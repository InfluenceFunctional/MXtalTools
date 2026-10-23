"""Helpers for point-cloud autoencoders: conversion of a decoder output into a weighted
point swarm, Gaussian-overlap reconstruction losses and evaluation overlaps, matched-point
distance statistics, and rotation-equivariance checks of an encoder and decoder."""
from typing import Union, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_scatter import scatter, scatter_softmax

from mxtaltools.models.functions.radial_graph import radius


def compute_gaussian_overlap(ref_types,
                             mol_batch,
                             decoded_data,
                             sigma,
                             nodewise_weights,
                             dist_to_self=False,
                             isolate_dimensions: list = None,
                             type_distance_scaling=0.1,
                             return_dists=False
                             ):
    """Sum weighted Gaussian overlaps from predicted points onto each reference node.

    Reference points are ``cat(mol_batch.pos, ref_types * type_distance_scaling)`` and
    predicted points are ``cat(decoded_data.pos, decoded_data.x * type_distance_scaling)``
    (the reference points themselves when ``dist_to_self`` is True), so positions and
    scaled type channels form one joint space. Pairs closer than ``4 * sigma`` within the
    same graph are found with ``torch_cluster.radius`` (at most 1000 reference points per
    predicted point). Each pair at joint-space distance ``d`` contributes
    ``exp(-(d / sigma) ** 2)`` times the weight of its predicted point, and the
    contributions are summed onto the reference node.

    Parameters
    ----------
    ref_types : torch.Tensor
        [num_nodes, num_types] type channels of the reference nodes (one-hot in
        :func:`ae_reconstruction_loss`).
    mol_batch : MolData
        Reference graphs; ``pos``, ``batch`` and ``num_nodes`` are read.
    decoded_data : MolData
        Predicted points; ``pos``, ``x`` and ``batch`` are read. With ``dist_to_self``
        only ``batch`` is read, as the graph index of the (reference) predicted points.
    sigma : float
        Gaussian width, in joint-space units.
    nodewise_weights : torch.Tensor
        [num_predicted_points] weight of each predicted point.
    dist_to_self : bool, optional
        Use the reference points as the predicted points. Default False.
    isolate_dimensions : list, optional
        ``[start, stop]``; if given, distances use only joint-space columns
        ``start:stop`` (the pair radius is still ``4 * sigma``). Default None, all columns.
    type_distance_scaling : float, optional
        Factor applied to the type channels. Default 0.1.
    return_dists : bool, optional
        Also return the pairs and their distances. Default False.

    Returns
    -------
    torch.Tensor or tuple
        ``nodewise_overlap`` [mol_batch.num_nodes]. With ``return_dists``,
        ``(nodewise_overlap, edges, dists)``: ``edges`` [2, num_pairs] holds predicted-point
        indices in row 0 and reference-node indices in row 1, and ``dists`` [num_pairs]
        the joint-space distances.
    """  # todo this could be simplified
    ref_points = torch.cat((mol_batch.pos, ref_types * type_distance_scaling), dim=1)

    if dist_to_self:
        pred_points = ref_points
    else:
        pred_types = decoded_data.x * type_distance_scaling  # nodes are already weighted at 1
        pred_points = torch.cat((decoded_data.pos, pred_types), dim=1)  # assume input x has already been normalized

    if isolate_dimensions is not None:  # only compute distances over certain dimensions
        ref_points = ref_points[:, isolate_dimensions[0]:isolate_dimensions[1]]
        pred_points = pred_points[:, isolate_dimensions[0]:isolate_dimensions[1]]

    edges = radius(ref_points, pred_points,
                   # r=2 * ref_points[:, :3].norm(dim=1).amax(),  # max range encompasses largest molecule in the batch
                   # alternatively any point which will have even a small overlap - should be faster by ignoring unimportant edges, where the gradient will anyway be vanishing
                   r=4 * sigma,
                   max_num_neighbors=1000,
                   batch_x=mol_batch.batch,
                   batch_y=decoded_data.batch)  # this step is slower than before
    dists = torch.linalg.norm(ref_points[edges[1]] - pred_points[edges[0]], dim=1)
    overlap = torch.exp(-torch.pow(dists / sigma, 2))
    scaled_overlap = overlap * nodewise_weights[edges[0]]  # reweight appropriately
    nodewise_overlap = scatter(scaled_overlap,
                               edges[1],
                               reduce='sum',
                               dim_size=mol_batch.num_nodes)

    if not return_dists:
        return nodewise_overlap
    else:
        return nodewise_overlap, edges, dists


def compute_type_evaluation_overlap(config,
                                    data,
                                    num_atom_types,
                                    decoded_data,
                                    true_nodes):
    """Compute type-channel-only overlaps at ``config.autoencoder.evaluation_sigma``.

    Both overlaps call :func:`compute_gaussian_overlap` with
    ``isolate_dimensions=[3, 3 + num_atom_types]`` (distances over the scaled type
    channels only) and ``type_distance_scaling=config.autoencoder.type_distance_scaling``.
    # todo could be more flexible

    Parameters
    ----------
    config : Namespace
        Reads ``config.autoencoder.evaluation_sigma`` and
        ``config.autoencoder.type_distance_scaling``.
    data : MolData
        Reference graphs; ``z`` is read for the unit self-overlap weights.
    num_atom_types : int
        Number of type channels.
    decoded_data : MolData
        Decoded swarm; ``aux_ind`` supplies the predicted-point weights.
    true_nodes : torch.Tensor
        [num_nodes, num_atom_types] reference type channels.

    Returns
    -------
    tuple of torch.Tensor
        ``(self_type_overlap, type_overlap)``, each [data.num_nodes]: the reference
        against itself with unit weights, and the decoded swarm against the reference.
        The order is the reverse of :func:`compute_coord_evaluation_overlap`.
    """
    type_overlap = compute_gaussian_overlap(
        true_nodes,
        data,
        decoded_data,
        config.autoencoder.evaluation_sigma,
        nodewise_weights=decoded_data.aux_ind,
        isolate_dimensions=[3, 3 + num_atom_types],
        type_distance_scaling=config.autoencoder.type_distance_scaling
    )
    self_type_overlap = compute_gaussian_overlap(
        true_nodes,
        data,
        data,
        config.autoencoder.evaluation_sigma,
        nodewise_weights=torch.ones(len(data.z), device=data.z.device, dtype=torch.float32),
        dist_to_self=True,
        isolate_dimensions=[3, 3 + num_atom_types],
        type_distance_scaling=config.autoencoder.type_distance_scaling
    )
    return self_type_overlap, type_overlap


def compute_coord_evaluation_overlap(
        config,
        data,
        decoded_data,
        true_nodes):
    """Compute position-only overlaps at ``config.autoencoder.evaluation_sigma``.

    Both overlaps call :func:`compute_gaussian_overlap` with ``isolate_dimensions=[0, 3]``
    (distances over positions only).
    # todo could be more flexible

    Parameters
    ----------
    config : Namespace
        Reads ``config.autoencoder.evaluation_sigma`` and
        ``config.autoencoder.type_distance_scaling``.
    data : MolData
        Reference graphs; ``z`` is read for the unit self-overlap weights.
    decoded_data : MolData
        Decoded swarm; ``aux_ind`` supplies the predicted-point weights.
    true_nodes : torch.Tensor
        [num_nodes, num_types] reference type channels.

    Returns
    -------
    tuple of torch.Tensor
        ``(coord_overlap, self_coord_overlap)``, each [data.num_nodes]: the decoded swarm
        against the reference, and the reference against itself with unit weights.
    """
    coord_overlap = compute_gaussian_overlap(
        true_nodes,
        data,
        decoded_data,
        config.autoencoder.evaluation_sigma,
        nodewise_weights=decoded_data.aux_ind,
        isolate_dimensions=[0, 3],
        type_distance_scaling=config.autoencoder.type_distance_scaling
    )
    self_coord_overlap = compute_gaussian_overlap(
        true_nodes,
        data,
        data,
        config.autoencoder.evaluation_sigma,
        nodewise_weights=torch.ones(len(data.z), device=data.z.device, dtype=torch.float32),
        dist_to_self=True,
        isolate_dimensions=[0, 3],
        type_distance_scaling=config.autoencoder.type_distance_scaling
    )
    return coord_overlap, self_coord_overlap


def compute_full_evaluation_overlap(mol_batch,
                                    decoded_mol_batch,
                                    true_nodes,
                                    sigma=None, distance_scaling=None):
    """Compute joint position-and-type overlaps of a decoded swarm and of the reference with itself.

    Parameters
    ----------
    mol_batch : MolData
        Reference graphs; ``z`` is read for the unit self-overlap weights.
    decoded_mol_batch : MolData
        Decoded swarm; ``aux_ind`` supplies the predicted-point weights.
    true_nodes : torch.Tensor
        [num_nodes, num_types] reference type channels.
    sigma : float
        Gaussian width passed to :func:`compute_gaussian_overlap`. The default None is not
        usable: ``4 * sigma`` raises TypeError.
    distance_scaling : float
        Passed as ``type_distance_scaling``. The default None is not usable: scaling the
        type channels by None raises TypeError.

    Returns
    -------
    tuple of torch.Tensor
        ``(full_overlap, self_overlap)``, each [mol_batch.num_nodes]: the decoded swarm
        against the reference, and the reference against itself with unit weights.
    """
    full_overlap = compute_gaussian_overlap(
        true_nodes, mol_batch, decoded_mol_batch, sigma,
        nodewise_weights=decoded_mol_batch.aux_ind,
        type_distance_scaling=distance_scaling,
    )
    self_overlap = compute_gaussian_overlap(
        true_nodes, mol_batch, mol_batch, sigma,
        nodewise_weights=torch.ones(len(mol_batch.z), device=mol_batch.z.device, dtype=torch.float32),
        dist_to_self=True,
        type_distance_scaling=distance_scaling)
    return full_overlap, self_overlap


def get_node_weights(mol_batch, decoded_mol_batch, decoding, num_decoder_nodes, node_weight_temperature):
    """Compute the weights of the decoded swarm points.

    Parameters
    ----------
    mol_batch : MolData
        Reference graphs; ``num_atoms`` [num_graphs] is read.
    decoded_mol_batch : MolData
        Decoded swarm; ``batch`` [num_graphs * num_decoder_nodes] is the softmax group
        index and ``num_nodes`` the ``dim_size`` passed to ``scatter_softmax``.
    decoding : torch.Tensor
        [num_graphs * num_decoder_nodes, output_depth] decoder output; the last column is
        the weight logit.
    num_decoder_nodes : int
        Swarm points per graph.
    node_weight_temperature : float
        Divisor applied to the weight logits before the softmax.

    Returns
    -------
    weight_per_swarm_point : torch.Tensor
        [num_graphs * num_decoder_nodes] uniform weight ``num_atoms / num_decoder_nodes``
        of each point's graph.
    nodewise_weights : torch.Tensor
        [num_graphs * num_decoder_nodes] softmax of the weight logits within each graph
        (sums to 1 per graph).
    nodewise_weights_tensor : torch.Tensor
        ``nodewise_weights`` multiplied by the graph's ``num_atoms`` (sums to
        ``num_atoms`` per graph).
    """
    # per-atom weights of each graph
    molwise_weight_per_swarm_point = mol_batch.num_atoms / num_decoder_nodes

    # cast to num_decoder_nodes
    weight_per_swarm_point = molwise_weight_per_swarm_point.repeat_interleave(num_decoder_nodes)

    # softmax over decoding weight dimension, adjusted by temperature
    nodewise_weights = scatter_softmax(decoding[:, -1] / node_weight_temperature,
                                       decoded_mol_batch.batch,
                                       dim=0,
                                       dim_size=decoded_mol_batch.num_nodes)

    # reweigh against the number of atoms
    nodewise_weights_tensor = nodewise_weights * mol_batch.num_atoms.repeat_interleave(
        num_decoder_nodes)

    return weight_per_swarm_point, nodewise_weights, nodewise_weights_tensor


def init_decoded_data(mol_batch, decoded_batch, device, num_nodes):
    """Return a clone of ``mol_batch`` carrying the decoded positions and swarm batch index.

    ``mol_batch.detach()`` reassigns every tensor attribute of the caller's ``mol_batch``
    to a detached tensor and returns ``mol_batch`` itself, which is then cloned. In the
    clone, ``pos`` is set to ``decoded_batch[:, :3]`` (a view that keeps its autograd
    history) and ``batch`` to ``arange(mol_batch.num_graphs)`` repeated ``num_nodes``
    times, on ``device``. All other attributes are the detached, cloned values from
    ``mol_batch``.
    """
    decoded_data = mol_batch.detach().clone()
    decoded_data.pos = decoded_batch[:, :3]
    decoded_data.batch = torch.arange(mol_batch.num_graphs).repeat_interleave(num_nodes).to(device)
    return decoded_data


def test_decoder_equivariance(data,
                              encoding: torch.Tensor,
                              rotated_encoding: torch.Tensor,
                              rotations: torch.Tensor,
                              autoencoder: nn.Module,
                              device: Union[torch.device, str]) -> torch.Tensor:
    """Measure the rotation equivariance of the decoded positions.

    Decodes ``encoding`` and ``rotated_encoding`` with ``autoencoder.decode``, rotates the
    first decoding's positions (columns 0:3) of graph ``i`` by ``rotations[i]``
    (``R @ p`` per point, graphs assumed to occupy consecutive blocks of
    ``autoencoder.num_decoder_nodes`` rows), and compares them with the positions of the
    second decoding. The remaining decoded columns are not compared.

    Parameters
    ----------
    data : MolData
        Only ``num_graphs`` is read.
    encoding : torch.Tensor
        [num_graphs, 3, k] encoding.
    rotated_encoding : torch.Tensor
        Rotated encoding with ``num_graphs * 3 * k`` elements, reshaped to
        [num_graphs, 3, k] (e.g. the [num_graphs, 3 * k] output of
        :func:`test_encoder_equivariance`).
    rotations : torch.Tensor
        [num_graphs, 3, 3] rotation matrices.
    autoencoder : nn.Module
        Model providing ``decode`` and ``num_decoder_nodes``.
    device : torch.device or str
        Device of the graph index built for the decoded rows.

    Returns
    -------
    torch.Tensor
        [num_graphs * num_decoder_nodes] relative error per decoded point, the mean over
        the three coordinates of ``abs(Rp1 - p2) / (1e-3 + abs(Rp1))``.
    """
    '''take a given embedding and decode it'''
    decoding = autoencoder.decode(encoding)

    '''rotate embedding and decode'''
    decoding2 = autoencoder.decode(
        rotated_encoding.reshape(data.num_graphs, 3, encoding.shape[-1]))

    '''rotate first decoding and compare'''
    decoded_batch = torch.arange(data.num_graphs).repeat_interleave(autoencoder.num_decoder_nodes).to(device)
    rotated_decoding_positions = torch.cat(
        [torch.einsum('ij, kj->ki', rotations[ind], decoding[:, :3][decoded_batch == ind])
         for ind in range(data.num_graphs)])
    rotated_decoding = decoding.clone()
    rotated_decoding[:, :3] = rotated_decoding_positions
    # first three dimensions should be equivariant and all trailing invariant
    decoder_equivariance_loss = (
            torch.abs(rotated_decoding[:, :3] - decoding2[:, :3]) / (1e-3 + torch.abs(rotated_decoding[:, :3])))
    return decoder_equivariance_loss.mean(-1)


def test_encoder_equivariance(data,
                              rotations: torch.Tensor,
                              autoencoder) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure the rotation equivariance of the encoder.

    Encodes a clone of ``data`` (``override_centering=True``) and rotates the encoding of
    graph ``n`` by ``rotations[n]``; then replaces ``data.pos`` with positions rotated
    per graph and encodes a clone of the rotated data. The per-graph position rotation
    concatenates ``data.pos[data.batch == ind]`` in graph order, so nodes must be sorted
    by graph.

    Parameters
    ----------
    data : MolData
        Input graphs. ``data.pos`` is replaced by the rotated positions (the caller's
        object is modified).
    rotations : torch.Tensor
        [num_graphs, 3, 3] rotation matrices.
    autoencoder : nn.Module
        Model providing ``encode(mol_batch, override_centering)`` that returns
        [num_graphs, 3, k].

    Returns
    -------
    encoder_equivariance_loss : torch.Tensor
        [num_graphs] mean over the ``3 * k`` components of
        ``abs(rotated_encoding - encoding2) / abs(rotated_encoding)``, where
        ``encoding2`` is the encoding of the rotated input. There is no epsilon in the
        denominator.
    encoding : torch.Tensor
        [num_graphs, 3, k] encoding of the unrotated input.
    rotated_encoding : torch.Tensor
        [num_graphs, 3 * k] rotated encoding, flattened.
    """
    '''embed the input data then rotate the embedding'''
    encoding = autoencoder.encode(data.clone(), override_centering=True)
    rotated_encoding = torch.einsum('nij, njk->nik',
                                    rotations,
                                    encoding
                                    )  # rotate in 3D

    rotated_encoding = rotated_encoding.reshape(data.num_graphs, rotated_encoding.shape[-1] * 3)
    '''rotate the input data and embed it'''
    data.pos = torch.cat([torch.einsum('ij, kj->ki', rotations[ind], data.pos[data.batch == ind])
                          for ind in range(data.num_graphs)])
    encoding2 = autoencoder.encode(data.clone(), override_centering=True)
    encoding2 = encoding2.reshape(data.num_graphs, encoding2.shape[-1] * 3)
    '''compare the embeddings - should be identical for an equivariant embedding'''
    encoder_equivariance_loss = (torch.abs(rotated_encoding - encoding2) / torch.abs(rotated_encoding)).mean(-1)
    return encoder_equivariance_loss, encoding, rotated_encoding


def decoding2mol_batch(mol_batch, decoding, num_decoder_nodes, node_weight_temperature, device):
    """Convert a decoder output into a weighted point swarm stored as a MolData batch.

    Parameters
    ----------
    mol_batch : MolData
        Reference graphs; ``num_graphs`` and ``num_atoms`` are read. Its tensor
        attributes are replaced by detached tensors (see :func:`init_decoded_data`), and
        ``mol_batch.aux_ind`` is then set in place to float32 ones [mol_batch.num_nodes]
        on ``device``.
    decoding : torch.Tensor
        [num_graphs * num_decoder_nodes, output_depth] decoder output, graph-major:
        columns 0:3 positions, then type logits, then one weight logit.
    num_decoder_nodes : int
        Swarm points per graph.
    node_weight_temperature : float
        Softmax temperature of the weight logits (see :func:`get_node_weights`).
    device : torch.device or str
        Device of the swarm ``batch`` index and of ``mol_batch.aux_ind``.

    Returns
    -------
    decoded_mol_batch : MolData
        Clone of ``mol_batch`` (see :func:`init_decoded_data`) with ``pos`` the decoded
        positions, ``batch`` the swarm graph index, ``x`` the softmax over the type logits
        ``decoding[:, 3:-1]``, ``num_nodes`` set to ``num_graphs * num_decoder_nodes``,
        and ``aux_ind`` the per-point weights summing to ``num_atoms`` per graph.
    nodewise_graph_weights : torch.Tensor
        [num_graphs * num_decoder_nodes] uniform weight ``num_atoms / num_decoder_nodes``.
    graph_weighted_node_weights : torch.Tensor
        [num_graphs * num_decoder_nodes] per-point weights summing to 1 per graph.
    node_weighted_node_weights : torch.Tensor
        [num_graphs * num_decoder_nodes] per-point weights summing to ``num_atoms`` per
        graph (the tensor stored as ``aux_ind``).
    """
    # generate input reconstructed as a data type
    decoded_mol_batch = init_decoded_data(mol_batch,
                                          decoding,
                                          device,
                                          num_decoder_nodes
                                          )
    # compute the distributional weight of each node
    nodewise_graph_weights, graph_weighted_node_weights, node_weighted_node_weights = \
        get_node_weights(mol_batch, decoded_mol_batch, decoding,
                         num_decoder_nodes,
                         node_weight_temperature)
    decoded_mol_batch.aux_ind = node_weighted_node_weights
    # input node weights are always 1 - corresponding each to an atom
    mol_batch.aux_ind = torch.ones(mol_batch.num_nodes, dtype=torch.float32, device=device)
    # get probability distribution over type dimensions
    decoded_mol_batch.x = F.softmax(decoding[:, 3:-1], dim=1)
    decoded_mol_batch.num_nodes = len(decoded_mol_batch.x)
    return decoded_mol_batch, nodewise_graph_weights, graph_weighted_node_weights, node_weighted_node_weights


def ae_reconstruction_loss(mol_batch,
                           decoding_batch,
                           graph_weighted_node_weights,
                           node_weighted_node_weights,
                           num_atom_types,
                           type_distance_scaling,
                           autoencoder_sigma,
                           ):
    """Compute the Gaussian-overlap reconstruction losses of a decoded swarm.

    Reference types are ``one_hot(mol_batch.x.flatten().long(), num_atom_types)``. The
    decoded overlap onto each atom (weights ``decoding_batch.aux_ind``) and the reference
    self-overlap (weights ``mol_batch.aux_ind``) come from
    :func:`compute_gaussian_overlap` with ``autoencoder_sigma`` and
    ``type_distance_scaling``; the pairs and joint-space distances of the decoded
    overlap feed the distance and clumping terms. Reductions use torch_scatter; under
    ``reduce='min'`` a point or atom with no pair receives 0.

    Parameters
    ----------
    mol_batch : MolData
        Reference graphs; ``x`` (type indices in ``[0, num_atom_types)``), ``pos``,
        ``batch``, ``num_nodes``, ``num_graphs`` and ``aux_ind`` are read (``aux_ind`` is
        set by :func:`decoding2mol_batch`).
    decoding_batch : MolData
        Decoded swarm from :func:`decoding2mol_batch`; ``pos``, ``x`` (type
        probabilities), ``batch``, ``num_nodes`` and ``aux_ind`` are read.
    graph_weighted_node_weights : torch.Tensor
        [num_decoded_points] point weights summing to 1 per graph; weight the predicted
        type composition.
    node_weighted_node_weights : torch.Tensor
        [num_decoded_points] point weights summing to ``num_atoms`` per graph; summed in
        the clumping term.
    num_atom_types : int
        Number of type classes.
    type_distance_scaling : float
        Factor applied to the type channels in the joint space.
    autoencoder_sigma : float
        Gaussian width of the overlaps; the pair radius is ``4 * autoencoder_sigma``.

    Returns
    -------
    nodewise_reconstruction_loss : torch.Tensor
        [num_atoms] ``smooth_l1_loss`` between the decoded overlap and the self-overlap.
    nodewise_type_loss : torch.Tensor
        Scalar: binary cross-entropy of the weighted predicted type composition per graph
        (clipped to ``[1e-6, 1 - 1e-6]``) against the true composition (mean one-hot per
        graph), minus the binary cross-entropy of the true composition with itself; both
        averaged over graphs and types.
    graph_reconstruction_loss : torch.Tensor
        [num_graphs] per-graph mean of ``nodewise_reconstruction_loss``.
    self_likelihoods : torch.Tensor
        [num_atoms] reference self-overlap.
    nearest_node_loss : torch.Tensor
        [num_graphs] per-graph mean, over decoded points, of the distance to the nearest
        paired atom (0 for points with no atom within the pair radius).
    graph_clumping_loss : torch.Tensor
        [num_graphs] per-graph mean of ``smooth_l1_loss`` between 1 and the summed
        ``node_weighted_node_weights`` of the decoded points paired with each atom at
        joint-space distance below 0.5 (pairs exist only within the pair radius).
    nearest_component_dist : torch.Tensor
        [num_atoms] distance from each atom to its nearest paired decoded point (0 when
        none is within the pair radius).
    nearest_component_loss : torch.Tensor
        [num_graphs] per-graph mean of ``nearest_component_dist``.
    """
    true_node_one_hot = F.one_hot(mol_batch.x.flatten().long(), num_classes=num_atom_types).float()

    decoder_likelihoods, input2output_edges, input2output_dists = (
        compute_gaussian_overlap(true_node_one_hot,
                                 mol_batch,
                                 decoding_batch,
                                 autoencoder_sigma,
                                 nodewise_weights=decoding_batch.aux_ind,
                                 type_distance_scaling=type_distance_scaling,
                                 return_dists=True
                                 ))

    # if sigma is too large, these can be > 1, so we map to the overlap of the true density with itself
    self_likelihoods = compute_gaussian_overlap(
        true_node_one_hot,
        mol_batch, mol_batch,
        autoencoder_sigma,
        nodewise_weights=mol_batch.aux_ind, dist_to_self=True,
        type_distance_scaling=type_distance_scaling)

    # typewise agreement for whole graph
    per_graph_true_types = scatter(
        true_node_one_hot, mol_batch.batch[:, None], dim=0, reduce='mean')
    per_graph_pred_types = scatter(
        decoding_batch.x * graph_weighted_node_weights[:, None], decoding_batch.batch[:, None], dim=0,
        reduce='sum')

    nodewise_type_loss = (
            F.binary_cross_entropy(per_graph_pred_types.clip(min=1e-6, max=1 - 1e-6), per_graph_true_types) -
            F.binary_cross_entropy(per_graph_true_types, per_graph_true_types))

    nodewise_reconstruction_loss = F.smooth_l1_loss(decoder_likelihoods, self_likelihoods, reduction='none')
    graph_reconstruction_loss = scatter(nodewise_reconstruction_loss, mol_batch.batch, reduce='mean')

    # new losses -
    # 1 penalize output components for distance to nearest atom
    nearest_node_dist = scatter(input2output_dists,
                                input2output_edges[0],
                                reduce='min',
                                dim_size=decoding_batch.num_nodes
                                )
    nearest_node_loss = scatter(nearest_node_dist,
                                decoding_batch.batch,
                                reduce='mean',
                                dim_size=mol_batch.num_graphs)
    # 1a also identify reciprocal distance from each atom to nearest component
    nearest_component_dist = scatter(input2output_dists,
                                input2output_edges[1],
                                reduce='min',
                                dim_size=mol_batch.num_nodes
                                )
    nearest_component_loss = scatter(nearest_component_dist,
                                     mol_batch.batch,
                                     reduce='mean',
                                     dim_size=mol_batch.num_graphs)
    # 2 penalize area near an atom for not being a part of an exactly atom-size clump
    collect_bools = input2output_dists < 0.5
    inds_within_cutoff = input2output_edges[0][collect_bools]
    inside_edge_nodes = input2output_edges[1][collect_bools]
    collected_particle_weights = node_weighted_node_weights[inds_within_cutoff]
    pred_particle_weights = scatter(collected_particle_weights,
                                    inside_edge_nodes,
                                    reduce='sum',
                                    dim_size=mol_batch.num_nodes,
                                    )

    nodewise_clumping_loss = F.smooth_l1_loss(pred_particle_weights, torch.ones_like(pred_particle_weights),
                                              reduction='none')
    graph_clumping_loss = scatter(nodewise_clumping_loss, mol_batch.batch, reduce='mean')

    return (nodewise_reconstruction_loss, nodewise_type_loss,
            graph_reconstruction_loss, self_likelihoods,
            nearest_node_loss, graph_clumping_loss,
            nearest_component_dist, nearest_component_loss)


def batch_rmsd(mol_batch,
               decoded_mol_batch,
               true_node_one_hot,
               intrapoint_cutoff: float = 0.5,
               probability_threshold: float = 0.25,
               type_distance_scaling: float = 2):
    """Match decoded swarm points to atoms and compute per-graph mean matched distances.

    Atoms and decoded points are placed in the joint space
    ``cat(pos, types * type_distance_scaling)`` (one-hot for atoms,
    ``decoded_mol_batch.x`` for decoded points). Each decoded point within
    ``intrapoint_cutoff`` of an atom in the same graph (found with
    ``torch_cluster.radius``, at most 1000 atoms per point) is collected onto that atom
    with its weight ``decoded_mol_batch.aux_ind``. An atom is matched when
    ``abs(1 - summed weight) < probability_threshold``; a graph is complete when all its
    atoms are matched. The predicted point of an atom is the weight-weighted sum (not
    renormalized by the summed weight) of its collected joint-space points.

    Parameters
    ----------
    mol_batch : MolData
        Reference graphs; ``pos``, ``batch``, ``num_nodes`` and ``num_graphs`` are read.
    decoded_mol_batch : MolData
        Decoded swarm from :func:`decoding2mol_batch`; ``pos``, ``x``, ``batch`` and
        ``aux_ind`` are read.
    true_node_one_hot : torch.Tensor
        [num_atoms, num_types] one-hot atom types.
    intrapoint_cutoff : float, optional
        Joint-space collection radius. Default 0.5.
    probability_threshold : float, optional
        Tolerance on the summed weight per atom. Default 0.25.
    type_distance_scaling : float, optional
        Factor applied to the type channels. Default 2.

    Returns
    -------
    rmsd : torch.Tensor
        [num_graphs] per-graph arithmetic mean of ``pred_dists`` (not a root mean square);
        NaN for incomplete graphs.
    pred_dists : torch.Tensor
        [num_atoms] joint-space distance (positions and scaled types) from each atom to
        its predicted point, computed before NaN masking.
    complete_graph_bools : torch.Tensor
        [num_graphs] bool, True when every atom of the graph is matched.
    matched_node_bools : torch.Tensor
        [num_atoms] bool, True for matched atoms.
    pred_particle_points : torch.Tensor
        [num_atoms, 3 + num_types] predicted joint-space points; rows of unmatched atoms
        are NaN.
    pred_particle_weights : torch.Tensor
        [num_atoms] summed weight collected onto each atom.
    """
    ref_types = true_node_one_hot.float()
    ref_points = torch.cat((mol_batch.pos, ref_types * type_distance_scaling), dim=1)
    pred_types = decoded_mol_batch.x * type_distance_scaling  # nodes are already weighted at 1
    pred_points = torch.cat((decoded_mol_batch.pos, pred_types), dim=1)  # assume input x has already been normalized
    nodewise_weights = decoded_mol_batch.aux_ind

    edges = radius(ref_points,
                   pred_points,
                   r=intrapoint_cutoff,
                   max_num_neighbors=1000,
                   batch_x=mol_batch.batch,
                   batch_y=decoded_mol_batch.batch)  # this step is slower than before
    dists = torch.linalg.norm(ref_points[edges[1]] - pred_points[edges[0]], dim=1)

    collect_bools = dists < intrapoint_cutoff
    inds_within_cutoff = edges[0][collect_bools]
    inside_edge_nodes = edges[1][collect_bools]
    collected_particles = pred_points[inds_within_cutoff]
    collected_particle_weights = nodewise_weights[inds_within_cutoff]
    # # confirm each output is mapped to a single input
    # a, b = torch.unique(edges[0][collect_bools], return_counts=True)
    # assert b.max() == 1
    pred_particle_weights = scatter(collected_particle_weights,
                                    inside_edge_nodes,
                                    reduce='sum',
                                    dim_size=mol_batch.num_nodes,
                                    )
    # filter here for where we do not match the scaffold (no nearby nodes, or insufficient probability mass)
    missing_particle_bools = (1 - pred_particle_weights).abs() >= probability_threshold
    complete_graph_bools = scatter((~missing_particle_bools).long(),
                                   mol_batch.batch,
                                   reduce='mul',
                                   dim_size=mol_batch.num_graphs,
                                   dim=0
                                   ).bool()
    pred_particle_points = scatter(collected_particles * collected_particle_weights[:, None],
                                   inside_edge_nodes,
                                   reduce='sum',
                                   dim=0,
                                   dim_size=mol_batch.num_nodes,
                                   )
    pred_dists = torch.linalg.norm(ref_points - pred_particle_points, dim=1)
    rmsd = scatter(pred_dists, mol_batch.batch, reduce='mean', dim_size=mol_batch.num_graphs)
    rmsd[~complete_graph_bools] = torch.nan
    pred_particle_points[missing_particle_bools] *= torch.nan

    return rmsd, pred_dists, complete_graph_bools, ~missing_particle_bools, pred_particle_points, pred_particle_weights

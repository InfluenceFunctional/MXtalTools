from typing import Union, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_scatter import scatter, scatter_softmax

from mxtaltools.models.functions.asymmetric_radius_graph import radius


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
    """
    compute distance between gaussian mixtures in high dimension, taking atom types as one-hot dimensions
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
                   max_num_neighbors=10000,
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
    """
    compute typewise overlaps at evaluation sigma
    # todo could be more flexible
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
    """
    compute positional overlaps at evaluation sigma
    # todo could be more flexible
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
    """
    compute overall overlaps at evaluation sigma
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
    """
    extract nodewise normed weights from decoder swarm
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
    """
    check decoder end-to-end equivariance
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
    """
    check encoder end-to-end equivariance
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
    return decoded_mol_batch, nodewise_graph_weights, graph_weighted_node_weights, node_weighted_node_weights


def ae_reconstruction_loss(mol_batch,
                           decoding_batch,
                           graph_weighted_node_weights,
                           node_weighted_node_weights,
                           num_atom_types,
                           type_distance_scaling,
                           autoencoder_sigma,
                           ):
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
    nearest_node_loss = scatter(nearest_node_dist, decoding_batch.batch, reduce='mean',
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
    ref_types = true_node_one_hot.float()
    ref_points = torch.cat((mol_batch.pos, ref_types * type_distance_scaling), dim=1)
    pred_types = decoded_mol_batch.x * type_distance_scaling  # nodes are already weighted at 1
    pred_points = torch.cat((decoded_mol_batch.pos, pred_types), dim=1)  # assume input x has already been normalized
    nodewise_weights = decoded_mol_batch.aux_ind

    edges = radius(ref_points,
                   pred_points,
                   r=intrapoint_cutoff,
                   max_num_neighbors=10000,
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

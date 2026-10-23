"""Geometry helpers for molecules and crystal cells.

Principal inertial axes of point clouds (unit masses), cell volumes and
fractional/Cartesian transform matrices, crystal-system constraints on cell
parameters, rotation-vector and rotation-matrix conversions, molecular volume
estimates, and distances between crystals in the latent cell parameterization.
Angles are in radians throughout.
"""
import sys
from math import sqrt
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch_scatter import scatter, scatter_max

from mxtaltools.common.sym_utils import init_sym_info
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.constants.space_group_info import LATTICE_TO_CODE
from mxtaltools.models.functions.radial_graph import radius


def compute_principal_axes_np(coords):
    """
    Compute the principal inertial axes of one point cloud, with unit masses.

    The coordinates are centred on their mean. Eigenvectors of the inertial
    tensor are sorted by ascending moment and stored as rows. Each axis is then
    flipped so that its overlap with the vector from the centroid to the
    farthest point is non-negative (an exactly zero overlap counts as positive;
    among points equidistant to 8 decimals the lowest index is used). If any
    overlap magnitude is below 1e-3 and the flipped basis is left-handed, the
    axis with the smallest overlap magnitude is negated. The signs are not
    unique for some symmetric point clouds.

    Parameters
    ----------
    coords : np.ndarray
        Point coordinates, [num_points, 3].

    Returns
    -------
    Ip : np.ndarray
        Principal axes as rows, sorted by ascending moment, [3, 3].
    Ipm : np.ndarray
        Principal moments in ascending order, in squared coordinate units, [3].
    I : np.ndarray
        Inertial tensor of the centred coordinates in the input frame, [3, 3].
    """  # todo harmonize with torch version - currently disagrees ~0.5% of the time
    points = coords - coords.mean(0)

    x, y, z = points.T
    Ixx = np.sum((y ** 2 + z ** 2))
    Iyy = np.sum((x ** 2 + z ** 2))
    Izz = np.sum((x ** 2 + y ** 2))
    Ixy = -np.sum(x * y)
    Iyz = -np.sum(y * z)
    Ixz = -np.sum(x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])  # inertial tensor
    Ipm, Ip = np.linalg.eig(I)  # principal inertial tensor
    Ipm, Ip = np.real(Ipm), np.real(Ip)
    sort_inds = np.argsort(Ipm)
    Ipm = Ipm[sort_inds]
    Ip = Ip.T[sort_inds]  # want eigenvectors to be sorted row-wise (rather than column-wise)

    # cardinal direction is vector from CoM to the farthest atom
    dists = np.linalg.norm(points, axis=1)
    max_ind = np.argmax(dists)
    max_equivs = np.argwhere(np.round(dists, 8) == np.round(dists[max_ind], 8))[:,
    0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(np.amin(max_equivs))
    direction = points[max_ind]
    direction = np.divide(direction, np.linalg.norm(direction))
    overlaps = Ip.dot(direction)  # check if the principal components point towards or away from the CoG
    signs = np.sign(overlaps)  # returns zero for zero overlap, but we want it to default to +1 in this case
    signs[signs == 0] = 1

    Ip = (Ip.T * signs).T  # if the vectors have negative overlap, flip the direction
    if np.any(
            np.abs(
                overlaps) < 1e-3):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
        # align the 'good' vectors
        fix_ind = np.argmin(np.abs(overlaps))  # vector with vanishing overlap
        if compute_Ip_handedness(Ip) < 0:  # make sure result is right handed
            Ip[fix_ind] = -Ip[fix_ind]

    return Ip, Ipm, I


def compute_inertial_tensor_torch(x: torch.tensor, y: torch.tensor, z: torch.tensor):
    """
    Build the inertial tensor of one point cloud (unit masses) and its eigendecomposition.

    The coordinates are used as given, without centring. The tensor is
    assembled with ``torch.tensor`` from the summed components, so it is a new
    tensor on ``x.device`` that is not connected to the autograd graph of the
    inputs.

    Parameters
    ----------
    x : torch.Tensor
        x coordinates, [num_points].
    y : torch.Tensor
        y coordinates, [num_points].
    z : torch.Tensor
        z coordinates, [num_points].

    Returns
    -------
    I : torch.Tensor
        Inertial tensor, [3, 3].
    Ip : torch.Tensor
        Complex eigenvectors from ``torch.linalg.eig``, one per column,
        unsorted, [3, 3].
    Ipm : torch.Tensor
        Complex eigenvalues from ``torch.linalg.eig``, unsorted, [3].
    """  # todo harmonize with numpy version - currently disagrees ~0.5% of the time
    Ixy = -torch.sum(x * y)
    Iyz = -torch.sum(y * z)
    Ixz = -torch.sum(x * z)
    # I = torch.tensor([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]],device=points.device)  # inertial tensor
    I = torch.tensor(
        [[torch.sum((y ** 2 + z ** 2)), Ixy, Ixz],
         [Ixy, torch.sum((x ** 2 + z ** 2)), Iyz],
         [Ixz, Iyz, torch.sum((x ** 2 + y ** 2))]], device=x.device)  # inertial tensor

    Ipm, Ip = torch.linalg.eig(I)  # principal inertial tensor

    return I, Ip, Ipm


def single_molecule_principal_axes_torch(coords: torch.tensor, masses=None, return_direction=False):
    """
    Compute the principal inertial axes of one point cloud (unit masses), with sign rules.

    The coordinates are not centred here: the inertial tensor and the reference
    direction are both taken about the origin, so ``coords`` should already be
    centred. Eigenvectors (real parts from ``torch.linalg.eig``) are sorted by
    ascending moment and stored as rows. The reference direction is the
    position of the point farthest from the origin (lowest index among exact
    ties), unnormalised. Exactly zero overlaps with it are set to 1e-9. Each
    axis is flipped to a non-negative overlap with the reference direction; if
    any overlap magnitude is below 1e-8, the axis with the smallest overlap
    magnitude is then replaced by the cross product of the other two axes,
    taken in increasing index order.

    Parameters
    ----------
    coords : torch.Tensor
        Point coordinates, [num_points, 3].
    masses : optional
        Not used. If not None, a message is printed and the calculation
        proceeds without masses.
    return_direction : bool
        If True, also return the reference direction.

    Returns
    -------
    Ip : torch.Tensor
        Principal axes as rows, sorted by ascending moment, [3, 3].
    Ipm : torch.Tensor
        Principal moments in ascending order, [3].
    I : torch.Tensor
        Inertial tensor about the origin, [3, 3].
    direction : torch.Tensor
        Coordinates of the farthest point, [3]; returned only when
        ``return_direction`` is True.
    """
    if masses is not None:
        print('Inertial tensor is purely geometric! Calculation will not account for varying masses')
    x, y, z = coords.T
    I, Ip, Ipm = compute_inertial_tensor_torch(x, y, z)

    Ipm, Ip = torch.real(Ipm), torch.real(Ip)
    sort_inds = torch.argsort(Ipm)
    Ipm = Ipm[sort_inds]
    Ip = Ip.T[sort_inds]  # want eigenvectors to be sorted row-wise (rather than column-wise)

    # cardinal direction is vector from CoM to farthest atom
    dists = torch.linalg.norm(coords, axis=1)  # CoM is at 0,0,0
    max_ind = torch.argmax(dists)
    max_equivs = torch.where(dists == dists[max_ind])[
        0]  # torch.where(torch.round(dists, decimals=8) == torch.round(dists[max_ind], decimals=8))[0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(torch.amin(max_equivs))
    direction = coords[max_ind]
    # direction = direction / torch.linalg.norm(direction) # magnitude doesn't matter, only the sign
    overlaps = torch.inner(Ip,
                           direction)  # Ip.dot(direction) # check if the principal components point towards or away from the CoG
    if any(overlaps == 0):  # exactly zero is invalid #
        overlaps[overlaps == 0] = 1e-9
    if any(torch.abs(
            overlaps) < 1e-8):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
        # align the 'good' vectors
        Ip = (Ip.T * torch.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction
        fix_ind = torch.argmin(torch.abs(overlaps))
        other_vectors = np.delete(np.arange(3), fix_ind)
        check_direction = torch.cross(Ip[other_vectors[0]], Ip[other_vectors[1]])
        # align the 'bad' vector
        Ip[fix_ind] = check_direction  # Ip[fix_ind] * torch.sign(torch.dot(check_direction, Ip[fix_ind]))
    else:
        Ip = (Ip.T * torch.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction

    if return_direction:
        return Ip, Ipm, I, direction
    else:
        return Ip, Ipm, I


def list_molecule_principal_axes_torch(coords_list: list = None,
                                       skip_centring=False):
    """
    Compute principal inertial axes (unit masses) for a list of point clouds of different sizes.

    Each cloud is centred on its mean unless ``skip_centring`` is True, the
    clouds are concatenated, and the axes are computed with
    ``scatter_compute_Ip`` (which adds Gaussian noise to the coordinates when
    they require grad). Each axis is then oriented against the vector from the
    origin to the cloud's farthest point by ``get_overlaps`` and
    ``correct_Ip_directions``.

    Parameters
    ----------
    coords_list : list of torch.Tensor
        Point clouds, each [num_points_i, 3].
    skip_centring : bool
        If True, use the coordinates as given (already centred).

    Returns
    -------
    Ip_fin : torch.Tensor
        Principal axes as rows, sorted by ascending moment, [num_graphs, 3, 3].
    Ipm_fin : torch.Tensor
        Principal moments in ascending order, [num_graphs, 3].
    I : torch.Tensor
        Inertial tensors, [num_graphs, 3, 3].
    """
    if not skip_centring:  # todo accelerate with scatter
        coords_list_centred = [coord - coord.mean(0) for coord in coords_list]
        all_coords = torch.cat(coords_list_centred)
    else:
        all_coords = torch.cat(coords_list)

    batch, ptrs = extract_batching_info(coords_list,
                                        all_coords.device)  # todo pass batch info as an argument instead calculating here

    Ip, Ipm_fin, I = scatter_compute_Ip(all_coords, batch)

    # cardinal direction is vector from CoM to the farthest atom
    direction = batch_get_furthest_node_vector(all_coords, batch, num_graphs=len(coords_list))
    normed_direction = direction / torch.linalg.norm(direction, dim=1)[:, None]
    overlaps, signs = get_overlaps(Ip, normed_direction)

    Ip_fin = correct_Ip_directions(Ip, overlaps,
                                   signs)  # somehow, fails for mirror planes, on top of symmetric and spherical tops

    return Ip_fin, Ipm_fin, I


def batch_molecule_principal_axes_torch(coords_i: torch.FloatTensor,
                                        batch: torch.LongTensor,
                                        num_graphs: int,
                                        nodes_per_graph: torch.LongTensor,
                                        heavy_atoms_only: bool = True,
                                        atom_types: Optional[torch.LongTensor] = None, ):
    """
    Compute principal inertial axes (unit masses) for a batch of molecules.

    Coordinates are centred per graph with ``center_batch``: on the centroid of
    the heavy atoms (``atom_types > 1``) when ``heavy_atoms_only`` is True,
    otherwise on the centroid of all atoms; every atom is translated. The
    inertial tensor and the reference direction then use the heavy atoms only,
    or all atoms when ``heavy_atoms_only`` is False. The reference direction is
    the unit vector from the centroid to the farthest of those atoms. Axes come
    from ``scatter_compute_Ip``, which adds Gaussian noise of standard
    deviation 0.05 to the coordinates when they require grad, and are oriented
    by ``get_overlaps`` and ``correct_Ip_directions`` so that each axis has a
    non-negative overlap with the reference direction. The handedness of the
    returned basis is not fixed; it can be left- or right-handed. The inputs
    are not modified.

    Parameters
    ----------
    coords_i : torch.FloatTensor
        Atom coordinates, [num_nodes, 3], grouped contiguously by graph in
        batch order.
    batch : torch.LongTensor
        Graph index of each atom, [num_nodes].
    num_graphs : int
        Number of graphs in the batch.
    nodes_per_graph : torch.LongTensor
        Atom count per graph, [num_graphs]; must equal ``bincount(batch)``.
    heavy_atoms_only : bool
        If True (default), centre on and compute the axes from atoms with
        ``atom_types > 1`` only. Every graph must then contain at least one
        such atom: ``scatter_compute_Ip`` sizes its output by the largest
        graph index among the selected atoms, and
        ``batch_get_furthest_node_vector`` has no node to select for an empty
        graph.
    atom_types : torch.LongTensor, optional
        Atomic numbers, [num_nodes]; required when ``heavy_atoms_only`` is True.

    Returns
    -------
    Ip_fin : torch.Tensor
        Principal axes as rows, sorted by ascending moment, [num_graphs, 3, 3].
    Ipm_fin : torch.Tensor
        Principal moments in ascending order, in squared coordinate units,
        [num_graphs, 3].
    I : torch.Tensor
        Inertial tensors of the centred (and, under grad, noised)
        coordinates, [num_graphs, 3, 3].

    Raises
    ------
    ValueError
        If ``bincount(batch, minlength=num_graphs)`` differs from
        ``nodes_per_graph`` (an explicit raise, so it holds under ``python -O``;
        before 2026-10-22 a bare assert).
    """
    bc = torch.bincount(batch, minlength=num_graphs)
    if not torch.equal(bc, nodes_per_graph):  # an explicit raise, so it holds under python -O (was a bare assert)
        raise ValueError(f"bincount(batch) vs num_atoms mismatch:\n  bincount={bc.tolist()}\n"
                         f"  num_atoms={nodes_per_graph.tolist()}")
    coords = center_batch(coords_i, batch, num_graphs, nodes_per_graph,
                          center_on_heavy_atoms=heavy_atoms_only, atom_types=atom_types)

    if heavy_atoms_only:
        mask = atom_types > 1
        Ip, Ipm_fin, I = scatter_compute_Ip(coords[mask], batch[mask])
        direction = batch_get_furthest_node_vector(coords[mask], batch[mask],
                                                   num_graphs)  # todo ON NEXT REPARAMETERIZATION SET THIS TO A MORE STABLE ANCHOR like median atom or some inner product

    else:
        Ip, Ipm_fin, I = scatter_compute_Ip(coords, batch)
        direction = batch_get_furthest_node_vector(coords, batch, num_graphs)

    # cardinal direction is vector from CoM to the farthest atom
    normed_direction = direction / torch.linalg.norm(direction, dim=1)[:, None]
    overlaps, signs = get_overlaps(Ip, normed_direction)
    Ip_fin = correct_Ip_directions(Ip, overlaps,
                                   signs)  # fails for mirror planes, on top of symmetric and spherical tops

    return Ip_fin, Ipm_fin, I

    '''  visualize clouds and axes for testing
    ind = 16
    x, y, z = coords[batch == ind].T.cpu().detach().numpy()
    types = atom_types[batch==ind].cpu().detach().numpy()
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode='markers'))
    a, b, c = torch.stack([torch.zeros_like(direction[ind]), direction[ind]]).T.cpu().detach().numpy()
    fig.add_trace(go.Scatter3d(x=a, y=b, z=c, marker_color=types))
    colors = ['red', 'green', 'blue']
    for i in range(3):
        a, b, c = torch.stack([torch.zeros_like(direction[ind]), Ip[ind, i]]).T.cpu().detach().numpy()
        fig.add_trace(go.Scatter3d(x=a, y=b, z=c, marker_color=colors[i], name='Initial principal axes',
                                   legendgroup='Initial principal axes', showlegend=i == 0))
    for i in range(3):
        a, b, c = torch.stack([torch.zeros_like(direction[ind]), Ip_fin[ind, i]]).T.cpu().detach().numpy()
        fig.add_trace(go.Scatter3d(x=a, y=b, z=c, marker_color=colors[i], name='Fixed principal axes',
                                   legendgroup='Fixed principal axes', showlegend=i == 0))
    fig.show(renderer='browser')
    '''


def correct_Ip_directions(Ip, overlaps, signs, overlap_threshold: float = 1e-5):
    """
    Flip principal axes to non-negative overlap with a reference direction (unreachable definition).

    A second function named ``correct_Ip_directions`` is defined later in this
    module and replaces this one at import, so this body is never called
    through the module attribute. See that definition for the behavior.
    """
    handedness = compute_Ip_handedness(Ip)
    # if the vectors have negative overlap, flip the direction,
    Ip_fixed = (Ip.permute(0, 2, 1) * signs[..., None]).permute(0, 2, 1)

    # if any overlaps are vanishing (up to 32 bit precision),
    # happens if the cardinal direction is too close to an existing principal axis
    # determine the direction via the RHR (if two overlaps are vanishing, this will not work)
    small_overlaps_bool = (overlaps.abs() < overlap_threshold).any(dim=1)
    # identify the smallest overlap if we need to flip a direction
    fix_ind = torch.argmin(overlaps.abs(), dim=-1)  # [B]

    left_handed = handedness < 0
    to_flip_bools = left_handed * small_overlaps_bool
    Ip_fixed[to_flip_bools, fix_ind[to_flip_bools]] *= -1

    Ip_fixed_list = []
    for ii, Ip_i in enumerate(Ip):
        # if the vectors have negative overlap, flip the direction,
        Ip_fixed = (Ip_i.T * signs[ii]).T
        # if any overlaps are vanishing (up to 32 bit precision),
        # happens if the cardinal direction is too close to an existing principal axis
        # determine the direction via the RHR (if two overlaps are vanishing, this will not work)
        if any(torch.abs(overlaps[ii]) < overlap_threshold):
            # enforce right-handedness in the free vector (vanishing overlap)
            fix_ind = torch.argmin(torch.abs(overlaps[ii]))  # vector with vanishing overlap
            if compute_Ip_handedness(Ip_i) < 0:  # if result is not right-handed, swap it
                Ip_fixed[fix_ind] = -Ip_fixed[fix_ind]

        Ip_fixed_list.append(Ip_fixed)
    Ip_fin = torch.stack(Ip_fixed_list)
    return Ip_fin


def correct_Ip_directions(Ip, overlaps, signs, overlap_threshold: float = 1e-5):
    """
    Flip principal axes to non-negative overlap with a reference direction.

    Row ``i`` of each ``Ip[n]`` is multiplied by ``signs[n, i]``. For graphs
    where any ``|overlaps[n]|`` is below ``overlap_threshold`` and the input
    basis ``Ip[n]`` (before the sign flips) is left-handed, the axis with the
    smallest ``|overlaps[n]|`` is additionally negated. The inputs are not
    modified.

    Parameters
    ----------
    Ip : torch.Tensor
        Principal axes as rows, [num_graphs, 3, 3].
    overlaps : torch.Tensor
        Overlap of each axis with the reference direction, [num_graphs, 3].
    signs : torch.Tensor
        Sign applied to each axis, [num_graphs, 3]; ``get_overlaps`` supplies
        the overlap signs with zeros mapped to +1.
    overlap_threshold : float
        Magnitude below which an overlap counts as vanishing.

    Returns
    -------
    Ip_fin : torch.Tensor
        Oriented principal axes as rows, [num_graphs, 3, 3].
    """
    handedness = compute_Ip_handedness(Ip)
    # if the vectors have negative overlap, flip the direction,
    Ip_fin = (Ip.permute(0, 2, 1) * signs[:, None, :]).permute(0, 2, 1)

    # if any overlaps are vanishing (up to 32 bit precision),
    # happens if the cardinal direction is too close to an existing principal axis
    # determine the direction via the RHR (if two overlaps are vanishing, this will not work)
    small_overlaps_bool = (overlaps.abs() < overlap_threshold).any(dim=1)
    # identify the smallest overlap if we need to flip a direction
    fix_ind = torch.argmin(overlaps.abs(), dim=-1)  # [B]

    left_handed = handedness < 0
    to_flip_bools = left_handed * small_overlaps_bool
    Ip_fin[to_flip_bools, fix_ind[to_flip_bools]] *= -1
    #
    # Ip_fixed_list = []
    # for ii, Ip_i in enumerate(Ip):
    #     # if the vectors have negative overlap, flip the direction,
    #     Ip_fixed = (Ip_i.T * signs[ii]).T
    #     # if any overlaps are vanishing (up to 32 bit precision),
    #     # happens if the cardinal direction is too close to an existing principal axis
    #     # determine the direction via the RHR (if two overlaps are vanishing, this will not work)
    #     if any(torch.abs(overlaps[ii]) < overlap_threshold):
    #         # enforce right-handedness in the free vector (vanishing overlap)
    #         fix_ind = torch.argmin(torch.abs(overlaps[ii]))  # vector with vanishing overlap
    #         if compute_Ip_handedness(Ip_i) < 0:  # if result is not right-handed, swap it
    #             Ip_fixed[fix_ind] = -Ip_fixed[fix_ind]
    #
    #     Ip_fixed_list.append(Ip_fixed)
    # Ip_fin = torch.stack(Ip_fixed_list)
    return Ip_fin


def get_overlaps(Ip, direction):
    """
    Compute the overlap of each principal axis with a reference direction, and its sign.

    Parameters
    ----------
    Ip : torch.Tensor
        Principal axes as rows, [num_graphs, 3, 3].
    direction : torch.Tensor
        Reference direction per graph, [num_graphs, 3].

    Returns
    -------
    overlaps : torch.Tensor
        ``Ip[n] @ direction[n]``, [num_graphs, 3].
    signs : torch.Tensor
        Sign of each overlap with exact zeros set to +1, [num_graphs, 3].
    """
    overlaps = torch.einsum('nij,nj->ni', (Ip,
                                           direction))  # Ip.dot(direction) # check if the principal components point towards or away from the CoG
    signs = torch.sign(overlaps)
    signs[signs == 0] = 1  # we want any exactly zero overlaps to come with positive signs
    return overlaps, signs


def batch_get_furthest_node_vector(all_coords: torch.FloatTensor, batch: torch.LongTensor, num_graphs: int) -> Tensor:
    """
    Return, for each graph, the coordinates of its node farthest from the origin.

    Distances are measured from the origin, so the result is the vector from
    each graph's centroid only when the coordinates are already centred per
    graph. Ties are resolved by ``torch_scatter.scatter_max``; the result is not
    unique for some symmetric inputs.

    Parameters
    ----------
    all_coords : torch.FloatTensor
        Node coordinates, [num_nodes, 3].
    batch : torch.LongTensor
        Graph index of each node, [num_nodes].
    num_graphs : int
        Number of graphs.

    Returns
    -------
    direction : torch.Tensor
        Farthest-node coordinates per graph, unnormalised, [num_graphs, 3].
    """
    dists = torch.linalg.norm(all_coords, axis=1)  # CoM is at 0,0,0 by construction
    max_dist, max_ind = scatter_max(dists, batch, dim_size=num_graphs)

    return all_coords[max_ind]


def nan_hook(name, tensor_ref, batch):
    """Return a tensor backward hook that raises on NaN gradients.

    The hook prints ``name``, ``tensor_ref`` and ``batch`` and raises
    ``AssertionError`` if the gradient contains NaN; otherwise it returns
    ``torch.nan_to_num(grad)``.
    """

    def _hook(grad):
        if torch.isnan(grad).any():
            print(f"NaNs in grad of {name}")
            print(f"Original tensor: {tensor_ref}")
            print(f"Batch: {batch}")
            assert False, "Stop the code!!"
        return torch.nan_to_num(grad)

    return _hook


def scatter_compute_Ip(all_coords, batch,
                       eps: float = 0.05,
                       add_noise: bool = False):
    """
    Compute inertial tensors (unit masses) and their principal axes for a batch of point clouds.

    The coordinates are used as given, without centring. When ``all_coords``
    requires grad or ``add_noise`` is True, Gaussian noise with standard
    deviation ``eps`` is added to the coordinates first, and every returned
    quantity is computed from the noised coordinates. The decomposition uses
    ``torch.linalg.eigh``; axis signs are as it returns them. If ``eigh``
    raises a RuntimeError on a CUDA tensor, it is repeated on CPU and the
    results are moved to ``'cuda'``. The number of graphs is
    ``batch.max() + 1``.

    Parameters
    ----------
    all_coords : torch.Tensor
        Point coordinates, [num_nodes, 3].
    batch : torch.LongTensor
        Graph index of each point, [num_nodes].
    eps : float
        Standard deviation of the added noise.
    add_noise : bool
        If True, add noise even when ``all_coords`` does not require grad.

    Returns
    -------
    Ip : torch.Tensor
        Principal axes (eigenvectors) as rows, sorted by ascending moment,
        [num_graphs, 3, 3].
    Ipm : torch.Tensor
        Principal moments (eigenvalues) in ascending order, [num_graphs, 3].
    inertial_tensor : torch.Tensor
        Inertial tensors, [num_graphs, 3, 3].

    Raises
    ------
    RuntimeError
        If ``torch.linalg.eigh`` raises a RuntimeError on a non-CUDA tensor (e.g.
        non-finite coordinates); the message names the graphs whose inertia tensor
        is not finite. An explicit raise, so it holds under ``python -O`` (before
        2026-10-22 a bare assert, and ``-O`` fell through to an UnboundLocalError).
    """

    """
    input is symmetric by construction
    when backpropagating, we need to ensure no degenerate eigenvalues
    empirically has to be bigger already than ~0.0127 to lift degeneracies
    taking default at 0.05 for safety
    """
    if all_coords.requires_grad or add_noise:
        coords_to_compute = all_coords + torch.randn_like(all_coords) * eps
    else:
        coords_to_compute = all_coords

    Ixy = -scatter(coords_to_compute[:, 0] * coords_to_compute[:, 1],
                   batch, reduce='sum')
    Iyz = -scatter(coords_to_compute[:, 1] * coords_to_compute[:, 2],
                   batch, reduce='sum')
    Ixz = -scatter(coords_to_compute[:, 0] * coords_to_compute[:, 2],
                   batch, reduce='sum')
    Ixx = scatter(coords_to_compute[:, 1] ** 2 + coords_to_compute[:, 2] ** 2,
                  batch, reduce='sum')
    Iyy = scatter(coords_to_compute[:, 0] ** 2 + coords_to_compute[:, 2] ** 2,
                  batch, reduce='sum')
    Izz = scatter(coords_to_compute[:, 0] ** 2 + coords_to_compute[:, 1] ** 2,
                  batch, reduce='sum')

    inertial_tensor = torch.cat(
        (torch.vstack((Ixx, Ixy, Ixz))[:, None, :].permute(2, 1, 0),
         torch.vstack((Ixy, Iyy, Iyz))[:, None, :].permute(2, 1, 0),
         torch.vstack((Ixz, Iyz, Izz))[:, None, :].permute(2, 1, 0)
         ), dim=-2)  # inertial tensor

    try:
        Ipm_c, Ip_c = torch.linalg.eigh(inertial_tensor)
    except RuntimeError as e:
        if 'cuda' in str(inertial_tensor.device):
            Ipm_c, Ip_c = torch.linalg.eigh(inertial_tensor.cpu())
            Ipm_c = Ipm_c.to('cuda')
            Ip_c = Ip_c.to('cuda')
        else:  # an explicit raise, so it holds under python -O (was `assert False, "Ipm error"`)
            bad = torch.nonzero(~torch.isfinite(inertial_tensor).flatten(1).all(1)).flatten().tolist()
            raise RuntimeError(f"scatter_compute_Ip: torch.linalg.eigh failed on the inertia tensors"
                               + (f"; graph(s) {bad[:8]}{' and more' if len(bad) > 8 else ''} of this batch have "
                                  f"non-finite coordinates" if bad else "") + f" ({e})") from e
    Ipms, Ip_o = torch.real(Ipm_c), torch.real(Ip_c)

    # Ip_o, Ipms, _ = torch.linalg.svd(inertial_tensor)  # superior numerical stability, yet equivalent for symmetric positive semi-definite

    Ips = Ip_o.permute(0, 2, 1)  # switch to row-wise eigenvectors

    sort_inds = torch.argsort(Ipms, dim=1)
    # too slow
    # Ipm = torch.stack([Ipms[i, sort_inds[i]] for i in range(len(sort_inds))])
    # Ip = torch.stack([Ips[i][sort_inds[i]] for i in range(len(sort_inds))])  # sort also the eigenvectors
    # much faster
    Ipm = torch.gather(Ipms, dim=1, index=sort_inds)
    Ip = torch.gather(Ips, dim=1, index=sort_inds.unsqueeze(2).expand(-1, -1, Ips.shape[2]))

    # if Ipms.requires_grad:
    #     coords_to_compute.retain_grad()
    #     coords_to_compute.register_hook(nan_hook("coords", coords_to_compute, batch))

    return Ip, Ipm, inertial_tensor


def extract_batching_info(nodes_list, device='cpu'):
    """
    Build a batch vector and cumulative pointer from a list of point sets.

    Parameters
    ----------
    nodes_list : list of torch.Tensor
        Point sets, each [num_points_i, ...]; only their lengths are used.
    device : str or torch.device
        Device of the returned tensors.

    Returns
    -------
    batch : torch.LongTensor
        Set index of each point, [num_nodes].
    ptrs : torch.Tensor
        Cumulative point counts starting at 0, [num_graphs + 1].
    """
    ptrs = [0]
    ptrs.extend([len(coord) for coord in nodes_list])
    ptrs = torch.tensor(ptrs, dtype=torch.int, device=device).cumsum(0)
    batch = torch.cat(
        [(i - 1) * torch.ones(ptrs[i] - ptrs[i - 1], dtype=torch.int64, device=device) for i in range(1, len(ptrs))])
    return batch, ptrs


def sph2cart_rotvec(angles):
    """
    Convert rotation vectors from spherical (theta, phi, r) to Cartesian components.

    ``rotvec = r * (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))``,
    with ``theta`` the polar angle of the rotation axis, ``phi`` its azimuthal
    angle and ``r`` the rotation angle, all in radians.

    Parameters
    ----------
    angles : np.ndarray or torch.Tensor
        Columns (theta, phi, r), [n, 3] or [3].

    Returns
    -------
    rotvec : np.ndarray or torch.Tensor or None
        Cartesian rotation vectors (x, y, z), same shape as ``angles``.
        ``None``, after printing a message, for any other input type.

    Notes
    -----
    The [3] torch branch passes the three scalar components to
    ``torch.Tensor(...)`` as separate arguments rather than stacking them;
    the [n, 3] torch branch and both numpy branches stack them.
    """

    if isinstance(angles, np.ndarray):
        if angles.ndim > 1:
            theta, phi, r = angles.T
            rotvec = r[:, None] * np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))).T
        else:
            theta, phi, r = angles
            rotvec = r * np.asarray((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))

        return rotvec

    elif torch.is_tensor(angles):
        if angles.ndim > 1:
            theta, phi, r = angles.T
            rotvec = r[:, None] * torch.stack((theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos())).T
        else:
            theta, phi, r = angles
            rotvec = r * torch.Tensor(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos())

        return rotvec

    else:
        print("Array type not supported! Must be np.ndarray or torch.tensor")
        return None


def cart2sph_rotvec(rotvec):
    """
    Convert Cartesian rotation vectors to spherical (theta, phi, r).

    ``r`` is the vector norm (the rotation angle), ``theta =
    atan2(sqrt(ux^2 + uy^2), uz)`` in [0, pi] is the polar angle of the unit
    axis ``u`` and ``phi = atan2(uy, ux)`` in [-pi, pi] is its azimuthal angle,
    all in radians. For torch input ``r = sqrt(|rotvec|^2 + 1e-12)`` and
    ``u = rotvec / max(r, 1e-8)``, so a zero vector gives finite values; for
    numpy input a zero vector divides by zero.

    Parameters
    ----------
    rotvec : np.ndarray or torch.Tensor
        Cartesian rotation vectors (x, y, z), [n, 3] or [3].

    Returns
    -------
    angles : np.ndarray or torch.Tensor or None
        Columns (theta, phi, r), [n, 3]. A [3] input is promoted to [1, 3] and
        the result keeps that shape. ``None``, after printing a message, for
        any other input type.
    """
    if isinstance(rotvec, np.ndarray):
        r = np.linalg.norm(rotvec, axis=-1)
        if rotvec.ndim == 1:
            rotvec = rotvec[None, :]
            r = np.asarray(r)[None]

        unit_vector = rotvec / r[:, None]

        # convert unit vector to angles
        theta = np.arctan2(np.sqrt(unit_vector[:, 0] ** 2 + unit_vector[:, 1] ** 2), unit_vector[:, 2])
        phi = np.arctan2(unit_vector[:, 1], unit_vector[:, 0])
        if rotvec.ndim == 1:
            return np.concatenate((theta, phi, r), axis=-1)  # polar, azimuthal, applied rotation
        else:
            return np.concatenate((theta[:, None], phi[:, None], r[:, None]),
                                  axis=-1)  # polar, azimuthal, applied rotation

    elif torch.is_tensor(rotvec):
        r = torch.sqrt((rotvec ** 2).sum(-1) + 1e-12)  # eps under sqrt → grad finite at 0, gradient-safe
        # r = torch.linalg.norm(rotvec, axis=-1)

        if rotvec.ndim == 1:
            rotvec = rotvec[None, :]
            r = torch.Tensor(r)[None]

        unit_vector = rotvec / r[:, None].clamp(min=1e-8)

        # convert unit vector to angles
        theta = torch.arctan2(torch.sqrt(unit_vector[:, 0] ** 2 + unit_vector[:, 1] ** 2), unit_vector[:, 2])
        phi = torch.arctan2(unit_vector[:, 1], unit_vector[:, 0])
        if rotvec.ndim == 1:
            return torch.cat((theta, phi, r), dim=-1)  # polar, azimuthal, applied rotation
        else:
            return torch.cat((theta[:, None], phi[:, None], r[:, None]), dim=-1)  # polar, azimuthal, applied rotation

    else:
        print("Array type not supported! Must be np.ndarray or torch.tensor")
        return None


def batch_compute_fractional_transform(cell_lengths, cell_angles):
    """
    Compute fractional-Cartesian transform matrices and cell volumes for a batch of cells.

    ``T_fc`` is upper triangular with the cell vectors as columns (a along x,
    b in the xy plane), so Cartesian coordinates are ``T_fc @ frac`` and
    fractional coordinates are ``T_cf @ cart``. The matrices are built with
    differentiable tensor operations, on the device and in the dtype of
    ``cell_lengths``. They use the signed volume ``sign(f) * a * b * c *
    sqrt(|f|)``, with ``f = 1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma) +
    2 cos(alpha) cos(beta) cos(gamma)``, which is negative for angle triples
    with ``f < 0``; the returned volume is its absolute value.

    Parameters
    ----------
    cell_lengths : torch.Tensor
        Cell lengths (a, b, c), [n, 3].
    cell_angles : torch.Tensor
        Cell angles (alpha, beta, gamma) in radians, [n, 3].

    Returns
    -------
    T_fc : torch.Tensor
        Fractional-to-Cartesian matrices, [n, 3, 3].
    T_cf : torch.Tensor
        Cartesian-to-fractional matrices, [n, 3, 3].
    cell_volumes : torch.Tensor
        Absolute cell volumes, in cubed length units, [n].
    """
    cos_a = torch.cos(cell_angles)
    sin_a = torch.sin(cell_angles)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[:, 0] ** 2 - cos_a[:, 1] ** 2 - cos_a[:, 2] ** 2 + 2.0 * cos_a[:, 0] * cos_a[:, 1] * cos_a[:, 2]

    vol = torch.sign(val) * torch.prod(cell_lengths, dim=1) * torch.sqrt(
        torch.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    T_fc_list = torch.zeros((len(cell_lengths), 3, 3), device=cell_lengths.device, dtype=cell_lengths.dtype)
    T_cf_list = torch.zeros((len(cell_lengths), 3, 3), device=cell_lengths.device, dtype=cell_lengths.dtype)

    ''' Converting from cartesian to fractional '''
    T_cf_list[:, 0, 0] = 1.0 / cell_lengths[:, 0]
    T_cf_list[:, 0, 1] = -cos_a[:, 2] / cell_lengths[:, 0] / sin_a[:, 2]
    T_cf_list[:, 0, 2] = cell_lengths[:, 1] * cell_lengths[:, 2] * (
            cos_a[:, 0] * cos_a[:, 2] - cos_a[:, 1]) / vol / sin_a[:, 2]
    T_cf_list[:, 1, 1] = 1.0 / cell_lengths[:, 1] / sin_a[:, 2]
    T_cf_list[:, 1, 2] = cell_lengths[:, 0] * cell_lengths[:, 2] * (
            cos_a[:, 1] * cos_a[:, 2] - cos_a[:, 0]) / vol / sin_a[:, 2]
    T_cf_list[:, 2, 2] = cell_lengths[:, 0] * cell_lengths[:, 1] * sin_a[:, 2] / vol
    ''' Converting from fractional to cartesian '''
    T_fc_list[:, 0, 0] = cell_lengths[:, 0]
    T_fc_list[:, 0, 1] = cell_lengths[:, 1] * cos_a[:, 2]
    T_fc_list[:, 0, 2] = cell_lengths[:, 2] * cos_a[:, 1]
    T_fc_list[:, 1, 1] = cell_lengths[:, 1] * sin_a[:, 2]
    T_fc_list[:, 1, 2] = cell_lengths[:, 2] * (cos_a[:, 0] - cos_a[:, 1] * cos_a[:, 2]) / sin_a[:, 2]
    T_fc_list[:, 2, 2] = vol / cell_lengths[:, 0] / cell_lengths[:, 1] / sin_a[:, 2]

    return T_fc_list, T_cf_list, torch.abs(vol)


def cell_vol_torch(v: torch.tensor, a: torch.tensor):
    """
    Compute the volume of one cell from its lengths and angles.

    ``a * b * c * sqrt(|1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma) +
    2 cos(alpha) cos(beta) cos(gamma)|)``.

    Parameters
    ----------
    v : torch.Tensor
        Cell lengths (a, b, c), [3].
    a : torch.Tensor
        Cell angles (alpha, beta, gamma) in radians, [3].

    Returns
    -------
    cell_volume : torch.Tensor
        Volume in cubed length units, 0-d.
    """
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(a)  # in natural units

    ''' Calculate volume of the unit cell '''
    vol = v[0] * v[1] * v[2] * torch.sqrt(
        torch.abs(1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]))

    return vol


def batch_cell_vol_torch(v: torch.tensor, a: torch.tensor):
    """
    Compute cell volumes for a batch of cells from their lengths and angles.

    Same formula as ``cell_vol_torch``, applied row-wise; the result is
    non-negative for positive lengths.

    Parameters
    ----------
    v : torch.Tensor
        Cell lengths (a, b, c), [n, 3].
    a : torch.Tensor
        Cell angles (alpha, beta, gamma) in radians, [n, 3].

    Returns
    -------
    cell_volumes : torch.Tensor
        Volumes in cubed length units, [n].
    """
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(a)  # in natural units

    ''' Calculate volume of the unit cell '''
    vol = v[:, 0] * v[:, 1] * v[:, 2] * torch.sqrt(
        torch.abs(
            1.0 - cos_a[:, 0] ** 2 - cos_a[:, 1] ** 2 - cos_a[:, 2] ** 2 + 2.0 * cos_a[:, 0] * cos_a[:, 1] * cos_a[:,
            2]))

    return vol


def cell_vol_angle_factor(cell_angles):
    """
    Compute the angular factor sqrt(1 - cos²α - cos²β - cos²γ + 2cosα cosβ cosγ) used in cell volume calculations.

    Parameters
    ----------
    cell_angles : torch.tensor(..., 3)
        [alpha, beta, gamma] in radians; leading batch dimensions are supported

    Returns
    -------
    factor : torch.tensor(...)
    """
    cos_a = torch.cos(cell_angles)  # in natural units

    return torch.sqrt(
        torch.abs(1.0 - cos_a[..., 0] ** 2 - cos_a[..., 1] ** 2 - cos_a[..., 2] ** 2
                  + 2.0 * cos_a[..., 0] * cos_a[..., 1] * cos_a[..., 2]))


def compute_Ip_handedness(Ip):
    """
    Return the handedness of axis bases as the sign of ``Ip[0] . (Ip[1] x Ip[2])``.

    Parameters
    ----------
    Ip : np.ndarray or torch.Tensor
        Axes as rows, [3, 3] or [n, 3, 3].

    Returns
    -------
    handedness : scalar, np.ndarray or torch.Tensor
        +1 for right-handed, -1 for left-handed, 0 for a degenerate basis.
        A scalar for [3, 3] input (a 0-d float tensor for torch) and [n] for
        [n, 3, 3] input. For numpy [n, 3, 3] input, entry ``i`` is the sign of
        ``sum_j Ip[i, 0] . (Ip[j, 1] x Ip[j, 2])``, summed over every sample
        ``j``, not only ``j = i``. Other dimensionalities return None.

    Raises
    ------
    SystemExit
        If ``Ip`` is neither a numpy array nor a torch tensor (a message is
        printed first).
    """
    if isinstance(Ip, np.ndarray):
        if Ip.ndim == 2:
            return np.sign(np.dot(Ip[0], np.cross(Ip[1], Ip[2])).sum())
        elif Ip.ndim == 3:
            return np.sign(np.dot(Ip[:, 0], np.cross(Ip[:, 1], Ip[:, 2], axis=1).T).sum(1))

    elif torch.is_tensor(Ip):
        if Ip.ndim == 2:
            return torch.sign(torch.mul(Ip[0], torch.cross(Ip[1], Ip[2], dim=0)).sum()).float()
        elif Ip.ndim == 3:
            return torch.sign(torch.mul(Ip[:, 0], torch.cross(Ip[:, 1], Ip[:, 2], dim=1)).sum(1))

    else:
        print("Ip handedness calculation failed! Inputs were neither torch.tensor or numpy.array")
        sys.exit()


def cell_vol_np(v, a):
    """
    Compute the volume of one cell from its lengths and angles, with numpy.

    Same formula as ``cell_vol_torch``; the result is non-negative for
    positive lengths.

    Parameters
    ----------
    v : np.ndarray
        Cell lengths (a, b, c), [3].
    a : np.ndarray
        Cell angles (alpha, beta, gamma) in radians, [3].

    Returns
    -------
    cell_volume : float
        Volume in cubed length units.
    """
    """ Calculate cos and sin of cell angles """
    cos_a = np.cos(a)  # in natural units

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = v[0] * v[1] * v[2] * np.sqrt(np.abs(val))  # technically a signed quanitity

    return vol


def coor_trans_matrix_np(opt, v, a, return_vol=False):
    """
    Compute one fractional-Cartesian transform matrix with numpy.

    Same matrix elements as ``batch_compute_fractional_transform`` for a
    single cell, in float64. Prints a warning if any angle exceeds pi.

    Parameters
    ----------
    opt : str
        ``'c_to_f'`` for Cartesian-to-fractional or ``'f_to_c'`` for
        fractional-to-Cartesian; any other value returns a zero matrix.
    v : np.ndarray
        Cell lengths (a, b, c), [3].
    a : np.ndarray
        Cell angles (alpha, beta, gamma) in radians, [3].
    return_vol : bool, optional
        If True, also return the absolute cell volume.

    Returns
    -------
    transform : np.ndarray
        Transform matrix, [3, 3].
    cell_volume : float
        Absolute cell volume; returned only when ``return_vol`` is True.
    """
    """ Calculate cos and sin of cell angles """  # todo test - enforce this agrees with the torch version
    if np.amax(a) > np.pi:
        print('Warning - large angles! Remember to convert to natural units!')

    cos_a = np.cos(a)
    sin_a = np.sin(a)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = np.sign(val) * v[0] * v[1] * v[2] * np.sqrt(np.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    m = np.zeros((3, 3), dtype=np.float64)
    if opt == 'c_to_f':
        ''' Converting from cartesian to fractional '''
        m[0, 0] = 1.0 / v[0]
        m[0, 1] = -cos_a[2] / v[0] / sin_a[2]
        m[0, 2] = v[1] * v[2] * (cos_a[0] * cos_a[2] - cos_a[1]) / vol / sin_a[2]
        m[1, 1] = 1.0 / v[1] / sin_a[2]
        m[1, 2] = v[0] * v[2] * (cos_a[1] * cos_a[2] - cos_a[0]) / vol / sin_a[2]
        m[2, 2] = v[0] * v[1] * sin_a[2] / vol
    elif opt == 'f_to_c':
        ''' Converting from fractional to cartesian '''
        m[0, 0] = v[0]
        m[0, 1] = v[1] * cos_a[2]
        m[0, 2] = v[2] * cos_a[1]
        m[1, 1] = v[1] * sin_a[2]
        m[1, 2] = v[2] * (cos_a[0] - cos_a[1] * cos_a[2]) / sin_a[2]
        m[2, 2] = vol / v[0] / v[1] / sin_a[2]

    if return_vol:
        return m, np.abs(vol)
    else:
        return m


def mol_batch_vdW_volume(mol_batch):
    """
    Estimate vdW volumes for a molecule batch with ``batch_compute_molecule_volume``.

    Reads ``mol_batch.z``, ``mol_batch.pos``, ``mol_batch.batch`` and
    ``mol_batch.num_graphs``, and passes the values of ``VDW_RADII`` (keyed by
    atomic number) as a float32 lookup on the device of ``mol_batch.z``.

    Parameters
    ----------
    mol_batch
        Batched molecule data object.

    Returns
    -------
    volumes : torch.Tensor
        Estimated molecular volumes, [num_graphs].
    """
    return batch_compute_molecule_volume(
        mol_batch.z,
        mol_batch.pos,
        mol_batch.batch,
        mol_batch.num_graphs,
        torch.tensor(list(VDW_RADII.values()),
                     device=mol_batch.z.device,
                     dtype=torch.float32))


def batch_compute_molecule_volume(
        atom_types: torch.LongTensor,
        pos: torch.FloatTensor,
        batch: torch.LongTensor,
        num_graphs: int,
        vdw_radii_tensor: torch.Tensor
):
    """
    Estimate molecular vdW volumes for a batch of molecules from atomic spheres and pairwise overlaps.

    The volume is the sum of atomic sphere volumes minus 0.7272 times the sum
    of pairwise sphere-sphere intersection (lens) volumes. Pairs are searched
    within ``2 * vdw_radii_tensor.max()`` inside each molecule (at most 100
    neighbours per atom), each unordered pair is counted once, and pairs
    farther apart than the sum of their radii contribute zero. Volumes are in
    cubed units of the radii and positions.

    Parameters
    ----------
    atom_types : torch.LongTensor
        Atomic numbers used to index ``vdw_radii_tensor``, [num_nodes].
    pos : torch.FloatTensor
        Atomic coordinates, [num_nodes, 3].
    batch : torch.LongTensor
        Graph index of each atom, [num_nodes].
    num_graphs : int
        Number of graphs.
    vdw_radii_tensor : torch.Tensor
        vdW radius lookup indexed by atomic number.

    Returns
    -------
    corrected_mol_volume : torch.FloatTensor
        Estimated volumes, [num_graphs].
    """
    atom_volumes = 4 / 3 * torch.pi * vdw_radii_tensor[atom_types] ** 3
    raw_vdw_volumes = scatter(atom_volumes, batch, dim=0, dim_size=num_graphs, reduce='sum')
    bonds_i, bonds_j = radius(pos, pos,
                              r=2 * vdw_radii_tensor.max(),
                              batch_x=batch,
                              batch_y=batch,
                              max_num_neighbors=100)
    mask = ~(bonds_i >= bonds_j)  # eliminate duplicates
    bonds_i, bonds_j = bonds_i[mask], bonds_j[mask]
    bond_lengths = torch.linalg.norm(pos[bonds_i] - pos[bonds_j], dim=1)
    radii_i, radii_j = vdw_radii_tensor[atom_types[bonds_i]], vdw_radii_tensor[atom_types[bonds_j]]
    # https://mathworld.wolfram.com/Sphere-SphereIntersection.html
    # c1 = pi*(r1+r2-d)^2
    # c2 = d^2 + 2dr1 + 2dr2 + 6r1r2 - 3r1^2 - 3r2^2
    # c3 = 12d
    c1 = torch.pi * (radii_i + radii_j - bond_lengths) ** 2
    c2 = bond_lengths ** 2 + 2 * bond_lengths * (
            radii_i + radii_j) - 3 * radii_i ** 2 - 3 * radii_j ** 2 + 6 * radii_i * radii_j
    c3 = 12 * bond_lengths
    # sphere_overlaps = (torch.pi * (radii_i + radii_j - bond_lengths) ** 2 *
    #                    (bond_lengths ** 2 + 2 * bond_lengths * radii_j - 3 * radii_j ** 2
    #                     + 2 * bond_lengths * radii_i + 6 * radii_j * radii_i - 3 * radii_i ** 2) / (12 * bond_lengths))
    sphere_overlaps = c1 * c2 / c3
    sphere_overlaps[bond_lengths > (radii_i + radii_j)] = 0
    molwise_sphere_overlaps = scatter(sphere_overlaps, batch[bonds_i], dim=0, dim_size=num_graphs,
                                      reduce='sum')

    # estimate correction factor (omits triple overlaps) by comparison of representative molecule batch with well-converged probe method
    corrected_mol_volume = raw_vdw_volumes - molwise_sphere_overlaps * 0.7272  # a very coarse correction factor
    return corrected_mol_volume


def probe_compute_molecule_volume(
        atom_types: torch.LongTensor,
        pos: torch.FloatTensor,
        batch: torch.LongTensor,
        num_graphs: int,
        vdw_radii_tensor: torch.Tensor,
        probes_per_mol: int = 100,
        eps: float = 1e-2,
        max_iters: int = 1000,
        min_iters: int = 5
):
    """
    Estimate molecular vdW volumes by Monte Carlo sampling in each molecule's bounding box.

    Each iteration draws ``probes_per_mol`` uniform points in each molecule's
    axis-aligned bounding box, padded on every side by
    ``vdw_radii_tensor.max()``, and takes the box volume times the fraction of
    probes inside at least one atomic sphere. Once more than ``min_iters``
    iterations have run, the loop stops when, for every molecule, the mean
    absolute step of the running mean (normalised by that molecule's average
    running mean) over the last ``min_iters`` steps is below ``eps``; it also
    stops after ``max_iters`` iterations. Volumes are in cubed units of the
    radii and positions.

    Parameters
    ----------
    atom_types : torch.LongTensor
        Atomic numbers used to index ``vdw_radii_tensor``, [num_nodes].
    pos : torch.FloatTensor
        Atomic coordinates, [num_nodes, 3].
    batch : torch.LongTensor
        Graph index of each atom, [num_nodes].
    num_graphs : int
        Number of graphs.
    vdw_radii_tensor : torch.Tensor
        vdW radius lookup indexed by atomic number.
    probes_per_mol : int
        Number of probes per molecule per iteration.
    eps : float
        Convergence threshold on the normalised step of the running mean.
    max_iters : int
        Maximum number of iterations; must exceed ``min_iters``.
    min_iters : int
        Number of iterations before convergence is checked, and the window of
        steps the check averages over.

    Returns
    -------
    volumes : torch.FloatTensor
        Mean of all per-iteration estimates, [num_graphs].

    Raises
    ------
    UnboundLocalError
        If the loop ends before any convergence check has run
        (``max_iters <= min_iters``).
    """
    volume_record = []
    converged = False
    iter = 0
    while not converged and iter < max_iters:
        iter += 1
        mol_min_corner = scatter(pos, batch, dim=0, reduce='min') - vdw_radii_tensor.max()
        mol_max_corner = scatter(pos, batch, dim=0, reduce='max') + vdw_radii_tensor.max()
        bounding_volumes = torch.prod(mol_max_corner - mol_min_corner, dim=1)
        probe_batch = torch.arange(num_graphs, device=batch.device).repeat_interleave(probes_per_mol)
        random_probes = mol_min_corner[probe_batch] + (
                mol_max_corner[probe_batch] - mol_min_corner[probe_batch]) * torch.rand(
            (probes_per_mol * num_graphs, 3), device=pos.device)
        # Compute squared distances between each probe and each atom
        edge_i, edge_j = radius(x=pos,
                                y=random_probes,
                                r=vdw_radii_tensor.max(),
                                batch_x=batch,
                                batch_y=probe_batch,
                                max_num_neighbors=probes_per_mol)
        dists = torch.linalg.norm(random_probes[edge_i] - pos[edge_j], dim=1)

        # Check if each probe is inside any sphere
        inside_any_sphere = (dists < vdw_radii_tensor[atom_types[edge_j]]).float()

        # we want a maximum of one contact per probe
        probe_has_a_contact = scatter(inside_any_sphere, edge_i, reduce='max', dim_size=probes_per_mol * num_graphs,
                                      dim=0)
        inside_sphere_frac = scatter(probe_has_a_contact, probe_batch, reduce='sum', dim=0,
                                     dim_size=num_graphs) / probes_per_mol

        # Estimate molecular volume
        mol_volume = bounding_volumes * inside_sphere_frac
        volume_record.append(mol_volume)
        if iter > min_iters:
            volumes = torch.stack(volume_record)
            cum_vols = torch.cumsum(volumes, dim=0)
            cum_iters = torch.arange(1, len(volumes) + 1, device=volumes.device)[:, None]
            cum_means = cum_vols / cum_iters
            rel_diffs = torch.diff(cum_means / cum_means.mean(0)[None, :], dim=0)
            criteria = rel_diffs[-min_iters:].abs().mean(0)
            if torch.all(criteria < eps):  # relative change in running average less than eps
                converged = True

    return cum_means[-1, :]


def grid_compute_molecule_volume(atom_types, pos, vdw_radii_tensor, eps):
    """
    Estimate the vdW volume of one molecule by counting occupied points on refining grids.

    The grid spans the atoms' bounding box padded by ``vdw_radii_tensor.amax()``.
    Its nominal spacing starts at 0.075 and is multiplied by 0.75 on each
    iteration; a grid point is occupied if it lies within the vdW radius of
    any atom. From the third iteration on, the loop stops when the change
    between the last two estimates divided by the second estimate is below
    ``eps``, printing the iteration index and that ratio at each check; it
    runs at most 11 iterations. Grids are created on the default device.

    Parameters
    ----------
    atom_types : torch.LongTensor
        Atomic numbers used to index ``vdw_radii_tensor``, [num_atoms].
    pos : torch.Tensor
        Atomic coordinates, [num_atoms, 3].
    vdw_radii_tensor : torch.Tensor
        vdW radius lookup indexed by atomic number.
    eps : float
        Convergence threshold on the relative change between iterations.

    Returns
    -------
    occupied_volume : torch.Tensor
        Box volume times the occupied fraction of the last grid, 0-d.
    """
    convergence_history = []
    dx = 0.1
    converged = False
    ind = -1
    max_iters = 10
    while converged is False and ind < max_iters:
        dx *= 0.75
        xmin, ymin, zmin = (pos.amin(0) - vdw_radii_tensor.amax())
        xmax, ymax, zmax = (pos.amax(0) + vdw_radii_tensor.amax())
        num_x = int((xmax - xmin) / dx)
        num_y = int((ymax - ymin) / dx)
        num_z = int((zmax - zmin) / dx)
        grid = torch.meshgrid(torch.linspace(xmin, xmax, num_x),
                              torch.linspace(ymin, ymax, num_y),
                              torch.linspace(zmin, zmax, num_z),
                              indexing='xy'
                              )
        grid = torch.stack(grid)
        grid_flat = grid.reshape(3, num_x * num_y * num_z).T

        edges = radius(x=grid_flat, y=pos, r=vdw_radii_tensor.amax(),
                       max_num_neighbors=int(1e12))

        dists = torch.linalg.norm(pos[edges[0]] - grid_flat[edges[1]], dim=1)
        close_enough = dists <= vdw_radii_tensor[atom_types[edges[0]]]

        overlapped_points = len(edges[1, close_enough].unique())
        box_volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        overlapping_fraction = overlapped_points / len(grid_flat)
        occupied_volume = box_volume * overlapping_fraction
        convergence_history.append(float(occupied_volume))
        ind += 1
        if ind > 1:
            conv = abs(convergence_history[-2] - convergence_history[-1]) / convergence_history[1]
            if conv < eps:
                converged = True

            print(ind)
            print(conv)

    return occupied_volume


def grid_compute_molecule_volume_pointwise(atom_types, pos, vdw_radii_tensor, eps):
    """
    Estimate the vdW volume of one molecule on refining grids.

    The body is identical to ``grid_compute_molecule_volume``; see that
    function for the procedure.

    Parameters
    ----------
    atom_types : torch.LongTensor
        Atomic numbers used to index ``vdw_radii_tensor``, [num_atoms].
    pos : torch.Tensor
        Atomic coordinates, [num_atoms, 3].
    vdw_radii_tensor : torch.Tensor
        vdW radius lookup indexed by atomic number.
    eps : float
        Convergence threshold on the relative change between iterations.

    Returns
    -------
    occupied_volume : torch.Tensor
        Box volume times the occupied fraction of the last grid, 0-d.
    """
    convergence_history = []
    dx = 0.1
    converged = False
    ind = -1
    max_iters = 10
    while converged is False and ind < max_iters:
        dx *= 0.75
        xmin, ymin, zmin = (pos.amin(0) - vdw_radii_tensor.amax())
        xmax, ymax, zmax = (pos.amax(0) + vdw_radii_tensor.amax())
        num_x = int((xmax - xmin) / dx)
        num_y = int((ymax - ymin) / dx)
        num_z = int((zmax - zmin) / dx)
        grid = torch.meshgrid(torch.linspace(xmin, xmax, num_x),
                              torch.linspace(ymin, ymax, num_y),
                              torch.linspace(zmin, zmax, num_z),
                              indexing='xy'
                              )
        grid = torch.stack(grid)
        grid_flat = grid.reshape(3, num_x * num_y * num_z).T

        edges = radius(x=grid_flat, y=pos, r=vdw_radii_tensor.amax(),
                       max_num_neighbors=int(1e12))

        dists = torch.linalg.norm(pos[edges[0]] - grid_flat[edges[1]], dim=1)
        close_enough = dists <= vdw_radii_tensor[atom_types[edges[0]]]

        overlapped_points = len(edges[1, close_enough].unique())
        box_volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        overlapping_fraction = overlapped_points / len(grid_flat)
        occupied_volume = box_volume * overlapping_fraction
        convergence_history.append(float(occupied_volume))
        ind += 1
        if ind > 1:
            conv = abs(convergence_history[-2] - convergence_history[-1]) / convergence_history[1]
            if conv < eps:
                converged = True

            print(ind)
            print(conv)

    return occupied_volume


def norm_circular_components(components: torch.tensor):
    """
    Scale each row of a two-component tensor to unit Euclidean norm.

    Parameters
    ----------
    components : torch.Tensor
        [n, 2].

    Returns
    -------
    normed_components : torch.Tensor
        [n, 2].
    """

    return components / torch.sqrt(torch.sum(components ** 2, dim=-1))[:, None]


def components2angle(components: torch.tensor, norm_components=True):
    """
    Convert (sin, cos) component pairs to angles.

    Returns ``atan2(components[:, 0], components[:, 1])`` in radians, in
    [-pi, pi], following
    https://ai.stackexchange.com/questions/38045/how-can-i-encode-angle-data-to-train-neural-networks.
    With ``norm_components`` the rows are first scaled to unit norm, which
    leaves the angle of any nonzero row unchanged.

    Parameters
    ----------
    components : torch.Tensor
        Unnormalised (sin, cos) pairs, [n, 2].
    norm_components : bool, optional
        If True (default), normalise the rows with
        ``norm_circular_components`` first.

    Returns
    -------
    angles : torch.Tensor
        [n].
    """

    if norm_components:
        normed_components = norm_circular_components(components)
        angles = torch.atan2(normed_components[:, 0], normed_components[:, 1])
    else:
        angles = torch.atan2(components[:, 0], components[:, 1])

    return angles


def angle2components(angle: torch.tensor):
    """
    Decompose angles into (sin, cos) columns.

    Parameters
    ----------
    angle : torch.Tensor
        Angles in radians, [n].

    Returns
    -------
    components : torch.Tensor
        Columns (sin(angle), cos(angle)), [n, 2].
    """

    return torch.cat((torch.sin(angle)[:, None], torch.cos(angle)[:, None]), dim=1)


def old_enforce_crystal_system(lattice_lengths,
                               lattice_angles,
                               sg_inds,
                               symmetries_dict: Optional[dict] = None
                               ):
    """
    Project cell parameters onto their crystal-system constraints, one sample at a time.

    Lattice names come from ``symmetries_dict['lattice_type']`` indexed by
    space group (case-insensitive). Unlike ``enforce_crystal_system``, tied
    parameters copy the first component instead of averaging: tetragonal and
    hexagonal set a and b to a; rhombohedral sets a, b and c to a and all
    angles to alpha; cubic sets a, b and c to a. Angles: monoclinic fixes
    alpha and gamma to pi/2; orthorhombic, tetragonal and cubic fix all angles
    to pi/2; hexagonal fixes alpha and beta to pi/2 and gamma to 2 pi/3.
    Triclinic is unchanged. Returns new tensors; the inputs are not modified.
    https://en.wikipedia.org/wiki/Crystal_system

    Parameters
    ----------
    lattice_lengths : torch.Tensor
        Cell lengths (a, b, c), [n, 3].
    lattice_angles : torch.Tensor
        Cell angles (alpha, beta, gamma) in radians, [n, 3].
    sg_inds : torch.Tensor
        Space-group indices, [n]; a 0-d index yields one lattice name, so only
        n == 1 works with it.
    symmetries_dict : dict, optional
        Symmetry info with a ``'lattice_type'`` lookup; built with
        ``init_sym_info()`` when None.

    Returns
    -------
    fixed_lengths : torch.Tensor
        [n, 3].
    fixed_angles : torch.Tensor
        [n, 3].

    Raises
    ------
    SystemExit
        If a lattice name is not one of the handled systems (a message is
        printed first).
    """  # todo vectorize this function, and clean it up in general

    if symmetries_dict is None:
        symmetries_dict = init_sym_info()
    if sg_inds.ndim == 0:
        lattices = [symmetries_dict['lattice_type'][int(sg_inds)]]
    else:
        lattices = [symmetries_dict['lattice_type'][int(sg_inds[n])] for n in range(len(lattice_lengths))]

    pi_tensor = torch.ones_like(lattice_lengths[0, 0]) * torch.pi

    fixed_lengths = torch.zeros_like(lattice_lengths)
    fixed_angles = torch.zeros_like(lattice_angles)

    for i in range(len(lattice_lengths)):
        lengths = lattice_lengths[i]
        angles = lattice_angles[i]
        lattice = lattices[i]
        # enforce agreement with crystal system
        if lattice.lower() == 'triclinic':  # anything goes
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = angles * 1

        elif lattice.lower() == 'monoclinic':  # fix alpha and gamma to pi/2
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, angles[1], pi_tensor.clone() / 2,
            ), dim=- 1)

        elif lattice.lower() == 'orthorhombic':  # fix all angles at pi/2
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)

        elif lattice.lower() == 'tetragonal':  # fix all angles pi/2 and a=b
            mean_tensor = lengths[0] * 1
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, lengths[2] * 1,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)

        elif lattice.lower() == 'hexagonal':
            # a=b
            # alpha beta are pi/2, gamma is 2pi/3

            mean_tensor = lengths[0] * 1
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, lengths[2] * 1,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() * 2 / 3,
            ), dim=- 1)

        # elif lattice.lower()  == 'trigonal':

        elif lattice.lower() == 'rhombohedral':
            # mean of abc vector lengths
            # mean of all angles

            mean_tensor = lengths[0] * 1
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, mean_tensor,
            ), dim=- 1)

            mean_angle = angles[0]
            fixed_angles[i] = torch.stack((
                mean_angle, mean_angle, mean_angle,
            ), dim=- 1)

        elif lattice.lower() == 'cubic':  # all angles 90 all lengths equal
            mean_tensor = lengths[0] * 1
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, mean_tensor,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)
        else:
            print(lattice + ' is not a valid crystal lattice!')
            sys.exit()

    return fixed_lengths, fixed_angles


def enforce_crystal_system2(lattice_lengths, lattice_angles, lattices):
    """
    Project cell parameters onto their crystal-system constraints, given lattice names.

    Per-sample loop. Triclinic: unchanged. Monoclinic: alpha = gamma = pi/2.
    Orthorhombic: all angles pi/2. Tetragonal: a and b set to their mean, all
    angles pi/2. Hexagonal: a and b set to their mean, alpha = beta = pi/2,
    gamma = 2 pi/3. Rhombohedral: a, b and c set to their mean, all angles set
    to their mean. Cubic: a, b and c set to their mean, all angles pi/2.
    Returns new tensors; the inputs are not modified.
    https://en.wikipedia.org/wiki/Crystal_system

    Parameters
    ----------
    lattice_lengths : torch.Tensor
        Cell lengths (a, b, c), [n, 3].
    lattice_angles : torch.Tensor
        Cell angles (alpha, beta, gamma) in radians, [n, 3].
    lattices : sequence of str
        Lattice name per sample (case-insensitive), length n.

    Returns
    -------
    fixed_lengths : torch.Tensor
        [n, 3].
    fixed_angles : torch.Tensor
        [n, 3].

    Raises
    ------
    SystemExit
        If a lattice name is not one of the handled systems (a message is
        printed first).
    """  # todo double check these limits

    pi_tensor = torch.ones_like(lattice_lengths[0, 0]) * torch.pi

    fixed_lengths = torch.zeros_like(lattice_lengths)
    fixed_angles = torch.zeros_like(lattice_angles)

    for i in range(len(lattice_lengths)):
        lengths = lattice_lengths[i]
        angles = lattice_angles[i]
        lattice = lattices[i]
        # enforce agreement with crystal system
        if lattice.lower() == 'triclinic':  # anything goes
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = angles * 1

        elif lattice.lower() == 'monoclinic':  # fix alpha and gamma to pi/2
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, angles[1], pi_tensor.clone() / 2,
            ), dim=- 1)
        elif lattice.lower() == 'orthorhombic':  # fix all angles at pi/2
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)
        elif lattice.lower() == 'tetragonal':  # fix all angles pi/2 and take the mean of a & b vectors
            mean_tensor = torch.mean(lengths[0:2])
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, lengths[2] * 1,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)

        elif lattice.lower() == 'hexagonal':
            # mean of ab, c is free
            # alpha beta are pi/2, gamma is 2pi/3

            mean_tensor = torch.mean(lengths[0:2])
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, lengths[2] * 1,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() * 2 / 3,
            ), dim=- 1)

        # elif lattice.lower()  == 'trigonal':

        elif lattice.lower() == 'rhombohedral':
            # mean of abc vector lengths
            # mean of all angles

            mean_tensor = torch.mean(lengths)
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, mean_tensor,
            ), dim=- 1)

            mean_angle = torch.mean(angles)
            fixed_angles[i] = torch.stack((
                mean_angle, mean_angle, mean_angle,
            ), dim=- 1)

        elif lattice.lower() == 'cubic':  # all angles 90 all lengths equal
            mean_tensor = torch.mean(lengths)
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, mean_tensor,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)
        else:
            print(lattice + ' is not a valid crystal lattice!')
            sys.exit()

    return fixed_lengths, fixed_angles


def enforce_crystal_system(
        lattice_lengths,
        lattice_angles,
        sg_inds,
        symmetries_dict: Optional[dict] = None,
):
    """
    Project cell parameters onto the constraints of each sample's crystal system.

    Vectorised over the batch; the inputs are cloned, not modified. The
    lattice system of each sample is ``symmetries_dict["lattice_code"]``
    indexed by its space group, with codes from ``LATTICE_TO_CODE``:

    - triclinic: unchanged;
    - monoclinic: alpha = gamma = pi/2;
    - orthorhombic: all angles pi/2;
    - tetragonal: a and b set to their mean, all angles pi/2;
    - hexagonal: a and b set to their mean, alpha = beta = pi/2,
      gamma = 2 pi/3;
    - rhombohedral: a, b and c set to their mean, all angles set to their mean;
    - cubic: a, b and c set to their mean, all angles pi/2.

    Parameters
    ----------
    lattice_lengths : torch.Tensor
        Cell lengths (a, b, c), [B, 3] or [3].
    lattice_angles : torch.Tensor
        Cell angles (alpha, beta, gamma) in radians, [B, 3] or [3].
    sg_inds : int or torch.Tensor
        Space-group indices, scalar or [B]; a scalar applies to every sample.
    symmetries_dict : dict, optional
        Must contain ``"lattice_code"``, a LongTensor mapping space-group index
        to lattice code; built with ``init_sym_info()`` when None.

    Returns
    -------
    fixed_lengths : torch.Tensor
        Same shape as ``lattice_lengths``.
    fixed_angles : torch.Tensor
        Same shape as ``lattice_angles``.

    Raises
    ------
    ValueError
        If any space-group index maps to a negative lattice code (with
        ``init_sym_info()``, index 0).
    """

    if symmetries_dict is None:
        symmetries_dict = init_sym_info()

    squeeze = False
    if lattice_lengths.ndim == 1:
        lattice_lengths = lattice_lengths[None, :]
        lattice_angles = lattice_angles[None, :]
        squeeze = True

    device = lattice_lengths.device
    dtype = lattice_lengths.dtype
    bsz = len(lattice_lengths)

    sg_inds = torch.as_tensor(sg_inds, device=device, dtype=torch.long)
    if sg_inds.ndim == 0:
        sg_inds = sg_inds.expand(bsz)

    lattice_codes = symmetries_dict["lattice_code"].to(sg_inds.device)[sg_inds.long()]

    if (lattice_codes < 0).any():
        bad_sgs = sg_inds[lattice_codes < 0].detach().cpu().unique().tolist()
        raise ValueError(f"Invalid or unmapped space group indices: {bad_sgs}")

    fixed_lengths = lattice_lengths.clone()
    fixed_angles = lattice_angles.clone()

    half_pi = lattice_lengths.new_tensor(torch.pi / 2)
    two_pi_over_three = lattice_lengths.new_tensor(2 * torch.pi / 3)

    # 0: triclinic, no-op

    # 1: monoclinic: alpha = gamma = 90
    m = lattice_codes == LATTICE_TO_CODE["monoclinic"]
    fixed_angles[m, 0] = half_pi
    fixed_angles[m, 2] = half_pi

    # 2: orthorhombic: alpha = beta = gamma = 90
    m = lattice_codes == LATTICE_TO_CODE["orthorhombic"]
    fixed_angles[m, :] = half_pi

    # 3: tetragonal: a = b, angles = 90
    m = lattice_codes == LATTICE_TO_CODE["tetragonal"]
    if m.any():
        ab_mean = lattice_lengths[m, :2].mean(dim=-1)
        fixed_lengths[m, 0] = ab_mean
        fixed_lengths[m, 1] = ab_mean
        fixed_angles[m, :] = half_pi

    # 4: hexagonal: a = b, alpha = beta = 90, gamma = 120
    m = lattice_codes == LATTICE_TO_CODE["hexagonal"]
    if m.any():
        ab_mean = lattice_lengths[m, :2].mean(dim=-1)
        fixed_lengths[m, 0] = ab_mean
        fixed_lengths[m, 1] = ab_mean
        fixed_angles[m, 0] = half_pi
        fixed_angles[m, 1] = half_pi
        fixed_angles[m, 2] = two_pi_over_three

    # 5: rhombohedral: a = b = c, alpha = beta = gamma
    m = lattice_codes == LATTICE_TO_CODE["rhombohedral"]
    if m.any():
        abc_mean = lattice_lengths[m].mean(dim=-1)
        angle_mean = lattice_angles[m].mean(dim=-1)

        fixed_lengths[m, :] = abc_mean[:, None]
        fixed_angles[m, :] = angle_mean[:, None]

    # 6: cubic: a = b = c, angles = 90
    m = lattice_codes == LATTICE_TO_CODE["cubic"]
    if m.any():
        abc_mean = lattice_lengths[m].mean(dim=-1)

        fixed_lengths[m, :] = abc_mean[:, None]
        fixed_angles[m, :] = half_pi

    if squeeze:
        fixed_lengths = fixed_lengths[0]
        fixed_angles = fixed_angles[0]

    return fixed_lengths, fixed_angles


def cell_parameters_to_box_vectors(opt: str,
                                   cell_lengths: torch.tensor,
                                   cell_angles: torch.tensor,
                                   return_vol: bool = False):
    """
    Compute one fractional-Cartesian transform matrix with torch.

    Initially borrowed from Nikos. Computes the same matrix elements as
    ``coor_trans_matrix_np``, and as ``batch_compute_fractional_transform``
    for a single cell; for ``'f_to_c'`` the columns are the cell vectors. The
    matrix is float32 on the device of ``cell_lengths``.

    Parameters
    ----------
    opt : str
        ``'c_to_f'`` for Cartesian-to-fractional or ``'f_to_c'`` for
        fractional-to-Cartesian; any other value returns a zero matrix.
    cell_lengths : torch.Tensor
        Cell lengths (a, b, c), [3].
    cell_angles : torch.Tensor
        Cell angles (alpha, beta, gamma) in radians, [3].
    return_vol : bool
        If True, also return the absolute cell volume.

    Returns
    -------
    transform : torch.Tensor
        Transform matrix, [3, 3].
    cell_volume : torch.Tensor
        Absolute cell volume, 0-d; returned only when ``return_vol`` is True.
    """
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(cell_angles)
    sin_a = torch.sin(cell_angles)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = torch.sign(val) * cell_lengths[0] * cell_lengths[1] * cell_lengths[2] * torch.sqrt(
        torch.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    m = torch.zeros((3, 3), dtype=torch.float32, device=cell_lengths.device)
    if opt == 'c_to_f':
        ''' Converting from cartesian to fractional '''
        m[0, 0] = 1.0 / cell_lengths[0]
        m[0, 1] = -cos_a[2] / cell_lengths[0] / sin_a[2]
        m[0, 2] = cell_lengths[1] * cell_lengths[2] * (cos_a[0] * cos_a[2] - cos_a[1]) / vol / sin_a[2]
        m[1, 1] = 1.0 / cell_lengths[1] / sin_a[2]
        m[1, 2] = cell_lengths[0] * cell_lengths[2] * (cos_a[1] * cos_a[2] - cos_a[0]) / vol / sin_a[2]
        m[2, 2] = cell_lengths[0] * cell_lengths[1] * sin_a[2] / vol
    elif opt == 'f_to_c':
        ''' Converting from fractional to cartesian '''
        m[0, 0] = cell_lengths[0]
        m[0, 1] = cell_lengths[1] * cos_a[2]
        m[0, 2] = cell_lengths[2] * cos_a[1]
        m[1, 1] = cell_lengths[1] * sin_a[2]
        m[1, 2] = cell_lengths[2] * (cos_a[0] - cos_a[1] * cos_a[2]) / sin_a[2]
        m[2, 2] = vol / cell_lengths[0] / cell_lengths[1] / sin_a[2]

    # todo create m in a single-step
    if return_vol:
        return m, torch.abs(vol)
    else:
        return m


def compute_mol_radius(coords: torch.FloatTensor) -> Tensor:
    """
    Compute the radius of a single molecule as the maximum distance from the centroid to any atom.

    Parameters
    ----------
    coords : torch.FloatTensor(n, 3)

    Returns
    -------
    radius : torch.FloatTensor scalar
    """
    centroid = coords.mean(0)
    return torch.amax(torch.linalg.norm(coords - centroid, dim=-1))


def batch_compute_mol_radius(coords: torch.FloatTensor,
                             batch: torch.LongTensor,
                             num_graphs: int,
                             nodes_per_graph: torch.LongTensor,
                             ) -> Tensor:
    """
    Compute, per graph, the maximum distance from the centroid of all nodes to any node.

    Centroids are expanded to nodes with ``repeat_interleave(nodes_per_graph)``,
    so nodes must be grouped contiguously by graph in batch order.

    Parameters
    ----------
    coords : torch.FloatTensor
        Node coordinates, [num_nodes, 3].
    batch : torch.LongTensor
        Graph index of each node, [num_nodes].
    num_graphs : int
        Number of graphs.
    nodes_per_graph : torch.LongTensor
        Node count per graph, [num_graphs].

    Returns
    -------
    radii : torch.FloatTensor
        [num_graphs].
    """
    centroids = get_batch_centroids(coords, batch, num_graphs)
    dists = torch.linalg.norm(coords - centroids.repeat_interleave(nodes_per_graph, 0), dim=-1)
    return scatter(dists, batch, dim=0, dim_size=num_graphs, reduce='max')


def get_batch_centroids(coords: torch.FloatTensor,
                        batch: torch.LongTensor,
                        num_graphs: int,
                        ) -> Tensor:
    """
    Compute the centroid (mean position) for each graph in a batch.

    Parameters
    ----------
    coords : torch.FloatTensor(n, 3)
    batch : torch.LongTensor(n)
    num_graphs : int

    Returns
    -------
    centroids : torch.FloatTensor(num_graphs, 3)
    """
    return scatter(coords, batch, dim=0, dim_size=num_graphs, reduce='mean')


def center_batch(coords: torch.FloatTensor,
                 batch: torch.LongTensor,
                 num_graphs: int,
                 nodes_per_graph: torch.LongTensor,
                 center_on_heavy_atoms: bool = False,
                 atom_types: Optional[torch.LongTensor] = None,
                 ) -> Tensor:
    """
    Subtract each graph's centroid from its coordinates, returning a new tensor.

    Centroids are expanded to nodes with ``repeat_interleave(nodes_per_graph)``,
    so nodes must be grouped contiguously by graph in batch order.

    Parameters
    ----------
    coords : torch.FloatTensor
        Node coordinates, [num_nodes, 3].
    batch : torch.LongTensor
        Graph index of each node, [num_nodes].
    num_graphs : int
        Number of graphs.
    nodes_per_graph : torch.LongTensor
        Node count per graph, [num_graphs].
    center_on_heavy_atoms : bool
        If True, compute the centroid from atoms with ``atom_types > 1`` only;
        all atoms are translated.
    atom_types : torch.LongTensor, optional
        Atomic numbers, [num_nodes]; required when ``center_on_heavy_atoms``
        is True.

    Returns
    -------
    coords_out : torch.FloatTensor
        Centred coordinates, [num_nodes, 3].
    """
    if center_on_heavy_atoms:
        mask = atom_types > 1
        centroids = get_batch_centroids(coords[mask], batch[mask], num_graphs)
    else:
        centroids = get_batch_centroids(coords, batch, num_graphs)
    coords_out = coords - centroids.repeat_interleave(nodes_per_graph, 0)
    return coords_out


def batch_compute_mol_mass(z: torch.LongTensor,
                           batch: torch.LongTensor,
                           masses_tensor: torch.FloatTensor,
                           num_graphs: int) -> Tensor:
    """
    Sum atomic masses for each graph in a batch.

    Parameters
    ----------
    z : torch.LongTensor(n)
        Atomic numbers used to index masses_tensor
    batch : torch.LongTensor(n)
    masses_tensor : torch.FloatTensor
        Lookup table of atomic masses indexed by atomic number
    num_graphs : int

    Returns
    -------
    masses : torch.FloatTensor(num_graphs)
    """
    return scatter(masses_tensor[z], batch, dim=0, dim_size=num_graphs, reduce='sum')


def compute_mol_mass(z: torch.LongTensor,
                     masses_tensor: torch.FloatTensor) -> Tensor:
    """
    Sum atomic masses for a single molecule.

    Parameters
    ----------
    z : torch.LongTensor(n)
        Atomic numbers
    masses_tensor : torch.FloatTensor
        Lookup table of atomic masses indexed by atomic number

    Returns
    -------
    mass : torch.FloatTensor scalar
    """
    return torch.sum(masses_tensor[z])


def rotvec2rotmat(mol_rotation: torch.tensor, basis='cartesian'):
    """
    Convert rotation vectors to rotation matrices with Rodrigues' formula.

    ``R = I + sin(r) K + (1 - cos(r)) K @ K``, where ``r`` is the rotation
    angle in radians and ``K`` is the cross-product matrix of the unit axis
    (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).

    Parameters
    ----------
    mol_rotation : torch.Tensor
        Rotation vectors, [n, 3]. For ``basis='cartesian'`` these are
        Cartesian axis-angle vectors (norm = angle); a zero vector gives NaN.
        For ``basis='spherical'`` the columns are (theta, phi, r): polar and
        azimuthal angle of the axis and the rotation angle, as produced by
        ``lat2sph_rotvec`` or ``cart2sph_rotvec``; r = 0 gives NaN.
    basis : str
        ``'cartesian'`` (default) or ``'spherical'``.

    Returns
    -------
    applied_rotation_list : torch.Tensor
        Rotation matrices, [n, 3, 3]. The identity term is float32, so the
        dtype follows type promotion with float32.

    Raises
    ------
    SystemExit
        If ``basis`` is not recognised (a message is printed first).
    """
    if basis == 'cartesian':
        r = torch.linalg.norm(mol_rotation, dim=1)
        unit_vector = mol_rotation / r[:, None]
    elif basis == 'spherical':  # psi, phi, (spherical unit vector) theta (rotation vector)
        r = mol_rotation[
            :, -1]  # third dimension in spherical basis is the norm #torch.linalg.norm(mol_rotation, dim=1)
        mol_rotation = sph2cart_rotvec(mol_rotation)
        unit_vector = mol_rotation / r[:, None]
    else:
        print(f'{basis} is not a valid orientation parameterization!')
        sys.exit()

    K = torch.stack((  # matrix representing rotation axis
        torch.stack((torch.zeros_like(unit_vector[:, 0]), -unit_vector[:, 2], unit_vector[:, 1]), dim=1),
        torch.stack((unit_vector[:, 2], torch.zeros_like(unit_vector[:, 0]), -unit_vector[:, 0]), dim=1),
        torch.stack((-unit_vector[:, 1], unit_vector[:, 0], torch.zeros_like(unit_vector[:, 0])), dim=1)
    ), dim=1)

    identity_batch = torch.eye(3, device=r.device, dtype=torch.float32)[None, :, :].tile(len(r), 1, 1)
    applied_rotation_list = identity_batch + torch.sin(r[:, None, None]) * K + (1 - torch.cos(r[:, None, None])) * (
            K @ K)

    return applied_rotation_list


def extract_rotmat(target_position: torch.FloatTensor,
                   original_position: torch.FloatTensor) -> Tensor:
    """
    Return ``target_position^T @ inv(original_position)`` for single or batched 3x3 matrices.

    The result ``M`` satisfies ``M @ original_position = target_position^T``;
    for a symmetric ``target_position`` (such as the identity) this is
    ``M @ original_position = target_position``.

    Parameters
    ----------
    target_position : torch.FloatTensor
        [n, 3, 3] or [3, 3].
    original_position : torch.FloatTensor
        Invertible matrices, same shape as ``target_position``.

    Returns
    -------
    rotmat : torch.FloatTensor
        [n, 3, 3] or [3, 3].

    Raises
    ------
    AssertionError
        If ``target_position`` is neither 2-D nor 3-D.
    """
    if target_position.ndim == 3:
        return torch.einsum('nji, njk -> nik', target_position, torch.linalg.inv(original_position))
    elif target_position.ndim == 2:
        return torch.einsum('ji, jk -> ik', target_position, torch.linalg.inv(original_position))
    else:
        assert False, "Target position must have at least 2 dimensions"


def apply_rotation_to_batch(elems, rotations, batch):
    """
    Apply per-graph rotation matrices to a batch of vectors.

    Parameters
    ----------
    elems : torch.FloatTensor(n, 3)
    rotations : torch.FloatTensor(num_graphs, 3, 3)
    batch : torch.LongTensor(n)

    Returns
    -------
    rotated : torch.FloatTensor(n, 3)
    """
    return torch.einsum('nij, nj -> ni', rotations[batch], elems)


def rotmat2rotvec_stable(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [n, 3, 3] (proper, orthogonal up to rounding) -> rotation vectors [n, 3], angle in [0, pi],
    in the input dtype. Accurate for EVERY rotation, including those at or near the identity and near a rotation by pi,
    where rotmat2rotvec's float32 trace test (``|(tr - 1)/2| >= 1``) replaces the rotation with pi about (1, 1, 1): a
    different orientation. Here the matrix goes to a unit quaternion by Shepperd's method in float64 (the largest of
    w^2, x^2, y^2, z^2 is taken from the diagonal, so no division by a small number), then
    ``angle = 2 atan2(|v|, w)`` with w >= 0."""
    R = rotation_matrices.double()
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    cand = torch.stack([tr, m00, m11, m22], dim=1)
    k = cand.argmax(dim=1)
    q = torch.empty(len(R), 4, dtype=torch.float64, device=R.device)  # (w, x, y, z)
    for i in range(4):
        sel = k == i
        if not sel.any():
            continue
        r = R[sel]
        if i == 0:
            s = torch.sqrt((1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]).clamp(min=0)) * 2  # s = 4w
            q[sel] = torch.stack([s / 4, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                                  (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif i == 1:
            s = torch.sqrt((1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]).clamp(min=0)) * 2  # s = 4x
            q[sel] = torch.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, s / 4, (r[:, 0, 1] + r[:, 1, 0]) / s,
                                  (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif i == 2:
            s = torch.sqrt((1 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2]).clamp(min=0)) * 2  # s = 4y
            q[sel] = torch.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, s / 4,
                                  (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            s = torch.sqrt((1 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2]).clamp(min=0)) * 2  # s = 4z
            q[sel] = torch.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                                  (r[:, 1, 2] + r[:, 2, 1]) / s, s / 4], 1)
    q = q / q.norm(dim=1, keepdim=True)
    q = torch.where(q[:, :1] < 0, -q, q)  # w >= 0: angle in [0, pi]
    v = q[:, 1:]
    vn = v.norm(dim=1)
    angle = 2 * torch.atan2(vn, q[:, 0])
    scale = torch.where(vn > 1e-12, angle / vn.clamp(min=1e-300), torch.full_like(vn, 2.0))  # small angle: 2 v
    return (v * scale[:, None]).to(rotation_matrices.dtype)


def rotmat2rotvec(rotation_matrix_list, warn_on_bad_determinant=True):
    """
    Convert rotation matrices to rotation vectors (unit axis times angle).

    The axis is the normalised skew-symmetric part
    ``(R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1])`` and the angle is
    ``arccos((trace - 1) / 2)`` in radians, in [0, pi]. Rows where
    ``|(trace - 1) / 2| >= 1``, where the angle or axis is NaN, or where the
    three axis components sum to exactly zero are returned as a rotation by
    pi about (1, 1, 1) / sqrt(3), whatever the input rotation. This includes
    the exact identity and exact rotations by pi; ``rotmat2rotvec_stable``
    handles those cases. The input is not modified.

    Parameters
    ----------
    rotation_matrix_list : torch.FloatTensor
        Rotation matrices, [n, 3, 3].
    warn_on_bad_determinant : bool
        If True, print a warning when any matrix has a negative determinant
        (a reflection); the conversion still proceeds.

    Returns
    -------
    rotvecs : torch.FloatTensor
        Rotation vectors, [n, 3].
    """
    # Fixed!
    if warn_on_bad_determinant:
        det = torch.linalg.det(rotation_matrix_list)
        bad_dets = det < 0.0
        if bad_dets.any():
            num_bad = bad_dets.sum().item()
            print(f"[rotmat2rotvec] WARNING: {num_bad} matrices have determinant < 0. "
                  "These are reflections (not in SO(3)) and cannot be converted to rotation vectors.")

    direction_vector_list = torch.stack([
        rotation_matrix_list[:, 2, 1] - rotation_matrix_list[:, 1, 2],
        rotation_matrix_list[:, 0, 2] - rotation_matrix_list[:, 2, 0],
        rotation_matrix_list[:, 1, 0] - rotation_matrix_list[:, 0, 1]],
    ).T
    trace = torch.einsum('nii->n', rotation_matrix_list)
    r_arg = (trace - 1) / 2
    r = torch.arccos(r_arg)

    bad_inds = torch.any(torch.stack([r_arg.abs() >= 1,
                                      torch.isnan(r),
                                      direction_vector_list.sum(1) == 0,
                                      torch.isnan(direction_vector_list).sum(dim=1) > 0]
                                     ).T,
                         dim=1)

    direction_vector_list[bad_inds, :] = torch.ones_like(direction_vector_list[bad_inds, :])
    r[bad_inds] = torch.pi

    rotvecs = direction_vector_list / (direction_vector_list.norm(dim=1, keepdim=True).clamp(min=1e-8)) * r[:, None]
    return rotvecs


"""
old version


        direction_vector = torch.tensor([
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1]],
            device=rotation_matrix.device, dtype=torch.float32)  # 32 precision is limiting here in some cases

        rotvec_list.append(direction_vector / torch.linalg.norm(direction_vector) * r)
        """


def sample_random_valid_rotvecs(num_samples):
    """
    Sample random rotation vectors with non-negative z component.

    Directions are Gaussian samples normalised to unit length, uniform on the
    sphere, then reflected into the z >= 0 hemisphere. Norms (rotation angles,
    radians) are uniform on [0, 2 pi) and clipped below at 0.05. The tensor is
    created on CPU in the default float dtype.

    Parameters
    ----------
    num_samples : int
        Number of rotation vectors.

    Returns
    -------
    random_vectors : torch.Tensor
        Rotation vectors, [num_samples, 3].
    """
    # random directions on the sphere, getting naturally the correct distribution of theta, phi
    random_vectors = torch.randn(size=(num_samples, 3))
    # set norms uniformly between 0-2pi
    norms = random_vectors.norm(dim=1)
    applied_norms = (torch.rand(num_samples) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
    random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]
    # restrict theta to upper half-sphere (positive z)
    random_vectors[:, 2] = torch.abs(random_vectors[:, 2])
    return random_vectors


def embed_vector_to_rank3(v):
    """
    Embed vectors as symmetric rank-3 tensors.

    ``P[n, i, j, k] = (delta_ij v[n, k] + delta_ik v[n, j] + delta_jk v[n, i]) / 3``,
    with the Kronecker delta ``torch.eye(3)`` created on CPU.

    Parameters
    ----------
    v : torch.Tensor
        [n, 3].

    Returns
    -------
    P : torch.Tensor
        [n, 3, 3, 3].
    """
    delta = torch.eye(3)
    P = torch.einsum('ij,nk->nijk', delta, v) + \
        torch.einsum('ik,nj->nijk', delta, v) + \
        torch.einsum('jk,ni->nijk', delta, v)
    return P / 3  # Normalization factor


def fractional_transform(coords, transform_matrix):
    """
    Apply a fractional-Cartesian transform matrix to coordinates (numpy or torch).

    Each coordinate vector ``x`` becomes ``transform_matrix @ x``. With a
    fractional-to-Cartesian matrix whose columns are the cell vectors (the
    transpose of row-stacked box vectors, as ``T_fc`` from
    ``batch_compute_fractional_transform``) this maps fractional to Cartesian;
    with ``T_cf`` it maps Cartesian to fractional. Dispatches to
    ``fractional_transform_np`` or ``fractional_transform_torch``.

    Parameters
    ----------
    coords : np.ndarray or torch.Tensor
        [n, 3] or [n, m, 3] with a [3, 3] matrix, or [n, 3] with one matrix
        per row, [n, 3, 3].
    transform_matrix : np.ndarray or torch.Tensor
        [3, 3] or [n, 3, 3], of the same array type as ``coords``.

    Returns
    -------
    transformed_coords : np.ndarray or torch.Tensor
        Same shape as ``coords``; None for any other combination of
        dimensions.

    Raises
    ------
    AssertionError
        If ``coords`` is neither a numpy array nor a torch tensor.
    """
    if isinstance(coords, np.ndarray):
        return fractional_transform_np(coords, transform_matrix)
    elif torch.is_tensor(coords):
        return fractional_transform_torch(coords, transform_matrix)
    else:
        assert False


def fractional_transform_np(coords, transform_matrix):
    """
    Apply a fractional/cartesian transform to numpy coordinate arrays.

    Each coordinate vector ``x`` becomes ``transform_matrix @ x``. Dispatches
    on dimensionality: [n, 3] or [n, m, 3] coordinates with one [3, 3]
    matrix, or [n, 3] coordinates with one matrix per row, [n, 3, 3]. Any
    other combination returns None.

    Parameters
    ----------
    coords : np.ndarray
        [n, 3] or [n, m, 3].
    transform_matrix : np.ndarray
        [3, 3] or [n, 3, 3].

    Returns
    -------
    transformed : np.ndarray
        Same shape as ``coords``.
    """
    if coords.ndim == 2 and transform_matrix.ndim == 2:
        return np.einsum('nj,ij->ni', coords, transform_matrix)
    elif coords.ndim == 3 and transform_matrix.ndim == 2:
        return np.einsum('nmj,ij->nmi', coords, transform_matrix)
    elif coords.ndim == 2 and transform_matrix.ndim == 3:
        return np.einsum('nj,nij->ni', coords, transform_matrix)


def fractional_transform_torch(coords, transform_matrix):
    """
    Apply a fractional/cartesian transform to torch coordinate tensors.

    Same dispatch as ``fractional_transform_np``: each coordinate vector ``x``
    becomes ``transform_matrix @ x`` for [n, 3] or [n, m, 3] coordinates with
    one [3, 3] matrix, or [n, 3] coordinates with one matrix per row,
    [n, 3, 3]. Any other combination returns None.

    Parameters
    ----------
    coords : torch.FloatTensor
        [n, 3] or [n, m, 3].
    transform_matrix : torch.FloatTensor
        [3, 3] or [n, 3, 3].

    Returns
    -------
    transformed : torch.FloatTensor
        Same shape as ``coords``.
    """
    if coords.ndim == 2 and transform_matrix.ndim == 2:
        return torch.einsum('nj,ij->ni', (coords, transform_matrix))
    elif coords.ndim == 3 and transform_matrix.ndim == 2:
        return torch.einsum('nmj,ij->nmi', (coords, transform_matrix))
    elif coords.ndim == 2 and transform_matrix.ndim == 3:
        return torch.einsum('nj,nij->ni', (coords, transform_matrix))


def compute_ellipsoid_volume(e):
    """
    Compute ellipsoid volumes from semi-axis vectors.

    Parameters
    ----------
    e : torch.FloatTensor(..., 3, 3)
        Each row is a semi-axis vector; volume = (4/3)π * product of semi-axis lengths

    Returns
    -------
    volumes : torch.FloatTensor(...)
    """
    return 4 / 3 * torch.pi * e.norm(dim=-1).prod(dim=-1)


def compute_cosine_similarity_matrix(e1, e2):
    """
    Compute dot products between the rows of paired batched matrices.

    ``out[n, i, k] = e1[n, i] . e2[n, k]``. The rows are not normalised, so
    these are cosines only for unit rows. Swapping ``e1`` and ``e2``
    transposes the last two dimensions of the result.

    Parameters
    ----------
    e1 : torch.Tensor
        [n, i, j].
    e2 : torch.Tensor
        [n, k, j].

    Returns
    -------
    overlaps : torch.Tensor
        [n, i, k].
    """
    return torch.einsum('nij, nkj -> nik', e1, e2)


def safe_batched_eigh(covs, chunk=10000):
    """
    Chunked symmetric eigendecomposition with a CPU fallback for CUSOLVER failures.

    Calls ``torch.linalg.eigh`` on consecutive chunks. A chunk that raises a
    RuntimeError containing ``CUSOLVER_STATUS_INVALID_VALUE`` is recomputed
    on CPU (after printing a message) and moved back to ``covs.device``; CUDA
    out-of-memory errors and other RuntimeErrors are re-raised. Outputs are
    cast to float32.

    Parameters
    ----------
    covs : torch.Tensor
        Batch of symmetric matrices, [n, d, d].
    chunk : int
        Number of matrices per ``eigh`` call.

    Returns
    -------
    eigenvalues : torch.FloatTensor
        Eigenvalues in ascending order, [n, d].
    eigenvectors : torch.FloatTensor
        Eigenvectors as columns, [n, d, d].
    """
    out_vals, out_vecs = [], []
    for i in range(0, covs.shape[0], chunk):
        cchunk = covs[i:i + chunk]
        try:
            ev, evec = torch.linalg.eigh(cchunk)
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as e:
            if "CUSOLVER_STATUS_INVALID_VALUE" in str(e):
                print("Invalid matrix to eigh! Switching to CPU.")
                ev, evec = torch.linalg.eigh(cchunk.cpu())
                ev, evec = ev.to(covs.device), evec.to(covs.device)
            else:
                raise e
        out_vals.append(ev)
        out_vecs.append(evec)
    return torch.cat(out_vals, dim=0).float(), torch.cat(out_vecs, dim=0).float()


def lat2sph_rotvec(lat_orientations, z_prime):
    """
    Map latent orientation parameters in [-1, 1] to spherical rotation vectors.

    Inverse of ``sph_rotvec2lat``. Each consecutive triplet ``(x0, x1, x2)``
    of the last dimension maps as:

    - theta = (pi/4) x0 + pi/4: [-1, 1] -> [0, pi/2], polar angle of the axis;
    - phi = pi x1: [-1, 1] -> [-pi, pi], azimuthal angle of the axis;
    - r = pi x2 + pi: [-1, 1] -> [0, 2 pi], rotation angle.

    Values outside [-1, 1] follow the same affine maps without clipping.

    Parameters
    ----------
    lat_orientations : torch.Tensor
        Latent orientations, [..., 3 * z_prime].
    z_prime : int
        Number of orientation triplets (Z') in the last dimension.

    Returns
    -------
    sph_rotvec : torch.Tensor
        (theta, phi, r) triplets in radians; a new tensor with the shape of
        ``lat_orientations``.
    """
    lat = lat_orientations.view(*lat_orientations.shape[:-1], z_prime, 3)

    # allocate output
    sph = torch.empty_like(lat)

    halfpi = torch.pi / 2

    # theta
    sph[..., 0] = lat[..., 0] * (torch.pi / 4) + (halfpi / 2)

    # phi
    sph[..., 1] = lat[..., 1] * torch.pi

    # r
    sph[..., 2] = lat[..., 2] * torch.pi + torch.pi

    return sph.view(*lat_orientations.shape)


def simple_latent_distance(l1: torch.Tensor,
                           l2: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance between latent vectors with the periodic orientation dimensions wrapped.

    The latent layout is taken as 6 cell dimensions, 3 * Zp centroid
    dimensions, then 3 * Zp orientation triplets (theta, phi, r; see
    ``lat2sph_rotvec``), with ``Zp = (l1.shape[-1] - 6) // 6``. In every
    orientation triplet the phi and r differences are wrapped into [-1, 1)
    (period 2) before the norm; all other dimensions, including theta, are
    not wrapped. The inputs are not modified.

    Parameters
    ----------
    l1 : torch.Tensor
        Latent vectors, [n, 6 + 6 * Zp].
    l2 : torch.Tensor
        Latent vectors broadcastable against ``l1``; the difference must be
        2-D.

    Returns
    -------
    dist : torch.Tensor
        [n] (the broadcast batch size).
    """
    max_z_prime = (l1.shape[-1] - 6) // 6
    angs = [False] * 6
    for zp in range(max_z_prime):
        angs.extend([False, False, False])
    for zp in range(max_z_prime):
        angs.extend([False, True, True])
        # phi and r dimensions arein rotational basis
    periodic_mask = torch.tensor(angs, device=l1.device)

    diff = l1 - l2

    # wrap periodic dims
    diff[:, periodic_mask] = ((diff[:, periodic_mask] + 1) % 2) - 1

    dist = diff.norm(dim=-1)
    return dist


def compute_latent_distance(latents1: torch.Tensor,
                            latents2: torch.Tensor) -> torch.Tensor:
    """
    Weighted distance between crystals in the latent cell parameterization.

    ``Zp = (latents1.shape[1] - 6) // 6`` is inferred from the width; the
    layout is 6 cell dimensions, 3 * Zp centroid dimensions, then 3 * Zp
    orientation dimensions (see ``lat2sph_rotvec``). The result is

    ``0.5 * box + 0.836 * 0.25 * pos / Zp + 0.293 * 0.25 * rot / Zp``

    where ``box`` is the Euclidean distance over the 6 cell dimensions,
    ``pos`` the Euclidean distance over all centroid dimensions (neither is
    wrapped), and ``rot`` the sum over the Zp orientation triplets of the
    geodesic angle ``arccos((trace(R1 @ R2^T) - 1) / 2)``, in radians, between
    the rotation matrices built by ``rotvec2rotmat(..., 'spherical')``. The
    cosine is clamped to [-1 + 1e-7, 1 - 1e-7], so identical inputs give a
    small positive distance rather than exactly zero.

    Parameters
    ----------
    latents1 : torch.Tensor
        Latent vectors, [n, 6 + 6 * Zp].
    latents2 : torch.Tensor
        Latent vectors with the same last dimension, broadcastable against
        ``latents1`` along the first dimension (e.g. [1, K] against [n, K]).

    Returns
    -------
    dists : torch.Tensor
        [n] (the broadcast batch size).

    Raises
    ------
    AssertionError
        If the last dimensions of ``latents1`` and ``latents2`` differ.
    """
    n_params = latents1.shape[1]
    assert latents1.shape[-1] == latents2.shape[-1]
    z_prime = (n_params - 6) // 6

    "Latent box parameters are the log lengths 0-2 and angles 3-5"
    box_params1 = latents1[..., :6]
    box_params2 = latents2[..., :6]
    box_dist = (box_params1 - box_params2).norm(dim=-1)

    "positions defined on [-1, 1]^3 * z_prime. Not periodic (different periodicity in every space group)"
    positions1 = latents1[..., 6:6 + 3 * z_prime]
    positions2 = latents2[..., 6:6 + 3 * z_prime]
    positions_dist = (positions1 - positions2).norm(dim=-1)

    "angles are defined in a spherical basis [0,pi/2], [-pi, pi], [0,2pi] and renormalized on [-1,1]"
    lat_orientations1 = latents1[..., 6 + 3 * z_prime:]
    lat_orientations2 = latents2[..., 6 + 3 * z_prime:]

    lat_sph_rotvec1 = lat2sph_rotvec(lat_orientations1, z_prime)  # [polar, azimuthal, length]
    lat_sph_rotvec2 = lat2sph_rotvec(lat_orientations2, z_prime)

    rot_dists = []
    for zp in range(z_prime):  # this should be replaced with a proper vector distance
        rmat1 = rotvec2rotmat(lat_sph_rotvec1[..., 3 * zp:3 * zp + 3], 'spherical')
        rmat2 = rotvec2rotmat(lat_sph_rotvec2[..., 3 * zp:3 * zp + 3], 'spherical')

        R_delta = rmat1 @ rmat2.transpose(-1, -2)

        trace = R_delta.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
        dist = torch.acos(cos_theta)
        rot_dists.append(dist)

    rot_dists = torch.stack(rot_dists).sum(0)

    "overall distance metric"
    # dists = 0.5 * box_dist + 0.25 * (positions_dist / z_prime / 2.5) + 0.25 * (rot_dists / z_prime / 2)
    # scales = [2 * sqrt(6), 2*sqrt(3), torch.pi] # maximum variation per dist
    scales = [1, 0.836, 0.293]  # [0.0127, 0.0152, 0.0433]  # empirical std over CSD samples
    dists = scales[0] * 0.5 * box_dist + scales[1] * 0.25 * (positions_dist / z_prime) + scales[2] * 0.25 * (
            rot_dists / z_prime)

    return dists


# def crystal_parameter_distmat(latents, max_batch_size = 2000):
#     """
#     Compute distance to self for a set of crystal latent vectors
#     :param latents:
#     :return:
#     """
#     N, K = latents.shape
#     if N < max_batch_size:
#         lat1 = latents[:, None, :].expand(N, N, K).reshape(N * N, K)
#         lat2 = latents[None, :, :].expand(N, N, K).reshape(N * N, K)
#
#         d = crystal_parameter_distance(lat1, lat2)  # (N*N,)
#         distmat = d.reshape(N, N)
#     else:
#         distmat = torch.zeros((N, N), device=latents.device)
#
#         for i in range(N):
#             lat1 = latents[i:i + 1].expand(N, -1)  # (N, K)
#             lat2 = latents  # (N, K)
#
#             distmat[i] = crystal_parameter_distance(lat1, lat2)
#
#
#     return distmat

def crystal_parameter_distmat(
        latents,
        target_entries=5_000_000,  # ≈ distances per call
        min_block=1,
        max_block=2048,
):
    """
    Pairwise ``compute_latent_distance`` matrix of a set of latent vectors, computed in row blocks.

    Rows are processed in blocks of
    ``B = max(min_block, min(max_block, target_entries // max(N, 1)))``, so each call
    to ``compute_latent_distance`` handles about ``B * N`` pairs. The diagonal
    holds the small positive self-distance of ``compute_latent_distance``,
    not zero.

    Parameters
    ----------
    latents : torch.Tensor
        Latent vectors, [N, K].
    target_entries : int
        Approximate number of distances per block.
    min_block : int
        Minimum rows per block.
    max_block : int
        Maximum rows per block.

    Returns
    -------
    distmat : torch.Tensor
        [N, N], allocated on ``latents.device`` in the default float dtype.
    """
    device = latents.device
    N, K = latents.shape

    # adaptive block size
    B = max(min_block, min(max_block, target_entries // max(N, 1)))

    distmat = torch.empty((N, N), device=device)

    for i in range(0, N, B):
        lat1 = latents[i:i + B]  # (B, K)
        b = lat1.shape[0]

        lat1_exp = lat1[:, None, :].expand(b, N, K).reshape(-1, K)
        lat2_exp = latents[None, :, :].expand(b, N, K).reshape(-1, K)

        d = compute_latent_distance(lat1_exp, lat2_exp)
        distmat[i:i + b] = d.view(b, N)

    return distmat


def sph_rotvec2lat(sph_rotvec, z_prime):
    """
    Map spherical rotation vectors to latent orientation parameters normalized to [-1, 1].

    Inverse of ``lat2sph_rotvec``. Each consecutive (theta, phi, r) triplet of
    the last dimension maps without clipping to
    ``((theta - pi/4) / (pi/4), phi / pi, (r - pi) / pi)``, so [0, pi/2],
    [-pi, pi] and [0, 2 pi] map to [-1, 1].

    Parameters
    ----------
    sph_rotvec : torch.Tensor
        (theta, phi, r) triplets in radians, [..., 3 * z_prime].
    z_prime : int
        Number of orientation triplets (Z') in the last dimension.

    Returns
    -------
    lat_orientations : torch.Tensor
        A new tensor with the shape of ``sph_rotvec``.
    """
    sph = sph_rotvec.view(*sph_rotvec.shape[:-1], z_prime, 3)

    # allocate output
    lat = torch.empty_like(sph)

    halfpi = torch.pi / 2

    # theta: [0, π/2] → [-1, 1]
    lat[..., 0] = (sph[..., 0] - (halfpi / 2)) / (torch.pi / 4)

    # phi: [-π, π] → [-1, 1]
    lat[..., 1] = sph[..., 1] / torch.pi

    # r: [0, 2π] → [-1, 1]
    lat[..., 2] = (sph[..., 2] - torch.pi) / torch.pi

    return lat.view(*sph_rotvec.shape)

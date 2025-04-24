from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter

from mxtaltools.models.functions.radial_graph import radius, build_radial_graph


def vdw_analysis(vdw_radii: torch.Tensor,
                 dist_dict: dict,
                 num_graphs: int,
                 ):
    """
    new version of the vdw_overlap function for analysis of intermolecular contacts
    """
    batch = dist_dict['intermolecular_dist_batch']
    lj_pot, normed_overlap, overlap = compute_lj_pot(dist_dict, vdw_radii)

    molwise_overlap = scatter(overlap, batch, reduce='sum', dim_size=num_graphs)
    molwise_normed_overlap = scatter(normed_overlap, batch, reduce='sum', dim_size=num_graphs)
    molwise_lj_pot = scatter(lj_pot, batch, reduce='sum', dim_size=num_graphs)

    molwise_loss = scale_molwise_lj_pot(molwise_lj_pot).clip(max=50)

    return molwise_overlap, molwise_normed_overlap, molwise_lj_pot, molwise_loss, lj_pot


def scale_molwise_lj_pot(vdw_potential: torch.Tensor,
                         ):

    rescaled_vdw_loss = vdw_potential.clone()
    rescaled_vdw_loss[rescaled_vdw_loss > 0] = torch.log(rescaled_vdw_loss[rescaled_vdw_loss > 0] + 1)

    return rescaled_vdw_loss

'''
import torch
import plotly.graph_objects as go
xx = torch.linspace(0.001, 5, 1001)
lj = 4 * (1/xx**12 - 1/xx**6)
scaled_lj = torch.log(lj+2) / torch.log(torch.Tensor([2])) - 1
fig = go.Figure()
#fig.add_scatter(x=xx,y=lj)
fig.add_scatter(x=xx,y=scaled_lj)
fig.update_layout(yaxis_range=[-5,10])
fig.show()
'''


def compute_lj_pot(dist_dict, vdw_radii):
    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']

    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]
    # only punish positives (meaning overlaps)
    overlap = F.relu(radii_sums - dists)
    # norm overlaps against internuclear distances
    normed_overlap = F.softplus((radii_sums - dists) / radii_sums, beta=10)
    # uniform lennard jones potential
    sigma_r6 = torch.pow(radii_sums / dists, 6)
    sigma_r12 = torch.pow(sigma_r6, 2)
    lj_pot = torch.nan_to_num(
        4 * 1 * (sigma_r12 - sigma_r6),
        nan=0.0, posinf=1e20, neginf=-1e-20
    )
    return lj_pot, normed_overlap, overlap



def old_compute_num_h_bonds(supercell_data, atom_acceptor_ind, atom_donor_ind, i):
    """
    compute the number of hydrogen bonds, up to a loose range (3.3 angstroms), and non-directionally
    @param atom_donor_ind: index in tracking_features to find donor status
    @param atom_acceptor_ind: index in tracking_features to find acceptor status
    @param supercell_data: crystal data
    @param i: cell index we are checking
    @return: sum of total hydrogen bonds for the canonical conformer
    """
    batch_inds = torch.arange(supercell_data.ptr[i], supercell_data.ptr[i + 1], device=supercell_data.x.device)

    # find the canonical conformers
    canonical_conformers_inds = torch.where(supercell_data.aux_ind[batch_inds] == 0)[0]
    outside_inds = torch.where(supercell_data.aux_ind[batch_inds] == 1)[0]

    # identify and count canonical conformer acceptors and intermolecular donors
    canonical_conformer_acceptors_inds = \
        torch.where(supercell_data.x[batch_inds[canonical_conformers_inds], atom_acceptor_ind] == 1)[0]
    outside_donors_inds = torch.where(supercell_data.x[batch_inds[outside_inds], atom_donor_ind] == 1)[0]

    donors_pos = supercell_data.pos[batch_inds[outside_inds[outside_donors_inds]]]
    acceptors_pos = supercell_data.pos[batch_inds[canonical_conformers_inds[canonical_conformer_acceptors_inds]]]

    return torch.sum(torch.cdist(donors_pos, acceptors_pos, p=2) < 3.3)


def electrostatic_analysis(dist_dict, num_graphs: int, cutoff: float = 0.92):
    """
    technically this is a yukawa potential
    https://en.wikipedia.org/wiki/Yukawa_potential
    """
    batch = dist_dict['intermolecular_dist_batch']
    dists = dist_dict['intermolecular_dist']
    charges = dist_dict['intermolecular_partial_charges']
    # default cutoff of 0.77 gets us 99% exponentially squashed by 6 angstroms distance
    estat_energy = ((charges[0] * charges[1]) / dists) * torch.exp(-(dists*cutoff))
    molwise_estat_energy = scatter(estat_energy, batch, reduce='sum', dim_size=num_graphs)

    return molwise_estat_energy

def buckingham_energy(dist_dict,
                      num_graphs,
                      vdw_radii,
                      A: float = -3.2619,
                      B: float = 1.0,
                      C: float = -0.2,
                      ):
    """
    buckingham potential with distance normalized by interatomic distances
    default ABC parameters fitted to very roughly match 12-6 LJ potential
    """
    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']
    batch = dist_dict['intermolecular_dist_batch']

    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]
    normed_dists = dists / radii_sums
    bh = torch.nan_to_num(A * torch.exp(-B * normed_dists) - C / normed_dists ** 6,
                          nan=0.0, posinf=1e20, neginf=-1e-20
                          )
    molwise_bh = scatter(bh, batch, reduce='sum', dim_size=num_graphs)

    return molwise_bh

    #
    # def fit_buckingham_to_lj(r0=1.0, epsilon=1.0, B=1.2):
    #     import numpy as np
    #     from scipy.optimize import fsolve
    #     def equations(vars):
    #         A, C = vars
    #         # Energy match at r0
    #         V = A * np.exp(-B * r0) - C / r0**6 + epsilon
    #         # Derivative match at r0 (minimum)
    #         dV = -A * B * np.exp(-B * r0) + 6 * C / r0**7
    #         return [V, dV]
    #     # Initial guess
    #     guess = [100.0, 1000.0]
    #     A, C = fsolve(equations, guess)
    #     return A, B, C
    #     """
    #     A, B, C = fit_buckingham_to_lj(r0=1.0, epsilon=1.0, B=1.2)
    #     xx = torch.linspace(0, 10, 1001)
    #     sigma = 3.0
    #     epsilon = 1.0
    #     lj = 4 * ((sigma/xx)**12 - (sigma/xx)**6)
    #     B = 1
    #     A, B, C = fit_buckingham_to_lj(sigma, epsilon, B)
    #     """



def old_new_hydrogen_bond_analysis(supercell_data, dist_dict, cutoff: float = 2.2):
    """
    dist_dict.keys()
    dict_keys(['edge_index', 'edge_index_inter',
    'inside_inds', 'outside_inds', 'inside_batch',
    'n_repeats', 'num_graphs','graph_size', 'outside_batch',
    'intramolecular_dist', 'intramolecular_dist_atoms',
    'intermolecular_dist', 'intermolecular_dist_batch',
    'intermolecular_dist_atoms'])
    """
    # identify donors as H in O-H or N-H
    proton_intra_edges_bool = dist_dict['intramolecular_dist_atoms'][0] == 1
    nitrogen_intra_edges_bool = dist_dict['intramolecular_dist_atoms'][1] == 7
    oxygen_intra_edges_bool = dist_dict['intramolecular_dist_atoms'][1] == 8

    intra_NH_bool = proton_intra_edges_bool * nitrogen_intra_edges_bool
    intra_OH_bool = proton_intra_edges_bool * oxygen_intra_edges_bool

    proton_is_NH_donor = dist_dict['edge_index'][0][(dist_dict['intramolecular_dist'] <= 1.5) * intra_NH_bool]
    proton_is_OH_donor = dist_dict['edge_index'][0][(dist_dict['intramolecular_dist'] <= 1.5) * intra_OH_bool]

    # identify acceptors in intermolecular edges
    N_acceptors = supercell_data.x.flatten() == 7
    O_acceptors = supercell_data.x.flatten() == 8

    # get hydrogen bonds
    NH_donors_pos = supercell_data.pos[proton_is_NH_donor]
    NH_donors_batch = supercell_data.batch[proton_is_NH_donor]
    OH_donors_pos = supercell_data.pos[proton_is_OH_donor]
    OH_donors_batch = supercell_data.batch[proton_is_OH_donor]

    outside_inds = supercell_data.aux_ind == 1
    N_acceptors_pos = supercell_data.pos[N_acceptors * outside_inds]
    N_acceptors_batch = supercell_data.batch[N_acceptors * outside_inds]
    O_acceptors_pos = supercell_data.pos[O_acceptors * outside_inds]
    O_acceptors_batch = supercell_data.batch[O_acceptors * outside_inds]

    NH_N_edge_index = radius(
        NH_donors_pos,
        N_acceptors_pos,
        cutoff,
        NH_donors_batch,
        N_acceptors_batch,
        max_num_neighbors=100,
    )
    NH_O_edge_index = radius(
        NH_donors_pos,
        O_acceptors_pos,
        cutoff,
        NH_donors_batch,
        O_acceptors_batch,
        max_num_neighbors=100,
    )
    OH_N_edge_index = radius(
        OH_donors_pos,
        N_acceptors_pos,
        cutoff,
        OH_donors_batch,
        N_acceptors_batch,
        max_num_neighbors=100,
    )
    OH_O_edge_index = radius(
        OH_donors_pos,
        O_acceptors_pos,
        cutoff,
        OH_donors_batch,
        O_acceptors_batch,
        max_num_neighbors=100,
    )

    # collect all bonds
    num_NH_N_bonds = scatter(torch.ones(NH_N_edge_index.shape[1]),
                             NH_donors_batch[NH_N_edge_index[1, :]],
                             dim_size=supercell_data.num_graphs)
    num_NH_O_bonds = scatter(torch.ones(NH_O_edge_index.shape[1]),
                             NH_donors_batch[NH_O_edge_index[1, :]],
                             dim_size=supercell_data.num_graphs)
    num_OH_N_bonds = scatter(torch.ones(OH_N_edge_index.shape[1]),
                             OH_donors_batch[OH_N_edge_index[1, :]],
                             dim_size=supercell_data.num_graphs)
    num_OH_O_bonds = scatter(torch.ones(OH_O_edge_index.shape[1]),
                             OH_donors_batch[OH_O_edge_index[1, :]],
                             dim_size=supercell_data.num_graphs)

    return num_NH_N_bonds, num_NH_O_bonds, num_OH_N_bonds, num_OH_O_bonds


def get_intermolecular_dists_dict(cluster_batch,
                                  conv_cutoff: float,
                                  max_num_neighbors: int = 10000):
    dist_dict = {}
    edges_dict = build_radial_graph(
        cluster_batch.pos,
        cluster_batch.batch,
        cluster_batch.ptr,
        conv_cutoff,
        max_num_neighbors,
        aux_ind=cluster_batch.aux_ind,
        mol_ind=cluster_batch.mol_ind,
    )
    dist_dict.update(edges_dict)
    dist_dict['num_graphs'] = cluster_batch.num_graphs
    dist_dict['graph_size'] = cluster_batch.num_atoms
    dist_dict['outside_batch'] = cluster_batch.batch
    dist_dict['intramolecular_dist'] = (
        (cluster_batch.pos[edges_dict['edge_index'][0]] - cluster_batch.pos[
            edges_dict['edge_index'][1]]).pow(2).sum(
            dim=-1).sqrt())

    dist_dict['intramolecular_dist_atoms'] = \
        [cluster_batch.z[edges_dict['edge_index'][0]].long(),
         cluster_batch.z[edges_dict['edge_index'][1]].long()]

    dist_dict['intermolecular_dist'] = (
        (cluster_batch.pos[edges_dict['edge_index_inter'][0]] - cluster_batch.pos[
            edges_dict['edge_index_inter'][1]]).pow(2).sum(
            dim=-1).sqrt())

    dist_dict['intermolecular_dist_batch'] = cluster_batch.batch[edges_dict['edge_index_inter'][0]]

    dist_dict['intermolecular_dist_atoms'] = \
        [cluster_batch.z[edges_dict['edge_index_inter'][0]].long(),
         cluster_batch.z[edges_dict['edge_index_inter'][1]].long()]

    # if we have partial charges in the batch (always first column in x)
    if cluster_batch.x is not None and len(cluster_batch.x) == len(cluster_batch.z):
        if cluster_batch.x.ndim == 2:
            dist_dict['intermolecular_partial_charges'] = \
                [cluster_batch.x[edges_dict['edge_index_inter'][0], 0],
                 cluster_batch.x[edges_dict['edge_index_inter'][1], 0]]

        else:
            dist_dict['intermolecular_partial_charges'] = \
                [cluster_batch.x[edges_dict['edge_index_inter'][0]],
                 cluster_batch.x[edges_dict['edge_index_inter'][1]]]

    return dist_dict


def silu_energy(dist_dict,
                      num_graphs,
                      vdw_radii,):
    """
    a shorter range and softer LJ-type energy
    """
    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']
    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]
    edgewise_potentials = (F.silu(-4 * (dists - radii_sums)) / 0.28) + torch.exp(-dists * 100) * 100
    molwise_silu_energy = scatter(edgewise_potentials, dist_dict['intermolecular_dist_batch'],
                       reduce='sum', dim_size=num_graphs)
    return molwise_silu_energy
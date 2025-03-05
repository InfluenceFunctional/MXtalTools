import sys
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter

from mxtaltools.dataset_utils.CrystalData import CrystalData
from mxtaltools.models.functions.asymmetric_radius_graph import asymmetric_radius_graph, radius


def vdw_overlap(vdw_radii: torch.Tensor,  # TODO replace/unify with vdw_analysis + electrostatics function
                dist_dict: Optional[dict] = None,
                dists: Optional[torch.Tensor] = None,
                batch_numbers: Optional[torch.LongTensor] = None,
                atomic_numbers: Optional[torch.LongTensor] = None,
                num_graphs: Optional[int] = None,
                crystaldata: Optional[CrystalData] = None,
                graph_sizes: Optional[int] = None,
                return_loss_only: bool = False,
                return_score_only: bool = False,
                loss_func: Optional[str] = None):
    assert not (return_loss_only and return_score_only)

    if dist_dict is not None:
        dists = dist_dict['intermolecular_dist']
        atomic_numbers = dist_dict['intermolecular_dist_atoms']
        batch_numbers = dist_dict['intermolecular_dist_batch']

    abs_overlaps, normed_overlaps, lj_potentials = raw_vdw_overlap(
        vdw_radii, crystaldata=crystaldata,
        dists=dists, batch_numbers=batch_numbers, atomic_numbers=atomic_numbers,
        num_graphs=num_graphs
    )

    if graph_sizes is not None:
        mol_sizes = graph_sizes
    elif crystaldata is not None:
        mol_sizes = crystaldata.num_atoms
    else:
        assert False

    if num_graphs is None:
        num_graphs = crystaldata.num_graphs

    if loss_func is None:  # treat all errors equally
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(normed_overlaps[ii]) for ii in range(num_graphs)]
            )) / mol_sizes
    elif loss_func == 'mse':  # penalize on big errors
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(normed_overlaps[ii] ** 2) for ii in range(num_graphs)]
            )) / mol_sizes
    elif loss_func == 'log':  # go easy on big errors
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(torch.log(1 + normed_overlaps[ii])) for ii in range(num_graphs)]
            )) / mol_sizes
    elif loss_func == 'max':  # cut the tallest dandelion
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.amax(normed_overlaps[ii]) for ii in range(num_graphs)]
            ))
    elif loss_func == 'inv':  # hard asymptote at 1, but we can't touch it
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(
                    1 / (-torch.minimum(0.99 * torch.ones_like(normed_overlaps[ii]), normed_overlaps[ii]) + 1) - 1
                ) for ii in range(num_graphs)]
            )) / mol_sizes
    elif loss_func == 'lj':  # mean LJ energy per molecule, with exponential rescaling to convert to an unnormalized probability
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(1 - torch.exp(-(1 + lj_potentials[ii]))) for ii in range(num_graphs)]
            )) / mol_sizes
    else:
        print(f'{loss_func} is not a valid loss function for vdw penalty')
        sys.exit()

    vdw_score = -torch.nan_to_num(
        torch.stack(
            [torch.sum(normed_overlaps[ii]) for ii in range(num_graphs)]
        )) / mol_sizes

    max_overlaps = torch.nan_to_num(
        torch.stack(
            [torch.amax(normed_overlaps[ii]) if len(normed_overlaps[ii] > 0) else
             torch.amax(torch.zeros(1, device=normed_overlaps[ii].device)) for ii in range(num_graphs)]
        ))

    if return_loss_only:
        return vdw_loss
    elif return_score_only:
        return vdw_score
    else:  # return everything
        return vdw_loss, vdw_score, max_overlaps, abs_overlaps, normed_overlaps, lj_potentials


def raw_vdw_overlap(vdw_radii, dists=None, batch_numbers=None,
                    atomic_numbers=None, num_graphs=None,
                    crystaldata=None):
    if crystaldata is not None:  # extract distances from the crystal
        if crystaldata.aux_ind is not None:
            in_inds = torch.where(crystaldata.aux_ind == 0)[0]
            # default to always intermolecular_distances
            out_inds = torch.where(crystaldata.aux_ind == 1)[0].to(crystaldata.pos.device)

        else:  # if we lack the info, just do it intramolecular
            in_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
            out_inds = in_inds

        '''
        compute all distances
        '''
        edges = asymmetric_radius_graph(crystaldata.pos,
                                        batch=crystaldata.batch,
                                        inside_inds=in_inds,
                                        convolve_inds=out_inds,
                                        r=6, max_num_neighbors=500, flow='source_to_target')

        crystal_number = crystaldata.batch[edges[0]]

        dists = (crystaldata.pos[edges[0]] - crystaldata.pos[edges[1]]).pow(2).sum(dim=-1).sqrt()
        elements = [crystaldata.x[edges[0], 0].long().to(dists.device),
                    crystaldata.x[edges[1], 0].long().to(dists.device)]
        num_graphs = crystaldata.num_graphs
        #molecule_sizes = torch.diff(crystaldata.ptr)
    elif dists is not None:  # precomputed intermolecular crystal distances
        crystal_number = batch_numbers
        elements = atomic_numbers
        num_graphs = num_graphs

    else:
        assert False  # must do one or the other

    '''
    compute vdw radii respectfulness
    '''
    if torch.is_tensor(vdw_radii):
        vdw_radii_vector = vdw_radii
    else:
        vdw_radii_vector = torch.Tensor(list(vdw_radii.values())).to(dists.device)
    atom_radii = [vdw_radii_vector[elements[0]], vdw_radii_vector[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]

    penalties = F.relu(radii_sums - dists)  # only punish positives (meaning overlaps)
    normed_penalties = F.relu((radii_sums - dists) / radii_sums)  # norm overlaps against internuclear distances

    """effective lennard jones potential"""  # todo functionalize and provide as separate option
    sigma_r6 = torch.pow(radii_sums / dists, 6)
    sigma_r12 = torch.pow(sigma_r6, 2)
    pot = 4 * 1 * (sigma_r12 - sigma_r6)

    return ([penalties[crystal_number == ii] for ii in range(num_graphs)],
            [normed_penalties[crystal_number == ii] for ii in range(num_graphs)],
            [pot[crystal_number == ii] for ii in range(num_graphs)])


def vdw_analysis(vdw_radii: torch.Tensor,
                 dist_dict: dict,
                 num_graphs: int,
                 num_atoms: torch.LongTensor,
                 ):
    """
    new version of the vdw_overlap function for analysis of intermolecular contacts
    """
    batch = dist_dict['intermolecular_dist_batch']
    lj_pot, normed_overlap, overlap = compute_lj_pot(dist_dict, vdw_radii)

    molwise_overlap = scatter(overlap, batch, reduce='sum', dim_size=num_graphs)
    molwise_normed_overlap = scatter(normed_overlap, batch, reduce='sum', dim_size=num_graphs)
    molwise_lj_pot = scatter(lj_pot, batch, reduce='sum', dim_size=num_graphs)

    molwise_loss = scale_molwise_lj_pot(molwise_lj_pot, num_atoms)

    return molwise_overlap, molwise_normed_overlap, molwise_lj_pot, molwise_loss, lj_pot


def scale_molwise_lj_pot(vdw_potential: torch.Tensor,
                         num_atoms: torch.LongTensor,
                         clip_max: float=50):

    rescaled_vdw_loss = vdw_potential.clone()
    rescaled_vdw_loss[rescaled_vdw_loss > 0] = torch.log(rescaled_vdw_loss[rescaled_vdw_loss > 0])

    if vdw_potential.ndim > 1:
        rescaled_vdw_loss = rescaled_vdw_loss / num_atoms[None, :]
    else:
        rescaled_vdw_loss = rescaled_vdw_loss / num_atoms

    return rescaled_vdw_loss.clip(max=clip_max)

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
    normed_overlap = F.relu((radii_sums - dists) / radii_sums)
    # uniform lennard jones potential
    sigma_r6 = torch.pow(radii_sums / dists, 6)
    sigma_r12 = torch.pow(sigma_r6, 2)
    lj_pot = torch.nan_to_num(
        4 * 1 * (sigma_r12 - sigma_r6),
        nan=0.0, posinf=1e20, neginf=-1e-20
    )
    return lj_pot, normed_overlap, overlap


def scale_edgewise_vdw_pot(lj_pot: Union[np.ndarray, torch.tensor],
                           clip_max: float = 100) \
        -> Union[np.ndarray, torch.tensor]:
    """

    """
    if torch.is_tensor(lj_pot):
        scaled_lj_pot = torch.log(2 + lj_pot) / np.log(2) - 1
        #scaled_lj_pot = lj_pot.clone()
        #scaled_lj_pot[high_bools] = turnover_pot + torch.log10(scaled_lj_pot[high_bools] + 1 - turnover_pot)
    else:
        scaled_lj_pot = torch.log(2 + lj_pot) / torch.log(torch.Tensor([2])) - 1
        #scaled_lj_pot = lj_pot.copy()
        #scaled_lj_pot[high_bools] = turnover_pot + np.log10(scaled_lj_pot[high_bools] + 1 - turnover_pot)
    return scaled_lj_pot.clip(max=clip_max)


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

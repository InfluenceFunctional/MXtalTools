import itertools
from typing import Tuple, Optional, Union

import numpy as np
import torch
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.utils import repeat_interleave, torch_ptp
from mxtaltools.models.functions.radial_graph import asymmetric_radius_graph


def old_crystal_rdf(crystal_batch,
                    precomputed_distances_dict=None,
                    rrange: list[int, int] = [0, 10],
                    bins: int = 100,
                    mode: str = 'all',
                    elementwise: bool = False,
                    raw_density: bool = False,
                    atomwise: bool = False,
                    cpu_detach: bool = False,
                    remove_radial_scaling: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    compute the RDF for all the supercells in a Molcrystal_batch object

    crystal_batch object containing a batch of molecular crystals relevant information
    rrange: range over which to compute radial distribution function
    bins: number of bins for RDF histogramming
    mode: 'intramolecular' consider only intramolecular edges, 'intermolecular' consider only intermolecular edges, 'all' consider both
    elementwise: compute a separate RDF for all pairs of all elements in each crystal (exclusive with atomwise)
    raw_dentiy: False estimate the crystal density. True use raw RDF outputs (same as setting density always to 1)
    atomwise: compute a separate RDF for all pairs of unique atom indexes per molecule in each crystal (exclusive with elementwise)
    cpu_detach: True return outputs as numpy array on cpu. False return outputs as torch tensor on the same device as the input crystal_batch
    """
    device = crystal_batch.pos.device
    #  whether to include intramolecular edges
    if crystal_batch.aux_ind is not None:
        in_inds = torch.where(crystal_batch.aux_ind == 0)[0]
        if mode == 'intermolecular':
            out_inds = torch.where(crystal_batch.aux_ind == 1)[0].to(device)
        elif mode == 'all':
            out_inds = torch.arange(len(crystal_batch.pos)).to(device)
        elif mode == 'intramolecular':
            out_inds = in_inds
        else:
            print(mode + ' is not a valid rdf mode!')
            assert False
    else:
        # if we have no inside/outside labels, just consider the whole thing one big molecule
        in_inds = torch.arange(len(crystal_batch.pos)).to(device)
        out_inds = in_inds

    if precomputed_distances_dict is not None and mode == 'intermolecular':
        edges = precomputed_distances_dict['edge_index_inter']
    else:
        edges = asymmetric_radius_graph(crystal_batch.pos,
                                        batch=crystal_batch.batch,
                                        inside_inds=in_inds,
                                        convolve_inds=out_inds,
                                        r=max(rrange), max_num_neighbors=500, flow='source_to_target')

    # track which edges go with which crystals
    edge_in_crystal_number = crystal_batch.batch[edges[0]]
    for i in range(crystal_batch.num_graphs):
        if i == 0:
            edges_ptr = torch.zeros(crystal_batch.num_graphs + 1, device=device, dtype=torch.long)
            edges_ptr[1] = torch.sum(edge_in_crystal_number == i)
        else:
            edges_ptr[i + 1] = edges_ptr[i] + torch.sum(edge_in_crystal_number == i)

    # compute all the dists
    dists = torch.linalg.norm(crystal_batch.pos[edges[0]] - crystal_batch.pos[edges[1]], dim=1)

    assert not (elementwise and atomwise)

    if elementwise:  # todo this could also be sped up
        relevant_elements = [5, 6, 7, 8, 9, 15, 16, 17, 35]
        element_symbols = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
        elements = [crystal_batch.z[edges[0]], crystal_batch.z[edges[1]]]
        rdfs_dict = {}
        rdfs_array = torch.zeros(
            (crystal_batch.num_graphs, int((len(relevant_elements) ** 2 + len(relevant_elements)) / 2), bins),
            device=dists.device)
        # prebuild pair lists
        elem_pair_list = []
        ind = 0
        for i, element1 in enumerate(relevant_elements):
            for j, element2 in enumerate(relevant_elements):
                if j >= i:
                    rdfs_dict[ind] = element_symbols[element1] + ' to ' + element_symbols[element2]
                    elem_pair_list.append((elements[0] == element1) * (elements[1] == element2))
                    ind += 1

        for i in range(len(elem_pair_list)):
            rdfs_array[:, i], rr = parallel_compute_rdf_torch(
                [dists[(edge_in_crystal_number == n) * elem_pair_list[i]] for n in range(crystal_batch.num_graphs)],
                rrange=rrange, bins=bins,
                raw_density=raw_density,
                remove_radial_scaling=remove_radial_scaling)

        if cpu_detach:
            rdfs_array = rdfs_array.cpu().detach().numpy()
            rr = rr.cpu().detach().numpy()

        return rdfs_array, rr, rdfs_dict

    elif atomwise:  # generate atomwise indices which are shared between samples - quite slow! # todo speed up
        rdfs_array_list = []
        rdfs_dict_list = []
        all_atom_inds = []
        # abs_atom_inds = []
        for i in range(crystal_batch.num_graphs):
            # assume only that the order of atoms is patterned in all images
            canonical_conformer_coords = crystal_batch.pos[
                                         crystal_batch.ptr[i]:crystal_batch.ptr[i] + int(crystal_batch.num_atoms[i])]
            centroid = canonical_conformer_coords.mean(0)
            mol_dists = torch.linalg.norm(canonical_conformer_coords - centroid[None, :], dim=-1)
            inds = torch.argsort(mol_dists)
            all_atom_inds.append(inds.tile(int((crystal_batch.batch == i).sum() // crystal_batch.num_atoms[i])))

        atom_inds = torch.cat(all_atom_inds)

        atoms_in_edges = [atom_inds[edges[0]], atom_inds[edges[1]]]

        ind = 0
        for n in range(crystal_batch.num_graphs):
            # all possible combinations of unique atoms on this graph
            atom_pairs = torch.Tensor(list(itertools.combinations(torch.arange(int(crystal_batch.num_atoms[n])), 2)))
            rdfs_dict_list.append(atom_pairs)  # record the pairs for reporting purposes

            in_crystal_inds = torch.where(edge_in_crystal_number == n)[0]  # atom indices in this crystal

            atom_locations = [[atoms_in_edges[0][in_crystal_inds] == m, atoms_in_edges[1][in_crystal_inds] == m] for m
                              in range(int(atom_pairs.max()) + 1)]

            relevant_atoms_dists_list = [
                dists[in_crystal_inds[torch.logical_and(atom_locations[int(atom_pairs[m, 0])][0],
                                                        atom_locations[int(atom_pairs[m, 1])][1])]]
                for m in range(len(atom_pairs))]

            rdfs_array, rr = parallel_compute_rdf_torch(relevant_atoms_dists_list,
                                                        rrange=rrange, bins=bins,
                                                        raw_density=raw_density,
                                                        remove_radial_scaling=remove_radial_scaling)
            ind += 1
            if cpu_detach:
                rdfs_array = rdfs_array.cpu().detach().numpy()
                rr = rr.cpu().detach().numpy()

            rdfs_array_list.append(rdfs_array)

        return rdfs_array_list, rr, rdfs_dict_list
    else:
        rdfs_array, rr = parallel_compute_rdf_torch(
            [dists[edge_in_crystal_number == n] for n in range(crystal_batch.num_graphs)],
            rrange=rrange, bins=bins,
            raw_density=raw_density, remove_radial_scaling=remove_radial_scaling)
        if cpu_detach:
            rdfs_array = rdfs_array.cpu().detach().numpy()
            rr = rr.cpu().detach().numpy()
        return rdfs_array, rr


def crystal_rdf(crystal_batch,
                precomputed_distances_dict=None,
                rrange: Tuple[float, float] = (0, 10),
                bins: int = 100,
                mode: str = 'all',
                elementwise: bool = False,
                raw_density: bool = False,
                atomwise: bool = False,
                cpu_detach: bool = False,
                atomic_numbers_override: Optional[torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    faster rdf calculation
    technically returns the radial density function, not the dimensionless g(r), as we aren't
    dividing by the true density in the calculation
    """
    device = crystal_batch.pos.device
    num_graphs = crystal_batch.num_graphs
    in_inds, out_inds = get_rdf_inds(crystal_batch, mode, device)

    if precomputed_distances_dict is not None and mode == 'intermolecular':
        edges = precomputed_distances_dict['edge_index_inter']
    else:
        edges = asymmetric_radius_graph(crystal_batch.pos,
                                        batch=crystal_batch.batch,
                                        inside_inds=in_inds,
                                        convolve_inds=out_inds,
                                        r=max(rrange),
                                        max_num_neighbors=100,
                                        flow='source_to_target')

    # track which edges go with which crystals
    edge_in_crystal_number = crystal_batch.batch[edges[0]]

    # compute all the dists
    dists = torch.linalg.norm(crystal_batch.pos[edges[0]] - crystal_batch.pos[edges[1]], dim=1)

    assert not (elementwise and atomwise)

    '''
    efficiently gather the relevant distances
    '''
    if elementwise:
        dists_per_hist, sorted_dists, rdfs_dict = get_elementwise_dists(crystal_batch.z,
                                                                        edges,
                                                                        dists,
                                                                        device,
                                                                        num_graphs,
                                                                        edge_in_crystal_number,
                                                                        atomic_numbers_override)
        num_pairs = len(rdfs_dict.keys())
        batch = torch.arange(len(dists_per_hist), device=device).repeat_interleave(dists_per_hist, dim=0)
        hist, bin_edges = batch_histogram_1d(sorted_dists, batch, num_graphs * num_pairs, rrange=rrange, nbins=bins)
        if raw_density:  # todo reimplement
            rdf_density = torch.ones(num_graphs * num_pairs, device=device, dtype=torch.float32)
        else:
            assert False, "non-raw density not implemented"
        # volume of the shell at radius r+dr
        shell_volumes = (4 / 3) * torch.pi * ((bin_edges[:-1] + torch.diff(bin_edges)) ** 3 - bin_edges[:-1] ** 3)
        rdf = hist / shell_volumes[None, :] / rdf_density[:, None]  # un-smoothed radial density
        rdf = rdf.reshape(num_graphs, num_pairs, -1)  # sample-wise indexing
    elif atomwise:  # todo this is only implemented for an identical atom indexing (assumes batch is repetition of same molecule)
        dists_per_hist, sorted_dists, rdfs_dict = get_atomwise_dists(crystal_batch.z, edges, dists, device,
                                                                     num_graphs,
                                                                     edge_in_crystal_number,
                                                                     crystal_batch.num_atoms)
        num_pairs = len(rdfs_dict.keys())
        batch = torch.arange(len(dists_per_hist), device=device).repeat_interleave(dists_per_hist, dim=0)
        hist, bin_edges = batch_histogram_1d(sorted_dists, batch, num_graphs * num_pairs, rrange=rrange, nbins=bins)
        if raw_density:  # todo reimplement
            rdf_density = torch.ones(num_graphs * num_pairs, device=device, dtype=torch.float32)
        else:
            assert False, "non-raw density not implemented"
        # volume of the shell at radius r+dr
        shell_volumes = (4 / 3) * torch.pi * ((bin_edges[:-1] + torch.diff(bin_edges)) ** 3 - bin_edges[:-1] ** 3)
        rdf = hist / shell_volumes[None, :] / rdf_density[:, None]  # un-smoothed radial density
        rdf = rdf.reshape(num_graphs, num_pairs, -1)  # sample-wise indexing    else:  # average over all atom types
    else:
        dists_per_hist = [torch.sum(edge_in_crystal_number == n) for n in range(num_graphs)]
        sorted_dists = torch.cat([dists[edge_in_crystal_number == n] for n in range(num_graphs)])
        rdfs_dict = {}
        batch = repeat_interleave(dists_per_hist, device='cpu').to(device)
        hist, bin_edges = batch_histogram_1d(sorted_dists, batch, num_graphs, rrange=rrange, nbins=bins)
        rdf_density = torch.ones(num_graphs, device=device, dtype=torch.float32)
        shell_volumes = (4 / 3) * torch.pi * ((bin_edges[:-1] + torch.diff(bin_edges)) ** 3 - bin_edges[
                                                                                              :-1] ** 3)  # volume of the shell at radius r+dr
        rdf = hist / shell_volumes[None, :] / rdf_density[:, None]  # un-smoothed radial density

    rr = (bin_edges[:-1] + torch.diff(bin_edges)).requires_grad_()

    if cpu_detach:
        rdf = rdf.cpu().detach().numpy()
        rr = rr.cpu().detach().numpy()

    return rdf, rr, rdfs_dict


def get_elementwise_dists(atom_types: torch.LongTensor,
                          edges: torch.LongTensor,
                          dists: torch.Tensor,
                          device: Union[torch.device, str],
                          num_graphs: int,
                          edge_in_crystal_number: torch.LongTensor,
                          atomic_numbers_override: Optional[torch.LongTensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    if atomic_numbers_override is None:
        relevant_elements = torch.tensor([5, 6, 7, 8, 9, 15, 16, 17, 35], dtype=torch.long, device=device)
        element_symbols = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
    else:
        relevant_elements = atomic_numbers_override
        element_symbols = {int(i): str(int(i)) for i in relevant_elements}
    num_relevant_elements = len(relevant_elements)

    elements = [atom_types[edges[0]].long(), atom_types[edges[1]].long()]
    rdfs_dict = {}
    num_pairs = int((len(relevant_elements) ** 2 + len(relevant_elements)) / 2)

    # prebuild pair lists
    elem_pair_list = torch.zeros((num_pairs, len(elements[0])), dtype=torch.bool, device=device)
    elem1 = elements[0].repeat(num_relevant_elements, 1) == relevant_elements[:, None].to(device)
    elem2 = elements[1].repeat(num_relevant_elements, 1) == relevant_elements[:, None].to(device)
    ind = 0
    for i, element1 in enumerate(relevant_elements):
        for j, element2 in enumerate(relevant_elements):
            if j >= i:
                rdfs_dict[ind] = element_symbols[int(element1)] + '_to_' + element_symbols[int(element2)]
                elem_pair_list[ind] = elem1[i] * elem2[j]
                ind += 1

    bool_list = torch.zeros((num_pairs * num_graphs, len(dists)), dtype=torch.bool, device=device)
    for i in range(num_graphs):
        bool_list[i * num_pairs:(i + 1) * num_pairs, :] = (edge_in_crystal_number == i
                                                           ).repeat(num_pairs, 1) * elem_pair_list

    sorted_dists = dists.repeat(num_graphs * num_pairs, 1)[bool_list]
    # test assert bool_list.sum() == len(sorted_dists)
    dists_per_hist = bool_list.sum(1)

    return dists_per_hist, sorted_dists, rdfs_dict


def get_atomwise_dists(atom_types: torch.LongTensor,
                       edges: torch.LongTensor,
                       dists: torch.Tensor,
                       device: Union[torch.device, str],
                       num_graphs: int,
                       edge_in_crystal_number: torch.LongTensor,
                       mol_num_atoms: torch.LongTensor,
                       ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    assert all(mol_num_atoms == mol_num_atoms[0]), "atomwise rdf not set up for variable molecule sizes"
    all_atoms = torch.arange(mol_num_atoms[0], device=atom_types.device)
    num_atoms = len(all_atoms)

    atoms = [atom_types[edges[0]].long(), atom_types[edges[1]].long()]
    rdfs_dict = {}
    num_pairs = int((len(all_atoms) ** 2 + len(all_atoms)) / 2)

    # prebuild pair lists
    elem_pair_list = torch.zeros((num_pairs, len(atoms[0])), dtype=torch.bool, device=device)
    elem1 = atoms[0].repeat(num_atoms, 1) == torch.Tensor(all_atoms)[:, None].to(device)
    elem2 = atoms[1].repeat(num_atoms, 1) == torch.Tensor(all_atoms)[:, None].to(device)
    ind = 0
    for i, element1 in enumerate(all_atoms):
        for j, element2 in enumerate(all_atoms):
            if j >= i:
                rdfs_dict[ind] = str(element1) + '_to_' + str(element2)
                elem_pair_list[ind] = elem1[i] * elem2[j]
                ind += 1

    bool_list = torch.zeros((num_pairs * num_graphs, len(dists)), dtype=torch.bool, device=device)
    for i in range(num_graphs):
        bool_list[i * num_pairs:(i + 1) * num_pairs, :] = (edge_in_crystal_number == i
                                                           ).repeat(num_pairs, 1) * elem_pair_list

    sorted_dists = dists.repeat(num_graphs * num_pairs, 1)[bool_list]
    # test assert bool_list.sum() == len(sorted_dists)
    dists_per_hist = bool_list.sum(1)

    return dists_per_hist, sorted_dists, rdfs_dict


def get_rdf_inds(crystal_batch,
                 mode: str,
                 device: Union[torch.device, str]
                 ) -> Tuple[torch.LongTensor, torch.LongTensor]:
    #  whether to include intramolecular edges

    if crystal_batch.aux_ind is not None:
        in_inds = torch.where(crystal_batch.aux_ind == 0)[0]
        if mode == 'intermolecular':
            out_inds = torch.where(crystal_batch.aux_ind == 1)[0].to(device)
        elif mode == 'all':
            out_inds = torch.arange(len(crystal_batch.pos)).to(device)
        elif mode == 'intramolecular':
            out_inds = in_inds
        else:
            print(mode + ' is not a valid rdf mode!')
            assert False
    else:
        # if we have no inside/outside labels, just consider the whole thing one big molecule
        in_inds = torch.arange(len(crystal_batch.pos)).to(device)
        out_inds = in_inds

    return in_inds, out_inds


# inspired by https://github.com/pytorch/pytorch/issues/99719#issuecomment-1664135524
def batch_histogram_1d(data_tensor: torch.Tensor,
                       batch: torch.LongTensor,
                       num_hists: int,
                       rrange: list[int, int] = [0, 10],
                       nbins: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    very fast approximate batch histogram accurate up to n digits where n is log10(nbins)

    data_tensor: 1d tensor of datapoints in torch.float32, from an arbitrary number of samples of arbitrary size
    batch: index of which sample each datapoint belongs to
    num_graphs: number of samples represented in data_tensor, equal to max(batch) + 1
    rrange: histogram range
    nbins: number of bins of resulting histogram - more bins reduces the distortion of the discretization procedure this hist uses
    """
    epsilon = (rrange[1] - rrange[0]) / nbins  # important to bracket bins
    ones = torch.ones_like(data_tensor)
    scatter_inds = batch.long() * nbins + (torch.round(
        data_tensor.clip(min=rrange[0], max=rrange[1] - epsilon) / rrange[1] * nbins)).long()  # convert float to long

    flat_hist = scatter(ones, scatter_inds, dim=0, dim_size=num_hists * nbins)
    hists = flat_hist.reshape(num_hists, nbins)

    return hists, torch.linspace(*rrange, nbins + 1, device=data_tensor.device)

    # a,b = torch.unique(batch, return_counts = True)  # test
    # for i, num in enumerate(a):
    #     assert hists[int(num)].sum() == b[i], f'{i} {num}'


def parallel_compute_rdf_torch(dists_list: torch.tensor, raw_density=True, rrange=None, bins=None,
                               remove_radial_scaling=False):
    """
    Compute the radial distribution given a list of distances with parallel execution for speed.

    Parameters
    ----------
    dists_list : list of torch tensors
    raw_density : bool
        If true, use uniform density of 1 everywhere, else estimate the density from the dist list.
    rrange : [min_range, max_range], optional
    bins : int
    remove_radial_scaling : removes inverse scaling in RDF, giving something more like the unnormalized radial CDF.

    Returns
    -------
    rdf : torch.tensor(n, n_bins)
    bin_edges : torch.tensor(n_bins + 1)
    """

    hist_range = [0.5, 10] if rrange is None else rrange
    hist_bins = 100 if bins is None else bins

    if not raw_density:  # estimate the density from the distances
        rdf_density = torch.zeros(len(dists_list)).to(dists_list[0].device)
        for i in range(len(dists_list)):
            dists = dists_list[i]
            try:
                sorted_dists = torch.sort(dists)[0][
                               :len(dists) // 2]  # we will use 1/2 the dist radius to avoid edge / cutoff effects
                volume = 4 / 3 * torch.pi * torch_ptp(
                    sorted_dists) ** 3  # volume of a sphere #np.ptp(sorted_dists[:, 0]) * np.ptp(sorted_dists[:, 1]) * np.ptp(sorted_dists[:, 2])
                # number of particles divided by the volume
                rdf_density[i] = len(sorted_dists) / volume
            except:
                rdf_density[i] = 1
    else:
        rdf_density = torch.ones(len(dists_list), device=dists_list[0].device, dtype=torch.float32)

    hh_list = torch.stack(
        [torch.histc(dists, min=hist_range[0], max=hist_range[1], bins=hist_bins) for dists in dists_list])
    rr = torch.linspace(hist_range[0], hist_range[1], hist_bins + 1, device=hh_list.device)
    if remove_radial_scaling:
        rdf = hh_list / rdf_density[:, None]  # un-smoothed radial density
    else:
        shell_volumes = (4 / 3) * torch.pi * (
                    (rr[:-1] + torch.diff(rr)) ** 3 - rr[:-1] ** 3)  # volume of the shell at radius r+dr
        rdf = hh_list / shell_volumes[None, :] / rdf_density[:, None]  # un-smoothed radial density

    return rdf, (rr[:-1] + torch.diff(rr)).requires_grad_()  # rdf and x-axis


def compute_rdf_distmat(rdf_record, rr, show_tqdm=True, chunk_size=250):
    # todo parallelize this function via async
    rdf_dists = torch.zeros(rdf_record.shape[0], rdf_record.shape[0])

    for i in tqdm(range(1, len(rdf_record)), disable=not show_tqdm):
        num_chunks = i // chunk_size + 1
        for j in range(num_chunks):
            start_ind = j * chunk_size
            stop_ind = min(i, (j + 1) * chunk_size)
            rdf_dists[i, start_ind:stop_ind] = compute_rdf_distance(
                rdf_record[i],
                rdf_record[start_ind:stop_ind],  # save on energy & memory
                rr,
                n_parallel_rdf2=stop_ind - start_ind)
    rdf_dists = rdf_dists + rdf_dists.T  # symmetric distance matrix
    rdf_dists = torch.log10(1 + rdf_dists)
    return rdf_dists


def compute_rdf_distmat_block(rdf_record: torch.Tensor,
                              rr,
                              i_range,
                              j_range) -> torch.Tensor:
    device = rdf_record.device

    num_rdfs = (j_range[1] - j_range[0]) * (i_range[1] - i_range[0])
    i_inds = torch.arange(i_range[0], min(i_range[1], num_rdfs), device=device)
    j_inds = torch.arange(j_range[0], min(j_range[1], num_rdfs), device=device)

    mg = torch.meshgrid(i_inds, j_inds)
    ind1, ind2 = mg[0].flatten(), mg[1].flatten()
    good_bools = ind1 > ind2
    ind1_g = ind1[good_bools]
    ind2_g = ind2[good_bools]
    dists = compute_rdf_distance(
        rdf_record[ind1_g],
        rdf_record[ind2_g],
        rr,
    )
    distmat = torch.zeros((i_range[1] - i_range[0], j_range[1] - j_range[0]), dtype=torch.float32, device=device)
    distmat[ind1_g - i_range[1], ind2_g - j_range[1]] = dists

    return distmat


def compute_rdf_distmat_parallel(rdf_record, rr, num_cpus, chunk_size=250):
    import multiprocessing as mp
    from math import ceil

    pool = mp.Pool(num_cpus)
    num_rdfs = len(rdf_record)
    num_chunks = ceil(len(rdf_record) / chunk_size)

    out = []
    for i in range(num_chunks):
        for j in range(num_chunks):
            out.append(
                pool.apply_async(
                    compute_rdf_distmat_block,
                    (rdf_record,
                     rr,
                     [i * chunk_size, min((i + 1) * chunk_size, num_rdfs)],
                     [j * chunk_size, min((j + 1) * chunk_size, num_rdfs)],
                     )
                )
            )
    pool.close()
    pool.join()
    out = [thing.get() for thing in out]

    rdf_dists = torch.zeros(rdf_record.shape[0], rdf_record.shape[0])
    ind = 0
    for i in range(num_chunks):
        for j in range(num_chunks):
            rdf_dists[i * chunk_size: min((i + 1) * chunk_size, num_rdfs),
            j * chunk_size: min((j + 1) * chunk_size, num_rdfs)] = out[ind]
            ind += 1

    rdf_dists = rdf_dists + rdf_dists.T  # symmetric distance matrix
    rdf_dists = torch.log10(1 + rdf_dists)
    return rdf_dists


def compute_rdf_distance(rdf1, rdf2, rr, n_parallel_rdf2: int = None, return_numpy: bool = False):
    """
    Compute a distance metric between two radial distribution functions including sub_rdfs where sub_rdfs are e.g., particular interatomic RDFS within a certain sample (elementwise or atomwise modes).

    If all inputs are numpy arrays, output will be a numpy array, and vice-versa with torch tensors.
    Parameters
    ----------
    rdf1 : array(n_sub_rdfs, n_bins)
    rdf2 : array(n_sub_rdfs, n_bins)
    rr : array(n_bins + 1)
        is the bin edges used for both rdfs
    n_parallel_rdf2: int, optional
        Optionally in parallel compare many rdf2's to a single rdf1
    Returns
    -------

    """

    if not torch.is_tensor(rdf1):
        torch_rdf1 = torch.Tensor(rdf1)
        torch_rdf2 = torch.Tensor(rdf2)
        return_numpy = True
    else:
        torch_rdf1 = rdf1
        torch_rdf2 = rdf2

    if not torch.is_tensor(rr):
        torch_range = torch.Tensor(rr)
    else:
        torch_range = rr

    if n_parallel_rdf2 is not None:
        torch_rdf1_f = torch_rdf1.tile(n_parallel_rdf2, 1, 1)
    else:
        torch_rdf1_f = torch_rdf1

    emd = earth_movers_distance_torch(torch_rdf1_f, torch_rdf2)

    # rescale the distance from units of bins to the real physical range
    range_normed_emd = emd / len(torch_range) ** 2 * (torch_range[-1] - torch_range[0])
    # do not adjust the above - distance is extensive weirdly extensive in bin scaling

    # aggregate rdf components according to pairwise mean weight
    aggregation_weight = (torch_rdf1_f.sum(-1) + torch_rdf2.sum(-1)) / 2
    distance = (range_normed_emd * aggregation_weight).mean(-1)

    assert torch.sum(torch.isnan(distance)) == 0, "NaN EMD Distances Computed"
    if return_numpy:
        distance = distance.cpu().detach().numpy()

    return distance


def earth_movers_distance_torch(pdf1: torch.tensor, pdf2: torch.tensor):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    Parameters
    ----------
    pdf1 : torch.tensor(n,i)
    pdf2 : torch.tensor(n,i)

    Returns
    -------
    emd: torch.tensor(n)
    """

    return torch.sum(torch.abs(torch.cumsum(pdf1, dim=-1) - torch.cumsum(pdf2, dim=-1)), dim=-1)


def earth_movers_distance_np(pdf1: np.ndarray, pdf2: np.ndarray):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    Parameters
    ----------
    pdf1 : np.array(n,i)
    pdf2 : np.array(n,i)

    Returns
    -------
    emd: np.array(n)
    """
    return np.sum(np.abs(np.cumsum(pdf1, axis=-1) - np.cumsum(pdf2, axis=-1)), axis=-1)

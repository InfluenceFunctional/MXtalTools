import itertools

import torch
from models.asymmetric_radius_graph import asymmetric_radius_graph
from common.rdf_calculation import parallel_compute_rdf_torch


def crystal_rdf(crystaldata, precomputed_distances_dict=None, rrange=[0, 10], bins=100, mode='all', elementwise=False, raw_density=False, atomwise=False, cpu_detach=False, remove_radial_scaling=False):
    """
    compute the RDF for all the supercells in a CrystalData object

    crystal_data: crystaldata object containing a batch of molecular crystals relevant information
    rrange: range over which to compute radial distribution function
    bins: number of bins for RDF histogramming
    mode: 'intramolecular' consider only intramolecular edges, 'intermolecular' consider only intermolecular edges, 'all' consider both
    elementwise: compute a separate RDF for all pairs of all elements in each crystal (exclusive with atomwise)
    raw_dentiy: False estimate the crystal density. True use raw RDF outputs (same as setting density always to 1)
    atomwise: compute a separate RDF for all pairs of unique atom indexes per molecule in each crystal (exclusive with elementwise)
    cpu_detach: True return outputs as numpy array on cpu. False return outputs as torch tensor on the same device as the input crystaldata
    """
    device = crystaldata.pos.device
    #  whether to include intramolecular edges
    if crystaldata.aux_ind is not None:
        in_inds = torch.where(crystaldata.aux_ind == 0)[0]
        if mode == 'intermolecular':
            out_inds = torch.where(crystaldata.aux_ind == 1)[0].to(device)
        elif mode == 'all':
            out_inds = torch.arange(len(crystaldata.pos)).to(device)
        elif mode == 'intramolecular':
            out_inds = in_inds
        else:
            print(mode + ' is not a valid rdf mode!')
            assert False
    else:
        # if we have no inside/outside labels, just consider the whole thing one big molecule
        in_inds = torch.arange(len(crystaldata.pos)).to(device)
        out_inds = in_inds

    if precomputed_distances_dict is not None and mode == 'intermolecular':
        edges = precomputed_distances_dict['edge_index_inter']
    else:
        edges = asymmetric_radius_graph(crystaldata.pos,
                                        batch=crystaldata.batch,
                                        inside_inds=in_inds,
                                        convolve_inds=out_inds,
                                        r=max(rrange), max_num_neighbors=500, flow='source_to_target')

    # track which edges go with which crystals
    edge_in_crystal_number = crystaldata.batch[edges[0]]
    for i in range(crystaldata.num_graphs):
        if i == 0:
            edges_ptr = torch.zeros(crystaldata.num_graphs + 1, device=device, dtype = torch.long)
            edges_ptr[1] = torch.sum(edge_in_crystal_number == i)
        else:
            edges_ptr[i + 1] = edges_ptr[i] + torch.sum(edge_in_crystal_number == i)

    # compute all the dists
    dists = torch.linalg.norm(crystaldata.pos[edges[0]] - crystaldata.pos[edges[1]], dim=1)

    assert not (elementwise and atomwise)

    if elementwise:  # todo this could also be sped up
        relevant_elements = [5, 6, 7, 8, 9, 15, 16, 17, 35]
        element_symbols = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
        elements = [crystaldata.x[edges[0], 0], crystaldata.x[edges[1], 0]]
        rdfs_dict = {}
        rdfs_array = torch.zeros((crystaldata.num_graphs, int((len(relevant_elements) ** 2 + len(relevant_elements)) / 2), bins), device=dists.device)

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
                [dists[(edge_in_crystal_number == n) * elem_pair_list[i]] for n in range(crystaldata.num_graphs)],
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
        for i in range(crystaldata.num_graphs):
            # all_atom_inds.append(torch.arange(crystal_data.mol_size[i]).tile(int((crystal_data.batch == i).sum() // crystal_data.mol_size[i])))
            # assume only that the order of atoms is patterned in all images
            canonical_conformer_coords = crystaldata.pos[crystaldata.ptr[i]:crystaldata.ptr[i] + int(crystaldata.mol_size[i])]
            centroid = canonical_conformer_coords.mean(0)
            mol_dists = torch.linalg.norm(canonical_conformer_coords - centroid[None, :], dim=-1)
            inds = torch.argsort(mol_dists)
            all_atom_inds.append(inds.tile(int((crystaldata.batch == i).sum() // crystaldata.mol_size[i])))

        atom_inds = torch.cat(all_atom_inds)

        atoms_in_edges = [atom_inds[edges[0]], atom_inds[edges[1]]]

        ind = 0
        for n in range(crystaldata.num_graphs):
            # all possible combinations of unique atoms on this graph
            atom_pairs = torch.Tensor(list(itertools.combinations(torch.arange(int(crystaldata.mol_size[n])), 2)))
            rdfs_dict_list.append(atom_pairs)  # record the pairs for reporting purposes

            in_crystal_inds = torch.where(edge_in_crystal_number == n)[0]  # atom indices in this crystal

            atom_locations = [[atoms_in_edges[0][in_crystal_inds] == m, atoms_in_edges[1][in_crystal_inds] == m] for m in range(int(atom_pairs.max()) + 1)]

            relevant_atoms_dists_list = [dists[in_crystal_inds[torch.logical_and(atom_locations[int(atom_pairs[m, 0])][0],
                                                                                 atom_locations[int(atom_pairs[m, 1])][1])]]
                                         for m in range(len(atom_pairs))]

            rdfs_array, rr = parallel_compute_rdf_torch(relevant_atoms_dists_list,
                                                        rrange=rrange, bins=bins,
                                                        raw_density=raw_density, remove_radial_scaling=remove_radial_scaling)
            ind += 1
            if cpu_detach:
                rdfs_array = rdfs_array.cpu().detach().numpy()
                rr = rr.cpu().detach().numpy()

            rdfs_array_list.append(rdfs_array)

        return rdfs_array_list, rr, rdfs_dict_list
    else:
        rdfs_array, rr = parallel_compute_rdf_torch([dists[edge_in_crystal_number == n] for n in range(crystaldata.num_graphs)],
                                                    rrange=rrange, bins=bins,
                                                    raw_density=raw_density, remove_radial_scaling=remove_radial_scaling)
        if cpu_detach:
            rdfs_array = rdfs_array.cpu().detach().numpy()
            rr = rr.cpu().detach().numpy()
        return rdfs_array, rr

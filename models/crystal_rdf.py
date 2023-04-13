import itertools

import torch
from models.asymmetric_radius_graph import asymmetric_radius_graph
from common.rdf_calculation import parallel_compute_rdf_torch


def crystal_rdf(crystal_data, rrange=[0, 10], bins=100, mode='all', elementwise=False, raw_density=False, atomwise=False):
    """
    compute the RDF for all the supercells in a CrystalData object
    """
    device = crystal_data.pos.device
    #  whether to include intramolecular edges
    if crystal_data.aux_ind is not None:
        in_inds = torch.where(crystal_data.aux_ind == 0)[0]
        if mode == 'intermolecular':
            out_inds = torch.where(crystal_data.aux_ind == 1)[0].to(device)
        elif mode == 'all':
            out_inds = torch.arange(len(crystal_data.pos)).to(device)
        elif mode == 'intramolecular':
            out_inds = in_inds
        else:
            print(mode + ' is not a valid rdf mode!')
            assert False
    else:
        # if we have no inside/outside labels, just consider the whole thing one big molecule
        in_inds = torch.arange(len(crystal_data.pos)).to(device)
        out_inds = in_inds

    edges = asymmetric_radius_graph(crystal_data.pos,
                                    batch=crystal_data.batch,
                                    inside_inds=in_inds,
                                    convolve_inds=out_inds,
                                    r=max(rrange), max_num_neighbors=500, flow='source_to_target')

    # track which edges go with which crystals
    edge_in_crystal_number = crystal_data.batch[edges[0]]

    # compute all the dists
    dists = (crystal_data.pos[edges[0]] - crystal_data.pos[edges[1]]).pow(2).sum(dim=-1).sqrt()

    assert not (elementwise and atomwise)

    if elementwise:
        relevant_elements = [5, 6, 7, 8, 9, 15, 16, 17, 35]
        element_symbols = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
        elements = [crystal_data.x[edges[0], 0], crystal_data.x[edges[1], 0]]
        rdfs_dict = {}
        rdfs_array = torch.zeros((crystal_data.num_graphs, int((len(relevant_elements) ** 2 + len(relevant_elements)) / 2), bins))
        ind = 0
        for i, element1 in enumerate(relevant_elements):
            for j, element2 in enumerate(relevant_elements):
                if j >= i:
                    rdfs_dict[ind] = element_symbols[element1] + ' to ' + element_symbols[element2]
                    rdfs_array[:, ind], rr = parallel_compute_rdf_torch([dists[(edge_in_crystal_number == n) * (elements[0] == element1) * (elements[1] == element2)] for n in range(crystal_data.num_graphs)],
                                                                        rrange=rrange, bins=bins, raw_density=raw_density)
                    ind += 1
        return rdfs_array, rr, rdfs_dict

    elif atomwise:  # generate atomwise indices which are shared between samples
        rdfs_array_list = []
        rdfs_dict_list = []
        all_atom_inds = []
        # abs_atom_inds = []
        for i in range(crystal_data.num_graphs):
            # all_atom_inds.append(torch.arange(crystal_data.mol_size[i]).tile(int((crystal_data.batch == i).sum() // crystal_data.mol_size[i])))
            # assume only that the order of atoms is patterned in all images
            canonical_conformer_coords = crystal_data.pos[crystal_data.ptr[i]:crystal_data.ptr[i] + int(crystal_data.mol_size[i])]
            centroid = canonical_conformer_coords.mean(0)
            mol_dists = torch.linalg.norm(canonical_conformer_coords - centroid[None, :], dim=-1)
            # TODO if any dists are within 32 bit uncertainty, we have to break the symmetry somehow
            inds = torch.argsort(mol_dists)
            all_atom_inds.append(inds.tile(int((crystal_data.batch == i).sum() // crystal_data.mol_size[i])))
            # abs_atom_inds.append(all_atom_inds[-1] + crystal_data.ptr[i])

        atom_inds = torch.cat(all_atom_inds)
        # abs_atom_inds = torch.cat(abs_atom_inds)

        atoms_in_edges = [atom_inds[edges[0]], atom_inds[edges[1]]]

        # t0 = time.time()
        # ind = 0
        # for n in range(crystal_data.num_graphs):
        #     # all possible combinations of unique atoms on this graph
        #     atom_pairs = torch.Tensor(list(itertools.combinations(torch.arange(int(crystal_data.mol_size[n])), 2)))
        #     rdfs_dict_list.append(atom_pairs)  # record the pairs for reporting purposes
        #
        #     dists_in_this_crystal = edge_in_crystal_number == n  # edge indices in this crystal
        #
        #     atoms_in_atom_pairs = [torch.logical_and(atoms_in_edges[0] == atom_pairs[m, 0], atoms_in_edges[1] == atom_pairs[m, 1]) for m in range(len(atom_pairs))]
        #     relevant_atoms_dists_list = [dists[torch.logical_and(dists_in_this_crystal, atoms_in_atom_pairs[m])]
        #                                  for m in range(len(atom_pairs))]

        # t1 = time.time()
        ind = 0
        for n in range(crystal_data.num_graphs):
            # this way is slightly faster than the above, and should scale well to larger batch sizes
            # all possible combinations of unique atoms on this graph
            atom_pairs = torch.Tensor(list(itertools.combinations(torch.arange(int(crystal_data.mol_size[n])), 2)))
            rdfs_dict_list.append(atom_pairs)  # record the pairs for reporting purposes

            in_crystal_inds = torch.where(edge_in_crystal_number == n)[0]  # atom indices in this crystal

            atom_locations = [[atoms_in_edges[0][in_crystal_inds] == m, atoms_in_edges[1][in_crystal_inds] == m] for m in range(int(atom_pairs.max()) + 1)]

            relevant_atoms_dists_list = [dists[in_crystal_inds[torch.logical_and(atom_locations[int(atom_pairs[m, 0])][0],
                                                                                 atom_locations[int(atom_pairs[m, 1])][1])]]
                                         for m in range(len(atom_pairs))]
            #
            # t2 = time.time()
            # aa = 0
            # for ii in range(len(relevant_atoms_dists_list)):
            #     aa += torch.sum(torch.abs(relevant_atoms_dists_list2[ii] - relevant_atoms_dists_list[ii]))
            # print(aa)
            # print(t1-t0)
            # print(t2-t1)

            rdfs_array, rr = parallel_compute_rdf_torch(relevant_atoms_dists_list,
                                                        rrange=rrange, bins=bins, raw_density=raw_density)
            ind += 1

            rdfs_array_list.append(rdfs_array)

        return rdfs_array_list, rr, rdfs_dict_list
    else:
        return parallel_compute_rdf_torch([dists[edge_in_crystal_number == n] for n in range(crystal_data.num_graphs)], rrange=rrange, bins=bins, raw_density=raw_density)

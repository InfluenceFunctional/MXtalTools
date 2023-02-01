import torch
from models.asymmetric_radius_graph import asymmetric_radius_graph
from utils import parallel_compute_rdf_torch

def crystal_rdf(crystaldata, rrange=[0, 10], bins=100, mode='all', elementwise=False, raw_density=False, atomwise=False):
    '''
    compute the RDF for all the supercells in a CrystalData object
    without respect for atom type
    '''

    # whether or not to include intramolecular edges
    if crystaldata.aux_ind is not None:
        in_inds = torch.where(crystaldata.aux_ind == 0)[0]
        if mode == 'intermolecular':
            out_inds = torch.where(crystaldata.aux_ind == 1)[0].to(crystaldata.pos.device)
        elif mode == 'all':
            out_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
        elif mode == 'intramolecular':
            out_inds = in_inds
        else:
            print(mode + ' is not a valid rdf mode!')
            assert False
    else:
        # if we have no inside/outside labels, just consider the whole thing one big molecule
        in_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
        out_inds = in_inds

    # get edges
    edges = asymmetric_radius_graph(crystaldata.pos,
                                    batch=crystaldata.batch,
                                    inside_inds=in_inds,
                                    convolve_inds=out_inds,
                                    r=max(rrange), max_num_neighbors=500, flow='source_to_target')

    # track which edges go with which crystals
    crystal_number = crystaldata.batch[edges[0]]

    # compute all the dists
    dists = (crystaldata.pos[edges[0]] - crystaldata.pos[edges[1]]).pow(2).sum(dim=-1).sqrt()

    assert not (elementwise and atomwise)

    if elementwise:
        if raw_density:
            density = torch.ones(crystaldata.num_graphs).to(dists.device)
        else:
            density = None
        relevant_elements = [5, 6, 7, 8, 9, 15, 16, 17, 35]
        element_symbols = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
        elements = [crystaldata.x[edges[0], 0], crystaldata.x[edges[1], 0]]
        rdfs_dict = {}
        rdfs_array = torch.zeros((crystaldata.num_graphs, int((len(relevant_elements) ** 2 + len(relevant_elements)) / 2), bins))
        ind = 0
        for i, element1 in enumerate(relevant_elements):
            for j, element2 in enumerate(relevant_elements):
                if j >= i:
                    rdfs_dict[ind] = element_symbols[element1] + ' to ' + element_symbols[element2]
                    rdfs_array[:, ind], rr = parallel_compute_rdf_torch([dists[(crystal_number == n) * (elements[0] == element1) * (elements[1] == element2)] for n in range(crystaldata.num_graphs)],
                                                                        rrange=rrange, bins=bins, density=density)
                    ind += 1
        return rdfs_array, rr, rdfs_dict
    elif atomwise: # assumes that atom indices are constantly tiled within samples and between samples
        # generate atomwise indices
        rdfs_array_list = []
        rdfs_dict_list = []
        all_atom_inds = []
        for i in range(crystaldata.num_graphs):
            all_atom_inds.append(torch.arange(crystaldata.mol_size[i]).tile(int((crystaldata.batch == i).sum() // crystaldata.mol_size[i])))
        atom_inds = torch.cat(all_atom_inds)
        atoms = [atom_inds[edges[0]].to(crystaldata.x.device), atom_inds[edges[1]].to(crystaldata.x.device)]
        ind = 0
        for n in range(crystaldata.num_graphs):
            # all possible combinations of atoms on this graph
            atom = []
            for i in range(int(crystaldata.mol_size[n])):
                for j in range(int(crystaldata.mol_size[n])):
                    if j >= i:
                        atom.append([i, j])
            atom = torch.Tensor(atom)
            rdfs_dict_list.append(atom) # record the pairs

            if raw_density:
                density = torch.ones(len(atom)).to(dists.device)
            else:
                density = None

            rdfs_array, rr = parallel_compute_rdf_torch([dists[(crystal_number == n) * (atoms[0] == atom[m, 0]) * (atoms[1] == atom[m, 1])]
                                                         for m in range(len(atom))],
                                                        rrange=rrange, bins=bins, density=density)
            ind += 1

            rdfs_array_list.append(rdfs_array)

        return rdfs_array_list, rr, rdfs_dict_list

    else:
        if raw_density:
            density = torch.ones(crystaldata.num_graphs).to(dists.device)
        else:
            density = None
        return parallel_compute_rdf_torch([dists[crystal_number == n] for n in range(crystaldata.num_graphs)], rrange=rrange, bins=bins, density=density)

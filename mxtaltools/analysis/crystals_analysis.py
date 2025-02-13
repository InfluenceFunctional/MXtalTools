from mxtaltools.dataset_utils.CrystalData import CrystalData
from mxtaltools.models.modules.components import construct_radial_graph


def get_intermolecular_dists_dict(supercell_data: CrystalData,
                                  conv_cutoff: float,
                                  max_num_neighbors: int = 10000):
    dist_dict = {}
    edges_dict = construct_radial_graph(
        supercell_data.pos,
        supercell_data.batch,
        supercell_data.ptr,
        conv_cutoff,
        max_num_neighbors,
        aux_ind=supercell_data.aux_ind,
        mol_ind=supercell_data.mol_ind,
    )
    dist_dict.update(edges_dict)
    dist_dict['num_graphs'] = supercell_data.num_graphs
    dist_dict['graph_size'] = supercell_data.num_atoms
    dist_dict['outside_batch'] = supercell_data.batch
    dist_dict['intramolecular_dist'] = (
        (supercell_data.pos[edges_dict['edge_index'][0]] - supercell_data.pos[
            edges_dict['edge_index'][1]]).pow(2).sum(
            dim=-1).sqrt())

    dist_dict['intramolecular_dist_atoms'] = \
        [supercell_data.x[edges_dict['edge_index'][0], 0].long(),
         supercell_data.x[edges_dict['edge_index'][1], 0].long()]

    dist_dict['intermolecular_dist'] = (
        (supercell_data.pos[edges_dict['edge_index_inter'][0]] - supercell_data.pos[
            edges_dict['edge_index_inter'][1]]).pow(2).sum(
            dim=-1).sqrt())

    dist_dict['intermolecular_dist_batch'] = supercell_data.batch[edges_dict['edge_index_inter'][0]]

    dist_dict['intermolecular_dist_atoms'] = \
        [supercell_data.x[edges_dict['edge_index_inter'][0], 0].long(),
         supercell_data.x[edges_dict['edge_index_inter'][1], 0].long()]

    dist_dict['intermolecular_partial_charges'] = \
        [supercell_data.p_charges[edges_dict['edge_index_inter'][0]],
         supercell_data.p_charges[edges_dict['edge_index_inter'][1]]]

    return dist_dict

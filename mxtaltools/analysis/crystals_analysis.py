from mxtaltools.dataset_utils.CrystalData import CrystalData
from mxtaltools.models.modules.components import construct_radial_graph


def get_intermolecular_dists_dict(cluster_batch,
                                  conv_cutoff: float,
                                  max_num_neighbors: int = 10000):
    dist_dict = {}
    edges_dict = construct_radial_graph(
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

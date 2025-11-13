import torch
from ase.build import niggli_reduce

from mxtaltools.common.ase_interface import data_batch_to_ase_mols_list
from ase.visualize import view
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

if __name__ == "__main__":
    data_path = r"D:\crystal_datasets\protonated_nicoam\_chunk_0.pkl"
    dataset = torch.load(data_path, weights_only=False)
    batch = collate_data_list(dataset)
    clusters = batch.mol2cluster(cutoff=6)
    clusters.visualize(mode='distance', cutoff=4)
    clusters.construct_radial_graph(cutoff=6)
    rdfs, bins, _ = clusters.compute_rdf()
    dists = compute_rdf_distance(rdfs[None,0], rdfs[None,1], bins)
    outs = clusters.compute(['lj','silu','niggli','qlj'])

    #metric tensor distance
    ref_metric_tensor = (batch.T_fc[0] @ batch.T_fc[0]).cpu()
    metric_tensor = (batch.T_fc[1] @ batch.T_fc[1]).cpu()
    mdist = torch.sqrt(torch.sum(torch.pow(ref_metric_tensor - metric_tensor, 2)))

    for data in dataset:
        udata = data.clone()
        udata.build_unit_cell()
        udata.pos = udata.unit_cell_pos
        udata.z = udata.z.repeat(udata.sym_mult)
        ase_mol = data_batch_to_ase_mols_list(udata)[0]
        ase_mol.pbc = True
        niggli_reduce(ase_mol)

    aa = 1
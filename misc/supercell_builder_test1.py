"""
1. Generate supercells using old and new methods
2. Confirm outputs are identical via RDF analysis
3. Visualize RDF analysis
4. Check timing
"""
from time import time

times = []
times.append(time())
supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
    unit_cell_to_convolution_cluster(supercell_data.unit_cell_pos,
                                     cell_vector_list,
                                     self.device,
                                     node_feats_list,
                                     supercell_data.sym_mult,
                                     sorted_fractional_translations=self.sorted_fractional_translations,
                                     supercell_scale=supercell_size, cutoff=graph_convolution_cutoff,
                                     pare_to_convolution_cluster=pare_to_convolution_cluster)

supercell_data1 = update_supercell_data(supercell_data.clone(),
                                        supercell_atoms_list,
                                        supercell_list,
                                        ref_mol_inds_list,
                                        supercell_data.unit_cell_pos)
times.append(time())
supercell_data2, n_copies2 = new_unit_cell_to_convolution_cluster(supercell_data.clone(), cell_vector_list,
                                                                  self.sorted_fractional_translations, self.device)
times.append(time())
from mxtaltools.models.functions.crystal_rdf import new_crystal_rdf
from mxtaltools.models.utils import get_intermolecular_dists_dict
from mxtaltools.common.utils import compute_rdf_distance
times.append(time())
dist_dict1 = get_intermolecular_dists_dict(supercell_data1, 6, 100)
times.append(time())
rdf1, rr, _ = new_crystal_rdf(supercell_data1, dist_dict1,
                              rrange=[0, 6], bins=2000,
                              mode='intermolecular', elementwise=True, raw_density=True,
                              cpu_detach=True)
times.append(time())
dist_dict2 = get_intermolecular_dists_dict(supercell_data2, 6, 100)
times.append(time())
rdf2, rr, _ = new_crystal_rdf(supercell_data2, dist_dict2,
                              rrange=[0, 6], bins=2000,
                              mode='intermolecular', elementwise=True, raw_density=True,
                              cpu_detach=True)

rdf_dists = torch.zeros(supercell_data1.num_graphs, device='cpu', dtype=torch.float32)
for i in range(supercell_data1.num_graphs):
    rdf_dists[i] = compute_rdf_distance(rdf1[i], rdf2[i], rr) / int(supercell_data1.num_atoms[i])
    # divides out the trivial size correlation

# should be small or ideally exactly zero
print(rdf_dists)

import plotly.graph_objects as go

fig = go.Figure()
fig.add_scattergl(x=rr[1:], y=rdf1[0].sum(0), mode='lines')
fig.add_scattergl(x=rr[1:], y=rdf2[0].sum(0), mode='lines')
fig.show()

for ind in range(1, len(times)):
    print(times[ind] - times[ind-1])
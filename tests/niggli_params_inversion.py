from mxtaltools.common.utils import std_normal_to_uniform, uniform_to_std_normal
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.crystal_building.random_crystal_sampling import niggli_box_vectors_to_randn, \
    randn_to_niggli_box_vectors, std_normal_to_aunit_orientations, aunit_orientations_to_std_normal
import torch
import torch.nn.functional as F

num_samples = 1000
sg_inds = torch.randint(1, 40, (num_samples,))
rands = torch.randn((num_samples, 12)) * 2
mol_radii = torch.rand(num_samples) * 10 + 4
asym_unit_dict = ASYM_UNITS.copy()
sgs_to_tensorize = asym_unit_dict.keys()
for key in sgs_to_tensorize:
    asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key])


"""
enforce this cycle is reversible
std normal --> niggli --> std normal
"""
"forward operation"
a, al, b, be, c, ga = randn_to_niggli_box_vectors(
    asym_unit_dict,
    mol_radii,
    rands,
    sg_inds
)
aunit_orientation = std_normal_to_aunit_orientations(rands[:, 9:])
aunit_scaled_centroids = std_normal_to_uniform(rands[:, 6:9])
aunit_centroid = aunit_scaled_centroids * torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])

"reverse operation"
a_s, al_s, b_s, be_s, c_s, ga_s = niggli_box_vectors_to_randn(
    asym_unit_dict,
    mol_radii,
    a, b, c, al, be, ga,
    sg_inds
)

aunit_scaled_centroids = aunit_centroid / torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])
std_aunit = uniform_to_std_normal(aunit_scaled_centroids)
std_orientation = aunit_orientations_to_std_normal(aunit_orientation)

new_rands = torch.stack([a_s, b_s, c_s, al_s, be_s, ga_s]).T
new_rands = torch.cat([new_rands, std_aunit, std_orientation], dim=1)

print((rands - new_rands).abs().amax(0))
print((rands - new_rands).abs().mean(0))


import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=4, cols=3)
for ind in range(12):
    row = ind // 3 + 1
    col = ind % 3 + 1
    fig.add_scatter(x=rands[:, ind], y=new_rands[:, ind], row=row,col=col, mode='markers')
    fig.add_scatter(x=[min(torch.amin(rands[:, ind]), torch.amin(new_rands[:, ind])),
                       max(torch.amax(rands[:, ind]), torch.amax(new_rands[:, ind]))],
                    y=[min(torch.amin(rands[:, ind]), torch.amin(new_rands[:, ind])),
                       max(torch.amax(rands[:, ind]), torch.amax(new_rands[:, ind]))],
                    row=row, col=col,
                    marker_color='black')
fig.show()

fig = make_subplots(rows=4, cols=3)
for ind in range(12):
    row = ind // 3 + 1
    col = ind % 3 + 1
    fig.add_histogram(x=rands[:, ind], showlegend=False, row=row, col=col, nbinsx=100, marker_color='red')
    fig.add_histogram(x=new_rands[:, ind], showlegend=False, row=row, col=col, nbinsx=100, marker_color='blue')
fig.show()



assert torch.all(torch.isclose(rands, new_rands))

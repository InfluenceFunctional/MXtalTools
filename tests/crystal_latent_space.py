import torch

from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.crystal_building.crystal_latent_transforms import AunitTransform, NiggliTransform, BoundedTransform, \
    StdNormalTransform, CompositeTransform, SquashingTransform
from mxtaltools.dataset_utils.utils import collate_data_list
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


def plot_all_dists(params_list):
    for samples in params_list:
        lattice_features = ['cell_a', 'cell_b', 'cell_c',
                            'cell_alpha', 'cell_beta', 'cell_gamma',
                            'aunit_x', 'aunit_y', 'aunit_z',
                            'orientation_1', 'orientation_2', 'orientation_3']
        # 1d Histograms
        n_crystal_features = 12
        colors = ['red', 'blue']
        fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
        for i in range(n_crystal_features):
            row = i // 3 + 1
            col = i % 3 + 1

            fig.add_trace(go.Violin(
                x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
                meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
                line_color=colors[0],
            ),
                row=row, col=col
            )

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
        fig.show()


def plot_reconstruction_agreement(cp1, cp2):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=4, cols=3)

    for ind in range(12):
        row = ind // 3 + 1
        col = ind % 3 + 1
        fig.add_scatter(x=cp1[:, ind], y=cp2[:, ind], row=row, col=col, mode='markers')
        fig.add_scatter(x=[min(torch.amin(cp1[:, ind]), torch.amin(cp2[:, ind])),
                           max(torch.amax(cp1[:, ind]), torch.amax(cp2[:, ind]))],
                        y=[min(torch.amin(cp1[:, ind]), torch.amin(cp2[:, ind])),
                           max(torch.amax(cp1[:, ind]), torch.amax(cp2[:, ind]))],
                        row=row, col=col,
                        marker_color='black')
    fig.show()


""" 
Load up test dataset, and mix in random crystals
"""
dataset = collate_data_list(torch.load(r'D:\crystal_datasets\test_CSD_dataset.pt '))

sg_inds = dataset.sg_ind
mol_radii = dataset.radius

# we want to keep orientations and densities, but resample new niggli-compliant cells
cell_params = dataset.cell_parameters()
aunit_centroids = dataset.scale_centroid_to_aunit()

dataset.sample_random_reduced_crystal_parameters(
    target_packing_coeff=dataset.packing_coeff
)
cell_params[:, :6] = dataset.cell_parameters()[:, :6]

latents = dataset.cell_params_to_gen_basis()
dataset.gen_basis_to_cell_params(latents)

# acute niggli cells in reasonable SGs only, and with correctly placed aunits
good_inds = (sg_inds == 2) * (cell_params[:, 3:6] <= torch.pi / 2).all(dim=1) * (aunit_centroids <= 1).all(dim=1)
sg_inds = sg_inds[good_inds]
mol_radii = mol_radii[good_inds]
cell_params = cell_params[good_inds]

asym_unit_dict = ASYM_UNITS.copy()
sgs_to_tensorize = asym_unit_dict.keys()
for key in sgs_to_tensorize:
    asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key])

"""
Transform steps
1) normalize raw cell parameters to aunits
2) niggli-normalize the box parameters
3) transform to std-normal space
4) transform to uniform space
5) denormalize to physical space
6) denormalize aunits

finally for actual training we need to bound this thing in a reasonable range
7) softly bound all outputs
"""

aunit_transform = AunitTransform(asym_unit_dict=asym_unit_dict)
niggli_transform = NiggliTransform()
std_transform = StdNormalTransform()
global_bound_transform = SquashingTransform(min_val=-6, max_val=6)
latent_transform = CompositeTransform([
    AunitTransform(asym_unit_dict=asym_unit_dict),
    NiggliTransform(),
    StdNormalTransform(),
    SquashingTransform(min_val=-6, max_val=6),
])

"""get cell parameters from randn noise"""
noise = torch.randn((len(sg_inds), 12)) * 2
cells = latent_transform.inverse(noise, sg_inds, mol_radii)
std_cell_params = latent_transform(cells, sg_inds, mol_radii)

lattice_features = ['cell_a', 'cell_b', 'cell_c',
                    'cell_alpha', 'cell_beta', 'cell_gamma',
                    'aunit_x', 'aunit_y', 'aunit_z',
                    'orientation_1', 'orientation_2', 'orientation_3']
# 1d Histograms
n_crystal_features = 12
colors = ['red', 'blue']
fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
for i in range(n_crystal_features):
    row = i // 3 + 1
    col = i % 3 + 1

    samples = cells.clone()
    fig.add_trace(go.Violin(
        x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
        meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
        name='random crystals', legendgroup='random crystals', showlegend=i == 0,
        line_color=colors[0],
    ),
        row=row, col=col
    )
    samples = cell_params.clone()
    fig.add_trace(go.Violin(
        x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
        meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
        name='good crystals', legendgroup='good crystals', showlegend=i == 0,
        line_color=colors[1],
    ),
        row=row, col=col
    )

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
fig.show()

n_crystal_features = 12
colors = ['red', 'blue']
fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
for i in range(n_crystal_features):
    row = i // 3 + 1
    col = i % 3 + 1

    samples = torch.nan_to_num(noise.clone())
    fig.add_trace(go.Violin(
        x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
        meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
        name='random crystals', legendgroup='random crystals', showlegend=i == 0,
        line_color=colors[0],
    ),
        row=row, col=col
    )
    samples = torch.nan_to_num(std_cell_params.clone())
    fig.add_trace(go.Violin(
        x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
        meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
        name='good crystals', legendgroup='good crystals', showlegend=i == 0,
        line_color=colors[1],
    ),
        row=row, col=col
    )

fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
fig.show()


aunit_params = aunit_transform.forward(cell_params, sg_inds)
cell_params_re = aunit_transform.inverse(aunit_params, sg_inds)

niggli_params = niggli_transform.forward(aunit_params, mol_radii)
aunit_params_re = niggli_transform.inverse(niggli_params, mol_radii)

std_params = std_transform.forward(niggli_params)
niggli_params_re = std_transform.inverse(std_params)

bounded_std_params = global_bound_transform.forward(std_params)
std_params_re = global_bound_transform.inverse(bounded_std_params)

bounded_std_params2 = global_bound_transform.inverse(std_params)
niggli_params2 = std_transform.inverse(std_params)
aunit_params2 = niggli_transform.inverse(niggli_params, mol_radii)
cell_params2 = aunit_transform.inverse(aunit_params, sg_inds)

"""composite transform"""
bounded_std_params3 = latent_transform.forward(cell_params, sg_inds, mol_radii)
cell_params3 = latent_transform.inverse(bounded_std_params3, sg_inds, mol_radii)
assert torch.isclose(cell_params, cell_params3, atol=1e-4).all()
assert torch.isclose(bounded_std_params, bounded_std_params3).all()

"""stepwise reversal assertions"""
assert torch.isclose(cell_params, cell_params_re).all()
assert torch.isclose(aunit_params, aunit_params_re).all()
assert torch.isclose(niggli_params, niggli_params_re, atol=1e-4).all()
#assert torch.isclose(std_params, std_params_re, atol=1e-4).all()

"""global reversal assertions"""
assert torch.isclose(cell_params, cell_params2).all()
assert torch.isclose(aunit_params, aunit_params2).all()
assert torch.isclose(niggli_params, niggli_params2, atol=1e-4).all()
#assert torch.isclose(std_params, std_params2, atol=1e-4).all()

plot_all_dists([cell_params, aunit_params, niggli_params, std_params, bounded_std_params])

#plot_reconstruction_agreement(cell_params, cell_params2)


aa = 1

# OLD METHODS
# """
# enforce this cycle is reversible
# std normal --> niggli --> std normal
# """
# "forward operation"
# a, al, b, be, c, ga = randn_to_niggli_box_vectors(
#     asym_unit_dict,
#     mol_radii,
#     cell_params,
#     sg_inds
# )
# aunit_orientation = std_normal_to_aunit_orientations(cell_params[:, 9:])
# aunit_scaled_centroids = std_normal_to_uniform(cell_params[:, 6:9])
# aunit_centroid = aunit_scaled_centroids * torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])
#
# "reverse operation"
# a_s, al_s, b_s, be_s, c_s, ga_s = niggli_box_vectors_to_randn(
#     asym_unit_dict,
#     mol_radii,
#     a, b, c, al, be, ga,
#     sg_inds
# )
#
# aunit_scaled_centroids = aunit_centroid / torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])
# std_aunit = uniform_to_std_normal(aunit_scaled_centroids)
# std_orientation = aunit_orientations_to_std_normal(aunit_orientation)
#
# new_cell_params = torch.stack([a_s, b_s, c_s, al_s, be_s, ga_s]).T
# new_cell_params = torch.cat([new_cell_params, std_aunit, std_orientation], dim=1)
#
# print((cell_params - new_cell_params).abs().amax(0))
# print((cell_params - new_cell_params).abs().mean(0))
#
#
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
#
# fig = make_subplots(rows=4, cols=3)
# for ind in range(12):
#     row = ind // 3 + 1
#     col = ind % 3 + 1
#     fig.add_scatter(x=cell_params[:, ind], y=new_cell_params[:, ind], row=row,col=col, mode='markers')
#     fig.add_scatter(x=[min(torch.amin(cell_params[:, ind]), torch.amin(new_cell_params[:, ind])),
#                        max(torch.amax(cell_params[:, ind]), torch.amax(new_cell_params[:, ind]))],
#                     y=[min(torch.amin(cell_params[:, ind]), torch.amin(new_cell_params[:, ind])),
#                        max(torch.amax(cell_params[:, ind]), torch.amax(new_cell_params[:, ind]))],
#                     row=row, col=col,
#                     marker_color='black')
# fig.show()
#
# fig = make_subplots(rows=4, cols=3)
# for ind in range(12):
#     row = ind // 3 + 1
#     col = ind % 3 + 1
#     fig.add_histogram(x=cell_params[:, ind], showlegend=False, row=row, col=col, nbinsx=100, marker_color='red')
#     fig.add_histogram(x=new_cell_params[:, ind], showlegend=False, row=row, col=col, nbinsx=100, marker_color='blue')
# fig.show()
#
#
#
# assert torch.all(torch.isclose(cell_params, new_cell_params))

"""
dataset cell parameter analysis
"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from torch_geometric.loader.dataloader import Collater
from tqdm import tqdm

from mxtaltools.common.geometry_calculations import rotvec2sph
from mxtaltools.common.utils import torch_ptp
from mxtaltools.constants.asymmetric_units import raw_asym_unit_dict
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.constants.space_group_info import LATTICE_TYPE

# dataset_path = Path('D:/crystal_datasets/CSD_dataset.pt')
# dataset = torch.load(dataset_path)
# dataset = dataset.to_data_list()

dataset_path = Path('D:/crystal_datasets/test_CSD_dataset.pt')
dataset = torch.load(dataset_path)


def prep_good_dataset():
    global dataset, vdw_radii_tensor, collater
    # filter Z'=1
    good_dataset = [elem for elem in dataset if elem.z_prime == 1]
    print(f"Z' filter took {len(dataset) - len(good_dataset)}")
    dataset = good_dataset
    # filter diffuse
    vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()))
    red_vol = torch.tensor([elem.reduced_volume for elem in dataset])
    atom_volumes = torch.tensor(
        [torch.sum(4 / 3 * torch.pi * vdw_radii_tensor[elem.x] ** 3) for elem in dataset])
    targets = red_vol / atom_volumes
    good_dataset = [elem for i, elem in enumerate(dataset) if 0.7 < targets[i] < 1.2]
    print(f"diffuse filter took {len(dataset) - len(good_dataset)}")
    dataset = good_dataset
    atom_volumes = torch.tensor(
        [torch.sum(4 / 3 * torch.pi * vdw_radii_tensor[elem.x] ** 3) for elem in dataset])
    collater = Collater(0, 0)
    dataset = collater(dataset)
    del good_dataset
    return atom_volumes


atom_volumes = prep_good_dataset()


def prep_auvs():
    global i, sg
    # prep asymmetric unit parameterization
    asym_unit_vectors = torch.zeros((len(dataset), 3), dtype=torch.float32)
    for i in tqdm(range(len(dataset))):
        if not dataset.nonstandard_symmetry[i]:
            vectors = dataset.cell_lengths[i]
            sg = dataset.sg_ind[i]
            try:
                aunit = raw_asym_unit_dict[str(int(sg))]
                asym_unit_vectors[i] = vectors * torch.Tensor(aunit)
            except KeyError:
                pass


prep_auvs()

"""
Analyze cell lengths
"""
# good_inds = torch.argwhere(asym_unit_vectors[:, 0] != 0).flatten()
# normed_auvecs = asym_unit_vectors / atom_volumes[:, None]
#
# fig = go.Figure()
# for i in range(3):
#     fig.add_trace(go.Violin(x=normed_auvecs[good_inds, i], side='positive', orientation='h',
#                             bandwidth=float(torch.nan_to_num(normed_auvecs).amax()) / 500, width=1))
# fig.show()

sgs = ['P-1', 'P21/c', 'P21', 'C2/c', 'P212121']
sg_inds = [2, 14, 4, 15, 19]
fig = make_subplots(rows=1, cols=6, subplot_titles=sgs)
normed_auvecs = dataset.cell_lengths / dataset.radius[:, None] / torch.pow(dataset.sym_mult[:, None], 1 / 3)

length_stds = normed_auvecs.std(0)
length_means = normed_auvecs.mean(0)
lengths_dist = normed_auvecs

maxval = normed_auvecs.amax()
for sgi, sg in enumerate(sgs):

    for i in range(3):
        good_inds = torch.argwhere(dataset.sg_ind == sg_inds[sgi]).flatten()

        fig.add_trace(go.Violin(x=normed_auvecs[good_inds, i], side='positive', orientation='h',
                                bandwidth=float(torch.nan_to_num(normed_auvecs).amax()) / 200, width=1), row=1,
                      col=sgi + 1)

for i in range(3):
    fig.add_trace(go.Violin(x=normed_auvecs[:, i],y=np.ones(len(normed_auvecs)) * i, side='positive', orientation='h',
                            bandwidth=float(torch.nan_to_num(normed_auvecs).amax()) / 200, width=1), row=1,
                  col=len(sgs) + 1)
    dist = torch.distributions.log_normal.LogNormal(np.log(length_means[i]), length_stds[i])

    sample = dist.sample((10000,)).flatten()
    fig.add_trace(go.Violin(x=sample, y=np.ones(len(sample)) * i, side='positive', orientation='h',
                            bandwidth=float(torch.nan_to_num(normed_auvecs).amax()) / 200, width=1), row=1,
                  col=len(sgs) + 1)

fig.update_xaxes(range=[0, maxval])
fig.show()

sgs, counts = torch.unique(dataset.sg_ind, return_counts=True)
sg_list = [int(sgs[i]) for i in range(len(sgs)) if counts[i] > 50]
means = []
fig = go.Figure()
for sgi, sg in enumerate(sg_list):
    good_inds = torch.argwhere(dataset.sg_ind == sg_list[sgi]).flatten()
    #means.append(float(torch.mean(dataset.reduced_volume[good_inds] / atom_volumes[good_inds])))
    norms = torch.prod(normed_auvecs[good_inds], dim=1)
    means.append(float(torch.mean(norms)))
    fig.add_trace(go.Violin(x=norms,
                            side='positive',
                            orientation='h',
                            width=4,
                            name=sg_list[sgi],
                            bandwidth=float(torch.prod(normed_auvecs, dim=1).amax()) / 300))

fig.show()

# alternate featurization, enforced positive, with cross covariance

sgs = ['P-1', 'P21/c', 'P21', 'C2/c', 'P212121']
sg_inds = [2, 14, 4, 15, 19]
fig = go.Figure()
normed_auvecs = dataset.cell_lengths / dataset.radius[:, None] / torch.pow(dataset.sym_mult[:, None], 1 / 3)

length_stds = normed_auvecs.std(0)
length_means = normed_auvecs.mean(0)
lengths_dist = normed_auvecs

shift = lambda x: F.softplus(x - 0.01, beta=5) + 0.01
s_normed_auvecs = shift(normed_auvecs)

s_length_stds = s_normed_auvecs.std(0)
s_length_means = s_normed_auvecs.mean(0)
s_lengths_dist = s_normed_auvecs

maxval = normed_auvecs.amax()

dist = torch.distributions.multivariate_normal.MultivariateNormal(length_means, torch.cov(normed_auvecs.T))
sample = shift(dist.sample((10000,)))

for i in range(3):
    fig.add_trace(
        go.Violin(x=normed_auvecs[:, i], y=np.ones(len(normed_auvecs)) * i, side='positive', orientation='h',
                  bandwidth=float(torch.nan_to_num(normed_auvecs).amax()) / 200, width=1))

    fig.add_trace(
        go.Violin(x=s_normed_auvecs[:, i], y=np.ones(len(normed_auvecs)) * i, side='positive', orientation='h',
                  bandwidth=float(torch.nan_to_num(normed_auvecs).amax()) / 200, width=1))

    fig.add_trace(go.Violin(x=sample[:, i], y=np.ones(len(sample)) * i, side='positive', orientation='h',
                            bandwidth=float(torch.nan_to_num(normed_auvecs).amax()) / 200, width=1))

fig.update_xaxes(range=[0, maxval])
fig.show()

fig = go.Figure(go.Bar(x=sg_list, y=means)).show()

# reduced auv calculation
# ((normed_auvecs * dataset.radius[:, None] * torch.pow(dataset.sym_mult[:, None], 1/3)).prod(dim=1)/dataset.sym_mult/atom_volumes).mean()

"""
we'll go with cell_length / radius / Z^(1/3) OR cell_length / (atomwise_volume * Z)^(1/3)
these are roughly equivalent functions - 
with radius it's:
normed_auvecs.std(0)
tensor([0.5163, 0.5930, 0.6284])
normed_auvecs.mean(0)
tensor([1.2740, 1.4319, 1.7752])
"""

lattices = [LATTICE_TYPE[int(dataset.sg_ind[ind])] for ind in range(len(dataset))]
css = np.unique(lattices)
fig = make_subplots(rows=1, cols=len(css) + 1, subplot_titles=css)
angles = dataset.cell_angles
maxval = angles.amax()
angles_stats = {}
for csi, cs in enumerate(css):
    good_inds = np.argwhere([lattice == cs for lattice in lattices]).flatten()

    for i in range(3):
        fig.add_trace(go.Violin(x=angles[good_inds, i], side='positive', orientation='h',
                                bandwidth=float(torch.nan_to_num(angles).amax()) / 200, width=1), row=1,
                      col=csi + 1)

    angles_stats[cs] = [angles[good_inds, :].mean(0), angles[good_inds, :].std(0)]

for i in range(3):
    good_inds = torch.argwhere(angles[:, 2] != torch.pi / 2).flatten()

    fig.add_trace(go.Violin(x=angles[good_inds, i], side='positive', orientation='h',
                            bandwidth=float(torch.nan_to_num(angles).amax()) / 200, width=1), row=1,
                  col=len(css) + 1)
fig.update_xaxes(range=[0, maxval])
fig.show()

angles_stds = angles.std(0)
angles_means = angles.mean(0)
angles_dist = angles

print(angles_stats)
"""
Angles basically follow crystal system. Almost always beta >=90 for monoclinic.
Alpha and gamma are more blended outside monoclinic.
Priors: 90/90/90, one sided normal, plain normal
"""

sgs = ['P-1', 'P21/c', 'P21', 'C2/c', 'P212121']
sg_inds = [2, 14, 4, 15, 19]
fig = make_subplots(rows=1, cols=6, subplot_titles=sgs)
positions = dataset.pose_params0[:, :3].clone()
for ind in range(len(positions)):
    try:
        positions[ind] = positions[ind] / torch.Tensor(raw_asym_unit_dict[str(int(dataset.sg_ind[ind]))])
    except:
        pass
maxval = positions.amax()
for sgi, sg in enumerate(sgs):

    for i in range(3):
        good_inds = torch.argwhere((dataset.sg_ind == sg_inds[sgi])).flatten()

        fig.add_trace(go.Violin(x=positions[good_inds, i], side='positive', orientation='h',
                                bandwidth=float(torch.nan_to_num(positions).amax()) / 200, width=1), row=1,
                      col=sgi + 1)

for i in range(3):
    good_inds = torch.argwhere(positions[:, 2] != torch.pi / 2).flatten()

    fig.add_trace(go.Violin(x=positions[good_inds, i], side='positive', orientation='h',
                            bandwidth=float(torch.nan_to_num(positions).amax()) / 500, width=1), row=1,
                  col=len(sgs) + 1)
fig.update_xaxes(range=[0, 1])
fig.show()

pos_stds = positions.std(0)
pos_means = positions.mean(0)
pos_dist = positions

"""
fractional positions are highly textured and variable between space groups
Typical points multiples of 1/8, 1/4, 1/2 with higher order in higher symmetry groups
The best prior is probably uniform
"""

random_vectors = torch.randn(size=(10000, 3))
norms = random_vectors.norm(dim=1)
applied_norms = torch.rand(10000) * 2 * torch.pi
random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]
random_vectors[:, 2] = torch.abs(random_vectors[:, 2])
sph_vectors = rotvec2sph(random_vectors)

fig = go.Figure()
for i in range(3):
    fig.add_trace(go.Violin(x=sph_vectors[:, i], side='positive', orientation='h', width=4,
                            bandwidth=4 * np.pi / 500, )
                  )
fig.update_layout(xaxis_range=[-2.5 * np.pi, 2.5 * np.pi])
fig.show()

"""
positive z-axis only so theta on 0->pi/2
uniform norms 0->2*pi
uniform azimuth -pi->pi
"""

fig = make_subplots(rows=3, cols=3)
randns = torch.randn(10000)

for row_i in range(3):
    dist = randns * length_stds[row_i] + length_means[row_i]
    fig.add_trace(go.Violin(x=dist, side='positive', orientation='h', width=4,
                            bandwidth=float(torch_ptp(dist) / 200)),
                  row=row_i + 1, col=1)
    fig.add_trace(go.Violin(x=lengths_dist[:, row_i], side='positive', orientation='h', width=4,
                            bandwidth=float(torch_ptp(dist) / 200)),
                  row=row_i + 1, col=1)

for row_i in range(3):
    dist = randns * angles_stds[row_i] + angles_means[row_i]
    fig.add_trace(go.Violin(x=dist, side='positive', orientation='h', width=4,
                            bandwidth=float(torch_ptp(dist) / 200)),
                  row=row_i + 1, col=2)
    fig.add_trace(go.Violin(x=angles_dist[:, row_i], side='positive', orientation='h', width=4,
                            bandwidth=float(torch_ptp(dist) / 200)),
                  row=row_i + 1, col=2)

for row_i in range(3):
    dist = torch.rand(10000)
    fig.add_trace(go.Violin(x=dist, side='positive', orientation='h', width=4,
                            bandwidth=float(torch_ptp(dist) / 200)),
                  row=row_i + 1, col=3)
    fig.add_trace(go.Violin(x=pos_dist[:, row_i].clip(max=1), side='positive', orientation='h', width=4,
                            bandwidth=float(torch_ptp(dist) / 200)),
                  row=row_i + 1, col=3)

fig.show()

aa = 1

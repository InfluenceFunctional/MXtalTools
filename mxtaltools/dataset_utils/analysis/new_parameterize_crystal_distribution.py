"""
dataset cell parameter analysis
this time, do the box vectors in Type I Niggli-reduced basis
Dataset
"""

from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import plotly.graph_objects as go

from mxtaltools.common.geometry_utils import cell_vol_angle_factor
from mxtaltools.common.utils import softplus_shift, sample_triangular_right
from mxtaltools.constants.asymmetric_units import RAW_ASYM_UNITS, ASYM_UNITS
from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.models.utils import enforce_1d_bound

if __name__ == '__main__':
    dataset_path = Path('D:/crystal_datasets/test_reduced_CSD_dataset.pt')
    dataset = torch.load(dataset_path)
    dataset = [elem for elem in dataset if 0.45 < elem.packing_coeff < 0.95]
    usable_aunits = list(RAW_ASYM_UNITS.keys())
    dataset = [elem for elem in dataset if str(int(elem.sg_ind)) in usable_aunits]

    """
    Box Vectors
    1) compute Metric tensor components
    2) confirm general niggli conditions
    C>=B
    B>=A
    xi<=B --> cos(alpha) <= b/2c (necessarily less than 1/2)
    eta<=A --> cos(beta) <= a/2b (necessarily less than 1/2)
    zeta<=A --> cos(gamma) <= a/2c (necessarily less than 1/2)   
    """
    mol_radii = torch.tensor([elem.radius for elem in dataset])
    sg_inds = torch.tensor([elem.sg_ind for elem in dataset])
    lattices = [LATTICE_TYPE[int(sample.sg_ind)] for sample in dataset]
    css = np.unique(lattices)
    al, be, ga = torch.cat([elem.cell_angles for elem in dataset], dim=0).split(1, dim=1)
    a, b, c = torch.cat([elem.cell_lengths for elem in dataset], dim=0).split(1, dim=1)
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()
    al = al.flatten()
    be = be.flatten()
    ga = ga.flatten()

    # metric tensor components
    A, B, C = a ** 2, b ** 2, c ** 2
    xi, eta, zeta = 2 * b * c * torch.cos(al), 2 * c * a * torch.cos(be), 2 * a * b * torch.cos(ga)
    # niggli conditions
    print("Checking Main Niggli Conditions")
    print(torch.sum(B <= C) / len(C))
    print(torch.sum(A <= B) / len(C))
    print(torch.sum(torch.cos(al) <= b / 2 / c) / len(C))
    print(torch.sum(torch.cos(be) <= a / 2 / c) / len(C))
    print(torch.sum(torch.cos(ga) <= a / 2 / b) / len(C))
    # parameterize & sample
    """
    Lengths - sample C in the diameter basis, then B as fraction of C, A as fraction of B
    """
    aunit_lengths = torch.cat([elem.scale_lengths_to_aunit() for elem in dataset], dim=0)
    auc = aunit_lengths[:, 2]
    auc_normed = auc / 2 / mol_radii
    c_normed = c / 2 / mol_radii
    log_c = torch.log(auc_normed)
    log_mean = log_c.mean()
    log_std = log_c.std()

    # a and b are right-triangle distributed from 0.2 to 1
    b_normed = (b / c)
    a_normed = (a / b)

    # synthetic samples
    asym_unit_dict = ASYM_UNITS.copy()
    sgs_to_tensorize = asym_unit_dict.keys()
    for key in sgs_to_tensorize:
        asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key])

    # sample c in the asymmetric unit basis for even better normalization
    auc_sampler = torch.distributions.LogNormal(log_c.mean(),
                                                log_c.std() / 2)  # factor of 2 here gets better denormed variance
    normed_auc_samples = auc_sampler.sample((len(dataset),))
    normed_c_samples = normed_auc_samples / torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])[:, 2]
    normed_a_samples = sample_triangular_right(len(dataset), 0.2, 1)
    normed_b_samples = sample_triangular_right(len(dataset), 0.2, 1)

    fig = go.Figure(go.Histogram(x=c, marker_color='red', nbinsx=100))
    fig.add_histogram(x=(b), marker_color='blue', nbinsx=100)
    fig.add_histogram(x=a, marker_color='green', nbinsx=100)
    c_samples = normed_c_samples * 2 * mol_radii
    fig.add_histogram(x=c_samples, marker_color='brown', nbinsx=100)
    fig.add_histogram(x=normed_b_samples * c_samples, marker_color='black', nbinsx=100)
    fig.add_histogram(x=normed_a_samples * normed_b_samples * c_samples, marker_color='magenta', nbinsx=100)
    fig.show()

    """
    Type I Niggli - all acute angles.
    Parameterize as fraction of avaiable range of cosines [0, a/2b], and so on
    Turns out the distribution over the maximum fractional value is pretty damn uniform
    """
    cos_al, cos_be, cos_ga = al.cos(), be.cos(), ga.cos()
    al_max = (b/2/c)
    be_max = (a/2/c)
    ga_max = (a/2/b)
    al_frac = (cos_al/al_max)[al < torch.pi/2]
    be_frac = (cos_be/be_max)[be < torch.pi/2]
    ga_frac = (cos_ga/ga_max)[ga < torch.pi/2]

    fig = go.Figure()
    fig.add_histogram(x=al_frac, nbinsx=100)
    fig.add_histogram(x=be_frac)
    fig.add_histogram(x=ga_frac)
    fig.show()

    """
    Niggli-enforcement pipeline for random inputs
    Imagine we are sampling from a gaussian distributed, standardized basis
    """

    rands = torch.randn((len(dataset), 6))
    # 1) destandardize, denormalize
    # normed aunit c lengths are lognormal distributed
    # we can help the representation by manually applying that here
    auc_normed = torch.exp((log_mean + log_std * rands[:, 0]).clip(max=np.log(4))) # hard clip at 4x the diameter, applied before exp for stability
    # then, convert to the absolute unit cell c, still normalized
    c_normed = auc_normed / torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])[:, 2]
    # then, denormalize
    c_denormed = c_normed * 2 * mol_radii

    # a and b are simple fractions of c, right-triangle distributed
    # sigmoid of N+1 looks kindof similar
    b_scale = F.sigmoid(rands[:, 1] + 1)
    a_scale = F.sigmoid(rands[:, 2] + 1)

    c_out = c_denormed
    b_out = b_scale * c_out
    a_out = a_scale * b_out

    # We sample the cosines of the cell angles, for a Type I (acute) Niggli cell
    # as a function of their maximum possible values
    # this is automatically a positive value within the required range
    cos_al_out_max = (b_out / 2 / c_out)
    cos_be_out_max = (a_out / 2 / c_out)
    cos_ga_out_max = (a_out / 2 / b_out)
    al_cos_out = F.sigmoid(rands[:, 3]*2) * cos_al_out_max
    be_cos_out = F.sigmoid(rands[:, 4]*2) * cos_be_out_max
    ga_cos_out = F.sigmoid(rands[:, 5]*2) * cos_ga_out_max
    al_out = torch.arccos(al_cos_out)
    be_out = torch.arccos(be_cos_out)
    ga_out = torch.arccos(ga_cos_out)

    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=3)
    for ind, (c1, c2) in enumerate(zip(
            [a, b, c, al[al<torch.pi/2], be[be<torch.pi/2], ga[ga<torch.pi/2]], [a_out, b_out, c_out, al_out, be_out, ga_out]

    )):
        row = ind // 3 + 1
        col = ind % 3 + 1
        fig.add_histogram(x=c1.flatten(), nbinsx=100, row=row, col=col, marker_color='red', )
        fig.add_histogram(x=c2.flatten(), nbinsx=100, row=row, col=col, marker_color='blue')

    fig.show()

    # """
    # fractional positions (we take this as a uniform distribution)
    # (this is exactly the same as the pre-Niggli reduced version)
    # """
    #
    # fractional_centroids = torch.zeros((len(dataset), 3))
    # for ind in range(len(fractional_centroids)):
    #     fractional_centroids[ind] = dataset[ind].scale_centroid_to_aunit()
    # # todo, this inspires another crystal filter - for some very nonstandard objects,
    # #  the fractional centroids won't be in the asym unit - filter these
    #
    # pos_stds = fractional_centroids.std(0)
    # pos_means = fractional_centroids.mean(0)
    #
    # # then we can resample it like so
    # random_centroids = torch.rand((len(dataset), 3))
    #
    # 'mol orientations (these are random)'
    # random_vectors = torch.randn(size=(10000, 3))
    # norms = random_vectors.norm(dim=1)
    # aunit_orientations = torch.cat([elem.aunit_orientation for elem in dataset], dim=0)
    # norm_std = aunit_orientations.norm(dim=1).std()
    #
    # # then we can resample it like so
    # applied_norms = (torch.randn(10000) * norm_std + torch.pi).clip(min=-2 * torch.pi + 0.1,
    #                                                                 max=2 * torch.pi - 0.1)  # the CSD rotation norms are gaussian-distributed, not uniform
    # random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]
    # random_vectors[:, 2] = torch.abs(random_vectors[:, 2])
    #
    # # compare sampled distributions to baseline
    # import plotly.graph_objects as go
    #
    # fig = go.Figure()
    # for ind in range(3):
    #     fig.add_histogram(x=normed_lengths[:, ind], nbinsx=100, marker_color='blue')
    #     fig.add_histogram(x=random_normed_aunit_lengths[:, ind], nbinsx=100, marker_color='red')
    # fig.update_layout(title='Normed Aunit Lengths')
    # fig.show()
    #
    # fig = go.Figure()
    # fig.add_histogram(x=packing_coeffs, nbinsx=100, marker_color='blue')
    # fig.add_histogram(x=random_packing_coeffs, nbinsx=100, marker_color='red')
    # fig.update_layout(title='Packing Coefficients')
    # fig.show()
    #
    # fig = go.Figure()
    # for ind in range(3):
    #     fig.add_histogram(x=aunit_lengths[:, ind], nbinsx=100, marker_color='blue')
    #     fig.add_histogram(x=random_aunit_lengths[:, ind], nbinsx=100, marker_color='red')
    # fig.update_layout(title='Aunit Lengths')
    # fig.show()
    #
    # fig = go.Figure()
    # for ind in range(3):
    #     fig.add_histogram(x=cell_lengths[:, ind], nbinsx=100, marker_color='blue')
    #     fig.add_histogram(x=random_cell_lengths[:, ind], nbinsx=100, marker_color='red')
    # fig.update_layout(title='Cell Lengths')
    # fig.show()
    #
    # fig = go.Figure()
    # for ind in range(3):
    #     fig.add_histogram(x=angles[:, ind], nbinsx=100, marker_color='blue')
    #     fig.add_histogram(x=random_angles[:, ind], nbinsx=100, marker_color='red')
    # fig.update_layout(title='Cell Angles')
    # fig.show()
    #
    # fig = go.Figure()
    # for ind in range(3):
    #     fig.add_histogram(x=fractional_centroids[:, ind], nbinsx=100, marker_color='blue')
    #     fig.add_histogram(x=random_centroids[:, ind], nbinsx=100, marker_color='red')
    # fig.update_layout(title='Aunit Centroids')
    # fig.show()
    #
    # fig = go.Figure()
    # fig.add_histogram(x=aunit_orientations.norm(dim=1), nbinsx=100)
    # fig.add_histogram(x=random_vectors.norm(dim=1), nbinsx=100)
    # fig.update_layout(title='Aunit Orientation Vector Norms')
    # fig.show()
    #
    # fig = go.Figure()
    # for ind in range(3):
    #     fig.add_histogram(x=aunit_orientations[:, ind], nbinsx=100, marker_color='blue')
    #     fig.add_histogram(x=random_vectors[:, ind], nbinsx=100, marker_color='red')
    # fig.update_layout(title='Aunit Orientation Vectors')
    # fig.show()
    #
    # finished = 1

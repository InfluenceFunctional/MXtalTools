"""
dataset cell parameter analysis
"""

from pathlib import Path

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from mxtaltools.common.geometry_utils import cell_vol_angle_factor
from mxtaltools.common.utils import softplus_shift
from mxtaltools.constants.asymmetric_units import RAW_ASYM_UNITS
from mxtaltools.constants.space_group_info import LATTICE_TYPE

if __name__ == '__main__':
    dataset_path = Path('D:/crystal_datasets/test_reduced_CSD_dataset.pt')
    dataset = torch.load(dataset_path)
    dataset = [elem for elem in dataset if 0.45 < elem.packing_coeff < 0.95]
    usable_aunits = list(RAW_ASYM_UNITS.keys())
    dataset = [elem for elem in dataset if str(int(elem.sg_ind)) in usable_aunits]

    'cell angles'  # todo evaluate these and make sure they make physical sense
    lattices = [LATTICE_TYPE[int(sample.sg_ind)] for sample in dataset]
    css = np.unique(lattices)
    angles = torch.cat([elem.cell_angles for elem in dataset], dim=0)
    angles_stats = {}
    for csi, cs in enumerate(css):
        good_inds = np.argwhere([lattice == cs for lattice in lattices]).flatten()
        angles_stats[cs] = [angles[good_inds, :].mean(0), angles[good_inds, :].std(0),
                            torch.cov(angles[good_inds, :].T)]

    angles_means = angles_stats['triclinic'][0]
    angles_stds = angles_stats['triclinic'][1]
    angles_cov = angles_stats['triclinic'][2]  # take the stats from triclinic

    print(angles_means)
    print(angles_stds)
    print(angles_stats)

    # we can then sample it like so
    angles_prior = MultivariateNormal(angles_means, angles_cov)
    random_angles = angles_prior.sample((len(dataset),))

    """
    cell lengths
    
    normed lengths are cell_length/prod(cell_length)^(1/3), so that the product of lengths is 1
    we then get the asymmetric unit cell lengths as F*normed_lengths
    with F=V_{mol} / (c_{pack} * c_{ang})
    for c_{ang} the angular component of the unit cell box volume calculation
    """
    cell_lengths = torch.cat([elem.cell_lengths for elem in dataset], dim=0)

    # convert to asymmetric unit lengths
    aunit_lengths = torch.cat([elem.scale_lengths_to_aunit() for elem in dataset], dim=0)
    normed_lengths = aunit_lengths / aunit_lengths.prod(dim=1, keepdim=True) ** (1 / 3)
    normed_length_stds = normed_lengths.std(0)
    normed_length_means = normed_lengths.mean(0)

    shifted_normed_cell_lengths = softplus_shift(normed_lengths)

    # normal distribution parameterization isn't great
    s_length_stds = shifted_normed_cell_lengths.std(0)
    s_length_means = shifted_normed_cell_lengths.mean(0)
    s_cov_mat = torch.cov(shifted_normed_cell_lengths.T)
    print(s_length_means)
    print(s_length_stds)
    print(s_cov_mat)

    # we'll rather try a lognormal
    log_data = torch.log(shifted_normed_cell_lengths)
    log_means = log_data.mean(0)
    log_cov_mat = torch.cov(log_data.T)

    # extract also packing coefficient statistics
    packing_coeffs = torch.tensor([elem.packing_coeff for elem in dataset])
    mol_volumes = torch.tensor([elem.mol_volume for elem in dataset])
    packing_mean = packing_coeffs.mean()
    packing_std = packing_coeffs.std()

    # it can then be sampled like so
    #normed_aunit_lengths_prior = MultivariateNormal(s_length_means, s_cov_mat)
    #random_normed_aunit_lengths = normed_aunit_lengths_prior.sample((len(dataset),))
    normed_aunit_lengths_prior = MultivariateNormal(log_means, log_cov_mat)
    random_normed_aunit_lengths = normed_aunit_lengths_prior.sample((len(dataset),)).exp()

    random_packing_coeffs = (torch.randn(len(dataset)) * packing_std + packing_mean).clip(min=0.4, max=1)
    angle_factors = cell_vol_angle_factor(random_angles).abs()
    volume_factors = mol_volumes / (random_packing_coeffs * angle_factors)

    random_aunit_lengths = random_normed_aunit_lengths * volume_factors[:, None] ** (1 / 3)
    random_cell_lengths = torch.stack(
        [elem.scale_aunit_lengths_to_unit_cell(random_aunit_lengths[ind]) for ind, elem in enumerate(dataset)])

    'fractional positions (we take this as a uniform distribution)'
    fractional_centroids = torch.zeros((len(dataset), 3))
    for ind in range(len(fractional_centroids)):
        fractional_centroids[ind] = dataset[ind].scale_centroid_to_aunit()
    # todo, this inspires another crystal filter - for some very nonstandard objects,
    #  the fractional centroids won't be in the asym unit - filter these

    pos_stds = fractional_centroids.std(0)
    pos_means = fractional_centroids.mean(0)

    # then we can resample it like so
    random_centroids = torch.rand((len(dataset), 3))

    'mol orientations (these are random)'
    random_vectors = torch.randn(size=(10000, 3))
    norms = random_vectors.norm(dim=1)
    aunit_orientations = torch.cat([elem.aunit_orientation for elem in dataset], dim=0)
    norm_std = aunit_orientations.norm(dim=1).std()

    # then we can resample it like so
    applied_norms = (torch.randn(10000) * norm_std + torch.pi).clip(min=-2 * torch.pi + 0.1,
                                                                    max=2 * torch.pi - 0.1)  # the CSD rotation norms are gaussian-distributed, not uniform
    random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]
    random_vectors[:, 2] = torch.abs(random_vectors[:, 2])

    # compare sampled distributions to baseline
    import plotly.graph_objects as go

    fig = go.Figure()
    for ind in range(3):
        fig.add_histogram(x=normed_lengths[:, ind], nbinsx=100, marker_color='blue')
        fig.add_histogram(x=random_normed_aunit_lengths[:, ind], nbinsx=100, marker_color='red')
    fig.update_layout(title='Normed Aunit Lengths')
    fig.show()

    fig = go.Figure()
    fig.add_histogram(x=packing_coeffs, nbinsx=100, marker_color='blue')
    fig.add_histogram(x=random_packing_coeffs, nbinsx=100, marker_color='red')
    fig.update_layout(title='Packing Coefficients')
    fig.show()

    fig = go.Figure()
    for ind in range(3):
        fig.add_histogram(x=aunit_lengths[:, ind], nbinsx=100, marker_color='blue')
        fig.add_histogram(x=random_aunit_lengths[:, ind], nbinsx=100, marker_color='red')
    fig.update_layout(title='Aunit Lengths')
    fig.show()

    fig = go.Figure()
    for ind in range(3):
        fig.add_histogram(x=cell_lengths[:, ind], nbinsx=100, marker_color='blue')
        fig.add_histogram(x=random_cell_lengths[:, ind], nbinsx=100, marker_color='red')
    fig.update_layout(title='Cell Lengths')
    fig.show()

    fig = go.Figure()
    for ind in range(3):
        fig.add_histogram(x=angles[:, ind], nbinsx=100, marker_color='blue')
        fig.add_histogram(x=random_angles[:, ind], nbinsx=100, marker_color='red')
    fig.update_layout(title='Cell Angles')
    fig.show()

    fig = go.Figure()
    for ind in range(3):
        fig.add_histogram(x=fractional_centroids[:, ind], nbinsx=100, marker_color='blue')
        fig.add_histogram(x=random_centroids[:, ind], nbinsx=100, marker_color='red')
    fig.update_layout(title='Aunit Centroids')
    fig.show()

    fig = go.Figure()
    fig.add_histogram(x=aunit_orientations.norm(dim=1), nbinsx=100)
    fig.add_histogram(x=random_vectors.norm(dim=1), nbinsx=100)
    fig.update_layout(title='Aunit Orientation Vector Norms')
    fig.show()

    fig = go.Figure()
    for ind in range(3):
        fig.add_histogram(x=aunit_orientations[:, ind], nbinsx=100, marker_color='blue')
        fig.add_histogram(x=random_vectors[:, ind], nbinsx=100, marker_color='red')
    fig.update_layout(title='Aunit Orientation Vectors')
    fig.show()

    finished = 1

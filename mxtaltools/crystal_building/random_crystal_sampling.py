"""
utilities for the sampling of random crystals according to some underlying statistics
based on analysis in parameterize_crystal_distribution.py
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from mxtaltools.common.geometry_utils import cell_vol_angle_factor, batch_cell_vol_torch
from mxtaltools.constants.asymmetric_units import ASYM_UNITS

# extracted from Z'=1 CSD structures
angles_means = torch.ones(3) * torch.pi / 2
angles_cov = torch.tensor([[0.0547, 0.0325, 0.0379],
                           [0.0325, 0.0431, 0.0345],
                           [0.0379, 0.0345, 0.0675]])
norm_std = 1.1333

# normed_aunit_length_log_means = torch.tensor([-0.0740, -0.4079,  0.4897])
# normed_aunit_length_log_covs = torch.tensor([[ 0.2673, -0.2373, -0.0261],
#         [-0.2373,  0.3543, -0.1232],
#         [-0.0261, -0.1232,  0.1517]])

# we use flattened and uncorrelated length relationships
normed_aunit_length_log_means = torch.tensor([0.0026, 0.0026, 0.0026])
normed_aunit_length_log_covs = torch.eye(3)

packing_coeff_mean = 0.6226
packing_coeff_std = 0.0447


def sample_cell_angles(num_samples):
    """
    returns angles[n, 3], [alpha, beta, gamma]
    normally distributed according to the statistics in the triclinic CSD crystals
    """
    angles_prior = MultivariateNormal(angles_means, angles_cov)
    return angles_prior.sample((num_samples,))


def sample_aunit_centroids(num_samples):
    """
    returns aunit_centroids[n, 3], [x, y, z]
    uniformly distributed on [0,1]
    """
    return torch.rand((num_samples, 3))


def sample_aunit_orientations(num_samples):
    """
    returns aunit_orientations[n, 3], [u, v, w]
    of vectors with random lengths and directions
    """
    # random directions
    random_vectors = torch.randn(size=(num_samples, 3))
    norms = random_vectors.norm(dim=1)

    # random lengths
    applied_norms = (torch.randn(num_samples) * norm_std + torch.pi
                     ).clip(min=-2 * torch.pi,
                            max=2 * torch.pi)  # the CSD rotation norms are gaussian-distributed, not uniform
    random_rotvecs = random_vectors / norms[:, None] * applied_norms[:, None]

    # z direction must be positive
    random_rotvecs[:, 2] = torch.abs(random_rotvecs[:, 2])

    return random_rotvecs


def sample_aunit_lengths(num_samples,
                         cell_angles,
                         mol_volumes,
                         target_packing_coeff: Optional = None,
                         clip_packing_min: float = 0.4,
                         clip_packing_max: float = 1.0,
                         eps: Optional[float] = 1e-5):
    """
    returns unit cell lengths[n, 3], [a, b, c]

    we first sample normed aunit lengths on a cube with unit volume
    then, for a target packing coefficient & given cell angles, convert these to
    asymmetric unit lengths, and then to unit cell lengths

    """
    normed_aunit_lengths_prior = MultivariateNormal(normed_aunit_length_log_means,
                                                    normed_aunit_length_log_covs + torch.eye(3) * eps)
    random_sampled_aunit_lengths = normed_aunit_lengths_prior.sample((num_samples,)).exp()
    random_normed_aunit_lengths = (
                random_sampled_aunit_lengths / random_sampled_aunit_lengths.prod(dim=1, keepdim=True) ** (1 / 3)).to(
        mol_volumes.device)

    if target_packing_coeff is not None:
        if isinstance(target_packing_coeff, float):
            packing_coeffs_to_build = target_packing_coeff * torch.ones(num_samples, dtype=torch.float32,
                                                                      device=cell_angles.device)
        elif torch.is_tensor(target_packing_coeff):
            if len(target_packing_coeff) > 1:
                packing_coeffs_to_build = target_packing_coeff * torch.ones(num_samples, dtype=torch.float32,
                                                                          device=cell_angles.device)
            else:
                packing_coeffs_to_build = target_packing_coeff
        else:
            assert False, 'target_packing_coeff must be a float or tensor'
    else:
        packing_coeffs_to_build = (
                    torch.randn(num_samples, device=mol_volumes.device) * packing_coeff_std + packing_coeff_mean
                    ).clip(min=clip_packing_min, max=clip_packing_max)

    angle_factors = torch.abs(cell_vol_angle_factor(cell_angles))
    volume_factors = mol_volumes / (packing_coeffs_to_build * angle_factors)
    random_aunit_lengths = random_normed_aunit_lengths * volume_factors[:, None] ** (1 / 3)

    return random_aunit_lengths


def sample_reduced_box_vectors(num_graphs,
                               mol_radii,
                               mol_volumes,
                               sg_inds,
                               sym_mult,
                               asym_unit_dict=None,
                               target_packing_coeff: Optional[float] = None):
    """
    Sample type-1 (acute) niggli reduced box paramete from a random prior
    Currently the main conditions are satisfied, not the symmetry breaking special conditions
    """
    if asym_unit_dict is None:
        asym_unit_dict = ASYM_UNITS.copy()
        sgs_to_tensorize = asym_unit_dict.keys()
        for key in sgs_to_tensorize:
            asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key], device=mol_radii.device)

    rands = torch.randn((num_graphs, 6), device=mol_radii.device)

    a_out, al_out, b_out, be_out, c_out, ga_out = nigglify_random_box_vectors(asym_unit_dict,
                                                                              mol_radii, rands, sg_inds)
    cell_lengths = torch.vstack([a_out, b_out, c_out]).T
    cell_angles = torch.vstack([al_out, be_out, ga_out]).T

    if target_packing_coeff is not None:
        vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
        cp1 = mol_volumes * sym_mult / vol1
        correction_ratio = cp1 / target_packing_coeff
        cell_lengths *= (correction_ratio**(1/3))[:, None]

    return cell_lengths, cell_angles


def nigglify_random_box_vectors(asym_unit_dict, mol_radii, rands, sg_inds):
    # lognormal distribution parameters for the asymmetric unit c-vector length in molecule diameter units
    log_mean = 0.24
    log_std = 0.3618
    # 1) destandardize, denormalize
    # normed aunit c lengths are lognormal distributed
    # we can help the representation by manually applying that here
    auc_normed = torch.exp((log_mean + log_std * rands[:, 0]).clip(
        max=np.log(4)))  # clip at 4x the diameter, applied before exp for stability
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
    al_cos_out = F.sigmoid(rands[:, 3] * 2) * cos_al_out_max
    be_cos_out = F.sigmoid(rands[:, 4] * 2) * cos_be_out_max
    ga_cos_out = F.sigmoid(rands[:, 5] * 2) * cos_ga_out_max
    al_out = torch.arccos(al_cos_out)
    be_out = torch.arccos(be_cos_out)
    ga_out = torch.arccos(ga_cos_out)
    return a_out, al_out, b_out, be_out, c_out, ga_out

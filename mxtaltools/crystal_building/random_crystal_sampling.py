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
from mxtaltools.common.utils import softplus_shift, invert_softplus_shift, siginv
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.models.utils import enforce_1d_bound, undo_1d_bound

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

    Can also be done by sampling the cdf of a standard normal
    meaning it can be converted into a std normal type distribution
    by the inverse operation
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
                     ).clip(min=-2 * torch.pi * 0.99,
                            max=2 * torch.pi * 0.99)  # the CSD rotation norms are gaussian-distributed, not uniform
    random_rotvecs = random_vectors / norms[:, None] * applied_norms[:, None]

    # z direction must be positive
    random_rotvecs[:, 2] = torch.abs(random_rotvecs[:, 2])

    return random_rotvecs


def std_normal_to_aunit_orientations(vectors):
    """
    generate randomly distributed random rotation vectors
    from a standard normal distribution

    Identical to sample_aunit_orientations but with the z axis uniquely mapped
    """
    # z direction must be positive - convert from std normal to uniform on 0-2pi
    # this is not ideal, as Z is std normal distributed on positive, but it works and is unique
    fixed_z = torch.distributions.Normal(0, 1).cdf(vectors[:, 2])[:, None] * 2 * torch.pi
    fixed_vectors = torch.cat([vectors[:, :2], fixed_z], dim=1)

    # set vector lengths within relevant limits
    norms = fixed_vectors.norm(dim=1)
    applied_norms = enforce_1d_bound(norms, torch.pi, torch.pi, mode='soft')

    rotvecs = fixed_vectors / norms[:, None] * applied_norms[:, None]

    return rotvecs


def aunit_orientations_to_std_normal(rotvecs):
    """
    inverts the above std_normal to aunit orientations
    """
    eps = 1e-5
    recovered_norms = rotvecs.norm(dim=1).clip(min=eps, max=torch.pi*2-eps)
    original_norms = undo_1d_bound(recovered_norms, torch.pi, torch.pi, mode='soft')

    # Undo the rescaling
    fixed_vectors = rotvecs / recovered_norms[:, None] * original_norms[:, None]

    # Recover the original z before CDF
    z_prime = (fixed_vectors[:, 2]/ 2 / torch.pi).clamp(min=1e-3, max=1 - 1e-3)  # avoid infs in icdf
    z_original = torch.distributions.Normal(0, 1).icdf(z_prime)

    # Reconstruct full original vector
    std_normal_vectors = torch.cat([fixed_vectors[:, :2], z_original[:, None]], dim=1)
    return std_normal_vectors


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
            asym_unit_dict[key] = torch.tensor(asym_unit_dict[key], device=mol_radii.device)

    rands = torch.randn((num_graphs, 6), device=mol_radii.device)

    a_out, al_out, b_out, be_out, c_out, ga_out = randn_to_niggli_box_vectors(asym_unit_dict,
                                                                              mol_radii, rands, sg_inds)
    cell_lengths = torch.vstack([a_out, b_out, c_out]).T
    cell_angles = torch.vstack([al_out, be_out, ga_out]).T

    if target_packing_coeff is not None:  # todo think of a way to do this that doesn't break our cells
        vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
        cp1 = mol_volumes * sym_mult / vol1
        correction_ratio = (cp1 / target_packing_coeff) ** (1 / 3)
        cell_lengths *= correction_ratio[:, None]
        min_c_length = 0.1 * 2 * mol_radii * torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])[:, 2]
        cell_lengths[:, 2] = cell_lengths[:, 2].clip(min=min_c_length)

    return cell_lengths, cell_angles


def randn_to_niggli_box_vectors(asym_unit_dict, mol_radii, rands,
                                sg_inds):  # TODO this could be written more efficiently
    # lognormal distribution parameters for the asymmetric unit c-vector length in molecule diameter units
    log_mean = 0.24
    log_std = 0.3618
    # 1) destandardize, denormalize
    # normed aunit c lengths are lognormal distributed
    # we can help the representation by manually applying that here
    # clip at 4x the diameter, applied before exp for stability
    destd_log = log_mean + log_std * rands[:, 2]
    min_val = 0.1
    max_val = 4
    log_auc = sigmoid_shift(destd_log, np.log(min_val), np.log(max_val))
    auc_normed = torch.exp(log_auc)
    # then, convert to the absolute unit cell c, still normalized
    c_normed = auc_normed / torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])[:, 2]
    # then, denormalize
    c_denormed = c_normed * 2 * mol_radii
    # a and b are simple fractions of c, right-triangle distributed
    # set a hard limit at 0.1 for practical purposes
    # sigmoid of N+1 looks kindof similar
    min_scale = 0.1
    b_scale = min_scale + (1-min_scale) * F.sigmoid(rands[:, 1] + 1)
    a_scale = min_scale + (1-min_scale) * F.sigmoid(rands[:, 0] + 1)
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


def niggli_box_vectors_to_randn(asym_unit_dict, mol_radii, a, b, c, al, be, ga, sg_inds):
    """
    inverts the nigglifying function back to a standardized tensor
    """
    # recover c
    log_mean = 0.24
    log_std = 0.3618
    c_normed = c / 2 / mol_radii
    auc_normed = c_normed * torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])[:, 2]
    log_auc = torch.log(auc_normed)
    min_val = 0.1
    max_val = 4
    eps = 1e-4
    c_lognorm = inverse_sigmoid_shift(log_auc.clip(min=np.log(min_val + eps), max=np.log(max_val - eps)), np.log(min_val), np.log(max_val))
    c_std = (c_lognorm - log_mean) / log_std

    # recover a and b
    min_scale = 0.1
    eps = 1e-3
    a_scale = (a / b).clip(min=min_scale+eps, max=1-eps)
    b_scale = (b / c).clip(min=min_scale+eps, max=1-eps)
    a_std = siginv((a_scale - min_scale)/(1-min_scale)) - 1
    b_std = siginv((b_scale - min_scale)/(1-min_scale)) - 1

    # recover angles
    cos_al_out_max = (b / 2 / c)
    cos_be_out_max = (a / 2 / c)
    cos_ga_out_max = (a / 2 / b)

    al_cos_out = torch.cos(al)
    be_cos_out = torch.cos(be)
    ga_cos_out = torch.cos(ga)
    sigmoid_al_scaled = (al_cos_out / cos_al_out_max).clip(min=eps, max=1-eps)
    sigmoid_be_scaled = (be_cos_out / cos_be_out_max).clip(min=eps, max=1-eps)
    sigmoid_ga_scaled = (ga_cos_out / cos_ga_out_max).clip(min=eps, max=1-eps)
    al_std = 0.5 * siginv(sigmoid_al_scaled)
    be_std = 0.5 * siginv(sigmoid_be_scaled)
    ga_std = 0.5 * siginv(sigmoid_ga_scaled)

    return a_std, al_std, b_std, be_std, c_std, ga_std


def sigmoid_shift(x, x_min, x_max):
    return x_min + (x_max - x_min) * torch.sigmoid(x)

def inverse_sigmoid_shift(x, x_min, x_max):
    return siginv((x - x_min) / (x_max - x_min))

    """
    # test inverse performance
    
    a, al, b, be, c, ga = randn_to_niggli_box_vectors(asym_unit_dict,
                                                                          mol_radii, rands, sg_inds)
    
    a_out, al_out, b_out, be_out, c_out, ga_out = niggli_box_vectors_to_randn(asym_unit_dict,
                                                                              mol_radii, a, b, c, al, be, ga, sg_inds)
    std_cell_lengths = torch.vstack([a_out, b_out, c_out]).T
    std_cell_angles = torch.vstack([al_out, be_out, ga_out]).T
    rebuild_std = torch.cat([std_cell_lengths, std_cell_angles], dim=-1)
    print(torch.mean(torch.abs(rebuild_std - rands)))
    """

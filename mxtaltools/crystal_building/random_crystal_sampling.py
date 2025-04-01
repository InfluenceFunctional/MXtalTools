"""
utilities for the sampling of random crystals according to some underlying statistics
based on analysis in parameterize_crystal_distribution.py
"""
from typing import Optional

import torch
from torch.distributions import MultivariateNormal

from mxtaltools.common.geometry_utils import cell_vol_angle_factor

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
                     ).clip(min=-2 * torch.pi, max=2 * torch.pi)  # the CSD rotation norms are gaussian-distributed, not uniform
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
    random_normed_aunit_lengths = (random_sampled_aunit_lengths / random_sampled_aunit_lengths.prod(dim=1, keepdim=True)**(1/3)).to(mol_volumes.device)

    if target_packing_coeff is not None:
        if isinstance(target_packing_coeff, float):
            random_packing_coeffs = target_packing_coeff * torch.ones(num_samples)
        elif torch.is_tensor(target_packing_coeff):
            if len(target_packing_coeff) > 1:
                random_packing_coeffs = target_packing_coeff * torch.ones(num_samples)
            else:
                random_packing_coeffs = target_packing_coeff
        else:
            assert False, 'target_packing_coeff must be a float or tensor'
    else:
        random_packing_coeffs = (torch.randn(num_samples, device=mol_volumes.device) * packing_coeff_std + packing_coeff_mean
                                 ).clip(min=clip_packing_min, max=clip_packing_max)

    angle_factors = torch.abs(cell_vol_angle_factor(cell_angles))
    volume_factors = mol_volumes / (random_packing_coeffs * angle_factors)

    random_aunit_lengths = random_normed_aunit_lengths * volume_factors[:, None] ** (1/3)

    return random_aunit_lengths

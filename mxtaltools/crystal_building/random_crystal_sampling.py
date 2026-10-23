"""
Utilities for sampling random crystal parameters according to fixed prior statistics.

The statistics are based on the analysis in
``mxtaltools/dataset_utils/analysis/parameterize_crystal_distribution.py`` and
are held in module-level constants: ``angles_means``, ``angles_cov``,
``norm_std``, ``normed_aunit_length_log_means``,
``normed_aunit_length_log_covs``, ``packing_coeff_mean`` and
``packing_coeff_std``. Angles are in radians throughout.
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
    Sample cell angles from a multivariate normal prior.

    The mean is ``angles_means`` (pi / 2 for each angle) and the covariance
    ``angles_cov``, from the statistics of triclinic CSD crystals. No
    clipping is applied.

    Parameters
    ----------
    num_samples : int
        Number of rows.

    Returns
    -------
    torch.Tensor
        [num_samples, 3] angles (alpha, beta, gamma) in radians.
    """
    angles_prior = MultivariateNormal(angles_means, angles_cov)
    return angles_prior.sample((num_samples,))


def sample_aunit_centroids(num_samples, z_prime: int=1):
    """
    Sample asymmetric-unit centroids uniformly on [0, 1).

    Parameters
    ----------
    num_samples : int
        Number of rows.
    z_prime : int, optional
        Number of centroid triples per row. Default 1.

    Returns
    -------
    torch.Tensor
        [num_samples, 3 * z_prime] from ``torch.rand``; each consecutive
        triple is one (x, y, z) centroid.

    Notes
    -----
    The same distribution results from applying the standard normal CDF to
    standard normal samples, so the inverse CDF converts these values to
    standard normal ones.
    """
    return torch.rand((num_samples, 3 * z_prime))


def sample_aunit_orientations(num_samples, z_prime: int=1):
    """
    Sample rotation vectors with random axes and normally distributed angles.

    For each of the ``z_prime`` slots the axis is a normalized standard
    normal 3-vector, and the length is drawn from N(pi, norm_std) and
    clipped to [-0.99 * 2 pi, 0.99 * 2 pi]. The z component of the scaled
    vector is then replaced by its absolute value. A negative length draw
    reverses the axis, so the returned norm is the absolute value of the
    draw.

    Parameters
    ----------
    num_samples : int
        Number of rows.
    z_prime : int, optional
        Number of rotation vectors per row. Default 1.

    Returns
    -------
    torch.Tensor
        [num_samples, 3 * z_prime]; each consecutive triple is one rotation
        vector (norm = rotation angle in radians) with non-negative z.
    """
    # random directions
    rvs = []
    for zp in range(z_prime):
        random_vectors = torch.randn(size=(num_samples, 3))
        norms = random_vectors.norm(dim=1)

        # random lengths
        # this should be changed to the correct Haar measure, proportional to sin^2(theta/2), but I don't know if it has an analytical PDF
        applied_norms = (torch.randn(num_samples) * norm_std + torch.pi
                         ).clip(min=-2 * torch.pi * 0.99,
                                max=2 * torch.pi * 0.99)  # the CSD rotation norms are gaussian-distributed, not uniform
        random_rotvecs = random_vectors / norms[:, None] * applied_norms[:, None]

        # z direction must be positive
        random_rotvecs[:, 2] = torch.abs(random_rotvecs[:, 2])
        rvs.append(random_rotvecs)

    return torch.cat(rvs, dim=1)


def std_normal_to_aunit_orientations(vectors):
    """
    Map standard-normal 3-vectors to rotation vectors with positive z.

    The z column is replaced by ``2 pi * Phi(z)`` (Phi the standard normal
    CDF), giving ``w = (x, y, 2 pi Phi(z))``. The output keeps the direction
    of ``w`` and sets its length to ``tanh((|w| - pi) / pi) * pi + pi``
    (``enforce_1d_bound`` in soft mode), which lies in (0, 2 pi).

    Parameters
    ----------
    vectors : torch.Tensor
        [n, 3]; only the first three columns are read.

    Returns
    -------
    torch.Tensor
        [n, 3] rotation vectors (norm = rotation angle in radians).

    Notes
    -----
    Unlike :func:`sample_aunit_orientations`, the length is a function of
    the input vector's own length rather than an independent draw, and z is
    made positive by the CDF map rather than by an absolute value. Inverted
    by :func:`aunit_orientations_to_std_normal`.
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
    Invert :func:`std_normal_to_aunit_orientations`.

    The norm is clamped to [1e-5, 2 pi - 1e-5] and mapped back with
    ``undo_1d_bound`` (soft mode, span and center pi). The z component of the
    rescaled vector is divided by 2 pi, clamped to [1e-3, 1 - 1e-3] and
    passed through the standard normal inverse CDF.

    Parameters
    ----------
    rotvecs : torch.Tensor
        [n, 3] rotation vectors.

    Returns
    -------
    torch.Tensor
        [n, 3] vectors in the standard-normal basis.
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
    Sample asymmetric-unit lengths for given cell angles and molecular volumes.

    Normed lengths are first sampled with unit product (a unit-volume cube
    before the angle factor): log-normal draws with log-mean
    ``normed_aunit_length_log_means`` and log-covariance
    ``normed_aunit_length_log_covs + eps * I``, divided by their geometric
    mean. Each row is then scaled by
    ``(mol_volumes / (packing_coeff * |f|)) ** (1 / 3)`` with
    ``f = cell_vol_angle_factor(cell_angles)``, so that the parallelepiped
    with these lengths and ``cell_angles`` has volume
    ``mol_volumes / packing_coeff``. Conversion to unit-cell lengths is left
    to the caller.

    Parameters
    ----------
    num_samples : int
        Number of rows.
    cell_angles : torch.Tensor
        [num_samples, 3] cell angles in radians.
    mol_volumes : torch.Tensor
        [num_samples] molecular volumes; the output lengths are in the length
        unit whose cube is the unit of ``mol_volumes``.
    target_packing_coeff : float or torch.Tensor, optional
        Packing coefficient. A float is broadcast to all rows; a tensor with
        ``len > 1`` is multiplied by ``ones(num_samples)``; a tensor with
        ``len == 1`` is used as is (a 0-d tensor raises ``TypeError`` at
        ``len``). Any other non-None type fails an ``assert``. If None
        (default), coefficients are drawn from
        N(packing_coeff_mean, packing_coeff_std) and clipped to
        [clip_packing_min, clip_packing_max].
    clip_packing_min, clip_packing_max : float, optional
        Clip range for sampled packing coefficients. Defaults 0.4 and 1.0.
    eps : float, optional
        Diagonal jitter added to the log-length covariance. Default 1e-5.

    Returns
    -------
    torch.Tensor
        [num_samples, 3] asymmetric-unit lengths, on the device of
        ``mol_volumes``.
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
    Sample type-I (acute) Niggli-reduced box parameters from a random prior.

    Draws [num_graphs, 6] standard normal values and maps them with
    :func:`randn_to_niggli_box_vectors`. Without ``target_packing_coeff``
    the sampled cells satisfy the main Niggli conditions (a < b < c, 0 < cos(alpha) < b / (2c),
    0 < cos(beta) < a / (2c), 0 < cos(gamma) < a / (2b)), not the
    symmetry-breaking special conditions.

    Parameters
    ----------
    num_graphs : int
        Number of rows.
    mol_radii : torch.Tensor
        [num_graphs] molecular radii; set the length scale and the device.
    mol_volumes : torch.Tensor
        [num_graphs] molecular volumes; used only with
        ``target_packing_coeff``.
    sg_inds : iterable
        [num_graphs] space-group numbers, looked up as ``str(int(ind))``.
    sym_mult : torch.Tensor
        [num_graphs] symmetry multiplicities; used only with
        ``target_packing_coeff``.
    asym_unit_dict : dict, optional
        Space-group number as a string -> tensor [3] of asymmetric-unit axis
        fractions. If None, built from ``ASYM_UNITS`` with values converted
        to tensors on the device of ``mol_radii``.
    target_packing_coeff : float, optional
        If given, each row's three lengths are scaled by
        ``(mol_volumes * sym_mult / V / target_packing_coeff) ** (1 / 3)``,
        V being the sampled cell volume, and c is then clipped below at
        ``0.1 * 2 * mol_radii * asym_unit_dict[sg][2]``. The sampler's own
        lower bound on c is ``0.1 * 2 * mol_radii / asym_unit_dict[sg][2]``.
        The clip can move the packing coefficient off the target and break
        the cosine bounds on alpha and beta.

    Returns
    -------
    cell_lengths : torch.Tensor
        [num_graphs, 3] (a, b, c), in the length unit of ``mol_radii``.
    cell_angles : torch.Tensor
        [num_graphs, 3] (alpha, beta, gamma) in radians, each in
        (pi / 3, pi / 2).
    """
    if asym_unit_dict is None:
        asym_unit_dict = ASYM_UNITS.copy()
        sgs_to_tensorize = asym_unit_dict.keys()
        for key in sgs_to_tensorize:
            asym_unit_dict[key] = torch.tensor(asym_unit_dict[key], device=mol_radii.device)

    rands = torch.randn((num_graphs, 6), device=mol_radii.device)

    a_out, al_out, b_out, be_out, c_out, ga_out = randn_to_niggli_box_vectors(asym_unit_dict,
                                                                              mol_radii,
                                                                              rands,
                                                                              sg_inds)
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
    """
    Map standard-normal draws to type-I (acute) Niggli-reduced box parameters.

    With ``r = rands`` and ``auv_c = asym_unit_dict[str(int(sg))][2]``:

    - c: the asymmetric-unit c length in units of ``2 * mol_radii`` is
      ``exp(log(0.1) + (log(4) - log(0.1)) * sigmoid(0.24 + 0.3618 * r[:, 2]))``,
      which lies in (0.1, 4); ``c`` is that value / ``auv_c`` * ``2 * mol_radii``.
    - ``b = (0.1 + 0.9 * sigmoid(r[:, 1] + 1)) * c`` and
      ``a = (0.1 + 0.9 * sigmoid(r[:, 0] + 1)) * b``, so a < b < c.
    - ``cos(alpha) = sigmoid(2 r[:, 3]) * b / (2c)``,
      ``cos(beta) = sigmoid(2 r[:, 4]) * a / (2c)``,
      ``cos(gamma) = sigmoid(2 r[:, 5]) * a / (2b)``; each angle is the
      arccos, in radians, and lies in (pi / 3, pi / 2).

    Parameters
    ----------
    asym_unit_dict : dict
        Space-group number as a string -> tensor [3] of asymmetric-unit axis
        fractions. Values must be tensors; they are combined with
        ``torch.stack``.
    mol_radii : torch.Tensor
        [n] molecular radii.
    rands : torch.Tensor
        [n, 6] standard-normal draws.
    sg_inds : iterable
        [n] space-group numbers.

    Returns
    -------
    tuple of torch.Tensor
        ``(a, alpha, b, beta, c, gamma)`` in that interleaved order, each
        [n]; lengths in the unit of ``mol_radii``, angles in radians.
    """
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
    Map box parameters back to the standard-normal basis of :func:`randn_to_niggli_box_vectors`.

    a, b and the angles use the inverses of the forward maps, with the
    ratios a / b and b / c clamped to [0.1 + 1e-3, 1 - 1e-3] and the scaled
    cosines clamped to [1e-3, 1 - 1e-3]. For c, the log asymmetric-unit
    length (in units of ``2 * mol_radii``) is clamped to
    [log(0.1 + 1e-4), log(8 - 1e-4)] and inverted over the range
    (log(0.1), log(8)), whereas the forward map uses (log(0.1), log(4)); the
    c component therefore does not round-trip.

    Parameters
    ----------
    asym_unit_dict : dict
        As in :func:`randn_to_niggli_box_vectors`.
    mol_radii : torch.Tensor
        [n] molecular radii.
    a, b, c : torch.Tensor
        [n] cell lengths, in the unit of ``mol_radii``.
    al, be, ga : torch.Tensor
        [n] cell angles in radians.
    sg_inds : iterable
        [n] space-group numbers.

    Returns
    -------
    tuple of torch.Tensor
        ``(a_std, al_std, b_std, be_std, c_std, ga_std)``, each [n], in the
        interleaved order of the forward function's outputs.
    """
    # recover c
    log_mean = 0.24
    log_std = 0.3618
    c_normed = c / 2 / mol_radii
    auc_normed = c_normed * torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])[:, 2]
    log_auc = torch.log(auc_normed)
    min_val = 0.1
    max_val = 8
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
    """Return ``x_min + (x_max - x_min) * sigmoid(x)``, in (x_min, x_max)."""
    return x_min + (x_max - x_min) * torch.sigmoid(x)

def inverse_sigmoid_shift(x, x_min, x_max):
    """
    Inverse of :func:`sigmoid_shift`: ``logit((x - x_min) / (x_max - x_min))``.

    No clamping is applied; ``x`` at or outside the bounds gives inf or nan.
    """
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

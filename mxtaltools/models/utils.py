"""Model-side helpers: classifier score transforms, bounding, decoding and rescaling of
generated cell parameters, proxy embeddings for crystal batches, autoencoder loading, and
parameter counting.
"""

from argparse import Namespace
from typing import Union, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.geometry_utils import cell_vol_torch, components2angle, enforce_crystal_system, \
    batch_molecule_principal_axes_torch
from mxtaltools.common.training_utils import get_n_config
from mxtaltools.common.utils import softmax_np
from mxtaltools.crystal_building.utils import descale_asymmetric_unit, rescale_asymmetric_unit
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.task_models.autoencoder_models import Mo3ENet


def softmax_and_score(raw_classwise_output, temperature=1, old_method=False, correct_discontinuity=True) -> Union[
    torch.Tensor, np.ndarray]:
    """
    Map two-class logits to a real-valued score.

    Default method: ``score = log10(p_true / p_false)`` with ``p = softmax(raw)`` over the
    last dimension (:func:`softmax_np` for numpy input); ``temperature`` is not used.

    Old method: with ``p = softmax(raw / temperature)[:, 1]``,
    ``score = sign * log10(c + |tan((p - 0.5) * pi)|)``, where ``sign`` is +1 if
    ``raw[:, 1] > raw[:, 0]`` else -1 and ``c`` is 1 if ``correct_discontinuity`` else 0.
    The numpy branch computes in float64.

    Parameters
    ----------
    raw_classwise_output : numpy.ndarray or torch.Tensor
        Unnormalized [false, true] logits, [n, 2].
    temperature : float, optional
        Softmax temperature for the old method. Default 1.
    old_method : bool, optional
        Use the tangent-based score. Default False.
    correct_discontinuity : bool, optional
        Old method only: add 1 inside the logarithm. Default True.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Scores, [n], of the input's type. With ``old_method`` and an input that is
        neither an ndarray nor a tensor, returns None.
    """
    if not old_method:  # turns out you get almost identically the same answer by simply dividing the activations, much simpler
        if torch.is_tensor(raw_classwise_output):
            soft_activation = F.softmax(raw_classwise_output, dim=-1)
            score = torch.log10(soft_activation[:, 1] / soft_activation[:, 0])
            # if torch.sum(torch.isnan(score)) > 0:
            #     raise ValueError("Numerical Error: discriminator output is not finite")
            return score
        else:
            soft_activation = softmax_np(raw_classwise_output)
            score = np.log10(soft_activation[:, 1] / soft_activation[:, 0])
            # if np.sum(np.isnan(score)) > 0:
            #     raise ValueError("Numerical Error: discriminator output is not finite")
            return score
    else:
        if correct_discontinuity:
            correction = 1
        else:
            correction = 0

        if isinstance(raw_classwise_output, np.ndarray):
            softmax_output = softmax_np(raw_classwise_output.astype('float64'), temperature)[:, 1].astype(
                'float64')  # values get too close to zero for float32
            tanned = np.tan((softmax_output - 0.5) * np.pi)
            sign = (raw_classwise_output[:, 1] > raw_classwise_output[:,
                                                 0]) * 2 - 1  # values very close to zero can realize a sign error
            return sign * np.log10(correction + np.abs(tanned))  # new factor of 1+ conditions the function about zero

        elif torch.is_tensor(raw_classwise_output):
            softmax_output = F.softmax(raw_classwise_output / temperature, dim=-1)[:, 1]
            tanned = torch.tan((softmax_output - 0.5) * torch.pi)
            sign = (raw_classwise_output[:, 1] > raw_classwise_output[:,
                                                 0]) * 2 - 1  # values very close to zero can realize a sign error
            return sign * torch.log10(correction + torch.abs(tanned))


def norm_scores(score, tracking_features, dataDims):
    """
    Divide ``score`` by the ``'molecule volume'`` column of ``tracking_features``.

    The column index is ``dataDims['tracking_features'].index('molecule volume')``.
    """
    volume = tracking_features[:, dataDims['tracking_features'].index('molecule volume')]

    return score / volume


def enforce_1d_bound(x: torch.tensor, x_span, x_center, mode='soft'):  # soft or hard
    """
    constrains function to range x_center plus/minus x_span

    Parameters
    ----------
    x : torch.Tensor
        Values to bound, any shape.
    x_span : float or torch.Tensor
        Half-width of the range.
    x_center : float or torch.Tensor
        Centre of the range.
    mode : str, optional
        ``'soft'``: ``tanh((x - x_center) / x_span) * x_span + x_center``, in the open
        interval. ``'hard'``: the same with ``hardtanh``, i.e. clamped to the closed
        interval. Default ``'soft'``.

    Returns
    -------
    torch.Tensor
        Bounded values, same shape as ``x``.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'soft'`` or ``'hard'``.
    """
    if mode == 'soft':  # smoothly converge to (center-span,center+span)
        bounded = F.tanh((x - x_center) / x_span) * x_span + x_center
    elif mode == 'hard':  # linear scaling to hard stop at [center-span, center+span]
        bounded = F.hardtanh((x - x_center) / x_span) * x_span + x_center
    else:
        raise ValueError("bound must be of type 'hard' or 'soft'")

    return bounded


def undo_1d_bound(x: torch.tensor, x_span, x_center, mode='soft'):
    """
    undo / rescale an enforced 1d bound
    only setup for soft rescaling

    Returns ``x_span * atanh((x - x_center) / x_span) + x_center``, the inverse of
    :func:`enforce_1d_bound` in ``'soft'`` mode; values on or outside the interval
    boundaries give infinite or NaN results.

    Raises
    ------
    ValueError
        If ``mode`` is ``'hard'`` (not implemented) or any value other than ``'soft'``.
    """
    # todo: write a version for hard bounds

    if mode == 'soft':
        out = x_span * torch.atanh((x - x_center) / x_span) + x_center
        return out
    elif mode == 'hard':  # linear scaling to hard stop at [center-span, center+span]
        raise ValueError("'hard' bound not yet implemented")
    else:
        raise ValueError("bound must be of type 'soft'")


def compute_reduced_volume_fraction(cell_lengths: torch.tensor,
                                    cell_angles: torch.tensor,
                                    atom_radii: torch.tensor,
                                    batch: torch.tensor,
                                    crystal_multiplicity: torch.tensor):
    """
    Reduced volume fraction: cell volume divided by ``crystal_multiplicity``, over the
    summed atomic sphere volumes of each sample.

    Returns ``(V_cell / crystal_multiplicity) / sum_atoms(4/3 * pi * atom_radii**3)``,
    with ``V_cell`` from :func:`cell_vol_torch` for each sample (computed in a loop, into
    a float32 tensor on the device of ``cell_lengths``) and the sum taken per sample over
    ``batch``.

    Parameters
    ----------
    cell_lengths : torch.Tensor
        Cell lengths, [n, 3].
    cell_angles : torch.Tensor
        Cell angles, [n, 3], in radians.
    atom_radii : torch.Tensor
        Radius of each atom, [num_atoms].
    batch : torch.Tensor
        Sample index of each atom, [num_atoms].
    crystal_multiplicity : torch.Tensor
        Divisor of each cell volume, [n].

    Returns
    -------
    torch.Tensor
        Reduced volume fractions, [n].

    Notes
    -----
    Marked for deprecation in favour of the packing coefficient.
    """

    cell_volumes = torch.zeros(len(cell_lengths), dtype=torch.float32, device=cell_lengths.device)
    for i in range(len(cell_lengths)):  # todo switch to the parallel version of this function
        cell_volumes[i] = cell_vol_torch(cell_lengths[i], cell_angles[i])

    return (cell_volumes / crystal_multiplicity) / scatter(4 / 3 * torch.pi * atom_radii ** 3, batch, reduce='sum')


def clean_generator_output(samples=None,
                           lattice_lengths=None,
                           lattice_angles=None,
                           mol_positions=None,
                           mol_orientations=None,
                           lattice_means=None,
                           lattice_stds=None,
                           destandardize=True,
                           mode='soft',
                           skip_angular_dof=False):
    """
    convert from raw model output to the actual cell parameters with appropriate bounds
    considering raw outputs to be in the standardized basis, we destandardize, then enforce bounds

    Components come from ``samples`` columns ``[:, :3]`` (lengths), ``[:, 3:6]``
    (angles), ``[:, 6:9]`` (positions) and ``[:, 9:]`` (orientations) when ``samples`` is
    given, otherwise from the separate arguments.

    With ``destandardize``, lengths, angles and positions become ``value * std + mean``
    using ``lattice_stds``/``lattice_means`` columns 0:3, 3:6 and 6:9; orientations with
    3 columns use columns 9:, and orientations with other widths are copied unchanged.

    Orientations: with 6 columns they are decoded by :func:`decode_to_sph_rotvec`. With
    3 columns and ``mode`` not None they are bounded with :func:`enforce_1d_bound` to
    theta in pi/4 +/- pi/4, phi in 0 +/- pi and r in pi +/- pi; with ``mode`` None,
    ``theta, phi, r = orientations`` unpacks along the first (sample) dimension, which
    succeeds only for exactly three samples and then yields rows rather than columns.
    Any other width leaves ``theta``, ``phi`` and ``r`` unassigned and raises
    UnboundLocalError. In every case ``r`` is then floored at 0.01.

    Bounds when ``mode`` is not None: lengths become ``softplus(L - 0.01) + 0.01``
    (``'soft'``) or ``max(relu(L), 1)`` (``'hard'``); angles are bounded to
    ``pi/2 +/- 0.4 pi`` and positions to ``0.5 +/- 0.5`` by :func:`enforce_1d_bound`
    with ``mode``, which raises ValueError for any other mode string. With ``mode``
    None, lengths, angles and positions are returned after the optional
    destandardization only.

    Parameters
    ----------
    samples : torch.Tensor, optional
        Stacked parameters, [n, 9 + orientation width].
    lattice_lengths, lattice_angles, mol_positions, mol_orientations : torch.Tensor, optional
        Separate components, [n, 3] each (orientations [n, 3] or [n, 6]); used when
        ``samples`` is None.
    lattice_means, lattice_stds : torch.Tensor, optional
        Per-column standardization statistics; read only when ``destandardize``.
    destandardize : bool, optional
        Default True.
    mode : str or None, optional
        ``'soft'``, ``'hard'`` or None. Default ``'soft'``.
    skip_angular_dof : bool, optional
        Not used.

    Returns
    -------
    tuple of torch.Tensor
        ``(lengths, angles, positions, orientations)``, each [n, 3]. The angle and
        orientation bounds are multiples of pi (radians); positions are bounded but
        not rescaled.

    Notes
    -----
    Marked for rewrite.
    """

    '''separate components'''
    if samples is not None:
        lattice_lengths = samples[:, :3]
        lattice_angles = samples[:, 3:6]
        mol_positions = samples[:, 6:9]
        mol_orientations = samples[:, 9:]

    '''destandardize & decode angles'''
    if destandardize:
        real_lattice_lengths = lattice_lengths * lattice_stds[:3] + lattice_means[:3]
        real_lattice_angles = lattice_angles * lattice_stds[3:6] + lattice_means[
                                                                   3:6]  # not bothering to encode as an angle
        real_mol_positions = mol_positions * lattice_stds[6:9] + lattice_means[6:9]
        if mol_orientations.shape[-1] == 3:
            real_mol_orientations = mol_orientations * lattice_stds[9:] + lattice_means[9:]
        else:
            real_mol_orientations = mol_orientations * 1
    else:  # optionally, skip destandardization if we are already in the real basis
        real_lattice_lengths = lattice_lengths * 1
        real_lattice_angles = lattice_angles * 1
        real_mol_positions = mol_positions * 1
        real_mol_orientations = mol_orientations * 1

    if mol_orientations.shape[-1] == 6:
        theta, phi, r_i = decode_to_sph_rotvec(real_mol_orientations)
    # already have angles, no need to decode  # todo deprecate - we will only use spherical components in future
    elif mol_orientations.shape[-1] == 3:
        if mode is not None:
            theta = enforce_1d_bound(real_mol_orientations[:, 0], x_span=torch.pi / 4, x_center=torch.pi / 4,
                                     mode=mode)[:, None]
            phi = enforce_1d_bound(real_mol_orientations[:, 1], x_span=torch.pi, x_center=0, mode=mode)[:, None]
            r_i = enforce_1d_bound(real_mol_orientations[:, 2], x_span=torch.pi, x_center=torch.pi, mode=mode)[:, None]
        else:
            theta, phi, r_i = real_mol_orientations

    r = torch.maximum(r_i, torch.ones_like(r_i) * 0.01)  # MUST be nonzero
    clean_mol_orientations = torch.cat((theta, phi, r), dim=-1)

    '''enforce physical bounds'''
    if mode is not None:
        if mode == 'soft':
            clean_lattice_lengths = F.softplus(real_lattice_lengths - 0.01) + 0.01  # smoothly enforces positive nonzero
        elif mode == 'hard':
            clean_lattice_lengths = torch.maximum(F.relu(real_lattice_lengths), torch.ones_like(
                real_lattice_lengths))  # harshly enforces positive nonzero

        clean_lattice_angles = enforce_1d_bound(real_lattice_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2,
                                                mode=mode)  # range from (0,pi) with 20% limit to prevent too-skinny cells
        clean_mol_positions = enforce_1d_bound(real_mol_positions, 0.5, 0.5,
                                               mode=mode)  # enforce fractional centroids between 0 and 1
    else:  # do nothing
        clean_lattice_lengths, clean_lattice_angles, clean_mol_positions = real_lattice_lengths, real_lattice_angles, real_mol_positions

    return clean_lattice_lengths, clean_lattice_angles, clean_mol_positions, clean_mol_orientations


def decode_to_sph_rotvec(mol_orientations):
    """
    each angle is predicted with 2 params
    we bound the encodings for theta on 0-1 to restrict the range of theta to [0,pi/2]

    Each angle is ``atan2`` of a normalised (sin, cos) pair via :func:`components2angle`:
    theta from ``sigmoid(o[:, 0:2])``, in [0, pi/2]; phi from ``o[:, 2:4]``, in
    [-pi, pi]; r from ``o[:, 4:6]`` plus pi, in [0, 2 pi].

    Parameters
    ----------
    mol_orientations : torch.Tensor
        Encoded orientations, [n, 6].

    Returns
    -------
    tuple of torch.Tensor
        ``(theta, phi, r)`` in radians, each [n, 1].
    """
    theta_encoding = F.sigmoid(mol_orientations[:, 0:2])  # restrict to positive quadrant
    real_orientation_theta = components2angle(theta_encoding)  # from the sigmoid, [0, pi/2]
    real_orientation_phi = components2angle(mol_orientations[:, 2:4])  # unrestricted [-pi,pi]
    real_orientation_r = components2angle(
        mol_orientations[:, 4:6]) + torch.pi  # shift from [-pi,pi] to [0, 2pi]  # want vector to have a positive norm

    return real_orientation_theta[:, None], real_orientation_phi[:, None], real_orientation_r[:, None]


def decode_to_sph_rotvec2(mol_orientation_components):
    """
    As :func:`decode_to_sph_rotvec`, but with theta given directly as a scalar.

    theta is column 0, passed through unchanged (not bounded); phi is
    :func:`components2angle` of columns 1:3, in [-pi, pi]; r is :func:`components2angle`
    of columns 3:5 plus pi, in [0, 2 pi].

    Parameters
    ----------
    mol_orientation_components : torch.Tensor
        Encoded orientations, [n, 5].

    Returns
    -------
    tuple of torch.Tensor
        ``(theta, phi, r)``, each [n, 1].

    Notes
    -----
    Marked as a candidate for deprecation.
    """
    # theta_encoding = F.sigmoid(mol_orientations[:, 0:2])  # restrict to positive quadrant
    # real_orientation_theta = components2angle(theta_encoding)  # from the sigmoid, [0, pi/2]
    real_orientation_phi = components2angle(mol_orientation_components[:, 1:3])  # unrestricted [-pi,pi]
    real_orientation_r = components2angle(mol_orientation_components[:,
                                          3:5]) + torch.pi  # shift from [-pi,pi] to [0, 2pi]  # want vector to have a positive norm

    return mol_orientation_components[:, 0, None], real_orientation_phi[:, None], real_orientation_r[:, None]


def get_regression_loss(regressor, data, targets, mean, std):
    """
    Elementwise smooth-L1 loss of a regressor, with destandardized predictions and targets.

    Parameters
    ----------
    regressor : callable
        Model; ``regressor(data)`` is flattened to the predictions.
    data : object
        Input passed to ``regressor``.
    targets : torch.Tensor
        Standardized targets, same shape as the flattened predictions.
    mean, std : float or torch.Tensor
        Target standardization statistics.

    Returns
    -------
    tuple of torch.Tensor
        ``(loss, predictions * std + mean, targets * std + mean)``: the unreduced
        smooth-L1 loss in the standardized basis, then detached destandardized
        predictions and targets.

    Raises
    ------
    AssertionError
        If the target and prediction shapes differ.
    """
    predictions = regressor(data).flatten()
    assert targets.shape == predictions.shape
    return (F.smooth_l1_loss(predictions, targets, reduction='none'),
            predictions.detach() * std + mean,
            targets.detach() * std + mean)


def dict_of_tensors_to_cpu_numpy(stats):
    """Replace, in place, tensor values of ``stats`` by CPU numpy arrays and values whose type name contains ``'DataBatch'`` by their ``.cpu()`` copies; returns None."""
    for key, value in stats.items():
        if torch.is_tensor(value):
            stats[key] = value.cpu().numpy()
        elif 'DataBatch' in str(type(value)):
            stats[key] = value.cpu()


def increment_value(value, increment, maxval, minval=0):
    """Return ``value + increment`` clamped to ``[minval, maxval]``."""
    return max(minval, min(value + increment, maxval))


def clean_cell_params(samples,
                      sg_inds,
                      lattice_means,
                      lattice_stds,
                      symmetries_dict,
                      asym_unit_dict,
                      rescale_asymmetric_unit=True,
                      destandardize=False,
                      mode='soft',
                      fractional_basis='asymmetric_unit',
                      skip_angular_dof=False):
    """
    An important function for enforcing physical limits on cell parameterization
    with randomly generated samples of different sources.

    Splits ``samples`` into lengths ``[:, :3]``, angles ``[:, 3:6]``, positions
    ``[:, 6:9]`` and orientations ``[:, 9:]``, bounds them with
    :func:`clean_generator_output`, applies :func:`enforce_crystal_system` to the lengths
    and angles, and optionally scales the positions by the asymmetric-unit extents.

    Parameters
    ----------
    samples : torch.Tensor
        Cell parameters, [n, 9 + orientation width] (orientation width 3 or 6).
    sg_inds : torch.LongTensor
        Space-group index of each sample, [n].
    lattice_means, lattice_stds : torch.Tensor
        Standardization statistics; read only when ``destandardize``.
    symmetries_dict : dict
        Symmetry information passed to :func:`enforce_crystal_system`.
    asym_unit_dict : dict
        Per-space-group asymmetric-unit extents keyed by ``str(sg_ind)``, used by
        :func:`descale_asymmetric_unit`.
    rescale_asymmetric_unit : bool, optional
        If True, the bounded positions are multiplied by the asymmetric-unit extents
        (:func:`descale_asymmetric_unit`); otherwise copied. Default True.
    destandardize : bool, optional
        Passed to :func:`clean_generator_output`. Default False.
    mode : str or None, optional
        Passed to :func:`clean_generator_output`. Default ``'soft'``.
    fractional_basis : str, optional
        ``'asymmetric_unit'``: positions are taken as given. ``'unit_cell'``: positions
        are first multiplied by the asymmetric-unit extents
        (:func:`descale_asymmetric_unit`) before bounding. Any other value raises
        AssertionError. Default ``'asymmetric_unit'``.
    skip_angular_dof : bool, optional
        Passed to :func:`clean_generator_output`, which does not use it.

    Returns
    -------
    torch.Tensor
        ``[lengths, angles, positions, orientations]`` concatenated, [n, 12].

    Notes
    -----
    Marked for deprecation.
    """
    lattice_lengths = samples[:, :3]
    lattice_angles = samples[:, 3:6]
    mol_orientations = samples[:, 9:]

    if fractional_basis == 'asymmetric_unit':  # basis is 0-1 within the asymmetric unit
        mol_positions = samples[:, 6:9]

    elif fractional_basis == 'unit_cell':  # basis is 0-1 within the unit cell
        mol_positions = descale_asymmetric_unit(asym_unit_dict, samples[:, 6:9], sg_inds)

    else:
        assert False, f"{fractional_basis} is not an implemented fractional basis"

    lattice_lengths, lattice_angles, mol_positions, mol_orientations \
        = clean_generator_output(lattice_lengths=lattice_lengths,
                                 lattice_angles=lattice_angles,
                                 mol_positions=mol_positions,
                                 mol_orientations=mol_orientations,
                                 lattice_means=lattice_means,
                                 lattice_stds=lattice_stds,
                                 destandardize=destandardize,
                                 mode=mode,
                                 skip_angular_dof=skip_angular_dof)

    fixed_lengths, fixed_angles = (
        enforce_crystal_system(lattice_lengths, lattice_angles, sg_inds, symmetries_dict))

    if rescale_asymmetric_unit:
        fixed_positions = descale_asymmetric_unit(asym_unit_dict, mol_positions, sg_inds)
    else:
        fixed_positions = mol_positions * 1

    '''collect'''
    final_samples = torch.cat((
        fixed_lengths,
        fixed_angles,
        fixed_positions,
        mol_orientations,
    ), dim=-1)

    return final_samples


def denormalize_generated_cell_params(
        normed_cell_samples: torch.FloatTensor,
        mol_data,
        asym_unit_dict: dict):
    """
    Convert normalized cell parameters to buildable ones.

    Lengths ``[:, :3]`` are multiplied by ``(mol_data.sym_mult * mol_data.mol_volume) **
    (1/3)``; positions ``[:, 6:9]`` are multiplied by the asymmetric-unit extents of
    ``mol_data.sg_ind`` (:func:`descale_asymmetric_unit`); angles ``[:, 3:6]`` and
    orientations ``[:, 9:12]`` are copied. Inverse of
    :func:`renormalize_generated_cell_params`.

    Parameters
    ----------
    normed_cell_samples : torch.FloatTensor
        Normalized parameters, [n, 12].
    mol_data : molecule batch
        Reads ``sym_mult``, ``mol_volume`` and ``sg_ind``, each [n].
    asym_unit_dict : dict
        Per-space-group asymmetric-unit extents keyed by ``str(sg_ind)``.

    Returns
    -------
    torch.Tensor
        [n, 12] parameters.
    """
    # denormalize the predicted cell lengths
    cell_lengths = torch.pow(mol_data.sym_mult * mol_data.mol_volume, 1 / 3)[:, None] * normed_cell_samples[:, :3]
    # rescale asymmetric units  # todo add assertions around these
    mol_positions = descale_asymmetric_unit(asym_unit_dict,
                                            normed_cell_samples[:, 6:9],
                                            mol_data.sg_ind)
    generated_samples_to_build = torch.cat(
        [cell_lengths, normed_cell_samples[:, 3:6], mol_positions, normed_cell_samples[:, 9:12]], dim=1)
    return generated_samples_to_build


def renormalize_generated_cell_params(
        generator_raw_samples,
        mol_data,
        asym_unit_dict):
    """
    Convert buildable cell parameters to normalized ones; inverse of
    :func:`denormalize_generated_cell_params`.

    Lengths ``[:, :3]`` are divided by ``(mol_data.sym_mult * mol_data.mol_volume) **
    (1/3)``; positions ``[:, 6:9]`` are divided by the asymmetric-unit extents of
    ``mol_data.sg_ind`` (:func:`rescale_asymmetric_unit`); angles ``[:, 3:6]`` and
    orientations ``[:, 9:12]`` are copied. Returns [n, 12].
    """
    # renormalize the predicted cell lengths
    cell_lengths = generator_raw_samples[:, :3] / torch.pow(mol_data.sym_mult * mol_data.mol_volume, 1 / 3)[:, None]
    # rescale asymmetric units  # todo add assertions around these
    mol_positions = rescale_asymmetric_unit(asym_unit_dict,
                                            generator_raw_samples[:, 6:9],
                                            mol_data.sg_ind)
    generated_samples_to_build = torch.cat(
        [cell_lengths, generator_raw_samples[:, 3:6], mol_positions, generator_raw_samples[:, 9:12]], dim=1)
    return generated_samples_to_build


def compute_prior_loss(norm_factors: torch.Tensor,
                       sg_inds: torch.LongTensor,
                       generator_raw_samples: torch.Tensor,
                       prior: torch.Tensor,
                       variation_factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Take the norm of the scaled distances between prior and generated samples,
    and apply a quadratic penalty when it is larger than variation_factor

    ``scaled_deviation = |prior - generator_raw_samples| / (norm_factors[sg_inds] + 1e-4)``
    and ``prior_loss = relu(||scaled_deviation||_2 - variation_factor) ** 2``, with the
    norm taken over dimension 1.

    Parameters
    ----------
    norm_factors : torch.Tensor
        Per-space-group scale factors, indexed by ``sg_inds`` along dimension 0.
    sg_inds : torch.LongTensor
        Space-group index of each sample, [n].
    generator_raw_samples : torch.Tensor
        Generated samples, [n, d].
    prior : torch.Tensor
        Prior samples, [n, d].
    variation_factor : torch.Tensor
        Tolerance subtracted from each norm; a scalar or [n].

    Returns
    -------
    tuple of torch.Tensor
        ``(prior_loss, scaled_deviation)``, [n] and [n, d].
    """
    scaling_factor = (norm_factors[sg_inds, :] + 1e-4)
    scaled_deviation = torch.abs(prior - generator_raw_samples) / scaling_factor
    prior_loss = F.relu(torch.linalg.norm(scaled_deviation, dim=1) - variation_factor) ** 2  # 'flashlight' search
    return prior_loss, scaled_deviation


def get_mol_embedding_for_proxy(crystal_batch,
                                embedding_type,
                                encoder: Optional = None):
    """
    Build a per-crystal molecule embedding: scalar part ``s`` and vector part ``v``.

    Modifies ``crystal_batch`` in place: ``pose_aunit()`` and ``recenter_molecules()``
    reassign ``crystal_batch.pos`` before the embedding is computed.

    Parameters
    ----------
    crystal_batch : crystal batch
        Reads ``pos``, ``batch``, ``num_graphs``, ``num_atoms``, ``z``, ``mol_volume``
        and ``device`` as the selected embedding requires.
    embedding_type : str or None
        ``'autoencoder'``: ``v = encoder.encode(crystal_batch.clone())`` and
        ``s = encoder.scalarizer(v)``.
        ``'principal_axes'``: ``s`` is the principal inertial moments over heavy atoms
        divided by (35, 91, 105), [num_graphs, 3]; ``v`` is the principal-axes matrix
        transposed, [num_graphs, 3, 3].
        ``'principal_moments'``: the same ``s``; ``v`` is zeros.
        ``'mol_volume'``: ``s`` is ``mol_volume / 70`` repeated three times; ``v`` is
        float32 zeros [num_graphs, 3, 3].
        None: ``s`` zeros [num_graphs, 3] and ``v`` float32 zeros [num_graphs, 3, 3].
        Any other value raises AssertionError.
    encoder : optional
        Model with ``encode`` and ``scalarizer`` methods; used only for
        ``'autoencoder'``.

    Returns
    -------
    torch.Tensor
        ``torch.cat([s, v.flatten(-2)], dim=-1)``: [num_graphs, 12] for every type except
        ``'autoencoder'``, whose width depends on the encoder.
    """
    crystal_batch.pose_aunit()  # get correct orientation
    crystal_batch.recenter_molecules()
    ipm_means = torch.tensor([35, 91, 105], dtype=torch.float32, device=crystal_batch.device)
    mol_volume_mean = 70
    if embedding_type == 'autoencoder':
        v_embedding = encoder.encode(crystal_batch.clone())
        s_embedding = encoder.scalarizer(v_embedding)
    elif embedding_type == 'principal_axes':
        v_embedding_i, s_embedding_i, _ = batch_molecule_principal_axes_torch(
            crystal_batch.pos,
            crystal_batch.batch,
            crystal_batch.num_graphs,
            crystal_batch.num_atoms,
            heavy_atoms_only=True,
            atom_types=crystal_batch.z
        )

        s_embedding = s_embedding_i / ipm_means[None, :]
        v_embedding = v_embedding_i.permute(0, 2, 1)

    elif embedding_type == 'principal_moments':
        Ip, s_embedding_i, _ = batch_molecule_principal_axes_torch(
            crystal_batch.pos,
            crystal_batch.batch,
            crystal_batch.num_graphs,
            crystal_batch.num_atoms,
            heavy_atoms_only=True,
            atom_types=crystal_batch.z
        )
        v_embedding = torch.zeros_like(Ip)

        s_embedding = s_embedding_i / ipm_means[None, :]

    elif embedding_type == 'mol_volume':
        s_embedding = crystal_batch.mol_volume[:, None].repeat(1, 3) / mol_volume_mean
        v_embedding = torch.zeros((crystal_batch.num_graphs, 3, 3), dtype=torch.float32, device=crystal_batch.device)
    elif embedding_type is None:

        s_embedding = torch.zeros_like(crystal_batch.mol_volume[:, None].repeat(1, 3))
        v_embedding = torch.zeros((crystal_batch.num_graphs, 3, 3), dtype=torch.float32, device=crystal_batch.device)
    else:
        assert False, f"{embedding_type} is not an implemented proxy discriminator embedding"

    return torch.cat([s_embedding, v_embedding.flatten(-2)], dim=-1)


def embed_crystal_list(
        batch_size: int,
        crystal_list: list,
        embedding_type: str,
        encoder_checkpoint_path: Optional = None,
        device: Optional[str] = 'cpu',
        redo_crystal_analysis: Optional[bool] = False
) -> list:
    """
    Compute proxy embeddings for a list of crystals and attach them to its elements.

    Crystals are collated in consecutive chunks of ``batch_size`` (list order kept) with
    :func:`collate_data_list`, moved to ``device``, and embedded with
    ``do_embedding(embedding_type, encoder)`` under ``torch.no_grad()``. Each element
    then gets ``embedding``, [1, embedding width], on ``device``. With
    ``redo_crystal_analysis``, each chunk is also analysed with
    ``build_and_analyze(cutoff=10, return_cluster=True)`` and
    ``compute_buckingham_energy()`` on the returned cluster, and each element gets
    ``lj``, ``es_pot``, ``scaled_lj`` and ``bh_pot`` as 0-d CPU tensors.

    Parameters
    ----------
    batch_size : int
        Chunk size.
    crystal_list : list
        Crystal data objects; modified in place.
    embedding_type : str
        Passed to ``do_embedding``; see :func:`get_mol_embedding_for_proxy`.
    encoder_checkpoint_path : str, optional
        If given, ``load_encoder(encoder_checkpoint_path).to(device)`` is the encoder. If
        None, the local ``encoder`` is never assigned and the first ``do_embedding`` call
        raises UnboundLocalError, whatever the embedding type.
    device : str, optional
        Device for the batches and the result tensors. Default ``'cpu'``.
    redo_crystal_analysis : bool, optional
        Recompute and attach the energies. Default False.

    Returns
    -------
    list
        The same ``crystal_list`` object.
    """
    if encoder_checkpoint_path is not None:
        encoder = load_encoder(encoder_checkpoint_path).to(device)
    embeddings = []
    num_chunks = len(crystal_list) // batch_size + int(len(crystal_list) % batch_size != 0)
    lj_pots, scaled_lj_pots, es_pots, bh_pots = (torch.zeros(len(crystal_list), dtype=torch.float32, device=device) for
                                                 _ in range(4))
    with torch.no_grad():
        for ind in tqdm(range(num_chunks)):  # do it this way so to avoid shuffling
            sample_inds = torch.arange(ind * batch_size, min((ind + 1) * batch_size, len(crystal_list)))
            crystal_batch = collate_data_list([crystal_list[ind] for ind in sample_inds]
                                              ).to(device)
            if redo_crystal_analysis:
                lj_pots[sample_inds], es_pots[sample_inds], scaled_lj_pots[
                    sample_inds], cluster_batch = crystal_batch.build_and_analyze(cutoff=10, return_cluster=True)
                bh_pots[sample_inds] = cluster_batch.compute_buckingham_energy()

            embedding = crystal_batch.do_embedding(embedding_type,
                                                   encoder
                                                   ).cpu().detach()
            if ind == 0:
                embeddings = torch.zeros(
                    (len(crystal_list), embedding.shape[-1]),
                    dtype=torch.float32,
                    device=device
                )
            embeddings[ind * batch_size:(ind + 1) * batch_size] = embedding

    for ind in range(len(crystal_list)):
        crystal_list[ind].embedding = embeddings[None, ind]
        if redo_crystal_analysis:
            crystal_list[ind].lj = lj_pots[ind].cpu()
            crystal_list[ind].es_pot = es_pots[ind].cpu()
            crystal_list[ind].scaled_lj = scaled_lj_pots[ind].cpu()
            crystal_list[ind].bh_pot = bh_pots[ind].cpu()

    return crystal_list


def load_encoder(checkpoint_path):
    """
    Load a :class:`Mo3ENet` autoencoder from a checkpoint.

    The checkpoint is read (twice) with ``torch.load(checkpoint_path,
    map_location='cpu', weights_only=False)``. ``checkpoint['config']`` is unpacked into a
    Namespace, and the model is built as ``Mo3ENet(0, config.model, 5, type_index, 1,
    protons_in_input=True)`` (seed, model config, number of atom types, atom-type index,
    radial normalization), where ``type_index`` is a long tensor of length 10 mapping
    atomic numbers 1, 6, 7, 8 and 9 to 0-4 and every other entry to -1. If the first key
    of ``checkpoint['model_state_dict']`` starts with ``'module'``, the first seven
    characters are removed from every key before ``load_state_dict`` (strict).

    Parameters
    ----------
    checkpoint_path : str or path-like
        Checkpoint with a ``'config'`` mapping (containing ``'model'``) and a
        ``'model_state_dict'``.

    Returns
    -------
    Mo3ENet
        The model, on CPU, with the checkpoint weights; its train/eval mode is not
        changed after construction.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model

    allowed_types = np.array([1, 6, 7, 8, 9])
    type_translation_index = np.zeros(allowed_types.max() + 1) - 1
    for ind, atype in enumerate(allowed_types):
        type_translation_index[atype] = ind
    autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')

    model = Mo3ENet(
        0,
        model_config.model,
        5,
        autoencoder_type_index,
        1,  # will get overwritten
        protons_in_input=True
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if list(checkpoint['model_state_dict'])[0][
       0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def get_model_sizes(models_dict: dict):
    """
    Count and print the parameters of each model.

    Returns ``{name + '_num_params': count}`` with ``count`` from :func:`get_n_config`
    (all parameters, trainable or not), and prints one line per entry.
    """
    num_params_dict = {model_name + "_num_params": get_n_config(model) for model_name, model in
                       models_dict.items()}
    [print(
        f'{model_name} {num_params_dict[model_name] / 1e6:.3f} million or {int(num_params_dict[model_name])} parameters')
        for model_name in num_params_dict.keys()]
    return num_params_dict


def test_gradient_flow(grad_params, operation):
    """
    Print the gradient of a scalar loss of ``operation(grad_params)`` with respect to ``grad_params.pos``.

    Sets ``grad_params.pos.requires_grad_(True)``, backpropagates
    ``(operation(grad_params) + 1).log().abs().sum()``, prints the gradient, the loss and
    the fraction of nonzero gradient entries, then zeroes ``grad_params.pos.grad`` in
    place. Returns None.
    """
    grad_params.pos.requires_grad_(True)
    output = operation(grad_params)
    grad_params.pos.retain_grad()

    loss = (output + 1).log().abs().sum()
    loss.backward()
    print(grad_params.pos.grad)
    print(loss)
    print(torch.count_nonzero(grad_params.pos.grad) / len(grad_params.pos.flatten()))
    grad_params.pos.grad.zero_()

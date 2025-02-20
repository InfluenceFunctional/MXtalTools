from typing import Union

import numpy as np
import torch
from torch.nn import functional as F
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import cell_vol_torch, components2angle, enforce_crystal_system
from mxtaltools.common.utils import softmax_np
from mxtaltools.crystal_building.utils import descale_asymmetric_unit, rescale_asymmetric_unit
from mxtaltools.dataset_utils.CrystalData import CrystalData


def softmax_and_score(raw_classwise_output, temperature=1, old_method=False, correct_discontinuity=True) -> Union[
    torch.Tensor, np.ndarray]:
    """
    Parameters
    ----------
    raw_classwise_output: numpy array or torch tensor with dimension [n,2], representing the non-normalized [false,true] probabilities
    temperature: softmax temperature
    old_method: use more complicated method from first paper
    correct_discontinuity: correct discontinuity at 0 only in the old method


    Returns
    -------
    score: linearizes the input probabilities from (0,1) to [-inf, inf] for easier visualization
    """
    if not old_method:  # turns out you get almost identically the same answer by simply dividing the activations, much simpler
        if torch.is_tensor(raw_classwise_output):
            soft_activation = F.softmax(raw_classwise_output, dim=-1)
            score = torch.log10(soft_activation[:, 1] / soft_activation[:, 0])
            assert torch.sum(torch.isnan(score)) == 0
            return score
        else:
            soft_activation = softmax_np(raw_classwise_output)
            score = np.log10(soft_activation[:, 1] / soft_activation[:, 0])
            assert np.sum(np.isnan(score)) == 0
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
    norm the incoming score according to some feature of the molecule (generally size)
    """
    volume = tracking_features[:, dataDims['tracking_features'].index('molecule volume')]

    return score / volume


def enforce_1d_bound(x: torch.tensor, x_span, x_center, mode='soft'):  # soft or hard
    """
    constrains function to range x_center plus/minus x_span
    Parameters
    ----------
    x
    x_span
    x_center
    mode

    Returns
    -------

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
    """
    # todo: write a version for hard bounds

    if mode == 'soft':
        return x_span * torch.atanh((x - x_center) / x_span) + x_center
    elif mode == 'hard':  # linear scaling to hard stop at [center-span, center+span]
        raise ValueError("'hard' bound not yet implemented")
    else:
        raise ValueError("bound must be of type 'soft'")


def compute_reduced_volume_fraction(cell_lengths: torch.tensor,
                                    cell_angles: torch.tensor,
                                    atom_radii: torch.tensor,
                                    batch: torch.tensor,
                                    crystal_multiplicity: torch.tensor):
    """  # TODO DEPRECATE IN FAVOUR OF PACKING COEFFICIENT

    Args:
        cell_lengths:
        cell_angles:
        atom_radii:
        crystal_multiplicity:

    Returns: asymmetric unit volume / sum of vdw volumes - so-called 'reduced volume fraction'

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
    """ # TODO rewrite - this is a very important function but it's currently a disaster
    convert from raw model output to the actual cell parameters with appropriate bounds
    considering raw outputs to be in the standardized basis, we destandardize, then enforce bounds
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
        real_lattice_angles = lattice_angles * lattice_stds[3:6] + lattice_means[3:6]  # not bothering to encode as an angle
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
    """
    theta_encoding = F.sigmoid(mol_orientations[:, 0:2])  # restrict to positive quadrant
    real_orientation_theta = components2angle(theta_encoding)  # from the sigmoid, [0, pi/2]
    real_orientation_phi = components2angle(mol_orientations[:, 2:4])  # unrestricted [-pi,pi]
    real_orientation_r = components2angle(
        mol_orientations[:, 4:6]) + torch.pi  # shift from [-pi,pi] to [0, 2pi]  # want vector to have a positive norm

    return real_orientation_theta[:, None], real_orientation_phi[:, None], real_orientation_r[:, None]


def decode_to_sph_rotvec2(mol_orientation_components):
    """  # todo decide whether to use/keep or deprecate this
    each angle is predicted with 2 params
    we bound the encodings for theta on 0-1 to restrict the range of theta to [0,pi/2]

    identical to the above, but considering theta as a simple scalar
    [n, 5] input to [n, 3] output
    """
    # theta_encoding = F.sigmoid(mol_orientations[:, 0:2])  # restrict to positive quadrant
    # real_orientation_theta = components2angle(theta_encoding)  # from the sigmoid, [0, pi/2]
    real_orientation_phi = components2angle(mol_orientation_components[:, 1:3])  # unrestricted [-pi,pi]
    real_orientation_r = components2angle(mol_orientation_components[:,
                                          3:5]) + torch.pi  # shift from [-pi,pi] to [0, 2pi]  # want vector to have a positive norm

    return mol_orientation_components[:, 0, None], real_orientation_phi[:, None], real_orientation_r[:, None]


def get_regression_loss(regressor, data, targets, mean, std):
    predictions = regressor(data).flatten()
    assert targets.shape == predictions.shape
    return (F.smooth_l1_loss(predictions, targets, reduction='none'),
            predictions.detach() * std + mean,
            targets.detach() * std + mean)


def dict_of_tensors_to_cpu_numpy(stats):
    for key, value in stats.items():
        if torch.is_tensor(value):
            stats[key] = value.cpu().numpy()
        elif 'CrystalDataBatch' in str(type(value)):
            stats[key] = value.cpu()


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
    """  # todo simplify and combine with clean_generator output
    An important function for enforcing physical limits on cell parameterization
    with randomly generated samples of different sources.


    Parameters
    ----------
    skip_angular_dof
    samples: torch.Tensor
    sg_inds: torch.LongTensor
    lattice_means: torch.Tensor
    lattice_stds: torch.Tensor
    symmetries_dict: dict
    asym_unit_dict: dict
    rescale_asymmetric_unit: bool
    destandardize: bool
    mode: str, "hard" or "soft"
    fractional_basis: bool

    Returns
    -------

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
        mol_data: CrystalData,
        asym_unit_dict: dict):
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
    Parameters
    ----------
    data
    generator_raw_samples
    prior
    variation_factor

    Returns
    -------

    """
    scaling_factor = (norm_factors[sg_inds, :] + 1e-4)
    scaled_deviation = torch.abs(prior - generator_raw_samples) / scaling_factor
    prior_loss = F.relu(torch.linalg.norm(scaled_deviation, dim=1) - variation_factor) ** 2  # 'flashlight' search
    return prior_loss, scaled_deviation




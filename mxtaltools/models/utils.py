import sys
from typing import Union, Tuple

import numpy as np
import torch
from scipy.stats import linregress
from torch import optim, nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler as lr_scheduler
from torch_scatter import scatter, scatter_softmax

from mxtaltools.common.geometry_calculations import cell_vol_torch
from mxtaltools.common.utils import softmax_np, components2angle
from mxtaltools.crystal_building.utils import descale_asymmetric_unit, descale_asymmetric_unit, rescale_asymmetric_unit
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.dataset_management.dataloader_utils import update_dataloader_batch_size
from mxtaltools.models.functions.asymmetric_radius_graph import radius
from mxtaltools.models.modules.components import construct_radial_graph


def set_lr(schedulers, optimizer, optimizer_config, err_tr, hit_max_lr, override_lr=None):
    if optimizer_config.lr_schedule and override_lr is None:
        lr = optimizer.param_groups[0]['lr']
        if lr > optimizer_config.min_lr:
            schedulers[0].step(np.mean(np.asarray(err_tr)))  # plateau scheduler

        if not hit_max_lr:
            schedulers[1].step()
        elif hit_max_lr:
            if lr > optimizer_config.min_lr:
                schedulers[2].step()  # start reducing lr
    elif override_lr is not None:
        for g in optimizer.param_groups:
            g['lr'] = override_lr

    lr = optimizer.param_groups[0]['lr']
    return optimizer, lr


def check_convergence(test_record, history, convergence_eps, epoch, minimum_epochs, overfit_tolerance,
                      train_record=None):
    """
    check if we are converged
    condition: test loss has increased or levelled out over the last several epochs
    :return: convergence flag
    """

    converged = False

    if epoch > minimum_epochs + 1:
        if type(test_record) is list:
            test_record = np.asarray([rec.mean() for rec in test_record])

        elif isinstance(test_record, np.ndarray):
            test_record = test_record.copy()

        if np.sum(np.isnan(test_record)) > 0:
            return True

        '''
        conditions
        1. not decreasing significantly quickly (log slope too shallow)
        XX not using2. not near global minimum
        3. train and test not significantly diverging
        '''

        lin_hist = test_record[-history:]
        if history > 20 and minimum_epochs > 20:  # scrub high outliers
            lin_hist = lin_hist[lin_hist < np.quantile(lin_hist, 0.95)]

        linreg = linregress(np.arange(len(lin_hist)), np.log10(lin_hist))
        converged = linreg.slope > -convergence_eps
        # if not converged:
        #     converged *= all(test_record[-history] > np.quantile(test_record, 0.05))
        if converged:
            print(f"Model Converged!: Slow convergence with log-slope of {linreg.slope:.5f}")
            return True

        if train_record is not None:
            if type(train_record) is list:
                train_record = np.asarray([rec.mean() for rec in train_record])

            elif isinstance(train_record, np.ndarray):
                train_record = train_record.copy()

            test_train_ratio = test_record / train_record
            if test_train_ratio[-history:].mean() > overfit_tolerance:
                print(f"Model Converged!: Overfit at {test_train_ratio[-history:].mean():.2f}")
                return True

    return converged


"""  game to help determine good values
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from plotly.subplots import make_subplots

x = np.linspace(0,1000,1000)
y = np.exp(-x/400)*(10*np.cos(x/100)**2 + 1) + np.random.randn(len(x))/2
history = 50
condition = np.zeros(len(x))
for ind in range(history,len(x)):
    linreg = linregress(x[ind-history:ind], np.log10(y[ind-history:ind]))
    condition[ind] = linreg.slope > -.0001
    # if not condition[ind]:
    #     condition[ind] *= all(y[ind-history:ind] > np.quantile(y[:ind], 0.05))

fig = make_subplots(rows=1,cols=2)
fig.add_scatter(x=x,y=np.log10(y),mode='markers',marker_color=condition.astype(float),row=1,col=1)
fig.add_scatter(x=x,y=y,mode='markers',marker_color=condition.astype(float),row=1,col=2)
fig.show(renderer='browser')
"""


def init_optimizer(model_name, optim_config, model, amsgrad=False, freeze_params=False):
    """
    initialize optimizers
    @param optim_config: config for a given optimizer
    @param model: model with params to be optimized
    @param freeze_params: whether parameters without requires_grad should be frozen
    @return: optimizer
    """
    if optim_config is None:
        beta1 = 0.9
        beta2 = 0.99
        weight_decay = 0.01
        momentum = 0
        optimizer = 'adam'
        init_lr = 1e-3
    else:
        beta1 = optim_config.beta1  # 0.9
        beta2 = optim_config.beta2  # 0.999
        weight_decay = optim_config.weight_decay  # 0.01
        optimizer = optim_config.optimizer
        init_lr = optim_config.init_lr

    amsgrad = amsgrad

    if model_name == 'autoencoder' and hasattr(model, 'encoder'):
        if freeze_params:
            assert False, "params freezing not implemented for autoencoder"

        params_dict = [
            {'params': model.encoder.parameters(), 'lr': optim_config.encoder_init_lr},
            {'params': model.decoder.parameters(), 'lr': optim_config.decoder_init_lr}
        ]

    else:
        if freeze_params:
            params_dict = [param for param in model.parameters() if param.requires_grad == True]
        else:
            params_dict = model.parameters()

    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(params_dict, amsgrad=amsgrad, lr=init_lr, betas=(beta1, beta2),
                               weight_decay=weight_decay)
    elif optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(params_dict, amsgrad=amsgrad, lr=init_lr, betas=(beta1, beta2),
                                weight_decay=weight_decay)
    elif optimizer.lower() == 'sgd':
        optimizer = optim.SGD(params_dict, lr=init_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print(optim_config.optimizer + ' is not a valid optimizer')
        sys.exit()

    return optimizer


def init_scheduler(optimizer, optimizer_config):
    """
    initialize a series of LR schedulers
    """
    if optimizer_config is not None:
        lr_shrink_lambda = optimizer_config.lr_shrink_lambda
        lr_growth_lambda = optimizer_config.lr_growth_lambda
        use_plateau_scheduler = optimizer_config.use_plateau_scheduler
    else:
        lr_shrink_lambda = 1  # no change
        lr_growth_lambda = 1
        use_plateau_scheduler = False

    if use_plateau_scheduler:
        scheduler1 = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=500,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=500
        )
    else:
        scheduler1 = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.00001,
            patience=5000000,
            threshold=1e-8,
            threshold_mode='rel',
            cooldown=5000000,
            min_lr=1
        )
    scheduler2 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_growth_lambda)
    scheduler3 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_shrink_lambda)

    return [scheduler1, scheduler2, scheduler3]


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


def reload_model(model, device, optimizer, path, reload_optimizer=False):
    """
    load model and state dict from path
    includes fix for potential dataparallel issue
    """
    checkpoint = torch.load(path, map_location=device)
    if list(checkpoint['model_state_dict'])[0][
       0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        if reload_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def compute_packing_coefficient(cell_params: torch.tensor, mol_volumes: torch.tensor,
                                crystal_multiplicity: torch.tensor):
    """
    @param cell_params: cell parameters using our standard scheme 0-5 are a,b,c,alpha,beta,gamma
    @param mol_volumes: molumes in cubic angstrom of each single molecule
    @param crystal_multiplicity: Z value for each crystal
    @return: crystal packing coefficient
    """
    volumes_list = []
    for i in range(len(cell_params)):
        volumes_list.append(cell_vol_torch(cell_params[i, 0:3], cell_params[i, 3:6]))
    cell_volumes = torch.stack(volumes_list)
    coeffs = crystal_multiplicity * mol_volumes / cell_volumes
    return coeffs


def compute_reduced_volume_fraction(cell_lengths: torch.tensor,
                                    cell_angles: torch.tensor,
                                    atom_radii: torch.tensor,
                                    batch: torch.tensor,
                                    crystal_multiplicity: torch.tensor):
    """

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


def compute_num_h_bonds(supercell_data, atom_acceptor_ind, atom_donor_ind, i):
    """
    compute the number of hydrogen bonds, up to a loose range (3.3 angstroms), and non-directionally
    @param atom_donor_ind: index in tracking_features to find donor status
    @param atom_acceptor_ind: index in tracking_features to find acceptor status
    @param supercell_data: crystal data
    @param i: cell index we are checking
    @return: sum of total hydrogen bonds for the canonical conformer
    """
    batch_inds = torch.arange(supercell_data.ptr[i], supercell_data.ptr[i + 1], device=supercell_data.x.device)

    # find the canonical conformers
    canonical_conformers_inds = torch.where(supercell_data.aux_ind[batch_inds] == 0)[0]
    outside_inds = torch.where(supercell_data.aux_ind[batch_inds] == 1)[0]

    # identify and count canonical conformer acceptors and intermolecular donors
    canonical_conformer_acceptors_inds = \
        torch.where(supercell_data.x[batch_inds[canonical_conformers_inds], atom_acceptor_ind] == 1)[0]
    outside_donors_inds = torch.where(supercell_data.x[batch_inds[outside_inds], atom_donor_ind] == 1)[0]

    donors_pos = supercell_data.pos[batch_inds[outside_inds[outside_donors_inds]]]
    acceptors_pos = supercell_data.pos[batch_inds[canonical_conformers_inds[canonical_conformer_acceptors_inds]]]

    return torch.sum(torch.cdist(donors_pos, acceptors_pos, p=2) < 3.3)


def save_checkpoint(epoch: int,
                    model: nn.Module,
                    optimizer,
                    config: dict,
                    save_path: str,
                    dataDims: dict):
    """

    Parameters
    ----------
    epoch
    model
    optimizer
    config
    save_path
    dataDims

    Returns
    -------

    """
    if torch.stack([torch.isfinite(p).any() for p in model.parameters()]).all():
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'dataDims': dataDims},
                   save_path)
    else:
        print("Did not save model - NaN parameters present")
        # todo add assertion here?
    return None


def weight_reset(m):
    if (isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.Conv3d)
            or isinstance(m, nn.ConvTranspose3d)):
        m.reset_parameters()


def get_n_config(model):
    """
    count parameters for a pytorch model
    :param model:
    :return:
    """
    pp = 0
    for p in list(model.parameters()):
        numm = 1
        for s in list(p.size()):
            numm = numm * s
        pp += numm
    return pp


def clean_generator_output(samples=None,  # TODO rewrite - this is a mess
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


def enforce_crystal_system(lattice_lengths, lattice_angles, sg_inds, symmetries_dict):
    """
    enforce physical bounds on cell parameters
    https://en.wikipedia.org/wiki/Crystal_system
    """  # todo double check these limits

    lattices = [symmetries_dict['lattice_type'][int(sg_inds[n])] for n in range(len(sg_inds))]

    pi_tensor = torch.tensor(torch.ones_like(lattice_lengths[0, 0]) * torch.pi)

    fixed_lengths = torch.zeros_like(lattice_lengths)
    fixed_angles = torch.zeros_like(lattice_angles)

    for i in range(len(lattice_lengths)):
        lengths = lattice_lengths[i]
        angles = lattice_angles[i]
        lattice = lattices[i]
        # enforce agreement with crystal system
        if lattice.lower() == 'triclinic':  # anything goes
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = angles * 1

        elif lattice.lower() == 'monoclinic':  # fix alpha and gamma to pi/2
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, angles[1], pi_tensor.clone() / 2,
            ), dim=- 1)
        elif lattice.lower() == 'orthorhombic':  # fix all angles at pi/2
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)
        elif lattice.lower() == 'tetragonal':  # fix all angles pi/2 and take the mean of a & b vectors
            mean_tensor = torch.mean(lengths[0:2])
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, lengths[2] * 1,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)

        elif lattice.lower() == 'hexagonal':
            # mean of ab, c is free
            # alpha beta are pi/2, gamma is 2pi/3

            mean_tensor = torch.mean(lengths[0:2])
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, lengths[2] * 1,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() * 2 / 3,
            ), dim=- 1)

        # elif lattice.lower()  == 'trigonal':

        elif lattice.lower() == 'rhombohedral':
            # mean of abc vector lengths
            # mean of all angles

            mean_tensor = torch.mean(lengths)
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, mean_tensor,
            ), dim=- 1)

            mean_angle = torch.mean(angles)
            fixed_angles[i] = torch.stack((
                mean_angle, mean_angle, mean_angle,
            ), dim=- 1)

        elif lattice.lower() == 'cubic':  # all angles 90 all lengths equal
            mean_tensor = torch.mean(lengths)
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, mean_tensor,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)
        else:
            print(lattice + ' is not a valid crystal lattice!')
            sys.exit()

    return fixed_lengths, fixed_angles


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
    """
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


def slash_batch(train_loader, test_loader, slash_fraction):
    slash_increment = max(4, int(train_loader.batch_size * slash_fraction))
    train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size - slash_increment)
    test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size - slash_increment)
    print('==============================')
    print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
    print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
    print('==============================')

    return train_loader, test_loader


def compute_gaussian_overlap(ref_types,
                             mol_batch,
                             decoded_data,
                             sigma,
                             nodewise_weights,
                             dist_to_self=False,
                             isolate_dimensions: list = None,
                             type_distance_scaling=0.1,
                             return_dists=False
                             ):
    """
    same as previous version
    except atom type differences are treated as high dimensional distances
    """
    ref_points = torch.cat((mol_batch.pos, ref_types * type_distance_scaling), dim=1)

    if dist_to_self:
        pred_points = ref_points
    else:
        pred_types = decoded_data.x * type_distance_scaling  # nodes are already weighted at 1
        pred_points = torch.cat((decoded_data.pos, pred_types), dim=1)  # assume input x has already been normalized

    if isolate_dimensions is not None:  # only compute distances over certain dimensions
        ref_points = ref_points[:, isolate_dimensions[0]:isolate_dimensions[1]]
        pred_points = pred_points[:, isolate_dimensions[0]:isolate_dimensions[1]]

    edges = radius(ref_points, pred_points,
                   # r=2 * ref_points[:, :3].norm(dim=1).amax(),  # max range encompasses largest molecule in the batch
                   # alternatively any point which will have even a small overlap - should be faster by ignoring unimportant edges, where the gradient will anyway be vanishing
                   r=4 * sigma,
                   max_num_neighbors=10000,
                   batch_x=mol_batch.batch,
                   batch_y=decoded_data.batch)  # this step is slower than before
    dists = torch.linalg.norm(ref_points[edges[1]] - pred_points[edges[0]], dim=1)
    overlap = torch.exp(-torch.pow(dists / sigma, 2))
    scaled_overlap = overlap * nodewise_weights[edges[0]]  # reweight appropriately
    nodewise_overlap = scatter(scaled_overlap,
                               edges[1],
                               reduce='sum',
                               dim_size=mol_batch.num_nodes)

    if not return_dists:
        return nodewise_overlap
    else:
        return nodewise_overlap, edges, dists


def batch_rmsd(mol_batch,
               decoded_mol_batch,
               true_node_one_hot,
               intrapoint_cutoff: float = 0.5,
               probability_threshold: float = 0.25,
               type_distance_scaling: float = 2):
    ref_types = true_node_one_hot.float()
    ref_points = torch.cat((mol_batch.pos, ref_types * type_distance_scaling), dim=1)
    pred_types = decoded_mol_batch.x * type_distance_scaling  # nodes are already weighted at 1
    pred_points = torch.cat((decoded_mol_batch.pos, pred_types), dim=1)  # assume input x has already been normalized
    nodewise_weights = decoded_mol_batch.aux_ind

    edges = radius(ref_points,
                   pred_points,
                   r=intrapoint_cutoff,
                   max_num_neighbors=10000,
                   batch_x=mol_batch.batch,
                   batch_y=decoded_mol_batch.batch)  # this step is slower than before
    dists = torch.linalg.norm(ref_points[edges[1]] - pred_points[edges[0]], dim=1)

    collect_bools = dists < intrapoint_cutoff
    inds_within_cutoff = edges[0][collect_bools]
    inside_edge_nodes = edges[1][collect_bools]
    collected_particles = pred_points[inds_within_cutoff]
    collected_particle_weights = nodewise_weights[inds_within_cutoff]
    # # confirm each output is mapped to a single input
    # a, b = torch.unique(edges[0][collect_bools], return_counts=True)
    # assert b.max() == 1
    pred_particle_weights = scatter(collected_particle_weights,
                                    inside_edge_nodes,
                                    reduce='sum',
                                    dim_size=mol_batch.num_nodes,
                                    )
    # filter here for where we do not match the scaffold (no nearby nodes, or insufficient probability mass)
    missing_particle_bools = (1 - pred_particle_weights).abs() >= probability_threshold
    complete_graph_bools = scatter((~missing_particle_bools).long(),
                                   mol_batch.batch,
                                   reduce='mul',
                                   dim_size=mol_batch.num_graphs,
                                   dim=0
                                   ).bool()
    pred_particle_points = scatter(collected_particles * collected_particle_weights[:, None],
                                   inside_edge_nodes,
                                   reduce='sum',
                                   dim=0,
                                   dim_size=mol_batch.num_nodes,
                                   )
    pred_dists = torch.linalg.norm(ref_points - pred_particle_points, dim=1)
    rmsd = scatter(pred_dists, mol_batch.batch, reduce='mean', dim_size=mol_batch.num_graphs)
    rmsd[~complete_graph_bools] = torch.nan
    pred_particle_points[missing_particle_bools] *= torch.nan

    return rmsd, pred_dists, complete_graph_bools, ~missing_particle_bools, pred_particle_points, pred_particle_weights


""" # new cheap RMSD calculation

# get radius about which to collect points for each input node

# intra_dists = torch.linalg.norm(ref_points[intra_edges[1]] - ref_points[intra_edges[0]], dim=1)
# intra_edges = radius(ref_points, ref_points,
#                # r=2 * ref_points[:, :3].norm(dim=1).amax(),  # max range encompasses largest molecule in the batch
#                # alternatively any point which will have even a small overlap - should be faster by ignoring unimportant edges, where the gradient will anyway be vanishing
#                r=4 * sigma,
#                max_num_neighbors=10000,
#                batch_x=data.batch,
#                batch_y=data.batch)  # this step is slower than before
# intrapoint_cutoff = intra_dists[intra_dists!=0].min()/2
intrapoint_cutoff = 0.6

collect_bools = dists < intrapoint_cutoff
collected_particles = pred_points[edges[0][collect_bools]]
collected_particle_weights = nodewise_weights[edges[0][collect_bools]]

# # confirm each output is mapped to a single input
# a, b = torch.unique(edges[0][collect_bools], return_counts=True)  
# assert b.max() == 1

pred_particle_weights = scatter(collected_particle_weights, 
                                edges[1][collect_bools],
                                reduce='sum',
                                dim_size=data.num_nodes,
                                )
pred_particle_points = scatter(collected_particles * collected_particle_weights[:, None],
                               edges[1][collect_bools],
                               reduce='sum',
                               dim = 0,
                               dim_size=data.num_nodes,
                               )

pred_dists = torch.linalg.norm(ref_points - pred_particle_points, dim=1)

rmsd = scatter(pred_dists, data.batch, reduce='mean', dim_size=data.num_graphs)
"""


def direction_coefficient(v):
    """
    norm vectors
    take inner product
    sum the gaussian-weighted dot product components
    """
    norms = torch.linalg.norm(v, dim=1)
    nv = v / (norms[:, None, :] + 1e-3)
    dp = torch.einsum('nik,nil->nkl', nv, nv)

    return torch.exp(-(1 - dp) ** 2).mean(-1)


def get_model_nans(model):
    if model is not None:
        nans = 0
        for parameter in model.parameters():
            nans += int(torch.sum(torch.isnan(parameter)))
        return nans
    else:
        return 0


def compute_type_evaluation_overlap(config,
                                    data,
                                    num_atom_types,
                                    decoded_data,
                                    true_nodes):
    type_overlap = compute_gaussian_overlap(
        true_nodes,
        data,
        decoded_data,
        config.autoencoder.evaluation_sigma,
        nodewise_weights=decoded_data.aux_ind,
        isolate_dimensions=[3, 3 + num_atom_types],
        type_distance_scaling=config.autoencoder.type_distance_scaling
    )
    self_type_overlap = compute_gaussian_overlap(
        true_nodes,
        data,
        data,
        config.autoencoder.evaluation_sigma,
        nodewise_weights=torch.ones(len(data.x), device=data.x.device, dtype=data.x.dtype),
        dist_to_self=True,
        isolate_dimensions=[3, 3 + num_atom_types],
        type_distance_scaling=config.autoencoder.type_distance_scaling
    )
    return self_type_overlap, type_overlap


def compute_coord_evaluation_overlap(
        config,
        data,
        decoded_data,
        true_nodes):
    coord_overlap = compute_gaussian_overlap(
        true_nodes,
        data,
        decoded_data,
        config.autoencoder.evaluation_sigma,
        nodewise_weights=decoded_data.aux_ind,
        isolate_dimensions=[0, 3],
        type_distance_scaling=config.autoencoder.type_distance_scaling
    )
    self_coord_overlap = compute_gaussian_overlap(
        true_nodes,
        data,
        data,
        config.autoencoder.evaluation_sigma,
        nodewise_weights=torch.ones(len(data.x), device=data.x.device, dtype=data.x.dtype),
        dist_to_self=True,
        isolate_dimensions=[0, 3],
        type_distance_scaling=config.autoencoder.type_distance_scaling
    )
    return coord_overlap, self_coord_overlap


def compute_full_evaluation_overlap(mol_batch,
                                    decoded_mol_batch,
                                    true_nodes,
                                    sigma=None, distance_scaling=None):
    full_overlap = compute_gaussian_overlap(
        true_nodes, mol_batch, decoded_mol_batch, sigma,
        nodewise_weights=decoded_mol_batch.aux_ind,
        type_distance_scaling=distance_scaling,
    )
    self_overlap = compute_gaussian_overlap(
        true_nodes, mol_batch, mol_batch, sigma,
        nodewise_weights=torch.ones(len(mol_batch.x), device=mol_batch.x.device, dtype=mol_batch.x.dtype),
        dist_to_self=True,
        type_distance_scaling=distance_scaling)
    return full_overlap, self_overlap


def get_node_weights(mol_batch, decoded_mol_batch, decoding, num_decoder_nodes, node_weight_temperature):
    # per-atom weights of each graph
    molwise_weight_per_swarm_point = mol_batch.num_atoms / num_decoder_nodes

    # cast to num_decoder_nodes
    weight_per_swarm_point = molwise_weight_per_swarm_point.repeat_interleave(num_decoder_nodes)

    # softmax over decoding weight dimension, adjusted by temperature
    nodewise_weights = scatter_softmax(decoding[:, -1] / node_weight_temperature,
                                       decoded_mol_batch.batch,
                                       dim=0,
                                       dim_size=decoded_mol_batch.num_nodes)

    # reweigh against the number of atoms
    nodewise_weights_tensor = nodewise_weights * mol_batch.num_atoms.repeat_interleave(
        num_decoder_nodes)

    return weight_per_swarm_point, nodewise_weights, nodewise_weights_tensor


def dict_of_tensors_to_cpu_numpy(stats):
    for key, value in stats.items():
        if torch.is_tensor(value):
            stats[key] = value.cpu().numpy()
        elif 'CrystalDataBatch' in str(type(value)):
            stats[key] = value.cpu()


def init_decoded_data(data, decoding, device, num_nodes):
    decoded_data = data.detach().clone()
    decoded_data.pos = decoding[:, :3]
    decoded_data.batch = torch.arange(data.num_graphs).repeat_interleave(num_nodes).to(device)
    return decoded_data


def test_decoder_equivariance(data: CrystalData,
                              encoding: torch.Tensor,
                              rotated_encoding: torch.Tensor,
                              rotations: torch.Tensor,
                              autoencoder: nn.Module,
                              device: Union[torch.device, str]) -> torch.Tensor:
    """
    check decoder end-to-end equivariance
    """
    '''take a given embedding and decode it'''
    decoding = autoencoder.decode(encoding)

    '''rotate embedding and decode'''
    decoding2 = autoencoder.decode(
        rotated_encoding.reshape(data.num_graphs, 3, encoding.shape[-1]))

    '''rotate first decoding and compare'''
    decoded_batch = torch.arange(data.num_graphs).repeat_interleave(autoencoder.num_decoder_nodes).to(device)
    rotated_decoding_positions = torch.cat(
        [torch.einsum('ij, kj->ki', rotations[ind], decoding[:, :3][decoded_batch == ind])
         for ind in range(data.num_graphs)])
    rotated_decoding = decoding.clone()
    rotated_decoding[:, :3] = rotated_decoding_positions
    # first three dimensions should be equivariant and all trailing invariant
    decoder_equivariance_loss = (
            torch.abs(rotated_decoding[:, :3] - decoding2[:, :3]) / (1e-3 + torch.abs(rotated_decoding[:, :3])))
    return decoder_equivariance_loss.mean(-1)


def test_encoder_equivariance(data: CrystalData,
                              rotations: torch.Tensor,
                              autoencoder) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    check encoder end-to-end equivariance
    """
    '''embed the input data then rotate the embedding'''
    encoding = autoencoder.encode(data.clone(), override_centering=True)
    rotated_encoding = torch.einsum('nij, njk->nik',
                                    rotations,
                                    encoding
                                    )  # rotate in 3D

    rotated_encoding = rotated_encoding.reshape(data.num_graphs, rotated_encoding.shape[-1] * 3)
    '''rotate the input data and embed it'''
    data.pos = torch.cat([torch.einsum('ij, kj->ki', rotations[ind], data.pos[data.batch == ind])
                          for ind in range(data.num_graphs)])
    encoding2 = autoencoder.encode(data.clone(), override_centering=True)
    encoding2 = encoding2.reshape(data.num_graphs, encoding2.shape[-1] * 3)
    '''compare the embeddings - should be identical for an equivariant embedding'''
    encoder_equivariance_loss = (torch.abs(rotated_encoding - encoding2) / torch.abs(rotated_encoding)).mean(-1)
    return encoder_equivariance_loss, encoding, rotated_encoding


def collate_decoded_data(data, decoding, num_decoder_nodes, node_weight_temperature, device):
    # generate input reconstructed as a data type
    decoded_mol_batch = init_decoded_data(data,
                                          decoding,
                                          device,
                                          num_decoder_nodes
                                          )
    # compute the distributional weight of each node
    nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = \
        get_node_weights(data, decoded_mol_batch, decoding,
                         num_decoder_nodes,
                         node_weight_temperature)
    decoded_mol_batch.aux_ind = nodewise_weights_tensor
    # input node weights are always 1 - corresponding each to an atom
    data.aux_ind = torch.ones(data.num_nodes, dtype=torch.float32, device=device)
    # get probability distribution over type dimensions
    decoded_mol_batch.x = F.softmax(decoding[:, 3:-1], dim=1)
    return decoded_mol_batch, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor


def ae_reconstruction_loss(mol_batch,
                           decoded_mol_batch,
                           graph_normed_nodewise_weights,
                           nodewise_weights_tensor,
                           num_atom_types,
                           type_distance_scaling,
                           autoencoder_sigma,
                           ):
    true_node_one_hot = F.one_hot(mol_batch.x.flatten().long(), num_classes=num_atom_types).float()

    decoder_likelihoods, input2output_edges, input2output_dists = (
        compute_gaussian_overlap(true_node_one_hot, mol_batch, decoded_mol_batch, autoencoder_sigma,
                                 nodewise_weights=decoded_mol_batch.aux_ind,
                                 type_distance_scaling=type_distance_scaling,
                                 return_dists=True
                                 ))

    # if sigma is too large, these can be > 1, so we map to the overlap of the true density with itself
    self_likelihoods = compute_gaussian_overlap(true_node_one_hot, mol_batch, mol_batch, autoencoder_sigma,
                                                nodewise_weights=mol_batch.aux_ind, dist_to_self=True,
                                                type_distance_scaling=type_distance_scaling)

    # typewise agreement for whole graph
    per_graph_true_types = scatter(
        true_node_one_hot, mol_batch.batch[:, None], dim=0, reduce='mean')
    per_graph_pred_types = scatter(
        decoded_mol_batch.x * graph_normed_nodewise_weights[:, None], decoded_mol_batch.batch[:, None], dim=0,
        reduce='sum')

    nodewise_type_loss = (
            F.binary_cross_entropy(per_graph_pred_types.clip(min=1e-6, max=1 - 1e-6), per_graph_true_types) -
            F.binary_cross_entropy(per_graph_true_types, per_graph_true_types))

    nodewise_reconstruction_loss = F.smooth_l1_loss(decoder_likelihoods, self_likelihoods, reduction='none')
    graph_reconstruction_loss = scatter(nodewise_reconstruction_loss, mol_batch.batch, reduce='mean')

    # new losses -
    # 1 penalize components for distance to nearest atom
    nearest_node_dist = scatter(input2output_dists,
                                input2output_edges[0],
                                reduce='min',
                                dim_size=decoded_mol_batch.num_nodes
                                )
    nearest_node_loss = scatter(nearest_node_dist, decoded_mol_batch.batch, reduce='mean',
                                dim_size=mol_batch.num_graphs)

    # 2 penalize area near an atom for not being a part of an exactly atom-size clump
    collect_bools = input2output_dists < 0.5
    inds_within_cutoff = input2output_edges[0][collect_bools]
    inside_edge_nodes = input2output_edges[1][collect_bools]
    collected_particle_weights = nodewise_weights_tensor[inds_within_cutoff]
    pred_particle_weights = scatter(collected_particle_weights,
                                    inside_edge_nodes,
                                    reduce='sum',
                                    dim_size=mol_batch.num_nodes,
                                    )

    nodewise_clumping_loss = F.smooth_l1_loss(pred_particle_weights, torch.ones_like(pred_particle_weights),
                                              reduction='none')
    graph_clumping_loss = scatter(nodewise_clumping_loss, mol_batch.batch, reduce='mean')

    return (nodewise_reconstruction_loss, nodewise_type_loss,
            graph_reconstruction_loss, self_likelihoods,
            nearest_node_loss, graph_clumping_loss)


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


def get_intermolecular_dists_dict(supercell_data: CrystalData,
                                  conv_cutoff: float,
                                  max_num_neighbors: int = 10000):
    dist_dict = {}
    edges_dict = construct_radial_graph(
        supercell_data.pos,
        supercell_data.batch,
        supercell_data.ptr,
        conv_cutoff,
        max_num_neighbors,
        aux_ind=supercell_data.aux_ind,
        mol_ind=supercell_data.mol_ind,
    )
    dist_dict.update(edges_dict)
    dist_dict['intermolecular_dist'] = (
        (supercell_data.pos[edges_dict['edge_index_inter'][0]] - supercell_data.pos[
            edges_dict['edge_index_inter'][1]]).pow(2).sum(
            dim=-1).sqrt())

    dist_dict['intermolecular_dist_batch'] = supercell_data.batch[edges_dict['edge_index_inter'][0]]

    dist_dict['intermolecular_dist_atoms'] = \
        [supercell_data.x[edges_dict['edge_index_inter'][0], 0].long(),
         supercell_data.x[edges_dict['edge_index_inter'][1], 0].long()]

    dist_dict['intermolecular_dist_inds'] = edges_dict['edge_index_inter']

    return dist_dict


def denormalize_generated_cell_params(
        generator_raw_samples,
        mol_data,
        asym_unit_dict):
    # denormalize the predicted cell lengths
    cell_lengths = torch.pow(mol_data.sym_mult * mol_data.mol_volume, 1 / 3)[:, None] * generator_raw_samples[:, :3]
    # rescale asymmetric units  # todo add assertions around these
    mol_positions = descale_asymmetric_unit(asym_unit_dict,
                                            generator_raw_samples[:, 6:9],
                                            mol_data.sg_ind)
    generated_samples_to_build = torch.cat(
        [cell_lengths, generator_raw_samples[:, 3:6], mol_positions, generator_raw_samples[:, 9:12]], dim=1)
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
                       variation_factor: torch.Tensor) -> torch.Tensor:
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


def coarse_crystal_filter(lj_record, lj_cutoff, packing_coeff_record, packing_cutoff):
    """filtering - samples with exactly 0 LJ energy are too diffuse, and more than CUTOFF are overlapping"""
    bad_inds = []
    bad_bools1 = lj_record == 0
    bad_bools2 = lj_record >= lj_cutoff
    bad_bools3 = packing_coeff_record >= packing_cutoff[-1]
    bad_bools4 = packing_coeff_record <= packing_cutoff[0]

    # if we got any of these, cut the sample
    good_bools = (~bad_bools1) * (~bad_bools2) * (~bad_bools3) * (~bad_bools4)
    good_inds = torch.argwhere(good_bools).flatten()

    print(f"{bad_bools1.sum()} with zero vdW, "
          f"{bad_bools2.sum()} above vdW cutoff, "
          f"{bad_bools3.sum()} outside density cutoff,"
          f"leaving {len(good_inds)} samples")

    return bad_inds, good_inds


'''
import plotly.graph_objects as go
fig = go.Figure(go.Heatmap(z=distmat)).show()
'''

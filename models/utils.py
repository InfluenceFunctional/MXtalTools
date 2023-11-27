import sys

import numpy as np
import torch
import wandb
from torch import optim, nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler as lr_scheduler

from common.geometry_calculations import cell_vol_torch
from common.utils import softmax_np, components2angle
from dataset_management.utils import update_dataloader_batch_size


def set_lr(schedulers, optimizer, lr_schedule, min_lr, err_tr, hit_max_lr):
    if lr_schedule:
        lr = optimizer.param_groups[0]['lr']
        if lr > min_lr:
            schedulers[0].step(np.mean(np.asarray(err_tr)))  # plateau scheduler

        if not hit_max_lr:
            schedulers[1].step()
        elif hit_max_lr:
            if lr > min_lr:
                schedulers[2].step()  # start reducing lr

    lr = optimizer.param_groups[0]['lr']
    return optimizer, lr


def check_convergence(record, history, convergence_eps):
    """
    check if we are converged
    condition: test loss has increased or levelled out over the last several epochs
    :return: convergence flag
    """

    converged = False
    if type(record) is list:
        try:
            record = np.concatenate(record)
        except ValueError:
            record = np.asarray(record)
    elif isinstance(record, np.ndarray):
        record = record.copy()

    if len(record) > 2 * history:
        #  all of last :history: points are above the 5% quantile of the training record
        if all(record[-history:] > np.quantile(record, 0.05)):
            converged = True
            print("Model converged, target diverging")

        criteria = np.abs(np.mean(record[-history:]) / np.mean(record[-2 * history:-history]))  # change in mean over 2 periods of history
        print('Convergence criteria at {:.3f}'.format(np.log10(criteria)))  # todo better rolling metric here - trailing exponential something
        if criteria < convergence_eps:
            converged = True
            print("Model converged, target stabilized")

    return converged


def save_model(model, optimizer):
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/model_ckpt')


def init_optimizer(optim_config, model, freeze_params=False):
    """
    initialize optimizers
    @param optim_config: config for a given optimizer
    @param model: model with params to be optimized
    @param freeze_params: whether parameters without requires_grad should be frozen
    @return: optimizer
    """

    amsgrad = True
    beta1 = optim_config.beta1  # 0.9
    beta2 = optim_config.beta2  # 0.999
    weight_decay = optim_config.weight_decay  # 0.01
    momentum = 0

    if freeze_params:
        model_params = [param for param in model.parameters() if param.requires_grad == True]
    else:
        model_params = model.parameters()

    if optim_config.optimizer == 'adam':
        optimizer = optim.Adam(model_params, amsgrad=amsgrad, lr=optim_config.init_lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif optim_config.optimizer == 'adamw':
        optimizer = optim.AdamW(model_params, amsgrad=amsgrad, lr=optim_config.init_lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif optim_config.optimizer == 'sgd':
        optimizer = optim.SGD(model_params, lr=optim_config.init_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print(optim_config.optimizer + ' is not a valid optimizer')
        sys.exit()

    return optimizer


def init_schedulers(optimizer, lr_shrink_lambda, lr_growth_lambda):
    """
    initialize a series of LR schedulers
    """
    scheduler1 = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=500,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=500
    )
    lr_lambda = lambda epoch: lr_growth_lambda
    scheduler2 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    lr_lambda2 = lambda epoch: lr_shrink_lambda
    scheduler3 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda2)

    return [scheduler1, scheduler2, scheduler3]


def softmax_and_score(raw_classwise_output, temperature=1, old_method=False, correct_discontinuity=True):
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
            softmax_output = softmax_np(raw_classwise_output.astype('float64'), temperature)[:, 1].astype('float64')  # values get too close to zero for float32
            tanned = np.tan((softmax_output - 0.5) * np.pi)
            sign = (raw_classwise_output[:, 1] > raw_classwise_output[:, 0]) * 2 - 1  # values very close to zero can realize a sign error
            return sign * np.log10(correction + np.abs(tanned))  # new factor of 1+ conditions the function about zero

        elif torch.is_tensor(raw_classwise_output):
            softmax_output = F.softmax(raw_classwise_output / temperature, dim=-1)[:, 1]
            tanned = torch.tan((softmax_output - 0.5) * torch.pi)
            sign = (raw_classwise_output[:, 1] > raw_classwise_output[:, 0]) * 2 - 1  # values very close to zero can realize a sign error
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


def reload_model(model, optimizer, path, reload_optimizer=False):
    """
    load model and state dict from path
    includes fix for potential dataparallel issue
    """
    checkpoint = torch.load(path)
    if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        if reload_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def compute_packing_coefficient(cell_params: torch.tensor, mol_volumes: torch.tensor, crystal_multiplicity: torch.tensor):
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
    canonical_conformer_acceptors_inds = torch.where(supercell_data.x[batch_inds[canonical_conformers_inds], atom_acceptor_ind] == 1)[0]
    outside_donors_inds = torch.where(supercell_data.x[batch_inds[outside_inds], atom_donor_ind] == 1)[0]

    donors_pos = supercell_data.pos[batch_inds[outside_inds[outside_donors_inds]]]
    acceptors_pos = supercell_data.pos[batch_inds[canonical_conformers_inds[canonical_conformer_acceptors_inds]]]

    return torch.sum(torch.cdist(donors_pos, acceptors_pos, p=2) < 3.3)


def save_checkpoint(epoch, model, optimizer, config, save_path):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config},
               save_path)
    return None


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
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


def clean_generator_output(samples=None, lattice_lengths=None, lattice_angles=None, mol_positions=None, mol_orientations=None, lattice_means=None, lattice_stds=None, destandardize=True, mode='soft'):
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
    elif mol_orientations.shape[-1] == 3:  # already have angles, no need to decode  # todo deprecate - we will only use spherical components in future
        if mode is not None:
            theta = enforce_1d_bound(real_mol_orientations[:, 0], x_span=torch.pi / 4, x_center=torch.pi / 4, mode=mode)[:, None]
            phi = enforce_1d_bound(real_mol_orientations[:, 1], x_span=torch.pi, x_center=0, mode=mode)[:, None]
            r_i = enforce_1d_bound(real_mol_orientations[:, 2], x_span=torch.pi, x_center=torch.pi, mode=mode)[:, None]
        else:
            theta, phi, r_i = real_mol_orientations
    r = torch.maximum(r_i, torch.ones_like(r_i) * 0.01)  # MUST be nonzero
    clean_mol_orientations = torch.cat((theta, phi, r), dim=-1)

    '''enforce physical bounds'''
    if mode is not None:
        if mode == 'soft':
            clean_lattice_lengths = F.softplus(real_lattice_lengths - 0.1) + 0.1  # smoothly enforces positive nonzero
        elif mode == 'hard':
            clean_lattice_lengths = torch.maximum(F.relu(real_lattice_lengths), torch.ones_like(real_lattice_lengths))  # harshly enforces positive nonzero

        clean_lattice_angles = enforce_1d_bound(real_lattice_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2, mode=mode)  # range from (0,pi) with 20% limit to prevent too-skinny cells
        clean_mol_positions = enforce_1d_bound(real_mol_positions, 0.5, 0.5, mode=mode)  # enforce fractional centroids between 0 and 1
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
    real_orientation_theta = components2angle(theta_encoding)
    real_orientation_phi = components2angle(mol_orientations[:, 2:4])  # unrestricted [-pi,pi
    real_orientation_r = components2angle(mol_orientations[:, 4:6]) + torch.pi  # shift from [-pi,pi] to [0, 2pi]  # want vector to have a positive norm

    # clean_mol_orientations = torch.cat((
    #     real_orientation_theta[:, None],
    #     real_orientation_phi[:, None],
    #     real_orientation_r[:, None]
    # ), dim=-1)

    return real_orientation_theta[:, None], real_orientation_phi[:, None], real_orientation_r[:, None]


def get_regression_loss(regressor, data, mean, std):
    predictions = regressor(data)[:, 0]
    targets = data.y
    return F.smooth_l1_loss(predictions, targets, reduction='none'), predictions.cpu().detach().numpy() * std + mean, targets.cpu().detach().numpy() * std + mean


def slash_batch(train_loader, test_loader, slash_fraction):
    slash_increment = max(4, int(train_loader.batch_size * slash_fraction))
    train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size - slash_increment)
    test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size - slash_increment)
    print('==============================')
    print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
    print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
    print('==============================')
    wandb.log({'batch size': train_loader.batch_size})

    return train_loader, test_loader

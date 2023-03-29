import sys

import numpy as np
import torch
from ase import Atoms
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler as lr_scheduler

from common.geometry_calculations import cell_vol_torch
from common.utils import np_softmax


def get_grad_norm(model):
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(params) == 0:
        norm = 0
    else:
        norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).cpu() for p in params]), 2.0).item()

    return norm


def set_lr(schedulers, optimizer, lr_schedule, min_lr, max_lr, err_tr, hit_max_lr):
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


def compute_F1_score(confusion_matrix, num_classes):
    true_positive = [confusion_matrix[i, i] for i in range(num_classes)]
    false_positive = [np.sum(confusion_matrix[i, :]) - confusion_matrix[i, i] for i in range(num_classes)]
    false_negative = [np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i] for i in range(num_classes)]

    accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
    recall = np.asarray([true_positive[i] / (true_positive[i] + false_positive[i]) for i in range(num_classes)])
    precision = np.asarray([true_positive[i] / (true_positive[i] + false_negative[i]) for i in range(num_classes)])
    F1 = np.asarray([2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(num_classes)])

    return accuracy, np.average(np.nan_to_num(precision, nan=0)), np.average(np.nan_to_num(recall, nan=0)), np.average(np.nan_to_num(F1, nan=0))


def compute_top_k_accuracy(config, probs, targets, X=5):
    # this actually computes the 'true positive rate' true positive / all positives
    correct_counter = np.zeros(config.dataDims['output classes'][0], dtype='uint64')
    incorrect_counter = np.zeros_like(correct_counter)

    for i in range(len(probs)):
        topXPredictions = np.argpartition(probs[i], -X)[-X:]  # not sorted
        if targets[i] in topXPredictions:
            correct_counter[targets[i]] += 1
        else:
            incorrect_counter[targets[i]] += 1

    overall_top_x_accuracy = correct_counter.sum() / (correct_counter.sum() + incorrect_counter.sum())
    by_group_top_x_accuracy = np.zeros(len(correct_counter))
    for i in range(len(correct_counter)):
        by_group_top_x_accuracy[i] = correct_counter[i] / (correct_counter[i] + incorrect_counter[i])

    return overall_top_x_accuracy, by_group_top_x_accuracy


def check_convergence(record, history, convergence_eps):
    """
    check if we are converged
    condition: test loss has increased or levelled out over the last several epochs
    :return: convergence flag
    """

    converged = False
    if type(record) == list:
        record = np.asarray(record)

    if len(record) > (history + 2):
        if all(record[-history:] >= np.amin(record)):
            converged = True
            print("Model converged, target diverging")

        criteria = np.var(record[-history:]) / np.abs(np.average(record[-history:]))
        print('Convergence criteria at {:.3f}'.format(np.log10(criteria)))  # todo better rolling metric here - trailing exponential something
        if criteria < convergence_eps:
            converged = True
            print("Model converged, target stabilized")

    return converged


def save_model(model, optimizer):
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/model_ckpt')


def crystals_to_ase_mols(crystaldata, max_ind=np.inf, highlight_aux=False, exclusion_level='distance', inclusion_distance=4):
    return [ase_mol_from_crystaldata(crystaldata, ii, highlight_canonical_conformer=highlight_aux, exclusion_level=exclusion_level, inclusion_distance=inclusion_distance)
            for ii in range(min(max_ind, crystaldata.num_graphs))]


def ase_mol_from_crystaldata(data, index=None, highlight_canonical_conformer=False, exclusion_level=None, inclusion_distance=4):
    """
    generate an ASE Atoms object from a crystaldata object, up to certain exclusions
    optionally highlight atoms in the asymmetric unit

    """
    data = data.clone().cpu().detach()
    if data.batch is not None:  # more than one crystal in the datafile
        atom_inds = torch.where(data.batch == index)[0]
    else:
        atom_inds = torch.arange(len(data.x))

    if exclusion_level == 'conformer':  # only the canonical conformer itself
        inside_inds = torch.where(data.aux_ind == 0)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = data.pos[atom_inds].cpu().detach().numpy()

    elif exclusion_level == 'unit cell':
        # assume that by construction the first Z molecules are the ones in the unit cell
        # todo THIS IS NOT ACTUALLY THE CASE - identify centroids and pick the ones inside
        inside_inds = (torch.arange(data.mol_size[index] * data.Z[index]) + data.ptr[index]).long()
        atom_inds = inside_inds
        coords = data.pos[inside_inds].cpu().detach().numpy()

    # todo mode for any atoms inside the unit cell

    elif exclusion_level == 'convolve with':  # atoms potentially in the convolutional field
        inside_inds = torch.where(data.aux_ind < 2)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = data.pos[atom_inds].cpu().detach().numpy()

    elif exclusion_level == 'distance':  # atoms within a certain distance of the conformer radius
        crystal_coords = data.pos[atom_inds]
        crystal_inds = data.aux_ind[atom_inds]

        canonical_conformer_inds = torch.where(crystal_inds == 0)[0]
        mol_centroid = crystal_coords[canonical_conformer_inds].mean(0)
        mol_radius = torch.max(torch.cdist(mol_centroid[None], crystal_coords[canonical_conformer_inds], p=2))
        in_range_inds = torch.where((torch.cdist(mol_centroid[None], crystal_coords, p=2) < (mol_radius + inclusion_distance))[0])[0]
        atom_inds = atom_inds[in_range_inds]
        coords = crystal_coords[in_range_inds].cpu().detach().numpy()
    else:
        coords = data.pos[atom_inds].cpu().detach().numpy()

    if highlight_canonical_conformer:  # highlight the atom aux index
        numbers = data.aux_ind[atom_inds].cpu().detach().numpy() + 6
    else:
        numbers = data.x[atom_inds, 0].cpu().detach().numpy()

    if index is not None:
        cell = data.T_fc[index].T.cpu().detach().numpy()
    else:
        cell = data.T_fc[0].T.cpu().detach().numpy()

    mol = Atoms(symbols=numbers, positions=coords, cell=cell)

    return mol


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


def init_schedulers(config, optimizer):
    """
    initialize a series of LR schedulers
    @param config: config for the given optimizer
    @param optimizer:
    @return: set of schedulers
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
    lr_lambda = lambda epoch: config.lr_growth_lambda
    scheduler2 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    lr_lambda2 = lambda epoch: config.lr_shrink_lambda
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
            soft_activation = np_softmax(raw_classwise_output)
            score = np.log10(soft_activation[:, 1] / soft_activation[:, 0])
            assert np.sum(np.isnan(score)) == 0
            return score
    else:
        if correct_discontinuity:
            correction = 1
        else:
            correction = 0

        if isinstance(raw_classwise_output, np.ndarray):
            softmax_output = np_softmax(raw_classwise_output.astype('float64'), temperature)[:, 1].astype('float64')  # values get too close to zero for float32
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
    volume = tracking_features[:, dataDims['tracking features dict'].index('molecule volume')]
    # radius = (3/4/np.pi * volume)**(1/3)
    # surface_area = 4*np.pi*radius**2
    # eccentricity = tracking_features[:,config.dataDims['tracking features dict'].index('molecule eccentricity')]
    # surface_area = tracking_features[:,config.dataDims['tracking features dict'].index('molecule freeSASA')]
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
    # todo: hard mode

    if mode == 'soft':
        return x_span * torch.atanh((x - x_center) / x_span) + x_center
    elif mode == 'hard':  # linear scaling to hard stop at [center-span, center+span]
        raise ValueError("'hard' bound undong not yet implemented")
    else:
        raise ValueError("bound must be of type 'soft'")


def reload_model(model, optimizer, path):
    """
    load model and state dict from path
    includes fix for potential dataparallel issue
    """
    checkpoint = torch.load(path)
    if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def compute_packing_coefficient(cell_params: torch.tensor, mol_volumes: torch.tensor, z_values: torch.tensor):
    """
    @param cell_params: cell parameters using our standard scheme 0-5 are a,b,c,alpha,beta,gamma
    @param mol_volumes: molumes in cubic angstrom of each single molecule
    @param z_values: Z value for each crystal
    @return: crystal packing coefficient
    """
    volumes_list = []
    for i in range(len(cell_params)):
        volumes_list.append(cell_vol_torch(cell_params[i, 0:3], cell_params[i, 3:6]))
    cell_volumes = torch.stack(volumes_list)
    coeffs = z_values * mol_volumes / cell_volumes
    return coeffs


def compute_num_h_bonds(supercell_data, dataDims, i):
    """
    compute the number of hydrogen bonds, up to a loose range (3.3 angstroms), and non-directionally
    @param supercell_data: crystal data
    @param dataDims: useful information
    @param i: cell index we are checking
    @return: sum of total hydrogen bonds for the canonical conformer
    """
    batch_inds = torch.arange(supercell_data.ptr[i], supercell_data.ptr[i + 1], device=supercell_data.x.device)

    # find the canonical conformers
    canonical_conformers_inds = torch.where(supercell_data.aux_ind[batch_inds] == 0)[0]
    outside_inds = torch.where(supercell_data.aux_ind[batch_inds] == 1)[0]

    # identify and count canonical conformer acceptors and intermolecular donors
    canonical_conformer_acceptors_inds = torch.where(supercell_data.x[batch_inds[canonical_conformers_inds], dataDims['atom features'].index('atom is H bond acceptor')] == 1)[0]
    outside_donors_inds = torch.where(supercell_data.x[batch_inds[outside_inds], dataDims['atom features'].index('atom is H bond donor')] == 1)[0]

    donors_pos = supercell_data.pos[batch_inds[outside_inds[outside_donors_inds]]]
    acceptors_pos = supercell_data.pos[batch_inds[canonical_conformers_inds[canonical_conformer_acceptors_inds]]]

    return torch.sum(torch.cdist(donors_pos, acceptors_pos, p=2) < 3.3)


def get_strides(n_target_bins, init_size=3):
    """
    compute the deconvolution stride size for an approximately fixed number of deconvolution steps
    required to achieve a certain output size
    @param n_target_bins: desired cubic edge length
    @param init_size: edge length of input
    @return: list of the desired strides
    """
    target_size = n_target_bins
    tolerance = -1
    converged = False
    while not converged and tolerance < 4:
        tolerance += 1
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        inds = [1, 2, 3, 4]
                        strides = [inds[i], inds[j], inds[k], inds[l]]
                        img_size = [init_size]
                        for ii, stride in enumerate(strides):
                            if stride == 1:
                                img_size += [img_size[ii] + 2]
                            elif stride == 2:
                                img_size += [img_size[ii] + img_size[ii] + 1]
                            elif stride == 3:
                                img_size += [img_size[ii] + 2 * img_size[ii]]
                            elif stride == 4:
                                img_size += [img_size[ii] + 3 * img_size[ii] - 1]

                        if (img_size[-1] == target_size - tolerance):
                            converged = True
                        if converged:
                            # print(strides)
                            # print(img_size[-1])
                            break
                    if converged:
                        break
                if converged:
                    break
            if converged:
                break

    for n in range(tolerance):
        strides += [1]

    if converged:
        return strides, img_size[-1] + 2 * tolerance
    else:
        assert False, 'could not manage this resolution with current strided setup'


def save_checkpoint(epoch, model, optimizer, config, model_name):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config},
               "../models/" + model_name)
    return None

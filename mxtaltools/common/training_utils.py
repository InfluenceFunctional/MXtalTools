import os
import sys
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
from scipy.stats import linregress
from torch import nn as nn, optim
from torch.optim import lr_scheduler as lr_scheduler

from mxtaltools.common.utils import flatten_dict, namespace2dict
from mxtaltools.dataset_utils.utils import update_dataloader_batch_size
from mxtaltools.models.graph_models.embedding_regression_models import EquivariantEmbeddingRegressor, \
    InvariantEmbeddingRegressor
from mxtaltools.models.task_models.autoencoder_models import Mo3ENet
from mxtaltools.models.task_models.crystal_models import MolecularCrystalModel
from mxtaltools.models.task_models.generator_models import CrystalGenerator
from mxtaltools.models.task_models.mol_classifier import MoleculeClusterClassifier
from mxtaltools.models.task_models.regression_models import MoleculeScalarRegressor


def instantiate_models(config: Namespace,
                       dataDims: dict,
                       model_names,
                       autoencoder_type_index,
                       sym_info,
                       compile=False) -> dict:
    print("Initializing model(s) for " + config.mode)
    models_dict = {}
    if config.mode in ['gan', 'search', 'generator']:
        assert config.model_paths.autoencoder is not None, "Must supply a pretrained encoder for the generator!"
        models_dict['generator'] = CrystalGenerator(config.seeds.model,
                                                    config.generator.model,
                                                    embedding_dim=config.autoencoder.model.bottleneck_dim,
                                                    z_prime=1,
                                                    sym_info=sym_info
                                                    )
        models_dict['discriminator'] = MolecularCrystalModel(
            config.seeds.model,
            config.discriminator.model,
            dataDims['atom_features'],
            dataDims['molecule_features'],
            output_dim=3,
            node_standardization_tensor=dataDims['node_standardization_vector'],
            graph_standardization_tensor=dataDims['graph_standardization_vector'])
    if config.mode == 'discriminator':
        models_dict['generator'] = nn.Linear(1, 1)
        models_dict['discriminator'] = MolecularCrystalModel(
            config.seeds.model,
            config.discriminator.model,
            dataDims['atom_features'],
            dataDims['molecule_features'],
            output_dim=3,
            node_standardization_tensor=dataDims['node_standardization_vector'],
            graph_standardization_tensor=dataDims['graph_standardization_vector'])
    if config.mode == 'regression' or config.model_paths.regressor is not None:
        models_dict['regressor'] = MoleculeScalarRegressor(
            config.regressor.model,
            dataDims['atom_features'],
            dataDims['molecule_features'],
            dataDims['node_standardization_vector'],
            dataDims['graph_standardization_vector'],
            torch.tensor([dataDims['target_mean'], dataDims['target_std']]),
            config.seeds.model,
        )
    if config.mode == 'autoencoder' or config.model_paths.autoencoder is not None:
        models_dict['autoencoder'] = Mo3ENet(
            config.seeds.model,
            config.autoencoder.model,
            int(torch.sum(autoencoder_type_index != -1)),
            autoencoder_type_index,
            config.autoencoder.molecule_radius_normalization,
            protons_in_input=not config.autoencoder.filter_protons
        )
    if config.mode == 'embedding_regression' or config.model_paths.embedding_regressor is not None:
        models_dict['autoencoder'] = Mo3ENet(
            config.seeds.model,
            config.autoencoder.model,
            int(torch.sum(autoencoder_type_index != -1)),
            autoencoder_type_index,
            config.autoencoder.molecule_radius_normalization,
            protons_in_input=not config.autoencoder.filter_protons
        )
        for param in models_dict['autoencoder'].parameters():  # freeze encoder
            param.requires_grad = False
        config.embedding_regressor.model.bottleneck_dim = config.autoencoder.model.bottleneck_dim
        models_dict['embedding_regressor'] = EquivariantEmbeddingRegressor(
            config.seeds.model,
            config.embedding_regressor.model,
            num_targets=config.embedding_regressor.num_targets,
            prediction_type=config.embedding_regressor.prediction_type,
        )
        assert config.model_paths.autoencoder is not None  # must preload the encoder
    if config.mode == 'polymorph_classification':
        models_dict['polymorph_classifier'] = MoleculeClusterClassifier(
            config.seeds.model,
            config.polymorph_classifier.model,
            config.polymorph_classifier.num_output_classes,
            dataDims['atom_features'],
            dataDims['molecule_features'],
            dataDims['node_standardization_vector'],
            dataDims['graph_standardization_vector'],
        )
    if config.mode == 'proxy_discriminator':
        if config.proxy_discriminator.embedding_type == 'autoencoder':
            config.proxy_discriminator.model.bottleneck_dim = 4 * config.autoencoder.model.bottleneck_dim

        elif config.proxy_discriminator.embedding_type == 'principal_axes':
            config.proxy_discriminator.model.bottleneck_dim = 3 * 4

        elif config.proxy_discriminator.embedding_type == 'principal_moments':
            config.proxy_discriminator.model.bottleneck_dim = 3 * 4

        elif config.proxy_discriminator.embedding_type == 'mol_volume':
            config.proxy_discriminator.model.bottleneck_dim = 3 * 4

        elif config.proxy_discriminator.embedding_type is None:
            config.proxy_discriminator.model.bottleneck_dim = 3 * 4

        else:
            assert False

        conditions_dim = 12
        models_dict['proxy_discriminator'] = InvariantEmbeddingRegressor(
            config.seeds.model,
            config.proxy_discriminator.model,
            num_targets=1,
            conditions_dim=conditions_dim,
        )
        assert config.model_paths.autoencoder is not None  # must preload the encoder

    null_models = {name: nn.Linear(1, 1) for name in model_names if
                   name not in models_dict.keys()}  # initialize null models
    models_dict.update(null_models)

    # # not currently working on any platform
    # # compile models
    # if compile:
    #     for key in models_dict.keys():
    #         models_dict[key].compile_self()

    if config.device.lower() == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        for model in models_dict.values():
            model.cuda()

    return models_dict


def update_stats_dict(dictionary: dict, keys, values, mode='append'):
    """
    Append/extend dict of key:list pairs or one at a time

    Parameters
    ----------
    dictionary
    keys
    values
    mode: 'append' or 'extend'

    Returns
    -------
    updated_dictionary
    """

    if isinstance(keys, list):
        for key, value in zip(keys, values):
            if key not in dictionary.keys():
                dictionary[key] = []

            if (mode == 'append') or ('crystaldata' in str(type(value)).lower()):
                dictionary[key].append(value)
            elif mode == 'extend':
                dictionary[key].extend(value)
    else:
        key, value = keys, values
        if key not in dictionary.keys():
            dictionary[key] = []

        if mode == 'append':
            dictionary[key].append(value)
        elif mode == 'extend':
            dictionary[key].extend(value)

    return dictionary


def make_sequential_directory(yaml_path, workdir):  # make working directory
    """
    make a new working directory labelled by the time & date
    hopefully does not overlap with any other workdirs
    :return:
    """
    run_identifier = str(yaml_path).split('.yaml')[0].split('configs')[1].replace('\\', '_').replace(
        '/', '_') + '_' + datetime.today().strftime("%d-%m-%H-%M-%S")
    working_directory = workdir + run_identifier
    os.mkdir(working_directory)
    return run_identifier, working_directory


def flatten_wandb_params(config):
    """Initialize "flat" config for wandb parameter logging"""
    flat_config_dict = flatten_dict(namespace2dict(config.__dict__), separator='_')
    for key in flat_config_dict.keys():
        if 'path' in str(type(flat_config_dict[key])).lower():
            flat_config_dict[key] = str(flat_config_dict[key])
    config.__dict__.update(flat_config_dict)
    return config


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
        optimizer_name = 'adam'
        init_lr = 1e-3
    else:
        beta1 = optim_config.beta1  # 0.9
        beta2 = optim_config.beta2  # 0.999
        weight_decay = optim_config.weight_decay  # 0.01
        optimizer_name = optim_config.optimizer
        init_lr = optim_config.init_lr

    amsgrad = amsgrad

    if model_name == 'autoencoder' and hasattr(model, 'encoder'):
        if freeze_params:
            assert False, "params freezing not implemented for autoencoder"

        params_dict = [
            {'params': list(model.scalarizer.parameters()) + list(model.encoder.parameters()), 'lr': optim_config.encoder_init_lr},
            {'params': model.decoder.parameters(), 'lr': optim_config.decoder_init_lr}
        ]

    else:
        if freeze_params:
            params_dict = [param for param in model.parameters() if param.requires_grad == True]
        else:
            params_dict = model.parameters()

    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(params_dict, amsgrad=amsgrad, lr=init_lr, betas=(beta1, beta2),
                               weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(params_dict, amsgrad=amsgrad, lr=init_lr, betas=(beta1, beta2),
                                weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
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


def slash_batch(train_loader, test_loader, slash_fraction):
    slash_increment = max(4, int(train_loader.batch_size * slash_fraction))
    train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size - slash_increment)
    test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size - slash_increment)
    print('==============================')
    print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
    print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
    print('==============================')

    return train_loader, test_loader


def get_model_nans(model):
    if model is not None:
        nans = 0
        for parameter in model.parameters():
            nans += int(torch.sum(torch.isnan(parameter)))
        return nans
    else:
        return 0

import os
import sys
import threading
import time
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
from scipy.stats import linregress
from torch import nn as nn, optim
from torch.optim import lr_scheduler as lr_scheduler

from mxtaltools.common.utils import flatten_dict, namespace2dict
from mxtaltools.dataset_utils.utils import update_dataloader_batch_size
from mxtaltools.models.task_models.autoencoder_models import Mo3ENet
from mxtaltools.models.task_models.crystal_models import MolecularCrystalModel
from mxtaltools.models.task_models.regression_models import MoleculeScalarRegressor

class OOMRetry:
    """
    Purely functional OOM handler.
    Expects a *mutable reference* to batch size, e.g. [batch_size].
    On OOM: scales bs_ref[0] *= factor, cleans CUDA, retries.
    """

    def __init__(self, bs_ref, factor=0.75, min_bs=1, context=""):
        self.bs_ref = bs_ref
        self.factor = factor
        self.min_bs = min_bs
        self.context = context

    def __enter__(self):
        self._orig = self.bs_ref[0]
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            return False  # nothing happened

        msg = str(exc).lower()
        if exc_type is RuntimeError and "out of memory" in msg:
            old = self.bs_ref[0]
            new = max(self.min_bs, int(old * self.factor))
            self.bs_ref[0] = new

            print(f"[OOMRetry:{self.context}] {old} → {new} due to OOM")

            if new <= self.min_bs:
                print(f"[OOMRetry:{self.context}] minimum batch reached; re-raising")
                return False

            # cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            return True  # suppress exception → retry the block

        return False


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
            if isinstance(value, int) or isinstance(value, float):
                value = np.array(value)[None]
            if (mode == 'append') or ('batch' in str(type(value)).lower()):
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
            {'params': list(model.scalarizer.parameters()) + list(model.encoder.parameters()),
             'lr': optim_config.encoder_init_lr},
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
    # todo we load a config here as well as weights so we will have to change how we do this in an upcoming pytorch release when they get rid of this option
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if list(checkpoint['model_state_dict'])[0][
       0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        if reload_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def load_molecule_scalar_regressor(checkpoint_path, device):
    """script to reload a regression model for molecule scalar properties"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
    #opt_config = model_config.optimizer
    model_config = model_config.model
    r_dataDims = checkpoint['dataDims']
    model = MoleculeScalarRegressor(seed=12345,
                                    config=model_config,
                                    atom_features=r_dataDims['atom_features'],
                                    molecule_features=r_dataDims['molecule_features'],
                                    )
    for param in model.parameters():  # freeze model
        param.requires_grad = False

    model, _ = reload_model(model, device=device, optimizer=None,
                            path=checkpoint_path)
    model.eval()
    model.to(device)
    return model


def load_crystal_score_model(checkpoint_path, device):
    """script to reload a regression model for molecule scalar properties"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
    #opt_config = model_config.optimizer
    model_config = model_config.model
    r_dataDims = checkpoint['dataDims']
    model = MolecularCrystalModel(
        12345,
        model_config,
        r_dataDims['atom_features'],
        r_dataDims['molecule_features'],
        output_dim=3,
        node_standardization_tensor=r_dataDims['node_standardization_vector'],
        graph_standardization_tensor=r_dataDims['graph_standardization_vector']
    )
    for param in model.parameters():  # freeze model
        param.requires_grad = False

    model, _ = reload_model(model, device=device, optimizer=None,
                            path=checkpoint_path)
    model.eval()
    model.to(device)
    return model




def load_molecule_autoencoder(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
    #opt_config = model_config.optimizer
    model_config = model_config.model
    r_dataDims = checkpoint['dataDims']
    allowed_types = r_dataDims['allowed_atom_types']
    num_atom_types = len(allowed_types)

    type_translation_index = np.zeros(np.array(allowed_types).max() + 1) - 1
    for ind, atype in enumerate(allowed_types):
        type_translation_index[atype] = ind
    autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')

    model = Mo3ENet(seed=12345,
                    config=model_config,
                    num_atom_types=num_atom_types,
                    atom_embedding_vector=autoencoder_type_index,
                    radial_normalization=1,  # overwritten on reload
                    protons_in_input=True,  # overwritten on reload
                    )
    for param in model.parameters():  # freeze model
        param.requires_grad = False

    model, _ = reload_model(model, device=device, optimizer=None,
                            path=checkpoint_path)
    model.eval()
    model.to(device)
    return model


def enable_dropout(model):
    """Enable dropout layers in evaluation mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()  # Enable dropout during inference

    return model


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


def spoof_gpu_memory():
    """Dynamically allocate memory only when needed."""
    total_mem = torch.cuda.get_device_properties(0).total_memory
    target_mem = 0.5 * total_mem

    while True:
        allocated = torch.cuda.memory_allocated()

        if allocated < target_mem:
            extra_needed = target_mem - allocated
            _ = torch.ones((int(extra_needed // 4),), dtype=torch.float32, device="cuda")  # 4 bytes per float32

        time.sleep(10)


def spoof_gpu_compute():
    stream = torch.cuda.Stream(priority=-1)  # Low-priority stream
    util_threshold = 0.4
    util_sleep = 0.5
    check_sleep = 5
    size = 12000
    with torch.cuda.stream(stream):
        while True:
            util = torch.cuda.utilization(0)  # GPU utilization in %
            if util < util_threshold * 100:
                A = torch.randn((size, size), device="cuda")
                B = torch.randn((size, size), device="cuda")
                for _ in range(100):  # Keeps GPU active for longer
                    A = torch.sin(A @ B) + torch.cos(B @ A)

                time.sleep(util_sleep)  # Avoid excessive interference
            else:
                time.sleep(check_sleep)  # Wait before checking again


def spoof_usage():
    threading.Thread(target=spoof_gpu_memory, daemon=True).start()
    threading.Thread(target=spoof_gpu_compute, daemon=True).start()

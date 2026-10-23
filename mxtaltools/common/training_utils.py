"""Helpers for model training: batch-size handling on out-of-memory errors, optimizer and
learning-rate scheduler construction, convergence checks, checkpoint saving and reloading,
loaders for frozen pretrained models, and background GPU load generators."""
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
    Context manager that shrinks a shared batch size when the managed block runs out of memory.

    The batch size is held in a mutable container, e.g. ``bs_ref = [batch_size]``. On exit with
    an exception whose type is exactly ``RuntimeError`` (subclasses are not matched) and whose
    lower-cased message contains ``'out of memory'``, ``bs_ref[0]`` is set to
    ``max(min_bs, int(bs_ref[0] * factor))`` and a message is printed. If the new value is
    greater than ``min_bs``, ``gc.collect()`` and, when CUDA is available,
    ``torch.cuda.empty_cache()`` and ``torch.cuda.ipc_collect()`` are called and the exception
    is suppressed. If the new value equals ``min_bs``, the exception propagates (``bs_ref[0]``
    has still been updated). Any other exception propagates unchanged. The block is not
    re-executed by this class; the caller loops to retry with the new ``bs_ref[0]``.

    Parameters
    ----------
    bs_ref : list
        Single-element container holding the current batch size; ``bs_ref[0]`` is modified in
        place.
    factor : float, optional
        Multiplier applied to the batch size on each handled error. Default 0.75.
    min_bs : int, optional
        Lower bound for the batch size; reaching it re-raises the error. Default 1.
    context : str, optional
        Label included in the printed messages. Default ``''``.
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
    Add values to a dict of lists, for one key or for several keys at a time.

    Parameters
    ----------
    dictionary : dict
        Mapping of key to list. Modified in place; a missing key is created with an empty list.
    keys : list or hashable
        If a ``list``, each element is a key paired with the corresponding element of
        ``values``. Any other object (including a tuple) is used as a single key and must be
        hashable.
    values : sequence or object
        The value(s) to add. In the list-of-keys form, a Python ``int`` or ``float`` value is
        first wrapped into a numpy array of shape [1].
    mode : str, optional
        ``'append'`` appends each value to its list; ``'extend'`` extends the list with the
        value's elements. In the list-of-keys form, a value whose ``str(type(value))``
        contains ``'batch'`` (case-insensitive) is always appended. Otherwise any other mode
        adds nothing. Default ``'append'``.

    Returns
    -------
    dict
        The same ``dictionary`` object.
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
    Create a new run directory named from the config path and the current date and time.

    The run identifier is taken from the part of ``str(yaml_path)`` before ``'.yaml'``: the
    text after the first occurrence of ``'configs'`` (up to any second occurrence), with
    backslashes and forward slashes replaced by ``'_'``, followed by ``'_'`` and the current
    local time formatted as ``'%d-%m-%H-%M-%S'``.

    Parameters
    ----------
    yaml_path : str or path-like
        Config file path; must contain ``'configs'``.
    workdir : str
        Prefix to which the identifier is appended by string concatenation (no separator is
        inserted).

    Returns
    -------
    run_identifier : str
        The identifier described above.
    working_directory : str
        ``workdir + run_identifier``, created with ``os.mkdir``.

    Raises
    ------
    IndexError
        If ``'configs'`` does not occur in ``yaml_path``.
    FileExistsError
        If the directory already exists.
    """
    run_identifier = str(yaml_path).split('.yaml')[0].split('configs')[1].replace('\\', '_').replace(
        '/', '_') + '_' + datetime.today().strftime("%d-%m-%H-%M-%S")
    working_directory = workdir + run_identifier
    os.mkdir(working_directory)
    return run_identifier, working_directory


def flatten_wandb_params(config):
    """
    Add flattened copies of all nested config entries to a config namespace, in place.

    ``config.__dict__`` is converted to a nested dict (nested namespaces become dicts; the
    original nested namespaces are not modified) and flattened with ``'_'`` joining the key
    path, list elements contributing their index as a key component (e.g.
    ``optimizer_init_lr``, ``layers_0``). Values whose type name contains ``'path'`` (e.g.
    ``pathlib.Path``) are converted to ``str``. The flattened entries are then written onto
    ``config`` with ``__dict__.update``: the original nested attributes remain alongside the
    new flat keys, and top-level non-nested attributes are overwritten with their (possibly
    stringified) values.

    Parameters
    ----------
    config : argparse.Namespace
        Config namespace, possibly nested. Modified in place.

    Returns
    -------
    argparse.Namespace
        The same ``config`` object.
    """
    flat_config_dict = flatten_dict(namespace2dict(config.__dict__), separator='_')
    for key in flat_config_dict.keys():
        if 'path' in str(type(flat_config_dict[key])).lower():
            flat_config_dict[key] = str(flat_config_dict[key])
    config.__dict__.update(flat_config_dict)
    return config


def set_lr(schedulers, optimizer, optimizer_config, err_tr, hit_max_lr, override_lr=None):
    """
    Step the learning-rate schedulers, or set an explicit learning rate.

    If ``optimizer_config.lr_schedule`` is truthy and ``override_lr`` is None, then, with
    ``lr`` the learning rate of the first parameter group before stepping: ``schedulers[0]``
    (plateau) is stepped with ``np.mean(err_tr)`` when ``lr > optimizer_config.min_lr``; then
    ``schedulers[1]`` (growth) is stepped if ``hit_max_lr`` is False, otherwise
    ``schedulers[2]`` (shrink) is stepped when ``lr > optimizer_config.min_lr``. If
    ``override_lr`` is not None, every parameter group's learning rate is set to it. Otherwise
    nothing is changed.

    Parameters
    ----------
    schedulers : list
        ``[plateau, growth, shrink]`` schedulers as returned by :func:`init_scheduler`.
    optimizer : torch.optim.Optimizer
        Modified in place.
    optimizer_config : argparse.Namespace
        Reads ``lr_schedule`` and ``min_lr``.
    err_tr : array-like
        Training losses; their mean is passed to the plateau scheduler.
    hit_max_lr : bool
        If True, the shrink scheduler is stepped instead of the growth scheduler.
    override_lr : float, optional
        Learning rate to set on all parameter groups.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The same optimizer object.
    lr : float
        Learning rate of the first parameter group after the update.
    """
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
    Decide whether training has converged from per-epoch test (and optionally train) losses.

    Returns False unless ``epoch > minimum_epochs + 1``. Otherwise a list record is reduced to
    an array of its elements' ``.mean()`` values and an ndarray record is copied, then:

    1. If the test record contains any NaN, returns True.
    2. The last ``history`` test values are taken; when both ``history`` and
       ``minimum_epochs`` exceed 20, values at or above their 0.95 quantile are dropped. A
       linear regression of ``log10`` of these values against their position is fitted; if
       its slope is greater than ``-convergence_eps``, a message is printed and True is
       returned.
    3. If ``train_record`` is given and the mean of ``test_record / train_record`` over the
       last ``history`` entries exceeds ``overfit_tolerance``, a message is printed and True
       is returned.

    Parameters
    ----------
    test_record : list or numpy.ndarray
        Per-epoch test losses; list elements must support ``.mean()``.
    history : int
        Number of trailing epochs considered.
    convergence_eps : float
        Threshold on the log10-loss slope per epoch.
    epoch : int
        Current epoch.
    minimum_epochs : int
        Convergence is not reported before ``epoch > minimum_epochs + 1``.
    overfit_tolerance : float
        Threshold on the mean test/train loss ratio.
    train_record : list or numpy.ndarray, optional
        Per-epoch train losses, in the same form as ``test_record``.

    Returns
    -------
    bool
        True if any criterion above is met, otherwise False.
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
    Build an Adam, AdamW or SGD optimizer for a model.

    With ``optim_config`` None: Adam, learning rate 1e-3, betas (0.9, 0.99), weight decay 0.01.
    Otherwise ``beta1``, ``beta2``, ``weight_decay``, ``optimizer`` (name, case-insensitive:
    ``'adam'``, ``'adamw'`` or ``'sgd'``) and ``init_lr`` are read from ``optim_config``.

    For ``model_name == 'autoencoder'`` and a model with an ``encoder`` attribute, two parameter
    groups are used: the ``scalarizer`` and ``encoder`` parameters with learning rate
    ``optim_config.encoder_init_lr``, and the ``decoder`` parameters with learning rate
    ``optim_config.decoder_init_lr``. Otherwise all model parameters are used or, if
    ``freeze_params`` is True, only those with ``requires_grad`` True.

    Parameters
    ----------
    model_name : str
        Only compared with ``'autoencoder'``.
    optim_config : argparse.Namespace or None
        Optimizer settings, as above.
    model : torch.nn.Module
        Model whose parameters are optimized.
    amsgrad : bool, optional
        Passed to Adam and AdamW. Default False.
    freeze_params : bool, optional
        If True, parameters with ``requires_grad`` False are excluded. For the autoencoder
        parameter grouping it must be False (an ``AssertionError`` is raised otherwise).
        Default False.

    Returns
    -------
    torch.optim.Optimizer

    Notes
    -----
    An unrecognized optimizer name prints a message and calls ``sys.exit()``. ``momentum`` is
    assigned only when ``optim_config`` is None, so selecting ``'sgd'`` through a non-None
    ``optim_config`` raises ``UnboundLocalError``.
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
    Build the three learning-rate schedulers used by :func:`set_lr`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer the schedulers act on.
    optimizer_config : argparse.Namespace or None
        Reads ``lr_shrink_lambda``, ``lr_growth_lambda`` and ``use_plateau_scheduler``. If
        None, both lambdas are 1 and ``use_plateau_scheduler`` is False.

    Returns
    -------
    list
        ``[plateau, growth, shrink]``. ``plateau`` is a ``ReduceLROnPlateau`` in mode
        ``'min'``: if ``use_plateau_scheduler``, with factor 0.5, patience 500, relative
        threshold 1e-4 and cooldown 500; otherwise with factor 1e-5, patience and cooldown
        5,000,000, threshold 1e-8 and ``min_lr=1``. ``growth`` and ``shrink`` are
        ``MultiplicativeLR`` schedulers that multiply the learning rate by
        ``lr_growth_lambda`` and ``lr_shrink_lambda`` respectively on each step.
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
    Load model (and optionally optimizer) state from a checkpoint file.

    The checkpoint is loaded with ``torch.load(path, map_location=device,
    weights_only=False)``. If the first key of ``checkpoint['model_state_dict']`` starts with
    ``'module'``, the first 7 characters (``'module.'``) are removed from every key before
    ``model.load_state_dict`` (strict) is called.

    Parameters
    ----------
    model : torch.nn.Module
        Receives the weights in place.
    device : str or torch.device
        ``map_location`` for loading.
    optimizer : torch.optim.Optimizer or None
        Optimizer to restore when ``reload_optimizer`` is True.
    path : str or path-like
        Checkpoint file: a dict with ``'model_state_dict'`` and, when the optimizer is
        restored, ``'optimizer_state_dict'``.
    reload_optimizer : bool, optional
        If True and ``optimizer`` is not None, ``checkpoint['optimizer_state_dict']`` is
        loaded into it. Default False.

    Returns
    -------
    model : torch.nn.Module
        The same model object.
    optimizer : torch.optim.Optimizer or None
        The same optimizer object.
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
    """
    Build a frozen ``MoleculeScalarRegressor`` from a checkpoint.

    The model is constructed with seed 12345 from ``checkpoint['config']['model']`` and
    ``checkpoint['dataDims']['atom_features']`` and ``['molecule_features']``; all parameters
    are set to ``requires_grad=False``, the weights are loaded with :func:`reload_model`, and
    the model is put in eval mode and moved to ``device``.

    Parameters
    ----------
    checkpoint_path : str or path-like
        Checkpoint in the layout written by :func:`save_checkpoint`.
    device : str or torch.device
        Device for ``map_location`` and for the returned model.

    Returns
    -------
    MoleculeScalarRegressor
    """
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
    """
    Build a frozen ``MolecularCrystalModel`` with three outputs from a checkpoint.

    The model is constructed with seed 12345, ``checkpoint['config']['model']``,
    ``output_dim=3`` and, from ``checkpoint['dataDims']``, ``atom_features``,
    ``molecule_features``, ``node_standardization_vector`` and
    ``graph_standardization_vector``; all parameters are set to ``requires_grad=False``, the
    weights are loaded with :func:`reload_model`, and the model is put in eval mode and moved
    to ``device``.

    Parameters
    ----------
    checkpoint_path : str or path-like
        Checkpoint in the layout written by :func:`save_checkpoint`.
    device : str or torch.device
        Device for ``map_location`` and for the returned model.

    Returns
    -------
    MolecularCrystalModel
    """
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
    """
    Build a frozen ``Mo3ENet`` molecule autoencoder from a checkpoint.

    ``checkpoint['dataDims']['allowed_atom_types']`` (atomic numbers) sets ``num_atom_types``
    and an int64 CPU lookup of length ``max(allowed_atom_types) + 1`` mapping each allowed
    atomic number to its position in that list (-1 elsewhere). The model is constructed with
    seed 12345 and ``checkpoint['config']['model']``; the ``radial_normalization=1`` and
    ``protons_in_input=True`` constructor arguments are placeholders whose registered buffers,
    like the lookup buffer, are replaced by the checkpoint's values when the weights are
    loaded with :func:`reload_model`. All parameters are set to ``requires_grad=False`` and
    the model is put in eval mode and moved to ``device``.

    Parameters
    ----------
    checkpoint_path : str or path-like
        Checkpoint in the layout written by :func:`save_checkpoint`.
    device : str or torch.device
        Device for ``map_location`` and for the returned model.

    Returns
    -------
    Mo3ENet
    """
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
    """Put every ``torch.nn.Dropout`` submodule of ``model`` in train mode, in place, and return ``model``."""
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
    Save a training checkpoint with ``torch.save``, unless a parameter tensor is entirely non-finite.

    The file holds a dict with keys ``'epoch'``, ``'model_state_dict'``,
    ``'optimizer_state_dict'``, ``'config'`` and ``'dataDims'``. It is written only if every
    parameter tensor of ``model`` contains at least one finite element; otherwise a message is
    printed and nothing is written.

    Parameters
    ----------
    epoch : int
        Stored under ``'epoch'``.
    model : torch.nn.Module
        Its ``state_dict()`` is stored.
    optimizer : torch.optim.Optimizer
        Its ``state_dict()`` is stored.
    config : dict
        Stored as-is; the loaders in this module read ``config['model']``.
    save_path : str
        Destination file path.
    dataDims : dict
        Stored as-is; the loaders in this module read model dimensions from it.

    Returns
    -------
    None
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
    """Call ``m.reset_parameters()`` if ``m`` is a ``Conv2d``, ``Linear``, ``Conv3d`` or ``ConvTranspose3d`` module."""
    if (isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.Conv3d)
            or isinstance(m, nn.ConvTranspose3d)):
        m.reset_parameters()


def get_n_config(model):
    """Return the total number of parameter elements in ``model``, including parameters with ``requires_grad`` False."""
    pp = 0
    for p in list(model.parameters()):
        numm = 1
        for s in list(p.size()):
            numm = numm * s
        pp += numm
    return pp


def slash_batch(train_loader, test_loader, slash_fraction):
    """
    Rebuild the train and test loaders with a smaller batch size.

    Both batch sizes are reduced by ``max(4, int(train_loader.batch_size * slash_fraction))``,
    computed from the train loader. The new loaders come from
    ``update_dataloader_batch_size`` (same dataset, ``num_workers``, ``pin_memory`` and
    ``drop_last``, with ``shuffle=True`` for both). No lower bound is applied to the new batch
    sizes. A message is printed.

    Parameters
    ----------
    train_loader, test_loader : torch_geometric.loader.DataLoader
        Loaders to rebuild.
    slash_fraction : float
        Fraction of the train batch size to remove (at least 4 is removed).

    Returns
    -------
    train_loader, test_loader : torch_geometric.loader.DataLoader
        New loader objects.
    """
    slash_increment = max(4, int(train_loader.batch_size * slash_fraction))
    train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size - slash_increment)
    test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size - slash_increment)
    print('==============================')
    print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
    print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
    print('==============================')

    return train_loader, test_loader


def get_model_nans(model):
    """Return the number of NaN elements across all parameters of ``model``, or 0 if ``model`` is None."""
    if model is not None:
        nans = 0
        for parameter in model.parameters():
            nans += int(torch.sum(torch.isnan(parameter)))
        return nans
    else:
        return 0


def spoof_gpu_memory():
    """
    Loop indefinitely, topping torch-allocated CUDA memory up to half of device 0's total memory.

    Every 10 s, if ``torch.cuda.memory_allocated()`` is below half of
    ``torch.cuda.get_device_properties(0).total_memory``, a float32 tensor sized to the
    difference is allocated on CUDA and kept referenced until the next such allocation. Never
    returns.
    """
    total_mem = torch.cuda.get_device_properties(0).total_memory
    target_mem = 0.5 * total_mem

    while True:
        allocated = torch.cuda.memory_allocated()

        if allocated < target_mem:
            extra_needed = target_mem - allocated
            _ = torch.ones((int(extra_needed // 4),), dtype=torch.float32, device="cuda")  # 4 bytes per float32

        time.sleep(10)


def spoof_gpu_compute():
    """
    Loop indefinitely, running matrix products on CUDA while device 0's utilization is low.

    Work is issued on a ``torch.cuda.Stream`` created with ``priority=-1``. Each cycle, if
    ``torch.cuda.utilization(0)`` is below 40 (percent), two 12000 x 12000 ``torch.randn``
    matrices are created on CUDA and ``A = sin(A @ B) + cos(B @ A)`` is iterated 100 times,
    followed by a 0.5 s sleep; otherwise the loop sleeps 5 s. Never returns.
    """
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
    """Start :func:`spoof_gpu_memory` and :func:`spoof_gpu_compute` in daemon threads and return immediately."""
    threading.Thread(target=spoof_gpu_memory, daemon=True).start()
    threading.Thread(target=spoof_gpu_compute, daemon=True).start()

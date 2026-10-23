"""Build the dictionary of named models for a training mode from a config namespace and dataset dimensions."""
from argparse import Namespace

import torch
from torch import nn as nn

from mxtaltools.models.task_models.embedding_regression_models import EquivariantEmbeddingRegressor, \
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
    """
    Instantiate the models required by ``config.mode``.

    Prints ``"Initializing model(s) for <mode>"``, then builds models by the
    following rules (a later rule overwrites an earlier entry with the same key):

    - ``mode`` in ``'gan'``, ``'search'``, ``'generator'``: ``'generator'``
      (:class:`CrystalGenerator` with ``embedding_dim =
      config.autoencoder.model.bottleneck_dim``, ``z_prime=1``) and
      ``'discriminator'`` (:class:`MolecularCrystalModel`, ``output_dim=3``).
    - ``mode == 'discriminator'``: ``'discriminator'``
      (:class:`MolecularCrystalModel`, ``output_dim=3``).
    - ``mode == 'crystal_regression'``: ``'crystal_regressor'``
      (:class:`MolecularCrystalModel`, ``output_dim=1``).
    - ``mode == 'regression'`` or ``config.model_paths.regressor`` not None:
      ``'regressor'`` (:class:`MoleculeScalarRegressor`, target standardization
      ``[dataDims['target_mean'], dataDims['target_std']]``).
    - ``mode == 'autoencoder'`` or ``config.model_paths.autoencoder`` not None:
      ``'autoencoder'`` (:class:`Mo3ENet`).
    - ``mode == 'embedding_regression'`` or
      ``config.model_paths.embedding_regressor`` not None: ``'autoencoder'``
      (:class:`Mo3ENet` with every parameter set to ``requires_grad=False``) and
      ``'embedding_regressor'`` (:class:`EquivariantEmbeddingRegressor`).
    - ``mode == 'polymorph_classification'``: ``'polymorph_classifier'``
      (:class:`MoleculeClusterClassifier`).
    - ``mode == 'proxy_discriminator'``: ``'proxy_discriminator'``
      (:class:`InvariantEmbeddingRegressor`, ``num_targets=1``,
      ``conditions_dim=12``).

    Every name in ``model_names`` not built above is filled with
    ``nn.Linear(1, 1)``. If ``config.device.lower() == 'cuda'``, sets
    ``torch.backends.cudnn.benchmark = True``, empties the CUDA cache and moves
    every model to CUDA.

    Parameters
    ----------
    config : Namespace
        Run configuration. Reads ``mode``, ``device``, ``seeds.model``,
        ``model_paths.*`` and, for the models built, their ``<model>.model``
        sub-configs and further per-model settings (e.g.
        ``autoencoder.filter_protons``,
        ``polymorph_classifier.num_output_classes``,
        ``proxy_discriminator.embedding_type``). Mutated in place: ``embedding_regressor.model.bottleneck_dim`` is
        set to ``autoencoder.model.bottleneck_dim`` in the embedding-regression
        branch, and ``proxy_discriminator.model.bottleneck_dim`` is set to
        ``4 * autoencoder.model.bottleneck_dim`` when
        ``proxy_discriminator.embedding_type == 'autoencoder'`` and to 12 when it
        is ``'principal_axes'``, ``'principal_moments'``, ``'mol_volume'`` or
        None.
    dataDims : dict
        Dataset dimensions; the branches that use it read ``'atom_features'``,
        ``'molecule_features'``, ``'node_standardization_vector'``,
        ``'graph_standardization_vector'``, ``'target_mean'`` and
        ``'target_std'``.
    model_names : iterable of str
        Keys that must be present in the returned dictionary.
    autoencoder_type_index : torch.Tensor
        Atom-type translation index passed to :class:`Mo3ENet`; the number of
        entries not equal to -1 is passed as the number of atom types.
    sym_info
        Symmetry information passed to :class:`CrystalGenerator`.
    compile : bool, optional
        Unused.

    Returns
    -------
    dict
        Model name to ``nn.Module``.

    Raises
    ------
    AssertionError
        If a generator mode (``'gan'``, ``'search'``, ``'generator'``), the
        embedding-regression branch or ``mode == 'proxy_discriminator'`` is
        selected while ``config.model_paths.autoencoder`` is None, or if
        ``proxy_discriminator.embedding_type`` is not one of the values listed
        above.
    """
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
        models_dict['discriminator'] = MolecularCrystalModel(
            config.seeds.model,
            config.discriminator.model,
            dataDims['atom_features'],
            dataDims['molecule_features'],
            output_dim=3,
            node_standardization_tensor=dataDims['node_standardization_vector'],
            graph_standardization_tensor=dataDims['graph_standardization_vector'])
    if config.mode == 'crystal_regression':
        models_dict['crystal_regressor'] = MolecularCrystalModel(
            config.seeds.model,
            config.crystal_regressor.model,
            dataDims['atom_features'],
            dataDims['molecule_features'],
            output_dim=1,
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

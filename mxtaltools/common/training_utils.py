from argparse import Namespace

import torch
from torch import nn as nn

from mxtaltools.models.graph_models.embedding_regression_models import EmbeddingRegressor
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
            config.seeds.model
        )
    if config.mode == 'autoencoder' or config.model_paths.autoencoder is not None:
        models_dict['autoencoder'] = Mo3ENet(
            config.seeds.model,
            config.autoencoder.model,
            int(torch.sum(autoencoder_type_index != -1)),
            autoencoder_type_index,
            config.autoencoder.molecule_radius_normalization,
            infer_protons=config.autoencoder.infer_protons,
            protons_in_input=not config.autoencoder.filter_protons
        )
    if config.mode == 'embedding_regression' or config.model_paths.embedding_regressor is not None:
        models_dict['autoencoder'] = Mo3ENet(
            config.seeds.model,
            config.autoencoder.model,
            int(torch.sum(autoencoder_type_index != -1)),
            autoencoder_type_index,
            config.autoencoder.molecule_radius_normalization,
            infer_protons=config.autoencoder.infer_protons,
            protons_in_input=not config.autoencoder.filter_protons
        )
        for param in models_dict['autoencoder'].parameters():  # freeze encoder
            param.requires_grad = False
        config.embedding_regressor.model.bottleneck_dim = config.autoencoder.model.bottleneck_dim
        models_dict['embedding_regressor'] = EmbeddingRegressor(
            config.seeds.model,
            config.embedding_regressor.model,
            num_targets=config.embedding_regressor.num_targets
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
        config.proxy_discriminator.model.bottleneck_dim = config.autoencoder.model.bottleneck_dim
        models_dict['proxy_discriminator'] = EmbeddingRegressor(
            config.seeds.model,
            config.proxy_discriminator.model,
            num_targets=1,
            conditions_dim=12,
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

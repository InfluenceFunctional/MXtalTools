import gc
import multiprocessing as mp
import os
from argparse import Namespace
from pathlib import Path
from time import time
from typing import Tuple

import numpy as np
import torch
import torch.random
import wandb
from scipy.spatial.transform import Rotation as R
from torch import backends
from torch.nn import functional as F
from torch_geometric.loader.dataloader import Collater
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance, new_crystal_rdf
from mxtaltools.analysis.crystals_analysis import get_intermolecular_dists_dict
from mxtaltools.analysis.vdw_analysis import vdw_analysis, vdw_overlap
from mxtaltools.common.geometry_utils import list_molecule_principal_axes_torch
from mxtaltools.common.geometry_utils import rotvec2sph
from mxtaltools.common.training_utils import instantiate_models, init_sym_info, make_sequential_directory, \
    flatten_wandb_params, set_lr, init_optimizer, init_scheduler, reload_model, save_checkpoint, slash_batch
from mxtaltools.common.utils import sample_uniform
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.atom_properties import VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD
from mxtaltools.crystal_building.builder import CrystalBuilder
from mxtaltools.crystal_building.utils import overwrite_symmetry_info
from mxtaltools.crystal_building.utils import (set_molecule_alignment)
from mxtaltools.crystal_search.sampling import Sampler
from mxtaltools.dataset_utils.CrystalData import CrystalData
from mxtaltools.dataset_utils.dataset_manager import DataManager
from mxtaltools.dataset_utils.synthesis.otf_dataset_synthesis import otf_synthesize_molecules, otf_synthesize_crystals
from mxtaltools.dataset_utils.utils import quick_combine_dataloaders, get_dataloaders, update_dataloader_batch_size
from mxtaltools.modelling.utils import get_model_sizes, copy_source_to_workdir
from mxtaltools.models.autoencoder_utils import compute_type_evaluation_overlap, compute_coord_evaluation_overlap, \
    compute_full_evaluation_overlap, test_decoder_equivariance, test_encoder_equivariance, collate_decoded_data, \
    ae_reconstruction_loss, batch_rmsd
from mxtaltools.models.task_models.generator_models import CSDPrior
from mxtaltools.models.utils import (softmax_and_score, get_regression_loss,
                                     compute_reduced_volume_fraction,
                                     dict_of_tensors_to_cpu_numpy,
                                     clean_cell_params, denormalize_generated_cell_params, compute_prior_loss,
                                     renormalize_generated_cell_params)
from mxtaltools.reporting.ae_reporting import scaffolded_decoder_clustering
from mxtaltools.reporting.logger import Logger


#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# noinspection PyAttributeOutsideInit

class Modeller:
    """
    Class for training models
    -: data loading
    -: model initialization & training
    -: model evaluation & reporting
    """

    def __init__(self, config, sweep_config=None):
        """
        initialize config, physical constants, SGs to be generated
        load dataset and statistics
        decide what models we are training
        """
        self.config = config
        self.times = {}
        self.sweep_config = sweep_config
        self.device = self.config.device
        self.separator_string = "⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅"
        self.always_do_analysis = False
        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        self.num_naned_epochs = 0
        self.load_physical_constants()
        self.config = flatten_wandb_params(self.config)

        self.crystal_builder = CrystalBuilder(device=self.config.device,
                                              rotation_basis='spherical' if self.config.mode != 'generator' else 'cartesian')

        self.nan_lr_shrink_lambda = 0.9
        self.overall_minimum_lr = 1e-7
        self.collater = Collater(None, None)

        self.train_models_dict = {
            'discriminator': False,
            'generator': False,
            'regressor': False,
            'autoencoder': False,
            'embedding_regressor': False,
            'proxy_discriminator': False,
        }

    def load_physical_constants(self):
        """get some physical constants"""
        self.atom_weights = ATOM_WEIGHTS
        self.vdw_radii = VDW_RADII
        self.sym_info = init_sym_info()
        for key, value in ELECTRONEGATIVITY.items():
            if value is None:
                ELECTRONEGATIVITY[key] = 0
        self.electronegativity_tensor = torch.tensor(list(ELECTRONEGATIVITY.values()), dtype=torch.float32,
                                                     device=self.config.device)
        self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        self.atom_weights_tensor = torch.tensor(list(ATOM_WEIGHTS.values()), device=self.device)
        self.electronegativity_tensor = torch.tensor(list(ELECTRONEGATIVITY.values()), device=self.device)
        self.group_tensor = torch.tensor(list(GROUP.values()), device=self.device)
        self.period_tensor = torch.tensor(list(PERIOD.values()), device=self.device)

    def prep_new_working_directory(self):
        """
        make a workdir
        copy the source directory to the new working directory
        """
        self.run_identifier, self.working_directory = make_sequential_directory(self.config.paths.yaml_path,
                                                                                self.config.workdir)
        copy_source_to_workdir(
            self.working_directory,
            self.config.paths.yaml_path,
            self.config)

    def initialize_models_optimizers_schedulers(self):
        """
        Initialize models, optimizers, schedulers
        for models we will not use, just set them as nn.Linear(1,1)
        :return:
        """
        self.times['init_models_start'] = time()
        self.model_names = self.config.model_names
        self.reload_model_checkpoint_configs()
        self.models_dict = instantiate_models(self.config,
                                              self.dataDims,
                                              self.model_names,
                                              self.autoencoder_type_index,
                                              self.sym_info,
                                              compile=self.config.machine == 'cluster',
                                              )

        self.init_optimizers()
        self.reload_models()
        self.init_schedulers()
        self.num_params_dict = get_model_sizes(self.models_dict)
        self.times['init_models_end'] = time()

    def init_schedulers(self):
        self.schedulers_dict = {model_name: init_scheduler(
            self.optimizers_dict[model_name], self.config.__dict__[model_name].optimizer)
            for model_name in self.model_names}

    def init_optimizers(self):
        self.optimizers_dict = {
            model_name: init_optimizer(
                model_name, self.config.__dict__[model_name].optimizer, model
            )
            for model_name, model in self.models_dict.items()
        }
        self.hit_max_lr_dict = {model_name: False for model_name in self.model_names}

    def reload_models(self):
        for model_name, model_path in self.config.model_paths.__dict__.items():
            if model_path is not None:
                self.models_dict[model_name], self.optimizers_dict[model_name] = reload_model(
                    self.models_dict[model_name], self.device, self.optimizers_dict[model_name],
                    model_path
                )

    def load_dataset_and_dataloaders(self,
                                     override_test_fraction=None,
                                     override_shuffle=None,
                                     override_batch_size=None):
        """
        use data manager to load and filter dataset
        use dataset builder to generate crystaldata objects
        return dataloaders
        """
        self.reload_model_checkpoint_configs()
        nonzero_positional_noise = sum(list(self.config.positional_noise.__dict__.values()))
        conv_cutoff = self.set_conv_cutoffs()

        """load and filter dataset"""  # todo add assertion that if we're evaluating a dumps list, it only has one element
        data_manager = DataManager(device=self.device,
                                   datasets_path=self.config.dataset_path,
                                   config=self.config.dataset)
        data_manager.load_dataset_for_modelling(
            dataset_name=self.config.dataset_name,
            filter_conditions=self.config.dataset.filter_conditions,
            filter_polymorphs=self.config.dataset.filter_polymorphs,
            filter_duplicate_molecules=self.config.dataset.filter_duplicate_molecules,
            filter_protons=self.config.autoencoder.filter_protons if self.train_models_dict['autoencoder'] else False,
            conv_cutoff=conv_cutoff,
            do_shuffle=True,
            precompute_edges=not nonzero_positional_noise and (  # TODO have a better think about this
                    self.config.mode not in ['gan', 'discriminator', 'generator']),
            single_identifier=self.config.dataset.single_identifier,
        )
        self.dataDims = data_manager.dataDims
        self.lattice_means = torch.tensor(self.dataDims['lattice_means'], device=self.device)
        self.lattice_stds = torch.tensor(self.dataDims['lattice_stds'], device=self.device)

        if self.config.mode == 'polymorph_classification':
            self.config.polymorph_classifier.num_output_classes = 7  #self.dataDims['num_polymorphs'] + self.dataDims['num_topologies']

        self.times['dataset_loading'] = data_manager.times

        # todo change this to 'if autoencoder exists' or some proxy
        if self.train_models_dict['autoencoder'] or self.config.model_paths.autoencoder is not None:
            self.config.autoencoder_sigma = self.config.autoencoder.init_sigma
            self.config.autoencoder.molecule_radius_normalization = self.dataDims['standardization_dict']['radius'][
                'max']

            allowed_types = np.array(self.dataDims['allowed_atom_types'])
            type_translation_index = np.zeros(allowed_types.max() + 1) - 1
            for ind, atype in enumerate(allowed_types):
                type_translation_index[atype] = ind
            self.autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')
        else:
            self.autoencoder_type_index = None

        """load 'extra' dataset for evaluation"""
        if self.config.extra_test_set_name is not None:
            blind_test_conditions = [['crystal_z_prime', 'in', [1]],  # very permissive
                                     ['crystal_z_value', 'range', [1, 32]],
                                     ['atom_atomic_numbers', 'range', [1, 100]]]

            # omit blind test 5 & 6 targets
            extra_data_manager = DataManager(device=self.device,
                                             datasets_path=self.config.dataset_path,
                                             config=self.config.dataset
                                             )
            extra_data_manager.load_dataset_for_modelling(
                dataset_name=self.config.extra_test_set_name,
                override_length=int(1e7),
                filter_conditions=blind_test_conditions,  # standard filtration conditions
                filter_polymorphs=False,
                # do not filter duplicates
                filter_duplicate_molecules=False,
                filter_protons=not self.models_dict['autoencoder'].protons_in_input,
            )
            self.times['extra_dataset_loading'] = data_manager.times
        else:
            extra_data_manager = None

        """return dataloaders"""
        if override_test_fraction is not None:
            test_fraction = override_test_fraction
        else:
            test_fraction = self.config.dataset.test_fraction

        if self.config.mode == 'polymorph_classification':
            override_batch_size = 1
            print("Setting batch size to 1 for bulk classification")

        return self.prep_dataloaders(data_manager, extra_data_manager, test_fraction,
                                     override_shuffle=override_shuffle,
                                     override_batch_size=override_batch_size)

    def set_conv_cutoffs(self):
        if self.config.mode == 'polymorph_classification':
            conv_cutoff = self.config.polymorph_classifier.model.graph.cutoff
        elif self.config.mode == 'regression':
            conv_cutoff = self.config.regressor.model.graph.cutoff
        elif self.config.mode == 'autoencoder':
            conv_cutoff = self.config.autoencoder.model.encoder.graph.cutoff
        elif self.config.mode in ['gan', 'generator', 'discriminator']:
            conv_cutoff = self.config.discriminator.model.graph.cutoff
        elif self.config.mode == 'proxy_discriminator' or self.config.mode == 'embedding_regression':
            conv_cutoff = self.config.autoencoder.model.encoder.graph.cutoff
        else:
            assert False, "Missing convolutional cutoff information"
        return conv_cutoff

    def prep_dataloaders(self, dataset_builder, extra_dataset_builder=None, test_fraction=0.2,
                         override_batch_size: int = None,
                         override_shuffle=None):
        """
        get training, test, and optionally extra validation dataloaders
        """

        self.times['dataloader_start'] = time()

        if override_batch_size is None:
            loader_batch_size = self.config.min_batch_size
        else:
            loader_batch_size = override_batch_size

        if override_shuffle is not None:
            shuffle = override_shuffle
        else:
            shuffle = True

        train_loader, test_loader = get_dataloaders(dataset_builder,
                                                    machine=self.config.machine,
                                                    batch_size=loader_batch_size,
                                                    test_fraction=test_fraction,
                                                    shuffle=shuffle)
        self.config.current_batch_size = loader_batch_size
        print("Initial training batch size set to {}".format(self.config.current_batch_size))
        del dataset_builder

        # data_loader for a secondary test set - analysis is hardcoded for CSD Blind Tests 5 & 6
        if extra_dataset_builder is not None:
            _, extra_test_loader = get_dataloaders(extra_dataset_builder,
                                                   machine=self.config.machine,
                                                   batch_size=loader_batch_size,
                                                   test_fraction=1,
                                                   shuffle=shuffle)
            del extra_dataset_builder
        else:
            extra_test_loader = None
        self.times['dataloader_end'] = time()
        return train_loader, test_loader, extra_test_loader

    def ae_embedding_analysis(self):
        """prep workdir"""
        self.source_directory = os.getcwd()  # todo fix
        self.prep_new_working_directory()

        self.train_models_dict = {
            'discriminator': False,
            'generator': False,
            'regressor': False,
            'autoencoder': True,
            'embedding_regressor': False,
        }

        '''initialize datasets and useful classes'''
        _, data_loader, extra_test_loader = self.load_dataset_and_dataloaders(override_test_fraction=1,
                                                                              override_shuffle=False)
        self.initialize_models_optimizers_schedulers()

        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         # online=False,
                         settings=wandb.Settings(code_dir="."))):
            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(self.num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(self.num_params_dict.values()))),
                       "Initial Batch Size": self.config.current_batch_size})

            self.models_dict['autoencoder'].eval()
            self.epoch_type = 'test'

            with torch.no_grad():
                for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
                    self.ae_embedding_step(data)

            # post epoch processing
            self.logger.concatenate_stats_dict(self.epoch_type)

            # save results
            np.save(self.config.checkpoint_dir_path + self.config.model_paths.autoencoder.split('autoencoder')[-1],
                    {'train_stats': self.logger.train_stats, 'test_stats': self.logger.test_stats})


            ''' >>> noisy embeddings
            data = data.to(self.device)
            data0 = data.clone()
            encodings = []
            num_samples = 100
            for ind in tqdm(range(10)):
                data.pos += torch.randn_like(data.pos) * 0.05
                encodings.append(self.models_dict['autoencoder'].encode(data.clone(),
                                                                        z= torch.zeros((data.num_graphs, 3,  # uniform prior for comparison
                                                                                                   self.config.autoencoder.model.bottleneck_dim),
                                                                                                  dtype=torch.float32,
                                                                                                  device=self.config.device)
                              ).cpu().detach().numpy())
            encodings = np.stack(encodings)
            scalar_encodings = np.linalg.norm(encodings, axis=2)[:, :num_samples, :].reshape(num_samples * 10, encodings.shape[-1])
            from sklearn.manifold import TSNE
            import plotly.graph_objects as go
            
            embedding = TSNE(n_components=2, learning_rate='auto', verbose=1, n_iter=20000,
                             init='pca', perplexity=30).fit_transform(scalar_encodings)
            
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=embedding[:, 0], y=embedding[:, 1],
                                       mode='markers',
                                       marker_color=np.arange(1000) % num_samples,  # np.concatenate(stats_dict[mol_key])[:max_num_samples],
                                       opacity=.75,
                                       # marker_colorbar=dict(title=mol_key),
                                       ))
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False,
                              xaxis_title='tSNE1', yaxis_title='tSNE2', xaxis_showticklabels=False, yaxis_showticklabels=False,
                              plot_bgcolor='rgba(0,0,0,0)')
            fig.show(renderer='browser')
            '''

    def embedding_regressor_analysis(self):
        """prep workdir"""
        self.source_directory = os.getcwd()
        self.prep_new_working_directory()

        self.train_models_dict = {
            'discriminator': False,
            'generator': False,
            'regressor': False,
            'autoencoder': True,
            'embedding_regressor': True,
        }

        '''initialize datasets and useful classes'''
        _, test_loader, extra_test_loader = self.load_dataset_and_dataloaders(override_test_fraction=0.2)
        self.initialize_models_optimizers_schedulers()

        self.config.autoencoder_sigma = self.config.autoencoder.init_sigma
        self.config.autoencoder.molecule_radius_normalization = self.models_dict[
            'autoencoder'].radial_normalization  #self.dataDims['standardization_dict']['radius']['max']

        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         # online=False,
                         settings=wandb.Settings(code_dir="."))):
            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(self.num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(self.num_params_dict.values()))),
                       "Initial Batch Size": self.config.current_batch_size})

            self.models_dict['autoencoder'].eval()
            self.models_dict['embedding_regressor'].eval()
            with torch.no_grad():
                # self.epoch_type = 'train'
                #
                # for i, data in enumerate(tqdm(train_loader, miniters=int(len(train_loader) / 25))):
                #     self.embedding_regression_step(data, update_weights=False)
                #
                # # post epoch processing
                # self.logger.concatenate_stats_dict(self.epoch_type)

                self.epoch_type = 'test'
                for i, data in enumerate(tqdm(test_loader, miniters=int(len(test_loader) / 25))):
                    self.embedding_regression_step(data, update_weights=False)

                # post epoch processing
                self.logger.concatenate_stats_dict(self.epoch_type)

            # save results
            np.save(
                r'C:\Users\mikem\crystals\CSP_runs\models\ae_draft2_models_and_artifacts\embedding_regressor\results/'
                + self.config.model_paths.embedding_regressor.split("\\")[-1] + '_results.npy',
                {'train_stats': self.logger.train_stats, 'test_stats': self.logger.test_stats})

    def ae_analysis(self, save_results=True, path_prepend=str()):
        """prep workdir"""
        self.source_directory = os.getcwd()
        self.prep_new_working_directory()

        self.train_models_dict = {
            'discriminator': False,
            'generator': False,
            'regressor': False,
            'autoencoder': True,
            'embedding_regressor': False,
        }

        '''initialize datasets and useful classes'''
        train_loader, test_loader, _ = self.load_dataset_and_dataloaders()
        self.initialize_models_optimizers_schedulers()

        self.config.autoencoder_sigma = self.config.autoencoder.evaluation_sigma
        self.config.autoencoder.molecule_radius_normalization = self.models_dict['autoencoder'].radial_normalization
        #self.dataDims['standardization_dict']['radius']['max']

        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         # online=False,
                         tags=[self.config.logger.experiment_tag],
                         settings=wandb.Settings(code_dir="."))):
            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(self.num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(self.num_params_dict.values()))),
                       "Initial Batch Size": self.config.current_batch_size})

            self.config.sample_reporting_frequency = 1
            self.config.stats_reporting_frequency = 1
            self.always_do_analysis = True
            self.models_dict['autoencoder'].eval()
            update_weights = False

            print(self.separator_string)
            print("Starting Evaluation")

            with torch.no_grad():
                self.epoch_type = 'train'

                if train_loader is not None:
                    for i, data in enumerate(tqdm(train_loader, miniters=int(len(train_loader) / 25))):
                        data = data.to(self.device)
                        data.x = data.x.flatten()
                        data, input_data = self.preprocess_ae_inputs(data,
                                                                     noise=0.01,
                                                                     no_noise=False)
                        self.ae_step(input_data, data, update_weights, step=i, last_step=True)

                    # post epoch processing
                    self.logger.concatenate_stats_dict(self.epoch_type)
                else:
                    self.logger.train_stats = None

                self.epoch_type = 'test'

                for i, data in enumerate(tqdm(test_loader, miniters=int(len(test_loader) / 25))):
                    data = data.to(self.device)
                    data.x = data.x.flatten()
                    data, input_data = self.preprocess_ae_inputs(data,
                                                                 noise=0.01,
                                                                 no_noise=False)
                    self.ae_step(input_data, data, update_weights, step=i, last_step=True)

                # post epoch processing
                self.logger.concatenate_stats_dict(self.epoch_type)

            # save results
            if save_results:
                np.save(self.config.model_paths.autoencoder[:-3] + path_prepend + '_results.npy',
                        {'train_stats': self.logger.train_stats, 'test_stats': self.logger.test_stats})

    def pd_analysis(self, samples_per_molecule: int = 1):
        """prep workdir"""
        self.source_directory = os.getcwd()
        self.prep_new_working_directory()

        self.train_models_dict = {
            'discriminator': False,
            'generator': False,
            'regressor': False,
            'autoencoder': True,
            'embedding_regressor': False,
            'proxy_discriminator': True
        }

        '''initialize datasets and useful classes'''
        train_loader, test_loader, _ = self.load_dataset_and_dataloaders(override_test_fraction=0.2)
        self.initialize_models_optimizers_schedulers()

        self.config.autoencoder_sigma = self.config.autoencoder.evaluation_sigma
        self.config.autoencoder.molecule_radius_normalization = self.models_dict[
            'autoencoder'].radial_normalization  #self.dataDims['standardization_dict']['radius']['max']

        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         # online=False,
                         settings=wandb.Settings(code_dir="."))):
            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(self.num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(self.num_params_dict.values()))),
                       "Initial Batch Size": self.config.current_batch_size})

            self.always_do_analysis = True
            self.models_dict['autoencoder'].eval()
            self.models_dict['proxy_discriminator'].eval()
            update_weights = False
            with torch.no_grad():
                self.init_gan_constants()

                self.epoch_type = 'test'
                for ind in range(samples_per_molecule):
                    for i, data in enumerate(tqdm(test_loader, miniters=int(len(test_loader) / 25))):
                        data = data.to(self.config.device)
                        self.pd_step(data, i, update_weights, skip_step=False)

                # post epoch processing
                self.logger.concatenate_stats_dict(self.epoch_type)

            # save results
            np.save(
                r'C:\Users\mikem\crystals\CSP_runs\models\ae_draft2_models_and_artifacts\proxy_discriminator\results/'
                + self.config.model_paths.proxy_discriminator.split("\\")[-1] + '_results.npy',
                {'train_stats': self.logger.train_stats, 'test_stats': self.logger.test_stats})

    def ae_embedding_step(self, data):
        data = data.to(self.device)
        data, input_data = self.preprocess_ae_inputs(data, no_noise=True)
        if (not self.models_dict['autoencoder'].protons_in_input and
                not self.models_dict['autoencoder'].inferring_protons):
            data = input_data.clone()  # deprotonate the reference if we are not inferring protons

        decoding, encoding = self.models_dict['autoencoder'](input_data.clone(), return_encoding=True)
        scalar_encoding = self.models_dict['autoencoder'].scalarizer(encoding)

        self.ae_evaluation_sample_analysis(data, decoding, encoding, scalar_encoding)

    def ae_evaluation_sample_analysis(self, mol_batch, decoding, encoding, scalar_encoding):
        """

        Parameters
        ----------
        mol_batch
        decoding
        encoding
        scalar_encoding

        Returns
        -------

        """
        'standard analysis'
        autoencoder_losses, stats, decoded_mol_batch = self.compute_autoencoder_loss(decoding, mol_batch.clone())

        'extra analysis'
        mol_batch.x = self.models_dict['autoencoder'].atom_embedding_vector[mol_batch.x]
        true_nodes = F.one_hot(mol_batch.x[:, 0].long(), num_classes=self.dataDims['num_atom_types']).float()

        full_overlap, self_overlap = compute_full_evaluation_overlap(
            mol_batch,
            decoded_mol_batch,
            true_nodes,
            sigma=self.config.autoencoder.evaluation_sigma,
            distance_scaling=self.config.autoencoder.type_distance_scaling
        )
        coord_overlap, self_coord_overlap = compute_coord_evaluation_overlap(
            self.config,
            mol_batch,
            decoded_mol_batch,
            true_nodes
        )
        self_type_overlap, type_overlap = compute_type_evaluation_overlap(
            self.config,
            mol_batch,
            self.dataDims['num_atom_types'],
            decoded_mol_batch,
            true_nodes
        )
        #
        # rmsd, nodewise_dist, matched_graph, matched_node = batch_rmsd(mol_batch,
        #                                                               decoded_mol_batch)

        Ip, Ipm, I = list_molecule_principal_axes_torch(
            [mol_batch.pos[mol_batch.batch == ind] for ind in range(mol_batch.num_graphs)])

        scaffold_rmsds, scaffold_max_dists, scaffold_matched = [], [], []
        #glom_rmsds, glom_max_dists = [], []
        for ind in range(mol_batch.num_graphs):  # somewhat slow
            rmsd, max_dist, weight_mean, match_successful = scaffolded_decoder_clustering(ind, mol_batch,
                                                                                          decoded_mol_batch,
                                                                                          self.dataDims[
                                                                                              'num_atom_types'],
                                                                                          return_fig=False)
            scaffold_rmsds.append(rmsd)
            scaffold_max_dists.append(max_dist)
            scaffold_matched.append(match_successful)
            #  very slow
            # coords_true, coords_pred, points_true, points_pred, sample_weights = (
            #     extract_true_and_predicted_points(data, decoded_data, ind, self.config.autoencoder.molecule_radius_normalization, self.dataDims['num_atom_types'], to_numpy=True))
            #
            # glom_points_pred, glom_pred_weights = decoder_agglomerative_clustering(points_pred, sample_weights, 0.75)
            # _, glom_max_dist, glom_rmsd = compute_point_cloud_rmsd(points_true, glom_pred_weights, glom_points_pred, weight_threshold=0.25)
            #
            # glom_rmsds.append(glom_rmsd)
            # glom_max_dists.append(glom_max_dist)

        stats_values = [encoding.cpu().detach().numpy(),
                        mol_batch.radius.cpu().detach().numpy(),
                        scalar_encoding.cpu().detach().numpy(),
                        scatter(full_overlap / self_overlap, mol_batch.batch, reduce='mean').cpu().detach().numpy(),
                        scatter(coord_overlap / self_coord_overlap, mol_batch.batch,
                                reduce='mean').cpu().detach().numpy(),
                        scatter(self_type_overlap / type_overlap, mol_batch.batch,
                                reduce='mean').cpu().detach().numpy(),
                        Ip.cpu().detach().numpy(),
                        Ipm.cpu().detach().numpy(),
                        np.asarray(scaffold_rmsds),
                        np.asarray(scaffold_max_dists),
                        np.asarray(scaffold_matched),
                        # np.asarray(glom_rmsds),
                        # np.asarray(glom_max_dists),
                        mol_batch.smiles
                        ]
        stats_keys = ['encoding',
                      'molecule_radius',
                      'scalar_encoding',
                      'evaluation_overlap',
                      'evaluation_coord_overlap',
                      'evaluation_type_overlap',
                      'principal_inertial_axes',
                      'principal_inertial_moments',
                      'scaffold_rmsds',
                      'scaffold_max_dists',
                      'scaffold_matched',
                      # 'glom_rmsds',
                      # 'glom_max_dists',
                      'molecule_smiles'
                      ]
        assert len(stats_keys) == len(stats_values)
        self.logger.update_stats_dict(self.epoch_type,
                                      stats_keys,
                                      stats_values,
                                      mode='append')

        dict_of_tensors_to_cpu_numpy(stats)

        self.logger.update_stats_dict(self.epoch_type,
                                      stats.keys(),
                                      stats.values(),
                                      mode='append')

    def fit_models(self):
        """
        train and/or evaluate one or more models given one of our training modes
        """

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         # online=False,
                         settings=wandb.Settings(code_dir="."))):
            self.process_sweep_config()
            self.source_directory = os.getcwd()
            self.prep_new_working_directory()
            self.get_training_mode()
            train_loader, test_loader, extra_test_loader = self.load_dataset_and_dataloaders(override_shuffle=True)
            self.initialize_models_optimizers_schedulers()
            converged, epoch, prev_epoch_failed = self.init_logging()

            with torch.autograd.set_detect_anomaly(self.config.anomaly_detection,
                                                   check_nan=self.config.anomaly_detection):
                while (epoch < self.config.max_epochs) and not converged:
                    print(self.separator_string)
                    print("Starting Epoch {}".format(epoch))  # index from 0
                    self.times['full_epoch_start'] = time()
                    self.logger.reset_for_new_epoch(epoch, test_loader.batch_size)

                    if epoch < self.config.num_early_epochs:
                        steps_override = self.config.early_epochs_step_override
                    else:
                        steps_override = self.config.max_epoch_steps

                    try:  # try this batch size
                        self.train_test_validate(epoch, extra_test_loader, steps_override, test_loader, train_loader)
                        self.post_epoch_logging_analysis(train_loader, test_loader, epoch)
                        if hasattr(self, 'train_loader_to_replace'):  # dynamically update train loader
                            train_loader = self.train_loader_to_replace
                            del self.train_loader_to_replace

                        if all(list(self.logger.converged_flags.values())):  # todo confirm this works
                            print('Training has converged!')
                            break

                        if self.config.mode != 'polymorph_classification':
                            train_loader, test_loader, extra_test_loader = \
                                self.increment_batch_size(train_loader, test_loader, extra_test_loader)

                        prev_epoch_failed = False

                    except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
                        if "CUDA out of memory" in str(
                                e) or "nonzero is not supported for tensors with more than INT_MAX elements" in str(e):
                            test_loader, train_loader, prev_epoch_failed = self.handle_oom(prev_epoch_failed,
                                                                                           test_loader, train_loader)
                        elif "numerical error" in str(e).lower():
                            self.handle_nan(e, epoch)
                        else:
                            raise e  # will simply raise error if other or if training on CPU

                    self.times['full_epoch_end'] = time()
                    self.logger.log_times(self.times)
                    self.times = {}
                    epoch += 1

                self.logger.polymorph_classification_eval_analysis(test_loader, self.config.mode)

    def handle_nan(self, e, epoch):
        print(e)
        if self.num_naned_epochs > 5:
            for model_name in self.model_names:
                if self.train_models_dict[model_name]:
                    print(f"Saving {model_name} crash checkpoint")
                    save_checkpoint(epoch,
                                    self.models_dict[model_name],
                                    self.optimizers_dict[model_name],
                                    self.config.__dict__[model_name].__dict__,
                                    self.config.checkpoint_dir_path + f'best_{model_name}' + self.run_identifier + '_crashed',
                                    self.dataDims)
            raise e
        self.num_naned_epochs += 1

        print("Reloading prior best checkpoint")

        self.reload_best_test_checkpoint(epoch)

        if self.num_naned_epochs > 0:
            print("Restarting training at low LR")
            # shrink learning rate
            override_lrs = {}
            for model_name in self.model_names:
                if self.config.__dict__[model_name].optimizer is not None:
                    current_lr = self.optimizers_dict[model_name].param_groups[0]['lr']
                    if current_lr > self.overall_minimum_lr:  # absolute minimum LR we will allow
                        override_lrs[model_name] = self.nan_lr_shrink_lambda * current_lr

            # resetting optimizer state
            self.optimizers_dict = {
                model_name: init_optimizer(
                    model_name, self.config.__dict__[model_name].optimizer, model
                )
                for model_name, model in self.models_dict.items()
            }
        else:
            override_lrs = None

        self.update_lr(override_lr=override_lrs)  # reduce LR and try again
        self.hit_max_lr_dict = {key: True for key in self.hit_max_lr_dict.keys()}

    def handle_oom(self, prev_epoch_failed, test_loader, train_loader):
        if prev_epoch_failed:
            gc.collect()  # TODO not clear to me that these are effective
            torch.cuda.empty_cache()
        train_loader, test_loader = slash_batch(train_loader, test_loader, slash_fraction=0.1)
        self.config.grow_batch_size = False  # stop growing the batch for the rest of the run
        prev_epoch_failed = True
        return test_loader, train_loader, prev_epoch_failed

    def evaluate_model(self):
        self.config.sample_reporting_frequency = 1
        self.config.stats_reporting_frequency = 1
        self.always_do_analysis = True

        if self.config.mode == 'generator':
            self.crystal_structure_prediction()
        else:
            with (wandb.init(config=self.config,
                             project=self.config.wandb.project_name,
                             entity=self.config.wandb.username,
                             tags=[self.config.logger.experiment_tag],
                             # online=False,
                             settings=wandb.Settings(code_dir="."))):
                self.source_directory = os.getcwd()
                self.prep_new_working_directory()
                self.get_training_mode()
                _, data_loader, _ = self.load_dataset_and_dataloaders(
                    override_test_fraction=1,
                    override_shuffle=False,
                )
                self.initialize_models_optimizers_schedulers()

                print(self.separator_string)
                print("Starting Evaluation")
                self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)
                self.times = {}  # reset for iterative looping
                self.logger.reset_for_new_epoch(0, data_loader.batch_size)

                with torch.no_grad():
                    self.run_epoch(epoch_type='test',
                                   data_loader=data_loader,
                                   update_weights=False,
                                   )

                self.post_epoch_logging_analysis(None, data_loader, 0)
                self.logger.polymorph_classification_eval_analysis(data_loader, self.config.mode)
                for model_name in self.model_names:
                    if self.train_models_dict[model_name]:
                        self.logger.save_stats_dict(prefix=f'best_{model_name}_')

                #self.logger.save_stats_dict(prefix=f'best_{model_name}_')

    def train_test_validate(self, epoch, extra_test_loader, steps_override, test_loader, train_loader):
        self.run_epoch(epoch_type='train',
                       data_loader=train_loader,
                       update_weights=True,
                       iteration_override=steps_override)

        for model in self.models_dict.values():
            if not torch.stack([torch.isfinite(p).any() for p in model.parameters()]).all():
                raise ValueError('Numerical Error: Model weights not all finite')

        with torch.no_grad():
            self.run_epoch(epoch_type='test',
                           data_loader=test_loader,
                           update_weights=False,
                           iteration_override=int(steps_override * self.config.dataset.test_fraction))

            if (extra_test_loader is not None) and \
                    (epoch % self.config.extra_test_period == 0) and \
                    (epoch > 0):
                self.run_epoch(epoch_type='extra',
                               data_loader=extra_test_loader,
                               update_weights=False,
                               iteration_override=None)

        self.num_naned_epochs = 0

    def post_epoch_logging_analysis(self, train_loader, test_loader, epoch):
        """check convergence status and record metrics & analysis"""
        self.times['reporting_start'] = time()
        self.logger.numpyize_current_losses()
        self.logger.update_loss_record()
        self.logger.log_training_metrics()
        if (self.logger.epoch % self.logger.sample_reporting_frequency) == 0:
            self.logger.log_detailed_analysis()
        if (time() - self.logger.last_logged_dataset_stats) > self.config.logger.dataset_reporting_time:
            if self.config.mode == 'autoencoder' or self.config.mode == 'proxy_discriminator':
                self.logger.log_dataset_analysis(train_loader, test_loader)

        self.logger.check_model_convergence()
        self.times['reporting_end'] = time()

        if self.config.save_checkpoints and epoch > 0:
            self.model_checkpointing(epoch)

        self.update_lr()

    def init_logging(self):
        """initialize some training metrics"""
        converged, epoch, prev_epoch_failed = self.config.max_epochs == 0, 0, False
        wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
        wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
        wandb.log(data=self.num_params_dict, commit=False)
        wandb.log(data={"All Models Parameters": np.sum(np.asarray(list(self.num_params_dict.values()))),
                        "Initial Batch Size": self.config.current_batch_size},
                  commit=False)
        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)
        self.logger.log_times(self.times)  # log initialization times
        self.times = {}  # reset for iterative looping
        return converged, epoch, prev_epoch_failed

    def get_training_mode(self):
        self.train_models_dict = {
            'discriminator': ((self.config.mode in ['gan', 'generator', 'discriminator']) and any(
                (self.config.discriminator.train_adversarially, self.config.discriminator.train_on_distorted,
                 self.config.discriminator.train_on_randn)))
                             or (self.config.model_paths.discriminator is not None),
            'generator': (self.config.mode in ['gan', 'generator']),
            'regressor': self.config.mode == 'regression',
            'autoencoder': self.config.mode == 'autoencoder',
            'embedding_regressor': self.config.mode == 'embedding_regression',
            'polymorph_classifier': self.config.mode == 'polymorph_classification',
            'proxy_discriminator': self.config.mode == 'proxy_discriminator',
        }

    def process_sweep_config(self):
        if self.sweep_config is not None:  # write sweep config features to working config # todo make more universal - I hate wandb configs
            def write_dict_to_namespace(d1, d2):
                """
                d1 is the first level dict of a namespace
                """
                for key in d1.__dict__.keys():
                    print(key)
                    if key in d2.keys():
                        if not isinstance(d1.__dict__[key], Namespace):
                            d1.__dict__[key] = d2[key]
                        elif key in ['autoencoder', 'model', 'optimizer']:
                            d1.__dict__[key], d2[key] = write_dict_to_namespace(d1.__dict__[key], d2[key])

                return d1, d2

            self.config, wandb.config = write_dict_to_namespace(self.config, wandb.config)

    def run_epoch(self,
                  epoch_type: str,
                  data_loader: CrystalData = None,
                  update_weights: bool = True,
                  iteration_override: int = None):
        self.epoch_type = epoch_type
        self.times[epoch_type + "_epoch_start"] = time()

        if self.config.mode in ['gan', 'generator']:
            if self.config.model_paths.regressor is not None:
                self.models_dict['regressor'].eval()  # using this to suggest densities to the generator

        if self.train_models_dict['discriminator']:
            self.discriminator_epoch(data_loader, update_weights, iteration_override)

        if self.train_models_dict['generator']:
            self.generator_epoch(data_loader, update_weights, iteration_override)

        elif self.config.mode == 'regression':
            self.regression_epoch(data_loader, update_weights, iteration_override)

        elif self.config.mode == 'autoencoder':
            self.ae_epoch(data_loader, update_weights, iteration_override)

        elif self.config.mode == 'embedding_regression':
            self.embedding_regression_epoch(data_loader, update_weights, iteration_override)

        elif self.config.mode == 'polymorph_classification':
            self.polymorph_classification_epoch(data_loader, update_weights, iteration_override)

        elif self.config.mode == 'proxy_discriminator':
            self.pd_epoch(data_loader, update_weights, iteration_override)

        self.times[epoch_type + "_epoch_end"] = time()

    def embedding_regression_epoch(self, data_loader, update_weights, iteration_override):
        if update_weights:
            self.models_dict['embedding_regressor'].train(True)
        else:
            self.models_dict['embedding_regressor'].eval()

        self.models_dict['autoencoder'].eval()

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            self.embedding_regression_step(data, update_weights)

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early

        # post epoch processing
        self.logger.concatenate_stats_dict(self.epoch_type)

    def embedding_regression_step(self, data, update_weights):
        data = data.to(self.device)
        _, data = self.preprocess_ae_inputs(data, no_noise=True, orientation_override=None)
        v_embedding = self.models_dict['autoencoder'].encode(data)
        s_embedding = self.models_dict['autoencoder'].scalarizer(v_embedding)

        s_predictions, v_predictions = self.models_dict['embedding_regressor'](s_embedding, v_embedding)
        if self.models_dict['embedding_regressor'].prediction_type == 'scalar':
            losses = F.smooth_l1_loss(s_predictions[..., 0], data.y, reduction='none')
            predictions = s_predictions

        elif self.models_dict['embedding_regressor'].prediction_type == 'vector':
            losses = F.smooth_l1_loss(v_predictions[..., 0], data.y, reduction='none')
            predictions = v_predictions

        elif self.models_dict['embedding_regressor'].prediction_type == '2-tensor':
            # generate a batch of rank 2 even (2e) symmetric tensors
            # do a weighted sum with learned weights
            a, b = v_predictions.split(v_predictions.shape[-1] // 2, dim=2)

            t1 = torch.einsum('nik,njk->nijk', a, b)
            t2 = torch.einsum('nik,njk->nijk', b, a)

            # linearly combine them, weighted by the scalar outputs
            weights = F.softmax(s_predictions[:, a.shape[-1]:], dim=1)
            isotropic_part = (s_predictions[:, :a.shape[-1], None, None].sum(1) *
                              torch.eye(3, device=self.device, dtype=torch.float32
                                        ).repeat(data.num_graphs, 1, 1))

            symmetric_tensor = ((t1 + t2) / 2)
            t_predictions = torch.sum(weights[:, None, None, :] * symmetric_tensor, dim=-1) + isotropic_part
            losses = F.smooth_l1_loss(t_predictions, data.y, reduction='none')
            predictions = t_predictions

        elif self.models_dict['embedding_regressor'].prediction_type == '3-tensor':
            # create a basis of rand 3 odd (3o) symmetric tensors
            # do a weighted sum with learned weights

            # construct symmetric 3-tensor
            a, b, c, d = v_predictions.split([v_predictions.shape[-1] // 4 for _ in range(4)], dim=2)

            t12 = torch.einsum('nik,njk->nijk', a, b)
            t21 = torch.einsum('nik,njk->nijk', b, a)
            t23 = torch.einsum('nik,njk->nijk', b, c)
            t32 = torch.einsum('nik,njk->nijk', c, b)
            t13 = torch.einsum('nik,njk->nijk', a, c)
            t31 = torch.einsum('nik,njk->nijk', c, a)

            t123 = torch.einsum('nijk,nlk->nijlk', t12, c)
            t213 = torch.einsum('nijk,nlk->nijlk', t21, c)
            t231 = torch.einsum('nijk,nlk->nijlk', t23, a)
            t321 = torch.einsum('nijk,nlk->nijlk', t32, a)
            t132 = torch.einsum('nijk,nlk->nijlk', t13, b)
            t312 = torch.einsum('nijk,nlk->nijlk', t31, b)

            symmetric_tensor = (1 / 6) * (t123 + t213 + t231 + t321 + t132 + t312)

            vec_to_3_tensor = torch.tensor([[0.7746, 0.0000, 0.0000, 0.0000, 0.2582, 0.0000, 0.0000, 0.0000, 0.2582,
                                             0.0000, 0.2582, 0.0000, 0.2582, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                             0.0000, 0.0000, 0.2582, 0.0000, 0.0000, 0.0000, 0.2582, 0.0000, 0.0000],
                                            [0.0000, 0.2582, 0.0000, 0.2582, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                             0.2582, 0.0000, 0.0000, 0.0000, 0.7746, 0.0000, 0.0000, 0.0000, 0.2582,
                                             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2582, 0.0000, 0.2582, 0.0000],
                                            [0.0000, 0.0000, 0.2582, 0.0000, 0.0000, 0.0000, 0.2582, 0.0000, 0.0000,
                                             0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2582, 0.0000, 0.2582, 0.0000,
                                             0.2582, 0.0000, 0.0000, 0.0000, 0.2582, 0.0000, 0.0000, 0.0000, 0.7746]],
                                           dtype=torch.float32,
                                           device=self.device)

            vec_embedding = torch.einsum('nik, ij -> njk', d, vec_to_3_tensor).reshape(data.num_graphs, 3, 3, 3,
                                                                                       d.shape[-1])

            # linearly combine them, weighted by the scalar outputs
            t_weights = F.softmax(s_predictions[:, :a.shape[-1]], dim=1)
            v_weights = F.softmax(s_predictions[:, -a.shape[-1]:], dim=1)

            t_predictions = (torch.sum(t_weights[:, None, None, None, :] * symmetric_tensor, dim=-1)
                             + torch.sum(v_weights[:, None, None, None, :] * vec_embedding, dim=-1)
                             )
            losses = F.smooth_l1_loss(t_predictions, data.y, reduction='none')
            predictions = t_predictions

        else:
            assert False, "Embedding regressor must be in scalar or vector mode"

        regression_loss = losses.mean()
        if update_weights:
            self.optimizers_dict['embedding_regressor'].zero_grad(set_to_none=True)
            regression_loss.backward()  # back-propagation
            self.optimizers_dict['embedding_regressor'].step()  # update parameters
        '''log losses and other tracking values'''
        self.logger.update_current_losses('embedding_regressor', self.epoch_type,
                                          regression_loss.cpu().detach().numpy(),
                                          losses.cpu().detach().numpy())
        stats_values = [
            (predictions.cpu().detach() * self.dataDims['target_std'] + self.dataDims['target_mean'])[
                None, ...].numpy(),
            (data.y.cpu().detach() * self.dataDims['target_std'] + self.dataDims['target_mean'])[None, ...].numpy()]

        self.logger.update_stats_dict(self.epoch_type,
                                      ['regressor_prediction', 'regressor_target'],
                                      stats_values,
                                      mode='extend')

    def ae_step(self, input_data, mol_batch, update_weights, step, last_step=False):
        if (not self.models_dict['autoencoder'].protons_in_input and
                not self.models_dict['autoencoder'].inferring_protons):
            mol_batch = input_data.detach().clone()

        if step % self.config.logger.stats_reporting_frequency == 0:
            skip_stats = False
        elif last_step:
            skip_stats = False
        else:
            skip_stats = True

        decoding = self.models_dict['autoencoder'](input_data.clone(), return_latent=False)
        if torch.sum(torch.isnan(decoding)) > 0:
            raise ValueError("Numerical Error: decoder output is not finite")

        losses, stats, decoded_mol_batch = self.compute_autoencoder_loss(decoding, mol_batch.clone(),
                                                                         skip_stats=skip_stats)

        mean_loss = losses.mean()
        if torch.sum(torch.logical_not(torch.isfinite(mean_loss))) > 0:
            print('loss is not finite')
            raise ValueError("Numerical Error: autoencoder loss is not finite")

        if update_weights:
            self.optimizers_dict['autoencoder'].zero_grad(set_to_none=True)
            mean_loss.backward()  # back-propagation
            if not torch.stack([torch.isfinite(p.grad).any() if p.grad is not None else torch.isfinite(torch.ones_like(p)).any() for p in self.models_dict['autoencoder'].parameters()]).all():
                raise ValueError("Numerical Error: model has NaN gradients!")

                '''
                pp = {name:bool(torch.isfinite(p.grad).any()) if p.grad is not None else None for name, p in self.models_dict['autoencoder'].named_parameters()}
                for key in pp.keys():
                    print(key, pp[key])
                '''
            torch.nn.utils.clip_grad_norm_(self.models_dict['autoencoder'].parameters(),
                                           self.config.gradient_norm_clip)  # gradient clipping by norm
            self.optimizers_dict['autoencoder'].step()  # update parameters
            if not torch.stack([torch.isfinite(p).any() for p in self.models_dict['autoencoder'].parameters()]).all():
                raise ValueError("Numerical Error: model has NaN weights!")

        if not skip_stats:
            if self.always_do_analysis:
                with torch.no_grad():
                    _, vector_embedding = self.models_dict['autoencoder'](input_data.clone(), return_latent=True)
                    scalar_embedding = self.models_dict['autoencoder'].scalarizer(vector_embedding)
                stats['scalar_embedding'] = scalar_embedding.detach()
                stats['vector_embedding'] = vector_embedding.detach()
                stats['molecule_smiles'] = input_data.smiles

            self.ae_stats_and_reporting(mol_batch,
                                        decoded_mol_batch,
                                        last_step,
                                        stats,
                                        step,
                                        override_do_analysis=self.always_do_analysis)

    def fix_autoencoder_protonation(self, data, override_deprotonate=False):
        if (not self.models_dict['autoencoder'].inferring_protons and
                self.models_dict['autoencoder'].protons_in_input and
                not override_deprotonate):
            input_cloud = data.detach().clone()
        else:
            heavy_atom_inds = torch.argwhere(data.x != 1).flatten()  # protons are atom type 1
            input_cloud = data.detach().clone()
            input_cloud.x = input_cloud.x[heavy_atom_inds]
            input_cloud.pos = input_cloud.pos[heavy_atom_inds]
            input_cloud.batch = input_cloud.batch[heavy_atom_inds]
            a, b = torch.unique(input_cloud.batch, return_counts=True)
            input_cloud.ptr = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(b, dim=0)]).long()
            input_cloud.num_atoms = torch.diff(input_cloud.ptr).long()

        return input_cloud

    def ae_stats_and_reporting(self,
                               data: CrystalData,
                               decoded_data: CrystalData,
                               last_step: bool,
                               stats: dict,
                               step: int,
                               override_do_analysis: bool = False):

        if any([step == 0, last_step, override_do_analysis]):
            self.detailed_autoencoder_step_analysis(data, decoded_data, stats)

        dict_of_tensors_to_cpu_numpy(stats)

        self.logger.update_stats_dict(self.epoch_type,
                                      stats.keys(),
                                      stats.values(),
                                      mode='append')

    def detailed_autoencoder_step_analysis(self,
                                           mol_batch,
                                           decoded_mol_batch,
                                           stats):
        # equivariance checks
        encoder_equivariance_loss, decoder_equivariance_loss = self.ae_equivariance_loss(mol_batch.clone())
        stats['encoder_equivariance_loss'] = encoder_equivariance_loss.mean().detach()
        stats['decoder_equivariance_loss'] = decoder_equivariance_loss.mean().detach()

        # do evaluation on current sample and save this as our loss for tracking purposes
        nodewise_weights_tensor = decoded_mol_batch.aux_ind
        true_nodes = F.one_hot(self.models_dict['autoencoder'].atom_embedding_vector[mol_batch.x.long()],
                               num_classes=self.dataDims['num_atom_types']).float()
        full_overlap, self_overlap = compute_full_evaluation_overlap(
            mol_batch, decoded_mol_batch,
            true_nodes,
            sigma=self.config.autoencoder.evaluation_sigma,
            distance_scaling=self.config.autoencoder.type_distance_scaling
        )

        '''log losses and other tracking values'''
        # for the purpose of convergence, we track the evaluation overlap rather than the loss, which is sigma-dependent
        # it's also expensive to compute so do it rarely
        overlap = (full_overlap / self_overlap).detach()
        tracking_loss = torch.abs(1 - overlap)
        stats['evaluation_overlap'] = scatter(overlap, mol_batch.batch, reduce='mean').detach()
        rmsd, nodewise_dists, matched_graphs, matched_nodes, _, pred_particle_weights = batch_rmsd(
            mol_batch,
            decoded_mol_batch,
            true_nodes,
        )
        stats['RMSD'] = rmsd[matched_graphs].mean().detach()
        stats['matching_graph_fraction'] = (torch.sum(matched_graphs) / len(matched_graphs)).detach()
        stats['matching_node_fraction'] = (torch.sum(matched_nodes) / len(matched_nodes)).detach()
        stats['nodewise_dists'] = nodewise_dists[matched_nodes].mean().detach()
        stats['nodewise_dists_dist'] = nodewise_dists.detach()
        stats['RMSD_dist'] = rmsd.detach()
        stats['matched_nodes'] = matched_nodes.detach()
        stats['matched_graphs'] = matched_graphs.detach()

        self.logger.update_current_losses('autoencoder', self.epoch_type,
                                          tracking_loss.mean().cpu().detach().numpy(),
                                          tracking_loss.cpu().detach().numpy())

        mol_samples, decoded_mol_samples = [], []
        for ind in range(mol_batch.num_graphs):
            b1 = mol_batch.batch == ind
            mol_samples.append(
                CrystalData(  # CRYTODO
                    x=mol_batch.x[b1],
                    pos=mol_batch.pos[b1],
                    radius=mol_batch.radius[ind],
                    smiles=mol_batch.smiles[ind],
                    identifier=mol_batch.identifier[ind],
                    mol_volume=mol_batch.mol_volume[ind],
                ).cpu().detach()
            )

            b2 = decoded_mol_batch.batch == ind
            decoded_mol_samples.append(
                CrystalData(  # CRYTODO
                    x=decoded_mol_batch.x[b2],
                    pos=decoded_mol_batch.pos[b2],
                    radius=decoded_mol_batch.radius[ind],
                    smiles=decoded_mol_batch.smiles[ind],
                    identifier=decoded_mol_batch.identifier[ind],
                    mol_volume=decoded_mol_batch.mol_volume[ind],
                    aux_ind=decoded_mol_batch.aux_ind[b2]
                ).cpu().detach()
            )

        self.logger.update_stats_dict(self.epoch_type,
                                      ['sample', 'decoded_sample'],
                                      [mol_samples, decoded_mol_samples],
                                      mode='extend')

    def ae_equivariance_loss(self, data: CrystalData) -> Tuple[torch.Tensor, torch.Tensor]:
        rotations = torch.tensor(
            R.random(data.num_graphs).as_matrix() *
            np.random.choice((-1, 1), replace=True, size=data.num_graphs)[:, None, None],
            dtype=torch.float,
            device=data.x.device)

        encoder_equivariance_loss, encoding, rotated_encoding = test_encoder_equivariance(data,
                                                                                          rotations,
                                                                                          self.models_dict[
                                                                                              'autoencoder'])

        decoder_equivariance_loss = test_decoder_equivariance(data,
                                                              encoding,
                                                              rotated_encoding,
                                                              rotations,
                                                              self.models_dict['autoencoder'],
                                                              self.config.device)

        # if self.epoch_type == 'test':
        #     if encoder_equivariance_loss.mean() > 0.01:
        #         aa = 1

        return encoder_equivariance_loss, decoder_equivariance_loss

    def ae_epoch(self,
                 data_loader,
                 update_weights: bool,
                 iteration_override: bool = None):

        if self.config.dataset.otf_build_size > 0 and self.epoch_type == 'train' and os.cpu_count() > 1:
            self.train_loader_to_replace = self.otf_molecule_dataset_generation(data_loader)
            data_loader = self.train_loader_to_replace

        if update_weights:
            self.models_dict['autoencoder'].train(True)
        else:
            self.models_dict['autoencoder'].eval()

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            data = data.to(self.device)
            data.x = data.x.flatten()
            data, input_data = self.preprocess_ae_inputs(data,
                                                         no_noise=False,
                                                         noise=self.config.positional_noise.autoencoder if self.epoch_type == 'train' else 0.01,
                                                         affine_scale=self.config.autoencoder.affine_scale_factor if self.epoch_type == 'train' else None
                                                         )
            self.ae_step(input_data, data, update_weights, step=i,
                         last_step=(i == len(data_loader) - 1) or (i == iteration_override))

            if iteration_override is not None:
                if i >= iteration_override:
                    break

        self.logger.concatenate_stats_dict(self.epoch_type)
        if self.epoch_type == 'train':
            self.ae_annealing()

    def otf_molecule_dataset_generation(self, data_loader):
        self.times['otf_refresh_start'] = time()
        temp_dataset_path = Path(self.working_directory).joinpath('otf_dataset.pt')
        # if previous batch is finished, or we are in first epoch,
        # initiate parallel otf conformer generation
        chunks_path = Path(self.working_directory).joinpath('chunks')
        if not os.path.exists(chunks_path):
            os.mkdir(chunks_path)

        if self.logger.epoch == 0:  # refresh
            [os.remove(chunks_path.joinpath(elem)) for elem in os.listdir(chunks_path)]
            self.integrated_dataset = False

        num_processes = self.config.dataset.num_processes
        if len(os.listdir(chunks_path)) == 0:  # only make a new batch if the previous batch has been integrated
            if self.logger.epoch == 0 or self.integrated_dataset == True:
                self.mp_pool = mp.Pool(num_processes)
                self.mp_pool = otf_synthesize_molecules(
                    self.config.dataset.otf_build_size,
                    self.config.dataset.smiles_source,
                    workdir=chunks_path,
                    allowed_atom_types=[1, 6, 7, 8, 9],
                    #list(self.dataDims['allowed_atom_types'].cpu().detach().numpy()),
                    num_processes=num_processes,
                    mp_pool=self.mp_pool, max_num_atoms=30,
                    max_num_heavy_atoms=9, pare_to_size=9,
                    max_radius=15, synchronize=False)
                self.integrated_dataset = False

        # if a batch is finished, merge it with our existing dataset
        if len(os.listdir(chunks_path)) == num_processes:  # only integrate when the batch is exactly complete
            self.times['otf_dataset_join_start'] = time()
            self.mp_pool.join()  # join only when the batch is already finished
            self.times['otf_dataset_join_end'] = time()
            # generate temporary training dataset
            miner = self.process_otf_molecules_dataset(chunks_path)
            # print("integrating otf dataset into dataloader")
            data_loader = quick_combine_dataloaders(miner.dataset,
                                                    data_loader,
                                                    data_loader.batch_size,
                                                    self.config.dataset.max_dataset_length)
            os.remove(temp_dataset_path)  # delete loaded dataset
            self.integrated_dataset = True
            num_atoms = np.sum([data.num_atoms for data in data_loader.dataset])
            stats = {
                'dataset_length': len(data_loader.dataset),
                'mean_molecule_size': num_atoms / len(data_loader.dataset),
                'mean_hydrogen_fraction': np.sum(
                    np.concatenate([data.x == 1 for data in data_loader.dataset])) / num_atoms,
                'mean_carbon_fraction': np.sum(
                    np.concatenate([data.x == 6 for data in data_loader.dataset])) / num_atoms,
                'mean_nitrogen_fraction': np.sum(
                    np.concatenate([data.x == 7 for data in data_loader.dataset])) / num_atoms,
                'mean_oxygen_fraction': np.sum(
                    np.concatenate([data.x == 8 for data in data_loader.dataset])) / num_atoms,
            }

            self.logger.update_stats_dict(self.epoch_type,
                                          stats.keys(),
                                          stats.values(),
                                          mode='append')

        os.chdir(self.working_directory)
        self.times['otf_refresh_end'] = time()
        return data_loader

    def otf_crystal_dataset_generation(self, data_loader):
        self.times['otf_refresh_start'] = time()
        temp_dataset_path = Path(self.working_directory).joinpath('otf_dataset.pt')
        # if previous batch is finished, or we are in first epoch,
        # initiate parallel otf conformer generation
        chunks_path = Path(self.working_directory).joinpath('chunks')
        if not os.path.exists(chunks_path):
            os.mkdir(chunks_path)

        if self.logger.epoch == 0:  # refresh
            [os.remove(chunks_path.joinpath(elem)) for elem in os.listdir(chunks_path)]
            self.integrated_dataset = False

        num_processes = self.config.dataset.num_processes
        if len(os.listdir(chunks_path)) == 0:  # only make a new batch if the previous batch has been integrated
            if self.logger.epoch == 0 or self.integrated_dataset == True:
                print('sending crystal opt jobs to mp pool')
                mp.set_start_method('spawn', force=True)
                self.mp_pool = mp.Pool(num_processes)
                self.mp_pool = otf_synthesize_crystals(
                    self.config.dataset.otf_build_size,
                    self.config.dataset.smiles_source,
                    workdir=chunks_path,
                    allowed_atom_types=[1, 6, 7, 8, 9],
                    #list(self.dataDims['allowed_atom_types'].cpu().detach().numpy()),
                    num_processes=num_processes,
                    mp_pool=self.mp_pool, max_num_atoms=30,
                    max_num_heavy_atoms=9, pare_to_size=9,
                    max_radius=15, synchronize=False)
                self.integrated_dataset = False

        #assert False, "stop for debugging"

        # if a batch is finished, merge it with our existing dataset
        if len(os.listdir(chunks_path)) >= num_processes:  # only integrate when the batch is exactly complete
            self.times['otf_dataset_join_start'] = time()
            self.mp_pool.join()  # join only when the batch is already finished
            self.times['otf_dataset_join_end'] = time()
            # generate temporary training dataset
            miner = self.process_otf_crystals_dataset(chunks_path)
            embeddings = []
            batch_size = data_loader.batch_size
            num_batches = len(miner.dataset) // batch_size
            if len(miner.dataset) % batch_size != 0:
                num_batches += 1
            with torch.no_grad():
                for ind in range(num_batches):
                    mols_to_embed = self.collater(miner.dataset[ind * batch_size:(ind + 1) * batch_size]).to(
                        self.device)
                    cell_params = torch.cat([
                        mols_to_embed.cell_lengths, mols_to_embed.cell_angles, mols_to_embed.pose_params0
                    ], dim=1)
                    embeddings.append(self._embed_mol(mols_to_embed, mols_to_embed.pos, cell_params).cpu().detach())

            embeddings = torch.cat(embeddings, dim=0)
            for ind in range(len(miner.dataset)):
                miner.dataset[ind].y = embeddings[None, ind]
            # print("integrating otf dataset into dataloader")
            data_loader = quick_combine_dataloaders(miner.dataset,
                                                    data_loader,
                                                    data_loader.batch_size,
                                                    self.config.dataset.max_dataset_length)
            os.remove(temp_dataset_path)  # delete loaded dataset
            self.integrated_dataset = True
            num_atoms = np.sum([data.num_atoms for data in data_loader.dataset])
            stats = {
                'dataset_length': len(data_loader.dataset),
                'mean_molecule_size': num_atoms / len(data_loader.dataset),
                'mean_hydrogen_fraction': np.sum(
                    np.concatenate([data.x == 1 for data in data_loader.dataset])) / num_atoms,
                'mean_carbon_fraction': np.sum(
                    np.concatenate([data.x == 6 for data in data_loader.dataset])) / num_atoms,
                'mean_nitrogen_fraction': np.sum(
                    np.concatenate([data.x == 7 for data in data_loader.dataset])) / num_atoms,
                'mean_oxygen_fraction': np.sum(
                    np.concatenate([data.x == 8 for data in data_loader.dataset])) / num_atoms,
            }

            self.logger.update_stats_dict(self.epoch_type,
                                          stats.keys(),
                                          stats.values(),
                                          mode='append')

        os.chdir(self.working_directory)
        self.times['otf_refresh_end'] = time()
        return data_loader

    def process_otf_molecules_dataset(self, chunks_path):
        miner = DataManager(device='cpu',
                            config=self.config.dataset,
                            datasets_path=self.working_directory,
                            chunks_path=chunks_path,
                            dataset_type='molecule', )
        miner.process_new_dataset(new_dataset_name='otf_dataset',
                                  chunks_patterns=['chunk'])
        del miner.dataset
        # kill old chunks so we don't re-use
        [os.remove(elem) for elem in os.listdir(chunks_path)]
        conv_cutoff = self.config.autoencoder.model.encoder.graph.cutoff
        nonzero_positional_noise = sum(list(self.config.positional_noise.__dict__.values()))
        miner.load_dataset_for_modelling(
            'otf_dataset.pt',
            filter_conditions=self.config.dataset.filter_conditions,
            filter_polymorphs=self.config.dataset.filter_polymorphs,
            filter_duplicate_molecules=self.config.dataset.filter_duplicate_molecules,
            filter_protons=self.config.autoencoder.filter_protons if self.train_models_dict['autoencoder'] else False,
            conv_cutoff=conv_cutoff,
            do_shuffle=True,
            precompute_edges=not nonzero_positional_noise and (
                    self.config.mode not in ['gan', 'discriminator', 'generator']),
            single_identifier=self.config.dataset.single_identifier,
        )
        return miner

    def process_otf_crystals_dataset(self, chunks_path):
        miner = DataManager(device='cpu',
                            config=self.config.dataset,
                            datasets_path=self.working_directory,
                            chunks_path=chunks_path,
                            dataset_type='crystal',
                            do_crystal_indexing=False)
        miner.process_new_dataset(new_dataset_name='otf_dataset',
                                  chunks_patterns=['chunk'])
        del miner.dataset
        # kill old chunks so we don't re-use
        [os.remove(elem) for elem in os.listdir(chunks_path)]
        conv_cutoff = self.config.autoencoder.model.encoder.graph.cutoff
        #nonzero_positional_noise = sum(list(self.config.positional_noise.__dict__.values()))
        miner.regression_target = None
        miner.load_dataset_for_modelling(
            'otf_dataset.pt',
            filter_conditions=None,
            filter_polymorphs=False,
            filter_duplicate_molecules=False,
            filter_protons=self.config.autoencoder.filter_protons if self.train_models_dict['autoencoder'] else False,
            conv_cutoff=conv_cutoff,
            do_shuffle=True,
            precompute_edges=False,
            single_identifier=None,

        )
        return miner

    def crystal_structure_prediction(self):
        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         # online=False,
                         settings=wandb.Settings(code_dir="."))):
            self.source_directory = os.getcwd()
            self.prep_new_working_directory()
            self.get_training_mode()
            _, data_loader, _ = self.load_dataset_and_dataloaders(
                override_test_fraction=1,
                override_shuffle=False,
            )
            self.initialize_models_optimizers_schedulers()

            print(self.separator_string)
            print("Starting Evaluation")
            self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)
            self.times = {}  # reset for iterative looping
            self.logger.reset_for_new_epoch(0, data_loader.batch_size)

            if not hasattr(self, 'packing_loss_coefficient'):  # first GAN epoch
                self.init_gan_constants()

            self.models_dict['generator'].eval()
            self.models_dict['autoencoder'].eval()
            self.models_dict['discriminator'].eval()

            # test autoencoder performance on this sample
            sample = self.collater(data_loader.dataset[0:200])
            data, input_data = self.preprocess_ae_inputs(sample.to(self.device), no_noise=True,
                                                         orientation_override=None)
            loss, rmsd, max_dist, tot_overlap, matched = self.models_dict['autoencoder'].check_embedding_quality(
                input_data.clone())
            # topk_samples = self.collater([data_loader.dataset[ind] for ind in torch.argsort(loss)])
            # sample = data_loader.dataset[torch.argmin(loss)]
            # 131369 is a good molecule, good reconstruction loss and flat-ish double ring
            ids = [data_loader.dataset[ind].identifier for ind in range(len(data_loader.dataset))]
            sample = data_loader.dataset[ids.index(131369)]

            # for use with QM9/CSD dataset
            # ids = [data_loader.dataset[ind].identifier for ind in range(len(data_loader.dataset))]
            #
            # sample = data_loader.dataset[ids.index('DUNVEN01')]

            # fairly rigid molecules: ampyrd, axosow, benzac, bepnit, bertoh, bzamid,
            # nicoam, DUNVEN, dovcat, NECMUD,

            sampler = Sampler(0,
                              self.device,
                              self.config.machine,
                              self.generator_prior,
                              self.models_dict['generator'],
                              self.models_dict['autoencoder'],
                              self.models_dict['discriminator'],
                              show_tqdm=True,
                              skip_rdf=True,
                              gd_score_func='discriminator',
                              )

            batch_size = self.config.max_batch_size
            num_samples = self.config.num_samples

            sampling_dict = sampler.crystal_search(sample,
                                                   'prior',
                                                   num_samples,
                                                   batch_size,
                                                   torch.LongTensor([14 for _ in range(batch_size)]),
                                                   packing_coeff_range=[0.5, 2.0],
                                                   vdw_threshold=10000,
                                                   rdf_cutoff=0.025,
                                                   opt_eps=1e-2,
                                                   cell_params_cutoff=0.05,
                                                   chain_length=10,
                                                   step_size=0.1,  # 0.1
                                                   parallel=True,
                                                   )
            np.save(f'../sampling_results/{str(int(sample.identifier))}_{self.run_identifier}_sampling_results',
                    sampling_dict)

    def ae_annealing(self):
        # if we have learned the existing distribution AND there are no orphaned nodes
        mean_loss = self.logger.train_stats['reconstruction_loss'][-100:].mean()
        if mean_loss < self.config.autoencoder.sigma_threshold:
            # and we more self-overlap than desired
            mean_self_overlap_loss = np.abs(1 - self.logger.train_stats['mean_self_overlap'][-100:]).mean()
            if mean_self_overlap_loss > self.config.autoencoder.overlap_eps.test:
                # tighten the target distribution
                self.config.autoencoder_sigma *= self.config.autoencoder.sigma_lambda

        # if we have way too much overlap, just tighten right away
        if (np.abs(1 - self.logger.train_stats['mean_self_overlap'][-100:]).mean()
                > self.config.autoencoder.max_overlap_threshold):
            self.config.autoencoder_sigma *= self.config.autoencoder.sigma_lambda

    def preprocess_ae_inputs(self,
                             mol_batch,
                             no_noise=False,
                             orientation_override=None,
                             noise=None,
                             deprotonation_override=False,
                             affine_scale=None):
        # atomwise random noise
        if not no_noise:  # TODO combine these args
            if noise is not None:
                mol_batch.pos += torch.randn_like(mol_batch.pos) * noise

            if affine_scale is not None:
                graph_scale_factor = torch.randn(mol_batch.num_graphs, device=mol_batch.pos.device, dtype=torch.float32)
                graph_scale_factor = (graph_scale_factor / 10 + affine_scale).clip(min=0.9, max=1.25)
                atomwise_scaling = graph_scale_factor.repeat_interleave(mol_batch.num_atoms)
                mol_batch.pos *= atomwise_scaling[:, None]

        # random global roto-inversion
        if orientation_override is not None:
            mol_batch = set_molecule_alignment(mol_batch,
                                               mode=orientation_override,
                                               right_handed=False,
                                               include_inversion=True)

        # optionally, deprotonate
        input_data = self.fix_autoencoder_protonation(mol_batch,
                                                      override_deprotonate=deprotonation_override)

        # subtract mean OF THE INPUT from BOTH reference and input
        centroids = scatter(input_data.pos, input_data.batch, reduce='mean', dim=0)
        mol_batch.pos -= torch.repeat_interleave(centroids, mol_batch.num_atoms, dim=0, output_size=mol_batch.num_nodes)
        input_data.pos -= torch.repeat_interleave(centroids, input_data.num_atoms, dim=0,
                                                  output_size=input_data.num_nodes)

        return mol_batch, input_data

    def compute_autoencoder_loss(self,
                                 decoding: torch.Tensor,
                                 mol_batch: CrystalData,
                                 skip_stats: bool = False,
                                 ) -> Tuple[torch.Tensor, dict, CrystalData]:
        """
        Function for analyzing autoencoder outputs and calculating loss & other key metrics
        1) process inputs and outputs into the correct format
        2) compute relevant losses, reconstruction, radial constraint, weight constraint
        Parameters
        ----------
        decoding : Tensor
            raw output for gaussian mixture
        mol_batch : CrystalData
            Input data to be reconstructed
        skip_stats : bool
            Whether to skip saving summary statistics for this step

        Returns
        -------

        """
        # reduce to relevant atom types
        mol_batch.x = self.models_dict['autoencoder'].atom_embedding_vector[mol_batch.x].flatten()
        decoded_mol_batch, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
            collate_decoded_data(mol_batch,
                                 decoding,
                                 self.models_dict['autoencoder'].num_decoder_nodes,
                                 self.config.autoencoder.node_weight_temperature,
                                 self.device))

        (nodewise_reconstruction_loss,
         nodewise_type_loss,
         reconstruction_loss,
         self_likelihoods,
         nearest_node_loss,
         clumping_loss,
         nearest_component_dist,
         nearest_component_loss,
         ) = ae_reconstruction_loss(
            mol_batch,
            decoded_mol_batch,
            nodewise_weights,
            nodewise_weights_tensor,
            self.dataDims['num_atom_types'],
            self.config.autoencoder.type_distance_scaling,
            self.config.autoencoder_sigma,
        )

        matching_nodes_fraction = torch.sum(
            nodewise_reconstruction_loss < 0.01) / mol_batch.num_nodes  # within 1% matching

        # node radius constraining loss
        decoded_dists = torch.linalg.norm(decoded_mol_batch.pos, dim=1)
        constraining_loss = scatter(
            F.relu(
                decoded_dists -  # self.models_dict['autoencoder'].radial_normalization),
                torch.repeat_interleave(mol_batch.radius, self.models_dict['autoencoder'].num_decoder_nodes, dim=0)),
            decoded_mol_batch.batch, reduce='mean')

        # node weight constraining loss
        equal_to_actual_difference = (nodewise_graph_weights - nodewise_weights_tensor) / nodewise_graph_weights
        # we don't want nodewise_weights_tensor to be too small, so equal_to_acutal_difference shouldn't be too positive
        nodewise_constraining_loss = F.relu(
            equal_to_actual_difference - self.config.autoencoder.weight_constraint_factor)
        node_weight_constraining_loss = scatter(
            nodewise_constraining_loss,
            decoded_mol_batch.batch,
            dim=0,
            dim_size=decoded_mol_batch.num_graphs,
            reduce='mean',
        )

        # sum losses
        losses = (reconstruction_loss +
                  constraining_loss +
                  node_weight_constraining_loss +
                  #self.config.autoencoder.nearest_node_loss_coefficient * nearest_node_loss**2 +
                  self.config.autoencoder.nearest_component_loss_coefficient * nearest_component_loss ** 2
                  #self.config.autoencoder.clumping_loss_coefficient * clumping_loss
                  )

        if not skip_stats:
            stats = {'constraining_loss': constraining_loss.mean().detach(),
                     'reconstruction_loss': reconstruction_loss.mean().detach(),
                     'nodewise_type_loss': nodewise_type_loss.detach(),
                     'scaled_reconstruction_loss': (
                             reconstruction_loss.mean() * self.config.autoencoder_sigma).detach(),
                     'sigma': self.config.autoencoder_sigma,
                     'mean_self_overlap': scatter(self_likelihoods, mol_batch.batch, reduce='mean').mean().detach(),
                     'matching_nodes_fraction': matching_nodes_fraction.detach(),
                     'matching_nodes_loss': 1 - matching_nodes_fraction.detach(),
                     'node_weight_constraining_loss': node_weight_constraining_loss.mean().detach(),
                     'nearest_node_loss': nearest_node_loss.detach().mean(),
                     'clumping_loss': clumping_loss.detach().mean(),
                     'nearest_component_max_dist': nearest_component_dist.max().detach(),
                     'nearest_component_loss': nearest_component_loss.mean().detach(),
                     }
        else:
            stats = {}

        return losses, stats, decoded_mol_batch

    def regression_epoch(self, data_loader, update_weights=True, iteration_override=None):
        if update_weights:
            self.models_dict['regressor'].train(True)
        else:
            self.models_dict['regressor'].eval()

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            if self.config.positional_noise.regressor > 0:
                data.pos += torch.randn_like(data.pos) * self.config.positional_noise.regressor

            data = data.to(self.device)

            regression_losses_list, predictions, targets = get_regression_loss(
                self.models_dict['regressor'], data, data.y, self.dataDims['target_mean'], self.dataDims['target_std'])
            regression_loss = regression_losses_list.mean()

            if update_weights:
                self.optimizers_dict['regressor'].zero_grad(set_to_none=True)  # reset gradients from previous passes
                regression_loss.backward()  # back-propagation
                self.optimizers_dict['regressor'].step()  # update parameters

            '''log losses and other tracking values'''
            self.logger.update_current_losses('regressor', self.epoch_type,
                                              regression_loss.cpu().detach().numpy(),
                                              regression_losses_list.cpu().detach().numpy())

            stats = {'regressor_prediction': predictions,
                     'regressor_target': targets}
            dict_of_tensors_to_cpu_numpy(stats)
            self.logger.update_stats_dict(self.epoch_type, stats.keys(), stats.values(), mode='extend')
            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early

        self.logger.concatenate_stats_dict(self.epoch_type)

    def polymorph_classification_epoch(self, data_loader, update_weights=True, iteration_override=None):
        if update_weights:
            self.models_dict['polymorph_classifier'].train(True)
        else:
            self.models_dict['polymorph_classifier'].eval()

        stats_keys = ['true_labels', 'probs']

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            if self.config.positional_noise.polymorph_classifier > 0:
                data.pos += torch.randn_like(data.pos) * self.config.positional_noise.polymorph_classifier

            data = data.to(self.device)
            output = self.models_dict['polymorph_classifier'](data.clone(),
                                                              return_latent=False,
                                                              return_embedding=False)
            loss = F.cross_entropy(output[:, :self.dataDims['num_polymorphs']], data.polymorph, reduction='none')  # +

            if update_weights:
                loss.mean().backward()
                # use gradient accumulation for synthetically larger batch sizes
                if i % data_loader.batch_size == 0 or i == len(data_loader) - 1:
                    self.optimizers_dict['polymorph_classifier'].step()  # update parameters
                    self.optimizers_dict['polymorph_classifier'].zero_grad(
                        set_to_none=True)  # reset gradients from previous passes

            if i % self.config.logger.stats_reporting_frequency == 0:
                '''log losses and other tracking values'''
                self.logger.update_current_losses('polymorph_classifier', self.epoch_type,
                                                  loss.mean().cpu().detach().numpy(),
                                                  loss.cpu().detach().numpy())

                stats_values = [data.polymorph.detach(), output.detach()]  #, data.cluster_type]
                stats = {key: value for key, value in zip(stats_keys, stats_values)}
                dict_of_tensors_to_cpu_numpy(stats)
                self.logger.update_stats_dict(self.epoch_type,
                                              stats.keys(),
                                              stats.values(),
                                              mode='append')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early

        self.logger.concatenate_stats_dict(self.epoch_type)

    def discriminator_epoch(self,
                            data_loader=None,
                            update_weights=True,
                            iteration_override=None):

        if not hasattr(self, 'generator_prior'):  # first GAN epoch
            self.init_gan_constants()

        if update_weights:
            self.models_dict['discriminator'].train(True)
        else:
            self.models_dict['discriminator'].eval()

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 10), mininterval=30)):
            data = data.to(self.config.device)

            '''
            train discriminator
            '''
            self.discriminator_step(data, i, update_weights, skip_step=False)

            '''
            record some stats
            '''
            self.logger.update_stats_dict(self.epoch_type, ['identifiers'], data.identifier, mode='extend')
            self.logger.update_stats_dict(self.epoch_type, ['smiles'], data.smiles, mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early

        self.logger.concatenate_stats_dict(self.epoch_type)

    def pd_epoch(self,
                 data_loader=None,
                 update_weights=True,
                 iteration_override=None):

        if not hasattr(self, 'generator_prior'):  # first GAN epoch
            self.init_gan_constants()

        if self.config.dataset.otf_build_size > 0 and self.epoch_type == 'train' and os.cpu_count() > 1:
            self.train_loader_to_replace = self.otf_crystal_dataset_generation(data_loader)
            data_loader = self.train_loader_to_replace

        self.models_dict['autoencoder'].eval()

        if update_weights:
            self.models_dict['proxy_discriminator'].train(True)
            if self.config.proxy_discriminator.embedding_type == 'autoencoder' and self.config.proxy_discriminator.train_encoder:
                self.models_dict['autoencoder'].train(True)
        else:
            self.models_dict['proxy_discriminator'].eval()
            if self.config.proxy_discriminator.embedding_type == 'autoencoder' and self.config.proxy_discriminator.train_encoder:
                self.models_dict['autoencoder'].eval()

        self.crystal_builder.rotation_basis = 'cartesian'

        if self.logger.epoch == 0:  # embed loaded dataset on first epoch
            self.embed_dataloader_dataset(data_loader)

        for step, mol_batch in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 10), mininterval=30)):
            if step % self.config.logger.stats_reporting_frequency == 0:
                skip_stats = False
            elif step == len(data_loader) - 1:
                skip_stats = False
            else:
                skip_stats = True

            mol_batch = mol_batch.to(self.config.device)
            '''
            train proxy discriminator
            '''
            self.pd_step(mol_batch, step, update_weights, skip_step=False, skip_stats=skip_stats)

            if iteration_override is not None:
                if step >= iteration_override:
                    break  # stop training early

        self.logger.concatenate_stats_dict(self.epoch_type)

    def embed_dataloader_dataset(self, data_loader):
        embeddings = []
        batch_size = data_loader.batch_size
        num_chunks = len(data_loader.dataset) // batch_size + int(len(data_loader.dataset) % batch_size != 0)
        with torch.no_grad():
            for ind in range(num_chunks):  # do it this way so to avoid shuffling
                mol_batch = self.collater(data_loader.dataset[ind * batch_size:(ind + 1) * batch_size]).to(self.device)
                cell_params = torch.cat([
                    mol_batch.cell_lengths, mol_batch.cell_angles, mol_batch.pose_params0
                ], dim=1)
                embeddings.append(self._embed_mol(mol_batch, mol_batch.pos, cell_params).cpu().detach())

        embeddings = torch.cat(embeddings, dim=0).to('cpu')
        for ind in range(len(data_loader.dataset)):
            data_loader.dataset[ind].y = embeddings[None, ind]

        '''
from mxtaltools.dataset_management.dataset_generation.generate_dataset_from_smiles import test_crystal_rebuild_from_embedding
mol_batch = next(iter(data_loader))

opt_vdw_pot = mol_batch.vdw_pot[None, ...].cpu()
opt_vdw_loss = mol_batch.vdw_loss[None, ...].cpu()
opt_aunits = mol_batch.pos[None,...].cpu()
cell_params = torch.cat([
    mol_batch.y[:, -9:].cpu(), torch.ones((mol_batch.num_graphs,3))], dim=1)[None, ...].cpu()

r_pot, r_loss, r_au = test_crystal_rebuild_from_embedding(
    mol_batch.clone().cpu(),
    opt_vdw_pot,
    opt_vdw_loss,
    opt_aunits,
    cell_params,
    denorm=True,
    destd=True,
    renorm=False,
    restd=False,
    make_figs=False,
)


'''

    # def update_buffers(self):
    #     # reprocess buffers
    #     if self.epoch_type == 'train':
    #         buffer = self.train_buffer
    #     elif self.epoch_type == 'test':
    #         buffer = self.test_buffer
    #     else:
    #         assert False
    #
    #     new_samples = torch.stack(buffer['new_samples'], dim=0)
    #     new_values = torch.tensor(buffer['new_values'], dtype=torch.float32, device='cpu')
    #
    #     if self.logger.epoch == 0:
    #         # first epoch, no new samples
    #         buffer['samples'] = new_samples
    #         buffer['values'] = new_values
    #
    #     elif len(buffer['new_samples']) + len(buffer['samples']) <= self.config.dataset.buffer_size:
    #         # take everything during the buffer building phase
    #         buffer['samples'] = torch.cat([buffer['samples'], new_samples], dim=0)
    #         buffer['values'] = torch.cat([buffer['values'], new_values], dim=0)
    #     else:
    #         # more samples than we need
    #         # replace old samples with new samples
    #         # weigh samples to replace by energy
    #
    #         temperature = 10
    #         old_samples = buffer['samples']
    #         old_values = buffer['values']
    #         samples_to_add = min(len(new_samples), self.config.dataset.buffer_size)
    #         # old_values = old_values.clip(max=100)
    #         # probs = np.exp(old_values.numpy() / temperature) / np.sum(
    #         #     np.exp(old_values.numpy() / temperature))
    #         # probs = np.nan_to_num(probs, posinf=0, neginf=0, nan=0)
    #         # probs /= probs.sum()
    #         old_rands = np.random.choice(len(old_samples),
    #                                      samples_to_add,
    #                                      #p=probs,
    #                                      replace=False
    #                                      )
    #         old_samples[old_rands] = new_samples[:self.config.dataset.buffer_size]
    #         old_values[old_rands] = new_values[:self.config.dataset.buffer_size]
    #         buffer['samples'] = old_samples[:self.config.dataset.buffer_size]
    #         buffer['values'] = old_values[:self.config.dataset.buffer_size]
    #
    #     buffer['new_samples'], buffer['new_values'] = [], []
    #     if self.epoch_type == 'train':
    #         self.logger.train_buffer_size = len(buffer['samples'])
    #     elif self.epoch_type == 'test':
    #         self.logger.test_buffer_size = len(buffer['samples'])

    # def init_sample_buffers(self):
    #     self.train_buffer = {'samples': torch.zeros((self.config.dataset.buffer_size,
    #                                                  self.config.proxy_discriminator.model.bottleneck_dim + 9),
    #                                                 dtype=torch.float32, device='cpu'),
    #                          'values': torch.zeros(self.config.proxy_discriminator.model.bottleneck_dim,
    #                                                dtype=torch.float32, device='cpu'),
    #                          'new_samples': [], 'new_values': []
    #                          }
    #     self.test_buffer = {'samples': torch.zeros((self.config.dataset.buffer_size,
    #                                                 self.config.proxy_discriminator.model.bottleneck_dim + 9),
    #                                                dtype=torch.float32, device='cpu'),
    #                         'values': torch.zeros(self.config.dataset.buffer_size,
    #                                               dtype=torch.float32, device='cpu'),
    #                         'new_samples': [], 'new_values': []
    #                         }

    def generator_epoch(self,
                        data_loader=None,
                        update_weights=True,
                        iteration_override=None):

        if not hasattr(self, 'generator_prior'):  # first GAN epoch
            self.init_gan_constants()

        if update_weights:
            self.models_dict['generator'].train(True)
        else:
            self.models_dict['generator'].eval()

        self.models_dict['autoencoder'].eval()

        for step, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 10), mininterval=30)):
            data = data.to(self.config.device)

            if step % self.config.logger.stats_reporting_frequency == 0:
                skip_stats = False
            elif step == len(data_loader) - 1:
                skip_stats = False
            else:
                skip_stats = True
            '''
            train_generator
            '''
            self.generator_step(data, step, update_weights, skip_stats)

            if iteration_override is not None:
                if step >= iteration_override:
                    break  # stop training early

        self.logger.concatenate_stats_dict(self.epoch_type)

    def init_gan_constants(self):
        self.prior_loss_coefficient = self.config.generator.prior_loss_coefficient
        self.prior_variation_scale = 0.05
        self.vdw_loss_coefficient = 0.01
        self.vdw_turnover_potential = 0.01

        '''set space groups to be included and generated'''
        if self.config.generate_sgs == 'all':
            self.config.generate_sgs = [self.sym_info['space_groups'][int(key)] for key in ASYM_UNITS.keys()]

        self.generator_prior = CSDPrior(
            sym_info=self.sym_info, device=self.config.device,
            cell_means=self.lattice_means[:6],
            cell_stds=self.lattice_stds[:6],
            lengths_cov_mat=self.dataDims['lattice_length_cov_mat'], )

    def adversarial_score(self, data, return_latent=False):
        """
        get the score from the discriminator on data
        """
        output, extra_outputs = self.models_dict['discriminator'](
            data.clone(), return_dists=True, return_latent=return_latent)

        if return_latent:
            return output, extra_outputs['dists_dict'], extra_outputs['final_activation']
        else:
            return output, extra_outputs['dists_dict']

    def discriminator_step(self, data, i, update_weights, skip_step):
        """
        execute a complete training step for the discriminator
        compute losses, do reporting, update gradients
        """
        if self.train_models_dict['discriminator']:
            (discriminator_output_on_real, discriminator_output_on_fake,
             real_fake_rdf_distances, stats) \
                = self.get_discriminator_output(data, i)

            discriminator_losses, loss_stats = self.aggregate_discriminator_losses(
                discriminator_output_on_real,
                discriminator_output_on_fake,
                real_fake_rdf_distances)

            stats.update(loss_stats)
            discriminator_loss = discriminator_losses.mean()

            if update_weights and (not skip_step):
                self.optimizers_dict['discriminator'].zero_grad(
                    set_to_none=True)  # reset gradients from previous passes
                discriminator_loss.backward()  # back-propagation
                torch.nn.utils.clip_grad_norm_(self.models_dict['discriminator'].parameters(),
                                               self.config.gradient_norm_clip)  # gradient clipping
                self.optimizers_dict['discriminator'].step()  # update parameters

            # don't move anything to the CPU until after the backward pass
            self.logger.update_current_losses('discriminator', self.epoch_type,
                                              discriminator_losses.mean().cpu().detach().numpy(),
                                              discriminator_losses.cpu().detach().numpy())

            dict_of_tensors_to_cpu_numpy(stats)

            self.logger.update_stats_dict(self.epoch_type,
                                          stats.keys(),
                                          stats.values(),
                                          mode='extend')

    def pd_step(self, mol_batch, i, update_weights, skip_step, skip_stats: bool = False):
        """
        execute a complete training step for the discriminator
        compute losses, do reporting, update gradients
        """
        if not hasattr(self.models_dict['proxy_discriminator'], 'target_std'):
            self.models_dict['proxy_discriminator'].target_std = 8.23
            self.models_dict['proxy_discriminator'].target_mean = -8.6

        if self.config.proxy_discriminator.embedding_type == 'autoencoder' and self.config.proxy_discriminator.train_encoder:
            cell_params = torch.cat([
                mol_batch.cell_lengths, mol_batch.cell_angles, mol_batch.pose_params0
            ], dim=1)
            mol_batch.y = self._embed_mol(mol_batch, mol_batch.pos, cell_params)

        prediction = self.models_dict['proxy_discriminator'](x=mol_batch.y[:, :-3])[:,0]  # cut trailing 3 dimensions for P1 modelling

        discriminator_losses = F.smooth_l1_loss(prediction.flatten(),
                                                (mol_batch.vdw_loss.flatten() - self.models_dict[
                                                    'proxy_discriminator'].target_mean) / self.models_dict[
                                                    'proxy_discriminator'].target_std,
                                                reduction='none')

        discriminator_loss = discriminator_losses.mean()

        if update_weights and (not skip_step):
            self.optimizers_dict['proxy_discriminator'].zero_grad(
                set_to_none=True)  # reset gradients from previous passes
            if self.config.proxy_discriminator.embedding_type == 'autoencoder' and self.config.proxy_discriminator.train_encoder:
                self.optimizers_dict['autoencoder'].zero_grad(
                    set_to_none=True)  # reset gradients from previous passes
            discriminator_loss.backward()  # back-propagation
            torch.nn.utils.clip_grad_norm_(self.models_dict['proxy_discriminator'].parameters(),
                                           self.config.gradient_norm_clip)  # gradient clipping
            self.optimizers_dict['proxy_discriminator'].step()  # update parameters
            if self.config.proxy_discriminator.embedding_type == 'autoencoder' and self.config.proxy_discriminator.train_encoder:
                torch.nn.utils.clip_grad_norm_(self.models_dict['autoencoder'].parameters(),
                                               self.config.gradient_norm_clip)  # gradient clipping
                self.optimizers_dict['autoencoder'].step()  # update parameters

        # don't move anything to the CPU until after the backward pass
        self.logger.update_current_losses('proxy_discriminator', self.epoch_type,
                                          discriminator_losses.mean().cpu().detach().numpy(),
                                          discriminator_losses.cpu().detach().numpy())

        if not skip_stats:
            stats = {
                'vdw_prediction': (prediction * self.models_dict['proxy_discriminator'].target_std + self.models_dict[
                    'proxy_discriminator'].target_mean).detach(),
                'vdw_score': mol_batch.vdw_loss.detach(),
            }

            dict_of_tensors_to_cpu_numpy(stats)

            self.logger.update_stats_dict(self.epoch_type,
                                          stats.keys(),
                                          stats.values(),
                                          mode='extend')

    def aggregate_discriminator_losses(self,
                                       discriminator_output_on_real,
                                       discriminator_output_on_fake,
                                       real_fake_rdf_distances):

        combined_outputs = torch.cat((discriminator_output_on_real, discriminator_output_on_fake), dim=0)

        'classification loss'
        classification_target = torch.cat((torch.ones_like(discriminator_output_on_real[:, 0]),
                                           torch.zeros_like(discriminator_output_on_fake[:, 0])))
        classification_losses = F.cross_entropy(combined_outputs[:, :2], classification_target.long(), reduction='none')

        'rdf distance loss'
        rdf_distance_target = torch.log10(1 + torch.cat((torch.zeros_like(discriminator_output_on_real[:, 0]),
                                                         real_fake_rdf_distances)))  # rescale on log(1+x)
        rdf_distance_prediction = F.softplus(combined_outputs[:, 2])
        rdf_distance_losses = F.smooth_l1_loss(rdf_distance_prediction, rdf_distance_target,
                                               reduction='none') * 10  # rescale w.r.t., classification loss

        score_on_real = softmax_and_score(discriminator_output_on_real[:, :2])
        score_on_fake = softmax_and_score(discriminator_output_on_fake[:, :2])

        stats = {'discriminator_real_score': score_on_real.detach(),
                 'discriminator_fake_score': score_on_fake.detach(),
                 'discriminator_fake_true_distance': torch.log10(1 + real_fake_rdf_distances).detach(),
                 'discriminator_fake_predicted_distance': F.softplus(discriminator_output_on_fake[:, 2]).detach(),
                 'discriminator_real_true_distance': torch.zeros_like(discriminator_output_on_real[:, 0]).detach(),
                 'discriminator_real_predicted_distance': F.softplus(discriminator_output_on_real[:, 2]).detach(),
                 'discriminator_classification_loss': classification_losses.detach(),
                 'discriminator_distance_loss': rdf_distance_losses.detach()}

        discriminator_losses_list = []
        if self.config.discriminator.use_classification_loss:
            discriminator_losses_list.append(classification_losses)

        if self.config.discriminator.use_rdf_distance_loss:
            discriminator_losses_list.append(rdf_distance_losses)

        discriminator_losses = torch.sum(torch.stack(discriminator_losses_list), dim=0)

        return discriminator_losses, stats

    def generator_step(self, mol_batch, step, update_weights, skip_stats):
        """
        execute a complete training step for the generator
        get sample losses, do reporting, update gradients
        """

        mol_data, scalar_mol_embedding, vector_mol_embedding = self.process_embed_for_generator(
            mol_batch,
            self.config.generator.canonical_conformer_orientation,
            self.config.generate_sgs,
            no_noise=False)

        prior, vdw_loss = self.sample_and_build_prior(mol_data)

        for ind in range(self.config.generator.samples_per_iter):
            if ind == 0:
                init_state = prior.detach().clone()
                prev_vdw_loss = vdw_loss.detach().clone() / mol_data.num_atoms
            else:
                init_state = generator_raw_samples.detach().clone()
                prev_vdw_loss = vdw_loss.detach().clone()

            (generated_samples_to_build, generator_raw_samples,
             scaling_factor, variation_factor) = self.sample_from_generator(
                mol_data, init_state,
                scalar_mol_embedding, vector_mol_embedding,
                self.prior_variation_scale, self.device)

            if torch.sum(torch.logical_not(torch.isfinite(generator_raw_samples))):
                self.vdw_turnover_potential *= 0.75  # soften vdW
                raise ValueError("Numerical Error: Generated samples contain NaN")

            supercell_data, generated_cell_volumes = (
                self.crystal_builder.build_zp1_supercells(
                    mol_batch=mol_data.clone(),
                    cell_parameters=generated_samples_to_build,
                    supercell_size=self.config.supercell_size,
                    graph_convolution_cutoff=self.config.discriminator.model.graph.cutoff,
                    align_to_standardized_orientation=False,
                    skip_refeaturization=True,
                ))

            (eval_vdw_loss, raw_vdw_loss, vdw_potential,
             molwise_normed_overlap, prior_loss,
             sample_packing_coeff, scaled_deviation) = self.analyze_generated_crystals(
                mol_data,
                generated_cell_volumes,
                supercell_data,
                generator_raw_samples,
                variation_factor=variation_factor,
                prior=init_state
            )

            vdw_score = -molwise_normed_overlap / mol_data.num_atoms
            vdw_loss = raw_vdw_loss / mol_data.num_atoms
            eval_vdw_loss = eval_vdw_loss / mol_data.num_atoms
            supercell_data.loss = vdw_loss

            vdw_delta = vdw_loss - prev_vdw_loss
            generator_losses = (prior_loss * self.prior_loss_coefficient +
                                (vdw_delta) * self.vdw_loss_coefficient)

            generator_loss = generator_losses.mean()
            if torch.sum(torch.logical_not(torch.isfinite(generator_losses))) > 0:
                raise ValueError("Numerical Error: NaN in generator losses")

            if update_weights:
                self.optimizers_dict['generator'].zero_grad(set_to_none=True)  # reset gradients from previous passes
                generator_loss.backward()  # back-propagation
                torch.nn.utils.clip_grad_norm_(self.models_dict['generator'].parameters(),
                                               self.config.gradient_norm_clip)  # gradient clipping
                self.optimizers_dict['generator'].step()  # update parameters

            # scale these against sampling from the prior (most difficult)
            if ind == 0 and step > 0:
                good_inds = np.argwhere(np.array(self.logger.train_stats['generator_sample_iter']) == 0).flatten()
                if len(good_inds) > 1000:
                    if self.epoch_type == 'train':
                        self.anneal_prior_loss(good_inds)
                        self.anneal_vdw_turnover(good_inds)

            if not skip_stats:
                self.logger.update_current_losses('generator', self.epoch_type,
                                                  generator_loss.data.detach().cpu().numpy(),
                                                  generator_losses.detach().cpu().numpy())
                stats = {
                    'generated_space_group_numbers': supercell_data.sg_ind.detach(),
                    'identifiers': mol_batch.identifier,
                    'smiles': mol_batch.smiles,
                    'generator_vdw_delta': vdw_delta.detach(),
                    'generator_per_mol_vdw_loss': eval_vdw_loss.detach(),
                    'generator_per_mol_raw_vdw_loss': vdw_loss.detach(),
                    'generator_per_mol_vdw_score': vdw_score.detach(),
                    'generator_prior_loss': prior_loss.detach(),
                    'generator_packing_prediction': sample_packing_coeff.detach(),
                    'generator_sample_iter': torch.ones(len(prior_loss)) * ind,
                    'generator_prior': prior.detach(),
                    'generator_scaling_factor': scaling_factor.detach(),
                    'generated_cell_parameters': torch.cat(
                        [generator_raw_samples[:, :9], supercell_data.cell_params[:, 9:]], dim=1).detach(),
                    'generator_scaled_deviation': scaled_deviation.detach(),
                    'generator_sample_vdw_energy': vdw_potential.detach(),
                    'generator_variation_factor': variation_factor.detach(),
                }
                if step == 0:
                    stats['generator_samples'] = supercell_data.detach()

                dict_of_tensors_to_cpu_numpy(stats)

                self.logger.update_stats_dict(self.epoch_type,
                                              stats.keys(),
                                              stats.values(),
                                              mode='extend')

    def sample_and_build_prior(self, mol_data):
        prior, descaled_prior = self.sample_from_prior(mol_data)
        supercell_data, generated_cell_volumes = (
            self.crystal_builder.build_zp1_supercells(
                mol_batch=mol_data,
                cell_parameters=descaled_prior,
                supercell_size=self.config.supercell_size,
                graph_convolution_cutoff=self.config.discriminator.model.graph.cutoff,
                align_to_standardized_orientation=False,
                skip_refeaturization=True,
            ))
        (eval_vdw_loss, vdw_loss, vdw_potential,
         molwise_normed_overlap, prior_loss,
         packing_coeff, scaled_deviation) = self.analyze_generated_crystals(
            mol_data.clone(),
            generated_cell_volumes,
            supercell_data,
        )
        return prior, vdw_loss

    def analyze_generated_crystals(self,
                                   mol_data,
                                   generated_cell_volumes,
                                   supercell_batch,
                                   generator_raw_samples=None,
                                   variation_factor=None,
                                   prior=None,
                                   return_dist_dict=False):
        if prior is not None:
            prior_loss, scaled_deviation = compute_prior_loss(
                self.generator_prior.norm_factors,
                mol_data.sg_ind,
                generator_raw_samples,
                prior,
                variation_factor)
        else:
            prior_loss = torch.zeros_like(generated_cell_volumes)
            scaled_deviation = torch.zeros_like(generated_cell_volumes)

        reduced_volume = generated_cell_volumes / supercell_batch.sym_mult
        packing_coeff = mol_data.mol_volume / reduced_volume

        dist_dict = get_intermolecular_dists_dict(supercell_batch, 6, 100)
        molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, lj_pot \
            = vdw_analysis(self.vdw_radii_tensor, dist_dict, mol_data.num_graphs)

        if return_dist_dict:
            return (vdw_potential, vdw_loss, vdw_potential,
                    molwise_normed_overlap, prior_loss,
                    packing_coeff, scaled_deviation, dist_dict)

        else:
            return (vdw_potential, vdw_loss, vdw_potential,
                    molwise_normed_overlap, prior_loss,
                    packing_coeff, scaled_deviation)

    def sample_from_generator(self,
                              mol_data,
                              prior,
                              scalar_mol_embedding,
                              vector_mol_embedding,
                              variation_scale,
                              device,
                              variation_factor=None
                              ):

        if variation_factor is None:
            variation_factor = sample_uniform(mol_data.num_graphs,
                                              variation_scale,
                                              device)

        scaling_factor = (self.generator_prior.norm_factors[mol_data.sg_ind, :] + 1e-4)

        absolute_reference = torch.eye(3, dtype=torch.float32, device=device
                                       ).reshape(1, 9).repeat(mol_data.num_graphs, 1)

        conditioning_vector = torch.cat((
            scalar_mol_embedding,
            prior,
            absolute_reference,
            variation_factor[:, None],
            scaling_factor),
            dim=1)

        generator_raw_samples = self.models_dict['generator'].forward(
            conditioning_vector,
            vector_mol_embedding,
            mol_data.sg_ind,
            prior,
        )

        generated_samples_to_build = denormalize_generated_cell_params(
            generator_raw_samples,
            mol_data,
            self.crystal_builder.asym_unit_dict
        )

        return generated_samples_to_build, generator_raw_samples, scaling_factor, variation_factor

    def sample_from_prior(self, mol_batch) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_samples = self.generator_prior(mol_batch.num_graphs, mol_batch.sg_ind).to(self.device)

        samples_to_build = denormalize_generated_cell_params(
            raw_samples,
            mol_batch,
            self.crystal_builder.asym_unit_dict
        )

        return raw_samples, samples_to_build

    def process_embed_for_generator(self, data, orientation, generate_sgs, no_noise=False):
        # clone, center, and orient the molecules
        _, mol_data = self.preprocess_ae_inputs(
            data,
            no_noise=any([no_noise, self.config.positional_noise.generator == 0]),
            noise=self.config.positional_noise.generator,
            orientation_override=orientation)

        # embed molecules
        with torch.no_grad():
            vector_mol_embedding = self.models_dict['autoencoder'].encode(mol_data.clone())
            scalar_mol_embedding = self.models_dict['autoencoder'].scalarizer(vector_mol_embedding)

        mol_data = overwrite_symmetry_info(mol_data,
                                           generate_sgs,
                                           self.sym_info,
                                           randomize_sgs=True)

        return mol_data, scalar_mol_embedding, vector_mol_embedding

    def get_discriminator_output(self, data, i):
        """
        generate real and fake crystals
        and score them
        """
        '''get real supercells'''  # todo revise this function with packing coeff, vdw overlap calc
        real_supercell_data = self.crystal_builder.prebuilt_unit_cell_to_supercell(
            data, self.config.supercell_size, self.config.discriminator.model.graph.cutoff)

        '''get fake supercells'''
        generated_samples_i, negative_type, generator_data, negatives_stats = \
            self.generate_discriminator_negatives(data,
                                                  train_adversarial=self.config.discriminator.train_adversarially,
                                                  train_on_randn=self.config.discriminator.train_on_randn,
                                                  train_on_distorted=self.config.discriminator.train_on_distorted,
                                                  orientation='random')

        fake_supercell_data, generated_cell_volumes = self.crystal_builder.build_zp1_supercells(
            generator_data, generated_samples_i, self.config.supercell_size,
            self.config.discriminator.model.graph.cutoff,
            align_to_standardized_orientation=(negative_type != 'generated'),  # take generator samples as-given
            target_handedness=generator_data.aunit_handedness,
            skip_refeaturization=True,
        )

        canonical_fake_cell_params = fake_supercell_data.cell_params

        '''apply noise'''
        if self.config.positional_noise.discriminator > 0:
            real_supercell_data.pos += \
                torch.randn_like(real_supercell_data.pos) * self.config.positional_noise.discriminator
            fake_supercell_data.pos += \
                torch.randn_like(fake_supercell_data.pos) * self.config.positional_noise.discriminator

        '''score'''
        discriminator_output_on_real, real_pairwise_distances_dict = self.adversarial_score(
            real_supercell_data, return_latent=False)
        discriminator_output_on_fake, fake_pairwise_distances_dict = self.adversarial_score(
            fake_supercell_data, return_latent=False)

        '''recompute reduced volumes'''
        # TODO convert this back to packing coefficient calculations
        real_volume_fractions = compute_reduced_volume_fraction(cell_lengths=real_supercell_data.cell_lengths,
                                                                cell_angles=real_supercell_data.cell_angles,
                                                                atom_radii=self.vdw_radii_tensor[
                                                                    data.x.long().flatten()],
                                                                batch=data.batch,
                                                                crystal_multiplicity=real_supercell_data.sym_mult)
        fake_volume_fractions = compute_reduced_volume_fraction(cell_lengths=fake_supercell_data.cell_lengths,
                                                                cell_angles=fake_supercell_data.cell_angles,
                                                                atom_radii=self.vdw_radii_tensor[
                                                                    data.x.long().flatten()],
                                                                batch=data.batch,
                                                                crystal_multiplicity=fake_supercell_data.sym_mult)

        '''distances'''
        if self.config.discriminator.use_rdf_distance_loss:
            real_rdf, rr, _ = new_crystal_rdf(real_supercell_data, real_pairwise_distances_dict,
                                              rrange=[0, self.config.discriminator.model.graph.cutoff], bins=2000,
                                              mode='intermolecular', elementwise=True, raw_density=True,
                                              cpu_detach=False)
            fake_rdf, _, _ = new_crystal_rdf(fake_supercell_data, fake_pairwise_distances_dict,
                                             rrange=[0, self.config.discriminator.model.graph.cutoff], bins=2000,
                                             mode='intermolecular', elementwise=True, raw_density=True,
                                             cpu_detach=False)

            rdf_dists = torch.zeros(real_supercell_data.num_graphs, device=self.config.device, dtype=torch.float32)
            for i in range(real_supercell_data.num_graphs):
                rdf_dists[i] = compute_rdf_distance(real_rdf[i], fake_rdf[i], rr) / real_supercell_data.num_atoms[i]
                # divides out the trivial size correlation
        else:
            rdf_dists = torch.randn(real_supercell_data.num_graphs, device=self.config.device,
                                    dtype=torch.float32).abs()  # dummy

        stats = {'real_vdw_penalty': -vdw_overlap(self.vdw_radii, crystaldata=real_supercell_data,
                                                  return_score_only=True).detach(),
                 'fake_vdw_penalty': -vdw_overlap(self.vdw_radii, crystaldata=fake_supercell_data,
                                                  return_score_only=True).detach(),
                 'real_cell_parameters': torch.cat([data.cell_lengths, data.cell_angles, data.pose_params0],
                                                   dim=1).detach(),
                 'generated_cell_parameters': canonical_fake_cell_params.detach(),
                 'real_volume_fractions': real_volume_fractions.detach(),
                 'generated_volume_fractions': fake_volume_fractions.detach(),
                 'fake_volume_fractions': fake_volume_fractions.detach()}

        stats.update(negatives_stats)

        return (discriminator_output_on_real, discriminator_output_on_fake,
                rdf_dists, stats)

    #
    # def sample_from_buffer(self, i):
    #     stats = {}
    #     sample_inds = torch.arange(i * self.config.current_batch_size, (i + 1) * self.config.current_batch_size)
    #     if self.epoch_type == 'train':
    #         embedding = self.train_buffer['samples'][sample_inds].to(self.device)
    #         vdw_loss = self.train_buffer['values'][sample_inds].to(self.device)
    #
    #     elif self.epoch_type == 'test':
    #         embedding = self.test_buffer['samples'][sample_inds].to(self.device)
    #         vdw_loss = self.test_buffer['values'][sample_inds].to(self.device)
    #     else:
    #         assert False
    #     return embedding, stats, vdw_loss
    #
    # def proxy_discriminator_sampling(self, mol_batch):
    #     with ((((torch.no_grad())))):
    #
    #         '''get initial random crystal'''
    #         mol_batch = set_molecule_alignment(mol_batch,
    #                                            mode='random',
    #                                            right_handed=False,
    #                                            include_inversion=True)
    #
    #         mol_batch = overwrite_symmetry_info(mol_batch,
    #                                             self.config.generate_sgs,
    #                                             self.sym_info,
    #                                             randomize_sgs=True)
    #
    #         raw_samples, generated_samples = self.sample_from_prior(mol_batch)
    #
    #         # even simpler for P1
    #         # generated_samples[:, 3:6] = torch.pi / 2
    #         # generated_samples[:, 6:9] = 0.5
    #
    #         supercell_batch, generated_cell_volumes = self.crystal_builder.build_zp1_supercells(
    #             mol_batch,
    #             generated_samples,
    #             self.config.supercell_size,
    #             self.config.proxy_discriminator.cutoff,
    #             align_to_standardized_orientation=False,  # take generator samples as-given
    #             target_handedness=None,
    #             skip_refeaturization=True,
    #         )
    #
    #         dist_dict = get_intermolecular_dists_dict(supercell_batch, 6, 100)
    #         _, _, vdw_potential, vdw_loss, lj_pot \
    #             = vdw_analysis(self.vdw_radii_tensor, dist_dict, mol_batch.num_graphs,
    #                            self.config.proxy_discriminator.vdw_turnover_potential, )
    #
    #         p1 = supercell_batch.pos[supercell_batch.aux_ind == 0]  # write crystal orientation to molecule
    #         p1 = torch.cat(
    #             [p1[mol_batch.batch == ind] - p1[mol_batch.batch == ind].mean(0) for ind in range(mol_batch.num_graphs)
    #              ])
    #         embedding = self._embed_mol(mol_batch, p1, supercell_batch.cell_params)
    #
    #         # reduced_volume = generated_cell_volumes / supercell_batch.sym_mult
    #         # packing_coeff = mol_batch.mol_volume / reduced_volume
    #         mol_batch.pos = p1  # reassign aunit according to supercell, with centring
    #
    #         '''generate additional samples by gradient descent'''
    #         self.mp_pool = mp.Pool(1)
    #         if os.path.exists('local_opt.pt') or :
    #             self.mp_pool.join()
    #             opt_dict = torch.load('local_opt.pt')
    #             opt_cell_params, opt_aunits, opt_packing_coeff, opt_vdw_loss, opt_vdw_pot = opt_dict['std_cell_params'], opt_dict['aunit_poses'], opt_dict['packing_coeff'], opt_dict['overall_loss'], opt_dict['vdw_potential']
    #             opt_embeddings = []
    #             for ind in range(len(opt_aunits)):
    #                 opt_embeddings.append(self._embed_mol(mol_batch.cuda(),
    #                                                       opt_aunits[ind].cuda(),
    #                                                       opt_cell_params[ind].cuda(),
    #                                                       norm_by_size=False,
    #                                                       ))
    #             self.otf_local_opt(mol_batch, p1, raw_samples)
    #
    #         stats = {
    #             'vdw_potential': (opt_vdw_pot / mol_batch.num_atoms[None, :].cpu()).detach(),
    #             'generated_cell_parameters': opt_cell_params.detach(),
    #             'packing_coeff': opt_packing_coeff.detach(),
    #             'vdw_loss': (opt_vdw_loss / mol_batch.num_atoms[None, :].cpu()).detach(),
    #         }
    #
    #         if self.epoch_type == 'train':
    #             buffer = self.train_buffer
    #         elif self.epoch_type == 'test':
    #             buffer = self.test_buffer
    #         buffer['new_samples'].append(torch.cat(opt_embeddings).cpu().detach())
    #         buffer['new_values'].extend(opt_vdw_loss.cpu().detach())
    #
    #     return mol_batch, embedding, stats, vdw_loss / mol_batch.num_atoms
    #
    # def otf_local_opt(self, mol_batch, raw_samples):
    #     sampler = Sampler(0,
    #                       'cpu',
    #                       self.config.machine,
    #                       None,
    #                       None,
    #                       None,
    #                       None,
    #                       show_tqdm=False,
    #                       skip_rdf=True,
    #                       gd_score_func='vdW',
    #                       num_cpus=1,
    #                       )
    #     opt_vdw_pot, opt_vdw_loss, opt_packing_coeff, opt_cell_params, opt_aunits = sampler.local_opt_for_proxy_discrim(
    #         mol_batch.clone().cpu(),
    #         raw_samples.cpu(),
    #         opt_eps=1e-1,
    #     )
    #
    #     return opt_cell_params, opt_aunits, opt_packing_coeff, opt_vdw_loss, opt_vdw_pot

    def _embed_mol(self, mol_batch, aunit_coordinates, cell_params, norm_by_size=True):
        mol_batch.pos = aunit_coordinates
        embedding = self.get_mol_embedding_for_proxy(mol_batch.clone())
        scaled_params = self.rescale_proxy_discrim_cell_params(mol_batch, cell_params, norm_by_size)
        embedding = torch.cat([embedding, scaled_params[:, :9]], dim=1)
        return embedding

    def rescale_proxy_discrim_cell_params(self, mol_batch, cell_params, norm_by_size=True):
        # rescale to nicer normalized basis
        if norm_by_size:
            normed_params = renormalize_generated_cell_params(
                cell_params,
                mol_batch,
                self.crystal_builder.asym_unit_dict
            )
        else:
            normed_params = cell_params.clone()

        # subtract means
        lattice_means = torch.tensor([
            1, 1, 1,
            torch.pi / 2, torch.pi / 2, torch.pi / 2,
            0.5, 0.5, 0.5,
            torch.pi / 4, 0, torch.pi / 2
        ], device=self.device, dtype=torch.float32)
        lattice_stds = torch.tensor([
            .35, .35, .35,
            .45, .45, .45,
            0.25, 0.25, 0.25,
            0.33, torch.pi / 2, torch.pi / 2
        ], device=self.device, dtype=torch.float32)
        scaled_params = (normed_params - lattice_means[None, :]) / lattice_stds[None, :]
        return scaled_params

    ''' test crystal rebuilding with different rescalings
    
from mxtaltools.dataset_management.dataset_generation.generate_dataset_from_smiles import test_crystal_rebuild_from_embedding

opt_vdw_pot = mol_batch.vdw_pot[None, ...].cpu()
opt_vdw_loss = mol_batch.vdw_loss[None, ...].cpu()
opt_aunits = mol_batch.pos[None,...].cpu()
cell_params=cell_params.cpu()
opt_cell_params = cell_params[None, ...]

normed_params = renormalize_generated_cell_params(
    cell_params.cuda(),
    mol_batch.cuda(),
    self.crystal_builder.asym_unit_dict
).cpu()

r_pot, r_loss, r_au = test_crystal_rebuild_from_embedding(
    mol_batch.clone().cpu(),
    opt_vdw_pot,
    opt_vdw_loss,
    opt_aunits,
    normed_params[None,...],
    denorm=True,
    destd=False,
    renorm=False,
    restd=False,
    make_figs=False,
)

# subtract means
lattice_means = torch.tensor([
    1, 1, 1,
    torch.pi / 2, torch.pi / 2, torch.pi / 2,
    0.5, 0.5, 0.5,
    torch.pi / 4, 0, torch.pi / 2
], device='cpu', dtype=torch.float32)
lattice_stds = torch.tensor([
    .35, .35, .35,
    .45, .45, .45,
    0.25, 0.25, 0.25,
    0.33, torch.pi / 2, torch.pi / 2
], device='cpu', dtype=torch.float32)
scaled_params = (normed_params - lattice_means[None, :]) / lattice_stds[None, :]

r_pot, r_loss, r_au = test_crystal_rebuild_from_embedding(
    mol_batch.clone().cpu(),
    opt_vdw_pot,
    opt_vdw_loss,
    opt_aunits,
    scaled_params[None,...],
    denorm=True,
    destd=True,
    renorm=False,
    restd=False,
    make_figs=False,
)

    '''

    def get_mol_embedding_for_proxy(self, mol_batch):
        centered_aunit_coordinates = torch.cat(
            [mol_batch.pos[mol_batch.batch == ind] - mol_batch.pos[mol_batch.batch == ind].mean(0) for ind in
             range(mol_batch.num_graphs)
             ])
        mol_batch.pos = centered_aunit_coordinates
        if self.config.proxy_discriminator.embedding_type == 'autoencoder':
            v_embedding = self.models_dict['autoencoder'].encode(mol_batch.clone())
            s_embedding = self.models_dict['autoencoder'].scalarizer(v_embedding)
        elif self.config.proxy_discriminator.embedding_type == 'principal_axes':
            v_embedding_i, s_embedding_i, _ = list_molecule_principal_axes_torch(
                [mol_batch.pos[mol_batch.batch == ind] for ind in range(mol_batch.num_graphs)])

            if not hasattr(self, 'ipm_means'):
                self.ipm_means = s_embedding_i.mean(0)

            s_embedding = s_embedding_i / self.ipm_means[None, :]
            v_embedding = v_embedding_i.permute(0, 2, 1)

        elif self.config.proxy_discriminator.embedding_type == 'principal_moments':
            Ip, s_embedding_i, _ = list_molecule_principal_axes_torch(
                [mol_batch.pos[mol_batch.batch == ind] for ind in range(mol_batch.num_graphs)])
            v_embedding = torch.zeros_like(Ip)

            if not hasattr(self, 'ipm_means'):
                self.ipm_means = s_embedding_i.mean(0)

            s_embedding = s_embedding_i / self.ipm_means[None, :]

        elif self.config.proxy_discriminator.embedding_type == 'mol_volume':
            if not hasattr(self, 'mol_volume_mean'):
                self.mol_volume_mean = mol_batch.mol_volume.mean()

            s_embedding = mol_batch.mol_volume[:, None].repeat(1, 3) / self.mol_volume_mean
            v_embedding = torch.zeros((mol_batch.num_graphs, 3, 3), dtype=torch.float32, device=self.device)
        elif self.config.proxy_discriminator.embedding_type is None:

            s_embedding = torch.zeros_like(mol_batch.mol_volume[:, None].repeat(1, 3))
            v_embedding = torch.zeros((mol_batch.num_graphs, 3, 3), dtype=torch.float32, device=self.device)
        else:
            assert False, f"{self.config.proxy_discriminator.embedding_type} is not an implemented proxy discriminator embedding"

        return torch.cat([s_embedding, v_embedding.flatten(-2)], dim=-1)

    def anneal_prior_loss(self, good_inds):
        """
        1. dynamically soften the packing loss when the model is doing well
        2. also increase the range of variability when the model is doing well
        3. if the model is not doing well, harden the loss
        """
        prior_loss = np.mean(np.array(self.logger.train_stats['generator_prior_loss'])[good_inds[-1000:]])

        if prior_loss.mean() < self.config.generator.prior_coefficient_threshold:
            if self.prior_loss_coefficient > 1:
                self.prior_loss_coefficient *= 0.99
            if self.prior_variation_scale < self.config.generator.variation_scale:
                self.prior_variation_scale += 0.01
        if (prior_loss.mean() > self.config.generator.prior_coefficient_threshold) and (
                self.prior_loss_coefficient < 10):
            self.prior_loss_coefficient *= 1.01
        self.logger.prior_loss_coefficient = self.prior_loss_coefficient
        self.logger.prior_variation_scale = self.prior_variation_scale

    def anneal_vdw_turnover(self, good_inds):
        """
        1. harden the potential when the model is doing well on the prior
        2. also boost the repulsion when the model is doing well on the prior AND at vdW business
        3. if doing poorly, soften everything
        """
        prior_loss = np.mean(np.array(self.logger.train_stats['generator_prior_loss'])[good_inds[-1000:]])
        vdw_loss = np.mean(np.array(self.logger.train_stats['generator_per_mol_raw_vdw_loss'])[good_inds[-1000:]])

        if prior_loss.mean() < self.config.generator.prior_coefficient_threshold:
            if self.vdw_loss_coefficient < self.config.generator.vdw_loss_coefficient:
                self.vdw_loss_coefficient += 0.01
            if (vdw_loss.mean() < 0) and (self.vdw_turnover_potential < self.config.generator.vdw_turnover_potential):
                self.vdw_turnover_potential += 0.01
        else:
            if self.vdw_loss_coefficient > 0.01:
                self.vdw_loss_coefficient -= 0.005
            if self.vdw_turnover_potential > 0.01:
                self.vdw_turnover_potential -= 0.005

        self.logger.vdw_turnover_potential = self.vdw_turnover_potential
        self.logger.vdw_loss_coefficient = self.vdw_loss_coefficient

    def what_generators_to_use(self, override_randn, override_distorted, override_adversarial):
        """
        pick what generator to use on a given step
        """  # todo adjust/delete generator option
        n_generators = sum((self.config.discriminator.train_on_randn or override_randn,
                            self.config.discriminator.train_on_distorted or override_distorted,
                            self.config.discriminator.train_adversarially or override_adversarial))

        gen_randint = np.random.randint(0, n_generators, 1)

        generator_ind_list = []
        if self.config.discriminator.train_adversarially or override_adversarial:
            generator_ind_list.append(1)
        if self.config.discriminator.train_on_randn or override_randn:
            generator_ind_list.append(2)
        if self.config.discriminator.train_on_distorted or override_distorted:
            generator_ind_list.append(3)

        generator_ind = generator_ind_list[
            int(gen_randint)]  # randomly select which generator to use from the available set

        return n_generators, generator_ind

    def generate_discriminator_negatives(self, real_data, train_adversarial=False,
                                         train_on_randn=False,
                                         train_on_distorted=False,
                                         orientation='random'):
        """
        use one of the available cell generation tools to sample cell parameters, to be fed to the discriminator
        """
        n_generators, generator_ind = self.what_generators_to_use(train_on_randn,
                                                                  train_on_distorted,
                                                                  train_adversarial)

        if False:  # TODO this is deprecated (self.config.discriminator.train_adversarially or override_adversarial) and (generator_ind == 1):
            negative_type = 'generator'
            with torch.no_grad():
                generated_samples, _, generator_data = self.old_get_generator_samples(real_data,
                                                                                      alignment_override=orientation)

            stats = {'generator_sample_source': np.zeros(len(generated_samples))}

        elif train_on_randn and (generator_ind == 2):
            generator_data = set_molecule_alignment(real_data.clone(), mode=orientation)
            negative_type = 'randn'
            _, generated_samples = self.sample_from_prior(real_data)
            generated_samples[:, 9:] = rotvec2sph(generated_samples[:, 9:])
            stats = {'generator_sample_source': np.ones(len(generated_samples))}

        elif train_on_distorted and (generator_ind == 3):
            # will be standardized anyway in cell builder
            generator_data = set_molecule_alignment(real_data.clone(), mode='as is')
            negative_type = 'distorted'

            generated_samples, distortion = self.make_distorted_samples(real_data)

            stats = {'generator_sample_source': 2 * np.ones(len(generated_samples)),
                     'distortion_level': torch.linalg.norm(distortion, axis=-1).detach()}

        else:
            print("No Generators set to make discriminator negatives!")
            assert False

        generator_data.cell_params = generated_samples

        return generated_samples.float().detach(), negative_type, generator_data, stats

    def make_distorted_samples(self, real_data, distortion_override=None):
        """
        given some cell params
        standardize them
        add noise in the standarized basis
        destandardize
        make sure samples are appropriately cleaned
        """
        real_cell_params = torch.cat([real_data.cell_reduced_lengths, real_data.cell_angles, real_data.pose_params0],
                                     dim=1)
        generated_samples_std = (real_cell_params - self.lattice_means) / self.lattice_stds

        if distortion_override is not None:
            distortion = torch.randn_like(generated_samples_std) * distortion_override
        else:
            # distortion types
            # pick n=[1,4] of the 4 cell param types and proportionally noise them
            distortion_mask = torch.randint(0, 2, size=(generated_samples_std.shape[0], 4),
                                            device=generated_samples_std.device, dtype=torch.long)
            distortion_mask[distortion_mask.sum(1) == 0] = 1  # any zero entries go to all

            distortion_mask = distortion_mask.repeat_interleave(3, dim=1)

            if self.config.discriminator.distortion_magnitude == -1:
                distortion_magnitude = torch.logspace(-1.5, 0.5, len(generated_samples_std)).to(
                    generated_samples_std.device)[:, None]  # wider range
            else:
                distortion_magnitude = self.config.discriminator.distortion_magnitude

            distortion = torch.randn_like(generated_samples_std) * distortion_magnitude * distortion_mask

        # add jitter and return in standardized basis
        distorted_samples_std = (generated_samples_std + distortion).to(self.device)

        distorted_samples_clean = clean_cell_params(
            distorted_samples_std, real_data.sg_ind,
            self.lattice_means, self.lattice_stds,
            self.sym_info, self.crystal_builder.asym_unit_dict,
            rescale_asymmetric_unit=False, destandardize=True, mode='hard')

        distorted_samples_clean[:, :3] *= torch.pow(real_data.mol_volume * real_data.sym_mult, 1 / 3)[:,
                                          None]  # denorm cell lengths

        return distorted_samples_clean, distortion

    def increment_batch_size(self, train_loader, test_loader, extra_test_loader):
        self.times['batch_resizing_start'] = time()
        if self.config.grow_batch_size:
            if (train_loader.batch_size < len(train_loader.dataset)) and (
                    train_loader.batch_size < self.config.max_batch_size):  # if the batch is smaller than the dataset
                increment = max(4,
                                int(train_loader.batch_size * self.config.batch_growth_increment))  # increment batch size
                train_loader, test_loader = (
                    update_dataloader_batch_size(train_loader, train_loader.batch_size + increment),
                    update_dataloader_batch_size(test_loader, test_loader.batch_size + increment))

                if extra_test_loader is not None:
                    extra_test_loader = update_dataloader_batch_size(extra_test_loader,
                                                                     extra_test_loader.batch_size + increment)
                print(f'Batch size incremented to {train_loader.batch_size}')
        self.config.current_batch_size = train_loader.batch_size
        self.times['batch_resizing_end'] = time()
        return train_loader, test_loader, extra_test_loader

    def model_checkpointing(self, epoch):
        self.times['checkpointing_start'] = time()
        loss_type_check = self.config.checkpointing_loss_type
        for model_name in self.model_names:
            if self.train_models_dict[model_name]:
                loss_record = self.logger.loss_record[model_name][f'mean_{loss_type_check}']
                past_mean_losses = [np.mean(record) for record in loss_record]  # load all prior epoch losses
                current_loss = np.average(self.logger.current_losses[model_name][f'mean_{loss_type_check}'])

                if current_loss <= np.amin(past_mean_losses):  # if current mean loss beats all prior epochs
                    print(f"Saving {model_name} checkpoint")
                    self.logger.save_stats_dict(prefix=f'best_{model_name}_')
                    save_checkpoint(epoch,
                                    self.models_dict[model_name],
                                    self.optimizers_dict[model_name],
                                    self.config.__dict__[model_name].__dict__,
                                    self.config.checkpoint_dir_path + f'best_{model_name}' + self.run_identifier,
                                    self.dataDims)
        self.times['checkpointing_end'] = time()

    def update_lr(self, override_lr: dict = None):
        for model_name in self.model_names:
            if self.config.__dict__[model_name].optimizer is not None:
                if override_lr is not None:
                    override = override_lr[model_name]
                else:
                    override = None
                self.optimizers_dict[model_name], learning_rate = set_lr(
                    self.schedulers_dict[model_name],
                    self.optimizers_dict[model_name],
                    self.config.__dict__[model_name].optimizer,
                    self.logger.current_losses[model_name]['mean_train'],
                    self.hit_max_lr_dict[model_name],
                    override)

                if learning_rate >= self.config.__dict__[model_name].optimizer.max_lr:
                    self.hit_max_lr_dict[model_name] = True

                self.logger.learning_rates[model_name] = learning_rate

    def reload_best_test_checkpoint(self, epoch):
        # reload best test for any existing model
        if epoch != 0:  # if we have trained at all, reload the best model
            best_checkpoints = {}
            for model_name in self.train_models_dict.keys():
                if self.train_models_dict[model_name]:
                    checkpoint_path = self.config.checkpoint_dir_path + f'best_{model_name}' + self.run_identifier
                    if os.path.exists(checkpoint_path):
                        best_checkpoints[model_name] = checkpoint_path
                    else:
                        assert False, f"No checkpoint to reload for {model_name}"

            for model_name, model_path in best_checkpoints.items():
                self.models_dict[model_name], self.optimizers_dict[model_name] = reload_model(
                    self.models_dict[model_name], self.device, self.optimizers_dict[model_name],
                    model_path
                )

                if not torch.stack([torch.isfinite(p).any() for p in self.models_dict[model_name].parameters()]).all():
                    assert False, "Reloaded model contains NaN weights! Something really wrong happened"

    def reload_model_checkpoint_configs(self):
        for model_name, model_path in self.config.model_paths.__dict__.items():
            if model_path is not None:
                checkpoint = torch.load(model_path, map_location=self.device)
                model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
                if self.config.__dict__[model_name].optimizer is not None:
                    if hasattr(self.config.__dict__[model_name].optimizer, 'overwrite_on_reload'):
                        if self.config.__dict__[model_name].optimizer.overwrite_on_reload:
                            self.config.__dict__[model_name].optimizer = model_config.optimizer
                    else:
                        pass
                else:
                    self.config.__dict__[model_name].optimizer = model_config.optimizer
                self.config.__dict__[model_name].model = model_config.model
                print(f"Reloading {model_name} {model_path}")

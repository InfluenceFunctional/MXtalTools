import gc
import os
from argparse import Namespace
from distutils.dir_util import copy_tree
from shutil import copy
from time import time
from typing import Tuple
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

from mxtaltools.common.geometry_calculations import batch_molecule_principal_axes_torch
from mxtaltools.common.utils import softmax_np, init_sym_info, compute_rdf_distance, make_sequential_directory, \
    flatten_wandb_params
from mxtaltools.constants.asymmetric_units import asym_unit_dict
from mxtaltools.constants.atom_properties import VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD
from mxtaltools.crystal_building.builder import SupercellBuilder
from mxtaltools.crystal_building.utils import (set_molecule_alignment, descale_asymmetric_unit)
from mxtaltools.crystal_building.utils import update_crystal_symmetry_elements
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.dataset_management.data_manager import DataManager
from mxtaltools.dataset_management.dataloader_utils import get_dataloaders, update_dataloader_batch_size
from mxtaltools.models.functions.crystal_rdf import new_crystal_rdf
from mxtaltools.models.functions.vdw_overlap import vdw_overlap, vdw_analysis
from mxtaltools.models.task_models.generator_models import IndependentGaussianGenerator, GeneratorPrior
from mxtaltools.models.utils import (reload_model, init_scheduler, softmax_and_score, save_checkpoint, set_lr,
                                     cell_vol_torch, init_optimizer, get_regression_loss,
                                     slash_batch, compute_type_evaluation_overlap,
                                     compute_coord_evaluation_overlap,
                                     compute_full_evaluation_overlap, compute_reduced_volume_fraction,
                                     dict_of_tensors_to_cpu_numpy,
                                     test_decoder_equivariance, test_encoder_equivariance, collate_decoded_data,
                                     ae_reconstruction_loss, clean_cell_params, get_intermolecular_dists_dict)
from mxtaltools.common.training_utils import instantiate_models
from mxtaltools.models.utils import (weight_reset, get_n_config)
from mxtaltools.reporting.ae_reporting import scaffolded_decoder_clustering
from mxtaltools.reporting.logger import Logger


# noinspection PyAttributeOutsideInit


class Modeller:
    """
    Main class brining together
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

        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        self.load_physical_constants()
        self.config = flatten_wandb_params(self.config)

        self.supercell_builder = SupercellBuilder(device=self.config.device,
                                                  rotation_basis='spherical' if self.config.mode != 'generator' else 'cartesian')
        self.collater = Collater(None, None)

        self.train_models_dict = {
            'discriminator': False,
            'generator': False,
            'regressor': False,
            'autoencoder': False,
            'embedding_regressor': False,
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
        self.copy_source_to_workdir()

    def copy_source_to_workdir(self):
        os.mkdir(self.working_directory + '/source')
        yaml_path = self.config.paths.yaml_path
        copy_tree("mxtaltools/common", self.working_directory + "/source/common")
        copy_tree("mxtaltools/crystal_building", self.working_directory + "/source/crystal_building")
        copy_tree("mxtaltools/dataset_management", self.working_directory + "/source/dataset_management")
        copy_tree("mxtaltools/models", self.working_directory + "/source/models")
        copy_tree("mxtaltools/reporting", self.working_directory + "/source/reporting")
        copy("mxtaltools/modeller.py", self.working_directory + "/source")
        copy("main.py", self.working_directory + "/source")
        np.save(self.working_directory + '/run_config', self.config)
        os.chdir(self.working_directory)  # move to working dir
        copy(yaml_path, os.getcwd())  # copy full config for reference
        print('Starting fresh run ' + self.working_directory)

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
                                              self.sym_info)
        self.init_optimizers()
        self.reload_models()
        self.init_schedulers()
        self.num_params_dict = self.get_model_sizes()
        self.times['init_models_end'] = time()

    def get_model_sizes(self):
        num_params_dict = {model_name + "_num_params": get_n_config(model) for model_name, model in
                           self.models_dict.items()}
        [print(
            f'{model_name} {num_params_dict[model_name] / 1e6:.3f} million or {int(num_params_dict[model_name])} parameters')
            for model_name in num_params_dict.keys()]
        return num_params_dict

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
                                     override_shuffle=False,
                                     override_batch_size=None):
        """
        use data manager to load and filter dataset
        use dataset builder to generate crystaldata objects
        return dataloaders
        """

        nonzero_positional_noise = sum(list(self.config.positional_noise.__dict__.values()))
        if self.config.mode == 'polymorph_classification':
            conv_cutoff = self.config.polymorph_classifier.model.graph.cutoff
        elif self.config.mode == 'regression':
            conv_cutoff = self.config.regressor.model.graph.cutoff
        elif self.config.mode == 'autoencoder':
            conv_cutoff = self.config.autoencoder.model.encoder.graph.cutoff
        elif self.config.mode in ['gan', 'generator', 'discriminator']:
            conv_cutoff = self.config.discriminator.model.graph.cutoff
        else:
            assert False, "Missing convolutional cutoff information"

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
            do_shuffle=override_shuffle,
            precompute_edges=not nonzero_positional_noise and (
                    self.config.mode not in ['gan', 'discriminator', 'generator']),
        )
        self.dataDims = data_manager.dataDims
        self.lattice_means = torch.tensor(self.dataDims['lattice_means'], device=self.device)
        self.lattice_stds = torch.tensor(self.dataDims['lattice_stds'], device=self.device)
        self.new_lattice_means = torch.tensor([1.2740, 1.4319, 1.7752,
                                               1.5619, 1.5691, 1.5509,
                                               0.5, 0.5, 0.5,
                                               torch.pi / 2, 0, torch.pi],
                                              dtype=torch.float32, device=self.device)
        self.new_lattice_stds = torch.tensor([0.5163, 0.5930, 0.6284,
                                              0.2363, 0.2046, 0.2624,
                                              0.2875, 0.2875, 0.2875,
                                              2.0942, 2.0942, 1.3804],
                                             dtype=torch.float32, device=self.device)
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

    def prep_dataloaders(self, dataset_builder, extra_dataset_builder=None, test_fraction=0.2,
                         override_batch_size: int = None,
                         override_shuffle=None):
        """
        get training, test, ane optionall extra validation dataloaders
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

        if self.config.dataset.on_disk_data_dir is not None:
            print(f"Loading on-disk dataset {self.config.dataset.on_disk_data_dir}")
            from mxtaltools.dataset_management.lmdb_dataset import lmdbDataset
            from torch_geometric.data import DataLoader
            train_dataset = lmdbDataset(self.config.dataset_path + self.config.dataset.on_disk_data_dir)
            num_workers = 2  #min(os.cpu_count(), 16)  # min(os.cpu_count(), 8)
            print(f'{num_workers} workers set for dataloaders')
            train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=shuffle,
                                      pin_memory=True, drop_last=False,
                                      num_workers=0,  #num_workers if self.config.machine == 'cluster' else 0,
                                      persistent_workers=False,  #True if self.config.machine == 'cluster' else False
                                      )
            del train_dataset
            #
            # test_dataset = lmdbDataset(
            #     self.config.dataset_path + self.config.dataset.on_disk_data_dir.replace('train', 'test'))
            # test_loader = DataLoader(test_dataset, batch_size=loader_batch_size, shuffle=shuffle,
            #                          pin_memory=True, drop_last=False,
            #                          num_workers=0,  #num_workers if self.config.machine == 'cluster' else 0,
            #                          persistent_workers=False,
            #                          #True, #True if self.config.machine == 'cluster' else False,
            #                          )

            #del test_dataset

            # test dataset is pre-generated
            test_loader, _ = get_dataloaders(dataset_builder,
                                             machine=self.config.machine,
                                             batch_size=loader_batch_size,
                                             test_fraction=0,
                                             shuffle=shuffle)
        else:
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

            # analysis & visualization
            # autoencoder_embedding_map(self.logger.test_stats)
            aa = 1

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
        _, data_loader, extra_test_loader = self.load_dataset_and_dataloaders(override_test_fraction=0.2)
        self.initialize_models_optimizers_schedulers()

        self.config.autoencoder_sigma = self.config.autoencoder.init_sigma
        self.config.autoencoder.molecule_radius_normalization = self.dataDims['standardization_dict']['radius']['max']

        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         settings=wandb.Settings(code_dir="."))):
            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(self.num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(self.num_params_dict.values()))),
                       "Initial Batch Size": self.config.current_batch_size})

            self.models_dict['autoencoder'].eval()
            self.models_dict['embedding_regressor'].eval()
            self.epoch_type = 'test'

            with torch.no_grad():
                for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
                    self.embedding_regression_step(data, update_weights=False)

            # post epoch processing
            self.logger.concatenate_stats_dict(self.epoch_type)

            # save results
            np.save(self.config.checkpoint_dir_path +
                    self.config.model_paths.EmbeddingRegressor.split('embedding_regressor')[-1],
                    {'train_stats': self.logger.train_stats, 'test_stats': self.logger.test_stats})

    def ae_embedding_step(self, data):
        data = data.to(self.device)
        data, input_data = self.preprocess_ae_inputs(data, no_noise=True)
        if not self.models_dict['autoencoder'].protons_in_input and not self.models_dict[
            'autoencoder'].inferring_protons:
            data = input_data.clone()  # deprotonate the reference if we are not inferring protons

        decoding, encoding = self.models_dict['autoencoder'](input_data.clone(), return_encoding=True)
        scalar_encoding = self.models_dict['autoencoder'].scalarizer(encoding)

        self.ae_evaluation_sample_analysis(data, decoding, encoding, scalar_encoding)

    def ae_evaluation_sample_analysis(self, data, decoding, encoding, scalar_encoding):
        """

        Parameters
        ----------
        data
        decoding
        encoding
        scalar_encoding

        Returns
        -------

        """
        'standard analysis'
        autoencoder_losses, stats, decoded_data = self.compute_autoencoder_loss(decoding, data.clone())

        'extra analysis'
        data.x = self.models_dict['autoencoder'].atom_embedding_vector[data.x]
        nodewise_weights_tensor = decoded_data.aux_ind
        true_nodes = F.one_hot(data.x[:, 0].long(), num_classes=self.dataDims['num_atom_types']).float()

        full_overlap, self_overlap = compute_full_evaluation_overlap(data, decoded_data, nodewise_weights_tensor,
                                                                     true_nodes,
                                                                     sigma=self.config.autoencoder.evaluation_sigma,
                                                                     distance_scaling=self.config.autoencoder.type_distance_scaling
                                                                     )
        coord_overlap, self_coord_overlap = compute_coord_evaluation_overlap(self.config, data, decoded_data,
                                                                             nodewise_weights_tensor, true_nodes)
        self_type_overlap, type_overlap = compute_type_evaluation_overlap(self.config, data,
                                                                          self.dataDims['num_atom_types'],
                                                                          decoded_data, nodewise_weights_tensor,
                                                                          true_nodes)

        Ip, Ipm, I = batch_molecule_principal_axes_torch(
            [data.pos[data.batch == ind] for ind in range(data.num_graphs)])

        scaffold_rmsds, scaffold_max_dists, scaffold_matched = [], [], []
        #glom_rmsds, glom_max_dists = [], []
        for ind in range(data.num_graphs):  # somewhat slow
            rmsd, max_dist, weight_mean, match_successful = scaffolded_decoder_clustering(ind, data, decoded_data,
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
                        data.radius.cpu().detach().numpy(),
                        scalar_encoding.cpu().detach().numpy(),
                        scatter(full_overlap / self_overlap, data.batch, reduce='mean').cpu().detach().numpy(),
                        scatter(coord_overlap / self_coord_overlap, data.batch, reduce='mean').cpu().detach().numpy(),
                        scatter(self_type_overlap / type_overlap, data.batch, reduce='mean').cpu().detach().numpy(),
                        Ip.cpu().detach().numpy(),
                        Ipm.cpu().detach().numpy(),
                        np.asarray(scaffold_rmsds),
                        np.asarray(scaffold_max_dists),
                        np.asarray(scaffold_matched),
                        # np.asarray(glom_rmsds),
                        # np.asarray(glom_max_dists),
                        data.smiles
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
                         settings=wandb.Settings(code_dir="."))):
            self.process_sweep_config()
            self.source_directory = os.getcwd()
            self.prep_new_working_directory()
            self.get_training_mode()
            train_loader, test_loader, extra_test_loader = self.load_dataset_and_dataloaders()
            self.initialize_models_optimizers_schedulers()
            converged, epoch, prev_epoch_failed = self.init_logging()

            #with torch.autograd.set_detect_anomaly(False):
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
                    self.post_epoch_logging_analysis(test_loader, epoch)

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
                    elif "Mean loss is NaN/Inf" == str(e):
                        self.handle_nan(e, epoch)
                    else:
                        raise e  # will simply raise error if other or if training on CPU

                self.times['full_epoch_end'] = time()
                self.logger.log_times(self.times)
                self.times = {}
                epoch += 1

            self.logger.evaluation_analysis(test_loader, self.config.mode)

    def handle_nan(self, e, epoch):
        print(e)
        print("Reloading prior best checkpoint and restarting training at low LR")
        self.reload_best_test_checkpoint(epoch)
        self.update_lr(update_lr_ratio=0.75)  # reduce LR and try again
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

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
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

            self.post_epoch_logging_analysis(data_loader, 0)

            self.logger.evaluation_analysis(data_loader, self.config.mode)

    def train_test_validate(self, epoch, extra_test_loader, steps_override, test_loader, train_loader):
        self.run_epoch(epoch_type='train',
                       data_loader=train_loader,
                       update_weights=True,
                       iteration_override=steps_override)
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

    def post_epoch_logging_analysis(self, test_loader, epoch):
        """check convergence status and record metrics & analysis"""
        self.times['reporting_start'] = time()
        self.logger.numpyize_current_losses()
        self.logger.update_loss_record()
        self.logger.log_training_metrics()
        self.logger.log_detailed_analysis(test_loader)
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
                 self.config.discriminator.train_on_randn, self.config.generator.train_adversarially)))
                             or (self.config.model_paths.discriminator is not None),
            'generator': (self.config.mode in ['gan', 'generator', 'discriminator']) and
                         any((self.config.generator.train_vdw,
                              self.config.generator.train_adversarially,
                              self.config.generator.train_h_bond)),
            'regressor': self.config.mode == 'regression',
            'autoencoder': self.config.mode == 'autoencoder',
            'embedding_regressor': self.config.mode == 'embedding_regression',
            'polymorph_classifier': self.config.mode == 'polymorph_classification',
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
        _, data = self.preprocess_ae_inputs(data, no_noise=True, orientation_override='random')
        v_embedding = self.models_dict['autoencoder'].encode(data)
        s_embedding = self.models_dict['autoencoder'].scalarizer(v_embedding)

        predictions = self.models_dict['embedding_regressor'](s_embedding, v_embedding)[:, 0]
        losses = F.smooth_l1_loss(predictions, data.y, reduction='none')
        predictions = predictions * self.dataDims['target_std'] + self.dataDims['target_mean']
        targets = data.y * self.dataDims['target_std'] + self.dataDims['target_mean']

        if self.config.embedding_regressor.prediction_type == 'vector':  # this is quite fast
            assert False, "Vector predictions are not implemented"

        regression_loss = losses.mean()
        if update_weights:
            self.optimizers_dict['embedding_regressor'].zero_grad(set_to_none=True)
            regression_loss.backward()  # back-propagation
            self.optimizers_dict['embedding_regressor'].step()  # update parameters
        '''log losses and other tracking values'''
        self.logger.update_current_losses('embedding_regressor', self.epoch_type,
                                          regression_loss.cpu().detach().numpy(),
                                          losses.cpu().detach().numpy())
        stats_values = [predictions.cpu().detach().numpy(), targets.cpu().detach().numpy()]

        self.logger.update_stats_dict(self.epoch_type,
                                      ['regressor_prediction', 'regressor_target'],
                                      stats_values,
                                      mode='extend')

    def ae_step(self, input_data, data, update_weights, step, last_step=False):
        if (not self.models_dict['autoencoder'].protons_in_input and
                not self.models_dict['autoencoder'].inferring_protons):
            data = input_data.detach().clone()

        if step % self.config.logger.stats_reporting_frequency == 0:
            skip_stats = False
        elif last_step:
            skip_stats = False
        else:
            skip_stats = True

        decoding = self.models_dict['autoencoder'](input_data.clone(), return_latent=False)
        losses, stats, decoded_data = self.compute_autoencoder_loss(decoding, data.clone(), skip_stats=skip_stats)

        mean_loss = losses.mean()
        if torch.sum(torch.logical_not(torch.isfinite(mean_loss))) > 0:
            raise ValueError("Mean loss is NaN/Inf")

        if update_weights:
            self.optimizers_dict['autoencoder'].zero_grad(set_to_none=True)
            mean_loss.backward()  # back-propagation
            torch.nn.utils.clip_grad_norm_(self.models_dict['autoencoder'].parameters(),
                                           self.config.gradient_norm_clip)  # gradient clipping by norm
            self.optimizers_dict['autoencoder'].step()  # update parameters

        if not skip_stats:
            self.ae_stats_and_reporting(data, decoded_data, last_step, stats, step)

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
                               step: int):
        # if self.logger.epoch % self.config.logger.sample_reporting_frequency == 0:
        #     if step % 10 == 0:
        #         stats['encoding'] = encoding.detach()

        if step == 0 or last_step:
            self.detailed_autoencoder_step_analysis(data, decoded_data, stats)

        dict_of_tensors_to_cpu_numpy(stats)

        self.logger.update_stats_dict(self.epoch_type,
                                      stats.keys(),
                                      stats.values(),
                                      mode='append')

    def detailed_autoencoder_step_analysis(self, data, decoded_data, stats):
        # equivariance checks
        encoder_equivariance_loss, decoder_equivariance_loss = self.ae_equivariance_loss(data.clone())
        stats['encoder_equivariance_loss'] = encoder_equivariance_loss.mean().detach()
        stats['decoder_equivariance_loss'] = decoder_equivariance_loss.mean().detach()

        # do evaluation on current sample and save this as our loss for tracking purposes
        nodewise_weights_tensor = decoded_data.aux_ind
        true_nodes = F.one_hot(self.models_dict['autoencoder'].atom_embedding_vector[data.x.long()],
                               num_classes=self.dataDims['num_atom_types']).float()
        full_overlap, self_overlap = compute_full_evaluation_overlap(data, decoded_data, nodewise_weights_tensor,
                                                                     true_nodes,
                                                                     sigma=self.config.autoencoder.evaluation_sigma,
                                                                     distance_scaling=self.config.autoencoder.type_distance_scaling
                                                                     )
        '''log losses and other tracking values'''
        # for the purpose of convergence, we track the evaluation overlap rather than the loss, which is sigma-dependent
        # it's also expensive to compute so do it rarely
        overlap = (full_overlap / self_overlap).detach()
        tracking_loss = torch.abs(1 - overlap)
        stats['evaluation_overlap'] = scatter(overlap, data.batch, reduce='mean').detach()
        self.logger.update_current_losses('autoencoder', self.epoch_type,
                                          tracking_loss.mean().cpu().detach().numpy(),
                                          tracking_loss.cpu().detach().numpy())

        self.logger.update_stats_dict(self.epoch_type,
                                      ['sample', 'decoded_sample'],
                                      [data.cpu().detach(), decoded_data.cpu().detach()
                                       ], mode='append')

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

        return encoder_equivariance_loss, decoder_equivariance_loss

    def ae_epoch(self,
                 data_loader,
                 update_weights: bool,
                 iteration_override: bool = None):
        if update_weights:
            self.models_dict['autoencoder'].train(True)
        else:
            self.models_dict['autoencoder'].eval()

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            data = data.to(self.device)
            data.x = data.x.flatten()

            data, input_data = self.preprocess_ae_inputs(data, no_noise=self.epoch_type == 'test')
            self.ae_step(input_data, data, update_weights, step=i,
                         last_step=(i == len(data_loader) - 1) or (i == iteration_override))

            if iteration_override is not None:
                if i >= iteration_override:
                    break

        self.logger.concatenate_stats_dict(self.epoch_type)
        self.ae_annealing()

    def ae_annealing(self):
        # if we have learned the existing distribution
        if self.logger.train_stats['reconstruction_loss'][-100:].mean() < self.config.autoencoder.sigma_threshold:
            # and we more self-overlap than desired
            if self.epoch_type == 'test':  # the overlap we ultimately care about is in the Test
                if np.abs(1 - self.logger.test_stats['mean_self_overlap'][
                              -100:]).mean() > self.config.autoencoder.overlap_eps.test:
                    # tighten the target distribution
                    self.config.autoencoder_sigma *= self.config.autoencoder.sigma_lambda

        # if we have way too much overlap, just tighten right away
        if np.abs(1 - self.logger.train_stats['mean_self_overlap'][
                      -100:]).mean() > self.config.autoencoder.max_overlap_threshold:
            self.config.autoencoder_sigma *= self.config.autoencoder.sigma_lambda

    def preprocess_ae_inputs(self, data, no_noise=False, orientation_override=None, noise_override=None):
        # atomwise random noise
        if not no_noise:
            if noise_override is not None:
                data.pos += torch.randn_like(data.pos) * noise_override
            elif self.config.positional_noise.autoencoder > 0:
                data.pos += torch.randn_like(data.pos) * self.config.positional_noise.autoencoder

        # random global roto-inversion
        if orientation_override is not None:
            data = set_molecule_alignment(data, mode=orientation_override, right_handed=False, include_inversion=True)

        # optionally, deprotonate
        input_data = self.fix_autoencoder_protonation(data)

        # subtract mean OF THE INPUT from BOTH reference and input
        centroids = scatter(input_data.pos, input_data.batch, reduce='mean', dim=0)
        data.pos -= torch.repeat_interleave(centroids, data.num_atoms, dim=0, output_size=data.num_nodes)
        input_data.pos -= torch.repeat_interleave(centroids, input_data.num_atoms, dim=0,
                                                  output_size=input_data.num_nodes)

        return data, input_data

    def compute_autoencoder_loss(self,
                                 decoding: torch.Tensor,
                                 data: CrystalData,
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
        data : CrystalData
            Input data to be reconstructed
        skip_stats : bool
            Whether to skip saving summary statistics for this step

        Returns
        -------

        """
        # reduce to relevant atom types
        data.x = self.models_dict['autoencoder'].atom_embedding_vector[data.x].flatten()
        decoded_data, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
            collate_decoded_data(data,
                                 decoding,
                                 self.models_dict['autoencoder'].num_decoder_nodes,
                                 self.config.autoencoder.node_weight_temperature,
                                 self.device))

        (nodewise_reconstruction_loss,
         nodewise_type_loss,
         reconstruction_loss,
         self_likelihoods) = ae_reconstruction_loss(data,
                                                    decoded_data,
                                                    nodewise_weights,
                                                    self.dataDims['num_atom_types'],
                                                    self.config.autoencoder.type_distance_scaling,
                                                    self.config.autoencoder_sigma)

        matching_nodes_fraction = torch.sum(nodewise_reconstruction_loss < 0.01) / data.num_nodes  # within 1% matching

        # node radius constraining loss
        decoded_dists = torch.linalg.norm(decoded_data.pos, dim=1)
        constraining_loss = scatter(
            F.relu(
                decoded_dists -  #self.models_dict['autoencoder'].radial_normalization),
                torch.repeat_interleave(data.radius, self.models_dict['autoencoder'].num_decoder_nodes, dim=0)),
            decoded_data.batch, reduce='mean')

        # node weight constraining loss
        node_weight_constraining_loss = scatter(
            F.relu(-torch.log10(nodewise_weights_tensor / torch.amin(nodewise_graph_weights)) - 2),
            decoded_data.batch)  # don't let these get too small

        # sum losses
        losses = reconstruction_loss + constraining_loss + node_weight_constraining_loss

        if not skip_stats:
            stats = {'constraining_loss': constraining_loss.mean().detach(),
                     'reconstruction_loss': reconstruction_loss.mean().detach(),
                     'nodewise_type_loss': nodewise_type_loss.detach(),
                     'scaled_reconstruction_loss': (
                             reconstruction_loss.mean() * self.config.autoencoder_sigma).detach(),
                     'sigma': self.config.autoencoder_sigma,
                     'mean_self_overlap': scatter(self_likelihoods, data.batch, reduce='mean').mean().detach(),
                     'matching_nodes_fraction': matching_nodes_fraction.detach(),
                     'matching_nodes_loss': 1 - matching_nodes_fraction.detach(),
                     'node_weight_constraining_loss': node_weight_constraining_loss.mean().detach(),
                     }
        else:
            stats = {}

        return losses, stats, decoded_data

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

        if not hasattr(self, 'packing_loss_coefficient'):  # first GAN epoch
            self.init_gan_constants()

        if update_weights:
            self.models_dict['generator'].train(True)
            self.models_dict['discriminator'].train(True)
        else:
            self.models_dict['generator'].eval()
            self.models_dict['discriminator'].eval()

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 10), mininterval=30)):
            data = data.to(self.config.device)

            '''
            train discriminator
            '''
            skip_discriminator_step = self.decide_whether_to_skip_discriminator(i, self.logger.get_stat_dict(
                self.epoch_type))

            self.discriminator_step(data, i, update_weights, skip_step=skip_discriminator_step)

            '''
            record some stats
            '''
            self.logger.update_stats_dict(self.epoch_type, ['identifiers'], data.identifier, mode='extend')
            self.logger.update_stats_dict(self.epoch_type, ['smiles'], data.smiles, mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early

        self.logger.concatenate_stats_dict(self.epoch_type)

    def generator_epoch(self,
                        data_loader=None,
                        update_weights=True,
                        iteration_override=None):

        if not hasattr(self, 'packing_loss_coefficient'):  # first GAN epoch
            self.init_gan_constants()

        if update_weights:
            self.models_dict['generator'].train(True)
        else:
            self.models_dict['generator'].eval()

        self.models_dict['autoencoder'].eval()

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 10), mininterval=30)):
            data = data.to(self.config.device)
            '''
            train_generator
            '''
            self.generator_step(data, i, i == len(data_loader) - 1, update_weights)

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early

        self.logger.concatenate_stats_dict(self.epoch_type)

    def init_gan_constants(self):
        self.packing_loss_coefficient = self.config.generator.packing_loss_coefficient
        self.prior_loss_coefficient = self.config.generator.prior_loss_coefficient
        self.vdw_loss_coefficient = self.config.generator.vdw_loss_coefficient
        self.vdw_turnover_potential = self.config.generator.vdw_turnover_potential
        '''set space groups to be included and generated'''
        if self.config.generate_sgs == 'all':
            self.config.generate_sgs = [self.sym_info['space_groups'][int(key)] for key in asym_unit_dict.keys()]

        '''compute the ratios between the norms of n-dimensional gaussians (means of chi distribution)'''
        m1 = torch.sqrt(torch.ones(1) * 2) * torch.exp(torch.lgamma(torch.ones(1) * (12 + 1) / 2)) / torch.exp(
            torch.lgamma(torch.ones(1) * 12 / 2))
        self.chi_scaling_factors = torch.zeros(4, dtype=torch.float, device=self.device)
        for ind, ni in enumerate([3, 6, 9, 12]):
            m2 = torch.sqrt(torch.ones(1) * 2) * torch.exp(torch.lgamma(torch.ones(1) * (ni + 1) / 2)) / torch.exp(
                torch.lgamma(torch.ones(1) * ni / 2))
            self.chi_scaling_factors[ind] = m1 / m2

        if self.train_models_dict['discriminator']:
            self.init_gaussian_generator()

        if self.train_models_dict['generator']:
            self.init_generator_prior()

    def decide_whether_to_skip_discriminator(self, i, epoch_stats_dict):
        """
        hold discriminator training when it's beating the generator
        """

        skip_discriminator_step = False
        if (i == 0) and self.config.generator.train_adversarially:
            skip_discriminator_step = True  # do not train except by express permission of the below condition
        if i > 0 and self.config.discriminator.train_adversarially:  # must skip first step since there will be no fake score to compare against
            generator_inds = np.argwhere(np.asarray(epoch_stats_dict['generator_sample_source']) == 0)[:, 0]
            if len(generator_inds > 0):
                if self.config.generator.adversarial_loss_func == 'score':
                    avg_generator_score = np.stack(epoch_stats_dict['discriminator_fake_score'])[generator_inds].mean()
                    if avg_generator_score < 0:
                        skip_discriminator_step = True
                else:
                    avg_generator_score = softmax_np(
                        np.stack(epoch_stats_dict['discriminator_fake_score'])[generator_inds])[:, 1].mean()
                    if avg_generator_score < 0.5:
                        skip_discriminator_step = True
            else:
                skip_discriminator_step = True
        return skip_discriminator_step

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

    def generator_step(self, data, step, last_step, update_weights):
        """
        execute a complete training step for the generator
        get sample losses, do reporting, update gradients
        """

        if step % self.config.logger.stats_reporting_frequency == 0:
            skip_stats = False
        elif last_step:
            skip_stats = False
        else:
            skip_stats = True

        # target_rauv = self.get_generator_density_target(data)  # doesn't work for QM9 data
        # target_rauv = (torch.ones_like(target_rauv)
        #                + torch.randn_like(target_rauv) * self.config.generator.packing_target_noise)

        # larger means we allow more freedom in the search
        variation_factor = torch.rand(size=(data.num_graphs,), device=self.device
                                      ).abs() * self.config.generator.variation_scale

        generated_samples, prior, generator_data \
            = self.get_generator_samples(data, variation_factor)

        # denormalize the predicted cell lengths
        cell_lengths = data.radius[:, None] * torch.pow(generator_data.sym_mult, 1 / 3)[:, None] * generated_samples[:,
                                                                                                   :3]
        # rescale asymmetric units  # todo add assertions around these
        mol_positions = descale_asymmetric_unit(self.supercell_builder.asym_unit_dict,
                                                generated_samples[:, 6:9],
                                                generator_data.sg_ind)
        generated_samples_to_build = torch.cat(
            [cell_lengths, generated_samples[:, 3:6], mol_positions, generated_samples[:, 9:12]], dim=1)
        if torch.sum(torch.isnan(generated_samples)) > 0:
            self.vdw_turnover_potential *= 0.75  # soften vdW
            raise ValueError("Mean loss is NaN/Inf")
        supercell_data, generated_cell_volumes = (
            self.supercell_builder.build_zp1_supercells(
                molecule_data=generator_data,
                cell_parameters=generated_samples_to_build,
                supercell_size=self.config.supercell_size,
                graph_convolution_cutoff=self.config.discriminator.model.graph.cutoff,
                align_to_standardized_orientation=False,
                skip_refeaturization=True,
            ))

        generator_losses, losses_stats, supercell_data = self.get_generator_losses(
            data,
            generated_samples,
            supercell_data,
            generated_cell_volumes,
            prior,
            variation_factor,
            skip_stats,
        )
        generator_loss = generator_losses.mean()
        if torch.sum(torch.logical_not(torch.isfinite(generator_losses))) > 0:
            raise ValueError("Mean loss is NaN/Inf")

        if update_weights:
            self.optimizers_dict['generator'].zero_grad(set_to_none=True)  # reset gradients from previous passes
            generator_loss.backward()  # back-propagation
            torch.nn.utils.clip_grad_norm_(self.models_dict['generator'].parameters(),
                                           self.config.gradient_norm_clip)  # gradient clipping
            self.optimizers_dict['generator'].step()  # update parameters

        if not skip_stats:
            self.logger.update_current_losses('generator', self.epoch_type,
                                              generator_loss.data.cpu().detach().numpy(),
                                              generator_losses.cpu().detach().numpy())

            stats = {
                'generated_space_group_numbers': supercell_data.sg_ind.detach(),
                'identifiers': data.identifier,
                'smiles': data.smiles,
            }
            if step == 0:
                stats['generator_samples'] = supercell_data.cpu().detach()

            stats.update(losses_stats)

            dict_of_tensors_to_cpu_numpy(stats)

            self.logger.update_stats_dict(self.epoch_type,
                                          stats.keys(),
                                          stats.values(),
                                          mode='extend')

    def get_generator_density_target(self, data):
        if self.config.model_paths.regressor is not None:
            # predict the crystal volume cofficient and feed it as an input to the generator
            with torch.no_grad():
                target_rauv = self.models_dict['regressor'](data.clone().detach().to(self.config.device)).detach()[:,
                              0]
                assert False, "Missing standardization for volume regressor"
        else:
            atom_volumes = scatter(4 / 3 * torch.pi * self.vdw_radii_tensor[data.x[:, 0]] ** 3, data.batch,
                                   reduce='sum')
            target_rauv = data.reduced_volume / atom_volumes

        target_rauv += torch.randn_like(target_rauv) * self.config.generator.packing_target_noise

        return torch.maximum(target_rauv, torch.ones_like(target_rauv) * 0.01)

    def get_discriminator_output(self, data, i):
        """
        generate real and fake crystals
        and score them
        """
        '''get real supercells'''
        real_supercell_data = self.supercell_builder.prebuilt_unit_cell_to_supercell(
            data, self.config.supercell_size, self.config.discriminator.model.graph.cutoff)

        '''get fake supercells'''
        generated_samples_i, negative_type, generator_data, negatives_stats = \
            self.generate_discriminator_negatives(
                data, i, orientation=self.config.generator.canonical_conformer_orientation)

        fake_supercell_data, generated_cell_volumes = self.supercell_builder.build_zp1_supercells(
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
                 #'generated_cell_parameters': generated_samples_i.detach(),
                 'generated_cell_parameters': canonical_fake_cell_params.detach(),
                 'real_volume_fractions': real_volume_fractions.detach(),
                 'generated_volume_fractions': fake_volume_fractions.detach(),
                 'fake_volume_fractions': fake_volume_fractions.detach()}

        stats.update(negatives_stats)

        return (discriminator_output_on_real, discriminator_output_on_fake,
                rdf_dists, stats)

    def get_generator_samples(self, data, variation_factor, alignment_override=None):
        """
        set conformer orientation, optionally add noise, set the space group & symmetry information
        pass to generator and get cell parameters
        """
        mol_data = data.clone()

        # conformer orientation setting
        mol_data = set_molecule_alignment(mol_data, mode=alignment_override)

        # noise injection
        if self.config.positional_noise.generator > 0:
            mol_data.pos += torch.randn_like(mol_data.pos) * self.config.positional_noise.generator

        # update symmetry information
        if self.config.generate_sgs is not None:
            mol_data = update_crystal_symmetry_elements(mol_data,
                                                        self.config.generate_sgs,
                                                        self.sym_info,
                                                        randomize_sgs=True)

        # generate the samples
        with torch.no_grad():
            # center the molecules
            mol_data, _ = self.preprocess_ae_inputs(mol_data, no_noise=True, orientation_override=None)
            prior = self.generator_prior(data.num_graphs, mol_data.sg_ind).to(self.device)
            vector_mol_embedding = self.models_dict['autoencoder'].encode(mol_data.clone())
            scalar_mol_embedding = self.models_dict['autoencoder'].scalarizer(vector_mol_embedding)

            # loss, rmsd = self.models_dict['autoencoder'].check_embedding_quality(mol_data.clone())
        scaling_factor = (self.generator_prior.norm_factors[mol_data.sg_ind, :] + 1e-4)

        # append scalar and vector features
        scalar_mol_embedding = torch.cat((scalar_mol_embedding,
                                          #target_auv[:, None],
                                          prior[:, :9],
                                          variation_factor[:, None],
                                          scaling_factor),
                                         dim=1)
        reference_vector = torch.eye(3, dtype=torch.float32, device=self.device
                                     ).reshape(1, 3, 3
                                               ).repeat(data.num_graphs, 1, 1)

        vector_mol_embedding = torch.cat((vector_mol_embedding,
                                          prior[:, 9:, None],
                                          reference_vector),
                                         dim=2)

        generated_samples = self.models_dict['generator'].forward(scalar_mol_embedding,
                                                                  vector_mol_embedding,
                                                                  mol_data.sg_ind,
                                                                  )

        return generated_samples, prior, mol_data

    def get_generator_losses(self,
                             data,
                             generated_samples,
                             supercell_data,
                             generated_cell_volumes,
                             prior,
                             variation_factor,
                             skip_stats,
                             ):
        scaling_factor = (self.generator_prior.norm_factors[data.sg_ind, :] + 1e-4)
        scaled_deviation = torch.abs(prior - generated_samples) / scaling_factor
        prior_loss = F.relu(torch.linalg.norm(scaled_deviation, dim=1) - variation_factor)  # 'flashlight' search
        # prior_loss = torch.log(1 + torch.pow(scaled_deviation.norm(dim=1) / variation_factor, 4))

        dist_dict = get_intermolecular_dists_dict(supercell_data, 6, 100)

        molwise_overlap, molwise_normed_overlap, lj_potential, lj_loss \
            = vdw_analysis(self.vdw_radii_tensor, dist_dict, data.num_graphs, self.vdw_turnover_potential)

        vdw_score = -molwise_normed_overlap / data.num_atoms

        vdw_loss = lj_loss / data.num_atoms

        reduced_volume = generated_cell_volumes / supercell_data.sym_mult
        sample_rauv = reduced_volume / scatter(4 / 3 * torch.pi * self.vdw_radii_tensor[data.x[:, 0]] ** 3, data.batch,
                                               reduce='sum')

        # self.anneal_packing_loss(packing_loss)
        self.anneal_prior_loss(prior_loss)
        self.anneal_vdw_turnover(vdw_loss)

        generator_losses = (prior_loss * self.prior_loss_coefficient +
                            vdw_loss * self.vdw_loss_coefficient)
        supercell_data.loss = vdw_loss

        if skip_stats:
            stats_dict = {}
        else:
            stats_dict = {
                'generator_per_mol_vdw_loss': vdw_loss.detach(),
                'generator_per_mol_vdw_score': vdw_score.detach(),
                'generator_prior_loss': prior_loss.detach(),
                # 'generator_packing_loss': packing_loss.detach(),
                'generator_packing_prediction': sample_rauv.detach(),
                #'generator_packing_target': target_rauv.detach(),
                'generator_prior': prior.detach(),
                'generator_scaling_factor': scaling_factor.detach(),
                'generated_cell_parameters': generated_samples.detach(),
                'generator_scaled_deviation': scaled_deviation.detach(),
                'generator_sample_lj_energy': lj_potential.detach(),
                'generator_sample_lj_loss': vdw_loss.detach(),
                'generator_variation_factor': variation_factor.detach(),
            }
        return generator_losses, stats_dict, supercell_data.detach()

    def anneal_packing_loss(self, packing_loss):
        # dynamically soften the packing loss when the model is doing well
        if packing_loss.mean() < 0.02:
            self.packing_loss_coefficient *= 0.99
        if (packing_loss.mean() > 0.03) and (self.packing_loss_coefficient < 10):
            self.packing_loss_coefficient *= 1.01
        self.logger.packing_loss_coefficient = self.packing_loss_coefficient

    def anneal_prior_loss(self, prior_loss):
        # dynamically soften the packing loss when the model is doing well
        if (prior_loss.mean() < self.config.generator.prior_coefficient_threshold) and (
                self.prior_loss_coefficient > 0.01):
            self.prior_loss_coefficient *= 0.99
        if (prior_loss.mean() > self.config.generator.prior_coefficient_threshold) and (
                self.prior_loss_coefficient < 10):
            self.prior_loss_coefficient *= 1.01
        self.logger.prior_loss_coefficient = self.prior_loss_coefficient

    def anneal_vdw_turnover(self, vdw_loss, prior_loss):
        # dynamically harden the LJ repulsive potential when the model is doing well
        # if doing well on prior and LJ, and not hit max value, dynamically increase
        if ((vdw_loss.mean() < 0) and
                (self.vdw_turnover_potential < 10) and
                (prior_loss.mean() < self.config.generator.prior_coefficient_threshold)):
            self.vdw_turnover_potential += 0.01
        # never soften - monotonic convergence
        self.logger.vdw_turnover_potential = self.vdw_turnover_potential

    def init_gaussian_generator(self):
        """
        init gaussian generator for cell parameter sampling
        """
        self.gaussian_generator = IndependentGaussianGenerator(input_dim=12,
                                                               means=self.dataDims['lattice_means'],
                                                               stds=self.dataDims['lattice_stds'],
                                                               sym_info=self.sym_info,
                                                               device=self.config.device,
                                                               cov_mat=self.dataDims['lattice_cov_mat'])

    def init_generator_prior(self):
        """
        Initialize a prior for the generator model
        """
        ''' 
        init gaussian generator for cell parameter sampling
        '''
        self.generator_prior = GeneratorPrior(sym_info=self.sym_info, device=self.config.device)

    def what_generators_to_use(self, override_randn, override_distorted, override_adversarial):
        """
        pick what generator to use on a given step
        """
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

    def generate_discriminator_negatives(self, real_data, i, override_adversarial=False, override_randn=False,
                                         override_distorted=False, orientation='random'):
        """
        use one of the available cell generation tools to sample cell parameters, to be fed to the discriminator
        """
        n_generators, generator_ind = self.what_generators_to_use(override_randn, override_distorted,
                                                                  override_adversarial)

        if (self.config.discriminator.train_adversarially or override_adversarial) and (generator_ind == 1):
            negative_type = 'generator'
            with torch.no_grad():
                generated_samples, _, generator_data = self.get_generator_samples(real_data,
                                                                                  alignment_override=orientation)

            stats = {'generator_sample_source': np.zeros(len(generated_samples))}

        elif (self.config.discriminator.train_on_randn or override_randn) and (generator_ind == 2):
            generator_data = set_molecule_alignment(real_data.clone(), mode=orientation)
            negative_type = 'randn'
            generated_samples = self.gaussian_generator.forward(real_data.num_graphs, real_data).to(self.config.device)

            stats = {'generator_sample_source': np.ones(len(generated_samples))}

        elif (self.config.discriminator.train_on_distorted or override_distorted) and (generator_ind == 3):
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
        real_cell_params = torch.cat([real_data.cell_lengths, real_data.cell_angles, real_data.pose_params0], dim=1)
        generated_samples_std = (real_cell_params - self.lattice_means) / self.lattice_stds

        if distortion_override is not None:
            distortion = torch.randn_like(generated_samples_std) * distortion_override
        else:
            # distortion types
            # pick n=[1,4] of the 4 cell param types and proportionally noise them
            distortion_mask = torch.randint(0, 2, size=(generated_samples_std.shape[0], 4),
                                            device=generated_samples_std.device, dtype=torch.long)
            distortion_mask[distortion_mask.sum(1) == 0] = 1  # any zero entries go to all
            distortion_mask = distortion_mask * self.chi_scaling_factors[distortion_mask.sum(1) - 1][:, None].float()
            distortion_mask = distortion_mask.repeat_interleave(3, dim=1)

            if self.config.discriminator.distortion_magnitude == -1:
                distortion_magnitude = torch.logspace(-1.5, 0.5, len(generated_samples_std)).to(
                    generated_samples_std.device)[:, None]  # wider range
            else:
                distortion_magnitude = self.config.discriminator.distortion_magnitude

            distortion = torch.randn_like(generated_samples_std) * distortion_magnitude * distortion_mask

        distorted_samples_std = (generated_samples_std + distortion).to(
            self.device)  # add jitter and return in standardized basis

        distorted_samples_clean = clean_cell_params(
            distorted_samples_std, real_data.sg_ind,
            self.lattice_means, self.lattice_stds,
            self.sym_info, self.supercell_builder.asym_unit_dict,
            rescale_asymmetric_unit=False, destandardize=True, mode='hard')

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

    def update_lr(self, update_lr_ratio=None):
        for model_name in self.model_names:
            if self.config.__dict__[model_name].optimizer is not None:

                if update_lr_ratio is not None:
                    current_lr = self.optimizers_dict[model_name].param_groups[0]['lr']
                    override_lr = current_lr * update_lr_ratio
                else:
                    override_lr = None

                self.optimizers_dict[model_name], learning_rate = set_lr(
                    self.schedulers_dict[model_name],
                    self.optimizers_dict[model_name],
                    self.config.__dict__[model_name].optimizer,
                    self.logger.current_losses[model_name]['mean_train'],
                    self.hit_max_lr_dict[model_name],
                    override_lr)

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

    def compute_similarity_penalty(self, generated_samples, prior, raw_samples):
        # DEPRECATED not currently useful
        """
        by hook or crook
        force samples to be more diverse

        Parameters
        ----------
        generated_samples
        prior

        Returns
        -------
        """
        # simply punish the model samples for deviating from the prior
        # TODO shift the prior to be more appropriate for this task
        # uniform angles
        # standardize cell lengths by molecule size

        # euclidean distance
        similarity_penalty = ((prior - raw_samples) ** 2).sum(1).sqrt()
        return similarity_penalty

    def score_adversarially(self, supercell_data, discriminator_noise=None, return_latent=False):
        """
        get a score for generated samples
        optionally add noise to the positions of all atoms before scoring
        option to return final layer activation of discriminator_model
        Parameters
        ----------
        supercell_data
        discriminator_noise
        return_latent

        Returns
        -------

        """
        if discriminator_noise is not None:
            supercell_data.pos += torch.randn_like(
                supercell_data.pos) * discriminator_noise
        else:
            if self.config.positional_noise.discriminator > 0:
                supercell_data.pos += torch.randn_like(
                    supercell_data.pos) * self.config.positional_noise.discriminator

        if (self.config.device.lower() == 'cuda') and (supercell_data.x.device != 'cuda'):
            supercell_data = supercell_data.cuda()

        discriminator_score, dist_dict, latent = self.adversarial_score(supercell_data, return_latent=True)

        if return_latent:
            return discriminator_score, dist_dict, latent
        else:
            return discriminator_score, dist_dict

    def reinitialize_models(self, generator, discriminator, regressor):
        """
        reset model weights, if we did not load it from a given path
        @param generator:
        @param discriminator:
        @param regressor:
        @return:
        """
        torch.manual_seed(self.config.seeds.model)
        print('Reinitializing models and optimizer')
        if self.config.model_paths.generator is None:
            generator.apply(weight_reset)
        if self.config.model_paths.discriminator is None:
            discriminator.apply(weight_reset)
        if self.config.model_paths.regressor is None:
            regressor.apply(weight_reset)

        return generator, discriminator, regressor

    def reload_model_checkpoint_configs(self):
        for model_name, model_path in self.config.model_paths.__dict__.items():
            if model_path is not None:
                checkpoint = torch.load(model_path)
                model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
                self.config.__dict__[model_name].optimizer = model_config.optimizer
                self.config.__dict__[model_name].model = model_config.model
                print(f"Reloading {model_name} {model_path}")

    #
    # def crystal_search(self, molecule_data, batch_size=None, data_contains_ground_truth=True):  # currently deprecated
    #     """
    #     execute a search for a single crystal target
    #     if the target is known, compare it to our best guesses
    #     """
    #     self.source_directory = os.getcwd()
    #     self.prep_new_working_directory()
    #
    #     with wandb.init(config=self.config,
    #                     project=self.config.wandb.project_name,
    #                     entity=self.config.wandb.username,
    #                     tags=[self.config.logger.experiment_tag],
    #                     settings=wandb.Settings(code_dir=".")):
    #
    #         wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
    #
    #         if batch_size is None:
    #             batch_size = self.config.min_batch_size
    #
    #         num_discriminator_opt_steps = 100
    #         num_mcmc_opt_steps = 100
    #         max_iters = 10
    #
    #         self.init_gaussian_generator()
    #         self.initialize_models_optimizers_schedulers()
    #
    #         self.models_dict['generator'].eval()
    #         self.models_dict['regressor'].eval()
    #         self.models_dict['discriminator'].eval()
    #
    #         '''instantiate batch'''
    #         crystaldata_batch = self.collater([molecule_data for _ in range(batch_size)]).to(self.device)
    #         refresh_inds = torch.arange(batch_size)
    #         converged_samples_list = []
    #         optimization_trajectories = []
    #
    #         for opt_iter in range(max_iters):
    #             crystaldata_batch = self.refresh_crystal_batch(crystaldata_batch, refresh_inds=refresh_inds)
    #
    #             crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(
    #                 crystaldata_batch,
    #                 mode='mcmc',
    #                 num_steps=num_mcmc_opt_steps,
    #                 temperature=0.05,
    #                 step_size=0.01)
    #             optimization_trajectories.append(opt_traj)
    #
    #             crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(
    #                 crystaldata_batch,
    #                 mode='discriminator',
    #                 num_steps=num_discriminator_opt_steps)
    #             optimization_trajectories.append(opt_traj)
    #
    #             crystaldata_batch, refresh_inds, converged_samples = self.prune_crystaldata_batch(crystaldata_batch,
    #                                                                                               optimization_trajectories)
    #
    #             converged_samples_list.extend(converged_samples)
    #
    #         aa = 1
    #         # do clustering
    #
    #         # compare to ground truth
    #         # add convergence flags based on completeness of sampling
    #
    #         # '''compare samples to ground truth'''
    #         # if data_contains_ground_truth:
    #         #     ground_truth_analysis = self.analyze_real_crystal(molecule_data)
    #         #

    # def prune_crystaldata_batch(self, crystaldata_batch, optimization_trajectories):
    #     """
    #     Identify trajectories which have converged.
    #     """
    #
    #     """
    #     combined_traj_dict = {key: np.concatenate(
    #         [traj[key] for traj in optimization_trajectories], axis=0)
    #         for key in optimization_trajectories[1].keys()}
    #
    #     from plotly.subplots import make_subplots
    #     import plotly.graph_objects as go
    #
    #     from plotly.subplots import make_subplots
    #     import plotly.graph_objects as go
    #     fig = make_subplots(cols=3, rows=1, subplot_titles=['score','vdw_score','packing_coeff'])
    #     for i in range(crystaldata_batch.num_graphs):
    #         for j, key in enumerate(['score','vdw_score','packing_coeff']):
    #             col = j % 3 + 1
    #             row = j // 3 + 1
    #             fig.add_scattergl(y=combined_traj_dict[key][:, i], name=i, legendgroup=i, showlegend=True if j == 0 else False, row=row, col=col)
    #     fig.show(renderer='browser')
    #
    #     """
    #
    #     refresh_inds = np.arange(crystaldata_batch.num_graphs)  # todo write a function that actually checks for this
    #     converged_samples = [crystaldata_batch[i] for i in refresh_inds.tolist()]
    #
    #     return crystaldata_batch, refresh_inds, converged_samples

    # def optimize_crystaldata_batch(self, batch, mode, num_steps, temperature=None, step_size=None):  # DEPRECATED todo redevelop
    #     """
    #     method which takes a batch of crystaldata objects
    #     and optimzies them according to a score model either
    #     with MCMC or gradient descent
    #     """
    #     if mode.lower() == 'mcmc':
    #         sampling_dict = mcmc_sampling(
    #             self.models_dict['discriminator'], batch,
    #             self.supercell_builder,
    #             num_steps, self.vdw_radii,
    #             supercell_size=5, cutoff=6,
    #             sampling_temperature=temperature,
    #             lattice_means=self.dataDims['lattice_means'],
    #             lattice_stds=self.dataDims['lattice_stds'],
    #             step_size=step_size,
    #         )
    #     elif mode.lower() == 'discriminator':
    #         sampling_dict = gradient_descent_sampling(
    #             self.models_dict['discriminator'], batch,
    #             self.supercell_builder,
    #             num_steps, 1e-3,
    #             torch.optim.Rprop, self.vdw_radii,
    #             lattice_means=self.dataDims['lattice_means'],
    #             lattice_stds=self.dataDims['lattice_stds'],
    #             supercell_size=5, cutoff=6,
    #         )
    #     else:
    #         assert False, f"{mode.lower()} is not a valid sampling mode!"
    #
    #     '''return best sample'''
    #     best_inds = np.argmax(sampling_dict['score'], axis=0)
    #     best_samples = sampling_dict['std_cell_params'][best_inds, np.arange(batch.num_graphs), :]
    #     supercell_data, _ = \
    #         self.supercell_builder.build_zp1_supercells(
    #             batch, torch.tensor(best_samples, dtype=torch.float32, device=batch.x.device),
    #             5, 6,
    #             align_to_standardized_orientation=True,
    #             target_handedness=batch.aunit_handedness)
    #
    #     output, proposed_dist_dict = self.models_dict['discriminator'](supercell_data.clone().cuda(), return_dists=True)
    #
    #     rebuilt_sample_scores = softmax_and_score(output[:, :2]).cpu().detach().numpy()
    #
    #     cell_params_difference = np.amax(
    #         np.sum(np.abs(supercell_data.cell_params.cpu().detach().numpy() - best_samples), axis=1))
    #     rebuilt_scores_difference = np.amax(np.abs(rebuilt_sample_scores - sampling_dict['score'].max(0)))
    #
    #     if rebuilt_scores_difference > 1e-2 or cell_params_difference > 1e-2:
    #         aa = 1
    #         assert False, "Best cell rebuilding failed!"  # confirm we rebuilt the cells correctly
    #
    #     sampling_dict['best_samples'] = best_samples
    #     sampling_dict['best_scores'] = sampling_dict['score'].max(0)
    #     sampling_dict['best_vdws'] = np.diag(sampling_dict['vdw_score'][best_inds, :])
    #
    #     best_batch = batch.clone()
    #     best_batch.cell_params = torch.tensor(best_samples, dtype=torch.float32, device=supercell_data.x.device)
    #
    #     return best_batch, sampling_dict
    #
    # def refresh_crystal_batch(self, crystaldata, refresh_inds, generator='gaussian', space_groups: torch.tensor = None):
    #     # crystaldata = self.set_molecule_alignment(crystaldata, right_handed=False, mode_override=mol_orientation)
    #
    #     if space_groups is not None:
    #         crystaldata.sg_ind = space_groups
    #
    #     if generator == 'gaussian':
    #         samples = self.gaussian_generator.forward(crystaldata.num_graphs, crystaldata).to(self.config.device)
    #         crystaldata.cell_params = samples[refresh_inds]
    #         # todo add option for generator here
    #
    #     return crystaldata

    def generator_density_matching_loss(self,
                                        target_rauv,
                                        data,
                                        sym_mult,
                                        samples,
                                        precomputed_volumes=None,
                                        loss_func='mse'):
        """
        compute packing coefficients for generated cells
        compute losses relating to packing density
        """
        if precomputed_volumes is None:
            volumes_list = []
            for i in range(len(samples)):  # todo implement parallel version
                volumes_list.append(cell_vol_torch(samples[i, 0:3], samples[i, 3:6]))
            cell_volume = torch.stack(volumes_list)
        else:
            cell_volume = precomputed_volumes

        reduced_volume = cell_volume / sym_mult
        atom_volumes = scatter(4 / 3 * torch.pi * self.vdw_radii_tensor[data.x[:, 0]] ** 3, data.batch,
                               reduce='sum')
        generated_rauv = reduced_volume / atom_volumes

        if loss_func == 'mse':
            packing_loss = F.mse_loss(generated_rauv, target_rauv, reduction='none')
        elif loss_func == 'l1':
            packing_loss = F.smooth_l1_loss(generated_rauv, target_rauv, reduction='none')
        else:
            assert False, "Must pick from the set of implemented packing loss functions 'mse', 'l1'"
        return packing_loss, generated_rauv

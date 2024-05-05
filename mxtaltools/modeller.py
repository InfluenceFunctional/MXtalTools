import gc
import os
import sys
from argparse import Namespace
from datetime import datetime
from distutils.dir_util import copy_tree
from shutil import copy
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.random
import wandb
from scipy.spatial.transform import Rotation as R
from torch import backends
from torch.nn import functional as F
from torch_geometric.loader.dataloader import Collater
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.utils import softmax_np, init_sym_info, compute_rdf_distance, flatten_dict, namespace2dict
from mxtaltools.constants.asymmetric_units import asym_unit_dict
from mxtaltools.constants.atom_properties import VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD
from mxtaltools.crystal_building.builder import SupercellBuilder
from mxtaltools.crystal_building.utils import (clean_cell_params, set_molecule_alignment)
from mxtaltools.crystal_building.utils import update_crystal_symmetry_elements
from mxtaltools.csp.SampleOptimization import gradient_descent_sampling, mcmc_sampling
from mxtaltools.dataset_management.data_manager import DataManager
from mxtaltools.dataset_management.dataloader_utils import get_dataloaders, update_dataloader_batch_size
#from mxtaltools.dataset_management.process_GEOM import geom_msgpack_to_minimal_dataset
from mxtaltools.models.autoencoder_models import PointAutoencoder
from mxtaltools.models.crystal_rdf import new_crystal_rdf
from mxtaltools.models.crystal_models import MolCrystal
from mxtaltools.models.embedding_regression_models import EmbeddingRegressor
from mxtaltools.models.generator_models import IndependentGaussianGenerator, CrystalGenerator
from mxtaltools.models.mol_classifier import PolymorphClassifier
from mxtaltools.models.regression_models import MoleculeRegressor
from mxtaltools.models.utils import (reload_model, init_schedulers, softmax_and_score, save_checkpoint, set_lr,
                                     cell_vol_torch, init_optimizer, get_regression_loss,
                                     slash_batch, compute_gaussian_overlap,
                                     compute_type_evaluation_overlap,
                                     compute_coord_evaluation_overlap,
                                     compute_full_evaluation_overlap, get_node_weights,
                                     compute_reduced_volume_fraction, dict_of_tensors_to_cpu_numpy)
from mxtaltools.models.utils import (weight_reset, get_n_config)
from mxtaltools.models.vdw_overlap import vdw_overlap
from mxtaltools.reporting.logger import Logger
from mxtaltools.reporting.ae_reporting import decoder_agglomerative_clustering, scaffolded_decoder_clustering, \
    extract_true_and_predicted_points


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
        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        self.packing_loss_coefficient = 1
        '''get some physical constants'''
        self.atom_weights = ATOM_WEIGHTS
        self.vdw_radii = VDW_RADII
        self.sym_info = init_sym_info()
        for key, value in ELECTRONEGATIVITY.items():
            if value is None:
                ELECTRONEGATIVITY[key] = 0
        self.electronegativity_tensor = torch.tensor(list(ELECTRONEGATIVITY.values()), dtype=torch.float32,
                                                     device=self.config.device)

        self.supercell_builder = SupercellBuilder(device=self.config.device, rotation_basis='spherical')

        self.train_models_dict = {
            'discriminator': False,
            'generator': False,
            'regressor': False,
            'autoencoder': False,
            'embedding_regressor': False,
        }

        '''set space groups to be included and generated'''
        if self.config.generate_sgs == 'all':
            self.config.generate_sgs = [self.sym_info['space_groups'][int(key)] for key in asym_unit_dict.keys()]

        self.collater = Collater(None, None)

        '''compute the ratios between the norms of n-dimensional gaussians (means of chi distribution)'''
        m1 = torch.sqrt(torch.ones(1) * 2) * torch.exp(torch.lgamma(torch.ones(1) * (12 + 1) / 2)) / torch.exp(
            torch.lgamma(torch.ones(1) * 12 / 2))
        self.chi_scaling_factors = torch.zeros(4, dtype=torch.float, device=self.device)
        for ind, ni in enumerate([3, 6, 9, 12]):
            m2 = torch.sqrt(torch.ones(1) * 2) * torch.exp(torch.lgamma(torch.ones(1) * (ni + 1) / 2)) / torch.exp(
                torch.lgamma(torch.ones(1) * ni / 2))
            self.chi_scaling_factors[ind] = m1 / m2

        self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        self.atom_weights_tensor = torch.tensor(list(ATOM_WEIGHTS.values()), device=self.device)
        self.electronegativity_tensor = torch.tensor(list(ELECTRONEGATIVITY.values()), device=self.device)
        self.group_tensor = torch.tensor(list(GROUP.values()), device=self.device)
        self.period_tensor = torch.tensor(list(PERIOD.values()), device=self.device)

        ''' # the distributions for the independent spherical coordinates of randn distributed vectors in 3d - chi 3, sine, and uniform
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # sine distribution https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/05%3A_Special_Distributions/5.27%3A_The_Sine_Distribution
        
        def to_spherical(vec):
            """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
            x, y, z = vec.T
            r = np.linalg.norm(vec, axis=1)
            theta = np.arctan2(np.sqrt(x * x + y * y), z)
            phi = np.arctan2(y, x)
            return r, theta, phi
        
        v = np.random.randn(100000,3)
        r, theta, phi = to_spherical(v)
        
        x = np.linspace(0, np.pi, 1001)
        
        theta2 = np.arcsin(2*(x - np.pi/2)) + np.pi/2
        fig = make_subplots(rows=1, cols=3, subplot_titles =['Radial Norm, Chi (3)','Polar Angle (sine)','Azimuthal Angle (uniform)'])
        fig.add_histogram(x=r, nbinsx=100, histnorm='probability density', row=1, col=1)
        fig.add_histogram(x=np.sqrt(np.random.chisquare(3.0, size=10000)), histnorm='probability density', row=1, col=1)
        for value in [theta, theta2]:
            fig.add_histogram(x=value,nbinsx=100,histnorm='probability density', row=1, col=2)
        fig.add_scattergl(x=x, y=1/2*np.sin(x), row=1,col=2)
        
        fig.add_histogram(x=phi, nbinsx=100, histnorm='probability density', row=1, col=3)
        fig.add_histogram(x=np.random.uniform(-np.pi,np.pi,size=10000), nbinsx=100, histnorm='probability density', row=1, col=3)
        
        fig.show(renderer='browser')
        '''

    def prep_new_working_directory(self):
        """
        make a workdir
        copy the source directory to the new working directory
        """
        self.make_sequential_directory()
        self.copy_source_to_workdir()

    def copy_source_to_workdir(self):
        os.mkdir(self.working_directory + '/source')
        yaml_path = self.config.paths.yaml_path
        copy_tree("mxtaltools/common", self.working_directory + "/source/common")
        copy_tree("mxtaltools/crystal_building", self.working_directory + "/source/crystal_building")
        copy_tree("mxtaltools/dataset_management", self.working_directory + "/source/dataset_management")
        copy_tree("mxtaltools/models", self.working_directory + "/source/models")
        copy_tree("mxtaltools/reporting", self.working_directory + "/source/reporting")
        copy_tree("mxtaltools/csp", self.working_directory + "/source/csp")
        copy("mxtaltools/modeller.py", self.working_directory + "/source")
        copy("main.py", self.working_directory + "/source")
        np.save(self.working_directory + '/run_config', self.config)
        os.chdir(self.working_directory)  # move to working dir
        copy(yaml_path, os.getcwd())  # copy full config for reference
        print('Starting fresh run ' + self.working_directory)

    def make_sequential_directory(self):  # make working directory
        """
        make a new working directory labelled by the time & date
        hopefully does not overlap with any other workdirs
        :return:
        """
        self.run_identifier = str(self.config.paths.yaml_path).split('.yaml')[0].split('configs')[1].replace('\\',
                                                                                                             '_').replace(
            '/', '_') + '_' + datetime.today().strftime("%d-%m-%H-%M-%S")
        self.working_directory = self.config.workdir + self.run_identifier
        os.mkdir(self.working_directory)

    def init_models(self):
        """
        Initialize models, optimizers, schedulers
        for models we will not use, just set them as nn.Linear(1,1)
        :return:
        """
        self.times['init_models_start'] = time()
        self.model_names = self.config.model_names
        self.reload_model_checkpoint_configs()

        print("Initializing model(s) for " + self.config.mode)
        self.models_dict = {}
        if self.config.mode == 'gan' or self.config.mode == 'search':
            self.models_dict['generator'] = CrystalGenerator(self.config.seeds.model,
                                                             self.device,
                                                             self.config.generator.model,
                                                             self.sym_info,
                                                             self.dataDims['atom_features'],
                                                             self.dataDims['molecule_features'],
                                                             self.dataDims['node_standardization_vector'],
                                                             self.dataDims['graph_standardization_vector'],
                                                             self.dataDims['lattice_means'],
                                                             self.dataDims['lattice_stds']
                                                             )
            self.models_dict['discriminator'] = MolCrystal(self.config.seeds.model,
                                                           self.config.discriminator.model,
                                                           self.dataDims['atom_features'],
                                                           self.dataDims['molecule_features'],
                                                           self.dataDims['node_standardization_vector'],
                                                           self.dataDims['graph_standardization_vector'])
        if self.config.mode == 'discriminator':
            self.models_dict['generator'] = nn.Linear(1, 1)
            self.models_dict['discriminator'] = MolCrystal(self.config.seeds.model,
                                                           self.config.discriminator.model,
                                                           self.dataDims['atom_features'],
                                                           self.dataDims['molecule_features'],
                                                           self.dataDims['node_standardization_vector'],
                                                           self.dataDims['graph_standardization_vector'])
        if self.config.mode == 'regression' or self.config.model_paths.regressor is not None:
            self.models_dict['regressor'] = MoleculeRegressor(self.config.seeds.model,
                                                              self.config.regressor.model,
                                                              self.dataDims['atom_features'],
                                                              self.dataDims['molecule_features'],
                                                              self.dataDims['node_standardization_vector'],
                                                              self.dataDims['graph_standardization_vector'])
        if self.config.mode == 'autoencoder' or self.config.model_paths.autoencoder is not None:
            self.models_dict['autoencoder'] = PointAutoencoder(self.config.seeds.model,
                                                               self.config.autoencoder.model,
                                                               int(torch.sum(self.autoencoder_type_index != -1)),
                                                               self.autoencoder_type_index,
                                                               self.config.autoencoder.molecule_radius_normalization,
                                                               infer_protons=self.config.autoencoder.infer_protons
                                                               )
        if self.config.mode == 'embedding_regression' or self.config.model_paths.embedding_regressor is not None:
            self.models_dict['autoencoder'] = PointAutoencoder(self.config.seeds.model,
                                                               self.config.autoencoder.model,
                                                               int(torch.sum(self.autoencoder_type_index != -1)),
                                                               self.autoencoder_type_index,
                                                               self.config.autoencoder.molecule_radius_normalization,
                                                               infer_protons=self.config.autoencoder.infer_protons
                                                               )
            for param in self.models_dict['autoencoder'].parameters():  # freeze encoder
                param.requires_grad = False
            self.config.EmbeddingRegressor.model.bottleneck_dim = self.config.autoencoder.model.bottleneck_dim
            self.models_dict['embedding_regressor'] = EmbeddingRegressor(self.config.seeds.model,
                                                                         self.config.EmbeddingRegressor.model,
                                                                         num_targets=self.config.EmbeddingRegressor.num_targets
                                                                         )
            assert self.config.model_paths.autoencoder is not None  # must preload the encoder

        if self.config.mode == 'polymorph_classification':
            self.models_dict['polymorph_classifier'] = PolymorphClassifier(self.config.seeds.model,
                                                                           self.config.polymorph_classifier.model,
                                                                           self.dataDims)

        null_models = {name: nn.Linear(1, 1) for name in self.model_names if
                       name not in self.models_dict.keys()}  # initialize null models
        self.models_dict.update(null_models)

        if self.config.device.lower() == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            for model in self.models_dict.values():
                model.cuda()

        self.optimizers_dict = \
            {model_name: init_optimizer(model_name, self.config.__dict__[model_name].optimizer, model)
             for model_name, model in self.models_dict.items()
             }

        for model_name, model_path in self.config.model_paths.__dict__.items():
            if model_path is not None:
                self.models_dict[model_name], self.optimizers_dict[model_name] = reload_model(
                    self.models_dict[model_name], self.optimizers_dict[model_name],
                    self.config.model_paths.__dict__[model_name]
                )

        self.schedulers_dict = {model_name: init_schedulers(
            self.optimizers_dict[model_name], self.config.__dict__[model_name].optimizer)
            for model_name in self.model_names}

        num_params_dict = {model_name + "_num_params": get_n_config(model) for model_name, model in
                           self.models_dict.items()}
        [print(
            f'{model_name} {num_params_dict[model_name] / 1e6:.3f} million or {int(num_params_dict[model_name])} parameters')
            for model_name in num_params_dict.keys()]

        self.times['init_models_end'] = time()
        return num_params_dict

    def load_dataset_and_dataloaders(self, override_test_fraction=None, override_dataset=None):
        """
        use data manager to load and filter dataset
        use dataset builder to generate crystaldata objects
        return dataloaders
        """

        """load and filter dataset"""
        data_manager = DataManager(device=self.device,
                                   datasets_path=self.config.dataset_path,
                                   config=self.config.dataset)
        data_manager.load_dataset_for_modelling(
            dataset_name=self.config.dataset_name,
            filter_conditions=self.config.dataset.filter_conditions,
            filter_polymorphs=self.config.dataset.filter_polymorphs,
            filter_duplicate_molecules=self.config.dataset.filter_duplicate_molecules,
            filter_protons=self.config.dataset.filter_protons,
            override_dataset=override_dataset
        )
        self.dataDims = data_manager.dataDims
        self.lattice_means = torch.tensor(self.dataDims['lattice_means'], device=self.device)
        self.lattice_stds = torch.tensor(self.dataDims['lattice_stds'], device=self.device)
        self.times['dataset_loading'] = data_manager.times

        if self.train_models_dict['autoencoder'] or self.train_models_dict[
            'embedding_regressor']:  # todo change this to 'if autoencoder exists' or some proxy
            self.config.autoencoder_sigma = self.config.autoencoder.init_sigma
            self.config.autoencoder.molecule_radius_normalization = self.dataDims['standardization_dict']['radius'][
                'max']

            allowed_types = np.array(self.dataDims['allowed_atom_types'])
            type_translation_index = np.zeros(allowed_types.max() + 1) - 1
            for ind, atype in enumerate(allowed_types):
                type_translation_index[atype] = ind
            self.autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')

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
                # do not filter duplicates - e.g., in Blind test they're almost all duplicates!
                filter_duplicate_molecules=False,
                filter_protons=self.config.dataset.filter_protons,
            )
            self.times['extra_dataset_loading'] = data_manager.times
        else:
            extra_data_manager = None

        """return dataloaders"""
        if override_test_fraction is not None:
            test_fraction = override_test_fraction
        else:
            test_fraction = self.config.dataset.test_fraction

        return self.prep_dataloaders(data_manager, extra_data_manager, test_fraction)

    def prep_dataloaders(self, dataset_builder, extra_dataset_builder=None, test_fraction=0.2,
                         override_batch_size: int = None):
        """
        get training, test, ane optionall extra validation dataloaders
        """
        self.times['dataloader_start'] = time()
        if override_batch_size is None:
            loader_batch_size = self.config.min_batch_size
        else:
            loader_batch_size = override_batch_size
        train_loader, test_loader = get_dataloaders(dataset_builder,
                                                    machine=self.config.machine,
                                                    batch_size=loader_batch_size,
                                                    test_fraction=test_fraction)
        self.config.current_batch_size = loader_batch_size
        print("Initial training batch size set to {}".format(self.config.current_batch_size))
        del dataset_builder

        # data_loader for a secondary test set - analysis is hardcoded for CSD Blind Tests 5 & 6
        if extra_dataset_builder is not None:
            _, extra_test_loader = get_dataloaders(extra_dataset_builder,
                                                   machine=self.config.machine,
                                                   batch_size=loader_batch_size,
                                                   test_fraction=1)
            del extra_dataset_builder
        else:
            extra_test_loader = None
        self.times['dataloader_end'] = time()
        return train_loader, test_loader, extra_test_loader

    def autoencoder_molecule_generation(self):

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
        _, data_loader, extra_test_loader = self.load_dataset_and_dataloaders(override_test_fraction=1)
        num_params_dict = self.init_models()

        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         settings=wandb.Settings(code_dir="."))):
            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(num_params_dict.values()))),
                       "Initial Batch Size": self.config.current_batch_size})

            self.models_dict['autoencoder'].eval()
            self.epoch_type = 'test'

            for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
                self.autoencoder_generation_step(data)

            # post epoch processing
            self.logger.concatenate_stats_dict(self.epoch_type)

    def autoencoder_generation_step(self, data):
        # TODO molecule validity checker
        data.to(self.device)
        import plotly.graph_objects as go
        decoding = self.models_dict['autoencoder'].decode(torch.randn(size=(
            data.num_graphs,
            3,
            self.config.autoencoder.model.bottleneck_dim
        ), dtype=torch.float32, device=self.device))

        decoded_data = data.clone()
        decoded_data.pos = decoding[:, :3]
        decoded_data.batch = torch.arange(data.num_graphs).repeat_interleave(
            self.config.autoencoder.model.num_decoder_nodes).to(self.config.device)

        nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = \
            get_node_weights(data, decoded_data, decoding, self.config.autoencoder.model.num_decoder_nodes,
                             self.config.autoencoder.node_weight_temperature)

        decoded_data.x = F.softmax(decoding[:, 3:-1], dim=1)
        decoded_data.aux_ind = nodewise_weights_tensor

        colors = ['rgb(229, 134, 6)', 'rgb(93, 105, 177)', 'rgb(82, 188, 163)', 'rgb(153, 201, 69)',
                  'rgb(204, 97, 176)', 'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)',
                  'rgb(118, 78, 159)', 'rgb(237, 100, 90)',
                  'rgb(165, 170, 153)'] * 10
        colorscales = [[[0, 'rgba(0, 0, 0, 0)'], [1, color]] for color in colors]
        cmax = 1
        for graph_ind in range(5):
            # points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()
            # fig = go.Figure()
            # for j in range(self.dataDims['num_atom_types']):
            #
            #     pred_type_weights = (decoded_data.aux_ind[decoded_data.batch == graph_ind] * decoded_data.x[decoded_data.batch == graph_ind, j]).cpu().detach().numpy()
            #
            #     fig.add_trace(go.Scatter3d(x=points_pred[:, 0] * self.config.autoencoder.molecule_radius_normalization,
            #                                y=points_pred[:, 1] * self.config.autoencoder.molecule_radius_normalization,
            #                                z=points_pred[:, 2] * self.config.autoencoder.molecule_radius_normalization,
            #                                mode='markers', marker=dict(size=10, color=pred_type_weights, colorscale=colorscales[j], cmax=cmax, cmin=0), opacity=1, marker_line_color='white',
            #                                showlegend=True,
            #                                name=f'Predicted type {j}',
            #                                legendgroup=f'Predicted type {j}',
            #                                ))

            fig = go.Figure()
            coords_true, coords_pred, points_true, points_pred, sample_weights = (
                extract_true_and_predicted_points(data, decoded_data, graph_ind,
                                                  self.config.autoencoder.molecule_radius_normalization,
                                                  self.dataDims['num_atom_types'], to_numpy=True))

            glom_points_pred, glom_pred_weights = decoder_agglomerative_clustering(points_pred, sample_weights, 0.75)

            for j in range(self.dataDims['num_atom_types']):
                type_inds = np.argwhere(np.argmax(glom_points_pred[:, 3:], axis=1) == j)[:, 0]

                pred_type_weights = glom_points_pred[type_inds, j + 3] * glom_pred_weights[type_inds]
                atom_type = int(torch.argwhere(self.autoencoder_type_index == j)) + 1

                fig.add_trace(go.Scatter3d(x=glom_points_pred[type_inds, 0], y=glom_points_pred[type_inds, 1],
                                           z=glom_points_pred[type_inds, 2],
                                           mode='markers',
                                           marker=dict(size=10, color=pred_type_weights, colorscale=colorscales[j],
                                                       cmax=cmax, cmin=0), opacity=1,
                                           marker_line_color='black', marker_line_width=30,
                                           showlegend=True,  # if j == 0 else False,
                                           name=f'Clustered Atoms Type {atom_type}',
                                           legendgroup=f'Clustered Atoms'
                                           ))
            fig.show(renderer='browser')

    def autoencoder_embedding_analysis(self):
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
        _, data_loader, extra_test_loader = self.load_dataset_and_dataloaders(override_test_fraction=1)
        num_params_dict = self.init_models()

        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         settings=wandb.Settings(code_dir="."))):
            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(num_params_dict.values()))),
                       "Initial Batch Size": self.config.current_batch_size})

            self.models_dict['autoencoder'].eval()
            self.epoch_type = 'test'

            with torch.no_grad():
                for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
                    self.autoencoder_embedding_step(data)

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
        num_params_dict = self.init_models()

        self.config.autoencoder_sigma = self.config.autoencoder.init_sigma
        self.config.autoencoder.molecule_radius_normalization = self.dataDims['max_molecule_radius']  # todo fix

        self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         settings=wandb.Settings(code_dir="."))):
            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(num_params_dict.values()))),
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

    def autoencoder_embedding_step(self, data):
        data = self.preprocess_ae_inputs(data, no_noise=True, orientation_override=None)
        data = data.to(self.device)

        input_cloud = self.fix_autoencoder_protonation(data)

        encoding = self.models_dict['autoencoder'].encode(input_cloud.clone())
        scalar_encoding = self.models_dict['autoencoder'].scalarizer(encoding)
        decoding = self.models_dict['autoencoder'].decode(encoding)

        self.autoencoder_evaluation_sample_analysis(data, decoding, encoding, scalar_encoding)

    def autoencoder_evaluation_sample_analysis(self, data, decoding, encoding, scalar_encoding):
        autoencoder_losses, stats, decoded_data = self.compute_autoencoder_loss(decoding, data.clone())

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

        # Ip, Ipm, I = batch_molecule_principal_axes_torch([data.pos[data.batch == ind] for ind in range(data.num_graphs)])

        scaffold_rmsds, scaffold_max_dists = [], []
        glom_rmsds, glom_max_dists = [], []
        for ind in range(data.num_graphs):  # somewhat slow
            rmsd, max_dist, weight_mean = scaffolded_decoder_clustering(ind, data, decoded_data,
                                                                        self.dataDims['max_molecule_radius'],
                                                                        self.dataDims['num_atom_types'],
                                                                        return_fig=False)
            scaffold_rmsds.append(rmsd)
            scaffold_max_dists.append(max_dist)
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
                        scalar_encoding.cpu().detach().numpy(),
                        scatter(full_overlap / self_overlap, data.batch, reduce='mean').cpu().detach().numpy(),
                        scatter(coord_overlap / self_coord_overlap, data.batch, reduce='mean').cpu().detach().numpy(),
                        scatter(self_type_overlap / type_overlap, data.batch, reduce='mean').cpu().detach().numpy(),
                        # Ip.cpu().detach().numpy(),
                        # Ipm.cpu().detach().numpy(),
                        np.asarray(scaffold_rmsds),
                        np.asarray(scaffold_max_dists),
                        # np.asarray(glom_rmsds),
                        # np.asarray(glom_max_dists),
                        data.smiles
                        ]
        stats_keys = ['encoding',
                      'scalar_encoding',
                      'evaluation_overlap',
                      'evaluation_coord_overlap',
                      'evaluation_type_overlap',
                      # 'principal_inertial_axes',
                      # 'principal_inertial_moments',
                      'scaffold_rmsds',
                      'scaffold_max_dists',
                      # 'glom_rmsds',
                      # 'glom_max_dists',
                      'molecule_smiles'
                      ]
        self.logger.update_stats_dict(self.epoch_type,
                                      stats_keys,
                                      stats_values,
                                      mode='append')

        dict_of_tensors_to_cpu_numpy(stats)

        self.logger.update_stats_dict(self.epoch_type,
                                      stats.keys(),
                                      stats.values(),
                                      mode='append')

    def train_models(self):
        """
        train and/or evaluate one or more models
        regressor
        GAN (generator and/or discriminator)
        autoencoder
        embedding_regressor
        """

        with (wandb.init(config=self.config,
                         project=self.config.wandb.project_name,
                         entity=self.config.wandb.username,
                         tags=[self.config.logger.experiment_tag],
                         settings=wandb.Settings(code_dir="."))):

            self.process_sweep_config()

            '''prep workdir'''
            self.source_directory = os.getcwd()
            self.prep_new_working_directory()

            self.train_models_dict = {
                'discriminator': (self.config.mode == 'gan' or self.config.mode == 'discriminator') and any(
                    (self.config.discriminator.train_adversarially, self.config.discriminator.train_on_distorted,
                     self.config.discriminator.train_on_randn)),
                'generator': (self.config.mode == 'gan') and any((self.config.generator.train_vdw,
                                                                  self.config.generator.train_adversarially,
                                                                  self.config.generator.train_h_bond)),
                'regressor': self.config.mode == 'regression',
                'autoencoder': self.config.mode == 'autoencoder',
                'embedding_regressor': self.config.mode == 'embedding_regression',
                'polymorph_classifier': self.config.mode == 'polymorph_classification',
            }

            '''initialize datasets and useful classes'''
            train_loader, test_loader, extra_test_loader = self.load_dataset_and_dataloaders()
            if self.train_models_dict['discriminator']:
                self.init_gaussian_generator()

            if self.config.mode == 'polymorph_classification':  # to refactor
                train_loader = update_dataloader_batch_size(train_loader, 1)
                test_loader = update_dataloader_batch_size(test_loader, 1)
                if extra_test_loader is not None:
                    extra_test_loader = update_dataloader_batch_size(extra_test_loader, 1)

            num_params_dict = self.init_models()

            '''initialize some training metrics'''
            self.hit_max_lr_dict = {model_name: False for model_name in self.model_names}
            converged, epoch, prev_epoch_failed = self.config.max_epochs == 0, 0, False

            flat_config_dict = flatten_dict(namespace2dict(self.config.__dict__), separator='_')
            for key in flat_config_dict.keys():
                if 'path' in str(type(flat_config_dict[key])).lower():
                    flat_config_dict[key] = str(flat_config_dict[key])
            self.config.__dict__.update(flat_config_dict)

            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            wandb.watch([model for model in self.models_dict.values()], log_graph=True, log_freq=100)
            wandb.log(num_params_dict)
            wandb.log({"All Models Parameters": np.sum(np.asarray(list(num_params_dict.values()))),
                       "Initial Batch Size": self.config.current_batch_size})
            self.logger = Logger(self.config, self.dataDims, wandb, self.model_names)
            self.logger.log_times(self.times)  # log initialization times
            self.times = {}  # reset for iterative looping

            # training loop
            # with torch.autograd.set_detect_anomaly(self.config.anomaly_detection):
            while (epoch < self.config.max_epochs) and not converged:
                print(
                    "⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅")
                print("Starting Epoch {}".format(epoch))  # index from 0
                self.times['full_epoch_start'] = time()
                self.logger.reset_for_new_epoch(epoch, test_loader.batch_size)

                # if epoch % self.config.dataset.refresh_interval:
                #     del train_loader, test_loader
                #     # todo alternate between GEOM and qm9 with some frequency
                #     # todo also keep track of where we are through GEOM and move through it at each iteration
                #     self.new_dataset = geom_msgpack_to_minimal_dataset(
                #         'DRUGS_crude.msgpack', self.config.dataset_path, self.config.dataset.max_dataset_length)
                #     train_loader, test_loader, extra_test_loader = self.load_dataset_and_dataloaders(
                #         override_dataset=self.new_dataset)
                #     del self.new_dataset

                if epoch < self.config.num_early_epochs:
                    early_epochs_step_override = self.config.early_epochs_step_override
                else:
                    early_epochs_step_override = None

                try:  # try this batch size
                    self.run_epoch(epoch_type='train', data_loader=train_loader,
                                   update_weights=True, iteration_override=early_epochs_step_override)

                    with torch.no_grad():
                        self.run_epoch(epoch_type='test', data_loader=test_loader,
                                       update_weights=False, iteration_override=early_epochs_step_override)

                        if (extra_test_loader is not None) and (epoch % self.config.extra_test_period == 0) and (
                                epoch > 0):
                            self.run_epoch(epoch_type='extra', data_loader=extra_test_loader,
                                           update_weights=False, iteration_override=None)  # compute loss on test set

                    '''save checkpoints'''
                    if self.config.save_checkpoints and epoch > 0:
                        self.model_checkpointing(epoch)

                    '''update learning rates'''
                    self.update_lr()

                    '''check convergence status and record metrics & analysis'''
                    self.times['reporting_start'] = time()
                    self.logger.numpyize_current_losses()
                    self.logger.update_loss_record()
                    self.logger.log_training_metrics()
                    self.logger.log_epoch_analysis(test_loader)
                    self.logger.check_model_convergence()
                    self.times['reporting_end'] = time()

                    if all(list(self.logger.converged_flags.values())):  # todo confirm this works
                        print('Training has converged!')
                        break

                    '''increment batch size'''
                    train_loader, test_loader, extra_test_loader = \
                        self.increment_batch_size(train_loader, test_loader, extra_test_loader)

                    prev_epoch_failed = False

                except RuntimeError as e:  # if we do hit OOM, slash the batch size
                    if "CUDA out of memory" in str(
                            e) or "nonzero is not supported for tensors with more than INT_MAX elements" in str(e):
                        if prev_epoch_failed:
                            gc.collect()  # TODO not clear to me that this is effective

                        train_loader, test_loader = slash_batch(train_loader, test_loader,
                                                                slash_fraction=0.25)  # shrink batch size
                        wandb.log({'batch size': train_loader.batch_size})

                        torch.cuda.empty_cache()
                        self.config.grow_batch_size = False  # stop growing the batch for the rest of the run
                        prev_epoch_failed = True
                    else:
                        raise e  # will simply raise error if training on CPU

                self.times['full_epoch_end'] = time()
                self.logger.log_times(self.times)
                self.times = {}
                epoch += 1

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

    # def post_run_evaluation(self, epoch, test_loader, extra_test_loader):
    #     if self.config.mode == 'gan':  # evaluation on test metrics
    #         self.gan_evaluation(epoch, test_loader, extra_test_loader)

    def run_epoch(self, epoch_type: str, data_loader=None, update_weights=True, iteration_override=None):
        self.epoch_type = epoch_type
        self.times[epoch_type + "_epoch_start"] = time()
        if self.config.mode == 'gan' or self.config.mode == 'discriminator':
            if self.config.model_paths.regressor is not None:
                self.models_dict['regressor'].eval()  # using this to suggest densities to the generator

            if self.train_models_dict['discriminator'] or self.train_models_dict['generator']:
                self.gan_epoch(data_loader, update_weights, iteration_override)

        elif self.config.mode == 'regression':
            self.regression_epoch(data_loader, update_weights, iteration_override)

        elif self.config.mode == 'autoencoder':
            self.autoencoder_epoch(data_loader, update_weights, iteration_override)

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
                    break  # stop training early - for debugging purposes

        # post epoch processing
        self.logger.concatenate_stats_dict(self.epoch_type)

    def embedding_regression_step(self, data, update_weights):
        data = self.preprocess_ae_inputs(data, no_noise=True, orientation_override='random')
        data = data.to(self.device)
        v_embedding = self.models_dict['autoencoder'].encode(data)
        s_embedding = self.models_dict['autoencoder'].scalarizer(v_embedding)
        # if self.config.embedding_regressor.prediction_type == 'vector':  # this is quite fast
        #     data.y = batch_compute_dipole(data.pos, data.batch, data.x[:, 0], self.electronegativity_tensor)
        #

        predictions = self.models_dict['embedding_regressor'](s_embedding, v_embedding)[:, 0]
        losses = F.smooth_l1_loss(predictions, data.y, reduction='none')
        predictions = predictions.cpu().detach().numpy() * self.dataDims['target_std'] + self.dataDims['target_mean']
        targets = data.y.cpu().detach().numpy() * self.dataDims['target_std'] + self.dataDims['target_mean']

        if self.config.EmbeddingRegressor.prediction_type == 'vector':  # this is quite fast
            predictions = np.linalg.norm(predictions, axis=-1)
            targets = np.linalg.norm(targets, axis=-1)
        regression_loss = losses.mean()
        if update_weights:
            self.optimizers_dict['embedding_regressor'].zero_grad(
                set_to_none=True)  # reset gradients from previous passes
            regression_loss.backward()  # back-propagation
            self.optimizers_dict['embedding_regressor'].step()  # update parameters
        '''log losses and other tracking values'''
        self.logger.update_current_losses('embedding_regressor', self.epoch_type,
                                          regression_loss.cpu().detach().numpy(),
                                          losses.cpu().detach().numpy())
        stats_values = [predictions, targets]

        self.logger.update_stats_dict(self.epoch_type, ['regressor_prediction', 'regressor_target'], stats_values,
                                      mode='extend')

    #
    # def generate_random_point_cloud_batch(self, batch_size):  # TODO DEPRECATE
    #     """
    #     generates a uniform random point cloud with maximum radius 1
    #     """
    #
    #     point_num_rands = np.random.randint(low=2,  # self.config.autoencoder.min_num_atoms,
    #                                         high=self.config.autoencoder.max_num_atoms + 1,
    #                                         size=batch_size)
    #
    #     # truly random point clouds within a sphere of fixed maximum radius of 1
    #     vectors = torch.rand(point_num_rands.sum(), 3, dtype=torch.float32, device=self.config.device)
    #     norms = torch.linalg.norm(vectors, dim=1)[:, None]
    #     lengths = torch.rand(point_num_rands.sum(), 1, dtype=torch.float32, device=self.config.device)
    #     coords_list = (vectors / norms * lengths).split(point_num_rands.tolist())
    #
    #     # coords_list = [coords - coords.mean(0) for coords in coords_list]
    #     types_list = torch.randint(self.dataDims['num_atom_types'],
    #                                size=(point_num_rands.sum(),),
    #                                device=self.config.device).split(point_num_rands.tolist())
    #
    #     data = self.collater([CrystalData(
    #         x=types_list[n][:, None],
    #         pos=coords_list[n],
    #         mol_size=torch.tensor(point_num_rands[n], dtype=torch.long, device=self.config.device),
    #     ) for n in range(batch_size)])
    #
    #     return data

    def autoencoder_step(self, input_data, data, update_weights, step, last_step=False):
        if self.config.dataset.filter_protons and not self.models_dict['autoencoder'].inferring_protons:
            data = input_data.clone()  # deprotonate the reference if we are not inferring protons
        decoding, encoding = self.models_dict['autoencoder'](input_data, return_encoding=True)
        autoencoder_losses, stats, decoded_data = self.compute_autoencoder_loss(decoding, data.clone())

        autoencoder_loss = autoencoder_losses.mean()
        if update_weights:
            self.optimizers_dict['autoencoder'].zero_grad(set_to_none=True)  # reset gradients from previous passes
            autoencoder_loss.backward()  # back-propagation
            torch.nn.utils.clip_grad_norm_(self.models_dict['autoencoder'].parameters(),
                                           self.config.gradient_norm_clip)  # gradient clipping by norm
            self.optimizers_dict['autoencoder'].step()  # update parameters

        self.autoencoder_stats_and_reporting(data, decoded_data, encoding, last_step, stats, step)

    def fix_autoencoder_protonation(self, data, override_deprotonate=False):
        if self.config.autoencoder.infer_protons or self.config.dataset.filter_protons or override_deprotonate:
            heavy_atom_inds = torch.argwhere(data.x != 1)[:, 0]  # protons are atom type 1
            input_cloud = data.clone()
            input_cloud.x = input_cloud.x[heavy_atom_inds]
            input_cloud.pos = input_cloud.pos[heavy_atom_inds]
            input_cloud.batch = input_cloud.batch[heavy_atom_inds]
            a, b = torch.unique(input_cloud.batch, return_counts=True)
            input_cloud.ptr = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(b, dim=0)]).long()
            input_cloud.num_atoms = torch.diff(input_cloud.ptr).long()
        else:
            input_cloud = data.clone()

        return input_cloud

    def autoencoder_stats_and_reporting(self, data, decoded_data, encoding, last_step, stats, step):
        if self.logger.epoch % self.config.logger.sample_reporting_frequency == 0:
            stats['encoding'] = encoding.cpu().detach().numpy()

        if last_step:
            self.detailed_autoencoder_step_analysis(data, decoded_data, stats)

        dict_of_tensors_to_cpu_numpy(stats)

        self.logger.update_stats_dict(self.epoch_type,
                                      stats.keys(),
                                      stats.values(),
                                      mode='append')

    def detailed_autoencoder_step_analysis(self, data, decoded_data, stats):
        '''equivariance checks'''
        encoder_equivariance_loss, decoder_equivariance_loss = self.equivariance_test(data.clone())
        stats['encoder_equivariance_loss'] = encoder_equivariance_loss.mean().detach()
        stats['decoder_equivariance_loss'] = decoder_equivariance_loss.mean().detach()
        # do evaluation on current sample and save this as our loss for tracking purposes
        nodewise_weights_tensor = decoded_data.aux_ind
        true_nodes = F.one_hot(self.models_dict['autoencoder'].atom_embedding_vector[data.x[:, 0].long()],
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
        data.pos *= self.models_dict['autoencoder'].radial_normalization
        decoded_data.pos *= self.models_dict['autoencoder'].radial_normalization
        self.logger.update_stats_dict(self.epoch_type,
                                      ['sample', 'decoded_sample'],
                                      [data.cpu().detach(), decoded_data.cpu().detach()
                                       ], mode='append')

    def equivariance_test(self, data):
        rotations = torch.tensor(
            R.random(data.num_graphs).as_matrix() *
            np.random.choice((-1, 1), replace=True, size=data.num_graphs)[:, None, None],
            dtype=torch.float,
            device=data.x.device)

        encoder_equivariance_loss, encoding, rotated_encoding = self.test_encoder_equivariance(data, rotations)

        decoder_equivariance_loss = self.test_decoder_equivariance(data, encoding, rotated_encoding, rotations)

        return encoder_equivariance_loss, decoder_equivariance_loss

    def test_decoder_equivariance(self, data, encoding, rotated_encoding, rotations):
        """
        check decoder end-to-end equivariance
        """
        '''take a given embedding and decoded it'''
        decoding = self.models_dict['autoencoder'].decode(encoding)
        '''rotate embedding and decode'''
        decoding2 = self.models_dict['autoencoder'].decode(
            rotated_encoding.reshape(data.num_graphs, 3, encoding.shape[-1]))
        '''rotate first decoding and compare'''
        decoded_batch = torch.arange(data.num_graphs).repeat_interleave(
            self.models_dict['autoencoder'].num_decoder_nodes).to(self.config.device)
        rotated_decoding_positions = torch.cat(
            [torch.einsum('ij, kj->ki', rotations[ind], decoding[:, :3][decoded_batch == ind])
             for ind in range(data.num_graphs)])
        rotated_decoding = decoding.clone()
        rotated_decoding[:, :3] = rotated_decoding_positions
        # first three dimensions should be equivariant and all trailing invariant
        decoder_equivariance_loss = (
                torch.abs(rotated_decoding[:, :3] - decoding2[:, :3]) / torch.abs(rotated_decoding[:, :3])).mean(-1)
        return decoder_equivariance_loss

    def test_encoder_equivariance(self, data, rotations):
        """
        check encoder end-to-end equivariance
        """
        '''embed the input data then rotate the embedding'''
        encoding = self.models_dict['autoencoder'].encode(data.clone())
        rotated_encoding = torch.einsum('nij, njk->nik',
                                        rotations,
                                        encoding
                                        )  # rotate in 3D

        rotated_encoding = rotated_encoding.reshape(data.num_graphs, rotated_encoding.shape[-1] * 3)
        '''rotate the input data and embed it'''
        data.pos = torch.cat([torch.einsum('ij, kj->ki', rotations[ind], data.pos[data.batch == ind])
                              for ind in range(data.num_graphs)])
        encoding2 = self.models_dict['autoencoder'].encode(data.clone())
        encoding2 = encoding2.reshape(data.num_graphs, encoding2.shape[-1] * 3)
        '''compare the embeddings - should be identical for an equivariant embedding'''
        encoder_equivariance_loss = (torch.abs(rotated_encoding - encoding2) / torch.abs(rotated_encoding)).mean(-1)
        return encoder_equivariance_loss, encoding, rotated_encoding

    def autoencoder_epoch(self, data_loader, update_weights, iteration_override=None):
        if update_weights:
            self.models_dict['autoencoder'].train(True)
        else:
            self.models_dict['autoencoder'].eval()

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            data = data.to(self.device)
            data.pos = data.pos / self.models_dict['autoencoder'].radial_normalization

            data, input_data = self.preprocess_ae_inputs(data, no_noise=self.epoch_type == 'test')
            self.autoencoder_step(input_data, data, update_weights, step=i, last_step=i == len(data_loader) - 1)

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        self.logger.concatenate_stats_dict(self.epoch_type)
        self.autoencoder_annealing()

    def autoencoder_annealing(self):
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
                data.pos += torch.randn_like(data.pos) * self.config.positional_noise.autoencoder
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
        input_data.pos -= torch.repeat_interleave(centroids, input_data.num_atoms, dim=0, output_size=input_data.num_nodes)

        return data, input_data

    def compute_autoencoder_loss(self, decoding, data):
        """
        Function for analyzing autoencoder and calculating loss & other key metrics
        1) process inputs and outputs into the correct format
        2) compute relevant losses, reconstruction, radial constraint, weight constraint
        Parameters
        ----------
        decoding
        data

        Returns
        -------

        """
        decoded_data, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
            self.prep_ae_io_for_analysis(data, decoding))

        nodewise_reconstruction_loss, nodewise_type_loss, reconstruction_loss, self_likelihoods = (
            self.get_reconstruction_loss(data, decoded_data, nodewise_weights))
        matching_nodes_fraction = torch.sum(nodewise_reconstruction_loss < 0.01) / data.num_nodes  # within 1% matching

        # node radius constraining loss
        decoded_dists = torch.linalg.norm(decoded_data.pos, dim=1)
        constraining_loss = scatter(
            F.relu(
                decoded_dists - 1), #torch.repeat_interleave(data.radius, self.models_dict['autoencoder'].num_decoder_nodes,
                                   #                     dim=0)),
            decoded_data.batch, reduce='mean')

        # node weight constraining loss
        node_weight_constraining_loss = scatter(
            F.relu(-torch.log10(nodewise_weights_tensor / torch.amin(nodewise_graph_weights)) - 2),
            decoded_data.batch)  # don't let these get too small
        losses = reconstruction_loss + constraining_loss + node_weight_constraining_loss

        stats = {'constraining_loss': constraining_loss.mean().detach(),
                 'reconstruction_loss': reconstruction_loss.mean().detach(),
                 'nodewise_type_loss': nodewise_type_loss.detach(),
                 'scaled_reconstruction_loss': (reconstruction_loss.mean() * self.config.autoencoder_sigma).detach(),
                 'sigma': self.config.autoencoder_sigma,
                 'mean_self_overlap': scatter(self_likelihoods, data.batch, reduce='mean').mean().detach(),
                 'matching_nodes_fraction': matching_nodes_fraction.detach(),
                 'matching_nodes_loss': 1 - matching_nodes_fraction.detach(),
                 'node_weight_constraining_loss': node_weight_constraining_loss.mean().detach(),
                 }

        assert torch.sum(torch.isnan(losses)) == 0, "NaN in Reconstruction Loss"

        return losses, stats, decoded_data

    def prep_ae_io_for_analysis(self, data, decoding):
        # reduce to relevant atom types
        data.x = self.models_dict['autoencoder'].atom_embedding_vector[data.x]
        # generate input reconstructed as a data type
        decoded_data = self.decoding2data(data, decoding)
        # compute the distributional weight of each node
        nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = \
            get_node_weights(data, decoded_data, decoding,
                             self.models_dict['autoencoder'].num_decoder_nodes,
                             self.config.autoencoder.node_weight_temperature)
        decoded_data.aux_ind = nodewise_weights_tensor
        # input node weights are always 1 - corresponding each to an atom
        data.aux_ind = torch.ones(data.num_nodes, dtype=torch.float32, device=self.config.device)
        # get probability distribution over type dimensions
        decoded_data.x = F.softmax(decoding[:, 3:-1], dim=1)
        return decoded_data, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor

    def decoding2data(self, data, decoding):
        decoded_data = data.clone()
        decoded_data.pos = decoding[:, :3]
        decoded_data.batch = torch.arange(data.num_graphs).repeat_interleave(
            self.models_dict['autoencoder'].num_decoder_nodes).to(self.config.device)
        return decoded_data

    def get_reconstruction_loss(self, data, decoded_data, nodewise_weights):
        true_node_one_hot = F.one_hot(data.x[:, 0].long(), num_classes=self.dataDims['num_atom_types']).float()

        decoder_likelihoods = compute_gaussian_overlap(true_node_one_hot, data, decoded_data,
                                                       self.config.autoencoder_sigma,
                                                       nodewise_weights=decoded_data.aux_ind,
                                                       overlap_type='gaussian', log_scale=False,
                                                       type_distance_scaling=self.config.autoencoder.type_distance_scaling)
        self_likelihoods = compute_gaussian_overlap(true_node_one_hot, data, data, self.config.autoencoder_sigma,
                                                    nodewise_weights=data.aux_ind,
                                                    overlap_type='gaussian', log_scale=False,
                                                    type_distance_scaling=self.config.autoencoder.type_distance_scaling,
                                                    dist_to_self=True)  # if sigma is too large, these can be > 1, so we map to the overlap of the true density with itself

        # typewise agreement for whole graph
        per_graph_true_types = scatter(
            true_node_one_hot, data.batch[:, None], dim=0, reduce='mean')
        per_graph_pred_types = scatter(
            decoded_data.x * nodewise_weights[:, None], decoded_data.batch[:, None], dim=0, reduce='sum')

        nodewise_type_loss = (F.binary_cross_entropy(per_graph_pred_types, per_graph_true_types) -
                              F.binary_cross_entropy(per_graph_true_types, per_graph_true_types))

        nodewise_reconstruction_loss = F.smooth_l1_loss(decoder_likelihoods, self_likelihoods, reduction='none')
        graph_reconstruction_loss = scatter(nodewise_reconstruction_loss, data.batch, reduce='mean')

        return nodewise_reconstruction_loss, nodewise_type_loss, graph_reconstruction_loss, self_likelihoods

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
                    break  # stop training early - for debugging purposes

        self.logger.concatenate_stats_dict(self.epoch_type)

    def polymorph_classification_epoch(self, data_loader, update_weights=True, iteration_override=None):
        if update_weights:
            self.models_dict['polymorph_classifier'].train(True)
        else:
            self.models_dict['polymorph_classifier'].eval()

        stats_keys = ['true_labels', 'probs', 'periodic']

        for i, data in enumerate(tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            if self.config.positional_noise.polymorph_classifier > 0:
                data.pos += torch.randn_like(data.pos) * self.config.positional_noise.polymorph_classifier

            data = data.to(self.device)
            output, extra_outputs = self.models_dict['polymorph_classifier'](data, return_latent=True,
                                                                             return_embedding=True)
            loss = (F.cross_entropy(output[:, :self.dataDims['num_polymorphs']], data.mol_y, reduction='none'))  # +
            #F.cross_entropy(output[:, self.dataDims['num_polymorphs']:], data.defect))

            if update_weights:
                loss.mean().backward()
                if i % data_loader.batch_size == 0 or i == len(
                        data_loader) - 1:  # use gradient accumulation for synthetically larger batch sizes
                    self.optimizers_dict['polymorph_classifier'].step()  # update parameters
                    self.optimizers_dict['polymorph_classifier'].zero_grad(
                        set_to_none=True)  # reset gradients from previous passes

            '''log losses and other tracking values'''
            self.logger.update_current_losses('polymorph_classifier', self.epoch_type,
                                              loss.mean().cpu().detach().numpy(),
                                              loss.cpu().detach().numpy())

            stats_values = [data.mol_y.detach(), output.detach(), data.periodic]
            stats = {key: value for key, value in zip(stats_keys, stats_values)}
            dict_of_tensors_to_cpu_numpy(stats)
            self.logger.update_stats_dict(self.epoch_type,
                                          stats.keys(),
                                          stats.values(),
                                          mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        self.logger.concatenate_stats_dict(self.epoch_type)

    def gan_epoch(self, data_loader=None, update_weights=True,
                  iteration_override=None):

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
            train_generator
            '''
            self.generator_step(data, i, update_weights)
            '''
            record some stats
            '''
            self.logger.update_stats_dict(self.epoch_type, ['identifiers'], data.identifier, mode='extend')
            self.logger.update_stats_dict(self.epoch_type, ['smiles'], data.smiles, mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        self.logger.concatenate_stats_dict(self.epoch_type)

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

        classification_target = torch.cat((torch.ones_like(discriminator_output_on_real[:, 0]),
                                           torch.zeros_like(discriminator_output_on_fake[:, 0])))

        classification_losses = F.cross_entropy(combined_outputs[:, :2], classification_target.long(), reduction='none')

        if real_fake_rdf_distances is not None:
            rdf_distance_target = torch.log10(1 + torch.cat((torch.zeros_like(discriminator_output_on_real[:, 0]),
                                                             real_fake_rdf_distances)))  # rescale on log(1+x)

            rdf_distance_losses = F.smooth_l1_loss(combined_outputs[:, 2], rdf_distance_target,
                                                   reduction='none') * 10  # rescale w.r.t., classification loss

        else:
            rdf_distance_losses = torch.zeros_like(classification_losses)

        score_on_real = softmax_and_score(discriminator_output_on_real[:, :2])
        score_on_fake = softmax_and_score(discriminator_output_on_fake[:, :2])

        stats = {'discriminator_real_score': score_on_real.detach(),
                 'discriminator_fake_score': score_on_fake.detach(),
                 'discriminator_fake_true_distance': torch.log10(1 + real_fake_rdf_distances).detach(),
                 'discriminator_fake_predicted_distance': discriminator_output_on_fake[:, 2].detach(),
                 'discriminator_real_true_distance': torch.zeros_like(discriminator_output_on_real[:, 0]).detach(),
                 'discriminator_real_predicted_distance': discriminator_output_on_real[:, 2].detach(),
                 'discriminator_classification_loss': classification_losses.detach(),
                 'discriminator_distance_loss': rdf_distance_losses.detach()}

        discriminator_losses_list = []
        if self.config.discriminator.use_classification_loss:
            discriminator_losses_list.append(classification_losses)

        if self.config.discriminator.use_rdf_distance_loss:
            discriminator_losses_list.append(rdf_distance_losses)

        discriminator_losses = torch.sum(torch.stack(discriminator_losses_list), dim=0)

        return discriminator_losses, stats

    def generator_step(self, data, i, update_weights):
        """
        execute a complete training step for the generator
        get sample losses, do reporting, update gradients
        """
        if self.train_models_dict['generator']:
            discriminator_raw_output, generated_samples, raw_samples, packing_loss, auv_prediction, auv_target, \
                vdw_loss, vdw_score, generated_dist_dict, supercell_examples, similarity_penalty, h_bond_score = \
                self.get_generator_losses(data)

            generator_losses, losses_stats = self.aggregate_generator_losses(
                packing_loss, discriminator_raw_output, vdw_loss, vdw_score,
                similarity_penalty, auv_prediction, auv_target, h_bond_score)

            generator_loss = generator_losses.mean()

            if update_weights:
                self.optimizers_dict['generator'].zero_grad(set_to_none=True)  # reset gradients from previous passes
                generator_loss.backward()  # back-propagation
                torch.nn.utils.clip_grad_norm_(self.models_dict['generator'].parameters(),
                                               self.config.gradient_norm_clip)  # gradient clipping
                self.optimizers_dict['generator'].step()  # update parameters

            self.logger.update_current_losses('generator', self.epoch_type,
                                              generator_loss.data.cpu().detach().numpy(),
                                              generator_losses.cpu().detach().numpy())

            stats = {
                'final_generated_cell_parameters': supercell_examples.cell_params.cpu().detach().numpy(),
                'generated_space_group_numbers': supercell_examples.sg_ind.cpu().detach().numpy(),
                'raw_generated_cell_parameters': raw_samples
            }
            stats.update(losses_stats)

            dict_of_tensors_to_cpu_numpy(stats)

            self.logger.update_stats_dict(self.epoch_type,
                                          stats.keys(),
                                          stats.values(),
                                          mode='extend')

            del supercell_examples, stats

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
        discriminator_output_on_real, real_pairwise_distances_dict, real_latent = self.adversarial_score(
            real_supercell_data, return_latent=True)
        discriminator_output_on_fake, fake_pairwise_distances_dict, fake_latent = self.adversarial_score(
            fake_supercell_data, return_latent=True)

        '''recompute packing coeffs'''  # todo get rid of these - we're not using mol_volume anymore

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
                                              rrange=[0, self.config.discriminator.model.graph.cutoff],
                                              bins=2000, raw_density=True, elementwise=True, mode='intermolecular',
                                              cpu_detach=False)
            fake_rdf, _, _ = new_crystal_rdf(fake_supercell_data, fake_pairwise_distances_dict,
                                             rrange=[0, self.config.discriminator.model.graph.cutoff],
                                             bins=2000, raw_density=True, elementwise=True, mode='intermolecular',
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
                 'generated_cell_parameters': generated_samples_i.detach(),
                 'final_generated_cell_parameters': canonical_fake_cell_params.detach(),
                 'real_volume_fractions': real_volume_fractions.detach(),
                 'generated_volume_fractions': fake_volume_fractions.detach(),
                 'real_volume_fractions': real_volume_fractions.detach(),
                 'fake_volume_fractions': fake_volume_fractions.detach()}

        stats.update(negatives_stats)

        return (discriminator_output_on_real, discriminator_output_on_fake,
                rdf_dists, stats)

    def get_generator_samples(self, data, alignment_override=None):
        """
        set conformer orientation, optionally add noise, set the space group & symmetry information
        optionally get the predicted density from a regression model
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
            mol_data = update_crystal_symmetry_elements(mol_data, self.config.generate_sgs, self.sym_info,
                                                        randomize_sgs=True)

        # update packing coefficient
        if self.config.model_paths.regressor is not None:  # todo ensure we have a regressor predicting the right thing here - i.e., cell_volume vs packing coeff
            # predict the crystal density and feed it as an input to the generator
            with torch.no_grad():
                standardized_target_aunit_volume = self.models_dict['regressor'](
                    mol_data.clone().detach().to(self.config.device)).detach()[:, 0]
        else:
            target_aunit_volume = mol_data.tracking[:, self.t_i_d[
                                                           'crystal_reduced_volume']]  # todo std dict and t_i_d are deprecated in current version
            standardized_target_aunit_volume = (
                    (target_aunit_volume - self.std_dict['crystal_reduced_volume'][0]) /
                    self.std_dict['crystal_reduced_volume'][1]).to(self.config.device)

        standardized_target_aunit_volume += torch.randn_like(
            standardized_target_aunit_volume) * self.config.generator.packing_target_noise

        # generate the samples
        [generated_samples, prior, condition, raw_samples] = self.models_dict['generator'].forward()

        return generated_samples, prior, standardized_target_aunit_volume, mol_data, raw_samples

    def get_generator_losses(self, data):
        """
        generate samples & score them
        """

        """get crystals"""
        generated_samples, prior, standardized_target_auv, generator_data, raw_samples = (
            self.get_generator_samples(data))

        supercell_data, generated_cell_volumes = (
            self.supercell_builder.build_zp1_supercells(
                generator_data, generated_samples, self.config.supercell_size,
                self.config.discriminator.model.graph.cutoff,
                align_to_standardized_orientation=False
            ))

        """get losses"""
        similarity_penalty = self.compute_similarity_penalty(generated_samples, prior, raw_samples)
        discriminator_raw_output, dist_dict = self.score_adversarially(supercell_data)
        h_bond_score = torch.ones_like(similarity_penalty)
        vdw_loss, vdw_score, _, _, _ = vdw_overlap(self.vdw_radii,
                                                   dist_dict=dist_dict,
                                                   num_graphs=generator_data.num_graphs,
                                                   graph_sizes=generator_data.num_atoms,
                                                   loss_func=self.config.generator.vdw_loss_func)
        packing_loss, auv_prediction, auv_target, auv_csd = \
            self.generator_density_matching_loss(
                standardized_target_auv, supercell_data, generated_samples,
                precomputed_volumes=generated_cell_volumes, loss_func=self.config.generator.density_loss_func)

        return discriminator_raw_output, generated_samples.detach(), raw_samples.detach(), \
            packing_loss, auv_prediction.detach(), \
            auv_target.detach(), \
            vdw_loss, vdw_score, dist_dict, \
            supercell_data, similarity_penalty, h_bond_score

    def init_gaussian_generator(self):
        """
        dataset_builder: for going from database to trainable dataset
        dataDims: contains key information about the dataset
        number of generators for discriminator training
        supercell_builder
        tracking indices for certain properties
        symmetry element indexing
        multivariate gaussian generator
        """
        ''' 
        init gaussian generator for cell parameter sampling
        we don't always use it but it's very cheap so just do it every time
        '''
        self.gaussian_generator = IndependentGaussianGenerator(input_dim=12,
                                                               means=self.dataDims['lattice_means'],
                                                               stds=self.dataDims['lattice_stds'],
                                                               sym_info=self.sym_info,
                                                               device=self.config.device,
                                                               cov_mat=self.dataDims['lattice_cov_mat'])

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
                generated_samples, _, _, generator_data, _ = self.get_generator_samples(real_data,
                                                                                        alignment_override=orientation)

            stats = {'generator_sample_source': np.zeros(len(generated_samples))}

        elif (self.config.discriminator.train_on_randn or override_randn) and (generator_ind == 2):
            generator_data = set_molecule_alignment(real_data.clone(), mode=orientation)
            negative_type = 'randn'
            generated_samples = self.gaussian_generator.forward(real_data.num_graphs, real_data).to(self.config.device)

            stats = {'generator_sample_source': np.ones(len(generated_samples))}

        elif (self.config.discriminator.train_on_distorted or override_distorted) and (generator_ind == 3):
            generator_data = set_molecule_alignment(real_data.clone(),
                                                    mode='as is')  # will be standardized anyway in cell builder
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
                train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size + increment)
                test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size + increment)
                if extra_test_loader is not None:
                    extra_test_loader = update_dataloader_batch_size(extra_test_loader,
                                                                     extra_test_loader.batch_size + increment)
                print(f'Batch size incremented to {train_loader.batch_size}')
        wandb.log({'batch size': train_loader.batch_size})
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
                if np.average(self.logger.current_losses[model_name][f'mean_{loss_type_check}']) == np.amin(
                        past_mean_losses):  # save if this is the best
                    print(f"Saving {model_name} checkpoint")
                    self.logger.save_stats_dict(prefix=f'best_{model_name}_')
                    save_checkpoint(epoch,
                                    self.models_dict[model_name],
                                    self.optimizers_dict[model_name],
                                    self.config.__dict__[model_name].__dict__,
                                    self.config.checkpoint_dir_path + f'best_{model_name}' + self.run_identifier,
                                    self.dataDims)
        self.times['checkpointing_end'] = time()

    def update_lr(self):
        for model_name in self.model_names:
            if self.config.__dict__[model_name].optimizer is not None:
                self.optimizers_dict[model_name], learning_rate = set_lr(
                    self.schedulers_dict[model_name],
                    self.optimizers_dict[model_name],
                    self.config.__dict__[model_name].optimizer,
                    self.logger.current_losses[model_name]['mean_train'],
                    self.hit_max_lr_dict[model_name])

                if learning_rate >= self.config.__dict__[model_name].optimizer.max_lr:
                    self.hit_max_lr_dict[model_name] = True

                self.logger.learning_rates[model_name] = learning_rate

    def reload_best_test_checkpoint(self, epoch):
        # reload best test  # todo rewrite to be more general
        if epoch != 0:  # if we have trained at all, reload the best model
            # generator_path = f'../models/best_generator_{self.run_identifier}'
            discriminator_path = f'../models/best_discriminator_{self.run_identifier}'

            # if os.path.exists(generator_path):
            #     generator_checkpoint = torch.load(generator_path)
            #     if list(generator_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
            #         for i in list(generator_checkpoint['model_state_dict']):
            #             generator_checkpoint['model_state_dict'][i[7:]] = generator_checkpoint['model_state_dict'].pop(i)
            #     self.models_dict['generator'].load_state_dict(generator_checkpoint['model_state_dict'])

            if os.path.exists(discriminator_path):
                discriminator_checkpoint = torch.load(discriminator_path)
                if list(discriminator_checkpoint['model_state_dict'])[0][
                   0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                    for i in list(discriminator_checkpoint['model_state_dict']):
                        discriminator_checkpoint['model_state_dict'][i[7:]] = discriminator_checkpoint[
                            'model_state_dict'].pop(i)
                self.models_dict['discriminator'].load_state_dict(discriminator_checkpoint['model_state_dict'])

    # def gan_evaluation(self, epoch, test_loader, extra_test_loader):
    #     """
    #     run post-training evaluation
    #     """
    #     self.reload_best_test_checkpoint(epoch)
    #     self.logger.reset_for_new_epoch(epoch, test_loader.batch_size)
    #
    #     # rerun test inference
    #     # self.models_dict['generator'].eval()
    #     self.models_dict['discriminator'].eval()
    #     with torch.no_grad():
    #         if self.train_models_dict['discriminator']:
    #             self.run_epoch(epoch_type='test', data_loader=test_loader, update_weights=False)  # compute loss on test set
    #
    #             if extra_test_loader is not None:
    #                 self.run_epoch(epoch_type='extra', data_loader=extra_test_loader, update_weights=False)  # compute loss on test set
    #
    #         # sometimes test the generator on a mini CSP problem
    #         if (self.config.mode == 'gan') and self.train_models_dict['generator']:
    #             self.batch_csp(extra_test_loader if extra_test_loader is not None else test_loader)
    #
    #     self.logger.log_epoch_analysis(test_loader)
    #
    #     return None

    def compute_similarity_penalty(self, generated_samples, prior, raw_samples):
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

    def aggregate_generator_losses(self, packing_loss, discriminator_raw_output, vdw_loss, vdw_score,
                                   similarity_penalty, auv_prediction, auv_target, h_bond_score):
        generator_losses_list = []
        stats_keys, stats_values = [], []
        if packing_loss is not None:
            packing_mae = torch.abs(auv_prediction - auv_target) / auv_target

            if packing_mae.mean() < (
                    0.02 + self.config.generator.packing_target_noise):  # dynamically soften the packing loss when the model is doing well
                self.packing_loss_coefficient *= 0.99
            if (packing_mae.mean() > (0.03 + self.config.generator.packing_target_noise)) and (
                    self.packing_loss_coefficient < 100):
                self.packing_loss_coefficient *= 1.01

            self.logger.packing_loss_coefficient = self.packing_loss_coefficient

            stats_keys += ['generator_packing_loss', 'generator_packing_prediction',
                           'generator_packing_target', 'generator_packing_mae']
            stats_values += [packing_loss.detach() * self.packing_loss_coefficient,
                             auv_prediction.detach(),
                             auv_target.detach(),
                             packing_mae.detach()]

            if True:  # enforce the target density all the time
                generator_losses_list.append(packing_loss.float() * self.packing_loss_coefficient)

        if discriminator_raw_output is not None:
            if self.config.generator.adversarial_loss_func == 'hot softmax':
                adversarial_loss = 1 - F.softmax(discriminator_raw_output / 5, dim=1)[:,
                                       1]  # high temp smears out the function over a wider range
                adversarial_score = softmax_and_score(discriminator_raw_output)

            elif self.config.generator.adversarial_loss_func == 'minimax':
                softmax_adversarial_score = F.softmax(discriminator_raw_output, dim=1)[:, 1]  # modified minimax
                adversarial_loss = -torch.log(softmax_adversarial_score)  # modified minimax
                adversarial_score = softmax_and_score(discriminator_raw_output)

            elif self.config.generator.adversarial_loss_func == 'score':
                adversarial_loss = -softmax_and_score(discriminator_raw_output)  # linearized score
                adversarial_score = softmax_and_score(discriminator_raw_output)

            elif self.config.generator.adversarial_loss_func == 'softmax':
                adversarial_loss = 1 - F.softmax(discriminator_raw_output, dim=1)[:, 1]
                adversarial_score = softmax_and_score(discriminator_raw_output)

            else:
                print(f'{self.config.generator.adversarial_loss_func} is not an implemented adversarial loss')
                sys.exit()

            stats_keys += ['generator_adversarial_loss']
            stats_values += [adversarial_loss.detach()]
            stats_keys += ['generator_adversarial_score']
            stats_values += [adversarial_score.detach()]

            if self.config.generator.train_adversarially:
                generator_losses_list.append(adversarial_loss)

        if vdw_loss is not None:
            stats_keys += ['generator_per_mol_vdw_loss', 'generator_per_mol_vdw_score']
            stats_values += [vdw_loss.detach()]
            stats_values += [vdw_score.detach()]

            if self.config.generator.train_vdw:
                generator_losses_list.append(vdw_loss)

        if h_bond_score is not None:
            if self.config.generator.train_h_bond:
                generator_losses_list.append(h_bond_score)

            stats_keys += ['generator h bond loss']
            stats_values += [h_bond_score.detach()]

        if similarity_penalty is not None:
            stats_keys += ['generator similarity loss']
            stats_values += [similarity_penalty.detach()]

            if self.config.generator.similarity_penalty != 0:
                if similarity_penalty is not None:
                    generator_losses_list.append(self.config.generator.similarity_penalty * similarity_penalty)
                else:
                    print('similarity penalty was none')

        generator_losses = torch.sum(torch.stack(generator_losses_list), dim=0)

        return generator_losses, {key: value for key, value in zip(stats_keys, stats_values)}

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

    def crystal_search(self, molecule_data, batch_size=None, data_contains_ground_truth=True):  # currently deprecated
        """
        execute a search for a single crystal target
        if the target is known, compare it to our best guesses
        """
        self.source_directory = os.getcwd()
        self.prep_new_working_directory()

        with wandb.init(config=self.config,
                        project=self.config.wandb.project_name,
                        entity=self.config.wandb.username,
                        tags=[self.config.logger.experiment_tag],
                        settings=wandb.Settings(code_dir=".")):

            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name

            if batch_size is None:
                batch_size = self.config.min_batch_size

            num_discriminator_opt_steps = 100
            num_mcmc_opt_steps = 100
            max_iters = 10

            self.init_gaussian_generator()
            self.init_models()

            self.models_dict['generator'].eval()
            self.models_dict['regressor'].eval()
            self.models_dict['discriminator'].eval()

            '''instantiate batch'''
            crystaldata_batch = self.collater([molecule_data for _ in range(batch_size)]).to(self.device)
            refresh_inds = torch.arange(batch_size)
            converged_samples_list = []
            optimization_trajectories = []

            for opt_iter in range(max_iters):
                crystaldata_batch = self.refresh_crystal_batch(crystaldata_batch, refresh_inds=refresh_inds)

                crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(
                    crystaldata_batch,
                    mode='mcmc',
                    num_steps=num_mcmc_opt_steps,
                    temperature=0.05,
                    step_size=0.01)
                optimization_trajectories.append(opt_traj)

                crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(
                    crystaldata_batch,
                    mode='discriminator',
                    num_steps=num_discriminator_opt_steps)
                optimization_trajectories.append(opt_traj)

                crystaldata_batch, refresh_inds, converged_samples = self.prune_crystaldata_batch(crystaldata_batch,
                                                                                                  optimization_trajectories)

                converged_samples_list.extend(converged_samples)

            aa = 1
            # do clustering

            # compare to ground truth
            # add convergence flags based on completeness of sampling

            # '''compare samples to ground truth'''
            # if data_contains_ground_truth:
            #     ground_truth_analysis = self.analyze_real_crystal(molecule_data)
            #

    def prune_crystaldata_batch(self, crystaldata_batch, optimization_trajectories):
        """
        Identify trajectories which have converged.
        """

        """
        combined_traj_dict = {key: np.concatenate(
            [traj[key] for traj in optimization_trajectories], axis=0)
            for key in optimization_trajectories[1].keys()}

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(cols=3, rows=1, subplot_titles=['score','vdw_score','packing_coeff'])
        for i in range(crystaldata_batch.num_graphs):
            for j, key in enumerate(['score','vdw_score','packing_coeff']):
                col = j % 3 + 1
                row = j // 3 + 1
                fig.add_scattergl(y=combined_traj_dict[key][:, i], name=i, legendgroup=i, showlegend=True if j == 0 else False, row=row, col=col)
        fig.show(renderer='browser')

        """

        refresh_inds = np.arange(crystaldata_batch.num_graphs)  # todo write a function that actually checks for this
        converged_samples = [crystaldata_batch[i] for i in refresh_inds.tolist()]

        return crystaldata_batch, refresh_inds, converged_samples

    def optimize_crystaldata_batch(self, batch, mode, num_steps, temperature=None, step_size=None):
        """
        method which takes a batch of crystaldata objects
        and optimzies them according to a score model either
        with MCMC or gradient descent
        """
        if mode.lower() == 'mcmc':
            sampling_dict = mcmc_sampling(
                self.models_dict['discriminator'], batch,
                self.supercell_builder,
                num_steps, self.vdw_radii,
                supercell_size=5, cutoff=6,
                sampling_temperature=temperature,
                lattice_means=self.dataDims['lattice_means'],
                lattice_stds=self.dataDims['lattice_stds'],
                step_size=step_size,
            )
        elif mode.lower() == 'discriminator':
            sampling_dict = gradient_descent_sampling(
                self.models_dict['discriminator'], batch,
                self.supercell_builder,
                num_steps, 1e-3,
                torch.optim.Rprop, self.vdw_radii,
                lattice_means=self.dataDims['lattice_means'],
                lattice_stds=self.dataDims['lattice_stds'],
                supercell_size=5, cutoff=6,
            )
        else:
            assert False, f"{mode.lower()} is not a valid sampling mode!"

        '''return best sample'''
        best_inds = np.argmax(sampling_dict['score'], axis=0)
        best_samples = sampling_dict['std_cell_params'][best_inds, np.arange(batch.num_graphs), :]
        supercell_data, _ = \
            self.supercell_builder.build_zp1_supercells(
                batch, torch.tensor(best_samples, dtype=torch.float32, device=batch.x.device),
                5, 6,
                align_to_standardized_orientation=True,
                target_handedness=batch.aunit_handedness)

        output, proposed_dist_dict = self.models_dict['discriminator'](supercell_data.clone().cuda(), return_dists=True)

        rebuilt_sample_scores = softmax_and_score(output[:, :2]).cpu().detach().numpy()

        cell_params_difference = np.amax(
            np.sum(np.abs(supercell_data.cell_params.cpu().detach().numpy() - best_samples), axis=1))
        rebuilt_scores_difference = np.amax(np.abs(rebuilt_sample_scores - sampling_dict['score'].max(0)))

        if rebuilt_scores_difference > 1e-2 or cell_params_difference > 1e-2:
            aa = 1
            assert False, "Best cell rebuilding failed!"  # confirm we rebuilt the cells correctly

        sampling_dict['best_samples'] = best_samples
        sampling_dict['best_scores'] = sampling_dict['score'].max(0)
        sampling_dict['best_vdws'] = np.diag(sampling_dict['vdw_score'][best_inds, :])

        best_batch = batch.clone()
        best_batch.cell_params = torch.tensor(best_samples, dtype=torch.float32, device=supercell_data.x.device)

        return best_batch, sampling_dict

    def refresh_crystal_batch(self, crystaldata, refresh_inds, generator='gaussian', space_groups: torch.tensor = None):
        # crystaldata = self.set_molecule_alignment(crystaldata, right_handed=False, mode_override=mol_orientation)

        if space_groups is not None:
            crystaldata.sg_ind = space_groups

        if generator == 'gaussian':
            samples = self.gaussian_generator.forward(crystaldata.num_graphs, crystaldata).to(self.config.device)
            crystaldata.cell_params = samples[refresh_inds]
            # todo add option for generator here

        return crystaldata

    def generator_density_matching_loss(self, standardized_target_auv,
                                        data, raw_sample,
                                        precomputed_volumes=None, loss_func='mse'):
        """
        compute packing coefficients for generated cells
        compute losses relating to packing density
        """
        if precomputed_volumes is None:
            volumes_list = []
            for i in range(len(raw_sample)):
                volumes_list.append(cell_vol_torch(data.cell_params[i, 0:3], data.cell_params[i, 3:6]))
            volumes = torch.stack(volumes_list)
        else:
            volumes = precomputed_volumes

        # generated_packing_coeffs = data.sym_mult * data.mol_volume / volumes
        generated_auvs = volumes / data.sym_mult
        # standardized_gen_packing_coeffs = (generated_packing_coeffs - self.std_dict['crystal_packing_coefficient'][0]) / \
        #                                   self.std_dict['crystal_packing_coefficient'][1]
        standardized_gen_auvs = (generated_auvs - self.std_dict['crystal_reduced_volume'][0]) / \
                                self.std_dict['crystal_reduced_volume'][
                                    1]  # todo std dict deprecated in current version
        #
        # target_packing_coeffs = standardized_target_auv * self.std_dict['crystal_packing_coefficient'][1] + \
        #                         self.std_dict['crystal_packing_coefficient'][0]
        target_auvs = standardized_target_auv * self.std_dict['crystal_reduced_volume'][1] + \
                      self.std_dict['crystal_reduced_volume'][0]

        # csd_packing_coeffs = data.tracking[:, self.t_i_d['crystal_packing_coefficient']]
        csd_auvs = data.tracking[:, self.t_i_d['crystal_reduced_volume']]  # TODO also deprecated

        # compute loss vs the target
        if loss_func == 'mse':
            packing_loss = F.mse_loss(standardized_gen_auvs, standardized_target_auv,
                                      reduction='none')  # allow for more error around the minimum
        elif loss_func == 'l1':
            packing_loss = F.smooth_l1_loss(standardized_gen_auvs, standardized_target_auv,
                                            reduction='none')
        else:
            assert False, "Must pick from the set of implemented packing loss functions 'mse', 'l1'"
        return packing_loss, generated_auvs, target_auvs, csd_auvs

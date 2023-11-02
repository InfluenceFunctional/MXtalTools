import os
from datetime import datetime
from argparse import Namespace
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime

import sys
import gc

import torch
import torch.random
import wandb
from torch import backends
import torch.nn as nn
import numpy as np
import tqdm
from shutil import copy
from distutils.dir_util import copy_tree
from torch.nn import functional as F
from torch_geometric.loader.dataloader import Collater

from constants.atom_properties import VDW_RADII, ATOM_WEIGHTS
from constants.asymmetric_units import asym_unit_dict
from csp.SampleOptimization import gradient_descent_sampling
from models.crystal_rdf import crystal_rdf, new_crystal_rdf

from models.discriminator_models import crystal_discriminator, crystal_proxy_discriminator
from models.generator_models import crystal_generator, independent_gaussian_model
from models.regression_models import molecule_regressor
from models.utils import (reload_model, init_schedulers, softmax_and_score, compute_packing_coefficient,
                          save_checkpoint, set_lr, cell_vol_torch, init_optimizer, get_regression_loss, compute_num_h_bonds, slash_batch)
from models.utils import (weight_reset, get_n_config)
from models.vdw_overlap import vdw_overlap

from crystal_building.utils import (random_crystaldata_alignment, align_crystaldata_to_principal_axes,
                                    batch_molecule_principal_axes_torch, compute_Ip_handedness, clean_cell_params)
from crystal_building.builder import SupercellBuilder
from crystal_building.utils import update_crystal_symmetry_elements

from dataset_management.manager import DataManager
from dataset_management.utils import (get_dataloaders, update_dataloader_batch_size)
from reporting.logger import Logger

from common.utils import softmax_np, init_sym_info, compute_rdf_distance


# https://www.ruppweb.org/Xray/tutorial/enantio.htm non enantiogenic groups
# https://dictionary.iucr.org/Sohncke_groups#:~:text=Sohncke%20groups%20are%20the%20three,in%20the%20chiral%20space%20groups.


class Modeller:
    """
    main class which handles everything
    """

    def __init__(self, config):
        """
        initialize config, physical constants, SGs to be generated
        load dataset and statistics
        decide what models we are training
        """
        self.config = config
        self.device = self.config.device
        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        self.packing_loss_coefficient = 1
        '''get some physical constants'''
        self.atom_weights = ATOM_WEIGHTS
        self.vdw_radii = VDW_RADII
        self.sym_info = init_sym_info()

        self.supercell_builder = SupercellBuilder(device=self.config.device, rotation_basis='spherical')

        '''set space groups to be included and generated'''
        if self.config.generate_sgs == 'all':
            self.config.generate_sgs = [self.sym_info['space_groups'][int(key)] for key in asym_unit_dict.keys()]

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
        copy_tree("common", self.working_directory + "/source/common")
        copy_tree("crystal_building", self.working_directory + "/source/crystal_building")
        copy_tree("dataset_management", self.working_directory + "/source/dataset_management")
        copy_tree("models", self.working_directory + "/source/models")
        copy_tree("reporting", self.working_directory + "/source/reporting")
        copy_tree("csp", self.working_directory + "/source/csp")
        copy("crystal_modeller.py", self.working_directory + "/source")
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
        self.run_identifier = str(self.config.paths.yaml_path).split('.yaml')[0].split('configs')[1].replace('\\', '_').replace('/', '_') + '_' + datetime.today().strftime("%d-%m-%H-%M-%S")
        self.working_directory = self.config.workdir + self.run_identifier
        os.mkdir(self.working_directory)

    def init_models(self):
        """
        Initialize models, optimizers, schedulers
        for models we will not use, just set them as nn.Linear(1,1)
        :return:
        """
        self.config = self.reload_model_checkpoints()

        self.generator, self.discriminator, self.regressor, self.proxy_discriminator = [nn.Linear(1, 1) for _ in range(4)]
        print("Initializing model(s) for " + self.config.mode)
        if self.config.mode == 'gan' or self.config.mode == 'sampling' or self.config.mode == 'embedding':
            self.generator = crystal_generator(self.config.seeds.model, self.device, self.config.generator.model, self.dataDims, self.sym_info)
            self.discriminator = crystal_discriminator(self.config.seeds.model, self.config.discriminator.model, self.dataDims)
            self.proxy_discriminator = crystal_proxy_discriminator(self.config.seeds.model, self.config.proxy_discriminator.model, self.dataDims)
        if self.config.mode == 'regression' or self.config.regressor_path is not None:
            self.regressor = molecule_regressor(self.config.seeds.model, self.config.regressor.model, self.dataDims)

        if self.config.device.lower() == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.regressor = self.regressor.cuda()
            self.proxy_discriminator = self.proxy_discriminator.cuda()

        self.generator_optimizer = init_optimizer(self.config.generator.optimizer, self.generator)
        self.discriminator_optimizer = init_optimizer(self.config.discriminator.optimizer, self.discriminator)
        self.regressor_optimizer = init_optimizer(self.config.regressor.optimizer, self.regressor)
        self.proxy_discriminator_optimizer = init_optimizer(self.config.regressor.optimizer, self.proxy_discriminator)

        if self.config.generator_path is not None and (self.config.mode == 'gan' or self.config.mode == 'embedding'):
            self.generator, self.generator_optimizer = reload_model(self.generator, self.generator_optimizer,
                                                                    self.config.generator_path)
        if self.config.discriminator_path is not None and (self.config.mode == 'gan' or self.config.mode == 'embedding'):
            self.discriminator, self.discriminator_optimizer = reload_model(self.discriminator, self.discriminator_optimizer,
                                                                            self.config.discriminator_path)
        if self.config.regressor_path is not None:
            self.regressor, self.regressor_optimizer = reload_model(self.regressor, self.regressor_optimizer,
                                                                    self.config.regressor_path)
        if self.config.proxy_discriminator_path is not None:
            self.proxy_discriminator, self.proxy_discriminator_optimizer = reload_model(self.proxy_discriminator, self.proxy_discriminator_optimizer,
                                                                                        self.config.proxy_discriminator_path)

        self.generator_schedulers = init_schedulers(self.generator_optimizer, self.config.generator.optimizer.lr_shrink_lambda, self.config.generator.optimizer.lr_growth_lambda)
        self.discriminator_schedulers = init_schedulers(self.discriminator_optimizer, self.config.discriminator.optimizer.lr_shrink_lambda, self.config.discriminator.optimizer.lr_growth_lambda)
        self.regressor_schedulers = init_schedulers(self.regressor_optimizer, self.config.regressor.optimizer.lr_shrink_lambda, self.config.regressor.optimizer.lr_growth_lambda)
        self.proxy_discriminator_schedulers = init_schedulers(self.proxy_discriminator_optimizer, self.config.proxy_discriminator.optimizer.lr_shrink_lambda, self.config.proxy_discriminator.optimizer.lr_growth_lambda)

        num_params = [get_n_config(model) for model in [self.generator, self.discriminator, self.regressor, self.proxy_discriminator]]
        print('Generator model has {:.3f} million or {} parameters'.format(num_params[0] / 1e6, int(num_params[0])))
        print('Discriminator model has {:.3f} million or {} parameters'.format(num_params[1] / 1e6, int(num_params[1])))
        print('Regressor model has {:.3f} million or {} parameters'.format(num_params[2] / 1e6, int(num_params[2])))
        print('Proxy discriminator model has {:.3f} million or {} parameters'.format(num_params[3] / 1e6, int(num_params[3])))

        wandb.watch((self.generator, self.discriminator, self.regressor, self.proxy_discriminator), log_graph=True, log_freq=100)
        wandb.log({"All Models Parameters": np.sum(np.asarray(num_params)),
                   "Initial Batch Size": self.config.current_batch_size})

    def load_dataset_and_dataloaders(self, override_test_fraction=None):
        """
        use data manager to load and filter dataset
        use dataset builder to generate crystaldata objects
        return dataloaders
        """

        """load and filter dataset"""
        data_manager = DataManager(device=self.device,
                                   datasets_path=self.config.dataset_path)
        data_manager.load_dataset_for_modelling(
            config=self.config.dataset,
            dataset_name=self.config.dataset_name,
            misc_dataset_name=self.config.misc_dataset_name,
            filter_conditions=self.config.dataset.filter_conditions,
            filter_polymorphs=self.config.dataset.filter_polymorphs,
            filter_duplicate_molecules=self.config.dataset.filter_duplicate_molecules
        )
        self.dataDims = data_manager.dataDims
        self.t_i_d = {feat: index for index, feat in enumerate(self.dataDims['tracking_features'])}  # tracking feature index dictionary
        self.lattice_means = torch.tensor(self.dataDims['lattice_means'], dtype=torch.float32, device=self.config.device)
        self.lattice_stds = torch.tensor(self.dataDims['lattice_stds'], dtype=torch.float32, device=self.config.device)
        self.std_dict = data_manager.standardization_dict

        if self.config.extra_test_set_name is not None:
            blind_test_conditions = [['crystal_z_prime', 'in', [1]],  # very permissive
                                     ['crystal_z_value', 'range', [1, 32]],
                                     ['atom_atomic_numbers', 'range', [1, 100]]]

            # omit blind test 5 & 6 targets
            extra_data_manager = DataManager(device=self.device,
                                             datasets_path=self.config.dataset_path
                                             )
            extra_data_manager.load_dataset_for_modelling(
                config=self.config.dataset,
                dataset_name=self.config.extra_test_set_name,
                misc_dataset_name=self.config.misc_dataset_name,
                override_length=int(1e7),
                filter_conditions=blind_test_conditions,  # standard filtration conditions
                filter_polymorphs=False,  # do not filter duplicates - e.g., in Blind test they're almost all duplicates!
                filter_duplicate_molecules=False,
            )

        else:
            extra_data_manager = None

        """return dataloaders"""
        if override_test_fraction is not None:
            test_fraction = override_test_fraction
        else:
            test_fraction = self.config.dataset.test_fraction

        return self.prep_dataloaders(data_manager, extra_data_manager, test_fraction)

    def prep_dataloaders(self, dataset_builder, extra_dataset_builder=None, test_fraction=0.2, override_batch_size: int = None):
        """
        get training, test, ane optionall extra validation dataloaders
        """
        if override_batch_size is None:
            loader_batch_size = self.config.min_batch_size
        else:
            loader_batch_size = override_batch_size
        train_loader, test_loader = get_dataloaders(dataset_builder,
                                                    machine=self.config.machine,
                                                    batch_size=loader_batch_size,
                                                    test_fraction=test_fraction)
        self.config.current_batch_size = self.config.min_batch_size
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

        return train_loader, test_loader, extra_test_loader

    #
    # def crystal_embedding_analysis(self):
    #     """
    #     analyze the embeddings of a given crystal dataset
    #     embeddings provided by pretrained model
    #     """
    #     """
    #             train and/or evaluate one or more models
    #             regressor
    #             GAN (generator and/or discriminator)
    #             """
    #     with wandb.init(config=self.config,
    #                     project=self.config.wandb.project_name,
    #                     entity=self.config.wandb.username,
    #                     tags=[self.config.logger.experiment_tag],
    #                     settings=wandb.Settings(code_dir=".")):
    #
    #         wandb.run.name = wandb.config.machine + '_' + str(self.config.mode) + '_' + str(wandb.config.run_num)  # overwrite procedurally generated run name with our run name
    #
    #         '''miscellaneous setup'''
    #         dataset_builder = self.misc_pre_training_items()
    #
    #         '''prep dataloaders'''
    #         from torch_geometric.loader import DataLoader
    #         test_dataset = []
    #         for i in range(len(dataset_builder)):
    #             test_dataset.append(dataset_builder[i])
    #
    #         self.config.current_batch_size = self.config.min_batch_size
    #         print("Training batch size set to {}".format(self.config.current_batch_size))
    #         del dataset_builder
    #         test_loader = DataLoader(test_dataset, batch_size=self.config.current_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    #
    #         '''instantiate models'''
    #         self.init_models()
    #
    #         '''initialize some training metrics'''
    #         with torch.autograd.set_detect_anomaly(self.config.anomaly_detection):
    #             # very cool
    #             print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
    #             print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
    #             print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
    #             # very cool
    #             print("Starting Embedding Analysis")
    #
    #             with torch.no_grad():
    #                 # compute test loss & save evaluation statistics on test samples
    #                 embedding_dict = self.embed_dataset(
    #                     data_loader=test_loader, generator=generator, discriminator=discriminator, regressor=regressor)
    #
    #                 np.save('embedding_dict', embedding_dict) #
    #
    # def embed_dataset(self, data_loader):
    #     t0 = time.time()
    #     discriminator.eval()
    #
    #     embedding_dict = {
    #         'tracking_features': [],
    #         'identifiers': [],
    #         'scores': [],
    #         'source': [],
    #         'final_activation': [],
    #     }
    #
    #     for i, data in enumerate(tqdm.tqdm(data_loader)):
    #         '''
    #         get discriminator embeddings
    #         '''
    #
    #         '''real data'''
    #         real_supercell_data = self.supercell_builder.prebuilt_unit_cell_to_supercell(
    #             data, self.config.supercell_size, self.config.discriminator.model.convolution_cutoff)
    #
    #         score_on_real, real_distances_dict, latent = \
    #             self.adversarial_score(real_supercell_data, return_latent=True)
    #
    #         embedding_dict['tracking_features'].extend(data.tracking.cpu().detach().numpy())
    #         embedding_dict['identifiers'].extend(data.csd_identifier)
    #         embedding_dict['scores'].extend(score_on_real.cpu().detach().numpy())
    #         embedding_dict['final_activation'].extend(latent)
    #         embedding_dict['source'].extend(['real' for _ in range(len(latent))])
    #
    #         '''fake data'''
    #         for j in tqdm.tqdm(range(100)):
    #             real_data = data.clone()
    #             generated_samples_i, negative_type, real_data = \
    #                 self.generate_discriminator_negatives(real_data, i, override_randn=True, override_distorted=True)
    #
    #             fake_supercell_data, generated_cell_volumes = self.supercell_builder.build_supercells(
    #                 real_data, generated_samples_i, self.config.supercell_size,
    #                 self.config.discriminator.model.convolution_cutoff,
    #                 align_to_standardized_orientation=(negative_type != 'generated'),
    #                 target_handedness=real_data.asym_unit_handedness,
    #             )
    #
    #             score_on_fake, fake_pairwise_distances_dict, fake_latent = self.adversarial_score(fake_supercell_data, return_latent=True)
    #
    #             embedding_dict['tracking_features'].extend(real_data.tracking.cpu().detach().numpy())
    #             embedding_dict['identifiers'].extend(real_data.csd_identifier)
    #             embedding_dict['scores'].extend(score_on_fake.cpu().detach().numpy())
    #             embedding_dict['final_activation'].extend(fake_latent)
    #             embedding_dict['source'].extend([negative_type for _ in range(len(latent))])
    #
    #     embedding_dict['scores'] = np.stack(embedding_dict['scores'])
    #     embedding_dict['tracking_features'] = np.stack(embedding_dict['tracking_features'])
    #     embedding_dict['final_activation'] = np.stack(embedding_dict['final_activation'])
    #
    #     total_time = time.time() - t0
    #     print(f"Embedding took {total_time:.1f} Seconds")
    #
    #     '''distance matrix'''
    #     scores = softmax_and_score(embedding_dict['scores'])
    #     latents = torch.Tensor(embedding_dict['final_activation'])
    #     overlaps = torch.inner(latents, latents) / torch.outer(torch.linalg.norm(latents, dim=-1), torch.linalg.norm(latents, dim=-1))
    #     distmat = torch.cdist(latents, latents)
    #
    #     sample_types = list(set(embedding_dict['source']))
    #     inds_dict = {}
    #     for source in sample_types:
    #         inds_dict[source] = np.argwhere(np.asarray(embedding_dict['source']) == source)[:, 0]
    #
    #     mean_overlap_to_real = {}
    #     mean_dist_to_real = {}
    #     mean_score = {}
    #     for source in sample_types:
    #         sample_dists = distmat[inds_dict[source]]
    #         sample_scores = scores[inds_dict[source]]
    #         sample_overlaps = overlaps[inds_dict[source]]
    #
    #         mean_dist_to_real[source] = sample_dists[:, inds_dict['real']].mean()
    #         mean_overlap_to_real[source] = sample_overlaps[:, inds_dict['real']].mean()
    #         mean_score[source] = sample_scores.mean()
    #
    #     import plotly.graph_objects as go
    #     from plotly.subplots import make_subplots
    #     from plotly.colors import n_colors
    #
    #     # '''distances'''
    #     # fig = make_subplots(rows=1, cols=2, subplot_titles=('distances', 'dot overlaps'))
    #     # fig.add_trace(go.Heatmap(z=distmat), row=1, col=1)
    #     # fig.add_trace(go.Heatmap(z=overlaps), row=1, col=2)
    #     # fig.show()
    #
    #     '''distance to real vs score'''
    #     colors = n_colors('rgb(250,0,5)', 'rgb(5,150,250)', len(inds_dict.keys()), colortype='rgb')
    #
    #     fig = make_subplots(rows=1, cols=2)
    #     for ii, source in enumerate(sample_types):
    #         fig.add_trace(go.Scattergl(
    #             x=distmat[inds_dict[source]][:, inds_dict['real']].mean(-1), y=scores[inds_dict[source]],
    #             mode='markers', marker=dict(color=colors[ii]), name=source), row=1, col=1
    #         )
    #
    #         fig.add_trace(go.Scattergl(
    #             x=overlaps[inds_dict[source]][:, inds_dict['real']].mean(-1), y=scores[inds_dict[source]],
    #             mode='markers', marker=dict(color=colors[ii]), showlegend=False), row=1, col=2
    #         )
    #
    #     fig.update_xaxes(title_text='mean distance to real', row=1, col=1)
    #     fig.update_yaxes(title_text='discriminator score', row=1, col=1)
    #
    #     fig.update_xaxes(title_text='mean overlap to real', row=1, col=2)
    #     fig.update_yaxes(title_text='discriminator score', row=1, col=2)
    #     fig.show()
    #
    #     return embedding_dict

    #
    # def prep_standalone_modelling_tools(self, batch_size, machine='local'):
    #     """
    #     to pass tools to another training pipeline
    #     """
    #     '''miscellaneous setup'''
    #     if machine == 'local':
    #         std_dataDims_path = '/home/mkilgour/mcrygan/old_dataset_management/standard_dataDims.npy'
    #     elif machine == 'cluster':
    #         std_dataDims_path = '/scratch/mk8347/mcrygan/old_dataset_management/standard_dataDims.npy'
    #
    #     standard_dataDims = np.load(std_dataDims_path, allow_pickle=True).item()  # maintain constant standardizations between runs
    #
    #     '''note this standard datadims construction will only work between runs with
    #     identical choice of features - there is a flag for this in the datasetbuilder'''
    #     dataset_builder = TrainingDataBuilder(self.config.dataset,
    #                                           preloaded_dataset=self.prepped_dataset,
    #                                           data_std_dict=self.std_dict,
    #                                           override_length=self.config.dataset.max_dataset_length)
    #
    #     self.dataDims = dataset_builder.dataDims
    #     del self.prepped_dataset  # we don't actually want this huge thing floating around
    #
    #     train_loader, test_loader, extra_test_loader = (
    #         self.prep_dataloaders(dataset_builder, test_fraction=0.2, override_batch_size=batch_size))
    #
    #     return train_loader, test_loader

    def train_crystal_models(self):
        """
        train and/or evaluate one or more models
        regressor
        GAN (generator and/or discriminator)
        """
        '''prep workdir'''
        self.source_directory = os.getcwd()
        self.prep_new_working_directory()

        with ((wandb.init(config=self.config,
                          project=self.config.wandb.project_name,
                          entity=self.config.wandb.username,
                          tags=[self.config.logger.experiment_tag],
                          settings=wandb.Settings(code_dir=".")))):

            wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
            # config = wandb.config # wandb configs don't support nested namespaces. look at the github thread to see if they eventually fix it
            # this means we also can't do wandb sweeps properly, as-is

            self.train_discriminator = (self.config.mode == 'gan') and any((self.config.discriminator.train_adversarially, self.config.discriminator.train_on_distorted, self.config.discriminator.train_on_randn))
            self.train_generator = (self.config.mode == 'gan') and any((self.config.generator.train_vdw, self.config.generator.train_adversarially, self.config.generator.train_h_bond))
            self.train_regressor = self.config.mode == 'regression'
            self.train_proxy_discriminator = (self.config.mode == 'gan') and self.config.proxy_discriminator.train

            '''initialize datasets and useful classes'''
            train_loader, test_loader, extra_test_loader = self.load_dataset_and_dataloaders()
            self.misc_pre_training_items()
            self.logger = Logger(self.config, self.dataDims, wandb)
            self.init_models()

            '''initialize some training metrics'''
            self.discriminator_hit_max_lr, self.generator_hit_max_lr, self.regressor_hit_max_lr, self.proxy_discriminator_hit_max_lr, converged, epoch, prev_epoch_failed = \
                (False, False, False, False,
                 self.config.max_epochs == 0, 0, False)

            # training loop
            with torch.autograd.set_detect_anomaly(self.config.anomaly_detection):
                while (epoch < self.config.max_epochs) and not converged:
                    print("⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅⋅.˳˳.⋅ॱ˙˙ॱ⋅.˳˳.⋅ॱ˙˙ॱᐧ.˳˳.⋅")
                    print("Starting Epoch {}".format(epoch))  # index from 0
                    self.logger.reset_for_new_epoch(epoch, test_loader.batch_size)

                    if epoch < self.config.num_early_epochs:
                        early_epochs_step_override = self.config.early_epochs_step_override
                    else:
                        early_epochs_step_override = None

                    try:  # try this batch size
                        self.run_epoch(epoch_type='train', data_loader=train_loader,
                                       update_gradients=True, iteration_override=early_epochs_step_override)

                        with torch.no_grad():
                            self.run_epoch(epoch_type='test', data_loader=test_loader,
                                           update_gradients=False, iteration_override=early_epochs_step_override)

                            if (extra_test_loader is not None) and (epoch % self.config.extra_test_period == 0) and (epoch > 0):
                                self.run_epoch(epoch_type='extra', data_loader=extra_test_loader,
                                               update_gradients=False, iteration_override=early_epochs_step_override)  # compute loss on test set

                        self.logger.numpyize_current_losses()
                        self.logger.update_loss_record()

                        '''update learning rates'''
                        self.update_lr()

                        '''save checkpoints'''
                        if self.config.save_checkpoints and epoch > 0:
                            self.model_checkpointing(epoch)

                        '''check convergence status'''
                        self.logger.check_model_convergence()

                        '''sometimes test the generator on a mini CSP problem'''
                        if (self.config.mode == 'gan') and (epoch % self.config.logger.mini_csp_frequency == 0) and \
                                self.train_generator and (epoch > 0):
                            pass  # self.batch_csp(extra_test_loader if extra_test_loader is not None else test_loader)

                        '''record metrics and analysis'''
                        self.logger.log_training_metrics()
                        self.logger.log_epoch_analysis(test_loader)

                        if (self.logger.generator_converged and self.logger.discriminator_converged and self.logger.regressor_converged) \
                                and (epoch > self.config.history + 2):
                            print('Training has converged!')
                            break

                        '''increment batch size'''
                        train_loader, test_loader, extra_test_loader = \
                            self.increment_batch_size(train_loader, test_loader, extra_test_loader)

                        prev_epoch_failed = False

                    except RuntimeError as e:  # if we do hit OOM, slash the batch size
                        if "CUDA out of memory" in str(e):
                            if prev_epoch_failed:
                                # print(torch.cuda.memory_summary())
                                gc.collect()
                                # for obj in gc.get_objects():
                                #     try:
                                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                #             print(type(obj), obj.size())
                                #     except:
                                #         pass
                            train_loader, test_loader = slash_batch(train_loader, test_loader, 0.05)  # shrink batch size
                            torch.cuda.empty_cache()
                            self.config.grow_batch_size = False  # stop growing the batch for the rest of the run
                            prev_epoch_failed = True
                        else:
                            raise e  # will simply raise error if training on CPU
                    epoch += 1

                    if self.config.device.lower() == 'cuda':
                        torch.cuda.empty_cache()  # clear GPU --- not clear this does anything

                if self.config.mode == 'gan':  # evaluation on test metrics
                    self.gan_evaluation(epoch, test_loader, extra_test_loader)

    def run_epoch(self, epoch_type, data_loader=None, update_gradients=True, iteration_override=None):
        self.epoch_type = epoch_type
        if self.config.mode == 'gan':
            if self.config.regressor_path is not None:
                self.regressor.eval()  # just using this to suggest densities to the generator

            if self.train_discriminator or self.train_generator:
                self.gan_epoch(data_loader, update_gradients, iteration_override)

            if self.train_proxy_discriminator:
                self.proxy_discriminator_epoch(data_loader, update_gradients, iteration_override)

        elif self.config.mode == 'regression':
            self.regression_epoch(data_loader, update_gradients, iteration_override)

    def regression_epoch(self, data_loader, update_gradients=True, iteration_override=None):
        if update_gradients:
            self.regressor.train(True)
        else:
            self.regressor.eval()

        stats_keys = ['regressor_prediction', 'regressor_target', 'tracking_features']

        for i, data in enumerate(tqdm.tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            if self.config.regressor_positional_noise > 0:
                data.pos += torch.randn_like(data.pos) * self.config.regressor_positional_noise

            data = data.to(self.device)

            regression_losses_list, predictions, targets = get_regression_loss(self.regressor, data, self.dataDims['target_mean'], self.dataDims['target_std'])
            regression_loss = regression_losses_list.mean()

            if update_gradients:
                self.regressor_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                regression_loss.backward()  # back-propagation
                self.regressor_optimizer.step()  # update parameters

            '''log losses and other tracking values'''
            self.logger.update_current_losses('regressor', self.epoch_type,
                                              regression_loss.cpu().detach().numpy(),
                                              regression_losses_list.cpu().detach().numpy())

            stats_values = [predictions, targets]
            self.logger.update_stats_dict(self.epoch_type, stats_keys, stats_values, mode='extend')
            self.logger.update_stats_dict(self.epoch_type, 'tracking_features', data.tracking.cpu().detach().numpy(), mode='append')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        self.logger.numpyize_stats_dict(self.epoch_type)

    def gan_epoch(self, data_loader=None, update_gradients=True,
                  iteration_override=None):

        if update_gradients:
            self.generator.train(True)
            self.discriminator.train(True)
        else:
            self.generator.eval()
            self.discriminator.eval()

        for i, data in enumerate(tqdm.tqdm(data_loader, miniters=int(len(data_loader) / 10), mininterval=30)):
            data = data.to(self.config.device)

            '''
            train discriminator
            '''
            skip_discriminator_step = self.decide_whether_to_skip_discriminator(i, self.logger.get_stat_dict(self.epoch_type))

            self.discriminator_step(data, i, update_gradients, skip_step=skip_discriminator_step)
            '''
            train_generator
            '''
            self.generator_step(data, i, update_gradients)
            '''
            record some stats
            '''
            self.logger.update_stats_dict(self.epoch_type, 'tracking_features', data.tracking.cpu().detach().numpy(), mode='append')
            self.logger.update_stats_dict(self.epoch_type, 'identifiers', data.csd_identifier, mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        self.logger.numpyize_stats_dict(self.epoch_type)

    def proxy_discriminator_epoch(self, data_loader=None, update_gradients=True,
                                  iteration_override=None):

        if update_gradients:
            self.proxy_discriminator.train(True)
            self.discriminator.train(True)
        else:
            self.proxy_discriminator.eval()
            self.discriminator.eval()

        for i, data in enumerate(tqdm.tqdm(data_loader, miniters=int(len(data_loader) / 10), mininterval=30)):
            data = data.to(self.config.device)

            self.proxy_discriminator_step(data, i, update_gradients)
            '''
            record some stats
            '''
            self.logger.update_stats_dict(self.epoch_type, 'tracking_features', data.tracking.cpu().detach().numpy(), mode='append')
            self.logger.update_stats_dict(self.epoch_type, 'identifiers', data.csd_identifier, mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        self.logger.numpyize_stats_dict(self.epoch_type)

    def proxy_discriminator_step(self, data, i, update_gradients):
        """
         execute a complete training step for the discriminator
         compute losses, do reporting, update gradients
         """
        '''get real supercells'''
        real_supercell_data = self.supercell_builder.prebuilt_unit_cell_to_supercell(
            data, self.config.supercell_size, self.config.discriminator.model.convolution_cutoff)

        '''get fake supercells'''
        generated_samples_i, negative_type, generator_data = \
            self.generate_discriminator_negatives(data, i, override_distorted=True, override_randn=True, orientation='as is')

        fake_supercell_data, generated_cell_volumes = self.supercell_builder.build_supercells(
            generator_data, generated_samples_i, self.config.supercell_size,
            self.config.discriminator.model.convolution_cutoff,
            align_to_standardized_orientation=False,  # take generator samples as-given
            target_handedness=generator_data.asym_unit_handedness,
        )

        generated_samples = fake_supercell_data.cell_params

        '''apply noise'''
        if self.config.discriminator_positional_noise > 0:
            real_supercell_data.pos += \
                torch.randn_like(real_supercell_data.pos) * self.config.discriminator_positional_noise
            fake_supercell_data.pos += \
                torch.randn_like(fake_supercell_data.pos) * self.config.discriminator_positional_noise

        '''score'''
        discriminator_output_on_real, real_pairwise_distances_dict, real_latent = self.adversarial_score(real_supercell_data, return_latent=True)
        discriminator_output_on_fake, fake_pairwise_distances_dict, fake_latent = self.adversarial_score(fake_supercell_data, return_latent=True)

        real_asym_unit_coords = torch.cat([real_supercell_data.pos[real_supercell_data.batch == ii, :][:int(data.mol_size[ii])] for ii in range(data.num_graphs)])
        data.pos = real_asym_unit_coords
        proxy_output_on_real = self.proxy_discriminator(data)
        proxy_output_on_fake = self.proxy_discriminator(generator_data)

        if False:  # test
            assert torch.sum(torch.abs(data.x - generator_data.x)) == 0
            assert torch.sum(torch.abs(data.pos - generator_data.pos)) == 0
            assert torch.sum(torch.abs(data.mol_x - generator_data.mol_x)) == 0

        '''recompute packing coeffs'''
        real_packing_coeffs = compute_packing_coefficient(cell_params=real_supercell_data.cell_params,
                                                          mol_volumes=real_supercell_data.mol_volume,
                                                          crystal_multiplicity=real_supercell_data.mult)
        fake_packing_coeffs = compute_packing_coefficient(cell_params=fake_supercell_data.cell_params,
                                                          mol_volumes=fake_supercell_data.mol_volume,
                                                          crystal_multiplicity=fake_supercell_data.mult)

        real_vdw_score = vdw_overlap(self.vdw_radii, crystaldata=real_supercell_data, return_score_only=True)
        fake_vdw_score = vdw_overlap(self.vdw_radii, crystaldata=fake_supercell_data, return_score_only=True)

        combined_outputs = torch.cat((proxy_output_on_real, proxy_output_on_fake))
        discriminator_target = torch.cat((torch.ones_like(proxy_output_on_real[:, 0]),
                                          torch.zeros_like(proxy_output_on_fake[:, 0])))

        score_on_real = softmax_and_score(discriminator_output_on_real)
        proxy_on_real = softmax_and_score(proxy_output_on_real)
        score_on_fake = softmax_and_score(discriminator_output_on_fake)
        proxy_on_fake = softmax_and_score(proxy_output_on_fake)
        proxy_losses = F.cross_entropy(combined_outputs, discriminator_target.long(), reduction='none') \
                       + F.smooth_l1_loss(torch.cat((score_on_real, score_on_fake)), torch.cat((proxy_on_real, proxy_on_fake)), reduction='none')

        proxy_loss = proxy_losses.mean()

        self.logger.update_current_losses('proxy_discriminator', self.epoch_type,
                                          proxy_loss.data.cpu().detach().numpy(),
                                          proxy_losses.cpu().detach().numpy())

        if update_gradients:
            self.proxy_discriminator_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
            torch.nn.utils.clip_grad_norm_(self.proxy_discriminator.parameters(),
                                           self.config.gradient_norm_clip)  # gradient clipping
            proxy_loss.backward()  # back-propagation
            self.proxy_discriminator_optimizer.step()  # update parameters

        stats_keys = ['discriminator_real_score', 'discriminator_fake_score',
                      'proxy_real_score', 'proxy_fake_score',
                      'real_vdw_penalty', 'fake_vdw_penalty',
                      'generated_cell_parameters', 'final_generated_cell_parameters',
                      'real_packing_coefficients', 'generated_packing_coefficients']
        stats_values = [score_on_real.cpu().detach().numpy(), score_on_fake.cpu().detach().numpy(),
                        proxy_on_real.cpu().detach().numpy(), proxy_on_fake.cpu().detach().numpy(),
                        -real_vdw_score.cpu().detach().numpy(), -fake_vdw_score.cpu().detach().numpy(),
                        generated_samples_i.cpu().detach().numpy(), generated_samples.cpu().detach().numpy(),
                        real_packing_coeffs.cpu().detach().numpy(), fake_packing_coeffs.cpu().detach().numpy()]
        self.logger.update_stats_dict(self.epoch_type, stats_keys, stats_values, mode='extend')

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
                    avg_generator_score = softmax_np(np.stack(epoch_stats_dict['discriminator_fake_score'])[generator_inds])[:, 1].mean()
                    if avg_generator_score < 0.5:
                        skip_discriminator_step = True
            else:
                skip_discriminator_step = True
        return skip_discriminator_step

    def adversarial_score(self, data, return_latent=False):
        """
        get the score from the discriminator on data
        """
        output, extra_outputs = self.discriminator(data.clone(), return_dists=True, return_latent=return_latent)  # reshape output from flat filters to channels * filters per channel
        if return_latent:
            return output, extra_outputs['dists_dict'], extra_outputs['final_activation']
        else:
            return output, extra_outputs['dists_dict']

    def discriminator_step(self, data, i, update_gradients, skip_step):
        """
        execute a complete training step for the discriminator
        compute losses, do reporting, update gradients
        """
        if self.train_discriminator:
            (discriminator_output_on_real, discriminator_output_on_fake,
             cell_distortion_size, real_fake_rdf_distances) \
                = self.get_discriminator_output(data, i)

            discriminator_losses = self.aggregate_discriminator_losses(
                discriminator_output_on_real,
                discriminator_output_on_fake,
                cell_distortion_size,
                real_fake_rdf_distances)

            discriminator_loss = discriminator_losses.mean()

            if update_gradients and (not skip_step):
                self.discriminator_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                               self.config.gradient_norm_clip)  # gradient clipping
                discriminator_loss.backward()  # back-propagation
                self.discriminator_optimizer.step()  # update parameters

    def aggregate_discriminator_losses(self,
                                       discriminator_output_on_real,
                                       discriminator_output_on_fake,
                                       cell_distortion_size,
                                       real_fake_rdf_distances):

        combined_outputs = torch.cat((discriminator_output_on_real, discriminator_output_on_fake))

        discriminator_target = torch.cat((torch.ones_like(discriminator_output_on_real[:, 0]),
                                          torch.zeros_like(discriminator_output_on_fake[:, 0])))
        distortion_target = torch.log10(1 + torch.cat((torch.zeros_like(discriminator_output_on_real[:, 0]),
                                                       cell_distortion_size)))  # rescale on log(1+x)

        classification_losses = F.cross_entropy(combined_outputs[:, :2], discriminator_target.long(), reduction='none')  # works much better
        distortion_losses = F.smooth_l1_loss(combined_outputs[:, 2], distortion_target, reduction='none')

        if real_fake_rdf_distances is not None:
            rdf_distance_target = torch.log10(1 + torch.cat((torch.zeros_like(discriminator_output_on_real[:, 0]),
                                                             real_fake_rdf_distances)))  # rescale on log(1+x)
            rdf_distance_losses = F.smooth_l1_loss(combined_outputs[:, 3], rdf_distance_target, reduction='none')

        else:
            rdf_distance_target = torch.zeros_like(discriminator_target)
            rdf_distance_losses = torch.zeros_like(classification_losses)

        score_on_real = softmax_and_score(discriminator_output_on_real[:, :2])
        score_on_fake = softmax_and_score(discriminator_output_on_fake[:, :2])

        stats_keys = ['discriminator_real_score',
                      'discriminator_fake_score',
                      'discriminator_fake_true_distance',
                      'discriminator_fake_predicted_distance',
                      'discriminator_real_true_distance',
                      'discriminator_real_predicted_distance',
                      'discriminator_classification_loss',
                      'discriminator_distortion_loss',
                      'discriminator_distance_loss']
        stats_values = [score_on_real.cpu().detach().numpy(),
                        score_on_fake.cpu().detach().numpy(),
                        torch.log10(1 + real_fake_rdf_distances).cpu().detach().numpy(),
                        discriminator_output_on_fake[:, 3].cpu().detach().numpy(),
                        torch.zeros_like(discriminator_output_on_real[:, 0]).cpu().detach().numpy(),
                        discriminator_output_on_real[:, 3].cpu().detach().numpy(),
                        classification_losses.cpu().detach().numpy(),
                        distortion_losses.cpu().detach().numpy(),
                        rdf_distance_losses.cpu().detach().numpy()]

        discriminator_losses_list = []
        if self.config.discriminator.use_classification_loss:
            discriminator_losses_list.append(classification_losses)

        if self.config.discriminator.use_rdf_distance_loss:
            discriminator_losses_list.append(rdf_distance_losses)

        if self.config.discriminator.use_cell_distance_loss:
            discriminator_losses_list.append(distortion_losses)

        discriminator_losses = torch.sum(torch.stack(discriminator_losses_list), dim=0)
        self.logger.update_stats_dict(self.epoch_type, stats_keys, stats_values, mode='extend')

        self.logger.update_current_losses('discriminator', self.epoch_type,
                                          discriminator_losses.mean().data.cpu().detach().numpy(),
                                          discriminator_losses.cpu().detach().numpy())

        return discriminator_losses

    def generator_step(self, data, i, update_gradients):
        """
        execute a complete training step for the generator
        get sample losses, do reporting, update gradients
        """
        if self.train_generator:
            discriminator_raw_output, generated_samples, raw_samples, packing_loss, packing_prediction, packing_target, \
                vdw_loss, vdw_score, generated_dist_dict, supercell_examples, similarity_penalty, h_bond_score = \
                self.get_generator_losses(data)

            generator_losses = self.aggregate_generator_losses(
                packing_loss, discriminator_raw_output, vdw_loss, vdw_score,
                similarity_penalty, packing_prediction, packing_target, h_bond_score)

            generator_loss = generator_losses.mean()
            self.logger.update_current_losses('generator', self.epoch_type,
                                              generator_loss.data.cpu().detach().numpy(),
                                              generator_losses.cpu().detach().numpy())

            if update_gradients:
                self.generator_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                               self.config.gradient_norm_clip)  # gradient clipping
                generator_loss.backward()  # back-propagation
                self.generator_optimizer.step()  # update parameters

            self.logger.update_stats_dict(self.epoch_type, ['final_generated_cell_parameters', 'generated_space_group_numbers', 'raw_generated_cell_parameters'],
                                          [supercell_examples.cell_params.cpu().detach().numpy(), supercell_examples.sg_ind.cpu().detach().numpy(), raw_samples], mode='extend')

            del supercell_examples

    def get_discriminator_output(self, data, i):
        """
        generate real and fake crystals
        and score them
        """
        '''get real supercells'''
        real_supercell_data = self.supercell_builder.prebuilt_unit_cell_to_supercell(
            data, self.config.supercell_size, self.config.discriminator.model.convolution_cutoff)

        '''get fake supercells'''
        generated_samples_i, negative_type, generator_data = \
            self.generate_discriminator_negatives(data, i, orientation=self.config.generator.canonical_conformer_orientation)

        fake_supercell_data, generated_cell_volumes = self.supercell_builder.build_supercells(
            generator_data, generated_samples_i, self.config.supercell_size,
            self.config.discriminator.model.convolution_cutoff,
            align_to_standardized_orientation=(negative_type != 'generated'),  # take generator samples as-given
            target_handedness=generator_data.asym_unit_handedness,
        )

        canonical_fake_cell_params = fake_supercell_data.cell_params
        cell_distortion_size = torch.linalg.norm((real_supercell_data.cell_params - self.lattice_means) / self.lattice_stds - (canonical_fake_cell_params - self.lattice_means) / self.lattice_stds, dim=1)

        '''apply noise'''
        if self.config.discriminator_positional_noise > 0:
            real_supercell_data.pos += \
                torch.randn_like(real_supercell_data.pos) * self.config.discriminator_positional_noise
            fake_supercell_data.pos += \
                torch.randn_like(fake_supercell_data.pos) * self.config.discriminator_positional_noise

        '''score'''
        discriminator_output_on_real, real_pairwise_distances_dict, real_latent = self.adversarial_score(real_supercell_data, return_latent=True)
        discriminator_output_on_fake, fake_pairwise_distances_dict, fake_latent = self.adversarial_score(fake_supercell_data, return_latent=True)

        '''recompute packing coeffs'''
        real_packing_coeffs = compute_packing_coefficient(cell_params=real_supercell_data.cell_params,
                                                          mol_volumes=real_supercell_data.mol_volume,
                                                          crystal_multiplicity=real_supercell_data.mult)
        fake_packing_coeffs = compute_packing_coefficient(cell_params=fake_supercell_data.cell_params,
                                                          mol_volumes=fake_supercell_data.mol_volume,
                                                          crystal_multiplicity=fake_supercell_data.mult)

        '''distances'''
        if self.config.discriminator.use_rdf_distance_loss:
            real_rdf, rr, _ = new_crystal_rdf(real_supercell_data, real_pairwise_distances_dict,
                                              rrange=[0, self.config.discriminator.model.convolution_cutoff],
                                              bins=2000, raw_density=True, elementwise=True, mode='intermolecular', cpu_detach=False)
            fake_rdf, _, _ = new_crystal_rdf(fake_supercell_data, fake_pairwise_distances_dict,
                                             rrange=[0, self.config.discriminator.model.convolution_cutoff],
                                             bins=2000, raw_density=True, elementwise=True, mode='intermolecular', cpu_detach=False)

            rdf_dists = torch.zeros(real_supercell_data.num_graphs, device=self.config.device, dtype=torch.float32)
            for i in range(real_supercell_data.num_graphs):
                rdf_dists[i] = compute_rdf_distance(real_rdf[i], fake_rdf[i], rr) / real_supercell_data.mol_size[i]  # divides out the trivial size correlation
        else:
            rdf_dists = torch.randn(real_supercell_data.num_graphs, device=self.config.device, dtype=torch.float32).abs()  # dummy

        stats_keys = ['real_vdw_penalty',
                      'fake_vdw_penalty',
                      'generated_cell_parameters', 'final_generated_cell_parameters',
                      'real_packing_coefficients', 'generated_packing_coefficients']
        stats_values = [-vdw_overlap(self.vdw_radii, crystaldata=real_supercell_data, return_score_only=True).cpu().detach().numpy(),
                        -vdw_overlap(self.vdw_radii, crystaldata=fake_supercell_data, return_score_only=True).cpu().detach().numpy(),
                        generated_samples_i.cpu().detach().numpy(), canonical_fake_cell_params.cpu().detach().numpy(),
                        real_packing_coeffs.cpu().detach().numpy(), fake_packing_coeffs.cpu().detach().numpy()]

        self.logger.update_stats_dict(self.epoch_type, stats_keys, stats_values, mode='extend')

        return (discriminator_output_on_real, discriminator_output_on_fake,
                cell_distortion_size, rdf_dists)

    def set_molecule_alignment(self, data, right_handed=False, mode_override=None):
        """
        set the position and orientation of the molecule with respect to the xyz axis
        'standardized' sets the molecule principal inertial axes equal to the xyz axis
        'random' sets a random orientation of the molecule
        in any case, the molecule centroid is set at (0,0,0)

        option to preserve the handedness of the molecule, e.g., by aligning with
        (x,y,-z) for a left-handed molecule
        """
        if mode_override is not None:
            mode = mode_override
        else:
            mode = self.config.generator.canonical_conformer_orientation

        if mode == 'standardized':
            data = align_crystaldata_to_principal_axes(data, handedness=data.asym_unit_handedness)
            # data.asym_unit_handedness = torch.ones_like(data.asym_unit_handedness)

        elif mode == 'random':
            data = random_crystaldata_alignment(data)
            if right_handed:
                coords_list = [data.pos[data.ptr[i]:data.ptr[i + 1]] for i in range(data.num_graphs)]
                coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(data.num_graphs)]
                principal_axes_list, _, _ = batch_molecule_principal_axes_torch(coords_list_centred)
                handedness = compute_Ip_handedness(principal_axes_list)
                for ind, hand in enumerate(handedness):
                    if hand == -1:
                        data.pos[data.batch == ind] = -data.pos[data.batch == ind]  # invert

                data.asym_unit_handedness = torch.ones_like(data.asym_unit_handedness)
        elif mode == 'as is':
            pass  # do nothing

        return data

    def get_generator_samples(self, data, alignment_override=None):
        """
        set conformer orientation, optionally add noise, set the space group & symmetry information
        optionally get the predicted density from a regression model
        pass to generator and get cell parameters
        """
        mol_data = data.clone()
        # conformer orientation setting
        mol_data = self.set_molecule_alignment(mol_data, mode_override=alignment_override)

        # noise injection
        if self.config.generator_positional_noise > 0:
            mol_data.pos += torch.randn_like(mol_data.pos) * self.config.generator_positional_noise

        # update symmetry information
        if self.config.generate_sgs is not None:
            mol_data = update_crystal_symmetry_elements(mol_data, self.config.generate_sgs, self.sym_info, randomize_sgs=True)

        # update packing coefficient
        if self.config.regressor_path is not None:  # todo ensure we have a regressor predicting the right thing here - i.e., cell_volume vs packing coeff
            # predict the crystal density and feed it as an input to the generator
            with torch.no_grad():
                standardized_target_packing_coeff = self.regressor(mol_data.clone().detach().to(self.config.device)).detach()[:, 0]
        else:
            target_packing_coeff = mol_data.tracking[:, self.t_i_d['crystal_packing_coefficient']]
            standardized_target_packing_coeff = ((target_packing_coeff - self.std_dict['crystal_packing_coefficient'][0]) / self.std_dict['crystal_packing_coefficient'][1]).to(self.config.device)

        standardized_target_packing_coeff += torch.randn_like(standardized_target_packing_coeff) * self.config.generator.packing_target_noise

        # generate the samples
        [generated_samples, prior, condition, raw_samples] = self.generator.forward(
            n_samples=mol_data.num_graphs, molecule_data=mol_data.to(self.config.device).clone(),
            return_condition=True, return_prior=True, return_raw_samples=True,
            target_packing=standardized_target_packing_coeff)

        return generated_samples, prior, standardized_target_packing_coeff, mol_data, raw_samples

    def get_generator_losses(self, data):
        """
        generate samples & score them
        """

        """get crystals"""
        generated_samples, prior, standardized_target_packing, generator_data, raw_samples = (
            self.get_generator_samples(data))

        supercell_data, generated_cell_volumes = (
            self.supercell_builder.build_supercells(
                generator_data, generated_samples, self.config.supercell_size,
                self.config.discriminator.model.convolution_cutoff,
                align_to_standardized_orientation=False
            ))

        """get losses"""
        similarity_penalty = self.compute_similarity_penalty(generated_samples, prior, raw_samples)
        discriminator_raw_output, dist_dict = self.score_adversarially(supercell_data)
        h_bond_score = self.compute_h_bond_score(supercell_data)
        vdw_loss, vdw_score, _, _, _ = vdw_overlap(self.vdw_radii,
                                                   dist_dict=dist_dict,
                                                   num_graphs=generator_data.num_graphs,
                                                   graph_sizes=generator_data.mol_size,
                                                   loss_func=self.config.generator.vdw_loss_func)
        packing_loss, packing_prediction, packing_target, packing_csd = \
            self.generator_density_matching_loss(
                standardized_target_packing, supercell_data, generated_samples,
                precomputed_volumes=generated_cell_volumes, loss_func=self.config.generator.density_loss_func)

        return discriminator_raw_output, generated_samples.cpu().detach().numpy(), raw_samples.cpu().detach().numpy(), \
            packing_loss, packing_prediction.cpu().detach().numpy(), \
            packing_target.cpu().detach().numpy(), \
            vdw_loss, vdw_score, dist_dict, \
            supercell_data, similarity_penalty, h_bond_score

    def misc_pre_training_items(self):
        """
        dataset_builder: for going from database to trainable dataset
        dataDims: contains key information about the dataset
        number of generators for discriminator training
        supercell_builder
        tracking indices for certain properties
        symmetry element indexing
        multivariate gaussian generator
        """
        # todo reconsider the need for this function
        ''' 
        init gaussian generator for cell parameter sampling
        we don't always use it but it's very cheap so just do it every time
        '''
        self.gaussian_generator = independent_gaussian_model(input_dim=self.dataDims['num_lattice_features'],
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

        generator_ind = generator_ind_list[int(gen_randint)]  # randomly select which generator to use from the available set

        return n_generators, generator_ind

    def generate_discriminator_negatives(self, real_data, i, override_adversarial=False, override_randn=False, override_distorted=False, orientation='random'):
        """
        use one of the available cell generation tools to sample cell parameters, to be fed to the discriminator
        """
        n_generators, generator_ind = self.what_generators_to_use(override_randn, override_distorted, override_adversarial)

        if (self.config.discriminator.train_adversarially or override_adversarial) and (generator_ind == 1):
            negative_type = 'generator'
            with torch.no_grad():
                generated_samples, _, _, generator_data, _ = self.get_generator_samples(real_data, alignment_override=orientation)

            self.logger.update_stats_dict(self.epoch_type, 'generator_sample_source', np.zeros(len(generated_samples)), mode='extend')

        elif (self.config.discriminator.train_on_randn or override_randn) and (generator_ind == 2):
            generator_data = self.set_molecule_alignment(real_data.clone(), mode_override=orientation)
            negative_type = 'randn'
            generated_samples = self.gaussian_generator.forward(real_data.num_graphs, real_data).to(self.config.device)

            self.logger.update_stats_dict(self.epoch_type, 'generator_sample_source', np.ones(len(generated_samples)), mode='extend')

        elif (self.config.discriminator.train_on_distorted or override_distorted) and (generator_ind == 3):
            generator_data = self.set_molecule_alignment(real_data.clone(), mode_override=orientation)
            negative_type = 'distorted'

            generated_samples, distortion = self.make_distorted_samples(real_data)

            self.logger.update_stats_dict(self.epoch_type, 'generator_sample_source', 2 * np.ones(len(generated_samples)), mode='extend')
            self.logger.update_stats_dict(self.epoch_type, 'distortion_level',
                                          torch.linalg.norm(distortion, axis=-1).cpu().detach().numpy(),
                                          mode='extend')
        else:
            print("No Generators set to make discriminator negatives!")
            assert False

        generator_data.cell_params = generated_samples

        return generated_samples.float().detach(), negative_type, generator_data

    def make_distorted_samples(self, real_data, distortion_override=None):
        """
        given some cell params
        standardize them
        add noise in the standarized basis
        destandardize
        make sure samples are appropriately cleaned
        """
        generated_samples_std = (real_data.cell_params - self.lattice_means) / self.lattice_stds

        if distortion_override is not None:
            distortion = torch.randn_like(generated_samples_std) * distortion_override
        else:
            if self.config.discriminator.distortion_magnitude == -1:
                distortion = torch.randn_like(generated_samples_std) * torch.logspace(-2, 1, len(generated_samples_std)).to(generated_samples_std.device)[:, None]  # wider range
            else:
                distortion = torch.randn_like(generated_samples_std) * self.config.discriminator.distortion_magnitude

        distorted_samples_std = (generated_samples_std + distortion).to(self.config.device)  # add jitter and return in standardized basis

        distorted_samples_clean = clean_cell_params(
            distorted_samples_std, real_data.sg_ind,
            self.lattice_means, self.lattice_stds,
            self.sym_info, self.supercell_builder.asym_unit_dict,
            rescale_asymmetric_unit=False, destandardize=True, mode='hard')

        return distorted_samples_clean, distortion

    def increment_batch_size(self, train_loader, test_loader, extra_test_loader):
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
        return train_loader, test_loader, extra_test_loader

    def model_checkpointing(self, epoch):
        if self.train_discriminator:
            model = 'discriminator'
            loss_record = self.logger.loss_record[model]['mean_test']
            past_mean_losses = [np.mean(record) for record in loss_record]
            if np.average(self.logger.current_losses[model]['mean_test']) == np.amin(past_mean_losses):
                print("Saving discriminator checkpoint")
                self.logger.save_stats_dict(prefix='best_discriminator_')
                save_checkpoint(epoch, self.discriminator, self.discriminator_optimizer, self.config.discriminator.__dict__,
                                self.config.checkpoint_dir_path + 'best_discriminator' + self.run_identifier)

        if self.train_generator:
            model = 'generator'
            loss_record = self.logger.loss_record[model]['mean_test']
            past_mean_losses = [np.mean(record) for record in loss_record]
            if np.average(self.logger.current_losses[model]['mean_test']) == np.amin(past_mean_losses):
                print("Saving generator checkpoint")
                self.logger.save_stats_dict(prefix='best_generator_')
                save_checkpoint(epoch, self.generator, self.generator_optimizer, self.config.generator.__dict__,
                                self.config.checkpoint_dir_path + 'best_generator' + self.run_identifier)

        if self.train_regressor:
            model = 'regressor'
            loss_record = self.logger.loss_record[model]['mean_test']
            past_mean_losses = [np.mean(record) for record in loss_record]
            if np.average(self.logger.current_losses[model]['mean_test']) == np.amin(past_mean_losses):
                print("Saving regressor checkpoint")
                self.logger.save_stats_dict(prefix='best_regressor_')
                save_checkpoint(epoch, self.regressor, self.regressor_optimizer, self.config.regressor.__dict__,
                                self.config.checkpoint_dir_path + 'best_regressor' + self.run_identifier)

        # todo checkpointing for proxy discriminator


    def update_lr(self):
        self.discriminator_optimizer, discriminator_lr = set_lr(self.discriminator_schedulers, self.discriminator_optimizer, self.config.discriminator.optimizer.lr_schedule, self.config.discriminator.optimizer.min_lr,
                                                                self.logger.current_losses['discriminator']['mean_train'], self.discriminator_hit_max_lr)

        self.generator_optimizer, generator_lr = set_lr(self.generator_schedulers, self.generator_optimizer, self.config.generator.optimizer.lr_schedule, self.config.generator.optimizer.min_lr,
                                                        self.logger.current_losses['generator']['mean_train'], self.generator_hit_max_lr)

        self.regressor_optimizer, regressor_lr = set_lr(self.regressor_schedulers, self.regressor_optimizer, self.config.regressor.optimizer.lr_schedule, self.config.regressor.optimizer.min_lr,
                                                        self.logger.current_losses['regressor']['mean_train'], self.regressor_hit_max_lr)

        self.proxy_discriminator_optimizer, proxy_discriminator_lr = set_lr(self.proxy_discriminator_schedulers, self.proxy_discriminator_optimizer, self.config.proxy_discriminator.optimizer.lr_schedule, self.config.proxy_discriminator.optimizer.min_lr,
                                                                            self.logger.current_losses['proxy_discriminator']['mean_train'], self.proxy_discriminator_hit_max_lr)

        discriminator_learning_rate = self.discriminator_optimizer.param_groups[0]['lr']
        if discriminator_learning_rate >= self.config.discriminator.optimizer.max_lr:
            self.discriminator_hit_max_lr = True
        generator_learning_rate = self.generator_optimizer.param_groups[0]['lr']
        if generator_learning_rate >= self.config.generator.optimizer.max_lr:
            self.generator_hit_max_lr = True
        regressor_learning_rate = self.regressor_optimizer.param_groups[0]['lr']
        if regressor_learning_rate >= self.config.regressor.optimizer.max_lr:
            self.regressor_hit_max_lr = True
        proxy_discriminator_learning_rate = self.proxy_discriminator_optimizer.param_groups[0]['lr']
        if proxy_discriminator_learning_rate >= self.config.proxy_discriminator.optimizer.max_lr:
            self.proxy_discriminator_hit_max_lr = True
        (self.logger.learning_rates['discriminator'], self.logger.learning_rates['generator'],
         self.logger.learning_rates['regressor'], self.logger.learning_rates['proxy_discriminator']) = (
            discriminator_learning_rate, generator_learning_rate, regressor_learning_rate, proxy_discriminator_learning_rate)

    def reload_best_test_checkpoint(self, epoch):
        # reload best test
        if epoch != 0:  # if we have trained at all, reload the best model
            generator_path = f'../models/best_generator_{self.run_identifier}'
            discriminator_path = f'../models/best_discriminator_{self.run_identifier}'

            if os.path.exists(generator_path):
                generator_checkpoint = torch.load(generator_path)
                if list(generator_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                    for i in list(generator_checkpoint['model_state_dict']):
                        generator_checkpoint['model_state_dict'][i[7:]] = generator_checkpoint['model_state_dict'].pop(i)
                self.generator.load_state_dict(generator_checkpoint['model_state_dict'])

            if os.path.exists(discriminator_path):
                discriminator_checkpoint = torch.load(discriminator_path)
                if list(discriminator_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                    for i in list(discriminator_checkpoint['model_state_dict']):
                        discriminator_checkpoint['model_state_dict'][i[7:]] = discriminator_checkpoint['model_state_dict'].pop(i)
                self.discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])

    def gan_evaluation(self, epoch, test_loader, extra_test_loader):
        """
        run post-training evaluation
        """
        self.reload_best_test_checkpoint(epoch)
        self.logger.reset_for_new_epoch(epoch, test_loader.batch_size)

        # rerun test inference
        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():  # todo add proxy evaluation when finished
            if self.train_discriminator:
                self.run_epoch(epoch_type='test', data_loader=test_loader, update_gradients=False)  # compute loss on test set

                if extra_test_loader is not None:
                    self.run_epoch(epoch_type='extra', data_loader=extra_test_loader, update_gradients=False)  # compute loss on test set

            # sometimes test the generator on a mini CSP problem
            if (self.config.mode == 'gan') and self.train_generator:
                pass  # self.batch_csp(extra_test_loader if extra_test_loader is not None else test_loader)

        self.logger.log_epoch_analysis(test_loader)

        return None

    def compute_similarity_penalty(self, generated_samples, prior, raw_samples):
        """
        punish batches in which the samples are too self-similar
        or on the basis of their statistics
        Parameters
        ----------
        generated_samples
        prior

        Returns
        -------
        """
        if len(generated_samples) >= 3:
            # enforce that the distance between samples is similar to the distance between priors
            # prior_dists = torch.cdist(prior, prior, p=2)
            # std_samples = (generated_samples - self.lattice_means) / self.lattice_stds
            # sample_dists = torch.cdist(std_samples, std_samples, p=2)
            # prior_distance_penalty = F.smooth_l1_loss(input=sample_dists, target=prior_dists, reduction='none').mean(1)  # align distances to all other samples
            #
            # prior_variance = prior.var(dim=0)
            # sample_variance = std_samples.var(dim=0)
            # variance_penalty = F.smooth_l1_loss(input=sample_variance, target=prior_variance, reduction='none').mean().tile(len(prior))

            # similarity_penalty = (prior_distance_penalty + variance_penalty)
            sample_stds = generated_samples[:, [0, 1, 2, 6, 7, 8]].std(0)
            sample_means = generated_samples[:, [0, 1, 2, 6, 7, 8]].mean(0)

            raw_sample_stds = raw_samples[:, [3, 4, 5, 9, 10, 11]].std(0)
            raw_sample_means = raw_samples[:, [3, 4, 5, 9, 10, 11]].mean(0)

            # enforce similar distribution
            standardization_losses = (F.smooth_l1_loss(sample_stds, self.lattice_stds[[0, 1, 2, 6, 7, 8]]) + F.smooth_l1_loss(sample_means, self.lattice_means[[0, 1, 2, 6, 7, 8]]) + \
                                      F.smooth_l1_loss(raw_sample_stds, self.lattice_stds[[3, 4, 5, 9, 10, 11]]) + F.smooth_l1_loss(raw_sample_means, self.lattice_means[[3, 4, 5, 9, 10, 11]]))
            # enforce similar range of fractional centroids
            mins = raw_samples[:, 6:9].amin(0)
            maxs = raw_samples[:, 6:9].amax(0)
            frac_range_losses = F.smooth_l1_loss(mins, torch.zeros_like(mins)) + F.smooth_l1_loss(maxs, torch.ones_like(maxs))

            similarity_penalty = (standardization_losses + frac_range_losses).tile(len(prior))
        else:
            similarity_penalty = None

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
            if self.config.discriminator_positional_noise > 0:
                supercell_data.pos += torch.randn_like(
                    supercell_data.pos) * self.config.discriminator_positional_noise

        if (self.config.device.lower() == 'cuda') and (supercell_data.x.device != 'cuda'):
            supercell_data = supercell_data.cuda()

        discriminator_score, dist_dict, latent = self.adversarial_score(supercell_data, return_latent=True)

        if return_latent:
            return discriminator_score, dist_dict, latent
        else:
            return discriminator_score, dist_dict

    def aggregate_generator_losses(self, packing_loss, discriminator_raw_output, vdw_loss, vdw_score,
                                   similarity_penalty, packing_prediction, packing_target, h_bond_score):
        generator_losses_list = []
        stats_keys, stats_values = [], []
        if packing_loss is not None:
            packing_mae = np.abs(packing_prediction - packing_target) / packing_target

            if packing_mae.mean() < (0.02 + self.config.generator.packing_target_noise):  # dynamically soften the packing loss when the model is doing well
                self.packing_loss_coefficient *= 0.99
            if (packing_mae.mean() > (0.03 + self.config.generator.packing_target_noise)) and (self.packing_loss_coefficient < 100):
                self.packing_loss_coefficient *= 1.01

            self.logger.packing_loss_coefficient = self.packing_loss_coefficient

            stats_keys += ['generator_packing_loss', 'generator_packing_prediction',
                           'generator_packing_target', 'generator_packing_mae']
            stats_values += [packing_loss.cpu().detach().numpy() * self.packing_loss_coefficient, packing_prediction,
                             packing_target, packing_mae]

            if True:  # enforce the target density all the time
                generator_losses_list.append(packing_loss.float() * self.packing_loss_coefficient)

        if discriminator_raw_output is not None:
            if self.config.generator.adversarial_loss_func == 'hot softmax':
                adversarial_loss = 1 - F.softmax(discriminator_raw_output / 5, dim=1)[:, 1]  # high temp smears out the function over a wider range
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
            stats_values += [adversarial_loss.cpu().detach().numpy()]
            stats_keys += ['generator_adversarial_score']
            stats_values += [adversarial_score.cpu().detach().numpy()]

            if self.config.generator.train_adversarially:
                generator_losses_list.append(adversarial_loss)

        if vdw_loss is not None:
            stats_keys += ['generator_per_mol_vdw_loss', 'generator_per_mol_vdw_score']
            stats_values += [vdw_loss.cpu().detach().numpy()]
            stats_values += [vdw_score.cpu().detach().numpy()]

            if self.config.generator.train_vdw:
                generator_losses_list.append(vdw_loss)

        if h_bond_score is not None:
            if self.config.generator.train_h_bond:
                generator_losses_list.append(h_bond_score)

            stats_keys += ['generator h bond loss']
            stats_values += [h_bond_score.cpu().detach().numpy()]

        if similarity_penalty is not None:
            stats_keys += ['generator similarity loss']
            stats_values += [similarity_penalty.cpu().detach().numpy()]

            if self.config.generator.similarity_penalty != 0:
                if similarity_penalty is not None:
                    generator_losses_list.append(self.config.generator.similarity_penalty * similarity_penalty)
                else:
                    print('similarity penalty was none')

        generator_losses = torch.sum(torch.stack(generator_losses_list), dim=0)
        self.logger.update_stats_dict(self.epoch_type, stats_keys, stats_values, mode='extend')

        return generator_losses

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
        if self.config.generator_path is None:
            generator.apply(weight_reset)
        if self.config.discriminator_path is None:
            discriminator.apply(weight_reset)
        if self.config.regressor_path is None:
            regressor.apply(weight_reset)

        return generator, discriminator, regressor

    def reload_model_checkpoints(self):
        if self.config.generator_path is not None:
            generator_checkpoint = torch.load(self.config.generator_path)
            generator_config = Namespace(**generator_checkpoint['config'])  # overwrite the settings for the model
            self.config.generator.optimizer = generator_config.optimizer
            self.config.generator.model = generator_config.model

        if self.config.discriminator_path is not None:
            discriminator_checkpoint = torch.load(self.config.discriminator_path)
            discriminator_config = Namespace(**discriminator_checkpoint['config'])  # overwrite the settings for the model
            self.config.discriminator.optimizer = discriminator_config.optimizer
            self.config.discriminator.model = discriminator_config.model

        if self.config.regressor_path is not None:
            regressor_checkpoint = torch.load(self.config.regressor_path)
            regressor_config = Namespace(**regressor_checkpoint['config'])  # overwrite the settings for the model
            self.config.regressor.optimizer = regressor_config.optimizer
            self.config.regressor.model = regressor_config.model

        return self.config

    def crystal_search(self, molecule_data, batch_size, data_contains_ground_truth=True):
        """
        execute a search for a single crystal target
        if the target is known, compare it to our best guesses
        """
        num_discriminator_opt_steps = 100
        num_mcmc_opt_steps = 100
        self.generator.eval()
        self.regressor.eval()
        self.discriminator.eval()

        '''instantiate batch'''
        collater = Collater(None, None)
        crystaldata_batch = collater([molecule_data for _ in range(batch_size)]).to(self.device)
        refresh_inds = torch.arange(batch_size)
        max_iters = 10
        converged_samples_list = []
        opt_trajectories = []
        for opt_iter in range(max_iters):
            crystaldata_batch = self.refresh_crystal_batch(crystaldata_batch, refresh_inds=refresh_inds)

            crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(crystaldata_batch, mode='discriminator', num_steps=num_discriminator_opt_steps)
            opt_trajectories.append(opt_traj)
            crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(crystaldata_batch, mode='mcmc', num_steps=num_mcmc_opt_steps)
            opt_trajectories.append(opt_traj)
            crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(crystaldata_batch, mode='discriminator', num_steps=num_discriminator_opt_steps)
            opt_trajectories.append(opt_traj)

            crystaldata_batch, refresh_inds, converged_samples = self.prune_crystaldata_batch(crystaldata_batch)

            converged_samples_list.append(converged_samples)

            # add convergence flags based on completeness of sampling

        '''compare samples to ground truth'''
        if data_contains_ground_truth:
            ground_truth_analysis = self.analyze_real_crystal(molecule_data)

    def optimize_crystaldata_batch(self, batch, mode, num_steps):
        """
        method which takes a batch of crystaldata objects
        and optimzies them according to a score model either
        with MCMC or gradient descent
        """
        num_crystals = batch.num_graphs
        traj_record = {'score': np.zeros((num_steps, num_crystals)),
                       'vdw_score': np.zeros((num_steps, num_crystals)),
                       'std_cell_params': np.zeros((num_steps, num_crystals, 12)),
                       'space_groups': np.zeros((num_steps, num_crystals)),
                       'rdf': [[] for _ in range(num_crystals)]
                       }

        if mode.lower() == 'mcmc':
            # todo build a clean MCMC here
            assert False
        elif mode.lower() == 'discriminator':
            batch, traj_record = gradient_descent_sampling(
                self.discriminator, batch,
                self.supercell_builder,
                num_steps, 1e-3,
                torch.optim.Rprop, self.vdw_radii,
                supercell_size=5, cutoff=6)

        traj_record['rdf'] = crystal_rdf(batch, mode='intermolecular', elementwise=True, cpu_detach=True, raw_density=True)

        return batch, traj_record

    def refresh_crystal_batch(self, batch, refresh_inds, generator='gaussian', mol_orientation='random', space_groups: torch.tensor = None):
        batch = self.set_molecule_alignment(batch, right_handed=False, mode_override=mol_orientation)

        if space_groups is not None:
            batch.sg_ind = space_groups

        if generator == 'gaussian':
            samples = self.gaussian_generator.forward(batch.num_graphs, batch).to(self.config.device)
            batch.cell_params = samples[refresh_inds]
            # todo add option for generator here

        return batch

    #
    # def batch_csp(self, data_loader):
    #     print('Starting Mini CSP')
    #     self.generator.eval()
    #     self.discriminator.eval()
    #     rdf_bins, rdf_range = 100, [0, 10]
    #
    #
    #     if self.config.target_identifiers is not None:  # analyse one or more particular crystals
    #         identifiers = [data_loader.dataset[ind].csd_identifier for ind in range(len(data_loader.dataset))]
    #         for i, identifier in enumerate(identifiers):
    #             '''prep data'''
    #             collater = Collater(None, None)
    #             real_data = collater([data_loader.dataset[i]]).to(self.config.device)
    #             real_data_for_sampling = collater([data_loader.dataset[i] for _ in range(data_loader.batch_size)]).to(self.config.device)  #
    #             real_samples_dict, real_supercell_data = self.analyze_real_crystals(real_data, rdf_bins, rdf_range)
    #             num_crystals, num_samples = 1, self.config.sample_steps
    #
    #             '''do sampling'''
    #             generated_samples_dict, rr = self.generate_mini_csp_samples(real_data_for_sampling, rdf_range, rdf_bins)
    #             # results from batch to single array format
    #             for key in generated_samples_dict.keys():
    #                 if not isinstance(generated_samples_dict[key], list):
    #                     generated_samples_dict[key] = np.concatenate(generated_samples_dict[key], axis=0)[None, ...]
    #                 elif isinstance(generated_samples_dict[key], list):
    #                     generated_samples_dict[key] = [[generated_samples_dict[key][i2][i1] for i1 in range(num_samples) for i2 in range(real_data_for_sampling.num_graphs)]]
    #
    #             '''results summary'''
    #             log_mini_csp_scores_distributions(self.config, wandb, generated_samples_dict, real_samples_dict, real_data, self.sym_info)
    #             log_csp_summary_stats(wandb, generated_samples_dict, self.sym_info)
    #             log_csp_cell_params(self.config, wandb, generated_samples_dict, real_samples_dict, identifier, crystal_ind=0)
    #
    #             '''compute intra-crystal and crystal-target distances'''
    #             real_dists_dict, intra_dists_dict = compute_csp_sample_distances(self.config, real_samples_dict, generated_samples_dict, num_crystals, num_samples * real_data_for_sampling.num_graphs, rr)
    #
    #             plot_mini_csp_dist_vs_score(real_dists_dict['real_sample_rdf_distance'],
    #                                         real_dists_dict['real_sample_cell_distance'],
    #                                         real_dists_dict['real_sample_latent_distance'],
    #                                         generated_samples_dict, real_samples_dict, wandb)
    #
    #             sample_density_funnel_plot(self.config, wandb, num_crystals, identifier, generated_samples_dict, real_samples_dict)
    #             sample_rdf_funnel_plot(self.config, wandb, num_crystals, identifier, generated_samples_dict['score'], real_samples_dict, real_dists_dict['real_sample_rdf_distance'])
    #
    #             '''cluster and identify interesting samples, then optimize them'''
    #             aa = 1
    #
    #     else:  # otherwise, a random batch from the dataset
    #         collater = Collater(None, None)
    #         real_data = collater(data_loader.dataset[0:min(50, len(data_loader.dataset))]).to(self.config.device)  # take a fixed number of samples
    #         real_samples_dict, real_supercell_data = self.analyze_real_crystals(real_data, rdf_bins, rdf_range)
    #         num_samples = self.config.sample_steps
    #         num_crystals = real_data.num_graphs
    #
    #         '''do sampling'''
    #         generated_samples_dict, rr = self.generate_mini_csp_samples(real_data, rdf_range, rdf_bins)
    #
    #         '''results summary'''
    #         log_mini_csp_scores_distributions(self.config, wandb, generated_samples_dict, real_samples_dict, real_data, self.sym_info)
    #         log_csp_summary_stats(wandb, generated_samples_dict, self.sym_info)
    #         for ii in range(real_data.num_graphs):
    #             log_csp_cell_params(self.config, wandb, generated_samples_dict, real_samples_dict, real_data.csd_identifier[ii], crystal_ind=ii)
    #
    #         '''compute intra-crystal and crystal-target distances'''
    #         real_dists_dict, intra_dists_dict = compute_csp_sample_distances(self.config, real_samples_dict, generated_samples_dict, num_crystals, num_samples, rr)
    #
    #         '''summary distances'''
    #         plot_mini_csp_dist_vs_score(real_dists_dict['real_sample_rdf_distance'],
    #                                     real_dists_dict['real_sample_cell_distance'],
    #                                     real_dists_dict['real_sample_latent_distance'],
    #                                     generated_samples_dict, real_samples_dict, wandb)
    #
    #         '''funnel plots'''
    #         sample_density_funnel_plot(self.config, wandb, num_crystals, real_data.csd_identifier, generated_samples_dict, real_samples_dict)
    #         sample_rdf_funnel_plot(self.config, wandb, num_crystals, real_data.csd_identifier, generated_samples_dict['score'], real_samples_dict, real_dists_dict['real_sample_rdf_distance'])
    #
    #     return None
    #
    # def analyze_real_crystals(self, real_data, rdf_bins, rdf_range):
    #     real_supercell_data = self.supercell_builder.prebuilt_unit_cell_to_supercell(real_data, self.config.supercell_size, self.config.discriminator.model.convolution_cutoff)
    #
    #     discriminator_score, dist_dict, discriminator_latent = self.score_adversarially(real_supercell_data.clone(), self.discriminator, return_latent=True)
    #     h_bond_score = self.compute_h_bond_score(real_supercell_data)
    #     _, vdw_score, _, _ = vdw_overlap(self.vdw_radii,
    #                                      dist_dict=dist_dict,
    #                                      num_graphs=real_data.num_graphs,
    #                                      graph_sizes=real_data.mol_size)
    #
    #     real_rdf, rr, atom_inds = crystal_rdf(real_supercell_data, rrange=rdf_range,
    #                                           bins=rdf_bins, mode='intermolecular',
    #                                           raw_density=True, atomwise=True, cpu_detach=True)
    #
    #     volumes_list = []
    #     for i in range(real_data.num_graphs):
    #         volumes_list.append(cell_vol_torch(real_data.cell_params[i, 0:3], real_data.cell_params[i, 3:6]))
    #     volumes = torch.stack(volumes_list)
    #     real_packing_coeffs = real_data.mult * real_data.mol_volume / volumes
    #
    #     real_samples_dict = {'score': softmax_and_score(discriminator_score).cpu().detach().numpy(),
    #                          'vdw overlap': -vdw_score.cpu().detach().numpy(),
    #                          'density': real_packing_coeffs.cpu().detach().numpy(),
    #                          'h bond score': h_bond_score.cpu().detach().numpy(),
    #                          'cell params': real_data.cell_params.cpu().detach().numpy(),
    #                          'space group': real_data.sg_ind.cpu().detach().numpy(),
    #                          'RDF': real_rdf,
    #                          'discriminator latent': discriminator_latent,
    #                          }
    #
    #     return real_samples_dict, real_supercell_data.cpu()
    #
    # def generate_mini_csp_samples(self, real_data, rdf_range, rdf_bins, sample_source='generator'):
    #     num_molecules = real_data.num_graphs
    #     n_sampling_iters = self.config.sample_steps
    #     sampling_dict = {'score': np.zeros((num_molecules, n_sampling_iters)),
    #                      'vdw overlap': np.zeros((num_molecules, n_sampling_iters)),
    #                      'density': np.zeros((num_molecules, n_sampling_iters)),
    #                      'h bond score': np.zeros((num_molecules, n_sampling_iters)),
    #                      'cell params': np.zeros((num_molecules, n_sampling_iters, 12)),
    #                      'space group': np.zeros((num_molecules, n_sampling_iters)),
    #                      'handedness': np.zeros((num_molecules, n_sampling_iters)),
    #                      'distortion_size': np.zeros((num_molecules, n_sampling_iters)),
    #                      'discriminator latent': np.zeros((num_molecules, n_sampling_iters, self.config.discriminator.fc_depth)),
    #                      'RDF': [[] for _ in range(num_molecules)]
    #                      }
    #
    #     with torch.no_grad():
    #         for ii in tqdm.tqdm(range(n_sampling_iters)):
    #             fake_data = real_data.clone().to(self.config.device)
    #
    #             if sample_source == 'generator':
    #                 # use generator to make samples
    #                 samples, prior, standardized_target_packing_coeff, fake_data = \
    #                     self.get_generator_samples(fake_data)
    #
    #                 fake_supercell_data, generated_cell_volumes = \
    #                     self.supercell_builder.build_supercells(
    #                         fake_data, samples, self.config.supercell_size,
    #                         self.config.discriminator.model.convolution_cutoff,
    #                         align_to_standardized_orientation=False,
    #                     )
    #
    #             elif sample_source == 'distorted':
    #                 # test - do slight distortions on existing crystals
    #                 generated_samples_ii = (real_data.cell_params - self.lattice_means) / self.lattice_stds
    #
    #                 if True:  # self.config.discriminator.distortion_magnitude == -1:
    #                     distortion = torch.randn_like(generated_samples_ii) * torch.logspace(-4, 1, len(generated_samples_ii)).to(generated_samples_ii.device)[:, None]  # wider range
    #                     distortion = distortion[torch.randperm(len(distortion))]
    #                 else:
    #                     distortion = torch.randn_like(generated_samples_ii) * self.config.discriminator.distortion_magnitude
    #
    #                 generated_samples_i_d = (generated_samples_ii + distortion).to(self.config.device)  # add jitter and return in standardized basis
    #
    #                 generated_samples_i = clean_cell_params(
    #                     generated_samples_i_d, real_data.sg_ind,
    #                     self.lattice_means, self.lattice_stds,
    #                     self.sym_info, self.supercell_builder.asym_unit_dict,
    #                     rescale_asymmetric_unit=False, destandardize=True, mode='hard')
    #
    #                 fake_supercell_data, generated_cell_volumes = self.supercell_builder.build_supercells(
    #                     fake_data, generated_samples_i, self.config.supercell_size,
    #                     self.config.discriminator.model.convolution_cutoff,
    #                     align_to_standardized_orientation=True,
    #                     target_handedness=real_data.asym_unit_handedness,
    #                 )
    #                 sampling_dict['distortion_size'][:, ii] = torch.linalg.norm(distortion, axis=-1).cpu().detach().numpy()
    #                 # end test
    #
    #             generated_rdf, rr, atom_inds = crystal_rdf(fake_supercell_data, rrange=rdf_range,
    #                                                        bins=rdf_bins, mode='intermolecular',
    #                                                        raw_density=True, atomwise=True, cpu_detach=True)
    #             discriminator_score, dist_dict, discriminator_latent = self.score_adversarially(fake_supercell_data.clone(), discriminator_noise=0, return_latent=True)
    #             h_bond_score = self.compute_h_bond_score(fake_supercell_data)
    #             vdw_score = vdw_overlap(self.vdw_radii,
    #                                     dist_dict=dist_dict,
    #                                     num_graphs=fake_data.num_graphs,
    #                                     graph_sizes=fake_data.mol_size,
    #                                     return_score_only=True)
    #
    #             volumes_list = []
    #             for i in range(fake_data.num_graphs):
    #                 volumes_list.append(
    #                     cell_vol_torch(fake_supercell_data.cell_params[i, 0:3], fake_supercell_data.cell_params[i, 3:6]))
    #             volumes = torch.stack(volumes_list)
    #
    #             fake_packing_coeffs = fake_supercell_data.mult * fake_supercell_data.mol_volume / volumes
    #
    #             sampling_dict['score'][:, ii] = softmax_and_score(discriminator_score).cpu().detach().numpy()
    #             sampling_dict['vdw overlap'][:, ii] = -vdw_score.cpu().detach().numpy()
    #             sampling_dict['density'][:, ii] = fake_packing_coeffs.cpu().detach().numpy()
    #             sampling_dict['h bond score'][:, ii] = h_bond_score.cpu().detach().numpy()
    #             sampling_dict['cell params'][:, ii, :] = fake_supercell_data.cell_params.cpu().detach().numpy()
    #             sampling_dict['space group'][:, ii] = fake_supercell_data.sg_ind.cpu().detach().numpy()
    #             sampling_dict['handedness'][:, ii] = fake_supercell_data.asym_unit_handedness.cpu().detach().numpy()
    #             sampling_dict['discriminator latent'][:, ii, :] = discriminator_latent
    #             for jj in range(num_molecules):
    #                 sampling_dict['RDF'][jj].append(generated_rdf[jj])
    #
    #     return sampling_dict, rr

    def compute_h_bond_score(self, supercell_data=None):
        if (supercell_data is not None) and ('atom_is_H_bond_donor' in self.dataDims['atom_features']) and (
                'molecule_num_donors' in self.dataDims['molecule_features']):  # supercell_data is not None: # do vdw computation even if we don't need it
            # get the total per-molecule counts
            mol_acceptors = supercell_data.tracking[:, self.t_i_d['molecule_num_acceptors']]
            mol_donors = supercell_data.tracking[:, self.t_i_d['molecule_num_donors']]

            '''
            count pairs within a close enough bubble ~2.7-3.3 Angstroms
            '''
            h_bonds_loss = []
            for i in range(supercell_data.num_graphs):
                if (mol_donors[i]) > 0 and (mol_acceptors[i] > 0):
                    h_bonds = compute_num_h_bonds(supercell_data,
                                                  self.dataDims['atom_features'].index('atom_is_H_bond_acceptor'),
                                                  self.dataDims['atom_features'].index('atom_is_H_bond_donor'), i)

                    bonds_per_possible_bond = h_bonds / min(mol_donors[i], mol_acceptors[i])
                    h_bond_loss = 1 - torch.tanh(2 * bonds_per_possible_bond)  # smoother gradient about 0

                    h_bonds_loss.append(h_bond_loss)
                else:
                    h_bonds_loss.append(torch.zeros(1)[0].to(supercell_data.x.device))
            h_bond_loss_f = torch.stack(h_bonds_loss)
        else:
            h_bond_loss_f = None

        return h_bond_loss_f

    def generator_density_matching_loss(self, standardized_target_packing,
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

        generated_packing_coeffs = data.mult * data.mol_volume / volumes
        standardized_gen_packing_coeffs = (generated_packing_coeffs - self.std_dict['crystal_packing_coefficient'][0]) / self.std_dict['crystal_packing_coefficient'][1]

        target_packing_coeffs = standardized_target_packing * self.std_dict['crystal_packing_coefficient'][1] + self.std_dict['crystal_packing_coefficient'][0]

        csd_packing_coeffs = data.tracking[:, self.t_i_d['crystal_packing_coefficient']]

        # compute loss vs the target
        if loss_func == 'mse':
            packing_loss = F.mse_loss(standardized_gen_packing_coeffs, standardized_target_packing,
                                      reduction='none')  # allow for more error around the minimum
        elif loss_func == 'l1':
            packing_loss = F.smooth_l1_loss(standardized_gen_packing_coeffs, standardized_target_packing,
                                            reduction='none')
        else:
            assert False, "Must pick from the set of implemented packing loss functions 'mse', 'l1'"
        return packing_loss, generated_packing_coeffs, target_packing_coeffs, csd_packing_coeffs

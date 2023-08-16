import glob
import os
import time
from argparse import Namespace
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime

import sys

import torch
import torch.random
import wandb
from torch import backends
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from shutil import copy
from distutils.dir_util import copy_tree

from constants.atom_properties import VDW_RADII, ATOM_WEIGHTS
from constants.asymmetric_units import asym_unit_dict
from constants.space_group_info import (POINT_GROUPS, LATTICE_TYPE, SPACE_GROUPS, SYM_OPS)

from models.discriminator_models import crystal_discriminator
from models.generator_models import crystal_generator, independent_gaussian_model
from models.regression_models import molecule_regressor
from models.utils import (reload_model, init_schedulers, softmax_and_score, compute_packing_coefficient,
                          check_convergence, save_checkpoint, set_lr, cell_vol_torch, init_optimizer)
from models.utils import (compute_h_bond_score, get_vdw_penalty, generator_density_matching_loss, weight_reset, get_n_config)
from models.vdw_overlap import vdw_overlap
from models.crystal_rdf import crystal_rdf

from crystal_building.utils import (random_crystaldata_alignment, align_crystaldata_to_principal_axes,
                                    batch_molecule_principal_axes_torch, compute_Ip_handedness, clean_cell_params)
from crystal_building.builder import SupercellBuilder
from crystal_building.utils import update_crystal_symmetry_elements

from dataset_management.manager import DataManager
from dataset_management.utils import (DatasetBuilder, get_dataloaders, update_dataloader_batch_size, get_extra_test_loader)

from reporting.online import (detailed_reporting)
from csp.utils import log_best_mini_csp_samples
from reporting.csp.utils import log_mini_csp_scores_distributions
from common.utils import (update_stats_dict, np_softmax)


# https://www.ruppweb.org/Xray/tutorial/enantio.htm non enantiogenic groups
# https://dictionary.iucr.org/Sohncke_groups#:~:text=Sohncke%20groups%20are%20the%20three,in%20the%20chiral%20space%20groups.


class Modeller:
    def __init__(self, config):
        self.config = config
        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        '''get some physical constants'''
        self.atom_weights = ATOM_WEIGHTS
        self.vdw_radii = VDW_RADII
        self.prep_symmetry_info()

        '''set space groups to be included and generated'''
        if self.config.generate_sgs == 'all':
            # generate samples in every space group in the asym dict (eventually, all sgs)
            self.config.generate_sgs = [self.space_groups[int(key)] for key in asym_unit_dict.keys()]

        if self.config.include_sgs is None:  # draw from all space groups we can parameterize
            self.config.include_sgs = [self.space_groups[int(key)] for key in asym_unit_dict.keys()]  # list(self.space_groups.values())


        '''prep workdir'''
        self.source_directory = os.getcwd()
        if (self.config.run_num == 0) or (self.config.explicit_run_enumeration == True):  # if making a new workdir
            self.prep_new_working_directory()
        else:
            print("Must provide a run_num if not creating a new workdir!")
            sys.exit()

        '''load dataset'''
        data_manager = DataManager(config=self.config, dataset_path=self.config.dataset_path, collect_chunks=False)  # dataminer for dataset construction

        if self.config.skip_saving_and_loading:  # transfer dataset directly from miner rather than saving and reloading
            self.prep_dataset = data_manager.load_for_modelling(return_dataset=True, save_dataset=False)
            del data_manager
        else:
            data_manager.load_for_modelling(return_dataset=False, save_dataset=True)
            self.prep_dataset = None

        self.train_discriminator = any((config.train_discriminator_adversarially, config.train_discriminator_on_distorted, config.train_discriminator_on_randn))
        self.train_generator = any((config.train_generator_vdw, config.train_generator_adversarially, config.train_generator_h_bond))

    def prep_symmetry_info(self):
        '''
        if we don't have the symmetry dict prepared already, generate it
        '''
        # if True: #os.path.exists('symmetry_info.npy'):
        # sym_info: dict = np.load('symmetry_info.npy', allow_pickle=True).item()
        self.sym_ops = SYM_OPS  # sym_info['sym_ops']
        self.point_groups = POINT_GROUPS
        self.lattice_type = LATTICE_TYPE
        self.space_groups = SPACE_GROUPS

        self.sym_info = {
            'sym_ops': self.sym_ops,
            'point_groups': self.point_groups,
            'lattice_type': self.lattice_type,
            'space_groups': self.space_groups}
        # else: # generate spacegroup information dict - should no longer be necessary
        #     from pyxtal import symmetry
        #     print('Pre-generating spacegroup symmetries')
        #     self.sym_ops = {}
        #     self.point_groups = {}
        #     self.lattice_type = {}
        #     self.space_groups = {}
        #     self.space_group_indices = {}
        #     for i in tqdm.tqdm(range(1, 231)):
        #         sym_group = symmetry.Group(i)
        #         general_position_syms = sym_group.wyckoffs_organized[0][0]
        #         self.sym_ops[i] = [general_position_syms[i].affine_matrix for i in range(
        #             len(general_position_syms))]  # first 0 index is for general position, second index is
        #         # superfluous, third index is the symmetry operation
        #         self.point_groups[i] = sym_group.point_group
        #         self.lattice_type[i] = sym_group.lattice_type
        #         self.space_groups[i] = sym_group.symbol
        #         self.space_group_indices[sym_group.symbol] = i
        #
        #     self.sym_info = {
        #         'sym_ops': self.sym_ops,
        #         'point_groups': self.point_groups,
        #         'lattice_type': self.lattice_type,
        #         'space_groups': self.space_groups,
        #         'space_group_indices': self.space_group_indices}
        #
        #     np.save('symmetry_info', self.sym_info)

    def prep_new_working_directory(self):
        if self.config.run_num == 0:
            self.make_sequential_directory()
        else:
            self.workDir = self.config.workdir + '/run%d' % self.config.run_num  # explicitly enumerate the new run directory
            os.mkdir(self.workDir)

        os.mkdir(self.workDir + '/ckpts')  # not used
        os.mkdir(self.workDir + '/datasets')  # not used
        os.mkdir(self.workDir + '/source')
        yaml_path = os.getcwd() + '/' + self.config.yaml_config

        # copy source to workdir for record keeping purposes
        copy_tree("common", self.workDir + "/source/common")
        copy_tree("crystal_building", self.workDir + "/source/crystal_building")
        copy_tree("dataset_management", self.workDir + "/source/dataset_management")
        copy_tree("models", self.workDir + "/source/models")
        copy_tree("reporting", self.workDir + "/source/reporting")
        copy_tree("sampling", self.workDir + "/source/sampling")
        copy("crystal_modeller.py", self.workDir + "/source")
        copy("main.py", self.workDir + "/source")

        os.chdir(self.workDir)  # move to working dir
        copy(yaml_path, os.getcwd())  # copy full config for reference
        print('Starting Fresh Run %d' % self.config.run_num)
        t0 = time.time()
        print('Initializing dataset took {} seconds'.format(int(time.time() - t0)))

    def make_sequential_directory(self):  # make working directory
        """
        make a new working directory
        non-overlapping previous entries
        or with a preset number
        :return:
        """
        workdirs = glob.glob(self.config.workdir + '/' + 'run*')  # check for prior working directories
        if len(workdirs) > 0:
            prev_runs = []
            for i in range(len(workdirs)):
                prev_runs.append(int(workdirs[i].split('run')[-1]))

            prev_max = max(prev_runs)
            self.workDir = self.config.workdir + '/' + 'run%d' % (prev_max + 1)
            self.config.workdir = self.workDir
            os.mkdir(self.workDir)
            self.config.run_num = int(prev_max + 1)
        else:
            self.workDir = self.config.workdir + '/' + 'run1'
            self.config.run_num = 1
            os.mkdir(self.workDir)

    def init_models(self):
        """
        Initialize models and optimizers and schedulers
        :return:
        """
        self.config = self.reload_model_checkpoints(self.config)

        generator, discriminator, regressor = nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(
            1, 1)  # init dummy models
        print("Initializing model(s) for " + self.config.mode)
        if self.config.mode == 'gan' or self.config.mode == 'sampling' or self.config.mode == 'embedding':
            generator = crystal_generator(self.config, self.config.dataDims, self.sym_info)
            discriminator = crystal_discriminator(self.config, self.config.dataDims)
        if self.config.mode == 'regression' or self.config.regressor_path is not None:
            regressor = molecule_regressor(self.config, self.config.dataDims)

        if self.config.device.lower() == 'cuda':
            print('Putting models on CUDA')
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            regressor = regressor.cuda()

        generator_optimizer = init_optimizer(self.config.generator_optimizer, generator)
        discriminator_optimizer = init_optimizer(self.config.discriminator_optimizer, discriminator)
        regressor_optimizer = init_optimizer(self.config.regressor_optimizer, regressor)

        if self.config.generator_path is not None and (self.config.mode == 'gan' or self.config.mode == 'embedding'):
            generator, generator_optimizer = reload_model(generator, generator_optimizer,
                                                          self.config.generator_path)
        if self.config.discriminator_path is not None and (self.config.mode == 'gan' or self.config.mode == 'embedding'):
            discriminator, discriminator_optimizer = reload_model(discriminator, discriminator_optimizer,
                                                                  self.config.discriminator_path)
        if self.config.regressor_path is not None:
            regressor, regressor_optimizer = reload_model(regressor, regressor_optimizer,
                                                          self.config.regressor_path)

        generator_schedulers = init_schedulers(self.config.generator_optimizer, generator_optimizer)
        discriminator_schedulers = init_schedulers(self.config.discriminator_optimizer, discriminator_optimizer)
        regressor_schedulers = init_schedulers(self.config.regressor_optimizer, regressor_optimizer)

        num_params = [get_n_config(model) for model in [generator, discriminator, regressor]]
        print('Generator model has {:.3f} million or {} parameters'.format(num_params[0] / 1e6, int(num_params[0])))
        print('Discriminator model has {:.3f} million or {} parameters'.format(num_params[1] / 1e6, int(num_params[1])))
        print('Regressor model has {:.3f} million or {} parameters'.format(num_params[2] / 1e6, int(num_params[2])))

        wandb.watch((generator, discriminator, regressor), log_graph=True, log_freq=100)
        wandb.log({"Model Num Parameters": np.sum(np.asarray(num_params)),
                   "Initial Batch Size": self.config.current_batch_size})

        return generator, discriminator, regressor, \
            generator_optimizer, generator_schedulers, \
            discriminator_optimizer, discriminator_schedulers, \
            regressor_optimizer, regressor_schedulers, \
            num_params

    def prep_dataloaders(self, dataset_builder, test_fraction=0.2):
        train_loader, test_loader = get_dataloaders(dataset_builder, machine=self.config.machine, batch_size=self.config.min_batch_size, test_fraction=test_fraction)
        self.config.current_batch_size = self.config.min_batch_size
        print("Training batch size set to {}".format(self.config.current_batch_size))
        del dataset_builder

        extra_test_loader = None  # data_loader for a secondary test set - analysis is hardcoded for CSD Blind Tests 5 & 6
        if self.config.extra_test_set_paths is not None:
            extra_test_loader = get_extra_test_loader(self.config,
                                                      self.config.extra_test_set_paths,
                                                      dataDims=self.config.dataDims,
                                                      pg_dict=self.point_groups,
                                                      sg_dict=self.space_groups,
                                                      lattice_dict=self.lattice_type,
                                                      sym_ops_dict=self.sym_ops)

        return train_loader, test_loader, extra_test_loader

    def crystal_embedding_analysis(self):
        """
        analyze the embeddings of a given crystal dataset
        embeddings provided by pretrained model
        """
        """
                train and/or evaluate one or more models
                regressor
                GAN (generator and/or discriminator)
                """
        with wandb.init(config=self.config,
                        project=self.config.wandb.project_name,
                        entity=self.config.wandb.username,
                        tags=[self.config.wandb.experiment_tag],
                        settings=wandb.Settings(code_dir=".")):

            wandb.run.name = wandb.config.machine + '_' + str(self.config.mode) + '_' + str(wandb.config.run_num)  # overwrite procedurally generated run name with our run name

            '''miscellaneous setup'''
            dataset_builder = self.misc_pre_training_items()

            '''prep dataloaders'''
            from torch_geometric.loader import DataLoader
            test_dataset = []
            for i in range(len(dataset_builder)):
                test_dataset.append(dataset_builder[i])

            self.config.current_batch_size = self.config.min_batch_size
            print("Training batch size set to {}".format(self.config.current_batch_size))
            del dataset_builder
            test_loader = DataLoader(test_dataset, batch_size=self.config.current_batch_size, shuffle=True, num_workers=0, pin_memory=True)

            '''instantiate models'''
            generator, discriminator, regressor, \
                generator_optimizer, generator_schedulers, \
                discriminator_optimizer, discriminator_schedulers, \
                regressor_optimizer, regressor_schedulers, \
                num_params = self.init_models()

            '''initialize some training metrics'''
            with torch.autograd.set_detect_anomaly(self.config.anomaly_detection):
                # very cool
                print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                # very cool
                print("Starting Embedding Analysis")

                with torch.no_grad():
                    # compute test loss & save evaluation statistics on test samples
                    embedding_dict = self.embed_dataset(
                        data_loader=test_loader, generator=generator, discriminator=discriminator, regressor=regressor)

                    np.save('embedding_dict', embedding_dict)

    def embed_dataset(self, data_loader, discriminator, generator=None, regressor=None):
        t0 = time.time()
        discriminator.eval()

        embedding_dict = {
            'tracking features': [],
            'identifiers': [],
            'scores': [],
            'source': [],
            'latent': [],
        }
        epoch_stats_dict = {}  # unused but necessary for some functions

        for i, data in enumerate(tqdm.tqdm(data_loader)):
            '''
            get discriminator embeddings
            '''

            '''real data'''
            real_supercell_data = \
                self.supercell_builder.unit_cell_to_supercell(
                    data, self.config.supercell_size, self.config.discriminator.graph_convolution_cutoff)

            score_on_real, real_distances_dict, latent = \
                self.adversarial_score(
                    discriminator, real_supercell_data, return_latent=True)

            embedding_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())
            embedding_dict['identifiers'].extend(data.csd_identifier)
            embedding_dict['scores'].extend(score_on_real.cpu().detach().numpy())
            embedding_dict['latent'].extend(latent)
            embedding_dict['source'].extend(['real' for _ in range(len(latent))])

            '''fake data'''
            for j in tqdm.tqdm(range(100)):
                real_data = data.clone()
                generated_samples_i, epoch_stats_dict, negative_type, real_data = \
                    self.generate_discriminator_negatives(epoch_stats_dict, real_data, generator, i, regressor,
                                                          override_randn=True, override_distorted=True)

                fake_supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
                    real_data, generated_samples_i, self.config.supercell_size,
                    self.config.discriminator.graph_convolution_cutoff,
                    align_molecules=(negative_type != 'generated'),
                    target_handedness=real_data.asym_unit_handedness,
                )

                score_on_fake, fake_pairwise_distances_dict, fake_latent = self.adversarial_score(discriminator, fake_supercell_data, return_latent=True)

                embedding_dict['tracking features'].extend(real_data.tracking.cpu().detach().numpy())
                embedding_dict['identifiers'].extend(real_data.csd_identifier)
                embedding_dict['scores'].extend(score_on_fake.cpu().detach().numpy())
                embedding_dict['latent'].extend(fake_latent)
                embedding_dict['source'].extend([negative_type for _ in range(len(latent))])

        embedding_dict['scores'] = np.stack(embedding_dict['scores'])
        embedding_dict['tracking features'] = np.stack(embedding_dict['tracking features'])
        embedding_dict['latent'] = np.stack(embedding_dict['latent'])

        total_time = time.time() - t0
        print(f"Embedding took {total_time:.1f} Seconds")

        '''distance matrix'''
        scores = softmax_and_score(embedding_dict['scores'])
        latents = torch.Tensor(embedding_dict['latent'])
        overlaps = torch.inner(latents, latents) / torch.outer(torch.linalg.norm(latents, dim=-1), torch.linalg.norm(latents, dim=-1))
        distmat = torch.cdist(latents, latents)

        sample_types = list(set(embedding_dict['source']))
        inds_dict = {}
        for source in sample_types:
            inds_dict[source] = np.argwhere(np.asarray(embedding_dict['source']) == source)[:, 0]

        mean_overlap_to_real = {}
        mean_dist_to_real = {}
        mean_score = {}
        for source in sample_types:
            sample_dists = distmat[inds_dict[source]]
            sample_scores = scores[inds_dict[source]]
            sample_overlaps = overlaps[inds_dict[source]]

            mean_dist_to_real[source] = sample_dists[:, inds_dict['real']].mean()
            mean_overlap_to_real[source] = sample_overlaps[:, inds_dict['real']].mean()
            mean_score[source] = sample_scores.mean()

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.colors import n_colors

        # '''distances'''
        # fig = make_subplots(rows=1, cols=2, subplot_titles=('distances', 'dot overlaps'))
        # fig.add_trace(go.Heatmap(z=distmat), row=1, col=1)
        # fig.add_trace(go.Heatmap(z=overlaps), row=1, col=2)
        # fig.show()

        '''distance to real vs score'''
        colors = n_colors('rgb(250,0,5)', 'rgb(5,150,250)', len(inds_dict.keys()), colortype='rgb')

        fig = make_subplots(rows=1, cols=2)
        for ii, source in enumerate(sample_types):
            fig.add_trace(go.Scattergl(
                x=distmat[inds_dict[source]][:, inds_dict['real']].mean(-1), y=scores[inds_dict[source]],
                mode='markers', marker=dict(color=colors[ii]), name=source), row=1, col=1
            )

            fig.add_trace(go.Scattergl(
                x=overlaps[inds_dict[source]][:, inds_dict['real']].mean(-1), y=scores[inds_dict[source]],
                mode='markers', marker=dict(color=colors[ii]), showlegend=False), row=1, col=2
            )

        fig.update_xaxes(title_text='mean distance to real', row=1, col=1)
        fig.update_yaxes(title_text='discriminator score', row=1, col=1)

        fig.update_xaxes(title_text='mean overlap to real', row=1, col=2)
        fig.update_yaxes(title_text='discriminator score', row=1, col=2)
        fig.show()

        return embedding_dict

    def train_crystal_models(self):
        """
        train and/or evaluate one or more models
        regressor
        GAN (generator and/or discriminator)
        """
        with wandb.init(config=self.config,
                        project=self.config.wandb.project_name,
                        entity=self.config.wandb.username,
                        tags=[self.config.wandb.experiment_tag],
                        settings=wandb.Settings(code_dir=".")):

            wandb.run.name = wandb.config.machine + '_' + str(self.config.mode) + '_' + str(wandb.config.run_num)  # overwrite procedurally generated run name with our run name
            # config = wandb.config # wandb configs don't support nested namespaces. look at the github thread to see if they eventually fix it
            # this means we also can't do wandb sweeps properly, for now

            '''miscellaneous setup'''
            dataset_builder = self.misc_pre_training_items()

            '''prep dataloaders'''
            train_loader, test_loader, extra_test_loader = self.prep_dataloaders(dataset_builder)

            '''instantiate models'''
            generator, discriminator, regressor, \
                generator_optimizer, generator_schedulers, \
                discriminator_optimizer, discriminator_schedulers, \
                regressor_optimizer, regressor_schedulers, \
                num_params = self.init_models()

            '''initialize some training metrics'''
            metrics_dict = {}
            generator_err_tr, discriminator_err_tr, regressor_err_tr = 0, 0, 0
            generator_err_te, discriminator_err_te, regressor_err_te = 0, 0, 0
            generator_tr_record, discriminator_tr_record, regressor_tr_record = [0], [0], [0]
            generator_te_record, discriminator_te_record, regressor_te_record = [0], [0], [0]
            discriminator_hit_max_lr, generator_hit_max_lr, regressor_hit_max_lr, converged, epoch = \
                False, False, False, self.config.max_epochs == 0, 0

            # training loop
            with torch.autograd.set_detect_anomaly(self.config.anomaly_detection):
                while (epoch < self.config.max_epochs) and not converged:
                    # very cool
                    print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                    print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                    print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                    # very cool
                    print("Starting Epoch {}".format(epoch))  # index from 0
                    train_epoch_stats_dict, test_epoch_stats_dict, extra_test_epoch_stats_dict = None, None, None
                    try:  # try this batch size - if it fits, increase on next epoch
                        # train & compute loss
                        # save space by skipping evaluation of train epoch stats
                        train_loss, train_loss_record, time_train = \
                            self.run_epoch(data_loader=train_loader,
                                           generator=generator, discriminator=discriminator, regressor=regressor,
                                           generator_optimizer=generator_optimizer,
                                           discriminator_optimizer=discriminator_optimizer,
                                           regressor_optimizer=regressor_optimizer,
                                           update_gradients=True, record_stats=False, epoch=epoch)

                        with torch.no_grad():
                            # compute test loss & save evaluation statistics on test samples
                            test_loss, test_loss_record, test_epoch_stats_dict, time_test = \
                                self.run_epoch(data_loader=test_loader,
                                               generator=generator, discriminator=discriminator, regressor=regressor,
                                               update_gradients=False, record_stats=True, epoch=epoch)

                            if (extra_test_loader is not None) and (epoch % self.config.extra_test_period == 0):
                                _, _, extra_test_epoch_stats_dict, extra_time_test = \
                                    self.run_epoch(data_loader=extra_test_loader,
                                                   generator=generator, discriminator=discriminator,
                                                   update_gradients=False, record_stats=True, epoch=epoch)  # compute loss on test set
                                np.save(f'../{self.config.run_num}_extra_test_dict', extra_test_epoch_stats_dict)

                        print('epoch={}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, time_train, time_test))

                        '''save losses'''
                        if self.config.mode == 'gan':
                            discriminator_err_tr, generator_err_tr = train_loss[0], train_loss[1]
                            discriminator_err_te, generator_err_te = test_loss[0], test_loss[1]
                            discriminator_tr_record, generator_tr_record = train_loss_record[0], train_loss_record[1]
                            discriminator_te_record, generator_te_record = test_loss_record[0], test_loss_record[1]
                        elif self.config.mode == 'regression':
                            regressor_err_tr, regressor_err_te = train_loss, test_loss
                            regressor_tr_record, regressor_te_record = train_loss_record, test_loss_record

                        '''update learning rates'''
                        discriminator_optimizer, discriminator_learning_rate, discriminator_hit_max_lr, \
                            generator_optimizer, generator_learning_rate, generator_hit_max_lr, \
                            regressor_optimizer, regressor_learning_rate, regressor_hit_max_lr = \
                            self.update_lr(discriminator_schedulers, discriminator_optimizer, discriminator_err_tr,
                                           discriminator_hit_max_lr,
                                           generator_schedulers, generator_optimizer, generator_err_tr,
                                           generator_hit_max_lr,
                                           regressor_schedulers, regressor_optimizer, regressor_err_tr,
                                           regressor_hit_max_lr)

                        '''save key metrics'''
                        metrics_dict = update_gan_metrics(epoch, metrics_dict,
                                                          discriminator_learning_rate, generator_learning_rate,
                                                          regressor_learning_rate,
                                                          discriminator_err_tr, discriminator_err_te,
                                                          generator_err_tr, generator_err_te,
                                                          regressor_err_tr, regressor_err_te)

                        '''log losses to wandb'''
                        self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                                          generator_tr_record, generator_te_record, discriminator_tr_record,
                                          discriminator_te_record,
                                          regressor_tr_record, regressor_te_record)

                        '''sometimes to detailed reporting'''
                        if (epoch % self.config.wandb.sample_reporting_frequency) == 0:
                            detailed_reporting(self.config, epoch, test_loader, train_epoch_stats_dict, test_epoch_stats_dict,
                                               extra_test_dict=extra_test_epoch_stats_dict)

                        '''sometimes test the generator on a mini CSP problem'''
                        if (self.config.mode == 'gan') and (epoch % self.config.wandb.mini_csp_frequency == 0) and \
                                self.train_generator and (epoch > 0):
                            self.mini_csp(extra_test_loader if extra_test_loader is not None else test_loader, generator, discriminator, regressor if self.config.regressor_path else None)

                        '''save checkpoints'''
                        self.model_checkpointing(epoch, self.config, discriminator, generator, regressor,
                                                 discriminator_optimizer, generator_optimizer,
                                                 regressor_optimizer,
                                                 generator_err_te, discriminator_err_te,
                                                 regressor_err_te, metrics_dict)

                        '''check convergence status'''
                        generator_converged, discriminator_converged, regressor_converged = \
                            self.check_model_convergence(metrics_dict)

                        if (generator_converged and discriminator_converged and regressor_converged) and (
                                epoch > self.config.history + 2):
                            print('Training has converged!')
                            break

                        '''increment batch size'''
                        train_loader, test_loader, extra_test_loader = \
                            self.increment_batch_size(train_loader, test_loader, extra_test_loader)

                    except RuntimeError as e:  # if we do hit OOM, slash the batch size
                        if "CUDA out of memory" in str(e):
                            train_loader, test_loader = self.slash_batch(train_loader, test_loader, 0.05)  # shrink batch size
                            self.config.grow_batch_size = False  # stop growing the batch for the rest of the run
                        else:
                            raise e
                    epoch += 1

                    if self.config.device.lower() == 'cuda':
                        torch.cuda.empty_cache()  # clear GPU, not clear this does anything

                if self.config.mode == 'gan':  # evaluation on test metrics
                    self.gan_evaluation(epoch, generator, discriminator,
                                        test_loader, extra_test_loader, regressor)

    def run_epoch(self, data_loader=None, generator=None, discriminator=None, regressor=None,
                  generator_optimizer=None, discriminator_optimizer=None, regressor_optimizer=None,
                  update_gradients=True, iteration_override=None, record_stats=False, epoch=None):

        if self.config.mode == 'gan':
            if self.config.regressor_path is not None:
                regressor.eval()  # just using this to suggest densities to the generator

            return self.gan_epoch(data_loader, generator, discriminator, generator_optimizer, discriminator_optimizer,
                                  update_gradients,
                                  iteration_override, record_stats, epoch, regressor if self.config.regressor_path else None)

        elif self.config.mode == 'regression':
            return self.regression_epoch(data_loader, regressor, regressor_optimizer, update_gradients,
                                         iteration_override, record_stats)

    def regression_epoch(self, data_loader, regressor, regressor_optimizer=None, update_gradients=True,
                         iteration_override=None, record_stats=False):

        t0 = time.time()
        if update_gradients:
            regressor.train(True)
        else:
            regressor.eval()

        loss = []
        loss_record = []
        epoch_stats_dict = {}

        for i, data in enumerate(tqdm.tqdm(data_loader, miniters=int(len(data_loader) / 25))):
            if self.config.regressor_positional_noise > 0:
                data.pos += torch.randn_like(data.pos) * self.config.regressor_positional_noise

            regression_losses_list, predictions, targets = self.regression_loss(regressor, data)

            stats_keys = ['regressor packing prediction', 'regressor packing target']
            stats_values = [predictions.cpu().detach().numpy(), targets.cpu().detach().numpy()]
            epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values, mode='extend')

            regression_loss = regression_losses_list.mean()
            loss.append(regression_loss.data.cpu().detach().numpy())  # average loss
            loss_record.extend(regression_losses_list.cpu().detach().numpy())  # loss distribution

            if update_gradients:
                regressor_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                regression_loss.backward()  # back-propagation
                regressor_optimizer.step()  # update parameters

            if record_stats:
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'tracking features',
                                                     data.tracking.cpu().detach().numpy(), mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0

        if record_stats:
            epoch_stats_dict['tracking features'] = np.stack(epoch_stats_dict['tracking features'])

            for key in epoch_stats_dict.keys():
                feature = epoch_stats_dict[key]
                if (feature == []) or (feature is None):
                    epoch_stats_dict[key] = None
                else:
                    try:
                        epoch_stats_dict[key] = np.asarray(feature)
                    except:
                        pass

            epoch_stats_dict['data dims'] = self.config.dataDims.copy()  # record explicitly all the tracking features

            return np.mean(loss), loss_record, epoch_stats_dict, total_time
        else:
            return np.mean(loss), loss_record, total_time

    def gan_epoch(self, data_loader=None, generator=None, discriminator=None, generator_optimizer=None,
                  discriminator_optimizer=None, update_gradients=True,
                  iteration_override=None, record_stats=False, epoch=None, regressor=None):
        t0 = time.time()

        if update_gradients:
            generator.train(True)
            discriminator.train(True)
        else:
            generator.eval()
            discriminator.eval()

        discriminator_err = []
        discriminator_loss_record = []
        generator_err = []
        generator_loss_record = []

        epoch_stats_dict = {}

        rand_batch_ind = np.random.randint(0, len(data_loader))

        for i, data in enumerate(tqdm.tqdm(data_loader, miniters=int(len(data_loader) / 10), mininterval=30)):
            '''
            train discriminator
            '''
            data = data.to(self.config.device)

            # hold discriminator training when it's beating the generator
            skip_discriminator_step = False
            if (i == 0) and self.config.train_generator_adversarially:
                skip_discriminator_step = True  # do not train except by express permission of the below condition
            if i > 0 and self.config.train_discriminator_adversarially:  # must skip first step since there will be no fake score to compare against
                avg_generator_score = np_softmax(np.stack(epoch_stats_dict['discriminator fake score'])[np.argwhere(np.asarray(epoch_stats_dict['generator sample source']) == 0)[:, 0]])[:, 1].mean()
                if avg_generator_score < 0.5:
                    skip_discriminator_step = True

            discriminator_err, discriminator_loss_record, epoch_stats_dict = \
                self.discriminator_step(discriminator, generator, epoch_stats_dict, data,
                                        discriminator_optimizer, i, update_gradients, discriminator_err,
                                        discriminator_loss_record,
                                        skip_step=skip_discriminator_step, regressor=regressor)
            '''
            train_generator
            '''
            generator_err, generator_loss_record, epoch_stats_dict = \
                self.generator_step(discriminator, generator, epoch_stats_dict, data,
                                    generator_optimizer, i, update_gradients, generator_err, generator_loss_record,
                                    rand_batch_ind, last_batch=i == (len(data_loader) - 1), regressor=regressor)
            '''
            record some stats
            '''
            if (len(epoch_stats_dict[
                        'generated cell parameters']) < i) and record_stats:  # make some samples for analysis if we have none so far from this step
                generated_samples = generator(data.num_graphs, z=None, conditions=data.to(self.config.device))
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generated cell parameters',
                                                     generated_samples.cpu().detach().numpy(), mode='extend')

            if record_stats:
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'tracking features',
                                                     data.tracking.cpu().detach().numpy(), mode='extend')
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'identifiers', data.csd_identifier,
                                                     mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0

        if record_stats:
            for key in epoch_stats_dict.keys():  # convert any lists to np arrays and empty lists to None
                if 'supercell' not in key:
                    feature = epoch_stats_dict[key]
                    if (feature == []) or (feature is None):
                        epoch_stats_dict[key] = None
                    else:
                        if isinstance(feature, list):
                            if isinstance(feature[0], list):
                                epoch_stats_dict[key] = np.concatenate(feature)
                            else:
                                epoch_stats_dict[key] = np.asarray(feature)

            epoch_stats_dict['data dims'] = self.config.dataDims.copy()  # record explicitly all the tracking features

            return [np.mean(discriminator_err), np.mean(generator_err)], [discriminator_loss_record, generator_loss_record], epoch_stats_dict, total_time
        else:
            return [np.mean(discriminator_err), np.mean(generator_err)], [discriminator_loss_record, generator_loss_record], total_time

    def discriminator_evaluation(self, data_loader=None, discriminator=None, iteration_override=None):  # todo write generator evaluation
        t0 = time.time()
        discriminator.eval()

        epoch_stats_dict = {
            'tracking features': [],
            'identifiers': [],
            'scores': [],
            'intermolecular rdf': [],
            'atomistic energy': [],
            'full rdf': [],
            'vdw penalty': [],
        }

        for i, data in enumerate(tqdm.tqdm(data_loader)):
            '''
            evaluate discriminator
            '''
            real_supercell_data = \
                self.supercell_builder.unit_cell_to_supercell(data, self.config.supercell_size, self.config.discriminator.graph_convolution_cutoff)

            if self.config.device.lower() == 'cuda':  # redundant
                real_supercell_data = real_supercell_data.cuda()

            if self.config.test_mode or self.config.anomaly_detection:
                assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"

            score_on_real, real_distances_dict = self.adversarial_score(discriminator, real_supercell_data)

            epoch_stats_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())
            epoch_stats_dict['identifiers'].extend(data.csd_identifier)  #
            epoch_stats_dict['scores'].extend(score_on_real.cpu().detach().numpy())

            epoch_stats_dict['vdw penalty'].extend(
                vdw_overlap(real_supercell_data, self.vdw_radii).cpu().detach().numpy())

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        epoch_stats_dict['scores'] = np.stack(epoch_stats_dict['scores'])
        epoch_stats_dict['tracking features'] = np.stack(epoch_stats_dict['tracking features'])
        # epoch_stats_dict['full rdf'] = np.stack(epoch_stats_dict['full rdf'])
        # epoch_stats_dict['intermolecular rdf'] = np.stack(epoch_stats_dict['intermolecular rdf'])
        epoch_stats_dict['vdw penalty'] = np.asarray(epoch_stats_dict['vdw penalty'])

        total_time = time.time() - t0

        return epoch_stats_dict, total_time

    @staticmethod
    def adversarial_score(discriminator, data, return_latent=False):
        output, extra_outputs = discriminator(data.clone(), return_dists=True, return_latent=return_latent)  # reshape output from flat filters to channels * filters per channel
        if return_latent:
            return output, extra_outputs['dists dict'], extra_outputs['latent']
        else:
            return output, extra_outputs['dists dict']

    @staticmethod
    def log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                     discriminator_tr_record, discriminator_te_record, generator_tr_record, generator_te_record,
                     regressor_tr_record, regressor_te_record):

        current_metrics = {}
        for key in metrics_dict.keys():
            current_metrics[key] = float(metrics_dict[key][-1])
            if 'loss' in key:  # log 'best' metrics
                current_metrics['best ' + key] = np.amin(metrics_dict[key])

            elif ('epoch' in key) or ('confusion' in key) or ('learning rate'):
                pass
            else:
                current_metrics['best ' + key] = np.amax(metrics_dict[key])

        for key in current_metrics.keys():
            current_metrics[key] = np.amax(
                current_metrics[key])  # just a formatting thing - nothing to do with the max of anything
        wandb.log(current_metrics)

        # log discriminator losses
        if discriminator_tr_record is not None:
            hist = np.histogram(discriminator_tr_record, bins=256,
                                range=(np.amin(discriminator_tr_record), np.quantile(discriminator_tr_record, 0.9)))
            wandb.log({"Discriminator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        if discriminator_te_record is not None:
            hist = np.histogram(discriminator_te_record, bins=256,
                                range=(np.amin(discriminator_te_record), np.quantile(discriminator_te_record, 0.9)))
            wandb.log({"Discriminator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        # log generator losses
        if generator_tr_record is not None:
            hist = np.histogram(generator_tr_record, bins=256,
                                range=(np.amin(generator_tr_record), np.quantile(generator_tr_record, 0.9)))
            wandb.log({"Generator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        if generator_te_record is not None:
            hist = np.histogram(generator_te_record, bins=256,
                                range=(np.amin(generator_te_record), np.quantile(generator_te_record, 0.9)))
            wandb.log({"Generator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        # log regressor losses
        if regressor_tr_record is not None:
            hist = np.histogram(regressor_tr_record, bins=256,
                                range=(np.amin(regressor_tr_record), np.quantile(regressor_tr_record, 0.9)))
            wandb.log({"Generator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        if regressor_te_record is not None:
            hist = np.histogram(regressor_te_record, bins=256,
                                range=(np.amin(regressor_te_record), np.quantile(regressor_te_record, 0.9)))
            wandb.log({"Generator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        # log special losses
        special_losses = {'epoch': current_metrics['epoch']}
        if train_epoch_stats_dict is not None:
            for key in train_epoch_stats_dict.keys():
                if isinstance(train_epoch_stats_dict[key], list):
                    train_epoch_stats_dict[key] = train_epoch_stats_dict[key]
                if ('loss' in key) and (train_epoch_stats_dict[key] is not None):
                    special_losses['Train ' + key] = np.average(train_epoch_stats_dict[key])
                if ('score' in key) and (train_epoch_stats_dict[key] is not None):
                    score = softmax_and_score(train_epoch_stats_dict[key])
                    special_losses['Train ' + key] = np.average(score)

        if test_epoch_stats_dict is not None:
            for key in test_epoch_stats_dict.keys():
                if ('loss' in key) and (test_epoch_stats_dict[key] is not None):
                    special_losses['Test ' + key] = np.average(test_epoch_stats_dict[key])
                if ('score' in key) and (test_epoch_stats_dict[key] is not None):
                    score = softmax_and_score(test_epoch_stats_dict[key])
                    special_losses['Test ' + key] = np.average(score)

        wandb.log(special_losses)

    def discriminator_step(self, discriminator, generator, epoch_stats_dict, data, discriminator_optimizer, i,
                           update_gradients, discriminator_err, discriminator_loss_record, skip_step, regressor):

        if self.train_discriminator:
            score_on_real, score_on_fake, generated_samples, \
                real_dist_dict, fake_dist_dict, real_vdw_score, fake_vdw_score, \
                real_packing_coeffs, fake_packing_coeffs, generated_samples_i \
                = self.get_discriminator_losses(discriminator, generator, data, i, epoch_stats_dict, regressor)

            discriminator_scores = torch.cat((score_on_real, score_on_fake))
            discriminator_target = torch.cat((torch.ones_like(score_on_real[:, 0]), torch.zeros_like(score_on_fake[:, 0])))
            discriminator_losses = F.cross_entropy(discriminator_scores, discriminator_target.long(), reduction='none')  # works much better

            discriminator_loss = discriminator_losses.mean()
            discriminator_err.append(discriminator_loss.data.cpu().detach().numpy())

            if update_gradients and (not skip_step):
                discriminator_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(),
                                               self.config.gradient_norm_clip)  # gradient clipping
                discriminator_loss.backward()  # back-propagation
                discriminator_optimizer.step()  # update parameters

            stats_keys = ['discriminator real score', 'discriminator fake score',
                          'real vdw penalty', 'fake vdw penalty',
                          'generated cell parameters', 'final generated cell parameters',
                          'real packing coefficients', 'generated packing coefficients']
            stats_values = [score_on_real.cpu().detach().numpy(), score_on_fake.cpu().detach().numpy(),
                            real_vdw_score.cpu().detach().numpy(), fake_vdw_score.cpu().detach().numpy(),
                            generated_samples_i.cpu().detach().numpy(), generated_samples,
                            real_packing_coeffs.cpu().detach().numpy(), fake_packing_coeffs.cpu().detach().numpy()]
            epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values, mode='extend')

            discriminator_loss_record.extend(discriminator_losses.cpu().detach().numpy())  # overall loss distribution
        else:
            discriminator_err.append(np.zeros(1))
            discriminator_loss_record.extend(np.zeros(data.num_graphs))

        return discriminator_err, discriminator_loss_record, epoch_stats_dict

    def generator_step(self, discriminator, generator, epoch_stats_dict, data, generator_optimizer, i, update_gradients,
                       generator_err, generator_loss_record, rand_batch_ind, last_batch, regressor):
        if self.train_generator:
            discriminator_raw_output, generated_samples, packing_loss, packing_prediction, packing_target, \
                vdw_loss, generated_dist_dict, supercell_examples, similarity_penalty, h_bond_score = \
                self.get_generator_losses(generator, discriminator, data, i, regressor)

            generator_losses, epoch_stats_dict = self.aggregate_generator_losses(
                epoch_stats_dict, packing_loss, discriminator_raw_output,
                vdw_loss, similarity_penalty, packing_prediction, packing_target, h_bond_score)

            generator_loss = generator_losses.mean()
            generator_err.append(generator_loss.data.cpu().detach().numpy())  # average loss

            if update_gradients:
                generator_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                torch.nn.utils.clip_grad_norm_(generator.parameters(),
                                               self.config.gradient_norm_clip)  # gradient clipping
                generator_loss.backward()  # back-propagation
                generator_optimizer.step()  # update parameters

            epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'final generated cell parameters',
                                                 supercell_examples.cell_params.cpu().detach().numpy(), mode='extend')
            generator_loss_record.extend(generator_losses.cpu().detach().numpy())  # loss distribution
            epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generated cell parameters', generated_samples,
                                                 mode='extend')
            del supercell_examples


        else:
            generator_err.append(np.zeros(1))
            generator_loss_record.extend(np.zeros(data.num_graphs))

        return generator_err, generator_loss_record, epoch_stats_dict

    def get_discriminator_losses(self, discriminator, generator, real_data, i, epoch_stats_dict, regressor):
        # generate fakes & create supercell data
        real_supercell_data = self.supercell_builder.unit_cell_to_supercell(real_data, self.config.supercell_size, self.config.discriminator.graph_convolution_cutoff)

        generated_samples_i, epoch_stats_dict, negative_type, real_data = \
            self.generate_discriminator_negatives(epoch_stats_dict, real_data, generator, i, regressor)

        fake_supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
            real_data, generated_samples_i, self.config.supercell_size,
            self.config.discriminator.graph_convolution_cutoff,
            align_molecules=(negative_type != 'generated'),
            target_handedness=real_data.asym_unit_handedness,
        )

        if self.config.discriminator_positional_noise > 0:
            real_supercell_data.pos += \
                torch.randn_like(real_supercell_data.pos) * self.config.discriminator_positional_noise
            fake_supercell_data.pos += \
                torch.randn_like(fake_supercell_data.pos) * self.config.discriminator_positional_noise

        score_on_real, real_distances_dict, real_latent = self.adversarial_score(discriminator, real_supercell_data, return_latent=True)
        score_on_fake, fake_pairwise_distances_dict, fake_latent = self.adversarial_score(discriminator, fake_supercell_data, return_latent=True)

        # todo assign z values properly in cell construction
        real_packing_coeffs = compute_packing_coefficient(cell_params=real_supercell_data.cell_params,
                                                          mol_volumes=real_supercell_data.tracking[:, self.tracking_mol_volume_ind],
                                                          z_values=torch.tensor([len(real_supercell_data.ref_cell_pos[ii]) for ii in range(real_supercell_data.num_graphs)],
                                                                                dtype=torch.float64, device=real_supercell_data.x.device))
        fake_packing_coeffs = compute_packing_coefficient(cell_params=fake_supercell_data.cell_params,
                                                          mol_volumes=fake_supercell_data.tracking[:, self.tracking_mol_volume_ind],
                                                          z_values=torch.tensor([len(fake_supercell_data.ref_cell_pos[ii]) for ii in range(fake_supercell_data.num_graphs)],
                                                                                dtype=torch.float64, device=fake_supercell_data.x.device))

        return score_on_real, score_on_fake, fake_supercell_data.cell_params.cpu().detach().numpy(), \
            real_distances_dict, fake_pairwise_distances_dict, \
            vdw_overlap(self.vdw_radii, crystaldata=real_supercell_data), \
            vdw_overlap(self.vdw_radii, crystaldata=fake_supercell_data), \
            real_packing_coeffs, fake_packing_coeffs, \
            generated_samples_i

    def set_molecule_alignment(self, data, right_handed=False, mode_override=None):
        if mode_override is not None:
            mode = mode_override
        else:
            mode = self.config.canonical_conformer_orientation

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

        return data

    def get_generator_samples(self, data, generator, regressor, alignment_override=None):
        """
        @param mol_data: CrystalData object containing information on the starting conformer
        @param generator:
        @param target_packing: standardized target packing coefficient
        @return:
        """
        mol_data = data.clone()
        # conformer orientation setting
        mol_data = self.set_molecule_alignment(mol_data, mode_override=alignment_override)

        # noise injection
        if self.config.generator_positional_noise > 0:
            mol_data.pos += torch.randn_like(mol_data.pos) * self.config.generator_positional_noise

        # update symmetry information
        if self.config.generate_sgs is not None:
            mol_data = update_crystal_symmetry_elements(mol_data, self.config.generate_sgs, self.config.dataDims,
                                                        self.sym_info, randomize_sgs=True)

        # update packing coefficient
        if regressor is not None:
            # predict the crystal density and feed it as an input to the generator
            with torch.no_grad():
                standardized_target_packing_coeff = regressor(mol_data.clone().detach().to(self.config.device)).detach()[:, 0]
        else:
            target_packing_coeff = mol_data.tracking[:, self.config.dataDims['tracking features dict'].index('crystal packing coefficient')]
            standardized_target_packing_coeff = ((target_packing_coeff - self.config.dataDims['target mean']) / self.config.dataDims['target std']).to(self.config.device)

        standardized_target_packing_coeff += torch.randn_like(standardized_target_packing_coeff) * self.config.packing_target_noise

        for ii in range(len(standardized_target_packing_coeff)):
            mol_inds = torch.arange(mol_data.ptr[ii], mol_data.ptr[ii + 1])
            mol_data.x[mol_inds, int(self.sym_info['packing_coefficient_ind'])] = standardized_target_packing_coeff[ii]  # assign target packing coefficient

        # generate the samples
        [generated_samples, latent, prior, condition] = generator.forward(
            n_samples=mol_data.num_graphs, conditions=mol_data.to(self.config.device).clone(),
            return_latent=True, return_condition=True, return_prior=True)

        return generated_samples, prior, standardized_target_packing_coeff, mol_data

    def get_generator_losses(self, generator, discriminator, data, i, regressor):
        """
        train the generator
        """

        if self.train_generator:
            '''
            build supercells
            '''

            generated_samples, prior, standardized_target_packing, sample_data = (
                self.get_generator_samples(data, generator, regressor))

            supercell_data, generated_cell_volumes, _ = (
                self.supercell_builder.build_supercells(
                    data, generated_samples, self.config.supercell_size,
                    self.config.discriminator.graph_convolution_cutoff,
                    align_molecules=False
                ))

            similarity_penalty = self.compute_similarity_penalty(generated_samples, prior)
            discriminator_raw_output, dist_dict = self.score_adversarially(supercell_data, discriminator)
            h_bond_score = compute_h_bond_score(self.config.feature_richness, self.tracking_atom_acceptor_ind, self.tracking_atom_donor_ind, self.tracking_num_acceptors_ind, self.tracking_num_donors_ind, supercell_data)
            vdw_penalty, normed_vdw_penalty = get_vdw_penalty(self.vdw_radii,
                                                              dist_dict=dist_dict,
                                                              num_graphs=data.num_graphs,
                                                              mol_sizes=data.mol_size,
                                                              loss_func=self.config.vdw_loss_func)

            packing_loss, packing_prediction, packing_target, packing_csd = \
                generator_density_matching_loss(
                    standardized_target_packing, self.config.dataDims['target mean'], self.config.dataDims['target std'],
                    self.tracking_mol_volume_ind,
                    self.config.dataDims['tracking features dict'].index('crystal packing coefficient'),
                    supercell_data, generated_samples,
                    precomputed_volumes=generated_cell_volumes, loss_func=self.config.density_loss_func)

            return discriminator_raw_output, generated_samples.cpu().detach().numpy(), \
                packing_loss, packing_prediction.cpu().detach().numpy(), \
                packing_target.cpu().detach().numpy(), \
                vdw_penalty, dist_dict, \
                supercell_data, similarity_penalty, h_bond_score

    def regression_loss(self, regressor, data):
        predictions = regressor(data.to(regressor.model.device))[:, 0]
        targets = data.y
        return F.smooth_l1_loss(predictions, targets, reduction='none'), predictions, targets

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
        std_dataDims_path = self.source_directory + r'/dataset_management/standard_dataDims.npy'
        if os.path.exists(std_dataDims_path):
            standard_dataDims = np.load(std_dataDims_path,allow_pickle=True).item()  # maintain constant standardizations between runs
        else:
            print("Premade Standardization Missing!")
            standard_dataDims = None

        '''note this standard datadims construction will only work between runs with
        identical choice of features - there is a flag for this in the datasetbuilder'''
        dataset_builder = DatasetBuilder(self.config, pg_dict=self.point_groups,
                                         sg_dict=self.space_groups,
                                         lattice_dict=self.lattice_type,
                                         premade_dataset=self.prep_dataset,
                                         replace_dataDims=standard_dataDims)

        del self.prep_dataset  # we don't actually want this huge thing floating around
        self.config.dataDims = dataset_builder.get_dimension()
        if False:  #standard_dataDims is None:  # set a standard if we don't have one
            np.save(self.source_directory + r'/dataset_management/standard_dataDims', self.config.dataDims)

        '''init lattice mean & std'''
        self.lattice_means = torch.tensor(self.config.dataDims['lattice means'], dtype=torch.float32, device=self.config.device)
        self.lattice_stds = torch.tensor(self.config.dataDims['lattice stds'], dtype=torch.float32, device=self.config.device)

        '''
        init supercell builder
        '''
        self.supercell_builder = SupercellBuilder(self.sym_info, self.config.dataDims, device=self.config.device, rotation_basis=self.config.rotation_basis)

        '''
        set tracking feature indices & property dicts we will use later
        '''
        self.tracking_mol_volume_ind = self.config.dataDims['tracking features dict'].index('molecule volume')

        # index in the input where we can manipulate the desired crystal values: packing coefficient, sg indices, crystal systems
        self.crystal_packing_ind = self.config.dataDims['num atomwise features'] + self.config.dataDims['molecule features'].index('crystal packing coefficient')
        self.sg_feature_ind_dict = {thing[14:]: ind + self.config.dataDims['num atomwise features'] for ind, thing in
                                    enumerate(self.config.dataDims['molecule features']) if 'sg is' in thing}
        self.crysys_ind_dict = {thing[18:]: ind + self.config.dataDims['num atomwise features'] for ind, thing in
                                enumerate(self.config.dataDims['molecule features']) if 'crystal system is' in thing}

        if self.config.feature_richness == 'full':
            self.tracking_num_acceptors_ind = self.config.dataDims['tracking features dict'].index('molecule num acceptors')
            self.tracking_num_donors_ind = self.config.dataDims['tracking features dict'].index('molecule num donors')
            self.tracking_atom_acceptor_ind = self.config.dataDims['atom features'].index('atom is H bond acceptor')
            self.tracking_atom_donor_ind = self.config.dataDims['atom features'].index('atom is H bond donor')

        '''
        add symmetry element indices to symmetry dict
        '''
        # todo build separately
        self.sym_info['packing_coefficient_ind'] = self.crystal_packing_ind
        self.sym_info['sg_feature_ind_dict'] = self.sg_feature_ind_dict  # SG indices in input features
        self.sym_info['crysys_ind_dict'] = self.crysys_ind_dict  # crysys indices in input features
        self.sym_info['crystal_z_value_ind'] = self.config.dataDims['num atomwise features'] + self.config.dataDims['molecule features'].index('crystal z value')  # Z value index in input features

        ''' 
        init gaussian generator for cell parameter sampling
        we don't always use it but it's very cheap so just do it every time
        '''
        self.gaussian_generator = independent_gaussian_model(input_dim=self.config.dataDims['num lattice features'],
                                                             means=self.config.dataDims['lattice means'],
                                                             stds=self.config.dataDims['lattice stds'],
                                                             normed_length_means=self.config.dataDims[
                                                                 'lattice normed length means'],
                                                             normed_length_stds=self.config.dataDims[
                                                                 'lattice normed length stds'],
                                                             sym_info=self.sym_info,
                                                             device=self.config.device,
                                                             cov_mat=self.config.dataDims['lattice cov mat'])

        return dataset_builder

    def generate_discriminator_negatives(self, epoch_stats_dict, real_data, generator, i, regressor, override_adversarial=False, override_randn=False, override_distorted=False):
        """
        use one of the available cell generation tools to sample cell parameters, to be fed to the discriminator
        @param epoch_stats_dict:
        @param real_data:
        @param generator:
        @param i:
        @return:
        """

        self.n_generators = sum((self.config.train_discriminator_on_randn or override_randn,
                                 self.config.train_discriminator_on_distorted or override_distorted,
                                 self.config.train_discriminator_adversarially or override_adversarial))

        gen_randint = np.random.randint(0, self.n_generators, 1)

        self.generator_ind_list = []
        if self.config.train_discriminator_adversarially or override_adversarial:
            self.generator_ind_list.append(1)
        if self.config.train_discriminator_on_randn or override_randn:
            self.generator_ind_list.append(2)
        if self.config.train_discriminator_on_distorted or override_distorted:
            self.generator_ind_list.append(3)

        generator_ind = self.generator_ind_list[int(gen_randint)]  # randomly select which generator to use from the available set

        if self.config.train_discriminator_adversarially or override_adversarial:
            if generator_ind == 1:  # randomly sample which generator to use at each iteration
                negative_type = 'generator'
                with torch.no_grad():
                    generated_samples_i, _, _, real_data = self.get_generator_samples(real_data, generator, regressor)
                    epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',
                                                         np.zeros(len(generated_samples_i)), mode='extend')

        if self.config.train_discriminator_on_randn or override_randn:
            if generator_ind == 2:
                negative_type = 'randn'
                generated_samples_i = self.gaussian_generator.forward(real_data.num_graphs, real_data).to(self.config.device)
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',
                                                     np.ones(len(generated_samples_i)), mode='extend')

        if self.config.train_discriminator_on_distorted or override_distorted:
            if generator_ind == 3:
                negative_type = 'distorted'

                generated_samples_ii = (real_data.cell_params - self.lattice_means) / self.lattice_stds

                if self.config.sample_distortion_magnitude == -1:
                    distortion = torch.randn_like(generated_samples_ii) * torch.logspace(-.5, 0.5, len(generated_samples_ii)).to(generated_samples_ii.device)[:, None]  # wider range
                else:
                    distortion = torch.randn_like(generated_samples_ii) * self.config.sample_distortion_magnitude

                generated_samples_i_d = (generated_samples_ii + distortion).to(self.config.device)  # add jitter and return in standardized basis
                generated_samples_i = clean_cell_params(
                    generated_samples_i_d, real_data.sg_ind,
                    self.lattice_means, self.lattice_stds,
                    self.sym_info, self.supercell_builder.asym_unit_dict,
                    rescale_asymmetric_unit=False, destandardize=True, mode='hard')

                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',
                                                     np.ones(len(generated_samples_i)) * 2, mode='extend')
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'distortion level',
                                                     torch.linalg.norm(distortion, axis=-1).cpu().detach().numpy(),
                                                     mode='extend')

        return generated_samples_i.float().detach(), epoch_stats_dict, negative_type, real_data

    def nov_22_figures(self):
        """
        make beautiful figures for the first paper
        """
        import plotly.io as pio
        pio.renderers.default = 'browser'

        # figures from the late 2022 JCTC draft submissions
        with wandb.init(config=self.config, project=self.config.wandb.project_name,
                        entity=self.config.wandb.username, tags=[self.config.wandb.experiment_tag]):
            wandb.run.name = wandb.config.machine + '_' + str(
                wandb.config.run_num)  # overwrite procedurally generated run name with our run name
            wandb.run.save()

            # self.nice_dataset_analysis(self.prep_dataset)
            self.misc_pre_training_items()
            from reporting.nov_22_regressor import nov_22_paper_regression_plots
            nov_22_paper_regression_plots(self.config)
            from reporting.nov_22_discriminator_final import nov_22_paper_discriminator_plots
            nov_22_paper_discriminator_plots(self.config, wandb)

        return

    def slash_batch(self, train_loader, test_loader, slash_fraction):
        slash_increment = max(4, int(train_loader.batch_size * slash_fraction))
        train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size - slash_increment)
        test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size - slash_increment)
        print('==============================')
        print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
        print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
        print('==============================')
        wandb.log({'batch size': train_loader.batch_size})

        return train_loader, test_loader

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

    def check_model_convergence(self, metrics_dict):
        generator_convergence = check_convergence(metrics_dict['generator test loss'], self.config.history,
                                                  self.config.generator_optimizer.convergence_eps)
        discriminator_convergence = check_convergence(metrics_dict['discriminator test loss'], self.config.history,
                                                      self.config.discriminator_optimizer.convergence_eps)
        regressor_convergence = check_convergence(metrics_dict['regressor test loss'], self.config.history,
                                                  self.config.regressor_optimizer.convergence_eps)

        return generator_convergence, discriminator_convergence, regressor_convergence

    def model_checkpointing(self, epoch, config, discriminator, generator, regressor,
                            discriminator_optimizer, generator_optimizer, regressor_optimizer,
                            generator_err_te, discriminator_err_te, regressor_err_te, metrics_dict):
        if config.save_checkpoints:  # config.machine == 'cluster':  # every 5 epochs, save a checkpoint
            # if (epoch > 0) and (epoch % 5 == 0):
            #     # saving early-stopping checkpoint
            #     save_checkpoint(epoch, discriminator, discriminator_optimizer, self.config.discriminator.__dict__, 'discriminator_' + str(config.run_num) + f'_epoch_{epoch}')
            #     save_checkpoint(epoch, generator, generator_optimizer, self.config.generator.__dict__, 'generator_' + str(config.run_num) + f'_epoch_{epoch}')

            # or save any checkpoint which is a new best
            if epoch > 0:
                if np.average(discriminator_err_te) < np.amin(metrics_dict['discriminator test loss'][:-1]):
                    print("Saving discriminator checkpoint")
                    save_checkpoint(epoch, discriminator, discriminator_optimizer, self.config.discriminator.__dict__,
                                    'best_discriminator_' + str(config.run_num))
                if np.average(generator_err_te) < np.amin(metrics_dict['generator test loss'][:-1]):
                    print("Saving generator checkpoint")
                    save_checkpoint(epoch, generator, generator_optimizer, self.config.generator.__dict__,
                                    'best_generator_' + str(config.run_num))
                if np.average(regressor_err_te) < np.amin(metrics_dict['regressor test loss'][:-1]):
                    print("Saving regressor checkpoint")
                    save_checkpoint(epoch, regressor, regressor_optimizer, self.config.regressor.__dict__,
                                    'best_regressor_' + str(config.run_num))

        return None

    def update_lr(self, discriminator_schedulers, discriminator_optimizer, discriminator_err_tr,
                  discriminator_hit_max_lr,
                  generator_schedulers, generator_optimizer, generator_err_tr, generator_hit_max_lr,
                  regressor_schedulers, regressor_optimizer, regressor_err_tr, regressor_hit_max_lr,
                  ):  # update learning rate

        discriminator_optimizer, discriminator_lr = set_lr(discriminator_schedulers, discriminator_optimizer,
                                                           self.config.discriminator_optimizer.lr_schedule,
                                                           self.config.discriminator_optimizer.min_lr,
                                                           self.config.discriminator_optimizer.max_lr,
                                                           discriminator_err_tr, discriminator_hit_max_lr)
        discriminator_learning_rate = discriminator_optimizer.param_groups[0]['lr']
        if discriminator_learning_rate >= self.config.discriminator_optimizer.max_lr:
            discriminator_hit_max_lr = True

        generator_optimizer, generator_lr = set_lr(generator_schedulers, generator_optimizer,
                                                   self.config.generator_optimizer.lr_schedule,
                                                   self.config.generator_optimizer.min_lr,
                                                   self.config.generator_optimizer.max_lr, generator_err_tr,
                                                   generator_hit_max_lr)
        generator_learning_rate = generator_optimizer.param_groups[0]['lr']
        if generator_learning_rate >= self.config.generator_optimizer.max_lr:
            generator_hit_max_lr = True

        regressor_optimizer, regressor_lr = set_lr(regressor_schedulers, regressor_optimizer,
                                                   self.config.regressor_optimizer.lr_schedule,
                                                   self.config.regressor_optimizer.min_lr,
                                                   self.config.regressor_optimizer.max_lr, regressor_err_tr,
                                                   regressor_hit_max_lr)
        regressor_learning_rate = regressor_optimizer.param_groups[0]['lr']
        if regressor_learning_rate >= self.config.regressor_optimizer.max_lr:
            regressor_hit_max_lr = True

        return discriminator_optimizer, discriminator_learning_rate, discriminator_hit_max_lr, \
            generator_optimizer, generator_learning_rate, generator_hit_max_lr, \
            regressor_optimizer, regressor_learning_rate, regressor_hit_max_lr

    def reload_best_test_checkpoint(self, epoch, generator, discriminator):
        # reload best test
        if epoch != 0:  # if we have trained at all, reload the best model
            generator_path = f'../models/generator_{self.config.run_num}'
            discriminator_path = f'../models/discriminator_{self.config.run_num}'
            if os.path.exists(generator_path):
                generator_checkpoint = torch.load(generator_path)
                if list(generator_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                    for i in list(generator_checkpoint['model_state_dict']):
                        generator_checkpoint['model_state_dict'][i[7:]] = generator_checkpoint['model_state_dict'].pop(i)
                generator.load_state_dict(generator_checkpoint['model_state_dict'])

            if os.path.exists(discriminator_path):
                discriminator_checkpoint = torch.load(discriminator_path)
                if list(discriminator_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                    for i in list(discriminator_checkpoint['model_state_dict']):
                        discriminator_checkpoint['model_state_dict'][i[7:]] = discriminator_checkpoint['model_state_dict'].pop(i)
                discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])

        return generator, discriminator

    def gan_evaluation(self, epoch, generator, discriminator, test_loader, extra_test_loader, regressor):
        """
        run post-training evaluation
        """
        generator, discriminator = self.reload_best_test_checkpoint(epoch, generator, discriminator)

        # rerun test inference
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            test_loss, test_loss_record, test_epoch_stats_dict, time_test = \
                self.run_epoch(data_loader=test_loader, generator=generator, discriminator=discriminator,
                               update_gradients=False, record_stats=True, epoch=epoch, regressor=regressor)  # compute loss on test set

            np.save(f'../{self.config.run_num}_test_epoch_stats_dict', test_epoch_stats_dict)

            # sometimes test the generator on a mini CSP problem
            if (self.config.mode == 'gan') and self.train_generator:
                self.mini_csp(extra_test_loader if extra_test_loader is not None else test_loader, generator, discriminator)

            if extra_test_loader is not None:
                # extra_test_epoch_stats_dict = np.load('C:/Users\mikem\crystals\CSP_runs/1513_extra_test_dict.npy',allow_pickle=True).item()  # we already have it

                _, _, extra_test_epoch_stats_dict, extra_time_test = \
                    self.run_epoch(data_loader=extra_test_loader, generator=generator, discriminator=discriminator,
                                   update_gradients=False, record_stats=True, epoch=epoch)  # compute loss on test set
                np.save(f'../{self.config.run_num}_extra_test_dict', extra_test_epoch_stats_dict)
            else:
                extra_test_epoch_stats_dict = None

        detailed_reporting(self.config, epoch, test_loader, None, test_epoch_stats_dict, extra_test_dict=extra_test_epoch_stats_dict)

    @staticmethod
    def compute_similarity_penalty(generated_samples, prior):
        """
        punish batches in which the samples are too self-similar

        Parameters
        ----------
        generated_samples
        prior

        Returns
        -------
        """
        if len(generated_samples) >= 3:
            # enforce that the distance between samples is similar to the distance between priors
            prior_dists = torch.cdist(prior, prior, p=2)
            sample_dists = torch.cdist(generated_samples, generated_samples, p=2)
            similarity_penalty = F.smooth_l1_loss(input=sample_dists, target=prior_dists, reduction='none').mean(
                1)  # align distances to all other samples

            # todo this metric isn't very good e.g., doesn't set the standardization for each space group individually (different stats and distances)
            # also the distance between the different cell params are not at all equally meaningful

        else:
            similarity_penalty = None

        return similarity_penalty

    def score_adversarially(self, supercell_data, discriminator, discriminator_noise=None):
        """
        get an adversarial score for generated samples

        Parameters
        ----------
        supercell_data
        discriminator

        Returns
        -------

        """
        if supercell_data is not None:  # if we built the supercells, we'll want to do this analysis anyway
            if discriminator_noise is not None:
                supercell_data.pos += torch.randn_like(
                    supercell_data.pos) * discriminator_noise
            else:
                if self.config.discriminator_positional_noise > 0:
                    supercell_data.pos += torch.randn_like(
                        supercell_data.pos) * self.config.discriminator_positional_noise

            if (self.config.device.lower() == 'cuda') and (supercell_data.x.device != 'cuda'):
                supercell_data = supercell_data.cuda()

            if self.config.test_mode or self.config.anomaly_detection:
                assert torch.sum(torch.isnan(supercell_data.x)) == 0, "NaN in training input"

            discriminator_score, dist_dict = self.adversarial_score(discriminator, supercell_data)
        else:
            discriminator_score = None
            dist_dict = None

        return discriminator_score, dist_dict

    def aggregate_generator_losses(self, epoch_stats_dict, packing_loss, discriminator_raw_output, vdw_loss,
                                   similarity_penalty, packing_prediction, packing_target, h_bond_score):
        generator_losses_list = []
        stats_keys, stats_values = [], []
        if packing_loss is not None:
            stats_keys += ['generator packing loss', 'generator packing prediction',
                           'generator packing target', 'generator packing mae']
            stats_values += [packing_loss.cpu().detach().numpy(), packing_prediction,
                             packing_target, np.abs(packing_prediction - packing_target) / packing_target]

            if True:  # enforce the target density all the time
                generator_losses_list.append(packing_loss.float())

        if discriminator_raw_output is not None:
            if self.config.generator_adversarial_loss_func == 'hot softmax':
                adversarial_loss = 1 - F.softmax(discriminator_raw_output / 5, dim=1)[:, 1]  # high temp smears out the function over a wider range
            elif self.config.generator_adversarial_loss_func == 'minimax':
                softmax_adversarial_score = F.softmax(discriminator_raw_output, dim=1)[:, 1]  # modified minimax
                adversarial_loss = -torch.log(softmax_adversarial_score)  # modified minimax
            elif self.config.generator_adversarial_loss_func == 'score':
                adversarial_loss = -softmax_and_score(discriminator_raw_output)  # linearized score
            elif self.config.generator_adversarial_loss_func == 'softmax':
                adversarial_loss = 1 - F.softmax(discriminator_raw_output, dim=1)[:, 1]
            else:
                print(f'{self.config.generator_adversarial_loss_func} is not an implemented adversarial loss')
                sys.exit()

            stats_keys += ['generator adversarial loss']
            stats_values += [adversarial_loss.cpu().detach().numpy()]
            stats_keys += ['generator adversarial score']
            stats_values += [discriminator_raw_output.cpu().detach().numpy()]

            if self.config.train_generator_adversarially:
                generator_losses_list.append(adversarial_loss)

        if vdw_loss is not None:
            stats_keys += ['generator per mol vdw loss']
            stats_values += [vdw_loss.cpu().detach().numpy()]

            if self.config.train_generator_vdw:
                # if self.config.vdw_loss_func == 'log':
                #     vdw_loss_f = torch.log(1 + vdw_loss)  # soft rescaling to be gentler on outliers
                # elif self.config.vdw_loss_func is None:
                #     vdw_loss_f = vdw_loss
                # elif self.config.vdw_loss_func == 'mse':
                #     vdw_loss_f = vdw_loss ** 2

                generator_losses_list.append(vdw_loss)

        if h_bond_score is not None:
            if self.config.train_generator_h_bond:
                generator_losses_list.append(h_bond_score)

            stats_keys += ['generator h bond loss']
            stats_values += [h_bond_score.cpu().detach().numpy()]

        if similarity_penalty is not None:
            stats_keys += ['generator similarity loss']
            stats_values += [similarity_penalty.cpu().detach().numpy()]

            if self.config.generator_similarity_penalty != 0:
                if similarity_penalty is not None:
                    generator_losses_list.append(self.config.generator_similarity_penalty * similarity_penalty)
                else:
                    print('similarity penalty was none')

        generator_losses = torch.sum(torch.stack(generator_losses_list), dim=0)
        epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values, mode='extend')

        return generator_losses, epoch_stats_dict

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

    def reload_model_checkpoints(self, config):
        if config.generator_path is not None:
            generator_checkpoint = torch.load(config.generator_path)
            config.generator = Namespace(**generator_checkpoint['config'])  # overwrite the settings for the model

        if config.discriminator_path is not None:
            discriminator_checkpoint = torch.load(config.discriminator_path)
            config.discriminator = Namespace(**discriminator_checkpoint['config'])

        if config.regressor_path is not None:
            regressor_checkpoint = torch.load(config.regressor_path)
            config.regressor = Namespace(**regressor_checkpoint['config'])  # overwrite the settings for the model

        return config

    def mini_csp(self, data_loader, generator, discriminator, regressor=None):
        print('Starting Mini CSP')
        generator.eval()
        discriminator.eval()
        real_data = next(iter(data_loader)).clone().detach().to(self.config.device)

        rdf_bins = 100
        rdf_range = [0, 10]

        real_samples_dict = self.analyze_real_crystals(real_data, discriminator, rdf_bins, rdf_range)
        generated_samples_dict = self.generate_mini_csp_samples(real_data, generator, discriminator, regressor)

        log_mini_csp_scores_distributions(self.config, wandb, generated_samples_dict, real_samples_dict, real_data)
        log_best_mini_csp_samples(self.config, wandb, discriminator, generated_samples_dict, real_samples_dict, real_data, self.supercell_builder, self.tracking_mol_volume_ind, self.sym_info, self.vdw_radii)

        return None

    def analyze_real_crystals(self, real_data, discriminator, rdf_bins, rdf_range):
        real_supercell_data = self.supercell_builder.unit_cell_to_supercell(real_data, self.config.supercell_size, self.config.discriminator.graph_convolution_cutoff)

        discriminator_score, dist_dict = self.score_adversarially(real_supercell_data.clone(), discriminator)
        h_bond_score = compute_h_bond_score(self.config.feature_richness, self.tracking_atom_acceptor_ind, self.tracking_atom_donor_ind, self.tracking_num_acceptors_ind, self.tracking_num_donors_ind, real_supercell_data)
        vdw_penalty, normed_vdw_penalty = get_vdw_penalty(self.vdw_radii, dist_dict, real_data.num_graphs, real_data.mol_size,
                                                          loss_func=None)
        real_rdf, rr, atom_inds = crystal_rdf(real_supercell_data, rrange=rdf_range,
                                              bins=rdf_bins, mode='intermolecular',
                                              raw_density=True, atomwise=True, cpu_detach=True)

        volumes_list = []
        for i in range(real_data.num_graphs):
            volumes_list.append(cell_vol_torch(real_data.cell_params[i, 0:3], real_data.cell_params[i, 3:6]))
        volumes = torch.stack(volumes_list)
        real_packing_coeffs = real_data.Z * real_data.tracking[:, self.tracking_mol_volume_ind] / volumes

        real_samples_dict = {'score': softmax_and_score(discriminator_score).cpu().detach().numpy(),
                             'vdw overlap': vdw_penalty.cpu().detach().numpy(),
                             'density': real_packing_coeffs.cpu().detach().numpy(),
                             'h bond score': h_bond_score.cpu().detach().numpy(),
                             'cell params': real_data.cell_params.cpu().detach().numpy(),
                             'space group': real_data.sg_ind.cpu().detach().numpy(),
                             'RDF': real_rdf
                             }
        return real_samples_dict

    def generate_mini_csp_samples(self, real_data, generator, discriminator, regressor, sample_source='generator'):
        num_molecules = real_data.num_graphs
        n_sampling_iters = self.config.sample_steps
        sampling_dict = {'score': np.zeros((num_molecules, n_sampling_iters)),
                         'vdw overlap': np.zeros((num_molecules, n_sampling_iters)),
                         'density': np.zeros((num_molecules, n_sampling_iters)),
                         'h bond score': np.zeros((num_molecules, n_sampling_iters)),
                         'cell params': np.zeros((num_molecules, n_sampling_iters, 12)),
                         'space group': np.zeros((num_molecules, n_sampling_iters)),
                         'handedness': np.zeros((num_molecules, n_sampling_iters)),
                         'distortion_size': np.zeros((num_molecules, n_sampling_iters)),
                         }

        with torch.no_grad():
            for ii in tqdm.tqdm(range(n_sampling_iters)):
                fake_data = real_data.clone().to(self.config.device)

                if sample_source == 'generator':
                    # use generator to make samples
                    samples, prior, standardized_target_packing_coeff, fake_data = \
                        self.get_generator_samples(fake_data, generator, regressor)

                    fake_supercell_data, generated_cell_volumes, _ = \
                        self.supercell_builder.build_supercells(
                            fake_data, samples, self.config.supercell_size,
                            self.config.discriminator.graph_convolution_cutoff,
                            align_molecules=False,
                        )

                elif sample_source == 'distorted':
                    # test - do slight distortions on existing crystals
                    generated_samples_ii = (real_data.cell_params - self.lattice_means) / self.lattice_stds

                    if True:  # self.config.sample_distortion_magnitude == -1:
                        distortion = torch.randn_like(generated_samples_ii) * torch.logspace(-4, 1, len(generated_samples_ii)).to(generated_samples_ii.device)[:, None]  # wider range
                        distortion = distortion[torch.randperm(len(distortion))]
                    else:
                        distortion = torch.randn_like(generated_samples_ii) * self.config.sample_distortion_magnitude

                    generated_samples_i_d = (generated_samples_ii + distortion).to(self.config.device)  # add jitter and return in standardized basis

                    generated_samples_i = clean_cell_params(
                        generated_samples_i_d, real_data.sg_ind,
                        self.lattice_means, self.lattice_stds,
                        self.sym_info, self.supercell_builder.asym_unit_dict,
                        rescale_asymmetric_unit=False, destandardize=True, mode='hard')

                    fake_supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
                        fake_data, generated_samples_i, self.config.supercell_size,
                        self.config.discriminator.graph_convolution_cutoff,
                        align_molecules=True,
                        target_handedness=real_data.asym_unit_handedness,
                    )
                    sampling_dict['distortion_size'][:, ii] = torch.linalg.norm(distortion, axis=-1).cpu().detach().numpy()
                    # end test

                discriminator_score, dist_dict = self.score_adversarially(fake_supercell_data.clone(), discriminator, discriminator_noise=0)
                h_bond_score = compute_h_bond_score(self.config.feature_richness, self.tracking_atom_acceptor_ind, self.tracking_atom_donor_ind,
                                                    self.tracking_num_acceptors_ind, self.tracking_num_donors_ind, fake_supercell_data)
                vdw_penalty, normed_vdw_penalty = get_vdw_penalty(self.vdw_radii, dist_dict, fake_data.num_graphs, fake_data.mol_size,
                                                                  loss_func=None)

                volumes_list = []
                for i in range(fake_data.num_graphs):
                    volumes_list.append(
                        cell_vol_torch(fake_supercell_data.cell_params[i, 0:3], fake_supercell_data.cell_params[i, 3:6]))
                volumes = torch.stack(volumes_list)

                # todo possible issue here with division by two - make sure Z assignment is consistent throughout
                fake_packing_coeffs = fake_supercell_data.Z * fake_supercell_data.tracking[:, self.tracking_mol_volume_ind] / volumes

                sampling_dict['score'][:, ii] = softmax_and_score(discriminator_score).cpu().detach().numpy()
                sampling_dict['vdw overlap'][:, ii] = vdw_penalty.cpu().detach().numpy()
                sampling_dict['density'][:, ii] = fake_packing_coeffs.cpu().detach().numpy()
                sampling_dict['h bond score'][:, ii] = h_bond_score.cpu().detach().numpy()
                sampling_dict['cell params'][:, ii, :] = fake_supercell_data.cell_params.cpu().detach().numpy()
                sampling_dict['space group'][:, ii] = fake_supercell_data.sg_ind.cpu().detach().numpy()
                sampling_dict['handedness'][:, ii] = fake_supercell_data.asym_unit_handedness.cpu().detach().numpy()

        return sampling_dict


def update_gan_metrics(epoch, metrics_dict,
                       discriminator_lr, generator_lr, regressor_lr,
                       discriminator_train_loss, discriminator_test_loss,
                       generator_train_loss, generator_test_loss,
                       regressor_train_loss, regressor_test_loss
                       ):

    metrics_keys = ['epoch',
                    'discriminator learning rate', 'generator learning rate',
                    'regressor learning rate',
                    'discriminator train loss', 'discriminator test loss',
                    'generator train loss', 'generator test loss',
                    'regressor train loss', 'regressor test loss'
                    ]
    metrics_vals = [epoch, discriminator_lr, generator_lr, regressor_lr,
                    discriminator_train_loss, discriminator_test_loss,
                    generator_train_loss, generator_test_loss,
                    regressor_train_loss, regressor_test_loss
                    ]

    metrics_dict = update_stats_dict(metrics_dict, metrics_keys, metrics_vals)

    return metrics_dict

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime
import numpy as np
import glob
from torch import backends, optim
from torch_geometric.loader.dataloader import Collater
from pyxtal import symmetry
import ase.io
from ase.visualize import view
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.Draw
from scipy.stats import linregress
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import n_colors
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from PIL import Image

from utils import *

from dataset_management.dataset_utils import BuildDataset, get_dataloaders, update_dataloader_batch_size, get_extra_test_loader
from dataset_management.dataset_manager import Miner

from crystal_building.crystal_builder_tools import *
from crystal_building.coordinate_transformations import cell_vol
from crystal_building.supercell_builders import SupercellBuilder, override_sg_info

from models.model_utils import *
from models.crystal_rdf import crystal_rdf
from models.vdw_overlap import vdw_overlap, raw_vdw_overlap
from models.generator_models import crystal_generator
from models.autoencoder_model import molecule_autoencoder
from models.discriminator_models import crystal_discriminator
from models.regression_models import molecule_regressor
from models.torch_models import independent_gaussian_model
from sampling.MCMC_Sampling import mcmcSampler
from sampling.SampleOptimization import gradient_descent_sampling

import wandb



class Modeller():
    def __init__(self, config):
        self.config = config
        setup_outputs = self.setup()
        if self.config.skip_saving_and_loading:
            self.prep_dataset = setup_outputs
        else:
            self.prep_dataset = None

    def setup(self):
        '''
        setup working directory
        move to relevant directory
        :return:
        '''

        '''
        load lots of relevant physical data
        '''

        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        periodicTable = rdkit.Chem.GetPeriodicTable()
        self.atom_weights = {}
        self.vdw_radii = {}
        for i in range(100):
            self.atom_weights[i] = periodicTable.GetAtomicWeight(i)
            self.vdw_radii[i] = periodicTable.GetRvdw(i)

        if os.path.exists('symmetry_info.npy'):
            sym_info = np.load('symmetry_info.npy', allow_pickle=True).item()
            self.sym_ops = sym_info['sym_ops']
            self.point_groups = sym_info['point_groups']
            self.lattice_type = sym_info['lattice_type']
            self.space_groups = sym_info['space_groups']

            self.sym_info = {}
            self.sym_info['sym_ops'] = self.sym_ops
            self.sym_info['point_groups'] = self.point_groups
            self.sym_info['lattice_type'] = self.lattice_type
            self.sym_info['space_groups'] = self.space_groups
        else:
            print('Pre-generating spacegroup symmetries')
            self.sym_ops = {}
            self.point_groups = {}
            self.lattice_type = {}
            self.space_groups = {}
            self.space_group_indices = {}
            for i in tqdm.tqdm(range(1, 231)):
                sym_group = symmetry.Group(i)
                general_position_syms = sym_group.wyckoffs_organized[0][0]
                self.sym_ops[i] = [general_position_syms[i].affine_matrix for i in range(len(general_position_syms))]  # first 0 index is for general position, second index is superfluous, third index is the symmetry operation
                self.point_groups[i] = sym_group.point_group
                self.lattice_type[i] = sym_group.lattice_type
                self.space_groups[i] = sym_group.symbol
                self.space_group_indices[sym_group.symbol] = i

            self.sym_info = {}
            self.sym_info['sym_ops'] = self.sym_ops
            self.sym_info['point_groups'] = self.point_groups
            self.sym_info['lattice_type'] = self.lattice_type
            self.sym_info['space_groups'] = self.space_groups
            self.sym_info['space_group_indices'] = self.space_group_indices
            np.save('symmetry_info', self.sym_info)

        # initialize fractional lattice vectors - should be exactly identical to what's in molecule_featurizer.py
        supercell_scale = 2  # t
        n_cells = (2 * supercell_scale + 1) ** 3

        fractional_translations = np.zeros((n_cells, 3))  # initialize the translations in fractional coords
        i = 0
        for xx in range(-supercell_scale, supercell_scale + 1):
            for yy in range(-supercell_scale, supercell_scale + 1):
                for zz in range(-supercell_scale, supercell_scale + 1):
                    fractional_translations[i] = np.array((xx, yy, zz))
                    i += 1
        self.lattice_vectors = torch.Tensor(fractional_translations[np.argsort(np.abs(fractional_translations).sum(1))][1:])  # leave out the 0,0,0 element
        self.normed_lattice_vectors = self.lattice_vectors / torch.linalg.norm(self.lattice_vectors, axis=1)[:, None]

        '''
        prepare to load dataset
        '''
        miner = Miner(config=self.config, dataset_path=self.config.dataset_path, collect_chunks=False)

        if (self.config.run_num == 0) or (self.config.explicit_run_enumeration == True):  # if making a new workdir
            if self.config.run_num == 0:
                self.makeNewWorkingDirectory()
            else:
                self.workDir = self.config.workdir + '/run%d' % self.config.run_num  # explicitly enumerate the new run directory
                os.mkdir(self.workDir)

            os.mkdir(self.workDir + '/ckpts')
            os.mkdir(self.workDir + '/datasets')
            os.chdir(self.workDir)  # move to working dir
            print('Starting Fresh Run %d' % self.config.run_num)
            t0 = time.time()
            if self.config.skip_saving_and_loading:
                dataset = miner.load_for_modelling(return_dataset=True, save_dataset=False)
                del miner
                return dataset
            else:
                miner.load_for_modelling(return_dataset=False, save_dataset=True)
            print('Initializing dataset took {} seconds'.format(int(time.time() - t0)))
        else:
            print("Must provide a run_num if not creating a new workdir!")

    def makeNewWorkingDirectory(self):  # make working directory
        '''
        make a new working directory
        non-overlapping previous entries
        :return:
        '''
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

    def update_gan_metrics(self, epoch, metrics_dict,
                           discriminator_lr, generator_lr, conditioner_lr, regressor_lr,
                           discriminator_train_loss, discriminator_test_loss,
                           generator_train_loss, generator_test_loss,
                           conditioner_train_loss, conditioner_test_loss,
                           regressor_train_loss, regressor_test_loss
                           ):

        metrics_keys = ['epoch',
                        'discriminator learning rate', 'generator learning rate',
                        'conditioner learning rate', 'regressor learning rate',
                        'discriminator train loss', 'discriminator test loss',
                        'generator train loss', 'generator test loss',
                        'conditioner train loss', 'conditioner test loss',
                        'regressor train loss', 'regressor test loss'
                        ]
        metrics_vals = [epoch, discriminator_lr, generator_lr, conditioner_lr, regressor_lr,
                        discriminator_train_loss, discriminator_test_loss,
                        generator_train_loss, generator_test_loss,
                        conditioner_train_loss, conditioner_test_loss,
                        regressor_train_loss, regressor_test_loss
                        ]

        metrics_dict = update_stats_dict(metrics_dict, metrics_keys, metrics_vals)

        return metrics_dict

    def init_models(self):
        '''
        Initialize model and optimizer
        :return:
        '''
        self.config = self.reload_model_checkpoints(self.config)

        generator, discriminator, conditioner, regressor = nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1) # init dummy models
        print("Initializing model(s) for " + self.config.mode)
        if self.config.mode == 'gan' or self.config.mode == 'sampling':
            generator = crystal_generator(self.config, self.config.dataDims)
            discriminator = crystal_discriminator(self.config, self.config.dataDims)
        elif self.config.mode == 'regression':
            regressor = molecule_regressor(self.config, self.config.dataDims)
        elif self.config.mode == 'autoencoder':
            self.init_conditioner_classes()
            conditioner = molecule_autoencoder(self.config, self.config.dataDims)
        else:
            print(f'{self.config.mode} is not an implemented method!')
            sys.exit()

        if self.config.device.lower() == 'cuda':
            print('Putting models on CUDA')
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            conditioner = conditioner.cuda()
            regressor = regressor.cuda()

        generator_optimizer = init_optimizer(self.config.generator_optimizer, generator)
        discriminator_optimizer = init_optimizer(self.config.discriminator_optimizer, discriminator)
        conditioner_optimizer = init_optimizer(self.config.conditioner_optimizer, conditioner)
        regressor_optimizer = init_optimizer(self.config.regressor_optimizer, regressor)

        if self.config.generator_path is not None and self.config.mode == 'gan':
            generator, generator_optimizer = reload_model(generator, generator_optimizer, self.config.generator_path)
        if self.config.discriminator_path is not None and self.config.mode == 'gan':
            discriminator, discriminator_optimizer = reload_model(discriminator, discriminator_optimizer, self.config.discriminator_path)
        if self.config.conditioner_path is not None and self.config.mode == 'autoencoder':
            conditioner, conditioner_optimizer = reload_model(conditioner, conditioner_optimizer, self.config.conditioner_path)
        if self.config.regressor_path is not None and self.config.mode == 'regression':
            regressor, regressor_optimizer = reload_model(discriminator, regressor_optimizer, self.config.regressor_path)

        if self.config.mode == 'gan' and self.config.conditioner_path is not None:
            checkpoint = torch.load(self.config.conditioner_path)
            if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(checkpoint['model_state_dict']):
                    checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

            conditioner_model_state = {key[12:]:value for key, value in checkpoint['model_state_dict'].items() if 'conditioner' in key}
            generator.conditioner.load_state_dict(conditioner_model_state)
            if self.config.freeze_generator_conditioner:
                generator.conditioner.requires_grad_(False) # todo confirm this works
                generator_optimizer = init_optimizer(self.config.generator_optimizer, generator, freeze_params = True)

        generator_schedulers = init_schedulers(self.config.generator_optimizer, generator_optimizer)
        discriminator_schedulers = init_schedulers(self.config.discriminator_optimizer, discriminator_optimizer)
        conditioner_schedulers = init_schedulers(self.config.conditioner_optimizer, conditioner_optimizer)
        regressor_schedulers = init_schedulers(self.config.regressor_optimizer, regressor_optimizer)

        num_params = [get_n_config(model) for model in [generator, discriminator, conditioner, regressor]]
        print('Generator model has {:.3f} million or {} parameters'.format(num_params[0] / 1e6, int(num_params[0])))
        print('Discriminator model has {:.3f} million or {} parameters'.format(num_params[1] / 1e6, int(num_params[1])))
        print('conditioner model has {:.3f} million or {} parameters'.format(num_params[2] / 1e6, int(num_params[2])))
        print('Regressor model has {:.3f} million or {} parameters'.format(num_params[3] / 1e6, int(num_params[3])))

        return generator, discriminator, conditioner, regressor, \
               generator_optimizer, generator_schedulers, \
               discriminator_optimizer, discriminator_schedulers, \
               conditioner_optimizer, conditioner_schedulers, \
               regressor_optimizer, regressor_schedulers, \
               num_params

    def get_batch_size(self, generator, discriminator, generator_optimizer, discriminator_optimizer, dataset, config):
        '''
        try larger batches until it crashes
        '''
        finished = False
        init_batch_size = self.config.min_batch_size.real
        max_batch_size = self.config.max_batch_size.real
        batch_reduction_factor = self.config.auto_batch_reduction

        train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=init_batch_size)

        increment = 1.5  # what fraction by which to increment the batch size
        batch_size = int(init_batch_size)

        while (not finished) and (batch_size < max_batch_size):
            self.config.current_batch_size = batch_size

            if self.config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU cache
                generator.cuda()
                discriminator.cuda()

            try:
                _ = self.run_epoch(dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                   generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                                   update_gradients=True, record_stats=True, iteration_override=2, epoch=1)

                # if successful, increase the batch and try again
                batch_size = max(batch_size + 5, int(batch_size * increment))
                train_loader = update_dataloader_batch_size(train_loader, batch_size)
                test_loader = update_dataloader_batch_size(test_loader, batch_size)
                # train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=batch_size)

                print('Training batch size increased to {}'.format(batch_size))

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    finished = True
                    batch_size = int(batch_size / increment)
                else:
                    raise e

        # once it hits OOM or the maximum batch size, take it as final
        # reduce by a certain factor to give room for different size graphs
        if batch_size < 10:
            leeway = batch_reduction_factor / 2
        elif batch_size > 20:
            leeway = batch_reduction_factor
        else:
            leeway = batch_reduction_factor / 1.33

        batch_size = max(1, int(batch_size * leeway))

        print('Final batch size is {}'.format(batch_size))

        tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)

        if self.config.device.lower() == 'cuda':
            torch.cuda.empty_cache()  # clear GPU cache

        return tr, te, batch_size

    def train(self):
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=[self.config.wandb.experiment_tag]):
            wandb.run.name = wandb.config.machine + '_' + str(wandb.config.run_num)  # overwrite procedurally generated run name with our run name
            wandb.run.save()
            # config = wandb.config # FLAG wandb configs don't support nested namespaces. Sweeps are officially broken - look at the github thread to see if they fix it

            dataset_builder = self.training_prep()
            generator, discriminator, conditioner, regressor, \
            generator_optimizer, generator_schedulers, \
            discriminator_optimizer, discriminator_schedulers, \
            conditioner_optimizer, conditioner_schedulers, \
            regressor_optimizer, regressor_schedulers, \
            num_params = self.init_models()

            # get batch size
            # if self.config.auto_batch_sizing: # todo rebuild for all types of model
            #     print('Finding optimal batch size')
            #     train_loader, test_loader, self.config.final_batch_size = \
            #         self.get_batch_size(generator, discriminator, generator_optimizer, discriminator_optimizer,
            #                             dataset_builder, self.config)
            #
            #     # reload preloaded models (there are weight updates in batch sizing)
            #     if self.config.generator_path is not None:
            #         generator, generator_optimizer = reload_model(generator, generator_optimizer, self.config.generator_path)
            #     if self.config.discriminator_path is not None:
            #         discriminator, discriminator_optimizer = reload_model(discriminator, discriminator_optimizer, self.config.discriminator_path)
            # else:
            print('Getting dataloaders for pre-determined batch size')
            train_loader, test_loader = get_dataloaders(dataset_builder, self.config)
            self.config.current_batch_size = self.config.min_batch_size

            del dataset_builder
            print("Training batch size set to {}".format(self.config.current_batch_size))

            extra_test_loader = None
            if (self.config.extra_test_set_paths is not None) and self.config.extra_test_evaluation:
                extra_test_loader = get_extra_test_loader(self.config, self.config.extra_test_set_paths, dataDims=self.config.dataDims,
                                                          pg_dict=self.point_groups, sg_dict=self.space_groups, lattice_dict=self.lattice_type)

            generator, discriminator, conditioner, regressor = self.reinitialize_models(
                generator, discriminator, conditioner, regressor)

            wandb.watch((generator, discriminator, conditioner, regressor), log_graph=True, log_freq=100)
            wandb.log({"Model Num Parameters": np.sum(np.asarray(num_params)),
                       "Final Batch Size": self.config.current_batch_size})

            metrics_dict = {}

            # training loop
            discriminator_hit_max_lr, generator_hit_max_lr, conditioner_hit_max_lr, regressor_hit_max_lr, converged, epoch = \
                False, False, False, False, self.config.max_epochs == 0, 0

            generator_err_tr, discriminator_err_tr, conditioner_err_tr, regressor_err_tr = 0,0,0,0# [0], [0], [0], [0]
            generator_err_te, discriminator_err_te, conditioner_err_te, regressor_err_te = 0,0,0,0#[0], [0], [0], [0]
            generator_tr_record, discriminator_tr_record, conditioner_tr_record, regressor_tr_record = [0], [0], [0], [0]
            generator_te_record, discriminator_te_record, conditioner_te_record, regressor_te_record = [0], [0], [0], [0]

            with torch.autograd.set_detect_anomaly(self.config.anomaly_detection):
                while (epoch < self.config.max_epochs) and not converged:
                    self.epoch = epoch
                    # very cool
                    print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                    print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                    print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                    # very cool
                    print("Starting Epoch {}".format(epoch))  # index from 0

                    extra_test_epoch_stats_dict = None
                    try:  # try this batch size
                        # train & compute train loss
                        train_loss, train_loss_record, train_epoch_stats_dict, time_train = \
                            self.run_epoch(dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                           regressor=regressor, conditioner=conditioner,
                                           generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                                           conditioner_optimizer=conditioner_optimizer, regressor_optimizer=regressor_optimizer,
                                           update_gradients=True, record_stats=True, epoch=epoch)

                        with torch.no_grad():
                            # compute test loss
                            test_loss, test_loss_record, test_epoch_stats_dict, time_test = \
                                self.run_epoch(dataLoader=test_loader, generator=generator, discriminator=discriminator,
                                               regressor=regressor, conditioner=conditioner,
                                               generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
                                               conditioner_optimizer=conditioner_optimizer, regressor_optimizer=regressor_optimizer,
                                               update_gradients=False, record_stats=True, epoch=epoch)

                            if (extra_test_loader is not None) and (epoch % self.config.extra_test_period == 0):
                                extra_test_epoch_stats_dict, time_test_ex = \
                                    self.discriminator_evaluation(dataLoader=extra_test_loader, discriminator=discriminator)  # compute loss on test set
                                print(f'Extra test evaluation took {time_test_ex:.1f} seconds')

                        print('epoch={}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, time_train, time_test))

                        if self.config.mode == 'gan':
                            discriminator_err_tr, generator_err_tr = train_loss[0], train_loss[1]
                            discriminator_err_te, generator_err_te = test_loss[0], test_loss[1]
                            discriminator_tr_record, generator_tr_record = train_loss_record[0], train_loss_record[1]
                            discriminator_te_record, generator_te_record = test_loss_record[0], test_loss_record[1]
                        elif self.config.mode == 'regression':
                            regressor_err_tr, regressor_err_te = train_loss, test_loss
                            regressor_tr_record, regressor_te_record = train_loss_record, test_loss_record
                        elif self.config.mode == 'autoencoder':
                            conditioner_err_tr, conditioner_err_te = train_loss, test_loss
                            conditioner_tr_record, conditioner_te_record = train_loss_record, test_loss_record

                        discriminator_optimizer, discriminator_learning_rate, discriminator_hit_max_lr, \
                        generator_optimizer, generator_learning_rate, generator_hit_max_lr, \
                        conditioner_optimizer, conditioner_learning_rate, conditioner_hit_max_lr, \
                        regressor_optimizer, regressor_learning_rate, regressor_hit_max_lr = \
                            self.update_lr(discriminator_schedulers, discriminator_optimizer, discriminator_err_tr, discriminator_hit_max_lr,
                                           generator_schedulers, generator_optimizer, generator_err_tr, generator_hit_max_lr,
                                           conditioner_schedulers, conditioner_optimizer, conditioner_err_tr, conditioner_hit_max_lr,
                                           regressor_schedulers, regressor_optimizer, regressor_err_tr, regressor_hit_max_lr)

                        metrics_dict = self.update_gan_metrics(epoch, metrics_dict,
                                                               discriminator_learning_rate, generator_learning_rate,
                                                               conditioner_learning_rate, regressor_learning_rate,
                                                               discriminator_err_tr, discriminator_err_te,
                                                               generator_err_tr, generator_err_te,
                                                               conditioner_err_tr, conditioner_err_te,
                                                               regressor_err_tr, regressor_err_te)

                        self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                                          generator_tr_record, generator_te_record, discriminator_tr_record, discriminator_te_record,
                                          conditioner_tr_record, conditioner_te_record, regressor_tr_record, regressor_te_record)

                        if epoch % self.config.wandb.sample_reporting_frequency == 0:
                            self.detailed_reporting(epoch, train_loader, train_epoch_stats_dict, test_epoch_stats_dict,
                                                    extra_test_dict=extra_test_epoch_stats_dict)

                        self.model_checkpointing(epoch, self.config, discriminator, generator, conditioner, regressor,
                                                 discriminator_optimizer, generator_optimizer, conditioner_optimizer, regressor_optimizer,
                                                 generator_err_te, discriminator_err_te, conditioner_err_te, regressor_err_te, metrics_dict)

                        generator_converged, discriminator_converged, conditioner_converged, regressor_converged = \
                            self.check_model_convergence(metrics_dict)

                        if (generator_converged and discriminator_converged and conditioner_converged and regressor_converged) and (epoch > self.config.history + 2):
                            print('Training has converged!')
                            break

                        if epoch % 5 == 0:
                            train_loader, test_loader, extra_test_loader = \
                                self.update_batch_size(train_loader, test_loader, extra_test_loader)

                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):  # if we do hit OOM, slash the batch size
                            train_loader, test_loader = self.slash_batch(train_loader, test_loader)
                        else:
                            raise e
                    epoch += 1

                    if self.config.device.lower() == 'cuda':
                        torch.cuda.empty_cache()  # clear GPU, possibly unnecessary

                # self.post_run_evaluation(epoch, generator, discriminator, discriminator_optimizer, generator_optimizer, metrics_dict, train_loader, test_loader, extra_test_loader) # todo revise

    def run_epoch(self, dataLoader=None, generator=None, discriminator=None, regressor=None, conditioner=None,
                  generator_optimizer=None, discriminator_optimizer=None, conditioner_optimizer=None, regressor_optimizer=None,
                  update_gradients=True, iteration_override=None, record_stats=False, epoch=None):

        if self.config.mode == 'gan':
            return self.gan_epoch(dataLoader, generator, discriminator, generator_optimizer, discriminator_optimizer, update_gradients,
                                  iteration_override, record_stats, epoch)
        elif self.config.mode == 'regression':
            return self.regression_epoch(dataLoader, regressor, regressor_optimizer, update_gradients,
                                         iteration_override, record_stats)
        elif self.config.mode == 'autoencoder':
            return self.conditioner_epoch(dataLoader, conditioner, conditioner_optimizer, update_gradients,
                                          iteration_override, record_stats, epoch)

    def regression_epoch(self, dataLoader, regressor, regressor_optimizer, update_gradients=True,
                         iteration_override=None, record_stats=False):

        t0 = time.time()
        if update_gradients:
            regressor.train(True)
        else:
            regressor.eval()

        loss = []
        loss_record = []
        epoch_stats_dict = {}

        for i, data in enumerate(tqdm.tqdm(dataLoader, miniters=int(len(dataLoader) / 25))):
            '''
            noise injection
            '''
            if self.config.regressor.positional_noise > 0:
                data.pos += torch.randn_like(data.pos) * self.config.regressor.positional_noise

            regression_losses_list, predictions, targets = self.regression_loss(regressor, data)

            stats_keys = ['regressor packing prediction', 'regressor packing target']
            stats_values = [predictions.cpu().detach().numpy(), targets.cpu().detach().numpy()]
            epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values, mode = 'extend')

            regression_loss = regression_losses_list.mean()
            loss.append(regression_loss.data.cpu().detach().numpy())  # average loss
            loss_record.extend(regression_losses_list.cpu().detach().numpy())  # loss distribution

            if update_gradients:
                regressor_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                regression_loss.backward()  # back-propagation
                regressor_optimizer.step()  # update parameters

            if record_stats:
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'tracking features', data.tracking.cpu().detach().numpy(), mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0
        epoch_stats_dict['tracking features'] = np.stack(epoch_stats_dict['tracking features'])

        if record_stats:
            for key in epoch_stats_dict.keys():
                feature = epoch_stats_dict[key]
                if (feature == []) or (feature is None):
                    epoch_stats_dict[key] = None
                else:
                    epoch_stats_dict[key] = np.asarray(feature)

            return np.mean(loss), loss_record, epoch_stats_dict, total_time
        else:
            return np.mean(loss), loss_record, total_time

    def conditioner_epoch(self, dataLoader, conditioner, conditioner_optimizer, update_gradients=True,
                          iteration_override=None, record_stats=False, epoch=None):
        t0 = time.time()
        if update_gradients:
            conditioner.train(True)
        else:
            conditioner.eval()

        loss = []
        loss_record = []
        epoch_stats_dict = {}

        for i, data in enumerate(tqdm.tqdm(dataLoader, miniters=int(len(dataLoader) / 10))):

            '''
            train_generator
            '''
            loss, loss_record, epoch_stats_dict = \
                self.conditioner_step(conditioner, epoch_stats_dict, data,
                                      conditioner_optimizer, i, epoch,
                                      update_gradients, loss, loss_record)

            if record_stats:
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'tracking features', data.tracking.cpu().detach().numpy(), mode='extend')
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'identifiers', data.csd_identifier, mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0

        if record_stats:
            for key in epoch_stats_dict.keys():
                if 'supercell' not in key:
                    feature = epoch_stats_dict[key]
                    if (feature == []) or (feature is None):
                        epoch_stats_dict[key] = None
                    else:
                        epoch_stats_dict[key] = np.asarray(feature)

            return np.mean(loss), loss_record, epoch_stats_dict, total_time
        else:
            return np.mean(loss), loss_record, total_time

    def gan_epoch(self, dataLoader=None, generator=None, discriminator=None, generator_optimizer=None, discriminator_optimizer=None, update_gradients=True,
                  iteration_override=None, record_stats=False, epoch=None):
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

        rand_batch_ind = np.random.randint(0, len(dataLoader))

        for i, data in enumerate(tqdm.tqdm(dataLoader, miniters=int(len(dataLoader) / 10))):

            '''
            train discriminator
            '''
            skip_step = (i % self.config.discriminator_optimizer.training_period) != 0  # only train the discriminator every XX steps, assuming n_steps per epoch is much larger than training period
            discriminator_err, discriminator_loss_record, epoch_stats_dict = \
                self.discriminator_step(discriminator, generator, epoch_stats_dict, data,
                                        discriminator_optimizer, i, update_gradients, discriminator_err, discriminator_loss_record,
                                        skip_step=skip_step, epoch=epoch, last_batch=i == (len(dataLoader) - 1))
            '''
            train_generator
            '''
            generator_err, generator_loss_record, epoch_stats_dict = \
                self.generator_step(discriminator, generator, epoch_stats_dict, data,
                                    generator_optimizer, i, update_gradients, generator_err, generator_loss_record,
                                    rand_batch_ind, last_batch=i == (len(dataLoader) - 1))
            '''
            record some stats
            '''
            if (len(epoch_stats_dict['generated cell parameters']) < i) and record_stats and not self.config.train_generator_conditioner:  # make some samples for analysis if we have none so far from this step
                generated_samples = generator(len(data.y), z=None, conditions=data.to(self.config.device))
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generated cell parameters',
                                                     generated_samples.cpu().detach().numpy(), mode='extend')

            if record_stats:
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'tracking features', data.tracking.cpu().detach().numpy(),mode='extend')
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'identifiers', data.csd_identifier,mode='extend')

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0

        if record_stats:
            for key in epoch_stats_dict.keys():
                if 'supercell' not in key:
                    feature = epoch_stats_dict[key]
                    if (feature == []) or (feature is None):
                        epoch_stats_dict[key] = None
                    else:
                        epoch_stats_dict[key] = np.asarray(feature)

            return [np.mean(discriminator_err), np.mean(generator_err)], [discriminator_loss_record, generator_loss_record], epoch_stats_dict, total_time
        else:
            return [np.mean(discriminator_err), np.mean(generator_err)], [discriminator_loss_record, generator_loss_record], total_time

    def discriminator_evaluation(self, dataLoader=None, discriminator=None, iteration_override=None):
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

        for i, data in enumerate(tqdm.tqdm(dataLoader)):
            '''
            evaluate discriminator
            '''

            real_supercell_data = \
                self.supercell_builder.real_cell_to_supercell(data, self.config)

            if self.config.device.lower() == 'cuda':  # redundant
                real_supercell_data = real_supercell_data.cuda()

            if self.config.test_mode or self.config.anomaly_detection:
                assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"

            score_on_real, real_distances_dict = self.adversarial_score(discriminator, real_supercell_data)

            # full_rdfs, rr, self.elementwise_correlations_labels = crystal_rdf(real_supercell_data, elementwise=True, raw_density=True, rrange=[0, 10], bins=500)
            # intermolecular_rdfs, rr, _ = crystal_rdf(real_supercell_data, intermolecular=True, elementwise=True, raw_density=True, rrange=[0, 10], bins=500)

            epoch_stats_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())
            epoch_stats_dict['identifiers'].extend(data.csd_identifier)  #
            epoch_stats_dict['scores'].extend(score_on_real.cpu().detach().numpy())
            # epoch_stats_dict['intermolecular rdf'].extend(intermolecular_rdfs.cpu().detach().numpy())
            # epoch_stats_dict['full rdf'].extend(full_rdfs.cpu().detach().numpy())
            epoch_stats_dict['vdw penalty'].extend(vdw_overlap(real_supercell_data, self.vdw_radii).cpu().detach().numpy())

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

    def adversarial_score(self, discriminator, data):
        output, extra_outputs = discriminator(data, return_dists=True)  # reshape output from flat filters to channels * filters per channel

        return output, extra_outputs['dists dict']

    def log_gan_loss(self, metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                     discriminator_tr_record, discriminator_te_record, generator_tr_record, generator_te_record,
                     conditioner_tr_record, conditioner_te_record, regressor_tr_record, regressor_te_record):

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
            current_metrics[key] = np.amax(current_metrics[key])  # just a formatting thing - nothing to do with the max of anything
        wandb.log(current_metrics)

        # log discriminator losses
        if discriminator_tr_record is not None:
            hist = np.histogram(discriminator_tr_record, bins=256, range=(np.amin(discriminator_tr_record), np.quantile(discriminator_tr_record, 0.9)))
            wandb.log({"Discriminator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        if discriminator_te_record is not None:
            hist = np.histogram(discriminator_te_record, bins=256, range=(np.amin(discriminator_te_record), np.quantile(discriminator_te_record, 0.9)))
            wandb.log({"Discriminator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        # log generator losses
        if generator_tr_record is not None:
            hist = np.histogram(generator_tr_record, bins=256, range=(np.amin(generator_tr_record), np.quantile(generator_tr_record, 0.9)))
            wandb.log({"Generator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        if generator_te_record is not None:
            hist = np.histogram(generator_te_record, bins=256, range=(np.amin(generator_te_record), np.quantile(generator_te_record, 0.9)))
            wandb.log({"Generator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        # log conditioner losses
        if conditioner_tr_record is not None:
            hist = np.histogram(conditioner_tr_record, bins=256, range=(np.amin(conditioner_tr_record), np.quantile(conditioner_tr_record, 0.9)))
            wandb.log({"Generator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        if conditioner_te_record is not None:
            hist = np.histogram(conditioner_te_record, bins=256, range=(np.amin(conditioner_te_record), np.quantile(conditioner_te_record, 0.9)))
            wandb.log({"Generator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        # log regressor losses
        if regressor_tr_record is not None:
            hist = np.histogram(regressor_tr_record, bins=256, range=(np.amin(regressor_tr_record), np.quantile(regressor_tr_record, 0.9)))
            wandb.log({"Generator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        if regressor_te_record is not None:
            hist = np.histogram(regressor_te_record, bins=256, range=(np.amin(regressor_te_record), np.quantile(regressor_te_record, 0.9)))
            wandb.log({"Generator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        # log special losses
        special_losses = {}
        special_losses['epoch'] = current_metrics['epoch']
        if train_epoch_stats_dict is not None:
            for key in train_epoch_stats_dict.keys():
                if ('loss' in key) and (train_epoch_stats_dict[key] is not None):
                    special_losses['Train ' + key] = np.average(np.concatenate(train_epoch_stats_dict[key]))
                if ('score' in key) and (train_epoch_stats_dict[key] is not None):
                    score = softmax_and_score(train_epoch_stats_dict[key])
                    special_losses['Train ' + key] = np.average(score)

        if test_epoch_stats_dict is not None:
            for key in test_epoch_stats_dict.keys():
                if ('loss' in key) and (test_epoch_stats_dict[key] is not None):
                    special_losses['Test ' + key] = np.average(np.concatenate(test_epoch_stats_dict[key]))
                if ('score' in key) and (test_epoch_stats_dict[key] is not None):
                    score = softmax_and_score(test_epoch_stats_dict[key])
                    special_losses['Test ' + key] = np.average(score)

        wandb.log(special_losses)

    def detailed_reporting(self, epoch, train_loader, train_epoch_stats_dict, test_epoch_stats_dict, extra_test_dict=None):
        '''
        Do analysis and upload results to w&b
        '''
        if (test_epoch_stats_dict is not None) and self.config.mode == 'gan':
            if test_epoch_stats_dict['generated cell parameters'] is not None:
                self.cell_params_analysis(train_loader, test_epoch_stats_dict)

            if self.config.train_generator_packing or self.config.train_generator_vdw or self.config.train_generator_adversarially or self.config.train_generator_combo:
                self.cell_generation_analysis(test_epoch_stats_dict)

            if self.config.train_discriminator_on_distorted or self.config.train_discriminator_on_randn or self.config.train_discriminator_adversarially:
                self.discriminator_analysis(test_epoch_stats_dict)

        if (test_epoch_stats_dict is not None) and self.config.mode == 'autoencoder':
            self.report_conditioner_training(test_epoch_stats_dict)

        elif self.config.mode == 'regression':
            self.log_regression_accuracy(train_epoch_stats_dict, test_epoch_stats_dict)

        if (extra_test_dict is not None) and (epoch % self.config.extra_test_period == 0):
            pass  # do reporting on an extra dataset
            # old reporting on nov_2022 discriminator development
            # from reporting.nov_22_discriminator import blind_test_analysis
            # blind_test_analysis(self.config, wandb, train_epoch_stats_dict, test_epoch_stats_dict, extra_test_dict)

        return None

    def discriminator_step(self, discriminator, generator, epoch_stats_dict, data, discriminator_optimizer, i, update_gradients, discriminator_err, discriminator_loss_record, skip_step, epoch, last_batch):
        if self.config.train_discriminator_adversarially or self.config.train_discriminator_on_distorted or self.config.train_discriminator_on_randn:
            generated_samples_i, handedness, epoch_stats_dict = \
                self.generate_discriminator_negatives(epoch_stats_dict, self.config, data, generator, i)


            # todo separate generated samples data objects from original real crystals
            score_on_real, score_on_fake, generated_samples, \
            real_dist_dict, fake_dist_dict, real_vdw_score, fake_vdw_score \
                = self.train_discriminator(generated_samples_i, discriminator, data, i, handedness)

            prediction = torch.cat((score_on_real, score_on_fake))
            target = torch.cat((torch.ones_like(score_on_real[:, 0]), torch.zeros_like(score_on_fake[:, 0])))
            discriminator_losses = F.cross_entropy(prediction, target.long(), reduction='none')  # works much better

            discriminator_loss = discriminator_losses.mean()
            discriminator_err.append(discriminator_loss.data.cpu().detach().numpy())  # average overall loss

            if update_gradients and (not skip_step):
                discriminator_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.config.gradient_norm_clip)  # gradient clipping
                discriminator_loss.backward()  # back-propagation
                discriminator_optimizer.step()  # update parameters

            stats_keys = ['discriminator real score', 'discriminator fake score',
                          'real vdw penalty', 'fake vdw penalty',
                          'generated cell parameters', 'final generated cell parameters']
            stats_values = [score_on_real.cpu().detach().numpy(), score_on_fake.cpu().detach().numpy(),
                            real_vdw_score.cpu().detach().numpy(), fake_vdw_score.cpu().detach().numpy(),
                            generated_samples_i.cpu().detach().numpy(), generated_samples]

            epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values, mode='extend')
            discriminator_loss_record.extend(discriminator_losses.cpu().detach().numpy())  # overall loss distribution

        else:
            discriminator_err.append(np.zeros(1))
            discriminator_loss_record.extend(np.zeros(data.num_graphs))

        return discriminator_err, discriminator_loss_record, epoch_stats_dict

    def generator_step(self, discriminator, generator, epoch_stats_dict, data, generator_optimizer, i, update_gradients, generator_err, generator_loss_record, rand_batch_ind, last_batch):
        if any((self.config.train_generator_packing, self.config.train_generator_adversarially, self.config.train_generator_vdw, self.config.train_generator_combo, self.config.train_generator_h_bond)):
            adversarial_score, generated_samples, packing_loss, packing_prediction, packing_target, \
            vdw_loss, generated_dist_dict, supercell_examples, similarity_penalty, h_bond_score, combo_score = \
                self.train_generator(generator, discriminator, data, i)

            generator_losses, epoch_stats_dict = self.aggregate_generator_losses(
                epoch_stats_dict, packing_loss, adversarial_score,
                vdw_loss, similarity_penalty, packing_prediction, packing_target, h_bond_score, combo_score)

            generator_loss = generator_losses.mean()
            generator_err.append(generator_loss.data.cpu().detach().numpy())  # average loss

            if update_gradients:
                generator_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                torch.nn.utils.clip_grad_norm_(generator.parameters(), self.config.gradient_norm_clip)  # gradient clipping
                generator_loss.backward()  # back-propagation
                generator_optimizer.step()  # update parameters

            epoch_stats_dict = self.log_supercell_examples(supercell_examples, i, rand_batch_ind, epoch_stats_dict)
            generator_loss_record.extend(generator_losses.cpu().detach().numpy())  # loss distribution
            epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generated cell parameters', generated_samples, mode='extend')

        else:
            generator_err.append(np.zeros(1))
            generator_loss_record.extend(np.zeros(data.num_graphs))

        return generator_err, generator_loss_record, epoch_stats_dict

    def conditioner_step(self, conditioner, epoch_stats_dict, data, conditioner_optimizer, i, epoch, update_gradients, loss, loss_record):
        # if (epoch < 50) and (i%2 == 0):
        #     rand_sample = True
        # else:
        #     rand_sample = False
        rand_sample = False # only real data

        packing_loss, reconstruction_loss, target_sample, prediction_sample, \
        packing_true, packing_pred, particle_dist_true, particle_dist_pred,\
            real_sample= \
            self.train_conditioner(conditioner, data, rand_sample)

        stats_keys = ['conditioner packing target', 'conditioner packing prediction',
                      'conditioner particle prediction', 'conditioner particle true',
                      'reconstruction loss', 'conditioner packing loss']
        stats_values = [packing_true, packing_pred,
                        particle_dist_pred, particle_dist_true,
                        reconstruction_loss.cpu().detach().numpy(), packing_loss.cpu().detach().numpy()]

        if i == 0:
            stats_keys += ['target_sample', 'prediction_sample']
            stats_values += [target_sample, prediction_sample]

        epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values)

        if not real_sample:
            conditioning_losses = reconstruction_loss  #+ packing_loss
        else:
            conditioning_losses = reconstruction_loss + packing_loss

        conditioner_loss = (conditioning_losses).mean()
        loss.append(conditioner_loss.data.cpu().detach().numpy())  # average loss

        if update_gradients:
            conditioner_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
            torch.nn.utils.clip_grad_norm_(conditioner.parameters(), self.config.gradient_norm_clip)  # gradient clipping
            conditioner_loss.backward()  # back-propagation
            conditioner_optimizer.step()  # update parameters

        loss_record.extend(conditioning_losses.cpu().detach().numpy())  # loss distribution

        return loss, loss_record, epoch_stats_dict

    def train_discriminator(self, generated_samples, discriminator, data, i, target_handedness=None, return_rdf=False):
        # generate fakes & create supercell data
        real_supercell_data = self.supercell_builder.real_cell_to_supercell(data, self.config)
        fake_supercell_data, generated_cell_volumes, overlaps_list = \
            self.supercell_builder.build_supercells(data, generated_samples,
                                                    self.config.supercell_size, self.config.discriminator.graph_convolution_cutoff,
                                                    override_sg=self.config.generate_sgs, target_handedness=target_handedness)

        if self.config.device.lower() == 'cuda':  # redundant
            real_supercell_data = real_supercell_data.cuda()
            fake_supercell_data = fake_supercell_data.cuda()

        if self.config.test_mode or self.config.anomaly_detection:
            assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"
            assert torch.sum(torch.isnan(fake_supercell_data.x)) == 0, "NaN in training input"

        if self.config.discriminator.positional_noise > 0:
            real_supercell_data.pos += torch.randn_like(real_supercell_data.pos) * self.config.discriminator.positional_noise
            fake_supercell_data.pos += torch.randn_like(fake_supercell_data.pos) * self.config.discriminator.positional_noise

        score_on_real, real_distances_dict = self.adversarial_score(discriminator, real_supercell_data)
        score_on_fake, fake_pairwise_distances_dict = self.adversarial_score(discriminator, fake_supercell_data)

        if return_rdf:
            real_rdf, rr, rdf_label_dict = crystal_rdf(real_supercell_data, rrange=[0, 6], bins=1000, intermolecular=True, elementwise=True, raw_density=True)
            fake_rdf, rr, rdf_label_dict = crystal_rdf(fake_supercell_data, rrange=[0, 6], bins=1000, intermolecular=True, elementwise=True, raw_density=True)

            real_rdf_dict = {'rdf': real_rdf, 'range': rr, 'labels': rdf_label_dict}
            fake_rdf_dict = {'rdf': fake_rdf, 'range': rr, 'labels': rdf_label_dict}

            return score_on_real, score_on_fake, fake_supercell_data.cell_params.cpu().detach().numpy(), real_rdf_dict, fake_rdf_dict

        else:
            return score_on_real, score_on_fake, fake_supercell_data.cell_params.cpu().detach().numpy(), \
                   real_distances_dict, fake_pairwise_distances_dict, \
                   vdw_overlap(self.vdw_radii, crystaldata=real_supercell_data), \
                   vdw_overlap(self.vdw_radii, crystaldata=fake_supercell_data)

    def set_molecule_alignment(self, data, right_handed=True, mode_override=None):
        if mode_override is not None:
            mode = mode_override
        else:
            mode = self.config.generator.canonical_conformer_orientation
        if mode == 'standardized':
            data = align_crystaldata_to_principal_axes(data)
        elif mode == 'random':
            data = random_crystaldata_alignment(data)
            if right_handed:
                coords_list = [data.pos[data.ptr[i]:data.ptr[i + 1]] for i in range(data.num_graphs)]
                coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(data.num_graphs)]
                principal_axes_list, _, _ = batch_molecule_principal_axes(coords_list_centred)
                handedness = compute_Ip_handedness(principal_axes_list)
                for ind, hand in enumerate(handedness):
                    if hand == -1:
                        data.pos[data.batch == ind] = -data.pos[data.batch == ind]  # invert

        return data

    def get_generator_samples(self, data, generator, return_condition=False):
        '''
        conformer orentation setting
        '''
        data = self.set_molecule_alignment(data)

        '''
        noise injection
        '''
        if self.config.generator.positional_noise > 0:
            data.pos += torch.randn_like(data.pos) * self.config.generator.positional_noise

        '''
        update symmetry information
        '''
        if self.config.generate_sgs is not None:
            override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index(self.config.generate_sgs) + 1  # indexing from 0
            sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(data.x.device) for i in range(data.num_graphs)]
            data = override_sg_info(self.config.generate_sgs, self.config.dataDims, data, self.supercell_builder.symmetries_dict, sym_ops_list)  # todo update the way we handle this

        '''
        generate samples
        '''
        [[generated_samples, latent], prior, condition] = generator.forward(
            n_samples=data.num_graphs, conditions=data.to(self.config.device).clone(),
            return_latent=True, return_condition=True, return_prior=True)

        if return_condition:
            return generated_samples, prior, condition
        else:
            return generated_samples, prior

    def train_generator(self, generator, discriminator, data, i):
        '''
        train the generator
        '''

        if self.config.train_generator_packing or self.config.train_generator_vdw or self.config.train_generator_combo or self.config.train_generator_adversarially or self.config.train_generator_h_bond:
            '''
            build supercells
            '''
            generated_samples, prior = self.get_generator_samples(data, generator, return_condition=False)

            supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
                data, generated_samples, self.config.supercell_size,
                self.config.discriminator.graph_convolution_cutoff,
                override_sg=self.config.generate_sgs,
                align_molecules=False,  # molecules are either random on purpose, or pre-aligned with set handedness
            )

            data.cell_params = supercell_data.cell_params

            '''
            #evaluate losses
            
            # look at cells
            from ase.visualize import view
            mols = [ase_mol_from_crystaldata(supercell_data, i, exclusion_level='convolve with', highlight_aux=True) for i in range(min(10, supercell_data.num_graphs))]
            view(mols)
            '''
            similarity_penalty = self.compute_similarity_penalty(generated_samples, prior)
            discriminator_score, dist_dict = self.score_adversarially(supercell_data.clone(), discriminator)
            h_bond_score = self.compute_h_bond_score(supercell_data)
            vdw_penalty, normed_vdw_penalty = self.get_vdw_penalty(dist_dict, data.num_graphs, data)
            packing_loss, packing_prediction, packing_target, = \
                self.cell_density_loss(data, generated_samples, precomputed_volumes=generated_cell_volumes)
            combo_score = torch.log(10 / (10 + ((vdw_penalty) ** 2 + packing_loss ** 2)))  # torch.exp(-(normed_vdw_overlap + outside_reasonable_packing_loss))

            return discriminator_score, generated_samples.cpu().detach().numpy(), \
                   packing_loss, packing_prediction.cpu().detach().numpy(), \
                   packing_target.cpu().detach().numpy(), \
                   vdw_penalty, dist_dict, \
                   supercell_data, similarity_penalty, h_bond_score, \
                   combo_score

    def train_conditioner(self, conditioner, data, rand_sample):
        # minimum resolution of 0.5 Angstrom, to start
        # limit target to set of useful classes
        sample_real = True
        if conditioner.training and rand_sample:  # replace training data with a random point cloud, and do test on real molecules
            sample_real = False
            # batch_size = data.num_graphs
            # avg_num_particles_per_sample = 15
            # cartesian_dimension = 3
            # n_particle_types = len(self.config.conditioner_classes)
            # batch = torch.randint(low=0, high=batch_size, size=(len(data.x), 1))[:, 0]  # batch index
            # batch = batch[torch.argsort(batch)]
            particle_coords = torch.rand(len(data.x), 3, device=data.pos.device, dtype=data.pos.dtype) * 2 - 1  # particle positions with mean of avg_num_particles_per_sample particles per sample
            particle_coords *= self.config.max_molecule_radius
            # particle_types = torch.randint(low=1, high=n_particle_types + 1, size=(len(data.x), data.x.shape[1])) # only first column is used anyway, later columns kept for indexing purposes

            data.pos = particle_coords
            # data.batch = batch
            # data.x = particle_types
            # data.ptr = torch.cat((torch.zeros(1),torch.argwhere(torch.diff(data.batch))[:,0] + 1,torch.ones(1) * len(data.x))).long()

        data = data.cuda()
        data = self.set_molecule_alignment(data, mode_override='random')
        if len(self.config.conditioner_classes) == 1:
            data.x[:, 0] = 1
        '''
        noise injection
        '''
        if self.config.conditioner.positional_noise > 0:
            data.pos += torch.randn_like(data.pos) * self.config.conditioner.positional_noise

        '''
        update symmetry information
        '''
        if self.config.generate_sgs is not None:
            override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index(self.config.generate_sgs) + 1  # indexing from 0
            sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(data.x.device) for i in range(data.num_graphs)]
            data = override_sg_info(self.config.generate_sgs, self.config.dataDims, data, self.supercell_builder.symmetries_dict, sym_ops_list)  # todo update the way we handle this

        point_cloud_prediction, packing_prediction = conditioner(data.clone())

        n_target_bins = int((self.config.max_molecule_radius) * 2 / self.config.conditioner.decoder_resolution) + 1  # make up for odd in stride
        _, n_target_bins = get_strides(n_target_bins, init_size=self.config.conditioner.init_decoder_size)  # automatically find the right number of strides within 4-5 steps (minimizes overall stack depth)
        batch_size = len(point_cloud_prediction)
        buckets = torch.bucketize(data.pos, torch.linspace(-self.config.max_molecule_radius, self.config.max_molecule_radius, n_target_bins - 1, device='cuda'))
        target = torch.zeros((batch_size, n_target_bins, n_target_bins, n_target_bins), dtype=torch.long, device=point_cloud_prediction.device)
        for ii in range(batch_size):
            target[ii, buckets[data.batch == ii, 0], buckets[data.batch == ii, 1], buckets[data.batch == ii, 2]] = data.x[data.batch == ii, 0].long()  # torch.clip(data.x[data.batch == ii,0],max=1).long()

        for ind, (key, value) in enumerate(self.config.conditioner_classes.items()):
            target[target == key] = -value  # negative, so this next step works quickly
        target[target > 0] = 1  # setting for 'other'
        target = -target
        target[target == -1] = 1

        packing_loss = F.smooth_l1_loss(packing_prediction[:, 0], data.y.float(), reduction='none')
        reconstruction_loss = F.cross_entropy(point_cloud_prediction, target, reduction='none').mean([-3, -2, -1]) / (torch.sum(target > 0) / len(target.flatten()))

        # true_dist = F.one_hot(target).permute(0, 4, 1, 2, 3).flatten(2, 4)
        # pred_dist = F.softmax(point_cloud_prediction.flatten(2, 4), dim=1)

        return packing_loss, reconstruction_loss, \
               target[0:8].cpu().detach().numpy(), point_cloud_prediction[0:8].cpu().detach().numpy(), \
               data.y.cpu().detach().numpy(), packing_prediction.cpu().detach().numpy(), \
               torch.mean(F.one_hot(target, num_classes = self.config.max_atomic_number + 1).permute(0, 4, 1, 2, 3).flatten(2, 4).float(), dim=(0, 2)).cpu().detach().numpy(), \
               torch.mean(F.softmax(point_cloud_prediction.flatten(2, 4), dim=1).float(), dim=(0, 2)).cpu().detach().numpy(),\
               sample_real

    def regression_loss(self, generator, data):
        predictions = generator(data.to(generator.model.device))[:, 0]
        targets = data.y
        return F.smooth_l1_loss(predictions, targets, reduction='none'), predictions, targets

    def cell_density_loss(self, data, raw_sample, precomputed_volumes=None):
        '''
        compute packing coefficients for generated cells
        '''
        if precomputed_volumes is None:
            volumes_list = []
            for i in range(len(raw_sample)):
                volumes_list.append(cell_vol_torch(data.cell_params[i, 0:3], data.cell_params[i, 3:6]))
            volumes = torch.stack(volumes_list)
        else:
            volumes = precomputed_volumes

        generated_packing_coeffs = data.Z * data.tracking[:, self.mol_volume_ind] / volumes
        standardized_gen_packing_coeffs = (generated_packing_coeffs - self.config.dataDims['target mean']) / self.config.dataDims['target std']

        csd_packing_coeffs = data.tracking[:, self.config.dataDims['tracking features dict'].index('crystal packing coefficient')]
        standardized_csd_packing_coeffs = (csd_packing_coeffs - self.config.dataDims['target mean']) / self.config.dataDims['target std']  # requires that packing coefficnet is set as regression target in main

        if self.config.packing_loss_rescaling == 'log':
            packing_loss = torch.log(1 + F.smooth_l1_loss(standardized_gen_packing_coeffs, standardized_csd_packing_coeffs, reduction='none'))  # log(1+loss) is a soft rescaling to avoid gigantic losses
        elif self.config.packing_loss_rescaling is None:
            packing_loss = F.smooth_l1_loss(standardized_gen_packing_coeffs, standardized_csd_packing_coeffs, reduction='none')
        elif self.config.packing_loss_rescaling == 'mse':
            packing_loss = F.mse_loss(standardized_gen_packing_coeffs, standardized_csd_packing_coeffs, reduction='none')

        if self.config.test_mode:
            assert torch.sum(torch.isnan(packing_loss)) == 0

        return packing_loss, generated_packing_coeffs, csd_packing_coeffs

    def training_prep(self):
        dataset_builder = BuildDataset(self.config, pg_dict=self.point_groups,
                                       sg_dict=self.space_groups,
                                       lattice_dict=self.lattice_type,
                                       premade_dataset=self.prep_dataset)

        del self.prep_dataset  # we don't actually want this huge thing floating around
        self.config.dataDims = dataset_builder.get_dimension()

        '''
        init supercell builder
        '''
        self.supercell_builder = SupercellBuilder(self.sym_ops, self.sym_info, self.normed_lattice_vectors, self.atom_weights, self.config.dataDims)

        '''
        set tracking feature indices & property dicts we will use later
        '''
        self.mol_volume_ind = self.config.dataDims['tracking features dict'].index('molecule volume')
        self.crystal_packing_ind = self.config.dataDims['tracking features dict'].index('crystal packing coefficient')
        self.crystal_density_ind = self.config.dataDims['tracking features dict'].index('crystal calculated density')
        self.mol_size_ind = self.config.dataDims['tracking features dict'].index('molecule num atoms')
        self.sg_feature_ind_dict = {thing[14:]: ind + self.config.dataDims['num atomwise features'] for ind, thing in enumerate(self.config.dataDims['molecule features']) if 'sg is' in thing}
        self.crysys_ind_dict = {thing[18:]: ind + self.config.dataDims['num atomwise features'] for ind, thing in enumerate(self.config.dataDims['molecule features']) if 'crystal system is' in thing}

        '''
        add symmetry element indices to symmetry dict
        '''
        self.sym_info['sg_feature_ind_dict'] = self.sg_feature_ind_dict
        self.sym_info['crysys_ind_dict'] = self.crysys_ind_dict

        ''' 
        init gaussian generator for cell parameter sampling
        we don't always use it but it's very cheap so just do it every time
        '''
        self.randn_generator = independent_gaussian_model(input_dim=self.config.dataDims['num lattice features'],
                                                          means=self.config.dataDims['lattice means'],
                                                          stds=self.config.dataDims['lattice stds'],
                                                          normed_length_means=self.config.dataDims['lattice normed length means'],
                                                          normed_length_stds=self.config.dataDims['lattice normed length stds'],
                                                          cov_mat=self.config.dataDims['lattice cov mat'])

        self.epoch = 0

        return dataset_builder

    def sampling_prep(self):
        dataset_builder = self.training_prep()
        del dataset_builder
        generator, discriminator, generator_optimizer, generator_schedulers, \
        discriminator_optimizer, discriminator_schedulers, params1, params2 \
            = self.init_models(self.config, self.config.dataDims)

        self.config.current_batch_size = self.config.min_batch_size

        extra_test_set_path = self.config.extra_test_set_paths
        extra_test_loader = get_extra_test_loader(self.config, extra_test_set_path, dataDims=self.config.dataDims,
                                                  pg_dict=self.point_groups, sg_dict=self.space_groups, lattice_dict=self.lattice_type)

        self.randn_generator = independent_gaussian_model(input_dim=self.config.dataDims['num lattice features'],
                                                          means=self.config.dataDims['lattice means'],
                                                          stds=self.config.dataDims['lattice stds'],
                                                          normed_length_means=self.config.dataDims['lattice normed length means'],
                                                          normed_length_stds=self.config.dataDims['lattice normed length stds'],
                                                          cov_mat=self.config.dataDims['lattice cov mat'])

        # blind_test_identifiers = [
        #     'OBEQUJ', 'OBEQOD','NACJAF'] # targets XVI, XVII, XXII
        #
        # single_mol_data = extra_test_loader.dataset[extra_test_loader.csd_identifier.index(blind_test_identifiers[-1])]

        return extra_test_loader, generator, discriminator, \
               generator_optimizer, generator_schedulers, discriminator_optimizer, discriminator_schedulers, \
               params1, params2

    def model_sampling(self):
        '''
        Stun MC annealing on a pretrained discriminator / generator
        '''
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=[self.config.wandb.experiment_tag]):
            extra_test_loader, generator, discriminator, \
            generator_optimizer, generator_schedulers, discriminator_optimizer, discriminator_schedulers, \
            params1, params2 = self.sampling_prep()  # todo rebuild this with new model_init

            generator.eval()
            discriminator.eval()

            smc_sampler = mcmcSampler(
                gammas=np.logspace(-4, 0, self.config.current_batch_size),
                seedInd=0,
                STUN_mode=False,
                debug=False,
                init_adaptive_step_size=self.config.sample_move_size,
                global_temperature=0.00001,  # essentially only allow downward moves
                generator=generator,
                supercell_size=self.config.supercell_size,
                graph_convolution_cutoff=self.config.discriminator.graph_convolution_cutoff,
                vdw_radii=self.vdw_radii,
                preset_minimum=None,
                spacegroup_to_search='P-1',  # self.config.generate_sgs,
                new_minimum_patience=25,
                reset_patience=50,
                conformer_orientation=self.config.generator.canonical_conformer_orientation,
            )

            '''
            run sampling
            '''
            # prep the conformers
            single_mol_data_0 = extra_test_loader.dataset[0]
            collater = Collater(None, None)
            single_mol_data = collater([single_mol_data_0 for n in range(self.config.current_batch_size)])
            single_mol_data = self.set_molecule_alignment(single_mol_data, mode_override='random')  # take all the same conformers for one run
            override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index('P-1') + 1
            sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(single_mol_data.x.device) for i in range(single_mol_data.num_graphs)]
            single_mol_data = override_sg_info('P-1', self.supercell_builder.dataDims, single_mol_data, self.supercell_builder.symmetries_dict, sym_ops_list)

            smc_sampling_dict = smc_sampler(discriminator, self.supercell_builder,
                                            single_mol_data.clone().cuda(), None,
                                            self.config.sample_steps)

            '''
            reporting
            '''

            self.sampling_telemetry_plot(smc_sampling_dict)
            self.cell_params_tracking_plot(smc_sampling_dict, collater, extra_test_loader)
            best_smc_samples, best_smc_samples_scores, best_smc_cells = self.sample_clustering(smc_sampling_dict, collater, extra_test_loader, discriminator)

            # destandardize samples
            unclean_best_samples = self.de_clean_samples(best_smc_samples, best_smc_cells.sg_ind)
            single_mol_data = collater([single_mol_data_0 for n in range(len(best_smc_samples))])
            gd_sampling_dict = gradient_descent_sampling(
                discriminator, unclean_best_samples, single_mol_data.clone(), self.supercell_builder,
                n_iter=500, lr=1e-3,
                optimizer_func=optim.Rprop,
                return_vdw=True, vdw_radii=self.vdw_radii,
                supercell_size=self.config.supercell_size,
                cutoff=self.config.discriminator.graph_convolution_cutoff,
                generate_sgs='P-1',  # self.config.generate_sgs
                align_molecules=True,  # always true here
            )
            gd_sampling_dict['canonical samples'] = gd_sampling_dict['samples']
            gd_sampling_dict['resampled state record'] = [[0] for _ in range(len(unclean_best_samples))]
            gd_sampling_dict['scores'] = gd_sampling_dict['scores'].T
            gd_sampling_dict['vdw penalties'] = gd_sampling_dict['vdw'].T
            gd_sampling_dict['canonical samples'] = np.swapaxes(gd_sampling_dict['canonical samples'], 0, 2)
            self.sampling_telemetry_plot(gd_sampling_dict)
            self.cell_params_tracking_plot(gd_sampling_dict, collater, extra_test_loader)
            best_gd_samples, best_gd_samples_scores, best_gd_cells = self.sample_clustering(gd_sampling_dict, collater, extra_test_loader, discriminator)

            # todo process refined samples

            # todo compare final samples to known minima

            extra_test_sample = next(iter(extra_test_loader)).cuda()
            sample_supercells = self.supercell_builder.real_cell_to_supercell(extra_test_sample, self.config)
            known_sample_scores = softmax_and_score(discriminator(sample_supercells.clone()))

            aa = 1
            plt.clf()
            plt.plot(gd_sampling_dict['scores'])

            # np.save(f'../sampling_output_run_{self.config.run_num}', sampling_dict)
            # self.report_sampling(sampling_dict)

    def de_clean_samples(self, samples, sg_inds):
        means = self.supercell_builder.dataDims['lattice means']
        stds = self.supercell_builder.dataDims['lattice stds']

        # soft clipping to ensure correct range with finite gradients
        cell_lengths = torch.Tensor(samples[:, :3] * stds[0:3] + means[0:3])
        cell_angles = torch.Tensor(samples[:, 3:6] * stds[3:6] + means[3:6])
        mol_position = torch.Tensor(samples[:, 6:9] * stds[6:9] + means[6:9])
        mol_rotation = torch.Tensor(samples[:, 9:12] * stds[9:12] + means[9:12])

        # descale asymmetric unit
        descaled_mol_position = mol_position.clone()
        for i, ind in enumerate(sg_inds):
            descaled_mol_position[i, :] = mol_position[i, :] / self.supercell_builder.asym_unit_dict[str(int(ind))].cpu()

        # undo cleaning
        unclean_cell_lengths = np.log(np.exp(cell_lengths) - np.exp(1) ** (1 / 10))
        unclean_cell_angles = undo_1d_bound(cell_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2, mode='soft')
        unclean_mol_position = undo_1d_bound(descaled_mol_position, 0.5, 0.5, mode='soft')
        norms = torch.linalg.norm(mol_rotation, dim=1)
        unclean_norms = undo_1d_bound(norms, torch.pi, torch.pi, mode='soft')
        unclean_mol_rotation = mol_rotation / norms[:, None] * unclean_norms[:, None]

        # restandardize samples
        unclean_cell_lengths = (unclean_cell_lengths.detach().numpy() - means[0:3]) / stds[0:3]
        unclean_cell_angles = (unclean_cell_angles.detach().numpy() - means[3:6]) / stds[3:6]
        unclean_mol_position = (unclean_mol_position.detach().numpy() - means[6:9]) / stds[6:9]
        unclean_mol_rotation = (unclean_mol_rotation.detach().numpy() - means[9:12]) / stds[9:12]

        unclean_best_samples = np.concatenate((unclean_cell_lengths, unclean_cell_angles, unclean_mol_position, unclean_mol_rotation), axis=1)
        return unclean_best_samples

    def sample_clustering(self, sampling_dict, collater, extra_test_loader, discriminator):

        # first level filter - remove subsequent duplicates
        n_runs = sampling_dict['canonical samples'].shape[1]
        n_steps = sampling_dict['canonical samples'].shape[2]
        filtered_samples = [[sampling_dict['canonical samples'][:, ii, 0]] for ii in range(n_runs)]
        filtered_samples_inds = [[0] for ii in range(n_runs)]
        for i in range(1, n_steps):
            for j in range(n_runs):
                if not all(sampling_dict['canonical samples'][:, j, i] == sampling_dict['canonical samples'][:, j, i - 1]):
                    filtered_samples[j].append(sampling_dict['canonical samples'][:, j, i])
                    filtered_samples_inds[j].append(i)
        filtered_samples = [torch.tensor(filtered_samples[ii], requires_grad=False, dtype=torch.float32) for ii in range(n_runs)]
        filtered_samples_inds = [np.asarray(filtered_samples_inds[ii]) for ii in range(n_runs)]
        filtered_samples_scores = [np.asarray(sampling_dict['scores'][ii, filtered_samples_inds[ii]]) for ii in range(n_runs)]

        all_filtered_samples = np.concatenate(filtered_samples)
        all_filtered_samples_scores = np.concatenate(filtered_samples_scores)
        dists = torch.cdist(torch.Tensor(all_filtered_samples), torch.Tensor(all_filtered_samples)).detach().numpy()

        model = AgglomerativeClustering(distance_threshold=1, linkage="average", affinity='euclidean', n_clusters=None)
        model = model.fit(all_filtered_samples)
        n_clusters = model.n_clusters_
        classes = model.labels_

        '''
        visualize classwise distances
        '''
        class_distances = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if j >= i:
                    class_distances[i, j] = np.mean(dists[classes == i][:, classes == j])

        # #plot the top three levels of the dendrogram
        # plt.clf()
        # plt.subplot(1,2,1)
        # plot_dendrogram(model, truncate_mode="level", p=3)
        # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        # plt.show()
        # plt.subplot(1,2,2)
        # plt.imshow(class_distances)

        '''
        pick out best samples in each class with reasonably good scoress
        '''
        best_samples = np.zeros((n_clusters, 12))
        best_samples_scores = np.zeros((n_clusters))
        for i in range(n_clusters):
            best_ind = np.argmax(all_filtered_samples_scores[classes == i])
            best_samples_scores[i] = all_filtered_samples_scores[classes == i][best_ind]
            best_samples[i] = all_filtered_samples[classes == i][best_ind]

        sort_inds = np.argsort(best_samples_scores)
        best_samples = best_samples[sort_inds]
        best_samples_scores = best_samples_scores[sort_inds]

        n_samples_to_build = min(100, len(best_samples))
        best_samples_to_build = best_samples[:n_samples_to_build]
        single_mol_data_0 = extra_test_loader.dataset[0]
        big_single_mol_data = collater([single_mol_data_0 for n in range(n_samples_to_build)]).cuda()
        override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index('P-1') + 1
        sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(big_single_mol_data.x.device) for i in range(big_single_mol_data.num_graphs)]
        big_single_mol_data = override_sg_info('P-1', self.supercell_builder.dataDims, big_single_mol_data, self.supercell_builder.symmetries_dict, sym_ops_list)

        best_cells, _, _ = self.supercell_builder.build_supercells(big_single_mol_data, torch.tensor(best_samples_to_build, device='cuda', dtype=torch.float32),
                                                                   supercell_size=self.config.supercell_size,
                                                                   graph_convolution_cutoff=self.config.discriminator.graph_convolution_cutoff,
                                                                   override_sg='P-1',
                                                                   align_molecules=True,
                                                                   skip_cell_cleaning=True,
                                                                   rescale_asymmetric_unit=False,
                                                                   standardized_sample=True, )

        assert np.mean(np.abs(best_cells.cell_params.cpu().detach().numpy() - (best_samples_to_build * self.supercell_builder.dataDims['lattice stds'] + self.supercell_builder.dataDims['lattice means']))) < 1e-4
        ss = softmax_and_score(discriminator(best_cells.clone().cuda())).cpu().detach().numpy()

        # mols = [ase_mol_from_crystaldata(best_cells, ii, highlight_aux=True, exclusion_level='distance', inclusion_distance=5) for ii in range(best_cells.num_graphs)]
        # view(mols)

        return best_samples, best_samples_scores, best_cells.cpu().detach()

    def sampling_telemetry_plot(self, sampling_dict):
        n_runs = len(sampling_dict['scores'])
        num_iters = sampling_dict['scores'].shape[1]

        self.layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=40,  # top margin
            )
        )

        '''
        full telemetry
        '''
        colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', n_runs, colortype='rgb')
        fig = make_subplots(cols=2, rows=1, subplot_titles=['Model Score', 'vdw Score'])
        for i in range(n_runs):
            x = np.arange(num_iters)
            y = sampling_dict['scores'][i]
            opacity = np.clip(1 - np.abs(np.amax(y) - np.amax(sampling_dict['scores'])) / np.amax(sampling_dict['scores']), a_min=0.1, a_max=1)
            fig.add_trace(go.Scattergl(x=x, y=y, opacity=opacity, line_color=colors[i], name=f'score_{i}'),
                          col=1, row=1)
            fig.add_trace(go.Scattergl(x=sampling_dict['resampled state record'][i], y=y[sampling_dict['resampled state record'][i]],
                                       mode='markers', line_color=colors[i], marker=dict(size=10), opacity=1, showlegend=False),
                          col=1, row=1)
        for i in range(n_runs):
            y = -sampling_dict['vdw penalties'][i]
            opacity = np.clip(1 - np.abs(np.amax(y) - np.amax(sampling_dict['scores'])) / np.amax(sampling_dict['scores']), a_min=0.1, a_max=1)
            fig.add_trace(go.Scattergl(x=x, y=y, opacity=opacity, line_color=colors[i], name=f'vdw_{i}'),
                          col=2, row=1)
            fig.add_trace(go.Scattergl(x=sampling_dict['resampled state record'][i], y=y[sampling_dict['resampled state record'][i]],
                                       mode='markers', line_color=colors[i], marker=dict(size=10), opacity=1, showlegend=False),
                          col=2, row=1)
        # for i in range(n_samples):
        #     opacity = np.clip(1 - np.abs(np.amax(sampling_dict['scores'][i]) - np.amax(sampling_dict['scores'])) / np.amax(sampling_dict['scores']),
        #                       a_min=0.1, a_max=1)
        #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=sampling_dict['acceptance ratio'][i], opacity=opacity, line_color=colors[i], name=f'run_{i}'),
        #                   col=1, row=2)
        # for i in range(n_samples):
        #     opacity = np.clip(1 - np.abs(np.amax(sampling_dict['scores'][i]) - np.amax(sampling_dict['scores'])) / np.amax(sampling_dict['scores']),
        #                       a_min=0.1, a_max=1)
        #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.log10(sampling_dict['step size'][i]), opacity=opacity, line_color=colors[i], name=f'run_{i}'),
        #                   col=2, row=2)
        fig.update_layout(showlegend=True)
        # fig.update_yaxes(range=[0, 1], row=1, col=2)
        fig.layout.margin = self.layout.margin
        # fig.write_image('../paper1_figs/sampling_telemetry.png')
        # wandb.log({'Sampling Telemetry': fig})
        if self.config.machine == 'local':
            import plotly.io as pio
            pio.renderers.default = 'browser'
            fig.show()

    def cell_params_tracking_plot(self, sampling_dict, collater, extra_test_loader):
        num_iters = sampling_dict['scores'].shape[1]
        n_runs = sampling_dict['canonical samples'].shape[1]

        all_samples = torch.tensor(sampling_dict['canonical samples'].reshape(12, n_runs * num_iters).T)
        single_mol_data_0 = extra_test_loader.dataset[0]
        big_single_mol_data = collater([single_mol_data_0 for n in range(len(all_samples))]).cuda()
        override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index('P-1') + 1
        sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(big_single_mol_data.x.device) for i in range(big_single_mol_data.num_graphs)]
        big_single_mol_data = override_sg_info('P-1', self.supercell_builder.dataDims, big_single_mol_data, self.supercell_builder.symmetries_dict, sym_ops_list)
        processed_cell_params = torch.cat(self.supercell_builder.process_cell_params(big_single_mol_data, all_samples.cuda(), rescale_asymmetric_unit=False, skip_cell_cleaning=True), dim=-1).T
        del big_single_mol_data

        processed_cell_params = processed_cell_params.reshape(12, n_runs, num_iters).cpu().detach().numpy()

        fig = make_subplots(rows=4, cols=3, subplot_titles=[
            'a', 'b', 'c', 'alpha', 'beta', 'gamma',
            'x', 'y', 'z', 'phi', 'psi', 'theta'
        ])
        colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', min(10, n_runs), colortype='rgb')
        for i in range(12):
            row = i // 3 + 1
            col = i % 3 + 1
            x = np.arange(num_iters * n_runs)
            for j in range(min(10, n_runs)):
                y = processed_cell_params[i, j]
                opacity = 0.75  # np.clip(1 - np.abs(np.ptp(y) - np.ptp(processed_cell_params[i])) / np.ptp(processed_cell_params[i]), a_min=0.1, a_max=1)
                fig.add_trace(go.Scattergl(x=x, y=y, line_color=colors[j], opacity=opacity),
                              row=row, col=col)
                fig.add_trace(go.Scattergl(x=sampling_dict['resampled state record'][j], y=y[sampling_dict['resampled state record'][j]],
                                           mode='markers', line_color=colors[j], marker=dict(size=7), opacity=opacity, showlegend=False),
                              row=row, col=col)

        fig.update_layout(showlegend=False)
        fig.layout.margin = self.layout.margin
        # fig.write_image('../paper1_figs/sampling_telemetry.png')
        # wandb.log({'Sampling Telemetry': fig})
        if self.config.machine == 'local':
            import plotly.io as pio
            pio.renderers.default = 'browser'
            fig.show()

    def generate_discriminator_negatives(self, epoch_stats_dict, config, data, generator, i):
        n_generators = sum([config.train_discriminator_adversarially, self.config.train_discriminator_on_distorted, self.config.train_discriminator_on_randn])
        gen_random_number = np.random.uniform(0, 1, 1)
        gen_randn_range = np.linspace(0, 1, n_generators + 1)

        if self.config.train_discriminator_adversarially:
            ii = i % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:  # randomly sample which generator to use at each iteration
                generated_samples_i, _ = self.get_generator_samples(data, generator)
                handedness = torch.ones(len(generated_samples_i), device=generated_samples_i.device)
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',np.zeros(len(generated_samples_i)), mode='extend')

        if self.config.train_discriminator_on_randn:
            ii = (i + 1) % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:
                generated_samples_i = self.randn_generator.forward(data.num_graphs, data).to(config.device)
                handedness = None
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',np.ones(len(generated_samples_i)), mode='extend')

        if self.config.train_discriminator_on_distorted:
            ii = (i + 2) % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:
                generated_samples_ii = (data.cell_params - torch.Tensor(self.config.dataDims['lattice means'])) / torch.Tensor(self.config.dataDims['lattice stds'])  # standardize
                if self.config.sample_distortion_magnitude == -1:
                    distortion = torch.randn_like(generated_samples_ii) * torch.logspace(-2.5, -0.5, len(generated_samples_ii)).to(generated_samples_ii.device)[:, None]  # wider range for evaluation mode
                else:
                    distortion = torch.randn_like(generated_samples_ii) * self.config.sample_distortion_magnitude
                generated_samples_i = (generated_samples_ii + distortion).to(config.device)  # add jitter and return in standardized basis
                handedness = data.asym_unit_handedness
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',np.ones(len(generated_samples_i)) * 2, mode='extend')
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'distortion level',torch.linalg.norm(distortion, axis=-1).cpu().detach().numpy(), mode='extend')

        return generated_samples_i, handedness, epoch_stats_dict

    def log_regression_accuracy(self, train_epoch_stats_dict, test_epoch_stats_dict):
        target_mean = self.config.dataDims['target mean']
        target_std = self.config.dataDims['target std']

        target = np.asarray(test_epoch_stats_dict['regressor packing target'])
        prediction = np.asarray(test_epoch_stats_dict['regressor packing prediction'])
        orig_target = target * target_std + target_mean
        orig_prediction = prediction * target_std + target_mean

        volume_ind = self.config.dataDims['tracking features dict'].index('molecule volume')
        mass_ind = self.config.dataDims['tracking features dict'].index('molecule mass')
        molwise_density = test_epoch_stats_dict['tracking features'][:, mass_ind] / test_epoch_stats_dict['tracking features'][:, volume_ind]
        target_density = molwise_density * orig_target * 1.66  # conversion from amu/A^3 to g/mL
        predicted_density = molwise_density * orig_prediction * 1.66

        if train_epoch_stats_dict is not None:
            train_target = np.asarray(train_epoch_stats_dict['regressor packing target'])
            train_prediction = np.asarray(train_epoch_stats_dict['regressor packing prediction'])
            train_orig_target = train_target * target_std + target_mean
            train_orig_prediction = train_prediction * target_std + target_mean

        losses = ['normed error', 'abs normed error', 'squared error']
        loss_dict = {}
        losses_dict = {}
        for loss in losses:
            if loss == 'normed error':
                loss_i = (orig_target - orig_prediction) / np.abs(orig_target)
            elif loss == 'abs normed error':
                loss_i = np.abs((orig_target - orig_prediction) / np.abs(orig_target))
            elif loss == 'squared error':
                loss_i = (orig_target - orig_prediction) ** 2
            losses_dict[loss] = loss_i  # huge unnecessary upload
            loss_dict[loss + ' mean'] = np.mean(loss_i)
            loss_dict[loss + ' std'] = np.std(loss_i)
            print(loss + ' mean: {:.3f} std: {:.3f}'.format(loss_dict[loss + ' mean'], loss_dict[loss + ' std']))

        linreg_result = linregress(orig_target, orig_prediction)
        loss_dict['Regression R'] = linreg_result.rvalue
        loss_dict['Regression slope'] = linreg_result.slope
        wandb.log(loss_dict)

        losses = ['density normed error', 'density abs normed error', 'density squared error']
        loss_dict = {}
        losses_dict = {}
        for loss in losses:
            if loss == 'density normed error':
                loss_i = (target_density - predicted_density) / np.abs(target_density)
            elif loss == 'density abs normed error':
                loss_i = np.abs((target_density - predicted_density) / np.abs(target_density))
            elif loss == 'density squared error':
                loss_i = (target_density - predicted_density) ** 2
            losses_dict[loss] = loss_i  # huge unnecessary upload
            loss_dict[loss + ' mean'] = np.mean(loss_i)
            loss_dict[loss + ' std'] = np.std(loss_i)
            print(loss + ' mean: {:.3f} std: {:.3f}'.format(loss_dict[loss + ' mean'], loss_dict[loss + ' std']))

        linreg_result = linregress(target_density, predicted_density)
        loss_dict['Density Regression R'] = linreg_result.rvalue
        loss_dict['Density Regression slope'] = linreg_result.slope
        wandb.log(loss_dict)

        # log loss distribution
        if self.config.wandb.log_figures:
            # predictions vs target trace
            xline = np.linspace(max(min(orig_target), min(orig_prediction)), min(max(orig_target), max(orig_prediction)), 10)
            fig = go.Figure()
            fig.add_trace(go.Histogram2dContour(x=orig_target, y=orig_prediction, ncontours=50, nbinsx=40, nbinsy=40, showlegend=True))
            fig.update_traces(contours_coloring="fill")
            fig.update_traces(contours_showlines=False)
            fig.add_trace(go.Scattergl(x=orig_target, y=orig_prediction, mode='markers', showlegend=True, opacity=0.5))
            fig.add_trace(go.Scattergl(x=xline, y=xline))
            fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
            fig.update_layout(showlegend=True)
            wandb.log({'Test Packing Coefficient': fig})

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=orig_prediction - orig_target,
                                       histnorm='probability density',
                                       nbinsx=100,
                                       name="Error Distribution",
                                       showlegend=False))
            wandb.log({'Packing Coefficient Error Distribution': fig})

            xline = np.linspace(max(min(target_density), min(predicted_density)), min(max(target_density), max(predicted_density)), 10)
            fig = go.Figure()
            fig.add_trace(go.Histogram2dContour(x=target_density, y=predicted_density, ncontours=50, nbinsx=40, nbinsy=40, showlegend=True))
            fig.update_traces(contours_coloring="fill")
            fig.update_traces(contours_showlines=False)
            fig.add_trace(go.Scattergl(x=target_density, y=predicted_density, mode='markers', showlegend=True, opacity=0.5))
            fig.add_trace(go.Scattergl(x=xline, y=xline))
            fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
            fig.update_layout(showlegend=True)
            wandb.log({'Test Density': fig})

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=predicted_density - target_density,
                                       histnorm='probability density',
                                       nbinsx=100,
                                       name="Error Distribution",
                                       showlegend=False))
            wandb.log({'Density Error Distribution': fig})

            if train_epoch_stats_dict is not None:
                xline = np.linspace(max(min(train_orig_target), min(train_orig_prediction)), min(max(train_orig_target), max(train_orig_prediction)), 10)
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(x=train_orig_target, y=train_orig_prediction, ncontours=50, nbinsx=40, nbinsy=40, showlegend=True))
                fig.update_traces(contours_coloring="fill")
                fig.update_traces(contours_showlines=False)
                fig.add_trace(go.Scattergl(x=train_orig_target, y=train_orig_prediction, mode='markers', showlegend=True, opacity=0.5))
                fig.add_trace(go.Scattergl(x=xline, y=xline))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                fig.update_layout(showlegend=True)
                wandb.log({'Train Packing Coefficient': fig})

            # correlate losses with molecular features
            tracking_features = np.asarray(test_epoch_stats_dict['tracking features'])
            generator_loss_correlations = np.zeros(self.config.dataDims['num tracking features'])
            features = []
            for i in range(self.config.dataDims['num tracking features']):  # not that interesting
                features.append(self.config.dataDims['tracking features dict'][i])
                generator_loss_correlations[i] = np.corrcoef(np.abs((orig_target - orig_prediction) / np.abs(orig_target)), tracking_features[:, i], rowvar=False)[0, 1]

            generator_sort_inds = np.argsort(generator_loss_correlations)
            generator_loss_correlations = generator_loss_correlations[generator_sort_inds]

            fig = go.Figure(go.Bar(
                y=[self.config.dataDims['tracking features dict'][i] for i in range(self.config.dataDims['num tracking features'])],
                x=[generator_loss_correlations[i] for i in range(self.config.dataDims['num tracking features'])],
                orientation='h',
            ))
            wandb.log({'Regressor Loss Correlates': fig})

        return None

    def nice_dataset_analysis(self, dataset):
        '''
        distributions of dataset features
        - molecule num atoms
        - num rings
        - num donors
        - num acceptors
        - atom fractions CNOFCl Metals
        -
        '''
        import plotly.io as pio
        pio.renderers.default = 'browser'

        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=20,  # top margin
            )
        )

        rows = 3
        cols = 4
        mol_feats = ['molecule num atoms', 'molecule num rings', 'molecule num donors', 'molecule num acceptors',
                     'molecule planarity', 'molecule C fraction', 'molecule N fraction', 'molecule O fraction',
                     'crystal packing coefficient', 'crystal lattice centring', 'crystal system', 'crystal z value']

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=mol_feats, horizontal_spacing=0.04, vertical_spacing=0.1)
        for ii, feat in enumerate(mol_feats):
            fig.add_trace(go.Histogram(x=dataset[feat],
                                       histnorm='probability density',
                                       nbinsx=50,
                                       showlegend=False,
                                       marker_color='#0c4dae'),
                          row=(ii) // cols + 1, col=(ii) % cols + 1)
        fig.update_layout(width=900, height=600)
        fig.layout.margin = layout.margin
        fig.write_image('../paper1_figs/dataset_statistics.png', scale=4)
        if self.config.machine == 'local':
            fig.show()

        return None

    def cell_params_analysis(self, train_loader, test_epoch_stats_dict):
        n_crystal_features = self.config.dataDims['num lattice features']
        generated_samples = test_epoch_stats_dict['generated cell parameters']
        if generated_samples.ndim == 3:
            generated_samples = generated_samples[0]
        means = self.config.dataDims['lattice means']
        stds = self.config.dataDims['lattice stds']

        # slightly expensive to do this every time
        dataset_cell_distribution = np.asarray([train_loader.dataset[ii].cell_params[0].cpu().detach().numpy() for ii in range(len(train_loader.dataset))])

        # raw outputs
        renormalized_samples = np.zeros_like(generated_samples)
        for i in range(generated_samples.shape[1]):
            renormalized_samples[:, i] = generated_samples[:, i] * stds[i] + means[i]

        cleaned_samples = test_epoch_stats_dict['final generated cell parameters']

        overlaps_1d = {}
        sample_means = {}
        sample_stds = {}
        for i, key in enumerate(self.config.dataDims['lattice features']):
            mini, maxi = np.amin(dataset_cell_distribution[:, i]), np.amax(dataset_cell_distribution[:, i])
            h1, r1 = np.histogram(dataset_cell_distribution[:, i], bins=100, range=(mini, maxi))
            h1 = h1 / len(dataset_cell_distribution[:, i])

            h2, r2 = np.histogram(cleaned_samples[:, i], bins=r1)
            h2 = h2 / len(cleaned_samples[:, i])

            overlaps_1d[f'{key} 1D Overlap'] = np.min(np.concatenate((h1[None], h2[None]), axis=0), axis=0).sum()

            sample_means[f'{key} mean'] = np.mean(cleaned_samples[:, i])
            sample_stds[f'{key} std'] = np.std(cleaned_samples[:, i])

        average_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys()])
        overlaps_1d['average 1D overlap'] = average_overlap
        wandb.log(overlaps_1d.copy())
        wandb.log(sample_means)
        wandb.log(sample_stds)
        print("1D Overlap With Data:{:.3f}".format(average_overlap))

        if self.config.wandb.log_figures:
            fig_dict = {}  # consider replacing by Joy plot

            # bar graph of 1d overlaps
            fig = go.Figure(go.Bar(
                y=list(overlaps_1d.keys()),
                x=[overlaps_1d[key] for key in overlaps_1d],
                orientation='h',
                marker=dict(color='red')
            ))
            fig_dict['1D overlaps'] = fig

            # 1d Histograms
            for i in range(n_crystal_features):
                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=dataset_cell_distribution[:, i],
                    histnorm='probability density',
                    nbinsx=100,
                    name="Dataset samples",
                    showlegend=True,
                ))

                fig.add_trace(go.Histogram(
                    x=renormalized_samples[:, i],
                    histnorm='probability density',
                    nbinsx=100,
                    name="Samples",
                    showlegend=True,
                ))
                fig.add_trace(go.Histogram(
                    x=cleaned_samples[:, i],
                    histnorm='probability density',
                    nbinsx=100,
                    name="Cleaned Samples",
                    showlegend=True,
                ))
                fig.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                fig.update_traces(opacity=0.5)

                fig_dict[self.config.dataDims['lattice features'][i] + ' distribution'] = fig

            wandb.log(fig_dict)

    def make_nice_figures(self):
        '''
        make beautiful figures for papers / presentation
        '''
        import plotly.io as pio
        pio.renderers.default = 'browser'

        figures = 'nov_22'  # figures from the late 2022 JCTC draft submissions
        if figures == 'nov_22':
            with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=[self.config.wandb.experiment_tag]):
                wandb.run.name = wandb.config.machine + '_' + str(wandb.config.run_num)  # overwrite procedurally generated run name with our run name
                wandb.run.save()

                # self.nice_dataset_analysis(self.prep_dataset)
                self.training_prep()
                from reporting.nov_22_regressor import nice_regression_plots
                nice_regression_plots(self.config)
                from reporting.nov_22_discriminator_final import nice_scoring_plots
                nice_scoring_plots(self.config, wandb)

        return

    def slash_batch(self, train_loader, test_loader):
        slash_increment = max(4, int(train_loader.batch_size * 0.1))
        train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size - slash_increment)
        test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size - slash_increment)
        print('==============================')
        print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
        print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
        print('==============================')
        wandb.log({'batch size': train_loader.batch_size})

        return train_loader, test_loader

    def update_batch_size(self, train_loader, test_loader, extra_test_loader):
        if self.config.grow_batch_size:
            if (train_loader.batch_size < len(train_loader.dataset)) and (train_loader.batch_size < self.config.max_batch_size):  # if the batch is smaller than the dataset
                increment = max(4, int(train_loader.batch_size * self.config.batch_growth_increment))  # increment batch size
                train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size + increment)
                test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size + increment)
                if extra_test_loader is not None:
                    extra_test_loader = update_dataloader_batch_size(extra_test_loader, extra_test_loader.batch_size + increment)
                print(f'Batch size incremented to {train_loader.batch_size}')
        wandb.log({'batch size': train_loader.batch_size})
        self.config.current_batch_size = train_loader.batch_size
        return train_loader, test_loader, extra_test_loader

    def check_model_convergence(self, metrics_dict):
        generator_convergence = checkConvergence(metrics_dict['generator test loss'], self.config.history, self.config.generator_optimizer.convergence_eps)
        discriminator_convergence = checkConvergence(metrics_dict['discriminator test loss'], self.config.history, self.config.discriminator_optimizer.convergence_eps)
        conditioner_convergence = checkConvergence(metrics_dict['conditioner test loss'], self.config.history, self.config.conditioner_optimizer.convergence_eps)
        regressor_convergence = checkConvergence(metrics_dict['regressor test loss'], self.config.history, self.config.regressor_optimizer.convergence_eps)

        # if generator_convergence:
        #     print('generator converged!')
        # if discriminator_convergence:
        #     print('discriminator converged!')
        # if conditioner_convergence:
        #     print('conditioner converged!')
        # if regressor_convergence:
        #     print('regressor converged!')

        return generator_convergence, discriminator_convergence, conditioner_convergence, regressor_convergence

    def model_checkpointing(self, epoch, config, discriminator, generator, conditioner, regressor,
                            discriminator_optimizer, generator_optimizer, conditioner_optimizer, regressor_optimizer,
                            generator_err_te, discriminator_err_te, conditioner_err_te, regressor_err_te, metrics_dict):
        if config.save_checkpoints:  # config.machine == 'cluster':  # every 5 epochs, save a checkpoint
            # if (epoch > 0) and (epoch % 5 == 0):
            #     # saving early-stopping checkpoint
            #     save_checkpoint(epoch, discriminator, discriminator_optimizer, self.config.discriminator.__dict__, 'discriminator_' + str(config.run_num) + f'_epoch_{epoch}')
            #     save_checkpoint(epoch, generator, generator_optimizer, self.config.generator.__dict__, 'generator_' + str(config.run_num) + f'_epoch_{epoch}')

            # or save any checkpoint which is a new best
            if epoch > 0:
                if np.average(discriminator_err_te) < np.amin(metrics_dict['discriminator test loss'][:-1]):
                    print("Saving discriminator checkpoint")
                    save_checkpoint(epoch, discriminator, discriminator_optimizer, self.config.discriminator.__dict__, 'best_discriminator_' + str(config.run_num))
                if np.average(generator_err_te) < np.amin(metrics_dict['generator test loss'][:-1]):
                    print("Saving generator checkpoint")
                    save_checkpoint(epoch, generator, generator_optimizer, self.config.generator.__dict__, 'best_generator_' + str(config.run_num))
                if np.average(conditioner_err_te) < np.amin(metrics_dict['conditioner test loss'][:-1]):
                    print("Saving conditioner checkpoint")
                    save_checkpoint(epoch, conditioner, conditioner_optimizer, self.config.conditioner.__dict__, 'best_conditioner_' + str(config.run_num))
                if np.average(regressor_err_te) < np.amin(metrics_dict['regressor test loss'][:-1]):
                    print("Saving regressor checkpoint")
                    save_checkpoint(epoch, regressor, regressor_optimizer, self.config.regressor.__dict__, 'best_regressor_' + str(config.run_num))

        return None

    def update_lr(self, discriminator_schedulers, discriminator_optimizer, discriminator_err_tr, discriminator_hit_max_lr,
                  generator_schedulers, generator_optimizer, generator_err_tr, generator_hit_max_lr,
                  conditioner_schedulers, conditioner_optimizer, conditioner_err_tr, conditioner_hit_max_lr,
                  regressor_schedulers, regressor_optimizer, regressor_err_tr, regressor_hit_max_lr,
                  ):  # update learning rate

        discriminator_optimizer, discriminator_lr = set_lr(discriminator_schedulers, discriminator_optimizer, self.config.discriminator_optimizer.lr_schedule,
                                                           self.config.discriminator_optimizer.min_lr, self.config.discriminator_optimizer.max_lr, discriminator_err_tr, discriminator_hit_max_lr)
        discriminator_learning_rate = discriminator_optimizer.param_groups[0]['lr']
        if discriminator_learning_rate >= self.config.discriminator_optimizer.max_lr: discriminator_hit_max_lr = True

        generator_optimizer, generator_lr = set_lr(generator_schedulers, generator_optimizer, self.config.generator_optimizer.lr_schedule,
                                                   self.config.generator_optimizer.min_lr, self.config.generator_optimizer.max_lr, generator_err_tr, generator_hit_max_lr)
        generator_learning_rate = generator_optimizer.param_groups[0]['lr']
        if generator_learning_rate >= self.config.generator_optimizer.max_lr: generator_hit_max_lr = True

        conditioner_optimizer, conditioner_lr = set_lr(conditioner_schedulers, conditioner_optimizer, self.config.conditioner_optimizer.lr_schedule,
                                                       self.config.conditioner_optimizer.min_lr, self.config.conditioner_optimizer.max_lr, conditioner_err_tr, conditioner_hit_max_lr)
        conditioner_learning_rate = conditioner_optimizer.param_groups[0]['lr']
        if conditioner_learning_rate >= self.config.conditioner_optimizer.max_lr: conditioner_hit_max_lr = True

        regressor_optimizer, regressor_lr = set_lr(regressor_schedulers, regressor_optimizer, self.config.regressor_optimizer.lr_schedule,
                                                   self.config.regressor_optimizer.min_lr, self.config.regressor_optimizer.max_lr, regressor_err_tr, regressor_hit_max_lr)
        regressor_learning_rate = regressor_optimizer.param_groups[0]['lr']
        if regressor_learning_rate >= self.config.regressor_optimizer.max_lr: regressor_hit_max_lr = True

        # print(f"Learning rates are d={discriminator_lr:.5f}, g={generator_lr:.5f}, a={conditioner_lr:.5f}, r={regressor_lr:.5f}")

        return discriminator_optimizer, discriminator_learning_rate, discriminator_hit_max_lr, \
               generator_optimizer, generator_learning_rate, generator_hit_max_lr, \
               conditioner_optimizer, conditioner_learning_rate, conditioner_hit_max_lr, \
               regressor_optimizer, regressor_learning_rate, regressor_hit_max_lr

    def post_run_evaluation(self, epoch, generator, discriminator, discriminator_optimizer, generator_optimizer, metrics_dict, train_loader, test_loader, extra_test_loader):
        '''
        run post-training evaluation
        '''
        # reload best test
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

        # rerun test inference
        with torch.no_grad():
            discriminator_err_te, discriminator_te_record, generator_err_te, generator_te_record, test_epoch_stats_dict, time_test = \
                self.run_epoch(dataLoader=test_loader, generator=generator, discriminator=discriminator,
                               update_gradients=False, record_stats=True, epoch=epoch)  # compute loss on test set
            np.save(f'../{self.config.run_num}_test_epoch_stats_dict', test_epoch_stats_dict)

            if extra_test_loader is not None:
                extra_test_epoch_stats_dict, time_test_ex = \
                    self.discriminator_evaluation(self.config, dataLoader=extra_test_loader, discriminator=discriminator)  # compute loss on test set

                np.save(f'../{self.config.run_num}_extra_test_dict', extra_test_epoch_stats_dict)
            else:
                extra_test_epoch_stats_dict = None

        # save results
        metrics_dict = self.update_gan_metrics(
            epoch, metrics_dict,
            np.zeros(10), discriminator_err_te,
            np.zeros(10), generator_err_te,
            discriminator_optimizer.defaults['lr'], generator_optimizer.defaults['lr'])

        self.log_gan_loss(metrics_dict, None, test_epoch_stats_dict,
                          None, discriminator_te_record, None, generator_te_record)

        self.detailed_reporting(epoch, train_loader, None, test_epoch_stats_dict,
                                extra_test_dict=extra_test_epoch_stats_dict)

    def compute_similarity_penalty(self, generated_samples, prior):
        '''
        punish batches in which the samples are too self-similar

        Parameters
        ----------
        generated_samples
        prior

        Returns
        -------
        '''
        if len(generated_samples) >= 3:
            # enforce that the distance between samples is similar to the distance between priors
            prior_dists = torch.cdist(prior, prior, p=2)
            sample_dists = torch.cdist(generated_samples, generated_samples, p=2)
            similarity_penalty = F.smooth_l1_loss(input=sample_dists, target=prior_dists, reduction='none').mean(1)  # align distances to all other samples

            # todo set the standardization for each space group individually (different stats and distances)

        else:
            similarity_penalty = None

        return similarity_penalty

    def score_adversarially(self, supercell_data, discriminator):
        '''
        get an adversarial score for generated samples

        Parameters
        ----------
        supercell_data
        discriminator

        Returns
        -------

        '''
        if supercell_data is not None:  # if we built the supercells, we'll want to do this analysis anyway
            if (self.config.device.lower() == 'cuda') and (supercell_data.x.device != 'cuda'):
                supercell_data = supercell_data.cuda()

            if self.config.test_mode or self.config.anomaly_detection:
                assert torch.sum(torch.isnan(supercell_data.x)) == 0, "NaN in training input"

            discriminator_score, dist_dict = self.adversarial_score(discriminator, supercell_data)
        else:
            discriminator_score = None
            dist_dict = None

        return discriminator_score, dist_dict

    def get_vdw_penalty(self, dist_dict=None, num_graphs=None, data=None):
        if dist_dict is not None:  # supercell_data is not None: # do vdw computation even if we don't need it
            vdw_overlap_sum, normed_vdw_overlap_sum, penalties = \
                raw_vdw_overlap(self.vdw_radii, dists=dist_dict['intermolecular dist'],
                                atomic_numbers=dist_dict['intermolecular dist atoms'],
                                batch_numbers=dist_dict['intermolecular dist batch'],
                                num_graphs=num_graphs)

            scores = torch.nan_to_num(
                torch.stack(
                    [torch.sum(penalties[ii]) for ii in range(num_graphs)]
                )) / torch.diff(data.ptr)
            #
            # top_scores = torch.nan_to_num(
            #     torch.stack(
            #         # [torch.mean(torch.topk(penalties[crystal_number == ii], 5)[0]) for ii in range(num_graphs)]
            #         [torch.max(penalties[ii]) if (len(penalties[ii]) > 0) else torch.zeros(1)[0].to(vdw_overlap_sum.device) for ii in range(num_graphs)]
            #     ))
            #
            # scores = (scores_i + top_scores) / 2

            return scores, normed_vdw_overlap_sum / torch.diff(data.ptr)

        else:
            return None, None

    def aggregate_generator_losses(self, epoch_stats_dict, packing_loss, adversarial_score, vdw_loss, similarity_penalty, packing_prediction, packing_target, h_bond_score, combo_score):
        generator_losses_list = []
        stats_keys, stats_values = [], []
        if self.config.train_generator_packing:
            generator_losses_list.append(packing_loss.float())

        if packing_loss is not None:
            stats_keys += ['generator packing loss', 'generator packing prediction',
                           'generator packing target', 'generator packing mae']
            stats_values += [packing_loss.cpu().detach().numpy(), packing_prediction,
                             packing_target, np.abs(packing_prediction - packing_target) / packing_target]

        if adversarial_score is not None:
            softmax_adversarial_score = F.softmax(adversarial_score, dim=1)[:, 1]  # modified minimax
            adversarial_loss = -torch.log(softmax_adversarial_score)  # modified minimax
            stats_keys += ['generator adversarial loss']
            stats_values += [adversarial_loss.cpu().detach().numpy()]

        if self.config.train_generator_adversarially:
            generator_losses_list.append(adversarial_loss)

        if vdw_loss is not None:
            stats_keys += ['generator per mol vdw loss']
            stats_values += [vdw_loss.cpu().detach().numpy()]

        if self.config.train_generator_vdw:
            if self.config.vdw_loss_rescaling == 'log':
                vdw_loss_f = torch.log(1 + vdw_loss)  # soft rescaling to be gentler on outliers
            elif self.config.vdw_loss_rescaling is None:
                vdw_loss_f = vdw_loss
            elif self.config.vdw_loss_rescaling == 'mse':
                vdw_loss_f = vdw_loss ** 2

            generator_losses_list.append(vdw_loss_f)

        if self.config.train_generator_h_bond:
            generator_losses_list.append(h_bond_score)
        if h_bond_score is not None:
            stats_keys += ['generator h bond loss']
            stats_values += [h_bond_score.cpu().detach().numpy()]

        if self.config.generator_similarity_penalty != 0:
            if similarity_penalty is not None:
                generator_losses_list.append(self.config.generator_similarity_penalty * similarity_penalty)
            else:
                print('similarity penalty was none')
        if similarity_penalty is not None:
            stats_keys += ['generator similarity loss']
            stats_values += [similarity_penalty.cpu().detach().numpy()]

        if self.config.train_generator_combo:
            generator_losses_list.append(-combo_score)
        if combo_score is not None:
            stats_keys += ['generator combo loss']
            stats_values += [1 - combo_score.cpu().detach().numpy()]

        generator_losses = torch.sum(torch.stack(generator_losses_list), dim=0)
        epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values)

        return generator_losses, epoch_stats_dict

    def cell_generation_analysis(self, epoch_stats_dict):
        '''
        do analysis and plotting for cell generator
        '''  # todo add loss correlates
        layout = self.plotly_setup()
        self.log_cubic_defect(epoch_stats_dict)
        wandb.log({"Generated cell parameter variation": epoch_stats_dict['generated cell parameters'].std(0).mean()})
        generator_losses, average_losses_dict = self.process_generator_losses(epoch_stats_dict)
        wandb.log(average_losses_dict)

        self.cell_density_plot(epoch_stats_dict, layout)
        self.all_losses_plot(epoch_stats_dict, generator_losses, layout)
        self.save_3d_structure_examples(epoch_stats_dict)
        self.sample_wise_analysis(epoch_stats_dict, layout)
        self.plot_generator_loss_correlates(epoch_stats_dict, generator_losses, layout)

        return None

    def log_supercell_examples(self, supercell_examples, i, rand_batch_ind, epoch_stats_dict):
        if (supercell_examples is not None) and (i == rand_batch_ind):  # for a random batch in the epoch
            epoch_stats_dict['generated supercell examples'] = supercell_examples.cpu().detach()
            # if supercell_examples.num_graphs > 100:  # todo find a way to take only the few that we need - maybe using the Collater
            #     print('WARNING. Saving over 100 supercells for analysis')

        epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'final generated cell parameters',
                                             supercell_examples.cell_params.cpu().detach().numpy(), mode='extend')

        del supercell_examples
        return epoch_stats_dict

    def compute_h_bond_score(self, supercell_data=None):
        if (supercell_data is not None) and (self.config.feature_richness == 'full'):  # supercell_data is not None: # do vdw computation even if we don't need it
            # get the total per-molecule counts
            mol_acceptors = supercell_data.tracking[:, self.config.dataDims['tracking features dict'].index('molecule num acceptors')]
            mol_donors = supercell_data.tracking[:, self.config.dataDims['tracking features dict'].index('molecule num donors')]

            '''
            count pairs within a close enough bubble ~2.7-3.3 Angstroms
            '''
            h_bonds_loss = []
            for i in range(supercell_data.num_graphs):
                if (mol_donors[i]) > 0 and (mol_acceptors[i] > 0):
                    h_bonds = compute_num_h_bonds(supercell_data, self.config.dataDims, i)

                    bonds_per_possible_bond = h_bonds / min(mol_donors[i], mol_acceptors[i])
                    h_bond_loss = 1 - torch.tanh(2 * bonds_per_possible_bond)  # smoother gradient about 0

                    h_bonds_loss.append(h_bond_loss)
                else:
                    h_bonds_loss.append(torch.zeros(1)[0].to(supercell_data.x.device))
            h_bond_loss_f = torch.stack(h_bonds_loss)
        else:
            h_bond_loss_f = None

        return h_bond_loss_f

    def log_cubic_defect(self, epoch_stats_dict):
        cleaned_samples = epoch_stats_dict['final generated cell parameters']
        cubic_distortion = np.abs(1 - np.nan_to_num(np.stack([cell_vol(cleaned_samples[i, 0:3], cleaned_samples[i, 3:6]) / np.prod(cleaned_samples[i, 0:3], axis=-1) for i in range(len(cleaned_samples))])))
        wandb.log({'Avg generated cubic distortion': np.average(cubic_distortion)})
        hist = np.histogram(cubic_distortion, bins=256, range=(0, 1))
        wandb.log({"Generated cubic distortions": wandb.Histogram(np_histogram=hist, num_bins=256)})

    def plotly_setup(self):
        if self.config.machine == 'local':
            import plotly.io as pio
            pio.renderers.default = 'browser'

        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=20,  # top margin
            )
        )
        return layout

    def process_generator_losses(self, epoch_stats_dict):
        generator_loss_keys = ['generator packing prediction', 'generator packing target', 'generator per mol vdw loss', 'generator adversarial loss', 'generator h bond loss', 'generator combo loss']
        generator_losses = {}
        for key in generator_loss_keys:
            if key in epoch_stats_dict.keys():
                if epoch_stats_dict[key] is not None:
                    if key == 'generator adversarial loss':
                        if self.config.train_generator_adversarially:
                            generator_losses[key[10:]] = np.concatenate(epoch_stats_dict[key])
                        else:
                            pass
                    else:
                        generator_losses[key[10:]] = np.concatenate(epoch_stats_dict[key])

                    if key == 'generator packing target':
                        generator_losses['packing normed mae'] = np.abs(generator_losses['packing prediction'] - generator_losses['packing target']) / generator_losses['packing target']
                        del generator_losses['packing prediction'], generator_losses['packing target']
                else:
                    generator_losses[key[10:]] = None

        return generator_losses, {key: np.average(value) for i, (key, value) in enumerate(generator_losses.items()) if value is not None}

    def cell_density_plot(self, epoch_stats_dict, layout):
        if epoch_stats_dict['generator packing prediction'] is not None and \
                epoch_stats_dict['generator packing target'] is not None:

            x = np.concatenate(epoch_stats_dict['generator packing target'])  # generator_losses['generator per mol vdw loss']
            y = np.concatenate(epoch_stats_dict['generator packing prediction'])  # generator_losses['generator packing loss']

            xy = np.vstack([x, y])
            try:
                z = gaussian_kde(xy)(xy)
            except:
                z = np.ones_like(x)

            xline = np.asarray([np.amin(x), np.amax(x)])
            linreg_result = linregress(x, y)
            yline = xline * linreg_result.slope + linreg_result.intercept

            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                                       mode='markers', marker=dict(color=z), opacity=1))

            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f' R={linreg_result.rvalue:.3f}, m={linreg_result.slope:.3f}'))

            fig.add_trace(go.Scattergl(x=xline, y=xline, marker_color='rgba(0,0,0,1)', showlegend=False))

            fig.layout.margin = layout.margin
            fig.update_layout(xaxis_title='packing target', yaxis_title='packing prediction')

            # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
            if self.config.wandb.log_figures:
                wandb.log({'Cell Packing': fig})
            if (self.config.machine == 'local') and False:
                fig.show()

    def all_losses_plot(self, epoch_stats_dict, generator_losses, layout):
        num_samples = min(10, epoch_stats_dict['generated supercell examples'].num_graphs)
        supercell_identifiers = [epoch_stats_dict['generated supercell examples'].csd_identifier[i] for i in range(num_samples)]
        supercell_inds = [np.argwhere(epoch_stats_dict['identifiers'] == ident)[0, 0] for ident in supercell_identifiers]

        generator_losses_i = {key: value[supercell_inds] for i, (key, value) in enumerate(generator_losses.items())}  # limit to 10 samples
        generator_losses_i['identifier'] = supercell_identifiers
        losses = list(generator_losses_i.keys())
        fig = px.bar(generator_losses_i, x="identifier", y=losses)

        fig.layout.margin = layout.margin
        fig.update_layout(xaxis_title='Sample', yaxis_title='Per-Sample Losses')

        # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
        if self.config.wandb.log_figures:
            wandb.log({'Cell Generation Losses': fig})
        if (self.config.machine == 'local') and False:
            fig.show()

        x = generator_losses['packing normed mae']
        y = generator_losses['per mol vdw loss']
        xy = np.vstack([x, y])
        try:
            z = gaussian_kde(xy)(xy)
        except:
            z = np.ones_like(x)

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                                   mode='markers', marker=dict(color=z), opacity=1))
        fig.layout.margin = layout.margin
        fig.update_layout(xaxis_title='Packing Loss', yaxis_title='vdW Loss')
        fig.update_layout(yaxis_range=[0, 10], xaxis_range=[0, 2])

        # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
        if self.config.wandb.log_figures:
            wandb.log({'Loss Balance': fig})
        if (self.config.machine == 'local') and False:
            fig.show()

    def save_3d_structure_examples(self, epoch_stats_dict):
        num_samples = min(10, epoch_stats_dict['generated supercell examples'].num_graphs)
        generated_supercell_examples = epoch_stats_dict['generated supercell examples']
        identifiers = [generated_supercell_examples.csd_identifier[i] for i in range(num_samples)]
        sgs = [str(int(generated_supercell_examples.sg_ind[i])) for i in range(num_samples)]

        crystals = [ase_mol_from_crystaldata(generated_supercell_examples, highlight_aux=False,
                                             index=i, exclusion_level='distance', inclusion_distance=4)
                    for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

        for i in range(len(crystals)):
            ase.io.write(f'supercell_{i}.pdb', crystals[i])
            wandb.log({'Generated Supercells': wandb.Molecule(open(f"supercell_{i}.pdb"), caption=identifiers[i] + ' ' + sgs[i])})

        mols = [ase_mol_from_crystaldata(generated_supercell_examples,
                                         index=i, exclusion_level='conformer')
                for i in range(min(num_samples, generated_supercell_examples.num_graphs))]
        for i in range(len(mols)):
            ase.io.write(f'conformer_{i}.pdb', mols[i])
            wandb.log({'Single Conformers': wandb.Molecule(open(f"conformer_{i}.pdb"), caption=identifiers[i])})

        return None

    def sample_wise_analysis(self, epoch_stats_dict, layout):
        num_samples = 10
        supercell_examples = epoch_stats_dict['generated supercell examples']
        vdw_loss, normed_vdw_loss, vdw_penalties = \
            vdw_overlap(self.vdw_radii, crystaldata=supercell_examples, return_atomwise=True, return_normed=True,
                        graph_sizes=supercell_examples.tracking[:, self.config.dataDims['tracking features dict'].index('molecule num atoms')])
        vdw_loss /= supercell_examples.tracking[:, self.config.dataDims['tracking features dict'].index('molecule num atoms')]

        # mol_acceptors = supercell_examples.tracking[:, self.config.dataDims['tracking features dict'].index('molecule num acceptors')]
        # mol_donors = supercell_examples.tracking[:, self.config.dataDims['tracking features dict'].index('molecule num donors')]
        # possible_h_bonds = torch.amin(torch.vstack((mol_acceptors, mol_donors)), dim=0)
        # num_h_bonds = torch.stack([compute_num_h_bonds(supercell_examples, self.config.dataDims, i) for i in range(supercell_examples.num_graphs)])

        volumes_list = []
        for i in range(supercell_examples.num_graphs):
            volumes_list.append(cell_vol_torch(supercell_examples.cell_params[i, 0:3], supercell_examples.cell_params[i, 3:6]))
        volumes = torch.stack(volumes_list)
        generated_packing_coeffs = (supercell_examples.Z * supercell_examples.tracking[:, self.mol_volume_ind] / volumes).cpu().detach().numpy()
        target_packing = (supercell_examples.y * self.config.dataDims['target std'] + self.config.dataDims['target mean']).cpu().detach().numpy()

        fig = go.Figure()
        for i in range(min(supercell_examples.num_graphs, num_samples)):
            pens = vdw_penalties[i].cpu().detach()
            fig.add_trace(go.Violin(x=pens[pens != 0], side='positive', orientation='h',
                                    bandwidth=0.01, width=1, showlegend=False, opacity=1,
                                    name=f'{supercell_examples.csd_identifier[i]} <br /> ' +
                                         f'c_t={target_packing[i]:.2f} c_p={generated_packing_coeffs[i]:.2f} <br /> ' +
                                         f'tot_norm_ov={normed_vdw_loss[i]:.2f}'),
                          )

            molecule = rdkit.Chem.MolFromSmiles(supercell_examples[i].smiles)
            try:
                rdkit.Chem.AllChem.Compute2DCoords(molecule)
                rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
                pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(500, 500))
                pil_image.save('mol_img.png', 'png')
                # Add trace
                img = Image.open("mol_img.png")
                # Add images
                fig.add_layout_image(
                    dict(
                        source=img,
                        xref="x domain", yref="y domain",
                        x=.1 + (.15 * (i % 2)), y=i / 10.5 + 0.05,
                        sizex=.15, sizey=.15,
                        xanchor="center",
                        yanchor="middle",
                        opacity=0.75
                    )
                )
            except:
                pass

        # fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
        # fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
        fig.update_layout(width=800, height=600, font=dict(size=12), xaxis_range=[-1, 4])
        fig.layout.margin = layout.margin
        fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
        fig.update_layout(xaxis_title='Nonzero vdW overlaps', yaxis_title='packing prediction')

        # fig.write_image('../paper1_figs/sampling_scores.png')
        wandb.log({'Generated Sample Analysis': fig})
        # fig.show()

        return None

    def report_conditioner_training(self, epoch_stats_dict):
        reconstruction_losses = np.concatenate(epoch_stats_dict['reconstruction loss']).flatten()
        pack_true = np.concatenate(epoch_stats_dict['conditioner packing target']).flatten() * self.config.dataDims['target std'] + self.config.dataDims['target mean']
        pack_pred = np.concatenate(epoch_stats_dict['conditioner packing prediction']).flatten() * self.config.dataDims['target std'] + self.config.dataDims['target mean']
        packing_mae = np.abs(pack_true - pack_pred) / pack_true

        wandb.log({
            'reconstruction loss': reconstruction_losses.mean(),
            'packing MAE': packing_mae.mean(),
        })
        layout = self.plotly_setup()
        x = packing_mae
        y = reconstruction_losses
        xy = np.vstack([x, y])
        #
        # try:
        #     z = gaussian_kde(xy)(xy)
        # except:
        #     z = np.ones_like(x)
        #
        # fig = go.Figure()
        # fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
        #                            mode='markers', marker=dict(color=z), opacity=0.5))
        # fig.layout.margin = layout.margin
        # fig.update_layout(xaxis_title='Packing Loss', yaxis_title='Reconstruction Loss')
        #
        # # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
        # if self.config.wandb.log_figures:
        #     wandb.log({'Conditioner Loss Balance': fig})
        # if (self.config.machine == 'local') and False:
        #     fig.show()

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]])

        for img_i in range(2):
            sample_guess = epoch_stats_dict['prediction_sample'][0, img_i].argmax(0)
            sample_density = np_softmax(epoch_stats_dict['prediction_sample'][0, img_i][None, ...])[0, 1:].sum(0)
            sample_true = epoch_stats_dict['target_sample'][0, img_i]

            X, Y, Z = (sample_guess + 1).nonzero()
            fig.add_trace(go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=sample_density.flatten(),
                isomin=0.00001,
                isomax=1,
                opacity=0.05,  # needs to be small to see through all surfaces
                surface_count=50,  # needs to be a large number for good volume rendering
                colorscale='Jet',
                cmin=0,
                showlegend=True,
                # caps=dict(x_show=False, y_show=False, z_show=False),
            ), row=1, col=img_i + 1)

            x, y, z = sample_true.nonzero()
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                showlegend=True,
                marker=dict(
                    size=10,
                    color=sample_true[x, y, z],
                    colorscale='Jet',
                    cmin=0, cmax=6,
                    opacity=0.5
                )), row=1, col=img_i + 1)
            fig.update_layout(showlegend=True)

        if self.config.wandb.log_figures:
            wandb.log({'Conditioner Reconstruction Samples': fig})
        if (self.config.machine == 'local') and False:
            fig.show()

        class_names = ['empty'] + [str(key) for key in self.config.conditioner_classes.keys()]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='True', x=class_names, y=epoch_stats_dict['conditioner particle true'].mean(0)))
        fig.add_trace(go.Bar(name='Predictions', x=class_names, y=epoch_stats_dict['conditioner particle prediction'].mean(0)))
        fig.update_layout(barmode='group')
        fig.update_yaxes(type='log')
        fig.layout.margin = layout.margin

        if self.config.wandb.log_figures:
            wandb.log({'Conditioner Classwise Density': fig})
        if (self.config.machine == 'local') and False:
            fig.show()


        x = pack_true  # generator_losses['generator per mol vdw loss']
        y = pack_pred  # generator_losses['generator packing loss']

        xy = np.vstack([x, y])
        try:
            z = gaussian_kde(xy)(xy)
        except:
            z = np.ones_like(x)

        xline = np.asarray([np.amin(x), np.amax(x)])
        linreg_result = linregress(x, y)
        yline = xline * linreg_result.slope + linreg_result.intercept

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                                   mode='markers', marker=dict(color=z), opacity=1))

        fig.add_trace(go.Scattergl(x=xline, y=yline, name=f' R={linreg_result.rvalue:.3f}, m={linreg_result.slope:.3f}'))

        fig.add_trace(go.Scattergl(x=xline, y=xline, marker_color='rgba(0,0,0,1)', showlegend=False))

        fig.layout.margin = layout.margin
        fig.update_layout(xaxis_title='packing target', yaxis_title='packing prediction')

        # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
        if self.config.wandb.log_figures:
            wandb.log({'Cell Packing': fig})
        if (self.config.machine == 'local') and False:
            fig.show()

        return None

    def discriminator_analysis(self, epoch_stats_dict):
        '''
        do analysis and plotting for cell discriminator

        -: scores distribution and vdw penalty by sample source
        -: loss correlates
        '''
        layout = self.plotly_setup()

        scores_dict, vdw_penalty_dict, tracking_features_dict = self.process_discriminator_outputs(epoch_stats_dict)
        self.discriminator_scores_plot(scores_dict, vdw_penalty_dict, layout)
        self.plot_discriminator_score_correlates(epoch_stats_dict, layout)

        return None

    def process_discriminator_outputs(self, epoch_stats_dict):
        scores_dict = {}
        vdw_penalty_dict = {}
        tracking_features_dict = {}

        generator_inds = np.where(epoch_stats_dict['generator sample source'] == 0)
        randn_inds = np.where(epoch_stats_dict['generator sample source'] == 1)[0]
        distorted_inds = np.where(epoch_stats_dict['generator sample source'] == 2)[0]

        scores_dict['CSD'] = softmax_and_score(epoch_stats_dict['discriminator real score'])
        scores_dict['Gaussian'] = softmax_and_score(epoch_stats_dict['discriminator fake score'][randn_inds])
        scores_dict['Generator'] = softmax_and_score(epoch_stats_dict['discriminator fake score'][generator_inds])
        scores_dict['Distorted'] = softmax_and_score(epoch_stats_dict['discriminator fake score'][distorted_inds])

        tracking_features_dict['CSD'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], epoch_stats_dict['tracking features'].T)}
        tracking_features_dict['Distorted'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], epoch_stats_dict['tracking features'][distorted_inds].T)}
        tracking_features_dict['Gaussian'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], epoch_stats_dict['tracking features'][randn_inds].T)}
        tracking_features_dict['Generator'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], epoch_stats_dict['tracking features'][generator_inds].T)}

        vdw_penalty_dict['CSD'] = epoch_stats_dict['real vdw penalty']
        vdw_penalty_dict['Gaussian'] = epoch_stats_dict['fake vdw penalty'][randn_inds]
        vdw_penalty_dict['Generator'] = epoch_stats_dict['fake vdw penalty'][generator_inds]
        vdw_penalty_dict['Distorted'] = epoch_stats_dict['fake vdw penalty'][distorted_inds]

        return scores_dict, vdw_penalty_dict, tracking_features_dict

    def discriminator_scores_plot(self, scores_dict, vdw_penalty_dict, layout):
        plot_color_dict = {}
        plot_color_dict['CSD'] = ('rgb(250,150,50)')  # test
        plot_color_dict['Generator'] = ('rgb(100,50,0)')  # test
        plot_color_dict['Gaussian'] = ('rgb(0,50,0)')  # fake csd
        plot_color_dict['Distorted'] = ('rgb(0,100,100)')  # fake distortion

        scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
        bandwidth1 = scores_range / 200

        bandwidth2 = 15 / 200
        viridis = px.colors.sequential.Viridis

        scores_labels = ['CSD', 'Gaussian', 'Distorted', 'Generator']
        fig = make_subplots(rows=2, cols=2, subplot_titles=('a)', 'b)', 'c)'),
                            specs=[[{}, {}], [{"colspan": 2}, None]], vertical_spacing=0.14)

        for i, label in enumerate(scores_labels):
            legend_label = label
            fig.add_trace(go.Violin(x=scores_dict[label], name=legend_label, line_color=plot_color_dict[label],
                                    side='positive', orientation='h', width=4,
                                    meanline_visible=True, bandwidth=bandwidth1, points=False),
                          row=1, col=1)
            fig.add_trace(go.Violin(x=-np.log10(vdw_penalty_dict[label] + 1e-3), name=legend_label, line_color=plot_color_dict[label],
                                    side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth2, points=False),
                          row=1, col=2)

        all_vdws = np.concatenate((vdw_penalty_dict['CSD'], vdw_penalty_dict['Gaussian'], vdw_penalty_dict['Distorted'], vdw_penalty_dict['Generator']))
        all_scores_i = np.concatenate((scores_dict['CSD'], scores_dict['Gaussian'], scores_dict['Distorted'], scores_dict['Generator']))

        rrange = np.logspace(3, 0, len(viridis))
        cscale = [[1 / rrange[i], viridis[i]] for i in range(len(rrange))]
        cscale[0][0] = 0

        fig.add_trace(go.Histogram2d(x=all_scores_i,
                                     y=-np.log10(all_vdws + 1e-3),
                                     showscale=False,
                                     nbinsy=50, nbinsx=200,
                                     colorscale=cscale,
                                     colorbar=dict(
                                         tick0=0,
                                         tickmode='array',
                                         tickvals=[0, 1000, 10000]
                                     )),
                      row=2, col=1)

        fig.update_layout(showlegend=False, yaxis_showgrid=True)
        fig.update_xaxes(title_text='Model Score', row=1, col=1)
        fig.update_xaxes(title_text='vdw Score', row=1, col=2)
        fig.update_xaxes(title_text='Model Score', row=2, col=1)
        fig.update_yaxes(title_text='vdw Score', row=2, col=1)

        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

        fig.layout.annotations[0].update(x=0.025)
        fig.layout.annotations[1].update(x=0.575)

        fig.layout.margin = layout.margin
        wandb.log({'Discriminator Scores Analysis': fig})

        return None

    def plot_generator_loss_correlates(self, epoch_stats_dict, generator_losses, layout):
        correlates_dict = {}
        generator_losses['all'] = np.vstack([generator_losses[key] for key in generator_losses.keys()]).T.sum(1)
        loss_labels = list(generator_losses.keys())

        tracking_features = np.asarray(epoch_stats_dict['tracking features'])

        for i in range(self.config.dataDims['num tracking features']):  # not that interesting
            if (np.average(tracking_features[:, i] != 0) > 0.05):
                corr_dict = {loss_label: np.corrcoef(generator_losses[loss_label], tracking_features[:, i], rowvar=False)[0, 1] for loss_label in loss_labels}
                correlates_dict[self.config.dataDims['tracking features dict'][i]] = corr_dict

        sort_inds = np.argsort(np.asarray([(correlates_dict[key]['all']) for key in correlates_dict.keys()]))
        keys_list = list(correlates_dict.keys())
        sorted_correlates_dict = {keys_list[ind]: correlates_dict[keys_list[ind]] for ind in sort_inds}

        fig = go.Figure()
        for label in loss_labels:
            fig.add_trace(go.Bar(name=label,
                                 y=list(sorted_correlates_dict.keys()),
                                 x=[corr[label] for corr in sorted_correlates_dict.values()],
                                 textposition='auto',
                                 orientation='h',
                                 text=[corr[label] for corr in sorted_correlates_dict.values()],
                                 ))
        fig.update_layout(barmode='relative')
        fig.update_traces(texttemplate='%{text:.2f}')
        fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=10))

        fig.layout.margin = layout.margin

        wandb.log({'Generator Loss Correlates': fig})

    def plot_discriminator_score_correlates(self, epoch_stats_dict, layout):
        correlates_dict = {}
        real_scores = softmax_and_score(epoch_stats_dict['discriminator real score'])
        tracking_features = np.asarray(epoch_stats_dict['tracking features'])

        for i in range(self.config.dataDims['num tracking features']):  # not that interesting
            if (np.average(tracking_features[:, i] != 0) > 0.05):
                corr = np.corrcoef(real_scores, tracking_features[:, i], rowvar=False)[0, 1]
                if np.abs(corr) > 0.05:
                    correlates_dict[self.config.dataDims['tracking features dict'][i]] = corr

        sort_inds = np.argsort(np.asarray([(correlates_dict[key]) for key in correlates_dict.keys()]))
        keys_list = list(correlates_dict.keys())
        sorted_correlates_dict = {keys_list[ind]: correlates_dict[keys_list[ind]] for ind in sort_inds}

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=list(sorted_correlates_dict.keys()),
            x=[corr for corr in sorted_correlates_dict.values()],
            textposition='auto',
            orientation='h',
        ))
        fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=10))

        fig.layout.margin = layout.margin

        wandb.log({'Discriminator Score Correlates': fig})

    def reinitialize_models(self, generator, discriminator, conditioner, regressor):
        print('Reinitializing models and optimizer')
        if self.config.generator_path is None:
            generator.apply(weight_reset)
        if self.config.discriminator_path is None:
            discriminator.apply(weight_reset)
        if self.config.conditioner_path is None:
            conditioner.apply(weight_reset)
        if self.config.regressor_path is None:
            regressor.apply(weight_reset)

        return generator, discriminator, conditioner, regressor

    def reload_model_checkpoints(self, config):
        if config.generator_path is not None:
            generator_checkpoint = torch.load(config.generator_path)
            config.generator = Namespace(**generator_checkpoint['config'])  # overwrite the settings for the model

        if config.discriminator_path is not None:
            discriminator_checkpoint = torch.load(config.discriminator_path)
            config.discriminator = Namespace(**discriminator_checkpoint['config'])

        if config.conditioner_path is not None:
            conditioner_checkpoint = torch.load(config.conditioner_path)
            config.conditioner = Namespace(**conditioner_checkpoint['config'])  # overwrite the settings for the model

        if config.regressor_path is not None:
            regressor_checkpoint = torch.load(config.regressor_path)
            config.regressor = Namespace(**regressor_checkpoint['config'])  # overwrite the settings for the model

        return config

    def init_conditioner_classes(self):
        if self.config.conditioner.decoder_classes == 'minimal':
            self.config.conditioner_classes = {  # only a few substantial atom types
                'other': 1,
            }
        elif self.config.conditioner.decoder_classes == 'full':
            self.config.conditioner_classes = {  # only a few substantial atom types
                'other': 1, # boron or smaller
            }
            for i in range(2, self.config.max_atomic_number + 1):
                self.config.conditioner_classes[i] = i

        conditioner_classes_dict = {i: self.config.conditioner_classes['other'] for i in range(self.config.max_atomic_number)}
        for i, (key, value) in enumerate(self.config.conditioner_classes.items()):
            if key != 'other':
                conditioner_classes_dict[key] = self.config.conditioner_classes[key]  # assign all atoms to type other, except these specific ones
        self.config.conditioner_classes_dict = conditioner_classes_dict

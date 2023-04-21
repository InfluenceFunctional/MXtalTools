import glob
import os
import time

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime

import ase.io
import ase.data
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import rdkit.Chem
# import rdkit.Chem.AllChem
# import rdkit.Chem.Draw
import wandb
# from PIL import Image
from pyxtal import symmetry
from scipy.stats import linregress
from torch import backends
from torch_geometric.loader.dataloader import Collater
import tqdm

from crystal_building.coordinate_transformations import cell_vol
from crystal_building.utils import *
from crystal_building.builder import SupercellBuilder, update_sg_to_all_crystals, update_crystal_symmetry_elements
from dataset_management.manager import Miner
from dataset_management.utils import BuildDataset, get_dataloaders, update_dataloader_batch_size, \
    get_extra_test_loader
from models.autoencoder_model import molecule_autoencoder
from models.crystal_rdf import crystal_rdf
from models.discriminator_models import crystal_discriminator
from models.generator_models import crystal_generator
from models.utils import *
from models.regression_models import molecule_regressor
from models.base_models import independent_gaussian_model
from models.utils import compute_h_bond_score, get_vdw_penalty, cell_density_loss, compute_combo_score
from models.vdw_overlap import vdw_overlap
from reporting.online import cell_params_analysis, plotly_setup, cell_density_plot, all_losses_plot, report_conditioner_training, process_discriminator_outputs, discriminator_scores_plot, \
    plot_generator_loss_correlates, plot_discriminator_score_correlates, log_mini_csp_scores_distributions, sampling_telemetry_plot, cell_params_tracking_plot, sample_wise_analysis, log_best_mini_csp_samples
from sampling.MCMC_Sampling import mcmcSampler
from sampling.SampleOptimization import gradient_descent_sampling
from common.utils import *
from common.utils import update_gan_metrics
from sampling.utils import de_clean_samples, sample_clustering


# https://www.ruppweb.org/Xray/tutorial/enantio.htm non enantiogenic groups
# https://dictionary.iucr.org/Sohncke_groups#:~:text=Sohncke%20groups%20are%20the%20three,in%20the%20chiral%20space%20groups.


class Modeller:
    def __init__(self, config):
        self.config = config
        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        setup_outputs = self.setup()
        if self.config.skip_saving_and_loading:
            self.prep_dataset = setup_outputs
        else:
            self.prep_dataset = None

    def setup(self):
        """
        load some physical data
        setup working directory
        move to it
        :return:
        """

        self.atom_weights = {}
        # self.vdw_radii = {}
        for i in range(100):
            self.atom_weights[i] = ase.data.atomic_masses[i]
            # self.vdw_radii[i] = ase.data.vdw_radii[i]

        # hardcoded from rdkit - ase module is missing many radii
        self.vdw_radii = {0: 0.0, 1: 1.2, 2: 1.4, 3: 2.2, 4: 1.9, 5: 1.8, 6: 1.7, 7: 1.6, 8: 1.55, 9: 1.5, 10: 1.54, 11: 2.4, 12: 2.2, 13: 2.1, 14: 2.1, 15: 1.95, 16: 1.8, 17: 1.8, 18: 1.88, 19: 2.8, 20: 2.4, 21: 2.3, 22: 2.15, 23: 2.05,
                          24: 2.05, 25: 2.05,
                          26: 2.05, 27: 2.0, 28: 2.0, 29: 2.0, 30: 2.1, 31: 2.1, 32: 2.1, 33: 2.05, 34: 1.9, 35: 1.9, 36: 2.02, 37: 2.9, 38: 2.55, 39: 2.4, 40: 2.3, 41: 2.15, 42: 2.1, 43: 2.05, 44: 2.05, 45: 2.0, 46: 2.05, 47: 2.1, 48: 2.2,
                          49: 2.2,
                          50: 2.25, 51: 2.2, 52: 2.1, 53: 2.1, 54: 2.16, 55: 3.0, 56: 2.7, 57: 2.5, 58: 2.48, 59: 2.47, 60: 2.45, 61: 2.43, 62: 2.42, 63: 2.4, 64: 2.38, 65: 2.37, 66: 2.35, 67: 2.33, 68: 2.32, 69: 2.3, 70: 2.28, 71: 2.27,
                          72: 2.25, 73: 2.2,
                          74: 2.1, 75: 2.05, 76: 2.0, 77: 2.0, 78: 2.05, 79: 2.1, 80: 2.05, 81: 2.2, 82: 2.3, 83: 2.3, 84: 2.0, 85: 2.0, 86: 2.0, 87: 2.0, 88: 2.0, 89: 2.0, 90: 2.4, 91: 2.0, 92: 2.3, 93: 2.0, 94: 2.0, 95: 2.0, 96: 2.0,
                          97: 2.0, 98: 2.0,
                          99: 2.0}

        # generate symmetry info dict if we don't already have it
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
                self.sym_ops[i] = [general_position_syms[i].affine_matrix for i in range(
                    len(general_position_syms))]  # first 0 index is for general position, second index is
                # superfluous, third index is the symmetry operation
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

        # set space groups to be included and generated
        if self.config.generate_sgs is None:
            # generate samples in every space group in the asym dict (eventually, all sgs)
            self.config.generate_sgs = [self.space_groups[int(key)] for key in asym_unit_dict.keys()]


        if self.config.include_sgs is None:
            self.config.include_sgs = [self.space_groups[int(key)] for key in asym_unit_dict.keys()]

            # initialize fractional lattice vectors - should be exactly identical to what's in molecule_featurizer.py
        # not currently used as we are not computing the overlaps
        # supercell_scale = self.config.supercell_size  # t
        # n_cells = (2 * supercell_scale + 1) ** 3
        #
        # fractional_translations = np.zeros((n_cells, 3))  # initialize the translations in fractional coords
        # i = 0
        # for xx in range(-supercell_scale, supercell_scale + 1):
        #     for yy in range(-supercell_scale, supercell_scale + 1):
        #         for zz in range(-supercell_scale, supercell_scale + 1):
        #             fractional_translations[i] = np.array((xx, yy, zz))
        #             i += 1
        # self.lattice_vectors = torch.Tensor(fractional_translations[np.argsort(np.abs(fractional_translations).sum(1))][1:])  # leave out the 0,0,0 element
        # self.normed_lattice_vectors = self.lattice_vectors / torch.linalg.norm(self.lattice_vectors, axis=1)[:, None]
        self.lattice_vectors, self.normed_lattice_vectors = None, None
        '''
        prepare to load dataset
        '''
        miner = Miner(config=self.config, dataset_path=self.config.dataset_path, collect_chunks=False)

        if (self.config.run_num == 0) or (self.config.explicit_run_enumeration == True):  # if making a new workdir
            if self.config.run_num == 0:
                self.make_new_working_directory()
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

    def make_new_working_directory(self):  # make working directory
        """
        make a new working directory
        non-overlapping previous entries
        or with a preset numter
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

        generator, discriminator, conditioner, regressor = nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(
            1, 1)  # init dummy models
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
            discriminator, discriminator_optimizer = reload_model(discriminator, discriminator_optimizer,
                                                                  self.config.discriminator_path)
        if self.config.conditioner_path is not None and self.config.mode == 'autoencoder':
            conditioner, conditioner_optimizer = reload_model(conditioner, conditioner_optimizer,
                                                              self.config.conditioner_path)
        if self.config.regressor_path is not None and self.config.mode == 'regression':
            regressor, regressor_optimizer = reload_model(discriminator, regressor_optimizer,
                                                          self.config.regressor_path)

        if self.config.mode == 'gan' and self.config.conditioner_path is not None:
            checkpoint = torch.load(self.config.conditioner_path)
            if list(checkpoint['model_state_dict'])[0][
               0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(checkpoint['model_state_dict']):
                    checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

            conditioner_model_state = {key[12:]: value for key, value in checkpoint['model_state_dict'].items() if
                                       'conditioner' in key}
            generator.conditioner.load_state_dict(conditioner_model_state)

        if self.config.freeze_generator_conditioner:
            generator.conditioner.requires_grad_(False)
            generator_optimizer = init_optimizer(self.config.generator_optimizer, generator, freeze_params=True)

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
        """
        try larger batches until it crashes
        DEPRECATED #todo fix this, or delete it
        """
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
                _ = self.run_epoch(data_loader=train_loader, generator=generator, discriminator=discriminator,
                                   generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   update_gradients=True, record_stats=True, iteration_override=2, epoch=1)

                # if successful, increase the batch and try again
                batch_size = max(batch_size + 5, int(batch_size * increment))
                train_loader = update_dataloader_batch_size(train_loader, batch_size)
                test_loader = update_dataloader_batch_size(test_loader, batch_size)
                # train_loader, test_loader = get_data_loaders(dataset, config, override_batch_size=batch_size)

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
        """
        train and/or evaluate one or more models
        regressor
        autoencoder
        GAN (generator and/or discriminator)
        """
        with wandb.init(config=self.config,
                        project=self.config.wandb.project_name,
                        entity=self.config.wandb.username,
                        tags=[self.config.wandb.experiment_tag]):
            wandb.run.name = wandb.config.machine + '_' + str(self.config.mode) + '_' + str(wandb.config.run_num)  # overwrite procedurally generated run name with our run name
            wandb.run.save()
            # config = wandb.config # wandb configs don't support nested namespaces. look at the github thread to see if they eventually fix it

            dataset_builder = self.misc_pre_training_items()
            generator, discriminator, conditioner, regressor, \
                generator_optimizer, generator_schedulers, \
                discriminator_optimizer, discriminator_schedulers, \
                conditioner_optimizer, conditioner_schedulers, \
                regressor_optimizer, regressor_schedulers, \
                num_params = self.init_models()

            train_loader, test_loader = get_dataloaders(dataset_builder, self.config)
            self.config.current_batch_size = self.config.min_batch_size
            print("Training batch size set to {}".format(self.config.current_batch_size))
            del dataset_builder

            extra_test_loader = None  # data_loader for a secondary test set - currently unused
            if (self.config.extra_test_set_paths is not None) and self.config.extra_test_evaluation:
                extra_test_loader = get_extra_test_loader(self.config, self.config.extra_test_set_paths,
                                                          dataDims=self.config.dataDims,
                                                          pg_dict=self.point_groups, sg_dict=self.space_groups,
                                                          lattice_dict=self.lattice_type)

            generator, discriminator, conditioner, regressor = self.reinitialize_models(
                generator, discriminator, conditioner, regressor)
            wandb.watch((generator, discriminator, conditioner, regressor), log_graph=True, log_freq=100)
            wandb.log({"Model Num Parameters": np.sum(np.asarray(num_params)),
                       "Initial Batch Size": self.config.current_batch_size})

            # initialize some metrics
            metrics_dict = {}
            generator_err_tr, discriminator_err_tr, conditioner_err_tr, regressor_err_tr = 0, 0, 0, 0
            generator_err_te, discriminator_err_te, conditioner_err_te, regressor_err_te = 0, 0, 0, 0
            generator_tr_record, discriminator_tr_record, conditioner_tr_record, regressor_tr_record = [0], [0], [0], [0]
            generator_te_record, discriminator_te_record, conditioner_te_record, regressor_te_record = [0], [0], [0], [0]
            discriminator_hit_max_lr, generator_hit_max_lr, conditioner_hit_max_lr, regressor_hit_max_lr, \
                converged, epoch = \
                False, False, False, False, self.config.max_epochs == 0, 0

            # training loop
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
                        # train & compute loss
                        train_loss, train_loss_record, train_epoch_stats_dict, time_train = \
                            self.run_epoch(data_loader=train_loader,
                                           generator=generator, discriminator=discriminator,
                                           regressor=regressor, conditioner=conditioner,
                                           generator_optimizer=generator_optimizer,
                                           discriminator_optimizer=discriminator_optimizer,
                                           conditioner_optimizer=conditioner_optimizer,
                                           regressor_optimizer=regressor_optimizer,
                                           update_gradients=True, record_stats=True, epoch=epoch)

                        with torch.no_grad():
                            # compute test loss
                            test_loss, test_loss_record, test_epoch_stats_dict, time_test = \
                                self.run_epoch(data_loader=test_loader,
                                               generator=generator, discriminator=discriminator,
                                               regressor=regressor, conditioner=conditioner,
                                               generator_optimizer=generator_optimizer,
                                               discriminator_optimizer=discriminator_optimizer,
                                               conditioner_optimizer=conditioner_optimizer,
                                               regressor_optimizer=regressor_optimizer,
                                               update_gradients=False, record_stats=True, epoch=epoch)

                        print('epoch={}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, time_train, time_test))

                        # save losses
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

                        # update learning rates
                        discriminator_optimizer, discriminator_learning_rate, discriminator_hit_max_lr, \
                            generator_optimizer, generator_learning_rate, generator_hit_max_lr, \
                            conditioner_optimizer, conditioner_learning_rate, conditioner_hit_max_lr, \
                            regressor_optimizer, regressor_learning_rate, regressor_hit_max_lr = \
                            self.update_lr(discriminator_schedulers, discriminator_optimizer, discriminator_err_tr,
                                           discriminator_hit_max_lr,
                                           generator_schedulers, generator_optimizer, generator_err_tr,
                                           generator_hit_max_lr,
                                           conditioner_schedulers, conditioner_optimizer, conditioner_err_tr,
                                           conditioner_hit_max_lr,
                                           regressor_schedulers, regressor_optimizer, regressor_err_tr,
                                           regressor_hit_max_lr)

                        # save key metrics
                        metrics_dict = update_gan_metrics(epoch, metrics_dict,
                                                          discriminator_learning_rate, generator_learning_rate,
                                                          conditioner_learning_rate, regressor_learning_rate,
                                                          discriminator_err_tr, discriminator_err_te,
                                                          generator_err_tr, generator_err_te,
                                                          conditioner_err_tr, conditioner_err_te,
                                                          regressor_err_tr, regressor_err_te)

                        # log losses to wandb
                        self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                                          generator_tr_record, generator_te_record, discriminator_tr_record,
                                          discriminator_te_record,
                                          conditioner_tr_record, conditioner_te_record, regressor_tr_record,
                                          regressor_te_record)

                        # sometimes to detailed reporting
                        if (epoch % self.config.wandb.sample_reporting_frequency) == 0:
                            self.detailed_reporting(epoch, train_loader, train_epoch_stats_dict, test_epoch_stats_dict,
                                                    extra_test_dict=extra_test_epoch_stats_dict)

                        # sometimes test the generator on a mini CSP problem
                        if (self.config.mode == 'gan') and (epoch % self.config.wandb.mini_csp_frequency == 0) and \
                                any((self.config.train_generator_packing, self.config.train_generator_adversarially,
                                     self.config.train_generator_vdw, self.config.train_generator_combo,
                                     self.config.train_generator_h_bond)):
                            self.mini_csp(extra_test_loader if extra_test_loader is not None else test_loader, generator, discriminator)

                        # save checkpoints
                        self.model_checkpointing(epoch, self.config, discriminator, generator, conditioner, regressor,
                                                 discriminator_optimizer, generator_optimizer, conditioner_optimizer,
                                                 regressor_optimizer,
                                                 generator_err_te, discriminator_err_te, conditioner_err_te,
                                                 regressor_err_te, metrics_dict)

                        # check convergence status
                        generator_converged, discriminator_converged, conditioner_converged, regressor_converged = \
                            self.check_model_convergence(metrics_dict)

                        if (generator_converged and discriminator_converged and conditioner_converged and regressor_converged) and (
                                epoch > self.config.history + 2):
                            print('Training has converged!')
                            break

                        train_loader, test_loader, extra_test_loader = \
                            self.update_batch_size(train_loader, test_loader, extra_test_loader)

                    except RuntimeError as e:  # if we do hit OOM, slash the batch size
                        if "CUDA out of memory" in str(e):
                            train_loader, test_loader = self.slash_batch(train_loader, test_loader)
                        else:
                            raise e
                    epoch += 1

                    if self.config.device.lower() == 'cuda':
                        torch.cuda.empty_cache()  # clear GPU, probably unnecessary

    def run_epoch(self, data_loader=None, generator=None, discriminator=None, regressor=None, conditioner=None,
                  generator_optimizer=None, discriminator_optimizer=None, conditioner_optimizer=None, regressor_optimizer=None,
                  update_gradients=True, iteration_override=None, record_stats=False, epoch=None):

        if self.config.mode == 'gan':
            return self.gan_epoch(data_loader, generator, discriminator, generator_optimizer, discriminator_optimizer,
                                  update_gradients,
                                  iteration_override, record_stats, epoch)

        elif self.config.mode == 'regression':
            return self.regression_epoch(data_loader, regressor, regressor_optimizer, update_gradients,
                                         iteration_override, record_stats)

        elif self.config.mode == 'autoencoder':
            return self.conditioner_epoch(data_loader, conditioner, conditioner_optimizer, update_gradients,
                                          iteration_override, record_stats, epoch)

    def regression_epoch(self, data_loader, regressor, regressor_optimizer, update_gradients=True,
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
            if self.config.regressor.positional_noise > 0:
                data.pos += torch.randn_like(data.pos) * self.config.regressor.positional_noise

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

    def conditioner_epoch(self, data_loader, conditioner, conditioner_optimizer, update_gradients=True,
                          iteration_override=None, record_stats=False, epoch=None):
        t0 = time.time()

        if update_gradients:
            conditioner.train(True)
        else:
            conditioner.eval()

        loss = []
        loss_record = []
        epoch_stats_dict = {}

        for i, data in enumerate(tqdm.tqdm(data_loader, miniters=int(len(data_loader) / 10))):
            loss, loss_record, epoch_stats_dict = \
                self.conditioner_step(conditioner, epoch_stats_dict, data,
                                      conditioner_optimizer, i, epoch,
                                      update_gradients, loss, loss_record)

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

    def gan_epoch(self, data_loader=None, generator=None, discriminator=None, generator_optimizer=None,
                  discriminator_optimizer=None, update_gradients=True,
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

        rand_batch_ind = np.random.randint(0, len(data_loader))

        for i, data in enumerate(tqdm.tqdm(data_loader, miniters=int(len(data_loader) / 10))):
            '''
            train discriminator
            '''
            skip_discriminator_step = False  # (i % self.config.discriminator_optimizer.training_period) != 0  # only train the discriminator every XX steps, assuming n_steps per epoch is much larger than training period
            if i > 0 and self.config.train_discriminator_adversarially:
                avg_generator_score = np_softmax(np.stack(epoch_stats_dict['discriminator fake score'])[np.argwhere(np.asarray(epoch_stats_dict['generator sample source']) == 0)[:, 0]])[:, 1].mean()
                if avg_generator_score < 0.5:
                    skip_discriminator_step = True

            discriminator_err, discriminator_loss_record, epoch_stats_dict = \
                self.discriminator_step(discriminator, generator, epoch_stats_dict, data,
                                        discriminator_optimizer, i, update_gradients, discriminator_err,
                                        discriminator_loss_record,
                                        skip_step=skip_discriminator_step)
            '''
            train_generator
            '''
            generator_err, generator_loss_record, epoch_stats_dict = \
                self.generator_step(discriminator, generator, epoch_stats_dict, data,
                                    generator_optimizer, i, update_gradients, generator_err, generator_loss_record,
                                    rand_batch_ind, last_batch=i == (len(data_loader) - 1))
            '''
            record some stats
            '''
            if (len(epoch_stats_dict[
                        'generated cell parameters']) < i) and record_stats and not self.config.train_generator_conditioner:  # make some samples for analysis if we have none so far from this step
                generated_samples = generator(len(data.y), z=None, conditions=data.to(self.config.device))
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
                        epoch_stats_dict[key] = np.asarray(feature)

            return [np.mean(discriminator_err), np.mean(generator_err)], [discriminator_loss_record, generator_loss_record], epoch_stats_dict, total_time
        else:
            return [np.mean(discriminator_err), np.mean(generator_err)], [discriminator_loss_record, generator_loss_record], total_time

    def discriminator_evaluation(self, data_loader=None, discriminator=None, iteration_override=None):  # todo revise to include generator
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
    def adversarial_score(discriminator, data):
        output, extra_outputs = discriminator(data.clone(), return_dists=True)  # reshape output from flat filters to channels * filters per channel
        return output, extra_outputs['dists dict']

    @staticmethod
    def log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
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

        # log conditioner losses
        if conditioner_tr_record is not None:
            hist = np.histogram(conditioner_tr_record, bins=256,
                                range=(np.amin(conditioner_tr_record), np.quantile(conditioner_tr_record, 0.9)))
            wandb.log({"Generator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        if conditioner_te_record is not None:
            hist = np.histogram(conditioner_te_record, bins=256,
                                range=(np.amin(conditioner_te_record), np.quantile(conditioner_te_record, 0.9)))
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
                    train_epoch_stats_dict[key] = np.concatenate(train_epoch_stats_dict[key])
                if ('loss' in key) and (train_epoch_stats_dict[key] is not None):
                    special_losses['Train ' + key] = np.average(np.concatenate(train_epoch_stats_dict[key]))
                if ('score' in key) and (train_epoch_stats_dict[key] is not None):
                    score = softmax_and_score(np.concatenate(train_epoch_stats_dict[key]))
                    special_losses['Train ' + key] = np.average(score)

        if test_epoch_stats_dict is not None:
            for key in test_epoch_stats_dict.keys():
                if ('loss' in key) and (test_epoch_stats_dict[key] is not None):
                    special_losses['Test ' + key] = np.average(np.concatenate(test_epoch_stats_dict[key]))
                if ('score' in key) and (test_epoch_stats_dict[key] is not None):
                    score = softmax_and_score(np.concatenate(test_epoch_stats_dict[key]))
                    special_losses['Test ' + key] = np.average(score)

        wandb.log(special_losses)

    def detailed_reporting(self, epoch, train_loader, train_epoch_stats_dict, test_epoch_stats_dict,
                           extra_test_dict=None):
        """
        Do analysis and upload results to w&b
        """
        if (test_epoch_stats_dict is not None) and self.config.mode == 'gan':
            if test_epoch_stats_dict['generated cell parameters'] is not None:
                cell_params_analysis(self.config, wandb, train_loader, test_epoch_stats_dict)

            if self.config.train_generator_packing or self.config.train_generator_vdw or self.config.train_generator_adversarially or self.config.train_generator_combo:
                self.cell_generation_analysis(test_epoch_stats_dict)

            if self.config.train_discriminator_on_distorted or self.config.train_discriminator_on_randn or self.config.train_discriminator_adversarially:
                self.discriminator_analysis(test_epoch_stats_dict)

        if (test_epoch_stats_dict is not None) and self.config.mode == 'autoencoder':
            report_conditioner_training(self.config, wandb, test_epoch_stats_dict)

        elif self.config.mode == 'regression':
            self.log_regression_accuracy(train_epoch_stats_dict, test_epoch_stats_dict)

        if (extra_test_dict is not None) and (epoch % self.config.extra_test_period == 0):
            pass  # do reporting on an extra dataset # todo redevelop evaluation or delete this

        return None

    def discriminator_step(self, discriminator, generator, epoch_stats_dict, data, discriminator_optimizer, i,
                           update_gradients, discriminator_err, discriminator_loss_record, skip_step):

        if any((self.config.train_discriminator_adversarially, self.config.train_discriminator_on_distorted, self.config.train_discriminator_on_randn)):
            score_on_real, score_on_fake, generated_samples, \
                real_dist_dict, fake_dist_dict, real_vdw_score, fake_vdw_score, \
                real_packing_coeffs, fake_packing_coeffs, generated_samples_i \
                = self.train_discriminator(discriminator, generator, data, i, epoch_stats_dict)

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
                       generator_err, generator_loss_record, rand_batch_ind, last_batch):
        if any((self.config.train_generator_packing, self.config.train_generator_adversarially,
                self.config.train_generator_vdw, self.config.train_generator_combo,
                self.config.train_generator_h_bond)):
            discriminator_raw_output, generated_samples, packing_loss, packing_prediction, packing_target, \
                vdw_loss, generated_dist_dict, supercell_examples, similarity_penalty, h_bond_score, combo_score = \
                self.train_generator(generator, discriminator, data, i)

            generator_losses, epoch_stats_dict = self.aggregate_generator_losses(
                epoch_stats_dict, packing_loss, discriminator_raw_output,
                vdw_loss, similarity_penalty, packing_prediction, packing_target, h_bond_score, combo_score)

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

    def conditioner_step(self, conditioner, epoch_stats_dict, data, conditioner_optimizer, i, epoch, update_gradients,
                         loss, loss_record):
        # if (epoch < 50) and (i%2 == 0):
        #     rand_sample = True
        # else:
        #     rand_sample = False
        rand_sample = False  # only real data

        packing_loss, reconstruction_loss, target_sample, prediction_sample, \
            packing_true, packing_pred, particle_dist_true, particle_dist_pred, \
            real_sample, classwise_loss = \
            self.train_conditioner(conditioner, data, rand_sample)

        stats_keys = ['conditioner packing target', 'conditioner packing prediction',
                      'conditioner particle prediction', 'conditioner particle true',
                      'reconstruction loss', 'conditioner packing loss', 'conditioner classwise loss',
                      'conditioner classwise bce']
        stats_values = [packing_true, packing_pred,
                        particle_dist_pred, particle_dist_true,
                        reconstruction_loss.cpu().detach().numpy(), packing_loss.cpu().detach().numpy(),
                        classwise_loss, classwise_loss]

        if i == 0:
            stats_keys += ['target_sample', 'prediction_sample']
            stats_values += [target_sample, prediction_sample]

        epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values)

        if not real_sample:
            conditioning_losses = reconstruction_loss  # + packing_loss
        else:
            conditioning_losses = reconstruction_loss + packing_loss

        conditioner_loss = (conditioning_losses).mean()
        loss.append(conditioner_loss.data.cpu().detach().numpy())  # average loss

        if update_gradients:
            conditioner_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
            torch.nn.utils.clip_grad_norm_(conditioner.parameters(),
                                           self.config.gradient_norm_clip)  # gradient clipping
            conditioner_loss.backward()  # back-propagation
            conditioner_optimizer.step()  # update parameters

        loss_record.extend(conditioning_losses.cpu().detach().numpy())  # loss distribution

        return loss, loss_record, epoch_stats_dict

    def train_discriminator(self, discriminator, generator, real_data, i, epoch_stats_dict):
        # generate fakes & create supercell data
        real_supercell_data = self.supercell_builder.real_cell_to_supercell(real_data, self.config)

        generated_samples_i, epoch_stats_dict, negative_type = \
            self.generate_discriminator_negatives(epoch_stats_dict, real_data, generator, i)

        fake_supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
            real_data, generated_samples_i, self.config.supercell_size,
            self.config.discriminator.graph_convolution_cutoff,
            align_molecules=(negative_type != 'generated'),
            rescale_asymmetric_unit=(negative_type != 'distorted'),
            skip_cell_cleaning=False,
            standardized_sample=True,
            target_handedness=real_data.asym_unit_handedness,
        )

        if self.config.device.lower() == 'cuda':  # redundant
            real_supercell_data = real_supercell_data.cuda()
            fake_supercell_data = fake_supercell_data.cuda()

        if self.config.test_mode or self.config.anomaly_detection:
            assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"
            assert torch.sum(torch.isnan(fake_supercell_data.x)) == 0, "NaN in training input"

        if self.config.discriminator.positional_noise > 0:
            real_supercell_data.pos += \
                torch.randn_like(real_supercell_data.pos) * self.config.discriminator.positional_noise
            fake_supercell_data.pos += \
                torch.randn_like(fake_supercell_data.pos) * self.config.discriminator.positional_noise

        score_on_real, real_distances_dict = self.adversarial_score(discriminator, real_supercell_data)
        score_on_fake, fake_pairwise_distances_dict = self.adversarial_score(discriminator, fake_supercell_data)

        # todo assign z values properly in cell construction
        real_packing_coeffs = compute_packing_coefficient(cell_params=real_supercell_data.cell_params,
                                                          mol_volumes=real_supercell_data.tracking[:, self.mol_volume_ind],
                                                          z_values=torch.tensor([len(real_supercell_data.ref_cell_pos[ii]) for ii in range(real_supercell_data.num_graphs)],
                                                                                dtype=torch.float64, device=real_supercell_data.x.device))
        fake_packing_coeffs = compute_packing_coefficient(cell_params=fake_supercell_data.cell_params,
                                                          mol_volumes=fake_supercell_data.tracking[:, self.mol_volume_ind],
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
            mode = self.config.generator.canonical_conformer_orientation

        if mode == 'standardized':
            data = align_crystaldata_to_principal_axes(data, handedness=data.asym_unit_handedness)
            # data.asym_unit_handedness = torch.ones_like(data.asym_unit_handedness)

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

                data.asym_unit_handedness = torch.ones_like(data.asym_unit_handedness)

        return data

    def get_generator_samples(self, mol_data, generator, alignment_override=None):
        """
        @param mol_data: CrystalData object containing information on the starting conformer
        @param generator:
        @return:
        """
        # conformer orentation setting
        mol_data = self.set_molecule_alignment(mol_data, mode_override=alignment_override)

        # noise injection
        if self.config.generator.positional_noise > 0:
            mol_data.pos += torch.randn_like(mol_data.pos) * self.config.generator.positional_noise

        # update symmetry information
        if self.config.generate_sgs is not None:
            mol_data = update_crystal_symmetry_elements(mol_data, self.config.generate_sgs, self.config.dataDims,
                                                        self.sym_info, randomize_sgs=True)

        # generate the samples
        [[generated_samples, latent], prior, condition] = generator.forward(
            n_samples=mol_data.num_graphs, conditions=mol_data.to(self.config.device).clone(),
            return_latent=True, return_condition=True, return_prior=True)

        return generated_samples, prior

    def train_generator(self, generator, discriminator, data, i):
        """
        train the generator
        """

        if any((self.config.train_generator_packing,
                self.config.train_generator_vdw,
                self.config.train_generator_combo,
                self.config.train_generator_adversarially,
                self.config.train_generator_h_bond)):
            '''
            build supercells
            '''
            generated_samples, prior = self.get_generator_samples(data, generator)

            supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
                data, generated_samples, self.config.supercell_size,
                self.config.discriminator.graph_convolution_cutoff,
                align_molecules=False,  # molecules are either random on purpose, or pre-aligned with set handedness
            )

            similarity_penalty = self.compute_similarity_penalty(generated_samples, prior)
            discriminator_raw_output, dist_dict = self.score_adversarially(supercell_data, discriminator)
            h_bond_score = compute_h_bond_score(self.config.feature_richness, self.atom_acceptor_ind, self.atom_donor_ind, self.num_acceptors_ind, self.num_donors_ind, supercell_data)
            vdw_penalty, normed_vdw_penalty = get_vdw_penalty(self.vdw_radii,
                                                              dist_dict=dist_dict,
                                                              num_graphs=data.num_graphs,
                                                              mol_sizes=data.mol_size)
            packing_loss, packing_prediction, packing_target, = cell_density_loss(
                self.config.packing_loss_rescaling,
                self.config.dataDims['tracking features dict'].index('crystal packing coefficient'),
                self.mol_volume_ind,
                self.config.dataDims['target mean'], self.config.dataDims['target std'],
                supercell_data, generated_samples, precomputed_volumes=generated_cell_volumes)
            combo_score = compute_combo_score(packing_prediction, vdw_penalty, discriminator_raw_output)

            '''  # visualize contributions
            fig = go.Figure()
            fig.add_trace(go.Bar(name='packing',x = np.arange(len(packing_range_loss)), y=packing_range_loss.cpu().detach().numpy()))
            fig.add_trace(go.Bar(name='vdw',x = np.arange(len(packing_range_loss)), y=vdw_range_loss.cpu().detach().numpy()))
            fig.add_trace(go.Bar(name='discrim',x = np.arange(len(packing_range_loss)), y=discriminator_loss.cpu().detach().numpy()))
            fig.update_layout(showlegend=True)
            fig.update_yaxes(type="log")
            fig.show()
            '''

            return discriminator_raw_output, generated_samples.cpu().detach().numpy(), \
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
            particle_coords = torch.rand(len(data.x), 3, device=data.pos.device,
                                         dtype=data.pos.dtype) * 2 - 1  # particle positions with mean of avg_num_particles_per_sample particles per sample
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
            override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index(
                self.config.generate_sgs) + 1  # indexing from 0
            sym_ops_list = [
                torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(data.x.device) for i
                in range(data.num_graphs)]
            data = update_sg_to_all_crystals(self.config.generate_sgs, self.config.dataDims, data,
                                             self.supercell_builder.symmetries_dict,
                                             sym_ops_list)  # todo update the way we handle this

        point_cloud_prediction, packing_prediction = conditioner(data.clone())

        n_target_bins = int((
                                self.config.max_molecule_radius) * 2 / self.config.conditioner.decoder_resolution) + 1  # make up for odd in stride
        _, n_target_bins = get_strides(n_target_bins,
                                       init_size=self.config.conditioner.init_decoder_size)  # automatically find the right number of strides within 4-5 steps (minimizes overall stack depth)
        batch_size = len(point_cloud_prediction)
        buckets = torch.bucketize(data.pos,
                                  torch.linspace(-self.config.max_molecule_radius, self.config.max_molecule_radius,
                                                 n_target_bins - 1, device='cuda'))
        target = torch.zeros((batch_size, n_target_bins, n_target_bins, n_target_bins), dtype=torch.long,
                             device=point_cloud_prediction.device)
        for ii in range(batch_size):
            target[ii, buckets[data.batch == ii, 0], buckets[data.batch == ii, 1], buckets[data.batch == ii, 2]] = \
                data.x[data.batch == ii, 0].long()  # torch.clip(data.x[data.batch == ii,0],max=1).long()

        for ind, (key, value) in enumerate(self.config.conditioner_classes.items()):
            target[target == key] = -value  # negative, so this next step works quickly
        target[target > 0] = 1  # setting for 'other'
        target = -target
        target[target == -1] = 1

        packing_loss = F.smooth_l1_loss(packing_prediction[:, 0], data.y.float(), reduction='none')
        reconstruction_loss = F.cross_entropy(point_cloud_prediction, target, reduction='none').mean([-3, -2, -1]) / (
                torch.sum(target > 0) / len(target.flatten()))

        target_one_hot = F.one_hot(target.detach(), num_classes=len(self.config.conditioner_classes) + 1).permute(0, 4,
                                                                                                                  1, 2,
                                                                                                                  3).flatten(
            2, 4).float()
        prediction_flattened = F.softmax(point_cloud_prediction.detach().flatten(2, 4), dim=1).float()
        classwise_loss = F.binary_cross_entropy_with_logits(prediction_flattened, target_one_hot,
                                                            reduction='none').mean((0, 2))

        return packing_loss, reconstruction_loss, \
            target[0:8].cpu().detach().numpy(), point_cloud_prediction[0:8].cpu().detach().numpy(), \
            data.y.cpu().detach().numpy(), packing_prediction.cpu().detach().numpy(), \
            torch.mean(target_one_hot, dim=(0, 2)).cpu().detach().numpy(), \
            torch.mean(prediction_flattened, dim=(0, 2)).cpu().detach().numpy(), \
            sample_real, classwise_loss.cpu().detach().numpy()

    def regression_loss(self, generator, data):
        predictions = generator(data.to(generator.model.device))[:, 0]
        targets = data.y
        return F.smooth_l1_loss(predictions, targets, reduction='none'), predictions, targets

    def misc_pre_training_items(self):
        dataset_builder = BuildDataset(self.config, pg_dict=self.point_groups,
                                       sg_dict=self.space_groups,
                                       lattice_dict=self.lattice_type,
                                       premade_dataset=self.prep_dataset)

        del self.prep_dataset  # we don't actually want this huge thing floating around
        self.config.dataDims = dataset_builder.get_dimension()

        self.n_generators = sum((self.config.train_discriminator_on_randn, self.config.train_discriminator_on_distorted, self.config.train_discriminator_adversarially))
        self.generator_ind_list = []
        if self.config.train_discriminator_adversarially:
            self.generator_ind_list.append(1)
        if self.config.train_discriminator_on_randn:
            self.generator_ind_list.append(2)
        if self.config.train_discriminator_on_distorted:
            self.generator_ind_list.append(3)
        '''
        init supercell builder
        '''
        self.supercell_builder = SupercellBuilder(self.sym_info, self.config.dataDims)

        '''
        set tracking feature indices & property dicts we will use later
        '''
        self.mol_volume_ind = self.config.dataDims['tracking features dict'].index('molecule volume')
        self.crystal_packing_ind = self.config.dataDims['tracking features dict'].index('crystal packing coefficient')
        self.crystal_density_ind = self.config.dataDims['tracking features dict'].index('crystal calculated density')
        self.mol_size_ind = self.config.dataDims['tracking features dict'].index('molecule num atoms')
        self.sg_feature_ind_dict = {thing[14:]: ind + self.config.dataDims['num atomwise features'] for ind, thing in
                                    enumerate(self.config.dataDims['molecule features']) if 'sg is' in thing}
        self.crysys_ind_dict = {thing[18:]: ind + self.config.dataDims['num atomwise features'] for ind, thing in
                                enumerate(self.config.dataDims['molecule features']) if 'crystal system is' in thing}

        if self.config.feature_richness == 'full':
            self.num_acceptors_ind = self.config.dataDims['tracking features dict'].index('molecule num acceptors')
            self.num_donors_ind = self.config.dataDims['tracking features dict'].index('molecule num donors')
            self.atom_acceptor_ind = self.config.dataDims['atom features'].index('atom is H bond acceptor')
            self.atom_donor_ind = self.config.dataDims['atom features'].index('atom is H bond donor')

        '''
        add symmetry element indices to symmetry dict
        '''
        # todo build separate
        self.sym_info['sg_feature_ind_dict'] = self.sg_feature_ind_dict  # SG indices in input features
        self.sym_info['crysys_ind_dict'] = self.crysys_ind_dict  # crysys indices in input features
        self.sym_info['crystal_z_value_ind'] = self.config.dataDims['num atomwise features'] + self.config.dataDims['molecule features'].index('crystal z value')  # Z value index in input features

        ''' 
        init gaussian generator for cell parameter sampling
        we don't always use it but it's very cheap so just do it every time
        '''
        self.randn_generator = independent_gaussian_model(input_dim=self.config.dataDims['num lattice features'],
                                                          means=self.config.dataDims['lattice means'],
                                                          stds=self.config.dataDims['lattice stds'],
                                                          normed_length_means=self.config.dataDims[
                                                              'lattice normed length means'],
                                                          normed_length_stds=self.config.dataDims[
                                                              'lattice normed length stds'],
                                                          cov_mat=self.config.dataDims['lattice cov mat'])

        self.epoch = 0

        return dataset_builder

    def sampling_prep(self):
        dataset_builder = self.misc_pre_training_items()
        del dataset_builder
        generator, discriminator, generator_optimizer, generator_schedulers, \
            discriminator_optimizer, discriminator_schedulers, params1, params2 \
            = self.init_models()

        self.config.current_batch_size = self.config.min_batch_size

        extra_test_set_path = self.config.extra_test_set_paths
        extra_test_loader = get_extra_test_loader(self.config, extra_test_set_path, dataDims=self.config.dataDims,
                                                  pg_dict=self.point_groups, sg_dict=self.space_groups,
                                                  lattice_dict=self.lattice_type)

        self.randn_generator = independent_gaussian_model(input_dim=self.config.dataDims['num lattice features'],
                                                          means=self.config.dataDims['lattice means'],
                                                          stds=self.config.dataDims['lattice stds'],
                                                          normed_length_means=self.config.dataDims[
                                                              'lattice normed length means'],
                                                          normed_length_stds=self.config.dataDims[
                                                              'lattice normed length stds'],
                                                          cov_mat=self.config.dataDims['lattice cov mat'])

        # blind_test_identifiers = [
        #     'OBEQUJ', 'OBEQOD','NACJAF'] # targets XVI, XVII, XXII
        #
        # single_mol_data = extra_test_loader.dataset[extra_test_loader.csd_identifier.index(blind_test_identifiers[-1])]

        return extra_test_loader, generator, discriminator, \
            generator_optimizer, generator_schedulers, discriminator_optimizer, discriminator_schedulers, \
            params1, params2

    def model_sampling(self):  # todo combine with mini-CSP module
        """ DEPRECATED
        Stun MC annealing on a pretrained discriminator / generator
        """
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username,
                        tags=[self.config.wandb.experiment_tag]):
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
            single_mol_data = self.set_molecule_alignment(single_mol_data,
                                                          mode_override='random')  # take all the same conformers for one run
            override_sg_ind = list(self.supercell_builder.symmetries_dict['space_groups'].values()).index('P-1') + 1
            sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(
                single_mol_data.x.device) for i in range(single_mol_data.num_graphs)]
            single_mol_data = update_sg_to_all_crystals('P-1', self.supercell_builder.dataDims, single_mol_data,
                                                        self.supercell_builder.symmetries_dict, sym_ops_list)

            smc_sampling_dict = smc_sampler(discriminator, self.supercell_builder,
                                            single_mol_data.clone().cuda(), None,
                                            self.config.sample_steps)

            '''
            reporting
            '''

            sampling_telemetry_plot(self.config, wandb, smc_sampling_dict)
            cell_params_tracking_plot(wandb, self.supercell_builder, self.layout, self.config, smc_sampling_dict, collater, extra_test_loader)
            best_smc_samples, best_smc_samples_scores, best_smc_cells = sample_clustering(self.supercell_builder, self.config, smc_sampling_dict,
                                                                                          collater,
                                                                                          extra_test_loader,
                                                                                          discriminator)
            # destandardize samples
            unclean_best_samples = de_clean_samples(self.supercell_builder, best_smc_samples, best_smc_cells.sg_ind)
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
            sampling_telemetry_plot(self.config, wandb, gd_sampling_dict)
            cell_params_tracking_plot(wandb, self.supercell_builder, self.layout, self.config, gd_sampling_dict, collater, extra_test_loader)
            best_gd_samples, best_gd_samples_scores, best_gd_cells = sample_clustering(self.supercell_builder, self.config, gd_sampling_dict, collater,
                                                                                       extra_test_loader,
                                                                                       discriminator)

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

    def generate_discriminator_negatives(self, epoch_stats_dict, real_data, generator, i):
        """
        use one of the available cell generation tools to sample cell parameters, to be fed to the discriminator
        @param epoch_stats_dict:
        @param real_data:
        @param generator:
        @param i:
        @return:
        """

        gen_randint = np.random.randint(0, self.n_generators, 1)
        generator_ind = self.generator_ind_list[int(gen_randint)]  # randomly select which generator to use from the available set

        if self.config.train_discriminator_adversarially:
            if generator_ind == 1:  # randomly sample which generator to use at each iteration
                negative_type = 'generator'
                generated_samples_i, _ = self.get_generator_samples(real_data, generator)
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',
                                                     np.zeros(len(generated_samples_i)), mode='extend')

        if self.config.train_discriminator_on_randn:
            if generator_ind == 2:
                negative_type = 'randn'
                generated_samples_i = self.randn_generator.forward(real_data.num_graphs, real_data).to(self.config.device)
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',
                                                     np.ones(len(generated_samples_i)), mode='extend')

        if self.config.train_discriminator_on_distorted:
            if generator_ind == 3:
                negative_type = 'distorted'
                lattice_means = torch.tensor(self.config.dataDims['lattice means'], device=real_data.cell_params.device)
                lattice_stds = torch.tensor(self.config.dataDims['lattice stds'], device=real_data.cell_params.device)  # standardize

                generated_samples_ii = (real_data.cell_params - lattice_means) / lattice_stds

                if i % 2 == 0:  # alternate between random distortions and specifically diffuse cells
                    if self.config.sample_distortion_magnitude == -1:
                        distortion = torch.randn_like(generated_samples_ii) * torch.logspace(-.5, 0.5, len(generated_samples_ii)).to(
                            generated_samples_ii.device)[:, None]  # wider range for evaluation mode
                    else:
                        distortion = torch.randn_like(generated_samples_ii) * self.config.sample_distortion_magnitude
                else:
                    # add a random fraction of the original cell length - make the cell larger
                    distortion = torch.zeros_like(generated_samples_ii)
                    distortion[:, 0:3] = torch.randn_like(distortion[:, 0:3]).abs()

                generated_samples_i = (generated_samples_ii + distortion).to(self.config.device)  # add jitter and return in standardized basis
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'generator sample source',
                                                     np.ones(len(generated_samples_i)) * 2, mode='extend')
                epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'distortion level',
                                                     torch.linalg.norm(distortion, axis=-1).cpu().detach().numpy(),
                                                     mode='extend')

        return generated_samples_i.float(), epoch_stats_dict, negative_type

    def log_regression_accuracy(self, train_epoch_stats_dict, test_epoch_stats_dict):
        target_mean = self.config.dataDims['target mean']
        target_std = self.config.dataDims['target std']

        target = np.asarray(test_epoch_stats_dict['regressor packing target'])
        prediction = np.asarray(test_epoch_stats_dict['regressor packing prediction'])
        orig_target = target * target_std + target_mean
        orig_prediction = prediction * target_std + target_mean

        volume_ind = self.config.dataDims['tracking features dict'].index('molecule volume')
        mass_ind = self.config.dataDims['tracking features dict'].index('molecule mass')
        molwise_density = test_epoch_stats_dict['tracking features'][:, mass_ind] / test_epoch_stats_dict[
                                                                                        'tracking features'][:,
                                                                                    volume_ind]
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
            xline = np.linspace(max(min(orig_target), min(orig_prediction)),
                                min(max(orig_target), max(orig_prediction)), 10)
            fig = go.Figure()
            fig.add_trace(go.Histogram2dContour(x=orig_target, y=orig_prediction, ncontours=50, nbinsx=40, nbinsy=40,
                                                showlegend=True))
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

            xline = np.linspace(max(min(target_density), min(predicted_density)),
                                min(max(target_density), max(predicted_density)), 10)
            fig = go.Figure()
            fig.add_trace(
                go.Histogram2dContour(x=target_density, y=predicted_density, ncontours=50, nbinsx=40, nbinsy=40,
                                      showlegend=True))
            fig.update_traces(contours_coloring="fill")
            fig.update_traces(contours_showlines=False)
            fig.add_trace(
                go.Scattergl(x=target_density, y=predicted_density, mode='markers', showlegend=True, opacity=0.5))
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
                xline = np.linspace(max(min(train_orig_target), min(train_orig_prediction)),
                                    min(max(train_orig_target), max(train_orig_prediction)), 10)
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram2dContour(x=train_orig_target, y=train_orig_prediction, ncontours=50, nbinsx=40,
                                          nbinsy=40, showlegend=True))
                fig.update_traces(contours_coloring="fill")
                fig.update_traces(contours_showlines=False)
                fig.add_trace(
                    go.Scattergl(x=train_orig_target, y=train_orig_prediction, mode='markers', showlegend=True,
                                 opacity=0.5))
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
                generator_loss_correlations[i] = \
                    np.corrcoef(np.abs((orig_target - orig_prediction) / np.abs(orig_target)), tracking_features[:, i],
                                rowvar=False)[0, 1]

            generator_sort_inds = np.argsort(generator_loss_correlations)
            generator_loss_correlations = generator_loss_correlations[generator_sort_inds]

            fig = go.Figure(go.Bar(
                y=[self.config.dataDims['tracking features dict'][i] for i in
                   range(self.config.dataDims['num tracking features'])],
                x=[generator_loss_correlations[i] for i in range(self.config.dataDims['num tracking features'])],
                orientation='h',
            ))
            wandb.log({'Regressor Loss Correlates': fig})

        return None

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
            # from reporting.nov_22_regressor import nice_regression_plots
            # nice_regression_plots(self.config)
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
        conditioner_convergence = check_convergence(metrics_dict['conditioner test loss'], self.config.history,
                                                    self.config.conditioner_optimizer.convergence_eps)
        regressor_convergence = check_convergence(metrics_dict['regressor test loss'], self.config.history,
                                                  self.config.regressor_optimizer.convergence_eps)

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
                    save_checkpoint(epoch, discriminator, discriminator_optimizer, self.config.discriminator.__dict__,
                                    'best_discriminator_' + str(config.run_num))
                if np.average(generator_err_te) < np.amin(metrics_dict['generator test loss'][:-1]):
                    print("Saving generator checkpoint")
                    save_checkpoint(epoch, generator, generator_optimizer, self.config.generator.__dict__,
                                    'best_generator_' + str(config.run_num))
                if np.average(conditioner_err_te) < np.amin(metrics_dict['conditioner test loss'][:-1]):
                    print("Saving conditioner checkpoint")
                    save_checkpoint(epoch, conditioner, conditioner_optimizer, self.config.conditioner.__dict__,
                                    'best_conditioner_' + str(config.run_num))
                if np.average(regressor_err_te) < np.amin(metrics_dict['regressor test loss'][:-1]):
                    print("Saving regressor checkpoint")
                    save_checkpoint(epoch, regressor, regressor_optimizer, self.config.regressor.__dict__,
                                    'best_regressor_' + str(config.run_num))

        return None

    def update_lr(self, discriminator_schedulers, discriminator_optimizer, discriminator_err_tr,
                  discriminator_hit_max_lr,
                  generator_schedulers, generator_optimizer, generator_err_tr, generator_hit_max_lr,
                  conditioner_schedulers, conditioner_optimizer, conditioner_err_tr, conditioner_hit_max_lr,
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

        conditioner_optimizer, conditioner_lr = set_lr(conditioner_schedulers, conditioner_optimizer,
                                                       self.config.conditioner_optimizer.lr_schedule,
                                                       self.config.conditioner_optimizer.min_lr,
                                                       self.config.conditioner_optimizer.max_lr, conditioner_err_tr,
                                                       conditioner_hit_max_lr)
        conditioner_learning_rate = conditioner_optimizer.param_groups[0]['lr']
        if conditioner_learning_rate >= self.config.conditioner_optimizer.max_lr:
            conditioner_hit_max_lr = True

        regressor_optimizer, regressor_lr = set_lr(regressor_schedulers, regressor_optimizer,
                                                   self.config.regressor_optimizer.lr_schedule,
                                                   self.config.regressor_optimizer.min_lr,
                                                   self.config.regressor_optimizer.max_lr, regressor_err_tr,
                                                   regressor_hit_max_lr)
        regressor_learning_rate = regressor_optimizer.param_groups[0]['lr']
        if regressor_learning_rate >= self.config.regressor_optimizer.max_lr:
            regressor_hit_max_lr = True

        # print(f"Learning rates are d={discriminator_lr:.5f}, g={generator_lr:.5f}, a={conditioner_lr:.5f}, r={regressor_lr:.5f}")

        return discriminator_optimizer, discriminator_learning_rate, discriminator_hit_max_lr, \
            generator_optimizer, generator_learning_rate, generator_hit_max_lr, \
            conditioner_optimizer, conditioner_learning_rate, conditioner_hit_max_lr, \
            regressor_optimizer, regressor_learning_rate, regressor_hit_max_lr

    def post_run_evaluation(self, epoch, generator, discriminator, discriminator_optimizer, generator_optimizer,
                            metrics_dict, train_loader, test_loader, extra_test_loader):  # todo revise or delete
        """
        run post-training evaluation
        """
        # reload best test
        generator_path = f'../models/generator_{self.config.run_num}'
        discriminator_path = f'../models/discriminator_{self.config.run_num}'
        if os.path.exists(generator_path):
            generator_checkpoint = torch.load(generator_path)
            if list(generator_checkpoint['model_state_dict'])[0][
               0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(generator_checkpoint['model_state_dict']):
                    generator_checkpoint['model_state_dict'][i[7:]] = generator_checkpoint['model_state_dict'].pop(i)
            generator.load_state_dict(generator_checkpoint['model_state_dict'])

        if os.path.exists(discriminator_path):
            discriminator_checkpoint = torch.load(discriminator_path)
            if list(discriminator_checkpoint['model_state_dict'])[0][
               0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(discriminator_checkpoint['model_state_dict']):
                    discriminator_checkpoint['model_state_dict'][i[7:]] = discriminator_checkpoint[
                        'model_state_dict'].pop(i)
            discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])

        # rerun test inference
        with torch.no_grad():
            discriminator_err_te, discriminator_te_record, generator_err_te, generator_te_record, test_epoch_stats_dict, time_test = \
                self.run_epoch(data_loader=test_loader, generator=generator, discriminator=discriminator,
                               update_gradients=False, record_stats=True, epoch=epoch)  # compute loss on test set
            np.save(f'../{self.config.run_num}_test_epoch_stats_dict', test_epoch_stats_dict)

            if extra_test_loader is not None:
                extra_test_epoch_stats_dict, time_test_ex = \
                    self.discriminator_evaluation(self.config, data_loader=extra_test_loader,
                                                  discriminator=discriminator)  # compute loss on test set

                np.save(f'../{self.config.run_num}_extra_test_dict', extra_test_epoch_stats_dict)
            else:
                extra_test_epoch_stats_dict = None

        # save results
        metrics_dict = update_gan_metrics(
            epoch, metrics_dict,
            np.zeros(10), discriminator_err_te,
            np.zeros(10), generator_err_te,
            discriminator_optimizer.defaults['lr'], generator_optimizer.defaults['lr'])

        self.log_gan_loss(metrics_dict, None, test_epoch_stats_dict,
                          None, discriminator_te_record, None, generator_te_record)

        self.detailed_reporting(epoch, train_loader, None, test_epoch_stats_dict,
                                extra_test_dict=extra_test_epoch_stats_dict)

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

            # todo set the standardization for each space group individually (different stats and distances)

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
                if self.config.discriminator.positional_noise > 0:
                    supercell_data.pos += torch.randn_like(
                        supercell_data.pos) * self.config.discriminator.positional_noise

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
                                   similarity_penalty, packing_prediction, packing_target, h_bond_score, combo_score):
        generator_losses_list = []
        stats_keys, stats_values = [], []
        if packing_loss is not None:
            stats_keys += ['generator packing loss', 'generator packing prediction',
                           'generator packing target', 'generator packing mae']
            stats_values += [packing_loss.cpu().detach().numpy(), packing_prediction,
                             packing_target, np.abs(packing_prediction - packing_target) / packing_target]

            if self.config.train_generator_packing:
                generator_losses_list.append(packing_loss.float())

        if discriminator_raw_output is not None:
            softmax_adversarial_score = F.softmax(discriminator_raw_output, dim=1)[:, 1]  # modified minimax
            # adversarial_loss = -torch.log(softmax_adversarial_score)  # modified minimax
            # adversarial_loss = 10 - softmax_and_score(discriminator_raw_output)  # linearized score
            # adversarial_loss = 1-softmax_adversarial_score  # simply maximize P(real) (small gradients near 0 and 1)
            adversarial_loss = 1 - F.softmax(discriminator_raw_output / 5, dim=1)[:, 1]  # high temp smears out the function over a wider range
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
                if self.config.vdw_loss_rescaling == 'log':
                    vdw_loss_f = torch.log(1 + vdw_loss)  # soft rescaling to be gentler on outliers
                elif self.config.vdw_loss_rescaling is None:
                    vdw_loss_f = vdw_loss
                elif self.config.vdw_loss_rescaling == 'mse':
                    vdw_loss_f = vdw_loss ** 2

                generator_losses_list.append(vdw_loss_f)

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

        if combo_score is not None:
            stats_keys += ['generator combo loss']
            stats_values += [1 - combo_score.cpu().detach().numpy()]

            if self.config.train_generator_combo:
                generator_losses_list.append(-combo_score)

        generator_losses = torch.sum(torch.stack(generator_losses_list), dim=0)
        epoch_stats_dict = update_stats_dict(epoch_stats_dict, stats_keys, stats_values)

        return generator_losses, epoch_stats_dict

    def cell_generation_analysis(self, epoch_stats_dict):
        """
        do analysis and plotting for cell generator
        """
        layout = plotly_setup(self.config)
        self.log_cubic_defect(epoch_stats_dict)
        wandb.log({"Generated cell parameter variation": epoch_stats_dict['generated cell parameters'].std(0).mean()})
        generator_losses, average_losses_dict = self.process_generator_losses(epoch_stats_dict)
        wandb.log(average_losses_dict)

        cell_density_plot(self.config, wandb, epoch_stats_dict, layout)
        plot_generator_loss_correlates(self.config, wandb, epoch_stats_dict, generator_losses, layout)

        return None

    def log_cubic_defect(self, epoch_stats_dict):
        cleaned_samples = epoch_stats_dict['final generated cell parameters']
        cubic_distortion = np.abs(1 - np.nan_to_num(np.stack(
            [cell_vol(cleaned_samples[i, 0:3], cleaned_samples[i, 3:6]) / np.prod(cleaned_samples[i, 0:3], axis=-1) for
             i in range(len(cleaned_samples))])))
        wandb.log({'Avg generated cubic distortion': np.average(cubic_distortion)})
        hist = np.histogram(cubic_distortion, bins=256, range=(0, 1))
        wandb.log({"Generated cubic distortions": wandb.Histogram(np_histogram=hist, num_bins=256)})

    def process_generator_losses(self, epoch_stats_dict):
        generator_loss_keys = ['generator packing prediction', 'generator packing target', 'generator per mol vdw loss',
                               'generator adversarial loss', 'generator h bond loss', 'generator combo loss']
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
                        generator_losses['packing normed mae'] = np.abs(
                            generator_losses['packing prediction'] - generator_losses['packing target']) / \
                                                                 generator_losses['packing target']
                        del generator_losses['packing prediction'], generator_losses['packing target']
                else:
                    generator_losses[key[10:]] = None

        return generator_losses, {key: np.average(value) for i, (key, value) in enumerate(generator_losses.items()) if
                                  value is not None}

    @staticmethod
    def save_3d_structure_examples(wandb, generated_supercell_examples):
        num_samples = min(10, generated_supercell_examples.num_graphs)
        identifiers = [generated_supercell_examples.csd_identifier[i] for i in range(num_samples)]
        sgs = [str(int(generated_supercell_examples.sg_ind[i])) for i in range(num_samples)]

        crystals = [ase_mol_from_crystaldata(generated_supercell_examples, highlight_canonical_conformer=False,
                                             index=i, exclusion_level='distance', inclusion_distance=4)
                    for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

        for i in range(len(crystals)):
            ase.io.write(f'supercell_{i}.pdb', crystals[i])
            wandb.log({'Generated Supercells': wandb.Molecule(open(f"supercell_{i}.pdb"),
                                                              caption=identifiers[i] + ' ' + sgs[i])})

        mols = [ase_mol_from_crystaldata(generated_supercell_examples,
                                         index=i, exclusion_level='conformer')
                for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

        for i in range(len(mols)):
            ase.io.write(f'conformer_{i}.pdb', mols[i])
            wandb.log({'Single Conformers': wandb.Molecule(open(f"conformer_{i}.pdb"), caption=identifiers[i])})

        return None

    def discriminator_analysis(self, epoch_stats_dict):
        '''
        do analysis and plotting for cell discriminator

        -: scores distribution and vdw penalty by sample source
        -: loss correlates
        '''
        layout = plotly_setup(self.config)

        scores_dict, vdw_penalty_dict, tracking_features_dict, packing_coeff_dict \
            = process_discriminator_outputs(self.config, epoch_stats_dict)
        discriminator_scores_plot(wandb, scores_dict, vdw_penalty_dict, packing_coeff_dict, layout)
        plot_discriminator_score_correlates(self.config, wandb, epoch_stats_dict, layout)

        return None

    def reinitialize_models(self, generator, discriminator, conditioner, regressor):
        """
        reset model weights, if we did not load it from a given path
        @param generator:
        @param discriminator:
        @param conditioner:
        @param regressor:
        @return:
        """
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
                'other': 1,  # boron or smaller
            }
            for i in range(2, self.config.max_atomic_number + 1):
                self.config.conditioner_classes[i] = i

        conditioner_classes_dict = {i: self.config.conditioner_classes['other'] for i in
                                    range(self.config.max_atomic_number)}
        for i, (key, value) in enumerate(self.config.conditioner_classes.items()):
            if key != 'other':
                conditioner_classes_dict[key] = self.config.conditioner_classes[
                    key]  # assign all atoms to type other, except these specific ones
        self.config.conditioner_classes_dict = conditioner_classes_dict

    def mini_csp(self, data_loader, generator, discriminator):
        print('Starting Mini CSP')
        generator.eval()
        discriminator.eval()
        real_data = next(iter(data_loader)).clone().detach().cuda()
        real_supercell_data = self.supercell_builder.real_cell_to_supercell(real_data, self.config)

        num_molecules = real_data.num_graphs
        n_sampling_iters = self.config.sample_steps
        rdf_bins = 100
        rdf_range = [0, 10]

        discriminator_score, dist_dict = self.score_adversarially(real_supercell_data.clone(), discriminator)
        h_bond_score = compute_h_bond_score(self.config.feature_richness, self.atom_acceptor_ind, self.atom_donor_ind, self.num_acceptors_ind, self.num_donors_ind, real_supercell_data)
        vdw_penalty, normed_vdw_penalty = get_vdw_penalty(self.vdw_radii, dist_dict, real_data.num_graphs, real_data.mol_size)
        real_rdf, rr, atom_inds = crystal_rdf(real_supercell_data, rrange=rdf_range,
                                              bins=rdf_bins, mode='intermolecular',
                                              raw_density=True, atomwise=True, cpu_detach=True)

        volumes_list = []
        for i in range(real_data.num_graphs):
            volumes_list.append(cell_vol_torch(real_data.cell_params[i, 0:3], real_data.cell_params[i, 3:6]))
        volumes = torch.stack(volumes_list)
        real_packing_coeffs = real_data.Z * real_data.tracking[:, self.mol_volume_ind] / volumes

        real_samples_dict = {'score': softmax_and_score(discriminator_score).cpu().detach().numpy(),
                             'vdw overlap': vdw_penalty.cpu().detach().numpy(),
                             'density': real_packing_coeffs.cpu().detach().numpy(),
                             'h bond score': h_bond_score.cpu().detach().numpy(),
                             'cell params': real_data.cell_params.cpu().detach().numpy(),
                             'space group': real_data.sg_ind.cpu().detach().numpy(),
                             'RDF': real_rdf
                             }

        sampling_dict = {'score': np.zeros((num_molecules, n_sampling_iters)),
                         'vdw overlap': np.zeros((num_molecules, n_sampling_iters)),
                         'density': np.zeros((num_molecules, n_sampling_iters)),
                         'h bond score': np.zeros((num_molecules, n_sampling_iters)),
                         'cell params': np.zeros((num_molecules, n_sampling_iters, 12)),
                         'space group': np.zeros((num_molecules, n_sampling_iters)),
                         'handedness': np.zeros((num_molecules, n_sampling_iters)),
                         'RDF': [],
                         'batch supercell coords': [],
                         'batch supercell inds': [],
                         }
        discriminator = discriminator.eval()
        generator = generator.eval()
        with torch.no_grad():
            for ii in tqdm.tqdm(range(n_sampling_iters)):
                fake_data = real_data.clone()
                samples, prior = self.get_generator_samples(fake_data, generator)

                fake_supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
                    fake_data, samples, self.config.supercell_size,
                    self.config.discriminator.graph_convolution_cutoff,
                    align_molecules=False,  # molecules are either random on purpose, or pre-aligned with set handedness
                )
                fake_supercell_data = fake_supercell_data.cuda()  # todo remove soon

                discriminator_score, dist_dict = self.score_adversarially(fake_supercell_data.clone(), discriminator, discriminator_noise=0)
                h_bond_score = compute_h_bond_score(self.config.feature_richness, self.atom_acceptor_ind, self.atom_donor_ind, self.num_acceptors_ind, self.num_donors_ind, fake_supercell_data)
                vdw_penalty, normed_vdw_penalty = get_vdw_penalty(self.vdw_radii, dist_dict, fake_data.num_graphs, fake_data.mol_size)
                # rdf, rr, dist_dict = crystal_rdf(fake_supercell_data, rrange=rdf_range, bins=rdf_bins,
                #                                 raw_density=True, atomwise=True, mode='intermolecular', cpu_detach=True)

                volumes_list = []
                for i in range(fake_data.num_graphs):
                    volumes_list.append(
                        cell_vol_torch(fake_supercell_data.cell_params[i, 0:3], fake_supercell_data.cell_params[i, 3:6]))
                volumes = torch.stack(volumes_list)

                # todo issue here with division by two - make sure Z assignment is consistent throughout
                fake_packing_coeffs = fake_supercell_data.Z * fake_supercell_data.tracking[:, self.mol_volume_ind] / volumes

                sampling_dict['score'][:, ii] = softmax_and_score(discriminator_score).cpu().detach().numpy()
                sampling_dict['vdw overlap'][:, ii] = vdw_penalty.cpu().detach().numpy()
                sampling_dict['density'][:, ii] = fake_packing_coeffs.cpu().detach().numpy()
                sampling_dict['h bond score'][:, ii] = h_bond_score.cpu().detach().numpy()
                sampling_dict['cell params'][:, ii, :] = fake_supercell_data.cell_params.cpu().detach().numpy()
                sampling_dict['space group'][:, ii] = fake_supercell_data.sg_ind.cpu().detach().numpy()
                sampling_dict['handedness'][:, ii] = fake_supercell_data.asym_unit_handedness.cpu().detach().numpy()
                # sampling_dict['batch supercell coords'].append(fake_supercell_data.pos.cpu().detach().numpy())
                # sampling_dict['batch supercell inds'].append(fake_supercell_data.batch.cpu().detach().numpy())
                # sampling_dict['RDF'].append(rdf)

        """ what do we want from reporting
        1) plot scores from the distribution of all samples (see our normal scores plot)
   
        """

        log_mini_csp_scores_distributions(self.config, wandb, sampling_dict, real_samples_dict, real_data)
        #log_best_mini_csp_samples(self.config, wandb, discriminator, sampling_dict, real_samples_dict, real_data, self.supercell_builder)

        return None

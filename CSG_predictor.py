import wandb
from utils import *
import glob
from model_utils import *
from dataset_management.CSD_data_manager import Miner
from torch import backends, optim
import torch
from dataset_utils import BuildDataset, get_dataloaders, update_batch_size
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.colors import n_colors
import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from nikos.coordinate_transformations import coor_trans, cell_vol
from pyxtal import symmetry
from ase import Atoms
from ase.visualize import view
import rdkit.Chem as Chem
from dataset_management.random_crystal_builder import *
from models.generator_models import crystal_generator
from models.discriminator_models import crystal_discriminator
import ase.io
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from dataset_management.random_crystal_builder import (build_supercells, build_supercells_from_dataset)


class Predictor():
    def __init__(self, config):
        self.config = config
        setup_outputs = self.setup()
        if config.skip_saving_and_loading:
            self.prep_dataset = setup_outputs
        else:
            self.prep_dataset = None

    def setup(self):
        '''
        setup working directory
        move to relevant directory
        :return:
        '''

        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        periodicTable = Chem.GetPeriodicTable()
        self.atom_weights = {}
        for i in range(100):
            self.atom_weights[i] = periodicTable.GetAtomicWeight(i)

        if os.path.exists('symmetry_info.npy'):
            sym_info = np.load('symmetry_info.npy', allow_pickle=True).item()
            self.sym_ops = sym_info['sym_ops']
            self.point_groups = sym_info['point_groups']
            self.lattice_type = sym_info['lattice_type']
            self.space_groups = sym_info['space_groups']
        else:
            print('Pre-generating spacegroup symmetries')
            self.sym_ops = {}
            self.point_groups = {}
            self.lattice_type = {}
            self.space_groups = {}
            for i in tqdm.tqdm(range(1, 231)):
                sym_group = symmetry.Group(i)
                general_position_syms = sym_group.wyckoffs_organized[0][0]
                self.sym_ops[i] = [general_position_syms[i].affine_matrix for i in range(len(general_position_syms))]  # first 0 index is for general position, second index is superfluous, third index is the symmetry operation
                self.point_groups[i] = sym_group.point_group
                self.lattice_type[i] = sym_group.lattice_type
                self.space_groups[i] = sym_group.symbol

            self.sym_info = {}
            self.sym_info['sym_ops'] = self.sym_ops
            self.sym_info['point_groups'] = self.point_groups
            self.sym_info['lattice_type'] = self.lattice_type
            self.sym_info['space_groups'] = self.space_groups
            #np.save('symmetry_info', self.sym_info)

        # initialize fractional lattice vectors - should be exactly identical to what's in featurize_crystal_data.py
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

        miner = Miner(config=self.config, dataset_path=self.config.dataset_path, collect_chunks=False)

        if not self.config.skip_run_init:
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
            if self.config.explicit_run_enumeration:  # todo this is deprecated
                # move to working dir
                self.workDir = self.config.workdir + '/' + 'run%d' % self.config.run_num
                os.chdir(self.workDir)
                self.class_labels = list(np.load('group_labels.npy', allow_pickle=True))
                print('Resuming run %d' % self.config.run_num)
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

    def prep_metrics(self, config=None):
        metrics_list = ['discriminator train loss', 'discriminator test loss', 'epoch', 'discriminator learning rate',
                        'generator train loss', 'generator test loss', 'generator learning rate']
        metrics_dict = initialize_metrics_dict(metrics_list)

        return metrics_dict

    def update_gan_metrics(self, epoch, metrics_dict, d_err_tr, d_err_te, g_err_tr, g_err_te, d_lr, g_lr):
        metrics_dict['epoch'].append(epoch)
        metrics_dict['discriminator train loss'].append(np.mean(np.asarray(d_err_tr)))
        metrics_dict['discriminator test loss'].append(np.mean(np.asarray(d_err_te)))
        metrics_dict['discriminator learning rate'].append(d_lr)
        metrics_dict['generator train loss'].append(np.mean(np.asarray(g_err_tr)))
        metrics_dict['generator test loss'].append(np.mean(np.asarray(g_err_te)))
        metrics_dict['generator learning rate'].append(g_lr)

        return metrics_dict

    def init_gan(self, config, dataDims, print_status=True):
        '''
        Initialize model and optimizer
        :return:
        '''
        # init model
        print("Initializing models for " + config.mode)
        if config.mode == 'gan':
            generator = crystal_generator(config, dataDims)
            discriminator = crystal_discriminator(config, dataDims)
        elif config.mode == 'regression':
            generator = molecule_graph_model(dataDims,
                                             seed=config.seeds.model,
                                             output_dimension=1,  # single-target regression
                                             activation=config.generator.conditioner_activation,
                                             num_fc_layers=config.generator.conditioner_num_fc_layers,
                                             fc_depth=config.generator.conditioner_fc_depth,
                                             fc_dropout_probability=config.generator.conditioner_fc_dropout_probability,
                                             fc_norm_mode=config.generator.conditioner_fc_norm_mode,
                                             graph_model=config.generator.graph_model,
                                             graph_filters=config.generator.graph_filters,
                                             graph_convolutional_layers=config.generator.graph_convolution_layers,
                                             concat_mol_to_atom_features=True,
                                             pooling=config.generator.pooling,
                                             graph_norm=config.generator.graph_norm,
                                             num_spherical=config.generator.num_spherical,
                                             num_radial=config.generator.num_radial,
                                             graph_convolution=config.generator.graph_convolution,
                                             num_attention_heads=config.generator.num_attention_heads,
                                             add_radial_basis=config.generator.add_radial_basis,
                                             atom_embedding_size=config.generator.atom_embedding_size,
                                             radial_function=config.generator.radial_function,
                                             max_num_neighbors=config.generator.max_num_neighbors,
                                             convolution_cutoff=config.generator.graph_convolution_cutoff,
                                             device=config.device,
                                             )
            discriminator = nn.Linear(1, 1)  # dummy model

        if config.device == 'cuda':
            generator = generator.cuda()
            discriminator = discriminator.cuda()

        # init optimizers
        amsgrad = False
        beta1 = config.generator.beta1  # 0.9
        beta2 = config.generator.beta2  # 0.999
        weight_decay = config.generator.weight_decay  # 0.01
        momentum = 0

        if config.generator.optimizer == 'adam':
            g_optimizer = optim.Adam(generator.parameters(), amsgrad=amsgrad, lr=config.generator.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif config.generator.optimizer == 'adamw':
            g_optimizer = optim.AdamW(generator.parameters(), amsgrad=amsgrad, lr=config.generator.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif config.generator.optimizer == 'sgd':
            g_optimizer = optim.SGD(generator.parameters(), lr=config.generator.learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            print(config.generator.optimizer + ' is not a valid optimizer')
            sys.exit()

        amsgrad = False
        beta1 = config.discriminator.beta1  # 0.9
        beta2 = config.discriminator.beta2  # 0.999
        weight_decay = config.discriminator.weight_decay  # 0.01
        momentum = 0

        if config.discriminator.optimizer == 'adam':
            d_optimizer = optim.Adam(discriminator.parameters(), amsgrad=amsgrad, lr=config.discriminator.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif config.discriminator.optimizer == 'adamw':
            d_optimizer = optim.AdamW(discriminator.parameters(), amsgrad=amsgrad, lr=config.discriminator.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif config.discriminator.optimizer == 'sgd':
            d_optimizer = optim.SGD(discriminator.parameters(), lr=config.discriminator.learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            print(config.discriminator.optimizer + ' is not a valid optimizer')
            sys.exit()

        # init schedulers
        scheduler1 = lr_scheduler.ReduceLROnPlateau(
            g_optimizer,
            mode='min',
            factor=0.1,
            patience=15,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=15
        )
        lr_lambda = lambda epoch: 1.25
        scheduler2 = lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lr_lambda)
        lr_lambda2 = lambda epoch: 0.95
        scheduler3 = lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lr_lambda2)

        scheduler4 = lr_scheduler.ReduceLROnPlateau(
            d_optimizer,
            mode='min',
            factor=0.1,
            patience=15,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=15
        )
        lr_lambda = lambda epoch: 1.25
        scheduler5 = lr_scheduler.MultiplicativeLR(d_optimizer, lr_lambda=lr_lambda)
        lr_lambda2 = lambda epoch: 0.95
        scheduler6 = lr_scheduler.MultiplicativeLR(d_optimizer, lr_lambda=lr_lambda2)

        g_scheduler = [scheduler1, scheduler2, scheduler3]
        d_scheduler = [scheduler4, scheduler5, scheduler6]

        params1 = get_n_config(generator)
        if print_status:
            print('Generator model has {:.3f} million or {} parameters'.format(params1 / 1e6, int(params1)))

        params2 = get_n_config(discriminator)
        if print_status:
            print('Discriminator model has {:.3f} million or {} parameters'.format(params2 / 1e6, int(params2)))

        return generator, discriminator, g_optimizer, g_scheduler, d_optimizer, d_scheduler, params1, params2

    def get_batch_size(self, dataset, config):
        '''
        try larger batches until it crashes
        '''
        finished = False
        init_batch_size = config.min_batch_size.real
        max_batch_size = config.max_batch_size.real
        batch_reduction_factor = config.auto_batch_reduction

        generator, discriminator, g_optimizer, \
        g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)
        train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=init_batch_size)

        increment = 1.5  # what fraction by which to increment the batch size
        batch_size = int(init_batch_size)

        while (not finished) and (batch_size < max_batch_size):
            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU cache
                generator.cuda()
                discriminator.cuda()

            # if config.add_radial_basis is False:  # initializing spherical basis is too expensive to do repetitively
            #     generator, discriminator, g_optimizer, \
            #     g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)

            try:
                _ = self.epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                               g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                               update_gradients=True, record_stats=True, iteration_override=2, epoch=1)

                # if successful, increase the batch and try again
                batch_size = max(batch_size + 5, int(batch_size * increment))
                train_loader = update_batch_size(train_loader, batch_size)
                test_loader = update_batch_size(test_loader, batch_size)
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

        if config.device.lower() == 'cuda':
            torch.cuda.empty_cache()  # clear GPU cache

        return tr, te, batch_size

    def cell_diagnostic(self):
        config, dataset_builder = self.train_boilerplate()
        config.batch_size = 250
        train_loader, test_loader = get_dataloaders(dataset_builder, config)

        '''
        slow cell-by-cell analysis & rebuild
        '''
        # from models.cell_generation_analysis import parallel_cell_build_analyze
        #
        # overlaps, good_fit, inds = [], [], []
        # for i in tqdm.tqdm(range(len(train_loader.dataset))):
        #     fit, overlap = parallel_cell_build_analyze(train_loader.dataset[i],self.sym_ops,self.atom_weights)
        #     overlaps.append(overlap)
        #     good_fit.append(fit)
        #     inds.append(i)

        '''
        production build & analysis
        '''
        csd_rdf, rebuild_rdf, random_rdf = [], [], []
        csd_overlaps, rebuild_overlaps, random_overlaps = [], [], []
        for i, data in enumerate(train_loader):
            # build supercells from the dataset, and compute their properties
            csd_supercells = build_supercells_from_dataset(data.clone(), config, return_overlaps=False)
            csd_rdf_i, rr = crystal_rdf(csd_supercells, rrange=[0, 10], bins=100)
            csd_rdf.extend(csd_rdf_i.cpu().detach().numpy())
            # csd_overlaps.extend(csd_overlaps_i)

            rebuild_supercells, vol, rebuild_overlaps_i = \
                build_supercells(data.clone(), config, None, self.dataDims, self.atom_weights, self.sym_info, skip_cell_cleaning=True, ref_data=csd_supercells)

            rebuild_supercells = rebuild_supercells.cpu()
            # compare supercell centroids # DISCOVERY some CSD cells were built with incorrect symmetry
            # for ind in range(rebuild_supercells.num_graphs):
            #     rebuild_molwise_pos = torch.stack(rebuild_supercells.pos[rebuild_supercells.batch == ind].split(len(data.x[data.batch == ind])))
            #     csd_molwise_pos = torch.stack(csd_supercells.pos[csd_supercells.batch == ind].split(len(data.x[data.batch == ind])))
            #     rebuild_centroids = rebuild_molwise_pos.mean(1)
            #     csd_centroids = csd_molwise_pos.mean(1)
            #     distmat = torch.cdist(rebuild_centroids, csd_centroids, p=2)
            #     assert (torch.sum(distmat < 0.05) / len(rebuild_centroids)) > 0.99

            rebuild_rdf_i, rr = crystal_rdf(rebuild_supercells, rrange=[0, 10], bins=100)
            rebuild_rdf.extend(rebuild_rdf_i.cpu().detach().numpy())
            rebuild_overlaps.extend(rebuild_overlaps_i.cpu().detach().numpy())

            # build random supercells and compute their properties
            random_supercells, vol, random_overlaps_i = build_supercells(data.clone(), config, self.randn_generator(data.num_graphs), self.dataDims, self.atom_weights, self.sym_info)
            random_supercells = random_supercells.cpu()
            random_rdf_i, rr = crystal_rdf(random_supercells, rrange=[0, 10], bins=100)

            random_rdf.extend(random_rdf_i.cpu().detach().numpy())
            random_overlaps.extend(random_overlaps_i.cpu().detach().numpy())

        csd_rdf = np.stack(csd_rdf)
        rebuild_rdf = np.stack(rebuild_rdf)
        random_rdf = np.stack(random_rdf)

        print(f'RDF error of {np.sum(np.abs(csd_rdf - rebuild_rdf)) / np.sum(csd_rdf):.3f}')
        frac_correct = np.mean(np.sum(np.abs(csd_rdf - rebuild_rdf), axis=1) / np.sum(csd_rdf, axis=1) < 0.05)
        print(f'RDF {frac_correct:.3f} accurate on a per-sample basis at 95%')
        frac_correct = np.mean(np.sum(np.abs(csd_rdf - rebuild_rdf), axis=1) / np.sum(csd_rdf, axis=1) < 0.01)
        print(f'RDF {frac_correct:.3f} accurate on a per-sample basis at 99%')
        print(f'vs a random divergence of {np.sum(np.abs(csd_rdf - random_rdf)) / np.sum(csd_rdf):.3f}')
        flag = 0
        #
        # # visualize
        # mol1 = Atoms(symbols=csd_supercells.x[csd_supercells.batch == 0, 0].cpu().detach().numpy(), positions=csd_supercells.pos[csd_supercells.batch == 0].cpu().detach().numpy(), cell=csd_supercells.T_fc[0].T.cpu().detach().numpy())
        # mol2 = Atoms(symbols=rebuild_supercells.x[rebuild_supercells.batch == 0, 0].cpu().detach().numpy(), positions=rebuild_supercells.pos[rebuild_supercells.batch == 0].cpu().detach().numpy(),
        #              cell=rebuild_supercells.T_fc[0].T.cpu().detach().numpy())
        # view((mol1, mol2))
        # # compare the above

    def train(self):
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=self.config.wandb.experiment_tag):
            # config = wandb.config # todo: wandb configs don't support nested namespaces. Sweeps are officially broken - look at the github thread

            config, dataset_builder = self.train_boilerplate()

            # get batch size
            if config.auto_batch_sizing:
                print('Finding optimal batch size')
                train_loader, test_loader, config.final_batch_size = self.get_batch_size(dataset_builder, config)
            else:
                print('Getting dataloaders for pre-determined batch size')
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.max_batch_size
            del dataset_builder

            print("Training batch size set to {}".format(config.final_batch_size))
            # model, optimizer, schedulers
            print('Reinitializing model and optimizer')
            generator, discriminator, g_optimizer, \
            g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)
            n_params = params1 + params2

            # cuda
            if config.device.lower() == 'cuda':
                print('Putting models on CUDA')
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
                generator.cuda()
                discriminator.cuda()

            wandb.watch((generator, discriminator), log_graph=True, log_freq=100)
            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_metrics(config=config)

            # training loop
            d_hit_max_lr, g_hit_max_lr, converged, epoch = False, False, False, 0
            with torch.autograd.set_detect_anomaly(config.anomaly_detection):
                while (epoch < config.max_epochs) and not converged:
                    # very cool
                    print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                    print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                    print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                    # very cool
                    print("Starting Epoch {}".format(epoch))  # index from 0, very cool

                    '''
                    train
                    '''
                    try:
                        d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                            self.epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                       g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                       update_gradients=True, record_stats=True, epoch=epoch)  # train & compute test loss

                        with torch.no_grad():
                            d_err_te, d_te_record, g_err_te, g_te_record, test_epoch_stats_dict, time_test = \
                                self.epoch(config, dataLoader=test_loader, generator=generator, discriminator=discriminator,
                                           update_gradients=False, record_stats=True, epoch=epoch)  # compute loss on test set

                        print('epoch={}; d_nll_tr={:.5f}; d_nll_te={:.5f}; g_nll_tr={:.5f}; g_nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
                            epoch, np.mean(np.asarray(d_err_tr)), np.mean(np.asarray(d_err_te)),
                            np.mean(np.asarray(g_err_tr)), np.mean(np.asarray(g_err_te)),
                            time_train, time_test))

                        '''
                        update LR
                        '''
                        # update learning rate
                        d_optimizer, d_lr = set_lr(d_schedulers, d_optimizer, config.discriminator.lr_schedule,
                                                   config.discriminator.learning_rate, config.discriminator.max_lr, d_err_tr, d_hit_max_lr)
                        d_learning_rate = d_optimizer.param_groups[0]['lr']
                        if d_learning_rate >= config.discriminator.max_lr: d_hit_max_lr = True

                        # update learning rate
                        g_optimizer, g_lr = set_lr(g_schedulers, g_optimizer, config.generator.lr_schedule,
                                                   config.generator.learning_rate, config.generator.max_lr, g_err_tr, g_hit_max_lr)
                        g_learning_rate = g_optimizer.param_groups[0]['lr']
                        if g_learning_rate >= config.generator.max_lr: g_hit_max_lr = True

                        print(f"Learning rates are d={d_lr:.5f}, g={g_lr:.5f}")

                        '''
                        logging
                        '''
                        # logging
                        metrics_dict = self.update_gan_metrics(
                            epoch, metrics_dict, d_err_tr, d_err_te,
                            g_err_tr, g_err_te, d_learning_rate, g_learning_rate)

                        self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict, d_tr_record, d_te_record, g_tr_record, g_te_record)
                        if epoch % config.wandb.sample_reporting_frequency == 0:
                            self.log_gan_accuracy(epoch, train_loader, test_loader,
                                                  metrics_dict, g_tr_record, g_te_record, d_tr_record, d_te_record,
                                                  train_epoch_stats_dict, test_epoch_stats_dict, config,
                                                  generator, discriminator, wandb_log_figures=config.wandb.log_figures)

                        '''
                        convergence checks
                        '''

                        # check for convergence
                        if checkConvergence(metrics_dict['generator test loss'], config.history, config.generator.convergence_eps) and (epoch > config.history + 2):
                            config.finished = True
                            self.log_gan_accuracy(epoch, train_loader, test_loader,
                                                  metrics_dict, g_tr_record, g_te_record, d_tr_record, d_te_record,
                                                  train_epoch_stats_dict, test_epoch_stats_dict, config,
                                                  generator, discriminator, wandb_log_figures=config.wandb.log_figures)
                            break

                        if epoch % 5 == 0:
                            if train_loader.batch_size < len(train_loader.dataset):  # if the batch is smaller than the dataset
                                increment = max(4, int(train_loader.batch_size * 0.05))  # increment batch size
                                train_loader = update_batch_size(train_loader, train_loader.batch_size + increment)
                                test_loader = update_batch_size(test_loader, test_loader.batch_size + increment)
                                print(f'Batch size incremented to {train_loader.batch_size}')
                            wandb.log({'batch size': train_loader.batch_size})

                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):  # if we do hit OOM, slash the batch size
                            slash_increment = max(4, int(train_loader.batch_size * 0.1))
                            train_loader = update_batch_size(train_loader, train_loader.batch_size - slash_increment)
                            test_loader = update_batch_size(test_loader, test_loader.batch_size - slash_increment)
                            print('==============================')
                            print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
                            print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
                            print('==============================')
                            wandb.log({'batch size': train_loader.batch_size})
                        else:
                            raise e
                    epoch += 1

                    # for working with a trained model
                    if np.mean(np.asarray(d_err_te)) < 0.1:
                        self.MCMC_sampling()

                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU

    def epoch(self, config, dataLoader=None, generator=None, discriminator=None, g_optimizer=None, d_optimizer=None, update_gradients=True,
              iteration_override=None, record_stats=False, epoch=None):

        if config.mode == 'gan':
            return self.gan_epoch(config, dataLoader, generator, discriminator, g_optimizer, d_optimizer, update_gradients,
                                  iteration_override, record_stats, epoch)
        elif config.mode == 'regression':
            return self.regression_epoch(config, dataLoader, generator, g_optimizer, update_gradients,
                                         iteration_override, record_stats)

    def regression_epoch(self, config, dataLoader=None, generator=None, g_optimizer=None, update_gradients=True,
                         iteration_override=None, record_stats=False):

        t0 = time.time()
        if update_gradients:
            generator.train(True)
        else:
            generator.eval()

        g_err = []
        g_loss_record = []
        g_den_pred = []
        g_den_true = []
        epoch_stats_dict = {
            'tracking features': [],
        }

        for i, data in enumerate(dataLoader):
            '''
            noise injection
            '''
            if config.generator.positional_noise > 0:
                data.pos += torch.randn_like(data.pos) * config.generator.positional_noise

            regression_losses_list, predictions, targets = self.regression_loss(generator, data)
            g_den_pred.append(predictions.cpu().detach().numpy())
            g_den_true.append(targets.cpu().detach().numpy())

            g_loss = regression_losses_list.mean()
            g_err.append(g_loss.data.cpu().detach().numpy())  # average loss
            g_loss_record.extend(regression_losses_list.cpu().detach().numpy())  # loss distribution

            if update_gradients:
                g_optimizer.zero_grad()  # reset gradients from previous passes
                g_loss.backward()  # back-propagation
                g_optimizer.step()  # update parameters

            if record_stats:
                epoch_stats_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0
        if record_stats:
            epoch_stats_dict['generator density prediction'] = np.concatenate(g_den_pred) if g_den_pred != [] else None
            epoch_stats_dict['generator density target'] = np.concatenate(g_den_true) if g_den_true != [] else None
            return g_err, g_loss_record, g_err, g_loss_record, epoch_stats_dict, total_time
        else:
            return g_err, g_loss_record, g_err, g_loss_record, total_time

    def gan_epoch(self, config, dataLoader=None, generator=None, discriminator=None, g_optimizer=None, d_optimizer=None, update_gradients=True,
                  iteration_override=None, record_stats=False, epoch=None):
        t0 = time.time()
        if update_gradients:
            generator.train(True)
            discriminator.train(True)
        else:
            generator.eval()
            discriminator.eval()

        d_err = []
        d_loss_record = []
        d_real_scores = []
        d_fake_scores = []
        g_err = []
        g_loss_record = []
        g_flow_err = []
        g_flow_loss_record = []
        g_den_losses = []
        g_adv_scores = []
        g_den_pred = []
        g_den_true = []
        generated_intra_dist = []
        generated_inter_dist = []
        g_g2_losses = []
        g_packing_losses = []
        g_similarity_losses = []
        real_intra_dist = []
        real_inter_dist = []
        generated_samples_list = []
        epoch_stats_dict = {
            'tracking features': [],
        }
        generated_supercell_examples_dict = {}

        rand_batch_ind = np.random.randint(0, len(dataLoader))

        for i, data in enumerate(dataLoader):
            '''
            train discriminator
            '''
            if epoch % config.discriminator.training_period == 0:  # only train the discriminator every XX epochs
                if config.train_discriminator_adversarially or config.train_discriminator_on_randn or config.train_discriminator_on_noise:
                    score_on_real, score_on_fake, generated_samples, real_dist_dict, fake_dist_dict \
                        = self.train_discriminator(generator, discriminator, config, data, i)  # alternately trains on real and fake samples
                    if config.gan_loss == 'wasserstein':
                        d_losses = -score_on_real + score_on_fake  # maximize score on real, minimize score on fake
                        d_real_scores.append(score_on_real.cpu().detach().numpy())
                        d_fake_scores.append(score_on_fake.cpu().detach().numpy())

                    elif config.gan_loss == 'standard':
                        prediction = torch.cat((score_on_real, score_on_fake))
                        target = torch.cat((torch.ones_like(score_on_real[:, 0]), torch.zeros_like(score_on_fake[:, 0])))
                        d_losses = F.cross_entropy(prediction, target.long(), reduction='none')  # works much better
                        d_real_scores.append(F.softmax(score_on_real, dim=1)[:, 1].cpu().detach().numpy())
                        d_fake_scores.append(F.softmax(score_on_fake, dim=1)[:, 1].cpu().detach().numpy())

                    else:
                        print(config.gan_loss + ' is not an implemented GAN loss function!')
                        sys.exit()

                    d_loss = d_losses.mean()
                    d_err.append(d_loss.data.cpu().detach().numpy())  # average overall loss
                    d_loss_record.extend(d_losses.cpu().detach().numpy())  # overall loss distribution

                    if update_gradients:
                        if True:  # epoch < 50:
                            d_optimizer.zero_grad()  # reset gradients from previous passes
                            d_loss.backward()  # back-propagation
                            d_optimizer.step()  # update parameters

                    generated_intra_dist.append(fake_dist_dict['intramolecular dist'].cpu().detach().numpy())
                    generated_inter_dist.append(fake_dist_dict['intermolecular dist'].cpu().detach().numpy())
                    real_intra_dist.append(real_dist_dict['intramolecular dist'].cpu().detach().numpy())
                    real_inter_dist.append(real_dist_dict['intermolecular dist'].cpu().detach().numpy())

                    generated_samples_list.append(generated_samples)
                else:
                    d_err.append(np.zeros(1))
                    d_loss_record.extend(np.zeros(data.num_graphs))

            '''
            train_generator
            '''
            if any((config.train_generator_density, config.train_generator_adversarially, config.train_generator_g2, config.train_generator_packing)):

                adversarial_score, generated_samples, density_loss, density_prediction, density_target, \
                packing_loss, g2_loss, generated_dist_dict, supercell_examples, similarity_penalty = \
                    self.train_generator(generator, discriminator, config, data, i)

                if (supercell_examples is not None) and (i == rand_batch_ind):  # for a random batch in the epoch
                    generated_supercell_examples_dict['n examples'] = min(5, len(supercell_examples.ptr) - 1)  # max of 5 examples per epoch
                    supercell_inds = np.random.choice(len(supercell_examples.ptr) - 1, size=generated_supercell_examples_dict['n examples'], replace=False)
                    generated_supercell_examples_dict['T fc list'] = [supercell_examples.T_fc[ind].cpu().detach().numpy() for ind in supercell_inds]
                    generated_supercell_examples_dict['crystal positions'] = [supercell_examples.pos[supercell_examples.batch == ind].cpu().detach().numpy() for ind in supercell_inds]
                    generated_supercell_examples_dict['atoms'] = [supercell_examples.x[supercell_examples.batch == ind, 0].cpu().detach().numpy() for ind in supercell_inds]
                    # todo record some vital statistics and report in a table below
                    del supercell_examples

                if adversarial_score is not None:
                    if config.gan_loss == 'wasserstein':
                        adversarial_loss = -adversarial_score  # generator wants to maximize the score (minimize the negative score)
                    elif config.gan_loss == 'standard':
                        adversarial_score = F.softmax(adversarial_score, dim=1)[:, 1]  # modified minimax
                        # adversarial_loss = torch.log(1-adversarial_score) # standard minimax
                        adversarial_loss = -torch.log(adversarial_score)  # modified minimax
                    else:
                        print(config.gan_loss + ' is not an implemented GAN loss function!')
                        sys.exit()

                g_den_pred.append(density_prediction)
                g_den_true.append(density_target)
                g_losses_list = []
                if config.train_generator_density:
                    g_losses_list.append(density_loss.float())
                    g_den_losses.append(density_loss.cpu().detach().numpy())

                if config.train_generator_adversarially:
                    g_losses_list.append(adversarial_loss)
                    g_adv_scores.append(adversarial_score.cpu().detach().numpy())

                if config.train_generator_g2:
                    g_losses_list.append(g2_loss)
                    g_g2_losses.append(g2_loss.cpu().detach().numpy())

                if config.train_generator_packing:
                    g_losses_list.append(packing_loss)
                    g_packing_losses.append(packing_loss.cpu().detach().numpy())

                if config.generator_similarity_penalty != 0:
                    if similarity_penalty is not None:
                        g_losses_list.append(similarity_penalty)
                        g_similarity_losses.append(similarity_penalty.cpu().detach().numpy())
                    else:
                        print('similarity penalty was none')

                if config.train_generator_adversarially or config.train_generator_g2:
                    generated_intra_dist.append(generated_dist_dict['intramolecular dist'].cpu().detach().numpy())
                    generated_inter_dist.append(generated_dist_dict['intermolecular dist'].cpu().detach().numpy())

                g_losses = torch.sum(torch.stack(g_losses_list), dim=0)

                g_loss = g_losses.mean()
                g_err.append(g_loss.data.cpu().detach().numpy())  # average loss
                g_loss_record.extend(g_losses.cpu().detach().numpy())  # loss distribution
                generated_samples_list.append(generated_samples)

                if update_gradients:
                    g_optimizer.zero_grad()  # reset gradients from previous passes
                    g_loss.backward()  # back-propagation
                    g_optimizer.step()  # update parameters
            else:
                g_err.append(np.zeros(1))
                g_loss_record.extend(np.zeros(data.num_graphs))

            # flow loss - probability maximization on the datapoints in the dataset
            if config.train_generator_as_flow:
                if (epoch < config.cut_max_prob_training_after):  # stop using max_prob training after a few initial epochs

                    g_flow_losses = self.flow_iter(generator, data.to(config.device))

                    g_flow_loss = g_flow_losses.mean()
                    g_flow_err.append(g_flow_loss.data.cpu())  # average loss
                    g_flow_loss_record.append(g_flow_losses.cpu().detach().numpy())  # loss distribution

                    if update_gradients:
                        g_optimizer.zero_grad()  # reset gradients from previous passes
                        g_flow_loss.backward()  # back-propagation
                        g_optimizer.step()  # update parameters

            if config.train_generator_on_randn:
                if (epoch < config.cut_max_prob_training_after):  # stop using max_prob training after a few initial epochs
                    g_max_prob_losses = self.max_prob_iter(generator, data)

                    g_max_prob_loss = g_max_prob_losses.mean()
                    # g_max_prob_err.append(g_max_prob_loss.data.cpu())  # average loss
                    # g_max_prob_loss_record.append(g_max_prob_losses.cpu().detach().numpy())  # loss distribution

                    if update_gradients:
                        g_optimizer.zero_grad()  # reset gradients from previous passes
                        g_max_prob_loss.backward()  # back-propagation
                        g_optimizer.step()  # update parameters

            if (len(generated_samples_list) < i) and record_stats:  # make some samples for analysis if we have none so far from this step
                generated_samples = generator(len(data.y), z=None, conditions=data.to(config.device))
                generated_samples_list.append(generated_samples.cpu().detach().numpy())

            if record_stats:
                epoch_stats_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0

        if record_stats:
            epoch_stats_dict['discriminator real score'] = np.concatenate(d_real_scores) if d_real_scores != [] else None
            epoch_stats_dict['discriminator fake score'] = np.concatenate(d_fake_scores) if d_fake_scores != [] else None
            epoch_stats_dict['generator density loss'] = np.concatenate(g_den_losses) if g_den_losses != [] else None
            epoch_stats_dict['generator adversarial score'] = np.concatenate(g_adv_scores) if g_adv_scores != [] else None
            epoch_stats_dict['generator flow loss'] = np.concatenate(g_flow_loss_record) if g_flow_loss_record != [] else None
            epoch_stats_dict['generator short range loss'] = np.concatenate(g_g2_losses) if g_g2_losses != [] else None
            epoch_stats_dict['generator packing loss'] = np.concatenate(g_packing_losses) if g_packing_losses != [] else None
            epoch_stats_dict['generator density prediction'] = np.concatenate(g_den_pred) if g_den_pred != [] else None
            epoch_stats_dict['generator density target'] = np.concatenate(g_den_true) if g_den_true != [] else None
            epoch_stats_dict['generator similarity loss'] = np.concatenate(g_similarity_losses) if g_similarity_losses != [] else None
            epoch_stats_dict['generated intra distance hist'] = np.histogram(np.concatenate(generated_intra_dist), bins=100, range=(0, config.discriminator.graph_convolution_cutoff), density=True) if generated_intra_dist != [] else None
            epoch_stats_dict['generated inter distance hist'] = np.histogram(np.concatenate(generated_inter_dist), bins=100, range=(0, config.discriminator.graph_convolution_cutoff), density=True) if generated_inter_dist != [] else None
            epoch_stats_dict['real intra distance hist'] = np.histogram(np.concatenate(real_intra_dist), bins=100, range=(0, config.discriminator.graph_convolution_cutoff), density=True) if real_intra_dist != [] else None
            epoch_stats_dict['real inter distance hist'] = np.histogram(np.concatenate(real_inter_dist), bins=100, range=(0, config.discriminator.graph_convolution_cutoff), density=True) if real_inter_dist != [] else None
            epoch_stats_dict['generated cell parameters'] = np.concatenate(generated_samples_list) if generated_samples_list != [] else None
            epoch_stats_dict['generated supercell examples dict'] = generated_supercell_examples_dict if generated_supercell_examples_dict != {} else None

            return d_err, d_loss_record, g_err, g_loss_record, epoch_stats_dict, total_time
        else:
            return d_err, d_loss_record, g_err, g_loss_record, total_time

    def adversarial_loss(self, discriminator, data, config):

        output, extra_outputs = discriminator(data, return_dists=True)  # reshape output from flat filters to channels * filters per channel

        # discriminator score
        if config.gan_loss == 'wasserstein':
            scores = output[:, 0]  # F.softplus(output[:, 0])  # critic score - higher meaning more plausible

        elif config.gan_loss == 'standard':
            scores = output  # F.softmax(output, dim=1)[:, -1]  # 'no' and 'yes' unnormalized activations

        else:
            print(config.gan_loss + ' is not a valid GAN loss function!')
            sys.exit()

        return scores, output, extra_outputs['dists dict']

    def pairwise_correlations_analysis(self, dataset_builder, config):
        '''
        check correlations in the data
        :param dataset_builder:
        :param config:
        :return:
        '''
        data = dataset_builder.datapoints
        keys = self.dataDims['crystal features']
        if config.generator.conditional_modelling:
            if (config.generator.conditioning_mode != 'graph model'):
                keys.extend(self.dataDims['conditional features'])
            else:
                data = np.asarray([(data[i].cell_params).detach().numpy() for i in range(len(data))])[:, 0, :]

        df = pd.DataFrame(data, columns=keys)
        correlations = df.corr()

        return correlations, keys

    def check_inversion_quality(self, model, test_loader, config):  # todo deprecated - need normed cell params
        # check for quality of the inversion
        if self.n_conditional_features > 0:
            if config.generator.conditioning_mode == 'molecule features':
                test_conditions = next(iter(test_loader)).to(config.device)
                test_sample = model.sample(test_conditions.num_graphs, conditions=test_conditions)
                test_conditions.y[0][:, :-self.n_conditional_features] = test_sample
                zs, _, _ = model.forward(test_conditions)
                test_conditions.y[0] = torch.cat((zs, test_conditions.y[0][:, -self.n_conditional_features:]), dim=1)
                test_sample2, _ = model.backward(test_conditions)
            elif config.generator.conditioning_mode == 'graph model':
                test_conditions = next(iter(test_loader)).to(config.device)
                test_sample = model.sample(test_conditions.num_graphs, conditions=test_conditions)
                test_conditions.y[0] = test_sample
                zs, _, _ = model.forward(test_conditions)
                test_conditions.y[0] = zs
                test_sample2, _ = model.backward(test_conditions)
        else:
            test_conditions = next(iter(test_loader)).to(config.device)
            test_sample = model.sample(test_conditions.num_graphs, conditions=None)
            test_conditions.y[0] = test_sample
            zs, _, _ = model.forward(test_conditions)
            test_conditions.y[0] = zs
            test_sample2, _ = model.backward(test_conditions)
        diff = torch.mean((torch.abs(test_sample - test_sample2))).cpu().detach().numpy()
        print('Average Inversion Error is {:.6f} per sample'.format(diff))
        if diff > 0.01:
            print("Warning! Inversion error is notably large! The flow is likely broken!")
        wandb.log({'Inversion error': diff})
        del zs, test_sample, test_sample2

    def get_sample_efficiency(self, dataDims, targets, renormalized_samples, sample_efficiency_dict, feature_accuracy_dict, sampler):
        assert renormalized_samples.ndim == 3
        samples = renormalized_samples[:len(targets)]
        targets = np.asarray(targets)[:len(samples)]
        renormalized_targets = np.zeros_like(targets)
        for i in range(dataDims['n crystal features']):
            renormalized_targets[:, i] = targets[:, i] * dataDims['stds'][i] + dataDims['means'][i]

        targets_rep = np.repeat(renormalized_targets[:, None, :], samples.shape[1], axis=1)
        # denominator = np.repeat(np.repeat(np.quantile(renormalized_targets,0.95,axis=0)[None,None,:],samples.shape[0],axis=0),samples.shape[1],axis=1)
        denominator = targets_rep.copy()
        for i in range(dataDims['n crystal features']):
            if dataDims['dtypes'][i] == 'bool':
                denominator[:, :, i] = 1

        errors = np.abs((targets_rep - samples) / denominator)
        feature_mae = np.mean(errors, axis=(0, 1))

        for i in range(dataDims['n crystal features']):
            feature_accuracy_dict[sampler + ' ' + dataDims['crystal features'][i] + ' mae'] = feature_mae[i]
            for cutoff in [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3]:
                feature_accuracy_dict[sampler + ' ' + dataDims['crystal features'][i] + ' efficiency at {}'.format(cutoff)] = np.average(errors[:, :, i] < cutoff)

        mae_error = np.mean(errors, axis=2)

        for cutoff in [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3]:
            sample_efficiency_dict[sampler + ' efficiency at {}'.format(cutoff)] = np.average(mae_error < cutoff)

        sample_efficiency_dict[sampler + ' average mae'] = np.average(mae_error)

        return sample_efficiency_dict, feature_accuracy_dict

    def get_generation_conditions(self, train_loader, test_loader, model, config):  # todo deprecated - fix data.y norming
        generation_conditions = []
        targets = []
        for i, data in enumerate(test_loader):
            generation_conditions.append(data.to(model.device))
            targets.extend(generation_conditions[-1].y[0].cpu().detach().numpy())

        targets = np.asarray(targets)

        train_data = train_loader.dataset
        train_data = np.asarray([(train_data[i].y[0]).detach().numpy() for i in range(len(train_data))])[:, 0, :]
        if (self.n_conditional_features > 0) and (config.generator.conditioning_mode == 'molecule features'):
            train_data = train_data[:, :-self.n_conditional_features]
            targets = targets[:, :-self.n_conditional_features]

        del generation_conditions
        return targets, train_data

    def sample_nf(self, n_repeats, config, model, test_loader):
        nf_samples = [[] for _ in range(n_repeats)]
        print('Sampling from NF')
        for j in tqdm.tqdm(range(n_repeats)):
            for i, data in enumerate(test_loader):
                minibatch_size = data.num_graphs
                if config.generator.conditional_modelling:
                    if config.device == 'cuda':
                        data = data.cuda()
                    nf_samples[j].extend(model.sample(
                        minibatch_size,
                        conditions=data
                    ).cpu().detach().numpy())
                else:
                    nf_samples[j].extend(model.sample(
                        minibatch_size,
                    ).cpu().detach().numpy())
        return np.asarray(nf_samples).transpose((1, 0, 2))  # molecule - n_samples - feature dimension

    def get_pc_scores(self, sample_dict, pca):
        # score everything via pca
        pc_scores_dict = {}
        for i, (key, value) in enumerate(sample_dict.items()):
            if value.ndim == 3:
                pc_scores_dict[key] = pca.score_samples(value.reshape(value.shape[0] * value.shape[1], value.shape[2]))
            else:
                pc_scores_dict[key] = pca.score_samples(value)
        return pc_scores_dict

    def get_nf_scores(self, sample_dict, model, config, dataloader, n_repeats, dataset_length):  # todo deprecated - fix data.y norming
        nf_scores_dict = {}
        for i, (key, value) in enumerate(sample_dict.items()):
            scores = []
            for n, data in enumerate(dataloader):
                sample = sample_dict[key]
                if sample.ndim == 2:
                    if sample.shape[0] == dataset_length * n_repeats:
                        sample = sample.reshape(dataset_length, n_repeats, sample.shape[-1])  # roll up the first dim for the indepenent and pc sampels
                    elif sample.shape[0] == dataset_length:
                        sample = sample[:, None, :]  # the real data only has one repeat
                sample = torch.Tensor(sample[n * self.sampling_batch_size:n * self.sampling_batch_size + self.sampling_batch_size:1]).to(config.device)
                for j in range(sample.shape[1]):  # todo this is very likely broken
                    if self.n_conditional_features > 0:
                        data.y[0] = sample[:, j]

                    scores.extend(model.score(data.to(config.device)).cpu().detach().numpy())
            nf_scores_dict[key] = np.asarray(scores)

        return nf_scores_dict

    def log_gan_loss(self, metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict, d_tr_record, d_te_record, g_tr_record, g_te_record):
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

        # log discriminator losses
        wandb.log(current_metrics)
        hist = np.histogram(d_tr_record, bins=256, range=(np.amin(d_tr_record), np.quantile(d_tr_record, 0.9)))
        wandb.log({"Discriminator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(d_te_record, bins=256, range=(np.amin(d_te_record), np.quantile(d_te_record, 0.9)))
        wandb.log({"Discriminator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        wandb.log({"D Train Loss Coeff. of Variation": np.sqrt(np.var(d_tr_record)) / np.average(d_tr_record)})
        wandb.log({"D Test Loss Coeff. of Variation": np.sqrt(np.var(d_te_record)) / np.average(d_te_record)})

        # log generator losses
        hist = np.histogram(g_tr_record, bins=256, range=(np.amin(g_tr_record), np.quantile(g_tr_record, 0.9)))
        wandb.log({"Generator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(g_te_record, bins=256, range=(np.amin(g_te_record), np.quantile(g_te_record, 0.9)))
        wandb.log({"Generator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        wandb.log({"G Train Loss Coeff. of Variation": np.sqrt(np.var(g_tr_record)) / np.average(g_tr_record)})
        wandb.log({"G Test Loss Coeff. of Variation": np.sqrt(np.var(g_te_record)) / np.average(g_te_record)})

        # log specific losses
        special_losses = {}
        special_losses['epoch'] = current_metrics['epoch']
        for key in train_epoch_stats_dict.keys():
            if ('loss' in key) and (train_epoch_stats_dict[key] is not None):
                special_losses['Train ' + key] = np.average(train_epoch_stats_dict[key])
            if ('loss' in key) and (test_epoch_stats_dict[key] is not None):
                special_losses['Test ' + key] = np.average(test_epoch_stats_dict[key])
            if ('score' in key) and (train_epoch_stats_dict[key] is not None):
                special_losses['Train ' + key] = np.average(train_epoch_stats_dict[key])
            if ('score' in key) and (test_epoch_stats_dict[key] is not None):
                special_losses['Test ' + key] = np.average(test_epoch_stats_dict[key])
        wandb.log(special_losses)


    def params_f_to_c(self, cell_lengths, cell_angles):
        cell_vector_a, cell_vector_b, cell_vector_c = \
            torch.tensor(coor_trans('f_to_c', np.array((1, 0, 0)), cell_lengths, cell_angles)), \
            torch.tensor(coor_trans('f_to_c', np.array((0, 1, 0)), cell_lengths, cell_angles)), \
            torch.tensor(coor_trans('f_to_c', np.array((0, 0, 1)), cell_lengths, cell_angles))

        return np.concatenate((cell_vector_a[None, :], cell_vector_b[None, :], cell_vector_c[None, :]), axis=0), cell_vol(cell_lengths, cell_angles)

    def get_CSD_crystal(self, reader, csd_identifier, mol_n_atoms, z_value):
        crystal = reader.crystal(csd_identifier)
        tile_len = 1
        n_tiles = tile_len ** 3
        ref_cell = crystal.packing(box_dimensions=((0, 0, 0), (tile_len, tile_len, tile_len)), inclusion='CentroidIncluded')

        ref_cell_coords_c = np.zeros((n_tiles * z_value, mol_n_atoms, 3), dtype=np.float_)
        ref_cell_centroids_c = np.zeros((n_tiles * z_value, 3), dtype=np.float_)

        for ind, component in enumerate(ref_cell.components):
            if ind < z_value:  # some cells have spurious little atoms counted as extra components. Just hope the early components are the good ones
                ref_cell_coords_c[ind, :] = np.asarray([atom.coordinates for atom in component.atoms if atom.atomic_number != 1])  # filter hydrogen
                ref_cell_centroids_c[ind, :] = np.average(ref_cell_coords_c[ind], axis=0)

        return ref_cell_coords_c

    def log_gan_accuracy(self, epoch, train_loader, test_loader,
                         metrics_dict, g_tr_record, g_te_record, d_tr_record, d_te_record,
                         train_epoch_stats_dict, test_epoch_stats_dict, config,
                         generator, discriminator, wandb_log_figures=True):
        '''
        Do analysis and upload results to w&b

        1: generator efficiency on test set
            -:> cell parameter rmsd
            -:> RDF / RMSD20 (from compack, CCDC)
            -:> cell density
            -:> 1D & 2D correlation overlaps
        Future: FF evaluations, polymorphs ranking

        :param epoch:
        :param train_loader:
        :param test_loader:
        :param metrics_dict:
        :param g_tr_record:
        :param g_te_record:
        :param d_tr_record:
        :param d_te_record:
        :param epoch_stats_dict:
        :param config:
        :param generator:
        :param discriminator:
        :return:
        '''
        # cell images
        # for i in range(len(bottomXsamples)):
        #     image = wandb.Image(image_set[i],
        #                         caption="Epoch {} Molecule5 {} bad sample {} classified as {}".format(epoch, smiles_set[i], self.groupLabels[int(true_set[i])], self.groupLabels[int(guess_set[i])]))
        #     wandb.log({"Bad examples": image})

        '''
        example generated unit cells
        '''
        if config.mode == 'gan':
            generated_supercell_examples_dict = test_epoch_stats_dict['generated supercell examples dict']
            if generated_supercell_examples_dict is not None:
                if config.wandb.log_figures:
                    for i in range(generated_supercell_examples_dict['n examples']):
                        cell_vectors = generated_supercell_examples_dict['T fc list'][i].T
                        supercell_pos = generated_supercell_examples_dict['crystal positions'][i]
                        supercell_atoms = generated_supercell_examples_dict['atoms'][i]

                        ref_cell_size = len(supercell_pos) // (2 * config.supercell_size + 1) ** 3
                        ref_cell_pos = supercell_pos[:ref_cell_size]

                        ref_cell_centroid = ref_cell_pos.mean(0)
                        max_range = np.amax(np.linalg.norm(cell_vectors, axis=0)) / 1.3
                        keep_pos = np.argwhere(cdist(supercell_pos, ref_cell_centroid[None, :]) < max_range)[:, 0]

                        ase.io.write(f'supercell_{i}.pdb', Atoms(symbols=supercell_atoms[keep_pos], positions=supercell_pos[keep_pos], cell=cell_vectors))
                        wandb.log({'Generated Supercells': wandb.Molecule(open(f"supercell_{i}.pdb"))})  # todo find the max number of atoms this thing will take for bonding

            # todo rebuild - make sure this is meaningful
            # # correlate losses with molecular features
            # tracking_features = np.asarray(test_epoch_stats_dict['tracking features'])
            # g_loss_correlations = np.zeros(config.dataDims['n tracking features'])
            # d_loss_correlations = np.zeros(config.dataDims['n tracking features'])
            # features = []
            # for i in range(config.dataDims['n tracking features']):  # not that interesting
            #     features.append(config.dataDims['tracking features dict'][i])
            #     g_loss_correlations[i] = np.corrcoef(g_te_record, tracking_features[:, i], rowvar=False)[0, 1]
            #     d_loss_correlations[i] = np.corrcoef(d_te_record, tracking_features[:, i], rowvar=False)[0, 1]
            #
            # g_sort_inds = np.argsort(g_loss_correlations)
            # g_loss_correlations = g_loss_correlations[g_sort_inds]
            #
            # d_sort_inds = np.argsort(d_loss_correlations)
            # d_loss_correlations = d_loss_correlations[d_sort_inds]
            #
            # if config.wandb.log_figures:
            #     fig = go.Figure(go.Bar(
            #         y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['n tracking features'])],
            #         x=[g_loss_correlations[i] for i in range(config.dataDims['n tracking features'])],
            #         orientation='h',
            #     ))
            #     wandb.log({'G Loss correlations': fig})
            #
            #     fig = go.Figure(go.Bar(
            #         y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['n tracking features'])],
            #         x=[d_loss_correlations[i] for i in range(config.dataDims['n tracking features'])],
            #         orientation='h',
            #     ))
            #     wandb.log({'D Loss correlations': fig})

            '''
            cell parameter analysis
            '''
            if train_epoch_stats_dict['generated cell parameters'] is not None:
                n_crystal_features = config.dataDims['n crystal features']
                generated_samples = test_epoch_stats_dict['generated cell parameters']
                means = config.dataDims['means']
                stds = config.dataDims['stds']

                # slightly expensive to do this every time
                dataset_cell_distribution = np.asarray([train_loader.dataset[ii].cell_params[0].cpu().detach().numpy() for ii in range(len(train_loader.dataset))])

                # raw outputs
                renormalized_samples = np.zeros_like(generated_samples)
                for i in range(generated_samples.shape[1]):
                    renormalized_samples[:, i] = generated_samples[:, i] * stds[i] + means[i]

                # samples as the builder sees them
                cell_lengths, cell_angles, rand_position, rand_rotation = torch.tensor(generated_samples).split(3, 1)
                cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
                    cell_lengths, cell_angles, rand_position, rand_rotation, None, config.dataDims, enforce_crystal_system=False)
                cleaned_samples = torch.cat((cell_lengths, cell_angles, rand_position, rand_rotation), dim=1).detach().numpy()

                overlaps_1d = {}
                sample_means = {}
                sample_stds = {}
                for i, key in enumerate(config.dataDims['crystal features']):
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

                if wandb_log_figures:
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

                        fig_dict[self.dataDims['crystal features'][i] + ' distribution'] = fig

                    wandb.log(fig_dict)

            '''
            cell atomic distances
            '''
            if train_epoch_stats_dict['generated inter distance hist'] is not None:  # todo update this
                hh2_test, rr = test_epoch_stats_dict['generated inter distance hist']
                hh2_train, _ = train_epoch_stats_dict['generated inter distance hist']
                if train_epoch_stats_dict['real inter distance hist'] is not None:  # if there is no discriminator training, we don't generate this
                    hh1, rr = train_epoch_stats_dict['real inter distance hist']
                else:
                    hh1 = hh2_test

                shell_volumes = (4 / 3) * torch.pi * ((rr[:-1] + np.diff(rr)) ** 3 - rr[:-1] ** 3)
                rdf1 = hh1 / shell_volumes
                rdf2 = hh2_test / shell_volumes
                rdf3 = hh2_train / shell_volumes
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rr, y=rdf1, name='real'))
                fig.add_trace(go.Scatter(x=rr, y=rdf2, name='gen, test'))
                fig.add_trace(go.Scatter(x=rr, y=rdf3, name='gen, train'))

                if config.wandb.log_figures:
                    wandb.log({'G2 Comparison': fig})

                range_analysis_dict = {}
                if train_epoch_stats_dict['real inter distance hist'] is not None:  # if there is no discriminator training, we don't generate this
                    # get histogram overlaps
                    range_analysis_dict['tr g2 overlap'] = np.min(np.concatenate((rdf1[None, :] / rdf1.sum(), rdf3[None, :] / rdf1.sum()), axis=0), axis=0).sum()
                    range_analysis_dict['te g2 overlap'] = np.min(np.concatenate((rdf1[None, :] / rdf1.sum(), rdf2[None, :] / rdf1.sum()), axis=0), axis=0).sum()

                # get probability mass at too-close range (should be ~zero)
                range_analysis_dict['tr short range density fraction'] = np.sum(rdf3[rr[1:] < 1.2] / rdf3.sum())
                range_analysis_dict['te short range density fraction'] = np.sum(rdf2[rr[1:] < 1.2] / rdf2.sum())

                wandb.log(range_analysis_dict)

        '''
        auxiliary regression target
        '''  # todo plot anything that we have here
        if test_epoch_stats_dict['generator density target'] is not None:  # config.train_generator_density:
            target_mean = self.dataDims['target mean']
            target_std = self.dataDims['target std']

            target = np.asarray(test_epoch_stats_dict['generator density target'])
            prediction = np.asarray(test_epoch_stats_dict['generator density prediction'])
            orig_target = target * target_std + target_mean
            orig_prediction = prediction * target_std + target_mean

            train_target = np.asarray(train_epoch_stats_dict['generator density target'])
            train_prediction = np.asarray(train_epoch_stats_dict['generator density prediction'])
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
            loss_dict['Regression R2'] = linreg_result.rvalue
            loss_dict['Regression slope'] = linreg_result.slope
            wandb.log(loss_dict)

            # log loss distribution
            if config.wandb.log_figures:  # todo clean
                # predictions vs target trace
                xline = np.linspace(max(min(orig_target), min(orig_prediction)), min(max(orig_target), max(orig_prediction)), 10)
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(x=orig_target, y=orig_prediction, ncontours=50, nbinsx=40, nbinsy=40, showlegend=True))
                fig.update_traces(contours_coloring="fill")
                fig.update_traces(contours_showlines=False)
                fig.add_trace(go.Scatter(x=orig_target, y=orig_prediction, mode='markers', showlegend=True, opacity=0.5))
                fig.add_trace(go.Scatter(x=xline, y=xline))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Test Packing Coefficient': fig})

                xline = np.linspace(max(min(train_orig_target), min(train_orig_prediction)), min(max(train_orig_target), max(train_orig_prediction)), 10)
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(x=train_orig_target, y=train_orig_prediction, ncontours=50, nbinsx=40, nbinsy=40, showlegend=True))
                fig.update_traces(contours_coloring="fill")
                fig.update_traces(contours_showlines=False)
                fig.add_trace(go.Scatter(x=train_orig_target, y=train_orig_prediction, mode='markers', showlegend=True, opacity=0.5))
                fig.add_trace(go.Scatter(x=xline, y=xline))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Train Packing Coefficient': fig})

        return None

    def train_discriminator(self, generator, discriminator, config, data, i):

        # generate fakes & create supercell data
        if config.train_discriminator_adversarially:
            generated_samples = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device))
        elif config.train_discriminator_on_randn:
            generated_samples = self.randn_generator.forward(data.num_graphs).to(generator.device)

        real_supercell_data = build_supercells_from_dataset(data.clone(), config)
        if not config.train_discriminator_on_noise:
            fake_supercell_data, generated_cell_volumes, overlaps_list = \
                build_supercells(data.clone().to(generated_samples.device), config, generated_samples, override_sg=config.generate_sgs)
        else:
            generated_samples = self.randn_generator.forward(data.num_graphs).to(generator.device)  # placeholder
            fake_supercell_data = real_supercell_data.clone()
            fake_supercell_data.pos += torch.randn_like(fake_supercell_data.pos) * 10  # huge amount of noise - should be basically nonsense
        if False:  # i == 0:
            print('Batch {} real + fake supercell generation took {:.2f} seconds for {} samples'.format(i, round(t1 - t0, 2), data.num_graphs))

        if config.device.lower() == 'cuda':  # redundant
            real_supercell_data = real_supercell_data.cuda()
            fake_supercell_data = fake_supercell_data.cuda()

        if config.test_mode or config.anomaly_detection:
            assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"
            assert torch.sum(torch.isnan(fake_supercell_data.x)) == 0, "NaN in training input"

        if config.discriminator.positional_noise > 0:
            real_supercell_data.pos += torch.randn_like(real_supercell_data.pos) * config.discriminator.positional_noise
            fake_supercell_data.pos += torch.randn_like(fake_supercell_data.pos) * config.discriminator.positional_noise

        score_on_real, raw_output_on_real, real_intra_distances_dict = self.adversarial_loss(discriminator, real_supercell_data, config)
        score_on_fake, raw_output_on_fake, fake_pairwise_distances_dict = self.adversarial_loss(discriminator, fake_supercell_data, config)

        return score_on_real, score_on_fake, generated_samples.cpu().detach().numpy(), real_intra_distances_dict, fake_pairwise_distances_dict

    def train_generator(self, generator, discriminator, config, data, i):
        # noise injection
        if config.generator.positional_noise > 0:
            data.pos += torch.randn_like(data.pos) * config.generator.positional_noise

        [[generated_samples, latent], prior, condition] = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device), return_latent=True, return_condition=True, return_prior=True)

        if (config.generator_similarity_penalty != 0) and (len(generated_samples) > 5):
            similarity_penalty_i = config.generator_similarity_penalty * F.mse_loss(generated_samples.std(0), prior.std(0))  # match variance with the input noise
            similarity_penalty = torch.ones(len(generated_samples)).to(generated_samples.device) * similarity_penalty_i  # copy across batch
        else:
            similarity_penalty = None

        if config.train_generator_adversarially or config.train_generator_packing or config.train_generator_g2 or config.train_generator_density:
            supercell_data, generated_cell_volumes, overlaps_list = build_supercells(data.clone(), config, generated_samples, self.dataDims, self.atom_weights, self.sym_info, override_sg=config.generate_sgs)
            data.cell_params = supercell_data.cell_params
        else:
            supercell_data = None

        if config.train_generator_adversarially or config.train_generator_g2:
            if config.device.lower() == 'cuda':
                supercell_data = supercell_data.cuda()

            if config.test_mode or config.anomaly_detection:
                assert torch.sum(torch.isnan(data.x)) == 0, "NaN in training input"

            discriminator_score, raw_output, dist_dict = self.adversarial_loss(discriminator, supercell_data, config)

            rdf_inter, bin_range = parallel_compute_rdf_torch([dist_dict['intermolecular dist'][dist_dict['intermolecular dist batch'] == n] for n in range(supercell_data.num_graphs)],
                                                              bins=50, density=torch.ones(supercell_data.num_graphs).to(config.device), rrange=[0, self.config.generator.graph_convolution_cutoff])

            vdw_range = 3  # rough vdW volume for all atoms
            g2_loss = 10 * (F.relu(-(bin_range - vdw_range)) / vdw_range * rdf_inter).mean(1).requires_grad_()  # smoothly penalize for disrespecting vdW volumes

            if self.config.test_mode:
                assert torch.sum(torch.isnan(g2_loss)) == 0
                if similarity_penalty is not None:
                    assert torch.sum(torch.isnan(similarity_penalty)) == 0

        else:
            discriminator_score = None
            g2_loss = None
            dist_dict = None

        density_loss, density_prediction, density_target, packing_loss = \
            self.generator_density_loss(data, generated_samples, config.dataDims)

        # '''
        # special one-time overlap loss ( just calling it the density loss )
        # '''
        # density_prediction = overlaps_list[:,2,0].cpu().detach().numpy()
        # density_target = torch.ones_like(overlaps_list[:,2,0]).cpu().detach().numpy()
        # density_loss = F.smooth_l1_loss(overlaps_list[:,2,0].cuda(), torch.ones_like(overlaps_list[:,2,0]).cuda(),reduction='none')

        if supercell_data is not None:
            return_supercell_data = supercell_data
        else:
            return_supercell_data = None

        return discriminator_score, generated_samples.cpu().detach().numpy(), \
               density_loss, density_prediction, \
               density_target, packing_loss, \
               g2_loss, dist_dict, \
               return_supercell_data, similarity_penalty

    def regression_loss(self, generator, data):
        predictions = generator(data.to(generator.device))[:, 0]
        targets = data.y
        return F.smooth_l1_loss(predictions, targets, reduction='none'), predictions, targets

    def generator_density_loss(self, data, raw_sample, dataDims, precomputed_volumes=None):
        # compute proposed density and evaluate loss against known volume

        gen_packing = []
        for i in range(len(raw_sample)):
            if precomputed_volumes is None:
                volume = cell_vol_torch(data.cell_params[i, 0:3], data.cell_params[i, 3:6])  # torch.prod(cell_lengths[i]) # toss angles #
            else:
                volume = precomputed_volumes[i]

            coeff = data.Z[i] * data.tracking[i, self.mol_volume_ind] / volume
            std_coeff = (coeff - self.dataDims['target mean']) / self.dataDims['target std']

            # volumes.append(volume)
            gen_packing.append(std_coeff)  # compute packing index from given volume and restandardize

        generated_packing_coefficients = torch.stack(gen_packing)
        # generated_packing_coefficients = raw_sample[:,0]
        csd_packing_coefficients = data.y

        # den_loss = F.smooth_l1_loss(generated_packing_coefficients, csd_packing_coefficients, reduction='none') # raw value
        # den_loss = torch.abs(torch.sqrt(F.smooth_l1_loss(generated_packing_coefficients, csd_packing_coefficients, reduction='none'))) # abs(sqrt()) is a soft rescaling to avoid gigantic losses
        den_loss = torch.log(1 + F.smooth_l1_loss(generated_packing_coefficients, csd_packing_coefficients, reduction='none'))  # log(1+loss) is a better soft rescaling to avoid gigantic losses
        #cutoff = (0.6 - self.dataDims['target mean']) / self.dataDims['target std']  # cutoff (0.55) in standardized basis
        #packing_loss = F.mse_loss(-(generated_packing_coefficients - cutoff))  # linear loss below a cutoff
        packing_loss = F.mse_loss(generated_packing_coefficients, torch.zeros_like(generated_packing_coefficients),reduction='none')  # since the data is standardized, we want it to regress towards 0 (the mean)


        if self.config.test_mode:
            assert torch.sum(torch.isnan(packing_loss)) == 0
            assert torch.sum(torch.isnan(den_loss)) == 0

        return den_loss, generated_packing_coefficients.cpu().detach().numpy(), csd_packing_coefficients.cpu().detach().numpy(), packing_loss

    def flow_iter(self, generator, data):  # todo update cell_params norming
        '''
        train the generator via standard normalizing flow loss
        # todo note that this approach will push to model toward good **raw** samples, not accounting for cell 'cleaning'
        '''
        zs, prior_logprob, log_det = generator.nf_forward(data.y[0], conditions=data)
        logprob = prior_logprob + log_det  # log probability of samples, which we want to maximize

        return -(logprob)  #

    def max_prob_iter(self, generator, data):
        samples = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device))
        latent_samples = self.randn_generator.backward(samples.cpu())
        log_probs = self.randn_generator.score(latent_samples)

        return -log_probs  # want to maximize this objective

    def train_boilerplate(self):
        config = self.config
        # dataset
        dataset_builder = BuildDataset(config, pg_dict=self.point_groups,
                                       sg_dict=self.space_groups,
                                       lattice_dict=self.lattice_type,
                                       premade_dataset=self.prep_dataset)
        del self.prep_dataset  # we don't actually want this huge thing floating around

        self.dataDims = dataset_builder.get_dimension()
        config.dataDims = self.dataDims

        self.mol_volume_ind = self.dataDims['tracking features dict'].index('molecule volume')
        self.crystal_packing_ind = self.dataDims['tracking features dict'].index('crystal packing coefficient')
        self.crystal_density_ind = self.dataDims['tracking features dict'].index('crystal calculated density')
        self.mol_size_ind = self.dataDims['tracking features dict'].index('molecule num atoms')
        self.pg_ind_dict = {thing[14:]: ind + self.dataDims['n atomwise features'] for ind, thing in enumerate(self.dataDims['mol features']) if 'pg is' in thing}
        self.sg_ind_dict = {thing[14:]: ind + self.dataDims['n atomwise features'] for ind, thing in enumerate(self.dataDims['mol features']) if 'sg is' in thing}  # todo simplify - allow all possibilities
        self.crysys_ind_dict = {thing[18:]: ind + self.dataDims['n atomwise features'] for ind, thing in enumerate(self.dataDims['mol features']) if 'crystal system is' in thing}

        self.sym_info['pg_ind_dict'] = self.pg_ind_dict
        self.sym_info['sg_ind_dict'] = self.sg_ind_dict
        self.sym_info['crysys_ind_dict'] = self.crysys_ind_dict

        ''' independent gaussian model to approximate the problem
        '''
        self.randn_generator = independent_gaussian_model(input_dim=self.dataDims['n crystal features'],
                                                          means=self.dataDims['means'],
                                                          stds=self.dataDims['stds'],
                                                          cov_mat=self.dataDims['cov mat'])

        return config, dataset_builder


    def MCMC_sampling(self, generator, discriminator, test_loader):
        '''
        Stun MC annealing on a pretrained discriminator
        '''
        from torch_geometric.loader.dataloader import Collater
        from STUN_MC import Sampler

        raw_scores = []
        samples = []
        too_close_frac = []
        volumes = []
        rdfs = []
        n_samples = 100
        single_mol_data = test_loader.dataset[0]
        # set single_mol_data to a bunch of copies of just the first molecule
        collater = Collater(None, None)
        single_mol_data = collater([single_mol_data for n in range(n_samples)])
        self.randn_generator.cpu()

        '''
        initialize sampler
        '''
        real_params = single_mol_data[0].cell_params
        smc_sampler = Sampler(
            gammas=np.logspace(-5, -1, n_samples),
            seedInd=0,
            acceptance_mode='stun',
            debug=True,
            init_temp=1,
        )

        '''
        random batch of initial conditions
        '''
        init_samples = self.randn_generator.forward(n_samples).to(generator.device)
        init_samples = generated_samples * self.randn_generator.stds.to(generator.device) + self.randn_generator.means.to(generator.device)

        '''
        run sampling
        '''
        sampling_dict = smc_sampler(discriminator, single_mol_data, init_samples)

        '''
        analyze outputs
        '''
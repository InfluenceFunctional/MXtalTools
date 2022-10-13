import torch
import wandb
from utils import *
import glob
from model_utils import *
from dataset_management.dataset_manager import Miner
from torch import backends, optim
from dataset_utils import BuildDataset, get_dataloaders, update_batch_size, get_extra_test_loader
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from nikos.coordinate_transformations import coor_trans, cell_vol
from pyxtal import symmetry
from ase import Atoms
import rdkit.Chem as Chem
from crystal_builder_tools import *
from models.generator_models import crystal_generator
from models.discriminator_models import crystal_discriminator
from models.regression_models import molecule_regressor
import ase.io
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from supercell_builders import SupercellBuilder
from STUN_MC import Sampler
from plotly.colors import n_colors


class Modeller():
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

        '''
        load lots of relevant physical data
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
        if config.g_model_path is not None:
            g_checkpoint = torch.load(config.g_model_path)
            config.generator = Namespace(**g_checkpoint['config'])  # overwrite the settings for the model
        if config.d_model_path is not None:
            d_checkpoint = torch.load(config.d_model_path)
            config.discriminator = Namespace(**d_checkpoint['config'])
        print("Initializing models for " + config.mode)
        if config.mode == 'gan':
            generator = crystal_generator(config, dataDims)
            discriminator = crystal_discriminator(config, dataDims)
        elif config.mode == 'regression':
            generator = molecule_regressor(config, dataDims)
            discriminator = nn.Linear(1, 1)  # dummy model
        else:
            print(f'{config.mode} is not an implemented method!')
            sys.exit()

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

        if config.g_model_path is not None:
            if list(g_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(g_checkpoint['model_state_dict']):
                    g_checkpoint['model_state_dict'][i[7:]] = g_checkpoint['model_state_dict'].pop(i)

            generator.load_state_dict(g_checkpoint['model_state_dict'])
            g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])

        if config.d_model_path is not None:
            if list(d_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(d_checkpoint['model_state_dict']):
                    d_checkpoint['model_state_dict'][i[7:]] = d_checkpoint['model_state_dict'].pop(i)
            discriminator.load_state_dict(d_checkpoint['model_state_dict'])
            d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])

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

    def get_batch_size(self, generator, discriminator, g_optimizer, d_optimizer, dataset, config):
        '''
        try larger batches until it crashes
        '''
        finished = False
        init_batch_size = config.min_batch_size.real
        max_batch_size = config.max_batch_size.real
        batch_reduction_factor = config.auto_batch_reduction

        train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=init_batch_size)

        increment = 1.5  # what fraction by which to increment the batch size
        batch_size = int(init_batch_size)

        while (not finished) and (batch_size < max_batch_size):
            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU cache
                generator.cuda()
                discriminator.cuda()

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

        # '''
        # slow cell-by-cell analysis & rebuild
        # '''
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
        sgs = []
        csd_energies = []
        random_energies = []
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            # build supercells from the dataset, and compute their properties
            csd_supercells = self.supercell_builder.build_supercells_from_dataset(data.clone(), config, return_overlaps=False, supercell_inclusion_level='ref mol')

            csd_rdf_i, rr = crystal_rdf(csd_supercells, rrange=[0, 10], bins=100, intermolecular=True)

            csd_rdf.extend(csd_rdf_i.cpu().detach().numpy())
            # csd_overlaps.extend(csd_overlaps_i)

            rebuild_supercells, vol, rebuild_overlaps_i = \
                self.supercell_builder.build_supercells(data.clone(), None,
                                                        config.supercell_size, config.discriminator.graph_convolution_cutoff,
                                                        skip_cell_cleaning=True, ref_data=data.clone(), debug=True, supercell_inclusion_level='ref mol')
            # csd_energies.append(en)
            rebuild_supercells = rebuild_supercells.cpu()

            rebuild_rdf_i, rr = crystal_rdf(rebuild_supercells, rrange=[0, 10], bins=100, intermolecular=True)
            rebuild_rdf.extend(rebuild_rdf_i.cpu().detach().numpy())
            rebuild_overlaps.extend(rebuild_overlaps_i.cpu().detach().numpy())

            sgs.append(rebuild_supercells.sg_ind.cpu().detach().numpy())

            if any(torch.sum(torch.abs(rebuild_rdf_i.to(csd_rdf_i.device) - csd_rdf_i), dim=1) / torch.sum(csd_rdf_i, dim=1) > 0.05):
                aa = 1
                # #examine
                # plt.clf()
                # diff = torch.sum(torch.abs(rebuild_rdf_i.to(csd_rdf_i.device) - csd_rdf_i), dim=1) / torch.sum(csd_rdf_i, dim=1)
                # ind = torch.argsort(diff)[-1]
                # plt.plot(csd_rdf_i[ind].cpu().detach())
                # plt.plot(rebuild_rdf_i[ind].cpu().detach())
                # mols = [ase_mol_from_crystaldata(csd_supercells.cpu(), ind.cpu().detach(), exclusion_level='convolve with'),
                #
                #         ase_mol_from_crystaldata(rebuild_supercells.cpu(), ind.cpu().detach(), exclusion_level='convolve with')]
                # from ase.visualize import view
                # view(mols)

            # assert torch.mean(torch.sum(torch.abs(rebuild_rdf_i.to(csd_rdf_i.device) - csd_rdf_i),dim=1) / torch.sum(csd_rdf_i,dim=1)) < 0.001
            #
            # # build random supercells and compute their properties
            # random_supercells, vol, random_overlaps_i = \
            #     self.supercell_builder.build_supercells(data.clone(), self.randn_generator(data, data.num_graphs),
            #                                             config.supercell_size, config.discriminator.graph_convolution_cutoff, return_energy = False)
            # #random_energies.append(en)
            # random_supercells = random_supercells.cpu()
            # random_rdf_i, rr = crystal_rdf(random_supercells, rrange=[0, 10], bins=100, intermolecular=True)
            #
            # random_rdf.extend(random_rdf_i.cpu().detach().numpy())
            # random_overlaps.extend(random_overlaps_i.cpu().detach().numpy())

        sgs = np.concatenate(sgs)
        csd_rdf = np.stack(csd_rdf)
        rebuild_rdf = np.stack(rebuild_rdf)
        # random_rdf = np.stack(random_rdf)

        print(f'RDF error of {np.sum(np.abs(csd_rdf - rebuild_rdf)) / np.sum(csd_rdf):.3f}')
        diff = np.sum(np.abs(csd_rdf - rebuild_rdf), axis=1) / np.sum(csd_rdf, axis=1)
        frac_correct = np.mean(diff < 0.05)
        print(f'RDF {frac_correct:.3f} accurate on a per-sample basis at 95%')

        frac_correct = np.mean(diff < 0.01)
        print(f'RDF {frac_correct:.3f} accurate on a per-sample basis at 99%')
        # print(f'vs a random divergence of {np.sum(np.abs(csd_rdf - random_rdf)) / np.sum(csd_rdf):.3f}')
        stop = 1

    def train(self):
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=[self.config.wandb.experiment_tag]):
            # config = wandb.config # todo: wandb configs don't support nested namespaces. Sweeps are officially broken - look at the github thread

            config, dataset_builder = self.train_boilerplate()
            generator, discriminator, g_optimizer, g_schedulers, d_optimizer, d_schedulers, params1, params2 \
                = self.init_gan(config, self.dataDims)  # todo change this to just resetting all the parameters of existing models

            # get batch size
            if config.auto_batch_sizing:
                print('Finding optimal batch size')
                train_loader, test_loader, config.final_batch_size = self.get_batch_size(generator, discriminator, g_optimizer, d_optimizer,
                                                                                         dataset_builder, config)
            else:
                print('Getting dataloaders for pre-determined batch size')
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.max_batch_size
            del dataset_builder

            if config.extra_test_set_paths is not None:
                extra_test_loader = get_extra_test_loader(config, config.extra_test_set_paths, dataDims=self.dataDims,
                                                          pg_dict=self.point_groups, sg_dict=self.space_groups, lattice_dict=self.lattice_type)

            print("Training batch size set to {}".format(config.final_batch_size))
            # model, optimizer, schedulers
            print('Reinitializing model and optimizer')
            generator.apply(weight_reset)
            discriminator.apply(weight_reset)
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
            d_hit_max_lr, g_hit_max_lr, converged, epoch = False, False, config.max_epochs == 0, 0  # for evaluation mode
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
                    extra_test_epoch_stats_dict = None
                    try:
                        d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                            self.epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                       g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                       update_gradients=True, record_stats=True, epoch=epoch)  # train & compute test loss

                        with torch.no_grad():
                            d_err_te, d_te_record, g_err_te, g_te_record, test_epoch_stats_dict, time_test = \
                                self.epoch(config, dataLoader=test_loader, generator=generator, discriminator=discriminator,
                                           update_gradients=False, record_stats=True, epoch=epoch)  # compute loss on test set

                            if config.extra_test_set_paths is not None:
                                if epoch % config.extra_test_period == 0:
                                    extra_test_epoch_stats_dict, time_test_ex = \
                                        self.discriminator_evaluation(config, dataLoader=extra_test_loader, discriminator=discriminator)  # compute loss on test set
                                    print(f'Extra test evaluation took {time_test_ex:.1f} seconds')
                            else:
                                extra_test_epoch_stats_dict = None

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
                            epoch, metrics_dict,
                            d_err_tr, d_err_te,
                            g_err_tr, g_err_te,
                            d_learning_rate, g_learning_rate)

                        self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                                          d_tr_record, d_te_record, g_tr_record, g_te_record)

                        if epoch % config.wandb.sample_reporting_frequency == 0:
                            self.log_gan_accuracy(epoch, train_loader, test_loader,
                                                  metrics_dict, g_tr_record, g_te_record, d_tr_record, d_te_record,
                                                  train_epoch_stats_dict, test_epoch_stats_dict, config,
                                                  generator, discriminator, wandb_log_figures=config.wandb.log_figures,
                                                  extra_test_dict=extra_test_epoch_stats_dict)

                        '''
                        save model if best
                        '''  # todo add more sophisticated convergence stat
                        if epoch > 0:
                            if np.average(d_err_te) < np.amin(metrics_dict['discriminator test loss'][:-1]):  # todo fix this
                                print("Saving discriminator checkpoint")
                                save_checkpoint(epoch, discriminator, d_optimizer, config.discriminator.__dict__, 'discriminator_' + str(config.run_num))
                            if np.average(g_err_te) < np.amin(metrics_dict['generator test loss'][:-1]):
                                print("Saving generator checkpoint")
                                save_checkpoint(epoch, generator, g_optimizer, config.generator.__dict__, 'generator_' + str(config.run_num))

                        '''
                        convergence checks
                        '''
                        generator_convergence = checkConvergence(metrics_dict['generator test loss'], config.history, config.generator.convergence_eps)
                        discriminator_convergence = checkConvergence(metrics_dict['discriminator test loss'], config.history, config.discriminator.convergence_eps)

                        if generator_convergence:
                            print('generator converged!')
                        if discriminator_convergence:
                            print('discriminator converged!')

                        if (generator_convergence and discriminator_convergence) and (epoch > config.history + 2):
                            print('Training has converged!')
                            config.finished = True
                            break

                        if epoch % 5 == 0:
                            if train_loader.batch_size < len(train_loader.dataset):  # if the batch is smaller than the dataset
                                increment = max(4, int(train_loader.batch_size * 0.05))  # increment batch size
                                train_loader = update_batch_size(train_loader, train_loader.batch_size + increment)
                                test_loader = update_batch_size(test_loader, test_loader.batch_size + increment)
                                if config.extra_test_set_paths is not None:
                                    extra_test_loader = update_batch_size(extra_test_loader, extra_test_loader.batch_size + increment)
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

                    if config.device.lower() == 'cuda':
                        torch.cuda.empty_cache()  # clear GPU

            '''
            run post-training evaluation
            '''
            with torch.no_grad():
                test_epoch_stats_dict, time_test_ex = \
                    self.discriminator_evaluation(config, dataLoader=test_loader, discriminator=discriminator)  # compute loss on test set

                extra_test_epoch_stats_dict, time_test_ex = \
                    self.discriminator_evaluation(config, dataLoader=extra_test_loader, discriminator=discriminator)  # compute loss on test set

            self.log_discriminator_analysis(epoch, test_loader, extra_test_loader,
                                            metrics_dict,
                                            test_epoch_stats_dict, config,
                                            extra_test_epoch_stats_dict)

            # for working with a trained model
            if config.sample_after_training:
                sampling_results = self.MCMC_sampling(generator, discriminator, test_loader)

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
        epoch_stats_dict['generator density prediction'] = []
        epoch_stats_dict['generator density target'] = []
        epoch_stats_dict = {
            'tracking features': [],
        }

        for i, data in enumerate(tqdm.tqdm(dataLoader)):
            '''
            noise injection
            '''
            if config.generator.positional_noise > 0:
                data.pos += torch.randn_like(data.pos) * config.generator.positional_noise

            regression_losses_list, predictions, targets = self.regression_loss(generator, data)
            epoch_stats_dict['generator density prediction'].append(predictions.cpu().detach().numpy())
            epoch_stats_dict['generator density target'].append(targets.cpu().detach().numpy())

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
        epoch_stats_dict['tracking features'] = np.stack(epoch_stats_dict['tracking features'])

        if record_stats:
            epoch_stats_dict['generator density prediction'] = np.concatenate(epoch_stats_dict['generator density prediction']) if epoch_stats_dict['generator density prediction'] != [] else None
            epoch_stats_dict['generator density target'] = np.concatenate(epoch_stats_dict['generator density target']) if epoch_stats_dict['generator density target'] != [] else None
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
        g_err = []
        g_loss_record = []
        g_flow_err = []

        epoch_stats_dict = {
            'tracking features': [],
            'identifiers': [],
            'discriminator real score': [],
            'discriminator fake score': [],
            'generator density loss': [],
            'generator adversarial score': [],
            'generator flow loss': [],
            'generator short range loss': [],
            'generator packing loss': [],
            'generator density prediction': [],
            'generator density target': [],
            'generator similarity loss': [],
            'generator intra distance hist': [],
            'generator inter distance hist': [],
            'real intra distance hist': [],
            'real inter distance hist': [],
            'generated cell parameters': [],
            'final generated cell parameters': [],
            'generated supercell examples dict': [],
            'generator sample source': [],
        }

        generated_supercell_examples_dict = {}

        rand_batch_ind = np.random.randint(0, len(dataLoader))

        for i, data in enumerate(tqdm.tqdm(dataLoader)):
            '''
            train flowmodel first just in case it's initialization is too wild for the discriminator
            '''
            if config.train_generator_as_flow:
                if (epoch < config.cut_max_prob_training_after):  # stop using max_prob training after a few initial epochs

                    g_flow_losses = self.flow_iter(generator, data.clone().to(config.device))

                    g_flow_loss = g_flow_losses.mean()
                    g_flow_err.append(g_flow_loss.data.cpu())  # average loss
                    epoch_stats_dict['generator flow loss'].append(g_flow_losses.cpu().detach().numpy())  # loss distribution

                    if update_gradients:
                        g_optimizer.zero_grad()  # reset gradients from previous passes
                        g_flow_loss.backward()  # back-propagation
                        g_optimizer.step()  # update parameters

            '''
            train discriminator
            '''
            if epoch % config.discriminator.training_period == 0:  # only train the discriminator every XX epochs
                if config.train_discriminator_adversarially or config.train_discriminator_on_noise or config.train_discriminator_on_randn:
                    generated_samples_i, handedness, epoch_stats_dict = self.generate_discriminator_negatives(epoch_stats_dict, config, data, generator, i)

                    score_on_real, score_on_fake, generated_samples, real_dist_dict, fake_dist_dict \
                        = self.train_discriminator(generated_samples_i, discriminator, config, data, i, handedness)  # alternately trains on real and fake samples

                    epoch_stats_dict['discriminator real score'].extend(score_on_real.cpu().detach().numpy())
                    epoch_stats_dict['discriminator fake score'].extend(score_on_fake.cpu().detach().numpy())

                    if config.gan_loss == 'wasserstein':
                        d_losses = -score_on_real + score_on_fake  # maximize score on real, minimize score on fake

                    elif config.gan_loss == 'standard':
                        prediction = torch.cat((score_on_real, score_on_fake))
                        target = torch.cat((torch.ones_like(score_on_real[:, 0]), torch.zeros_like(score_on_fake[:, 0])))
                        d_losses = F.cross_entropy(prediction, target.long(), reduction='none')  # works much better

                    else:
                        print(config.gan_loss + ' is not an implemented GAN loss function!')
                        sys.exit()

                    d_loss = d_losses.mean()
                    d_err.append(d_loss.data.cpu().detach().numpy())  # average overall loss
                    d_loss_record.extend(d_losses.cpu().detach().numpy())  # overall loss distribution

                    if update_gradients:
                        d_optimizer.zero_grad()  # reset gradients from previous passes
                        d_loss.backward()  # back-propagation
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.gradient_norm_clip)  # gradient clipping
                        d_optimizer.step()  # update parameters

                    # generated_intra_dist.append(fake_dist_dict['intramolecular dist'].cpu().detach().numpy())
                    # generated_inter_dist.append(fake_dist_dict['intermolecular dist'].cpu().detach().numpy())
                    # real_intra_dist.append(real_dist_dict['intramolecular dist'].cpu().detach().numpy())
                    # real_inter_dist.append(real_dist_dict['intermolecular dist'].cpu().detach().numpy())

                    epoch_stats_dict['generated cell parameters'].extend(generated_samples_i.cpu().detach().numpy())
                    epoch_stats_dict['final generated cell parameters'].extend(generated_samples)

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

                epoch_stats_dict['generator density prediction'].append(density_prediction)
                epoch_stats_dict['generator density target'].append(density_target)
                g_losses_list = []
                if config.train_generator_density:
                    g_losses_list.append(density_loss.float())
                    epoch_stats_dict['generator density loss'].append(density_loss.cpu().detach().numpy())

                if config.train_generator_adversarially:
                    g_losses_list.append(adversarial_loss)
                    epoch_stats_dict['generator adversarial score'].append(adversarial_score.cpu().detach().numpy())

                if config.train_generator_g2:
                    g_losses_list.append(g2_loss)
                    epoch_stats_dict['generator short range loss'].append(g2_loss.cpu().detach().numpy())

                if config.train_generator_packing:
                    g_losses_list.append(packing_loss)
                    epoch_stats_dict['generator packing loss'].append(packing_loss.cpu().detach().numpy())

                if config.generator_similarity_penalty != 0:
                    if similarity_penalty is not None:
                        g_losses_list.append(similarity_penalty)
                        epoch_stats_dict['generator similarity loss'].append(similarity_penalty.cpu().detach().numpy())
                    else:
                        print('similarity penalty was none')

                if config.train_generator_adversarially or config.train_generator_g2:
                    print('deprecated!')
                    sys.exit()
                    # generated_intra_dist.append(generated_dist_dict['intramolecular dist'].cpu().detach().numpy())
                    # generated_inter_dist.append(generated_dist_dict['intermolecular dist'].cpu().detach().numpy())

                g_losses = torch.sum(torch.stack(g_losses_list), dim=0)

                g_loss = g_losses.mean()
                g_err.append(g_loss.data.cpu().detach().numpy())  # average loss
                g_loss_record.extend(g_losses.cpu().detach().numpy())  # loss distribution
                epoch_stats_dict['generated cell parameters'].append(generated_samples)

                if update_gradients:
                    g_optimizer.zero_grad()  # reset gradients from previous passes
                    g_loss.backward()  # back-propagation
                    g_optimizer.step()  # update parameters
            else:
                g_err.append(np.zeros(1))
                g_loss_record.extend(np.zeros(data.num_graphs))

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

            if (len(epoch_stats_dict['generated cell parameters']) < i) and record_stats:  # make some samples for analysis if we have none so far from this step
                generated_samples = generator(len(data.y), z=None, conditions=data.to(config.device))
                epoch_stats_dict['generated cell parameters'].append(generated_samples.cpu().detach().numpy())

            if record_stats:
                epoch_stats_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())
                epoch_stats_dict['identifiers'].extend(data.csd_identifier)

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0

        # epoch_stats_dict['generated intra distance hist'] = np.histogram(np.concatenate(generated_intra_dist), bins=100, range=(0, config.discriminator.graph_convolution_cutoff), density=True) if generated_intra_dist != [] else None
        # epoch_stats_dict['generated inter distance hist'] = np.histogram(np.concatenate(generated_inter_dist), bins=100, range=(0, config.discriminator.graph_convolution_cutoff), density=True) if generated_inter_dist != [] else None
        # epoch_stats_dict['real intra distance hist'] = np.histogram(np.concatenate(real_intra_dist), bins=100, range=(0, config.discriminator.graph_convolution_cutoff), density=True) if real_intra_dist != [] else None
        # epoch_stats_dict['real inter distance hist'] = np.histogram(np.concatenate(real_inter_dist), bins=100, range=(0, config.discriminator.graph_convolution_cutoff), density=True) if real_inter_dist != [] else None

        if record_stats:
            for key in epoch_stats_dict.keys():
                if 'supercell' not in key:
                    feature = epoch_stats_dict[key]
                    if (feature == []) or (feature is None):
                        epoch_stats_dict[key] = None
                    else:
                        epoch_stats_dict[key] = np.asarray(feature)

            epoch_stats_dict['generated supercell examples dict'] = generated_supercell_examples_dict if generated_supercell_examples_dict != {} else None

            return d_err, d_loss_record, g_err, g_loss_record, epoch_stats_dict, total_time
        else:
            return d_err, d_loss_record, g_err, g_loss_record, total_time

    def discriminator_evaluation(self, config, dataLoader=None, discriminator=None, iteration_override=None, compute_LJ_energy=False):
        t0 = time.time()
        discriminator.eval()

        epoch_stats_dict = {
            'tracking features': [],
            'identifiers': [],
            'scores': [],
            'intermolecular rdf': [],
            'atomistic energy': [],
            'full rdf': [],
        }

        for i, data in enumerate(tqdm.tqdm(dataLoader)):
            '''
            evaluate discriminator
            '''
            if compute_LJ_energy:  # only compute LJ energy on the first run or when specifically asked
                real_supercell_data, atomwise_energy = self.supercell_builder.build_supercells_from_dataset(data.clone(), config, return_energy=True)
            else:
                real_supercell_data = self.supercell_builder.build_supercells_from_dataset(data.clone(), config, return_energy=False)

            if config.device.lower() == 'cuda':  # redundant
                real_supercell_data = real_supercell_data.cuda()

            if config.test_mode or config.anomaly_detection:
                assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"

            score_on_real, real_distances_dict = self.adversarial_loss(discriminator, real_supercell_data, config)

            full_rdfs, rr, self.elementwise_correlations_labels = crystal_rdf(real_supercell_data, elementwise=True, raw_density=True)
            intermolecular_rdfs, rr, _ = crystal_rdf(real_supercell_data, intermolecular=True, elementwise=True, raw_density=True)

            epoch_stats_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())
            epoch_stats_dict['identifiers'].extend(data.csd_identifier)  #
            epoch_stats_dict['scores'].extend(score_on_real.cpu().detach().numpy())
            epoch_stats_dict['intermolecular rdf'].extend(intermolecular_rdfs.cpu().detach().numpy())
            epoch_stats_dict['full rdf'].extend(full_rdfs.cpu().detach().numpy())
            if compute_LJ_energy:
                epoch_stats_dict['atomistic energy'].extend(atomwise_energy)

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        epoch_stats_dict['scores'] = np.stack(epoch_stats_dict['scores'])
        epoch_stats_dict['tracking features'] = np.stack(epoch_stats_dict['tracking features'])
        epoch_stats_dict['full rdf'] = np.stack(epoch_stats_dict['full rdf'])
        epoch_stats_dict['intermolecular rdf'] = np.stack(epoch_stats_dict['intermolecular rdf'])
        if compute_LJ_energy:
            epoch_stats_dict['atomistic energy'] = np.asarray(epoch_stats_dict['atomistic energy'])
        else:
            epoch_stats_dict['atomistic energy'] = None

        total_time = time.time() - t0

        return epoch_stats_dict, total_time

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

        assert torch.sum(torch.isnan(scores)) == 0

        return scores, extra_outputs['dists dict']

    def pairwise_correlations_analysis(self, dataset_builder, config):
        '''
        check correlations in the data
        :param dataset_builder:
        :param config:
        :return:
        '''
        data = dataset_builder.datapoints
        keys = self.dataDims['lattice features']
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
        for i in range(dataDims['num lattice features']):
            renormalized_targets[:, i] = targets[:, i] * dataDims['lattice stds'][i] + dataDims['lattice means'][i]

        targets_rep = np.repeat(renormalized_targets[:, None, :], samples.shape[1], axis=1)
        # denominator = np.repeat(np.repeat(np.quantile(renormalized_targets,0.95,axis=0)[None,None,:],samples.shape[0],axis=0),samples.shape[1],axis=1)
        denominator = targets_rep.copy()
        for i in range(dataDims['num lattice features']):
            if dataDims['lattice dtypes'][i] == 'bool':
                denominator[:, :, i] = 1

        errors = np.abs((targets_rep - samples) / denominator)
        feature_mae = np.mean(errors, axis=(0, 1))

        for i in range(dataDims['num lattice features']):
            feature_accuracy_dict[sampler + ' ' + dataDims['lattice features'][i] + ' mae'] = feature_mae[i]
            for cutoff in [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3]:
                feature_accuracy_dict[sampler + ' ' + dataDims['lattice features'][i] + ' efficiency at {}'.format(cutoff)] = np.average(errors[:, :, i] < cutoff)

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

    def log_gan_loss(self, metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                     d_tr_record, d_te_record, g_tr_record, g_te_record):
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
                if self.config.gan_loss == 'wasserstein':
                    score = train_epoch_stats_dict[key]
                elif self.config.gan_loss == 'standard':
                    score = np_softmax(train_epoch_stats_dict[key])[:, 0]
                special_losses['Train ' + key] = np.average(score)
            if ('score' in key) and (test_epoch_stats_dict[key] is not None):
                if self.config.gan_loss == 'wasserstein':
                    score = test_epoch_stats_dict[key]
                elif self.config.gan_loss == 'standard':
                    score = np_softmax(test_epoch_stats_dict[key])[:, 0]
                special_losses['Test ' + key] = np.average(score)
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
                         generator, discriminator, wandb_log_figures=True,
                         extra_test_dict=None):
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

            # # correlate losses with molecular features
            # tracking_features = np.asarray(test_epoch_stats_dict['tracking features'])
            # g_loss_correlations = np.zeros(config.dataDims['num tracking features'])
            # d_loss_correlations = np.zeros(config.dataDims['num tracking features'])
            # features = []
            # for i in range(config.dataDims['num tracking features']):  # not that interesting
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
            #         y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['num tracking features'])],
            #         x=[g_loss_correlations[i] for i in range(config.dataDims['num tracking features'])],
            #         orientation='h',
            #     ))
            #     wandb.log({'G Loss correlations': fig})
            #
            #     fig = go.Figure(go.Bar(
            #         y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['num tracking features'])],
            #         x=[d_loss_correlations[i] for i in range(config.dataDims['num tracking features'])],
            #         orientation='h',
            #     ))
            #     wandb.log({'D Loss correlations': fig})

            '''
            cell parameter analysis
            '''
            if train_epoch_stats_dict['generated cell parameters'] is not None:
                n_crystal_features = config.dataDims['num lattice features']
                generated_samples = test_epoch_stats_dict['generated cell parameters']
                means = config.dataDims['lattice means']
                stds = config.dataDims['lattice stds']

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
                for i, key in enumerate(config.dataDims['lattice features']):
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

                        fig_dict[self.dataDims['lattice features'][i] + ' distribution'] = fig

                    wandb.log(fig_dict)

            # '''
            # cell atomic distances
            # '''
            # if train_epoch_stats_dict['generated inter distance hist'] is not None:  # todo update this
            #     hh2_test, rr = test_epoch_stats_dict['generated inter distance hist']
            #     hh2_train, _ = train_epoch_stats_dict['generated inter distance hist']
            #     if train_epoch_stats_dict['real inter distance hist'] is not None:  # if there is no discriminator training, we don't generate this
            #         hh1, rr = train_epoch_stats_dict['real inter distance hist']
            #     else:
            #         hh1 = hh2_test
            #
            #     shell_volumes = (4 / 3) * torch.pi * ((rr[:-1] + np.diff(rr)) ** 3 - rr[:-1] ** 3)
            #     rdf1 = hh1 / shell_volumes
            #     rdf2 = hh2_test / shell_volumes
            #     rdf3 = hh2_train / shell_volumes
            #     fig = go.Figure()
            #     fig.add_trace(go.Scattergl(x=rr, y=rdf1, name='real'))
            #     fig.add_trace(go.Scattergl(x=rr, y=rdf2, name='gen, test'))
            #     fig.add_trace(go.Scattergl(x=rr, y=rdf3, name='gen, train'))
            #
            #     if config.wandb.log_figures:
            #         wandb.log({'G2 Comparison': fig})
            #
            #     range_analysis_dict = {}
            #     if train_epoch_stats_dict['real inter distance hist'] is not None:  # if there is no discriminator training, we don't generate this
            #         # get histogram overlaps
            #         range_analysis_dict['tr g2 overlap'] = np.min(np.concatenate((rdf1[None, :] / rdf1.sum(), rdf3[None, :] / rdf1.sum()), axis=0), axis=0).sum()
            #         range_analysis_dict['te g2 overlap'] = np.min(np.concatenate((rdf1[None, :] / rdf1.sum(), rdf2[None, :] / rdf1.sum()), axis=0), axis=0).sum()
            #
            #     # get probability mass at too-close range (should be ~zero)
            #     range_analysis_dict['tr short range density fraction'] = np.sum(rdf3[rr[1:] < 1.2] / rdf3.sum())
            #     range_analysis_dict['te short range density fraction'] = np.sum(rdf2[rr[1:] < 1.2] / rdf2.sum())
            #
            #     wandb.log(range_analysis_dict)

        '''
        auxiliary regression target
        '''
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
                fig.add_trace(go.Scattergl(x=orig_target, y=orig_prediction, mode='markers', showlegend=True, opacity=0.5))
                fig.add_trace(go.Scattergl(x=xline, y=xline))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Test Packing Coefficient': fig})

                xline = np.linspace(max(min(train_orig_target), min(train_orig_prediction)), min(max(train_orig_target), max(train_orig_prediction)), 10)
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(x=train_orig_target, y=train_orig_prediction, ncontours=50, nbinsx=40, nbinsy=40, showlegend=True))
                fig.update_traces(contours_coloring="fill")
                fig.update_traces(contours_showlines=False)
                fig.add_trace(go.Scattergl(x=train_orig_target, y=train_orig_prediction, mode='markers', showlegend=True, opacity=0.5))
                fig.add_trace(go.Scattergl(x=xline, y=xline))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Train Packing Coefficient': fig})

                fig = go.Figure()
                fig.add_trace(go.Histogram(x=train_orig_prediction - train_orig_target,
                                           histnorm='probability density',
                                           nbinsx=100,
                                           name="Error Distribution",
                                           showlegend=False))
                wandb.log({'Regression Error Distribution': fig})

                # correlate losses with molecular features
                tracking_features = np.asarray(test_epoch_stats_dict['tracking features'])
                g_loss_correlations = np.zeros(config.dataDims['num tracking features'])
                features = []
                for i in range(config.dataDims['num tracking features']):  # not that interesting
                    features.append(config.dataDims['tracking features dict'][i])
                    g_loss_correlations[i] = np.corrcoef(np.abs((orig_target - orig_prediction) / np.abs(orig_target)), tracking_features[:, i], rowvar=False)[0, 1]

                g_sort_inds = np.argsort(g_loss_correlations)
                g_loss_correlations = g_loss_correlations[g_sort_inds]

                fig = go.Figure(go.Bar(
                    y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['num tracking features'])],
                    x=[g_loss_correlations[i] for i in range(config.dataDims['num tracking features'])],
                    orientation='h',
                ))
                wandb.log({'Regressor Loss Correlates': fig})

        if (extra_test_dict is not None) and (epoch % config.extra_test_period == 0):
            # need to identify targets
            # need to distinguish between experimental and proposed structures
            blind_test_targets = [  # 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
                'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', ]

            bt_lj_energy_dict = np.load('../datasets/BT_LJ_energies.npy', allow_pickle=True).item()
            extra_test_dict['atomistic energy'] = np.asarray([bt_lj_energy_dict[ident] for ident in extra_test_dict['identifiers']])

            '''
            determine which samples go with which targets
            '''
            all_identifiers = {key: [] for key in blind_test_targets}
            for i in range(len(extra_test_dict['identifiers'])):
                item = extra_test_dict['identifiers'][i]
                for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
                    if blind_test_targets[-1 - j] in item:
                        all_identifiers[blind_test_targets[-1 - j]].append(i)
                        break

            '''
            get the identifiers for all the BT targets
            '''
            # CSD identifiers for the blind test targets
            target_identifiers = {
                'XVI': 'OBEQUJ',
                'XVII': 'OBEQOD',
                'XVIII': 'OBEQET',
                'XIX': 'XATJOT',
                'XX': 'OBEQIX',
                'XXI': 'KONTIQ',
                'XXII': 'NACJAF',
                'XXIII': 'XAFPAY',
                'XXIV': 'XAFQON',
                'XXVI': 'XAFQIH'
            }

            target_identifiers_inds = {key: [] for key in blind_test_targets}
            for i in range(len(extra_test_dict['identifiers'])):
                item = extra_test_dict['identifiers'][i]
                for key in target_identifiers.keys():
                    if item == target_identifiers[key]:
                        target_identifiers_inds[key] = i

            '''
            record all the stats for the CSD data
            '''
            scores_dict = {}
            # nf_inds = np.where(test_epoch_stats_dict['generator sample source'] == 0)
            randn_inds = np.where(test_epoch_stats_dict['generator sample source'] == 1)
            distorted_inds = np.where(test_epoch_stats_dict['generator sample source'] == 2)

            if self.config.gan_loss == 'wasserstein':
                scores_dict['Train Real'] = train_epoch_stats_dict['discriminator real score']
                scores_dict['Test Real'] = test_epoch_stats_dict['discriminator real score']
                scores_dict['Test Randn'] = test_epoch_stats_dict['discriminator fake score'][randn_inds]
                # scores_dict['Test NF'] = test_epoch_stats_dict['discriminator fake score'][nf_inds]
                scores_dict['Test Distorted'] = test_epoch_stats_dict['discriminator fake score'][distorted_inds]

            elif self.config.gan_loss == 'standard':
                scores_dict['Train Real'] = np_softmax(train_epoch_stats_dict['discriminator real score'])[:, 1]
                scores_dict['Test Real'] = np_softmax(test_epoch_stats_dict['discriminator real score'])[:, 1]
                scores_dict['Test Randn'] = np_softmax(test_epoch_stats_dict['discriminator fake score'][randn_inds])[:, 1]
                # scores_dict['Test NF'] = np_softmax(test_epoch_stats_dict['discriminator fake score'][nf_inds])[:, 1]
                scores_dict['Test Distorted'] = np_softmax(test_epoch_stats_dict['discriminator fake score'][distorted_inds])[:, 1]

            wandb.log({'Average Train score': np.average(scores_dict['Train Real'])})
            wandb.log({'Average Test score': np.average(scores_dict['Test Real'])})
            wandb.log({'Average Randn Fake score': np.average(scores_dict['Test Randn'])})
            # wandb.log({'Average NF Fake score': np.average(scores_dict['Test NF'])})
            wandb.log({'Average Distorted Fake score': np.average(scores_dict['Test Distorted'])})

            loss_correlations_dict = {}
            rdf_full_distance_dict = {}
            rdf_inter_distance_dict = {}
            energy_dict = {}

            '''
            build property dicts for the submissions and BT targets
            '''
            for target in all_identifiers.keys():  # run the analysis for each target
                if target_identifiers_inds[target] != []:

                    target_index = target_identifiers_inds[target]
                    raw_scores = extra_test_dict['scores'][target_index]
                    if self.config.gan_loss == 'wasserstein':
                        scores = raw_scores
                    elif self.config.gan_loss == 'standard':
                        scores = np_softmax(raw_scores)[:, 1]
                    scores_dict[target + ' exp'] = scores
                    energy_dict[target + ' exp'] = extra_test_dict['atomistic energy'][target_index]

                    wandb.log({f'Average {target} exp score': np.average(scores)})

                    target_full_rdf = extra_test_dict['full rdf'][target_index]
                    target_inter_rdf = extra_test_dict['intermolecular rdf'][target_index]

                if all_identifiers[target] != []:

                    target_indices = all_identifiers[target]
                    raw_scores = extra_test_dict['scores'][target_indices]
                    if self.config.gan_loss == 'wasserstein':
                        scores = raw_scores
                    elif self.config.gan_loss == 'standard':
                        scores = np_softmax(raw_scores)[:, 1]
                    scores_dict[target] = scores
                    energy_dict[target] = extra_test_dict['atomistic energy'][target_indices]

                    wandb.log({f'Average {target} score': np.average(scores)})

                    submission_full_rdf = extra_test_dict['full rdf'][target_indices]
                    submission_inter_rdf = extra_test_dict['intermolecular rdf'][target_indices]

                    # compute distance between target & submission RDFs
                    rr = np.linspace(0, 10, 100)
                    sigma = 1
                    smoothed_target_full_rdf = gaussian_filter1d(target_full_rdf, sigma=sigma)
                    smoothed_target_inter_rdf = gaussian_filter1d(target_inter_rdf, sigma=sigma)
                    smoothed_submission_full_rdf = gaussian_filter1d(np.clip(submission_full_rdf, a_min=0, a_max=10), sigma=sigma)
                    smoothed_submission_inter_rdf = gaussian_filter1d(np.clip(submission_inter_rdf, a_min=0, a_max=10), sigma=sigma)

                    rdf_full_distance_dict[target] = 1 - np.sum(np.minimum(smoothed_target_full_rdf, smoothed_submission_full_rdf), axis=(1, 2)) / np.sum(
                        smoothed_target_full_rdf)  # get histogram overlap over all atom pairs & scale by total target density
                    rdf_inter_distance_dict[target] = 1 - np.sum(np.minimum(smoothed_target_inter_rdf, smoothed_submission_inter_rdf), axis=(1, 2)) / np.sum(smoothed_target_inter_rdf)

                    # correlate losses with molecular features
                    tracking_features = np.asarray(extra_test_dict['tracking features'])
                    loss_correlations = np.zeros(config.dataDims['num tracking features'])
                    features = []
                    for j in range(config.dataDims['num tracking features']):  # not that interesting
                        features.append(config.dataDims['tracking features dict'][j])
                        loss_correlations[j] = np.corrcoef(scores, tracking_features[target_indices, j], rowvar=False)[0, 1]

                    loss_correlations_dict[target] = loss_correlations

            # normed_energy_dict = {key: normalize(energy_dict[key]) for key in energy_dict.keys() if 'exp' not in key}
            normed_energy_dict = {key: standardize(energy_dict[key]) for key in energy_dict.keys() if 'exp' not in key}

            '''
            prep violin figure colors
            '''
            lens = [len(val) for val in all_identifiers.values()]
            colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', max(np.count_nonzero(lens), np.count_nonzero(list(target_identifiers_inds.values()))), colortype='rgb')

            plot_color_dict = {}
            plot_color_dict['Train Real'] = ('rgb(250,50,50)')  # train
            plot_color_dict['Test Real'] = ('rgb(250,150,50)')  # test
            plot_color_dict['Test Randn'] = ('rgb(0,50,0)')  # fake csd
            plot_color_dict['Test NF'] = ('rgb(0,150,0)')  # fake nf
            plot_color_dict['Test Distorted'] = ('rgb(0,100,100)')  # fake distortion
            ind = 0
            for target in all_identifiers.keys():
                if all_identifiers[target] != []:
                    plot_color_dict[target] = colors[ind]
                    plot_color_dict[target + ' exp'] = colors[ind]

                    ind += 1

            '''
            violin scores plot
            '''
            if self.config.gan_loss == 'wasserstein':
                bandwidth = np.concatenate(list(scores_dict.values())).std()
            elif self.config.gan_loss == 'standard':
                bandwidth = 0.0025

            fig = go.Figure()
            for i, label in enumerate(scores_dict.keys()):
                if 'exp' in label:
                    fig.add_trace(go.Violin(x=np.array(scores_dict[label]), name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=6))
                else:
                    fig.add_trace(go.Violin(x=np.array(scores_dict[label]), name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False))
            fig.update_layout(legend_traceorder='reversed', yaxis_showgrid=True)
            wandb.log({'Discriminator Test Scores': fig})

            '''
            loss correlates - not useful right now
            '''

            loss_correlations = np.zeros(config.dataDims['num tracking features'])
            features = []
            for j in range(config.dataDims['num tracking features']):  # not that interesting
                features.append(config.dataDims['tracking features dict'][j])
                loss_correlations[j] = np.corrcoef(scores_dict['Test Real'], test_epoch_stats_dict['tracking features'][:, j], rowvar=False)[0, 1]

            loss_correlations_dict['Test Real'] = loss_correlations

            fig = go.Figure()
            color_dict = {
                'XVI': 'rgb(250,50,5)',
                'XVII': 'rgb(250,50,5)',
                'XVIII': 'rgb(250,50,5)',
                'XIX': 'rgb(250,50,5)',
                'XX': 'rgb(250,50,5)',
                'XXI': 'rgb(250,50,5)',

                'XXII': 'rgb(5,50,250)',
                'XXIII': 'rgb(5,50,250)',
                'XXIV': 'rgb(5,50,250)',
                'XXV': 'rgb(5,50,250)',
                'XXVI': 'rgb(5,50,250)',

                'Train Real': 'rgb(5,250,50)',
                'Test Real': 'rgb(5,250,50)'
            }
            for target in loss_correlations_dict.keys():
                fig.add_trace(go.Bar(
                    y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['num tracking features'])],
                    x=[loss_correlations_dict[target][i] for i in range(config.dataDims['num tracking features'])],
                    orientation='h',
                    name=target,
                    showlegend=True,
                    marker_color=color_dict[target],
                ))
            fig.update_layout(showlegend=True)
            wandb.log({'Test loss correlates': fig})

            '''
            rdf distance vs score vs energy
            '''
            '''
            for each target
            '''
            for i, label in enumerate(rdf_full_distance_dict.keys()):
                fig = make_subplots(rows=1, cols=3)
                xline = np.asarray([np.amin(normed_energy_dict[label]), np.amax(normed_energy_dict[label])])
                linreg_result = linregress((normed_energy_dict[label]), scores_dict[label])
                yline = xline * linreg_result.slope + linreg_result.intercept
                fig.add_trace(go.Scattergl(x=normed_energy_dict[label], y=scores_dict[label], showlegend=False,
                                           mode='markers', marker=dict(size=4, color=rdf_full_distance_dict[label], colorscale='Viridis', showscale=False)),
                              row=1, col=1)
                fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Energy R={linreg_result.rvalue:.3f}'), row=1, col=1)

                xline = np.asarray([np.amin(rdf_full_distance_dict[label]), np.amax(rdf_full_distance_dict[label])])
                linreg_result = linregress(rdf_full_distance_dict[label], scores_dict[label])
                yline = xline * linreg_result.slope + linreg_result.intercept
                fig.add_trace(go.Scattergl(x=rdf_full_distance_dict[label], y=scores_dict[label], showlegend=False,
                                           mode='markers', marker=dict(size=4, color=normed_energy_dict[label], colorscale='Viridis', showscale=False)),
                              row=1, col=2)
                fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Full RDF R={linreg_result.rvalue:.3f}'), row=1, col=2)

                xline = np.asarray([np.amin(rdf_inter_distance_dict[label]), np.amax(rdf_inter_distance_dict[label])])
                linreg_result = linregress(rdf_inter_distance_dict[label], scores_dict[label])
                yline = xline * linreg_result.slope + linreg_result.intercept
                fig.add_trace(go.Scattergl(x=rdf_inter_distance_dict[label], y=scores_dict[label], showlegend=False,
                                           mode='markers', marker=dict(size=4, color=normed_energy_dict[label], colorscale='Viridis', showscale=False)),
                              row=1, col=3)
                fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Intermolecular RDF R={linreg_result.rvalue:.3f}'), row=1, col=3)
                fig.update_layout(title=label)
                fig.update_xaxes(title_text='Normed LJ Energy', row=1, col=1)
                fig.update_yaxes(title_text='Model score', row=1, col=1)
                fig.update_xaxes(title_text='Full RDF Distance', row=1, col=2)
                fig.update_xaxes(title_text='Intermolecular RDF Distance', row=1, col=3)

                # fig.show()
                wandb.log({f'Target Analysis': fig})
            '''
            for all targets
            '''
            fig = make_subplots(rows=1, cols=3)
            energies = np.concatenate([val for val in normed_energy_dict.values()])
            full_rdf = np.concatenate([val for val in rdf_full_distance_dict.values()])
            inter_rdf = np.concatenate([val for val in rdf_inter_distance_dict.values()])
            normed_score = np.concatenate([normalize(scores_dict[key]) for key in scores_dict.keys() if key in normed_energy_dict.keys()])

            xline = np.asarray([np.amin(energies), np.amax(energies)])
            linreg_result = linregress((energies), normed_score)
            yline = xline * linreg_result.slope + linreg_result.intercept
            fig.add_trace(go.Scattergl(x=energies, y=normed_score, showlegend=False,
                                       mode='markers', marker=dict(size=4, color=full_rdf, colorscale='Viridis', showscale=False)),
                          row=1, col=1)
            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Energy R={linreg_result.rvalue:.3f}'), row=1, col=1)

            xline = np.asarray([np.amin(full_rdf), np.amax(full_rdf)])
            linreg_result = linregress(full_rdf, normed_score)
            yline = xline * linreg_result.slope + linreg_result.intercept
            fig.add_trace(go.Scattergl(x=full_rdf, y=normed_score, showlegend=False,
                                       mode='markers', marker=dict(size=4, color=energies, colorscale='Viridis', showscale=False)),
                          row=1, col=2)
            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Full RDF R={linreg_result.rvalue:.3f}'), row=1, col=2)

            xline = np.asarray([np.amin(inter_rdf), np.amax(inter_rdf)])
            linreg_result = linregress(inter_rdf, normed_score)
            yline = xline * linreg_result.slope + linreg_result.intercept
            fig.add_trace(go.Scattergl(x=inter_rdf, y=normed_score, showlegend=False,
                                       mode='markers', marker=dict(size=4, color=energies, colorscale='Viridis', showscale=False)),
                          row=1, col=3)
            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Intermolecular RDF R={linreg_result.rvalue:.3f}'), row=1, col=3)

            fig.update_layout(title='All BT Targets')
            fig.update_xaxes(title_text='Normed Energy', row=1, col=1)
            fig.update_yaxes(title_text='Model score', row=1, col=1)
            fig.update_xaxes(title_text='Full RDF Distance', row=1, col=2)
            fig.update_xaxes(title_text='Intermolecular RDF Distance', row=1, col=3)
            wandb.log({f'Target Analysis': fig})

            '''
            within-submission score vs rankings
            file formats are different between BT 5 and BT6
            '''
            target_identifiers = {}
            rankings = {}
            group = {}
            list_num = {}
            for label in ['XXII', 'XXIII', 'XXVI']:
                target_identifiers[label] = [extra_test_dict['identifiers'][all_identifiers[label][n]] for n in range(len(all_identifiers[label]))]
                rankings[label] = []
                group[label] = []
                list_num[label] = []
                for ident in target_identifiers[label]:
                    if 'edited' in ident:
                        ident = ident[7:]

                    long_ident = ident.split('_')
                    list_num[label].append(int(ident[len(label) + 1]))
                    rankings[label].append(int(long_ident[-1]) + 1)
                    group[label].append(long_ident[1])

            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=(['XXII', 'XXIII', 'XXVI']))

            for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
                xline = np.asarray([0, 100])
                linreg_result = linregress(rankings[label], scores_dict[label])
                yline = xline * linreg_result.slope + linreg_result.intercept

                fig.add_trace(go.Scattergl(x=rankings[label], y=scores_dict[label], showlegend=False,
                                           mode='markers', marker=dict(size=6, color=list_num[label], colorscale='portland', showscale=False)),
                              row=1, col=i + 1)

                fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=1, col=i + 1)
            fig.update_xaxes(title_text='Submission Rank', row=1, col=1)
            fig.update_yaxes(title_text='Model score', row=1, col=1)
            fig.update_xaxes(title_text='Submission Rank', row=1, col=2)
            fig.update_xaxes(title_text='Submission Rank', row=1, col=3)

            # fig.show()
            wandb.log({'Target Score Rankings': fig})

            for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
                names = np.unique(list(group[label]))
                uniques = len(names)
                rows = int(np.floor(np.sqrt(uniques)))
                cols = int(np.ceil(np.sqrt(uniques)) + 1)
                fig = make_subplots(rows=rows, cols=cols,
                                    subplot_titles=(names))

                for j, group_name in enumerate(np.unique(group[label])):
                    good_inds = np.where(np.asarray(group[label]) == group_name)
                    xline = np.asarray([0, 100])
                    linreg_result = linregress(np.asarray(rankings[label])[good_inds], np.asarray(scores_dict[label])[good_inds])
                    yline = xline * linreg_result.slope + linreg_result.intercept

                    fig.add_trace(go.Scattergl(x=np.asarray(rankings[label])[good_inds], y=np.asarray(scores_dict[label])[good_inds], showlegend=False,
                                               mode='markers', marker=dict(size=6, color=np.asarray(list_num[label])[good_inds], colorscale='portland', cmax=2, cmin=1, showscale=False)),
                                  row=j // cols + 1, col=j % cols + 1)

                    fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=j // cols + 1, col=j % cols + 1)

                fig.update_layout(title=label)
                wandb.log({f"{label} Groupwise Analysis": fig})

        return None

    def log_discriminator_analysis(self, epoch, test_loader, extra_test_loader,
                                   metrics_dict,
                                   test_epoch_stats_dict, config,
                                   extra_test_dict):
        '''
        Do analysis and upload results to w&b
        '''

        softmax_temperature = 10

        # load up precomputed lennard-jones energies for the blind test submissions
        bt_lj_energy_dict = np.load('../datasets/BT_LJ_energies.npy', allow_pickle=True).item()
        extra_test_dict['atomistic energy'] = np.asarray([bt_lj_energy_dict[ident] for ident in extra_test_dict['identifiers']])

        # need to identify targets
        # need to distinguish between experimental and proposed structures
        blind_test_targets = [  # 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
            'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
            'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', ]

        '''
        determine which samples go with which targets
        '''
        all_identifiers = {key: [] for key in blind_test_targets}
        for i in range(len(extra_test_dict['identifiers'])):
            item = extra_test_dict['identifiers'][i]
            for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
                if blind_test_targets[-1 - j] in item:
                    all_identifiers[blind_test_targets[-1 - j]].append(i)
                    break

        '''
        get the identifiers for all the BT targets
        '''
        # CSD identifiers for the blind test targets
        target_identifiers = {
            'XVI': 'OBEQUJ',
            'XVII': 'OBEQOD',
            'XVIII': 'OBEQET',
            'XIX': 'XATJOT',
            'XX': 'OBEQIX',
            'XXI': 'KONTIQ',
            'XXII': 'NACJAF',
            'XXIII': 'XAFPAY',
            'XXIV': 'XAFQON',
            'XXVI': 'XAFQIH'
        }

        target_identifiers_inds = {key: [] for key in blind_test_targets}
        for i in range(len(extra_test_dict['identifiers'])):
            item = extra_test_dict['identifiers'][i]
            for key in target_identifiers.keys():
                if item == target_identifiers[key]:
                    target_identifiers_inds[key] = i

        '''
        record all the stats for the CSD data
        '''
        scores_dict = {}

        if self.config.gan_loss == 'wasserstein':
            scores_dict['Test Real'] = test_epoch_stats_dict['scores']

        elif self.config.gan_loss == 'standard':
            scores_dict['Test Real'] = np_softmax(test_epoch_stats_dict['scores'], temperature=softmax_temperature)[:, 1]

        wandb.log({'Average Test score': np.average(scores_dict['Test Real'])})

        loss_correlations_dict = {}
        rdf_full_distance_dict = {}
        rdf_inter_distance_dict = {}
        energy_dict = {}

        '''
        build property dicts for the submissions and BT targets
        '''
        for target in all_identifiers.keys():  # run the analysis for each target
            if target_identifiers_inds[target] != []:

                target_index = target_identifiers_inds[target]
                raw_scores = extra_test_dict['scores'][target_index]
                if self.config.gan_loss == 'wasserstein':
                    scores = raw_scores
                elif self.config.gan_loss == 'standard':
                    scores = np_softmax(raw_scores, temperature=softmax_temperature)[:, 1]
                scores_dict[target + ' exp'] = scores
                energy_dict[target + ' exp'] = extra_test_dict['atomistic energy'][target_index]

                wandb.log({f'Average {target} exp score': np.average(scores)})

                target_full_rdf = extra_test_dict['full rdf'][target_index]
                target_inter_rdf = extra_test_dict['intermolecular rdf'][target_index]

            if all_identifiers[target] != []:

                target_indices = all_identifiers[target]
                raw_scores = extra_test_dict['scores'][target_indices]
                if self.config.gan_loss == 'wasserstein':
                    scores = raw_scores
                elif self.config.gan_loss == 'standard':
                    scores = np_softmax(raw_scores, temperature=softmax_temperature)[:, 1]
                scores_dict[target] = scores
                energy_dict[target] = extra_test_dict['atomistic energy'][target_indices]

                wandb.log({f'Average {target} score': np.average(scores)})

                submission_full_rdf = extra_test_dict['full rdf'][target_indices]
                submission_inter_rdf = extra_test_dict['intermolecular rdf'][target_indices]

                # compute distance between target & submission RDFs
                rr = np.linspace(0, 10, 100)
                sigma = 1
                smoothed_target_full_rdf = gaussian_filter1d(target_full_rdf, sigma=sigma)
                smoothed_target_inter_rdf = gaussian_filter1d(target_inter_rdf, sigma=sigma)
                smoothed_submission_full_rdf = gaussian_filter1d(np.clip(submission_full_rdf, a_min=0, a_max=10), sigma=sigma)
                smoothed_submission_inter_rdf = gaussian_filter1d(np.clip(submission_inter_rdf, a_min=0, a_max=10), sigma=sigma)

                rdf_full_distance_dict[target] = 1 - np.sum(np.minimum(smoothed_target_full_rdf, smoothed_submission_full_rdf), axis=(1, 2)) / np.sum(
                    smoothed_target_full_rdf)  # get histogram overlap over all atom pairs & scale by total target density
                rdf_inter_distance_dict[target] = 1 - np.sum(np.minimum(smoothed_target_inter_rdf, smoothed_submission_inter_rdf), axis=(1, 2)) / np.sum(smoothed_target_inter_rdf)

                # correlate losses with molecular features
                tracking_features = np.asarray(extra_test_dict['tracking features'])
                loss_correlations = np.zeros(config.dataDims['num tracking features'])
                features = []
                for j in range(config.dataDims['num tracking features']):  # not that interesting
                    features.append(config.dataDims['tracking features dict'][j])
                    loss_correlations[j] = np.corrcoef(scores, tracking_features[target_indices, j], rowvar=False)[0, 1]

                loss_correlations_dict[target] = loss_correlations

        # normed_energy_dict = {key: normalize(energy_dict[key]) for key in energy_dict.keys() if 'exp' not in key}
        normed_energy_dict = {key: standardize(energy_dict[key]) for key in energy_dict.keys() if 'exp' not in key}

        '''
        prep violin figure colors
        '''
        lens = [len(val) for val in all_identifiers.values()]
        colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', max(np.count_nonzero(lens), np.count_nonzero(list(target_identifiers_inds.values()))), colortype='rgb')

        plot_color_dict = {}
        plot_color_dict['Test Real'] = ('rgb(250,150,50)')  # test

        ind = 0
        for target in all_identifiers.keys():
            if all_identifiers[target] != []:
                plot_color_dict[target] = colors[ind]
                plot_color_dict[target + ' exp'] = colors[ind]

                ind += 1

        '''
        violin scores plot
        '''
        if self.config.gan_loss == 'wasserstein':
            bandwidth = np.concatenate(list(scores_dict.values())).std()
        elif self.config.gan_loss == 'standard':
            bandwidth = 0.0025

        fig = go.Figure()
        for i, label in enumerate(scores_dict.keys()):
            if 'exp' in label:
                fig.add_trace(go.Violin(x=np.array(scores_dict[label]), name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=6))
            else:
                fig.add_trace(go.Violin(x=np.array(scores_dict[label]), name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False))
        fig.update_layout(legend_traceorder='reversed', yaxis_showgrid=True)
        wandb.log({'Discriminator Test Scores': fig})

        '''
        rdf distance vs score vs energy
        for each target
        '''

        energies = np.concatenate([val for val in normed_energy_dict.values()])
        full_rdf = np.concatenate([val for val in rdf_full_distance_dict.values()])
        inter_rdf = np.concatenate([val for val in rdf_inter_distance_dict.values()])
        normed_score = np.concatenate([normalize(scores_dict[key]) for key in scores_dict.keys() if key in normed_energy_dict.keys()])

        fig = make_subplots(rows=2, cols=4,
                            subplot_titles=(list(rdf_full_distance_dict.keys())) + ['All'])

        for i, label in enumerate(rdf_full_distance_dict.keys()):
            row = i // 4 + 1
            col = i % 4 + 1
            xline = np.asarray([np.amin(rdf_full_distance_dict[label]), np.amax(rdf_full_distance_dict[label])])
            linreg_result = linregress(rdf_full_distance_dict[label], scores_dict[label])
            yline = xline * linreg_result.slope + linreg_result.intercept

            fig.add_trace(go.Scattergl(x=rdf_full_distance_dict[label], y=scores_dict[label], showlegend=False,
                                       mode='markers', marker=dict(size=4, color=normed_energy_dict[label], colorscale='Viridis', showscale=False)),
                          row=row, col=col)

            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=row, col=col)

        xline = np.asarray([np.amin(full_rdf), np.amax(full_rdf)])
        linreg_result = linregress(full_rdf, normed_score)
        yline = xline * linreg_result.slope + linreg_result.intercept
        fig.add_trace(go.Scattergl(x=full_rdf, y=normed_score, showlegend=False,
                                   mode='markers', marker=dict(size=4, color=energies, colorscale='Viridis', showscale=False)),
                      row=2, col=4)
        fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'All Targets R={linreg_result.rvalue:.3f}'), row=2, col=4)

        # fig.show()
        wandb.log({f'Target Analysis': fig})

        '''
        within-submission score vs rankings
        file formats are different between BT 5 and BT6
        '''
        target_identifiers = {}
        rankings = {}
        group = {}
        list_num = {}
        for label in ['XXII', 'XXIII', 'XXVI']:
            target_identifiers[label] = [extra_test_dict['identifiers'][all_identifiers[label][n]] for n in range(len(all_identifiers[label]))]
            rankings[label] = []
            group[label] = []
            list_num[label] = []
            for ident in target_identifiers[label]:
                if 'edited' in ident:
                    ident = ident[7:]

                long_ident = ident.split('_')
                list_num[label].append(int(ident[len(label) + 1]))
                rankings[label].append(int(long_ident[-1]) + 1)
                group[label].append(long_ident[1])

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=(['XXII', 'XXIII', 'XXVI']))

        for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
            xline = np.asarray([0, 100])
            linreg_result = linregress(rankings[label], scores_dict[label])
            yline = xline * linreg_result.slope + linreg_result.intercept

            fig.add_trace(go.Scattergl(x=rankings[label], y=scores_dict[label], showlegend=False,
                                       mode='markers', marker=dict(size=6, color=list_num[label], colorscale='portland', showscale=False)),
                          row=1, col=i + 1)

            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=1, col=i + 1)
        fig.update_xaxes(title_text='Submission Rank', row=1, col=1)
        fig.update_yaxes(title_text='Model score', row=1, col=1)
        fig.update_xaxes(title_text='Submission Rank', row=1, col=2)
        fig.update_xaxes(title_text='Submission Rank', row=1, col=3)

        # fig.show()
        wandb.log({'Target Score Rankings': fig})

        for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
            names = np.unique(list(group[label]))
            uniques = len(names)
            rows = int(np.floor(np.sqrt(uniques)))
            cols = int(np.ceil(np.sqrt(uniques)) + 1)
            fig = make_subplots(rows=rows, cols=cols,
                                subplot_titles=(names))
            colormap = {1: 'rgb(0,0,100)',
                        2: 'rgb(100,0,0)'}

            for j, group_name in enumerate(np.unique(group[label])):
                good_inds = np.where(np.asarray(group[label]) == group_name)
                xline = np.asarray([0, 100])
                linreg_result = linregress(np.asarray(rankings[label])[good_inds], np.asarray(scores_dict[label])[good_inds])
                yline = xline * linreg_result.slope + linreg_result.intercept

                fig.add_trace(go.Scattergl(x=np.asarray(rankings[label])[good_inds], y=np.asarray(scores_dict[label])[good_inds], showlegend=False,
                                           mode='markers', marker=dict(size=6, color=np.asarray(list_num[label])[good_inds], colorscale='portland', cmax=2, cmin=1, showscale=False)),
                              row=j // cols + 1, col=j % cols + 1)

                fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=j // cols + 1, col=j % cols + 1)

            fig.update_layout(title=label)
            wandb.log({f"{label} Groupwise Analysis": fig})

        finished = 1
        return None

    def train_discriminator(self, generated_samples, discriminator, config, data, i, target_handedness=None):
        # generate fakes & create supercell data
        real_supercell_data = self.supercell_builder.build_supercells_from_dataset(data.clone(), config)
        fake_supercell_data, generated_cell_volumes, overlaps_list = \
            self.supercell_builder.build_supercells(data.clone().to(generated_samples.device), generated_samples,
                                                    config.supercell_size, config.discriminator.graph_convolution_cutoff, override_sg=config.generate_sgs, target_handedness=target_handedness)

        if config.device.lower() == 'cuda':  # redundant
            real_supercell_data = real_supercell_data.cuda()
            fake_supercell_data = fake_supercell_data.cuda()

        if config.test_mode or config.anomaly_detection:
            assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"
            assert torch.sum(torch.isnan(fake_supercell_data.x)) == 0, "NaN in training input"

        if config.discriminator.positional_noise > 0:
            real_supercell_data.pos += torch.randn_like(real_supercell_data.pos) * config.discriminator.positional_noise
            fake_supercell_data.pos += torch.randn_like(fake_supercell_data.pos) * config.discriminator.positional_noise

        score_on_real, real_distances_dict = self.adversarial_loss(discriminator, real_supercell_data, config)
        score_on_fake, fake_pairwise_distances_dict = self.adversarial_loss(discriminator, fake_supercell_data, config)

        return score_on_real, score_on_fake, fake_supercell_data.cell_params.cpu().detach().numpy(), real_distances_dict, fake_pairwise_distances_dict

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
            supercell_data, generated_cell_volumes, overlaps_list = self.supercell_builder.build_supercells(data.clone(), generated_samples, config.supercell_size, config.discriminator.graph_convolution_cutoff,
                                                                                                            override_sg=config.generate_sgs)
            data.cell_params = supercell_data.cell_params
        else:
            supercell_data = None

        if config.train_generator_adversarially or config.train_generator_g2:
            if config.device.lower() == 'cuda':
                supercell_data = supercell_data.cuda()

            if config.test_mode or config.anomaly_detection:
                assert torch.sum(torch.isnan(data.x)) == 0, "NaN in training input"

            discriminator_score, dist_dict = self.adversarial_loss(discriminator, supercell_data, config)

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
        predictions = generator(data.to(generator.model.device))[:, 0]
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
        # cutoff = (0.6 - self.dataDims['target mean']) / self.dataDims['target std']  # cutoff (0.55) in standardized basis
        # packing_loss = F.mse_loss(-(generated_packing_coefficients - cutoff))  # linear loss below a cutoff
        packing_loss = F.mse_loss(generated_packing_coefficients, torch.zeros_like(generated_packing_coefficients), reduction='none')  # since the data is standardized, we want it to regress towards 0 (the mean)

        if self.config.test_mode:
            assert torch.sum(torch.isnan(packing_loss)) == 0
            assert torch.sum(torch.isnan(den_loss)) == 0

        return den_loss, generated_packing_coefficients.cpu().detach().numpy(), csd_packing_coefficients.cpu().detach().numpy(), packing_loss

    def flow_iter(self, generator, data):  # todo update cell_params norming
        '''
        train the generator via standard normalizing flow loss
        # todo note that this approach will push to model toward good **raw** samples, not accounting for cell 'cleaning'
        '''
        normed_lengths = data.cell_params[:, 0:3] / (data.Z[:, None] ** (1 / 3)) / (data.mol_volume[:, None] ** (1 / 3))
        normed_samples = data.cell_params.clone()
        normed_samples[:, :3] = normed_lengths

        means = torch.Tensor(self.dataDims['lattice means'])
        stds = torch.Tensor(self.dataDims['lattice stds'])
        means[:3] = torch.Tensor(self.dataDims['lattice normed length means'])
        stds[:3] = torch.Tensor(self.dataDims['lattice normed length stds'])
        std_samples = (normed_samples - means.to(normed_samples.device)) / stds.to(normed_samples.device)

        zs, prior_logprob, log_det = generator.nf_forward(std_samples, conditions=data)
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

        '''
        init supercell builder class
        '''
        self.supercell_builder = SupercellBuilder(self.sym_ops, self.sym_info, self.normed_lattice_vectors, self.atom_weights, self.dataDims)

        self.mol_volume_ind = self.dataDims['tracking features dict'].index('molecule volume')
        self.crystal_packing_ind = self.dataDims['tracking features dict'].index('crystal packing coefficient')
        self.crystal_density_ind = self.dataDims['tracking features dict'].index('crystal calculated density')
        self.mol_size_ind = self.dataDims['tracking features dict'].index('molecule num atoms')
        self.pg_ind_dict = {thing[14:]: ind + self.dataDims['num atomwise features'] for ind, thing in enumerate(self.dataDims['molecule features']) if 'pg is' in thing}
        self.sg_ind_dict = {thing[14:]: ind + self.dataDims['num atomwise features'] for ind, thing in enumerate(self.dataDims['molecule features']) if 'sg is' in thing}  # todo simplify - allow all possibilities
        self.crysys_ind_dict = {thing[18:]: ind + self.dataDims['num atomwise features'] for ind, thing in enumerate(self.dataDims['molecule features']) if 'crystal system is' in thing}

        self.sym_info['pg_ind_dict'] = self.pg_ind_dict
        self.sym_info['sg_ind_dict'] = self.sg_ind_dict
        self.sym_info['crysys_ind_dict'] = self.crysys_ind_dict

        ''' independent gaussian model to approximate the problem
        '''
        self.randn_generator = independent_gaussian_model(input_dim=self.dataDims['num lattice features'],
                                                          means=self.dataDims['lattice means'],
                                                          stds=self.dataDims['lattice stds'],
                                                          normed_length_means=self.dataDims['lattice normed length means'],
                                                          normed_length_stds=self.dataDims['lattice normed length stds'],
                                                          cov_mat=self.dataDims['lattice cov mat'])

        return config, dataset_builder

    def MCMC_sampling(self, generator, discriminator, test_loader):
        '''
        Stun MC annealing on a pretrained discriminator
        '''
        from torch_geometric.loader.dataloader import Collater

        n_samples = self.config.final_batch_size
        single_mol_data = test_loader.dataset[0]
        collater = Collater(None, None)
        single_mol_data = collater([single_mol_data for n in range(n_samples)])
        self.randn_generator.cpu()

        '''
        initialize sampler
        '''
        smc_sampler = Sampler(
            gammas=np.logspace(-7, 2, n_samples),
            seedInd=0,
            acceptance_mode='stun',
            debug=True,
            init_temp=1,
            random_generator=self.randn_generator,
            move_size=.5,
            supercell_size=self.config.supercell_size,
            graph_convolution_cutoff=self.config.discriminator.graph_convolution_cutoff
        )

        '''
        random batch of initial conditions
        '''
        init_samples = self.randn_generator.forward(single_mol_data, n_samples).cpu().detach().numpy()

        # debug test
        supercells1, _, _ = self.supercell_builder.build_supercells(
            supercell_data=single_mol_data.clone(), cell_sample=torch.Tensor(init_samples), supercell_size=3, graph_convolution_cutoff=7, target_handedness=torch.ones(10))

        supercells2, _, _ = self.supercell_builder.build_supercells(
            supercell_data=single_mol_data.clone(), cell_sample=torch.Tensor(init_samples), supercell_size=3, graph_convolution_cutoff=7, target_handedness=-torch.ones(10))

        '''
        run sampling
        '''
        num_iters = 1000
        sampling_dict = smc_sampler(discriminator, self.supercell_builder,
                                    single_mol_data, init_samples, num_iters)

        flag = 1
        '''
        analyze outputs
        '''

        plt.figure(5)
        plt.clf()
        plt.subplot(2, 3, 1)
        plt.plot(sampling_dict['scores'].T)
        plt.ylabel('scores')
        plt.subplot(2, 3, 2)
        plt.plot(sampling_dict['stun score'].T)
        plt.ylabel('stun score')
        plt.ylim([-5, 5])
        plt.subplot(2, 3, 3)
        plt.plot(np.exp(-sampling_dict['scores'].T))
        plt.ylabel('probabilities')
        plt.subplot(2, 2, 3)
        plt.plot(sampling_dict['acceptance ratio'].T)
        plt.ylabel('acc rat')
        plt.subplot(2, 2, 4)
        plt.semilogy(sampling_dict['temperature'].T)
        plt.ylabel('temps')
        plt.tight_layout()

        best_inds = np.argsort(sampling_dict['scores'].flatten())
        best_samples = sampling_dict['samples'].reshape(12, n_samples * num_iters)[:, best_inds[:n_samples]].T

        best_supercells, _, _ = self.supercell_builder.build_supercells(single_mol_data.clone(), torch.Tensor(best_samples).cuda(),
                                                                        supercell_size=1, graph_convolution_cutoff=7)

        best_rdfs, rr = crystal_rdf(best_supercells, rrange=[0, 10], bins=100, intermolecular=True)

        return sampling_dict

    def generate_discriminator_negatives(self, epoch_stats_dict, config, data, generator, i):
        n_generators = sum([config.train_discriminator_adversarially, config.train_discriminator_on_noise, config.train_discriminator_on_randn])
        gen_random_number = np.random.uniform(0, 1, 1)
        gen_randn_range = np.linspace(0, 1, n_generators + 1)

        if config.train_discriminator_adversarially:
            ii = i % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:  # randomly sample which generator to use at each iteration
                generated_samples_i = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device))
                handedness = None
                epoch_stats_dict['generator sample source'].extend(np.zeros(len(generated_samples_i)))

        if config.train_discriminator_on_randn:
            ii = (i + 1) % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:
                generated_samples_i = self.randn_generator.forward(data, data.num_graphs).to(generator.device)
                handedness = None
                epoch_stats_dict['generator sample source'].extend(np.ones(len(generated_samples_i)))

        if config.train_discriminator_on_noise:
            ii = (i + 2) % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:
                generated_samples_ii = (data.cell_params - torch.Tensor(self.dataDims['lattice means'])) / torch.Tensor(self.dataDims['lattice stds'])  # standardize
                generated_samples_i = ((generated_samples_ii + torch.randn_like(generated_samples_ii) * config.generator_noise_level)).to(generator.device)  # add jitter and return in standardized basis
                handedness = data.asym_unit_handedness
                epoch_stats_dict['generator sample source'].extend(np.ones(len(generated_samples_i)) * 2)

        return generated_samples_i, handedness, epoch_stats_dict

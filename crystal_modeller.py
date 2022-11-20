import torch
import wandb
from utils import *
import glob
from model_utils import *
from dataset_management.dataset_manager import Miner
from torch import backends, optim
from dataset_utils import BuildDataset, get_dataloaders, update_batch_size, get_extra_test_loader
import matplotlib.pyplot as plt
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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering


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
            # save learning rates so we can un-overwrite them
            max_lr = config.generator.max_lr * 1
            lr = config.generator.learning_rate * 1
            config.generator = Namespace(**g_checkpoint['config'])  # overwrite the settings for the model
            config.generator.learning_rate = lr
            config.generator.max_lr = max_lr
        if config.d_model_path is not None:
            d_checkpoint = torch.load(config.d_model_path)
            max_lr = config.discriminator.max_lr * 1
            lr = config.discriminator.learning_rate * 1
            config.discriminator = Namespace(**d_checkpoint['config'])
            config.discriminator.learning_rate = lr
            config.discriminator.max_lr = max_lr
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

        if config.device.lower() == 'cuda':
            print('Putting models on CUDA')
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
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

        # cuda
        if config.device.lower() == 'cuda':
            pass
            # generator = gnn.DataParallel(generator)
            # discriminator = gnn.DataParallel(discriminator)

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
            wandb.run.name = wandb.config.machine + '_' + str(wandb.config.run_num)  # overwrite procedurally generated run name with our run name
            wandb.run.save()
            # config = wandb.config # todo: wandb configs don't support nested namespaces. Sweeps are officially broken - look at the github thread

            config, dataset_builder = self.train_boilerplate()
            generator, discriminator, g_optimizer, g_schedulers, d_optimizer, d_schedulers, params1, params2 \
                = self.init_gan(config, self.dataDims)  # todo change this to just resetting all the parameters of existing models

            # get batch size
            if config.auto_batch_sizing:
                print('Finding optimal batch size')
                train_loader, test_loader, config.final_batch_size = \
                    self.get_batch_size(generator, discriminator, g_optimizer, d_optimizer,
                                        dataset_builder, config)
                # reload original models
                if config.g_model_path is not None:
                    g_checkpoint = torch.load(config.g_model_path)
                    if list(g_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                        for i in list(g_checkpoint['model_state_dict']):
                            g_checkpoint['model_state_dict'][i[7:]] = g_checkpoint['model_state_dict'].pop(i)

                    generator.load_state_dict(g_checkpoint['model_state_dict'])
                    g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])

                if config.d_model_path is not None:
                    d_checkpoint = torch.load(config.d_model_path)
                    if list(d_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                        for i in list(d_checkpoint['model_state_dict']):
                            d_checkpoint['model_state_dict'][i[7:]] = d_checkpoint['model_state_dict'].pop(i)
                    discriminator.load_state_dict(d_checkpoint['model_state_dict'])
                    d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])

            else:
                print('Getting dataloaders for pre-determined batch size')
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.max_batch_size
            del dataset_builder

            if config.extra_test_set_paths is not None:
                extra_test_loader = get_extra_test_loader(config, config.extra_test_set_paths, dataDims=self.dataDims,
                                                          pg_dict=self.point_groups, sg_dict=self.space_groups, lattice_dict=self.lattice_type)
            else:
                extra_test_loader = None

            print("Training batch size set to {}".format(config.final_batch_size))

            # model, optimizer, schedulers
            print('Reinitializing model and optimizer')
            if config.g_model_path is None:
                generator.apply(weight_reset)
            if config.d_model_path is None:
                discriminator.apply(weight_reset)
            n_params = params1 + params2

            wandb.watch((generator, discriminator), log_graph=True, log_freq=100)
            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_metrics(config=config)

            # training loop
            d_hit_max_lr, g_hit_max_lr, converged, epoch = False, False, config.max_epochs == 0, 0  # for evaluation mode
            if config.max_epochs == -1:  # skip evaluation altogether
                self.make_nice_figures(config)
            else:
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

                                        np.save(f'../{config.run_num}_extra_test_dict', extra_test_epoch_stats_dict)  # save
                                        np.save(f'../{config.run_num}_test_epoch_stats_dict', test_epoch_stats_dict)

                                else:
                                    extra_test_epoch_stats_dict = None

                            print('epoch={}; d_nll_tr={:.5f}; d_nll_te={:.5f}; g_nll_tr={:.5f}; g_nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
                                epoch, np.mean(np.asarray(d_err_tr)), np.mean(np.asarray(d_err_te)),
                                np.mean(np.asarray(g_err_tr)), np.mean(np.asarray(g_err_te)),
                                time_train, time_test))

                            d_optimizer, d_learning_rate, d_hit_max_lr, g_optimizer, g_learning_rate, g_hit_max_lr = \
                                self.update_lr(config, d_schedulers, d_optimizer, d_err_tr, d_hit_max_lr,
                                               g_schedulers, g_optimizer, g_err_tr, g_hit_max_lr)

                            metrics_dict = \
                                self.update_gan_metrics(epoch, metrics_dict, d_err_tr, d_err_te,
                                                        g_err_tr, g_err_te, d_learning_rate, g_learning_rate)

                            self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                                              d_tr_record, d_te_record, g_tr_record, g_te_record)

                            if epoch % config.wandb.sample_reporting_frequency == 0:
                                self.log_gan_accuracy(epoch, train_loader,
                                                      train_epoch_stats_dict, test_epoch_stats_dict, config,
                                                      extra_test_dict=extra_test_epoch_stats_dict)

                            self.model_checkpointing(epoch, config, discriminator, generator, d_optimizer, g_optimizer, g_err_te, d_err_te, metrics_dict)

                            generator_convergence, discriminator_convergence = \
                                self.check_model_convergence(metrics_dict, config, epoch)
                            if (generator_convergence and discriminator_convergence) and (epoch > config.history + 2):
                                print('Training has converged!')
                                config.finished = True
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

                        if config.device.lower() == 'cuda':
                            torch.cuda.empty_cache()  # clear GPU

                '''
                run post-training evaluation
                '''
                # reload best test
                g_path = f'../models/generator_{config.run_num}'
                d_path = f'../models/discriminator_{config.run_num}'
                if os.path.exists(g_path):
                    g_checkpoint = torch.load(g_path)
                    if list(g_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                        for i in list(g_checkpoint['model_state_dict']):
                            g_checkpoint['model_state_dict'][i[7:]] = g_checkpoint['model_state_dict'].pop(i)
                    generator.load_state_dict(g_checkpoint['model_state_dict'])

                if os.path.exists(d_path):
                    d_checkpoint = torch.load(d_path)
                    if list(d_checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                        for i in list(d_checkpoint['model_state_dict']):
                            d_checkpoint['model_state_dict'][i[7:]] = d_checkpoint['model_state_dict'].pop(i)
                    discriminator.load_state_dict(d_checkpoint['model_state_dict'])

                with torch.no_grad():
                    d_err_te, d_te_record, g_err_te, g_te_record, test_epoch_stats_dict, time_test = \
                        self.epoch(config, dataLoader=test_loader, generator=generator, discriminator=discriminator,
                                   update_gradients=False, record_stats=True, epoch=epoch)  # compute loss on test set
                    np.save(f'../{config.run_num}_test_epoch_stats_dict', test_epoch_stats_dict)

                    if config.extra_test_set_paths is not None:
                        extra_test_epoch_stats_dict, time_test_ex = \
                            self.discriminator_evaluation(config, dataLoader=extra_test_loader, discriminator=discriminator)  # compute loss on test set

                        np.save(f'../{config.run_num}_extra_test_dict', extra_test_epoch_stats_dict)
                    else:
                        extra_test_epoch_stats_dict = None

                metrics_dict = self.update_gan_metrics(
                    epoch, metrics_dict,
                    np.zeros(10), d_err_te,
                    np.zeros(10), g_err_te,
                    d_optimizer.defaults['lr'], g_optimizer.defaults['lr'])

                self.log_gan_loss(metrics_dict, None, test_epoch_stats_dict,
                                  None, d_te_record, None, g_te_record)

                self.log_gan_accuracy(epoch, train_loader,
                                      None, test_epoch_stats_dict, config,
                                      extra_test_dict=extra_test_epoch_stats_dict)

                # # for working with a trained model # deprecated
                if config.sample_after_training:
                    sampling_dict = self.MCMC_sampling(discriminator, test_loader, config.sample_ind, config.sample_steps, config.sample_move_size, test_epoch_stats_dict, config)

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
        epoch_stats_dict = {
            'tracking features': [],
            'generator density target': [],
            'generator density prediction': [],

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
            'generated sample distances': [],
            'distortion level': [],
            'real vdW penalty': [],
            'fake vdW penalty': [],
        }

        generated_supercell_examples_dict = {}

        rand_batch_ind = np.random.randint(0, len(dataLoader))

        for i, data in enumerate(tqdm.tqdm(dataLoader)):

            '''
            train discriminator
            '''
            if epoch % config.discriminator.training_period == 0:  # only train the discriminator every XX epochs
                if config.train_discriminator_adversarially or config.train_discriminator_on_noise or config.train_discriminator_on_randn:
                    generated_samples_i, handedness, epoch_stats_dict = self.generate_discriminator_negatives(epoch_stats_dict, config, data, generator, i)

                    score_on_real, score_on_fake, generated_samples, real_dist_dict, fake_dist_dict, real_vdW_score, fake_vdW_score \
                        = self.train_discriminator(generated_samples_i, discriminator, config, data, i, handedness, return_rdf=config.gan_loss == 'distance')  # alternately trains on real and fake samples

                    epoch_stats_dict['discriminator real score'].extend(score_on_real.cpu().detach().numpy())
                    epoch_stats_dict['discriminator fake score'].extend(score_on_fake.cpu().detach().numpy())
                    epoch_stats_dict['real vdW penalty'].extend(real_vdW_score.cpu().detach().numpy())
                    epoch_stats_dict['fake vdW penalty'].extend(fake_vdW_score.cpu().detach().numpy())

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

                    epoch_stats_dict['generated cell parameters'].extend(generated_samples_i.cpu().detach().numpy())
                    epoch_stats_dict['final generated cell parameters'].extend(generated_samples)

                else:
                    d_err.append(np.zeros(1))
                    d_loss_record.extend(np.zeros(data.num_graphs))

            '''
            train flowmodel first just in case it's initialization is too wild for the discriminator # DEPRECATED
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
            train_generator # todo update # DEPRECATED
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
                    elif config.gan_loss == 'distance':
                        assert False  # implement something here
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
            'vdW penalty': [],
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

            full_rdfs, rr, self.elementwise_correlations_labels = crystal_rdf(real_supercell_data, elementwise=True, raw_density=True, rrange=[0, 10], bins=500)
            intermolecular_rdfs, rr, _ = crystal_rdf(real_supercell_data, intermolecular=True, elementwise=True, raw_density=True, rrange=[0, 10], bins=500)

            epoch_stats_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())
            epoch_stats_dict['identifiers'].extend(data.csd_identifier)  #
            epoch_stats_dict['scores'].extend(score_on_real.cpu().detach().numpy())
            epoch_stats_dict['intermolecular rdf'].extend(intermolecular_rdfs.cpu().detach().numpy())
            epoch_stats_dict['full rdf'].extend(full_rdfs.cpu().detach().numpy())
            epoch_stats_dict['vdW penalty'].extend(vdW_penalty(real_supercell_data, self.vdw_radii).cpu().detach().numpy())

            if compute_LJ_energy:
                epoch_stats_dict['atomistic energy'].extend(atomwise_energy)

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        epoch_stats_dict['scores'] = np.stack(epoch_stats_dict['scores'])
        epoch_stats_dict['tracking features'] = np.stack(epoch_stats_dict['tracking features'])
        epoch_stats_dict['full rdf'] = np.stack(epoch_stats_dict['full rdf'])
        epoch_stats_dict['intermolecular rdf'] = np.stack(epoch_stats_dict['intermolecular rdf'])
        epoch_stats_dict['vdW penalty'] = np.asarray(epoch_stats_dict['vdW penalty'])
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
        elif config.gan_loss == 'distance':
            scores = output[:, 0]  # the raw distance output
        else:
            print(config.gan_loss + ' is not a valid GAN loss function!')
            sys.exit()

        assert torch.sum(torch.isnan(scores)) == 0

        return scores, extra_outputs['dists dict']

    #
    # def compute_dataset_rmsd(self, target_identifiers, target_identifiers_list, extra_test_loader):
    #
    #     if os.path.exists('../../test_data_rmsds.npy'):
    #         rmsd_dict = np.load('../../test_data_rmsds.npy', allow_pickle=True).item()
    #     else:
    #         rmsd_dict = {}
    #
    #     '''
    #     initialize analyzer
    #     '''
    #
    #     sim_engine = PackingSimilarity()
    #     sim_engine.settings.packing_shell_size = 20
    #     sim_engine.settings.allow_molecular_differences = True
    #
    #     target_crystals = {}
    #     target_inds = {}
    #
    #     '''
    #     pre generate all the reference structures & print CIFs & collect crystal objects
    #     '''
    #
    #     for i, data in enumerate(tqdm.tqdm(extra_test_loader.dataset)):
    #         item = data.csd_identifier
    #         for key in target_identifiers.keys():
    #             if item == target_identifiers[key]:
    #                 target_inds[key] = i
    #
    #     for key in target_inds.keys():
    #         data = extra_test_loader.dataset[target_inds[key]]
    #         pymat_struct = structure.IStructure(species=data.x[:, 0].repeat(data.Z),
    #                                             coords=data.ref_cell_pos.reshape(int(data.Z * len(data.pos)), 3),
    #                                             lattice=lattice.Lattice(data.T_fc.T.type(dtype=torch.float16)),
    #                                             coords_are_cartesian=True)
    #         writer1 = cif.CifWriter(pymat_struct, symprec=0.1)
    #         writer1.write_file(key + '.cif')
    #         target_crystals[key] = CrystalReader(key + '.cif', format='cif')[0]
    #
    #     '''
    #     do the rmsd calculation for each sample in the dataset
    #     '''
    #
    #     identifiers_inds = {key: [] for key in target_identifiers_list}
    #
    #     for i, data in enumerate(tqdm.tqdm(extra_test_loader.dataset)):
    #         '''
    #         identify the reference
    #         -> need also to account for the 3 structures of target 31
    #         '''
    #         crystal_target = None
    #         item = data.csd_identifier
    #         if item not in list(rmsd_dict.keys()):
    #             for j in range(len(target_identifiers_list)):  # go in reverse to account for roman numerals system of duplication
    #                 if target_identifiers_list[-1 - j] in item:
    #                     crystal_target = target_identifiers_list[-1 - j]
    #                     identifiers_inds[crystal_target].append(i)
    #                     break
    #
    #             try:
    #                 '''
    #                 generate cif & compare
    #                 '''
    #                 pymat_struct = structure.IStructure(species=data.x[:, 0].repeat(data.Z),
    #                                                     coords=data.ref_cell_pos.reshape(int(data.Z * len(data.pos)), 3),
    #                                                     lattice=lattice.Lattice(data.T_fc.T.type(dtype=torch.float16)),
    #                                                     coords_are_cartesian=True)
    #                 writer1 = cif.CifWriter(pymat_struct, symprec=0.1)
    #                 writer1.write_file('crystal.cif')
    #
    #                 test_crystal = CrystalReader('crystal.cif', format='cif')[0]
    #
    #                 if crystal_target == 't31':
    #                     comparison_output1 = sim_engine.compare(target_crystals['XXXI_1'], test_crystal)
    #                     comparison_output2 = sim_engine.compare(target_crystals['XXXI_2'], test_crystal)
    #                     # comparison_output3 = sim_engine.compare(target_crystals['XXXI_3'], test_crystal)
    #                     rmsd_dict[data.csd_identifier] = [comparison_output1.rmsd,
    #                                                       comparison_output2.rmsd, ]
    #                     # comparison_output3.rmsd]
    #
    #                 elif crystal_target is not None:
    #                     comparison_output = sim_engine.compare(target_crystals[crystal_target], test_crystal)
    #                     rmsd_dict[data.csd_identifier] = comparison_output.rmsd
    #             except:
    #                 rmsd_dict[data.csd_identifier] = 'error'  # todo pymatgen and ccdc disagree on some space group names
    #
    #         if i % 1000 == 0:
    #             np.save('../../test_data_rmsds', rmsd_dict)
    #
    #     np.save('../../test_data_rmsds', rmsd_dict)
    #     return rmsd_dict

    def get_dists(self, config, extra_test_loader):
        '''
        compute distance metrics for extra test data
        run from the end of dataset_utils
        '''

        from pymatgen.core import (structure, lattice)
        # from ccdc.crystal import PackingSimilarity
        # from ccdc.io import CrystalReader
        from pymatgen.io import cif
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)  # annoying numpy error
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # annoying numpy error

        blind_test_targets = [  # 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
            'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
            'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', ]

        target_identifiers = {
            'XVI': 'OBEQUJ',
            'XVII': 'OBEQOD',
            'XVIII': 'OBEQET',
            'XIX': 'XATJOT',
            'XX': 'OBEQIX',
            'XXI': 'KONTIQ',
            'XXII': 'NACJAF',
            'XXIII': 'XAFPAY',
            'XXIII_1': 'XAFPAY01',
            'XXIII_2': 'XAFPAY02',
            'XXXIII_3': 'XAFPAY03',
            'XXXIII_4': 'XAFPAY04',
            'XXIV': 'XAFQON',
            'XXVI': 'XAFQIH',
            'XXXI_1': '2199671_p10167_1_0',
            'XXXI_2': '2199673_1_0',
            # 'XXXI_3': '2199672_1_0',
        }

        target_identifiers_list = [
            'XVI',
            'XVII',
            'XVIII',
            'XIX',
            'XX',
            'XXI',
            'XXII',
            'XXIII',
            'XXIV',
            'XXVI',
            't31']

        rmsd_dict = self.compute_dataset_rmsd(target_identifiers, target_identifiers_list, extra_test_loader)

        '''
        Get our RDF metric
        '''
        builder = SupercellBuilder(self.sym_ops, self.sym_info, self.normed_lattice_vectors, self.atom_weights, config.dataDims)

        rdfs_dict = {}

        for i, data in enumerate(tqdm.tqdm(extra_test_loader)):
            supercell_data = builder.build_supercells_from_dataset(data.clone(), config, return_energy=False, override_supercell_size=1)
            rdfs, rr, corr_labels = crystal_rdf(supercell_data, rrange=[0, 5], bins=500, elementwise=True, atomwise=False, intermolecular=False, raw_density=True)

            for j, ident in enumerate(supercell_data.csd_identifier):
                rdfs_dict[ident] = rdfs[j].cpu().detach().numpy()
            if i % 1000 == 0:
                np.save('../../test_data_rdfs', rdfs_dict)

        np.save('../../test_data_rdfs', rdfs_dict)
        '''
        generate rdf diffs
        '''
        rdf_diffs_dict = {}
        sigma = 10

        compare = lambda a, b: np.sum(np.abs(a - b)) / np.sum(np.abs(a))

        for i, key in enumerate(tqdm.tqdm(rdfs_dict)):
            '''
            identify the reference
            -> need also to account for the 3 structures of target 31
            '''
            crystal_target = None
            item = key
            for j in range(len(target_identifiers_list)):  # go in reverse to account for roman numerals system of duplication
                if target_identifiers_list[-1 - j] in item:
                    crystal_target = target_identifiers_list[-1 - j]
                    break

            '''
            compare rdfs
            '''

            sample_rdf = gaussian_filter1d(rdfs_dict[key], sigma=sigma)
            if crystal_target == 't31':
                target_rdf1 = gaussian_filter1d(rdfs_dict['2199671_p10167_1_0'], sigma=sigma)
                target_rdf2 = gaussian_filter1d(rdfs_dict['2199673_1_0'], sigma=sigma)
                # target_rdf3 = gaussian_filter1d(rdfs_dict['2199672_1_0'], sigma=sigma)

                rdf_diffs_dict[key] = [compare(target_rdf1, sample_rdf),
                                       compare(target_rdf2, sample_rdf)]
                # compare(target_rdf3, sample_rdf),]

            elif crystal_target is not None:
                target_rdf1 = gaussian_filter1d(rdfs_dict[target_identifiers[crystal_target]], sigma=sigma)
                rdf_diffs_dict[key] = compare(target_rdf1, sample_rdf)

        '''
        collate & generate diffs
        '''

        # omit the third entry, as it's not the right molecule
        datapoints = {key: [] for key in target_identifiers_list}

        for i, key in enumerate(rdf_diffs_dict):
            if rmsd_dict[key] != 'error':
                crystal_target = None
                item = key
                for j in range(len(target_identifiers_list)):  # go in reverse to account for roman numerals system of duplication
                    if target_identifiers_list[-1 - j] in item:
                        crystal_target = target_identifiers_list[-1 - j]
                        if crystal_target == 't31':
                            datapoints[crystal_target].append([rmsd_dict[key][:2], rdf_diffs_dict[key]])
                        else:
                            datapoints[crystal_target].append([rmsd_dict[key], rdf_diffs_dict[key]])
                        break
        ll = list(datapoints.keys())
        for key in ll:
            if len(datapoints[key]) == 0:
                datapoints.pop(key)
            else:
                datapoints[key] = np.asarray(datapoints[key])

        plt.figure(3)
        plt.clf()
        vals = []
        [vals.extend(aa[:, 1].flatten()) for aa in list(datapoints.values())]
        vals2 = []
        [vals2.extend(aa[:, 0].flatten()) for aa in list(datapoints.values())]
        ylim = max(vals)
        xlim = max(vals2)
        ind = 1
        for i, key in enumerate(datapoints.keys()):
            if key != 't31':
                if datapoints[key] != []:
                    data = datapoints[key]
                    plt.subplot(3, 3, ind)
                    ind += 1
                    plt.title(key)
                    plt.scatter(data[:, 0], data[:, 1], marker='.')
                    xline = np.asarray([0, np.amax(data[:, 0])])
                    linreg_result = linregress(data[:, 0], data[:, 1])
                    yline = xline * linreg_result.slope + linreg_result.intercept
                    plt.plot(xline, yline, label=f'R={linreg_result.rvalue:.3f}')
                    plt.legend()
                    plt.axis([0, xlim, 0, ylim])

            else:
                data1 = datapoints[key][..., 0]
                data2 = datapoints[key][..., 1]
                # data3 = datapoints[key][..., 2]

                plt.subplot(3, 3, ind)
                plt.title('XXXI_1')
                ind += 1
                plt.scatter(data1[:, 0], data1[:, 1], marker='.')
                xline = np.asarray([0, np.amax(data1[:, 0])])
                linreg_result = linregress(data1[:, 0], data1[:, 1])
                yline = xline * linreg_result.slope + linreg_result.intercept
                plt.plot(xline, yline, label=f'R={linreg_result.rvalue:.3f}')
                plt.legend()
                plt.axis([0, xlim, 0, ylim])

                plt.subplot(3, 3, ind)
                plt.title('XXXI_2')
                ind += 1
                plt.scatter(data2[:, 0], data2[:, 1], marker='.')
                xline = np.asarray([0, np.amax(data2[:, 0])])
                linreg_result = linregress(data2[:, 0], data2[:, 1])
                yline = xline * linreg_result.slope + linreg_result.intercept
                plt.plot(xline, yline, label=f'R={linreg_result.rvalue:.3f}')
                plt.legend()
                plt.axis([0, xlim, 0, ylim])

        plt.tight_layout()

        return None

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
        if d_tr_record is not None:
            hist = np.histogram(d_tr_record, bins=256, range=(np.amin(d_tr_record), np.quantile(d_tr_record, 0.9)))
            wandb.log({"Discriminator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(d_te_record, bins=256, range=(np.amin(d_te_record), np.quantile(d_te_record, 0.9)))
        wandb.log({"Discriminator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        if d_tr_record is not None:
            wandb.log({"D Train Loss Coeff. of Variation": np.sqrt(np.var(d_tr_record)) / np.average(d_tr_record)})
        wandb.log({"D Test Loss Coeff. of Variation": np.sqrt(np.var(d_te_record)) / np.average(d_te_record)})

        # log generator losses
        if g_tr_record is not None:
            hist = np.histogram(g_tr_record, bins=256, range=(np.amin(g_tr_record), np.quantile(g_tr_record, 0.9)))
            wandb.log({"Generator Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(g_te_record, bins=256, range=(np.amin(g_te_record), np.quantile(g_te_record, 0.9)))
        wandb.log({"Generator Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        if g_tr_record is not None:
            wandb.log({"G Train Loss Coeff. of Variation": np.sqrt(np.var(g_tr_record)) / np.average(g_tr_record)})
        wandb.log({"G Test Loss Coeff. of Variation": np.sqrt(np.var(g_te_record)) / np.average(g_te_record)})

        # log specific losses
        special_losses = {}
        special_losses['epoch'] = current_metrics['epoch']
        if train_epoch_stats_dict is not None:
            for key in train_epoch_stats_dict.keys():
                if ('loss' in key) and (train_epoch_stats_dict[key] is not None):
                    special_losses['Train ' + key] = np.average(train_epoch_stats_dict[key])
                if ('score' in key) and (train_epoch_stats_dict[key] is not None):
                    if (self.config.gan_loss == 'wasserstein') or (self.config.gan_loss == 'distance'):
                        score = train_epoch_stats_dict[key]
                    elif self.config.gan_loss == 'standard':
                        score = softmax_and_score(train_epoch_stats_dict[key])
                    special_losses['Train ' + key] = np.average(score)

        if test_epoch_stats_dict is not None:
            for key in test_epoch_stats_dict.keys():
                if ('loss' in key) and (test_epoch_stats_dict[key] is not None):
                    special_losses['Test ' + key] = np.average(test_epoch_stats_dict[key])
                if ('score' in key) and (test_epoch_stats_dict[key] is not None):
                    if (self.config.gan_loss == 'wasserstein') or (self.config.gan_loss == 'distance'):
                        score = test_epoch_stats_dict[key]
                    elif self.config.gan_loss == 'standard':
                        score = softmax_and_score(test_epoch_stats_dict[key])
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

    def log_gan_accuracy(self, epoch, train_loader,
                         train_epoch_stats_dict, test_epoch_stats_dict, config,
                         extra_test_dict=None):
        '''
        Do analysis and upload results to w&b
        '''
        if config.mode == 'gan':
            # if test_epoch_stats_dict['generated supercell examples dict'] is not None:
            #    self.log_molecules(config, test_epoch_stats_dict['generated supercell examples dict'])

            # if self.config.gan_loss == 'distance': # DEPRECATED
            #     self.gan_distance_regression_analysis(config, test_epoch_stats_dict)

            if train_epoch_stats_dict is not None:
                if test_epoch_stats_dict['generated cell parameters'] is not None:  # config.train_generator_density:
                    self.cell_params_analysis(config, train_loader, test_epoch_stats_dict)

                if train_epoch_stats_dict['generator density target'] is not None:  # config.train_generator_density:
                    self.log_aux_regression(config, train_epoch_stats_dict, test_epoch_stats_dict)
        elif config.mode == 'regression':
            self.log_regression_accuracy(config, train_epoch_stats_dict, test_epoch_stats_dict)

        if (extra_test_dict is not None) and (epoch % config.extra_test_period == 0):
            self.blind_test_analysis(config, train_epoch_stats_dict, test_epoch_stats_dict, extra_test_dict)

        return None

    def train_discriminator(self, generated_samples, discriminator, config, data, i, target_handedness=None, return_rdf=False):
        # generate fakes & create supercell data
        real_supercell_data = self.supercell_builder.build_supercells_from_dataset(data.clone(), config)
        fake_supercell_data, generated_cell_volumes, overlaps_list = \
            self.supercell_builder.build_supercells(data.clone().to(generated_samples.device), generated_samples,
                                                    config.supercell_size, config.discriminator.graph_convolution_cutoff,
                                                    override_sg=config.generate_sgs, target_handedness=target_handedness)

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
        '''
        plt.clf()
        real_rdf, rr, rdf_label_dict = crystal_rdf(real_supercell_data, rrange=[0, 10], bins=1000, intermolecular=False, elementwise=True, raw_density=True)
        plt.plot(rr.cpu().detach(), gaussian_filter1d(real_rdf[:,9,:].mean(0).cpu().detach(),5))
        plt.xlabel('Range (Angstrom)')
        plt.ylabel('RDF')
        '''
        if return_rdf:
            real_rdf, rr, rdf_label_dict = crystal_rdf(real_supercell_data, rrange=[0, 6], bins=1000, intermolecular=True, elementwise=True, raw_density=True)
            fake_rdf, rr, rdf_label_dict = crystal_rdf(fake_supercell_data, rrange=[0, 6], bins=1000, intermolecular=True, elementwise=True, raw_density=True)

            real_rdf_dict = {'rdf': real_rdf, 'range': rr, 'labels': rdf_label_dict}
            fake_rdf_dict = {'rdf': fake_rdf, 'range': rr, 'labels': rdf_label_dict}

            return score_on_real, score_on_fake, fake_supercell_data.cell_params.cpu().detach().numpy(), real_rdf_dict, fake_rdf_dict

        else:
            return score_on_real, score_on_fake, fake_supercell_data.cell_params.cpu().detach().numpy(), \
                   real_distances_dict, fake_pairwise_distances_dict, vdW_penalty(real_supercell_data, self.vdw_radii), vdW_penalty(fake_supercell_data, self.vdw_radii)

    def train_generator(self, generator, discriminator, config, data, i):
        # noise injection
        if config.generator.positional_noise > 0:
            data.pos += torch.randn_like(data.pos) * config.generator.positional_noise

        [[generated_samples, latent], prior, condition] = generator.forward(n_samples=data.num_graphs, conditions=data.to(config.device), return_latent=True, return_condition=True, return_prior=True)

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
        samples = generator.forward(n_samples=data.num_graphs, conditions=data.to(self.config.device))
        latent_samples = self.randn_generator.backward(samples.cpu())
        log_probs = self.randn_generator.score(latent_samples)

        return -log_probs  # want to maximize this objective

    def train_boilerplate(self):
        config = self.config
        # dataset
        if config.max_epochs == -1:
            pass  # self.nice_dataset_analysis(config, self.prep_dataset)

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

    def MCMC_sampling(self, discriminator, test_loader, sample_ind, sample_steps, move_size, test_epoch_stats_dict, config):
        '''
        Stun MC annealing on a pretrained discriminator
        '''
        from torch_geometric.loader.dataloader import Collater

        n_samples = self.config.final_batch_size
        single_mol_data = test_loader.dataset[sample_ind]
        collater = Collater(None, None)
        single_mol_data = collater([single_mol_data for n in range(n_samples)])
        self.randn_generator.cpu()

        '''
        initialize sampler
        '''
        smc_sampler = Sampler(
            gammas=np.logspace(-4, 0, n_samples),
            seedInd=0,
            acceptance_mode='stun',
            debug=True,
            init_temp=1,
            random_generator=self.randn_generator,
            move_size=move_size,
            supercell_size=self.config.supercell_size,
            graph_convolution_cutoff=self.config.discriminator.graph_convolution_cutoff,
            vdw_radii=self.vdw_radii,
            preset_minimum = np.quantile(softmax_and_score(test_epoch_stats_dict['discriminator real score']),0.05)
        )

        '''
        random batch of initial conditions
        '''
        init_samples = self.randn_generator.forward(single_mol_data, n_samples).cpu().detach().numpy()

        '''
        run sampling
        '''
        num_iters = sample_steps
        sampling_dict = smc_sampler(discriminator, self.supercell_builder,
                                    single_mol_data, init_samples, num_iters)

        np.save(f'../sampling_output_run_{self.config.run_num}', sampling_dict)

        # best_inds = np.argsort(sampling_dict['scores'].flatten())
        # best_samples = sampling_dict['samples'].reshape(12, n_samples * num_iters)[:, best_inds[:n_samples]].T
        #
        # best_supercells, _, _ = self.supercell_builder.build_supercells(single_mol_data.clone(), torch.Tensor(best_samples).cuda(),
        #                                                                 supercell_size=1, graph_convolution_cutoff=7)
        #
        # best_rdfs, rr = crystal_rdf(best_supercells, rrange=[0, 10], bins=100, intermolecular=True)
        self.report_sampling(test_epoch_stats_dict, sampling_dict, config.sample_ind)

        return sampling_dict

    def generate_discriminator_negatives(self, epoch_stats_dict, config, data, generator, i):
        n_generators = sum([config.train_discriminator_adversarially, config.train_discriminator_on_noise, config.train_discriminator_on_randn])
        gen_random_number = np.random.uniform(0, 1, 1)
        gen_randn_range = np.linspace(0, 1, n_generators + 1)

        if config.train_discriminator_adversarially:
            ii = i % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:  # randomly sample which generator to use at each iteration
                generated_samples_i = generator.forward(n_samples=data.num_graphs, conditions=data.to(config.device))
                handedness = None
                epoch_stats_dict['generator sample source'].extend(np.zeros(len(generated_samples_i)))

        if config.train_discriminator_on_randn:
            ii = (i + 1) % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:
                generated_samples_i = self.randn_generator.forward(data, data.num_graphs).to(config.device)
                handedness = None
                epoch_stats_dict['generator sample source'].extend(np.ones(len(generated_samples_i)))

        if config.train_discriminator_on_noise:
            ii = (i + 2) % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:
                generated_samples_ii = (data.cell_params - torch.Tensor(self.dataDims['lattice means'])) / torch.Tensor(self.dataDims['lattice stds'])  # standardize
                if config.generator_noise_level == -1:
                    distortion = torch.randn_like(generated_samples_ii) * torch.logspace(-2.5, -0.5, len(generated_samples_ii)).to(generated_samples_ii.device)[:, None]  # wider range for evaluation mode
                else:
                    distortion = torch.randn_like(generated_samples_ii) * config.generator_noise_level
                generated_samples_i = (generated_samples_ii + distortion).to(config.device)  # add jitter and return in standardized basis
                handedness = data.asym_unit_handedness
                epoch_stats_dict['generator sample source'].extend(np.ones(len(generated_samples_i)) * 2)
                epoch_stats_dict['distortion level'].extend(torch.linalg.norm(distortion, axis=-1).cpu().detach().numpy())

        return generated_samples_i, handedness, epoch_stats_dict

    def log_aux_regression(self, config, train_epoch_stats_dict, test_epoch_stats_dict):
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
        loss_dict['Regression R'] = linreg_result.rvalue
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

    def log_regression_accuracy(self, config, train_epoch_stats_dict, test_epoch_stats_dict):
        target_mean = self.dataDims['target mean']
        target_std = self.dataDims['target std']

        target = np.asarray(test_epoch_stats_dict['generator density target'])
        prediction = np.asarray(test_epoch_stats_dict['generator density prediction'])
        orig_target = target * target_std + target_mean
        orig_prediction = prediction * target_std + target_mean

        volume_ind = config.dataDims['tracking features dict'].index('molecule volume')
        mass_ind = config.dataDims['tracking features dict'].index('molecule mass')
        molwise_density = test_epoch_stats_dict['tracking features'][:, mass_ind] / test_epoch_stats_dict['tracking features'][:, volume_ind]
        target_density = molwise_density * orig_target * 1.66  # conversion from amu/A^3 to g/mL
        predicted_density = molwise_density * orig_prediction * 1.66

        if train_epoch_stats_dict is not None:
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
        if config.wandb.log_figures:
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

        return None

    def nice_dataset_analysis(self, config, dataset):
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
        fig.write_image('../paper1_figs/dataset_statistics.png')
        if config.machine == 'local':
            fig.show()

        aa = 1
        return None

    def nice_regression_plots(self, config, test_epoch_stats_dict):

        '''
        Calculations
        '''
        target_mean = self.dataDims['target mean']
        target_std = self.dataDims['target std']

        target = np.asarray(test_epoch_stats_dict['generator density target'])
        prediction = np.asarray(test_epoch_stats_dict['generator density prediction'])
        orig_target = target * target_std + target_mean
        orig_prediction = prediction * target_std + target_mean

        volume_ind = config.dataDims['tracking features dict'].index('molecule volume')
        mass_ind = config.dataDims['tracking features dict'].index('molecule mass')
        molwise_density = test_epoch_stats_dict['tracking features'][:, mass_ind] / test_epoch_stats_dict['tracking features'][:, volume_ind]
        target_density = molwise_density * orig_target * 1.66  # conversion from amu/A^3 to g/mL
        predicted_density = molwise_density * orig_prediction * 1.66

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

        losses = ['density normed error', 'density abs normed error', 'density squared error']
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

        '''
        summary table
        '''
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=20,  # top margin
            )
        )

        fig = go.Figure(data=go.Table(
            header=dict(values=['Metric', '$C_{pack}$', 'density']),
            cells=dict(values=[['MAE', '$\sigma$', 'R', 'Slope'],
                               [loss_dict['abs normed error mean'], loss_dict['abs normed error std'], loss_dict['Regression R'], loss_dict['Regression slope']],
                               [loss_dict['density abs normed error mean'], loss_dict['density abs normed error std'], loss_dict['Density Regression R'], loss_dict['Density Regression slope']],
                               ], format=["", ".3", ".3"])))
        fig.update_layout(width=300)
        fig.layout.margin = layout.margin
        fig.write_image('../paper1_figs/regression_topline.png')
        if config.machine == 'local':
            fig.show()

        '''
        4-panel error distribution
        '''
        from scipy.stats import gaussian_kde
        xy = np.vstack([orig_target, orig_prediction])
        z = gaussian_kde(xy)(xy)
        xy2 = np.vstack([target_density, predicted_density])
        z2 = gaussian_kde(xy)(xy)

        fig = make_subplots(rows=2, cols=2, subplot_titles=('a)', 'b)', 'c)', 'd)'),vertical_spacing=0.12)
        xline = np.linspace(max(min(orig_target), min(orig_prediction)), min(max(orig_target), max(orig_prediction)), 10)


        fig.add_trace(go.Scattergl(x=orig_target, y=orig_prediction, mode='markers',marker=dict(color=z), opacity=0.1),
                      row=1, col=1)
        fig.add_trace(go.Scattergl(x=xline, y=xline, marker_color='rgba(0,0,0,1)'), row=1, col=1)
        fig.update_layout(xaxis_title='targets', yaxis_title='predictions')

        fig.add_trace(go.Histogram(x=orig_target - orig_prediction,
                                   histnorm='probability density',
                                   nbinsx=500,
                                   name="Error Distribution",
                                   marker_color='rgba(0,0,100,1)'), row=2, col=1)

        xline = np.linspace(max(min(target_density), min(predicted_density)), min(max(target_density), max(predicted_density)), 10)
        fig.add_trace(go.Scattergl(x=target_density, y=predicted_density, mode='markers',marker=dict(color=z2), opacity = 0.1),
                      row=1, col=2)
        fig.add_trace(go.Scattergl(x=xline, y=xline, marker_color='rgba(0,0,0,1)'), row=1, col=2)
        fig.update_layout(xaxis_title='targets', yaxis_title='predictions')


        fig.add_trace(go.Histogram(x=target_density - predicted_density,
                                   histnorm='probability density',
                                   nbinsx=500,
                                   name="Error Distribution",
                                   marker_color='rgba(0,0,100,1)', ), row=2, col=2)
        fig.update_layout(showlegend=False)

        fig.update_yaxes(title_text='Predicted Packing Coefficient', row=1, col=1, dtick=0.05)
        fig.update_yaxes(title_text=r'$\text{Predicted Density }(g/cm^3)$', row=1, col=2, dtick=0.5)
        fig.update_xaxes(title_text='True Packing Coefficient', row=1, col=1, dtick=0.05)
        fig.update_xaxes(title_text=r'$\text{True Density }(g/cm^3)$', row=1, col=2, dtick=0.5)
        fig.update_xaxes(title_text='True-Predicted Packing Coefficient', row=2, col=1, dtick=0.05)
        fig.update_xaxes(title_text=r'$\text{True-Predicted Density }(g/cm^3)$', row=2, col=2, dtick=0.1)

        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

        fig.layout.annotations[0].update(x=0.025)
        fig.layout.annotations[2].update(x=0.025)
        fig.layout.annotations[1].update(x=0.575)
        fig.layout.annotations[3].update(x=0.575)

        fig.update_layout(height=600, width=800)
        fig.layout.margin = layout.margin

        fig.write_image('../paper1_figs/regression_distributions.png')
        if config.machine == 'local':
            fig.show()


        '''
        Error correlates
        '''
        # correlate losses with molecular features
        tracking_features = np.asarray(test_epoch_stats_dict['tracking features'])
        g_loss_correlations = np.zeros(config.dataDims['num tracking features'])
        features = []
        ind = 0
        for i in range(config.dataDims['num tracking features']):  # not that interesting
            if (np.average(tracking_features[:, i] != 0) > 0.05) and \
                    (config.dataDims['tracking features dict'][i] != 'crystal z prime') and \
                    (config.dataDims['tracking features dict'][i] != 'molecule point group is C1') and \
                    (config.dataDims['tracking features dict'][i] != 'crystal calculated density'):  # if we have at least 1# relevance

                coeff = np.corrcoef(np.abs((orig_target - orig_prediction) / np.abs(orig_target)), tracking_features[:, i], rowvar=False)[0, 1]
                if np.abs(coeff) > 0.05:
                    features.append(config.dataDims['tracking features dict'][i])
                    g_loss_correlations[ind] = coeff
                    ind += 1
        g_loss_correlations = g_loss_correlations[:ind]

        g_sort_inds = np.argsort(g_loss_correlations)
        g_loss_correlations = g_loss_correlations[g_sort_inds]
        features_sorted = [features[i] for i in g_sort_inds]
        features_sorted_cleaned_i = [feat.replace('molecule','mol') for feat in features_sorted]
        features_sorted_cleaned = [feat.replace('crystal','crys') for feat in features_sorted_cleaned_i]

        fig = go.Figure(go.Bar(
            y=features_sorted_cleaned,
            x=[corr for corr in g_loss_correlations],
            orientation='h',
            text=np.asarray([corr for corr in g_loss_correlations]).astype('float16'),
            textposition='auto',
            texttemplate='%{text:.2}'
        ))
        fig.layout.margin = layout.margin
        fig.update_layout(width=500, height=600, font=dict(size=12))
        fig.update_layout(xaxis_title='R Value')

        fig.write_image('../paper1_figs/regression_correlates.png')
        if config.machine == 'local':
            fig.show()

        return None

    def nice_scoring_plots(self, config):
        test_epoch_stats_dict = np.load('C:/Users\mikem\Desktop\CSP_runs\discriminator_713_test_epoch_stats_dict.npy', allow_pickle=True).item()
        extra_test_dict = np.load('C:/Users\mikem\Desktop\CSP_runs\discriminator_713_extra_test_dict.npy', allow_pickle=True).item()

        tracking_features = test_epoch_stats_dict['tracking features']
        identifiers_list = extra_test_dict['identifiers']
        score_correlations_dict, rdf_full_distance_dict, rdf_inter_distance_dict, \
        scores_dict, all_identifiers, blind_test_targets, target_identifiers, \
        target_identifiers_inds, BT_target_scores, BT_submission_scores, \
        BT_scores_dists, BT_balanced_dist, vdW_penalty_dict, tracking_features_dict = \
            self.process_discriminator_evaluation_data(config, extra_test_dict,
                                                       test_epoch_stats_dict, None, size_normed_score=False)

        del test_epoch_stats_dict
        del extra_test_dict

        '''
        new scoring decorrection?
        
        plt.clf()
        molecule_feature = 'molecule principal moment 1'
        x = tracking_features_dict['Test Real'][molecule_feature]
        y = scores_dict['Test Real']
        linreg_result = linregress(x,y)
        
        slope = linreg_result.slope
        intercept = linreg_result.intercept
        
        decor_y = (y - intercept) / (x)
        print(linreg_result.rvalue)
        linreg_result2 = linregress(x,decor_y)
        print(linreg_result2.rvalue)
        
        plt.subplot(1,2,1)
        plt.scatter(x,y/np.abs(np.amax(y)),alpha=0.025)
        plt.scatter(x,decor_y/np.abs(np.amax(decor_y)), alpha=0.025)

        '''

        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=20,  # top margin
            )
        )

        '''
        4. true-false model scores distribution
        '''
        lens = [len(val) for val in all_identifiers.values()]
        colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', max(np.count_nonzero(lens), np.count_nonzero(list(target_identifiers_inds.values()))), colortype='rgb')

        plot_color_dict = {}
        plot_color_dict['Test Real'] = ('rgb(250,150,50)')  # test
        plot_color_dict['Test Randn'] = ('rgb(0,50,0)')  # fake csd
        plot_color_dict['Test Distorted'] = ('rgb(0,100,100)')  # fake distortion
        ind = 0
        for target in all_identifiers.keys():
            if all_identifiers[target] != []:
                plot_color_dict[target] = colors[ind]
                plot_color_dict[target + ' exp'] = colors[ind]
                ind += 1

        scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
        bandwidth1 = scores_range / 200

        bandwidth2 = 15 / 200
        viridis = px.colors.sequential.Viridis

        scores_labels = {'Test Real': 'Real', 'Test Randn': 'Gaussian', 'Test Distorted': 'Distorted'}
        fig = make_subplots(rows=2, cols=2, subplot_titles=('a)', 'b)', 'c)'),
                            specs=[[{}, {}], [{"colspan": 2}, None]], vertical_spacing=0.14)

        for i, label in enumerate(scores_labels):
            legend_label = scores_labels[label]
            fig.add_trace(go.Violin(x=scores_dict[label], name=legend_label, line_color=plot_color_dict[label],
                                    side='positive', orientation='h', width=4,
                                    meanline_visible=True, bandwidth=bandwidth1, points=False),
                          row=1, col=1)
            fig.add_trace(go.Violin(x=-np.log(vdW_penalty_dict[label] + 1e-6), name=legend_label, line_color=plot_color_dict[label],
                                    side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth2, points=False),
                          row=1, col=2)

        all_vdws = np.concatenate((vdW_penalty_dict['Test Real'], vdW_penalty_dict['Test Randn'], vdW_penalty_dict['Test Distorted']))
        all_scores_i = np.concatenate((scores_dict['Test Real'], scores_dict['Test Randn'], scores_dict['Test Distorted']))

        rrange = np.logspace(3,0,len(viridis))
        cscale = [[1/rrange[i], viridis[i]] for i in range(len(rrange))]
        cscale[0][0]=0
        # colorscale = [
        #     [0, viridis[0]],
        #     [1. / 1000000, viridis[2]],
        #     [1. / 10000, viridis[4]],
        #     [1. / 100, viridis[7]],
        #     [1., viridis[9]],

        fig.add_trace(go.Histogram2d(x=all_scores_i,
                                     y=-np.log(all_vdws + 1e-6),
                                     showscale=False,
                                     nbinsy=50, nbinsx=200,
                                     colorscale=cscale,
                                     colorbar=dict(
                                         tick0=0,
                                         tickmode='array',
                                         tickvals=[0, 1000, 10000]
                                     )),
                      row=2, col=1)

        fig.update_layout(showlegend=False, yaxis_showgrid=True, width=800, height=500)
        fig.update_xaxes(title_text='Model Score', row=1, col=1)
        fig.update_xaxes(title_text='vdW Score', row=1, col=2)
        fig.update_xaxes(title_text='Model Score', row=2, col=1)
        fig.update_yaxes(title_text='vdW Score', row=2, col=1)

        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

        fig.layout.annotations[0].update(x=0.025)
        fig.layout.annotations[1].update(x=0.575)

        fig.layout.margin = layout.margin
        fig.write_image('../paper1_figs/real_vs_fake_scores.png')
        if config.machine == 'local':
            fig.show()

        '''
        5. BT scores distributions w aggregate inset
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

        # plot 1
        scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
        bandwidth = scores_range / 200

        fig = make_subplots(cols=2, rows=2, horizontal_spacing=0.15, subplot_titles=('a)', 'b)', 'c)'),
                            specs=[[{"rowspan": 2}, {}], [None, {}]], vertical_spacing=0.12)
        fig.layout.annotations[0].update(x=0.025)
        fig.layout.annotations[1].update(x=0.525)
        fig.layout.annotations[2].update(x=0.525)

        for i, label in enumerate(scores_dict.keys()):
            # if 'X' in label and 'exp' not in label:
            #     agrees_with_exp_sg = tracking_features_dict[label]['crystal spacegroup number'] == tracking_features_dict[label + ' exp']['crystal spacegroup number']
            #     fig.add_trace(go.Violin(x=scores_dict[label][agrees_with_exp_sg], name=label, line_color=plot_color_dict['Test NF'], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False),
            #                   row=1, col=1)

            if 'exp' in label:
                fig.add_trace(go.Violin(x=scores_dict[label], name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=6),
                              row=1, col=1)
            else:
                fig.add_trace(go.Violin(x=scores_dict[label], name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False),
                              row=1, col=1)

        # plot2 inset
        plot_color_dict = {}
        plot_color_dict['Test Real'] = ('rgb(200,0,50)')  # test
        plot_color_dict['BT Targets'] = ('rgb(50,0,50)')
        plot_color_dict['BT Submissions'] = ('rgb(50,150,250)')

        scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
        bandwidth = scores_range / 200

        # test data
        fig.add_trace(go.Violin(x=scores_dict['Test Real'], name='CSD Test',
                                line_color=plot_color_dict['Test Real'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=1, col=2)

        # BT distribution
        fig.add_trace(go.Violin(x=BT_target_scores, name='BT Targets',
                                line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth / 100, points=False), row=1, col=2)
        # Submissions
        fig.add_trace(go.Violin(x=BT_submission_scores, name='BT Submissions',
                                line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=1, col=2)

        quantiles = [np.quantile(scores_dict['Test Real'], 0.01), np.quantile(scores_dict['Test Real'], 0.05), np.quantile(scores_dict['Test Real'], 0.1)]
        fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['Test Real'], row=1, col=2)
        fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['Test Real'], row=1, col=2)
        fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['Test Real'], row=1, col=2)

        normed_scores_dict = scores_dict.copy()
        for key in normed_scores_dict.keys():
            normed_scores_dict[key] = normed_scores_dict[key] / tracking_features_dict[key]['molecule num atoms']

        normed_BT_target_scores = np.concatenate([normed_scores_dict[key] for key in normed_scores_dict.keys() if 'exp' in key])
        normed_BT_submission_scores = np.concatenate([normed_scores_dict[key] for key in normed_scores_dict.keys() if key in all_identifiers.keys()])
        scores_range = np.ptp(np.concatenate(list(normed_scores_dict.values())))
        bandwidth = scores_range / 200
        # test data
        fig.add_trace(go.Violin(x=normed_scores_dict['Test Real'], name='CSD Test',
                                line_color=plot_color_dict['Test Real'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=2, col=2)

        # BT distribution
        fig.add_trace(go.Violin(x=normed_BT_target_scores, name='BT Targets',
                                line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth / 100, points=False), row=2, col=2)
        # Submissions
        fig.add_trace(go.Violin(x=normed_BT_submission_scores, name='BT Submissions',
                                line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=2, col=2)

        quantiles = [np.quantile(normed_scores_dict['Test Real'], 0.01), np.quantile(normed_scores_dict['Test Real'], 0.05), np.quantile(normed_scores_dict['Test Real'], 0.1)]
        fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['Test Real'], row=2, col=2)
        fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['Test Real'], row=2, col=2)
        fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['Test Real'], row=2, col=2)

        fig.update_layout(showlegend=False, yaxis_showgrid=True)

        fig.update_layout(showlegend=False, yaxis_showgrid=True, width=1000, height=500)
        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_xaxes(title_text='Model Score', row=1, col=2)
        fig.update_xaxes(title_text='Model Score', row=1, col=1)
        fig.update_xaxes(title_text='Model Score / molecule # atoms', row=2, col=2)

        fig.layout.margin = layout.margin
        fig.write_image('../paper1_figs/bt_submissions_distribution.png')
        if config.machine == 'local':
            fig.show()

        '''
        7. Table of BT separation statistics
        '''
        vals = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
        quantiles = np.quantile(scores_dict['Test Real'], vals)
        submissions_fraction_below_csd_quantile = {value: np.average(BT_submission_scores < cutoff) for value, cutoff in zip(vals, quantiles)}

        normed_quantiles = np.quantile(normed_scores_dict['Test Real'], vals)
        normed_submissions_fraction_below_csd_quantile = {value: np.average(normed_BT_submission_scores < cutoff) for value, cutoff in zip(vals, normed_quantiles)}

        submissions_fraction_below_target = {key: np.average(scores_dict[key] < scores_dict[key + ' exp']) for key in all_identifiers.keys() if key in scores_dict.keys()}
        submissions_average_below_target = np.average(list(submissions_fraction_below_target.values()))

        fig = go.Figure(data=go.Table(
            header=dict(values=['CSD Test Quantile', 'Fraction of Submissions']),
            cells=dict(values=[list(submissions_fraction_below_csd_quantile.keys()),
                               list(submissions_fraction_below_csd_quantile.values()),
                               ], format=[".3", ".3"])))
        fig.update_layout(width=200)
        fig.layout.margin = layout.margin
        fig.write_image('../paper1_figs/scores_separation_table.png')
        if config.machine == 'local':
            fig.show()

        '''
        8. Functional group analysis
        '''
        tracking_features_names = config.dataDims['tracking features dict']
        # get the indices for each functional group
        functional_group_inds = {}
        fraction_dict = {}
        for ii, key in enumerate(tracking_features_names):
            if ('molecule' in key and 'fraction' in key):
                if np.average(tracking_features[:, ii] > 0) > 0.01:
                    fraction_dict[key.split()[1]] = np.average(tracking_features[:, ii] > 0)
                    functional_group_inds[key.split()[1]] = np.argwhere(tracking_features[:, ii] > 0)[:, 0]
            elif 'molecule has' in key:
                if np.average(tracking_features[:, ii] > 0) > 0.01:
                    fraction_dict[key.split()[2]] = np.average(tracking_features[:, ii] > 0)
                    functional_group_inds[key.split()[2]] = np.argwhere(tracking_features[:, ii] > 0)[:, 0]

        sort_order = np.argsort(list(fraction_dict.values()))[-1::-1]
        sorted_functional_group_keys = [list(functional_group_inds.keys())[i] for i in sort_order]
        #
        # colors = n_colors('rgb(100,10,5)', 'rgb(5,110,200)', len(list(functional_group_inds.keys())), colortype='rgb')
        # plot_color_dict = {}
        # for ind, target in enumerate(sorted_functional_group_keys):
        #     plot_color_dict[target] = colors[ind]
        #
        #
        # scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
        # bandwidth = scores_range / 200
        #
        # fig = go.Figure()
        # fig.add_trace(go.Violin(x=scores_dict['Test Real'], name='CSD Test',
        #                         line_color='#0c4dae', side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))
        #
        # for ii, label in enumerate(sorted_functional_group_keys):
        #     fraction = fraction_dict[label]
        #     if fraction > 0.01:
        #         fig.add_trace(go.Violin(x=scores_dict['Test Real'][functional_group_inds[label]], name=f'Fraction containing {label}={fraction:.2f}',
        #                                 line_color=plot_color_dict[label], side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))
        #
        # fig.update_layout(legend_traceorder='reversed', yaxis_showgrid=True)
        # fig.update_layout(xaxis_title='Model Score')
        # fig.update_layout(showlegend=False)
        #
        # fig.layout.margin = layout.margin
        # fig.write_image('../paper1_figs/scores_separation_table.png')
        # if config.machine == 'local':
        #     fig.show()

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=[f'{key} {fraction_dict[key]:.2f}' for key in sorted_functional_group_keys],
                                   y=[np.average(scores_dict['Test Real'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                                   error_y=dict(type='data',
                                                array=[np.std(scores_dict['Test Real'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                                                visible=True
                                                ),
                                   showlegend=False,
                                   mode='markers'))

        fig.update_layout(yaxis_title='Mean Score and Standard Deviation')
        fig.update_layout(width=1400, height=600)
        fig.update_layout(font=dict(size=12))
        fig.layout.margin = layout.margin
        fig.write_image('../paper1_figs/functional_group_scores.png')
        if config.machine == 'local':
            fig.show()

        '''
        9. Score vs. EMD on BT submissions
        '''

        all_scores = np.concatenate([(scores_dict[key]) for key in scores_dict.keys() if key in blind_test_targets])  # if key in normed_energy_dict.keys()])
        full_rdf = np.concatenate([val for val in rdf_full_distance_dict.values()])
        inter_rdf = np.concatenate([val for val in rdf_inter_distance_dict.values()])

        clip = np.quantile(full_rdf, 0.99) * 1.9
        # full_rdf = np.clip(full_rdf, a_min=0, a_max=clip)

        fig = make_subplots(rows=2, cols=4,
                            vertical_spacing=0.075,
                            subplot_titles=(list(rdf_full_distance_dict.keys())),
                            x_title='Distance from Target',
                            y_title='Model Score')  # + ['All'])

        from scipy.stats import gaussian_kde
        for i, label in enumerate(rdf_full_distance_dict.keys()):
            row = i // 4 + 1
            col = i % 4 + 1
            dist = rdf_full_distance_dict[label]
            dist = np.clip(dist, a_min=0, a_max=clip)
            xline = np.asarray([np.amin(dist), np.amax(dist)])
            linreg_result = linregress(dist, scores_dict[label])
            yline = xline * linreg_result.slope + linreg_result.intercept

            xy = np.vstack([dist, scores_dict[label]])
            z = gaussian_kde(xy)(xy)

            fig.add_trace(go.Scattergl(x=dist, y=scores_dict[label], showlegend=False,
                                       mode='markers', marker=dict(color=z), opacity=0.1),
                          row=row, col=col)
            fig.add_trace(go.Scattergl(x=np.zeros(1), y=scores_dict[label + ' exp'], showlegend=False, mode='markers',
                                       marker=dict(color='Black', size=10, line=dict(color='White', width=2))), row=row, col=col)

            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=row, col=col)
            fig.update_xaxes(range=[-5, clip], row=row, col=col)
        #
        # xline = np.asarray([np.amin(full_rdf), np.amax(full_rdf)])
        # linreg_result = linregress(full_rdf, all_scores)
        # yline = xline * linreg_result.slope + linreg_result.intercept
        # fig.add_trace(go.Scattergl(x=full_rdf, y=all_scores, showlegend=False,
        #                            mode='markers', ),
        #               row=2, col=4)
        # fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'All Targets R={linreg_result.rvalue:.3f}'), row=2, col=4)
        # fig.update_xaxes(range=[-5, clip], row=2, col=4)


        fig.update_layout(width=1000, height=500)
        fig.layout.margin = layout.margin
        fig.layout.margin.b = 60
        fig.layout.margin.l = 90
        fig.write_image('../paper1_figs/scores_vs_emd.png')
        if config.machine == 'local':
            fig.show()

        '''
        10. Interesting Group-wise analysis
        '''

        target_identifiers = {}
        rankings = {}
        group = {}
        list_num = {}
        for label in ['XXII', 'XXIII', 'XXVI']:
            target_identifiers[label] = [identifiers_list[all_identifiers[label][n]] for n in range(len(all_identifiers[label]))]
            rankings[label] = []
            group[label] = []
            list_num[label] = []
            for ident in target_identifiers[label]:
                if 'edited' in ident:
                    ident = ident[7:]

                long_ident = ident.split('_')
                list_num[label].append(int(ident[len(label) + 1]))
                rankings[label].append(int(long_ident[-1]) + 1)
                rankings[label].append(int(long_ident[-1]) + 1)
                group[label].append(long_ident[1])

        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.075, subplot_titles=(
            ['Brandenburg XXII', 'Brandenburg XXIII', 'Brandenburg XXVI', 'Facelli XXII']),
                            x_title='Model Score')

        quantiles = [np.quantile(normed_scores_dict['Test Real'], 0.01), np.quantile(normed_scores_dict['Test Real'], 0.05), np.quantile(normed_scores_dict['Test Real'], 0.1)]

        for ii, label in enumerate(['XXII', 'XXIII', 'XXVI']):
            good_inds = np.where(np.asarray(group[label]) == 'Brandenburg')[0]
            submissions_list_num = np.asarray(list_num[label])[good_inds]
            list1_inds = np.where(submissions_list_num == 1)[0]
            list2_inds = np.where(submissions_list_num == 2)[0]

            fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
                                       histnorm='probability density',
                                       nbinsx=50,
                                       name="Submission 1 Score",
                                       showlegend=False,
                                       marker_color='#0c4dae'),
                          row=(ii) // 2 + 1, col=(ii) % 2 + 1)

            fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
                                       histnorm='probability density',
                                       nbinsx=50,
                                       name="Submission 2 Score",
                                       showlegend=False,
                                       marker_color='#d60000'),
                          row=(ii) // 2 + 1, col=(ii) % 2 + 1)

        label = 'XXII'
        good_inds = np.where(np.asarray(group[label]) == 'Facelli')[0]
        submissions_list_num = np.asarray(list_num[label])[good_inds]
        list1_inds = np.where(submissions_list_num == 1)[0]
        list2_inds = np.where(submissions_list_num == 2)[0]

        fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
                                   histnorm='probability density',
                                   nbinsx=50,
                                   name="Submission 1 Score",
                                   showlegend=False,
                                   marker_color='#0c4dae'), row=2, col=2)
        fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
                                   histnorm='probability density',
                                   nbinsx=50,
                                   name="Submission 2 Score",
                                   showlegend=False,
                                   marker_color='#d60000'), row=2, col=2)

        fig.add_vline(x=quantiles[1], line_dash='dash', line_color='black', row=1, col=1)
        fig.add_vline(x=quantiles[1], line_dash='dash', line_color='black', row=1, col=2)
        fig.add_vline(x=quantiles[1], line_dash='dash', line_color='black', row=2, col=1)
        fig.add_vline(x=quantiles[1], line_dash='dash', line_color='black', row=2, col=2)

        fig.update_layout(width=1000, height=500)
        fig.layout.margin = layout.margin
        fig.layout.margin.b = 50
        fig.write_image('../paper1_figs/interesting_groups.png')
        if config.machine == 'local':
            fig.show()

        '''
        S1. All group-wise analysis
        '''

        for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
            names = np.unique(list(group[label]))
            uniques = len(names)
            rows = int(np.floor(np.sqrt(uniques)))
            cols = int(np.ceil(np.sqrt(uniques)) + 1)
            fig = make_subplots(rows=rows, cols=cols,
                                subplot_titles=(names), x_title='Group Ranking', y_title='Model Score', vertical_spacing=0.1)

            for j, group_name in enumerate(np.unique(group[label])):
                good_inds = np.where(np.asarray(group[label]) == group_name)[0]
                submissions_list_num = np.asarray(list_num[label])[good_inds]
                list1_inds = np.where(submissions_list_num == 1)[0]
                list2_inds = np.where(submissions_list_num == 2)[0]

                xline = np.asarray([0, max(np.asarray(rankings[label])[good_inds[list1_inds]])])
                linreg_result = linregress(np.asarray(rankings[label])[good_inds[list1_inds]], np.asarray(scores_dict[label])[good_inds[list1_inds]])
                yline = xline * linreg_result.slope + linreg_result.intercept

                fig.add_trace(go.Scattergl(x=np.asarray(rankings[label])[good_inds], y=np.asarray(scores_dict[label])[good_inds], showlegend=False,
                                           mode='markers', opacity=0.5, marker=dict(size=6, color=submissions_list_num, colorscale='portland', cmax=2, cmin=1, showscale=False)),
                              row=j // cols + 1, col=j % cols + 1)

                fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{group_name} R={linreg_result.rvalue:.3f}', line=dict(color='#0c4dae')), row=j // cols + 1, col=j % cols + 1)

                if len(list2_inds) > 0:
                    xline = np.asarray([0, max(np.asarray(rankings[label])[good_inds[list2_inds]])])
                    linreg_result2 = linregress(np.asarray(rankings[label])[good_inds[list2_inds]], np.asarray(scores_dict[label])[good_inds[list2_inds]])
                    yline2 = xline * linreg_result2.slope + linreg_result2.intercept
                    fig.add_trace(go.Scattergl(x=xline, y=yline2, name=f'{group_name} R={linreg_result2.rvalue:.3f}', line=dict(color='#d60000')), row=j // cols + 1, col=j % cols + 1)

            fig.update_layout(title=label)

            fig.update_layout(width=1200, height=600)
            fig.layout.margin = layout.margin
            fig.layout.margin.t = 50
            fig.layout.margin.b = 55
            fig.layout.margin.l = 60
            fig.write_image(f'../paper1_figs/groupwise_analysis_{i}.png')
            if config.machine == 'local':
                fig.show()

        '''
        S2.  score correlates
        '''

        # correlate losses with molecular features
        tracking_features = np.asarray(tracking_features)
        g_loss_correlations = np.zeros(config.dataDims['num tracking features'])
        features = []
        ind = 0
        for i in range(config.dataDims['num tracking features']):  # not that interesting
            if 'spacegroup' not in config.dataDims['tracking features dict'][i]:
                if (np.average(tracking_features[:, i] != 0) > 0.05) and \
                        (config.dataDims['tracking features dict'][i] != 'crystal z prime') and \
                        (config.dataDims['tracking features dict'][i] != 'molecule point group is C1'):  # if we have at least 1# relevance
                    corr = np.corrcoef(scores_dict['Test Real'], tracking_features[:, i], rowvar=False)[0, 1]
                    if np.abs(corr) > 0.05:
                        features.append(config.dataDims['tracking features dict'][i])
                        g_loss_correlations[ind] = corr
                        ind += 1

        g_loss_correlations = g_loss_correlations[:ind]

        g_sort_inds = np.argsort(g_loss_correlations)
        g_loss_correlations = g_loss_correlations[g_sort_inds]
        features_sorted = [features[i] for i in g_sort_inds]
        features_sorted_cleaned_i = [feat.replace('molecule','mol') for feat in features_sorted]
        features_sorted_cleaned_ii = [feat.replace('crystal','crys') for feat in features_sorted_cleaned_i]
        features_sorted_cleaned = [feat.replace('mol atom heavier than','>') for feat in features_sorted_cleaned_ii]

        functional_group_dict = {
            'NH0': 'tert amine',
            'para_hydroxylation': 'para-hydroxylation',
            'Ar_N': 'aromatic N',
            'aryl_methyl': 'aryl methyl',
            'Al_OH_noTert': 'non-tert al-hydroxyl',
            'C_O': 'carbonyl O',
            'Al_OH': 'al-hydroxyl',
        }
        ff = []
        for feat in features_sorted_cleaned:
            for func in functional_group_dict.keys():
                if func in feat:
                    feat = feat.replace(func,functional_group_dict[func])
            ff.append(feat)
        features_sorted_cleaned = ff

        g_loss_dict = {feat: corr for feat, corr in zip(features_sorted, g_loss_correlations)}

        fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14, subplot_titles=('a) Molecule & Crystal Features', 'b) Atom Fractions', 'c) Contains Functional Groups'), x_title='R Value')

        fig.add_trace(go.Bar(
            y=[feat for feat in features_sorted_cleaned if 'has' not in feat and 'fraction' not in feat],
            x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' not in feat],
            orientation='h',
            text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' not in feat]).astype('float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(100,0,0,1)')
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            y=[feat.replace('mol ', '').replace('fraction','') for feat in features_sorted_cleaned if 'has' not in feat and 'fraction' in feat],
            x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' in feat],
            orientation='h',
            text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' in feat]).astype('float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(0,0,100,1)')
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            y=[feat.replace('mol has ', '') for feat in features_sorted_cleaned if 'has' in feat and 'fraction' not in feat],
            x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' in feat and 'fraction' not in feat],
            orientation='h',
            text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' in feat and 'fraction' not in feat]).astype('float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(0,100,0,1)')
        ), row=1, col=3)

        fig.update_yaxes(tickfont=dict(size=14), row=1, col=1)
        fig.update_yaxes(tickfont=dict(size=14), row=1, col=2)
        fig.update_yaxes(tickfont=dict(size=13), row=1, col=3)

        fig.layout.annotations[0].update(x=0.12)
        fig.layout.annotations[1].update(x=0.45)
        fig.layout.annotations[2].update(x=0.88)

        fig.layout.margin = layout.margin
        fig.layout.margin.b = 50
        fig.update_xaxes(range=[np.amin(list(g_loss_dict.values())), np.amax(list(g_loss_dict.values()))])
        fig.update_layout(width=1200, height=400)
        fig.update_layout(showlegend=False)
        fig.write_image('../paper1_figs/scores_correlates.png')
        if config.machine == 'local':
            fig.show()

        aa = 1

    def cell_params_analysis(self, config, train_loader, test_epoch_stats_dict):
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

        if config.wandb.log_figures:
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

    def log_molecules(self, config, generated_supercell_examples_dict):
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

    def gan_distance_regression_analysis(self, config, test_epoch_stats_dict):
        test_fake_predictions = test_epoch_stats_dict['discriminator fake score']
        test_real_predictions = test_epoch_stats_dict['discriminator real score']
        test_fake_distances = test_epoch_stats_dict['generated sample distances']
        test_real_distances = np.zeros_like(test_real_predictions)
        orig_target = np.concatenate((test_fake_distances, test_real_distances))
        orig_prediction = np.concatenate((test_fake_predictions, test_real_predictions))

        xline = np.linspace(max(min(orig_target), min(orig_prediction)), min(max(orig_target), max(orig_prediction)), 10)
        fig = go.Figure()
        fig.add_trace(go.Histogram2dContour(x=orig_target, y=orig_prediction, ncontours=50, nbinsx=40, nbinsy=40, showlegend=True))
        fig.update_traces(contours_coloring="fill")
        fig.update_traces(contours_showlines=False)
        fig.add_trace(go.Scattergl(x=orig_target, y=orig_prediction, mode='markers', showlegend=True, opacity=0.5))
        fig.add_trace(go.Scattergl(x=xline, y=xline))
        fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
        wandb.log({'Distance Trace': fig})

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

    def blind_test_analysis(self, config, train_epoch_stats_dict, test_epoch_stats_dict, extra_test_dict):
        '''
        analyze and plot
        '''
        aa = 1
        identifiers_list = extra_test_dict['identifiers']
        score_correlations_dict, rdf_full_distance_dict, rdf_inter_distance_dict, \
        scores_dict, all_identifiers, blind_test_targets, target_identifiers, \
        target_identifiers_inds, BT_target_scores, BT_submission_scores, \
        BT_scores_dists, BT_balanced_dist, vdW_penalty_dict, tracking_features_dict = \
            self.process_discriminator_evaluation_data(config, extra_test_dict, test_epoch_stats_dict, train_epoch_stats_dict)

        self.layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=20,  # top margin
            )
        )

        self.violin_scores_plot(all_identifiers, scores_dict, target_identifiers_inds)
        self.violin_vdW_plot(all_identifiers, vdW_penalty_dict, target_identifiers_inds)
        self.violin_scores_plot2(all_identifiers, scores_dict, BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist)
        self.functional_group_violin_plot(scores_dict, tracking_features_names=config.dataDims['tracking features dict'], tracking_features=test_epoch_stats_dict['tracking features'])
        self.scores_distributions_plot(all_identifiers, scores_dict, BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist)
        self.score_correlations_plot(test_epoch_stats_dict['tracking features'], scores_dict, config)
        self.distance_vs_score_plot(rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, blind_test_targets)
        self.targetwise_distance_vs_score_plot(rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, blind_test_targets)
        group, rankings, list_num = self.target_ranking_analysis(identifiers_list, scores_dict, all_identifiers)
        self.groupwise_target_ranking_analysis(group, rankings, list_num, scores_dict)

        # self.bt_clustering(all_identifiers, scores_dict, extra_test_dict)
        '''
        plt.clf()
        plt.scatter(scores_dict['Test Distorted'],np.log10(vdW_penalty_dict['Test Distorted']+np.amin(vdW_penalty_dict['Test Distorted'][np.nonzero(vdW_penalty_dict['Test Distorted'])])),c=np.log(test_epoch_stats_dict['distortion level']))
        plt.ylabel('vdW Penalty')
        plt.xlabel('Model score')
        plt.title('scores vs distortion size')
        
        plt.clf()
        ind = 1
        for key in vdW_penalty_dict.keys():
            if 'exp' not in key:
                plt.subplot(3,4,ind)
                ind += 1
                plt.scatter(x=scores_dict[key],
                            y=np.log10(vdW_penalty_dict[key] + 1e-6))
                plt.ylim([-7,1])
                plt.xlim([-16,10])
                
                xline = np.asarray([np.amin(scores_dict[key]), np.amax(scores_dict[key])])
                linreg_result = linregress(scores_dict[key], vdW_penalty_dict[key])
                yline = xline * linreg_result.slope + linreg_result.intercept
                plt.plot(xline,np.log10(yline),'r.-')
                plt.title(key + f' {linreg_result.rvalue:.3f}')
        '''
        aa = 1

    def make_nice_figures(self, config):
        '''
        make beautiful figures for paper / presentation
        '''
        import plotly.io as pio
        pio.renderers.default = 'browser'
        '''
        Required figures
        Data
        0. Dataset statistics analysis
        
        Regression
        1x. 4-panel packing coefficient & density, prediction trace & error distribution
        2x. Table of packing coefficient & density metrics
        3x. Packing coefficient error correlates

        Scoring
        4. true-false model scores distribution
        6. BT scores distributions w aggregate inset
        7. Table of BT separation statistics
        8. Functional group analysis
        9. Score vs. EMD on BT submissions
        10. Interesting Group-wise analysis
        
        SI
        S1. All group-wise analysis
        S2. Scoring score correlates
        '''

        regression_test_epoch_stats_dict = np.load('C:/Users\mikem\Desktop\CSP_runs/good_regression_test_epoch_stats_dict.npy', allow_pickle=True).item()
        self.nice_regression_plots(config, regression_test_epoch_stats_dict)
        del regression_test_epoch_stats_dict

        self.nice_scoring_plots(config)

    def process_discriminator_evaluation_data(self, config, extra_test_dict, test_epoch_stats_dict, train_epoch_stats_dict, size_normed_score=False):
        blind_test_targets = [  # 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
            'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
            'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI', 'XXXII']

        target_identifiers = {
            'XVI': 'OBEQUJ',
            'XVII': 'OBEQOD',
            'XVIII': 'OBEQET',
            'XIX': 'XATJOT',
            'XX': 'OBEQIX',
            'XXI': 'KONTIQ',
            'XXII': 'NACJAF',
            'XXIII': 'XAFPAY',
            'XXIII_1': 'XAFPAY01',
            'XXIII_2': 'XAFPAY02',
            'XXXIII_3': 'XAFPAY03',
            'XXXIII_4': 'XAFPAY04',
            'XXIV': 'XAFQON',
            'XXVI': 'XAFQIH',
            'XXXI_1': '2199671_p10167_1_0',
            'XXXI_2': '2199673_1_0',
            # 'XXXI_3': '2199672_1_0',
        }

        # determine which samples go with which targets
        all_identifiers = {key: [] for key in blind_test_targets}
        for i in range(len(extra_test_dict['identifiers'])):
            item = extra_test_dict['identifiers'][i]
            for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
                if blind_test_targets[-1 - j] in item:
                    all_identifiers[blind_test_targets[-1 - j]].append(i)
                    break

        # determine which samples ARE the targets (mixed in the dataloader)
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
        vdW_penalty_dict = {}
        tracking_features_dict = {}
        # nf_inds = np.where(test_epoch_stats_dict['generator sample source'] == 0)
        randn_inds = np.where(test_epoch_stats_dict['generator sample source'] == 1)[0]
        distorted_inds = np.where(test_epoch_stats_dict['generator sample source'] == 2)[0]

        if self.config.gan_loss == 'standard':
            scores_dict['Test Real'] = softmax_and_score(test_epoch_stats_dict['discriminator real score'])
            scores_dict['Test Randn'] = softmax_and_score(test_epoch_stats_dict['discriminator fake score'][randn_inds])
            # scores_dict['Test NF'] = np_softmax(test_epoch_stats_dict['discriminator fake score'][nf_inds])[:, 1]
            scores_dict['Test Distorted'] = softmax_and_score(test_epoch_stats_dict['discriminator fake score'][distorted_inds])

            tracking_features_dict['Test Real'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'].T)}
            tracking_features_dict['Test Distorted'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'][distorted_inds].T)}
            tracking_features_dict['Test Randn'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'][randn_inds].T)}

            if size_normed_score:
                scores_dict['Test Real'] = norm_scores(scores_dict['Test Real'], test_epoch_stats_dict['tracking features'], config)
                scores_dict['Test Randn'] = norm_scores(scores_dict['Test Randn'], test_epoch_stats_dict['tracking features'][randn_inds], config)
                scores_dict['Test Distorted'] = norm_scores(scores_dict['Test Distorted'], test_epoch_stats_dict['tracking features'][distorted_inds], config)

            if train_epoch_stats_dict is not None:
                scores_dict['Train Real'] = softmax_and_score(train_epoch_stats_dict['discriminator real score'])
                tracking_features_dict['Train Real'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], train_epoch_stats_dict['tracking features'].T)}

                if size_normed_score:
                    scores_dict['Train Real'] = norm_scores(scores_dict['Train Real'], train_epoch_stats_dict['tracking features'], config)

                vdW_penalty_dict['Train Real'] = train_epoch_stats_dict['real vdW penalty']
                wandb.log({'Average Train score': np.average(scores_dict['Train Real'])})
                wandb.log({'Train score std': np.std(scores_dict['Train Real'])})

            vdW_penalty_dict['Test Real'] = test_epoch_stats_dict['real vdW penalty']
            vdW_penalty_dict['Test Randn'] = test_epoch_stats_dict['fake vdW penalty'][randn_inds]
            vdW_penalty_dict['Test Distorted'] = test_epoch_stats_dict['fake vdW penalty'][distorted_inds]

            wandb.log({'Average Test score': np.average(scores_dict['Test Real'])})
            wandb.log({'Average Randn Fake score': np.average(scores_dict['Test Randn'])})
            # wandb.log({'Average NF Fake score': np.average(scores_dict['Test NF'])})
            wandb.log({'Average Distorted Fake score': np.average(scores_dict['Test Distorted'])})

            wandb.log({'Test Real std': np.std(scores_dict['Test Real'])})
            wandb.log({'Distorted Fake score std': np.std(scores_dict['Test Distorted'])})
            wandb.log({'Randn score std': np.std(scores_dict['Test Randn'])})

        else:
            print("Analysis only setup for cross entropy loss")
            assert False

        '''
        build property dicts for the submissions and BT targets
        '''
        score_correlations_dict = {}
        rdf_full_distance_dict = {}
        rdf_inter_distance_dict = {}

        for target in all_identifiers.keys():  # run the analysis for each target
            if target_identifiers_inds[target] != []:  # record target data

                target_index = target_identifiers_inds[target]
                raw_scores = extra_test_dict['scores'][target_index]
                scores = softmax_and_score(raw_scores)
                scores_dict[target + ' exp'] = scores

                tracking_features_dict[target + ' exp'] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], extra_test_dict['tracking features'][target_index][None, :].T)}

                if size_normed_score:
                    scores_dict[target + ' exp'] = norm_scores(scores_dict[target + ' exp'], extra_test_dict['tracking features'][target_index][None, :], config)

                vdW_penalty_dict[target + ' exp'] = extra_test_dict['vdW penalty'][target_index][None]

                wandb.log({f'Average {target} exp score': np.average(scores)})

                target_full_rdf = extra_test_dict['full rdf'][target_index]
                target_inter_rdf = extra_test_dict['intermolecular rdf'][target_index]

            if all_identifiers[target] != []:  # record sample data
                target_indices = all_identifiers[target]
                raw_scores = extra_test_dict['scores'][target_indices]
                scores = softmax_and_score(raw_scores)
                scores_dict[target] = scores
                tracking_features_dict[target] = {feat: vec for feat, vec in zip(self.config.dataDims['tracking features dict'], extra_test_dict['tracking features'][target_indices].T)}

                if size_normed_score:
                    scores_dict[target] = norm_scores(scores_dict[target], extra_test_dict['tracking features'][target_indices], config)

                # energy_dict[target] = extra_test_dict['atomistic energy'][target_indices]
                vdW_penalty_dict[target] = extra_test_dict['vdW penalty'][target_indices]

                wandb.log({f'Average {target} score': np.average(scores)})
                wandb.log({f'Average {target} std': np.std(scores)})

                submission_full_rdf = extra_test_dict['full rdf'][target_indices]
                submission_inter_rdf = extra_test_dict['intermolecular rdf'][target_indices]

                rdf_full_distance_dict[target] = compute_rdf_distance(target_full_rdf, submission_full_rdf)
                rdf_inter_distance_dict[target] = compute_rdf_distance(target_inter_rdf, submission_inter_rdf)

                # correlate losses with molecular features
                tracking_features = np.asarray(extra_test_dict['tracking features'])
                loss_correlations = np.zeros(config.dataDims['num tracking features'])
                features = []
                for j in range(tracking_features.shape[-1]):  # not that interesting
                    features.append(config.dataDims['tracking features dict'][j])
                    loss_correlations[j] = np.corrcoef(scores, tracking_features[target_indices, j], rowvar=False)[0, 1]

                score_correlations_dict[target] = loss_correlations

        # compute loss correlates
        loss_correlations = np.zeros(config.dataDims['num tracking features'])
        features = []
        for j in range(config.dataDims['num tracking features']):  # not that interesting
            features.append(config.dataDims['tracking features dict'][j])
            loss_correlations[j] = np.corrcoef(scores_dict['Test Real'], test_epoch_stats_dict['tracking features'][:, j], rowvar=False)[0, 1]
        score_correlations_dict['Test Real'] = loss_correlations

        # collect all BT targets & submissions into single dicts
        BT_target_scores = np.concatenate([scores_dict[key] for key in scores_dict.keys() if 'exp' in key])
        BT_submission_scores = np.concatenate([scores_dict[key] for key in scores_dict.keys() if key in all_identifiers.keys()])
        BT_scores_dists = {key: np.histogram(scores_dict[key], bins=200, range=[-15, 15])[0] / len(scores_dict[key]) for key in scores_dict.keys() if key in all_identifiers.keys()}
        BT_balanced_dist = np.average(np.stack(list(BT_scores_dists.values())), axis=0)

        wandb.log({'Average BT submission score': np.average(BT_submission_scores)})
        wandb.log({'Average BT target score': np.average(BT_target_scores)})
        wandb.log({'BT submission score std': np.std(BT_target_scores)})
        wandb.log({'BT target score std': np.std(BT_target_scores)})

        return score_correlations_dict, rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, \
               all_identifiers, blind_test_targets, target_identifiers, target_identifiers_inds, \
               BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist, \
               vdW_penalty_dict, tracking_features_dict

    def violin_scores_plot(self, all_identifiers, scores_dict, target_identifiers_inds):
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
        scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
        bandwidth = scores_range / 200

        fig = go.Figure()
        for i, label in enumerate(scores_dict.keys()):
            if 'exp' in label:
                fig.add_trace(go.Violin(x=scores_dict[label], name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=6))
            else:
                fig.add_trace(go.Violin(x=scores_dict[label], name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False))

            quantiles = [np.quantile(scores_dict['Test Real'], 0.01), np.quantile(scores_dict['Test Real'], 0.05), np.quantile(scores_dict['Test Real'], 0.1)]
        fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['Test Real'])
        fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['Test Real'])
        fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['Test Real'])

        fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
        fig.update_layout(xaxis_title='Model Score')
        fig.update_layout(font=dict(size=18))
        fig.layout.margin = self.layout.margin
        wandb.log({'Discriminator Test Scores': fig})

    def violin_vdW_plot(self, all_identifiers, vdW_penalty_dict, target_identifiers_inds):
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
        scores_range = np.ptp(np.log10(1e-6 + np.concatenate(list(vdW_penalty_dict.values()))))
        bandwidth = scores_range / 200

        fig = go.Figure()
        for i, label in enumerate(vdW_penalty_dict.keys()):
            if 'exp' in label:
                fig.add_trace(go.Violin(x=np.log10(vdW_penalty_dict[label] + 1e-6), name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=6))
            else:
                fig.add_trace(go.Violin(x=np.log10(vdW_penalty_dict[label] + 1e-6), name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False))

        fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
        fig.update_layout(xaxis_title='Model Score')
        fig.update_layout(font=dict(size=18))
        fig.layout.margin = self.layout.margin
        fig.update_layout(xaxis_title='log vdW Penalty')
        wandb.log({'vdW Penalty': fig})

    def violin_scores_plot2(self, all_identifiers, scores_dict, BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist):

        plot_color_dict = {}
        plot_color_dict['Test Real'] = ('rgb(200,0,50)')  # test
        plot_color_dict['BT Targets'] = ('rgb(50,0,50)')
        plot_color_dict['BT Submissions'] = ('rgb(50,150,250)')

        '''
        violin scores plot
        '''
        scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
        bandwidth = scores_range / 200

        fig = go.Figure()
        # 99 and 95 quantiles
        quantiles = [np.quantile(scores_dict['Test Real'], 0.01), np.quantile(scores_dict['Test Real'], 0.05), np.quantile(scores_dict['Test Real'], 0.1)]
        fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['Test Real'])
        fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['Test Real'])
        fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['Test Real'])
        # test data
        fig.add_trace(go.Violin(x=scores_dict['Test Real'], name='CSD Test', line_color=plot_color_dict['Test Real'], side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))

        # BT distribution
        fig.add_trace(go.Violin(x=BT_target_scores, name='BT Targets', line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1, meanline_visible=True, bandwidth=bandwidth / 100, points=False))
        # Submissions
        fig.add_trace(go.Violin(x=BT_submission_scores, name='BT Submissions', line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))
        quantiles = [np.quantile(BT_submission_scores, 0.01), np.quantile(BT_submission_scores, 0.05), np.quantile(BT_submission_scores, 0.1)]
        fig.add_shape(type="line",
                      x0=quantiles[0], y0=2, x1=quantiles[0], y1=3,
                      line=dict(color=plot_color_dict['BT Submissions'], dash='dash'))
        fig.add_shape(type="line",
                      x0=quantiles[1], y0=2, x1=quantiles[1], y1=3,
                      line=dict(color=plot_color_dict['BT Submissions'], dash='dash'))
        fig.add_shape(type="line",
                      x0=quantiles[2], y0=2, x1=quantiles[2], y1=3,
                      line=dict(color=plot_color_dict['BT Submissions'], dash='dash'))
        fig.update_layout(legend_traceorder='reversed', yaxis_showgrid=True)
        fig.update_layout(xaxis_title='Model Score')
        # fig.show()
        fig.update_layout(title='Scores and 0.01, 0.05, 0.1 quantiles')
        fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
        fig.update_layout(xaxis_title='Model Score')
        fig.update_layout(font=dict(size=18))
        fig.layout.margin = self.layout.margin
        fig.layout.margin.t = 50
        wandb.log({'Scores Distribution': fig})

        return None

    def functional_group_violin_plot(self, scores_dict, tracking_features_names, tracking_features):
        '''
        plot scores distributions for different functional groups
        '''

        # get the indices for each functional group
        functional_group_inds = {}
        fraction_dict = {}
        for ii, key in enumerate(tracking_features_names):
            if ('molecule' in key and 'fraction' in key):
                if np.average(tracking_features[:, ii] > 0) > 0.01:
                    fraction_dict[key.split()[1]] = np.average(tracking_features[:, ii] > 0)
                    functional_group_inds[key.split()[1]] = np.argwhere(tracking_features[:, ii] > 0)[:, 0]
            elif 'molecule has' in key:
                if np.average(tracking_features[:, ii] > 0) > 0.01:
                    fraction_dict[key.split()[2]] = np.average(tracking_features[:, ii] > 0)
                    functional_group_inds[key.split()[2]] = np.argwhere(tracking_features[:, ii] > 0)[:, 0]

        sort_order = np.argsort(list(fraction_dict.values()))[-1::-1]
        sorted_functional_group_keys = [list(functional_group_inds.keys())[i] for i in sort_order]

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=[f'{key} {fraction_dict[key]:.2f}' for key in sorted_functional_group_keys],
                                   y=[np.average(scores_dict['Test Real'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                                   error_y=dict(type='data',
                                                array=[np.std(scores_dict['Test Real'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                                                visible=True
                                                ),
                                   showlegend=False,
                                   mode='markers'))

        fig.update_layout(xaxis_title='Molecule Containing Functional Groups & Elements')
        fig.update_layout(yaxis_title='Mean Score and Standard Deviation')
        fig.layout.margin = self.layout.margin

        wandb.log({'Functional Group Scores Statistics': fig})

        # '''
        # violin scores plot
        # '''
        # scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
        # bandwidth = scores_range / 200
        #
        # fig = go.Figure()
        # fig.add_trace(go.Violin(x=scores_dict['Test Real'], name='CSD Test',
        #                         line_color='#0c4dae', side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))
        #
        # for ii, label in enumerate(sorted_functional_group_keys):
        #     fraction = fraction_dict[label]
        #     if fraction > 0.01:
        #         fig.add_trace(go.Violin(x=scores_dict['Test Real'][functional_group_inds[label]], name=f'Fraction containing {label}={fraction:.2f}',
        #                                 line_color=plot_color_dict[label], side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))
        #
        # fig.update_layout(legend_traceorder='reversed', yaxis_showgrid=True)
        # fig.update_layout(xaxis_title='Model Score')
        # fig.update_layout(showlegend=False)
        # fig.show()

        # wandb.log({'Functional Group Scores Distributions': fig})

        return None

    def scores_distributions_plot(self, all_identifiers, scores_dict, BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist):
        '''
        compute fraction of submissions below each quantile of the CSD data
        compute fraction of submissions above & below each experimental structrue
        '''
        csd_scores = scores_dict['Test Real']
        hists = {}
        scores_hist, rr = np.histogram(csd_scores, bins=200, range=[-15, 15], density=True)
        hists['Test Real'] = scores_hist / scores_hist.sum()

        submissions_hist, _ = np.histogram(BT_submission_scores, bins=200, range=[-15, 15], density=True)
        hists['BT submissions'] = submissions_hist / submissions_hist.sum()

        distorted_hist, rr = np.histogram(scores_dict['Test Distorted'], bins=200, range=[-15, 15], density=True)
        hists['Test Distorted'] = distorted_hist / distorted_hist.sum()

        randn_hist, rr = np.histogram(scores_dict['Test Randn'], bins=200, range=[-15, 15], density=True)
        hists['Test Randn'] = randn_hist / randn_hist.sum()

        emds = {}
        overlaps = {}
        for i, label1 in enumerate(hists.keys()):
            for j, label2 in enumerate(hists.keys()):
                if i > j:
                    emds[f'{label1} <-> {label2} emd'] = earth_movers_distance_np(hists[label1], hists[label2])
                    overlaps[f'{label1} <-> {label2} overlap'] = histogram_overlap(hists[label1], hists[label2])

        wandb.log(emds)
        wandb.log(overlaps)

        vals = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
        quantiles = np.quantile(csd_scores, vals)
        submissions_fraction_below_csd_quantile = {value: np.average(BT_submission_scores < cutoff) for value, cutoff in zip(vals, quantiles)}
        targets_fraction_below_csd_quantile = {value: np.average(BT_target_scores < cutoff) for value, cutoff in zip(vals, quantiles)}

        submissions_fraction_below_target = {key: np.average(scores_dict[key] < scores_dict[key + ' exp']) for key in all_identifiers.keys() if key in scores_dict.keys()}
        submissions_average_below_target = np.average(list(submissions_fraction_below_target.values()))

        distributions_dict = {
            'Submissions below CSD quantile': submissions_fraction_below_csd_quantile,
            'Targets below CSD quantile': targets_fraction_below_csd_quantile,
            'Submissions below target': submissions_fraction_below_target,
            'Submissions below target mean': submissions_average_below_target,
        }
        wandb.log(distributions_dict)

        return None

    def score_correlations_plot(self, tracking_features, scores_dict, config):

        # correlate losses with molecular features
        tracking_features = np.asarray(tracking_features)
        g_loss_correlations = np.zeros(config.dataDims['num tracking features'])
        features = []
        ind = 0
        for i in range(config.dataDims['num tracking features']):  # not that interesting
            if 'spacegroup' not in config.dataDims['tracking features dict'][i]:
                if (np.average(tracking_features[:, i] != 0) > 0.01) and \
                        (config.dataDims['tracking features dict'][i] != 'crystal z prime') and \
                        (config.dataDims['tracking features dict'][i] != 'molecule point group is C1'):  # if we have at least 1# relevance
                    corr = np.corrcoef(scores_dict['Test Real'], tracking_features[:, i], rowvar=False)[0, 1]
                    if np.abs(corr) > 0.05:
                        features.append(config.dataDims['tracking features dict'][i])
                        g_loss_correlations[ind] = corr
                        ind += 1

        g_loss_correlations = g_loss_correlations[:ind]

        g_sort_inds = np.argsort(g_loss_correlations)
        g_loss_correlations = g_loss_correlations[g_sort_inds]
        features_sorted = [features[i] for i in g_sort_inds]
        g_loss_dict = {feat: corr for feat, corr in zip(features_sorted, g_loss_correlations)}

        fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14, subplot_titles=('a)', 'b)', 'c)'), x_title='R Value')

        fig.add_trace(go.Bar(
            y=[feat for feat in features_sorted if 'has' not in feat and 'fraction' not in feat],
            x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' not in feat],
            orientation='h',
            text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' not in feat]).astype('float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(100,0,0,1)')
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            y=[feat for feat in features_sorted if 'has' in feat and 'fraction' not in feat],
            x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' in feat and 'fraction' not in feat],
            orientation='h',
            text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' in feat and 'fraction' not in feat]).astype('float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(0,100,0,1)')
        ), row=1, col=3)
        fig.add_trace(go.Bar(
            y=[feat for feat in features_sorted if 'has' not in feat and 'fraction' in feat],
            x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' in feat],
            orientation='h',
            text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' in feat]).astype('float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(0,0,100,1)')
        ), row=1, col=2)

        fig.update_yaxes(tickfont=dict(size=14), row=1, col=1)
        fig.update_yaxes(tickfont=dict(size=14), row=1, col=2)
        fig.update_yaxes(tickfont=dict(size=10), row=1, col=3)
        fig.update_xaxes(range=[np.amin(list(g_loss_dict.values())), np.amax(list(g_loss_dict.values()))])

        fig.layout.annotations[0].update(x=0.02)
        fig.layout.annotations[1].update(x=0.358)
        fig.layout.annotations[2].update(x=0.75)

        fig.layout.margin = self.layout.margin
        fig.layout.margin.b = 50

        wandb.log({'Test loss correlates': fig})

    def distance_vs_score_plot(self, rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, blind_test_targets):

        '''
        rdf distance vs score
        '''
        fig = make_subplots(rows=1, cols=2)
        full_rdf = np.concatenate([val for val in rdf_full_distance_dict.values()])
        inter_rdf = np.concatenate([val for val in rdf_inter_distance_dict.values()])
        normed_score = np.concatenate([normalize(scores_dict[key]) for key in scores_dict.keys() if key in blind_test_targets])  # if key in normed_energy_dict.keys()])

        clip = np.quantile(full_rdf, 0.99) * 2
        full_rdf = np.clip(full_rdf, a_min=0, a_max=clip)
        inter_rdf = np.clip(inter_rdf, a_min=0, a_max=clip)

        xline = np.asarray([np.amin(full_rdf), np.amax(full_rdf)])
        linreg_result = linregress(full_rdf, normed_score)
        yline = xline * linreg_result.slope + linreg_result.intercept
        fig.add_trace(go.Scattergl(x=full_rdf, y=normed_score, showlegend=False,
                                   mode='markers'),
                      row=1, col=1)

        fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Full RDF R={linreg_result.rvalue:.3f}'), row=1, col=1)

        xline = np.asarray([np.amin(inter_rdf), np.amax(inter_rdf)])
        linreg_result = linregress(inter_rdf, normed_score)
        yline = xline * linreg_result.slope + linreg_result.intercept
        fig.add_trace(go.Scattergl(x=inter_rdf, y=normed_score, showlegend=False,
                                   mode='markers'),
                      row=1, col=2)
        fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Intermolecular RDF R={linreg_result.rvalue:.3f}'), row=1, col=2)

        fig.update_layout(title='All BT Targets')
        fig.update_yaxes(title_text='Target-wise Normed Model score', row=1, col=1)
        fig.update_xaxes(title_text='Full RDF Distance', row=1, col=1)
        fig.update_xaxes(title_text='Intermolecular RDF Distance', row=1, col=2)
        wandb.log({f'Distance vs. Score': fig})

    def targetwise_distance_vs_score_plot(self, rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, blind_test_targets):
        all_scores = np.concatenate([(scores_dict[key]) for key in scores_dict.keys() if key in blind_test_targets])  # if key in normed_energy_dict.keys()])
        full_rdf = np.concatenate([val for val in rdf_full_distance_dict.values()])
        inter_rdf = np.concatenate([val for val in rdf_inter_distance_dict.values()])

        clip = np.quantile(full_rdf, 0.99) * 1.9
        # full_rdf = np.clip(full_rdf, a_min=0, a_max=clip)

        fig = make_subplots(rows=2, cols=4,
                            vertical_spacing=0.075,
                            subplot_titles=(list(rdf_full_distance_dict.keys())))  # + ['All'])

        for i, label in enumerate(rdf_full_distance_dict.keys()):
            row = i // 4 + 1
            col = i % 4 + 1
            dist = rdf_full_distance_dict[label]
            dist = np.clip(dist, a_min=0, a_max=clip)
            xline = np.asarray([np.amin(dist), np.amax(dist)])
            linreg_result = linregress(dist, scores_dict[label])
            yline = xline * linreg_result.slope + linreg_result.intercept

            fig.add_trace(go.Scattergl(x=dist, y=scores_dict[label], showlegend=False,
                                       mode='markers', marker_color='rgba(100,0,0,1)', opacity=0.05),
                          row=row, col=col)
            fig.add_trace(go.Scattergl(x=np.zeros(1), y=scores_dict[label + ' exp'], showlegend=False, mode='markers',
                                       marker=dict(color='Black', size=10, line=dict(color='White', width=2))), row=row, col=col)

            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=row, col=col)
            fig.update_xaxes(range=[-5, clip], row=row, col=col)
        #
        # xline = np.asarray([np.amin(full_rdf), np.amax(full_rdf)])
        # linreg_result = linregress(full_rdf, all_scores)
        # yline = xline * linreg_result.slope + linreg_result.intercept
        # fig.add_trace(go.Scattergl(x=full_rdf, y=all_scores, showlegend=False,
        #                            mode='markers', ),
        #               row=2, col=4)
        # fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'All Targets R={linreg_result.rvalue:.3f}'), row=2, col=4)
        # fig.update_xaxes(range=[-5, clip], row=2, col=4)

        fig.update_yaxes(title_text='Model Score', row=1, col=1)
        fig.update_yaxes(title_text='Model Score', row=2, col=1)
        fig.layout.margin = self.layout.margin

        wandb.log({f'Targetwise Distance vs. Score': fig})

    def target_ranking_analysis(self, identifiers_list, scores_dict, all_identifiers):
        '''
        within-submission score vs rankings
        file formats are different between BT 5 and BT6
        '''
        target_identifiers = {}
        rankings = {}
        group = {}
        list_num = {}
        for label in ['XXII', 'XXIII', 'XXVI']:
            target_identifiers[label] = [identifiers_list[all_identifiers[label][n]] for n in range(len(all_identifiers[label]))]
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
        return group, rankings, list_num

    def groupwise_target_ranking_analysis(self, group, rankings, list_num, scores_dict):

        for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
            names = np.unique(list(group[label]))
            uniques = len(names)
            rows = int(np.floor(np.sqrt(uniques)))
            cols = int(np.ceil(np.sqrt(uniques)) + 1)
            fig = make_subplots(rows=rows, cols=cols,
                                subplot_titles=(names), x_title='Group Ranking', y_title='Model Score', vertical_spacing=0.1)

            for j, group_name in enumerate(np.unique(group[label])):
                good_inds = np.where(np.asarray(group[label]) == group_name)[0]
                submissions_list_num = np.asarray(list_num[label])[good_inds]
                list1_inds = np.where(submissions_list_num == 1)[0]
                list2_inds = np.where(submissions_list_num == 2)[0]

                xline = np.asarray([0, max(np.asarray(rankings[label])[good_inds[list1_inds]])])
                linreg_result = linregress(np.asarray(rankings[label])[good_inds[list1_inds]], np.asarray(scores_dict[label])[good_inds[list1_inds]])
                yline = xline * linreg_result.slope + linreg_result.intercept

                fig.add_trace(go.Scattergl(x=np.asarray(rankings[label])[good_inds], y=np.asarray(scores_dict[label])[good_inds], showlegend=False,
                                           mode='markers', opacity=0.5, marker=dict(size=6, color=submissions_list_num, colorscale='portland', cmax=2, cmin=1, showscale=False)),
                              row=j // cols + 1, col=j % cols + 1)

                fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{group_name} R={linreg_result.rvalue:.3f}', line=dict(color='#0c4dae')), row=j // cols + 1, col=j % cols + 1)

                if len(list2_inds) > 0:
                    xline = np.asarray([0, max(np.asarray(rankings[label])[good_inds[list2_inds]])])
                    linreg_result2 = linregress(np.asarray(rankings[label])[good_inds[list2_inds]], np.asarray(scores_dict[label])[good_inds[list2_inds]])
                    yline2 = xline * linreg_result2.slope + linreg_result2.intercept
                    fig.add_trace(go.Scattergl(x=xline, y=yline2, name=f'{group_name} #2 R={linreg_result2.rvalue:.3f}', line=dict(color='#d60000')), row=j // cols + 1, col=j % cols + 1)

            fig.update_layout(title=label)

            fig.update_layout(width=1000, height=500)
            fig.layout.margin = self.layout.margin
            fig.layout.margin.t = 50
            fig.layout.margin.b = 55
            fig.layout.margin.l = 60
            wandb.log({f"{label} Groupwise Analysis": fig})

        # specifically interesting groups & targets
        # brandenberg XXVI
        # Brandenberg XXII
        # Facelli XXII
        # Price XXII
        # Goto XXII
        # Brandenberg XXIII

        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.075, subplot_titles=(
            ['Brandenburg XXII', 'Brandenburg XXIII', 'Brandenburg XXVI', 'Facelli XXII']),
                            x_title='Model Score')

        for ii, label in enumerate(['XXII', 'XXIII', 'XXVI']):
            good_inds = np.where(np.asarray(group[label]) == 'Brandenburg')[0]
            submissions_list_num = np.asarray(list_num[label])[good_inds]
            list1_inds = np.where(submissions_list_num == 1)[0]
            list2_inds = np.where(submissions_list_num == 2)[0]

            fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
                                       histnorm='probability density',
                                       nbinsx=50,
                                       name="Submission 1 Score",
                                       showlegend=False,
                                       marker_color='#0c4dae'),
                          row=(ii) // 2 + 1, col=(ii) % 2 + 1)
            fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
                                       histnorm='probability density',
                                       nbinsx=50,
                                       name="Submission 2 Score",
                                       showlegend=False,
                                       marker_color='#d60000'),
                          row=(ii) // 2 + 1, col=(ii) % 2 + 1)

        label = 'XXII'
        good_inds = np.where(np.asarray(group[label]) == 'Facelli')[0]
        submissions_list_num = np.asarray(list_num[label])[good_inds]
        list1_inds = np.where(submissions_list_num == 1)[0]
        list2_inds = np.where(submissions_list_num == 2)[0]

        fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
                                   histnorm='probability density',
                                   nbinsx=50,
                                   name="Submission 1 Score",
                                   showlegend=False,
                                   marker_color='#0c4dae'), row=2, col=2)
        fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
                                   histnorm='probability density',
                                   nbinsx=50,
                                   name="Submission 2 Score",
                                   showlegend=False,
                                   marker_color='#d60000'), row=2, col=2)

        fig.layout.margin = self.layout.margin
        fig.layout.margin.b = 50

        wandb.log({'Group Submissions Analysis': fig})

        #
        # label = 'XXII'
        # good_inds = np.where(np.asarray(group[label]) == 'Price')[0]
        # submissions_list_num = np.asarray(list_num[label])[good_inds]
        # list1_inds = np.where(submissions_list_num == 1)[0]
        # list2_inds = np.where(submissions_list_num == 2)[0]
        #
        # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
        #                            histnorm='probability density',
        #                            nbinsx=50,
        #                            name="Submission 1 Score",
        #                            showlegend=False,
        #                            marker_color='#0c4dae'),
        #               row=2, col=2)
        # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
        #                            histnorm='probability density',
        #                            nbinsx=50,
        #                            name="Submission 2 Score",
        #                            showlegend=False,
        #                            marker_color='#d60000'),
        #               row=2, col=2)
        #
        # label = 'XXII'
        # good_inds = np.where(np.asarray(group[label]) == 'Goto')[0]
        # submissions_list_num = np.asarray(list_num[label])[good_inds]
        # list1_inds = np.where(submissions_list_num == 1)[0]
        # list2_inds = np.where(submissions_list_num == 2)[0]
        #
        # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
        #                            histnorm='probability density',
        #                            nbinsx=50,
        #                            name="Submission 1 Score",
        #                            showlegend=False,
        #                            marker_color='#0c4dae'),
        #               row=2, col=3)
        # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
        #                            histnorm='probability density',
        #                            nbinsx=50,
        #                            name="Submission 2 Score",
        #                            showlegend=False,
        #                            marker_color='#d60000'),
        #               row=2, col=3)

        return None

    def bt_clustering(self, all_identifiers, scores_dict, extra_test_dict):
        # compute pairwise distances
        aa = 1
        if True:  # not os.path.exists('../bt_submissions_distances.npy'):  # expensive - should be precomputed
            submissions_dists_dict = {}
            for key in all_identifiers.keys():
                if True:  # key not in ['XVI', 'XVII', 'XVIII']:  # already did these ones
                    if all_identifiers[key] != []:
                        print(key)
                        rdfs = extra_test_dict['full rdf'][all_identifiers[key]]
                        dists = np.zeros((len(rdfs), len(rdfs)))
                        for i in tqdm.tqdm(range(len(rdfs))):
                            dists[i, :i + 1] = compute_rdf_distance_metric_torch(torch.Tensor(rdfs[i]).cuda(), torch.Tensor(rdfs[:i + 1]).cuda()).cpu().detach().numpy()  # many-to-one - faster via torch
                        submissions_dists_dict[key] = dists
                        submissions_dists_dict[key] += dists.T
                np.save('../bt_submissions_distances.npy', submissions_dists_dict)
        else:
            submissions_dists_dict = np.load('../bt_submissions_distances.npy', allow_pickle=True).item()

    def slash_batch(self, train_loader, test_loader):
        slash_increment = max(4, int(train_loader.batch_size * 0.1))
        train_loader = update_batch_size(train_loader, train_loader.batch_size - slash_increment)
        test_loader = update_batch_size(test_loader, test_loader.batch_size - slash_increment)
        print('==============================')
        print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
        print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
        print('==============================')
        wandb.log({'batch size': train_loader.batch_size})

        return train_loader, test_loader

    def update_batch_size(self, train_loader, test_loader, extra_test_loader):
        if train_loader.batch_size < len(train_loader.dataset):  # if the batch is smaller than the dataset
            increment = max(4, int(train_loader.batch_size * 0.05))  # increment batch size
            train_loader = update_batch_size(train_loader, train_loader.batch_size + increment)
            test_loader = update_batch_size(test_loader, test_loader.batch_size + increment)
            if self.config.extra_test_set_paths is not None:
                extra_test_loader = update_batch_size(extra_test_loader, extra_test_loader.batch_size + increment)
            print(f'Batch size incremented to {train_loader.batch_size}')
        wandb.log({'batch size': train_loader.batch_size})
        return train_loader, test_loader, extra_test_loader

    def check_model_convergence(self, metrics_dict, config, epoch):
        generator_convergence = checkConvergence(metrics_dict['generator test loss'], config.history, config.generator.convergence_eps)
        discriminator_convergence = checkConvergence(metrics_dict['discriminator test loss'], config.history, config.discriminator.convergence_eps)
        if generator_convergence:
            print('generator converged!')
        if discriminator_convergence:
            print('discriminator converged!')

        return generator_convergence, discriminator_convergence

    def model_checkpointing(self, epoch, config, discriminator, generator, d_optimizer, g_optimizer, g_err_te, d_err_te, metrics_dict):
        if epoch > 0:
            if epoch % 5 == 0:  # every 5 epochs, save a checkpoint
                # saving early-stopping checkpoint
                save_checkpoint(epoch, discriminator, d_optimizer, config.discriminator.__dict__, 'discriminator_' + str(config.run_num) + f'_epoch_{epoch}')
                save_checkpoint(epoch, generator, g_optimizer, config.generator.__dict__, 'generator_' + str(config.run_num) + f'_epoch_{epoch}')

            if np.average(d_err_te) < np.amin(metrics_dict['discriminator test loss'][:-1]):  # todo fix this
                print("Saving discriminator checkpoint")
                save_checkpoint(epoch, discriminator, d_optimizer, config.discriminator.__dict__, 'discriminator_' + str(config.run_num))
            if np.average(g_err_te) < np.amin(metrics_dict['generator test loss'][:-1]):
                print("Saving generator checkpoint")
                save_checkpoint(epoch, generator, g_optimizer, config.generator.__dict__, 'generator_' + str(config.run_num))

    def update_lr(self, config, d_schedulers, d_optimizer, d_err_tr, d_hit_max_lr,
                  g_schedulers, g_optimizer, g_err_tr, g_hit_max_lr):
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

        return d_optimizer, d_learning_rate, d_hit_max_lr, g_optimizer, g_learning_rate, g_hit_max_lr

        # agglomerative clustering and dendrogram
        # setting distance_threshold=0 ensures we compute the full tree.
        # plt.clf()
        # for i, label in enumerate(submissions_dists_dict.keys()):
        #     model = AgglomerativeClustering(distance_threshold=0, linkage="average", affinity='precomputed', n_clusters=None)
        #     model = model.fit(submissions_dists_dict[label])
        #
        #     plt.title("Hierarchical Clustering Dendrogram")
        #     plt.subplot(2, 4, i + 1)
        #     # plot the top three levels of the dendrogram
        #     plot_dendrogram(model, truncate_mode="level", p=3)
        #     plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        #     plt.show()

    def report_sampling(self, test_epoch_stats_dict, sampling_dict, sample_ind):
        '''
        score
        stun
        acceptance rate
        temperature
        overall distribution
        '''
        #if False:
        # files = [
        #     'C:/Users\mikem\Desktop\CSP_runs\sampling_1/sampling_output_run_901.npy',
        #     'C:/Users\mikem\Desktop\CSP_runs\sampling_1/sampling_output_run_902.npy',
        #     'C:/Users\mikem\Desktop\CSP_runs\sampling_1/sampling_output_run_903.npy',
        #     'C:/Users\mikem\Desktop\CSP_runs\sampling_1/sampling_output_run_904.npy',
        #     'C:/Users\mikem\Desktop\CSP_runs\sampling_1/sampling_output_run_905.npy',
        #     'C:/Users\mikem\Desktop\CSP_runs\sampling_1/sampling_output_run_906.npy']
        files = [
            'D:\sampling_2/sampling_output_run_910.npy',
            'D:\sampling_2/sampling_output_run_911.npy',
            'D:\sampling_2/sampling_output_run_912.npy',
            'D:\sampling_2/sampling_output_run_917.npy',
            'D:\sampling_2/sampling_output_run_918.npy'
        ]
        #sampling_dict = np.load(files[0],allow_pickle=True).item()
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=40,  # top margin
            )
        )
        n_samples = len(sampling_dict['scores'])
        num_iters = sampling_dict['scores'].shape[1]

        '''
        telemetry summary
        '''
        fig = make_subplots(cols=3, rows=2, subplot_titles=['Best Model Score', 'Average STUN Score', 'Min vdW Score', 'Mean Acceptance Rate', 'Mean Temperature'])
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=-np.amin(sampling_dict['scores'], axis=0)), col=1, row=1)
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.mean(sampling_dict['stun score'], axis=0)), col=2, row=1)
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=-np.amin(sampling_dict['vdw penalties'], axis=0)), col=3, row=1)
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.mean(sampling_dict['acceptance ratio'], axis=0)), col=1, row=2)
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.mean(np.log10(sampling_dict['temperature']), axis=0)), col=2, row=2)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[-1, 1], row=1, col=2)
        fig.update_yaxes(range=[-2,0],row=1,col=3)
        fig.update_yaxes(range=[-16,16],row=1,col=1)
        fig.layout.margin = layout.margin
        #fig.write_image('../paper1_figs/sampling_telemetry_summary.png')
        wandb.log({'Sampling Telemetry Summary': fig})
        if self.config.machine == 'local':
            import plotly.io as pio
            pio.renderers.default = 'browser'
            fig.show()

        crystal_identifier = test_epoch_stats_dict['identifiers'][sample_ind]

        '''
        plt.clf()
        import matplotlib as mpl
        for i,file in enumerate(files):
            plt.subplot(2,3,i+1)
            sampling_dict = np.load(file, allow_pickle=True).item()
            plt.hist2d(x=np.nan_to_num(-sampling_dict['scores'].flatten()),
                   y=np.nan_to_num(-sampling_dict['vdw penalties'].flatten()),
                   bins = 50,
                   range=[[-16,16],[-5,0.2]],
                   norm=mpl.colors.LogNorm())           
            
        '''

        fig = go.Figure()
        viridis = px.colors.sequential.Viridis
        fig.add_trace(go.Histogram2d(x=-sampling_dict['scores'].flatten(),
                                     y=-sampling_dict['vdw penalties'].flatten(),
                                     xbins=dict(start=-16, end=16, size=32 / 50),
                                     ybins=dict(start=-2, end=0.1, size=1 / 50),
                                     showscale=False,
                                     colorscale=[
                                         [0, viridis[0]],
                                         [1. / 1000000, viridis[2]],
                                         [1. / 10000, viridis[4]],
                                         [1. / 100, viridis[7]],
                                         [1., viridis[9]],
                                     ],
                                     colorbar=dict(
                                         tick0=0,
                                         tickmode='array',
                                         tickvals=[0, 1000, 10000]
                                     )))

        fig.add_trace(go.Scattergl(x=softmax_and_score(test_epoch_stats_dict['discriminator real score'][sample_ind]),
                                   y=test_epoch_stats_dict['real vdW penalty'][sample_ind][None],
                                   mode='markers',
                                   showlegend=False,
                                   marker=dict(
                                       symbol='circle',
                                       color='white',
                                       size=25,
                                       line=dict(width=1, color='black')
                                   )))

        fig.layout.margin = layout.margin
        fig.update_layout(title=crystal_identifier)
        #fig.write_image('../paper1_figs/sampling_scores.png')
        wandb.log({'Sampling Scores': fig})
        if self.config.machine == 'local':
            import plotly.io as pio
            pio.renderers.default = 'browser'
            fig.show()

        aa = 1


        '''
        full telemetry
        '''
        # fig = make_subplots(cols=3, rows=2, subplot_titles=['Model Score', 'STUN Score','vdW Score', 'Acceptance Rate', 'Temperature'])
        # for i in range(n_samples):
        #     opacity = np.clip(1 - np.abs(np.amin(sampling_dict['scores'][i]) - np.amin(sampling_dict['scores'])) / np.amin(sampling_dict['scores']), a_min = 0.1, a_max = 1)
        #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=-sampling_dict['scores'][i], opacity = opacity), col=1, row=1)
        #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=sampling_dict['stun score'][i], opacity = opacity), col=2, row=1)
        #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=-sampling_dict['vdw penalties'][i], opacity = opacity), col=3, row=1)
        #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=sampling_dict['acceptance ratio'][i], opacity=opacity), col=1, row=2)
        #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.log10(sampling_dict['temperature'][i]), opacity= opacity), col=2, row=2)
        # fig.update_layout(showlegend=False)
        # fig.update_yaxes(range=[0, 1], row=1, col=2)
        # fig.layout.margin = layout.margin
        # fig.write_image('../paper1_figs/sampling_telemetry.png')
        # wandb.log({'Sampling Telemetry': fig})
        # if self.config.machine == 'local':
        #     import plotly.io as pio
        #     pio.renderers.default = 'browser'
        #     fig.show()
        #
        return None


''' extra stuff
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


         
        # '''
# CSP pipeline analysis
# '''
# sapt_inds = np.asarray([i for i, ind in enumerate(extra_test_dict['identifiers']) if ('\sapt' in ind) and ('-220224' not in ind)])
# sapt_22_inds = np.asarray([i for i, ind in enumerate(extra_test_dict['identifiers']) if '\sapt-220224' in ind])
#
# if len(sapt_inds > 0):
#
#     sapt_energies = extra_test_dict['atomistic energy'][sapt_inds]
#     sapt_22_energies = extra_test_dict['atomistic energy'][sapt_22_inds]
#
#     outlier_inds = np.concatenate((np.where(sapt_energies > 0)[0], np.where(sapt_energies < -150)[0]))
#     sapt_inds = np.asarray([ind for i, ind in enumerate(sapt_inds) if not (i in outlier_inds)])
#     sapt_energies = extra_test_dict['atomistic energy'][sapt_inds]
#
#     outlier_inds = np.concatenate((np.where(sapt_22_energies > 0)[0], np.where(sapt_22_energies < -150)[0]))
#     sapt_22_inds = np.asarray([ind for i, ind in enumerate(sapt_22_inds) if not (i in outlier_inds)])
#     sapt_22_energies = extra_test_dict['atomistic energy'][sapt_22_inds]
#
#     if config.gan_loss == 'standard':
#         sapt_scores = -np.log10(1 - np_softmax(extra_test_dict['scores'][sapt_inds], temperature=softmax_temperature)[:, 1])
#         sapt_22_scores = -np.log10(1 - np_softmax(extra_test_dict['scores'][sapt_22_inds], temperature=softmax_temperature)[:, 1])
#     elif config.gan_loss == 'wasserstein':
#         sapt_scores = extra_test_dict['scores'][sapt_inds]
#         sapt_22_scores = extra_test_dict['scores'][sapt_22_inds]
#
#     '''
#     plot overall energy-score correlation
#     '''
#     fig = make_subplots(rows=1, cols=2,
#                         subplot_titles=(['Sapt', 'Sapt 22']))
#
#     xline = np.asarray([min(sapt_energies), max(sapt_energies)])
#     linreg_result = linregress(sapt_energies, sapt_scores)
#     yline = xline * linreg_result.slope + linreg_result.intercept
#
#     fig.add_trace(go.Scattergl(x=sapt_energies, y=sapt_scores, showlegend=False,
#                                mode='markers'),  # , marker=dict(size=6, color=list_num[label], colorscale='portland', showscale=False)),
#                   row=1, col=1)
#
#     fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'sapt R={linreg_result.rvalue:.3f}'), row=1, col=1)
#
#     xline = np.asarray([min(sapt_22_energies), max(sapt_22_energies)])
#     linreg_result = linregress(sapt_22_energies, sapt_22_scores)
#     yline = xline * linreg_result.slope + linreg_result.intercept
#
#     fig.add_trace(go.Scattergl(x=sapt_22_energies, y=sapt_22_scores, showlegend=False,
#                                mode='markers'),  # , marker=dict(size=6, color=list_num[label], colorscale='portland', showscale=False)),
#                   row=1, col=2)
#
#     fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'sapt-22 R={linreg_result.rvalue:.3f}'), row=1, col=2)
#     wandb.log({'Target 31 analysis': fig})
#
#     '''
#     plot within-conformer energy-score correlation
#     '''
#     conformers = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10']
#     conformers_inds_dict = {}
#     conformers_inds_dict_22 = {}
#     conformers_energies_dict = {}
#     conformers_energies_dict_22 = {}
#     conformers_scores_dict = {}
#     conformers_scores_dict_22 = {}
#     for conformer in conformers:
#         conformers_inds_dict[conformer] = np.asarray([i for i, ind in enumerate(extra_test_dict['identifiers']) if ('\sapt' in ind) and ('-220224' not in ind) and (conformer in ind)])
#         conformers_inds_dict_22[conformer] = np.asarray([i for i, ind in enumerate(extra_test_dict['identifiers']) if ('\sapt-220224' in ind) and (conformer in ind)])
#         conformers_energies_dict[conformer] = extra_test_dict['atomistic energy'][conformers_inds_dict[conformer]]
#
#         outlier_inds = np.concatenate((np.where(conformers_energies_dict[conformer] > 0)[0], np.where(conformers_energies_dict[conformer] < -150)[0]))
#         conformers_inds_dict[conformer] = np.asarray([ind for i, ind in enumerate(conformers_inds_dict[conformer]) if not (i in outlier_inds)])
#         conformers_energies_dict[conformer] = extra_test_dict['atomistic energy'][conformers_inds_dict[conformer]]
#
#         conformers_energies_dict_22[conformer] = extra_test_dict['atomistic energy'][conformers_inds_dict_22[conformer]]
#
#         outlier_inds = np.concatenate((np.where(conformers_energies_dict_22[conformer] > 0)[0], np.where(conformers_energies_dict_22[conformer] < -150)[0]))
#         conformers_inds_dict_22[conformer] = np.asarray([ind for i, ind in enumerate(conformers_inds_dict_22[conformer]) if not (i in outlier_inds)])
#         conformers_energies_dict_22[conformer] = extra_test_dict['atomistic energy'][conformers_inds_dict_22[conformer]]
#
#         conformers_scores_dict[conformer] = -np.log10(1 - np_softmax(extra_test_dict['scores'][conformers_inds_dict[conformer]], temperature=softmax_temperature)[:, 1])
#         conformers_scores_dict_22[conformer] = -np.log10(1 - np_softmax(extra_test_dict['scores'][conformers_inds_dict_22[conformer]], temperature=softmax_temperature)[:, 1])
#
#     fig = make_subplots(rows=2, cols=10,
#                         )  # subplot_titles=(['Sapt','Sapt 22']))
#
#     for ii, conformer in enumerate(conformers_inds_dict.keys()):
#         score = conformers_scores_dict[conformer]
#         score_22 = conformers_scores_dict_22[conformer]
#         energy = conformers_energies_dict[conformer]
#         energy_22 = conformers_energies_dict_22[conformer]
#
#         xline = np.asarray([min(energy), max(energy)])
#         linreg_result = linregress(energy, score)
#         yline = xline * linreg_result.slope + linreg_result.intercept
#
#         fig.add_trace(go.Scattergl(x=energy, y=score, showlegend=False,
#                                    mode='markers'),  # , marker=dict(size=6, color=list_num[label], colorscale='portland', showscale=False)),
#                       row=1, col=ii + 1)
#
#         fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'sapt R={linreg_result.rvalue:.3f}'), row=1, col=ii + 1)
#
#         xline = np.asarray([min(energy_22), max(energy_22)])
#         linreg_result = linregress(energy_22, score_22)
#         yline = xline * linreg_result.slope + linreg_result.intercept
#
#         fig.add_trace(go.Scattergl(x=energy_22, y=score_22, showlegend=False,
#                                    mode='markers'),  # , marker=dict(size=6, color=list_num[label], colorscale='portland', showscale=False)),
#                       row=2, col=ii + 1)
#
#         fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'sapt-22 R={linreg_result.rvalue:.3f}'), row=2, col=ii + 1)
#
#     wandb.log({'Conformer-wise analysis': fig})
#
#     target_31_inds = np.asarray([i for i, ind in enumerate(extra_test_dict['identifiers']) if ('219967' in ind)])

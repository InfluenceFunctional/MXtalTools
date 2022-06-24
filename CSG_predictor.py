import wandb
from utils import *
import glob
from model_utils import *
from dataset_management.CSD_data_manager import Miner
from torch import backends, optim
import torch
from dataset_utils import BuildDataset, get_dataloaders
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from nikos.coordinate_transformations import coor_trans, cell_vol, coor_trans_matrix
from pyxtal import symmetry
from ase.visualize import view
from ase import Atoms
import rdkit.Chem as Chem
from dataset_management.random_crystal_builder import *
from models.generator_models import crystal_generator
from models.discriminator_models import crystal_discriminator


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

        if 'cell' in self.config.mode:
            print('Pre-generating spacegroup symmetries')
            self.sym_ops = {}
            self.point_groups = {}
            self.lattice_type = {}
            for i in tqdm.tqdm(range(1, 231)):
                sym_group = symmetry.Group(i)
                general_position_syms = sym_group.wyckoffs_organized[0][0]
                self.sym_ops[i] = [general_position_syms[i].affine_matrix for i in range(len(general_position_syms))]  # first 0 index is for general position, second index is superfluous, third index is the symmetry operation
                self.point_groups[i] = sym_group.point_group
                self.lattice_type[i] = sym_group.lattice_type

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
        if 'gan' in config.mode:
            metrics_list = ['discriminator train loss', 'discriminator test loss', 'epoch', 'discriminator learning rate',
                            'generator train loss', 'generator test loss', 'generator learning rate']
            metrics_dict = initialize_metrics_dict(metrics_list)
        else:
            metrics_list = ['train loss', 'test loss', 'epoch', 'learning rate']
            metrics_dict = initialize_metrics_dict(metrics_list)

        return metrics_dict

    def update_metrics(self, epoch, metrics_dict, err_tr, err_te, lr):
        metrics_dict['train loss'].append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
        metrics_dict['test loss'].append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
        metrics_dict['epoch'].append(epoch)
        metrics_dict['learning rate'].append(lr)

        return metrics_dict

    def update_gan_metrics(self, epoch, metrics_dict, d_err_tr, d_err_te, g_err_tr, g_err_te, d_lr, g_lr):
        metrics_dict['epoch'].append(epoch)
        metrics_dict['discriminator train loss'].append(torch.mean(torch.stack(d_err_tr)).cpu().detach().numpy())
        metrics_dict['discriminator test loss'].append(torch.mean(torch.stack(d_err_te)).cpu().detach().numpy())
        metrics_dict['discriminator learning rate'].append(d_lr)
        metrics_dict['generator train loss'].append(torch.mean(torch.stack(g_err_tr)).cpu().detach().numpy())
        metrics_dict['generator test loss'].append(torch.mean(torch.stack(g_err_te)).cpu().detach().numpy())
        metrics_dict['generator learning rate'].append(g_lr)

        return metrics_dict

    def init_model(self, config, dataDims, print_status=True):
        '''
        Initialize model and optimizer
        :return:
        '''
        # init model
        print("Initializing model for " + config.mode)
        if config.mode == 'joint modelling':
            model = FlowModel(config, dataDims)
        elif 'molecule' in config.mode:
            model = molecule_graph_model(config, dataDims, crystal_mode=False)
        elif 'cell' in config.mode:
            model = molecule_graph_model(config, dataDims, crystal_mode=True)
        else:
            print(config.mode + ' is not a valid model mode!')
            sys.exit()

        if config.device == 'cuda':
            model = model.cuda()

        # init optimizers
        amsgrad = False
        beta1 = config.beta1  # 0.9
        beta2 = config.beta2  # 0.999
        weight_decay = config.weight_decay  # 0.01
        momentum = 0

        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), amsgrad=amsgrad, lr=config.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif config.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), amsgrad=amsgrad, lr=config.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif config.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            print(config.optimizer + ' is not a valid optimizer')
            sys.exit()

        # init schedulers
        scheduler1 = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=15,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=15
        )
        lr_lambda = lambda epoch: 1.25
        scheduler3 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
        lr_lambda2 = lambda epoch: 0.95
        scheduler4 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda2)

        nconfig = get_n_config(model)
        if print_status:
            print('Proxy model has {:.3f} million or {} parameters'.format(nconfig / 1e6, int(nconfig)))

        return model, optimizer, [scheduler1, scheduler3, scheduler4], nconfig

    def init_gan(self, config, dataDims, print_status=True):
        '''
        Initialize model and optimizer
        :return:
        '''
        # init model
        print("Initializing models for " + config.mode)
        generator = crystal_generator(config, dataDims)
        discriminator = crystal_discriminator(config, dataDims)

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
            print('Discriminator model has {:.3f} million or {} parameters'.format(params1 / 1e6, int(params2)))

        return generator, discriminator, g_optimizer, g_scheduler, d_optimizer, d_scheduler, params1, params2

    def get_batch_size(self, dataset, config):
        finished = 0
        batch_size = config.initial_batch_size.real
        batch_reduction_factor = config.auto_batch_reduction

        if 'gan'.lower() in config.mode:
            generator, discriminator, g_optimizer, \
            g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)

            while finished == 0:
                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU cache
                    generator.cuda()
                    discriminator.cuda()

                if config.add_spherical_basis is False:  # initializing spherical basis is too expensive to do repetitively
                    generator, discriminator, g_optimizer, \
                    g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)

                try:
                    train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=batch_size)
                    d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                        self.gan_epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                       g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                       update_gradients=True, record_stats=True, iteration_override=2)  # train & compute test loss

                    finished = 1

                    if batch_size < 10:
                        leeway = batch_reduction_factor / 2
                    elif batch_size > 20:
                        leeway = batch_reduction_factor
                    else:
                        leeway = batch_reduction_factor / 1.33

                    batch_size = max(1, int(batch_size * leeway))  # give a margin for molecule sizes - larger margin for smaller batch sizes

                    print('Final batch size is {}'.format(batch_size))

                    tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)

                    if config.device.lower() == 'cuda':
                        torch.cuda.empty_cache()  # clear GPU cache

                    return tr, te, batch_size
                except:  # MemoryError or RuntimeError:
                    batch_size = int(batch_size * 0.95)
                    print('Training batch size reduced to {}'.format(batch_size))
                    if batch_size <= 2:
                        print('Model is too big! (or otherwise broken)')
                        if config.device.lower() == 'cuda':
                            torch.cuda.empty_cache()  # clear GPU cache

                        # for debugging purposes
                        train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=batch_size)
                        d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                            self.gan_epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                           g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                           update_gradients=True, record_stats=True, iteration_override=2)  # train & compute test loss

                        sys.exit()

        else:

            model, optimizer, schedulers, n_params = self.init_model(config, config.dataDims, print_status=False)

            while finished == 0:
                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU cache

                if config.add_spherical_basis is False:  # initializing spherical basis is too expensive to do repetitively
                    model, optimizer, schedulers, n_params = self.init_model(config, config.dataDims, print_status=False)  # for some reason necessary for memory reasons

                try:
                    tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)
                    self.model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss

                    finished = 1

                    if batch_size < 10:
                        leeway = batch_reduction_factor / 2
                    elif batch_size > 20:
                        leeway = batch_reduction_factor
                    else:
                        leeway = batch_reduction_factor / 1.33

                    batch_size = max(1, int(batch_size * leeway))  # give a margin for molecule sizes - larger margin for smaller batch sizes

                    print('Final batch size is {}'.format(batch_size))

                    tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)

                    if config.device.lower() == 'cuda':
                        torch.cuda.empty_cache()  # clear GPU cache

                    return tr, te, batch_size
                except:  # MemoryError or RuntimeError:
                    batch_size = int(batch_size * 0.95)
                    print('Training batch size reduced to {}'.format(batch_size))
                    if batch_size <= 2:
                        print('Model is too big! (or otherwise broken)')
                        if config.device.lower() == 'cuda':
                            torch.cuda.empty_cache()  # clear GPU cache

                        # for debugging purposes
                        tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)
                        self.model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss

                        sys.exit()

    def train(self):
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=self.config.wandb.experiment_tag):
            # config = wandb.config # todo: wandb configs don't support nested namespaces. Sweeps are officially broken
            # print(config)
            '''
            #hypothetical fix
            from omegaconf import DictConfig
            import argparse
            
            def parse_unknown_cmd(config: DictConfig) -> DictConfig:
                #parses unknown command line arguments to overwrite the config file
                #:param config:
                #:return:
                parser = argparse.ArgumentParser()
                args, unknown = parser.parse_known_args()
            
                for flag in unknown:
                    arg = flag.find('--')
                    set = flag.find('=')
            
                    if (arg >= 0) and (set >= 0):
                        arg = flag[arg + 2:set]
                        val = flag[set + 1:]
            
            
                        if val.isdigit():
                            val = int(val)
                        elif val.replace('.', '', 1).isdigit():
                            val = float(val)
                        elif val.lower() in ('true', 'false'):
                            val = bool(val)
                        else:
                            val = f'"{val}"'
            
                        print(f'Executing ... config.{arg}={val}')
                        exec(f'config.{arg}={val}')
                    else:
                        print(f'Not executing {flag}')
            
                return config
            '''
            config = self.config  # go with hand-built config

            # dataset
            dataset_builder = BuildDataset(config, self.point_groups, premade_dataset=self.prep_dataset)
            del self.prep_dataset  # we don't actually want this huge thing floating around

            config.dataDims = dataset_builder.get_dimension()
            self.dataDims = dataset_builder.get_dimension()
            if 'classification' in config.mode:  # for convenience
                self.class_labels = self.dataDims['class labels']
                self.class_weights = self.dataDims['class weights']
            if config.mode == 'joint modelling':
                self.lattice_features = dataset_builder.lattice_keys
                self.n_crystal_dims = self.dataDims['n crystal features']
                if config.generator.conditional_modelling:
                    self.n_conditional_features = self.dataDims['n conditional features']
                else:
                    self.n_conditional_features = 0
            if 'cell' in config.mode:  # get relevant indices
                self.cell_angle_keys = ['crystal alpha', 'crystal beta', 'crystal gamma']
                self.cell_angle_inds = [self.dataDims['tracking features dict'].index(key) for key in self.cell_angle_keys]
                self.cell_length_keys = ['crystal cell a', 'crystal cell b', 'crystal cell c']
                self.cell_length_inds = [self.dataDims['tracking features dict'].index(key) for key in self.cell_length_keys]
                self.z_value_ind = self.dataDims['tracking features dict'].index('crystal z value')
                self.sg_number_ind = self.dataDims['tracking features dict'].index('crystal spacegroup number')
                self.mol_volume_ind = self.dataDims['tracking features dict'].index('molecule volume')
                self.crystal_packing_ind = self.dataDims['tracking features dict'].index('crystal packing coefficient')
                self.crystal_density_ind = self.dataDims['tracking features dict'].index('crystal calculated density')

            # get batch size
            if config.auto_batch_sizing:
                print('Finding optimal batch size')
                train_loader, test_loader, config.final_batch_size = self.get_batch_size(dataset_builder, config)
            else:
                print('Getting dataloaders for pre-determined batch size')
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.initial_batch_size

            print("Training batch size set to {}".format(config.final_batch_size))
            # model, optimizer, schedulers
            print('Reinitializing model and optimizer')
            if 'gan'.lower() in config.mode:
                generator, discriminator, g_optimizer, \
                g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)
                n_params = params1 + params2
            else:
                model, optimizer, schedulers, n_params = self.init_model(config, self.dataDims)

            # cuda
            if config.device.lower() == 'cuda':
                print('Putting model on CUDA')
                torch.backends.cudnn.benchmark = True
                # model = torch.nn.DataParallel(model) # send to multiple GPUs - not always working with wandb
                if 'gan'.lower() in config.mode:
                    generator.cuda()
                    discriminator.cuda()
                else:
                    model.cuda()

            if 'gan'.lower() in config.mode:
                wandb.watch(generator, log_graph=True)
                wandb.watch(discriminator, log_graph=True)
            else:
                wandb.watch(model, log_graph=True)

            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_metrics(config=config)

            # training loop
            d_hit_max_lr, g_hit_max_lr, converged, epoch = False, False, False, 0
            # if config.anomaly_detection:
            #     torch.autograd.set_detect_anomaly = True
            with torch.autograd.set_detect_anomaly(True):
                while (epoch < config.max_epochs) and not converged:
                    # very cool
                    print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                    print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                    print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                    # very cool
                    print("Starting Epoch {}".format(epoch))  # index from 0, very cool

                    if 'gan'.lower() in config.mode:
                        '''
                        train
                        '''
                        d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                            self.gan_epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                           g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                           update_gradients=True, record_stats=True)  # train & compute test loss

                        d_err_te, d_te_record, g_err_te, g_te_record, test_epoch_stats_dict, time_test = \
                            self.gan_epoch(config, dataLoader=test_loader, generator=generator, discriminator=discriminator,
                                           update_gradients=False, record_stats=True)  # compute loss on test set

                        print('epoch={}; d_nll_tr={:.5f}; d_nll_te={:.5f}; g_nll_tr={:.5f}; g_nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
                            epoch, torch.mean(torch.stack(d_err_tr)), torch.mean(torch.stack(d_err_te)),
                            torch.mean(torch.stack(g_err_tr)), torch.mean(torch.stack(g_err_te)),
                            time_train, time_test))

                        '''
                        update LR
                        '''
                        # update learning rate
                        g_optimizer = set_lr(g_schedulers, g_optimizer, config.generator.lr_schedule,
                                             config.generator.learning_rate, config.generator.max_lr,g_err_tr, g_hit_max_lr)
                        g_learning_rate = g_optimizer.param_groups[0]['lr']
                        if g_learning_rate >= config.generator.max_lr: g_hit_max_lr = True

                        # update learning rate
                        d_optimizer = set_lr(d_schedulers, d_optimizer, config.discriminator.lr_schedule,
                                             config.discriminator.learning_rate, config.discriminator.max_lr,d_err_tr, d_hit_max_lr)
                        d_learning_rate = d_optimizer.param_groups[0]['lr']
                        if d_learning_rate >= config.discriminator.max_lr: d_hit_max_lr = True

                        '''
                        logging
                        '''
                        # logging
                        metrics_dict = self.update_gan_metrics(
                            epoch, metrics_dict, d_err_tr, d_err_te,
                            g_err_tr, g_err_te, d_learning_rate, d_learning_rate)

                        self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict, d_tr_record, d_te_record, g_tr_record, g_te_record)
                        if epoch % config.wandb.sample_reporting_frequency == 0:
                            self.log_gan_accuracy(epoch, dataset_builder, train_loader, test_loader,
                                                  metrics_dict, g_tr_record, g_te_record, d_tr_record, d_te_record,
                                                  train_epoch_stats_dict, test_epoch_stats_dict, config,
                                                  generator, discriminator, wandb_log_figures=config.wandb.log_figures)

                        '''
                        convergence checks
                        '''

                        # check for convergence
                        if checkConvergence(metrics_dict['generator test loss'], config.history, config.generator.convergence_eps) and (epoch > config.history + 2):
                            config.finished = True
                            # self.log_gan_accuracy(epoch, dataset_builder, train_loader, test_loader,
                            #                   te_record, epoch_stats_dict,
                            #                   config, model, wandb_log_figures=True)  # always log figures at end of run
                            break

                    else:  # todo officially deprecate the old way of doing this #flow model will likely no longer function on its own here
                        err_tr, tr_record, time_train = \
                            self.model_epoch(config, dataLoader=train_loader, model=model,
                                             optimizer=optimizer, update_gradients=True)  # train & compute test loss

                        err_te, te_record, epoch_stats_dict, time_test = \
                            self.model_epoch(config, dataLoader=test_loader, model=model,
                                             update_gradients=False, record_stats=True)  # compute loss on test set

                        print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_train, time_test))

                        # update learning rate
                        optimizer = set_lr(schedulers, optimizer, config, err_tr, hit_max_lr)
                        learning_rate = optimizer.param_groups[0]['lr']
                        if learning_rate >= config.max_lr: hit_max_lr = True

                        # logging
                        self.update_metrics(epoch, metrics_dict, err_tr, err_te, learning_rate)
                        self.log_loss(metrics_dict, tr_record, te_record)
                        if epoch % config.wandb.sample_reporting_frequency == 0:
                            self.log_accuracy(epoch, dataset_builder, train_loader, test_loader,
                                              te_record, epoch_stats_dict,
                                              config, model, wandb_log_figures=config.wandb.log_figures)

                        # check for convergence
                        if checkConvergence(config, metrics_dict['test loss']) and (epoch > config.history + 2):
                            config.finished = True
                            self.log_accuracy(epoch, dataset_builder, train_loader, test_loader,
                                              te_record, epoch_stats_dict,
                                              config, model, wandb_log_figures=True)  # always log figures at end of run
                            break

                    epoch += 1

                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU

    def gan_epoch(self, config, dataLoader=None, generator=None, discriminator=None, g_optimizer=None, d_optimizer=None, update_gradients=True,
                  iteration_override=None, record_stats=False):
        t0 = time.time()
        if update_gradients:
            generator.train(True)
            discriminator.train(True)
        else:
            generator.eval()
            discriminator.eval()

        d_err = []
        d_loss_record = []
        d_real_losses = []
        d_fake_losses = []
        g_err = []
        g_loss_record = []
        g_flow_err = []
        g_flow_loss_record = []
        g_aux_losses = []
        g_adv_losses = []
        g_aux_pred = []
        g_aux_true = []
        epoch_stats_dict = {
            'tracking features': [],
        }
        for i, data in enumerate(dataLoader):
            '''
            train discriminator
            '''
            if config.train_discriminator_adversarially:
                score_on_real, score_on_fake = self.train_discriminator(generator, discriminator, config, data, i)  # alternately trains on real and fake samples

                if config.gan_loss == 'wasserstein':
                    d_losses = -score_on_real + score_on_fake  # maximize score on real, minimize score on fake
                    d_real_losses.append(-score_on_real.cpu().detach().numpy())

                elif config.gan_loss == 'standard':
                    d_losses = (1 - score_on_real) + score_on_fake  # maximize probability on real(normed on 0-1) and minimize score on fake
                    d_real_losses.append(1 - score_on_real.cpu().detach().numpy())

                else:
                    print(config.gan_loss + ' is not an implemented GAN loss function!')
                    sys.exit()

                d_fake_losses.append(score_on_fake.cpu().detach().numpy())

                d_loss = d_losses.mean()
                d_err.append(d_loss.data.cpu())  # average overall loss
                d_loss_record.extend(d_losses.cpu().detach().numpy())  # overall loss distribution

                if update_gradients:
                    d_optimizer.zero_grad()  # reset gradients from previous passes
                    d_loss.backward()  # back-propagation
                    d_optimizer.step()  # update parameters
            else:
                d_err.append(torch.zeros(1))
                d_loss_record.extend(torch.zeros(data.num_graphs))

            '''
            train_generator
            '''
            if any((config.train_generator_density, config.train_generator_adversarially)):

                adversarial_score, raw_sample, auxiliary_loss, auxiliary_prediction, auxiliary_target = \
                    self.train_generator(generator, discriminator, config, data, i)

                if adversarial_score is not None:
                    if config.gan_loss == 'wasserstein':
                        adversarial_loss = -adversarial_score  # generator wants to maximize the score
                    elif config.gan_loss == 'standard':
                        adversarial_loss = 1 - adversarial_score
                    else:
                        print(config.gan_loss + ' is not an implemented GAN loss function!')
                        sys.exit()

                if config.train_generator_density and config.train_generator_adversarially:
                    g_losses = adversarial_loss + auxiliary_loss.float()
                    g_adv_losses.append(adversarial_loss.cpu().detach().numpy())
                    g_aux_losses.append(auxiliary_loss.cpu().detach().numpy())
                    g_aux_pred.append(auxiliary_prediction)
                    g_aux_true.append(auxiliary_target)

                elif config.train_generator_density and not config.train_generator_adversarially:
                    g_losses = auxiliary_loss.float()
                    g_aux_losses.append(auxiliary_loss.cpu().detach().numpy())
                    g_aux_losses.append(auxiliary_loss.cpu().detach().numpy())
                    g_aux_pred.append(auxiliary_prediction)
                    g_aux_true.append(auxiliary_target)

                elif config.train_generator_adversarially and not config.train_generator_density:
                    g_losses = adversarial_loss
                    g_adv_losses.append(adversarial_loss.cpu().detach().numpy())

                g_loss = g_losses.mean()
                g_err.append(g_loss.data.cpu())  # average loss
                g_loss_record.extend(g_losses.cpu().detach().numpy())  # loss distribution

                if update_gradients:
                    g_optimizer.zero_grad()  # reset gradients from previous passes
                    g_loss.backward()  # back-propagation
                    g_optimizer.step()  # update parameters
            else:
                g_err.append(torch.zeros(1))
                g_loss_record.extend(torch.zeros(data.num_graphs))

            # flow loss # totally separate thing
            if config.train_generator_as_flow and ('flow' in config.generator.model_type):
                g_flow_losses = self.flow_iter(generator, data)

                g_flow_loss = g_flow_losses.mean()
                g_flow_err.append(g_flow_loss.data.cpu())  # average loss
                g_flow_loss_record.extend(g_flow_losses.cpu().detach().numpy())  # loss distribution

                if update_gradients:
                    g_optimizer.zero_grad()  # reset gradients from previous passes
                    g_flow_loss.backward()  # back-propagation
                    g_optimizer.step()  # update parameters

            if record_stats:
                epoch_stats_dict['tracking features'].extend(data.y[2])

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0

        if record_stats:
            epoch_stats_dict['discriminator on real loss'] = np.concatenate(d_real_losses) if d_real_losses != [] else None
            epoch_stats_dict['discriminator on fake loss'] = np.concatenate(d_fake_losses) if d_fake_losses != [] else None
            epoch_stats_dict['generator auxiliary loss'] = np.concatenate(g_aux_losses) if g_aux_losses != [] else None
            epoch_stats_dict['generator adversarial loss'] = np.concatenate(g_adv_losses) if g_adv_losses != [] else None
            epoch_stats_dict['generator flow loss'] = np.concatenate(g_flow_loss_record) if g_flow_loss_record != [] else None
            epoch_stats_dict['generator auxiliary prediction'] =np.concatenate(g_aux_pred) if g_aux_pred != [] else None
            epoch_stats_dict['generator auxiliary target'] = np.concatenate(g_aux_true) if g_aux_true != [] else None

            return d_err, d_loss_record, g_err, g_loss_record, epoch_stats_dict, total_time
        else:
            return d_err, d_loss_record, g_err, g_loss_record, total_time


    def adversarial_loss(self, discriminator, data, config):
        output = discriminator(data)  # reshape output from flat filters to channels * filters per channel

        # discriminator score
        if config.gan_loss == 'wasserstein':
            scores = F.softplus(output[:, 0])  # critic score - higher meaning better

        elif config.gan_loss == 'standard':
            scores = F.softmax(output, dim=1)[:, -1]  # probability of 'yes'

        else:
            print(config.gan_loss + ' is not a valid GAN loss function!')
            sys.exit()

        return scores, output

    def pairwise_correlations_analysis(self, dataset_builder, config):
        '''
        check pairwise correlations in the data
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
                data = np.asarray([(data[i].y[0]).detach().numpy() for i in range(len(data))])[:, 0, :]

        df = pd.DataFrame(data, columns=keys)
        correlations = df.corr()

        return correlations, keys

    def check_inversion_quality(self, model, test_loader, config):
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

    def get_generation_conditions(self, train_loader, test_loader, model, config):
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

    def get_nf_scores(self, sample_dict, model, config, dataloader, n_repeats, dataset_length):
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
        wandb.log({"Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(d_te_record, bins=256, range=(np.amin(d_te_record), np.quantile(d_te_record, 0.9)))
        wandb.log({"Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        wandb.log({"Train Loss Coeff. of Variation": np.sqrt(np.var(d_tr_record)) / np.average(d_tr_record)})
        wandb.log({"Test Loss Coeff. of Variation": np.sqrt(np.var(d_te_record)) / np.average(d_te_record)})

        # log generator losses
        hist = np.histogram(g_tr_record, bins=256, range=(np.amin(g_tr_record), np.quantile(g_tr_record, 0.9)))
        wandb.log({"Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(g_te_record, bins=256, range=(np.amin(g_te_record), np.quantile(g_te_record, 0.9)))
        wandb.log({"Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        wandb.log({"Train Loss Coeff. of Variation": np.sqrt(np.var(g_tr_record)) / np.average(g_tr_record)})
        wandb.log({"Test Loss Coeff. of Variation": np.sqrt(np.var(g_te_record)) / np.average(g_te_record)})

        # log specific losses
        special_losses = {}
        special_losses['epoch'] = current_metrics['epoch']
        for key in train_epoch_stats_dict.keys():
            if ('loss' in key) and (train_epoch_stats_dict[key] is not None):
                special_losses['Train ' + key] = np.average(train_epoch_stats_dict[key])
            if ('loss' in key) and (test_epoch_stats_dict[key] is not None):
                special_losses['Test ' + key] = np.average(test_epoch_stats_dict[key])
        wandb.log(special_losses)

    def log_loss(self, metrics_dict, tr_record, te_record):
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
        hist = np.histogram(tr_record, bins=256, range=(np.amin(tr_record), np.quantile(tr_record, 0.9)))
        wandb.log({"Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(te_record, bins=256, range=(np.amin(te_record), np.quantile(te_record, 0.9)))
        wandb.log({"Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        wandb.log({"Train Loss Coeff. of Variation": np.sqrt(np.var(tr_record)) / np.average(tr_record)})
        wandb.log({"Test Loss Coeff. of Variation": np.sqrt(np.var(te_record)) / np.average(te_record)})


    def generated_supercells(self, data, config, generator):
        '''
              test code for on-the-fly cell generation
              data = self.build_supercells(data)
              0. extract molecule and cell parameters
              1. find centroid
              2. find principal axis & angular component
              3. place centroid & align axes
              4. apply point symmetry
              5. tile supercell
              '''
        sg_numbers = [int(data.y[2][i][self.sg_number_ind]) for i in range(data.num_graphs)]
        lattices = [self.lattice_type[number] for number in sg_numbers]

        cell_sample = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device)).cpu()
        cell_lengths, cell_angles, rand_position, rand_rotation = cell_sample.split(3, 1)
        cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
            cell_lengths, cell_angles, rand_position, rand_rotation, lattices, config.dataDims)

        for i in range(data.num_graphs):
            # 0 extract molecule and cell parameters
            atoms = np.asarray(data.x[data.batch == i].cpu().detach())
            atomic_numbers = np.asarray(atoms[:, 0])
            heavy_inds = np.where(atomic_numbers != 1)
            atoms = atoms[heavy_inds]

            # get symmetry info
            sym_ops = np.asarray(self.sym_ops[sg_numbers[i]])  # symmetry operations between the general positions for this space group
            z_value = len(sym_ops)  # number of molecules in the reference cell

            # prep conformer
            coords = np.asarray(data.pos[data.batch == i].cpu().detach())
            weights = np.asarray([self.atom_weights[int(number)] for number in atomic_numbers])
            coords = coords[heavy_inds]
            weights = weights[heavy_inds]

            # # option to get the cell itself from the CSD
            # cell_lengths = data.y[2][i][self.cell_length_inds]  # pull cell params from tracking inds
            # cell_angles = data.y[2][i][self.cell_angle_inds]

            T_fc = coor_trans_matrix('f_to_c', cell_lengths[i].detach().numpy(), cell_angles[i].detach().numpy())
            T_cf = coor_trans_matrix('c_to_f', cell_lengths[i].detach().numpy(), cell_angles[i].detach().numpy())
            cell_vectors = T_fc.dot(np.eye(3)).T

            random_coords = randomize_molecule_position_and_orientation(
                coords.astype(float), weights.astype(float), T_fc.astype(float), T_cf.astype(float),
                np.asarray(self.sym_ops[sg_numbers[i]], dtype=float), set_position=rand_position[i].detach().numpy(), set_rotation=rand_rotation[i].detach().numpy())

            reference_cell, ref_cell_f = build_random_crystal(T_cf, T_fc, random_coords, np.asarray(self.sym_ops[sg_numbers[i]], dtype=float), z_value)

            supercell_atoms, supercell_coords = ref_to_supercell(reference_cell, z_value, atoms, cell_vectors)

            supercell_batch = torch.ones(len(supercell_atoms)).int() * i

            # append supercell info to the data class
            if i == 0:
                new_x = supercell_atoms
                new_coords = supercell_coords
                new_batch = supercell_batch
                new_ptr = torch.zeros(data.num_graphs)
            else:
                new_x = torch.cat((new_x, supercell_atoms), dim=0)
                new_coords = torch.cat((new_coords, supercell_coords), dim=0)
                new_batch = torch.cat((new_batch, supercell_batch))
                new_ptr[i] = new_ptr[-1] + len(new_x)

        # update dataloader with cell info
        data.x = new_x.type(dtype=torch.float32)
        data.pos = new_coords.type(dtype=torch.float32)
        data.batch = new_batch.type(dtype=torch.int64)
        data.ptr = new_ptr.type(dtype=torch.int64)

        return data, cell_sample

    def differentiable_generated_supercells(self, cell_sample, supercell_data, config, override_position=None, override_orientation=None, override_cell_length=None, override_cell_angle=None):
        '''
        convert cell parameters to reference cell
        convert reference cell to 3x3 supercell
        all using differentiable torch functions
        '''
        volumes = []
        z_values = []
        sg_numbers = [int(supercell_data.y[2][i][self.sg_number_ind]) for i in range(supercell_data.num_graphs)]
        lattices = [self.lattice_type[number] for number in sg_numbers]

        cell_lengths, cell_angles, rand_position, rand_rotation = cell_sample.split(3, 1)

        cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
            cell_lengths, cell_angles, rand_position, rand_rotation, lattices, config.dataDims, enforce_crystal_system=False)

        if override_position is not None:
            rand_position = torch.tensor(override_position).to(rand_position.device)
        if override_orientation is not None:
            rand_rotation = torch.tensor(override_orientation).to(rand_rotation.device)
        if override_cell_length is not None:
            cell_lengths = torch.tensor(override_cell_length).to(rand_position.device)
        if override_cell_angle is not None:
            cell_angles = torch.tensor(override_cell_angle).to(rand_rotation.device)

        for i in range(supercell_data.num_graphs):
            atoms = supercell_data.x[supercell_data.batch == i]
            atomic_numbers = atoms[:, 0]
            # heavy_atom_inds = torch.argwhere(atomic_numbers > 1)[:, 0]
            # assert torch.sum(atomic_numbers == 1) == 0, 'hydrogens in supercell_dataset!'
            # atoms = atoms_i[heavy_atom_inds]
            coords = supercell_data.pos[supercell_data.batch == i, :]
            weights = torch.tensor([self.atom_weights[int(number)] for number in atomic_numbers]).to(config.device)

            sym_ops = torch.tensor(self.sym_ops[sg_numbers[i]], dtype=coords.dtype).to(coords.device)
            z_value = len(sym_ops)  # number of molecules in the reference cell
            z_values.append(z_value)

            T_fc, vol = coor_trans_matrix_torch('f_to_c', cell_lengths[i], cell_angles[i], return_vol=True)
            T_fc = T_fc.to(config.device)
            T_cf = torch.linalg.inv(T_fc)  # faster #coor_trans_matrix_torch('c_to_f', cell_lengths[i], cell_angles[i]).to(config.device)
            cell_vectors = torch.inner(T_fc, torch.eye(3).to(config.device)).T  # T_fc.dot(torch.eye(3)).T
            volumes.append(vol)

            random_coords = randomize_molecule_position_and_orientation_torch(
                coords, weights, T_fc, sym_ops,
                set_position=rand_position[i], set_rotation=rand_rotation[i])

            reference_cell = build_random_crystal_torch(T_cf, T_fc, random_coords, sym_ops, z_value)

            supercell_atoms, supercell_coords = ref_to_supercell_torch(reference_cell, z_value, atoms, cell_vectors)

            supercell_batch = torch.ones(len(supercell_atoms)).int() * i

            # append supercell info to the data class #
            if i == 0:
                new_x = supercell_atoms
                new_coords = supercell_coords
                new_batch = supercell_batch
                new_ptr = torch.zeros(supercell_data.num_graphs)
            else:
                new_x = torch.cat((new_x, supercell_atoms), dim=0)
                new_coords = torch.cat((new_coords, supercell_coords), dim=0)
                new_batch = torch.cat((new_batch, supercell_batch))
                new_ptr[i] = new_ptr[-1] + len(new_x)

        # update dataloader with cell info
        supercell_data.x = new_x.type(dtype=torch.float32)
        supercell_data.pos = new_coords.type(dtype=torch.float32)
        supercell_data.batch = new_batch.type(dtype=torch.int64)
        supercell_data.ptr = new_ptr.type(dtype=torch.int64)

        return supercell_data, z_values, volumes

    def fast_differentiable_generated_supercells(self, data, config, generator, do_cpu=True, override_position=None, override_orientation=None, override_cell_length=None, override_cell_angle=None):
        '''
        convert cell parameters to reference cell
        convert reference cell to 3x3 supercell
        all using differentiable torch functions
        note: currently it seems a bit faster on CPU
        '''
        supercell_data = data.clone()
        sg_numbers = [int(supercell_data.y[2][i][self.sg_number_ind]) for i in range(supercell_data.num_graphs)]
        # lattices = [self.lattice_type[number] for number in sg_numbers]
        # t0 = time.time()
        cell_sample = generator.forward(n_samples=supercell_data.num_graphs, conditions=supercell_data.to(generator.device))

        if do_cpu:
            supercell_data = supercell_data.cpu()
            cell_sample = cell_sample.cpu()

        cell_lengths, cell_angles, mol_position, mol_rotation = cell_sample.split(3, 1)

        cell_lengths, cell_angles, mol_position, mol_rotation = clean_cell_output(
            cell_lengths, cell_angles, mol_position, mol_rotation, None, config.dataDims, enforce_crystal_system=False)

        if override_position is not None:
            mol_position = torch.tensor(override_position).to(mol_position.device)
        if override_orientation is not None:
            mol_rotation = torch.tensor(override_orientation).to(mol_position.device)
        if override_cell_length is not None:
            cell_lengths = torch.tensor(override_cell_length).to(mol_position.device)
        if override_cell_angle is not None:
            cell_angles = torch.tensor(override_cell_angle).to(mol_position.device)

        # t05 = time.time()
        T_fc_list, T_cf_list, generated_cell_volumes = fast_differentiable_coor_trans_matrix(cell_lengths, cell_angles)

        # t075 = time.time()
        coords_list = []
        masses_list = []
        atoms_list = []
        for i in range(supercell_data.num_graphs):
            atoms_i = supercell_data.x[supercell_data.batch == i]
            atomic_numbers = atoms_i[:, 0]
            # heavy_atom_inds = torch.argwhere(atomic_numbers > 1)[:, 0]
            atoms_list.append(atoms_i)
            coords_list.append(supercell_data.pos[supercell_data.batch == i])
            masses_list.append(torch.tensor([self.atom_weights[int(number)] for number in atomic_numbers]).to(supercell_data.x.device))

        sym_ops_list = [torch.Tensor(self.sym_ops[sg_numbers[i]]).to(supercell_data.x.device) for i in range(len(sg_numbers))]
        z_values = [len(sym_ops) for sym_ops in sym_ops_list]

        # t1 = time.time()
        standardization_rotation_list = fast_differentiable_standard_rotation_matrix(masses_list, coords_list, T_fc_list)
        # t2 = time.time()
        applied_rotation_list = fast_differentiable_applied_rotation_matrix(mol_rotation)
        # t3 = time.time()
        canonical_mol_position = fast_differentiable_get_canonical_coords(mol_position, sym_ops_list)
        # t4 = time.time()
        final_coords_list = fast_differentiable_apply_rotations_and_translations(
            standardization_rotation_list, applied_rotation_list, coords_list, masses_list, T_fc_list, canonical_mol_position)
        # t5 = time.time()
        reference_cell_list = fast_differentiable_apply_point_symmetry(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values)
        # t6 = time.time()
        cell_vector_list = fast_differentiable_cell_vectors(T_fc_list)
        # t7 = time.time()
        supercell_list, supercell_atoms_list = fast_differentiable_ref_to_supercell(reference_cell_list, cell_vector_list, T_fc_list, atoms_list, z_values)
        # t8 = time.time()

        # print(f'generator took {t05-t0:.1f}')
        # print(f'Tfc took {t075-t05:.1f}')
        # print(f'listicles took {t1-t075:.1f}')
        # print(f'std mat took {t2-t1:.1f}')
        # print(f'rot mat took {t3-t2:.1f}')
        # print(f'canonical coords took {t4-t3:.1f}')
        # print(f'rot/trans took {t5-t4:.1f}')
        # print(f'point syms took {t6-t5:.1f}')
        # print(f'cell vectors took {t7-t6:.1f}')
        # print(f'supercell tiling took {t8-t7:.1f}')

        # append supercell info to the data class #
        for i in range(data.num_graphs):
            if i == 0:
                new_batch = torch.ones(len(supercell_atoms_list[i])).int() * i
                new_ptr = torch.zeros(supercell_data.num_graphs)
            else:
                new_batch = torch.cat((new_batch, torch.ones(len(supercell_atoms_list[i])).int() * i))
                new_ptr[i] = new_ptr[-1] + len(supercell_list[i])

        # update dataloader with cell info
        supercell_data.x = torch.cat(supercell_atoms_list).type(dtype=torch.float32)
        supercell_data.pos = torch.cat(supercell_list).type(dtype=torch.float32)
        supercell_data.batch = new_batch.type(dtype=torch.int64)
        supercell_data.ptr = new_ptr.type(dtype=torch.int64)

        return supercell_data.to(config.device), cell_sample.to(config.device), z_values, generated_cell_volumes.to(config.device)

    def real_supercells(self, supercell_data, config):
        '''
        test code for on-the-fly cell generation
        data = self.build_supercells(data)
        0. extract molecule and cell parameters
        1. find centroid
        2. find principal axis & angular component
        3. place centroid & align axes
        4. apply point symmetry
        5. tile supercell
        '''
        for i in range(supercell_data.num_graphs):
            # 0 extract molecule and cell parameters
            atoms = np.asarray(supercell_data.x[supercell_data.batch == i])
            # pre-enforce hydrogen cleanliness - shouldn't be necessary but you never know
            atomic_numbers = np.asarray(atoms[:, 0])  # this is currently atomic number - convert to masses
            heavy_inds = np.where(atomic_numbers != 1)
            atoms = atoms[heavy_inds]

            # get reference cell positions
            # first 3 columns are cartesian coords, last 3 are fractional

            cell_lengths = supercell_data.y[2][i][self.cell_length_inds]  # pull cell params from tracking inds
            cell_angles = supercell_data.y[2][i][self.cell_angle_inds]
            z_value = int(supercell_data.y[2][i][self.z_value_ind])
            T_fc = coor_trans_matrix('f_to_c', cell_lengths, cell_angles)
            cell_vectors = T_fc.dot(np.eye(3)).T

            reference_cell = supercell_data.y[3][i][:, :, :3]  # we're now pre-storing the packing rather than grabbing it from the CSD at runtime
            # #self.get_CSD_crystal(reader, csd_identifier=csd_identifier, mol_n_atoms=len(coords), z_value=z_value) # use CSD directly - slow
            # alternately, we could use the above random_coords functions with the known parameters to create the CSD cells (>99% accurate from a couple tests)

            supercell_atoms, supercell_coords = ref_to_supercell(reference_cell, z_value, atoms, cell_vectors)

            supercell_batch = torch.ones(len(supercell_atoms)).int() * i

            # append supercell info to the supercell_data class
            if i == 0:
                new_x = supercell_atoms
                new_coords = supercell_coords
                new_batch = supercell_batch
                new_ptr = torch.zeros(supercell_data.num_graphs)
            else:
                new_x = torch.cat((new_x, supercell_atoms), dim=0)
                new_coords = torch.cat((new_coords, supercell_coords), dim=0)
                new_batch = torch.cat((new_batch, supercell_batch))
                new_ptr[i] = new_ptr[-1] + len(new_x)

        # update dataloader with cell info
        supercell_data.x = new_x.type(dtype=torch.float32)
        supercell_data.pos = new_coords.type(dtype=torch.float32)
        supercell_data.batch = new_batch.type(dtype=torch.int64)
        supercell_data.ptr = new_ptr.type(dtype=torch.int64)

        return supercell_data

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

    def log_gan_accuracy(self, epoch, dataset_builder, train_loader, test_loader,
                         metrics_dict, g_tr_record, g_te_record, d_tr_record, d_te_record,
                         train_epoch_stats_dict, test_epoch_stats_dict, config,
                         generator, discriminator, wandb_log_figures = True):
        '''
        Do analysis and upload results to w&b

        1: generator efficiency on test set
            -:> cell parameter rmsd
            -:> RDF / RMSD20 (from compack, CCDC)
            -:> cell density
        Future: FF evaluations, polymorphs ranking

        :param epoch:
        :param dataset_builder:
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


        '''
        auxiliary regression loss
        '''
        if config.train_generator_density:
            target_mean = 0.673 # pacing coefficient mean #config.dataDims['mean']
            target_std = 0.0271# packing coefficient std dev #config.dataDims['std']

            target = np.asarray(test_epoch_stats_dict['generator auxiliary target'])
            prediction = np.asarray(test_epoch_stats_dict['generator auxiliary prediction'])
            orig_target = target * target_std + target_mean
            orig_prediction = prediction * target_std + target_mean

            train_target = np.asarray(train_epoch_stats_dict['generator auxiliary target'])
            train_prediction = np.asarray(train_epoch_stats_dict['generator auxiliary prediction'])
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

            wandb.log(loss_dict)

            # log loss distribution
            if config.wandb.log_figures:
                # predictions vs target trace
                xline = np.linspace(min(min(orig_target), min(orig_prediction)), max(max(orig_target), max(orig_prediction)), 1000)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=orig_target, y=orig_prediction, mode='markers', showlegend=False))
                fig.add_trace(go.Scatter(x=xline, y=xline, showlegend=False))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Test Prediction Trace': fig})

                xline = np.linspace(min(min(train_orig_target), min(train_orig_prediction)), max(max(train_orig_target), max(train_orig_prediction)), 1000)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_orig_target, y=train_orig_prediction, mode='markers', showlegend=False))
                fig.add_trace(go.Scatter(x=xline, y=xline, showlegend=False))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Train Prediction Trace': fig})

        return None

    def train_discriminator(self, generator, discriminator, config, data, i):
        t0 = time.time()

        real_supercell_data = self.real_supercells(data.clone(), config)
        generated_samples = generator.forward(n_samples=data.num_graphs, conditions=data.clone().to(generator.device))
        fake_supercell_data, z_values, generated_cell_volumes = self.differentiable_generated_supercells(generated_samples, data.clone(), config)
        # fake_supercell_data, raw_generation, z_values, generated_cell_volumes = self.fast_differentiable_generated_supercells(data, config, generator=generator)
        '''
        # supercell method comparison
        supercell_data1, raw_generation, z_values = self.differentiable_generated_supercells(data, config, generator=generator, override_position = torch.ones((50,3))/4, override_orientation = torch.ones((50,3)) * torch.pi / 2)
        supercell_data2, raw_generation, z_values, generated_cell_volumes = self.fast_differentiable_generated_supercells(data, config, generator=generator,  override_position = torch.ones((50,3))/4, override_orientation = torch.ones((50,3)) * torch.pi / 2)
        mol1 = Atoms(numbers=supercell_data1.x[supercell_data1.batch==0,0].cpu().detach().numpy(),positions=supercell_data1.pos[supercell_data1.batch==0,:].cpu().detach().numpy())
        mol2 = Atoms(numbers=supercell_data2.x[supercell_data2.batch==0,0].cpu().detach().numpy(),positions=supercell_data2.pos[supercell_data2.batch==0,:].cpu().detach().numpy())
        view((mol1,mol2))
        '''

        if i % 5 == 0:
            print('Batch {} supercell generation took {:.2f} seconds for {} samples'.format(i, round(time.time() - t0, 2), data.num_graphs))

        if config.device.lower() == 'cuda':
            real_supercell_data = real_supercell_data.cuda()
            fake_supercell_data = fake_supercell_data.cuda()

        if config.test_mode or config.anomaly_detection:
            assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"
            assert torch.sum(torch.isnan(fake_supercell_data.x)) == 0, "NaN in training input"

        score_on_real, raw_output_on_real = self.adversarial_loss(discriminator, real_supercell_data, config)
        score_on_fake, raw_output_on_fake = self.adversarial_loss(discriminator, fake_supercell_data, config)

        return score_on_real, score_on_fake

    def train_generator(self, generator, discriminator, config, data, i):
        t0 = time.time()

        # data, raw_generation = self.generated_supercells(data, config, generator = generator)
        generated_samples = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device))
        if config.train_generator_adversarially:
            supercell_data, z_values, generated_cell_volumes = self.differentiable_generated_supercells(generated_samples, data.clone(), config)
            # supercell_data, raw_generation, z_values, generated_cell_volumes = self.fast_differentiable_generated_supercells(data, config, generator=generator)

            if i % 5 == 0:
                print('Batch {} supercell generation took {:.2f} seconds for {} samples'.format(i, round(time.time() - t0, 2), data.num_graphs))

            if config.device.lower() == 'cuda':
                supercell_data = supercell_data.cuda()

            if config.test_mode or config.anomaly_detection:
                assert torch.sum(torch.isnan(data.x)) == 0, "NaN in training input"

            discriminator_score, raw_output = self.adversarial_loss(discriminator, supercell_data, config)

        else:
            discriminator_score = None

        if config.train_generator_density:
            if not config.train_generator_adversarially:
                sg_numbers = [int(data.y[2][i][self.sg_number_ind]) for i in range(data.num_graphs)]
                z_values = [len(self.sym_ops[sg_numbers[i]]) for i in range(data.num_graphs)]
                auxiliary_loss, auxiliary_prediction, auxiliary_target = self.generator_auxiliary_loss(data, generated_samples, z_values, config.dataDims)
            else:
                auxiliary_loss, auxiliary_prediction, auxiliary_target = self.generator_auxiliary_loss(data, generated_samples, z_values, config.dataDims, precomputed_volumes=generated_cell_volumes)
        else:
            auxiliary_loss, auxiliary_prediction, auxiliary_target = None, None, None

        return discriminator_score, generated_samples, auxiliary_loss, auxiliary_prediction, auxiliary_target

    def generator_auxiliary_loss(self, data, raw_sample, z_values, dataDims, precomputed_volumes=None):
        # compute proposed density and evaluate loss against known volume

        if precomputed_volumes is None:
            cell_lengths, cell_angles, rand_position, rand_rotation = raw_sample.split(3, 1)
            cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
                cell_lengths, cell_angles, rand_position, rand_rotation, None, dataDims)

        gen_packing = []
        csd_packing = []
        #volumes = []
        for i in range(len(raw_sample)):
            if precomputed_volumes is None:
                volume = cell_vol_torch(cell_lengths[i], cell_angles[i]) # torch.prod(cell_lengths[i]) # toss angles #
            else:
                volume = precomputed_volumes[i]

            mol_volume = data.y[2][i][self.z_value_ind] * data.y[2][i][self.mol_volume_ind]

            coeff = mol_volume / volume
            std_coeff = raw_sample[i,0] #(coeff - 0.673) / 0.0271

            #volumes.append(volume)
            gen_packing.append(std_coeff)  # compute packing index from given volume and restandardize
            csd_packing.append((data.y[2][i][self.crystal_packing_ind] - 0.673) / 0.0271)  # standardized true packing coefficient
            #gen_packing.append(raw_sample[i,0]) # first dimension of output
            #csd_packing.append((data.y[2][i][self.mol_volume_ind] - 348.12) / 114.61) # molecule volume substitute toy target (a number which is directly given to the model)

        generated_packing_coefficients = torch.stack(gen_packing)
        csd_packing_coefficients = torch.Tensor(csd_packing).to(generated_packing_coefficients.device)

        return (F.smooth_l1_loss(generated_packing_coefficients, csd_packing_coefficients, reduction='none'),
                generated_packing_coefficients.cpu().detach().numpy(), csd_packing_coefficients.cpu().detach().numpy())

    def flow_iter(self, generator, data):
        '''
        train the generator via standard normalizing flow loss
        '''

        zs, prior_logprob, log_det = generator(data)
        logprob = prior_logprob + log_det

        return -(logprob)  # , prior_logprob.cpu().detach().numpy()

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
import logging


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
            self.sym_ops_pg = {}
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

            # if config.add_spherical_basis is False:  # initializing spherical basis is too expensive to do repetitively
            #     generator, discriminator, g_optimizer, \
            #     g_schedulers, d_optimizer, d_schedulers, params1, params2 = self.init_gan(config, self.dataDims)

            try:
                _ = self.gan_epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                   g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                   update_gradients=True, record_stats=True, iteration_override=2, epoch = 1)

                # if successful, increase the batch and try again
                batch_size = max(batch_size + 5, int(batch_size * increment))
                train_loader = update_batch_size(train_loader,batch_size)
                test_loader = update_batch_size(test_loader,batch_size)
                #train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=batch_size)

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

    def train(self):
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=self.config.wandb.experiment_tag):
            # config = wandb.config # todo: wandb configs don't support nested namespaces. Sweeps are officially broken
            # print(config)

            config = self.config  # go with hand-built config

            # dataset
            dataset_builder = BuildDataset(config, self.point_groups, premade_dataset=self.prep_dataset)
            del self.prep_dataset  # we don't actually want this huge thing floating around

            config.dataDims = dataset_builder.get_dimension()
            self.dataDims = dataset_builder.get_dimension()

            self.cell_angle_keys = ['crystal alpha', 'crystal beta', 'crystal gamma']
            self.cell_angle_inds = [self.dataDims['tracking features dict'].index(key) for key in self.cell_angle_keys]
            self.cell_length_keys = ['crystal cell a', 'crystal cell b', 'crystal cell c']
            self.cell_length_inds = [self.dataDims['tracking features dict'].index(key) for key in self.cell_length_keys]
            self.z_value_ind = self.dataDims['tracking features dict'].index('crystal z value')
            self.sg_number_ind = self.dataDims['tracking features dict'].index('crystal spacegroup number')
            self.mol_volume_ind = self.dataDims['tracking features dict'].index('molecule volume')
            self.crystal_packing_ind = self.dataDims['tracking features dict'].index('crystal packing coefficient')
            self.crystal_density_ind = self.dataDims['tracking features dict'].index('crystal calculated density')
            self.mol_size_ind = self.dataDims['tracking features dict'].index('molecule num atoms')
            self.pg_ind_dict = {thing[14:]:ind + self.dataDims['n atomwise features'] for ind,thing in enumerate(self.dataDims['mol features']) if 'pg' in thing}

            self.randn_generator = independent_gaussian_model(input_dim = self.dataDims['n crystal features'],
                                                              means = self.dataDims['means'],
                                                              stds = self.dataDims['stds'])
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
                            self.gan_epoch(config, dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                           g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                           update_gradients=True, record_stats=True, epoch = epoch)  # train & compute test loss

                        with torch.no_grad():
                            d_err_te, d_te_record, g_err_te, g_te_record, test_epoch_stats_dict, time_test = \
                                self.gan_epoch(config, dataLoader=test_loader, generator=generator, discriminator=discriminator,
                                               update_gradients=False, record_stats=True, epoch = epoch)  # compute loss on test set


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
                            increment = max(4, int(train_loader.batch_size * 0.05)) # increment batch size
                            train_loader = update_batch_size(train_loader, train_loader.batch_size + increment)
                            test_loader = update_batch_size(test_loader, test_loader.batch_size + increment)
                            print(f'Batch size incremented to {train_loader.batch_size}')
                            wandb.log({'batch size':train_loader.batch_size})

                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e): # if we do hit OOM, slash the batch size
                            slash_increment = max(4, int(train_loader.batch_size * 0.1))
                            train_loader = update_batch_size(train_loader, train_loader.batch_size - slash_increment)
                            test_loader = update_batch_size(test_loader, test_loader.batch_size - slash_increment)
                            print('==============================')
                            print('OOMOOMOOMOOMOOMOOMOOMOOMOOMOOM')
                            print(f'Batch size slashed to {train_loader.batch_size} due to OOM')
                            print('==============================')
                            wandb.log({'batch size':train_loader.batch_size})
                        else:
                            raise e
                    epoch += 1

                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU

    def gan_epoch(self, config, dataLoader=None, generator=None, discriminator=None, g_optimizer=None, d_optimizer=None, update_gradients=True,
                  iteration_override=None, record_stats=False, epoch = None):
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
        g_den_losses = []
        g_adv_losses = []
        g_den_pred = []
        g_den_true = []
        g_pairwise_dist = []
        g_short_range_losses = []
        g_packing_losses = []
        real_pairwise_dist = []
        generated_samples_list = []
        epoch_stats_dict = {
            'tracking features': [],
        }
        for i, data in enumerate(dataLoader):
            '''
            noise injection
            '''
            if config.positional_noise > 0:
                data.pos += torch.randn_like(data.pos) * config.positional_noise
            '''
            train discriminator
            '''
            if config.train_discriminator_adversarially:
                score_on_real, score_on_fake, generated_samples, real_pairwise_dists, fake_pairwise_dists\
                    = self.train_discriminator(generator, discriminator, config, data, i)  # alternately trains on real and fake samples
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
                d_err.append(d_loss.data.cpu().detach().numpy())  # average overall loss
                d_loss_record.extend(d_losses.cpu().detach().numpy())  # overall loss distribution

                if update_gradients:
                    d_optimizer.zero_grad()  # reset gradients from previous passes
                    d_loss.backward()  # back-propagation
                    d_optimizer.step()  # update parameters

                g_pairwise_dist.append(fake_pairwise_dists.cpu().detach().numpy())
                real_pairwise_dist.append(real_pairwise_dists.cpu().detach().numpy())
                #generated_samples_list.append(generated_samples) # ignore for now - mixture of generator and other garbage
            else:
                d_err.append(np.zeros(1))
                d_loss_record.extend(np.zeros(data.num_graphs))

            '''
            train_generator
            '''
            if any((config.train_generator_density, config.train_generator_adversarially, config.train_generator_range_cutoff, config.train_generator_packing)):

                adversarial_score, generated_samples, density_loss, density_prediction, density_target, packing_loss, short_range_loss, generated_pairwise_distances = \
                    self.train_generator(generator, discriminator, config, data, i)

                if adversarial_score is not None:
                    if config.gan_loss == 'wasserstein':
                        adversarial_loss = -adversarial_score  # generator wants to maximize the score
                    elif config.gan_loss == 'standard':
                        adversarial_loss = 1 - adversarial_score
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
                    g_adv_losses.append(adversarial_loss.cpu().detach().numpy())

                if config.train_generator_range_cutoff:
                    g_losses_list.append(short_range_loss)
                    g_short_range_losses.append(short_range_loss.cpu().detach().numpy())

                if config.train_generator_packing:
                    g_losses_list.append(packing_loss)
                    g_packing_losses.append(packing_loss.cpu().detach().numpy())

                if config.train_generator_adversarially or config.train_generator_range_cutoff:
                    g_pairwise_dist.append(generated_pairwise_distances.cpu().detach().numpy())


                g_losses = torch.mean(torch.stack(g_losses_list),dim=0)

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
                if (epoch < config.cut_max_prob_training_after): # stop using max_prob training after a few initial epochs

                    g_flow_losses = self.flow_iter(generator, data.to(config.device))

                    g_flow_loss = g_flow_losses.mean()
                    g_flow_err.append(g_flow_loss.data.cpu())  # average loss
                    g_flow_loss_record.append(g_flow_losses.cpu().detach().numpy())  # loss distribution

                    if update_gradients:
                        g_optimizer.zero_grad()  # reset gradients from previous passes
                        g_flow_loss.backward()  # back-propagation
                        g_optimizer.step()  # update parameters

            if config.train_generator_on_randn:
                if (epoch < config.cut_max_prob_training_after): # stop using max_prob training after a few initial epochs
                    g_max_prob_losses = self.max_prob_iter(generator, data)

                    g_max_prob_loss = g_max_prob_losses.mean()
                    #g_max_prob_err.append(g_max_prob_loss.data.cpu())  # average loss
                    #g_max_prob_loss_record.append(g_max_prob_losses.cpu().detach().numpy())  # loss distribution

                    if update_gradients:
                        g_optimizer.zero_grad()  # reset gradients from previous passes
                        g_max_prob_loss.backward()  # back-propagation
                        g_optimizer.step()  # update parameters

            if (len(generated_samples_list) < i) and record_stats: # make some samples for analysis if we have none so far from this step
                generated_samples = generator(len(data.y[0]), z=None, conditions=data.to(config.device))
                generated_samples_list.append(generated_samples.cpu().detach().numpy())

            if record_stats:
                epoch_stats_dict['tracking features'].extend(data.y[2])

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0

        if record_stats:
            epoch_stats_dict['discriminator on real loss'] = np.concatenate(d_real_losses) if d_real_losses != [] else None
            epoch_stats_dict['discriminator on fake loss'] = np.concatenate(d_fake_losses) if d_fake_losses != [] else None
            epoch_stats_dict['generator density loss'] = np.concatenate(g_den_losses) if g_den_losses != [] else None
            epoch_stats_dict['generator adversarial loss'] = np.concatenate(g_adv_losses) if g_adv_losses != [] else None
            epoch_stats_dict['generator flow loss'] = np.concatenate(g_flow_loss_record) if g_flow_loss_record != [] else None
            epoch_stats_dict['generator short range loss'] = np.concatenate(g_short_range_losses) if g_short_range_losses != [] else None
            epoch_stats_dict['generator packing loss'] = np.concatenate(g_packing_losses) if g_packing_losses != [] else None
            epoch_stats_dict['generator density prediction'] = np.concatenate(g_den_pred) if g_den_pred != [] else None
            epoch_stats_dict['generator density target'] = np.concatenate(g_den_true) if g_den_true != [] else None
            epoch_stats_dict['generated pairwise distance hist'] = np.histogram(np.concatenate(g_pairwise_dist), bins=50, range=(0, 5), density=True) if g_pairwise_dist != [] else None
            epoch_stats_dict['real pairwise distance hist'] = np.histogram(np.concatenate(real_pairwise_dist), bins=50, range=(0, 5), density=True) if real_pairwise_dist != [] else None
            epoch_stats_dict['generated cell parameters'] = np.concatenate(generated_samples_list) if generated_samples_list != [] else None

            return d_err, d_loss_record, g_err, g_loss_record, epoch_stats_dict, total_time
        else:
            return d_err, d_loss_record, g_err, g_loss_record, total_time

    def adversarial_loss(self, discriminator, data, config):

        output, dists = discriminator(data, return_dists=True)  # reshape output from flat filters to channels * filters per channel

        # discriminator score
        if config.gan_loss == 'wasserstein':
            scores = output[:,0] #F.softplus(output[:, 0])  # critic score - higher meaning more plausible

        elif config.gan_loss == 'standard':
            scores = F.softmax(output, dim=1)[:, -1]  # probability of 'yes'

        else:
            print(config.gan_loss + ' is not a valid GAN loss function!')
            sys.exit()

        return scores, output, dists

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
        wandb.log(special_losses)


    def fast_differentiable_generated_supercells(self, supercell_data, config, cell_sample, do_cpu=True, override_position=None, override_orientation=None, override_cell_length=None, override_cell_angle=None, override_pg = None):
        '''
        convert cell parameters to reference cell
        convert reference cell to 3x3 supercell
        all using differentiable torch functions
        note: currently it seems a bit faster on CPU
        '''

        if do_cpu:
            supercell_data = supercell_data.cpu()
            cell_sample = cell_sample.cpu()

        if override_pg is not None:
            pg_ind = self.pg_ind_dict[override_pg]
            supercell_data.x[:,min(self.pg_ind_dict.values()):max(self.pg_ind_dict.values())] = 0 # set all pgs to 0
            supercell_data.x[:,pg_ind] = 1 # set all molecules to the given pg
            pick_sg_ind = list(self.point_groups.keys())[list(self.point_groups.values()).index(override_pg)]
            sg_numbers = [pick_sg_ind for i in range(supercell_data.num_graphs)]
            sym_ops_list = [torch.Tensor(self.sym_ops[sg_numbers[i]]).to(supercell_data.x.device) for i in range(len(sg_numbers))]
            z_values = [len(sym_ops) for sym_ops in sym_ops_list]

            # update to correct Z values
            #z_value_vec = supercell_data.x[:,self.z_value_ind].clone()
            for i in range(len(z_values)):
                supercell_data.x[supercell_data.batch==i, self.z_value_ind] = z_values[i] # todo norm/std

        else:
            sg_numbers = [int(supercell_data.y[2][i][self.sg_number_ind]) for i in range(supercell_data.num_graphs)]
            sym_ops_list = [torch.Tensor(self.sym_ops[sg_numbers[i]]).to(supercell_data.x.device) for i in range(len(sg_numbers))]
            z_values = [len(sym_ops) for sym_ops in sym_ops_list]

        # lattices = [self.lattice_type[number] for number in sg_numbers]

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

        T_fc_list, T_cf_list, generated_cell_volumes = fast_differentiable_coor_trans_matrix(cell_lengths, cell_angles)

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

        standardization_rotation_list = fast_differentiable_standard_rotation_matrix(masses_list, coords_list, T_fc_list)
        applied_rotation_list = fast_differentiable_applied_rotation_matrix(mol_rotation)
        canonical_mol_position = fast_differentiable_get_canonical_coords(mol_position, sym_ops_list, z_values)
        final_coords_list = fast_differentiable_apply_rotations_and_translations(
            standardization_rotation_list, applied_rotation_list, coords_list, masses_list, T_fc_list, canonical_mol_position)
        reference_cell_list = fast_differentiable_apply_point_symmetry(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values)
        cell_vector_list = fast_differentiable_cell_vectors(T_fc_list)
        supercell_list, supercell_atoms_list = fast_differentiable_ref_to_supercell(reference_cell_list, cell_vector_list, T_fc_list, atoms_list, z_values, supercell_scale=config.supercell_size)

        # append supercell info to the data class #
        for i in range(supercell_data.num_graphs):
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

        return supercell_data.to(config.device), z_values, generated_cell_volumes.to(config.device)


    def fast_real_supercells(self, supercell_data, config, do_cpu=True):
        '''
        should be faster than the old way
        pretty quick on cpu
        '''
        sg_numbers = [int(supercell_data.y[2][i][self.sg_number_ind]) for i in range(supercell_data.num_graphs)]
        # lattices = [self.lattice_type[number] for number in sg_numbers]
        # t0 = time.time()

        if do_cpu:
            supercell_data = supercell_data.cpu()

        cell_lengths = torch.stack([torch.tensor(supercell_data.y[2][i][self.cell_length_inds]) for i in range(supercell_data.num_graphs)]).to(supercell_data.x.device)
        cell_angles = torch.stack([torch.tensor(supercell_data.y[2][i][self.cell_angle_inds]) for i in range(supercell_data.num_graphs)]).to(supercell_data.x.device)

        T_fc_list, T_cf_list, generated_cell_volumes = fast_differentiable_coor_trans_matrix(cell_lengths, cell_angles)

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

        reference_cell_list = [torch.tensor(supercell_data.y[3][i][:, :, :3]).to(supercell_data.x.device) for i in range(supercell_data.num_graphs)]
        cell_vector_list = fast_differentiable_cell_vectors(T_fc_list)
        supercell_list, supercell_atoms_list = fast_differentiable_ref_to_supercell(reference_cell_list, cell_vector_list, T_fc_list, atoms_list, z_values, supercell_scale = config.supercell_size)

        for i in range(supercell_data.num_graphs):
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

        return supercell_data.to(config.device)


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
        #                         caption="Epoch {} Molecule {} bad sample {} classified as {}".format(epoch, smiles_set[i], self.groupLabels[int(true_set[i])], self.groupLabels[int(guess_set[i])]))
        #     wandb.log({"Bad examples": image})


        '''
        cell parameter analysis
        '''
        # correlate losses with molecular features
        tracking_features = np.asarray(test_epoch_stats_dict['tracking features'])
        g_loss_correlations = np.zeros(config.dataDims['n tracking features'])
        d_loss_correlations = np.zeros(config.dataDims['n tracking features'])
        features = []
        for i in range(config.dataDims['n tracking features']): # not that interesting
            features.append(config.dataDims['tracking features dict'][i])
            g_loss_correlations[i] = np.corrcoef(g_te_record, tracking_features[:, i], rowvar=False)[0, 1]
            d_loss_correlations[i] = np.corrcoef(d_te_record, tracking_features[:, i], rowvar=False)[0, 1]

        g_sort_inds = np.argsort(g_loss_correlations)
        g_loss_correlations = g_loss_correlations[g_sort_inds]

        d_sort_inds = np.argsort(d_loss_correlations)
        d_loss_correlations = d_loss_correlations[d_sort_inds]

        if config.wandb.log_figures:
            fig = go.Figure(go.Bar(
                y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['n tracking features'])],
                x=[g_loss_correlations[i] for i in range(config.dataDims['n tracking features'])],
                orientation='h',
            ))
            wandb.log({'G Loss correlations': fig})

            fig = go.Figure(go.Bar(
                y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['n tracking features'])],
                x=[d_loss_correlations[i] for i in range(config.dataDims['n tracking features'])],
                orientation='h',
            ))
            wandb.log({'D Loss correlations': fig})

        if train_epoch_stats_dict['generated cell parameters'] is not None:
            n_crystal_features = config.dataDims['n crystal features']
            generated_samples = test_epoch_stats_dict['generated cell parameters']
            means = config.dataDims['means']
            stds = config.dataDims['stds']

            if not hasattr(self, 'dataset_cell_distribution'):
                self.dataset_cell_distribution = np.concatenate([datapoint.y[0].cpu().detach().numpy() for datapoint in train_loader.dataset])  # build it
                for i in range(self.dataset_cell_distribution.shape[1]):
                    self.dataset_cell_distribution[:, i] = self.dataset_cell_distribution[:, i] * stds[i] + means[i]

            # raw outputs
            renormalized_samples = np.zeros_like(generated_samples)
            for i in range(generated_samples.shape[1]):
                renormalized_samples[:, i] = generated_samples[:, i] * stds[i] + means[i]

            # samples as the builder sees them
            cell_lengths, cell_angles, rand_position, rand_rotation = torch.tensor(generated_samples).split(3, 1)
            cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
                cell_lengths, cell_angles, rand_position, rand_rotation, None, config.dataDims, enforce_crystal_system=False)
            cleaned_samples = torch.cat((cell_lengths, cell_angles, rand_position, rand_rotation),dim=1).detach().numpy()

            overlaps_1d = {}
            for i, key in enumerate(config.dataDims['crystal features']):
                mini, maxi = np.amin(self.dataset_cell_distribution[:, i]), np.amax(self.dataset_cell_distribution[:, i])
                h1, r1 = np.histogram(self.dataset_cell_distribution[:, i], bins=100, range=(mini, maxi))
                h1 = h1 / len(self.dataset_cell_distribution[:, i])

                h2, r2 = np.histogram(renormalized_samples[:, i], bins=r1)
                h2 = h2 / len(renormalized_samples[:, i])

                overlaps_1d[f'{key} 1D Overlap'] = np.min(np.concatenate((h1[None], h2[None]), axis=0), axis=0).sum()

            average_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys()])
            overlaps_1d['average 1D overlap'] = average_overlap
            wandb.log(overlaps_1d.copy())
            print("1D Overlap With Data:{:.3f}".format(average_overlap))

            if wandb_log_figures:
                fig_dict = {}

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
                        x=self.dataset_cell_distribution[:, i],
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
        cell pairwise atomic distances
        '''
        if train_epoch_stats_dict['generated pairwise distance hist'] is not None:
            wandb.log({"Train Close Range RDF": wandb.Histogram(np_histogram=train_epoch_stats_dict['generated pairwise distance hist'])})
            wandb.log({"Test Close Range RDF": wandb.Histogram(np_histogram=test_epoch_stats_dict['generated pairwise distance hist'])})

        if train_epoch_stats_dict['real pairwise distance hist'] is not None:
            tr_real_g2, rrange = train_epoch_stats_dict['real pairwise distance hist']  # same range by construction
            te_real_g2, _ = test_epoch_stats_dict['real pairwise distance hist']
            tr_fake_g2, _ = train_epoch_stats_dict['generated pairwise distance hist']
            te_fake_g2, _ = test_epoch_stats_dict['generated pairwise distance hist']
            bin_width = rrange[1] - rrange[0]

            range_analysis_dict = {}
            # get histogram overlaps
            range_analysis_dict['tr g2 overlap'] = np.min(np.concatenate((tr_real_g2[None, :] * bin_width, tr_fake_g2[None, :] * bin_width), axis=0), axis=0).sum()
            range_analysis_dict['te g2 overlap'] = np.min(np.concatenate((te_real_g2[None, :] * bin_width, te_fake_g2[None, :] * bin_width), axis=0), axis=0).sum()

            # get probability mass at too-close range (should be ~zero)
            range_analysis_dict['tr short range density'] = np.sum(tr_fake_g2[(rrange < 1)[:-1]] * bin_width)
            range_analysis_dict['te short range density'] = np.sum(te_fake_g2[(rrange < 1)[:-1]] * bin_width)

            wandb.log(range_analysis_dict)

        '''
        auxiliary regression target
        '''
        if test_epoch_stats_dict['generator density target'] is not None: #config.train_generator_density:
            target_mean = 0.673  # pacing coefficient mean #config.dataDims['mean']
            target_std = 0.0271  # packing coefficient std dev #config.dataDims['std']

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

            wandb.log(loss_dict)

            # log loss distribution
            if config.wandb.log_figures:
                # predictions vs target trace
                xline = np.linspace(max(min(orig_target), min(orig_prediction)), min(max(orig_target), max(orig_prediction)), 10)
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(x=orig_target,y=orig_prediction,ncontours=50,nbinsx=20,nbinsy=20,showlegend=True))
                fig.add_trace(go.Scatter(x=orig_target, y=orig_prediction, mode='markers', showlegend=True,opacity=0.5))
                fig.add_trace(go.Scatter(x=xline, y=xline))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Test Prediction Trace': fig})


                xline = np.linspace(max(min(train_orig_target), min(train_orig_prediction)), min(max(train_orig_target), max(train_orig_prediction)), 10)
                fig = go.Figure()
                fig.add_trace(go.Histogram2dContour(x=train_orig_target,y=train_orig_prediction,ncontours=50,nbinsx=20,nbinsy=20,showlegend=True))
                fig.add_trace(go.Scatter(x=train_orig_target, y=train_orig_prediction, mode='markers', showlegend=True,opacity=0.5))
                fig.add_trace(go.Scatter(x=xline, y=xline))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Train Prediction Trace': fig})

        return None

    def train_discriminator(self, generator, discriminator, config, data, i, sample_type = 'generator'):

        if sample_type == 'generator':
            generated_samples = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device))
        elif sample_type == 'randn':
            generated_samples = self.randn_generator.forward(data.num_graphs).to(generator.device)
        if not (sample_type == 'noisy'):
            real_supercell_data = self.fast_real_supercells(data.clone(), config)
            fake_supercell_data, z_values, generated_cell_volumes = self.fast_differentiable_generated_supercells(data.clone().to(generated_samples.device), config, generated_samples, override_pg = config.generate_pgs)
        else:
            real_supercell_data = self.fast_real_supercells(data.clone(), config)
            fake_supercell_data = real_supercell_data.clone()
            fake_supercell_data.pos += torch.randn_like(fake_supercell_data) * 10 # huge amount of noise - should be basically nonsense
        if False:#i == 0:
            print('Batch {} real + fake supercell generation took {:.2f} seconds for {} samples'.format(i, round(t1 - t0, 2), data.num_graphs))

        if config.device.lower() == 'cuda': # redundant
            real_supercell_data = real_supercell_data.cuda()
            fake_supercell_data = fake_supercell_data.cuda()

        if config.test_mode or config.anomaly_detection:
            assert torch.sum(torch.isnan(real_supercell_data.x)) == 0, "NaN in training input"
            assert torch.sum(torch.isnan(fake_supercell_data.x)) == 0, "NaN in training input"

        score_on_real, raw_output_on_real, real_pairwise_distances = self.adversarial_loss(discriminator, real_supercell_data, config)
        score_on_fake, raw_output_on_fake, fake_pairwise_distances = self.adversarial_loss(discriminator, fake_supercell_data, config)

        return score_on_real, score_on_fake, generated_samples.cpu().detach().numpy(), real_pairwise_distances, fake_pairwise_distances

    def train_generator(self, generator, discriminator, config, data, i):
        pairwise_distances = None
        # data, raw_generation = self.generated_supercells(data, config, generator = generator)
        generated_samples = generator.forward(n_samples=data.num_graphs, conditions=data.to(generator.device))
        if config.train_generator_adversarially or config.train_generator_packing or config.train_generator_range_cutoff:

            supercell_data, z_values, generated_cell_volumes = self.fast_differentiable_generated_supercells(data.clone(), config, generated_samples, override_pg = config.generate_pgs)

            if False:#:i == 0:
                print('Batch {} fake supercell generation took {:.2f} seconds for {} samples'.format(i, round(t1 - t0, 2), data.num_graphs))

        if config.train_generator_adversarially or config.train_generator_range_cutoff:
            if config.device.lower() == 'cuda':
                supercell_data = supercell_data.cuda()

            if config.test_mode or config.anomaly_detection:
                assert torch.sum(torch.isnan(data.x)) == 0, "NaN in training input"

            discriminator_score, raw_output, pairwise_distances = self.adversarial_loss(discriminator, supercell_data, config)
        else:
            discriminator_score = None


        # always check the density
        #sg_numbers = [int(data.y[2][i][self.sg_number_ind]) for i in range(data.num_graphs)]
        #z_values = [len(self.sym_ops[sg_numbers[i]]) for i in range(data.num_graphs)]
        density_loss, density_prediction, density_target, packing_loss = self.generator_density_loss(data, generated_samples, z_values, config.dataDims)
        #torch.mean(F.relu(-(pairwise_distances - 1)))  # alternate
        short_range_loss_i = torch.mean(torch.sinc(torch.clip(pairwise_distances,max=1))) # atoms should not be too close - not split out sample-wise
        short_range_loss = torch.ones_like(packing_loss) * short_range_loss_i * 1000 # scaling required to balance vs other losses

        if self.config.test_mode:
            assert torch.sum(torch.isnan(short_range_loss)) == 0

        return discriminator_score, generated_samples.cpu().detach().numpy(), density_loss, density_prediction, density_target, packing_loss, short_range_loss, pairwise_distances

    def generator_density_loss(self, data, raw_sample, z_values, dataDims, precomputed_volumes=None):
        # compute proposed density and evaluate loss against known volume

        if precomputed_volumes is None:
            cell_lengths, cell_angles, rand_position, rand_rotation = raw_sample.split(3, 1)
            cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
                cell_lengths, cell_angles, rand_position, rand_rotation, None, dataDims)

        gen_packing = []
        csd_packing = []
        # volumes = []
        for i in range(len(raw_sample)):
            if precomputed_volumes is None:
                volume = cell_vol_torch(cell_lengths[i], cell_angles[i])  # torch.prod(cell_lengths[i]) # toss angles #
            else:
                volume = precomputed_volumes[i]

            mol_volume = z_values[i] * data.y[2][i][self.mol_volume_ind]

            coeff = mol_volume / volume
            std_coeff = (coeff - 0.673) / 0.0271 # todo replace these with values from the dataset

            # volumes.append(volume)
            gen_packing.append(std_coeff)  # compute packing index from given volume and restandardize
            csd_packing.append((data.y[2][i][self.crystal_packing_ind] - 0.673) / 0.0271)  # standardized true packing coefficient # todo this too
            # gen_packing.append(raw_sample[i,0]) # first dimension of output
            # csd_packing.append((data.y[2][i][self.mol_volume_ind] - 348.12) / 114.61) # molecule volume substitute toy target (a number which is directly given to the model)

        generated_packing_coefficients = torch.stack(gen_packing)
        # generated_packing_coefficients = raw_sample[:,0]
        csd_packing_coefficients = torch.Tensor(csd_packing).to(generated_packing_coefficients.device)

        #den_loss = F.smooth_l1_loss(generated_packing_coefficients, csd_packing_coefficients, reduction='none') # raw value
        #den_loss = torch.abs(torch.sqrt(F.smooth_l1_loss(generated_packing_coefficients, csd_packing_coefficients, reduction='none'))) # abs(sqrt()) is a soft rescaling to avoid gigantic losses
        den_loss = torch.log(1 + F.smooth_l1_loss(generated_packing_coefficients, csd_packing_coefficients, reduction='none')) # log(1+loss) is a better soft rescaling to avoid gigantic losses
        cutoff = (0.55 - 0.673) / 0.0271 # cutoff in standardized basis
        packing_loss = F.relu(-(generated_packing_coefficients - cutoff)) # linear loss below a cutoff

        if self.config.test_mode:
            assert torch.sum(torch.isnan(packing_loss)) == 0
            assert torch.sum(torch.isnan(den_loss)) == 0

        return den_loss, generated_packing_coefficients.cpu().detach().numpy(), csd_packing_coefficients.cpu().detach().numpy(), packing_loss

    def flow_iter(self, generator, data):
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

        return -log_probs # want to maximize this objective
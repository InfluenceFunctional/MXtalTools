import wandb
import glob
from torch import backends, optim
import torch.optim.lr_scheduler as lr_scheduler
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
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from PIL import Image

from utils import *

from dataset_management.dataset_utils import BuildDataset, get_dataloaders, update_dataloader_batch_size, get_extra_test_loader
from dataset_management.dataset_manager import Miner

from crystal_building.crystal_builder_tools import *
from crystal_building.coordinate_transformations import cell_vol
from crystal_building.supercell_builders import SupercellBuilder, override_sg_info

from models.model_utils import *
from models.crystal_rdf import crystal_rdf
from models.vdw_overlap import vdw_overlap
from sampling.SampleOptimization import gradient_descent_sampling
from models.generator_models import crystal_generator
from models.discriminator_models import crystal_discriminator
from models.regression_models import molecule_regressor
from models.torch_models import independent_gaussian_model
from sampling.STUN_MC import mcmcSampler


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

    def prep_metrics(self):
        '''
        initialize key metrics to follow during training
        Returns
        -------

        '''
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
        if self.config.g_model_path is not None:
            g_checkpoint = torch.load(config.g_model_path)
            # save learning rates so we can un-overwrite them
            max_lr = self.config.generator.max_lr * 1
            lr = self.config.generator.learning_rate * 1
            self.config.generator = Namespace(**g_checkpoint['config'])  # overwrite the settings for the model
            self.config.generator.learning_rate = lr
            self.config.generator.max_lr = max_lr
        if self.config.d_model_path is not None:
            d_checkpoint = torch.load(config.d_model_path)
            max_lr = self.config.discriminator.max_lr * 1
            lr = self.config.discriminator.learning_rate * 1
            self.config.discriminator = Namespace(**d_checkpoint['config'])
            self.config.discriminator.learning_rate = lr
            self.config.discriminator.max_lr = max_lr
        print("Initializing models for " + self.config.mode)
        if self.config.mode == 'gan' or self.config.mode == 'sampling':
            generator = crystal_generator(config, dataDims)
            discriminator = crystal_discriminator(config, dataDims)
        elif self.config.mode == 'regression':
            generator = molecule_regressor(config, dataDims)
            discriminator = nn.Linear(1, 1)  # dummy model
        else:
            print(f'{config.mode} is not an implemented method!')
            sys.exit()

        if self.config.device.lower() == 'cuda':
            print('Putting models on CUDA')
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            generator = generator.cuda()
            discriminator = discriminator.cuda()

        # init optimizers
        amsgrad = True
        beta1 = self.config.generator.beta1  # 0.9
        beta2 = self.config.generator.beta2  # 0.999
        weight_decay = self.config.generator.weight_decay  # 0.01
        momentum = 0

        if self.config.generator.optimizer == 'adam':
            g_optimizer = optim.Adam(generator.parameters(), amsgrad=amsgrad, lr=config.generator.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif self.config.generator.optimizer == 'adamw':
            g_optimizer = optim.AdamW(generator.parameters(), amsgrad=amsgrad, lr=config.generator.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif self.config.generator.optimizer == 'sgd':
            g_optimizer = optim.SGD(generator.parameters(), lr=config.generator.learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            print(config.generator.optimizer + ' is not a valid optimizer')
            sys.exit()

        amsgrad = False
        beta1 = self.config.discriminator.beta1  # 0.9
        beta2 = self.config.discriminator.beta2  # 0.999
        weight_decay = self.config.discriminator.weight_decay  # 0.01
        momentum = 0

        if self.config.discriminator.optimizer == 'adam':
            d_optimizer = optim.Adam(discriminator.parameters(), amsgrad=amsgrad, lr=config.discriminator.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif self.config.discriminator.optimizer == 'adamw':
            d_optimizer = optim.AdamW(discriminator.parameters(), amsgrad=amsgrad, lr=config.discriminator.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif self.config.discriminator.optimizer == 'sgd':
            d_optimizer = optim.SGD(discriminator.parameters(), lr=config.discriminator.learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            print(config.discriminator.optimizer + ' is not a valid optimizer')
            sys.exit()

        if self.config.g_model_path is not None:
            generator, g_optimizer = reload_model(generator, g_optimizer, self.config.g_model_path)
        if self.config.d_model_path is not None:
            discriminator, d_optimizer = reload_model(discriminator, d_optimizer, self.config.d_model_path)

        # cuda
        if self.config.device.lower() == 'cuda':
            pass
            # generator = gnn.DataParallel(generator)
            # discriminator = gnn.DataParallel(discriminator)

        # init schedulers
        scheduler1 = lr_scheduler.ReduceLROnPlateau(
            g_optimizer,
            mode='min',
            factor=0.1,
            patience=100,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=15
        )
        lr_lambda = lambda epoch: self.config.lr_growth_lambda
        scheduler2 = lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lr_lambda)
        lr_lambda2 = lambda epoch: self.config.lr_shrink_lambda
        scheduler3 = lr_scheduler.MultiplicativeLR(g_optimizer, lr_lambda=lr_lambda2)

        scheduler4 = lr_scheduler.ReduceLROnPlateau(
            d_optimizer,
            mode='min',
            factor=0.1,
            patience=100,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=15
        )
        lr_lambda = lambda epoch: self.config.lr_growth_lambda
        scheduler5 = lr_scheduler.MultiplicativeLR(d_optimizer, lr_lambda=lr_lambda)
        lr_lambda2 = lambda epoch: self.config.lr_shrink_lambda
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
        init_batch_size = self.config.min_batch_size.real
        max_batch_size = self.config.max_batch_size.real
        batch_reduction_factor = self.config.auto_batch_reduction

        train_loader, test_loader = get_dataloaders(dataset, config, override_batch_size=init_batch_size)

        increment = 1.5  # what fraction by which to increment the batch size
        batch_size = int(init_batch_size)

        while (not finished) and (batch_size < max_batch_size):
            self.config.final_batch_size = batch_size

            if self.config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU cache
                generator.cuda()
                discriminator.cuda()

            try:
                _ = self.epoch(dataLoader=train_loader, generator=generator, discriminator=discriminator,
                               g_optimizer=g_optimizer, d_optimizer=d_optimizer,
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
            generator, discriminator, g_optimizer, g_schedulers, d_optimizer, d_schedulers, params1, params2 \
                = self.init_gan(self.config, self.config.dataDims)

            # get batch size
            if self.config.auto_batch_sizing:
                print('Finding optimal batch size')
                train_loader, test_loader, self.config.final_batch_size = \
                    self.get_batch_size(generator, discriminator, g_optimizer, d_optimizer,
                                        dataset_builder, self.config)

                # reload preloaded models (there are weight updates in batch sizing)
                if self.config.g_model_path is not None:
                    generator, g_optimizer = reload_model(generator, g_optimizer, self.config.g_model_path)
                if self.config.d_model_path is not None:
                    discriminator, d_optimizer = reload_model(discriminator, d_optimizer, self.config.d_model_path)
            else:
                print('Getting dataloaders for pre-determined batch size')
                train_loader, test_loader = get_dataloaders(dataset_builder, self.config)
                self.config.final_batch_size = self.config.max_batch_size

            del dataset_builder
            print("Training batch size set to {}".format(self.config.final_batch_size))

            if (self.config.extra_test_set_paths is not None) and self.config.extra_test_evaluation:
                extra_test_loader = get_extra_test_loader(self.config, self.config.extra_test_set_paths, dataDims=self.config.dataDims,
                                                          pg_dict=self.point_groups, sg_dict=self.space_groups, lattice_dict=self.lattice_type)
            else:
                extra_test_loader = None

            # model, optimizer, schedulers
            print('Reinitializing model and optimizer')
            if self.config.g_model_path is None:
                generator.apply(weight_reset)
            if self.config.d_model_path is None:
                discriminator.apply(weight_reset)
            n_params = params1 + params2

            wandb.watch((generator, discriminator), log_graph=True, log_freq=100)
            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": self.config.final_batch_size})

            metrics_dict = self.prep_metrics()

            # training loop
            d_hit_max_lr, g_hit_max_lr, converged, epoch = False, False, self.config.max_epochs == 0, 0  # for evaluation mode
            with torch.autograd.set_detect_anomaly(self.config.anomaly_detection):
                while (epoch < self.config.max_epochs) and not converged:
                    # very cool
                    print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                    print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                    print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                    # very cool
                    print("Starting Epoch {}".format(epoch))  # index from 0, very cool

                    extra_test_epoch_stats_dict = None
                    try:
                        # train & compute train loss
                        d_err_tr, d_tr_record, g_err_tr, g_tr_record, train_epoch_stats_dict, time_train = \
                            self.epoch(dataLoader=train_loader, generator=generator, discriminator=discriminator,
                                       g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                                       update_gradients=True, record_stats=True, epoch=epoch)

                        with torch.no_grad():
                            # compute test loss
                            d_err_te, d_te_record, g_err_te, g_te_record, test_epoch_stats_dict, time_test = \
                                self.epoch(dataLoader=test_loader, generator=generator, discriminator=discriminator,
                                           update_gradients=False, record_stats=True, epoch=epoch)

                            if (extra_test_loader is not None) and (epoch % self.config.extra_test_period == 0):
                                extra_test_epoch_stats_dict, time_test_ex = \
                                    self.discriminator_evaluation(dataLoader=extra_test_loader, discriminator=discriminator)  # compute loss on test set
                                print(f'Extra test evaluation took {time_test_ex:.1f} seconds')

                        print('epoch={}; d_nll_tr={:.5f}; d_nll_te={:.5f}; g_nll_tr={:.5f}; g_nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
                            epoch, np.mean(np.asarray(d_err_tr)), np.mean(np.asarray(d_err_te)),
                            np.mean(np.asarray(g_err_tr)), np.mean(np.asarray(g_err_te)),
                            time_train, time_test))

                        d_optimizer, d_learning_rate, d_hit_max_lr, g_optimizer, g_learning_rate, g_hit_max_lr = \
                            self.update_lr(d_schedulers, d_optimizer, d_err_tr, d_hit_max_lr,
                                           g_schedulers, g_optimizer, g_err_tr, g_hit_max_lr)

                        metrics_dict = self.update_gan_metrics(epoch, metrics_dict, d_err_tr, d_err_te,
                                                               g_err_tr, g_err_te, d_learning_rate, g_learning_rate)

                        self.log_gan_loss(metrics_dict, train_epoch_stats_dict, test_epoch_stats_dict,
                                          d_tr_record, d_te_record, g_tr_record, g_te_record)

                        if epoch % self.config.wandb.sample_reporting_frequency == 0:
                            self.gan_reporting(epoch, train_loader, train_epoch_stats_dict, test_epoch_stats_dict,
                                               extra_test_dict=extra_test_epoch_stats_dict)

                        self.model_checkpointing(epoch, self.config, discriminator, generator, d_optimizer, g_optimizer, g_err_te, d_err_te, metrics_dict)

                        generator_converged, discriminator_converged = \
                            self.check_model_convergence(metrics_dict, self.config, epoch)

                        if (generator_converged and discriminator_converged) and (epoch > self.config.history + 2):
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

                self.post_run_evaluation(epoch, generator, discriminator, d_optimizer, g_optimizer, metrics_dict, train_loader, test_loader, extra_test_loader)

    def epoch(self, dataLoader=None, generator=None, discriminator=None, g_optimizer=None, d_optimizer=None, update_gradients=True,
              iteration_override=None, record_stats=False, epoch=None):

        if self.config.mode == 'gan':
            return self.gan_epoch(dataLoader, generator, discriminator, g_optimizer, d_optimizer, update_gradients,
                                  iteration_override, record_stats, epoch)
        elif self.config.mode == 'regression':
            return self.regression_epoch(dataLoader, generator, g_optimizer, update_gradients,
                                         iteration_override, record_stats)

    def regression_epoch(self, dataLoader=None, generator=None, g_optimizer=None, update_gradients=True,
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
            'generator packing target': [],
            'generator packing prediction': [],

        }

        for i, data in enumerate(tqdm.tqdm(dataLoader, miniters=int(len(dataLoader) / 25))):
            '''
            noise injection
            '''
            if self.config.generator.positional_noise > 0:
                data.pos += torch.randn_like(data.pos) * self.config.generator.positional_noise

            regression_losses_list, predictions, targets = self.regression_loss(generator, data)
            epoch_stats_dict['generator packing prediction'].append(predictions.cpu().detach().numpy())
            epoch_stats_dict['generator packing target'].append(targets.cpu().detach().numpy())

            g_loss = regression_losses_list.mean()
            g_err.append(g_loss.data.cpu().detach().numpy())  # average loss
            g_loss_record.extend(regression_losses_list.cpu().detach().numpy())  # loss distribution

            if update_gradients:
                g_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
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
            epoch_stats_dict['generator packing prediction'] = np.concatenate(epoch_stats_dict['generator packing prediction']) if epoch_stats_dict['generator packing prediction'] != [] else None
            epoch_stats_dict['generator packing target'] = np.concatenate(epoch_stats_dict['generator packing target']) if epoch_stats_dict['generator packing target'] != [] else None
            return g_err, g_loss_record, g_err, g_loss_record, epoch_stats_dict, total_time
        else:
            return g_err, g_loss_record, g_err, g_loss_record, total_time

    def gan_epoch(self, dataLoader=None, generator=None, discriminator=None, g_optimizer=None, d_optimizer=None, update_gradients=True,
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

        epoch_stats_dict = {
            'tracking features': [],
            'identifiers': [],
            'discriminator real score': [],
            'discriminator fake score': [],
            'generator adversarial loss': [],
            'generator per mol vdw loss': [],
            'generator h bond loss': [],
            'generator packing loss': [],
            'generator packing prediction': [],
            'generator packing target': [],
            'generator packing mae': [],
            'generator similarity loss': [],
            'generator combo loss': [],
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
            'real vdw penalty': [],
            'fake vdw penalty': [],
            'generated supercell examples': None,
        }

        rand_batch_ind = np.random.randint(0, len(dataLoader))
        self.n_samples_in_grad_buffer = 0

        if update_gradients:
            g_optimizer.zero_grad(set_to_none=True)
            d_optimizer.zero_grad(set_to_none=True)

        for i, data in enumerate(tqdm.tqdm(dataLoader, miniters=int(len(dataLoader) / 10))):

            '''
            train discriminator
            '''
            d_err, d_loss_record, epoch_stats_dict = \
                self.discriminator_step(discriminator, generator, epoch_stats_dict, data,
                                        d_optimizer, i, update_gradients, d_err, d_loss_record,
                                        epoch, last_batch=i == (len(dataLoader) - 1))

            '''
            train_generator
            '''
            g_err, g_loss_record, epoch_stats_dict = \
                self.generator_step(discriminator, generator, epoch_stats_dict, data,
                                    g_optimizer, i, update_gradients, g_err, g_loss_record,
                                    rand_batch_ind, last_batch=i == (len(dataLoader) - 1))

            '''
            record some stats
            '''
            if (len(epoch_stats_dict['generated cell parameters']) < i) and record_stats:  # make some samples for analysis if we have none so far from this step
                generated_samples = generator(len(data.y), z=None, conditions=data.to(self.config.device))
                epoch_stats_dict['generated cell parameters'].extend(generated_samples.cpu().detach().numpy())

            if record_stats:
                epoch_stats_dict['tracking features'].extend(data.tracking.cpu().detach().numpy())
                epoch_stats_dict['identifiers'].extend(data.csd_identifier)

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

            return d_err, d_loss_record, g_err, g_loss_record, epoch_stats_dict, total_time
        else:
            return d_err, d_loss_record, g_err, g_loss_record, total_time

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

    def pairwise_correlations_analysis(self, dataset_builder, config):
        '''
        check correlations in the data
        :param dataset_builder:
        :param config:
        :return:
        '''
        data = dataset_builder.datapoints
        keys = self.config.dataDims['lattice features']
        if self.config.generator.conditional_modelling:
            if (config.generator.conditioning_mode != 'graph model'):
                keys.extend(self.config.dataDims['conditional features'])
            else:
                data = np.asarray([(data[i].cell_params).detach().numpy() for i in range(len(data))])[:, 0, :]

        df = pd.DataFrame(data, columns=keys)
        correlations = df.corr()

        return correlations, keys

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

    def gan_reporting(self, epoch, train_loader, train_epoch_stats_dict, test_epoch_stats_dict, extra_test_dict=None):
        '''
        Do analysis and upload results to w&b
        '''
        if self.config.mode == 'gan':
            if test_epoch_stats_dict is not None:
                if test_epoch_stats_dict['generated cell parameters'] is not None:
                    self.cell_params_analysis(train_loader, test_epoch_stats_dict)

                if self.config.train_generator_packing or self.config.train_generator_vdw or self.config.train_generator_adversarially or self.config.train_generator_combo:
                    self.cell_generation_analysis(test_epoch_stats_dict)

                if self.config.train_discriminator_on_noise or self.config.train_discriminator_on_randn or self.config.train_discriminator_adversarially:
                    self.discriminator_analysis(test_epoch_stats_dict)

        elif self.config.mode == 'regression':
            self.log_regression_accuracy(train_epoch_stats_dict, test_epoch_stats_dict)

        if (extra_test_dict is not None) and (epoch % self.config.extra_test_period == 0):
            pass  # do reporting on an extra dataset
            # old reporting on nov_2022 discriminator development
            # from reporting.nov_22_discriminator import blind_test_analysis
            # blind_test_analysis(self.config, wandb, train_epoch_stats_dict, test_epoch_stats_dict, extra_test_dict)

        return None

    def discriminator_step(self, discriminator, generator, epoch_stats_dict, data, d_optimizer, i, update_gradients, d_err, d_loss_record, epoch, last_batch):
        if epoch % self.config.discriminator.training_period == 0:  # only train the discriminator every XX epochs
            if self.config.train_discriminator_adversarially or self.config.train_discriminator_on_noise or self.config.train_discriminator_on_randn:
                generated_samples_i, handedness, epoch_stats_dict = self.generate_discriminator_negatives(epoch_stats_dict, self.config, data, generator, i)

                score_on_real, score_on_fake, generated_samples, real_dist_dict, fake_dist_dict, real_vdw_score, fake_vdw_score \
                    = self.train_discriminator(generated_samples_i, discriminator, data, i, handedness)

                epoch_stats_dict['discriminator real score'].extend(score_on_real.cpu().detach().numpy())
                epoch_stats_dict['discriminator fake score'].extend(score_on_fake.cpu().detach().numpy())
                epoch_stats_dict['real vdw penalty'].extend(real_vdw_score.cpu().detach().numpy())
                epoch_stats_dict['fake vdw penalty'].extend(fake_vdw_score.cpu().detach().numpy())

                prediction = torch.cat((score_on_real, score_on_fake))
                target = torch.cat((torch.ones_like(score_on_real[:, 0]), torch.zeros_like(score_on_fake[:, 0])))
                d_losses = F.cross_entropy(prediction, target.long(), reduction='none')  # works much better

                # d_loss = d_losses.mean()
                d_loss = (d_losses / torch.diff(data.ptr.to(d_losses.device)).tile(2) * (data.num_nodes / data.num_graphs)).mean()  # norm losses according to graph size
                d_err.append(d_loss.data.cpu().detach().numpy())  # average overall loss
                d_loss_record.extend(d_losses.cpu().detach().numpy())  # overall loss distribution

                # if update_gradients:
                #     d_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                #     d_loss = d_loss / data.num_nodes  # normalize the loss by mean graph size
                #     d_loss.backward()  # back-propagation
                #     torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.config.gradient_norm_clip)  # gradient clipping
                #     d_optimizer.step()  # update parameters

                if update_gradients:
                    if self.config.accumulate_gradients:
                        d_loss = d_loss / self.config.accumulate_batch_size * self.config.final_batch_size
                    d_loss.backward()  # back-propagation
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), self.config.gradient_norm_clip)  # gradient clipping
                    if self.config.accumulate_gradients:
                        self.n_samples_in_grad_buffer += data.num_graphs
                        if (self.n_samples_in_grad_buffer > self.config.accumulate_batch_size) or last_batch:
                            d_optimizer.step()
                            d_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                            self.n_samples_in_grad_buffer = 0
                    else:
                        d_optimizer.step()  # update parameters
                        d_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes

                epoch_stats_dict['generated cell parameters'].extend(generated_samples_i.cpu().detach().numpy())
                epoch_stats_dict['final generated cell parameters'].extend(generated_samples)

            else:
                d_err.append(np.zeros(1))
                d_loss_record.extend(np.zeros(data.num_graphs))

        return d_err, d_loss_record, epoch_stats_dict

    def generator_step(self, discriminator, generator, epoch_stats_dict, data, g_optimizer, i, update_gradients, g_err, g_loss_record, rand_batch_ind, last_batch):
        if any((self.config.train_generator_packing, self.config.train_generator_adversarially, self.config.train_generator_vdw, self.config.train_generator_combo)):
            adversarial_score, generated_samples, packing_loss, packing_prediction, packing_target, \
            vdw_loss, generated_dist_dict, supercell_examples, similarity_penalty, h_bond_score, combo_score = \
                self.train_generator(generator, discriminator, data, i)

            epoch_stats_dict = self.log_supercell_examples(supercell_examples, i, rand_batch_ind, epoch_stats_dict)

            g_losses, epoch_stats_dict = self.aggregate_generator_losses(
                epoch_stats_dict, packing_loss, adversarial_score, adversarial_score,
                vdw_loss, similarity_penalty, packing_prediction, packing_target, h_bond_score, combo_score)

            # g_loss = g_losses.mean()
            g_loss = (g_losses / torch.diff(data.ptr) * (data.num_nodes / data.num_graphs)).mean()  # norm losses according to graph size
            g_err.append(g_loss.data.cpu().detach().numpy())  # average loss
            g_loss_record.extend(g_losses.cpu().detach().numpy())  # loss distribution
            epoch_stats_dict['generated cell parameters'].extend(generated_samples)

            if update_gradients:
                if self.config.accumulate_gradients:
                    g_loss = g_loss / self.config.accumulate_batch_size * self.config.final_batch_size
                g_loss.backward()  # back-propagation
                torch.nn.utils.clip_grad_norm_(generator.parameters(), self.config.gradient_norm_clip)  # gradient clipping
                if self.config.accumulate_gradients:
                    self.n_samples_in_grad_buffer += data.num_graphs
                    if (self.n_samples_in_grad_buffer > self.config.accumulate_batch_size) or last_batch:
                        g_optimizer.step()
                        g_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes
                        self.n_samples_in_grad_buffer = 0
                else:
                    g_optimizer.step()  # update parameters
                    g_optimizer.zero_grad(set_to_none=True)  # reset gradients from previous passes

        else:
            g_err.append(np.zeros(1))
            g_loss_record.extend(np.zeros(data.num_graphs))

        return g_err, g_loss_record, epoch_stats_dict

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

    def train_generator(self, generator, discriminator, data, i):
        '''
        train the generator
        '''

        '''
        conformer orentation setting
        '''
        if self.config.generator.canonical_conformer_orientation == 'standardized':
            data = align_crystaldata_to_principal_axes(data)
        elif self.config.generator.canonical_conformer_orientation == 'random':
            data = random_crystaldata_alignment(data)

        '''
        noise injection
        '''
        if self.config.generator.positional_noise > 0:
            data.pos += torch.randn_like(data.pos) * self.config.generator.positional_noise

        '''
        update symmetry information
        '''
        if self.config.generate_sgs is not None:
            override_sg_ind = self.supercell_builder.symmetries_dict['sg_ind_dict'][self.config.generate_sgs]
            sym_ops_list = [torch.Tensor(self.supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(data.x.device) for i in range(data.num_graphs)]
            data = override_sg_info(self.config.generate_sgs, self.config.dataDims, data, self.supercell_builder.symmetries_dict, sym_ops_list)  # todo update the way we handle this

        '''
        generate samples
        '''
        [[generated_samples, latent], prior, condition] = generator.forward(
            n_samples=data.num_graphs, conditions=data.to(self.config.device),
            return_latent=True, return_condition=True, return_prior=True)

        '''
        build supercells
        '''
        if self.config.train_generator_adversarially or self.config.train_generator_vdw or self.config.train_generator_combo:
            supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
                data, generated_samples, self.config.supercell_size,
                self.config.discriminator.graph_convolution_cutoff,
                override_sg=self.config.generate_sgs,
                align_molecules=False,  # molecules are either random on purpose, or pre-aligned with set handedness
            )

            data.cell_params = supercell_data.cell_params
        else:
            supercell_data, generated_cell_volumes = None, None

        '''
        #evaluate losses
        
        # look at cells
        from ase.visualize import view
        mols = [ase_mol_from_crystaldata(supercell_data, i, exclusion_level='convolve with', highlight_aux=True) for i in range(min(10, supercell_data.num_graphs))]
        view(mols)
        '''
        similarity_penalty = self.compute_similarity_penalty(generated_samples, supercell_data.sg_ind, prior)
        discriminator_score, dist_dict = self.score_adversarially(supercell_data, discriminator)
        h_bond_score = self.compute_h_bond_score(supercell_data)
        total_vdw_overlap, normed_vdw_overlap = self.get_vdw_penalty(dist_dict, supercell_data.num_graphs, data)
        vdw_overlap = total_vdw_overlap / torch.diff(data.ptr) # norm by molecule size
        packing_loss, packing_prediction, packing_target, = \
            self.cell_density_loss(data, generated_samples, precomputed_volumes=generated_cell_volumes)

        if normed_vdw_overlap is not None:
            #packing_range = [0.55, 0.75]
            #outside_reasonable_packing_loss = F.smooth_l1_loss(packing_prediction, packing_target, beta=.1, reduction='none')
            combo_score = 1/(1 + (normed_vdw_overlap**2 + packing_loss**2)) #torch.exp(-(normed_vdw_overlap + outside_reasonable_packing_loss))
        else:
            combo_score = None

        return discriminator_score, generated_samples.cpu().detach().numpy(), \
               packing_loss, packing_prediction.cpu().detach().numpy(), \
               packing_target.cpu().detach().numpy(), \
               vdw_overlap, dist_dict, \
               supercell_data, similarity_penalty, h_bond_score,\
               combo_score

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
            packing_loss = torch.log(1 + F.smooth_l1_loss(standardized_gen_packing_coeffs, standardized_csd_packing_coeffs, reduction='none')) ** 2  # log(1+loss) is a soft rescaling to avoid gigantic losses
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
        self.sg_ind_dict = {thing[14:]: ind + self.config.dataDims['num atomwise features'] for ind, thing in enumerate(self.config.dataDims['molecule features']) if 'sg is' in thing}
        self.crysys_ind_dict = {thing[18:]: ind + self.config.dataDims['num atomwise features'] for ind, thing in enumerate(self.config.dataDims['molecule features']) if 'crystal system is' in thing}

        '''
        add symmetry element indices to symmetry dict
        '''
        self.sym_info['sg_ind_dict'] = self.sg_ind_dict
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

        return dataset_builder

    def model_sampling(self):
        '''
        Stun MC annealing on a pretrained discriminator / generator
        '''
        with wandb.init(config=self.config, project=self.config.wandb.project_name, entity=self.config.wandb.username, tags=[self.config.wandb.experiment_tag]):
            dataset_builder = self.training_prep()
            del dataset_builder
            generator, discriminator, g_optimizer, g_schedulers, \
            d_optimizer, d_schedulers, params1, params2 \
                = self.init_gan(self.config, self.config.dataDims)

            self.config.final_batch_size = self.config.min_batch_size

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

            n_samples = 200  # self.config.min_batch_size
            single_mol_data = extra_test_loader.dataset[0]
            collater = Collater(None, None)
            single_mol_data = collater([single_mol_data for n in range(n_samples)])

            # init_samples = self.randn_generator.forward(n_samples, single_mol_data)
            init_samples = generator.forward(n_samples=single_mol_data.num_graphs,
                                             conditions=single_mol_data.cuda()
                                             )

            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                smc_sampler = mcmcSampler(
                    gammas=np.logspace(-4, 0, n_samples),
                    seedInd=0,
                    acceptance_mode=None,
                    debug=True,
                    init_temp=.001,
                    generator=generator,
                    move_size=self.config.sample_move_size,
                    supercell_size=self.config.supercell_size,
                    graph_convolution_cutoff=self.config.discriminator.graph_convolution_cutoff,
                    vdw_radii=self.vdw_radii,
                    preset_minimum=None,
                    spacegroup_to_search=self.config.generate_sgs,
                    new_minimum_patience=25,
                    reset_patience=50,
                )

                '''
                run sampling
                '''
                sampling_dict = smc_sampler(discriminator, self.supercell_builder,
                                            single_mol_data, init_samples.cpu().detach().numpy(), self.config.sample_steps)

            # np.save(f'../sampling_output_run_{self.config.run_num}', sampling_dict)

            # best_inds = np.argsort(sampling_dict['scores'].flatten())
            # best_samples = sampling_dict['samples'].reshape(12, n_samples * num_iters)[:, best_inds[:n_samples]].T
            #
            # best_supercells, _, _ = self.supercell_builder.build_supercells(single_mol_data.clone(), torch.Tensor(best_samples).cuda(),
            #                                                                 supercell_size=1, graph_convolution_cutoff=7)
            #
            # best_rdfs, rr = crystal_rdf(best_supercells, rrange=[0, 10], bins=100, intermolecular=True)
            self.report_sampling(sampling_dict)

        return sampling_dict

    def generate_discriminator_negatives(self, epoch_stats_dict, config, data, generator, i):
        n_generators = sum([config.train_discriminator_adversarially, self.config.train_discriminator_on_noise, self.config.train_discriminator_on_randn])
        gen_random_number = np.random.uniform(0, 1, 1)
        gen_randn_range = np.linspace(0, 1, n_generators + 1)

        if self.config.train_discriminator_adversarially:
            ii = i % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:  # randomly sample which generator to use at each iteration
                generated_samples_i = generator.forward(n_samples=data.num_graphs, conditions=data.to(config.device))
                handedness = None
                epoch_stats_dict['generator sample source'].extend(np.zeros(len(generated_samples_i)))

        if self.config.train_discriminator_on_randn:
            ii = (i + 1) % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:
                generated_samples_i = self.randn_generator.forward(data.num_graphs, data).to(config.device)
                handedness = None
                epoch_stats_dict['generator sample source'].extend(np.ones(len(generated_samples_i)))

        if self.config.train_discriminator_on_noise:
            ii = (i + 2) % n_generators
            if gen_randn_range[ii] < gen_random_number < gen_randn_range[ii + 1]:
                generated_samples_ii = (data.cell_params - torch.Tensor(self.config.dataDims['lattice means'])) / torch.Tensor(self.config.dataDims['lattice stds'])  # standardize
                if self.config.generator_noise_level == -1:
                    distortion = torch.randn_like(generated_samples_ii) * torch.logspace(-2.5, -0.5, len(generated_samples_ii)).to(generated_samples_ii.device)[:, None]  # wider range for evaluation mode
                else:
                    distortion = torch.randn_like(generated_samples_ii) * self.config.generator_noise_level
                generated_samples_i = (generated_samples_ii + distortion).to(config.device)  # add jitter and return in standardized basis
                handedness = data.asym_unit_handedness
                epoch_stats_dict['generator sample source'].extend(np.ones(len(generated_samples_i)) * 2)
                epoch_stats_dict['distortion level'].extend(torch.linalg.norm(distortion, axis=-1).cpu().detach().numpy())

        return generated_samples_i, handedness, epoch_stats_dict

    def log_regression_accuracy(self, train_epoch_stats_dict, test_epoch_stats_dict):
        target_mean = self.config.dataDims['target mean']
        target_std = self.config.dataDims['target std']

        target = np.asarray(test_epoch_stats_dict['generator packing target'])
        prediction = np.asarray(test_epoch_stats_dict['generator packing prediction'])
        orig_target = target * target_std + target_mean
        orig_prediction = prediction * target_std + target_mean

        volume_ind = self.config.dataDims['tracking features dict'].index('molecule volume')
        mass_ind = self.config.dataDims['tracking features dict'].index('molecule mass')
        molwise_density = test_epoch_stats_dict['tracking features'][:, mass_ind] / test_epoch_stats_dict['tracking features'][:, volume_ind]
        target_density = molwise_density * orig_target * 1.66  # conversion from amu/A^3 to g/mL
        predicted_density = molwise_density * orig_prediction * 1.66

        if train_epoch_stats_dict is not None:
            train_target = np.asarray(train_epoch_stats_dict['generator packing target'])
            train_prediction = np.asarray(train_epoch_stats_dict['generator packing prediction'])
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
            g_loss_correlations = np.zeros(self.config.dataDims['num tracking features'])
            features = []
            for i in range(self.config.dataDims['num tracking features']):  # not that interesting
                features.append(self.config.dataDims['tracking features dict'][i])
                g_loss_correlations[i] = np.corrcoef(np.abs((orig_target - orig_prediction) / np.abs(orig_target)), tracking_features[:, i], rowvar=False)[0, 1]

            g_sort_inds = np.argsort(g_loss_correlations)
            g_loss_correlations = g_loss_correlations[g_sort_inds]

            fig = go.Figure(go.Bar(
                y=[self.config.dataDims['tracking features dict'][i] for i in range(self.config.dataDims['num tracking features'])],
                x=[g_loss_correlations[i] for i in range(self.config.dataDims['num tracking features'])],
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
                nice_scoring_plots(self.config)

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
        if self.config.auto_batch_sizing:
            if (train_loader.batch_size < len(train_loader.dataset)) and (train_loader.batch_size < self.config.max_batch_size):  # if the batch is smaller than the dataset
                increment = max(4, int(train_loader.batch_size * self.config.batch_growth_increment))  # increment batch size
                train_loader = update_dataloader_batch_size(train_loader, train_loader.batch_size + increment)
                test_loader = update_dataloader_batch_size(test_loader, test_loader.batch_size + increment)
                if extra_test_loader is not None:
                    extra_test_loader = update_dataloader_batch_size(extra_test_loader, extra_test_loader.batch_size + increment)
                print(f'Batch size incremented to {train_loader.batch_size}')
        wandb.log({'batch size': train_loader.batch_size})
        self.config.final_batch_size = train_loader.batch_size
        return train_loader, test_loader, extra_test_loader

    def check_model_convergence(self, metrics_dict, config, epoch):
        generator_convergence = checkConvergence(metrics_dict['generator test loss'], self.config.history, self.config.generator.convergence_eps)
        discriminator_convergence = checkConvergence(metrics_dict['discriminator test loss'], self.config.history, self.config.discriminator.convergence_eps)
        if generator_convergence:
            print('generator converged!')
        if discriminator_convergence:
            print('discriminator converged!')

        return generator_convergence, discriminator_convergence

    def model_checkpointing(self, epoch, config, discriminator, generator, d_optimizer, g_optimizer, g_err_te, d_err_te, metrics_dict):
        if (epoch > 0) and (epoch % 5 == 0):
            if config.machine == 'cluster':  # every 5 epochs, save a checkpoint
                # saving early-stopping checkpoint
                save_checkpoint(epoch, discriminator, d_optimizer, self.config.discriminator.__dict__, 'discriminator_' + str(config.run_num) + f'_epoch_{epoch}')
                save_checkpoint(epoch, generator, g_optimizer, self.config.generator.__dict__, 'generator_' + str(config.run_num) + f'_epoch_{epoch}')

                if np.average(d_err_te) < np.amin(metrics_dict['discriminator test loss'][:-1]):
                    print("Saving discriminator checkpoint")
                    save_checkpoint(epoch, discriminator, d_optimizer, self.config.discriminator.__dict__, 'discriminator_' + str(config.run_num))
                if np.average(g_err_te) < np.amin(metrics_dict['generator test loss'][:-1]):
                    print("Saving generator checkpoint")
                    save_checkpoint(epoch, generator, g_optimizer, self.config.generator.__dict__, 'generator_' + str(config.run_num))

        return None

    def update_lr(self, d_schedulers, d_optimizer, d_err_tr, d_hit_max_lr,
                  g_schedulers, g_optimizer, g_err_tr, g_hit_max_lr):
        # update learning rate
        d_optimizer, d_lr = set_lr(d_schedulers, d_optimizer, self.config.discriminator.lr_schedule,
                                   self.config.discriminator.learning_rate, self.config.discriminator.max_lr, d_err_tr, d_hit_max_lr)
        d_learning_rate = d_optimizer.param_groups[0]['lr']
        if d_learning_rate >= self.config.discriminator.max_lr: d_hit_max_lr = True

        # update learning rate
        g_optimizer, g_lr = set_lr(g_schedulers, g_optimizer, self.config.generator.lr_schedule,
                                   self.config.generator.learning_rate, self.config.generator.max_lr, g_err_tr, g_hit_max_lr)
        g_learning_rate = g_optimizer.param_groups[0]['lr']
        if g_learning_rate >= self.config.generator.max_lr: g_hit_max_lr = True

        print(f"Learning rates are d={d_lr:.5f}, g={g_lr:.5f}")

        return d_optimizer, d_learning_rate, d_hit_max_lr, g_optimizer, g_learning_rate, g_hit_max_lr

    def report_sampling(self, sampling_dict):
        '''
        score
        stun
        acceptance rate
        temperature
        overall distribution
        CLUSTERING
        '''
        # todo fix this up

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
        fig = make_subplots(cols=3, rows=2, subplot_titles=['Best Model Score', 'Average STUN Score', 'Min vdw Score', 'Mean Acceptance Rate', 'Mean Temperature'])
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=-np.amin(sampling_dict['scores'], axis=0)), col=1, row=1)
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.mean(sampling_dict['stun score'], axis=0)), col=2, row=1)
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=-np.amin(sampling_dict['vdw penalties'], axis=0)), col=3, row=1)
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.mean(sampling_dict['acceptance ratio'], axis=0)), col=1, row=2)
        fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.mean(np.log10(sampling_dict['temperature']), axis=0)), col=2, row=2)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[-1, 1], row=1, col=2)
        fig.update_yaxes(range=[-2, 0], row=1, col=3)
        fig.update_yaxes(range=[-16, 16], row=1, col=1)
        fig.layout.margin = layout.margin
        # fig.write_image('../paper1_figs/sampling_telemetry_summary.png')
        wandb.log({'Sampling Telemetry Summary': fig})
        if self.config.machine == 'local':
            import plotly.io as pio
            pio.renderers.default = 'browser'
            fig.show()

        # crystal_identifier = test_epoch_stats_dict['identifiers'][sample_ind]
        #
        # '''
        # plt.clf()
        # import matplotlib as mpl
        # for i,file in enumerate(files):
        #     plt.subplot(2,3,i+1)
        #     sampling_dict = np.load(file, allow_pickle=True).item()
        #     plt.hist2d(x=np.nan_to_num(-sampling_dict['scores'].flatten()),
        #            y=np.nan_to_num(-sampling_dict['vdw penalties'].flatten()),
        #            bins = 50,
        #            range=[[-16,16],[-5,0.2]],
        #            norm=mpl.colors.LogNorm())
        #
        # '''
        #
        # fig = go.Figure()
        # viridis = px.colors.sequential.Viridis
        # fig.add_trace(go.Histogram2d(x=-sampling_dict['scores'].flatten(),
        #                              y=-sampling_dict['vdw penalties'].flatten(),
        #                              xbins=dict(start=-16, end=16, size=32 / 50),
        #                              ybins=dict(start=-2, end=0.1, size=1 / 50),
        #                              showscale=False,
        #                              colorscale=[
        #                                  [0, viridis[0]],
        #                                  [1. / 1000000, viridis[2]],
        #                                  [1. / 10000, viridis[4]],
        #                                  [1. / 100, viridis[7]],
        #                                  [1., viridis[9]],
        #                              ],
        #                              colorbar=dict(
        #                                  tick0=0,
        #                                  tickmode='array',
        #                                  tickvals=[0, 1000, 10000]
        #                              )))
        #
        # fig.add_trace(go.Scattergl(x=softmax_and_score(test_epoch_stats_dict['discriminator real score'][sample_ind]),
        #                            y=test_epoch_stats_dict['real vdw penalty'][sample_ind][None],
        #                            mode='markers',
        #                            showlegend=False,
        #                            marker=dict(
        #                                symbol='circle',
        #                                color='white',
        #                                size=25,
        #                                line=dict(width=1, color='black')
        #                            )))
        #
        # fig.layout.margin = layout.margin
        # fig.update_layout(title=crystal_identifier)
        # # fig.write_image('../paper1_figs/sampling_scores.png')
        # wandb.log({'Sampling Scores': fig})
        # if self.config.machine == 'local':
        #     import plotly.io as pio
        #     pio.renderers.default = 'browser'
        #     fig.show()

        '''
        full telemetry
        '''
        # fig = make_subplots(cols=3, rows=2, subplot_titles=['Model Score', 'STUN Score','vdw Score', 'Acceptance Rate', 'Temperature'])
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

        # supercell_data, generated_cell_volumes, overlaps_list = \
        #     self.supercell_builder.build_supercells(single_mol_data.clone().to(init_samples.device), init_samples,
        #                                             self.config.supercell_size, self.config.discriminator.graph_convolution_cutoff,
        #                                             override_sg = self.config.generate_sgs)
        #
        # rdfs_list, rr, rdfs_dict_list = crystal_rdf(supercell_data, rrange=[0, self.config.discriminator.graph_convolution_cutoff],
        #                                             bins=200, mode='intermolecular', raw_density=True, atomwise=True)
        # dists = torch.zeros((supercell_data.num_graphs, supercell_data.num_graphs))
        # for i in range(supercell_data.num_graphs):
        #     for j in range(i, supercell_data.num_graphs):
        #         dists[i, j] = compute_rdf_distance(rdfs_list[i], rdfs_list[j], rr)
        #
        # rdfs_list2, rr, rdfs_dict_list = crystal_rdf(supercell_data, rrange=[0, self.config.discriminator.graph_convolution_cutoff],
        #                                              bins=200, mode='intermolecular', raw_density=True, atomwise=False, elementwise=True)
        # dists2 = torch.zeros((supercell_data.num_graphs, supercell_data.num_graphs))
        # for i in range(supercell_data.num_graphs):
        #     for j in range(i, supercell_data.num_graphs):
        #         dists2[i, j] = compute_rdf_distance(rdfs_list2[i], rdfs_list2[j], rr)

        # agglomerative clustering and dendrogram
        # setting distance_threshold=0 ensures we compute the full tree.
        # from sklear.cluster import AgglomerativeClustering
        # plt.clf()
        # model = AgglomerativeClustering(distance_threshold=0, linkage="average", affinity='precomputed', n_clusters=None)
        # model = model.fit(dists.cpu().detach().numpy())
        #
        # # plot the top three levels of the dendrogram
        # plot_dendrogram(model, truncate_mode="level", p=3)
        # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        # plt.show()

        return None

    def post_run_evaluation(self, epoch, generator, discriminator, d_optimizer, g_optimizer, metrics_dict, train_loader, test_loader, extra_test_loader):
        '''
        run post-training evaluation
        '''
        # reload best test
        g_path = f'../models/generator_{self.config.run_num}'
        d_path = f'../models/discriminator_{self.config.run_num}'
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

        # rerun test inference
        with torch.no_grad():
            d_err_te, d_te_record, g_err_te, g_te_record, test_epoch_stats_dict, time_test = \
                self.epoch(dataLoader=test_loader, generator=generator, discriminator=discriminator,
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
            np.zeros(10), d_err_te,
            np.zeros(10), g_err_te,
            d_optimizer.defaults['lr'], g_optimizer.defaults['lr'])

        self.log_gan_loss(metrics_dict, None, test_epoch_stats_dict,
                          None, d_te_record, None, g_te_record)

        self.gan_reporting(epoch, train_loader, None, test_epoch_stats_dict,
                           extra_test_dict=extra_test_epoch_stats_dict)

    def compute_similarity_penalty(self, generated_samples, sg_ind, prior):
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

    def get_vdw_penalty(self, dist_dict=None, num_graphs=None, data = None):
        if dist_dict is not None:  # supercell_data is not None: # do vdw computation even if we don't need it
            return vdw_overlap(self.vdw_radii, dists=dist_dict['intermolecular dist'],
                               atomic_numbers=dist_dict['intermolecular dist atoms'],
                               batch_numbers=dist_dict['intermolecular dist batch'],
                               num_graphs=num_graphs,
                               graph_sizes = torch.diff(data.ptr),
                               return_normed = True)
        else:
            return None, None

    def aggregate_generator_losses(self, epoch_stats_dict, packing_loss, adversarial_score, adversarial_loss, vdw_loss, similarity_penalty, packing_prediction, packing_target, h_bond_score, combo_score):
        g_losses_list = []
        if self.config.train_generator_packing:
            g_losses_list.append(packing_loss.float())

        if packing_loss is not None:
            epoch_stats_dict['generator packing loss'].append(packing_loss.cpu().detach().numpy())
            epoch_stats_dict['generator packing prediction'].append(packing_prediction)
            epoch_stats_dict['generator packing target'].append(packing_target)
            epoch_stats_dict['generator packing mae'].append(np.abs(packing_prediction - packing_target) / packing_target)

        if adversarial_score is not None:
            softmax_adversarial_score = F.softmax(adversarial_score, dim=1)[:, 1]  # modified minimax
            adversarial_loss = -torch.log(softmax_adversarial_score)  # modified minimax
            epoch_stats_dict['generator adversarial loss'].append(adversarial_loss.cpu().detach().numpy())
        if self.config.train_generator_adversarially:
            g_losses_list.append(adversarial_loss)

        if self.config.train_generator_vdw:
            if self.config.vdw_loss_rescaling == 'log':
                vdw_loss_f = torch.log(1 + vdw_loss) ** 2  # soft rescaling
            elif self.config.vdw_loss_rescaling is None:
                vdw_loss_f = vdw_loss
            elif self.config.vdw_loss_rescaling == 'mse':
                vdw_loss_f = vdw_loss ** 4 # todo GET RID OF THIS
            g_losses_list.append(vdw_loss_f)
        if vdw_loss is not None:
            epoch_stats_dict['generator per mol vdw loss'].append(vdw_loss.cpu().detach().numpy())

        if self.config.train_generator_h_bond:
            g_losses_list.append(h_bond_score)
        if vdw_loss is not None:
            epoch_stats_dict['generator h bond loss'].append(h_bond_score.cpu().detach().numpy())

        if self.config.generator_similarity_penalty != 0:
            if similarity_penalty is not None:
                g_losses_list.append(self.config.generator_similarity_penalty * similarity_penalty)
            else:
                print('similarity penalty was none')
        if similarity_penalty is not None:
            epoch_stats_dict['generator similarity loss'].append(similarity_penalty.cpu().detach().numpy())

        if self.config.train_generator_combo:
            g_losses_list.append(-combo_score)
        if combo_score is not None:
            epoch_stats_dict['generator combo loss'].append(1-combo_score.cpu().detach().numpy())

        g_losses = torch.sum(torch.stack(g_losses_list), dim=0)

        return g_losses, epoch_stats_dict

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
            if supercell_examples.num_graphs > 100:  # todo find a way to take only the few that we need - maybe using the Collater
                print('WARNING. Saving over 100 supercells for analysis')
        epoch_stats_dict['final generated cell parameters'].extend(supercell_examples.cell_params.cpu().detach().numpy())
        del supercell_examples
        return epoch_stats_dict

    def compute_h_bond_score(self, supercell_data=None):
        if supercell_data is not None:  # supercell_data is not None: # do vdw computation even if we don't need it
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
            z = gaussian_kde(xy)(xy)

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
                                         index=i, exclusion_level='ref only')
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
                        graph_sizes = supercell_examples.tracking[:,self.config.dataDims['tracking features dict'].index('molecule num atoms')])
        vdw_loss /= supercell_examples.tracking[:,self.config.dataDims['tracking features dict'].index('molecule num atoms')]

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
            fig.add_trace(go.Violin(x=pens[pens!=0], side='positive', orientation='h',
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
        fig.update_layout(width=800, height=600, font=dict(size=12))
        fig.layout.margin = layout.margin
        fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
        fig.update_layout(xaxis_title='Nonzero vdW overlaps', yaxis_title='packing prediction')

        # fig.write_image('../paper1_figs/sampling_scores.png')
        wandb.log({'Generated Sample Analysis': fig})
        # fig.show()

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
            fig.add_trace(go.Violin(x=-np.log(vdw_penalty_dict[label] + 1e-6), name=legend_label, line_color=plot_color_dict[label],
                                    side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth2, points=False),
                          row=1, col=2)

        all_vdws = np.concatenate((vdw_penalty_dict['CSD'], vdw_penalty_dict['Gaussian'], vdw_penalty_dict['Distorted'], vdw_penalty_dict['Generator']))
        all_scores_i = np.concatenate((scores_dict['CSD'], scores_dict['Gaussian'], scores_dict['Distorted'], scores_dict['Generator']))

        rrange = np.logspace(3, 0, len(viridis))
        cscale = [[1 / rrange[i], viridis[i]] for i in range(len(rrange))]
        cscale[0][0] = 0

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
                                 ))
        fig.update_layout(barmode='relative')
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

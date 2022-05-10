import wandb
from utils import *
import glob
from model_utils import *
from dataset_management.CSD_data_manager import Miner
from torch import backends
from dataset_utils import BuildDataset, get_dataloaders
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler


class Predictor():
    def __init__(self, config):
        self.config = config
        self.setup()

    def setup(self):
        '''
        setup working directory
        move to relevant directory
        :return:
        '''

        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

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
                miner.load_for_modelling()
                print('Initializing dataset took {} seconds'.format(int(time.time() - t0)))
        else:
            if self.config.explicit_run_enumeration:
                # move to working dir
                self.workDir = self.config.workdir + '/' + 'run%d' % self.config.run_num
                os.chdir(self.workDir)
                self.groupLabels = list(np.load('group_labels.npy', allow_pickle=True))
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

    def prep_metrics(self):
        metrics_list = ['train loss', 'test loss', 'epoch', 'learning rate']
        metrics_dict = initialize_metrics_dict(metrics_list)

        return metrics_dict

    def update_metrics(self, epoch, metrics_dict, err_tr, err_te, lr):
        metrics_dict['train loss'].append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
        metrics_dict['test loss'].append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
        metrics_dict['epoch'].append(epoch)
        metrics_dict['learning rate'].append(lr)

        return metrics_dict

    def init_model(self, config, dataDims, print_status=True):
        '''
        Initialize model and optimizer
        :return:
        '''
        # init model
        if config.mode == 'joint modelling':
            model = FlowModel(config, dataDims)
        elif 'molecule' in config.mode:
            model = molecule_graph_model(config, dataDims)
        elif 'cell' in config.mode:
            model = molecular_crystal_graph_model(config, dataDims)

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


    def get_batch_size(self, dataset, config):
        finished = 0
        batch_size = config.initial_batch_size.real
        batch_reduction_factor = config.auto_batch_reduction

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
        with wandb.init(config=self.config, project=self.config.project_name, entity=self.config.wandb_username, tags=self.config.experiment_tag):
            config = wandb.config
            print(config)

            # dataset
            dataset_builder = BuildDataset(config)
            config.dataDims = dataset_builder.get_dimension()
            self.dataDims = dataset_builder.get_dimension()
            if 'classification' in config.mode:
                self.groupLabels =dataset_builder.group_labels
            if config.mode == 'joint modelling':
                self.lattice_features = dataset_builder.lattice_keys
                self.dataDims = dataset_builder.get_dimension()
                self.n_crystal_dims = self.dataDims['n crystal features']
                if config.conditional_modelling:
                    self.n_conditional_features = self.dataDims['n conditional features']
                else:
                    self.n_conditional_features = 0


            # get batch size
            if config.auto_batch_sizing:
                train_loader, test_loader, config.final_batch_size = self.get_batch_size(dataset_builder, config)
            else:
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.initial_batch_size

            # model, optimizer, schedulers
            model, optimizer, schedulers, n_params = self.init_model(config, self.dataDims)

            print("Training batch size set to {}".format(config.final_batch_size))

            # cuda
            if config.device.lower() == 'cuda':
                torch.backends.cudnn.benchmark = True
                # model = torch.nn.DataParallel(model) # send to multiple GPUs - not always working with wandb
                model.cuda()

            wandb.watch(model, log_graph=True)

            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_metrics()

            # training loop
            hit_max_lr, converged, epoch = False, False, 0
            while (epoch < config.max_epochs) and not converged:
                # very cool
                print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                # very cool
                print("Starting Epoch {}".format(epoch))  # index from 0, very cool

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
                self.log_accuracy(epoch, dataset_builder, train_loader, test_loader,
                                  metrics_dict, tr_record, te_record, epoch_stats_dict,
                                  config, model, log_figures=config.log_figures)

                # check for convergence
                if checkConvergence(config, metrics_dict['test loss']) and (epoch > config.history + 2):
                    config.finished = True
                    self.log_accuracy(epoch, dataset_builder, train_loader, test_loader,
                                      metrics_dict, tr_record, te_record, epoch_stats_dict,
                                      config, model, log_figures=True) # always log figures at end of run
                    break

                epoch += 1

            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU

    def model_epoch(self, config, dataLoader=None, model=None, optimizer=None, update_gradients=True,
                    iteration_override=None, record_stats=False):
        t0 = time.time()
        if update_gradients:
            model.train(True)
        else:
            model.eval()

        err = []
        loss_record = []
        epoch_stats_dict = {
            'predictions': [],
            'targets': [],
            'tracking features': [],
        }

        for i, data in enumerate(dataLoader):

            if config.device.lower() == 'cuda':
                data = data.cuda()

            losses, predictions = self.get_loss(model, data, config)

            loss = losses.mean()
            err.append(loss.data.cpu())  # average loss
            loss_record.extend(losses.cpu().detach().numpy())  # loss distribution

            if record_stats:
                epoch_stats_dict['predictions'].extend(predictions)
                epoch_stats_dict['targets'].extend(data.y[0])
                epoch_stats_dict['tracking features'].extend(data.y[2])

            if update_gradients:
                optimizer.zero_grad()  # reset gradients from previous passes
                loss.backward()  # back-propagation
                optimizer.step()  # update parameters

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0
        if record_stats:
            return err, loss_record, epoch_stats_dict, total_time
        else:
            return err, loss_record, total_time

    def get_loss(self, model, data, config):
        if (config.mode == 'single molecule regression') or (config.mode == 'cell regression'):
            pred = model(data)
            targets = data.y[0]
            if targets.ndim > 1:
                targets = targets[:, 0]

            losses = F.smooth_l1_loss(pred, targets.float(), reduction='none')
            return losses, pred.cpu().detach().numpy()

        elif (config.mode == 'single molecule classification') or (config.mode == 'cell classification'):
            output = model(data)  # reshape output from flat filters to channels * filters per channel
            targets = data.y[0]

            if targets.ndim > 1:
                targets = targets[:, 0]

            losses = F.cross_entropy(output, targets.long(), reduction='none')
            probs = F.softmax(output, dim=1).cpu().detach().numpy()

            return losses, probs

        elif config.mode == 'joint modelling':
            zs, prior_logprob, log_det = model(data)
            logprob = prior_logprob + log_det

            return -(logprob), prior_logprob.cpu().detach().numpy()



    def pairwise_correlations_analysis(self, dataset_builder, config):
        '''
        check pairwise correlations in the data
        :param dataset_builder:
        :param config:
        :return:
        '''
        data = dataset_builder.datapoints
        keys = self.dataDims['crystal features']
        if config.conditional_modelling:
            if (config.conditioning_mode != 'graph model'):
                keys.extend(self.dataDims['conditional features'])
            else:
                data = np.asarray([(data[i].y[0]).detach().numpy() for i in range(len(data))])[:, 0, :]

        df = pd.DataFrame(data, columns=keys)
        correlations = df.corr()

        return correlations, keys

    def check_inversion_quality(self, model, test_loader, config):
        # check for quality of the inversion
        if self.n_conditional_features > 0:
            test_conditions = next(iter(test_loader)).to(config.device)
            if config.conditioning_mode == 'graph model':
                test_sample = model.sample(test_conditions.num_graphs, conditions=test_conditions)
            else:
                test_sample = model.sample(len(test_conditions), conditions=test_conditions[:, -self.n_conditional_features:])
            if config.conditioning_mode == 'graph model':
                test_conditions.y[0] = test_sample
                zs, _, _ = model.forward(test_conditions)
                test_conditions.y[0] = zs
                test_sample2, _ = model.backward(test_conditions)
            else:
                zs, _, _ = model.forward(torch.cat((test_sample, test_conditions[:, -self.n_conditional_features:].to(test_sample.device)), dim=-1))
                test_sample2, _ = model.backward(torch.cat((zs, test_conditions[:, -self.n_conditional_features:].to(zs.device)), dim=-1))
        else:
            test_sample = model.sample(self.sampling_batch_size)
            zs, _, _ = model.forward(test_sample)
            test_sample2, _ = model.backward(zs)
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
            if config.conditioning_mode == 'graph model':
                generation_conditions.append(data.to(model.device))
                targets.extend(generation_conditions[-1].y[0].cpu().detach().numpy())
            else:
                batch_placeholder = data
                if config.conditional_modelling:
                    generation_conditions.append(batch_placeholder[:, -self.n_conditional_features:].float())
                    targets.extend(batch_placeholder[:, :-self.n_conditional_features].cpu().detach().numpy())
                else:
                    generation_conditions.append(batch_placeholder.float())
                    targets.extend(batch_placeholder.cpu().detach().numpy())

        train_data = train_loader.dataset
        if (self.n_conditional_features > 0):
            if (config.conditioning_mode != 'graph model'):
                train_data = np.asarray(train_data)[:, :-self.n_conditional_features]
            else:
                train_data = np.asarray([(train_data[i].y[0]).detach().numpy() for i in range(len(train_data))])[:, 0, :]
        else:
            train_data = np.asarray(train_data)

        del generation_conditions
        return np.asarray(targets), train_data

    def sample_nf(self, n_repeats, config, model, test_loader):
        nf_samples = [[] for _ in range(n_repeats)]
        print('Sampling from NF')
        for j in tqdm.tqdm(range(n_repeats)):
            for i, data in enumerate(test_loader):
                if config.conditional_modelling:
                    if config.device == 'cuda':
                        data = data.cuda()
                    if config.conditioning_mode == 'graph model':
                        minibatch_size = data.num_graphs
                    else:
                        minibatch_size = len(data)
                        data = data[:, -self.n_conditional_features:]
                    nf_samples[j].extend(model.sample(
                        minibatch_size,
                        conditions=data
                    ).cpu().detach().numpy())
                else:
                    nf_samples[j].extend(model.sample(
                        len(data)
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
                for j in range(sample.shape[1]):
                    if self.n_conditional_features > 0:
                        if config.conditioning_mode == 'graph model':
                            data.y[0] = sample[:, j]
                        else:
                            data = torch.cat((sample[:, j, :], data[:, -self.n_conditional_features:].to(sample.device)), dim=1)

                    scores.extend(model.score(data.to(config.device)).cpu().detach().numpy())
            nf_scores_dict[key] = np.asarray(scores)

        return nf_scores_dict

    def log_accuracy(self, epoch, dataset_builder, train_loader, test_loader,
                     metrics_dict, tr_record, te_record, epoch_stats_dict, config, model, log_figures=True):
        t0 = time.time()
        current_metrics = {}
        for key in metrics_dict.keys():
            current_metrics[key] = float(metrics_dict[key][-1])
            if 'loss' in key:  # log 'best' metrics
                current_metrics['best ' + key] = np.amin(metrics_dict[key])

            elif ('epoch' in key) or ('confusion' in key) or ('lr'):
                pass
            else:
                current_metrics['best ' + key] = np.amax(metrics_dict[key])

        for key in current_metrics.keys():
            current_metrics[key] = np.amax(current_metrics[key])  # just a formatting thing - nothing to do with the max of anything

        wandb.log(current_metrics)
        hist = np.histogram(tr_record, bins=256, range=(0, np.quantile(tr_record, 0.9)))
        wandb.log({"Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
        hist = np.histogram(te_record, bins=256, range=(0, np.quantile(te_record, 0.9)))
        wandb.log({"Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

        wandb.log({"Train Loss Coeff. of Variation": np.sqrt(np.var(tr_record)) / np.average(tr_record)})
        wandb.log({"Test Loss Coeff. of Variation": np.sqrt(np.var(te_record)) / np.average(te_record)})

        # correlate losses with molecular features
        tracking_features = np.asarray(epoch_stats_dict['tracking features'])
        loss_correlations = np.zeros(config.dataDims['n tracking features'])
        features = []
        for i in range(config.dataDims['n tracking features']):
            features.append(config.dataDims['tracking features dict'][i])
            loss_correlations[i] = np.corrcoef(te_record, tracking_features[:, i], rowvar=False)[0, 1]

        sort_inds = np.argsort(loss_correlations)
        loss_correlations = loss_correlations[sort_inds]

        if log_figures:
            fig = go.Figure(go.Bar(
                y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['n tracking features'])],
                x=[loss_correlations[i] for i in range(config.dataDims['n tracking features'])],
                orientation='h',
            ))
            wandb.log({'Loss correlations': fig})

        if 'regression' in config.mode:
            target_mean = config.dataDims['mean']
            target_std = config.dataDims['std']

            target = np.asarray(epoch_stats_dict['target'])
            prediction = np.asarray(epoch_stats_dict['prediction'])
            orig_target = target * target_std + target_mean
            orig_prediction = prediction * target_std + target_mean

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
            if log_figures:
                fig = go.Figure()
                for loss in losses:
                    fig.add_trace(go.Histogram(
                        x=losses_dict[loss],
                        histnorm='probability density',
                        nbinsx=100,
                        name=loss,
                        showlegend=True,
                        opacity=0.55
                    ))
                fig.update_layout(barmode='overlay')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                wandb.log({'loss histograms': fig})

                # log target distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=orig_target,
                    histnorm='probability density',
                    nbinsx=100,
                    name='target',
                    showlegend=True,
                    opacity=1
                ))
                fig.add_trace(go.Histogram(
                    x=orig_prediction,
                    histnorm='probability density',
                    nbinsx=100,
                    name='prediction',
                    showlegend=True,
                    opacity=0.65
                ))
                fig.update_layout(barmode='overlay')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                wandb.log({'target distribution': fig})

                # predictions vs target trace
                xline = np.linspace(min(min(orig_target), min(orig_prediction)), max(max(orig_target), max(orig_prediction)), 1000)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=orig_target, y=orig_prediction, mode='markers', showlegend=False))
                fig.add_trace(go.Scatter(x=xline, y=xline, showlegend=False))
                fig.update_layout(xaxis_title='targets', yaxis_title='predictions')
                wandb.log({'Prediction Trace': fig})

        elif 'classification' in config.mode:
            probs = np.asarray(epoch_stats_dict['predictions'])
            targets = np.asarray(epoch_stats_dict['targets']).astype(int)
            predictions = np.argmax(probs, axis=1)
            nClasses = config.dataDims['output classes'][0]

            # get classwise accuracy
            overallTop1Accuracy, byGroupTop1Accuracy = computeTopXAccuracy(config, probs, targets, X=1)
            X = min(nClasses // 2, 5)
            overallTopXAccuracy, byGroupTopXAccuracy = computeTopXAccuracy(config, probs, targets, X=X)

            if targets.ndim > 1:
                targets = targets[:, 0]

            # get confusion matrix
            prob_matrix = np.zeros((nClasses, nClasses))
            target_inds = [np.where(targets == i) for i in range(nClasses)]
            for i in range(nClasses):
                prob_matrix[i, :] = np.sum(probs[target_inds[i]], axis=0)

            confusion_matrix = metrics.confusion_matrix(targets, predictions)

            avgProbAccuracy, avgProbPrecision, avgProbRecall, avgProbF1 = computeF1(prob_matrix.astype(int), nClasses)
            avgAccuracy, avgPrecision, avgRecall, avgF1 = computeF1(confusion_matrix, nClasses)

            if probs.shape[1] == 2:
                roc_score = metrics.roc_auc_score(y_true=targets, y_score=probs[:, 1], average='macro', multi_class='ovo')
            else:
                roc_score = metrics.roc_auc_score(y_true=targets, y_score=probs, average='macro', multi_class='ovo')

            # report scores
            print("Probability Matrix:")
            normed_prob_mat = prob_matrix / np.sum(prob_matrix)
            print('{}'.format((normed_prob_mat * 100 * 100).astype(int)))
            print('Prob based Accuracy {:.3f} Precision {:.3f} Recall {:.3f} F1 {:.3f}'.format(avgProbAccuracy, avgProbPrecision, avgProbRecall, avgProbF1))
            print("Confusion Matrix:")
            print('{}'.format(confusion_matrix))
            print('Accuracy {:.3f} Precision {:.3f} Recall {:.3f} F1 {:.3f}'.format(avgAccuracy, avgPrecision, avgRecall, avgF1))
            print('Top 1 Accuracy Overall: {:.3f} By Group: {:.3f}'.format(overallTop1Accuracy, np.average(byGroupTop1Accuracy)))
            print('Top {} Accuracy Overall: {:.3f} By Group: {:.3f}'.format(X, overallTopXAccuracy, np.average(byGroupTopXAccuracy)))
            print('ROC AUC Score {:.3f}'.format(roc_score))

            metrics_dict = {
                'ROC AUC': roc_score,
                'F1': avgF1,
                'P F1': avgProbF1,
                'precision': avgPrecision,
                'P precision': avgProbPrecision,
                'recall': avgRecall,
                'P recall': avgProbRecall,
                'accuracy': avgAccuracy,
                'P accuracy': avgProbAccuracy,
                'confusion matrix': confusion_matrix,
                'P confusion matrix': prob_matrix,
                'average top 1 accuracy': overallTop1Accuracy,
                'average top {} accuracy'.format(X): overallTopXAccuracy
            }
            accuracy_dict = {}
            if 'symbol' in config.target:
                for i in range(len(config.groupLabels)):
                    accuracy_dict[config.groupLabels[i] + ' top 1 accuracy'] = byGroupTop1Accuracy[i]
                    accuracy_dict[config.groupLabels[i] + ' top {} accuracy'.format(X)] = byGroupTopXAccuracy[i]
            else:
                for i in range(len(byGroupTop1Accuracy)):
                    accuracy_dict['Class {} top 1 accuracy'.format(i)] = byGroupTop1Accuracy[i]
                    accuracy_dict['Class {} top {} accuracy'.format(i, X)] = byGroupTopXAccuracy[i]

            if log_figures:
                xaxis = self.groupLabels
                yaxis = self.groupLabels[-1::-1]
                zaxis = confusion_matrix / np.sum(confusion_matrix)
                fig = go.Figure(data=go.Heatmap(z=np.flipud(zaxis),
                                                x=xaxis,
                                                y=yaxis,
                                                zmin=0,
                                                # zmax=1,
                                                ))
                wandb.log({"Confusion Matrix": fig})

                xaxis = self.groupLabels
                yaxis = self.groupLabels[-1::-1]
                zaxis = prob_matrix / np.sum(prob_matrix)
                fig = go.Figure(data=go.Heatmap(z=np.flipud(zaxis),
                                                x=xaxis,
                                                y=yaxis,
                                                zmin=0,
                                                # zmax=1,
                                                ))
                wandb.log({"P Confusion Matrix": fig})


                rands = np.random.choice(len(targets), size=len(targets), replace=False)  # min(len(targets), 9999)
                wandb.log({"roc {}".format(epoch): wandb.plot.roc_curve(
                    y_true=targets[rands], y_probas=probs[rands], labels=self.groupLabels, title='Epoch {}'.format(epoch))})

        elif config.mode == 'joint modelling':
            '''
            Get the samples
            '''

            n_samples = config.num_samples
            n_dims = self.dataDims['n crystal features']
            dataset_length = len(test_loader.dataset)
            self.sampling_batch_size = min(dataset_length, config.final_batch_size)
            n_repeats = n_samples // dataset_length
            n_samples = n_repeats * dataset_length
            full_dataDims = dataset_builder.get_full_dimension()
            model.eval()

            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU

            # boilerplate
            targets, train_data = self.get_generation_conditions(train_loader, test_loader, model, config)
            self.check_inversion_quality(model, test_loader, config)
            pca = model.fit_pca(train_data, print_variance=(epoch == 0))
            full_pca = model.fit_pca(dataset_builder.full_dataset_data, print_variance=(epoch == 0))
            # get all our samples
            sample_dict = {}
            sample_dict['data'] = targets
            sample_dict['independent gaussian'] = model.prior.sample((n_samples,)).detach().numpy()
            # sample_dict['full independent gaussian'] = model.prior.sample((n_samples,)).detach().numpy()
            sample_dict['pca gaussian'] = model.pca_sampling(pca, n_samples)
            # sample_dict['full pca gaussian'] = model.pca_sampling(full_pca,n_samples)
            sample_dict['nf gaussian'] = self.sample_nf(n_repeats, config, model, test_loader)
            renormalized_sample_dict = {}
            for key in sample_dict.keys():
                if 'full' in key:
                    renormalized_sample_dict[key] = model.destandardize_samples(sample_dict[key], full_dataDims)
                else:
                    renormalized_sample_dict[key] = model.destandardize_samples(sample_dict[key], self.dataDims)

            # skipping because it's expensive
            # # get PC and NF scores
            # pc_scores_dict = self.get_pc_scores(sample_dict, pca)
            # nf_scores_dict = self.get_nf_scores(sample_dict, model, config, test_loader, n_repeats, dataset_length)
            #
            # wandb.log({
            #     'data nf score': np.average(nf_scores_dict['data']),
            #     'pca gaussian nf score': np.average(nf_scores_dict['pca gaussian']),
            #     'independent gaussian nf score': np.average(nf_scores_dict['independent gaussian']),
            #     'nf gaussian nf score': np.average(nf_scores_dict['nf gaussian']),
            #     'data pc score': np.average(pc_scores_dict['data']),
            #     'pca gaussian pc score': np.average(pc_scores_dict['pca gaussian']),
            #     'independent gaussian pc score': np.average(pc_scores_dict['independent gaussian']),
            #     'nf gaussian pc score': np.average(pc_scores_dict['nf gaussian']),
            # })

            # check sample efficiency
            print("Computing sample efficiency")
            sample_efficiency_dict = {}
            feature_accuracy_dict = {}
            for i, (key, sample) in enumerate(renormalized_sample_dict.items()):
                if sample.ndim == 2:
                    if sample.shape[0] == dataset_length * n_repeats:
                        sample = sample.reshape(dataset_length, n_repeats, n_dims)  # roll up the first dim for the indepenent and pc sampels
                    elif sample.shape[0] == dataset_length:
                        sample = sample[:, None, :]  # the real data only has one repeat

                sample_efficiency_dict, feature_accuracy_dict = self.get_sample_efficiency(self.dataDims, targets, sample, sample_efficiency_dict, feature_accuracy_dict, key)
                print(key + ' average error {:.3f}'.format(sample_efficiency_dict[key + ' average mae']))

            wandb.log(sample_efficiency_dict)
            '''
            compute and report some overlaps
            '''
            # flatten sample dimension for the nf data for subsequent analysis
            sample_shape = renormalized_sample_dict['nf gaussian'].shape
            sample_dict['nf gaussian'] = sample_dict['nf gaussian'].reshape(sample_shape[0] * sample_shape[1], sample_shape[2])
            renormalized_sample_dict['nf gaussian'] = renormalized_sample_dict['nf gaussian'].reshape(sample_shape[0] * sample_shape[1], sample_shape[2])

            # 1D histogram overlaps
            overlaps_1d = {}
            for j in range(n_dims):
                mini, maxi = np.amin(renormalized_sample_dict['data'][:, j]), np.amax(renormalized_sample_dict['data'][:, j])
                h1, r1 = np.histogram(renormalized_sample_dict['data'][:, j], bins=100, range=(mini, maxi))
                h1 = h1 / len(renormalized_sample_dict['data'][:, j])
                for i, key in enumerate(['independent gaussian', 'pca gaussian', 'nf gaussian']):
                    h2, r2 = np.histogram(renormalized_sample_dict[key][:, j], bins=r1)
                    h2 = h2 / len(renormalized_sample_dict[key][:, j])
                    overlaps_1d[key + ' ' + self.dataDims['crystal features'][j]] = np.min(np.concatenate((h1[None], h2[None]), axis=0), axis=0).sum()

            average_independent_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys() if 'independent' in key])
            average_pc_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys() if 'pc' in key])
            average_nf_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys() if 'nf' in key])

            print("1D Overlaps With Data: Ind. {:.3f} PC {:.3f} NF {:.3f}".format(average_independent_overlap, average_pc_overlap, average_nf_overlap))
            wandb.log({
                'independent 1D overlap': average_independent_overlap,
                'pc 1D overlap': average_pc_overlap,
                'nf 1D overlap': average_nf_overlap
            })

            # 2D histogram overlaps
            overlaps_2d = {}
            for i in range(n_dims):
                minx, maxx = np.amin(renormalized_sample_dict['data'][:, i]), np.amax(renormalized_sample_dict['data'][:, i])

                for j in range(n_dims):
                    if i != j:
                        miny, maxy = np.amin(renormalized_sample_dict['data'][:, j]), np.amax(renormalized_sample_dict['data'][:, j])
                        h1, x1, y1 = np.histogram2d(renormalized_sample_dict['data'][:, i], renormalized_sample_dict['data'][:, j], bins=100, range=((minx, maxx), (miny, maxy)))
                        h1 = h1 / len(renormalized_sample_dict['data'][:, j])
                        for key in ['independent gaussian', 'pca gaussian', 'nf gaussian']:
                            h2, x2, y2 = np.histogram2d(renormalized_sample_dict[key][:, i], renormalized_sample_dict[key][:, j], bins=(x1, y1))
                            h2 = h2 / len(renormalized_sample_dict[key][:, j])
                            overlaps_2d[key + ' ' + self.dataDims['crystal features'][i] + ' vs ' + self.dataDims['crystal features'][j]] = np.min(np.concatenate((h1.flatten()[None], h2.flatten()[None]), axis=0), axis=0).sum()

            average_independent_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'independent' in key])
            average_pc_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'pc' in key])
            average_nf_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'nf' in key])

            print("2D Overlaps With Data: Ind. {:.3f} PC {:.3f} NF {:.3f}".format(average_independent_overlap, average_pc_overlap, average_nf_overlap))
            wandb.log({
                'independent 2D overlap': average_independent_overlap,
                'pc 2D overlap': average_pc_overlap,
                'nf 2D overlap': average_nf_overlap
            })

            if log_figures:
                fig_dict = {}

                # bar graph of feature-wise sample accuracy
                feat_keys = list(feature_accuracy_dict.keys())
                for key in feat_keys:
                    if 'data' in key:
                        feature_accuracy_dict.pop(key)
                feat_keys = list(feature_accuracy_dict.keys())
                indy_list = [key for key in feat_keys if 'independent gaussian' in key]
                pc_list = [key for key in feat_keys if 'pca gaussian' in key]
                nf_list = [key for key in feat_keys if 'nf gaussian' in key]
                feat_keys = [val for triplet in zip(*[indy_list, pc_list, nf_list]) for val in triplet]
                color = []
                for key in [key for key in feat_keys if 'mae' in key]:
                    if 'independent' in key:
                        color.append('red')
                    elif 'pc' in key:
                        color.append('blue')
                    elif 'nf' in key:
                        color.append('green')
                fig = go.Figure(go.Bar(
                    y=[key for key in feat_keys if 'mae' in key],
                    x=[feature_accuracy_dict[key] for key in feat_keys if 'mae' in key],
                    orientation='h',
                    marker=dict(color=color)
                ))
                fig_dict['Feature-wise MAE'] = fig
                color = []
                for key in [key for key in feat_keys if '0.05' in key]:
                    if 'independent' in key:
                        color.append('red')
                    elif 'pc' in key:
                        color.append('blue')
                    elif 'nf' in key:
                        color.append('green')
                fig = go.Figure(go.Bar(
                    y=[key for key in feat_keys if '0.05' in key],
                    x=[feature_accuracy_dict[key] for key in feat_keys if '0.05' in key],
                    orientation='h',
                    marker=dict(color=color)
                ))
                fig_dict['Feature-wise 0.05 efficiency'] = fig

                # bar graph of 1d overlaps
                color = []
                for key in overlaps_1d.keys():
                    if 'independent' in key:
                        color.append('red')
                    elif 'pc' in key:
                        color.append('blue')
                    elif 'nf' in key:
                        color.append('green')
                fig = go.Figure(go.Bar(
                    y=list(overlaps_1d.keys()),
                    x=[overlaps_1d[key] for key in overlaps_1d],
                    orientation='h',
                    marker=dict(color=color)
                ))
                fig_dict['1D overlaps'] = fig

                color = []
                for key in overlaps_2d.keys():
                    if 'independent' in key:
                        color.append('red')
                    elif 'pc' in key:
                        color.append('blue')
                    elif 'nf' in key:
                        color.append('green')
                fig = go.Figure(go.Bar(
                    y=list(overlaps_2d.keys()),
                    x=[overlaps_2d[key] for key in overlaps_2d],
                    orientation='h',
                    marker=dict(color=color)
                ))
                fig_dict['2D overlaps'] = fig

                # 1d Histograms
                for i in range(n_dims):
                    fig = go.Figure()
                    for key in renormalized_sample_dict.keys():
                        fig.add_trace(go.Histogram(
                            x=renormalized_sample_dict[key][:, i],
                            histnorm='probability density',
                            nbinsx=100,
                            name=key,
                            showlegend=True,
                        ))
                    fig.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    fig.update_traces(opacity=0.5)

                    fig_dict[self.dataDims['crystal features'][i] + ' distribution'] = fig

                # 2d Scatterplots -
                pairs = []  # pick some dims randomly
                np.random.seed(1)
                for i in range(min(self.n_crystal_dims, self.n_crystal_dims ** 2 // 2)):
                    pairs.append(np.random.choice(np.arange(self.n_crystal_dims), size=2, replace=False))
                colors = 'black', 'green', 'orange', 'red'
                nbins = 50
                for n in range(len(pairs)):
                    i, j = pairs[n]
                    fig = go.Figure()
                    for k, key in enumerate(['data', 'independent gaussian', 'pca gaussian', 'nf gaussian']):  # renormalized_sample_dict.keys()):
                        if key == 'data':
                            opacity = 1
                        else:
                            opacity = 0.75
                        fig.add_trace(go.Histogram2dContour(
                            x=renormalized_sample_dict[key][:, i],
                            y=renormalized_sample_dict[key][:, j],
                            histnorm='probability density',
                            name=key,
                            showlegend=True,
                            nbinsx=nbins,
                            nbinsy=nbins,
                            contours=dict(coloring="none"),
                            line_color=colors[k],
                            line_width=0.5,
                            ncontours=50,
                            opacity=opacity
                        ))

                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', )
                    fig_dict[self.dataDims['crystal features'][i] + ' vs ' + self.dataDims['crystal features'][j]] = fig
                    #
                    # # Score Histograms
                    # for n in range(2):
                    #     fig = go.Figure()
                    #     for i, key in enumerate(pc_scores_dict.keys()):
                    #         if n == 0:
                    #             values = pc_scores_dict[key]
                    #
                    #         elif n == 1:
                    #             values = nf_scores_dict[key]
                    #
                    #         if i == 0:
                    #             opacity = 1
                    #         else:
                    #             opacity = 0.5
                    #
                    #         if i == 0:
                    #             xstart = np.quantile(values, 0.01)
                    #             xend = np.quantile(values, 0.99)
                    #         values = np.clip(values, a_min=xstart, a_max=xend)
                    #         fig.add_trace(go.Histogram(
                    #             x=values,
                    #             histnorm='probability density',
                    #             nbinsx=100,
                    #             name=key,
                    #             showlegend=True,
                    #             opacity=opacity,
                    #         ))
                    #         fig.update_layout(barmode='overlay')
                    #     fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    #     # fig.show()
                    #     if n == 0:
                    #         fig_dict['PCA log-P'] = fig
                    #     elif n == 1:
                    #         fig_dict['NF log-P'] = fig

                    keys = list(fig_dict.keys())
                    for key in keys:
                        if '/' in key:
                            new_key = key.replace('/', '')
                            fig_dict[new_key] = fig_dict.pop(key)
                    wandb.log(fig_dict)

        else:
            print(config.mode + ' is not a real model! how did you get this far?')

        print('Analysis took {} seconds'.format(int(time.time() - t0)))

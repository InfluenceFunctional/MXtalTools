import wandb
from utils import *
import glob
from model_utils import *
from CSD_data_manager import Miner
from torch import backends
from dataset_utils import BuildDataset, get_dataloaders
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm


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

        miner = Miner(self.config, dataset_path=self.config.dataset_path)

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

    def prep_metrics(self, config):
        # record metrics
        X = min(self.dataDims['output classes'][0] // 2, 5)
        metrics_list = ['train loss', 'test loss',
                        'F1', 'precision', 'recall', 'accuracy', 'P F1', 'P precision', 'P recall', 'P accuracy',
                        'ROC AUC', 'quasi L2',
                        'confusion matrix', 'P confusion matrix',
                        'average top 1 accuracy', 'average top {} accuracy'.format(X)]
        if 'symbol' in config.target:
            metrics_list.extend(['{} top 1 accuracy'.format(groupLabel) for groupLabel in self.groupLabels])
            metrics_list.extend(['{} top {} accuracy'.format(groupLabel, X) for groupLabel in self.groupLabels])
        else:
            metrics_list.extend(['Class {} top 1 accuracy'.format(i) for i in range(self.dataDims['output classes'][0])])
            metrics_list.extend(['Class {} top {} accuracy'.format(i, X) for i in range(self.dataDims['output classes'][0])])
            self.groupLabels = list(np.arange(self.dataDims['output classes'][0]).astype(str))

        metrics_list.extend(['epoch'])
        metrics_dict = initialize_metrics_dict(metrics_list)

        return metrics_dict

    def prep_flow_metrics(self, config):
        # record metrics
        metrics_list = ['train loss', 'test loss', 'epoch']
        metrics_dict = initialize_metrics_dict(metrics_list)

        return metrics_dict


    def prep_regression_metrics(self, config):
        # record metrics
        metrics_list = ['train loss', 'test loss', 'epoch']
        metrics_dict = initialize_metrics_dict(metrics_list)

        return metrics_dict

    def wandb_logging(self, optimizer, epoch, tr_record, te_record, test_sample_data, metrics_dict, config):
        logging = True
        if logging:
            probs = test_sample_data['model probabilities']
            targets = test_sample_data['sample targets']
            # predictions = test_sample_data['model predictions']

            current_metrics = {}
            for key in metrics_dict.keys():
                if 'confusion' in key:
                    pass
                else:
                    current_metrics[key] = float(metrics_dict[key][-1])

                if 'loss' in key:  # log 'best' metrics
                    current_metrics['best ' + key] = np.amin(metrics_dict[key])

                elif ('epoch' in key) or ('confusion' in key) or ('lr'):
                    pass
                else:
                    current_metrics['best ' + key] = np.amax(metrics_dict[key])
                    # pass

            for key in current_metrics.keys():
                current_metrics[key] = np.amax(current_metrics[key])  # just a formatting thing - nothing to do with the max of anything

            current_metrics['learning rate'] = optimizer.param_groups[0]['lr']

            wandb.log(current_metrics)
            hist = np.histogram(tr_record, range=(0, min(max(len(self.groupLabels) / 2, 2), np.amax(tr_record))), bins=256)
            wandb.log({"Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
            hist = np.histogram(te_record, range=(0, min(max(len(self.groupLabels) / 2, 2), np.amax(te_record))), bins=256)
            wandb.log({"Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
            # wandb.log({"Train Losses": wandb.Histogram(tr_record,num_bins=512)})
            # wandb.log({"Test Losses" : wandb.Histogram(te_record,num_bins=512)})

            wandb.log({"Train Loss Coeff. of Variation": np.sqrt(np.var(tr_record)) / np.average(tr_record)})
            wandb.log({"Test Loss Coeff. of Variation": np.sqrt(np.var(te_record)) / np.average(te_record)})

            xaxis = self.groupLabels
            yaxis = self.groupLabels[-1::-1]
            zaxis = metrics_dict['confusion matrix'][-1] / np.sum(metrics_dict['confusion matrix'][-1])
            fig = go.Figure(data=go.Heatmap(z=np.flipud(zaxis),
                                            x=xaxis,
                                            y=yaxis,
                                            zmin=0,
                                            # zmax=1,
                                            ))
            wandb.log({"Confusion Matrix": fig})

            xaxis = self.groupLabels
            yaxis = self.groupLabels[-1::-1]
            zaxis = metrics_dict['P confusion matrix'][-1] / np.sum(metrics_dict['confusion matrix'][-1])
            fig = go.Figure(data=go.Heatmap(z=np.flipud(zaxis),
                                            x=xaxis,
                                            y=yaxis,
                                            zmin=0,
                                            # zmax=1,
                                            ))
            wandb.log({"P Confusion Matrix": fig})

            # log sorted 1D table of loss correlation coefficients
            corr_coefs = test_sample_data['sample stats']['loss correlation coefficients']
            corr_values = np.nan_to_num(np.asarray([corr_coefs[key] for key in corr_coefs.keys()]), nan=0)
            corr_keys = np.asarray([key for key in corr_coefs.keys()])
            corr_sort_inds = np.argsort(corr_values)

            fig = go.Figure(go.Bar(
                y=corr_keys[corr_sort_inds],
                x=corr_values[corr_sort_inds],
                orientation='h'
            ))
            wandb.log({"Loss Correlations": fig})
            wandb.log(corr_coefs)

            ########## DETAILED REPORTING ################
            if config.detailed_reporting:
                # wandb.log({"confusion matrix {}".format(epoch) : wandb.plot.confusion_matrix(
                #    probs = probs, y_true = targets, class_names = self.groupLabels, title='Epoch {} Confusion Matrix'.format(epoch))})

                rands = np.random.choice(len(targets), size=len(targets), replace=False)  # min(len(targets), 9999)
                wandb.log({"roc {}".format(epoch): wandb.plot.roc_curve(
                    y_true=targets[rands], y_probas=probs[rands], labels=self.groupLabels, title='Epoch {}'.format(epoch))})

                if epoch % config.sample_reporting_frequency == 0:
                    bottomXsamples = test_sample_data['sample stats']['worst samples']
                    topXsamples = test_sample_data['sample stats']['best samples']

                    image_set = [np.array(sample[0]) for sample in bottomXsamples]
                    smiles_set = [sample[3] for sample in bottomXsamples]
                    guess_set = [sample[2] for sample in bottomXsamples]
                    true_set = [sample[1] for sample in bottomXsamples]

                    for i in range(len(bottomXsamples)):
                        image = wandb.Image(image_set[i],
                                            caption="Epoch {} Molecule {} bad sample {} classified as {}".format(epoch, smiles_set[i], self.groupLabels[int(true_set[i])], self.groupLabels[int(guess_set[i])]))
                        wandb.log({"Bad examples": image})

                    image_set = [np.array(sample[0]) for sample in topXsamples]
                    smiles_set = [sample[3] for sample in topXsamples]
                    guess_set = [sample[2] for sample in topXsamples]
                    true_set = [sample[1] for sample in topXsamples]

                    for i in range(len(topXsamples)):
                        image = wandb.Image(image_set[i],
                                            caption="Epoch {} Molecule {} good sample {} classified as {}".format(epoch, smiles_set[i], self.groupLabels[int(true_set[i])], self.groupLabels[int(guess_set[i])]))
                        wandb.log({"Good examples": image})

                    # log 2d histograms of loss vs features
                    hist_data = test_sample_data['sample stats']['loss correlation histograms']
                    for key in hist_data.keys():
                        xaxis = hist_data[key][2][:-1] + np.diff(hist_data[key][2][0:2])
                        yaxis = hist_data[key][1][:-1] + np.diff(hist_data[key][1][0:2])
                        fig = go.Figure(data=go.Heatmap(z=hist_data[key][0],
                                                        x=xaxis,
                                                        y=yaxis
                                                        ))
                        wandb.log({key + " vs Loss Histogram": fig})

    def update_metrics(self, epoch, metrics_dict, err_tr, err_te, test_sample_data, taskwise_train_losses, taskwise_test_losses, config):
        metrics_dict['train loss'].append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
        metrics_dict['test loss'].append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
        metrics_dict = update_metrics_dict(metrics_dict, test_sample_data['accuracy metrics'])
        metrics_dict['epoch'].append(epoch)
        if config.multi_crystal_tasks or config.multi_molecule_tasks:
            metrics_dict = add_multi_task_loss_to_metrics_dict(
                metrics_dict, taskwise_train_losses, taskwise_test_losses, self.dataDims['target features dict']['target feature keys'])

        return metrics_dict

    def update_flow_metrics(self, epoch, metrics_dict, err_tr, err_te):
        metrics_dict['train loss'].append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
        metrics_dict['test loss'].append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
        metrics_dict['epoch'].append(epoch)

        return metrics_dict

    def update_regression_metrics(self, epoch, metrics_dict, err_tr, err_te):
        metrics_dict['train loss'].append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
        metrics_dict['test loss'].append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
        metrics_dict['epoch'].append(epoch)

        return metrics_dict

    def train(self):
        if self.config.mode == 'single molecule classification':
            self.train_predictor()
        elif self.config.mode == 'single molecule regression':
            self.regression()
        elif self.config.mode == 'joint modelling':
            self.joint_modelling()


    def train_predictor(self):
        with wandb.init(config=self.config, project=self.config.project_name, entity=self.config.wandb_username, tags=self.config.experiment_tag):
            config = wandb.config
            print(config)

            # dataset
            dataset_builder = BuildDataset(config)
            config.dataDims = dataset_builder.get_dimension()
            self.dataDims = dataset_builder.get_dimension()
            config.groupLabels = dataset_builder.group_labels
            self.groupLabels = dataset_builder.group_labels
            config.classData = dataset_builder.get_class_weights()

            # get batch size
            if config.auto_batch_sizing:
                train_loader, test_loader, config.final_batch_size = get_batch_size(dataset_builder, config)
            else:
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.initial_batch_size

            # model, optimizer, schedulers
            model, optimizer, schedulers, n_params = init_model(config, self.dataDims)

            print("Training batch size set to {}".format(config.final_batch_size))
            # getNaiveTestLoss(next(iter(test_loader)), config)

            # cuda
            if config.device.lower() == 'cuda':
                torch.backends.cudnn.benchmark = True
                # model = torch.nn.DataParallel(model) # send to multiple GPUs - not always working with wandb
                model.cuda()

            wandb.watch(model, log_graph=True)

            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_metrics(config)

            # training loop
            hit_max_lr, converged, epoch = False, False, 0
            while (epoch < config.max_epochs) and not converged:
                print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                print("Starting Epoch {}".format(epoch))

                t0 = time.time()
                err_tr, tr_record, taskwise_train_losses = \
                    model_epoch(config, dataLoader=train_loader, model=model,
                                optimizer=optimizer, update_gradients=True)  # train & compute loss
                time_train = int(time.time() - t0)

                t0 = time.time()
                err_te, te_record, test_sample_data, taskwise_test_losses = model_epoch(
                    config, dataLoader=test_loader, model=model,
                    update_gradients=False, accuracy_calculation=True, log_sample_statistics=epoch % config.sample_reporting_frequency == 0
                )  # compute loss on test set
                time_test = int(time.time() - t0)

                self.update_metrics(epoch, metrics_dict, err_tr, err_te, test_sample_data, taskwise_train_losses, taskwise_test_losses, config)
                print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_train, time_test))

                optimizer = set_lr(schedulers, optimizer, config, err_tr, hit_max_lr)
                learning_rate = optimizer.param_groups[0]['lr']
                if learning_rate >= config.max_lr: hit_max_lr = True

                # do wandb logging
                self.wandb_logging(optimizer, epoch, tr_record, te_record, test_sample_data, metrics_dict, config)

                if checkConvergence(config, metrics_dict['test loss']) and (epoch > config.history + 2):
                    config.finished = True
                    break

                epoch += 1

            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU



    def regression(self):
        with wandb.init(config=self.config, project=self.config.project_name, entity=self.config.wandb_username, tags=self.config.experiment_tag):
            config = wandb.config
            print(config)

            # dataset
            dataset_builder = BuildDataset(config)
            config.dataDims = dataset_builder.get_dimension()
            self.dataDims = dataset_builder.get_dimension()

            # get batch size
            if config.auto_batch_sizing:
                train_loader, test_loader, config.final_batch_size = get_batch_size(dataset_builder, config)
            else:
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.initial_batch_size

            # model, optimizer, schedulers
            model, optimizer, schedulers, n_params = init_model(config, self.dataDims)

            print("Training batch size set to {}".format(config.final_batch_size))
            # getNaiveTestLoss(next(iter(test_loader)), config)

            # cuda
            if config.device.lower() == 'cuda':
                torch.backends.cudnn.benchmark = True
                # model = torch.nn.DataParallel(model) # send to multiple GPUs - not always working with wandb
                model.cuda()

            wandb.watch(model, log_graph=True)

            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_regression_metrics(config)

            # training loop
            hit_max_lr, converged, epoch = False, False, 0
            while (epoch < config.max_epochs) and not converged:
                print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                print("Starting Epoch {}".format(epoch))

                t0 = time.time()
                err_tr, tr_record = \
                    regression_model_epoch(config, dataLoader=train_loader, model=model,
                                           optimizer=optimizer, update_gradients=True)  # train & compute loss
                time_train = int(time.time() - t0)

                t0 = time.time()
                err_te, te_record, epoch_stats_dict =\
                    regression_model_epoch(config, dataLoader=test_loader, model=model,
                                           update_gradients=False, record_stats = True)  # compute loss on test set
                time_test = int(time.time() - t0)

                self.update_regression_metrics(epoch, metrics_dict, err_tr, err_te)
                print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_train, time_test))

                optimizer = set_lr(schedulers, optimizer, config, err_tr, hit_max_lr)
                learning_rate = optimizer.param_groups[0]['lr']
                if learning_rate >= config.max_lr: hit_max_lr = True

                # do wandb logging

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

                current_metrics['learning rate'] = optimizer.param_groups[0]['lr']

                wandb.log(current_metrics)
                hist = np.histogram(tr_record, bins=256, range=(0,np.quantile(tr_record,0.95)))
                wandb.log({"Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
                hist = np.histogram(te_record, bins=256, range=(0,np.quantile(te_record,0.95)))
                wandb.log({"Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

                if epoch % config.sample_reporting_frequency == 0:
                    t0 = time.time()
                    self.regression_analysis(config, epoch_stats_dict)
                    print('Analysis took {} seconds'.format(int(time.time() - t0)))

                if checkConvergence(config, metrics_dict['test loss']) and (epoch > config.history + 2):
                    config.finished = True
                    self.regression_analysis(config, epoch_stats_dict)

                    break

                epoch += 1

            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU



    def joint_modelling(self):
        '''
        model the joint distribution of crystal cell parameters, conditioned on molecule-level inputs
        :return:
        '''
        with wandb.init(config=self.config, project=self.config.project_name, entity=self.config.wandb_username, tags=self.config.experiment_tag):
            config = wandb.config
            print(config)

            # dataset
            dataset_builder = BuildDataset(config)
            config.dataDims = dataset_builder.get_dimension()
            config.lattice_features = dataset_builder.lattice_keys
            self.lattice_features = dataset_builder.lattice_keys
            self.dataDims = dataset_builder.get_dimension()
            self.n_crystal_dims = self.dataDims['n crystal features']
            if config.conditional_modelling:
                self.n_conditional_features = self.dataDims['n conditional features']
            else:
                self.n_conditional_features = 0

            # get batch size
            if config.auto_batch_sizing:
                train_loader, test_loader, config.final_batch_size = get_batch_size(dataset_builder, config)
            else:
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.initial_batch_size

            # model, optimizer, schedulers
            model, optimizer, schedulers, n_params = init_model(config, self.dataDims)
            print("Training batch size set to {}".format(config.final_batch_size))

            # cuda
            if config.device.lower() == 'cuda':
                torch.backends.cudnn.benchmark = True
                # model = torch.nn.DataParallel(model) # send to multiple GPUs - not always working with wandb
                model.cuda()

            wandb.watch(model, log_graph=True)
            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_flow_metrics(config)

            # sampling_dict = self.pre_run_sampling(dataset_builder, test_loader, config)
            pair_correlations, combined_keys = self.pairwise_correlations_analysis(dataset_builder, config)
            fig = go.Figure(data=go.Heatmap(
                z=np.abs(np.flipud(pair_correlations - np.eye(len(combined_keys)))),
                x=combined_keys, y=combined_keys[-1::-1], zmin=0))  # xaxis labelling is broken - I have no idea why but it ignroes the last column
            wandb.log({"Input pair correlations": fig})

            # training loop
            hit_max_lr, converged, epoch = False, False, 0
            while (epoch < config.max_epochs) and not converged:
                print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                print("Starting Epoch {}".format(epoch))

                t0 = time.time()
                err_tr, tr_record = \
                    flow_model_epoch(config, dataLoader=train_loader, model=model,
                                     optimizer=optimizer, update_gradients=True)  # train & compute loss
                time_train = int(time.time() - t0)

                t0 = time.time()
                err_te, te_record = \
                    flow_model_epoch(config, dataLoader=test_loader, model=model,
                                     update_gradients=False,
                                     )  # compute loss on test set
                time_test = int(time.time() - t0)

                self.update_flow_metrics(epoch, metrics_dict, err_tr, err_te)

                print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_train, time_test))

                optimizer = set_lr(schedulers, optimizer, config, err_tr, hit_max_lr)
                learning_rate = optimizer.param_groups[0]['lr']
                if learning_rate >= config.max_lr:
                    hit_max_lr = True

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

                current_metrics['learning rate'] = optimizer.param_groups[0]['lr']

                wandb.log(current_metrics)
                hist = np.histogram(tr_record, bins=256)
                wandb.log({"Train Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})
                hist = np.histogram(te_record, bins=256)
                wandb.log({"Test Losses": wandb.Histogram(np_histogram=hist, num_bins=256)})

                if (epoch % config.sample_reporting_frequency == 0):
                    t0 = time.time()
                    self.sample_analysis(model, self.dataDims, config, epoch, dataset_builder, train_loader=train_loader, test_loader=test_loader, log_to_wandb=True)
                    print("Sample analysis took {} seconds".format(int(time.time() - t0)))

                if checkConvergence(config, metrics_dict['test loss']) and (epoch > config.history + 2):
                    config.finished = True
                    # do sample analysis at the final epoch
                    t0 = time.time()
                    self.sample_analysis(model, self.dataDims, config, epoch, dataset_builder, train_loader=train_loader, test_loader=test_loader, log_to_wandb=True)
                    print("Sample analysis took {} seconds".format(int(time.time() - t0)))
                    break

                epoch += 1

            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU


    def sample_analysis(self, model, dataDims, config, epoch, dataset_builder, train_loader, test_loader, log_to_wandb=False):
        '''
        collect samples from random sampling, PCA, normalizing flow
        score them
        visualize scores
        visualize 1D and 2D histograms
        compute per-sample loss on a batch of samples
        :param model:
        :param dataset_builder:
        :return:
        '''
        print("Running sample generation and analysis")
        '''
        Get the samples
        '''

        n_samples = config.num_samples
        n_dims = dataDims['n crystal features']
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
        #sample_dict['full independent gaussian'] = model.prior.sample((n_samples,)).detach().numpy()
        sample_dict['pca gaussian'] = model.pca_sampling(pca, n_samples)
        #sample_dict['full pca gaussian'] = model.pca_sampling(full_pca,n_samples)
        sample_dict['nf gaussian'] = self.sample_nf(n_repeats, config, model, test_loader)
        renormalized_sample_dict = {}
        for key in sample_dict.keys():
            if 'full' in key:
                renormalized_sample_dict[key] = model.destandardize_samples(sample_dict[key], full_dataDims)
            else:
                renormalized_sample_dict[key] = model.destandardize_samples(sample_dict[key], dataDims)
        #
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

            sample_efficiency_dict, feature_accuracy_dict = self.get_sample_efficiency(dataDims, targets, sample, sample_efficiency_dict, feature_accuracy_dict, key)
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
                overlaps_1d[key + ' ' + dataDims['crystal features'][j]] = np.min(np.concatenate((h1[None], h2[None]), axis=0), axis=0).sum()

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
                        overlaps_2d[key + ' ' + dataDims['crystal features'][i] + ' vs ' + dataDims['crystal features'][j]] = np.min(np.concatenate((h1.flatten()[None], h2.flatten()[None]), axis=0), axis=0).sum()

        average_independent_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'independent' in key])
        average_pc_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'pc' in key])
        average_nf_overlap = np.average([overlaps_2d[key] for key in overlaps_2d.keys() if 'nf' in key])

        print("2D Overlaps With Data: Ind. {:.3f} PC {:.3f} NF {:.3f}".format(average_independent_overlap, average_pc_overlap, average_nf_overlap))
        wandb.log({
            'independent 2D overlap': average_independent_overlap,
            'pc 2D overlap': average_pc_overlap,
            'nf 2D overlap': average_nf_overlap
        })

        '''
        plot the samples and / or log analysis to wandb
        '''

        if log_to_wandb:
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

                fig_dict[dataDims['crystal features'][i] + ' distribution'] = fig

            # 2d Scatterplots - #TODO generalize for different crystal inputs
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
                # fig.show()
                fig_dict[dataDims['crystal features'][i] + ' vs ' + dataDims['crystal features'][j]] = fig
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

            t0 = time.time()
            keys = list(fig_dict.keys())
            for key in keys:
                if '/' in key:
                    new_key = key.replace('/','')
                    fig_dict[new_key] = fig_dict.pop(key)
            wandb.log(fig_dict)
            print('Logging figures took {} seconds'.format(int(time.time() - t0)))
        #
        # if pyplot_plot:
        #     # visualize the original data via 1D histograms
        #     plt.figure(1)
        #     plt.clf()
        #     for i in range(n_dims):
        #         plt.subplot(4, 4, i + 1)
        #         plt.title(dataDims['crystal features'][i])
        #         for key in renormalized_sample_dict.keys():
        #             if 'data' in key:
        #                 alpha = 1
        #             else:
        #                 alpha = 0.5
        #             if dataDims['dtypes'][i] == 'float64':
        #                 bins = 100
        #                 plt.hist(renormalized_sample_dict[key][:, i], density=True, log=False, bins=bins, alpha=alpha, label=key)
        #             else:
        #                 bins = 10
        #                 plt.hist(np.round(renormalized_sample_dict[key][:, i]), density=True, log=False, bins=bins, alpha=alpha, label=key)
        #
        #         if i == 0:
        #             plt.legend()
        #     plt.tight_layout()
        #
        #     # visualize the original data via 2D scatterplots
        #     plt.figure(2)
        #     plt.clf()
        #     pairs = [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 6], [0, 6], [0, 7], [0, 8]]
        #     for n in range(9):
        #         plt.subplot(3, 3, n + 1)
        #         i, j = pairs[n]
        #         plt.title(dataDims['crystal features'][i] + ' vs. ' + dataDims['crystal features'][j])
        #         plt.scatter(renormalized_sample_dict['data'][:, i], renormalized_sample_dict['data'][:, j], alpha=0.1, marker='.')
        #         # plt.scatter(renormalized_sample_dict['pca gaussian'][:, i], renormalized_sample_dict['pca gaussian'][:, j], alpha=0.05, marker='.')
        #         # plt.scatter(renormalized_sample_dict['independent gaussian'][:, i], renormalized_sample_dict['independent gaussian'][:, j], alpha=0.05, marker='.')
        #         plt.scatter(renormalized_sample_dict['nf gaussian'][:, i], renormalized_sample_dict['nf gaussian'][:, j], alpha=0.05, marker='.')
        #         if n == 0:
        #             plt.legend(('data', 'pc', 'independent', 'nf'))
        #     plt.tight_layout()
        #
        #     # compare scores under the PC and NF model
        #     plt.figure(3)
        #     plt.clf()
        #     for i, key in enumerate(pc_scores_dict.keys()):
        #         plt.subplot(4, 2, 2 * i + 1)
        #         plt.hist(pc_scores_dict[key], bins=100, density=True, log=True, label=key, alpha=1)
        #         plt.hist(nf_scores_dict[key], bins=100, density=True, log=True, label=key, alpha=0.5)
        #         plt.xlim(right=0, left=np.amin(pc_scores_dict['data']) * 1.5)
        #         plt.legend()
        #         if i == 0:
        #             plt.title('log-likelihood under PCA & NF model')
        #
        #     for i, key in enumerate(pc_scores_dict.keys()):
        #         plt.subplot(4, 2, 2 * i + 2)
        #         plt.hist(np.exp(pc_scores_dict[key]), bins=100, density=True, log=True, label=key, alpha=1)
        #         plt.hist(np.exp(nf_scores_dict[key]), bins=100, density=True, log=True, label=key, alpha=.5)
        #         plt.xlim(left=0, right=np.amax(np.exp(pc_scores_dict['data'])) * 1.5)
        #         if i == 0:
        #             plt.title('likelihood under PCA & NF model')

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
        #denominator = np.repeat(np.repeat(np.quantile(renormalized_targets,0.95,axis=0)[None,None,:],samples.shape[0],axis=0),samples.shape[1],axis=1)
        denominator = targets_rep.copy()
        for i in range(dataDims['n crystal features']):
            if dataDims['dtypes'][i] == 'bool':
                denominator[:,:,i] = 1

        errors = np.abs((targets_rep - samples) / denominator)
        feature_mae = np.mean(errors, axis=(0,1))

        for i in range(dataDims['n crystal features']):
            feature_accuracy_dict[sampler + ' ' + dataDims['crystal features'][i] + ' mae'] = feature_mae[i]
            for cutoff in [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3]:
                feature_accuracy_dict[sampler + ' ' + dataDims['crystal features'][i] + ' efficiency at {}'.format(cutoff)] = np.average(errors[:,:,i] < cutoff)

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
                        data = data[:,-self.n_conditional_features:]
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


    def regression_analysis(self,config,epoch_stats_dict):
        '''
        return MAE and MSE losses in original basis
        means and stds
        visualize distributions
        opt: correlate with molecule features
        :param config:
        :param epoch_stats_dict:
        :return:
        '''
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
                loss_i = (orig_target - orig_prediction)**2
            losses_dict[loss] = loss_i # huge unnecessary upload
            loss_dict[loss + ' mean'] = np.mean(loss_i)
            loss_dict[loss + ' std'] = np.std(loss_i)
            print(loss + ' mean: {:.3f} std: {:.3f}'.format(loss_dict[loss + ' mean'], loss_dict[loss + ' std']))

        wandb.log(loss_dict)

        # log loss distribution
        fig = go.Figure()
        for loss in losses:
            fig.add_trace(go.Histogram(
                x = losses_dict[loss],
                histnorm = 'probability density',
                nbinsx = 100,
                name = loss,
                showlegend = True,
                opacity = 0.55
            ))
        fig.update_layout(barmode = 'overlay')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        wandb.log({'loss histograms': fig})

        # log target distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x = orig_target,
            histnorm = 'probability density',
            nbinsx = 100,
            name = 'target',
            showlegend = True,
            opacity = 1
        ))
        fig.add_trace(go.Histogram(
            x = orig_prediction,
            histnorm = 'probability density',
            nbinsx = 100,
            name = 'prediction',
            showlegend = True,
            opacity = 0.65
        ))
        fig.update_layout(barmode = 'overlay')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        wandb.log({'target distribution': fig})

        # predictions vs target trace
        xline = np.linspace(min(min(orig_target),min(orig_prediction)),max(max(orig_target),max(orig_prediction)),1000)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=orig_target,y=orig_prediction, mode = 'markers',showlegend=False))
        fig.add_trace(go.Scatter(x = xline, y = xline,showlegend=False))
        fig.update_layout(xaxis_title = 'targets', yaxis_title = 'predictions')
        wandb.log({'Prediction Trace':fig})

        # correlate losses with molecular features
        tracking_features = np.asarray(epoch_stats_dict['tracking features'])
        loss_correlations = np.zeros(config.dataDims['n tracking features'])
        features = []
        for i in range(config.dataDims['n tracking features']):
            features.append(config.dataDims['tracking features dict'][i])
            loss_correlations[i] = np.corrcoef(losses_dict['abs normed error'], tracking_features[:,i], rowvar = False)[0,1]

        sort_inds = np.argsort(loss_correlations)
        loss_correlations = loss_correlations[sort_inds]
        features = [features[i] for i in sort_inds]
        fig = go.Figure(go.Bar(
            y=[config.dataDims['tracking features dict'][i] for i in range(config.dataDims['n tracking features'])],
            x=[loss_correlations[i] for i in range(config.dataDims['n tracking features'])],
            orientation='h',
        ))
        wandb.log({'Loss correlations':fig})
import numpy as np

from mxtaltools.common.utils import update_stats_dict, softmax_np
from mxtaltools.models.utils import check_convergence, softmax_and_score
from mxtaltools.reporting.online import detailed_reporting, polymorph_classification_trajectory_analysis


class Logger:
    """
    class for storing, processing and reporting metrics
    interface to external reporters e.g., self.wandb
    """

    def __init__(self, config, dataDims, wandb, model_names):
        self.config = config
        self.dataDims = dataDims
        self.wandb = wandb
        self.model_names = model_names
        self.log_figs_to_self = config.logger.log_figures
        self.save_figs_to_local = False
        self.reset_stats_dicts()
        self.init_loss_records()
        self.sample_reporting_frequency = config.logger.sample_reporting_frequency

        self.packing_loss_coefficient = None
        self.epoch = None
        self.learning_rates = {name: None for name in self.model_names}
        self.batch_size = None

        self.converged_flags = {model_name: False for model_name in self.model_names}

    def init_loss_records(self):
        self.current_losses = {}
        for key in self.model_names:
            self.current_losses[key] = {}
            for loss in ['mean_train', 'mean_test', 'all_train', 'all_test', 'mean_extra', 'all_extra']:
                self.current_losses[key][loss] = []

        if not hasattr(self, 'loss_record'):  # initialize it just once
            self.loss_record = {k1: {k2: [] for k2 in self.current_losses[k1].keys() if 'mean' in k2} for k1 in
                                self.current_losses.keys()}

    def reset_for_new_epoch(self, epoch, batch_size):
        self.init_loss_records()
        self.reset_stats_dicts()

        self.epoch = epoch
        self.batch_size = batch_size
        self.wandb.log({'epoch': epoch}, step=epoch, commit=True)

    def update_current_losses(self, model_name, epoch_type, mean_loss, all_loss):
        self.current_losses[model_name]['mean_' + epoch_type].append(mean_loss)
        self.current_losses[model_name]['all_' + epoch_type].extend(all_loss)

    def update_loss_record(self):
        for key in self.loss_record.keys():
            if len(self.loss_record[key]['mean_train']) > 2 * self.config.history:
                self.loss_record[key]['mean_train'].pop(0)
                self.loss_record[key]['mean_test'].pop(0)

            self.loss_record[key]['mean_train'].append(self.current_losses[key]['mean_train'])
            self.loss_record[key]['mean_test'].append(self.current_losses[key]['mean_test'])

    def reset_stats_dicts(self):
        self.train_stats, self.test_stats, self.extra_stats = {}, {}, {}

    def get_stat_dict(self, epoch_type):
        if epoch_type == 'train':
            stat_dict = self.train_stats
        elif epoch_type == 'test':
            stat_dict = self.test_stats
        elif epoch_type == 'extra':
            stat_dict = self.extra_stats
        else:
            print(f"{epoch_type} is not a valid epoch type!")
            import sys
            sys.exit()
        return stat_dict

    def update_stats_dict(self, epoch_type, keys, values, mode='extend'):
        keys_type = type({}.keys())
        values_type = type({}.values())
        if type(keys) == keys_type:
            keys = list(keys)
        if type(values) != values_type:
            values = list(values)
        stat_dict = self.get_stat_dict(epoch_type)
        stat_dict = update_stats_dict(stat_dict, keys, values, mode=mode)

    def numpyize_current_losses(self):
        for k1 in self.current_losses.keys():
            for k2 in self.current_losses[k1].keys():
                if isinstance(self.current_losses[k1][k2], list):
                    self.current_losses[k1][k2] = np.asarray(self.current_losses[k1][k2])

    def log_times(self, times: dict, commit=False):
        elapsed_times = {}
        for start_key in times.keys():
            if isinstance(times[start_key], dict):
                self.log_times(times[start_key])
            elif 'start' in start_key:
                end_key = start_key.split('_start')[0] + '_end'
                if end_key in times.keys():
                    elapsed_times[start_key.split('_start')[0] + '_time'] = times[end_key] - times[start_key]
        self.wandb.log(data=elapsed_times, commit=commit)

    def concatenate_stats_dict(self, epoch_type):
        stat_dict = self.get_stat_dict(epoch_type)
        for _, (key, value) in enumerate(stat_dict.items()):
            if isinstance(value, list):
                if isinstance(value[0], list):  # list of lists
                    stat_dict[key] = np.concatenate(value)
                if isinstance(value[0], np.ndarray):
                    if value[0].ndim > 1:
                        stat_dict[key] = np.concatenate(value)
                    elif len(value) > 1 and value[0].ndim > 0:
                        if len(value[0]) != len(value[1]):
                            stat_dict[key] = np.concatenate(value)
                    else:
                        stat_dict[key] = np.asarray(value)
                elif 'crystaldata' in str(type(value[0])).lower():
                    pass  # do not concatenate lists of crystaldata objects
                else:  # just a list
                    try:
                        stat_dict[key] = np.asarray(value)
                    except ValueError:
                        stat_dict[key] = np.concatenate(value)

    def save_stats_dict(self, prefix=None):
        save_path = prefix + r'test_stats_dict'
        np.save(save_path, self.test_stats)

    def check_model_convergence(self):
        self.converged_flags = {model_name: check_convergence(self.loss_record[model_name]['mean_test'],
                                                              self.config.history,
                                                              self.config.__dict__[
                                                                  model_name].optimizer.convergence_eps,
                                                              self.epoch,
                                                              self.config.minimum_epochs,
                                                              self.config.overfit_tolerance,
                                                              train_record=self.loss_record[model_name]['mean_train'])
                                for model_name in self.model_names if
                                self.config.__dict__[model_name].optimizer is not None}

    def log_fig_dict(self, fig_dict, commit=False):
        if self.log_figs_to_self:
            self.wandb.log(data=fig_dict, commit=commit)

        if self.save_figs_to_local:
            for key in fig_dict.keys():
                fig_dict[key].write_image(key)  # assume these are all plotly figs

    def log_training_metrics(self, commit=False):
        if self.log_figs_to_self:
            # key metrics
            self.wandb.log(data=self.collate_current_metrics(), commit=commit)

            # loss histograms
            for key in self.current_losses.keys():
                if 'all_train' in self.current_losses[key].keys():
                    train_loss = self.current_losses[key]['all_train']
                    self.wandb.log(
                        data={key + '_train_loss_distribution': self.wandb.Histogram(train_loss)},
                        commit=commit)
                    self.wandb.log(
                        data={key + '_log_train_loss_distribution': self.wandb.Histogram(np.nan_to_num(
                            np.log10(np.abs(train_loss)), neginf=0, posinf=0
                        ))},
                        commit=commit)

                if 'all_test' in self.current_losses[key].keys():
                    test_loss = self.current_losses[key]['all_test']
                    self.wandb.log(
                        data={key + '_test_loss_distribution': self.wandb.Histogram(test_loss)},
                        commit=commit)
                    self.wandb.log(
                        data={key + '_log_test_loss_distribution': self.wandb.Histogram(np.nan_to_num(
                            np.log10(np.abs(test_loss)), neginf=0, posinf=0
                        ))},
                        commit=commit)

    def collate_current_metrics(self):
        # general metrics

        metrics_to_log = {'epoch': self.epoch,
                          'packing_loss_coefficient': self.packing_loss_coefficient,
                          'batch_size': self.batch_size}

        for key in self.learning_rates.keys():
            metrics_to_log[f'{key}_learning_rate'] = self.learning_rates[key]

        # losses
        for key in self.current_losses.keys():
            for key2 in self.current_losses[key].keys():
                if isinstance(self.current_losses[key][key2], np.ndarray) and (
                        len(self.current_losses[key][key2] > 0)):  # log 'best' metrics
                    if 'train' in key2:
                        ttype = 'train'
                    elif 'test' in key2:
                        ttype = 'test'
                    elif 'extra' in key2:
                        ttype = 'extra'

                    metrics_to_log[key + '_' + ttype + '_loss'] = np.average(self.current_losses[key][key2])

        # special losses, scores, and miscellaneous items
        for name, stats_dict in zip(['train', 'test', 'extra'], [self.train_stats, self.test_stats, self.extra_stats]):
            if len(stats_dict) > 0:
                for key in stats_dict.keys():
                    if 'loss' in key:
                        metrics_to_log[f'{name}_{key}'] = np.average(stats_dict[key])
                    elif 'score' in key:
                        if stats_dict[key].ndim == 2:  # 2d inputs are generally classwise output scores
                            score = softmax_and_score(stats_dict[key])
                        else:
                            score = stats_dict[key]
                        metrics_to_log[f'{name}_{key}'] = np.average(score)
                    elif isinstance(stats_dict[key], np.ndarray):  # keep any other 1d arrays
                        if stats_dict[key].ndim == 1:
                            if '<U' not in str(stats_dict[key].dtype):  # if it is not a list of strings
                                metrics_to_log[f'{name}_{key}'] = np.average(stats_dict[key])
                    else:  # ignore other objects
                        pass

        for model in self.loss_record.keys():
            for loss in self.loss_record[model].keys():
                record = self.loss_record[model][loss]
                if len(record) > 0:
                    mean_record = np.asarray([rec.mean() for rec in record])
                    metrics_to_log[model + '_best_' + loss] = np.amin(mean_record)

        return metrics_to_log

    def log_detailed_analysis(self, test_loader):
        """sometimes do detailed reporting"""
        if (self.epoch % self.sample_reporting_frequency) == 0:
            detailed_reporting(self.config, self.dataDims,
                               self.train_stats, self.test_stats, extra_test_dict=self.extra_stats)

    def evaluation_analysis(self, test_loader, mode):
        """
        log analysis of an evaluation dataset
        Parameters
        ----------
        test_loader

        Returns
        -------


        """

        if mode == 'polymorph_classification':
            assert len(self.config.dataset.dumps_dirs) == 1, "Polymorph classification trajectory analysis only implemented for single trajectory files"
            polymorph_classification_trajectory_analysis(test_loader, self.test_stats, traj_name=self.config.dataset.dumps_dirs)

            aa = 1

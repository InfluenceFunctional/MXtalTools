import numpy as np

from common.utils import update_stats_dict
from models.utils import check_convergence, softmax_and_score
from reporting.online import detailed_reporting


class Logger:
    """
    class for storing, processing and reporting metrics
    interface to external reporters e.g., self.wandb
    """

    def __init__(self, config, dataDims, wandb):
        self.config = config
        self.dataDims = dataDims
        self.wandb = wandb
        self.model_names = ['generator',
                            'discriminator',
                            'regressor',
                            'proxy_discriminator']
        self.log_figs_to_self = config.logger.log_figures
        self.save_figs_to_local = False
        self.reset_stats_dicts()
        self.init_loss_records()
        self.sample_reporting_frequency = config.logger.sample_reporting_frequency


        self.packing_loss_coefficient = None
        self.epoch = None
        self.learning_rates = {name: None for name in self.model_names}
        self.batch_size = None

    def init_loss_records(self):
        self.current_losses = {}
        for key in self.model_names:
            self.current_losses[key] = {}
            for loss in ['mean_train', 'mean_test', 'all_train', 'all_test', 'mean_extra', 'all_extra']:
                self.current_losses[key][loss] = []

        if not hasattr(self, 'loss_record'):  # initialize it just once
            self.loss_record = {k1: {k2: [] for k2 in self.current_losses[k1].keys() if 'mean' in k2} for k1 in self.current_losses.keys()}

    def reset_for_new_epoch(self, epoch, batch_size):
        self.init_loss_records()
        self.reset_stats_dicts()

        self.epoch = epoch
        self.batch_size = batch_size

    def update_current_losses(self, model_name, epoch_type, mean_loss, all_loss):
        self.current_losses[model_name]['mean_' + epoch_type].append(mean_loss)
        self.current_losses[model_name]['all_' + epoch_type].extend(all_loss)

    def update_loss_record(self):
        for key in self.loss_record.keys():
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
        stat_dict = self.get_stat_dict(epoch_type)
        stat_dict = update_stats_dict(stat_dict, keys, values, mode=mode)

    def numpyize_current_losses(self):
        for k1 in self.current_losses.keys():
            for k2 in self.current_losses[k1].keys():
                if isinstance(self.current_losses[k1][k2], list):
                    self.current_losses[k1][k2] = np.asarray(self.current_losses[k1][k2])

    def numpyize_stats_dict(self, epoch_type):
        stat_dict = self.get_stat_dict(epoch_type)
        for _, (key, value) in enumerate(stat_dict.items()):
            if isinstance(value, list):
                if isinstance(value[0], list):  # list of lists
                    stat_dict[key] = np.concatenate(value)
                if isinstance(value[0], np.ndarray):
                    if value[0].ndim > 1:
                        stat_dict[key] = np.concatenate(value)
                    else:
                        stat_dict[key] = np.asarray(value)
                else:  # just a list
                    stat_dict[key] = np.asarray(value)

    def save_stats_dict(self, prefix=None):
        save_path = prefix + r'test_stats_dict'
        np.save(save_path, self.test_stats)

    def check_model_convergence(self):
        generator_convergence = check_convergence(self.loss_record['generator']['mean_test'], self.config.history,
                                                  self.config.generator.optimizer.convergence_eps)
        discriminator_convergence = check_convergence(self.loss_record['discriminator']['mean_test'], self.config.history,
                                                      self.config.discriminator.optimizer.convergence_eps)
        regressor_convergence = check_convergence(self.loss_record['regressor']['mean_test'], self.config.history,
                                                  self.config.regressor.optimizer.convergence_eps)
        proxy_convergence = check_convergence(self.loss_record['proxy_discriminator']['mean_test'], self.config.history,
                                              self.config.proxy_discriminator.optimizer.convergence_eps)

        return generator_convergence, discriminator_convergence, regressor_convergence, proxy_convergence

    def log_fig_dict(self, fig_dict):
        if self.log_figs_to_self:
            self.wandb.log(fig_dict)

        if self.save_figs_to_local:
            for key in fig_dict.keys():
                fig_dict[key].write_image(key)  # assume these are all plotly figs

    def log_training_metrics(self):
        if self.log_figs_to_self:
            self.wandb.log(self.collate_current_metrics())

    def collate_current_metrics(self):
        metrics_to_log = {}
        # general metrics
        metrics_to_log['epoch'] = self.epoch
        metrics_to_log['packing_loss_coefficient'] = self.packing_loss_coefficient
        for key in self.learning_rates.keys():
            metrics_to_log[f'{key} learning rate'] = self.learning_rates[key]
        metrics_to_log['batch size'] = self.batch_size

        # losses
        for key in self.current_losses.keys():
            for key2 in self.current_losses[key].keys():
                if isinstance(self.current_losses[key][key2], np.ndarray) and (len(self.current_losses[key][key2] > 0)):  # log 'best' metrics
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

        return metrics_to_log

    def log_epoch_analysis(self, test_loader):
        """sometimes do detailed reporting"""
        if (self.epoch % self.sample_reporting_frequency) == 0:
            detailed_reporting(self.config, self.dataDims, test_loader, self.test_stats, extra_test_dict=self.extra_stats)

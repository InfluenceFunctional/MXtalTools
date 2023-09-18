# TODO finish

class Logger:
    """
    class for storing, processing and reporting metrics
    interface to external reporters e.g., wandb
    """
    def __init__(self, config, wandb, **kwargs):
        self.config = config
        self.wandb = wandb
        self.log_figs_to_wandb = True
        self.save_figs_to_local = False

        self.reset_stats_dicts()

    def reset_stats_dicts(self):
        self.train_stats, self.test_stats, self.extra_stats = {}, {}, {}

    def update_stats_dict(self, dictionary: dict, keys, values, mode='append'):
        """
        update dict of running statistics in batches of key:list pairs or one at a time
        """
        if isinstance(keys, list):
            for key, value in zip(keys, values):
                if key not in dictionary.keys():
                    dictionary[key] = []
                if mode == 'append':
                    dictionary[key].append(value)
                elif mode == 'extend':
                    dictionary[key].extend(value)
        else:
            key, value = keys, values
            if key not in dictionary.keys():
                dictionary[key] = []
            if mode == 'append':
                dictionary[key].append(value)
            elif mode == 'extend':
                dictionary[key].extend(value)

        return dictionary

    def log_fig_dict(self, fig_dict):
        if self.log_figs_to_wandb:
            self.wandb.log(fig_dict)

        if self.save_figs_to_local:
            for key in fig_dict.keys():
                fig_dict[key].write_image(key)

    def log_metrics_dict(self, metrics_dict):
        if self.log_figs_to_wandb:
            self.wandb.log(metrics_dict)


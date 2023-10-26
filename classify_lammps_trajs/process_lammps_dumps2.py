import os
import warnings
import torch.optim as optim
import wandb
import argparse
import classify_lammps_trajs.test_configs as test_configs

from classify_lammps_trajs.utils import (generate_dataset_from_dumps, class_names,
                                         collect_to_traj_dataloaders, init_classifier,
                                         train_classifier, reload_model)

warnings.filterwarnings("ignore", category=FutureWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # ignore ovito error

parser = argparse.ArgumentParser()
args = parser.parse_known_args()[1]
if '--config' in args:
    config = getattr(test_configs, args[1])
else:
    config = test_configs.dev

if __name__ == "__main__":
    dataset_name = 'nicotinamide_trajectories_dataset_full'
    datasets_path = config['datasets_path']
    dataset_path = f'{datasets_path}{dataset_name}.pkl'
    dumps_dirs = [config['dumps_path'] + r'bulk_trajs1/T_100',
                  config['dumps_path'] + r'bulk_trajs1/T_350',
                  config['dumps_path'] + r'hot_trajs/melted_trajs_T_950']

    os.chdir(config['runs_path'])

    if not os.path.exists(dataset_path):
        generate_dataset_from_dumps(dumps_dirs, dataset_path)

    train_loader, _ = collect_to_traj_dataloaders(
        dataset_path, config['dataset_size'], batch_size=1, temperatures=[100, 950], test_fraction=0.01)
    test_loader, _ = collect_to_traj_dataloaders(
        dataset_path, config['dataset_size'], batch_size=1, temperatures=[350], test_fraction=0.01)

    classifier = init_classifier(config['conv_cutoff'], config['num_convs'],
                                 config['embedding_depth'], config['dropout'],
                                 config['graph_norm'], config['fc_norm'],
                                 config['num_fcs'], config['message_depth'])

    optimizer = optim.Adam(classifier.parameters(), lr=config['learning_rate'])

    if config['classifier_path'] is not None:
        reload_model(classifier, optimizer, config['classifier_path'], reload_optimizer=True)

    classifier.to(config['device'])
    if config['train_model']:
        train_classifier(config, classifier, optimizer,
                         train_loader, test_loader,
                         config['num_epochs'], wandb,
                         class_names, config['device'],
                         config['batch_size'], config['reporting_frequency'],
                         config['runs_path'], config['run_name']
                         )

    #  todo add evaluation utils & pretty graphs

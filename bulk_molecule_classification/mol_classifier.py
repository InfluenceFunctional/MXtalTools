import os
import warnings
import torch.optim as optim
import wandb
import argparse
from bulk_molecule_classification.dev_configs import configs, dev
from random import shuffle
import torch
import numpy as np

from bulk_molecule_classification.utils import (collect_to_traj_dataloaders, init_classifier,
                                                reload_model)
from bulk_molecule_classification.workflows import train_classifier, classifier_evaluation, trajectory_analysis
from bulk_molecule_classification.NICOAM_constants import nic_class_names, nic_ordered_class_names, urea_class_names, urea_ordered_class_names
from bulk_molecule_classification.dump_data_processing import generate_dataset_from_dumps

warnings.filterwarnings("ignore", category=FutureWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # ignore ovito error

parser = argparse.ArgumentParser()
args = parser.parse_known_args()[1]

if '--config' in args:  # new format
    config = configs[int(args[1])]
else:
    config = dev

if __name__ == "__main__":
    """init model"""
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    """get dataset"""
    if config['train_model'] or config['do_classifier_evaluation']:
        dataset_name = config['dataset_name']
        datasets_path = config['datasets_path']
        dataset_path = f'{datasets_path}{dataset_name}.pkl'
        dumps_dirs = [config['dumps_path'] + dir_name for dir_name in config['dumps_dirs']]

    if 'urea' in config['run_name']:
        class_names = urea_class_names
        ordered_class_names = urea_ordered_class_names
        config['num_forms'] = 7
        config['mol_num_atoms'] = 8
    else:
        class_names = nic_class_names
        ordered_class_names = nic_ordered_class_names
        config['num_forms'] = 10
        config['mol_num_atoms'] = 15

    config['num_topologies'] = 2

    classifier = init_classifier(config['conv_cutoff'], config['num_convs'],
                                 config['embedding_depth'], config['dropout'],
                                 config['graph_norm'], config['fc_norm'],
                                 config['num_fcs'], config['message_depth'],
                                 config['num_forms'], config['num_topologies'],
                                 config['seed'])

    optimizer = optim.Adam(classifier.parameters(), lr=config['learning_rate'])

    if config['classifier_path'] is not None:
        reload_model(classifier, config['device'], optimizer, config['classifier_path'], reload_optimizer=True)

    classifier.to(config['device'])
    os.chdir(config['runs_path'])

    """
    training
    """
    if config['train_model']:
        os.chdir(config['runs_path'])

        if not os.path.exists(dataset_path):
            generate_dataset_from_dumps(dumps_dirs, dataset_path)
            os.chdir(config['runs_path'])

        if 'nic' in dataset_path.lower():
            melt_frac = 1/9
        elif 'urea' in dataset_path.lower():
            melt_frac = 1/6

        _, train_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                      dataset_path, config['dataset_size'],
                                                      conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                      temperatures=[config['training_temps'][0]], test_fraction=1,
                                                      no_melt=True)
        _, test_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                     dataset_path, int(config['dataset_size'] * 0.2),
                                                     conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                     temperatures=[config['training_temps'][1]], test_fraction=1,
                                                     no_melt=True)
        _, hot_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                    dataset_path, int(config['dataset_size'] * melt_frac),
                                                    conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                    temperatures=[config['training_temps'][-1]], test_fraction=1,
                                                    melt_only=True)

        # split the hot trajs equally
        hot_length = len(hot_loader)
        train_loader.dataset.extend(hot_loader.dataset[:hot_length // 2])
        test_loader.dataset.extend(hot_loader.dataset[hot_length // 2:])
        del hot_loader

        train_classifier(config, classifier, optimizer,
                         train_loader, test_loader,
                         config['num_epochs'], wandb,
                         class_names, ordered_class_names,
                         config['device'],
                         config['batch_size'], config['reporting_frequency'],
                         config['runs_path'], config['run_name']
                         )

    """
    Evaluation & analysis
    """
    if config['do_classifier_evaluation']:
        if not os.path.exists(dataset_path):
            generate_dataset_from_dumps(dumps_dirs, dataset_path)

        if 'nic' in dataset_path.lower():
            melt_frac = 1/9
        elif 'urea' in dataset_path.lower():
            melt_frac = 1/6

        _, train_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                      dataset_path, config['dataset_size'],
                                                      conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                      temperatures=[config['training_temps'][0]], test_fraction=1,
                                                      no_melt=True)
        _, test_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                     dataset_path, int(config['dataset_size'] * 0.2),
                                                     conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                     temperatures=[config['training_temps'][1]], test_fraction=1,
                                                     no_melt=True)
        _, hot_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                    dataset_path, int(config['dataset_size'] * melt_frac),
                                                    conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                    temperatures=[config['training_temps'][-1]], test_fraction=1,
                                                    melt_only=True)

        # split the hot trajs equally
        hot_length = len(hot_loader)
        train_loader.dataset.extend(hot_loader.dataset[:hot_length // 2])
        test_loader.dataset.extend(hot_loader.dataset[hot_length // 2:])
        classifier_evaluation(config, classifier, train_loader, test_loader, wandb, class_names, ordered_class_names, config['device'], config['run_name'])

    """
    Trajectory Classification & Analysis
    """
    if config['trajs_to_analyze_list'] is not None:
        with wandb.init(project='cluster_classifier', entity='mkilgour'):
            wandb.run.name = config['run_name'] + '_trajectory_analysis'
            wandb.log({'config': config})
            dumps_list = config['trajs_to_analyze_list']
            # shuffle(dumps_list)  # this speeds up lazy parallel evaluation
            for dump_dir in config['trajs_to_analyze_list']:

                if 'urea' in dump_dir:
                    class_names = urea_class_names
                    ordered_class_names = urea_ordered_class_names
                    config['num_forms'] = 7
                    config['mol_num_atoms'] = 8
                else:
                    class_names = nic_class_names
                    ordered_class_names = nic_ordered_class_names
                    config['num_forms'] = 10
                    config['mol_num_atoms'] = 15

                config['num_topologies'] = 2

                print(f"Processing dump {dump_dir}")
                trajectory_analysis(config, classifier, config['run_name'],
                                    wandb, config['device'],
                                    dumps_dir=dump_dir)

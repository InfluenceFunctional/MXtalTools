import os
import warnings
import torch.optim as optim
import wandb
import argparse
from configs.mol_classifier_configs import configs, dev
import torch
import numpy as np

from mxtaltools.common.mol_classifier_utils import (reload_model)
from bulk_molecule_classification.deprecated.classifier_dataset_prep import collate_training_dataloaders
from bulk_molecule_classification.deprecated.classifier_workflows import train_classifier, classifier_evaluation, trajectory_analysis
from mxtaltools.constants.classifier_constants import nic_class_names, nic_ordered_class_names, urea_class_names, urea_ordered_class_names
from mxtaltools.dataset_management.md_data_processing import generate_dataset_from_dumps
from mxtaltools.models.mol_classifier import PolymorphClassifier
from mxtaltools.models.utils import get_n_config

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

    classifier = PolymorphClassifier(config, dataDims)
    num_params = get_n_config(classifier)
    print(f"Model has {num_params} parameters")

    classifier.to(config['device'])
    optimizer = optim.Adam(classifier.parameters(), lr=config['learning_rate'])

    if config['classifier_path'] is not None:
        reload_model(classifier, config['device'], optimizer, config['classifier_path'], reload_optimizer=True)

    os.chdir(config['runs_path'])

    """
    training
    """
    if config['train_model']:
        os.chdir(config['runs_path'])

        if not os.path.exists(dataset_path):
            generate_dataset_from_dumps(dumps_dirs, dataset_path)
            os.chdir(config['runs_path'])

        train_loader, test_loader = collate_training_dataloaders(config, dataset_path, mode=config['dataset_temperature'])

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

        train_loader, test_loader = collate_training_dataloaders(config, dataset_path)

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
                trajectory_analysis(config, classifier, wandb, config['device'], dumps_dir=dump_dir)
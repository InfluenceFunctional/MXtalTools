from argparse import Namespace
from pathlib import Path

import yaml


def dict2namespace(data_dict: dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


def get_config(override_args):
    """
    Combines YAML configuration file, command line arguments and default arguments into
    a single configuration dictionary.

    - Command line arguments override values in YAML file

    # todo confirm behavior of nested command line overrides
    # todo option to override dataset path in main yaml file
    # todo rename 'dev' / main / experiments officially to main

    Returns
    -------
        Namespace
    """

    '''get user-specific configs'''
    override_keys = [
        arg.strip("--").split("=")[0] for arg in override_args if "--" in arg
    ]
    override_values = [
        arg for arg in override_args if "--" not in arg
    ]
    override_args = dict2namespace({key: val for key, val in zip(override_keys, override_values)})

    user_path = f'configs/users/{override_args.user}.yaml'  # this is a necessary cmd line argument
    user_config = load_yaml(user_path)

    # Read YAML config
    if hasattr(override_args, 'yaml_config'):
        yaml_path = Path(override_args.yaml_config)
    else:
        yaml_path = Path(user_config['paths']['yaml_path'])

    config = load_yaml(yaml_path)

    config['paths'] = user_config['paths']
    config['wandb'] = user_config['wandb']

    for arg in override_args.__dict__.keys():  # todo not sure if this will work for nested args
        if hasattr(config, arg):
            config[arg] = override_args.__dict__[arg]

    # update user paths
    if config['test_mode']:  # todo put dataset name in the config yaml
        dataset_name = 'test_dataset.pkl'
    else:
        dataset_name = 'dataset.pkl'

    config['dataset_name'] = dataset_name
    if config['machine'] == 'local':
        config['workdir'] = user_config['paths']['local_workdir_path']
        config['dataset_path'] = user_config['paths']['local_dataset_dir_path']
        config['checkpoint_dir_path'] = user_config['paths']['local_checkpoint_dir_path']

    elif config['machine'] == 'cluster':
        config['workdir'] = user_config['paths']['cluster_workdir_path']
        config['dataset_path'] = user_config['paths']['cluster_dataset_dir_path']
        config['checkpoint_dir_path'] = user_config['paths']['cluster_checkpoint_dir_path']
        config['save_checkpoints'] = True  # always save checkpoints on cluster

    dataset_yaml_path = Path(config['dataset_yaml_path'])
    dataset_config = load_yaml(dataset_yaml_path)
    config['dataset'] = dataset_config

    return dict2namespace(config)


def load_yaml(path):
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict

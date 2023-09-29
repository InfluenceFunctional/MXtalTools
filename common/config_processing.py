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
    override_args = dict2namespace({key:val for key,val in zip(override_keys, override_values)})

    user_path = f'configs/users/{override_args.user}.yaml'  # this is a necessary cmd line argument
    yaml_path = Path(user_path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        user_config = yaml.safe_load(f)

    # Read YAML config
    if hasattr(override_args, 'yaml_config'):
        yaml_path = Path(override_args.yaml_config)
    else:
        yaml_path = Path(user_config['paths']['yaml_path'])

    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        config = yaml.safe_load(f)

    config['paths'] = user_config['paths']
    config['wandb'] = user_config['wandb']

    for arg in override_args.__dict__.keys():  # todo not sure if this will work for nested args
        if hasattr(config, arg):
            config[arg] = override_args.__dict__[arg]

    # update user paths
    if config['test_mode']:  # todo put this in the config yaml
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
    assert dataset_yaml_path.exists()
    assert dataset_yaml_path.suffix in {".yaml", ".yml"}
    with dataset_yaml_path.open("r") as f:
        dataset_config = yaml.safe_load(f)

    config['dataset'] = dataset_config

    config = dict2namespace(config)

    return config


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def add_arg_list(parser, arg_list):
    for entry in arg_list:
        if entry['type'] == 'bool':
            add_bool_arg(parser, entry['name'], entry['default'])
        else:
            parser.add_argument('--' + entry['name'], type=entry['type'], default=entry['default'])

    return parser


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


def load_yaml(path, append_config_dir=True):
    if append_config_dir:
        path = "configs/" + path
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict

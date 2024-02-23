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


def get_user_config(override_args=None, user_yaml_path=None):
    if user_yaml_path is None:
        assert override_args is not None, "Must provide a user yaml path on command line if not directly to get_config"
        '''get user-specific configs'''
        override_keys = [
            arg.strip("--").split("=")[0] for arg in override_args if "--" in arg
        ]
        override_values = [
            arg for arg in override_args if "--" not in arg
        ]
        override_args = dict2namespace({key: val for key, val in zip(override_keys, override_values)})

        user_path = f'configs/users/{override_args.user}.yaml'  # this is a necessary cmd line argument
    else:
        user_path = user_yaml_path

    return load_yaml(user_path), override_args


def get_main_config(user_config, override_args=None, main_yaml_path=None):
    # Read main YAML config
    if hasattr(override_args, 'yaml_config'):
        yaml_path = Path(override_args.yaml_config)
    elif main_yaml_path is not None:
        yaml_path = Path(main_yaml_path)
    else:
        yaml_path = Path(user_config['paths']['yaml_path'])

    return load_yaml(yaml_path), yaml_path


def print_dict(v, prefix='', keys_list=[]):
    """
    https://stackoverflow.com/questions/10756427/loop-through-all-nested-dictionary-values
    """
    if isinstance(v, dict):
        for k, v2 in v.items():
            p2 = "{}['{}']".format(prefix, k)
            keys_list = print_dict(v2, p2, keys_list)
    elif isinstance(v, list):
        for i, v2 in enumerate(v):
            p2 = "{}[{}]".format(prefix, i)
            keys_list = print_dict(v2, p2, keys_list)
    else:
        keys_list.append(prefix)
        # print('{} = {}'.format(prefix, repr(v)))

    return keys_list


def write_non_overlapping_configs(c1, c2):
    """
    write any items in c2 onto c1 if they are not already there
    """
    for key in c2.keys():
        if key in c1.keys():
            if isinstance(c1[key], dict):
                c1[key] = write_non_overlapping_configs(c1[key], c2[key])

        elif key not in c1.keys():
            c1[key] = c2[key]

    return c1


def get_config(override_args=None, user_yaml_path=None, main_yaml_path=None):
    """
    Combines YAML configuration file, command line arguments and default arguments into
    a single configuration dictionary.

    Returns
    -------
        Namespace
    """
    # load user and main configs
    user_config, override_args = get_user_config(override_args, user_yaml_path)
    main_config, main_config_path = get_main_config(user_config, override_args, main_yaml_path)

    # combine main and user configs
    main_config['paths'] = user_config['paths']
    main_config['paths']['yaml_path'] = main_config_path  # overwrite here
    main_config['wandb'] = user_config['wandb']

    # apply command line override args # todo this does not work for nested override_args
    if override_args is not None:
        for arg in override_args.__dict__.keys():
            if arg in main_config.keys():
                main_config[arg] = override_args.__dict__[arg]

    # generate machine-appropriate paths
    machine_type = main_config['machine']
    main_config['workdir'] = user_config['paths'][machine_type + '_workdir_path']
    main_config['dataset_path'] = user_config['paths'][machine_type + '_dataset_dir_path']
    main_config['checkpoint_dir_path'] = user_config['paths'][machine_type + '_checkpoint_dir_path']
    main_config['config_path'] = user_config['paths'][machine_type + '_config_path']

    # update any missing values from base config
    if 'base_config_path' in main_config.keys() and main_config['base_config_path'] is not None:
        base_config_path = main_config['config_path'] + main_config['base_config_path']
        base_config = load_yaml(base_config_path)
        main_config = write_non_overlapping_configs(main_config, base_config)  # add elements from base into main if they are missing

    for model in main_config['model_paths'].keys():
        if main_config['model_paths'][model] is not None:
            main_config['model_paths'][model] = user_config['paths'][machine_type + '_checkpoints_path'] + main_config['model_paths'][model]

    # always save checkpoints on cluster
    if machine_type == 'cluster':
        main_config['save_checkpoints'] = True

    # load dataset config - but do not overwrite any settings from main config
    dataset_yaml_path = Path(main_config['config_path'] + main_config['dataset_yaml_path'])
    dataset_config = load_yaml(dataset_yaml_path)
    if 'dataset' in main_config.keys():
        for key in dataset_config.keys():
            if key not in main_config['dataset']:
                main_config['dataset'][key] = dataset_config[key]
    else:
        main_config['dataset'] = dataset_config

    return dict2namespace(main_config)


def load_yaml(path):
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict

"""Loading and merging of YAML run configurations (user, main, base and dataset configs) into
nested ``argparse.Namespace`` objects."""
from argparse import Namespace
from pathlib import Path
import yaml


def dict2namespace(data_dict: dict):
    """
    Recursively convert a dictionary and its nested dictionaries into ``argparse.Namespace``
    objects.

    The input is modified in place: each nested dict value in ``data_dict`` is replaced by its
    namespace. Dicts inside lists are not converted.

    Parameters
    ----------
    data_dict : dict
        String-keyed input dictionary. Modified in place.

    Returns
    -------
    data_namespace : argparse.Namespace
        Namespace whose attributes are the (converted) items of ``data_dict``.
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


def get_user_config(override_args=None, user_yaml_path=None):
    """
    Load the user YAML config, locating it from command-line arguments if no path is given.

    If ``user_yaml_path`` is None, ``override_args`` is parsed as a list of command-line
    strings: the arguments containing ``'--'`` give the keys (leading and trailing ``'-'``
    stripped, anything from the first ``'='`` dropped), the remaining arguments give the
    values, paired in order, and the result is converted with :func:`dict2namespace`. The user
    config is then loaded from ``configs/users/{override_args.user}.yaml``, relative to the
    current working directory.

    Parameters
    ----------
    override_args : list of str, optional
        Command-line arguments; required if ``user_yaml_path`` is None, otherwise returned
        unchanged.
    user_yaml_path : str or path-like, optional
        Path of the user YAML config.

    Returns
    -------
    user_config : dict
        Loaded with :func:`load_yaml`.
    override_args : argparse.Namespace or object
        The parsed namespace if ``user_yaml_path`` is None, otherwise the input unchanged.

    Raises
    ------
    AssertionError
        If both arguments are None, or from :func:`load_yaml`.
    """
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
    """
    Load the main YAML config.

    The path is taken, in order of precedence, from ``override_args.yaml_config`` (if that
    attribute exists), from ``main_yaml_path``, or from ``user_config['paths']['yaml_path']``.

    Parameters
    ----------
    user_config : dict
        User config; read only in the last case.
    override_args : object, optional
        Checked for a ``yaml_config`` attribute.
    main_yaml_path : str or path-like, optional
        Path of the main YAML config.

    Returns
    -------
    main_config : dict
        Loaded with :func:`load_yaml`.
    yaml_path : pathlib.Path
        The path that was loaded.
    """
    if hasattr(override_args, 'yaml_config'):
        yaml_path = Path(override_args.yaml_config)
    elif main_yaml_path is not None:
        yaml_path = Path(main_yaml_path)
    else:
        yaml_path = Path(user_config['paths']['yaml_path'])

    return load_yaml(yaml_path), yaml_path


def print_dict(v, prefix='', keys_list=[]):
    """
    Attribution:
    https://stackoverflow.com/questions/10756427/loop-through-all-nested-dictionary-values

    Recursively collect the access path of every leaf of a nested dict/list structure.

    Dict keys are rendered as ``['key']`` and list positions as ``[i]``, appended to
    ``prefix``; each value that is neither a dict nor a list appends its path string to
    ``keys_list``. Nothing is printed.

    Parameters
    ----------
    v : dict, list or object
        Structure to walk.
    prefix : str, optional
        Path of ``v`` itself. Default ``''``.
    keys_list : list, optional
        List to append to; modified in place and returned. The default is a single list
        created when the function is defined, so calls that omit this argument append to
        the same list.

    Returns
    -------
    list of str
        ``keys_list`` with the leaf paths appended, e.g. ``"['a']['b'][0]"``.
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

    return keys_list


def write_non_overlapping_configs(c1, c2):
    """
    Write any items in c2 onto c1 if they are not already there.
    Compatible with nested configs.

    Where both hold a dict under the same key, the two are merged recursively; otherwise the
    value in ``c1`` is kept. Values taken from ``c2`` are inserted by reference, not copied.

    Parameters
    ----------
    c1 : dict
        Target config. Modified in place.
    c2 : dict
        Source of missing items.

    Returns
    -------
    c1 : dict
        The same ``c1`` object.
    """
    for key in c2.keys():
        if key in c1.keys():
            if isinstance(c1[key], dict) and isinstance(c2[key], dict):
                c1[key] = write_non_overlapping_configs(c1[key], c2[key])

        elif key not in c1.keys():
            c1[key] = c2[key]

    return c1


def process_main_config(override_args=None, user_yaml_path=None, main_yaml_path=None, machine=None, append_model_paths=True):
    """
    Combine the user, main, base and dataset YAML configs and command-line overrides into a
    single configuration namespace.

    Steps, in order:

    1. Load the user config (parsing ``override_args``) with :func:`get_user_config` and the
       main config with :func:`get_main_config`.
    2. Set ``paths`` and ``wandb`` from the user config, with ``paths['yaml_path']`` set to
       the main config path.
    3. For each attribute of the override namespace whose name is a top-level key of the main
       config, overwrite that key with the override value (nested keys are not reached).
    4. With ``machine_type`` equal to ``machine`` if given, else ``main_config['machine']``,
       set ``workdir``, ``dataset_path``, ``checkpoint_dir_path`` and ``config_path`` from the
       user config's ``paths`` entries ``{machine_type}_workdir_path``,
       ``{machine_type}_dataset_dir_path``, ``{machine_type}_checkpoint_dir_path`` and
       ``{machine_type}_config_path``.
    5. If ``base_config_path`` is set, fill missing keys from the YAML at
       ``config_path + base_config_path``; then fill missing keys from
       ``config_path + '/experiments/full_base.yaml'`` (both with
       :func:`write_non_overlapping_configs`).
    6. If ``append_model_paths``, prefix every non-None ``model_paths`` entry with the user
       config's ``paths[f'{machine_type}_checkpoints_path']``.
    7. If ``machine_type == 'cluster'``, set ``save_checkpoints`` to True.
    8. Load ``config_path + '/dataset/' + dataset_yaml_path`` and add its top-level keys that
       are missing from ``main_config['dataset']``, or use it as ``dataset`` if that key is
       absent.

    File paths in steps 5, 6 and 8 are joined by string concatenation.

    Parameters
    ----------
    override_args : list of str or argparse.Namespace, optional
        Command-line arguments; see :func:`get_user_config`. When ``user_yaml_path`` is
        given, this must be None or an object with a ``__dict__``.
    user_yaml_path : str or path-like, optional
        Path of the user YAML config.
    main_yaml_path : str or path-like, optional
        Path of the main YAML config, used if the override arguments carry no
        ``yaml_config``.
    machine : str, optional
        Overrides ``main_config['machine']`` for path selection.
    append_model_paths : bool, optional
        Whether to prefix ``model_paths`` entries (step 6). Default True.

    Returns
    -------
    argparse.Namespace
        The merged config, converted with :func:`dict2namespace`.
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
    if machine is not None:
        machine_type = machine
    else:
        machine_type = main_config['machine']

    main_config['workdir'] = user_config['paths'][machine_type + '_workdir_path']
    main_config['dataset_path'] = user_config['paths'][machine_type + '_dataset_dir_path']
    main_config['checkpoint_dir_path'] = user_config['paths'][machine_type + '_checkpoint_dir_path']
    main_config['config_path'] = user_config['paths'][machine_type + '_config_path']

    # update any missing values from this type of base config
    if 'base_config_path' in main_config.keys() and main_config['base_config_path'] is not None:
        base_config_path = main_config['config_path'] + main_config['base_config_path']
        base_config = load_yaml(base_config_path)
        main_config = write_non_overlapping_configs(main_config, base_config)  # add elements from base into main if they are missing

    # update any missing values from the all-inclusive base config
    base_config = load_yaml(main_config['config_path'] + '/experiments/full_base.yaml')
    main_config = write_non_overlapping_configs(main_config, base_config)  # add elements from base into main if they are missing

    for model in main_config['model_paths'].keys():
        if main_config['model_paths'][model] is not None:
            if append_model_paths:
                main_config['model_paths'][model] = user_config['paths'][machine_type + '_checkpoints_path'] + main_config['model_paths'][model]


    # always save checkpoints on cluster
    if machine_type == 'cluster':
        main_config['save_checkpoints'] = True

    # load dataset config - but do not overwrite any settings from main config
    dataset_yaml_path = Path(main_config['config_path'] + '/dataset/' + main_config['dataset_yaml_path'])
    dataset_config = load_yaml(dataset_yaml_path)
    if 'dataset' in main_config.keys():
        for key in dataset_config.keys():
            if key not in main_config['dataset']:
                main_config['dataset'][key] = dataset_config[key]
    else:
        main_config['dataset'] = dataset_config

    return dict2namespace(main_config)


def load_yaml(path):
    """
    Load a YAML file with ``yaml.safe_load``.

    Parameters
    ----------
    path : str or path-like
        Must exist and have the suffix ``.yaml`` or ``.yml``.

    Returns
    -------
    object
        The parsed document: a dict for a top-level mapping, None for an empty file.

    Raises
    ------
    AssertionError
        If the file does not exist or has another suffix.
    """
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict

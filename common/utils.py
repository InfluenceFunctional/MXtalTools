"""import statement"""
import numpy as np
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem import AllChem
from argparse import Namespace
import yaml
from pathlib import Path
import torch
import torch.nn as nn
# from ase.calculators import lj
# from pymatgen.core import (structure, lattice)
# from ccdc.crystal import PackingSimilarity
# from ccdc.io import CrystalReader
# from pymatgen.io import cif
from scipy.cluster.hierarchy import dendrogram

'''
general utilities
'''


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        m.reset_parameters()


def initialize_dict_of_lists(keys):
    m_dict = {}
    for metric in keys:
        m_dict[metric] = []

    return m_dict


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


def get_n_config(model):
    """
    count parameters for a pytorch model
    :param model:
    :return:
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def dict2namespace(data_dict):
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


def delete_from_dataframe(df, inds):
    df = df.drop(index=inds)
    if 'level_0' in df.columns:  # delete unwanted samples
        df = df.drop(columns='level_0')
    df = df.reset_index()

    return df


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


def get_config(args, override_args, args2config):
    """
    Combines YAML configuration file, command line arguments and default arguments into
    a single configuration dictionary.

    - Values in YAML file override default values
    - Command line arguments override values in YAML file

    Returns
    -------
        Namespace
    """

    def _update_config(arg, val, config, override=False):
        config_aux = config
        for k in args2config[arg]:
            if k not in config_aux:
                if k is args2config[arg][-1]:
                    config_aux.update({k: val})
                else:
                    config_aux.update({k: {}})
                    config_aux = config_aux[k]
            else:
                if k is args2config[arg][-1] and override:
                    config_aux[k] = val
                else:
                    config_aux = config_aux[k]

    # Read YAML config
    if args.yaml_config:
        yaml_path = Path(args.yaml_config)
        assert yaml_path.exists()
        assert yaml_path.suffix in {".yaml", ".yml"}
        with yaml_path.open("r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    # Add args to config: add if not provided; override if in command line
    override_args = [
        arg.strip("--").split("=")[0] for arg in override_args if "--" in arg
    ]
    override_args_extra = []
    for k1 in override_args:
        if k1 in args2config:
            v1 = args2config[k1]
            for k2, v2 in args2config.items():
                if v2 == v1 and k2 != k1:
                    override_args_extra.append(k2)
    override_args = override_args + override_args_extra
    for k, v in vars(args).items():
        if k in override_args:
            _update_config(k, v, config, override=True)
        else:
            _update_config(k, v, config, override=False)
    return dict2namespace(config)


def torch_ptp(tensor):
    return torch.max(tensor) - torch.min(tensor)


def standardize(data: np.ndarray, return_std: bool = False, known_mean=None, known_std=None):
    data = data.astype('float32')
    if known_mean is not None:
        mean = known_mean
    else:
        mean = np.mean(data)

    if known_std is not None:
        std = known_std
    else:
        std = np.std(data)

    if std == 0:
        std = 0.01  # hard stop to back up all-one-value inputs

    std_data = (data - mean) / std

    if return_std:
        return std_data, mean, std
    else:
        return std_data


def normalize(x: np.ndarray):
    """
    normalize an input by its span
    """
    min_x = np.amin(x)
    max_x = np.amax(x)
    span = max_x - min_x
    normed_x = (x - min_x) / span
    return normed_x


def np_softmax(x: np.ndarray, temperature=1):
    """
    softmax function implemented in numpy
    """
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype(float)
    probabilities = np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=1)[:, None]
    assert np.sum(np.isnan(probabilities)) == 0
    return probabilities


def earth_movers_distance_torch(x: torch.tensor, y: torch.tensor):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    """
    return torch.sum(torch.abs(torch.cumsum(x, dim=-1) - torch.cumsum(y, dim=-1)), dim=-1)


def earth_movers_distance_np(d1: np.ndarray, d2: np.ndarray):
    '''

    Parameters
    ----------
    d1
    d2

    Returns
    -------
    earth mover's distance (Wasserstein metric) between 1d PDFs (pre-normalized)
    '''
    return np.sum(np.abs(np.cumsum(d1) - np.cumsum(d2)))


def histogram_overlap(d1: np.ndarray, d2: np.ndarray):
    """
    compute the symmetric overlap of two histograms
    """
    return np.sum(np.minimum(d1, d2)) / np.average((d1.sum(), d2.sum()))


def update_stats_dict(dict, keys, values, mode='append'):
    """
    update our running statistics
    """
    if isinstance(keys, list):
        for key, value in zip(keys, values):
            # if isinstance(value, list):
            #     value = np.stack(value)

            if key not in dict.keys():
                dict[key] = []

            if mode == 'append':
                dict[key].append(value)
            elif mode == 'extend':
                dict[key].extend(value)
    else:
        key, value = keys, values
        if key not in dict.keys():
            dict[key] = []
        #
        # if isinstance(value, list):
        #     value = np.stack(value)

        if mode == 'append':
            dict[key].append(value)
        elif mode == 'extend':
            dict[key].extend(value)

    return dict


def update_gan_metrics(epoch, metrics_dict,
                       discriminator_lr, generator_lr, conditioner_lr, regressor_lr,
                       discriminator_train_loss, discriminator_test_loss,
                       generator_train_loss, generator_test_loss,
                       conditioner_train_loss, conditioner_test_loss,
                       regressor_train_loss, regressor_test_loss
                       ):

    metrics_keys = ['epoch',
                    'discriminator learning rate', 'generator learning rate',
                    'conditioner learning rate', 'regressor learning rate',
                    'discriminator train loss', 'discriminator test loss',
                    'generator train loss', 'generator test loss',
                    'conditioner train loss', 'conditioner test loss',
                    'regressor train loss', 'regressor test loss'
                    ]
    metrics_vals = [epoch, discriminator_lr, generator_lr, conditioner_lr, regressor_lr,
                    discriminator_train_loss, discriminator_test_loss,
                    generator_train_loss, generator_test_loss,
                    conditioner_train_loss, conditioner_test_loss,
                    regressor_train_loss, regressor_test_loss
                    ]

    metrics_dict = update_stats_dict(metrics_dict, metrics_keys, metrics_vals)

    return metrics_dict

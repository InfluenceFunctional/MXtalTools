import os
from datetime import datetime

import numpy as np

import torch
from torch_scatter import scatter
from scipy.interpolate import interpn
from typing import List, Optional
import collections
from copy import copy

from mxtaltools.constants.space_group_info import SYM_OPS, LATTICE_TYPE, POINT_GROUPS, SPACE_GROUPS

'''
general utilities
'''


def batch_compute_dipole(pos, batch, z, electronegativity_tensor):
    """
    Compute a rough dipole moment for a flat batch of molecules, as the simple weighted sum of electronegativities.
    Do this in a batch fashion assuming the input is a flat list of graphs, indexed by batch

    Parameters
    ----------
    pos : torch.tensor(n,3)
        3d coordinates
    batch : torch.tensor(n)
        batch index
    z : torch.tensor(n)
        atom types
    electronegativity_tensor
        fixed electronegativities for each atom type

    Returns
    -------
    dipole_moment : torch.tensor(num_graphs,3)
    """
    centers_of_geometry = scatter(pos, batch, dim=0, reduce='mean')
    centers_of_charge = scatter(electronegativity_tensor[z.long()][:, None] * pos, batch, dim=0, reduce='mean')

    return centers_of_charge - centers_of_geometry


def get_point_density(xy, bins=35):
    """
    Interpolate a local density function over 2d points.

    Parameters
    ----------
    xy : torch.tensor(n,2)
    bins : int

    Returns
    -------
    z : normalized local density function
    """

    x, y = xy
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data,
                np.vstack([x, y]).T,
                method="cubic",
                bounds_error=False,
                fill_value=None)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    return np.sqrt(z / z.max())


def chunkify(lst: list, n: int):
    """
    break up a list into n chunks of equal size (up to last chunk)

    Parameters
    ----------
    lst : list
    n : int

    Returns
    -------
    list_of_chunks : [n]
    """
    return [lst[i::n] for i in range(n)]


def torch_ptp(tensor: torch.tensor):
    """
    torch implementation of np.ptp

    Parameters
    ----------
    tensor

    Returns
    -------
    ptp : float
    """

    return torch.max(tensor) - torch.min(tensor)


def standardize_np(data: np.ndarray, return_standardization: bool = False, known_mean=None, known_std=None):
    """
    standardize an input 1D array by subtracting mean and dividing by standard deviation

    Parameters
    ----------
    data : np.array
    return_standardization : bool
        return the mean and std

    known_mean : float, optional
        optionally use precomputed mean

    known_std : float, optional
            optionally use precomputed standard deviation

    Returns
    -------
    std_data : np.array
    mean and standard deviation : optional floats
    """
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
        std = 0.01  # if std == 0, data-mean will be all zeros. Doesn't matter what this value is.

    std_data = (data - mean) / std

    if return_standardization:
        return std_data, mean, std
    else:
        return std_data


def normalize_np(x: np.ndarray):
    """
    normalize an input by its span
    subtract min_x so that range is fixed on [0,1]

    Parameters
    ----------
    x : np.array

    Returns
    -------
    normed_x : np.array
    """
    normed_x = (x - np.amin(x)) / np.ptp(x)
    return normed_x


def softmax_np(x: np.ndarray, temperature: float = 1):
    """
    softmax function implemented in numpy

    Parameters
    ----------
    x : np.array
    temperature : float
        softmax temperature

    Returns
    -------
    softmax_x : np.array
    """
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype(float)
    x -= x.max()
    probabilities = np.exp(x / temperature) / (np.sum(np.exp(x / temperature), axis=1) + 1e-16)[:, None]
    return probabilities


def compute_rdf_distance(rdf1, rdf2, rr, n_parallel_rdf2: int = None):
    """
    Compute a distance metric between two radial distribution functions including sub_rdfs where sub_rdfs are e.g., particular interatomic RDFS within a certain sample (elementwise or atomwise modes).

    If all inputs are numpy arrays, output will be a numpy array, and vice-versa with torch tensors.
    Parameters
    ----------
    rdf1 : array(n_sub_rdfs, n_bins)
    rdf2 : array(n_sub_rdfs, n_bins)
    rr : array(n_bins + 1)
        is the bin edges used for both rdfs
    n_parallel_rdf2: int, optional
        Optionally in parallel compare many rdf2's to a single rdf1
    Returns
    -------

    """

    return_numpy = False
    if not torch.is_tensor(rdf1):
        torch_rdf1 = torch.Tensor(rdf1)
        torch_rdf2 = torch.Tensor(rdf2)
        return_numpy = True
    else:
        torch_rdf1 = rdf1
        torch_rdf2 = rdf2

    if not torch.is_tensor(rr):
        torch_range = torch.Tensor(rr)
    else:
        torch_range = rr

    if n_parallel_rdf2 is not None:
        torch_rdf1_f = torch_rdf1.tile(n_parallel_rdf2, 1, 1)
    else:
        torch_rdf1_f = torch_rdf1

    emd = earth_movers_distance_torch(torch_rdf1_f, torch_rdf2)

    range_normed_emd = emd / len(torch_range) ** 2 * (
                torch_range[-1] - torch_range[0])  # rescale the distance from units of bins to the real physical range
    # do not adjust the above - distance is extensive weirdly extensive in bin scaling
    aggregation_weight = (rdf1.sum(1) + rdf2.sum(1)) / 2  # aggregate rdf components according to pairwise mean weight
    distance = (range_normed_emd * aggregation_weight).mean()

    assert torch.sum(torch.isnan(distance)) == 0
    if return_numpy:
        distance = distance.cpu().detach().numpy()

    return distance


def earth_movers_distance_torch(pdf1: torch.tensor, pdf2: torch.tensor):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    Parameters
    ----------
    pdf1 : torch.tensor(n,i)
    pdf2 : torch.tensor(n,i)

    Returns
    -------
    emd: torch.tensor(n)
    """

    return torch.sum(torch.abs(torch.cumsum(pdf1, dim=-1) - torch.cumsum(pdf2, dim=-1)), dim=-1)


def earth_movers_distance_np(pdf1: np.ndarray, pdf2: np.ndarray):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    Parameters
    ----------
    pdf1 : np.array(n,i)
    pdf2 : np.array(n,i)

    Returns
    -------
    emd: np.array(n)
    """
    return np.sum(np.abs(np.cumsum(pdf1, axis=-1) - np.cumsum(pdf2, axis=-1)), axis=-1)


def histogram_overlap_np(d1: np.ndarray, d2: np.ndarray):
    """
    Compute the symmetric overlap of two histograms

    Parameters
    ----------
    d1 : np.array(n)
    d2 : np.array(n)

    Returns
    -------
    overlap : float
    """
    return np.sum(np.minimum(d1, d2)) / np.average((d1.sum(), d2.sum()))


def update_stats_dict(dictionary: dict, keys, values, mode='append'):
    """
    Append/extend dict of key:list pairs or one at a time

    Parameters
    ----------
    dictionary
    keys
    values
    mode: 'append' or 'extend'

    Returns
    -------
    updated_dictionary
    """

    if isinstance(keys, list):
        for key, value in zip(keys, values):
            if key not in dictionary.keys():
                dictionary[key] = []

            if (mode == 'append') or ('crystaldata' in str(type(value)).lower()):
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


def init_sym_info():
    """
    Initialize dict containing symmetry info for crystals with standard settings and general positions.

    Returns
    -------
    sym_info : dict
    """
    sym_ops = SYM_OPS
    point_groups = POINT_GROUPS
    lattice_type = LATTICE_TYPE
    space_groups = SPACE_GROUPS
    sym_info = {  # collect space group info into single dict
        'sym_ops': sym_ops,
        'point_groups': point_groups,
        'lattice_type': lattice_type,
        'space_groups': space_groups}

    return sym_info


def norm_circular_components(components: torch.tensor):
    """
    Use Pythagoras to norm the sum of squares to the unit circle.
    Parameters
    ----------
    components : torch.tensor(n, 2)

    Returns
    -------
    normed_components : torch.tensor(n, 2)
    """

    return components / torch.sqrt(torch.sum(components ** 2, dim=-1))[:, None]


def components2angle(components: torch.tensor, norm_components=True):
    """
    Take two non-normalized components[n, 2] representing sin(angle) and cos(angle), compute the resulting angle,
    following     https://ai.stackexchange.com/questions/38045/how-can-i-encode-angle-data-to-train-neural-networks

    Optionally norm the sum of squares - doesn't appear to do much though.

    Parameters
    ----------
    components : torch.tensor(n, 2)
    norm_components : bool, optional

    Returns
    -------
    angles : torch.tensor(n, 2)
    """

    if norm_components:
        normed_components = norm_circular_components(components)
        angles = torch.atan2(normed_components[:, 0], normed_components[:, 1])
    else:
        angles = torch.atan2(components[:, 0], components[:, 1])

    return angles


def angle2components(angle: torch.tensor):
    """
    Tecompose an angle input into sin(angle) and cos(angle)

    Parameters
    ----------
    angle : torch.tensor(n)

    Returns
    -------
    sin(angle), cos(angle) : torch.tensor, torch.tensor
    """

    return torch.cat((torch.sin(angle)[:, None], torch.cos(angle)[:, None]), dim=1)


def repeat_interleave(
        repeats: List[int],
        device: Optional[torch.device] = None,
):
    """
    # todo why do we have this? There are builtin methods.
    Alternate implementation of torch.repeat_interleave
    borrowed from torch_geometric.data.collate

    Parameters
    ----------
    repeats : list of ints
    device : str or torch.device

    Returns
    -------
    Repeated tensor which has the same shape as input, except along the given axis.

    """

    outs = [torch.full((n,), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


def namespace2dict(namespace_dict, higher_level=''):
    """
    Convert a dict from an optionally nested namespace to an optionally nested dict.

    Parameters
    ----------
    namespace_dict : dict for the higher level namespace
    higher_level : key for high level dict

    Returns
    -------
    Dict matching higher level namespace
    """
    copied_dict = copy(namespace_dict)
    for key in copied_dict.keys():
        if 'namespace' in str(type(copied_dict[key])).lower():
            copied_dict[key] = namespace2dict(copied_dict[key].__dict__, higher_level=key)
        else:
            pass

    return copied_dict


def flatten_dict(dictionary, parent_key=False, separator='_'):
    """
    From : https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    Recursively convert a nested dictionary into a flattened dictionary

    Parameters
    ----------
    dictionary
    parent_key
    separator

    Returns
    -------
    Dict with all nested dict flattened, with longer keys instead of nesting.
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten_dict(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten_dict({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def make_sequential_directory(yaml_path, workdir):  # make working directory
    """
    make a new working directory labelled by the time & date
    hopefully does not overlap with any other workdirs
    :return:
    """
    run_identifier = str(yaml_path).split('.yaml')[0].split('configs')[1].replace('\\',
                                                                                                         '_').replace(
        '/', '_') + '_' + datetime.today().strftime("%d-%m-%H-%M-%S")
    working_directory = workdir + run_identifier
    os.mkdir(working_directory)
    return run_identifier, working_directory


def flatten_wandb_params(config):
    """Initialize "flat" config for wandb parameter logging"""
    flat_config_dict = flatten_dict(namespace2dict(config.__dict__), separator='_')
    for key in flat_config_dict.keys():
        if 'path' in str(type(flat_config_dict[key])).lower():
            flat_config_dict[key] = str(flat_config_dict[key])
    config.__dict__.update(flat_config_dict)
    return config


def scale_lj_pot(lj_pot: torch.tensor) -> torch.tensor:
    if torch.is_tensor(lj_pot):
        scaled_lj_pot = lj_pot.clone()
        scaled_lj_pot[scaled_lj_pot > 0] = torch.log(scaled_lj_pot[scaled_lj_pot > 0]) + torch.log(torch.ones_like(scaled_lj_pot[scaled_lj_pot>0]))

    else:
        scaled_lj_pot = lj_pot.copy()
        scaled_lj_pot[scaled_lj_pot > 0] = np.log(scaled_lj_pot[scaled_lj_pot > 0]) + np.log(np.ones_like(scaled_lj_pot[scaled_lj_pot>0]))

    # alternate GAUSS = 10 * np.exp(-(xx)**8/0.4) - np.exp(-(xx - 1)**2/0.25)
    return scaled_lj_pot

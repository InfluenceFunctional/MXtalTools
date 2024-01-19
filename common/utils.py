import numpy as np

import torch
from torch_scatter import scatter
import pandas as pd
from scipy.interpolate import interpn
from typing import List, Optional
import collections
from copy import copy

from constants.space_group_info import SYM_OPS, LATTICE_TYPE, POINT_GROUPS, SPACE_GROUPS

'''
general utilities
'''


def batch_compute_dipole(pos, batch, z, electronegativity_tensor):
    centers_of_geometry = scatter(pos, batch, dim=0, reduce='mean')
    centers_of_charge = scatter(electronegativity_tensor[z.long()][:, None] * pos, batch, dim=0, reduce='mean')

    return centers_of_charge - centers_of_geometry


def get_point_density(xy, bins=1000):
    """
    Scatter plot colored by 2d histogram
    """
    x, y = xy
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    return np.sqrt(z / z.max())


def chunkify(lst: list, n: int):
    """
    break up a list into n chunks of equal size (up to last chunk)
    """
    return [lst[i::n] for i in range(n)]


def delete_from_dataframe(df: pd.DataFrame, inds):
    """
    delete rows "inds" from dataframe df
    """
    df.drop(index=inds, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def torch_ptp(tensor: torch.tensor):
    """
    torch implementation of np.ptp
    """
    return torch.max(tensor) - torch.min(tensor)


def standardize_np(data: np.ndarray, return_standardization: bool = False, known_mean=None, known_std=None):
    """
    standardize an input 1D array by subtracting mean and dividing by standard deviation
    optionally use precomputed mean and standard deviation (useful to compare data between datasets)
    optionally return standardization parameters
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
    """
    normed_x = (x - np.amin(x)) / np.ptp(x)
    return normed_x


def softmax_np(x: np.ndarray, temperature: float = 1):
    """
    softmax function implemented in numpy
    """
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype(float)
    probabilities = np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=1)[:, None]
    return probabilities


def compute_rdf_distance(rdf1, rdf2, rr, n_parallel_rdf2: int = None):
    """
    compute a distance metric between two radial distribution functions with shapes
    [num_sub_rdfs, num_bins] where sub_rdfs are e.g., particular interatomic RDFS within a certain sample (elementwise or atomwise modes)
    rr is the bin edges used for both rdfs

    option for input to be torch tensors or numpy arrays, but has to be the same either way
    computation is done in torch
    range rr can be independently either np.array or torch.tensor
    will return same format as given
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

    if n_parallel_rdf2 is not None:  # we can in parallel compare many rdf2's to a single rdf1
        torch_rdf1_f = torch_rdf1.tile(n_parallel_rdf2, 1, 1)
    else:
        torch_rdf1_f = torch_rdf1

    emd = earth_movers_distance_torch(torch_rdf1_f, torch_rdf2)

    range_normed_emd = emd / len(torch_range) ** 2 * (torch_range[-1] - torch_range[0])  # rescale the distance from units of bins to the real physical range
    # do not adjust the above - distance is extensive weirdly extensive in bin scaling
    aggregation_weight = (rdf1.sum(1) + rdf2.sum(1)) / 2  # aggregate rdf components according to pairwise mean weight
    distance = (range_normed_emd * aggregation_weight).mean()

    assert torch.sum(torch.isnan(distance)) == 0
    if return_numpy:
        distance = distance.cpu().detach().numpy()

    return distance


def earth_movers_distance_torch(x: torch.tensor, y: torch.tensor):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    """
    return torch.sum(torch.abs(torch.cumsum(x, dim=-1) - torch.cumsum(y, dim=-1)), dim=-1)


def earth_movers_distance_np(d1: np.ndarray, d2: np.ndarray):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    """
    return np.sum(np.abs(np.cumsum(d1, axis=-1) - np.cumsum(d2, axis=-1)), axis=-1)


def histogram_overlap_np(d1: np.ndarray, d2: np.ndarray):
    """
    compute the symmetric overlap of two histograms
    """
    return np.sum(np.minimum(d1, d2)) / np.average((d1.sum(), d2.sum()))


def update_stats_dict(dictionary: dict, keys, values, mode='append'):
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


def init_sym_info():
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
    """use softmax to norm the sum of squares, and multiply by the signs to keep all 4 quadrants"""

    return components / torch.sqrt(torch.sum(components ** 2, dim=-1))[:, None]


def components2angle(components: torch.tensor, norm_components=True):
    """  # todo decide whether we actually care about the norming
    take two non-normalized components[n_samples, 2] representing
    sin(angle) and cos(angle), compute the resulting angle, following
    https://ai.stackexchange.com/questions/38045/how-can-i-encode-angle-data-to-train-neural-networks

    norm the sum of squares via softmax to enforce prediction on the unit circle
    """
    if norm_components:
        normed_components = norm_circular_components(components)
        angles = torch.atan2(normed_components[:, 0], normed_components[:, 1])
    else:
        angles = torch.atan2(components[:, 0], components[:, 1])

    return angles


def angle2components(angle: torch.tensor):
    """
    decompose an angle input into sin(angle) and cos(angle)
    """
    return torch.cat((torch.sin(angle)[:, None], torch.cos(angle)[:, None]), dim=1)


# def prep_symmetry_info():
#     """
#     if we don't have the symmetry dict prepared already, generate it
#     DEPRECATED USAGE - LEFT IN TO DEMONSTRATE HOW TO GENERATE THESE DATA
#     """
#
#     from pyxtal import symmetry
#     print('Pre-generating spacegroup symmetries')
#     sym_ops = {}
#     point_groups = {}
#     lattice_type = {}
#     space_groups = {}
#     space_group_indices = {}
#     for i in tqdm.tqdm(range(1, 231)):
#         sym_group = symmetry.Group(i)
#         general_position_syms = sym_group.wyckoffs_organized[0][0]
#         sym_ops[i] = [general_position_syms[i].affine_matrix for i in range(
#             len(general_position_syms))]  # first 0 index is for general position, second index is
#         # superfluous, third index is the symmetry operation
#         point_groups[i] = sym_group.point_group
#         lattice_type[i] = sym_group.lattice_type
#         space_groups[i] = sym_group.symbol
#         space_group_indices[sym_group.symbol] = i
#
#     sym_info = {
#         'sym_ops': sym_ops,
#         'point_groups': point_groups,
#         'lattice_type': lattice_type,
#         'space_groups': space_groups,
#         'space_group_indices': space_group_indices}
#
#     np.save('symmetry_info', sym_info)


def repeat_interleave(
        repeats: List[int],
        device: Optional[torch.device] = None,
):
    """
    borrowed from torch_geometric.data.collate
    """
    outs = [torch.full((n,), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


def namespace2dict(namespace_dict, higher_level=''):
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
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
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

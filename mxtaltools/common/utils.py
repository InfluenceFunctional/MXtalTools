import numpy as np

import torch
from torch.nn import functional as F
from torch_scatter import scatter
from scipy.interpolate import interpn
from typing import List, Optional, Union
import collections
from copy import copy

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


def repeat_interleave(
        repeats: List[int],
        device: Optional[torch.device] = None,
):
    """
    # todo why do we have this? There are builtin methods in torch.
    # TODO replace and deprecate
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


def signed_log(y: Union[torch.tensor, np.ndarray]
               ) -> Union[torch.tensor, np.ndarray]:
    if torch.is_tensor(y):
        out = torch.sign(y) * torch.log(1 + torch.abs(y))

    else:
        out = np.sign(y) * np.log(1 + np.abs(y))

    return out


def sample_uniform(num_samples, max_value, device):
    return torch.rand(size=(num_samples,), device=device) * max_value


def parse_to_torch(array: Union[torch.Tensor, np.ndarray, list],
                   device: Union[torch.device, str],
                   dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(array):
        return torch.tensor(array.clone().detach(), dtype=dtype, device=device)
    elif isinstance(array, np.ndarray):
        return torch.tensor(array, dtype=dtype, device=device)
    elif isinstance(array, list):
        if torch.is_tensor(array[0]):
            return torch.cat(array, dim=0).to(device)
        else:
            return torch.tensor(array, dtype=dtype, device=device)


def softplus_shift(x: torch.Tensor,
                   beta: Optional[float] = 5) -> torch.Tensor:
    return F.softplus(x - 0.01, beta=beta) + 0.01

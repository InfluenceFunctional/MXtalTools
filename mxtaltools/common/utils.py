import json
import sys

import numpy as np
import plotly

import torch
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from torch_scatter import scatter
from scipy.interpolate import interpn
from typing import List, Optional, Union
import collections
from copy import copy

'''
general utilities
'''

def get_point_density_knn(xy, k=30, alpha=20):
    """
    Per-point density score from the k-nearest-neighbour distance, log-compressed so the densest point scores 1.

    Parameters
    ----------
    xy : np.ndarray
        [num_dims, num_points] coordinates, typically [2, num_points]. A float64 copy is used, with each dimension
        centred and divided by its standard deviation before the neighbour search.
    k : int, optional
        ``n_neighbors`` of ``sklearn.neighbors.NearestNeighbors``. The queried points are the fitted points, so each
        point is its own first neighbour. Requires at least ``k`` points. Default 30.
    alpha : float, optional
        Log-compression strength. Default 20.

    Returns
    -------
    z : np.ndarray
        [num_points] float64 in (0, 1]. With ``r_k`` the distance to the last of the ``k`` neighbours,
        ``1 / (r_k**2 + 1e-12)`` is divided by its maximum and mapped through ``log1p(alpha * z) / log1p(alpha)``.
    """
    X = xy.T.astype(np.float64)

    # whiten
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True) + 1e-12
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)

    rk = dists[:, -1]          # distance to k-th neighbor
    z = 1.0 / (rk**2 + 1e-12)  # 2D density proxy

    z /= z.max()
    z = np.log1p(alpha * z) / np.log1p(alpha)
    return z

def batch_compute_dipole(pos, batch, z, electronegativity_tensor):
    """
    Rough electronegativity-weighted dipole vector per molecule for a flat batch of molecular graphs.

    For each graph, returns the mean over its atoms of ``electronegativity_tensor[z] * pos`` minus the mean over its
    atoms of ``pos``. The weighted term is a plain mean, not normalised by the summed weights, so the result shifts
    with the coordinate origin unless the graph's mean weight is 1.

    Parameters
    ----------
    pos : torch.Tensor
        [num_nodes, 3] atom coordinates.
    batch : torch.Tensor
        [num_nodes] graph index of each atom.
    z : torch.Tensor
        [num_nodes] atom types, cast to long to index ``electronegativity_tensor``.
    electronegativity_tensor : torch.Tensor
        [num_atom_types] fixed electronegativity per atom type.

    Returns
    -------
    dipole_moment : torch.Tensor
        [num_graphs, 3], with ``num_graphs = batch.max() + 1``.
    """
    centers_of_geometry = scatter(pos, batch, dim=0, reduce='mean')
    centers_of_charge = scatter(electronegativity_tensor[z.long()][:, None] * pos, batch, dim=0, reduce='mean')

    return centers_of_charge - centers_of_geometry


def get_point_density(xy, bins=100, sigma=1.5, alpha=20):
    """
    Per-point density score from a smoothed 2D histogram, floored and log-compressed so the densest point scores 1.

    A density-normalised ``np.histogram2d`` is smoothed with ``scipy.ndimage.gaussian_filter`` (mode ``'nearest'``)
    and linearly interpolated at each point, with points clipped to the range of the bin centres. Values are floored
    at 1e-4 times their maximum, divided by the maximum and mapped through ``log1p(alpha * z) / log1p(alpha)``.

    Parameters
    ----------
    xy : np.ndarray
        [2, num_points] x and y coordinates.
    bins : int, optional
        Histogram bins per axis. Default 100.
    sigma : float, optional
        Gaussian smoothing width, in bins. Default 1.5.
    alpha : float, optional
        Log-compression strength. Default 20.

    Returns
    -------
    z : np.ndarray
        [num_points], in [log1p(1e-4 * alpha) / log1p(alpha), 1].
    """
    x, y = xy

    hist, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    hist = gaussian_filter(hist, sigma=sigma, mode="nearest")

    xc = 0.5 * (x_e[1:] + x_e[:-1])
    yc = 0.5 * (y_e[1:] + y_e[:-1])

    x_clipped = np.clip(x, xc[0], xc[-1])
    y_clipped = np.clip(y, yc[0], yc[-1])

    z = interpn((xc, yc), hist, np.vstack([x_clipped, y_clipped]).T,
                bounds_error=False, fill_value=0.0)

    # floor + log compression
    z = np.clip(z, 1e-4 * z.max(), None)
    z /= z.max()
    z = np.log1p(alpha * z) / np.log1p(alpha)
    return z


def chunkify(lst: list, n: int):
    """
    Split a list into ``n`` interleaved chunks: chunk ``i`` is ``lst[i::n]``.

    Chunk sizes differ by at most one; the first ``len(lst) % n`` chunks are the longer ones.

    Parameters
    ----------
    lst : list
    n : int
        Number of chunks.

    Returns
    -------
    list_of_chunks : list
        ``n`` chunks, each of the same type as ``lst``.
    """
    return [lst[i::n] for i in range(n)]


def torch_ptp(tensor: torch.tensor):
    """
    Range (maximum minus minimum) over all elements, as ``np.ptp`` without an axis.

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    ptp : torch.Tensor
        0-d tensor.
    """

    return torch.max(tensor) - torch.min(tensor)


def standardize_np(data: np.ndarray, return_standardization: bool = False, known_mean=None, known_std=None):
    """
    Standardize an array with one scalar mean and standard deviation taken over all elements.

    Parameters
    ----------
    data : np.ndarray
        Any shape; converted to a float32 copy before standardising.
    return_standardization : bool, optional
        Also return the mean and standard deviation used. Default False.
    known_mean : float, optional
        Mean to use instead of ``np.mean(data)``.
    known_std : float, optional
        Standard deviation to use instead of ``np.std(data)``.

    Returns
    -------
    std_data : np.ndarray
        ``(data - mean) / std``, same shape as ``data``. A standard deviation of 0, computed or given, is replaced by
        0.01.
    mean, std : float
        Only when ``return_standardization`` is True; ``std`` is the value actually used.
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
    Min-max scale an array to [0, 1]: ``(x - min(x)) / ptp(x)`` over all elements.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    normed_x : np.ndarray
        Same shape as ``x``; all NaN when ``x`` is constant.
    """
    normed_x = (x - np.amin(x)) / np.ptp(x)
    return normed_x


def softmax_np(x: np.ndarray, temperature: float = 1):
    """
    Row-wise softmax with temperature.

    Parameters
    ----------
    x : np.ndarray
        [num_rows, num_classes] logits, or [num_classes], treated as one row. A float64 copy is used.
    temperature : float, optional
        Divides the logits. Default 1.

    Returns
    -------
    softmax_x : np.ndarray
        [num_rows, num_classes] float64, 2-D also for 1-D input. Each row is ``exp(x / temperature)`` divided by its
        row sum plus 1e-16. The single maximum over all of ``x``, not each row's own maximum, is subtracted before
        exponentiation.
    """
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype(float)
    x -= x.max()
    probabilities = np.exp(x / temperature) / (np.sum(np.exp(x / temperature), axis=1) + 1e-16)[:, None]
    return probabilities


def histogram_overlap_np(d1: np.ndarray, d2: np.ndarray):
    """
    Symmetric overlap of two histograms: ``sum(min(d1, d2)) / mean(sum(d1), sum(d2))``.

    Parameters
    ----------
    d1 : np.ndarray
        [n] histogram values.
    d2 : np.ndarray
        [n] histogram values on the same bins as ``d1``.

    Returns
    -------
    overlap : float
        For non-negative histograms with positive sums: 1 when they are identical, 0 when no bin is nonzero in both.
    """
    return np.sum(np.minimum(d1, d2)) / np.average((d1.sum(), d2.sum()))


def repeat_interleave(
        repeats: List[int],
        device: Optional[torch.device] = None,
):
    """
    Index vector in which value ``i`` is repeated ``repeats[i]`` times, in order.

    Alternate implementation of ``torch.repeat_interleave(torch.arange(len(repeats)), torch.tensor(repeats))``,
    borrowed from ``torch_geometric.data.collate``.

    # todo why do we have this? There are builtin methods in torch.
    # TODO replace and deprecate

    Parameters
    ----------
    repeats : list of int
        Repeat count for each index.
    device : str or torch.device, optional
        Device of the output.

    Returns
    -------
    out : torch.Tensor
        [sum(repeats)] int64.
    """

    outs = [torch.full((n,), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


def namespace2dict(namespace_dict, higher_level=''):
    """
    Convert a namespace's attribute dict to a dict, recursively converting namespace-valued entries.

    A value counts as a namespace when its type's string representation contains ``'namespace'``
    (case-insensitive); it is replaced by ``namespace2dict(value.__dict__)``. Other values, including dicts and lists
    that contain namespaces, are kept unchanged.

    Parameters
    ----------
    namespace_dict : dict
        Attribute dict of the top-level namespace, e.g. ``namespace.__dict__``. It is not modified.
    higher_level : str, optional
        Key of this level in its parent; passed down the recursion and otherwise unused.

    Returns
    -------
    dict
        Shallow copy of ``namespace_dict`` with namespace values converted.
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
    Recursively flatten nested mappings and lists into a single-level dict.

    From : https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Entries of a nested ``collections.abc.MutableMapping`` get the key ``<parent><separator><key>``; other values,
    including tuples and immutable mappings, are leaves. Elements of a list get the key
    ``<parent>_<index>``: ``separator`` is not forwarded into lists, so every key at or below a list element is joined
    with ``'_'``. Empty mappings and lists contribute no entries. Keys below the top level must be str.

    Parameters
    ----------
    dictionary : Mapping
    parent_key : str or bool, optional
        Prefix for the keys of this level; a falsy value (default False) means no prefix.
    separator : str, optional
        Joins a mapping key to its parent key. Default ``'_'``.

    Returns
    -------
    dict
        New dict of leaf values keyed by the joined keys.
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
    """Signed log compression ``sign(y) * log(1 + |y|)``, elementwise; returns a tensor for a tensor input and the
    result of the numpy ufuncs otherwise."""
    if torch.is_tensor(y):
        out = torch.sign(y) * torch.log(1 + torch.abs(y))

    else:
        out = np.sign(y) * np.log(1 + np.abs(y))

    return out


def sample_uniform(num_samples, max_value, device):
    """[num_samples] values uniform on [0, max_value), drawn with ``torch.rand`` on ``device``."""
    return torch.rand(size=(num_samples,), device=device) * max_value


def parse_to_torch(array: Union[torch.Tensor, np.ndarray, list],
                   device: Union[torch.device, str],
                   dtype=torch.float32) -> torch.Tensor:
    """
    Convert a tensor, numpy array or list to a tensor on ``device``.

    Parameters
    ----------
    array : torch.Tensor, np.ndarray or list
        A tensor is cloned and detached into a new tensor. A list whose first element is a tensor is concatenated
        along dim 0; any other non-empty list, and a numpy array, is passed to ``torch.tensor``. An empty list raises
        ``IndexError``.
    device : torch.device or str
    dtype : torch.dtype, optional
        Output dtype, default ``torch.float32``. Not applied to a list of tensors, which keeps the dtype
        ``torch.cat`` produces.

    Returns
    -------
    torch.Tensor
        None for an input of any other type.
    """
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
                   min_val: Optional[Union[float, torch.Tensor]] = 0.01,
                   beta: Optional[float] = 10) -> torch.Tensor:
    """Smooth floor at ``min_val``: ``softplus(x - min_val, beta=beta) + min_val``, elementwise. Never below
    ``min_val`` and approaching ``x`` for ``x`` well above ``min_val``."""
    return F.softplus(x - min_val, beta=beta) + min_val


def invert_softplus_shift(y: torch.Tensor,
                          min_val: Optional[Union[float, torch.Tensor]] = 0.01,
                          beta: Optional[float] = 10) -> torch.Tensor:
    """Inverse of ``softplus_shift``: ``log(exp(beta * (y - min_val)) - 1) / beta + min_val``, elementwise. Finite
    only for ``y > min_val`` (``y == min_val`` gives -inf, ``y < min_val`` NaN). The exponential is evaluated
    directly, so the result is inf once ``beta * (y - min_val)`` overflows ``exp`` in the input dtype (above about
    88.7 in float32)."""
    return (1.0 / beta) * torch.log(torch.exp(beta * (y - min_val)) - 1.0) + min_val


def scale_bh_energy(bh_energy, beta: Optional[float] = 100):
    """Log-compress the positive entries of a tensor, ``e -> beta * log(1 + e / beta)`` for ``e > 0``; other entries
    are unchanged. Returns a new tensor (the input is cloned)."""
    scaled_en = bh_energy.clone()
    scaled_en[scaled_en > 0] = torch.log(1 + scaled_en[scaled_en > 0] / beta) * beta
    return scaled_en


def smooth_constraint(value: torch.Tensor, threshold: float, mode: 'str', hardness: float):
    """
    Squared-softplus penalty for a one-sided threshold, elementwise.

    Parameters
    ----------
    value : torch.Tensor
    threshold : float
    mode : str
        ``'greater than'``: ``softplus(threshold - value, beta=hardness)**2``, growing as ``value`` falls below
        ``threshold``. ``'less than'``: ``softplus(value - threshold, beta=hardness)**2``, growing as ``value`` rises
        above it. Any other string returns None.
    hardness : float
        softplus ``beta``.

    Returns
    -------
    torch.Tensor
        Same shape as ``value``. Not zero at the threshold, where it equals ``(log(2) / hardness)**2``; it decays
        towards zero on the satisfied side.
    """
    if mode == 'greater than':
        return F.softplus(-(value - threshold), beta=hardness) ** 2
    elif mode == 'less than':
        return F.softplus(value - threshold, beta=hardness) ** 2


def sample_triangular_right(n_samples, start, stop, device='cpu'):
    """
    Sample the right-aligned triangular distribution on [start, stop], whose density rises linearly from 0 at
    ``start``, by inverse-CDF sampling ``start + (stop - start) * sqrt(U)`` with ``U`` from ``torch.rand``.
    Returns [n_samples] on ``device``.
    """
    U = torch.rand(n_samples, device=device)
    return start + (stop - start) * torch.sqrt(U)


def siginv(x):
    """Inverts the sigmoid function (logit): ``log(x / (1 - x))``, elementwise, finite for x in (0, 1)."""
    return torch.log(x/(1-x))


def std_normal_to_uniform(rands):
    """Standard-normal CDF, elementwise: maps standard-normal values to [0, 1]."""
    return torch.distributions.Normal(0, 1).cdf(rands)


def uniform_to_std_normal(uniform):
    """Standard-normal inverse CDF, elementwise, after clipping the input to [1e-5, 1 - 1e-5]; outputs therefore lie
    within about +-4.26."""
    return torch.distributions.Normal(0, 1).icdf(uniform.clip(min=1e-5, max=1-1e-5))  # prevent exploding values


def get_plotly_fig_size_mb(fig) -> float:
    """Size in MiB, ``sys.getsizeof(s) / 1024**2``, of the Python str ``s`` holding ``fig`` serialised with
    ``json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)``. This is the str object's in-memory size, not an encoded
    byte count."""
    # Convert Plotly figure to JSON string
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return sys.getsizeof(fig_json) / (1024 * 1024)


def block_repeat_interleave(
                            lengths: torch.Tensor,
                            repeats: torch.Tensor) -> torch.Tensor:
    """
    Source indices that repeat each ragged block of a flat tensor as a whole block.

    Block ``i`` occupies indices ``offsets[i]`` to ``offsets[i] + lengths[i] - 1`` of the flat tensor, where
    ``offsets`` is the exclusive cumulative sum of ``lengths``. The output lists, block by block, those indices
    ``repeats[i]`` times over (``offsets[i] + 0, 1, ..., lengths[i] - 1, 0, 1, ...``), so ``flat[src_ids]`` is the
    block-repeated tensor.

    Parameters
    ----------
    lengths : torch.Tensor
        [num_blocks] integer block lengths.
    repeats : torch.Tensor
        [num_blocks] integer repeat count per block.

    Returns
    -------
    src_ids : torch.Tensor
        [sum(lengths * repeats)] integer indices into the flat tensor, on ``lengths.device``.
    """
    device = lengths.device
    # start offset of each block in flat
    offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device), lengths[:-1]]), 0)

    # output size contributed per block
    block_out_sizes = lengths * repeats
    total_out = block_out_sizes.sum()

    # which block each output element belongs to
    block_ids = torch.repeat_interleave(torch.arange(len(lengths), device=device),
                                        block_out_sizes)

    # output start offsets
    out_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device),
                                          block_out_sizes[:-1]]), 0)

    # local position within block (resets with modulo!)
    local_index = torch.arange(total_out, device=device) - out_offsets[block_ids]
    pos_in_block = local_index % lengths[block_ids]

    # map back to flat indices
    src_ids = offsets[block_ids] + pos_in_block
    return src_ids


def log_rescale_positive(y: torch.Tensor,
                         cutoff: float = 0
                         ):
    """Log-compress values above ``cutoff``: ``cutoff + log1p(y - cutoff)`` where ``y > cutoff``, ``y`` elsewhere.
    Elementwise, continuous with unit slope at ``cutoff``; returns a new tensor. ``torch.where`` evaluates both
    branches, and at ``y == cutoff - 1`` exactly the unselected ``log1p`` branch makes the gradient NaN."""
    return torch.where(
        y > cutoff,  # where above cutoff
        cutoff + torch.log1p(y - cutoff),  # return log(1+x)
        y  # else return x
    )


def is_cuda_oom(e: Exception) -> bool:
    """
    Classify an exception as a CUDA out-of-memory condition.

    True for a ``torch.cuda.OutOfMemoryError``, or when the lower-cased message contains ``'cuda'`` and ``'memory'``,
    ``'cublas'`` and ``'alloc'``, ``'cusolver'`` and ``'alloc'``, ``'out of memory'``, or
    ``'nonzero is not supported for tensors with more than int_max elements'``. The ``'cuda'`` and ``'memory'`` rule
    also matches other CUDA errors whose message mentions memory, such as an illegal memory access.

    Parameters
    ----------
    e : Exception

    Returns
    -------
    bool
    """
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    s = str(e).lower()
    return (
            ("cuda" in s and "memory" in s)
            or ("cublas" in s and "alloc" in s)
            or ("cusolver" in s and "alloc" in s)
            or ("out of memory" in s)
            or ("nonzero is not supported for tensors with more than int_max elements" in s)
    )

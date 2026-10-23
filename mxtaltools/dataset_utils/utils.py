"""Collation and data-loading helpers for MXtalTools data objects."""
import copy

import torch.nn.functional as F
from random import shuffle
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader, Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader.dataloader import DataLoader, Collater


def fill_unit_maps(data_list, caller: str = 'fill_unit_maps'):
    """``data_list`` with the crystal unit map written onto every element that lacks it, when some carry it.

    Rows written before stage 2, GFN's blank-batch rows and older stores carry no ``unit_ind``
    (``crystal_building.utils.UNIT_MAP_KEY``), and every reader gives such a row ``z_prime`` equal contiguous
    blocks. When some elements carry the map and others do not, each map-less element is replaced by a shallow copy
    (``copy.copy``) holding exactly that layout written out (``crystal_building.utils.derived_unit_ind``; -1 on a row
    whose units cannot be read, which every reader refuses either way), so PyG's collation, which takes the keys from
    the first element, neither raises KeyError nor drops the map, in any order. The caller's elements are not
    modified. A list whose elements all carry the map, or none, is returned as given (the same list object).
    Elements without a PyG store (``_store``) count as carrying none.

    Used by ``collate_data_list`` and by the DataLoaders MXtalTools builds (``MXtalDataLoader``; stage 3 and its
    review, 2026-10-23).

    Raises
    ------
    ValueError
        From ``derived_unit_ind``: a map-less element whose ``num_atoms`` do not sum to its atoms.
    """
    from mxtaltools.crystal_building.utils import UNIT_MAP_KEY, derived_unit_ind
    with_map = [UNIT_MAP_KEY in getattr(d, '_store', ()) for d in data_list]
    if not any(with_map) or all(with_map):
        return data_list
    filled = []
    for i, (d, has) in enumerate(zip(data_list, with_map)):
        if not has:
            d = copy.copy(d)
            d[UNIT_MAP_KEY] = derived_unit_ind(d, f'{caller} (element {i}, without {UNIT_MAP_KEY})')
        filled.append(d)
    return filled


class UnitMapCollater(Collater):
    """``torch_geometric``'s ``Collater``, with ``fill_unit_maps`` applied to a list of data objects first, so rows
    with and without the crystal unit map collate as ``collate_data_list`` collates them (PyG's own collation raised
    KeyError when the first row carried the map, and dropped it for the whole batch when it did not). Any other
    list, and a list of data objects that all carry the map or none, is collated exactly as PyG collates it."""

    def __call__(self, batch):
        if batch and isinstance(batch[0], BaseData):
            batch = fill_unit_maps(batch, 'DataLoader collation')
        return super().__call__(batch)


class MXtalDataLoader(DataLoader):
    """``torch_geometric``'s ``DataLoader`` collating through ``UnitMapCollater``: the loader MXtalTools builds
    (``get_dataloaders``, ``quick_combine_dataloaders``, ``update_dataloader_batch_size``; since the stage-3 review,
    2026-10-23), so a loader mixing rows of different vintages carries the unit map for every row. Only the map
    fill differs from PyG's loader: ``max_z_prime`` widening, slot truncation and the other steps of
    ``collate_data_list`` are not applied. Arguments as ``torch_geometric.loader.DataLoader``."""

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False, follow_batch=None, exclude_keys=None,
                 **kwargs):
        super().__init__(dataset, batch_size, shuffle, follow_batch=follow_batch, exclude_keys=exclude_keys,
                         **kwargs)
        # PyG's DataLoader drops a collate_fn argument and installs its own Collater: replaced after construction
        self.collate_fn = UnitMapCollater(dataset, follow_batch, exclude_keys)


def collate_data_list(data_list, exclude_unit_cell: bool = True,
                      max_z_prime: Optional[int] = None,
                      exclude_keys: Optional[list] = None,
                      skip_default_exclusion: bool = False):
    """
    Collate MXtalTools data objects into a PyG ``Batch``.

    Parameters
    ----------
    data_list : list of MolData or MolCrystalData, or a single data object
        Objects to collate. An object whose ``is_batch`` is True is returned unchanged
        after printing ``"Already batched"``. Any other input that is not a ``list``
        (e.g. a single data object) is wrapped in a one-element list.
    exclude_unit_cell : bool
        If True (default), also exclude ``unit_cell_pos``.
    max_z_prime : int, optional
        Crystal batches only: the value given to ``batch.max_z_prime``. Defaults to
        ``int(batch.z_prime.amax())``. It may drop pad slots (rows built wider than
        their ``z_prime``) but not a row's units, and not add slots beyond the widest
        row's (see Raises).
    exclude_keys : list of str, optional
        Keys to exclude in addition to the default list. The caller's list is not
        modified.
    skip_default_exclusion : bool
        If True, do not apply the default exclude list.

    Returns
    -------
    Batch
        ``Batch.from_data_list(data_list, exclude_keys=...)``; its class is a ``Batch``
        subclass of ``data_list[0]``'s class. Excluded keys are absent from it.

    Raises
    ------
    AssertionError
        If ``data_list`` is empty.
    ValueError
        Crystal batches: if ``max_z_prime`` is below some row's ``z_prime`` (its pose
        slots would be truncated and its units posed as one rigid body; audit #4), or
        if the rows carry fewer than ``3 * max_z_prime`` ``aunit_centroid`` columns.
        Explicit raises, so they hold under ``python -O`` (before 2026-10-22 the first
        truncated silently and the second was a bare ``assert``).

    Notes
    -----
    Default excluded keys: ``edges_dict``, ``niggli_energy``, ``core_energy``,
    ``density_energy``, ``lj_energy``, ``bounding_energy``, ``es_pot``, ``gfn_energy``,
    ``lj_pot``, ``normed_lj_pot``, ``qlj_pot``, ``elj_pot``, ``silu_pot``.

    If ``data_list[0]`` has a ``max_z_prime`` attribute, every element whose
    ``max_z_prime`` is below the largest in the list is padded with zeros to it
    (``aunit_centroid`` and ``aunit_orientation`` by 3 columns per missing slot,
    ``aunit_handedness`` by 1) and its ``max_z_prime`` is updated. The padding is done
    on a shallow copy (``copy.copy``) of the element: the input objects are not
    modified (before 2026-10-22 they were widened in place).

    The crystal unit map ``unit_ind`` (``crystal_building.utils.UNIT_MAP_KEY``): when some
    elements carry it and others do not, each element without it gets, on a shallow copy,
    ``crystal_building.utils.derived_unit_ind`` -- the ``z_prime`` equal contiguous blocks
    every reader gives a row without a map (-1 on a row whose units cannot be read) -- so
    the batch carries the map for every row, in any element order (since stage 3,
    2026-10-23; before, PyG took the keys from the first element: KeyError when it carried
    the map, the whole batch's map dropped when it did not). Raises ValueError
    (``derived_unit_ind``) for an element without a map whose ``num_atoms`` do not sum to
    its atoms.

    Per-unit values (one entry per unit, ``[z_prime, ...]`` per element) collate to a
    ``[sum(z_prime), ...]`` tensor whose slice entry is the unit pointer: the batch
    helpers read it as the ``UNIT`` level (``MXtalBase``'s level contract).

    If the batch's repr contains ``'crystal'`` (case-insensitive; true for a
    ``MolCrystalData`` batch, whose class name is in the repr), the batch then gets:

    - ``max_z_prime``, set to the argument as given, or to ``int(z_prime.amax())`` when
      the argument is None; a row whose ``z_prime`` exceeds it is refused (ValueError);
    - ``aunit_centroid`` and ``aunit_orientation`` truncated to ``3 * max_z_prime``
      columns and ``aunit_handedness`` to ``max_z_prime`` columns, when
      ``aunit_centroid`` is stored;
    - ``reset_sg_info(sg_ind)`` when ``hasattr(batch, 'symmetry_operators')`` is False.
      ``MolCrystalData`` defines ``symmetry_operators`` as a property that returns None
      for a missing key, so for it this condition is never met and the call does not
      happen.

    Per-key concatenation and index offsets follow ``MXtalBase.__cat_dim__`` and
    ``MXtalBase.__inc__``.
    """
    if len(data_list) == 0:
        assert False, "Data list is empty!"

    if hasattr(data_list, 'is_batch'):
        if data_list.is_batch:
            print("Already batched")
            return data_list

    if not isinstance(data_list, list):
        data_list = [data_list]

    # Optionally exclude known keys  # todo this really needs to be fixed up
    if not skip_default_exclusion:
        exclude_keys_i = ['edges_dict',
                          'niggli_energy',
                          'core_energy',
                          'density_energy',
                          'lj_energy',
                          'bounding_energy',
                          'es_pot',
                          'gfn_energy',
                          'lj_pot',
                          'normed_lj_pot',
                          'qlj_pot',
                          'elj_pot',
                          'silu_pot',
                          ]
    else:
        exclude_keys_i = []
    if exclude_keys is not None:
        exclude_keys_i.extend(exclude_keys)

    if exclude_unit_cell:
        exclude_keys_i.append('unit_cell_pos')

    if hasattr(data_list[0], 'max_z_prime'):
        batch_max_z_prime = max(int(d.max_z_prime) for d in data_list)
        widened = []
        for d in data_list:
            d_zp = int(d.max_z_prime)
            if d_zp < batch_max_z_prime:
                pad = batch_max_z_prime - d_zp
                # a shallow copy carries the padded slots: the caller's object is not widened in place (before
                # 2026-10-22 it was, so its pad values depended on its collation history; audit section 7)
                d = copy.copy(d)
                d.aunit_centroid = F.pad(d.aunit_centroid, (0, 3 * pad))
                d.aunit_orientation = F.pad(d.aunit_orientation, (0, 3 * pad))
                d.aunit_handedness = F.pad(d.aunit_handedness, (0, pad))
                d.max_z_prime = batch_max_z_prime
            widened.append(d)
        data_list = widened

    data_list = fill_unit_maps(data_list, 'collate_data_list')

    batch = Batch.from_data_list(data_list,
                                 exclude_keys=list(exclude_keys_i),
                                 )

    # if hasattr(batch, 'max_z_prime'):
    #     if isinstance(batch.max_z_prime, int):
    #         pass
    # else:
    if 'crystal' in batch.__str__().lower():
        if max_z_prime is not None:
            batch.max_z_prime = max_z_prime
        else:
            batch.max_z_prime = int(batch.z_prime.amax())

        z_prime = getattr(batch, 'z_prime', None)
        if z_prime is not None:
            # a row with more units than slots would be truncated to its first max_z_prime units and then posed as
            # one rigid body (audit #4): refuse it
            zp = z_prime.reshape(-1)
            wide = zp > batch.max_z_prime
            if bool(wide.any()):  # the row list only when refusing
                wide = torch.nonzero(wide).flatten().tolist()
                ident = getattr(batch, 'identifier', None)
                ident = ident if isinstance(ident, (list, tuple)) else None
                rows = ', '.join(f'row {i}' + (f' ({ident[i]})' if ident is not None and i < len(ident) else '')
                                 + f' (z_prime {int(zp[i])})' for i in wide[:8])
                raise ValueError(f"collate_data_list: max_z_prime={batch.max_z_prime} is below the z_prime of "
                                 f"{len(wide)} row(s): {rows}{' and more' if len(wide) > 8 else ''}; their pose slots "
                                 f"would be truncated. Collate at max_z_prime >= {int(zp.amax())}, or leave it None")

        if hasattr(batch, 'aunit_centroid'):
            if batch.aunit_centroid.shape[1] < 3 * batch.max_z_prime:
                raise ValueError(f"collate_data_list: max_z_prime={batch.max_z_prime} needs "
                                 f"{3 * batch.max_z_prime} aunit_centroid columns, but the rows carry "
                                 f"{batch.aunit_centroid.shape[1] // 3} pose slot(s); rows are padded to the widest "
                                 f"row's max_z_prime, never beyond it (build the rows at that width)")
            batch.aunit_centroid = batch.aunit_centroid[:, :3 * batch.max_z_prime]
            batch.aunit_orientation = batch.aunit_orientation[:, :3 * batch.max_z_prime]
            batch.aunit_handedness = batch.aunit_handedness[:, :batch.max_z_prime]
        if not hasattr(batch, 'symmetry_operators'):
            batch.reset_sg_info(batch.sg_ind)

    return batch


def quick_combine_dataloaders(dataset, data_loader, batch_size, max_size):
    """
    Prepend ``dataset`` to a loader's shuffled dataset and return a new loader.

    ``data_loader.dataset`` is shuffled in place and ``dataset`` is extended in place
    with it; the first ``max_size`` entries of the result go into a new
    ``MXtalDataLoader`` (``torch_geometric``'s ``DataLoader``, filling the unit map of
    rows that lack it beside rows that carry it) with ``shuffle=True`` and
    ``data_loader``'s ``num_workers``, ``pin_memory`` and ``drop_last``.

    Parameters
    ----------
    dataset : list
        New samples; kept first.
    data_loader : DataLoader
        Loader whose ``dataset`` (a mutable sequence) supplies the older samples.
    batch_size : int
    max_size : int
        Maximum number of samples in the returned loader's dataset.

    Returns
    -------
    DataLoader
    """
    shuffle(data_loader.dataset)  # randomize order of old dataset
    dataset.extend(data_loader.dataset)  # append old dataset to new one
    dataset = dataset[:max_size]  # truncate from the end of the old dataset
    dataloader = MXtalDataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=data_loader.num_workers,
                                 pin_memory=data_loader.pin_memory,
                                 drop_last=data_loader.drop_last)

    return dataloader


def quick_combine_crystal_embedding_dataloaders(dataset, data_loader, batch_size, max_size):
    """
    Put a new ``(x, y)`` dataset ahead of a permuted copy of a loader's; return a new loader.

    The rows of ``data_loader.dataset.x`` and ``.y`` are randomly permuted (the loader's
    dataset is not modified), concatenated after ``dataset.x`` / ``dataset.y`` along
    dim 0, truncated to ``max_size`` rows, wrapped in a :class:`SimpleDataset` and
    served by a new ``torch_geometric`` ``DataLoader`` with ``shuffle=True`` and
    ``data_loader``'s ``num_workers``, ``pin_memory`` and ``drop_last``.

    Parameters
    ----------
    dataset : object with tensor attributes ``x`` and ``y``
        New samples; kept first.
    data_loader : DataLoader
        Loader whose dataset has tensor attributes ``x`` and ``y``.
    batch_size : int
    max_size : int
        Maximum number of rows kept.

    Returns
    -------
    DataLoader
    """
    x, y = data_loader.dataset.x, data_loader.dataset.y  # randomize order of old dataset
    rands = torch.tensor(np.random.choice(len(x), len(x), replace=False), dtype=torch.long)
    x = x[rands]
    y = y[rands]

    new_x, new_y = dataset.x, dataset.y  # prepend the new dataset
    new_x = torch.cat((new_x, x), dim=0)[:max_size]
    new_y = torch.cat((new_y, y), dim=0)[:max_size]

    new_dataset = SimpleDataset(x=new_x, y=new_y)

    dataloader = DataLoader(new_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=data_loader.num_workers,
                            pin_memory=data_loader.pin_memory,
                            drop_last=data_loader.drop_last)

    return dataloader


def filter_graph_nodewise(data, keep_index=None, delete_index=None):
    """
    Function to get subgraph of data. Effectively filtering by nodes.

    NOTE this does not work because of our custom data structure: it calls
    ``data.subgraph(keep_index)``, which the MXtalTools data classes do not define.

    Parameters
    ----------
    data : torch_geometric data object
        Graph or batch to filter. If ``data.edge_index`` is None it is set in place to
        ``torch.arange(2)`` before the call.
    keep_index : bool mask or sequence of int, optional
        Nodes to keep.
    delete_index : sequence of int, optional
        Nodes to drop; used only when ``keep_index`` is None, in which case
        ``keep_index`` becomes every index in ``range(len(data))`` not in it.

    Returns
    -------
    object
        The result of ``data.subgraph(keep_index)``.

    Raises
    ------
    AssertionError
        If both ``keep_index`` and ``delete_index`` are None.
    """
    assert keep_index is not None or delete_index is not None
    if keep_index is None and delete_index is not None:
        keep_index = [ind for ind in range(len(data)) if ind not in delete_index]

    if data.edge_index is None:
        data.edge_index = torch.arange(2)  # necessary dummy
    return data.subgraph(keep_index)


def basic_stats(values: torch.tensor) -> dict[str, Tensor]:
    """
    Summary statistics of a tensor.

    Parameters
    ----------
    values : torch.Tensor
        Values to summarise.

    Returns
    -------
    dict
        ``'max'``, ``'min'``, ``'mean'`` and ``'std'`` of ``values`` (mean and std in
        float); ``'tight_mean'`` and ``'tight_std'`` of ``values`` clamped to its 5th and
        95th percentiles, which are computed from the first ``int(16e6)`` entries along
        dim 0; ``'histogram'``, the ``torch.histogram`` of ``values`` with 50 bins; and
        ``'uniques'``, ``torch.unique(values, return_counts=True)`` for long tensors and
        ``(None, None)`` otherwise.
    """
    clipped_values = values.clip(min=torch.quantile(values[:int(16e6)].float(), 0.05),
                                 max=torch.quantile(values[:int(16e6)].float(), 0.95))

    return {'max': torch.amax(values),
            'min': torch.amin(values),
            'mean': torch.mean(values.float()),
            'std': torch.std(values.float()),
            'tight_mean': torch.mean(clipped_values),
            'tight_std': torch.std(clipped_values),
            'histogram': torch.histogram(values.float(), bins=50),
            'uniques': torch.unique(values, return_counts=True) if values.dtype == torch.long else (None, None),
            }


def get_dataloaders(dataset_builder, machine, batch_size, test_fraction=0.2,
                    shuffle=True,
                    num_workers: int = 0):
    """
    Split an indexable dataset into train and test ``MXtalDataLoader`` s (``torch_geometric``
    DataLoaders that fill the unit map of rows lacking it beside rows that carry it).

    The first ``len(dataset_builder) - int((1 - test_fraction) * len(dataset_builder))``
    items form the test set and the remaining items the training set; the split is by
    position, not random. Both loaders use ``pin_memory=True`` and ``drop_last=False``.

    Parameters
    ----------
    dataset_builder : indexable sized dataset
        Source of the samples.
    machine : str
        ``'cluster'`` uses ``num_workers`` workers; any other value uses 0.
    batch_size : int
    test_fraction : float
        Fraction of the items placed in the test set. Default 0.2.
    shuffle : bool
        Passed to both loaders.
    num_workers : int
        Worker count, used only when ``machine == 'cluster'``.

    Returns
    -------
    tuple
        ``(train_loader, test_loader)``; ``train_loader`` is None when the training set
        is empty.
    """
    batch_size = batch_size
    train_size = int((1 - test_fraction) * len(dataset_builder))  # split data into training and test sets
    test_size = len(dataset_builder) - train_size

    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size):
        train_dataset.append(dataset_builder[i])
    for i in range(test_size):
        test_dataset.append(dataset_builder[i])

    if machine == 'cluster':  # faster dataloading on cluster with more workers
        if len(train_dataset) > 0:
            tr = MXtalDataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                 pin_memory=True, drop_last=False)
        else:
            tr = None
        te = MXtalDataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             pin_memory=True, drop_last=False)
    else:
        if len(train_dataset) > 0:
            tr = MXtalDataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                                 pin_memory=True, drop_last=False)
        else:
            tr = None
        te = MXtalDataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True,
                             drop_last=False)

    return tr, te


def update_dataloader_batch_size(loader, new_batch_size):
    """
    Return a new ``MXtalDataLoader`` (``torch_geometric``'s DataLoader, filling the unit map
    of rows that lack it beside rows that carry it) over ``loader.dataset`` with
    ``new_batch_size``.

    The new loader keeps ``loader``'s ``num_workers``, ``pin_memory`` and ``drop_last``
    and always uses ``shuffle=True``.
    """
    return MXtalDataLoader(loader.dataset,
                           batch_size=new_batch_size,
                           shuffle=True,
                           num_workers=loader.num_workers,
                           pin_memory=loader.pin_memory,
                           drop_last=loader.drop_last)


class SimpleDataset(Dataset):
    """
    Map-style dataset over paired ``x`` and ``y`` rows.

    Both inputs are converted with ``torch.tensor(..., dtype=torch.float32)``.
    ``len`` is ``len(x)`` and item ``idx`` is ``(x[idx], y[idx])``.
    """
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

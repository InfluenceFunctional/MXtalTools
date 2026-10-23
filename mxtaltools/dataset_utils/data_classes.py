"""
Graph data objects for molecules and molecular crystals.

``MXtalBase`` is a single-store ``torch_geometric`` ``BaseData`` subclass carrying the
collation rules and batch-manipulation helpers shared by both data classes.
``MolData`` holds one molecule, and ``MolCrystalData`` adds space-group, unit-cell and
asymmetric-unit parameters to it. Lists of either are collated with
:func:`mxtaltools.dataset_utils.utils.collate_data_list`, which returns a ``Batch``
subclass of the element class (e.g. ``MolCrystalDataBatch``); ``is_batch`` is True on
those objects.
"""
import copy
from collections.abc import Mapping, Sequence
from typing import (Any, Dict, Iterable, List, NamedTuple, Optional, Union)

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                          GlobalStorage, NodeStorage)
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor

from mxtaltools.constants.space_group_info import SPACE_GROUP_NAMES
from mxtaltools.crystal_building.utils import UNIT_MAP_KEY, unit_layout, refuse_unit_map_disagreeing_with_z_prime, \
    derived_unit_ind
from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import MolCrystalAnalysis
from mxtaltools.dataset_utils.data_class_methods.crystal_building import MolCrystalBuilding
from mxtaltools.dataset_utils.data_class_methods.crystal_chem import MolCrystalChem
from mxtaltools.dataset_utils.data_class_methods.crystal_io import MolCrystalIO
from mxtaltools.dataset_utils.data_class_methods.crystal_ops import MolCrystalOps
from mxtaltools.dataset_utils.data_class_methods.crystal_recipes import MolCrystalRecipes
from mxtaltools.dataset_utils.data_class_methods.crystal_store import MolCrystalStore
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import MolCrystalSteps
from mxtaltools.dataset_utils.data_class_methods.ellipsoid_ops import MolCrystalEllipsoidOps
from mxtaltools.dataset_utils.data_class_methods.conformer_methods import MolConformerMethods
from mxtaltools.dataset_utils.data_class_methods.mol_methods import MolDataMethods
from mxtaltools.dataset_utils.utils import collate_data_list


###############################################################################
# the level contract of the batch helpers (stage 3 of the units plan, 2026-10-23)
###############################################################################

#: The levels a stored value of a batch has (``MXtalBase``'s Notes: how each level is sliced, gathered, appended
#: and recorded, and how a value declares its level).
ATOM = 'atom'      #: one entry per atom along dim 0, sliced by ``ptr``
UNIT = 'unit'      #: one entry per unit (ragged), along dim 0, sliced by the unit pointer of ``z_prime``
EDGE = 'edge'      #: per row a segment of its own length (bonds, angles, torsions...), sliced by its recorded counts
ROW = 'row'        #: one entry per row (crystal), incl. the padded [B, max_z_prime, ...] per-unit slots
BATCH = 'batch'    #: batch-level / constant: carried whole, no slice entry
LEVELS = (ATOM, UNIT, EDGE, ROW, BATCH)
#: A registered level the helpers do not slice: per unit-cell atom (``build_unit_cell``'s output, laid out
#: ``[unit][image][atom]``). Every helper that moves values refuses it; rebuild the unit cell on the result.
UNIT_CELL = 'unit_cell'

#: The level registry: stored keys whose level is fixed by name, whatever their length or bookkeeping. A key here
#: takes this level before any other rule (``MXtalBase._level_plan``), and a value that does not fit it is refused.
#: Extend it with ``declare_level`` (process-wide), not by editing callers.
KEY_LEVELS = {
    # the atom -> unit map of a crystal row (crystal_building.utils.UNIT_MAP_KEY)
    UNIT_MAP_KEY: ATOM,
    # python int: the batch's pose-slot count
    'max_z_prime': BATCH,
    # space-group lookup tables (231 rows, indexed by space-group number) that crystal_ops attaches lazily: constant,
    # whatever the batch's row or atom count (a 231-row batch had them gathered as rows by the length rule)
    'asym_unit_lut': BATCH, 'sym_mult_lut': BATCH, 'free_axis_lut': BATCH, 'asym_unit_dict': BATCH,
    # build_conformer_tree's per-angle / per-torsion / per-closure values (no 'index' in their names, so collation
    # concatenates them on dim 0 without offset): per edge, sliced by the counts their slice entry records
    'tree_angle_is_linear': EDGE, 'tree_torsion_is_proper': EDGE, 'tree_torsion_frame_is_linear': EDGE,
    'tree_closure_r0': EDGE,
    # the unit cell: per unit-cell atom, a level the helpers do not slice
    'unit_cell_pos': UNIT_CELL, 'unit_cell_batch': UNIT_CELL, 'unit_cell_mol_ind': UNIT_CELL,
}


class BatchLevelError(ValueError):
    """A stored value of a batch that the batch helpers cannot classify, or cannot move at its level (the level
    contract: ``MXtalBase``'s Notes). Raised by ``raise`` statements, so it holds under ``python -O``."""


def declare_level(key: str, level: str):
    """Register the level of a stored key, process-wide (``KEY_LEVELS``).

    For a value whose bookkeeping cannot carry its level: a per-unit or per-edge value set on a batch without a
    slice entry, a constant whose length may equal the batch's atom, row or unit count, or a per-unit-cell value.
    ``level`` is one of ``LEVELS`` or ``UNIT_CELL``. Re-declaring a key at another level is refused (one key, one
    level).
    """
    if level not in LEVELS + (UNIT_CELL,):
        raise BatchLevelError(f'declare_level: {level!r} is not a level; one of {LEVELS + (UNIT_CELL,)}')
    if not isinstance(key, str) or not key or key in ('ptr', 'batch'):
        raise BatchLevelError(f'declare_level: {key!r} cannot be declared (ptr and batch are rebuilt by every helper)')
    known = KEY_LEVELS.get(key)
    if known is not None and known != level:
        raise BatchLevelError(f'declare_level: {key!r} is registered at level {known!r}; it cannot also be {level!r}')
    KEY_LEVELS[key] = level


def _offset_by_atoms(key: str) -> bool:
    """Whether collation offsets ``key``'s values by the atom counts: an 'index' / 'face' key (``MXtalBase.__inc__``;
    a key that also holds 'batch' is offset by its maximum instead)."""
    return ('index' in key or 'face' in key) and 'batch' not in key


def _length_level(value, num_nodes: int, num_graphs: int):
    """The level the length rule gives a stored value: ``ATOM``, ``ROW`` or ``BATCH``.

    Tensors: 0-dim or length 1 along dim 0 -> ``BATCH``; ``size(0) == num_nodes`` -> ``ATOM``
    (tested first); ``size(0) == num_graphs`` -> ``ROW``; any other length -> ``BATCH``.
    Lists and tuples: ``ROW`` when their length is ``num_graphs``, else ``BATCH``. Anything
    else (ints, strings, dicts, arrays): ``BATCH``. The rule for values without a usable
    slice entry (``MXtalBase._level_plan``); it never gives ``UNIT`` or ``EDGE``.
    """
    if torch.is_tensor(value):
        if value.ndim == 0 or value.size(0) == 1:
            return BATCH
        if value.size(0) == num_nodes:
            return ATOM
        if value.size(0) == num_graphs:
            return ROW
        return BATCH
    if isinstance(value, (list, tuple)):
        return ROW if len(value) == num_graphs else BATCH
    return BATCH


def _level_fits(level, value, num_nodes: int, num_graphs: int) -> bool:
    """Whether ``level`` (``ATOM``, ``ROW`` or ``BATCH``) can describe ``value`` on a batch of these counts.

    ``BATCH`` fits anything; ``ATOM`` a tensor with ``num_nodes`` rows; ``ROW`` a tensor
    with ``num_graphs`` rows, or a list or tuple of ``num_graphs`` entries.
    """
    if level == BATCH:
        return True
    if level not in (ATOM, ROW):
        return False
    if torch.is_tensor(value):
        return value.ndim > 0 and value.size(0) == (num_nodes if level == ATOM else num_graphs)
    return level == ROW and isinstance(value, (list, tuple)) and len(value) == num_graphs


def _cat_axis(batch, key, value) -> int:
    """The tensor axis ``value`` is concatenated along on collation (``__cat_dim__``; -1 is its last axis)."""
    d = batch.__cat_dim__(key, value)
    return value.ndim - 1 if d == -1 else d


def _entry_pointer(batch, key, value, slices, num_graphs: int):
    """The host pointer [num_graphs + 1] of ``slices`` (``key``'s ``_slice_dict`` entry) when it DESCRIBES the tensor
    ``value``: one-dimensional, ``num_graphs + 1`` entries, starting at 0, non-decreasing, ending at ``value``'s size
    along its collation axis. Else None (no entry, or one left from before the value or the batch changed)."""
    if not torch.is_tensor(slices) or slices.ndim != 1 or slices.numel() != num_graphs + 1:
        return None
    if not torch.is_tensor(value) or value.ndim == 0:
        return None
    s = slices.detach().to('cpu').long()
    if int(s[0]) != 0 or int(s[-1]) != value.size(_cat_axis(batch, key, value)):
        return None
    if num_graphs and bool((s[1:] < s[:-1]).any()):
        return None
    return s


def _increments_are_atom_offsets(inc, ptr_cpu) -> bool:
    """Whether ``inc`` (an ``_inc_dict`` entry) is each row's atom offset ``ptr[:-1]``: what collation writes for an
    'index' / 'face' key (``MXtalBase.__inc__``), so its values are atom indices offset per row and can be re-offset."""
    if not torch.is_tensor(inc) or inc.numel() != len(ptr_cpu) - 1:
        return False
    return torch.equal(inc.detach().to('cpu').long().reshape(-1), ptr_cpu[:-1].long())


def _edge_pointer_from_values(value, ptr_cpu, num_nodes: int, num_graphs: int):
    """Per-row counts of an index tensor without a usable entry, from its values: column ``j`` belongs to the row of
    the atoms it indexes. None unless every value is an atom index of the batch, every column's indices lie in one
    row, and the columns are grouped by row in row order (as collation lays them out). Read only for a ``[k, E]``
    tensor with ``k >= 2`` (``MXtalBase._edge_pointer``): a one-dimensional or ``[1, E]`` value without an entry may
    as well hold one integer per row, which this would read as atom indices."""
    v = value.detach().to('cpu')
    if v.is_floating_point() or v.is_complex() or v.dtype == torch.bool:
        return None
    v = v.long().reshape(-1, value.shape[-1])
    E = v.shape[1]
    if E == 0:
        return torch.zeros(num_graphs + 1, dtype=torch.long)
    if bool((v < 0).any()) or bool((v >= num_nodes).any()):
        return None
    row_of_atom = torch.repeat_interleave(torch.arange(num_graphs), ptr_cpu[1:] - ptr_cpu[:-1],
                                          output_size=num_nodes)
    rows = row_of_atom[v]                                       # [k, E]
    if not bool((rows == rows[:1]).all()) or bool((rows[0, 1:] < rows[0, :-1]).any()):
        return None
    counts = torch.bincount(rows[0], minlength=num_graphs)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


def _segments(pointer, idx_cpu):
    """The selected rows' segments of a value sliced by ``pointer`` [B + 1] (host long), in ``idx_cpu`` order.

    Returns ``(src, new_pointer, seg)``: the source positions [total] along the value's axis, the pointer [k + 1] of
    the result, and the result row of each position [total] (all host long)."""
    lengths = pointer[idx_cpu + 1] - pointer[idx_cpu]
    new_pointer = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    total = int(new_pointer[-1])
    seg = torch.repeat_interleave(torch.arange(len(idx_cpu)), lengths, output_size=total)
    src = pointer[idx_cpu][seg] + torch.arange(total) - new_pointer[seg]
    return src, new_pointer, seg


def _gather_segments(batch, key, value, pointer, idx_cpu, old_atom_ptr, new_atom_ptr):
    """``value`` (a ``UNIT`` or ``EDGE`` tensor sliced by ``pointer``) for the rows ``idx_cpu``, and its bookkeeping.

    The selected rows' segments along the value's collation axis, in ``idx_cpu`` order; an 'index' / 'face' key's
    values (atom indices, offset per row by collation) re-offset from each row's old atom offset to its new one.
    Returns ``(value, (slice entry, increments))``: increments are the new atom offsets for an offset key (what
    collation writes, so ``batch_to_list`` gives row-local indices), zeros otherwise. Shared by
    ``subsample_new_batch`` and ``MolCrystalIO.select_rows``.
    """
    axis = _cat_axis(batch, key, value)
    src, new_pointer, seg = _segments(pointer, idx_cpu)
    out = value.index_select(axis, src.to(value.device))
    k = len(idx_cpu)
    if _offset_by_atoms(key):
        shift = new_atom_ptr[:-1][seg] - old_atom_ptr[idx_cpu][seg]
        shape = [1] * out.ndim
        shape[axis] = len(seg)
        out = out + shift.to(device=out.device, dtype=out.dtype).reshape(shape)
        inc = new_atom_ptr[:-1].clone()
    else:
        inc = torch.zeros(k, dtype=torch.long)
    return out, (new_pointer, inc)


def _unit_cell_refusal(key, caller):
    return BatchLevelError(f"{caller}: {key!r} is per unit-cell atom (build_unit_cell), a level the batch helpers do "
                           f"not slice; delete unit_cell_pos / unit_cell_batch / unit_cell_mol_ind before it and "
                           f"rebuild the unit cell (mol2ucell) on the result")


class _UnitPointer:
    """The batch's unit pointer ``[0, z'_0, z'_0 + z'_1, ...]`` (host long [B + 1]), read from ``z_prime`` on first
    use only (one host read of a device-resident ``z_prime``); None when the batch stores no ``z_prime`` of one value
    per row. ``z_prime < 0`` counts as no unit."""

    def __init__(self, batch, num_graphs: int):
        self._batch, self._B, self._value, self._done = batch, num_graphs, None, False

    def get(self):
        if not self._done:
            self._done = True
            zp = self._batch._store['z_prime'] if 'z_prime' in self._batch._store else None
            if torch.is_tensor(zp) and zp.numel() == self._B and not zp.is_floating_point():
                zp = zp.detach().to('cpu').reshape(-1).long().clamp(min=0)
                self._value = torch.cat([zp.new_zeros(1), zp.cumsum(0)])
        return self._value


def _levels_coincide(ptr_cpu, num_nodes: int, num_graphs: int) -> bool:
    """Whether every graph of the batch holds exactly one node (``ptr == arange(num_graphs + 1)``).

    On such a batch a per-node and a per-graph tensor have the same length, gather alike
    and get the same ``_slice_dict`` entry, so neither tells their levels apart:
    ``_level_plan`` files both per atom, which is harmless there but says nothing
    about the value elsewhere (``append_batch``). Other batches pay one integer
    comparison.
    """
    if num_nodes != num_graphs:
        return False
    return torch.equal(ptr_cpu.long(), torch.arange(num_graphs + 1, dtype=torch.long))


###############################################################################

# noinspection PyPropertyAccess


class MXtalBase(BaseData):
    """
    Single-store ``BaseData`` base class for MXtalTools graph data objects.

    All attributes live in one ``GlobalStorage`` (``self._store``). Attribute access,
    assignment and deletion are routed to it, except ``_num_graphs``, ``_slice_dict`` and
    ``_inc_dict``, which are read and written in the instance ``__dict__``.

    Notes
    -----
    Collation rules applied by ``Batch.from_data_list`` (``__cat_dim__`` / ``__inc__``):

    - a ``SparseTensor`` whose key contains ``'adj'`` is concatenated along ``(0, 1)``;
    - keys containing ``'index'`` or ``'face'`` are concatenated along the last dimension
      (``[k, n]`` layout, as ``edge_index``) and offset by ``num_nodes``;
    - keys containing ``'batch'`` are concatenated along dim 0 and offset by
      ``int(value.max()) + 1``;
    - every other key is concatenated along dim 0 without offset.

    The level contract (stage 3 of the units plan, 2026-10-23). The batch helpers
    (``subsample_new_batch``, ``append_batch``, ``rebuild_simple_slice_inc_``,
    ``batch_to_list``; ``MolCrystalIO.select_rows``; the store's level map) give every
    stored value except ``ptr`` and ``batch`` (rebuilt) one level (``_level_plan``):

    - ``ATOM``: one entry per atom along dim 0, sliced by ``ptr``;
    - ``UNIT``: one entry per unit along dim 0 (ragged: ``sum(z_prime)`` entries), sliced
      by the unit pointer ``[0, z'_0, z'_0 + z'_1, ...]`` of ``z_prime``;
    - ``EDGE``: per row a segment of its own length (bonds, angles, torsions, ...) along
      the value's collation axis (the last axis for an 'index' / 'face' key, dim 0
      otherwise), sliced by the per-row counts its slice entry records; an 'index' /
      'face' key's values are atom indices, offset by the atom counts as collation offsets
      them (``__inc__``), and re-offset by every helper;
    - ``ROW``: one entry per row (crystal) along dim 0, a list or tuple of one entry per
      row included; the padded per-unit slots ``[B, max_z_prime, ...]`` (``aunit_*``,
      ``unit_radius``, ...) are row values;
    - ``BATCH``: batch-level or constant, carried whole without a slice entry.

    A value gets its level from the first of these that applies:

    1. the registry ``KEY_LEVELS`` (``declare_level``): the level is fixed by name, and a
       value that does not fit it is refused (``BatchLevelError``); keys registered
       ``UNIT_CELL`` (the unit cell) are refused by every helper that moves values;
    2. the naming contract of collation: an 'index' / 'face' tensor key (concatenated on
       its last axis, ``__cat_dim__``) is ``EDGE``, never classified by its length; its
       per-row counts are its slice entry's, whose increments must then be the rows' atom
       offsets ``ptr[:-1]`` (what collation writes: its values are atom indices offset per
       row, which every helper re-offsets), or, without a usable entry, read from the
       values of a ``[k, E]`` tensor with ``k >= 2`` (the row of the atoms each column
       indexes, columns grouped by row); else it is refused (a one-dimensional index value
       without an entry, an index value attached with zero increments);
    3. a tensor whose slice entry describes it (``B + 1`` entries from 0, non-decreasing,
       ending at its length; collation writes one for every key it collates,
       ``add_unit_attr`` for a per-unit value set on a batch, the helpers keep them) gets
       the level the entry records, whatever its length coincides with: ``ATOM`` when the
       entry is ``ptr`` and the length the atom count (tested first), ``ROW`` when it is
       ``arange`` and the length the row count, else ``UNIT`` when it is the unit pointer
       of ``z_prime`` and ``EDGE`` otherwise (``_entry_level``; since the stage-3 review,
       2026-10-23, before the length: a ragged value whose total equalled the atom or row
       count was read per atom or per row). An entry held on a device (``add_graph_attr``
       / ``add_node_attr`` write them on the batch's device) is not read where the length
       is the atom or row count (no host sync on the hot path; ``_recorded_level`` reads
       it where that length is ambiguous): a ragged value must carry a host entry
       (collation's, ``add_unit_attr``'s) or be registered;
    4. the length, for a tensor without a usable entry: the atom count is ``ATOM``, the
       row count ``ROW`` (atom first);
    5. everything else is ``BATCH``: 0-dim tensors, python values, dicts, arrays, lists
       or tuples of another length than the row count, and tensors of another length
       without a usable entry (a constant table set on the batch, or a value whose entry
       went stale). Carried whole by ``subsample_new_batch`` / ``append_batch``, where
       ``batch_to_list`` refuses a tensor whose entry is stale: an accepted gap of the
       contract, pending the owner's decision on refusing such values (the
       characterization README, "The batch helpers' level contract").

    How a new attribute declares its level: collate it from rows (collation writes the
    slice entry), attach it on a batch with ``add_node_attr`` / ``add_graph_attr`` /
    ``add_unit_attr``, give an atom-index tensor an 'index' key (and the increments
    ``ptr[:-1]`` when it is attached on a batch with an entry), or register the key
    (``declare_level``). A ragged value set on a batch with neither a usable entry nor a
    registration is ``BATCH`` and carried whole when its length is another than the atom
    or row count, and read at that level when its total coincides with one (exact only
    where every row's segments are its atoms or its one entry, e.g. ``z_prime`` 1 on every
    row; otherwise give it an entry or register it). Values are never re-offset except
    'index' / 'face' keys; 'batch' keys (``aunit_batch``) keep the values collation gave
    them.
    """
    # PyG Batch bookkeeping that some methods here (subsample_new_batch,
    # append_batch, rebuild_simple_slice_inc_) already write straight into
    # instance.__dict__. Routing ordinary `self.attr = value` for these same
    # keys through self._store instead (the default below) would silently
    # write to a second, shadowed location - self._store lands them in
    # self._store.__dict__ (underscore-prefixed keys never enter the PyG
    # mapping), which instance.__dict__ always wins over. That split-brain
    # let a rebuilt batch "remember" a stale num_graphs/slice/inc from before
    # a subsample/clone. Keeping every write for these keys in
    # instance.__dict__ removes the second location entirely.
    _INSTANCE_DICT_ATTRS = frozenset({'_num_graphs', '_slice_dict', '_inc_dict'})

    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        if key in self._INSTANCE_DICT_ATTRS:
            self.__dict__[key] = value
        else:
            setattr(self._store, key, value)

    def __delattr__(self, key: str):
        if key in self._INSTANCE_DICT_ATTRS:
            self.__dict__.pop(key, None)
        else:
            delattr(self._store, key)

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        has_dict = any([isinstance(v, Mapping) for v in self._store.values()])

        if not has_dict:
            info = [size_repr(k, v) for k, v in self._store.items()]
            info = ', '.join(info)
            return f'{cls}({info})'
        else:
            info = [size_repr(k, v, indent=2) for k, v in self._store.items()]
            info = ',\n'.join(info)
            return f'{cls}(\n{info}\n)'

    def stores_as(self, data: 'Data'):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        """Return the stored attributes as a dict (``self._store.to_dict()``)."""
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        """Return the stored attributes as a namedtuple (``self._store.to_namedtuple()``)."""
        return self._store.to_namedtuple()

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]):
        r"""Construct an instance of this class from a dictionary, as ``cls(**mapping)``."""
        return cls(**mapping)

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the graph."""
        return self._store.num_node_features

    @property
    def device(self):
        """Device of ``self.z``; raises ``AttributeError`` when ``z`` is absent."""
        return self.z.device

    @property
    def is_batch(self):
        """True when the class name contains ``'Batch'``, as for collated objects."""
        return 'Batch' in self.__class__.__name__

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the graph.
        Alias for :py:attr:`~num_node_features`."""
        return self._store.num_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the graph."""
        return self._store.num_edge_features

    def __iter__(self) -> Iterable:
        r"""Iterates over all attributes in the data, yielding their attribute
        names and values."""
        for key, value in self._store.items():
            yield key, value

    def __call__(self, *args: List[str]) -> Iterable:
        r"""Iterates over all attributes :obj:`*args` in the data, yielding
        their attribute names and values.
        If :obj:`*args` is not given, will iterate over all attributes."""
        for key, value in self._store.items(*args):
            yield key, value

    @staticmethod
    def to_batch(data_list):
        """Collate ``data_list`` with :func:`collate_data_list` and its default arguments."""
        return collate_data_list(data_list)

    def add_graph_attr(self, values: torch.Tensor, name: str, slice_dict=None, inc_dict=None):
        """
        Attach a per-graph attribute to this Batch so it survives ``to_data_list()``.

        Sets ``self.<name> = values`` and registers ``self._slice_dict[name]`` and
        ``self._inc_dict[name]``. Modifies ``self`` in place.

        Parameters
        ----------
        values : torch.Tensor
            Per-graph values, [num_graphs, ...].
        name : str
            Attribute name to assign.
        slice_dict : torch.Tensor, optional
            Slice entry to register; defaults to ``arange(num_graphs + 1)`` on
            ``self.device``.
        inc_dict : torch.Tensor, optional
            Increment entry to register; defaults to long zeros [num_graphs] on
            ``self.device``.

        Returns
        -------
        MXtalBase
            ``self``.

        Raises
        ------
        TypeError
            If ``self`` is not a batch.
        AssertionError
            If ``values.size(0) != self.num_graphs``.
        """
        if not self.is_batch:
            raise TypeError("add_graph_attr only works on Batch objects")

        num_graphs = self.num_graphs
        assert values.size(0) == num_graphs, \
            f"{name} must have shape [num_graphs, ...], got {values.shape}"

        # Store attribute on the batch
        setattr(self, name, values)

        # Tell PyG how to split it back into Data objects
        if slice_dict is not None:
            self._slice_dict[name] = slice_dict
        else:
            self._slice_dict[name] = torch.arange(0, num_graphs + 1, 1, device=self.device)

        if inc_dict is not None:
            self._inc_dict[name] = inc_dict
        else:
            self._inc_dict[name] = torch.zeros(num_graphs, dtype=torch.long, device=self.device)

        return self

    @staticmethod
    def node_slice_dict(num_nodes_per_graph, device):
        """
        [0, n_0, n_0+n_1, ...], the _slice_dict entry for a node-wise attribute.

        Parameters
        ----------
        num_nodes_per_graph : torch.Tensor or sequence of int
            Node count of each graph, [num_graphs].
        device : torch.device or str
            Device of the returned tensor.

        Returns
        -------
        torch.Tensor
            Long tensor [num_graphs + 1] of cumulative node offsets, starting at 0.

        Notes
        -----
        Built on-device, and the reason matters: the previous form was
        ``torch.cumsum(torch.tensor([0] + list(num_nodes_per_graph)), dim=0)``
        and num_nodes_per_graph is normally num_atoms, a CUDA tensor, so list()
        yields one 0-dim CUDA tensor per graph and torch.tensor() then pulls each
        of them to the host separately -- num_graphs DEVICE SYNCS per call, paid
        once per node-wise key. Profiled 2026-09-22 on a conditional run at ~1500
        graphs it was 17.9% of main-thread time, the largest single entry, and
        Tensor.__iter__ was reached from nowhere else in the program. Verified
        bit-identical on uniform, ragged, n=1, single-graph and 2000-graph
        batches and for a plain python list; 20.5 ms -> 0.09 ms at 1500 graphs.
        """
        t = torch.as_tensor(num_nodes_per_graph, device=device, dtype=torch.long)
        return torch.cat([t.new_zeros(1), t.cumsum(0)])

    def add_node_attr(self, values: torch.Tensor, name: str, num_nodes_per_graph,
                      slice_dict=None, inc_dict=None):
        """
        Attach a per-node attribute to this Batch so it survives to_data_list().

        Sets ``self.<name> = values`` and registers ``self._slice_dict[name]`` and
        ``self._inc_dict[name]``. Modifies ``self`` in place and returns None. The length
        of ``values`` is not checked against ``num_nodes_per_graph``.

        slice_dict/inc_dict mirror add_graph_attr: both are identical for every
        node-wise key on a given batch, so a caller attaching several should build
        them once and pass them in rather than paying for them per key.

        Parameters
        ----------
        values : torch.Tensor
            Per-node values, [num_nodes, ...].
        name : str
            Attribute name to assign.
        num_nodes_per_graph : torch.Tensor or sequence of int
            Node count of each graph, [num_graphs]; used only to build the defaults.
        slice_dict : torch.Tensor, optional
            Slice entry; defaults to ``node_slice_dict(num_nodes_per_graph,
            values.device)``.
        inc_dict : torch.Tensor, optional
            Increment entry; defaults to long zeros [num_graphs] on ``values.device``.

        Raises
        ------
        TypeError
            If ``self`` is not a batch.
        """
        if not self.is_batch:
            raise TypeError("add_node_attr only works on Batch objects")

        # concat all node tensors already; now define slices
        setattr(self, name, values)

        if slice_dict is None:
            slice_dict = self.node_slice_dict(num_nodes_per_graph, values.device)
        self._slice_dict[name] = slice_dict

        if inc_dict is None:
            inc_dict = torch.zeros(len(num_nodes_per_graph),
                                   dtype=torch.long, device=values.device)
        self._inc_dict[name] = inc_dict

    def add_unit_attr(self, values: torch.Tensor, name: str):
        """
        Attach a per-unit value to this crystal batch: one entry per unit, ``[sum(z_prime), ...]``, rows in order and
        each row's units in unit order (the ``UNIT`` level of the level contract).

        Sets ``self.<name> = values`` and registers ``self._slice_dict[name]`` = the unit pointer
        ``[0, z'_0, z'_0 + z'_1, ...]`` of ``z_prime`` and ``self._inc_dict[name]`` = zeros (host long), so
        ``subsample_new_batch``, ``append_batch``, ``select_rows`` and ``batch_to_list`` slice it by unit.

        Raises
        ------
        TypeError
            If ``self`` is not a batch.
        BatchLevelError (a ValueError)
            If the batch stores no ``z_prime`` with one value per row, ``values`` is not a tensor with a dim 0, or
            its dim 0 is not ``sum(z_prime)``. Explicit raises: they hold under ``python -O``.
        """
        if not self.is_batch:
            raise TypeError("add_unit_attr only works on Batch objects")
        unit_ptr = _UnitPointer(self, self.num_graphs).get()
        if unit_ptr is None:
            raise BatchLevelError(f'add_unit_attr({name!r}): the batch stores no integer z_prime with one value per '
                                  f'row ({self.num_graphs}), so it has no unit pointer')
        if not torch.is_tensor(values) or values.ndim == 0 or values.size(0) != int(unit_ptr[-1]):
            shape = tuple(values.shape) if torch.is_tensor(values) else type(values).__name__
            raise BatchLevelError(f'add_unit_attr({name!r}): needs one entry per unit, [{int(unit_ptr[-1])}, ...] '
                                  f'(sum of z_prime), got {shape}')
        setattr(self, name, values)
        self._slice_dict[name] = unit_ptr
        self._inc_dict[name] = torch.zeros(self.num_graphs, dtype=torch.long)
        return self

    def _ptr_cpu(self):
        """
        Host mirror of self.ptr, cached by tensor IDENTITY. Reading ptr values
        off a CUDA-resident batch (int(ptr[-1]), .cpu(), .tolist()) stalls the
        host on the whole CUDA stream -- and the stall costs whatever happens
        to be queued, which is what made fixed-size buffer churn take 10ms or
        2500ms at random. All ptr/size arithmetic goes through this mirror
        instead. ptr is only ever REBUILT, never mutated in place, so object
        identity captures staleness exactly; a missed cache is a redundant
        recompute, never stale data. subsample_new_batch/append_batch stamp
        mirrors for the ptr tensors they construct, so churned batches never
        pay the initial D2H at all.
        """
        ptr = self.ptr
        pair = self.__dict__.get('_ptr_cpu_pair')
        if pair is None or pair[0] is not ptr:
            pair = (ptr, ptr.detach().to('cpu').long())
            self.__dict__['_ptr_cpu_pair'] = pair
        return pair[1]

    def _stamp_ptr_cpu(self, ptr_dev, ptr_cpu):
        """Record a host mirror for a freshly built ptr (see _ptr_cpu)."""
        self.__dict__['_ptr_cpu_pair'] = (ptr_dev, ptr_cpu.long())

    def _level_plan(self, ptr_cpu, num_nodes, num_graphs, skip=()):
        """
        The level of every stored value except ``ptr``, ``batch`` and the keys in ``skip``: the level contract
        (class Notes), rules 1-5 in order.

        Returns a dict ``{key: (level, pointer)}``: ``level`` one of ``ATOM``, ``UNIT``, ``EDGE``, ``ROW``,
        ``BATCH`` or ``UNIT_CELL``; ``pointer`` the host long [num_graphs + 1] slicing it (``ptr_cpu`` for ``ATOM``,
        ``arange`` for ``ROW``, the unit pointer for ``UNIT``, the recorded counts for ``EDGE``), None for
        ``BATCH`` and ``UNIT_CELL``.

        Cost. A slice entry held on the host (collation, the helpers' rebuild and ``add_unit_attr`` write them
        there) is compared with ``ptr`` / ``arange`` once per entry object (``_entry_level``; the helpers' rebuild
        shares one object among all atom keys and one among all row keys), which needs no device sync. An entry
        held on a device (``add_graph_attr`` / ``add_node_attr`` write them on the batch's device) is read only where
        the value's length is neither the atom nor the row count, or is ambiguous (``_recorded_level``), as before
        stage 3's review: the hot path pays no host read of a device-resident entry. ``z_prime`` is read (once) only
        to test an entry against the unit pointer.

        Raises
        ------
        BatchLevelError (a ValueError)
            A registered key whose value does not fit its level; an 'index' / 'face' key whose per-row counts are
            unknown (no entry describes it and its values do not give them), whose entry describes it but whose
            increments are not its rows' atom offsets, or that also holds 'batch'.
        """
        slice_dict = self.__dict__.get('_slice_dict') or {}
        inc_dict = self.__dict__.get('_inc_dict') or {}
        graph_slices = torch.arange(num_graphs + 1, dtype=torch.long)
        units = _UnitPointer(self, num_graphs)
        seen = {}
        plan = {}
        for key, val in self._store.items():
            if key in ('ptr', 'batch') or key in skip:
                continue
            declared = KEY_LEVELS.get(key)
            if declared is not None:
                plan[key] = self._declared_level(key, val, declared, slice_dict.get(key), ptr_cpu, num_nodes,
                                                 num_graphs, graph_slices, units, inc_dict.get(key))
                continue
            if torch.is_tensor(val) and val.ndim > 0 and self.__cat_dim__(key, val) == -1:
                plan[key] = (EDGE, self._edge_pointer(key, val, slice_dict.get(key), ptr_cpu, num_nodes, num_graphs,
                                                      inc_dict.get(key)))
                continue
            level = _length_level(val, num_nodes, num_graphs)
            if torch.is_tensor(val) and val.ndim > 0:
                slices = slice_dict.get(key)
                on_device = torch.is_tensor(slices) and slices.device.type != 'cpu'
                if not on_device or val.size(0) not in (num_nodes, num_graphs):
                    # rule 3: an entry that describes the value decides its level, whatever its length coincides with
                    found = self._entry_level(key, val, slices, ptr_cpu, num_nodes, num_graphs, graph_slices, units,
                                              seen)
                    if found is not None:
                        plan[key] = found
                        continue
                elif val.size(0) == 1 or val.size(0) == num_nodes == num_graphs:
                    # a device-resident entry, read only where the length is ambiguous (as before the review)
                    recorded = self._recorded_level(key, val, slices, ptr_cpu, num_nodes, num_graphs)
                    if recorded is not None:
                        level = recorded
            plan[key] = (level, ptr_cpu if level == ATOM else graph_slices if level == ROW else None)
        return plan

    def _entry_level(self, key, val, slices, ptr_cpu, num_nodes, num_graphs, graph_slices, units, seen):
        """
        ``(level, pointer)`` that the slice entry ``slices`` of the dim-0 tensor ``val`` records, when it describes
        ``val``; else None (no entry, or a stale one: the length rule then decides). Rule 3 of the level contract.

        ``ATOM`` when the entry is ``ptr`` and ``val`` has ``num_nodes`` rows (tested first: where ``ptr`` is
        ``arange``, one atom per row, the two gathers coincide); ``ROW`` when it is ``arange(num_graphs + 1)`` and
        ``val`` has ``num_graphs`` rows; otherwise, when it describes ``val`` (``_entry_pointer``: ``num_graphs + 1``
        entries from 0, non-decreasing, ending at ``val``'s length), ``UNIT`` when it is the unit pointer of
        ``z_prime`` and ``EDGE`` otherwise, whatever ``val``'s length coincides with (since the stage-3 review,
        2026-10-23: a ragged value whose total equalled the atom or row count was sliced per atom or per row).
        ``seen`` memoises the result per (entry object, length).
        """
        if not torch.is_tensor(slices) or slices.ndim != 1 or slices.numel() != num_graphs + 1:
            return None
        length = val.size(0)
        memo = seen.get((id(slices), length))
        if memo is not None and memo[0] is slices:
            return memo[1]
        s = slices.detach().to('cpu').long()
        if length == num_nodes and (slices is ptr_cpu or torch.equal(s, ptr_cpu)):
            found = (ATOM, ptr_cpu)
        elif length == num_graphs and torch.equal(s, graph_slices):
            found = (ROW, graph_slices)
        else:
            pointer = _entry_pointer(self, key, val, s, num_graphs)
            if pointer is None:
                found = None
            else:
                unit_ptr = units.get()
                found = (UNIT if unit_ptr is not None and torch.equal(pointer, unit_ptr) else EDGE, pointer)
        seen[(id(slices), length)] = (slices, found)
        return found

    def _declared_level(self, key, val, declared, slices, ptr_cpu, num_nodes, num_graphs, graph_slices, units,
                        inc=None):
        """``(level, pointer)`` of a key registered in ``KEY_LEVELS``; refuses a value that does not fit it."""
        if declared in (BATCH, UNIT_CELL):
            return declared, None
        if declared == EDGE:
            if not (torch.is_tensor(val) and val.ndim > 0):
                raise BatchLevelError(f'{key!r} is registered per edge but holds a {type(val).__name__} '
                                      f'{tuple(val.shape) if torch.is_tensor(val) else ""}')
            return EDGE, self._edge_pointer(key, val, slices, ptr_cpu, num_nodes, num_graphs, inc)
        if declared == UNIT:
            unit_ptr = units.get()
            if unit_ptr is None:
                raise BatchLevelError(f'{key!r} is registered per unit, but the batch stores no integer z_prime with '
                                      f'one value per row ({num_graphs})')
            if torch.is_tensor(val) and val.ndim > 0 and val.size(0) == int(unit_ptr[-1]):
                return UNIT, unit_ptr
            want = f'[{int(unit_ptr[-1])}, ...] (sum of z_prime)'
        elif declared == ATOM:
            if torch.is_tensor(val) and val.ndim > 0 and val.size(0) == num_nodes:
                return ATOM, ptr_cpu
            want = f'[{num_nodes}, ...] (the atom count)'
        else:   # ROW
            if _level_fits(ROW, val, num_nodes, num_graphs):
                return ROW, graph_slices
            want = f'[{num_graphs}, ...] (the row count)'
        got = tuple(val.shape) if torch.is_tensor(val) else (f'{type(val).__name__} of {len(val)}'
                                                             if isinstance(val, (list, tuple)) else type(val).__name__)
        raise BatchLevelError(f'{key!r} is registered at level {declared!r} (KEY_LEVELS) and needs {want}, but '
                              f'holds {got}; a stale value (its rows or atoms changed without it) is refused rather '
                              f'than carried whole')

    def _edge_pointer(self, key, val, slices, ptr_cpu, num_nodes, num_graphs, inc=None):
        """The per-row counts (host long pointer [num_graphs + 1]) of an ``EDGE`` tensor: its slice entry where it
        describes the value; for an 'index' / 'face' key without one, from its values (``_edge_pointer_from_values``,
        a ``[k, E]`` tensor with ``k >= 2`` only). Refused otherwise.

        An 'index' / 'face' key's values are atom indices that the helpers re-offset from each row's old atom offset
        to its new one, which is right only where they hold those offsets: its increments ``inc`` must be the rows'
        atom offsets ``ptr[:-1]``, as collation writes them (``MXtalBase.__inc__``), or it is refused, as
        ``MolCrystalIO.select_rows`` refuses it (since the stage-3 review, 2026-10-23: a per-row value attached with
        ``add_graph_attr``, increments zero, was shifted by the atom offsets, ``[9]`` becoming ``[-9]``). Values whose
        counts are read from the values are atom indices of the batch by that reading."""
        if 'batch' in key and self.__cat_dim__(key, val) == -1:
            raise BatchLevelError(f'{key!r} is concatenated on its last axis and offset by its maximum on collation '
                                  f"('index' / 'face' and 'batch' in one name), which the batch helpers do not "
                                  f'reproduce; rename it')
        pointer = _entry_pointer(self, key, val, slices, num_graphs)
        if pointer is not None:
            if _offset_by_atoms(key) and not _increments_are_atom_offsets(inc, ptr_cpu):
                got = tuple(inc.shape) if torch.is_tensor(inc) and inc.numel() != len(ptr_cpu) - 1 else \
                    (inc.detach().to('cpu').reshape(-1)[:6].tolist() if torch.is_tensor(inc) else inc)
                raise BatchLevelError(f"{key!r} {tuple(val.shape)} is an 'index' / 'face' key (atom indices, offset "
                                      f"per row by collation), but its increments are not its rows' atom offsets "
                                      f"ptr[:-1] (got {got}), so its values cannot be re-offset; collate it from "
                                      f"rows, give it the increments ptr[:-1] if its values are batch atom indices, or "
                                      f"rename it if they are not atom indices")
            return pointer
        if _offset_by_atoms(key) and val.ndim >= 2 and val.size(0) >= 2:
            pointer = _edge_pointer_from_values(val, ptr_cpu, num_nodes, num_graphs)
            if pointer is not None:
                return pointer
        if _offset_by_atoms(key):
            why = (' and its values are not atom indices grouped by row' if val.ndim >= 2 and val.size(0) >= 2 else
                   f' (a {tuple(val.shape)} value without one is not read as atom indices: it may hold one integer '
                   f'per row)')
        else:
            why = ''
        raise BatchLevelError(f'{key!r} {tuple(val.shape)} is per edge (concatenated along its '
                              f'{"last" if self.__cat_dim__(key, val) == -1 else "first"} axis), but its per-row '
                              f'counts are unknown: no slice entry describes it' + why
                              + '; collate it from rows, or set its _slice_dict entry')

    def _attribute_levels(self, ptr_cpu, num_nodes, num_graphs):
        """``{key: level}`` of every stored value except ``ptr`` and ``batch`` (``_level_plan`` without the
        pointers)."""
        return {key: level for key, (level, _) in self._level_plan(ptr_cpu, num_nodes, num_graphs).items()}

    def _recorded_level(self, key, val, slices, ptr_cpu, num_nodes, num_graphs):
        """
        ``ATOM`` or ``ROW`` when ``slices`` (the ``_slice_dict`` entry of ``key``) records
        that level for the tensor ``val`` on this batch, else None. Read only where the
        value's length is ambiguous and its entry is held on a device (``_level_plan``; a
        host-held entry is read by ``_entry_level``).

        Usable: a one-dimensional tensor of ``num_graphs + 1`` entries, for a value
        concatenated along dim 0 (``__cat_dim__``; not an ``*index`` / ``*face`` key),
        equal to ``ptr`` with ``val`` holding ``num_nodes`` rows (``ATOM``, tested first),
        or equal to ``arange(num_graphs + 1)`` with ``val`` holding ``num_graphs`` rows
        (``ROW``). When ``ptr == arange`` (one node per graph, ``_levels_coincide``) the
        entry is both and ``ATOM`` is returned: the two gathers coincide on this batch, but
        the level says nothing about the value outside it. Any other entry is not used
        here (``_level_plan`` reads it as ``UNIT`` / ``EDGE`` where it describes the
        value). Increments are not consulted: they say how collation offset the VALUES
        (``aunit_batch``), not how the value is sliced.
        """
        if not torch.is_tensor(slices) or slices.ndim != 1 or slices.numel() != num_graphs + 1:
            return None
        if self.__cat_dim__(key, val) != 0:
            return None
        s = slices.detach().to('cpu').long()
        if val.size(0) == num_nodes and torch.equal(s, ptr_cpu):
            return ATOM
        if val.size(0) == num_graphs and torch.equal(s, torch.arange(num_graphs + 1)):
            return ROW
        return None

    def rebuild_simple_slice_inc_(self, ptr_cpu=None, num_nodes=None, levels=None, entries=None):
        """
        Rebuild PyG batch bookkeeping from the levels of the stored values.

        Replaces ``self.__dict__['_slice_dict']`` and ``self.__dict__['_inc_dict']`` in
        place so that ``to_data_list()`` can split the batch again. ``ptr`` and ``batch``
        get no entry. Each other stored value gets its entry from ``entries`` when given
        there, else its level from ``levels`` when given there and consistent with the
        value, else from ``_level_plan`` (the level contract, class Notes), and then:

          - ``ATOM``: sliced by ``ptr_cpu``;
          - ``ROW``: sliced by ``arange(g + 1)``;
          - ``UNIT`` / ``EDGE``: sliced by its pointer (the unit pointer, or the counts
            its own entry records);
          - ``BATCH`` (ints, strings, dicts, 0-dim tensors, tensors and lists of any other
            length without a usable entry, length-1 tensors without a recorded level) and
            ``UNIT_CELL``: no entry.

        So a batch whose bookkeeping records its levels keeps them through a rebuild,
        including the per-graph tensors of a one-graph batch and the per-graph tensors
        of a batch with as many nodes as graphs, which the length alone files as shared
        and as per-node. Increments are zero (values are not re-offset), except an
        'index' / 'face' key's: each row's atom offset, ``ptr_cpu[:-1]``, as collation
        writes them, so ``to_data_list()`` gives each row its own atom indices.
        _slice_dict/_inc_dict stay on CPU.

        Parameters
        ----------
        ptr_cpu : torch.Tensor, optional
            Host copy of ``self.ptr`` [num_graphs + 1]; defaults to ``self._ptr_cpu()``.
        num_nodes : int, optional
            Total node count ``n``; defaults to ``int(ptr_cpu[-1])``.
            ptr_cpu/num_nodes, when the caller already has them host-side (see
            _ptr_cpu), skip the device sync this used to pay on CUDA batches.
        levels : dict, optional
            ``{key: ATOM | ROW}``: levels the caller knows, e.g. those of
            the batch it gathered ``self`` from (``subsample_new_batch``,
            ``append_batch``), whose bookkeeping ``self`` does not have. A level that
            does not fit the stored value (``ATOM`` on a tensor without ``n`` rows,
            ``ROW`` on a value without ``g`` rows), or that is given for a value
            ``_level_plan`` files ``UNIT`` / ``EDGE`` / ``UNIT_CELL``, is ignored.
        entries : dict, optional
            ``{key: (slice entry, increments)}`` the caller built (the ``UNIT`` /
            ``EDGE`` values it gathered or joined); written as given, not classified.

        Returns
        -------
        MXtalBase
            ``self``.

        Raises
        ------
        AssertionError
            If ``self`` is not a batch.
        BatchLevelError (a ValueError)
            From ``_level_plan``, for a value it refuses.
        """
        assert self.is_batch

        g = self.num_graphs
        if ptr_cpu is None:
            ptr_cpu = self._ptr_cpu()
        ptr_cpu = ptr_cpu.long()
        n = int(ptr_cpu[-1]) if num_nodes is None else num_nodes
        graph_slices = torch.arange(g + 1, dtype=torch.long, device='cpu')
        graph_inc = torch.zeros(g, dtype=torch.long, device='cpu')
        entries = entries or {}

        slice_dict = {}
        inc_dict = {}

        own = self._level_plan(ptr_cpu, n, g, skip=entries)
        for key in self._store.keys():
            if key in entries:
                slice_dict[key], inc_dict[key] = entries[key]
                continue
            if key not in own:
                continue
            level, pointer = own[key]
            if levels is not None and key in levels and level in (ATOM, ROW, BATCH) and \
                    levels[key] in (ATOM, ROW) and _level_fits(levels[key], self._store[key], n, g):
                level = levels[key]
            if level == ATOM:
                slice_dict[key] = ptr_cpu
                inc_dict[key] = graph_inc
            elif level == ROW:
                slice_dict[key] = graph_slices
                inc_dict[key] = graph_inc
            elif level in (UNIT, EDGE):
                slice_dict[key] = pointer
                inc_dict[key] = ptr_cpu[:-1].clone() if level == EDGE and _offset_by_atoms(key) else graph_inc

        self.__dict__['_slice_dict'] = slice_dict
        self.__dict__['_inc_dict'] = inc_dict

        return self

    def batch_to_list(self):
        """
        Split a crystal batch into a list of single data objects.

        Wraps PyG's default ``to_data_list`` with custom handling of ``max_z_prime`` and
        ``is_well_defined``: both are removed from a clone of ``self`` before splitting,
        then every element gets ``max_z_prime = int(self.max_z_prime)`` and
        ``is_well_defined = self.is_well_defined[i].unsqueeze(0)`` (shape [1]). ``self``
        is not modified.

        Every other value is split at the level its ``_slice_dict`` entry records (PyG's
        ``separate``: tensors narrowed by the entry along their collation axis and
        decremented by their increments, lists indexed by row); a value without an entry
        is not carried onto the elements. The batch helpers keep the entries
        (``rebuild_simple_slice_inc_``), so a one-graph batch, or one with as many atoms
        as graphs, splits correctly when its bookkeeping records its levels, and so do
        ``UNIT`` values (by the unit pointer) and ``EDGE`` values (by their counts; an
        'index' / 'face' key's increments are its rows' atom offsets, so every element
        gets its own atom indices).

        Returns
        -------
        list
            One data object per graph, in batch order.

        Raises
        ------
        AssertionError
            If ``self`` is not a batch.
        AttributeError
            If ``max_z_prime`` or ``is_well_defined`` is absent.
        BatchLevelError (a ValueError)
            If a tensor's slice entry does not describe it (``num_graphs + 1`` entries from
            0, non-decreasing, ending at its size along its collation axis): PyG would cut
            it at stale offsets, silently or not (since stage 3, 2026-10-23).
        """
        assert self.is_batch
        for key, slices in (self.__dict__.get('_slice_dict') or {}).items():
            value = self._store[key] if key in self._store else None
            if torch.is_tensor(value) and value.ndim > 0 and key not in ('max_z_prime', 'is_well_defined') and \
                    _entry_pointer(self, key, value, slices, self.num_graphs) is None:
                raise BatchLevelError(f'batch_to_list: the slice entry of {key!r} {tuple(value.shape)} does not '
                                      f'describe it (stale bookkeeping: rebuild_simple_slice_inc_, or a helper, '
                                      f'after changing the value)')
        zp = int(self.max_z_prime)
        is_well_defined = self.is_well_defined.clone()
        batch_to_listify = self.clone()
        del batch_to_listify.max_z_prime, batch_to_listify.is_well_defined

        samples_list = batch_to_listify.to_data_list()
        for ind, elem in enumerate(samples_list):
            elem.max_z_prime = zp
            elem.is_well_defined = is_well_defined[ind].unsqueeze(0)
        del batch_to_listify
        return samples_list

    def subsample_new_batch(self, idx):
        """
        Return a new batch holding the graphs selected by ``idx``, in ``idx`` order.

        ``ptr`` and ``batch`` are rebuilt for the selection. Every other stored value is
        handled by its level (``_level_plan``, the level contract of the class Notes):

        - ``ATOM``: atoms of the selected graphs gathered;
        - ``ROW``: rows ``idx`` gathered (a list or tuple keeps its type);
        - ``UNIT``: the selected rows' units gathered (by the unit pointer of ``z_prime``);
        - ``EDGE``: the selected rows' segments gathered along the value's collation axis
          (by the counts its slice entry records); an 'index' / 'face' key's values
          (atom indices) re-offset to the selected rows' new atom offsets (since stage 3,
          2026-10-23: such a key was gathered by row on a 2-graph batch, [2, E] becoming
          [1, E], and carried whole and un-offset otherwise);
        - ``BATCH``: copied by reference.

        On a batch whose lengths are unambiguous (``num_graphs > 1``, ``num_nodes > 1``,
        ``num_nodes != num_graphs``) and that holds no ragged or 'index' / 'face' value
        (every tensor's entry, where it has one, is ``ptr`` or ``arange``), this is the
        length rule exactly, as before stage 3. The slice/increment
        bookkeeping of the result is rebuilt with ``rebuild_simple_slice_inc_``, given the
        atom and row levels used here and the pointers of the ``UNIT`` / ``EDGE`` values,
        so the result records them even where its own lengths are ambiguous (one selected
        graph; as many selected nodes as graphs). A ``BATCH`` value gets the length rule
        on the result, as before. ``self`` is not modified apart from its cached host copy
        of ``ptr``.

        Parameters
        ----------
        idx : slice, int, torch.Tensor, np.ndarray or sequence of int
            Graphs to keep; integer indices may repeat. Only a bool ``torch.Tensor`` is
            treated as a mask; a bool numpy array or list is converted to integer indices
            0/1. A device tensor is copied to the host. Integer indices are used as given,
            not normalised: ``ptr`` has ``num_graphs + 1`` entries, so a negative index
            either raises or pairs the node range of one graph with the per-graph rows of
            another.

        Returns
        -------
        MXtalBase
            New object of the same class holding the selected graphs, on
            ``self.device``.

        Raises
        ------
        BatchLevelError (a ValueError)
            A value ``_level_plan`` refuses, or a ``UNIT_CELL`` value (the unit cell:
            rebuild it on the result).
        """
        device = self.device

        # Normalize idx on the HOST first: every size/pointer quantity below is
        # computed against the host ptr mirror (_ptr_cpu), so a CUDA-resident
        # batch pays NO stream sync here -- the previous device-side path hid
        # three (int(ptr[-1]) x2 and repeat_interleave's data-dependent output
        # size), each of which stalled the host on everything queued ahead.
        # Only H2D copies remain, and those don't wait on the stream.
        if isinstance(idx, slice):
            idx_cpu = torch.arange(*idx.indices(self.num_graphs), dtype=torch.long)
        elif isinstance(idx, int):
            idx_cpu = torch.tensor([idx], dtype=torch.long)
        elif torch.is_tensor(idx):
            if idx.dtype == torch.bool:
                idx = idx.nonzero(as_tuple=False)
            # device-resident idx is the one input that still costs a sync (its
            # values must come to the host); buffer callers pass numpy/cpu
            idx_cpu = idx.detach().reshape(-1).to(device='cpu', dtype=torch.long)
        else:  # np.ndarray / list / other sequence
            idx_cpu = torch.as_tensor(idx, dtype=torch.long).reshape(-1)

        # host list indices for python list/tuple attrs, built lazily (tolist
        # on the host mirror is free; tensor-only stores skip it entirely)
        idx_list = None

        k = idx_cpu.numel()

        ptr_cpu = self._ptr_cpu()
        lengths_cpu = ptr_cpu[idx_cpu + 1] - ptr_cpu[idx_cpu]
        new_ptr_cpu = torch.cat([lengths_cpu.new_zeros(1), lengths_cpu.cumsum(0)])

        total_nodes = int(new_ptr_cpu[-1])
        old_num_nodes = int(ptr_cpu[-1])
        old_num_graphs = self.num_graphs

        old_ptr = self.ptr
        if torch.device(device).type == 'cuda':
            # one PINNED staging tensor + non_blocking for all three uploads:
            # pageable H2D does a stream synchronize before copying (same stall
            # as a D2H read), pinned + non_blocking is truly async. torch's
            # caching host allocator event-tracks the pinned block, so its
            # reuse is deferred until the copy lands.
            packed = torch.cat([idx_cpu, lengths_cpu, new_ptr_cpu]).pin_memory()
            packed_dev = packed.to(device, non_blocking=True)
            idx = packed_dev[:k]
            lengths = packed_dev[k:2 * k]
            # ptr persists on the returned batch: clone off the staging storage
            new_ptr = packed_dev[2 * k:].clone()
        else:
            # cpu: all no-ops; any other device: plain moves, as before
            idx = idx_cpu.to(device)
            lengths = lengths_cpu.to(device)
            new_ptr = new_ptr_cpu.to(device)

        seg = torch.repeat_interleave(
            torch.arange(k, device=device, dtype=torch.long),
            lengths,
            output_size=total_nodes,  # host-precomputed: no device sync
        )

        node_src = old_ptr[idx][seg] + (
                torch.arange(total_nodes, device=device) - new_ptr[seg]
        )

        out = self.__class__.__new__(self.__class__)
        out.__dict__['_store'] = GlobalStorage(_parent=out)

        def index_tensor_first_dim(x, index):
            return x[index.to(x.device)]

        # one level per value (the level contract), from self's bookkeeping where the length does not decide it
        plan = self._level_plan(ptr_cpu, old_num_nodes, old_num_graphs)
        entries = {}
        for key, val in self._store.items():
            if key in ('ptr', 'batch'):
                continue
            level, pointer = plan[key]

            if level == ATOM:
                # Atom-level tensor (only tensors get this level).
                out[key] = index_tensor_first_dim(val, node_src)

            elif level == ROW:
                if torch.is_tensor(val):
                    out[key] = index_tensor_first_dim(val, idx)
                else:
                    # Graph-level Python attr: identifier, symmetry_operators, etc.
                    if idx_list is None:
                        idx_list = idx_cpu.tolist()
                    out[key] = tuple(val[i] for i in idx_list) if isinstance(val, tuple) \
                        else [val[i] for i in idx_list]

            elif level in (UNIT, EDGE):
                # ragged: the selected rows' segments (an index key re-offset to the new atom offsets)
                out[key], entries[key] = _gather_segments(self, key, val, pointer, idx_cpu, ptr_cpu, new_ptr_cpu)

            elif level == UNIT_CELL:
                raise _unit_cell_refusal(key, 'subsample_new_batch')

            else:
                # Batch-level / constant.
                out[key] = val

        out.batch = seg
        out.ptr = new_ptr
        out._stamp_ptr_cpu(new_ptr, new_ptr_cpu)
        out.__dict__['_num_graphs'] = k

        out.rebuild_simple_slice_inc_(ptr_cpu=new_ptr_cpu, num_nodes=total_nodes,
                                      levels={key: lv for key, (lv, _) in plan.items() if lv in (ATOM, ROW)},
                                      entries=entries)

        return out

    def append_batch(self, other, *, validate=True):
        """
        Append another compatible batch without using PyG slice/inc metadata.

        Returns a new object of ``self``'s class holding ``self``'s graphs followed by
        ``other``'s; ``self`` is not modified. ``other`` is first passed through
        ``other.to(self.device)``. Keys are handled, in this order:

          - ``ptr`` and ``batch``: rebuilt, ``other``'s offset by ``self``'s node and
            graph counts
          - 0-dim tensor: shared metadata, keep self's value, optionally validate shape,
            dtype and equality
          - node-level tensors (``size(0)`` equal to each batch's node count): cat along
            dim 0
          - graph-level tensors (``size(0)`` equal to each batch's graph count): cat
            along dim 0
          - any other tensor: shared metadata, keep self's value, optionally validate
            shape, dtype and equality
          - graph-level lists/tuples: concatenate; other lists/tuples are shared
            metadata, keep self's value, optionally validate equality
          - dicts: keep self's value, not compared
          - anything else: keep self's value, optionally validate ``==``
          - the crystal unit map ``unit_ind`` carried by one side only: the other side
            gets ``crystal_building.utils.derived_unit_ind``, the equal-block layout every
            reader gives a row without a map (-1 on rows whose units cannot be read), and
            the two are concatenated, with or without ``validate`` (since the stage-2
            review, 2026-10-23; before, KeyError or dropped)
          - a ragged value (``UNIT`` or ``EDGE`` in either batch, by the level contract of
            the class Notes; checked before the length rules above): concatenated along
            its collation axis, ``other``'s 'index' / 'face' values offset by ``self``'s
            atom count, with a slice entry joining the two batches' pointers (since stage
            3, 2026-10-23: an index tensor of a 2-graph batch was concatenated as rows,
            never offset, and a per-unit value was kept once as shared metadata). The
            other batch's level may be ``ATOM`` or ``ROW`` where its length coincides
            (its entry, where it has one, gives the pointer); a ``BATCH`` value there is
            refused, as is a ``UNIT_CELL`` value in either batch. Since the stage-3 review
            (2026-10-23) a value's slice entry decides its level before its length
            (``_level_plan``), so a ragged value whose total equals a batch's atom or row
            count is joined here too (it was concatenated by the node / graph length rules
            above and given the result's ``ptr`` / ``arange`` entry)
          - a key registered ``BATCH`` (``KEY_LEVELS``: ``max_z_prime``, the space-group
            lookup tables) holding a tensor, list or tuple: shared metadata whatever its
            length, checked before the node / graph length rules above; keep self's value,
            optionally validate type, shape, dtype and equality (since the stage-3 review,
            2026-10-23; a 231-row batch's ``sym_mult_lut`` was concatenated as rows)

        The result's bookkeeping gives a concatenated atom or row value the level both
        batches record for it (``_level_plan``), so a result with as many nodes as graphs
        keeps its per-graph entries. A batch with one node in every graph
        (``_levels_coincide``), or one that files the value as batch-level, records
        nothing for it, and the other batch's level is used alone. A value whose levels
        conflict, or that neither batch records, and every value not concatenated, get
        the length rule on the result.

        Parameters
        ----------
        other : MXtalBase
            Batch to append.
        validate : bool
            If True (default), raise on keys missing from either batch and on shared
            values that differ. If False, keys missing from either batch are dropped
            from the result and shared values are taken from ``self`` unchecked.

        Returns
        -------
        MXtalBase
            New batch with ``num_graphs = self.num_graphs + other.num_graphs`` and
            rebuilt slice/increment bookkeeping.

        Raises
        ------
        KeyError
            With ``validate``, if a key is present in only one batch (``unit_ind`` excepted).
        TypeError
            If a key holds a tensor, list, tuple or dict in ``self`` and a different
            type in ``other``.
        ValueError
            With ``validate``, if a shared field differs between the batches. Whether or
            not ``validate``, if one side carries ``unit_ind`` and the other's row atom
            counts (``num_atoms``) do not sum to its atoms (no map can be derived for it).
        BatchLevelError (a ValueError)
            A value ``_level_plan`` refuses in either batch; a ragged value that the
            other batch holds batch-level, as a non-tensor, or with other trailing
            dimensions; a ``UNIT_CELL`` value.
        """
        device = self.device
        other = other.to(device)

        # host ptr mirrors: no device sync for the size math (see _ptr_cpu)
        self_ptr_cpu = self._ptr_cpu()
        other_ptr_cpu = other._ptr_cpu()
        n0 = int(self_ptr_cpu[-1])
        n1 = int(other_ptr_cpu[-1])
        g0 = self.num_graphs
        g1 = other.num_graphs

        out = self.__class__.__new__(self.__class__)
        out.__dict__['_store'] = GlobalStorage(_parent=out)

        # each batch's levels, for the result's bookkeeping (concatenation of an atom or row value is decided by
        # length below and is the same for either level) and for the ragged values (UNIT / EDGE)
        plan0 = self._level_plan(self_ptr_cpu, n0, g0)
        plan1 = other._level_plan(other_ptr_cpu, n1, g1)
        levels0 = {k: lv for k, (lv, _) in plan0.items()}
        levels1 = {k: lv for k, (lv, _) in plan1.items()}
        new_ptr_cpu = torch.cat([self_ptr_cpu, other_ptr_cpu[1:] + n0])
        joined = set()
        joined_entries = {}

        keys = set(self._store.keys()) | set(other._store.keys())

        # The crystal unit map: a side without one (older rows, GFN's blank batch, a bare from_arrays batch) is read by
        # every reader as z_prime equal contiguous blocks, so it gets exactly that map written out (-1 on rows whose
        # units cannot be read, which every reader refuses either way). Before the stage-2 review (2026-10-23) the
        # key was refused (KeyError) or dropped for the whole result.
        derived_maps = {}
        if (UNIT_MAP_KEY in self._store) != (UNIT_MAP_KEY in other._store):
            lacking = 'self' if UNIT_MAP_KEY not in self._store else 'other'
            derived_maps[lacking] = derived_unit_ind(self if lacking == 'self' else other,
                                                     f'append_batch ({lacking}, without {UNIT_MAP_KEY})')

        for key in sorted(keys):
            if key in ('ptr', 'batch'):
                continue

            if key == UNIT_MAP_KEY and derived_maps:
                v0 = derived_maps['self'].to(device) if 'self' in derived_maps else self[key]
                v1 = derived_maps['other'].to(device) if 'other' in derived_maps else other[key]
                out[key] = torch.cat([v0, v1.to(v0.device)], dim=0)
                joined.add(key)
                continue

            if key not in self._store or key not in other._store:
                if validate:
                    raise KeyError(f"Key {key!r} not present in both batches.")
                continue

            v0 = self[key]
            v1 = other[key]

            l0, l1 = levels0[key], levels1[key]
            if UNIT_CELL in (l0, l1):
                raise _unit_cell_refusal(key, 'append_batch')
            if l0 in (UNIT, EDGE) or l1 in (UNIT, EDGE):
                out[key], joined_entries[key] = self._join_segments(key, v0, v1, plan0[key], plan1[key], other,
                                                                    n0, g0, g1, new_ptr_cpu)
                continue

            if KEY_LEVELS.get(key) == BATCH and isinstance(v0, (torch.Tensor, list, tuple)):
                # a registered constant (KEY_LEVELS) is shared metadata whatever its length: not concatenated where
                # its length is both batches' atom or row count (since the stage-3 review, 2026-10-23: a 231-row
                # batch's sym_mult_lut was joined to 462 entries, where subsample_new_batch kept it whole)
                if type(v1) is not type(v0) and not (torch.is_tensor(v0) and torch.is_tensor(v1)):
                    raise TypeError(f"Field {key!r} {type(v0).__name__}/{type(v1).__name__} mismatch.")
                if validate:
                    if torch.is_tensor(v0):
                        v1 = v1.to(v0.device)
                        if v0.shape != v1.shape or v0.dtype != v1.dtype:
                            raise ValueError(f"Shared tensor field {key!r} differs in shape/dtype: "
                                             f"{tuple(v0.shape)}, {v0.dtype} vs {tuple(v1.shape)}, {v1.dtype}")
                        if v0.numel() > 0 and not torch.equal(v0, v1):
                            raise ValueError(f"Shared tensor field {key!r} differs.")
                    elif v0 != v1:
                        raise ValueError(f"Shared {type(v0).__name__} field {key!r} differs.")
                out[key] = v0
                continue

            if torch.is_tensor(v0):
                if not torch.is_tensor(v1):
                    raise TypeError(f"Field {key!r} tensor/non-tensor mismatch.")

                v1 = v1.to(v0.device)

                # Scalar metadata.
                if v0.ndim == 0:
                    if validate:
                        if v0.shape != v1.shape or v0.dtype != v1.dtype:
                            raise ValueError(
                                f"Shared tensor field {key!r} differs in shape/dtype."
                            )
                        if v0.numel() > 0 and not torch.equal(v0, v1):
                            raise ValueError(f"Shared tensor field {key!r} differs.")
                    out[key] = v0

                # Node-level tensor. Checked before the length-1 metadata case so
                # that a single-node batch (n0 == 1) is still concatenated.
                elif v0.size(0) == n0 and v1.size(0) == n1:
                    out[key] = torch.cat([v0, v1], dim=0)
                    joined.add(key)

                # Graph-level tensor. Checked before the length-1 metadata case so
                # that a single-graph batch (g0 == 1) is still concatenated; genuine
                # shared metadata stays size-1 in both batches and so cannot match
                # here (it requires v1.size(0) == g1).
                elif v0.size(0) == g0 and v1.size(0) == g1:
                    out[key] = torch.cat([v0, v1], dim=0)
                    joined.add(key)

                # Shared / length-1 tensor metadata.
                else:
                    if validate:
                        if v0.shape != v1.shape or v0.dtype != v1.dtype:
                            raise ValueError(
                                f"Shared tensor field {key!r} differs in shape/dtype: "
                                f"{tuple(v0.shape)}, {v0.dtype} vs "
                                f"{tuple(v1.shape)}, {v1.dtype}"
                            )
                        if v0.numel() > 0 and not torch.equal(v0, v1):
                            raise ValueError(f"Shared tensor field {key!r} differs.")
                    out[key] = v0

            elif isinstance(v0, list):
                if not isinstance(v1, list):
                    raise TypeError(f"Field {key!r} list/non-list mismatch.")

                if len(v0) == g0 and len(v1) == g1:
                    out[key] = v0 + v1
                    joined.add(key)
                else:
                    if validate and v0 != v1:
                        raise ValueError(f"Shared list field {key!r} differs.")
                    out[key] = v0

            elif isinstance(v0, tuple):
                if not isinstance(v1, tuple):
                    raise TypeError(f"Field {key!r} tuple/non-tuple mismatch.")

                if len(v0) == g0 and len(v1) == g1:
                    out[key] = v0 + v1
                    joined.add(key)
                else:
                    if validate and v0 != v1:
                        raise ValueError(f"Shared tuple field {key!r} differs.")
                    out[key] = v0

            elif isinstance(v0, dict):
                if not isinstance(v1, dict):
                    raise TypeError(f"Field {key!r} dict/non-dict mismatch.")
                out[key] = v0

            else:
                if validate and v0 != v1:
                    raise ValueError(f"Shared field {key!r} differs: {v0!r} vs {v1!r}")
                out[key] = v0

        new_ptr = torch.cat([
            self.ptr,
            other.ptr[1:].to(device) + n0,
        ])
        out.ptr = new_ptr
        out._stamp_ptr_cpu(new_ptr, new_ptr_cpu)

        out.batch = torch.cat([
            self.batch,
            other.batch.to(device) + g0,
        ])

        out.__dict__['_num_graphs'] = g0 + g1
        # A concatenated value keeps the level the two batches record for it. A side
        # says nothing about a value when it has one atom in every row (per-atom and
        # per-row coincide there, _levels_coincide) or files the value as shared (a
        # length-1 value without an entry, concatenated all the same); the other side's
        # level then stands alone. Levels that conflict, a value neither side knows, and
        # every value not concatenated get the result's own levels (the length rule, as
        # it has no bookkeeping yet).
        known0 = {} if _levels_coincide(self_ptr_cpu, n0, g0) else levels0
        known1 = {} if _levels_coincide(other_ptr_cpu, n1, g1) else levels1
        joined_levels = {}
        for k in joined:
            l0 = known0.get(k) if known0.get(k) in (ATOM, ROW) else None
            l1 = known1.get(k) if known1.get(k) in (ATOM, ROW) else None
            if l0 is None or l1 is None or l0 == l1:
                level = l1 if l0 is None else l0
                if level is not None:
                    joined_levels[k] = level
        out.rebuild_simple_slice_inc_(num_nodes=n0 + n1, levels=joined_levels, entries=joined_entries)

        return out

    def _join_segments(self, key, v0, v1, lp0, lp1, other, n0, g0, g1, new_ptr_cpu):
        """``append_batch`` of a ragged (``UNIT`` / ``EDGE``) value: ``(joined value, (slice entry, increments))``.

        Each side's pointer is its ``UNIT`` / ``EDGE`` pointer, or, where that side files the value ``ATOM`` /
        ``ROW`` (its entry is ``ptr`` / ``arange``, or it has no usable host entry and its length decides), the
        counts its own slice entry records, else ``ptr`` / ``arange``. The values
        are concatenated along the collation axis, ``other``'s 'index' / 'face' values offset by ``self``'s atom
        count ``n0``; the entry joins the two pointers; the increments are the joined rows' atom offsets for an
        offset key (as collation writes them), zeros otherwise. Refuses a side that holds the value batch-level, a
        non-tensor, or other trailing dimensions.
        """
        sides = []
        for batch, v, (level, pointer), g in ((self, v0, lp0, g0), (other, v1, lp1, g1)):
            if not torch.is_tensor(v) or v.ndim == 0 or level not in (ATOM, ROW, UNIT, EDGE):
                what = (f'{level} {tuple(v.shape)}' if torch.is_tensor(v) else type(v).__name__)
                raise BatchLevelError(f'append_batch: {key!r} is ragged (per unit or per edge) in one batch and '
                                      f'{what} in the other; the two cannot be joined')
            if level in (ATOM, ROW):
                entry = _entry_pointer(batch, key, v, (batch.__dict__.get('_slice_dict') or {}).get(key), g)
                pointer = entry if entry is not None else pointer
            sides.append(pointer)
        p0, p1 = sides
        axis = _cat_axis(self, key, v0)
        if v0.ndim != v1.ndim or _cat_axis(other, key, v1) != axis or \
                [s for d, s in enumerate(v0.shape) if d != axis] != [s for d, s in enumerate(v1.shape) if d != axis]:
            raise BatchLevelError(f'append_batch: {key!r} is {tuple(v0.shape)} in self and {tuple(v1.shape)} in '
                                  f'other; they join only along their collation axis')
        v1 = v1.to(v0.device)
        if _offset_by_atoms(key):
            v1 = v1 + torch.as_tensor(n0, dtype=v1.dtype, device=v1.device)
        joined = torch.cat([v0, v1], dim=axis)
        pointer = torch.cat([p0, p1[1:] + p0[-1]])
        inc = new_ptr_cpu[:-1].clone() if _offset_by_atoms(key) else torch.zeros(g0 + g1, dtype=torch.long)
        return joined, (pointer, inc)


# noinspection PyPropertyAccess


class MolData(MXtalBase, MolConformerMethods, MolDataMethods):
    r"""
    A graph representing a single molecule; after collation, a batch of molecules.

    Methods come from ``MolConformerMethods`` and ``MolDataMethods``. The properties
    ``x``, ``z``, ``graph_x``, ``edge_index``, ``y``, ``pos``, ``radius``, ``mass``,
    ``mol_volume``, ``batch``, ``ptr`` and ``smiles`` return None when the key is not
    stored; any other absent attribute raises ``AttributeError``.

    Notes
    -----
    Stored attributes of one molecule with ``num_atoms`` atoms. Coordinates are
    Cartesian, in Angstrom. Arguments passed as None are not stored at all.

    Per-node; collated by concatenation to [num_nodes, ...]:

    ``z`` : torch.Tensor [num_atoms], long
        Atomic numbers.
    ``pos`` : torch.Tensor [num_atoms, 3]
        Atom positions.
    ``x`` : torch.Tensor [num_atoms, ...], optional
        Node features, stored as given (``from_smiles`` stores Gasteiger partial
        charges, or zeros when charges are not computed, [num_atoms] float32).

    Per-graph; 0-dim tensors and python numbers collate to [num_graphs], strings to
    lists of length num_graphs:

    ``num_atoms`` : torch.Tensor [], long
        Atom count, set to ``len(z)`` when ``z`` is given and ``num_atoms`` is not.
    ``smiles``, ``identifier`` : str, optional
        SMILES string and identifier, stored as given.
    ``radius`` : torch.Tensor [] or float, optional
        Largest distance of a heavy atom (``z > 1``) from the heavy-atom centroid, as
        computed by ``mol_analysis``.
    ``mass`` : torch.Tensor [] or float, optional
        Sum of ``ATOM_WEIGHTS`` over the atoms, as computed by ``mol_analysis``.
    ``mol_volume`` : torch.Tensor [] or float, optional
        Overlap-corrected van der Waals volume (``VDW_RADII``), cubic Angstrom, as
        computed by ``mol_analysis``.
    ``graph_x``, ``y`` : optional
        Graph-level features and targets, stored as given.

    Edges; index keys are offset by the node count on collation:

    ``edge_index`` : torch.Tensor [2, num_edges], optional
        Source and target node indices, stored as given.

    Batch only:

    ``batch`` : torch.Tensor [num_nodes], long
        Graph index of each node.
    ``ptr`` : torch.Tensor [num_graphs + 1], long
        Offset of each graph's first node, followed by ``num_nodes``.

    ``build_conformer_tree`` adds internal-coordinate tree keys (``tree_*``,
    ``mol_bond_index``); see ``MolConformerMethods``.
    """

    def __init__(self,
                 z: Optional[torch.Tensor] = None,
                 pos: Optional[torch.Tensor] = None,
                 x: OptTensor = None,
                 graph_x: OptTensor = None,
                 y: OptTensor = None,
                 edge_index: OptTensor = None,
                 smiles: str = None,
                 identifier: str = None,
                 mol_volume: Optional[float] = None,
                 mass: Optional[float] = None,
                 radius: Optional[float] = None,
                 do_mol_analysis: Optional[bool] = False,
                 build_tree: Optional[bool] = False,
                 **kwargs):
        """
        Store the given molecular attributes; optionally analyse the molecule.

        Parameters
        ----------
        z : torch.Tensor, optional
            Atomic numbers [num_atoms], long.
        pos : torch.Tensor, optional
            Cartesian positions [num_atoms, 3], Angstrom.
        x : torch.Tensor, optional
            Node features [num_atoms, ...].
        graph_x : torch.Tensor, optional
            Graph-level features, stored as given.
        y : torch.Tensor, optional
            Targets, stored as given.
        edge_index : torch.Tensor, optional
            Edges [2, num_edges].
        smiles : str, optional
        identifier : str, optional
        mol_volume : float or torch.Tensor, optional
            Stored as given.
        mass : float or torch.Tensor, optional
            Stored as given.
        radius : float or torch.Tensor, optional
            Stored as given.
        do_mol_analysis : bool
            If True, call ``mol_analysis()``, which computes ``radius``, ``mass`` and
            ``mol_volume`` for whichever of them is None. Needs ``z`` and ``pos``.
        build_tree : bool
            If True, call ``build_conformer_tree()``. Needs ``pos``.
        **kwargs
            Every entry that is not None is stored as an attribute.

        Notes
        -----
        ``num_atoms`` is set to a 0-dim long tensor ``len(z)`` when ``z`` is given and
        ``num_atoms`` was not passed in ``kwargs``.
        """
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        # set attributes
        kwargs.update(
            z=z,
            pos=pos,
            x=x,
            y=y,
            graph_x=graph_x,
            edge_index=edge_index,
            smiles=smiles,
            identifier=identifier,
            mol_volume=mol_volume,
            mass=mass,
            radius=radius
        )
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        # only piece of analysis for init
        if self.z is not None and not hasattr(self, 'num_atoms'):
            self.num_atoms = torch.tensor(len(z), dtype=torch.long)

        if do_mol_analysis:
            self.mol_analysis()

        # opt-in: needs `pos` (there is no stored bond topology to build from) and can
        # legitimately fail on a disconnected or fully-linear molecule, so it is not
        # forced on every construction. Accessors build it lazily on first use anyway.
        if build_tree:
            self.build_conformer_tree()

    @property
    def x(self) -> Any:
        return self['x'] if 'x' in self._store else None

    @property
    def z(self) -> Any:
        return self['z'] if 'z' in self._store else None

    @property
    def graph_x(self) -> Any:
        return self['graph_x'] if 'graph_x' in self._store else None

    @property
    def edge_index(self) -> Any:
        return self['edge_index'] if 'edge_index' in self._store else None

    @property
    def y(self) -> Any:
        return self['y'] if 'y' in self._store else None

    @property
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def radius(self) -> Any:
        return self['radius'] if 'radius' in self._store else None

    @property
    def mass(self) -> Any:
        return self['mass'] if 'mass' in self._store else None

    @property
    def mol_volume(self) -> Any:
        return self['mol_volume'] if 'mol_volume' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    @property
    def ptr(self) -> Any:
        return self['ptr'] if 'ptr' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None


# noinspection PyPropertyAccess
class MolCrystalData(  # order crystal ops first to overwrite any mol ops
    MolCrystalOps,
    MolCrystalBuilding,
    MolCrystalAnalysis,
    MolCrystalEllipsoidOps,
    MolCrystalIO,
    MolCrystalStore,
    MolCrystalSteps,
    MolCrystalChem,
    MolCrystalRecipes,
    MolData,
):
    r"""
    A data object representing a molecular crystal; after collation, a batch of crystals.

    A crystal is a molecule (for Z' > 1, the Z' asymmetric-unit molecules merged into
    one graph) plus a space group, cell parameters and the pose of each
    asymmetric-unit molecule. The pose parameters hold ``max_z_prime`` molecule slots;
    ``z_prime`` says how many are used. Methods come from ``MolCrystalOps``,
    ``MolCrystalBuilding``, ``MolCrystalAnalysis``, ``MolCrystalEllipsoidOps``,
    ``MolCrystalIO`` (the batch constructors ``read`` and ``from_arrays``) and
    ``MolCrystalStore`` (``save`` and ``load`` of the ingest store) and ``MolCrystalSteps``
    (the structural ingestion steps ``find_molecules``, ``parameterize_pose``,
    ``check_rebuild``, ``check_validity``) and ``MolCrystalChem`` (the chemistry steps
    ``sanitize`` and ``features``) and ``MolCrystalRecipes`` (``run_chain`` of a named recipe,
    and ``canonicalize`` to the fundamental-domain product), which precede
    ``MolData`` in the MRO and so take precedence over same-named molecule methods.

    The properties defined here and on ``MolData`` return None when their key is not
    stored; any other absent attribute raises ``AttributeError``. ``unit_cell_batch``
    and ``unit_cell_mol_ind`` test for ``unit_cell_pos`` instead of their own key.

    Notes
    -----
    Stored attributes of one crystal, shapes before collation. Leading dimensions of 1
    and 0-dim values become [num_graphs] rows in a batch. Lengths are in Angstrom and
    cell angles in radians.

    Molecule: ``z``, ``pos``, ``x``, ``num_atoms``, ``smiles``, ``identifier``,
    ``radius``, ``mass``, ``mol_volume``, as on :class:`MolData`. When ``molecule`` is a
    list (Z' > 1) they cover all listed molecules: tensors with a leading dimension are
    concatenated along dim 0, and 0-dim tensors and python numbers are summed. The four
    row scalars ``num_atoms``, ``mass``, ``mol_volume`` and ``radius`` are sums over the
    molecules whether each molecule holds them 0-dim (``MolData.__init__``,
    ``mol_analysis``) or [1] (a row taken out of a collated batch; since 2026-10-22), and
    are stored 0-dim either way (since the stage-2 review, 2026-10-23).

    Units (stage 2 of the units plan, 2026-10-22):

    ``unit_ind`` : torch.Tensor [num_atoms], long
        The atom -> unit map, the single authority for which asymmetric-unit molecule
        (unit) an atom belongs to: the row-local unit index 0 .. z_prime - 1. Written by
        the constructor (molecule ``k`` of a list is unit ``k``; a single molecule is unit
        0; a map-less Z'>1 crystal row passed as the molecule keeps its equal blocks; an
        explicit ``unit_ind`` keyword overrides), by ``from_arrays`` with a pose and by
        ``find_molecules`` (since stage 3, 2026-10-23; ``sanitize`` fills it for the
        hydrogens it places), and checked against ``z_prime`` at construction. A row
        without it (an older file, GFN's blank batch) has ``z_prime`` equal contiguous
        blocks (``crystal_building.utils.derived_unit_ind`` writes them out); every reader
        goes through ``crystal_building.utils.unit_layout``, which also requires, at this
        stage, equal contiguous units in unit order. Per atom (registered ``ATOM`` in
        ``KEY_LEVELS``), so it is sliced, subsampled, appended and collated with the atoms
        (``append_batch`` and ``collate_data_list`` give a map-less row its derived map when
        another row carries one); clusters do not carry it. ``num_units``, ``unit_ptr``,
        ``unit_batch`` and ``unit_atom_counts`` are derived from it (not stored). A ragged
        per-unit value, one entry per unit, is attached with ``add_unit_attr`` (or collated
        from rows that hold ``[z_prime, ...]`` each).

    Symmetry; set when ``sg_ind`` or ``sg_name`` is given:

    ``sg_ind`` : torch.Tensor, long
        Space-group number; [] when given as an int, otherwise the given shape.
    ``symmetry_operators`` : np.ndarray [sym_mult, 4, 4]
        Affine symmetry operators acting on fractional coordinates. A batch holds a
        list with one array per graph.
    ``sym_mult`` : torch.Tensor [1], long
        Number of symmetry operators (molecules per unit cell for Z' = 1).
    ``z_prime`` : torch.Tensor [1] (or the shape of a tensor given), long
        Molecules in the asymmetric unit; ``ones(1)`` when not given; a python int is
        stored [1] (0-dim before 2026-10-22, which did not collate beside the default).
    ``nonstandard_symmetry`` : bool
        True when the operators were supplied with ``nonstandard_symmetry=True``.

    Cell and asymmetric-unit pose:

    ``cell_lengths`` : torch.Tensor [1, 3]
        a, b, c in Angstrom.
    ``cell_angles`` : torch.Tensor [1, 3]
        alpha, beta, gamma in radians.
    ``aunit_centroid`` : torch.Tensor [1, 3 * max_z_prime]
        Fractional coordinates of each molecule's centroid, 3 columns per slot; unused
        slots padded with 0.
    ``aunit_orientation`` : torch.Tensor [1, 3 * max_z_prime]
        Rotation vector (axis times angle in radians) of each molecule, 3 columns per
        slot; unused slots padded with 1 by the constructor and with 0 by
        ``collate_data_list``.
    ``aunit_handedness`` : torch.Tensor [1, max_z_prime]
        +1 or -1 per molecule, the sign given to the x axis of the standard molecular
        frame; unused slots padded with 1 by the constructor and with 0 by
        ``collate_data_list``.
    ``max_z_prime`` : int
        Number of pose slots. A python int on batches as well.
    ``is_well_defined`` : bool
        Stored as given (default True); a tensor [num_graphs] in a batch.

    Derived by ``box_analysis``:

    ``T_fc``, ``T_cf`` : torch.Tensor [1, 3, 3]
        Fractional-to-Cartesian and Cartesian-to-fractional transforms.
    ``cell_volume`` : torch.Tensor [1]
        Cubic Angstrom.
    ``packing_coeff`` : torch.Tensor
        ``mol_volume * sym_mult / cell_volume``.
    ``density`` : torch.Tensor
        ``mass * sym_mult / cell_volume * 1.66054``, g/cm^3.

    Unit cell; set by ``build_unit_cell``, ``num_atoms * sym_mult`` rows per crystal:

    ``unit_cell_pos`` : torch.Tensor [num_unit_cell_atoms, 3]
        Cartesian positions of every atom in the unit cell.
    ``unit_cell_batch`` : torch.Tensor [num_unit_cell_atoms], long
        Crystal index of each unit-cell atom.
    ``unit_cell_mol_ind`` : torch.Tensor [num_unit_cell_atoms], long
        Molecule index of each unit-cell atom, counted over the whole batch.

    Cluster; per node of the cluster batch returned by ``mol2cluster`` or
    ``build_cluster``, whose ``pos``/``z``/``batch`` cover all cluster atoms:

    ``aux_ind`` : torch.Tensor [num_nodes], long
        0 for atoms of molecule 0 (``mol_ind == 0``), 1 for other molecules, 2 for
        molecules the cluster builder flags as beyond its distance cutoff.
    ``mol_ind`` : torch.Tensor [num_nodes], long
        Molecule index within each cluster (for Z' = 1, molecule 0 is the
        asymmetric-unit molecule).
    """

    def __init__(self,
                 molecule: Optional[Union[list, MolData]] = None,
                 sg_ind: Optional[Union[int, torch.Tensor]] = None,
                 sg_name: Optional[str] = None,
                 z_prime: Optional[torch.Tensor] = None,
                 cell_lengths: Optional[torch.Tensor] = None,
                 cell_angles: Optional[torch.Tensor] = None,
                 aunit_centroid: Optional[torch.Tensor] = None,
                 aunit_orientation: Optional[torch.Tensor] = None,
                 aunit_handedness: Optional[Union[torch.Tensor, int]] = None,
                 identifier: Optional[str] = None,
                 nonstandard_symmetry: Optional[bool] = False,
                 symmetry_operators: Optional[list] = None,
                 is_well_defined: Optional[bool] = True,
                 aux_ind: Optional[torch.Tensor] = None,
                 mol_ind: Optional[torch.Tensor] = None,
                 do_box_analysis: Optional[bool] = False,
                 max_z_prime: Optional[int] = 1,
                 **kwargs):
        """
        Initialize a molecular crystal from an existing molecule.

        Does not natively enforce physicality or symmetry correctness: the given values
        are stored as they are, apart from the reshaping and padding described below.
        Do clean_cell_parameters and/or box_analysis for further processing.

        Parameters
        ----------
        molecule : MolData or list of MolData, optional
            Source of the molecular attributes, copied by ``set_mol_attrs``. A list
            (Z' > 1) is merged into one graph: node tensors concatenated, 0-dim tensors
            and numbers summed (``num_atoms``, ``radius``, ``mass``, ``mol_volume`` also
            when [1]-shaped), strings joined with ``'|'``, and ``identifier`` taken
            from the first molecule; molecule ``k`` becomes unit ``k`` of the unit map
            ``unit_ind``. A single molecule is one unit (``unit_ind`` zeros) unless it
            carries a map; a merged multi-unit molecule needs ``unit_ind`` passed as a
            keyword. A collated molecule batch is not supported here:
            ``set_mol_attrs`` then writes per-graph and per-node values through
            ``add_graph_attr`` / ``add_node_attr`` and ``self._slice_dict``, which
            fail (``TypeError`` or ``AttributeError``) on the non-batch object being
            constructed.
        sg_ind : int or torch.Tensor, optional
            Space-group number. Setting a space group reads the device from ``z``, so
            ``z`` must be set, from ``molecule`` or ``kwargs``.
        sg_name : str, optional
            Space-group symbol, looked up in ``SPACE_GROUP_NAMES``; replaces ``sg_ind``.
        z_prime : int or torch.Tensor, optional
            Molecules in the asymmetric unit, stored as a long tensor; ``ones(1)`` when
            omitted. Only used when a space group is given.
        cell_lengths : torch.Tensor, optional
            a, b, c in Angstrom, [3] or [1, 3]; stored as [1, 3].
        cell_angles : torch.Tensor, optional
            alpha, beta, gamma in radians, [3] or [1, 3]; stored as [1, 3].
        aunit_centroid : torch.Tensor, optional
            Fractional centroid coordinates, [3 * k] or [1, 3 * k] with
            k <= max_z_prime; padded with 0 to [1, 3 * max_z_prime].
        aunit_orientation : torch.Tensor, optional
            Rotation vectors, same shapes as ``aunit_centroid``; padded with 1 to
            [1, 3 * max_z_prime].
        aunit_handedness : int or torch.Tensor, optional
            +1 or -1 per molecule, scalar, [k] or [1, k]; padded with 1 to
            [1, max_z_prime]. A python number is converted on ``self.device``, which
            needs ``z``.
        identifier : str, optional
            Replaces any identifier copied from ``molecule``.
        nonstandard_symmetry : bool
            Stored as the ``nonstandard_symmetry`` flag when a space group is given. If
            True, ``symmetry_operators`` must be given.
        symmetry_operators : np.ndarray, optional
            Operators [n_ops, 4, 4]; defaults to ``np.stack(SYM_OPS[sg_ind])``.
        is_well_defined : bool
            Stored as given. Default True.
        aux_ind : torch.Tensor, optional
            Per-node cluster labels, stored as given.
        mol_ind : torch.Tensor, optional
            Per-node molecule indices, stored as given.
        do_box_analysis : bool
            If True, call ``box_analysis()`` at the end, which sets ``T_fc``, ``T_cf``,
            ``cell_volume``, ``packing_coeff`` and ``density``; it needs the cell
            parameters, a space group, ``mol_volume`` and ``mass``.
        max_z_prime : int
            Number of asymmetric-unit slots in the pose parameters. Default 1.
        **kwargs
            Every entry that is not None is stored as an attribute.

        Raises
        ------
        ValueError
            If ``sg_name`` is not in ``SPACE_GROUP_NAMES``, or if ``do_box_analysis``
            is set and ``mol_volume`` or ``mass`` is missing; if ``z_prime`` disagrees
            with the unit map (it must hold exactly the units 0 .. z_prime - 1: a
            one-molecule crystal given ``z_prime=2`` is refused, audit #2); if the map is
            not a one-dimensional integer (non-bool) tensor with one entry per atom, or
            ``z_prime`` holds more than one value (since the stage-2 review); if a listed
            molecule's ``num_atoms`` / ``radius`` / ``mass`` / ``mol_volume`` is not one
            value; or if a listed molecule carries a map of several units.
        RuntimeError
            If ``nonstandard_symmetry`` is True and ``symmetry_operators`` is None.
        AssertionError
            If ``symmetry_operators`` is given and is not an ``np.ndarray``.
        AttributeError
            If ``cell_lengths`` is given without all of ``cell_angles``,
            ``aunit_centroid``, ``aunit_orientation`` and ``aunit_handedness``, or if a
            space group is given while ``z`` is not set.
        """
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        if molecule is not None:
            self.set_mol_attrs(molecule)

        kwargs.update(
            identifier=identifier,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
            aux_ind=aux_ind,
            mol_ind=mol_ind,
            is_well_defined=is_well_defined,
            max_z_prime=max_z_prime,
        )

        if aunit_centroid is not None:
            self.aunit_centroid = self._pad_tensor(
                aunit_centroid, (1, 3 * max_z_prime), 0)
        if aunit_orientation is not None:
            self.aunit_orientation = self._pad_tensor(
                aunit_orientation, (1, 3 * max_z_prime), 1)
        if aunit_handedness is not None:
            self.aunit_handedness = self._pad_tensor(
                aunit_handedness, (1, max_z_prime), 1)

        # set attrs
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        # functions / analysis
        # will raise error if sg_ind is None, but then, it can't be a real crystal
        if sg_ind is not None or sg_name is not None:
            if sg_name is not None:
                valid_sgs = list(SPACE_GROUP_NAMES.keys())
                if sg_name in valid_sgs:
                    sg_ind = SPACE_GROUP_NAMES[sg_name]
                else:
                    raise ValueError(f"{sg_name} is not in the set of canonical space group names!"
                                     f"{valid_sgs}")
            self.set_symmetry_attrs(nonstandard_symmetry=nonstandard_symmetry,
                                    sg_ind=sg_ind,
                                    symmetry_operators=symmetry_operators,
                                    z_prime=z_prime)

        # cell parameters need a leading dimension [n, 3]
        # will raise error if self.cell_lengths is None, but then, it can't be a real crystal
        if self.cell_lengths is not None:
            if self.cell_lengths.ndim == 1:
                self.cell_lengths = self.cell_lengths[None, ...]
            if self.cell_angles.ndim == 1:
                self.cell_angles = self.cell_angles[None, ...]
            if self.aunit_centroid.ndim == 1:
                self.aunit_centroid = self.aunit_centroid[None, ...]
            if self.aunit_orientation.ndim == 1:
                self.aunit_orientation = self.aunit_orientation[None, ...]
            if self.aunit_handedness.ndim == 1:
                self.aunit_handedness = self.aunit_handedness[None, ...]

        # z_prime against the unit map, where cheap (one row): the map must hold exactly the units 0 .. z_prime - 1
        # (a one-molecule row given z_prime 2 is refused, audit #2). Unit sizes and contiguity are checked when a
        # reader reads the map (crystal_building.utils.unit_layout).
        if UNIT_MAP_KEY in self._store and self.z_prime is not None and not self.is_batch:
            atoms = self.z if self.z is not None else self.pos
            refuse_unit_map_disagreeing_with_z_prime(self[UNIT_MAP_KEY], self.z_prime,
                                                     where=f'MolCrystalData({self.identifier!r})'
                                                     if isinstance(self.identifier, str) else 'MolCrystalData',
                                                     n_atoms=None if atoms is None else len(atoms))

        if do_box_analysis:
            self.box_analysis()

    # ---- the unit level, derived from the unit map (crystal_building.utils.unit_layout); nothing here is stored

    @property
    def num_units(self) -> Tensor:
        """[num_graphs] long: the units of each row, ``z_prime`` (the unit map is checked: ``unit_layout``)."""
        return unit_layout(self, 'num_units').units_per_row

    @property
    def unit_ptr(self) -> Tensor:
        """[num_graphs + 1] long: cumulative unit offsets over the batch, ``[0, z'_0, z'_0 + z'_1, ...]``."""
        return unit_layout(self, 'unit_ptr').unit_ptr

    @property
    def unit_batch(self) -> Tensor:
        """[sum(num_atoms)] long: the batch-wide unit of every atom, ``unit_ptr[row] + unit_ind`` (from the stored map,
        or the equal-block layout of a row without one)."""
        return unit_layout(self, 'unit_batch').unit_batch

    @property
    def unit_atom_counts(self) -> Tensor:
        """[sum(z_prime)] long: the atom count of every unit, rows in order."""
        return unit_layout(self, 'unit_atom_counts').unit_atom_counts

    @property
    def x(self) -> Any:
        return self['x'] if 'x' in self._store else None

    @property
    def z(self) -> Any:
        return self['z'] if 'z' in self._store else None

    @property
    def graph_x(self) -> Any:
        return self['graph_x'] if 'graph_x' in self._store else None

    @property
    def edge_index(self) -> Any:
        return self['edge_index'] if 'edge_index' in self._store else None

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def edge_attr(self) -> Any:
        return self['edge_attr'] if 'edge_attr' in self._store else None

    @property
    def y(self) -> Any:
        return self['y'] if 'y' in self._store else None

    @property
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def radius(self) -> Any:
        return self['radius'] if 'radius' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    @property
    def sg_ind(self) -> Any:
        return self['sg_ind'] if 'sg_ind' in self._store else None

    @property
    def symmetry_operators(self) -> Any:
        return self['symmetry_operators'] if 'symmetry_operators' in self._store else None

    @property
    def cell_lengths(self) -> Any:
        return self['cell_lengths'] if 'cell_lengths' in self._store else None

    @property
    def z_prime(self) -> Any:
        return self['z_prime'] if 'z_prime' in self._store else None

    @property
    def cell_angles(self) -> Any:
        return self['cell_angles'] if 'cell_angles' in self._store else None

    @property
    def T_fc(self) -> Any:
        return self['T_fc'] if 'T_fc' in self._store else None

    @property
    def T_cf(self) -> Any:
        return self['T_cf'] if 'T_cf' in self._store else None

    @property
    def identifier(self) -> Any:
        return self['identifier'] if 'identifier' in self._store else None

    @property
    def mol_ind(self) -> Any:
        return self['mol_ind'] if 'mol_ind' in self._store else None

    @property
    def unit_ind(self) -> Any:
        """The stored unit map, or None (a row without one has the equal-block layout: ``unit_layout``)."""
        return self[UNIT_MAP_KEY] if UNIT_MAP_KEY in self._store else None

    @property
    def aux_ind(self) -> Any:
        return self['aux_ind'] if 'aux_ind' in self._store else None

    @property
    def cell_volume(self) -> Any:
        return self['cell_volume'] if 'cell_volume' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None

    @property
    def packing_coeff(self) -> Any:
        return self['packing_coeff'] if 'packing_coeff' in self._store else None

    @property
    def density(self) -> Any:
        return self['density'] if 'density' in self._store else None

    @property
    def unit_cell_pos(self) -> Any:
        return self['unit_cell_pos'] if 'unit_cell_pos' in self._store else None

    @property
    def unit_cell_batch(self) -> Any:
        """Stored ``unit_cell_batch``, or None when ``unit_cell_pos`` is not stored.

        The check is on ``unit_cell_pos``: a stored ``unit_cell_batch`` without
        ``unit_cell_pos`` reads as None, and ``unit_cell_pos`` without
        ``unit_cell_batch`` raises ``KeyError``.
        """
        return self['unit_cell_batch'] if 'unit_cell_pos' in self._store else None

    @property
    def unit_cell_mol_ind(self) -> Any:
        """Stored ``unit_cell_mol_ind``, or None when ``unit_cell_pos`` is not stored.

        The check is on ``unit_cell_pos``, as for ``unit_cell_batch``.
        """
        return self['unit_cell_mol_ind'] if 'unit_cell_pos' in self._store else None

    @property
    def aunit_handedness(self) -> Any:
        return self['aunit_handedness'] if 'aunit_handedness' in self._store else None

    @property
    def is_well_defined(self) -> Any:
        return self['is_well_defined'] if 'is_well_defined' in self._store else None

    @property
    def nonstandard_symmetry(self) -> Any:
        return self['nonstandard_symmetry'] if 'nonstandard_symmetry' in self._store else None

    @property
    def cocrystal(self) -> Any:
        return self['cocrystal'] if 'cocrystal' in self._store else None


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    """
    Format one ``key=value`` entry for ``MXtalBase.__repr__``.

    0-dim tensors show their value, tensors and arrays their shape, ``SparseTensor``
    its sizes and ``nnz``, strings are quoted, other sequences show their length and
    mappings are formatted recursively. A ``BaseStorage`` value's key is wrapped in
    ANSI bold escapes.
    """
    pad = ' ' * indent
    if isinstance(value, Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, Tensor):
        out = str(list(value.size()))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, SparseTensor):
        out = str(value.sizes())[:-1] + f', nnz={value.nnz()}]'
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = '{}'
    elif (isinstance(value, Mapping) and len(value) == 1
          and not isinstance(list(value.values())[0], Mapping)):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = '{ ' + ', '.join(lines) + ' }'
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    if isinstance(value, BaseStorage):
        return f'{pad}\033[1m{key}\033[0m={out}'
    else:
        return f'{pad}{key}={out}'

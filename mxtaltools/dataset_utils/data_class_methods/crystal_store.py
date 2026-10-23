"""The ingest store, mixed into MolCrystalData: save a batch to a folder and load it back exactly.

For batches made by ``MolCrystalData.read`` / ``from_arrays`` (``crystal_io.py``), bare or
posed, with whatever attributes later steps have added. ``batch.save(path)`` writes one
folder; ``MolCrystalData.load(path)`` returns a batch equal to the saved one in every
stored attribute (python type, dtype, shape, values bit for bit, including NaN), in the
same store order, with the same ``_num_graphs``, ``_slice_dict`` and ``_inc_dict``. It is
the ingested store of the ingestion contract (§5 of ``docs/design/crystal_ingestion.md``
in the GFN repository), schema ``mxtaltools.ingest/1``.

Layout
------
``data.pt``
    One dict, readable with ``torch.load(path, weights_only=True)``: tensors, strings,
    python numbers and bools, None, and lists / tuples / dicts of them. No pickled
    classes and no numpy arrays. Keys: ``schema``; ``num_graphs``; ``keys`` (the store's
    keys in order); ``attrs`` ({key: encoded value}); ``slice_dict`` and ``inc_dict`` (the
    batch's PyG bookkeeping, tensors and None, or None when the batch had none).
    An encoded value is a dict whose ``kind`` says how to rebuild it:

    - ``tensor``: ``value``, the tensor (on CPU; ``requires_grad`` is not kept);
    - ``ndarray``: ``value`` as a tensor and the numpy ``dtype`` name;
    - ``scalar``: a python bool, int, float or str (``max_z_prime``);
    - ``str_list``: a list / tuple of str or None (``identifier``, ``source_path``,
      ``message``): ``values`` with '' in place of None, and the bool mask ``none``;
    - ``str_lists``: a list / tuple whose entries are lists (or tuples) of str, or None
      (``atom_labels``): the strings flat in ``values``, entry ``b`` being
      ``values[ptr[b]:ptr[b + 1]]``, and the mask ``none``;
    - ``array_list``: a list / tuple of numpy arrays or tensors, or None, of one dtype
      and rank (``symmetry_operators``, ``file_bonds``): with ``layout='axis'`` they are
      concatenated along the one ``axis`` whose size varies (axis 0 when none does) and
      entry ``b`` is the slice ``ptr[b]:ptr[b + 1]`` along it, so the operators of a
      batch are one [M, 4, 4] tensor with their row pointer; with ``layout='flat'`` (more
      than one axis varies, or rank 0) they are flattened and ``shapes`` holds each
      entry's shape. ``none`` masks None entries, ``lib`` says numpy or torch;
    - ``plain``: any other nesting of lists, tuples and dicts of None, bool, int, float
      and str, stored as it is.

    Anything else (another class, an object array, a sparse tensor) is refused by
    ``save``, which names the attribute.
``manifest.csv``
    One line per row: ``row, identifier, source_path, block_name, status, message`` (the
    status by name; '' where the batch has no such attribute or holds None).
``meta.json``
    ``schema``; ``rows``, ``atoms``; ``batch_class``; ``device`` the batch was saved
    from; ``status_names`` (the status table the codes were written against,
    ``status_table()``: entry ``i`` names code ``i``) and ``status_counts`` (every name of
    that table, with its count); ``done_counts`` (rows per ``done`` bitmask); ``levels``
    (per attribute: its level from the bookkeeping, ``atom``, ``row``, ``atom_or_row``
    when every row holds one atom so the two coincide, ``unit`` for an entry that is the
    unit pointer of ``z_prime``, ``edge`` for any other entry that describes the tensor
    (``MXtalBase``'s level contract; since stage 3, 2026-10-23), ``sliced`` for an entry
    that describes no level of it, ``batch`` for a value without one, ``bookkeeping`` for
    ``ptr`` / ``batch``; its
    encoding; dtype and shape for a tensor); ``steps_run`` (the arguments of every step
    run, as the caller gives them); ``code`` (``code_digest()``: sha256 per ingestion
    module); ``versions`` (torch, rdkit, spglib, ccdc); ``files`` (sha256 and size of
    ``data.pt`` and ``manifest.csv``); ``extra`` (the caller's); ``written_utc``.

Rows that failed a step are stored like any other: ``save`` takes every code of the status
table, the read's (``READ_STATUS``) and those the steps append (``crystal_steps.STATUS``
and any later chain module's ``STATUS``), looked up in one place, ``status_table()``.

The folder is written under a temporary name and renamed into place; ``save`` refuses an
existing path (as ``crystal_fd.dataset.save_dataset`` does). ``load`` checks every file
against its sha256 unless ``verify=False``, and always reads with ``weights_only=True``.
Tensors are loaded on the CPU, whatever device the batch was saved from.
"""
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data.storage import GlobalStorage

from mxtaltools.dataset_utils.data_class_methods.crystal_io import _batch_class
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import STATUS as _STEPS_STATUS

__all__ = ['MolCrystalStore', 'SCHEMA', 'StoreError', 'code_digest', 'read_meta', 'status_table']

SCHEMA = 'mxtaltools.ingest/1'
_DATA, _META, _MANIFEST = 'data.pt', 'meta.json', 'manifest.csv'

#: The ingestion modules of ``data_class_methods``, in chain order: the required ones, then
#: the later steps' modules, used when their file exists. ``code_digest`` hashes every one
#: present. Each module from ``crystal_steps`` on that defines ``STATUS`` declares the whole
#: status table so far (the previous module's table followed by its own codes, as
#: ``crystal_steps.STATUS`` is ``READ_STATUS`` followed by ``STEP_STATUS``); the status
#: table is the last of these (``status_table``).
_REQUIRED_MODULES = ('crystal_io', 'crystal_store', 'crystal_steps')
_LATER_MODULES = ('crystal_chem', 'crystal_recipes')
_PACKAGE = 'mxtaltools.dataset_utils.data_class_methods'

#: Instance attributes a batch keeps outside its store: saved (the first three) or
#: dropped and recomputed on demand (``_ptr_cpu_pair``, a host copy of ``ptr``).
_SAVED_DICT_KEYS = ('_num_graphs', '_slice_dict', '_inc_dict')
_CACHE_DICT_KEYS = ('_ptr_cpu_pair',)

#: Attributes the store needs: the row bookkeeping and the ingestion status.
_REQUIRED = ('ptr', 'batch', 'status', 'message', 'done')

#: Columns of manifest.csv, after the row index.
_MANIFEST_COLUMNS = ('identifier', 'source_path', 'block_name', 'status', 'message')

#: Deepest nesting a ``plain`` value may have (a list that holds itself is refused).
_MAX_DEPTH = 8

#: numpy dtype kinds an array list may hold: bool, signed and unsigned integers, floats, complex.
_ARRAY_KINDS = 'biufc'

#: Package distributions whose versions meta.json records (read from the installed
#: metadata, so nothing is imported: importing ccdc checks its licence).
_VERSION_DISTRIBUTIONS = (('rdkit', ('rdkit', 'rdkit-pypi')), ('spglib', ('spglib',)),
                          ('ccdc', ('csd-python-api', 'ccdc')))


class StoreError(ValueError):
    """A store folder, or a batch given to ``save``, does not satisfy the schema (named file, key or row)."""


# noinspection PyAttributeOutsideInit
class MolCrystalStore:
    """Mixin providing ``save`` and the classmethod ``load`` (schema ``mxtaltools.ingest/1``)."""

    def save(self, path, *, steps_run=None, extra_meta: Optional[dict] = None) -> dict:
        """Write this batch to a new folder ``path``.

        Parameters
        ----------
        path : str or os.PathLike
            The folder to create; it must not exist. Written as ``path.partial-<pid>``
            first and renamed into place, so a reader never sees a partial store.
        steps_run : list of dict, optional
            The arguments of every step that made this batch, in order, each a dict with
            a ``'step'`` name, e.g. ``[{'step': 'read', 'args': {'format': 'cif', ...}}]``;
            JSON-serialisable (no NaN). Recorded in meta.json as given; default none.
        extra_meta : dict, optional
            JSON-serialisable, string keys; recorded in meta.json under ``extra``.

        Returns
        -------
        dict
            The meta.json written (module docstring).

        Raises
        ------
        FileExistsError
            If ``path`` exists.
        StoreError
            ``self`` is not a batch, lacks one of ``ptr``, ``batch``, ``status``,
            ``message``, ``done``, holds a status outside the status table
            (``status_table()``), or holds an attribute the store cannot encode exactly
            (the module docstring's kinds); nothing is written.
        TypeError, ValueError
            ``steps_run`` or ``extra_meta`` of the wrong type, or not JSON-serialisable.
        """
        path = _as_path(path)
        if os.path.exists(path):
            raise FileExistsError(f'{path} exists; a store is never overwritten')
        steps = _check_steps_run(steps_run)
        extra = _check_extra_meta(extra_meta)
        table = status_table()
        ptr_cpu, status_cpu = _check_batch(self, table)
        B, N = len(ptr_cpu) - 1, int(ptr_cpu[-1])

        # encode everything before touching the disk: a value that cannot be stored refuses the call
        clones = {}
        keys = list(self._store.keys())
        attrs = {key: _encode(key, self._store[key], clones) for key in keys}
        bookkeeping = {}
        for name in ('_slice_dict', '_inc_dict'):
            entries = self.__dict__.get(name)
            bookkeeping[name] = None if entries is None else {
                key: _encode_bookkeeping(name, key, value, clones) for key, value in entries.items()}
        data = dict(schema=SCHEMA, num_graphs=self.__dict__.get('_num_graphs'), keys=keys, attrs=attrs,
                    slice_dict=bookkeeping['_slice_dict'], inc_dict=bookkeeping['_inc_dict'])

        status_names = [table[s] for s in status_cpu.tolist()]
        status_counts = torch.bincount(status_cpu, minlength=len(table)).tolist()
        done_values, done_counts = torch.unique(self['done'].detach().to('cpu').long(), return_counts=True)
        meta = dict(
            schema=SCHEMA, rows=B, atoms=N, batch_class=type(self).__name__, device=str(self['ptr'].device),
            status_names=list(table),
            status_counts=dict(zip(table, status_counts)),
            done_counts={str(int(v)): int(c) for v, c in zip(done_values.tolist(), done_counts.tolist())},
            levels=_level_map(self, attrs, ptr_cpu, B),
            steps_run=steps, code=code_digest(), versions=_versions(), files={}, extra=extra,
            written_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))

        tmp = f'{path}.partial-{os.getpid()}'
        if os.path.exists(tmp):                     # left by a process that died with this pid
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        try:
            torch.save(data, os.path.join(tmp, _DATA))
            _write_manifest(os.path.join(tmp, _MANIFEST), self, status_names, B)
            for name in (_DATA, _MANIFEST):
                file = os.path.join(tmp, name)
                meta['files'][name] = dict(sha256=_sha256(file), bytes=os.path.getsize(file))
            with open(os.path.join(tmp, _META), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=1, allow_nan=False)
            if os.path.exists(path):                # created meanwhile by someone else
                raise FileExistsError(f'{path} exists; a store is never overwritten')
            os.rename(tmp, path)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return meta

    @classmethod
    def load(cls, path, *, verify: bool = True):
        """Read a store folder back into the batch that was saved.

        Parameters
        ----------
        path : str or os.PathLike
            A folder written by ``save``.
        verify : bool
            Check ``data.pt`` and ``manifest.csv`` against their sha256 in meta.json
            before reading, and the rebuilt batch's row and atom counts against it.

        Returns
        -------
        MolCrystalDataBatch
            Every stored attribute rebuilt exactly (module docstring), in the saved
            order, with ``_num_graphs``, ``_slice_dict`` and ``_inc_dict``; tensors on
            the CPU.

        Raises
        ------
        FileNotFoundError
            ``path`` is not a folder, or a file of the store is missing.
        StoreError
            Another schema, a digest that does not match (a corrupted or edited file),
            a store saved from another class, or contents that do not decode.
        """
        path = _as_path(path)
        meta = read_meta(path)
        if verify:
            unlisted = sorted({_DATA, _MANIFEST} - set(meta.get('files') or {}))
            if unlisted:
                raise StoreError(f'{path}/{_META} lists no sha256 for {unlisted}')
            for name, record in meta['files'].items():
                got = _sha256(os.path.join(path, name))
                if got != record['sha256']:
                    raise StoreError(f'{path}/{name}: sha256 {got} differs from meta.json ({record["sha256"]}); '
                                     'the file changed after it was written')
        # weights_only: the file can hold tensors and plain python values, never code
        data = torch.load(os.path.join(path, _DATA), weights_only=True, map_location='cpu')
        if not isinstance(data, dict) or data.get('schema') != SCHEMA:
            raise StoreError(f'{path}/{_DATA} is not a {SCHEMA!r} data file')
        batch_cls = _batch_class(cls)
        if meta['batch_class'] != batch_cls.__name__:
            raise StoreError(f'{path} holds a {meta["batch_class"]}; {cls.__name__}.load builds a {batch_cls.__name__}')

        out = batch_cls.__new__(batch_cls)
        out.__dict__['_store'] = GlobalStorage(_parent=out)
        missing = [key for key in data['keys'] if key not in data['attrs']]
        if missing:
            raise StoreError(f'{path}/{_DATA} lists {missing} without their values')
        for key in data['keys']:
            out[key] = _decode(key, data['attrs'][key])
        if data['num_graphs'] is not None:
            out.__dict__['_num_graphs'] = data['num_graphs']
        for name, stored in (('_slice_dict', data['slice_dict']), ('_inc_dict', data['inc_dict'])):
            if stored is not None:
                out.__dict__[name] = dict(stored)
        if verify:
            rows, atoms = len(out['ptr']) - 1, int(out['ptr'][-1])
            if (rows, atoms) != (meta['rows'], meta['atoms']):
                raise StoreError(f'{path}: {rows} rows and {atoms} atoms rebuilt, meta.json records '
                                 f'{meta["rows"]} and {meta["atoms"]}')
        return out


###############################################################################
# folder, meta and code identity
###############################################################################

def read_meta(path) -> dict:
    """meta.json of the store at ``path``; raises ``FileNotFoundError`` or ``StoreError`` (another schema)."""
    path = _as_path(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f'{path} is not a store folder')
    with open(os.path.join(path, _META), encoding='utf-8') as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError as e:
            raise StoreError(f'{path}/{_META} is not JSON: {e}') from e
    if not isinstance(meta, dict) or meta.get('schema') != SCHEMA:
        raise StoreError(f'{path}: schema {meta.get("schema") if isinstance(meta, dict) else None!r}, '
                         f'expected {SCHEMA!r}')
    return meta


def code_digest() -> dict:
    """The identity of the ingestion code: ``{'digest', 'files', 'scope'}``.

    ``files`` maps each module's path (relative to the ``mxtaltools`` package, '/'
    separated) to its sha256, for the ingestion modules of ``data_class_methods``
    (``crystal_io.py``, ``crystal_store.py``, ``crystal_steps.py``, and ``crystal_chem.py``
    and ``crystal_recipes.py`` where their file exists) and every ``*.py`` of the native CIF
    reader (``dataset_utils/construction/cif/``); ``scope`` lists them. ``digest`` is the
    sha256 over the sorted ``path:sha256`` lines. It names bytes, not behaviour: an edited
    comment changes it too.
    """
    package = Path(__file__).resolve().parents[2]                 # .../mxtaltools
    modules = _ingest_modules()
    here = Path(__file__).resolve().parent
    paths = [here / f'{name}.py' for name in modules]
    paths += sorted((package / 'dataset_utils' / 'construction' / 'cif').glob('*.py'))
    files = {p.relative_to(package).as_posix(): _sha256(str(p)) for p in paths}
    lines = '\n'.join(f'{k}:{v}' for k, v in sorted(files.items()))
    scope = 'mxtaltools: data_class_methods/' + ', '.join(f'{name}.py' for name in modules) + ', construction/cif/*.py'
    return dict(digest=hashlib.sha256(lines.encode()).hexdigest(), files=files, scope=scope)


def status_table() -> Tuple[str, ...]:
    """The row status table: ``status`` code ``i`` is named ``status_table()[i]``.

    The last ``STATUS`` declared along the chain of ingestion modules: ``crystal_steps.STATUS``
    (``READ_STATUS`` followed by the structural steps' codes), extended by the ``STATUS`` of
    ``crystal_chem`` and then ``crystal_recipes`` where their file exists and defines one. The
    one place the store looks a status code up.

    Returns
    -------
    tuple of str

    Raises
    ------
    RuntimeError
        A later module's ``STATUS`` does not extend the table before it (codes are only ever
        appended, so an index keeps its meaning in every stored batch).
    """
    tables = [('crystal_steps', tuple(_STEPS_STATUS))]
    for name in _ingest_modules()[len(_REQUIRED_MODULES):]:
        table = getattr(importlib.import_module(f'{_PACKAGE}.{name}'), 'STATUS', None)
        if table is not None:
            tables.append((name, tuple(table)))
    return _extend_status(tables)


def _extend_status(tables: Sequence[Tuple[str, Tuple[str, ...]]]) -> Tuple[str, ...]:
    """The last of ``[(module, table), ...]``, after checking that each table starts with the one before it."""
    for (prev_name, prev), (name, table) in zip(tables[:-1], tables[1:]):
        if table[:len(prev)] != prev:
            raise RuntimeError(f'{name}.STATUS does not start with {prev_name}.STATUS: status codes are only ever '
                               'appended')
    return tables[-1][1]


def _ingest_modules() -> Tuple[str, ...]:
    """``_REQUIRED_MODULES``, then those of ``_LATER_MODULES`` whose file exists next to this one."""
    here = Path(__file__).resolve().parent
    return _REQUIRED_MODULES + tuple(name for name in _LATER_MODULES if (here / f'{name}.py').is_file())


def _versions() -> dict:
    """``{'torch', 'rdkit', 'spglib', 'ccdc'}``: installed versions, None where not installed."""
    out = {'torch': torch.__version__}
    for name, distributions in _VERSION_DISTRIBUTIONS:
        out[name] = None
        for dist in distributions:
            try:
                out[name] = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    return out


def _sha256(path: str, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


def _as_path(path) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f'path must be a str or os.PathLike, got {type(path).__name__}')
    return os.path.abspath(os.fspath(path))


def _check_steps_run(steps_run) -> list:
    """``steps_run`` as a JSON-ready list of dicts, each naming its step."""
    if steps_run is None:
        return []
    if not isinstance(steps_run, (list, tuple)) or not all(isinstance(s, dict) for s in steps_run):
        raise TypeError('steps_run must be a list of dicts, one per step run')
    for i, step in enumerate(steps_run):
        if not isinstance(step.get('step'), str) or not step['step']:
            raise ValueError(f"steps_run[{i}] needs a 'step' name (a non-empty str)")
    return json.loads(_to_json('steps_run', list(steps_run)))


def _check_extra_meta(extra_meta) -> dict:
    if extra_meta is None:
        return {}
    if not isinstance(extra_meta, dict) or not all(isinstance(k, str) for k in extra_meta):
        raise TypeError('extra_meta must be a dict with str keys')
    return json.loads(_to_json('extra_meta', extra_meta))


def _to_json(name, value) -> str:
    try:
        return json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f'{name} must be JSON-serialisable (no NaN or infinity): {e}') from e


def _check_batch(batch, table):
    """``(ptr_cpu, status_cpu)`` of a batch the store accepts (codes of ``table``); raises ``StoreError`` otherwise."""
    if not getattr(batch, 'is_batch', False):
        raise StoreError(f'save stores a batch (MolCrystalData.read / from_arrays), not a {type(batch).__name__}')
    missing = [k for k in _REQUIRED if k not in batch._store]
    if missing:
        raise StoreError(f'the batch lacks {missing}: the store needs the row bookkeeping and the ingestion '
                         'status of read / from_arrays')
    unknown = sorted(set(batch.__dict__) - {'_store'} - set(_SAVED_DICT_KEYS) - set(_CACHE_DICT_KEYS))
    if unknown:
        raise StoreError(f'instance attributes {unknown} live outside the store and cannot be saved')
    ptr_cpu = batch['ptr'].detach().to('cpu').long()
    B = len(ptr_cpu) - 1
    status = batch['status']
    if not torch.is_tensor(status) or status.shape != (B,) or status.is_floating_point():
        raise StoreError(f'status must be an integer tensor [{B}]')
    status_cpu = status.detach().to('cpu').long()
    bad = sorted(set(status_cpu[(status_cpu < 0) | (status_cpu >= len(table))].tolist()))
    if bad:
        raise StoreError(f'status codes {bad} are outside the status table (status_table(), 0-{len(table) - 1})')
    if not isinstance(batch['message'], (list, tuple)) or len(batch['message']) != B:
        raise StoreError(f'message must be a list of {B} strings')
    if not torch.is_tensor(batch['done']) or batch['done'].shape != (B,):
        raise StoreError(f'done must be a tensor [{B}]')
    return ptr_cpu, status_cpu


def _write_manifest(file, batch, status_names, B):
    """manifest.csv: one line per row (``_MANIFEST_COLUMNS``)."""
    columns = {}
    for name in ('identifier', 'source_path', 'block_name', 'message'):
        value = batch._store.get(name, None)
        columns[name] = list(value) if isinstance(value, (list, tuple)) and len(value) == B else [''] * B
    columns['status'] = status_names
    with open(file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('row',) + _MANIFEST_COLUMNS)
        for b in range(B):
            writer.writerow([b] + ['' if columns[c][b] is None else str(columns[c][b]) for c in _MANIFEST_COLUMNS])


def _level_map(batch, attrs, ptr_cpu, B) -> dict:
    """meta.json's ``levels``: per attribute, its level from ``_slice_dict``, encoding, and dtype / shape.

    The levels of ``MXtalBase``'s level contract, read from the bookkeeping alone: ``atom`` (sliced by ``ptr``),
    ``row`` (by ``arange``), ``atom_or_row`` (both coincide), ``unit`` (by the unit pointer of ``z_prime``), ``edge``
    (any other entry that describes the tensor: its own per-row counts; the unit and edge names since stage 3,
    2026-10-23, both were ``sliced``), ``sliced`` (an entry that describes no level of it: a stale entry, or one on a
    non-tensor), ``batch`` (no entry) and ``bookkeeping`` (``ptr``, ``batch``).
    """
    from mxtaltools.dataset_utils.data_classes import _UnitPointer, _entry_pointer   # (data_classes imports this)
    slice_dict = batch.__dict__.get('_slice_dict') or {}
    row_slices = torch.arange(B + 1)
    units = _UnitPointer(batch, B)
    out = {}
    for key, enc in attrs.items():
        slices = slice_dict.get(key)
        if key in ('ptr', 'batch'):
            level = 'bookkeeping'
        elif not torch.is_tensor(slices):
            level = 'batch'
        else:
            s = slices.detach().to('cpu').long()
            atom = s.shape == ptr_cpu.shape and torch.equal(s, ptr_cpu)
            row = s.shape == row_slices.shape and torch.equal(s, row_slices)
            level = 'atom_or_row' if atom and row else 'atom' if atom else 'row' if row else 'sliced'
            value = batch._store[key]
            if level == 'sliced' and _entry_pointer(batch, key, value, slices, B) is not None:
                unit_ptr = units.get()
                level = 'unit' if unit_ptr is not None and torch.equal(s, unit_ptr) else 'edge'
        entry = dict(level=level, encoding=enc['kind'])
        if enc['kind'] in ('tensor', 'ndarray'):
            entry.update(dtype=enc.get('dtype', str(enc['value'].dtype).replace('torch.', '')),
                         shape=list(enc['value'].shape))
        out[key] = entry
    return out


###############################################################################
# encoding
###############################################################################

def _encode(key, value, clones) -> dict:
    """One stored value as a weights_only-safe dict (module docstring, ``kind``)."""
    if torch.is_tensor(value):
        return dict(kind='tensor', value=_tensor_copy(f'attribute {key!r}', value, clones))
    if isinstance(value, np.ndarray):
        _check_array_dtype(key, value.dtype)
        return dict(kind='ndarray', value=torch.from_numpy(value.copy()), dtype=value.dtype.name)
    if type(value) in (bool, int, float, str):
        return dict(kind='scalar', value=value)
    if type(value) in (list, tuple):
        return _encode_sequence(key, value)
    if type(value) is dict and _is_plain(value):
        return dict(kind='plain', value=value)
    raise StoreError(f'attribute {key!r}: a {type(value).__name__} cannot be stored exactly (module docstring)')


def _encode_bookkeeping(name, key, value, clones):
    """A ``_slice_dict`` / ``_inc_dict`` entry: a tensor (shared with the store where it is the same object) or None."""
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise StoreError(f'{name}[{key!r}] is a {type(value).__name__}; the store keeps tensors and None')
    return _tensor_copy(f'{name}[{key!r}]', value, clones)


def _tensor_copy(key, t, clones):
    """A detached CPU copy of ``t``, made once per tensor object so objects shared in the batch stay shared."""
    if type(t) is not torch.Tensor or t.layout != torch.strided or t.is_quantized:
        raise StoreError(f'{key}: a {type(t).__name__} with layout {t.layout} cannot be stored exactly; '
                         'the store keeps plain dense tensors')
    hit = clones.get(id(t))
    if hit is None:
        # the original is kept with its copy so its id cannot be reused within this save
        hit = clones[id(t)] = (t, t.detach().to('cpu', copy=True))
    return hit[1]


def _check_array_dtype(key, dtype):
    if dtype.kind not in _ARRAY_KINDS or not dtype.isnative:
        raise StoreError(f'attribute {key!r}: numpy dtype {dtype} cannot be stored exactly (bool, integer, float '
                         'or complex in native byte order only)')


def _mask(items) -> torch.Tensor:
    return torch.tensor([v is None for v in items], dtype=torch.bool)


def _ptr_from_counts(counts) -> torch.Tensor:
    return torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(counts, dtype=torch.long).cumsum(0)])


def _encode_sequence(key, seq) -> dict:
    """A list or tuple: str_list, str_lists, array_list or plain, in that order of preference."""
    container = type(seq).__name__
    items = list(seq)
    present = [v for v in items if v is not None]
    if all(type(v) is str for v in present):
        return dict(kind='str_list', container=container, values=['' if v is None else v for v in items],
                    none=_mask(items))
    if all(type(v) in (list, tuple) and all(type(s) is str for s in v) for v in present) \
            and len({type(v) for v in present}) == 1:
        return dict(kind='str_lists', container=container, inner=type(present[0]).__name__,
                    values=[s for v in present for s in v], ptr=_ptr_from_counts([len(v or ()) for v in items]),
                    none=_mask(items))
    if all(isinstance(v, np.ndarray) or torch.is_tensor(v) for v in present):
        return _encode_arrays(key, items, container)
    if _is_plain(seq):
        return dict(kind='plain', value=seq)
    raise StoreError(f'attribute {key!r}: a {container} of {sorted({type(v).__name__ for v in present})} cannot be '
                     'stored exactly (module docstring)')


def _encode_arrays(key, items, container) -> dict:
    """A list of numpy arrays or tensors (None allowed) of one dtype and rank, concatenated with a pointer."""
    present = [v for v in items if v is not None]
    if all(isinstance(v, np.ndarray) for v in present):
        lib = 'numpy'
        dtypes = {v.dtype for v in present}
        if len(dtypes) != 1:
            raise StoreError(f'attribute {key!r}: arrays of several dtypes {sorted(map(str, dtypes))}')
        _check_array_dtype(key, present[0].dtype)
        dtype = present[0].dtype.name
        tensors = [torch.from_numpy(v.copy()) for v in present]
    elif all(torch.is_tensor(v) for v in present):
        lib = 'torch'
        dtypes = {v.dtype for v in present}
        if len(dtypes) != 1:
            raise StoreError(f'attribute {key!r}: tensors of several dtypes {sorted(map(str, dtypes))}')
        tensors = [_tensor_copy(f'attribute {key!r}', v, {}) for v in present]
        dtype = str(present[0].dtype).replace('torch.', '')
    else:
        raise StoreError(f'attribute {key!r}: a list mixing numpy arrays and tensors')
    ranks = {t.ndim for t in tensors}
    if len(ranks) != 1:
        raise StoreError(f'attribute {key!r}: arrays of several ranks {sorted(ranks)}')
    rank = ranks.pop()
    shapes = [tuple(t.shape) for t in tensors]
    varying = [ax for ax in range(rank) if len({s[ax] for s in shapes}) > 1]

    enc = dict(kind='array_list', container=container, lib=lib, dtype=dtype, none=_mask(items))
    shape_of = iter(shapes)
    if rank >= 1 and len(varying) <= 1:
        # one varying axis (or none: axis 0): one tensor along it, entry b = ptr[b]:ptr[b + 1]
        axis = varying[0] if varying else 0
        counts = [0 if v is None else next(shape_of)[axis] for v in items]
        enc.update(layout='axis', axis=axis, values=torch.cat(tensors, dim=axis), ptr=_ptr_from_counts(counts))
    else:
        # several varying axes, or rank 0: flat values, each entry's shape kept
        row_shapes = [(0,) * rank if v is None else next(shape_of) for v in items]
        counts = [0 if v is None else int(np.prod(s, dtype=np.int64)) for v, s in zip(items, row_shapes)]
        enc.update(layout='flat', values=torch.cat([t.reshape(-1) for t in tensors]),
                   ptr=_ptr_from_counts(counts),
                   shapes=torch.tensor(row_shapes, dtype=torch.long).reshape(len(items), rank))
    return enc


def _is_plain(value, depth: int = 0) -> bool:
    """None, bool, int, float, str, or lists / tuples / dicts (str or int keys) of them, at most ``_MAX_DEPTH`` deep."""
    if value is None or type(value) in (bool, int, float, str):
        return True
    if depth >= _MAX_DEPTH:
        return False
    if type(value) in (list, tuple):
        return all(_is_plain(v, depth + 1) for v in value)
    if type(value) is dict:
        return all(type(k) in (str, int) and _is_plain(v, depth + 1) for k, v in value.items())
    return False


###############################################################################
# decoding
###############################################################################

_CONTAINERS = {'list': list, 'tuple': tuple}


def _decode(key, enc):
    """The value ``_encode`` encoded as ``enc``."""
    try:
        kind = enc['kind']
        if kind == 'tensor':
            return enc['value']
        if kind == 'ndarray':
            return _as_numpy(key, enc['value'], enc['dtype'])
        if kind in ('scalar', 'plain'):
            return enc['value']
        container = _CONTAINERS[enc['container']]
        none = enc['none'].tolist()
        if kind == 'str_list':
            return container(None if n else v for v, n in zip(enc['values'], none))
        ptr = enc['ptr'].tolist()
        if kind == 'str_lists':
            inner = _CONTAINERS[enc['inner']]
            values = enc['values']
            return container(None if none[b] else inner(values[ptr[b]:ptr[b + 1]]) for b in range(len(none)))
        if kind == 'array_list':
            return container(_decode_array_entry(key, enc, ptr, b) if not none[b] else None for b in range(len(none)))
    except (KeyError, TypeError, IndexError, AttributeError) as e:
        raise StoreError(f'attribute {key!r} does not decode: {type(e).__name__}: {e}') from e
    raise StoreError(f'attribute {key!r} has unknown kind {kind!r}')


def _decode_array_entry(key, enc, ptr, b):
    values = enc['values']
    if enc['layout'] == 'axis':
        t = values.narrow(enc['axis'], ptr[b], ptr[b + 1] - ptr[b])
    else:
        t = values[ptr[b]:ptr[b + 1]].reshape(enc['shapes'][b].tolist())
    return _as_numpy(key, t, enc['dtype']) if enc['lib'] == 'numpy' else t.clone()


def _as_numpy(key, t, dtype_name):
    """A C-contiguous numpy copy of ``t``, checked to have the saved dtype."""
    out = t.numpy().copy()
    if out.dtype.name != dtype_name:
        raise StoreError(f'attribute {key!r}: rebuilt dtype {out.dtype.name}, saved {dtype_name}')
    return out

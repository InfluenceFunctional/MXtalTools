"""Canonical crystal datasets on disk: save, load, and convert back to MolCrystalData rows.

A dataset is a folder holding one row per crystal: a Z' = 1 ``CrystalDescription`` in the standard setting
``SYM_OPS[sg]`` (in practice a canonical form: ``ops.reduce_cell`` where it applies, then ``domains.fold``), the row's
reference molecule (a ``MoleculeRef``) with its ``reference_handedness``, an identifier, a status and any per-row
columns, and optionally a manifest that accounts for every row of the source the dataset was built from.

What a row means
----------------
Asymmetric unit at ``T_fc t + R diag(h, 1, 1) X``, ``X`` the reference molecule's body coordinates (Angstrom, centred
on the heavy-atom centroid, exactly ``MoleculeRef.pos``). ``reference_handedness`` is the handedness under which
``align_mol_batch_to_standard_axes`` (GFN: ``orient_molecule(mode='std', target_handedness=...)``) leaves ``X``
where it is: -1 for a reference that is the mirror image of the aligned stored molecule (owner decision D2 of
2026-10-16, ``io_mxt.ImportInfo.mirrored``), +1 otherwise. Aligning such an ``X`` at +1 returns its mirror image, and a
crystal built from that is a different, usually clashing, crystal (``io_mxt`` module docstring, Handedness). A consumer
must therefore keep the saved molecule and its ``reference_handedness``; it must not re-derive them from coordinates.

Layout (schema ``crystal_fd.dataset/1``)
----------------------------------------
``meta.json``
    ``schema``; ``rows`` and ``atoms``; ``status_names`` (the meaning of each status code); ``columns`` (per column:
    dtype, shape and the writer's description); ``table_hash`` (``tables.TABLE_HASH`` when written: canonical forms are
    defined by the table, so ``load_dataset`` refuses a dataset written against another table unless told otherwise);
    ``files`` (sha256 and size of each data file); ``manifest`` (row count and code counts, when one was written);
    ``code`` (``code_digest()`` when written: a sha256 over this package's ``*.py`` and ``data/*`` files and one per
    file; canonical forms also depend on code outside the table, e.g. ``ops.reduce_cell``'s tie-breaks, so
    ``load_dataset`` warns, or with ``code='raise'`` refuses, when the package has changed since and names the files);
    ``provenance`` (free-form, from the writer).
``data.pt``
    A dict of tensors and lists of strings (``torch.load(..., weights_only=True)`` reads it): ``sg`` [B] long; ``cell``
    [B, 6] float64 (a, b, c in Angstrom; alpha, beta, gamma in radians); ``t`` [B, 3] float64 (fractional); ``R``
    [B, 3, 3] float64 (proper); ``h`` [B] long; ``reference_handedness`` [B] long; ``status`` [B] long (index into
    ``status_names``); ``identifier`` list[B] of str; ``mol_ptr`` [B + 1] long; ``mol_pos`` [N, 3] float64; ``mol_z``
    [N] uint8; ``mol_hash`` list[B] of str (``MoleculeRef.hash``); ``columns`` {name: tensor with leading dimension B,
    or list[B]}.
``manifest.pt`` and ``manifest.csv`` (optional)
    One entry per source row: ``source_row`` [M] long (a permutation of 0..M-1), ``identifier`` list[M], ``code``
    list[M] of str, ``dataset_row`` [M] long (the row of ``data.pt`` it became, -1 when it is not in the dataset;
    every dataset row is named by exactly one source row) and ``detail`` {name: [M] tensor or list}.

The folder is written under a temporary name and renamed into place; ``save_dataset`` refuses an existing path.

Code identity
-------------
``save_dataset`` records ``code_digest()`` of the package as it is when the folder is written. A writer that
canonicalised its rows in another process (or earlier) must itself check that the code was the same then (the CSD
build records the digest in every shard and refuses to assemble shards whose digest differs from the code's). The
digest names bytes, not behaviour: an edit to a comment changes it too, which is why a mismatch warns by default.

Conversion back to MolCrystalData
---------------------------------
``to_mol_crystal_data`` writes rows the live builder builds into the described crystal. ``aunit_handedness`` is
``h * reference_handedness``: MolCrystalData's handedness is relative to the molecule's standard (aligned) frame, which
for a mirrored reference is the mirror image of ``X``. ``mode='body'`` (default) stores ``pos = X diag(h, 1, 1)``,
which ``mol2ucell`` / ``analyze`` build as described with ``std_orientation=False`` (no re-alignment; the trainer's
convention) and, through the alignment, with ``std_orientation=True``. ``mode='literal'`` stores the posed asymmetric
unit (Cartesian, as the CSD dataset builders write ``pos``), for ``std_orientation=True`` only. Both rely, under
``std_orientation=True``, on the alignment returning ``X`` (a fixed point; rounding decides it for molecules with
near-degenerate principal moments, ``io_mxt`` module docstring). Centroids are wrapped to [0, 1) first; the live
builder clips them to [0, 1 - ``BUILDER_CLIP``], so rows with a wrapped coordinate in that band (``in_clip_band``) are
refused unless ``clip='allow'``.
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import warnings
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import torch

from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef, molecule_hash
from mxtaltools.crystal_fd.tables import TABLE_HASH

F64 = torch.float64
SCHEMA = 'crystal_fd.dataset/1'
#: the live builder clips fractional centroids to [0, 1 - BUILDER_CLIP] (``MolCrystalOps.assign_aunit_centroid``)
BUILDER_CLIP = 1e-4
#: keys of ``data.pt`` other than ``columns``; column names may not repeat them
CORE_KEYS = ('sg', 'cell', 't', 'R', 'h', 'reference_handedness', 'status', 'identifier', 'mol_ptr', 'mol_pos', 'mol_z',
             'mol_hash')
MODES = ('body', 'literal')
_DATA, _META, _MANIFEST, _MANIFEST_CSV = 'data.pt', 'meta.json', 'manifest.pt', 'manifest.csv'


class DatasetError(ValueError):
    """A dataset folder or its inputs do not satisfy the schema (named field and rows)."""


class CodeMismatchWarning(UserWarning):
    """The crystal_fd package differs from the one that wrote a dataset (``code_mismatch`` names the files)."""


CODE_MODES = ('warn', 'raise', 'ignore')


# ---------------------------------------------------------------------------
# code identity
# ---------------------------------------------------------------------------

def code_digest(package_dir: Optional[str] = None) -> dict:
    """The identity of the crystal_fd code: {'digest': sha256 over the sorted 'path:sha256' lines, 'files': {path:
    sha256}, 'scope': ...} for every ``*.py`` file of the package and every file under its ``data/`` folder (paths
    relative to the package, '/'-separated; README and caches excluded)."""
    root = os.path.dirname(os.path.abspath(__file__)) if package_dir is None else os.path.abspath(package_dir)
    paths = sorted(set(glob.glob(os.path.join(root, '*.py')) + glob.glob(os.path.join(root, 'data', '*'))))
    files = {}
    for p in paths:
        if os.path.isfile(p):
            files[os.path.relpath(p, root).replace(os.sep, '/')] = _sha256(p)
    lines = '\n'.join(f'{k}:{v}' for k, v in sorted(files.items()))
    return dict(digest=hashlib.sha256(lines.encode()).hexdigest(), files=files,
                scope='mxtaltools/crystal_fd: *.py and data/*')


def code_mismatch(meta: dict, current: Optional[dict] = None) -> Optional[list]:
    """Files whose content differs between the code that wrote ``meta`` and ``current`` (default ``code_digest()``):
    [] when the digests agree, None when ``meta`` records no code digest, else a sorted list of 'changed: f',
    'added: f' and 'removed: f'."""
    rec = meta.get('code')
    if not rec or 'digest' not in rec:
        return None
    cur = code_digest() if current is None else current
    if rec['digest'] == cur['digest']:
        return []
    old, new = rec.get('files', {}), cur['files']
    out = [f'changed: {f}' for f in old if f in new and old[f] != new[f]]
    out += [f'removed: {f}' for f in old if f not in new] + [f'added: {f}' for f in new if f not in old]
    return sorted(out) or ['digest differs']


# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------

def pack_molecules(mols: Sequence[MoleculeRef]):
    """(pos [N, 3] float64, z [N] uint8, ptr [B + 1] long, hashes list[B]) of ``MoleculeRef``s, coordinates bit-exact."""
    if len(mols) == 0:
        return torch.zeros(0, 3, dtype=F64), torch.zeros(0, dtype=torch.uint8), torch.zeros(1, dtype=torch.long), []
    n = torch.tensor([len(m.z) for m in mols], dtype=torch.long)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(n, 0)])
    z = torch.cat([torch.as_tensor(m.z).reshape(-1).long() for m in mols])
    if bool(((z < 1) | (z > 255)).any()):
        raise DatasetError('atomic numbers must lie in 1-255 to be stored as uint8')
    pos = torch.cat([torch.as_tensor(m.pos).reshape(-1, 3).to(F64) for m in mols])
    return pos, z.to(torch.uint8), ptr, [m.hash for m in mols]


def molecule_at(pos: torch.Tensor, z: torch.Tensor, ptr: torch.Tensor, hashes: Sequence[str], i: int) -> MoleculeRef:
    """The ``MoleculeRef`` of packed row ``i``, exactly as saved (no recentring, which would move its bits)."""
    a, b = int(ptr[i]), int(ptr[i + 1])
    return MoleculeRef(pos=pos[a:b], z=z[a:b].long(), hash=hashes[i])


# ---------------------------------------------------------------------------
# the in-memory dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FDDataset:
    """Rows of a saved dataset (all, or those of chosen statuses), in saved order.

    ``desc`` (Z' = 1, float64); packed molecules ``mol_pos``, ``mol_z``, ``mol_ptr``, ``mol_hash`` (``molecule(i)``);
    ``reference_handedness`` [B] long; ``identifier`` list[B]; ``status`` [B] long into ``status_names``;
    ``columns`` {name: [B] tensor or list}; ``index`` [B] long, each row's position in ``data.pt``; ``meta`` the
    folder's ``meta.json``.
    """
    desc: CrystalDescription
    mol_pos: torch.Tensor
    mol_z: torch.Tensor
    mol_ptr: torch.Tensor
    mol_hash: list
    reference_handedness: torch.Tensor
    identifier: list
    status: torch.Tensor
    status_names: tuple
    columns: dict
    index: torch.Tensor
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.desc.num_rows

    def molecule(self, i: int) -> MoleculeRef:
        return molecule_at(self.mol_pos, self.mol_z, self.mol_ptr, self.mol_hash, int(i))

    def molecules(self, rows=None) -> list:
        rows = range(len(self)) if rows is None else [int(r) for r in torch.as_tensor(rows).reshape(-1).tolist()]
        return [self.molecule(i) for i in rows]

    def status_of(self, i: int) -> str:
        return self.status_names[int(self.status[i])]

    def select(self, rows) -> 'FDDataset':
        """The dataset restricted to ``rows`` (positions in this dataset; bool mask or indices), molecules repacked."""
        rows = torch.as_tensor(rows)
        if rows.dtype == torch.bool:
            rows = torch.nonzero(rows)[:, 0]
        rows = rows.reshape(-1).long()
        mols = self.molecules(rows)
        pos, z, ptr, hashes = pack_molecules(mols)
        r = rows.tolist()
        cols = {k: ([v[i] for i in r] if isinstance(v, list) else v[rows]) for k, v in self.columns.items()}
        return replace(self, desc=self.desc[rows], mol_pos=pos, mol_z=z, mol_ptr=ptr, mol_hash=hashes,
                       reference_handedness=self.reference_handedness[rows], identifier=[self.identifier[i] for i in r],
                       status=self.status[rows], columns=cols, index=self.index[rows])

    def to_mol_crystal_data(self, rows=None, mode: str = 'body', clip: str = 'raise', extra_columns=()) -> list:
        """``to_mol_crystal_data`` of these rows (all by default) with their identifiers; ``extra_columns`` names
        list columns (e.g. ``'smiles'``) copied onto each row as attributes."""
        sub = self if rows is None else self.select(rows)
        extra = {k: sub.columns[k] for k in extra_columns}
        return to_mol_crystal_data(sub.desc, sub.molecules(), sub.reference_handedness, mode=mode,
                                   identifier=sub.identifier, clip=clip, extra=extra)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------

def _sha256(path: str, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                return h.hexdigest()
            h.update(b)


def _check_column(name: str, v, B: int):
    if name in CORE_KEYS:
        raise DatasetError(f'column {name!r} repeats a core key')
    if isinstance(v, (list, tuple)):
        if len(v) != B:
            raise DatasetError(f'column {name!r} has {len(v)} entries, expected {B}')
        if not all(x is None or isinstance(x, (str, int, float, bool)) for x in v):
            raise DatasetError(f'list column {name!r} must hold str, int, float, bool or None')
        return list(v)
    v = torch.as_tensor(v)
    if v.ndim == 0 or v.shape[0] != B:
        raise DatasetError(f'column {name!r} has shape {tuple(v.shape)}, expected leading dimension {B}')
    return v.detach().cpu().clone()


def _describe(v) -> dict:
    if isinstance(v, list):
        return dict(dtype='list', shape=[len(v)])
    return dict(dtype=str(v.dtype).replace('torch.', ''), shape=list(v.shape))


def _check_manifest(man: dict, B: int, identifier: list) -> dict:
    for k in ('source_row', 'identifier', 'code', 'dataset_row'):
        if k not in man:
            raise DatasetError(f'manifest lacks {k!r}')
    src = torch.as_tensor(man['source_row']).long().reshape(-1)
    dst = torch.as_tensor(man['dataset_row']).long().reshape(-1)
    M = len(src)
    code, ident = list(man['code']), list(man['identifier'])
    if len(dst) != M or len(code) != M or len(ident) != M:
        raise DatasetError('manifest fields have different lengths')
    if not torch.equal(torch.sort(src).values, torch.arange(M)):
        raise DatasetError('manifest source_row must name every source row 0..M-1 exactly once')
    if not all(isinstance(c, str) and c for c in code):
        raise DatasetError('every manifest code must be a non-empty string')
    if bool(((dst < -1) | (dst >= B)).any()):
        raise DatasetError(f'manifest dataset_row must lie in [-1, {B})')
    named = dst[dst >= 0]
    if not torch.equal(torch.sort(named).values, torch.arange(B)):
        raise DatasetError('every dataset row must be named by exactly one manifest entry')
    for j in torch.nonzero(dst >= 0)[:, 0].tolist():
        if str(ident[j]) != str(identifier[int(dst[j])]):
            raise DatasetError(f'manifest entry {j} names dataset row {int(dst[j])} with identifier {ident[j]!r}, '
                               f'the dataset row has {identifier[int(dst[j])]!r}')
    detail = {k: _check_column(f'detail.{k}', v, M) for k, v in (man.get('detail') or {}).items()}
    return dict(source_row=src, identifier=[str(x) for x in ident], code=code, dataset_row=dst, detail=detail)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def save_dataset(path: str, desc: CrystalDescription, mols: Sequence[MoleculeRef], identifier: Sequence[str],
                 reference_handedness=None, status=None, status_names: Sequence[str] = ('canonical',),
                 columns: Optional[dict] = None, column_docs: Optional[dict] = None, manifest: Optional[dict] = None,
                 provenance: Optional[dict] = None) -> dict:
    """Write a dataset folder at ``path`` (which must not exist); returns its ``meta.json`` as a dict, which records
    ``code_digest()`` of the package at this call (module docstring, Code identity).

    Parameters
    ----------
    desc : CrystalDescription
        Z' = 1 rows, h = +-1, proper R (``validate``), sg 1-230; stored float64.
    mols : sequence of MoleculeRef
        One per row, saved bit-exactly (``MoleculeRef.pos`` float64, ``z``, ``hash``); each hash is checked.
    identifier : sequence of str
        One per row.
    reference_handedness : [B] +-1, optional
        Default +1 (module docstring).
    status, status_names
        [B] integer codes into ``status_names`` (default: every row ``status_names[0]``).
    columns, column_docs
        Per-row columns ({name: tensor with leading dimension B, or list[B] of str/int/float/bool/None}) and a
        description per column, written to ``meta.json``.
    manifest : dict, optional
        ``source_row``, ``identifier``, ``code``, ``dataset_row`` and optional ``detail`` (module docstring); checked.
    provenance : dict, optional
        JSON-serialisable; stored in ``meta.json``.

    Raises
    ------
    FileExistsError
        If ``path`` exists.
    DatasetError
        If an input violates the schema.
    """
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; a dataset is never overwritten')
    d = desc.double().to('cpu')
    B = d.num_rows
    if bool((d.z_prime != 1).any()) or d.max_z_prime != 1:
        raise DatasetError("only Z' = 1 descriptions can be saved")
    try:
        d.validate()
    except ValueError as e:
        raise DatasetError(f'descriptions do not validate: {e}') from e
    mols = list(mols)
    identifier = [str(x) for x in identifier]
    if len(mols) != B or len(identifier) != B:
        raise DatasetError(f'{B} descriptions, {len(mols)} molecules, {len(identifier)} identifiers')
    rh = torch.ones(B, dtype=torch.long) if reference_handedness is None \
        else torch.as_tensor(reference_handedness).reshape(-1).long().cpu()
    if len(rh) != B or not bool((rh.abs() == 1).all()):
        raise DatasetError('reference_handedness must hold +1 or -1 per row')
    status_names = tuple(str(s) for s in status_names)
    if not status_names or len(set(status_names)) != len(status_names):
        raise DatasetError('status_names must be non-empty and distinct')
    st = torch.zeros(B, dtype=torch.long) if status is None else torch.as_tensor(status).reshape(-1).long().cpu()
    if len(st) != B or bool(((st < 0) | (st >= len(status_names))).any()):
        raise DatasetError(f'status must hold codes 0..{len(status_names) - 1}, one per row')
    pos, z, ptr, hashes = pack_molecules(mols)
    for i, m in enumerate(mols):
        if molecule_hash(m.pos, m.z) != m.hash:
            raise DatasetError(f'molecule of row {i} does not match its hash')
    cols = {k: _check_column(k, v, B) for k, v in (columns or {}).items()}
    man = None if manifest is None else _check_manifest(manifest, B, identifier)

    tmp = f'{path}.partial-{os.getpid()}'
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    data = dict(sg=d.sg.clone(), cell=d.cell.clone(), t=d.t[:, 0].clone(), R=d.R[:, 0].clone(), h=d.h[:, 0].clone(),
                reference_handedness=rh, status=st, identifier=identifier, mol_ptr=ptr, mol_pos=pos, mol_z=z,
                mol_hash=hashes, columns=cols)
    torch.save(data, os.path.join(tmp, _DATA))
    files = {_DATA: None}
    man_meta = None
    if man is not None:
        torch.save(man, os.path.join(tmp, _MANIFEST))
        files[_MANIFEST] = None
        _write_manifest_csv(os.path.join(tmp, _MANIFEST_CSV), man)
        files[_MANIFEST_CSV] = None
        codes = {}
        for c in man['code']:
            codes[c] = codes.get(c, 0) + 1
        man_meta = dict(rows=len(man['code']), in_dataset=int((man['dataset_row'] >= 0).sum()),
                        codes=dict(sorted(codes.items(), key=lambda kv: -kv[1])))
    for f in files:
        files[f] = dict(sha256=_sha256(os.path.join(tmp, f)), bytes=os.path.getsize(os.path.join(tmp, f)))
    column_docs = column_docs or {}
    counts = {s: int((st == k).sum()) for k, s in enumerate(status_names)}
    meta = dict(schema=SCHEMA, rows=B, atoms=int(len(z)), status_names=list(status_names), status_counts=counts,
                table_hash=TABLE_HASH, code=code_digest(), files=files,
                columns={k: dict(**_describe(v), doc=str(column_docs.get(k, ''))) for k, v in cols.items()},
                manifest=man_meta, provenance=provenance or {},
                row_meaning="asymmetric unit at T_fc t + R diag(h, 1, 1) X, X = mol_pos[mol_ptr[i]:mol_ptr[i+1]] "
                            "(Angstrom, heavy-atom centred); standard setting SYM_OPS[sg]; cell angles in radians; "
                            "reference_handedness = the alignment handedness that leaves X fixed (-1: mirrored "
                            "reference, owner decision D2)")
    with open(os.path.join(tmp, _META), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1, default=str)
    if os.path.exists(path):                                       # created meanwhile by someone else
        shutil.rmtree(tmp)
        raise FileExistsError(f'{path} exists; a dataset is never overwritten')
    os.rename(tmp, path)
    return meta


def _write_manifest_csv(path: str, man: dict):
    detail = man['detail']
    names = list(detail)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['source_row', 'identifier', 'code', 'dataset_row'] + names)
        cols = [detail[k] if isinstance(detail[k], list) else detail[k].tolist() for k in names]
        for j in range(len(man['code'])):
            w.writerow([int(man['source_row'][j]), man['identifier'][j], man['code'][j], int(man['dataset_row'][j])]
                       + [c[j] for c in cols])


def read_meta(path: str) -> dict:
    with open(os.path.join(path, _META), encoding='utf-8') as f:
        meta = json.load(f)
    if meta.get('schema') != SCHEMA:
        raise DatasetError(f'{path}: schema {meta.get("schema")!r}, expected {SCHEMA!r}')
    return meta


def _verify_files(path: str, meta: dict, names):
    for name in names:
        rec = meta['files'].get(name)
        if rec is None:
            raise DatasetError(f'{path}: meta.json lists no digest for {name}')
        got = _sha256(os.path.join(path, name))
        if got != rec['sha256']:
            raise DatasetError(f'{path}/{name}: sha256 {got} differs from meta.json ({rec["sha256"]})')


def check_code(path: str, meta: dict, code: str = 'warn') -> Optional[list]:
    """Compare the code that wrote the dataset at ``path`` (``meta['code']``) with this package (``code_mismatch``):
    ``'warn'`` issues a ``CodeMismatchWarning``, ``'raise'`` a ``DatasetError``, ``'ignore'`` nothing, when the
    digests differ or none was recorded. Files added since, with every recorded file unchanged, do neither: the
    recorded code cannot call a module it did not have without one of its files changing. Returns the mismatch list
    (None: no digest recorded)."""
    if code not in CODE_MODES:
        raise ValueError(f'code must be one of {CODE_MODES}')
    if code == 'ignore':
        return code_mismatch(meta)
    diff = code_mismatch(meta)
    if diff == [] or (diff is not None and all(x.startswith('added: ') for x in diff)):
        return diff
    msg = (f'{path} records no crystal_fd code digest: whether its rows are this code\'s canonical forms is unknown'
           if diff is None else
           f'{path} was written by other crystal_fd code ({len(diff)} files differ: {diff[:8]}'
           f'{" ..." if len(diff) > 8 else ""}); its rows may not be this code\'s canonical forms')
    if code == 'raise':
        raise DatasetError(msg)
    warnings.warn(msg, CodeMismatchWarning, stacklevel=3)
    return diff


def load_dataset(path: str, status='canonical', verify: bool = True, verify_molecules: bool = False,
                 allow_table_mismatch: bool = False, code: str = 'warn') -> FDDataset:
    """Read a dataset folder.

    Parameters
    ----------
    status : str, sequence of str, or None
        Keep rows whose status name is given (default ``'canonical'``); None keeps every row.
    verify : bool
        Check ``data.pt`` against its sha256 in ``meta.json`` and the stored arrays against the schema.
    verify_molecules : bool
        Also recompute every molecule hash (slower).
    allow_table_mismatch : bool
        Read a dataset written against another ``TABLE_HASH`` (its canonical forms may not be this table's).
    code : str
        What to do when the crystal_fd code differs from the code that wrote the dataset, or the dataset records no
        code digest (``check_code``): ``'warn'`` (default), ``'raise'`` or ``'ignore'``.

    Raises
    ------
    DatasetError
        On a schema, digest, hash or table mismatch, an unknown status name, or (``code='raise'``) a code mismatch.
    """
    meta = read_meta(path)
    if meta['table_hash'] != TABLE_HASH and not allow_table_mismatch:
        raise DatasetError(f'{path} was written against table {meta["table_hash"]}, this package has {TABLE_HASH}')
    check_code(path, meta, code)
    if verify:
        _verify_files(path, meta, [_DATA])
    data = torch.load(os.path.join(path, _DATA), weights_only=True)
    B = int(meta['rows'])
    names = tuple(meta['status_names'])
    if verify:
        for k in CORE_KEYS:
            if k not in data:
                raise DatasetError(f'{path}: data.pt lacks {k!r}')
        if data['sg'].shape != (B,) or data['cell'].shape != (B, 6) or data['t'].shape != (B, 3) \
                or data['R'].shape != (B, 3, 3) or data['mol_ptr'].shape != (B + 1,) or len(data['identifier']) != B:
            raise DatasetError(f'{path}: array shapes do not match {B} rows')
        if int(data['mol_ptr'][-1]) != len(data['mol_z']) or len(data['mol_pos']) != len(data['mol_z']):
            raise DatasetError(f'{path}: molecule arrays do not match mol_ptr')
    if verify_molecules:
        for i in range(B):
            a, b = int(data['mol_ptr'][i]), int(data['mol_ptr'][i + 1])
            if molecule_hash(data['mol_pos'][a:b], data['mol_z'][a:b].long()) != data['mol_hash'][i]:
                raise DatasetError(f'{path}: molecule of row {i} does not match its hash')
    desc = CrystalDescription.from_parts(data['sg'], data['cell'], data['t'], data['R'], data['h'])
    full = FDDataset(desc=desc, mol_pos=data['mol_pos'], mol_z=data['mol_z'], mol_ptr=data['mol_ptr'],
                     mol_hash=list(data['mol_hash']), reference_handedness=data['reference_handedness'],
                     identifier=list(data['identifier']), status=data['status'], status_names=names,
                     columns=dict(data['columns']), index=torch.arange(B), meta=meta)
    if status is None:
        return full
    want = (status,) if isinstance(status, str) else tuple(status)
    unknown = [s for s in want if s not in names]
    if unknown:
        raise DatasetError(f'{path}: unknown status {unknown}; the dataset has {list(names)}')
    keep = torch.zeros(B, dtype=torch.bool)
    for s in want:
        keep |= data['status'] == names.index(s)
    return full if bool(keep.all()) else full.select(keep)


def load_manifest(path: str, verify: bool = True) -> dict:
    """The folder's manifest (``source_row``, ``identifier``, ``code``, ``dataset_row``, ``detail``); raises
    ``DatasetError`` when there is none or it fails its digest."""
    meta = read_meta(path)
    if not meta.get('manifest'):
        raise DatasetError(f'{path} has no manifest')
    if verify:
        _verify_files(path, meta, [_MANIFEST])
    return torch.load(os.path.join(path, _MANIFEST), weights_only=True)


# ---------------------------------------------------------------------------
# back to MolCrystalData
# ---------------------------------------------------------------------------

def wrap_centroids(t: torch.Tensor) -> torch.Tensor:
    """Fractional centroids wrapped to [0, 1), a value within 1e-9 below 1 (rounding of a centroid on the 0 face) moved
    to just below 0, which the builder's clip returns to 0."""
    w = t - torch.floor(t)
    return torch.where(w > 1 - 1e-9, w - 1, w)


def in_clip_band(desc: CrystalDescription) -> torch.Tensor:
    """[B] bool: a wrapped centroid coordinate lies in (1 - BUILDER_CLIP, 1 - 1e-9], where the live builder's clip
    would move it (by up to BUILDER_CLIP, fractional)."""
    return (wrap_centroids(desc.double().t[:, 0]) > 1 - BUILDER_CLIP).any(1)


def to_mol_crystal_data(desc: CrystalDescription, mols, reference_handedness=None, mode: str = 'body',
                        identifier: Optional[Sequence[str]] = None, clip: str = 'raise',
                        extra: Optional[dict] = None) -> list:
    """MolCrystalData rows (float32, ``do_box_analysis=True``) that the live builder builds into ``desc``'s crystals.

    Parameters
    ----------
    desc : CrystalDescription
        Z' = 1 rows in the standard setting; centroids are wrapped (``wrap_centroids``) first.
    mols : MoleculeRef or sequence of MoleculeRef
        One per row, or one for every row.
    reference_handedness : [B] +-1, optional
        Default +1. Written as ``aunit_handedness = h * reference_handedness`` and as the attribute
        ``reference_handedness`` ([1] long) on each row.
    mode : str
        ``'body'``: ``pos = X diag(h, 1, 1)``; build with ``std_orientation=False`` (exact; the trainer's convention)
        or ``True`` (through the alignment). ``'literal'``: ``pos`` the posed asymmetric unit in Cartesian
        coordinates; build with ``std_orientation=True`` only (module docstring).
    identifier : sequence of str, optional
    clip : str
        ``'raise'`` (default) refuses rows ``in_clip_band``; ``'allow'`` writes them (the builder will clip them).
    extra : dict, optional
        {attribute: list[B] or [B] tensor} copied onto each row (e.g. smiles).
    """
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if clip not in ('raise', 'allow'):
        raise ValueError("clip must be 'raise' or 'allow'")
    d = desc.double().to('cpu')
    B = d.num_rows
    if bool((d.z_prime != 1).any()):
        raise NotImplementedError("Z' > 1")
    if isinstance(mols, MoleculeRef):
        mols = [mols] * B
    mols = list(mols)
    if len(mols) != B:
        raise ValueError(f'{B} descriptions, {len(mols)} molecules')
    rh = torch.ones(B, dtype=torch.long) if reference_handedness is None \
        else torch.as_tensor(reference_handedness).reshape(-1).long().cpu()
    if len(rh) != B or not bool((rh.abs() == 1).all()):
        raise ValueError('reference_handedness must hold +1 or -1 per row')
    t = wrap_centroids(d.t[:, 0])
    band = (t > 1 - BUILDER_CLIP).any(1)
    if clip == 'raise' and bool(band.any()):
        bad = torch.nonzero(band)[:, 0].tolist()
        raise ValueError(f'{len(bad)} rows have a wrapped centroid in the builder clip band (1 - {BUILDER_CLIP}, 1): '
                         f'rows {bad[:10]}{" ..." if len(bad) > 10 else ""}')
    d = d.replace(t=t[:, None])
    L, A, cc, rv, h = d.crystal_params()
    h = h[:, 0]
    T = d.cell_matrix()
    extra = extra or {}
    for k, v in extra.items():
        if len(v) != B:
            raise ValueError(f'extra {k!r} has {len(v)} entries, expected {B}')
    rows = []
    for i in range(B):
        X = torch.as_tensor(mols[i].pos, dtype=F64)
        flip = torch.tensor([float(h[i]), 1.0, 1.0], dtype=F64)
        if mode == 'body':
            pos = X * flip
        else:
            pos = (d.R[i, 0] @ (X * flip).T).T + T[i] @ t[i]
        z = torch.as_tensor(mols[i].z).long()
        m = MolData(z=z.clone(), pos=pos.float(), x=torch.zeros(len(z), 1), do_mol_analysis=True)
        row = MolCrystalData(molecule=m, sg_ind=int(d.sg[i]), cell_lengths=L[i].float(), cell_angles=A[i].float(),
                             aunit_centroid=cc[i].float(), aunit_orientation=rv[i].float(),
                             aunit_handedness=(h[i:i + 1] * rh[i:i + 1]).long(),
                             identifier=None if identifier is None else str(identifier[i]), do_box_analysis=True)
        row.reference_handedness = rh[i:i + 1].clone()
        for k, v in extra.items():
            setattr(row, k, v[i])
        rows.append(row)
    return rows

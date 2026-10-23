"""The CSD provenance side table: per refcode, what the database records about the study.

The CCDC-exported CIFs carry no temperature, pressure, R factor or deposition number;
the database entry (``ccdc.io.EntryReader('CSD').entry(refcode)``) does. One pass over
the refcodes writes them to a side table keyed by refcode, for deduplication and quality
choices (§5, Table 9, of ``docs/design/crystal_ingestion.md`` in the GFN repository).

``build_provenance(refcodes, out_dir)`` reads the entries in chunks, writes each chunk
under ``out_dir/chunks/`` (temporary name, then rename), and once every chunk is on disk
merges them into the finished folder and removes ``chunks/``. A run that stops is
resumed by calling it again with the same refcodes and chunk size: finished chunks are
kept, never re-read. Every temporary file, those of ``data.pt`` and ``meta.json``
included, is written inside ``chunks/``, so a run killed at any point leaves nothing
outside it; a folder holding only a ``chunks/`` of temporary files (a run killed before
its plan was written) starts afresh.

Layout
------
``data.pt``
    One dict, readable with ``torch.load(path, weights_only=True)``: ``schema``, ``rows``
    and one column per field, row ``i`` describing ``refcode[i]`` in the order given.

    - ``refcode`` (list of str), ``family`` (list of str): the refcode's first six
      letters, shared by every redetermination of one compound;
    - ``found`` (bool): the refcode is in the database. A row with ``found`` False holds
      the absent value in every other column;
    - ``temperature_raw`` (list of str): the database string as given, e.g.
      ``'at 185 K'``; ``temperature_k`` (float64): kelvin parsed from it
      (``parse_temperature_kelvin``), NaN when absent or unparseable;
    - ``pressure_raw`` (list of str), e.g. ``'0.51 GPa'``; not parsed;
    - ``r_factor`` (float64), in percent as the database gives it, NaN when absent;
    - ``is_powder_study``, ``has_disorder``, ``is_polymeric`` (bool): True only where
      the database says True; each with ``<flag>_known`` (bool), False where the
      database holds no value or the refcode is not found. The database leaves
      ``is_powder_study`` empty for about half the entries;
    - ``radiation_source`` (list of str), e.g. ``'X-ray'``, ``'Unknown'``;
    - ``ccdc_number`` (int64), -1 when absent;
    - ``deposition_date`` (list of str), ISO ``YYYY-MM-DD``;
    - ``polymorph`` (list of str), the database's label, e.g. ``'polymorph II'``;
    - ``database_name`` (list of str), the database the entry came from;
    - ``message`` (list of str): ``'not in the database'`` for a missing refcode, the
      attribute and exception for a field the CCDC API failed to return, else ''.

    An absent string is ''.
``meta.json``
    ``schema``; ``rows``; ``counts`` (found, not found, values present per field, flags
    true / false / unknown, value counts of the low-cardinality strings, temperatures
    present but unparsed with their most common strings); ``refcodes_sha256`` (over the
    refcodes joined by newlines); ``chunk_size`` and ``chunks`` (index, start, stop,
    seconds spent reading entries, when written); ``read_seconds`` (their sum);
    ``code`` (sha256 of this module); ``versions`` (torch, csd-python-api, the CSD
    data version); ``files`` (sha256 and size of ``data.pt``); ``written_utc``.

Missing refcodes are rows with ``found`` False, never an exception; any other failure to
open an entry (a licence or database fault) raises, so it cannot pass as "not found".
"""
import collections
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

__all__ = ['SCHEMA', 'build_provenance', 'load_provenance', 'parse_temperature_kelvin']

SCHEMA = 'mxtaltools.csd_provenance/1'
_DATA, _META, _CHUNKS, _PLAN = 'data.pt', 'meta.json', 'chunks', 'plan.json'

#: The CCDC API's message for a refcode the database does not hold.
_NOT_FOUND = 'is not in the database'

_STR_FIELDS = ('family', 'temperature_raw', 'pressure_raw', 'radiation_source', 'deposition_date', 'polymorph',
               'database_name', 'message')
_FLOAT_FIELDS = ('temperature_k', 'r_factor')
_FLAGS = ('is_powder_study', 'has_disorder', 'is_polymeric')
_INT_FIELDS = ('ccdc_number',)
_COLUMNS = ('refcode', 'found') + _STR_FIELDS + _FLOAT_FIELDS + _FLAGS + tuple(f + '_known' for f in _FLAGS) \
           + _INT_FIELDS

#: String fields whose distinct values meta.json counts (a handful each).
_VALUE_COUNTED = ('radiation_source', 'database_name')

#: Entry attribute -> column, for the plain-string fields.
_STR_ATTRS = (('temperature', 'temperature_raw'), ('pressure', 'pressure_raw'),
              ('radiation_source', 'radiation_source'), ('polymorph', 'polymorph'), ('database_name', 'database_name'))

#: ``[at] <number>[(esd)] K`` or ``[at] <number>[(esd)] deg.C``, optional trailing full stop.
_TEMPERATURE = re.compile(r'^\s*(?:at\s+)?(-?\d+(?:\.\d*)?)(?:\(\d+\))?\s*(K|deg\.?\s*C)\s*\.?\s*$', re.IGNORECASE)

#: The suffix ``_tmp_path`` gives a file before its rename.
_TMP = re.compile(r'\.tmp-\d+$')

#: How many distinct unparsed temperature strings meta.json lists.
_N_UNPARSED_EXAMPLES = 25


def parse_temperature_kelvin(raw) -> float:
    """Kelvin from a CSD temperature string, NaN when absent or not one temperature.

    Parameters
    ----------
    raw : str or None
        The database string: ``'240 K'``, ``'at 185 K'``, ``'at 98. K'``, ``'100(2) K'``,
        ``'at -100 deg.C'``, ``'at 23 deg.C.'`` (forms seen in the CSD).

    Returns
    -------
    float
        The temperature in kelvin (deg.C + 273.15). NaN for None, '', a range
        (``'240-260 K'``), a phrase (``'at room temperature'``), any other form, or a
        value at or below 0 K.
    """
    if not raw:
        return math.nan
    m = _TEMPERATURE.match(raw)
    if m is None:
        return math.nan
    value = float(m.group(1))
    if not m.group(2).upper().startswith('K'):
        value += 273.15
    return value if value > 0 else math.nan


def build_provenance(refcodes, out_dir, *, chunk_size: int = 20000, resume: bool = True) -> dict:
    """Read the CSD entry of every refcode and write the provenance table to ``out_dir``.

    Parameters
    ----------
    refcodes : sequence of str
        CSD refcodes, distinct; the table keeps their order.
    out_dir : str or os.PathLike
        The folder to write (module docstring). Created if absent.
    chunk_size : int
        Refcodes per chunk file; the unit of resumption.
    resume : bool
        If ``out_dir`` exists: True continues a run of the same refcodes and chunk size
        (keeping its finished chunks) or returns the finished table's meta; False refuses.

    Returns
    -------
    dict
        The meta.json written (or found, for a finished run).

    Raises
    ------
    TypeError
        ``refcodes`` holds a non-str, or ``chunk_size`` is not an int.
    ValueError
        Duplicate refcodes, ``chunk_size`` below 1, or ``out_dir`` holding a run of other
        refcodes or another chunk size, or other files.
    FileExistsError
        ``out_dir`` exists and ``resume`` is False.
    RuntimeError
        The CCDC API fails to open an entry for any reason other than a missing refcode.
    """
    refcodes = _check_refcodes(refcodes)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError(f'chunk_size must be an int, got {type(chunk_size).__name__}')
    if chunk_size < 1:
        raise ValueError(f'chunk_size must be at least 1, got {chunk_size}')
    out_dir = os.path.abspath(os.fspath(out_dir))
    refcodes_sha256 = _refcodes_sha256(refcodes)
    plan = dict(schema=SCHEMA, rows=len(refcodes), chunk_size=chunk_size, refcodes_sha256=refcodes_sha256)
    chunks_dir = os.path.join(out_dir, _CHUNKS)

    if os.path.exists(out_dir):
        if not resume:
            raise FileExistsError(f'{out_dir} exists; pass resume=True to continue a run there')
        finished = os.path.join(out_dir, _META)
        if os.path.exists(finished):
            meta = _read_json(finished)
            if meta.get('schema') != SCHEMA or meta.get('refcodes_sha256') != refcodes_sha256:
                raise ValueError(f'{out_dir} holds a finished table of other refcodes '
                                 f'({meta.get("refcodes_sha256")}, not {refcodes_sha256})')
            if os.path.isdir(chunks_dir):           # a run that stopped after writing meta.json
                shutil.rmtree(chunks_dir)
            return meta
        plan_file = os.path.join(chunks_dir, _PLAN)
        if os.path.exists(plan_file):
            if _read_json(plan_file) != plan:
                raise ValueError(f'{out_dir} holds an unfinished run of other refcodes or chunk size: '
                                 f'{_read_json(plan_file)}, not {plan}')
        elif not _unplanned(out_dir, chunks_dir):
            raise ValueError(f'{out_dir} holds files that are not a provenance run: {sorted(os.listdir(out_dir))[:5]}')
    os.makedirs(chunks_dir, exist_ok=True)
    if not os.path.exists(os.path.join(chunks_dir, _PLAN)):
        _atomic_json(os.path.join(chunks_dir, _PLAN), plan)

    from ccdc import io as ccdc_io                  # imported here: importing ccdc checks the licence
    reader = ccdc_io.EntryReader('CSD')
    bounds = [(start, min(start + chunk_size, len(refcodes))) for start in range(0, len(refcodes), chunk_size)]
    try:
        for index, (start, stop) in enumerate(bounds):
            file = _chunk_file(chunks_dir, index)
            if os.path.exists(file):
                continue
            t0 = time.perf_counter()
            columns = _read_chunk(reader, refcodes[start:stop])
            chunk = dict(schema=SCHEMA, index=index, start=start, stop=stop, columns=columns,
                         seconds=time.perf_counter() - t0,
                         written_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            _atomic_torch_save(file, chunk)
    finally:
        reader.close()

    columns, chunk_records = _merge_chunks(chunks_dir, bounds, refcodes)
    data = dict(schema=SCHEMA, rows=len(refcodes), **columns)
    _atomic_torch_save(os.path.join(out_dir, _DATA), data, tmp_dir=chunks_dir)
    meta = dict(schema=SCHEMA, rows=len(refcodes), counts=_counts(columns), refcodes_sha256=refcodes_sha256,
                chunk_size=chunk_size, chunks=chunk_records,
                read_seconds=round(sum(c['seconds'] for c in chunk_records), 3),
                code=dict(file='dataset_utils/construction/csd_provenance.py', sha256=_sha256(__file__)),
                versions=_versions(ccdc_io),
                files={_DATA: dict(sha256=_sha256(os.path.join(out_dir, _DATA)),
                                   bytes=os.path.getsize(os.path.join(out_dir, _DATA)))},
                written_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    _atomic_json(os.path.join(out_dir, _META), meta, tmp_dir=chunks_dir)
    shutil.rmtree(chunks_dir)
    return meta


def load_provenance(path, *, verify: bool = True) -> dict:
    """The data.pt dict of a finished provenance folder.

    Parameters
    ----------
    path : str or os.PathLike
        A folder written by ``build_provenance``.
    verify : bool
        Check ``data.pt`` against its sha256 in meta.json first.

    Returns
    -------
    dict
        The columns (module docstring), with ``schema`` and ``rows``.

    Raises
    ------
    FileNotFoundError
        ``path`` holds no finished table.
    ValueError
        Another schema, or a ``data.pt`` that differs from its recorded sha256.
    """
    path = os.path.abspath(os.fspath(path))
    meta = _read_json(os.path.join(path, _META))
    if meta.get('schema') != SCHEMA:
        raise ValueError(f'{path}: schema {meta.get("schema")!r}, expected {SCHEMA!r}')
    file = os.path.join(path, _DATA)
    if verify and _sha256(file) != meta['files'][_DATA]['sha256']:
        raise ValueError(f'{file} differs from the sha256 in meta.json; it changed after it was written')
    data = torch.load(file, weights_only=True, map_location='cpu')
    if data.get('schema') != SCHEMA or data.get('rows') != meta['rows']:
        raise ValueError(f'{file} is not the {meta["rows"]}-row {SCHEMA!r} table meta.json describes')
    return data


###############################################################################
# reading entries
###############################################################################

def _read_chunk(reader, refcodes: list) -> dict:
    """The columns for ``refcodes``, one database lookup each."""
    rows = [_entry_record(reader, refcode) for refcode in refcodes]
    out = dict(refcode=list(refcodes), found=torch.tensor([r['found'] for r in rows], dtype=torch.bool))
    for f in _STR_FIELDS:
        out[f] = [r[f] for r in rows]
    for f in _FLOAT_FIELDS:
        out[f] = torch.tensor([r[f] for r in rows], dtype=torch.float64)
    for f in _FLAGS:
        out[f] = torch.tensor([r[f] is True for r in rows], dtype=torch.bool)
        out[f + '_known'] = torch.tensor([r[f] is not None for r in rows], dtype=torch.bool)
    for f in _INT_FIELDS:
        out[f] = torch.tensor([r[f] for r in rows], dtype=torch.int64)
    return out


def _entry_record(reader, refcode: str) -> dict:
    """One refcode's fields; the absent values (and ``found`` False) when the database lacks it."""
    record = dict(found=False, family=refcode[:6], temperature_raw='', pressure_raw='', radiation_source='',
                  deposition_date='', polymorph='', database_name='', message='', temperature_k=math.nan,
                  r_factor=math.nan, is_powder_study=None, has_disorder=None, is_polymeric=None, ccdc_number=-1)
    try:
        entry = reader.entry(refcode)
    except RuntimeError as e:
        if _NOT_FOUND not in str(e):
            raise
        record['message'] = 'not in the database'
        return record
    record['found'] = True
    errors = []

    def get(attr):
        try:
            return getattr(entry, attr)
        except Exception as e:                      # one field the API cannot return: recorded, counted in meta
            errors.append(f'{attr}: {type(e).__name__}: {e}')
            return None

    for attr, column in _STR_ATTRS:
        value = get(attr)
        record[column] = '' if value is None else str(value)
    record['temperature_k'] = parse_temperature_kelvin(record['temperature_raw'])
    r_factor = get('r_factor')
    record['r_factor'] = math.nan if r_factor is None else float(r_factor)
    for f in _FLAGS:
        value = get(f)
        record[f] = None if value is None else bool(value)
    number = get('ccdc_number')
    record['ccdc_number'] = -1 if number is None else int(number)
    date = get('deposition_date')
    record['deposition_date'] = '' if date is None else date.isoformat() if hasattr(date, 'isoformat') else str(date)
    record['message'] = '; '.join(errors)
    return record


###############################################################################
# chunks, merge, counts
###############################################################################

def _chunk_file(chunks_dir: str, index: int) -> str:
    return os.path.join(chunks_dir, f'chunk_{index:05d}.pt')


def _merge_chunks(chunks_dir: str, bounds: list, refcodes: list):
    """All chunks concatenated in order, each checked against the refcodes it was planned to hold."""
    parts, records = [], []
    for index, (start, stop) in enumerate(bounds):
        file = _chunk_file(chunks_dir, index)
        chunk = torch.load(file, weights_only=True, map_location='cpu')
        if (chunk.get('schema'), chunk.get('start'), chunk.get('stop')) != (SCHEMA, start, stop) \
                or chunk['columns']['refcode'] != refcodes[start:stop]:
            raise ValueError(f'{file} does not hold refcodes {start}:{stop} of this run')
        parts.append(chunk['columns'])
        records.append(dict(index=index, start=start, stop=stop, seconds=round(chunk['seconds'], 3),
                            written_utc=chunk['written_utc']))
    columns = {}
    for key in _COLUMNS:
        values = [p[key] for p in parts]
        if not values:
            columns[key] = _empty(key)
        elif isinstance(values[0], torch.Tensor):
            columns[key] = torch.cat(values)
        else:
            columns[key] = [v for part in values for v in part]
    return columns, records


def _empty(key: str):
    if key == 'refcode' or key in _STR_FIELDS:
        return []
    dtype = (torch.float64 if key in _FLOAT_FIELDS else torch.int64 if key in _INT_FIELDS else torch.bool)
    return torch.zeros(0, dtype=dtype)


def _counts(columns: dict) -> dict:
    """What meta.json reports about the table (module docstring)."""
    found = columns['found']
    n_found = int(found.sum())
    present = {f: sum(1 for v in columns[f] if v) for f in _STR_FIELDS if f not in ('family', 'message')}
    present['temperature_k'] = int((~torch.isnan(columns['temperature_k'])).sum())
    present['r_factor'] = int((~torch.isnan(columns['r_factor'])).sum())
    present['ccdc_number'] = int((columns['ccdc_number'] >= 0).sum())
    flags = {}
    for f in _FLAGS:
        known = columns[f + '_known']
        flags[f] = dict(true=int((columns[f] & known).sum()), false=int((~columns[f] & known).sum()),
                        unknown=int((~known).sum()))
    unparsed = collections.Counter(raw for raw, k in zip(columns['temperature_raw'], columns['temperature_k'].tolist())
                                   if raw and math.isnan(k))
    field_errors = sum(1 for m, f in zip(columns['message'], found.tolist()) if f and m)
    return dict(rows=len(columns['refcode']), found=n_found, not_found=len(columns['refcode']) - n_found,
                families=len(set(columns['family'])), present=present, flags=flags,
                values={f: dict(collections.Counter(columns[f]).most_common()) for f in _VALUE_COUNTED},
                temperature_unparsed=sum(unparsed.values()),
                temperature_unparsed_examples=dict(unparsed.most_common(_N_UNPARSED_EXAMPLES)),
                rows_with_field_errors=field_errors)


###############################################################################
# files
###############################################################################

def _check_refcodes(refcodes) -> list:
    if isinstance(refcodes, str):
        raise TypeError('refcodes must be a sequence of str, not one str')
    refcodes = list(refcodes)
    bad = [r for r in refcodes if not isinstance(r, str)]
    if bad:
        raise TypeError(f'refcodes must be str; got {type(bad[0]).__name__} ({bad[0]!r})')
    if len(set(refcodes)) != len(refcodes):
        dupes = [r for r, n in collections.Counter(refcodes).items() if n > 1]
        raise ValueError(f'{len(dupes)} refcodes repeat, e.g. {dupes[:5]}')
    return refcodes


def _refcodes_sha256(refcodes: list) -> str:
    return hashlib.sha256('\n'.join(refcodes).encode()).hexdigest()


def _versions(ccdc_io) -> dict:
    out = dict(torch=torch.__version__, csd_python_api=None, csd_version=None)
    try:
        out['csd_python_api'] = importlib.metadata.version('csd-python-api')
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        out['csd_version'] = str(ccdc_io.csd_version())
    except Exception:                               # not every API version has it; the meta records None
        pass
    return out


def _unplanned(out_dir: str, chunks_dir: str) -> bool:
    """``out_dir`` is empty, or holds only a ``chunks/`` of temporary files: a run killed before its plan landed."""
    contents = os.listdir(out_dir)
    if not contents:
        return True
    if contents != [_CHUNKS] or not os.path.isdir(chunks_dir):
        return False
    return all(_TMP.search(name) for name in os.listdir(chunks_dir))


def _tmp_path(path: str, tmp_dir) -> str:
    """The temporary name ``path`` is written under before its rename, in ``tmp_dir`` (default: beside it)."""
    return os.path.join(tmp_dir or os.path.dirname(path), f'{os.path.basename(path)}.tmp-{os.getpid()}')


def _atomic_torch_save(path: str, obj, tmp_dir=None) -> None:
    tmp = _tmp_path(path, tmp_dir)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_json(path: str, obj, tmp_dir=None) -> None:
    tmp = _tmp_path(path, tmp_dir)
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=1, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _sha256(path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(Path(path), 'rb') as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)

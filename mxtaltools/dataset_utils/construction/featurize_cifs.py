"""Featurize CIF files at CSD scale: a named recipe over chunks, resumable, concurrent, collated explicitly.

The CSD-scale runner of the ingestion contract (§4 of ``docs/design/crystal_ingestion.md``
in the GFN repository). ``featurize_cifs(paths, out_dir, recipe=..., chunk_size=...)`` reads
the files chunk by chunk with ``MolCrystalData.read(..., full=recipe)`` and saves each chunk
as an ingest store (``crystal_store``); ``collate(out_dir, recipe=...)`` joins the chunk
stores into one; ``reclaim(out_dir)`` frees the chunks of processes that died. It fixes the
old featurizer's faults: every chunk rewritten on every run, collation of whatever chunks
exist, refusals sent only to stdout.

Rules
-----
- The sorted path list and the chunk map are written first (``paths.txt``, ``run.json``); a
  later call with another list or chunk size for the same ``out_dir`` is refused.
- Any number of processes may run at once. A process takes a chunk by creating its claim
  file atomically, and never waits on a claim: a chunk claimed by another is skipped.
- Code identity (``run_identity``): the sha256 of every ``*.py`` of the mxtaltools package
  (not only the ingestion modules: the stored values also come from geometry, building and
  constants modules) and the installed versions of the libraries that compute them (torch,
  numpy, scipy, torch_geometric, rdkit, spglib, ccdc), and python's. Chunks are kept per
  (recipe, identity): ``<runkey>/`` with ``runkey = '<recipe>-' + sha256(identity digest,
  recipe digest)[:12]``, so a code or recipe change starts a fresh set beside the old one,
  which is never deleted and stays visible (``run_status(...)['runs']``).
- Written, one definition for every caller (``_chunk_done``): the chunk's store and manifest
  exist and match its run key's ``recipe.json`` (recipe digest, identity digest, and the
  ingestion-module digest the store's meta records), the chunk's paths and the store's
  data digest. A worker skips such a chunk; status, collate and reclaim judge by it.
- A worker re-reads the identity before each chunk it would claim and after each chunk it
  saves. When the code changed since it started (an edit, an install), it stops: a chunk
  saved meanwhile goes to ``chunks/orphans/`` and its claim to ``claims/freed/`` (its store's
  meta would name code that did not produce it). Start a worker on the new code to continue
  (a new run key), or restore the old code. Run production from a tree nobody edits.
- Termination: each process walks the finite chunk list at most once and processes a chunk
  at most once. A chunk whose processing raises is recorded as failed and keeps its claim; a
  crashed process's claims stay until an explicit ``reclaim``, never freed by waiting.
- ``collate`` refuses unless every chunk of the map is written; ``allow_partial=True``
  collates the written ones, and the collated store records that and which are missing. It
  collates the current code's run key, or a named ``runkey``; a run key made by other code
  than is installed needs ``allow_other_code=True``, recorded with the differences.
- Writes: only under ``out_dir``, which must not contain any input file. Stores are written
  under a temporary name and renamed; manifests and records by a temporary file and a
  rename. Nothing is deleted: a freed claim, an orphaned store or manifest (not written as
  above) and a dead process's partial store are moved into ``freed/`` and ``orphans/``
  folders.
- Paths: absolute, no duplicates, and no control or format character (a newline, a tab, a
  byte-order mark, a zero-width space). The command line reads its paths file as UTF-8 with
  an optional byte-order mark.

Layout of ``out_dir``
---------------------
``paths.txt``
    The sorted paths, one per line (UTF-8).
``run.json``
    ``schema``, ``n_paths``, ``paths_sha256`` (of ``paths.txt``), ``chunk_size`` and
    ``chunks``: ``[start, stop)`` per chunk, indices into ``paths.txt``.
``<runkey>/recipe.json``
    The recipe's spec and digest and the identity (``run_identity``) of the chunks below.
``<runkey>/claims/chunk_NNNNNN.json``
    The claim: worker, pid, host, process start time, UTC time. ``claims/freed/`` keeps
    the claims ``reclaim`` freed.
``<runkey>/chunks/chunk_NNNNNN/``, ``chunk_NNNNNN.json``
    The chunk's store, then its manifest (rows, atoms, statuses, seconds, digests).
    ``chunk_NNNNNN.failed.json`` records a chunk whose processing raised.
``<runkey>/logs/<worker>.log``
    One line per chunk: files, rows, statuses, seconds.
``<runkey>/collated/`` (``collated_partial_<k>of<n>/``)
    The collated store.

Command line
------------
::

    python -m mxtaltools.dataset_utils.construction.featurize_cifs --paths-file F --out D \\
        --recipe csd_v1 --chunk-size N [--worker-id K] [--threads T] [--cif-dir DIR]
    python -m mxtaltools.dataset_utils.construction.featurize_cifs --out D --recipe csd_v1 --status [--runkey R]
    python -m mxtaltools.dataset_utils.construction.featurize_cifs --out D --reclaim [--chunks I ...]
    python -m mxtaltools.dataset_utils.construction.featurize_cifs --out D --recipe csd_v1 --collate \\
        [--allow-partial] [--runkey R [--allow-other-code]]
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import socket
import sys
import time
import traceback
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch_geometric.data.storage import GlobalStorage

from mxtaltools.dataset_utils.data_class_methods.crystal_recipes import get_recipe
from mxtaltools.dataset_utils.data_class_methods.crystal_store import (StoreError, code_digest, read_meta,
                                                                       status_table)

__all__ = ['featurize_cifs', 'collate', 'reclaim', 'run_status', 'run_identity', 'concat_batches', 'RunnerError',
           'ROW_PADS', 'SCHEMA', 'main']

SCHEMA = 'mxtaltools.featurize_cifs/2'

#: The mxtaltools package folder, every ``*.py`` of which ``run_identity`` hashes.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

#: Installed distributions whose versions are part of the identity (read from the installed
#: metadata, so nothing is imported: importing ccdc checks its licence).
_IDENTITY_DISTRIBUTIONS = (('torch', ('torch',)), ('numpy', ('numpy',)), ('scipy', ('scipy',)),
                           ('torch_geometric', ('torch-geometric', 'torch_geometric')),
                           ('rdkit', ('rdkit', 'rdkit-pypi')), ('spglib', ('spglib',)),
                           ('ccdc', ('csd-python-api', 'ccdc')))

#: The value each step writes in the unused slots of a row attribute Z'max wide (``z_prime``
#: below the batch's largest): ``concat_batches`` pads a chunk's narrower attribute with it,
#: which is what one batch of all the rows holds there. Another attribute whose trailing
#: shape differs between batches is refused.
ROW_PADS = {'aunit_centroid': 0, 'aunit_orientation': 0, 'aunit_handedness': 0, 'aunit_op': -1,
            'unit_radius': float('nan'), 'unit_mass': float('nan'), 'unit_volume': float('nan'),
            'unit_fingerprint': 0}

_NAME_RE = re.compile(r'[A-Za-z0-9_.-]{1,64}')
_RUNKEY_RE = re.compile(r'[A-Za-z0-9_.-]+-[0-9a-f]{12}')
_PATHS, _RUN, _RECIPE = 'paths.txt', 'run.json', 'recipe.json'
_RUN_KEYS = ('schema', 'n_paths', 'paths_sha256', 'chunk_size', 'chunks')


class RunnerError(RuntimeError):
    """The state of an ``out_dir`` does not allow the call (named chunks and why)."""


class _CodeChanged(RunnerError):
    """The code changed under a running worker (``featurize_cifs`` stops)."""


###############################################################################
# code identity
###############################################################################

def run_identity() -> dict:
    """The identity of the code a worker runs: every module of the mxtaltools package and the libraries' versions.

    Returns
    -------
    dict
        ``digest``: sha256 over the sorted ``files`` lines, ``versions`` and ``python``;
        ``files``: {path relative to the mxtaltools package, '/' separated: sha256} for every
        ``*.py`` of the package (all of it, so a module a later chunk loads, or one loaded
        inside a function, is covered as well as the ingestion modules); ``versions``:
        {name: installed version, or None} of ``_IDENTITY_DISTRIBUTIONS``; ``python``
        (``major.minor.micro``); ``store_code``: ``crystal_store.code_digest()['digest']``,
        the narrower digest every chunk store's meta.json records.

    Notes
    -----
    It names bytes, not behaviour: any edit to the package, a comment included, gives a
    new identity and so a new run key.
    """
    files = {p.relative_to(_PACKAGE_ROOT).as_posix(): _sha_file(p) for p in sorted(_PACKAGE_ROOT.rglob('*.py'))}
    versions = {}
    for name, distributions in _IDENTITY_DISTRIBUTIONS:
        versions[name] = None
        for dist in distributions:
            try:
                versions[name] = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    python = '.'.join(str(v) for v in sys.version_info[:3])
    lines = ([f'file:{k}:{v}' for k, v in sorted(files.items())] + [f'version:{k}:{v}' for k, v in sorted(versions.items())]
             + [f'python:{python}'])
    return dict(digest=hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest(), files=files, versions=versions,
                python=python, store_code=code_digest()['digest'])


def _identity_changes(was: dict, now: dict) -> list:
    """What differs from identity ``was`` to ``now``: 'changed: f', 'added: f', 'removed: f', 'version: n a -> b'."""
    a, b = was.get('files', {}), now.get('files', {})
    out = [f'changed: {f}' for f in sorted(set(a) & set(b)) if a[f] != b[f]]
    out += [f'added: {f}' for f in sorted(set(b) - set(a))] + [f'removed: {f}' for f in sorted(set(a) - set(b))]
    va, vb = was.get('versions', {}), now.get('versions', {})
    out += [f'version: {n} {va.get(n)} -> {vb.get(n)}' for n in sorted(set(va) | set(vb)) if va.get(n) != vb.get(n)]
    if was.get('python') != now.get('python'):
        out.append(f'python: {was.get("python")} -> {now.get("python")}')
    if not out and was.get('digest') != now.get('digest'):
        out.append('digest differs')
    return out


def _summary(changes, n: int = 5) -> str:
    return ', '.join(changes[:n]) + (f' and {len(changes) - n} more' if len(changes) > n else '')


###############################################################################
# the worker
###############################################################################

def featurize_cifs(paths: Sequence, out_dir, *, recipe, chunk_size: int, worker_id: Optional[str] = None,
                   log: Optional[Callable[[str], None]] = print) -> dict:
    """Featurize ``paths`` into chunk stores under ``out_dir``, taking every chunk no other process holds.

    Parameters
    ----------
    paths : sequence of str or os.PathLike
        Absolute paths of CIF files, no duplicates, none containing a newline; sorted here.
        Each file becomes one or more rows (``MolCrystalData.read``).
    out_dir : str or os.PathLike
        Created if absent. It must not contain any input file, and an existing one must be a
        folder of this runner (or empty).
    recipe : str or Recipe
        A name in ``crystal_recipes.RECIPES`` (``'csd_v1'``) or a ``Recipe``.
    chunk_size : int
        Files per chunk (positive).
    worker_id : str, optional
        This process's label in claims and logs (1-64 of ``[A-Za-z0-9_.-]``); default
        ``<host>-<pid>``.
    log : callable or None
        Called with each log line (default ``print``); every line also goes to
        ``<runkey>/logs/<worker>.log``.

    Returns
    -------
    dict
        ``runkey``, ``worker``, ``n_chunks``; chunk lists ``written``, ``failed``,
        ``skipped_done``, ``skipped_claimed`` (this call's outcome per chunk); ``stopped``:
        None, or why the walk stopped early (the code changed since the call started; the
        chunks after it were not visited).

    Raises
    ------
    ValueError
        Bad ``paths``, ``chunk_size`` or ``worker_id``; an ``out_dir`` holding another path
        list or chunk size, or input files, or that is not a folder of this runner.
    TypeError, NotImplementedError
        From ``get_recipe``.
    """
    recipe = get_recipe(recipe)
    if not _NAME_RE.fullmatch(recipe.name):
        raise ValueError(f'recipe name {recipe.name!r} must be 1-64 of [A-Za-z0-9_.-] (it names a folder)')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f'chunk_size must be a positive int, got {chunk_size!r}')
    worker = f'{socket.gethostname()}-{os.getpid()}' if worker_id is None else worker_id
    if not isinstance(worker, str) or not _NAME_RE.fullmatch(worker):
        raise ValueError(f'worker_id must be 1-64 of [A-Za-z0-9_.-], got {worker!r}')
    paths = _sorted_paths(paths)
    out_dir = _out_dir(out_dir, paths)
    run = _prepare_run(out_dir, paths, chunk_size)
    identity = run_identity()
    key = _runkey(recipe, identity)
    rdir = os.path.join(out_dir, key)
    for sub in ('claims', 'chunks', 'logs'):
        os.makedirs(os.path.join(rdir, sub), exist_ok=True)
    record = _record_recipe(rdir, recipe, identity)
    log_file = os.path.join(rdir, 'logs', f'{worker}.log')

    def say(line):
        line = f'{_utc()} {worker} {line}'
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        if log is not None:
            log(line)

    chunks = run['chunks']
    out = dict(runkey=key, worker=worker, n_chunks=len(chunks), written=[], failed=[], skipped_done=[],
               skipped_claimed=[], stopped=None)
    say(f'start: {len(paths)} files, {len(chunks)} chunks of {chunk_size}, recipe {recipe.name} '
        f'({recipe.digest[:12]}), code {identity["digest"][:12]}')
    for k, (start, stop) in enumerate(chunks):          # bounded: at most one pass over a finite list
        chunk_paths = paths[start:stop]
        if _chunk_done(rdir, k, _sha_lines(chunk_paths), record):
            out['skipped_done'].append(k)
            continue
        changes = _identity_changes(identity, run_identity())
        if changes:
            out['stopped'] = (f'the code changed since this worker started ({_summary(changes)}); start a worker on '
                              'the new code (a new run key) or restore the old code')
            say(f'STOPPED before chunk {k}/{len(chunks)}: {out["stopped"]}')
            break
        if not _claim(rdir, k, worker):
            out['skipped_claimed'].append(k)
            continue
        try:
            line = _run_chunk(rdir, k, len(chunks), chunk_paths, recipe, identity, key, worker)
            out['written'].append(k)
            say(line)
        except _CodeChanged as e:
            out['stopped'] = str(e)
            say(f'STOPPED at chunk {k}/{len(chunks)}: {e}')
            break
        except Exception as e:                              # a chunk's failure is recorded; the walk goes on
            _record_failure(rdir, k, worker, e)
            out['failed'].append(k)
            say(f'chunk {k}/{len(chunks)} FAILED: {type(e).__name__}: {str(e)[:300]} (claim kept; see '
                f'chunks/{_chunk_name(k)}.failed.json)')
    say(f'end: written {len(out["written"])}, failed {len(out["failed"])}, already done '
        f'{len(out["skipped_done"])}, claimed by others {len(out["skipped_claimed"])}'
        + ('' if out['stopped'] is None else '; stopped early (code changed)'))
    return out


def _run_chunk(rdir, k, n_chunks, chunk_paths, recipe, identity, key, worker) -> str:
    """Read and run the recipe on one chunk, save its store and then its manifest; the log line.

    Raises ``_CodeChanged`` (after moving the store aside and freeing the claim) when the
    code changed while the chunk ran: ``crystal_store.save`` records the ingestion-module
    digest of the files at save time, which then names code that did not make the rows.
    """
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    started = _utc()
    read = recipe.args('read')
    fmt, dtype = read.pop('format'), read.pop('dtype')
    t0 = time.time()
    batch = MolCrystalData.read(list(chunk_paths), fmt, dtype=dtype, full=recipe, **read)
    t1 = time.time()
    chunks = os.path.join(rdir, 'chunks')
    store = os.path.join(chunks, _chunk_name(k))
    for leftover in (store, os.path.join(chunks, f'{_chunk_name(k)}.json')):
        if os.path.exists(leftover):    # not written (a process died between them, or other code): kept aside, not reused
            _move_aside(leftover, os.path.join(chunks, 'orphans'))
    chunk_sha = _sha_lines(chunk_paths)
    steps_run = [dict(step='read', args=recipe.spec()['steps'][0]['args'])] + list(batch['steps_run'])
    meta = batch.save(store, steps_run=steps_run, extra_meta=dict(featurize_cifs=dict(
        chunk=k, runkey=key, recipe=recipe.spec(), recipe_digest=recipe.digest, paths_sha256=chunk_sha,
        identity_digest=identity['digest'], worker=worker)))
    t2 = time.time()
    changes = _identity_changes(identity, run_identity())
    if meta['code']['digest'] != identity['store_code']:
        changes = changes or [f'the store records ingestion code {meta["code"]["digest"][:12]}, this worker started '
                              f'with {identity["store_code"][:12]}']
    if changes:
        _move_aside(store, os.path.join(chunks, 'orphans'))
        claim = os.path.join(rdir, 'claims', f'{_chunk_name(k)}.json')
        if os.path.exists(claim):                   # absent only when reclaim(chunks=[k]) freed it meanwhile
            _move_aside(claim, os.path.join(rdir, 'claims', 'freed'))
        raise _CodeChanged(f'chunk {k}: the code changed while it ran ({_summary(changes)}); its store went to '
                           f'chunks/orphans and its claim to claims/freed, so it is unwritten and unclaimed under {key}; '
                           'start a worker on the new code (a new run key) or restore the old code')
    counts = {name: n for name, n in meta['status_counts'].items() if n}
    manifest = dict(schema=SCHEMA, chunk=k, runkey=key, recipe=recipe.name, recipe_digest=recipe.digest,
                    identity_digest=identity['digest'], code_digest=identity['store_code'], paths_sha256=chunk_sha,
                    n_paths=len(chunk_paths),
                    first_path=chunk_paths[0], last_path=chunk_paths[-1], rows=meta['rows'], atoms=meta['atoms'],
                    status_counts=counts, done_counts=meta['done_counts'],
                    seconds=dict(read_and_chain=round(t1 - t0, 3), save=round(t2 - t1, 3)),
                    store_data_sha256=meta['files']['data.pt']['sha256'], worker=worker, pid=os.getpid(),
                    host=socket.gethostname(), started_utc=started, finished_utc=_utc())
    _write_atomic(os.path.join(chunks, f'{_chunk_name(k)}.json'), _json_bytes(manifest))
    statuses = ', '.join(f'{n} {name}' for name, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    return (f'chunk {k}/{n_chunks}: {len(chunk_paths)} files -> {meta["rows"]} rows ({statuses}); '
            f'{t1 - t0:.1f} s read and chain, {t2 - t1:.1f} s save')


###############################################################################
# state, reclaim, collate
###############################################################################

def run_status(out_dir, *, recipe, runkey: Optional[str] = None) -> dict:
    """The state of every chunk of ``out_dir`` for ``recipe``, under the current code or a named run key.

    Parameters
    ----------
    recipe : str or Recipe
    runkey : str, optional
        A run key under ``out_dir`` (``runs`` lists them); default the current code's.

    Returns
    -------
    dict
        ``runkey``; ``current`` (it is the installed code's run key); ``n_chunks``; chunk
        lists ``done`` (written, ``_chunk_done``), ``failed`` (a failure record and not
        written), ``claimed`` (claimed, neither written nor failed: running, or its process
        died), ``todo`` (not claimed); ``claims`` {chunk: claim record} for ``claimed`` and
        ``failed``; ``runs`` {run key: summary} for EVERY run key under ``out_dir``, so a set
        written under other code or another recipe is never invisible: ``recipe``,
        ``recipe_digest`` and ``identity`` (12 characters), ``current``, ``done`` (written
        chunks, None when the folder holds no identity record), ``code_changes`` (how the
        installed code differs from the one that wrote it), ``collated`` (its collated
        stores).

    Raises
    ------
    RunnerError
        ``out_dir`` holds no run; a named ``runkey`` that is not a run of this runner there,
        or that ran another recipe.
    ValueError
        A ``runkey`` that is not a run key.
    """
    recipe = get_recipe(recipe)
    out_dir = os.path.abspath(os.fspath(out_dir))
    run, paths = _load_run(out_dir)
    identity = run_identity()
    current = _runkey(recipe, identity)
    if runkey is None:
        key = current
    elif not isinstance(runkey, str) or not _RUNKEY_RE.fullmatch(runkey):
        raise ValueError(f'runkey must be a run key ("<recipe>-" and 12 hex digits), got {runkey!r}')
    else:
        key = runkey
    rdir = os.path.join(out_dir, key)
    record = _read_record(rdir)
    if runkey is not None:
        if record is None:
            raise RunnerError(f'{out_dir} holds no run {runkey} of this runner (no readable {_RECIPE} with an identity)')
        if record['recipe_digest'] != recipe.digest:
            raise RunnerError(f'{runkey} ran recipe {record["recipe"].get("name")!r} ({record["recipe_digest"][:12]}), '
                              f'not {recipe.name!r} ({recipe.digest[:12]})')
    if record is None:                                   # the current code has not started here
        record = dict(recipe=recipe.spec(), recipe_digest=recipe.digest, identity=identity)
    out = dict(runkey=key, current=key == current, n_chunks=len(run['chunks']), done=[], failed=[], claimed=[],
               todo=[], claims={}, runs=_runs(out_dir, run, paths, identity, current))
    for k, (start, stop) in enumerate(run['chunks']):
        if _chunk_done(rdir, k, _sha_lines(paths[start:stop]), record):
            out['done'].append(k)
            continue
        claim = _read_claim(os.path.join(rdir, 'claims', f'{_chunk_name(k)}.json'))
        if os.path.isfile(os.path.join(rdir, 'chunks', f'{_chunk_name(k)}.failed.json')):
            out['failed'].append(k)
        elif claim is not None:
            out['claimed'].append(k)
        else:
            out['todo'].append(k)
        if claim is not None:
            out['claims'][k] = claim
    return out


def reclaim(out_dir, *, chunks: Optional[Sequence[int]] = None) -> dict:
    """Free the claims of chunks that are not written and whose process has died.

    Every run key under ``out_dir`` is scanned. A chunk is written exactly as for
    ``run_status``, ``collate`` and the workers (``_chunk_done`` against its run key's
    ``recipe.json``); the claim of a written chunk is kept (a record of who wrote it). Any
    other claim made on this host is freed when its process is gone (no such pid, or the pid
    now belongs to a process started at another time); a claim from another host cannot be
    judged and is kept unless its chunk is named in ``chunks``. A freed claim is moved to
    ``claims/freed/``; whatever the chunk left that is not a written chunk (its store, its
    manifest, the dead process's partial store ``chunk_NNNNNN.partial-<pid>``) to
    ``chunks/orphans/``, so the chunk is free to be redone. A run key without an identity
    record (another runner version) is reported, not touched.

    Parameters
    ----------
    chunks : sequence of int, optional
        Chunks to free whatever their process's state (a claim on another host, or a
        process known to be stuck). Freeing a live process's claim lets a second process
        take the chunk; whichever saves its store second fails that chunk.

    Returns
    -------
    dict
        Per runkey: ``freed``, ``alive``, ``other_host``, ``unreadable`` chunk lists (and
        ``unjudged``, the reason, for a run key without an identity record).

    Raises
    ------
    RunnerError
        ``out_dir`` holds no run.
    ValueError
        ``chunks`` that are not ints.
    """
    import psutil              # judges whether a pid is alive; os.kill(pid, 0) would end a process on Windows
    out_dir = os.path.abspath(os.fspath(out_dir))
    run, paths = _load_run(out_dir)
    named = set()
    if chunks is not None:
        if any(isinstance(c, bool) or not isinstance(c, int) for c in chunks):
            raise ValueError(f'chunks must be ints, got {list(chunks)[:10]}')
        named = set(chunks)
    host = socket.gethostname()
    report = {}
    for key in sorted(os.listdir(out_dir)):
        rdir = os.path.join(out_dir, key)
        claims = os.path.join(rdir, 'claims')
        if not (_RUNKEY_RE.fullmatch(key) and os.path.isdir(claims)):
            continue
        rep = dict(freed=[], alive=[], other_host=[], unreadable=[])
        record = _read_record(rdir)
        if record is None:
            rep['unjudged'] = f'no readable {_RECIPE} with an identity (not a run of this runner version)'
            report[key] = rep
            continue
        for name in sorted(os.listdir(claims)):
            m = re.fullmatch(r'chunk_(\d{6})\.json', name)
            if not m or int(m.group(1)) >= len(run['chunks']):
                continue
            k = int(m.group(1))
            start, stop = run['chunks'][k]
            if _chunk_done(rdir, k, _sha_lines(paths[start:stop]), record):
                continue                                    # written: the claim is its record
            claim = _read_claim(os.path.join(claims, name))
            if claim is None and k not in named:
                rep['unreadable'].append(k)
                continue
            if k not in named:
                if claim.get('host') != host:
                    rep['other_host'].append(k)
                    continue
                if _alive(psutil, claim):
                    rep['alive'].append(k)
                    continue
            _move_aside(os.path.join(claims, name), os.path.join(claims, 'freed'))
            pid = None if claim is None else claim.get('pid')
            leftovers = [os.path.join(rdir, 'chunks', f'{_chunk_name(k)}.json'), os.path.join(rdir, 'chunks', _chunk_name(k))]
            if pid is not None:
                leftovers.append(os.path.join(rdir, 'chunks', f'{_chunk_name(k)}.partial-{pid}'))
            for leftover in leftovers:
                if os.path.exists(leftover):
                    _move_aside(leftover, os.path.join(rdir, 'chunks', 'orphans'))
            rep['freed'].append(k)
        report[key] = rep
    return report


def collate(out_dir, *, recipe, allow_partial: bool = False, runkey: Optional[str] = None,
            allow_other_code: bool = False) -> str:
    """Join the chunk stores of ``recipe`` under one run key into one store, in path order.

    Parameters
    ----------
    recipe : str or Recipe
    allow_partial : bool
        Collate the written chunks when some are not; the store's meta records
        ``partial``, ``allow_partial`` and the missing chunks.
    runkey : str, optional
        The run key to collate (``run_status(...)['runs']``); default the installed code's.
    allow_other_code : bool
        Collate a run key written by other code than is installed. The collated store's
        ``code`` then names the installed ingestion modules, so its meta also records the
        rows' identity, the installed one and how they differ.

    Returns
    -------
    str
        The collated store's folder: ``<runkey>/collated`` (every chunk), or
        ``<runkey>/collated_partial_<k>of<n>``.

    Raises
    ------
    RunnerError
        A chunk is not written (without ``allow_partial``; the message names the claimed,
        failed and never-claimed chunks), no chunk is written, the chunks were run with
        different step arguments, or the run key was written by other code (without
        ``allow_other_code``); and as ``run_status``.
    FileExistsError
        The collated store exists (it is never overwritten).
    ValueError
        ``allow_partial`` or ``allow_other_code`` not a bool; chunk batches that cannot be
        joined (``concat_batches``).
    """
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    for name, value in (('allow_partial', allow_partial), ('allow_other_code', allow_other_code)):
        if not isinstance(value, bool):
            raise ValueError(f'{name} must be a bool, got {value!r}')
    recipe = get_recipe(recipe)
    state = run_status(out_dir, recipe=recipe, runkey=runkey)
    out_dir = os.path.abspath(os.fspath(out_dir))
    run, paths = _load_run(out_dir)
    n = state['n_chunks']
    rdir = os.path.join(out_dir, state['runkey'])
    record = _read_record(rdir)
    installed = run_identity()
    changes = [] if record is None else _identity_changes(record['identity'], installed)
    if changes and not allow_other_code:
        raise RunnerError(f'collate refuses: {state["runkey"]} was written by other code than is installed '
                          f'({_summary(changes)}); restore that code, or pass allow_other_code=True (recorded)')
    missing = sorted(state['claimed'] + state['failed'] + state['todo'])
    if missing and not allow_partial:
        others = {k: v['done'] for k, v in state['runs'].items() if k != state['runkey'] and v['done']}
        raise RunnerError(f'collate refuses: {len(missing)} of {n} chunks are not written under {state["runkey"]}: '
                          f'claimed but unwritten {state["claimed"][:20]} (free dead processes\' claims with reclaim), '
                          f'failed {state["failed"][:20]}, never claimed {state["todo"][:20]}; pass allow_partial=True '
                          'to collate the written ones'
                          + (f'; other run keys here hold written chunks {others} (run_status runs)' if others else ''))
    if not state['done']:
        raise RunnerError(f'no chunk of {n} is written for {state["runkey"]}')
    target = os.path.join(rdir, 'collated' if not missing else f'collated_partial_{len(state["done"])}of{n}')
    if os.path.exists(target):
        raise FileExistsError(f'{target} exists; a store is never overwritten')
    stores = [os.path.join(rdir, 'chunks', _chunk_name(k)) for k in state['done']]
    steps = [read_meta(s)['steps_run'] for s in stores]
    if any(s != steps[0] for s in steps):
        raise RunnerError('the chunk stores record different steps_run; they were not run with one recipe')
    batch = concat_batches([MolCrystalData.load(s) for s in stores])
    chunk_sha = {str(k): read_meta(s)['files']['data.pt']['sha256'] for k, s in zip(state['done'], stores)}
    batch.save(target, steps_run=steps[0], extra_meta=dict(featurize_cifs=dict(
        runkey=state['runkey'], recipe=recipe.spec(), recipe_digest=recipe.digest,
        paths_sha256=run['paths_sha256'], n_paths=run['n_paths'], chunk_size=run['chunk_size'], n_chunks=n,
        partial=bool(missing), allow_partial=allow_partial, missing_chunks=missing, chunk_data_sha256=chunk_sha,
        rows_identity=record['identity']['digest'], rows_code=record['identity']['store_code'],
        installed_identity=installed['digest'], allow_other_code=allow_other_code, code_changes=changes[:200],
        n_code_changes=len(changes))))
    return target


###############################################################################
# joining batches
###############################################################################

def concat_batches(batches: Sequence):
    """One batch holding the rows of ``batches`` in order, as one read of all their sources would hold them.

    For batches of the ingestion chain (``read`` and the steps) made by the same steps: the
    same attributes, in the same order. Per attribute, by its level in ``_slice_dict``: per
    atom and per row values are concatenated (tensors along dim 0, lists joined), a row
    tensor's trailing shape grown to the widest with its ``ROW_PADS`` value; batch-level
    values (no slice entry) must be equal, except ``max_z_prime`` (the largest). ``ptr``,
    ``batch``, ``_num_graphs`` and the slice bookkeeping (per atom: the new ``ptr``; per row:
    ``arange(B + 1)``; increments long zeros for tensors, None for lists) are rebuilt. Inputs
    are not changed.

    Parameters
    ----------
    batches : sequence of MolCrystalDataBatch
        At least one; one class.

    Returns
    -------
    MolCrystalDataBatch

    Raises
    ------
    ValueError
        No batches; different classes or attribute lists; an attribute at different levels,
        with non-zero increments, of different types or dtypes, a trailing shape that
        differs outside ``ROW_PADS``, or a batch-level value that differs.
    """
    batches = list(batches)
    if not batches:
        raise ValueError('concat_batches needs at least one batch')
    first = batches[0]
    cls = type(first)
    keys = list(first._store.keys())
    for i, b in enumerate(batches):
        if type(b) is not cls:
            raise ValueError(f'batch {i} is a {type(b).__name__}, batch 0 a {cls.__name__}')
        if '_slice_dict' not in b.__dict__ or '_inc_dict' not in b.__dict__:
            raise ValueError(f'batch {i} has no _slice_dict / _inc_dict bookkeeping')
        if list(b._store.keys()) != keys:
            raise ValueError(f'batch {i} holds other attributes than batch 0: '
                             f'{sorted(set(b._store.keys()) ^ set(keys))[:10] or "a different order"}')
    ptrs = [b['ptr'].detach().to('cpu').long() for b in batches]
    counts = torch.cat([p[1:] - p[:-1] for p in ptrs])
    B = len(counts)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    row_slices = torch.arange(B + 1)
    out = cls.__new__(cls)
    out.__dict__['_store'] = GlobalStorage(_parent=out)
    slice_dict, inc_dict = {}, {}
    for key in keys:
        if key == 'ptr':
            out['ptr'] = ptr.to(first['ptr'].dtype).to(first['ptr'].device)
            continue
        if key == 'batch':
            out['batch'] = torch.arange(B).repeat_interleave(counts).to(first['batch'].device)
            continue
        values = [b[key] for b in batches]
        levels = [_level(b, key, p) for b, p in zip(batches, ptrs)]
        if all(level is None for level in levels):
            out[key] = _batch_level_value(key, values)
            continue
        if any(level is None for level in levels):
            raise ValueError(f'{key!r} has a slice entry in some batches only')
        known = {level for level in levels if level != 'either'}
        if len(known) > 1:
            raise ValueError(f'{key!r} is stored per atom in some batches and per row in others')
        level = known.pop() if known else 'atom'
        out[key] = _join(key, values, level)
        slice_dict[key] = ptr.clone() if level == 'atom' else row_slices.clone()
        incs = [b._inc_dict.get(key) for b in batches]
        if torch.is_tensor(out[key]):
            if not all(torch.is_tensor(i) and not bool(i.any()) for i in incs):
                raise ValueError(f'{key!r} carries collation increments; concat_batches does not re-offset them')
            inc_dict[key] = torch.zeros(B, dtype=torch.long)
        else:
            if any(i is not None for i in incs):
                raise ValueError(f'{key!r} is a list with increments')
            inc_dict[key] = None
    out.__dict__['_num_graphs'] = B
    out.__dict__['_slice_dict'] = slice_dict
    out.__dict__['_inc_dict'] = inc_dict
    return out


def _level(batch, key, ptr):
    """'atom', 'row', 'either' (one atom per row: the two coincide) or None (no slice entry)."""
    s = batch._slice_dict.get(key)
    if s is None:
        return None
    s = s.detach().to('cpu').long()
    rows = torch.arange(len(ptr))
    is_atom, is_row = torch.equal(s, ptr), torch.equal(s, rows)
    if is_atom and is_row:
        return 'either'
    if is_atom or is_row:
        return 'atom' if is_atom else 'row'
    raise ValueError(f'{key!r} is sliced neither per atom nor per row')


def _batch_level_value(key, values):
    if key == 'max_z_prime':
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise ValueError('max_z_prime must be a python int in every batch')
        return max(values)
    for v in values[1:]:
        same = torch.equal(v, values[0]) if torch.is_tensor(v) and torch.is_tensor(values[0]) else v == values[0]
        if not same:
            raise ValueError(f'the batch-level value {key!r} differs between batches')
    return values[0]


def _join(key, values, level):
    if all(torch.is_tensor(v) for v in values):
        dtypes = {v.dtype for v in values}
        if len(dtypes) > 1 or len({v.dim() for v in values}) > 1:
            raise ValueError(f'{key!r} has different dtypes or ranks between batches')
        trailing = tuple(max(v.shape[d] for v in values) for d in range(1, values[0].dim()))
        parts = []
        for v in values:
            if tuple(v.shape[1:]) != trailing:
                if level != 'row' or key not in ROW_PADS:
                    raise ValueError(f'{key!r} has trailing shape {tuple(v.shape[1:])} in one batch and '
                                     f'{trailing} in another; only the row attributes of ROW_PADS are padded')
                padded = torch.full((v.shape[0],) + trailing, ROW_PADS[key], dtype=v.dtype, device=v.device)
                padded[(slice(None),) + tuple(slice(0, n) for n in v.shape[1:])] = v
                v = padded
            parts.append(v)
        return torch.cat(parts, 0)
    if all(isinstance(v, (list, tuple)) for v in values):
        if len({type(v) for v in values}) > 1:
            raise ValueError(f'{key!r} is a list in some batches and a tuple in others')
        joined = [x for v in values for x in v]
        return type(values[0])(joined)
    raise ValueError(f'{key!r} has a slice entry but is neither a tensor nor a list in every batch')


###############################################################################
# files
###############################################################################

def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _json_bytes(data) -> bytes:
    return (json.dumps(data, indent=1, allow_nan=False) + '\n').encode('utf-8')


def _sha_lines(lines) -> str:
    return hashlib.sha256(('\n'.join(lines) + '\n').encode('utf-8')).hexdigest()


def _chunk_name(k: int) -> str:
    return f'chunk_{k:06d}'


def _sha_file(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _runkey(recipe, identity) -> str:
    return f'{recipe.name}-' + hashlib.sha256(f'{identity["digest"]}|{recipe.digest}'.encode()).hexdigest()[:12]


def _sorted_paths(paths) -> list:
    if isinstance(paths, (str, bytes, os.PathLike)) or not hasattr(paths, '__iter__'):
        raise ValueError('paths must be a sequence of paths')
    out = []
    for p in paths:
        if not isinstance(p, (str, os.PathLike)):
            raise ValueError(f'a path must be a str or os.PathLike, got {type(p).__name__}')
        p = os.fspath(p)
        if not p or any(unicodedata.category(c) in ('Cc', 'Cf') for c in p):
            raise ValueError(f'a path must be non-empty and hold no control or format character (newline, tab, '
                             f'byte-order mark, zero-width space), got {p!r:.80}')
        if not os.path.isabs(p):
            raise ValueError(f'paths must be absolute (workers may start elsewhere), got {p!r:.80}')
        out.append(p)
    if not out:
        raise ValueError('no paths')
    out.sort()
    dup = [a for a, b in zip(out[:-1], out[1:]) if a == b]
    if dup:
        raise ValueError(f'{len(dup)} duplicate path(s), e.g. {dup[0]!r}')
    return out


def _out_dir(out_dir, paths) -> str:
    """The absolute ``out_dir``, refused when it contains an input file or is a folder of something else."""
    if not isinstance(out_dir, (str, os.PathLike)):
        raise ValueError(f'out_dir must be a str or os.PathLike, got {type(out_dir).__name__}')
    out_dir = os.path.abspath(os.fspath(out_dir))
    norm = os.path.normcase(out_dir)
    for parent in sorted({os.path.normcase(os.path.dirname(os.path.abspath(p))) for p in paths}):
        if parent == norm or parent.startswith(norm.rstrip(os.sep) + os.sep):
            raise ValueError(f'out_dir {out_dir} contains input files ({parent}); outputs never go next to the inputs')
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            raise ValueError(f'out_dir {out_dir} exists and is not a folder')
        foreign = [n for n in os.listdir(out_dir) if not (n in (_PATHS, _RUN) or _RUNKEY_RE.fullmatch(n)
                                                         or re.fullmatch(r'\.(paths\.txt|run\.json)\.tmp-.+', n))]
        if foreign:
            raise ValueError(f'out_dir {out_dir} holds files of something else ({foreign[:5]}); use a new folder')
    return out_dir


def _prepare_run(out_dir, paths, chunk_size) -> dict:
    """Write paths.txt and run.json, or check that the ones there describe these paths and chunks."""
    os.makedirs(out_dir, exist_ok=True)
    text = ('\n'.join(paths) + '\n').encode('utf-8')
    if not _create_exclusive(os.path.join(out_dir, _PATHS), text):
        with open(os.path.join(out_dir, _PATHS), 'rb') as f:
            there = f.read()
        if there != text:
            n_there = there.count(b'\n')
            raise ValueError(f'{out_dir} holds another path list ({n_there} paths, sha256 '
                             f'{hashlib.sha256(there).hexdigest()[:12]}; this call: {len(paths)}, '
                             f'{hashlib.sha256(text).hexdigest()[:12]}); use a new out_dir')
    n_chunks = math.ceil(len(paths) / chunk_size)
    run = dict(schema=SCHEMA, n_paths=len(paths), paths_sha256=hashlib.sha256(text).hexdigest(), chunk_size=chunk_size,
               chunks=[[i * chunk_size, min((i + 1) * chunk_size, len(paths))] for i in range(n_chunks)],
               created_utc=_utc())
    if not _create_exclusive(os.path.join(out_dir, _RUN), _json_bytes(run)):
        there, _ = _load_run(out_dir)
        differ = [k for k in _RUN_KEYS if there.get(k) != run[k]]
        if differ:
            raise ValueError(f'{out_dir} holds another chunk map (differs in {differ}: e.g. chunk_size '
                             f'{there.get("chunk_size")} there, {chunk_size} here); use a new out_dir')
        return there
    return run


def _load_run(out_dir):
    """(run.json, the path list) of ``out_dir``; RunnerError when it holds no run."""
    try:
        with open(os.path.join(out_dir, _RUN), encoding='utf-8') as f:
            run = json.load(f)
        with open(os.path.join(out_dir, _PATHS), encoding='utf-8') as f:
            paths = f.read().split('\n')[:-1]
    except (OSError, json.JSONDecodeError) as e:
        raise RunnerError(f'{out_dir} holds no featurize_cifs run ({type(e).__name__}: {e})') from e
    if run.get('schema') != SCHEMA or len(paths) != run.get('n_paths'):
        raise RunnerError(f'{out_dir}: run.json is not a {SCHEMA} run of {len(paths)} paths')
    return run, paths


def _record_recipe(rdir, recipe, identity) -> dict:
    """Write ``<runkey>/recipe.json``, or check the one there; the record."""
    record = dict(schema=SCHEMA, recipe=recipe.spec(), recipe_digest=recipe.digest, identity=identity)
    if not _create_exclusive(os.path.join(rdir, _RECIPE), _json_bytes(record)):
        there = _read_record(rdir)
        if there is None or there['recipe_digest'] != recipe.digest or \
                there['identity']['digest'] != identity['digest']:
            raise RunnerError(f'{rdir}/{_RECIPE} records another recipe or code under the same run key')
    return record


def _read_record(rdir) -> Optional[dict]:
    """``<runkey>/recipe.json`` when it is a record of this runner (recipe digest and identity), else None."""
    try:
        with open(os.path.join(rdir, _RECIPE), encoding='utf-8') as f:
            record = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    ok = (isinstance(record, dict) and record.get('schema') == SCHEMA and isinstance(record.get('recipe'), dict)
          and isinstance(record.get('recipe_digest'), str) and isinstance(record.get('identity'), dict)
          and all(isinstance(record['identity'].get(k), str) for k in ('digest', 'store_code')))
    return record if ok else None


def _chunk_done(rdir, k, chunk_sha, record) -> bool:
    """Written: the chunk's store and manifest exist and match its run key's record and the chunk's paths.

    The one definition the workers, ``run_status``, ``collate`` and ``reclaim`` use. ``record``
    is the run key's ``recipe.json``: the manifest must name its recipe and identity, the
    store's meta its ingestion-module digest (``crystal_store`` records it at save time).
    """
    chunks = os.path.join(rdir, 'chunks')
    man_path, store = os.path.join(chunks, f'{_chunk_name(k)}.json'), os.path.join(chunks, _chunk_name(k))
    if not (os.path.isfile(man_path) and os.path.isdir(store)):
        return False
    try:
        with open(man_path, encoding='utf-8') as f:
            man = json.load(f)
        meta = read_meta(store)
    except (OSError, json.JSONDecodeError, StoreError):
        return False
    identity = record['identity']
    return (isinstance(man, dict) and man.get('identity_digest') == identity['digest']
            and man.get('code_digest') == identity['store_code'] and man.get('recipe_digest') == record['recipe_digest']
            and man.get('paths_sha256') == chunk_sha and meta.get('code', {}).get('digest') == identity['store_code']
            and meta.get('files', {}).get('data.pt', {}).get('sha256') == man.get('store_data_sha256')
            and meta.get('rows') == man.get('rows'))


def _runs(out_dir, run, paths, identity, current) -> dict:
    """``run_status``'s ``runs``: every run key under ``out_dir`` with its written chunks."""
    shas = [_sha_lines(paths[start:stop]) for start, stop in run['chunks']]
    out = {}
    for key in sorted(os.listdir(out_dir)):
        rdir = os.path.join(out_dir, key)
        if not (_RUNKEY_RE.fullmatch(key) and os.path.isdir(rdir)):
            continue
        record = _read_record(rdir)
        collated = sorted(n for n in os.listdir(rdir) if n == 'collated' or n.startswith('collated_partial_'))
        if record is None:
            out[key] = dict(recipe=None, recipe_digest=None, identity=None, current=False, done=None,
                            code_changes=None, collated=collated,
                            note=f'no readable {_RECIPE} with an identity (not a run of this runner version)')
            continue
        changes = _identity_changes(record['identity'], identity)
        out[key] = dict(recipe=record['recipe'].get('name'), recipe_digest=record['recipe_digest'][:12],
                        identity=record['identity']['digest'][:12], current=key == current,
                        done=sum(_chunk_done(rdir, k, sha, record) for k, sha in enumerate(shas)),
                        code_changes=_summary(changes) if changes else '', collated=collated)
    return out


def _claim(rdir, k, worker) -> bool:
    """Create the chunk's claim atomically; False when it exists (another process holds or held it)."""
    try:
        import psutil
        create_time = psutil.Process(os.getpid()).create_time()
    except Exception:          # psutil absent: reclaim then cannot tell a reused pid
        create_time = None
    claim = dict(worker=worker, pid=os.getpid(), host=socket.gethostname(), create_time=create_time,
                 claimed_utc=_utc())
    return _create_exclusive(os.path.join(rdir, 'claims', f'{_chunk_name(k)}.json'), _json_bytes(claim))


def _read_claim(path) -> Optional[dict]:
    try:
        with open(path, encoding='utf-8') as f:
            claim = json.load(f)
        return claim if isinstance(claim, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _alive(psutil, claim) -> bool:
    pid = claim.get('pid')
    if not isinstance(pid, int) or not psutil.pid_exists(pid):
        return False
    try:
        started = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return False
    recorded = claim.get('create_time')
    return recorded is None or abs(started - recorded) < 1.0       # another start time: the pid was reused


def _record_failure(rdir, k, worker, error):
    record = dict(chunk=k, worker=worker, pid=os.getpid(), host=socket.gethostname(), utc=_utc(),
                  error=f'{type(error).__name__}: {error}',
                  traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)))
    _write_atomic(os.path.join(rdir, 'chunks', f'{_chunk_name(k)}.failed.json'), _json_bytes(record))


def _create_exclusive(path, data: bytes) -> bool:
    """Write ``path`` whole, only if it does not exist (a hard link from a temporary file); False if it exists."""
    folder, name = os.path.split(path)
    tmp = os.path.join(folder, f'.{name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}')
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        os.remove(tmp)


def _write_atomic(path, data: bytes):
    """Write ``path`` through a temporary file and a rename (replacing this runner's own earlier record)."""
    folder, name = os.path.split(path)
    tmp = os.path.join(folder, f'.{name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}')
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _move_aside(path, folder):
    """Move ``path`` into ``folder`` under a new name (UTC time and pid appended); nothing is deleted."""
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    os.rename(path, os.path.join(folder, f'{os.path.basename(path)}.{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}'))


###############################################################################
# command line
###############################################################################

def main(argv=None) -> int:
    """The command line (module docstring); returns the exit code (1 when a chunk failed, the worker stopped
    because the code changed, or collate refused)."""
    ap = argparse.ArgumentParser(prog='python -m mxtaltools.dataset_utils.construction.featurize_cifs',
                                 description='Featurize CIF files with a named recipe, chunk by chunk. Run from a '
                                             'tree nobody edits meanwhile: a worker stops when the code changes.')
    ap.add_argument('--out', required=True, help='the run folder')
    ap.add_argument('--recipe', help="a recipe name, e.g. 'csd_v1' (all actions but --reclaim)")
    ap.add_argument('--paths-file', help='one CIF path per line (absolute, or relative to --cif-dir); UTF-8, a '
                                         'leading byte-order mark allowed')
    ap.add_argument('--cif-dir', help='folder that relative lines of --paths-file name files in')
    ap.add_argument('--chunk-size', type=int)
    ap.add_argument('--worker-id')
    ap.add_argument('--threads', type=int, default=1, help='torch threads of this process (default 1)')
    act = ap.add_mutually_exclusive_group()
    act.add_argument('--status', action='store_true', help='print the state of every chunk')
    act.add_argument('--reclaim', action='store_true', help="free the claims of dead processes")
    act.add_argument('--collate', action='store_true', help='join the chunk stores into one')
    ap.add_argument('--chunks', type=int, nargs='*', help='with --reclaim: chunks to free whatever their state')
    ap.add_argument('--allow-partial', action='store_true', help='with --collate: collate the written chunks only')
    ap.add_argument('--runkey', help='with --status or --collate: a run key under --out (default: the installed '
                                     "code's); --status lists them all under 'runs'")
    ap.add_argument('--allow-other-code', action='store_true',
                    help='with --collate --runkey: collate a run key written by other code than is installed (recorded)')
    a = ap.parse_args(argv)
    if a.threads < 1:
        ap.error('--threads must be at least 1')
    if a.runkey is not None and not (a.status or a.collate):
        ap.error('--runkey goes with --status or --collate')
    if a.allow_other_code and not a.collate:
        ap.error('--allow-other-code goes with --collate')
    torch.set_num_threads(a.threads)
    if a.reclaim:
        print(json.dumps(reclaim(a.out, chunks=a.chunks), indent=1))
        return 0
    if a.recipe is None:
        ap.error('--recipe is required')
    if a.status:
        state = run_status(a.out, recipe=a.recipe, runkey=a.runkey)
        print(json.dumps({k: (v if k != 'claims' else {str(c): r for c, r in v.items()}) for k, v in state.items()},
                         indent=1))
        return 0
    if a.collate:
        try:
            print(collate(a.out, recipe=a.recipe, allow_partial=a.allow_partial, runkey=a.runkey,
                          allow_other_code=a.allow_other_code))
        except RunnerError as e:
            print(f'collate refused: {e}', file=sys.stderr)
            return 1
        return 0
    if a.paths_file is None or a.chunk_size is None:
        ap.error('featurizing needs --paths-file and --chunk-size')
    with open(a.paths_file, encoding='utf-8-sig') as f:         # utf-8-sig: a leading byte-order mark is not a name
        lines = [line.strip() for line in f if line.strip()]
    if a.cif_dir is not None:
        lines = [line if os.path.isabs(line) else os.path.join(a.cif_dir, line) for line in lines]
    summary = featurize_cifs(lines, a.out, recipe=a.recipe, chunk_size=a.chunk_size, worker_id=a.worker_id)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in summary.items()}))
    return 1 if summary['failed'] or summary['stopped'] else 0


if __name__ == '__main__':
    sys.exit(main())

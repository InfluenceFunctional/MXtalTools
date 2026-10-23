"""Contract tests for the CSD-scale chunk runner (dataset_utils/construction/featurize_cifs.py).

CPU only, CCDC-free: a recipe with ``sanitize(chemistry='none')`` on the 16 CIF fixtures.
Checked: the collated store equals one read of every file in one batch (padding of Z'max-wide
attributes across chunks included), resume skips written chunks, claims are never waited on,
collate refuses missing chunks unless ``allow_partial``, reclaim frees dead processes' claims
only, failures are recorded and keep their claim, a code or recipe change starts a new set,
the path list and chunk map are fixed per folder, outputs never go next to the inputs, and the
command line. The code identity covers the whole package and the libraries' versions; an edit while a
worker runs (simulated by patching the identity or the store's digest) stops it and leaves no chunk
stuck; reclaim, status and collate share one definition of written; status lists every run key and
collate takes one by name; a paths file with a byte-order mark loses no file. Real concurrency (several
processes) is exercised by the gate, not here.
"""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from mxtaltools.dataset_utils.construction import featurize_cifs as fc
from mxtaltools.dataset_utils.construction.featurize_cifs import (ROW_PADS, RunnerError, collate, concat_batches,
                                                                  featurize_cifs, reclaim, run_status)
from mxtaltools.dataset_utils.data_class_methods import crystal_recipes
from mxtaltools.dataset_utils.data_class_methods.crystal_recipes import Recipe
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import _set
from mxtaltools.dataset_utils.data_class_methods.crystal_store import code_digest, read_meta, status_table
from mxtaltools.dataset_utils.data_classes import MolCrystalData

FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
PATHS = sorted(str(p) for p in FIXTURES.glob('*.cif'))
F64 = torch.float64
CHEAP = Recipe('cheap_test', (
    ('read', dict(format='cif', dtype=F64, max_atomic_number=100)),
    ('find_molecules', dict(bonds='file', bond_scale=(1.15, 1.35), iso_cap=5000, special_tol=0.1)),
    ('sanitize', dict(chemistry='none', hydrogens='as_read', on_protonation_change='flag')),
    ('parameterize_pose', dict(handedness='measured', degenerate_tol=1e-4, handedness_tol=1e-3)),
    ('check_rebuild', dict(max_dev=0.05, mean_dev=0.01, match_elements=True)),
    ('check_validity', dict(max_atomic_number=100, max_heavy_atoms=100, heavy_nn_range=(0.9, 3.0), occupancy=1e-3,
                            disorder=None, cell_constraints=1e-4, min_contact=0.5, composition='heavy')),
))
CHUNK = 4                       # 16 fixtures -> 4 chunks; only one holds the Z' = 2 row, so widths differ


def _quiet(line):
    pass


def _run(out, **kw):
    return featurize_cifs(kw.pop('paths', PATHS), out, recipe=kw.pop('recipe', CHEAP),
                          chunk_size=kw.pop('chunk_size', CHUNK), log=_quiet, **kw)


def _same(a, b):
    import numpy as np
    if torch.is_tensor(a):
        if not (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape):
            return False
        if a.is_floating_point():
            return bool(torch.equal(torch.isnan(a), torch.isnan(b))) and bool(torch.equal(a[~torch.isnan(a)],
                                                                                         b[~torch.isnan(b)]))
        return bool(torch.equal(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def assert_same_batch(a, b):
    assert list(a._store.keys()) == list(b._store.keys())
    for k in a._store.keys():
        assert _same(a._store[k], b._store[k]), k
    assert a.__dict__.get('_num_graphs') == b.__dict__.get('_num_graphs')
    for name in ('_slice_dict', '_inc_dict'):
        assert a.__dict__[name].keys() == b.__dict__[name].keys(), name
        for k in a.__dict__[name]:
            assert _same(a.__dict__[name][k], b.__dict__[name][k]), (name, k)


@pytest.fixture(scope='module')
def single():
    """One read of every fixture, in sorted order, in one batch."""
    return MolCrystalData.read(PATHS, dtype=F64, full=CHEAP)


def _dead_pid():
    done = subprocess.run([sys.executable, '-c', 'import os; print(os.getpid())'], capture_output=True, text=True,
                          check=True)
    return int(done.stdout.strip())


def _write_claim(out, k, **claim):
    rdir = Path(out) / fc._runkey(CHEAP, fc.run_identity()) / 'claims'
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f'chunk_{k:06d}.json').write_text(json.dumps(claim), encoding='utf-8')


# ======================================================================================
# the run and the collated store
# ======================================================================================

def test_collated_store_equals_one_batch_of_every_file(tmp_path, single):
    out = tmp_path / 'run'
    summary = _run(out)
    assert summary['written'] == [0, 1, 2, 3] and not summary['failed'] and summary['n_chunks'] == 4
    rdir = out / summary['runkey']
    widths = {MolCrystalData.load(rdir / 'chunks' / f'chunk_{k:06d}').unit_radius.shape[1] for k in range(4)}
    assert widths == {1, 2}                                   # the padding path is exercised
    target = collate(out, recipe=CHEAP)
    assert Path(target) == rdir / 'collated'
    collated = MolCrystalData.load(target)
    assert_same_batch(collated, single)
    meta = read_meta(target)
    run = meta['extra']['featurize_cifs']
    assert run['partial'] is False and run['missing_chunks'] == [] and run['n_chunks'] == 4
    assert run['recipe_digest'] == CHEAP.digest and len(run['chunk_data_sha256']) == 4
    assert meta['steps_run'][0] == dict(step='read', args=dict(format='cif', dtype='float64', max_atomic_number=100))
    assert [s['step'] for s in meta['steps_run'][1:]] == list(CHEAP.step_names[1:]) + ['run_chain']
    assert (out / 'paths.txt').read_text(encoding='utf-8').split('\n')[:-1] == PATHS
    with pytest.raises(FileExistsError):
        collate(out, recipe=CHEAP)


def test_chunk_manifests_and_logs_name_rows_statuses_and_seconds(tmp_path):
    out = tmp_path / 'run'
    summary = _run(out, worker_id='w1')
    rdir = out / summary['runkey']
    for k in range(4):
        man = json.loads((rdir / 'chunks' / f'chunk_{k:06d}.json').read_text(encoding='utf-8'))
        assert man['n_paths'] == 4 and man['rows'] == 4 and man['worker'] == 'w1'
        assert sum(man['status_counts'].values()) == 4 and man['seconds']['read_and_chain'] >= 0
        assert man['code_digest'] == code_digest()['digest'] and man['recipe_digest'] == CHEAP.digest
        assert man['identity_digest'] == fc.run_identity()['digest']
    lines = (rdir / 'logs' / 'w1.log').read_text(encoding='utf-8').splitlines()
    assert sum(' rows (' in line and ' s read and chain' in line for line in lines) == 4
    recipe = json.loads((rdir / 'recipe.json').read_text(encoding='utf-8'))
    assert recipe['recipe_digest'] == CHEAP.digest and recipe['identity']['digest'] == fc.run_identity()['digest']
    assert recipe['identity']['store_code'] == code_digest()['digest']


def test_a_second_run_skips_written_chunks(tmp_path):
    out = tmp_path / 'run'
    _run(out)
    rdir = out / fc._runkey(CHEAP, fc.run_identity())
    before = {p: p.stat().st_mtime_ns for p in (rdir / 'chunks').rglob('*')}
    again = _run(out)
    assert again['skipped_done'] == [0, 1, 2, 3] and not again['written']
    assert {p: p.stat().st_mtime_ns for p in (rdir / 'chunks').rglob('*')} == before


def test_the_path_list_and_chunk_map_are_fixed_per_folder(tmp_path):
    out = tmp_path / 'run'
    _run(out, paths=PATHS[:8])
    with pytest.raises(ValueError, match='another path list'):
        _run(out, paths=PATHS[:9])
    with pytest.raises(ValueError, match='another chunk map'):
        _run(out, paths=PATHS[:8], chunk_size=3)
    assert _run(out, paths=list(reversed(PATHS[:8])))['skipped_done'] == [0, 1]       # sorted: the same list


def test_bad_arguments_raise(tmp_path):
    with pytest.raises(ValueError, match='absolute'):
        _run(tmp_path / 'a', paths=['x.cif'])
    with pytest.raises(ValueError, match='duplicate'):
        _run(tmp_path / 'a', paths=PATHS[:2] + PATHS[:1])
    with pytest.raises(ValueError, match='no paths'):
        _run(tmp_path / 'a', paths=[])
    with pytest.raises(ValueError, match='chunk_size'):
        _run(tmp_path / 'a', chunk_size=0)
    with pytest.raises(ValueError, match='worker_id'):
        _run(tmp_path / 'a', worker_id='a b')
    with pytest.raises(ValueError, match='contains input files'):
        _run(FIXTURES.parent)                                   # the inputs' own folder's parent
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'notes.txt').write_text('mine', encoding='utf-8')
    with pytest.raises(ValueError, match='something else'):
        _run(foreign)
    assert (foreign / 'notes.txt').read_text(encoding='utf-8') == 'mine' and len(list(foreign.iterdir())) == 1
    with pytest.raises(NotImplementedError):
        _run(tmp_path / 'a', recipe=True)


# ======================================================================================
# claims, reclaim, partial collation, failures
# ======================================================================================

def test_claims_are_skipped_collate_refuses_and_reclaim_frees_only_dead_processes(tmp_path, single):
    import psutil
    out = tmp_path / 'run'
    host = socket.gethostname()
    me = psutil.Process(os.getpid())
    _write_claim(out, 1, worker='dead', pid=_dead_pid(), host=host, create_time=1.0, claimed_utc='x')
    _write_claim(out, 2, worker='far', pid=os.getpid(), host='another-host', create_time=None, claimed_utc='x')
    _write_claim(out, 3, worker='alive', pid=os.getpid(), host=host, create_time=me.create_time(), claimed_utc='x')
    first = _run(out)
    assert first['written'] == [0] and first['skipped_claimed'] == [1, 2, 3]
    state = run_status(out, recipe=CHEAP)
    assert state['done'] == [0] and state['claimed'] == [1, 2, 3] and set(state['claims']) == {1, 2, 3}
    with pytest.raises(RunnerError, match='claimed but unwritten \\[1, 2, 3\\]'):
        collate(out, recipe=CHEAP)
    partial = collate(out, recipe=CHEAP, allow_partial=True)
    assert Path(partial).name == 'collated_partial_1of4'
    extra = read_meta(partial)['extra']['featurize_cifs']
    assert extra['partial'] is True and extra['allow_partial'] is True and extra['missing_chunks'] == [1, 2, 3]
    assert MolCrystalData.load(partial).num_graphs == 4

    report = reclaim(out)[fc._runkey(CHEAP, fc.run_identity())]
    assert report == dict(freed=[1], alive=[3], other_host=[2], unreadable=[])
    rdir = out / fc._runkey(CHEAP, fc.run_identity())
    assert len(list((rdir / 'claims' / 'freed').iterdir())) == 1          # moved aside, not deleted
    assert _run(out)['written'] == [1]
    report = reclaim(out, chunks=[2, 3])[fc._runkey(CHEAP, fc.run_identity())]
    assert sorted(report['freed']) == [2, 3]
    assert _run(out)['written'] == [2, 3]
    assert_same_batch(MolCrystalData.load(collate(out, recipe=CHEAP)), single)


def test_reclaim_moves_a_dead_process_partial_store_aside(tmp_path):
    out = tmp_path / 'run'
    pid = _dead_pid()
    _write_claim(out, 0, worker='dead', pid=pid, host=socket.gethostname(), create_time=1.0, claimed_utc='x')
    _run(out)
    rdir = out / fc._runkey(CHEAP, fc.run_identity())
    partial = rdir / 'chunks' / f'chunk_000000.partial-{pid}'
    partial.mkdir()
    (partial / 'data.pt').write_bytes(b'half')
    assert reclaim(out)[rdir.name]['freed'] == [0]
    assert not partial.exists() and len(list((rdir / 'chunks' / 'orphans').iterdir())) == 1
    assert _run(out)['written'] == [0]


def test_a_failing_chunk_is_recorded_keeps_its_claim_and_the_walk_goes_on(tmp_path, monkeypatch, single):
    out = tmp_path / 'run'
    real = crystal_recipes.run_chain_after_read

    def flaky(batch, recipe):
        if 'NUKCOL' in batch.identifier:
            raise RuntimeError('injected failure')
        return real(batch, recipe)

    monkeypatch.setattr(crystal_recipes, 'run_chain_after_read', flaky)
    summary = _run(out)
    k = PATHS.index(str(FIXTURES / 'NUKCOL.cif')) // CHUNK
    assert summary['failed'] == [k] and sorted(summary['written']) == sorted(set(range(4)) - {k})
    rdir = out / summary['runkey']
    record = json.loads((rdir / 'chunks' / f'chunk_{k:06d}.failed.json').read_text(encoding='utf-8'))
    assert 'injected failure' in record['error'] and 'Traceback' in record['traceback']
    assert run_status(out, recipe=CHEAP)['failed'] == [k]
    with pytest.raises(RunnerError, match=f'failed \\[{k}\\]'):
        collate(out, recipe=CHEAP)
    monkeypatch.setattr(crystal_recipes, 'run_chain_after_read', real)
    assert _run(out)['skipped_claimed'] == [k]                  # the claim stays: no retry without reclaim
    assert reclaim(out)[rdir.name]['alive'] == [k]              # this process holds it
    assert reclaim(out, chunks=[k])[rdir.name]['freed'] == [k]
    assert _run(out)['written'] == [k]
    assert_same_batch(MolCrystalData.load(collate(out, recipe=CHEAP)), single)


def test_an_orphaned_store_is_moved_aside_not_reused(tmp_path):
    out = tmp_path / 'run'
    summary = _run(out)
    rdir = out / summary['runkey']
    manifest = rdir / 'chunks' / 'chunk_000002.json'
    manifest.rename(rdir / 'chunks' / 'manifest_moved_by_test.json')     # as if the writer died before it
    assert run_status(out, recipe=CHEAP)['claimed'] == [2]
    assert reclaim(out, chunks=[2])[rdir.name]['freed'] == [2]
    assert _run(out)['written'] == [2]
    assert len(list((rdir / 'chunks' / 'orphans').iterdir())) == 1


def test_a_code_or_recipe_change_starts_a_new_set_beside_the_old(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    old = _run(out)['runkey']
    other = Recipe('cheap_test', [(s, dict(CHEAP.args(s), max_dev=0.06) if s == 'check_rebuild' else CHEAP.args(s))
                                  for s in CHEAP.step_names])
    new = _run(out, recipe=other)
    assert new['runkey'] != old and new['written'] == [0, 1, 2, 3]
    fake = dict(fc.run_identity(), digest='0' * 64)
    monkeypatch.setattr(fc, 'run_identity', lambda: fake)
    assert fc._runkey(CHEAP, fake) != old
    assert run_status(out, recipe=CHEAP)['todo'] == [0, 1, 2, 3]     # nothing written under that code
    monkeypatch.undo()
    assert run_status(out, recipe=CHEAP)['done'] == [0, 1, 2, 3]    # the old set is untouched


def test_the_identity_covers_the_whole_package_and_the_libraries(monkeypatch):
    """Not only the ingestion modules: geometry, building, constants and data_classes compute stored values too."""
    ident = fc.run_identity()
    package = Path(fc.__file__).resolve().parents[2]
    assert set(ident['files']) == {p.relative_to(package).as_posix() for p in package.rglob('*.py')}
    assert {'common/geometry_utils.py', 'crystal_building/utils.py', 'constants/asymmetric_units.py',
            'constants/atom_properties.py', 'constants/space_group_info.py', 'dataset_utils/data_classes.py',
            'dataset_utils/data_class_methods/crystal_steps.py'} <= set(ident['files'])
    assert {'torch', 'numpy', 'rdkit', 'spglib', 'ccdc'} <= set(ident['versions'])
    assert ident['store_code'] == code_digest()['digest'] and fc.run_identity()['digest'] == ident['digest']
    real = fc.importlib.metadata.version
    monkeypatch.setattr(fc.importlib.metadata, 'version', lambda d: '0.0.test' if d == 'spglib' else real(d))
    other = fc.run_identity()
    assert other['digest'] != ident['digest'] and fc._runkey(CHEAP, other) != fc._runkey(CHEAP, ident)
    assert [c for c in fc._identity_changes(ident, other) if c.startswith('version: spglib')]


def _edited(identity, name='common/geometry_utils.py'):
    """``identity`` as if ``name`` had been edited (a module outside crystal_store.code_digest's scope)."""
    files = dict(identity['files'], **{name: '0' * 64})
    return dict(identity, files=files, digest='e' * 64)


def test_a_code_edit_between_chunks_stops_the_worker_before_it_claims(tmp_path, monkeypatch, single):
    out = tmp_path / 'run'
    real = fc.run_identity
    state = dict(edited=False)
    monkeypatch.setattr(fc, 'run_identity', lambda: _edited(real()) if state['edited'] else real())

    def log(line):
        if ' chunk 0/4: ' in line:                                  # chunk 0 is written; the file is edited now
            state['edited'] = True

    summary = featurize_cifs(PATHS, out, recipe=CHEAP, chunk_size=CHUNK, log=log)
    assert summary['written'] == [0] and 'common/geometry_utils.py' in summary['stopped']
    assert not summary['skipped_claimed'] and not summary['failed']
    rdir = out / summary['runkey']
    assert sorted(p.name for p in (rdir / 'claims').glob('chunk_*.json')) == ['chunk_000000.json']   # nothing else
    state['edited'] = False                                          # the edit is reverted
    status = run_status(out, recipe=CHEAP)
    assert status['done'] == [0] and status['todo'] == [1, 2, 3] and status['claimed'] == []
    assert _run(out)['written'] == [1, 2, 3]
    assert_same_batch(MolCrystalData.load(collate(out, recipe=CHEAP)), single)


def test_a_code_edit_during_a_chunk_discards_it_and_frees_its_claim(tmp_path, monkeypatch, single):
    """The edit lands while chunk 1 runs: crystal_store.save records the edited ingestion modules' digest, which
    names code that did not make the rows. The chunk must end unwritten and unclaimed, never stuck."""
    from mxtaltools.dataset_utils.data_class_methods import crystal_store
    out = tmp_path / 'run'
    real_chain, real_digest = crystal_recipes.run_chain_after_read, crystal_store.code_digest

    def chain(batch, recipe):
        if str(FIXTURES / 'IDOVUT.cif') in batch.source_path:       # chunk 1
            monkeypatch.setattr(crystal_store, 'code_digest', lambda: dict(real_digest(), digest='d' * 64))
        return real_chain(batch, recipe)

    monkeypatch.setattr(crystal_recipes, 'run_chain_after_read', chain)
    assert PATHS.index(str(FIXTURES / 'IDOVUT.cif')) // CHUNK == 1
    summary = _run(out)
    assert summary['written'] == [0] and 'ingestion code dddddddddddd' in summary['stopped'] and not summary['failed']
    rdir = out / summary['runkey']
    assert not (rdir / 'chunks' / 'chunk_000001').exists() and not (rdir / 'chunks' / 'chunk_000001.json').exists()
    assert len(list((rdir / 'chunks' / 'orphans').iterdir())) == 1
    assert len(list((rdir / 'claims' / 'freed').iterdir())) == 1
    monkeypatch.undo()
    status = run_status(out, recipe=CHEAP)
    assert status['done'] == [0] and status['todo'] == [1, 2, 3] and status['claimed'] == []
    assert _run(out)['written'] == [1, 2, 3]
    assert_same_batch(MolCrystalData.load(collate(out, recipe=CHEAP)), single)


def test_reclaim_frees_a_chunk_whose_manifest_exists_but_which_is_not_written(tmp_path, single):
    """One definition of written: a manifest without a matching store (the state an earlier runner left after a
    mid-run edit) is not written, so reclaim frees its dead claim and moves manifest and store aside together."""
    out = tmp_path / 'run'
    summary = _run(out)
    rdir = out / summary['runkey']
    manifest = rdir / 'chunks' / 'chunk_000002.json'
    record = json.loads(manifest.read_text(encoding='utf-8'))
    manifest.write_text(json.dumps(dict(record, code_digest='0' * 64)), encoding='utf-8')     # names other code
    claim = rdir / 'claims' / 'chunk_000002.json'
    claim.write_text(json.dumps(dict(worker='dead', pid=_dead_pid(), host=socket.gethostname(), create_time=1.0,
                                     claimed_utc='x')), encoding='utf-8')
    assert run_status(out, recipe=CHEAP)['claimed'] == [2]
    assert _run(out)['skipped_claimed'] == [2]
    assert reclaim(out)[rdir.name] == dict(freed=[2], alive=[], other_host=[], unreadable=[])
    assert not manifest.exists() and not (rdir / 'chunks' / 'chunk_000002').exists()
    assert len(list((rdir / 'chunks' / 'orphans').iterdir())) == 2                          # manifest and store
    assert run_status(out, recipe=CHEAP)['todo'] == [2]
    assert _run(out)['written'] == [2]
    assert_same_batch(MolCrystalData.load(collate(out, recipe=CHEAP)), single)
    # a named chunk is freed the same way when its process is alive (this one)
    manifest.write_text(json.dumps(dict(record, identity_digest='0' * 64)), encoding='utf-8')
    claim.unlink()
    assert fc._claim(str(rdir), 2, 'me')
    assert reclaim(out)[rdir.name]['alive'] == [2]
    assert reclaim(out, chunks=[2])[rdir.name]['freed'] == [2] and not manifest.exists()


def test_status_lists_every_run_key_and_collate_takes_one_by_name(tmp_path, monkeypatch, single):
    out = tmp_path / 'run'
    old = _run(out)['runkey']
    real = fc.run_identity
    monkeypatch.setattr(fc, 'run_identity', lambda: _edited(real()))       # the installed code changed since
    state = run_status(out, recipe=CHEAP)
    assert state['runkey'] != old and state['current'] and state['todo'] == [0, 1, 2, 3]
    assert state['runs'][old]['done'] == 4 and not state['runs'][old]['current']
    assert 'changed: common/geometry_utils.py' in state['runs'][old]['code_changes']
    with pytest.raises(RunnerError, match=f"other run keys here hold written chunks {{'{old}': 4}}"):
        collate(out, recipe=CHEAP)
    with pytest.raises(RunnerError, match='written by other code'):
        collate(out, recipe=CHEAP, runkey=old)
    named = run_status(out, recipe=CHEAP, runkey=old)
    assert named['runkey'] == old and not named['current'] and named['done'] == [0, 1, 2, 3]
    target = collate(out, recipe=CHEAP, runkey=old, allow_other_code=True)
    assert Path(target) == out / old / 'collated'
    assert_same_batch(MolCrystalData.load(target), single)
    extra = read_meta(target)['extra']['featurize_cifs']
    assert extra['allow_other_code'] is True and extra['code_changes'] == ['changed: common/geometry_utils.py']
    assert extra['rows_identity'] == real()['digest'] and extra['installed_identity'] == 'e' * 64
    monkeypatch.undo()
    assert run_status(out, recipe=CHEAP)['runs'][old]['collated'] == ['collated']
    other = Recipe('other_test', [(step, CHEAP.args(step)) for step in CHEAP.step_names])
    with pytest.raises(RunnerError, match='ran recipe'):
        run_status(out, recipe=other, runkey=old)
    with pytest.raises(RunnerError, match='no run'):
        run_status(out, recipe=CHEAP, runkey='cheap_test-000000000000')
    with pytest.raises(ValueError, match='run key'):
        run_status(out, recipe=CHEAP, runkey='../x')
    with pytest.raises(ValueError, match='allow_other_code'):
        collate(out, recipe=CHEAP, allow_other_code='yes')


def test_claim_creation_is_exclusive(tmp_path):
    (tmp_path / 'claims').mkdir()
    assert fc._claim(str(tmp_path), 5, 'a') is True
    assert fc._claim(str(tmp_path), 5, 'b') is False
    claim = json.loads((tmp_path / 'claims' / 'chunk_000005.json').read_text(encoding='utf-8'))
    assert claim['worker'] == 'a' and claim['pid'] == os.getpid()
    assert sorted(p.name for p in (tmp_path / 'claims').iterdir()) == ['chunk_000005.json']   # no temporary left


def test_collate_and_status_need_a_run(tmp_path):
    with pytest.raises(RunnerError, match='no featurize_cifs run'):
        run_status(tmp_path, recipe=CHEAP)
    with pytest.raises(RunnerError):
        collate(tmp_path, recipe=CHEAP)


# ======================================================================================
# concat_batches
# ======================================================================================

def test_concat_batches_refuses_what_it_cannot_join(single):
    a, b = single.select_rows([0, 1]), single.select_rows([2, 3])
    with pytest.raises(ValueError, match='at least one'):
        concat_batches([])
    c = single.select_rows([4])
    _set(c, 'extra', torch.zeros(1, 2), 'row')
    with pytest.raises(ValueError, match='other attributes'):
        concat_batches([a, c])
    d, e = single.select_rows([0]), single.select_rows([1])
    _set(d, 'extra', torch.zeros(1, 2), 'row')
    _set(e, 'extra', torch.zeros(1, 3), 'row')
    with pytest.raises(ValueError, match='ROW_PADS'):
        concat_batches([d, e])
    f = single.select_rows([0])
    f['steps_run'] = f['steps_run'] + [dict(step='other', args={})]
    with pytest.raises(ValueError, match="'steps_run' differs"):
        concat_batches([f, single.select_rows([1])])
    assert_same_batch(concat_batches([a, b]), single.select_rows([0, 1, 2, 3]))
    assert set(ROW_PADS) >= {'aunit_centroid', 'aunit_op', 'unit_radius', 'unit_fingerprint'}


# ======================================================================================
# command line
# ======================================================================================

def test_command_line(tmp_path, monkeypatch, capsys, single):
    monkeypatch.setattr(crystal_recipes, 'RECIPES', MappingProxyType({'cheap_test': CHEAP}))
    paths_file = tmp_path / 'paths.txt'
    paths_file.write_text('\n'.join(Path(p).name for p in PATHS) + '\n', encoding='utf-8')
    out = str(tmp_path / 'run')
    assert fc.main(['--out', out, '--recipe', 'cheap_test', '--paths-file', str(paths_file), '--cif-dir',
                    str(FIXTURES), '--chunk-size', '5', '--worker-id', 'cli1']) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['written'] == 4 and printed['failed'] == 0
    assert fc.main(['--out', out, '--recipe', 'cheap_test', '--status']) == 0
    assert json.loads(capsys.readouterr().out)['done'] == [0, 1, 2, 3]
    assert fc.main(['--out', out, '--reclaim']) == 0
    capsys.readouterr()
    assert fc.main(['--out', out, '--recipe', 'cheap_test', '--collate']) == 0
    target = capsys.readouterr().out.strip()
    assert_same_batch(MolCrystalData.load(target), single)
    assert fc.main(['--out', str(tmp_path / 'none'), '--recipe', 'cheap_test', '--collate']) == 1
    with pytest.raises(SystemExit):
        fc.main(['--out', out, '--status'])                     # --recipe missing
    runkey = Path(target).parent.name
    assert fc.main(['--out', out, '--recipe', 'cheap_test', '--status', '--runkey', runkey]) == 0
    assert json.loads(capsys.readouterr().out)['runs'][runkey]['done'] == 4
    with pytest.raises(SystemExit):
        fc.main(['--out', out, '--reclaim', '--runkey', runkey])  # --runkey goes with --status or --collate


def test_command_line_reads_a_paths_file_with_a_byte_order_mark(tmp_path, monkeypatch, capsys):
    """PowerShell's > writes UTF-8 with a byte-order mark: the first name must not become '\\ufeffNAME.cif'."""
    monkeypatch.setattr(crystal_recipes, 'RECIPES', MappingProxyType({'cheap_test': CHEAP}))
    names = [Path(p).name for p in PATHS[:4]]
    paths_file = tmp_path / 'paths_bom.txt'
    paths_file.write_bytes(b'\xef\xbb\xbf' + '\r\n'.join(names).encode('utf-8') + b'\r\n')
    out = tmp_path / 'run'
    assert fc.main(['--out', str(out), '--recipe', 'cheap_test', '--paths-file', str(paths_file), '--cif-dir',
                    str(FIXTURES), '--chunk-size', '4']) == 0
    capsys.readouterr()
    assert (out / 'paths.txt').read_text(encoding='utf-8').split('\n')[:-1] == sorted(str(FIXTURES / n) for n in names)
    batch = MolCrystalData.load(next(out.glob('cheap_test-*')) / 'chunks' / 'chunk_000000')
    table = status_table()
    assert batch.num_graphs == 4 and all(table[s] != 'unreadable_file' for s in batch.status.tolist())
    hidden = tmp_path / 'paths_hidden.txt'                     # a format character inside a name is refused loudly
    hidden.write_text(names[0] + '\n' + names[1][:2] + '​' + names[1][2:] + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='control or format'):
        fc.main(['--out', str(tmp_path / 'run2'), '--recipe', 'cheap_test', '--paths-file', str(hidden), '--cif-dir',
                 str(FIXTURES), '--chunk-size', '4'])
    with pytest.raises(ValueError, match='control or format'):
        _run(tmp_path / 'run3', paths=[PATHS[0], '﻿' + PATHS[1]])
    assert not (tmp_path / 'run2').exists() and not (tmp_path / 'run3').exists()

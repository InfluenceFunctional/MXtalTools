"""
crystal_search run-state safety (run_search.py + crystal_search/run_state.py).

1. Saves are atomic: a save that dies part-way leaves the previous complete file.
2. A kill between the output save and the progress save leaves one uncommitted batch; the resume drops it and redoes
   it, so no batch is appended twice.
3. One writer per run: a second owner is refused while the lease is fresh, may take over a stale lease, and a writer that
   has lost its lease stops before writing. The same SLURM job (a requeue) re-acquires its own lease.
4. A stop request (stop file or SIGUSR1) ends the run cleanly between batches; the resume finishes the rest.

CPU only; the optimiser is a scripted stand-in.
"""
import json
from pathlib import Path

import pytest
import torch

import mxtaltools.crystal_search.run_search as rs
import mxtaltools.crystal_search.run_state as st
from mxtaltools.common.config_processing import dict2namespace
from mxtaltools.dataset_utils.data_classes import MolCrystalData

ACRIDINE = Path(__file__).resolve().parent / 'datasets' / 'mini_acridine.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']


def _seeds(n):
    if not ACRIDINE.exists():
        pytest.skip(f'{ACRIDINE} not present')
    base = [c for c in torch.load(ACRIDINE, weights_only=False) if int(c.sg_ind) == 14 and int(c.z_prime) == 2][0]
    base = base.clone()
    for k in DROP:
        if k in base.keys():
            delattr(base, k)
    out = []
    for i in range(n):
        c = base.clone()
        L = c.cell_lengths.clone()
        L.reshape(-1)[0] = 10.0 + i  # seed i is traceable by its a length
        c.cell_lengths = L
        out.append(c)
    return out


def _config(tmp_path, n, **kw):
    cfg = {'device': 'cpu', 'target_path': None, 'out_dir': str(tmp_path), 'run_name': 'statetest',
           'force_restart_run': False, 'batch_size': 3, 'grow_batch_size': False, 'save_trajs': False,
           'num_samples': n, 'init_sample_method': 'data', 'init_reduced': True, 'init_target_cp': 'std',
           'opt_seed': 0, 'mol_seed': 0, 'dataset_path': None, 'lease_settle_s': 0.0,
           'opt': [{'optim_target': 'elj', 'init_lr': 0.01, 'max_num_steps': 3, 'optimizer_func': 'rprop',
                    'show_tqdm': False}]}
    cfg.update(kw)
    return dict2namespace(cfg)


def _identity_optimiser(seen):
    def fake_optimize(self, return_record=False, **kwargs):
        seen.append([float(x) for x in self.cell_lengths[:, 0]])
        out = self.batch_to_list()
        return (out, {}) if return_record else out
    return fake_optimize


def _setup(monkeypatch, n):
    seeds = _seeds(n)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    seen = []
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _identity_optimiser(seen))
    return seen


def _a_lengths(outs):
    return [round(float(o.cell_lengths.reshape(-1)[0]), 3) for o in outs]


# ---------------------------------------------------------------------------
# 1-2
# ---------------------------------------------------------------------------

def test_atomic_save_keeps_the_previous_file_when_a_save_dies(tmp_path, monkeypatch):
    path = tmp_path / 'x.pt'
    st.atomic_torch_save([1, 2, 3], path)
    real = torch.save

    def dying_save(obj, f):
        real(obj, f)  # the temporary file is written...
        raise OSError('killed')  # ...and the process dies before the rename
    monkeypatch.setattr(st.torch, 'save', dying_save)
    with pytest.raises(OSError):
        st.atomic_torch_save([9], path)
    assert torch.load(path) == [1, 2, 3], 'the previous complete file must survive'
    assert list(tmp_path.glob('*.tmp')) == [], 'the temporary file must not be left behind'


def test_kill_between_output_and_progress_saves_is_undone_on_resume(tmp_path, monkeypatch):
    n = 9
    seen = _setup(monkeypatch, n)
    calls = {'n': 0}
    real_dump = rs.atomic_json_dump

    def dump_dies_on_second_batch(obj, path):
        calls['n'] += 1  # call 1 commits the empty state, call 2 batch 1's progress, call 3 batch 2's
        if calls['n'] == 3 and str(path).endswith('_progress.json'):
            raise KeyboardInterrupt  # the job is killed after batch 2's outputs were saved, before its progress
        return real_dump(obj, path)
    monkeypatch.setattr(rs, 'atomic_json_dump', dump_dies_on_second_batch)
    with pytest.raises(KeyboardInterrupt):
        rs.crystal_search(_config(tmp_path, n))
    saved = torch.load(tmp_path / 'statetest.pt', weights_only=False)
    progress = json.loads((tmp_path / 'statetest_progress.json').read_text())
    assert len(saved) == 6 and progress['n_out'] == 3, 'setup: batch 2 is in the outputs but not in the progress file'

    monkeypatch.setattr(rs, 'atomic_json_dump', real_dump)
    outs = rs.crystal_search(_config(tmp_path, n))
    assert _a_lengths(outs) == [10.0 + i for i in range(n)], 'each seed exactly once, in order: no batch appended twice'


def test_committed_outputs_refuses_an_output_older_than_its_progress():
    assert st.committed_outputs(list(range(5)), {'n_out': 3}) == [0, 1, 2]
    assert st.committed_outputs(list(range(5)), {}) == list(range(5))
    with pytest.raises(RuntimeError, match='older than its progress'):
        st.committed_outputs(list(range(2)), {'n_out': 3})


# ---------------------------------------------------------------------------
# 3. lease
# ---------------------------------------------------------------------------

def test_second_writer_is_refused_while_the_lease_is_fresh_and_may_take_over_a_stale_one(tmp_path, monkeypatch):
    stem = str(tmp_path / 'run')
    monkeypatch.setenv('SLURM_JOB_ID', '111')
    a = st.RunLease(stem, stale_after_s=600, settle_s=0)
    a.acquire()
    monkeypatch.setenv('SLURM_JOB_ID', '222')
    b = st.RunLease(stem, stale_after_s=600, settle_s=0)
    with pytest.raises(st.RunLeaseLost, match='held by slurm:111'):
        b.acquire()
    stale_b = st.RunLease(stem, stale_after_s=-1, settle_s=0)  # the lease counts as stale
    stale_b.acquire()
    monkeypatch.setenv('SLURM_JOB_ID', '111')
    with pytest.raises(st.RunLeaseLost, match='held by slurm:222'):
        a.check()  # the old owner finds it lost and must not write
    requeued = st.RunLease(stem, stale_after_s=600, settle_s=0)
    monkeypatch.setenv('SLURM_JOB_ID', '222')
    st.RunLease(stem, stale_after_s=600, settle_s=0).acquire()  # the same job (a requeue) takes its own lease back
    monkeypatch.setenv('SLURM_JOB_ID', '111')
    with pytest.raises(st.RunLeaseLost):
        requeued.acquire()


def test_run_search_refuses_to_start_on_a_live_lease_and_releases_its_own(tmp_path, monkeypatch):
    n = 4
    _setup(monkeypatch, n)
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    other = st.RunLease(str(tmp_path / 'statetest'), stale_after_s=600, settle_s=0)
    other.acquire()
    monkeypatch.setenv('SLURM_JOB_ID', '1000')
    with pytest.raises(st.RunLeaseLost):
        rs.crystal_search(_config(tmp_path, n))
    assert not (tmp_path / 'statetest.pt').exists(), 'a refused run must not write'
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    other.release()
    monkeypatch.setenv('SLURM_JOB_ID', '1000')
    rs.crystal_search(_config(tmp_path, n))
    assert not (tmp_path / 'statetest_owner.json').exists(), 'a finished run releases its lease'


def test_losing_the_lease_mid_run_stops_before_writing(tmp_path, monkeypatch):
    n = 9
    seeds = _seeds(n)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    calls = {'n': 0}

    def optimise_then_get_taken_over(self, return_record=False, **kwargs):
        calls['n'] += 1
        if calls['n'] == 2:  # during batch 2 another job takes the run over
            st.atomic_json_dump({'owner': 'slurm:2'}, str(tmp_path / 'statetest_owner.json'))
        out = self.batch_to_list()
        return (out, {}) if return_record else out
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', optimise_then_get_taken_over)
    with pytest.raises(st.RunLeaseLost):
        rs.crystal_search(_config(tmp_path, n))
    assert len(torch.load(tmp_path / 'statetest.pt', weights_only=False)) == 3, 'batch 2 must not have been written'
    assert json.loads((tmp_path / 'statetest_owner.json').read_text())['owner'] == 'slurm:2', \
        'the loser must not release (delete) the new owner\'s lease'


# ---------------------------------------------------------------------------
# 4. stop requests
# ---------------------------------------------------------------------------

def test_stop_file_ends_the_run_between_batches_and_the_resume_finishes(tmp_path, monkeypatch):
    n = 9
    seeds = _seeds(n)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    stop_file = tmp_path / 'STOP'
    calls = {'n': 0}

    def optimise_and_request_stop(self, return_record=False, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            stop_file.touch()  # the coordinator asks for a stop while batch 1 runs
        out = self.batch_to_list()
        return (out, {}) if return_record else out
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', optimise_and_request_stop)
    first = rs.crystal_search(_config(tmp_path, n, stop_files=[str(stop_file)]))
    assert len(first) == 3, 'batch 1 completes and is saved; batch 2 never starts'
    stop_file.unlink()
    outs = rs.crystal_search(_config(tmp_path, n, stop_files=[str(stop_file)]))
    assert _a_lengths(outs) == [10.0 + i for i in range(n)]


def test_sigusr1_requests_a_stop():
    import signal
    if not hasattr(signal, 'SIGUSR1'):
        pytest.skip('no SIGUSR1 on this platform')
    s = st.StopRequest()
    try:
        assert s.reason() is None
        signal.raise_signal(signal.SIGUSR1)
        assert s.reason() == f'signal {signal.SIGUSR1}'
    finally:
        s.close()


# ---------------------------------------------------------------------------
# 5. search features: keep_lowest_fraction (in-job pre-screen) and the OOM batch-size ceiling
# ---------------------------------------------------------------------------

def test_keep_lowest_fraction_passes_only_the_lowest_rows_on(tmp_path, monkeypatch):
    n = 8
    seeds = _seeds(n)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    stage_sizes = []

    def optimise(self, return_record=False, **kwargs):
        stage_sizes.append(self.num_graphs)
        out = self.batch_to_list()
        for c in out:  # energy = a length: seeds 10..17, lower is better
            c.elj = torch.tensor([float(c.cell_lengths.reshape(-1)[0])])
        return (out, {}) if return_record else out
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', optimise)
    stages = [{'optim_target': 'elj', 'init_lr': 0.01, 'max_num_steps': 3, 'optimizer_func': 'rprop',
               'show_tqdm': False, 'keep_lowest_fraction': 0.5},
              {'optim_target': 'elj', 'init_lr': 0.01, 'max_num_steps': 3, 'optimizer_func': 'rprop',
               'show_tqdm': False}]
    outs = rs.crystal_search(_config(tmp_path, n, batch_size=4, opt=stages))
    assert stage_sizes == [4, 2, 4, 2], 'the second stage sees the kept half of each batch'
    assert _a_lengths(outs) == [10.0, 11.0, 14.0, 15.0], 'the lowest half of each batch is kept'
    assert 'keep_lowest_fraction' in stages[0], 'the config itself must not be mutated'


def test_oom_ceiling_stops_repeated_ooms(tmp_path, monkeypatch):
    n = 60
    seeds = _seeds(n)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    sizes = []

    def optimise(self, return_record=False, **kwargs):
        sizes.append(self.num_graphs)
        if self.num_graphs > 5:  # the GPU holds at most 5 of these crystals
            raise RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB')
        out = self.batch_to_list()
        return (out, {}) if return_record else out
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', optimise)
    counts = {}
    for flag in (False, True):
        sizes.clear()
        (tmp_path / str(flag)).mkdir()
        rs.crystal_search(_config(tmp_path / str(flag), n, batch_size=5, grow_batch_size=True, oom_ceiling=flag))
        counts[flag] = sum(s > 5 for s in sizes)
    assert counts[True] <= 3 < counts[False], f'OOM attempts with / without the ceiling: {counts}'


# ---------------------------------------------------------------------------
# 6. lease robustness (review fixes)
# ---------------------------------------------------------------------------

def test_requeue_takes_over_at_once_but_a_concurrent_process_of_the_same_job_is_refused(tmp_path, monkeypatch):
    stem = str(tmp_path / 'run')
    other = dict(owner='slurm:5:0:node1:999', job='5', restart=0, host='node1', pid=999)
    st.atomic_json_dump(other, stem + '_owner.json')  # a live process of job 5, first run
    monkeypatch.setenv('SLURM_JOB_ID', '5')
    monkeypatch.setenv('SLURM_RESTART_COUNT', '0')
    with pytest.raises(st.RunLeaseLost, match='slurm:5:0:node1:999'):
        st.RunLease(stem, stale_after_s=600, settle_s=0).acquire()
    monkeypatch.setenv('SLURM_RESTART_COUNT', '1')  # the same job after a requeue
    lease = st.RunLease(stem, stale_after_s=600, settle_s=0)
    lease.acquire()
    assert st.read_json(stem + '_owner.json')['restart'] == 1
    lease.release()


def test_heartbeat_keeps_the_lease_fresh_through_a_long_batch(tmp_path, monkeypatch):
    import time
    stem = str(tmp_path / 'run')
    monkeypatch.setenv('SLURM_JOB_ID', '11')
    lease = st.RunLease(stem, stale_after_s=1.0, settle_s=0, heartbeat_s=0.1)
    lease.acquire()
    time.sleep(2.5)  # a batch far longer than the staleness window, with no save in between
    monkeypatch.setenv('SLURM_JOB_ID', '12')
    with pytest.raises(st.RunLeaseLost):
        st.RunLease(stem, stale_after_s=1.0, settle_s=0).acquire()
    lease.release()
    st.RunLease(stem, stale_after_s=1.0, settle_s=0).acquire()  # released: free


def test_sigterm_releases_the_lease_and_the_run_resumes(tmp_path, monkeypatch):
    import signal
    n = 9
    seeds = _seeds(n)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    calls = {'n': 0}

    def optimise_then_terminated(self, return_record=False, **kwargs):
        calls['n'] += 1
        if calls['n'] == 2:
            signal.raise_signal(signal.SIGTERM)  # scancel / walltime during batch 2
        out = self.batch_to_list()
        return (out, {}) if return_record else out
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', optimise_then_terminated)
    with pytest.raises(SystemExit):
        rs.crystal_search(_config(tmp_path, n))
    assert not (tmp_path / 'statetest_owner.json').exists(), 'a terminated run must release its lease'
    assert len(torch.load(tmp_path / 'statetest.pt', weights_only=False)) == 3
    monkeypatch.setenv('SLURM_JOB_ID', '4242')  # a different job resubmits at once
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _identity_optimiser([]))
    outs = rs.crystal_search(_config(tmp_path, n))
    assert _a_lengths(outs) == [10.0 + i for i in range(n)]


def test_kill_in_the_first_batch_between_saves_does_not_duplicate_rows(tmp_path, monkeypatch):
    n = 6
    _setup(monkeypatch, n)
    real_dump = rs.atomic_json_dump
    calls = {'n': 0}

    def dump(obj, path):
        calls['n'] += 1
        if calls['n'] == 2:  # call 1 commits the empty state; call 2 is batch 1's progress: the kill lands here
            raise KeyboardInterrupt
        return real_dump(obj, path)
    monkeypatch.setattr(rs, 'atomic_json_dump', dump)
    with pytest.raises(KeyboardInterrupt):
        rs.crystal_search(_config(tmp_path, n))
    assert len(torch.load(tmp_path / 'statetest.pt', weights_only=False)) == 3, 'setup: batch 1 saved, not committed'
    monkeypatch.setattr(rs, 'atomic_json_dump', real_dump)
    outs = rs.crystal_search(_config(tmp_path, n))
    assert _a_lengths(outs) == [10.0 + i for i in range(n)]


def test_windows_replace_waits_for_a_reader_to_close(tmp_path):
    import sys
    import threading
    if sys.platform != 'win32':
        pytest.skip('Windows-only behaviour')
    path = tmp_path / 'x.pt'
    st.atomic_torch_save([1], path)
    fh = open(path, 'rb')  # another process reading the outputs
    threading.Timer(0.5, fh.close).start()
    st.atomic_torch_save([2], path)  # retried until the reader closed
    assert torch.load(path) == [2]


def test_a_final_partial_batch_ends_the_cursor_at_num_samples(tmp_path, monkeypatch):
    """5 samples in batches of 3: the progress cursor ends at 5, not 6, so a later run with more samples starts at the
    first sample not yet relaxed rather than skipping one."""
    seeds = _seeds(7)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _identity_optimiser([]))
    rs.crystal_search(_config(tmp_path, 5))
    assert json.loads((tmp_path / 'statetest_progress.json').read_text())['cursor'] == 5
    outs = rs.crystal_search(_config(tmp_path, 7))
    assert len(outs) == 7

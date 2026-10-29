"""crystal_search.stream_search (the search config's stream block).

Pins: the config check refuses what the stream does not reproduce; the per-row convergence test is
check_convergence's without the quorum; a small two-stage stream on CPU relaxes every start once, through both
stages, with a minima library, and a resumed run finds nothing left to do; slots refill (more starts than slots).
CPU, eLJ, the test molecule of the refold tests.
"""
from types import SimpleNamespace

import pytest
import torch

import mxtaltools.crystal_search.run_search as rs
from mxtaltools.common.config_processing import dict2namespace
from mxtaltools.crystal_search import stream_search as ss

from tests.test_fd_refold import _case, _rows, mol  # noqa: F401  (the fixture and row builders of the refold tests)

STAGE = dict(optim_target='elj', optimizer_func='rprop', centroid_boundary='wrap', init_lr=0.01, max_num_steps=10,
             convergence_eps=0.0, show_tqdm=False)


def _cfg(**over):
    c = dict(batch_size=4, init_sample_method='random', stream={'slots': 4}, opt=[dict(STAGE)])
    c.update(over)
    return SimpleNamespace(**c)


def test_config_check():
    assert ss.check_stream_config(SimpleNamespace(opt=[STAGE])) is None
    ok = ss.check_stream_config(_cfg())
    assert ok['slots'] == 4 and ok['polish_chunk'] == 16 and ok['check_every'] == 5
    assert ss.check_stream_config(_cfg(stream={}))['slots'] == 4  # slots default to batch_size
    for over in [dict(coord_dir='/x'), dict(init_sample_method='data'), dict(stream={'slotz': 3}),
                 dict(opt=[dict(STAGE, optimizer_func='adam')]), dict(opt=[dict(STAGE, centroid_boundary='clamp')]),
                 dict(opt=[dict(STAGE, enforce_reduced=True)]), dict(opt=[dict(STAGE, convergence='grad')]),
                 dict(opt=[dict(STAGE, keep_lowest_fraction=0.5)]), dict(opt=[dict(STAGE, fd_refold=True)]),
                 dict(opt=[dict(STAGE), dict(STAGE, optim_target='lj')])]:
        with pytest.raises(ValueError):
            ss.check_stream_config(_cfg(**over))


def test_per_row_convergence_matches_check_convergence():
    from mxtaltools.crystal_search.crystal_opt_utils import check_convergence
    torch.manual_seed(0)
    n, d, T = 3, 12, 80
    traj = torch.zeros(T, n, d)
    traj[:, 1] = torch.linspace(0, 1, T)[:, None]  # row 1 keeps moving
    traj[:, 2] = 0.001 * torch.randn(T, d)  # row 2 jitters slightly
    slots = ss._Slots(n, d, 'cpu')
    for t in range(T):
        x = traj[t]
        slots.ema_num = (1 - ss.EMA_ALPHA) * slots.ema_num + x
        slots.ema_den = (1 - ss.EMA_ALPHA) * slots.ema_den + 1
        slots.smooth[torch.arange(n), slots.nstep % ss.CONV_WINDOW] = slots.ema_num / slots.ema_den[:, None]
        slots.nstep += 1
    eps = 1e-3
    ref = check_convergence(traj, T, eps, None, None, active=torch.ones(n, dtype=torch.bool))  # quorum not reached
    assert [ss._converged(slots, i, eps) for i in range(n)] == ref.tolist() == [True, False, True]
    assert not ss._converged(slots, 0, 0.0)  # eps <= 0: never


def _placeholders(mol, n):
    _, p, h = _case('normal', n=n, seed=5)
    return _rows(mol, 2, p, h).batch_to_list()


def _run_cfg(tmp_path, n, force=True):
    return dict2namespace({
        'device': 'cpu', 'target_path': None, 'out_dir': str(tmp_path), 'run_name': 'streamtest',
        'force_restart_run': force, 'batch_size': 3, 'grow_batch_size': False, 'save_trajs': False,
        'num_samples': n, 'init_sample_method': 'random', 'init_reduced': True, 'init_target_cp': 'std',
        'opt_seed': 0, 'mol_seed': 0, 'dataset_path': None,
        'stream': {'slots': 3, 'check_every': 2, 'polish_chunk': 2},
        'minima': {'kT': 6.86, 'max_iter': 20},
        'opt': [dict(STAGE, compression_factor=1.0, max_num_steps=8),
                dict(STAGE, max_num_steps=6, early_stop=[[2, 1000.0]], early_stop_kT=6.86)]})


def test_stream_runs_every_start_once_and_resumes(mol, tmp_path, monkeypatch):
    n = 7  # more starts than slots: slots refill
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    out = rs.crystal_search(_run_cfg(tmp_path, n))
    assert sorted(int(r.start_index) for r in out) == list(range(n))
    for r in out:
        st = r.stage_steps.reshape(-1).tolist()
        assert st[0] >= 7 and st[1] >= 1  # stage 1 to its cap (convergence off), then stage 2
        assert 'minimum_id' in r.keys() and r.hessian.shape == (1, 12, 12)
    again = rs.crystal_search(_run_cfg(tmp_path, n, force=False))  # resume: nothing left
    assert len(again) == n
    assert sorted(int(r.start_index) for r in again) == list(range(n))


def test_out_of_memory_halves_the_chunk_and_retries(mol, tmp_path, monkeypatch):
    """One CUDA out-of-memory error in a descent step and one in the polish: the stream halves its evaluation chunk
    (or the polish's energy chunk, after undoing the group's library adds) and finishes every start once."""
    from mxtaltools.crystal_search import minima as mn
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    n = 5
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    real_analyze, real_finish = MolCrystalData.analyze, mn.finish_minima
    calls = dict(step=0, finish=0)

    def analyze(self, *a, **k):
        if k.get('return_cluster') and self.num_graphs > 1 and calls['step'] == 0:
            calls['step'] += 1
            raise RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB')
        return real_analyze(self, *a, **k)

    def finish(*a, **k):
        calls['finish'] += 1
        if calls['finish'] == 1:
            assert k['energy_chunk'] == 256
            raise RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB')
        assert k['energy_chunk'] == 128
        return real_finish(*a, **k)

    monkeypatch.setattr(MolCrystalData, 'analyze', analyze)
    monkeypatch.setattr(mn, 'finish_minima', finish)
    out = rs.crystal_search(_run_cfg(tmp_path, n))
    assert calls['step'] == 1 and calls['finish'] >= 2
    assert sorted(int(r.start_index) for r in out) == list(range(n))

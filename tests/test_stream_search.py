"""crystal_search.stream_search (the search config's stream block).

Pins: the config check refuses what the stream does not reproduce; the per-row convergence test is
check_convergence's without the quorum; a small two-stage stream on CPU relaxes every start once, through both
stages, with a minima library, and a resumed run finds nothing left to do; slots refill (more starts than slots);
seeds go first, straight to the polish; stop_p ends the random starts, and the seeds' own bound ends the seeds.
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


def test_example_config_passes_the_checks():
    """configs/crystal_searches/minima_stream/mipcas_elj.yaml, the example of a streaming search with minima."""
    from pathlib import Path
    from mxtaltools.common.config_processing import load_yaml
    path = Path(rs.__file__).resolve().parents[2] / 'configs' / 'crystal_searches' / 'minima_stream' / 'mipcas_elj.yaml'
    cfg = dict2namespace(load_yaml(str(path)))
    s = ss.check_stream_config(cfg)
    assert s['slots'] == 250 and s['stop_p'] == 0.01 and s['stop_band_kT'] == 5.0
    m = rs.check_minima_config(cfg)
    assert m['capture_d2'] == 9.0 and m['kT'] > 0


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


def test_seeds_go_first_straight_to_the_polish(mol, tmp_path, monkeypatch):
    """A first run's outputs as the seeds of a second: each seed is placed with its own cell parameters and
    handedness on the search's copy of its molecule, is polished once with a negative start index and no stage step,
    and the run writes one row per minimum with its hits."""
    n = 4
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    first = rs.crystal_search(_run_cfg(tmp_path / 'a', n))
    torch.save(first, tmp_path / 'seeds.pt')
    placed = ss._seed_starts(str(tmp_path / 'seeds.pt'), _placeholders(mol, n), 'cpu', done=set())
    assert [k for k, _ in placed] == [-1, -2, -3, -4]
    other = _placeholders(mol, 1)
    other[0].identifier = 'NOT_THIS_MOLECULE'
    with pytest.raises(ValueError):  # seeds are placed on the search's copy of their own molecule only
        ss._seed_starts(str(tmp_path / 'seeds.pt'), other, 'cpu', done=set())
    for (_, r), s in zip(placed, first):
        assert torch.allclose(r.full_cell_parameters(), s.full_cell_parameters(), atol=1e-5)
        assert torch.equal(r.aunit_handedness.reshape(-1), s.aunit_handedness.reshape(-1))
    cfg = _run_cfg(tmp_path / 'b', n)
    cfg.stream.seeds = str(tmp_path / 'seeds.pt')
    out = rs.crystal_search(cfg)
    assert sorted(int(r.start_index) for r in out) == list(range(-n, n))  # the n seeds and the n random starts
    for r in out:
        st = r.stage_steps.reshape(-1).tolist()
        assert st == [0, 0] if int(r.start_index) < 0 else st[0] >= 7
        assert 'minimum_id' in r.keys()
    table = torch.load(tmp_path / 'b' / 'streamtest_minima.pt', weights_only=False)
    assert sum(int(r.hits) + int(r.seed_hits) for r in table) == sum(1 for r in out if int(r.minimum_id) >= 0)



def test_stop_p_skips_the_remaining_seeds(mol, tmp_path, monkeypatch):
    """With stop_p the seeds' own bound (missing_mass over the seeds) is checked before each seed group: once it is
    met the remaining seeds are skipped, the random starts run, and a resumed run skips those seeds too."""
    from mxtaltools.crystal_search import minima as mn
    n = 4
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    torch.save(rs.crystal_search(_run_cfg(tmp_path / 'a', n)), tmp_path / 'seeds.pt')
    real = mn.missing_mass

    def bound(rows, lib, band_kT, seeds=False):  # the seeds' bound is met once any seed is in; the starts' never
        mm = real(rows, lib, band_kT, seeds=seeds)
        return dict(mm, p_hi=0.0 if seeds and mm['n'] else float('inf'))

    monkeypatch.setattr(mn, 'missing_mass', bound)
    cfg = _run_cfg(tmp_path / 'b', n)
    cfg.stream.seeds = str(tmp_path / 'seeds.pt')
    cfg.stream.stop_p = 0.01
    out = rs.crystal_search(cfg)
    assert sum(int(r.start_index) < 0 for r in out) == 2  # one group of polish_chunk = 2 seeds
    assert sorted(int(r.start_index) for r in out if int(r.start_index) >= 0) == list(range(n))
    with open(tmp_path / 'b' / 'streamtest_stream_effort.json') as fh:
        import json
        assert json.load(fh)['seeds_stopped']['skipped'] == 2
    cfg.force_restart_run = False
    assert len(rs.crystal_search(cfg)) == len(out)  # resumed: the skipped seeds stay skipped


def test_stop_p_draws_no_new_starts(mol, tmp_path, monkeypatch):
    """Once the missing-mass bound is below stop_p, the slots in flight finish and no start is drawn after."""
    from mxtaltools.crystal_search import minima as mn
    n = 12
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    monkeypatch.setattr(mn, 'missing_mass', lambda rows, lib, band_kT: dict(n=len(rows), f1=0, p_hi=0.0))
    cfg = _run_cfg(tmp_path, n)
    cfg.stream.stop_p = 0.01
    out = rs.crystal_search(cfg)
    assert len(out) < n  # the first finished group (2 rows) set the stop; the 3 slots then in flight finished
    with open(tmp_path / 'streamtest_stream_effort.json') as fh:
        import json
        assert json.load(fh)['stopped']['reason'] == 'stop_p'
    two_groups = _placeholders(mol, 4)
    for r, sg in zip(two_groups, (1, 2, 1, 2)):
        r.sg_ind = torch.tensor([sg])
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: two_groups)
    with pytest.raises(ValueError, match='one molecule, space group'):  # the band needs one lowest minimum
        rs.crystal_search(cfg)
    with pytest.raises(ValueError):
        cfg.minima = None
        ss.check_stream_config(cfg)


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

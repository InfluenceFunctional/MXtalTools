"""crystal_search.stream_search (the search config's stream block).

Pins: the config check refuses what the stream does not reproduce; the per-row convergence test is
check_convergence's without the quorum; a small two-stage stream on CPU relaxes every start once, through both
stages, with a minima library, and a resumed run finds nothing left to do; slots refill (more starts than slots);
seeds go first, straight to the polish; stop_p ends the random starts, and the seeds' own bound ends the seeds;
_bfgs_step converges on a quadratic, rejects a rise and always ends; a bfgs last stage leaves the polish less to do;
each stage runs at its own cutoff with the envelope; starts at known minima are captured in the last stage, not
in stage 1; a flat cell is dropped before its step.
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
    assert ok['slots'] == 4 and ok['polish_chunk'] == 1000 and ok['check_every'] == 5
    assert ss.check_stream_config(_cfg(stream={}))['slots'] == 4  # slots default to batch_size
    for over in [dict(coord_dir='/x'), dict(init_sample_method='data'), dict(stream={'slotz': 3}),
                 dict(opt=[dict(STAGE, optimizer_func='adam')]), dict(opt=[dict(STAGE, centroid_boundary='clamp')]),
                 dict(opt=[dict(STAGE, enforce_reduced=True)]), dict(opt=[dict(STAGE, convergence='grad')]),
                 dict(opt=[dict(STAGE, keep_lowest_fraction=0.5)]), dict(opt=[dict(STAGE, fd_refold=True)]),
                 dict(opt=[dict(STAGE), dict(STAGE, optim_target='lj')]), dict(stream={'first_polish_chunk': -1})]:
        with pytest.raises(ValueError):
            ss.check_stream_config(_cfg(**over))
    assert ss.check_stream_config(_cfg(stream={'polish_chunk': 8}))['first_polish_chunk'] == 8  # default: the same
    bfgs = dict(STAGE, optimizer_func='bfgs', convergence_eps=1e-4)
    assert ss.check_stream_config(_cfg(opt=[dict(STAGE), bfgs], minima={'kT': 1.0})) is not None  # last stage only
    for over in [dict(opt=[dict(STAGE), bfgs]),  # no minima block: no kT for its decrement
                 dict(opt=[bfgs, dict(STAGE)], minima={'kT': 1.0}),  # not the last stage
                 dict(opt=[dict(STAGE), dict(bfgs, compression_factor=1.0)], minima={'kT': 1.0}),
                 dict(opt=[dict(STAGE), dict(bfgs, convergence_eps=0.0)], minima={'kT': 1.0}),
                 dict(opt=[{k: v for k, v in STAGE.items() if k != 'init_lr'}]),  # rprop needs its lr
                 dict(opt=[dict(STAGE, cutoff=1.0)], minima={'kT': 1.0})]:  # cutoff inside the 1 A envelope
        with pytest.raises(ValueError):
            ss.check_stream_config(_cfg(**over))


def _quadratic(x, xs, A):
    dx = (x.double() - xs)
    return 0.5 * torch.einsum('ni,ij,nj->n', dx, A, dx), dx @ A


def test_bfgs_step_converges_rejects_and_gives_up():
    """_bfgs_step on a quadratic: converges (done, at the minimum) from a start several caps away; a trial that does
    not decrease the loss leaves the accepted point unchanged and halves the trust; a row that never decreases is done
    after two runs of POLISH_TRIES rejections (the reset in between), so a bfgs row always leaves its stage."""
    from mxtaltools.crystal_search.minima import POLISH_TRIES
    torch.manual_seed(3)
    d = 12
    A = torch.diag(torch.logspace(0, 3, d, dtype=torch.float64))
    xs = torch.rand(2, d, dtype=torch.float64)
    slots = ss._Slots(2, d, 'cpu')
    ib = torch.arange(2)
    for i in range(2):
        slots.begin_stage(i, (xs[i] + 0.3).float(), 0.0)
    for it in range(400):
        f, g = _quadratic(slots.X[ib], xs, A)
        ss._bfgs_step(slots, ib, f, g, kT=1.0, tol=1e-10, d=d)
        if bool(slots.bdone.all()):
            break
    assert bool(slots.bdone.all()) and it < 399 and not bool(slots.breset.any())  # converged, not given up
    assert float((slots.bx - xs).abs().max()) < 1e-3 and bool((slots.bnupd >= ss.BFGS_MIN_UPDATES).all())
    one = ss._Slots(1, d, 'cpu')
    one.begin_stage(0, xs[0].float() + 0.3, 0.0)
    f, g = _quadratic(one.X[:1], xs[:1], A)
    ss._bfgs_step(one, torch.arange(1), f, g, kT=1.0, tol=1e-10, d=d)  # the first evaluation: accepted
    x_acc, t0 = one.bx.clone(), float(one.btrust[0])
    step1 = (one.X[0].double() - x_acc[0]).norm()
    ss._bfgs_step(one, torch.arange(1), f + 1.0, g, kT=1.0, tol=1e-10, d=d)  # a trial that rose
    assert torch.equal(one.bx, x_acc) and float(one.btrust[0]) == t0 / 2 and int(one.btries[0]) == 1
    step2 = (one.X[0].double() - x_acc[0]).norm()  # the retry is a shorter step, even where no cap binds
    assert 0 < float(step2) <= ss.BFGS_SHRINK * float(step1) * (1 + 1e-5)
    for _ in range(2 * POLISH_TRIES):
        if bool(one.bdone[0]):
            break
        ss._bfgs_step(one, torch.arange(1), f + 1.0, g, kT=1.0, tol=1e-10, d=d)
    assert bool(one.bdone[0]) and bool(one.breset[0])


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


def test_a_bfgs_last_stage_hands_the_polish_converged_rows(mol, tmp_path, monkeypatch):
    """The same starts through rprop (40 steps) then rprop, and through rprop then bfgs, the last stage 100 steps at
    most either way: every start is relaxed once, bfgs rows leave their stage converged before the cap, and the polish
    has less left to do after bfgs."""
    import json
    n = 6
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    evals = {}
    for name, last in (('rprop', dict(STAGE, max_num_steps=100)),
                       ('bfgs', dict(STAGE, optimizer_func='bfgs', max_num_steps=100, convergence_eps=1e-4))):
        (tmp_path / name).mkdir()
        cfg = _run_cfg(tmp_path / name, n)
        cfg.opt = [dict(cfg.opt[0], max_num_steps=40), last]
        out = rs.crystal_search(cfg)
        assert sorted(int(r.start_index) for r in out) == list(range(n))
        assert all('minimum_id' in r.keys() for r in out)
        with open(tmp_path / name / 'streamtest_stream_effort.json') as fh:
            evals[name] = json.load(fh)['polish_evals']
        if name == 'bfgs':
            assert any(0 < int(r.stage_steps.reshape(-1)[1]) < 99 for r in out)  # left before the cap
    assert evals['bfgs'] < evals['rprop']


def test_each_stage_at_its_own_cutoff_with_the_envelope(mol, tmp_path, monkeypatch):
    """Every stage's evaluations use that stage's cutoff, and the minima block's lj_envelope in every stage."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import MolCrystalAnalysis
    n = 3
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    seen = set()
    real = MolCrystalAnalysis.analyze

    def spy(self, *a, **k):
        if k.get('elementwise') and k.get('return_cluster'):  # the stream's step evaluations
            seen.add((float(k['cutoff']), k.get('lj_envelope')))
        return real(self, *a, **k)

    monkeypatch.setattr(MolCrystalAnalysis, 'analyze', spy)
    cfg = _run_cfg(tmp_path, n)
    cfg.opt = [dict(cfg.opt[0], cutoff=6), cfg.opt[1]]
    rs.crystal_search(cfg)
    assert seen == {(6.0, 1.0), (10.0, 1.0)}


def test_capture_is_in_the_last_stage(mol, tmp_path, monkeypatch):
    """Starts placed at minima the seeds put in the library are not captured in stage 1 (no compression, a tiny lr:
    they sit inside the ellipsoids throughout it) but in the last stage, and are reported as those minima."""
    import mxtaltools.crystal_search.utils as su
    n = 3
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    first = rs.crystal_search(_run_cfg(tmp_path / 'a', n))
    torch.save(first, tmp_path / 'seeds.pt')

    def at_minima(config, batch, device, b0):
        src = [first[(b0 * 3 + j) % len(first)] for j in range(batch.num_graphs)]
        batch.aunit_handedness = torch.stack([r.aunit_handedness.reshape(-1) for r in src]).reshape(
            batch.aunit_handedness.shape).to(batch.aunit_handedness)
        batch.set_cell_parameters(torch.stack([r.full_cell_parameters().reshape(-1) for r in src]).to(device),
                                  skip_box_analysis=False)
        return batch

    monkeypatch.setattr(su, 'get_initial_state', at_minima)
    cfg = _run_cfg(tmp_path / 'b', n)
    cfg.stream.seeds = str(tmp_path / 'seeds.pt')
    cfg.minima.capture_every = 2
    cfg.opt = [dict(cfg.opt[0], init_lr=1e-6, compression_factor=0.0), cfg.opt[1]]
    out = rs.crystal_search(cfg)
    starts = [r for r in out if int(r.start_index) >= 0]
    assert len(starts) == n
    for r in starts:
        st = r.stage_steps.reshape(-1).tolist()
        assert int(r.polish_status) == 0 and bool(r.recapture) and st[0] > 0 and st[1] > 0



def test_a_flat_cell_is_dropped_before_its_step(mol, tmp_path, monkeypatch):
    """A start whose angles are each in range but together nearly flat (volume / abc 0.008) is dropped before its
    first evaluation, counted in the effort file, and every other start is relaxed."""
    import json
    import math
    import mxtaltools.crystal_search.utils as su
    n = 4
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    real = su.get_initial_state

    def one_flat(config, batch, device, b0):
        batch = real(config, batch, device, b0)
        if b0 == 0:
            x = batch.full_cell_parameters().clone()
            x[0, 3:6] = torch.tensor([math.radians(v) for v in (52.9, 88.4, 36.0)], device=x.device)
            batch.set_cell_parameters(x, skip_box_analysis=False)
        return batch

    monkeypatch.setattr(su, 'get_initial_state', one_flat)
    out = rs.crystal_search(_run_cfg(tmp_path, n))
    assert sorted(int(r.start_index) for r in out) == [1, 2, 3]
    with open(tmp_path / 'streamtest_stream_effort.json') as fh:
        assert json.load(fh)['flat_dropped'] == 1


def test_stop_p_draws_no_new_starts(mol, tmp_path, monkeypatch):
    """Once the missing-mass bound is below stop_p the run ends at once: no start is drawn after, and the rows in
    flight are dropped (counted in the effort file)."""
    from mxtaltools.crystal_search import minima as mn
    n = 12
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _placeholders(mol, n))
    monkeypatch.setattr(mn, 'missing_mass', lambda rows, lib, band_kT: dict(n=len(rows), f1=0, p_hi=0.0))
    cfg = _run_cfg(tmp_path, n)
    cfg.stream.stop_p = 0.01
    out = rs.crystal_search(cfg)
    assert len(out) == 2  # the first finished group (polish_chunk 2) set the stop; nothing else was finished
    with open(tmp_path / 'streamtest_stream_effort.json') as fh:
        import json
        stopped = json.load(fh)['stopped']
        assert stopped['reason'] == 'stop_p' and stopped['dropped'] >= 1
    cfg.stream.first_polish_chunk = 1  # a smaller first group: the stop is checked after one row
    assert len(rs.crystal_search(cfg)) == 1
    cfg.stream.first_polish_chunk = None
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

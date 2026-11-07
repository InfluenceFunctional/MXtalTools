"""The streaming search's endpoint mode (minima.polish 'none'): balls of the FD identity metric instead of Hessian
ellipsoids, a bfgs last stage that stops on its force, and the kept trajectory states.

Pins: metric_factor measures a step in Angstrom as crystal_fd's identity metric does; a ball library holds no
Hessian, matches and captures by radius, and is rebuilt from its rows (never mixed with polished minima); ball_net is
a radius-net; _bfgs_step with fmax stops on the force in the metric, and with energy_noise converges where energy
noise makes the energy test reject every short step; the config check refuses what the mode cannot run; a small CPU
stream in the mode relaxes every start once, reports endpoints (no Hessians) and resumes; seeds start in the last
stage, before the random starts; kept trajectory states belong to finished starts, and a band set and a basin set
without Hessians are written and read back.
CPU, eLJ, the test molecule of the refold tests.
"""
import json
import math
from types import SimpleNamespace

import pytest
import torch

import mxtaltools.crystal_search.run_search as rs
from mxtaltools.crystal_search import minima as mn
from mxtaltools.crystal_search import stream_search as ss

from tests.test_fd_refold import mol  # noqa: F401  (the fixture of the refold tests)
from tests.test_stream_search import STAGE, _cfg, _placeholders, _run_cfg

X0 = torch.tensor([7.0, 8.0, 9.0, 1.4, 1.6, 1.7, 0.2, 0.3, 0.4, 0.5, -0.3, 0.8], dtype=torch.float64)


def test_metric_factor_measures_angstrom():
    """|W^T dx| = |A dx| with A: lengths as they are, angles x sqrt(bc), sqrt(ac), sqrt(ab), centres through T_fc,
    rotation vectors x rg; rg per row or shared."""
    from mxtaltools.crystal_fd.ops import cell_matrices
    rg = 2.5
    W = mn.metric_factor(X0[None], rg)[0]
    a, b, c = X0[:3]
    for k, scale in [(0, 1.0), (2, 1.0), (3, float(torch.sqrt(b * c))), (5, float(torch.sqrt(a * b)))]:
        dx = torch.zeros(12, dtype=torch.float64)
        dx[k] = 1e-3
        assert float((W.T @ dx).norm()) == pytest.approx(1e-3 * scale, rel=1e-9)
    from scipy.spatial.transform import Rotation
    for k in (9, 10, 11):  # the rotation block: rg times the geodesic angle between the two orientations
        dx = torch.zeros(12, dtype=torch.float64)
        dx[k] = 1e-5
        r0 = X0[9:12].numpy()
        ang = (Rotation.from_rotvec(r0).inv() * Rotation.from_rotvec(r0 + dx[9:12].numpy())).magnitude()
        assert float((W.T @ dx).norm()) == pytest.approx(rg * ang, rel=1e-6)
    dx = torch.zeros(12, dtype=torch.float64)
    dx[6:9] = torch.tensor([1e-3, -2e-3, 5e-4])
    assert float((W.T @ dx).norm()) == pytest.approx(float((cell_matrices(X0[None, :6])[0] @ dx[6:9]).norm()))
    two = mn.metric_factor(torch.stack([X0, X0]), torch.tensor([1.0, 3.0]))
    assert torch.allclose(two[1, 9:12, 9:12], 3.0 * two[0, 9:12, 9:12])
    with pytest.raises(NotImplementedError):
        mn.metric_factor(torch.zeros(1, 18), 1.0)
    assert torch.allclose(mn.so3_jacobian(torch.zeros(1, 3))[0], torch.eye(3, dtype=torch.float64))


def _key():
    return ('MOL', 2, 1, (1,))


def test_ball_library_matches_and_captures_by_radius():
    lib = mn.MinimaLibrary(ball=True)
    W = mn.metric_factor(X0[None], 2.0)[0]
    lib.add(X0, None, -5.0, _key(), dict(x=X0, hand=torch.ones(1)), W=W)
    with pytest.raises(ValueError):  # a ball library takes W, not H
        lib.add(X0, torch.eye(12), -5.0, _key(), dict(x=X0, hand=torch.ones(1)))
    near, far = X0.clone(), X0.clone()
    near[0] += 0.05  # 0.05 A along a
    far[0] += 0.5
    ids, d2 = lib.nearest(torch.stack([near, far]), [_key(), _key()])
    assert ids.tolist() == [0, 0] and float(d2[0]) == pytest.approx(0.05 ** 2) and float(d2[1]) == pytest.approx(0.25)
    assert lib.match(near, W, _key(), 0.1 ** 2) == 0 and lib.match(far, W, _key(), 0.1 ** 2, f=-1.0) == -1
    shifted = X0.clone()
    shifted[6] += 0.5  # an equivalent origin of P-1 (half a cell): the same crystal
    assert float(lib.nearest(shifted[None], [_key()])[1][0]) < 1e-12
    with pytest.raises(ValueError):  # finish_minima needs Hessians
        mn.finish_minima([], lib, 'elj', 1.0)


def test_ball_net_is_a_radius_net():
    torch.manual_seed(1)
    x = X0[None] + 0.02 * torch.randn(300, 12, dtype=torch.float64)
    keys = [_key()] * len(x)
    r = 0.15
    kept = mn.ball_net(x, keys, torch.full((len(x),), 2.0), r)
    assert 0 in kept.tolist() and 1 < len(kept) < len(x)
    W = mn.metric_factor(x[kept], 2.0)
    dist = torch.einsum('mji,bmj->bmi', W, mn.min_image(x[:, None] - x[kept][None])).norm(dim=-1)  # [n, kept]
    assert float(dist.min(1).values.max()) <= r + 1e-9  # every row within r of a kept one
    kd = dist[kept]  # kept rows against each other: apart, the earlier one's factor
    assert float((kd + torch.eye(len(kept), dtype=torch.float64) * 1e9).min()) > r * 0.5


def _quadratic_y(x, xs, lam, rg):
    """E/kT = 0.5 sum lam_i y_i^2 with y = A (x - xs), A the metric map at xs; gradient in the parameters."""
    W = mn.metric_factor(xs, rg)
    y = torch.einsum('nji,nj->ni', W, x.double() - xs)
    f = 0.5 * (lam * y ** 2).sum(1)
    g = torch.einsum('nij,nj->ni', W, lam * y)
    return f, g


def test_bfgs_fmax_and_energy_noise():
    """A quadratic in the metric's coordinates, 800 times stiffer along one axis than another, with energy noise of
    3e-5 kT: with fmax and energy_noise every row ends by the force test, near its minimum; with fmax alone and no
    noise tolerance the noise leaves rows giving up instead."""
    torch.manual_seed(2)
    d, n = 12, 6
    lam = torch.logspace(math.log10(0.4), math.log10(320), d, dtype=torch.float64)
    xs = X0[None].repeat(n, 1)
    xs[:, :3] += torch.rand(n, 3, dtype=torch.float64)
    gen = torch.Generator().manual_seed(4)

    def run(noise):
        slots = ss._Slots(n, d, 'cpu')
        ib = torch.arange(n)
        for i in range(n):
            slots.begin_stage(i, (xs[i] + 0.02 * torch.randn(d, dtype=torch.float64)).float(), 0.0)
            slots.rg[i] = 2.0
            slots.start[i] = i
        gave_up = torch.zeros(n, dtype=torch.bool)
        for it in range(3000):
            f, g = _quadratic_y(slots.X[ib], xs, lam, 2.0)
            f = f + 3e-5 * torch.randn(n, generator=gen, dtype=torch.float64)
            before = slots.bdone.clone()
            ss._bfgs_step(slots, ib, f, g, kT=1.0, tol=1e-4, d=d, fmax=0.01, noise=noise)
            gave_up |= slots.bdone & ~before & slots.breset
            if bool(slots.bdone.all()):
                break
        _, g = _quadratic_y(slots.bx, xs, lam, 2.0)
        force = torch.linalg.solve(mn.metric_factor(slots.bx, 2.0), g[..., None])[..., 0].abs().amax(1)
        return slots, gave_up, force

    slots, gave_up, force = run(1e-4)
    assert bool(slots.bdone.all()) and not bool(gave_up.any()) and float(force.max()) < 0.01
    _, gave_up0, _ = run(None)
    assert int(gave_up0.sum()) > 0  # the energy test alone: the noise ends some rows by giving up


def _endpoint_cfg(tmp_path, n, force=True, **stream):
    cfg = _run_cfg(tmp_path, n, force)
    cfg.minima.polish = 'none'
    cfg.opt = [dict(cfg.opt[0], max_num_steps=8),
               dict(STAGE, optimizer_func='bfgs', max_num_steps=300, fmax=0.05, energy_noise=1e-4,
                    early_stop=[[2, 1000.0]], early_stop_kT=6.86)]
    for k, v in stream.items():
        setattr(cfg.stream, k, v)
    return cfg


def test_endpoint_config_checks():
    bfgs = dict(STAGE, optimizer_func='bfgs', fmax=0.05)
    ok = _cfg(opt=[dict(STAGE), bfgs], minima={'kT': 1.0, 'polish': 'none'})
    assert ss.check_stream_config(ok)['polish_chunk'] == ss.ENDPOINT_POLISH_CHUNK
    m = rs.check_minima_config(SimpleNamespace(**dict(vars(ok), minima={'kT': 1.0, 'polish': 'none'})))
    assert m['capture_A'] == 0.2 and m['merge_A'] == 0.2
    for over in [dict(opt=[dict(STAGE), dict(bfgs, fmax=None)], minima={'kT': 1.0, 'polish': 'none'}),  # no fmax
                 dict(opt=[dict(STAGE), dict(STAGE)], minima={'kT': 1.0, 'polish': 'none'}),  # not bfgs
                 dict(opt=[dict(STAGE, fmax=0.1), bfgs], minima={'kT': 1.0}),  # fmax on rprop
                 dict(opt=[dict(STAGE), dict(bfgs, fmax=-1.0)], minima={'kT': 1.0}),
                 dict(opt=[dict(STAGE), bfgs], stream={'slots': 4, 'trajectories': {'every': 2}}),  # no minima
                 dict(opt=[dict(STAGE), bfgs], minima={'kT': 1.0},
                      stream={'slots': 4, 'trajectories': {'evry': 2}})]:
        with pytest.raises(ValueError):
            ss.check_stream_config(_cfg(**over))
    with pytest.raises(ValueError):  # the metric is Z' = 1's
        ss.check_stream_config(SimpleNamespace(**dict(vars(ok), zp_to_search=[1, 2])))
    with pytest.raises(ValueError):  # the endpoint mode is the stream's
        rs.check_minima_config(SimpleNamespace(opt=[dict(STAGE, centroid_boundary='wrap')],
                                               minima={'kT': 1.0, 'polish': 'none'}))


def test_endpoint_stream_runs_every_start_once_and_resumes(mol, tmp_path, monkeypatch):
    n = 7
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None, num_samples=None: _placeholders(mol, n))
    out = rs.crystal_search(_endpoint_cfg(tmp_path, n))
    assert sorted(int(r.start_index) for r in out) == list(range(n))
    for r in out:
        assert int(r.polish_status) in (0, 2, 4, 5, 6) and 'hessian' not in r.keys() and 'id_hessian' not in r.keys()
        assert 'id_params' in r.keys() and math.isfinite(float(r.minimum_f))
    own = [r for r in out if int(r.minimum_id) >= 0 and not bool(r.recapture)]
    assert sorted(int(r.minimum_id) for r in own) == list(range(len(own))) and own
    with open(tmp_path / 'streamtest_stream_effort.json') as fh:
        eff = json.load(fh)
    assert eff['polish_evals'] < sum(eff['stage_evals']) / 4  # an energy per endpoint, a Hessian per new one
    again = rs.crystal_search(_endpoint_cfg(tmp_path, n, force=False))  # resume: the ball library rebuilds
    assert len(again) == n
    with pytest.raises(ValueError):  # a polished run cannot resume these endpoints
        mn.MinimaLibrary.from_rows(out, ball=False)


def test_endpoint_seeds_start_in_the_last_stage(mol, tmp_path, monkeypatch):
    n = 4
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None, num_samples=None: _placeholders(mol, n))
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    first = rs.crystal_search(_endpoint_cfg(tmp_path / 'a', n))
    torch.save(first, tmp_path / 'seeds.pt')
    out = rs.crystal_search(_endpoint_cfg(tmp_path / 'b', n, seeds=str(tmp_path / 'seeds.pt')))
    assert sorted(int(r.start_index) for r in out) == list(range(-n, n))
    for r in out:
        st = r.stage_steps.reshape(-1).tolist()
        if int(r.start_index) < 0:
            assert st[0] == 0 and st[1] >= 1  # straight into the bfgs stage
        else:
            assert st[0] >= 7
    order = [int(r.start_index) for r in out]
    assert max(i for i, k in enumerate(order) if k < 0) < max(i for i, k in enumerate(order) if k >= 0)


def test_trajectories_and_band_set(mol, tmp_path, monkeypatch):
    from mxtaltools.crystal_search.basin_set import read_basin_set
    n = 6

    def named(config, target=None, num_samples=None):  # a basin set's rows carry their molecule's identifier
        rows = _placeholders(mol, n)
        for r in rows:
            r.identifier = 'A'
        return rows

    monkeypatch.setattr(rs, 'init_samples_to_optim', named)
    cfg = _endpoint_cfg(tmp_path, n, basin_set=True, trajectories={'every': 2, 'band_kT': 1000.0, 'net_A': 0.05})
    out = rs.crystal_search(cfg)
    tr = torch.load(tmp_path / 'streamtest_traj.pt', weights_only=False)
    assert tr['format'] == 'stream_trajectories/1' and len(tr['f_kT']) > n
    starts = {int(r.start_index) for r in out}
    assert set(tr['start_index'].tolist()) <= starts and bool(torch.isfinite(tr['f_kT']).all())
    assert tr['grad_kT'].shape == tr['params'].shape and bool(torch.isfinite(tr['grad_kT']).all())
    per = {}
    for s_, st_, n_ in zip(tr['start_index'].tolist(), tr['stage'].tolist(), tr['step'].tolist()):
        per.setdefault((s_, st_), []).append(n_)
    assert all(sorted(v) == list(range(0, 2 * len(v), 2)) for (s_, st_), v in per.items() if st_ == 0)  # every 2nd step
    mid = {int(r.start_index): int(r.minimum_id) for r in out}
    assert all(int(m) == mid[int(s)] for s, m in zip(tr['start_index'], tr['minimum_id']))
    assert not (tmp_path / 'streamtest_traj_parts').exists()
    ds, header = read_basin_set(str(tmp_path / 'streamtest_basins'))  # endpoint minima: no Hessian column
    assert header['kind'] == 'minima' and 'H' not in ds.columns and header['identity']['mode'] == 'endpoint balls'
    band, bh = read_basin_set(str(tmp_path / 'streamtest_band'))
    assert bh['kind'] == 'band' and 'H' not in band.columns and len(band) >= 1
    assert {'group_id', 'start_index', 'stage', 'step'} <= set(band.columns)
    assert set(band.columns['start_index'].tolist()) <= starts
    assert bh['search']['band']['net'] == len(band) + sum(bh['counts']['refused'].values())


def test_the_endpoint_example_config_passes_the_checks():
    """configs/crystal_searches/minima_stream/mipcas_elj_balls.yaml, the example of the endpoint mode."""
    from pathlib import Path
    from mxtaltools.common.config_processing import dict2namespace, load_yaml
    path = Path(rs.__file__).resolve().parents[2] / 'configs' / 'crystal_searches' / 'minima_stream' / \
        'mipcas_elj_balls.yaml'
    cfg = dict2namespace(load_yaml(str(path)))
    s = ss.check_stream_config(cfg)
    m = rs.check_minima_config(cfg)
    assert m['polish'] == 'none' and m['capture_A'] == 0.2 and m['merge_A'] == 0.2
    assert s['polish_chunk'] == ss.ENDPOINT_POLISH_CHUNK and s['trajectories']['net_A'] == 0.1 and s['basin_set']
    assert cfg.opt[-1]['optimizer_func'] == 'bfgs' and cfg.opt[-1]['fmax'] == 0.02


def test_unconverged_endpoints_are_no_minima(mol, tmp_path, monkeypatch):
    """A bfgs stage cut at 3 steps (before BFGS_MIN_UPDATES updates can pass the force test): every endpoint leaves
    unconverged, is kept with status 2 or 4, and no minimum is made of it."""
    n = 4
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None, num_samples=None: _placeholders(mol, n))
    cfg = _endpoint_cfg(tmp_path, n)
    cfg.opt[1]['max_num_steps'] = 3
    out = rs.crystal_search(cfg)
    assert len(out) == n and all(int(r.minimum_id) == -1 and int(r.polish_status) in (2, 4) for r in out)
    assert 'stage_finished' not in out[0].keys()
    assert torch.load(tmp_path / 'streamtest_minima.pt', weights_only=False) == []


def test_trajectory_merge_keeps_one_attempt_per_start(tmp_path):
    """_merge_trajectories: a start relaxed twice keeps only the records of the attempt its output row names, wherever
    they were saved (its first attempt in part 0, the redo split over parts 0 and 1); a start of an earlier
    invocation's merged file survives a merge of new parts; unfinished starts are dropped."""
    def rec(start, step, f, att):
        return dict(start=start, c=0, hand=torch.ones(1), stage=1, step=step, att=att, g=torch.zeros(12, dtype=torch.float64),
                    x=torch.full((12,), float(f), dtype=torch.float64), f=f)
    rows = []
    for k, att in ((0, (5, 2)), (1, (5, 1)), (2, (6, 1))):
        r = SimpleNamespace(start_index=torch.tensor([k]), minimum_id=torch.tensor([k]),
                            traj_attempt=torch.tensor([att]))
        r.keys = lambda r=r: ['start_index', 'minimum_id', 'traj_attempt']
        rows.append(r)
    d = tmp_path / 'traj_parts'
    d.mkdir()
    torch.save([rec(0, 0, 1.0, (4, 9)), rec(0, 10, 2.0, (4, 9)), rec(1, 0, 3.0, (5, 1)), rec(0, 0, 1.5, (5, 2)),
                rec(9, 0, 9.0, (5, 3))], d / 'part_000000.pt')
    torch.save([rec(0, 20, 2.5, (5, 2))], d / 'part_000001.pt')
    path = str(tmp_path / 'run_traj.pt')
    ss._merge_trajectories(str(d), path, rows[:2], [('A', 2, 1)], {'every': 10}, 1.0)
    t = torch.load(path, weights_only=False)
    got = sorted(zip(t['start_index'].tolist(), t['step'].tolist(), t['f_kT'].tolist()))
    assert got == [(0, 0, 1.5), (0, 20, 2.5), (1, 0, 3.0)] and not d.exists()
    d.mkdir()
    torch.save([rec(2, 0, 4.0, (6, 1))], d / 'part_000000.pt')  # a later invocation: one new start
    ss._merge_trajectories(str(d), path, rows, [('A', 2, 1)], {'every': 10}, 1.0)
    t = torch.load(path, weights_only=False)
    assert sorted(t['start_index'].tolist()) == [0, 0, 1, 2] and t['attempt'].shape == (4, 2)


def test_a_new_candidate_at_a_saddle_is_descended(mol, tmp_path, monkeypatch):
    """Every new candidate takes one Hessian; one that is not positive semi-definite (here every one, candidate_is_minimum
    False) is finished by bfgs_polish and stored at the minimum it reaches (status 6), the saddle kept on the row.
    Repeats take no Hessian (their library entry was checked); without saddles saddle_f is NaN."""
    n = 4
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None, num_samples=None: _placeholders(mol, n))
    (tmp_path / 'a').mkdir()
    out = rs.crystal_search(_endpoint_cfg(tmp_path / 'a', n))
    assert all(math.isnan(float(r.saddle_f)) for r in out)  # no saddle here (a status 6 is a candidate short of tol)
    calls = []
    real = mn.bfgs_polish

    def spy(*a, **k):
        calls.append(len(a[1]))
        return real(*a, **k)

    monkeypatch.setattr(mn, 'candidate_is_minimum', lambda H: torch.zeros(len(H), dtype=torch.bool))
    monkeypatch.setattr(mn, 'bfgs_polish', spy)
    (tmp_path / 'b').mkdir()
    out = rs.crystal_search(_endpoint_cfg(tmp_path / 'b', n))
    own = [r for r in out if int(r.minimum_id) >= 0 and not bool(r.recapture)]
    assert own and all(int(r.polish_status) == 6 and math.isfinite(float(r.saddle_f)) for r in own)
    assert not (tmp_path / 'b' / 'streamtest_traj.pt').exists()  # no trajectories asked for: none written
    assert sum(calls) == sum(1 for r in out if math.isfinite(float(r.saddle_f)))  # one polish per saddle
    for r in own:  # the polish moved down from the endpoint, not up
        assert float(r.minimum_f) <= float(r.saddle_f) + 1e-6
    with open(tmp_path / 'b' / 'streamtest_stream_effort.json') as fh:
        assert json.load(fh)['polish_evals'] > n  # the Hessians and the polish are counted


def test_saddle_slides_join_the_trajectories(mol, tmp_path, monkeypatch):
    """With trajectories, a saddle candidate's descent in the finish step adds its accepted steps as states of stage
    n_st (here 2) of that start; every kept state's start names its own attempt."""
    n = 4
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None, num_samples=None: _placeholders(mol, n))
    monkeypatch.setattr(mn, 'candidate_is_minimum', lambda H: torch.zeros(len(H), dtype=torch.bool))
    out = rs.crystal_search(_endpoint_cfg(tmp_path, n, trajectories={'band_kT': 1000.0}))
    tr = torch.load(tmp_path / 'streamtest_traj.pt', weights_only=False)
    slid = {int(r.start_index) for r in out if int(r.polish_status) == 6}
    assert slid and set(tr['start_index'][tr['stage'] == 2].tolist()) <= slid
    assert int((tr['stage'] == 2).sum()) > 0
    att = {int(r.start_index): tuple(r.traj_attempt.reshape(-1).tolist()) for r in out}
    assert all(tuple(a) == att[int(s)] for s, a in zip(tr['start_index'], tr['attempt'].tolist()))

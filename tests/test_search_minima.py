"""crystal_search.minima and the capture rule of gradient_descent_optimization (cfg:minima of run_search).

Pins: StageEnergy's double-backward Hessian agrees with finite differences of its gradient; newton_polish ends on a
positive semi-definite Hessian with a small Newton decrement and never raises the energy; the library's d^2 is
dx^T H dx with centre lattice translations removed and keys kept apart; a row started inside a known minimum's
ellipsoid is captured at step 0 and leaves the batch while the others descend; no library means no capture_min;
finish_minima reports a captured row as the library minimum and merges a second polish of a known minimum;
two cell settings of one crystal are one minimum; the run_search config check refuses what the polish cannot follow;
truncate undoes a batch's adds.
CPU, eLJ, a handful of rows of the test molecule (as test_grad_convergence).
"""
import pytest
import torch

from mxtaltools.crystal_search import crystal_opt_utils as cou
from mxtaltools.crystal_search import minima as mn

from tests.test_fd_refold import _case, _rows, mol  # noqa: F401  (the fixture and row builders of the refold tests)

KT = 6.86  # eLJ units (the MIPCAS campaigns' kT)


@pytest.fixture(scope='module')
def relaxed(mol):
    """Four sg 2 rows of the test molecule after a short wrap-mode descent, and their polish."""
    torch.manual_seed(0)
    _, p, h = _case('normal', n=4, seed=31)
    b = _rows(mol, 2, p, h)
    out, _ = cou.gradient_descent_optimization(b.full_cell_parameters(), b, optimizer_func='rprop', init_lr=0.01,
                                               max_num_steps=120, optim_target='elj', cutoff=10,
                                               centroid_boundary='wrap', convergence_eps=1e-4)
    E = mn.StageEnergy(out, 'elj', kT=KT, lj_envelope=1.0)
    x = torch.stack([o.full_cell_parameters().reshape(-1) for o in out]).double()
    return out, E, x, mn.newton_polish(E, x, max_iter=60)


def test_hessian_matches_finite_differences(relaxed):
    _, E, x, _ = relaxed
    _, _, H = E.fgh(x[:1], torch.tensor([0]))
    h, d = 1e-3, x.shape[1]
    fd = torch.zeros(d, d, dtype=torch.float64)
    for k in range(d):
        e = torch.zeros(d, dtype=torch.float64)
        e[k] = h
        _, gp, _ = E.fgh(x[:1] + e, torch.tensor([0]))
        _, gm, _ = E.fgh(x[:1] - e, torch.tensor([0]))
        fd[k] = (gp[0] - gm[0]) / (2 * h)
    fd = 0.5 * (fd + fd.T)
    assert float((H[0] - fd).norm() / fd.norm()) < 0.05  # float32 energies: FD noise is a few percent


def test_polish_converges_downhill(relaxed):
    _, E, x, res = relaxed
    f0 = E.f(x, torch.arange(len(x)))
    assert bool((res['f'] <= f0 + 1e-6).all())
    assert res['status'].tolist() == [1, 1, 1, 1]
    assert bool(mn.is_psd(res['H']).all())
    lam, V = torch.linalg.eigh(res['H'])
    dec = 0.5 * (torch.einsum('nij,ni->nj', V, res['g']) ** 2 / lam.abs()).sum(1)
    assert bool((dec < 1e-4).all())


def _report(x, H):
    return dict(x=x, hand=torch.ones(1), H=H)


def test_library_d2_is_the_ellipsoid_with_centre_translations_removed():
    torch.manual_seed(1)
    A = torch.randn(12, 12, dtype=torch.float64)
    H = A @ A.T + torch.eye(12, dtype=torch.float64)
    x0 = torch.rand(12, dtype=torch.float64)
    lib = mn.MinimaLibrary()
    k = ('MOL', 2, 1, (1,))
    assert lib.add(x0, H, -10.0, k, _report(x0, H)) == 0
    dx = 0.01 * torch.randn(12, dtype=torch.float64)
    shifted = x0 + dx
    shifted[6:9] += torch.tensor([1.0, -2.0, 0.0], dtype=torch.float64)  # a lattice translation of the centre
    ids, d2 = lib.nearest(torch.stack([x0 + dx, shifted]), [k, k])
    assert ids.tolist() == [0, 0]
    assert torch.allclose(d2, (dx @ H @ dx).expand(2), rtol=1e-8)
    for other in [('MOL', 14, 1, (1,)), ('MOL', 2, 1, (-1,))]:  # another group or handedness: never the same minimum
        ids, d2 = lib.nearest((x0 + dx)[None], [other])
        assert ids.tolist() == [-1] and bool(torch.isinf(d2).all())


def test_truncate_undoes_adds():
    lib = mn.MinimaLibrary()
    k = ('MOL', 2, 1, (1,))
    for i in range(3):
        lib.add(torch.full((12,), float(i)), torch.eye(12), float(i), k, _report(torch.zeros(12), torch.eye(12)))
    lib.truncate(1)
    assert len(lib) == 1 and len(lib.report) == 1
    ids, _ = lib.nearest(torch.full((1, 12), 2.0, dtype=torch.float64), [k])
    assert ids.tolist() == [0]


@pytest.fixture(scope='module')
def finished(relaxed):
    out, _, _, _ = relaxed
    lib = mn.MinimaLibrary()
    rows, eff = mn.finish_minima([o.clone() for o in out], lib, 'elj', KT, max_iter=60)
    return lib, rows, eff


def test_finish_minima_rows_and_library(finished):
    lib, rows, eff = finished
    assert eff['n_h'] > 0 and len(lib) == sum(1 for r in rows if not bool(r.recapture) and int(r.minimum_id) >= 0)
    for r in rows:
        assert int(r.polish_status) == 1
        assert r.hessian.shape == (1, 12, 12) and r.id_hessian.shape == (1, 12, 12) and r.id_params.shape == (1, 12)
    rebuilt = mn.MinimaLibrary.from_rows(rows)
    assert len(rebuilt) == len(lib) and all(torch.allclose(a, b.float().double(), atol=1e-5)
                                           for a, b in zip(rebuilt.x, lib.x))


def test_restart_at_a_minimum_is_captured_at_step_zero(finished, mol):
    lib, rows, _ = finished
    b = _rows(mol, 2, torch.stack([r.full_cell_parameters().reshape(-1) for r in rows]),
              torch.stack([r.aunit_handedness.reshape(-1) for r in rows]))
    keys = mn.batch_keys(b)
    assert keys == [mn.row_key(r) for r in rows]
    o2, rec = cou.gradient_descent_optimization(b.full_cell_parameters(), b, optimizer_func='rprop', init_lr=0.01,
                                                max_num_steps=20, optim_target='elj', cutoff=10,
                                                centroid_boundary='wrap', convergence_eps=-1,
                                                capture_library=lib, capture_keys=keys, capture_every=1)
    assert [int(o.capture_min) for o in o2] == [int(r.minimum_id) for r in rows]
    assert bool(torch.isinf(rec['loss'][1:]).all())  # every row left the batch after step 0


def test_captured_and_repeated_rows_are_recaptures(finished, relaxed):
    lib, first, _ = finished
    out, _, _, _ = relaxed
    n_lib = len(lib)
    again = [o.clone() for o in out]  # the same descents again: every converged row is a repeat
    for o in again:
        o.capture_min = torch.tensor([-1])
    again[0].capture_min = torch.tensor([int(first[0].minimum_id)])  # and row 0 as if captured
    second, _ = mn.finish_minima(again, lib, 'elj', KT, max_iter=60)
    assert len(lib) == n_lib
    assert all(bool(r.recapture) for r in second)
    assert [int(r.minimum_id) for r in second] == [int(r.minimum_id) for r in first]
    m0 = int(first[0].minimum_id)
    assert torch.allclose(second[0].full_cell_parameters().reshape(-1).double(), lib.report[m0]['x'], atol=1e-5)
    assert int(second[0].polish_status) == 0


def _sheared(row):
    """The same crystal with its c axis replaced by c + a (or c - a): new lengths and angles, centres re-expressed,
    Cartesian frame (a along x, b in the xy plane) and so the pose unchanged."""
    T = row.T_fc[0].double()  # columns: a, b, c
    f = row.aunit_centroid.reshape(-1)[:3].double()
    for s in (1.0, -1.0):
        N = torch.tensor([[1.0, 0.0, s], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        a, b, c = (T @ N).T
        ang = lambda u, v: torch.arccos(torch.dot(u, v) / (u.norm() * v.norm()))
        angles = torch.stack([ang(b, c), ang(a, c), ang(a, b)])
        if bool(((angles > 0.25 * torch.pi) & (angles < 0.75 * torch.pi)).all()):
            fc = torch.linalg.solve(N, f)
            fc = fc - torch.floor(fc)
            x = torch.cat([torch.stack([a.norm(), b.norm(), c.norm()]), angles, fc,
                           row.aunit_orientation.reshape(-1)[:3].double()])
            return x
    return None


def test_two_cell_settings_of_one_crystal_are_one_minimum(finished, mol):
    _, rows, _ = finished
    lib = mn.MinimaLibrary()
    r = rows[0]
    x2 = _sheared(r)
    assert x2 is not None
    xs = torch.stack([r.full_cell_parameters().reshape(-1).double(), x2])
    b = _rows(mol, 2, xs, r.aunit_handedness.reshape(-1).repeat(2))
    E = mn.StageEnergy(b.batch_to_list(), 'elj', kT=KT)
    e = E.f(xs, torch.arange(2))
    assert abs(float(e[0] - e[1])) < 1e-3, 'the sheared description must be the same crystal'
    out, _ = mn.finish_minima(b.batch_to_list(), lib, 'elj', KT, max_iter=60)
    assert len(lib) == 1
    assert [int(o.minimum_id) for o in out] == [0, 0]
    assert sorted(bool(o.recapture) for o in out) == [False, True]  # the lower polished energy is added first


def test_config_check_refuses_what_the_polish_cannot_follow():
    from types import SimpleNamespace
    from mxtaltools.crystal_search.run_search import check_minima_config
    stage = dict(optim_target='elj', centroid_boundary='wrap')
    assert check_minima_config(SimpleNamespace(opt=[stage])) is None
    ok = check_minima_config(SimpleNamespace(opt=[stage], minima={'kT': KT}))
    assert ok['capture_d2'] == 9.0 and ok['merge_d2'] == 1.0
    for bad_stage, block in [(dict(stage, centroid_boundary='clamp'), {'kT': KT}),
                             (dict(stage, enforce_reduced=True), {'kT': KT}),
                             (dict(stage, fd_refold=True), {'kT': KT}),
                             (dict(stage, optim_target='rdf_dist'), {'kT': KT}),
                             (stage, {}), (stage, {'kT': -1.0}), (stage, {'kT': KT, 'capture': 9})]:
        with pytest.raises(ValueError):
            check_minima_config(SimpleNamespace(opt=[bad_stage], minima=block))

"""crystal_search.minima and the capture rule of gradient_descent_optimization (cfg:minima of run_search).

Pins: StageEnergy's double-backward Hessian agrees with finite differences of its gradient; newton_polish ends on a
positive semi-definite Hessian with a small Newton decrement (flat directions left out) and never raises the energy;
the library's d^2 is dx^T H dx with centre lattice translations and the group's origin shifts removed (half a cell
in P-1, none in P1) and keys kept apart; a minimum in another cell setting merges by the glued FD distance; a row
started inside a known minimum's ellipsoid is captured at step 0 and leaves the batch while the others descend; no
library means no capture_min; finish_minima reports a captured row as the library minimum and merges a second polish
of a known minimum; a row converged in the polish captures the rest of its group; two cell settings of one crystal
are one minimum; the run_search config check refuses what the polish cannot follow; truncate undoes a batch's adds.
CPU, eLJ, a handful of rows of the test molecule (as test_grad_convergence).
"""
import itertools

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



def test_finite_difference_hessian_matches_the_double_backward(relaxed):
    """StageEnergy(hessian='fd'), the MLIP targets' default, agrees with the batched double backward on eLJ."""
    out, _, x, _ = relaxed
    a = mn.StageEnergy(out, 'elj', kT=KT, lj_envelope=1.0)
    b = mn.StageEnergy(out, 'elj', kT=KT, lj_envelope=1.0, hessian='fd')
    assert mn.StageEnergy(out, 'mace', kT=1.0).hessian == 'fd' and a.hessian == 'autograd'
    _, _, Ha = a.fgh(x, torch.arange(len(x)))
    Fb, Gb, Hb = b.fgh(x, torch.arange(len(x)))
    assert torch.allclose(Hb, Ha, rtol=2e-2, atol=2e-2 * float(Ha.abs().max()))


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



class _Quadratic:
    """E/kT = 0.5 x^T A x + b^T x, as StageEnergy's f, fg, fgh (rows independent)."""

    def __init__(self, A, b):
        self.A, self.b = A, b

    def f(self, x, idx):
        return 0.5 * torch.einsum('ni,ij,nj->n', x, self.A, x) + x @ self.b

    def fg(self, x, idx):
        return self.f(x, idx), x @ self.A + self.b

    def fgh(self, x, idx):
        return self.f(x, idx), x @ self.A + self.b, self.A.expand(len(x), -1, -1).clone()


def test_the_polish_converges_where_the_energy_is_flat_along_some_directions():
    """Two directions of exactly zero curvature (a crystal system's fixed angle, a layer's spacing across a vacuum gap):
    a gradient component of 1e-6 along one must not hold the decrement up (it used to, 1e-12 / 1e-12, so such rows
    ended stalled or capped); a real slope (0.1) along it is not converged."""
    A = torch.diag(torch.tensor([10.0, 3.0, 1.0, 0.0, 5.0, 0.0, 2.0, 7.0, 4.0, 1.5, 6.0, 9.0], dtype=torch.float64))
    for slope, converged in ((1e-6, True), (0.1, False)):
        b = torch.zeros(12, dtype=torch.float64)
        b[3] = slope
        res = mn.newton_polish(_Quadratic(A, b), torch.zeros(2, 12, dtype=torch.float64) + 0.01, max_iter=10)
        assert (res['status'] == 1).all().item() == converged


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



def test_centre_shifts_are_the_groups_equivalent_origins():
    half = [[float(v) for v in t] for t in itertools.product((0.0, 0.5), repeat=3)]
    for sg in (2, 14, 15, 19):  # P-1, P21/c, C2/c (centring and normalizer together), P212121
        assert mn.centre_shifts(sg).tolist() == half
    assert mn.centre_shifts(1).tolist() == [[0.0, 0.0, 0.0]]  # P1: the origin moves continuously, no discrete shift
    assert mn.centre_shifts(4).tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5]]


def test_a_half_cell_origin_shift_is_the_same_minimum_in_p_1_but_not_in_p1():
    """A P-1 crystal with its centre moved half a cell is the same crystal about another inversion centre (one stored
    minimum had been counted twice this way, 2026-10-29); in P1 the same move is another crystal."""
    torch.manual_seed(2)
    A = torch.randn(12, 12, dtype=torch.float64)
    H = A @ A.T + torch.eye(12, dtype=torch.float64)
    W = mn.ellipsoid_factor(H)
    x0 = torch.rand(12, dtype=torch.float64)
    dx = 0.01 * torch.randn(12, dtype=torch.float64)
    x = x0 + dx
    x[6:9] += torch.tensor([0.5, 0.0, -0.5], dtype=torch.float64)
    for sg, same in ((2, True), (1, False)):
        lib = mn.MinimaLibrary()
        k = ('MOL', sg, 1, (1,))
        lib.add(x0, H, -10.0, k, _report(x0, H))
        ids, d2 = lib.nearest(x[None], [k])
        assert (int(ids[0]) == 0) and (bool(torch.isclose(d2[0], dx @ H @ dx, rtol=1e-8)) == same)
        assert lib.match(x, W, k, merge_d2=1.0) == (0 if same else -1)



def test_a_minimum_in_another_cell_setting_merges_by_the_glued_fd_distance():
    """One P2_1/c crystal in two cell settings (c and c + 2a, a basis change of its setting group): far apart in the
    parameters, so the ellipsoids miss, but the glued FD distance is zero, so match merges them -- inside the energy
    window, with the molecule's r_g, and only with fd_merge on."""
    import math
    import numpy as np
    from mxtaltools.crystal_fd import ops
    from mxtaltools.crystal_fd.description import CrystalDescription
    from mxtaltools.crystal_fd.metric import canonical
    from mxtaltools.crystal_search.standardize import setting_shift
    F64 = torch.float64
    sg = 14
    p = torch.tensor([[5.0, 7.0, 10.0, math.pi / 2, math.radians(100.0), math.pi / 2, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1]],
                     dtype=F64)
    can = canonical(sg, p, torch.ones(1))
    d1 = CrystalDescription.from_parts(sg, can['cell'], can['t'], can['R'], can['h'])
    N = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]])
    shift = setting_shift(N, sg)
    assert shift is not None  # c -> c + 2a keeps P2_1/c's c glide
    d2 = ops.change_basis(d1, torch.as_tensor(N, dtype=F64)[None], torch.as_tensor(shift, dtype=F64)[None])
    params = lambda d: torch.cat([torch.as_tensor(v).double().reshape(1, -1) for v in d.crystal_params()[:4]], 1)[0]
    x1, x2 = params(d1), params(d2)
    assert float((x1 - x2).abs().max()) > 1.0  # the same crystal, far apart in the parameters
    H = 100.0 * torch.eye(12, dtype=F64)
    W = mn.ellipsoid_factor(H)
    key = ('MOL', sg, 1, (int(can['h'][0]),))
    lib = mn.MinimaLibrary()
    lib.add(x1, H, -10.0, key, {})
    assert lib.match(x2, W, key, 1.0) == -1  # no energy or r_g: the ellipsoids alone
    assert lib.match(x2, W, key, 1.0, f=-10.0, rg=2.0) == 0
    assert lib.match(x2, W, key, 1.0, f=-10.0 + 2 * mn.FD_MERGE_KT, rg=2.0) == -1  # outside the energy window
    off = mn.MinimaLibrary(fd_merge=None)
    off.add(x1, H, -10.0, key, {})
    assert off.match(x2, W, key, 1.0, f=-10.0, rg=2.0) == -1


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


def test_the_polish_captures_rows_inside_a_known_minimum(finished, relaxed):
    """The polished minima finished again with capture_d2: each is at its known minimum before the polish's first
    evaluation, so it is captured there (status 0, no Hessian taken) as that minimum."""
    lib, first, _ = finished
    out, E, x, _ = relaxed
    n_lib = len(lib)
    second, eff = mn.finish_minima([r.clone() for r in first], lib, 'elj', KT, max_iter=60, capture_d2=9.0)
    assert len(lib) == n_lib and eff['n_h'] == 0
    assert [int(r.polish_status) for r in second] == [0] * len(first)
    assert [int(r.minimum_id) for r in second] == [int(r.minimum_id) for r in first]
    # newton_polish's own rule: a row the callable reports inside stops with status 0 and is not evaluated
    res = mn.newton_polish(E, x, max_iter=60, capture_d2=9.0,
                           capture=lambda xc, ids: (torch.where(ids == 0, 0, -1), torch.where(ids == 0, 0.0, 1e9)))
    assert int(res['status'][0]) == 0 and int(res['capture'][0]) == 0 and int(res['n_h'][0]) == 0
    assert res['status'][1:].tolist() == [1] * (len(x) - 1)



def test_rows_converged_in_the_polish_capture_the_rest_of_their_group(finished, relaxed):
    """A group polished against an empty library: a row already at its minimum converges at once and becomes a
    capture target, and a row of the same group descending into that minimum is captured there (status 0) and
    reported as that minimum, instead of being polished to the end and merged."""
    _, first, _ = finished
    out, _, _, _ = relaxed
    at_min = first[0].clone()
    for k in set(at_min.keys()) - set(out[0].keys()):
        delattr(at_min, k)
    lib = mn.MinimaLibrary()
    rows, eff = mn.finish_minima([at_min, out[0].clone()], lib, 'elj', KT, max_iter=60, capture_d2=9.0)
    assert len(lib) == 1
    assert [int(r.polish_status) for r in rows] == [1, 0]
    assert [int(r.minimum_id) for r in rows] == [0, 0] and bool(rows[1].recapture)
    assert torch.equal(rows[1].id_params, rows[0].id_params)


class _Row:
    """A stand-in output row: tensors as attributes, ``keys`` and ``clone`` as on crystal rows."""

    def __init__(self, **kw):
        self.__dict__.update({k: torch.tensor([v]) for k, v in kw.items()})

    def keys(self):
        return list(self.__dict__)

    def clone(self):
        r = _Row()
        r.__dict__.update({k: v.clone() for k, v in self.__dict__.items()})
        return r


def test_missing_mass_and_minimum_rows():
    lib = mn.MinimaLibrary()
    for f in (0.0, 1.0, 10.0):  # E/kT: minima 0 and 1 are within 5 kT of the lowest, minimum 2 is not
        lib.add(torch.zeros(12), torch.eye(12), f, ('m', 2, 1, (1,)), dict(x=torch.zeros(12)))
    rows = [_Row(start_index=0, minimum_id=0, recapture=False), _Row(start_index=1, minimum_id=0, recapture=True),
            _Row(start_index=2, minimum_id=1, recapture=False), _Row(start_index=3, minimum_id=2, recapture=False),
            _Row(start_index=4, minimum_id=-1, recapture=False),  # a random start that did not converge
            _Row(start_index=-1, minimum_id=1, recapture=True)]  # a seed: not in the sample
    mm = mn.missing_mass(rows, lib, band_kT=5.0)
    from mxtaltools.crystal_search.coordinator import poisson_upper
    assert (mm['n'], mm['f1'], mm['band_minima']) == (5, 1, 2)  # minimum 1 is the band's one singleton
    assert mm['p'] == pytest.approx(1 / 5) and mm['p_hi'] == pytest.approx(poisson_upper(1) / 5)
    seeds = mn.missing_mass(rows, lib, band_kT=5.0, seeds=True)  # the seed's row alone: minimum 1 once
    assert (seeds['n'], seeds['f1']) == (1, 1)
    table = mn.minimum_rows(rows)
    assert [int(r.minimum_id) for r in table] == [0, 1, 2]
    assert [int(r.hits) for r in table] == [2, 1, 1] and [int(r.seed_hits) for r in table] == [0, 1, 0]


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


def _skewed(row):
    """The same crystal with c replaced by c + n a (n = 1..6), the first whose angles leave the cleaning clamp
    (some angle above 0.8 pi or below 0.2 pi), centres re-expressed, pose unchanged."""
    T = row.T_fc[0].double()
    f = row.aunit_centroid.reshape(-1)[:3].double()
    for n_ in range(1, 7):
        for s in (n_, -n_):
            N = torch.tensor([[1.0, 0.0, float(s)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
            a, b, c = (T @ N).T
            ang = lambda u, v: torch.arccos(torch.dot(u, v) / (u.norm() * v.norm()))
            angles = torch.stack([ang(b, c), ang(a, c), ang(a, b)])
            if bool(((angles > 0.8 * torch.pi) | (angles < 0.2 * torch.pi)).any()):
                fc = torch.linalg.solve(N, f)
                return torch.cat([torch.stack([a.norm(), b.norm(), c.norm()]), angles, fc - torch.floor(fc),
                                  row.aunit_orientation.reshape(-1)[:3].double()])
    return None


def test_a_row_past_the_angle_clamp_polishes_to_its_own_minimum(finished, mol):
    lib, rows, _ = finished
    r = rows[0]
    x = _skewed(r)
    assert x is not None
    b = _rows(mol, 2, x[None], r.aunit_handedness.reshape(-1))
    with torch.no_grad():  # the skewed description, built without cleaning, is the same crystal
        e_skew = float(b.clone().analyze(computes=['elj'], cutoff=10, return_cluster=False)['elj'])
    assert abs(e_skew - float(r.elj)) < 1e-2 * abs(float(r.elj))
    n_lib = len(lib)
    out, _ = mn.finish_minima(b.batch_to_list(), lib, 'elj', KT, max_iter=60)
    assert len(lib) == n_lib  # nothing new: it is a known minimum
    assert int(out[0].polish_status) == 1 and bool(out[0].recapture)
    assert int(out[0].minimum_id) == int(r.minimum_id)

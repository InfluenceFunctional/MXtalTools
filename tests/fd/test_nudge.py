"""crystal_fd.canon: ``canonicalize(..., nudge=delta)``, moving near-face rows off the face (owner decision 2026-10-21).

Rows are placed about 1e-6 from a face (and some on it) in the units of ``near_face``: cell-chart coordinates for the
cell (``canon._cell_chart``), fractional for the centroid.

1. Triclinic cell faces (r1, r2, v; r1 = 1 exactly, tied candidates), monoclinic sg 14 faces (s = 0 and s = 1 in the
   window where the row is canonical, not a special metric) and the sg 4 FREE a <= c face: nudged; t and R kept;
   every chart face at least delta away; near_face 0; the move is delta less the start distance (a FREE face:
   divided by sqrt 2); the nudged row is its own canonical form. Faces whose rows are special metrics (sg 14 at
   1e-6, triclinic s_beta = 0) are left bit for bit as nudge=None leaves them.
2. Centroid faces of ASU(Gamma+) in sg 2, sg 14 (a face, and an edge on a 2-fold axis of Gamma+, where the fold chose
   the pose by its orientation rule) and sg 19 (canonical since 2026-10-21, pose_only before): nudged; cell and R
   kept; slack at least delta.
3. Float32 re-descriptions (a random Gamma+ element and lattice shift, sg 1-15 also a random near-reduced basis; cast
   to float32) of every nudged row canonicalise back to it within 1e-5 with near_face 0; the unnudged rows on a face,
   re-described the same way, do land on another row (the check has power).
4. Rows nudge does not move are bit-identical to nudge=None (a batch mixing every status); NEAR_FOLD_BAND alone is
   not nudged; nudging a nudged row changes nothing; fd_status takes nudge.
5. Refusals keep the unmoved row, record no move and say why: a domain thinner than 2 delta (cell chart and ASU), and
   a nudged row that fails the fixed-point check. ``_min_norm_push`` against hand-solved cases; bad arguments raise.
   Sheets -1 and None are kept. sg >= 16 rows at special positions nudge to one crystal from float64, float32 and
   angles 1e-7 rad off (the orientation rule re-decided on the exact metric); since the cell step snaps sg >= 16
   cells to the exact metric (2026-10-21), canonicalize without the nudge lands on one pose too, while a fold of
   the unsnapped cells does not.
   The nudge's cell faces are those near_face measures (sg 16-74: ``axis_order.face_values``; ``_cell_face_distance``
   takes sg 1-15 only). Orthorhombic cell faces (the axis order in log lengths, Pbca's seams):
   tests/fd/test_higher_sg_cell.py.
6. The limits README states, pinned: an on-face row's re-descriptions nudge to more than one crystal; the near_tol
   edge; a c/a = 40 triclinic cell outruns the chart-unit step in float32.
"""
import math

import numpy as np
import pytest
import torch

from mxtaltools.common.sym_utils import MONO_CLASS
from mxtaltools.crystal_fd import axis_order, canon, domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table
from tests.fd import _helpers as H
from tests.fd import test_canon as TC

F64 = torch.float64
DELTA = 1e-4
EPS = 1e-6
FIELDS = ('N', 's', 'n_auto', 'n_tied', 'violation', 'on_sheet', 'sheet_moved', 'element', 'lattice', 'free_shift',
          'slack', 'tie', 'near_face', 'moved', 'status')
TRI_Y = (math.log(9.0), 0.8, 0.85, -0.3, -0.4, 0.5)          # an interior triclinic chart point
MONO_Y = (math.log(9.0), math.log(6.0), math.log(7.0), 0.4)   # an interior monoclinic chart point (log c, log a, log b, s)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cell(sg, **set_y):
    """[1, 6] cell at the interior chart point with some chart coordinates replaced (by column index)."""
    chart = canon._cell_chart(sg)
    y = torch.tensor([TRI_Y if sg <= 2 else MONO_Y], dtype=F64)
    for k, v in set_y.items():
        y[0, int(k[1:])] = v
    return chart.map(y)[0]


def _row(sg, cell, t, gen):
    return CrystalDescription.from_parts(sg, cell, torch.tensor([t], dtype=F64), so3.random_rotations(1, gen))


def _chart_faces(sg, cell):
    """[b, k] face values of reduced cells in the chart of ``near_face``."""
    chart = canon._cell_chart(sg)
    A, o = canon._cell_faces(chart)
    return chart.unmap(cell)[0] @ A.T + o


def _slack(d):
    """[B] slack of each row's centroid in ASU(Gamma+), free coordinates at the gauge."""
    out = []
    for i in range(d.num_rows):
        sec = domains.translation_section(int(d.sg[i]))
        out.append(sec.region.slack(torch.where(sec.free, sec.gauge, d.t[i, 0])))
    return torch.stack(out)


def _bits_equal(a, b):
    """Same dtype, shape and bits (NaN payloads and the sign of zero included)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.is_floating_point():
        view = {torch.float64: torch.int64, torch.float32: torch.int32}[a.dtype]
        return torch.equal(a.contiguous().view(view), b.contiguous().view(view))
    return torch.equal(a, b)


def _assert_same(a, b, rows=None):
    """Every field of two Canonicalizations equal bit for bit, dtypes included (on ``rows``)."""
    rows = torch.arange(a.desc.num_rows) if rows is None else rows
    for f in ('cell', 't', 'R', 'h', 'sg', 'z_prime', 'unit_mask'):
        assert _bits_equal(getattr(a.desc, f)[rows], getattr(b.desc, f)[rows]), f
    for f in FIELDS:
        assert _bits_equal(getattr(a, f)[rows], getattr(b, f)[rows]), f
    assert [a.message[i] for i in rows.tolist()] == [b.message[i] for i in rows.tolist()]


def test_bits_equal_sees_dtype_and_signed_zero():
    x = torch.tensor([0.0, 1.0, float('nan')], dtype=F64)
    assert _bits_equal(x, x.clone()) and TC._nan_equal(x, x.float())          # the old comparison ignored dtype
    assert not _bits_equal(x, x.float())
    zero, neg = torch.tensor([0.0], dtype=F64), torch.tensor([-0.0], dtype=F64)
    assert TC._nan_equal(zero, neg) and not _bits_equal(zero, neg)
    i8 = torch.zeros(2, dtype=torch.int8)
    assert TC._nan_equal(i8, i8.long()) and not _bits_equal(i8, i8.long())


def _mono_change(sg, rng):
    """[1, 3, 3] long: one ac change the class of ``sg`` allows (a shear by -1, 0 or 1, or a sign change)."""
    cls = torch.tensor(int(MONO_CLASS[sg]))
    for _ in range(1000):                                   # bounded: at least a third of the draws are allowed
        kind, k = int(rng.integers(3)), int(rng.integers(-1, 2))
        E = (np.array([[1, 0], [k, 1]]), np.array([[1, k], [0, 1]]), np.diag(rng.choice([-1, 1], 2)))[kind]
        if bool(ops.mono_parity(torch.as_tensor(E), cls)):
            N = np.zeros((1, 3, 3), dtype=np.int64)
            N[0, 0, 0], N[0, 0, 2], N[0, 2, 0], N[0, 2, 2] = E[0, 0], E[0, 1], E[1, 0], E[1, 1]
            N[0, 1, 1] = E[0, 0] * E[1, 1] - E[0, 1] * E[1, 0]
            return torch.as_tensor(N)
    raise RuntimeError('no allowed change in 1000 draws')


def _redescribe32(d, rng):
    """Each row at a random Gamma+ element and lattice shift (and a free-axis shift), sg 1-15 also in a random
    near-reduced basis (triclinic: entries in [-1, 1]; monoclinic: one allowed change with its origin shift); float32."""
    parts = []
    for i in range(d.num_rows):
        r = d[i:i + 1]
        sg = int(r.sg[0])
        tab = group_table(sg)
        k = int(tab.gamma_plus[int(rng.integers(len(tab.gamma_plus)))])
        w = tab.gamma_w[k].to(F64) + torch.as_tensor(rng.integers(-1, 2, 3), dtype=F64) \
            + torch.as_tensor(rng.random(3), dtype=F64) * tab.free.to(F64)
        r = ops.apply(r, tab.gamma_W[k].to(F64), w)
        if sg <= 2:
            r = ops.change_basis(r, H.random_unimodular(1, rng, span=1), torch.zeros(1, 3, dtype=F64))
        elif sg <= 15:
            r = ops.change_basis(r, _mono_change(sg, rng))
        parts.append(r)
    return CrystalDescription.cat(parts).to(dtype=torch.float32)


def _deviation(a, b):
    """[B] largest difference of unit 0: cell lengths relative, angles, t, R entries."""
    return torch.stack([((a.cell[:, :3] - b.cell[:, :3]).abs() / b.cell[:, :3]).amax(1),
                        (a.cell[:, 3:] - b.cell[:, 3:]).abs().amax(1), (a.t[:, 0] - b.t[:, 0]).abs().amax(1),
                        (a.R[:, 0] - b.R[:, 0]).abs().amax((1, 2))], 1).amax(1)


def _cell_cases(gen):
    """(batch, expected status, nudged?, start distance or None, free-face?) of rows near cell-chart faces."""
    cases = [
        (2, dict(y2=1 - EPS), 'canonical', True, EPS, False),       # r1 = 1 - eps (a = b)
        (2, dict(y1=1 - EPS), 'canonical', True, EPS, False),       # r2 = 1 - eps (b = c)
        (2, dict(y5=EPS), 'canonical', True, EPS, False),           # v = eps
        (2, dict(y5=1 - EPS), 'canonical', True, EPS, False),       # v = 1 - eps
        (2, dict(y2=1.0), 'canonical', True, None, False),          # r1 = 1: tied candidates
        (2, dict(y3=-EPS), 'special_metric', False, None, False),   # s_beta = 0: a lattice mirror within face_tol
        (14, dict(y3=8e-6), 'canonical', True, 8e-6, False),        # beta = 90 deg side, outside face_tol's band
        (14, dict(y3=1 - 8e-6), 'canonical', True, 8e-6, False),    # the class bound on -cos beta
        (14, dict(y3=EPS), 'special_metric', False, None, False),   # within face_tol: a special metric
        (4, dict(y1=MONO_Y[0] - 8e-6), 'canonical', True, 8e-6, True),   # FREE: a = c (1 - 8e-6)
    ]
    rows = [_row(sg, _cell(sg, **y), [0.21, 0.13, 0.17], gen) for sg, y, *_ in cases]
    return CrystalDescription.cat(rows), cases


def _centroid_cases(gen):
    cell2, cell14 = _cell(2), _cell(14)
    cell19 = torch.tensor([[5.0, 7.0, 9.0, math.pi / 2, math.pi / 2, math.pi / 2]], dtype=F64)
    cases = [
        (2, cell2, [EPS, 0.2, 0.3], 'canonical', EPS),
        (2, cell2, [0.0, 0.2, 0.3], 'canonical', 0.0),
        (2, cell2, [0.0, 0.0, 0.3], 'canonical', 0.0),               # an edge: two faces
        (14, cell14, [0.2, 0.3, EPS], 'canonical', EPS),
        (14, cell14, [0.2, 0.3, 0.0], 'canonical', 0.0),
        (14, cell14, [0.0, 0.3, 0.0], 'canonical', 0.0),             # on a 2-fold axis of Gamma+: two poses tie
        (19, cell19, [0.1, EPS, 0.3], 'canonical', EPS),               # pose_only until 2026-10-21
        (19, cell19, [0.1, 0.0, 0.3], 'canonical', 0.0),
    ]
    return CrystalDescription.cat([_row(sg, c, t, gen) for sg, c, t, *_ in cases]), cases


# ---------------------------------------------------------------------------
# 1, 2: cell and centroid faces
# ---------------------------------------------------------------------------

def test_cell_faces_are_nudged():
    d, cases = _cell_cases(torch.Generator().manual_seed(1))
    c0 = canon.canonicalize(d)
    c = canon.canonicalize(d, nudge=DELTA)
    assert c0.names() == [case[2] for case in cases]
    assert bool((c0.near_face & canon.NEAR_CELL).all())
    assert int(c0.n_tied[4]) > 1                                          # the r1 = 1 row ties
    want = torch.tensor([case[3] for case in cases])
    assert torch.equal(c.nudged, want) and c.names() == c0.names()
    assert all(m == '' for m in c.nudge_message)
    _assert_same(c, c0, torch.nonzero(~want)[:, 0])                       # special metrics: bit for bit
    i = torch.nonzero(want)[:, 0]
    x, x0 = c.desc[i], c0.desc[i]
    assert torch.equal(x.t, x0.t) and torch.equal(x.R, x0.R) and torch.equal(x.h, x0.h)
    assert bool((c.near_face[i] == 0).all()) and bool((c.n_tied[i] >= 1).all())
    assert bool((c.nudge_t[i] == 0).all()) and bool(c.moved[i].all())
    for j, k in enumerate(i.tolist()):
        sg = int(d.sg[k])
        assert float(_chart_faces(sg, x.cell[j:j + 1]).min()) >= DELTA * (1 - 1e-9)
        start, free = cases[k][4], cases[k][5]
        if start is not None:                                             # one face: the move is delta less the start
            want_move = (DELTA - start) / (math.sqrt(2) if free else 1.0)
            assert abs(float(c.nudge_cell[k]) - want_move) <= 1e-12
        else:
            assert DELTA * (1 - 1e-9) <= float(c.nudge_cell[k]) <= DELTA * math.sqrt(8)
    again = canon.canonicalize(x)                                         # a fixed point
    assert again.names() == [c.names()[k] for k in i.tolist()]
    assert float(_deviation(again.desc, x).max()) <= 1e-12 and not bool(again.moved.any())
    assert bool((again.near_face == 0).all()) and bool((again.n_tied == 1).all())


def test_centroid_faces_are_nudged():
    d, cases = _centroid_cases(torch.Generator().manual_seed(2))
    c0 = canon.canonicalize(d)
    c = canon.canonicalize(d, nudge=DELTA)
    assert c0.names() == [case[3] for case in cases] and c.names() == c0.names()
    assert bool((c0.near_face & canon.NEAR_CENTROID).all())
    assert int(c0.tie[5]) == 2                                            # the 2-fold axis: the orientation rule chose
    assert bool(c.nudged.all()) and all(m == '' for m in c.nudge_message)
    assert torch.equal(c.desc.cell, c0.desc.cell) and torch.equal(c.desc.R, c0.desc.R)
    assert bool((c.near_face == 0).all()) and bool((c.nudge_cell == 0).all())
    assert float(_slack(c.desc).min()) >= DELTA * (1 - 1e-9)
    for k, case in enumerate(cases):
        start = case[4]
        if start > 0:
            assert abs(float(c.nudge_t[k]) - (DELTA - start)) <= 1e-12
        else:
            assert DELTA * (1 - 1e-9) <= float(c.nudge_t[k]) <= DELTA * (1 + 1e-9)
    again = canon.canonicalize(c.desc)
    assert again.names() == c.names() and float(_deviation(again.desc, c.desc).max()) <= 1e-12
    assert bool((again.near_face == 0).all()) and bool((again.tie == 0).all())


@pytest.mark.parametrize('sheet', [-1, None])
def test_nudge_keeps_the_callers_sheet(sheet):
    """Cell and centroid rows (half of them at h = -1 for sheet=None) nudged on sheet -1 or None: every eligible row
    is moved, keeps the handedness canonicalize gives it, and is a fixed point of canonicalize on the same sheet."""
    g = torch.Generator().manual_seed(10)
    d = CrystalDescription.cat([_cell_cases(g)[0], _centroid_cases(g)[0]])
    if sheet is None:
        h = d.h.clone()
        h[::2] = -1
        d = d.replace(h=h)
    c0 = canon.canonicalize(d, sheet=sheet)
    c = canon.canonicalize(d, sheet=sheet, nudge=DELTA)
    eligible = torch.tensor([n in ('canonical', 'pose_only') for n in c0.names()]) & \
        ((c0.near_face.long() & (canon.NEAR_CELL | canon.NEAR_CENTROID)) != 0)
    assert int(eligible.sum()) >= 12 and bool((c0.desc.h[eligible, 0] == -1).sum() >= 6)
    assert torch.equal(c.nudged, eligible) and all(m == '' for m in c.nudge_message)
    assert torch.equal(c.desc.h, c0.desc.h) and c.names() == c0.names()
    x = c.desc[eligible]
    again = canon.canonicalize(x, sheet=sheet)
    assert again.names() == [n for n, e in zip(c.names(), eligible.tolist()) if e]
    assert float(_deviation(again.desc, x).max()) <= 1e-12 and torch.equal(again.desc.h, x.h)
    assert bool((again.near_face == 0).all())


def test_special_positions_nudge_to_one_crystal_from_rounded_cells(monkeypatch):
    """sg 19 and 33 rows at special positions of Gamma+ (fold tie 2) with random poses. On a float32 cell or angles
    1e-7 rad off 90 deg, fold's orientation rule is decided by rounding (a fold of those cells as given picks other
    poses); canonicalize snaps the cell to the group's exact metric first (since 2026-10-21), so the float64 row, its
    float32 cast and both angle changes canonicalise to one pose, and the nudge (which re-decides the rule on the
    exact metric) moves them to one crystal. The pose is an image of the unnudged one under an element of Gamma+, and
    the nudged rows are fixed points."""
    gen = torch.Generator().manual_seed(11)
    parts = []
    for sg, abc, t in ((19, [5.0, 7.0, 9.0], [0.1, 0.0, 0.0]), (33, [13.9581, 13.1337, 4.011], [0.0688, 0.25, 0.5]),
                       (33, [13.9581, 13.1337, 4.011], [0.0, 0.0, 0.3])):
        cell = torch.tensor([abc + [math.pi / 2] * 3], dtype=F64).expand(6, 6)
        parts.append(CrystalDescription.from_parts(sg, cell, torch.tensor([t] * 6, dtype=F64),
                                                   so3.random_rotations(6, gen)))
    d = CrystalDescription.cat(parts)
    c0 = canon.canonicalize(d)
    assert set(c0.names()) == {'canonical'} and bool((c0.tie == 2).all())
    n = canon.canonicalize(d, nudge=DELTA)
    assert bool(n.nudged.all()) and bool((n.near_face == 0).all())
    flips = 0
    for v in (d.to(dtype=torch.float32), d.replace(cell=d.cell + torch.tensor([0, 0, 0, 1e-7, -1e-7, 1e-7], dtype=F64)),
              d.replace(cell=d.cell + torch.tensor([0, 0, 0, -1e-7, 1e-7, -1e-7], dtype=F64))):
        flips += int(((domains.fold(v).desc.R - c0.desc.R).abs().amax((1, 2, 3)) > 1e-3).sum())
        cv = canon.canonicalize(v)                              # the snap: one pose without the nudge
        assert float((cv.desc.t - c0.desc.t).abs().max()) <= 1e-6 and float((cv.desc.R - c0.desc.R).abs().max()) <= 1e-6
        nv = canon.canonicalize(v, nudge=DELTA)
        assert bool(nv.nudged.all())
        assert float((nv.desc.t - n.desc.t).abs().max()) <= 1e-6 and float((nv.desc.R - n.desc.R).abs().max()) <= 1e-6
    assert flips >= 10                                          # a fold of the cells as given: rounding picks poses
    for i in range(d.num_rows):                                 # an image of the unnudged pose
        sec = domains.translation_section(int(d.sg[i]))
        Q = ops.cartesian_rotation(c0.desc[i:i + 1].cell_matrix().expand(sec.order, 3, 3), sec.W)
        assert float((Q @ c0.desc.R[i, 0] - n.desc.R[i, 0]).abs().amax((1, 2)).min()) <= 1e-12
    again = canon.canonicalize(n.desc)
    assert float(_deviation(again.desc, n.desc).max()) <= 1e-12 and bool((again.tie == 0).all())
    # where the exact-metric fold does not tie, the row is refused and left as nudge=None leaves it
    monkeypatch.setattr(canon, '_exact_metric_pose', lambda row, tol: None)
    c = canon.canonicalize(d, nudge=DELTA)
    assert not bool(c.nudged.any()) and all('exact metric' in m for m in c.nudge_message)
    _assert_same(c, c0)
    assert bool((c.nudge_cell == 0).all()) and bool((c.nudge_t == 0).all())


# ---------------------------------------------------------------------------
# 3: float32 re-descriptions
# ---------------------------------------------------------------------------

def test_nudged_rows_survive_float32_redescriptions():
    gen = torch.Generator().manual_seed(3)
    dc, _ = _cell_cases(gen)
    dt, _ = _centroid_cases(gen)
    d = CrystalDescription.cat([dc, dt])
    c0 = canon.canonicalize(d)
    c = canon.canonicalize(d, nudge=DELTA)
    k = torch.nonzero(c.nudged)[:, 0]
    x, x0 = c.desc[k], c0.desc[k]
    rng = np.random.default_rng(3)
    worst, control_flips = 0.0, 0
    for _ in range(12):
        back = canon.canonicalize(_redescribe32(x, rng))
        assert back.names() == [c.names()[i] for i in k.tolist()]
        assert bool((back.near_face == 0).all())
        worst = max(worst, float(_deviation(back.desc, x).max()))
        ctl = canon.canonicalize(_redescribe32(x0, rng))
        control_flips += int((_deviation(ctl.desc, x0) > 1e-3).sum())
    assert worst <= 1e-5
    assert control_flips >= 5                     # unnudged rows on a face do come back as another row


# ---------------------------------------------------------------------------
# 4: everything else is untouched; idempotence; fd_status
# ---------------------------------------------------------------------------

def test_rows_not_nudged_are_bit_identical():
    d, want, _ = TC._mixed_batch()
    g = torch.Generator().manual_seed(4)
    near, _ = _centroid_cases(g)
    cells, _ = _cell_cases(g)
    pad = CrystalDescription.cat([near, cells])
    pad = CrystalDescription.from_parts(pad.sg, pad.cell, torch.cat([pad.t, torch.zeros_like(pad.t)], 1),
                                        torch.cat([pad.R, pad.R], 1), torch.cat([pad.h, pad.h], 1),
                                        z_prime=torch.ones(pad.num_rows, dtype=torch.long))
    rand = canon.canonicalize(TC._rand(14, 20, g, 2)).desc
    batch = CrystalDescription.cat([d, pad, rand])
    c0 = canon.canonicalize(batch)
    c = canon.canonicalize(batch, nudge=DELTA)
    assert c.names() == c0.names() and c.names()[:len(want)] == want
    assert int(c.nudged.sum()) >= 16
    _assert_same(c, c0, torch.nonzero(~c.nudged)[:, 0])
    eligible = torch.tensor([n in ('canonical', 'pose_only') for n in c0.names()]) & \
        ((c0.near_face.long() & (canon.NEAR_CELL | canon.NEAR_CENTROID)) != 0)
    assert torch.equal(c.nudged, eligible)
    assert TC._nan_equal(c.desc.t[:, 1:], c0.desc.t[:, 1:])            # padded slots untouched
    assert TC._nan_equal(c.desc.R[:, 1:], c0.desc.R[:, 1:])
    off = c0.status.long() == canon.STATUS.index('special_metric')
    assert bool(off.any()) and not bool(c.nudged[off].any())
    none = canon.canonicalize(batch, nudge=None)
    _assert_same(none, c0)
    assert not bool(none.nudged.any()) and float(none.nudge_cell.abs().max()) == 0 and none.nudge_message == [''] * \
        batch.num_rows


def test_fold_band_alone_is_not_nudged(monkeypatch):
    d = canon.canonicalize(TC._rand(2, 6, torch.Generator().manual_seed(5))).desc
    real_fold = domains.fold

    def fold_tie3(x, *a, **kw):
        fr = real_fold(x, *a, **kw)
        return type(fr)(desc=fr.desc, element=fr.element, lattice=fr.lattice, free_shift=fr.free_shift,
                        slack=fr.slack, tie=torch.full_like(fr.tie, 3))
    monkeypatch.setattr(canon.domains, 'fold', fold_tie3)
    c = canon.canonicalize(d, nudge=DELTA)
    assert bool((c.near_face == canon.NEAR_FOLD_BAND).all()) and not bool(c.nudged.any())
    assert c.nudge_message == [''] * d.num_rows


def test_nudge_is_idempotent_and_fd_status_takes_it():
    g = torch.Generator().manual_seed(6)
    d = CrystalDescription.cat([_cell_cases(g)[0], _centroid_cases(g)[0]])
    c = canon.canonicalize(d, nudge=DELTA)
    twice = canon.canonicalize(c.desc, nudge=DELTA)
    assert not bool(twice.nudged.any()) and torch.equal(twice.near_face, c.near_face)
    assert not bool(twice.moved.any()) and float(_deviation(twice.desc, c.desc).max()) <= 1e-12
    # fd_status: an unnudged near-face canonical row is in the domain, but not in the nudged one
    plain = canon.canonicalize(d).desc
    moved = c.nudged
    assert bool(moved[d.sg >= 16].any())                       # sg 19 rows: decided since 2026-10-21
    assert bool(canon.fd_status(plain).is_fd[moved].all())
    st = canon.fd_status(plain, nudge=DELTA)
    assert not bool(st.is_fd[moved].any())
    st = canon.fd_status(c.desc, nudge=DELTA)
    assert bool(st.is_fd[moved].all()) and not bool(st.near_face[moved].any())


# ---------------------------------------------------------------------------
# 5: refusals, the push, arguments
# ---------------------------------------------------------------------------

def test_thin_domains_are_refused():
    g = torch.Generator().manual_seed(7)
    d = CrystalDescription.cat([_row(2, _cell(2, y2=1 - EPS), [0.21, 0.13, 0.17], g),     # cell chart widths are 1
                                _row(2, _cell(2), [EPS, 0.2, 0.3], g)])                   # ASU of sg 2 is 1/2 wide
    c0 = canon.canonicalize(d)
    c = canon.canonicalize(d, nudge=0.6)
    assert not bool(c.nudged.any())
    _assert_same(c, c0)
    assert 'cell chart' in c.nudge_message[0] and 'thinner' in c.nudge_message[0]
    assert 'ASU' in c.nudge_message[1] and 'thinner' in c.nudge_message[1]
    assert bool((c.nudge_cell == 0).all() & (c.nudge_t == 0).all())


def test_a_nudge_that_fails_its_check_keeps_the_row(monkeypatch):
    d, _ = _centroid_cases(torch.Generator().manual_seed(8))
    c0 = canon.canonicalize(d)
    with monkeypatch.context() as m:
        m.setattr(canon, '_min_norm_push', lambda A, b, eps=1e-13: torch.zeros(A.shape[1], dtype=F64))
        c = canon.canonicalize(d, nudge=DELTA)                 # nothing moves: still near a face
    assert not bool(c.nudged.any()) and all('still near a face' in m for m in c.nudge_message)
    _assert_same(c, c0)
    assert bool((c.nudge_cell == 0).all()) and bool((c.nudge_t == 0).all())

    def wrong_way(A, b, eps=1e-13):                            # outward: the row folds to another image
        return -torch.ones(A.shape[1], dtype=F64) * 0.05
    with monkeypatch.context() as m:
        m.setattr(canon, '_min_norm_push', wrong_way)
        c = canon.canonicalize(d, nudge=DELTA)
    assert not bool(c.nudged.any()) and all(m.startswith('nudge: ') for m in c.nudge_message)
    _assert_same(c, c0)
    assert bool((c.nudge_cell == 0).all()) and bool((c.nudge_t == 0).all())     # a refused move records no size
    # every tried row (cell and centroid) refused by the fixed-point check: the move sizes go back to 0
    g = torch.Generator().manual_seed(8)
    d = CrystalDescription.cat([_cell_cases(g)[0], _centroid_cases(g)[0]])
    c0 = canon.canonicalize(d)
    with monkeypatch.context() as m:
        m.setattr(canon, 'NUDGE_FIXED_TOL', -1.0)
        c = canon.canonicalize(d, nudge=DELTA)
    tried = [i for i, msg in enumerate(c.nudge_message) if msg]
    assert len(tried) >= 15 and all('not its own canonical form' in c.nudge_message[i] for i in tried)
    assert not bool(c.nudged.any())
    _assert_same(c, c0)
    assert bool((c.nudge_cell == 0).all()) and bool((c.nudge_t == 0).all())


def test_min_norm_push_against_hand_solutions():
    push = canon._min_norm_push
    A = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=F64)
    assert torch.allclose(push(A, torch.tensor([0.1, 0.2, -1., -1.], dtype=F64)), torch.tensor([0.1, 0.2], dtype=F64))
    assert torch.equal(push(A, torch.tensor([-0.1, -0.2, -1., -1.], dtype=F64)), torch.zeros(2, dtype=F64))
    assert push(A, torch.tensor([0.1, 0.0, -0.05, -1.], dtype=F64)) is None              # x >= 0.1 and x <= 0.05
    # a row the first answer breaks joins the play: x >= 1, then x + y <= 0.5
    B = torch.tensor([[1., 0.], [-1., -1.]], dtype=F64)
    assert torch.allclose(push(B, torch.tensor([1., -0.5], dtype=F64)), torch.tensor([1., -0.5], dtype=F64))
    # a wedge (unit normals at +-45 deg about x): the shortest move is along x, by delta sqrt 2
    s = 1 / math.sqrt(2)
    W = torch.tensor([[s, s], [s, -s]], dtype=F64)
    assert torch.allclose(push(W, torch.tensor([1e-4, 1e-4], dtype=F64)), torch.tensor([2 * s * 1e-4, 0.], dtype=F64))
    # duplicate rows and a zero row
    D = torch.tensor([[1., 0.], [1., 0.]], dtype=F64)
    assert torch.allclose(push(D, torch.tensor([0.3, 0.3], dtype=F64)), torch.tensor([0.3, 0.], dtype=F64))
    assert push(torch.zeros(1, 2, dtype=F64), torch.tensor([0.1], dtype=F64)) is None


def test_cell_faces_match_the_near_face_distance():
    """The faces the nudge pushes against are the ones near_face measures: sg 1-15 those of ``_cell_face_distance``,
    sg 16-74 those of ``axis_order.face_values`` (the axis-order step's; ``_cell_face_distance`` refuses sg >= 16, so
    there is one face definition per group)."""
    g = torch.Generator().manual_seed(9)
    for sg in (2, 14, 4, 15):
        d = canon.canonicalize(TC._rand(sg, 50, g)).desc
        assert torch.equal(_chart_faces(sg, d.cell).amin(1), canon._cell_face_distance(sg, d.cell))
    for sg in (19, 18, 61, 33):                                     # k = 6, 2, 3, 1
        c = canon.canonicalize(TC._rand(sg, 50, g))
        fv = axis_order.face_values(sg, c.desc.cell)
        assert torch.equal(_chart_faces(sg, c.desc.cell), fv), sg
        if fv.shape[1]:
            near = (fv.amin(1) <= ops.FACE_TOL) | (c.n_tied > 1)
            assert torch.equal(c.near_face.long() & canon.NEAR_CELL != 0, near), sg
        with pytest.raises(ValueError, match='1-15'):
            canon._cell_face_distance(sg, c.desc.cell)
    with pytest.raises(ValueError, match='1-15'):
        canon._cell_face_distance(80, torch.tensor([[5., 5., 7., math.pi / 2, math.pi / 2, math.pi / 2]], dtype=F64))


# ---------------------------------------------------------------------------
# 6: the limits README states (pinned so a change to them is seen)
# ---------------------------------------------------------------------------

def test_limit_an_on_face_row_nudges_to_the_side_it_rounds_to():
    """An sg 2 row exactly on the a = b face: its float32 re-descriptions canonicalise to either side, and each is
    nudged inward, so they land on more than one crystal, each its own fixed point."""
    d = _row(2, _cell(2, y2=1.0), [0.21, 0.13, 0.17], torch.Generator().manual_seed(12))
    n = canon.canonicalize(d, nudge=DELTA)
    assert bool(n.nudged[0])
    back = canon.canonicalize(_redescribe32(d[torch.zeros(20, dtype=torch.long)], np.random.default_rng(12)),
                              nudge=DELTA)
    assert bool(back.nudged.all())
    far = _deviation(back.desc, n.desc[torch.zeros(20, dtype=torch.long)]) > 1e-3
    assert 0 < int(far.sum()) < 20
    again = canon.canonicalize(back.desc[far])
    assert float(_deviation(again.desc, back.desc[far]).max()) <= 1e-9 and bool((again.near_face == 0).all())


def test_limit_the_near_tol_edge_jumps():
    """A centroid 1e-5 + 1e-13 from a face is outside near_tol and not moved; its float32 cast is inside and is moved
    by delta less its distance, so fd_status with nudge calls the cast of the stored row not in the domain."""
    row = _row(2, _cell(2), [1e-5 + 1e-13, 0.2, 0.3], torch.Generator().manual_seed(13))
    c64 = canon.canonicalize(row, nudge=DELTA)
    c32 = canon.canonicalize(row.to(dtype=torch.float32), nudge=DELTA)
    assert not bool(c64.nudged[0]) and bool(c32.nudged[0])
    assert abs(float((c32.desc.t - c64.desc.t).abs().max()) - (DELTA - 1e-5)) <= 1e-9
    x32 = c64.desc.to(dtype=torch.float32)
    assert bool(canon.fd_status(x32).is_fd[0]) and not bool(canon.fd_status(x32, nudge=DELTA).is_fd[0])


def test_limit_elongated_triclinic_cells_outrun_the_step():
    """The cell step is in chart units, whose float32 rounding grows about as (c/a)^2: a row 1e-6 inside the
    s_beta = 0 face, nudged, and re-described in 200 span-1 bases in float32. None comes back as another row at
    c/a = 20; some do at c/a = 40."""
    chart = canon._cell_chart(2)
    rng = np.random.default_rng(0)
    N = []
    while len(N) < 200:
        M = rng.integers(-1, 2, (3, 3))
        if round(np.linalg.det(M)) == 1:
            N.append(M)
    N = torch.as_tensor(np.stack(N))
    flips = {}
    for ratio in (20.0, 40.0):
        a, b, c = 4.0, 6.0, 4.0 * ratio
        y = torch.tensor([[math.log(c), b / c, a / b, -1e-6, -0.4, 0.37]], dtype=F64)
        d = CrystalDescription.from_parts(2, chart.map(y)[0], torch.tensor([[0.21, 0.13, 0.17]], dtype=F64),
                                          torch.eye(3, dtype=F64)[None])
        n = canon.canonicalize(d, nudge=DELTA)
        assert n.names() == ['canonical'] and bool(n.nudged[0])
        x = n.desc[torch.zeros(200, dtype=torch.long)]
        back = canon.canonicalize(ops.change_basis(x, N, torch.zeros(200, 3, dtype=F64)).to(dtype=torch.float32))
        flips[ratio] = int((_deviation(back.desc, x) > 1e-3).sum())
    assert flips[20.0] == 0 and flips[40.0] > 0, flips


@pytest.mark.parametrize('nudge', [1e-5, 1e-6, -1e-4, float('nan'), float('inf'), True, 0.0])
def test_bad_nudge_raises(nudge):
    d = TC._rand(2, 2, torch.Generator().manual_seed(1))
    with pytest.raises(ValueError):
        canon.canonicalize(d, nudge=nudge)

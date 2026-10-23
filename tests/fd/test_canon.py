"""crystal_fd.canon: batch canonicalisation with a status per row, and the computed fundamental-domain check.

1. One call on a batch that mixes space groups, Z', both sheets, special metrics, padded unit slots (zero, NaN, h = 0)
   and every kind of bad row, sorted-out rows first and interleaved, returns the stated status per row without
   raising; rows not canonicalised come back bit for bit with empty records, padded slots are untouched, and each
   canonicalised row equals the recipe (the cell step: reduce_cell for sg 1-15, the exact-metric snap and the
   orthorhombic axis order chosen here by a plain lexicographic minimum for sg >= 16; to_sheet; fold) run on its
   first unit alone. An sg 19 row off the sheet is off_sheet (until 2026-10-21 every sg >= 16 row was pose_only).
2. Scrambled crystals (random proper bases, random Gamma+ images) come back to one canonical form in sixteen space
   groups; a second pass moves no row that is not a special metric.
3. fd_status against independently stated expectations: a canonical row; its Gamma+ image (cell reduced, pose not
   folded); a folded pose in an unreduced basis; the same crystal on the other sheet (sheet=1, -1, None); an
   off-sheet Sohncke row; a special metric; Z' = 2; sg >= 16 (decided since 2026-10-21; the same crystal in another
   axis order is not in the domain, its cell not canonical); float32 rows, and float32 values passed as float64.
4. A row a step raises on is isolated (reduce, sheet and fold: failed status, message, row unchanged) and every other
   row's description and records equal an unpatched run; other exceptions propagate; bad arguments raise.
5. near_face: centroid on a domain face, a triclinic cell with a = b, a FREE-class monoclinic cell with a = c, the
   fold's tolerance-band bit; an empty batch.
6. Rows of sg 1-15 come out bit for bit the same whether or not rows of sg >= 16 share their batch, in every mode
   (flag, nudge, sheets +1, -1, None; float64 and float32): the cell step for sg >= 16 does not reach them.
"""
import math

import numpy as np
import pytest
import torch

from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import axis_order, canon, checks, domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table
from tests.fd import _helpers as H

F64 = torch.float64


def _rand(sg, n, g, zmax=1, family=None):
    cells = checks.random_cells(family or LATTICE_TYPE[sg], n, g)
    t = torch.rand(n, zmax, 3, generator=g, dtype=F64)
    R = so3.random_rotations(n * zmax, g).reshape(n, zmax, 3, 3)
    return CrystalDescription.from_parts(sg, cells, t, R, z_prime=torch.ones(n, dtype=torch.long))


def _unit0(d):
    return CrystalDescription.from_parts(d.sg, d.cell, d.t[:, :1], d.R[:, :1], d.h[:, :1])


def _high_cell(x):
    """The sg >= 16 cell step of one group's rows, written plainly: the exact-metric snap, then (sg 16-74) of the
    group's allowed orders the one whose lengths are the lexicographically smallest tuple (Python's min; the rows here
    have no ties), applied with its (N, s)."""
    sg = int(x.sg[0])
    x = x.replace(cell=canon._exact_metric_cell(sg, x.cell))
    if sg > 74:
        return x
    o = axis_order.axis_orders(sg)
    rows = []
    for i in range(x.num_rows):
        L = x.cell[i, :3].tolist()
        j = min(range(o.k), key=lambda q: tuple(L[p] for p in o.perm[q].tolist()))
        r = x[i:i + 1]
        if j:
            moved = ops.change_basis(r, o.N[j][None], o.s[j][None])
            r = moved.replace(cell=r.cell[:, torch.cat([o.perm[j], o.perm[j] + 3])])
        rows.append(r)
    return CrystalDescription.cat(rows)


def _recipe(d, sheet=1):
    x = _unit0(d).double()
    if bool((x.sg <= 15).all()):
        x = ops.reduce_cell(x).desc
    elif bool((x.sg >= 16).all()) and len(torch.unique(x.sg)) == 1:
        x = _high_cell(x)
    if sheet is not None:
        x, _ = domains.to_sheet(x, sheet)
    return domains.fold(x).desc


def _canonical(sg, n, g):
    return canon.canonicalize(_rand(sg, n, g)).desc


def _gamma_plus_images(d, rng, nonidentity=False):
    """The same crystals at a random Gamma+ element and lattice shift (one space group)."""
    tab = group_table(int(d.sg[0]))
    gp = tab.gamma_plus[1:] if nonidentity else tab.gamma_plus
    k = gp[torch.as_tensor(rng.integers(0, len(gp), d.num_rows))]
    shift = torch.as_tensor(rng.integers(-1, 2, (d.num_rows, 3)), dtype=F64)
    return ops.apply(d, tab.gamma_W[k].to(F64), tab.gamma_w[k].to(F64) + shift)


def _set(d, i, **fields):
    """Row i of d with some fields replaced (tensors indexed [i])."""
    out = {}
    for name, v in fields.items():
        a = getattr(d, name).clone()
        a[i] = v
        out[name] = a
    return d.replace(**out)


def _mixed_batch():
    """(Zmax 2 batch, expected status per row, padded-slot rows). Sorted-out rows come first and between others."""
    g = torch.Generator().manual_seed(11)
    rows, want = [], []

    def add(d, name):
        rows.append(d)
        want.extend([name] * d.num_rows)
    tri = math.radians
    add(_set(_rand(2, 1, g, 2), 0, t=torch.tensor([[float('nan')] * 3, [0.1, 0.2, 0.3]], dtype=F64)), 'bad_values')
    zp2 = _rand(2, 1, g, 2)
    add(CrystalDescription.from_parts(2, zp2.cell, zp2.t, zp2.R, zp2.h, z_prime=torch.tensor([2])), 'z_prime_gt1')
    add(_rand(19, 2, g, 2), 'canonical')
    add(_rand(2, 3, g, 2), 'canonical')
    add(_set(_rand(2, 1, g, 2), 0, cell=torch.tensor([5.0, 6.0, 7.0, math.pi / 2, math.pi / 2, math.pi], dtype=F64)),
        'unreal_cell')
    add(_rand(14, 3, g, 2), 'canonical')
    p21 = _rand(4, 2, g, 2)
    add(p21.replace(h=-torch.ones_like(p21.h)), 'off_sheet')                       # Sohncke at h = -1
    off = _rand(14, 1, g, 2)
    add(_set(off, 0, cell=torch.cat([off.cell[0, :3], torch.tensor([tri(91.0), off.cell[0, 4], tri(90.0)],
                                                                   dtype=F64)])), 'mono_angle_off')
    p21c = _rand(14, 2, g, 2)
    add(p21c.replace(h=-torch.ones_like(p21c.h)), 'canonical')                      # moves through an improper element
    special = torch.tensor([5.0, 5.0, 9.0, tri(100), tri(100), tri(110)], dtype=F64)
    add(_set(_rand(2, 1, g, 2), 0, cell=special), 'special_metric')                  # the 3.6 example
    p21_special = torch.tensor([5.0, 7.0, 5.0, tri(90), tri(110), tri(90)], dtype=F64)  # FREE class, a = c: n_auto > 1
    sp = _rand(4, 1, g, 2)
    add(_set(sp, 0, cell=p21_special, h=torch.tensor([-1, -1])), 'off_sheet')        # off sheet beats special
    s19 = _rand(19, 1, g, 2)
    add(s19.replace(h=-torch.ones_like(s19.h)), 'off_sheet')                         # Sohncke sg 19 at h = -1
    bad19 = _rand(19, 1, g, 2)
    add(_set(bad19, 0, cell=torch.cat([bad19.cell[0, :3], torch.tensor([tri(91.0), tri(90), tri(90)], dtype=F64)])),
        'cell_off_group')
    add(_set(_rand(2, 1, g, 2), 0, sg=torch.tensor(231)), 'bad_values')
    det_m1 = _rand(2, 1, g, 2)
    add(_set(det_m1, 0, R=torch.stack([-det_m1.R[0, 0], det_m1.R[0, 1]])), 'bad_values')   # det R = -1
    add(_set(_rand(2, 1, g, 2), 0, h=torch.tensor([0, 1])), 'bad_values')
    add(_set(_rand(2, 1, g, 2), 0, h=torch.tensor([2, 1])), 'bad_values')
    z0 = _rand(2, 1, g, 2)
    add(CrystalDescription.from_parts(2, z0.cell, z0.t, z0.R, z0.h, z_prime=torch.tensor([0])), 'bad_values')
    pad = _rand(14, 2, g, 2)                                                         # padded slots at h = -1 rows
    t_pad, R_pad, h_pad = pad.t.clone(), pad.R.clone(), pad.h.clone()
    t_pad[0, 1] = float('nan')
    R_pad[0, 1] = float('nan')
    R_pad[1, 1] = 0.0
    h_pad[:, 0], h_pad[:, 1] = -1, 0
    add(pad.replace(t=t_pad, R=R_pad, h=h_pad), 'canonical')
    padded = [len(want) - 2, len(want) - 1]
    return CrystalDescription.cat(rows), want, padded


def _nan_equal(a, b):
    return torch.equal(torch.nan_to_num(a.double(), nan=7.0), torch.nan_to_num(b.double(), nan=7.0))


def test_mixed_batch_one_call():
    d, want, padded = _mixed_batch()
    c = canon.canonicalize(d)
    assert c.names() == want
    d64 = d.double()
    for i, name in enumerate(want):
        if not bool(c.done[i]):                     # returned exactly as it came in, with empty records
            for f in ('sg', 'cell', 't', 'R', 'h'):
                assert _nan_equal(getattr(c.desc, f)[i], getattr(d64, f)[i]), (i, name, f)
            assert int(c.n_auto[i]) == -1 and int(c.element[i]) == -1 and not bool(c.on_sheet[i])
            continue
        ref = _recipe(d[i:i + 1])                   # equals the recipe on its first unit alone
        assert (c.desc.t[i, 0] - ref.t[0, 0]).abs().max() <= 1e-12
        assert (c.desc.R[i, 0] - ref.R[0, 0]).abs().max() <= 1e-12
        assert (c.desc.cell[i] - ref.cell[0]).abs().max() <= 1e-12 and int(c.desc.h[i, 0]) == int(ref.h[0, 0])
        assert _nan_equal(c.desc.t[i, 1], d64.t[i, 1]) and _nan_equal(c.desc.R[i, 1], d64.R[i, 1])   # padding kept
        assert int(c.desc.h[i, 1]) == int(d64.h[i, 1])
    names = c.names()
    moved_sheet = [i for i in range(len(want)) if names[i] == 'canonical' and int(d.sg[i]) == 14
                   and int(d.h[i, 0]) == -1]
    assert len(moved_sheet) == 4 and bool(c.sheet_moved[moved_sheet].all())
    assert bool((c.desc.h[moved_sheet, 0] == 1).all()) and bool(c.on_sheet[moved_sheet].all())
    assert set(padded) <= set(moved_sheet)
    off = [i for i, n in enumerate(names) if n == 'off_sheet']
    p21_off = [i for i in off if int(d.sg[i]) == 4]
    assert not bool(c.on_sheet[off].any()) and int(c.n_auto[p21_off[-1]]) > 1      # the special one stays off_sheet
    assert int(c.n_auto[names.index('special_metric')]) > 1
    s19 = [i for i, n in enumerate(names) if n == 'off_sheet' and int(d.sg[i]) == 19]
    assert len(s19) == 1 and not bool(c.on_sheet[s19[0]]) and int(c.n_auto[s19[0]]) == 1
    high = [i for i in range(len(want)) if int(d.sg[i]) >= 16 and bool(c.done[i])]
    assert len(high) == 3 and not bool(torch.isnan(c.cell_snap[high]).any())       # the snap is recorded
    assert bool(torch.isnan(c.cell_snap[[i for i in range(len(want)) if int(d.sg[i]) <= 15]]).all())


SCRAMBLE_GROUPS = [1, 2, 4, 5, 7, 9, 13, 14, 15, 19, 29, 33, 61, 92, 148, 227]


@pytest.mark.parametrize('sg', SCRAMBLE_GROUPS)
def test_scrambled_rows_return_to_one_form(sg):
    g = torch.Generator().manual_seed(sg)
    rng = np.random.default_rng(sg)
    c0 = canon.canonicalize(_rand(sg, 60, g))
    keep = torch.nonzero(c0.n_auto <= 1)[:, 0]                 # special metrics have no unique pose
    d = c0.desc[keep]
    s = _gamma_plus_images(d, rng)
    if sg <= 2:
        s = H.scramble(s, rng)
    c = canon.canonicalize(s)
    assert bool(c.done.all())
    assert (c.desc.t[:, 0] - d.t[:, 0]).abs().max() <= 1e-9 and (c.desc.R[:, 0] - d.R[:, 0]).abs().max() <= 1e-9
    assert ((c.desc.cell - d.cell).abs() / d.cell.abs()).max() <= 1e-9
    ref = _recipe(s)
    assert (c.desc.t[:, 0] - ref.t[:, 0]).abs().max() <= 1e-12 and (c.desc.R[:, 0] - ref.R[:, 0]).abs().max() <= 1e-12
    again = canon.canonicalize(c.desc)
    assert not bool((again.moved & (again.n_auto <= 1)).any())


def test_fd_status_independent_expectations():
    g = torch.Generator().manual_seed(3)
    rng = np.random.default_rng(3)
    p1 = _canonical(2, 6, g)
    st = canon.fd_status(p1)
    assert bool(st.is_fd.all() & st.decided.all() & st.cell_reduced.all() & st.pose_folded.all())
    # a Gamma+ image (not the identity): the same crystal, cell still reduced, pose not folded
    img = _gamma_plus_images(p1, rng, nonidentity=True)
    st = canon.fd_status(img)
    assert not bool(st.is_fd.any()) and bool(st.cell_reduced.all()) and not bool(st.pose_folded.any())
    # a folded pose in an unreduced basis: pose folded in its own basis, cell not reduced
    N = torch.tensor([[1, 1, 0], [0, 1, 0], [0, 0, 1]]).repeat(6, 1, 1)
    unred = domains.fold(ops.change_basis(p1, N, torch.zeros(6, 3, dtype=F64))).desc
    st = canon.fd_status(unred)
    assert not bool(st.is_fd.any()) and not bool(st.cell_reduced.any()) and bool(st.pose_folded.all())
    # the same P2_1/c crystals on sheet -1: in the domain only for the sheet asked for
    p21c = _canonical(14, 6, g)
    minus = canon.canonicalize(p21c, sheet=-1).desc
    assert bool((minus.h[:, 0] == -1).all())
    assert not bool(canon.fd_status(minus, sheet=1).is_fd.any()) and not bool(canon.fd_status(minus).on_sheet.any())
    assert bool(canon.fd_status(minus, sheet=-1).is_fd.all())
    assert bool(canon.fd_status(minus, sheet=None).is_fd.all() & canon.fd_status(p21c, sheet=None).is_fd.all())
    # an off-sheet Sohncke row: decided, not on the sheet, not in the domain
    p21 = _rand(4, 4, g)
    p21 = canon.canonicalize(p21.replace(h=-torch.ones_like(p21.h))).desc
    st = canon.fd_status(p21)
    assert bool(st.decided.all()) and not bool(st.on_sheet.any()) and not bool(st.is_fd.any())
    # a special metric: decided, flagged, not in the domain
    sp = canon.canonicalize(_set(_rand(2, 1, g), 0, cell=torch.tensor(
        [5.0, 5.0, 9.0, math.radians(100), math.radians(100), math.radians(110)], dtype=F64))).desc
    st = canon.fd_status(sp)
    assert bool(st.decided.all() & st.special_metric.all()) and not bool(st.is_fd.any())
    # Z' = 2: undecided; sg >= 16: decided since 2026-10-21, and another axis order of the same crystal is not in it
    zp2 = _rand(2, 1, g, 2)
    st = canon.fd_status(CrystalDescription.from_parts(2, zp2.cell, zp2.t, zp2.R, zp2.h, z_prime=torch.tensor([2])))
    assert not bool(st.zp1.any() | st.decided.any() | st.is_fd.any())
    s19 = _canonical(19, 4, g)
    st = canon.fd_status(s19)
    assert bool(st.decided.all() & st.is_fd.all() & st.cell_reduced.all() & st.pose_folded.all())
    o = axis_order.axis_orders(19)
    j = [q for q in range(o.k) if o.perm[q].tolist() == [1, 0, 2]][0]                   # b, a, c
    swapped = domains.fold(ops.change_basis(s19, o.N[j].expand(4, 3, 3), o.s[j].expand(4, 3))).desc
    st = canon.fd_status(swapped)
    assert bool(st.decided.all() & st.pose_folded.all()) and not bool(st.is_fd.any() | st.cell_reduced.any())
    # float32 rows at the float32 tolerance; float32 values passed as float64 need tol set (monoclinic alpha, gamma)
    both = CrystalDescription.cat([p1, p21c])
    f32 = both.to(dtype=torch.float32)
    st32, c32 = canon.fd_status(f32), canon.canonicalize(f32)
    assert bool((st32.is_fd | st32.near_face).all()) and not bool((c32.moved & (c32.near_face == 0)).any())
    up = f32.double()
    mono = up.sg == 14
    assert not bool(canon.fd_status(up).is_fd[mono].any())
    st = canon.fd_status(up, tol=1e-5)
    assert bool((st.is_fd | st.near_face).all())


def test_failed_rows_are_isolated(monkeypatch):
    g = torch.Generator().manual_seed(5)
    zp2 = _rand(2, 1, g, 2)
    zp2 = CrystalDescription.from_parts(2, zp2.cell, zp2.t, zp2.R, zp2.h, z_prime=torch.tensor([2]))
    # a sorted-out row and an sg >= 16 row first, so the failing row's index differs in every step's sub-batch
    d = CrystalDescription.cat([zp2, _rand(19, 1, g, 2), _rand(2, 9, g, 2)])
    clean = canon.canonicalize(d)
    assert clean.names()[:2] == ['z_prime_gt1', 'canonical']
    target = 6
    sentinel = d.cell[target, 0].item()
    red_cell = clean.desc.cell[target, 0].item()
    real = dict(reduce_cell=ops.reduce_cell, to_sheet=domains.to_sheet, fold=domains.fold)

    def raising(name, mod, value, exc):
        def fn(x, *a, **kw):
            if bool((x.cell[:, 0] == value).any()):
                raise exc('sentinel')
            return real[name](x, *a, **kw)
        monkeypatch.setattr(mod, name, fn)

    fields = ('N', 's', 'n_auto', 'n_tied', 'on_sheet', 'sheet_moved', 'element', 'lattice', 'free_shift', 'tie')
    for name, mod, value, status in (('reduce_cell', canon.ops, sentinel, 'reduce_failed'),
                                     ('to_sheet', canon.domains, red_cell, 'sheet_failed'),
                                     ('fold', canon.domains, red_cell, 'fold_failed')):
        raising(name, mod, value, RuntimeError)
        c = canon.canonicalize(d)
        monkeypatch.setattr(mod, name, real[name])
        names = c.names()
        assert names[target] == status and 'sentinel' in c.message[target]
        assert names[:target] + names[target + 1:] == clean.names()[:target] + clean.names()[target + 1:]
        assert torch.equal(c.desc.t[target], d.t[target].double()) and torch.equal(c.desc.cell[target],
                                                                                 d.cell[target].double())
        others = [i for i in range(d.num_rows) if i != target]
        for f in ('cell', 't', 'R', 'h'):
            assert torch.equal(getattr(c.desc, f)[others], getattr(clean.desc, f)[others]), (name, f)
        for f in fields:
            assert torch.equal(getattr(c, f)[others], getattr(clean, f)[others]), (name, f)
    raising('fold', canon.domains, red_cell, TypeError)          # not a row's problem: propagates
    with pytest.raises(TypeError):
        canon.canonicalize(d)


def test_bad_arguments_raise():
    d = _rand(2, 2, torch.Generator().manual_seed(1))
    for kw in (dict(cell_tol=1e-4), dict(tol=-1.0), dict(near_tol=float('nan')), dict(sheet=0)):
        with pytest.raises(ValueError):
            canon.canonicalize(d, **kw)
    with pytest.raises(ValueError):
        canon.fd_status(d, tol=0.0)


def test_near_face_and_empty(monkeypatch):
    g = torch.Generator().manual_seed(9)
    d = _canonical(2, 4, g)
    t = d.t.clone()
    t[0, 0, 0] = 0.0                                   # centroid on the x face of [0, 1/2)^3
    cell = d.cell.clone()
    cell[1, 1] = cell[1, 0]                            # a = b: the r1 = 1 face of the triclinic cell chart
    c = canon.canonicalize(d.replace(t=t, cell=cell))
    assert int(c.near_face[0]) & canon.NEAR_CENTROID
    assert int(c.near_face[1]) & canon.NEAR_CELL
    assert int(c.near_face[2]) == 0 and int(c.near_face[3]) == 0
    p21 = _rand(4, 1, g)                               # FREE monoclinic class, a = c (1 - 1e-7): the a <= c face
    a, c_len = 6.0, 6.0 * (1 - 1e-7)
    p21 = _set(p21, 0, cell=torch.tensor([c_len, 7.0, a, math.pi / 2, math.radians(105), math.pi / 2], dtype=F64))
    assert int(canon.canonicalize(p21).near_face[0]) & canon.NEAR_CELL
    real_fold = domains.fold

    def fold_tie3(x, *a, **kw):
        fr = real_fold(x, *a, **kw)
        return type(fr)(desc=fr.desc, element=fr.element, lattice=fr.lattice, free_shift=fr.free_shift,
                        slack=fr.slack, tie=torch.full_like(fr.tie, 3))
    monkeypatch.setattr(canon.domains, 'fold', fold_tie3)
    assert bool((canon.canonicalize(d).near_face & canon.NEAR_FOLD_BAND).all())
    monkeypatch.setattr(canon.domains, 'fold', real_fold)
    e = canon.canonicalize(d[:0])
    assert e.desc.num_rows == 0 and len(e.status) == 0 and e.message == []
    assert canon.fd_status(d[:0]).is_fd.shape == (0,)


@pytest.mark.parametrize('kw', [dict(), dict(nudge=1e-4), dict(sheet=-1), dict(sheet=None),
                                dict(sheet=None, nudge=1e-4)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
def test_rows_of_sg_1_15_do_not_see_the_higher_cell_step(kw, dtype):
    """Every Canonicalization field of the sg 1-15 rows (the mixed batch, near-face rows, random rows of each of sg
    1-15 at both handedness) is bit for bit the same alone and interleaved with sg >= 16 rows of every family."""
    from tests.fd import test_nudge as TN

    def zmax2(x):                                     # Z' = 1 rows with a second, unused unit slot
        return CrystalDescription.from_parts(x.sg, x.cell, torch.cat([x.t, torch.zeros_like(x.t)], 1),
                                             torch.cat([x.R, x.R], 1), torch.cat([x.h, x.h], 1),
                                             z_prime=torch.ones(x.num_rows, dtype=torch.long))
    g = torch.Generator().manual_seed(21)
    d, _, _ = _mixed_batch()
    low = [d[torch.nonzero(d.sg <= 15)[:, 0]], zmax2(TN._cell_cases(g)[0]), zmax2(TN._centroid_cases(g)[0])]
    for sg in range(1, 16):
        r = _rand(sg, 4, g, 2)
        low.append(r.replace(h=torch.where(torch.rand(4, 1, generator=g) < 0.5, -1, 1).to(r.h.dtype).repeat(1, 2)))
    low = CrystalDescription.cat(low).to(dtype=dtype)
    low = low[torch.nonzero(low.sg <= 15)[:, 0]]                                    # the centroid cases hold sg 19
    high = CrystalDescription.cat([_rand(sg, 2, g, 2) for sg in (16, 19, 33, 61, 70, 92, 146, 148, 167, 178, 194,
                                                                  205, 227, 230)]).to(dtype=dtype)
    both = CrystalDescription.cat([high[:9], low[:20], high[9:], low[20:]])
    pos = torch.cat([torch.arange(9, 29), torch.arange(29 + high.num_rows - 9, both.num_rows)])
    alone, mixed = canon.canonicalize(low, **kw), canon.canonicalize(both, **kw)
    assert len(pos) == low.num_rows and set(alone.names()) >= {'canonical', 'special_metric', 'bad_values'}
    bits = TN._bits_equal
    for f in ('cell', 't', 'R', 'h', 'sg', 'z_prime', 'unit_mask'):
        assert bits(getattr(alone.desc, f), getattr(mixed.desc, f)[pos]), f
    for f in TN.FIELDS + ('nudged', 'nudge_cell', 'nudge_t', 'cell_snap'):
        assert bits(getattr(alone, f), getattr(mixed, f)[pos]), f
    assert alone.message == [mixed.message[i] for i in pos.tolist()]
    assert alone.nudge_message == [mixed.nudge_message[i] for i in pos.tolist()]
    assert bool(torch.isnan(alone.cell_snap).all())
    if 'nudge' in kw:
        assert bool(alone.nudged.any())

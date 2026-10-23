"""crystal_fd: the cell step of ``canon.canonicalize`` for space groups 16-230 (2026-10-21).

Orthorhombic groups (16-74) get the canonical axis order of ``axis_order``; every group 16-230 gets its cell snapped
to the group's exact metric (``canon._exact_metric_cell``).

1. The derivation: the classes are the survey's (21 / 26 / 1 / 11 groups with 1 / 2 / 3 / 6 orders; the two-order
   class is a <-> b with c fixed, Pbca's the three cyclic orders); every kept (N, s) is a det +1 signed axis
   permutation with a quarter-grid shift that ``settings.setting_matches`` accepts, every sign choice of a kept
   permutation is admitted and every permutation left out is refused with every sign choice; the faces are the
   class's rule; the status codes keep their order; classes off the survey raise, through canonicalize too.
2. Every orthorhombic class (a representative set; all 59 groups in the slow tier): random crystals in every allowed
   axis order and sign choice, each at a random Gamma+ image, canonicalise to one row within 1e-12; the canonical
   cell is ordered by the class's rule (k = 1: the input's cell); status canonical, n_auto 1; ``N`` and ``s`` are the
   cell step (the fold of change_basis(snap(x), N, s) is the row) and undo the re-description up to a diagonal sign
   change; the canonical row's unit cell (built from ``group_table``'s G, the literal oracle) is the input's up to a
   proper isometry and a translation within 1e-9 Angstrom, while a shifted origin or a flipped handedness is not.
3. Ties. Mirror faces (k = 6: a = b, b = c, a = b = c; k = 2: a = b; exactly, and within ``face_tol``) are
   special_metric with n_auto 2 / 2 / 6 / 2 and are never nudged; just outside ``face_tol`` the row is canonical with
   no flag; with ``near_tol`` above ``face_tol`` a row between them is canonical, flagged and nudged. Pbca: the seams
   a = b and a = c (exactly, and 1e-6 off) are canonical with NEAR_CELL (n_tied 2 when exact), every cyclic order of
   a seam row canonicalises to one row, the nudge moves them by the shortest log-length step to delta from both
   faces, they are fixed points and their float32 re-descriptions come back within 1e-5, while rows within the tie
   tolerance of the seam do land on another cyclic order; the cone a = b = c is special_metric with n_auto 3.
   n_auto is the order of the subgroup the allowed orders within ``face_tol`` generate, against a plain count on
   chains of near-ties (b - a and c - b within ``face_tol``, c - a not): 6 in the six-order class, 2 in a <-> b, 1 in
   Pbca (canonical, NEAR_CELL); statuses follow n_auto > 1.
4. Groups 75-230 (a representative set; all 156 in the slow tier): every Gamma image (Gamma+ and, in a non-Sohncke
   group, the improper elements too; in a Sohncke group the improper images are off_sheet and fold to one row of
   their own) canonicalises to one row; a cell perturbed by about 1e-7 is snapped to the exact metric (deviation at
   rounding), ``cell_snap`` is the perturbation's size, and the snapped crystal's atoms are within a few snap sizes
   times the cell of the input's; an exact cell is snapped by rounding only.
5. Float32 re-descriptions (an allowed axis order, a Gamma+ element with lattice and free shifts, a cast) of the
   canonical rows of every class land back on the canonical row within 1e-5, except flagged rows.
6. A row the new step raises on is isolated (reduce_failed, message, every other row as in a clean run); rows of sg
   6-9 (K nontrivial) keep their statuses and come out bit for bit as reduce_cell, to_sheet and fold give them.
"""
import math

import numpy as np
import pytest
import torch

from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import axis_order, canon, checks, domains, io_mxt, ops, settings, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table
from mxtaltools.crystal_search.standardize import setting_shift

F64 = torch.float64
DELTA = 1e-4
SIGNS = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
ORTHO_FAST = (19, 22, 70, 18, 43, 56, 61, 33, 62)          # k = 6, 6, 6, 2, 2, 2, 3, 1, 1 (P, F, I and C/A lattices)
HIGH_FAST = (75, 80, 88, 92, 96, 122, 142, 143, 146, 148, 152, 155, 160, 166, 167, 169, 178, 191, 194, 195, 198, 205,
             210, 212, 225, 227, 230)
ORTHO_SLOW = tuple(sg for sg in axis_order.ORTHORHOMBIC if sg not in ORTHO_FAST)
HIGH_SLOW = tuple(sg for sg in range(75, 231) if sg not in HIGH_FAST)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rows(sg, n, g, cells=None):
    cells = checks.random_cells(LATTICE_TYPE[sg], n, g) if cells is None else torch.as_tensor(cells, dtype=F64)
    return CrystalDescription.from_parts(sg, cells, torch.rand(n, 1, 3, generator=g, dtype=F64),
                                         so3.random_rotations(n, g).reshape(n, 1, 3, 3))


def _ortho(sg, abc, g):
    """Rows of group ``sg`` with the given orthorhombic lengths (list of triples), random poses."""
    cells = torch.tensor([list(x) + [math.pi / 2] * 3 for x in abc], dtype=F64)
    return _rows(sg, len(abc), g, cells)


def _deviation(a, b):
    """[B] largest difference of unit 0: cell lengths relative, angles, t, R entries (and inf where h differs)."""
    a, b = a.double(), b.double()
    dev = torch.stack([((a.cell[:, :3] - b.cell[:, :3]).abs() / b.cell[:, :3]).amax(1),
                       (a.cell[:, 3:] - b.cell[:, 3:]).abs().amax(1), (a.t[:, 0] - b.t[:, 0]).abs().amax(1),
                       (a.R[:, 0] - b.R[:, 0]).abs().amax((1, 2))], 1).amax(1)
    return torch.where(a.h[:, 0] == b.h[:, 0], dev, torch.full_like(dev, float('inf')))


def _variants(sg, j):
    """[(N [3, 3] long, s [3])] order j of group ``sg`` with every det +1 sign choice, each with its own shift."""
    o = axis_order.axis_orders(sg)
    out = []
    for D in SIGNS:
        N = o.N[j] @ torch.diag(torch.tensor(D))
        s = setting_shift(N.numpy().astype(np.float64), sg)
        assert s is not None, (sg, j, D)
        out.append((N, torch.as_tensor(s, dtype=F64)))
    return out


def _reorder(d, N, s):
    return ops.change_basis(d, N.expand(d.num_rows, 3, 3), s.expand(d.num_rows, 3))


def _gamma_image(d, rng, plus=True):
    """The rows (one group) at a random Gamma+ (or Gamma) element, lattice shift and free-axis shift."""
    tab = group_table(int(d.sg[0]))
    pool = tab.gamma_plus if plus else torch.arange(len(tab.gamma_W))
    k = pool[torch.as_tensor(rng.integers(0, len(pool), d.num_rows))]
    w = tab.gamma_w[k] + torch.as_tensor(rng.integers(-1, 2, (d.num_rows, 3)), dtype=F64) \
        + torch.as_tensor(rng.random((d.num_rows, 3)), dtype=F64) * tab.free.to(F64)
    return ops.apply(d, tab.gamma_W[k], w)


def _faces_ok(sg, cell, tol=0.0):
    """[b] the canonical order's rule holds within ``tol`` (relative)."""
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    kind = axis_order.axis_orders(sg).kind
    if kind == 'all':
        return (a <= b * (1 + tol)) & (b <= c * (1 + tol))
    if kind == 'swap_ab':
        return a <= b * (1 + tol)
    if kind == 'cyclic':
        return (a <= b * (1 + tol)) & (a <= c * (1 + tol))
    return torch.ones(len(cell), dtype=torch.bool)


_SIGNED = [N for _, _, N in axis_order._signed_permutations()]


def _cartesian_cell(desc, X, oracle):
    """([Z, n, 3] Cartesian atoms of the unit cell of a one-row description, built literally from group_table's G, its
    T_fc)."""
    tab = group_table(int(desc.sg[0]))
    cell = desc.cell[0].double().numpy()
    atoms = oracle.unit_cell(cell, desc.t[0, 0].double().numpy(), desc.pose[0, 0].double().numpy(), X,
                             tab.g_W.numpy(), tab.g_w.numpy())
    return atoms, oracle.cell_matrix(cell)


def _isometry_deviation(a, b, X, oracle):
    """Smallest largest atom deviation (Angstrom) between the unit cells of the one-row descriptions a and b over the
    proper isometries that map b's lattice basis onto a det +1 signed permutation of a's (f_a = M f_b + tau, M^T G_a M
    = G_b) and the translations that put atom 0 of a molecule of b on atom 0 of molecule 0 of a."""
    A, Ta = _cartesian_cell(a, X, oracle)
    B, Tb = _cartesian_cell(b, X, oracle)
    Fa = np.linalg.solve(Ta, A.reshape(-1, 3).T).T.reshape(A.shape)
    Fb = np.linalg.solve(Tb, B.reshape(-1, 3).T).T.reshape(B.shape)
    Ga, Gb = Ta.T @ Ta, Tb.T @ Tb
    best = float('inf')
    for M in _SIGNED:
        M = M.astype(np.float64)
        if np.abs(M.T @ Ga @ M - Gb).max() > 1e-9 * np.abs(Gb).max():
            continue
        Fm = Fb @ M.T
        for k in range(len(Fm)):
            Fc = Fm + (Fa[0, 0] - Fm[k, 0])
            best = min(best, oracle.max_atom_deviation(A, Fc @ Ta.T, a.cell[0].double().numpy()))
    return best


# ---------------------------------------------------------------------------
# 1: the derivation
# ---------------------------------------------------------------------------

def test_classes_are_the_survey_s():
    cl = axis_order.classes()
    assert {k: len(v) for k, v in cl.items()} == {1: 21, 2: 26, 3: 1, 6: 11}
    assert cl[3] == (61,)
    assert cl[6] == (16, 19, 22, 23, 24, 47, 48, 69, 70, 71, 73)
    assert cl[1] == (26, 28, 29, 30, 31, 33, 36, 38, 39, 40, 41, 46, 51, 52, 53, 54, 57, 60, 62, 63, 64)
    for sg in axis_order.ORTHORHOMBIC:
        o = axis_order.axis_orders(sg)
        perms = [tuple(p) for p in o.perm.tolist()]
        assert perms[0] == (0, 1, 2) and len(set(perms)) == o.k, sg
        want = {1: [(0, 1, 2)], 2: [(0, 1, 2), (1, 0, 2)], 3: [(0, 1, 2), (1, 2, 0), (2, 0, 1)]}.get(o.k)
        if want is not None:
            assert perms == want, sg
        assert o.kind == axis_order.KINDS[o.k] and o.mirror == (o.k in (2, 6))
        assert o.faces.shape == ({1: 0, 2: 1, 3: 2, 6: 2}[o.k], 3)
        for j, p in enumerate(perms):
            N = o.N[j]
            assert round(float(torch.linalg.det(N.double()))) == 1
            assert torch.equal(N.abs(), torch.eye(3, dtype=torch.long)[:, list(p)]), (sg, j)  # new axis i = old p[i]
            assert settings.setting_matches(settings.std_ops(sg), sg, N.double().numpy(), o.s[j].numpy()), (sg, j)
            assert bool(((o.s[j] * 4) == torch.round(o.s[j] * 4)).all())                 # the quarter grid
        for i, p in enumerate(perms):                                                      # the composition table
            for q_i, q in enumerate(perms):
                assert perms[int(o.comp[i, q_i])] == tuple(p[x] for x in q), (sg, i, q_i)
        for perm, signs, N in axis_order._signed_permutations():                          # the rest are refused
            admitted = setting_shift(N.astype(np.float64), sg) is not None
            assert admitted == (perm in perms), (sg, perm, signs)
    with pytest.raises(ValueError):
        axis_order.axis_orders(75)
    assert canon.STATUS == ('canonical', 'special_metric', 'off_sheet', 'pose_only', 'bad_values', 'unreal_cell',
                            'z_prime_gt1', 'mono_angle_off', 'cell_off_group', 'reduce_failed', 'sheet_failed',
                            'fold_failed')


def test_tables_off_the_survey_raise(monkeypatch):
    """A derivation that does not give the survey's classes raises, and canonicalize lets it propagate (it is not a
    row's problem, so no row is marked reduce_failed for it)."""
    monkeypatch.setattr(axis_order, '_ORDERS', {})
    monkeypatch.setattr(axis_order, 'SURVEY_COUNTS', {1: 22, 2: 25, 3: 1, 6: 11})
    with pytest.raises(RuntimeError, match='survey'):
        axis_order.axis_orders(19)
    d = CrystalDescription.cat([_rows(2, 1, torch.Generator().manual_seed(0)),
                                _rows(19, 2, torch.Generator().manual_seed(0))])
    with pytest.raises(RuntimeError, match='survey'):
        canon.canonicalize(d)


def test_choose_against_the_rule():
    """The lexicographic choice gives each class's rule on random lengths, with ties broken as stated."""
    g = torch.Generator().manual_seed(1)
    L = 4 + 12 * torch.rand(400, 3, generator=g, dtype=F64)
    cell = torch.cat([L, torch.full_like(L, math.pi / 2)], 1)
    for sg in (19, 18, 61, 33):
        o = axis_order.axis_orders(sg)
        pick, n_tied, n_auto, viol, face = axis_order.choose(sg, cell, 1e-10, 1e-5)
        new = L[torch.arange(400)[:, None], o.perm[pick]]
        assert bool(_faces_ok(sg, new).all()) and bool((n_tied == 1).all()) and bool((n_auto == 1).all())
        assert bool((viol == 0).all())
        if o.k == 6:
            assert torch.equal(new, torch.sort(L, 1).values)
        if o.k == 1:
            assert torch.equal(new, L) and bool(torch.isnan(face).all())
        else:
            assert torch.equal(face, axis_order.face_values(sg, torch.cat([new, cell[:, 3:]], 1)).amin(1))
    # Pbca: a = b exactly -> the cyclic order with b <= c; a = c exactly -> likewise
    for abc, want in (([5., 5., 9.], [5., 5., 9.]), ([5., 9., 5.], [5., 5., 9.]), ([9., 5., 5.], [5., 5., 9.])):
        c = torch.tensor([abc + [math.pi / 2] * 3], dtype=F64)
        pick, n_tied, n_auto, _, face = axis_order.choose(61, c, 1e-10, 1e-5)
        o = axis_order.axis_orders(61)
        assert c[0, :3][o.perm[pick[0]]].tolist() == want and int(n_tied[0]) == 2 and int(n_auto[0]) == 1
        assert float(face[0]) == 0.0


# ---------------------------------------------------------------------------
# 2: every allowed order of one crystal -> one row, ordered, the same crystal
# ---------------------------------------------------------------------------

def _one_row_from_every_order(sg, chiral_molecule, literal_oracle, n=5, check_atoms=2):
    g = torch.Generator().manual_seed(sg)
    rng = np.random.default_rng(sg)
    d = _rows(sg, n, g)
    c0 = canon.canonicalize(d)
    assert c0.names() == ['canonical'] * n and bool((c0.n_auto == 1).all()), (sg, c0.names())
    assert bool(_faces_ok(sg, c0.desc.cell).all()), sg
    o = axis_order.axis_orders(sg)
    if o.k == 1:
        assert float(((c0.desc.cell[:, :3] - d.cell[:, :3]).abs() / d.cell[:, :3]).max()) <= 1e-15
    # the records are the cell step: fold(change_basis(snap(x), N, s)) is the row
    snapped = d.replace(cell=canon._exact_metric_cell(sg, d.cell))
    again = domains.fold(ops.change_basis(snapped, c0.N, c0.s)).desc
    assert float(_deviation(again, c0.desc).max()) <= 1e-12, sg
    X = chiral_molecule['pos']
    for j in range(o.k):
        for N, s in _variants(sg, j):
            x = _gamma_image(_reorder(d, N, s), rng)
            c = canon.canonicalize(x)
            assert c.names() == ['canonical'] * n, (sg, j)
            assert float(_deviation(c.desc, c0.desc).max()) <= 1e-12, (sg, j, N.tolist())
            both = N.double() @ c.N.double()             # re-description then cell step: the direct step up to signs
            assert bool((both.abs() == c0.N.double().abs()).all()), (sg, j)
    for i in range(check_atoms):                                   # the canonical row is the input crystal
        dev = _isometry_deviation(d[i:i + 1], c0.desc[i:i + 1], X, literal_oracle)
        assert dev <= 1e-9, (sg, i, dev)
    return d, c0


@pytest.mark.parametrize('sg', ORTHO_FAST)
def test_every_allowed_order_gives_one_row(sg, chiral_molecule, literal_oracle):
    _one_row_from_every_order(sg, chiral_molecule, literal_oracle)


@pytest.mark.slow
@pytest.mark.parametrize('sg', ORTHO_SLOW)
def test_every_allowed_order_gives_one_row_all_groups(sg, chiral_molecule, literal_oracle):
    _one_row_from_every_order(sg, chiral_molecule, literal_oracle, n=3, check_atoms=1)


def test_the_isometry_check_has_power(chiral_molecule, literal_oracle):
    """A canonical P212121 / Pbca row with its origin shifted by a quarter, or with h flipped, is not the input."""
    X = chiral_molecule['pos']
    for sg in (19, 61):
        d = _rows(sg, 1, torch.Generator().manual_seed(100 + sg))
        c = canon.canonicalize(d).desc
        assert _isometry_deviation(d, c, X, literal_oracle) <= 1e-9
        shifted = c.replace(t=c.t + torch.tensor([0.25, 0.0, 0.0], dtype=F64))
        assert _isometry_deviation(d, shifted, X, literal_oracle) > 0.1
        assert _isometry_deviation(d, c.replace(h=-c.h), X, literal_oracle) > 0.1
        o = axis_order.axis_orders(sg)                  # an allowed order without its origin shift (where it has one)
        j = [q for q in range(o.k) if bool(o.s[q].abs().max() > 0)]
        if j:
            wrong = ops.change_basis(d, o.N[j[0]][None], torch.zeros(1, 3, dtype=F64))
            assert _isometry_deviation(d, wrong, X, literal_oracle) > 0.1


# ---------------------------------------------------------------------------
# 3: ties: mirror faces, Pbca's seams and cone
# ---------------------------------------------------------------------------

def test_mirror_faces_are_special_metrics():
    g = torch.Generator().manual_seed(3)
    near, far = 1 + 5e-6, 1 + 2e-5                     # within face_tol (log 5e-6) and outside it (log 2e-5)
    cases = [  # sg, lengths, status, n_auto
        (19, (5., 5., 9.), 'special_metric', 2), (19, (9., 5., 5.), 'special_metric', 2),
        (19, (6., 6., 6.), 'special_metric', 6), (19, (5., 5 * near, 9.), 'special_metric', 2),
        (19, (5., 9., 9. * near), 'special_metric', 2), (19, (5., 5 * far, 9.), 'canonical', 1),
        (18, (5., 5., 9.), 'special_metric', 2), (18, (5 * near, 5., 9.), 'special_metric', 2),
        (18, (5., 5 * far, 9.), 'canonical', 1), (18, (5., 9., 9.), 'canonical', 1),      # b = c: no mirror in k = 2
        (22, (7., 7., 7.), 'special_metric', 6), (56, (6., 6., 4.), 'special_metric', 2),
        (33, (6., 6., 6.), 'canonical', 1),                                              # one order: no automorphism
    ]
    d = CrystalDescription.cat([_ortho(sg, [abc], g) for sg, abc, *_ in cases])
    c = canon.canonicalize(d)
    assert c.names() == [x[2] for x in cases]
    assert c.n_auto.tolist() == [x[3] for x in cases]
    mirror_faced = [i for i, x in enumerate(cases) if x[2] == 'special_metric']
    assert bool((c.near_face[mirror_faced].long() & canon.NEAR_CELL != 0).all())        # near a face, as for sg 1-15
    assert int(c.near_face[5]) == 0 and int(c.near_face[8]) == 0
    assert bool(_faces_ok(19, c.desc.cell[:6], 1e-10).all())
    n = canon.canonicalize(d, nudge=DELTA)                                              # never nudged
    assert not bool(n.nudged.any()) and all(m == '' for m in n.nudge_message)
    assert torch.equal(n.desc.cell, c.desc.cell) and torch.equal(n.desc.t, c.desc.t) and torch.equal(n.desc.R, c.desc.R)
    # near_tol above face_tol: a row between them is canonical, flagged, and nudged off the mirror face
    row = _ortho(19, [(5., 5 * (1 + 5e-6), 9.)], g)
    kw = dict(face_tol=1e-6, near_tol=2e-5)
    c1 = canon.canonicalize(row, **kw)
    assert c1.names() == ['canonical'] and int(c1.near_face[0]) & canon.NEAR_CELL
    n1 = canon.canonicalize(row, nudge=DELTA, **kw)
    assert bool(n1.nudged[0]) and int(n1.near_face[0]) == 0
    assert abs(float(n1.nudge_cell[0]) - (DELTA - math.log(1 + 5e-6)) / math.sqrt(2)) <= 1e-12
    assert float(axis_order.face_values(19, n1.desc.cell).min()) >= DELTA * (1 - 1e-9)


def _seam_rows(g):
    x, y, eps = 6.0, 9.0, 1e-6
    abc = [(x, x, y), (x, x * (1 + eps), y), (x, y, x), (x, y, x * (1 + eps)), (y, x, x * (1 + eps))]
    return _ortho(61, abc, g), abc


def test_pbca_seams_are_flagged_and_nudged():
    g = torch.Generator().manual_seed(4)
    d, abc = _seam_rows(g)
    c = canon.canonicalize(d)
    assert c.names() == ['canonical'] * len(abc) and bool((c.n_auto == 1).all())
    assert bool((c.near_face.long() & canon.NEAR_CELL != 0).all())
    assert c.n_tied.tolist() == [2, 1, 2, 1, 1]
    assert bool(_faces_ok(61, c.desc.cell).all())
    assert c.desc.cell[0, :3].tolist() == [6.0, 6.0, 9.0] and c.desc.cell[2, :3].tolist() == [6.0, 6.0, 9.0]
    # every cyclic order of a seam row (exact and 1e-6 off) canonicalises to one row
    o = axis_order.axis_orders(61)
    for j in range(o.k):
        for N, s in _variants(61, j):
            cj = canon.canonicalize(_reorder(d, N, s))
            assert float(_deviation(cj.desc, c.desc).max()) <= 1e-12, (j, N.tolist())
    # the nudge: the shortest log-length move to delta from both faces; t and R kept; a fixed point
    n = canon.canonicalize(d, nudge=DELTA)
    assert bool(n.nudged.all()) and all(m == '' for m in n.nudge_message) and bool((n.near_face == 0).all())
    assert torch.equal(n.desc.t, c.desc.t) and torch.equal(n.desc.R, c.desc.R) and bool((n.nudge_t == 0).all())
    fv = axis_order.face_values(61, n.desc.cell)
    assert float(fv.min()) >= DELTA * (1 - 1e-9)
    start = axis_order.face_values(61, c.desc.cell).amin(1)
    assert torch.allclose(n.nudge_cell, (DELTA - start) / math.sqrt(2), rtol=0, atol=1e-12)
    assert bool(n.moved.all()) and torch.equal(n.status, c.status) and torch.equal(n.N, c.N)
    again = canon.canonicalize(n.desc)
    assert float(_deviation(again.desc, n.desc).max()) <= 1e-12 and not bool(again.moved.any())
    assert bool((again.near_face == 0).all()) and bool((again.n_tied == 1).all())
    twice = canon.canonicalize(n.desc, nudge=DELTA)
    assert not bool(twice.nudged.any()) and float(_deviation(twice.desc, n.desc).max()) <= 1e-12
    st = canon.fd_status(n.desc, nudge=DELTA)
    assert bool(st.is_fd.all()) and not bool(canon.fd_status(c.desc, nudge=DELTA).is_fd.any())
    # float32 re-descriptions (a cyclic order with a sign choice, a Gamma+ image, a cast) of the nudged rows
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(10):
        j = int(rng.integers(o.k))
        N, s = _variants(61, j)[int(rng.integers(4))]
        x = _gamma_image(_reorder(n.desc, N, s), rng).to(dtype=torch.float32)
        back = canon.canonicalize(x)
        assert back.names() == ['canonical'] * d.num_rows and bool((back.near_face == 0).all())
        worst = max(worst, float(_deviation(back.desc, n.desc).max()))
    assert worst <= 1e-5


def test_pbca_rows_within_the_tie_tolerance_need_the_nudge():
    """A Pbca row with a and b 4e-11 apart (inside the tie tolerance 1e-10, so the tie-break puts a first although a >
    b) on either side of a float32 rounding boundary: cast to float32, a rounds up and b down, the order is strict and
    the shortest axis b comes first, another cyclic order (rounding is monotone, so this is the only way a seam row
    flips). Nudged, the row comes back from every float32 re-description."""
    g = torch.Generator().manual_seed(5)
    m = 6.0 + 2.0 ** -22                                       # the midpoint of 6 and the next float32
    a, b = m * (1 + 2e-11), m * (1 - 2e-11)
    assert float(torch.tensor(a).float()) > float(torch.tensor(b).float())
    d = _ortho(61, [(a, b, 9.0)] * 4, g)
    c = canon.canonicalize(d)
    assert c.names() == ['canonical'] * 4 and bool((c.desc.cell[:, 0] > c.desc.cell[:, 1]).all())
    assert bool((c.n_tied == 2).all()) and bool((c.near_face.long() & canon.NEAR_CELL != 0).all())
    n = canon.canonicalize(d, nudge=DELTA)
    assert bool(n.nudged.all())
    rng = np.random.default_rng(5)
    o = axis_order.axis_orders(61)
    flips, worst = 0, 0.0
    for _ in range(6):
        j = int(rng.integers(o.k))
        N, s = _variants(61, j)[int(rng.integers(4))]
        x0 = _gamma_image(_reorder(c.desc, N, s), rng).to(dtype=torch.float32)
        flips += int((_deviation(canon.canonicalize(x0).desc, c.desc) > 1e-3).sum())
        xn = _gamma_image(_reorder(n.desc, N, s), rng).to(dtype=torch.float32)
        worst = max(worst, float(_deviation(canon.canonicalize(xn).desc, n.desc).max()))
    assert flips == 24 and worst <= 1e-5, (flips, worst)


def test_pbca_cone_is_a_special_metric():
    g = torch.Generator().manual_seed(6)
    d = _ortho(61, [(7.0, 7.0, 7.0), (7.0, 7.0 * (1 + 3e-6), 7.0 * (1 - 3e-6)), (7.0, 7.0, 7.0 * (1 + 3e-5))], g)
    c = canon.canonicalize(d)
    assert c.names() == ['special_metric', 'special_metric', 'canonical'] and c.n_auto.tolist() == [3, 3, 1]
    n = canon.canonicalize(d, nudge=DELTA)
    assert n.nudged.tolist() == [False, False, True]
    assert torch.equal(n.desc.cell[:2], c.desc.cell[:2]) and torch.equal(n.desc.R[:2], c.desc.R[:2])


def _plain_n_auto(sg, cell, face_tol):
    """([b] orders within face_tol, [b] order of the subgroup they generate) of canonical cells, written plainly: the
    allowed orders as permutation tuples whose lengths match the cell's within face_tol in log length, closed by
    composing pairs until nothing is added."""
    perms = [tuple(p) for p in axis_order.axis_orders(sg).perm.tolist()]
    raw, closed = [], []
    for y in torch.log(cell[:, :3]).tolist():
        S = {p for p in perms if max(abs(y[p[i]] - y[i]) for i in range(3)) <= face_tol}
        raw.append(len(S))
        while True:
            T = S | {tuple(p[q[i]] for i in range(3)) for p in S for q in S}
            if T == S:
                break
            S = T
        closed.append(len(S))
    return raw, closed


def test_n_auto_is_the_order_of_a_subgroup():
    """A chain of near-ties (b - a = c - b = 6e-6, c - a = 1.2e-5, face_tol 1e-5): the transpositions are within
    face_tol and the 3-cycles are not. They generate all six orders (n_auto 6, not the 3 of a plain count); in a <-> b
    only the a = b mirror counts (2); in Pbca no 3-cycle is within face_tol, so the row is canonical with NEAR_CELL.
    Random chains in every class: n_auto is the plain subgroup order, a divisor of k, and special_metric iff > 1."""
    g = torch.Generator().manual_seed(11)
    ft = ops.FACE_TOL
    chain = (6.0, 6.0 * (1 + 6e-6), 6.0 * (1 + 1.2e-5))
    named = [(19, chain, 'special_metric', 6), (16, chain[::-1], 'special_metric', 6),
             (22, (chain[1], chain[2], chain[0]), 'special_metric', 6), (18, chain, 'special_metric', 2),
             (61, chain, 'canonical', 1), (33, chain, 'canonical', 1)]
    d = CrystalDescription.cat([_ortho(sg, [abc], g) for sg, abc, *_ in named])
    c = canon.canonicalize(d)
    assert c.names() == [x[2] for x in named] and c.n_auto.tolist() == [x[3] for x in named]
    assert int(c.near_face[4]) & canon.NEAR_CELL
    raw, _ = _plain_n_auto(19, c.desc.cell[:1], ft)
    assert raw == [3]                                          # the plain count the subgroup replaces
    # random chains: gaps in log length below, near and above face_tol, in random input order
    gaps = torch.tensor([0.0, 0.3, 0.6, 0.9, 1.2, 2.0, 1e4], dtype=F64) * ft
    chains_seen = 0
    for sg in (19, 22, 18, 56, 61, 33):
        n = 80
        y0 = torch.log(4 + 8 * torch.rand(n, generator=g, dtype=F64))
        d1 = gaps[torch.randint(len(gaps), (n,), generator=g)]
        d2 = gaps[torch.randint(len(gaps), (n,), generator=g)]
        L = torch.exp(torch.stack([y0, y0 + d1, y0 + d1 + d2], 1))
        L = L[torch.arange(n)[:, None], torch.argsort(torch.rand(n, 3, generator=g), 1)]
        c = canon.canonicalize(_ortho(sg, L.tolist(), g))
        raw, closed = _plain_n_auto(sg, c.desc.cell, ft)
        k = axis_order.axis_orders(sg).k
        assert c.n_auto.tolist() == closed, sg
        assert all(k % m == 0 for m in closed), sg
        assert c.names() == ['special_metric' if m > 1 else 'canonical' for m in closed], sg
        chains_seen += sum(1 for r, m in zip(raw, closed) if r != m)
        if k != 6:
            assert raw == closed, sg                           # a <-> b and Pbca: already subgroups
    assert chains_seen >= 5, chains_seen


# ---------------------------------------------------------------------------
# 4: groups 75-230: every Gamma image to one row; the snap
# ---------------------------------------------------------------------------

def _gamma_images_and_snap(sg, chiral_molecule, literal_oracle, n=3):
    g = torch.Generator().manual_seed(1000 + sg)
    rng = np.random.default_rng(1000 + sg)
    tab = group_table(sg)
    d = _rows(sg, n, g)
    c0 = canon.canonicalize(d)
    assert c0.names() == ['canonical'] * n and bool((c0.n_auto == 1).all()), (sg, c0.names())
    assert bool((c0.N == torch.eye(3, dtype=torch.long)).all()) and float(c0.s.abs().max()) == 0
    assert float(c0.cell_snap.max()) <= 1e-14
    assert float(canon._metric_deviation(c0.desc.sg, c0.desc.cell).max()) <= 1e-14
    sohncke = io_mxt.is_sohncke(sg)
    for k in range(len(tab.gamma_W)):
        e = tab.gamma_W[k].expand(n, 3, 3)
        w = tab.gamma_w[k] + torch.as_tensor(rng.integers(-1, 2, (n, 3)), dtype=F64) \
            + torch.as_tensor(rng.random((n, 3)), dtype=F64) * tab.free.to(F64)
        x = ops.apply(d, e, w)
        c = canon.canonicalize(x)
        if int(tab.gamma_det[k]) == 1 or not sohncke:
            assert c.names() == ['canonical'] * n, (sg, k, c.names())
            assert float(_deviation(c.desc, c0.desc).max()) <= 1e-10, (sg, k)
        else:                                   # the mirror image of a Sohncke crystal: off the sheet, one row too
            assert c.names() == ['off_sheet'] * n, (sg, k)
            if k == int(torch.nonzero(tab.gamma_det < 0)[0, 0]):
                first = c.desc
            assert float(_deviation(c.desc, first).max()) <= 1e-10, (sg, k)
    # the snap: a cell about 1e-7 off the exact metric
    noise = torch.cat([1 + 1e-7 * torch.randn(n, 3, generator=g, dtype=F64),
                       1e-7 * torch.randn(n, 3, generator=g, dtype=F64)], 1)
    off = d.replace(cell=torch.cat([d.cell[:, :3] * noise[:, :3], d.cell[:, 3:] + noise[:, 3:]], 1))
    c = canon.canonicalize(off)
    assert c.names() == ['canonical'] * n
    assert float(canon._metric_deviation(c.desc.sg, c.desc.cell).max()) <= 1e-14
    assert bool((c.cell_snap > 1e-9).all() & (c.cell_snap < 1e-6).all()), c.cell_snap
    assert float(_deviation(c.desc, c0.desc).max()) <= 1e-5
    X = chiral_molecule['pos']
    for i in range(n):                          # the snapped crystal (input pose, snapped cell) against the input
        A, _ = _cartesian_cell(off[i:i + 1], X, literal_oracle)
        B, _ = _cartesian_cell(off[i:i + 1].replace(cell=c.desc.cell[i:i + 1]), X, literal_oracle)
        L = float(off.cell[i, :3].max())
        assert float(np.abs(A - B).max()) <= 10 * float(c.cell_snap[i]) * (L + 5.0), (sg, i)


@pytest.mark.parametrize('sg', HIGH_FAST)
def test_higher_groups_gamma_images_and_snap(sg, chiral_molecule, literal_oracle):
    _gamma_images_and_snap(sg, chiral_molecule, literal_oracle)


@pytest.mark.slow
@pytest.mark.parametrize('sg', HIGH_SLOW)
def test_higher_groups_gamma_images_and_snap_all_groups(sg, chiral_molecule, literal_oracle):
    _gamma_images_and_snap(sg, chiral_molecule, literal_oracle, n=2)


def test_snap_is_recorded_and_bounded():
    """cell_snap is the largest change of a length (relative) or angle; a cell off by more than metric_tol is not
    snapped but cell_off_group; sg 1-15 rows have none."""
    g = torch.Generator().manual_seed(7)
    d = CrystalDescription.cat([_rows(92, 1, g), _rows(19, 1, g), _rows(2, 1, g)])
    cell = d.cell.clone()
    cell[0, 1] = cell[0, 0] * (1 + 3e-6)                     # tetragonal a != b by 3e-6
    cell[1, 4] = math.pi / 2 + 2e-6                          # orthorhombic beta off by 2e-6 rad
    c = canon.canonicalize(d.replace(cell=cell))
    assert c.names() == ['canonical'] * 3
    assert abs(float(c.cell_snap[0]) - 1.5e-6) <= 1e-9 and abs(float(c.cell_snap[1]) - 2e-6) <= 1e-12
    assert math.isnan(float(c.cell_snap[2]))
    assert float((c.desc.cell[0, :2] - math.sqrt((cell[0, 0] ** 2 + cell[0, 1] ** 2) / 2)).abs().max()) <= 1e-12
    cell[0, 1] = cell[0, 0] * 1.01
    c = canon.canonicalize(d.replace(cell=cell))
    assert c.names()[0] == 'cell_off_group' and math.isnan(float(c.cell_snap[0]))


# ---------------------------------------------------------------------------
# 5: float32 re-descriptions of every class
# ---------------------------------------------------------------------------

def test_float32_redescriptions_land_on_the_canonical_row():
    g = torch.Generator().manual_seed(8)
    rng = np.random.default_rng(8)
    groups = ORTHO_FAST + (92, 148, 167, 178, 194, 227)
    d = CrystalDescription.cat([_rows(sg, 6, g) for sg in groups])
    c0 = canon.canonicalize(d)
    flagged = (c0.status.long() != canon.STATUS.index('canonical')) | (c0.near_face != 0)
    assert int(flagged.sum()) <= 2
    worst = 0.0
    for _ in range(6):
        parts = []
        for i in range(d.num_rows):
            r = c0.desc[i:i + 1]
            sg = int(r.sg[0])
            if sg in axis_order.ORTHORHOMBIC:
                o = axis_order.axis_orders(sg)
                N, s = _variants(sg, int(rng.integers(o.k)))[int(rng.integers(4))]
                r = _reorder(r, N, s)
            parts.append(_gamma_image(r, rng))
        x = CrystalDescription.cat(parts).to(dtype=torch.float32)
        back = canon.canonicalize(x)
        ok = ~flagged
        assert [n for n, k in zip(back.names(), ok.tolist()) if k] == ['canonical'] * int(ok.sum())
        worst = max(worst, float(_deviation(back.desc[ok], c0.desc[ok]).max()))
    assert worst <= 1e-5, worst


# ---------------------------------------------------------------------------
# 6: isolation; sg 6-9 unchanged
# ---------------------------------------------------------------------------

def test_a_row_the_new_step_raises_on_is_isolated(monkeypatch):
    g = torch.Generator().manual_seed(9)
    d = CrystalDescription.cat([_rows(2, 2, g), _rows(19, 3, g), _rows(148, 2, g), _rows(61, 2, g)])
    clean = canon.canonicalize(d)
    target = 3
    sentinel = float(d.cell[target, 0])
    real = canon._high_cell

    def raising(x, *a, **kw):
        if bool((x.cell[:, 0] == sentinel).any()):
            raise RuntimeError('sentinel')
        return real(x, *a, **kw)
    monkeypatch.setattr(canon, '_high_cell', raising)
    c = canon.canonicalize(d)
    monkeypatch.setattr(canon, '_high_cell', real)
    assert c.names()[target] == 'reduce_failed' and c.message[target].startswith('cell step (sg >= 16): ')
    assert 'sentinel' in c.message[target] and torch.equal(c.desc.cell[target], d.cell[target])
    others = [i for i in range(d.num_rows) if i != target]
    assert [c.names()[i] for i in others] == [clean.names()[i] for i in others]
    for f in ('cell', 't', 'R', 'h'):
        assert torch.equal(getattr(c.desc, f)[others], getattr(clean.desc, f)[others]), f
    for f in ('N', 's', 'n_auto', 'n_tied', 'violation', 'cell_snap', 'element', 'tie', 'near_face'):
        a, b = getattr(c, f)[others], getattr(clean, f)[others]
        assert torch.equal(torch.nan_to_num(a.double(), nan=7.0), torch.nan_to_num(b.double(), nan=7.0)), f
    assert math.isnan(float(c.cell_snap[target])) and int(c.n_auto[target]) == -1


@pytest.mark.parametrize('sheet', [1, -1, None])
def test_sg_6_to_9_keep_their_behaviour(sheet):
    """Rows of sg 6-9 (nontrivial K), random and on special metrics (beta = 90 deg), at both handedness: statuses as the
    rule gives them (off_sheet, special_metric, canonical) and every row reduce_cell, to_sheet and fold run on it alone
    (within 1e-12: the batched reduction rounds otherwise than a single row, as before 2026-10-21; the bit-for-bit
    comparison with the code before the sg >= 16 cell step was made outside the suite, and
    test_canon.test_rows_of_sg_1_15_do_not_see_the_higher_cell_step pins that the new step does not reach them)."""
    g = torch.Generator().manual_seed(10)
    parts = []
    for sg in (6, 7, 8, 9):
        r = _rows(sg, 6, g)
        cell = r.cell.clone()
        cell[0, 4] = math.pi / 2
        parts.append(r.replace(cell=cell, h=torch.tensor([[1], [-1], [1], [-1], [1], [-1]])))
    d = CrystalDescription.cat(parts)
    c = canon.canonicalize(d, sheet=sheet)
    for i in range(d.num_rows):
        red = ops.reduce_cell(d[i:i + 1])
        x = red.desc
        ok = True
        if sheet is not None:
            x, ok_t = domains.to_sheet(x, sheet)
            ok = bool(ok_t[0])
        f = domains.fold(x).desc
        want = 'off_sheet' if not ok else ('special_metric' if int(red.n_auto[0]) > 1 else 'canonical')
        assert c.names()[i] == want, i
        assert float(_deviation(c.desc[i:i + 1], f).max()) <= 1e-12, i
        assert torch.equal(c.N[i], red.N[0]) and int(c.n_auto[i]) == int(red.n_auto[0])
        assert math.isnan(float(c.cell_snap[i]))
    assert 'special_metric' in c.names() and 'canonical' in c.names()

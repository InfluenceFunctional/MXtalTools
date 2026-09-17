"""
Monoclinic cell-reduction walls: mxtaltools/common/sym_utils.py::mono_reduction_penalty, via cell_reduction_penalty.

Contract. For sg 3-15 the SYM_OPS operator list is kept by a subgroup of the ac-plane basis changes
a' = p a + q c, c' = r a + s c (det +-1, b' = det b): the sg's SETTING GROUP. At margin 0 the penalty is 0 on exactly
one cell per lattice within that group (counted up to the sign of the basis, which leaves a, c, beta unchanged), and
that cell is spglib's standard cell. Non-monoclinic rows are pinned bit for bit to the formulas written out below.

The setting groups are derived here from SYM_OPS, not read from sym_utils.MONO_CLASS. SYM_OPS translations for
sg 3-15 are multiples of 1/2 and the rotation parts are diagonal, so whether a basis change keeps the operator list
depends only on (p, q, r, s) mod 2; the preconditions are asserted and the reduction is cross-checked on every matrix
with |entries| <= 2.

CPU only.
"""
import functools
import itertools
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mxtaltools.common.sym_utils import (MONO_CENTRED, MONO_CLASS, MONO_FREE, MONO_GLIDE, cell_reduction_penalty,
                                         mono_reduction_penalty)
from mxtaltools.constants.space_group_info import SYM_OPS

MONO_SGS = tuple(range(3, 16))
HALF_PI = torch.pi / 2
CLASSES = ('GLIDE', 'CENTRED', 'FREE')


# ---------------------------------------------------------------------------
# Setting groups from SYM_OPS
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _mats(k):
    """(p, q, r, s) with |entries| <= k and det +-1."""
    return tuple(m for m in itertools.product(range(-k, k + 1), repeat=4) if abs(m[0] * m[3] - m[1] * m[2]) == 1)


def _res(m):
    return tuple(x % 2 for x in m)


_SHIFTS8 = np.array(list(itertools.product(range(8), repeat=3)))  # origin shifts on the 1/8 grid, in eighths


def _op_codes(w_diag, t8):
    """one integer per operator: diagonal rotation signs and translation in eighths (mod 1)"""
    return ((w_diag + 1) // 2) @ np.array([4, 2, 1]) * 512 + (t8 % 8) @ np.array([64, 8, 1])


def keeps_sym_ops(sg, p, q, r, s):
    """True if the basis change keeps the SYM_OPS operator list of sg, up to an origin shift on the 1/8 grid."""
    ops = np.array(SYM_OPS[sg], dtype=np.float64)
    rot, trans = ops[:, :3, :3], ops[:, :3, 3]
    d = p * s - q * r
    B = np.array([[p, 0, r], [0, d, 0], [q, 0, s]], dtype=np.float64)  # columns: new a, b, c in old fractional coords
    Bi = np.linalg.inv(B)
    assert np.allclose(np.einsum('ij,njk,kl->nil', Bi, rot, B), rot)  # rotation parts unchanged
    w_diag = np.rint(np.diagonal(rot, axis1=1, axis2=2)).astype(int)
    assert np.allclose(rot, np.eye(3) * w_diag[:, None, :])  # diagonal rotation parts
    ref8 = np.rint(8 * trans).astype(int)
    assert np.allclose(8 * trans, ref8) and (ref8 % 4 == 0).all()  # translations are multiples of 1/2
    new8 = np.rint(8 * (trans @ Bi.T)).astype(int)
    shifted = new8[None] + (w_diag - 1)[None] * _SHIFTS8[:, None, :]  # t + (W - I) o
    codes = np.sort(_op_codes(w_diag, shifted), axis=1)
    return bool((codes == np.sort(_op_codes(w_diag, ref8))[None]).all(axis=1).any())


@functools.lru_cache(maxsize=None)
def setting_group_residues(sg):
    """GL(2, Z/2) residues of the basis changes that keep SYM_OPS[sg]."""
    reps = {}
    for m in _mats(1):
        reps.setdefault(_res(m), m)
    assert len(reps) == 6
    return frozenset(res for res, m in reps.items() if keeps_sym_ops(sg, *m))


@functools.lru_cache(maxsize=None)
def setting_class(sg):
    res = setting_group_residues(sg)
    gl2 = {_res(m) for m in _mats(1)}
    if res == gl2:
        return 'FREE'
    if res == {m for m in gl2 if m[2] == 0}:
        return 'GLIDE'  # r even
    if res == {m for m in gl2 if m[1] == 0}:
        return 'CENTRED'  # q even
    raise AssertionError(f'sg {sg}: unexpected setting group {sorted(res)}')


def class_sgs(name):
    return tuple(sg for sg in MONO_SGS if setting_class(sg) == name)


# ---------------------------------------------------------------------------
# Hard walls and cells
# ---------------------------------------------------------------------------

def hard_domain(name, a, c, cb):
    """closed class domain in Gram form (D = a c cos beta), independent of the penalty's algebra"""
    A2, C2, D = a * a, c * c, a * c * cb
    if name == 'GLIDE':
        return (D <= 0) & (-D <= A2) & (-2 * D <= C2)
    if name == 'CENTRED':
        return (D <= 0) & (-D <= C2) & (-2 * D <= A2)
    return (D <= 0) & (A2 <= C2) & (-2 * D <= A2)


def rel_slack(name, a, c, cb):
    """distance to the nearest wall, relative"""
    s1 = -cb
    if name == 'GLIDE':
        s2, s3 = 1 - c * cb.abs() / a, 1 - 2 * a * cb.abs() / c
    elif name == 'CENTRED':
        s2, s3 = 1 - a * cb.abs() / c, 1 - 2 * c * cb.abs() / a
    else:
        s2, s3 = 1 - a / c, 1 - 2 * c * cb.abs() / a
    return torch.stack([s1, s2, s3]).abs().min(0).values


def random_cells(n, sgs, gen, dtype=torch.float64, lo=0.5, hi=200.0, exact_angles=True):
    """exact_angles=False: alpha and gamma drawn independently in [0.3 pi, 0.7 pi] on every second row"""
    logu = lambda: torch.exp(torch.empty(n, dtype=torch.float64).uniform_(math.log(lo), math.log(hi), generator=gen))
    a, b, c = logu(), logu(), logu()
    be = torch.empty(n, dtype=torch.float64).uniform_(0.02, math.pi - 0.02, generator=gen)
    sg = torch.tensor(sgs, dtype=torch.long)[torch.randint(len(sgs), (n,), generator=gen)]
    L = torch.stack([a, b, c], 1).to(dtype)
    al, ga = torch.full_like(be, HALF_PI), torch.full_like(be, HALF_PI)
    if not exact_angles:
        off = torch.arange(n) % 2 == 1
        al[off] = torch.empty(int(off.sum()), dtype=torch.float64).uniform_(0.3 * math.pi, 0.7 * math.pi, generator=gen)
        ga[off] = torch.empty(int(off.sum()), dtype=torch.float64).uniform_(0.3 * math.pi, 0.7 * math.pi, generator=gen)
    A = torch.stack([al, be, ga], 1).to(dtype)
    return L, A, sg


def bounding(x, lower, upper, margin):
    return (F.relu(x - (upper - margin)) ** 2) + (F.relu((lower + margin) - x) ** 2)


def lambda1(a, c, cb):
    """shortest lattice vector length, Lagrange-Gauss reduction on the Gram entries"""
    g11, g22, g12 = a * a, c * c, a * c * cb
    for _ in range(200):
        swap = g22 < g11
        g11, g22 = torch.where(swap, g22, g11), torch.where(swap, g11, g22)
        k = torch.round(g12 / g11)
        if not swap.any() and (k == 0).all():
            return g11.sqrt()
        g22 = g22 - 2 * k * g12 + k * k * g11
        g12 = g12 - k * g11
    raise AssertionError('Gauss reduction did not converge')


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_class_table_matches_setting_groups_from_sym_ops():
    names = {MONO_GLIDE: 'GLIDE', MONO_CENTRED: 'CENTRED', MONO_FREE: 'FREE'}
    for sg in MONO_SGS:
        assert names[int(MONO_CLASS[sg])] == setting_class(sg), sg
        res = setting_group_residues(sg)
        for m in _mats(2):  # the mod-2 reduction holds on every matrix with |entries| <= 2
            assert keeps_sym_ops(sg, *m) == (_res(m) in res), (sg, m)
    assert (MONO_CLASS[[s for s in range(231) if not 3 <= s <= 15]] == -1).all()
    assert {name: len(class_sgs(name)) for name in CLASSES} == {'GLIDE': 3, 'CENTRED': 5, 'FREE': 5}


@pytest.mark.parametrize('sg', MONO_SGS)
def test_exactly_one_zero_penalty_cell_per_lattice(sg):
    """Random lattices, each written in a random basis taken as its SYM_OPS-setting cell. Every setting-group basis
    change with |entries| <= K is scored at margin 0 (float64); exactly one must score 0.
    Bound: every zero-penalty cell has sin^2(beta) >= 1/2 (max beta 135 deg, test_zero_set_is_the_class_domain), so
    |a'|, |c'| <= sqrt2 * area / lambda1 and every coefficient is <= sqrt2 * max(a, c) / lambda1; asserted <= K."""
    K, n = 10, 4000
    gen = torch.Generator().manual_seed(1000 + sg)
    s = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(math.log(0.1), math.log(100), generator=gen))
    ratio = lambda: torch.exp(torch.empty(n, dtype=torch.float64).uniform_(0, math.log(6), generator=gen))
    a, b, c = s * ratio(), s * ratio(), s * ratio()
    cb = torch.cos(torch.deg2rad(torch.empty(n, dtype=torch.float64).uniform_(15, 165, generator=gen)))
    k_needed = torch.floor(math.sqrt(2) * torch.maximum(a, c) / lambda1(a, c, cb) * (1 + 1e-9))
    assert (k_needed <= K).all(), int(k_needed.max())

    res = setting_group_residues(sg)
    mats = torch.tensor([m for m in _mats(K) if _res(m) in res and (m[0] > 0 or (m[0] == 0 and m[1] > 0))],
                        dtype=torch.float64)  # one of +-M: same a, c, beta
    p, q, r, t = (x[None, :] for x in mats.T)
    counts = []
    for i in range(0, n, 500):
        A2, C2, D = (x[i:i + 500, None] for x in (a * a, c * c, a * c * cb))
        ga = p * p * A2 + 2 * p * q * D + q * q * C2
        gc = r * r * A2 + 2 * r * t * D + t * t * C2
        gac = p * r * A2 + (p * t + q * r) * D + q * t * C2
        beta = torch.arccos((gac / (ga * gc).sqrt()).clamp(-1, 1))
        L = torch.stack([ga.sqrt(), b[i:i + 500, None].expand_as(ga), gc.sqrt()], -1).reshape(-1, 3)
        A = torch.stack([torch.full_like(beta, HALF_PI), beta, torch.full_like(beta, HALF_PI)], -1).reshape(-1, 3)
        E = mono_reduction_penalty(L, A, torch.full((len(L),), sg, dtype=torch.long), 0.0)
        counts.append((E.reshape(ga.shape) == 0).sum(1))
    counts = torch.cat(counts)
    assert len(counts) == n
    assert (counts == 1).all(), dict(zip(*[x.tolist() for x in torch.unique(counts, return_counts=True)]))


@pytest.mark.parametrize('margin', [0.0, 0.1])
def test_non_monoclinic_rows_bit_identical(margin):
    """float32 rows over every sg; non-monoclinic rows must equal the per-system formulas bit for bit."""
    gen = torch.Generator().manual_seed(1)
    n = 20000
    L = torch.exp(torch.empty(n, 3).uniform_(0, 4, generator=gen))
    A = torch.empty(n, 3).uniform_(0.2 * math.pi, 0.8 * math.pi, generator=gen)
    sg = torch.randint(1, 231, (n,), generator=gen)
    A[(torch.arange(n) % 2 == 0) & (sg >= 3)] = HALF_PI  # crystal-system-exact angles on half the rows
    E = cell_reduction_penalty(A, L, sg, margin)
    assert E.dtype == torch.float32 and E.shape == (n,)

    ref = torch.full((n,), float('nan'))
    # triclinic: Niggli main conditions, beta/gamma-obtuse convention (tri_niggli_reduction_penalty written out)
    m = (sg == 1) | (sg == 2)
    a, b, c = L[m].clamp(min=1e-6).unbind(1)
    al, be, ga = A[m].unbind(1)
    Asq, Bsq = a.square(), b.square()
    xi, eta, zeta = 2 * b * c * al.cos(), 2 * a * c * be.cos(), 2 * a * b * ga.cos()
    q = torch.stack((xi / Bsq, eta / Asq, zeta / Asq), dim=-1)
    ref[m] = (F.relu(a / b - (1 - margin)).square() + F.relu(b / c - (1 - margin)).square()
              + F.relu(q.abs() - (1 - margin)).square().sum(dim=-1)
              + (F.relu(eta / Asq).square() + F.relu(zeta / Asq).square())
              + F.relu(margin - (Asq + Bsq + xi + eta + zeta) / (Asq + Bsq)).square())
    right = lambda x: (x - HALF_PI) ** 2
    # orthorhombic
    m = (sg >= 16) & (sg <= 74)
    ref[m] = right(A[m, 0]) + right(A[m, 1]) + right(A[m, 2])
    # tetragonal
    m = (sg >= 75) & (sg <= 142)
    ref[m] = (L[m, 0] - L[m, 1]) ** 2 + right(A[m, 0]) + right(A[m, 1]) + right(A[m, 2])
    # trigonal and hexagonal
    m = (sg >= 143) & (sg <= 194)
    ref[m] = (L[m, 0] - L[m, 1]) ** 2 + right(A[m, 0]) + right(A[m, 1]) + (A[m, 2] - 2 * torch.pi / 3) ** 2
    # cubic
    m = sg >= 195
    ref[m] = (L[m, 0] - L[m, 1]) ** 2 + (L[m, 1] - L[m, 2]) ** 2 + right(A[m, 0]) + right(A[m, 1]) + right(A[m, 2])

    nonmono = ~((sg >= 3) & (sg <= 15))
    assert nonmono.sum() > 15000
    assert torch.equal(E[nonmono], ref[nonmono])


def test_cell_reduction_penalty_passes_each_rows_sg():
    """monoclinic rows interleaved with other systems get the walls of their own sg"""
    gen = torch.Generator().manual_seed(9)
    L, A, sg = random_cells(20000, tuple(range(1, 40)), gen, dtype=torch.float32)
    mono = (sg >= 3) & (sg <= 15)
    E = cell_reduction_penalty(A, L, sg, 0.0)
    assert torch.equal(E[mono], mono_reduction_penalty(L[mono], A[mono], sg[mono], 0.0))
    shuffled = sg[mono][torch.randperm(int(mono.sum()), generator=gen)]
    assert not torch.equal(E[mono], mono_reduction_penalty(L[mono], A[mono], shuffled, 0.0))  # control


@pytest.mark.parametrize('name', CLASSES)
def test_zero_set_is_the_class_domain(name):
    gen = torch.Generator().manual_seed(2)
    L, A, sg = random_cells(200000, class_sgs(name), gen)
    E = mono_reduction_penalty(L, A, sg, 0.0)
    a, c, cb = L[:, 0], L[:, 2], torch.cos(A[:, 1])
    inside = hard_domain(name, a, c, cb)
    disagree = (E == 0) != inside
    assert inside.float().mean() > 0.005 and (~inside).float().mean() > 0.5
    assert (rel_slack(name, a, c, cb)[disagree] < 1e-12).all(), int(disagree.sum())
    max_beta = {'GLIDE': 135.0, 'CENTRED': 135.0, 'FREE': 120.0}[name]
    assert torch.rad2deg(A[E == 0, 1]).max() <= max_beta + 1e-9


@pytest.mark.parametrize('margin', [0.0, 0.1])
def test_explicit_class_formulas(margin):
    """sg 14 (GLIDE) is W1 + W2 + W3 as one bounded cos(beta); CENTRED and FREE likewise; bit for bit. alpha and gamma
    are off 90 deg on half the rows, so the alpha = gamma = 90 pins are part of the comparison."""
    gen = torch.Generator().manual_seed(4)
    for name in CLASSES:
        L, A, sg = random_cells(50000, class_sgs(name), gen, dtype=torch.float32, exact_angles=False)
        assert ((A[:, 0] - A[:, 2]).abs() > 1e-3).sum() > 20000  # alpha, gamma off 90 and different from each other
        a, c, cb = L[:, 0], L[:, 2], torch.cos(A[:, 1])
        if name == 'GLIDE':
            ref = bounding(cb, (-torch.minimum(a / c, c / (2 * a))).clamp(min=-1, max=0), 0, margin)
        elif name == 'CENTRED':
            ref = bounding(cb, (-torch.minimum(c / a, a / (2 * c))).clamp(min=-1, max=0), 0, margin)
        else:
            ref = bounding(cb, (-(a / (2 * c))).clamp(min=-1, max=0), 0, margin) + F.relu(a / c - (1 - margin)) ** 2
        ref = ref + (A[:, 0] - HALF_PI) ** 2 + (A[:, 2] - HALF_PI) ** 2
        assert torch.equal(mono_reduction_penalty(L, A, sg, margin), ref), name
    assert 14 in class_sgs('GLIDE')


@pytest.mark.parametrize('margin', [0.0, 0.05, 0.1])
def test_never_below_the_w1_w2_walls_and_equal_where_w2_binds(margin):
    """the previous walls (every sg: cos(beta) in [-a/c, 0]) bound the new penalty from below, for a, c >= 1e-6 (the
    eps clamp; lengths here are >= 0.5)"""
    gen = torch.Generator().manual_seed(3)
    L, A, sg = random_cells(100000, MONO_SGS, gen, dtype=torch.float32)
    a, c = L[:, 0], L[:, 2]
    old = bounding(torch.cos(A[:, 1]), (-a / c).clamp(min=-1), 0, margin) + (A[:, 0] - HALF_PI) ** 2 + (A[:, 2] - HALF_PI) ** 2
    new = mono_reduction_penalty(L, A, sg, margin)
    assert (new >= old).all()
    w2_binds = torch.isin(sg, torch.tensor(class_sgs('GLIDE'))) & (a / c <= c / (2 * a))
    assert w2_binds.sum() > 1000
    assert torch.equal(new[w2_binds], old[w2_binds])


def test_margin_zero_set_is_inside_the_domain():
    gen = torch.Generator().manual_seed(6)
    for name in CLASSES:
        L, A, sg = random_cells(100000, class_sgs(name), gen)
        E = mono_reduction_penalty(L, A, sg, 0.05)
        inside = hard_domain(name, L[:, 0], L[:, 2], torch.cos(A[:, 1]))
        assert (E == 0).sum() > 100
        assert inside[E == 0].all(), name


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('margin', [0.0, 0.1])
def test_values_and_gradients_finite_at_extremes_and_ties(dtype, margin):
    rows = []
    for a, c in itertools.product([0.0, 1e-8, 1e-3, 1.0, 7.0, 1e3, 1e8], repeat=2):
        for be in [0.0, 1e-7, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi - 1e-7, math.pi]:
            rows.append((a, c, be))
    for a in [0.5, 3.0, 20.0]:
        rows += [(a, math.sqrt(2) * a, 0.75 * math.pi),  # GLIDE corner: a/c = c/(2a), beta 135
                 (math.sqrt(2) * a, a, 0.75 * math.pi),  # CENTRED corner
                 (a, a, 2 * math.pi / 3),  # FREE corner: a = c, beta 120
                 (a, a, math.pi / 2)]
    base = torch.tensor(rows, dtype=torch.float64)
    n = len(base)
    for sg in MONO_SGS:
        L = torch.stack([base[:, 0], torch.full((n,), 5.0, dtype=torch.float64), base[:, 1]], 1).to(dtype).requires_grad_(True)
        A = torch.stack([torch.full((n,), HALF_PI, dtype=torch.float64), base[:, 2],
                         torch.full((n,), HALF_PI, dtype=torch.float64)], 1).to(dtype).requires_grad_(True)
        E = mono_reduction_penalty(L, A, torch.full((n,), sg, dtype=torch.long), margin)
        assert torch.isfinite(E).all(), sg
        gL, gA = torch.autograd.grad(E.sum(), (L, A))
        assert torch.isfinite(gL).all() and torch.isfinite(gA).all(), sg
        if dtype == torch.float32:  # the same rows through the float32 dispatch
            assert torch.isfinite(cell_reduction_penalty(A.detach(), L.detach(), torch.full((n,), sg), margin)).all()


def test_sg_outside_the_table_is_loud():
    L = torch.tensor([[5.0, 6.0, 7.0]] * 5)
    A = torch.tensor([[HALF_PI, 1.9, HALF_PI]] * 5)
    E = mono_reduction_penalty(L, A, torch.tensor([0, 1, 2, 16, 230]), 0.0)
    assert torch.isnan(E).all()


@pytest.mark.parametrize('dtype', [torch.int32, torch.int16, torch.uint8, torch.float32, torch.float64])
def test_sg_dtypes_score_like_long(dtype):
    """cell_reduction_penalty takes sg in any integer or float dtype the other systems' masks take. A uint8 sg must not
    act as a boolean mask: with exactly 231 monoclinic rows that silently mis-scores the batch. (int8 is not supported
    by the masks themselves: sg <= 142 overflows the constant, so tetragonal rows score 0.)"""
    gen = torch.Generator().manual_seed(10)
    L, A, sg = random_cells(2000, tuple(range(1, 128)), gen, dtype=torch.float32)
    assert ((sg >= 3) & (sg <= 15)).sum() > 100
    assert torch.equal(cell_reduction_penalty(A, L, sg.to(dtype), 0.0), cell_reduction_penalty(A, L, sg, 0.0))
    L, A, sg = random_cells(231, MONO_SGS, gen, dtype=torch.float32)
    assert torch.equal(cell_reduction_penalty(A, L, sg.to(dtype), 0.0), cell_reduction_penalty(A, L, sg, 0.0))


def _cellpar_rad32(lattices):
    """cell parameters as compute_standard_cell stores them: ase degrees -> float32 radians"""
    from ase.geometry import cell_to_cellpar
    p = np.stack([cell_to_cellpar(lat) for lat in lattices])
    return torch.tensor(p[:, :3], dtype=torch.float32), torch.tensor(p[:, 3:] / 180 * np.pi, dtype=torch.float32)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')  # spglib OLD_ERROR_HANDLING notice
def test_spglib_standard_cells_score_zero():
    """Point-atom crystals in every sg 3-15, standardized with compute_standard_cell's spglib call from their own basis
    and from a random GL(2,Z) re-setting; every output scores exactly 0 in float32 (compute_standard_cell
    accepts < 1e-3)."""
    spglib = pytest.importorskip('spglib')
    pytest.importorskip('ase')
    rng = np.random.default_rng(7)
    lattices, sgs = [], []
    for sg in MONO_SGS:
        ops = np.array(SYM_OPS[sg], dtype=np.float64)
        for _ in range(25):
            a, c = np.exp(rng.uniform(np.log(3), np.log(60), 2))
            b = float(np.exp(rng.uniform(np.log(4), np.log(20))))
            cb = math.cos(math.radians(rng.uniform(60, 150)))
            lat = np.array([[a, 0, 0], [0, b, 0], [c * cb, 0, c * math.sqrt(1 - cb * cb)]])
            while True:  # 3 atoms in general position, images >= 0.8 A apart
                x = rng.uniform(0, 1, (3, 3))
                frac = np.mod(np.einsum('oij,aj->aoi', ops[:, :3, :3], x) + ops[None, :, :3, 3], 1).reshape(-1, 3)
                d = frac[:, None] - frac[None]
                d -= np.round(d)
                if (np.linalg.norm(d @ lat, axis=-1) + 1e9 * np.eye(len(frac))).min() > 0.8:
                    break
            nums = np.repeat([6, 7, 8], len(ops))
            while True:
                p, q, r, s = (int(v) for v in rng.integers(-3, 4, 4))
                if abs(p * s - q * r) == 1:
                    break
            for (p_, q_, r_, s_) in ((1, 0, 0, 1), (p, q, r, s)):
                det = p_ * s_ - q_ * r_
                lat2 = np.stack([p_ * lat[0] + q_ * lat[2], det * lat[1], r_ * lat[0] + s_ * lat[2]])
                frac2 = np.mod(frac @ lat @ np.linalg.inv(lat2), 1)
                lat_std, frac_std, nums_std = spglib.standardize_cell((lat2, frac2, nums), to_primitive=False,
                                                                      no_idealize=True)
                assert spglib.get_symmetry_dataset((lat_std, frac_std, nums_std)).number == sg
                lattices.append(lat_std)
                sgs.append(sg)
    L, A = _cellpar_rad32(lattices)
    E = cell_reduction_penalty(A, L, torch.tensor(sgs), 0.0)
    assert len(E) == 2 * 25 * len(MONO_SGS)
    assert (E == 0).all(), int((E != 0).sum())


def test_neighbouring_setting_cells_score_positive():
    """Control for the test above: domain cells moved to a neighbouring setting-group cell (GLIDE a -> a + c,
    CENTRED c -> c + a, FREE a <-> c; then the twin to beta >= 90) score > 0 away from a wall."""
    gen = torch.Generator().manual_seed(8)
    for name in CLASSES:
        L, A, sg = random_cells(60000, class_sgs(name), gen)
        inside = hard_domain(name, L[:, 0], L[:, 2], torch.cos(A[:, 1]))
        L, A, sg = L[inside], A[inside], sg[inside]
        a, c = L[:, 0], L[:, 2]
        e1 = torch.stack([a, torch.zeros_like(a)], 1)
        e2 = torch.stack([c * torch.cos(A[:, 1]), c * torch.sin(A[:, 1])], 1)
        n1, n2 = {'GLIDE': (e1 + e2, e2), 'CENTRED': (e1, e2 + e1), 'FREE': (e2, e1)}[name]
        la, lc = n1.norm(dim=1), n2.norm(dim=1)
        cbn = (n1 * n2).sum(1) / (la * lc)
        cbn = torch.where(cbn > 0, -cbn, cbn)
        L2 = torch.stack([la, L[:, 1], lc], 1)
        A2 = torch.stack([A[:, 0], torch.arccos(cbn), A[:, 2]], 1)
        E = mono_reduction_penalty(L2, A2, sg, 0.0)
        away = rel_slack(name, la, lc, cbn) > 1e-9
        assert away.sum() > 1000
        assert (E[away] > 0).all(), name


# Real cells, as stored (float32; each value round-trips to the stored bits). 'CSD <refcode>': the stored cell in its own
# basis, from new_csd.pt (Z'=1) or tests/datasets/mini_new_csd.pt. 'GFN p12 row <i>': sg 14 samples of a trained GFN,
# D:/crystal_datasets/gfn_results/p12_acr_lr0p05_best_10k.pt. Every row scores exactly 0 under the previous walls
# (cos(beta) in [-a/c, 0] for every sg); the rows with a nonzero pinned value are cells the class walls move. The gfn
# crystal/protocol tests pass under both walls, so this is where a change to the walls on real data shows up.
REAL_ROWS = (  # source, sg, a, b, c (A), alpha, beta, gamma (rad), penalty at margin 0
    ('CSD KEYVOC', 3, 14.1009998, 12.8719997, 11.0109997, 1.57079637, 1.85039806, 1.57079637, 0.07875232398509979),
    ('CSD ZINCII', 4, 16.4640007, 9.75399971, 6.46010017, 1.57079637, 1.70326686, 1.57079637, 2.3980605602264404),
    ('CSD DMANTL12', 4, 5.08940983, 18.2504005, 4.91701984, 1.57079637, 2.0647769, 1.57079637, 0.0012291902676224709),
    ('CSD PLBULD', 4, 11.2110004, 6.82200003, 8.71000004, 1.57079637, 1.74358392, 1.57079637, 0.08245006948709488),
    ('CSD DIVPON', 4, 8.52900028, 11.7349997, 4.92500019, 1.57079637, 1.65125597, 1.57079637, 0.5354970097541809),
    ('CSD FUNGIE', 4, 13.1829996, 6.40700006, 11.842, 1.57079637, 2.00852489, 1.57079637, 0.012823514640331268),
    ('CSD AGLCAM', 4, 9.22900009, 5.1789999, 8.26099968, 1.57079637, 1.79594386, 1.57079637, 0.013730479404330254),
    ('CSD IKURUB', 4, 7.16200018, 14.4799995, 6.08400011, 1.57079637, 1.95075452, 1.57079637, 0.03139488399028778),
    ('CSD REQHUQ', 4, 8.6260004, 9.92399979, 5.1789999, 1.57079637, 1.810238, 1.57079637, 0.44298693537712097),
    ('CSD RULWEA', 4, 18.5270004, 6.07299995, 12.4619999, 1.57079637, 1.7919296, 1.57079637, 0.23685698211193085),
    ('CSD FPOLCR', 4, 9.77000046, 13.1899996, 6.65999985, 1.57079637, 1.82735968, 1.57079637, 0.2180582731962204),
    ('CSD JOVYUO', 4, 13.9899998, 6.03700018, 12.6429996, 1.57079637, 1.59697628, 1.57079637, 0.011351018212735653),
    ('CSD NASRUV', 4, 10.8610001, 6.52799988, 8.58300018, 1.57079637, 1.86558247, 1.57079637, 0.07044161856174469),
    ('CSD YALXEQ', 5, 18.5900002, 7.43100023, 15.967, 1.57079637, 2.27852726, 1.57079637, 0.004620347172021866),
    ('CSD TAVBEB', 5, 32.7280998, 6.53022003, 16.5179005, 1.57079637, 2.25603008, 1.57079637, 0.01642322912812233),
    ('CSD AMESNC', 9, 9.13140011, 12.8671999, 13.0317001, 1.57079637, 2.20963931, 1.57079637, 0.06047350913286209),
    ('CSD PHSRUB', 9, 15.7700005, 9.91699982, 19.6399994, 1.57079637, 2.16979337, 1.57079637, 0.026353564113378525),
    ('CSD BEGTAI', 9, 31.3099995, 5.62900019, 18.0249996, 1.57079637, 2.42914915, 1.57079637, 0.03278713300824165),
    ('CSD BOXMHF10', 14, 11.4910002, 11.1969995, 13.1999998, 1.57079637, 2.23140335, 1.57079637, 0.0015392866916954517),
    ('CSD PEWZUO', 15, 12.1947002, 20.3994999, 17.7565002, 1.57079637, 2.0026257, 1.57079637, 0.005646924022585154),
    ('CSD DUCMIX', 15, 20.5909996, 20.5170002, 12.4659996, 1.57079637, 2.4570744, 1.57079637, 0.028667118400335312),
    ('CSD VORGOY', 15, 18.9769993, 10.2460003, 34.6500015, 1.57079637, 1.91549885, 1.57079637, 0.004106036387383938),
    ('CSD GIBCAX', 15, 16.9659996, 6.69999981, 28.0049992, 1.57079637, 1.92478406, 1.57079637, 0.0019123851088806987),
    ('CSD NEJFEP (mini_new_csd)', 4, 9.97999954, 9.84889984, 12.7770004, 1.57079637, 1.82561445, 1.57079637, 0.0),
    ('CSD APAPUF (mini_new_csd)', 4, 10.3339996, 7.93289995, 11.0349998, 1.57079637, 1.67972231, 1.57079637, 0.0),
    ('CSD EBEXUI (mini_new_csd)', 15, 14.4054003, 9.35270023, 27.5100002, 1.57079637, 1.70719385, 1.57079637, 0.0),
    ('CSD KEQQON (mini_new_csd)', 15, 22.8320007, 4.75600004, 20.5149994, 1.57079637, 2.02004409, 1.57079637, 0.0),
    ('CSD AFEGIF (mini_new_csd)', 14, 12.9259005, 18.4871006, 14.7918997, 1.57079637, 1.76430094, 1.57079637, 0.0),
    ('CSD HONWUG (mini_new_csd)', 14, 13.6337996, 11.7589998, 13.1680002, 1.57079637, 2.00206709, 1.57079637, 0.0),
    ('GFN p12 row 5596', 14, 9.18811417, 30.3139458, 3.99655747, 1.57079637, 1.82276511, 1.57079637, 0.0010128860594704747),
    ('GFN p12 row 6478', 14, 8.00722122, 18.9245796, 6.8301053, 1.57079637, 2.11949158, 1.57079637, 0.009039795957505703),
    ('GFN p12 row 7043', 14, 8.31464195, 18.4494247, 7.34314394, 1.57079637, 2.19601297, 1.57079637, 0.020647935569286346),
    ('GFN p12 row 6719', 14, 6.69290829, 41.2497253, 3.76336408, 1.57079637, 2.05909824, 1.57079637, 0.03533696010708809),
    ('GFN p12 row 4083', 14, 6.84004927, 40.8667793, 3.79042053, 1.57079637, 2.11524796, 1.57079637, 0.058020077645778656),
    ('GFN p12 row 4536', 14, 7.45144463, 39.0168114, 3.69453382, 1.57079637, 2.17693496, 1.57079637, 0.10354944318532944),
    ('GFN p12 row 5867', 14, 7.71333027, 38.8889694, 3.83268356, 1.57079637, 2.26343203, 1.57079637, 0.15219540894031525),
    ('GFN p12 row 6479', 14, 14.9045811, 30.5939274, 3.77396631, 1.57079637, 2.45416451, 1.57079637, 0.4176730513572693),
    ('GFN p12 row 5090', 14, 3.87396669, 39.9608955, 7.53560877, 1.57079637, 1.93406594, 1.57079637, 0.0),
    ('GFN p12 row 1059', 14, 3.68654752, 28.1940556, 10.790967, 1.57079637, 1.82145071, 1.57079637, 0.0),
    ('GFN p12 row 4009', 14, 6.50803709, 40.3921013, 4.26518011, 1.57079637, 1.62360024, 1.57079637, 0.0),
    ('GFN p12 row 8292', 14, 3.80113649, 27.3970661, 10.18332, 1.57079637, 1.70849466, 1.57079637, 0.0),
)


def test_real_cells_pinned():
    sg = torch.tensor([r[1] for r in REAL_ROWS])
    L = torch.tensor([r[2:5] for r in REAL_ROWS], dtype=torch.float32)
    A = torch.tensor([r[5:8] for r in REAL_ROWS], dtype=torch.float32)
    pinned = torch.tensor([r[8] for r in REAL_ROWS], dtype=torch.float32)
    E = cell_reduction_penalty(A, L, sg, 0.0)
    old = bounding(torch.cos(A[:, 1]), (-L[:, 0] / L[:, 2]).clamp(min=-1), 0, 0.0) + (A[:, 0] - HALF_PI) ** 2 \
        + (A[:, 2] - HALF_PI) ** 2
    moved = pinned > 0
    assert (old == 0).all()
    assert moved.sum() >= 30 and (~moved).sum() >= 10 and set(sg[moved].tolist()) >= {3, 4, 5, 9, 14, 15}
    assert torch.equal(E == 0, ~moved), [r[0] for r, e in zip(REAL_ROWS, E.tolist()) if (e == 0) != (r[8] == 0)]
    assert torch.allclose(E[moved], pinned[moved], rtol=1e-4, atol=0), (E[moved] / pinned[moved] - 1).abs().max()
    # zero exactly on the hard class domain (float64 Gram form, class derived from SYM_OPS)
    inside = torch.tensor([bool(hard_domain(setting_class(int(s)), l[0].double(), l[2].double(), torch.cos(ang[1].double())))
                           for s, l, ang in zip(sg, L, A)])
    assert torch.equal(inside, ~moved)

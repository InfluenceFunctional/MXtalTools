"""GATE 1 (fd_tooling.md section 4): exactly one representative per orbit.

Pose (every space group; the fold supports all 230, sg 6-9 through the K cut in SO(3)):
- generic points: 10^4 random (t, R) per group, every Gamma+ image with every
  lattice shift that reaches the asymmetric unit, counted by ``in_asu``
  (membership by slack and tie rules, independent of ``fold``). Zero rows may
  count other than 1 outside a 1e-6 face slab. ``fold`` of the point and of a
  random image agree, the result is in the domain, and ``unfold`` inverts it.
- special points: exact rational points on facets, on edges (two facets), at
  vertices and on the fixed sets of Gamma+ elements (special positions); the
  canonical form is the same from every Gamma+ image, and exactly one image is in
  the domain (``in_asu``).
- band points: the special points moved by 3e-10 to 5e-8 (around the tol = 1e-9
  band, where plane tests within a tolerance are not equivariant): every Gamma+
  image folds to one description, which is in the domain, and exactly one
  enumerated image of the orbit is that description. (A slack within rounding of
  exactly +-tol can still flip between images; offsets avoid tol itself.)
- mutation control: a facet pushed out by 0.01 double-counts, pulled in by 0.01
  loses points (``in_asu``).
- K cut (sg 6-9): of R and Q_K R exactly one is in the domain, and the domain
  holds half of SO(3).
- no molecule argument: a planar molecule's improper image, re-expressed through
  the molecular mirror, is the same crystal and stays a distinct domain point.
- non-finite rows are refused.

The 230-group sweeps are marked ``slow`` outside a representative set
(``FAST_SG``: every P+ kind, sg 6-9, free-axis, piecewise-glued and the largest
groups); ``-m "not slow"`` runs that set.

Cell (sg 1, 2): canonicalise(scramble(c)) == canonicalise(c) within 1e-9 (Angstrom,
degrees) in the interior and at 1e-7, 1e-4 and 1e-2 (rad, or relative for the two
length faces) on both sides of every face of the tri_niggli domain, and on every
face (offset 0 and +-1e-12), where candidates tie and the tie-break decides. A
control shows the old ``standardize.choose_basis`` tie-break fails that check at
1e-7 rad from the beta = 90 deg face; another that a tie-break taking the first
valid candidate fails it on the faces. Special metrics (a proper lattice
automorphism) are flagged by ``CellReduction.n_auto``; unflagged rows, faces
included, have one canonical description (cell and pose) from every basis; a
special metric stored in float32 keeps its flag.
"""
import copy
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from mxtaltools.common.sym_utils import tri_niggli_reduction_penalty
from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.exact import polytope_vertices, solve, to_fraction
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
SLAB = 1e-6
TOL = 1e-9
N_GENERIC = 10_000
#: groups run in the default lane: C1/C2/D2/D3/D4/D6/T/O point groups, K groups 6-9, free axes (1, 4, 75, 144,
#: 168), piecewise-glued facets (88, 92, 142, 148, 198, 205) and the largest Gamma+ (196, 227)
FAST_SG = (1, 2, 4, 6, 7, 8, 9, 14, 15, 19, 61, 75, 88, 92, 142, 144, 148, 152, 166, 168, 196, 198, 205, 227, 230)
ALL_SG = [sg if sg in FAST_SG else pytest.param(sg, marks=pytest.mark.slow) for sg in range(1, 231)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cells(sg, n, seed, cell_sampler):
    return torch.tensor(cell_sampler(LATTICE_TYPE[sg], n, seed=seed), dtype=F64)


def _desc(sg, cell, t, R):
    return CrystalDescription.from_parts(sg, cell, t, R)


def _cart(cell):
    return ops.cell_matrices(cell)


def _orbit(sg, cell, t, R, lo, hi):
    """Every Gamma+ image (W t + w + n, Q R) whose non-free centroid lies in [lo, hi] (padded by SLAB).

    Free coordinates are put at the gauge (half the trainer box). Written apart
    from ``domains`` so that the count does not share its enumeration.
    Returns (cell, t, R, row) of the images.
    """
    tab = group_table(sg)
    W, w, free = tab.gamma_plus_W, tab.gamma_plus_w, tab.free
    gauge = tab.trainer_box / 2
    b, m = len(t), len(W)
    T = _cart(cell)
    Q = T[:, None] @ W[None] @ torch.linalg.inv(T)[:, None]                 # [b, m, 3, 3]
    y = torch.einsum('mij,bj->bmi', W, t) + w
    y = torch.where(free, gauge, y)
    base = torch.where(free, torch.zeros_like(y), torch.ceil(lo - SLAB - y))
    ext = torch.floor(hi - lo + 2 * SLAB).to(torch.long) + 1
    out_t, out_R, out_row = [], [], []
    for k in itertools.product(*[range(int(e)) if not bool(f) else range(1) for e, f in zip(ext, free)]):
        x = y + base + torch.tensor(k, dtype=F64)
        ok = ((x <= hi + SLAB) | free).all(-1)
        rows, els = torch.nonzero(ok, as_tuple=True)
        out_t.append(x[rows, els])
        out_R.append(Q[rows, els] @ R[rows])
        out_row.append(rows)
    rows = torch.cat(out_row)
    return cell[rows], torch.cat(out_t), torch.cat(out_R), rows


def _count(sg, cell, t, R, section=None):
    """Per row: number of images in the domain by the rules (``in_asu``), and the smallest |slack| over its images."""
    sec = domains.translation_section(sg) if section is None else section
    c_img, t_img, R_img, rows = _orbit(sg, cell, t, R, sec.region.lo, sec.region.hi)
    inside, slack = domains.in_asu(_desc(sg, c_img, t_img, R_img), sec, TOL)
    count = torch.zeros(len(t), dtype=torch.long).index_add_(0, rows, inside.long())
    near = torch.full((len(t),), float('inf'), dtype=F64).scatter_reduce_(0, rows, slack.abs(), 'amin')
    return count, near, (t_img[inside], R_img[inside], rows[inside])


def _chunks(n, sg):
    m = len(group_table(sg).gamma_plus)
    step = max(50, 400_000 // (8 * m))
    return [(lo, min(n, lo + step)) for lo in range(0, n, step)]


# ---------------------------------------------------------------------------
# pose, generic points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', ALL_SG)
def test_pose_generic_orbits(sg, cell_sampler):
    """10^4 random poses: exactly one Gamma+ image in the domain (outside the 1e-6 slab); fold agrees and inverts."""
    g = torch.Generator().manual_seed(1000 + sg)
    cell = _cells(sg, N_GENERIC, sg, cell_sampler)
    t = torch.rand(N_GENERIC, 3, generator=g, dtype=F64) * 3 - 1
    R = so3.random_rotations(N_GENERIC, g)
    bad, in_slab = 0, 0
    for lo, hi in _chunks(N_GENERIC, sg):
        c, tt, RR = cell[lo:hi], t[lo:hi], R[lo:hi]
        count, near, (t_in, R_in, row_in) = _count(sg, c, tt, RR)
        slab = near < SLAB
        in_slab += int(slab.sum())
        bad += int(((count != 1) & ~slab).sum())

        d = _desc(sg, c, tt, RR)
        fr = domains.fold(d)
        assert bool((fr.tie != 3).all()), 'a row had no candidate in the domain'
        inside, _ = domains.in_domain(fr.desc)
        assert bool(inside[~slab].all())
        one = (count == 1) & ~slab
        ref_t = torch.zeros_like(tt)
        ref_R = torch.zeros_like(RR)
        ref_t[row_in], ref_R[row_in] = t_in, R_in
        assert (fr.desc.t[one, 0] - ref_t[one]).abs().max() <= TOL
        assert (fr.desc.R[one, 0] - ref_R[one]).abs().max() <= TOL

        # fold of a random image is the same canonical point
        tab = group_table(sg)
        k = torch.randint(len(tab.gamma_plus), (len(tt),), generator=g)
        W, w = tab.gamma_plus_W[k], tab.gamma_plus_w[k]
        shift = torch.randint(-2, 3, (len(tt), 3), generator=g).to(F64)
        Q = ops.cartesian_rotation(_cart(c), W)
        img = _desc(sg, c, torch.einsum('bij,bj->bi', W, tt) + w + shift, Q @ RR)
        fi = domains.fold(img)
        assert (fi.desc.t[~slab] - fr.desc.t[~slab]).abs().max() <= TOL
        assert (fi.desc.R[~slab] - fr.desc.R[~slab]).abs().max() <= TOL

        back = domains.unfold(fr.desc, fr.element, fr.lattice, fr.free_shift)
        assert (back.t[:, 0] - tt).abs().max() <= 1e-12
        assert (back.R[:, 0] - RR).abs().max() <= 1e-12
    assert bad == 0, f'sg {sg}: {bad} of {N_GENERIC} poses outside the slab had other than one image in the domain'
    assert in_slab <= 0.01 * N_GENERIC


# ---------------------------------------------------------------------------
# pose, exact special points
# ---------------------------------------------------------------------------

def _nullspace_solve(A, b):
    """Exact (particular solution, nullspace basis) of A x = b, or None if inconsistent."""
    rows, cols = len(A), len(A[0])
    M = [[Fraction(v) for v in r] + [Fraction(bi)] for r, bi in zip(A, b)]
    piv_cols, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        M[r] = [v / M[r][c] for v in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * x for a, x in zip(M[i], M[r])]
        piv_cols.append(c)
        r += 1
    if any(all(v == 0 for v in M[i][:cols]) and M[i][cols] != 0 for i in range(rows)):
        return None
    x0 = [Fraction(0)] * cols
    for i, c in enumerate(piv_cols):
        x0[c] = M[i][cols]
    basis = []
    for fcol in (c for c in range(cols) if c not in piv_cols):
        v = [Fraction(0)] * cols
        v[fcol] = Fraction(1)
        for i, c in enumerate(piv_cols):
            v[c] = -M[i][fcol]
        basis.append(v)
    return x0, basis


def _special_points(sg, rng, per_kind=6):
    """Exact rational points on facets, edges, vertices and element fixed sets of ASU(Gamma+) (free coords at gauge)."""
    tab = group_table(sg)
    raw = tab.raw
    free = [bool(f) for f in raw['free']]
    nf = [i for i in range(3) if not free[i]]
    gauge = [to_fraction(b) / 2 if free[i] else Fraction(0) for i, b in enumerate(raw['trainer_box']['box'])]
    cuts = raw['asu']['cuts']
    normals = [[Fraction(v) for v in c['n']] for c in cuts]
    offsets = [to_fraction(c['c']) for c in cuts]
    planes = [i for i, c in enumerate(cuts) if raw['asu']['facet'][i] and all(normals[i][j] == 0 for j in range(3) if free[j])]
    verts = [[to_fraction(v) for v in p] for p in raw['asu']['vertices']]
    lo = [min(p[i] for p in verts) for i in range(3)]
    hi = [max(p[i] for p in verts) for i in range(3)]

    def closed(p):
        return all(sum(n[j] * p[j] for j in range(3)) + c >= 0 for n, c in zip(normals, offsets))

    def grid(i):
        return lo[i] + (hi[i] - lo[i]) * Fraction(int(rng.integers(0, 49)), 48)

    def base():
        return [gauge[i] if free[i] else grid(i) for i in range(3)]

    pts = []
    for i in planes:                                    # facets
        got = 0
        for _ in range(400):
            p = base()
            d = max(nf, key=lambda j: abs(normals[i][j]))
            p[d] = -(offsets[i] + sum(normals[i][j] * p[j] for j in range(3) if j != d)) / normals[i][d]
            if closed(p):
                pts.append(('facet', p))
                got += 1
                if got >= per_kind:
                    break
    if len(nf) == 3:                                    # edges
        got = 0
        for _ in range(4000):
            i, j = rng.choice(planes, 2, replace=False)
            k = int(rng.integers(0, 3))
            others = [a for a in range(3) if a != k]
            p = base()
            A = [[normals[i][a] for a in others], [normals[j][a] for a in others]]
            rhs = [-(offsets[i] + normals[i][k] * p[k]), -(offsets[j] + normals[j][k] * p[k])]
            sol = solve(A, rhs)
            if sol is None:
                continue
            p[others[0]], p[others[1]] = sol
            if closed(p):
                pts.append(('edge', p))
                got += 1
                if got >= per_kind:
                    break
    for v in verts:                                     # vertices
        pts.append(('vertex', [gauge[i] if free[i] else v[i] for i in range(3)]))
    Wx = [np.asarray(W, dtype=np.int64).reshape(3, 3) for W in raw['gamma']['W']]
    wx = [[to_fraction(v) for v in w] for w in raw['gamma']['w']]
    fixed = []
    for e in raw['gamma_plus'][1:]:                     # special positions
        W, w = Wx[e], wx[e]
        for n in itertools.product((-1, 0, 1), repeat=len(nf)):
            A = [[int(W[a][b]) - (a == b) for b in nf] for a in nf]
            rhs = [-(w[a] + n[k]) for k, a in enumerate(nf)]
            res = _nullspace_solve(A, rhs)
            if res is not None:
                fixed.append(res)
    rng.shuffle(fixed)
    for x0, basis in fixed[:per_kind]:
        p = [gauge[i] for i in range(3)]
        q = list(x0)
        for v in basis:
            s = Fraction(int(rng.integers(-24, 25)), 24)
            q = [a + s * b for a, b in zip(q, v)]
        for k, a in enumerate(nf):
            p[a] = q[k]
        pts.append(('special', p))
    return pts


@pytest.mark.parametrize('sg', ALL_SG)
def test_pose_special_points(sg, cell_sampler):
    """Exact rational face, edge, vertex and special-position points: one canonical form from every Gamma+ image."""
    rng = np.random.default_rng(2000 + sg)
    g = torch.Generator().manual_seed(2000 + sg)
    tab = group_table(sg)
    pts = _special_points(sg, rng)
    assert pts
    kinds = {k for k, _ in pts}
    rotations = bool((tab.gamma_plus_W != torch.eye(3, dtype=F64)).any())
    assert 'vertex' in kinds and ('special' in kinds or not rotations)
    P = len(pts)
    m = len(tab.gamma_plus)
    t = torch.tensor([[float(v) for v in p] for _, p in pts], dtype=F64)
    cell = _cells(sg, P, 3000 + sg, cell_sampler)
    R = so3.random_rotations(P, g)
    # every element, a random lattice shift, a random free-axis offset
    W, w = tab.gamma_plus_W, tab.gamma_plus_w
    shift = torch.randint(-1, 2, (P, m, 3), generator=g).to(F64)
    shift = shift + torch.where(tab.free, torch.rand(P, m, 3, generator=g, dtype=F64), torch.zeros(3, dtype=F64))
    T = _cart(cell)
    Q = T[:, None] @ W[None] @ torch.linalg.inv(T)[:, None]
    t_img = torch.einsum('mij,pj->pmi', W, t) + w + shift
    R_img = Q @ R[:, None]
    d_img = _desc(sg, cell[:, None].expand(P, m, 6).reshape(-1, 6), t_img.reshape(-1, 3), R_img.reshape(-1, 3, 3))
    fr = domains.fold(d_img)
    assert bool((fr.tie != 3).all())
    ct = fr.desc.t[:, 0].reshape(P, m, 3)
    cR = fr.desc.R[:, 0].reshape(P, m, 3, 3)
    dev_t = (ct - ct[:, :1]).abs().amax((1, 2))
    dev_R = (cR - cR[:, :1]).abs().amax((1, 2, 3))
    assert dev_t.max() <= TOL and dev_R.max() <= TOL, \
        f'sg {sg}: canonical forms differ across images by {dev_t.max():.2e} (t), {dev_R.max():.2e} (R)'
    inside, _ = domains.in_domain(fr.desc[torch.arange(0, P * m, m)])
    assert bool(inside.all())
    count, _, _ = _count(sg, cell, t, R)
    assert bool((count == 1).all()), f'sg {sg}: in-domain image counts {count.tolist()}'


# ---------------------------------------------------------------------------
# pose, points in the tolerance band
# ---------------------------------------------------------------------------

#: offsets from the exact special points (fractional), on both sides of the fold's tol = 1e-9. Not tol itself: a slack
#: within rounding (about 1e-16) of +-tol is the one place where an image's plane test can flip
BAND = (3e-10, 7e-10, 1.3e-9, 3e-9, 5e-8)


@pytest.mark.parametrize('sg', ALL_SG)
def test_pose_band(sg, cell_sampler):
    """Points 3e-10 to 5e-8 off facets, edges, vertices and special positions: one canonical form from every image."""
    rng = np.random.default_rng(7000 + sg)
    g = torch.Generator().manual_seed(7000 + sg)
    tab = group_table(sg)
    sec = domains.translation_section(sg)
    pts = _special_points(sg, rng)
    pts += [p for p in pts if p[0] == 'vertex'] * 3                 # vertices, where most cuts meet, weighted up
    base = torch.tensor([[float(v) for v in p] for _, p in pts], dtype=F64).repeat(4, 1)   # 4 directions each
    t0 = base.repeat(len(BAND), 1)
    delta = torch.tensor(BAND, dtype=F64).repeat_interleave(len(base))
    dirs = torch.where(tab.free, 0.0, torch.randn(len(t0), 3, generator=g, dtype=F64))
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-300)
    t = torch.where(tab.free, sec.gauge, t0 + delta[:, None] * dirs)
    m = len(tab.gamma_plus)
    t = t[torch.randperm(len(t), generator=g)[:max(100, 40_000 // m)]]   # at most ~4e4 images to fold
    n = len(t)
    cell = _cells(sg, n, 7000 + sg, cell_sampler)
    R = so3.random_rotations(n, g)
    fr = domains.fold(_desc(sg, cell, t, R))
    assert bool(domains.in_domain(fr.desc)[0].all())
    assert float(fr.slack.min()) >= -2 * TOL                   # within the band of ASU(Gamma+), whatever the tie code
    # every Gamma+ element, a random lattice shift and a random free-axis offset: the same canonical form
    W, w = tab.gamma_plus_W, tab.gamma_plus_w
    shift = torch.randint(-1, 2, (n, m, 3), generator=g).to(F64)
    shift = shift + torch.where(tab.free, torch.rand(n, m, 3, generator=g, dtype=F64), torch.zeros(3, dtype=F64))
    T = _cart(cell)
    Q = T[:, None] @ W[None] @ torch.linalg.inv(T)[:, None]
    t_img = torch.einsum('mij,pj->pmi', W, t) + w + shift
    d_img = _desc(sg, cell[:, None].expand(n, m, 6).reshape(-1, 6), t_img.reshape(-1, 3),
                  (Q @ R[:, None]).reshape(-1, 3, 3))
    fi = domains.fold(d_img)
    dev_t = (fi.desc.t[:, 0].reshape(n, m, 3) - fr.desc.t[:, :1]).abs().amax((1, 2))
    dev_R = (fi.desc.R[:, 0].reshape(n, m, 3, 3) - fr.desc.R[:, :1]).abs().amax((1, 2, 3))
    assert dev_t.max() <= TOL and dev_R.max() <= TOL, \
        f'sg {sg}: canonical forms differ across images by {dev_t.max():.2e} (t), {dev_R.max():.2e} (R)'
    # exactly one enumerated image of each orbit is the canonical description
    _, ti, Ri, rows = _orbit(sg, cell, t, R, sec.region.lo, sec.region.hi)
    same = ((ti - fr.desc.t[rows, 0]).abs() <= TOL).all(1) & ((Ri - fr.desc.R[rows, 0]).abs() <= TOL).all(-1).all(-1)
    count = torch.zeros(n, dtype=torch.long).index_add_(0, rows, same.long())
    assert bool((count == 1).all()), f'sg {sg}: canonical-image counts {sorted(set(count.tolist()))}'


def test_fold_refuses_non_finite_rows():
    """A NaN centroid or a cell with no real metric is refused by name, never folded to a garbage row."""
    t = torch.tensor([[0.1, 0.2, 0.3], [float('nan'), 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=F64)
    cell = torch.tensor([[6.0, 7.0, 8.0, 1.7, 1.8, 1.9]] * 3, dtype=F64)
    cell[2, 3:] = torch.deg2rad(torch.tensor([60.0, 60.0, 170.0], dtype=F64))
    d = _desc(2, cell, t, so3.random_rotations(3, torch.Generator().manual_seed(0)))
    with pytest.raises(ValueError, match=r'rows \[1, 2\]'):
        domains.fold(d)
    with pytest.raises(ValueError, match=r'rows \[1, 2\]'):
        domains.in_domain(d)
    assert bool(domains.in_domain(domains.fold(d[:1]).desc)[0].all())


# ---------------------------------------------------------------------------
# mutation control
# ---------------------------------------------------------------------------

def _mutated_section(sg, cut, delta):
    tab = group_table(sg)
    sec = domains.translation_section(sg)
    cuts = copy.deepcopy(list(tab.asu.cuts))
    n = [Fraction(v) for v in cuts[cut]['n']]
    norm = math.sqrt(sum(float(v) ** 2 for v in n))
    cuts[cut]['c'] = str(to_fraction(cuts[cut]['c']) + Fraction(delta * norm).limit_denominator(10 ** 12))
    normals = [[Fraction(v) for v in c['n']] for c in cuts]
    offsets = [to_fraction(c['c']) for c in cuts]
    verts = torch.tensor([[float(v) for v in p] for p in polytope_vertices(normals, offsets)], dtype=F64)
    return sec.with_region(domains.Region.from_cuts(cuts, verts))


@pytest.mark.parametrize('sg', [2, 14, 19, 92, 148, 198])
def test_mutation_control(sg, cell_sampler):
    """Every non-free facet pushed out by 0.01 double-counts some orbits; pulled in by 0.01, some orbits are lost."""
    tab = group_table(sg)
    n = 4000
    g = torch.Generator().manual_seed(4000 + sg)
    cell = _cells(sg, n, 4000 + sg, cell_sampler)
    t = torch.rand(n, 3, generator=g, dtype=F64)
    R = so3.random_rotations(n, g)
    count, _, _ = _count(sg, cell, t, R)
    assert bool((count == 1).all())
    free = tab.free.tolist()
    for i, cut in enumerate(tab.asu.cuts):
        if not tab.raw['asu']['facet'][i] or any(cut['n'][j] != 0 for j in range(3) if free[j]):
            continue
        out, _, _ = _count(sg, cell, t, R, _mutated_section(sg, i, 0.01))
        assert int((out == 2).sum()) > 0, f'sg {sg}: pushing cut {i} out by 0.01 went unnoticed'
        inn, _, _ = _count(sg, cell, t, R, _mutated_section(sg, i, -0.01))
        assert int((inn == 0).sum()) > 0, f'sg {sg}: pulling cut {i} in by 0.01 went unnoticed'


# ---------------------------------------------------------------------------
# K cut (sg 6-9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', [6, 7, 8, 9])
def test_k_cut(sg, cell_sampler):
    """K = {1, k} fixes the projected centroid: of R and Q_k R exactly one is in the domain; the cut holds half of SO(3)."""
    tab = group_table(sg)
    assert len(tab.kernel) == 2
    k = int(tab.kernel[1])
    n = 20_000
    g = torch.Generator().manual_seed(5000 + sg)
    cell = _cells(sg, n, 5000 + sg, cell_sampler)
    sec = domains.translation_section(sg)
    lo, hi = sec.region.lo, sec.region.hi
    t = torch.where(sec.free, sec.gauge, lo + (hi - lo) * torch.rand(n, 3, generator=g, dtype=F64))
    R = so3.random_rotations(n, g)
    Qk = ops.cartesian_rotation(_cart(cell), tab.gamma_W[k])
    a, _ = domains.in_asu(_desc(sg, cell, t, R))
    b, _ = domains.in_asu(_desc(sg, cell, t, Qk @ R))
    assert bool((a ^ b).all())
    assert torch.equal(domains.in_domain(_desc(sg, cell, t, R))[0], a)
    frac = a.double().mean().item()
    assert abs(frac - 0.5) <= 5 * math.sqrt(0.25 / n)


@pytest.mark.parametrize('sg', [2, 14, 15])
def test_planar_mirror_images_stay_distinct(sg, planar_molecule, cell_sampler, literal_oracle):
    """The fold takes no molecule: an improper image of G re-expressed through the molecule's mirror plane is the same
    crystal (literal atoms) but folds to a different domain point (the exact multiplicity of an achiral molecule)."""
    tab = group_table(sg)
    X = planar_molecule['pos']
    assert np.abs(X[:, 2]).max() == 0                               # its mirror is z -> -z in the body frame
    sigma = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=F64))
    k = int(torch.nonzero(torch.round(torch.linalg.det(tab.g_W)) < 0)[0, 0])
    n = 200
    g = torch.Generator().manual_seed(6000 + sg)
    cell = _cells(sg, n, 6000 + sg, cell_sampler)
    d = _desc(sg, cell, torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g))
    img = ops.apply(d, tab.g_W[k], tab.g_w[k])
    assert bool((img.h == -1).all())
    mirrored = img.replace(R=img.pose @ sigma, h=torch.ones_like(img.h))  # P X = (P sigma)(sigma X), sigma X = X
    assert (torch.linalg.det(mirrored.R) - 1).abs().max() <= 1e-12
    W, w = tab.g_W.numpy(), tab.g_w.numpy()
    for i in range(10):
        a = literal_oracle.unit_cell(cell[i].numpy(), d.t[i, 0].numpy(), d.pose[i, 0].numpy(), X, W, w)
        b = literal_oracle.unit_cell(cell[i].numpy(), mirrored.t[i, 0].numpy(), mirrored.pose[i, 0].numpy(), X, W, w)
        assert literal_oracle.max_atom_deviation(a, b, cell[i].numpy()) <= 1e-9
    fa, fb = domains.fold(d), domains.fold(mirrored)
    gap = torch.maximum((fa.desc.t - fb.desc.t).abs().amax((1, 2)), (fa.desc.R - fb.desc.R).abs().amax((1, 2, 3)))
    assert gap.min() > 1e-3
    assert bool(domains.in_domain(fb.desc)[0].all())


# ---------------------------------------------------------------------------
# cell (triclinic)
# ---------------------------------------------------------------------------

FACES = ('a=b', 'b=c', 'xi=+B', 'xi=-B', 'eta=-A', 'zeta=-A', 'beta=90', 'gamma=90', 'three')
OFFSETS = (1e-7, 1e-4, 1e-2)


def _conditions(cells):
    return ops.tri_niggli_conditions(ops.metric_tensor(torch.as_tensor(cells, dtype=F64)))


def _on_face(rng, face, d, n, band=None):
    """n triclinic cells at signed offset d (> 0 inside) from ``face``, at least 2e-3 inside every other wall.

    With ``band``, the face's condition is only required to lie within ``band`` of 0 (d = 0 or +-1e-12: on the face
    up to rounding).
    """
    which = FACES.index(face)
    wall = {'a=b': 0, 'b=c': 1, 'xi=+B': 2, 'xi=-B': 2, 'eta=-A': 3, 'zeta=-A': 4, 'beta=90': 5, 'gamma=90': 6,
            'three': 7}[face]
    out, rounds = [], 0
    while len(out) < n:
        m = 4096
        L = np.sort(rng.uniform(4.0, 16.0, (m, 3)), axis=1)
        A = np.radians(np.stack([rng.uniform(60, 120, m), rng.uniform(90, 125, m), rng.uniform(90, 125, m)], 1))
        a, b, c = L.T
        with np.errstate(invalid='ignore'):
            if face == 'a=b':
                L[:, 0] = b * (1 - d)
            elif face == 'b=c':
                L[:, 1] = c * (1 - d)
                L[:, 0] = np.minimum(a, L[:, 1] * 0.99)
            elif face == 'xi=+B':
                A[:, 0] = np.arccos(b / (2 * c)) + d
            elif face == 'xi=-B':
                A[:, 0] = np.arccos(-b / (2 * c)) - d
            elif face == 'eta=-A':
                A[:, 1] = np.arccos(-a / (2 * c)) - d
            elif face == 'zeta=-A':
                A[:, 2] = np.arccos(-a / (2 * b)) - d
            elif face == 'beta=90':
                A[:, 1] = np.pi / 2 + d
            elif face == 'gamma=90':
                A[:, 2] = np.pi / 2 + d
            else:
                eta, zeta = 2 * a * c * np.cos(A[:, 1]), 2 * a * b * np.cos(A[:, 2])
                A[:, 0] = np.arccos(-(a * a + b * b + eta + zeta) / (2 * b * c)) - d
        cells = np.concatenate([L, A], 1)
        ok = np.isfinite(cells).all(1)
        ca, cb, cg = np.cos(A).T
        ok &= 1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg > 1e-3
        cells = cells[ok]
        g = _conditions(cells).numpy()
        others = np.delete(g, wall, axis=1)
        keep = (others >= 2e-3).all(1)
        if band is not None:
            keep &= np.abs(g[:, wall] - d) <= band
        elif d > 0:
            keep &= (g[:, wall] > 0)
        else:
            keep &= (g[:, wall] < 0)
        out.extend(cells[keep][: n - len(out)])
        rounds += 1
        if rounds > 200:
            raise RuntimeError(f'{face} at {d}: {len(out)} of {n} cells after 200 rounds')
    return np.array(out[:n]), which


def _interior(rng, n, margin=1e-2):
    """n triclinic cells at least ``margin`` inside every wall of the tri_niggli domain."""
    out = []
    for _ in range(200):                                     # bounded: about 1 in 20 draws is kept
        if len(out) >= n:
            break
        m = 4096
        L = np.sort(rng.uniform(4.0, 16.0, (m, 3)), axis=1)
        A = np.radians(np.stack([rng.uniform(60, 120, m), rng.uniform(90, 125, m), rng.uniform(90, 125, m)], 1))
        cells = np.concatenate([L, A], 1)
        ca, cb, cg = np.cos(A).T
        cells = cells[1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg > 1e-3]
        keep = (_conditions(cells) >= margin).all(1).numpy()
        out.extend(cells[keep][: n - len(out)])
    if len(out) < n:
        raise RuntimeError(f'_interior: {len(out)} of {n} cells after 200 rounds')
    return np.array(out[:n])


def _scramble(rng, n):
    """n random integer matrices with det +1 (products of signed permutations and shears with entries in [-2, 2])."""
    perms = [np.eye(3, dtype=np.int64)[list(p)] for p in itertools.permutations(range(3))]
    out = []
    for _ in range(100 * n):                                 # bounded: about half the products have det +1
        if len(out) >= n:
            break
        N = np.eye(3, dtype=np.int64)
        for _ in range(3):
            E = np.eye(3, dtype=np.int64)
            i, j = rng.choice(3, 2, replace=False)
            E[i, j] = rng.integers(-2, 3)
            P = perms[rng.integers(6)] * rng.choice([-1, 1], 3)[None]
            N = N @ E @ P
        if round(np.linalg.det(N)) == 1:
            out.append(N)
    if len(out) < n:
        raise RuntimeError(f'_scramble: {len(out)} of {n} det +1 matrices')
    return np.stack(out)


def _scrambled(cells, Ns):
    G = ops.metric_tensor(torch.as_tensor(cells, dtype=F64))
    Nf = torch.as_tensor(Ns, dtype=F64)
    return ops.cell_from_metric_tensor(Nf.transpose(1, 2) @ G @ Nf)


def _cell_dev(c1, c2):
    """Largest difference: lengths in Angstrom, angles in degrees."""
    return max((c1[:, :3] - c2[:, :3]).abs().max().item(), torch.rad2deg((c1[:, 3:] - c2[:, 3:]).abs()).max().item())


@pytest.mark.parametrize('face, sign', [('interior', 1)] + [(f, s) for f in FACES for s in (1, -1)])
def test_cell_canonicalisation_invariant(face, sign):
    """canonicalise(scramble(c)) == canonicalise(c) within 1e-9 at 1e-7, 1e-4, 1e-2 from each face (both sides)."""
    rng = np.random.default_rng(100 * (('interior',) + FACES).index(face) + (sign > 0))
    offsets = (None,) if face == 'interior' else OFFSETS
    for d in offsets:
        cells = _interior(rng, 60) if face == 'interior' else _on_face(rng, face, sign * d, 60)[0]
        base = torch.as_tensor(cells, dtype=F64)
        N0, ref, tied0, v0 = ops.triclinic_basis(base)
        assert v0.max() <= 1e-10
        for _ in range(3):
            Ns = _scramble(rng, len(cells))
            N1, out, tied1, v1 = ops.triclinic_basis(_scrambled(cells, Ns))
            assert v1.max() <= 1e-10
            dev = _cell_dev(out, ref)
            assert dev <= 1e-9, f'{face} at {sign * d}: canonical cells differ by {dev:.2e}'
        pen = tri_niggli_reduction_penalty(ref[:, :3], ref[:, 3:], margin=0.0)
        assert pen.max() <= 1e-18
        if face != 'interior' and sign > 0:
            assert _cell_dev(ref, base) <= 1e-9  # a cell inside the domain is its own canonical form


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_old_tie_break_control():
    """Control: standardize.choose_basis is not invariant within 1e-7 rad of beta = 90 deg; triclinic_basis is."""
    from mxtaltools.crystal_search.standardize import choose_basis
    rng = np.random.default_rng(77)
    cells = _on_face(rng, 'beta=90', 1e-7, 150)[0]
    Ns = _scramble(rng, len(cells))
    scr = _scrambled(cells, Ns).numpy()
    old_bad = 0
    for c, s in zip(cells, scr):
        _, L0, A0, _ = choose_basis(2, c[:3], c[3:])
        _, L1, A1, _ = choose_basis(2, s[:3], s[3:])
        old_bad += int(max(np.abs(L0 - L1).max(), np.degrees(np.abs(A0 - A1)).max()) > 1e-6)
    _, ref, _, _ = ops.triclinic_basis(torch.as_tensor(cells))
    _, new, _, _ = ops.triclinic_basis(torch.as_tensor(scr))
    assert old_bad > 0, 'the old tie-break control no longer fails; re-check what this control shows'
    assert _cell_dev(new, ref) <= 1e-9


#: offsets inside the tie band (tol = 1e-10): candidates on both sides of the face are valid and tie
EXACT = (0.0, 1e-12, -1e-12)


def _face_invariance(face, rng, n=60):
    """(largest canonical-cell difference over 3 scrambles, rows with tied candidates) for cells on ``face``."""
    worst, tied = 0.0, 0
    for d in EXACT:
        cells = _on_face(rng, face, d, n, band=1e-11)[0]
        _, ref, tied0, v0 = ops.triclinic_basis(torch.as_tensor(cells, dtype=F64))
        assert v0.max() <= 1e-10
        tied += int((tied0 > 1).sum())
        for _ in range(3):
            _, out, tied1, v1 = ops.triclinic_basis(_scrambled(cells, _scramble(rng, len(cells))))
            assert v1.max() <= 1e-10
            tied += int((tied1 > 1).sum())
            worst = max(worst, _cell_dev(out, ref))
    return worst, tied


@pytest.mark.parametrize('face', FACES)
def test_cell_tie_break_on_faces(face):
    """Cells on each face (offset 0, +-1e-12): candidates tie, and the tie-break returns one cell from every basis."""
    worst, tied = _face_invariance(face, np.random.default_rng(300 + FACES.index(face)))
    assert tied > 0, f'{face}: no row had tied candidates; the tie-break path did not run'
    assert worst <= 1e-9, f'{face}: canonical cells differ by {worst:.2e}'


def test_cell_tie_break_control(monkeypatch):
    """Control: a tie-break that takes the first valid candidate (the input basis decides) fails on the faces."""
    monkeypatch.setattr(ops, '_lex_select', lambda keys, valid, tol, largest: valid.to(torch.int8).argmax(1))
    for face in ('beta=90', 'a=b'):
        worst, tied = _face_invariance(face, np.random.default_rng(400 + FACES.index(face)))
        assert tied > 0 and worst > 1e-3, f'{face}: the mutated tie-break went unnoticed ({worst:.2e})'


def _cells_deg(n, a, b, c, al, be, ga, rng, jitter):
    """n cells (Angstrom, radians) with the given lengths and angles; ``jitter`` entries get a random +-5% / +-5 deg."""
    v = np.tile(np.array([a, b, c, al, be, ga], dtype=float), (n, 1))
    for k in jitter:
        if k < 3:
            v[:, k] *= 1 + 0.05 * rng.uniform(-1, 1, n)
        else:
            v[:, k] += 5 * rng.uniform(-1, 1, n)
    return torch.as_tensor(np.concatenate([v[:, :3], np.radians(v[:, 3:])], 1), dtype=F64)


#: (cells, proper lattice automorphisms): generic and single-face metrics are unflagged, special metrics flagged
METRICS = {
    'generic': (lambda n, r: _cells_deg(n, 6.0, 7.5, 8.0, 100, 95, 97, r, (0, 1, 2, 3, 4, 5)), 1),
    'beta=90': (lambda n, r: _cells_deg(n, 6.0, 7.5, 8.0, 100, 90, 97, r, (0, 1, 2, 3, 5)), 1),
    'a=b': (lambda n, r: _cells_deg(n, 7.0, 7.0, 8.0, 100, 95, 97, r, (2, 3, 4, 5)), 1),
    'beta=gamma=90': (lambda n, r: _cells_deg(n, 6.0, 7.5, 8.0, 100, 90, 90, r, (0, 1, 2, 3)), 2),
    'a=b, alpha=beta': (lambda n, r: _cells_deg(n, 7.0, 7.0, 8.0, 100, 100, 97, r, ()), 2),
    'all 90': (lambda n, r: _cells_deg(n, 6.0, 7.5, 8.0, 90, 90, 90, r, (0, 1, 2)), 4),
    'hexagonal': (lambda n, r: _cells_deg(n, 7.0, 7.0, 9.0, 90, 90, 120, r, ()), 12),
    'cubic': (lambda n, r: _cells_deg(n, 7.0, 7.0, 7.0, 90, 90, 90, r, ()), 24),
}


@pytest.mark.parametrize('sg', [1, 2])
@pytest.mark.parametrize('metric', list(METRICS))
def test_special_metrics_are_flagged(sg, metric):
    """n_auto counts the proper lattice automorphisms from any basis; unflagged rows (faces included) have one
    canonical description, cell and pose, from every basis."""
    rng = np.random.default_rng(500 + 10 * list(METRICS).index(metric) + sg)
    g = torch.Generator().manual_seed(500 + sg)
    make, expected = METRICS[metric]
    n = 40
    d = _desc(sg, make(n, rng), torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g))
    red0 = ops.reduce_cell(d)
    a = domains.fold(red0.desc).desc
    assert bool((red0.n_auto == expected).all()), (metric, sorted(set(red0.n_auto.tolist())))
    for _ in range(2):
        red = ops.reduce_cell(ops.change_basis(d, torch.as_tensor(_scramble(rng, n)), torch.zeros(n, 3, dtype=F64)))
        assert bool((red.n_auto == expected).all())
        assert _cell_dev(red.desc.cell, red0.desc.cell) <= 1e-9
        if expected == 1:
            b = domains.fold(red.desc).desc
            assert (b.t - a.t).abs().max() <= 1e-9 and (b.R - a.R).abs().max() <= 1e-9, metric


@pytest.mark.parametrize('metric', ['beta=gamma=90', 'all 90', 'hexagonal'])
def test_float32_special_metrics_are_flagged(metric):
    """A special metric stored in float32, as the CSD datasets store cells (a 90 deg angle is 4.4e-8 rad off), keeps
    its n_auto: the flag's tolerance (``ops.FACE_TOL``) is wider than float32 rounding."""
    rng = np.random.default_rng(700 + list(METRICS).index(metric))
    g = torch.Generator().manual_seed(700)
    make, expected = METRICS[metric]
    n = 60
    d = _desc(2, make(n, rng), torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g))
    f32 = d.to(dtype=torch.float32)
    assert f32.cell.dtype == torch.float32
    red = ops.reduce_cell(f32)
    assert bool((red.n_auto == expected).all()), (metric, sorted(set(red.n_auto.tolist())))


@pytest.mark.parametrize('sg', [1, 2])
def test_cell_then_pose_composition(sg, cell_sampler):
    """reduce_cell then fold gives one description per crystal from any proper basis (generic triclinic cells)."""
    rng = np.random.default_rng(sg)
    g = torch.Generator().manual_seed(sg)
    n = 300
    cells = torch.as_tensor(_interior(rng, n), dtype=F64)
    d = _desc(sg, cells, torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g))
    a = domains.fold(ops.reduce_cell(d).desc)
    for _ in range(2):
        scr = ops.change_basis(d, torch.as_tensor(_scramble(rng, n)), torch.zeros(n, 3, dtype=F64))
        red = ops.reduce_cell(scr)
        assert red.violation.max() <= 1e-10
        b = domains.fold(red.desc)
        assert _cell_dev(b.desc.cell, a.desc.cell) <= 1e-9
        assert (b.desc.t - a.desc.t).abs().max() <= 1e-9
        assert (b.desc.R - a.desc.R).abs().max() <= 1e-9

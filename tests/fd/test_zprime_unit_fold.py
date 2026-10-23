"""Stage Z2 of the Z' > 1 design (energy_sampling/docs/design/zprime_fd.md section 9): one unit into ASU(G) x SO(3).

``domains.fold_unit`` folds units, one per row, under G with its improper elements: (W, w) with lattice shift L acts
by (t, R, q) -> (W t + w + L, M_W R D^s, sigma^s q), s = [det W = -1], M_W = T W T^-1; the fold returns the folded
(t, R), the element, the shift and ``mirrored`` = s, and the caller applies sigma to q. Checks:

- one representative per unit orbit: every image of a random unit under all |G| elements, each with a random lattice
  shift in -2..2, folds to the unit's own fold within 1e-9, and ``mirrored`` of an image differs from the unit's
  exactly when the image's element is improper;
- that representative is the orbit's only image in ASU(G), counted apart from the fold (``Region`` of ``g_asu`` with
  its tie cuts, over every element and lattice translate; rows within 1e-6 of a facet are left out of the count);
- the folded centroid is in ASU(G) and R stays proper; the records reproduce the result (t = W t_in + w + lattice,
  R = M_W R_in D^s, ``mirrored`` = det W < 0);
- idempotence: a folded unit folds to itself, bitwise, by the identity with zero shift;
- exact special points (vertices, edge midpoints and facet centroids of ASU(G)) and points 3e-10 and 5e-8 off them
  (around the tol = 1e-9 band): every G image folds to one representative, with the same parity rule;
- literal atoms: a chain built from its internal coordinates in the seed frame (atom 0 at the origin, atom 1 on +x,
  atom 2 in the xy-plane, so X(sigma q) = D X(q) with D = diag(1, 1, -1)), placed at the folded (t, R) with sigma
  applied where ``mirrored``, equals the image of the input unit's atoms under the element and shift within 1e-9 A;
  controls without sigma, and with crystal_fd's x-reflection as D, fail;
- mixed space groups in one call; refusals (non-finite rows, improper R, no real cell, a D that is not a reflection).

The orbit test runs every group: ``FAST_SG`` in the default lane, the others marked ``slow``.
"""
import itertools

import numpy as np
import pytest
import torch

from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import domains, exact, ops, so3
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
TOL = 1e-9
SLAB = 1e-6
PAD = 1e-6
D_BODY = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=F64))
#: triclinic, monoclinic (sg 9: K), orthorhombic, a pair group (76, 92), tetragonal, trigonal (148, 166) and cubic
#: (198, 205, 227: the largest G) groups
FAST_SG = (1, 2, 4, 9, 14, 15, 19, 61, 76, 88, 92, 148, 166, 198, 205, 227)
ALL_SG = [sg if sg in FAST_SG else pytest.param(sg, marks=pytest.mark.slow) for sg in range(1, 231)]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cells(sg, n, seed, cell_sampler):
    return torch.tensor(cell_sampler(LATTICE_TYPE[sg], n, seed=seed), dtype=F64)


def _poses(cell, W, R, mirrored, D=D_BODY):
    """M_W R D^s for rows: cell [n, 6], W [n, 3, 3], R [n, 3, 3], mirrored [n] bool."""
    P = ops.cartesian_rotation(ops.cell_matrices(cell), W) @ R
    return torch.where(mirrored[:, None, None], P @ D, P)


def _images(sg, cell, t, R, gen, D=D_BODY):
    """Every G image of every unit with a random lattice shift in -2..2: (rows, element, t, R) over n x |G| rows."""
    tab = group_table(sg)
    n, m = len(t), len(tab.g_W)
    rows = torch.arange(n).repeat_interleave(m)
    el = torch.arange(m).repeat(n)
    shift = torch.randint(-2, 3, (n * m, 3), generator=gen).to(F64)
    ti = torch.einsum('bij,bj->bi', tab.g_W[el], t[rows]) + tab.g_w[el] + shift
    Ri = _poses(cell[rows], tab.g_W[el], R[rows], torch.linalg.det(tab.g_W[el]) < 0, D)
    return rows, el, ti, Ri


def _units(sg, n, seed, cell_sampler):
    gen = torch.Generator().manual_seed(seed)
    cell = _cells(sg, n, seed, cell_sampler)
    t = torch.rand(n, 3, generator=gen, dtype=F64) * 3 - 1
    return gen, cell, t, so3.random_rotations(n, gen)


def _g_asu(sg):
    tab = group_table(sg)
    return domains.Region.from_cuts(tab.g_asu.cuts, tab.g_asu.vertices)


def _count_in_asu(sg, cell, t, R):
    """Apart from ``fold_unit``: per unit, the images (every G element, every lattice translate that can reach the
    bounding box) in ASU(G) by ``Region.contains`` with its tie cuts. Returns count [n], the last in-domain image's
    centroid [n, 3], pose [n, 3, 3] and parity [n], and the smallest |slack| over the images [n]."""
    tab, reg = group_table(sg), _g_asu(sg)
    n, m = len(t), len(tab.g_W)
    ext = torch.floor(reg.hi - reg.lo + 2 * PAD).to(torch.long) + 1
    grid = torch.stack(torch.meshgrid(*[torch.arange(int(k), dtype=F64) for k in ext], indexing='ij'), -1).reshape(-1, 3)
    y = torch.einsum('mij,nj->nmi', tab.g_W, t) + tab.g_w
    x = y[:, :, None] + torch.ceil(reg.lo - PAD - y)[:, :, None] + grid          # [n, m, S, 3]
    s = reg.slack(x)
    inside = reg.contains(x, TOL, s)
    hit = torch.nonzero(inside)
    t_in = torch.zeros(n, 3, dtype=F64)
    el = torch.zeros(n, dtype=torch.long)
    t_in[hit[:, 0]] = x[hit[:, 0], hit[:, 1], hit[:, 2]]
    el[hit[:, 0]] = hit[:, 1]
    mir = torch.linalg.det(tab.g_W[el]) < 0
    return inside.sum((1, 2)), t_in, _poses(cell, tab.g_W[el], R, mir), mir, s.abs().amin((1, 2))


def _identity(sg):
    sec = domains.unit_section(sg)
    k = torch.nonzero((sec.W == torch.eye(3, dtype=F64)).all(-1).all(-1) & (sec.w == 0).all(-1))[:, 0]
    assert len(k) == 1
    return int(k[0])


def _assert_orbit_folds_once(sg, cell, t, R, gen, fu=None):
    """Every image of every unit folds to the unit's fold; ``mirrored`` follows the image's parity."""
    fu = domains.fold_unit(sg, cell, t, R) if fu is None else fu
    rows, el, ti, Ri = _images(sg, cell, t, R, gen)
    fi = domains.fold_unit(sg, cell[rows], ti, Ri)
    dt = (fi.t - fu.t[rows]).abs().amax(-1)
    dR = (fi.R - fu.R[rows]).abs().amax((-1, -2))
    assert float(dt.max()) <= 1e-9 and float(dR.max()) <= 1e-9, (sg, float(dt.max()), float(dR.max()))
    parity = torch.linalg.det(group_table(sg).g_W[el]) < 0
    assert torch.equal(fi.mirrored, fu.mirrored[rows] ^ parity), sg
    return fu


def _special_points(sg):
    """Exact vertices, edge midpoints and facet centroids of ASU(G) (from the table's rationals), float64 [k, 3]."""
    normals, offsets = exact.cut_planes(group_table(sg).g_asu.cuts)
    verts = exact.polytope_vertices(normals, offsets)
    tight = [frozenset(i for i, (n, c) in enumerate(zip(normals, offsets)) if exact.evaluate(n, c, v) == 0)
             for v in verts]
    pts = set(verts)
    for (a, ta), (b, tb) in itertools.combinations(zip(verts, tight), 2):
        if exact.rank([normals[i] for i in ta & tb]) == 2:
            pts.add(tuple((p + q) / 2 for p, q in zip(a, b)))
    for i in range(len(normals)):
        face = [v for v, tv in zip(verts, tight) if i in tv]
        if len(face) >= 3:
            pts.add(tuple(sum(v[j] for v in face) / len(face) for j in range(3)))
    return torch.tensor([[float(q) for q in p] for p in sorted(pts)], dtype=F64)


def _nerf(bonds, angles, dihedrals):
    """Chain atoms [n, 3] in the seed frame: atom 0 at the origin, atom 1 on +x, atom 2 in the xy-plane (y > 0)."""
    n = len(bonds) + 1
    X = np.zeros((n, 3))
    X[1] = [bonds[0], 0, 0]
    X[2] = X[1] + bonds[1] * np.array([-np.cos(angles[0]), np.sin(angles[0]), 0.0])
    for i in range(3, n):
        a, b, c = X[i - 3], X[i - 2], X[i - 1]
        bc = (c - b) / np.linalg.norm(c - b)
        nv = np.cross(b - a, bc)
        nv /= np.linalg.norm(nv)
        mv = np.cross(nv, bc)
        L, th, ph = bonds[i - 1], angles[i - 2], dihedrals[i - 3]
        X[i] = c - L * np.cos(th) * bc + L * np.sin(th) * np.cos(ph) * mv + L * np.sin(th) * np.sin(ph) * nv
    return X


def _chain(q):
    """Centred atoms [6, 3] of the 6-atom chain q = (5 bonds, 4 angles, 3 dihedrals)."""
    X = _nerf(q[:5], q[5:9], q[9:])
    return torch.tensor(X - X.mean(0), dtype=F64)


SIGMA_CHAIN = np.r_[np.ones(9), -np.ones(3)]


# ---------------------------------------------------------------------------
# one representative per unit orbit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', ALL_SG)
def test_one_representative_per_unit_orbit(sg, cell_sampler):
    """Random units: one image in ASU(G) by the cut trees, which the fold returns; every G image folds to it."""
    m = len(group_table(sg).g_W)
    n = max(12, 2400 // m)
    gen, cell, t, R = _units(sg, n, 50_000 + sg, cell_sampler)
    fu = _assert_orbit_folds_once(sg, cell, t, R, gen)
    count, t_in, P_in, mir_in, near = _count_in_asu(sg, cell, t, R)
    far = near > SLAB
    assert int(far.sum()) >= n - 2
    assert bool((count[far] == 1).all()), (sg, torch.unique(count[far], return_counts=True))
    assert float((fu.t - t_in)[far].abs().max()) <= 1e-9
    assert float((fu.R - P_in)[far].abs().max()) <= 1e-9
    assert torch.equal(fu.mirrored[far], mir_in[far])
    assert bool((fu.tie[far] == 0).all())


@pytest.mark.parametrize('sg', FAST_SG)
def test_fold_records_domain_and_properness(sg, cell_sampler):
    """The folded centroid is in ASU(G); R is proper; the records reproduce the fold; fold(fold(u)) == fold(u)."""
    sec = domains.unit_section(sg)
    gen, cell, t, R = _units(sg, 300, 51_000 + sg, cell_sampler)
    fu = domains.fold_unit(sg, cell, t, R)
    assert bool(_g_asu(sg).contains(fu.t, TOL).all())
    assert float((torch.linalg.det(fu.R) - 1).abs().max()) <= 1e-12
    assert float((fu.R.transpose(1, 2) @ fu.R - torch.eye(3, dtype=F64)).abs().max()) <= 1e-12
    assert torch.equal(fu.W, sec.W[fu.element]) and torch.equal(fu.w, sec.w[fu.element])
    assert torch.equal(fu.mirrored, sec.det[fu.element] < 0)
    assert float((torch.einsum('bij,bj->bi', fu.W, t) + fu.w + fu.lattice - fu.t).abs().max()) <= 1e-12
    assert float((_poses(cell, fu.W, R, fu.mirrored) - fu.R).abs().max()) <= 1e-12
    assert float((fu.slack - _g_asu(sg).slack(fu.t)).abs().max()) <= 1e-15
    again = domains.fold_unit(sg, cell, fu.t, fu.R)
    assert torch.equal(again.t, fu.t) and torch.equal(again.R, fu.R)
    assert bool((again.element == _identity(sg)).all()) and not bool(again.lattice.any())
    assert not bool(again.mirrored.any())


@pytest.mark.parametrize('sg', ALL_SG)
def test_special_and_band_points(sg, cell_sampler):
    """Exact vertices, edge midpoints and facet centroids of ASU(G), and points 3e-10 and 5e-8 off them: every G
    image folds to one representative, in ASU(G) for the exact points, with the parity rule."""
    gen = torch.Generator().manual_seed(52_000 + sg)
    pts = _special_points(sg)
    d = torch.randn(len(pts), 3, generator=gen, dtype=F64)
    d = d / d.norm(dim=1, keepdim=True)
    t = torch.cat([pts, pts + 3e-10 * d, pts + 5e-8 * d])
    cell = _cells(sg, len(t), 52_000 + sg, cell_sampler)
    R = so3.random_rotations(len(t), gen)
    fu = _assert_orbit_folds_once(sg, cell, t, R, gen)
    exact_pts = torch.arange(len(t)) < len(pts)
    assert bool(_g_asu(sg).contains(fu.t[exact_pts], TOL).all())
    assert bool((fu.tie[exact_pts] > 0).all())
    # every row, band rows included, ends no further than tol outside ASU(G): a fold that keeps only the base lattice
    # shift per element (dropping the shift grid) sends points near a full-period face out (stage Z2 review)
    assert float(fu.slack.min()) >= -TOL, (sg, float(fu.slack.min()))


# ---------------------------------------------------------------------------
# the conformer mirror, literally
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', (2, 14, 19, 61, 148))
def test_literal_atoms(sg, cell_sampler):
    """The folded unit's atoms, with sigma applied where mirrored, are the image of the input atoms (1e-9 A)."""
    n = 200
    gen, cell, t, R = _units(sg, n, 53_000 + sg, cell_sampler)
    rng = np.random.default_rng(53_000 + sg)
    q = np.c_[rng.uniform(1.3, 1.6, (n, 5)), np.radians(rng.uniform(100, 120, (n, 4))),
              rng.choice([-1, 1], (n, 3)) * rng.uniform(0.3, 2.8, (n, 3))]
    X = torch.stack([_chain(v) for v in q])                                      # [n, 6, 3]
    Xs = torch.stack([_chain(v * SIGMA_CHAIN) for v in q])
    assert float((Xs - X @ D_BODY).abs().max()) <= 1e-12                         # X(sigma q) = D X(q)
    T = ops.cell_matrices(cell)

    def deviation(fu, apply_sigma=True):
        M = T @ fu.W @ torch.linalg.inv(T)                                       # the isometry, unprojected
        image = X @ (M @ R).transpose(1, 2)
        Xq = torch.where(fu.mirrored[:, None, None] & apply_sigma, Xs, X)
        return (Xq @ fu.R.transpose(1, 2) - image).norm(dim=-1).amax(-1)

    fu = domains.fold_unit(sg, cell, t, R)
    assert float(deviation(fu).max()) <= 1e-9
    assert float((torch.einsum('bij,bj->bi', fu.W, t) + fu.w + fu.lattice - fu.t).abs().max()) <= 1e-12
    if bool((torch.linalg.det(group_table(sg).g_W) > 0).all()):
        assert not bool(fu.mirrored.any())                                      # Sohncke: G is proper
        return
    mir = fu.mirrored
    assert 0.2 * n < int(mir.sum()) < 0.8 * n
    assert float(deviation(fu, apply_sigma=False)[mir].min()) > 0.05            # control: sigma omitted
    fx = domains.fold_unit(sg, cell, t, R, D=torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=F64)))
    assert torch.equal(fx.element, fu.element) and torch.equal(fx.t, fu.t)
    assert float(deviation(fx)[mir].min()) > 0.05                               # control: the x-reflection as D
    assert float(deviation(fx)[~mir].max()) <= 1e-9


def test_default_d_is_the_seed_plane_reflection(cell_sampler):
    gen, cell, t, R = _units(14, 50, 54_014, cell_sampler)
    a, b = domains.fold_unit(14, cell, t, R), domains.fold_unit(14, cell, t, R, D=D_BODY)
    assert torch.equal(a.R, b.R) and torch.equal(a.t, b.t) and bool(a.mirrored.any())


# ---------------------------------------------------------------------------
# batches and refusals
# ---------------------------------------------------------------------------

def test_mixed_space_groups(cell_sampler):
    """Rows of several groups in one call equal the per-group folds."""
    parts = [(sg,) + _units(sg, 40, 55_000 + sg, cell_sampler)[1:] for sg in (2, 14, 19, 148)]
    sg = torch.cat([torch.full((len(p[1]),), p[0], dtype=torch.long) for p in parts])
    fu = domains.fold_unit(sg, torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]),
                           torch.cat([p[3] for p in parts]))
    lo = 0
    for s, cell, t, R in parts:
        one = domains.fold_unit(s, cell, t, R)
        hi = lo + len(t)
        assert torch.equal(fu.t[lo:hi], one.t) and torch.equal(fu.R[lo:hi], one.R)
        assert torch.equal(fu.element[lo:hi], one.element) and torch.equal(fu.mirrored[lo:hi], one.mirrored)
        lo = hi


def test_refusals(cell_sampler):
    gen, cell, t, R = _units(14, 4, 56_014, cell_sampler)
    bad = t.clone()
    bad[2, 1] = float('nan')
    with pytest.raises(ValueError, match='rows'):
        domains.fold_unit(14, cell, bad, R)
    improper = R.clone()
    improper[1] = improper[1] @ D_BODY
    with pytest.raises(ValueError, match='rows'):
        domains.fold_unit(14, cell, t, improper)
    flat = cell.clone()
    flat[3, 3:] = torch.tensor([0.0, 0.0, 0.0], dtype=F64)
    with pytest.raises(ValueError, match='rows'):
        domains.fold_unit(14, flat, t, R)
    for D in (torch.eye(3, dtype=F64), so3.random_rotations(1, gen)[0] @ D_BODY,
              torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=F64))):
        with pytest.raises(ValueError, match='reflection'):
            domains.fold_unit(14, cell, t, R, D=D)

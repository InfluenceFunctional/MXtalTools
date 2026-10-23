"""Monoclinic cells (sg 3-15): the cell canonicaliser and the monoclinic CellChart (fd_tooling.md sections 2.6, 4).

The setting is SYM_OPS (b unique, cell choice 1). A basis change is a' = p a + q c, c' = r a + s c, b' = (ps - qr) b
with ps - qr = +-1, carrying the origin shift ``standardize.setting_shift`` gives it; the setting classes of
``sym_utils.MONO_CLASS`` restrict it to r even (GLIDE: 7, 13, 14), q even (CENTRED: 5, 8, 9, 12, 15) or nothing
(FREE: 3, 4, 6, 10, 11). Groups in the fast lane: 14, 15, 4, 9, 5, 7 (every class; 9 and 15 need origin shifts; 7 and
9 have free axes and K = C2(b)); the other seven are ``slow``.

1. Setting group: ``ops.mono_parity`` accepts exactly the changes ``setting_shift`` accepts (every change with
   entries in [-3, 3] and det +-1, in each of sg 3-15).
2. Zero set: ``ops.mono_conditions`` >= 0 exactly where the live ``mono_reduction_penalty`` at margin 0 is 0.
3. Cell invariance (gate 1, cell): descriptions in the interior and at 1e-7, 1e-4 and 1e-2 on both sides of every
   face of their class (W1 beta = 90 deg; W2, W3 the class's two bounds on -cos beta; FREE a = c; offsets in rad
   for W1, relative otherwise), each re-described by three random class-allowed basis changes (products of three
   shears, sign changes and swaps, with their origin shifts), canonicalise to one cell within 1e-9 (Angstrom,
   degrees), and, after the fold, to one pose within 1e-9; unflagged (n_auto = 1) beyond the flag's band
   (``ops.FACE_TOL`` = 1e-5; at 1e-7 and 1e-4 rows may be flagged). A cell inside the domain is its own canonical
   form (N = I).
4. Faces (offset 0 and +-1e-12, inside the tie band tol = 1e-10): candidates tie, the canonical cell is one from
   every basis, every row is flagged n_auto = 2 (each face is a mirror whose reflection is in the setting class),
   and the flag is real: some rows get two canonical poses from two bases. The tie band (3e-11 to 1e-8 on both sides
   of every face): every row is flagged from every description, the least violating candidate (the one inside the
   domain) is taken, so cell and pose are one from every basis, and the canonical cell is in the chart's bounds.
   Cells stored in float32 on a face (beta = 90 deg as CIFs report it, the other setting (a, -b, -c) of it, and
   every other face) are flagged.
5. Special metrics: n_auto equals an independent count (every 2x2 change with entries in [-4, 4] that keeps the
   reduced metric and the class allows, modulo +-I) for generic (1), face (2), square (a = c, beta = 90 deg: 2, FREE
   4) and hexagonal (a = c, beta = 120 deg: 2, FREE 6) ac nets, from any basis.
6. Composition (cell, then pose): a crystal and a random Gamma+ image of it in a random allowed basis have one
   canonical description (cell, t, R within 1e-9) and literally the same atoms (``checks.same_crystal`` <= 1e-9
   Angstrom). The reduction moves no atom: the old atoms carried into the new basis by x -> N^-1 (x - s), with no
   frame rotation, are the new description's atoms within 1e-9 Angstrom (a pose left un-re-expressed is > 0.1 off).
   The fold moves the crystal by its recorded normaliser element only (unfold returns the reduced description).
7. Search and guards: the correction span 1 and the default span 3 choose the same bases on strongly skewed
   lattices (the search has a margin); a parity check switched off picks bases outside the setting group, which
   reduce_cell refuses; cells with alpha or gamma off 90 deg are refused, float32 rounding is accepted, and alpha
   and gamma within ``angle_tol`` of 90 deg are set to 90 deg before the pose is re-expressed (the result is the
   snapped row's, atoms included); mixed triclinic and monoclinic batches reduce row by row as alone; other systems
   are refused; empty batches give empty results; ``face_tol`` below ``tol`` is refused; in a fresh interpreter the
   monoclinic path loads no cctbx or GFN module.
8. Chart (gates 3, 5, 8): y -> cell -> y within 1e-12 (interior and 1e-9, 1e-6, 1e-3 and 0 from every bound);
   log|det| = log m - log sin(beta) written out, against autograd and fourth-order central differences (1e-8),
   lower triangular, at the kink of m; the chart's image is the live penalty's zero set within the trainer's length
   box (both ways), and a face pushed out by 1% breaks that; sg 10 (FREE, whose trainer box has lo_a > lo_c, so the
   a <= c bound on log c binds) in the round trip and zero-set tests; CrystalChart.for_latent_v1 points are canonical.
9. Energy (gate 4): a crystal's canonical form, reached from a Gamma+ image in a random basis, has its eLJ through
   ``MolCrystalData.analyze(['elj'])`` (relative 1e-3 or absolute 5e-3 raw), as does an allowed unreduced
   description; a basis change outside the setting group, an origin shift withheld (sg 9, 15) and a pose left
   un-re-expressed each change it (controls at >= 0.1 raw on all rows but one, median >= 0.5).
"""
import itertools
import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mxtaltools.common.sym_utils import MONO_CENTRED, MONO_CLASS, MONO_FREE, MONO_GLIDE, cell_reduction_penalty
from mxtaltools.crystal_fd import charts, checks, domains, measure, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_search.standardize import setting_shift
from tests.fd import _helpers as H
from tests.fd.test_gate4_energy import _assert_control, _close, _length_box
from tests.fd.test_gate5_jacobians import _check_square

F64 = torch.float64
RADIUS = 1.7561
SGS = (14, 15, 4, 9, 5, 7)
ALL_MONO = [sg if sg in SGS else pytest.param(sg, marks=pytest.mark.slow) for sg in range(3, 16)]
FACES = {MONO_GLIDE: ('W1', 'W2', 'W3'), MONO_CENTRED: ('W1', 'W2', 'W3'), MONO_FREE: ('W1', 'W3', 'order')}
WALL = {'W1': 0, 'W2': 1, 'W3': 2, 'order': 3}
OFFSETS = (1e-7, 1e-4, 1e-2)
EXACT = (0.0, 1e-12, -1e-12)
CASES = [(sg, f) for sg in SGS for f in FACES[int(MONO_CLASS[sg])]]
_SWAP = np.array([[0, 1], [1, 0]])


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cls(sg):
    return int(MONO_CLASS[sg])


def _embed(M):
    """[n, 2, 2] integer ac changes -> [n, 3, 3] long N = [[p, 0, r], [0, ps - qr, 0], [q, 0, s]]."""
    M = np.asarray(M, dtype=np.int64)
    N = np.zeros((len(M), 3, 3), dtype=np.int64)
    N[:, 0, 0], N[:, 0, 2], N[:, 2, 0], N[:, 2, 2] = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
    N[:, 1, 1] = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    return torch.as_tensor(N)


def _elementary(cls, rng):
    """One ac change the class allows: a' = a + k c or c' = c + k a (k in [-2, 2]) or a sign change of a and/or c,
    each followed by the swap of a and c with probability 0.3."""
    for _ in range(1000):                                    # bounded: about half the draws are allowed
        kind, k = int(rng.integers(3)), int(rng.integers(-2, 3))
        E = (np.array([[1, 0], [k, 1]]), np.array([[1, k], [0, 1]]), np.diag(rng.choice([-1, 1], 2)))[kind]
        if rng.random() < 0.3:
            E = E @ _SWAP
        if bool(ops.mono_parity(torch.as_tensor(E), torch.tensor(cls))):
            return E
    raise RuntimeError('no allowed elementary change in 1000 draws')


def _scramble(sg, n, rng, depth=3):
    """[n, 3, 3] long random allowed basis changes: products of ``depth`` elementary ones (entries up to ~12)."""
    out = []
    for _ in range(n):
        M = np.eye(2, dtype=np.int64)
        for _ in range(depth):
            M = M @ _elementary(_cls(sg), rng)
        out.append(M)
    return _embed(np.stack(out))


def _mono_cells(a, b, c, cb):
    m = len(a)
    return np.stack([a, b, c, np.full(m, np.pi / 2), np.arccos(np.clip(cb, -1, 1)), np.full(m, np.pi / 2)], 1)


def _conditions(cells, sg):
    return ops.mono_conditions(ops.metric_tensor(torch.as_tensor(cells, dtype=F64)), sg)


def _cells_at(sg, face, d, n, rng, band=None):
    """n cells (Angstrom, rad; alpha = gamma = 90 deg) whose ``face`` condition is d (> 0 inside), at least 2e-3
    inside every other wall of the class. With ``band`` the face condition must lie within ``band`` of d (d = 0 or
    +-1e-12: on the face up to rounding); otherwise it must have the sign of d."""
    cls, w = _cls(sg), WALL[face]
    out = []
    for _ in range(200):                                     # bounded: a few percent of draws are kept at worst
        if len(out) >= n:
            break
        m = 4096
        a, b, c = (rng.uniform(4.0, 16.0, m) for _ in range(3))
        cb = -rng.uniform(0.0, 0.75, m)
        if face == 'W1':
            cb = np.full(m, -d)
        elif face == 'W2':
            cb = -(1 - d) * (a / c if cls == MONO_GLIDE else c / a)
        elif face == 'W3':
            cb = -(1 - d) * (c / (2 * a) if cls == MONO_GLIDE else a / (2 * c))
        else:
            a = c * (1 - d)
        cells = _mono_cells(a, b, c, cb)
        g = _conditions(cells, sg).numpy()
        keep = (np.abs(cb) < 0.99) & (np.delete(g, w, axis=1) >= 2e-3).all(1)
        if band is not None:
            keep &= np.abs(g[:, w] - d) <= band
        else:
            keep &= (g[:, w] > 0) if d > 0 else (g[:, w] < 0)
        out.extend(cells[keep][: n - len(out)])
    if len(out) < n:
        raise RuntimeError(f'sg {sg} {face} at {d}: {len(out)} of {n} cells after 200 rounds')
    return torch.as_tensor(np.array(out[:n]), dtype=F64)


def _interior(sg, n, rng, margin=1e-2):
    """n cells at least ``margin`` inside every wall of the class's domain."""
    out = []
    for _ in range(200):
        if len(out) >= n:
            break
        m = 4096
        cells = _mono_cells(*(rng.uniform(4.0, 16.0, m) for _ in range(3)), -rng.uniform(0.0, 0.75, m))
        out.extend(cells[(_conditions(cells, sg) >= margin).all(1).numpy()][: n - len(out)])
    if len(out) < n:
        raise RuntimeError(f'_interior: {len(out)} of {n} cells after 200 rounds')
    return torch.as_tensor(np.array(out[:n]), dtype=F64)


def _desc(sg, cells, gen):
    n = len(cells)
    return CrystalDescription.from_parts(sg, cells, torch.rand(n, 3, generator=gen, dtype=F64),
                                         so3.random_rotations(n, gen))


def _cell_dev(c1, c2):
    """[B] largest difference: lengths in Angstrom, angles in degrees."""
    return torch.maximum((c1[:, :3] - c2[:, :3]).abs().amax(1), torch.rad2deg((c1[:, 3:] - c2[:, 3:]).abs()).amax(1))


def _pose_dev(a, b):
    """[B] largest difference of fractional centroids and rotation-matrix entries."""
    return torch.maximum((a.t - b.t).abs().amax((1, 2)), (a.R - b.R).abs().amax((1, 2, 3)))


def _penalty0(cells, sg):
    c = torch.as_tensor(cells, dtype=F64)
    return cell_reduction_penalty(c[:, 3:], c[:, :3], torch.full((len(c),), sg), margin=0.0)


def _random_image(d, gen):
    """One random Gamma+ image of each row of a one-group batch (``domains.images``: row i's images are consecutive)."""
    imgs, src, _ = domains.images(d)
    k = len(src) // d.num_rows
    assert torch.equal(src, torch.arange(d.num_rows).repeat_interleave(k))
    return imgs[torch.arange(d.num_rows) * k + torch.randint(k, (d.num_rows,), generator=gen)]


def _basis_change_deviation(old, new, N, s, mol):
    """[B] Angstrom: the atoms of ``old`` carried into ``new``'s basis by x -> N^-1 (x - s) (fractional; no frame
    rotation is involved) against the atoms ``new`` describes. Atoms are compared in order within a molecule,
    molecules matched to their nearest counterpart both ways, differences at the minimum image in ``new``'s cell."""
    fo = checks.unit_cell_fractional(old, mol)                                  # [B, k, n, 3]
    fn = checks.unit_cell_fractional(new, mol)
    Ninv = torch.linalg.inv(torch.as_tensor(N).to(F64))
    fo = torch.einsum('bij,bknj->bkni', Ninv, fo - torch.as_tensor(s, dtype=F64)[:, None, None])
    df = fn[:, :, None] - fo[:, None]                                           # [B, k, l, n, 3]
    df = df - torch.round(df)
    dist = torch.einsum('bij,bklnj->bklni', new.cell_matrix(), df).norm(dim=-1).amax(-1)
    return torch.maximum(dist.amin(2).amax(1), dist.amin(1).amax(1))


# ---------------------------------------------------------------------------
# 1, 2: the setting group and the zero set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', range(3, 16))
def test_class_parity_is_the_setting_group(sg):
    M = np.array(list(itertools.product(range(-3, 4), repeat=4)), dtype=np.int64).reshape(-1, 2, 2)
    M = M[np.abs(M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]) == 1]
    par = ops.mono_parity(torch.as_tensor(M), torch.tensor(_cls(sg)))
    kept = torch.tensor([setting_shift(N, sg) is not None for N in _embed(M).numpy()])
    assert len(M) == 232 and torch.equal(par, kept), int((par != kept).sum())
    assert 0 < int(kept.sum()) <= 232 and (int(kept.sum()) == 232) == (_cls(sg) == MONO_FREE)


@pytest.mark.parametrize('sg', range(3, 16))
def test_conditions_are_the_penalty_zero_set(sg):
    rng = np.random.default_rng(10 + sg)
    m = 200_000
    a, b, c = (np.exp(rng.uniform(np.log(2.0), np.log(40.0), m)) for _ in range(3))
    cells = _mono_cells(a, b, c, np.cos(np.radians(rng.uniform(60.0, 150.0, m))))
    g = _conditions(cells, sg)
    zero = _penalty0(cells, sg) == 0
    clear = g.abs().amin(1) > 1e-12                          # rows not within rounding of a wall
    assert int(clear.sum()) > 0.99 * m and int((zero & clear).sum()) > 1000
    assert torch.equal((g >= 0).all(1)[clear], zero[clear])


# ---------------------------------------------------------------------------
# 3, 4: cell invariance, faces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg, face, sign', [(sg, 'interior', 1) for sg in SGS] +
                         [(sg, f, s) for sg, f in CASES for s in (1, -1)])
def test_cell_canonicalisation_invariant(sg, face, sign):
    """canonicalise(scramble(c)) == canonicalise(c) within 1e-9 at 1e-7, 1e-4, 1e-2 from each face (both sides);
    unflagged rows, so the canonical pose is one from every basis too."""
    seed = 1000 * sg + 10 * (['interior', 'W1', 'W2', 'W3', 'order'].index(face)) + (sign > 0)
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    for d in ((None,) if face == 'interior' else OFFSETS):
        cells = _interior(sg, 60, rng) if face == 'interior' else _cells_at(sg, face, sign * d, 60, rng)
        desc = _desc(sg, cells, gen)
        red0 = ops.reduce_cell(desc)
        can0 = domains.fold(red0.desc).desc
        flag_free = d is None or d >= 100 * ops.FACE_TOL         # beyond the flag's band (nearer, rows may be flagged)
        assert float(red0.violation.max()) <= 1e-10 and bool((red0.n_tied == 1).all()), (face, d)
        assert not flag_free or bool((red0.n_auto == 1).all()), (face, d)
        assert float(_penalty0(red0.desc.cell, sg).max()) <= 1e-20
        assert bool((red0.desc.cell[:, [3, 5]] == math.pi / 2).all())
        if sign > 0:                                         # a cell inside the domain is its own canonical form
            assert float(_cell_dev(red0.desc.cell, cells).max()) <= 1e-9
            assert bool((red0.N == torch.eye(3, dtype=torch.long)).all())
        for _ in range(3):
            red = ops.reduce_cell(ops.change_basis(desc, _scramble(sg, len(cells), rng)))
            assert float(red.violation.max()) <= 1e-10 and (not flag_free or bool((red.n_auto == 1).all()))
            dev = float(_cell_dev(red.desc.cell, red0.desc.cell).max())
            assert dev <= 1e-9, f'sg {sg} {face} at {sign * d}: canonical cells differ by {dev:.2e}'
            pose = float(_pose_dev(domains.fold(red.desc).desc, can0).max())
            assert pose <= 1e-9, f'sg {sg} {face} at {sign * d}: canonical poses differ by {pose:.2e}'


@pytest.mark.parametrize('sg, face', CASES)
def test_faces_tie_and_are_flagged(sg, face):
    """On a face (offset 0, +-1e-12) candidates tie; the cell is one from every basis and every row is flagged."""
    seed = 2000 * sg + WALL[face]
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    tied = differ = 0
    for d in EXACT:
        cells = _cells_at(sg, face, d, 40, rng, band=1e-11)
        desc = _desc(sg, cells, gen)
        red0 = ops.reduce_cell(desc)
        can0 = domains.fold(red0.desc).desc
        assert float(red0.violation.max()) <= 1e-10 and bool((red0.n_auto == 2).all()), sorted(set(red0.n_auto.tolist()))
        tied += int((red0.n_tied > 1).sum())
        for _ in range(3):
            red = ops.reduce_cell(ops.change_basis(desc, _scramble(sg, len(cells), rng)))
            assert bool((red.n_auto == 2).all())
            tied += int((red.n_tied > 1).sum())
            assert float(_cell_dev(red.desc.cell, red0.desc.cell).max()) <= 1e-9
            differ += int((_pose_dev(domains.fold(red.desc).desc, can0) > 1e-6).sum())
    assert tied > 0, f'sg {sg} {face}: no row had tied candidates; the tie path did not run'
    assert differ > 0, f'sg {sg} {face}: no flagged row had two canonical poses; re-check what the flag marks'


#: offsets (relative, rad for W1) across the tie band (tol = 1e-10) and out to 1e-8, both sides of the face
BAND = (3e-11, 6e-11, 1e-10, 1.5e-10, 3e-10, 1e-9, 1e-8)


def _in_chart_bounds(cell, sg):
    """[B] bool: the monoclinic chart coordinates of ``cell`` lie within its bounds (a wide length box)."""
    ch = charts.CellChart('monoclinic', (torch.full((3,), 0.1, dtype=F64), torch.full((3,), 1e3, dtype=F64)), sg=sg)
    y, _ = ch.unmap(cell)
    ok = torch.ones(len(y), dtype=torch.bool)
    for i in range(4):
        lo, hi = ch.bounds(y[:, :i], i)
        ok &= (y[:, i] >= lo) & (y[:, i] <= hi)
    return ok


@pytest.mark.parametrize('sg, face', CASES)
def test_tie_band_is_flagged_and_canonical(sg, face):
    """Rows 3e-11 to 1e-8 from a face (inside and around the tie band), as a random Gamma+ image in random allowed
    bases: flagged from every description; one canonical cell and pose (the least violating candidate, the one
    inside the domain, is taken); the canonical cell is on the zero set and in the chart's bounds. Exact faces
    (offset 0, +-1e-12) are flagged and in bounds too."""
    seed = 2500 * sg + WALL[face]
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    for d in [s * x for x in BAND for s in (1, -1)] + list(EXACT):
        exact = d in EXACT
        cells = _cells_at(sg, face, d, 30, rng, band=1e-11 if exact else None)
        desc = _desc(sg, cells, gen)
        red0 = ops.reduce_cell(desc)
        can0 = domains.fold(red0.desc).desc
        for k in range(3):
            red = red0 if k == 0 else ops.reduce_cell(ops.change_basis(_random_image(desc, gen),
                                                                      _scramble(sg, len(cells), rng)))
            assert bool((red.n_auto >= 2).all()), (d, sorted(set(red.n_auto.tolist())))
            assert float(_penalty0(red.desc.cell, sg).max()) <= 1e-20
            assert bool(_in_chart_bounds(red.desc.cell, sg).all()), d
            assert float(_cell_dev(red.desc.cell, red0.desc.cell).max()) <= 1e-9, d
            if not exact:
                pose = float(_pose_dev(domains.fold(red.desc).desc, can0).max())
                assert pose <= 1e-9, f'sg {sg} {face} at {d}: canonical poses differ by {pose:.2e}'


@pytest.mark.parametrize('sg', SGS)
def test_float32_faces_are_flagged(sg):
    """Cells on a face stored in float32, as the CSD datasets store them (a 90 deg angle is 4.4e-8 rad off): flagged.
    For beta = 90 deg both settings a source may report, (a, b, c) and (a, -b, -c) (beta' = 180 deg - beta) with
    beta' written as 90 deg again, are flagged; their canonical poses may differ, which is what the flag marks."""
    rng, gen = np.random.default_rng(3500 + sg), torch.Generator().manual_seed(3500 + sg)
    for face in FACES[_cls(sg)]:
        d = _desc(sg, _cells_at(sg, face, 0.0, 30, rng, band=1e-11), gen)
        rows = [d.to(dtype=torch.float32)]
        if face == 'W1':
            other = ops.change_basis(d, torch.diag(torch.tensor([1, -1, -1])))
            c = other.cell.clone()
            c[:, 4] = math.pi / 2
            rows.append(other.replace(cell=c).to(dtype=torch.float32))
        for r in rows:
            assert r.cell.dtype == torch.float32
            red = ops.reduce_cell(r)
            assert bool((red.n_auto >= 2).all()), (face, sorted(set(red.n_auto.tolist())))
            assert bool(_in_chart_bounds(red.desc.cell, sg).all()), face
    with pytest.raises(ValueError, match='face_tol'):
        ops.reduce_cell(d, face_tol=1e-11)


def _brute_n_auto(cell, sg):
    """[B] changes with entries in [-4, 4], det +-1, that keep the ac metric (relative 1e-9) and the class allows,
    modulo +-I (independent of ``monoclinic_basis``'s conjugation)."""
    M = np.array(list(itertools.product(range(-4, 5), repeat=4)), dtype=np.int64).reshape(-1, 2, 2)
    M = torch.as_tensor(M[np.abs(M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]) == 1])
    c = torch.as_tensor(cell, dtype=F64)
    gac = c[:, 0] * c[:, 2] * torch.cos(c[:, 4])
    g = torch.stack([c[:, 0] ** 2, gac, gac, c[:, 2] ** 2], 1).reshape(-1, 2, 2)
    Mf = M.to(F64)
    gm = Mf.transpose(1, 2)[None] @ g[:, None] @ Mf[None]
    scale = (c[:, 0] ** 2 + c[:, 2] ** 2)[:, None, None, None] / 2
    same = ((gm - g[:, None]).abs() <= 1e-9 * scale).all(-1).all(-1)
    return (same & ops.mono_parity(M, torch.tensor(_cls(sg)))[None]).sum(1) // 2


def _special(kind, sg, n, rng):
    """n cells of a metric ``kind``: 'generic', a face name, 'square' (a = c, beta = 90) or 'hexagonal' (a = c, beta
    = 120); lengths jittered, the defining relation kept exactly."""
    if kind == 'generic':
        return _interior(sg, n, rng)
    if kind in WALL:
        return _cells_at(sg, kind, 0.0, n, rng, band=1e-11)
    a = rng.uniform(5.0, 12.0, n)
    beta = {'square': 90.0, 'hexagonal': 120.0}[kind]
    return torch.as_tensor(_mono_cells(a, rng.uniform(5.0, 12.0, n), a, np.full(n, math.cos(math.radians(beta)))),
                           dtype=F64)


@pytest.mark.parametrize('sg', SGS)
def test_special_metrics_are_flagged(sg):
    """n_auto equals the brute-force count from every basis: 1 generic, 2 on each face, more on the square and
    hexagonal ac nets; unflagged rows have one canonical description from every basis."""
    rng, gen = np.random.default_rng(3000 + sg), torch.Generator().manual_seed(3000 + sg)
    seen = {}
    for kind in ('generic',) + FACES[_cls(sg)] + ('square', 'hexagonal'):
        d = _desc(sg, _special(kind, sg, 30, rng), gen)
        red0 = ops.reduce_cell(d)
        brute = _brute_n_auto(red0.desc.cell, sg)
        assert torch.equal(red0.n_auto, brute), (kind, red0.n_auto.tolist(), brute.tolist())
        seen[kind] = sorted(set(brute.tolist()))
        a = domains.fold(red0.desc).desc
        for _ in range(2):
            red = ops.reduce_cell(ops.change_basis(d, _scramble(sg, d.num_rows, rng)))
            assert torch.equal(red.n_auto, red0.n_auto)
            assert float(_cell_dev(red.desc.cell, red0.desc.cell).max()) <= 1e-9
            if kind == 'generic':
                assert float(_pose_dev(domains.fold(red.desc).desc, a).max()) <= 1e-9
    assert seen['generic'] == [1] and all(seen[f] == [2] for f in FACES[_cls(sg)])
    assert seen['square'] == ([4] if _cls(sg) == MONO_FREE else [2])
    assert seen['hexagonal'] == ([6] if _cls(sg) == MONO_FREE else [2])


# ---------------------------------------------------------------------------
# 6: composition with the pose fold, atoms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', ALL_MONO)
def test_cell_then_pose_composition(sg, chiral_molecule):
    mol = H.mol_ref(chiral_molecule)
    rng, gen = np.random.default_rng(4000 + sg), torch.Generator().manual_seed(4000 + sg)
    near = [_cells_at(sg, f, s * 1e-4, 20, rng) for f in FACES[_cls(sg)] for s in (1, -1)]
    cells = torch.cat([_interior(sg, 120, rng)] + near)
    d = _desc(sg, cells, gen)
    d2 = ops.change_basis(_random_image(d, gen), _scramble(sg, d.num_rows, rng))
    r1, r2 = ops.reduce_cell(d), ops.reduce_cell(d2)
    f1, f2 = domains.fold(r1.desc), domains.fold(r2.desc)
    assert bool((r1.n_auto == 1).all()) and bool((r2.n_auto == 1).all())
    # one canonical description
    assert float(_cell_dev(f1.desc.cell, f2.desc.cell).max()) <= 1e-9
    assert float(_pose_dev(f1.desc, f2.desc).max()) <= 1e-9
    # literally one crystal
    assert float(checks.same_crystal(f1.desc, f2.desc, mol).max()) <= 1e-9
    # the reduction moved no atom (old atoms carried by x -> N^-1 (x - s), no frame rotation involved)
    dev = _basis_change_deviation(d2, r2.desc, r2.N, r2.s, mol)
    assert float(dev.max()) <= 1e-9, float(dev.max())
    # the fold moved the crystal by its recorded normaliser element only
    back = domains.unfold(f2.desc, f2.element, f2.lattice, f2.free_shift)
    assert float(checks.same_crystal(back, r2.desc, mol).max()) <= 1e-9
    # controls, each > 0.1 Angstrom on every row: the rotation of the Cartesian frame withheld from R (rows whose frame
    # turns by more than 0.3 rad), a change outside the setting class read with zero shift, an origin shift withheld
    turned = _frame_turn(d2, r2) > 0.3
    unrotated = r2.desc.replace(R=d2.R)
    ctl = _basis_change_deviation(d2[turned], unrotated[turned], r2.N[turned], r2.s[turned], mol)
    assert int(turned.sum()) >= 0.3 * d.num_rows and float(ctl.min()) > 0.1, float(ctl.min())
    zero = torch.zeros(d.num_rows, 3, dtype=F64)
    if FORBIDDEN[_cls(sg)] is not None:
        N = _embed(np.array([FORBIDDEN[_cls(sg)]] * d.num_rows))
        ctl = _basis_change_deviation(r2.desc, ops.change_basis(r2.desc, N, zero), N, zero, mol)
        assert float(ctl.min()) > 0.1, float(ctl.min())
    N = _embed(np.array([ALLOWED[_cls(sg)]] * d.num_rows))
    s = ops.mono_setting_shifts(N, r2.desc.sg)
    assert float(_basis_change_deviation(r2.desc, ops.change_basis(r2.desc, N), N, s, mol).max()) <= 1e-9
    if sg in (9, 15):
        ctl = _basis_change_deviation(r2.desc, ops.change_basis(r2.desc, N, zero), N, zero, mol)
        assert float(ctl.min()) > 0.1, float(ctl.min())


def _frame_turn(old, red):
    """[B] angle (rad) of the change of Cartesian frame Q = T_new N^-1 T_old^-1 of a reduction."""
    Q = red.desc.cell_matrix() @ torch.linalg.inv(red.N.to(F64)) @ torch.linalg.inv(old.cell_matrix())
    return torch.arccos(((Q.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1))


# ---------------------------------------------------------------------------
# 7: search margin and guards
# ---------------------------------------------------------------------------

def test_search_span_has_a_margin(monkeypatch):
    """Strongly skewed lattices (a, c log-uniform on [2, 60] Angstrom, beta in [20, 160] deg): correction span 1 and
    the default span 3 choose the same bases, so the default searches past what is needed."""
    rng = np.random.default_rng(5)
    out = {}
    for span in (3, 1):
        if span == 1:
            ops.monoclinic_basis(torch.tensor([[6.0, 7, 8, math.pi / 2, 1.8, math.pi / 2]], dtype=F64), 14)
            monkeypatch.setattr(ops, '_MONO_CANDIDATES', ops._square_neighbourhood(1))
        rng = np.random.default_rng(5)
        for sg in (14, 15, 4):
            m = 50_000
            a, c = (np.exp(rng.uniform(np.log(2.0), np.log(60.0), m)) for _ in range(2))
            cells = torch.as_tensor(_mono_cells(a, np.full(m, 7.0), c, np.cos(np.radians(rng.uniform(20, 160, m)))))
            N, cell, _, viol, _ = ops.monoclinic_basis(cells, sg)
            assert float(viol.max()) <= 1e-10
            out[span, sg] = (N, cell)
    assert len(ops._square_neighbourhood(1)) < len(ops._square_neighbourhood(3))
    for sg in (14, 15, 4):
        assert torch.equal(out[1, sg][0], out[3, sg][0]) and torch.equal(out[1, sg][1], out[3, sg][1])


def test_parity_check_is_load_bearing(monkeypatch):
    """Control: with the class parity switched off, the search picks bases outside the setting group (a GLIDE or
    CENTRED domain holds three GL2(Z) images of each lattice), and reduce_cell refuses them."""
    for sg in (14, 15):
        rng, gen = np.random.default_rng(sg), torch.Generator().manual_seed(sg)
        d = _desc(sg, _interior(sg, 60, rng), gen)
        d = ops.change_basis(d, _scramble(sg, 60, rng))
        ops.reduce_cell(d)
        with monkeypatch.context() as mp:
            mp.setattr(ops, 'mono_parity', lambda M, cls: torch.ones(M.shape[:-2], dtype=torch.bool, device=M.device))
            with pytest.raises(RuntimeError, match='outside the setting group'):
                ops.reduce_cell(d)
    with pytest.raises(ValueError, match='not in the setting group'):
        ops.change_basis(d[:1], _embed(np.array([[[1, 0], [1, 1]]])))     # q odd: outside CENTRED (sg 15)


def test_guards_and_mixed_batches():
    gen = torch.Generator().manual_seed(9)
    rng = np.random.default_rng(9)
    cells = _interior(14, 6, rng)
    d = _desc(14, cells, gen)
    off = d.cell.clone()
    off[2, 3] += 1e-3
    with pytest.raises(ValueError, match=r'rows \[2\]'):
        ops.reduce_cell(d.replace(cell=off))
    f32 = d.to(dtype=torch.float32)                                  # alpha, gamma 4e-8 rad off: accepted, snapped
    red = ops.reduce_cell(f32)
    assert bool((red.desc.cell[:, [3, 5]] == math.pi / 2).all()) and float(red.violation.max()) <= 1e-10
    tri = CrystalDescription.from_parts(2, torch.tensor([[6.0, 7.5, 8.0, 1.75, 1.66, 1.69]] * 4, dtype=F64),
                                        torch.rand(4, 3, generator=gen, dtype=F64), so3.random_rotations(4, gen))
    mixed = CrystalDescription.cat([tri[:2], d[:3], tri[2:], d[3:]])
    rm = ops.reduce_cell(mixed)
    alone = (ops.reduce_cell(tri), ops.reduce_cell(d))
    rows_t, rows_m = [0, 1, 5, 6], [2, 3, 4, 7, 8, 9]
    for rows, ref in ((rows_t, alone[0]), (rows_m, alone[1])):
        for f in ('N', 'n_tied', 'n_auto'):
            assert torch.equal(getattr(rm, f)[rows], getattr(ref, f)), f
        for a, b in ((rm.s[rows], ref.s), (rm.violation[rows], ref.violation), (rm.desc.cell[rows], ref.desc.cell),
                     (rm.desc.t[rows], ref.desc.t), (rm.desc.R[rows], ref.desc.R)):
            assert float((a - b).abs().max()) <= 1e-13
    bad = mixed.cell.clone()
    bad[8, 5] -= 1e-3                                                 # a monoclinic row, named by its batch index
    with pytest.raises(ValueError, match=r'rows \[8\]'):
        ops.reduce_cell(mixed.replace(cell=bad))
    with pytest.raises(NotImplementedError):
        ops.reduce_cell(d.replace(sg=torch.full_like(d.sg, 19)))



@pytest.mark.parametrize('sg', (14, 15, 4))
def test_alpha_gamma_are_set_to_90_before_the_pose(sg, chiral_molecule):
    """Rows with alpha or gamma 5e-7 rad off 90 deg (within angle_tol) reduce exactly as the same rows at 90 deg: the
    angles are set before the pose is re-expressed, so cell, centroid, R and atoms are the snapped row's (re-expressed
    in the unsnapped frame, R would move by about 1e-7). Inputs are scrambled, so most basis changes are not I."""
    mol = H.mol_ref(chiral_molecule)
    rng, gen = np.random.default_rng(6000 + sg), torch.Generator().manual_seed(6000 + sg)
    d = ops.change_basis(_desc(sg, _interior(sg, 40, rng), gen), _scramble(sg, 40, rng))
    cell = d.cell.clone()
    cell[:, 3] = cell[:, 5] = math.pi / 2
    d = d.replace(cell=cell)
    off = cell.clone()
    off[:20, 3] += 5e-7
    off[20:, 5] -= 5e-7
    r_off, r_ref = ops.reduce_cell(d.replace(cell=off)), ops.reduce_cell(d)
    assert int((r_ref.N != torch.eye(3, dtype=torch.long)).any((1, 2)).sum()) >= 30   # rows with N = I test nothing
    assert torch.equal(r_off.N, r_ref.N) and torch.equal(r_off.desc.cell, r_ref.desc.cell)
    assert float((r_off.desc.t - r_ref.desc.t).abs().max()) <= 1e-12
    assert float((r_off.desc.R - r_ref.desc.R).abs().max()) <= 1e-12
    assert float(checks.same_crystal(r_off.desc, r_ref.desc, mol).max()) <= 1e-9


def test_empty_batches():
    """An empty batch (e.g. a per-group mask that selects nothing) gives empty results in both systems."""
    gen = torch.Generator().manual_seed(11)
    none = torch.zeros(0, dtype=torch.long)
    for sg, cells in ((14, _interior(14, 3, np.random.default_rng(11))),
                      (2, torch.tensor([[6.0, 7.5, 8.0, 1.75, 1.66, 1.69]] * 3, dtype=F64))):
        red = ops.reduce_cell(_desc(sg, cells, gen)[none])
        assert red.desc.num_rows == 0 and red.N.shape == (0, 3, 3) and red.s.shape == (0, 3)
        assert red.n_tied.shape == red.violation.shape == red.n_auto.shape == (0,)
    N, cell, tied, viol, auto = ops.monoclinic_basis(torch.zeros(0, 6, dtype=F64), 14)
    assert N.shape == (0, 3, 3) and cell.shape == (0, 6) and tied.shape == viol.shape == auto.shape == (0,)
    N, cell, tied, viol = ops.triclinic_basis(torch.zeros(0, 6, dtype=F64))
    assert N.shape == (0, 3, 3) and cell.shape == (0, 6) and tied.shape == viol.shape == (0,)

_FRESH = '''
import sys, torch
from mxtaltools.crystal_fd import charts, domains, ops
ch = charts.CrystalChart.for_latent_v1(14, 1.7, 'euler_zyz')
m = ch.forward(charts.from_unit(ch, torch.rand(8, ch.ndim, dtype=torch.float64))[0])
d = ops.change_basis(m.desc, torch.tensor([[1, 0, 2], [0, 1, 0], [0, 0, 1]]))
domains.fold(ops.reduce_cell(d).desc)
ops.reduce_cell(d.replace(sg=torch.full_like(d.sg, 15)))
bad = [k for k in sys.modules if k.split('.')[0] in ('cctbx', 'libtbx', 'scitbx', 'boost_adaptbx')]
assert not bad, bad
gfn = sys.argv[1]
files = [str(getattr(v, '__file__', None) or '').replace(chr(92), '/').lower() for v in list(sys.modules.values())]
loaded = [f for f in files if f.startswith(gfn)]
assert not loaded, loaded
'''


def test_monoclinic_path_loads_no_cctbx_or_gfn():
    """In a fresh interpreter the monoclinic reduction, chart and fold load neither cctbx nor any GFN module."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(ops.__file__).parents[2]
    gfn = str(root.parent / 'gfn_diffusion').replace(chr(92), '/').lower()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1',
               PYTHONPATH=os.pathsep.join([str(root), os.environ.get('PYTHONPATH', '')]))
    r = subprocess.run([sys.executable, '-c', _FRESH, gfn], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


# ---------------------------------------------------------------------------
# 8: the chart
# ---------------------------------------------------------------------------

CHART_SGS = (14, 15, 4)                                      # one per class
#: plus sg 10: FREE, and its trainer box has lo_a > lo_c, so the chart's a <= c bound on log c binds
BOX_SGS = CHART_SGS + (10,)


def _chart(sg, radius=RADIUS):
    return charts.CellChart('monoclinic', charts.latent_v1_length_box(sg, radius), sg=sg)


def _unit(k, gen, n=500, edges=(1e-9, 1e-6, 1e-3, 0.0)):
    u = [0.01 + 0.98 * torch.rand(n, k, generator=gen, dtype=F64)]
    for i in range(k):
        for e in edges:
            for x in (e, 1 - e):
                v = 0.01 + 0.98 * torch.rand(20, k, generator=gen, dtype=F64)
                v[:, i] = x
                u.append(v)
    return torch.cat(u)


@pytest.mark.parametrize('sg', BOX_SGS)
def test_chart_round_trip(sg):
    ch = _chart(sg)
    assert [d.kind for d in ch.dims] == ['interval'] * 4 and ch.ndim == 4
    y, _ = charts.from_unit(ch, _unit(4, torch.Generator().manual_seed(sg)))
    cell, ld = ch.map(y)
    yy, ldi = ch.unmap(cell)
    assert float((yy - y).abs().max()) <= 1e-12 and float((ld + ldi).abs().max()) <= 1e-12
    cell2, _ = ch.map(yy)
    assert float(((cell2 - cell).abs() / cell.abs()).max()) <= 1e-12
    assert bool((cell[:, [3, 5]] == math.pi / 2).all())


def _natural(ch):
    def f(y):
        cell, _ = ch.map(y)
        return torch.stack([torch.log(cell[:, 2]), torch.log(cell[:, 0]), torch.log(cell[:, 1]), cell[:, 4]], 1)
    return f


@pytest.mark.parametrize('sg', CHART_SGS)
def test_chart_jacobian(sg):
    """Against autograd and central differences, lower triangular; at the kink of m (GLIDE c^2 = 2 a^2, CENTRED
    a^2 = 2 c^2), both sides, which enters below the diagonal only."""
    ch = _chart(sg)
    g = torch.Generator().manual_seed(40 + sg)
    y, _ = charts.from_unit(ch, _unit(4, g, edges=(1e-6,)))
    if ch.mono_class != MONO_FREE:
        k = y[:60].clone()
        sign = 1 if ch.mono_class == MONO_GLIDE else -1           # log a = log c -+ log(2) / 2 at the kink
        k[:, 1] = k[:, 0] - sign * 0.5 * math.log(2) + torch.cat([torch.full((30,), 1e-6), torch.full((30,), -1e-6)]).to(F64)
        y = torch.cat([y, k])
    cell, ld = ch.map(y)
    _check_square(_natural(ch), y, ld)
    m = ops.mono_bound(cell[:, 0], cell[:, 2], torch.full((len(y),), ch.mono_class))
    assert float((ld - (torch.log(m) - torch.log(torch.sin(cell[:, 4])))).abs().max()) <= 1e-12


@pytest.mark.parametrize('radius', [1.7561, 1.1])
@pytest.mark.parametrize('sg', BOX_SGS)
def test_chart_image_is_the_zero_set_in_the_length_box(sg, radius):
    lo, hi = charts.latent_v1_length_box(sg, radius)
    if sg == 10:
        assert float(lo.reshape(-1)[0]) > float(lo.reshape(-1)[2]), 'sg 10 has lo_a <= lo_c; pick a box where it binds'
    ch = charts.CellChart('monoclinic', (lo, hi), sg=sg)
    g = torch.Generator().manual_seed(int(100 * radius) + sg)
    _zero_set_equality(ch, sg, lo, hi, g, expect=True)


class _Pushed(charts.CellChart):
    """Mutation control: the class bound m taken 1% wider (the beta face pushed out)."""
    def _mono_bound(self, a, c):
        return 1.01 * super()._mono_bound(a, c)


def test_chart_zero_set_control():
    lo, hi = charts.latent_v1_length_box(14, RADIUS)
    _zero_set_equality(_Pushed('monoclinic', (lo, hi), sg=14), 14, lo, hi, torch.Generator().manual_seed(7),
                       expect=False)


def _zero_set_equality(ch, sg, lo, hi, g, expect):
    u = torch.rand(100_000, 4, generator=g, dtype=F64)
    u[:6000] = torch.where(torch.rand(6000, 4, generator=g) < 0.3, torch.randint(0, 2, (6000, 4), generator=g).to(F64),
                           u[:6000])
    cells, _ = ch.map(charts.from_unit(ch, u)[0])
    pen = _penalty0(cells, sg)
    rel = 1e-12
    in_box = bool(((cells[:, :3] >= lo * (1 - rel)) & (cells[:, :3] <= hi * (1 + rel))).all())
    n = 400_000
    v = torch.rand(n, 4, generator=g, dtype=F64)
    logL = torch.log(lo / 2) + v[:, :3] * (torch.log(2 * hi) - torch.log(lo / 2))
    beta = torch.deg2rad(50.0 + 100.0 * v[:, 3])
    half = torch.full((n,), math.pi / 2, dtype=F64)
    cand = torch.cat([torch.exp(logL), torch.stack([half, beta, half], 1)], 1)
    on = (_penalty0(cand, sg) == 0) & H.in_box(cand, lo, hi)
    yc, _ = ch.unmap(cand)
    inside = torch.ones(n, dtype=torch.bool)
    for i in range(4):
        blo, bhi = ch.bounds(yc[:, :i], i)
        inside &= (yc[:, i] >= blo - 1e-12) & (yc[:, i] <= bhi + 1e-12)
    if expect:
        assert float(pen.max()) <= 1e-20, float(pen.max())
        assert in_box and int(on.sum()) > 1000
        assert torch.equal(inside, on), (int((inside & ~on).sum()), int((on & ~inside).sum()))
    else:
        assert float(pen.max()) > 0 and not torch.equal(inside, on)


@pytest.mark.parametrize('sg', (14, 15, 4, 5))
def test_crystal_chart_points_are_canonical(sg):
    ch = charts.CrystalChart.for_latent_v1(sg, RADIUS, 'euler_zyz')
    assert ch.cell.family == 'monoclinic' and ch.ndim == 4 + ch.centroid.ndim + 3
    g = torch.Generator().manual_seed(60 + sg)
    y, _ = charts.from_unit(ch, 0.01 + 0.98 * torch.rand(500, ch.ndim, generator=g, dtype=F64))
    m = ch.forward(y)
    yy, ldi = ch.inverse(m.desc)
    assert float((yy - y).abs().max()) <= 1e-12 and float((m.logabsdet + ldi).abs().max()) <= 1e-12
    red = ops.reduce_cell(m.desc)
    assert bool((red.N == torch.eye(3, dtype=torch.long)).all())
    assert float(_cell_dev(red.desc.cell, m.desc.cell).max()) <= 1e-12
    assert bool(domains.in_domain(m.desc)[0].all())
    lp = measure.log_reference_density(m.desc, ch, 'latent_v1_lebesgue')
    assert bool(torch.isfinite(lp).all())
    with pytest.raises(NotImplementedError):
        measure.log_reference_density(m.desc, ch, 'metric_invariant')


def test_crystal_chart_guards():
    lo, hi = charts.latent_v1_length_box(14, RADIUS)
    with pytest.raises(ValueError):
        charts.CrystalChart(15, charts.CellChart('monoclinic', (lo, hi), sg=14), charts.PolytopeARChart(15),
                            charts.OrientationChart('euler_zyz'))           # GLIDE chart, CENTRED group
    for sg in (6, 7, 8, 9):                                                  # K = C2(b): no SO(3)/K chart
        with pytest.raises(NotImplementedError):
            charts.CrystalChart.for_latent_v1(sg, RADIUS, 'euler_zyz')
    with pytest.raises(ValueError):
        charts.CellChart('monoclinic', (lo, hi))                             # no setting class
    with pytest.raises(ValueError):
        charts.CellChart('orthorhombic', (lo, hi))                           # no axis-order class (built 2026-10-21)
    with pytest.raises(ValueError):
        charts.CellChart('rhombohedral', (lo, hi))                           # not a family


# ---------------------------------------------------------------------------
# 9: energy
# ---------------------------------------------------------------------------

N_E = 16
#: (allowed unreduced change, forbidden change read with zero shift) per class, as (p, r, q, s)-shaped 2x2 [[p, r], [q, s]]
ALLOWED = {MONO_GLIDE: [[1, 2], [0, 1]], MONO_CENTRED: [[1, 1], [0, 1]], MONO_FREE: [[1, 1], [0, 1]]}
FORBIDDEN = {MONO_GLIDE: [[1, 1], [0, 1]], MONO_CENTRED: [[1, 0], [1, 1]], MONO_FREE: None}


@pytest.fixture(scope='module', params=SGS, ids=lambda sg: f'sg{sg}')
def energy_case(request, chiral_molecule):
    sg = request.param
    mol = H.mol_ref(chiral_molecule)
    g = torch.Generator().manual_seed(70 + sg)
    lo, hi = _length_box(sg, mol)
    ch = charts.CellChart('monoclinic', (lo, hi), sg=sg)
    cells, _ = ch.map(charts.from_unit(ch, 0.02 + 0.96 * torch.rand(N_E, 4, generator=g, dtype=F64))[0])
    R = torch.as_tensor(Rotation.random(N_E, random_state=70 + sg).as_matrix(), dtype=F64)
    d = CrystalDescription.from_parts(sg, cells, 0.001 + 0.997 * torch.rand(N_E, 3, generator=g, dtype=F64), R)
    return sg, mol, d, H.elj(d, mol)


def test_canonical_form_has_the_source_energy(energy_case):
    sg, mol, d, e0 = energy_case
    g = torch.Generator().manual_seed(80 + sg)
    d2 = ops.change_basis(_random_image(d, g), _scramble(sg, N_E, np.random.default_rng(80 + sg)))
    red = ops.reduce_cell(d2)
    canon = domains.fold(red.desc).desc
    e = H.elj(canon, mol)
    assert bool(_close(e, e0).all()), float((e - e0).abs().max())
    allowed = ops.change_basis(d, _embed(np.array([ALLOWED[_cls(sg)]] * N_E)))   # unreduced; the builder copes
    e1 = H.elj(allowed, mol)
    assert bool(_close(e1, e0).all()), float((e1 - e0).abs().max())


def test_wrong_basis_changes_change_the_energy(energy_case):
    sg, mol, d, e0 = energy_case
    zero = torch.zeros(N_E, 3, dtype=F64)
    if FORBIDDEN[_cls(sg)] is not None:                                         # outside the setting group
        bad = ops.change_basis(d, _embed(np.array([FORBIDDEN[_cls(sg)]] * N_E)), zero)
        _assert_control((H.elj(bad, mol) - e0).abs(), 'outside the setting group')
    if sg in (9, 15):                                                           # its origin shift withheld
        N = _embed(np.array([ALLOWED[_cls(sg)]] * N_E))
        assert bool((ops.mono_setting_shifts(N, d.sg).abs().sum(1) > 0).all())
        _assert_control((H.elj(ops.change_basis(d, N, zero), mol) - e0).abs(), 'origin shift withheld')
    d2 = ops.change_basis(CrystalDescription.cat([d] * 4), _scramble(sg, 4 * N_E, np.random.default_rng(90 + sg)))
    red = ops.reduce_cell(d2)
    turned = torch.nonzero(_frame_turn(d2, red) > 0.3)[:, 0][:N_E]          # frames turned by more than 0.3 rad
    assert len(turned) >= N_E // 2
    unrotated = red.desc.replace(R=d2.R)[turned]                                # cell and centroid re-expressed, R not
    _assert_control((H.elj(unrotated, mol) - e0[turned % N_E]).abs(), 'frame rotation withheld from R')

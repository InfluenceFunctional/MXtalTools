"""GATE 3 (fd_tooling.md section 4): round trips through the charts and the float64 literal-atom oracle, sg 1 and 2.

1. chart -> description -> chart, float64, for the domain chart of sg 1 and 2 with each orientation chart:
   interior points, every coordinate at 1e-9, 1e-6 and 1e-3 from both ends of its range, circle seams (0 and 1
   of a circle's range; circle coordinates compare modulo their period), and the orientation charts near their
   singular sets (r -> 0, pi, 2 pi; theta -> 0 and the equator; cos beta -> +-1; |v| -> pi).
   - The description returns within 1e-12 everywhere (cell relative, t fractional, R entries).
   - Every coordinate returns within 1e-12 where the chart is regular. At a coordinate singularity a matrix holds
     an ill-conditioned coordinate only to about eps / (distance to the singular set): there each coordinate is
     held to 1e-12 + 1e-15 * its condition number (1 / sin(r/2), 1 / (sin(r/2) sin theta), 1 / (1 - |cos beta|)).
   - Every chart point is canonical: in the domain, and its own fold.
2. description -> canonical -> atoms, against the conftest float64 literal oracle, for random crystals in a random
   basis: the reduced crystal's atoms, carried back by the change of Cartesian frame Q = T_new N^-1 T_old^-1, are
   the original atoms within 1e-10 Angstrom (and the reduced cell re-expressed in the original basis is the same
   crystal by checks.same_crystal), and the canonical crystal's atoms are the reduced crystal's atoms moved by the
   fold's normaliser element (Q x + T (w + lattice + free shift)) within 1e-10 Angstrom. Control: an improper
   normaliser element read at +1 moves atoms by more than 0.1 Angstrom, by both measures.
3. The float32 builder path: descriptions, their folds and their Gamma+ images, centroids wrapped, built by the live
   builder (MolCrystalData, mol2ucell with std_orientation=False) put every atom within 1e-3 Angstrom of the float64
   oracle; the same descriptions with R transposed (a rotation-convention error) are more than 0.1 Angstrom off.
   This pins CrystalDescription.crystal_params / from_crystal_params to the trainer's builder.
4. The trainer latent in float32: latent -> description -> latent within 1e-6 latent units where the orientation
   coordinates are well conditioned (theta, r >= 0.1), and the rotation itself within 1e-6 everywhere.
5. Coordinate kinds: a chart coordinate is labelled a circle (with its period) iff its two ends describe one crystal
   (the descriptions there fold to one point), for PolytopeARChart in sg 2, 14, 19 and 92 and LatentV1Chart in
   sg 1 and 2; LatentV1Chart's 'fixed' rows are exactly the free-axis centroid rows.
6. checks.orbit_check counts one image per orbit (and a pushed facet does not); checks.round_trip.
"""
import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import charts, checks, domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.domains import Region
from mxtaltools.crystal_fd.tables import group_table
from tests.fd import _helpers as H

F64 = torch.float64
RADIUS = 1.7561
EDGES = (1e-9, 1e-6, 1e-3)
ORIENTATIONS = ('hemisphere', 'euler_zyz', 'axis_angle')


def _unit_points(chart, gen, n_interior=2000):
    k = chart.ndim
    u = [0.01 + 0.98 * torch.rand(n_interior, k, generator=gen, dtype=F64)]
    for i, d in enumerate(chart.dims):
        edges = [e for x in EDGES for e in (x, 1 - x)] + ([0.0, 1.0] if d.kind == 'circle' else [])
        for e in edges:
            v = 0.01 + 0.98 * torch.rand(30, k, generator=gen, dtype=F64)
            v[:, i] = e
            u.append(v)
    o = chart.offsets[2]
    name = chart.orientation.name
    if name == 'hemisphere':                                    # r near pi (both sides), theta at the equator and r
        for e in (1e-9, 1e-6, 1e-3):
            v = 0.01 + 0.98 * torch.rand(60, k, generator=gen, dtype=F64)
            v[:30, o + 2], v[30:, o + 2] = 0.5 - e, 0.5 + e
            u.append(v)
    return torch.cat(u)


def _condition(name, y):
    """[n, 3] condition numbers of the orientation coordinates at y (1 where the chart is regular)."""
    one = torch.ones(len(y), dtype=F64)
    if name == 'hemisphere':
        s = torch.sin(y[:, 2] / 2).abs()
        return torch.stack([1 / s, 1 / (s * torch.sin(y[:, 0])), one], 1)
    if name == 'euler_zyz':
        k = 1 / (1 - y[:, 1].abs())
        return torch.stack([k, one, k], 1)
    return torch.stack([one, one, one], 1)


def _wrapped_diff(chart, a, b):
    d = (a - b).abs()
    for i, dim in enumerate(chart.dims):
        if dim.kind == 'circle':
            p = dim.period
            d[:, i] = torch.remainder(a[:, i] - b[:, i] + p / 2, p) - p / 2
            d[:, i] = d[:, i].abs()
    return d


@pytest.mark.parametrize('sg', [1, 2])
@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_chart_description_chart(sg, orientation):
    chart = charts.CrystalChart.for_latent_v1(sg, RADIUS, orientation)
    g = torch.Generator().manual_seed(sg)
    y, _ = charts.from_unit(chart, _unit_points(chart, g))
    if orientation == 'axis_angle':                             # |v| -> pi from inside, and v -> 0
        o = chart.offsets[2]
        n = torch.randn(60, 3, generator=g, dtype=F64)
        n = n / n.norm(dim=1, keepdim=True)
        extra = y[:60].clone()
        extra[:, o:] = n * torch.tensor([math.pi - 1e-9, math.pi - 1e-6, 1e-9] * 20, dtype=F64)[:, None]
        y = torch.cat([y, extra])
    m = chart.forward(y)
    y2, ld2 = chart.inverse(m.desc)
    back = chart.forward(y2).desc
    o = chart.offsets[2]
    assert ((back.cell - m.desc.cell).abs() / m.desc.cell.abs()).max() <= 1e-12
    assert (back.t - m.desc.t).abs().max() <= 1e-12
    eR = (back.R - m.desc.R).abs().amax((1, 2, 3))
    if orientation == 'euler_zyz':      # a float64 cos(beta) holds beta only to eps / sin(beta) near the poles
        assert bool((eR <= 1e-12 + 1e-15 / torch.sqrt(1 - y[:, o + 1] ** 2)).all()), float(eR.max())
    else:
        assert float(eR.max()) <= 1e-12
    err = _wrapped_diff(chart, y2, y)
    allowed = torch.full_like(err, 1e-12)
    allowed[:, o:] = 1e-12 + 1e-15 * _condition(orientation, y[:, o:])
    assert bool((err <= allowed).all()), float((err - allowed).max())
    regular = (_condition(orientation, y[:, o:]) <= 1e3).all(1)
    assert regular.float().mean() > 0.9 and float(err[regular].max()) <= 1e-12
    assert (ld2 + m.logabsdet)[regular].abs().max() <= 1e-9
    # chart points are canonical, except within the fold's face tolerance (1e-9) of an excluded (seam) face,
    # where the fold returns the glued point: the same pose moved by the seam's translation
    inside, _ = domains.in_domain(m.desc)
    fr = domains.fold(m.desc)
    dt = fr.desc.t - m.desc.t
    assert (dt[inside].abs().max() if inside.any() else 0) <= 1e-12
    assert (fr.desc.R - m.desc.R).abs().max() <= 1e-12
    if sg == 2:
        assert [d.kind for d in chart.centroid.dims] == ['circle'] * 3       # the seam branch below runs
        seam = ~inside
        assert bool(seam.any()) and seam.float().mean() < 0.05
        assert bool(((m.desc.t[seam, 0] - 0.5).abs() <= 1e-9).any(1).all())     # on an upper face, within tol
        assert (dt[seam] - torch.round(2 * dt[seam]) / 2).abs().max() <= 1e-12   # moved by halves (the gluing)
    else:
        assert bool(inside.all())


def _random_crystals(sg, n, seed):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    lo, hi = torch.tensor([4.0, 4.0, 4.0], dtype=F64), torch.tensor([12.0, 12.0, 12.0], dtype=F64)
    cells = H.physical_cells(lo, hi, n, 'latent_v1_lebesgue', g)
    R = torch.as_tensor(Rotation.random(n, random_state=seed).as_matrix(), dtype=F64)
    d0 = CrystalDescription.from_parts(sg, cells, torch.rand(n, 3, generator=g, dtype=F64), R)
    return H.scramble(d0, rng)


def _oracle_atoms(desc, i, X, tab, oracle):
    return oracle.unit_cell(desc.cell[i].numpy(), desc.t[i, 0].numpy(), desc.pose[i, 0].numpy(), X,
                            tab.g_W.numpy(), tab.g_w.numpy())


@pytest.mark.parametrize('sg', [1, 2])
def test_canonical_atoms_match_the_literal_oracle(sg, chiral_molecule, literal_oracle):
    """The cell step moves no atom; the fold moves the crystal by exactly its normaliser element (an isometry)."""
    mol = H.mol_ref(chiral_molecule)
    d = _random_crystals(sg, 300, 30 + sg)
    canon, red, fr = H.canonicalise(d)
    back = ops.change_basis(red.desc, torch.linalg.inv(red.N.to(F64)).round(), torch.zeros(d.num_rows, 3, dtype=F64))
    dev = checks.same_crystal(d, back, mol)
    assert float(dev.max()) <= 1e-10, float(dev.max())
    tab = group_table(sg)
    sec = domains.translation_section(sg)
    X = mol.pos.numpy()
    worst = 0.0
    T_old, T_red = d.cell_matrix().numpy(), red.desc.cell_matrix().numpy()
    for i in range(d.num_rows):                                  # the conftest oracle, row by row
        A = _oracle_atoms(d, i, X, tab, literal_oracle)
        B = _oracle_atoms(back, i, X, tab, literal_oracle)
        worst = max(worst, literal_oracle.max_atom_deviation(A, B, d.cell[i].numpy()))
        # the reduced crystal itself, carried back to the old Cartesian frame by Q^T, Q = T_new N^-1 T_old^-1
        Q = T_red[i] @ np.linalg.inv(red.N[i].double().numpy()) @ np.linalg.inv(T_old[i])
        C = _oracle_atoms(red.desc, i, X, tab, literal_oracle) @ Q
        worst = max(worst, literal_oracle.max_atom_deviation(A, C, d.cell[i].numpy()))
        # reduced -> canonical: the image of the reduced crystal under the fold's element, lattice and free shift
        T = red.desc.cell_matrix()[i].numpy()
        e = int(fr.element[i])
        W = sec.W[e].numpy()
        shift = (sec.w[e] + fr.lattice[i] + fr.free_shift[i]).numpy()
        Q = T @ W @ np.linalg.inv(T)
        image = _oracle_atoms(red.desc, i, X, tab, literal_oracle) @ Q.T + T @ shift
        C = _oracle_atoms(canon, i, X, tab, literal_oracle)
        worst = max(worst, literal_oracle.max_atom_deviation(image, C, canon.cell[i].numpy()))
    assert worst <= 1e-10, worst
    # control: an improper normaliser element (the inversion), its handedness flip withheld (read at +1)
    k = int(torch.nonzero(tab.gamma_det < 0)[0, 0])
    moved = ops.apply(canon, tab.gamma_W[k], tab.gamma_w[k], wrap=True)
    wrong = moved.replace(h=torch.ones_like(moved.h))
    ctl = checks.same_crystal(canon, wrong, mol)
    assert float(ctl.min()) > 0.1, float(ctl.min())
    if k < len(tab.g_W):                                         # an element of G: with the flip, the same crystal
        assert float(checks.same_crystal(canon, moved, mol).max()) <= 1e-10
    for i in range(0, d.num_rows, 10):     # the oracle (rounded image only) bounds same_crystal (27 images) above
        A, B = _oracle_atoms(canon, i, X, tab, literal_oracle), _oracle_atoms(wrong, i, X, tab, literal_oracle)
        ref = literal_oracle.max_atom_deviation(A, B, canon.cell[i].numpy())
        assert 0.1 < float(ctl[i]) <= ref + 1e-9


@pytest.mark.parametrize('sg', [1, 2])
def test_latent_v1_float32_round_trip(sg):
    ch = charts.LatentV1Chart(sg, RADIUS)
    g = torch.Generator().manual_seed(40 + sg)
    lo, hi = ch.clamp_bounds()
    y = (lo + (hi - lo) * torch.rand(5000, 12, generator=g)).float()
    if sg == 1:
        y[:, 6:9] = 0.0                                          # free axes: the gauge is latent 0
    m = ch.forward(y)
    assert not bool(m.clamped.any())
    y2, ld2 = ch.inverse(m.desc)
    sph = so3.matrix_to_hemisphere(m.desc.R[:, 0])
    good = (sph[:, 0] >= 0.1) & (sph[:, 2] >= 0.1) & (sph[:, 2] <= 2 * math.pi - 0.1)
    d = (y2 - y).abs()
    d[:, 10] = torch.remainder(y2[:, 10] - y[:, 10] + 1, 2).sub(1).abs()   # phi is a circle of period 2
    assert float(d[good].max()) <= 1e-6, float(d[good].max())
    assert float(d[:, :9].max()) <= 1e-6
    R2 = ch.forward(y2).desc.R
    assert (R2 - m.desc.R).abs().max() <= 1e-6
    assert (ld2 + m.logabsdet)[good].abs().max() <= 1e-4


# ---------------------------------------------------------------------------
# the float32 builder path
# ---------------------------------------------------------------------------

def _builder_deviation(built_from, truth, mol, tab, oracle):
    """[B] largest atom deviation (Angstrom) between the live builder's unit cell of ``built_from`` and the oracle's
    unit cell of ``truth`` (same cells)."""
    b = H.crystal_batch(built_from, mol)
    with torch.no_grad():
        b.mol2ucell(std_orientation=False)
    k, na = len(tab.g_W), len(mol.z)
    built = b.unit_cell_pos.double().reshape(built_from.num_rows, k, na, 3).numpy()
    X = mol.pos.numpy()
    return np.array([oracle.max_atom_deviation(built[i], _oracle_atoms(truth, i, X, tab, oracle), truth.cell[i].numpy())
                     for i in range(truth.num_rows)])


@pytest.mark.parametrize('sg', [1, 2])
def test_builder_atoms_match_the_literal_oracle(sg, chiral_molecule, literal_oracle):
    mol = H.mol_ref(chiral_molecule)
    g = torch.Generator().manual_seed(80 + sg)
    n = 24
    cells = H.physical_cells(torch.full((3,), 6.0, dtype=F64), torch.full((3,), 9.0, dtype=F64), n,
                             'latent_v1_lebesgue', g)
    d = CrystalDescription.from_parts(sg, cells, torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g))
    imgs, _, _ = domains.images(d)
    descs = CrystalDescription.cat([d, domains.fold(d).desc, imgs])
    wrapped = descs.t - torch.floor(descs.t)
    descs = descs[~(wrapped > 1 - 2 * H.CLIP).any(-1).any(-1)]                 # clear of the builder's clip band
    assert descs.num_rows >= n * len(group_table(sg).gamma_plus)
    tab = group_table(sg)
    dev = _builder_deviation(descs, descs, mol, tab, literal_oracle)
    assert dev.max() <= 1e-3, dev.max()
    # control: R transposed (the rotation convention read the other way), on rows where that moves R noticeably
    flipped = descs.replace(R=descs.R.transpose(-1, -2))
    moved = (descs.R - flipped.R).abs().amax((1, 2, 3)) > 0.2
    assert int(moved.sum()) > descs.num_rows // 2
    ctl = _builder_deviation(flipped[moved], descs[moved], mol, tab, literal_oracle)
    assert ctl.min() > 0.1, ctl.min()


# ---------------------------------------------------------------------------
# coordinate kinds
# ---------------------------------------------------------------------------

def _same(a, b, tol):
    """[B] True where two descriptions agree within ``tol`` in cell (relative), t and R."""
    return (((a.cell - b.cell).abs() / a.cell.abs()).amax(1) <= tol) & ((a.t - b.t).abs().amax((1, 2)) <= tol) \
        & ((a.R - b.R).abs().amax((1, 2, 3)) <= tol)


@pytest.mark.parametrize('sg', [2, 14, 19, 92])
def test_centroid_chart_kinds(sg, cell_sampler):
    """A centroid coordinate is a circle of period 1 in u iff u_i = 0 and u_i = 1 (the rest equal) fold to one point."""
    ch = charts.PolytopeARChart(sg)
    g = torch.Generator().manual_seed(90 + sg)
    n = 200
    cell = torch.tensor(cell_sampler(LATTICE_TYPE[sg], n, seed=90 + sg), dtype=F64)
    R = so3.random_rotations(n, g)
    for i, dim in enumerate(ch.dims):
        u = 0.05 + 0.9 * torch.rand(n, ch.ndim, generator=g, dtype=F64)
        ends = []
        for e in (0.0, 1.0):
            v = u.clone()
            v[:, i] = e
            ends.append(domains.fold(CrystalDescription.from_parts(sg, cell, ch.map(v)[0], R)).desc)
        same = _same(*ends, 1e-9)
        if dim.kind == 'circle':
            assert dim.period == 1.0 and bool(same.all()), (sg, dim)
        else:
            assert dim.kind == 'interval' and not bool(same.any()), (sg, dim)
    kinds = [d.kind for d in ch.dims]
    assert kinds == {2: ['circle'] * 3, 14: ['circle', 'circle', 'interval']}.get(sg, kinds)


@pytest.mark.parametrize('sg', [1, 2])
def test_latent_chart_kinds(sg):
    """LatentV1Chart: 'fixed' exactly on free-axis centroid rows; any other row is a circle (period = its clamp
    range, 2) iff its two clamp ends (the rest equal) describe one crystal (float32 parameters: within 1e-5)."""
    ch = charts.LatentV1Chart(sg, RADIUS)
    tab = group_table(sg)
    lo, hi = ch.clamp_bounds()
    g = torch.Generator().manual_seed(95 + sg)
    n = 200
    kinds = []
    for i, dim in enumerate(ch.dims):
        if 6 <= i < 9 and bool(tab.free[i - 6]):
            assert dim.kind == 'fixed', dim
            kinds.append(dim.kind)
            continue
        y = lo + (hi - lo) * (0.05 + 0.9 * torch.rand(n, 12, generator=g))
        y[:, 3:6] = 0.2 * (2 * torch.rand(n, 3, generator=g) - 1)                 # angles near 90 deg: real cells
        ends = []
        for e in (lo[i], hi[i]):
            v = y.clone()
            v[:, i] = e
            d, clamped = ch.describe(v)
            assert not bool(clamped.any())
            ends.append(domains.fold(d).desc)
        same = _same(*ends, 1e-5)
        if dim.kind == 'circle':
            assert dim.period == float(hi[i] - lo[i]) == 2.0 and bool(same.all()), dim
        else:
            assert dim.kind == 'interval' and not bool(same.any()), dim
        kinds.append(dim.kind)
    assert kinds.count('circle') == (1 if sg == 1 else 4)                       # phi; and the sg 2 3-torus


# ---------------------------------------------------------------------------
# package checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', [1, 2, 14, 92])
def test_orbit_check(sg):
    res = checks.orbit_check(sg, n=2000, seed=sg)
    assert res.n_not_one == 0 and res.min_count == res.max_count == 1


def test_orbit_check_detects_a_pushed_facet():
    sec = domains.translation_section(2)
    tab = group_table(2)
    cuts = [dict(c) for c in tab.asu.cuts]
    cuts[1] = dict(cuts[1], c='51/50')                            # x <= 1/2 -> x <= 0.51
    from mxtaltools.crystal_fd.exact import polytope_vertices, to_fraction
    verts = polytope_vertices([c['n'] for c in cuts], [to_fraction(c['c']) for c in cuts])
    region = Region.from_cuts(cuts, torch.tensor([[float(v) for v in p] for p in verts], dtype=F64))
    res = checks.orbit_check(2, n=2000, section=sec.with_region(region))
    assert res.max_count == 2 and res.n_not_one > 0


def test_round_trip_and_same_crystal(chiral_molecule):
    mol = H.mol_ref(chiral_molecule)
    chart = charts.CrystalChart.for_latent_v1(2, RADIUS, 'euler_zyz')
    y, _ = charts.from_unit(chart, torch.rand(200, chart.ndim, generator=torch.Generator().manual_seed(5),
                                              dtype=F64))
    d = chart.forward(y).desc
    rt = checks.round_trip(d, chart, mol)
    assert max(rt.cell, rt.t, rt.R, rt.atoms) <= 1e-11
    rt = checks.round_trip(d, lambda x: domains.fold(x).desc, mol)
    assert rt.atoms <= 1e-12
    with pytest.raises(ValueError):
        checks.same_crystal(d, d.replace(cell=d.cell * 1.01), mol)

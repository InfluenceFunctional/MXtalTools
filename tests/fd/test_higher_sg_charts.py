"""crystal_fd: charts and the reference measure for space groups 16-230, and the sg 6-9 double cover (2026-10-21).

Owner decisions (2026-10-21): one chart coordinate per free cell parameter, the reference cell measure flat in each
free log length ('latent_v1_lebesgue' in every family), the trainer's averaged length rows not mirrored; sg 6-9 only
through ``orientation_cover='double'``.

1. Families: ``cell_family`` is ``LATTICE_TYPE`` for all 230 groups; every group builds a ``CrystalChart`` with an
   explicit length box (sg 6-9 refused with the K message unless ``'double'``), cell blocks of 6 / 4 / 3 / 2 / 2 / 1
   interval coordinates; ``for_latent_v1`` uses the trainer's length box bit for bit and nothing else of its latent.
2. Round trips (fast set: 19, 61, 18, 33, 92, 88, 148, 169, 198, 9; every group 6-9 and 16-230 ``slow``):
   inverse(forward(y)) = y and forward(inverse(desc)) = desc within 1e-10, the log-Jacobians cancel; mapped cells
   are exact (angles ``HALF_PI``, gamma ``HEX_GAMMA``, b = a and c = a bit for bit) with cell log|det| 0.
3. The image is the canonical domain: every drawn crystal is its own canonical form (``canon.canonicalize``: cell
   step the identity, cell, t and R within 1e-10; on the double cover, it or its K image), and every canonical form
   of a random crystal charts inside the bounds exactly when its cell lies in the length box.
4. ``unmap``: face rounding within ``SLACK`` is put on the face, larger excursions are not; cells off the family's
   metric by more than ``FAMILY_TOL`` are refused, float32-sized deviations are not.
5. log|det| of the whole chart against autograd and central differences, through the natural coordinates (free log
   lengths, the monoclinic beta, the non-free centroid) plus the Haar density of the orientation chart.
6. ``log_reference_density`` is the chart's log-Jacobian + log V (V in closed form per family), finite;
   'metric_invariant' is refused; a Monte Carlo integral of exp(density) over the chart matches the measure's closed
   form (ASU volume x the integral of V over the cell domain) for one group of every class and family, and a chart
   with a face removed does not.
7. Orthorhombic faces: mirror faces in the a <-> b and six-order classes, Pbca's two seams glued by the 3-cycle
   (``CellChart.gluings``), and the gluing is physical (a seam cell re-described by its order canonicalises to the
   same row as the cell it lands on).
8. sg 6-9 double cover: multiplicity 2 (chart, ``images_per_crystal``, ``log_multiplicity``), the K image of every
   draw lies in the cover, is the same crystal, has the same reference density in Euler coordinates, K twice is the
   identity, exactly one of the two is in the fundamental domain and half the cover's draws are; in every orientation
   chart the reference densities at a draw and its K image differ by the Haar ratio alone (K-invariance is relative to
   the reference measure; the ratio is 0 in Euler coordinates only).
9. ``LatentV1Chart`` refuses its log-Jacobian for sg >= 16 with ``LATENT_ABOVE_MONOCLINIC`` (sg 3-15 keep their
   message); ``images_per_crystal('latent_v1_box')`` refuses orthorhombic groups with more than one axis order.
10. Local data (``slow``; ``MXT_FD_CSD_DATASET`` or ``D:/crystal_datasets/csd_fd_v4``, skipped when absent, read only):
    every canonical and special-metric row of sg 6-9 and 16-230 charts inside its group's bounds (cell coordinates
    exactly, centroid within the fold tolerance) and maps back to itself within 1e-10.
11. Translation seams (all 230 groups): one circle of period 1 per facet pair of ASU(Gamma+) glued by a pure
    translation (read off ``GroupTable.facets``) and no other; points on either facet and their translates chart to
    coordinates equal modulo the period with equal log-Jacobians; only sg 88, 91, 92, 95, 96, 98, 122, 141, 142 take
    a sheared basis. In those, in the whole chart: the seam's two ends differ by the translation alone, fold to one
    point and have one reference density, interval ends do not; ASU(Gamma+) is a box in s = M t, |det M| = 2. A
    ``stages`` override is read in the AR coordinates: the table's t-bounds are refused in a sheared group unless
    ``stages_in='s'``, and reproduce the default chart elsewhere.
"""
import math
import os

import pytest
import torch

from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import axis_order, canon, charts, domains, measure, ops, so3
from mxtaltools.crystal_fd.charts import HALF_PI, HEX_GAMMA, HIGHER_FAMILIES
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
FAST = (19, 61, 18, 33, 92, 88, 148, 169, 198, 9)
SLOW = tuple(sg for sg in (6, 7, 8) + tuple(range(16, 231)) if sg not in FAST)
#: an asymmetric box (Angstrom) that leaves every family a cell: tetragonal a in [4, 14], cubic a in [4, 12]
BOX = (torch.tensor([3.0, 4.0, 2.5], dtype=F64), torch.tensor([14.0, 18.0, 12.0], dtype=F64))
NDIM_CELL = {'triclinic': 6, 'monoclinic': 4, 'orthorhombic': 3, 'tetragonal': 2, 'hexagonal': 2, 'cubic': 1}
K_GROUPS = (6, 7, 8, 9)


def _cover(sg):
    return 'double' if sg in K_GROUPS else None


def _chart(sg, orientation='axis_angle', box=BOX):
    return charts.CrystalChart.for_length_box(sg, box, orientation, orientation_cover=_cover(sg))


def _draw(ch, n, seed, margin=0.01):
    g = torch.Generator().manual_seed(seed)
    y, logv = charts.from_unit(ch, margin + (1 - 2 * margin) * torch.rand(n, ch.ndim, generator=g, dtype=F64))
    return y, logv


def _deviation(a, b):
    """[B] largest difference of unit 0: cell lengths relative, angles, t, R entries (inf where h differs)."""
    a, b = a.double(), b.double()
    dev = torch.stack([((a.cell[:, :3] - b.cell[:, :3]).abs() / b.cell[:, :3]).amax(1),
                       (a.cell[:, 3:] - b.cell[:, 3:]).abs().amax(1), (a.t[:, 0] - b.t[:, 0]).abs().amax(1),
                       (a.R[:, 0] - b.R[:, 0]).abs().amax((1, 2))], 1).amax(1)
    return torch.where(a.h[:, 0] == b.h[:, 0], dev, torch.full_like(dev, float('inf')))


def _in_bounds(ch, y, tol=0.0, cols=None):
    """[B] every column (or those in ``cols``) of y within its AR bounds, within ``tol``."""
    ok = torch.ones(y.shape[0], dtype=torch.bool)
    for i in range(ch.ndim) if cols is None else cols:
        lo, hi = ch.bounds(y[:, :i], i)
        ok &= (y[:, i] >= lo - tol) & (y[:, i] <= hi + tol)
    return ok


def _natural_cell(cell, family):
    """[B, k] the natural coordinates of a family's cell measure: free log lengths and free angles."""
    L = torch.log(cell[:, :3])
    if family == 'triclinic':
        return torch.cat([L, cell[:, 3:]], 1)
    if family == 'monoclinic':
        return torch.cat([L, cell[:, 4:5]], 1)
    if family == 'orthorhombic':
        return L
    if family in ('tetragonal', 'hexagonal'):
        return L[:, [0, 2]]
    return L[:, :1]


def _volume_closed(cell, family):
    """[B] cell volume from the family's free parameters alone (Angstrom^3)."""
    a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
    if family == 'orthorhombic':
        return a * b * c
    if family == 'tetragonal':
        return a * a * c
    if family == 'hexagonal':
        return a * a * c * math.sqrt(3) / 2
    if family == 'cubic':
        return a ** 3
    if family == 'monoclinic':
        return a * b * c * torch.sin(cell[:, 4])
    raise ValueError(family)


# ---------------------------------------------------------------------------
# 1: families
# ---------------------------------------------------------------------------

def test_families_dims_and_the_trainer_box():
    for sg in range(1, 231):
        fam = charts.cell_family(sg)
        assert fam == LATTICE_TYPE[sg], sg
        if sg in K_GROUPS:
            with pytest.raises(NotImplementedError, match='cuts K out of SO'):
                charts.CrystalChart.for_length_box(sg, BOX, 'euler_zyz')
        ch = _chart(sg, 'euler_zyz')
        assert ch.cell.family == fam and ch.cell.ndim == NDIM_CELL[fam]
        assert all(d.kind == 'interval' for d in ch.cell.dims)
        assert ch.ndim == ch.cell.ndim + ch.centroid.ndim + 3
        assert (ch.multiplicity, ch.domain) == ((2, 'double_cover') if sg in K_GROUPS else (1, 'fundamental'))
        if fam in HIGHER_FAMILIES:
            names = {'orthorhombic': ('log_a', 'log_b', 'log_c'), 'cubic': ('log_a',)}.get(fam, ('log_a', 'log_c'))
            assert tuple(d.name for d in ch.cell.dims) == names
    with pytest.raises(ValueError):
        charts.cell_family(231)
    # for_latent_v1 above monoclinic: the trainer's length box as a box, bit for bit, and nothing else
    for sg in (19, 61, 92, 148, 198):
        radius = torch.tensor([1.3, 1.9], dtype=torch.float32)
        ch = charts.CrystalChart.for_latent_v1(sg, radius, 'euler_zyz')
        lo, hi = charts.latent_v1_length_box(sg, radius)
        assert torch.equal(ch.cell.lo, lo) and torch.equal(ch.cell.hi, hi)
        ref = charts.CrystalChart.for_length_box(sg, (lo, hi), 'euler_zyz')
        y, _ = charts.from_unit(ref, torch.rand(2, ref.ndim, dtype=F64))
        assert torch.equal(ch.forward(y).desc.cell, ref.forward(y).desc.cell)


# ---------------------------------------------------------------------------
# 2: round trips and exact cells
# ---------------------------------------------------------------------------

def _round_trips(sg, n=400):
    for orientation in ('axis_angle', 'euler_zyz'):
        ch = _chart(sg, orientation)
        y, _ = _draw(ch, n, 10 + sg)
        m = ch.forward(y)
        yy, ldi = ch.inverse(m.desc)
        assert float((yy - y).abs().max()) <= 1e-10
        assert float((m.logabsdet + ldi).abs().max()) <= 1e-10
        back = ch.forward(yy).desc
        assert float(_deviation(back, m.desc).max()) <= 1e-10
    cell, ld = ch.cell.map(y[:, :ch.cell.ndim])
    fam = ch.cell.family
    if fam in HIGHER_FAMILIES:
        assert torch.equal(ld, torch.zeros_like(ld))
        assert bool((cell[:, 3:5] == HALF_PI).all())
        assert bool((cell[:, 5] == (HEX_GAMMA if fam == 'hexagonal' else HALF_PI)).all())
        if fam != 'orthorhombic':
            assert torch.equal(cell[:, 1], cell[:, 0])
        if fam == 'cubic':
            assert torch.equal(cell[:, 2], cell[:, 0])
        assert float((ch.cell.family_deviation(cell)).max()) == 0.0


@pytest.mark.parametrize('sg', FAST)
def test_round_trips(sg):
    _round_trips(sg)


@pytest.mark.slow
@pytest.mark.parametrize('sg', SLOW)
def test_round_trips_all_groups(sg):
    _round_trips(sg, n=200)


# ---------------------------------------------------------------------------
# 3: the image is the canonical domain
# ---------------------------------------------------------------------------

def _own_canonical_form(sg, n=400):
    ch = _chart(sg, 'euler_zyz')
    y, _ = _draw(ch, n, 20 + sg)
    d = ch.forward(y).desc
    c = canon.canonicalize(d)
    names = c.names()
    assert set(names) <= {'canonical', 'special_metric'}, set(names)
    assert names.count('canonical') >= 0.95 * n
    assert bool((c.N == torch.eye(3, dtype=torch.long)).all())
    dev = _deviation(c.desc, d)
    if sg in K_GROUPS:
        dev_k = _deviation(c.desc, ch.k_image(d))
        assert bool((torch.minimum(dev, dev_k) <= 1e-10).all())
        assert bool(((dev <= 1e-10) ^ (dev_k <= 1e-10)).all())
    else:
        assert float(dev.max()) <= 1e-10, float(dev.max())
        st = canon.fd_status(d)
        assert bool(st.is_fd[torch.tensor([s == 'canonical' for s in names])].all())


def _canonical_forms_chart_inside(sg, n=400):
    """Random crystals (lengths log-uniform over a box wider than BOX, any axis order), canonicalised: their chart
    coordinates lie in the bounds exactly when the canonical cell lies in BOX."""
    ch = _chart(sg, 'axis_angle')
    g = torch.Generator().manual_seed(30 + sg)
    fam = ch.cell.family
    L = torch.exp(math.log(2.0) + (math.log(22.0) - math.log(2.0)) * torch.rand(n, 3, generator=g, dtype=F64))
    if fam in ('tetragonal', 'hexagonal'):
        L[:, 1] = L[:, 0]
    if fam == 'cubic':
        L[:, 1] = L[:, 2] = L[:, 0]
    ang = torch.full((n, 3), HALF_PI, dtype=F64)
    if fam == 'hexagonal':
        ang[:, 2] = HEX_GAMMA
    if fam == 'monoclinic':
        ang[:, 1] = math.radians(95) + math.radians(30) * torch.rand(n, generator=g, dtype=F64)
    d = CrystalDescription.from_parts(sg, torch.cat([L, ang], 1), torch.rand(n, 3, generator=g, dtype=F64),
                                      so3.random_rotations(n, g))
    c = canon.canonicalize(d)
    assert set(c.names()) <= {'canonical', 'special_metric'}
    y, _ = ch.inverse(c.desc)
    k = ch.cell.ndim
    cell_in = _in_bounds(ch, y, 0.0, range(k))
    lo, hi = BOX
    in_box = ((c.desc.cell[:, :3] >= lo) & (c.desc.cell[:, :3] <= hi)).all(1)
    if fam == 'monoclinic':     # the shape coordinate s of every canonical (reduced) cell is in [0, 1], exactly
        assert bool(_in_bounds(ch, y, 0.0, [3]).all())
    assert torch.equal(cell_in, in_box), (int((cell_in & ~in_box).sum()), int((in_box & ~cell_in).sum()))
    assert int(in_box.sum()) >= n // 10
    assert bool(_in_bounds(ch, y, 1e-7, range(k, ch.ndim)).all())        # centroid (fold tolerance) and orientation


@pytest.mark.parametrize('sg', FAST)
def test_drawn_crystals_are_their_own_canonical_form(sg):
    _own_canonical_form(sg)


@pytest.mark.parametrize('sg', FAST)
def test_canonical_forms_chart_inside_the_bounds(sg):
    _canonical_forms_chart_inside(sg)


@pytest.mark.slow
@pytest.mark.parametrize('sg', SLOW)
def test_image_is_the_canonical_domain_all_groups(sg):
    _own_canonical_form(sg, n=200)
    _canonical_forms_chart_inside(sg, n=200)


# ---------------------------------------------------------------------------
# 4: unmap on faces and off the family metric
# ---------------------------------------------------------------------------

def _ortho_cells(rows):
    return torch.tensor([list(r) + [HALF_PI] * 3 for r in rows], dtype=F64)


def test_unmap_puts_face_rounding_on_the_face():
    wide = (torch.full((3,), 1.0, dtype=F64), torch.full((3,), 100.0, dtype=F64))
    eps, far = 1e-12, 1e-6
    a = 7.0
    # (sg, cells with a log length eps or far below its face bound, the column moved, the column it is put on)
    cases = ((18, [(a, a * (1 - eps), 9.0), (a, a * (1 - far), 9.0)], 1, 0),          # a <-> b: b onto a
             (19, [(5.0, a, a * (1 - eps)), (5.0, a, a * (1 - far))], 2, 1),          # six orders: c onto b
             (19, [(a, a * (1 - eps), 9.0), (a, a * (1 - far), 9.0)], 1, 0),          # six orders: b onto a
             (61, [(a, 9.0, a * (1 - eps)), (a, 9.0, a * (1 - far))], 2, 0),          # Pbca: c onto a
             (61, [(a, a * (1 - eps), 9.0), (a, a * (1 - far), 9.0)], 1, 0))          # Pbca: b onto a
    for sg, rows, col, onto in cases:
        ch = charts.CellChart('orthorhombic', wide, sg=sg)
        y, ld = ch.unmap(_ortho_cells(rows))
        assert torch.equal(ld, torch.zeros(2, dtype=F64))
        assert float(y[0, col]) == float(y[0, onto]) and bool(_in_bounds(ch, y[:1]).all()), (sg, col)
        assert float(y[1, col]) == math.log(rows[1][col]) and not bool(_in_bounds(ch, y[1:]).all()), (sg, col)
    ch = charts.CellChart('orthorhombic', wide, sg=33)                               # one order: nothing moved
    cells = _ortho_cells([(9.0, 7.0, 5.0)])
    assert torch.equal(ch.unmap(cells)[0], torch.log(cells[:, :3])) and bool(_in_bounds(ch, ch.unmap(cells)[0]).all())


def test_unmap_refuses_cells_off_the_family_metric():
    wide = (torch.full((3,), 1.0, dtype=F64), torch.full((3,), 100.0, dtype=F64))
    tet = charts.CellChart('tetragonal', wide)
    hexa = charts.CellChart('hexagonal', wide)
    cub = charts.CellChart('cubic', wide)
    ortho = charts.CellChart('orthorhombic', wide, sg=19)
    good = torch.tensor([[7.0, 7.0 * (1 + 1e-7), 9.0, HALF_PI, HALF_PI + 4.4e-8, HALF_PI]], dtype=F64)
    assert torch.equal(tet.unmap(good)[0], torch.log(good[:, [0, 2]]))      # float32-sized deviations pass; a is read
    hx = torch.tensor([[7.0, 7.0, 9.0, HALF_PI, HALF_PI, 2 * math.pi / 3]], dtype=F64)  # the other float64 of 120 deg
    assert torch.equal(hexa.unmap(hx)[0], torch.log(hx[:, [0, 2]]))
    for chart, cell in ((tet, [7.0, 7.0 * (1 + 1e-5), 9.0, HALF_PI, HALF_PI, HALF_PI]),
                        (tet, [7.0, 7.0, 9.0, HALF_PI, HALF_PI, HALF_PI + 1e-5]),
                        (tet, [7.0, 7.0, 9.0, HALF_PI, HALF_PI, HEX_GAMMA]),
                        (hexa, [7.0, 7.0, 9.0, HALF_PI, HALF_PI, HALF_PI]),
                        (cub, [7.0, 7.0, 7.0 * (1 + 1e-5), HALF_PI, HALF_PI, HALF_PI]),
                        (ortho, [5.0, 7.0, 9.0, HALF_PI + 1e-5, HALF_PI, HALF_PI])):
        with pytest.raises(ValueError, match='FAMILY_TOL'):
            chart.unmap(torch.tensor([cell], dtype=F64))


# ---------------------------------------------------------------------------
# 5: log|det| against autograd and central differences
# ---------------------------------------------------------------------------

def _natural(ch, z):
    """[B, k] natural coordinates of the cell and centroid blocks (free log lengths, free angles, non-free t) at the
    chart coordinates z [B, k] of those two blocks."""
    k0 = ch.cell.ndim
    cell, _ = ch.cell.map(z[:, :k0])
    t, _ = ch.centroid.map(z[:, k0:])
    return torch.cat([_natural_cell(cell, ch.cell.family), t[:, list(ch.centroid.order)]], 1)


def _log_det_checks(sg, n=12):
    ch = _chart(sg, 'euler_zyz')
    y, _ = _draw(ch, n, 40 + sg, margin=0.05)
    k = ch.cell.ndim + ch.centroid.ndim
    ld = ch.forward(y).logabsdet
    haar = so3.haar_log_density('euler_zyz', y[:, k:])
    for i in range(n):
        def f(z):
            return _natural(ch, z)[0]
        J = torch.autograd.functional.jacobian(f, y[i:i + 1, :k].clone())[:, 0, :]
        ref = torch.linalg.slogdet(J)[1] + haar[i]
        assert abs(float(ld[i] - ref)) <= 1e-10, (sg, i, float(ld[i]), float(ref))
        h = 1e-6
        cols = []
        for j in range(k):
            e = torch.zeros(1, k, dtype=F64)
            e[0, j] = h
            cols.append((f(y[i:i + 1, :k] + e) - f(y[i:i + 1, :k] - e)) / (2 * h))
        Jc = torch.stack(cols, 1)
        assert abs(float(ld[i] - torch.linalg.slogdet(Jc)[1] - haar[i])) <= 1e-6


@pytest.mark.parametrize('sg', FAST)
def test_log_abs_det_against_autograd_and_central_differences(sg):
    _log_det_checks(sg)


@pytest.mark.slow
@pytest.mark.parametrize('sg', SLOW)
def test_log_abs_det_all_groups(sg):
    _log_det_checks(sg, n=4)


# ---------------------------------------------------------------------------
# 6: the reference density
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', FAST)
def test_reference_density_is_the_measure(sg):
    ch = _chart(sg, 'hemisphere')
    y, _ = _draw(ch, 300, 50 + sg)
    d = ch.forward(y).desc
    lp = measure.log_reference_density(d, ch, 'latent_v1_lebesgue')
    assert bool(torch.isfinite(lp).all())
    k0, k1 = ch.cell.ndim, ch.cell.ndim + ch.centroid.ndim
    _, ld_cell = ch.cell.map(y[:, :k0])
    _, ld_cen = ch.centroid.map(y[:, k0:k1])
    ref = ld_cell + ld_cen + so3.haar_log_density('hemisphere', y[:, k1:]) \
        + torch.log(_volume_closed(d.cell, ch.cell.family))
    assert float((lp - ref).abs().max()) <= 1e-12 * float(lp.abs().max() + 1)
    with pytest.raises(NotImplementedError):
        measure.log_reference_density(d, ch, 'metric_invariant')


def _mc_mass(ch, n, seed):
    """(estimate, standard error) of the integral of exp(log_reference_density) over the chart (uniform u)."""
    g = torch.Generator().manual_seed(seed)
    y, logv = charts.from_unit(ch, torch.rand(n, ch.ndim, generator=g, dtype=F64))
    w = torch.exp(measure.log_reference_density(ch.forward(y).desc, ch, 'latent_v1_lebesgue') + logv)
    return float(w.mean()), float(w.std() / math.sqrt(n))


#: (sg, closed-form integral of V over the cell domain in the box [l, h]^3, as a function of (l, h))
MASS_CASES = ((33, lambda l, h: (h - l) ** 3), (18, lambda l, h: (h - l) ** 3 / 2), (61, lambda l, h: (h - l) ** 3 / 3),
              (19, lambda l, h: (h - l) ** 3 / 6), (92, lambda l, h: (h * h - l * l) / 2 * (h - l)),
              (148, lambda l, h: (h * h - l * l) / 2 * (h - l) * math.sqrt(3) / 2),
              (169, lambda l, h: (h * h - l * l) / 2 * (h - l) * math.sqrt(3) / 2),
              (198, lambda l, h: (h ** 3 - l ** 3) / 3))


@pytest.mark.parametrize('sg,cell_mass', MASS_CASES, ids=[f'sg{c[0]}' for c in MASS_CASES])
def test_the_measure_integrates_to_its_closed_form(sg, cell_mass):
    """Integral over the chart = vol(ASU(Gamma+)) (non-free coordinates) x 1 (normalised Haar) x the integral of V
    d(free log lengths) over the cell domain: the Lebesgue volume of the domain in the lengths for orthorhombic cells
    (the box over the number of allowed orders), a^2 c (sqrt(3) / 2 hexagonal) or a^3 integrated otherwise."""
    l, h = 4.0, 9.0
    box = (torch.full((3,), l, dtype=F64), torch.full((3,), h, dtype=F64))
    ch = charts.CrystalChart.for_length_box(sg, box, 'axis_angle')
    est, se = _mc_mass(ch, 200_000, 60 + sg)
    expected = float(group_table(sg).asu.volume) * cell_mass(l, h)
    assert abs(est - expected) <= 5 * se, (est, expected, se)
    if ch.cell.family == 'orthorhombic' and ch.cell.axis_order_kind != 'one':
        ch.cell.axis_order_kind = 'one'                         # control: the faces removed, every order admitted
        est2, se2 = _mc_mass(ch, 50_000, 70 + sg)
        assert abs(est2 - expected) >= 20 * se2, (est2, expected, se2)


# ---------------------------------------------------------------------------
# 7: orthorhombic faces and the Pbca seams
# ---------------------------------------------------------------------------

def test_orthorhombic_faces_and_gluings():
    wide = (torch.full((3,), 1.0, dtype=F64), torch.full((3,), 100.0, dtype=F64))
    expect = {'one': ((), ()), 'swap_ab': (('mirror',), ()), 'all': (('mirror', 'mirror'), ())}
    for sg in axis_order.ORTHORHOMBIC:
        ch = charts.CellChart('orthorhombic', wide, sg=sg)
        o = axis_order.axis_orders(sg)
        assert torch.equal(ch.faces, o.faces)
        if o.kind == 'cyclic':
            assert ch.face_kinds == ('seam', 'seam')
            assert [(g.face, g.lands_on, g.perm) for g in ch.gluings] == [(0, 1, (1, 2, 0)), (1, 0, (2, 0, 1))]
            assert all(tuple(o.perm[g.order].tolist()) == g.perm for g in ch.gluings)
        else:
            assert (ch.face_kinds, ch.gluings) == expect[o.kind]
    # the Pbca gluing is physical: a cell on face 1 (a = c < b) re-described by the gluing's order lands on face 0, and
    # the two descriptions canonicalise to one row (the face-0 description)
    ch = charts.CellChart('orthorhombic', wide, sg=61)
    g = torch.Generator().manual_seed(80)
    n = 50
    a = 5 + 3 * torch.rand(n, generator=g, dtype=F64)
    b = a * (1.2 + torch.rand(n, generator=g, dtype=F64))
    cells = torch.stack([a, b, a, *[torch.full((n,), HALF_PI, dtype=F64)] * 3], 1)
    d = CrystalDescription.from_parts(61, cells, torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g))
    glue = ch.gluings[1]
    assert (glue.face, glue.lands_on) == (1, 0)
    o = axis_order.axis_orders(61)
    moved = ops.change_basis(d, o.N[glue.order].expand(n, 3, 3), o.s[glue.order].expand(n, 3))
    y, _ = ch.unmap(moved.cell)
    assert float((y @ ch.faces[0]).abs().max()) <= 1e-12                   # on face 0
    assert float((y - torch.log(cells[:, list(glue.perm)][:, :3])).abs().max()) <= 1e-12
    c1, c2 = canon.canonicalize(d), canon.canonicalize(moved)
    assert float(_deviation(c1.desc, c2.desc).max()) <= 1e-9
    assert float(_deviation(c1.desc, domains.fold(moved).desc).max()) <= 1e-9


# ---------------------------------------------------------------------------
# 8: the sg 6-9 double cover
# ---------------------------------------------------------------------------

def _double_cover(sg, n=2000):
    for name in ('euler_zyz', 'hemisphere'):
        with pytest.raises(NotImplementedError, match='cuts K out of SO'):
            charts.CrystalChart.for_latent_v1(sg, 1.7, name)
    ch = charts.CrystalChart.for_latent_v1(sg, 1.7, 'euler_zyz', orientation_cover='double')
    assert (ch.multiplicity, ch.domain, ch.orientation_cover) == (2, 'double_cover', 'double')
    assert ch.log_multiplicity == math.log(2)
    assert measure.images_per_crystal(sg, 'double_cover') == 2
    assert measure.log_multiplicity(sg, 'fundamental', 'double_cover') == math.log(2)
    y, _ = _draw(ch, n, 90 + sg)
    d = ch.forward(y).desc
    k = ch.k_image(d)
    tab = group_table(sg)
    (kk,) = [int(i) for i in tab.kernel if int(i) != 0]
    img = ops.apply(d, tab.gamma_W[kk], tab.gamma_w[kk])
    assert torch.equal(img.t[..., ~tab.free], d.t[..., ~tab.free])       # K acts trivially on the non-free centroid
    assert torch.equal(k.R, img.R) and bool((k.h == 1).all())
    yk, ldk = ch.inverse(k)
    assert bool(_in_bounds(ch, yk, 1e-12).all())                          # the K image is in the cover
    assert float(_deviation(ch.forward(yk).desc, k).max()) <= 1e-10
    assert float(_deviation(ch.k_image(k), d).max()) <= 1e-12              # K twice is the identity
    lp, lpk = (measure.log_reference_density(x, ch, 'latent_v1_lebesgue') for x in (d, k))
    assert float((lp - lpk).abs().max()) <= 1e-12                          # K-invariant (Euler: Haar is uniform)
    c, ck = canon.canonicalize(d), canon.canonicalize(k)
    assert float(_deviation(c.desc, ck.desc).max()) <= 1e-9                # the same crystal
    ind, ink = domains.in_domain(d)[0], domains.in_domain(k)[0]
    assert bool((ind ^ ink).all())
    frac = float(ind.double().mean())
    assert abs(frac - 0.5) <= 5 * math.sqrt(0.25 / n), frac


@pytest.mark.parametrize('sg', K_GROUPS)
def test_double_cover(sg):
    _double_cover(sg, n=2000 if sg == 9 else 500)


@pytest.mark.parametrize('sg', (6, 9))
def test_double_cover_k_invariance_is_relative_to_the_reference_measure(sg):
    """The condition ``CrystalChart``'s docstring states: K-invariance of a density relative to the reference measure.
    The reference density itself is K-invariant in that sense, so in every orientation chart log ref(y) - log ref(y_K)
    is the Haar ratio haar(y) - haar(y_K) alone (cell, non-free t and V equal), and log ref - logabsdet is equal at the
    two points. That ratio is 0 in Euler coordinates only: in hemisphere and axis_angle a K-invariant distribution has
    unequal Lebesgue densities at y and y_K, so p_y(y) = p_y(y_K) would be the wrong condition there."""
    for name in ('euler_zyz', 'hemisphere', 'axis_angle'):
        ch = charts.CrystalChart.for_length_box(sg, BOX, name, orientation_cover='double')
        y, _ = _draw(ch, 500, 95 + sg)
        m = ch.forward(y)
        k = ch.k_image(m.desc)
        yk, ldk_inv = ch.inverse(k)
        k1 = ch.offsets[2]
        lp, lpk = (measure.log_reference_density(x, ch, 'latent_v1_lebesgue') for x in (m.desc, k))
        haar = so3.haar_log_density(name, y[:, k1:]) - so3.haar_log_density(name, yk[:, k1:])
        assert float((lp - lpk - haar).abs().max()) <= 1e-12, name
        assert float(((lp - m.logabsdet) - (lpk + ldk_inv)).abs().max()) <= 1e-12, name
        if name == 'euler_zyz':
            assert float(haar.abs().max()) == 0.0
        else:
            assert float(haar.abs().max()) > 0.5, name                  # the Lebesgue densities differ


def test_double_cover_guards():
    with pytest.raises(ValueError, match='K is trivial'):
        charts.CrystalChart.for_length_box(14, BOX, 'euler_zyz', orientation_cover='double')
    with pytest.raises(ValueError, match='orientation_cover'):
        charts.CrystalChart.for_length_box(9, BOX, 'euler_zyz', orientation_cover='quotient')
    with pytest.raises(ValueError, match='K is trivial'):
        measure.images_per_crystal(14, 'double_cover')
    with pytest.raises(ValueError):
        _chart(19).k_image(_chart(19).forward(_draw(_chart(19), 2, 0)[0]).desc)


# ---------------------------------------------------------------------------
# 9: the trainer latent and the measure's counts above monoclinic; guards
# ---------------------------------------------------------------------------

def test_latent_chart_refuses_above_monoclinic():
    y = (-0.9 + 1.8 * torch.rand(20, 12, generator=torch.Generator().manual_seed(1))).float()
    for sg in (19, 61, 92, 148, 198):
        ch = charts.LatentV1Chart(sg, 1.5)
        desc, _ = ch.describe(y)                                           # the live maps still read
        assert ch.latents(desc).shape == (20, 12)
        for call in (lambda: ch.forward(y), lambda: ch.inverse(desc), lambda: ch.log_abs_det(y)):
            with pytest.raises(NotImplementedError) as e:
                call()
            assert str(e.value) == charts.LATENT_ABOVE_MONOCLINIC
        with pytest.raises(NotImplementedError, match='not built for sg >= 16'):
            measure.log_reference_density(desc, ch, 'latent_v1_lebesgue')
    with pytest.raises(NotImplementedError, match=r'triclinic groups only \(stage 0\)'):
        charts.LatentV1Chart(14, 1.5).forward(y)


def test_measure_counts_and_guards():
    for sg in (19, 61, 92, 148, 198):
        assert measure.images_per_crystal(sg, 'fundamental') == 1
        assert measure.images_per_crystal(sg, 'unit_cell') == len(group_table(sg).gamma_plus)
    for sg in axis_order.ORTHORHOMBIC:
        tab = group_table(sg)
        if axis_order.axis_orders(sg).k > 1:
            with pytest.raises(ValueError):
                measure.images_per_crystal(sg, 'latent_v1_box')
        elif tab.box_coverage == 1 and tab.k_box is not None:
            assert measure.images_per_crystal(sg, 'latent_v1_box') == tab.k_box
    with pytest.raises(ValueError, match='axis orders'):
        measure.log_multiplicity(19, 'fundamental', 'latent_v1_box')
    assert measure.images_per_crystal(92, 'latent_v1_box') == group_table(92).k_box
    lo, hi = BOX
    with pytest.raises(ValueError, match='needs sg in 16-74'):
        charts.CellChart('orthorhombic', BOX)
    with pytest.raises(ValueError):
        charts.CellChart('tetragonal', BOX, sg=19)
    with pytest.raises(ValueError):
        charts.CellChart('rhombohedral', BOX)
    with pytest.raises(ValueError, match='leaves no'):
        charts.CellChart('cubic', (torch.tensor([5., 1, 1]), torch.tensor([9., 9, 4])))
    with pytest.raises(ValueError, match='leaves no'):
        charts.CellChart('tetragonal', (torch.tensor([5., 1, 1]), torch.tensor([9., 4, 9])))
    with pytest.raises(ValueError, match='leaves no'):
        charts.CellChart('orthorhombic', (torch.tensor([5., 1, 1]), torch.tensor([9., 9, 4])), sg=19)
    charts.CellChart('orthorhombic', (torch.tensor([5., 1, 1]), torch.tensor([9., 9, 4])), sg=33)   # one order: a box
    with pytest.raises(ValueError, match='axis-order class'):
        charts.CrystalChart(19, charts.CellChart('orthorhombic', BOX, sg=18), charts.PolytopeARChart(19),
                            charts.OrientationChart('euler_zyz'))
    charts.CrystalChart(16, charts.CellChart('orthorhombic', BOX, sg=19), charts.PolytopeARChart(16),
                        charts.OrientationChart('euler_zyz'))              # one chart per axis-order class
    with pytest.raises(ValueError, match='is not hexagonal'):
        charts.CrystalChart(92, charts.CellChart('hexagonal', BOX), charts.PolytopeARChart(92),
                            charts.OrientationChart('euler_zyz'))
    with pytest.raises(ValueError, match='is not triclinic'):
        charts.CrystalChart(19, charts.CellChart('triclinic', BOX), charts.PolytopeARChart(19),
                            charts.OrientationChart('euler_zyz'))


# ---------------------------------------------------------------------------
# 10: canonical CSD rows (local data, read only)
# ---------------------------------------------------------------------------

_V4 = os.environ.get('MXT_FD_CSD_DATASET', 'D:/crystal_datasets/csd_fd_v4')


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(_V4), reason='canonical CSD dataset not available')
def test_canonical_csd_rows_chart_inside_and_back():
    from mxtaltools.crystal_fd import dataset
    ds = dataset.load_dataset(_V4, status=('canonical', 'special_metric'), verify=False, code='ignore')
    d = ds.desc
    sgs = sorted(set(d.sg.tolist()) & (set(K_GROUPS) | set(range(16, 231))))
    assert len(sgs) >= 80                                                  # 89 groups in csd_fd_v4
    wide = (torch.full((3,), 0.5, dtype=F64), torch.full((3,), 1e3, dtype=F64))
    n_rows = 0
    for sg in sgs:
        rows = d[d.sg == sg].double()
        ch = charts.CrystalChart.for_length_box(sg, wide, 'axis_angle', orientation_cover=_cover(sg))
        y, _ = ch.inverse(rows)
        k, k1 = ch.cell.ndim, ch.cell.ndim + ch.centroid.ndim
        assert bool(_in_bounds(ch, y, 0.0, range(k)).all()), sg                  # cell: exactly (SLACK convention)
        assert bool(_in_bounds(ch, y, 1e-7, range(k, k1)).all()), sg             # centroid: the fold tolerance in u
        assert bool(_in_bounds(ch, y, 1e-6, range(k1, ch.ndim)).all()), sg      # the pi ball, through a square root
        back = ch.forward(y).desc
        assert float(_deviation(back, rows).max()) <= 1e-10, sg
        n_rows += rows.num_rows
    assert n_rows == int(torch.isin(d.sg, torch.tensor(sgs)).sum()) and n_rows > 80_000


# ---------------------------------------------------------------------------
# 11: translation seams are circles
# ---------------------------------------------------------------------------

#: the groups whose translation seam runs along [1, +-1, 0]: their centroid chart has a sheared basis
SHEARED = (88, 91, 92, 95, 96, 98, 122, 141, 142)


def _translation_gluings(tab):
    """[(cut, lands_on, w exact [3])] every facet gluing of ASU(Gamma+) by a pure translation, read off the table:
    kind 'translation', W = I, lattice shift included (both facets of a pair, each with its own translation)."""
    from mxtaltools.crystal_fd.exact import to_fraction
    raw = tab.raw
    out = []
    for f in raw['facets']:
        for gl in f['gluings']:
            if gl['kind'] != 'translation':
                continue
            assert list(raw['gamma']['W'][gl['element']]) == [1, 0, 0, 0, 1, 0, 0, 0, 1]
            out.append((f['cut'], gl['lands_on'],
                        [to_fraction(v) + s for v, s in zip(raw['gamma']['w'][gl['element']], gl['shift'])]))
    return out


def _facet_points(tab, cut, n, seed):
    """[n, 3] generic points of the facet ``cut`` of ASU(Gamma+): random convex combinations of its exact vertices."""
    from fractions import Fraction
    from mxtaltools.crystal_fd.exact import evaluate, to_fraction
    c = tab.raw['asu']['cuts'][cut]
    verts = [[to_fraction(v) for v in p] for p in tab.raw['asu']['vertices']]
    on = [p for p in verts if evaluate(c['n'], Fraction(c['c']), p) == 0]
    assert len(on) >= 2, (tab.sg, cut)
    V = torch.tensor([[float(v) for v in p] for p in on], dtype=F64)
    w = torch.distributions.Dirichlet(torch.ones(len(on), dtype=F64)).sample((n,))
    return w @ V


def test_every_translation_gluing_is_a_circle():
    """In all 230 groups: the centroid chart has one circle (period 1) per facet pair glued by a pure translation and
    no other; a point on either facet and its translate (the gluing's own translation) chart to coordinates equal
    modulo the period (0 on ``seams[k].lower``, 1 on ``upper``), every other coordinate equal, with equal
    log-Jacobians (the density is continuous across the seam). Only the groups in ``SHEARED`` take a sheared basis."""
    torch.manual_seed(110)
    sheared = []
    for sg in range(1, 231):
        tab = group_table(sg)
        ch = charts.PolytopeARChart(sg)
        if ch.basis is not None:
            sheared.append(sg)
        glue = _translation_gluings(tab)
        circles = [i for i, d in enumerate(ch.dims) if d.kind == 'circle']
        assert all(ch.dims[i].period == 1.0 for i in circles), sg
        assert len(circles) == len(glue) // 2 == len(ch.seams), (sg, circles, len(glue))
        assert sorted(s.dim for s in ch.seams) == circles, sg
        for cut, lands_on, w in glue:
            (seam,) = [s for s in ch.seams if {s.lower, s.upper} == {cut, lands_on}]
            t = _facet_points(tab, cut, 64, sg)
            t2 = t + torch.tensor([float(v) for v in w], dtype=F64)
            u1, ld1 = ch.unmap(t)
            u2, ld2 = ch.unmap(t2)
            step = 1.0 if cut == seam.lower else -1.0
            expect = torch.zeros(ch.ndim, dtype=F64)
            expect[seam.dim] = step
            assert float((u2 - u1 - expect).abs().max()) <= 1e-12, (sg, cut)
            assert float((u1[:, seam.dim] - (0.0 if step > 0 else 1.0)).abs().max()) <= 1e-12, (sg, cut)
            assert float(((u1 < -1e-12) | (u1 > 1 + 1e-12)).sum()) == 0, (sg, cut)
            assert float((ld1 - ld2).abs().max()) <= 1e-12, (sg, cut)
            if step > 0:
                assert [float(v) for v in seam.w] == [float(v) for v in w]
    assert tuple(sheared) == SHEARED


def test_stages_override_is_read_in_the_ar_coordinates():
    """A ``stages`` override is bounds over the chart's AR coordinates. Unsheared (2, 19): the table's t-bounds are
    those, and ``stages=group_table(sg).ar`` reproduces the default chart bit for bit, under either ``stages_in``.
    Sheared (92, 88): the table's t-bounds are not bounds over s, so they are refused (default 't', and explicit 't'),
    and the chart's own ``stages`` with ``stages_in='s'`` reproduce the default chart bit for bit."""
    u = torch.rand(2000, 3, generator=torch.Generator().manual_seed(130), dtype=F64)

    def same(a, b):
        k = len(a.order)
        (ta, la), (tb, lb) = a.map(u[:, :k]), b.map(u[:, :k])
        return torch.equal(ta, tb) and torch.equal(la, lb) and a.dims == b.dims and a.seams == b.seams
    for sg in (2, 19):
        ref = charts.PolytopeARChart(sg)
        assert ref.basis is None
        for stages_in in ('t', 's'):
            assert same(ref, charts.PolytopeARChart(sg, stages=group_table(sg).ar, stages_in=stages_in)), sg
    for sg in (92, 88):
        ref = charts.PolytopeARChart(sg)
        assert ref.basis is not None
        for kw in ({}, {'stages_in': 't'}):
            with pytest.raises(ValueError, match="stages_in='s'"):
                charts.PolytopeARChart(sg, stages=group_table(sg).ar, **kw)
        assert same(ref, charts.PolytopeARChart(sg, stages=ref.stages, stages_in='s')), sg
    with pytest.raises(ValueError, match='stages_in must be'):
        charts.PolytopeARChart(19, stages_in='u')


@pytest.mark.parametrize('sg', SHEARED)
def test_sheared_seams_whole_chart(sg):
    """The sheared centroid chart of the [1, +-1, 0] seam groups, in the whole CrystalChart: the seam coordinate's two
    ends describe one crystal (the descriptions differ by the seam translation alone, fold to one point and have one
    reference density), each interval coordinate's two ends do not; ASU(Gamma+) is a box in s = M t with |det M| = 2
    (exp(log|det dt/du|) = vol(ASU) at every point)."""
    ch = _chart(sg, 'euler_zyz')
    cen = ch.centroid
    assert cen.basis is not None and cen.log_det_basis == math.log(2)
    assert [d.kind for d in cen.dims] == ['circle', 'interval', 'interval'] and cen.dims[2].name == 't_z'
    assert cen.order == (0, 1, 2) and cen.seams[0].dim == 0
    vol = float(group_table(sg).asu.volume)
    _, ld = cen.map(torch.rand(500, 3, generator=torch.Generator().manual_seed(sg), dtype=F64))
    assert float((ld - math.log(vol)).abs().max()) <= 1e-14
    for st in cen.stages:                                   # a box: constant bounds, one row each side
        assert float(st.lower_coef.abs().sum() + st.upper_coef.abs().sum()) == 0
        assert len(st.lower_const) == len(st.upper_const) == 1
    y, _ = _draw(ch, 200, 120 + sg, margin=0.05)
    k0 = ch.offsets[1]
    for i, dim in enumerate(cen.dims):
        ends = []
        for e in (0.0, 1.0):
            v = y.clone()
            v[:, k0 + i] = e
            ends.append(ch.forward(v).desc)
        folded = [domains.fold(d).desc for d in ends]
        same = _deviation(*folded) <= 1e-9
        if dim.kind == 'circle':
            w = torch.tensor([float(x) for x in cen.seams[0].w], dtype=F64)
            assert float((ends[1].t[:, 0] - ends[0].t[:, 0] - w).abs().max()) <= 1e-12
            assert torch.equal(ends[0].cell, ends[1].cell) and torch.equal(ends[0].R, ends[1].R)
            assert bool(same.all()), (sg, i)
            lp = [measure.log_reference_density(d, ch, 'latent_v1_lebesgue') for d in ends]
            assert float((lp[0] - lp[1]).abs().max()) <= 1e-12
        else:
            assert not bool(same.any()), (sg, i)

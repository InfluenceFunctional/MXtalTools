"""GATE 2 (fd_tooling.md section 4): the reference measure on the domain charts of sg 1 and 2.

(a) Volumes. The centroid boxes integrate exactly to vol(ASU(Gamma+)) (sg 1: 1, sg 2: 1/8); the generic
    Fourier-Motzkin chart matches the table volume by Monte Carlo in oblique groups (sg 148, 227), and sg 92's
    sheared chart is a box of that volume (the shear's 1 / |det M| included). Each orientation
    chart's Haar density integrates to 1 (quadrature, and Monte Carlo through the chart's AR bounds). The cell
    chart's integral of exp(logabsdet) equals the (log L, angle) volume of the tri_niggli zero set within the
    trainer's length box, counted independently with the live penalty.
(b) Histogram test. Independent physical crystals (cells from a reference sampler of mu_cell x V on the zero set,
    t uniform on [0, 1)^3, scipy Haar R) are re-described in a random proper basis, canonicalised (reduce_cell,
    then fold) and charted; they are compared with draws from the chart's own density
    exp(log_reference_density), per space group, cell measure and orientation chart. Two-sample KS on every
    coordinate, chi-square on the cell, centroid and orientation blocks and on consecutive cell coordinates;
    the family-wise level is 1e-3 (Bonferroni). 1e5 per side in the slow tier, 2e4 in the fast tier. The slow tier
    (the spec's 1e5) is the gate's acceptance: the fast tier is a smoke lane, too weak for cell-measure factors that
    vary by about 1.5x over the domain (a dropped prod sin in 'metric_invariant', or a dropped sin(gamma) in the
    cell Jacobian, fails a minority of fast cases and nearly all slow ones; gate 5 and gate 8 pin both formulas in
    the fast lane).
(c) Orientation: every chart with its measure term (hemisphere, Euler ZYZ, axis-angle in (b); the quaternion
    hemisphere against normalised Gaussian 4-vectors).
(d) Cell: both cell measures, each sampled where it is Lebesgue (``_helpers.physical_cells``).
Mutation controls must each fail: the sin(theta) factor dropped from the hemisphere density; the cell Jacobian
with R3's r1^2 r2^2 in place of r1 r2 (fast tier) or without its (1 - Lambda) factor (at 1e5; at 2e4 that factor,
between 1 and 2, gives p ~ 1e-3, short of the family-wise level); a centroid face pushed out (0.05 in the fast
tier, 0.01 at 1e5).
"""
import math
from fractions import Fraction

import numpy as np
import pytest
import torch
from scipy import stats
from scipy.spatial.transform import Rotation

from mxtaltools.crystal_fd import charts, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import ARStage, group_table
from tests.fd import _helpers as H

F64 = torch.float64
RADIUS = 1.7561            # the chiral fixture molecule's MolData radius; sets the trainer's length box
FAMILY_ALPHA = 1e-3
N_FAST, N_SLOW = 20_000, 100_000
ORIENTATIONS = ('hemisphere', 'euler_zyz', 'axis_angle')
MEASURES = ('latent_v1_lebesgue', 'metric_invariant')


# ---------------------------------------------------------------------------
# (a) volumes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', [1, 2, 14])
def test_box_volume_is_exact(sg):
    """A box chart's density is the constant 1 / vol: exp(logabsdet) equals vol(ASU(Gamma+)) as a fraction."""
    ch = charts.PolytopeARChart(sg)
    u = torch.rand(1000, ch.ndim, dtype=F64)
    t, ld = ch.map(u)
    vol = group_table(sg).asu.volume
    assert Fraction(float(torch.exp(ld).max())).limit_denominator(1000) == vol
    assert (torch.exp(ld) - float(vol)).abs().max() <= 1e-15
    assert vol == {1: Fraction(1), 2: Fraction(1, 8), 14: Fraction(1, 16)}[sg]


@pytest.mark.parametrize('sg', [148, 227])
def test_polytope_volume_monte_carlo(sg):
    """Oblique cuts (148, 227): E_u[exp(logabsdet)] = vol(ASU(Gamma+)) within 5 standard errors (1e6 points). sg 92,
    oblique in t, is a box in its sheared basis s = M t: exp(logabsdet) = 1/32 at every point, the 1/|det M| = 1/2 of
    the shear included."""
    ch = charts.PolytopeARChart(sg)
    g = torch.Generator().manual_seed(1)
    _, ld = ch.map(torch.rand(1_000_000, ch.ndim, generator=g, dtype=F64))
    J = torch.exp(ld)
    assert float(J.std()) > 0
    z = (J.mean() - float(group_table(sg).asu.volume)) / (J.std() / math.sqrt(len(J)))
    assert abs(float(z)) < 5
    _, ld = charts.PolytopeARChart(92).map(torch.rand(1000, 3, generator=g, dtype=F64))
    assert (torch.exp(ld) - 1 / 32).abs().max() <= 1e-15


def _gauss_legendre(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor((b - a) / 2 * x + (a + b) / 2), torch.tensor((b - a) / 2 * w)


@pytest.mark.parametrize('name', ['hemisphere', 'euler_zyz'])
def test_orientation_density_integrates_to_one_quadrature(name):
    ch = charts.OrientationChart(name)
    grids = [_gauss_legendre(48, *(float(b) for b in ch.bounds(torch.zeros(1, i, dtype=F64), i))) for i in range(3)]
    X = torch.stack(torch.meshgrid(*[g[0] for g in grids], indexing='ij'), -1).reshape(-1, 3)
    Wt = torch.einsum('i,j,k->ijk', *[g[1] for g in grids]).reshape(-1)
    _, ld = ch.map(X)
    assert abs(float((torch.exp(ld) * Wt).sum()) - 1) <= 1e-12


def test_axis_angle_density_integrates_to_one():
    """Quadrature in |v| (4 pi int_0^pi rho(w) w^2 dw), and Monte Carlo through the ball's AR bounds."""
    w, wt = _gauss_legendre(64, 0.0, math.pi)
    rho = torch.exp(so3.haar_log_density('axis_angle', torch.stack([w, 0 * w, 0 * w], 1)))
    assert abs(float(4 * math.pi * (rho * w ** 2 * wt).sum()) - 1) <= 1e-12
    ch = charts.OrientationChart('axis_angle')
    g = torch.Generator().manual_seed(2)
    y, logv = charts.from_unit(ch, torch.rand(1_000_000, 3, generator=g, dtype=F64))
    J = torch.exp(ch.map(y)[1] + logv)
    assert abs(float((J.mean() - 1) / (J.std() / 1000))) < 5


@pytest.mark.parametrize('sg', [1, 2])
def test_cell_chart_volume_matches_the_zero_set(sg):
    """int exp(logabsdet) dy over the cell chart = (log L, angle) volume of zero set x length box (1e6 points each)."""
    lo, hi = charts.latent_v1_length_box(sg, RADIUS)
    ch = charts.CellChart('triclinic', (lo, hi))
    g = torch.Generator().manual_seed(3 + sg)
    n = 1_000_000
    y, logv = charts.from_unit(ch, torch.rand(n, 6, generator=g, dtype=F64))
    J = torch.exp(ch.map(y)[1] + logv)
    chart_vol, chart_se = float(J.mean()), float(J.std()) / math.sqrt(n)
    # independent count: uniform (log a, log b, log c, alpha, beta, gamma) on a box holding the domain
    ra, rb, rc = H._reduced_ranges(lo[0].tolist(), hi[0].tolist())
    box = [(math.log(r[0]), math.log(r[1])) for r in (ra, rb, rc)] + [H.ALPHA_RANGE, H.OBTUSE_RANGE, H.OBTUSE_RANGE]
    u = torch.rand(n, 6, generator=g, dtype=F64)
    x = torch.stack([b[0] + (b[1] - b[0]) * u[:, k] for k, b in enumerate(box)], 1)
    cells = torch.cat([torch.exp(x[:, :3]), x[:, 3:]], 1)
    hit = (H.on_zero_set(cells) & H.in_box(cells, lo, hi)).to(F64)
    box_vol = math.prod(b[1] - b[0] for b in box)
    count_vol, count_se = float(hit.mean()) * box_vol, float(hit.std()) / math.sqrt(n) * box_vol
    z = (chart_vol - count_vol) / math.hypot(chart_se, count_se)
    assert abs(z) < 5, (chart_vol, count_vol, z)


# ---------------------------------------------------------------------------
# (b)-(d) histogram tests
# ---------------------------------------------------------------------------

_PHYSICAL = {}


def physical_canonical(sg, cell_measure, n):
    """Canonical descriptions of n independent physical crystals (cached per sg, measure and n)."""
    key = (sg, cell_measure, n)
    if key not in _PHYSICAL:
        lo, hi = charts.latent_v1_length_box(sg, RADIUS)
        g = torch.Generator().manual_seed(100 * sg + n % 97 + (7 if cell_measure == 'metric_invariant' else 0))
        rng = np.random.default_rng(int(g.initial_seed()))
        cells = H.physical_cells(lo, hi, n, cell_measure, g)
        R = torch.as_tensor(Rotation.random(n, random_state=rng.integers(1 << 31)).as_matrix(), dtype=F64)
        d0 = CrystalDescription.from_parts(sg, cells, torch.rand(n, 3, generator=g, dtype=F64), R)
        canon, red, _ = H.canonicalise(H.scramble(d0, rng))
        assert float(red.violation.max()) <= 1e-10
        assert float(((canon.cell - cells).abs() / cells.abs()).max()) <= 1e-9   # same reduced cell back
        _PHYSICAL[key] = canon
    return _PHYSICAL[key]


def _bins(pooled, k):
    q = torch.quantile(pooled, torch.linspace(0, 1, k + 1, dtype=F64)[1:-1].to(pooled.dtype), dim=0)
    return q.T                                                                   # [d, k - 1]


def _chi2(a, b, k):
    """Two-sample chi-square on a joint grid of per-coordinate pooled-quantile bins."""
    pooled = torch.cat([a, b])
    edges = _bins(pooled, k)
    def idx(x):
        i = torch.zeros(len(x), dtype=torch.long)
        for d in range(x.shape[1]):
            i = i * k + torch.searchsorted(edges[d].contiguous(), x[:, d].contiguous())
        return i
    ha = torch.bincount(idx(a), minlength=k ** a.shape[1])
    hb = torch.bincount(idx(b), minlength=k ** a.shape[1])
    keep = (ha + hb) > 0
    return float(stats.chi2_contingency(torch.stack([ha[keep], hb[keep]]).numpy())[1])


def compare(phys, draw, chart, n):
    """p-values {test name: p} of the two-sample tests between y samples [n, ndim]."""
    p = {}
    for i, d in enumerate(chart.dims):
        p[f'KS {d.name}'] = float(stats.ks_2samp(phys[:, i].numpy(), draw[:, i].numpy()).pvalue)
    k_cell = 3 if n >= N_SLOW else 2
    o_t, o_r = chart.offsets[1], chart.offsets[2]
    p['chi2 cell block'] = _chi2(phys[:, :6], draw[:, :6], k_cell)
    for i in range(5):
        p[f'chi2 cell pair {chart.dims[i].name},{chart.dims[i + 1].name}'] = _chi2(phys[:, i:i + 2], draw[:, i:i + 2], 6)
    if chart.centroid.ndim:
        p['chi2 centroid block'] = _chi2(phys[:, o_t:o_r], draw[:, o_t:o_r], 4)
    p['chi2 orientation block'] = _chi2(phys[:, o_r:], draw[:, o_r:], 4)
    return p


def _run(sg, cell_measure, orientation, n, chart=None, seed=0):
    chart = chart or charts.CrystalChart.for_latent_v1(sg, RADIUS, orientation)
    phys, _ = chart.inverse(physical_canonical(sg, cell_measure, n))
    g = torch.Generator().manual_seed(1000 + seed + sg)
    draw = H.chart_draws(chart, n, cell_measure, g)
    return compare(phys, draw, chart, n)


def _assert_pass(p):
    worst = min(p, key=p.get)
    assert p[worst] >= FAMILY_ALPHA / len(p), \
        f'{worst}: p = {p[worst]:.3g} < {FAMILY_ALPHA}/{len(p)}; all: ' + ', '.join(f'{k} {v:.3g}' for k, v in p.items())


CASES = [(sg, m, o) for sg in (1, 2) for m in MEASURES for o in ORIENTATIONS]


@pytest.mark.parametrize('sg, cell_measure, orientation', CASES)
def test_histogram_fast(sg, cell_measure, orientation):
    _assert_pass(_run(sg, cell_measure, orientation, N_FAST))


@pytest.mark.slow
@pytest.mark.parametrize('sg, cell_measure, orientation', CASES)
def test_histogram_1e5(sg, cell_measure, orientation):
    _assert_pass(_run(sg, cell_measure, orientation, N_SLOW))


def test_quaternion_hemisphere_is_haar():
    """scipy Haar rotations -> quaternions vs normalised Gaussian 4-vectors on the w >= 0 hemisphere (1e5 each)."""
    n = N_SLOW
    rng = np.random.default_rng(11)
    R = torch.as_tensor(Rotation.random(n, random_state=12).as_matrix(), dtype=F64)
    q, ld = charts.OrientationChart('quaternion').unmap(R)
    ref = torch.as_tensor(rng.normal(size=(n, 4)))
    ref = ref / ref.norm(dim=1, keepdim=True)
    ref = torch.where(ref[:, :1] < 0, -ref, ref)
    p = {f'KS {c}': float(stats.ks_2samp(q[:, i].numpy(), ref[:, i].numpy()).pvalue) for i, c in enumerate('wxyz')}
    p['chi2'] = _chi2(q, ref, 3)
    _assert_pass(p)
    assert (ld - math.log(math.pi ** 2)).abs().max() <= 1e-15       # constant density on the hemisphere


# ---------------------------------------------------------------------------
# mutation controls
# ---------------------------------------------------------------------------

class _NoSinTheta(charts.OrientationChart):
    def map(self, y):
        R, ld = super().map(y)
        return R, ld - torch.log(torch.sin(y[:, 0]))

    def unmap(self, R):
        y, ld = super().unmap(R)
        return y, ld + torch.log(torch.sin(y[:, 0]))


class _NoLambda(charts.CellChart):
    def map(self, y):
        cell, ld = super().map(y)
        return cell, ld - torch.log(1 - self.lam(y[:, 2], y[:, 3], y[:, 4]))

    def unmap(self, cell):
        y, ld = super().unmap(cell)
        return y, ld + torch.log(1 - self.lam(y[:, 2], y[:, 3], y[:, 4]))


class _R3Jacobian(charts.CellChart):
    """The Jacobian as R3 wrote it, (1 - Lambda) r1^2 r2^2 / (8 prod sin): right for log-ratio length rows, not r1, r2."""
    def map(self, y):
        cell, ld = super().map(y)
        return cell, ld + torch.log(y[:, 1] * y[:, 2])

    def unmap(self, cell):
        y, ld = super().unmap(cell)
        return y, ld - torch.log(y[:, 1] * y[:, 2])


def _pushed(sg, push):
    """The sg centroid chart with the first AR coordinate's upper face pushed out by ``push`` (fractional)."""
    st = charts.PolytopeARChart(sg).stages                  # the chart's AR coordinates (s = M t in a sheared group)
    s0 = st[0]
    return charts.PolytopeARChart(sg, stages=(ARStage(s0.coord, s0.lower_coef, s0.lower_const, s0.upper_coef,
                                                       s0.upper_const + push),) + tuple(st[1:]), stages_in='s')


def _mutants(sg, push):
    lo, hi = charts.latent_v1_length_box(sg, RADIUS)
    cell, cen, ori = (charts.CellChart('triclinic', (lo, hi)), charts.PolytopeARChart(sg),
                      charts.OrientationChart('hemisphere'))
    return {'no sin(theta)': charts.CrystalChart(sg, cell, cen, _NoSinTheta('hemisphere')),
            'no (1 - Lambda)': charts.CrystalChart(sg, _NoLambda('triclinic', (lo, hi)), cen, ori),
            'r1^2 r2^2 Jacobian': charts.CrystalChart(sg, _R3Jacobian('triclinic', (lo, hi)), cen, ori),
            f'face pushed {push}': charts.CrystalChart(sg, cell, _pushed(sg, push), ori)}


def _assert_fail(p, name):
    worst = min(p, key=p.get)
    assert p[worst] < FAMILY_ALPHA / len(p), f'mutation "{name}" was not detected: smallest p {worst} = {p[worst]:.3g}'


@pytest.mark.parametrize('mutation', ['no sin(theta)', 'r1^2 r2^2 Jacobian', 'face pushed 0.05'])
def test_mutations_fail_fast(mutation):
    chart = _mutants(2, 0.05)[mutation]
    _assert_fail(_run(2, 'latent_v1_lebesgue', 'hemisphere', N_FAST, chart=chart), mutation)


@pytest.mark.slow
@pytest.mark.parametrize('mutation', ['no sin(theta)', 'no (1 - Lambda)', 'face pushed 0.01'])
def test_mutations_fail_1e5(mutation):
    chart = _mutants(2, 0.01)[mutation]
    _assert_fail(_run(2, 'latent_v1_lebesgue', 'hemisphere', N_SLOW, chart=chart), mutation)

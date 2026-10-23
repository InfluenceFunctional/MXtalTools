"""GATE 5 (fd_tooling.md section 4): every chart's closed-form log-Jacobian against autograd and central differences.

Float64, tolerance 1e-8 (nats), at 500 interior points plus points within 1e-6 of every bound and near each
chart's singular sets. Central differences are fourth order (step 1e-5; 1e-6 and 1e-7 for the hemisphere and
Euler charts, whose derivatives grow near their poles); where a chart is only piecewise smooth (the max / min of
Fourier-Motzkin rows) they skip points within 1e-4 of a kink, and autograd covers them. Square Jacobians must be lower triangular
in the chart's AR order (the derivatives of conditional bounds stay below the diagonal; their upper entries are
exactly zero under autograd).

- CellChart (R3's triclinic map): y -> (log c, log b, log a, beta, gamma, alpha); also near the kink of Lambda
  (s_beta + s_gamma = -1) and near beta, gamma = 90 deg (s -> 0).
- PolytopeARChart: u -> the AR coordinates (triangular) and u -> the non-free centroid coordinates (the shear's
  log|det M| included), sg 2 (box), 92 (a box in its sheared basis s = M t) and the oblique polytopes of 148, 227.
- OrientationChart: the Haar density is sqrt(det g) / (8 pi^2), g the bi-invariant metric tr(dR^T dR) / 2 pulled
  back by the chart's Jacobian.
- Euler (alpha, cos beta, gamma) -> the trainer's orientation latent rows 9-11: log|det| = -log(sin^2(r/2) sin
  theta) - 3 log pi (fd_tooling.md section 2.7), equal to the difference of the two charts' Haar densities.
- LatentV1Chart: the live inverse latent map, evaluated in float64, on the live rows.
"""
import math

import pytest
import torch

from mxtaltools.common.geometry_utils import sph_rotvec2lat
from mxtaltools.crystal_fd import charts, so3

F64 = torch.float64
TOL = 1e-8
H = 1e-5


def jacobian_autograd(f, y):
    """[n, m, k] Jacobian of a row-wise map f: [n, k] -> [n, m] (rows independent)."""
    y = y.clone().requires_grad_(True)
    out = f(y)
    rows = [torch.autograd.grad(out[:, j].sum(), y, retain_graph=True)[0] for j in range(out.shape[1])]
    return torch.stack(rows, 1).detach()


def jacobian_fd(f, y, h=H):
    """Fourth-order central differences."""
    cols = []
    for i in range(y.shape[1]):
        e = torch.zeros_like(y)
        e[:, i] = h
        cols.append((-f(y + 2 * e) + 8 * f(y + e) - 8 * f(y - e) + f(y - 2 * e)) / (12 * h))
    return torch.stack(cols, 2)


def _check_square(f, y, closed, triangular=True, fd_ok=None):
    """Closed form vs autograd at every point, vs central differences where ``fd_ok`` (default all)."""
    Ja = jacobian_autograd(f, y)
    fd_ok = torch.ones(len(y), dtype=torch.bool) if fd_ok is None else fd_ok
    Jf = jacobian_fd(f, y[fd_ok])
    la = torch.linalg.slogdet(Ja)[1]
    # triangular (asserted below on the exact autograd Jacobian): the determinant is the diagonal's product, which
    # keeps central-difference noise in the zero upper entries out of it where a diagonal entry is tiny (pinches)
    lf = torch.log(torch.diagonal(Jf, dim1=1, dim2=2).abs()).sum(1) if triangular else torch.linalg.slogdet(Jf)[1]
    assert (closed - la).abs().max() <= TOL, float((closed - la).abs().max())
    assert (closed[fd_ok] - lf).abs().max() <= TOL, float((closed[fd_ok] - lf).abs().max())
    if triangular:
        assert bool((torch.triu(Ja, diagonal=1) == 0).all()), 'Jacobian is not lower triangular in AR order'
    return float((closed - la).abs().max()), float((closed[fd_ok] - lf).abs().max())


def _points(chart, n_interior, gen, delta=1e-6, extra=None):
    """Interior points, plus every coordinate in turn at delta and 1 - delta of its range (others interior)."""
    k = chart.ndim
    u = [0.02 + 0.96 * torch.rand(n_interior, k, generator=gen, dtype=F64)]
    for i in range(k):
        for edge in (delta, 1 - delta):
            v = 0.02 + 0.96 * torch.rand(20, k, generator=gen, dtype=F64)
            v[:, i] = edge
            u.append(v)
    if extra is not None:
        u.append(extra)
    return charts.from_unit(chart, torch.cat(u))[0]


# ---------------------------------------------------------------------------
# cell
# ---------------------------------------------------------------------------

def _cell_natural(ch):
    def f(y):
        cell, _ = ch.map(y)
        return torch.stack([torch.log(cell[:, 2]), torch.log(cell[:, 1]), torch.log(cell[:, 0]),
                            cell[:, 4], cell[:, 5], cell[:, 3]], 1)
    return f


@pytest.mark.parametrize('sg', [1, 2])
def test_cell_chart_jacobian(sg):
    lo, hi = charts.latent_v1_length_box(sg, 1.7561)
    ch = charts.CellChart('triclinic', (lo, hi))
    g = torch.Generator().manual_seed(sg)
    y = _points(ch, 500, g)
    # near the kink of Lambda (s_beta + s_gamma = -1, both sides) and near s_beta, s_gamma = 0 (beta, gamma = 90 deg)
    base = y[:40].clone()
    sb = -0.2 - 0.6 * torch.rand(40, generator=g, dtype=F64)
    kink = torch.cat([base.clone(), base.clone()])
    kink[:, 3] = torch.cat([sb, sb])
    kink[:, 4] = -1 - kink[:, 3] + torch.cat([torch.full((40,), 1e-6), torch.full((40,), -1e-6)]).to(F64)
    right = base.clone()
    right[:, 3], right[:, 4] = -1e-6, -1e-6
    y = torch.cat([y, kink, right])
    _check_square(_cell_natural(ch), y, ch.map(y)[1])
    # the inverse map's log-Jacobian is the negative at the image
    yy, ld = ch.unmap(ch.map(y)[0])
    assert (yy - y).abs().max() <= 1e-12 and (ld + ch.map(y)[1]).abs().max() <= 1e-12


def test_cell_chart_closed_form():
    """log|det| = log((1 - Lambda) r1 r2 / (8 sin alpha sin beta sin gamma)) written out."""
    ch = charts.CellChart('triclinic', (torch.tensor([1.0, 1, 1]), torch.tensor([20.0, 20, 20])))
    y = _points(ch, 200, torch.Generator().manual_seed(7))
    cell, ld = ch.map(y)
    lam = ch.lam(y[:, 2], y[:, 3], y[:, 4])
    ref = torch.log((1 - lam) * y[:, 1] * y[:, 2] / (8 * torch.sin(cell[:, 3:]).prod(1)))
    assert (ld - ref).abs().max() <= 1e-12


# ---------------------------------------------------------------------------
# centroid
# ---------------------------------------------------------------------------

def _kink_margin(ch, u):
    """[n] smallest gap (fractional) between the active and next bound rows of any AR stage, or 10x its width."""
    s = ch.ar_coordinates(ch.map(u)[0])          # t[:, order], or (M t)[:, order] in a sheared group (sg 92)
    margin = torch.full((len(u),), float('inf'), dtype=F64)
    for i, st in enumerate(ch.stages):
        prev = s[:, :i]
        for vals in (prev @ st.lower_coef.T + st.lower_const, -(prev @ st.upper_coef.T + st.upper_const)):
            if vals.shape[1] > 1:
                top = vals.topk(2, dim=1).values
                margin = torch.minimum(margin, top[:, 0] - top[:, 1])
        lo, hi = ch.t_bounds(prev, i)
        margin = torch.minimum(margin, 10 * (hi - lo))   # pinch: a width of 1e-5 leaves differences of 1e-10
    return margin


@pytest.mark.parametrize('sg', [2, 92, 148, 227])
def test_polytope_chart_jacobian(sg):
    """Bounds are piecewise linear (max / min of rows): central differences skip points within 1e-4 of a kink, and
    where an AR interval is narrower than 1e-5 (next to a vertex, roundoff / width exceeds 1e-8 there)."""
    ch = charts.PolytopeARChart(sg)
    y = _points(ch, 500, torch.Generator().manual_seed(sg))
    order = list(ch.order)
    s = lambda u: ch.ar_coordinates(ch.map(u)[0])       # the AR coordinates: t[:, order], or (M t)[:, order]
    fd_ok = _kink_margin(ch, y) > 1e-4
    assert fd_ok[:500].float().mean() >= 0.95 and fd_ok.float().mean() >= 0.8   # interior points; all points
    _check_square(s, y, ch.map(y)[1] + ch.log_det_basis, fd_ok=fd_ok)
    # the natural coordinates t: the shear's -log|det M| included (not triangular in a sheared group)
    _check_square(lambda u: ch.map(u)[0][:, order], y, ch.map(y)[1], triangular=False, fd_ok=fd_ok)
    Ja = jacobian_autograd(s, y)
    if sg == 92:                                        # a box in s = (x + y, x - y, z), |det M| = 2
        assert ch.basis is not None and ch.log_det_basis == math.log(2)
        assert bool((torch.tril(Ja, diagonal=-1) == 0).all())
    elif sg != 2:
        assert ch.basis is None
        assert bool((torch.tril(Ja, diagonal=-1) != 0).any()), 'expected conditional bounds below the diagonal'


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------

def _haar_from_metric(name, J):
    """log(sqrt(det g) / 8 pi^2), g = J^T J / 2 with J = d vec(R) / dy [n, 9, 3]."""
    g = J.transpose(1, 2) @ J / 2
    return 0.5 * torch.logdet(g) - math.log(8 * math.pi ** 2)


@pytest.mark.parametrize('name', ['hemisphere', 'euler_zyz', 'axis_angle'])
def test_orientation_density_is_the_metric_volume(name):
    ch = charts.OrientationChart(name)
    g = torch.Generator().manual_seed(11)
    y = _points(ch, 500, g, delta=1e-4)
    if name == 'axis_angle':
        y = torch.cat([y, torch.tensor([[1e-4, 2e-4, -1e-4], [0.0, 0.0, 1e-3]], dtype=F64)])
    f = lambda x: ch.map(x)[0].reshape(-1, 9)
    closed = ch.map(y)[1]
    la = _haar_from_metric(name, jacobian_autograd(f, y))
    lf = _haar_from_metric(name, jacobian_fd(f, y, h={'hemisphere': 1e-6, 'euler_zyz': 1e-7, 'axis_angle': 1e-5}[name]))
    assert (closed - la).abs().max() <= TOL, float((closed - la).abs().max())
    assert (closed - lf).abs().max() <= TOL, float((closed - lf).abs().max())


def test_euler_to_trainer_rotvec_rows():
    """Euler (alpha, u, gamma) -> latent rows 9-11 (theta, phi, r scaled to [-1, 1]); closed form of section 2.7."""
    g = torch.Generator().manual_seed(12)
    e = torch.stack([2 * math.pi * torch.rand(4000, generator=g, dtype=F64),
                     -0.98 + 1.96 * torch.rand(4000, generator=g, dtype=F64),
                     2 * math.pi * torch.rand(4000, generator=g, dtype=F64)], 1)
    sph = so3.matrix_to_hemisphere(so3.euler_zyz_to_matrix(e))
    keep = ((sph[:, 0] > 0.02) & (sph[:, 0] < math.pi / 2 - 0.02) & (sph[:, 2] > 0.02)
            & (sph[:, 2] < 2 * math.pi - 0.02) & (sph[:, 1].abs() < math.pi - 0.02))
    e, sph = e[keep][:500], sph[keep][:500]
    f = lambda x: sph_rotvec2lat(so3.matrix_to_hemisphere(so3.euler_zyz_to_matrix(x)), 1)
    theta, r = sph[:, 0], sph[:, 2]
    closed = -torch.log(torch.sin(r / 2) ** 2 * torch.sin(theta)) - 3 * math.log(math.pi)
    by_charts = (so3.haar_log_density('euler_zyz', e)
                 - (so3.haar_log_density('hemisphere', sph) + math.log(math.pi ** 3 / 4)))
    assert (closed - by_charts).abs().max() <= 1e-12
    _check_square(f, e, closed, triangular=False)


# ---------------------------------------------------------------------------
# trainer latent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', [1, 2])
def test_latent_v1_jacobian(sg):
    """LatentV1Chart.log_abs_det against the live inverse latent map (float64) on the live rows, plus Haar."""
    from mxtaltools.common.geometry_utils import lat2sph_rotvec
    ch = charts.LatentV1Chart(sg, 1.7561)
    live = ch.live_rows(sg)
    g = torch.Generator().manual_seed(20 + sg)
    y = -0.9 + 1.8 * torch.rand(500, 12, generator=g, dtype=F64)
    y[:, 6:9] = torch.where(live[6:9], y[:, 6:9], torch.zeros_like(y[:, 6:9]))
    view = charts._make_view(torch.full((len(y),), sg), torch.full((len(y),), 1.7561), torch.device('cpu'))

    def nat(x):
        full = y.clone()
        full = full.masked_scatter(live.expand_as(full), x) if x.shape[1] < 12 else x
        p = view.inv_latent_transform(full)
        out = torch.cat([torch.log(p[:, :3]), p[:, 3:9], lat2sph_rotvec(full[:, 9:12], 1)], 1)
        return out[:, live]

    x = y[:, live]
    Ja, Jf = jacobian_autograd(nat, x), jacobian_fd(nat, x)
    sph = lat2sph_rotvec(y[:, 9:12], 1)
    haar = so3.haar_log_density('hemisphere', sph)
    closed = ch.log_abs_det(y)
    assert (closed - (torch.linalg.slogdet(Ja)[1] + haar)).abs().max() <= TOL
    assert (closed - (torch.linalg.slogdet(Jf)[1] + haar)).abs().max() <= TOL

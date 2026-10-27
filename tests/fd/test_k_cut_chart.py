"""crystal_fd: the K cut of sg 6-9 as a chart (``charts.KCutHemisphereChart``, 2026-10-26).

Owner request (2026-10-26): a unique chart of the fundamental domain for sg 6-9 (Pm, Pc, Cm, Cc), not only the double
cover; orientation chart 'hemisphere'; seams handled by consumers. K = C2(b) acts on R alone (R -> Q_K R) and the
fold's rule 2 keeps the lexicographically larger of R and Q_K R, i.e. R00 >= 0; in the hemisphere chart with r signed
in (-pi, pi] that is |r| <= rho(theta, phi) = 2 atan2(1, sqrt(max(0, 1 - 2 u_x^2))), u_x = sin theta cos phi.

1. Q_K: the group table's W_K is diag(-1, 1, -1) in sg 6-9 and T W_K T^-1 is the 2-fold about Cartesian y on every
   monoclinic cell (``k_rotation``); a cell whose Q_K is off it is refused; rho against the arccos form.
2. Round trips: inverse(forward(y)) = y and forward(inverse(desc)) = desc within 1e-10, log-Jacobians cancel.
3. The image is the fundamental domain: every draw is its own canonical form and in the domain, its K image is
   outside the bounds and not in the domain; every canonical form of a random crystal charts inside the orientation
   bounds exactly.
4. Adversarial rows (R00 near 0, theta at and near pi/2, r near +-pi, u_x^2 near 1/2, r on the cut): the bound's
   inside/outside equals the fold's choice wherever the fold decides by R00's sign (|R00| above the fold's tie band
   tol/2; a bound without the (1 - u_x^2) denominator does not); the chart's ``fold_keeps`` is the fold's choice on
   every row; every fold output charts inside the bounds exactly and back to itself (tie-band rows onto the face,
   rounding at the cut snapped); raw rows the fold keeps or rejects by a later entry chart as the fold's choice, and
   those it rejects by its first key are returned outside.
5. Measure: the cut has Haar mass 1/2 (quadrature and Monte Carlo; the wrong bound's is not; the plain chart's is 1);
   log Z of the double cover is log Z of the K-cut
   chart + log 2 (Monte Carlo over the whole chart, and ``log_multiplicity``); the reference density of a canonical
   row is the same on both charts.
6. Gluings: the equator (phi, r) ~ (phi + pi, -r) and r = -pi ~ pi (where rho = pi) are one rotation with one bound
   and one density; the cut face R00 = 0 is mapped to itself by K.
   Float32-stored canonical rows are accepted and agree with fd_status; canonical rows at the cusp whose R is off
   SO(3) (float32 origin, or 1e-7 noise) chart inside and come back within O(eps); the r bound is float64.
7. Guards (Q_K 1e-5 off refused, float32 storage accepted); groups other than 6-9 keep the plain orientation chart
   (``k_cut`` False, the block maps bit for bit).
8. Local data (``slow``; ``MXT_FD_CSD_DATASET`` or ``D:/crystal_datasets/csd_fd_v4``, read only): every canonical and
   special-metric row of sg 6-9 charts inside the K-cut bounds and maps back within 1e-10.
"""
import math
import os

import pytest
import torch

from mxtaltools.crystal_fd import canon, charts, domains, measure, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
K_GROUPS = (6, 7, 8, 9)
BOX = (torch.tensor([3.0, 4.0, 2.5], dtype=F64), torch.tensor([14.0, 18.0, 12.0], dtype=F64))
Q_B = torch.tensor(charts.K_B_ROTATION, dtype=F64)
FOLD_TOL = 1e-9


def _chart(sg, box=BOX):
    return charts.CrystalChart.for_length_box(sg, box, 'hemisphere')


def _draw(ch, n, seed, margin=0.01):
    g = torch.Generator().manual_seed(seed)
    return charts.from_unit(ch, margin + (1 - 2 * margin) * torch.rand(n, ch.ndim, generator=g, dtype=F64))


def _raw_signed(R):
    """(theta, phi, r signed, rho) of rotations, without the chart's band and snap rules."""
    th, ph, r = so3.matrix_to_hemisphere(R).unbind(1)
    r = torch.where(r > math.pi, r - 2 * math.pi, r)
    return th, ph, r, charts.KCutHemisphereChart.rho(th, ph)


def _in_bounds(ch, y, cols, tol=0.0):
    ok = torch.ones(y.shape[0], dtype=torch.bool)
    for i in cols:
        lo, hi = ch.bounds(y[:, :i], i)
        ok &= (y[:, i] >= lo - tol) & (y[:, i] <= hi + tol)
    return ok


def _rdev(a, b):
    return (a.double().R[:, 0] - b.double().R[:, 0]).abs().amax((1, 2))


def _deviation(a, b):
    a, b = a.double(), b.double()
    return torch.stack([((a.cell[:, :3] - b.cell[:, :3]).abs() / b.cell[:, :3]).amax(1),
                        (a.cell[:, 3:] - b.cell[:, 3:]).abs().amax(1), (a.t[:, 0] - b.t[:, 0]).abs().amax(1),
                        _rdev(a, b)], 1).amax(1)


# ---------------------------------------------------------------------------
# 1: Q_K and rho
# ---------------------------------------------------------------------------

def test_q_k_is_the_two_fold_about_b_and_rho_is_the_arccos_form():
    g = torch.Generator().manual_seed(1)
    n = 500
    cells = torch.cat([2 + 20 * torch.rand(n, 3, generator=g, dtype=F64), torch.full((n, 1), math.pi / 2, dtype=F64),
                       math.radians(90) + math.radians(45) * torch.rand(n, 1, generator=g, dtype=F64),
                       torch.full((n, 1), math.pi / 2, dtype=F64)], 1)
    for sg in K_GROUPS:
        k = charts.k_element(sg)
        assert list(group_table(sg).raw['gamma']['W'][k]) == [-1, 0, 0, 0, 1, 0, 0, 0, -1]
        assert torch.equal(group_table(sg).gamma_W[k], torch.diag(torch.tensor([-1., 1., -1.], dtype=F64)))
        assert float((charts.k_rotation(sg, cells) - Q_B).abs().max()) <= 1e-12
        assert charts.KCutHemisphereChart(sg).k_element == k
    for sg in (1, 2, 14, 19):
        with pytest.raises(ValueError, match='K is trivial'):
            charts.k_element(sg)
    # rho: arccos(-u^2 / (1 - u^2)) where u^2 < 1/2 (the arccos is ill-conditioned at the cusp: compared below 0.45),
    # pi where u^2 >= 1/2; in [pi/2, pi]
    th = math.pi / 2 * torch.rand(20000, generator=g, dtype=F64)
    ph = -math.pi + 2 * math.pi * torch.rand(20000, generator=g, dtype=F64)
    rho = charts.KCutHemisphereChart.rho(th, ph)
    u2 = (torch.sin(th) * torch.cos(ph)) ** 2
    ref = torch.arccos(-u2 / (1 - u2))
    lo = u2 < 0.45
    assert float((rho[lo] - ref[lo]).abs().max()) <= 1e-12
    assert bool((rho[u2 >= 0.5] == math.pi).all())
    assert float(rho.min()) >= math.pi / 2 and float(rho.max()) <= math.pi
    # axes along z, y and in the yz plane: rho = pi/2 where u_x = 0 (to the rounding of cos(pi/2))
    r0 = charts.KCutHemisphereChart.rho(torch.tensor([0., math.pi / 2, 0.3], dtype=F64),
                                        torch.tensor([0., math.pi / 2, math.pi / 2], dtype=F64))
    assert float((r0 - math.pi / 2).abs().max()) <= 1e-15


# ---------------------------------------------------------------------------
# 2-3: round trips; the image is the fundamental domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', K_GROUPS)
def test_round_trips_and_the_image_is_the_fundamental_domain(sg):
    ch = _chart(sg)
    assert (ch.k_cut, ch.domain, ch.multiplicity, ch.log_multiplicity) == (True, 'fundamental', 1, 0.0)
    assert isinstance(ch.orientation, charts.KCutHemisphereChart) and ch.orientation.name == 'hemisphere'
    assert [d.name for d in ch.orientation.dims] == ['theta', 'phi', 'r_signed']
    assert [d.kind for d in ch.orientation.dims] == ['interval', 'circle', 'interval']
    n = 1000
    for margin in (0.01, 0.0):
        y, _ = _draw(ch, n, 10 + sg, margin)
        m = ch.forward(y)
        yy, ldi = ch.inverse(m.desc)
        assert float((yy - y).abs().max()) <= 1e-10
        assert float((m.logabsdet + ldi).abs().max()) <= 1e-10
        assert float(_deviation(ch.forward(yy).desc, m.desc).max()) <= 1e-10
    d = m.desc
    assert bool((d.R[:, 0, 0, 0] >= 0).all())
    # every draw is its own canonical form and in the domain
    c = canon.canonicalize(d)
    names = c.names()
    assert set(names) <= {'canonical', 'special_metric'} and names.count('canonical') >= 0.95 * n
    assert float(_deviation(c.desc, d).max()) <= 1e-10
    assert bool(domains.in_domain(d)[0].all())
    # its K image: the same crystal, outside the r bound, not in the domain
    k = ch.k_image(d)
    assert float(_rdev(k, d.replace(R=(Q_B @ d.R[:, 0])[:, None])).max()) <= 1e-12
    yk, _ = ch.inverse(k)
    far = d.R[:, 0, 0, 0] > charts.KCutHemisphereChart.BAND_TOL
    assert int(far.sum()) == n
    assert not bool(_in_bounds(ch, yk, [ch.ndim - 1]).any())
    assert not bool(domains.in_domain(k)[0].any())
    assert float(_deviation(canon.canonicalize(k).desc, c.desc).max()) <= 1e-9
    # measure: the K-cut chart's reference density is finite and log-Jacobian + log V
    lp = measure.log_reference_density(d, ch, 'latent_v1_lebesgue')
    assert bool(torch.isfinite(lp).all())
    assert float((lp - (m.logabsdet + torch.log(measure.cell_volume(d.cell)))).abs().max()) <= 1e-10


@pytest.mark.parametrize('sg', K_GROUPS)
def test_canonical_forms_of_random_crystals_chart_inside(sg):
    """Random crystals (monoclinic cells, any t, Haar R), canonicalised: R00 >= -tol/2, the orientation coordinates
    inside the bounds exactly, and forward(inverse) the canonical row itself (no row lands in the tie band here)."""
    g = torch.Generator().manual_seed(30 + sg)
    n = 2000
    L = torch.exp(math.log(3.0) + (math.log(14.0) - math.log(3.0)) * torch.rand(n, 3, generator=g, dtype=F64))
    ang = torch.full((n, 3), math.pi / 2, dtype=F64)
    ang[:, 1] = math.radians(95) + math.radians(30) * torch.rand(n, generator=g, dtype=F64)
    d = CrystalDescription.from_parts(sg, torch.cat([L, ang], 1), torch.rand(n, 3, generator=g, dtype=F64),
                                      so3.random_rotations(n, g))
    c = canon.canonicalize(d)
    assert set(c.names()) <= {'canonical', 'special_metric'}
    wide = (torch.full((3,), 0.5, dtype=F64), torch.full((3,), 1e3, dtype=F64))
    ch = _chart(sg, wide)
    y, _ = ch.inverse(c.desc)
    k1 = ch.offsets[2]
    assert bool(_in_bounds(ch, y, range(k1, ch.ndim)).all())
    assert float(_deviation(ch.forward(y).desc, c.desc).max()) <= 1e-10
    assert float(c.desc.R[:, 0, 0, 0].min()) >= -FOLD_TOL / 2


# ---------------------------------------------------------------------------
# 4: adversarial rows
# ---------------------------------------------------------------------------

def _frame_with_r00(v, g):
    """[n, 3, 3] rotations with R[0, 0] = v exactly (v [n]): a random unit first column with x-component v, a random
    orthonormal completion."""
    n = len(v)
    a = 2 * math.pi * torch.rand(n, generator=g, dtype=F64)
    s = torch.sqrt((1 - v * v).clamp(min=0))
    c0 = torch.stack([v, s * torch.cos(a), s * torch.sin(a)], 1)
    w = torch.randn(n, 3, generator=g, dtype=F64)
    c1 = w - (w * c0).sum(1, keepdim=True) * c0
    c1 = c1 / c1.norm(dim=1, keepdim=True)
    c2 = torch.linalg.cross(c0, c1)
    R = torch.stack([c0, c1, c2], 2)
    assert torch.equal(R[:, 0, 0], v)
    return R


def _adversarial_rotations(g):
    """Rotations at the places the cut is delicate (each set also with its K image)."""
    out = []
    eps = torch.tensor([0.3, 1e-3, 1e-6, 1e-8, 2e-9, 1e-9, 6e-10, 4e-10, 1e-10, 1e-14, 0.0], dtype=F64)
    v = torch.cat([eps, -eps]).repeat_interleave(20)
    out.append(_frame_with_r00(v, g))
    pi = math.pi
    rows = []
    for th in (pi / 2, pi / 2 - 1e-12, pi / 2 - 1e-7, pi / 2 - 1e-3, 1e-9, 0.0):     # equator, near it, near the pole
        for ux2 in (0.5, 0.5 - 1e-14, 0.5 - 1e-10, 0.5 - 1e-6, 0.5 + 1e-10, 0.25, 0.0, 0.9):
            st = math.sin(th)
            c = math.sqrt(ux2) / st if st > 0 else 0.0
            if c > 1:
                continue
            for ph in (math.acos(c), -math.acos(c), pi - math.acos(c)):
                rho = float(charts.KCutHemisphereChart.rho(torch.tensor([th], dtype=F64), torch.tensor([ph], dtype=F64)))
                for r in (rho, -rho, rho * (1 - 1e-9), rho * (1 + 1e-9), rho - 1e-6, rho + 1e-6, rho + 1e-3,
                          pi, -pi, pi - 1e-12, -(pi - 1e-8), pi - 1e-4, 0.5):
                    if abs(r) <= pi:
                        rows.append((th, ph, r))
    out.append(so3.hemisphere_to_matrix(torch.tensor(rows, dtype=F64)))
    R = torch.cat(out)
    return torch.cat([R, Q_B @ R])


@pytest.mark.parametrize('sg', (6, 9))
def test_the_bound_is_the_folds_choice_on_adversarial_rows(sg):
    g = torch.Generator().manual_seed(40 + sg)
    R = _adversarial_rotations(g)
    n = len(R)
    ch = _chart(sg)
    base = ch.forward(_draw(ch, n, 50 + sg, margin=0.1)[0]).desc       # generic canonical cells and centroids
    d = base.replace(R=R[:, None])
    fr = domains.fold(d)
    Rc = fr.desc.R[:, 0]
    keeps = (Rc - R).abs().amax((1, 2)) <= 1e-12
    takes_k = (Rc - Q_B @ R).abs().amax((1, 2)) <= 1e-12
    assert bool((keeps | takes_k).all())                                    # the stabiliser of a generic t is K
    # wherever the fold decides by the sign of R00 (outside its tie band |R00| <= tol/2), the bound agrees with it
    th, ph, r, rho = _raw_signed(R)
    inside = r.abs() <= rho
    decided = R[:, 0, 0].abs() > FOLD_TOL / 2
    assert int(decided.sum()) > n // 2
    assert torch.equal(inside[decided], keeps[decided] & ~takes_k[decided]), \
        int((inside[decided] != keeps[decided]).sum())
    assert torch.equal(inside[decided], R[decided][:, 0, 0] > 0)
    # control: the bound without the (1 - u_x^2) denominator, arccos(-u_x^2), disagrees with the fold on many rows
    rho_m = torch.arccos(-(torch.sin(th) * torch.cos(ph)) ** 2)
    assert int(((r.abs() <= rho_m)[decided] != keeps[decided]).sum()) > 100
    # the chart's own decision is the fold's: KCutHemisphereChart.fold_keeps equals domains.fold on every row
    assert torch.equal(ch.orientation.fold_keeps(R), keeps & ~takes_k)
    # every fold output charts inside the bounds exactly, and back to itself (tie-band rows turned onto the face
    # R00 = 0, a move of |R00| <= tol/2; rounding left at the cusp u_x^2 = 1/2, |r| = pi put on rho: ~2e-8)
    y, _ = ch.inverse(fr.desc)
    k1 = ch.offsets[2]
    assert bool(_in_bounds(ch, y, range(k1, ch.ndim)).all())
    back = ch.forward(y).desc
    same = _rdev(back, fr.desc)
    assert float(same.max()) <= 1e-7
    assert float(same[fr.desc.R[:, 0, 0, 0] > 1e-6].max()) <= 1e-12
    # the raw rows: those the fold keeps, and the K images it rejects by a later entry (tie band), chart as the
    # fold's choice; the K images it rejects by its first key (R00 < -tol/2) are returned outside the r bound
    yr, _ = ch.inverse(d)
    tie = takes_k & (R[:, 0, 0] >= -FOLD_TOL / 2)
    assert int(tie.sum()) > 0
    assert float(_rdev(ch.forward(yr[keeps | tie]).desc, back[keeps | tie]).max()) <= 1e-7
    rejected = takes_k & ~tie
    assert int(rejected.sum()) > n // 4
    assert not bool(_in_bounds(ch, yr[rejected], [ch.ndim - 1]).any())


def _geodesic(Ra, Rb):
    """[B] rotation angle between two [B, 3, 3] rotations (from the Frobenius distance, well conditioned near 0)."""
    return 2 * torch.asin(((Ra - Rb).flatten(1).norm(dim=1) / (2 * math.sqrt(2))).clamp(max=1))


@pytest.mark.parametrize('sg', (7, 8))
def test_float32_and_off_so3_canonical_rows_chart_inside(sg):
    """Review findings of 2026-10-26: canonical rows stored in float32 (alpha and gamma ~4.4e-8 rad off 90 deg, R
    ~1e-7 off SO(3)) are accepted and agree with fd_status; at the cusp (u_x^2 ~ 1/2, |r| ~ rho ~ pi) canonical rows
    whose R is off SO(3) (float32 origin, or float64 within canonicalize's rot_tol) chart inside the bounds, and
    forward(inverse) moves R by O(eps) (eps their distance from SO(3); an r-only snap moved them by up to 2 sqrt(eps),
    1.4e-3 rad); rho and the r bound are float64 for a float32 prefix."""
    pi = math.pi
    ch = _chart(sg, (torch.full((3,), 0.5, dtype=F64), torch.full((3,), 1e3, dtype=F64)))
    k1 = ch.offsets[2]
    g = torch.Generator().manual_seed(200 + sg)
    # (a) float32-stored canonical rows, generic and within 1e-6 of the cut
    n = 600
    y, _ = _draw(ch, n, 210 + sg, margin=0.02)
    d = ch.forward(y).desc.replace(R=so3.random_rotations(n, g)[:, None])
    v = 10 ** (-9 + 3 * torch.rand(n, generator=g, dtype=F64)) * torch.sign(torch.rand(n, generator=g, dtype=F64) - .5)
    d = CrystalDescription.cat([d, d.replace(R=_frame_with_r00(v, g)[:, None])])
    c = canon.canonicalize(d)
    c32 = c.desc[torch.tensor([s == 'canonical' for s in c.names()])].to(dtype=torch.float32)
    assert c32.num_rows > n
    st = canon.fd_status(c32)
    assert bool(st.is_fd.all())
    y32, _ = ch.inverse(c32)
    inside = _in_bounds(ch, y32, range(k1, ch.ndim))
    assert bool(inside.all())
    assert float(_deviation(ch.forward(y32).desc, c32).max()) <= 1e-5          # fd_status's float32 tolerance
    # (b) rows at the cusp, R00 small and positive
    m = 1500
    th = pi / 4 + (pi / 4) * torch.rand(m, generator=g, dtype=F64)
    ux = torch.sqrt(0.5 - 10 ** (-12 + 8 * torch.rand(m, generator=g, dtype=F64)))
    ph = torch.arccos((ux / torch.sin(th)).clamp(max=1)) * torch.sign(torch.rand(m, generator=g, dtype=F64) - .5)
    rho = charts.KCutHemisphereChart.rho(th, ph)
    r = rho * (1 - 10 ** (-10 + 6 * torch.rand(m, generator=g, dtype=F64))) \
        * torch.sign(torch.rand(m, generator=g, dtype=F64) - .5)
    R = so3.hemisphere_to_matrix(torch.stack([th, ph, r], 1))
    base = ch.forward(_draw(ch, m, 220 + sg, margin=0.1)[0]).desc
    noise = 1e-7 * torch.randn(m, 3, 3, generator=g, dtype=F64)
    noise[:, 0, 0] = 0
    for x in (base.replace(R=R[:, None]).to(dtype=torch.float32), base.replace(R=(R + noise)[:, None])):
        c = canon.canonicalize(x)
        cd = c.desc[torch.tensor([s == 'canonical' for s in c.names()])]
        assert cd.num_rows > m // 2
        assert bool(canon.fd_status(cd).is_fd.all())
        yc, _ = ch.inverse(cd)
        assert bool(_in_bounds(ch, yc, range(k1, ch.ndim)).all())
        Rm = cd.R[:, 0]
        eps = (Rm @ Rm.transpose(1, 2) - torch.eye(3, dtype=F64)).abs().amax((1, 2))
        move = _geodesic(ch.forward(yc).desc.R[:, 0], so3.nearest_rotation(Rm))
        assert bool((move <= 3 * eps + 1e-7).all()), float((move - 3 * eps).max())
    # (c) dtypes
    yp = torch.tensor([[0.7, 0.3], [1.2, -2.0]], dtype=torch.float32)
    lo, hi = ch.orientation.bounds(yp, 2)
    assert lo.dtype == hi.dtype == F64
    assert float((hi - charts.KCutHemisphereChart.rho(yp[:, 0].double(), yp[:, 1].double())).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 5: measure
# ---------------------------------------------------------------------------

def test_the_cut_has_haar_mass_one_half():
    # quadrature: (1 / 2 pi^2) * integral of (rho - sin rho) sin theta dtheta dphi (midpoint rule; the cusp limits it)
    n = 2000
    th = (torch.arange(n, dtype=F64) + 0.5) * (math.pi / 2) / n
    ph = -math.pi + (torch.arange(2 * n, dtype=F64) + 0.5) * math.pi / n
    TH, PH = torch.meshgrid(th, ph, indexing='ij')
    rho = charts.KCutHemisphereChart.rho(TH, PH)
    mass = float(((rho - torch.sin(rho)) * torch.sin(TH)).sum() * (math.pi / 2 / n) * (math.pi / n)
                 / (2 * math.pi ** 2))
    assert abs(mass - 0.5) <= 1e-6, mass
    rho_m = torch.arccos(-(torch.sin(TH) * torch.cos(PH)) ** 2)         # control: a wrong bound has another mass
    mass_m = float(((rho_m - torch.sin(rho_m)) * torch.sin(TH)).sum() * (math.pi / 2 / n) * (math.pi / n)
                   / (2 * math.pi ** 2))
    assert abs(mass_m - 0.5) >= 0.01, mass_m
    # Monte Carlo through the AR bounds (uniform u, the chart's log volume and Haar density)
    ori = charts.KCutHemisphereChart(9)
    u = torch.rand(400_000, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
    y, logv = charts.from_unit(ori, u)
    w = torch.exp(logv + ori.map(y)[1])
    est, se = float(w.mean()), float(w.std() / math.sqrt(len(w)))
    assert abs(est - 0.5) <= 5 * se, (est, se)
    # the plain hemisphere chart, as a control, has mass 1
    pl = charts.OrientationChart('hemisphere')
    y, logv = charts.from_unit(pl, u)
    w = torch.exp(logv + pl.map(y)[1])
    assert abs(float(w.mean()) - 1.0) <= 5 * float(w.std() / math.sqrt(len(w)))
    # Haar rotations: inside the bound exactly where R00 >= 0, half of them
    R = so3.random_rotations(200_000, torch.Generator().manual_seed(3))
    _, _, r, rho = _raw_signed(R)
    inside = r.abs() <= rho
    assert torch.equal(inside, R[:, 0, 0] >= 0)
    assert abs(float(inside.double().mean()) - 0.5) <= 5 * math.sqrt(0.25 / len(R))


def _mc_mass(ch, n, seed):
    g = torch.Generator().manual_seed(seed)
    y, logv = charts.from_unit(ch, torch.rand(n, ch.ndim, generator=g, dtype=F64))
    w = torch.exp(measure.log_reference_density(ch.forward(y).desc, ch, 'latent_v1_lebesgue') + logv)
    return float(w.mean()), float(w.std() / math.sqrt(n))


@pytest.mark.parametrize('sg', (7, 8))
def test_log_z_of_the_double_cover_is_the_k_cut_plus_log_2(sg):
    box = (torch.full((3,), 4.0, dtype=F64), torch.full((3,), 9.0, dtype=F64))
    fd = charts.CrystalChart.for_length_box(sg, box, 'hemisphere')
    dc = charts.CrystalChart.for_length_box(sg, box, 'hemisphere', orientation_cover='double')
    assert fd.k_cut and not dc.k_cut and (fd.domain, dc.domain) == ('fundamental', 'double_cover')
    assert dc.log_multiplicity - fd.log_multiplicity == math.log(2)
    assert measure.log_multiplicity(sg, 'fundamental', 'double_cover') == math.log(2)
    assert measure.images_per_crystal(sg, 'fundamental') == 1
    z_fd, se_fd = _mc_mass(fd, 300_000, 70 + sg)
    z_dc, se_dc = _mc_mass(dc, 300_000, 80 + sg)
    diff = math.log(z_dc) - math.log(z_fd) - math.log(2)
    assert abs(diff) <= 5 * math.hypot(se_fd / z_fd, se_dc / z_dc), (z_fd, z_dc, diff)
    # pointwise: a canonical row has one reference density on both charts (sin^2(r/2) is even in r and 2 pi periodic)
    d = fd.forward(_draw(fd, 2000, 90 + sg)[0]).desc
    lp_fd = measure.log_reference_density(d, fd, 'latent_v1_lebesgue')
    lp_dc = measure.log_reference_density(d, dc, 'latent_v1_lebesgue')
    assert float((lp_fd - lp_dc).abs().max()) <= 1e-10


# ---------------------------------------------------------------------------
# 6: gluings
# ---------------------------------------------------------------------------

def test_equator_pi_end_and_cut_face_gluings():
    ori = charts.KCutHemisphereChart(6)
    g = torch.Generator().manual_seed(5)
    n = 2000
    ph = -math.pi + 2 * math.pi * torch.rand(n, generator=g, dtype=F64)
    th = torch.full((n,), math.pi / 2, dtype=F64)
    rho = ori.rho(th, ph)
    r = rho * (2 * torch.rand(n, generator=g, dtype=F64) - 1)
    a = torch.stack([th, ph, r], 1)
    b = torch.stack([th, ph + math.pi, -r], 1)                         # the equator gluing
    assert float((ori.rho(b[:, 0], b[:, 1]) - rho).abs().max()) <= 1e-12
    (Ra, la), (Rb, lb) = ori.map(a), ori.map(b)
    assert float((Ra - Rb).abs().max()) <= 1e-12 and float((la - lb).abs().max()) <= 1e-12
    # the pi end: r = -pi and pi where rho = pi are one rotation (u_x^2 >= 1/2 here)
    th2 = math.pi / 2 - 0.3 * torch.rand(n, generator=g, dtype=F64)
    ph2 = 0.3 * (2 * torch.rand(n, generator=g, dtype=F64) - 1)
    keep = ori.rho(th2, ph2) == math.pi
    assert int(keep.sum()) > n // 2
    p = torch.stack([th2, ph2, torch.full((n,), math.pi, dtype=F64)], 1)[keep]
    q = p.clone()
    q[:, 2] = -math.pi
    (Rp, lpp), (Rq, lq) = ori.map(p), ori.map(q)
    assert float((Rp - Rq).abs().max()) <= 1e-12 and float((lpp - lq).abs().max()) <= 1e-12
    # the cut face: R00 = 0 there, and K maps it to itself (R00 -> -R00 = 0, inside the closed bound after the snap)
    th3 = math.pi / 2 * torch.rand(n, generator=g, dtype=F64)
    ph3 = -math.pi + 2 * math.pi * torch.rand(n, generator=g, dtype=F64)
    rho3 = ori.rho(th3, ph3)
    face = (rho3 < math.pi - 1e-3)
    y3 = torch.stack([th3, ph3, rho3], 1)[face]
    R3, _ = ori.map(y3)
    assert float(R3[:, 0, 0].abs().max()) <= 1e-12
    yk, _ = ori.unmap(ori.k_image(R3))
    lo, hi = ori.bounds(yk[:, :2], 2)
    assert float(((yk[:, 2].abs() - hi).abs()).max()) <= 1e-6


# ---------------------------------------------------------------------------
# 7: guards; other groups unchanged
# ---------------------------------------------------------------------------

def test_guards():
    for name in ('euler_zyz', 'axis_angle', 'quaternion'):
        with pytest.raises(NotImplementedError, match='hemisphere chart only'):
            charts.CrystalChart.for_length_box(7, BOX, name)
        charts.CrystalChart.for_length_box(7, BOX, name, orientation_cover='double')        # the cover still builds
    with pytest.raises(ValueError, match='K is trivial'):
        charts.KCutHemisphereChart(14)
    cell6 = charts.CellChart('monoclinic', BOX, sg=6)
    with pytest.raises(ValueError, match='KCutHemisphereChart'):                 # a K-cut block where K is trivial
        charts.CrystalChart(10, charts.CellChart('monoclinic', BOX, sg=10), charts.PolytopeARChart(10),
                            charts.KCutHemisphereChart(6))
    with pytest.raises(ValueError, match='KCutHemisphereChart'):                 # ... or covered
        charts.CrystalChart(6, cell6, charts.PolytopeARChart(6), charts.KCutHemisphereChart(6),
                            orientation_cover='double')
    with pytest.raises(ValueError, match='is for sg 9'):
        charts.CrystalChart(6, cell6, charts.PolytopeARChart(6), charts.KCutHemisphereChart(9))
    explicit = charts.CrystalChart(6, cell6, charts.PolytopeARChart(6), charts.KCutHemisphereChart(6))
    assert explicit.k_cut and explicit.domain == 'fundamental'

    class _Mutant(charts.OrientationChart):          # a subclass named 'hemisphere' is not silently replaced
        pass
    with pytest.raises(NotImplementedError, match='cuts K out of SO'):
        charts.CrystalChart(6, cell6, charts.PolytopeARChart(6), _Mutant('hemisphere'))
    # a row whose Q_K is off the 2-fold about y (gamma 1e-5 rad off 90 deg) is refused; float32 storage (alpha and
    # gamma ~4.4e-8 rad off) and float64 rounding are not
    assert charts.KCutHemisphereChart.FRAME_TOL == 1e-6
    ch = _chart(6)
    d = ch.forward(_draw(ch, 4, 3)[0]).desc
    cell = d.cell.clone()
    cell[0, 5] += 1e-5
    with pytest.raises(ValueError, match='FRAME_TOL'):
        ch.inverse(d.replace(cell=cell))
    cell[0, 5] = math.pi / 2 + 1e-15
    ch.inverse(d.replace(cell=cell))
    f32_half_pi = float(torch.tensor(math.pi / 2, dtype=torch.float32))
    assert abs(f32_half_pi - math.pi / 2) > 4e-8
    cell[:, [3, 5]] = f32_half_pi
    ch.inverse(d.replace(cell=cell))
    ch.inverse(d.to(dtype=torch.float32))
    with pytest.raises(ValueError, match='K is nontrivial'):
        _chart(14).k_image(d.replace(sg=torch.full_like(d.sg, 14)))


NON_K = (1, 2, 3, 4, 5, 10, 14, 15, 19, 61, 92, 148, 198)


@pytest.mark.parametrize('sg', NON_K)
def test_other_groups_keep_the_plain_orientation_chart(sg):
    ch = charts.CrystalChart.for_length_box(sg, BOX, 'hemisphere')
    assert type(ch.orientation) is charts.OrientationChart and not ch.k_cut and ch.domain == 'fundamental'
    assert ch.orientation.dims == charts._ORIENTATION_DIMS['hemisphere']
    y, _ = _draw(ch, 300, 100 + sg)
    m = ch.forward(y)
    k0, k1 = ch.offsets[1], ch.offsets[2]
    cell, l0 = ch.cell.map(y[:, :k0])
    t, l1 = ch.centroid.map(y[:, k0:k1])
    R, l2 = charts.OrientationChart('hemisphere').map(y[:, k1:])
    assert torch.equal(m.desc.cell, cell) and torch.equal(m.desc.t[:, 0], t) and torch.equal(m.desc.R[:, 0], R)
    assert torch.equal(m.logabsdet, l0 + l1 + l2)
    yy, ld = ch.inverse(m.desc)
    y2, m2 = charts.OrientationChart('hemisphere').unmap(m.desc.R[:, 0])
    assert torch.equal(yy[:, k1:], y2)


# ---------------------------------------------------------------------------
# 8: canonical CSD rows (local data, read only)
# ---------------------------------------------------------------------------

_V4 = os.environ.get('MXT_FD_CSD_DATASET', 'D:/crystal_datasets/csd_fd_v4')


@pytest.mark.slow
@pytest.mark.skipif(not os.path.isdir(_V4), reason='canonical CSD dataset not available')
def test_canonical_csd_rows_chart_inside_the_k_cut():
    from mxtaltools.crystal_fd import dataset
    ds = dataset.load_dataset(_V4, status=('canonical', 'special_metric'), verify=False, code='ignore')
    wide = (torch.full((3,), 0.5, dtype=F64), torch.full((3,), 1e3, dtype=F64))
    n_rows = 0
    for sg in K_GROUPS:
        rows = ds.desc[ds.desc.sg == sg].double()
        if rows.num_rows == 0:
            continue
        ch = _chart(sg, wide)
        y, _ = ch.inverse(rows)
        k = ch.cell.ndim
        k1 = ch.offsets[2]
        assert bool(_in_bounds(ch, y, range(k)).all()), sg
        assert bool(_in_bounds(ch, y, range(k, k1), 1e-7).all()), sg              # centroid: the fold tolerance in u
        assert bool(_in_bounds(ch, y, range(k1, ch.ndim)).all()), sg             # orientation: exactly
        assert float(_deviation(ch.forward(y).desc, rows).max()) <= 1e-10, sg
        n_rows += rows.num_rows
    assert n_rows > 1000

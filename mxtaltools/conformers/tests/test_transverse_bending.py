"""Transverse bending coordinates: a chart that is regular at a linear centre.

THE PROBLEM. An atom placed by the NeRF chain uses ``(theta, phi)``, a POLAR chart on the
bend direction. ``theta = pi`` is its pole: ``phi`` is undefined there and the volume element
``sin(theta) d theta d phi`` collapses, so ``log sin(theta)`` diverges to ``-inf``. An sp
centre -- alkyne, nitrile, allene -- has its EQUILIBRIUM at exactly that point. The chart
therefore places an infinitely repulsive wall on the geometry the molecule actually prefers,
and the previous treatment was to drop those rows, freezing a real degree of freedom.

THE CHART. With ``rho = pi - theta`` the bend magnitude and ``phi`` the bending plane,

    u = rho cos(phi)    v = rho sin(phi)

are the Cartesian components of the same bend: two coordinates in, two out, nothing added or
removed. They are regular at the pole -- both zero, with no undefined direction -- and

    sin(theta) d theta d phi = (sin(rho) / rho) du dv

so the divergent ``log sin(theta)`` becomes ``log sinc(rho)``, which is smooth and ZERO at the
linear geometry.

WHAT IS AND IS NOT ESTABLISHED HERE. Rank and measure are checked SEPARATELY, at the linear
reference and at perturbations, because a chart can have the right rank and the wrong density
and neither is implied by the number of coordinates. ``3N - 6`` is treated as the expected
PHYSICAL rank to be measured, not as something the column count proves.

Two other degeneracies are NOT covered by this change, and ``test_uncovered_degeneracies``
pins that rather than leaving it implied:

  * a linear angle at the THIRD SEED ATOM, whose out-of-plane component is the sixth external
    DoF -- when atoms 0-1-2 are collinear the "third atom in the xy half-plane" convention
    does not fix a frame at all;
  * ``torsion_frame_is_linear``, where the a-b-c reference triple is collinear so the normal
    defining ``phi`` is arbitrary. That needs a smooth frame construction, not a convention
    evaluated at exactly zero bend.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mxtaltools.conformers import build, collate, log_jacobian, measure, spec_from_smiles
from mxtaltools.conformers.builder import _transverse_atom_mask
from mxtaltools.conformers.geometry import (log_sinc, measure_transverse, place_nerf,
                                            place_nerf_transverse, polar_from_transverse,
                                            transverse_from_polar)

DTYPE = torch.float64

#: ``CC#N`` is the one molecule here whose linear centre is FULLY covered -- its linear angle
#: places a non-seed atom. The alkynes are deliberately included because they are NOT, and the
#: coverage test is what says so out loud.
COVERED = ["CC#N", "CCC#N"]
UNCOVERED = ["CC#C", "CC#CCO", "CCC#CC"]
PLAIN = ["CCCCO", "C1CCCCC1O"]
EPS = 1e-3


@pytest.fixture(autouse=True)
def _default_dtype():
    """Double precision, per test -- see the note in test_conformers.py on why not at import."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# --------------------------------------------------------------------------- helpers

def _load(smiles, seed=7):
    mol, spec = spec_from_smiles(smiles, seed=seed)
    tree = collate([spec])
    pos = torch.tensor(np.asarray(mol.GetConformer().GetPositions(),
                                  dtype=np.float64)[spec.perm])
    return spec, tree, pos


def _eligible(spec, tree):
    """Angle rows that HAVE a torsion partner -- the rows a transverse pair can occupy."""
    has_phi = torch.zeros(spec.n_atoms, dtype=torch.bool)
    has_phi[tree.torsion_index[:, 3]] = True
    return has_phi[tree.angle_index[:, 2]]


def _covered(spec, tree):
    """Linear angle rows the transverse pair actually reaches -- ALL THREE conditions.

    Must match `ConformerTorsions.transverse_angles` exactly, or this file tests a rule the
    energy does not apply. The third condition is the easy one to drop: a row whose reference
    frame a-b-c is collinear has no well-defined normal to bend against, so flagging it would
    put the transverse kernel on a frame that is itself degenerate. It happens to pass
    numerically at these geometries, which is precisely why it needs to be excluded on
    purpose rather than by luck.
    """
    lin = torch.from_numpy(np.asarray(spec.angle_is_linear, dtype=bool))
    frame = torch.from_numpy(np.asarray(spec.torsion_frame_is_linear, dtype=bool))
    tor_atom = tree.torsion_index[:, 3]
    frame_at = torch.zeros(spec.n_atoms, dtype=torch.bool).index_copy(0, tor_atom, frame)
    return lin & _eligible(spec, tree) & ~frame_at[tree.angle_index[:, 2]]


def _to_transverse(spec, tree, th, ph, mask):
    """Replace the flagged rows' ``(theta, phi)`` with ``(u, v)``, leaving the rest alone.

    Routed through the ATOM each row places: theta and phi index different arrays, so pairing
    them by row number would silently mix up which bend belongs to which atom.
    """
    n = spec.n_atoms
    ph_at = torch.zeros(n, dtype=DTYPE).index_copy(0, tree.torsion_index[:, 3], ph)
    u, v = transverse_from_polar(th, ph_at[tree.angle_index[:, 2]])
    v_at = torch.zeros(n, dtype=DTYPE).index_copy(0, tree.angle_index[:, 2], v)
    m_at = torch.zeros(n, dtype=torch.bool).index_copy(0, tree.angle_index[:, 2], mask)
    return (torch.where(mask, u, th),
            torch.where(m_at[tree.torsion_index[:, 3]], v_at[tree.torsion_index[:, 3]], ph))


def _free(n_atoms):
    """The 3N-6 Cartesian components the seed convention leaves free."""
    m = np.zeros((n_atoms, 3), dtype=bool)
    m[1, 0] = True
    m[2, :2] = True
    m[3:, :] = True
    return torch.from_numpy(m.reshape(-1))


def _jac(tree, r, th, ph, tv, keep):
    f = lambda a, b, c: build(tree, a, b, c, transverse=tv).reshape(-1)[keep]
    return torch.cat(torch.autograd.functional.jacobian(f, (r, th, ph), vectorize=True), 1)


def _gauge(tree, r, th):
    """``2 log r1 + log r2 + log sin(theta2)``: the internals the frame convention consumes.

    The same factor ``test_log_jacobian`` in test_conformers.py already accounts for -- atom 1
    is pinned to +x so it contributes 1 rather than r1^2, atom 2 to the xy-plane so it
    contributes r2 rather than r2^2 sin(theta2). Restated here because the measure check below
    is only meaningful once the reference measure is named: ``log_jacobian`` returns the BAT
    element with the six rigid motions INTEGRATED OUT, while differentiating ``build`` gives
    the volume of one SECTION of that quotient. Comparing them without this term is a category
    error, not a discrepancy.
    """
    r_at = torch.zeros(tree.n_atoms, dtype=r.dtype).index_copy(0, tree.bond_index[:, 1], r)
    th_at = torch.zeros(tree.n_atoms, dtype=th.dtype).index_copy(0, tree.angle_index[:, 2], th)
    return 2 * torch.log(r_at[1]) + torch.log(r_at[2]) + torch.log(torch.sin(th_at[2]))


# --------------------------------------------------------------------------- the kernel

@pytest.mark.parametrize("lo,hi", [(0.3, 2.8), (3.0, 3.13), (3.1415, 3.14159)])
def test_kernel_agrees_with_the_polar_placement(lo, hi):
    """Same geometry, all the way in to rho ~ 6e-5. Not an approximation of the old chart."""
    g = torch.Generator().manual_seed(0)
    pa, pb, pc = (torch.randn(64, 3, generator=g, dtype=DTYPE) for _ in range(3))
    r = 1.0 + 0.3 * torch.rand(64, generator=g, dtype=DTYPE)
    th = lo + (hi - lo) * torch.rand(64, generator=g, dtype=DTYPE)
    ph = (2 * torch.rand(64, generator=g, dtype=DTYPE) - 1) * torch.pi
    u, v = transverse_from_polar(th, ph)
    a = place_nerf(pa, pb, pc, r, th, ph)
    b = place_nerf_transverse(pa, pb, pc, r, u, v)
    assert float((a - b).abs().max()) < 1e-12


@pytest.mark.parametrize("amp", [1e-1, 1e-3, 1e-6, 0.0])
def test_kernel_gradient_is_finite_and_correct_through_the_pole(amp):
    """THE defect this chart exists to remove, including at u = v = 0 EXACTLY.

    A first implementation divided by an unclamped ``rho2`` inside one branch of a
    ``torch.where``. Its VALUE was exact to 4e-16 everywhere -- the test above passed -- and
    its GRADIENT at the pole was NaN, because ``where`` differentiates the branch it discards
    and ``0 * NaN`` is NaN. Checking the value alone would have shipped it.
    """
    g = torch.Generator().manual_seed(1)
    pa, pb, pc = (torch.randn(32, 3, generator=g, dtype=DTYPE) for _ in range(3))
    r = 1.0 + 0.3 * torch.rand(32, generator=g, dtype=DTYPE)
    u = torch.full((32,), amp, dtype=DTYPE, requires_grad=True)
    v = torch.full((32,), amp * 0.5, dtype=DTYPE, requires_grad=True)

    out = place_nerf_transverse(pa, pb, pc, r, u, v)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all()

    h = 1e-6
    f = lambda a, b: place_nerf_transverse(pa, pb, pc, r, a, b).sum()
    with torch.no_grad():
        fd_u = (f(u + h, v) - f(u - h, v)) / (2 * h)
        fd_v = (f(u, v + h) - f(u, v - h)) / (2 * h)
    assert abs(float(u.grad.sum()) - float(fd_u)) < 1e-6
    assert abs(float(v.grad.sum()) - float(fd_v)) < 1e-6
    # the bend MOVES atoms at the pole; a zero gradient there would be a different bug
    assert float(u.grad.abs().mean()) > 1e-3


def test_the_kernel_survives_float32():
    """THE DTYPE THE RUNS ACTUALLY USE. Every other test here forces float64.

    That is right for the rank and measure checks -- an SVD and a log-determinant have no
    margin in single -- but it means the whole file would otherwise certify a precision the
    training loop never runs at. `conformer_modeller` sets float32 globally, and the guard
    thresholds here (`small_eps = 1e-8`, clamped to `1e-16`) are absolute, so whether they
    still separate the branches in single is a question, not an inference.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        g = torch.Generator().manual_seed(3)
        pa, pb, pc = (torch.randn(64, 3, generator=g) for _ in range(3))
        r = 1.0 + 0.3 * torch.rand(64, generator=g)
        for lo, hi in ((0.3, 2.8), (3.14159, 3.141592)):
            th = lo + (hi - lo) * torch.rand(64, generator=g)
            ph = (2 * torch.rand(64, generator=g) - 1) * torch.pi
            u, v = transverse_from_polar(th, ph)
            d = (place_nerf(pa, pb, pc, r, th, ph)
                 - place_nerf_transverse(pa, pb, pc, r, u, v)).abs().max()
            assert float(d) < 1e-5, f'float32 disagreement {float(d):.2e} at theta ~ {lo}'

        u = torch.zeros(64, requires_grad=True)
        v = torch.zeros(64, requires_grad=True)
        out = place_nerf_transverse(pa, pb, pc, r, u, v)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all()
        assert float(u.grad.abs().mean()) > 1e-3
        assert torch.isfinite(log_sinc(u.detach(), v.detach())).all()
    finally:
        torch.set_default_dtype(prev)


def test_the_polar_round_trip_is_the_thing_being_avoided():
    """``atan2(v, u)`` is why the placement is written in (u, v) directly.

    Pinned as a NEGATIVE result: recovering ``(theta, phi)`` and calling the ordinary
    placement would agree on values and reintroduce the singularity into the gradient, which
    is the failure mode that looks like it works.
    """
    u = torch.zeros(8, dtype=DTYPE, requires_grad=True)
    v = torch.zeros(8, dtype=DTYPE, requires_grad=True)
    theta, phi = polar_from_transverse(u, v)
    (theta.sum() + phi.sum()).backward()
    assert not torch.isfinite(u.grad).all()


def test_measurement_inverts_the_kernel_at_the_pole():
    """``measure_transverse`` must not go through the azimuthal atan2 either."""
    g = torch.Generator().manual_seed(2)
    pa, pb, pc = (torch.randn(64, 3, generator=g, dtype=DTYPE) for _ in range(3))
    r = 1.0 + 0.3 * torch.rand(64, generator=g, dtype=DTYPE)
    for amp in (0.5, 1e-3, 1e-8, 0.0):
        u = torch.full((64,), amp, dtype=DTYPE)
        v = torch.full((64,), -amp * 0.3, dtype=DTYPE)
        pn = place_nerf_transverse(pa, pb, pc, r, u, v)
        u2, v2 = measure_transverse(pa, pb, pc, pn)
        assert float((u2 - u).abs().max()) < 1e-9, f"u round trip at amp {amp}"
        assert float((v2 - v).abs().max()) < 1e-9, f"v round trip at amp {amp}"


# --------------------------------------------------------------------------- the measure

def test_log_sinc_is_finite_where_log_sin_diverges():
    """The headline: a smooth measure at the geometry the polar chart forbids."""
    rho = torch.tensor([0.5, 1e-2, 1e-8, 0.0], dtype=DTYPE)
    val = log_sinc(rho, torch.zeros_like(rho))
    assert torch.isfinite(val).all()
    assert abs(float(val[-1])) < 1e-15                      # exactly zero at the pole
    assert float(torch.log(torch.sin(torch.pi - rho[-2:]))[-1]) < -30   # the wall, for contrast


def test_log_sinc_gradient_vanishes_at_the_pole():
    """A flat measure at a bend with no preferred azimuth -- and finite, not merely small."""
    u = torch.zeros(4, dtype=DTYPE, requires_grad=True)
    v = torch.zeros(4, dtype=DTYPE, requires_grad=True)
    log_sinc(u, v).sum().backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all()
    assert float(u.grad.abs().max()) < 1e-12


def test_log_sinc_is_minus_inf_outside_the_disc_never_nan():
    """The chart is valid on rho < pi. Out of domain must be a visible zero, not a NaN.

    rho = pi is theta = 0 -- the placed atom folded back onto the b-c axis, on top of its own
    grandparent. A vanishing measure there is CORRECT: the singularity is moved from a real
    equilibrium to a sterically forbidden point, which is the trade the chart makes.
    """
    u = torch.tensor([3.0, np.pi, 4.0, 10.0], dtype=DTYPE)
    out = log_sinc(u, torch.zeros_like(u))
    assert not torch.isnan(out).any()
    assert torch.isfinite(out[0]) and bool((out[1:] == -float("inf")).all())


# --------------------------------------------------------------------------- the builder

@pytest.mark.parametrize("smiles", COVERED + UNCOVERED + PLAIN)
def test_an_unflagged_build_is_untouched(smiles):
    """``transverse=None`` and an all-False mask must be the SAME bytes as before."""
    spec, tree, pos = _load(smiles)
    r, th, ph = measure(tree, pos)
    off = build(tree, r, th, ph)
    allf = build(tree, r, th, ph, transverse=torch.zeros(tree.angle_index.shape[0],
                                                         dtype=torch.bool))
    assert float((off - allf).abs().max()) == 0.0


@pytest.mark.parametrize("smiles", COVERED + UNCOVERED + PLAIN)
def test_every_eligible_row_may_be_transverse_and_inverts(smiles):
    """Flag EVERY eligible row, not just the linear ones: same geometry, and measure inverts.

    Exercised beyond the linear rows on purpose. The chart is supposed to be a change of
    coordinates valid everywhere, so restricting the test to near-linear rows would leave the
    general claim unchecked and hide an error that only shows at ordinary bond angles.
    """
    spec, tree, pos = _load(smiles)
    r, th, ph = measure(tree, pos)
    tv = _eligible(spec, tree)
    th_t, ph_t = _to_transverse(spec, tree, th, ph, tv)
    pos_t = build(tree, r, th_t, ph_t, transverse=tv)
    assert float((build(tree, r, th, ph) - pos_t).norm(dim=-1).max()) < 1e-12

    r2, th2, ph2 = measure(tree, pos_t, transverse=tv)
    assert float((r2 - r).abs().max()) < 1e-9
    assert float((th2 - th_t).abs().max()) < 1e-9
    assert float((ph2 - ph_t).abs().max()) < 1e-9


@pytest.mark.parametrize("smiles", COVERED + UNCOVERED + PLAIN)
@pytest.mark.parametrize("pert", [0.0, 0.02])
def test_rank_is_3n_minus_6_in_both_charts(smiles, pert):
    """RANK, kept separate from measure. Measured by SVD, not inferred from column count.

    ``pert = 0`` is the embedded reference, which for the alkynes and the nitrile IS the
    near-linear geometry; ``pert = 0.02`` is a random displacement off it. Both are required:
    a chart can be full rank away from the pole and drop rank on it.
    """
    spec, tree, pos = _load(smiles)
    keep = _free(spec.n_atoms)
    r, th, ph = measure(tree, pos)
    cov = _covered(spec, tree)
    expect = 3 * spec.n_atoms - 6
    # THREE charts, not two. `cov` is empty for a molecule with no linear centre, so testing
    # only it would quietly re-run the polar case twice and leave the transverse chart
    # unexercised on most of the list. `_eligible` puts EVERY non-seed row in the transverse
    # form, which is the general claim; `cov` is the production configuration.
    for tv in (None, cov if bool(cov.any()) else None, _eligible(spec, tree)):
        b_th, b_ph = (th, ph) if tv is None else _to_transverse(spec, tree, th, ph, tv)
        g = torch.Generator().manual_seed(0)
        a = (r + pert * torch.randn(r.shape, generator=g, dtype=DTYPE)).requires_grad_()
        b = (b_th + pert * torch.randn(b_th.shape, generator=g, dtype=DTYPE)).requires_grad_()
        c = (b_ph + pert * torch.randn(b_ph.shape, generator=g, dtype=DTYPE)).requires_grad_()
        s = torch.linalg.svdvals(_jac(tree, a, b, c, tv, keep))
        assert int((s > 1e-8 * s[0]).sum()) == expect, (
            f"{smiles}: rank {int((s > 1e-8 * s[0]).sum())} against 3N-6 = {expect}")


@pytest.mark.parametrize("smiles", COVERED + UNCOVERED + PLAIN)
@pytest.mark.parametrize("pert", [0.0, 0.01, 0.05])
def test_measure_matches_the_section_volume_times_the_gauge(smiles, pert):
    """MEASURE, kept separate from rank, against a NAMED reference measure.

    ``log_jacobian`` is the BAT element with rigid motions integrated out; the numerical
    object is the k-volume of one section of that quotient, taken as ``sqrt(det(J^T J))``
    because the map is rectangular once a ring closure removes a tree DoF -- a raw determinant
    is not the check. The two differ by :func:`_gauge` and by nothing else, which is a
    prediction with no free parameters.
    """
    spec, tree, pos = _load(smiles)
    keep = _free(spec.n_atoms)
    r0, th0, ph0 = measure(tree, pos)
    cov = _covered(spec, tree)
    charts = [None, _eligible(spec, tree)] + ([cov] if bool(cov.any()) else [])
    for tv in charts:
        b_th, b_ph = (th0, ph0) if tv is None else _to_transverse(spec, tree, th0, ph0, tv)
        g = torch.Generator().manual_seed(1)
        a = (r0 + pert * torch.randn(r0.shape, generator=g, dtype=DTYPE)).requires_grad_()
        b = b_th + pert * torch.randn(b_th.shape, generator=g, dtype=DTYPE)
        c = b_ph + pert * torch.randn(b_ph.shape, generator=g, dtype=DTYPE)
        # polar rows must stay inside (0, pi); an out-of-domain angle is an artifact of the
        # perturbation, not of the chart, and the sampler clamps them for the same reason
        polar = torch.ones_like(b, dtype=torch.bool) if tv is None else ~tv
        b = torch.where(polar, b.clamp(EPS, np.pi - EPS), b).requires_grad_()
        c = c.requires_grad_()

        analytic = log_jacobian(tree, a, b, c if tv is not None else None, transverse=tv)
        numeric = torch.log(torch.linalg.svdvals(_jac(tree, a, b, c, tv, keep))).sum()
        residual = float(analytic.sum() - numeric - _gauge(tree, a, b))
        assert abs(residual) < 1e-9, f"{smiles} pert {pert}: residual {residual:.3e}"


def test_the_wall_is_gone_at_a_real_sp_centre():
    """End to end on acetonitrile: drive the bend through zero and compare the two charts."""
    spec, tree, pos = _load("CC#N")
    r0, th0, ph0 = measure(tree, pos)
    cov = _covered(spec, tree)
    assert bool(cov.any()), "CC#N should have a covered linear centre"
    row = int(cov.nonzero()[0])
    atom = int(tree.angle_index[row, 2])
    trow = int((tree.torsion_index[:, 3] == atom).nonzero()[0])
    u0, v0 = _to_transverse(spec, tree, th0, ph0, cov)

    polar, trans = [], []
    for rho in (1e-1, 1e-2, 1e-4, 1e-8, 0.0):
        th = th0.clone()
        th[row] = np.pi - rho
        th = th.requires_grad_()
        lp = log_jacobian(tree, r0, th)
        polar.append((float(lp), float(torch.autograd.grad(lp.sum(), th)[0][row])))

        u = u0.clone()
        u[row] = rho
        u = u.requires_grad_()
        v = v0.clone()
        v[trow] = 0.0
        lt = log_jacobian(tree, r0, u, v, transverse=cov)
        trans.append((float(lt), float(torch.autograd.grad(lt.sum(), u)[0][row])))

    # the polar measure falls without bound and its gradient grows as 1/rho
    assert polar[0][0] > polar[-2][0] + 15
    assert abs(polar[-2][1]) > 1e7
    # the transverse measure is flat and finite, and its gradient GOES TO ZERO
    assert all(np.isfinite([x for p in trans for x in p]))
    assert max(t[0] for t in trans) - min(t[0] for t in trans) < 0.02
    assert abs(trans[-1][1]) < 1e-12


# --------------------------------------------------------------------------- the limits

def test_uncovered_degeneracies_are_counted_not_assumed():
    """What the transverse pair does NOT fix, pinned so the claim cannot drift.

    Every alkyne here has a linear angle AT THE THIRD SEED ATOM and one or more collinear
    torsion frames. Neither is addressed by this change: the first is the sixth external DoF
    under a frame convention that stops fixing a frame when atoms 0-1-2 are collinear, the
    second needs a smooth frame construction. A molecule in this list must not be reported as
    a fully free `full` chart on the strength of the transverse pair alone.
    """
    for smiles in UNCOVERED:
        spec, tree, pos = _load(smiles)
        lin = torch.from_numpy(np.asarray(spec.angle_is_linear, dtype=bool))
        frame = torch.from_numpy(np.asarray(spec.torsion_frame_is_linear, dtype=bool))
        uncovered_at_seed = int(lin.sum()) - int(_covered(spec, tree).sum())
        assert uncovered_at_seed > 0 or int(frame.sum()) > 0, (
            f"{smiles} is listed as uncovered but shows no uncovered degeneracy")

    for smiles in COVERED:
        spec, tree, pos = _load(smiles)
        lin = torch.from_numpy(np.asarray(spec.angle_is_linear, dtype=bool))
        assert int(_covered(spec, tree).sum()) == int(lin.sum()) > 0
        assert not bool(torch.from_numpy(
            np.asarray(spec.torsion_frame_is_linear, dtype=bool)).any())


def test_a_seed_row_cannot_be_transverse():
    """The frame seed has no torsion slot, so a pair there would read another atom's phi."""
    spec, tree, pos = _load("CC#C")
    lin = torch.from_numpy(np.asarray(spec.angle_is_linear, dtype=bool))
    r, th, ph = measure(tree, pos)
    with pytest.raises(ValueError, match="no torsion row"):
        build(tree, r, th, ph, transverse=lin)


def test_a_mask_of_the_wrong_length_is_refused():
    spec, tree, pos = _load("CCCCO")
    r, th, ph = measure(tree, pos)
    with pytest.raises(ValueError, match="angle rows"):
        build(tree, r, th, ph, transverse=torch.zeros(3, dtype=torch.bool))


def test_log_jacobian_refuses_a_mask_without_phi():
    """A transverse row's measure reads BOTH components; phi is not optional there.

    Passing the mask to ``build`` but not to ``log_jacobian`` gives a correct geometry under a
    wrong density, and both still return finite numbers of the right shape -- so this has to
    be an error rather than a default.
    """
    spec, tree, pos = _load("CC#N")
    r, th, ph = measure(tree, pos)
    with pytest.raises(ValueError, match="both components"):
        log_jacobian(tree, r, th, transverse=_covered(spec, tree))


def test_the_atom_mask_helper_pairs_theta_and_phi_by_atom():
    """theta and phi live in differently-ordered arrays; the pairing is by placed atom."""
    spec, tree, pos = _load("CCCCO")
    tv = _eligible(spec, tree)
    at = _transverse_atom_mask(tree, tv)
    assert bool(at[tree.angle_index[tv, 2]].all())
    assert int(at.sum()) == int(tv.sum())
    # every flagged atom owns a torsion row, which is what makes the pair well defined
    has_phi = torch.zeros(spec.n_atoms, dtype=torch.bool)
    has_phi[tree.torsion_index[:, 3]] = True
    assert bool((at & ~has_phi).sum() == 0)

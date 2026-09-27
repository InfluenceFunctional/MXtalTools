"""The Z-matrix dummy frame and the sp-root rule: a complete chart through an alkyne.

THE PROBLEM. A dihedral a-b-c-d whose frame runs THROUGH an sp centre b has a, b, c on one
axis, so the plane that defines phi does not exist; and a tree rooted on an sp carbon puts its
two seed neighbours on one axis, so the frame convention fixes no plane at all. Both hold rows
at the linear reference, and at `full` that is a different distribution from the target.

THE FIX, in two parts, each tested here on the property it has to deliver:

  * ``spec_from_graph(avoid_sp_root=True)`` never roots on a carbon with two neighbours, and
    moves the root ONLY when the default pick is one -- every other molecule keeps its tree;
  * ``build``/``measure`` with a ``dummy_frame`` mask measure a flagged row's phi against the
    Z-matrix dummy X = b + m2(P, q, b), at 90 deg to the axis, whose azimuth reference P makes
    a TREE ANGLE (or is the previous dummy of a chain) -- never a derived angle, which can go
    collinear inside the sampler's box.

What must hold: d = 3N-6 with nothing held; measure and build exact inverses; the volume
element UNCHANGED (checked against a central-difference determinant, independently of
autograd); negating every phi and transverse v is the mirror image; and the None path is the
same bytes as before.

What must ALSO hold, and none of those can see: the chart is WELL CONDITIONED. Any X built
from earlier atoms keeps log J, the round trip and the mirror exact, so an X anchored on the
wrong atom passes all of them while the chart goes singular. That is pinned separately, on the
frames read from inside ``build`` and on the build Jacobian's smallest singular value.
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mxtaltools.conformers import build, collate, log_jacobian, measure
from mxtaltools.conformers.builder import dummy_frame_anchor_atoms, dummy_frame_refs
from mxtaltools.conformers.geometry import (dummy_reference, nerf_frame, place_nerf,
                                            transverse_from_polar)
from mxtaltools.conformers.topology import canonical_rank, choose_root, spec_from_graph

DTYPE = torch.float64

#: propyne and but-2-yne have an sp ROOT by default; the tetrayne is a chain of dummies; the
#: epoxide and the two bis-alkynes have a branch point next to the axis -- the last two were
#: the worst in-box frames of the design this replaced (0.42 deg from collinear). The diyne
#: CC#CC#CC is a chain whose on-axis atom is c's own torsion reference, so a chained X
#: anchored on a REAL atom instead of the previous X would be collinear at the pole.
ALKYNES = ["CC#C", "CC#CC", "CCC#CC#CC#C", "C#CC1CO1", "CC(=O)C(C#C)C#C",
           "C#CC(C#C)N1CCC1", "NC(C#C)(C#N)C#N", "CC#CC#CC"]
PLAIN = ["CCCO", "CC#N", "C1CCCCC1O"]
DELTA_R, DELTA_TH = 0.30, 0.50          # the sampler's box half-widths (r in A, angles in rad)


@pytest.fixture(autouse=True)
def _default_dtype():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# --------------------------------------------------------------------------- helpers

def _embed(smiles, seed=0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, p) == 0
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]).T
    return z, bonds, np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)


def _linear(pos, triples):
    u = pos[triples[:, 0]] - pos[triples[:, 1]]
    v = pos[triples[:, 2]] - pos[triples[:, 1]]
    ang = np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u * v).sum(-1)))
    return ang > 175.0


class Chart:
    """The complete chart for one molecule: tree with the root rule, dummy and transverse
    masks chosen the way a caller must (fixed point over the rows X can be built for)."""

    def __init__(self, smiles, avoid_sp_root=True):
        z, bonds, pos = _embed(smiles)
        self.spec = spec = spec_from_graph(z, bonds, pos, use_geometry=False,
                                           avoid_sp_root=avoid_sp_root)
        self.tree = tree = collate([spec])
        self.pos = torch.as_tensor(pos[spec.perm])
        po = pos[spec.perm]
        ti, ai = np.asarray(spec.torsion_index), np.asarray(spec.angle_index)
        self.ang_lin = _linear(po, ai)
        dm = _linear(po, ti[:, :3])
        for _ in range(len(dm) + 1):
            refs, ok = dummy_frame_refs(tree, torch.as_tensor(dm), strict=False)
            anc = refs.anchor.numpy()[ti[:, 3]]
            unch = dm & ~refs.chained.numpy()[ti[:, 3]] & (anc >= 0)
            own = np.maximum(ti[:, 1], anc) - 2
            anc_lin = np.zeros_like(dm)
            anc_lin[unch] = self.ang_lin[own[unch]]
            keep = dm & ok.numpy() & ~anc_lin
            if (keep == dm).all():
                break
            dm = keep
        self.frame_lin = _linear(po, ti[:, :3])
        self.dummy = torch.as_tensor(dm)
        slot_of = {int(a): i for i, a in enumerate(ti[:, 3])}
        partner = np.array([slot_of.get(int(a), -1) for a in ai[:, 2]])
        held = self.frame_lin & ~dm
        tv = self.ang_lin & (partner >= 0)
        tv[partner >= 0] &= ~held[partner[partner >= 0]]
        self.tv = torch.as_tensor(tv)
        self.partner = partner
        self.held = int(held.sum()) + int((self.ang_lin & ~tv).sum())

    def internals(self, pos=None):
        """``(r, th, ph)`` in chart units: (u, v) on transverse rows."""
        return measure(self.tree, self.pos if pos is None else pos,
                       transverse=self.tv, dummy_frame=self.dummy)

    def build(self, r, th, ph):
        return build(self.tree, r, th, ph, transverse=self.tv, dummy_frame=self.dummy)

    def box(self, rng, q0, n):
        """``n`` uniform draws of the sampler's box around the reference ``q0``."""
        r0, t0, p0 = q0
        tvr = self.tv.numpy()
        vr = np.zeros(len(p0), dtype=bool)
        vr[self.partner[tvr]] = True
        dr = torch.as_tensor(rng.uniform(-1, 1, (n, len(r0)))) * DELTA_R
        dt = torch.as_tensor(rng.uniform(-1, 1, (n, len(t0)))) * DELTA_TH
        dp = torch.as_tensor(rng.uniform(-1, 1, (n, len(p0))))
        dp = dp * torch.as_tensor(np.where(vr, DELTA_TH, np.pi))
        return r0 + dr, t0 + dt, p0 + dp

    def pole(self, q0):
        """``q0`` with every transverse bend at its POLE, u = v = 0: every axis exactly linear.

        The point where a frame built from real atoms on the axis is exactly collinear, so
        any construction that leans on one is singular here rather than merely ill-conditioned.
        """
        r0, t0, p0 = (t.clone() for t in q0)
        tv = self.tv.nonzero().flatten()
        t0[tv] = 0.0
        p0[torch.as_tensor(self.partner)[tv]] = 0.0
        return r0, t0, p0


def _canon(ch):
    """The SE(3)-reduced build as a map q -> R^(3N-6), for the volume check."""
    nr, nt = ch.spec.n_atoms - 1, ch.spec.n_atoms - 2

    def f(q):
        pos = ch.build(q[:nr], q[nr:nr + nt], q[nr + nt:])
        return torch.cat([pos[1, :1], pos[2, :2], pos[3:].reshape(-1)])
    return f


def _fd_logdet_residual(ch, q, h=1e-6):
    """log|det| of the build by CENTRAL DIFFERENCES, minus (log_jacobian - seed gauge).

    Independent of autograd on purpose: a dummy frame changes the graph autograd walks, so a
    Jacobian taken through it would share any error in it. The seed gauge 2 log r1 + log r2 +
    log sin(theta2) is what the frame convention consumes (see test_transverse_bending.py).
    """
    f = _canon(ch)
    cols = []
    for i in range(q.numel()):
        e = torch.zeros_like(q)
        e[i] = h
        cols.append((f(q + e) - f(q - e)) / (2 * h))
    _, ld = torch.linalg.slogdet(torch.stack(cols, 1))
    nr, nt = ch.spec.n_atoms - 1, ch.spec.n_atoms - 2
    lj = log_jacobian(ch.tree, q[:nr], q[nr:nr + nt], q[nr + nt:], transverse=ch.tv)[0]
    gauge = 2 * torch.log(q[0]) + torch.log(q[1]) + torch.log(torch.sin(q[nr]))
    return float(ld) - float(lj - gauge)


# --------------------------------------------------------------------------- the frame helper

def test_the_dummy_is_the_z_matrix_line_x_b_1_q_90_p_0():
    """X = b + m2(p, q, b) IS place_nerf(p, q, b, r=1, theta=pi/2, phi=0), up to cos(pi/2)."""
    g = torch.Generator().manual_seed(0)
    p, q, b = (torch.randn(16, 3, generator=g) for _ in range(3))
    x = dummy_reference(p, q, b)
    one = torch.ones(16)
    ref = place_nerf(p, q, b, one, torch.full((16,), np.pi / 2), torch.zeros(16))
    assert float((x - ref).abs().max()) < 1e-15
    bc, n, m2 = nerf_frame(p, q, b)
    # unit, at 90 degrees to the q -> b axis, in the (p, q, b) plane on p's side
    assert float(((x - b).norm(dim=-1) - 1).abs().max()) < 1e-14
    assert float(((x - b) * bc).sum(-1).abs().max()) < 1e-14
    assert float(((x - b) * n).sum(-1).abs().max()) < 1e-14
    assert bool((((x - b) * (p - q)).sum(-1) > 0).all())


# --------------------------------------------------------------------------- the root rule

def _roots(smiles):
    z, bonds, _ = _embed(smiles)
    g = nx.Graph([(int(i), int(j)) for i, j in bonds.T])
    nbrs = [sorted(g.neighbors(i)) for i in range(len(z))]
    rank = canonical_rank(z, nbrs)
    return g, z, choose_root(g, z, rank), choose_root(g, z, rank, avoid_sp_carbon=True)


@pytest.mark.parametrize("smiles", ["CC#C", "CC#CC", "CC#CCO"])
def test_the_root_moves_off_an_sp_carbon(smiles):
    g, z, default, moved = _roots(smiles)
    assert int(z[default]) == 6 and g.degree(default) == 2
    assert moved != default
    assert not (int(z[moved]) == 6 and g.degree(moved) == 2) and g.degree(moved) >= 2
    # to the MOST CENTRAL eligible atom, which is what keeps the round count (and so the
    # number of sequential kernel launches) as low as the default rule's. On CC#CCO the
    # candidates differ: the CH2 has eccentricity 4, the methyl carbon and the O have 5.
    ecc = nx.eccentricity(g)
    cand = [i for i in g.nodes if g.degree(i) >= 2 and not (int(z[i]) == 6 and g.degree(i) == 2)]
    assert ecc[moved] == min(ecc[i] for i in cand)


@pytest.mark.parametrize("smiles", ["CC#N", "CCCO", "C1CCCCC1O", "C#CC1CO1"])
def test_the_root_stays_where_the_default_is_not_an_sp_carbon(smiles):
    """The rule moves ONLY an sp-carbon root, so every other tree is the identical tree."""
    _, _, default, moved = _roots(smiles)
    assert moved == default
    z, bonds, pos = _embed(smiles)
    a = spec_from_graph(z, bonds, pos, use_geometry=False)
    b = spec_from_graph(z, bonds, pos, use_geometry=False, avoid_sp_root=True)
    for f in ("perm", "ref_a", "ref_b", "ref_c", "round_id", "torsion_index"):
        assert np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))), f
    assert b.root_moved_from == -1


@pytest.mark.parametrize("smiles", ["C#C", "N#CC#N", "C#CC#N"])
def test_a_wholly_linear_molecule_keeps_its_root(smiles):
    """No atom of degree >= 2 is off the axis, so nothing can help; the caller must refuse."""
    _, _, default, moved = _roots(smiles)
    assert moved == default


def test_the_rule_is_off_by_default_and_records_what_it_moved():
    z, bonds, pos = _embed("CC#C")
    off = spec_from_graph(z, bonds, pos, use_geometry=False)
    on = spec_from_graph(z, bonds, pos, use_geometry=False, avoid_sp_root=True)
    assert off.root_moved_from == -1
    assert on.root_moved_from == int(off.perm[0]) and int(on.perm[0]) != int(off.perm[0])


# --------------------------------------------------------------------------- the chart

@pytest.mark.parametrize("smiles", ALKYNES)
def test_the_chart_is_complete(smiles):
    """d = 3N-6 with NOTHING held: every linear angle is a transverse pair, every collinear
    frame a dummy row -- and the default root would have left at least one of them held."""
    ch = Chart(smiles)
    assert ch.held == 0, f"{smiles}: {ch.held} row(s) still held"
    assert bool(ch.dummy.any())
    assert int(ch.dummy.sum()) == int(ch.frame_lin.sum())


@pytest.mark.parametrize("smiles", ALKYNES)
def test_build_reproduces_the_reference_and_measure_inverts_it(smiles):
    ch = Chart(smiles)
    q0 = ch.internals()
    p0 = ch.build(*q0)
    d0 = torch.cdist(ch.pos, ch.pos)
    assert float((torch.cdist(p0, p0) - d0).abs().max()) < 1e-12
    rng = np.random.default_rng(0)
    r, th, ph = ch.box(rng, q0, 64)
    for i in range(r.shape[0]):
        q = (r[i], th[i], ph[i])
        back = ch.internals(ch.build(*q))
        for got, want in zip(back, q):
            diff = torch.remainder(got - want + np.pi, 2 * np.pi) - np.pi
            # v rows are not periodic, but |v| < pi in the box, so the wrap is the identity there
            assert float(diff.abs().max()) < 1e-10


@pytest.mark.parametrize("smiles", ALKYNES + PLAIN)
def test_an_unflagged_build_is_untouched(smiles):
    """``dummy_frame=None`` and an all-False mask are the SAME bytes, in build and measure."""
    ch = Chart(smiles, avoid_sp_root=False)
    r, th, ph = measure(ch.tree, ch.pos)
    off = build(ch.tree, r, th, ph)
    allf = build(ch.tree, r, th, ph,
                 dummy_frame=torch.zeros(ch.tree.torsion_index.shape[0], dtype=torch.bool))
    assert torch.equal(off, allf)
    m_off = measure(ch.tree, ch.pos)
    m_allf = measure(ch.tree, ch.pos,
                     dummy_frame=torch.zeros(ch.tree.torsion_index.shape[0], dtype=torch.bool))
    assert all(torch.equal(a, b) for a, b in zip(m_off, m_allf))


@pytest.mark.parametrize("smiles", ALKYNES)
def test_the_volume_element_is_unchanged(smiles):
    """log_jacobian (no dummy argument) against a CENTRAL-DIFFERENCE determinant of the build,
    at the reference, a random box point and a box corner. The critique's own check."""
    ch = Chart(smiles)
    q0 = torch.cat(ch.internals())
    rng = np.random.default_rng(1)
    r, th, ph = ch.box(rng, ch.internals(), 1)
    corner = torch.cat(ch.box(np.random.default_rng(2), ch.internals(), 1), -1)[0]
    nr = ch.spec.n_atoms - 1
    corner = q0 + torch.sign(corner - q0) * torch.cat(
        [torch.full((nr,), DELTA_R), torch.full((q0.numel() - nr,), DELTA_TH)])
    for q in (q0, torch.cat([r[0], th[0], ph[0]]), corner):
        res = _fd_logdet_residual(ch, q)
        assert abs(res) < 1e-7, f"{smiles}: log-det residual {res:.2e}"


@pytest.mark.parametrize("smiles", ALKYNES)
def test_negating_every_phi_and_v_is_the_mirror_image(smiles):
    """X lies along m2, a TRUE vector, so the reflection identity survives the dummy."""
    ch = Chart(smiles)
    r, th, ph = ch.box(np.random.default_rng(3), ch.internals(), 8)
    for i in range(r.shape[0]):
        a = ch.build(r[i], th[i], ph[i])
        b = ch.build(r[i], th[i], -ph[i])
        assert float((b - a * torch.tensor([1.0, 1.0, -1.0])).abs().max()) < 1e-12


def _frames_the_build_used(monkeypatch, ch, q):
    """Build ``q`` and return, in degrees from collinear, the two frame angles of every dummy
    row AS THE BUILDER CONSTRUCTED THEM: ``angle(P, q, b)``, X's azimuth frame, and
    ``angle(X, b, c)``, the frame the row's atom is placed in.

    READ FROM INSIDE ``build``, by recording what ``_dummy_points`` hands ``dummy_reference``,
    not recomputed here from ``dummy_frame_refs``. A recomputation checks the anchor RULE;
    only the recorded arguments check the X that ``build`` and ``measure`` actually use -- and
    any X built from earlier atoms keeps log J, the round trip and the mirror exact, so only
    this and the pole-conditioning test see a construction that anchors on the wrong atom.
    """
    import mxtaltools.conformers.builder as B
    calls = []
    real_points, real_ref = B._dummy_points, B.dummy_reference

    def ref(pp, pq, pb):
        x = real_ref(pp, pq, pb)
        calls[-1].update(p=pp, q=pq, b=pb, x=x)
        return x

    def points(tree, refs, pos, xs, s):
        calls.append(dict(c=tree.ref_c[s]))
        return real_points(tree, refs, pos, xs, s)

    monkeypatch.setattr(B, "dummy_reference", ref)
    monkeypatch.setattr(B, "_dummy_points", points)
    try:
        pos = ch.build(*q)
    finally:
        monkeypatch.setattr(B, "dummy_reference", real_ref)
        monkeypatch.setattr(B, "_dummy_points", real_points)
    assert calls and all("x" in k for k in calls), "build never constructed a dummy"

    def off_axis(u, w):
        ang = torch.rad2deg(torch.atan2(torch.linalg.cross(u, w, dim=-1).norm(dim=-1),
                                        (u * w).sum(-1)))
        return torch.minimum(ang, 180 - ang)
    azimuth = torch.cat([off_axis(k["p"] - k["q"], k["b"] - k["q"]) for k in calls])
    own = torch.cat([off_axis(k["x"] - k["b"], pos[k["c"]] - k["b"]) for k in calls])
    return azimuth, own


@pytest.mark.parametrize("smiles", ALKYNES)
def test_every_dummy_frame_stays_regular_over_the_box(smiles, monkeypatch):
    """Both frames a dummy row touches stay away from collinear on EVERY box draw, and at the
    pole, where every axis is exactly linear.

    X_d's own frame angle(X_d, b, c) is pi/2 +- rho_c. X_d's azimuth frame angle(P_d, q, b)
    is a tree angle (theta0 +- 0.5 rad) or, chained, pi/2 +- rho_b. So neither can approach
    0 or 180 deg in the box -- the design this replaced took P from c's own frame and reached
    0.42 deg on CC(=O)C(C#C)C#C. A floor of 20 deg is loose against both bounds (~23 and
    ~46 deg here) and far above 0.42. A chained X anchored on the real atom on the axis
    instead of the previous X reads 0 deg at the pole.
    """
    ch = Chart(smiles)
    q0 = ch.internals()
    r, th, ph = ch.box(np.random.default_rng(4), q0, 256)
    worst = 180.0
    for q in [ch.pole(q0)] + [(r[i], th[i], ph[i]) for i in range(r.shape[0])]:
        azimuth, own = _frames_the_build_used(monkeypatch, ch, q)
        worst = min(worst, float(azimuth.min()), float(own.min()))
    assert worst > 20.0, f"{smiles}: a dummy frame came within {worst:.2f} deg of collinear"


@pytest.mark.parametrize("smiles", ALKYNES)
def test_the_chart_is_well_conditioned_at_the_pole(smiles):
    """The smallest singular value of the SE(3)-reduced build Jacobian d(canonical)/dq at the
    pole stays O(0.1), i.e. the chart is a regular map there, not merely a finite one.

    Measured 0.28 to 0.44 over these molecules. A chained dummy anchored on the real atom on
    the axis makes it exactly singular (2.7e-17 on CC#CC#CC, with the largest singular value
    7e15), yet leaves log J, the round trip and the mirror identity passing -- so this and the
    frame-angle test above are what pin the anchor choice."""
    ch = Chart(smiles)
    q = torch.cat(ch.pole(ch.internals()))
    s = torch.linalg.svdvals(torch.autograd.functional.jacobian(_canon(ch), q))
    assert float(s.min()) > 0.1, f"{smiles}: sigma_min {float(s.min()):.3g} at the pole"


def test_a_chain_resolves_to_a_real_anchor():
    """Every dummy of a polyyne points, through its chain, at a real atom off the axis."""
    ch = Chart("CCC#CC#CC#C")
    df = dummy_frame_refs(ch.tree, ch.dummy)
    assert bool(df.chained.any())
    real = dummy_frame_anchor_atoms(ch.tree, ch.dummy)
    d = ch.tree.torsion_index[ch.dummy, 3]
    assert bool((real[d] >= 0).all())
    # a chained row inherits its parent's anchor; an unchained one is its own tree-angle partner
    ch_d, un_d = d[df.chained[d]], d[~df.chained[d]]
    assert torch.equal(real[ch_d], real[ch.tree.ref_c[ch_d]])
    assert torch.equal(real[un_d], df.anchor[un_d])


# --------------------------------------------------------------------------- refusals

def test_a_mask_of_the_wrong_length_is_refused():
    ch = Chart("CC#CC")
    r, th, ph = ch.internals()
    with pytest.raises(ValueError, match="aligned with torsion_index"):
        build(ch.tree, r, th, ph, transverse=ch.tv, dummy_frame=torch.zeros(2, dtype=torch.bool))


def test_a_row_rooted_at_the_root_is_refused():
    """With the DEFAULT sp root, a collinear frame through the root has no parent axis: its X
    would be built from a frame that is not the one the chart describes."""
    ch = Chart("CC#C", avoid_sp_root=False)
    r, th, ph = measure(ch.tree, ch.pos)
    with pytest.raises(ValueError, match="no well-defined dummy"):
        build(ch.tree, r, th, ph, dummy_frame=torch.as_tensor(ch.frame_lin))


def test_non_strict_resolution_reports_instead_of_raising():
    ch = Chart("CC#C", avoid_sp_root=False)
    refs, ok = dummy_frame_refs(ch.tree, torch.as_tensor(ch.frame_lin), strict=False)
    assert not bool(ok[torch.as_tensor(ch.frame_lin)].all())


def test_the_transverse_reference_through_a_dummy_matches_the_polar_one():
    """(u0, v0) converted from the polar (theta0, phi0 against X) equals the direct measure."""
    ch = Chart("CC#CC")
    r, th, ph = measure(ch.tree, ch.pos, dummy_frame=ch.dummy)
    _, th_t, ph_t = ch.internals()
    rows = ch.tv.nonzero().flatten()
    u, v = transverse_from_polar(th[rows], ph[torch.as_tensor(ch.partner)[rows]])
    assert float((u - th_t[rows]).abs().max()) < 1e-9
    assert float((v - ph_t[torch.as_tensor(ch.partner)[rows]]).abs().max()) < 1e-9

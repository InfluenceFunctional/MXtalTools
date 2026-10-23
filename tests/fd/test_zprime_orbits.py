"""GATE G1 of the Z' > 1 design (energy_sampling/docs/design/zprime_fd.md section 8; stage Z0): f per labelling.

Units (zprime_fd.md 2.1). A unit is (compound c, fractional centroid t, proper rotation R, conformer q). q is a
sigma-odd stand-in, as in the design's independent verifier: sigma negates its dihedral rows, which are drawn with
|value| in [0.3, 2.8], so q != sigma q. An affine element (W, w) acts on a unit by

    (t, R, q) -> (W t + w, M R D^s, sigma^s q),   s = [det W = -1],   M = T W T^-1,   D = diag(1, 1, -1).

Group H (2.2, 2.3): one n in Gamma (the table's N_E(G) modulo Z^3 and free shifts, improper elements included)
applied to every unit with one common free shift; an independent g_k in G, with its own lattice shift, for every
unit but the anchor; compound-preserving relabellings.

Domain (3.1, 3.3, 3.4): slot 0, the anchor, in ASU(Gamma+) x SO(3) with the K cut and its free coordinates at the
gauge, by ``domains.in_asu`` on the anchor as a Z' = 1 row; every other slot in cctbx's ASU(G)
(``group_table(sg).g_asu``, by ``domains.Region`` with its tie cuts) x SO(3) x Q. Labels kept; no unit ordered.

Count: per crystal and labelling, the number of elements of H (n, every g_k and every lattice shift) that put the
description in the domain, summed over n as [anchor in] x prod_k (number of (g_k, shift) putting unit k in ASU(G)).
Expected f = 2 where Gamma has an improper element and f = 1 in the 22 enantiomorphic-pair groups (checked against
the table). On the first crystals of each case the in-domain descriptions are listed: f x (number of labellings)
distinct points (t, R, q of every slot). On two crystals per fast case the whole orbit is enumerated element by
element and has |Gamma| x |G|^(Z'-1) x (number of labellings) distinct points (2.2), so H acts freely there and the
count runs over each description once.

Negative controls, each of which must break the count: other units in ASU(Gamma+), the anchor's polytope; other
units in ASU(G) u g0.ASU(G) (g0 improper), a fundamental domain of G+ (fd_tooling.md 2.8's residual), 2f in a
non-Sohncke group; the K cut dropped in sg 9 (the anchor tested by its centroid alone), 2f; the improper part of
Gamma dropped, 1.

Fast lane: Z' = 2 in sg 1, 2, 4, 9, 14, 19 and 76; Z' = 3 in sg 14; B.A in sg 14; the controls. Slow: Z' = 2 in all
230 groups; Z' = 3 in G1's 19 groups; B.A, B.A2 and A2.W in its 8 composition groups.
"""
import itertools
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
TOL = 1e-9
PAD = 1e-6
D_BODY = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=F64))
#: the 11 enantiomorphic pairs: Gamma has no improper element, so f = 1
PAIR_GROUPS = (76, 78, 91, 92, 95, 96, 144, 145, 151, 152, 153, 154, 169, 170, 171, 172, 178, 179, 180, 181, 212, 213)
#: sigma per stand-in compound (rows: bond lengths and angles +1, dihedrals -1); 0 = A, 1 = B, 2 = W
SIGMA = {0: (1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1), 1: (1, 1, 1, 1, 1, 1, 1, -1, -1), 2: (1, 1, 1, 1, 1, -1)}
FAST_Z2 = (1, 2, 4, 9, 14, 19, 76)
Z2_SG = [sg if sg in FAST_Z2 else pytest.param(sg, marks=pytest.mark.slow) for sg in range(1, 231)]
Z3_SG = (1, 2, 4, 5, 7, 9, 14, 15, 19, 29, 33, 61, 88, 148, 198, 205, 76, 92, 169)
CO_SG = (1, 2, 4, 9, 14, 19, 33, 76)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(sg):
    return 1 if sg in PAIR_GROUPS else 2


def _labellings(comp):
    """Compound-preserving permutations p of the slots: slot k holds unit p[k]."""
    Z = len(comp)
    return [p for p in itertools.permutations(range(Z)) if all(comp[p[k]] == comp[k] for k in range(Z))]


@lru_cache(maxsize=None)
def _group(sg):
    tab = group_table(sg)
    g_det = torch.round(torch.linalg.det(tab.g_W)).to(torch.long)
    improper = torch.nonzero(g_det < 0)[:, 0]
    return SimpleNamespace(
        sg=sg, free=tab.free, gauge=torch.where(tab.free, tab.trainer_box / 2, torch.zeros(3, dtype=F64)),
        AW=tab.gamma_W, Aw=tab.gamma_w, Adet=tab.gamma_det, GW=tab.g_W, Gw=tab.g_w, Gdet=g_det,
        g0=None if len(improper) == 0 else int(improper[0]), section=domains.translation_section(sg),
        reg_g=domains.Region.from_cuts(tab.g_asu.cuts, tab.g_asu.vertices))


def _cart(T, W):
    """[N, m, 3, 3] M = T W T^-1 per crystal and element, projected onto O(3)."""
    N, m = T.shape[0], W.shape[0]
    return ops.cartesian_rotation(T[:, None].expand(N, m, 3, 3).reshape(-1, 3, 3),
                                  W[None].expand(N, m, 3, 3).reshape(-1, 3, 3)).reshape(N, m, 3, 3)


def _mirror(P, s):
    """P [..., m, 3, 3] with D on the right where s [m] is True."""
    return torch.where(s[:, None, None], P @ D_BODY, P)


def _stand_in(c, n, gen):
    """[n, d] sigma-odd stand-in conformers of compound c."""
    sig = torch.tensor(SIGMA[c], dtype=F64)
    u = torch.rand(n, len(sig), generator=gen, dtype=F64)
    sign = torch.where(torch.rand(n, len(sig), generator=gen) < 0.5, -1.0, 1.0).to(F64)
    return torch.where(sig > 0, 0.9 + 0.7 * u, sign * (0.3 + 2.5 * u))


def _crystals(sg, comp, n, seed, cell_sampler):
    """Random cells [n, 6], centroids [n, Z, 3] in [-1, 2), rotations [n, Z, 3, 3] and stand-ins (one per slot)."""
    gen = torch.Generator().manual_seed(seed)
    cell = torch.tensor(cell_sampler(LATTICE_TYPE[sg], n, seed=seed), dtype=F64)
    t = torch.rand(n, len(comp), 3, generator=gen, dtype=F64) * 3 - 1
    R = so3.random_rotations(n * len(comp), gen).reshape(n, len(comp), 3, 3)
    return cell, t, R, [_stand_in(c, n, gen) for c in comp]


def _translates(region, y, fixed):
    """Lattice translates [..., S, 3] of y [..., 3] that can lie in ``region``; axes in ``fixed`` [3] are not moved."""
    m = torch.where(fixed, 1, torch.floor(region.hi - region.lo + 2 * PAD).to(torch.long) + 1)
    grid = torch.stack(torch.meshgrid(*[torch.arange(int(k), dtype=F64) for k in m], indexing='ij'), -1).reshape(-1, 3)
    base = torch.where(fixed, torch.zeros_like(y), torch.ceil(region.lo - PAD - y))
    return y[..., None, :] + base[..., None, :] + grid


def _anchor_in(grp, cell, y, RA, k_cut):
    """Anchor images, y [N, m, 3] with free coordinates at the gauge and RA [N, m, 3, 3]: per image the number of
    lattice translates in the anchor domain, and the in-domain centroid.

    With the K cut, membership is ``domains.in_asu`` (rows with slack below -PAD are outside and skipped); without
    it, the centroid's cut trees alone (``Region.contains``).
    """
    N, m = y.shape[:2]
    region = grp.section.region
    x = _translates(region, y, grp.free)                                    # [N, m, S, 3]
    S = x.shape[2]
    if k_cut:
        inside = torch.zeros(N * m * S, dtype=torch.bool)
        idx = torch.nonzero((region.slack(x) >= -PAD).reshape(-1))[:, 0]
        if len(idx):
            desc = CrystalDescription.from_parts(grp.sg, cell[idx // (m * S)], x.reshape(-1, 3)[idx],
                                                 RA.reshape(-1, 3, 3)[idx // S])
            inside[idx] = domains.in_asu(desc, grp.section, TOL)[0]
        inside = inside.reshape(N, m, S)
    else:
        inside = region.contains(x, TOL)
    return inside.sum(-1), (inside[..., None] * x).sum(-2)


def _other_in(grp, z, mode):
    """Centroids z [..., 3] of a non-anchor unit: the number of lattice translates in its domain, and the in-domain one.

    ``'g_asu'``: cctbx's ASU(G) (the design). Controls: ``'gamma_plus_asu'``, the anchor's polytope ASU(Gamma+);
    ``'g_plus'``: ASU(G) u g0.ASU(G), g0 an improper element of G, a fundamental domain of G+.
    """
    if mode == 'g_plus':
        c, pick = _other_in(grp, z, 'g_asu')
        if grp.g0 is None:
            return c, pick
        Wi = torch.linalg.inv(grp.GW[grp.g0])
        c0, _ = _other_in(grp, torch.einsum('ij,...j->...i', Wi, z - grp.Gw[grp.g0]), 'g_asu')
        return c + c0, pick
    region = grp.reg_g if mode == 'g_asu' else grp.section.region
    x = _translates(region, z, torch.zeros(3, dtype=torch.bool))
    inside = region.contains(x, TOL)
    return inside.sum(-1), (inside[..., None] * x).sum(-2)


def _n_distinct(points, tol=1e-7):
    """Points [k, d] not within ``tol`` (max-abs) of an earlier point."""
    if not points:
        return 0
    P = torch.stack(points)
    close = (P[:, None] - P[None]).abs().amax(-1) <= tol
    return int((~torch.tril(close, diagonal=-1).any(1)).sum())


def _count(sg, comp, cell, t, R, q, others='g_asu', k_cut=True, improper_gamma=True, listed=0):
    """Per crystal and labelling, the number of elements of H putting the description in the domain.

    t [N, Z, 3], R [N, Z, 3, 3], q one [N, d] per slot. Returns (counts [N, P], distinct in-domain descriptions of
    each of the first ``listed`` crystals, over all labellings).
    """
    grp = _group(sg)
    N = len(t)
    T = ops.cell_matrices(cell)
    keep = torch.arange(len(grp.AW)) if improper_gamma else torch.nonzero(grp.Adet > 0)[:, 0]
    AW, Aw, s_n = grp.AW[keep], grp.Aw[keep], grp.Adet[keep] < 0
    MA, MG = _cart(T, AW), _cart(T, grp.GW)
    s_g = grp.Gdet < 0
    perms = _labellings(comp)
    counts = torch.zeros(N, len(perms), dtype=torch.long)
    found = [[] for _ in range(listed)]
    for j, p in enumerate(perms):
        y = torch.einsum('mij,nj->nmi', AW, t[:, p[0]]) + Aw
        v = torch.where(grp.free, grp.gauge - y, torch.zeros_like(y))       # the common free shift
        RA = _mirror(MA @ R[:, p[0], None], s_n)
        a_cnt, a_x = _anchor_in(grp, cell, y + v, RA, k_cut)                 # [N, m]
        tot = a_cnt
        units = []
        for k in p[1:]:
            x = torch.einsum('mij,nj->nmi', AW, t[:, k]) + Aw + v
            z = torch.einsum('gij,nmj->nmgi', grp.GW, x) + grp.Gw           # [N, m, |G|, 3]
            c, zx = _other_in(grp, z, others)
            tot = tot * c.sum(-1)
            units.append((k, c, zx))
        counts[:, j] = tot.sum(1)
        for r in range(listed):
            for n in torch.nonzero(a_cnt[r])[:, 0].tolist():
                sn = bool(s_n[n])
                head = [a_x[r, n], RA[r, n].reshape(9), q[p[0]][r] * (torch.tensor(SIGMA[comp[0]]) if sn else 1)]
                for gs in itertools.product(*[torch.nonzero(c[r, n])[:, 0].tolist() for _, c, _ in units]):
                    vec = list(head)
                    for (k, c, zx), gi in zip(units, gs):
                        s = sn ^ bool(s_g[gi])
                        Rk = MG[r, gi] @ MA[r, n] @ R[r, k]
                        vec += [zx[r, n, gi], (Rk @ D_BODY if s else Rk).reshape(9),
                                q[k][r] * (torch.tensor(SIGMA[comp[k]]) if s else 1)]
                    found[r].append(torch.cat(vec))
    return counts, [_n_distinct(f) for f in found]


def _orbit_size(sg, comp, cell, t, R, q):
    """Distinct points of one crystal's orbit under H, every element enumerated: n in Gamma, g_k in G for every
    non-anchor slot, every labelling. Lattice shifts are modded out (wrapped coordinates as cos, sin of 2 pi t), the
    anchor's free coordinates sit at the gauge and the others carry the same free shift; points rounded to 1e-6."""
    grp = _group(sg)
    T = ops.cell_matrices(cell[None])
    MA, MG = _cart(T, grp.AW)[0], _cart(T, grp.GW)[0]
    s_n, s_g = grp.Adet < 0, grp.Gdet < 0

    def feats(x, P, qq):
        a = 2 * torch.pi * x
        return torch.cat([torch.cos(a), torch.sin(a), P.flatten(-2), qq], -1)

    points = []
    for p in _labellings(comp):
        y = torch.einsum('mij,j->mi', grp.AW, t[p[0]]) + grp.Aw
        v = torch.where(grp.free, grp.gauge - y, torch.zeros_like(y))
        sig = torch.tensor(SIGMA[comp[0]], dtype=F64)
        cur = feats(y + v, _mirror(MA @ R[p[0]], s_n), torch.where(s_n[:, None], q[p[0]] * sig, q[p[0]]))  # [m, d]
        for k in p[1:]:
            z = torch.einsum('gij,mj->mgi', grp.GW, torch.einsum('mij,j->mi', grp.AW, t[k]) + grp.Aw + v) + grp.Gw
            s = s_n[:, None] ^ s_g[None]                                        # [m, |G|]
            P = MG[None] @ MA[:, None] @ R[k]
            P = torch.where(s[..., None, None], P @ D_BODY, P)
            sig = torch.tensor(SIGMA[comp[k]], dtype=F64)
            u = feats(z, P, torch.where(s[..., None], q[k] * sig, q[k]))           # [m, |G|, d_k]
            shape = cur.shape[:-1] + (u.shape[1],)
            cur = torch.cat([cur[..., None, :].expand(*shape, cur.shape[-1]),
                             u.reshape(u.shape[0], *([1] * (cur.dim() - 2)), u.shape[1], u.shape[2]).expand(
                                 *shape, u.shape[2])], -1)
        points.append(cur.reshape(-1, cur.shape[-1]))
    return len(torch.unique(torch.round(torch.cat(points) * 1e6), dim=0))


def _hist(counts):
    u, c = torch.unique(counts, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def _check(sg, comp, n, listed, seed, cell_sampler):
    cell, t, R, q = _crystals(sg, comp, n, seed, cell_sampler)
    counts, distinct = _count(sg, comp, cell, t, R, q, listed=listed)
    f, P = _f(sg), len(_labellings(comp))
    assert counts.shape == (n, P)
    assert bool((counts == f).all()), f'sg {sg} {comp}: per-labelling counts {_hist(counts)}, expected {f}'
    assert distinct == [f * P] * listed, f'sg {sg} {comp}: distinct in-domain descriptions {distinct}'


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def test_pair_groups_are_the_groups_without_an_improper_normaliser_element():
    """f = 1 exactly in the 22 enantiomorphic-pair groups: the table's Gamma has no det -1 element there only."""
    assert tuple(sg for sg in range(1, 231) if not bool((group_table(sg).gamma_det < 0).any())) == PAIR_GROUPS


@pytest.mark.parametrize('sg', Z2_SG)
def test_z2_one_compound(sg, cell_sampler):
    """Z' = 2, one compound: f per labelling on 40 crystals; 2f distinct in-domain descriptions on 4."""
    _check(sg, (0, 0), 40, 4, 41_000 + sg, cell_sampler)


def test_z3_one_compound_sg14(cell_sampler):
    """Z' = 3 in sg 14: f per labelling, 6 labellings, 12 distinct descriptions per crystal."""
    _check(14, (0, 0, 0), 20, 3, 43_014, cell_sampler)


def test_b_a_sg14(cell_sampler):
    """B.A in sg 14 (B the anchor): one labelling, f = 2 descriptions per crystal."""
    _check(14, (1, 0), 40, 4, 44_014, cell_sampler)


@pytest.mark.parametrize('sg, comp', [(sg, (0, 0)) for sg in FAST_Z2] + [(14, (0, 0, 0)), (14, (1, 0))],
                         ids=[f'sg{sg}-A2' for sg in FAST_Z2] + ['sg14-A3', 'sg14-B.A'])
def test_orbit_size(sg, comp, cell_sampler):
    """The enumerated orbit has |Gamma| x |G|^(Z'-1) x (labellings) distinct points (zprime_fd.md 2.2): H acts
    freely on generic crystals, so the count above runs over the whole orbit without repeats."""
    cell, t, R, q = _crystals(sg, comp, 2, 42_000 + sg + len(comp), cell_sampler)
    grp = _group(sg)
    expected = len(grp.AW) * len(grp.GW) ** (len(comp) - 1) * len(_labellings(comp))
    for i in range(2):
        assert _orbit_size(sg, comp, cell[i], t[i], R[i], [v[i] for v in q]) == expected


@pytest.mark.slow
@pytest.mark.parametrize('sg', Z3_SG)
def test_z3_one_compound(sg, cell_sampler):
    """Z' = 3 in G1's 19 groups: 6f descriptions per crystal (12, or 6 in sg 76, 92, 169)."""
    _check(sg, (0, 0, 0), 20, 2, 43_000 + sg, cell_sampler)


@pytest.mark.slow
@pytest.mark.parametrize('sg', CO_SG)
@pytest.mark.parametrize('comp', [(1, 0), (1, 0, 0), (0, 0, 2)], ids=['B.A', 'B.A2', 'A2.W'])
def test_compositions(sg, comp, cell_sampler):
    """B.A, B.A2 and A2.W (W never the anchor): f, 2f and 2f descriptions per crystal."""
    _check(sg, comp, 30, 2, 45_000 + 10 * sg + len(comp) + comp[0], cell_sampler)


# ---------------------------------------------------------------------------
# negative controls: each must break the count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', (2, 14, 19))
def test_control_others_in_gamma_plus_asu(sg, cell_sampler):
    """Other units in ASU(Gamma+), the anchor's polytope, instead of ASU(G): the count is not f."""
    cell, t, R, q = _crystals(sg, (0, 0), 40, 46_000 + sg, cell_sampler)
    counts, _ = _count(sg, (0, 0), cell, t, R, q, others='gamma_plus_asu')
    assert not bool((counts == _f(sg)).all()), _hist(counts)


@pytest.mark.parametrize('sg', (2, 14))
def test_control_g_plus_residual(sg, cell_sampler):
    """Other units in a fundamental domain of G+ (fd_tooling.md 2.8's residual): 2f per labelling, not f."""
    assert _group(sg).g0 is not None
    cell, t, R, q = _crystals(sg, (0, 0), 40, 47_000 + sg, cell_sampler)
    counts, _ = _count(sg, (0, 0), cell, t, R, q, others='g_plus')
    assert bool((counts == 2 * _f(sg)).all()), _hist(counts)


def test_control_k_dropped_sg9(cell_sampler):
    """sg 9 with the anchor tested by its centroid alone (no K cut): |K| f = 4 per labelling, not 2."""
    assert len(group_table(9).kernel) == 2
    cell, t, R, q = _crystals(9, (0, 0), 40, 48_009, cell_sampler)
    counts, _ = _count(9, (0, 0), cell, t, R, q, k_cut=False)
    assert bool((counts == 4).all()), _hist(counts)


@pytest.mark.parametrize('sg', (1, 14))
def test_control_improper_gamma_dropped(sg, cell_sampler):
    """Gamma restricted to det +1: 1 per labelling, not f = 2 (the improper coset is the anchor's copy)."""
    cell, t, R, q = _crystals(sg, (0, 0), 40, 49_000 + sg, cell_sampler)
    counts, _ = _count(sg, (0, 0), cell, t, R, q, improper_gamma=False)
    assert bool((counts == 1).all()), _hist(counts)

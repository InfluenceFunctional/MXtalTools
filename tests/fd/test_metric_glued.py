"""crystal_fd.metric: GluedFDMetric's internal properties (2026-11-17; gfn_diffusion docs/design/fd_si_draft.md
section 7.3, fix M4). The orthorhombic axis-order faces have their own file, test_metric_axis_order.py.

The reference for a pair of perturbed descriptions of one crystal is their same-basis distance: the plain FDMetric
between the two descriptions as given, in their common unreduced basis. That is a valid description pair, so a
glued distance that finds every nearby image cannot exceed it by more than the formula's basis dependence.

1. Re-description invariance: every description of a crystal (a Gamma+ element, a lattice shift and, in sg 1-15, a
   basis change of the setting group) is at glued distance ~0 from its canonical form, in 7 groups.
2. Symmetry: d(A, B) = d(B, A) exactly.
3. Off the cell faces: in clusters of 0.1% perturbations, glued d never exceeds the same-basis distance, and the
   triangle inequality holds.
4. On the cell faces: the triclinic Niggli faces b = c and a = b read within 5% of the same-basis distance (measured
   up to 3.2%), the monoclinic face beta = 90 deg exactly, and the FREE class's face a = c within 10% (measured
   4.8%). The monoclinic face |c cos beta| = a reads up to 1.5x (sg 14) and 1.3x (sg 4, 15). Both excesses come from
   the cell term's dependence on the basis (fix M2, not built): the bounds pin today's behaviour, and a fix of M2
   should tighten them to ~1.
5. Control: on every face of 4 the unglued FDMetric between the canonical forms reads more than 10x the same-basis
   distance (measured 460-620x), so 4 tests the gluing.
"""
import math

import pytest
import torch

from mxtaltools.crystal_fd import axis_order, ops, so3, tables
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.metric import FDMetric, GluedFDMetric, _setting_candidates, canonical_desc
from mxtaltools.crystal_search.standardize import setting_shift

F64 = torch.float64
RG = 3.0
D2R = math.pi / 180

GENERIC = {2: [6.5, 8.0, 9.5, 100.0, 105.0, 95.0], 4: [9.0, 7.0, 13.0, 90.0, 105.0, 90.0],
           14: [9.0, 7.0, 13.0, 90.0, 105.0, 90.0], 15: [9.0, 7.0, 13.0, 90.0, 105.0, 90.0],
           19: [7.0, 9.5, 12.5, 90.0, 90.0, 90.0], 61: [7.0, 9.5, 12.5, 90.0, 90.0, 90.0],
           148: [12.0, 12.0, 8.0, 90.0, 90.0, 120.0]}


def _cell(c):
    return torch.tensor(c[:3] + [x * D2R for x in c[3:]], dtype=F64)


def _crystals(sg, cell, n, gen):
    t = 0.05 + 0.9 * torch.rand(n, 3, generator=gen, dtype=F64)
    return CrystalDescription.from_parts(sg, cell.expand(n, 6).clone(), t, so3.random_rotations(n, generator=gen))


def _as_dict(desc):
    d = desc.double()
    return dict(cell=d.cell, t=d.t[:, 0], R=d.R[:, 0], h=d.h[:, 0])


def _glued(met, A, B):
    n = len(A['cell'])
    both = {k: torch.cat([A[k], B[k]]) for k in ('cell', 't', 'R', 'h')}
    return met.distance(met.prepare_glued(both), torch.arange(n), torch.arange(n, 2 * n))


def _plain(sg, A, B):
    met = FDMetric(sg, RG)
    n = len(A['cell'])
    P = met.prepare({k: torch.cat([A[k], B[k]]) for k in ('cell', 't', 'R', 'h')})
    return met.pairs(P, torch.arange(n), P, torch.arange(n, 2 * n))


def _basis_changes(sg):
    """(N [k, 3, 3], s [k, 3]) re-descriptions of the setting group: triclinic every unimodular matrix with entries in
    {-1, 0, 1}, monoclinic the setting candidates with entries up to 2, orthorhombic the allowed axis orders."""
    if sg <= 2:
        N = ops._neighbourhood().to(F64)
        return N, torch.zeros(len(N), 3, dtype=F64)
    if sg <= 15:
        Ns, ss = [], []
        for N in _setting_candidates(sg):
            o = setting_shift(N, sg)
            if o is not None:
                Ns.append(torch.as_tensor(N, dtype=F64))
                ss.append(torch.as_tensor(o, dtype=F64))
        return torch.stack(Ns), torch.stack(ss)
    if sg <= 74:
        o = axis_order.axis_orders(sg)
        return o.N.to(F64), o.s.to(F64)
    return torch.eye(3, dtype=F64)[None], torch.zeros(1, 3, dtype=F64)


@pytest.mark.parametrize('sg', sorted(GENERIC))
def test_every_description_is_at_distance_zero(sg):
    gen = torch.Generator().manual_seed(sg)
    n = 24
    A = _crystals(sg, _cell(GENERIC[sg]), n, gen)
    tab = tables.group_table(sg)
    k = torch.randint(len(tab.gamma_plus_W), (n,), generator=gen)
    shift = torch.randint(-1, 2, (n, 3), generator=gen).to(F64)
    B = ops.apply(A, tab.gamma_plus_W.to(F64)[k], tab.gamma_plus_w.to(F64)[k] + shift)
    N, s = _basis_changes(sg)
    j = torch.randint(len(N), (n,), generator=gen)
    B = ops.change_basis(B, N[j], s[j])
    cA, cB = canonical_desc(A), canonical_desc(B)
    assert set(cA['status']) == set(cB['status']) == {'canonical'}
    d = _glued(GluedFDMetric(sg, RG), cA, cB)
    assert float(d.max()) < 1e-6, d


@pytest.mark.parametrize('sg', [2, 14])
def test_distance_is_symmetric(sg):
    gen = torch.Generator().manual_seed(7)
    met = GluedFDMetric(sg, RG)
    c = canonical_desc(_crystals(sg, _cell(GENERIC[sg]), 12, gen))
    P = met.prepare_glued(c)
    ia, ib = torch.triu_indices(12, 12, 1)
    assert torch.equal(met.distance(P, ia, ib), met.distance(P, ib, ia))


def _cluster(sg, cell, free_angles, scale, seed=3, n=12):
    """A crystal and n perturbations of it (relative scale in lengths and free angles, absolute in t and R), all in one
    basis; returns (canonical dicts of the canonical rows, same-basis dicts of the same rows)."""
    gen = torch.Generator().manual_seed(seed)
    base = _crystals(sg, cell, 1, gen)
    cells = cell.expand(n + 1, 6).clone()
    cells[1:, :3] *= 1 + scale * torch.randn(n, 3, generator=gen, dtype=F64)
    if sg >= 75:
        cells[:, 1] = cells[:, 0]
    for a in free_angles:
        cells[1:, 3 + a] += scale * torch.randn(n, generator=gen, dtype=F64)
    t = base.t[:, 0].expand(n + 1, 3) + torch.cat([torch.zeros(1, 3, dtype=F64),
                                                   scale * torch.randn(n, 3, generator=gen, dtype=F64)])
    R = base.R[:, 0].expand(n + 1, 3, 3).clone()
    axis = torch.randn(n, 3, generator=gen, dtype=F64)
    R[1:] = _rotate(axis, scale) @ R[1:]
    desc = CrystalDescription.from_parts(sg, cells, t, R)
    can = canonical_desc(desc)
    keep = torch.tensor([s == 'canonical' for s in can['status']])
    raw = _as_dict(desc)
    return ({k: v[keep] for k, v in can.items() if k in ('cell', 't', 'R', 'h')},
            {k: v[keep] for k, v in raw.items()})


def _rotate(axis, angle):
    """[n, 3, 3] rotations by ``angle`` (rad) about the given axes (Rodrigues)."""
    u = axis / axis.norm(dim=1, keepdim=True)
    K = torch.zeros(len(u), 3, 3, dtype=F64)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -u[:, 2], u[:, 1], -u[:, 0]
    K = K - K.transpose(1, 2)
    return torch.eye(3, dtype=F64) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _pairs_and_triangle(sg, can, raw):
    m = len(can['cell'])
    ia, ib = torch.triu_indices(m, m, 1)
    met = GluedFDMetric(sg, RG)
    d = met.distance(met.prepare_glued(can), ia, ib)
    same = _plain(sg, {k: v[ia] for k, v in raw.items()}, {k: v[ib] for k, v in raw.items()})
    D = torch.zeros(m, m, dtype=F64)
    D[ia, ib] = d
    D[ib, ia] = d
    viol = D[:, None, :] - (D[:, :, None] + D[None, :, :])
    return d, same, float(viol.max())


FREE_ANGLES = {2: [0, 1, 2], 4: [1], 14: [1], 15: [1], 19: [], 61: [], 148: []}


@pytest.mark.parametrize('sg', sorted(GENERIC))
def test_off_faces_glued_never_exceeds_the_same_basis_distance(sg):
    can, raw = _cluster(sg, _cell(GENERIC[sg]), FREE_ANGLES[sg], 1e-3)
    d, same, tri = _pairs_and_triangle(sg, can, raw)
    assert len(d) >= 45
    assert float((d - same).max()) < 1e-9, (d - same).max()
    assert tri < 1e-9


FACES = [  # (name, sg, cell, free angles, bound on glued / same-basis)
    ('triclinic b = c', 2, [6.0, 8.0, 8.0, 80.0, 95.0, 100.0], [0, 1, 2], 1.05),
    ('triclinic a = b', 2, [8.0, 8.0, 10.0, 80.0, 95.0, 100.0], [0, 1, 2], 1.05),
    ('monoclinic beta = 90', 14, [9.0, 7.0, 11.0, 90.0, 90.0, 90.0], [1], 1.0 + 1e-9),
    ('monoclinic a = c, FREE', 4, [9.0, 7.0, 9.0, 90.0, 110.0, 90.0], [1], 1.1),
    ('monoclinic |c cos beta| = a, GLIDE', 14, [9.0, 7.0, 15.0, 90.0, math.acos(-0.6) / D2R, 90.0], [1], 1.6),
    ('monoclinic |c cos beta| = a, CENTRED', 15, [9.0, 7.0, 15.0, 90.0, math.acos(-0.6) / D2R, 90.0], [1], 1.6),
    ('monoclinic |c cos beta| = a, FREE', 4, [9.0, 7.0, 15.0, 90.0, math.acos(-0.6) / D2R, 90.0], [1], 1.6),
]


@pytest.mark.parametrize('name, sg, cell, free, bound', FACES, ids=[f[0] for f in FACES])
def test_on_cell_faces(name, sg, cell, free, bound):
    can, raw = _cluster(sg, _cell(cell), free, 1e-3)
    d, same, tri = _pairs_and_triangle(sg, can, raw)
    ratio = float((d / same).max())
    assert ratio <= bound, (name, ratio)
    assert tri < 1e-9


@pytest.mark.parametrize('name, sg, cell, free, bound', FACES, ids=[f[0] for f in FACES])
def test_control_unglued_metric_fails_on_the_faces(name, sg, cell, free, bound):
    can, raw = _cluster(sg, _cell(cell), free, 1e-3)
    m = len(can['cell'])
    ia, ib = torch.triu_indices(m, m, 1)
    same = _plain(sg, {k: v[ia] for k, v in raw.items()}, {k: v[ib] for k, v in raw.items()})
    unglued = _plain(sg, {k: v[ia] for k, v in can.items()}, {k: v[ib] for k, v in can.items()})
    assert float((unglued / same).max()) > 10.0, name

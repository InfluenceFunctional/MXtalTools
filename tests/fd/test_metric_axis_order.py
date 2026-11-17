"""crystal_fd.metric: GluedFDMetric glues the orthorhombic axis-order faces (2026-11-17).

Two crystals a small step apart across an axis-order face (a = b, b = c, and Pbca's seams a = b, a = c) canonicalise
into different axis orders, so the plain ``FDMetric`` measures them several Angstrom apart. ``GluedFDMetric`` expands
one side over the orders the group allows (``axis_order.axis_orders``) and must read the step itself.

1. Across each face of sg 18 (a <-> b), 19 (all six orders) and 61 (cyclic), the glued distance equals the same-basis
   distance of the step, 2 sqrt(2) delta, within 1%; the plain FDMetric reads more than 1 A (the control).
2. Off the faces, a crystal re-described in every allowed axis order is at glued distance ~0 from itself.
3. A one-order group (sg 33) keeps the identity basis only: glued and plain distances are equal.
"""
import math

import pytest
import torch

from mxtaltools.crystal_fd import axis_order, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.metric import FDMetric, GluedFDMetric, canonical_desc

F64 = torch.float64
RG = 3.0
DELTA = 1e-3
HALF_PI = math.pi / 2


def _crystals(sg, lengths, n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = torch.as_tensor(lengths, dtype=F64).expand(n, 3)
    cell = torch.cat([L, torch.full((n, 3), HALF_PI, dtype=F64)], 1)
    t = 0.05 + 0.9 * torch.rand(n, 3, generator=g, dtype=F64)
    R = so3.random_rotations(n, generator=g)
    return CrystalDescription.from_parts(sg, cell, t, R)


def _distance(met, A, B):
    """d(A[k], B[k]) for canonical dicts A and B of n rows each."""
    n = len(A['cell'])
    both = {k: torch.cat([A[k], B[k]]) for k in ('cell', 't', 'R', 'h')}
    ia, ib = torch.arange(n), torch.arange(n, 2 * n)
    if isinstance(met, GluedFDMetric):
        return met.distance(met.prepare_glued(both), ia, ib)
    P = met.prepare(both)
    return met.pairs(P, ia, P, ib)


#: a cell on each face (Pbca's faces join the shortest axis to another, so its a = c face needs a = c < b)
FACE_CELLS = {(0, 1): [9.0, 9.0, 12.0], (1, 2): [7.0, 10.0, 10.0], (0, 2): [9.0, 11.0, 9.0]}


@pytest.mark.parametrize('sg, face', [(18, (0, 1)), (19, (0, 1)), (19, (1, 2)), (61, (0, 1)), (61, (0, 2))])
def test_step_across_an_axis_order_face(sg, face):
    i, j = face
    base = FACE_CELLS[face]
    lo, hi = list(base), list(base)
    lo[i], lo[j] = base[i] - DELTA, base[j] + DELTA
    hi[i], hi[j] = base[i] + DELTA, base[j] - DELTA
    A = _crystals(sg, lo)
    B = CrystalDescription.from_parts(sg, _crystals(sg, hi).cell, A.t[:, 0], A.R[:, 0])  # same pose, lengths swapped
    cA, cB = canonical_desc(A), canonical_desc(B)
    assert set(cA['status']) == set(cB['status']) == {'canonical'}
    step = 2 * math.sqrt(2) * DELTA  # the same-basis distance: two lengths moved by 2 delta, pose unchanged
    glued = _distance(GluedFDMetric(sg, RG), cA, cB)
    plain = _distance(FDMetric(sg, RG), cA, cB)
    assert float(glued.max()) <= 1.01 * step, (glued, step)
    assert float(glued.min()) >= 0.99 * step
    assert float(plain.min()) > 1.0, plain  # the face is real: unglued, the two sides are far apart


@pytest.mark.parametrize('sg', [18, 19, 61])
def test_every_allowed_order_is_the_same_crystal(sg):
    A = _crystals(sg, [7.0, 9.5, 12.5], seed=1)
    cA = canonical_desc(A)
    orders = axis_order.axis_orders(sg)
    met = GluedFDMetric(sg, RG)
    for N, s in zip(orders.N, orders.s):
        B = ops.change_basis(A, N.to(F64), s)
        d = _distance(met, cA, canonical_desc(B))
        assert float(d.max()) < 1e-6, (sg, N.tolist(), d)


def test_one_order_group_keeps_the_identity_basis():
    met = GluedFDMetric(33, RG)
    assert met.N.shape[0] == 1 and torch.equal(met.N[0], torch.eye(3, dtype=F64))
    A = canonical_desc(_crystals(33, [8.0, 8.0 + DELTA, 12.0], seed=2))
    B = canonical_desc(_crystals(33, [8.0 + DELTA, 8.0, 12.0], seed=2))
    assert torch.allclose(_distance(met, A, B), _distance(FDMetric(33, RG), A, B), atol=1e-12)

"""GATE 4 (fd_tooling.md section 4): descriptions of one crystal have one energy, through MXtalTools' own route.

Route: MolCrystalData built from the description (centroids wrapped to [0, 1) by lattice translation first; the
live builder clips, it does not wrap), then ``analyze(['elj'], cutoff=10, supercell_size=10,
std_orientation=False)`` on CPU: the eLJ call of the trainer's reward, without its other terms.

For random sg 1 and 2 crystals of a chiral and of a planar synthetic molecule (16 crystals each; reduced cells with
lengths 0.85-1.25 times the cube root of Z mol_volume / 0.6, so that the pose matters), the source, each
of its Gamma+ images, its canonical form (fold) and the canonical form of the same crystal in a random basis
(reduce_cell, then fold) must agree within relative 1e-3 or absolute 5e-3 raw eLJ units, whichever is larger.
Reference: float32 build noise is about 1e-5 raw on |eLJ| of 1-100.

Controls: at least 15 of the 16 crystals at 0.1 raw or more from the source (20x the absolute tolerance) and the
median at 0.5 or more (100x). One crystal is allowed below: an energy is one scalar, and a different crystal can
match it by chance (gate 3's literal-atom control holds on every row).
- the wrong gluing: an improper normaliser element applied with its handedness flip withheld (read at +1);
- a seam glued with a twist that is not in Gamma+: t moved by the seam translation (sg 2) or kept (sg 1), R
  turned by 90 degrees about Cartesian x;
- a canonical centroid passed unwrapped (t - 1 on x, rows with t_x >= 0.1): the builder clips it to 0 instead of
  wrapping, moving the molecule by t_x of a cell edge.
"""
import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mxtaltools.crystal_fd import domains, ops
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table
from tests.fd import _helpers as H

F64 = torch.float64
N = 16
RTOL, ATOL, CONTROL, CONTROL_MEDIAN = 1e-3, 5e-3, 0.1, 0.5


def _length_box(sg, mol):
    """Cell lengths around the cube of a packing coefficient near 0.6 (dense enough that a pose matters)."""
    from mxtaltools.dataset_utils.data_classes import MolData
    m = MolData(z=mol.z, pos=mol.pos.float(), x=torch.zeros(len(mol.z), 1), do_mol_analysis=True)
    edge = (len(group_table(sg).g_W) * float(m.mol_volume) / 0.6) ** (1 / 3)
    return torch.full((3,), 0.85 * edge, dtype=F64), torch.full((3,), 1.25 * edge, dtype=F64)


def _safe_crystals(sg, mol, n, seed):
    """Random crystals whose every Gamma+ image, wrapped, stays out of the builder's clip band."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = _length_box(sg, mol)
    out = []
    for _ in range(H.MAX_CHUNKS):
        if sum(o.num_rows for o in out) >= n:
            break
        cells = H.physical_cells(lo, hi, 4 * n, 'latent_v1_lebesgue', g)
        R = torch.as_tensor(Rotation.random(4 * n, random_state=int(torch.randint(1 << 30, (1,), generator=g)))
                            .as_matrix(), dtype=F64)
        d = CrystalDescription.from_parts(sg, cells, torch.rand(4 * n, 3, generator=g, dtype=F64), R)
        imgs, src, _ = domains.images(d)
        bad = torch.zeros(d.num_rows, dtype=torch.bool)
        bad[src[(imgs.t[:, 0] > 1 - 2 * H.CLIP).any(1)]] = True
        out.append(d[~bad])
    assert sum(o.num_rows for o in out) >= n, 'too few crystals clear of the clip band'
    return CrystalDescription.cat(out)[:n]


def _close(a, b):
    return (a - b).abs() <= torch.maximum(torch.full_like(a, ATOL), RTOL * b.abs())


@pytest.fixture(scope='module', params=[(sg, m) for sg in (1, 2) for m in ('chiral', 'planar')],
                ids=lambda p: f'sg{p[0]}-{p[1]}')
def case(request, chiral_molecule, planar_molecule):
    sg, which = request.param
    mol = H.mol_ref(chiral_molecule if which == 'chiral' else planar_molecule)
    d = _safe_crystals(sg, mol, N, 50 + sg)
    return sg, mol, d, H.elj(d, mol)


def test_gamma_plus_images_have_the_source_energy(case):
    sg, mol, d, e0 = case
    imgs, src, _ = domains.images(d)
    assert len(imgs.t) == N * len(group_table(sg).gamma_plus)
    e = H.elj(imgs, mol)
    ok = _close(e, e0[src])
    assert bool(ok.all()), float((e - e0[src]).abs().max())


def test_canonical_forms_have_the_source_energy(case):
    sg, mol, d, e0 = case
    canon = domains.fold(d).desc
    assert bool(_close(H.elj(canon, mol), e0).all())
    rng = np.random.default_rng(60 + sg)
    canon2, _, _ = H.canonicalise(H.scramble(d, rng))
    e2 = H.elj(canon2, mol)
    assert bool(_close(e2, e0).all()), float((e2 - e0).abs().max())


def test_wrong_gluings_change_the_energy(case):
    sg, mol, d, e0 = case
    tab = group_table(sg)
    canon = domains.fold(d).desc
    k = int(torch.nonzero(tab.gamma_det < 0)[0, 0])
    improper = ops.apply(canon, tab.gamma_W[k], tab.gamma_w[k], wrap=True)
    improper = improper.replace(h=torch.ones_like(improper.h))
    rx = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], dtype=F64)
    shift = torch.tensor([0.5, 0.0, 0.0], dtype=F64) if sg == 2 else torch.zeros(3, dtype=F64)
    twisted = canon.replace(t=canon.t + shift, R=rx @ canon.R)
    for name, bad in (('improper read at +1', improper), ('seam with a twist', twisted)):
        _assert_control((H.elj(bad, mol) - e0).abs(), name)


def _assert_control(dE, name):
    assert int((dE >= CONTROL).sum()) >= len(dE) - 1 and float(dE.median()) >= CONTROL_MEDIAN,         f'{name}: |dE| min {float(dE.min()):.3g}, median {float(dE.median()):.3g}, below {CONTROL}: {int((dE < CONTROL).sum())}'


def test_an_unwrapped_centroid_is_clipped_not_wrapped(case):
    sg, mol, d, e0 = case
    if sg == 1:
        pytest.skip('sg 1: every centroid coordinate is a free axis (the clip only translates the crystal)')
    canon = domains.fold(d).desc
    shifted = canon.replace(t=canon.t - torch.tensor([1.0, 0.0, 0.0], dtype=F64))
    assert bool(_close(H.elj(shifted, mol), e0).all())                   # wrapped first: the same crystal
    rows = canon.t[:, 0, 0] >= 0.1                                        # the clip moves the molecule by t_x
    assert int(rows.sum()) >= N // 2
    dE = (H.elj(shifted[rows], mol, wrap=False) - e0[rows]).abs()         # passed as is: clipped to x = 0
    _assert_control(dE, 'unwrapped centroid')


def _skewed(d, seed):
    """The same crystals in random proper bases far from reduced (products of two unimodular matrices with entries in
    [-2, 2]), keeping the rows whose wrapped centroid stays out of the builder's clip band."""
    rng = np.random.default_rng(seed)
    n = d.num_rows
    M = torch.einsum('bij,bjk->bik', H.random_unimodular(n, rng), H.random_unimodular(n, rng))
    s = ops.change_basis(d, M, torch.zeros(n, 3, dtype=F64))
    wrapped = s.t - torch.floor(s.t)
    ok = ~(wrapped > 1 - 2 * H.CLIP).any(-1).any(-1)
    return s[ok], ok


#: skewed-basis rows: the builder makes T_fc from float32 cell parameters in float32, and for a strongly oblique basis
#: that matrix is off by up to ~5e-3 A (measured 2026-10-18: a c-vector 0.44 A high in a 42 A cell), which moves
#: pair distances by ~1e-2 A and eLJ by up to ~0.15 raw. The tolerance below covers that; the old range missed by
#: up to 36 raw (median ~1 raw) on these rows.
SKEW_ATOL, SKEW_RTOL, SKEW_CONTROL = 0.25, 1e-2, 5.0


def _close_skewed(a, b):
    return (a - b).abs() <= torch.maximum(torch.full_like(a, SKEW_ATOL), SKEW_RTOL * b.abs())


def test_skewed_bases_have_the_source_energy_alone_and_batched(case):
    """The builder's lattice-translation range (crystal_building.utils.get_cart_translations) is searched per crystal in
    a reduced basis of its lattice, so a crystal written in an oblique basis gets every neighbour, and its energy does
    not depend on its batch-mates. Before 2026-10-18 the range was one cube per batch in the given basis, bounded
    through the lattice-vector lengths: rows like these scored up to tens of raw units off, by amounts that moved with
    the batch. Alone and batched must agree exactly (same float32 build); both must match the source within the
    float32 cell-matrix limit above."""
    sg, mol, d, e0 = case
    s, ok = _skewed(d, 70 + sg)
    assert s.num_rows >= N - 2
    batched = H.elj(s, mol)
    alone = torch.cat([H.elj(s[i:i + 1], mol) for i in range(s.num_rows)])
    assert float((batched - alone).abs().max()) <= 1e-4, float((batched - alone).abs().max())
    assert bool(_close_skewed(batched, e0[ok]).all()), float((batched - e0[ok]).abs().max())


def test_legacy_range_misses_neighbours_in_skewed_bases(case, monkeypatch):
    """Control: the pre-2026-10-18 range on the same skewed rows loses neighbours: some row is off by SKEW_CONTROL raw
    or more (20x SKEW_ATOL)."""
    import mxtaltools.crystal_building.utils as CBU
    from tests.test_cart_translations import _legacy
    sg, mol, d, e0 = case
    s, ok = _skewed(d, 70 + sg)
    monkeypatch.setattr(CBU, 'get_cart_translations', _legacy)
    dE = (H.elj(s, mol) - e0[ok]).abs()
    assert float(dE.max()) >= SKEW_CONTROL, f'legacy range: largest |dE| {float(dE.max()):.3g} raw; the control is not exercised'

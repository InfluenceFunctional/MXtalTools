"""GATE 4 for space groups 16-230 (2026-10-21): descriptions of one crystal have one energy, through MXtalTools' route.

Crystals: canonical rows drawn from ``CrystalChart.for_length_box`` (u in [0.02, 0.98]) for one group of every Gamma+
point-group type above monoclinic (D2 19, D4 92, D3 146 (R3, hexagonal axes), D6 169, O 198, T 205) and 29, 33, 61,
62; a synthetic chiral molecule; lengths 0.85-1.25 times the cube root of Z mol_volume / 0.6, so that the pose
matters. Rows are built with ``crystal_fd.dataset.to_mol_crystal_data`` (mode 'body', centroids wrapped; crystals with
a variant in the builder's clip band are not drawn) and scored by ``MolCrystalData.analyze(['elj'], cutoff=10,
supercell_size=10, std_orientation=False)`` on CPU. Tolerance: relative 1e-3 or absolute 5e-3 raw eLJ, whichever is
larger (gate 4's; float32 build noise is about 1e-5 raw).

1. Every Gamma+ image of each crystal (``domains.images``, all |Gamma+| elements) has the source's eLJ.
2. Orthorhombic groups: the crystal re-described in every allowed axis order (``axis_order`` (N, s), the cell step's
   basis changes and the Pbca seam gluing) has the source's eLJ.
3. sg 9 on the double cover: the K image (``CrystalChart.k_image``) has the source's eLJ.

Controls (at least N - 1 of N crystals 0.1 raw or more from the source, median 0.5 or more): R turned by 90 degrees
about Cartesian x (not an element of Gamma+); an axis order applied without its origin shift, in groups where it has
one.
"""
import math

import pytest
import torch

from mxtaltools.crystal_fd import axis_order, charts, dataset, domains, ops
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table
from tests.fd import _helpers as H

pytestmark = pytest.mark.slow

F64 = torch.float64
N = 8
RTOL, ATOL, CONTROL, CONTROL_MEDIAN = 1e-3, 5e-3, 0.1, 0.5
GROUPS = (19, 92, 146, 169, 198, 205, 29, 33, 61, 62)
RX = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], dtype=F64)


def _elj(desc, mol, chunk=256):
    """[B] eLJ of descriptions built through dataset.to_mol_crystal_data (CPU, the trainer's eLJ kwargs)."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    out = []
    for lo in range(0, desc.num_rows, chunk):
        batch = collate_data_list(dataset.to_mol_crystal_data(desc[lo:lo + chunk], mol))
        with torch.no_grad():
            out.append(batch.analyze(['elj'], cutoff=10, supercell_size=10, std_orientation=False)['elj'].double())
    return torch.cat(out)


def _length_box(sg, mol):
    from mxtaltools.dataset_utils.data_classes import MolData
    m = MolData(z=mol.z, pos=mol.pos.float(), x=torch.zeros(len(mol.z), 1), do_mol_analysis=True)
    edge = (len(group_table(sg).g_W) * float(m.mol_volume) / 0.6) ** (1 / 3)
    return torch.full((3,), 0.85 * edge, dtype=F64), torch.full((3,), 1.25 * edge, dtype=F64)


def _clip_safe(d):
    """[B] no wrapped centroid coordinate of the rows within twice the builder's clip band of 1."""
    t = d.t - torch.floor(d.t)
    return ~(t > 1 - 2 * H.CLIP).any(-1).any(-1)


def _orders(sg):
    """[(N [3, 3], s [3], has_shift)] every allowed non-identity axis order of an orthorhombic group (none else)."""
    if sg not in axis_order.ORTHORHOMBIC:
        return []
    o = axis_order.axis_orders(sg)
    return [(o.N[j], o.s[j], bool(o.s[j].abs().max() > 0)) for j in range(1, o.k)]


def _variants(d, sg):
    """The descriptions each crystal must score alike: (Gamma+ images with their source rows, axis-order rows)."""
    imgs, src, _ = domains.images(d)
    reordered = [ops.change_basis(d, Nj.expand(d.num_rows, 3, 3), sj.expand(d.num_rows, 3))
                 for Nj, sj, _ in _orders(sg)]
    return imgs, src, reordered


def _crystals(sg, mol, n, seed):
    """n canonical crystals of ``sg`` drawn from its chart, every variant clear of the clip band."""
    ch = charts.CrystalChart.for_length_box(sg, _length_box(sg, mol), 'euler_zyz')
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(H.MAX_CHUNKS):
        if sum(o.num_rows for o in out) >= n:
            break
        y, _ = charts.from_unit(ch, 0.02 + 0.96 * torch.rand(4 * n, ch.ndim, generator=g, dtype=F64))
        d = ch.forward(y).desc
        imgs, src, reordered = _variants(d, sg)
        ok = _clip_safe(d) & _clip_safe(d.replace(R=RX @ d.R))
        bad = torch.zeros(d.num_rows, dtype=torch.bool)
        bad[src[~_clip_safe(imgs)]] = True
        ok &= ~bad
        for (Nj, sj, has_shift), r in zip(_orders(sg), reordered):
            ok &= _clip_safe(r)
            if has_shift:
                ok &= _clip_safe(ops.change_basis(d, Nj.expand(d.num_rows, 3, 3), torch.zeros(d.num_rows, 3, dtype=F64)))
        out.append(d[ok])
    assert sum(o.num_rows for o in out) >= n, 'too few crystals clear of the clip band'
    return CrystalDescription.cat(out)[:n]


def _close(a, b):
    return (a - b).abs() <= torch.maximum(torch.full_like(a, ATOL), RTOL * b.abs())


def _assert_control(dE, name):
    assert int((dE >= CONTROL).sum()) >= len(dE) - 1 and float(dE.median()) >= CONTROL_MEDIAN, \
        f'{name}: |dE| min {float(dE.min()):.3g}, median {float(dE.median()):.3g}'


@pytest.fixture(scope='module', params=GROUPS, ids=lambda sg: f'sg{sg}')
def case(request, chiral_molecule):
    sg = request.param
    mol = H.mol_ref(chiral_molecule)
    d = _crystals(sg, mol, N, 400 + sg)
    return sg, mol, d, _elj(d, mol)


def test_gamma_plus_images_have_the_source_energy(case):
    sg, mol, d, e0 = case
    imgs, src, _ = domains.images(d)
    assert imgs.num_rows == N * len(group_table(sg).gamma_plus)
    e = _elj(imgs, mol)
    ok = _close(e, e0[src])
    assert bool(ok.all()), (int((~ok).sum()), float((e - e0[src]).abs().max()))


def test_axis_orders_have_the_source_energy(case):
    sg, mol, d, e0 = case
    orders = _orders(sg)
    if not orders:
        pytest.skip(f'sg {sg} has one axis order (or is not orthorhombic)')
    assert len(orders) == axis_order.axis_orders(sg).k - 1
    _, _, reordered = _variants(d, sg)
    for (Nj, sj, has_shift), r in zip(orders, reordered):
        e = _elj(r, mol)
        assert bool(_close(e, e0).all()), (Nj.tolist(), float((e - e0).abs().max()))
        if has_shift:                                       # control: the order without its origin shift
            bad = ops.change_basis(d, Nj.expand(N, 3, 3), torch.zeros(N, 3, dtype=F64))
            _assert_control((_elj(bad, mol) - e0).abs(), f'order {Nj.tolist()} without its shift')


def test_a_rotation_outside_gamma_plus_changes_the_energy(case):
    sg, mol, d, e0 = case
    _assert_control((_elj(d.replace(R=RX @ d.R), mol) - e0).abs(), 'R turned 90 deg about x')


def test_double_cover_k_image_has_the_source_energy(chiral_molecule):
    sg = 9
    mol = H.mol_ref(chiral_molecule)
    ch = charts.CrystalChart.for_length_box(sg, _length_box(sg, mol), 'euler_zyz', orientation_cover='double')
    g = torch.Generator().manual_seed(409)
    y, _ = charts.from_unit(ch, 0.02 + 0.96 * torch.rand(8 * N, ch.ndim, generator=g, dtype=F64))
    d = ch.forward(y).desc
    k = ch.k_image(d)
    ok = _clip_safe(d) & _clip_safe(k) & _clip_safe(d.replace(R=RX @ d.R))
    d, k = d[ok][:N], k[ok][:N]
    assert d.num_rows == N
    e0, ek = _elj(d, mol), _elj(k, mol)
    assert bool(_close(ek, e0).all()), float((ek - e0).abs().max())
    _assert_control((_elj(d.replace(R=RX @ d.R), mol) - e0).abs(), 'sg 9: R turned 90 deg about x')
    assert math.isclose(ch.log_multiplicity, math.log(2))

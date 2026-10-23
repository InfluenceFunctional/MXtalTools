"""GATE 8 (fd_tooling.md section 4), MXtalTools side: the trainer-latent chart is the live map, bit for bit.

1. ``LatentV1Chart.cell_params`` / ``describe`` equal ``MolCrystalData.latent_to_cell_params`` on a real batch
   (torch.equal), clamp included, for latents inside and outside the clamp box, two molecules of different radius,
   and sg 1, 2, 14 and 92 (the map is group-generic; its log-Jacobian is built for sg 1 and 2). The returned mask
   is exactly the rows the clamp changes, against the clamp bounds as documented in ``latent_to_cell_params``.
2. ``LatentV1Chart.latents`` (and ``inverse``) equals ``MolCrystalData.latent_params()`` (free-axis gauge,
   transform, clip) bit for bit after the same clip.
3. ``CellChart``'s image is the margin-0 zero set of the live ``cell_reduction_penalty`` within the trainer's length
   box: every chart point is on it, every cell on it charts inside the chart's bounds, and every cell off it charts
   outside them.
4. ``log_reference_density`` for ``LatentV1Chart`` with ``'latent_v1_lebesgue'`` equals
   2 log sin(r/2) + log sin(theta) + log V (GFN's ``-jacobian_energy / T`` up to log sym_mult), read off the live
   batch, up to one constant per space group, and that constant is the one the latent ranges documented in
   ``latent_transform`` predict (derived here, not read from the chart). (GFN's own function is pinned in GFN
   ``tests/crystal/test_fd_measure_contract.py``.)
5. ``cell_measure`` is required; ``log_multiplicity`` counts images per crystal.
"""
import math

import pytest
import torch

from mxtaltools.common.geometry_utils import lat2sph_rotvec
from mxtaltools.common.sym_utils import cell_reduction_penalty
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.crystal_fd import charts, measure
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from tests.fd import _helpers as H

F64 = torch.float64
#: the clamp as documented in MolCrystalOps.latent_to_cell_params (Z' = 1 layout)
DOC_LO = torch.tensor([-0.99] * 3 + [-1.0] * 6 + [-0.99, -1.0, -0.99])
DOC_HI = torch.tensor([1 - 1e-4] * 2 + [1.0] * 10)


def _molecules(chiral_molecule, planar_molecule):
    return [MolData(z=torch.as_tensor(m['z']), pos=torch.as_tensor(m['pos'], dtype=torch.float32),
                    x=torch.zeros(len(m['z']), 1), do_mol_analysis=True) for m in (chiral_molecule, planar_molecule)]


def _batch(mols, sg, B, params=None):
    """A real MolCrystalData batch, molecules alternating; cell parameters from ``params`` [B, 12] or placeholders."""
    rows = []
    for i in range(B):
        p = params[i] if params is not None else torch.tensor([6., 7, 8, 1.6, 1.7, 1.8, .3, .3, .3, .2, .3, .4])
        rows.append(MolCrystalData(molecule=mols[i % 2].clone(), sg_ind=int(sg), cell_lengths=p[:3],
                                   cell_angles=p[3:6], aunit_centroid=p[6:9], aunit_orientation=p[9:12],
                                   aunit_handedness=torch.tensor([1.0]), do_box_analysis=True))
    return collate_data_list(rows)


@pytest.mark.parametrize('sg', [1, 2, 14, 92])
def test_forward_is_the_live_map_bit_for_bit(sg, chiral_molecule, planar_molecule):
    mols = _molecules(chiral_molecule, planar_molecule)
    B = 2000
    g = torch.Generator().manual_seed(sg)
    y = (-1.3 + 2.6 * torch.rand(B, 12, generator=g)).float()
    y[:B // 2] = y[:B // 2].clamp(-0.98, 0.98)                    # half the rows inside the clamp box
    batch = _batch(mols, sg, B)
    ch = charts.LatentV1Chart(sg, batch.radius)
    lo, hi = ch.clamp_bounds()
    assert torch.equal(lo, DOC_LO.to(lo.dtype)) and torch.equal(hi, DOC_HI.to(hi.dtype))
    params, clamped, _ = ch.cell_params(y)
    live = batch.clone()
    live.latent_to_cell_params(y)
    for name, sl in (('cell_lengths', slice(0, 3)), ('cell_angles', slice(3, 6)), ('aunit_centroid', slice(6, 9)),
                     ('aunit_orientation', slice(9, 12))):
        assert torch.equal(params[:, sl], getattr(live, name)), name
    assert torch.equal(clamped, ((y < DOC_LO) | (y > DOC_HI)).any(1))
    assert not bool(clamped[:B // 2].any()) and bool(clamped[B // 2:].any())
    desc, mask = ch.describe(y)
    assert torch.equal(mask, clamped)
    assert torch.equal(desc.cell.float(), torch.cat([live.cell_lengths, live.cell_angles], 1))
    assert torch.equal(desc.t[:, 0].float(), live.aunit_centroid)
    assert bool((desc.h == 1).all())
    if sg in (1, 2):
        m = ch.forward(y)
        assert torch.equal(m.clamped, clamped) and torch.equal(m.desc.cell, desc.cell)
    else:
        with pytest.raises(NotImplementedError) as e:
            ch.forward(y)
        # sg >= 16: refused for good (the trainer's latent there is to be replaced; no averaged-row Jacobian)
        assert (str(e.value) == charts.LATENT_ABOVE_MONOCLINIC) == (sg >= 16)


@pytest.mark.parametrize('sg', [1, 2, 14, 92])
def test_inverse_is_latent_params_bit_for_bit(sg, chiral_molecule, planar_molecule):
    mols = _molecules(chiral_molecule, planar_molecule)
    B = 1000
    g = torch.Generator().manual_seed(10 + sg)
    y = (-0.97 + 1.94 * torch.rand(B, 12, generator=g)).float()
    batch = _batch(mols, sg, B)
    ch = charts.LatentV1Chart(sg, batch.radius)
    desc, _ = ch.describe(y)
    L, A, c, rv, h = desc.crystal_params()
    live = _batch(mols, sg, B, torch.cat([L, A, c, rv], 1).float())
    ref = live.latent_params()
    lat = ch.latents(desc)
    assert torch.equal(lat.clamp(-1, 1), ref)
    if sg in (1, 2):
        assert torch.equal(ch.inverse(desc)[0], lat)


@pytest.mark.parametrize('radius', [1.7561, 1.1])
@pytest.mark.parametrize('sg', [1, 2])
def test_cell_chart_image_is_the_zero_set_in_the_length_box(sg, radius):
    lo, hi = charts.latent_v1_length_box(sg, radius)
    ch = charts.CellChart('triclinic', (lo, hi))
    g = torch.Generator().manual_seed(int(100 * radius) + sg)
    # chart -> cells: on the zero set (penalty exactly 0 at margin 0) and in the box, edges of every range included
    u = torch.rand(100_000, 6, generator=g, dtype=F64)
    u[:6000] = torch.where(torch.rand(6000, 6, generator=g) < 0.3, torch.randint(0, 2, (6000, 6), generator=g).to(F64),
                           u[:6000])
    y, _ = charts.from_unit(ch, u)
    cells, _ = ch.map(y)
    pen = cell_reduction_penalty(cells[:, 3:], cells[:, :3], torch.full((len(cells),), sg), margin=0.0)
    assert float(pen.max()) <= 1e-20, float(pen.max())
    rel = 1e-12
    assert bool(((cells[:, :3] >= lo * (1 - rel)) & (cells[:, :3] <= hi * (1 + rel))).all())
    # cells -> chart: uniform in (log L, angle) on a box wider than the domain
    n = 400_000
    v = torch.rand(n, 6, generator=g, dtype=F64)
    logL = torch.log(lo / 2) + v[:, :3] * (torch.log(2 * hi) - torch.log(lo / 2))
    ang = torch.deg2rad(torch.tensor(50.0, dtype=F64) + 80 * v[:, 3:])
    cand = torch.cat([torch.exp(logL), ang], 1)
    ok = measure.cell_volume(cand).isfinite() & (measure.cell_volume(cand) > 0)
    cand = cand[ok]
    on = (cell_reduction_penalty(cand[:, 3:], cand[:, :3], torch.full((len(cand),), sg), margin=0.0) == 0) \
        & H.in_box(cand, lo, hi)
    assert int(on.sum()) > 100
    yc, _ = ch.unmap(cand)
    inside = torch.ones(len(cand), dtype=torch.bool)
    for i in range(6):
        blo, bhi = ch.bounds(yc[:, :i], i)
        inside &= (yc[:, i] >= blo - 1e-12) & (yc[:, i] <= bhi + 1e-12)
    assert torch.equal(inside, on), (int((inside & ~on).sum()), int((on & ~inside).sum()))


def _live_log_density(batch):
    """2 log sin(r/2) + log sin(theta) + log V, from the batch's own latent_params (GFN compute_jacobian's terms)."""
    lat = batch.clone().latent_params()[:, -3:]
    sph = lat2sph_rotvec(lat, 1).double()
    return 2 * torch.log(torch.sin(sph[:, 2] / 2)) + torch.log(torch.sin(sph[:, 0])) + torch.log(batch.cell_volume.double())


@pytest.mark.parametrize('sg', [1, 2])
def test_reference_density_is_the_trainer_measure(sg, chiral_molecule, planar_molecule):
    mols = _molecules(chiral_molecule, planar_molecule)
    B = 2000
    g = torch.Generator().manual_seed(20 + sg)
    y = (-0.99 + 1.98 * torch.rand(B, 12, generator=g)).float()
    y[:200, 9] = -0.99                                             # theta at the floor (pi / 400)
    y[200:400, 11] = -0.99                                         # r at the floor (pi / 100)
    batch = _batch(mols, sg, B)
    batch.latent_to_cell_params(y)
    ch = charts.LatentV1Chart(sg, batch.radius)
    desc = CrystalDescription.from_crystal_params(batch.sg_ind, batch.cell_lengths, batch.cell_angles,
                                                  batch.aunit_centroid, batch.aunit_orientation)
    ours = measure.log_reference_density(desc, ch, 'latent_v1_lebesgue')
    valid = torch.isfinite(measure.cell_volume(desc.cell))         # angle triples with a real cell (Gram > 0)
    assert int(valid.sum()) > B // 2
    diff = (ours - _live_log_density(batch))[valid]
    assert float(diff.max() - diff.min()) <= 1e-4, float(diff.max() - diff.min())
    # the constant, from the documented latent ranges (not from the chart): minus the live rows' log slopes, plus
    # the Haar normalisation log(2 pi^2)
    const = -documented_log_slopes(sg) + math.log(2 * math.pi ** 2)
    assert abs(float(diff.mean()) + const) <= 1e-4
    assert abs(ch._log_slopes(sg) - documented_log_slopes(sg)) <= 1e-6       # probed in float64 from float32 constants


def documented_log_slopes(sg: int) -> float:
    """Sum of log |d(natural coordinate) / d(latent)| over the live rows, from the ranges documented in
    MolCrystalOps.latent_transform and lat2sph_rotvec: each log-length spans log 40 over 2 latent units, each angle
    0.6 pi, each non-free centroid ASYM_UNITS[sg], theta pi / 2, phi 2 pi, r 2 pi (P1: every centroid axis is free)."""
    box = [float(v) for v in ASYM_UNITS[str(sg)]]
    free = {1: (True,) * 3, 2: (False,) * 3}[sg]
    return (3 * math.log(math.log(40) / 2) + 3 * math.log(0.3 * math.pi)
            + sum(math.log(b / 2) for b, f in zip(box, free) if not f)
            + math.log(math.pi / 4) + 2 * math.log(math.pi))


def test_cell_measure_is_required_and_multiplicities():
    ch = charts.CrystalChart.for_latent_v1(2, 1.7561, 'euler_zyz')
    d = ch.forward(charts.from_unit(ch, torch.rand(4, ch.ndim, dtype=F64))[0]).desc
    with pytest.raises(TypeError):
        measure.log_reference_density(d, ch)                       # no default
    with pytest.raises(ValueError):
        measure.log_reference_density(d, ch, 'lebesgue')
    with pytest.raises(TypeError):
        measure.log_reference_density(d, ch.centroid, 'latent_v1_lebesgue')
    assert measure.log_multiplicity(2, 'fundamental', 'latent_v1_box') == pytest.approx(math.log(4), abs=1e-15)
    assert measure.log_multiplicity(2, 'fundamental', 'unit_cell') == pytest.approx(math.log(8), abs=1e-15)
    assert measure.log_multiplicity(1, 'fundamental', 'latent_v1_box') == 0.0
    assert measure.log_multiplicity(2, 'latent_v1_box', 'fundamental') == pytest.approx(-math.log(4), abs=1e-15)
    with pytest.raises(ValueError):
        measure.log_multiplicity(14, 'fundamental', 'latent_v1_box')   # the sg 14 box covers half the crystals
    # metric_invariant minus latent_v1_lebesgue is log(8 (abc)^4 prod sin / V)
    a = measure.log_reference_density(d, ch, 'metric_invariant') - measure.log_reference_density(d, ch, 'latent_v1_lebesgue')
    c = d.cell
    ref = math.log(8) + 4 * torch.log(c[:, :3]).sum(1) + torch.log(torch.sin(c[:, 3:])).sum(1) \
        - torch.log(measure.cell_volume(c))
    assert (a - ref).abs().max() <= 1e-12

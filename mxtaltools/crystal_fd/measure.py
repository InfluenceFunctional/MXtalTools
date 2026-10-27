"""Reference measures on Z' = 1 crystal descriptions, and image multiplicities between domains.

The reference measure is Haar(dR) x (V d^k t) x mu_cell(dc): normalised Haar
measure on the orientation, Lebesgue measure on the non-free fractional
centroid coordinates times the cell volume V (free coordinates integrate to 1
over their period), and a cell measure chosen explicitly, with no default:

``'latent_v1_lebesgue'``
    Flat in each free log length and in each free angle of the cell's family, in every
    family: d(log a) d(log b) d(log c) d(alpha) d(beta) d(gamma) triclinic; d(log a)
    d(log b) d(log c) d(beta) monoclinic (alpha and gamma fixed); d(log a) d(log b)
    d(log c) orthorhombic; d(log a) d(log c) tetragonal and hexagonal (trigonal P and R
    groups on hexagonal axes included; b = a); d(log a) cubic (b = c = a). A length or
    angle the family fixes carries no factor. For sg 1-15 it is the measure the
    trainer's latent chart puts on the cell today (no cell term in GFN's
    ``compute_jacobian``); from orthorhombic up it is not (the trainer leaves the
    orthorhombic axis order free and, from tetragonal up, averages length rows), and the
    trainer is to change to it, not this package to the trainer (owner, 2026-10-21).
    The name is kept. Not basis-invariant: restricted to a different reduced domain it
    reweights lattices.
``'metric_invariant'``
    dG / V, G the Gram matrix: invariant under every basis change; relative to the
    measure above its density is 8 (abc)^4 sin(alpha) sin(beta) sin(gamma) / V. Built
    for triclinic groups only.

Multiplicities (``images_per_crystal``, ``log_multiplicity``) are hard-wall counts of
the descriptions of one crystal a domain holds; a ``CrystalChart`` names its own domain
(``CrystalChart.domain``: 'fundamental', sg 6-9 included through the K cut of
``charts.KCutHemisphereChart``, or 'double_cover' for sg 6-9 with
``orientation_cover='double'``).
"""
import math

import torch

from mxtaltools.crystal_fd import axis_order
from mxtaltools.crystal_fd.charts import CrystalChart, LatentV1Chart
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
CELL_MEASURES = ('latent_v1_lebesgue', 'metric_invariant')
DOMAINS = ('fundamental', 'double_cover', 'latent_v1_box', 'unit_cell')


def cell_volume(cell: torch.Tensor) -> torch.Tensor:
    """[B, 6] (a, b, c Angstrom; angles radians) -> [B] volume (Angstrom^3), float64."""
    c = torch.as_tensor(cell, dtype=F64)
    ca, cb, cg = torch.cos(c[:, 3]), torch.cos(c[:, 4]), torch.cos(c[:, 5])
    return c[:, 0] * c[:, 1] * c[:, 2] * torch.sqrt(1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg)


def log_cell_density(cell: torch.Tensor, cell_measure: str) -> torch.Tensor:
    """[B] log-density of ``cell_measure`` relative to the family's d(free log lengths) d(free angles).

    ``'latent_v1_lebesgue'`` is that measure (0 in every family); ``'metric_invariant'`` is computed from all six
    parameters and is meaningful for triclinic cells only (``log_reference_density`` refuses it elsewhere)."""
    _check_measure(cell_measure)
    c = torch.as_tensor(cell, dtype=F64)
    if cell_measure == 'latent_v1_lebesgue':
        return torch.zeros(c.shape[0], dtype=F64, device=c.device)
    return (math.log(8) + 4 * torch.log(c[:, :3]).sum(1) + torch.log(torch.sin(c[:, 3:])).sum(1)
            - torch.log(cell_volume(c)))


def _check_measure(cell_measure):
    if cell_measure not in CELL_MEASURES:
        raise ValueError(f'cell_measure must be one of {CELL_MEASURES} (it has no default), not {cell_measure!r}')


def log_reference_density(desc: CrystalDescription, chart, cell_measure: str) -> torch.Tensor:
    """[B] log-density of the reference measure relative to Lebesgue measure on ``chart``'s coordinates of ``desc``.

    Parameters
    ----------
    desc : CrystalDescription
        Z' = 1 rows in the chart's domain (``CrystalChart``) or on the +1 sheet
        (``LatentV1Chart``).
    chart : CrystalChart or LatentV1Chart
        A chart of the whole description (cell, centroid and orientation).
    cell_measure : str
        ``'latent_v1_lebesgue'`` or ``'metric_invariant'``; required.

    Notes
    -----
    For ``LatentV1Chart`` with ``'latent_v1_lebesgue'`` this is GFN's
    ``-jacobian_energy / T`` (``MolecularCrystal.compute_jacobian``:
    2 log sin(r/2) + log sin(theta) + Z' log(V / sym_mult)) up to a constant per
    space group (sg 1 and 2; the chart refuses other groups).

    For a ``CrystalChart`` of any family it is the chart's log-Jacobian (cell, centroid and
    orientation blocks) + Z' log V + ``log_cell_density`` (0 for ``'latent_v1_lebesgue'``);
    from orthorhombic up the cell block's log-Jacobian is 0 as well, the coordinates being
    the free log lengths. On sg 6-9 the K cut (``KCutHemisphereChart``, signed r) has the
    hemisphere Haar density, even in r, and integrates to the fundamental domain's mass (its
    orientation block to 1/2); on the double cover the density is the same pointwise and
    its integral over the chart is ``chart.multiplicity`` (2) times the fundamental
    domain's.
    """
    _check_measure(cell_measure)
    if not isinstance(chart, (CrystalChart, LatentV1Chart)):
        raise TypeError('log_reference_density needs a chart of the whole description (CrystalChart or LatentV1Chart)')
    d = desc.double()
    if cell_measure == 'metric_invariant' and not bool(((d.sg == 1) | (d.sg == 2)).all()):
        raise NotImplementedError("the 'metric_invariant' cell measure is built for triclinic groups only (stage 0)")
    _, ld_inv = chart.inverse(d)
    ld_inv = ld_inv.to(F64)
    V = cell_volume(d.cell)
    return -ld_inv + d.z_prime.to(F64) * torch.log(V) + log_cell_density(d.cell, cell_measure)


def images_per_crystal(sg: int, domain: str) -> int:
    """Descriptions per crystal that ``domain`` holds, for a chiral molecule on the +1 sheet (hard walls).

    Every domain holds each crystal's canonical cell only (sg 16-74: in its canonical axis order);
    ``'latent_v1_box'`` holds the trainer's cell, which is that cell in the hard-wall limit except in the orthorhombic
    groups refused below.

    ``'fundamental'``: 1 (``CrystalChart``'s image; sg 6-9: ASU(Gamma+) x the K cut of SO(3),
    ``charts.KCutHemisphereChart``). ``'double_cover'``: ASU(Gamma+) x SO(3)
    without the K cut, |K| = 2, the image of ``CrystalChart(..., orientation_cover='double')``; defined for sg 6-9
    only (elsewhere K is trivial: ValueError). ``'unit_cell'``: [0, 1)^k (non-free coordinates) x SO(3),
    |Gamma+| (counted modulo Z^3 and free shifts). ``'latent_v1_box'``: the trainer's box (``ASYM_UNITS``) with the
    trainer's cell, k_box; refused where the box does not reach every crystal (coverage < 1), since its count is then
    not a multiple of the fundamental domain's, and in the orthorhombic groups with more than one allowed axis order
    (``axis_order``): the trainer fixes no axis order, so its latent holds every allowed order whose lengths fit its
    length box (not the same box on every axis), between 1 and k orders depending on the cell.
    """
    if domain not in DOMAINS:
        raise ValueError(f'domain must be one of {DOMAINS}, not {domain!r}')
    tab = group_table(sg)
    if domain == 'fundamental':
        return 1
    if domain == 'double_cover':
        if len(tab.kernel) == 1:
            raise ValueError(f"sg {sg}: K is trivial, so there is no double cover (it is defined for sg 6-9, "
                             "CrystalChart orientation_cover='double')")
        return int(len(tab.kernel))
    if domain == 'unit_cell':
        return int(len(tab.gamma_plus))
    if tab.box_coverage != 1 or tab.k_box is None:
        raise ValueError(f'sg {sg}: the trainer box covers {tab.box_coverage} of the crystals; '
                         'its image count is not a constant multiple')
    if int(sg) in axis_order.ORTHORHOMBIC and axis_order.axis_orders(sg).k > 1:
        raise ValueError(f'sg {sg}: the trainer fixes no orthorhombic axis order, so its latent holds between 1 and '
                         f'{axis_order.axis_orders(sg).k} axis orders of a crystal, depending on the cell and the '
                         'length box; its image count is not a constant multiple')
    return int(tab.k_box)


def log_multiplicity(sg: int, src: str, dst: str) -> float:
    """log k with log Z_dst = log Z_src + log k: the ratio of images per crystal of two domains of ``sg``.

    Valid in the hard-wall limit only (a soft wall admits extra mass past it).
    """
    return math.log(images_per_crystal(sg, dst)) - math.log(images_per_crystal(sg, src))

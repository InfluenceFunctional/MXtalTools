"""Stage 5 of the Z'>1 units plan: the cluster reach is the all-atom reach, per unit, about the cluster's own molecule 0
(gfn_diffusion energy_sampling/docs/design/crystal_units_audit.md, items #11, #16; section 5, stage 5).

The contract: ``mol2cluster`` + ``construct_radial_graph`` find every intermolecular pair within the cutoff,
hydrogens included, at Z'=1 and for every unit of a Z'>1 crystal; the cluster of a row does not depend on its
batch-mates or pad slots; and ``crystal_building.image_pairs`` keeps the same images.

What is pinned here, on synthetic P1 crystals built so that a wrong rule visibly fails. The molecule (``_spiky``) has
three heavy atoms and six hydrogens: two on the x axis at +-hx, which set its reach and make the contacts, and four
shorter ones on y and z, the +y one reaching ``lop`` further, so that its all-atom centroid lies about 0.27 A from its
heavy-atom centroid (the point pose_aunit places at the slot) while the reach bound stays tight along x. The decisive
contact sits 0.01 A inside the cutoff, 50 times the oracle's 2e-4 boundary band:

- ``molecule_reach`` is the largest distance from the all-atom centroid to an atom;
- the rule itself: a molecule of the cluster is unflagged (aux_ind < 2) exactly when its all-atom centroid lies within
  ``cutoff + zp_buffer + 2 R + REACH_MARGIN`` of molecule 0's, and every unit-cell image within that distance is
  built (an independent enumeration);
- Z'=1, hydrogens past the heavy-atom radius: the contact lies beyond ``cutoff + 2 * radius + 0.1`` (the reach
  before 2026-10-23, which missed it), and the pair set equals the float64 brute-force lattice sum;
- a centroid component at 1.0, which the unit cell clips to 1 - 1e-4: the pair set still equals brute force taken
  around the cluster's own molecule 0 (the stage-5 review found the reach centred on the unclipped pose, 1.5e-3 A
  off in a 15 A cell, dropping a pair 3e-4 A inside the cutoff);
- Z'=2 and Z'=3 rows whose units have unequal reaches and unequal centroid offsets, the far unit between the near
  unit and its lattice image: the pair sets equal brute force;
- sensitivity controls: with the Z'>1 buffer replaced by the symmetric guess ``max_u |c_u - c_v|``, or with
  ``REACH_MARGIN`` set to -0.05 A, the comparisons FAIL;
- ``MolCrystalBuilding._unit_reach_buffer`` equals its definition, max_u (R_u + |c_u - c_v|) - R_v over each unit's
  molecule 0 in its unit cell, is exactly 0 for a Z'=1 row, and reads neither pad slots nor batch-mates;
- ``image_pairs`` gives the builder's pairs on the Z'=1 crystal (its fixture in tests/test_image_pairs.py has no
  hydrogens, so it cannot see the reach);
- ``get_cart_translations(rho=...)`` gives the result of the radius form for the same distance, and refuses
  ``rho`` without molecule centroids.

The real-row cases (phenacetin, pyridone, the CSD fixtures, two-compound crystals) are the characterization
oracle's (tests/characterization/test_characterization_oracle.py), whose stage-0 xfails this stage flipped.
CPU only, under a minute.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def _charlib():
    spec = importlib.util.spec_from_file_location('charlib_for_cluster_reach',
                                                  str(REPO / 'tests' / 'characterization' / 'charlib.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CL = _charlib()
EPS = 0.01                         # the decisive contact sits this far inside the cutoff (A)


@pytest.fixture(autouse=True, scope='module')
def _deterministic_cpu():
    with CL.deterministic():
        yield


# three heavy atoms (a frame); hydrogens at +-hx on x, at hy + lop and -hy on y, and at +-hy on z, all about the
# heavy-atom centroid: the all-atom centroid lies about lop / 9 from the heavy-atom centroid along y, and with
# hy + lop < hx the x hydrogens set the reach on both sides of the contact line
_HEAVY_Z = [6, 6, 7]
_HEAVY = [[-0.6, -0.35, 0.0], [0.6, -0.35, 0.0], [0.0, 0.7, 0.1]]


def _spiky(hx, hy=1.0, lop=0.0):
    from mxtaltools.dataset_utils.data_classes import MolData
    hs = [[hx, 0, 0], [-hx, 0, 0], [0, hy + lop, 0], [0, -hy, 0], [0, 0, hy], [0, 0, -hy]]
    z = torch.tensor(_HEAVY_Z + [1] * 6)
    pos = torch.tensor(_HEAVY + hs, dtype=torch.float32)
    pos = pos - pos[:3].mean(0)
    return MolData(z=z, pos=pos, x=torch.zeros(len(z), 1), do_mol_analysis=True)


_BIG, _SMALL, _MID = (3.5, 1.0, 2.4), (1.1, 0.8, 0.0), (1.6, 0.9, 1.2)    # (hx, hy, lop)


_ORIENT = (0.02, 0.01, 0.015)      # a small rotation: the hydrogens stay close to the cell axes


def _row(a, units, centroids, bc=(16.0, 17.0), max_z_prime=None):
    """A P1 crystal with an orthogonal cell (a, *bc), one unit per (hx, hy, lop) of ``units``."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    zp = len(units)
    mols = [_spiky(*u) for u in units]
    return MolCrystalData(molecule=mols if zp > 1 else mols[0], sg_ind=1, z_prime=torch.tensor([zp]),
                          max_z_prime=max_z_prime or zp, cell_lengths=torch.tensor([float(a), *bc]),
                          cell_angles=torch.full((3,), torch.pi / 2), aunit_centroid=torch.tensor(centroids),
                          aunit_orientation=torch.tensor(_ORIENT * zp), aunit_handedness=torch.ones(zp).long(),
                          identifier='SPIKY', do_box_analysis=True)


def _molecule0(batch, std=False):
    """Each unit's molecule 0 for row 0 of a one-row P1 batch: its unit cell, one image per unit, unit by unit (the
    cluster's aux_ind 0 molecules at the zero translation), and their elements."""
    assert batch.num_graphs == 1 and int(batch.sym_mult[0]) == 1
    c = batch.clone()
    c.mol2ucell(std_orientation=std)
    zp, n = int(c.z_prime[0]), int(c.num_atoms[0])
    pos = c.unit_cell_pos.reshape(zp, n // zp, 3)
    return list(pos), list(c.z.reshape(zp, n // zp))


def _closest(batch, i, j, shift):
    """Closest atom distance between molecule-0 unit i and molecule-0 unit j moved by the Cartesian ``shift``."""
    units, _ = _molecule0(batch)
    a, b = units[i].double(), units[j].double() + torch.as_tensor(shift, dtype=torch.float64)
    return float((a[:, None] - b[None]).norm(dim=-1).min())


def _solve(make, i, j, shift_of, target, lo, hi):
    """The cell length at which ``_closest`` equals ``target`` (bisection; it increases with the length here)."""
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _closest(make(mid), i, j, shift_of(mid)) < target else (lo, mid)
    return 0.5 * (lo + hi)


def _compare(batch, cutoff, std=False):
    """Builder against brute force around the cluster's own molecule 0 (the reference the reach test must be exact
    for): charlib.compare_pair_sets' dict."""
    _, cl = batch.clone().analyze(['elj'], cutoff=cutoff, std_orientation=std, return_cluster=True)
    m = (cl.aux_ind == 0) & (cl.batch == 0)
    pos, z, mol = cl.pos[m], cl.z[m], cl.mol_ind[m]
    units = [pos[mol == k] for k in torch.unique(mol)]
    zs = [z[mol == k] for k in torch.unique(mol)]
    oracle = CL.oracle_pairs(units, zs, batch.T_fc[0], batch.symmetry_operators[0], cutoff)
    return CL.compare_pair_sets(CL.builder_pairs(cl, 0, torch.cat(units)), oracle, cutoff)


def _check(cmp):
    if cmp['missing'] or cmp['extra'] or cmp['distance_mismatch'] or cmp['element_mismatch']:
        pytest.fail(f'builder {cmp["n_builder"]} pairs, brute force {cmp["n_oracle"]}: {cmp["missing"]} missing, '
                    f'{cmp["extra"]} extra; atoms {cmp["count_mismatch_atoms"][:4]}')


# ------------------------------------------------------------------------------------------------
# the cases
# ------------------------------------------------------------------------------------------------

_ZP1_CEN = (0.2, 0.3, 0.4)


def _zp1_case(cutoff, centroid=_ZP1_CEN, unit=_BIG):
    """Z'=1: the closest contact with the +a translate at cutoff - EPS."""
    a = _solve(lambda a: CL.collate([_row(a, [unit], centroid)]), 0, 0, lambda a: (a, 0.0, 0.0),
               cutoff - EPS, 2 * unit[0] + 0.5, cutoff + 4 * unit[0])
    return CL.collate([_row(a, [unit], centroid)]), a


def _zp2_case(cutoff, units=(_SMALL, _BIG)):
    """Unit 0 (reach ~1.1 A, all-atom centroid on its heavy-atom centroid) and unit 1 (reach ~3.5 A, centroid offset
    ~0.27 A), unit 1 halfway between unit 0 and unit 0 + a x: the closest contact of that image with unit 1 at
    cutoff - EPS."""
    cen = (0.2, 0.3, 0.4, 0.7, 0.3, 0.4)

    def make(a):
        return CL.collate([_row(a, list(units), cen, bc=(2 * cutoff + 12, 2 * cutoff + 13))])

    a = _solve(make, 1, 0, lambda a: (a, 0.0, 0.0), cutoff - EPS, 2 * cutoff, 2 * cutoff + 30)
    return make(a), a


def _zp3_case(cutoff):
    """Three units of unequal reach and offset on the x axis, unit 2 (the farthest-reaching) last; the contact of
    unit 2 with unit 0's +a image at cutoff - EPS. For unit 1 the maximising unit of its buffer is unit 2, neither
    slot 0 nor its nearest neighbour."""
    units = [_SMALL, _MID, _BIG]
    cen = (0.10, 0.3, 0.4, 0.30, 0.3, 0.4, 0.62, 0.3, 0.4)

    def make(a):
        return CL.collate([_row(a, units, cen, bc=(2 * cutoff + 12, 2 * cutoff + 13))])

    a = _solve(make, 2, 0, lambda a: (a, 0.0, 0.0), cutoff - EPS, cutoff + 4, 3 * cutoff + 30)
    return make(a), a


# ------------------------------------------------------------------------------------------------

def test_molecule_reach_is_the_all_atom_reach():
    from mxtaltools.crystal_building.utils import molecule_reach
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(11, 3, generator=g) * 2
    batch = torch.tensor([0] * 4 + [1] * 7)
    cen, reach = molecule_reach(pos, batch, 2)
    for k, sl in enumerate((slice(0, 4), slice(4, 11))):
        p = pos[sl].double()
        c = p.mean(0)
        assert torch.allclose(cen[k].double(), c, atol=1e-6)
        assert abs(float(reach[k]) - float((p - c).norm(dim=1).max())) < 1e-5


def test_spiky_all_atom_centroid_is_off_the_heavy_atom_centroid():
    """The fixture can tell the two centroids apart (the review's point: a symmetric molecule cannot), and the x
    hydrogens set the reach, so the bound 2 R is tight along the contact line (within 0.03 A)."""
    m = _spiky(*_BIG)
    c = m.pos.double().mean(0)
    off = float((c - m.pos[:3].double().mean(0)).norm())
    assert off > 0.25, off
    reach = float((m.pos.double() - c).norm(dim=1).max())
    x_ext = float(m.pos[3, 0] - m.pos[4, 0])                 # the two x hydrogens
    assert 0 <= 2 * reach - x_ext < 0.03, (reach, x_ext)


@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_unflagged_molecules_are_exactly_those_within_the_reach(cutoff):
    """The rule, pinned directly: aux_ind < 2 iff the molecule's all-atom centroid lies within rho of molecule 0's;
    and every unit-cell image within rho is in the cluster (independent enumeration of the lattice)."""
    from mxtaltools.crystal_building.utils import REACH_MARGIN
    b, a = _zp1_case(cutoff)
    _, cl = b.clone().analyze(['elj'], cutoff=cutoff, std_orientation=False, return_cluster=True)
    units, _ = _molecule0(b)
    c0 = units[0].double().mean(0)
    R = float((units[0].double() - c0).norm(dim=1).max())
    rho = cutoff + 2 * R + REACH_MARGIN
    mol = cl.mol_ind
    cents = torch.stack([cl.pos[mol == k].double().mean(0) for k in range(int(mol.max()) + 1)])
    flag = torch.stack([cl.aux_ind[mol == k].max() for k in range(int(mol.max()) + 1)])
    d = (cents - c0).norm(dim=1)
    band = (d - rho).abs() < 1e-4
    assert bool(((d <= rho) == (flag < 2))[~band].all())
    # every lattice image within rho - 1e-4 is built and unflagged (P1: one molecule per cell)
    T = b.T_fc[0].double()
    rng = torch.arange(-4, 5, dtype=torch.float64)
    n = torch.cartesian_prod(rng, rng, rng)
    want = ((c0 + n @ T.T - c0).norm(dim=1) <= rho - 1e-4).sum()
    assert int((flag < 2).sum()) >= int(want)
    assert int(want) > 1                                 # the case has neighbours inside the reach


@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_hydrogens_past_the_heavy_atom_radius_are_reached_at_zp1(cutoff):
    b, a = _zp1_case(cutoff)
    assert abs(_closest(b, 0, 0, (a, 0.0, 0.0)) - (cutoff - EPS)) < 1e-3
    # the contact is one the heavy-atom rule could not see: its molecules' centroids are farther apart than it
    assert a > cutoff + 2 * float(b.radius[0]) + 0.1 + 0.5, (a, float(b.radius[0]))
    _check(_compare(b, cutoff))


@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_a_centroid_at_the_clip_is_still_exact(cutoff):
    """aunit_centroid x = 1.0: pos is posed at 1.0, the unit cell (and so the cluster's molecule 0) at 1 - 1e-4. The
    reach must be centred on molecule 0 (reference_molecule_reach); centred on pos it is 1e-4 a away, more than
    REACH_MARGIN in this cell, and the -a image's cell went unbuilt (the stage-5 review's probe, on y). The contact
    sits 3e-4 A inside the cutoff, outside the oracle's 2e-4 band."""
    from mxtaltools.crystal_building.utils import REACH_MARGIN
    # no centroid offset: the x hydrogens then sit at the reach on both sides, and the bound is tight to ~1e-5 A
    unit, cen = (6.0, 1.0, 0.0), (1.0, 0.3, 0.4)        # a ~ cutoff + 12 A: the clip moves molecule 0 >= 1.8e-3 A

    def make(a):
        from mxtaltools.dataset_utils.data_classes import MolCrystalData
        row = MolCrystalData(molecule=_spiky(*unit), sg_ind=1, z_prime=torch.tensor([1]), max_z_prime=1,
                             cell_lengths=torch.tensor([a, 30.0, 31.0]), cell_angles=torch.full((3,), torch.pi / 2),
                             aunit_centroid=torch.tensor(cen), aunit_orientation=torch.tensor((0.0, 0.0, 1e-3)),
                             aunit_handedness=torch.ones(1).long(), identifier='CLIP', do_box_analysis=True)
        return CL.collate([row])

    a = _solve(make, 0, 0, lambda a: (a, 0.0, 0.0), cutoff - 3e-4, 2 * unit[0] + 0.5, cutoff + 4 * unit[0])
    assert a > 1.5 * REACH_MARGIN / 1e-4                   # 1e-4 a, the clip's displacement, exceeds the margin
    batch = make(a)
    posed = batch.clone()
    posed.mol2ucell(std_orientation=False)
    units, _ = _molecule0(batch)
    off = float((units[0].double().mean(0) - posed.pos.double().mean(0)).norm())
    assert off > 1.2 * REACH_MARGIN, off                 # the case is sensitive to the centre
    _check(_compare(batch, cutoff))


@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_every_unit_reaches_every_unit_of_its_crystal_at_zp2(cutoff):
    b, a = _zp2_case(cutoff)
    _check(_compare(b, cutoff))


@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_every_unit_reaches_every_unit_of_its_crystal_at_zp3(cutoff):
    b, a = _zp3_case(cutoff)
    _check(_compare(b, cutoff))


def _symmetric_buffer(self, zp1_batch):
    from mxtaltools.crystal_building.utils import reference_molecule_reach
    c, _ = reference_molecule_reach(zp1_batch)
    return (c[:, None] - c[None]).norm(dim=-1).amax(dim=0)      # one row in these tests


@pytest.mark.parametrize('case', ['zp2', 'zp3'])
@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_a_symmetric_buffer_misses_the_far_units_contacts(case, cutoff, monkeypatch):
    """Negative control: the cases above are sensitive to the R_u - R_v term. With the buffer max_u |c_u - c_v|
    (each unit's cluster reaching only cutoff + 2 R_v + d) the near unit's cluster stops short of the image that
    touches the far unit, and the comparison fails."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_building import MolCrystalBuilding
    b, _ = (_zp2_case if case == 'zp2' else _zp3_case)(cutoff)
    monkeypatch.setattr(MolCrystalBuilding, '_unit_reach_buffer', _symmetric_buffer)
    cmp = _compare(b, cutoff)
    assert cmp['missing'] > 0 and cmp['extra'] == 0, cmp['missing']


@pytest.mark.parametrize('case', ['zp1', 'zp2'])
@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_a_reach_short_by_005_misses_a_pair(case, cutoff, monkeypatch):
    """Negative control on the margin: with REACH_MARGIN at -0.05 A (rho 0.051 A short) the decisive contact, 0.01 A
    inside the cutoff with its atoms nearly on the centroid line, is lost: the cases resolve the reach to a few
    hundredths of an Angstrom."""
    import mxtaltools.crystal_building.utils as CBU
    b, _ = (_zp1_case if case == 'zp1' else _zp2_case)(cutoff)
    monkeypatch.setattr(CBU, 'REACH_MARGIN', -0.05)
    cmp = _compare(b, cutoff)
    assert cmp['missing'] > 0, cmp


# ------------------------------------------------------------------------------------------------

def _reach_buffer(batch, std=False):
    s = batch.split_to_zp1_batch()
    s.pose_aunit(std_orientation=std)
    s.build_unit_cell()
    return batch._unit_reach_buffer(s), s


def _reference_buffer(batch, s):
    """max_u (R_u + |c_u - c_v|) - R_v from each unit's molecule 0 (the first num_atoms rows of its unit cell)."""
    ref, start, ustart = [], 0, 0
    n_u = s.num_atoms.tolist()
    m_u = s.sym_mult.reshape(-1).tolist()
    blocks = []
    for k in range(len(n_u)):
        blocks.append(s.unit_cell_pos[ustart:ustart + n_u[k]].double())
        ustart += n_u[k] * m_u[k]
    for zp in batch.z_prime.reshape(-1).tolist():
        units = blocks[start:start + zp]
        c = [u.mean(0) for u in units]
        r = [float((u - cu).norm(dim=1).max()) for u, cu in zip(units, c)]
        for v in range(zp):
            ref.append(max(r[u] + float((c[u] - c[v]).norm()) for u in range(zp)) - r[v])
        start += zp
    return torch.tensor(ref, dtype=torch.float64)


def _slot_buffer(batch):
    """The pre-stage-5 buffer, for contrast: the largest distance between the row's own centroid slots."""
    from mxtaltools.common.geometry_utils import fractional_transform
    zp = int(batch.max_z_prime)
    cart = fractional_transform(batch.aunit_centroid.reshape(-1, 3), batch.T_fc.repeat_interleave(zp, dim=0))
    cart = cart.reshape(batch.num_graphs, zp, 3)
    return (cart[:, :, None] - cart[:, None]).norm(dim=-1)


def test_unit_reach_buffer_is_its_definition():
    zp2, _ = _zp2_case(6.0)
    zp3, _ = _zp3_case(6.0)
    wide = CL.collate([_row(12.0, [_BIG], (0.2, 0.3, 0.4)),
                       _row(13.0, [_SMALL, _BIG], (0.1, 0.2, 0.3, 0.6, 0.5, 0.4)),
                       _row(14.0, [_MID, _SMALL], (0.15, 0.25, 0.35, 0.45, 0.75, 0.6))], max_z_prime=2)
    for batch in (zp2, zp3, wide):
        buf, s = _reach_buffer(batch)
        ref = _reference_buffer(batch, s)
        assert buf.shape == (int(batch.z_prime.sum()),)
        assert torch.allclose(buf.double(), ref, atol=1e-5), (buf, ref)
        assert bool((buf >= 0).all())
    assert float(_reach_buffer(wide)[0][0]) == 0.0                     # the Z'=1 row: exactly 0
    # the definition reads each unit's all-atom molecule 0, not its slot (the heavy-atom centroid): on wide's row 1,
    # whose units are separated obliquely to _BIG's 0.27 A centroid offset, the slot distances give another buffer
    from mxtaltools.crystal_building.utils import reference_molecule_reach
    buf, s = _reach_buffer(wide)
    _, R = reference_molecule_reach(s)
    slot = float(_slot_buffer(wide)[1, 0, 1])
    r1, r2 = float(R[1]), float(R[2])                                  # row 1's units are unit-graphs 1 and 2
    slot_rule = torch.tensor([max(r1, r2 + slot) - r1, max(r2, r1 + slot) - r2], dtype=torch.float64)
    assert float((slot_rule - buf[1:3].double()).abs().max()) > 0.05, (slot_rule, buf[1:3])


def test_unit_reach_buffer_reads_neither_pads_nor_batch_mates():
    zp2 = _row(13.0, [_SMALL, _BIG], (0.1, 0.2, 0.3, 0.6, 0.5, 0.4))
    alone, _ = _reach_buffer(CL.collate([zp2]))
    zp2w = _row(13.0, [_SMALL, _BIG], (0.1, 0.2, 0.3, 0.6, 0.5, 0.4), max_z_prime=3)
    mixed, _ = _reach_buffer(CL.collate([_row(12.0, [_BIG], (0.9, 0.9, 0.9)), zp2w], max_z_prime=3))
    assert float(mixed[0]) == 0.0
    assert torch.equal(mixed[1:], alone)


# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_image_pairs_reaches_the_hydrogens(cutoff):
    """crystal_building.image_pairs keeps the builder's images on a crystal whose contact the heavy-atom reach misses:
    same pair count and sorted distances inside cutoff - 1e-3 (its own test's convention)."""
    from mxtaltools.crystal_building.image_pairs import build_image_tables, image_pairs
    b, _ = _zp1_case(cutoff)
    _, cl = b.clone().analyze(['elj'], cutoff=cutoff, std_orientation=False, return_cluster=True)
    d_old = cl.edges_dict['intermolecular_dist']
    with torch.no_grad():
        new = image_pairs(build_image_tables(b), b.T_fc, b.T_cf, b.aunit_centroid, b.aunit_orientation, cutoff)
    d_new = new['dist']
    d_old, d_new = (torch.sort(d[d < cutoff - 1e-3]).values for d in (d_old, d_new))
    assert d_old.shape == d_new.shape, (len(d_old), len(d_new))
    assert float((d_old - d_new).abs().max()) < 2e-4
    assert float(d_new.max()) > cutoff - 2 * EPS                     # the decisive contact is among them


def test_cart_translations_rho_equals_the_radius_form():
    from mxtaltools.crystal_building.utils import get_cart_translations
    g = torch.Generator().manual_seed(1)
    T = torch.diag_embed(torch.rand(3, 3, generator=g) * 4 + 7)
    cc = torch.rand(3, 3, generator=g) * 3
    radius = torch.rand(3, generator=g) + 2.5
    mol = torch.cat([cc + 0.0, cc + torch.tensor([1.5, 2.0, 2.5])])[torch.tensor([0, 3, 1, 4, 2, 5])]
    per = torch.tensor([2, 2, 2])
    ref = get_cart_translations(cc, T, radius, 6.0, mol_centroids=mol, mols_per_crystal=per)
    got = get_cart_translations(cc, T, None, None, mol_centroids=mol, mols_per_crystal=per,
                                rho=radius * 2 + 6.0 + 0.1)
    assert all(torch.equal(x, y) for x, y in zip(ref, got))
    with pytest.raises(ValueError, match='mol_centroids'):
        get_cart_translations(cc, T, None, None, rho=radius * 2 + 6.1)

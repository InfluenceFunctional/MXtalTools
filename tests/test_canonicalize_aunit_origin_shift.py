"""
crystal_opt_utils.canonicalize_aunit in P4_2/n (sg 86), Z' = 1.

SYM_OPS[86] is ITA origin choice 2 (inversion at the origin) while ASYM_UNITS['86'] is the origin-choice-1
asymmetric unit, so about half of all centres have no symmetry image in the box. canonicalize_aunit then moves the
crystal by a half-cell origin shift that keeps the operator list (ORIGIN_SHIFTS) before folding.

Contracts.
1. Control: the chosen centres include some with no image in the box (origin_shift=False raises on them).
2. With the default, every row is canonicalised: centre inside the box, rotvec on the +z hemisphere.
3. The result is the same crystal: its unit cell is the input's translated by one ORIGIN_SHIFTS vector, atom for
   atom; the zero shift alone does not match a shifted row (the comparison can fail); the eLJ energy is unchanged.
4. Rows that have an image in the box are canonicalised exactly as with origin_shift=False.

Synthetic chiral molecule, no data files. CPU only.
"""
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.crystal_search.crystal_opt_utils import ORIGIN_SHIFTS, canonicalize_aunit
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

SG = 86
N_ROWS = 24
ATOM_TOL = 1e-3  # Angstrom; float32 builds agree to ~1e-5
CONTROL = 0.1    # Angstrom: a different set of atoms is at least this far off
# a rigid chiral molecule (no mirror): C at the origin with four different substituents, one of them a short chain
Z = torch.tensor([6, 7, 8, 9, 6, 1])
POS = torch.tensor([[0.00, 0.00, 0.00],
                    [1.45, 0.05, -0.10],
                    [-0.50, 1.35, 0.15],
                    [-0.45, -0.70, 1.20],
                    [-0.55, -0.75, -1.30],
                    [-1.60, -0.70, -1.45]])


def _crystals(seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(N_ROWS):
        rv = torch.randn(3, generator=g)
        rv = rv / rv.norm() * (0.3 + 2.5 * torch.rand(1, generator=g))
        m = MolData(z=Z.clone(), pos=POS.clone(), x=torch.zeros(len(Z), 1), do_mol_analysis=True)
        out.append(MolCrystalData(molecule=m, sg_ind=SG, cell_lengths=torch.tensor([11.0, 11.0, 13.5]),
                                  cell_angles=torch.full((3,), torch.pi / 2),
                                  aunit_centroid=torch.rand(3, generator=g),
                                  aunit_orientation=rv,
                                  aunit_handedness=torch.tensor([1.0 if bool(torch.rand(1, generator=g) < 0.5)
                                                                 else -1.0]),
                                  do_box_analysis=True))
    return out


def _unit_cell(c):
    """(fractional unit-cell atoms wrapped into [0, 1) [N, 3] float64, element per atom [N], T_fc) via mol2ucell."""
    b = collate_data_list([c.clone()])
    with torch.no_grad():
        b.mol2ucell(std_orientation=True)
    f = b.unit_cell_pos.double() @ b.T_cf[0].double().T
    z = c.z[None].expand(int(b.sym_mult[0]), -1).reshape(-1)
    return f - torch.floor(f), z.numpy(), b.T_fc[0].double()


def _deviation(a, b, z, t_fc, shift):
    """Largest distance (Angstrom) between atoms of a moved by `shift` (fractional) and matched atoms of b."""
    worst = 0.0
    for e in np.unique(z):
        d = (a[z == e] + torch.as_tensor(shift, dtype=torch.float64))[:, None] - b[z == e][None]
        d = d - torch.round(d)
        dist = torch.linalg.norm(d @ t_fc.T, dim=-1).numpy()
        r, cidx = linear_sum_assignment(dist)
        worst = max(worst, float(dist[r, cidx].max()))
    return worst


def _elj(b):
    with torch.no_grad():
        return b.analyze(['elj'], cutoff=6, supercell_size=5)['elj'].double().reshape(-1)


@pytest.fixture(scope='module')
def rows():
    return _crystals()


def test_p42n_canonicalisation_keeps_the_crystal(rows):
    box = torch.tensor(ASYM_UNITS[str(SG)], dtype=torch.float64)
    needs_shift = []
    for c in rows:  # 1. control: which rows have no image in the box without a shift
        try:
            canonicalize_aunit(collate_data_list([c.clone()]), origin_shift=False)
            needs_shift.append(False)
        except ValueError:
            needs_shift.append(True)
    assert 4 <= sum(needs_shift) <= N_ROWS - 4, needs_shift  # about half; both kinds must be exercised

    before = collate_data_list([c.clone() for c in rows])
    after = before.clone()
    canonicalize_aunit(after)  # 2. never raises in sg 86
    u = after.aunit_centroid.double() / box
    assert bool(((u >= -1e-6) & (u <= 1 + 1e-6)).all()), 'canonical centre left the box'
    assert bool((after.aunit_orientation[:, 2] >= -1e-6).all()), 'canonical rotvecs must sit on the +z hemisphere'

    for i, (c, shifted) in enumerate(zip(rows, needs_shift)):  # 3. the same crystal, translated by one shift
        a, z, t_fc = _unit_cell(c)
        b, _, _ = _unit_cell(after.batch_to_list()[i])
        devs = [_deviation(a, b, z, t_fc, v) for v in ORIGIN_SHIFTS]
        assert min(devs) < ATOM_TOL, (i, devs)
        if shifted:
            assert devs[0] > CONTROL, (i, devs)  # the shift is real: the atoms moved
        else:
            assert devs[0] < ATOM_TOL, (i, devs)  # an in-box row is never shifted
    e0, e1 = _elj(before), _elj(after)
    assert torch.allclose(e0, e1, rtol=1e-4, atol=1e-4), (e0, e1)


def test_unshifted_rows_match_origin_shift_off(rows):
    keep = []
    for c in rows:
        try:
            canonicalize_aunit(collate_data_list([c.clone()]), origin_shift=False)
            keep.append(c)
        except ValueError:
            pass
    off = collate_data_list([c.clone() for c in keep])
    on = off.clone()
    canonicalize_aunit(off, origin_shift=False)
    canonicalize_aunit(on)
    assert torch.equal(off.aunit_centroid, on.aunit_centroid)
    assert torch.equal(off.aunit_orientation, on.aunit_orientation)
    assert torch.equal(off.aunit_handedness, on.aunit_handedness)

"""B -- the eLJ reward, pinned ABSOLUTELY.

This is the energy GFN's canonical config trains on (`mk_dev.yaml`:
`energy_function: 'elj'`), and **nothing in either repo pinned its scale**. A
change to `stiffness = repulsion * 2.5`, to VDW_RADII, or to the cutoff shifts
every reward uniformly -- which a GFN run absorbs as a temperature change and
never reports. Existing coverage is `test_batch_invariance.py` (path A == path B)
plus an `isfinite` assert; a uniform rescale passes both.

The three literal anchors below each isolate ONE thing:

    E(r=0)      identical for H..I -- sigma CANCELS, so it pins `stiffness` alone
    E(sigma)    exactly 0          -- pins the zero crossing
    E(r_min)    exactly -1         -- pins epsilon = 1, HARDCODED and dimensionless
"""

import numpy as np
import pytest
import torch

from mxtaltools.analysis.vdw_analysis import exponential_edgewise_lj_energy
from mxtaltools.constants.atom_properties import VDW_RADII

pytestmark = pytest.mark.golden

VDW = torch.tensor([VDW_RADII[z] for z in range(100)], dtype=torch.float32)
PAIRS = [(1, 1), (6, 6), (6, 1), (7, 8), (17, 17), (53, 53)]


def _energy(zi, zj, r):
    dd = {'intermolecular_dist': torch.tensor([r], dtype=torch.float32),
          'intermolecular_dist_atoms': [torch.tensor([zi]), torch.tensor([zj])],
          'intermolecular_dist_batch': torch.tensor([0])}
    return float(exponential_edgewise_lj_energy(VDW, dd, 2.5)[0])


def test_energy_at_zero_separation_is_sigma_independent(golden):
    """E(r=0) is the SAME for hydrogen and iodine -- sigma cancels exactly.

    That is what makes it the one anchor isolating `stiffness` from `VDW_RADII`:
    a radius change cannot move it, a stiffness change must.
    """
    got = [_energy(zi, zj, 0.0) for zi, zj in PAIRS]
    spread = max(got) - min(got)
    assert spread < 1e-3, f'sigma should cancel at r=0; spread across pairs is {spread:.2e}'

    ref = golden('elj', 'B2_energy_at_zero', got[0])
    assert got[0] == pytest.approx(ref, abs=1e-3)

    # closed form (24/k)(e^k - 1) at k = 2.5, independent of sigma
    closed = (24.0 / 2.5) * (np.exp(2.5) - 1.0)
    assert got[0] == pytest.approx(closed, abs=1e-3), (
        f'measured {got[0]} but the exponential form predicts {closed}')


def test_energy_at_sigma_is_exactly_zero():
    """The zero crossing. Bit-exact, so no tolerance is needed or wanted."""
    for zi, zj in PAIRS:
        sigma = float(VDW[zi] + VDW[zj])
        assert _energy(zi, zj, sigma) == 0.0, (
            f'E(sigma) != 0 for z=({zi},{zj}); the zero crossing has moved')


def test_well_depth_is_exactly_one_epsilon_is_hardcoded(golden):
    """E at r_min is -1 for EVERY pair, because epsilon is hardcoded to 1.

    Worth stating plainly: the reward is in DIMENSIONLESS epsilon units, while
    surrounding prose calls it kJ/mol. Any epsilon recalibration, or swapping in
    Buckingham or SiLU with a different well depth, is invisible without this.
    """
    got = [_energy(zi, zj, float(VDW[zi] + VDW[zj]) * 2 ** (1 / 6)) for zi, zj in PAIRS]
    ref = golden('elj', 'B3_well_depth', got[0])
    for (zi, zj), v in zip(PAIRS, got):
        assert v == pytest.approx(ref, abs=1e-6), (
            f'well depth is {v} for z=({zi},{zj}), not {ref}; epsilon is no longer 1')


# --------------------------------------------------------------------------
# the canonical route, on real crystals
# --------------------------------------------------------------------------

CANONICAL = ['NUSGEN', 'KEQQON', 'MOPJEG']


def _analyze(mini_by_id, std_orientation, supercell_size=10):
    from mxtaltools.dataset_utils.utils import collate_data_list
    missing = [k for k in CANONICAL if k not in mini_by_id]
    if missing:
        pytest.skip(f'fixture crystals absent: {missing}')
    batch = collate_data_list([mini_by_id[k].clone() for k in CANONICAL])
    out = batch.analyze(['reduction_en', 'elj'], cutoff=10,
                        supercell_size=supercell_size, std_orientation=std_orientation)
    return [float(v) for v in out['elj']], [float(v) for v in out['reduction_en']]


def test_canonical_elj_route(golden, mini_by_id):
    """The exact call GFN's reward makes (`energies/molecular_crystal.py:492-495`)."""
    elj, reduction = _analyze(mini_by_id, std_orientation=False)
    ref = golden('elj', 'B4_canonical_elj', elj)
    rtol, atol = golden.tol('elj', 'B4_canonical_elj', 1e-4, 1e-3)
    assert elj == pytest.approx(ref, rel=rtol, abs=atol)
    assert reduction == pytest.approx([0.0, 0.0, 0.0], abs=1e-8), (
        'a reduced CSD cell should carry no reduction penalty')


def test_std_orientation_is_not_a_gauge(golden, mini_by_id):
    """`std_orientation` is documented as an orientation CONVENTION. It is a
    two-fold reward lever on crystals whose `pos` is not standard-oriented.

    POPULATION MATTERS, and this test pins the population as much as the number.
    These are STORED CSD crystals, whose deposited `pos` is not standard-oriented,
    so the flag re-orients the molecule and changes the structure. GFN's GENERATED
    crystals arrive standard-oriented, where the flag is a measured no-op (4e-5 A).
    Quoting the swing below as a production risk would be reading the wrong
    population -- which is precisely the mistake this test exists to make hard.
    """
    elj_false, _ = _analyze(mini_by_id, std_orientation=False)
    elj_true, _ = _analyze(mini_by_id, std_orientation=True)

    ref = golden('elj', 'B5_std_orientation_true', elj_true)
    rtol, atol = golden.tol('elj', 'B5_std_orientation_true', 1e-4, 1e-3)
    assert elj_true == pytest.approx(ref, rel=rtol, abs=atol)

    swing = max(abs(a - b) for a, b in zip(elj_false, elj_true))
    assert swing > 100.0, (
        f'the largest std_orientation swing is now {swing:.1f} kJ/mol. It was 806.9 '
        'on this fixture; if it has collapsed, either the flag stopped doing '
        'anything or the fixture stopped being non-standard-oriented')
    assert any(a * b < 0 for a, b in zip(elj_false, elj_true)), (
        'the flag used to FLIP THE SIGN of eLJ on 2 of these 3 crystals')


def test_supercell_size_is_converged_at_five(mini_by_id):
    """The canonical route pays for supercell_size=10; 5 is already converged.

    Not a golden value -- a claim about the cost of the default. If this starts
    failing, the neighbour-list construction changed.
    """
    e5, _ = _analyze(mini_by_id, std_orientation=False, supercell_size=5)
    e10, _ = _analyze(mini_by_id, std_orientation=False, supercell_size=10)
    worst = max(abs(a - b) for a, b in zip(e5, e10))
    assert worst < 1e-3, f'supercell 5 vs 10 differ by {worst:.2e} kJ/mol'

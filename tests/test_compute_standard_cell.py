"""
MolCrystalOps.compute_standard_cell, checked atom by atom.

spglib's standard cell is x_s = P x + p: a change of basis P AND an origin shift p. The
function used to apply only P, which rebuilds a DIFFERENT crystal whenever the shift is not
a translation that keeps SYM_OPS (e.g. C-centred Cc/C2/c in an ac-plane setting that needs
one). Adding p in spglib's default setting is not enough either: in the 24 groups with two
origin choices SYM_OPS uses origin choice 2 and spglib defaults to 1. The function also ran
at spglib's default symprec of 1e-5, where float32 crystals with long oblique cells are
assigned a subgroup.

Each crystal is compared at the atom level, at spglib's own (P, p) computed on the exact
positions the function standardized:
  - the standardized unit cell as written, before reparameterisation;
  - the crystal rebuilt from its new cell parameters and pose with the standard operators.
The rebuild reproduces the atoms only if they are invariant under SYM_OPS at the chosen
origin, so it tests the setting and the origin, not just the arithmetic.

CPU-only, no GPU, no checkpoint.
"""
import os

import numpy as np
import pytest
import torch
from ase.geometry import cellpar_to_cell

from mxtaltools.common.geometry_utils import fractional_transform
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.dataset_utils.data_class_methods.crystal_ops import spglib_hall_number
from mxtaltools.dataset_utils.utils import collate_data_list

spglib = pytest.importorskip('spglib')

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
TOL = 1e-3  # Angstrom
SYMPREC = 1e-3


@pytest.fixture(scope='module')
def molecules():
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    mols = [c for c in torch.load(DATASET, weights_only=False, map_location='cpu')
            if int(c.z_prime) == 1 and int(c.num_atoms) <= 30]
    if len(mols) < 10:
        pytest.skip(f'only {len(mols)} small Z\'=1 molecules in the fixture')
    return mols


def _crystals(mols, sg, n, seed, scramble=0):
    """Z'=1 crystals built in the SYM_OPS setting with random, unreduced cells. scramble re-sets the monoclinic
    (a, c) lattice by a unimodular integer matrix with entries in [-scramble, scramble]."""
    rng = np.random.default_rng(seed)
    lens = 10 ** rng.uniform(np.log10(7), np.log10(25), (n, 3))
    beta = np.radians(rng.uniform(60, 150, n))
    if sg >= 16:
        beta[:] = np.pi / 2
        scramble = 0
    if 75 <= sg <= 142:
        lens[:, 1] = lens[:, 0]
    for k in range(n if scramble else 0):
        while True:
            S = rng.integers(-scramble, scramble + 1, (2, 2))
            if abs(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]) == 1:
                break
        V = S @ np.array([[lens[k, 0], 0.0], [lens[k, 2] * np.cos(beta[k]), lens[k, 2] * np.sin(beta[k])]])
        lens[k, 0], lens[k, 2] = np.linalg.norm(V[0]), np.linalg.norm(V[1])
        beta[k] = np.arccos(np.clip(V[0] @ V[1] / (lens[k, 0] * lens[k, 2]), -1, 1))
    ori = rng.normal(size=(n, 3))
    ori = ori / np.linalg.norm(ori, axis=1, keepdims=True) * rng.uniform(0.3, 3.0, (n, 1))
    ori[:, 2] = np.abs(ori[:, 2])

    b = collate_data_list([mols[int(k)].clone() for k in rng.integers(len(mols), size=n)])
    b.reset_sg_info(sg)
    b.cell_lengths = torch.tensor(lens, dtype=torch.float32)
    b.cell_angles = torch.tensor(np.stack([np.full(n, np.pi / 2), beta, np.full(n, np.pi / 2)], 1),
                                 dtype=torch.float32)
    b.aunit_centroid = torch.tensor(rng.uniform(0, 0.999, (n, 3)), dtype=torch.float32)
    b.aunit_orientation = torch.tensor(ori, dtype=torch.float32)
    b.aunit_handedness = torch.tensor(rng.choice([-1, 1], (n, 1)))
    b.box_analysis()
    b.mol2ucell()
    return b


def _spglib_cell(b, i):
    """the (lattice, fractional positions, numbers) compute_standard_cell hands spglib for crystal i"""
    cellpar = b.full_cell_parameters()[i, :6].clone().numpy()
    cellpar[3:] *= 180 / np.pi
    positions = fractional_transform(b.unit_cell_pos[b.unit_cell_batch == i], b.T_cf[i]).numpy()
    numbers = b.z[b.batch == i].repeat(b.sym_mult[i]).numpy()
    return cellpar_to_cell(cellpar), positions, numbers


def _frac(b, i):
    T_fc = b.T_fc[i].double().numpy()
    return np.linalg.solve(T_fc, b.unit_cell_pos[b.unit_cell_batch == i].double().numpy().T).T, T_fc


def _mismatch(frac_a, frac_b, z, T_fc):
    """largest distance (A) from any atom of either set to its nearest same-element atom of the other, over lattice
    translations"""
    worst = 0.0
    for el in np.unique(z):
        d = frac_a[z == el][:, None] - frac_b[z == el][None]
        d -= np.round(d)
        dist = np.linalg.norm(d @ T_fc.T, axis=-1)
        worst = max(worst, dist.min(0).max(), dist.min(1).max())
    return worst


def _sym_ops_mismatch(frac, z, sg, T_fc):
    ops = np.asarray(SYM_OPS[sg], dtype=np.float64)
    return max(_mismatch(frac @ op[:3, :3].T + op[:3, 3], frac, z, T_fc) for op in ops)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')  # spglib OLD_ERROR_HANDLING notice
@pytest.mark.parametrize('sg, n, scramble, min_old_wrong, min_default_setting_wrong', [
    (9, 16, 2, 3, 0),  # Cc
    (15, 16, 2, 3, 0),  # C2/c
    (14, 16, 2, 0, 0),  # P2_1/c: every half-cell shift keeps SYM_OPS
    (86, 4, 0, 0, 4),  # P4_2/n: origin choice 2 in SYM_OPS
])
def test_standard_cell_keeps_every_atom(molecules, sg, n, scramble, min_old_wrong, min_default_setting_wrong):
    b = _crystals(molecules, sg, n, seed=sg, scramble=scramble)
    std = b.compute_standard_cell(symprec=SYMPREC)  # on b itself: b.unit_cell_pos is now exactly spglib's input
    rebuilt = std.clone()
    rebuilt.mol2ucell()

    old_wrong, default_setting_wrong = 0, 0
    for i in range(n):
        lattice, positions, numbers = _spglib_cell(b, i)
        dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=SYMPREC,
                                              hall_number=spglib_hall_number(sg))
        assert dataset.number == sg
        P, p = dataset.transformation_matrix, dataset.origin_shift
        written, T_fc = _frac(std, i)
        rebuilt_frac, T_fc_rebuilt = _frac(rebuilt, i)
        assert np.allclose(T_fc, T_fc_rebuilt)

        dev_written = _mismatch(written, positions @ P.T + p, numbers, T_fc)
        dev_rebuilt = _mismatch(rebuilt_frac, positions @ P.T + p, numbers, T_fc)
        assert dev_written < TOL, (i, dev_written)
        assert dev_rebuilt < TOL, (i, dev_rebuilt)

        # spglib's default setting: basis change only (the old code), and basis change plus origin shift
        default = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=SYMPREC)
        P_d, p_d = default.transformation_matrix, default.origin_shift
        T_fc_d = (np.linalg.inv(P_d).T @ lattice).T
        old_wrong += int(_sym_ops_mismatch(positions @ P_d.T, numbers, sg, T_fc_d) > TOL)
        default_setting_wrong += int(_sym_ops_mismatch(positions @ P_d.T + p_d, numbers, sg, T_fc_d) > TOL)

    # both failure modes must actually occur in these crystals, or the test proves nothing
    assert old_wrong >= min_old_wrong, old_wrong
    assert default_setting_wrong >= min_default_setting_wrong, default_setting_wrong


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_space_group_mismatch_raises(molecules):
    """At symprec 1e-5 spglib assigns a subgroup to some float32 crystals with long oblique cells; that must raise,
    and the same batch must standardize at the default symprec."""
    sg, n = 14, 16
    b = _crystals(molecules, sg, n, seed=0, scramble=3)
    with pytest.raises(ValueError, match='disagrees with the crystal'):
        b.compute_standard_cell(symprec=1e-5)  # on b itself: b.unit_cell_pos is now exactly spglib's input
    wrong = [spglib.get_symmetry_dataset(_spglib_cell(b, i), symprec=1e-5) for i in range(n)]
    assert sum(d is None or d.number != sg for d in wrong) >= 1
    b.clone().compute_standard_cell()


def test_hall_number_reproduces_sym_ops_for_every_space_group():
    for sg in range(1, 231):
        assert spglib.get_spacegroup_type(spglib_hall_number(sg)).number == sg

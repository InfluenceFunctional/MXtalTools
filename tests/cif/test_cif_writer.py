"""Contract tests for the symmetry-bearing CIF writer.

The bulk is licence-free and runs on any machine.  The CCDC round trip is
gated and skips cleanly -- but a skip there proves nothing, so the licence-free
half carries the real assertions.

See docs/design/cif_reader_design.md §8.
"""

from fractions import Fraction

import numpy as np
import pytest
import torch

from mxtaltools.common.cif_io import cif_block_from_crystal, sym_op_to_xyz
from mxtaltools.constants.space_group_info import SPACE_GROUPS, SYM_OPS

_AXIS = {'x': 0, 'y': 1, 'z': 2}


def xyz_to_sym_op(expr: str) -> np.ndarray:
    """Parse a CIF 'x,y,z' operator string back into a 4x4 affine.

    Deliberately an INDEPENDENT implementation, not a call back into the writer:
    a round trip through the writer's own inverse would pass for any self
    consistent but wrong convention.
    """
    op = np.zeros((4, 4))
    op[3, 3] = 1.0
    for row, term in enumerate(expr.split(',')):
        term = term.strip().replace('-', '+-')
        for part in [p for p in term.split('+') if p]:
            if part[-1] in _AXIS:                       # coefficient * axis
                col = _AXIS[part[-1]]
                coeff = part[:-1].rstrip('*')
                if coeff in ('', '-'):
                    op[row, col] += -1.0 if coeff == '-' else 1.0
                else:
                    op[row, col] += float(Fraction(coeff))
            else:                                        # bare translation
                op[row, 3] += float(Fraction(part))
    return op


def test_every_symmetry_operator_round_trips():
    """Render then re-parse every operator of every space group in the tables.

    This is the writer's strongest licence-free assertion: 230 space groups,
    every operator, exact equality.  A sign error, an axis transposition or a
    dropped translation anywhere fails here.
    """
    checked = 0
    for sg_ind, ops in SYM_OPS.items():
        for op in ops:
            original = np.asarray(op, dtype=float)
            rendered = sym_op_to_xyz(original)
            reparsed = xyz_to_sym_op(rendered)
            assert np.allclose(original, reparsed, atol=1e-9), (
                f"space group {sg_ind} ({SPACE_GROUPS.get(sg_ind)}): "
                f"{original.tolist()} rendered as {rendered!r} which re-parses to "
                f"{reparsed.tolist()}")
            checked += 1
    assert checked > 1000, f"only {checked} operators checked; the tables look truncated"


@pytest.mark.parametrize('sg_ind,expected', [
    (1, ['x,y,z']),
    (2, ['x,y,z', '-x,-y,-z']),
    (14, ['x,y,z', '-x,y+1/2,-z+1/2', '-x,-y,-z', 'x,-y+1/2,z+1/2']),   # P21/c
    (19, ['x,y,z', '-x+1/2,-y,z+1/2', '-x,y+1/2,-z+1/2', 'x+1/2,-y+1/2,-z']),  # P212121
])
def test_known_space_group_operator_strings(sg_ind, expected):
    """Pin the literal strings for groups whose operators are textbook.

    The round-trip test above proves self-consistency; these pin the CONVENTION,
    so a change that is internally consistent but no longer standard still fails.
    """
    assert [sym_op_to_xyz(o) for o in SYM_OPS[sg_ind]] == expected


def _one_crystal(sg_ind: int):
    """Build a single posed crystal without touching any dataset or checkpoint."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list

    mol = MolData().from_smiles('c1ccccc1C(=O)O', protonate=True, minimize=False,
                                do_mol_analysis=True)
    if mol is None:
        pytest.skip('rdkit failed to build the fixture molecule')

    crystal = MolCrystalData(
        molecule=mol,
        sg_ind=sg_ind,
        cell_lengths=torch.tensor([11.0, 13.0, 9.0]),
        cell_angles=torch.tensor([np.pi / 2, np.pi / 2 + 0.05, np.pi / 2]),
        aunit_centroid=torch.tensor([0.3, 0.4, 0.35]),
        aunit_orientation=torch.tensor([0.4, 0.2, 1.1]),
        aunit_handedness=1,
        identifier=f'fixture_sg{sg_ind}',
        do_box_analysis=True,
    )
    batch = collate_data_list([crystal])
    batch.aunit_centroid = batch.scale_centroid_to_unit_cell(batch.aunit_centroid)
    batch.pose_aunit()
    return batch


@pytest.mark.parametrize('sg_ind', [2, 14, 19, 61])
def test_written_block_declares_symmetry(sg_ind):
    """The written block must carry the group, its operators, and Z.

    The incumbent ASE-backed writer emits P1 with no space group at all, so each
    assertion here is a defect it would fail.
    """
    batch = _one_crystal(sg_ind)
    text = cif_block_from_crystal(batch, index=0)

    assert f'_space_group_IT_number              {sg_ind}' in text
    assert SPACE_GROUPS[sg_ind] in text
    assert '_space_group_symop_operation_xyz' in text
    # both dialects, so a reader never has to guess which it is holding
    assert '_symmetry_Int_Tables_number' in text

    op_lines = [ln for ln in text.splitlines() if ln.strip().startswith(("1  '", "2  '", "3  '",
                                                                        "4  '", "5  '", "6  '",
                                                                        "7  '", "8  '"))]
    assert len(op_lines) == len(SYM_OPS[sg_ind]), (
        f'wrote {len(op_lines)} operator lines, group has {len(SYM_OPS[sg_ind])}')

    z_line = [ln for ln in text.splitlines() if ln.startswith('_cell_formula_units_Z')][0]
    assert int(z_line.split()[-1]) == len(SYM_OPS[sg_ind]), 'Z should be sym_mult * Zprime'


def test_written_atoms_are_the_asymmetric_unit_not_the_cell():
    """Atom count must equal the aunit, NOT the symmetry-expanded cell.

    This is the distinction the whole writer exists for: an expanded P1 cell
    would carry sym_mult times as many sites and no operators.
    """
    sg_ind = 14
    batch = _one_crystal(sg_ind)
    text = cif_block_from_crystal(batch, index=0)

    start = text.index('_atom_site_occupancy')
    site_lines = [ln for ln in text[start:].splitlines()[1:] if ln.strip()]
    assert len(site_lines) == int(batch.num_atoms[0]), (
        f'wrote {len(site_lines)} sites for an asymmetric unit of '
        f'{int(batch.num_atoms[0])} atoms')
    assert len(site_lines) * len(SYM_OPS[sg_ind]) != len(site_lines), 'sanity'


def test_fractional_coordinates_are_in_range():
    """Sites are fractional, so they must not carry Cartesian magnitudes.

    Catches the failure where `pos` is written straight out without the T_cf
    transform -- the values would still be finite and plausibly formatted.
    """
    batch = _one_crystal(14)
    text = cif_block_from_crystal(batch, index=0)
    start = text.index('_atom_site_occupancy')
    coords = [[float(v) for v in ln.split()[2:5]]
              for ln in text[start:].splitlines()[1:] if ln.strip()]
    arr = np.asarray(coords)
    assert np.all(np.abs(arr) < 3.0), (
        f'fractional coordinates out of range (max |f| = {np.abs(arr).max():.2f}); '
        'these look Cartesian')


@pytest.mark.ccdc
def test_ccdc_reads_back_what_we_wrote(tmp_path):
    """External round trip: CCDC must recover the group, operators, Z and cell.

    An independent reader is the only real proof the file says what we think.
    """
    ccdc_io = pytest.importorskip('ccdc.io', reason='CCDC licence required')

    for sg_ind in (2, 14, 19, 61):
        batch = _one_crystal(sg_ind)
        batch.write_cif([0], str(tmp_path / f'sg{sg_ind}'), mode='asymmetric unit')
        path = tmp_path / f'sg{sg_ind}_0.cif'
        assert path.exists()

        read = ccdc_io.CrystalReader(str(path), format='cif')[0]
        assert read.spacegroup_symbol.replace(' ', '') == SPACE_GROUPS[sg_ind].replace(' ', ''), \
            f'wrote {SPACE_GROUPS[sg_ind]}, CCDC read {read.spacegroup_symbol}'
        assert len(read.symmetry_operators) == len(SYM_OPS[sg_ind])
        assert int(read.z_value) == len(SYM_OPS[sg_ind])

        expected = ([float(v) for v in batch.cell_lengths[0]]
                    + [float(np.degrees(float(v))) for v in batch.cell_angles[0]])
        got = list(read.cell_lengths) + list(read.cell_angles)
        assert np.allclose(got, expected, atol=1e-4), f'cell params drifted: {got} vs {expected}'


# ---------------------------------------------------------------------------
# NON-STANDARD SETTINGS
#
# Everything above builds crystals with `MolCrystalData(sg_ind=...)` and no
# `symmetry_operators`, so the constructor supplies the SYM_OPS fallback
# (crystal_ops.py:50-62) and every fixture is standard BY CONSTRUCTION.  That
# blindness let a real defect through: the writer emitted SYM_OPS[sg_ind]
# instead of the operators the crystal was built from.
#
# Measured over 400 random CSD entries, only 62.5% carry operators identical to
# the standard table in the same order; 18.2% are the same set reordered and
# 19.2% are a DIFFERENT SET (alternate settings, e.g. P2_1/n filed under IT 14).
# For that 19.2% the written file would describe a different crystal and still
# read back cleanly.
# ---------------------------------------------------------------------------

# P2_1/n operators filed under IT number 14 -- the real, common CSD case
# (YAMRAI, RAXPOZ, BAGDUM all carry exactly this).
P21N_OPS = np.stack([
    np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
    np.array([[-1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]]),
    np.array([[-1, 0, 0, 0.0], [0, -1, 0, 0.0], [0, 0, -1, 0.0], [0, 0, 0, 1]]),
    np.array([[1, 0, 0, 0.5], [0, -1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 1]]),
])


def _crystal_with_operators(sg_ind: int, ops):
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list

    mol = MolData().from_smiles('c1ccccc1C(=O)O', protonate=True, minimize=False,
                                do_mol_analysis=True)
    if mol is None:
        pytest.skip('rdkit failed to build the fixture molecule')
    crystal = MolCrystalData(
        molecule=mol, sg_ind=sg_ind, symmetry_operators=ops,
        cell_lengths=torch.tensor([11.0, 13.0, 9.0]),
        cell_angles=torch.tensor([np.pi / 2, np.pi / 2 + 0.05, np.pi / 2]),
        aunit_centroid=torch.tensor([0.3, 0.4, 0.35]),
        aunit_orientation=torch.tensor([0.4, 0.2, 1.1]),
        aunit_handedness=1, identifier='nonstandard', do_box_analysis=True)
    batch = collate_data_list([crystal])
    batch.aunit_centroid = batch.scale_centroid_to_unit_cell(batch.aunit_centroid)
    batch.pose_aunit()
    return batch


def _written_operator_strings(text: str) -> list:
    start = text.index('_space_group_symop_operation_xyz')
    out = []
    for line in text[start:].splitlines()[1:]:
        line = line.strip()
        if not line or line.startswith('loop_') or line.startswith('_'):
            break
        out.append(line.split("'")[1])
    return out


def test_nonstandard_setting_is_written_not_the_standard_table():
    """A P2_1/n crystal filed under IT 14 must emit ITS operators, not P2_1/c's.

    This is the test the original suite could not contain: its fixtures never
    passed `symmetry_operators`, so the standard fallback was the only path.
    """
    from mxtaltools.common.cif_io import operators_are_standard

    batch = _crystal_with_operators(14, P21N_OPS)
    assert not operators_are_standard(batch, 0), 'fixture is not actually non-standard'

    written = _written_operator_strings(cif_block_from_crystal(batch, index=0))
    expected = [sym_op_to_xyz(o) for o in P21N_OPS]
    standard = [sym_op_to_xyz(o) for o in SYM_OPS[14]]

    assert written == expected, f'wrote {written}, crystal was built from {expected}'
    assert written != standard, 'wrote the standard table instead of the actual operators'


def test_operator_count_inconsistent_with_sym_mult_is_refused():
    """Refuse rather than emit a cell whose operator count contradicts sym_mult."""
    batch = _crystal_with_operators(14, P21N_OPS)
    batch.symmetry_operators = [P21N_OPS[:2]]          # 2 ops, sym_mult still says 4
    with pytest.raises(ValueError, match='sym_mult'):
        cif_block_from_crystal(batch, index=0)


@pytest.mark.ccdc
def test_real_csd_operators_survive_the_write(tmp_path):
    """Write REAL CSD crystals and assert the operators are the ones they carry.

    Deliberately samples enough entries to include non-standard settings, and
    ASSERTS THE SAMPLE CONTAINS THEM -- otherwise the test silently degrades into
    the standard-only case that hid the original defect.
    """
    pytest.importorskip('ccdc', reason='CCDC licence required')
    import os
    import random

    from mxtaltools.common.cif_io import operators_are_standard
    from mxtaltools.dataset_utils.construction.featurize_cif_chunks import process_chunk

    root = r'D:\crystal_datasets\CSD_dump'
    if not os.path.isdir(root):
        pytest.skip(f'CSD corpus not present at {root}')

    files = [f for f in os.listdir(root) if f.endswith('.cif')]
    random.Random(0).shuffle(files)
    chunk = [os.path.join(root, f) for f in files[:60]]

    data_list = process_chunk(chunk, chunk_ind=0, use_filenames_for_identifiers=False,
                              protonation_state='deprotonated', max_z_prime=1)
    if len(data_list) < 10:
        pytest.skip(f'only {len(data_list)} crystals survived filtering')

    from mxtaltools.dataset_utils.utils import collate_data_list
    batch = collate_data_list(data_list)

    n_nonstandard = 0
    for i in range(batch.num_graphs):
        stored = [np.asarray(o, float) for o in batch.symmetry_operators[i]]
        written = _written_operator_strings(cif_block_from_crystal(batch, index=i))
        assert written == [sym_op_to_xyz(o) for o in stored], (
            f'crystal {batch.identifier[i]}: wrote {written}, '
            f'built from {[sym_op_to_xyz(o) for o in stored]}')
        n_nonstandard += not operators_are_standard(batch, i)

    assert n_nonstandard > 0, (
        f'none of {batch.num_graphs} sampled CSD crystals carried non-standard '
        'operators -- this test is blind as written; enlarge the sample')

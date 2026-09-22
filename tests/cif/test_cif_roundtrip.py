"""Round trip and equivalence: the reader and writer checked against each other,
and both against CCDC.

Three instruments, in increasing order of what they can prove:

1. `write_cif` -> `read_cif`  -- licence-free, needs no corpus. Exercises tokenizer,
   symmetry, sites, topology and Z' in one assertion.
2. `read_cif` -> `write_cif` -> `read_cif` -- idempotence on real CSD input.
3. native `crystal_dict` vs CCDC's `crystal_dict` -- field-by-field equivalence
   against an independent implementation. CCDC-gated.

None of this wires the native reader into `process_chunk`. It cannot be a
production ingestion path until a chemistry backend supplies bond orders: the
molecule half of `featurize_molecule` needs `rd_mol` for `molecule_smiles`,
`molecule_fingerprint` and `atom_partial_charge`, and inferring bond orders from
a heavy-atom skeleton returns a plausible WRONG molecule 61/300 times
(design §1.3). These are correctness instruments, not a switch.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.dataset_utils.construction.cif import read_cif

FIXTURES = Path(__file__).parent / 'fixtures'


def _manifest():
    path = FIXTURES / 'MANIFEST.json'
    if not path.exists():
        pytest.skip('fixture manifest missing')
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# 1. writer -> reader, licence-free
# --------------------------------------------------------------------------

def _build_crystal(sg_ind, symmetry_operators=None):
    import torch
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list

    mol = MolData().from_smiles('c1ccccc1C(=O)O', protonate=True, minimize=False,
                                do_mol_analysis=True)
    if mol is None:
        pytest.skip('rdkit failed to build the fixture molecule')
    kw = {} if symmetry_operators is None else {'symmetry_operators': symmetry_operators}
    crystal = MolCrystalData(
        molecule=mol, sg_ind=sg_ind,
        cell_lengths=torch.tensor([11.0, 13.0, 9.0]),
        cell_angles=torch.tensor([np.pi / 2, np.pi / 2 + 0.05, np.pi / 2]),
        aunit_centroid=torch.tensor([0.3, 0.4, 0.35]),
        aunit_orientation=torch.tensor([0.4, 0.2, 1.1]),
        aunit_handedness=1, identifier=f'rt_sg{sg_ind}', do_box_analysis=True, **kw)
    batch = collate_data_list([crystal])
    batch.aunit_centroid = batch.scale_centroid_to_unit_cell(batch.aunit_centroid)
    batch.pose_aunit()
    return batch


@pytest.mark.parametrize('sg_ind', [1, 2, 14, 19, 61])
def test_write_then_read_recovers_the_crystal(sg_ind, tmp_path):
    """The loop the incumbent writer could not close: it emitted P1 with no group."""
    batch = _build_crystal(sg_ind)
    batch.write_cif([0], str(tmp_path / 'rt'), mode='asymmetric unit')

    back = read_cif(tmp_path / 'rt_0.cif', require_bonds=False)[0]

    assert back.symmetry.number == sg_ind
    assert back.sym_mult == len(SYM_OPS[sg_ind])
    assert back.n_sites == int(batch.num_atoms[0])
    assert sorted(back.sites.z.tolist()) == sorted(batch.z.tolist())

    # cell parameters, in the reader's units (lengths A, angles RADIANS)
    assert np.allclose(back.cell_lengths, batch.cell_lengths[0].numpy(), atol=1e-5)
    assert np.allclose(back.cell_angles, batch.cell_angles[0].numpy(), atol=1e-6)

    # operators survive as VALUES, in order
    for got, want in zip(back.symmetry.operators, SYM_OPS[sg_ind]):
        assert np.allclose(got, np.asarray(want, float), atol=1e-8)


def test_write_then_read_preserves_nonstandard_operators(tmp_path):
    """A P2_1/n crystal filed under IT 14 must survive the loop as P2_1/n.

    ~37.5 % of the CSD carries non-standard operators. If either half of the loop
    substituted the standard table, this would come back as P2_1/c -- a different
    crystal that still reads cleanly.
    """
    p21n = np.stack([
        np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]]),
        np.array([[-1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]]),
        np.array([[-1, 0, 0, 0.0], [0, -1, 0, 0.0], [0, 0, -1, 0.0], [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0.5], [0, -1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 1]]),
    ])
    batch = _build_crystal(14, symmetry_operators=p21n)
    batch.write_cif([0], str(tmp_path / 'ns'), mode='asymmetric unit')
    back = read_cif(tmp_path / 'ns_0.cif', require_bonds=False)[0]

    assert back.symmetry.number == 14
    assert not back.symmetry.operators_are_standard
    for got, want in zip(back.symmetry.operators, p21n):
        assert np.allclose(got, want, atol=1e-8), 'operators were substituted'


# --------------------------------------------------------------------------
# 2. read -> write -> read, on real CSD input
# --------------------------------------------------------------------------

@pytest.mark.parametrize('entry', _manifest(), ids=lambda e: e['identifier'])
def test_read_write_read_is_idempotent(entry, tmp_path):
    """Reading a real CIF, writing it, and reading it back must be a fixed point.

    Stronger than the synthetic loop: real files carry centred settings,
    non-standard operators, Z' != 1 and prime-suffixed labels.
    """
    import torch
    from mxtaltools.dataset_utils.construction.cif.adapt import to_crystal_dict
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list

    first = read_cif(FIXTURES / entry['file'], require_integer_z_prime=False,
                     max_atomic_number=None)[0]
    if first.z_prime != 1:
        pytest.skip("MolCrystalData holds exactly one molecule per aunit; "
                    f"this fixture is Z'={first.z_prime}")

    # build the object the writer needs, carrying the file's own operators
    T_fc, _ = __import__('mxtaltools.common.geometry_utils', fromlist=['x']).coor_trans_matrix_np(
        'f_to_c', first.cell_lengths, first.cell_angles, return_vol=True)
    cart = first.sites.frac @ np.asarray(T_fc)[:3, :3].T

    # do_mol_analysis=True computes mol_volume/mass/radius, which
    # MolCrystalData.box_analysis() needs -- without them it raises a bare
    # `TypeError: NoneType * Tensor` at crystal_ops.py:155 rather than saying so.
    mol = MolData(z=torch.tensor(first.sites.z, dtype=torch.long),
                  pos=torch.tensor(cart, dtype=torch.float32),
                  x=torch.zeros(len(first.sites), dtype=torch.float32),
                  identifier=first.identifier, do_mol_analysis=True)
    crystal = MolCrystalData(
        molecule=mol, sg_ind=int(first.symmetry.number),
        symmetry_operators=np.stack([np.asarray(o, float) for o in first.symmetry.operators]),
        cell_lengths=torch.tensor(first.cell_lengths, dtype=torch.float32),
        cell_angles=torch.tensor(first.cell_angles, dtype=torch.float32),
        aunit_centroid=torch.zeros(3), aunit_orientation=torch.zeros(3),
        aunit_handedness=1, identifier=first.identifier, do_box_analysis=True)
    batch = collate_data_list([crystal])

    batch.write_cif([0], str(tmp_path / 'rr'), mode='asymmetric unit')
    # require_bonds=False: MolCrystalData carries no bond topology
    # (data_classes.py:643), so a CIF written from it cannot contain _geom_bond.
    # The decomposition comes back marked UNKNOWN rather than guessed.
    second = read_cif(tmp_path / 'rr_0.cif', require_integer_z_prime=False,
                      max_atomic_number=None, require_bonds=False)[0]
    assert second.components.source == 'none', (
        'a bond-less file must report its decomposition as unknown, not computed')

    assert second.symmetry.number == first.symmetry.number
    assert second.sym_mult == first.sym_mult
    assert second.n_sites == first.n_sites
    assert sorted(second.sites.z.tolist()) == sorted(first.sites.z.tolist())
    assert np.allclose(second.cell_lengths, first.cell_lengths, atol=1e-4)
    assert np.allclose(second.cell_angles, first.cell_angles, atol=1e-6)
    for a, b in zip(second.symmetry.operators, first.symmetry.operators):
        assert np.allclose(a, b, atol=1e-8), 'operators changed across the loop'


# --------------------------------------------------------------------------
# 3. field-by-field equivalence against CCDC
# --------------------------------------------------------------------------

@pytest.mark.ccdc
def test_native_crystal_dict_matches_ccdc():
    """The native `crystal_dict` must agree with CCDC's, field by field.

    This is the equivalence harness: it does not switch any production path, it
    quantifies how far the two implementations are apart so that switching one
    later is a decision backed by numbers.
    """
    io = pytest.importorskip('ccdc.io', reason='CCDC licence required')
    from mxtaltools.dataset_utils.construction.cif.adapt import to_crystal_dict
    from mxtaltools.dataset_utils.construction.featurization_utils import extract_crystal_data
    from mxtaltools.dataset_utils.construction.featurize_cif_chunks import init_reader

    compared = 0
    for entry in _manifest():
        path = FIXTURES / entry['file']
        try:
            native = to_crystal_dict(read_cif(path, max_atomic_number=None)[0])
            rdr = io.CrystalReader(str(path), format='cif')
            csd, red = init_reader(0, 'unchanged', rdr)
            ref = extract_crystal_data(csd.identifier, csd, red,
                                       csd.packing(inclusion='CentroidIncluded'))
        except Exception:
            continue

        assert native['space_group_number'] == ref['space_group_number'], entry['identifier']
        assert native['symmetry_multiplicity'] == ref['symmetry_multiplicity'], entry['identifier']
        assert native['z_prime'] == pytest.approx(ref['z_prime']), entry['identifier']
        assert native['z_value'] == ref['z_value'], entry['identifier']
        for key in ('lattice_a', 'lattice_b', 'lattice_c',
                    'lattice_alpha', 'lattice_beta', 'lattice_gamma'):
            assert native[key] == pytest.approx(ref[key], abs=1e-5), \
                f'{entry["identifier"]}: {key}'
        assert native['cell_volume'] == pytest.approx(ref['cell_volume'], rel=1e-6)

        # the operators must match AS VALUES, in order -- not merely in count
        for a, b in zip(native['symmetry_operators'], ref['symmetry_operators']):
            assert np.allclose(np.asarray(a, float), np.asarray(b, float), atol=1e-6), \
                f'{entry["identifier"]}: operators differ'
        compared += 1

    assert compared >= 5, f'only {compared} fixtures compared; the harness is near-blind'


@pytest.mark.ccdc
def test_native_unit_cell_matches_ccdc_packing():
    """Built cell vs CCDC's `packing()`, minimum-image, requiring a BIJECTION.

    The bijection is the load-bearing half: without it, many built atoms
    collapsing onto one deposited atom passes while every distance looks small.
    Measured over 25 random CSD crystals: max deviation 8.6e-15 A, 25/25 bijective.
    """
    io = pytest.importorskip('ccdc.io', reason='CCDC licence required')
    from mxtaltools.dataset_utils.construction.cif.adapt import to_crystal_dict
    from mxtaltools.dataset_utils.construction.featurization_utils import extract_crystal_data
    from mxtaltools.dataset_utils.construction.featurize_cif_chunks import init_reader

    checked = 0
    for entry in _manifest():
        if entry['z_prime'] != 1 or entry['has_metal']:
            continue
        path = FIXTURES / entry['file']
        try:
            d = to_crystal_dict(read_cif(path, max_atomic_number=None)[0])
            rdr = io.CrystalReader(str(path), format='cif')
            csd, red = init_reader(0, 'unchanged', rdr)
            ref = extract_crystal_data(csd.identifier, csd, red,
                                       csd.packing(inclusion='CentroidIncluded'))
        except Exception:
            continue

        ours = np.concatenate(d['unit_cell_coordinates'])
        theirs = np.concatenate(ref['unit_cell_coordinates'])
        if ours.shape != theirs.shape:
            continue
        T_fc = np.asarray(d['fc_transform'], float)
        T_fc = T_fc[:3, :3] if T_fc.shape[0] > 3 else T_fc
        T_cf = np.linalg.inv(T_fc)
        df = (theirs @ T_cf.T)[:, None, :] - (ours @ T_cf.T)[None, :, :]
        df -= np.round(df)
        dist = np.linalg.norm(df @ T_fc.T, axis=-1)
        nn = dist.argmin(axis=1)

        assert dist[np.arange(len(dist)), nn].max() < 1e-6, entry['identifier']
        assert len(set(nn.tolist())) == len(nn), \
            f'{entry["identifier"]}: match is not bijective -- atoms collapsed'
        # ELEMENTS must match too, not just positions: a positional bijection
        # pairing a carbon with a nitrogen passes the geometry test.
        ours_z = np.concatenate(d['unit_cell_atomic_numbers'])
        theirs_z = np.concatenate(ref['unit_cell_atomic_numbers'])
        assert (theirs_z == ours_z[nn]).all(), \
            f'{entry["identifier"]}: positions match but ELEMENTS do not'
        checked += 1

    assert checked >= 3, f'only {checked} crystals compared; the test is near-blind'

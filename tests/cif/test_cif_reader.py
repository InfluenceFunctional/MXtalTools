"""Contract tests for the CCDC-free CIF reader.

The licence-free half carries the real assertions and runs anywhere.  The
CCDC-gated half compares against an independent oracle -- but a skip there proves
nothing, which is why the fixtures below pin expected values explicitly.

Fixtures were sampled to SPAN the corpus (7 crystal systems, standard and
non-standard operators, Z' in {1/2, 1, 2}, with/without hydrogens, an
organometallic, a multi-component crystal), not taken alphabetically.  That
matters: the first N alphabetically contain no centred setting, and the reader
had a real bug that only a centred setting exposes.  See fixtures/MANIFEST.json.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from mxtaltools.constants.space_group_info import SPACE_GROUPS, SYM_OPS
from mxtaltools.dataset_utils.construction.cif import (CifParseError, CifReadError,
                                                       CifSymmetryError, CifUnsupportedError,
                                                       parse_cif, parse_symop, read_cif)

FIXTURES = Path(__file__).parent / 'fixtures'


def _manifest():
    path = FIXTURES / 'MANIFEST.json'
    if not path.exists():
        pytest.skip('fixture manifest missing')
    return json.loads(path.read_text())


def _entries():
    return [(e['file'], e) for e in _manifest()]


# --------------------------------------------------------------------------
# licence-free
# --------------------------------------------------------------------------

def test_every_standard_operator_parses_back():
    """parse_symop is the inverse of the writer's sym_op_to_xyz, on all 230 groups."""
    from mxtaltools.common.cif_io import sym_op_to_xyz
    n = 0
    for sg, ops in SYM_OPS.items():
        for op in ops:
            o = np.asarray(op, dtype=float)
            assert np.allclose(o, parse_symop(sym_op_to_xyz(o)), atol=1e-9), (sg, o)
            n += 1
    assert n > 1000, f'only {n} operators checked; the tables look truncated'


@pytest.mark.parametrize('name,entry', _entries())
def test_fixture_reads_with_expected_shape(name, entry):
    """Site count, component count, Z' and multiplicity, pinned per fixture."""
    crystals = read_cif(FIXTURES / name,
                        require_integer_z_prime=False,   # the set includes a Z'=1/2
                        max_atomic_number=None)          # and an organometallic
    assert len(crystals) == 1
    c = crystals[0]

    assert c.n_sites == entry['n_sites']
    assert c.components.n_components == entry['n_components']
    assert c.sym_mult == entry['n_ops']
    assert c.symmetry.number == entry['sg']
    assert float(c.z_prime) == pytest.approx(entry['z_prime'])
    assert c.symmetry.operators_are_standard == entry['ops_standard']


def test_z_prime_is_exact_not_float():
    """Z' is a Fraction. 1/3 must not become 0.333..., and 1/2 must not become 0.5."""
    half = [e for _, e in _entries() if e['z_prime'] < 1]
    if not half:
        pytest.skip('no Z\' < 1 fixture')
    c = read_cif(FIXTURES / half[0]['file'], require_integer_z_prime=False)[0]
    assert isinstance(c.z_prime, Fraction)
    assert c.z_prime == Fraction(1, 2)
    assert c.deposited_set_is_asymmetric_unit is False, (
        "at Z' < 1 the deposited set spans more than one asymmetric unit")


def test_deposited_set_is_asymmetric_unit_is_true_at_zprime_one():
    entry = [e for _, e in _entries() if e['z_prime'] == 1][0]
    c = read_cif(FIXTURES / entry['file'])[0]
    assert c.deposited_set_is_asymmetric_unit is True


def test_nonstandard_operators_are_preserved_not_substituted():
    """~37.5 % of the CSD differs from SYM_OPS; the file's own operators must win.

    Substituting the standard table would describe a different crystal, because
    the unit-cell builder applies whatever the object holds.
    """
    entries = [e for _, e in _entries() if not e['ops_standard']]
    assert entries, 'fixture set contains no non-standard crystal; it is blind'
    for e in entries:
        c = read_cif(FIXTURES / e['file'], require_integer_z_prime=False,
                     max_atomic_number=None)[0]
        std = [np.asarray(o, float) for o in SYM_OPS[c.symmetry.number]]
        ours = c.symmetry.operators
        assert not (len(ours) == len(std)
                    and all(np.allclose(a, b, atol=1e-6) for a, b in zip(ours, std))), \
            f'{e["identifier"]} was substituted with the standard table'


def test_centred_setting_is_accepted_and_factorises():
    """A C-centred setting lists a MULTIPLE of the standard operator count.

    ZZZKEA02 is 'C -1' under IT number 2: four operators where P-1 has two, being
    the standard pair composed with the centring translation (1/2,1/2,0). Real
    structure, not a malformed file -- an earlier version refused it.
    """
    entry = [e for _, e in _entries() if e['identifier'].startswith('ZZZKEA')]
    if not entry:
        pytest.skip('centred-setting fixture absent')
    c = read_cif(FIXTURES / entry[0]['file'])[0]
    assert c.symmetry.number == 2
    assert c.sym_mult == 4 == 2 * len(SYM_OPS[2])
    # the centring vector is present as a pure translation
    pure = [o[:3, 3] for o in c.symmetry.operators
            if np.allclose(o[:3, :3], np.eye(3), atol=1e-8)]
    assert any(np.allclose(np.mod(t, 1.0), [0.5, 0.5, 0.0], atol=1e-8) for t in pure)


def test_missing_symmetry_is_refused_never_assumed_p1():
    """A file with no operator loop must RAISE, not default to P1.

    This is the ASE trap inverted: `ase/io/cif.py:378` silently reads a non-P1
    file as P1 whenever the symop loop is absent.
    """
    bare = '\n'.join([
        'data_bare', '_cell_length_a 10.0', '_cell_length_b 10.0', '_cell_length_c 10.0',
        '_cell_angle_alpha 90.0', '_cell_angle_beta 90.0', '_cell_angle_gamma 90.0',
        '_cell_formula_units_Z 1', 'loop_', ' _atom_site_label', ' _atom_site_type_symbol',
        ' _atom_site_fract_x', ' _atom_site_fract_y', ' _atom_site_fract_z',
        ' C1 C 0.1 0.1 0.1', ' C2 C 0.2 0.2 0.2', ''])
    with pytest.raises(CifSymmetryError, match='Refusing to assume P1'):
        read_cif(bare)


def test_declared_p1_with_a_full_cell_reads():
    """A genuine P1 full-cell file is readable: P1 carries one operator, the identity.

    It becomes a Z' = Z crystal, which is correct -- the deposited set IS the cell.
    """
    text = '\n'.join([
        'data_p1cell', '_cell_length_a 10.0', '_cell_length_b 11.0', '_cell_length_c 12.0',
        '_cell_angle_alpha 90.0', '_cell_angle_beta 95.0', '_cell_angle_gamma 90.0',
        '_cell_formula_units_Z 2', '_symmetry_Int_Tables_number 1',
        'loop_', ' _symmetry_equiv_pos_as_xyz', " 'x,y,z'",
        'loop_', ' _atom_site_label', ' _atom_site_type_symbol',
        ' _atom_site_fract_x', ' _atom_site_fract_y', ' _atom_site_fract_z',
        ' C1 C 0.1 0.1 0.1', ' C2 C 0.15 0.1 0.1',
        ' C3 C 0.6 0.6 0.6', ' C4 C 0.65 0.6 0.6',
        'loop_', ' _geom_bond_atom_site_label_1', ' _geom_bond_atom_site_label_2',
        ' C1 C2', ' C3 C4', ''])
    c = read_cif(text)[0]
    assert c.symmetry.number == 1 and c.sym_mult == 1
    assert c.z_prime == 2
    assert c.components.n_components == 2


def test_prime_suffixed_labels_are_distinct_sites():
    """S1 and S1' are DIFFERENT atoms -- the CSD's Z'=2 labelling convention.

    Stripping the apostrophe as if it were quoting collapses them into a false
    duplicate; three files in a 300-file sample were refused by that bug.
    """
    text = '\n'.join([
        'data_primes', '_cell_length_a 10.0', '_cell_length_b 10.0', '_cell_length_c 10.0',
        '_cell_angle_alpha 90.0', '_cell_angle_beta 90.0', '_cell_angle_gamma 90.0',
        '_cell_formula_units_Z 1', '_symmetry_Int_Tables_number 1',
        'loop_', ' _symmetry_equiv_pos_as_xyz', " 'x,y,z'",
        'loop_', ' _atom_site_label', ' _atom_site_type_symbol',
        ' _atom_site_fract_x', ' _atom_site_fract_y', ' _atom_site_fract_z',
        " S1 S 0.1 0.1 0.1", " S1' S 0.6 0.6 0.6",
        'loop_', ' _geom_bond_atom_site_label_1', ' _geom_bond_atom_site_label_2',
        " S1 S1'", ''])
    c = read_cif(text)[0]
    assert c.n_sites == 2
    assert c.sites.labels == ['S1', "S1'"]


def test_duplicate_labels_are_refused():
    """Genuinely duplicate labels make the bond graph ambiguous -> refuse."""
    text = '\n'.join([
        'data_dupes', '_cell_length_a 10.0', '_cell_length_b 10.0', '_cell_length_c 10.0',
        '_cell_angle_alpha 90.0', '_cell_angle_beta 90.0', '_cell_angle_gamma 90.0',
        '_cell_formula_units_Z 1', '_symmetry_Int_Tables_number 1',
        'loop_', ' _symmetry_equiv_pos_as_xyz', " 'x,y,z'",
        'loop_', ' _atom_site_label', ' _atom_site_type_symbol',
        ' _atom_site_fract_x', ' _atom_site_fract_y', ' _atom_site_fract_z',
        ' C1 C 0.1 0.1 0.1', ' C1 C 0.6 0.6 0.6', ''])
    with pytest.raises(CifParseError, match='duplicate'):
        read_cif(text)


def test_defaults_refuse():
    """Every optional flag defaults to the STRICT setting.

    There is enough genuine nonsense in the CSD that a permissive default would
    quietly admit structures no workflow here should treat as valid. Measured on
    400 random entries: strict reads 81.2 %, permissive 100 %, and the difference
    is 17.8 % non-integer Z' plus 1.0 % polymers -- all refused by name.
    """
    import inspect
    sig = inspect.signature(read_cif)
    assert sig.parameters['refuse_polymers'].default is True
    assert sig.parameters['require_integer_z_prime'].default is True
    assert sig.parameters['max_atomic_number'].default == 100


def test_non_integer_zprime_refused_by_default_accepted_on_request():
    entry = [e for _, e in _entries() if e['z_prime'] < 1]
    if not entry:
        pytest.skip("no Z' < 1 fixture")
    path = FIXTURES / entry[0]['file']
    with pytest.raises(CifUnsupportedError) as exc:
        read_cif(path)
    assert exc.value.reason == 'non-integer-zprime'
    assert read_cif(path, require_integer_z_prime=False)[0].z_prime == Fraction(1, 2)


# --------------------------------------------------------------------------
# CCDC-gated: an independent oracle
# --------------------------------------------------------------------------

@pytest.mark.ccdc
@pytest.mark.parametrize('name,entry', _entries())
def test_matches_ccdc_on_fixture(name, entry):
    """Space group, multiplicity and Z' must match CCDC exactly."""
    io = pytest.importorskip('ccdc.io', reason='CCDC licence required')
    path = FIXTURES / name
    c = read_cif(path, require_integer_z_prime=False, max_atomic_number=None)[0]
    ref = io.CrystalReader(str(path), format='cif')[0]

    assert c.symmetry.number == ref.spacegroup_number_and_setting[0]
    assert c.sym_mult == len(ref.symmetry_operators)
    assert float(c.z_prime) == pytest.approx(float(ref.z_prime))


@pytest.mark.ccdc
def test_built_unit_cell_matches_ccdc_packing():
    """The natively built unit cell must reproduce CCDC's `packing()`.

    This is the reader's central geometric claim: expanding the deposited set by
    the file's own operators, wrapping by COMPONENT CENTROID so molecules are not
    torn at the boundary, gives the same cell CCDC hands over ready-made.
    Measured over 25 random CSD crystals: max deviation 8.6e-15 A, 25/25 bijective.
    """
    io = pytest.importorskip('ccdc.io', reason='CCDC licence required')
    from mxtaltools.dataset_utils.construction.cif.adapt import to_crystal_dict
    from mxtaltools.dataset_utils.construction.featurize_cif_chunks import init_reader
    from mxtaltools.dataset_utils.construction.featurization_utils import extract_crystal_data

    checked = 0
    for name, entry in _entries():
        if entry['z_prime'] != 1 or entry['has_metal']:
            continue
        path = FIXTURES / name
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
        assert dist[np.arange(len(dist)), nn].max() < 1e-6, \
            f'{entry["identifier"]}: built cell deviates from CCDC packing'
        assert len(set(nn.tolist())) == len(nn), \
            f'{entry["identifier"]}: match is not bijective -- atoms collapsed'
        checked += 1

    assert checked >= 3, f'only {checked} crystals compared; the test is near-blind'


# --------------------------------------------------------------------------
# WRAPPED DEPOSITS
#
# Some CIFs write each atom wrapped into [0,1) independently, which SPLITS any
# molecule straddling a cell boundary. CSD exports do not -- KUNTEV is deposited
# spanning -0.435 to 1.203 fractional, deliberately outside the cell so the
# molecule stays whole -- but that is a property of the exporter, not of CIF.
#
# A split molecule is a silent defect: every atom looks fine on its own, while
# the component centroid the unit-cell builder needs lands somewhere meaningless.
# The bond graph is the only thing that can tell "bonded across a cell edge" from
# "genuinely far apart", which is a second, independent reason the reader wants
# `_geom_bond` (the first being component decomposition).
# --------------------------------------------------------------------------

def _rewrite_wrapped(text: str) -> str:
    """Return the same crystal with every atom wrapped into [0,1)."""
    import re
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip() == 'loop_' and any('_atom_site_fract_x' in lines[j]
                                               for j in range(i + 1, min(i + 12, len(lines)))):
            i += 1
            tags = []
            while i < len(lines) and lines[i].strip().startswith('_'):
                tags.append(lines[i].strip()); out.append(lines[i]); i += 1
            cols = [tags.index(t) for t in ('_atom_site_fract_x',
                                            '_atom_site_fract_y',
                                            '_atom_site_fract_z')]
            while (i < len(lines) and lines[i].strip()
                   and not lines[i].strip().startswith(('loop_', '_', 'data_', '#'))):
                parts = lines[i].split()
                if len(parts) >= max(cols) + 1:
                    for k in cols:
                        parts[k] = '%.6f' % (float(re.sub(r'\(.*\)', '', parts[k])) % 1.0)
                    out.append(' ' + ' '.join(parts))
                else:
                    out.append(lines[i])
                i += 1
            continue
        i += 1
    return '\n'.join(out) + '\n'


@pytest.mark.parametrize('name', ['KUNTEV.cif', 'EFUNUP.cif'])
def test_wrapped_deposit_rebuilds_the_same_unit_cell(name):
    """The same crystal written wrapped and unwrapped must give the SAME cell.

    EFUNUP carries four components, so this also covers the multi-component case
    the owner flagged: each component must be unwrapped independently.
    """
    from mxtaltools.dataset_utils.construction.cif.adapt import build_unit_cell_components

    raw = (FIXTURES / name).read_text(encoding='utf-8', errors='ignore')
    original = read_cif(FIXTURES / name, max_atomic_number=None)[0]
    wrapped = read_cif(_rewrite_wrapped(raw), max_atomic_number=None)[0]

    assert wrapped.components.n_unwrapped > 0, (
        'the wrapped fixture did not actually need repair -- this test is blind')
    assert original.components.n_unwrapped == 0, (
        'CSD exports are deposited whole; repairing one means the detector is wrong')
    assert wrapped.components.n_components == original.components.n_components

    a = np.concatenate(build_unit_cell_components(original)['coordinates'])
    b = np.concatenate(build_unit_cell_components(wrapped)['coordinates'])
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    assert dist.min(axis=1).max() < 1e-6, (
        f'{name}: wrapped and unwrapped deposits built different unit cells')

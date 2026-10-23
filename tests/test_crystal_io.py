"""Contract tests for the batch constructors MolCrystalData.read and MolCrystalData.from_arrays.

CPU only, no licence. The reader half checks `read` against the native CIF reader's own
outputs on the 16 CIF fixtures, and pins one status code per kind of bad source. The
array half checks the bare and posed states, and the gate the ingestion contract sets for
`from_arrays`: on posed rows from a real dataset it must equal `collate_data_list` of the
same rows attribute for attribute and reproduce `mol2ucell` and
`analyze(['reduction_en', 'elj'])` to float32 rounding.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from mxtaltools.common.geometry_utils import coor_trans_matrix_np
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.dataset_utils.construction.cif import CifUnsupportedError, parse_cif, read_cif
from mxtaltools.dataset_utils.construction.cif.sites import sites_from_block
from mxtaltools.dataset_utils.data_class_methods.crystal_io import DONE_READ, READ_STATUS
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
MINI_CSD = Path(__file__).parent / 'datasets' / 'mini_new_csd.pt'
OK = READ_STATUS.index('ok')


def _manifest():
    return {e['identifier']: e for e in json.loads((FIXTURES / 'MANIFEST.json').read_text())}


def _fixture_paths():
    return sorted(FIXTURES.glob('*.cif'))


def _status(batch):
    return [READ_STATUS[s] for s in batch.status.tolist()]


def _cif(name='syn', *, ops=('x,y,z', '-x,-y,-z'), number=2, hm='P -1', hall='-P 1',
         cell=(10.0, 11.0, 12.0, 90.0, 95.0, 90.0), z_value=2,
         sites=(('C1', 'C', 0.10, 0.10, 0.10), ('C2', 'C', 0.20, 0.10, 0.10), ('H1', 'H', 0.25, 0.15, 0.10)),
         occupancy=None, bonds=(('C1', 'C2'), ('C2', 'H1')), bond_codes=None, moiety=None):
    """Minimal CIF text; each keyword set to None leaves its tag or loop out.

    ``bond_codes``: one (code_1, code_2) pair per bond, written as the site-symmetry columns.
    """
    lines = [f'data_{name}']
    if moiety is not None:
        lines.append(f"_chemical_formula_moiety '{moiety}'")
    if number is not None:
        lines.append(f'_symmetry_Int_Tables_number {number}')
    if hm is not None:
        lines.append(f"_symmetry_space_group_name_H-M '{hm}'")
    if hall is not None:
        lines.append(f"_space_group_name_Hall '{hall}'")
    if ops is not None:
        lines += ['loop_', '_symmetry_equiv_pos_as_xyz'] + [f"'{o}'" for o in ops]
    for tag, v in zip(('_cell_length_a', '_cell_length_b', '_cell_length_c',
                       '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma'), cell):
        if v is not None:
            lines.append(f'{tag} {v}')
    if z_value is not None:
        lines.append(f'_cell_formula_units_Z {z_value}')
    lines += ['loop_', '_atom_site_label', '_atom_site_type_symbol',
              '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
    if occupancy is not None:
        lines.append('_atom_site_occupancy')
    for i, s in enumerate(sites):
        lines.append(' '.join(str(v) for v in s) + (f' {occupancy[i]}' if occupancy is not None else ''))
    if bonds is not None:
        lines += ['loop_', '_geom_bond_atom_site_label_1', '_geom_bond_atom_site_label_2']
        if bond_codes is None:
            lines += [f'{a} {b}' for a, b in bonds]
        else:
            lines += ['_geom_bond_site_symmetry_1', '_geom_bond_site_symmetry_2']
            lines += [f'{a} {b} {ca} {cb}' for (a, b), (ca, cb) in zip(bonds, bond_codes)]
    return '\n'.join(lines) + '\n'


def _strip_operator_loop(text):
    """The same CIF with its symmetry-operator loop removed (the loop_ line, its tags, its rows)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == 'loop_' and i + 1 < len(lines) and lines[i + 1].strip().startswith('_symmetry_equiv_pos'):
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('_symmetry_equiv_pos'):
                j += 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(('_', 'loop_', '#')):
                j += 1
            return '\n'.join(lines[:i] + lines[j:]) + '\n'
    raise AssertionError('no operator loop found; the helper is blind')


def _row(batch, i):
    """Row i of a read batch as plain values, for row-to-row comparisons."""
    s, e = int(batch.ptr[i]), int(batch.ptr[i + 1])
    return dict(z=batch.z[s:e], pos=batch.pos[s:e], cell_lengths=batch.cell_lengths[i],
                cell_angles=batch.cell_angles[i], T_fc=batch.T_fc[i], sg_ind=batch.sg_ind[i],
                ops=batch.symmetry_operators[i], setting=batch.setting[i], status=batch.status[i],
                labels=batch.atom_labels[i], identifier=batch.identifier[i], z_value=batch.z_value[i])


def _same_row(a, b):
    for k in a:
        if torch.is_tensor(a[k]):
            assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
        elif isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


# ======================================================================================
# read
# ======================================================================================

def test_read_matches_the_native_reader_on_every_fixture():
    """One call on the 16 fixtures: cell, operators (file order), atoms and positions per row."""
    paths = _fixture_paths()
    assert len(paths) == 16, 'fixture set changed; the test is pinned to 16'
    manifest = _manifest()
    batch = MolCrystalData.read([str(p) for p in paths])

    assert type(batch).__name__ == 'MolCrystalDataBatch' and batch.num_graphs == 16
    assert _status(batch) == ['ok'] * 16
    assert batch.done.tolist() == [DONE_READ] * 16
    for i, p in enumerate(paths):
        ref = read_cif(p, require_integer_z_prime=False, max_atomic_number=None)[0]
        entry = manifest[p.stem]
        s, e = int(batch.ptr[i]), int(batch.ptr[i + 1])
        assert batch.identifier[i] == p.stem == ref.identifier
        assert batch.source_path[i] == str(p) and batch.block_name[i] == ref.block_name
        assert int(batch.num_atoms[i]) == ref.n_sites == entry['n_sites']
        assert batch.z[s:e].tolist() == ref.sites.z.tolist()
        assert batch.atom_labels[i] == ref.sites.labels

        # the file's own operators, in file order, never the table's
        ops = batch.symmetry_operators[i]
        assert ops.dtype == np.float64 and ops.shape == (ref.sym_mult, 4, 4)
        assert all(np.array_equal(a, b) for a, b in zip(ops, ref.symmetry.operators))
        assert int(batch.sym_mult[i]) == entry['n_ops'] and int(batch.sg_ind[i]) == entry['sg']
        assert (int(batch.setting[i]) == 0) == entry['ops_standard'] == ref.symmetry.operators_are_standard
        assert bool(batch.nonstandard_symmetry[i]) == (not entry['ops_standard'])

        np.testing.assert_allclose(batch.cell_lengths[i].double().numpy(), ref.cell_lengths, rtol=1e-6)
        np.testing.assert_allclose(batch.cell_angles[i].double().numpy(), ref.cell_angles, rtol=1e-6)
        # the deposited set as written: CSD exports are never unwrapped by the reader either
        assert ref.components.n_unwrapped == 0
        T = coor_trans_matrix_np('f_to_c', ref.cell_lengths, ref.cell_angles)
        np.testing.assert_allclose(batch.pos[s:e].double().numpy(), ref.sites.frac @ T.T, atol=2e-6)

        assert int(batch.z_value[i]) == int(entry['z_value'])
        assert batch.hall_symbol[i] is not None and batch.hm_symbol[i] is not None
        assert batch.moiety[i] and batch.moiety[i] == ref.formula_moiety
        bonds = batch.file_bonds[i]
        n_bond_rows = len(parse_cif(p.read_text(encoding='utf-8', errors='ignore'))[0]
                          .loop_with('_geom_bond_atom_site_label_1'))
        assert bonds.dtype == np.int64 and bonds.shape == (2, n_bond_rows)
        assert bonds.min() >= 0 and bonds.max() < ref.n_sites
        assert not bool(batch.has_occupancy[i]) and bool((batch.occupancy[s:e] == 1).all())

    # derived fields from the stored cell, and the declared-level bookkeeping
    from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
    T_fc, T_cf, vol = batch_compute_fractional_transform(batch.cell_lengths, batch.cell_angles)
    assert torch.equal(T_fc, batch.T_fc) and torch.equal(T_cf, batch.T_cf) and torch.equal(vol, batch.cell_volume)
    assert torch.equal(batch._slice_dict['pos'], batch.ptr) and torch.equal(batch._slice_dict['occupancy'], batch.ptr)
    assert torch.equal(batch._slice_dict['cell_lengths'], torch.arange(17))
    assert batch._inc_dict['symmetry_operators'] is None and torch.equal(batch._inc_dict['z'], torch.zeros(16).long())


def test_read_text_equals_path_and_float64():
    path = FIXTURES / 'NUKCOL.cif'
    from_path = MolCrystalData.read(path)
    from_text = MolCrystalData.read(path.read_text(encoding='utf-8', errors='ignore'))
    assert from_text.source_path == [None] and from_path.source_path == [str(path)]
    _same_row(_row(from_path, 0), _row(from_text, 0))

    wide = MolCrystalData.read(path, dtype=torch.float64)
    ref = read_cif(path)[0]
    T = coor_trans_matrix_np('f_to_c', ref.cell_lengths, ref.cell_angles)
    assert wide.pos.dtype == torch.float64 and wide.T_fc.dtype == torch.float64
    np.testing.assert_allclose(wide.pos.numpy(), ref.sites.frac @ T.T, atol=1e-10)


def test_read_upper_case_suffix_is_read_as_a_path(tmp_path):
    for name in ('NUKCOL.CIF', 'NUKCOL.Cif'):
        p = tmp_path / name
        shutil.copy(FIXTURES / 'NUKCOL.cif', p)
        batch = MolCrystalData.read(str(p))
        assert _status(batch) == ['ok'] and batch.source_path == [str(p)] and int(batch.num_atoms[0]) == 25


def test_read_missing_path_is_a_row():
    batch = MolCrystalData.read('no/such/dir/ABSENT.cif')
    assert _status(batch) == ['unreadable_file'] and batch.identifier == ['ABSENT']
    assert int(batch.num_atoms[0]) == 0 and batch.symmetry_operators[0].shape == (0, 4, 4)
    assert int(batch.sg_ind[0]) == 0 and int(batch.sym_mult[0]) == 0 and int(batch.setting[0]) == -1
    assert bool(torch.isnan(batch.cell_lengths).all()) and 'ABSENT' in batch.message[0]


def test_read_malformed_file_is_a_row(tmp_path):
    p = tmp_path / 'broken.cif'
    p.write_text('data_broken\nloop_\n_atom_site_label\n_atom_site_fract_x\nC1 0.1\nC2\n')   # ragged loop
    batch = MolCrystalData.read(p)
    assert _status(batch) == ['parse_error'] and int(batch.num_atoms[0]) == 0
    assert batch.identifier == ['broken'] and 'ragged' in batch.message[0]


def test_read_multi_block_file_gives_one_row_per_block(tmp_path):
    p = tmp_path / 'two.cif'
    p.write_text((FIXTURES / 'NUKCOL.cif').read_text(encoding='utf-8', errors='ignore') + '\n'
                 + (FIXTURES / 'CEKZOH.cif').read_text(encoding='utf-8', errors='ignore'))
    batch = MolCrystalData.read(p)
    assert _status(batch) == ['ok', 'ok'] and batch.block_name == ['NUKCOL', 'CEKZOH']
    assert batch.identifier == ['NUKCOL', 'CEKZOH'] and batch.num_atoms.tolist() == [25, 21]
    named = MolCrystalData.read(p, identifiers=['pair'])
    assert named.identifier == ['pair:NUKCOL', 'pair:CEKZOH']
    assert MolCrystalData.read(FIXTURES / 'NUKCOL.cif', identifiers=['fluorescein']).identifier == ['fluorescein']


def test_read_file_without_operator_loop_is_no_operators(tmp_path):
    """Operators are never guessed from the symbol or the IT number (owner: a symmetry-detection step later)."""
    p = tmp_path / 'NUKCOL_noops.cif'
    p.write_text(_strip_operator_loop((FIXTURES / 'NUKCOL.cif').read_text(encoding='utf-8', errors='ignore')))
    assert read_cif(FIXTURES / 'NUKCOL.cif')        # the source itself is fine
    batch = MolCrystalData.read(p)
    assert _status(batch) == ['no_operators'] and int(batch.num_atoms[0]) == 0
    assert 'never guessed' in batch.message[0]


def test_read_list_mixing_good_and_bad_sources(tmp_path):
    """One call, a status per row, and every good row identical to reading it alone."""
    bad = tmp_path / 'bad.cif'
    bad.write_text(_cif(name='bad', cell=(10.0, None, 12.0, 90.0, 95.0, 90.0)))      # no cell length b
    upper = tmp_path / 'KUNTEV.CIF'
    shutil.copy(FIXTURES / 'KUNTEV.cif', upper)
    sources = [FIXTURES / 'NUKCOL.cif', 'nowhere/MISSING.cif', 'not cif at all\nreally', str(upper),
               _cif(ops=None), str(bad), _cif(name='good_text'), FIXTURES / 'IDOVUT.cif']
    batch = MolCrystalData.read(sources)
    assert _status(batch) == ['ok', 'unreadable_file', 'parse_error', 'ok', 'no_operators', 'missing_data',
                              'ok', 'ok']
    assert batch.num_graphs == len(sources) and len(batch.message) == len(sources)
    for i in (0, 3, 6, 7):
        alone = MolCrystalData.read(sources[i])
        _same_row(_row(batch, i), _row(alone, 0))

    good = batch.select_rows(batch.status == OK)
    assert good.identifier == ['NUKCOL', 'KUNTEV', 'good_text', 'IDOVUT']
    assert torch.equal(good.pos, torch.cat([_row(batch, i)['pos'] for i in (0, 3, 6, 7)]))


@pytest.mark.parametrize('expected,kwargs,options', [
    ('ok', {}, {}),
    ('duplicate_label', dict(sites=(('C1', 'C', 0.1, 0.1, 0.1), ('C1', 'C', 0.2, 0.1, 0.1))), {}),
    ('bad_element', dict(sites=(('C1', 'Qq', 0.1, 0.1, 0.1),)), {}),
    # the status comes from the data, not from the message, which carries the block name
    ('bad_element', dict(name='duplicate_x', sites=(('C1', 'Qq', 0.1, 0.1, 0.1),), bonds=None), {}),
    # ASE's placeholder 'X' maps to atomic number 0, which no site can hold
    ('bad_element', dict(sites=(('C1', 'C', 0.1, 0.1, 0.1), ('X1', 'X', 0.2, 0.1, 0.1)), bonds=None), {}),
    ('bad_element', dict(sites=(('U1', 'U', 0.1, 0.1, 0.1),), bonds=None), dict(max_atomic_number=50)),
    ('bad_cell', dict(cell=(10, 11, 12, 90, 190, 90)), {}),
    ('bad_cell', dict(cell=(10, 11, 12, 10, 10, 170)), {}),        # angles of no real cell
    ('bad_cell', dict(cell=(0, 11, 12, 90, 95, 90)), {}),
    ('bad_cell', dict(cell=('nan', 11, 12, 90, 95, 90)), {}),
    ('missing_data', dict(cell=(None, 11, 12, 90, 95, 90)), {}),
    ('missing_data', dict(number=None, hm=None), {}),
    ('missing_data', dict(sites=(('C1', 'C', 0.1, '?', 0.1),), bonds=None), {}),
    ('no_operators', dict(ops=None), {}),
    ('inconsistent_symmetry', dict(number=14), {}),                 # 2 operators under a 4-operator group
    ('inconsistent_symmetry', dict(ops=('x,y,z', 'p,q,r')), {}),
    ('inconsistent_symmetry', dict(ops=('-x,-y,-z', '-x,y,-z')), dict()),   # no identity: refused, not fixed
    ('parse_error', dict(sites=(('C1', 'C', 0.1, 'abc', 0.1),), bonds=None), {}),
    # float() reads these, but they are not CIF numbers and no site sits there
    ('parse_error', dict(sites=(('C1', 'C', 0.1, 'nan', 0.1),), bonds=None), {}),
    ('parse_error', dict(sites=(('C1', 'C', 0.1, '1e400', 0.1),), bonds=None), {}),
    ('parse_error', dict(occupancy=(1.0, 'nan', 1.0)), {}),
    ('parse_error', dict(bonds=(('C1', 'C9'),)), {}),
    ('parse_error', dict(z_value='4.5'), {}),
    ('parse_error', dict(number='fourteen'), {}),
    ('partial_occupancy', dict(occupancy=(1.0, 0.5, 1.0)), {}),
])
def test_read_status_codes(expected, kwargs, options):
    batch = MolCrystalData.read(_cif(**kwargs), **options)
    assert _status(batch) == [expected], batch.message
    if expected in ('ok', 'partial_occupancy'):
        assert int(batch.num_atoms[0]) == len(kwargs.get('sites', (0, 0, 0)))
        assert int(batch.sg_ind[0]) == 2 and int(batch.setting[0]) == 0
    else:
        assert int(batch.num_atoms[0]) == 0 and int(batch.setting[0]) == -1
        assert batch.message[0], 'a refusal must say why'


def test_read_partial_occupancy_keeps_atoms_and_occupancy():
    batch = MolCrystalData.read(_cif(occupancy=(1.0, 0.5, 0.25)))
    assert _status(batch) == ['partial_occupancy'] and bool(batch.has_occupancy[0])
    assert batch.occupancy.tolist() == [1.0, 0.5, 0.25] and int(batch.num_atoms[0]) == 3


def test_read_does_not_require_z_bonds_or_hall():
    batch = MolCrystalData.read(_cif(z_value=None, bonds=None, hall=None))
    assert _status(batch) == ['ok'] and int(batch.z_value[0]) == -1
    assert batch.file_bonds == [None] and batch.hall_symbol == [None] and batch.hm_symbol == ['P -1']
    with_all = MolCrystalData.read(_cif())
    assert int(with_all.z_value[0]) == 2 and with_all.hall_symbol == ['-P 1']
    assert with_all.file_bonds[0].tolist() == [[0, 1], [1, 2]]


def test_read_bad_arguments_raise():
    with pytest.raises(ValueError, match='dtype'):       # csd_v1 reads float64; the default float32 conflicts
        MolCrystalData.read(FIXTURES / 'NUKCOL.cif', full='csd_v1')
    with pytest.raises(NotImplementedError):
        MolCrystalData.read(FIXTURES / 'NUKCOL.cif', full=True)
    with pytest.raises(ValueError):
        MolCrystalData.read(FIXTURES / 'NUKCOL.cif', format='xyz')
    with pytest.raises(TypeError, match='format option'):
        MolCrystalData.read(FIXTURES / 'NUKCOL.cif', require_bonds=False)
    with pytest.raises(ValueError, match='identifiers'):
        MolCrystalData.read([FIXTURES / 'NUKCOL.cif'], identifiers=['a', 'b'])
    with pytest.raises(TypeError):
        MolCrystalData.read(42)
    with pytest.raises(TypeError):
        MolCrystalData.read([FIXTURES / 'NUKCOL.cif', 42])
    with pytest.raises(ValueError):
        MolCrystalData.read([])
    with pytest.raises(TypeError, match='dtype'):
        MolCrystalData.read(FIXTURES / 'NUKCOL.cif', dtype=torch.long)


def test_sites_refuse_partial_occupancy_option():
    """The one reader change: keyword-only, default refuses exactly as before."""
    import inspect
    p = inspect.signature(sites_from_block).parameters['refuse_partial_occupancy']
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True
    block = parse_cif(_cif(occupancy=(1.0, 0.5, 1.0)))[0]
    with pytest.raises(CifUnsupportedError) as exc:
        sites_from_block(block, 'syn')
    assert exc.value.reason == 'partial-occupancy'
    sites = sites_from_block(block, 'syn', refuse_partial_occupancy=False)
    assert sites.has_occupancy and sites.occupancy.tolist() == [1.0, 0.5, 1.0]


def test_read_path_or_text_rule(tmp_path):
    """Bad paths of every kind are unreadable_file rows carrying their path: never parse_error, never a raise."""
    cases = [str(tmp_path), tmp_path, str(tmp_path / 'ABCDEF'), 'no_such_dir/structure.res',
             'no_such_dir/ABSENT.cif\n', 'abc\x00def.cif', '']
    batch = MolCrystalData.read(cases)
    assert _status(batch) == ['unreadable_file'] * len(cases), batch.message
    assert batch.source_path[:5] == [str(tmp_path), str(tmp_path), str(tmp_path / 'ABCDEF'),
                                     'no_such_dir/structure.res', 'no_such_dir/ABSENT.cif']
    assert batch.identifier[2:5] == ['ABCDEF', 'structure', 'ABSENT']

    fixture = FIXTURES / 'NUKCOL.cif'
    assert _status(MolCrystalData.read(str(fixture) + '\n')) == ['ok']        # a path read with its newline
    # one-line CIF text is still text: it starts with data_ and holds whitespace
    one_line = ' '.join(_cif(name='oneline').splitlines())
    inline = MolCrystalData.read(one_line)
    assert '\n' not in one_line and _status(inline) == ['ok'], inline.message
    assert inline.source_path == [None] and inline.identifier == ['oneline'] and int(inline.num_atoms[0]) == 3


def test_read_byte_order_mark(tmp_path):
    """Windows tools write a UTF-8 byte-order mark by default; it is not CIF content."""
    text = (FIXTURES / 'NUKCOL.cif').read_text(encoding='utf-8', errors='ignore')
    p = tmp_path / 'NUKCOL.cif'
    p.write_bytes(b'\xef\xbb\xbf' + text.encode('utf-8'))
    plain = MolCrystalData.read(FIXTURES / 'NUKCOL.cif')
    for batch in (MolCrystalData.read(p), MolCrystalData.read('\ufeff' + text)):
        assert _status(batch) == ['ok'], batch.message
        _same_row(_row(batch, 0), _row(plain, 0))


def test_read_file_bonds_site_symmetry_codes():
    """'.' is the CIF default code, which SHELXL writes within the deposited set: kept, like 1_555."""
    kept = MolCrystalData.read(_cif(bond_codes=(('.', '.'), ('1_555', '.'))))
    assert _status(kept) == ['ok'] and kept.file_bonds[0].tolist() == [[0, 1], [1, 2]] and kept.message == ['']
    image = MolCrystalData.read(_cif(bond_codes=(('.', '.'), ('1_555', '2_655'))))
    assert image.file_bonds[0].tolist() == [[0], [1]] and '1 _geom_bond row(s)' in image.message[0]
    unknown = MolCrystalData.read(_cif(bond_codes=(('?', '.'), ('.', '.'))))
    assert unknown.file_bonds[0].tolist() == [[1], [2]]


def test_read_moiety():
    assert MolCrystalData.read(_cif(moiety='C2 H1')).moiety == ['C2 H1']
    assert MolCrystalData.read(_cif()).moiety == [None]
    unknown = _cif().replace('data_syn\n', 'data_syn\n_chemical_formula_moiety ?\n')
    assert MolCrystalData.read(unknown).moiety == [None]
    assert MolCrystalData.read('nowhere/MISSING.cif').moiety == [None]


#: R -3 on rhombohedral axes, as the CSD files it (DIOXSP's operators, cell and Hall symbol); Z' = 1.
_R3_RHOMBOHEDRAL = dict(ops=('x,y,z', 'z,x,y', 'y,z,x', '-x,-y,-z', '-z,-x,-y', '-y,-z,-x'), number=148, hm='R -3',
                        hall='-P 3*', cell=(7.62, 7.62, 7.62, 91.2, 91.2, 91.2), z_value=6)


def test_read_reexpresses_an_r_group_on_rhombohedral_axes():
    """Refused as inconsistent_symmetry before the re-expression: 6 operators under a group SYM_OPS lists with 18."""
    batch = MolCrystalData.read(_cif(**_R3_RHOMBOHEDRAL), dtype=torch.float64)
    assert _status(batch) == ['ok'] and int(batch.setting[0]) == 0 and int(batch.sym_mult[0]) == 18
    assert batch.message[0].startswith('[syn] re-expressed from rhombohedral axes')
    assert np.array_equal(batch.symmetry_operators[0], np.stack(SYM_OPS[148]).astype(np.float64))
    assert int(batch.z_value[0]) == 18 and batch.hall_symbol == ['-R 3'] and batch.hm_symbol == ['R -3']
    a, alpha = 7.62, np.radians(91.2)
    expected = [2 * a * np.sin(alpha / 2)] * 2 + [a * np.sqrt(3 * (1 + 2 * np.cos(alpha)))]
    assert torch.allclose(batch.cell_lengths[0], torch.tensor(expected, dtype=torch.float64), rtol=1e-14, atol=0)
    assert torch.equal(batch.cell_angles[0], torch.tensor(np.radians([90.0, 90.0, 120.0])))
    assert abs(float(batch.cell_volume[0]) / (a ** 3 * np.sqrt(1 - 3 * np.cos(alpha) ** 2 + 2 * np.cos(alpha) ** 3))
               - 3) < 1e-12
    # the same atoms: their distances equal those on the file's own axes
    T_fc, _ = coor_trans_matrix_np('f_to_c', np.array([a] * 3), np.radians([91.2] * 3), return_vol=True)
    frac = np.array([s[2:] for s in (('C1', 'C', 0.10, 0.10, 0.10), ('C2', 'C', 0.20, 0.10, 0.10),
                                     ('H1', 'H', 0.25, 0.15, 0.10))])
    xyz = frac @ T_fc.T
    ours = batch.pos.numpy()
    assert np.abs(np.linalg.norm(ours[:, None] - ours[None], axis=-1)
                  - np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)).max() < 1e-12
    assert batch.file_bonds[0].tolist() == [[0, 1], [1, 2]]


@pytest.mark.parametrize('change,match', [
    (dict(cell=(7.62, 7.70, 7.62, 91.2, 91.2, 91.2)), 'is not rhombohedral'),
    (dict(hall='-R 3'), 'Hall symbol'),
])
def test_read_rhombohedral_count_that_does_not_reexpress_is_a_row_status(change, match):
    batch = MolCrystalData.read(_cif(**{**_R3_RHOMBOHEDRAL, **change}))
    assert _status(batch) == ['inconsistent_symmetry'] and match in batch.message[0]
    assert int(batch.num_atoms[0]) == 0 and int(batch.setting[0]) == -1


def test_read_rows_beside_a_reexpressed_row_are_unchanged():
    path = _fixture_paths()[0]
    both = MolCrystalData.read([_cif(**_R3_RHOMBOHEDRAL), path])
    alone = MolCrystalData.read(path)
    assert _status(both)[0] == 'ok' and both.message[1] == alone.message[0]
    _same_row(_row(both, 1), _row(alone, 0))


@pytest.mark.parametrize('cell', [_R3_RHOMBOHEDRAL['cell'], (13.0, 13.0, 9.0, 90.0, 90.0, 120.0)],
                         ids=['rhombohedral', 'hexagonal'])
def test_read_r_group_with_an_overflowing_operator_is_a_row_status(cell):
    """'+1e400' in an operator under an R group, on either axes: parse_error, as under P -1, never an exception
    out of read (the re-expression step once let the parser's OverflowError through)."""
    ops = _R3_RHOMBOHEDRAL['ops'][:5] + ('-y,-z,-x+1e400',)
    batch = MolCrystalData.read(_cif(**{**_R3_RHOMBOHEDRAL, 'ops': ops, 'cell': cell}))
    control = MolCrystalData.read(_cif(ops=('x,y,z', '-x,-y,-z+1e400')))
    assert _status(batch) == _status(control) == ['parse_error'] and 'OverflowError' in batch.message[0]


def test_read_reexpression_that_does_not_rebuild_is_a_row_status(monkeypatch):
    """A defect in the re-expression's coordinate transform (here P^-T for P^-1) is caught by its rebuild check
    and becomes inconsistent_symmetry, not a row on wrong coordinates."""
    from mxtaltools.dataset_utils.construction.cif import rhombohedral
    monkeypatch.setattr(rhombohedral, '_hexagonal_frac', lambda f: f @ rhombohedral._P_INV3 / 3)
    batch = MolCrystalData.read(_cif(**_R3_RHOMBOHEDRAL))
    assert _status(batch) == ['inconsistent_symmetry'] and 'does not rebuild' in batch.message[0]
    assert int(batch.num_atoms[0]) == 0


def test_select_rows_reads_levels_when_atoms_equal_rows():
    """24 failed rows (no atoms) and one of 25 atoms: N == B, where inferring a level from a length mixes rows."""
    fixture = FIXTURES / 'NUKCOL.cif'
    batch = MolCrystalData.read([f'missing/ROW{i:02d}.cif' for i in range(24)] + [fixture])
    assert batch.num_graphs == int(batch.ptr[-1]) == 25
    good = batch.select_rows(batch.status == OK)
    alone = MolCrystalData.read(fixture)
    assert good.num_graphs == 1 and set(good.keys()) == set(alone.keys())
    for k in alone.keys():
        assert _identical(good[k], alone[k]), k
    assert set(good._slice_dict) == set(alone._slice_dict)
    for k in alone._slice_dict:
        assert _identical(good._slice_dict[k], alone._slice_dict[k]), k
        assert (good._inc_dict[k] is None and alone._inc_dict[k] is None) or \
               _identical(good._inc_dict[k], alone._inc_dict[k]), k

    # rows reorder and repeat; a failed row keeps its shape
    picked = batch.select_rows([24, 3, 24])
    assert picked.identifier == ['NUKCOL', 'ROW03', 'NUKCOL'] and picked.num_atoms.tolist() == [25, 0, 25]
    assert picked.cell_lengths.shape == (3, 3) and bool(torch.isnan(picked.cell_lengths[1]).all())
    assert torch.equal(picked.pos, torch.cat([alone.pos, alone.pos]))


# ======================================================================================
# from_arrays
# ======================================================================================

def _fixture_arrays(names=('NUKCOL', 'CEKZOH', 'KUNTEV')):
    batch = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in names], dtype=torch.float64)
    cell = torch.cat([batch.cell_lengths, batch.cell_angles], dim=1)
    return batch, cell


def test_from_arrays_bare_equals_the_read_batch():
    read, cell = _fixture_arrays()
    bare = MolCrystalData.from_arrays(cell, read.pos, read.z, ptr=read.ptr, sg=read.sg_ind,
                                      ops=read.symmetry_operators, dtype=torch.float64,
                                      identifier=list(read.identifier))
    for k in ('z', 'pos', 'batch', 'ptr', 'num_atoms', 'cell_lengths', 'cell_angles', 'T_fc', 'T_cf',
              'cell_volume', 'sg_ind', 'sym_mult', 'setting', 'nonstandard_symmetry', 'status', 'done'):
        assert bare[k].dtype == read[k].dtype and torch.equal(bare[k], read[k]), k
    assert all(np.array_equal(a, b) for a, b in zip(bare.symmetry_operators, read.symmetry_operators))
    assert bare.identifier == read.identifier
    for k in ('max_z_prime', 'x', 'mol_ind', 'radius', 'mass', 'mol_volume', 'z_prime', 'aunit_centroid'):
        assert k not in bare.keys(), f'a bare batch must not carry {k}'


def test_from_arrays_lattice_rows_are_rotated_into_the_T_fc_frame():
    read, cell = _fixture_arrays()
    g = torch.Generator().manual_seed(0)
    q, r = torch.linalg.qr(torch.randn(3, 3, 3, generator=g, dtype=torch.float64))
    R = q * torch.sign(torch.linalg.det(q))[:, None, None]                  # proper rotations
    lattice_rows = (R @ read.T_fc).transpose(1, 2)                          # row i = rotated lattice vector i
    atom_row = read.batch
    xyz_rotated = torch.einsum('nij,nj->ni', R[atom_row], read.pos)
    out = MolCrystalData.from_arrays(lattice_rows, xyz_rotated, read.z, ptr=read.ptr, sg=read.sg_ind,
                                     ops=read.symmetry_operators, dtype=torch.float64)
    assert _status(out) == ['ok'] * 3
    torch.testing.assert_close(out.cell_lengths, read.cell_lengths, rtol=0, atol=1e-10)
    torch.testing.assert_close(out.cell_angles, read.cell_angles, rtol=0, atol=1e-10)
    torch.testing.assert_close(out.pos, read.pos, rtol=0, atol=1e-9)

    # a left-handed basis cannot be rotated onto T_fc: a status, coordinates left as given
    mirrored = lattice_rows.clone()
    mirrored[1] = -mirrored[1]
    out = MolCrystalData.from_arrays(mirrored, xyz_rotated, read.z, ptr=read.ptr, sg=read.sg_ind,
                                     ops=read.symmetry_operators, dtype=torch.float64)
    assert _status(out) == ['ok', 'bad_cell', 'ok'] and 'right-handed' in out.message[1]
    s, e = int(read.ptr[1]), int(read.ptr[2])
    assert torch.equal(out.pos[s:e], xyz_rotated[s:e])


def test_from_arrays_bad_cells_are_row_statuses():
    read, cell = _fixture_arrays()
    cell = cell.clone()
    cell[0, 1] = -3.0
    cell[2, 3:] = torch.deg2rad(torch.tensor([10.0, 10.0, 170.0], dtype=torch.float64))
    out = MolCrystalData.from_arrays(cell, read.pos, read.z, ptr=read.ptr, sg=read.sg_ind,
                                     ops=read.symmetry_operators)
    assert _status(out) == ['bad_cell', 'ok', 'bad_cell']
    assert 'positive' in out.message[0] and 'no real cell' in out.message[2]


def test_from_arrays_setting_codes():
    n, xyz, z, ptr = 4, torch.rand(4, 3), torch.full((4,), 6), torch.arange(5)
    cell = torch.tensor([[8.0, 9.0, 10.0, 1.5708, 1.8, 1.5708]]).repeat(n, 1)
    std = np.stack(SYM_OPS[14])
    reordered = std[[0, 2, 1, 3]]
    shifted = std.copy()
    shifted[1, :3, 3] += 1.0                                           # same element, lattice-shifted
    p21n = np.stack([np.eye(4), std[1], std[2], std[3]])
    p21n[1, :3, 3] = [0.5, 0.5, 0.5]                                   # -x+1/2, y+1/2, -z+1/2: P 1 21/n 1
    p21n[3, :3, 3] = [0.5, 0.5, 0.5]
    out = MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=14, ops=[std, reordered, shifted, p21n])
    assert out.setting.tolist() == [0, 1, 1, 2] and out.setting.dtype == torch.int8
    assert out.nonstandard_symmetry.tolist() == [False, True, True, True]

    standard = MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=[14, 2, 19, 14], setting='standard')
    assert standard.setting.tolist() == [0] * 4 and standard.sym_mult.tolist() == [4, 2, 4, 4]
    assert all(np.array_equal(o, np.stack(SYM_OPS[s])) for o, s in zip(standard.symmetry_operators, [14, 2, 19, 14]))
    shared = MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=14, ops=std)
    assert shared.setting.tolist() == [0] * 4


def test_from_arrays_operator_and_shape_arguments_raise():
    cell, xyz, z, ptr = torch.tensor([[8.0, 9.0, 10.0, 1.5708, 1.8, 1.5708]] * 2), torch.rand(3, 3), torch.ones(3), \
        torch.tensor([0, 1, 3])
    ops = np.stack(SYM_OPS[2])
    with pytest.raises(ValueError, match='ops are required'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2)
    with pytest.raises(ValueError, match='not both'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, ops=ops, setting='standard')
    with pytest.raises(ValueError, match='setting'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='reexpress')
    with pytest.raises(ValueError, match='outside'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=231, ops=ops)
    with pytest.raises(ValueError, match='one operator array per row'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, ops=[ops])
    with pytest.raises(ValueError, match=r'\[m >= 1, 4, 4\]'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, ops=[ops, ops[:, :3]])
    with pytest.raises(ValueError, match='ptr'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=torch.tensor([1, 1, 3]), sg=2, ops=ops)
    with pytest.raises(ValueError, match='ptr'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=torch.tensor([0, 2, 1]), sg=2, ops=ops)
    with pytest.raises(ValueError, match='xyz'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=torch.tensor([0, 1, 2]), sg=2, ops=ops)
    with pytest.raises(ValueError, match='z must'):
        MolCrystalData.from_arrays(cell, xyz, z[:2], ptr=ptr, sg=2, ops=ops)
    with pytest.raises(ValueError, match='cell'):
        MolCrystalData.from_arrays(cell[:1], xyz, z, ptr=ptr, sg=2, ops=ops)
    with pytest.raises(TypeError, match='dtype'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, ops=ops, dtype=torch.int32)


def _two_unit_rows():
    """Three rows: Z'=1 (3 atoms), Z'=2 (two 3-atom units), and a Z'=2 row of 5 atoms (no split)."""
    unit = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.0, 0.9, 0.0]])
    z_unit = torch.tensor([6, 6, 1])
    xyz = torch.cat([unit, unit + 3.0, unit @ torch.tensor([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]) - 1.0,
                     unit[:2] + 5.0, unit])
    z = torch.cat([z_unit, z_unit, z_unit, z_unit[:2], z_unit])
    ptr = torch.tensor([0, 3, 9, 14])
    cell = torch.tensor([[8.0, 9.0, 10.0, 1.5708, 1.8, 1.5708]]).repeat(3, 1)
    return cell, xyz, z, ptr, unit, z_unit


def test_from_arrays_posed_units_and_row_sums():
    cell, xyz, z, ptr, unit, z_unit = _two_unit_rows()
    out = MolCrystalData.from_arrays(
        cell, xyz, z, ptr=ptr, sg=2, setting='standard',
        z_prime=torch.tensor([1, 2, 2]),
        aunit_centroid=torch.full((3, 9), 0.5), aunit_orientation=torch.ones(3, 9),
        aunit_handedness=torch.ones(3, 3, dtype=torch.long),
        is_well_defined=[True, True, False], identifier=['a', 'b', 'c'])
    assert _status(out) == ['ok', 'ok', 'z_prime_mismatch'] and "Z' = 2" in out.message[2]
    assert out.max_z_prime == 2 and type(out.max_z_prime) is int
    # truncated to max Z' slots; slots past each row's Z' padded with 0, as collate pads
    assert out.aunit_centroid.shape == (3, 6) and out.aunit_handedness.shape == (3, 2)
    assert out.aunit_centroid[0].tolist() == [0.5, 0.5, 0.5, 0, 0, 0]
    # the z_prime_mismatch row has no units, so none of its slots is used
    assert out.aunit_handedness.tolist() == [[1, 0], [1, 1], [0, 0]] and out.aunit_handedness.dtype == torch.long
    assert out.mol_ind.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    assert out.x.dtype == torch.float32 and bool((out.x == 0).all())
    assert out.is_well_defined.dtype == torch.bool and out.is_well_defined.tolist() == [True, True, False]
    assert out.done.tolist() == [DONE_READ] * 3

    # row sums of the per-unit values, with MolData.mol_analysis's single-molecule definitions
    def single(pos, zz):
        m = MolData(z=zz, pos=pos)
        return torch.stack([m.radius_calculation(), m.mass_calculation(), m.volume_calculation()])

    rot = torch.tensor([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    expect = [single(unit, z_unit), single(unit + 3.0, z_unit) + single(unit @ rot - 1.0, z_unit)]
    for b in range(2):
        got = torch.stack([out.radius[b], out.mass[b], out.mol_volume[b]])
        torch.testing.assert_close(got, expect[b], rtol=2e-6, atol=1e-5)
    assert bool(torch.isnan(torch.stack([out.radius[2], out.mass[2], out.mol_volume[2]])).all())


def test_from_arrays_pose_arguments_raise():
    cell, xyz, z, ptr, _, _ = _two_unit_rows()
    pose = dict(z_prime=torch.tensor([1, 2, 1]), aunit_centroid=torch.zeros(3, 6),
                aunit_orientation=torch.zeros(3, 6), aunit_handedness=torch.ones(3, 2))
    partial = {k: v for k, v in pose.items() if k != 'aunit_handedness'}
    with pytest.raises(ValueError, match='partial pose'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard', **partial)
    narrow = dict(pose, aunit_centroid=torch.zeros(3, 3))
    with pytest.raises(ValueError, match='aunit_centroid'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard', **narrow)
    with pytest.raises(ValueError, match='computed'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard', radius=torch.ones(3), **pose)
    with pytest.raises(ValueError, match='computed'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard', sg_ind=torch.ones(3))
    with pytest.raises(ValueError, match='leading dimension'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard', density=torch.ones(4))
    with pytest.raises(ValueError, match='entries'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard', smiles=['C', 'CC'])
    with pytest.raises(ValueError, match='leading dimension'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard',
                                   atom_attrs={'charge': torch.ones(3)})
    with pytest.raises(ValueError, match='offset'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard',
                                   atom_attrs={'aunit_batch': torch.zeros(14)})


def test_from_arrays_declared_levels_set_the_slice_bookkeeping():
    cell, xyz, z, ptr, _, _ = _two_unit_rows()
    labels = [str(i) for i in range(14)]
    out = MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard',
                                     tag=['a', 'b', 'c'], flag=[True, False, True], count=[1, 2, 3],
                                     score=np.array([0.5, 1.5, 2.5]),
                                     atom_attrs={'charge': torch.arange(14.0), 'label': labels,
                                                 'occ': [1.0] * 14})
    assert out.tag == ['a', 'b', 'c'] and out.flag.dtype == torch.bool and out.count.dtype == torch.long
    assert out.score.dtype == torch.float64 and out.charge.tolist() == list(range(14))
    # a per-atom list of strings is held as collate holds rows' lists: one list per row, row level
    assert out.label == [labels[0:3], labels[3:9], labels[9:14]]
    assert out.occ.dtype == torch.float32 and out.occ.shape == (14,)          # python numbers: a per-atom tensor
    graph, node = torch.arange(4), ptr
    for k in ('tag', 'flag', 'count', 'score', 'sg_ind', 'status', 'message', 'label'):
        assert torch.equal(out._slice_dict[k], graph), k
    for k in ('charge', 'occ', 'z', 'pos'):
        assert torch.equal(out._slice_dict[k], node), k
    assert out._inc_dict['tag'] is None and out._inc_dict['label'] is None
    assert torch.equal(out._inc_dict['charge'], torch.zeros(3, dtype=torch.long))

    # row access splits every level correctly: PyG's separate, and select_rows
    for i, row in enumerate(out.to_data_list()):
        s, e = int(ptr[i]), int(ptr[i + 1])
        assert row.label == labels[s:e] and row.charge.tolist() == list(range(s, e)) and row.tag == 'abc'[i]
    assert out.get_example(1).label == labels[3:9]
    picked = out.select_rows([2, 0])
    assert picked.label == [labels[9:14], labels[0:3]] and picked.tag == ['c', 'a']
    assert picked.charge.tolist() == list(range(9, 14)) + [0, 1, 2]


def _posed_subset(order, cell, xyz, z, ptr, pose):
    """from_arrays on rows ``order`` of the arrays, each row's atoms and pose carried with it."""
    blocks = [torch.arange(int(ptr[i]), int(ptr[i + 1])) for i in order]
    atoms = torch.cat(blocks)
    counts = torch.tensor([len(b) for b in blocks])
    return MolCrystalData.from_arrays(
        cell[order], xyz[atoms], z[atoms], ptr=torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]),
        sg=torch.tensor([2, 14, 2])[order], setting='standard', identifier=[f'r{i}' for i in order],
        **{k: v[order] for k, v in pose.items()},
        atom_attrs={'label': [f'a{int(j)}' for j in atoms], 'charge': atoms.double()})


def test_select_rows_equals_building_the_subset():
    cell, xyz, z, ptr, _, _ = _two_unit_rows()
    g = torch.Generator().manual_seed(1)
    pose = dict(z_prime=torch.tensor([1, 2, 1]), aunit_centroid=torch.rand(3, 6, generator=g),
                aunit_orientation=torch.rand(3, 6, generator=g), aunit_handedness=torch.ones(3, 2))
    full = _posed_subset([0, 1, 2], cell, xyz, z, ptr, pose)
    want = _posed_subset([2, 0, 1], cell, xyz, z, ptr, pose)
    got = full.select_rows(torch.tensor([2, 0, 1]))
    assert _status(full) == ['ok'] * 3 and got.max_z_prime == want.max_z_prime == 2
    assert set(got.keys()) == set(want.keys())
    for k in want.keys():
        assert _identical(got[k], want[k]), k
    assert set(got._slice_dict) == set(want._slice_dict)
    for k in want._slice_dict:
        assert _identical(got._slice_dict[k], want._slice_dict[k]), k
        assert (got._inc_dict[k] is None and want._inc_dict[k] is None) or \
               _identical(got._inc_dict[k], want._inc_dict[k]), k


def test_select_rows_bad_arguments_raise():
    batch = MolCrystalData.read([FIXTURES / 'NUKCOL.cif', 'missing/X.cif'])
    with pytest.raises(ValueError, match='mask'):
        batch.select_rows(torch.tensor([True]))
    with pytest.raises(ValueError, match='lie in'):
        batch.select_rows([2])
    with pytest.raises(ValueError, match='lie in'):
        batch.select_rows([-1])
    with pytest.raises(ValueError, match='integers'):
        batch.select_rows([0.5])
    assert batch.select_rows([]).num_graphs == 0
    if MINI_CSD.exists():
        # a collated dataset batch carries aunit_batch, which collation offsets per row
        collated = collate_data_list([r.clone() for r in torch.load(MINI_CSD, weights_only=False)[:2]])
        with pytest.raises(ValueError, match='increments'):
            collated.select_rows([1])


def test_from_arrays_pose_with_lattice_rows_raises():
    """The pose is defined in the T_fc frame; lattice rows rotate xyz, not the pose."""
    cell, xyz, z, ptr, _, _ = _two_unit_rows()
    lattice = torch.diag_embed(cell[:, :3])
    pose = dict(z_prime=torch.tensor([1, 2, 1]), aunit_centroid=torch.zeros(3, 6),
                aunit_orientation=torch.zeros(3, 6), aunit_handedness=torch.ones(3, 2))
    with pytest.raises(ValueError, match='T_fc frame'):
        MolCrystalData.from_arrays(lattice, xyz, z, ptr=ptr, sg=2, setting='standard', **pose)
    assert _status(MolCrystalData.from_arrays(lattice, xyz, z, ptr=ptr, sg=2, setting='standard')) == ['ok'] * 3


def test_from_arrays_row_values_give_statuses_not_raises():
    """Non-finite coordinates and atomic numbers outside the tables flag their row; other rows are untouched."""
    cell, xyz, z, ptr, _, _ = _two_unit_rows()                      # rows of 3, 6 and 5 atoms
    pose = dict(z_prime=torch.tensor([1, 2, 1]), aunit_centroid=torch.zeros(3, 6),
                aunit_orientation=torch.zeros(3, 6), aunit_handedness=torch.ones(3, 2))

    def props(b):
        return torch.stack([b.radius, b.mass, b.mol_volume])

    clean = MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard', **pose)
    assert _status(clean) == ['ok'] * 3 and not bool(torch.isnan(props(clean)).any())

    nan_xyz = xyz.clone()
    nan_xyz[0, 0] = float('nan')                                     # a heavy atom of row 0
    out = MolCrystalData.from_arrays(cell, nan_xyz, z, ptr=ptr, sg=2, setting='standard', **pose)
    assert _status(out) == ['bad_coordinates', 'ok', 'ok'] and '[0]' in out.message[0]
    assert bool(torch.isnan(props(out)[:, 0]).all())
    assert torch.equal(props(out)[:, 1:], props(clean)[:, 1:])

    for bad in (-1, 0, 100, 119):
        zz = z.clone()
        zz[4] = bad                                                  # an atom of row 1
        out = MolCrystalData.from_arrays(cell, xyz, zz, ptr=ptr, sg=2, setting='standard', **pose)
        assert _status(out) == ['ok', 'bad_element', 'ok'] and str(bad) in out.message[1]
        assert bool(torch.isnan(props(out)[:, 1]).all())
        assert torch.equal(props(out)[:, [0, 2]], props(clean)[:, [0, 2]])
        bare = MolCrystalData.from_arrays(cell, xyz, zz, ptr=ptr, sg=2, setting='standard')
        assert _status(bare) == ['ok', 'bad_element', 'ok']


def test_from_arrays_operator_content_is_checked_per_row():
    """The checks read applies to a file's operators, as row statuses; a centred setting is accepted as read accepts it."""
    n, xyz, z, ptr = 6, torch.rand(6, 3), torch.full((6,), 6), torch.arange(7)
    cell = torch.tensor([[8.0, 9.0, 10.0, 1.5708, 1.8, 1.5708]]).repeat(n, 1)
    p1bar = np.stack(SYM_OPS[2])
    nan_ops = p1bar.copy()
    nan_ops[1, 0, 3] = np.nan
    not_affine = p1bar.copy()
    not_affine[1, 3, 0] = 0.5
    centred = p1bar.copy()
    centred[:, :3, 3] += [0.5, 0.5, 0.0]
    ops = [p1bar, nan_ops, p1bar[1:], np.stack(SYM_OPS[14]), not_affine, np.concatenate([p1bar, centred])]
    out = MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, ops=ops)
    assert _status(out) == ['ok'] + ['inconsistent_symmetry'] * 4 + ['ok'], out.message
    assert 'non-finite' in out.message[1] and 'identity' in out.message[2]
    assert 'centred' in out.message[3] and 'affine' in out.message[4]
    assert out.sym_mult.tolist() == [2, 2, 1, 4, 2, 4] and int(out.setting[5]) == 2


def test_from_arrays_bookkeeping_arguments_raise():
    cell, xyz, z, ptr, _, _ = _two_unit_rows()
    kw = dict(sg=2, setting='standard')
    with pytest.raises(ValueError, match='one-dimensional'):
        MolCrystalData.from_arrays(cell[:2], xyz, z, ptr=torch.tensor([[0, 3], [9, 14]]), **kw)
    with pytest.raises(ValueError, match='ptr must hold integers'):
        MolCrystalData.from_arrays(cell[:1], xyz[:1], z[:1], ptr=torch.tensor([False, True]), **kw)
    with pytest.raises(ValueError, match='sg must hold integers'):
        MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=True, setting='standard')
    with pytest.raises(ValueError, match='z must hold integer'):
        MolCrystalData.from_arrays(cell, xyz, z.bool(), ptr=ptr, **kw)
    z_inf = z.double()
    z_inf[0] = float('inf')
    with pytest.raises(ValueError, match='z must hold integer'):
        MolCrystalData.from_arrays(cell, xyz, z_inf, ptr=ptr, **kw)


def test_from_arrays_a_bad_z_prime_does_not_set_the_batch_width():
    """A z_prime_mismatch row neither makes the call raise nor widens max_z_prime; its slots are zero."""
    cell, xyz, z, ptr, _, _ = _two_unit_rows()                      # rows of 3, 6 and 5 atoms
    narrow = dict(aunit_centroid=torch.ones(3, 6), aunit_orientation=torch.ones(3, 6),
                  aunit_handedness=torch.ones(3, 2))
    out = MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard',
                                     z_prime=torch.tensor([1, 2, 7]), **narrow)
    assert _status(out) == ['ok', 'ok', 'z_prime_mismatch'] and out.max_z_prime == 2
    assert out.z_prime.tolist() == [1, 2, 7]
    assert out.aunit_centroid[2].tolist() == [0.0] * 6 and out.aunit_handedness[2].tolist() == [0.0, 0.0]
    wide = dict(aunit_centroid=torch.ones(3, 18), aunit_orientation=torch.ones(3, 18),
                aunit_handedness=torch.ones(3, 6))
    out = MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=2, setting='standard',
                                     z_prime=torch.tensor([1, 1, 6]), **wide)
    assert _status(out) == ['ok', 'ok', 'z_prime_mismatch'] and out.max_z_prime == 1
    assert out.aunit_centroid.shape == (3, 3) and out.aunit_handedness.shape == (3, 1)


# --------------------------------------------------------------------------------------
# The gate: posed rows of a real dataset, against collate_data_list of the same rows
# --------------------------------------------------------------------------------------

#: Attributes allowed to differ, and why. Everything else both batches define must be
#: identical: name, python type, dtype, shape, values, _slice_dict and _inc_dict entries.
GATE_TOLERANCES = {
    # recomputed from the stored cell by batch_compute_fractional_transform; the stored
    # values came from another evaluation and differ by 1 float32 ulp (measured 1.5e-8)
    'T_cf': dict(rtol=1e-6, atol=1e-7),
    # recomputed per unit by the batch path and summed per row; the stored values are the
    # featurizer's per-molecule sums (measured on all 100 rows: 8.5e-4 mass, 2.9e-6 A radius, 1.2e-4 A^3 volume)
    'radius': dict(rtol=2e-6, atol=1e-5),
    'mass': dict(rtol=2e-6, atol=1e-5),
    'mol_volume': dict(rtol=2e-6, atol=1e-5),
}
#: computed as zeros by from_arrays (charges are the features step's output, which has not
#: run; the contract's posed state), while featurized rows carry Gasteiger charges
GATE_ZEROS = {'x'}
#: present only on from_arrays batches. 'unit_ind', the unit map (crystal_building.utils.UNIT_MAP_KEY), since stage 3 of
#: the units plan (2026-10-23): a posed from_arrays batch writes its pose's layout (Z' equal contiguous blocks) as the
#: map, and the collated reference rows (mini_new_csd.pt, written before stage 2) carry none; every reader gives
#: those rows the same layout (crystal_building.utils.unit_layout), so the two batches still describe the same units
GATE_NEW_ONLY = {'setting', 'status', 'message', 'done', 'unit_ind'}
#: present only on the collated rows: an old-schema per-atom float whose name ('batch')
#: makes collation offset it per row; from_arrays refuses such names, so it is not passed
GATE_REF_ONLY = {'aunit_batch'}


def _gate_rows(kind):
    """Every row of the file ('all'), or every Z' = 1 row ('zp1', so max_z_prime is 1)."""
    if not MINI_CSD.exists():
        pytest.skip(f'{MINI_CSD} absent')
    rows = torch.load(MINI_CSD, weights_only=False)
    zp = [int(r.z_prime.reshape(-1)[0]) for r in rows]
    assert zp.count(2) >= 2, "the gate needs Z' = 2 rows"
    return rows if kind == 'all' else [r for r, k in zip(rows, zp) if k == 1]


def _from_rows(rs):
    counts = torch.tensor([int(r.num_atoms) for r in rs])
    return MolCrystalData.from_arrays(
        torch.cat([torch.cat([r.cell_lengths, r.cell_angles], 1) for r in rs]),
        torch.cat([r.pos for r in rs]), torch.cat([r.z for r in rs]),
        ptr=torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]),
        sg=torch.stack([r.sg_ind.reshape(()) for r in rs]),
        ops=[r.symmetry_operators for r in rs],
        z_prime=torch.cat([r.z_prime.reshape(1) for r in rs]),
        aunit_centroid=torch.cat([r.aunit_centroid for r in rs]),        # [B, 18]: truncated to max Z'
        aunit_orientation=torch.cat([r.aunit_orientation for r in rs]),
        aunit_handedness=torch.cat([r.aunit_handedness for r in rs]),
        identifier=[r.identifier for r in rs], smiles=[r.smiles for r in rs],
        fingerprint=torch.cat([r.fingerprint for r in rs]),
        cocrystal=[r.cocrystal for r in rs], is_well_defined=[r.is_well_defined for r in rs],
        density=torch.cat([r.density for r in rs]), packing_coeff=torch.cat([r.packing_coeff for r in rs]))


def _identical(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


@pytest.mark.parametrize('kind', ['zp1', 'all'])
def test_from_arrays_matches_collate_on_dataset_rows(kind):
    rs = _gate_rows(kind)
    ref = collate_data_list([r.clone() for r in rs])
    new = _from_rows(rs)
    if kind == 'all':
        # the allowances below must be exercised, not vacuous
        assert bool((ref.x != 0).any()) and bool((ref.nonstandard_symmetry != new.nonstandard_symmetry).any())

    assert type(new) is type(ref)
    assert set(new.__dict__) == set(ref.__dict__)
    assert new.num_graphs == ref.num_graphs
    assert set(new.keys()) - set(ref.keys()) == GATE_NEW_ONLY
    assert set(ref.keys()) - set(new.keys()) == GATE_REF_ONLY
    assert _status(new) == ['ok'] * len(rs)
    assert type(new.max_z_prime) is int and new.max_z_prime == ref.max_z_prime == (1 if kind == 'zp1' else 2)

    for k in sorted(set(ref.keys()) & set(new.keys())):
        a, b = ref[k], new[k]
        assert type(a) is type(b), k
        if k in GATE_TOLERANCES:
            assert a.dtype == b.dtype and a.shape == b.shape, k
            torch.testing.assert_close(b, a, **GATE_TOLERANCES[k], msg=k)
        elif k in GATE_ZEROS:
            assert a.dtype == b.dtype and a.shape == b.shape and not bool(b.any()), k
        elif k == 'nonstandard_symmetry':
            # the stored flag came from EXACT float equality with SYM_OPS; setting 0 allows 1e-6, so rows
            # whose operators equal the table to 3.3e-9 read as standard here. Every other row is identical.
            differ = torch.nonzero(a != b).reshape(-1).tolist()
            assert a.dtype == b.dtype and len(differ) <= 2, differ     # BOGFUA and ESOZEU in this file
            for i in differ:
                ops, std = new.symmetry_operators[i], np.stack(SYM_OPS[int(new.sg_ind[i])])
                assert bool(a[i]) and not bool(b[i])
                assert np.allclose(ops, std, atol=1e-6) and not np.array_equal(ops, std)
        else:
            assert _identical(a, b), k
        if k in ref._slice_dict:
            assert _identical(ref._slice_dict[k], new._slice_dict[k]), f'_slice_dict[{k}]'
            assert (ref._inc_dict[k] is None and new._inc_dict[k] is None) or \
                   _identical(ref._inc_dict[k], new._inc_dict[k]), f'_inc_dict[{k}]'
    assert set(new._slice_dict) - set(ref._slice_dict) == GATE_NEW_ONLY

    # the same crystals: unit cell and energy to float32 rounding
    c_ref, c_new = ref.clone(), new.clone()
    c_ref.mol2ucell()
    c_new.mol2ucell()
    assert torch.equal(c_ref.unit_cell_batch, c_new.unit_cell_batch)
    torch.testing.assert_close(c_new.unit_cell_pos, c_ref.unit_cell_pos, rtol=0, atol=1e-4)
    e_ref = ref.clone().analyze(['reduction_en', 'elj'], cutoff=6.0)
    e_new = new.clone().analyze(['reduction_en', 'elj'], cutoff=6.0)
    for k in ('reduction_en', 'elj'):
        torch.testing.assert_close(e_new[k], e_ref[k], rtol=1e-5, atol=1e-3, msg=k)

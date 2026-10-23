"""Contract tests for the ingest store: MolCrystalData.save / MolCrystalData.load.

CPU only. A store must rebuild the saved batch exactly: every stored attribute with its
python type, dtype, shape and bits (NaN included), the store order, ``_num_graphs``,
``_slice_dict`` and ``_inc_dict``. Checked on a read batch mixing good and failed rows
(all 16 CIF fixtures plus refused sources), on a posed from_arrays batch of every row of
``mini_new_csd.pt`` (Z' = 1 and Z' = 2), on one-row batches, on attributes of every
kind a later step may add, and on rows holding every code of the status table
(``status_table()``, the steps' codes included). The data file must load with
``weights_only=True``; an existing path is refused, and so is a corrupted file on load,
and a status code outside the table on save.
"""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mxtaltools.dataset_utils.data_class_methods import crystal_store
from mxtaltools.dataset_utils.data_class_methods.crystal_io import READ_STATUS
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import STATUS
from mxtaltools.dataset_utils.data_class_methods.crystal_store import (SCHEMA, StoreError, code_digest,
                                                                        read_meta, status_table)
from mxtaltools.dataset_utils.data_classes import MolCrystalData

FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
MINI_CSD = Path(__file__).parent / 'datasets' / 'mini_new_csd.pt'
OK = READ_STATUS.index('ok')
#: the status table the store names codes by (READ_STATUS is its prefix)
TABLE = status_table()


def _text_cif(name, *, ops=('x,y,z', '-x,-y,-z'), occupancy=None, bonds=True):
    """A small P-1 CIF as text; ``ops=None`` drops the operator loop."""
    lines = [f'data_{name}', '_symmetry_Int_Tables_number 2']
    if ops is not None:
        lines += ['loop_', '_symmetry_equiv_pos_as_xyz'] + [f"'{o}'" for o in ops]
    lines += ['_cell_length_a 10.0', '_cell_length_b 11.0', '_cell_length_c 12.0',
              '_cell_angle_alpha 90.0', '_cell_angle_beta 95.0', '_cell_angle_gamma 90.0',
              'loop_', '_atom_site_label', '_atom_site_type_symbol',
              '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
    if occupancy is not None:
        lines.append('_atom_site_occupancy')
    sites = (('C1', 'C', 0.1, 0.1, 0.1), ('C2', 'C', 0.2, 0.1, 0.1), ('H1', 'H', 0.25, 0.15, 0.1))
    for i, s in enumerate(sites):
        lines.append(' '.join(map(str, s)) + (f' {occupancy[i]}' if occupancy is not None else ''))
    if bonds:
        lines += ['loop_', '_geom_bond_atom_site_label_1', '_geom_bond_atom_site_label_2', 'C1 C2', 'C2 H1']
    return '\n'.join(lines) + '\n'


def _mixed_read_batch():
    """All 16 fixtures with refused sources between them: every read-level attribute kind, None included."""
    fixtures = sorted(FIXTURES.glob('*.cif'))
    assert len(fixtures) == 16
    sources = fixtures[:5] + ['nowhere/MISSING.cif', 'not cif\nat all', _text_cif('no_ops', ops=None),
                              _text_cif('partial', occupancy=(1.0, 0.5, 1.0)), _text_cif('no_bonds', bonds=False)]
    sources += fixtures[5:]
    return MolCrystalData.read(sources)


def _same(a, b) -> bool:
    """Exactly equal: python type, dtype, shape and bits (NaN equal to NaN); containers recursively."""
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return (a.dtype == b.dtype and a.shape == b.shape and a.device == b.device
                and a.detach().cpu().numpy().tobytes() == b.detach().cpu().numpy().tobytes())
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def _assert_same_batch(saved, loaded):
    assert type(loaded) is type(saved)
    assert list(loaded._store.keys()) == list(saved._store.keys())
    for k in saved._store.keys():
        assert _same(saved._store[k], loaded._store[k]), k
    kept = lambda d: {k for k in d if k != '_ptr_cpu_pair'}
    assert kept(loaded.__dict__) == kept(saved.__dict__)
    assert type(loaded.__dict__['_num_graphs']) is int and loaded.num_graphs == saved.num_graphs
    for name in ('_slice_dict', '_inc_dict'):
        assert _same(saved.__dict__[name], loaded.__dict__[name]), name


def _round_trip(batch, path, **kw):
    meta = batch.save(path, **kw)
    return meta, MolCrystalData.load(path)


# ======================================================================================
# exact round trips
# ======================================================================================

def test_read_batch_with_failed_rows_round_trips(tmp_path):
    batch = _mixed_read_batch()
    names = [READ_STATUS[s] for s in batch.status.tolist()]
    assert names.count('ok') == 17 and {'unreadable_file', 'parse_error', 'no_operators',
                                        'partial_occupancy'} <= set(names)
    assert any(b is None for b in batch.file_bonds) and any(b is not None for b in batch.file_bonds)
    assert None in batch.source_path and bool(torch.isnan(batch.cell_lengths).any())
    before = batch.clone()

    steps = [{'step': 'read', 'args': {'format': 'cif', 'dtype': 'float32'}}]
    meta, loaded = _round_trip(batch, tmp_path / 'store', steps_run=steps, extra_meta={'note': 'mixed'})
    _assert_same_batch(batch, loaded)
    _assert_same_batch(before, batch)                         # save does not touch the batch

    # the loaded batch is usable: the row filter gives the same rows
    _assert_same_batch(batch.select_rows(batch.status == OK), loaded.select_rows(loaded.status == OK))

    # meta: counts, levels, steps, extra; the operators packed as one [M, 4, 4] tensor with a pointer
    assert meta == read_meta(tmp_path / 'store')
    assert meta['schema'] == SCHEMA and meta['rows'] == batch.num_graphs and meta['atoms'] == int(batch.ptr[-1])
    assert meta['status_names'] == list(TABLE) and meta['status_counts'] == {n: names.count(n) for n in TABLE}
    assert meta['steps_run'] == steps and meta['extra'] == {'note': 'mixed'}
    assert meta['levels']['pos']['level'] == 'atom' and meta['levels']['cell_lengths']['level'] == 'row'
    assert meta['levels']['symmetry_operators']['encoding'] == 'array_list'
    assert meta['levels']['ptr']['level'] == 'bookkeeping'
    data = torch.load(tmp_path / 'store' / 'data.pt', weights_only=True)
    ops = data['attrs']['symmetry_operators']
    assert ops['values'].shape == (int(batch.sym_mult.sum()), 4, 4) and ops['values'].dtype == torch.float64
    assert ops['ptr'].tolist() == [0] + batch.sym_mult.cumsum(0).tolist()
    bonds = data['attrs']['file_bonds']                       # [2, E] per row: packed along the edge axis
    assert bonds['layout'] == 'axis' and bonds['axis'] == 1 and bonds['none'].tolist() == [b is None for b in
                                                                                          batch.file_bonds]

    # the manifest names every row
    with open(tmp_path / 'store' / 'manifest.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert [r['status'] for r in rows] == names and [r['identifier'] for r in rows] == batch.identifier
    assert [r['message'] for r in rows] == batch.message
    assert [r['source_path'] for r in rows] == ['' if p is None else p for p in batch.source_path]


def _posed_mini_batch():
    if not MINI_CSD.exists():
        pytest.skip(f'{MINI_CSD} absent')
    rows = torch.load(MINI_CSD, weights_only=False)
    counts = torch.tensor([int(r.num_atoms) for r in rows])
    return MolCrystalData.from_arrays(
        torch.cat([torch.cat([r.cell_lengths, r.cell_angles], 1) for r in rows]),
        torch.cat([r.pos for r in rows]), torch.cat([r.z for r in rows]),
        ptr=torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]),
        sg=torch.stack([r.sg_ind.reshape(()) for r in rows]), ops=[r.symmetry_operators for r in rows],
        z_prime=torch.cat([r.z_prime.reshape(1) for r in rows]),
        aunit_centroid=torch.cat([r.aunit_centroid for r in rows]),
        aunit_orientation=torch.cat([r.aunit_orientation for r in rows]),
        aunit_handedness=torch.cat([r.aunit_handedness for r in rows]),
        identifier=[r.identifier for r in rows], smiles=[r.smiles for r in rows],
        fingerprint=torch.cat([r.fingerprint for r in rows]), is_well_defined=[r.is_well_defined for r in rows],
        density=torch.cat([r.density for r in rows]), packing_coeff=torch.cat([r.packing_coeff for r in rows]))


def test_posed_from_arrays_batch_round_trips(tmp_path):
    batch = _posed_mini_batch()
    assert batch.max_z_prime == 2 and int((batch.z_prime == 2).sum()) >= 2 and bool((batch.z_prime == 1).any())
    meta, loaded = _round_trip(batch, tmp_path / 'posed')
    _assert_same_batch(batch, loaded)
    assert type(loaded.max_z_prime) is int and meta['levels']['max_z_prime'] == dict(level='batch',
                                                                                      encoding='scalar')
    # the same crystals: the unit cell builds identically from the loaded batch
    a, b = batch.clone(), loaded.clone()
    a.mol2ucell()
    b.mol2ucell()
    assert torch.equal(a.unit_cell_pos, b.unit_cell_pos)


def test_one_row_batches_round_trip(tmp_path):
    read = MolCrystalData.read(FIXTURES / 'NUKCOL.cif')
    _assert_same_batch(read, _round_trip(read, tmp_path / 'read1')[1])
    failed = MolCrystalData.read('nowhere/GONE.cif')
    _assert_same_batch(failed, _round_trip(failed, tmp_path / 'failed1')[1])
    posed = MolCrystalData.from_arrays(torch.tensor([[8.0, 9.0, 10.0, 1.5708, 1.8, 1.5708]]),
                                       torch.rand(3, 3), torch.tensor([6, 6, 1]), ptr=torch.tensor([0, 3]),
                                       sg=14, setting='standard', z_prime=torch.tensor([1]),
                                       aunit_centroid=torch.rand(1, 3), aunit_orientation=torch.rand(1, 3),
                                       aunit_handedness=torch.ones(1, 1))
    _assert_same_batch(posed, _round_trip(posed, tmp_path / 'posed1')[1])


def test_later_step_attributes_round_trip(tmp_path):
    """Attributes of every kind the store keeps, as later steps may add them, with None entries."""
    batch = MolCrystalData.read([FIXTURES / 'NUKCOL.cif', 'nowhere/X.cif', FIXTURES / 'CEKZOH.cif'])
    batch.add_graph_attr(torch.tensor([0.5, float('nan'), -1.0], dtype=torch.float64), 'score')
    batch.unit_graphs = [np.array([[0, 1], [1, 2]]), None, np.zeros((2, 0), dtype=np.int64)]  # one varying axis
    batch.blocks = [np.ones((2, 3), dtype=np.float32), np.zeros((4, 1), dtype=np.float32), None]  # two: flat
    batch.scalars = [np.array(1.5), np.array(2.0), None]                                  # rank 0: flat
    batch.unit_tensors = [torch.arange(4).reshape(2, 2), None, torch.arange(2).reshape(1, 2)]
    batch.unit_smiles = [('CC', 'CO'), None, ('C',)]                                      # tuples of str
    batch.empty_labels = [[], [], []]
    batch.flags = (True, False, True)
    batch.mixed = [1, None, 2.5]
    batch.table = np.arange(6, dtype=np.int16).reshape(2, 3)
    batch.config = {'tol': 0.1, 'names': ['a', 'b'], 'nested': {'k': (1, 2)}}
    batch.threshold = 0.25
    batch.recipe = 'csd_v1'
    _assert_same_batch(batch, _round_trip(batch, tmp_path / 'later')[1])
    data = torch.load(tmp_path / 'later' / 'data.pt', weights_only=True)
    assert data['attrs']['blocks']['layout'] == 'flat' and data['attrs']['unit_graphs']['axis'] == 1


def test_rows_of_every_status_code_round_trip(tmp_path):
    """A row may hold any code of the status table, a step's included; meta.json and the manifest name it by it."""
    assert TABLE[:len(STATUS)] == STATUS and STATUS[:len(READ_STATUS)] == READ_STATUS
    mixed = _mixed_read_batch()
    batch = mixed.select_rows(torch.arange(len(TABLE)) % mixed.num_graphs)
    batch.status.copy_(torch.arange(len(TABLE), dtype=batch.status.dtype))
    meta, loaded = _round_trip(batch, tmp_path / 'every')
    _assert_same_batch(batch, loaded)
    assert meta['status_names'] == list(TABLE) and meta['status_counts'] == {n: 1 for n in TABLE}
    with open(tmp_path / 'every' / 'manifest.csv', newline='', encoding='utf-8') as f:
        assert [r['status'] for r in csv.DictReader(f)] == list(TABLE)


def test_a_later_status_table_must_extend_the_one_before():
    steps = ('crystal_steps', STATUS)
    assert crystal_store._extend_status([steps]) == STATUS
    assert crystal_store._extend_status([steps, ('crystal_chem', STATUS + ('later_code',))])[-1] == 'later_code'
    with pytest.raises(RuntimeError, match='appended'):
        crystal_store._extend_status([steps, ('crystal_chem', ('ok', 'later_code'))])


# ======================================================================================
# the file format and refusals
# ======================================================================================

_SAFE = (dict, list, tuple, str, int, float, bool, type(None), torch.Tensor)


def _walk(value, depth=0):
    """Every leaf type in a loaded data file (bounded depth)."""
    assert depth < 12
    if isinstance(value, dict):
        for k, v in value.items():
            yield type(k)
            yield from _walk(v, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v, depth + 1)
    else:
        yield type(value)


def test_data_file_loads_with_weights_only(tmp_path):
    _mixed_read_batch().save(tmp_path / 'store')
    data = torch.load(tmp_path / 'store' / 'data.pt', weights_only=True)
    assert data['schema'] == SCHEMA and set(data) == {'schema', 'num_graphs', 'keys', 'attrs', 'slice_dict',
                                                      'inc_dict'}
    leaves = set(_walk(data))
    assert leaves <= set(_SAFE), leaves - set(_SAFE)
    assert np.ndarray not in leaves


def test_save_refuses_an_existing_path(tmp_path):
    batch = MolCrystalData.read(FIXTURES / 'NUKCOL.cif')
    (tmp_path / 'taken').mkdir()
    with pytest.raises(FileExistsError):
        batch.save(tmp_path / 'taken')
    assert list((tmp_path / 'taken').iterdir()) == []
    batch.save(tmp_path / 'store')
    stamp = (tmp_path / 'store' / 'meta.json').read_bytes()
    with pytest.raises(FileExistsError):
        batch.save(str(tmp_path / 'store'))
    assert (tmp_path / 'store' / 'meta.json').read_bytes() == stamp
    assert sorted(p.name for p in tmp_path.iterdir()) == ['store', 'taken']        # no partial folder left


def test_load_refuses_a_corrupted_file(tmp_path):
    MolCrystalData.read([FIXTURES / 'NUKCOL.cif', 'nowhere/X.cif']).save(tmp_path / 'store')
    data = tmp_path / 'store' / 'data.pt'
    raw = bytearray(data.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    data.write_bytes(bytes(raw))
    with pytest.raises(StoreError, match='sha256'):
        MolCrystalData.load(tmp_path / 'store')

    MolCrystalData.read(FIXTURES / 'NUKCOL.cif').save(tmp_path / 'other')
    manifest = tmp_path / 'other' / 'manifest.csv'
    manifest.write_text(manifest.read_text(encoding='utf-8').replace('NUKCOL', 'NUKCOX'), encoding='utf-8')
    with pytest.raises(StoreError, match='manifest.csv'):
        MolCrystalData.load(tmp_path / 'other')
    assert MolCrystalData.load(tmp_path / 'other', verify=False).identifier == ['NUKCOL']


def test_load_refuses_another_schema_or_class(tmp_path):
    MolCrystalData.read(FIXTURES / 'NUKCOL.cif').save(tmp_path / 'store')
    meta_path = tmp_path / 'store' / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta_path.write_text(json.dumps(dict(meta, batch_class='MolDataBatch')), encoding='utf-8')
    with pytest.raises(StoreError, match='MolDataBatch'):
        MolCrystalData.load(tmp_path / 'store')
    meta_path.write_text(json.dumps(dict(meta, schema='crystal_fd.dataset/1')), encoding='utf-8')
    with pytest.raises(StoreError, match='schema'):
        MolCrystalData.load(tmp_path / 'store')
    with pytest.raises(FileNotFoundError):
        MolCrystalData.load(tmp_path / 'absent')


def test_save_refuses_what_it_cannot_store(tmp_path):
    batch = MolCrystalData.read([FIXTURES / 'NUKCOL.cif', 'nowhere/X.cif'])
    for name, value in (('owner', object()), ('labels', {'a', 'b'}), ('mixed_arrays', [np.zeros(2), torch.zeros(2)]),
                        ('dtypes', [np.zeros(2), np.zeros(2, dtype=np.int64)]),
                        ('objects', [np.array(['a'], dtype=object), None]),
                        ('sparse', torch.eye(2).to_sparse())):
        b = batch.clone()
        setattr(b, name, value)
        with pytest.raises(StoreError, match=name):
            b.save(tmp_path / name)
        assert not (tmp_path / name).exists()
    assert list(tmp_path.iterdir()) == []                                          # nothing partial left

    with pytest.raises(TypeError, match='steps_run'):
        batch.save(tmp_path / 's', steps_run={'step': 'read'})
    with pytest.raises(ValueError, match="'step'"):
        batch.save(tmp_path / 's', steps_run=[{'args': {}}])
    with pytest.raises(ValueError, match='JSON'):
        batch.save(tmp_path / 's', steps_run=[{'step': 'read', 'args': {'tol': float('nan')}}])
    with pytest.raises(TypeError, match='extra_meta'):
        batch.save(tmp_path / 's', extra_meta=['a'])
    with pytest.raises(StoreError, match='batch'):
        batch.get_example(0).save(tmp_path / 's')
    for code in (len(TABLE), -1):
        b = batch.clone()
        b.status[1] = code
        with pytest.raises(StoreError, match='status table'):
            b.save(tmp_path / 's')
    if MINI_CSD.exists():
        from mxtaltools.dataset_utils.utils import collate_data_list
        collated = collate_data_list([r.clone() for r in torch.load(MINI_CSD, weights_only=False)[:2]])
        with pytest.raises(StoreError, match='status'):
            collated.save(tmp_path / 's')
    assert list(tmp_path.iterdir()) == []


def test_meta_records_the_code_and_versions(tmp_path):
    meta = MolCrystalData.read(FIXTURES / 'NUKCOL.cif').save(tmp_path / 'store')
    files = meta['code']['files']
    methods = 'dataset_utils/data_class_methods/'
    for f in (methods + 'crystal_io.py', methods + 'crystal_store.py', methods + 'crystal_steps.py',
              'dataset_utils/construction/cif/reader.py', 'dataset_utils/construction/cif/symmetry.py'):
        assert f in files, f
    here = Path(crystal_store.__file__).parent
    for later in ('crystal_chem.py', 'crystal_recipes.py'):              # hashed exactly when the module exists
        assert (methods + later in files) == (here / later).is_file(), later
        assert (later in meta['code']['scope']) == (here / later).is_file(), later
    assert meta['code'] == code_digest()
    assert meta['versions']['torch'] == torch.__version__ and set(meta['versions']) == {'torch', 'rdkit', 'spglib',
                                                                                        'ccdc'}
    for name, record in meta['files'].items():
        assert record['bytes'] == os.path.getsize(tmp_path / 'store' / name)
    assert meta['done_counts'] == {'1': 1} and meta['device'] == 'cpu'

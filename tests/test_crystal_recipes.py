"""Contract tests for crystal_recipes.py: recipes, run_chain, read(full=...) and canonicalize.

CPU only. Most tests run a CCDC-free recipe (``sanitize(chemistry='none', hydrogens='as_read')``)
on the 16 CIF fixtures; the CSD recipe itself runs where the CSD Python API imports (skipped
otherwise). ``canonicalize`` is checked against the importer and ``canon.canonicalize`` called
directly, and through a ``crystal_fd.dataset`` round trip; ``off_sheet='enantiomorph'`` against
the D2 product (columns, references, smiles stereo), alone and with ``near_face='nudge'``, its column
docs' scope, and the smiles stereo rule on a local csd_v1 chunk store (opened read-only; skipped
when absent). The product's ``features_fallback`` column carries the batch's flag (0 without one),
on a flag written onto the fixtures and on the rows of that chunk store the fallback recovers.
"""
import copy
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mxtaltools.dataset_utils.data_class_methods import crystal_recipes as cr
from mxtaltools.dataset_utils.data_class_methods.crystal_chem import FEATURES_FALLBACK, MolCrystalChem
from mxtaltools.dataset_utils.data_class_methods.crystal_chem import STATUS as CHEM_TABLE
from mxtaltools.dataset_utils.data_class_methods.crystal_io import DONE_READ
from mxtaltools.dataset_utils.data_class_methods.crystal_recipes import CHAIN, CSD_V1, FDProduct, Recipe, get_recipe
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import (DONE_FIND_MOLECULES, DONE_POSE, DONE_REBUILD,
                                                                       DONE_SANITIZE, DONE_VALIDITY)
from mxtaltools.dataset_utils.data_class_methods.crystal_store import status_table
from mxtaltools.dataset_utils.data_classes import MolCrystalData

FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
PATHS = sorted(str(p) for p in FIXTURES.glob('*.cif'))
F64 = torch.float64
CANON = dict(setting='reexpress', sheet=1, off_sheet='mirror_reference', near_face='flag', zp_gt1='refuse')

#: CSD_V1 with CCDC-free chemistry and no features: every structural step, no licence needed.
CHEAP = Recipe('cheap_test', (
    ('read', dict(format='cif', dtype=F64, max_atomic_number=100)),
    ('find_molecules', dict(bonds='file', bond_scale=(1.15, 1.35), iso_cap=5000, special_tol=0.1)),
    ('sanitize', dict(chemistry='none', hydrogens='as_read', on_protonation_change='flag')),
    ('parameterize_pose', dict(handedness='measured', degenerate_tol=1e-4, handedness_tol=1e-3)),
    ('check_rebuild', dict(max_dev=0.05, mean_dev=0.01, match_elements=True)),
    ('check_validity', dict(max_atomic_number=100, max_heavy_atoms=100, heavy_nn_range=(0.9, 3.0), occupancy=1e-3,
                            disorder=None, cell_constraints=1e-4, min_contact=0.5, composition='heavy')),
))


def _have_ccdc():
    from mxtaltools.dataset_utils.data_class_methods import crystal_chem
    return crystal_chem._backends()[0] is not None


needs_ccdc = pytest.mark.skipif(not _have_ccdc(), reason='CSD Python API unavailable or unlicensed')


def _names(batch):
    table = status_table()
    return [table[int(s)] for s in batch.status.tolist()]


def _same(a, b):
    if torch.is_tensor(a):
        if not (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape):
            return False
        if a.is_floating_point():
            return bool(torch.equal(torch.isnan(a), torch.isnan(b))) and bool(torch.equal(a[~torch.isnan(a)],
                                                                                         b[~torch.isnan(b)]))
        return bool(torch.equal(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def assert_same_batch(a, b):
    """Every stored attribute, in order, and the bookkeeping, equal (NaN equal to NaN)."""
    assert list(a._store.keys()) == list(b._store.keys())
    for k in a._store.keys():
        assert _same(a._store[k], b._store[k]), k
    assert a.__dict__.get('_num_graphs') == b.__dict__.get('_num_graphs')
    for name in ('_slice_dict', '_inc_dict'):
        da, db = a.__dict__[name], b.__dict__[name]
        assert da.keys() == db.keys(), name
        for k in da:
            assert _same(da[k], db[k]), (name, k)


def _snapshot(batch):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in batch._store.items()}


@pytest.fixture(scope='module')
def cheap():
    return MolCrystalData.read(PATHS, dtype=F64).run_chain(CHEAP)


# ======================================================================================
# recipes
# ======================================================================================

def test_csd_v1_lists_every_argument_of_every_step_in_chain_order():
    assert CSD_V1.step_names == CHAIN
    assert 'to_standard_setting' not in CSD_V1.step_names
    for step in CHAIN[1:]:
        params = inspect.signature(getattr(MolCrystalData, step)).parameters.values()
        assert set(CSD_V1.args(step)) == {p.name for p in params if p.kind == p.KEYWORD_ONLY}, step
    assert CSD_V1.args('read') == dict(format='cif', dtype=F64, max_atomic_number=100)
    assert CSD_V1.args('sanitize') == dict(chemistry='ccdc', hydrogens='regenerate', on_protonation_change='flag')
    assert CSD_V1.args('find_molecules')['bonds'] == 'file'
    assert CSD_V1.args('check_validity')['composition'] == 'heavy'
    assert CSD_V1.args('features') == dict(fingerprint=dict(radius=2, n_bits=2048, chirality=False),
                                           charges='gasteiger')


def test_recipe_is_frozen_and_its_args_are_copies():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CSD_V1.name = 'other'
    args = CSD_V1.args('features')
    args['fingerprint']['radius'] = 9
    assert CSD_V1.args('features')['fingerprint']['radius'] == 2
    with pytest.raises(TypeError):
        CSD_V1.steps[-1][1]['fingerprint']['radius'] = 9            # read-only mapping
    assert get_recipe('csd_v1') is CSD_V1 and get_recipe(CSD_V1) is CSD_V1


def test_recipe_spec_is_json_and_the_digest_names_it():
    spec = CSD_V1.spec()
    text = json.dumps(spec, sort_keys=True, separators=(',', ':'))
    assert json.loads(text) == spec and spec['steps'][0]['args']['dtype'] == 'float64'
    assert CSD_V1.digest == hashlib.sha256(text.encode()).hexdigest()
    assert [s['step'] for s in spec['steps']] == list(CHAIN)


def test_recipe_digest_is_stable_and_sensitive():
    same = Recipe('csd_v1', [(s, CSD_V1.args(s)) for s in CSD_V1.step_names])
    assert same == CSD_V1 and same.digest == CSD_V1.digest and hash(same) == hash(CSD_V1)
    args = [(s, CSD_V1.args(s)) for s in CSD_V1.step_names]
    args[4][1]['max_dev'] = 0.06
    assert Recipe('csd_v1', args).digest != CSD_V1.digest
    assert Recipe('other', [(s, CSD_V1.args(s)) for s in CSD_V1.step_names]).digest != CSD_V1.digest


@pytest.mark.parametrize('steps, error, match', [
    ([], ValueError, 'non-empty'),
    ([('find_molecules', CSD_V1.args('find_molecules'))], ValueError, "start with 'read'"),
    ([('read', CSD_V1.args('read')), ('sanitize', CSD_V1.args('sanitize')),
      ('find_molecules', CSD_V1.args('find_molecules'))], ValueError, 'out of chain order'),
    ([('read', CSD_V1.args('read')), ('check_rebuild', CSD_V1.args('check_rebuild')),
      ('check_rebuild', CSD_V1.args('check_rebuild'))], ValueError, 'repeated'),
    ([('read', CSD_V1.args('read')), ('dedup', {})], ValueError, 'unknown step'),
    ([('read', CSD_V1.args('read')), ('canonicalize', {})], ValueError, 'derived product'),
    ([('read', CSD_V1.args('read')), ('to_standard_setting', {})], NotImplementedError, 'Q4'),
    ([('read', CSD_V1.args('read')), ('check_rebuild', dict(max_dev=0.05, mean_dev=0.01))], ValueError,
     "missing \\['match_elements'\\]"),
    ([('read', CSD_V1.args('read')), ('check_rebuild', dict(CSD_V1.args('check_rebuild'), extra=1))], ValueError,
     "unknown \\['extra'\\]"),
    ([('read', dict(format='cif', dtype=F64))], ValueError, 'max_atomic_number'),
    ([('read', dict(format='xyz', dtype=F64, max_atomic_number=100))], ValueError, 'format'),
    ([('read', dict(format='cif', dtype='float64', max_atomic_number=100))], ValueError, 'dtype'),
    ([('read', dict(format='cif', dtype=torch.long, max_atomic_number=100))], ValueError, 'dtype'),
    ([('read', CSD_V1.args('read')), ('check_rebuild', dict(max_dev=0.05, mean_dev=0.01, match_elements={1}))],
     TypeError, 'recipe argument'),
    ([('read', CSD_V1.args('read')), ('check_rebuild', [0.05])], TypeError, 'dict'),
])
def test_bad_recipes_raise(steps, error, match):
    with pytest.raises(error, match=match):
        Recipe('bad', steps)


def test_recipe_name_is_required():
    with pytest.raises(ValueError, match='name'):
        Recipe('', [('read', CSD_V1.args('read'))])


def test_get_recipe_refuses_what_names_no_recipe():
    with pytest.raises(NotImplementedError, match='Q7'):
        get_recipe(True)
    with pytest.raises(ValueError, match='unknown recipe'):
        get_recipe('csd_v0')
    for bad in (False, 1, None, {'name': 'csd_v1'}):
        with pytest.raises(TypeError):
            get_recipe(bad)


# ======================================================================================
# read(full=...) and run_chain
# ======================================================================================

def test_read_full_must_name_a_recipe_and_match_its_read_arguments():
    with pytest.raises(NotImplementedError, match='Q7'):
        MolCrystalData.read(PATHS[0], full=True)
    with pytest.raises(TypeError):
        MolCrystalData.read(PATHS[0], full=False)
    with pytest.raises(ValueError, match='unknown recipe'):
        MolCrystalData.read(PATHS[0], full='nope')
    with pytest.raises(ValueError, match='dtype'):                       # csd_v1 reads float64; the default is float32
        MolCrystalData.read(PATHS[0], full='csd_v1')
    with pytest.raises(ValueError, match='max_atomic_number'):
        MolCrystalData.read(PATHS[0], dtype=F64, full='csd_v1', max_atomic_number=90)
    assert _names(MolCrystalData.read(PATHS[0])) == ['ok']             # full=None: read only, as before
    assert int(MolCrystalData.read(PATHS[0]).done[0]) == DONE_READ


def test_read_full_equals_read_then_run_chain(cheap):
    """The same rows; only the record differs: read(full=...) checked its read arguments, run_chain could not."""
    full = MolCrystalData.read(PATHS, dtype=F64, full=CHEAP)
    want = copy.deepcopy(list(cheap.steps_run))
    want[-1]['args']['read_checked'] = True
    assert full.steps_run == want
    full['steps_run'] = cheap.steps_run
    assert_same_batch(full, cheap)


def test_run_chain_records_whether_its_read_arguments_were_checked():
    """A batch read with other options than the recipe's: run_chain cannot tell, so it must not claim it checked."""
    other = MolCrystalData.read(PATHS[:2], dtype=F64, max_atomic_number=5)
    assert _names(other) == ['bad_element', 'bad_element']                 # carbon is 6
    record = other.run_chain(CHEAP).steps_run[-1]['args']
    assert record['read'] == dict(format='cif', dtype='float64', max_atomic_number=100)
    assert record['read_checked'] is False
    assert MolCrystalData.read(PATHS[:2], dtype=F64, full=CHEAP).steps_run[-1]['args']['read_checked'] is True
    with pytest.raises(ValueError, match='max_atomic_number'):
        MolCrystalData.read(PATHS[:2], dtype=F64, max_atomic_number=5, full=CHEAP)


def _without_moiety(tmp_path, name='CEKZOH'):
    """A copy of a fixture whose _chemical_formula_moiety is '?' (read gives moiety None)."""
    lines = (FIXTURES / f'{name}.cif').read_text(encoding='utf-8', errors='ignore').split('\n')
    hit = [i for i, line in enumerate(lines) if line.lower().startswith('_chemical_formula_moiety')]
    assert len(hit) == 1 and len(lines[hit[0]].split()) > 1                # the value is on the same line
    lines[hit[0]] = '_chemical_formula_moiety ?'
    out = tmp_path / f'{name}_nomoiety.cif'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return str(out)


def test_a_rows_validity_does_not_depend_on_the_other_rows_of_its_batch(tmp_path):
    """check_validity(composition=...) raises for a batch none of whose ok rows holds a moiety; run_chain must give
    such a row, alone, exactly what it gets beside a moiety-bearing row: ok, composition unchecked, the same record."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_steps import VALIDITY_CRITERIA
    bare = _without_moiety(tmp_path)
    fresh = MolCrystalData.read([bare], dtype=F64)
    assert fresh.moiety == [None]
    staged = fresh.find_molecules(**CHEAP.args('find_molecules')).sanitize(**CHEAP.args('sanitize')) \
        .parameterize_pose(**CHEAP.args('parameterize_pose')).check_rebuild(**CHEAP.args('check_rebuild'))
    with pytest.raises(ValueError, match='composition'):                 # the step's own batch-level refusal
        staged.check_validity(**CHEAP.args('check_validity'))
    alone = MolCrystalData.read([bare], dtype=F64, full=CHEAP)
    pair = MolCrystalData.read([bare, str(FIXTURES / 'CEKZOH.cif')], dtype=F64, full=CHEAP)
    assert _names(alone) == ['ok'] and _names(pair) == ['ok', 'ok']
    composition = 1 << VALIDITY_CRITERIA.index('composition')
    assert int(alone.validity_checked[0]) & composition == 0 and int(pair.validity_checked[1]) & composition
    assert alone.steps_run == pair.steps_run
    assert alone.steps_run[-2] == dict(step='check_validity', args=dict(
        max_atomic_number=100, max_heavy_atoms=100, heavy_nn_range=[0.9, 3.0], occupancy=1e-3, cell_constraints=1e-4,
        min_contact=0.5, composition='heavy'))
    first = pair.select_rows([0])
    for k in alone._store.keys():
        if k not in ('steps_run', 'max_z_prime'):
            assert _same(alone._store[k], first._store[k]), k


def test_run_chain_runs_every_step_and_records_the_recipe(cheap):
    steps = [s['step'] for s in cheap.steps_run]
    assert steps == list(CHEAP.step_names[1:]) + ['run_chain']
    assert cheap.steps_run[-1]['args'] == dict(recipe='cheap_test', digest=CHEAP.digest,
                                               read=dict(format='cif', dtype='float64', max_atomic_number=100),
                                               read_checked=False)
    for record in cheap.steps_run[:-1]:
        want = json.loads(json.dumps(CHEAP.spec()['steps'][CHEAP.step_names.index(record['step'])]['args']))
        assert json.loads(json.dumps(record['args'])) == {k: v for k, v in want.items() if v is not None}
    ok = [i for i, s in enumerate(_names(cheap)) if s == 'ok']
    assert ok and all(int(cheap.done[i]) == DONE_READ | DONE_FIND_MOLECULES | DONE_SANITIZE | DONE_POSE
                      | DONE_REBUILD | DONE_VALIDITY for i in ok)
    assert cheap.pos.dtype == F64


def test_run_chain_does_not_change_its_input():
    fresh = MolCrystalData.read(PATHS, dtype=F64)
    snap = _snapshot(fresh)
    fresh.run_chain(CHEAP)
    assert list(fresh._store.keys()) == list(snap.keys())
    assert all(_same(fresh._store[k], v) for k, v in snap.items())


def test_run_chain_refuses_a_processed_batch_or_another_dtype(cheap):
    with pytest.raises(ValueError, match='only read'):
        cheap.run_chain(CHEAP)
    found = MolCrystalData.read(PATHS, dtype=F64).find_molecules(**CHEAP.args('find_molecules'))
    with pytest.raises(ValueError, match='only read'):
        found.run_chain(CHEAP)
    with pytest.raises(ValueError, match='dtype'):
        MolCrystalData.read(PATHS).run_chain(CHEAP)                   # float32 batch, float64 recipe
    with pytest.raises(ValueError, match='unknown recipe'):
        MolCrystalData.read(PATHS, dtype=F64).run_chain('nope')


def test_a_row_a_later_step_refused_passes_through_the_structural_steps(monkeypatch, cheap):
    """A row sanitize refuses carries a chemistry code, which crystal_steps' own check does not accept yet: the
    chain must still run, pass the row through bit-identical and keep its code; every other row ends as without it."""
    victim = _names(cheap).index('ok')
    code = CHEM_TABLE.index('chemistry_failed')
    real = MolCrystalChem.sanitize

    def refusing(self, **kw):
        out = real(self, **kw)
        status = out.status.clone()
        status[victim] = code
        out['status'] = status
        out['message'] = [m if i != victim else 'chemistry_failed: injected' for i, m in enumerate(out.message)]
        return out

    monkeypatch.setattr(MolCrystalChem, 'sanitize', refusing)
    got = MolCrystalData.read(PATHS, dtype=F64).run_chain(CHEAP)
    assert _names(got)[victim] == 'chemistry_failed' and got.message[victim] == 'chemistry_failed: injected'
    assert int(got.done[victim]) == DONE_READ | DONE_FIND_MOLECULES | DONE_SANITIZE      # nothing after sanitize
    # the victim's atoms are sanitize's output, untouched by the pose (cheap's own row was posed)
    monkeypatch.setattr(MolCrystalChem, 'sanitize', real)
    sanitized = MolCrystalData.read(PATHS, dtype=F64).find_molecules(**CHEAP.args('find_molecules')) \
        .sanitize(**CHEAP.args('sanitize'))
    a, b = sanitized.select_rows([victim]), got.select_rows([victim])
    for k in ('z', 'pos', 'mol_ind', 'deposit_frac'):
        assert _same(a[k], b[k]), k
    others = [i for i in range(cheap.num_graphs) if i != victim]
    for k in ('status', 'done', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness', 'rebuild_max_dev',
              'validity_checked'):
        assert _same(cheap[k][others], got[k][others]), k
    assert _same(cheap.select_rows(others).pos, got.select_rows(others).pos)


@needs_ccdc
def test_csd_v1_runs_end_to_end_on_the_fixtures():
    got = MolCrystalData.read(PATHS, dtype=F64, full='csd_v1')
    chained = MolCrystalData.read(PATHS, dtype=F64).run_chain(CSD_V1)
    assert got.steps_run[-1]['args']['read_checked'] is True and chained.steps_run[-1]['args']['read_checked'] is False
    chained['steps_run'] = got.steps_run
    assert_same_batch(got, chained)
    names = _names(got)
    assert names.count('ok') >= 10
    ok = [i for i, s in enumerate(names) if s == 'ok']
    assert all(int(got.done[i]) == 127 for i in ok)                   # read .. features: bits 0-6
    assert all(got.unit_smiles[i] is not None for i in ok)
    assert got.steps_run[-1]['args']['recipe'] == 'csd_v1'
    product = got.canonicalize(**CANON)
    assert len(product) == sum(c in ('canonical', 'special_metric', 'off_sheet', 'pose_only')
                               for c in product.manifest['code'])
    assert all(s is not None for s in product.columns['smiles'])


# ======================================================================================
# canonicalize
# ======================================================================================

def _direct(batch, rows, sheet=1, mirror=True, off_sheet=None):
    """The importer and canon.canonicalize called directly on ``rows`` (float32 positions, as the wrapper gives)."""
    from mxtaltools.crystal_fd import canon, io_mxt
    picked = batch.select_rows(torch.tensor(rows))
    sub = cr._view(picked, pos=picked.pos.float())
    desc, info = io_mxt.from_mol_crystal_data(sub, sheet=sheet, pose_tol=1e-2, special_tol=0.1, op_tol=1e-6,
                                              allow_zp_gt1=False, nonstandard='reexpress', mirror_reference=mirror,
                                              metric_tol=1e-6, off_sheet=off_sheet)
    c = canon.canonicalize(desc, sheet=sheet, tol=1e-9, cell_tol=1e-10, angle_tol=1e-6, face_tol=1e-5, near_tol=1e-5,
                           rot_tol=1e-6, metric_tol=1e-4, moved_tol=1e-9)
    return info, c


def test_canonicalize_equals_the_importer_and_canon_called_directly(cheap):
    from mxtaltools.crystal_fd import canon
    snap = _snapshot(cheap)
    product = cheap.canonicalize(**CANON)
    assert all(_same(cheap._store[k], v) for k, v in snap.items())             # the batch is not changed
    names = _names(cheap)
    zp = cheap.z_prime.tolist()
    eligible = [i for i in range(cheap.num_graphs) if names[i] == 'ok' and zp[i] == 1]
    info, c = _direct(cheap, eligible)
    done = c.done
    want_rows = [eligible[r] for j, r in enumerate(info.row.tolist()) if bool(done[j])]
    assert product.columns['batch_row'].tolist() == want_rows and len(want_rows) >= 5
    keep = torch.nonzero(done).reshape(-1)
    for field in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(product.desc, field), getattr(c.desc[keep], field)), field
    assert [m.hash for m in product.mols] == [info.mols[j].hash for j in keep.tolist()]
    assert torch.equal(product.reference_handedness, info.reference_handedness[keep])
    assert product.identifier == [cheap.identifier[b] for b in want_rows]
    assert [product.status_names[s] for s in product.status.tolist()] == [c.names()[j] for j in keep.tolist()]
    assert product.status_names == tuple(canon.STATUS)
    assert torch.equal(product.columns['near_face_kind'], c.near_face[keep])
    # every input row has one manifest entry naming its outcome
    man = product.manifest
    assert man['source_row'].tolist() == list(range(cheap.num_graphs)) and man['identifier'] == list(cheap.identifier)
    for b in range(cheap.num_graphs):
        code = man['code'][b]
        if names[b] != 'ok':
            assert code == f'skipped:{names[b]}'
        elif zp[b] != 1:
            assert code == 'refused:z_prime_gt1'
        elif b in want_rows:
            assert code == product.status_names[int(product.status[want_rows.index(b)])]
            assert int(man['dataset_row'][b]) == want_rows.index(b)
        else:
            assert code.startswith(('refused:', 'failed:'))
        if b not in want_rows:
            assert int(man['dataset_row'][b]) == -1
    assert set(product.columns) == set(product.column_docs)
    assert product.args['importer']['nonstandard'] == 'reexpress' and product.args['canonicalize']['face_tol'] == 1e-5


def test_canonicalize_keep_off_sheet_leaves_the_mirrored_rows_off_the_sheet(cheap):
    mirrored = cheap.canonicalize(**CANON)
    kept = cheap.canonicalize(**dict(CANON, off_sheet='keep'))
    m = mirrored.columns['mirrored']
    assert torch.equal(mirrored.columns['batch_row'], kept.columns['batch_row'])
    assert not bool(kept.columns['mirrored'].any())
    assert bool(m.any())                                         # MIHTEE52 (P6_1) and another Sohncke row
    assert torch.equal(~kept.columns['on_sheet'], m)             # status off_sheet (any sg since 2026-10-21)
    off = torch.tensor([kept.status_names[s] == 'off_sheet' for s in kept.status.tolist()])
    assert not bool((off & ~m).any())
    unsheeted = cheap.canonicalize(**dict(CANON, sheet=None))
    assert len(unsheeted) == len(kept) and not bool(unsheeted.columns['mirrored'].any())


def test_canonicalize_save_round_trips_through_crystal_fd(cheap, tmp_path):
    from mxtaltools.crystal_fd.dataset import load_dataset, load_manifest
    product = cheap.canonicalize(**CANON, path=tmp_path / 'fd', provenance=dict(note='test'))
    ds = load_dataset(str(tmp_path / 'fd'), status=None, code='raise')
    for field in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(ds.desc, field), getattr(product.desc, field)), field
    assert ds.identifier == product.identifier and ds.mol_hash == [m.hash for m in product.mols]
    assert torch.equal(ds.reference_handedness, product.reference_handedness)
    assert ds.status_names == product.status_names and torch.equal(ds.status, product.status)
    assert ds.meta['provenance']['note'] == 'test'
    assert ds.meta['provenance']['parent'] == json.loads(json.dumps(product.parent))
    assert ds.meta['provenance']['canonicalize']['off_sheet'] == 'mirror_reference'
    assert load_manifest(str(tmp_path / 'fd'))['code'] == product.manifest['code']
    with pytest.raises(FileExistsError):
        product.save(tmp_path / 'fd')


def test_canonicalize_records_its_parent_batch_and_store(cheap, tmp_path):
    """Contract §5: a derived product records its parent (schema, sha256) and the policy arguments."""
    from mxtaltools.crystal_fd.dataset import load_dataset
    from mxtaltools.dataset_utils.data_class_methods.crystal_store import code_digest, read_meta
    product = cheap.canonicalize(**CANON)
    parent = product.parent
    assert parent['rows'] == cheap.num_graphs and parent['atoms'] == int(cheap.ptr[-1]) and parent['store'] is None
    assert parent['run_chain'] == dict(recipe='cheap_test', digest=CHEAP.digest, read_checked=False,
                                       read=dict(format='cif', dtype='float64', max_atomic_number=100))
    assert [s['step'] for s in parent['steps_run']] == list(CHEAP.step_names[1:]) + ['run_chain']
    assert parent['identifiers_sha256'] == hashlib.sha256('\n'.join(cheap.identifier).encode()).hexdigest()
    assert sum(parent['status_counts'].values()) == cheap.num_graphs
    assert parent['ingestion_code'] == code_digest()['digest']
    store = tmp_path / 'store'
    meta = cheap.save(store, steps_run=[dict(step='read', args={})] + list(cheap.steps_run))
    loaded = MolCrystalData.load(store)
    stored = loaded.canonicalize(**CANON, parent_store=store, path=tmp_path / 'fd')
    assert stored.parent['store']['data_sha256'] == meta['files']['data.pt']['sha256']
    assert stored.parent['store']['schema'] == meta['schema'] and stored.parent['store']['path'] == str(store)
    assert stored.parent['store']['steps_run'][0] == dict(step='read', args={})
    written = load_dataset(str(tmp_path / 'fd'), status=None, code='raise').meta['provenance']['parent']
    assert written == json.loads(json.dumps(stored.parent))
    with pytest.raises(ValueError, match='does not hold this batch'):
        loaded.select_rows(list(range(1, cheap.num_graphs))).canonicalize(**CANON, parent_store=store)
    with pytest.raises(FileNotFoundError):
        loaded.canonicalize(**CANON, parent_store=tmp_path / 'nowhere')


def test_canonicalize_smiles_without_chemistry_is_none(cheap):
    product = cheap.canonicalize(**CANON)
    assert len(product) and all(s is None for s in product.columns['smiles'])


@needs_ccdc
def test_canonicalize_smiles_describe_the_reference_molecule():
    """The column is the SMILES of mols[k], the molecule a consumer places with the FD pose: its stereo follows
    the reference's coordinates (a mirrored reference gives the enantiomer), not the deposited unit's."""
    from rdkit import Chem
    from mxtaltools.crystal_fd.description import MoleculeRef

    def canon(smiles):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    def enantiomer(smiles):
        m = Chem.MolFromSmiles(smiles)
        for atom in m.GetAtoms():
            if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        return Chem.MolToSmiles(m)

    batch = MolCrystalData.read(PATHS, dtype=F64, full='csd_v1')
    product = batch.canonicalize(**CANON)
    flip = torch.tensor([-1.0, 1.0, 1.0], dtype=F64)
    chiral = 0
    for k, b in enumerate(product.columns['batch_row'].tolist()):
        mol = product.mols[k]
        smiles = product.columns['smiles'][k]
        assert smiles is not None and smiles == cr._reference_smiles(batch, b, mol)
        assert product.columns['smiles_deposited'][k] == batch.unit_smiles[b][0]
        mirror = cr._reference_smiles(batch, b, MoleculeRef.from_coords(mol.pos * flip, mol.z))
        assert canon(mirror) == canon(enantiomer(smiles)), product.identifier[k]
        chiral += canon(mirror) != canon(smiles)
    assert chiral >= 3                                                   # DELDEG, HECVAP, IYOHIP, MIHTEE52, ...


def test_canonicalize_refuses_an_existing_path_before_any_work(cheap, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('worked before checking the path')

    monkeypatch.setattr(cr, '_canonical_rows', boom)
    (tmp_path / 'there').mkdir()
    with pytest.raises(FileExistsError):
        cheap.canonicalize(**CANON, path=tmp_path / 'there')


@pytest.mark.parametrize('change, error, match', [
    (dict(near_face='nudge'), ValueError, 'explicit nudge_step'),
    (dict(nudge_step=1e-4), ValueError, "nudge_step is for near_face='nudge'"),
    (dict(near_face='nudge', nudge_step=1e-5), ValueError, 'near-face tolerance'),
    (dict(near_face='nudge', nudge_step=float('nan')), ValueError, 'near-face tolerance'),
    (dict(near_face='nudge', nudge_step=True), ValueError, 'near-face tolerance'),
    (dict(near_face='nudge', nudge_step='1e-4'), ValueError, 'near-face tolerance'),
    (dict(setting='standard'), ValueError, 'setting'),
    (dict(sheet=True), ValueError, 'sheet'),
    (dict(sheet=0), ValueError, 'sheet'),
    (dict(off_sheet='mirror'), ValueError, 'off_sheet'),
    (dict(near_face='ignore'), ValueError, 'near_face'),
    (dict(zp_gt1='allow'), ValueError, 'D3'),
    (dict(provenance=['x']), TypeError, 'provenance'),
    (dict(provenance=dict(parent='mine')), ValueError, 'parent_store'),
    (dict(provenance=dict(canonicalize='mine')), ValueError, 'written by canonicalize'),
    (dict(parent_store=3), TypeError, 'parent_store'),
])
def test_canonicalize_bad_choices_raise(cheap, change, error, match):
    with pytest.raises(error, match=match):
        cheap.canonicalize(**dict(CANON, **change))


def test_canonicalize_choices_are_required(cheap):
    with pytest.raises(TypeError):
        cheap.canonicalize(setting='reexpress', sheet=1, off_sheet='mirror_reference', near_face='flag')


def test_canonicalize_needs_posed_rebuild_checked_rows():
    found = MolCrystalData.read(PATHS, dtype=F64).find_molecules(**CHEAP.args('find_molecules'))
    with pytest.raises(ValueError, match='parameterize_pose and check_rebuild'):
        found.canonicalize(**CANON)
    posed = found.sanitize(**CHEAP.args('sanitize')).parameterize_pose(**CHEAP.args('parameterize_pose'))
    with pytest.raises(ValueError, match='check_rebuild'):
        posed.canonicalize(**CANON)


def test_canonicalize_with_no_eligible_row_returns_an_empty_product(cheap, tmp_path):
    bad = [i for i, s in enumerate(_names(cheap)) if s != 'ok']
    product = cheap.select_rows(bad).canonicalize(**CANON, path=tmp_path / 'empty')
    assert isinstance(product, FDProduct) and len(product) == 0 and product.desc.num_rows == 0
    assert all(c.startswith('skipped:') for c in product.manifest['code'])
    assert (tmp_path / 'empty' / 'meta.json').is_file()


NUDGE = dict(CANON, near_face='nudge')
NUDGE_COLUMNS = ('nudged', 'nudge_cell', 'nudge_t', 'nudge_shift', 'nudge_message')


def test_canonicalize_nudge_leaves_rows_off_faces_as_flag_does(cheap):
    flag = cheap.canonicalize(**CANON)
    nudge = cheap.canonicalize(**NUDGE, nudge_step=1e-4)
    assert not bool(flag.columns['near_face'].any())                 # no fixture row is near a face
    for field in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(nudge.desc, field), getattr(flag.desc, field)), field
    assert set(nudge.columns) == set(flag.columns) | set(NUDGE_COLUMNS) and not set(flag.columns) & set(NUDGE_COLUMNS)
    for k in flag.columns:
        assert _same(nudge.columns[k], flag.columns[k]), k
    assert not bool(nudge.columns['nudged'].any()) and nudge.columns['nudge_message'] == [''] * len(nudge)
    for k in ('nudge_cell', 'nudge_t', 'nudge_shift'):
        assert float(nudge.columns[k].abs().max()) == 0.0, k
    assert set(nudge.column_docs) == set(nudge.columns)
    # the records of a nudged row are those of its unmoved canonical form, and the nudge-mode docs say so
    records = ('reduce_N', 'reduce_s', 'cell_n_auto', 'cell_n_tied', 'cell_violation', 'cell_snap', 'fold_element',
               'fold_lattice', 'fold_free_shift', 'fold_slack', 'fold_tie')
    for k in records:
        assert 'unmoved canonical form' in nudge.column_docs[k] and 'unmoved' not in flag.column_docs[k], k
        assert nudge.column_docs[k].startswith(flag.column_docs[k]), k
    for k in set(flag.columns) - set(records) - {'near_face', 'near_face_kind', 'moved'}:
        assert nudge.column_docs[k] == flag.column_docs[k], k
    assert nudge.manifest['code'] == flag.manifest['code']
    assert 'nudge_step' not in flag.args and 'nudge' not in flag.args['canonicalize']
    assert nudge.args['nudge_step'] == 1e-4 and nudge.args['canonicalize']['nudge'] == 1e-4
    assert nudge.args['near_face'] == 'nudge'


def _literal_atoms(desc, mol):
    """[|G|, n, 3] Cartesian atoms of the unit cell of a Z' = 1 row from SYM_OPS: molecule g at its centroid
    c_g = W t + w wrapped to [0, 1)^3, atoms T (c_g + W T^-1 P X). A cell change moves an atom in proportion to its
    distance from the origin, so the placement is part of the definition of nudge_shift."""
    from mxtaltools.constants.space_group_info import SYM_OPS
    T = desc.cell_matrix()[0]
    off = torch.linalg.solve(T, desc.pose[0, 0] @ mol.pos.to(F64).T).T                  # [n, 3] fractional
    ops_ = torch.as_tensor(np.stack(SYM_OPS[int(desc.sg[0])]), dtype=F64)
    cen = torch.einsum('gij,j->gi', ops_[:, :3, :3], desc.t[0, 0]) + ops_[:, :3, 3]
    f = (cen - torch.floor(cen))[:, None] + torch.einsum('gij,nj->gni', ops_[:, :3, :3], off)
    return torch.einsum('ij,gnj->gni', T, f)


def test_canonicalize_nudge_moves_near_face_rows(cheap, monkeypatch, tmp_path):
    """Two P-1 rows of the import put on faces: one centroid on a face of ASU(Gamma+), one cell 1e-6 from the a = b
    face of the cell chart. Flag marks them; nudge moves them, clears near_face and records the move."""
    from mxtaltools.crystal_fd import canon, io_mxt
    from mxtaltools.crystal_fd.dataset import load_dataset
    real = io_mxt.from_mol_crystal_data

    def on_faces(sub, **kw):
        desc, info = real(sub, **kw)
        p1 = torch.nonzero(desc.sg == 2)[:, 0].tolist()
        assert len(p1) >= 2
        c = canon.canonicalize(desc[torch.tensor(p1[:2])])
        t, cell = desc.t.clone(), desc.cell.clone()
        R, h = desc.R.clone(), desc.h.clone()
        for q, i in enumerate(p1[:2]):
            cell[i], t[i], R[i], h[i] = c.desc.cell[q], c.desc.t[q], c.desc.R[q], c.desc.h[q]
        t[p1[0], 0, 0] = 0.0                                                    # centroid on the x = 0 face
        chart = canon._cell_chart(2)
        y = chart.unmap(cell[p1[1]:p1[1] + 1])[0]
        y[0, 2] = 1 - 1e-6                                                      # a = b (1 - 1e-6)
        cell[p1[1]] = chart.map(y)[0][0]
        on_faces.rows = p1[:2]
        return desc.replace(cell=cell, t=t, R=R, h=h), info

    monkeypatch.setattr(io_mxt, 'from_mol_crystal_data', on_faces)
    flag = cheap.canonicalize(**CANON)
    nudge = cheap.canonicalize(**NUDGE, nudge_step=1e-4, path=tmp_path / 'fd', provenance=dict(note='nudge'))
    kinds = flag.columns['near_face_kind'].tolist()
    k_t, k_c = kinds.index(canon.NEAR_CENTROID), kinds.index(canon.NEAR_CELL)
    assert sum(1 for v in kinds if v) == 2 and int(flag.desc.sg[k_t]) == 2 and int(flag.desc.sg[k_c]) == 2
    assert torch.equal(nudge.columns['nudged'], flag.columns['near_face'])
    assert not bool(nudge.columns['near_face'].any()) and nudge.columns['nudge_message'] == [''] * len(nudge)
    assert bool(nudge.columns['moved'][[k_t, k_c]].all())
    assert torch.equal(nudge.status, flag.status) and nudge.manifest['code'] == flag.manifest['code']
    others = [k for k in range(len(flag)) if k not in (k_t, k_c)]
    for field in ('cell', 't', 'R', 'h'):
        assert torch.equal(getattr(nudge.desc, field)[others], getattr(flag.desc, field)[others]), field
    # the centroid row: t moved by the step off the x face, cell and pose kept; every atom moves by T dt
    dt = nudge.desc.t[k_t, 0] - flag.desc.t[k_t, 0]
    assert abs(float(nudge.columns['nudge_t'][k_t]) - 1e-4) <= 1e-12 and float(nudge.columns['nudge_cell'][k_t]) == 0
    assert torch.equal(nudge.desc.cell[k_t], flag.desc.cell[k_t]) and torch.equal(nudge.desc.R[k_t], flag.desc.R[k_t])
    want = float((flag.desc[k_t:k_t + 1].cell_matrix()[0] @ dt).norm())
    assert abs(float(nudge.columns['nudge_shift'][k_t]) - want) <= 1e-9
    # the cell row: the chart point moved by the step less 1e-6, t and R kept; atoms compared literally
    assert abs(float(nudge.columns['nudge_cell'][k_c]) - (1e-4 - 1e-6)) <= 1e-10
    assert float(nudge.columns['nudge_t'][k_c]) == 0 and torch.equal(nudge.desc.t[k_c], flag.desc.t[k_c])
    lit = (_literal_atoms(nudge.desc[k_c:k_c + 1], nudge.mols[k_c])
           - _literal_atoms(flag.desc[k_c:k_c + 1], flag.mols[k_c])).norm(dim=-1).max()
    assert abs(float(nudge.columns['nudge_shift'][k_c]) - float(lit)) <= 1e-9
    assert 1e-5 < float(nudge.columns['nudge_shift'][k_c]) < 1e-2
    # written and read back: the columns, and the choice in the provenance
    ds = load_dataset(str(tmp_path / 'fd'), status=None, code='raise')
    for k in NUDGE_COLUMNS:
        assert _same(ds.columns[k], nudge.columns[k]), k
    rec = ds.meta['provenance']['canonicalize']
    assert rec['near_face'] == 'nudge' and rec['nudge_step'] == 1e-4 and rec['canonicalize']['nudge'] == 1e-4
    assert ds.meta['provenance']['note'] == 'nudge'


def test_canonicalize_nudge_shift_is_the_move_when_the_pose_is_redecided(cheap, monkeypatch):
    """A P212121 row of the import put on a 2-fold axis of Gamma+ (t = (0.1, 0, 0): fold tie 2) with its angles 1e-7
    rad off 90 deg and a pose for which rounding picks another orientation than the exact metric does (a fold of the
    row as given). Since 2026-10-21 the cell step snaps the cell to the group's exact metric, so the flag product
    already holds the exact-metric pose (not the as-given fold's) and records the snap; the nudge keeps that pose and
    moves the centroid off the axis; nudge_shift is the move alone."""
    from mxtaltools.crystal_fd import domains, io_mxt, so3
    real = io_mxt.from_mol_crystal_data

    def on_axis(sub, **kw):
        desc, info = real(sub, **kw)
        gen = torch.Generator().manual_seed(0)                  # the same row in every product
        i = int(torch.nonzero(desc.sg == 19)[0, 0])
        cell, t, R = desc.cell.clone(), desc.t.clone(), desc.R.clone()
        t[i, 0] = torch.tensor([0.1, 0.0, 0.0], dtype=t.dtype)
        exact = cell[i:i + 1].clone()
        exact[0, 3:] = torch.pi / 2
        for _ in range(50):                                     # bounded: about half of the poses flip
            R[i, 0] = so3.random_rotations(1, gen)[0].to(R.dtype)
            for sign in (1.0, -1.0):
                cell[i] = exact[0]
                cell[i, 3:] += sign * torch.tensor([1e-7, -1e-7, 1e-7], dtype=cell.dtype)
                row = desc[i:i + 1].replace(cell=cell[i:i + 1], t=t[i:i + 1], R=R[i:i + 1])
                if float((domains.fold(row).desc.R - domains.fold(row.replace(cell=exact)).desc.R)
                         .abs().max()) > 1e-3:
                    on_axis.row, on_axis.exact = row, exact
                    return desc.replace(cell=cell, t=t, R=R), info
        raise AssertionError('no pose whose orientation rounding decides')

    monkeypatch.setattr(io_mxt, 'from_mol_crystal_data', on_axis)
    flag = cheap.canonicalize(**CANON)
    nudge = cheap.canonicalize(**NUDGE, nudge_step=1e-4)
    near = torch.nonzero(flag.columns['near_face'])[:, 0].tolist()
    assert len(near) == 1
    k = near[0]
    assert int(flag.desc.sg[k]) == 19 and int(flag.columns['fold_tie'][k]) == 2
    assert torch.equal(flag.columns['reduce_N'][k], torch.eye(3, dtype=torch.long))      # already in axis order
    as_given = domains.fold(on_axis.row).desc.R[0, 0]
    exact = domains.fold(on_axis.row.replace(cell=on_axis.exact)).desc.R[0, 0]
    assert float((flag.desc.R[k, 0] - as_given).abs().max()) > 1e-3                # the snap re-decided the pose
    assert float((flag.desc.R[k, 0] - exact).abs().max()) <= 1e-9
    assert 0.5e-7 <= float(flag.columns['cell_snap'][k]) <= 2e-7                  # the angles' 1e-7
    assert bool(nudge.columns['nudged'][k]) and nudge.columns['nudge_message'][k] == ''
    assert float((nudge.desc.R[k] - flag.desc.R[k]).abs().max()) <= 1e-9         # the nudge keeps the snapped pose
    dt = nudge.desc.t[k, 0] - flag.desc.t[k, 0]
    assert abs(float(nudge.columns['nudge_t'][k]) - 1e-4) <= 1e-12 and abs(float(dt.abs().max()) - 1e-4) <= 1e-12
    want = float((flag.desc[k:k + 1].cell_matrix()[0] @ dt).norm())
    assert abs(float(nudge.columns['nudge_shift'][k]) - want) <= 1e-9 and want < 1e-2


def test_canonicalize_gives_sg_16_and_up_a_cell_step(cheap, monkeypatch):
    """Since 2026-10-21 rows of sg >= 16 have a cell step: the fixture's sg >= 16 rows are canonical with the snap to
    the exact metric recorded (cell_snap finite exactly there, NaN for sg 1-15) and a reduce_N that is a det +1 signed
    axis permutation. Its two P212121 rows, re-described by the importer in each other allowed axis order, come back
    as the same rows (within 1e-9) with reduce_N undoing the re-description up to a diagonal sign change (an element
    of Gamma+, which the fold absorbs); every other row is bit for bit the plain product's."""
    from mxtaltools.crystal_fd import axis_order, io_mxt, ops
    flag = cheap.canonicalize(**CANON)
    sg = flag.desc.sg
    high = sg >= 16
    assert int(high.sum()) >= 5
    assert all(flag.status_names[s] == 'canonical' for s in flag.status[high].tolist())
    snap = flag.columns['cell_snap']
    assert bool(torch.isfinite(snap[high]).all()) and bool(torch.isnan(snap[~high]).all()) and float(snap.nan_to_num()
                                                                                                      .max()) < 1e-6
    N = flag.columns['reduce_N'][high].double()
    assert bool((N.abs().sum(1) == 1).all() & (N.abs().sum(2) == 1).all()) and bool((torch.linalg.det(N) == 1).all())
    o = axis_order.axis_orders(19)
    k19 = torch.nonzero(sg == 19)[:, 0]
    assert len(k19) == 2 and o.k == 6
    real = io_mxt.from_mol_crystal_data
    eye = torch.eye(3, dtype=F64)
    for j in range(1, o.k):
        def reordered(sub, **kw):
            desc, info = real(sub, **kw)
            rows = torch.nonzero(desc.sg == 19)[:, 0]
            moved = ops.change_basis(desc[rows], o.N[j].expand(len(rows), 3, 3), o.s[j].expand(len(rows), 3))
            cell, t, R = desc.cell.clone(), desc.t.clone(), desc.R.clone()
            cell[rows], t[rows], R[rows] = moved.cell.to(cell.dtype), moved.t.to(t.dtype), moved.R.to(R.dtype)
            return desc.replace(cell=cell, t=t, R=R), info

        monkeypatch.setattr(io_mxt, 'from_mol_crystal_data', reordered)
        p = cheap.canonicalize(**CANON)
        monkeypatch.setattr(io_mxt, 'from_mol_crystal_data', real)
        assert torch.equal(p.columns['batch_row'], flag.columns['batch_row']) and torch.equal(p.status, flag.status)
        a, b = p.desc[k19], flag.desc[k19]
        assert float(((a.cell[:, :3] - b.cell[:, :3]).abs() / b.cell[:, :3]).max()) <= 1e-9, j
        assert float((a.cell[:, 3:] - b.cell[:, 3:]).abs().max()) <= 1e-9 and float((a.t - b.t).abs().max()) <= 1e-9
        assert float((a.R - b.R).abs().max()) <= 1e-9, j
        undo = o.N[j].double() @ p.columns['reduce_N'][k19].double()                 # input basis -> canonical
        assert bool((undo.abs() == eye).all()), (j, undo)
        rest = torch.nonzero(sg != 19)[:, 0]
        for field in ('sg', 'cell', 't', 'R', 'h'):
            assert torch.equal(getattr(p.desc, field)[rest], getattr(flag.desc, field)[rest]), field


# ======================================================================================
# canonicalize(off_sheet='enantiomorph') (owner decision 2026-10-21)
# ======================================================================================

ENANT = dict(CANON, off_sheet='enantiomorph')
ENANT_COLUMNS = ('enantiomorph', 'sg_source')


def _eligible(batch):
    names = _names(batch)
    return [i for i in range(batch.num_graphs) if names[i] == 'ok' and int(batch.z_prime[i]) == 1]


def test_canonicalize_enantiomorph_stores_the_mirror_image_crystal(cheap):
    """The rows D2 mirrors are stored as their enantiomorph (reference X = X' diag(-1, 1, 1), h = +1, the partner
    space group in a pair), every other row as D2 stores it; the product is the importer and canon.canonicalize
    called directly with off_sheet='enantiomorph'; the record and the column docs say so."""
    from mxtaltools.crystal_fd.enantiomorph import enantiomorph_map
    mirror = cheap.canonicalize(**CANON)
    enant = cheap.canonicalize(**ENANT)
    e = enant.columns['enantiomorph']
    src = enant.columns['sg_source']
    assert torch.equal(enant.columns['batch_row'], mirror.columns['batch_row'])
    assert e.dtype == torch.bool and torch.equal(e, mirror.columns['mirrored']) and bool(e.any())
    assert src.dtype == torch.long and torch.equal(src, mirror.desc.sg)
    assert enant.desc.sg.tolist() == [enantiomorph_map(s).sg_out if x else s for s, x in zip(src.tolist(), e.tolist())]
    assert not bool(enant.columns['mirrored'].any()) and bool((enant.reference_handedness == 1).all())
    assert bool(enant.columns['on_sheet'].all()) and bool((enant.desc.h[:, 0] == 1).all())
    assert set(enant.columns) == set(mirror.columns) | set(ENANT_COLUMNS)
    assert set(enant.column_docs) == set(enant.columns)
    assert all(enant.status_names[s] == 'canonical' for s in enant.status[e].tolist())
    assert torch.equal(enant.status[~e], mirror.status[~e])
    flip = torch.tensor([-1.0, 1.0, 1.0], dtype=F64)
    for k in range(len(enant)):
        if bool(e[k]):
            assert torch.equal(enant.mols[k].pos, mirror.mols[k].pos * flip) and enant.mols[k].hash != mirror.mols[k].hash
        else:
            assert enant.mols[k].hash == mirror.mols[k].hash
    rest = torch.nonzero(~e)[:, 0]
    for field in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(enant.desc, field)[rest], getattr(mirror.desc, field)[rest]), field
    for k, v in mirror.columns.items():
        w = enant.columns[k]
        if torch.is_tensor(v):
            assert _same(w[rest], v[rest]), k
        else:
            assert [w[i] for i in rest.tolist()] == [v[i] for i in rest.tolist()], k
    assert enant.manifest['code'] == mirror.manifest['code']
    # the importer and canon.canonicalize called directly
    info, c = _direct(cheap, _eligible(cheap), mirror=False, off_sheet='enantiomorph')
    keep = torch.nonzero(c.done).reshape(-1)
    for field in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(enant.desc, field), getattr(c.desc[keep], field)), field
    assert torch.equal(e, info.enantiomorph[keep]) and torch.equal(src, info.sg_source[keep])
    assert [m.hash for m in enant.mols] == [info.mols[j].hash for j in keep.tolist()]
    # the record and the docs
    assert enant.args['off_sheet'] == 'enantiomorph' and enant.args['importer']['off_sheet'] == 'enantiomorph'
    assert enant.args['importer']['mirror_reference'] is False
    assert 'never set' in enant.column_docs['mirrored'] and 'mirror-image' in enant.column_docs['enantiomorph']
    assert 'before the enantiomorph map' in enant.column_docs['sg_source']
    for k in set(mirror.column_docs) - set(cr._ENANTIOMORPH_COLUMN_DOCS):
        assert enant.column_docs[k] == mirror.column_docs[k], k


def test_canonicalize_other_modes_carry_no_enantiomorph_record(cheap):
    """'mirror_reference' and 'keep' (either sheet or none) products have neither column nor importer record of the
    mode, and their docs are the base docs; with sheet=None the enantiomorph mode maps nothing."""
    for kw in (CANON, dict(CANON, off_sheet='keep'), dict(CANON, sheet=None), dict(CANON, sheet=-1)):
        p = cheap.canonicalize(**kw)
        assert not set(ENANT_COLUMNS) & set(p.columns) and 'off_sheet' not in p.args['importer'], kw
        assert all(p.column_docs[k] == cr._COLUMN_DOCS[k] for k in p.column_docs), kw
    plain = cheap.canonicalize(**dict(CANON, sheet=None))
    none = cheap.canonicalize(**dict(ENANT, sheet=None))
    assert not bool(none.columns['enantiomorph'].any()) and torch.equal(none.columns['sg_source'], none.desc.sg)
    for field in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(none.desc, field), getattr(plain.desc, field)), field


def test_canonicalize_enantiomorph_save_round_trips(cheap, tmp_path):
    from mxtaltools.crystal_fd.dataset import load_dataset
    product = cheap.canonicalize(**ENANT, path=tmp_path / 'fd')
    ds = load_dataset(str(tmp_path / 'fd'), status=None, code='raise')
    for field in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(ds.desc, field), getattr(product.desc, field)), field
    for k in ENANT_COLUMNS + ('mirrored',):
        assert _same(ds.columns[k], product.columns[k]), k
    assert ds.mol_hash == [m.hash for m in product.mols] and bool((ds.reference_handedness == 1).all())
    rec = ds.meta['provenance']['canonicalize']
    assert rec['off_sheet'] == 'enantiomorph' and rec['importer']['off_sheet'] == 'enantiomorph'


def test_canonicalize_enantiomorph_with_nudge(cheap, monkeypatch):
    """near_face='nudge' with off_sheet='enantiomorph': an enantiomorph row put with its canonical centroid on the
    nearest facet of ASU(Gamma+) is flagged by 'flag' and moved by 'nudge'; both products carry both column sets."""
    from mxtaltools.crystal_fd import canon, io_mxt
    from mxtaltools.crystal_fd.tables import group_table
    real = io_mxt.from_mol_crystal_data

    def on_face(sub, **kw):
        desc, info = real(sub, **kw)
        assert kw['off_sheet'] == 'enantiomorph'
        j = int(torch.nonzero(info.enantiomorph)[0, 0])
        row = canon.canonicalize(desc[j:j + 1]).desc
        tab = group_table(int(row.sg[0]))
        x = row.t[0, 0]
        n, c = tab.asu.normals, tab.asu.offsets
        s = (n @ x + c) / n.norm(dim=1)
        i = int(torch.nonzero(tab.asu.facet)[torch.argmin(s[tab.asu.facet])])
        cell, t, R = desc.cell.clone(), desc.t.clone(), desc.R.clone()
        cell[j] = row.cell[0]
        t[j, 0] = x - s[i] * n[i] / n[i].norm()                       # onto the facet
        R[j, 0] = row.R[0, 0]
        on_face.row = int(info.row[j])
        return desc.replace(cell=cell, t=t, R=R), info

    monkeypatch.setattr(io_mxt, 'from_mol_crystal_data', on_face)
    flag = cheap.canonicalize(**ENANT)
    nudge = cheap.canonicalize(**dict(ENANT, near_face='nudge'), nudge_step=1e-4)
    k = flag.columns['batch_row'].tolist().index(_eligible(cheap)[on_face.row])
    assert bool(flag.columns['enantiomorph'][k]) and int(flag.columns['near_face_kind'][k]) & canon.NEAR_CENTROID
    assert torch.equal(nudge.columns['nudged'], flag.columns['near_face']) and bool(nudge.columns['nudged'][k])
    assert not bool(nudge.columns['near_face'][k]) and nudge.columns['nudge_message'][k] == ''
    assert 0 < float(nudge.columns['nudge_t'][k]) <= 1e-4 + 1e-12 and float(nudge.columns['nudge_shift'][k]) > 0
    for col in ENANT_COLUMNS:
        assert torch.equal(nudge.columns[col], flag.columns[col]), col
    assert set(nudge.columns) == set(flag.columns) | {'nudged', 'nudge_cell', 'nudge_t', 'nudge_shift', 'nudge_message'}
    assert set(nudge.column_docs) == set(nudge.columns) and 'never set' in nudge.column_docs['mirrored']
    others = [q for q in range(len(flag)) if q != k]
    for field in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(nudge.desc, field)[others], getattr(flag.desc, field)[others]), field
    assert nudge.args['importer']['off_sheet'] == 'enantiomorph' and nudge.args['canonicalize']['nudge'] == 1e-4


def _canon(smiles):
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _enantiomer(smiles):
    """The SMILES with every tetrahedral tag inverted (other stereo tags are not inverted)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    for atom in m.GetAtoms():
        if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(m)


def test_enantiomorph_column_docs_state_their_scope():
    """The 'enantiomorph' doc names the origin shift and the groups where the map derives one; the smiles and
    smiles_deposited docs state the h_input stereo rule for Sohncke groups only (elsewhere the posed unit may be an
    improper image of the deposited unit: test_canonicalize_enantiomorph_smiles_rule_on_real_rows)."""
    from mxtaltools.crystal_fd.enantiomorph import enantiomorph_map
    from mxtaltools.crystal_fd.io_mxt import is_sohncke
    shifted = [s for s in range(1, 231) if is_sohncke(s) and bool(np.any(enantiomorph_map(s).p != 0))]
    docs = cr._ENANTIOMORPH_COLUMN_DOCS
    assert f"origin shift p in sg {', '.join(map(str, shifted))}" in docs['enantiomorph'], shifted
    for k in ('smiles', 'smiles_deposited'):
        assert 'in a Sohncke group' in docs[k] and 'in other groups' in docs[k], k


@needs_ccdc
def test_canonicalize_enantiomorph_smiles_carry_the_reference_stereo():
    """With CCDC chemistry: smiles is the SMILES of the stored reference X; on enantiomorph rows (a chiral molecule
    among them, and a partner-group row) it is the enantiomer of the D2 product's smiles and of smiles_deposited;
    on every Sohncke row it equals smiles_deposited where h_input is +1 and its enantiomer where -1 (in other groups
    the posed unit may be an improper image of the deposited unit, so only one of the two is required)."""
    from mxtaltools.crystal_fd.io_mxt import is_sohncke
    canon, enantiomer = _canon, _enantiomer
    batch = MolCrystalData.read(PATHS, dtype=F64, full='csd_v1')
    mirror = batch.canonicalize(**CANON)
    enant = batch.canonicalize(**ENANT)
    e = enant.columns['enantiomorph']
    assert torch.equal(e, mirror.columns['mirrored']) and bool(e.any())
    chiral_mapped, partner, sohncke = 0, 0, 0
    for k, b in enumerate(enant.columns['batch_row'].tolist()):
        s, s_m, s_dep = enant.columns['smiles'][k], mirror.columns['smiles'][k], enant.columns['smiles_deposited'][k]
        assert s is not None and s == cr._reference_smiles(batch, b, enant.mols[k])
        if is_sohncke(int(enant.columns['sg_source'][k])):
            want_dep = canon(s_dep) if int(enant.columns['h_input'][k]) == 1 else canon(enantiomer(s_dep))
            assert canon(s) == want_dep, enant.identifier[k]
            sohncke += 1
        else:
            assert canon(s) in (canon(s_dep), canon(enantiomer(s_dep))), enant.identifier[k]
        if bool(e[k]):
            assert canon(s) == canon(enantiomer(s_m)), enant.identifier[k]
            chiral_mapped += canon(s) != canon(s_m)
            partner += int(enant.desc.sg[k]) != int(enant.columns['sg_source'][k])
        else:
            assert s == s_m, enant.identifier[k]
    assert chiral_mapped >= 1 and partner >= 1, (chiral_mapped, partner)            # HECVAP (P4_3 -> P4_1), IYOHIP
    assert sohncke >= chiral_mapped
    assert 'enantiomer of smiles_deposited' in enant.column_docs['smiles']


#: a csd_v1 chunk store (local data, opened read-only by MolCrystalData.load)
CHUNK = (Path(os.environ.get('MXT_CSD_DATA', 'D:/crystal_datasets')) / 'ingest_csd_v1' / 'csd_v1-1b8fd7793a9f'
         / 'chunks' / 'chunk_001499')


def test_canonicalize_enantiomorph_smiles_rule_on_real_rows():
    """``CHUNK`` (skipped when absent), off_sheet='enantiomorph': on every row smiles is the SMILES of the posed unit 0
    (the batch's coordinates) where h_input is +1 and of its mirror image (coordinates x -> -x) where -1. In Sohncke
    groups (the enantiomorph rows among them) the posed unit has the stereo of smiles_deposited, so there the rule
    holds against smiles_deposited; in other groups it does not on every row (WAMYUI, P-1: the posed unit is the
    inverted deposited unit), which is why the column docs state that rule for Sohncke groups only. Mirror images are
    built from coordinates, not by inverting SMILES tags (which misses non-tetrahedral stereo and, on WANYAM, gives
    another stereoisomer)."""
    from mxtaltools.crystal_fd.io_mxt import is_sohncke
    if not CHUNK.exists():
        pytest.skip(f'{CHUNK} not on this machine')
    pytest.importorskip('rdkit')
    batch = MolCrystalData.load(str(CHUNK))
    enant = batch.canonicalize(**ENANT)
    e = enant.columns['enantiomorph']
    flip = torch.tensor([-1.0, 1.0, 1.0], dtype=F64)
    n_rows, n_sohncke, n_mapped, other_breaks = 0, 0, 0, []
    for k, b in enumerate(enant.columns['batch_row'].tolist()):
        s, s_dep = enant.columns['smiles'][k], enant.columns['smiles_deposited'][k]
        if s is None or s_dep is None:
            continue
        lo, hi = int(batch.ptr[b]), int(batch.ptr[b + 1])
        pos, z = batch.pos[lo:hi].detach().double(), batch.z[lo:hi]
        posed = cr._reference_smiles(batch, b, SimpleNamespace(pos=pos, z=z))
        mirror = cr._reference_smiles(batch, b, SimpleNamespace(pos=pos * flip, z=z))
        want = posed if int(enant.columns['h_input'][k]) == 1 else mirror
        assert _canon(s) == _canon(want), enant.identifier[k]
        n_rows += 1
        if is_sohncke(int(enant.columns['sg_source'][k])):
            assert _canon(posed) == _canon(s_dep), enant.identifier[k]
            n_sohncke += 1
            n_mapped += bool(e[k])
        elif _canon(posed) != _canon(s_dep):
            other_breaks.append(enant.identifier[k])
    assert n_rows >= 150 and n_sohncke >= 30 and n_mapped >= 10, (n_rows, n_sohncke, n_mapped)
    assert 'WAMYUI' in other_breaks, other_breaks


# ======================================================================================
# features_fallback
# ======================================================================================

def _with_fallback(batch, codes):
    """A copy of ``batch`` holding ``features_fallback`` [B] int8 ``codes``, written as ``features`` writes it."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_steps import _copy_batch, _set
    out = _copy_batch(batch)
    _set(out, 'features_fallback', torch.as_tensor(codes, dtype=torch.int8), 'row')
    return out


def test_canonicalize_carries_features_fallback_into_the_product(cheap, tmp_path):
    """A row featurized by the partial-sanitization fallback keeps its flag (int8, an index into FEATURES_FALLBACK)
    in the product, through the batch store and the FD dataset; the column doc names the codes. The rest of the
    product is that of the same batch without the attribute, whose column is 0 on every row."""
    from mxtaltools.crystal_fd.dataset import load_dataset
    plain = cheap.canonicalize(**CANON)
    assert 'features_fallback' not in cheap._store
    kept = plain.columns['batch_row'].tolist()
    assert len(kept) >= 5
    codes = [0] * cheap.num_graphs
    codes[kept[0]], codes[kept[-1]] = FEATURES_FALLBACK.index('valence'), FEATURES_FALLBACK.index('kekule')
    codes[next(b for b in range(cheap.num_graphs) if b not in kept)] = 2          # a row the product does not keep
    flagged = _with_fallback(cheap, codes)
    product = flagged.canonicalize(**CANON)
    col = product.columns['features_fallback']
    assert col.dtype == torch.int8 and col.tolist() == [1] + [0] * (len(kept) - 2) + [2]
    zero = plain.columns['features_fallback']
    assert zero.dtype == torch.int8 and zero.tolist() == [0] * len(kept)
    doc = product.column_docs['features_fallback']
    assert repr(FEATURES_FALLBACK) in doc and 'without the attribute' in doc
    for field in dataclasses.fields(FDProduct):
        a, b = getattr(product, field.name), getattr(plain, field.name)
        if field.name == 'columns':
            a, b = {k: v for k, v in a.items() if k != 'features_fallback'}, \
                {k: v for k, v in b.items() if k != 'features_fallback'}
            assert list(a) == list(b)
        if field.name == 'desc':
            assert all(torch.equal(getattr(a, f), getattr(b, f)) for f in ('sg', 'cell', 't', 'R', 'h'))
        elif field.name == 'mols':
            assert [m.hash for m in a] == [m.hash for m in b]
        else:
            assert _same(a, b), field.name
    for mode in (dict(off_sheet='enantiomorph'), dict(near_face='nudge', nudge_step=1e-4)):
        assert torch.equal(flagged.canonicalize(**dict(CANON, **mode)).columns['features_fallback'], col), mode
    # the batch store keeps the flag, and the FD dataset writes it with its doc
    store = tmp_path / 'store'
    flagged.save(store, steps_run=[dict(step='read', args={})] + list(flagged.steps_run))
    again = MolCrystalData.load(store).canonicalize(**CANON, parent_store=store, path=tmp_path / 'fd')
    assert torch.equal(again.columns['features_fallback'], col)
    ds = load_dataset(str(tmp_path / 'fd'), status=None, code='raise')
    assert torch.equal(ds.columns['features_fallback'], col)
    assert ds.meta['columns']['features_fallback'] == dict(dtype='int8', shape=[len(kept)], doc=doc)


def test_canonicalize_refuses_a_features_fallback_that_is_not_codes(cheap):
    B = cheap.num_graphs
    for value, match in ((torch.full((B,), len(FEATURES_FALLBACK), dtype=torch.int8), 'codes outside'),
                         (torch.full((B,), -1, dtype=torch.int8), 'codes outside'),
                         (torch.zeros(B, dtype=torch.float32), 'integer tensor'),
                         (torch.zeros(B, dtype=torch.bool), 'integer tensor')):
        batch = _with_fallback(cheap, [0] * B)
        batch['features_fallback'] = value
        with pytest.raises(ValueError, match=match):
            batch.canonicalize(**CANON)


@needs_ccdc
def test_canonicalize_carries_the_flag_of_rows_the_fallback_recovers():
    """``CHUNK`` (skipped when absent) was stored before the fallback: no features_fallback, so its product's column
    is 0, and its features_failed rows are not in it. Those rows read again through CSD_V1 are recovered (most of them
    organometallic) and enter the product with their tier, the batch's flag of each kept row."""
    if not CHUNK.exists():
        pytest.skip(f'{CHUNK} not on this machine')
    stored = MolCrystalData.load(str(CHUNK))
    assert 'features_fallback' not in stored._store
    old = stored.canonicalize(**CANON)
    assert old.columns['features_fallback'].dtype == torch.int8 and not bool(old.columns['features_fallback'].any())
    table = status_table()
    rows = [b for b, s in enumerate(stored.status.tolist()) if table[s] == 'features_failed']
    paths = [str(stored.source_path[b]) for b in rows]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("CHUNK's source CIFs are not on this machine")
    batch = MolCrystalData.read(paths, dtype=F64, full='csd_v1')
    product = batch.canonicalize(**CANON)
    col = product.columns['features_fallback']
    assert col.dtype == torch.int8
    assert torch.equal(col, batch.features_fallback[product.columns['batch_row']].cpu())
    tiers = {FEATURES_FALLBACK.index('valence'), FEATURES_FALLBACK.index('kekule')}
    assert len(col) >= 20 and set(col.tolist()) == tiers
    assert not set(product.identifier) & set(old.identifier)

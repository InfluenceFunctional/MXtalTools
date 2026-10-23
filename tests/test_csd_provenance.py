"""The CSD provenance side table (`mxtaltools/dataset_utils/construction/csd_provenance.py`).

The database tests need a CCDC licence and an installed CSD; they skip when
``EntryReader('CSD')`` cannot open ACRDIN04. A skip is not evidence the table is right.
The temperature parser and the input checks run without the CCDC.

Expected ACRDIN04 values were read from the database by the ingestion survey
(``artifacts/fd_design_oct15/ingestion_survey/designer/entry_probe.py`` in the GFN
repository): 'at 185 K', R 3.81, CCDC 259366, deposited 2005-03-29, 'polymorph II'.
NUKCOL is a powder study.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from mxtaltools.dataset_utils.construction import csd_provenance
from mxtaltools.dataset_utils.construction.csd_provenance import (SCHEMA, build_provenance, load_provenance,
                                                                  parse_temperature_kelvin)

FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
MISSING = 'QQQQQQ99'

#: Runs build_provenance(argv[1] split on ',', argv[2], chunk_size=5) and dies with exit code 9, like an abrupt
#: kill, at the rename onto the file named argv[3]: its temporary file is fully written, the rename never happens.
_KILLED_CHILD = '''
import os, sys
from mxtaltools.dataset_utils.construction import csd_provenance
refcodes, out, target = sys.argv[1].split(','), sys.argv[2], sys.argv[3]
rename = os.replace
def killed_at_target(src, dst):
    if os.path.basename(dst) == target:
        os._exit(9)
    return rename(src, dst)
csd_provenance.os.replace = killed_at_target
csd_provenance.build_provenance(refcodes, out, chunk_size=5)
'''


def _assert_same_table(reference: dict, other: dict) -> None:
    assert reference.keys() == other.keys()
    for key, value in reference.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value.nan_to_num(-7.0), other[key].nan_to_num(-7.0)), key
        else:
            assert value == other[key], key


@pytest.fixture(scope='module')
def csd():
    """Skip unless the CCDC API can open the CSD."""
    try:
        from ccdc import io
        reader = io.EntryReader('CSD')
        reader.entry('ACRDIN04')
        reader.close()
    except Exception as e:                          # ImportError, licence or database faults
        pytest.skip(f'the CCDC cannot open the CSD: {type(e).__name__}: {e}')


@pytest.fixture(scope='module')
def refcodes():
    names = sorted(p.stem for p in FIXTURES.glob('*.cif'))
    assert len(names) == 16, names
    return names + ['ACRDIN04', MISSING]


@pytest.fixture(scope='module')
def table(csd, refcodes, tmp_path_factory):
    out = tmp_path_factory.mktemp('prov') / 'table'
    meta = build_provenance(refcodes, out, chunk_size=5)
    return out, meta, load_provenance(out)


@pytest.mark.parametrize('raw, kelvin', [
    ('240 K', 240.0), ('at 185 K', 185.0), ('100.2 K', 100.2), ('at 193K', 193.0), ('at 98. K', 98.0),
    ('100(2) K', 100.0), ('at -100 deg.C', 173.15), ('at -135deg.C', 138.15), ('at 23 deg.C.', 296.15),
    ('at 20.0deg.C', 293.15), ('240-260 K', math.nan), ('at room temperature', math.nan), ('', math.nan),
    (None, math.nan), ('at -300 deg.C', math.nan)])
def test_temperature_parser(raw, kelvin):
    got = parse_temperature_kelvin(raw)
    assert (math.isnan(got) and math.isnan(kelvin)) or got == pytest.approx(kelvin, abs=1e-9)


def test_inputs_are_checked_before_the_database(tmp_path):
    with pytest.raises(ValueError, match='repeat'):
        build_provenance(['ACRDIN04', 'ACRDIN04'], tmp_path / 'a')
    with pytest.raises(TypeError, match='str'):
        build_provenance(['ACRDIN04', 3], tmp_path / 'a')
    with pytest.raises(TypeError, match='one str'):
        build_provenance('ACRDIN04', tmp_path / 'a')
    with pytest.raises(ValueError, match='chunk_size'):
        build_provenance(['ACRDIN04'], tmp_path / 'a', chunk_size=0)
    assert list(tmp_path.iterdir()) == []


def test_folders_of_other_files_are_refused_before_the_database(tmp_path):
    """Only an empty folder, or one holding just a chunks/ of temporary files, counts as a fresh start."""
    foreign = tmp_path / 'foreign_in_chunks'
    (foreign / 'chunks').mkdir(parents=True)
    (foreign / 'chunks' / 'chunk_00000.pt').write_bytes(b'x')
    beside = tmp_path / 'temporary_plus_other'
    (beside / 'chunks').mkdir(parents=True)
    (beside / 'chunks' / 'plan.json.tmp-123').write_text('{')
    (beside / 'notes.txt').write_text('x')
    for out in (foreign, beside):
        with pytest.raises(ValueError, match='not a provenance run'):
            build_provenance(['ACRDIN04'], out)
    assert sorted(p.name for p in (foreign / 'chunks').iterdir()) == ['chunk_00000.pt']
    assert sorted(p.name for p in beside.rglob('*')) == ['chunks', 'notes.txt', 'plan.json.tmp-123']


@pytest.mark.ccdc
def test_table_values(table, refcodes):
    out, meta, data = table
    assert sorted(os.listdir(out)) == ['data.pt', 'meta.json']          # chunks merged and removed
    assert data['schema'] == SCHEMA and data['rows'] == len(refcodes) == meta['rows']
    assert data['refcode'] == refcodes and data['family'] == [r[:6] for r in refcodes]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            assert value.shape == (len(refcodes),), key
        elif isinstance(value, list):
            assert len(value) == len(refcodes) and all(isinstance(v, str) for v in value), key

    row = {r: i for i, r in enumerate(refcodes)}
    assert data['found'].tolist() == [r != MISSING for r in refcodes]

    a = row['ACRDIN04']
    assert data['temperature_raw'][a] == 'at 185 K' and data['temperature_k'][a].item() == 185.0
    assert data['r_factor'][a].item() == pytest.approx(3.81)
    assert data['ccdc_number'][a].item() == 259366
    assert data['deposition_date'][a] == '2005-03-29'
    assert data['polymorph'][a] == 'polymorph II'
    assert data['family'][a] == 'ACRDIN' and data['radiation_source'][a] == 'X-ray' and data['database_name'][a]
    assert data['is_powder_study_known'][a] and not data['is_powder_study'][a]
    assert data['message'][a] == ''

    n = row['NUKCOL']
    assert data['is_powder_study'][n] and data['is_powder_study_known'][n]

    m = row[MISSING]
    assert data['message'][m] == 'not in the database'
    assert math.isnan(data['temperature_k'][m]) and math.isnan(data['r_factor'][m])
    assert data['ccdc_number'][m].item() == -1
    for f in ('is_powder_study', 'has_disorder', 'is_polymeric'):
        assert not data[f][m] and not data[f + '_known'][m]
    assert all(data[f][m] == '' for f in ('temperature_raw', 'pressure_raw', 'radiation_source', 'deposition_date',
                                         'polymorph', 'database_name'))

    found = data['found']
    assert (data['ccdc_number'][found] > 0).all()
    assert all(data['deposition_date'][i] for i in range(len(refcodes)) if found[i])

    c = meta['counts']
    assert (c['found'], c['not_found'], c['rows_with_field_errors']) == (17, 1, 0)
    assert c['present']['r_factor'] == int((~data['r_factor'].isnan()).sum())
    assert c['flags']['is_powder_study']['true'] >= 1
    assert [ch['start'] for ch in meta['chunks']] == [0, 5, 10, 15]


@pytest.mark.ccdc
def test_finished_table_is_returned_not_rebuilt(table, refcodes, tmp_path):
    out, meta, _ = table
    stamp = os.path.getmtime(out / 'data.pt')
    assert build_provenance(refcodes, out, chunk_size=5) == meta
    assert os.path.getmtime(out / 'data.pt') == stamp
    with pytest.raises(ValueError, match='other refcodes'):
        build_provenance(refcodes[:3], out)
    with pytest.raises(FileExistsError):
        build_provenance(refcodes, out, resume=False)
    (tmp_path / 'foreign').mkdir()
    (tmp_path / 'foreign' / 'x.txt').write_text('x')
    with pytest.raises(ValueError, match='not a provenance run'):
        build_provenance(refcodes, tmp_path / 'foreign')


@pytest.mark.ccdc
def test_interrupted_run_resumes_from_its_chunks(csd, table, refcodes, tmp_path, monkeypatch):
    """A run stopped after two chunks re-reads only the other two and gives the same table."""
    real = csd_provenance._read_chunk
    calls = []

    def stop_after_two(reader, codes):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append(codes[0])
        return real(reader, codes)

    monkeypatch.setattr(csd_provenance, '_read_chunk', stop_after_two)
    out = tmp_path / 'resumed'
    with pytest.raises(KeyboardInterrupt):
        build_provenance(refcodes, out, chunk_size=5)
    assert sorted(os.listdir(out / 'chunks')) == ['chunk_00000.pt', 'chunk_00001.pt', 'plan.json']
    with pytest.raises(ValueError, match='chunk size'):
        build_provenance(refcodes, out, chunk_size=4)

    calls.clear()
    monkeypatch.setattr(csd_provenance, '_read_chunk', lambda reader, codes: calls.append(codes[0]) or real(reader, codes))
    build_provenance(refcodes, out, chunk_size=5)
    assert calls == [refcodes[10], refcodes[15]]
    _assert_same_table(table[2], load_provenance(out))


@pytest.mark.ccdc
@pytest.mark.parametrize('target', ['plan.json', 'chunk_00002.pt', 'data.pt', 'meta.json'])
def test_killed_run_leaves_temporaries_only_in_chunks_and_resumes(csd, table, refcodes, tmp_path, target):
    """A process killed between writing a temporary file and renaming it resumes to the same, clean table."""
    out = tmp_path / 'killed'
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='4',
               PYTHONPATH=os.pathsep.join(filter(None, [str(root), os.environ.get('PYTHONPATH')])))
    child = subprocess.run([sys.executable, '-c', _KILLED_CHILD, ','.join(refcodes), str(out), target],
                           cwd=root, env=env, capture_output=True, text=True, timeout=300)
    assert child.returncode == 9, child.stderr[-2000:]

    left = sorted(str(p.relative_to(out)).replace(os.sep, '/') for p in out.rglob('*') if p.is_file())
    assert f'chunks/{target}.tmp-' in ' '.join(left), left                # killed after the temporary was written
    assert all(name.startswith('chunks/') or name == 'data.pt' for name in left), left
    if target == 'plan.json':
        assert len(left) == 1, left                                           # nothing but the plan's temporary

    build_provenance(refcodes, out, chunk_size=5)
    assert sorted(os.listdir(out)) == ['data.pt', 'meta.json']
    _assert_same_table(table[2], load_provenance(out))

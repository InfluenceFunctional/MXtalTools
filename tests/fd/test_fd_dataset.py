"""Canonical crystal datasets on disk (``crystal_fd.dataset``): save / load and the converter back to MolCrystalData.

1. Save then load returns every field bit for bit (descriptions, molecules with their hashes, reference handedness,
   status, identifiers, tensor and list columns, manifest), on synthetic canonical rows of several groups with
   molecules of different sizes, a mirrored reference (reference_handedness -1) and a set-aside status;
   ``load_dataset`` keeps 'canonical' rows by default and selects by status name; ``select`` repacks molecules.
2. Refusals: an existing path (also an empty folder) is never overwritten; Z' > 1, a molecule that does not match its
   hash, a column that repeats a core key or has the wrong length, a bad status code, and manifests that miss or repeat
   a source row, name a dataset row twice or with another identifier are each refused (``DatasetError``).
3. Corruption: one changed byte of data.pt fails the digest; a dataset written against another table hash is refused
   unless ``allow_table_mismatch``; an unknown status name is refused; the CSV manifest has one line per source row.
   Code identity: meta.json records ``code_digest()`` (every *.py and data file of the package, README excluded);
   a dataset whose recorded code differs, or that records none, loads with a ``CodeMismatchWarning`` naming the changed
   files, is refused with ``code='raise'`` and loads silently with ``code='ignore'``.
4. The converter: rows from ``to_mol_crystal_data`` built by the live builder (``mol2ucell``) put every atom within
   1e-3 Angstrom of the float64 literal unit cell of the description (``checks.unit_cell_fractional``) - mode 'body'
   with ``std_orientation`` False and True, mode 'literal' with True - in sg 1, 2, 4, 14, 19 and 61, for references in
   the aligned frame (+1) and mirrored references (-1), and for h = -1 descriptions. Controls, each more than 0.1
   Angstrom off: a mirrored reference converted without its reference_handedness and built with
   ``std_orientation=True``; mode 'literal' built with ``std_orientation=False``. Centroids in the builder clip band
   are refused unless ``clip='allow'``; a centroid at -1e-17 (rounding on the 0 face) wraps without entering it.
5. Local data (skips when absent; ``MXT_FD_CSD_DATASET`` or D:/crystal_datasets/csd_fd_v1b): the canonical CSD
   dataset loads with its digest and records a code digest; its manifest names every source row once and its code
   counts equal meta.json's; every row is on the +1 sheet with reference_handedness -1 exactly where mirrored; canonical
   rows of a seeded sample are their own fold and (sg <= 15) their own reduced cell with n_auto <= 1; 150 seeded rows
   convert and build within 1e-3 Angstrom. Derived flags: cell_auto_tol is set exactly on set-aside rows; near_face
   covers every row within the stored tolerance of a cell-chart or centroid face; axis_order_k is 6 / 3 / 1 for
   P212121 / Pbca / Pna21 and 1 outside the orthorhombic groups; coincide groups are numbered 0.. with two or more rows
   of one space group each.
6. Importing and using ``crystal_fd.dataset`` loads neither cctbx nor a GFN module.
"""
import csv
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from mxtaltools.crystal_fd import checks, dataset as fdd, domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
from mxtaltools.crystal_fd.io_mxt import _Units

F64 = torch.float64
NB = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3).to(F64)
#: Angstrom: the live float32 builder against the float64 literal unit cell (gate 3's bound; measured about 1e-5)
BUILD_TOL = 1e-3
CSD_DATASET = Path(os.environ.get('MXT_FD_CSD_DATASET', 'D:/crystal_datasets/csd_fd_v1b'))


# ---------------------------------------------------------------------------
# synthetic rows
# ---------------------------------------------------------------------------

def _aligned(pos, z):
    """The molecule in the live builder's standard frame (align_mol_batch_to_standard_axes at +1), heavy-centred."""
    from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes
    pos = torch.as_tensor(pos, dtype=torch.float32).reshape(-1, 3)
    z = torch.as_tensor(z).long()
    u = _Units(pos.clone(), z, torch.zeros(len(z), dtype=torch.long), 1)
    align_mol_batch_to_standard_axes(u, handedness=None)
    return MoleculeRef.from_coords(u.pos.double(), z)


def _molecules(n, seed):
    """n distinct chiral molecules (5-9 atoms) in the aligned frame."""
    g = np.random.default_rng(seed)
    out = []
    for k in range(n):
        m = 5 + k % 5
        z = np.array([6, 7, 8, 9, 17, 35, 6, 8, 16][:m])
        z[-1] = 1                                    # one hydrogen
        pos = g.normal(size=(m, 3)) * 1.4
        out.append(_aligned(pos, z))
    return out


def _mirror(mol):
    return MoleculeRef.from_coords(mol.pos * torch.tensor([-1.0, 1.0, 1.0], dtype=F64), mol.z)


def _random_canonical(sg, n, seed, h=1):
    """n canonical Z' = 1 descriptions of sg (reduce_cell for sg <= 15, then fold), centroids clear of the clip band."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    fam = 'triclinic' if sg <= 2 else 'monoclinic' if sg <= 15 else 'orthorhombic'
    cells = torch.as_tensor(checks.random_cells(fam, 4 * n, g), dtype=F64)
    cells[:, :3] = torch.as_tensor(rng.uniform(7.0, 14.0, (4 * n, 3)))
    if fam == 'triclinic':
        ok = ops.real_cells(cells)
        cells = cells[ok]
    cells = cells[:n]
    d = CrystalDescription.from_parts(sg, cells, torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g),
                                      torch.full((n,), h, dtype=torch.long))
    if sg <= 15:
        d = ops.reduce_cell(d).desc
    d = domains.fold(d).desc
    return d[~fdd.in_clip_band(d)]


def _dataset_inputs(seed=7):
    descs, mols, rh, status, ident = [], [], [], [], []
    base = _molecules(12, seed)
    for k, sg in enumerate((1, 2, 4, 14, 19, 9)):
        d = _random_canonical(sg, 4, seed + k)
        for i in range(d.num_rows):
            m = base[(2 * k + i) % len(base)]
            mirrored = sg in (4, 19) and i % 2 == 1
            descs.append(d[i])
            mols.append(_mirror(m) if mirrored else m)
            rh.append(-1 if mirrored else 1)
            status.append(1 if i == 3 else 0)
            ident.append(f'X{sg:03d}_{i}')
    return CrystalDescription.cat(descs), mols, torch.tensor(rh), torch.tensor(status), ident


def _manifest(ident, extra_refused=3):
    """Source rows: the dataset rows interleaved with refused rows."""
    M = len(ident) + extra_refused
    order = list(range(M))
    refused = set(order[1::max(1, M // extra_refused)][:extra_refused])
    code, idl, drow, k = [], [], [], 0
    for j in order:
        if j in refused:
            code.append('refused:z_prime_gt1')
            idl.append(f'R{j}')
            drow.append(-1)
        else:
            code.append('canonical')
            idl.append(ident[k])
            drow.append(k)
            k += 1
    return dict(source_row=torch.arange(M), identifier=idl, code=code, dataset_row=torch.tensor(drow),
                detail=dict(z_prime=torch.ones(M, dtype=torch.long), note=['n'] * M))


def _save(tmp_path, name='ds', **kw):
    desc, mols, rh, status, ident = _dataset_inputs()
    args = dict(reference_handedness=rh, status=status, status_names=('canonical', 'set_aside_special_metric'),
                columns=dict(score=torch.arange(len(ident), dtype=F64) / 3, flag=status == 1,
                             smiles=[f's{i}' for i in range(len(ident))]),
                column_docs=dict(score='a float column'), provenance=dict(test=True))
    args.update(kw)
    path = tmp_path / name
    meta = fdd.save_dataset(str(path), desc, mols, ident, **args)
    return path, meta, (desc, mols, rh, status, ident)


# ---------------------------------------------------------------------------
# 1-3: save / load
# ---------------------------------------------------------------------------

def test_save_load_round_trip_bitwise(tmp_path):
    desc, mols, rh, status, ident = _dataset_inputs()
    man = _manifest(ident)
    path, meta, _ = _save(tmp_path, manifest=man)
    assert meta['rows'] == len(ident) and meta['manifest']['rows'] == len(man['code'])
    ds = fdd.load_dataset(str(path), status=None, verify_molecules=True)
    assert len(ds) == len(ident)
    for f in ('sg', 'cell', 't', 'R', 'h'):
        assert torch.equal(getattr(ds.desc, f), getattr(desc, f)), f
    assert torch.equal(ds.reference_handedness, rh) and torch.equal(ds.status, status)
    assert ds.identifier == ident
    for i, m in enumerate(mols):
        got = ds.molecule(i)
        assert torch.equal(got.pos, m.pos) and torch.equal(got.z, m.z) and got.hash == m.hash
    assert torch.equal(ds.columns['score'], torch.arange(len(ident), dtype=F64) / 3)
    assert ds.columns['smiles'] == [f's{i}' for i in range(len(ident))]
    assert ds.meta['columns']['score']['doc'] == 'a float column'
    # default: canonical rows only; select repacks molecules exactly
    can = fdd.load_dataset(str(path))
    keep = torch.nonzero(status == 0)[:, 0]
    assert len(can) == len(keep) and torch.equal(can.index, keep)
    for k, i in enumerate(keep.tolist()):
        assert torch.equal(can.molecule(k).pos, mols[i].pos) and can.identifier[k] == ident[i]
    aside = fdd.load_dataset(str(path), status='set_aside_special_metric')
    assert len(aside) == int((status == 1).sum())
    m2 = fdd.load_manifest(str(path))
    assert m2['code'] == man['code'] and torch.equal(m2['dataset_row'], man['dataset_row'])
    assert m2['detail']['note'] == man['detail']['note']


def test_save_refuses_existing_path_and_bad_inputs(tmp_path):
    desc, mols, rh, status, ident = _dataset_inputs()
    (tmp_path / 'exists').mkdir()
    with pytest.raises(FileExistsError):
        fdd.save_dataset(str(tmp_path / 'exists'), desc, mols, ident)
    path, _, _ = _save(tmp_path, 'once')
    with pytest.raises(FileExistsError):
        fdd.save_dataset(str(path), desc, mols, ident)
    bad = fdd.DatasetError
    zp2 = CrystalDescription.from_parts(desc.sg[:1], desc.cell[:1], desc.t[:1].repeat(1, 2, 1),
                                        desc.R[:1].repeat(1, 2, 1, 1))
    cases = {
        'zp2': lambda p: fdd.save_dataset(p, zp2, mols[:1], ident[:1]),
        'hash': lambda p: fdd.save_dataset(p, desc, [MoleculeRef(pos=mols[0].pos + 1e-3, z=mols[0].z,
                                                                  hash=mols[0].hash)] + mols[1:], ident),
        'core column': lambda p: fdd.save_dataset(p, desc, mols, ident, columns=dict(sg=desc.sg)),
        'column length': lambda p: fdd.save_dataset(p, desc, mols, ident, columns=dict(x=torch.zeros(3))),
        'status': lambda p: fdd.save_dataset(p, desc, mols, ident, status=torch.full((len(ident),), 2)),
        'rh': lambda p: fdd.save_dataset(p, desc, mols, ident, reference_handedness=torch.zeros(len(ident))),
        'lengths': lambda p: fdd.save_dataset(p, desc, mols[:-1], ident),
    }
    man = _manifest(ident)
    m_missing = dict(man, source_row=torch.cat([man['source_row'][:-1], man['source_row'][:1]]))
    m_twice = dict(man, dataset_row=torch.where(man['dataset_row'] == 1, 0, man['dataset_row']))
    m_ident = dict(man, identifier=[('Z' if d == 0 else s) for s, d in zip(man['identifier'],
                                                                            man['dataset_row'].tolist())])
    for name, m in (('manifest missing row', m_missing), ('manifest twice', m_twice), ('manifest ident', m_ident)):
        cases[name] = (lambda mm: (lambda p: fdd.save_dataset(p, desc, mols, ident, manifest=mm)))(m)
    for name, fn in cases.items():
        p = str(tmp_path / f'bad_{name.replace(" ", "_")}')
        with pytest.raises(bad):
            fn(p)
        assert not os.path.exists(p), name                       # nothing is left behind
        assert not any(x.name.startswith(Path(p).name + '.partial') for x in tmp_path.iterdir()), name


def test_load_detects_corruption(tmp_path):
    path, meta, (desc, mols, rh, status, ident) = _save(tmp_path, manifest=_manifest(_dataset_inputs()[4]))
    fdd.load_dataset(str(path))
    with pytest.raises(fdd.DatasetError):
        fdd.load_dataset(str(path), status='no_such_status')
    rows = list(csv.reader(open(path / 'manifest.csv', encoding='utf-8')))
    assert len(rows) == 1 + meta['manifest']['rows'] and rows[0][:4] == ['source_row', 'identifier', 'code',
                                                                          'dataset_row']
    # another table hash
    mj = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    mj['table_hash'] = '0' * 64
    (path / 'meta.json').write_text(json.dumps(mj), encoding='utf-8')
    with pytest.raises(fdd.DatasetError):
        fdd.load_dataset(str(path))
    assert len(fdd.load_dataset(str(path), allow_table_mismatch=True)) == int((status == 0).sum())
    # one changed byte
    raw = bytearray((path / 'data.pt').read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (path / 'data.pt').write_bytes(bytes(raw))
    with pytest.raises(fdd.DatasetError):
        fdd.load_dataset(str(path), allow_table_mismatch=True)


def test_code_digest_recorded_and_checked(tmp_path):
    import warnings
    path, meta, (desc, mols, rh, status, ident) = _save(tmp_path)
    cur = fdd.code_digest()
    assert meta['code']['digest'] == cur['digest'] and meta['code']['files'] == cur['files']
    assert 'dataset.py' in cur['files'] and 'ops.py' in cur['files'] and 'data/fd_tables_v1.json' in cur['files']
    assert not any(f.endswith('README.md') for f in cur['files'])
    with warnings.catch_warnings():
        warnings.simplefilter('error')                           # the writing code: no warning
        fdd.load_dataset(str(path), code='raise')
    mj = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    mj['code']['files']['ops.py'] = '0' * 64                     # as if ops.py had changed since the write
    mj['code']['files']['gone.py'] = '1' * 64
    mj['code']['digest'] = '2' * 64
    (path / 'meta.json').write_text(json.dumps(mj), encoding='utf-8')
    assert fdd.code_mismatch(mj) == ['changed: ops.py', 'removed: gone.py']
    with pytest.warns(fdd.CodeMismatchWarning, match='changed: ops.py'):
        assert len(fdd.load_dataset(str(path))) == int((status == 0).sum())
    with pytest.raises(fdd.DatasetError, match='changed: ops.py'):
        fdd.load_dataset(str(path), code='raise')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fdd.load_dataset(str(path), code='ignore')
    del mj['code']                                               # a dataset that records no digest
    (path / 'meta.json').write_text(json.dumps(mj), encoding='utf-8')
    assert fdd.code_mismatch(mj) is None
    with pytest.warns(fdd.CodeMismatchWarning, match='no crystal_fd code digest'):
        fdd.load_dataset(str(path))
    with pytest.raises(ValueError):
        fdd.load_dataset(str(path), code='sometimes')


def test_code_digest_names_changed_files(tmp_path):
    pkg = tmp_path / 'pkg'
    (pkg / 'data').mkdir(parents=True)
    (pkg / 'a.py').write_bytes(b'x = 1\n')
    (pkg / 'data' / 't.json').write_bytes(b'{}')
    (pkg / 'README.md').write_bytes(b'not code')
    d0 = fdd.code_digest(str(pkg))
    assert sorted(d0['files']) == ['a.py', 'data/t.json']
    (pkg / 'README.md').write_bytes(b'still not code')
    assert fdd.code_digest(str(pkg))['digest'] == d0['digest']
    (pkg / 'a.py').write_bytes(b'x = 2\n')
    (pkg / 'b.py').write_bytes(b'')
    d1 = fdd.code_digest(str(pkg))
    assert d1['digest'] != d0['digest']
    assert fdd.code_mismatch(dict(code=d0), d1) == ['added: b.py', 'changed: a.py']


def test_added_files_alone_do_not_warn(tmp_path, monkeypatch):
    """A module added to the package, with every recorded file unchanged, cannot change what the recorded code
    computes: check_code neither warns nor raises, and still lists it. A changed file does both."""
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    (pkg / 'a.py').write_bytes(b'x = 1\n')
    digest = fdd.code_digest
    d0 = digest(str(pkg))
    (pkg / 'b.py').write_bytes(b'y = 2\n')
    d1 = digest(str(pkg))
    monkeypatch.setattr(fdd, 'code_digest', lambda package_dir=None: d1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', fdd.CodeMismatchWarning)
        assert fdd.check_code('ds', dict(code=d0), code='warn') == ['added: b.py']
        assert fdd.check_code('ds', dict(code=d0), code='raise') == ['added: b.py']
    (pkg / 'a.py').write_bytes(b'x = 3\n')
    d2 = digest(str(pkg))
    monkeypatch.setattr(fdd, 'code_digest', lambda package_dir=None: d2)
    with pytest.warns(fdd.CodeMismatchWarning, match='changed: a.py'):
        fdd.check_code('ds', dict(code=d0), code='warn')
    with pytest.raises(fdd.DatasetError):
        fdd.check_code('ds', dict(code=d0), code='raise')


# ---------------------------------------------------------------------------
# 4: the converter
# ---------------------------------------------------------------------------

def _builder_dev(desc, mols, rh, mode, std_orientation):
    """[B] largest atom deviation (Angstrom), live builder unit cell against the float64 literal unit cell."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    rows = fdd.to_mol_crystal_data(desc, mols, rh, mode=mode)
    b = collate_data_list(rows)
    with torch.no_grad():
        b.mol2ucell(std_orientation=std_orientation)
    pos, bat = b.unit_cell_pos.double(), b.unit_cell_batch
    d = desc.double()
    out = torch.empty(d.num_rows, dtype=F64)
    for i in range(d.num_rows):
        T = d[i].cell_matrix()[0]
        ref = checks.unit_cell_fractional(d[i], mols[i])[0]                       # [k, n, 3]
        got = torch.linalg.solve(T, pos[bat == i].T).T.reshape(-1, len(mols[i].z), 3)
        df = got[:, None] - ref[None]
        df = df - torch.round(df)
        dist = torch.einsum('ij,abnkj->abnki', T, df[:, :, :, None] + NB).norm(dim=-1).amin(-1).amax(-1)
        out[i] = torch.maximum(dist.amin(1).amax(), dist.amin(0).amax())
    return out


@pytest.mark.parametrize('sg', [1, 2, 4, 14, 19, 61])
def test_converter_builds_the_described_crystal(sg):
    base = _molecules(6, 100 + sg)
    d = _random_canonical(sg, 12, 200 + sg)
    n = d.num_rows
    mirrored = torch.arange(n) % 2 == 1
    mols = [_mirror(base[i % 6]) if bool(mirrored[i]) else base[i % 6] for i in range(n)]
    rh = torch.where(mirrored, -1, 1)
    for mode, so in (('body', False), ('body', True), ('literal', True)):
        dev = _builder_dev(d, mols, rh, mode, so)
        assert float(dev.max()) <= BUILD_TOL, (mode, so, float(dev.max()))
    # h = -1 descriptions (off the sheet) build as described too
    dm = d.replace(h=-d.h)
    for mode, so in (('body', False), ('body', True), ('literal', True)):
        assert float(_builder_dev(dm, mols, rh, mode, so).max()) <= BUILD_TOL, (mode, so)
    # controls
    mi = torch.nonzero(mirrored)[:, 0]
    ctl = _builder_dev(d[mi], [mols[i] for i in mi.tolist()], torch.ones(len(mi), dtype=torch.long), 'body', True)
    assert float(ctl.min()) > 0.1, float(ctl.min())
    ctl2 = _builder_dev(d, mols, rh, 'literal', False)
    assert float(ctl2.min()) > 0.1, float(ctl2.min())


def test_converter_rows_and_clip_band():
    mol = _molecules(1, 5)[0]
    d = _random_canonical(14, 3, 9)
    rows = fdd.to_mol_crystal_data(d, mol, torch.tensor([1, -1, 1]), identifier=['a', 'b', 'c'],
                                   extra=dict(smiles=['x', 'y', 'z']))
    assert [r.identifier for r in rows] == ['a', 'b', 'c'] and rows[1].smiles == 'y'
    assert int(rows[1].aunit_handedness.reshape(-1)[0]) == -1 and int(rows[1].reference_handedness[0]) == -1
    assert all(int(r.sg_ind) == 14 for r in rows)
    band = d.replace(t=d.t.clone())
    band.t[0, 0, 0] = 1 - 5e-5
    assert bool(fdd.in_clip_band(band)[0]) and not bool(fdd.in_clip_band(band)[1:].any())
    with pytest.raises(ValueError):
        fdd.to_mol_crystal_data(band, mol)
    assert len(fdd.to_mol_crystal_data(band, mol, clip='allow')) == 3
    edge = d.replace(t=d.t.clone())
    edge.t[0, 0, 1] = -1e-17                                    # rounding of a centroid on the 0 face
    assert not bool(fdd.in_clip_band(edge).any())
    assert float(_builder_dev(edge[:1], [mol], torch.ones(1, dtype=torch.long), 'body', False)[0]) <= BUILD_TOL
    with pytest.raises(ValueError):
        fdd.to_mol_crystal_data(d, mol, mode='posed')


# ---------------------------------------------------------------------------
# 5: the canonical CSD dataset (local data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def csd():
    if not (CSD_DATASET / 'meta.json').exists():
        pytest.skip(f'{CSD_DATASET} is absent (local data)')
    return fdd.load_dataset(str(CSD_DATASET), status=None)


def test_csd_dataset_manifest_and_flags(csd):
    meta = csd.meta
    assert sum(meta['status_counts'].values()) == len(csd) == meta['rows']
    man = fdd.load_manifest(str(CSD_DATASET))
    M = len(man['code'])
    assert torch.equal(torch.sort(man['source_row']).values, torch.arange(M))
    counts = {}
    for c in man['code']:
        counts[c] = counts.get(c, 0) + 1
    assert counts == meta['manifest']['codes'] == meta['provenance']['code_counts']
    named = man['dataset_row'][man['dataset_row'] >= 0]
    assert torch.equal(torch.sort(named).values, torch.arange(len(csd)))
    assert bool((csd.desc.h == 1).all())
    assert torch.equal(csd.reference_handedness, torch.where(csd.columns['mirrored'], -1, 1))
    assert bool(csd.columns['in_domain'].all())
    st = torch.tensor([csd.status_names.index(c) for c in ('canonical', 'set_aside_special_metric')])
    nauto = csd.columns['cell_n_auto']
    assert bool((nauto[csd.status == st[1]] > 1).all()) and bool((nauto[csd.status == st[0]] <= 1).all())
    assert len(meta['code']['digest']) == 64 and 'ops.py' in meta['code']['files']


def test_csd_dataset_derived_flags(csd):
    C, sg = csd.columns, csd.desc.sg
    aside = csd.status == csd.status_names.index('set_aside_special_metric')
    tols = csd.meta['provenance']['cell_auto_tols']
    cat = C['cell_auto_tol']
    assert bool(torch.isnan(cat[~aside]).all()) and set(cat[aside].tolist()) <= set(tols)
    # near_face covers every row within the tolerance of a cell-chart or centroid face
    tol = csd.meta['provenance']['near_face_tol']
    kind = C['near_face_kind']
    assert torch.equal(C['near_face'], kind != 0)
    cell_near = (sg <= 15) & ((C['cell_face_dist'] <= tol) | (C['cell_n_tied'] > 1))
    assert torch.equal(cell_near, (kind & 1) != 0)
    assert torch.equal(C['fold_slack'] <= tol, (kind & 2) != 0)
    # axis orders that keep the symbol: the ITA setting counts of P212121, Pbca, Pna21; none outside orthorhombic
    k = C['axis_order_k'].long()
    for g, want in ((19, 6), (61, 3), (33, 1)):
        assert bool((k[sg == g] == want).all()), g
    assert bool((k[(sg <= 15) | (sg >= 75)] == 1).all())
    assert torch.equal(C['axis_order_gap'], k > 1)
    assert not bool(C['cell_canonical'][sg >= 16].any()) and bool(C['cell_canonical'][sg <= 15].all())
    # coincide groups: ids 0..G-1, each at least two rows of one space group
    g = C['coincide_group']
    ids, n = torch.unique(g[g >= 0], return_counts=True)
    assert torch.equal(ids, torch.arange(len(ids))) and bool((n >= 2).all())
    for i in ids[:200].tolist():
        assert len(torch.unique(sg[g == i])) == 1


def test_csd_dataset_rows_are_canonical_and_build(csd):
    g = torch.Generator().manual_seed(1017)
    can = torch.nonzero(csd.status == csd.status_names.index('canonical'))[:, 0]
    rows = can[torch.randperm(len(can), generator=g)[:2000]]
    d = csd.desc[rows]
    f = domains.fold(d)
    assert float((f.desc.t - d.t).abs().max()) <= 1e-9 and float((f.desc.R - d.R).abs().max()) <= 1e-9
    red = torch.nonzero(d.sg <= 15)[:, 0]
    r = ops.reduce_cell(d[red])
    assert bool((r.N == torch.eye(3, dtype=torch.long)).all()) and bool((r.n_auto <= 1).all())
    assert float((r.desc.cell - d[red].cell).abs().max()) <= 1e-9
    sub = rows[:150]
    sub = sub[~csd.columns['clip_band'][sub]]
    mols = csd.molecules(sub)
    dev = _builder_dev(csd.desc[sub], mols, csd.reference_handedness[sub], 'body', False)
    assert float(dev.max()) <= BUILD_TOL, float(dev.max())


# ---------------------------------------------------------------------------
# 6: import isolation
# ---------------------------------------------------------------------------

_FRESH = '''
import sys, tempfile, os, torch
from mxtaltools.crystal_fd import dataset as fdd
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
d = CrystalDescription.from_parts(2, torch.tensor([[7.0, 8.0, 9.0, 1.7, 1.8, 1.9]], dtype=torch.float64),
                                  torch.full((1, 3), 0.1, dtype=torch.float64), torch.eye(3, dtype=torch.float64)[None])
m = MoleculeRef.from_coords(torch.tensor([[0.0, 0, 0], [1.2, 0, 0], [0, 1.3, 0.2]]), torch.tensor([6, 8, 1]))
p = os.path.join(tempfile.mkdtemp(), 'ds')
fdd.save_dataset(p, d, [m], ['x'])
ds = fdd.load_dataset(p)
rows = ds.to_mol_crystal_data()
bad = [k for k in sys.modules if k.split('.')[0] in ('cctbx', 'libtbx', 'scitbx', 'boost_adaptbx')]
assert not bad, bad
gfn = sys.argv[1]
files = [str(getattr(v, '__file__', None) or '').replace(chr(92), '/').lower() for v in list(sys.modules.values())]
loaded = [f for f in files if f.startswith(gfn)]
assert not loaded, loaded
'''


def test_dataset_imports_no_cctbx_or_gfn():
    root = Path(fdd.__file__).parents[2]
    gfn = str(root.parent / 'gfn_diffusion').replace(chr(92), '/').lower()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONPATH=str(root))
    r = subprocess.run([sys.executable, '-c', _FRESH, gfn], env=env, capture_output=True, text=True, timeout=300,
                       cwd=str(root))
    assert r.returncode == 0, r.stderr[-2000:]

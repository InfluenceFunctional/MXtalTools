"""Build the stage-0 characterization fixtures (``data/fixtures.pt``) from real rows on D: and a few
RDKit-embedded audit molecules. Run once; the tests never read D:.

    cd MXtalTools
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python tests/characterization/build_fixtures.py            # first build
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python tests/characterization/build_fixtures.py --check    # rebuild in memory, compare, write nothing
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python tests/characterization/build_fixtures.py --regenerate   # overwrite (review + justify)

Deterministic: the rows are named (identifier and, for the ingest store, chunk and row), every
RDKit embedding is seeded, and each selection criterion is asserted. Refuses to overwrite an
existing fixture file unless ``--regenerate`` is given. Writes only ``data/fixtures.pt`` and
``data/fixtures_manifest.json``. The goldens depend on these fixtures: regenerating them means
regenerating the goldens too (``build_goldens.py --regenerate``), reviewed in the same change.

Sources (read only):
- F1: ``D:/crystal_datasets/csd_fd_v5`` (crystal_fd dataset; ``to_mol_crystal_data(mode='body')``).
- F3: the r2 ingest store ``D:/crystal_datasets/ingest_csd_v1_r2/csd_v1-6080f8b8a11a/chunks``.
- F3b: ``zp_doubling.double_zp1_crystal`` of the two sg 14 F1 rows (computed, not read).
- AUDIT: the hand-posed crystals of the 2026-10-22 audit probes (``artifacts/fd_design_oct15/
  units_audit_oct22``), molecules re-embedded with the probes' seeds.
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charlib  # noqa: E402

FD_PATH = 'D:/crystal_datasets/csd_fd_v5'
STORE_CHUNKS = 'D:/crystal_datasets/ingest_csd_v1_r2/csd_v1-6080f8b8a11a/chunks'

#: Z'=1 rows: C1 molecules (no point symmetry), 18-30 atoms, organic elements, sg 14 / 2 / 9 / 14
F1_IDENTIFIERS = ('PYRAZB21', 'YIRLEU', 'OYIQIX', 'HIFQUJ')
F1_EXPECT_SG = {'PYRAZB21': 14, 'YIRLEU': 2, 'OYIQIX': 9, 'HIFQUJ': 14}
#: Z'=2 ingestion rows: (chunk, row, identifier, sg); one compound, per-unit conformers differing by > 0.3 A
F3_ROWS = (('chunk_000022', 35, 'AHUMIZ', 2), ('chunk_000013', 329, 'AFEVOA', 14))
ORGANIC = {1, 6, 7, 8, 9, 16, 17}


# ------------------------------------------------------------------------------------------------
# selection checks
# ------------------------------------------------------------------------------------------------

def _kabsch_rmsd(P, Q, improper=False):
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, -d if improper else d])
    R = Vt.T @ D @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))


def point_symmetric(pos, z, tol=0.1) -> bool:
    """True when some graph automorphism of the molecule (bonds from distances) is realised by a
    rotation, or the identity or any automorphism by an improper rotation, within ``tol`` A RMSD."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds
    RDLogger.DisableLog('rdApp.*')
    pos = np.asarray(pos, dtype=float)
    z = [int(v) for v in z]
    pt = Chem.GetPeriodicTable()
    block = f'{len(z)}\n\n' + '\n'.join(f'{pt.GetElementSymbol(e)} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}'
                                         for e, p in zip(z, pos))
    m = Chem.MolFromXYZBlock(block)
    rdDetermineBonds.DetermineConnectivity(m)
    ident = tuple(range(len(z)))
    for p in m.GetSubstructMatches(m, uniquify=False, useChirality=False, maxMatches=20000):
        Q = pos[list(p)]
        if p != ident and _kabsch_rmsd(Q, pos) < tol:
            return True
        if _kabsch_rmsd(Q, pos, improper=True) < tol:
            return True
    return False


def _sha_meta(path):
    try:
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)
        return meta.get('files', {}).get('data.pt', {}).get('sha256')
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------
# F1
# ------------------------------------------------------------------------------------------------

def build_f1(log):
    from mxtaltools.crystal_fd.dataset import load_dataset
    fd = load_dataset(FD_PATH, status=None, verify=True, code='warn')
    ident = fd.identifier
    recs = []
    for name in F1_IDENTIFIERS:
        hits = [i for i, s in enumerate(ident) if s == name]
        assert len(hits) == 1, (name, hits)
        i = hits[0]
        assert fd.status_of(i) == 'canonical', (name, fd.status_of(i))
        mol = fd.molecule(i)
        z = torch.as_tensor(mol.z).long()
        assert set(z.tolist()) <= ORGANIC, name
        assert 18 <= len(z) <= 30, (name, len(z))
        assert not point_symmetric(torch.as_tensor(mol.pos).numpy(), z.tolist()), f'{name} has point symmetry'
        row = fd.to_mol_crystal_data(rows=[i], mode='body')[0]
        assert int(row.sg_ind) == F1_EXPECT_SG[name], (name, int(row.sg_ind))
        rec = {
            'identifier': name, 'sg': torch.tensor(int(row.sg_ind)),
            'z': row.z.clone(), 'pos': row.pos.clone(),
            'cell_lengths': row.cell_lengths.reshape(3).clone(), 'cell_angles': row.cell_angles.reshape(3).clone(),
            'aunit_centroid': row.aunit_centroid.reshape(3).clone(),
            'aunit_orientation': row.aunit_orientation.reshape(3).clone(),
            'aunit_handedness': row.aunit_handedness.reshape(1).clone(),
            'reference_handedness': row.reference_handedness.reshape(1).clone(),
            'smiles': fd.columns['smiles'][i],
            'source': {'dataset': FD_PATH, 'dataset_row': int(fd.index[i]), 'mol_hash': fd.mol_hash[i],
                       'data_sha256': fd.meta.get('files', {}).get('data.pt', {}).get('sha256')},
        }
        # the fixture must rebuild the very row to_mol_crystal_data returns
        again = charlib.f1_row(rec)
        for k in row.keys():
            a, b = row[k], again[k] if k in again._store else None
            if torch.is_tensor(a):
                assert torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b), (name, k)
            elif isinstance(a, np.ndarray):
                assert np.array_equal(a, b), (name, k)
            else:
                assert a == b, (name, k, a, b)
        assert set(row.keys()) == set(again.keys()), (name, set(row.keys()) ^ set(again.keys()))
        log(f'F1 {name}: dataset row {int(fd.index[i])}, sg {int(row.sg_ind)}, {len(z)} atoms, '
            f'handedness {int(row.aunit_handedness.reshape(-1)[0])}, C1')
        recs.append(rec)
    return recs


# ------------------------------------------------------------------------------------------------
# F3
# ------------------------------------------------------------------------------------------------

def build_f3(log):
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    from mxtaltools.constants.space_group_info import SYM_OPS
    recs = []
    for chunk, row, name, sg in F3_ROWS:
        path = os.path.join(STORE_CHUNKS, chunk)
        b = MolCrystalData.load(path, verify=True)
        assert b.identifier[row] == name, (chunk, row, b.identifier[row])
        assert int(b.status[row]) == 0 and int(b.z_prime[row]) == 2 and int(b.sg_ind[row]) == sg, name
        assert int(b.setting[row]) == 0 and not bool(b.nonstandard_symmetry[row]), name
        a, e = int(b.ptr[row]), int(b.ptr[row + 1])
        z = b.z[a:e].long().clone()
        pos = b.pos[a:e].double().clone()
        k = (e - a) // 2
        assert torch.equal(z[:k], z[k:]), name
        assert set(z.tolist()) <= ORGANIC, name
        dconf = float((torch.cdist(pos[:k], pos[:k]) - torch.cdist(pos[k:], pos[k:])).abs().max())
        assert dconf > 0.3, (name, dconf)
        assert not point_symmetric(pos[:k].numpy(), z[:k].tolist()), f'{name} unit 0 has point symmetry'
        ops = torch.as_tensor(np.asarray(b.symmetry_operators[row]), dtype=torch.float64).clone()
        standard = np.stack(SYM_OPS[sg]).astype(np.float64)
        rec = {
            'identifier': name, 'sg': torch.tensor(sg), 'z': z, 'pos': pos,
            'cell_lengths': b.cell_lengths[row].double().clone(), 'cell_angles': b.cell_angles[row].double().clone(),
            'ops': ops, 'z_prime': torch.tensor(2),
            'aunit_centroid': b.aunit_centroid[row, :6].double().clone(),
            'aunit_orientation': b.aunit_orientation[row, :6].double().clone(),
            'aunit_handedness': b.aunit_handedness[row, :2].long().clone(),
            'unit_radius': b.unit_radius[row, :2].double().clone(), 'unit_mass': b.unit_mass[row, :2].double().clone(),
            'unit_volume': b.unit_volume[row, :2].double().clone(),
            'unit_smiles': [str(s) for s in b.unit_smiles[row][:2]],
            'mol_ind': b.mol_ind[a:e].long().clone(),
            'source': {'store_chunk': path, 'row': row, 'rebuild_max_dev': float(b.rebuild_max_dev[row]),
                       'ops_equal_SYM_OPS': bool(ops.shape == standard.shape and np.allclose(ops.numpy(), standard)),
                       'per_unit_conformer_max_internal_distance_difference': dconf},
        }
        log(f'F3 {name}: {chunk} row {row}, sg {sg}, {e - a} atoms, handedness {rec["aunit_handedness"].tolist()}, '
            f'per-unit conformers differ by {dconf:.3f} A, centroid norms '
            f'{[round(float(x), 3) for x in rec["aunit_centroid"].reshape(2, 3).norm(dim=1)]}')
        recs.append(rec)
    built = charlib.f3_batch(recs)
    assert (built.status == 0).all(), built.message
    for j, rec in enumerate(recs):
        assert torch.equal(built.pos[built.ptr[j]:built.ptr[j + 1]], rec['pos'].float()), rec['identifier']
    return recs


# ------------------------------------------------------------------------------------------------
# F3b
# ------------------------------------------------------------------------------------------------

def build_f3b(f1_recs, log):
    from mxtaltools.crystal_building.zp_doubling import double_zp1_crystal, _molecule_of
    recs = []
    for rec in f1_recs:
        if int(rec['sg']) != 14:
            continue
        parent = charlib.f1_row(rec)
        doubled, params = double_zp1_crystal(parent, reduce=True, origin='best', check=True)
        mol = _molecule_of(parent)
        out = {
            'identifier': rec['identifier'], 'sg': torch.tensor(14),
            'mol': {'z': mol.z.clone(), 'pos': mol.pos.clone(), 'x': mol.x.clone(), 'identifier': rec['identifier'],
                    'mol_volume': mol.mol_volume.clone(), 'mass': mol.mass.clone(), 'radius': mol.radius.clone()},
            'cell_lengths': params['lengths'].float().clone(), 'cell_angles': params['angles'].float().clone(),
            'aunit_centroid': params['centroid'].reshape(-1).float().clone(),
            'aunit_orientation': params['rotvec'].reshape(-1).float().clone(),
            'aunit_handedness': params['handedness'].float().clone(),
            'source': {'parent': rec['identifier'], 'how': 'zp_doubling.double_zp1_crystal(reduce=True, origin=best, check=True)'},
        }
        again = charlib.f3b_row(out)
        assert set(again.keys()) == set(doubled.keys()), set(again.keys()) ^ set(doubled.keys())
        for k in doubled.keys():
            a, b = doubled[k], again[k]
            if torch.is_tensor(a):
                assert a.dtype == b.dtype and torch.equal(a, b), (rec['identifier'], k)
            elif isinstance(a, np.ndarray):
                assert np.array_equal(a, b), k
            else:
                assert a == b, (k, a, b)
        log(f'F3b {rec["identifier"]}: doubled to Z\'=2, cell {[round(float(x), 3) for x in out["cell_lengths"]]}, '
            f'handedness {out["aunit_handedness"].tolist()}')
        recs.append(out)
    return recs


# ------------------------------------------------------------------------------------------------
# AUDIT: the hand-posed crystals of the audit probes
# ------------------------------------------------------------------------------------------------

_EMBED_SEQUENCE = r'''
import os, sys, json, random
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np, torch
from mxtaltools.dataset_utils.data_classes import MolData
out = []
for smi, seed, charges in json.loads(sys.argv[1]):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = MolData.from_smiles(smi, do_mol_analysis=True, compute_partial_charges=charges)
    out.append({'z': m.z.tolist(), 'pos': m.pos.tolist(), 'x': m.x.tolist()})
sys.stdout.write('@@EMBEDDED@@' + json.dumps(out))
'''


def _embed_sequence(sequence):
    """MolData.from_smiles of each (smiles, seed, charges), in order, in a FRESH python process.

    from_smiles embeds with RDKit's unseeded EmbedMolecule, whose generator state persists across
    calls within a process: a probe's geometry depends on every embedding made before it in that
    process, not only on the torch / numpy seeds it set. Replaying the probe's own sequence in a
    new process reproduces its molecules exactly (deterministic for a given RDKit build)."""
    import subprocess
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1',
               PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    res = subprocess.run([sys.executable, '-c', _EMBED_SEQUENCE, json.dumps(sequence)], capture_output=True,
                         text=True, env=env, check=True, cwd=root)
    payload = json.loads(res.stdout.split('@@EMBEDDED@@', 1)[1])
    out = []
    for (smi, seed, charges), m in zip(sequence, payload):
        out.append({'z': torch.tensor(m['z'], dtype=torch.long), 'pos': torch.tensor(m['pos'], dtype=torch.float32),
                    'x': torch.tensor(m['x'], dtype=torch.float32), 'smiles': smi,
                    'how': f'MolData.from_smiles(seed {seed}, compute_partial_charges={charges}), embedding '
                           f'{len(out) + 1} of the fresh-process sequence {[s[0] for s in sequence]}'})
    return out


def _rdkit_mmff(smi, seed):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    z = torch.tensor([a.GetAtomicNum() for a in m.GetAtoms()], dtype=torch.long)
    pos = torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float32)
    return {'z': z, 'pos': pos, 'x': torch.zeros(len(z)), 'smiles': smi,
            'how': f'RDKit EmbedMolecule(randomSeed={seed}) + MMFFOptimizeMolecule'}


def _crystal(mols, sg, lengths, angles, cents, rotvecs, hand, source):
    return {'mols': mols, 'sg': torch.tensor(sg), 'cell_lengths': torch.tensor(lengths, dtype=torch.float32),
            'cell_angles': torch.tensor(angles, dtype=torch.float32),
            'aunit_centroid': torch.tensor(cents, dtype=torch.float32).flatten(),
            'aunit_orientation': torch.tensor(rotvecs, dtype=torch.float32).flatten(),
            'aunit_handedness': torch.tensor(hand, dtype=torch.long), 'source': source}


def build_audit(log):
    half = math.pi / 2
    deg = math.pi / 180
    phen = 'CCOc1ccc(NC(C)=O)cc1'
    pyr = 'O=c1cccc[nH]1'
    # verify/v1_index_levels_and_reach.py embeds CCO, CCN, phenacetin, pyridone, pyridone in this order
    _, _, phen0, pyr0, pyr3 = _embed_sequence([['CCO', 0, False], ['CCN', 1, False], [phen, 0, False],
                                               [pyr, 0, False], [pyr, 3, False]])
    # builders/probe1b_missing_pair.py embeds phenacetin first
    (phen1,) = _embed_sequence([[phen, 1, True]])
    # builders/probe2_two_compounds.py embeds pyridone then fluorobenzene first
    pyr0c, flb0c = _embed_sequence([[pyr, 0, True], ['Fc1ccccc1', 0, True]])
    eth = _rdkit_mmff('CCO', 3)
    ether = _rdkit_mmff('COC', 3)
    meth = _rdkit_mmff('C', 3)
    prop = _rdkit_mmff('CCC', 3)
    out = {
        'phen_p1': _crystal([phen0], 2, [7.0, 8.5, 9.5], [1.4, 1.6, 1.9], [0.21, 0.33, 0.18], [0.4, -0.7, 1.1], [1],
                            'verify/v1 (b): phenacetin P-1; misses 1 pair at cutoff 10'),
        'phen_p21c': _crystal([phen0], 14, [13.0, 9.6, 7.1], [half, 1.75, half], [0.2, 0.1, 0.3], [1.0, 0.3, -0.5], [1],
                              'verify/v1 (b): phenacetin P21/c; misses 4 pairs at cutoff 6'),
        'pyr_p21c': _crystal([pyr0], 14, [7.4, 5.6, 11.5], [half, 1.70, half], [0.25, 0.15, 0.2], [0.6, 1.2, 0.2], [1],
                             'verify/v1 (b, c): pyridone P21/c; misses 2 pairs at cutoff 10'),
        'phen_probe1b': _crystal([phen1], 14, [14.0, 9.5, 16.0], [half, 1.9, half], [0.3, 0.2, 0.1], [0.4, 0.9, 1.5],
                                 [1], 'builders/probe1b: phenacetin sg14; one H...H pair at 5.81 A missed at cutoff 6'),
        'pyr_zp2': _crystal([pyr0, pyr3], 14, [10.0, 7.5, 12.0], [half, 1.85, half],
                            [0.12, 0.10, 0.20, 0.40, 0.15, 0.35], [0.3, 0.5, 1.2, 1.0, -0.4, 2.0], [1, 1],
                            "verify/v1 (c): the Z'=2 batch-mate of pyr_p21c"),
        'two_eth_meth': _crystal([eth, meth], 2, [8.0, 9.0, 10.0], [90 * deg, 95 * deg, 90 * deg],
                                 [[0.2, 0.25, 0.3], [0.65, 0.7, 0.2]], [[0.3, 0.2, 1.0], [1.2, -0.4, 0.5]], [1, 1],
                                 'analysis/probe_zp2_analysis P5: ethanol (9) + methane (5) split 7/7'),
        'two_eth_prop': _crystal([eth, prop], 2, [8.0, 9.0, 10.0], [90 * deg, 95 * deg, 90 * deg],
                                 [[0.2, 0.25, 0.3], [0.65, 0.7, 0.2]], [[0.3, 0.2, 1.0], [1.2, -0.4, 0.5]], [1, 1],
                                 "fix review 2026-10-22: ethanol (9) + propane (11) = 20, split 10/10; two_eth_meth's "
                                 "cell and slots with >= 3 non-collinear heavy atoms per unit, so each unit's frame "
                                 "(std_orientation) is defined, unlike methane's"),
        'two_eth_ether': _crystal([eth, ether], 2, [8.0, 9.0, 10.0], [90 * deg, 95 * deg, 90 * deg],
                                  [[0.2, 0.25, 0.3], [0.65, 0.7, 0.2]], [[0.3, 0.2, 1.0], [1.2, -0.4, 0.5]], [1, 1],
                                  'analysis/probe_zp2_analysis P4: ethanol + dimethyl ether, 4 of 36 MLIP elements wrong'),
        'two_pyr_flb': _crystal([pyr0c, flb0c], 14, [10.0, 7.5, 12.0], [half, 1.85, half],
                                [0.12, 0.10, 0.20, 0.40, 0.15, 0.35], [0.3, 0.5, 1.2, 1.0, -0.4, 2.0], [1, 1],
                                'builders/probe2 (a): pyridone + fluorobenzene, 8 of 96 MLIP elements wrong'),
    }
    for k, v in out.items():
        log(f'AUDIT {k}: sg {int(v["sg"])}, units {[len(m["z"]) for m in v["mols"]]} atoms; {v["source"]}')
    return out


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------

def build_all(log):
    with charlib.deterministic():
        f1 = build_f1(log)
        f3 = build_f3(log)
        f3b = build_f3b(f1, log)
        audit = build_audit(log)
    return {'F1': f1, 'F3': f3, 'F3b': f3b, 'AUDIT': audit}


def same(a, b, path='') -> list:
    """Paths at which two fixture structures differ (tensors bit for bit)."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a, b)):
            return [path]
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f'{path}/{k} (only one side)' for k in set(a) ^ set(b)]
        for k in a:
            if k in b:
                out += same(a[k], b[k], f'{path}/{k}')
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f'{path} (length)']
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += same(x, y, f'{path}[{i}]')
        return out
    return [] if a == b else [path]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--regenerate', action='store_true', help='overwrite the existing fixture files')
    g.add_argument('--check', action='store_true', help='rebuild in memory and compare with the saved file')
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s)
        lines.append(s)

    data = build_all(log)
    out = charlib.assert_inside_data(charlib.FIXTURES_PATH)
    man = charlib.assert_inside_data(charlib.FIXTURES_MANIFEST)
    if args.check:
        saved = torch.load(out, weights_only=True)
        diff = same({k: saved[k] for k in data}, data)
        added = [d for d in diff if d.endswith('(only one side)')]
        changed = [d for d in diff if d not in added]
        if added:
            print(f'entries present on one side only (added or removed by a generator change): {added}')
        print('every fixture entry present in both rebuilds identically' if not changed
              else f'fixtures DIFFER at {changed[:20]}')
        sys.exit(1 if diff else 0)
    manifest = {
        'what': 'stage-0 characterization fixtures; see README.md',
        'built_with': charlib.environment(),
        'sources': {'F1': FD_PATH, 'F1_data_sha256': _sha_meta(FD_PATH), 'F3': STORE_CHUNKS},
        'log': lines,
        'F1': [{'identifier': r['identifier'], 'sg': int(r['sg']), 'atoms': len(r['z']), **r['source']} for r in data['F1']],
        'F3': [{'identifier': r['identifier'], 'sg': int(r['sg']), 'atoms': len(r['z']), **r['source']} for r in data['F3']],
        'F3b': [{'identifier': r['identifier'], **r['source']} for r in data['F3b']],
        'AUDIT': {k: {'sg': int(v['sg']), 'units': [{'smiles': m['smiles'], 'atoms': len(m['z']), 'how': m['how']}
                                                   for m in v['mols']], 'source': v['source']}
                  for k, v in data['AUDIT'].items()},
    }
    mode = 'wb' if args.regenerate else 'xb'
    try:
        with open(out, mode) as f:
            torch.save(data, f)
    except FileExistsError:
        sys.exit(f'{out} exists; pass --regenerate to overwrite it (and regenerate the goldens), or --check')
    with open(man, 'w' if args.regenerate else 'x') as f:
        json.dump(manifest, f, indent=1)
    print(f'wrote {out} ({os.path.getsize(out)} bytes) and {man}')


if __name__ == '__main__':
    main()

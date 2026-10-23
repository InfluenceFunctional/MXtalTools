"""Write the stage-0 golden pins (``data/golden_<case>.pt`` and ``data/goldens_manifest.json``) from
the fixtures, with the same code the tests check them with (``charlib.compute_group``).

    cd MXtalTools
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python tests/characterization/build_goldens.py              # first build
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python tests/characterization/build_goldens.py --check      # recompute, compare, write nothing
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python tests/characterization/build_goldens.py --regenerate [--case F1 ...] [--group mlip ...]

A regeneration replaces what the current code computes as the reference: it is only legitimate in
a change that alters behaviour ON PURPOSE, and that change must say which pins moved and why
(``--check`` lists them before you regenerate; README.md has the per-stage table of the groups
each stage is expected to move). ``--case`` and ``--group`` (group kinds: box, mlip, cluster, ...)
narrow a regeneration to the pins that moved; every other pin keeps its stored record. A partial
regeneration is refused when any record it would keep was written under a different environment
(the bit-for-bit pins do not transfer); regenerate everything then. Refuses to overwrite unless
``--regenerate`` is given; writes nothing but the files above.
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charlib  # noqa: E402

DESCRIPTIONS = {
    'box': 'the constructed batch: per-row sums (radius, mass, mol_volume, num_atoms), symmetry, cell, '
           'box_analysis (T_fc, T_cf, cell_volume, packing_coeff, density), pose slots and atoms; where the '
           'constructor ran no box_analysis (from_arrays), box_analysis on a clone too',
    'canonicalize': 'canonicalize_zp_aunits() on a clone: slot order (centroid, orientation, handedness) and the '
                    'atom blocks moved with it (pos, z, x, mol_ind, unit_* per-slot values, unit_smiles)',
    'latent': 'latent_params(gauge_fix_free_axes=g) on a clone (latents and the canonicalised state it leaves), '
              'then latent_to_cell_params(latents) on a copy (cell, pose, box)',
    'ucell': 'mol2ucell(std_orientation=s) on a clone: posed pos, unit_cell_pos / batch / mol_ind, z',
    'cluster': 'mol2cluster(cutoff, std_orientation=s) (pos, z, x, aux_ind, mol_ind, ptr, batch, unit cell) and '
               'construct_radial_graph(cutoff): intra and inter edge sets sorted (target-major), per-graph counts, '
               'inter distances in raw edge order',
    'analyze': "analyze(['reduction_en', 'elj'], cutoff, std_orientation=s) on a clone",
    'rdf': "analyze(['rdf'], cutoff, std_orientation=s, rdf_mode=m) on a clone: rdf, bin positions, channel labels",
    'mlip': 'MLIP input assembly without a model: batch_to_fairchem_batch (after mol2ucell, and force_rebuild), '
            'the list path collated, batch_to_mace_input_dict and the collated batch_to_mace_atomicdata (stub with '
            f'r_max {charlib.STUB_R_MAX}, elements 1-100); analyze([uma]) / analyze([mace]) and, on the cluster, '
            "analyze(['elj', uma|mace], cutoff 6) with stub predictors that return charlib.stub_energy of their input "
            '(the result, the stored *_pot / *_gas_pot, and every input handed over, gas leg then crystal leg; the '
            f'UMA stub asks for the external graph at cutoff {charlib.STUB_UMA_CUTOFF}); ase unit-cell assembly',
    'reduction': "analyze(['reduction_en']) on every row under cell variants and space groups off the reduced setting",
    'helpers': 'subsample_new_batch([1, 0]) and ([0]), append_batch(clone), batch_to_list and its re-collation: every '
               'key and the _slice_dict / _inc_dict entries',
    'zp_doubling': 'zp_doubling: double_zp1_crystal of each sg 14 F1 row (params, doubled crystal); _unit_cell_frac '
                   'of each F3b row',
    'stdcell': 'compute_standard_cell() on a clone (spglib): cell, pose, pos, unit cell, operators, or the exception',
    'reparam': 'mol2ucell(std_orientation=True) then reparameterize_unit_cell() on a clone: its five outputs',
    'ingest': 'the F3 rows as bare deposited sets (wrapped, units interleaved, unit 1 reversed) through '
              'find_molecules, parameterize_pose and check_rebuild (csd_v1 recipe arguments, bonds by distance): '
              'every key after each step',
    'exp6': "analyze(['exp6'], cutoff, std_orientation=s) on a clone",
    'split': "split_to_zp1_batch() on a clone: the per-unit graphs (atoms, slots, row sums)",
    'elements': 'per MLIP input builder and the ase unit-cell assembly: elements and positions per crystal, and the '
                'number of rows whose element differs from the geometric truth (charlib.element_truth)',
}


def compute_case(case, kinds=None):
    out = OrderedDict()
    for group in charlib.case_groups(case):
        if kinds is not None and charlib.group_kind(group) not in kinds:
            continue
        t = time.time()
        res = charlib.compute_group(case, group)
        for k, v in res.items():
            out[k] = charlib.encode(v)
        print(f'  {case} {group}: {len(res)} pins, {time.time() - t:.2f} s', flush=True)
    return out


def _load(case):
    path = charlib.DATA / f'golden_{case}.pt'
    return torch.load(path, weights_only=True) if path.exists() else None


def _env_problems(saved) -> list:
    """Why the records of a saved golden file may not be kept beside newly computed ones (empty if they may)."""
    if saved is None:
        return []
    out = [f'{k}: {a!r} -> {b!r}' for k, a, b in charlib.environment_mismatch(saved.get('__environment__', {}))]
    old_groups = saved.get('__group_environments__', {})
    now_groups = charlib.all_group_environments()
    out += [f'{kind}: {old_groups.get(kind)!r} -> {now_groups[kind]!r}' for kind in now_groups
            if kind in old_groups and old_groups.get(kind) != now_groups[kind]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--regenerate', action='store_true', help='overwrite existing golden files')
    g.add_argument('--check', action='store_true', help='recompute and compare with the saved goldens')
    ap.add_argument('--case', nargs='*', default=None, help='cases to (re)write; default all')
    ap.add_argument('--group', nargs='*', default=None,
                    help='group kinds to (re)write (box, canonicalize, latent, ucell, cluster, analyze, rdf, mlip, '
                         'reduction, helpers, zp_doubling, stdcell, reparam, ingest, exp6, split, elements); '
                         'default all')
    args = ap.parse_args()
    cases = args.case or list(charlib.CASES)
    for c in cases:
        if c not in charlib.CASES:
            sys.exit(f'unknown case {c}; known: {list(charlib.CASES)}')
    kinds = set(args.group) if args.group else None
    if kinds:
        unknown = kinds - set(DESCRIPTIONS)
        if unknown:
            sys.exit(f'unknown group kinds {sorted(unknown)}; known: {sorted(DESCRIPTIONS)}')

    with charlib.deterministic():
        computed = {case: compute_case(case, kinds) for case in cases}

    if args.check:
        bad = 0
        for case, pins in computed.items():
            saved_file = _load(case) or {}
            saved = {k: v for k, v in saved_file.items() if not k.startswith('__')
                     and (kinds is None or k.split('/')[0] in kinds)}
            for k in sorted(set(saved) | set(pins)):
                if k not in saved or k not in pins:
                    print(f'  {case} {k}: only in {"saved" if k in saved else "recomputed"}')
                    bad += 1
                    continue
                if not _same_record(saved[k], pins[k]):
                    print(f'  {case} {k}: golden {charlib.describe(saved[k])} | now {charlib.describe(pins[k])}')
                    bad += 1
        print('goldens reproduce bit for bit' if not bad else f'{bad} pins differ')
        sys.exit(1 if bad else 0)

    manifest_path = charlib.assert_inside_data(charlib.GOLDEN_MANIFEST)
    if manifest_path.exists() and not args.regenerate:
        sys.exit(f'{manifest_path} exists; pass --regenerate to overwrite (see the module docstring), or --check')
    old = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            old = json.load(f)

    # a record kept beside new ones must have been written under this environment
    full = args.case is None and kinds is None
    if not full:
        kept = [c for c in charlib.CASES if c not in cases] + (cases if kinds is not None else [])
        problems = {c: _env_problems(_load(c)) for c in kept}
        problems = {c: p for c, p in problems.items() if p}
        if problems:
            sys.exit(f'refusing a partial regeneration: kept goldens were written under another environment '
                     f'{problems}; regenerate every case and group (no --case / --group)')

    entries = dict(old.get('entries', {}))
    files = dict(old.get('files', {}))
    for case, pins in computed.items():
        path = charlib.assert_inside_data(charlib.DATA / f'golden_{case}.pt')
        records = OrderedDict()
        saved = _load(case)
        if kinds is not None and saved is not None:
            # keep every saved record of the kinds not being regenerated, in the case's group order
            records.update((k, v) for k, v in saved.items()
                           if not k.startswith('__') and k.split('/')[0] not in kinds)
        records.update(pins)
        order = {grp: i for i, grp in enumerate(charlib.case_groups(case))}
        records = OrderedDict(sorted(records.items(), key=lambda kv: order.get(_group_of(kv[0], case), len(order))))
        with open(path, 'wb' if args.regenerate else 'xb') as f:
            # the environment rides in each golden file (keys starting '__' are not pins): the pin tests compare
            # every file's own record, so files written at different times are each judged by their own
            torch.save({'__environment__': charlib.environment(),
                        '__group_environments__': charlib.all_group_environments(), **records}, f)
        files[case] = {'file': path.name, 'bytes': os.path.getsize(path), 'pins': len(records),
                       'case': charlib.CASES[case][2], 'environment': charlib.environment(),
                       'group_environments': charlib.all_group_environments()}
        for k in [k for k in entries if k.startswith(f'{case}/')
                  and (kinds is None or k.split('/')[1] in kinds)]:
            del entries[k]
        for k, rec in pins.items():
            kind = k.split('/')[0]
            entries[f'{case}/{k}'] = {'stored': rec['kind'], 'value': charlib.describe(rec)}
        print(f'wrote {path} ({os.path.getsize(path)} bytes, {len(records)} pins, {len(pins)} recomputed)')
    for rec in entries.values():
        rec.pop('what', None)           # what a group pins is said once, under 'group_descriptions'
    entries = OrderedDict(sorted(entries.items(), key=lambda kv: (list(charlib.CASES).index(kv[0].split('/')[0])
                                                                   if kv[0].split('/')[0] in charlib.CASES else 99,)))

    manifest = {
        'what': 'stage-0 golden pins of MXtalTools behaviour; README.md says how to read and regenerate them',
        'environment': charlib.environment(),
        'group_environments': charlib.all_group_environments(),
        'source_digest_at_generation': charlib.source_digest(),
        'settings': {'threads': 1, 'deterministic_algorithms': True, 'default_dtype': 'float32',
                     'cutoffs': list(charlib.CUTOFFS), 'rdf_modes': [str(m) for m in charlib.RDF_MODES],
                     'hash_above_bytes': charlib.HASH_ABOVE_BYTES},
        'cases': {c: {'std_orientation': list(charlib.CASES[c][1]), 'description': charlib.CASES[c][2],
                      'groups': charlib.case_groups(c)} for c in charlib.CASES},
        'group_descriptions': DESCRIPTIONS,
        'files': files,
        'entries': entries,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'wrote {manifest_path}')


def _group_of(key, case):
    """The pin group a stored key belongs to (the longest group of the case that prefixes it)."""
    best = ''
    for grp in charlib.case_groups(case):
        if key.startswith(grp + '/') and len(grp) > len(best):
            best = grp
    return best


def _same_record(a, b):
    """Two stored records pin the same value (the rule compare() applies to a live value)."""
    if a['kind'] != b['kind']:
        return False
    if a['kind'] == 'digest':
        return a['sha256'] == b['sha256']
    if a['kind'] == 'raises':
        return a['type'] == b['type']
    if a['kind'] == 'tensor':
        return charlib.compare(a, b['value'])[0]
    return a['value'] == b['value']


if __name__ == '__main__':
    main()

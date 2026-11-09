"""Reduce one finished stream run (cfg:stream.trajectories on, basin_set off) to <run>_slim.pt, then delete the bulky
files it was read from.

    python slim_outputs.py RUN_YAML [--keep]

RUN_YAML is the run's rendered config (out_dir/<run>.yaml; mxt_commit is read from it if present). Read from out_dir:
<run>_traj.pt (the merged trajectories: one attempt per start, ``stream_search._merge_trajectories``), <run>_minima.pt
(``minima.minimum_rows``), <run>_conditions.json, <run>_stream_effort.json, and the molecule file mol_path.

<run>_slim.pt is a dict, format 'stream_slim/1':
  mxt_commit, config (the run config as a dict), kT, energy / search / identity (``stream_search._basin_records``, the
  records a basin or band set's header carries), trajectory_settings, effort (the stream effort record);
  keys          [(identifier, space group, Z')] per condition, the trajectories' order;
  conditions    <run>_conditions.json rows (starts, minima, minima in the band, lowest E/kT, quota met), keys' order;
  molecules     the MolData of mol_path whose identifiers are conditions, as written (the frame the search used);
  traj          per state: start_index int32, condition int32 (into keys), handedness int8, stage int8 (a finish-step
                saddle descent is stage = number of stages, in the identity description; other stages are in their
                own raw description), step int16, params float32 [., 12], f_kT float32, grad_kT float32 [., 12],
                attempt int64 [., 2], minimum_id int32 (-1 if the start ended at none);
  minima        per minimum: condition int32, minimum_id int64, id_params float64 [., 12] (identity description),
                id_handedness int8, minimum_f float64 (E/kT under the minima block's envelope), polish_status int8,
                hits int32, seed_hits int32.
The file is written atomically and read back; its counts must equal the sources'. Only then are <run>.pt,
<run>_minima.pt and <run>_traj.pt deleted (not with --keep). A run whose <run>_slim.pt exists is done.
"""
import argparse
import json
import os
import sys

import torch
import yaml

FORMAT = 'stream_slim/1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_yaml')
    ap.add_argument('--keep', action='store_true', help='do not delete the source files')
    a = ap.parse_args()
    from mxtaltools.common.config_processing import dict2namespace
    from mxtaltools.crystal_search import minima as mn
    from mxtaltools.crystal_search.run_search import check_minima_config
    from mxtaltools.crystal_search.stream_search import _basin_records, check_stream_config
    from mxtaltools.crystal_search.run_state import atomic_torch_save

    raw = yaml.safe_load(open(a.run_yaml))
    config = dict2namespace(raw)
    stem = os.path.join(raw['out_dir'], raw['run_name'])
    out = stem + '_slim.pt'
    if os.path.exists(out):
        print(f'{out} exists: nothing to do')
        return
    scfg, mcfg = check_stream_config(config), check_minima_config(config)
    stages = [dict(st if isinstance(st, dict) else vars(st)) for st in config.opt]
    assert scfg['trajectories'] is not None, 'cfg:stream.trajectories is off: nothing to slim'

    tr = torch.load(stem + '_traj.pt', weights_only=False)
    assert tr['format'] == 'stream_trajectories/1', tr['format']
    rows = torch.load(stem + '_minima.pt', weights_only=False)
    conditions = json.load(open(stem + '_conditions.json'))
    effort = json.load(open(stem + '_stream_effort.json'))
    keys = [tuple(k) for k in tr['keys']]
    cidx = {k: i for i, k in enumerate(keys)}
    assert [(c['identifier'], c['sg'], c['z_prime']) for c in conditions] == keys, 'conditions and keys differ'

    h = tr['handedness']
    assert torch.is_tensor(h) and h.numel() == len(tr['f_kT']), "handedness is ragged (Z' > 1?): not handled"
    traj = dict(start_index=tr['start_index'].to(torch.int32), condition=tr['condition'].to(torch.int32),
                handedness=h.reshape(-1).round().to(torch.int8), stage=tr['stage'].to(torch.int8),
                step=tr['step'].to(torch.int16), params=tr['params'].to(torch.float32),
                f_kT=tr['f_kT'].to(torch.float32), grad_kT=tr['grad_kT'].to(torch.float32),
                attempt=tr['attempt'].to(torch.int64), minimum_id=tr['minimum_id'].to(torch.int32))
    if len(tr['step']):
        assert int(tr['step'].max()) < 2 ** 15 and int(tr['stage'].max()) < 2 ** 7, 'step or stage overflows'

    def col(name, dtype):
        return torch.stack([torch.as_tensor(getattr(r, name)).reshape(-1)[0] for r in rows]).to(dtype) \
            if rows else torch.zeros(0, dtype=dtype)
    minima = dict(condition=torch.tensor([cidx[tuple(mn.row_key(r)[:3])] for r in rows], dtype=torch.int32),
                  minimum_id=col('minimum_id', torch.int64),
                  id_params=torch.stack([torch.as_tensor(r.id_params, dtype=torch.float64).reshape(-1) for r in rows])
                  if rows else torch.zeros(0, 12, dtype=torch.float64),
                  id_handedness=col('id_handedness', torch.float64).round().to(torch.int8),
                  minimum_f=col('minimum_f', torch.float64), polish_status=col('polish_status', torch.int8),
                  hits=col('hits', torch.int32), seed_hits=col('seed_hits', torch.int32))

    ids = {k[0] for k in keys}
    mols = [m for m in torch.load(raw['mol_path'], weights_only=False) if str(m.identifier) in ids]
    assert len(mols) == len(ids), f'{len(ids) - len(mols)} conditions have no molecule in {raw["mol_path"]}'

    counts = dict(conditions=len(keys), rows=effort.get('rows'), minima=len(rows))
    energy, search, identity = _basin_records(config, stages, mcfg, scfg, counts)
    slim = dict(format=FORMAT, mxt_commit=raw.get('mxt_commit'), config=raw, kT=float(tr['kT']), energy=energy,
                search=search, identity=identity, trajectory_settings=dict(tr['settings']), effort=effort,
                keys=keys, conditions=conditions, molecules=mols, traj=traj, minima=minima)
    atomic_torch_save(slim, out)

    back = torch.load(out, weights_only=False)
    assert len(back['traj']['f_kT']) == len(tr['f_kT']) and len(back['minima']['minimum_f']) == len(rows)
    assert len(back['molecules']) == len(ids) and back['keys'] == keys
    src = [stem + '.pt', stem + '_minima.pt', stem + '_traj.pt']
    before = sum(os.path.getsize(p) for p in src if os.path.exists(p))
    print(f"{out}: {len(tr['f_kT']):,} states, {len(rows):,} minima, {len(keys)} conditions; "
          f"{os.path.getsize(out) / 1e6:.1f} MB from {before / 1e6:.1f} MB")
    if not a.keep:
        for p in src:
            if os.path.exists(p):
                os.remove(p)


if __name__ == '__main__':
    sys.exit(main())

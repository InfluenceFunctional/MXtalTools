"""
acr_fm_oct27, step 2: the campaign of each calibrated model -- a coordinated search for acridine P2_1/c (sg 14), Z'=1,
under that model with its own gas-phase conformer, run until every Good-Turing stream stops (coordinator.py) or the hard
cap. The models, molecules and schedule are in campaign.py.

    python make_campaign.py [MODEL ...]   # writes <MODEL>/coord.yaml, streams/*.yaml, model.env, MANIFEST.md

A model is generated only from its calibration (calibration/<MODEL>.json, the calibrate.py result.json of its cluster
job): the identity cut is that file's 1 kT thermal radius, and its energy model id (the cluster checkpoint's) must equal
the id of the dev-box copy (campaign.MODELS[MODEL]['local']).

Streams (GPU array jobs writing shards to <ROOT>/<MODEL>):
  seeded   seed_pool.pt (51,690 distinct states of the earlier acridine landscapes, build_seed_pool.py) and the known forms
           ACRDIN04 and ACRDIN12, rebuilt with the model's conformer (preflight.py writes seeds/seeds_<task>.pt); every
           seed relaxed through stage 2, the polish stage on the rows within POLISH_KT of the job's best. A fixed list:
           no stopping rule, it ends when its list is done.
  random   random reduced starts; wrap boundary, no reduction wall; stage-2 energy cascade (drop walkers > 10 / 8 / 5 kT
           above the job's best at steps 25 / 50 / 100); then the polish stage
  hops     latent kicks of registry basins within 2 kT, <= 32 per basin, <= 3 generations deep; kick length log-uniform on
           [10^-1.7, 10^-0.3] = [0.02, 0.5], recorded; each hop job takes 1 / (hop tasks) of a parent's remaining kicks
           per listing (coord_hop_share; the kick-limit fix after acr_zp1_sep30 ran 54 kicks per parent, median)
Identity: atomwise RDF leader clustering at the model's 1 kT thermal radius (owner 2026-09-28: the identity cut is the
thermal radius). Atomwise keeps acridine's C2v relabelling twins apart (owner 2026-10-27: accepted).
Energy: lattice energy, kJ/mol, of the model; kT 2.494; reference the lowest registry basin (moving); bands 2 kT and
1 kT; window 3 kT.
Stopping: as acr_zp1_sep30 -- per stream with a rule, effort / 90%-upper-bound(basins seen once from it) > Z (2 kT: 1e5
row-evaluations; 1 kT: 1e6) after >= 2000 relaxations and >= 20 states in the band, on two consecutive passes after new
work, or once the test holds and the stream has brought no new work for max(1 h, 3 x its longest shard); campaign STOP
when both stop or at the hard cap (1e8 row-evaluations).
Jobs: 6 h walltime, SIGUSR1 15 min before; a relaunch resumes every task (launch.sh MODEL).
"""
import json
import sys
from argparse import Namespace

import yaml

import campaign as C

SEEDS = {'random': 1_000_000_000, 'seeded': 1_300_000_000, 'hops': 1_600_000_000}
TASK_SEED_STEP = 15_000_000  # per array task; a task's own stream uses opt_seed + batch_idx * 1e4 (< 1500 batches)
NUM_SAMPLES = 20_000  # random starts per task (a task stops earlier on STOP, the walltime or its stream's stop)
LOG_NOISE = [-1.7, -0.3]
Z = {'2kT': 1e5, '1kT': 1e6}


def calibration(model):
    path = C.HERE / 'calibration' / f'{model}.json'
    if not path.exists():
        sys.exit(f'{model}: no {path} (copy the calibration job\'s result.json there)')
    cal = json.loads(path.read_text())
    assert cal['model'] == model, f'{path} is the calibration of {cal["model"]}'
    return cal


def local_model_id(model):
    from mxtaltools.crystal_search.coordinator import energy_model_id
    m = C.MODELS[model]
    key = 'mace_predictor_path' if m['kind'] == 'mace' else 'uma_predictor_path'
    return energy_model_id({'optim_target': m['kind']}, Namespace(**{key: m['local']}))


def stream_cfg(model, stream):
    m, camp = C.MODELS[model], f'{C.ROOT}/{model}'
    cfg = yaml.safe_load(C.BASE.read_text())
    cfg.update(device='cuda', mol_path=C.mol_path(model), out_dir=f'{camp}/runs', save_trajs=False,
               grow_batch_size=True, oom_ceiling=True, batch_size=C.BATCH[model], num_samples=NUM_SAMPLES,
               force_restart_run=False, sgs_to_search=[14], zp_to_search=[1], coord_dir=camp, coord_stream=stream,
               coord_curate_every_s=3600, coord_hop_wait_s=7200, init_sample_method='random', dataset_path=None,
               opt_seed=SEEDS[stream], run_name=f'{stream}_TASK', target_path=None, **C.predictor_paths(model))
    cfg['opt'] = C.stages(m['kind'], cascade=stream != 'seeded')
    if stream == 'hops':
        cfg.update(init_sample_method='hops', coord_hop_share=1.0 / C.TASKS[model]['hops'])
    if stream == 'seeded':  # the job script replaces TASK with the array index; preflight.py writes the files
        cfg.update(init_sample_method='data', dataset_path=f'{camp}/seeds/seeds_TASK.pt')
    return cfg


def coord_yaml(model, cal, mid):
    th = cal['thermal']['table']['1kT']
    rng = lambda v: f'{min(v):.4f}, {max(v):.4f}'  # noqa: E731
    n_draw = cal['thermal'].get('n_draw', 32)  # 32 in calibrate.py as first pushed (16950107), which did not record it
    note = (f"atomwise thermal radius under {model} (calibrate.py, {cal['env']['gpu']}, MXtalTools "
            f"{cal['env']['mxt_commit']}): median over {len(th['rdf_distance'])} distinct relaxed low basins "
            f"({', '.join(cal['thermal']['anchors'])}) of the atomwise RDF distance at which {n_draw} hop-style kicks "
            f"per size raise the crystal energy by a median 1 kT: {cal['identity_cut']:.4f} [{rng(th['rdf_distance'])}]; latent "
            f"kick {cal['latent_kick_1kT']:.4f} [{rng(th['latent_kick'])}]")
    return dict(mol_path=C.mol_path(model), sg=14, z_prime=1, energy_key=C.MODELS[model]['kind'], energy_model_id=mid,
                kT=C.KT, bands_kT={'2kT': 2.0, '1kT': 1.0}, identity_cut=round(float(cal['identity_cut']), 4),
                rdf_mode='atomwise', identity_calibrated=True, identity_note=note, energy_ref=None, window_kT=3.0,
                hard_cap=1e8, passes_to_confirm=2, confirm_settle_s=3600.0, rdf_batch=32,
                streams={s: dict(Z=dict(Z), min_relaxations=2000, min_hits=20) for s in ('random', 'hops')},
                hops=dict(stream='hops', window_kT=2.0, max_per_basin=32, max_generation=3, log_noise=LOG_NOISE))


def check(model, streams, coord):
    m, camp, tasks = C.MODELS[model], f'{C.ROOT}/{model}', C.TASKS[model]
    assert 0.0 < coord['identity_cut'] < 0.3, f"identity cut {coord['identity_cut']}"
    for s, c in streams.items():
        flat = yaml.dump(c)
        assert 'D:' not in flat and '\\\\' not in flat, f'{s}: local path'
        assert c['coord_stream'] == s and c['coord_dir'] == camp and c['mol_path'] == coord['mol_path']
        assert all(st['optim_target'] == m['kind'] and st['centroid_boundary'] == 'wrap' and not st['enforce_reduced']
                   for st in c['opt'])
        assert c['opt'][-1]['optim_target'] == coord['energy_key']
        assert c['mace_predictor_path' if m['kind'] == 'mace' else 'uma_predictor_path'] == m['path']
        assert c['uma_activation_checkpointing'] is False
        assert c['sgs_to_search'] == [coord['sg']] and c['zp_to_search'] == [1]
        assert (s == 'seeded') == (c['init_sample_method'] == 'data')
        assert (s == 'seeded') == ('early_stop' not in c['opt'][-2]), 'stage-2 cascade on random and hops only'
        assert c['opt'][-1]['early_stop'] == [[0, C.POLISH_KT]] and c['opt'][-1]['max_num_steps'] == C.POLISH_STEPS
        assert NUM_SAMPLES / c['batch_size'] < 1500, 'batches would overrun the per-task seed range'
        assert SEEDS[s] + 15 * TASK_SEED_STEP < 2 ** 31, 'seeds must stay below 2^31'
        assert tasks[s] <= 15, 'task indices beyond 14 overlap the next stream seed range'
    seeds = sorted(SEEDS.values())
    assert min(b - a for a, b in zip(seeds, seeds[1:])) > 15 * TASK_SEED_STEP, 'stream seed ranges overlap'
    assert set(coord['streams']) == {'random', 'hops'}, 'the seeded stream is a finite list: no stopping rule'
    assert 'D:' not in yaml.dump(coord)


def write(model):
    cal = calibration(model)
    mid = local_model_id(model)
    assert cal['energy_model_id'] == mid, f"{model}: the cluster checkpoint is {cal['energy_model_id']}, the dev " \
                                          f"box's {mid}"
    out = C.HERE / model
    (out / 'streams').mkdir(parents=True, exist_ok=True)
    streams = {s: stream_cfg(model, s) for s in SEEDS}
    coord = coord_yaml(model, cal, mid)
    for s, c in streams.items():
        (out / 'streams' / f'{s}.yaml').write_text(yaml.dump(c, default_flow_style=False, sort_keys=False))
    (out / 'coord.yaml').write_text(yaml.dump(coord, default_flow_style=False, sort_keys=False))
    check(model, {s: yaml.safe_load((out / 'streams' / f'{s}.yaml').read_text()) for s in streams},
          yaml.safe_load((out / 'coord.yaml').read_text()))
    m, t = C.MODELS[model], C.TASKS[model]
    env = dict(MODEL=model, KIND=m['kind'], MODEL_PATH=m['path'], MODEL_ID=mid, PYEXTRA=m['pyextra'],
               CDIR=f'{C.ROOT}/{model}', N_RANDOM=t['random'], N_HOPS=t['hops'], N_SEEDED=t['seeded'])
    (out / 'model.env').write_text('# generated by make_campaign.py; launch.sh sources it\n'
                                   + ''.join(f'{k}={v}\n' for k, v in env.items()))
    rows = [f'# {C.NAME}/{model} manifest (generated by make_campaign.py)', '',
            '| stream | start method | stages (target, max steps) | batch (starts, first) | array tasks '
            '| opt_seed of task k |', '|---|---|---|---|---|---|']
    for s, c in streams.items():
        stages = ', '.join(f"{st['optim_target']} {st['max_num_steps']}" + (' cascade' if st.get('early_stop') else '')
                           for st in c['opt'])
        rows.append(f"| {s} | {c['init_sample_method']} | {stages} | {c['batch_size']} | {t[s]} | "
                    f"{SEEDS[s]} + k * {TASK_SEED_STEP} |")
    rows += ['', f"Energy model `{mid}`. Molecule `{C.mol_path(model)}`. Identity cut {coord['identity_cut']} "
                 f"(atomwise). Hop kick lengths 10^{LOG_NOISE}. Campaign directory `{C.ROOT}/{model}`.",
             '', f"Calibration throughput: {cal['throughput']['relax_per_gpu_hour']:.0f} relaxations per GPU-hour "
                 f"({cal['throughput']['starts']} random starts at batch {cal['throughput']['batch']}, "
                 f"{cal['env']['gpu']})."]
    (out / 'MANIFEST.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    for mod in sys.argv[1:] or list(C.MODELS):
        write(mod)

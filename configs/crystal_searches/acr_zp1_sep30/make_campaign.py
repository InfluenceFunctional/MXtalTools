"""
acr_zp1_sep30: a fresh coordinated search for acridine P2_1/c (sg 14), Z'=1, under the new MACE checkpoint
(acr_newmodel.model), with the universal acridine conformer (acr_newmodel_conformer.pt: the gas-phase minimum under that
model; owner 2026-09-30). Run until every Good-Turing stream stops (coordinator.py) or the hard cap.

Streams (GPU array jobs writing shards to <DATA>/campaigns/acr_zp1_sep30):
  seeded   earlier search states re-relaxed under the new model (seed_pool.pt): every sg14 Z'=1 end state of the May
           production search (acr_production, old MACE; within 10 kT of its minimum) and the lowest-LJ 20% of the
           December production search (LJ minima), physical cells only, thinned lowest-energy-first in latent space at
           the 1 kT kick radius; plus the known P2_1/c Z'=1 forms ACRDIN04 and ACRDIN12 (known_forms.pt). Each is
           rebuilt from its cell parameters with the new conformer (preflight.py writes seeds/seeds_<task>.pt). A fixed
           list: no stopping rule, it ends when its list is done.
  random   random reduced starts; wrap boundary, no reduction wall (outputs re-standardised by the coordinator);
           stage-2 energy cascade (drop walkers > 10 / 8 / 5 kT above the job's best at steps 25 / 50 / 100)
  hops     latent kicks of registry basins within 2 kT, <= 32 per basin, <= 3 generations deep; kick length drawn
           log-uniformly on [10^-1.7, 10^-0.3] = [0.02, 0.5] per start and recorded (coordinator reg['hit_kick'])
Identity: ATOMWISE RDF leader clustering at the 1 kT thermal radius under this model (IDENTITY_CUT: the median atomwise
RDF distance at which hop-style kicks of relaxed low basins raise the energy by a median 1 kT; owner 2026-09-28: the
identity cut is the thermal radius).
Energy: lattice energy, kJ/mol (MACE); kT 2.494; the reference is the lowest registry basin (moving); bands 2 kT and
1 kT; window 3 kT.
Stopping: per stream with a rule (random, hops), effort / 90%-upper-bound(basins seen once from it) > Z (2 kT: 1e5
row-evaluations; 1 kT: 1e6) after >= 2000 relaxations and >= 20 states in the band, on two consecutive passes after new
work, or once the test holds and the stream has brought no new work for max(1 h, 3 x its longest shard); campaign STOP
when both stop or at the hard cap (1e8 row-evaluations).
Jobs: short (6 h walltime, SIGUSR1 15 min before), many tasks; a relaunch resumes every task (launch.sh).

    python make_campaign.py      # writes coord.yaml, streams/*.yaml, MANIFEST.md; asserts the checks
"""
from argparse import Namespace
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
NAME = HERE.name
BASE = HERE.parent / 'acr_finish_sep27' / '0.yaml'  # the no-wall wrap schedule of the acridine campaigns
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
CAMP = f'{DATA}/campaigns/{NAME}'
MOL = f'{DATA}/acr_newmodel_conformer.pt'
MODEL = '/scratch/mk8347/data/acr_newmodel.model'
LOCAL_MODEL = 'D:/crystal_datasets/acr_newmodel.model'  # the same file as the cluster's (launch.sh checks the id)
KT = 2.494
IDENTITY_CUT = 0.134
IDENTITY_NOTE = ("atomwise thermal radius under acr_newmodel, 2026-09-30: median over 6 distinct relaxed low basins "
                 "(16 lowest Z'=1 prior anchors rebuilt with the universal conformer and relaxed on this schedule) of "
                 "the atomwise RDF distance at which 32 hop-style kicks per size raise the crystal energy by a median "
                 "1 kT: 0.134 [0.121, 0.144]; latent kick 0.031 [0.028, 0.035]")
SEEDS = {'random': 1_000_000_000, 'seeded': 1_300_000_000, 'hops': 1_600_000_000}
TASK_SEED_STEP = 15_000_000  # per array task; a task's own stream uses opt_seed + batch_idx * 1e4 (< 1500 batches)
TASKS = {'random': 12, 'hops': 12, 'seeded': 8}  # array tasks per stream (launch.sh defaults: keep in step)
NUM_SAMPLES = 20_000  # random starts per task (a task stops earlier on STOP, the walltime or its stream's stop)
BATCH = 96
CASCADE = dict(early_stop=[[25, 10.0], [50, 8.0], [100, 5.0]], early_stop_kT=KT)
LOG_NOISE = [-1.7, -0.3]


def base_cfg(stream):
    cfg = yaml.safe_load(BASE.read_text())
    cfg.update(mol_path=MOL, mace_predictor_path=MODEL, uma_predictor_path=None, out_dir=f'{CAMP}/runs',
               save_trajs=False, grow_batch_size=True, oom_ceiling=True, batch_size=BATCH, num_samples=NUM_SAMPLES,
               force_restart_run=False, sgs_to_search=[14], zp_to_search=[1], coord_dir=CAMP, coord_stream=stream,
               coord_curate_every_s=3600, coord_hop_wait_s=7200, init_sample_method='random', dataset_path=None,
               opt_seed=SEEDS[stream], run_name=f'{stream}_TASK')
    for st in cfg['opt']:
        assert st['centroid_boundary'] == 'wrap' and st['enforce_reduced'] is False
        st['optim_target'] = 'mace'
        st['show_tqdm'] = False
        # the base thresholds were tuned when ema_trajectory returned 0.1 x the smoothed trajectory (before
        # MXtalTools fe283818); x10 keeps relaxations stopping where the earlier acridine campaigns' did
        st['convergence_eps'] = 10 * float(st['convergence_eps'])
    cfg['opt'][-1].update(CASCADE)
    return cfg


def stream_cfg(stream):
    cfg = base_cfg(stream)
    if stream == 'hops':
        cfg['init_sample_method'] = 'hops'
    if stream == 'seeded':  # the job script replaces TASK with the array index; preflight.py writes the files
        cfg.update(init_sample_method='data', dataset_path=f'{CAMP}/seeds/seeds_TASK.pt')
        cfg['opt'][-1].pop('early_stop')  # every seed is relaxed to the end: a seed is not a draw to be culled
        cfg['opt'][-1].pop('early_stop_kT')
    return cfg


def coord_yaml():
    from mxtaltools.crystal_search.coordinator import energy_model_id
    mid = energy_model_id({'optim_target': 'mace'}, Namespace(mace_predictor_path=LOCAL_MODEL))
    return dict(mol_path=MOL, sg=14, z_prime=1, energy_key='mace', energy_model_id=mid, kT=KT,
                bands_kT={'2kT': 2.0, '1kT': 1.0}, identity_cut=IDENTITY_CUT, rdf_mode='atomwise',
                identity_calibrated=True, identity_note=IDENTITY_NOTE, energy_ref=None, window_kT=3.0, hard_cap=1e8,
                passes_to_confirm=2, confirm_settle_s=3600.0, rdf_batch=32,
                streams={s: dict(Z={'2kT': 1e5, '1kT': 1e6}, min_relaxations=2000, min_hits=20)
                         for s in ('random', 'hops')},
                hops=dict(stream='hops', window_kT=2.0, max_per_basin=32, max_generation=3, log_noise=LOG_NOISE))


def check(streams, coord):
    assert coord['identity_cut'] is not None and 0.0 < coord['identity_cut'] < 0.3, 'set IDENTITY_CUT'
    for s, c in streams.items():
        flat = yaml.dump(c)
        assert 'D:' not in flat and '\\\\' not in flat, f'{s}: local path'
        assert c['coord_stream'] == s and c['coord_dir'] == CAMP and c['mol_path'] == coord['mol_path']
        assert all(st['optim_target'] == 'mace' and st['centroid_boundary'] == 'wrap' and not st['enforce_reduced']
                   for st in c['opt'])
        assert c['opt'][-1]['optim_target'] == coord['energy_key'] and c['mace_predictor_path'] == MODEL
        assert c['sgs_to_search'] == [coord['sg']] and c['zp_to_search'] == [1]
        assert (s == 'seeded') == (c['init_sample_method'] == 'data') and (s == 'seeded') == ('early_stop' not in c['opt'][-1])
        assert NUM_SAMPLES / BATCH < 1500, 'batches would overrun the per-task seed range'
        assert SEEDS[s] + 15 * TASK_SEED_STEP < 2 ** 31, 'seeds must stay below 2^31'
        assert TASKS[s] <= 15, 'task indices beyond 14 overlap the next stream seed range'
    seeds = sorted(SEEDS.values())
    assert min(b - a for a, b in zip(seeds, seeds[1:])) > 15 * TASK_SEED_STEP, 'stream seed ranges overlap'
    assert set(coord['streams']) == {'random', 'hops'}, 'the seeded stream is a finite list: no stopping rule'
    pre = (HERE / 'preflight.py').read_text()
    assert f"N_TASKS = {TASKS['seeded']}" in pre and f"N_S=${{N_SEEDED:-{TASKS['seeded']}}}" in (HERE / 'launch.sh').read_text(),         'preflight.py N_TASKS and launch.sh N_SEEDED must equal TASKS[seeded]'
    assert 'D:' not in yaml.dump(coord)
    print(f'checks passed on {len(streams)} stream configs')


def main():
    streams = {s: stream_cfg(s) for s in SEEDS}
    coord = coord_yaml()
    (HERE / 'streams').mkdir(exist_ok=True)
    for s, c in streams.items():
        (HERE / 'streams' / f'{s}.yaml').write_text(yaml.dump(c, default_flow_style=False, sort_keys=False))
    (HERE / 'coord.yaml').write_text(yaml.dump(coord, default_flow_style=False, sort_keys=False))
    check({s: yaml.safe_load((HERE / 'streams' / f'{s}.yaml').read_text()) for s in streams},
          yaml.safe_load((HERE / 'coord.yaml').read_text()))
    rows = [f'# {NAME} manifest (generated by make_campaign.py)', '',
            '| stream | start method | stages (target, max steps) | batch (starts) | tasks (launch.sh default) '
            '| opt_seed of task k |', '|---|---|---|---|---|---|']
    for s, c in streams.items():
        stages = ', '.join(f"{st['optim_target']} {st['max_num_steps']}" + (' cascade' if st.get('early_stop') else '')
                           for st in c['opt'])
        rows.append(f"| {s} | {c['init_sample_method']} | {stages} | {c['batch_size']} | {TASKS[s]} | "
                    f"{SEEDS[s]} + k * {TASK_SEED_STEP} |")
    rows += ['', f"Energy model: `{coord['energy_model_id']}`. Molecule: `{MOL}`. Identity cut {coord['identity_cut']} "
                 f"(atomwise). Hop kick lengths 10^{LOG_NOISE}. Campaign directory: `{CAMP}`."]
    (HERE / 'MANIFEST.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()

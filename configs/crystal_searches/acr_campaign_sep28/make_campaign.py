"""
acr_campaign_sep28: a coordinated search for all low-energy acridine sg14 Z'=2 structures, old MACE checkpoint
(acr_112025_mh1_stagetwo.model), until the expected effort per new basin exceeds Z for every stream.

Three proposal streams, each an array of GPU jobs writing shards to one campaign directory (DATA/campaigns/<this>):
  random     random starts; wrap boundary; no reduction wall (outputs are re-standardised by the coordinator);
             stage 2 energy cascade (drop walkers still > 10 / 8 / 5 kT above the job's best at steps 25 / 50 / 100)
  eljstart   the same, preceded in each batch by the eLJ schedule on 4x the starts, keeping the lowest-eLJ quarter
  hops       latent log-noise kicks (-0.5) of registry basins within 2 kT, <= 32 per basin, <= 3 generations deep
All jobs: OOM batch-size ceiling, run lease, SIGUSR1 900 s before the walltime, and any job may run the campaign's curate
pass between batches every 15 min (coordinator.maybe_curate); no separate coordinator job is needed.

Stopping (coordinator): per stream, stop when effort / 90%-upper-bound(basins seen once from it) exceeds Z, for both
bands (2 kT: Z = 1e5 MACE row-evaluations, ~230 random relaxations per new basin; 1 kT: Z = 1e6), after >= 2000
relaxations and >= 20 states in the band, on two consecutive passes; campaign STOP when every stream stops or at the
hard cap 3e7 row-evaluations (~7e4 relaxations). Basins: RDF leader clustering at 0.050 (acridine envwise; the COMPACK
rule of gfn_diffusion eval/campaign_compack.py -- one packing = 20/20 molecules, cut = largest distance with isotonic
P(match) >= 0.95 -- on 240 pairs of 3000 sep27 end states, 2026-09-27), within 3 kT of -62.812, seeded with
priors/known_map.pth (make_priors.py).

    python make_campaign.py      # writes coord.yaml, streams/*.yaml, MANIFEST.md; asserts the checks
"""
from argparse import Namespace
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
NAME = HERE.name
BASE = HERE.parent / 'acr_finish_sep27' / '0.yaml'  # the no-wall wrap schedule of the sep27 no-reduction arms
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
CAMP = f'{DATA}/campaigns/{NAME}'
MXT_ON_CLUSTER = '/scratch/mk8347/projects/gfn_cond/MXtalTools'
LOCAL_MODEL = 'D:/crystal_datasets/acr_112025_mh1_stagetwo.model'  # the same file as the cluster's (launch.sh checks)
KT = 2.494
SEEDS = {'random': 1_000_000_000, 'eljstart': 1_300_000_000, 'hops': 1_600_000_000}
TASK_SEED_STEP = 15_000_000  # per array task; a task's own stream uses opt_seed + batch_idx * 1e4 (< 1500 attempts)
CASCADE = dict(early_stop=[[25, 10.0], [50, 8.0], [100, 5.0]], early_stop_kT=KT)
COORD = dict(sg=14, z_prime=2, energy_key='mace', kT=KT, bands_kT={'2kT': 2.0, '1kT': 1.0}, identity_cut=0.05,
             identity_calibrated=True,
             identity_note='COMPACK rule (20/20 molecules; isotonic P(match) >= 0.95), campaign_compack calibrate on 240 '
                           'pairs of 3000 sep27 end states, 2026-09-27: 0.050 [0.040, 0.060]',
             energy_ref=-62.812,
             window_kT=3.0, hard_cap=3e7, passes_to_confirm=2, rdf_batch=32,
             streams={s: dict(Z={'2kT': 1e5, '1kT': 1e6}, min_relaxations=2000, min_hits=20) for s in SEEDS},
             hops=dict(stream='hops', window_kT=2.0, max_per_basin=32, max_generation=3, log_noise=[-0.5, -0.5]))


def base_cfg():
    cfg = yaml.safe_load(BASE.read_text())
    cfg.update(out_dir=f'{CAMP}/runs', save_trajs=False, grow_batch_size=True, oom_ceiling=True, batch_size=48,
               num_samples=30000, force_restart_run=False, coord_dir=CAMP, coord_curate_every_s=900,
               init_sample_method='random', dataset_path=None)
    for st in cfg['opt']:
        assert st['centroid_boundary'] == 'wrap' and st['enforce_reduced'] is False
        st['show_tqdm'] = False
    cfg['opt'][-1].update(CASCADE)
    return cfg


def stream_cfg(stream):
    cfg = base_cfg()
    cfg.update(coord_stream=stream, opt_seed=SEEDS[stream], run_name=f'{stream}_TASK')
    if stream == 'eljstart':
        s1, s2 = (dict(st) for st in cfg['opt'])
        e1 = dict(s1, optim_target='elj')
        e2 = {k: v for k, v in dict(s2, optim_target='elj', keep_lowest_fraction=0.25).items()
              if k not in CASCADE}
        cfg['opt'] = [e1, e2, s1, s2]
        cfg['batch_size'] = 160  # starts; the MACE stages see a quarter
    if stream == 'hops':
        cfg['init_sample_method'] = 'hops'
    return cfg


def coord_yaml():
    from mxtaltools.crystal_search.coordinator import energy_model_id
    mid = energy_model_id({'optim_target': 'mace'}, Namespace(mace_predictor_path=LOCAL_MODEL))
    return dict(COORD, mol_path=f'{DATA}/opt_acridine_conformer.pt', energy_model_id=mid,
                priors=[f'{MXT_ON_CLUSTER}/configs/crystal_searches/{NAME}/priors/known_map.pth'])


def check(streams, coord):
    for s, c in streams.items():
        flat = yaml.dump(c)
        assert 'D:' not in flat and '\\\\' not in flat, f'{s}: local path'
        assert 'umbrella' not in flat
        assert c['coord_stream'] == s and s in coord['streams']
        assert all(st['centroid_boundary'] == 'wrap' and st['enforce_reduced'] is False for st in c['opt'])
        mace = [st for st in c['opt'] if st['optim_target'] == 'mace']
        assert len(mace) == 2 and mace[-1]['early_stop'] == CASCADE['early_stop']
        assert c['opt'][-1]['optim_target'] == coord['energy_key']
        assert SEEDS[s] + 15 * TASK_SEED_STEP < 2 ** 31, 'seeds must stay below 2^31'
    seeds = sorted(SEEDS.values())
    assert min(b - a for a, b in zip(seeds, seeds[1:])) > 15 * TASK_SEED_STEP, 'stream seed ranges overlap'
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
            '| stream | stages (target, max steps) | batch (starts) | opt_seed of task k | start method |',
            '|---|---|---|---|---|']
    for s, c in streams.items():
        stages = ', '.join(f"{st['optim_target']} {st['max_num_steps']}"
                           + (' keep 25%' if st.get('keep_lowest_fraction') else '')
                           + (' cascade' if st.get('early_stop') else '') for st in c['opt'])
        rows.append(f"| {s} | {stages} | {c['batch_size']} | {SEEDS[s]} + k * {TASK_SEED_STEP} | "
                    f"{c['init_sample_method']} |")
    rows += ['', f"Energy model: `{coord['energy_model_id']}`. Campaign directory: `{CAMP}`."]
    (HERE / 'MANIFEST.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()

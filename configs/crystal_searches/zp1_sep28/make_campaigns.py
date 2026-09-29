"""
zp1_sep28: four fresh coordinated campaigns, {MIPCAS P-1, NEHZOR P2_1/c} x {eLJ, UMA esen_s}, Z'=1, no priors, each run
until every stream's Good-Turing rule stops it (coordinator.py) or its hard cap.

Streams per campaign (GPU array jobs writing shards to <DATA>/<mol>/campaigns/<campaign>):
  random   random starts; the acr_campaign_sep28 schedule: wrap boundary, no reduction wall (outputs are re-standardised
           by the coordinator), stage-2 energy cascade (drop walkers > 10 / 8 / 5 kT above the job's best at steps
           25 / 50 / 100)
  hops     latent log-noise kicks (-0.5) of registry basins within 2 kT, <= 32 per basin, <= 3 generations deep
Identity: ATOMWISE RDF leader clustering (owner, 2026-09-28: atomwise everywhere, molecular symmetry not quotiented), at a
provisional cut per molecule, the smaller of the UMA 1 kT thermal width and the COMPACK d(P=0.95) of the paper
calibration (energy_sampling eval/paper1_results/calibrate_rdf_metric.py docstring): MIPCAS min(0.071, 0.116) = 0.071,
NEHZOR min(0.122, 0.133) = 0.122; the same cut for eLJ and UMA (a geometric criterion). Final identity: campaign_compack
on each campaign's own end states.
Energy: the reference is the lowest registry basin (moving); kT 2.494 kJ/mol for UMA; for eLJ, raw eLJ units, kT = 2.494 /
thermal_scaling_factor of the GFN prior datasets (MIPCAS 0.36358, NEHZOR 0.15558: 1 raw unit in kJ/mol). Bands 2 kT and
1 kT, window 3 kT.
Stopping: per stream, effort / 90%-upper-bound(basins seen once from it) > Z (2 kT: 1e5 row-evaluations; 1 kT: 1e6),
after >= 2000 relaxations and >= 20 states in the band, on two consecutive passes each after new work; campaign STOP
when every stream stops or at the hard cap (eLJ 3e8, UMA 5e7 row-evaluations).

    python make_campaigns.py      # writes <campaign>/coord.yaml, <campaign>/streams/*.yaml, MANIFEST.md; asserts checks
"""
from argparse import Namespace
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'acr_finish_sep27' / '0.yaml'  # the no-wall wrap schedule of the acridine campaign
DATA = '/scratch/mk8347/data/crystal_datasets'
UMA_CLUSTER = '/scratch/mk8347/models/uma/esen_s.pt'
UMA_LOCAL = 'D:/crystal_datasets/esen_s.pt'  # the same file as the cluster's (launch.sh checks the id)
KT = 2.494
SEEDS = {'random': 1_000_000_000, 'hops': 1_600_000_000}
TASK_SEED_STEP = 15_000_000  # per array task; a task's own stream uses opt_seed + batch_idx * 1e4 (< 1500 batches)
MOLECULES = {
    'mipcas': dict(mol_path=f'{DATA}/mipcas/MIPCAS_standardized.pt', sg=2, cut=0.071, elj_tsf=0.3635836825309169),
    'nehzor': dict(mol_path=f'{DATA}/nehzor/NEHZOR0_std_conf.pt', sg=14, cut=0.122, elj_tsf=0.15557874739170074),
}
ENERGIES = {  # batch_size: starts per batch (the OOM ceiling and grow_batch_size adapt it); num_samples: per task
    'elj': dict(batch_size=2000, num_samples=1_000_000, hard_cap=3e8, tasks={'random': 2, 'hops': 2}),
    'uma': dict(batch_size=500, num_samples=300_000, hard_cap=5e7, tasks={'random': 3, 'hops': 3}),
}
CAMPAIGNS = {f'{m}_{e}': (m, e) for m in MOLECULES for e in ENERGIES}


def kT_of(mol, energy):
    return KT if energy == 'uma' else KT / MOLECULES[mol]['elj_tsf']


def camp_dir(name):
    mol, _ = CAMPAIGNS[name]
    return f'{DATA}/{mol}/campaigns/{name}_sep28'


def stream_cfg(name, stream):
    mol, energy = CAMPAIGNS[name]
    m, e, kt = MOLECULES[mol], ENERGIES[energy], kT_of(mol, energy)
    cfg = yaml.safe_load(BASE.read_text())
    cfg.pop('mace_predictor_path', None)
    cfg.update(mol_path=m['mol_path'], sgs_to_search=[m['sg']], zp_to_search=[1], out_dir=f'{camp_dir(name)}/runs',
               save_trajs=False, grow_batch_size=True, oom_ceiling=True, batch_size=e['batch_size'],
               num_samples=e['num_samples'], force_restart_run=False, coord_dir=camp_dir(name), coord_stream=stream,
               coord_curate_every_s=3600, coord_hop_wait_s=7200, init_sample_method='random', dataset_path=None,
               uma_predictor_path=UMA_CLUSTER, opt_seed=SEEDS[stream], run_name=f'{stream}_TASK')
    for st in cfg['opt']:
        assert st['centroid_boundary'] == 'wrap' and st['enforce_reduced'] is False
        st['optim_target'] = energy
        st['show_tqdm'] = False
        # the base thresholds were tuned when ema_trajectory returned 0.1 x the smoothed trajectory (before
        # MXtalTools fe283818); x10 keeps relaxations stopping where the acridine campaign's did
        st['convergence_eps'] = 10 * float(st['convergence_eps'])
    cfg['opt'][-1].update(early_stop=[[25, 10.0], [50, 8.0], [100, 5.0]], early_stop_kT=kt)
    if stream == 'hops':
        cfg['init_sample_method'] = 'hops'
    return cfg


def coord_yaml(name):
    from mxtaltools.crystal_search.coordinator import energy_model_id
    mol, energy = CAMPAIGNS[name]
    m = MOLECULES[mol]
    mid = energy_model_id({'optim_target': energy}, Namespace(uma_predictor_path=UMA_LOCAL))
    return dict(mol_path=m['mol_path'], sg=m['sg'], z_prime=1, energy_key=energy, energy_model_id=mid,
                kT=kT_of(mol, energy), bands_kT={'2kT': 2.0, '1kT': 1.0}, identity_cut=m['cut'], rdf_mode='atomwise',
                identity_calibrated=False,
                identity_note=f"provisional atomwise cut for {mol.upper()}: min(UMA 1 kT thermal width, COMPACK "
                              f"d(P=0.95)) of the paper calibration; recalibrate with campaign_compack on this campaign",
                energy_ref=None, window_kT=3.0, hard_cap=ENERGIES[energy]['hard_cap'], passes_to_confirm=2,
                rdf_batch=64,
                streams={s: dict(Z={'2kT': 1e5, '1kT': 1e6}, min_relaxations=2000, min_hits=20) for s in SEEDS},
                hops=dict(stream='hops', window_kT=2.0, max_per_basin=32, max_generation=3, log_noise=[-0.5, -0.5]))


def check(name, streams, coord):
    mol, energy = CAMPAIGNS[name]
    for s, c in streams.items():
        flat = yaml.dump(c)
        assert 'D:' not in flat and '\\\\' not in flat, f'{name}/{s}: local path'
        assert c['coord_stream'] == s and s in coord['streams'] and c['coord_dir'] == camp_dir(name)
        assert all(st['optim_target'] == energy and st['centroid_boundary'] == 'wrap' and not st['enforce_reduced']
                   for st in c['opt'])
        assert c['opt'][-1]['optim_target'] == coord['energy_key']
        assert c['sgs_to_search'] == [coord['sg']] and c['zp_to_search'] == [1] and c['mol_path'] == coord['mol_path']
        n_batches = c['num_samples'] / c['batch_size']
        assert n_batches < 1500, f'{name}/{s}: {n_batches} batches would overrun the per-task seed range'
        assert SEEDS[s] + 15 * TASK_SEED_STEP < 2 ** 31, 'seeds must stay below 2^31'
        assert ENERGIES[energy]['tasks'][s] <= 15, 'task indices beyond 14 overlap the next stream seed range'
    seeds = sorted(SEEDS.values())
    assert min(b - a for a, b in zip(seeds, seeds[1:])) > 15 * TASK_SEED_STEP, 'stream seed ranges overlap'
    assert 'D:' not in yaml.dump(coord)


def main():
    rows = ['# zp1_sep28 manifest (generated by make_campaigns.py)', '',
            '| campaign | directory | energy model | kT (energy units) | identity cut (atomwise) | tasks (random + hops) '
            '| batch (starts) | hard cap (row-evaluations) |',
            '|---|---|---|---|---|---|---|---|']
    for name in CAMPAIGNS:
        d = HERE / name
        (d / 'streams').mkdir(parents=True, exist_ok=True)
        streams = {s: stream_cfg(name, s) for s in SEEDS}
        for s, c in streams.items():
            (d / 'streams' / f'{s}.yaml').write_text(yaml.dump(c, default_flow_style=False, sort_keys=False))
        coord = coord_yaml(name)
        (d / 'coord.yaml').write_text(yaml.dump(coord, default_flow_style=False, sort_keys=False))
        check(name, {s: yaml.safe_load((d / 'streams' / f'{s}.yaml').read_text()) for s in SEEDS},
              yaml.safe_load((d / 'coord.yaml').read_text()))
        mol, energy = CAMPAIGNS[name]
        t = ENERGIES[energy]['tasks']
        rows.append(f"| {name} | `{camp_dir(name)}` | `{coord['energy_model_id']}` | {coord['kT']:.4g} | "
                    f"{coord['identity_cut']} | {t['random']} + {t['hops']} | {ENERGIES[energy]['batch_size']} | "
                    f"{coord['hard_cap']:.3g} |")
    rows += ['', 'Tasks per stream are launch.sh defaults (TASKS in launch.sh mirrors ENERGIES[...]["tasks"]).']
    (HERE / 'MANIFEST.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()

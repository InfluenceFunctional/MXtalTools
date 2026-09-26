"""
acr_finish_sep27: 4 jobs, acridine sg14 Z'=2 (MACE), each < 1 day on one GPU. One question, at two levels: how much
does the reduction-penalty wall (10^4 x relu(reduction_en), enforce_reduced) hold walkers away from lower energy?
  Search level: random starts under wrap with enforce_reduced off in both stages (acr_wrap_sep26's no-reduction arm
    doubled the <=2 kT fraction, but its data were contaminated). Compare with acr_proposals_sep27's wrap arms.
  End-state level, paired: the same acr_proposals_sep27 wrap end states continued for up to 300 more steps of the
    search's stage 2 (Rprop), once with the penalty (the control: how converged stage 2 already is) and once without
    it (how far the wall held each state). Every state is its own baseline; trajectories are saved.
Local basis (8 sep26 wrap end states, 100 steps): with the wall, <= 0.13 kJ/mol more for 7 of 8; without it, 3 of 8
dropped 0.9-2.5 kJ/mol.

Job (array index) -> config:
  0  nored      random starts, wrap, no reduction penalty, seed S3             4000
  1  nored      random starts, wrap, no reduction penalty, seed S4             4000
  2  finwall    wrap end states (build_finish_seeds), stage 2 + penalty, 300    2564
  3  finnowall  the same end states, stage 2 without the penalty, 300 steps     2564
Seeds are 1e8 apart and from acr_proposals_sep27's (1e8, 2e8): starts are drawn with opt_seed + batch_idx * 10000.
4000 random starts per no-reduction job fit 20 h on the slower GPUs seen in acr_proposals_sep27 (~4 per minute).

    python make_configs.py      # writes the yamls + MANIFEST.md and asserts the checks below
"""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_finish_seeds import SOURCES, expected_rows  # noqa: E402

BASE = HERE.parent / 'acr_rerun_aug21' / 'acridine.yaml'
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
S3, S4 = 300_000_000, 400_000_000
PRIOR_SEEDS = (100_000_000, 200_000_000)  # acr_proposals_sep27 random arms
SG, ZP = 14, 2
FINISH_STEPS = 300


def base_cfg():
    cfg = yaml.safe_load(BASE.read_text())
    cfg.pop('umbrella_path', None)
    for st in cfg['opt']:
        for k in [k for k in st if k.startswith('umbrella')]:
            st.pop(k)
    cfg.update(sgs_to_search=[SG], zp_to_search=[ZP], init_target_cp='std', batch_size=48, grow_batch_size=True,
               out_dir=f'{DATA}/opt_outs', force_restart_run=False)
    return cfg


def nored(chunk, opt_seed, n):
    cfg = base_cfg()
    for st in cfg['opt']:
        st['centroid_boundary'] = 'wrap'
        st['enforce_reduced'] = False
    cfg.update(opt_seed=opt_seed, num_samples=n, save_trajs=chunk == 0,
               run_name=f'sep27nored_acridine_sg{SG}_zp{ZP}_{chunk}')
    return cfg


def finish(wall):
    cfg = base_cfg()
    stage = dict(cfg['opt'][-1])  # the search's final stage: MACE, Rprop, no compression
    stage.update(centroid_boundary='wrap', max_num_steps=FINISH_STEPS, enforce_reduced=wall)
    cfg['opt'] = [stage]
    cfg.update(init_sample_method='data', dataset_path=f'{DATA}/sep27_finish/finish_wrap.pt', mol_seed=0,
               opt_seed=0, num_samples=expected_rows('wrap'), save_trajs=True,
               run_name=f"sep27fin{'wall' if wall else 'nowall'}_acridine_sg{SG}_zp{ZP}_0")
    return cfg


JOBS = {'0': nored(0, S3, 4000), '1': nored(1, S4, 4000), '2': finish(True), '3': finish(False)}


def check(jobs):
    base = base_cfg()
    names = [c['run_name'] for c in jobs.values()]
    assert len(set(names)) == len(names), 'run_name collision'
    for key, c in jobs.items():
        flat = yaml.dump(c)
        assert 'D:' not in flat and '\\' not in flat, f'{key}: local Windows path leaked'
        assert 'umbrella' not in flat, f'{key}: retired umbrella key'
        assert c['sgs_to_search'] == [SG] and c['zp_to_search'] == [ZP]
        assert c['mace_predictor_path'] == base['mace_predictor_path'], 'the sep27 end states were scored by this model'
        assert all(st['optimizer_func'] == 'rprop' for st in c['opt'])
    seeds = [jobs['0']['opt_seed'], jobs['1']['opt_seed'], *PRIOR_SEEDS]
    assert min(abs(a - b) for i, a in enumerate(seeds) for b in seeds[i + 1:]) >= 10 ** 7, 'seed streams overlap'
    other = lambda stages: [{k: v for k, v in st.items() if k not in ('centroid_boundary', 'enforce_reduced')}
                            for st in stages]
    for key in ('0', '1'):
        assert all(st['centroid_boundary'] == 'wrap' and st['enforce_reduced'] is False for st in jobs[key]['opt'])
        assert other(jobs[key]['opt']) == other(base['opt']), 'no-reduction arm differs from the base schedule elsewhere'
    wall, nowall = jobs['2'], jobs['3']
    assert wall['dataset_path'] == nowall['dataset_path'] and wall['num_samples'] == nowall['num_samples'] == \
        expected_rows('wrap')
    assert all(run.startswith('sep27rwrap') or run.startswith('sep27seedwrap') for run, _ in SOURCES['wrap'])
    assert wall['opt'][0] == dict(base['opt'][-1], centroid_boundary='wrap', max_num_steps=FINISH_STEPS), \
        'the control must be the search stage 2 with only the step cap and boundary set'
    assert nowall['opt'][0] == dict(wall['opt'][0], enforce_reduced=False), 'the pair must differ only in the penalty'
    assert wall['opt'][0]['enforce_reduced'] is True and wall['opt'][0]['compression_factor'] == 0
    print(f'checks passed on {len(jobs)} configs')


def main():
    for old in HERE.glob('*.yaml'):
        old.unlink()
    for key, cfg in JOBS.items():
        (HERE / f'{key}.yaml').write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    check({k: yaml.safe_load((HERE / f'{k}.yaml').read_text()) for k in JOBS})
    rows = ['# acr_finish_sep27 manifest (generated by make_configs.py)', '',
            '| config | run_name | stages | reduction penalty | opt_seed | samples | starts |',
            '|---|---|---|---|---|---|---|']
    for key, c in JOBS.items():
        rows.append(f"| {key}.yaml | {c['run_name']} | {len(c['opt'])} (max steps "
                    f"{', '.join(str(s['max_num_steps']) for s in c['opt'])}) | {c['opt'][-1]['enforce_reduced']} | "
                    f"{c['opt_seed']} | {c['num_samples']} | {c.get('dataset_path') or 'random'} |")
    (HERE / 'MANIFEST.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()

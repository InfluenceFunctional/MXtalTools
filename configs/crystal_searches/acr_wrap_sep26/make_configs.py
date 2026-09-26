"""
acr_wrap_sep26: 8 test jobs, acridine sg14 Z'=2 (MACE), each < 1 day on one A100.

Question: does the opt-in centroid boundary (`centroid_boundary: wrap`, mxtaltools 4d13543b) change what
the crystal search finds, against today's asymmetric-unit clamp ('clamp', the default), under the same code?
The aug21 outputs are NOT the control: they predate the third monoclinic reduction wall and the no_grad fix
(c7204ced), so every comparison here is inside this battery.

Settings are the aug21 `base` schedule (../acr_rerun_aug21/acridine.yaml, sg14 Z'=2 overrides) except:
  - centroid_boundary set explicitly on every stage;
  - batch_size 48 with grow_batch_size on, so the out-of-memory back-off (x0.9) can climb back (x1.2 per set)
    instead of stranding a job at a small batch;
  - no umbrella keys (removed from mxtaltools; run_search rejects them).

Starts are drawn per batch with seed opt_seed + batch_index * 10000 for that batch's size, so arms sharing
an opt_seed get identical starts only while their batch sizes match; under the out-of-memory back-off the
pairing is statistical, not start-by-start.

Job (array index) -> configs, run in order by submit.sbatch:
  0  rclamp   random starts, clamp, seed A                6000
  1  rwrap    random starts, wrap,  seed A                6000
  2  rclamp   random starts, clamp, seed B                6000
  3  rwrap    random starts, wrap,  seed B                6000
  4  rwrapnr  random starts, wrap,  seed A, no reduction penalty (enforce_reduced false, both stages)   6000
  5  rwrapl2  random starts, wrap,  seed A, stage 2 capped at 400 steps instead of 150                   4000
  6a/6b  seedwrap   seeded return test, wrap: seed shards 0 (ACRDIN07) and 1 (ACRDIN06)
  7a/7b  seedclamp  seeded return test, clamp: the same shards

    python make_configs.py      # writes <job>.yaml / <job>a.yaml, <job>b.yaml and MANIFEST.md, then checks them
"""
from copy import deepcopy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'acr_rerun_aug21' / 'acridine.yaml'
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
SEED_A, SEED_B = 260926, 360926
SG, ZP = 14, 2


def base_cfg():
    cfg = yaml.safe_load(BASE.read_text())
    cfg.pop('umbrella_path', None)
    for st in cfg['opt']:
        for k in [k for k in st if k.startswith('umbrella')]:
            st.pop(k)
    cfg.update(sgs_to_search=[SG], zp_to_search=[ZP], init_target_cp='std', batch_size=48, grow_batch_size=True,
               out_dir=f'{DATA}/opt_outs', force_restart_run=False)
    return cfg


def arm(stem, chunk, boundary, opt_seed, n, trajs=False, seeds_shard=None, stage=lambda i, s: s):
    cfg = base_cfg()
    for i, st in enumerate(cfg['opt']):
        st['centroid_boundary'] = boundary
        cfg['opt'][i] = stage(i, st)
    cfg.update(opt_seed=opt_seed, num_samples=n, save_trajs=trajs, run_name=f'sep26{stem}_acridine_sg{SG}_zp{ZP}_{chunk}')
    if seeds_shard is not None:
        cfg.update(init_sample_method='data', dataset_path=f'{DATA}/seeds_sg{SG}_zp{ZP}_{seeds_shard}.pt', mol_seed=0)
    return cfg


def no_reduction(i, st):
    st['enforce_reduced'] = False
    return st


def long_stage2(i, st):
    if i == 1:
        st['max_num_steps'] = 400
    return st


JOBS = {
    '0': arm('rclamp', 0, 'clamp', SEED_A, 6000, trajs=True),
    '1': arm('rwrap', 0, 'wrap', SEED_A, 6000, trajs=True),
    '2': arm('rclamp', 1, 'clamp', SEED_B, 6000),
    '3': arm('rwrap', 1, 'wrap', SEED_B, 6000),
    '4': arm('rwrapnr', 0, 'wrap', SEED_A, 6000, stage=no_reduction),
    '5': arm('rwrapl2', 0, 'wrap', SEED_A, 4000, stage=long_stage2),
    '6a': arm('seedwrap', 0, 'wrap', 106, 2000, trajs=True, seeds_shard=0),
    '6b': arm('seedwrap', 1, 'wrap', 107, 2000, trajs=True, seeds_shard=1),
    '7a': arm('seedclamp', 0, 'clamp', 106, 2000, trajs=True, seeds_shard=0),
    '7b': arm('seedclamp', 1, 'clamp', 107, 2000, trajs=True, seeds_shard=1),
}


def check(jobs):
    """What the manifest claims, asserted on the written configs."""
    names = [c['run_name'] for c in jobs.values()]
    assert len(set(names)) == len(names), 'run_name collision: one arm would clobber another'
    for key, c in jobs.items():
        flat = yaml.dump(c)
        assert 'D:' not in flat and '\\' not in flat, f'{key}: local Windows path leaked into a cluster config'
        assert 'umbrella' not in flat, f'{key}: retired umbrella key (run_search would reject it)'
        assert c['sgs_to_search'] == [SG] and c['zp_to_search'] == [ZP]
        assert all(st['centroid_boundary'] in ('clamp', 'wrap') for st in c['opt'])
        assert len({st['centroid_boundary'] for st in c['opt']}) == 1, f'{key}: stages disagree on the boundary'
    same = lambda a, b, drop: {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}
    strip_b = lambda c: dict(c, opt=[{k: v for k, v in st.items() if k != 'centroid_boundary'} for st in c['opt']])
    # clamp vs wrap pairs differ ONLY in the boundary (and run_name / trajs)
    for a, b in (('0', '1'), ('2', '3'), ('7a', '6a'), ('7b', '6b')):
        assert same(strip_b(jobs[a]), strip_b(jobs[b]), {'run_name', 'save_trajs'}), f'{a} vs {b} differ beyond the boundary'
        assert jobs[a]['opt'][0]['centroid_boundary'] == 'clamp' and jobs[b]['opt'][0]['centroid_boundary'] == 'wrap'
    assert jobs['0']['opt_seed'] == jobs['1']['opt_seed'] == jobs['4']['opt_seed'] == jobs['5']['opt_seed'] != jobs['2']['opt_seed']
    assert [st['enforce_reduced'] for st in jobs['4']['opt']] == [False, False]
    assert [st['enforce_reduced'] for st in jobs['1']['opt']] == [True, True]
    assert [st['max_num_steps'] for st in jobs['5']['opt']] == [350, 400]
    assert [st['max_num_steps'] for st in jobs['1']['opt']] == [350, 150]
    print(f'checks passed on {len(jobs)} configs')


def main():
    for old in HERE.glob('*.yaml'):
        old.unlink()
    for key, cfg in JOBS.items():
        (HERE / f'{key}.yaml').write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    check({k: yaml.safe_load((HERE / f'{k}.yaml').read_text()) for k in JOBS})
    rows = ['# acr_wrap_sep26 manifest (generated by make_configs.py)', '',
            '| config | run_name | boundary | opt_seed | samples | enforce_reduced | stage-2 steps | trajs | seeds |',
            '|---|---|---|---|---|---|---|---|---|']
    for key, c in JOBS.items():
        rows.append(f"| {key}.yaml | {c['run_name']} | {c['opt'][0]['centroid_boundary']} | {c['opt_seed']} | "
                    f"{c['num_samples']} | {c['opt'][0]['enforce_reduced']} | {c['opt'][1]['max_num_steps']} | "
                    f"{c['save_trajs']} | {c.get('dataset_path') or '-'} |")
    (HERE / 'MANIFEST.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()

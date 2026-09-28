"""
acr_proposals_sep27: 9 jobs, acridine sg14 Z'=2 (MACE), each < 1 day on one GPU.

Requires the stale-intermediates fix in run_search (discard_intermediates): the acr_wrap_sep26 battery lost about
half its relaxations to it. Two questions:
  A. clamp vs wrap, cleanly: independent random-start seeds, run to completion, plus the seeded return test;
  B. proposal arms, all under wrap, judged against the random wrap arms by new low-energy families per unit of MACE
     compute: hops around known low-energy families, doubled Z'=1 structures with symmetry-breaking kicks, and the
     lowest-energy half of an eLJ pre-search as starts.

Job (array index) -> configs, run in order by submit.sbatch:
  0     rclamp        random starts, clamp, seed S1              6000
  1     rwrap         random starts, wrap,  seed S1              6000
  2     rclamp        random starts, clamp, seed S2              6000
  3     rwrap         random starts, wrap,  seed S2              6000
  4a/4b seedwrap      seeded return test, wrap:  seed shards 0 (241 ACRDIN07 + 41 ACRDIN06) and 1 (200 ACRDIN06 + 82 nik00009)
  5a/5b seedclamp     the same shards under the clamp
  6     hops          wrap; seeds/hops.pth (kicked copies of 297 low-energy families, log-noise -1.0 and -0.5)
  7     dblkick       wrap; seeds/doubled_kicked.pth (121 doubled Z'=1 families: unkicked + log-noise -2.0/-1.5/-1.0)
  8     eljstart      wrap; seeds/elj_starts.pth (lowest-eLJ half of a local eLJ pre-search)
Starts are drawn per batch with seed opt_seed + batch_idx * 10000 (batch_idx counts attempts, OOM retries included,
a few hundred per job), so S1 and S2 are 1e8 apart and far from any seed used before. The proposal seeds are compact
parameter files; submit.sbatch expands them on the cluster with build_seeds.py into DATA/sep27_seeds/.

    python make_configs.py      # writes the yamls + MANIFEST.md and asserts the checks below
"""
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'acr_rerun_aug21' / 'acridine.yaml'
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
S1, S2 = 100_000_000, 200_000_000
SG, ZP = 14, 2
#: compact seed files are .pth (not LFS-tracked); submit.sbatch expands seeds/<name>.pth to DATA/sep27_seeds/<name>.pt
PROPOSALS = {'6': ('hops', 'hops'), '7': ('dblkick', 'doubled_kicked'), '8': ('eljstart', 'elj_starts')}


def base_cfg():
    cfg = yaml.safe_load(BASE.read_text())
    cfg.pop('umbrella_path', None)
    for st in cfg['opt']:
        for k in [k for k in st if k.startswith('umbrella')]:
            st.pop(k)
    cfg.update(sgs_to_search=[SG], zp_to_search=[ZP], init_target_cp='std', batch_size=48, grow_batch_size=True,
               out_dir=f'{DATA}/opt_outs', force_restart_run=False)
    return cfg


def arm(stem, chunk, boundary, opt_seed, n, trajs=False, dataset=None):
    cfg = base_cfg()
    for st in cfg['opt']:
        st['centroid_boundary'] = boundary
    cfg.update(opt_seed=opt_seed, num_samples=n, save_trajs=trajs,
               run_name=f'sep27{stem}_acridine_sg{SG}_zp{ZP}_{chunk}')
    if dataset is not None:
        cfg.update(init_sample_method='data', dataset_path=dataset, mol_seed=0)
    return cfg


def seed_count(name):
    return len(torch.load(HERE / 'seeds' / f'{name}.pth', weights_only=False)['params'])


JOBS = {
    '0': arm('rclamp', 0, 'clamp', S1, 6000, trajs=True),
    '1': arm('rwrap', 0, 'wrap', S1, 6000, trajs=True),
    '2': arm('rclamp', 1, 'clamp', S2, 6000),
    '3': arm('rwrap', 1, 'wrap', S2, 6000),
    '4a': arm('seedwrap', 0, 'wrap', 0, 2000, trajs=True, dataset=f'{DATA}/seeds_sg{SG}_zp{ZP}_0.pt'),
    '4b': arm('seedwrap', 1, 'wrap', 0, 2000, trajs=True, dataset=f'{DATA}/seeds_sg{SG}_zp{ZP}_1.pt'),
    '5a': arm('seedclamp', 0, 'clamp', 0, 2000, trajs=True, dataset=f'{DATA}/seeds_sg{SG}_zp{ZP}_0.pt'),
    '5b': arm('seedclamp', 1, 'clamp', 0, 2000, trajs=True, dataset=f'{DATA}/seeds_sg{SG}_zp{ZP}_1.pt'),
}
for key, (stem, fname) in PROPOSALS.items():
    JOBS[key] = arm(stem, 0, 'wrap', 0, seed_count(fname), dataset=f'{DATA}/sep27_seeds/{fname}.pt')


def check(jobs):
    names = [c['run_name'] for c in jobs.values()]
    assert len(set(names)) == len(names), 'run_name collision'
    for key, c in jobs.items():
        flat = yaml.dump(c)
        assert 'D:' not in flat and '\\' not in flat, f'{key}: local Windows path leaked'
        assert 'umbrella' not in flat, f'{key}: retired umbrella key'
        assert c['sgs_to_search'] == [SG] and c['zp_to_search'] == [ZP]
        assert len({st['centroid_boundary'] for st in c['opt']}) == 1
    strip = lambda c: dict(c, opt=[{k: v for k, v in st.items() if k != 'centroid_boundary'} for st in c['opt']])
    same = lambda a, b: {k: v for k, v in strip(a).items() if k not in ('run_name', 'save_trajs')} == \
        {k: v for k, v in strip(b).items() if k not in ('run_name', 'save_trajs')}
    for a, b in (('0', '1'), ('2', '3'), ('5a', '4a'), ('5b', '4b')):
        assert same(jobs[a], jobs[b]), f'{a} vs {b} differ beyond the boundary'
        assert jobs[a]['opt'][0]['centroid_boundary'] == 'clamp' and jobs[b]['opt'][0]['centroid_boundary'] == 'wrap'
    assert abs(jobs['2']['opt_seed'] - jobs['0']['opt_seed']) >= 10 ** 7, 'random-start seed streams would overlap'
    for key, (stem, fname) in PROPOSALS.items():
        c = jobs[key]
        assert c['opt'][0]['centroid_boundary'] == 'wrap' and c['init_sample_method'] == 'data'
        assert c['dataset_path'].endswith(f'/sep27_seeds/{fname}.pt') and (HERE / 'seeds' / f'{fname}.pth').exists()
        assert c['num_samples'] == seed_count(fname)
    print(f'checks passed on {len(jobs)} configs')


def main():
    for old in HERE.glob('*.yaml'):
        old.unlink()
    for key, cfg in JOBS.items():
        (HERE / f'{key}.yaml').write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
    check({k: yaml.safe_load((HERE / f'{k}.yaml').read_text()) for k in JOBS})
    rows = ['# acr_proposals_sep27 manifest (generated by make_configs.py)', '',
            '| config | run_name | boundary | opt_seed | samples | trajs | starts |', '|---|---|---|---|---|---|---|']
    for key, c in JOBS.items():
        rows.append(f"| {key}.yaml | {c['run_name']} | {c['opt'][0]['centroid_boundary']} | {c['opt_seed']} | "
                    f"{c['num_samples']} | {c['save_trajs']} | {c.get('dataset_path') or 'random'} |")
    (HERE / 'MANIFEST.md').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()

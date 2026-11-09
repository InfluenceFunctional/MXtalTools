"""
pool8_nov09: streaming eLJ searches in the endpoint (ball) mode, SG2 (P-1), Z'=1, on every molecule of GFN training
pool v1 with at most 8 heavy atoms that has a conformer (72,464 conditions), for the conditional prior. CPU only.

Molecules (built on the dev box with gfn-diffusion energy_sampling, branch conditional):
  1. data_processing/training_pool_conformers.py 8 <dir>: one MMFF94 conformer per pool key (8 ETKDGv3 embeddings
     seeded by the key, lowest converged MMFF energy kept, stereochemistry checked); 90,685 keys -> 75,572
     (failures listed in pool8_failed.tsv: 13,344 no embedding, 1,645 stereo changed, 124 no MMFF parameters).
  2. prep_qm9_anchor_mols.py --src pool8_all.pt --n-mols 0 --chunk-size 25 --chunk-stem pool8_mols: the
     orient_molecule(mode='std') fixed point; 3,108 that move under a second standardization dropped; 72,464 kept,
     shuffled (seed 0), 2,899 chunks of 25 (the last 14). Identifier = pool key (InChIKey -SRel -FixedH).

Search (minima_stream/mipcas_elj_balls.yaml): Rprop, then BFGS stopped on force (fmax 0.02 kT/A), endpoints are the
minima (polish none, one Hessian per new candidate, a saddle descended), balls of 0.2 A; NUM_SAMPLES random starts per
molecule, no stop rule; every descent state recorded (cfg:stream.trajectories); no basin or band set (basin_set false).
One task = one chunk, all of its molecules' starts, so each condition's minima library is complete within its task.
At the end of a task slim_outputs.py writes <run>_slim.pt and deletes <run>.pt, _minima.pt and _traj.pt.

    python make_battery.py     # writes tasks/<i>.yaml, INDEX.tsv and the job script's array range; asserts the checks
"""
import math
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'minima_stream' / 'mipcas_elj_balls.yaml'
SBATCH = HERE / 'submit_pool8_nov09.sbatch'
N_CHUNKS, LAST_CHUNK_SIZE, CHUNK_SIZE = 2899, 14, 25
DATA = '/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09'
MOL_PATH = DATA + '/mols/pool8_mols_chunk{k}.pt'
OUT_DIR = DATA + '/search'
NUM_SAMPLES = 100           # random starts per molecule
SLOTS = 50                  # starts in flight on one core (= batch_size)
TASKS_PER_JOB = 8           # tasks run side by side in one job, one core each
CONCURRENT = 100            # array throttle: jobs at once
SEED_BASE = 200_000_000_000  # above every rc_oct22 / qm9_full_sep29 seed; task k uses SEED_BASE + k * 1e6


def task_cfg(k):
    cfg = yaml.safe_load(BASE.read_text())
    cfg.update(device='cpu', mol_path=MOL_PATH.format(k=k), out_dir=OUT_DIR, run_name=f'pool8_c{k}',
               sampling_mode='all', mols_to_sample=0, num_samples=NUM_SAMPLES, batch_size=SLOTS,
               opt_seed=SEED_BASE + k * 1_000_000, force_restart_run=False)
    cfg['stream'].update(slots=SLOTS, stop_p=None, basin_set=False, seeds=None, quota=None)
    assert cfg['minima']['polish'] == 'none' and cfg['stream']['trajectories'] is not None
    assert cfg['sgs_to_search'] == [2] and cfg['zp_to_search'] == [1]
    for st in cfg['opt']:
        assert st['optim_target'] == 'elj' and st['centroid_boundary'] == 'wrap'
    return cfg


def _strings(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)
    elif isinstance(node, str):
        yield node


def main():
    cfgs = [task_cfg(k) for k in range(N_CHUNKS)]
    sizes = [CHUNK_SIZE] * (N_CHUNKS - 1) + [LAST_CHUNK_SIZE]
    tasks_dir = HERE / 'tasks'
    tasks_dir.mkdir(exist_ok=True)
    for stale in tasks_dir.glob('*.yaml'):
        stale.unlink()
    rows = ['task\tchunk\tmolecules\tstarts\trun_name\topt_seed\tmol_path']
    for k, (cfg, n) in enumerate(zip(cfgs, sizes)):
        (tasks_dir / f'{k}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        rows.append(f"{k}\t{k}\t{n}\t{n * NUM_SAMPLES}\t{cfg['run_name']}\t{cfg['opt_seed']}\t{cfg['mol_path']}")
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    jobs = math.ceil(N_CHUNKS / TASKS_PER_JOB)
    text = SBATCH.read_text(encoding='utf-8')
    text, hits = re.subn(r'(?m)^#SBATCH --array=\S+$', f'#SBATCH --array=0-{jobs - 1}%{CONCURRENT}', text)
    assert hits == 1
    text, hits = re.subn(r'(?m)^TASKS_PER_JOB=\d+$', f'TASKS_PER_JOB={TASKS_PER_JOB}', text)
    assert hits == 1
    text, hits = re.subn(r'(?m)^N_TASKS=\d+$', f'N_TASKS={N_CHUNKS}', text)
    assert hits == 1
    SBATCH.write_text(text, encoding='utf-8', newline='\n')

    assert len({c['run_name'] for c in cfgs}) == len(cfgs) and len({c['opt_seed'] for c in cfgs}) == len(cfgs)
    # the stream seeds condition c's block b at opt_seed + 1e4 * (c * blocks per condition + b): inside 1e6 per task
    blocks = math.ceil(NUM_SAMPLES / min(SLOTS, NUM_SAMPLES))
    assert CHUNK_SIZE * blocks * 10_000 < 1_000_000, 'a task could reach the next task\'s seeds'
    for c in cfgs:
        for v in _strings(c):
            assert ':' not in v[:3] and '\\' not in v, f"local path in {c['run_name']}: {v!r}"
        assert c['mol_path'].startswith('/scratch/') and c['out_dir'].startswith('/scratch/')
    print(f'wrote {len(cfgs)} task configs, INDEX.tsv; {jobs} jobs of {TASKS_PER_JOB} tasks, '
          f'--array=0-{jobs - 1}%{CONCURRENT}; {sum(sizes):,} molecules x {NUM_SAMPLES} starts = '
          f'{sum(sizes) * NUM_SAMPLES:,} starts')


if __name__ == '__main__':
    main()

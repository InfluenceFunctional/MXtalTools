"""
pool8sg_nov15: the pool8_nov09 search battery (streaming eLJ searches, endpoint/ball mode, Z' = 1, 100 random starts
per molecule, every descent state recorded, CPU only) repeated in seven more space groups on the same 72,464 pool8
molecules and the same 2,899 chunks: P2_1/c (14), C2/c (15), P2_1 2_1 2_1 (19), P2_1 (4), Pbca (61), Pna2_1 (33), Cc (9).
Owner, 2026-11-14/15: more space groups before bigger molecules. P-1 (2) is pool8_nov09.

One battery per space group: task k of group g searches chunk k (pool8_mols_chunk<k>.pt) in g alone, run name
pool8sg<g>_c<k>, opt_seed SEED_BASE + g x 1e10 + k x 1e6 (distinct per (group, task); the stream's block seeds stay
inside 1e6, the tasks of one group inside 1e10). The search settings are pool8_nov09's (task_cfg below asserts it
against that battery's task 0), with only sgs_to_search, run_name, out_dir and opt_seed changed.

Molecules: pool8_nov09's chunk files as they are. run_search reads them unchanged (utils.init_samples_to_optim) and
re-orients every drawn start with orient_molecule('standardized', target_handedness=aunit_handedness) (stream_search
``drawn``), which at handedness +1 is align_mol_batch_to_standard_axes with its default frame: the frame the files
already hold (their fixed point, up to atom relabelling, for all but 13 of 72,464 molecules at b6155c4a) and the
one GFN's trainer applies at load.

Crystal handedness is +1 in every row: init_samples_to_optim builds each template with aunit_handedness 1, and the
random cell samplers redraw handedness only when some Z' > 1 (MolCrystalOps.sample_random_crystal_parameters,
sample_random_reduced_crystal_parameters); no config key changes it at Z' = 1, so task_cfg asserts zp_to_search == [1]
and the random start method instead.

    python make_battery.py     # writes tasks/sg<g>/<k>.yaml, INDEX.tsv, submit_sg<g>.sbatch, band_sg<g>.sbatch

The job scripts are rendered from submit_template.sbatch and band_template.sbatch (placeholders @NAME@); slimming and
banding reuse pool8_nov09's slim_outputs.py and band_from_slim.py unchanged. The search job bands each task after its
search by default (BAND_AFTER=1); band_sg<g>.sbatch is for slims searched with BAND_AFTER=0.
"""
import math
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
POOL8 = HERE.parent / 'pool8_nov09'
BASE = HERE.parent / 'minima_stream' / 'mipcas_elj_balls.yaml'
#: space group -> (Hermann-Mauguin symbol, wall time per search job). A job runs 8 tasks side by side, each its search
#: then its band step. pool8_nov09's P-1 searches took 25 min per task on the cluster (Xeon 8592+, smoke job 22430710).
#: Search wall time per start relative to P-1, on the dev box (5 pool8 molecules x 100 starts, 50 slots, 3 runs side
#: by side, 2026-11-15; README "Cost"): 0.8 to 1.1 in every group; the band step is under 5 min per task.
GROUPS = {14: ('P2_1/c', '02:00:00'), 15: ('C2/c', '02:00:00'), 19: ('P2_12_12_1', '02:00:00'), 4: ('P2_1', '02:00:00'),
          61: ('Pbca', '02:00:00'), 33: ('Pna2_1', '02:00:00'), 9: ('Cc', '02:00:00')}
N_CHUNKS, LAST_CHUNK_SIZE, CHUNK_SIZE = 2899, 14, 25
MOLS = '/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/mols'   # pool8_nov09's molecule files, shared
DATA = '/scratch/mk8347/data/crystal_datasets/conditional/pool8sg_nov15'
MOL_PATH = MOLS + '/pool8_mols_chunk{k}.pt'
NUM_SAMPLES = 100           # random starts per molecule (as pool8_nov09)
SLOTS = 50                  # starts in flight on one core (= batch_size)
TASKS_PER_JOB = 8           # tasks run side by side in one job, one core each
CONCURRENT = 100            # array throttle per group: jobs at once
SEED_BASE = 300_000_000_000  # above every pool8_nov09 seed (2e11 + k x 1e6 < 2.03e11)
SEED_GROUP = 10_000_000_000  # group g's tasks use SEED_BASE + g x 1e10 + k x 1e6
TEMPLATE_KEYS = ('@SG@', '@SYMBOL@', '@TIME@', '@ARRAY@', '@TASKS_PER_JOB@', '@N_TASKS@')


def out_dir(sg):
    return f'{DATA}/search/sg{sg}'


def task_cfg(sg, k, *, mol_path=None, out=None, num_samples=NUM_SAMPLES, slots=SLOTS):
    """Task k of group sg: pool8_nov09's task-k config with the space group, run name, output folder and seed
    changed. mol_path / out / num_samples / slots override for a local smoke run."""
    cfg = yaml.safe_load(BASE.read_text())
    cfg.update(device='cpu', mol_path=mol_path or MOL_PATH.format(k=k), out_dir=out or out_dir(sg),
               run_name=f'pool8sg{sg}_c{k}', sampling_mode='all', mols_to_sample=0, num_samples=num_samples,
               batch_size=slots, sgs_to_search=[int(sg)], opt_seed=SEED_BASE + int(sg) * SEED_GROUP + k * 1_000_000,
               force_restart_run=False)
    cfg['stream'].update(slots=slots, stop_p=None, basin_set=False, seeds=None, quota=None)
    assert cfg['minima']['polish'] == 'none' and cfg['stream']['trajectories'] is not None
    assert cfg['zp_to_search'] == [1], "Z' > 1 draws a random handedness per unit"
    assert cfg['init_sample_method'] == 'random' and cfg['init_reduced'] is True
    for st in cfg['opt']:
        assert st['optim_target'] == 'elj' and st['centroid_boundary'] == 'wrap'
    return cfg


def same_as_pool8(cfg):
    """Every setting but the ones this battery changes equals pool8_nov09's task 0 (asserted, not assumed)."""
    ref = yaml.safe_load((POOL8 / 'tasks' / '0.yaml').read_text())
    changed = {'mol_path', 'out_dir', 'run_name', 'sgs_to_search', 'opt_seed'}
    for key in set(ref) | set(cfg):
        if key not in changed:
            assert ref.get(key) == cfg.get(key), f'{key}: {cfg.get(key)!r} differs from pool8_nov09 {ref.get(key)!r}'


def _strings(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)
    elif isinstance(node, str):
        yield node


def render(template, sg, jobs):
    sym, wall = GROUPS[sg]
    text = (HERE / template).read_text(encoding='utf-8')
    for k, v in zip(TEMPLATE_KEYS, (str(sg), sym, wall, f'0-{jobs - 1}%{CONCURRENT}', str(TASKS_PER_JOB),
                                     str(N_CHUNKS))):
        text = text.replace(k, v)
    assert not re.search(r'@[A-Z_]+@', text), f'{template}: unfilled placeholder'
    return text


def main():
    sizes = [CHUNK_SIZE] * (N_CHUNKS - 1) + [LAST_CHUNK_SIZE]
    jobs = math.ceil(N_CHUNKS / TASKS_PER_JOB)
    rows = ['sg\ttask\tchunk\tmolecules\tstarts\trun_name\topt_seed\tmol_path']
    seeds, names = set(), set()
    for sg in GROUPS:
        cfgs = [task_cfg(sg, k) for k in range(N_CHUNKS)]
        same_as_pool8(cfgs[0])
        d = HERE / 'tasks' / f'sg{sg}'
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob('*.yaml'):
            stale.unlink()
        for k, (cfg, n) in enumerate(zip(cfgs, sizes)):
            (d / f'{k}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
            rows.append(f"{sg}\t{k}\t{k}\t{n}\t{n * NUM_SAMPLES}\t{cfg['run_name']}\t{cfg['opt_seed']}\t"
                        f"{cfg['mol_path']}")
            seeds.add(cfg['opt_seed'])
            names.add(cfg['run_name'])
            for v in _strings(cfg):
                assert ':' not in v[:3] and '\\' not in v, f"local path in {cfg['run_name']}: {v!r}"
            assert cfg['mol_path'].startswith('/scratch/') and cfg['out_dir'].startswith('/scratch/')
        for tmpl, name in (('submit_template.sbatch', f'submit_sg{sg}.sbatch'),
                           ('band_template.sbatch', f'band_sg{sg}.sbatch')):
            (HERE / name).write_text(render(tmpl, sg, jobs if tmpl.startswith('submit') else
                                            math.ceil(N_CHUNKS / 64)), encoding='utf-8', newline='\n')
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')

    n_tasks = N_CHUNKS * len(GROUPS)
    assert len(names) == n_tasks and len(seeds) == n_tasks, 'run names or seeds collide'
    # the stream seeds condition c's block b at opt_seed + 1e4 * (c * blocks per condition + b): inside 1e6 per task
    blocks = math.ceil(NUM_SAMPLES / min(SLOTS, NUM_SAMPLES))
    assert CHUNK_SIZE * blocks * 10_000 < 1_000_000, "a task could reach the next task's seeds"
    assert N_CHUNKS * 1_000_000 < SEED_GROUP, "a group's tasks could reach the next group's seeds"
    assert SEED_BASE > 200_000_000_000 + N_CHUNKS * 1_000_000, 'seeds overlap pool8_nov09'
    print(f'wrote {n_tasks} task configs ({len(GROUPS)} groups x {N_CHUNKS}), INDEX.tsv, '
          f'{len(GROUPS)} search and band job scripts; per group {jobs} jobs of {TASKS_PER_JOB} tasks, '
          f'--array=0-{jobs - 1}%{CONCURRENT}; {sum(sizes):,} molecules x {NUM_SAMPLES} starts x {len(GROUPS)} '
          f'groups = {sum(sizes) * NUM_SAMPLES * len(GROUPS):,} starts')


if __name__ == '__main__':
    main()

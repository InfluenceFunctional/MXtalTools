"""
pool8sg_nov15 control: 1,000 starts per molecule on two pool8 chunks (50 molecules) in all eight groups, to measure
what the battery's 100 starts miss. Each task is the battery's task for that chunk and group (make_battery.task_cfg:
pool8_nov09's settings) with num_samples 1000, its own run name, output folder and seed block.

Chunks 1287 and 1571 are pool8_nov09's reference-panel tasks: their 100-start P-1 searches are the production ones
(38 train, 7 heldout_random and 5 heldout_scaffold molecules). The stream numbers start j of condition c as
c x num_samples + j, so the first 100 starts of each condition are, statistically, a 100-start search with the
battery's settings (pool8_nov09/band_from_slim.py --starts 100 subsets them); the question per (molecule, group) is whether the
lowest minimum of 1,000 starts is among the minima of the first 100, and how much in-window minimum mass 100 miss.

Seeds: CONTROL_SEED + g x 1e10 + j x 1e7 for chunk j of the two; one task spans 25 x 20 blocks x 1e4 = 5e6 < 1e7.
They are distinct from the battery's (3e11 + g x 1e10 + k x 1e6 < 9.2e11) and pool8_nov09's (< 2.03e11).

usage: python make_control.py   (writes tasks/<i>.yaml, INDEX.tsv and submit_control1000.sbatch here)
"""
import math
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_battery import DATA, GROUPS, MOL_PATH, task_cfg  # noqa: E402

CHUNKS = (1287, 1571)
SGS = (2,) + tuple(GROUPS)       # P-1 too: pool8_nov09's production searches of these chunks are its 100-start runs
NUM_SAMPLES = 1000
SLOTS = 50
CONTROL_SEED = 1_000_000_000_000
OUT = DATA + '/control1000/search/sg{sg}'


def control_cfg(sg, j, k):
    cfg = task_cfg(sg, k, num_samples=NUM_SAMPLES, slots=SLOTS)
    cfg.update(run_name=f'pool8ctl{sg}_c{k}', out_dir=OUT.format(sg=sg),
               opt_seed=CONTROL_SEED + int(sg) * 10_000_000_000 + j * 10_000_000)
    assert cfg['mol_path'] == MOL_PATH.format(k=k) and cfg['sgs_to_search'] == [sg]
    return cfg


def main(time, mem):
    d = HERE / 'tasks'
    d.mkdir(exist_ok=True)
    for stale in d.glob('*.yaml'):
        stale.unlink()
    rows = ['task\tsg\tchunk\tstarts_per_molecule\trun_name\topt_seed\tout_dir']
    seeds = set()
    i = 0
    for sg in SGS:
        for j, k in enumerate(CHUNKS):
            cfg = control_cfg(sg, j, k)
            (d / f'{i}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
            rows.append(f"{i}\t{sg}\t{k}\t{NUM_SAMPLES}\t{cfg['run_name']}\t{cfg['opt_seed']}\t{cfg['out_dir']}")
            seeds.add(cfg['opt_seed'])
            i += 1
    assert len(seeds) == i
    blocks = math.ceil(NUM_SAMPLES / SLOTS)
    assert 25 * blocks * 10_000 < 10_000_000, 'a task could reach the next task seeds'
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    text = (HERE / 'control_template.sbatch').read_text(encoding='utf-8')
    for key, v in (('@TIME@', time), ('@MEM@', mem), ('@ARRAY@', f'0-{i - 1}'), ('@N_TASKS@', str(i))):
        text = text.replace(key, v)
    assert not re.search(r'@[A-Z_]+@', text)
    (HERE / 'submit_control1000.sbatch').write_text(text, encoding='utf-8', newline='\n')
    print(f'{i} control tasks ({len(SGS)} groups x {len(CHUNKS)} chunks, {NUM_SAMPLES} starts per molecule), '
          f'array 0-{i - 1}, {time}, {mem}')


if __name__ == '__main__':
    # one task = 25 molecules x 1,000 starts. Measured 2026-11-15 on the dev box: 3 molecules x 1,000 starts in C2/c,
    # 0.37 s/start on one core (1,118 s), 16.4 KB of slim per start. pool8_nov09 ran 0.6 s/start on the cluster's
    # Xeon 8592+, so ~4.2 h per task; 12 h leaves room. 24 GB: the battery's tasks fit 8 GB at a tenth of the states.
    main(time='12:00:00', mem='24G')

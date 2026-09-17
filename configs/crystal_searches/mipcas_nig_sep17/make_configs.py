"""
MIPCAS sg 2 Z'=1: 50k more ELJ and 50k more UMA searches under the triclinic Niggli penalty (MXT_NIGGLI_TRICLINIC=1, set by
submit.sbatch). 10 arms x 5000 per energy, array indices 0-9 ELJ and 10-19 UMA.

RUN NAMES. load_search_chunks globs <run_name>_*.pt and keeps all-digit tails, so each energy gets its own stem
(mipcas_nig_elj_<arm>, mipcas_nig_uma_<arm>), distinct from the March chunks (mipcas_elj_*, mipcas_uma_*): the old and new
penalties write different cell conventions, and they are collated together only on purpose.

UMA CHUNKS ARE UNSTAMPED. run_search writes plain lists; load_search_chunks(require_uma_state=True) refuses a UMA chunk without
uma_energy_state >= 2 (F-047). These energies come from the fixed route, but they still need stamping (or a rescore) before
collate_prior will read them.

    python make_configs.py      # writes 0.yaml .. 19.yaml + MANIFEST.md
"""
from copy import deepcopy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / 'mipcas_nig.yaml'
ARMS_PER_ENERGY = 10
ARM_SAMPLES = 5000
#: UMA: a local smoke peaked at 9.1 GB for 50 rows (0.09 starts/s on an RTX 5080), so 500 is near an 80 GB card's ceiling;
#: run_search's OOM handler shrinks it by 10% per OOM if the GPU is smaller. ELJ: 200 rows peaked at 2.5 GB.
BATCH_SIZE = {'elj': 5000, 'uma': 500}
TRAJ_ARMS = 1                    # save_trajs on the first arm of each energy only
SEED0 = {'elj': 5000, 'uma': 6000}   # unused by earlier MIPCAS runs (0-9 March, 0-4 and 3000+ local)


def assert_no_local_paths(node, trail='cfg'):
    """No Windows drive path may survive into a cluster arm (see gfn_diffusion configs/prod_aug26/make.py)."""
    if isinstance(node, dict):
        for k, v in node.items():
            assert_no_local_paths(v, f'{trail}.{k}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            assert_no_local_paths(v, f'{trail}[{i}]')
    elif isinstance(node, str):
        assert not (len(node) > 2 and node[1] == ':' and node[2] in '\\/'), \
            f'local drive path leaked into a cluster arm at {trail}: {node!r}'


def main():
    base = yaml.safe_load(BASE.read_text())
    stale = [q for q in HERE.glob('*.yaml') if q.stem.isdigit()]
    for q in stale:
        q.unlink()
    ind, rows, names, seeds = 0, [], set(), set()
    for energy in ('elj', 'uma'):
        for arm in range(ARMS_PER_ENERGY):
            cfg = deepcopy(base)
            for stage in cfg['opt']:
                stage['optim_target'] = energy
            cfg['num_samples'] = ARM_SAMPLES
            cfg['batch_size'] = BATCH_SIZE[energy]
            cfg['opt_seed'] = SEED0[energy] + arm
            cfg['save_trajs'] = arm < TRAJ_ARMS
            cfg['run_name'] = f'mipcas_nig_{energy}_{arm}'
            if energy != 'uma':
                cfg['uma_predictor_path'] = None
            assert_no_local_paths(cfg)
            assert cfg['run_name'] not in names and cfg['opt_seed'] not in seeds
            names.add(cfg['run_name']); seeds.add(cfg['opt_seed'])
            (HERE / f'{ind}.yaml').write_text(yaml.dump(cfg, default_flow_style=False))
            rows.append((ind, energy, arm, cfg['run_name'], cfg['opt_seed'], cfg['save_trajs']))
            ind += 1
    lines = ['# mipcas_nig_sep17', '',
             f'{ind} arms: {ARMS_PER_ENERGY} x {ARM_SAMPLES} starts per energy (ELJ, UMA), sg 2, Z\'=1, triclinic Niggli penalty '
             '(MXT_NIGGLI_TRICLINIC=1 in submit.sbatch).', '',
             '| array index | energy | arm | run_name | opt_seed | batch | save_trajs |', '|---|---|---|---|---|---|---|']
    lines += [f'| {i} | {e} | {a} | `{n}` | {s} | {BATCH_SIZE[e]} | {t} |' for i, e, a, n, s, t in rows]
    (HERE / 'MANIFEST.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {ind} arm configs + MANIFEST.md (removed {len(stale)} previously generated)')


if __name__ == '__main__':
    main()

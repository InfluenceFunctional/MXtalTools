"""
Assert the generated arms are what MANIFEST.md claims.

The failure mode for a battery like this is arms that differ only by NAME -- a
variation block that silently carries the baseline schedule looks like a
successful control and reports "no effect" for a knob that was never turned.
So every claim the manifest makes is checked against the YAML on disk.

    python check_configs.py
"""
import glob
from collections import defaultdict
from pathlib import Path

import yaml

#: modes that are SUPPOSED to carry the unmodified acr_production schedule.
#: `base` is the control; `seed` varies the starting point, not the schedule.
BASELINE_STEMS = {'aug21base', 'aug21seed'}
#: modes that must force a near-target start rather than inheriting the default
STD_START_STEMS = {'aug21stdnocomp'}


def main():
    arms = defaultdict(list)
    for path in sorted(glob.glob('*.yaml'), key=lambda p: p.name if hasattr(p, 'name') else p):
        if path == 'acridine.yaml':
            continue
        cfg = yaml.safe_load(Path(path).read_text())
        stem, _, arm = cfg['run_name'].rpartition('_')
        arms[stem].append((int(arm), cfg, path))

    base = yaml.safe_load(Path('acridine.yaml').read_text())
    problems = []

    # 1. arm index must be the final component and all digits, or
    #    load_search_chunks silently drops the file
    for stem, entries in arms.items():
        for arm, cfg, path in entries:
            tail = cfg['run_name'][len(stem) + 1:]
            if not tail.isdigit():
                problems.append(f"{path}: run_name tail {tail!r} is not all digits "
                                f"-- load_search_chunks would skip it")

    # 2. no stem may be another stem followed by digits, or two blocks collate
    #    into one
    for a in arms:
        for b in arms:
            if a != b and b.startswith(a + '_') and b[len(a) + 1:].isdigit():
                problems.append(f"stem {b!r} collides with {a!r}")

    # 3. seeds within a block must be distinct, or arms duplicate each other
    for stem, entries in arms.items():
        seeds = [c['opt_seed'] for _, c, _ in entries]
        if len(set(seeds)) != len(seeds):
            problems.append(f"{stem}: duplicate opt_seed across arms")

    # 4. every variation block must actually differ from the baseline schedule
    base_opt = base['opt']
    for stem, entries in arms.items():
        tag = stem.split('_acridine_')[0]
        _, cfg, path = entries[0]
        same = cfg['opt'] == base_opt
        if tag in BASELINE_STEMS and not same:
            problems.append(f"{stem}: expected the baseline schedule, got a "
                            f"modified one")
        if tag not in BASELINE_STEMS and same:
            problems.append(f"{stem}: VARIATION BLOCK CARRIES THE BASELINE "
                            f"SCHEDULE -- the knob was never turned")

    # 5. the std* modes must actually set init_target_cp, or they are just
    #    compression variants wearing the wrong name
    for stem, entries in arms.items():
        tag = stem.split('_acridine_')[0]
        if tag in STD_START_STEMS:
            for arm, cfg, path in entries:
                if cfg.get('init_target_cp') != 'std':
                    problems.append(f"{path}: {tag} must set init_target_cp='std', "
                                    f"got {cfg.get('init_target_cp')!r}")

    # 6. seeded block must actually be seeded
    for stem, entries in arms.items():
        if stem.startswith('aug21seed'):
            for arm, cfg, path in entries:
                if cfg.get('init_sample_method') != 'data':
                    problems.append(f"{path}: seeded block is not init_sample_method=data")
                if not cfg.get('dataset_path'):
                    problems.append(f"{path}: seeded block has no dataset_path")
            #: the seeded arms are sharded -- one seed FILE each, mol_seed 0 --
            #: because init_samples_to_optim clips num_samples to the seed count
            #: and then indexes arange(mol_seed*n, (mol_seed+1)*n), which runs off
            #: the end for mol_seed > 0. So what must be distinct is the PATH.
            paths = [c['dataset_path'] for _, c, _ in entries]
            if len(set(paths)) != len(paths):
                problems.append(f"{stem}: arms share dataset_path, so they would "
                                f"optimize the SAME seeds")
            for arm, cfg, path in entries:
                if cfg.get('mol_seed', 0) != 0:
                    problems.append(f"{path}: sharded seeds require mol_seed 0; "
                                    f"got {cfg['mol_seed']}, which would index "
                                    f"past the end of the shard")

    print(f"{sum(len(v) for v in arms.values())} arms across {len(arms)} blocks\n")
    print(f"{'stem':40s} {'arms':>5} {'stage lrs':>18} {'compression':>12} "
          f"{'stages':>7} {'trajs':>6}")
    for stem in sorted(arms):
        entries = sorted(arms[stem])
        _, cfg, _ = entries[0]
        lrs = "/".join(str(s['init_lr']) for s in cfg['opt'])
        comp = "/".join(str(s['compression_factor']) for s in cfg['opt'])
        ntraj = sum(1 for _, c, _ in entries if c.get('save_trajs'))
        print(f"{stem:40s} {len(entries):5d} {lrs:>18} {comp:>12} "
              f"{len(cfg['opt']):7d} {ntraj:6d}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == '__main__':
    main()

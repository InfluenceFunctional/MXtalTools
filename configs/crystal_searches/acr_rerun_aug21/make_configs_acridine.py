"""
Acridine search battery, August 2026 re-run.

Five sampling modes PARTITION a fixed nominal budget per SG/Z' -- they do not add
to it -- so every landscape gets the diversified protocol and the per-combo total
stays comparable to `../acr_production` (`may_acridine_*`).

  base       the acr_production schedule, untouched. The control, and what makes
             this battery comparable to the last one.
  stdnocomp  start near target density and DON'T compress. Compression exists to
             drag a diffuse start up to a physical density; from a near-target
             start it is doing something else.
  lightcomp  the BASE start with light compression, so the compression axis is a
             clean single-variable change off the control.
  lowlr      stage 2 at a finer learning rate. Tests whether the final descent is
             too coarse to settle into a shallow basin.

Plus `seed`, a targeted probe that starts AT perturbed copies of known basins.
It is NOT part of the budget: its sample count is however many seeds exist.

WHY THESE. sg14-Z'2 missed BOTH of its known forms. Across all 64,927 raw samples
of the previous search the closest thing to ACRDIN07 sits at RDF distance 0.1428
and matches 11 of 20 molecules under COMPACK; ACRDIN06 the same. Already excluded,
so deliberately NOT varied here: thinning (the raw search is no closer than the
thinned prior), domain/Niggli exclusion (both forms have cell reduction penalty
exactly 0.0, no latent dimension outside the prior's range), parameterization (all
seven known forms rebuild from their own latent parameters at 20/20), pseudo-Z'=1
collapse (zero such samples), and symmetry (standard SYM_OPS throughout).

NOMINAL IS NOT WHAT LANDS. acr_production asked for 100k per combo and wrote
55-86k: two of fifty arms produced no output at all in each sg14 run, one chunk is
zero bytes, and per-arm yield ranged from 18 to 1993 of 2000 (medians 1094 / 1917
/ 1942 / 1680 for sg14-Z'1, sg14-Z'2, sg9-Z'2, sg19-Z'3). Budget against what
arrives, not what was requested.

RUN NAMES. `load_search_chunks` globs `<run_name>_*.pt` and keeps only all-digit
tails; `run_search` writes one file per arm as `<run_name>.pt`. So the arm index
must be the final component and every mode needs its own stem, or one mode's
chunks collate into another's.

    python make_configs_acridine.py       # writes 0.yaml .. N.yaml + MANIFEST.md
    python check_configs.py               # asserts the arms are what is claimed
"""
import json
import os
from copy import deepcopy
from pathlib import Path

import yaml

BASE_PATH = 'acridine.yaml'

#: the SG/Z' combinations Nikos' 15 accepted structures occupy
COVERAGE_COMBOS = [
    (14, 1),   # his zprime_1/P21n         (3)  our best-validated landscape
    (14, 2),   # his zprime_2/P21c + P21n  (5)  the failing landscape
    (9, 2),    # his zprime_2/Cc           (5)  validated, 5/5 of his matched
    (5, 2),    # his zprime_2/C2           (1)  NEVER SEARCHED
    (19, 3),   # his zprime_3/P212121      (1)  searched, never collated
]

COMBO_TOTAL = 100_000       # nominal samples per SG/Z', partitioned by the modes
ARM_SAMPLES = 2000          # per arm, as in acr_production
BATCH_SIZE = 1000
TRAJ_ARMS = 2               # arms per mode that keep mid-trajectory states
SEED_SHARDS = 6             # must equal make_seeds.py --shards
SEED_DIR = os.path.join('D:', os.sep, 'crystal_datasets', 'acridine')

LIGHT_COMPRESSION = 0.25    # vs 1.0 in the acr_production stage 1

#: save_trajs writes records['params'], (steps, batch, n_params) per stage per
#: batch -- ~25 MB each at batch 1000 / 350 steps / 18 params, so 100-200 MB per
#: arm. On for the first TRAJ_ARMS arms of each mode only.


def _stage(base_stage, **over):
    s = deepcopy(base_stage)
    s.update(over)
    return s


def modes(base):
    """
    The five sampling modes: budget share, optimizer schedule, config overrides.

    `frac` values are shares of COMBO_TOTAL; `base` absorbs the rounding
    remainder so the arms sum exactly.
    """
    s1, s2 = base['opt'][0], base['opt'][1]
    return {
        'base': dict(
            frac=0.40, opt=None, cfg={},
            q='unchanged acr_production schedule -- the control'),
        'stdnocomp': dict(
            frac=0.20,
            opt=[_stage(s1, compression_factor=0.0), s2],
            cfg={'init_target_cp': 'std'},
            q='start near target density, no stage-1 compression'),
        'lightcomp': dict(
            frac=0.20,
            opt=[_stage(s1, compression_factor=LIGHT_COMPRESSION), s2],
            cfg={},
            q=f'the BASE start with light compression ({LIGHT_COMPRESSION} vs '
              f'1.0) -- a single-variable change off the control'),
        'lowlr': dict(
            frac=0.20,
            opt=[s1, _stage(s2, init_lr=0.001, convergence_eps=1e-6,
                            max_num_steps=400)],
            cfg={},
            q='finer stage-2 descent: lr 0.001, tighter convergence, 400 steps'),
    }


SEED_QUESTION = (
    'are known basins reachable and HOLDABLE at all? starts AT perturbed copies '
    'of the known forms and his structures. NOT part of COMBO_TOTAL -- a '
    'targeted probe, not random sampling')


def arms_per_mode(mode_defs):
    """Split COMBO_TOTAL across the modes, exactly."""
    total_arms = COMBO_TOTAL // ARM_SAMPLES
    counts, assigned = {}, 0
    for name, spec in mode_defs.items():
        if name == 'base':
            continue
        counts[name] = round(total_arms * spec['frac'])
        assigned += counts[name]
    counts['base'] = total_arms - assigned
    assert counts['base'] > 0, "the mode fractions leave no arms for the control"
    assert sum(counts.values()) == total_arms
    return counts


def _shard_counts(sg, zp):
    """
    Seeds per shard, from the sidecar make_seeds.py writes.

    JSON rather than unpickling the shards: they hold MolCrystalData, so counting
    directly needs mxtaltools importable, and this generator normally runs from
    the config directory without it. That failure returned zero SILENTLY and
    reported "0 seeds" for a block holding 1407, so a missing sidecar raises.
    """
    p = Path(SEED_DIR) / f'seeds_sg{sg}_zp{zp}_counts.json'
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run make_seeds.py --sg {sg} --zp {zp} -- it writes the "
            f"sidecar alongside the shards.")
    return json.loads(p.read_text())


def main():
    base = yaml.safe_load(Path(BASE_PATH).read_text())
    #: clear previously generated arms. Regenerating a SMALLER battery over a
    #: larger one leaves stale arms behind, and a stale arm is indistinguishable
    #: from a live one to the launcher.
    stale = [q for q in Path('.').glob('*.yaml') if q.stem.isdigit()]
    for q in stale:
        q.unlink()
    if stale:
        print(f"removed {len(stale)} previously generated arm configs")

    mode_defs = modes(base)
    counts = arms_per_mode(mode_defs)
    ind = 0
    manifest = []

    def emit(sg, zp, mode, arm, opt, cfg_over, seeded=False):
        nonlocal ind
        cfg = deepcopy(base)
        cfg['sgs_to_search'] = [sg]
        cfg['zp_to_search'] = [zp]
        cfg['batch_size'] = BATCH_SIZE
        cfg['num_samples'] = ARM_SAMPLES
        cfg['opt_seed'] = ind
        #: acr_production's own rule: fixed 0.1 for Z'=1, 'std' for Z'>1. Kept for
        #: `base` so it stays comparable. NOTE the confound this creates at Z'=1:
        #: there `base` starts diffuse, so the std* modes differ from it in TWO
        #: ways at once (start density AND compression). At Z'>1 base is already
        #: 'std' and the comparison is clean.
        if zp > 1:
            cfg['init_target_cp'] = 'std'
        cfg.update(cfg_over)
        if opt is not None:
            cfg['opt'] = deepcopy(opt)
        if seeded:
            cfg['init_sample_method'] = 'data'
            #: one shard per arm, mol_seed 0. A single shared file would crash
            #: every arm but the first: init_samples_to_optim clips num_samples to
            #: the seed count then indexes arange(mol_seed*n, (mol_seed+1)*n).
            cfg['dataset_path'] = (
                '/scratch/mk8347/data/crystal_datasets/acridine/'
                f'seeds_sg{sg}_zp{zp}_{arm}.pt')
            cfg['mol_seed'] = 0
        cfg['save_trajs'] = arm < TRAJ_ARMS
        stem = f"aug21{mode}_acridine_sg{sg}_zp{zp}"
        cfg['run_name'] = f"{stem}_{arm}"
        Path(f'{ind}.yaml').write_text(yaml.dump(cfg, default_flow_style=False))
        ind += 1
        return stem

    for sg, zp in COVERAGE_COMBOS:
        for name, spec in mode_defs.items():
            stem = None
            for arm in range(counts[name]):
                stem = emit(sg, zp, name, arm, spec['opt'], spec['cfg'])
            manifest.append((name, sg, zp, counts[name],
                             counts[name] * ARM_SAMPLES, spec['q'], stem))

        if (Path(SEED_DIR) / f'seeds_sg{sg}_zp{zp}_0.pt').exists():
            shard_sizes = _shard_counts(sg, zp)
            if len(shard_sizes) != SEED_SHARDS:
                raise ValueError(
                    f"sg{sg} zp{zp}: make_seeds.py wrote {len(shard_sizes)} shards "
                    f"but SEED_SHARDS is {SEED_SHARDS}; arms would point at files "
                    f"that do not exist")
            stem = None
            for arm in range(SEED_SHARDS):
                stem = emit(sg, zp, 'seed', arm, None, {}, seeded=True)
            manifest.append(('seed', sg, zp, SEED_SHARDS, sum(shard_sizes),
                             SEED_QUESTION, stem))
        else:
            print(f"   no seed shards for sg{sg} zp{zp} -- run "
                  f"make_seeds.py --sg {sg} --zp {zp} to add the seeded probe")

    lines = [
        "# acridine re-run battery, August 2026", "",
        f"{ind} arms. Every stem is distinct and `load_search_chunks` keeps only",
        "all-digit tails, so modes cannot bleed into one another at collation.", "",
        f"The five sampling modes PARTITION {COMBO_TOTAL:,} nominal samples per",
        "SG/Z' -- they do not add to it. `seed` is a targeted probe on top, sized",
        "by the seed file rather than by arm count.", "",
        "**Nominal is not what lands.** acr_production asked for 100k per combo",
        "and wrote 55-86k (57 / 68 / 86 / 68 %): two of fifty arms produced no",
        "output at all in each sg14 run, one chunk is zero bytes, and per-arm",
        "yield ranged from 18 to 1993 of 2000. Count what arrives.", "",
        "**Confound at Z'=1.** `base` keeps acr_production's rule of",
        "`init_target_cp: 0.1` for Z'=1, so there the `std*` modes differ from it",
        "in two ways at once (start density and compression). At Z'>1 `base` is",
        "already `'std'` and the compression comparison is clean.", "",
        "| stem | sg | Z' | arms | samples | question |",
        "|---|---|---|---|---|---|"]
    for tag, sg, zp, n, s_, q, stem in manifest:
        kind = "seeds" if tag == "seed" else "nominal"
        lines.append(f"| `{stem}` | {sg} | {zp} | {n} | {s_:,} {kind} | {q} |")
    lines += ["",
              f"`save_trajs` is on for the first {TRAJ_ARMS} arms of each mode.",
              "", f"Seeded arms need `make_seeds.py --sg <sg> --zp <zp>` first, "
              f"and SEED_SHARDS ({SEED_SHARDS}) must equal its `--shards`.",
              ]
    Path('MANIFEST.md').write_text("\n".join(lines) + "\n")

    print(f"\nwrote {ind} arm configs + MANIFEST.md")
    per_combo = {}
    for tag, sg, zp, n, s_, q, stem in manifest:
        per_combo.setdefault((sg, zp), []).append((tag, n, s_))
    for (sg, zp), rows in per_combo.items():
        rand = sum(s_ for t, n, s_ in rows if t != 'seed')
        seeded = sum(s_ for t, n, s_ in rows if t == 'seed')
        n_rand = sum(n for t, n, _ in rows if t != 'seed')
        print(f"\n   sg{sg} Z'={zp}: {rand:,} nominal across {n_rand} arms"
              + (f", +{seeded:,} seeded" if seeded else ""))
        for t, n, s_ in rows:
            print(f"      {t:10s} {n:3d} arms  {s_:7,d} "
                  f"{'seeds' if t == 'seed' else 'nominal'}")
    total = sum(m[4] for m in manifest)
    print(f"\n   {ind} arms, {total:,} samples = {COMBO_TOTAL:,} nominal x "
          f"{len(COVERAGE_COMBOS)} combos + seeded probes")


if __name__ == '__main__':
    main()

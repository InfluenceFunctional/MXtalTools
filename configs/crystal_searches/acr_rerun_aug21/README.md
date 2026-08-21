# Acridine search battery — August 2026 re-run

```bash
python make_seeds.py --sg 14 --zp 2     # seed shards for the aug21seed block
python make_configs_acridine.py         # 256 arm configs + MANIFEST.md
python check_configs.py                 # asserts the arms are what MANIFEST claims
```

**256 arms, 100,000 nominal samples per SG/Z'** (plus 1,407 seeded). Predecessor
is [`../acr_production`](../acr_production) (`may_acridine_*`), which used the
same 100k per combo on a single mode.

**Nominal is not what lands.** acr_production asked for 100k per combo and wrote
**55-86k** (57 / 68 / 86 / 68 %): two of fifty arms produced no output at all in
each sg14 run, one chunk is zero bytes, and per-arm yield ranged from 18 to 1993
of 2000. Budget against what arrives. Unexplained and worth a look: sg14-Z'1's
median arm yielded 1094 of 2000 while sg9-Z'2's yielded 1942.

## Two questions, kept apart

**Coverage** — the landscape in each SG/Z′ that Nikos' 15 accepted structures
occupy. Two are effectively new: **sg5 (C2) Z′=2 was never searched**, and sg19
Z′=3 was searched but never collated. The `base` mode is byte-identical to
`acr_production`, so the two batteries stay directly comparable.

**Diagnosis** — why sg14-Z′2 missed *both* of its known forms. Across all 64,927
raw samples of the previous search, the closest thing to ACRDIN07 sits at RDF
distance 0.1428 and matches 11 of 20 molecules under COMPACK; ACRDIN06 the same.
Already excluded, so *not* varied here:

| ruled out | evidence |
|---|---|
| thinning artifact | raw 64,927 no closer than the thinned 12,624 |
| domain / Niggli exclusion | both forms have cell reduction penalty exactly 0.0; no latent dim outside the prior's range |
| parameterization error | all 7 known forms rebuild from their own latent params at 20/20 |
| pseudo-Z′=1 collapse | zero such samples in either Z′=2 search |
| wrong symmetry | standard `SYM_OPS`, `sym_mult` 4, no nonstandard settings |
| `init_target_cp` | sg14-Z′1 ran at 0.1 and found both forms; sg9-Z′2 ran at `'std'` and found both; sg14-Z′2 ran at `'std'` and found neither — it does not discriminate |

So the modes vary only what remains: the start density, the stage-1 compression,
the stage-2 descent, the optimizer itself, and whether the basins are reachable at
all.

## Modes

Four sampling modes **partition** 100,000 nominal samples per SG/Z′ — they do not
add to it — and all four run on all five combos (250 arms). `seed` sits outside
the budget.

| mode | arms/combo | samples | what it changes |
|---|---|---|---|
| `base` | 20 | 40,000 | nothing — the acr_production control |
| `stdnocomp` | 10 | 20,000 | `init_target_cp: 'std'`, stage-1 compression **0.0** |
| `lightcomp` | 10 | 20,000 | base start, stage-1 compression **0.25** |
| `lowlr` | 10 | 20,000 | stage 2 at lr **0.001**, eps 1e-6, 400 steps |
| `seed` | 6 | 1,687 seeds | starts *at* perturbed known basins (sg14-Z′2 so far) |

`stdnocomp` pairs "start near target density" with "don't compress", because
compression is what drags a *diffuse* start up to a physical density — switching
it off from a diffuse start tests nothing. `lightcomp` keeps the base start so the
compression axis is also available as a clean single-variable change.

### Confound at Z′=1

`base` keeps acr_production's rule of `init_target_cp: 0.1` for Z′=1, so at
sg14-Z′1 `stdnocomp` differs from it in *two* ways at once — start density and
compression. `lightcomp` is clean everywhere. At Z′>1 `base` is already `'std'`
and every comparison is single-variable. If you want a clean read at Z′=1, move
`base` to `'std'` there and accept losing byte-comparability with the last
battery.

## The seeded block

`make_seeds.py` builds perturbed copies of each known sg14-Z′2 form (ACRDIN07,
ACRDIN06) **and** Nikos' five accepted structures there — 7 anchors × (1
undisplaced + 6 displacement levels × 40) = 1687 seeds. Displacement uses
`log_noise_latent_parameters`, the same operator `new_calibrate_prior_noise` uses,
so magnitudes are in units the pipeline already speaks.

**The seeded arms keep the PRODUCTION optimizer, deliberately.** They exist to
explain why the production search missed these forms, so heating the learning rate
would measure a search nobody runs — and would confound "narrow basin" with
"optimizer too hot to settle". The displacement ladder is what stops every seed
relaxing back to the same place; it is the exploration. If everything returns even
from the top rung (10^-0.5), extend `LOG_NOISE` rather than the learning rate.

Genuine MC dynamics are *not* currently an option: `optimizer_func: 'mcmc'` exists
in `init_opt` but its path in `crystal_opt_utils.py` carries two `assert False`
statements (lines 190 and 262) and would crash on the first step. It would need
repair before it could be used.

Read the result as a radius of attraction:

- **returns from large displacement** → wide basin; the previous miss is genuinely
  surprising and points at coverage.
- **returns only from small displacement** → narrow basin; random starts rarely
  land it and more samples is the fix.
- **does not return even from zero displacement** → the optimizer cannot hold the
  structure it was handed. No amount of sampling would have reported it, and the
  schedule itself has to change.

The undisplaced row is the single most informative one, and it is why `None` is
first in `LOG_NOISE`.

## Things that will bite

**Seeds are sharded, one file per arm, `mol_seed: 0`.** Not cosmetic:
`init_samples_to_optim` clips `num_samples` to the seed count and then indexes
`arange(mol_seed*n, (mol_seed+1)*n)`, so a single shared seed file makes every arm
with `mol_seed > 0` index past the end. `SEED_SHARDS` in the generator must equal
`make_seeds.py --shards`; `check_configs.py` enforces both the distinct paths and
`mol_seed == 0`.

**Every mode has its own stem.** `load_search_chunks` globs `<run_name>_*.pt` and
keeps only all-digit tails, and `run_search` writes one file per arm as
`<run_name>.pt`. The arm index must therefore be the final component, and no stem
may be another stem followed by digits — otherwise one mode's chunks collate
into another's. `check_configs.py` asserts this.

**`save_trajs` is on for the first 2 arms of each mode only.**
`records['params']` is (steps, batch, n_params) per stage per batch — about 25 MB
each at `batch_size` 1000, so 100–200 MB per arm and 5–10 GB per combo if enabled
everywhere. Two arms per mode is enough to characterise trajectories. Raise
`TRAJ_ARMS` if you want the full picture and have the disk.

**The generator deletes previously generated arms before writing.** Regenerating
a smaller battery over a larger one used to leave stale arms behind, and a stale
arm is indistinguishable from a live one to the launcher.

**Validate each new landscape before reporting from it.** A search that cannot
recover its own known forms cannot judge anyone else's structures, and the failure
is silent — the query structures simply come out unmatched, which reads as a
statement about them. Run
`energy_sampling/eval/nikos_comparison/landscape_check.py --prior <name> --raw`
against the raw chunks, not the thinned prior.

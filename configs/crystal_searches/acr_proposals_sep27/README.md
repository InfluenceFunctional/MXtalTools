# acr_proposals_sep27: clean clamp vs wrap, and three proposal arms (2026-09-26)

Acridine, sg14 Z'=2, MACE. 9 jobs, each one GPU for under 20 h (6,000 relaxations took about 11 h in acr_wrap_sep26).

**Prerequisite.** This battery needs the `run_search` fixes: per-run intermediates discarded after each batch, a progress file for resume, and `dataset_index` on seeded outputs. `submit.sbatch` refuses to run without them. In acr_wrap_sep26, a retry after a step-0 out-of-memory error reloaded an earlier batch's saved walkers. As a result, 46–55% of relaxations repeated earlier starts, and seeded outputs landed in the wrong seed's slots.

*Jobs, their arms, and what each is compared with. All arms use the aug21 base schedule. Samples are relaxations.*

| job | arm (`run_name` stem) | boundary | starts | samples | compare with | question |
|---|---|---|---|---|---|---|
| 0 | `sep27rclamp` chunk 0 | clamp | random, seed 1e8 | 6000 | 1 | control |
| 1 | `sep27rwrap` chunk 0 | wrap | random, seed 1e8 | 6000 | 0 | Does wrap raise yield within 2 kT per unit of compute? Does it discover families faster? |
| 2 | `sep27rclamp` chunk 1 | clamp | random, seed 2e8 | 6000 | 3 | independent replicate |
| 3 | `sep27rwrap` chunk 1 | wrap | random, seed 2e8 | 6000 | 2 | independent replicate |
| 4 (a, b) | `sep27seedwrap` chunks 0, 1 | wrap | seed shards 0 (ACRDIN07) and 1 (ACRDIN06) | ~560 | 5 | the forms' return rate against displacement |
| 5 (a, b) | `sep27seedclamp` chunks 0, 1 | clamp | the same seeds | ~560 | 4 | control |
| 6 | `sep27hops` | wrap | kicked copies of 268 low-energy families (latent log-noise −1.0, −0.5; 8 each) | 4288 | 1, 3 | Do hops from known basins find new low-energy families faster than random starts? |
| 7 | `sep27dblkick` | wrap | 121 doubled Z'=1 families unkicked; the 102 latent-representable ones also kicked at log-noise −2.0/−1.5/−1.0 (11 each) | 3487 | 1, 3 | Does breaking a doubled structure's symmetry lead to new genuinely Z'=2 minima, or back to the parent? |
| 8 | `sep27eljstart` | wrap | the lowest-eLJ half (2128) of 4257 distinct eLJ minima from a local eLJ pre-search | 2128 | 1, 3 | Are cheap-energy minima better MACE starts than random ones? |

**Seeds.**
- The random-start seeds are 1e8 apart. Starts are drawn with seed `opt_seed + batch_idx * 10000`, and `batch_idx` counts attempts, so the two streams cannot overlap.
- The proposal seeds live in `seeds/` as compact parameter files (`.pth`: `*.pt` is Git LFS-tracked here and might not reach the cluster), made by `make_seeds.py`, a local provenance script with fixed torch seeds. Parents whose cell the latent cannot represent get no kicked copies: 29 hop families and 19 doubled families. For those, the latent clips a long a or c axis, so a "kick" would compress it by 20–30%. `submit.sbatch` expands each one with `build_seeds.py` into `DATA/sep27_seeds/`, rebuilding it if the committed file changed. The builder uses the search's own conformer and construction, and its round trip reproduces eLJ to 7e-7 relative.
- How the seeds were made:
  - hops and the doubled kicks: `MolCrystalData.log_noise_latent_parameters`, the operator the aug21 seeded ladder used;
  - families: the pooled acr_wrap_sep26 analysis clustering;
  - doubled set: `D:/crystal_datasets/acridine/doubled_zp1_sg14_2026-09-26.pt`;
  - eLJ starts: a local eLJ search with the same schedule, under wrap.

```bash
python make_configs.py   # regenerates the yamls + MANIFEST.md; asserts unique run names, no local paths, no umbrella
                         # keys, clamp/wrap pairs differ only in the boundary, non-overlapping seed streams, seed files present
sbatch submit.sbatch     # array 0-8
```

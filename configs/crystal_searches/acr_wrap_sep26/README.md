# acr_wrap_sep26 — clamp vs wrap centroid boundary, 8 test jobs (2026-09-26)

Acridine, sg14 Z'=2, MACE. Each job is one A100 GPU for under 20 h. The battery tests whether the opt-in `centroid_boundary: wrap` (mxtaltools `4d13543b`) changes what the search finds, compared with the default asymmetric-unit clamp, under the same code. That code includes the `no_grad` memory fix `c7204ced`. The aug21 outputs are not the control: they predate the third monoclinic reduction wall and that fix.

*Jobs, their arms, and the comparison each serves. Samples are relaxations. Paired arms share `opt_seed`. They draw starts per batch, so pairing is start-by-start only while batch sizes match; under the out-of-memory back-off it is statistical.*

| job | arm (run_name stem) | boundary | samples | compare with | question |
|---|---|---|---|---|---|
| 0 | `sep26rclamp` chunk 0 | clamp | 6000 | 1 | control |
| 1 | `sep26rwrap` chunk 0 | wrap | 6000 | 0 | Does wrap change the in-band fraction (1–3 kT above the floor), the families per band, the floor, face occupancy, and whether the forms appear? |
| 2 | `sep26rclamp` chunk 1 | clamp | 6000 | 3 | replicate of 0 with a second seed |
| 3 | `sep26rwrap` chunk 1 | wrap | 6000 | 2 | replicate of 1 |
| 4 | `sep26rwrapnr` | wrap, `enforce_reduced: false` | 6000 | 1 | Is the β = 90.57° reduction wall a second trap? |
| 5 | `sep26rwrapl2` | wrap, stage-2 cap 400 | 4000 | 1 | What does letting cut-off walkers finish buy? |
| 6 (a, b) | `sep26seedwrap` chunks 0, 1 | wrap | seed shards 0 (ACRDIN07), 1 (ACRDIN06) | 7 | Forms' return rate against displacement, 6 noise levels × 40 seeds |
| 7 (a, b) | `sep26seedclamp` chunks 0, 1 | clamp | the same shards | 6 | control |

**Common settings.** These are the aug21 `base` schedule: stage 1 rprop at lr 0.05 for up to 350 steps, compression 1.0, `enforce_reduced`; stage 2 at lr 0.01 annealed for up to 150 steps. Differences from aug21:
- `batch_size: 48` with `grow_batch_size: true`, so the out-of-memory back-off (×0.9) can climb back (×1.2 per set);
- no umbrella keys;
- the boundary is set on every stage.

Trajectories are saved for jobs 0, 1, 6 and 7.

```bash
python make_configs.py   # regenerates the yamls + MANIFEST.md and asserts: unique run names, no local paths,
                         # no umbrella keys, clamp/wrap pairs differ only in the boundary
sbatch submit.sbatch     # array 0-7
```

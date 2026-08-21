# acridine re-run battery, August 2026

280 arms. Every stem is distinct and `load_search_chunks` keeps only
all-digit tails, so modes cannot bleed into one another at collation.

The five sampling modes PARTITION 100,000 nominal samples per
SG/Z' -- they do not add to it. `seed` is a targeted probe on top, sized
by the seed file rather than by arm count.

**Nominal is not what lands.** acr_production asked for 100k per combo
and wrote 55-86k (57 / 68 / 86 / 68 %): two of fifty arms produced no
output at all in each sg14 run, one chunk is zero bytes, and per-arm
yield ranged from 18 to 1993 of 2000. Count what arrives.

**Confound at Z'=1.** `base` keeps acr_production's rule of
`init_target_cp: 0.1` for Z'=1, so there the `std*` modes differ from it
in two ways at once (start density and compression). At Z'>1 `base` is
already `'std'` and the compression comparison is clean.

| stem | sg | Z' | arms | samples | question |
|---|---|---|---|---|---|
| `aug21base_acridine_sg14_zp1` | 14 | 1 | 20 | 40,000 nominal | unchanged acr_production schedule -- the control |
| `aug21stdnocomp_acridine_sg14_zp1` | 14 | 1 | 10 | 20,000 nominal | start near target density, no stage-1 compression |
| `aug21lightcomp_acridine_sg14_zp1` | 14 | 1 | 10 | 20,000 nominal | the BASE start with light compression (0.25 vs 1.0) -- a single-variable change off the control |
| `aug21lowlr_acridine_sg14_zp1` | 14 | 1 | 10 | 20,000 nominal | finer stage-2 descent: lr 0.001, tighter convergence, 400 steps |
| `aug21seed_acridine_sg14_zp1` | 14 | 1 | 6 | 1,205 seeds | are known basins reachable and HOLDABLE at all? starts AT perturbed copies of the known forms and his structures. NOT part of COMBO_TOTAL -- a targeted probe, not random sampling |
| `aug21base_acridine_sg14_zp2` | 14 | 2 | 20 | 40,000 nominal | unchanged acr_production schedule -- the control |
| `aug21stdnocomp_acridine_sg14_zp2` | 14 | 2 | 10 | 20,000 nominal | start near target density, no stage-1 compression |
| `aug21lightcomp_acridine_sg14_zp2` | 14 | 2 | 10 | 20,000 nominal | the BASE start with light compression (0.25 vs 1.0) -- a single-variable change off the control |
| `aug21lowlr_acridine_sg14_zp2` | 14 | 2 | 10 | 20,000 nominal | finer stage-2 descent: lr 0.001, tighter convergence, 400 steps |
| `aug21seed_acridine_sg14_zp2` | 14 | 2 | 6 | 1,687 seeds | are known basins reachable and HOLDABLE at all? starts AT perturbed copies of the known forms and his structures. NOT part of COMBO_TOTAL -- a targeted probe, not random sampling |
| `aug21base_acridine_sg9_zp2` | 9 | 2 | 20 | 40,000 nominal | unchanged acr_production schedule -- the control |
| `aug21stdnocomp_acridine_sg9_zp2` | 9 | 2 | 10 | 20,000 nominal | start near target density, no stage-1 compression |
| `aug21lightcomp_acridine_sg9_zp2` | 9 | 2 | 10 | 20,000 nominal | the BASE start with light compression (0.25 vs 1.0) -- a single-variable change off the control |
| `aug21lowlr_acridine_sg9_zp2` | 9 | 2 | 10 | 20,000 nominal | finer stage-2 descent: lr 0.001, tighter convergence, 400 steps |
| `aug21seed_acridine_sg9_zp2` | 9 | 2 | 6 | 1,687 seeds | are known basins reachable and HOLDABLE at all? starts AT perturbed copies of the known forms and his structures. NOT part of COMBO_TOTAL -- a targeted probe, not random sampling |
| `aug21base_acridine_sg5_zp2` | 5 | 2 | 20 | 40,000 nominal | unchanged acr_production schedule -- the control |
| `aug21stdnocomp_acridine_sg5_zp2` | 5 | 2 | 10 | 20,000 nominal | start near target density, no stage-1 compression |
| `aug21lightcomp_acridine_sg5_zp2` | 5 | 2 | 10 | 20,000 nominal | the BASE start with light compression (0.25 vs 1.0) -- a single-variable change off the control |
| `aug21lowlr_acridine_sg5_zp2` | 5 | 2 | 10 | 20,000 nominal | finer stage-2 descent: lr 0.001, tighter convergence, 400 steps |
| `aug21seed_acridine_sg5_zp2` | 5 | 2 | 6 | 241 seeds | are known basins reachable and HOLDABLE at all? starts AT perturbed copies of the known forms and his structures. NOT part of COMBO_TOTAL -- a targeted probe, not random sampling |
| `aug21base_acridine_sg19_zp3` | 19 | 3 | 20 | 40,000 nominal | unchanged acr_production schedule -- the control |
| `aug21stdnocomp_acridine_sg19_zp3` | 19 | 3 | 10 | 20,000 nominal | start near target density, no stage-1 compression |
| `aug21lightcomp_acridine_sg19_zp3` | 19 | 3 | 10 | 20,000 nominal | the BASE start with light compression (0.25 vs 1.0) -- a single-variable change off the control |
| `aug21lowlr_acridine_sg19_zp3` | 19 | 3 | 10 | 20,000 nominal | finer stage-2 descent: lr 0.001, tighter convergence, 400 steps |
| `aug21seed_acridine_sg19_zp3` | 19 | 3 | 6 | 482 seeds | are known basins reachable and HOLDABLE at all? starts AT perturbed copies of the known forms and his structures. NOT part of COMBO_TOTAL -- a targeted probe, not random sampling |

`save_trajs` is on for the first 2 arms of each mode.

Seeded arms need `make_seeds.py --sg <sg> --zp <zp>` first, and SEED_SHARDS (6) must equal its `--shards`.

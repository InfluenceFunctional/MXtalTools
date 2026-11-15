# pool8sg_nov15 / control1000

What do the battery's 100 starts per molecule miss? 1,000 starts per molecule on pool8 chunks 1287 and 1571 (50
molecules: 38 train, 7 heldout_random, 5 heldout_scaffold; pool8_nov09's reference panel) in all eight groups
(2, 14, 15, 19, 4, 61, 33, 9), with the battery's settings otherwise (`make_control.py`, built on
`../make_battery.py` `task_cfg`). The first 100 starts of each molecule are, statistically, a 100-start search, so one
run per (molecule, group) gives both: is the 1,000-start lowest minimum among the first 100's minima, how far above it
the best of 100 lies, and what share of the minima within the band window 100 starts reach.

- 16 one-core array jobs (`submit_control1000.sbatch`, `--array=0-15`), 12 h and 24 GB each; no band step, the slims
  are the result (about 16 KB per start, so about 0.4 GB per task, 7 GB in all).
- Cost: about 4 h per task on the cluster (0.37 s/start on the dev box, 0.6 s/start on the cluster for pool8_nov09),
  so about 70 core-hours.
- Same frame guard as the battery (blob `ff4e836e`, clean `mxtaltools/`); seeds 1e12 + g x 1e10 + j x 1e7.
- Outputs: `/scratch/mk8347/data/crystal_datasets/conditional/pool8sg_nov15/control1000/search/sg<g>/pool8ctl<g>_c<k>_slim.pt`.

**Local preview (2026-11-15, dev box, C2/c only, 3 molecules of chunk 0, 1,000 starts each).** In the slim's own
energy unit (kT_s = 6.86 raw eLJ), per molecule:

| molecule (chunk 0 order) | minima, 1,000 starts | minima, first 100 starts | best of first 100 above best of 1,000 (kT_s) | minima within 5 kT_s of the best | of those, reached by the first 100 |
|---|---|---|---|---|---|
| 0 | 321 | 63 | 1.94 | 122 | 21 |
| 1 | 467 | 81 | 0.47 | 368 | 60 |
| 2 | 182 | 32 | 1.44 | 59 | 6 |

On these three molecules 100 starts never found the 1,000-start lowest minimum, and reached 10 to 17% of the minima
in the 5 kT_s window. Three molecules in one group are not a measurement of the battery; this control is.

    cd /scratch/mk8347/projects/gfn_cond/MXtalTools/configs/crystal_searches/pool8sg_nov15/control1000
    sbatch submit_control1000.sbatch

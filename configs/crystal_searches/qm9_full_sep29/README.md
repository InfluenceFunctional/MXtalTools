# qm9_full_sep29

The search that builds the full-QM9 conditional prior: random-start eLJ searches, SG2 (P-1), Z'=1, on all 130,310
standardized QM9 molecules. Chunks 0-49 get 20 starts per molecule and chunks 50-204 get 10. Chunk 0 also gets 180
more, which measure how often n starts miss a molecule's lowest minimum.

| tasks | chunks | molecules | seeds | random starts per molecule |
|---|---|---|---|---|
| 0 | 0 | 195 | 0 | 20 |
| 1-3 | 0 | 195 | 1-3 | +60 each, 200 in total |
| 4-52 | 1-49 | 9,555 | 0 | 20 |
| 53-207 | 50-204 | 120,560 | 0 | 10 |

*Chunks 0-49 hold 195 molecules each and are the set the current conditional prior was built from, which has 10 starts
per molecule on the August schedule (`qm9_anchors`, tag `qm9c100k`). Chunks 50-204 hold 780 each (440 in chunk 204):
the rest of the pool. Task i >= 4 runs chunk i - 3.*

1,435,700 relaxations in all, about 33 GPU-hours at the measured 12 per second. The longest task is 11,700
relaxations (tasks 1-3), about 15 minutes.

## Measured (tasks 0-52 that ran, 2026-09-30)

- Throughput: 12 relaxations per second per GPU, pooled over 42 tasks on A100, L40S and H200 nodes, from job start to
  end (container start included). A 3,900-relaxation task takes about 5.6 minutes.
- Every crystal is finite, bound (eLJ < 0) and well-defined. A third of the cells have an angle outside 60-120 degrees,
  as expected without a reduction wall.
- How many starts are enough, from chunk 0's 195 molecules at 200 starts each, exact over every n-start subset
  ("best" = lowest of the 200; 1 kT = 6.9 raw eLJ units at the current training temperature):

| starts per molecule | chance of a structure within 1 kT of the best | within 3 kT | shortfall of the lowest from the best, median (kT) | 90th percentile (kT) |
|---|---|---|---|---|
| 5 | 29% | 71% | 2.05 | 3.44 |
| 10 | 47% | 87% | 1.24 | 2.30 |
| 20 | 68% | 96% | 0.68 | 1.53 |
| 50 | 89% | 99.5% | 0.21 | 0.74 |

- Against the current prior on its own molecules (4,875 of chunks 0-29), the median new minimum is 1.54 kT lower at
  10 starts and 2.07 kT lower at 20.
- Starts rarely coincide: 19 of a molecule's 20 end at distinct energies (0.1 raw-unit tolerance; a proxy, not a
  structural comparison).

## Molecules

One chunk family, `qm9_cluster_mols_chunk<k>.pt`. Chunks 50-204 and their combined file `qm9_cluster_mols_rest.pt` come
from gfn-diffusion `energy_sampling/prep_qm9_anchor_mols.py`, run on the dev box:

```
python prep_qm9_anchor_mols.py --n-mols 0 --exclude D:\crystal_datasets\conditional\priors\qm9_cluster_mols.pt --chunk-size 780 --chunk-start 50 --chunk-stem qm9_cluster_mols --out D:\crystal_datasets\conditional\priors\qm9_cluster_mols_rest.pt
```

- The pool is `csd_free_qm9_dataset.pt`, 133,728 molecules. The 9,750 of chunks 0-49 are excluded by identifier, and no
  remaining molecule shares a SMILES with them.
- 3,418 of the remaining 123,978 molecules fail the standardization fixed-point check and are dropped. That leaves
  120,560, shuffled with seed 0, so any run of chunks is a random sample.
- Rerunning the command reproduces the files. The script refuses to replace a chunk file that holds other molecules.

## Schedule

`acr_finish_sep27/0.yaml` with `optim_target: elj`: stage 1 Rprop at lr 0.05 with compression (max 350 steps), stage
2 Rprop at lr 0.01 with annealing (max 150 steps).

- `centroid_boundary: wrap`, no reduction wall: outputs are not in the reduced cell; run `standardize_cells` on them.
- `convergence_eps` x10, as in `zp1_sep28`, for the MXtalTools fe283818 EMA change.
- `init_target_cp: std`, `init_reduced: true`, `batch_size: 2000` with `grow_batch_size` and `oom_ceiling`.
- No `early_stop` cascade. Its reference is the lowest energy of the whole run, and one run holds up to 780 molecules.
- Energy: raw eLJ (the searcher does not stamp `lj_coeff`).

## Files

- `make_battery.py` writes `tasks/<task>.yaml`, `INDEX.tsv` (chunk, seed, molecules, starts and relaxations per task),
  and the array range in the job script. It asserts distinct run names and seeds, seed slots that no task can overrun,
  and that no path is local.
- `submit_qm9_full.sbatch`: array 0-207, at most 16 at once, 4 h walltime, `USR1` 10 min before the end. Resubmitting
  resumes. A task whose molecule file is missing exits at once.
- To add starts later, append tasks to `TASKS` (a new seed on the chunks that need it) and submit only the new indices.
  Never reorder: an array index names a task.

## Launch (cluster)

The molecule files must be on the cluster first. Copy them with Globus from the dev box's
`D:\crystal_datasets\conditional\priors\` to `/scratch/mk8347/data/crystal_datasets/conditional/priors/`: chunks 0-49
are already there, and chunks 50-204 need `qm9_cluster_mols_chunk50.pt` to `qm9_cluster_mols_chunk204.pt`, plus
`qm9_cluster_mols_rest.pt` for the prior build.

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools && git pull
```

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools/configs/crystal_searches/qm9_full_sep29 && sbatch submit_qm9_full.sbatch
```

## Runs

- 2026-09-30, job 18835847 (all 208 tasks, 20 starts on every chunk as then configured): 42 of tasks 0-52 finished
  (187,200 crystals, 7,605 molecules). Tasks 20, 24, 30-36, 38 and 39 (chunks 17, 21, 27-33, 35, 36) never started.
  Tasks 53-207 exited at the molecule-file check because chunks 50-204 had not been uploaded; they wrote nothing, so
  they were changed to 10 starts before running.
- Resubmission of what did not run: `scancel 18835847`, then `sbatch --array=20,24,30-36,38,39,53-207%16
  submit_qm9_full.sbatch`.

## Outputs

`/scratch/mk8347/data/crystal_datasets/conditional/anchors/qm9_full_sep29/qm9full_c<k>_<seed>.pt` (lists of
`MolCrystalData`, about 3 KB per crystal), with `_progress.json` beside each and the rendered `<run>.yaml`. One chunk's
seeds gather with gfn-diffusion `energy_sampling/data_processing/utils.py::load_search_chunks(<dir>, 'qm9full_c<k>')`.

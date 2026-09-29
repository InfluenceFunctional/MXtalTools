# qm9_full_sep29

The search that builds the full-QM9 conditional prior: random-start eLJ searches, SG2 (P-1), Z'=1, on all 130,310
standardized QM9 molecules. Each molecule gets 20 starts; chunk 0 also gets 180 more, which are there to measure how
often 20 starts miss a molecule's lowest minimum.

| tasks | chunks | molecules | seeds | random starts per molecule |
|---|---|---|---|---|
| 0 | 0 | 195 | 0 | 20 |
| 1-3 | 0 | 195 | 1-3 | +60 each, 200 in total |
| 4-52 | 1-49 | 9,555 | 0 | 20 |
| 53-207 | 50-204 | 120,560 | 0 | 20 |

*Chunks 0-49 hold 195 molecules each and are the set the current conditional prior was built from, which has 10 starts
per molecule on the August schedule (`qm9_anchors`, tag `qm9c100k`). Chunks 50-204 hold 780 each (440 in chunk 204):
the rest of the pool. Task i >= 4 runs chunk i - 3.*

2,641,300 relaxations. At 3 per second, the rate `qm9_anchors` measured in August on the old schedule at batch 1000,
that is about 245 GPU-hours. The longest task is 15,600 relaxations, about 1.5 h at that rate, so it fits the 4 h
walltime. The rate on this schedule and batch size has not been measured yet.

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
- `submit_qm9_full.sbatch`: array 0-207, at most 32 at once, 4 h walltime, `USR1` 10 min before the end. Resubmitting
  resumes. A task whose molecule file is missing exits at once.
- To add starts later, append tasks to `TASKS` (a new seed on the chunks that need it) and submit only the new indices.
  Never reorder: an array index names a task.

## Launch (cluster)

Upload the new molecule files from the dev box (155 chunks of 2.4 MB, and the 375 MB combined file that a prior build
embeds):

```bash
cd /d/crystal_datasets/conditional/priors && scp qm9_cluster_mols_rest.pt qm9_cluster_mols_chunk{50..204}.pt mk8347@login.torch.hpc.nyu.edu:/scratch/mk8347/data/crystal_datasets/conditional/priors/
```

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools && git pull
```

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools/configs/crystal_searches/qm9_full_sep29 && sbatch submit_qm9_full.sbatch
```

## Outputs

`/scratch/mk8347/data/crystal_datasets/conditional/anchors/qm9_full_sep29/qm9full_c<k>_<seed>.pt` (lists of
`MolCrystalData`, about 8 GB in all), with `_progress.json` and `_owner.json` beside each and the rendered `<run>.yaml`.
One chunk's seeds gather with gfn-diffusion `energy_sampling/data_processing/utils.py::load_search_chunks(<dir>,
'qm9full_c<k>')`.

## Read first

- **Throughput:** the first finished tasks give the real rate, and therefore the real cost of the whole battery.
- **Saturation (tasks 0-3):** for each chunk 0 molecule, subsample its 200 starts and see how the lowest energy falls
  with the number of starts. Top up the other chunks only if 20 starts often miss the 200-start minimum by more than
  about 1 kT (6.9 raw eLJ units at the current training temperature).

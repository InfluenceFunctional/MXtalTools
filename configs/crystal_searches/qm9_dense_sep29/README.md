# qm9_dense_sep29

Pilot of the full-QM9 prior search: dense random-start eLJ searches, SG2 (P-1), Z'=1, on the first four chunks of
the standardized QM9 molecule set the current conditional prior was built from (195 molecules per chunk).

| tasks | chunk | seeds | starts per molecule | molecules |
|---|---|---|---|---|
| 0-3 | 0-3 | 0 | 50 | 780 |
| 4-6 | 0 | 1-3 | +150 (200 in total on chunk 0) | 195 |

Each task: `run_search.py`, `sampling_mode: all`, 50 random starts for every molecule of its chunk file, 9,750
relaxations. The same molecules have 10 random starts each in the current prior (`qm9_anchors`, tag `qm9c100k`).

## Schedule

`acr_finish_sep27/0.yaml` with `optim_target: elj`: stage 1 Rprop at lr 0.05 with compression (max 350 steps), stage
2 Rprop at lr 0.01 with annealing (max 150 steps).

- `centroid_boundary: wrap`, no reduction wall: outputs are not in the reduced cell; run `standardize_cells` on them.
- `convergence_eps` x10, as in `zp1_sep28`, for the MXtalTools fe283818 EMA change.
- `init_target_cp: std`, `init_reduced: true`, `batch_size: 2000` with `grow_batch_size` and `oom_ceiling`.
- No `early_stop` cascade. Its reference is the lowest energy of the whole run, and one run holds 195 molecules.
- Energy: raw eLJ (the searcher does not stamp `lj_coeff`).

## Files

- `make_battery.py`: writes `tasks/<task>.yaml` and `INDEX.tsv`. It asserts distinct run names and seeds, and that no
  path is local.
- `submit_qm9_dense.sbatch`: array 0-6, 4 h walltime, `USR1` 10 min before the end. Resubmitting resumes.

## Launch (cluster)

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools && git pull
```

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools/configs/crystal_searches/qm9_dense_sep29 && sbatch submit_qm9_dense.sbatch
```

## Outputs

`/scratch/mk8347/data/crystal_datasets/conditional/anchors/qm9_dense_sep29/qm9dense_c<k>_<seed>.pt` (lists of
`MolCrystalData`), with `_progress.json` and `_owner.json` beside each and the rendered `<run>.yaml`. One chunk's seeds
gather with `data_processing/utils.py::load_search_chunks('qm9dense_c<k>')`.

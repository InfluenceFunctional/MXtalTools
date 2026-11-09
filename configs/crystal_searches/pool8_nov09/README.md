# pool8_nov09

Streaming eLJ searches in the endpoint (ball) mode, P-1 (sg 2), Z' = 1, one condition per molecule of GFN training
pool v1 with at most 8 heavy atoms: 72,464 conditions, 100 random starts each (7.2M starts). CPU only. Built and
launched for the first size rung of the conditional prior (owner, 2026-11-09).

## Molecules

From gfn-diffusion `energy_sampling` (branch `conditional`), on the dev box:

| Step | Kept | Dropped |
|---|---|---|
| pool v1 keys with at most 8 heavy atoms (InChIKey -SRel -FixedH: mirror images merged, diastereomers, E/Z isomers and tautomers separate; Day's skeletons already excluded) | 90,685 | |
| one MMFF94 conformer per key (`data_processing/training_pool_conformers.py 8 <dir>`: 8 ETKDGv3 embeddings seeded by the key, lowest converged MMFF energy, stereochemistry checked against the SMILES) | 75,572 | 13,344 no embedding (11,344 of them enumerated stereoisomers that small fused rings cannot close), 1,645 stereo changed under MMFF, 124 no MMFF parameters |
| standard frame (`prep_qm9_anchor_mols.py --src pool8_all.pt --n-mols 0 --chunk-size 25 --chunk-stem pool8_mols`: the `orient_molecule(mode='std')` fixed point) | 72,464 | 3,108 that move under a second standardization (near-degenerate frames) |

The kept molecules are shuffled (seed 0) into 2,899 chunks of 25 (the last 14), `pool8_mols_chunk<k>.pt`. Identifier
= pool key; the pool split (train / heldout_random / heldout_scaffold) is in `pool8_mols.tsv` beside the molecule
files (dev box: `D:\crystal_datasets\conditional\pool8_nov09\mols\`), with the failure list `pool8_failed.tsv`.

## Search

`minima_stream/mipcas_elj_balls.yaml` with: device cpu, 50 slots (= batch), 100 random starts per molecule, no stop
rule, `basin_set: false` (no basin or band set: the prior constructor derives them from the trajectories), every
descent state recorded (`stream.trajectories`). One task = one chunk with all of its molecules' starts, so every
condition's minima library is complete inside its task. Task k: `opt_seed` = 2e11 + k x 1e6 (the stream's block
seeds stay inside 1e6).

At the end of a task, `slim_outputs.py` writes `<run>_slim.pt` (format `stream_slim/1`; contents in its docstring:
trajectories in float32 with every field the prior constructor reads, the minima's identity descriptions, the
conditions, molecules, energy/search/identity records and the run config) and deletes `<run>.pt`, `_minima.pt` and
`_traj.pt`. A task with a `_slim.pt` is done.

Measured on the dev box (Intel Core Ultra 9 275HX, tasks 0 and 1 run side by side through the job script, 2026-11-09):
each task 2,500 starts in about 20 min on one core, peak 2.7 GB, 320k recorded states, about 45 minima per molecule;
`_slim.pt` 43 MB per task (17 KB per start; 110 MB of sources deleted). Whole battery: about 1,000 core-hours on that
CPU (cluster cores may be slower) and about 125 GB of `_slim.pt` files.

## Files

- `make_battery.py`: writes `tasks/<k>.yaml`, `INDEX.tsv` and the job script's array range; asserts distinct run
  names and seeds, seed slots no task can overrun, and no local path.
- `submit_pool8_nov09.sbatch`: one array job = 8 tasks side by side, one core each (32 GB per job); job j runs tasks
  8j..8j+7; a task with a `_slim.pt` is skipped, so resubmitting the array finishes what is left. Each task logs to
  `<out_dir>/<run>.log`; the job log prints one status line per task.
- `slim_outputs.py`: the converter.

## Launch (cluster)

1. Upload `D:\crystal_datasets\conditional\pool8_nov09\mols\chunks\pool8_mols_chunk*.pt` (2,899 files, ~170 MB) to
   `/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/mols/`.
2. Pull MXtalTools, smoke job 0 (tasks 0-7), then the rest behind it.

Outputs: `/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/search/pool8_c<k>_slim.pt`.
Done when 2,899 `_slim.pt` files exist: `ls <out>/*_slim.pt | wc -l`.

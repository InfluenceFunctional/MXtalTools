# rc_oct22

Random-start eLJ searches, SG2 (P-1), Z'=1, on molecules whose geometry is **one random conformer per SMILES** in
place of a DFT geometry, for a larger conditional prior. **CPU only: one core per task.** One chunk family per molecule
source; each family's list is shuffled, so any run of its chunks is a random sample of it, and **the share of each
source is chosen at submission by how many of its chunks run**.

| family | molecules from | tasks (array indices) | chunks | molecules | relaxations | core-hours |
|---|---|---|---|---|---|---|
| qm9rc | the qm9full SMILES | 0-329 | 66 | 127,744 | 1,277,440 | 473 |
| csdrc | CSD | 330-469 | 28 | 55,099 | 550,990 | 204 |
| zincrc | ZINC20, MW row A | 470-929 | 92 | 182,419 | 1,824,190 | 676 |
| gdbrc | GDB-11 | 930-3314 | 477 | 950,076 | 9,500,760 | 3,519 |
| pubrc | PubChem | 3315-4734 | 284 | 553,570 | 5,535,700 | 2,050 |
| all | | 0-4734 | 947 | 1,868,908 | 18,689,080 | 6,922 |

*Task (family's first index in the table) + 5k + s runs seed s (0-4) of chunk k of that family: 2 random starts for every molecule of the chunk (2,000
molecules; the last chunk of each 20,000-molecule shard holds the remainder), so 4,000 relaxations at most, about 1.5 h
on one core. Core-hours at 0.75 relaxations per second per core, measured on the dev box (one core, batch 50, 200
relaxations of qm9rc chunk 0); the cluster's cores may be slower.*

## Molecules

Built on the dev box with gfn-diffusion `energy_sampling` (branch `conditional`), in four steps.

1. **SMILES pools** (`data_processing/smiles_pool.py`): neutral H/C/N/O/F components of CSD, ZINC20 (MW row A), GDB-11
   and PubChem, each molecule of 4-9 heavy atoms taken whole and every larger one (up to 40 heavy atoms) cut at single
   non-ring bonds into every connected union of pieces of 4-9 heavy atoms, at most 32 per parent, H-capped. Pools:
   `D:\crystal_datasets\smiles_sources\pool\<source>\pool.tsv`.
2. **Selection** (`data_processing/smiles_mixture.py`): every QM9 molecule excluded (146,946 canonical stereo-free
   SMILES: GDB and relaxed SMILES of the QM9 xyz files plus the qm9full identifiers); sources drawn in the order csd,
   zinc20A, gdb11, pubchem, each molecule once; a fixed 5% held-out side per SMILES (salted hash) in the `split`
   column. `qm9rc` is the qm9full SMILES with the prior's split (`selection\qm9_oct22\selected.tsv`).
   Commands: `python -m data_processing.smiles_mixture --qm9 --out D:\crystal_datasets\smiles_sources\selection\qm9_oct22`
   and `python -m data_processing.smiles_mixture --pool D:\crystal_datasets\smiles_sources\pool --quota csd=all
   zinc20A=all gdb11=all pubchem=1000000 --out D:\crystal_datasets\smiles_sources\selection\new_oct22` (the PubChem
   quota exceeds what is left of its pool, so every source is taken whole). Each family's list is
   `selection\new_oct22\by_source\<source>.tsv`.
3. **Conformers** (`data_processing/random_conformers.py`): ETKDGv3 seeded by the SMILES, every single non-ring rotor
   spun uniformly, redrawn up to 20 times until it passes the validity checks (no clash, bonds intact, stereo and
   double bonds as embedded); no force-field relief. Failures are listed per shard in
   `D:\crystal_datasets\conditional\rc_oct22\raw\<family>_failed_shard<s>.tsv`.
4. **Chunks** (`prep_qm9_anchor_mols.py`, per raw shard s: `--n-mols 0 --chunk-size 2000 --chunk-start 10s
   --chunk-stem <family>_mols`): standardized to the `orient_molecule(mode='std')` fixed point; about 0.3% that move
   under a second standardization are dropped. `make_battery.py --scan` writes `chunks.tsv` from the files.

| family | SMILES selected | valid conformers | conformer failures (%) | molecules in chunks |
|---|---|---|---|---|
| qm9rc | 130,229 | 128,157 | 1.6 | 127,744 |
| csdrc | 55,312 | 55,276 | 0.07 | 55,099 |
| zincrc | 183,822 | 182,774 | 0.6 | 182,419 |
| gdbrc | 953,942 | 952,231 | 0.2 | 950,076 |
| pubrc | 567,029 | 554,963 | 2.1 | 553,570 |

*Failures are mostly embedding failures of strained polycyclic cages (qm9rc: 1,736 of its 2,072; a second embedding
seed or more ETKDG iterations recovers only about 10-20% of them, so they are left out). The last column is after the
fixed-point check.*

Identifiers are `<canonical SMILES>_c0`; `smiles` is the canonical stereo-free SMILES (the key any split groups on).

## Schedule

The relaxation schedule is qm9_full_sep29's: `acr_finish_sep27/0.yaml` with `optim_target: elj`, two Rprop stages
(compression, then annealing), `centroid_boundary: wrap`, no reduction wall (outputs are not in the reduced cell),
`convergence_eps` x10, no `early_stop` cascade, raw eLJ, `init_target_cp: std`, `init_reduced: true`. What differs is
where it runs: `device: cpu`, a fixed `batch_size: 50` (no `grow_batch_size`, no `oom_ceiling`), and 2 starts per task
with 5 seeds per chunk in place of 10 starts in one task.

Why batch 50 on one core: the eLJ route writes out every crystal's image cluster (about 12,100 atoms per crystal on
QM9-sized molecules, of which about 6,200 atom pairs fall inside the 10 A cutoff), so a task's memory is about 0.8 GB
plus 16-20 MB per crystal in the batch (measured: 1.6 GB at batch 50, 4.8 GB at batch 200); extra threads per task
barely help (4 threads 1.0 relaxations per s against 0.58 for one, batch 200). A task's batches draw their seeds from a
slot of 100 batches, so a task holds at most 100 x 50 = 5,000 relaxations: hence 2 starts per task.

## Files

- `make_battery.py` writes `tasks/<task>.yaml`, `INDEX.tsv` (family, chunk, seed, molecules, starts, relaxations,
  run name, opt_seed, molecule path per task) and the array line of the job script; `--scan <dir>` first writes
  `chunks.tsv` from the chunk files. It asserts distinct run names and seeds, seed slots no task can overrun, no local
  path, a chunk for every family (`--allow-missing` to override), and that a rewrite keeps every existing `INDEX.tsv`
  row (`--renumber` to override, only before anything is submitted): **an array index names a task**.
- `submit_rc_oct22.sbatch`: CPU only (1 core, 3 GB, no GPU, `OMP_NUM_THREADS=1`); task = array index + `TASK_OFFSET`
  (default 0), at most 250 at once per submitted array, 4 h walltime, `USR1` 10 min before the end. Resubmitting
  resumes; a finished task exits at once; a task whose molecule file is missing exits at once. Each log prints the CPU
  model on a `DEVICE` line.
- `check_battery.py`: every task's state from the progress files and the newest log of each task (matched by the
  `task=` the log prints), crystal counts, output size and throughput; standard library only (`python3 check_battery.py`
  on the login node). `--rerun 0-1429,3315-3814` (the submitted ranges) adds the sbatch lines that resubmit only the
  tasks in them that are not done, once no rcoct22 job is queued; resubmitted tasks resume from their progress files.
  Installment 1 lost about 50 tasks to cluster faults (srun allocation timeouts while ~1,900 tasks started at once,
  node failures), none to the code.

Known behaviour, inherited from run_search: the molecules x starts list is shuffled with an unseeded generator, so a
task that is stopped and resumed relaxes a different permutation for its remaining starts, and per-molecule start
counts in that task scatter around 2. A task is sized to finish well inside its 4 h walltime, so this only arises after
a node failure or preemption.

## Launch (cluster)

1. Upload the chunk files of the installment with Globus, from the dev box's
   `D:\crystal_datasets\conditional\rc_oct22\mols\` to
   `/scratch/mk8347/data/crystal_datasets/conditional/rc_oct22/mols/`: only `<family>_mols_chunk<k>.pt` for the chunks
   to be run (about 4.7 MB each, 4.3 GB for all 947); the combined `<family>_mols_shard<s>.pt` files are for the prior
   build and can wait. (Done 2026-10-23 for installment 1: qm9rc, csdrc and zincrc whole, gdbrc and pubrc chunks 0-99.)
2. Pull, print the array limit, smoke one small task (task 325: qm9rc chunk 65, seed 0, 38 molecules x 2 starts,
   about 2 minutes), and queue installment 1 behind it: all of qm9rc, csdrc and zincrc plus the first 100 chunks of
   gdbrc and of pubrc (1,930 tasks, 759,631 molecules, about 2,800 core-hours). Three arrays, each index below 1,000,
   each at most 250 at once:

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools && git pull && scontrol show config | grep -i MaxArraySize && cd configs/crystal_searches/rc_oct22 && [ -z "$(squeue -h -u $USER -n rcoct22)" ] && [ -z "$(ls -A /scratch/mk8347/data/crystal_datasets/conditional/anchors/rc_oct22 2>/dev/null)" ] && J=$(sbatch --parsable --array=325 submit_rc_oct22.sbatch) && sbatch --kill-on-invalid-dep=yes --dependency=afterok:$J --array=0-324,326-999%250 submit_rc_oct22.sbatch && sbatch --kill-on-invalid-dep=yes --dependency=afterok:$J --export=ALL,TASK_OFFSET=1000 --array=0-429%250 submit_rc_oct22.sbatch && sbatch --kill-on-invalid-dep=yes --dependency=afterok:$J --export=ALL,TASK_OFFSET=3315 --array=0-499%250 submit_rc_oct22.sbatch
```

   The two `[ -z ... ]` tests stop the line if any rcoct22 job is already queued or the output directory is not empty
   (the first CPU revision reuses the run names and seeds of the never-submitted GPU revision for seed 0, and run_search
   resumes an existing output without comparing its start count). A failed smoke cancels the three arrays
   (`--kill-on-invalid-dep`).

   Any later range of a family's chunks is a further random sample of it; a range of tasks [a, b] runs as
   `--export=ALL,TASK_OFFSET=a --array=0-(b-a)`. Raise or lower `%250` to match how many cores the queue gives.

## Outputs

`/scratch/mk8347/data/crystal_datasets/conditional/anchors/rc_oct22/<family>_c<k>_<seed>.pt`, seeds 0-4 (lists of
`MolCrystalData`), with `_progress.json` beside each and the rendered `<run>.yaml`. A chunk's 10 starts per molecule
are the union over its 5 seeds; gfn-diffusion `data_processing/utils.py::load_search_chunks(<dir>, '<family>_c<k>')`
gathers them.

## Before a prior is built from these

The qm9full prior builder (`build_qm9_full_prior.py`) cannot read these outputs unchanged, and must not draw its own
held-out set: take the side of each molecule from the `split` column of the selection files (QM9 molecules keep the
qm9full prior's side), so the qm9full held-out molecules stay out of training and the held-out set does not reshuffle
between installments. The `cond_panel` evaluation tools parse identifiers as SMILES and need the `_c0` suffix stripped.

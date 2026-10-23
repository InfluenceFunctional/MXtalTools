# rc_oct22

Random-start eLJ searches, SG2 (P-1), Z'=1, on molecules whose geometry is **one random conformer per SMILES** in
place of a DFT geometry, for a larger conditional prior. One chunk family per molecule source; each family's list is
shuffled, so any run of its chunks is a random sample of it, and **the share of each source is chosen at submission
by how many of its chunks run**.

| family | molecules from | tasks (array indices) | chunks | molecules | relaxations | GPU-hours |
|---|---|---|---|---|---|---|
| qm9rc | the qm9full SMILES | 0-65 | 66 | 127,744 | 1,277,440 | 30 |
| csdrc | CSD | 66-93 | 28 | 55,099 | 550,990 | 13 |
| zincrc | ZINC20, MW row A | 94-185 | 92 | 182,419 | 1,824,190 | 42 |
| gdbrc | GDB-11 | 186-662 | 477 | 950,076 | 9,500,760 | 220 |
| pubrc | PubChem | 663-946 | 284 | 553,570 | 5,535,700 | 128 |
| all | | 0-946 | 947 | 1,868,908 | 18,689,080 | 433 |

*Task i runs 10 random starts for every molecule of one chunk file (2,000 molecules, the last chunk of each 20,000-
molecule shard holds the remainder). GPU-hours at the 12 relaxations per second measured on qm9_full_sep29; about 28
minutes per full task.*

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

Identical to qm9_full_sep29 apart from paths, run names and seeds (checked by diffing a task config):
`acr_finish_sep27/0.yaml` with `optim_target: elj`, two Rprop stages (compression, then annealing), `centroid_boundary:
wrap`, no reduction wall (outputs are not in the reduced cell), `convergence_eps` x10, no `early_stop` cascade, raw eLJ,
`init_target_cp: std`, `init_reduced: true`, `batch_size: 2000` with `grow_batch_size` and `oom_ceiling`.

## Files

- `make_battery.py` writes `tasks/<task>.yaml`, `INDEX.tsv` (family, chunk, seed, molecules, starts, relaxations,
  run name, opt_seed, molecule path per task) and the array line of the job script; `--scan <dir>` first writes
  `chunks.tsv` from the chunk files. It asserts distinct run names and seeds, seed slots no task can overrun, no local
  path, a chunk for every family (`--allow-missing` to override), and that a rewrite keeps every existing `INDEX.tsv`
  row (`--renumber` to override, only before anything is submitted): **an array index names a task**.
- `submit_rc_oct22.sbatch`: task = array index + `TASK_OFFSET` (default 0), at most 16 at once, 4 h walltime, `USR1` 10
  min before the end. Resubmitting resumes; a finished task exits at once; a task whose molecule file is missing exits
  at once.
- `check_battery.py`: every task's state from the progress files and the newest log of each task (matched by the
  `task=` the log prints), crystal counts, output size and throughput; standard library only (`python3 check_battery.py`
  on the login node).

Known behaviour, inherited from run_search: the molecules x starts list is shuffled with an unseeded generator, so a
task that is stopped and resumed relaxes a different permutation for its remaining starts; per-molecule start counts
then scatter around 10 (sd about 2 for a stop halfway). A full task takes about 28 minutes of a 4 h walltime, so this
only arises after a node failure or preemption.

## Launch (cluster)

1. Upload the chunk files of the installment with Globus, from the dev box's
   `D:\crystal_datasets\conditional\rc_oct22\mols\` to
   `/scratch/mk8347/data/crystal_datasets/conditional/rc_oct22/mols/`: only `<family>_mols_chunk<k>.pt` for the chunks
   to be run (about 4.7 MB each, 4.3 GB for all 947); the combined `<family>_mols_shard<s>.pt` files are for the prior
   build and can wait.
2. Pull and check the array limit:

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools && git pull && scontrol show config | grep -i MaxArraySize
```

3. Smoke one small task first (qm9rc chunk 65, 38 molecules, about a minute):

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools/configs/crystal_searches/rc_oct22 && sbatch --array=65 submit_rc_oct22.sbatch
```

4. Submit the installment by index ranges. For example, all of qm9rc, csdrc and zincrc plus the first 100 chunks of
   gdbrc and of pubrc (759,631 molecules, 176 GPU-hours, about 11 h at 16 at once):

```bash
cd /scratch/mk8347/projects/gfn_cond/MXtalTools/configs/crystal_searches/rc_oct22 && sbatch --array=0-285,663-762%16 submit_rc_oct22.sbatch
```

   Any later range of a family's chunks is a further random sample of it. Indices above the cluster's MaxArraySize run
   with `--export=ALL,TASK_OFFSET=<n>` (see the job script).

## Outputs

`/scratch/mk8347/data/crystal_datasets/conditional/anchors/rc_oct22/<family>_c<k>_0.pt` (lists of `MolCrystalData`),
with `_progress.json` beside each and the rendered `<run>.yaml`.

## Before a prior is built from these

The qm9full prior builder (`build_qm9_full_prior.py`) cannot read these outputs unchanged, and must not draw its own
held-out set: take the side of each molecule from the `split` column of the selection files (QM9 molecules keep the
qm9full prior's side), so the qm9full held-out molecules stay out of training and the held-out set does not reshuffle
between installments. The `cond_panel` evaluation tools parse identifiers as SMILES and need the `_c0` suffix stripped.

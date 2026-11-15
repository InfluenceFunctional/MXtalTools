# pool8sg_nov15

The pool8_nov09 search battery repeated in seven more space groups: P2_1/c (14), C2/c (15), P2_12_12_1 (19), P2_1 (4),
Pbca (61), Pna2_1 (33) and Cc (9). Same 72,464 pool8 molecules, same 2,899 chunks of 25, Z' = 1, 100 random starts per molecule,
streaming eLJ searches in the endpoint (ball) mode with every descent state recorded, CPU only. Built 2026-11-15 for
the multi-group conditional prior (owner: more space groups before bigger molecules). P-1 (2) is pool8_nov09.

## What is the same as pool8_nov09, and what differs

`make_battery.py` builds each task from `minima_stream/mipcas_elj_balls.yaml` as pool8_nov09 did. It then asserts that
task 0 of every group equals pool8_nov09's task 0 in every key except `mol_path`, `out_dir`, `run_name`,
`sgs_to_search` and `opt_seed` (`same_as_pool8`). Task k of group g:

- searches chunk k (`pool8_nov09/mols/pool8_mols_chunk<k>.pt`, the same files) in g alone (`sgs_to_search: [g]`);
- is named `pool8sg<g>_c<k>` and writes to `pool8sg_nov15/search/sg<g>/`;
- uses `opt_seed` = 3e11 + g x 1e10 + k x 1e6. The seeds are distinct across all 20,293 tasks and pool8_nov09's 2,899
  (asserted). As in pool8_nov09, a start redrawn on attempt a of task k reuses the seed block of
  task k + 1000a, so redrawn starts are not independent across tasks; redraws are rare and the effect negligible.

`INDEX.tsv` lists every task (group, chunk, molecules, starts, run name, seed, molecule file).

## Molecule frame and crystal handedness

- **Molecules.** run_search reads the chunk files unchanged (`utils.init_samples_to_optim`). Every drawn start is
  re-oriented with `orient_molecule('standardized', target_handedness=aunit_handedness)` (`stream_search.py` `drawn`,
  line 1020). At handedness +1 that is `align_mol_batch_to_standard_axes` with its default frame: the frame the files
  already hold, and the one the GFN trainer applies at load. On chunk 0 the stored positions move by at most 6.5e-6 Å
  under that map, at MXtalTools b6155c4a. Over all 72,464 chunk molecules, re-alignment at b6155c4a returns the same
  point set (atoms possibly relabelled by the molecule's own symmetry) for all but 13 molecules, which are searched as
  a slightly different pose of the same conformer (axis-sign ties; worst 1.8 Å element-matched). The search orients with whatever axis-sign rule the checked-out commit has.
  Each task records that commit (`mxt_commit` in `<run>.yaml` and `_slim.pt`), and the trainer that loads the band sets
  must use the same rule.
- **Handedness.** Every crystal is searched with handedness +1:
  - each template is built with `aunit_handedness` 1 (`utils.py:526`);
  - the random cell samplers redraw handedness only when some Z' > 1 (`MolCrystalOps.sample_random_crystal_parameters`
    and `sample_random_reduced_crystal_parameters`).

  No config key controls this at Z' = 1, so `task_cfg` asserts `zp_to_search == [1]` and the random start method
  instead. In the smoke, every trajectory state, library minimum and written band row had handedness +1, in all seven
  groups.
- **Why +1 is enough.**
  - In 2, 14, 15, 61, 33 and 9 an improper element of the group re-describes a handedness -1 crystal as the same crystal
    at +1.
  - In the Sohncke groups 4 and 19, the -1 crystal is the enantiomorph of the mirror-image molecule. The trainer's -1
    augmentation covers it, and eLJ is mirror-invariant.

## Cost

All timings below are on the dev box (Intel Core Ultra 9 275HX, MXtalTools b6155c4a, 2026-11-15). Each run used the
first 5 molecules of chunk 0, 100 starts each, 50 slots and one core, with three runs side by side. That is pool8_nov09's
per-molecule load, on 5 molecules instead of 25.

*Search wall time per start (s), `_slim.pt` size per start (KB), and the band step for one 5-molecule task (s). Each
cell is one run. The two s/start columns are two rounds with different neighbours; the P-1 row is the reference for
each round. Between rounds the same configuration varied by about 25% with box load.*

| space group | search s/start, round 1 | search s/start, round 2 | `_slim.pt` (KB/start) | band step (s/task) | search peak RSS (GB) |
|---|---|---|---|---|---|
| 2 P-1 (reference) | 0.511 | 0.396 | 17.4 | 51 / 38 | 2.3 |
| 15 C2/c | 0.473 | 0.434 | 18.8 | 16 / 17 | 2.5 |
| 19 P2_12_12_1 | 0.476 | | 16.4 | 26 | 2.2 |
| 4 P2_1 | 0.288 (ran alone) | 0.320 | 14.7 | 14 / 17 | 2.3 |
| 61 Pbca | 0.460 | | 17.4 | 19 | 2.3 |
| 33 Pna2_1 | 0.446 | | 15.9 | 20 | 2.2 |
| 9 Cc | 0.437 | | 15.6 | 19 | 2.2 |

Every group costs 0.8 to 1.1 times P-1 per start, so the budget below takes P-1's measured battery for each group:

- **pool8_nov09 on the cluster:** 25 min per task on a Xeon 8592+, so about 1,200 core-hours for 2,899 tasks.
  `_slim.pt` totalled 117 GB.
- **Band sets:** the pool8_nov09 band step took 212 s per task (dev box), about 170 core-hours per group, or less: here
  it ran at half P-1's time. About 1.1 KB per row on disk, so about 2.5 GB per group.

*Planned battery, seven groups, 100 starts per molecule. Each figure is pool8_nov09's measured P-1 battery times the
number of groups; per-group ratios to P-1 are 0.8 to 1.1 (table above; P2_1/c, added after the cost runs, was not
timed at 100 starts: its smoke ran at P-1's speed).*

| plan | search (core-hours) | band (core-hours) | `_slim.pt` kept (GB) | band sets (GB) |
|---|---|---|---|---|
| 100 starts, slims kept (BAND_AFTER=1) | ~8,400 | ≤ 1,200 | ~760 (100-126 per group) | ~18 |
| 100 starts, DELETE_SLIM=1 | ~8,400 | ≤ 1,200 | the tasks in flight (≤ 40 MB each), plus `_minima.pt` (~25 GB) | ~18 |
| 50 starts, slims kept (NUM_SAMPLES = 50, rerun make_battery.py) | ~4,200 | ≤ 1,200 | ~380 | ~18 |

`/scratch` is 5 TB and was 92% full in October. Without deleting slims, 760 GB probably does not fit. Deleting them
keeps the band sets and, per task, `<run>_minima.pt`: the slim without its trajectories (the minima library with hit
counts, the search effort and settings; about 3% of the slim, measured 18.5 of 532 KB on a smoke task). It gives up
re-banding with other settings and `band_from_slim.py --starts` (depth versus breadth), which need the trajectories.
That is the owner's call: `DELETE_SLIM` defaults to 0.

## Job scripts

`make_battery.py` renders the job scripts from `submit_template.sbatch` and `band_template.sbatch` into
`submit_sg<g>.sbatch` and `band_sg<g>.sbatch`. Edit the templates and rerun `make_battery.py`; do not edit the
rendered files. Account `torch_pr_226_chemistry`, no partition line (as pool8_nov09), 64 GB per job (pool8_nov09:
32 GB ran one task out of memory).

- **`submit_sg<g>.sbatch`:** 363 array jobs (`%100`) of 8 tasks side by side, one core each, 2 h per job. A task:
  1. runs run_search;
  2. runs pool8_nov09's `slim_outputs.py`, which writes `<run>_slim.pt` and deletes the bulk;
  3. with `BAND_AFTER=1` (default), runs pool8_nov09's `band_from_slim.py` into `band32/sg<g>/`, the same output as
     pool8_nov09's band step;
  4. with `DELETE_SLIM=1` (default 0), removes `<run>_slim.pt` once `band32/sg<g>/done/<run>.json` exists.

  Both variables are read from the environment. `DELETE_SLIM=1` without `BAND_AFTER=1` is refused. A task whose band
  record exists, or whose `_slim.pt` exists with `BAND_AFTER=0`, is skipped. A task with a `_slim.pt` and no band
  record runs only the band step. Resubmitting the same array therefore finishes only what is left.
- **`band_sg<g>.sbatch`:** for slims searched with `BAND_AFTER=0`; as pool8_nov09's `band_pool8_nov09.sbatch`.

## Launch (cluster)

The molecule files and the split file are pool8_nov09's, already on the cluster
(`/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/mols/`). Pull MXtalTools at the commit that carries
this folder.

Each job refuses to start unless `mxtaltools/common/geometry_utils.py` is the frame rule the pool8 data were built
with (git blob `ff4e836e`, as at b6155c4a) and the `mxtaltools/` tree has no local changes: the search orients every
start with that rule and the GFN trainer re-applies it at load. Since 2026-11-15 the packing-target draw of each start
block is seeded (`utils.get_initial_state`), so a task reproduces bit for bit from its config (checked on a smoke task
run twice; before, two runs of one config gave different trajectories). Run one smoke job per group, check it, then submit the rest:

    cd /scratch/mk8347/projects/gfn_cond/MXtalTools/configs/crystal_searches/pool8sg_nov15
    sbatch --array=0 submit_sg15.sbatch                          # smoke: tasks 0-7 of C2/c
    sbatch --array=1-362%100 submit_sg15.sbatch                  # the rest of the group
    sbatch --export=ALL,DELETE_SLIM=1 --array=1-362%100 submit_sg15.sbatch   # the same, deleting each slim once banded

Repeat for 14, 19, 4, 61, 33 and 9. Seven groups at `%100` would put up to 700 eight-core jobs in the queue at once; the
Slurm limit for the account has not been checked.

**Outputs**, under `/scratch/mk8347/data/crystal_datasets/conditional/pool8sg_nov15/`:

- `search/sg<g>/pool8sg<g>_c<k>_slim.pt` and `.log`;
- `band32/sg<g>/<split>/pool8sg<g>_c<k>_band/`.

A group is done when `ls band32/sg<g>/done | wc -l` is 2899. Then write one list file per group and split for
`build_noised_prior.py --basins` (GFN `energy_sampling/data_processing/`), one build per group with
`--set space_groups=[g]`:

    cd /scratch/mk8347/data/crystal_datasets/conditional/pool8sg_nov15/band32
    for g in 14 15 19 4 61 33 9; do for s in train heldout_random heldout_scaffold; do
        ls -d $PWD/sg$g/$s/pool8sg${g}_c*_band > sg${g}_${s}_list.txt; done; done

## Local smoke (2026-11-15)

The smoke ran on the dev box at MXtalTools b6155c4a, through run_search, `slim_outputs.py` and `band_from_slim.py`,
on the first 3 molecules of chunk 0 with 10 starts each and 10 slots.

*Per group: library minima per molecule; band rows written (32 per molecule requested) and importer refusals; the
space group spglib finds on each written row, rebuilt with `to_mol_crystal_data(mode='body')` from all atoms at
symprec 0.01 Å; and the wall time of the three steps. One run per group.*

| space group | minima per molecule | band rows written / refused | spglib group of the written rows (rows) | search + slim + band (s) |
|---|---|---|---|---|
| 2 P-1 | 9, 10, 9 | 96 / 0 | 2 (96) | 40 + 13 + 20 |
| 14 P2_1/c | 9, 9, 7 | 96 / 0 | 14 (96) | 31 + 13 + 11 |
| 15 C2/c | 9, 10, 7 | 96 / 0 | 15 (96) | 41 + 15 + 18 |
| 19 P2_12_12_1 | 10, 9, 7 | 91 / 0 | 19 (90), 62 (1) | 40 + 15 + 19 |
| 4 P2_1 | 10, 9, 8 | 96 / 0 | 4 (96) | 35 + 12 + 17 |
| 61 Pbca | 10, 10, 8 | 95 / 0 | 61 (95) | 45 + 15 + 18 |
| 33 Pna2_1 | 7, 10, 8 | 96 / 0 | 33 (93), 62 (3) | 37 + 14 + 19 |
| 9 Cc | 9, 10, 8 | 96 / 0 | 9 (96) | 37 + 14 + 18 |

- Every written row is `canonical` in its group, with handedness +1 and reference handedness +1.
- Rebuilt and rescored on eLJ at the hard cutoff, each row reproduces its stored E to within 0.005 (raw eLJ units).
- A row spglib places in a supergroup (62 Pnma over 19 and 33) is a crystal with more symmetry than its group requires.
- On the 5-molecule cost runs (32 rows requested per molecule), the importer refused 9 to 26 of 160 picks per group as
  `pose_mismatch`, P-1 included (26). This is the near-degenerate-frame refusal pool8_nov09 already shows, not specific
  to the new groups.

## Files

- `make_battery.py`: writes `tasks/sg<g>/<k>.yaml`, `INDEX.tsv` and the rendered job scripts. It asserts the checks
  above, plus no local path in any task.
- `submit_template.sbatch`, `band_template.sbatch`: the job script templates.
- `submit_sg<g>.sbatch`, `band_sg<g>.sbatch`: rendered from the templates.
- Slimming and banding reuse `../pool8_nov09/slim_outputs.py` and `../pool8_nov09/band_from_slim.py` unchanged.

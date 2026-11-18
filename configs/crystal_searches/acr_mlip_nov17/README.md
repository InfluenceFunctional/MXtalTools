# acr_mlip_nov17: acridine P2_1/c Z'=1 minima searches under three MLIPs

The acridine landscape is searched once per energy model with the streaming minima search
(`crystal_search/stream_search.py`). The models are MACE-OFF24 medium (`maceoff24m`), UMA-s 1.1 (`umas1p1`) and UMA-s
1.2 (`umas1p2`). This folder replaces acr_fm_oct27's coordinated campaigns (batch search, hops, curator), which never
launched. It keeps that campaign's models, checkpoints, per-model gas-phase conformers and seed pool, all already on the
cluster.

## Per model, two runs (`configs/`, written by `make_configs.py` from `models.py`)
- **`<model>_known`:** the known forms ACRDIN04 and ACRDIN12 (`../acr_zp1_sep30/known_forms.pt`) are the only seeds.
  There is no random start and no stop rule. The result is each form's minimum under the model, which the search's
  recall is read against.
- **`<model>_search`:** first the seed pool, then random starts.
  - The seed pool is `../acr_fm_oct27/seed_pool.pt`: 21,882 distinct states of the earlier acridine landscapes (the
    acr_newmodel prior rebuild's premerge basins and the old prior's anchors).
  - Random starts run after the seeds, up to 2,000 (`models.N_RANDOM`).
  - Stopping: the seeds and the random starts each end by the Good-Turing stop rule (`stop_p` 0.01 on a new minimum
    within 4 kT of the lowest).
  - Output: one row per minimum with Hessians (`<run>_minima.pt`). The basin set is off: acridine's third moment about
    one principal axis is about zero, so crystal_fd's importer rebuilds some rows in a frame turned 180° and refuses
    them (`pose_mismatch`). The UMA-s 1.1 smoke lost 4 of 8 minima that way.

Every run shares these settings:
- **Stages:** exp6_recall_nov10's two stages (Rprop 350 steps, then per-row BFGS 200 steps with the energy cascade), a
  10 Å cutoff, 96 slots, kT = RT at 300 K, and energies in kJ/mol per molecule.
- **Same starts for every model:** one `opt_seed` for all three, so they draw the same random starts and take the seeds
  in the same order.
- **Contact filter:** `min_intermolecular_contact` 1.5 Å. A row whose closest intermolecular atom pair is nearer is
  dropped before the polish, so an MLIP hole never enters the library, the cascade reference or the stop rule. Each
  run's effort file counts the drops (`contact_dropped`). Why: in acr_fm_oct27's calibration, MACE-OFF24 relaxed one of
  192 random starts into interpenetrating molecules at −238 kJ/mol, against −96 for ACRDIN12.
- **UMA only:** `mlip_gas_reference: mirror_mean`, because UMA is not reflection-equivariant. P2_1/c has an improper
  element, so the run needs no mirror average. Activation checkpointing is off.

## Run (cluster)
From the MXtalTools checkout, after `git pull` (Git LFS carries the seed pool):

    sbatch configs/crystal_searches/acr_mlip_nov17/submit.sbatch            # all three models, one GPU each
    sbatch --array=1 configs/crystal_searches/acr_mlip_nov17/submit.sbatch  # one model (0 maceoff24m, 1 umas1p1, 2 umas1p2)

`run.py MODEL` runs one model's whole chain:
- It checks the conformer's SHA1, that the model file exists, and that UMA 1.2's job imports fairchem 2.16 (from
  `/scratch/mk8347/pylibs_fc216`).
- It builds `seeds_<model>_{known,pool}.pt` in the output folder once. Beside each file, a `.json` records the origin of
  the seed with start index −(j+1).
- It then runs `<model>_known` and `<model>_search`.

Outputs and seed files go to `/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_mlip_nov17/`; logs to
`/scratch/mk8347/logs/acrmlip-<job>_<task>.out`. Jobs run 24 h. Resubmitting resumes each search from its committed
outputs, and a finished search is not redone. Fifteen minutes before the walltime the job gets SIGUSR1 and
the stream stops drawing starts; SLURM's SIGTERM releases the run's lease (`<run>_owner.json`). A process killed
outright (SIGKILL, a node failure) leaves the lease live for 30 minutes, and a resubmission inside that window refuses
to start (`RunLeaseLost`). To extend a search past its start cap, raise `models.N_RANDOM`, run
`make_configs.py`, and resubmit.

## Smoke (dev box)
`python run.py MODEL --smoke DIR` runs the same chain on the local files: 4 pool seeds, 4 random starts, 4 slots, and
stages capped at 40 steps. It runs on the CPU when CUDA is hidden (`CUDA_VISIBLE_DEVICES=-1`); on the GPU its memory is
capped at 40%.

# acr_fm_oct27: acridine P2_1/c Z'=1 landscapes under foundation models

One coordinated campaign (`crystal_search/coordinator.py`) per energy model: MACE-OFF24 medium (`maceoff24m`), UMA-s 1.1 (`umas1p1`) and UMA-s 1.2 (`umas1p2`). The models, their files and the relaxation schedule are in `campaign.py`. Each campaign has three streams with acr_zp1_sep30's coordinator settings:
- **seeded:** re-relaxes every distinct state of the earlier acridine landscapes under the model, plus ACRDIN04 and ACRDIN12. That is 51,690 states; `seed_pool.pt` comes from `build_seed_pool.py`.
- **random.**
- **hops:** kicks of registry basins.

Random and hops stop by their Good-Turing rules. Each campaign's identity cut is the model's own 1 kT thermal radius.

## Per model
- **Molecule:** the model's gas-phase minimum, built by `build_conformer.py` into `acr_<model>_conformer.pt`. It keeps the N-first atom order of every seed file.
- **Schedule:** acr_zp1_sep30's two stages, plus a polish stage. The polish runs to `convergence_eps / 100`, at most 800 steps, on rows within 3.5 kT of the job's best. Unconverged end states inflated acr_zp1_sep30's basin count.
- **UMA:** the searches run it with activation checkpointing off (`uma_activation_checkpointing: false`). With it on, the relaxation's backward pass raised torch's CheckpointError.
- **UMA 1.2 needs fairchem 2.16.** The container's fairchem 2.13 cannot build the model (HydraModel rejects `model_id`). Its jobs put `/scratch/mk8347/pylibs_fc216` first on PYTHONPATH: fairchem-core 2.16.0, pure Python, installed with `pip install --no-deps --target`.

## Step 1: calibration (one GPU job per model)
Run from the MXtalTools checkout on the cluster:

    bash configs/crystal_searches/acr_fm_oct27/calibrate.sh

For each model, `calibrate.py` does three things:
- relaxes the known forms and 14 distinct low anchors of the old prior (`calib_anchors.pt`);
- measures the 1 kT thermal radius (atomwise RDF), which becomes the identity cut;
- times 192 random starts at batch 96 on the stream schedule.

Output: `/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_fm_oct27/calibration/<model>/result.json`; logs in `/scratch/mk8347/logs/fm_calib_<model>-<job>.out`. A copy of each `result.json` goes to `calibration/<model>.json` here. That copy sets the campaign's identity cut and sizes its jobs.

## Built locally (dev box, CPU)
    python build_conformer.py mace|uma MODEL_PATH LABEL --write   # D:/crystal_datasets/acridine/acr_<LABEL>_conformer.pt
    python build_seed_pool.py --write                              # seed_pool.pt
    python build_seed_pool.py --anchors 14                         # calib_anchors.pt

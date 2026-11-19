# acr_mlip_nov19: acridine P2_1/c Z'=1 endpoint-mode searches under six MLIPs, with trajectories

The acridine landscape is searched once per energy model with the streaming search (`crystal_search/stream_search.py`)
in its endpoint mode, recording every descent's states, so that band sets (training priors) can be cut from the
trajectories afterwards. Owner, 2026-11-19: the endpoint recipe (pool8's), the three acr_mlip_nov17 models plus three
more, and acr_mlip_nov17's minima added to the seed set.

Models (`models.py`; array task in this order):

| task | name | model | route |
|---|---|---|---|
| 0 | `maceoff24m` | MACE-OFF24 medium | mace |
| 1 | `umas1p1` | UMA-s 1.1 | uma (task omc) |
| 2 | `umas1p2` | UMA-s 1.2 | uma; fairchem 2.16 from `/scratch/mk8347/pylibs_fc216` |
| 3 | `umam1p1` | UMA-m 1.1 | uma; 32 slots (not measured); its conformer is built by the job |
| 4 | `esens` | eSEN-S (OMC25) | uma (task omc); compressed aromatic crystals in the 2026-11-10 check |
| 5 | `macecspl` | MACE-CSP-25.2 large + D3(BJ, PBE) | mace with `mace_dispersion`; torch-dftd from `/scratch/mk8347/pylibs_dftd` |

## Per model, two runs (`configs/`, written by `make_configs.py` from `models.py`)
- **`<model>_known`:** ACRDIN04 and ACRDIN12 (`../acr_zp1_sep30/known_forms.pt`) as the only seeds; no random start,
  no stop rule.
- **`<model>_search`:** the seed pool, then up to 2,000 random starts (`models.N_RANDOM`); the seeds and the starts
  each end by the Good-Turing stop rule (`stop_p` 0.01 on a new minimum within 4 kT of the lowest).
  - **Seed pool** (`seed_pool.pt`, `build_seed_pool.py`): acr_mlip_nov17's 9,226 search minima (MACE-OFF24, UMA-s
    1.1, UMA-s 1.2; one per model in turn, each in ascending energy), then acr_fm_oct27's 21,882 states. Each is
    rebuilt on the model's own conformer.

## Search settings (every run)
- **Stages:** Rprop 350 steps, then per-row BFGS (600 steps) that stops on its force in the FD identity metric (`fmax`
  0.02 kT/Å) with the energy cascade at 10 / 8 / 5 kT after 25 / 50 / 100 steps; 10 Å cutoff; kT = RT at 300 K;
  energies in kJ/mol per molecule. `energy_noise` 1e-3 kT is set from MACE's float32 jitter, not measured per model.
- **Minima:** `polish: none`. Each BFGS endpoint is its minimum, identified by a 0.2 Å ball in the FD metric; a new one
  takes one Hessian to tell a minimum from a saddle (a saddle's descent is finished). No polish.
- **Trajectories:** every state of every descent within 15 kT of the lowest, thinned at 0.1 Å
  (`<run>_traj.pt`). `basin_set` is off: band sets are cut afterwards, when the prior's band is chosen.
- **Contact filter:** `min_intermolecular_contact` 1.5 Å.
- **No mirror keys:** acridine is planar (UMA-s gas energy of the molecule and its mirror image differ by 0.003-0.004
  kJ/mol) and P2_1/c has an improper element.
- **Same starts for every model:** one `opt_seed`.

## Run (cluster)
From the MXtalTools checkout, after `git pull` (Git LFS carries `seed_pool.pt`):

    sbatch configs/crystal_searches/acr_mlip_nov19/submit.sbatch            # all six models, one GPU each
    sbatch --array=3 configs/crystal_searches/acr_mlip_nov19/submit.sbatch  # one model (task numbers above)

`run.py MODEL` runs one model's chain: UMA-m's conformer when missing (`../acr_fm_oct27/build_conformer.py` on the CPU,
from `opt_acridine_conformer.pt` in the data folder; its SHA1 is printed), the checks (conformer SHA1 where pinned,
model file, fairchem version, torch-dftd for MACE-CSP), the seed files (`seeds_<model>_{known,pool}.pt` and a `.json`
of origins), then `<model>_known` and `<model>_search`.

Outputs go to `/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_mlip_nov19/`; logs to
`/scratch/mk8347/logs/acrmlip19-<job>_<task>.out`. Jobs run 24 h and resume from committed outputs when resubmitted;
SIGUSR1 15 min before the walltime drains the stream, SIGTERM releases the run's lease (`<run>_owner.json`).

## Smoke (dev box)
`python run.py MODEL --smoke DIR`: 4 pool seeds, 4 random starts, 4 slots, stages capped at 40 steps; CPU when CUDA is
hidden. UMA-m has no local checkpoint.

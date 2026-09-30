# acr_zp1_sep30: fresh acridine P2_1/c Z'=1 campaign under acr_newmodel

One coordinated campaign (`crystal_search/coordinator.py`) with three streams:
- **seeded:** re-relaxes earlier search states under the new model: the May and December sg14 Z'=1 searches, filtered and thinned in latent space (`seed_pool.pt`), plus the known forms ACRDIN04 and ACRDIN12, all rebuilt with the new conformer.
- **random.**
- **hops:** kick length drawn from 0.02 to 0.5, and recorded.

It runs until random and hops stop by their Good-Turing rules, or at the hard cap. The seeded stream is a fixed list with no rule. The settings, and where each value comes from, are in the docstring of `make_campaign.py`; `MANIFEST.md` lists the streams, tasks and seeds.

- Molecule: `acr_newmodel_conformer.pt`, the universal acridine conformer: the gas-phase minimum under `acr_newmodel.model`, with N first. `preflight.py` checks its SHA1.
- Identity: atomwise RDF leader clustering at the 1 kT thermal radius measured under this model (`identity_note` in `coord.yaml`).
- Jobs are short: 6 h walltime, 12 + 12 + 8 array tasks. Relaunch with the same line to continue; every task resumes. Keep relaunching until `stats.md` shows STOP, or until the campaign directory holds a `STOP` file.
- A CPU preflight job runs before the GPU arrays (`preflight.py`). It checks the container imports, the molecule and the model, builds `seeds/seeds_<k>.pt` from `seed_pool.pt`, and runs the coordinator's per-row path. If it fails, the arrays stay pending on the dependency: read `/scratch/mk8347/logs/a1_preflight-<job>.out`, then `scancel` them.

Launch, and every relaunch, from the MXtalTools checkout on the cluster:

    bash configs/crystal_searches/acr_zp1_sep30/launch.sh

Progress: `/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_zp1_sep30/stats.md`. Stop by hand: `touch <campaign directory>/STOP`.

    python make_campaign.py    # regenerates the configs (locally: needs D:/crystal_datasets/acr_newmodel.model for the model id)

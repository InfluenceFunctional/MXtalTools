# exp-6 recall test, stage B pilot (2026-11-10)

The question, the plan and stage A are in gfn_diffusion `energy_sampling/docs/design/exp6_recall_test.md`. This folder
holds what the cluster pilot reads.

- `panel/`: rigid conformers (`<name>_mol.pt`) and experimental anchors (`<name>_anchor.pt`) for imidazole (P2₁/c,
  IMAZOL06) and acetic acid (Pna2₁, ACETAC07), from gfn_diffusion `eval/exp6_recall/panel.py`.
- `configs/`: one streaming minima search per molecule × arm, from gfn_diffusion `eval/exp6_recall/make_configs.py`
  (`--site cluster`):
  - `<name>_uma_search`: the reference, UMA-s 1.1, at most 2,000 random starts;
  - `<name>_{exp6,elj}_search`: the cheap searches, 2,000 starts;
  - `<name>_uma_seeds_from_{exp6,elj}`: UMA-s 1.1 relaxing the 300 lowest minima of the cheap search.

  Every UMA-s run sets `mlip_gas_reference: mirror_mean`. Both groups have an improper element, so
  `mlip_mirror_average` is false.
- `run_pilot.py NAME reference|cheap`: the chain for one molecule; `--smoke` is a tiny local run of it.
- `submit_pilot.sbatch`: 4 array tasks (2 molecules × reference, cheap), one GPU each, 24 h.

Outputs go to `/scratch/mk8347/data/crystal_datasets/exp6_recall_nov10/pilot/`. The local smoke run (imidazole, RTX
5080 at 40%, 2026-11-10) completed every step, with every UMA-s polish converged.

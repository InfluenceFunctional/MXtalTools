# Project notes

## Before generating a new test battery — remind the user

When asked to make a new test battery / sweep (e.g. a new `battery`/`setN` directory
under `configs/experiments/`), **remind the user** that they wanted to sweep over:

- **Shorter trajectories** — the trajectory-length knobs live in the gfn-diffusion repo
  (`min_traj_length` / `max_traj_length`); mirror any equivalent rollout-length setting here.
- **Smaller gradient accumulation** — accumulation is currently implicit in
  `mxtaltools/modeller.py:1833` (steps every `data_loader.batch_size` iterations); sweeping
  it may mean exposing an explicit accumulation setting.
- **Smaller minimum batch sizes** — `min_batch_size` (currently 25 in
  `configs/experiments/dev.yaml`), alongside `grow_batch_size`, `max_batch_size`,
  `batch_growth_increment`.

Point: find out what we can get away with — how far these can be cut before results
degrade.

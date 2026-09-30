Local (dev-box) scripts that produced `../seed_pool.pt` and the identity cut, 2026-09-30; kept for provenance, not run
on the cluster. `pool_zp1.py` reads the May and December sg14 Z'=1 search outputs into `pool.pt`; `thin_zp1.py 0.031
--write` filters and thins them (0.031 = the latent 1 kT kick); `thermal_newmodel.py` (THERMAL_DEVICE=cuda) measured
the thermal radius. They name D:/ paths and write under POOL_DIR (default: this directory).

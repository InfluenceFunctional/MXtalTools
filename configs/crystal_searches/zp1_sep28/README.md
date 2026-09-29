# zp1_sep28: fresh Z'=1 campaigns for MIPCAS and NEHZOR, eLJ and UMA

Four coordinated campaigns (`crystal_search/coordinator.py`), each run until every stream's Good-Turing rule stops it or
its hard cap: `mipcas_elj`, `mipcas_uma` (P-1), `nehzor_elj`, `nehzor_uma` (P2_1/c), Z'=1, no priors. The settings, and
where each value comes from, are in the docstring of `make_campaigns.py`; `MANIFEST.md` lists directories, energy models,
kT, identity cuts, tasks and caps.

- Streams: random starts and hops (kicks of registry basins within 2 kT). Relaxation: the acr_campaign_sep28 schedule
  (wrap boundary, no reduction wall, stage-2 energy cascade, convergence thresholds x10 for the fe283818 EMA).
- Identity: ATOMWISE RDF leader clustering (`rdf_mode: atomwise`) at a provisional per-molecule cut (MIPCAS 0.071,
  NEHZOR 0.122); final identity by `campaign_compack` on each campaign's own end states.
- Energy reference: the lowest basin found (moving). kT: 2.494 kJ/mol (UMA); raw eLJ units for eLJ (2.494 / the GFN
  prior's thermal_scaling_factor).
- A hop job waits (up to `coord_hop_wait_s` = 2 h) while the registry has no basin yet: without priors the first curate
  pass can run before any shard exists.

Launch (from the MXtalTools checkout on the cluster):

    bash configs/crystal_searches/zp1_sep28/launch.sh all

Relaunch after a walltime: the same line (every task resumes; refused while a campaign's tasks are live). Progress:
`<campaign directory>/stats.md`. Stop one campaign: `touch <campaign directory>/STOP`.

    python make_campaigns.py    # regenerates the configs (locally: needs D:/crystal_datasets/esen_s.pt for the UMA id)

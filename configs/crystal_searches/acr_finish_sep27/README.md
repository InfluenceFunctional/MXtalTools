# acr_finish_sep27: how far does the reduction wall hold walkers? (2026-09-26)

Acridine, sg14 Z'=2, MACE (the same model as acr_proposals_sep27). 4 jobs, each one GPU for under 20 h. No library change: the search's own Rprop schedule, with the reduction penalty (`enforce_reduced`: $10^4 \times$ the sum of the squared monoclinic wall violations, `sym_utils.py::mono_reduction_penalty`) switched off where stated.

**Why.** In a local check (8 sep26 wrap end states, 100 more steps of stage 2 on an RTX 5080), continuing with the penalty moved 7 of 8 states by at most 0.13 kJ/mol. Without it, 3 of the 8 dropped by 0.9, 0.9 and 2.5 kJ/mol. The fourth outlier was an unrelaxed clash state. In acr_wrap_sep26, about 28% of end states sat on the β = 90.57° wall under both boundaries, and the no-reduction arm doubled the fraction within 2 kT. Those data were contaminated, so this battery repeats both measurements cleanly.

*Jobs, their arms, and what each is compared with. Samples are relaxations (jobs 0, 1) or continued end states (jobs 2, 3).*

| job | `run_name` | starts | reduction penalty | samples | compare with | question |
|---|---|---|---|---|---|---|
| 0 | `sep27nored_..._0` | random, seed 3e8 | off in both stages | 4000 | acr_proposals_sep27 jobs 1, 3 (`sep27rwrap`) | Does the search without the wall give more new low-energy families per unit of compute? |
| 1 | `sep27nored_..._1` | random, seed 4e8 | off | 4000 | as job 0 | independent replicate |
| 2 | `sep27finwall_..._0` | 2564 sep27 wrap end states | on, 300 more stage-2 steps | 2564 | each state's own end energy | How converged is stage 2 already? This is the control for job 3. |
| 3 | `sep27finnowall_..._0` | the same 2564 states | off, 300 more stage-2 steps | 2564 | job 2, state by state | How much lower does each state go once the wall is gone, and does it change family? |

- **Finish sources:** the first 1000 end states of `sep27rwrap_..._0` and `_1`, plus all 282 of `sep27seedwrap_..._0` and `_1` (`build_finish_seeds.py`). Rows are only ever appended, so both finish jobs continue identical states. Each state is rebuilt from its cell parameters with the search's conformer; in the local check the rebuilt MACE energy matched the recorded one within 0.01 kJ/mol. `DATA/sep27_finish/finish_wrap_compact.pth` maps each finish row (its `dataset_index`) to its source run, row and recorded energy.
- **Prerequisite:** the finish jobs need each random source to have at least 1000 rows. `sep27rwrap_..._1` was the slowest (about 4 per minute from about 14:20). If a source is short, the job stops with a message and nothing is written; resubmit with `sbatch --array=2,3 submit.sbatch`. The no-reduction jobs do not depend on this.
- **Outputs without the penalty** may be non-reduced cells (β < 90° or past the a/c walls). RDF families and energies do not depend on the cell choice.
- **Not in this battery: BFGS (`BatchedBFGS`, local only).** Near a minimum, MACE energies are quantised at about 0.0015 kJ/mol per molecule. The molecule-centre coordinates are very stiff: 0.03 Å costs about 0.05–0.1 kJ/mol. Steepest-descent steps short enough not to overshoot change the energy by less than one quantum, so an energy-tested step is rejected and BFGS stops. Locally it lost to Rprop on every near-converged state, and won only on the clash state. The gradients themselves were checked against finite differences along the stiff coordinates, which were smooth parabolas.

```bash
python make_configs.py   # regenerates the yamls + MANIFEST.md; asserts unique run names, no local paths, no umbrella keys,
                         # non-overlapping seed streams, the finish pair differing only in the penalty, the control = stage 2
sbatch submit.sbatch     # array 0-3
```

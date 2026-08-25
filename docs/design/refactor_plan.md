# MXtalTools refactor project — scope and sequence

**Status: PROPOSED. Owner decision required on §8 (the `AGENTS.md` amendment) and on WS-7 before Phase C.**
**Measured: 2026-08-23, main working tree, `.claude/worktrees/` and `__pycache__` excluded from every count.**
**Interpreter for all executed measurements: `C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe` with `PYTHONPATH` covering both repos.**

This document is a **decision** in the `AGENTS.md` sense: a scoped project selected by the owner, recorded here because it is not expressible in code or config. Its measurements are **observations** dated above. Its claims about contracted surfaces are **interfaces**. §9 classifies every material claim by type; §10 lists what is not yet proved.

**How to read it.** §1 is the diagnosis and is the only part that must be read in full. §2 is the work. §3 says what order and argues with the tiering that produced it. §4–§7 are the guards. §8–§10 are governance and honesty.

**Numbers in this document are not authority.** Every one was measured on 2026-08-23 against a tree with no CI. Re-measure before acting on any of them; §10 names the ones already known to be soft.

---

## 1. Diagnosis — five generators

Not a symptom list. These five produce almost everything else.

### G1 — There is no executable gate, so the tree cannot distinguish "works" from "has not been run"

`.github/` does not exist. `pyproject.toml` carries no `[tool.pytest.ini_options]`.

`python -m pytest --collect-only -q` at the repo root **exits 2**: `272 tests collected, 7 errors`, and the errors interrupt the run, so the 272 healthy node ids never execute. Four of the seven trace to one symbol — `sample_about_crystal`, deleted in `8dea6b56` on **2025-12-10** and still imported at `dataset_utils/construction/parallel_synthesis.py:12`.

Consequence, verified by direct import:

```
python -c "import mxtaltools.modeller"   → ImportError: cannot import name 'sample_about_crystal'
python -c "import main"                  → same
```

**`mxtaltools.modeller` and `main.py` — the trainer and the documented entry point — have been un-importable for roughly 8.5 months.**

The suite is not slow. With the seven broken files ignored: **236 passed, 35 skipped, 22.6 s** on CPU. A gate costing 23 seconds would have caught this the day it landed. This is not "we should add tests"; the repository has no mechanism that separates a live file from a dead one, and has run without one long enough for the trainer to rot in silence.

Everything below is downstream of G1.

### G2 — Every quantity crossing the boundary is a *difference* or a *roundtrip*, so whole families of wrong absolute values pass by construction

Lattice energy is `crystal_pot − gas_pot`. Latent transforms are `f⁻¹(f(x))`. Standardization is `destd(std(x))`. The MLIP vectorisation suites — 155 node ids, the best-instrumented part of the repo — assert *path A == path B*, so a defect present in both is invisible **by design**, not by oversight.

Live instances:

| Site | Defect | Why it passes today |
|---|---|---|
| `dataset_utils/data_class_methods/crystal_analysis.py` (MACE leg) | MACE lattice energy carries a per-molecule E₀ offset of **+11836.127 kJ/mol** | Cancels in every difference. Recorded only downstream, in GFN's `nikos_comparison/README.md` — a violation of `AGENTS.md:57` (downstream is not the home of MXtalTools rationale) |
| `dataset_utils/data_class_methods/crystal_ops.py:1136-1137, :1153-1154` | `standardize_cell_lengths` / `destandardize_cell_lengths` re-declare the same three literals inline, then collapse them with `.mean()*ones_like` — discarding the per-axis anisotropy the docstring advertises | Symmetric, so the roundtrip is exact **for any pair of values**. A test cannot tell the intended flattening from a typo |
| `common/sym_utils.py:209, :224` | `niggli_reduction_penalty(cell_lengths, cell_angles)` vs `cell_reduction_penalty(cell_angles, cell_lengths)` — **opposite positional order**, both `.split(1, dim=1)` into three | A swapped bind returns finite, plausible numbers |
| `mlip_interfaces/AL_mace_utils.py:230-231`, `uma_utils.py:382-383, :684-685` | A crashed MLIP forward returns **zeros**, not NaN and not an exception | `(0/(sym_mult·z_prime) − gas_pot)·96.485` is a number |

*Table: measured instances of the difference/roundtrip blind spot, main tree, 2026-08-23. Each row is a path on which a wrong absolute value survives every check currently in the repository.*

One test in the repo breaks the pattern: `conformers/tests/test_conformers.py:471` pins torsion barriers at 2.80 and 1.00 kcal/mol against GAFF. It is the template.

### G3 — Conventions live at call sites and in implicit state, not on the objects

`std_orientation` is a caller argument, never a stored property. On a dataset-shaped batch the two settings give eLJ **−20.7 kJ/mol** (`True`) versus **−14.5 / −10.5** (`False`), with max vdW overlap **0.000 Å** versus **0.43 / 0.63 Å** — real clashes. GFN passes `False`. Both legs return finite numbers and raise nothing. *(Agent-measured; re-measure before relying on the magnitudes.)*

Same generator at the other end of the tree: `Modeller` creates ~22 attributes outside `__init__`; `train_models_dict` has six writers with inconsistent key sets while `run_epoch` unconditionally indexes three keys; `train_loader_to_replace` uses *attribute existence* as control flow. `radius` and `mol_volume` are frozen scalars that do not track `pos`.

And the same generator produces the four one-line defects in WS-0, each of which is a convention asserted nowhere:

- `dataset_utils/data_class_methods/mol_methods.py:88` — `scatter(self.z > 1, ..., reduce='sum')` on a **bool** tensor. Boolean sum-scatter saturates: returns `[1,1]` where the truth is the heavy-atom count. The batch branch of `radius_calculation` therefore raises downstream and **has never executed**; every production `radius` comes from the stored dataset value.
- `crystal_analysis.py:131` — `'niggli_overlap'` is registered pointing at the raw free function `niggli_reduction_penalty`, not a bound method.
- `crystal_ops.py:1748` — `_pad_tensor` reads `.dtype` off an `int` its own type hint permits.
- `conformers/prior.py:249` stores ring banks under `(sig, n_dof)`; `:320` and `:355` look up `self.rings.get(sigs[s])` — a bare signature. The lookup **always returns `None`**, so every ring is silently drawn uniform over 0.6–2.6 Å bonds.

### G4 — The GFN boundary is declared in prose and crossed by deep import at 215 sites

Measured by AST over the GFN tree:

| Quantity | Value |
|---|---|
| Distinct `(module, symbol)` pairs imported from MXtalTools | **75** |
| `from … import` statements | 215 |
| Distinct MXtalTools modules reached | 32 |
| Distinct GFN files importing | 73 |
| `from mxtaltools import …` (package-level) | **0** |
| Private symbols crossing | 1 — `uma_utils._predictor_wants_external_graph` |
| GFN files reaching `collate_data_list` | 53 |
| Symbols pinned by the one behavioural test | **4 of 75 (5.3 %)**, and it asserts no numeric value |

*Table: the GFN→MXtalTools import surface, measured by `ast.ImportFrom` over `gfn_diffusion/**/*.py` (worktrees, `__pycache__`, `wandb` excluded), 2026-08-23. Every pair is a coupling that a rename breaks.*

Three further facts make this worse than the count suggests:

- **Undeclared couplings.** 18 pairs sit in no `AGENTS.md` boundary group, including `common/utils.log_rescale_positive` (the reward-shaping transform), `is_cuda_oom` (the OOM branch), and `flatten_wandb_params` (both GFN training entry points).
- **`examples/` is part of the boundary and nobody knows it.** GFN imports `examples.crystal_search_reporting.batch_compack` at two sites. `examples/` is a tracked top-level package with an `__init__.py` and is **absent from `pyproject.toml`'s `include`**.
- **Reverse coupling exists.** `dataset_utils/utils.py:37` hard-codes the GFN-owned string `'gfn_energy'` in the collation exclude list; `tests/conftest.py:46` does `import gpu_guard` — a **GFN module** — against `AGENTS.md:69`; and MXtalTools rationale is homed downstream at 7 sites citing `gfn docs/decisions.md D33` and `findings.md F-008`.

### G5 — The library and the experiment log are the same tree, and nothing marks which is which

**4142 of 4387 tracked files (94.4 %) are under `configs/`.** Of 4018 YAMLs, **1138 declare a `base_config_path`** and **876 of those point at a file that does not exist** (871 → `/experiments/base.yaml`, 5 → `/experiments/base/GEOM_autoencoder.yaml`), so they raise `FileNotFoundError` before training starts.

But `configs/` is **not** an inert log. `configs/crystal_searches/` holds 1821 tracked files and received **368 of 368** config files added in the last 60 days — 100 % of recent config activity. It is the active research surface. Any deletion rule phrased over `configs/**` lands on live work.

Elsewhere: `reporting/papers/` is ~4.3k LOC with zero external importers and is already unimportable; `reporting/old_figures.py` is 840 lines reachable only on a config value no config sets; `models/*/old/` is 281 lines already broken and already wheel-excluded; `modeller.py` carries 393 lines of dead generator code; `docs/source/dataset_creation.rst` names five scripts, none of which exist.

And one governance fact that belongs here: **`AGENTS.md` is untracked.**

```
git ls-files --error-unmatch AGENTS.md → did not match any file(s) known to git
```

The constitution does not reach the cluster, a collaborator, or a fresh clone.

---

## 2. Workstreams

Sizes are days of focused work for a single maintainer who is also running research. **Recommended cut line: end of Phase B.** §7 states what cutting there actually costs, in the owner's own vocabulary.

Every workstream below names its **safe stopping points** — the commits at which an interruption leaves the tree correct, just unfinished. A workstream with no safe stopping point is a design flaw, not a schedule risk.

### Phase A — foundation (≈14 d)

---

**WS-0 · REPAIR — reach a green, measurable baseline** — 2–3 d, no deps

*Problem:* G1. Collection exits 2, so no measurement of any later change is trustworthy.

**IN:**
- `git add AGENTS.md`. Nothing in §8 means anything until the constitution is in version control.
- **Fix `mol_methods.py:88` FIRST**, before deleting anything. `tests/test_data_classes_basics.py:185-186` executes at collection time and is the **only** call of `mol_analysis()` on a batch in either repo — it is the sole live detector of this defect. Watch it go from raising to passing, *then* convert it into a proper test. The plan's original order deleted the witness on day 1 and fixed the defect on day 2.
- The other three one-line defects, each with a failing-first test: `crystal_analysis.py:131` (bind `niggli_overlap` as a method), `crystal_ops.py:1748` (accept `int` handedness, allocate on the batch device), `prior.py:320`/`:355` (`self.rings.get((sigs[s], len(order)))`). The ring test must **fail when the bank is never consulted** — asserting the numbers are finite proves nothing, since uniform draws are finite too.
- Restore or delete `sample_about_crystal` and `batch_aunit_pose_analysis`.
- **Add the missing `__init__.py` files.** Eleven directories under `mxtaltools/` contain `.py` files and no `__init__.py`, including `analysis/`, `reporting/`, and `dataset_utils/{data_class_methods, construction, analysis, md_analysis, normalizer_reduction}`. A `pkgutil.walk_packages` module walk therefore reaches **67 of 145** modules — 46 %. The unreached half contains the *entire* GFN boundary (`data_class_methods/`), both files carrying G2's named numeric defects, and `parallel_synthesis.py` itself. **A pkgutil-based smoke gate would not have caught the founding example of this whole document.**
- Delete `tests/old_bad/` and the three dead test files — **but freeze the importer census first** (see WS-4 and R9).

**OUT:** any behavioural change to a live numeric path except the four fixes.

**Exit:** `pytest --collect-only -q` exits 0 with no `--ignore`; `import mxtaltools.modeller` succeeds; a **filesystem-walk** import smoke over all 145 modules reports zero failures other than genuinely optional backends.

**Safe stopping points:** every fix is independent. Stop anywhere.

---

**WS-1 · THE GATE — hook first, CI second** — 2–3 d, deps WS-0

*Problem:* G1.

The original scoping proposed four GitHub Actions tiers. **That is architecture for a problem this repo does not have.** The full collectible suite is 22.6 s. The expensive and risky part of CI here is *installing* torch + torch_scatter + torch_cluster + e3nn + rdkit + fairchem + mace on a runner — per-CUDA PyG wheels, which WS-9 itself calls the unfixable half of the install problem. Budgeting 3–4 d for CI without naming that is how CI work consumes a fortnight.

**IN, in this order:**
1. **A local pre-push hook running the 23 s suite.** This delivers most of G1's value on day one at zero install risk, and it works on the machine the research actually runs on.
2. `[tool.pytest.ini_options]`: `testpaths`, `markers = ["gpu","mlip","ccdc","slow"]`, `--strict-markers`, `norecursedirs`.
3. Convert four hard third-party imports to `pytest.importorskip`.
4. **Make a no-MLIP CPU tier possible at all** by deferring the three module-level MLIP imports (`mol_methods.py:14`, `crystal_analysis.py:14-15`) into their calling functions. Today `import mxtaltools.dataset_utils.data_classes` pulls ase, e3nn, fairchem, mace, matplotlib, numba, plotly, rdkit, sklearn, torch_cluster, torch_scatter and wandb. This is WS-1-internal ordering, not a dependency on WS-4.
5. **One** GitHub Actions job: filesystem import-walk + `--collect-only` + the CPU suite. Add tiers only when something demands one.
6. Root `conftest.py` — **but not by moving the current one**, which does `import gpu_guard` from GFN. Extract the GPU pre-flight to a MXtalTools-local helper first, or the move widens a reverse coupling `AGENTS.md:69` forbids.

**OUT:** GPU runners; anything needing CCDC or `D:/crystal_datasets`; a nightly tier until there is a nightly to run.

**Exit:** a deliberately-introduced `sample_about_crystal`-shaped deletion fails the hook in under a minute. If a nightly MLIP tier is ever added, it must **fail** when `$MACE_CHECKPOINT` / `$UMA_CHECKPOINT` are unset rather than skipping green — a silently-skipping nightly reads as a pass.

**Safe stopping points:** each of the six items ships alone.

---

**WS-2 · BOUNDARY FREEZE** — 4–5 d, deps WS-1

*Problem:* G4. **This is the load-bearing change of the project.** The data-class work rewrites the objects that *are* the boundary; doing that with 5.3 % pinned is the single most likely path to a silent GFN break.

**IN:**
- Generate `mxtaltools/api.py` **from the measured 75-pair table, not by hand**, partitioned by `AGENTS.md` boundary group, with `api_mlip.py` using lazy imports so MACE/UMA stay optional.
- **`api.py` ships as pure re-exports.** Deep imports keep working. This is what makes every intermediate commit a valid stopping point.
- `tests/test_api_surface.py` asserting: exact `__all__` set equality (catches *additions*, not only removals); frozen `inspect.signature` per symbol; `'mol2cluster' in vars(MolCrystalBuilding)` and `'construct_radial_graph' in vars(MolCrystalAnalysis)` — the two class attributes the GFN boundary test monkeypatches; `COMPUTES_REQUIRE_CLUSTER` / `_UNIT_CELL` contents; and `DEFAULT_COLLATE_EXCLUSIONS` hoisted out of `dataset_utils/utils.py:30-43` and pinned **by value, including the `'gfn_energy'` literal**, so the reverse coupling becomes visible and dated rather than buried.
- **Contracted entry points must reject unknown keywords.** `analyze`, `compute_eLJ_energy`, `compute_lattice_uma` and `compute_lattice_mace` all carry `**kwargs`, and the comment at `crystal_analysis.py:379-384` records that this has already fired once: a `std_orienation` typo swallowed GFN's `std_orientation=False`, so every UMA run scored std-oriented crystals while MACE and eLJ runs did not. **A frozen signature does not catch that.** Drop the `**kwargs` or raise on leftovers, and test that a misspelled keyword raises `TypeError`.
- Add `examples/` to the boundary inventory.
- A GFN-side mirror test that AST-scans for `from mxtaltools.` imports not targeting `api`/`api_mlip` — **shipped as a report, not a gate**, becoming a gate only when its waiver list empties.
- **A cross-repo lockstep rule, written down:** no symbol retirement without the matching GFN commit in the same cluster pull. The deploy pulls both repos; a mid-battery mismatch costs 16 A100s.

**OUT:** the conformer group (`AGENTS.md:53`); reporting/clustering/encoder-loading (`:55`); `dataset_utils.construction` (`:65`) — but note GFN's `eval/nikos_comparison/ingest.py` imports 9 symbols from it, so carry that as a **dated waiver**, not as silence.

**Exit:** renaming any frozen symbol or editing `DEFAULT_COLLATE_EXCLUSIONS` fails the CPU tier in under 30 s.

**Safe stopping points:** every commit, by construction, because `api.py` is additive.

---

**WS-3 · GOLDEN VALUES — break the difference/roundtrip trap** — 5–7 d, deps WS-1; runs parallel to WS-2

*Problem:* G2.

**The single most important correction to the original scoping:** derive the golden list from **what GFN actually calls**, not from what looks numerically fragile inside MXtalTools. The first draft pinned `standardize_cell_lengths`, which has **zero GFN callers**, while leaving `reduction_en` — which enters the reward — unpinned.

**Mechanism:** `tests/golden/generate_refs.py` → `refs.json` recording (value, tolerance, generator commit, backend, checkpoint id, dtype, device); `test_golden_values.py` reads it. **Regeneration is an explicit flag whose output is reviewed as a diff, never a silent overwrite** — otherwise the suite re-baselines the bug.

**Tranche 1 — the reward path GFN rides. All CPU.**

1. **eLJ absolute** — `compute_eLJ_energy` on a fixed synthetic batch, 6 s.f. This is the reward `mk_dev.yaml` trains on and **nothing in either repo pins its scale.** A change to `stiffness`, `VDW_RADII`, or the cutoff shifts every reward uniformly, which a GFN run absorbs as a temperature change and never reports.
2. **`reduction_en` absolute** for a named triclinic and a named monoclinic cell, **plus an explicit asymmetry assertion** that `f(lengths, angles) != f(angles, lengths)` on the fixture. Given `sym_utils.py:209` and `:224` take their arguments in opposite orders, a swap must *fail*, not merely differ.
3. **The inverse latent direction** — fixed latent → physical `(a,b,c,α,β,γ)` to 6 s.f., at Z′=1 and Z′=2, for a space group with a non-trivial `asym_unit_lut` row. The live literals GFN rides (`au_range`, `ang_range`) are **duplicated inline** at `crystal_ops.py:361-364` and `:396-399`; because both copies move together, the roundtrip stays exact for any value while the physical cell each latent decodes to changes — invalidating every stored prior, buffer row, and checkpoint policy with no error.
4. **Table-indexing invariants.** `crystal_analysis.py:107` builds `torch.tensor(list(VDW_RADII.values()))` and indexes it **by atomic number**. That is only correct because the keys happen to be contiguous 0–99. Assert `tensor(list(VDW_RADII.values()))[z] == VDW_RADII[z]` for all z, same for `ATOM_WEIGHTS`. Free, and it kills the whole class.
5. **`compute_rdf_distance`** — self-distance exactly 0, plus at least five named distances straddling the calibrated 0.085 / 0.147 thresholds (two below, two above, one between). GFN imports this module at 24 sites and the thresholds are load-bearing.
6. **`adaptive_batched_analysis` partition invariance** — same input at two `initial_batch_size` values, energies agree to a stated tolerance. Its knobs read as pure performance settings and would be retuned freely; UMA is not bit-reproducible on GPU, so a "tuning" edit can move scored energies.
7. **Molecule scalars** for named QM9 molecules — pinned to **post-fix** values, since WS-0 fixes `mol_methods.py:88` first. Prove the tier works by *deliberate re-introduction*, not by shipping a known-failing ref.

**Tranche 2 — structure and symmetry.** `sym_mult == len(SYM_OPS[sg_ind])`; asymmetric-unit invariants (AU fractional volume × `sym_mult` == 1; every op has |det| == 1; the set closes under composition mod 1); `T_fc`/`T_cf`/`cell_volume` for a **triclinic** cell against an independent oracle (existing coverage is cubic-diagonal and self-roundtrips only); the closed-form `log_partition_latent` at T==1 — the oracle GFN's own log-Z estimators are validated against, currently unpinned; the soft-wall `log(2+√(π/k))`.

**Note:** the AU invariant is not new work. `dataset_utils/normalizer_reduction/validate_asym_units.py` is already a Monte-Carlo check that each `RAW_ASYM_UNITS` box is a genuine fundamental domain of `SYM_OPS[sg_ind]`. It prints and flags. **Promote it into an assertion.** The claim "17020 lines of literal data carry zero tests" should read "zero *assertions* — one unrun script."

**Tranche 3 — MLIP, nightly.** `mace_pot`, `mace_gas_pot`, `uma_pot`, `uma_gas_pot` pinned **separately, not only as their difference**. This is the one structure that would have caught the 11836.127 offset. Plus an invariance test: score the same physical cell at two `z_prime`/`sym_mult` factorisations and assert the per-molecule lattice energy is unchanged — an E₀ residue moves with the divisor, a correct cancellation does not. Run it on the **eLJ** leg too, since GFN divides eLJ by `z_prime` itself while MXtalTools divides the MLIP legs by `sym_mult · z_prime`.

**Restate the 11836.127 kJ/mol observation locally in MXtalTools**, with its measurement conditions, per `AGENTS.md:57`.

**OUT:** anything requiring `D:/crystal_datasets` or CCDC. **No Z′>1 MLIP reference until the `mol2ucell` kwarg drop is decided** (see R11) — restrict to Z′=1 and record the restriction in `refs.json`.

**Exit:** deliberately re-introducing each of the four WS-0 defects produces a **failing** test, verified by actual re-introduction.

**Safe stopping points:** every reference is independent.

---

**WS-4 · QUARANTINE — separate the library from the experiment log** — 2–3 d, deps WS-0

*Problem:* G5. This is deletions and moves with measured zero importers, not an architectural project.

**Discipline first, because the plan can otherwise manufacture its own evidence:** freeze and date the importer census (the AST scan) **before any deletion commit**, and require every "zero importers" claim to cite the frozen census rather than a live grep. The original scoping deleted `tests/old_bad/` on day 1 and then, on the strength of a later grep, concluded `crystal_building/crystal_latent_transforms.py` was unreferenced — but `tests/old_bad/crystal_latent_space.py:8` was its only reference in either tree.

**IN:** move `reporting/papers/` out of the package tree; delete `old_figures.py` and its two references; `git rm -r models/*/old/`; delete the untracked root artifacts (`*fig.png`, `*.npy`, `opt_intermediates.pt`) and extend `.gitignore`; delete `crystal_search/testout.pt` (22.4 MB), the constants generator artifacts, and `tests/datasets/misc_data_for_new_csd.npy` (636 MB, gitignored, **zero references repo-wide** — so nothing depending on it has worked for any collaborator); delete `mkdocs.yml` (a second doc system, never built); delete `mlip_interfaces/requirements.txt` (a shadow dependency declaration no build tool reads — and the **only** place `mace-torch` is written down, so record it in `pyproject.toml` first).

**Configs — with a carve-out.** Delete the 876 provably-unloadable configs **by full resolved path, never by basename** (41 files under `old/` are named `base.yaml` and collide with a live reference), and re-run the `base_config_path` existence check at deletion time rather than trusting this document. **`configs/crystal_searches/` is excluded from every deletion rule** — it is the active research surface.

**OUT:** the ~3.5k machine-generated sweep children, pending an unverified fact (whether the generators reproduce their checked-in output — verify one first). Also out: `git filter-repo` on the 155 MB pack. It rewrites every SHA and breaks four active worktrees. It is a `.git` problem, not a working-tree problem.

**Exit:** `git status` clean; `--collect-only` still exits 0; the import smoke still green.

**Safe stopping points:** each deletion is independent.

---

### Phase B — the architectural work (≈22 d)

---

**WS-5a · GEOMETRY NUMERICS** — 4 d, deps WS-3, safe to run anytime

*Problem:* G2 + G3 in the geometry layer. Contains the highest-value single test found in the whole survey.

1. **`batch_molecule_principal_axes_torch` returns a different answer with gradients on.** `geometry_utils.py:454`, inside `scatter_compute_Ip`, adds `randn_like(coords) * eps` whenever `requires_grad`. Measured over 500 molecules: **median axis deviation 1.03°, p95 5.09°, max 29.98°, and 4.4 % of molecules get an axis sign flip.** This is a contracted GFN-facing function with a train/eval discrepancy announced nowhere. A two-line test exposes it. *(Agent-measured; the `randn_like` call is verified, the distribution is not.)*
2. **Handedness is effectively random** — ~50 % right-handed across all three implementations, and three different "vanishing overlap" thresholds (`1e-3` normalised, `1e-8` raw, `1e-5` normalised) exist for one test.
3. **`correct_Ip_directions` is defined twice in one file** — `geometry_utils.py:275` and `:324`. The live one is `:324`. **Do not "clean up the duplicate" by keeping the first definition.** Ship a collection-time guard against duplicate module-level names — or, more cheaply, adopt a linter: this is ruff/flake8 `F811`, obtained free from a tool the repo does not yet run. Pin `correct_Ip_directions` against an explicit expected sign-per-axis matrix, **never against the other definition**.
4. **The tested `enforce_crystal_system` is not the one that runs.** `enforce_crystal_system2` has 0 production callers and 7 test callers; `enforce_crystal_system` has 5 production callers and 0 test callers. Move the tests.
5. **Units guard.** Promote the existing `print` at `geometry_utils.py:827-828` into a shared `assert (angles < pi).all()` used by all six cell functions. Degrees where radians are expected gives 466.66 Å³ for a 1000 Å³ cube — finite, positive, plausible, undiagnosed.
6. Delete the confirmed dead duplicate pairs.

**OUT:** `space_group_info.py` → `.npz`. Measured cost is 226 ms cold / **5.9 ms warm**. Not worth a packaging change that cannot be verified without a built wheel.

**Exit:** grad-jitter and handedness tests pass; deliberately re-introducing the `signs[..., None]` broadcast fails.

---

**WS-5b · MLIP SEMANTICS + RNG** — 2–3 d, deps WS-5a, **gated on a cluster window**

Separated from WS-5a because both items change live production failure modes on the exact MACE/UMA path the cluster runs.

1. **MLIP crash returns zeros → make it raise or return NaN**, with a test that monkeypatches a non-OOM `RuntimeError` and asserts the result is **not finite**. Land this **before** any MLIP golden reference is generated, or the refs pin the zeros-return as truth.
2. **`AL_mace_utils.py:24-32` replaces `torch.load` process-globally with `weights_only=False` and never restores it**, rebinding `_original_torch_load` to the *previous* patch on each call so wrappers nest. Any GFN run touching MACE silently changes all of GFN's own `torch.load` sites. Wrap in `try/finally`. Note the local environment sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, which **masks this locally** — a fresh CI runner will not.
3. **Determinism (issue #156).** 13 library `__init__`s call `torch.manual_seed`, so building one model changes the next global `randn`. Any GFN run loading a frozen encoder has its global torch RNG overwritten by a seed literal baked into MXtalTools. **Fix by save/restore**, which preserves current init weights bit-for-bit. Thread explicit `Generator`s only at the two sites with no GFN callers. Generator-threading everywhere changes every initialised weight and invalidates every checkpoint — a separate decision, taken deliberately or not at all.

---

**WS-6 · DATA-CLASS CONVENTIONS + INTERNAL-DoF CARRIER** — 12–18 d, deps WS-2 (hard), WS-3 (hard), WS-5a (soft)

*Problem:* G3 in the data layer; the flexible-molecule blocker.

**The scoping correction that matters:** the flexible-molecule work is a **small, enumerable, additive** set of data-class changes — *not* a featurization rebuild. `MolConformerMethods` is already on `MolCrystalData` via the MRO, and stores its index arrays `[k,n]` so PyG batches them with no extra code. **The DoF carrier is not the missing piece; the conventions are.** Bundling this with the CCDC/CIF rebuild (WS-10) makes the crystal+conformer joint-DoF payoff hostage to hydrogen placement. Split them.

**IN, permissive-first, so nothing before step 4 changes GFN-visible behaviour:**

1. **Pure module splits.** `crystal_ops.py` (1970 lines) → symmetry / latents / sampling / fundamental-domain / **plots** (458 lines of plotly, 23 % of the file); `crystal_analysis.py` → lift the GFN toy-energy surface out, and the CCDC blocks into their own module. **Methods stay bound to the same classes** — the GFN contract is method-level. Note: the split is safer than the stale comment at `data_classes.py:699` implies — the method-name intersections between `crystal_ops`, `mol_methods`, `conformer_methods` and `crystal_analysis` are all **empty**. Say so in the commit; do not leave it unclaimed.
2. **Cache/payload separation.** `computes`, `edges_dict`, the LUTs and `ellipsoid_model` currently ride through `clone`/`subsample_new_batch`/`append_batch`. **This is the step most likely to go silently wrong**: `compute_lattice_uma` skips re-scoring whenever `uma_pot` already exists on the object, and `_pre_compute_checks` refuses only on the **absence** of `edges_dict`, never on staleness. Changing which attributes survive a clone changes whether a batch reuses a previous batch's energies. Guard with a test that analyzes the same object twice at different `std_orientation` and asserts the result **changes**.
3. **Convention attributes defaulting to today's behaviour.** `orientation_frame` defaults to the frame GFN currently gets, and the `pose_aunit(std_orientation=…)` kwarg still wins for one release while **asserting agreement** rather than overriding. Land the measured `True`/`False` eLJ gap as a test.
4. **`mol_pos` split** — separate the three meanings of `pos` (reference conformer / posed aunit / cluster), defaulting to a view of `pos` so existing datasets load unchanged.
5. **Turn silent wrong answers into refusals.** `crystal_rdf`'s `atomwise`/`envwise` paths must raise unless `num_atoms` is uniform. Measured today: the same 7-atom crystal reports **28 atomwise channels alone, 15 in batch `[5,7]`, 28 in batch `[7,5]`** — the answer depends on which graph is first — with a finite plausible array and no warning in every case. `compute_rdf_distance` takes a `channel_key`, not just `shape[-2]`, and the change must leave the WS-3 pins bit-identical on a uniform batch. **This is what unblocks flexible molecules**, because it forces the pair alphabet to become per-graph.
6. **Z′ ragged pointer** — replace the unchecked `num_atoms // z_prime` integer division with a stored `zp_num_atoms` defaulting to the uniform split, so every existing dataset is bit-identical.
7. **`internal_latents` alongside, never inside, the cell vector.** Widening the cell vector breaks `SG_FEATURE_TENSOR`, the dead-row machinery, and every stored buffer.
8. Vectorise the four `scale_*_to_aunit` python loops with the LUT that already exists (measured 26.5× at N=256 CPU).

**OUT:** renaming or unbinding any contracted method; widening `full_cell_parameters`/`latent_params`; touching `MXtalBase`'s batch bookkeeping (load-bearing CUDA-sync-avoidance work); `dataset_manager.py`, which self-labels as "almost certainly no longer works".

**Exit:** the API-surface test and the GFN boundary test green at every commit; the mixed-batch RDF table becomes a refusal test; joint crystal+conformer DoF optimization runs end-to-end on one molecule.

**Safe stopping points: steps 1, 4, 6, 7, 8 are individually shippable. Steps 2, 3, 5 are NOT** — an interrupt mid-step-2 leaves `_store`/`_INSTANCE_DICT_ATTRS` half-migrated, which is a silent-wrong-answer state, not a stopped state. Each of those three must land as a single commit or not at all.

---

**WS-7 · CONFORMER REPATRIATION** — 3 d, deps WS-0

*Problem:* the `AGENTS.md` wait has already been breached, in the direction the constitution forbids. See §8.

**IN — defect fixes and file moves, no new design:** move GFN's `tests/conformer/test_mmff_matches_rdkit.py` (237 lines, imports only `mxtaltools.conformers.*` + rdkit + torch — **no GFN import**) into MXtalTools, making `energy.py:42` and `:463` cite proof that can actually run; the ring-bank key fix from WS-0 plus a test asserting the bank is **reached**; export `spec_from_graph`, the primary GFN entry, currently absent from `__all__`; fix three stale docstrings (`ff_from_graph` claims no torsion term while the code sets one; `log_jacobian` mislabels the BAT volume element); test `infer_bond_index` against RDKit's bond **set**, not just plausible lengths (current coverage passes if bonds are *missed*); resolve the `gradient_descent_optimization` name collision between `conformers/optimize.py` and `crystal_search/crystal_opt_utils.py`.

**OUT — explicitly, each with a revisit condition:** `InternalPrior.sample`/`log_prob` as a draw policy (missing every measured correction); `RingBank` vs `RingModes` (two incompatible models, one with no `log_prob`); `optimize.py` (**zero tests anywhere**, and measured slower than the GFN optimizer); the chart (`dof_from_state`/`state_from_dof`) — the ladder is at step 2 of 7; anything reward-adjacent, since `log|dq/dx|` is missing and spreads ~9.8 nats across molecules.

**Exit:** the moved test passes standalone with no GFN on `sys.path`; the ring-bank test **fails** when the key is reverted.

---

**WS-8 · INSTALL — lockfile, doctor, honest dependencies** — 3–4 d, deps WS-0, WS-4

The owner's reframing is correct and the evidence supports it: the *unfixable* half (per-CUDA PyG wheels, licensed `ccdc`) is real, and the *fixable* half is small.

**IN:** declare the two undeclared dependencies (`umap-learn`, `spglib`) — or better, make the `umap` import lazy, since it enters through `figures.py` → `old_figures` → `umap` and WS-4 deletes `old_figures.py` anyway. Add a `mace` extra mirroring the existing `uma` one. Delete the seven declared-but-never-imported dependencies, notably **`pathtools`**, which is abandoned and whose `setup.py` uses `imp`, removed in Python 3.12 — a clean-install hazard from a package nothing imports. Move `pytest` to a dev group. **Commit `poetry.lock`** (currently absent). Fix `exclude = ["**/*.pt"]`, which strips `dataset_utils/ellipsoid_overlap_model.pt`, a file `torch.load`ed at runtime. Drop `configs/**`, `tests/**`, `misc/**`, `main.py` from `include` — they install as top-level namespaces. **Add `examples/`** or decide deliberately that it is not shipped, given GFN imports it.

**Ship `python -m mxtaltools.doctor`** (~80 LOC) reporting torch/CUDA/PyG-extension versions and which optional backends are present. This is the highest-leverage item because it makes the *unfixable* half legible instead of cryptic.

**Exit:** `poetry build` once and settle every `include`/`exclude` claim **against `unzip -l dist/*.whl`, not against the config**; a clean venv install imports every externally-reached module.

---

### Phase C — deferrable (≈25 d). **Cut here if time is short.**

**WS-9 · MODELLER DECOMPOSITION — 8–12 d. Recommendation: cut, or flag-gate.**

The evidence against scheduling this is strong enough to state plainly: `mxtaltools.modeller` has **zero GFN importers** (its only importers are dead `reporting/papers` files, dead test files, and `main.py`), **zero tests**, and no boundary constrains it. It blocks exactly one thing — four collection errors — and those are fixed by repairing one import, not by decomposing a 2718-line class.

Worse, its core step is **explicitly non-resumable**: `hit_max_lr_dict` is created as a side effect of `init_optimizers`, and `handle_nan` **replaces the whole `optimizers_dict` object**, so a half-migrated `ModelBundle` silently trains the wrong parameters. For an 8–12 day item at the interrupt-prone end of a sole-maintainer schedule, that is a design flaw. If it is done, both paths stay live behind a flag.

Before any decomposition, note that `evaluate_model` **cannot execute for any mode today**: it calls a logger method that is commented out, and a `self.crystal_structure_prediction` that does not exist. Separately, `increment_batch_size` reads a config key present in **zero** yaml files, raising `AttributeError` outside a `try` — while **641 non-`old/` configs set `grow_batch_size: True`**.

**WS-10 · CIF I/O + CCDC DECOUPLING — 8–12 d, with research risk.** A gemmi-backed reader producing the same `crystal_dict`, keeping CCDC as a second backend behind one interface with `crystal_rebuild_checks` as the equivalence test. A symmetric writer emitting SG + aunit rather than P1 — today `write_cif` loses `sg_ind`, `symmetry_operators`, `z_prime`, all `aunit_*`, `identifier`, `is_well_defined`, and partial charges. **The hard core is `assign_bonds`/`add_hydrogens` on crystal geometry, which has no drop-in replacement.** Scope it as "gemmi covers cell/SG/ops/components; H placement stays CCDC-gated until proven." COMPACK stays an optional CCDC-only eval dependency.

**WS-11 · MODEL DEPLOYMENT — 4–5 d.** One `load_model(path, device, *, eval_mode, freeze)` plus a `manifest.json` sidecar, replacing five divergent loaders. Two of them load the **same** autoencoder checkpoint differently — one reads `dataDims['allowed_atom_types']`, the other hard-codes the atom list, reads the 105.7 MB file **twice**, and returns a model that is not `.eval()`, not frozen, and not on-device. The `weights_only=True` migration is blocked by exactly two pickled globals, both located: `argparse.Namespace` and `torch.return_types.histogram` — and **no loader reads the dict containing the second**, so dropping it removes one blocker outright. A working precedent already exists in-tree.

---

### Phase D — named so it is deferred rather than invisible (size unknown)

**WS-12 · DATASET CONSTRUCTION AND THE SECOND CONFORMER STACK.**

`dataset_utils/construction/` (18 files), `dataset_utils/analysis/` (3), `md_analysis/` (1), `mol_building.py`, and `normalizer_reduction/` (679 LOC) are the largest unmapped region of the tree. Declaring them "migration debt" and moving on makes them invisible, not deferred. **Three of the owner's own items live entirely here**, plus 86 hardcoded absolute `D:\` paths across 20 files — which *is* the flexible-I/O problem, stated in the tree.

Two facts worth surfacing now:

- **There is a random *conformer* generator; there is no random *molecular graph* generator.** The distinction is the whole item. `dataset_utils/mol_building.py` (`smiles2conformer`, `generate_random_conformers_from_smiles`, `scramble_dihedral_angles`, `pare_molecule_skeleton`) is entirely SMILES→3D: every entry point takes a `smile: str`, and the randomness is conformational. **The graphs themselves are read off disk.** `parallel_synthesis.py:153` `generate_smiles_dataset` walks a ZINC22 directory and samples *files* with `p = file_sizes / sum(file_sizes)`, then reads their lines. So the only "controllable statistic" available today is the file-size distribution of a third-party corpus — which is not a chemical control at all, and cannot be steered toward the size, composition, or ring statistics an autoencoder curriculum needs. A random SMILES generator with controllable statistics is therefore **genuinely net-new construction**, feeding the existing SMILES→conformer machinery, which is reusable as-is.
  Three properties of the current supply path constrain the design: it requires ZINC22 present on disk at a hardcoded path (`D:\crystal_datasets\zinc22` local, `/scratch/mk8347/zinc22` cluster); it does `os.chdir` into that directory, mutating process CWD; and it lives in `parallel_synthesis.py`, **the file whose dead `sample_about_crystal` import has been breaking the trainer for 8.5 months** — so this path has not been exercised in that time either. Its `seed` parameter is also dead: `:154` is `#np.random.seed(seed)`, commented out, while `generate_otf_crystal_chunk.py:30` passes `seed=chunk_ind` expecting per-chunk determinism.
- **`mol_building.py` is nonetheless a second, unreconciled conformer stack** sitting beside `mxtaltools/conformers/` (12 files, 3814 LOC). Any flexible-molecule work that ignores the rdkit torsion-scrambling path will be redone when the two are unified.
- **The fundamental-domain code is here, not in `crystal_search/`.** `normalizer_reduction/validate_asym_units.py` and `validate_composition_full.py` are the fundamental-domain validators. Any fence that protects `crystal_search/` protects the wrong directory.

**WS-13 · SCIENCE FENCE (protected, not scheduled).**

Fundamental-domain work and a better search suite are protected by a structural fact: **no GFN file imports `mxtaltools.crystal_search`**. Its five extension seams already exist as pure functions — `compute_loss`/`compute_auxiliary_loss` (registry-ready), `CrystalParams.update_params(mask=)` (the accepted-set hook a Metropolis step needs), `get_initial_state` (a four-way switch, and the point of highest scientific leverage given that search misses are an initialiser problem rather than a basin problem), the six clustering functions `run_search.py` never calls, and the result store.

**Fence rule, corrected:** no refactor workstream may open a file under `mxtaltools/crystal_search/`, `configs/crystal_searches/`, or `dataset_utils/normalizer_reduction/` — except WS-0's one import repair. The original wording fenced only the first, which is the one that is *not* in flight.

Two caveats on the fence's founding claim: GFN **does** import `examples.crystal_search_reporting`, and six MXtalTools modules import `crystal_search`, including `crystal_ops.py` (which WS-6 splits) and `figures.py` (on a live GFN path). The fence is about GFN importing `mxtaltools.crystal_search`, not about isolation.

*Note for the science, not the refactor:* four config-reachable `optim_target` values (`classification_score`, `inter_overlaps`, `latent_dist`, `rdf_score`) raise `KeyError` *after* paying for a checkpoint load.

---

## 3. Sequencing — and where the evidence contradicts the tiering

**Real dependencies, each with its reason:**

- **WS-0 → WS-1.** You cannot install a gate while collection exits 2. Any "green" CI built on `--ignore` flags is a gate that ignores what broke.
- **WS-1 → everything.** Every subsequent claim in this document is a claim about behaviour, and behaviour claims need an executor.
- **WS-2 → WS-6, WS-9, WS-10.** Anything that moves a contracted symbol moves against a frozen surface. 71 of 75 symbols are currently unpinned.
- **WS-3 → WS-5, WS-6.** *You cannot refactor numerics against roundtrip tests.* A roundtrip lets you move a bug from `f` into `f⁻¹` and stay green. This is the same structure as the E₀ offset, and it is why the golden tier must **precede** the numerics work rather than follow it.
- **WS-6 → flexible-molecule science.** The DoF carrier exists; the conventions do not.

**Five places where the evidence contradicts the original tiering:**

**(a) Modeller decomposition is not foundation — demote from Tier 1 to Phase C, and consider cutting.** Zero GFN importers, zero tests, no boundary constraint, one non-resumable core step. Doing it early spends 8–12 days on the part of the tree with the least downstream leverage while the boundary is still unfrozen. It is worth doing eventually; it is a prerequisite for nothing.

**(b) The public API is not Tier 3 — promote it to Phase A, before the data-class work.** The data classes *are* the boundary: `MolCrystalData`, `MolData` and `collate_data_list` account for a large share of the 215 import sites, and `collate_data_list` alone reaches 53 of 73 GFN files. Rewriting them with 5.3 % pinned is the highest-probability silent break in this plan.

**(c) "Featurization rebuild + flexible molecules as ONE project" is the wrong seam — split it.** They share the word "featurization" and nothing else. The flexible-molecule blockers are five additive data-class changes whose migration path is no-op-by-construction through step 3. The featurization rebuild's hard core — CCDC `assign_bonds`/`add_hydrogens` on crystal geometry — has no drop-in replacement and is a research problem. They touch the same directory, not the same problem.

**(d) "Conformer promotion" is not one item — three of its parts belong in week 1.** The ring-bank key mismatch is a live correctness bug *in the very code `AGENTS.md` proposes to contract*. Moving the MMFF test is a pure file move that discharges an `AGENTS.md:69` violation. The docstring fixes are free. Everything else genuinely waits, with stated revisit conditions.

**(e) Numerics debt is absent from the tiering entirely, and it contains the highest-value single test found.** `batch_molecule_principal_axes_torch` returning a different answer when `requires_grad` is set, on a contracted GFN-facing function, is a two-line test exposing a discrepancy nothing in either repo measures.

**One thing the tiering gets exactly right:** the installation reframing. The unfixable half is real and the fixable half is small and concrete. Do not attempt more.

---

## 4. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Editing `DEFAULT_COLLATE_EXCLUSIONS` changes what survives collation. Note the invariant runs the opposite way from the obvious reading: `'gfn_energy'` is in the **exclude** list, so collation *drops* it. The hazard is **removing** it, which starts carrying a stale per-sample reward into the replay buffer. | Critical, silent | WS-2 pins the list by value and adds a **positive** test: collate a batch carrying `gfn_energy`, assert the attribute is absent. |
| R2 | Resplitting the data-class mixins breaks `mol2cluster` / `construct_radial_graph` as *class attributes*, which the GFN boundary test monkeypatches — so the failure presents as a boundary regression and misdirects diagnosis. | High | Assert `'mol2cluster' in vars(MolCrystalBuilding)` directly. Method splits keep bindings; only module homes move. |
| R3 | GFN calls MXtalTools **private methods unbound with `self=None`**. Giving them a real `self`, renaming them, or moving plotting out of `MolCrystalOps` breaks GFN **at figure time**, with no import-level warning. | High, silent | WS-6 lifts plotting into a mixin, keeping methods bound. Add the names to the frozen surface as a dated waiver, or give GFN public accessors first. |
| R4 | The golden suite re-baselines the bug: a regeneration flag silently overwrites `refs.json` and the next run is green against wrong values. | Critical | Regeneration is explicit and reviewed as a **diff**. Refs carry generator commit, backend, checkpoint id, dtype, device. Never regenerate in CI. |
| R5 | Golden MLIP references generated on a machine where a forward **crashed** pin the zeros-return as truth. | High, silent | Land WS-5b item 1 **before** generating any MLIP reference. Assert crash-count == 0 in the generator. |
| R6 | `AL_mace_utils.py` patches `torch.load` process-globally and never restores it; wrappers nest on repeat calls. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` masks it locally but not on a runner. | High | `try/finally` restore; a test asserting `torch.load` is unpatched after `load_mace_model`; CI explicitly unsets the env var. |
| R7 | Fixing the 13 library `manual_seed` calls by threading `Generator`s changes every initialised weight and invalidates every checkpoint. | Medium | Save/restore preserves init bit-for-bit. Generator-threading is a separate, deliberate decision. |
| R8 | Changing the orientation default shifts every eLJ reward by a measured 30–50 % with real clashes appearing. **GFN absorbs a uniform reward shift as a temperature change and never reports it.** | Critical, silent | WS-6 step 3 defaults to today's GFN behaviour and *asserts agreement* with the kwarg rather than overriding. |
| R9 | A deletion pass manufactures its own evidence: delete the only importer, then measure zero importers, then conclude dead. | Medium | Freeze and date the AST importer census **before** any deletion commit; every "zero importers" claim cites the frozen census. |
| R10 | Deleting configs by **basename**: 41 files under `old/` are named `base.yaml` and collide with a live reference. Deleting by directory rule lands on `configs/crystal_searches/`, which received 100 % of the last 60 days of config activity. | Medium | Full resolved paths only; `crystal_searches/` carved out entirely; re-run the existence check at deletion time. |
| R11 | Z′>1 kwarg drop: `crystal_building.py:230` calls `pose_aunit()` bare while the Z′=1 branch forwards `std_orientation` — on the contracted MACE acridine route. Phase-A MLIP refs could pin it as truth. | Medium | No Z′>1 MLIP golden ref until it is measured and decided; record the restriction in `refs.json`. |
| R12 | `conformer_prior_v2.pt` pickles a **GFN class** (`RingModes`) inside a MXtalTools `InternalPrior` — the one-way dependency broken at the artifact layer. Moving `prior.py` breaks loading it. | Medium | WS-7 does not touch `prior.py`'s module path. Any ring promotion first moves `RingModes` into MXtalTools or stops the artifact carrying it. |
| R13 | **Cross-repo lockstep.** The cluster deploy pulls both repos at HEAD. A MXtalTools commit changing eLJ (R8, WS-5, WS-6) lands mid-battery on 16 concurrent A100s. `gfn_diffusion/requirements.txt` does not name mxtaltools; GFN resolves it by sibling `sys.path`, with no pin, tag, or revision recorded in any run. | **Critical** | A freeze-window rule: no numeric-path commit while a battery is in flight. Record the MXtalTools revision in each GFN run's config. No symbol retirement without a matched GFN commit in the same pull. |
| R14 | `mxtaltools` runs **from the source tree, not installed** — there is no staging buffer. Every edit is live to every running local job on save. | High | Do numeric-path work on a branch, or during a declared cluster window. |
| R15 | Fixture coverage gap: the mini dataset's space groups are all triclinic/monoclinic/orthorhombic. `enforce_crystal_system` has **no real tetragonal/hexagonal/rhombohedral/cubic case**, and `LATTICE_TYPE` maps no space group to `'rhombohedral'`, making that branch unreachable. | Medium | Add ~10 higher-symmetry crystals during WS-3; document the hexagonal-setting convention for sg 143–167 as a local invariant. |
| R16 | The worktrees under `.claude/worktrees/` carry **divergent copies** (`crystal_ops.py` is 1612 lines there vs 1970 in the main tree). Any survey grep that did not exclude them returned wrong counts. | Medium | Every measurement excludes them, per `AGENTS.md`. Re-derive any number whose provenance is unclear. |

*Table: risks ranked by whether a failure would be silent. "Silent" means the failure produces a finite, plausible number and no diagnostic — the class of failure this repository is structurally prone to.*

---

## 5. The first week

**Day 1 — measure, unblock, and fix in the right order.**
`git add AGENTS.md`. Fix `mol_methods.py:88` **first**; confirm the module-scope call in `test_data_classes_basics.py` goes from raising to passing; *then* convert it to a real test. Decide `sample_about_crystal` and `batch_aunit_pose_analysis` (restore from `8dea6b56^` or delete the call sites). Freeze the AST importer census before any deletion. **Target: `--collect-only` exits 0 with no `--ignore` flags.**

**Day 2 — the remaining three one-line fixes, each failing-first.**
`crystal_analysis.py:131`, `crystal_ops.py:1748`, and the ring-bank key — with a test that fails when the bank is never consulted, not one that checks the numbers are finite.

**Day 3 — the gate, cheapest first.**
A pre-push hook running the 23-second suite. Then `[tool.pytest.ini_options]`. Then the eleven missing `__init__.py` files, and an import smoke that **walks the filesystem**, not `pkgutil` — otherwise the gate is blind to 46 % of the package, including the entire boundary. **Prove it works by deliberately deleting a symbol and watching it fail.**

**Day 4 — make a no-MLIP tier possible.**
Defer the three module-level MLIP imports. Extract the GPU pre-flight from `tests/conftest.py` into a MXtalTools-local helper so the root `conftest.py` does not import GFN's `gpu_guard`. Land one CI job.

**Day 5 — the boundary freeze, generated not hand-written.**
Run the AST scan that produced the 75-pair table and emit `mxtaltools/api.py` from it as **pure re-exports**. Hoist `DEFAULT_COLLATE_EXCLUSIONS`. Write `test_api_surface.py`. Land it **green, before any refactor commit**. Ship the GFN mirror test as a report.

**Spillover, low-risk, no decisions:** WS-4's zero-importer deletions, once the census is frozen.

**What week 1 deliberately does not touch:** any numeric path except the four fixes; anything under `crystal_search/`, `configs/crystal_searches/`, or `normalizer_reduction/`; `modeller.py` beyond the import repair; the 876 dangling configs.

---

## 6. Cross-cutting concerns that no workstream owns

These are not scheduled. They are recorded so that "the refactor is done" cannot be claimed while they stand.

| Concern | Measured state | Coverage |
|---|---|---|
| Logging | **250 `print(` in `mxtaltools/`; 0 modules import `logging`; 2 `warnings.warn` total** | WS-5a promotes one print to an assert. No policy, so the other 249 stay. |
| Error handling | **38 bare `except:`** plus ~12 `except Exception` | WS-5b fixes one returns-zeros site. The bare excepts are the same failure class and are untouched. |
| Configuration | `dict2namespace` turns YAML into `argparse.Namespace` with **no schema and no validation**; 876 configs declare a nonexistent base | WS-4 deletes the 876; nothing prevents the 877th. This `Namespace` is *also* WS-11's `weights_only` blocker — one defect seen twice. |
| Lint | **No `.pre-commit-config.yaml`, no ruff/flake8/tox**; 121 TODO/FIXME markers; 38 % docstring coverage (505/1314 functions and classes) | Not scheduled. WS-5a's bespoke duplicate-name AST guard is `F811`, free from a linter. |
| Versioning | `version = "0.1.0"`, **zero git tags**, no CHANGELOG | WS-8 commits a lockfile and stops. |
| Cross-repo reproducibility | GFN does not pin, tag, or record a MXtalTools revision in any run | R13. **The load-bearing gap.** |
| Performance regression | Only harnesses are `examples/timing_benchmark.py` and `examples/profile_mlip_energy.py`, the latter self-documented as never executed | No CI tier is performance. Yet `AGENTS.md` requires "a named benchmark, not a comment," and this document makes ~15 performance claims. None becomes reproducible under the plan as written. |
| Published API reference | `docs/source/modules.rst` omits `conformers`, `analysis`, `reporting`, and `data_class_methods` — i.e. every subsystem the refactor spends money on, including where the entire boundary lives | Nobody owns it, and no CI builds docs, so it goes stale exactly where the work lands. |

*Table: cross-cutting concerns, measured 2026-08-23 over `mxtaltools/` excluding `reporting/papers/`. "Coverage" states what the plan above does about each, which in most rows is nothing.*

---

## 7. What this plan does not deliver

Stated in the owner's own vocabulary, because a cut line is a descoping decision and should read as one.

| Owner item | Where it lands |
|---|---|
| Improved modularization | WS-6 (data classes) in Phase B; WS-9 (Modeller) in Phase C, **recommended cut** |
| Crystal building/analysis testing and efficiency | WS-3, WS-5a, WS-6 |
| Formalize GFN-side conformer work | WS-7 — **partial**; the chart, prior policy, ring model and optimizer explicitly wait |
| Flexible molecules, crystal+conformer joint DoF | WS-6 — this is the Phase B payoff |
| Easy installation | WS-8, reframed to lockfile + doctor + honest deps |
| Robust CIF I/O (aunit and unit cell) | WS-10, **below the cut line** |
| Featurization rebuild, drop CCDC | WS-10 + WS-12, **below the cut line**; the H-placement core has no known replacement |
| Efficient deployment of existing models | WS-11, **below the cut line** |
| More sophisticated search suite | WS-13, **protected but not scheduled** |
| Fundamental domain work | WS-13 + WS-12, **protected but not scheduled**; note the code is in `normalizer_reduction/`, not `crystal_search/` |
| Update workflows to new batching/analysis standards | WS-12, Phase D, **size unknown** |
| Seamless MLIP integration | Partially WS-5b; the duplicate-MACE question (`mace_utils.py` vs `AL_mace_utils.py`) is **unasked** |
| Random molecule generator with controllable statistics (SMILES→conformer at scale) | **No workstream. Net-new construction.** The *conformer* half exists (`mol_building.py`); the *graph* half does not — molecular graphs are read from ZINC22 on disk, sampled by file size (`parallel_synthesis.py:153`). See WS-12 for the constraints the current supply path imposes |
| Data-driven diffusion workflow | **No workstream. Not deliverable under this plan.** `grep -ril diffusion` returns two files repo-wide, neither an implementation |
| Docs and docstring updates | Scoped to a handful of items; `modules.rst` unowned (§6) |
| General UX | Emergent from WS-2 + WS-8; not a workstream |

*Table: coverage of the owner's stated to-do list by the workstreams above. Rows marked below the cut line are recommended not-now. Two rows have no workstream at all — both are net-new construction rather than repair, and this plan is a repair plan; they should be scheduled as their own projects.*

Also unscheduled and not counted in any size estimate: **the GFN-side work this plan creates.** WS-2 retires GFN-only symbols by GFN-side edits and adds a GFN mirror test; WS-7 moves a GFN test file; R3 may need GFN public accessors; R12 may need `RingModes` moved. None of it is in the day counts, and the sole maintainer is concurrently running the LR-controller validation queue and cluster batteries. **Assume less than 100 % of a person.**

---

## 8. `AGENTS.md` — two conflicts, surfaced rather than overridden

`AGENTS.md:35` requires that a conflict between sources be **reported**, not silently resolved. Two apply.

### Conflict 1 — the second-tier clause

`AGENTS.md:7` places broad modernization and user-friendliness in the second tier "unless explicitly selected." **This project is that explicit selection**, made by the owner on 2026-08-23.

The correct disposition is a **dated, owner-signed amendment to `AGENTS.md`** recording: (i) that a scoped refactor project was selected on 2026-08-23; (ii) its bounded scope — WS-0 through WS-13 as listed; (iii) that the selection **expires on completion**, restoring the second-tier default; and (iv) the science fence as an explicit carve-out.

Without that amendment this document reads as a subordinate file overriding the constitution — exactly the failure mode `AGENTS.md` exists to prevent. **And the amendment is currently impossible to make: `AGENTS.md` is untracked.** `git add AGENTS.md` is WS-0, day 1.

### Conflict 2 — the conformer wait, already breached

`AGENTS.md:7` says conformer integration should wait for the GFN conformer refactor; `:53` contracts the conformer group "after the conformer refactor settles." Measured, the wait has already been violated **in the direction the constitution forbids** — GFN reaching into MXtalTools, and MXtalTools depending on GFN:

- `conformers/energy.py:42` and `:463` cite `test_mmff_matches_rdkit` as their proof. That file does not exist in MXtalTools; it is in `gfn_diffusion/energy_sampling/tests/conformer/`. This breaches `AGENTS.md:57` (downstream is not the home of MXtalTools rationale) **and** `:69` (independently testable without GFN). MXtalTools source currently cites proof it cannot run.
- `gfn_diffusion/energy_sampling/build_ring_banks.py:314` sets a **GFN class** (`RingModes`) onto a `mxtaltools.conformers.prior.InternalPrior`, and `conformer_prior_v2.pt` pickles it there. The one-way dependency is broken at the artifact layer.
- GFN calls the private `prior._layout` at three sites.

The wait as written is being read as a blanket hold, and under it three one-way breaches have accumulated.

**Recommended amendment to `AGENTS.md:7` and `:53`:** re-scope from *"wait for the refactor"* to *"Tier-1 repatriation is authorized now — moving proofs into MXtalTools, fixing defects in code the constitution proposes to contract, and exporting entry points already in use. The chart, the prior draw policy, the ring model, `optimize.py`, and anything reward-adjacent continue to wait, each with a stated revisit condition."*

The alternatives, stated so the choice is real: **(b)** leave the hold and accept that `energy.py` cites unrunnable proof indefinitely; **(c)** promote the whole `:53` list now, which contracts `InternalPrior.sample` *with* the ring bug and `optimize.py` with zero tests.

**If this goes unanswered, the safe default is WS-7's first three items only** — a test move, a one-line key fix with its test, and two docstring corrections. None touches a moving interface, and all three reduce the breach.

### A third item, smaller: one `AGENTS.md` sentence is stale

`AGENTS.md:65` says `pbc_neighbours.py` belongs to the MACE adapter. Measured, `uma_utils.py:322-329` calls `batched_pbc_neighbour_list` directly with the external-graph flag defaulting on. The constitution's *intent* — that UMA must not inherit MACE's indexing contract by inference — is honoured in code by the explicit per-backend convention mapping. The *ownership sentence* is not. **Restate it, do not delete it:** "shared by both MLIP adapters under per-backend convention mappings; still not the ELJ path, and still not a universal indexing replacement."

---

## 9. Knowledge-type classification

Per `AGENTS.md` §"Authority by knowledge type", for this document's own material claims.

**INVARIANT** — must hold; violation is a defect. The GFN dependency is one-way. Every quantity crossing the boundary is a difference or a roundtrip, and therefore admits a family of wrong absolute values (G2). `hit_max_lr_dict` is a side effect of `init_optimizers` and `handle_nan` replaces the whole `optimizers_dict`, so any `ModelBundle` extraction must be atomic. `self.config` is mutated at runtime and is scratch space, not settings.

**INTERFACE** — contracted; changing it is a coordinated cross-repo change. The 75 `(module, symbol)` pairs. The `MolCrystalData` instance surface GFN reads off objects. `COMPUTES_REQUIRE_CLUSTER` / `_UNIT_CELL` contents. `DEFAULT_COLLATE_EXCLUSIONS` contents. `mol2cluster` and `construct_radial_graph` **as attributes of their owning classes**. `examples.crystal_search_reporting.batch_compack`.

**DEFAULT** — current value, changeable by decision. The orientation frame GFN currently receives. `std_orientation=False` on the gas-phase legs of both backends, set for *different* reasons per backend. `optimizer_func: 'rprop'`. The MLIP fast paths, on since 2026-08-19.

**WORKFLOW** — the gate ordering in WS-1 (hook before CI); the permissive-first ordering in WS-6; "golden refs regenerate only behind an explicit flag, reviewed as a diff"; "freeze the importer census before any deletion"; the cross-repo lockstep and freeze-window rules (R13).

**DECISION** — the owner's, recorded here, revisable. Demote Modeller decomposition to Phase C and consider cutting it. Promote the boundary freeze to Phase A. Split featurization from flexible molecules. Split WS-5 into a safe half and a cluster-gated half. Defer `git filter-repo` and the `space_group_info` → `.npz` conversion. Scope installation to lockfile + doctor + honest deps. Carve `configs/crystal_searches/` and `normalizer_reduction/` into the fence.

**WORKING ASSUMPTION** — believed, not proved; revisit when contradicted. That `pathtools` actually fails on Python 3.12 in this environment. That the sweep generators reproduce their checked-in output. That gemmi covers cell/SG/ops/components cleanly enough to make WS-10's non-CCDC reader tractable. That the `dataset_creation.rst` old→new script mapping is what name similarity suggests.

**OBSERVATION** — measured, dated 2026-08-23, no policy authority. Everything in §1 and every number in §2, §6 and §7.

**HISTORY** — was true; no current authority. `reporting/papers/`; `tests/old_bad/`; `models/*/old/`; `configs/**/old/**`; the `latent_transform.inverse` call pattern, dead on both sides — **delete it, do not shim it into life**; the stale mixin-collision comment at `data_classes.py:699`; `AGENTS.md:65`'s `pbc_neighbours.py` ownership sentence.

---

## 10. What is not verified

Named so that no one mistakes this document for measurement.

**Verified by execution in this session:** the collection failure and its exit code; the 22.6 s suite; `import mxtaltools.modeller` and `import main` both failing; the removal date of `sample_about_crystal`; the 75-pair / 215-statement / 32-module / 73-file boundary and its one private symbol; `AGENTS.md` being untracked; the config counts and the 876 dangling bases; `configs/crystal_searches/` receiving 368 of 368 recent config additions; GFN's two imports of `examples/`; the eleven directories missing `__init__.py`; the existence and contents of `mol_building.py` and `normalizer_reduction/`; `conftest.py` importing GFN's `gpu_guard`; the 250/38/0 logging census; the 86 hardcoded absolute paths; and each of the six named code defects (bool scatter, unbound `niggli_overlap`, the reversed `sym_utils` argument orders, the duplicated `correct_Ip_directions`, the `randn_like` grad jitter, the ring-bank key mismatch).

**Reported by survey, not re-executed here — treat as leads:** the 1.03°/29.98° grad-jitter distribution; the ~50 % handedness randomness; the 30–50 % eLJ gap between orientation settings; the mixed-batch RDF channel counts (28 / 15 / 28); the 26.5× LUT speedup; the 226 ms cold / 5.9 ms warm `space_group_info` import; the 11836.127 kJ/mol E₀ offset; the conformer prior's kcal/mol figures.

**Explicitly unknown, and what each blocks:**

- **Wheel contents.** Neither `poetry` nor `build` was available. Every `include`/`exclude` consequence in WS-8 is inferred from config, not from an artifact. Settle against `unzip -l dist/*.whl`.
- **Sweep-generator idempotence.** Blocks deleting ~3.5k config files.
- **Whether a Sphinx build currently succeeds.** `conf.py`'s mock list omits `fairchem`, `spglib`, `umap`, `plotly`, `cuequivariance*`, and ReadTheDocs does `pip install .`, which supplies none of the first three.
- **The Z′>1 `mol2ucell` kwarg drop.** Code-evident, **not measured** — a Z′=2 crystal was never constructed. It sits on the contracted MACE acridine route. Measure before relying on the diagnosis, and before pinning any Z′>1 MLIP reference (R11).
- **`torch`'s `weights_only` allowlist membership** for the numpy globals. Blocks the WS-11 scope estimate; the two *blocking* globals are located and confirmed.
- **The duplicate MACE adapter.** `mlip_interfaces/mace_utils.py` is a second, distinct file from `AL_mace_utils.py`. No survey asked which is canonical.
- **CI wall times.** No CI was run. The 22.6 s local figure is the only timing evidence.

**When this document goes stale:** every number here comes from a tree with no CI, and the surveys already disagreed once on a test count. Re-measure at each phase boundary, and treat any number older than the last phase transition as a lead rather than a fact. `AGENTS.md` names `$audit-active-context` as the milestone mechanism; four phase boundaries are defined above and it is invoked at none of them. It should be.

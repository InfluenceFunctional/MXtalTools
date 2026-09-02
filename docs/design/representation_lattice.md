# The representation lattice — CIF ⇄ object ⇄ pose ⇄ unit cell ⇄ cluster

**Status: OBSERVATIONS (measured 2026-08-24) plus a scoped BUILD ORDER (§5) awaiting owner sign-off on B4.**
**Corpus: `D:/crystal_datasets/CSD_dump` (60-CIF draws, two independent shuffles), `mxtaltools/mini_datasets/mini_new_csd.pt` and `tests/datasets/mini_new_csd.pt` (100 crystals: 95 Z′=1, 5 Z′=2), `tests/cif/` fixtures.**
**Interpreter: `C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe`; `ccdc` 3.4.1 live and licensed; `spglib` 2.7.0.**
**Produced by five parallel edge probes and integrated here. Every file:line anchor in this document was re-checked against the working tree by the integrator. No tracked file was modified by the investigation.**

Per `AGENTS.md`, §6 classifies each material claim by knowledge type and names what is unverified. Read §3 first if you read only one section — it is the part that costs energies.

---

## 0. The frame and the verdict

| | representation |
|---|---|
| **R1** | CIF with symmetry operators |
| **R2** | `MolCrystalData`: atoms (`z`, `pos`), `symmetry_operators`, box (`cell_lengths`, `cell_angles`) |
| **R3** | R2 + pose (`aunit_centroid`, `aunit_orientation`, `aunit_handedness`) |
| **R4** | the physical unit cell (`unit_cell_pos`, `unit_cell_batch`) |
| **R5** | supercell / cluster (`pos`, `aux_ind`, `mol_ind`) |

**Direction convention, stated because the probes did not agree on it.** An arrow `X→Y` means *produce Y from X*. So `R2→R3` is **extraction** (the object gains pose parameters) and `R3→R2` is **realization** (`pose_aunit` writes `pos`). One edge report called realization "forward"; this document does not.

**The chain has four edges, hence eight directed rows — not ten.** Six measured shortcuts bring the table to fourteen.

**Headline.** *The arithmetic of this lattice is sound everywhere it was measured.* Round-trip residuals are float32 round-off: 2.2e-5 Å over 19 312 unit-cell atoms, 6e-7 fractional on centroids, 2e-6 rad on orientations. **Every serious defect is a precondition defect, not an arithmetic defect** — fifteen of them are silent, and six land at 0.5–13 Å on real crystals. The contract this lattice runs on is almost entirely unwritten and unchecked.

---

## 1. The lattice table

*Table 1: every directed conversion between R1–R5 that exists in the package, with the strongest executed measurement available for it. Fidelity cells state the deviation and its unit; `n` is the number of real crystals the cell rests on. Sources: 60 CSD CIFs (two independent draws), the 100-crystal `mini_new_csd` fixture, and 19 written-and-re-read CIFs. The reference bar throughout is `crystal_rebuild_checks` (`featurize_cif_chunks.py:226`): max nearest-neighbour distance < 0.05 Å, mean < 0.01 Å, bijective match. "not measured" means no probe attempted it.*

| # | direction | function (file:line) | exists? | measured fidelity (units) | n | failure rate | what is lost |
|---|---|---|---|---|---|---|---|
| 1 | **R1→R2** | `extract_crystal_info` `dataset_utils/construction/featurize_cif_chunks.py:196` → `extract_crystal_data` `construction/featurization_utils.py:80` → `init_zp1_crystals` `:578` → `instantiate_crystal` `:513` | YES — **CCDC-only**, no other CIF reader in the package | not invertible alone; measured jointly as row 2 | 60+60 | ingest yield **19/60 (31.7 %)** and **28/60 (46.7 %)** in two draws — a *filter*, not an error (see D1); 0 crashes | occupancy, disorder, ADPs, R-factor, temperature, radiation, deposition no.; `z_value` (`:90`) and `space_group_setting` (`:94`) extracted then never passed on — dead; `zzp_cost`/`contact_overlap_cost` (`:416`) have no reader |
| 2 | **R2→R1** | `MolCrystalData.write_cif` `data_class_methods/crystal_ops.py:1985` → `write_asymmetric_unit_cif` `common/cif_io.py:258` → `cif_block_from_crystal` `cif_io.py:146` | YES — new; `mode='asymmetric unit'` only | cell lengths max **1.0e-6 Å**, angles max **7.0e-6 °**; sg 19/19, operator list *in order* 19/19. Full R2→R1→R2: aunit `pos` max **1.90e-5 Å** (mean 1.13e-5), rebuilt cell max **3.49e-5 Å**, cell params **bit-identical**, centroid max 1.79e-7 frac, orientation max 5.25e-6 rad | 19 CIF + 100 fixture | **0/19 and 0/100 for Z′=1**; **5/5 FAIL for Z′>1** (re-read as `filter:Polymer`) | `x` (partial charges), `smiles`, `fingerprint`, `is_well_defined`, **all** `aunit_*`; occupancy hardcoded `1.000` (`cif_io.py:252`); **no `_geom_bond` loop** → any consumer must re-perceive connectivity from coordinates. *The probe's "2/19 (10.5 %) inverted chirality" is **unreproduced**: on 26 deprotonated CSD crystals the integrator got 0 comparable pairs, because CCDC cannot derive a SMILES from a heavy-atom-only bond-less file at all. The mechanism — no bond loop — is confirmed; the 10.5 % figure is not. This is the same re-perception problem as §1.3 of `cif_reader_design.md`, and it is the writer's one real gap* |
| 3 | **R2→R3** (extract) | **no single function.** `extract_aunit_orientation` `crystal_building/utils.py:491`; the centroid inverse is *inline* at `utils.py:564`, reachable only via `parameterize_crystal_batch` `:437` / `reparameterize_unit_cell` `crystal_ops.py:767` — **both require R4** | PARTIAL | direct route (caller writes `fractional_transform(heavy_centroid(pos), T_cf)`): centroid max **5.96e-7 frac** (mean 7.07e-8); orientation on the non-degenerate 99.3 %: max **2.68e-4** Frobenius ≈ 1.9e-4 rad, mean 2.4e-6 | 5 700 trials (95 crystals × 30 poses × 2 handedness); 95 stored | orientation **40/5700 = 0.702 %** catastrophic (up to a full 180°, **1.94 Å** atom RMSD); handedness 38/5700 = 0.667 %; centroid **0/5700**. On *stored real* parameters **0/95** | the rotvec **representative** (canonicalization picks one of two); the incoming cartesian frame |
| 4 | **R3→R2** (pose) | `pose_aunit` `data_class_methods/crystal_building.py:116` → `get_aunit_positions` `crystal_building/utils.py:756` → `align_mol_batch_to_standard_axes` `:597` | YES | exact by construction. **Idempotent at `std_orientation=True`** (Δ 4.4e-5 Å on repeat); **NOT idempotent at `False`** — successive calls move atoms **7.05 → 7.53 → 7.78 Å** | 95 | 0 | the incoming `pos` frame (many-to-one) when `std_orientation=True`; `override_handedness` poses correctly but **does not update `aunit_handedness`**, leaving the object describing a pose it does not hold |
| 5 | **R3→R4** | `mol2ucell` `crystal_building.py:226` → `pose_aunit` + `build_unit_cell` `:135` → `aunit2ucell` `crystal_building/utils.py:646` | YES | vs the CCDC-packed cell: max nn **2.21e-5 Å**, mean 6.90e-6, median 6.02e-6. R3→R4→R3→R4 closure max **1.88e-5 Å** | 28 CIF (6 366 ucell atoms) + 100 fixture (12 946) | **0/28** against `crystal_rebuild_checks` — the 0.05 Å bar cleared by ~2 300× | **Z-to-one by design** (§2, D2); the aunit's absolute cartesian frame *and* translation (recentred on the heavy-atom centroid); centroid **clipped, not wrapped**; float32 only |
| 6 | **R4→R3** | `reparameterize_unit_cell` `crystal_ops.py:767` → `parameterize_crystal_batch` `crystal_building/utils.py:437` → `identify_canonical_asymmetric_unit` `:564` + `extract_aunit_orientation` `:491` | YES — as a **canonicalizing projection**, not a set-theoretic inverse | centroid max **1.17e-5 Å** (2.98e-7 fractional), orientation max **1.29e-4 rad** (worst: CABUND, sg 33), handedness 100/100 | 100 fixture + 28 CIF | 0 exceptions; **4/95 return `is_well_defined=False`** (all sg 145/148, `ASYM_UNITS=[1,1,1]`) | *which* of the Z images you came from; the rotvec branch — `cleanup_invalid_rotvecs` `:537` **fabricates** `rotvec=[1,1,1], θ=π` near trace ≥ 1; the box (R4 alone is not self-contained); **mutates `self.pos`** on the Z′=1 path only (`utils.py:472`) |
| 7 | **R4→R5** | `build_cluster` `crystal_building.py:141` → `ucell2cluster` `crystal_building/utils.py:29` → `get_cart_translations` `:384` → `_pare_cluster_molwise` `:87` | YES | lossless-in-arithmetic. **151.0×** atom inflation (2 652 aunit → 12 034 ucell → **1 817 694** cluster atoms); `aux_ind` 0/1/2 = **0.15 % / 7.90 % / 91.96 %** (92 % of every cluster is edgeless ballast) | 95 | 0; but `supercell_size=10` **binds and silently truncates 4/95 (4.2 %)** — required translations/dim median 5, **max 12** | the per-replica integer `(nx,ny,nz)` offset — computed as `atomwise_translation`, **discarded at `utils.py:336`, unrecoverable** (the blocker named at `examples/profile_mlip_energy.py:45`); `num_atoms` keeps **aunit** semantics on a cluster; `aunit_batch` goes stale; `unit_cell_pos` becomes a frozen snapshot |
| 8 | **R5→R4** | **NO named function.** Three unnamed inverses | MISSING by name / EXACT in arithmetic | (A) leading-block slice `pos[ptr[g] : ptr[g]+num_atoms[g]*sym_mult[g]]`: **bitwise 0.000e+00 Å**, 95/95, stable across `supercell_size` and re-entrancy. (C) heavy-atom-centroid-in-cell test: **0.0 Å**, 95/95 (Z′=1) and 5/5 (Z′=2). (B) `aux_ind==0` is **not** the unit cell — it is the aunit, matching only **1/95** | 95 + 5 | inverse A **fails 5/5** on the joined Z′>1 route (21.5–29.1 Å) — `join_zp1_cluster_batch` `:80` discards the subunit `ptr` | `de_cluster` `:149` returns the **aunit**, not the cell |
| 9 | **R1→R4** (shortcut) | `extract_crystal_data` `featurization_utils.py:113,117` reads the CCDC-packed cell straight into `unit_cell_coordinates` | YES | this **is** the ground truth row 5 was scored against: **2.21e-5 Å** | 28 | 0 | consumed by `reparameterize_unit_cell`, then dropped from the object |
| 10 | **R2→R4** (shortcut) | `build_unit_cell` `crystal_building.py:135` called directly | YES, **unguarded** | building from an **unposed** `pos`: max **11.746 Å**, mean 8.370, median 8.611 — **20/20 over the bar, no exception**, and `reparameterize` then returns `is_well_defined=True` on the wrong cell | 20 | **silent 20/20** | — (this is gap S2) |
| 11 | **R3→R5** (shortcut) | `mol2cluster` `crystal_building.py:168` | YES | Z′=1 route matches the staged build; Z′>1 route works 5/5 | 100 | 0 | `std_orientation` is **honoured here** (`:214`, `:222`) and **dropped in `mol2ucell`** (`:230`) — gap S1 |
| 12 | **R5→R3** (shortcut) | `de_cluster` `crystal_building.py:149` (geometry only) + `reparameterize_unit_cell` (needs R4) | PARTIAL | `de_cluster` vs unit-cell molecule 0: **bitwise 0.0**; vs the original R3 aunit: max **1.53e-5 Å**, mean 2.02e-6 (that residue is `aunit2ucell`'s float32, not this edge) | 95 | 0 | nulls `aux_ind`/`mol_ind`/`edges_dict`; returns **no pose parameters** |
| 13 | **R5→R2** (shortcut) | free — `_instantiate_cluster` clones at `crystal_building/utils.py:333` | YES | box, `symmetry_operators`, `sg_ind`, `sym_mult`, `z_prime`, `identifier`, `fingerprint`, `T_fc`, `T_cf` all **bitwise identical** | 95 | 0 | nothing |
| 14 | **R1→R5 / R5→R1** | none | MISSING | not measured | — | — | — |

---

## 2. Verdict per edge

| edge | verdict |
|---|---|
| **R1 ⇄ R2** | **Lossy-but-usable, Z′=1 and Z′>1 alike — closable for the first time.** ⚠ *Corrected by the integrator 2026-08-24: the edge probe reported "one-way for Z′>1, 0/5 survive". **Refuted.** All 5 Z′>1 fixtures (HAQXUW, AVENUL, IHEKOV02, ABESEG01, HEFMOV) write and re-read correctly — space group 5/5, full aunit site count 5/5 (42/48/84/26/70, both components). What the probe measured was **re-ingestion through `crystal_filter`**, which drops Z′>1 at `max_z_prime=1` and can classify a bond-less file as `Polymer`. That is a filter outcome, not a writer failure.* The forward is a CCDC licence dependency wearing a parser's clothes: it filters 53–68 % of input before an object exists. The reverse is new, numerically exact on the lattice, and loses *chemistry provenance* — not lattice. |
| **R2 ⇄ R3** | **Bidirectional, exact to 6e-7 fractional and ~2e-6 rad on 99.3 % of poses — and wrong by a mirror on the other 0.70 %.** Not "lossy at the margins": on the failing fraction it is off by 180° and 1.94 Å. Caveat that matters: the only *shipped* inverse routes through R4 and is therefore a projection (D2), so the direct inverse must be written by the caller. |
| **R3 ⇄ R4** | **Bidirectional and exact, up to a deliberate Z-to-one quotient.** The tightest edge in the lattice — residual is pure float32. `R4→R3` is a canonicalizer, and that is intended, proven constructively. |
| **R4 ⇄ R5** | **Forward lossless; inverse missing-by-name but exact-by-arithmetic.** "Bidirectional in the arithmetic, one-way in the API." One field is genuinely unrecoverable: the per-replica cell offset. |

### Where the probes disagreed, adjudicated

**D1 — ingest yield: 31.7 % vs 46.7 %.** Two independent 60-CIF draws with incompatible filter tallies (19 vs 22 `Z' != mol components`; 12 vs 9 `Z' < 1`). Neither was checked against the other. **Unresolved. Quote the range 32–47 %, not a point value, until one fixed shuffle is re-measured.**

**D2 — "R4 contamination" vs "the intended quotient".** Both probes are right and they measured the same thing. The R3–R4 probe proved the quotient constructively: all **8** unit-cell images of EBEXUI (sg 15, Z=8) parameterize to 8 distinct R3 vectors, build the *same* R4 to **7.5e-6 Å**, and return to the *same* canonical R3. The R2–R3 probe measured the consequence: routing an R2↔R3 question through `reparameterize_unit_cell` gives centroid failures **31/950**, max **0.978 fractional** — a different molecule. **Ruling: the quotient is intended; using it as an R2↔R3 inverse is not.** Anything comparing two R3 parameter vectors for equality must canonicalize first.

**D3 — is the canonical-aunit pick stable?** Yes: **0/95** flips under a 1e-3 fractional centroid nudge; margin from the selected centroid to the nearest `ASYM_UNITS` face is median 0.069, 5th pct 6.1e-3, min 4.08e-4. The competing result — **1/95** fails by >0.1 Å under 1e-4 Å *cartesian* noise — is a different mechanism: the **frame** is unstable, not the pick (gap S3). No contradiction.

**D4 — the stored `is_well_defined` is stale.** `mini_new_csd.pt` records `True` for BOGFUA, MUZSUW, ESOZEU and JUKLIM (sg 145/148), which recompute `False`. **Source of truth is the code.** Do not filter training data on the stored flag without recomputing.

**D5 — asymmetric return shape, reconciled by reading.** `parameterize_crystal_batch` returns `well_defined_asym_unit_list` as a **python list** and handedness of shape `(n,)`, passed straight through on the Z′=1 path (`crystal_building/utils.py:486` → `crystal_ops.py:817`). The Z′>1 path builds a **bool tensor** (`crystal_ops.py:798`) and handedness `(n, max_z_prime)`. So `iwd.sum()` raises `AttributeError: 'list' object has no attribute 'sum'` **for Z′=1 only**, and `handedness[:, :1]` raises `IndexError: too many indices` for Z′=1 only. Both probes were right about different halves.

---

## 2b. THE STANDARD-ORIENTATION CONVENTION — owner-stated, verified, unwritten until now

**INVARIANT (owner decision, 2026-08-31).** When posing a molecule and building a crystal,
**always assume standard orientation**: the cell and pose parameters are defined relative
to a standard-oriented asymmetric unit. Anything that poses a crystal must therefore derive
its parameters under that convention, never pose deposited coordinates directly.

**Verified.** Featurising a CIF and re-posing the derived parameters against the deposited
unit cell (the latter built by the native reader, itself verified against CCDC to 8.6e-15 Å):

| pose convention | deviation from the deposited cell |
|---|---|
| `std_orientation=True` | **0.0000 Å** |
| `std_orientation=False` | 2.5783 Å |

*Table: NUKCOL, the one Z′=1 fixture with no hydrogens and therefore the only one whose
atom count survives the featurizer's deprotonation for a like-for-like comparison. The
other ten Z′=1 fixtures give a shape mismatch, not a disagreement — widening this needs an
H-strip on the reader side.*

**WORKING ASSUMPTION, owner-stated and explicitly not yet true:** historic datasets do not
all comply. There is currently no way to tell a compliant stored crystal from a
non-compliant one, which is what makes the convention dangerous rather than merely
undocumented.

**Two known limits.**

- **High-symmetry molecules are genuinely ambiguous.** When the two largest principal
  moments are close, standard orientation is not unique. Measured (gap S3): on one fixture
  **1e-4 Å of noise flips the handedness and moves the rebuilt cell 2.4 Å**, with
  `is_well_defined` staying `True` throughout — it covers the centroid-in-box ambiguity and
  never the frame. The owner's position is to accept this for now.
- **A correction, recorded so it is not repeated.** An audit asking "is the stored `pos`
  standard-oriented?" returns 0/95 and is **the wrong question** — `pos` is the deposited
  crystal-frame position, so it never is. The convention lives in the *parameters*.

**FOR LATER — the owner's own framing: "more automatic, or in the manual."** Two separable
pieces, neither started:

1. *Automatic.* A compliance check — pose the stored parameters both ways, compare against
   the deposited cell, record which convention reproduces it. Cheap, and it would let a
   historic dataset be audited rather than assumed. It could then be asserted at load.
2. *Manual.* This section is the seed; the convention belongs wherever the data model is
   described for a reader, not only in a design doc.

---

## 3. The gaps, ranked by whether failure is silent

**Silent first.** Ordered by exposure × magnitude. Every entry was executed; the three marked **[read-verified]** were additionally confirmed by the integrator reading the lines.

### S1 — `mol2ucell` drops `std_orientation` for Z′>1 — **LATENT in production, LIVE for anything read from CIF** ⚠ *downgraded 2026-08-24*

> **Correction.** This gap was first written as "every Z′>1 crystal currently gets its MLIP energy and its LJ energy from different geometries." **That overstates it for the production path**, and the owner flagged it: the delta would have been noticed.
>
> The flag only matters when `pos` is *not* already standard-oriented, because `std_orientation=True` just calls `align_mol_batch_to_standard_axes` first — **and that operation is idempotent, measured to 3.6e-5 Å** (first application on stored CSD crystals moves atoms 5.79–12.33 Å; second application moves them 1.6e-6–3.6e-5 Å).
>
> **GFN's generated crystals carry standard-oriented `pos`** (owner-confirmed), so `True` and `False` coincide and the two legs agree. The 6–11 Å figures below were measured on *stored CSD* fixtures, whose `pos` is not standard-oriented — the wrong population for the production claim.
>
> **It is still a real defect, and it becomes live the moment a non-standard-oriented conformer reaches a Z′>1 crystal — which is exactly what a CIF reader produces.** Fix the line; treat "aunit `pos` is standard-oriented" as an invariant that is currently assumed and nowhere stated or checked.
>
> *(Also checked and ruled out: the owner's alternative hypothesis that the aunit is posed before reaching the MLIP. It is not — `analyze_crystal_batch` goes straight from `instantiate_crystals` to `analyze(std_orientation=False)` with no intervening pose.)*

`crystal_building.py:230` is `zp1_batch.pose_aunit()` with no argument. Lines `:214`, `:222` and `:237` all forward the flag. Measured: `mol2ucell(False)` vs `mol2cluster(False)` aunits differ by **6.72 / 6.25 / 10.31 / 6.71 / 9.63 Å** on all five Z′=2 crystals; Z′=1 controls **0.0000**. `analyze()` routes cluster computes (elj/lj/es) through `mol2cluster` (`crystal_analysis.py:321`) and MLIP computes (mace/uma) through `mol2ucell` (`:337`), and GFN's `analyze_kwargs` requests `std_orientation=False`. **Every Z′>1 crystal currently gets its MLIP energy and its LJ energy from different geometries.** Same class as the `std_orienation` typo already recorded in that file, live on a different axis.
*What would catch it:* forwarding the flag (one line) plus a test asserting the two routes pose the same aunit for Z′>1.

### S2 — `build_unit_cell` assumes `pos` is already posed; nothing records whether it is
`aunit2ucell` (`crystal_building/utils.py:646`) enforces the centroid but takes the orientation from `pos` as-is; the assumption is stated only in a comment at `:39-41`. Unposed build vs posed: max **11.746 Å**, mean 8.370, **20/20** over the bar, no exception, and `reparameterize_unit_cell` then succeeds on the wrong cell and reports `is_well_defined=True`.
*What would catch it:* `pose_aunit(std_orientation=True)` is idempotent on already-posed data (0.0000 Å shift), so calling it defensively inside `build_unit_cell` is free.

### S3 — `aunit_handedness` is the sign of a LAPACK eigenvector convention; it flips under 1e-4 Å
`compute_Ip_handedness` (`common/geometry_utils.py:748`) takes `det` of `linalg.eigh`'s eigenvectors; `align_mol_batch_to_standard_axes` (`crystal_building/utils.py:597`) re-runs `eigh` on the stored conformer at rebuild time and assumes the same sign returns. On KUTMIW (principal moments 132.4 / 448.4 / 522.3 — the top pair only 14 % apart), 1e-4 Å isotropic noise flips handedness and the rebuilt cell moves **2.396 Å**; `is_well_defined` stays **True** throughout, because it covers only the centroid-in-box ambiguity, never the frame. Rate at 1e-3 Å noise across 95 crystals: median 3.86e-3 Å, p90 6.21e-3, **max 2.40 Å, 1/95 (1.1 %) over 0.1 Å**. This is also the entire 0.702 % pose-degeneracy of Table 1 row 3 — same crystal.
Root cause pinned: `overlap_threshold=1e-5` in `correct_Ip_directions` (`geometry_utils.py:324` — note `:275` defines the **same function** and is shadowed dead code), anchored on `batch_get_furthest_node_vector` (`:394`), which carries its own in-repo TODO at `:241` ("SET THIS TO A MORE STABLE ANCHOR"). Population exposure: `min|overlap| < 1e-5` for **1/95**, `< 1e-3` for 2/95, `< 1e-2` for **18/95 (18.9 %)**.
*What would catch it:* the rebuild-and-assert check (B1).

### S4 — `assign_aunit_centroid` **clips** instead of wrapping, and `pose_aunit` does neither
`crystal_ops.py:34`: `values.clip(min=0, max=1-1e-4)`. Input `[1.2, -0.3, 0.5]` → `[0.9999, 0.0, 0.5]` (a wrap gives `[0.2, 0.7, 0.5]`). A centroid shifted by +1.0 — a *physically identical* crystal — builds a cell **3.83–8.43 Å** away, and +2.0 clips to the same corner. `get_aunit_positions` (`:756`) reads `aunit_centroid[:, :3]` raw and does not clip at all, so at centroid+1.0 the **posed aunit sits 5.98 Å outside its own unit cell**: `pos` feeds the gas-phase MLIP leg and `aux_ind==0`, `unit_cell_pos` feeds the periodic leg, and they describe different molecules. The guard exists — `validate_cell_params_ranges` `crystal_ops.py:715` raises `AssertionError: Aunit centroids must be less than 0.999` — and is **not wired into** `mol2ucell`/`build_unit_cell`; only `crystal_rebuild_checks` calls it, at featurization time. A generative sampler emitting out-of-range centroids gets saturation, not periodicity, and no error.

### S5 — `max_z_prime` zero-padding is read as a real Z′ slot; a Z′=1 crystal's cluster depends on its batch-mates
`mol2cluster` (`crystal_building.py:176-211`) reshapes `aunit_centroid` to `(num_graphs*max_z_prime, 3)` and takes `dists.amax` as `zp_buffer` with **no `z_prime` mask**; the pad is `0`, i.e. the cell origin. Same crystal alone vs collated with one Z′=2: **1.25×–2.48× more atoms** (EBEXUI 24 408 → 36 504; NILKIC 16 692 → 41 184; `aux_ind==1` 864 → 2 862). LJ at a 6 Å graph cutoff is invariant (≤ 6e-5 kJ/mol), so today this is an OOM/batch-sizer hazard; it becomes a correctness hazard for anything consuming the cluster rather than `edges_dict`.
Pad fill is inconsistent across four sites: `_pad_tensor` fills orientation and handedness with **1** (`data_classes.py:751-758`, `crystal_ops.py:1748`); `collate_data_list` `F.pad` fills **0** (`dataset_utils/utils.py:58-60`); `reparameterize_unit_cell` `new_zeros` fills **0** (`crystal_ops.py:786-788`); on-disk data is **0**. Handedness 0 is not a legal handedness (S6). `canonicalize_aunit_order` (`crystal_building/utils.py:787`) assumes a 1-fill for its sort and a 0.5-fill for its gather. Parameter *width* is also a function of batch composition: stored objects carry no `max_z_prime`, so `dataset_utils/utils.py:74-79` sets it from `z_prime.amax()` and truncates — 3 columns in an all-Z′=1 batch, 6 if one Z′=2 crystal is present — and `collate_data_list` **mutates the input list elements in place** to match, permanently.

### S6 — `aunit_handedness` outside {−1,+1} deforms molecules, and 0 is reachable live
Handedness lands in `eye[:, 0, 0]` at `crystal_building/utils.py:614`, making a non-orthogonal alignment target: `extract_rotmat` then yields a scaling, not a rotation. Measured bond-length changes: h=0 → **8.26 Å**, h=+2 → **9.09 Å**, h=−3 → **18.22 Å**; positions all finite, no error, and extraction cheerfully returns {−1,+1}. h=0 gives a cell **3.8700 Å** wrong and non-bijective. The live route: `_transform_aunit_params` (`crystal_ops.py:819`) returns `det * h_old = 0.99999994`, and `.long()` **truncates to 0** (`round().long()` gives 1 and rebuilds bitwise).

### S7 — atom **order** between aunit and unit-cell images is assumed everywhere and checked nowhere; the one check is commented out, **and it is wrong as written**
`parameterize_crystal_batch:452` does `z[batch==i].repeat(sym_mult[i])`; `:456` and `:466` reshape to `(sym_mult, n_atoms, 3)`. Permuting unit-cell rows (same atoms, bijection intact) moves the centroid `[0.415, 0.321, 0.137] → [0.459, 0.446, 0.424]` with `is_well_defined=True` and no exception. The only guard that would catch it is commented out at `featurize_cif_chunks.py:261`; with the aunit's `z` permuted, the surviving geometric test passes verbatim (`nn max 0.000e+00`, `single_matches True`) while **40/216 atoms carry the wrong element** and `rebuild_successful` is True.
**Do not reinstate line 261 as written.** `z.repeat(sym_mult)` **tiles** `[z_zp0, z_zp1, z_zp0, …]` while `aunit2ucell` produces **blocked** order `[z_zp0 × Z][z_zp1 × Z]` — and CCDC's `unit_cell.components` are blocked too (8/8 match blocked, 4/8 interleaved). At the CIF stage, before `overwrite_conformer_to_zp` (`featurize_cif_chunks.py:173`) normalises the orderings, the as-written check fires falsely on **ADEYAN** (8 mismatches) and **AGESOX** (12/128 = 9.4 %) — ~25 % of the Z′>1 sample — while the blocked ordering gives 0 in both. **The same tiling bug is live and uncommented at `crystal_ops.py:1909`**, where it labels the sites handed to spglib inside `compute_standard_cell`.

### S8 — `reset_sg_info` substitutes standard operators — ⚠ **REFUTED as a live hazard 2026-08-24**

> The gap was written as "reached automatically from `collate_data_list` whenever `symmetry_operators` is absent". **It is not reachable at all.** The guard at `dataset_utils/utils.py:83` is `if not hasattr(batch, 'symmetry_operators')`, and `symmetry_operators` is a **property** (`data_classes.py:842`) that returns `None` when unset — so `hasattr` is *always* True and the branch never executes. Verified directly.
>
> Two consequences, opposite in sign:
> - **Good:** collation never silently substitutes standard operators. Measured: **10/10 non-standard crystals keep their own operators through `collate_data_list`**, which is what the CIF reader depends on.
> - **Bad:** the fallback that was meant to supply operators when absent is a dead branch — the `feedback_no_runtime_gate_on_a_retired_key` pattern. A crystal with no operators reaches the builder holding `None`. Not currently reachable, because `MolCrystalData.__init__` (`crystal_ops.py:50-62`) always sets them, but the guard is not doing the job it appears to do.
>
> **Not fixed.** Changing `hasattr` to an `is None` test would make a dead branch live, which is its own behaviour change and needs its own justification. Recorded, not actioned.
>
> Separately, every one of the ~20 `reset_sg_info` call sites passes a *deliberately different* `sg_ind` (P1 for big cells, sampled groups for fakes). Nobody calls it with the crystal's own `sg_ind` expecting a no-op, so the "reads as a no-op" framing overstates the exposure. The 4.044 Å figure is real but describes a call nobody makes.

### S8b — the original measurement, retained as evidence
`crystal_ops.py:1786-1794` unconditionally overwrites `symmetry_operators` with `SYM_OPS[sg_ind]` **and** `nonstandard_symmetry` with `False`, erasing the evidence. On the 50 nonstandard crystals in the fixture: median 0.000 Å, **max 4.044 Å**, **>0.5 Å in 19/50 = 38 %** (NOKCAT01 4.044, JADLIJ 4.038, FUJXAJ 4.034, NAQVUA 3.918, FACXIQ 3.871, AFEGIF 3.849). Reached automatically from `collate_data_list` (`dataset_utils/utils.py:82`) whenever `symmetry_operators` is absent from the batch. Corroborated independently: substituting standard ops on 20 nonstandard crystals changes the built cell in **7** (2.27–4.04 Å, all non-bijective, all silent); the other 13 are pure reorderings and build the identical cell — so *"ops differ from standard" is not by itself diagnostic; you must rebuild to know.* Nonstandard rate here: 7/19 (36.8 %) on CIFs and 50/100 (49 %) on the fixture, bracketing the known 37.5 %.

### S9 — `build_cluster` on a cluster; the guard is commented out
`crystal_building.py:13-14` carries `# assert self.aux_ind is None, "Not implemented for cluster objects"`, commented. Re-clustering returns without error and gives a *different* cluster (179 328 → 175 884 atoms) because `cc_centroids` at `utils.py:41` now averages the whole cluster rather than the aunit. `z` stays bitwise equal and the leading block still inverts exactly, so it looks fine.

### S10 — cluster atom **order** is a function of the batch
`utils.py:400` takes `.amax()` of `ceil(cutoff_distance / box_lengths)` across the **whole batch**, so `actual_supercell_size` — and hence the within-shell tie order out of `generate_sorted_fractional_translations` (`:17`) — depends on batch composition. Byte-identical R4, solo vs in a batch of 20: **22 248 / 24 408 atoms (91.1 %) at a different index**, max 86.95 Å at a given index, same point set to 1e-4 Å, identical `aux_ind` histogram. Physics is safe (LJ varies **3.05e-5 kJ/mol** on −396.58 — float32 summation order), but **any index into `cluster.pos` persisted beyond the batch that produced it is unsound.** Only `mol_ind`, `aux_ind` and the leading unit-cell block survive re-batching.

### S11 — unequal-size Z′ components are re-cut at the wrong atom
`num_atoms // z_prime` at `crystal_building.py:23`, `:43` and `crystal_building/utils.py:370`. A CH₄(5)+C₂H₅(7) Z′=2 crystal splits `[6, 6]`; the `assert len(zp1_batch.pos) == len(new_batch)` at `:29` passes because the total is divisible. `mol2ucell` then runs and produces a `(24, 3)` cell with no exception. Gated only by `cocrystal_check` (`featurization_utils.py:642`) inside `process_chunk`; `MolCrystalData.__init__` accepts a heterogeneous `molecule=[...]` list with no check.

### S12 — `radius` for Z′>1 is the **sum** of component radii, and it sets the cluster cutoff
`featurization_utils.py:528` and `crystal_ops.py:96-100`. Consumed geometrically at `crystal_building/utils.py:45` (supercell selection) and `:104` (`cluster_cutoff = (cutoff + 2*radius + 0.1)**2`). Against the true max |r − heavy centroid| of the assembled Z′=2 aunit: **−19.2 %, +50.8 %, +36.9 %, +23.0 %, +22.2 %** — the sign is not fixed, so it is **not conservative**; the −19 % case truncates a neighbour list. `split_to_zp1_batch` (`:44-46`) then assigns the summed `radius`, `mass` and `mol_volume` to each single-molecule subunit unchanged.

### S13 — four dead branches that read as guarantees
- **`clean_cell_parameters(canonicalize_orientations=True)` is an identity** `crystal_ops.py:1056-1060` **[read-verified]** — it slices `aunit_orientation` per-Z′ and re-concatenates. Executed: `[0.3, 0.4, −1.0]` returns unchanged, z still negative; `canonicalize_rotvec` would give `[−1.386, −1.848, +4.620]`. Since `parameterize_crystal_batch` *does* canonicalize (`utils.py:481`), **two rotvec conventions are live in one pipeline**.
- **The `identify_canonical_asymmetric_unit` tie-break is unreachable** `crystal_building/utils.py:583` and `:587` **[read-verified]** — `len(set(dists)) < len(dists)` on a torch tensor is always False (iterating a tensor yields distinct 0-d objects), so the per-dimension ladder at `:584-591` never runs and ties fall to `argmin` (first index).
- **`_molwise_indexing` `utils.py:353` is dead code** and is the only *written* definition of `aux_ind` — it emits a two-valued label, contradicting the live three-valued one (`utils.py:347-348` and `:155`: 0 = aunit, 1 = in the convolutional field, 2 = outside, edgeless).
- **`z_value` (`featurization_utils.py:90`), `space_group_setting` (`:94`), `zzp_cost`/`contact_overlap_cost` (`:416`)** are extracted and never consumed.

### S14 — `unit_cell_batch` / `unit_cell_mol_ind` gate on the wrong key
`data_classes.py:899` and `:903` both test `'unit_cell_pos' in self._store`: with `unit_cell_pos` set but `unit_cell_batch` absent you get `KeyError: 'unit_cell_batch'` from a property that promises `None`; with `unit_cell_batch` present and `unit_cell_pos` deleted it returns `None` while the data exists.

### S15 — `is_well_defined` is structurally unreachable for ~47 % of space groups
It is `True` iff **exactly one** of the `sym_mult` centroids lands in the `ASYM_UNITS[sg]` box (`utils.py:564`, `find_coord_in_box_torch` `:426`, `epsilon=0`). **108/230 space groups have `ASYM_UNITS == [1,1,1]`** — the whole cell, a placeholder for non-parallelepiped aunits (`constants/asymmetric_units.py:151`) — so for 107 of them (all but sg 1, whose `sym_mult` is 1) every image is inside the box and the flag is **always False**. `dataset_manager.py:783` filters on it and `tests/test_data_classes_basics.py:216-222` asserts the round trip only where it is True: the shipped contract quietly excludes half the space groups from ever being trainable. Fixture rate 4/95 = 4.2 %; the stored flag disagrees with recomputation on exactly those 4 (D4).

### Loud failures — these are safe

*Table 2: violations that raise. Listed so nobody re-derives them; none of these is a gap. Executed on the fixture and on deliberately mutated objects.*

| invariant violated | raise site | message |
|---|---|---|
| `n_ucell == num_atoms * sym_mult` | `crystal_building/utils.py:455` (the **boolean mask**, not the reshape at `:456`) | `IndexError: The shape of the mask [216] at index 0 does not match the shape of the indexed tensor [213, 3]` — uncaught, ~1 % of crystals (e.g. HOBFAH) |
| `sym_mult != len(symmetry_operators)` during cell build | `crystal_building/utils.py:695` | `RuntimeError: einsum(): subscript n has size 4 for operand 1 which does not broadcast with previously seen size 8` |
| lied `sym_mult` on the cluster path | `crystal_building/utils.py:160+` | `RuntimeError: The size of tensor a (108) must match the size of tensor b (216)` |
| `build_cluster` on Z′>1 | `crystal_building/utils.py:249` | `AssertionError: Don't try cluster construction for Z'>1 crystals` |
| direct `build_unit_cell` on a joined Z′>1 object | `crystal_building/utils.py:689` | `RuntimeError: einsum(): subscript j has size 7 ... does not broadcast with previously seen size 4` |
| `z_prime` > actual components | `crystal_building.py:29` | bare `AssertionError`, no message |
| float64 batch anywhere | `crystal_building/utils.py:677` | `RuntimeError: expected scalar type Float but found Double` — the whole lattice is float32-only (`:677-678`, `:684` hard-code it) |
| `sg_ind ∉ SYM_OPS` on write | `common/cif_io.py:180` | `KeyError: space group 999 is not in SYM_OPS` |
| `len(ops) != sym_mult` on write | `common/cif_io.py:126` | `ValueError: crystal 0 carries 4 symmetry operators but sym_mult is 7` |
| `de_cluster` on a non-cluster | `crystal_building.py:166` | `RuntimeError: can't de-cluster - this is already not a cluster` |
| `z_prime.amax() > max_z_prime` at collation | `dataset_utils/utils.py:80` | `AssertionError: Batch max z prime must agree with parameterization` |

**Two write-side preconditions are silent, not loud:** a **stale `T_cf`** (cell changed without re-running `box_analysis`) writes `_cell_length_a 22.812` against unchanged fractional coordinates — 8× the cell volume, and the file reads back as a valid crystal; and an **unposed `pos`** writes negative fractionals, which are legal CIF, producing a physically absurd crystal that parses cleanly. `symmetry_operators = None` silently substitutes `SYM_OPS[sg_ind]` (`cif_io.py:111`), which for the 37–49 % nonstandard population emits operators that do not match the coordinates. Also latent: `identifier` apostrophes are emitted unescaped inside single quotes (`_chemical_name_common 'it's a 'test''`) — strictly malformed CIF 1.1; `ccdc` tolerated all four probes, a stricter parser will not.

---

## 4. The bare-unit-cell case (R1 without symmetry operators)

**Verdict: viable, and measured end to end — spglib inference is not merely usable, it is exact on CSD-quality cells.** The full route bare-CIF → R2/R4 was executed on 19 crystals with **zero failures**.

*Table 3: recovery of symmetry and geometry from an ASE-written "unit cell" CIF (cell + all Z·N atoms + `_space_group_IT_number 1` + one identity operator, nothing else) via spglib, on 19 real CSD crystals. Reference is the CCDC-derived R2/R4 the file was written from.*

| quantity | result |
|---|---|
| reached R2 | **19/19**, zero failures |
| space-group number | **19/19** |
| operator set exact (rotations + translations mod 1) | **19/19 — including all 7 nonstandard settings** |
| Z′ | 19/19 |
| atoms per molecule / element multiset | 19/19 / 19/19 |
| operator → cell-atom match error | max **4.48e-6 Å** |
| recovered cell lengths / angles | max **2.60e-6 Å** / **1.05e-5 °** |
| rebuilt unit cell vs original (bijective 19/19) | max **1.31e-5 Å**, mean 6.63e-6 |
| stable across `symprec` 1e-5 … 0.3 | yes, all-atom and heavy-only |

**Three rules that come out of the measurement, not from theory:**

1. **Never touch `transformation_matrix` / `std_lattice` / `origin_shift`.** `origin_shift == 0` in only **3/19**; `transformation_matrix == I` in only **13/19**. The raw `.rotations` / `.translations` are in the *input* basis and match the deposit; the standardized description does not. Using it re-introduces the 37–49 % nonstandard-setting problem from the other side.
2. **`symprec` must scale with the coordinate noise; a fixed value fails silently.** Over a σ × symprec grid (isotropic Gaussian noise on cartesian coordinates, n=19 per cell), the rule is **`symprec ≳ 10σ`**, and above **σ ≈ 0.02 Å nothing works at any symprec**. Recommended usage: sweep `symprec` upward and take the largest group stable across two adjacent values.
3. **Score operators by rotation-set match + translation drift, never by exact equality.** A scoring bug (exact equality with translations rounded to 1e-3) made spglib read as brittle — 8/19 at σ=0.005 — when in fact every rotation was correct and every translation was within 1.4e-3 fractional. That retraction is recorded here so it is not re-derived.

**What spglib does not supply, and what would have to be written (~60 lines, all verified to 4.5e-6 Å in the probe):** PBC connected components (`ase.neighborlist` + `scipy.connected_components`) → molecule count; `Z′ = n_components / n_ops`; whole-molecule unwrapping by minimum image about atom 0; and consistent cross-image atom ordering by applying each operator to the reference molecule and matching to cell atoms.

**What the bare route cannot recover:**

- **R3.** With the physical cell agreeing to **1.31e-5 Å**, the pose parameters do not agree at all: centroid max **0.535 fractional**, orientation max **5.79 rad**, handedness **14/19**. Cause isolated: **not** operator order (reversing the list leaves centroid, orientation and handedness bit-identical, 8/8) — the canonicalization simply selected a different, equally valid symmetry image. This is D2 again, and it is not a defect. It does mean **bare-cell ingestion cannot reproduce a stored R3 parameterization**, only an equivalent one.
- **Chemistry.** Bond orders, SMILES, partial charges and fingerprints are out of reach from a heavy-atom-only bare cell. `docs/design/cif_reader_design.md` §1.3 measures **61/300** heavy-atom skeletons returning a *plausible wrong* molecule (BARBOL yields a fully sanitized, radical-free aromatic SMILES). **The hydrogens-present precondition is a hard gate, not a formality.**
- Chirality is not at risk from the image choice in chiral groups: sg 4 (P2₁) and 5 (C2) have all det=+1 operators so every image shares one enantiomer; sg 2/14/15 contain det=−1 operators, so both enantiomers are present in the cell by construction.

---

## 5. What to build, in order

Scoped to CIF/crystal conversion. Everything outside that scope is listed at the end as a finding, not a task.

**B1 — one rebuild-and-assert check.** After `reparameterize_unit_cell`, rebuild the unit cell from the returned parameters and compare against the input R4 at the `crystal_rebuild_checks` bar. Cost: one extra `aunit2ucell`. On the 95-crystal fixture it flags **exactly the one genuinely broken crystal**. **This single check catches S3, S4, S7 and S8 at once** — the best value per line in the whole list. Ship it as a test first, then as an opt-in `validate=True`.

**B2 — forward `std_orientation` in `mol2ucell` (`crystal_building.py:230`).** One line, plus a Z′>1 test asserting `mol2ucell(False)` and `mol2cluster(False)` pose the same aunit. Closes **S1**, the only gap that currently changes a published energy.

**B3 — wire the contract into the two build entry points.** In `build_unit_cell` / `aunit2ucell`: assert `unit_cell_pos.shape[0] == num_atoms * sym_mult` (turning the uncaught `IndexError` at `utils.py:455` into a named error), assert `aunit_handedness ∈ {−1, +1}`, use `round().long()` at `crystal_ops.py:819`, and call the **already-written** `validate_cell_params_ranges` (`crystal_ops.py:715`). Closes **S6**, names the ~1 % `IndexError`, and makes **S4** loud.

**B4 — wrap, do not clip, the fractional centroid** (`crystal_ops.py:34`), and apply the same convention in `get_aunit_positions` (`utils.py:756`). Closes **S4** and makes the parameterization periodic in the centroid, which it currently is not. ⚠ **Owner decision required:** this changes numerical behaviour for out-of-range centroids, and a generative sampler currently sees saturation. Recommended answer: wrap. Alternative: keep clipping and raise instead, which is safe but rejects samples. Default if unanswered: raise (B3 already installs the assertion).

**B5 — reinstate the atom-type check at `featurize_cif_chunks.py:261` with the *blocked* ordering** — `z[block_repeat_interleave(num_atoms, sym_mult)]`, not `z.repeat(sym_mult)` — and fix the identical live tiling bug at `crystal_ops.py:1909`. Closes **S7**. Reinstating it as written would falsely reject ~25 % of Z′>1 crystals.

**B6 — name the R5→R4 inverse.** Add `cluster2ucell`: leading-block slice for Z′=1 (bitwise, 95/95), heavy-atom-centroid-in-cell fallback for Z′>1 (5/5). Restore the commented-out cluster guard at `crystal_building.py:13-14`. Closes **S9** and removes the folklore. Explicitly document that `aux_ind == 0` is the **aunit**, not the cell (it matches 1/95).

**B7 — Z′>1 CIF writing: emit a `_geom_bond_*` loop.** The written file is geometrically correct (nearest-neighbour within a molecule 1.41–2.14 Å, min inter-molecule 2.89–8.58 Å) and CCDC reads `Z` and `Z′` back correctly, but re-perceives bonds across symmetry and fuses everything: atom counts inflate 42→56, 84→**424**, one component reported instead of two, and **5/5 re-featurize as `filter:Polymer`**. Writing the topology the object already holds also fixes the **2/19 chirality inversion** on Z′=1, which has the same cause.

**B8 — the bare-unit-cell reader, if and only if CCDC obviation is selected.** ~60 lines on top of spglib, all verified in §4; `symprec` swept upward with a stability criterion; must **not** be used to reproduce a stored R3; must hard-gate on hydrogens present.

**Deliberately not tasks — findings, out of scope.** The `max_z_prime` padding/collation family (S5, S11, S12), the LAPACK eigenvector anchor (S3's root cause, `geometry_utils.py:241` TODO and the shadowed duplicate at `:275`), the batch-dependent cluster ordering (S10), `reset_sg_info` (S8), the dead branches (S13), the property key bug (S14), and the `is_well_defined` coverage question (S15). Each is real and measured; none is CIF/crystal-conversion work.

---

## 6. Knowledge-type classification (per `AGENTS.md`)

*Table 4: every material claim in this document, its knowledge class, its proof, and whether the proof is adequate. "Integrator-verified" means the file:line and the code behaviour were re-read against the working tree while assembling this document, not merely reported by a probe.*

| claim | class | proof | status |
|---|---|---|---|
| Round-trip residuals are float32 round-off (2.2e-5 Å / 6e-7 frac / 2e-6 rad) | observation | 5 probes over 28 CIF + 100 fixture + 19 written crystals, 19 312 ucell atoms | **measured**; scratch scripts are **not tracked** — reproducibility risk |
| Signatures, call chains and required object state (Table 1) | interface | implementation read + executed | **integrator-verified** (all anchors re-checked) |
| R3→R4 is Z-to-one; R4→R3 is a canonicalizing projection | invariant | constructive: 8/8 images of EBEXUI → same cell to 7.5e-6 Å → same canonical R3 | **verified** |
| "Centroid" means the **heavy-atom** centroid on both sides, consistently | invariant | relabelling 30 % of atoms as H moves the all-atom centroid 1.551 Å while the parameter holds at 2.98e-7 | **verified**; **nowhere documented and nowhere asserted in code** |
| `std_orientation` dropped at `crystal_building.py:230` (S1) | defect | read + 6–10 Å measured on 5/5 Z′=2 | **integrator-verified** |
| `clean_cell_parameters` canonicalization is an identity; the `:583` tie-break is unreachable (S13) | defect | read | **integrator-verified** |
| `is_well_defined` is a python list (Z′=1) / bool tensor (Z′>1) (D5) | defect | read: `crystal_building/utils.py:486` vs `crystal_ops.py:798` | **integrator-verified** |
| Ingest yield 32–47 % | observation | two 60-CIF draws with incompatible tallies | **unresolved** — do not quote a point value |
| 0.702 % pose degeneracy | observation | 5 700 **random** re-poses of 95 real crystals | **measured**, but the rate is over a random pose distribution; on stored real parameters it is **0/95**. ⚠ **working assumption:** do not transfer 0.7 % to a training run without re-measuring on that run's pose distribution |
| spglib exact 19/19 across `symprec` 1e-5–0.3; `symprec ≳ 10σ` | observation | n=19 per grid cell | **measured but small**; not sufficient to set a production `symprec` |
| Writer fidelity 19/19 CIF + 100/100 fixture | observation | executed; `tests/cif/test_cif_writer.py` 15 passed in 14.1 s | ⚠ **that test file is UNTRACKED** — the claim has no repository home until it is committed |
| 37.5 % of CSD carries nonstandard operators | history, corroborated | 7/19 (36.8 %) and 50/100 (49 %) here | **range 37–49 %** |
| 108/230 space groups have `ASYM_UNITS == [1,1,1]` | invariant | `constants/asymmetric_units.py:151` + enumeration | **verified** |
| The build order in §5 | decision | owner scope; no code proof | **awaiting sign-off on B4** |

**Not measured, called out explicitly.**
- **R1→R5 and R5→R1** do not exist and were not attempted.
- **No probe exercised a GPU path for the CIF edges.** The device audit (all outputs stay on `cuda:0`, no hidden `.cpu()`) covers R3/R4/R5 only. Two per-call host round-trips remain on the unit-cell hot path (`crystal_building/utils.py:677` `np.concatenate` of the operators; `:684` a CPU `torch.ones` then `.to(device)`).
- **float64 is unreachable** anywhere in the lattice (`utils.py:677-678`, `:684`). No probe measured whether the 1e-5 Å residual matters to any downstream consumer.
- **Every Z′>1 ordering claim rests on two CIF-stage crystals** (ADEYAN, AGESOX). The five fixture Z′=2 crystals have `z_zp0 == z_zp1` post-`overwrite_conformer_to_zp`, so tiled and blocked orderings coincide there and the hazard is invisible.
- **No acceptance benchmark exists** for any of B1–B8; §5 is a plan, not a validated sequence.

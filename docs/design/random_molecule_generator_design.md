# Random molecular-graph generator with controllable statistics

**Status: PROPOSED. No owner decision blocking; three defects below must be fixed before Stage 1 ships.**
**Measured 2026-08-23 with rdkit 2025.03.5, Python 3.11.9. A working prototype was built and re-executed by an independent reviewer.**
**Revised 2026-08-24 on owner direction — "tier T may also include functional group or larger fragment statistics; independent heteroatom draws are not structured enough" — and on the further direction that the random conformer generator will likely be replaced by a prior model from the conformer-GFlowNet work, so the output contract is SMILES and nothing here couples to `mol_building.py`. §3 Tier T is rewritten from scratch against three independent re-measurements; §2, §4, §5, §6, §7 and §8 change in consequence. Where the three rebuilds disagreed, the disagreement is stated rather than averaged.**

This document is a **decision** plus its supporting **observations**, per `AGENTS.md`. It was corrected against an adversarial review that reproduced the original design's tables *to the digit* and then found two critical defects in the mechanism underneath them. Where a number was retracted, the corrected value is given and the superseded one named.

---

## 1. What is missing, precisely

A random **conformer** generator exists — `dataset_utils/mol_building.py` is entirely SMILES→3D, every entry point takes a `smile: str`, and the randomness is torsional. What does not exist is the **graph** half. Today molecular graphs are read off disk: `parallel_synthesis.py:153` `generate_smiles_dataset` walks a ZINC22 directory, samples *files* with `p = file_sizes / sum(file_sizes)`, and reads their lines.

So the only controllable statistic available today is the file-size distribution of a third-party corpus. That is not a chemical control, and it cannot be steered toward the size regime the molecule autoencoder needs.

---

## 2. Approach

**Chosen: BRICS fragment assembly with a size-targeting controller.** RDKit only, no new dependency.

Measured against the alternatives on the prototype: fragment assembly gives valid, connected, sanitizable molecules by construction, tracks a heavy-atom setpoint closely, and costs 0.4–2.4 ms per molecule. Random valence-constrained assembly produced markedly higher MMFF strain per heavy atom.

**The generator's output contract is a `list[str]` of SMILES. It does not import `mol_building.py` and must not.** The random *conformer* generator is expected to be replaced by a prior model from the conformer-GFlowNet work, so any coupling to the SMILES→3D path would be coupling to a component with a known replacement date. Consequences are in §5 (import surface) and §7 (Stage 5 no longer measures a 3D quantity).

**SELFIES: rejected, with a stated flip condition.** `import selfies` → `ModuleNotFoundError` in `csd_mxt_gfn`; it would be a new runtime dependency. Its guarantee — every token string decodes to a valence-legal graph — is worth **zero** here, measured: 1000 molecules at heavy 32, seed 0 — `grow()` raised 0, SMILES round-trip failed 0, `SanitizeMol` failed 0, non-CHNOF 0, non-neutral 0, disconnected 0, duplicates 0, at 1.34 ms/molecule. Validity is already constructive. Worse for this design's actual requirement: a SELFIES token is an atom-with-incoming-bond-order whose meaning depends on the derivation state, so **there is no token-level feature `f(t)` whose expectation is a functional-group expectation**, and no fragment unit on which to place a larger-fragment target at all. Steering would be sample→decode→measure→reject against statistics with 20–60 % prevalence — Tier R, the most expensive tier, for everything §3 does constructively. There is also no SELFIES analogue of the join veto that takes the acetal family to 0.00 % (§4), because there is no join. *(SELFIES-side claims are UNMEASURED — reasoned from the specification, since the package is absent.)*
**Flip condition, checkable:** adopt SELFIES when the generator stops being an assembler and becomes a *learned* model — i.e. when the conformer-GFN prior, or any trained string/graph generator, is what emits SMILES. At that point validity-under-arbitrary-model-output becomes binding and statistics control moves into the loss/reward. Secondary flip: a genetic or latent-space search over molecules. Note against it even then: SELFIES has changed alphabet and derivation semantics across major versions, so a fixed seed does not pin output across a dependency upgrade — a reproducibility hazard given §6's seed policy.

*(One retraction: the original design rejected RDKit's built-in `BRICSBuild` partly because it "has no seed argument, so it cannot be made reproducible." That is false — it uses Python's global `random`, and `random.seed(n)` makes it fully reproducible. The decision survives on the other grounds — global reseeding conflicts with the seed policy in §6, plus the enumeration and `maxDepth` objections — but the stated justification was wrong.)*

### Three critical defects found in the prototype

D1 and D2 land in Stage 1, D3 in Stage 2. All three are fixed cheaply. None is optional — and **D3 was latent until Tier T restricted the seed pool**, which is the reason it appears in a revision about functional-group control.

**D1 — every BRICS label-7 join silently converts a C=C into a C–C.**

`BRICS.reactionDefs` has 46 label pairs. **45 specify a single bond; exactly one — group 5, labels `7a`/`7b` — specifies a double bond.** The prototype hard-codes `Chem.BondType.SINGLE` in the join.

```
BRICSDecompose(CC=CCc1ccccc1) = ['[16*]c1ccccc1', '[7*]CC', '[7*]CC[8*]']
prototype join  -> CCCc1ccccc1        (single bond)
BRICS-correct   -> CC=CCc1ccccc1      (double bond)
```

Realised rate: 7.5 % of molecules at target 20 heavy atoms, **15.2 % at target 48**. Every one sanitizes, is CHNOF, is connected, is charge-neutral, passes every blacklist filter, and embeds. **An alkene reassembled as an alkane, indistinguishable from an intended saturated chain** — the archetypal silent wrong answer, and invisible to every test tier that checks bond types are *in the vocabulary* rather than *correct*.

Fix: `BRICS_COMPATIBILITY` must map `label → {(partner, bond_order)}`. The originally declared type, `dict[int, frozenset[int]]`, **cannot express the correct answer**.

**D2 — `Fragment.n_heavy` is undercounted by `n_attach` for all 734 fragments, and the compensator was being shipped as a feature.**

`sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1) - n_attach`. Dummy atoms are Z=0 and are *already* excluded by `> 1`; the subtraction double-counts. Correcting it and re-running the identical controller:

| target heavy | prototype: achieved (bias), sd | corrected: achieved (bias), sd |
|---|---|---|
| 20 | 24.30 (+4.30), 3.57 | 21.83 (+1.83), 3.64 |
| 30 | 33.98 (+3.98), 3.21 | 30.62 (**+0.62**), **2.51** |
| 48 | 52.33 (+4.33), 3.46 | 48.31 (**+0.31**), **2.11** |
| 64 | 67.88 (+3.88), 3.03 | 64.28 (**+0.28**), **2.48** |

*Table: heavy-atom setpoint tracking, 300 draws per target, seed 0. "Prototype" carries the counting bug; "corrected" fixes the arithmetic only — same controller, same library.*

Three consequences. The constant ~4 overshoot **is the bug**, so the `size_offset` / `calibrate` / `calibration_target` / `calibration_draws` API existed only to hide it and **should be deleted, not shipped**. The headline "flat sd ≈ 3.2" understated the method by 30–40 %. And the hard floor at ~18 heavy atoms needs re-measuring before it is written into error logic.

**D3 — the size setpoint never scored the seed fragment, and it is the whole tail.** Interior steps project size; the seed did not. Because Tier T's cheapest mechanism *restricts the seed pool* (§3, T-B seed), D3 stops being latent the moment FG control is switched on. Traced to completion:

```
seed  [7*]CC(=O)O                              ( 4 heavy, 1 dummy, label 7)
join  [7*]CC(O)(C#Cc1ccccc1)C#Cc1ccccc1        (19 heavy, TERMINAL)
out   O=C(O)C=CC(O)(C#Cc1ccccc1)C#Cc1ccccc1    (23 heavy, target was 48)
```

**Label 7 is compatible only with itself** (38 slots in the library, 21 terminal, mean cap 7.33 heavy), so a molecule whose only open dummy is a label 7 cannot physically reach 48.

| target heavy | `seed_size_aware = False` | `True` |
|---|---|---|
| 20 | 19.93 ± 3.04 | 20.12 ± 2.98 |
| 32 | 31.03 ± 3.23 | 30.93 ± 2.85 |
| 48 | **45.83 ± 6.56** (bias −2.17, 1st pct 23) | **46.94 ± 2.88** (bias −1.06) |

Fix: score the seed by `n_heavy(F) + Σ_open_labels cap_heavy(label)` against the target with the same Gaussian used at interior steps. Same shape as D2 — the size controller reporting health while an unmodelled structural bottleneck produces the tail.

### The library build recipe is now pinned, and it was ambiguous

Two independent rebuilds produced **734** and **504** fragments from the same corpus. Both are correct for their recipe; only one reproduces this document's other digits. **Normative recipe (734):**

```
corpus = RDConfig.RDDataDir/NCI/first_5K.smi, first 1500 lines   # NOT CHNOF-filtered here
frags  = BRICS.BRICSDecompose(m, minFragmentSize=2)               # NOT the default 1
keep   = (>=1 attachment dummy) AND (elements subset of {*,H,C,N,O,F})
                                                                  -> 1451 unique -> 734
```

Confirmed by two digits the design already pins: the CHNOF reference subset is **n = 954**, and label 3 attaches to oxygen in **143/143** dummy instances (143 instances across 111 fragments, every one bonded to O). Measured sensitivity: `minFragmentSize=1` → 561; CHNOF-filtering the *corpus* first → 730; requiring neutrality → 730; dropping the attachment requirement → 1015. The 504 build filters the corpus to CHNOF + neutral + single-component and uses the default `minFragmentSize`; **the claim that 734 = 504 + 277 uncleavable whole molecules is false** — the 734 recipe already requires ≥1 attachment point. The library-size floor in §7 was inherited from a third number (888) and is invalid for either build.

**Both libraries are used below and every table names its own.** Library shapes: 734 — `n_heavy` mean 7.37 / median 7 / sd 3.64 / max 31, `n_attach` {1:434, 2:214, 3:68, 4:12, 5:6}, mean 1.559. 504 — `n_heavy` mean 7.15 / median 7 / max 30, `n_attach` {1:320, 2:135, 3:36, 4:10, 5:3}, mean 1.494, largest ring 7 atoms, 77 Murcko scaffolds. Directional conclusions reproduce on both; absolute prevalences do not, which is §8's point.

D1 and D2 were re-verified on both builds: `reactionDefs` = 46 pairs, `Counter({'-': 45, '=': 1})`, the double being `('7a','7b','=')`; round-trip of `CC=CCc1ccccc1` matches with the label table and mismatches with hard-coded SINGLE. D2: `n_heavy == GetNumHeavyAtoms()` for 504/504 and for 734/734, against **29 of 734 fragments reporting `n_heavy ≤ 0`** (min −2) under the bug.

---

## 3. The statistics-control interface

Four tiers, distinguished by *how* each statistic is achieved. This distinction is the point: a knob that silently becomes a 200× slowdown is a defect, so **every knob declares its mechanism — DIRECT (constructive, rejection rate 0), SETPOINT (closed-loop), TILT (expectation only), or REJECTION (with its measured rate)** — and that declaration is part of the public API, not documentation. Measured examples of the gap it protects: `0 aromatic rings` costs **161.7** attempts/molecule as a rejection knob and **1.00** as a constructive one; `exactly 3 COOH` costs **1187** and 2.67.

**Tier D — direct, constructive, rejection rate 0.** Element set (by library construction — this is not a soft filter: an out-of-vocabulary Z becomes `F.one_hot(-1)` and raises downstream rather than degrading), valence, single connected component, bond types within the vocabulary, **and ring systems / larger fragments (new — see T-D below).**

**Tier S — setpoint, closed-loop.** Heavy-atom count. After D2 and D3, tracks to −1.2…+0.6 with sd ≈ 2.4–3.2.

**Tier T — functional-group and fragment statistics.** Four mechanisms, selected by attribution, replacing the single scalar tilt. See below.

**Tier R — rejection only, budgeted.** Net formal charge, forbidden motifs, and any FG constraint with no constructive counterpart. Costs 2.4–3.5 % on the base spec; **every Tier-R knob must declare an attempts-per-molecule budget and abort, not stall.**

### Tier T — functional-group and fragment statistics

**Owner direction: "tier T may also include functional group or larger fragment statistics. independent heteroatom draws are not structured enough."** The scalar per-fragment heteroatom-fraction tilt is **deleted**, and so is the bisection-on-β solve that went with it. Two measurements say the deletion is free rather than a loss. Element counts are **100 % additive** over source fragments, so a heteroatom-fraction statistic is blind to every bond a join makes — which is exactly why it read 0.245–0.260 while acetal content sextupled (§4). And the replacement subsumes it anyway: driving the fragment-class knob alone spans heteroatom fraction **0.117** (aromatic-carbocycle β=+4) to **0.290** (polar-acyclic β=+4) against an NCI reference of 0.246, as a side effect of controlling something chemically meaningful *(library 504, target 32)*.

#### T-0. The attribution measurement that determines which mechanism applies

Every FG match in a generated molecule is tagged by whether its matched atoms lie inside one source-fragment instance or span a join bond. Occurrence-weighted, the aggregate is reassuring — **91.0 / 91.2 / 91.6 %** of all FG occurrences are fragment-internal at targets 20/32/48 *(library 734, 90 SMARTS, 500 molecules/target)* — **and that aggregate is misleading**, because the join share is concentrated in a few groups that happen to be the interesting ones.

The operational discriminator is not the occurrence split but the **suppression floor**: zero-weight every carrier fragment and measure what remains. That is what a fragment-level knob can never remove *(library 504, target 32, mean matches per molecule)*:

| statistic | carriers | **floor** (all carriers zeroed) | uniform | ceiling (carriers only) | heavy @ ceiling | class |
|---|---|---|---|---|---|---|
| benzene | 114 | **0.000** | 0.767 | 3.790 | 29.9 | inherited |
| aromatic_N | 45 | **0.000** | 0.492 | 6.018 | 30.8 | inherited |
| hydroxyl | 126 | **0.000** | 0.932 | 4.965 | 31.0 | inherited |
| carboxylic_acid | 36 | **0.000** | 0.177 | 1.093 | **7.7** | inherited |
| amine_prim | 43 | **0.000** | 0.247 | 3.540 | 30.8 | inherited |
| nitrile | 28 | **0.000** | 0.155 | 2.510 | **17.4** | inherited |
| aldehyde | 93 | **0.000** | 0.055 | 2.172 | **49.4** | inherited |
| fluoro | 4 | **0.000** | 0.005 | infeasible | — | inherited |
| n_rings | 223 | **0.000** | 2.862 | 5.013 | 31.7 | inherited |
| n_aromatic_rings | 148 | **0.000** | 1.125 | 4.668 | 32.1 | inherited |
| ether | 35 | **0.823** | 1.475 | 5.668 | 28.0 | **join-made** |
| amide | 21 | **0.765** | 0.978 | 6.385 | 30.1 | **join-made** |
| amine_tert | 3 | **0.740** | 0.907 | 3.000 | **17.0** | **join-made** |
| alkene | 18 | **0.492** | 0.497 | 3.000 | **25.5** | **join-made** |
| ester | 7 | **0.477** | 0.532 | 3.440 | **18.6** | **join-made** |
| ketone | 29 | **0.352** | 0.598 | 4.230 | 32.7 | **join-made** |
| N,O-acetal | 1 | **0.170** | 0.315 | — | — | **join-made** |
| aminal | 1 | **0.158** | 0.230 | — | — | **join-made** |
| gem-ether/acetal | 19 | 0.052 | 0.223 | 7.380 | 30.9 | mixed, fragment-dominant |

**Read this as the design's central structural limit: a per-fragment weight of any form cannot push the bottom nine statistics below their floor.** Draw-time control is therefore not merely imprecise on them, it is bounded away from the target, and the design handles them at the join table (T-C) instead. Independent confirmation: zeroing every ring-bearing fragment gives **exactly 0.000 rings/molecule**, so the separation is measured, not argued.

**Disputed, and stated as disputed.** Two rebuilds disagree on `amide`: occurrence-tagging on library 734 puts it at **70 % fragment-internal**, occurrence-tagging on library 504 puts it at **70 % join-made**. They cannot both describe one library. The floor test resolves the operational question regardless — on 504 the amide floor is 0.765/molecule and a tilt bottoms out at 23.5 % molecule-level presence — so **amide is treated as join-reconstitutable and routed to T-C**. Re-run the floor table on whichever library ships; it is 20 minutes of compute and the routing depends on it.

The join is where alkenes come from too: **alkene is 92 % join-made**, via the single `(7,7)` double-bond pair. D1 therefore did not merely mislabel bonds — it destroyed almost all alkene content in the output.

#### T-D. Ring systems and larger fragments — DIRECT, rejection rate 0

**Tree assembly adds one bond between two disjoint components, so it cannot close a ring.** Every ring system in an output molecule was carried intact by a library fragment. Measured three ways:

- ring count, aromatic-ring count, heteroaromatic-ring count: **100.0 %** additive over source fragments (0.00 mean discrepancy, sd 0); ring-system multiset **99.2 %** exact *(library 734, 500 molecules, target 32)*.
- ring systems emitted that are outside the library's 59-skeleton vocabulary: **1/328, 0/616, 1/989, 0/1375** at targets 20/32/48/64. Both apparent escapes are acridine, which *is* in the library as `[16*]c1c2ccccc2nc2ccccc12` — a skeleton-canonicalisation artifact, not ring formation *(library 504)*.
- vocabulary restriction is exact: dropping the 50 aromatic-heteroatom fragments takes molecules containing an aromatic heteroatom from 92/400 to **0/400**, with heavy 31.25 → 31.61 *(library 504, target 32)*.

**So "larger fragment statistics" are Tier D, not Tier T** — controlled by an allow/deny list over the ring-system vocabulary, exact, zero rejection, zero size cost. This is a stronger guarantee than any tilt and it is the direct answer to the owner's direction. Vocabulary (library 734): `R0_acyclic` 55.7 %, `R_arom_carbocycle` 19.2 %, `R_sat_heterocycle` 7.9 %, `R_heteroarom_6_fused` 4.8 %, `R_sat_carbocycle` 3.5 %, `R_arom_carbocycle_fused` 3.0 %, `R_heteroarom_6` 2.6 %, `R_heteroarom_5` 1.8 %, `R_heteroarom_5_fused` 0.5 %, `R_other` 1.0 %.

**What is not controllable at any price:** any ring system absent from the library (BRICS assembly concatenates rings, never forms one); rings larger than 7 atoms and all macrocycles (largest library ring is 7); ring-fusion topology beyond what fragments already carry. These are corpus properties. Widening the corpus is the only lever, and §8 already says what that invalidates.

#### T-A. Fragment-class rates — SETPOINT (closed-loop on realised usage)

**The weight vector is not the draw.** At uniform weights, realised fragment-class usage differs from library share by **TV = 0.2775** *(library 504)*: `amine` 0.0516 → 0.2287 (×4.43), `aromatic_heterocycle` 0.0833 → 0.0310 (×0.37). The BRICS label graph and size feasibility set the draw, not the weights. On library 734 the same effect: 657/734 fragments ever drawn, effective library 476 of 734, `fr_ether` marginal 0.061 → 0.125 realised. **Any open-loop weight assignment is wrong by construction; T-A is closed-loop or it is nothing.**

Mechanism: IPF on realised usage over a **coarse fragment-class partition**, 25 iterations × 200 molecules, **6 s total** *(library 504, target 32, 8 classes)*:

| class | target | uniform | after IPF | fitted weight |
|---|---|---|---|---|
| aromatic_carbocycle | 0.2111 | 0.0880 | **0.2052** | 4.497 |
| aromatic_heterocycle | 0.0474 | 0.0310 | 0.0516 | 1.800 |
| amide_ester | 0.0266 | 0.0289 | 0.0249 | 0.930 |
| acid_alcohol | 0.1317 | 0.1567 | 0.1315 | 0.033 |
| amine | 0.0913 | 0.2287 | 0.1032 | 0.010 |
| ether_acetal | 0.0163 | 0.0459 | 0.0263 | 0.001 |
| carbonyl_nitrile | 0.1174 | 0.1262 | 0.1191 | 0.340 |
| aliphatic | 0.3582 | 0.2946 | 0.3381 | 0.389 |
| **TV** | | **0.2031** | **0.0279** | |

Downstream, molecule-level FG error against the corpus reference falls **0.403 → 0.204** (benzene 0.803 → 0.038, amide 0.554 → 0.026, ester 0.156 → 0.019), i.e. it also improves two majority-join-made statistics, because class usage changes *which joins occur*. Stable across fit seeds: TV 0.017–0.027, 5.4–5.6 s. Size cost **−2.4 heavy atoms**, which must be reported (§3, interaction).

**Granularity is a knob with a real failure mode.** The same IPF on per-signature classes (one class per distinct FG signature) **diverges**: TV 0.2547 → 0.1255 at iteration 10 → **0.2597** at iteration 29, effective library 504 → 64 → 29 → 7 → **3**, downstream error barely moving (0.317 → 0.296). **Cap the control dimension at ~10 classes and guard on participation ratio.** Honest residual: the classes IPF has to crush (`acid_alcohol` → 0.033, `amine` → 0.010) are exactly the ones whose downstream statistics regress (hydroxyl 0.294 → 0.616 error) — within-class composition is unconstrained. That argues for a moderate class-count increase, not per-signature.

#### T-B. Individual functional groups — SEED (direct), BAN (direct), or TILT (expectation)

Three sub-mechanisms, chosen by the constraint shape, all applied **inside the per-step conditional proposal** (not as a one-shot library reweighting, which the TV = 0.2775 result forbids):

- **T-B-seed — `require ≥1 X`. DIRECT, 1.00–1.10 attempts/molecule.** Force the seed fragment to carry X. Valid iff ≥1 library fragment carries X. Compare rejection for the same constraint *(library 504, target 32 / 48 attempts per accepted molecule)*: `exactly 1 COOH` **8.09 / 5.68 → 1.10 / 1.03**; `≥1 amide` 3.17 / 2.40 → 1.00; `0 aromatic rings` **46.5 / 161.7 → 1.02 / 1.00**; `≥1 fluoro` **58.8 / 45.9 → 1.00**; `≥2 fluoro` **247.5 / 296.8 → 1.03 / 1.02**; `≥1 pyridine` 4.04 → 1.00; `≥1 naphthalene` 19.6 → 1.00; `≥1 indole` **214.5 → 1.00**.
  **Yield is not the health metric here.** With one indole fragment in the library, seeding gives 100 % yield *degenerately* — every molecule contains the same indole and the effective library falls to **43.6** (baseline 74.8). `tilt β=+4 + reject` is the honest alternative at 3.7–4.4 attempts/molecule and effective library 56.3. **Every seeded knob must report effective library, and the spec must declare `min_seed_pool`** (below ~6 carriers the sample is seed-degenerate). Corpus size does not fix this: widening 1500 → 4999 NCI lines takes F-carriers 4 → 13 and `≥2 fluoro` rejection only 247 → 46 attempts/molecule, while the seed stays at 1.00.
- **T-B-ban — `at most k of X`. DIRECT, 1.0–1.1 attempts/molecule.** Exclude X-carrying candidates from the proposal after the k-th. **Valid only for X the join cannot reconstitute** — the reconstitutable set (`ether`, `amide`, `ester`, `ketone`, `amine_tert`, `alkene`, `aniline`, and the acetal family) is enumerated at library-build time from the T-0 floor table and a ban on any member is **refused**, not silently under-delivered.
- **T-B-tilt — `target rate`. EXPECTATION ONLY, ~1.00 attempts/molecule at |β| ≤ 3.** `w_i ∝ count_i · exp(Σ_c β_c f_c(F_i))`, folded into the per-step conditional proposal alongside the size kernel. Monotone and roughly 10× dynamic range over β ∈ [−2, +2] on library 734 (aromatic rings 0.45 → 5.22, saturated rings 0.10 → 6.48, `fr_amide` 0.23 → 6.74, `fr_Al_OH` 0.27 → 10.76), and monotone over the full β ∈ [−6, +6] on library 504 for all 11 knobs swept (class usage and molecule-level FG presence tables below).

  | β | arom_heterocycle usage | carbonyl usage | nitro presence | nitrile presence | COOH presence | ether presence |
  |---|---|---|---|---|---|---|
  | −6 | 0.08 % | 0.04 % | 0.0 % | 0.0 % | 0.0 % | 0.5 % |
  | −2 | 1.07 % | 2.55 % | 4.0 % | 1.5 % | 2.5 % | 18.8 % |
  | 0 | 7.89 % | 13.94 % | 22.0 % | 9.8 % | 18.2 % | 63.2 % |
  | +2 | 38.96 % | 37.28 % | 77.2 % | 41.5 % | 54.2 % | 96.0 % |
  | +6 | **96.20 %** | **56.72 %** *(ceiling)* | 100 % | 100 % | 99.8 % | 100 % |

  **Three hard constraints on this mechanism, each measured, each a refusal condition rather than a tuning note.**
  1. **Ceilings are class-specific and set by the BRICS label graph, not by β.** `arom_heterocycle` and `aliph_ring` reach 94–96 % usage; `carbonyl` saturates at **56.7 %** and `apolar_acyclic` at **54.0 %** at β = +6, because those classes carry labels incompatible with many open attachment points. A target above a class ceiling must be refused, not chased with larger β.
  2. **Floors are set by T-0.** Suppression to ~0 works for any statistic whose floor is 0.000; for join-reconstituted groups the tilt bottoms out (amide 23.5 % presence at zero carrier usage; `acetal_NO` 23.5 % → 15.0 % and `aminal` 17.5 % → 16.5 % even at β = −8).
  3. **Large β collapses the library and the response reverses.** Driving `benzene` on library 504 with a size projection that competes with the tilt: E[benzene] 0.780 (β=0) → 0.943 (+1) → 0.460 (+2) → 0.405 (+4), while the participation ratio of the tilted weight vector goes 504 → 132 → **3.1** → **1.0** and achieved heavy 31.25 → 21.93 → 16.47. Mass concentrates on a few large carriers, the setpoint cannot place them, small terminals get drawn, the statistic reverses. Two independent implementations that fold the tilt into the per-step conditional proposal saw monotone response to |β| = 6; the one that did not, saw 7 of 12 features go non-monotone. **The design's position: monotonicity is an implementation property, not a property of the mechanism, so the solver must not assume it** — see the API below.

  Diversity band: effective library 95 at β = 0, 89–112 at |β| ≤ 2, **35.6 at β = +6**. **Ship `abs(beta) <= 3` as a hard cap.**

  Cross-coupling is real and must be reported, not hidden: driving `hydroxyl` to β=+1 multiplies its own rate ×3.10 but also `acetal_OO` ×1.85 and cuts `nitrile` ×0.62; driving `n_rings` moves its own target **+9 %** while halving benzene and cutting `acetal_OO` 73 % — **ring count is effectively uncontrollable by tilt and must be routed to T-D (vocabulary) instead.**

#### T-C. Join classes — DIRECT at the join table, rejection rate 0

This is the mechanism the previous design lacked, and it is the only one that reaches join-manufactured content. The label pair of a join is chosen at draw time, together with the site and the partner, so constraining it costs nothing. Six of the 46 pairs involve S and are unreachable in CHNOF; the remaining 40 partition into 11 join classes whose FG yield is near-deterministic *(library 734, 1500 molecules, target 32, FGs created per 100 joins of that class)*:

| join class | label pairs | FGs manufactured / 100 joins |
|---|---|---|
| amide/carbamate/urea | (1,5) (1,10) | **amide 100**, imide 10, urea 6, carbamate 2 |
| ester | (1,3) | **ester 91**, ether 96 |
| ether | (3,4) (3,14) (3,15) (3,16) | **ether 96–100** |
| ketone | (6,13) (6,14) (6,15) (6,16) | **ketone 86–93** |
| arylamine | (5,14) (5,16) | **aniline 98** |
| alkylamine | (4,5) (5,15) | N,O-acetal 7 / aminal 5, else none |
| **anomeric_O** | **(3,13)** | **acetal 114, orthoester 30**, ether 99, N,O-acetal 18 |
| **anomeric_N** | **(5,13)** | **N,O-acetal 114**, aminal 13 |
| azole/lactam N-alkyl | (8,9) (8,10) (9,13…16) (10,13…16) | none |
| C–C | (8,13…16) (13,14…16) (14,14…16) (15,16) (16,16) | none |
| alkene (**DOUBLE**) | (7,7) | none — and D1 lives here |

A join-class weight vector (including zeros) is a first-class Tier-T knob. **The predicate must be a precomputed boolean table over `(fragment, dummy)` attachment environments, evaluated as an array test.** The same guard written as a per-candidate `HasSubstructMatch` loop over ~120 candidates cost **5×** (2.5 → 13.0 ms/attempt at target 48). That difference is the difference between a knob and a defect.

#### The Tier-T knob table

| statistic | definition | mechanism | class | measured cost |
|---|---|---|---|---|
| ring-system vocabulary | multiset of ring skeletons, canonical SMILES of each ring system | allow/deny list at library build | **DIRECT** | 0 rejection, 0 size cost, 0.00–0.30 % escape (canonicalisation artifacts) |
| ring / aromatic-ring counts | `RingInfo.NumRings()`, aromatic subset | vocabulary + fragment-class IPF | **DIRECT / SETPOINT** | exactly additive (100 %); tilt is *not* an option here (see cross-coupling) |
| fragment-class rates | realised usage share over an ≤10-class partition | IPF on realised usage | **SETPOINT** | 6 s fit, TV 0.203 → 0.028, −2.4 heavy |
| `require ≥1 FG` | ≥1 SMARTS match in the emitted molecule | seed-forcing | **DIRECT** | 1.00–1.10 attempts/mol; requires `min_seed_pool`; report effective library |
| `at most k FG` | ≤ k SMARTS matches | candidate ban | **DIRECT** | 1.0–1.1 attempts/mol; refused for join-reconstitutable FGs |
| FG rate (inherited FGs) | E[matches/molecule] | exponential tilt in the conditional proposal | **TILT** | ~1.00 attempts/mol at \|β\| ≤ 3; bounded by floor/ceiling |
| FG rate (join-made FGs) | E[matches/molecule] | join-class weights | **DIRECT** | 0 rejection; +0.26–0.35 ms/mol |
| acetal family = 0 | 4 SMARTS, §4 | library filter + het-gem join veto | **DIRECT** | 3.1 % yield loss, 0 size cost — §4 |
| peroxide = 0 | `[OX2][OX2]` | drop 2 fragments at build | **DIRECT** | 0 |
| exact FG count `== k` | exact match count | seed + ban + tilt, residual reject | **REJECTION** | `==1`: 1.03–1.10; `==3`: **2.67 / 4.27**; `k ≥ 4` unmeasured, assume unaffordable |
| Murcko scaffold identity | Bemis–Murcko framework of the product | — | **REJECTION only** | molecule-only: only **17 %** of products share any source fragment's scaffold; 417 distinct scaffolds per 500 molecules |
| heteroatom fraction | (N+O+F)/heavy | *deleted* | — | 100 % additive ⇒ blind to joins; spanned as a side effect of T-A |

#### Interaction with the heavy-atom setpoint — real numbers

Tier T degrades Tier S, always in the same direction, and the amount is the acceptance criterion *(library 504, 500 accepts/cell, seed 0)*:

| configuration | heavy @20 | heavy @32 | heavy @48 |
|---|---|---|---|
| Tier T off (setpoint only) | 19.15 ± 2.61 (−0.85) | 31.21 ± 2.44 (−0.79) | 47.24 ± 2.48 (−0.76) |
| class tilt β=+2 | 19.71 ± 2.79 (−0.29) | 31.60 ± 2.57 (−0.40) | 47.91 ± 2.39 (−0.09) |
| FG tilt β=+2 (COOH) | 19.14 ± 2.65 (−0.86) | 31.11 ± 2.68 (−0.89) | 46.93 ± 2.57 (−1.07) |
| FG tilt β=+4 (COOH) | 18.67 ± 2.88 (−1.33) | 30.52 ± 2.64 (−1.48) | 46.59 ± 2.59 (−1.41) |
| hard spec (1 COOH, ≥2 ArRing) | 19.66 ± 3.26 (−0.34) | 30.93 ± 2.85 (−1.07) | 46.94 ± 2.88 (−1.06) |
| **same, D3 unfixed** | 19.93 ± 3.04 | 31.03 ± 3.23 | **45.83 ± 6.56 (−2.17)** |
| class IPF (T-A) | — | 29.22 ± 4.10 (−2.4) | 45.80 ± 4.59 |
| tilt driven into collapse (β ≥ +2, PR → 3) | — | **21.93 ± 4.78 (−9.3)** | — |
| **production** (clean library + join veto + hard spec) | 19.89 ± 3.19 (−0.11) | 30.83 ± 3.02 (−1.17) | 47.07 ± 2.86 (−0.93) |

**Budget: |bias| ≤ 1.5 and sd ≤ 3.5 with Tier T engaged, given D3 fixed.** T-A costs −2.4 and is the expensive one; |β| ≤ 2 costs ≤ −1.1; β = +4 costs −1.4; collapse costs −9.3 and is what the participation-ratio guard exists to prevent. Production throughput 0.98 / 1.32 / 2.10 ms per accepted molecule at yields 87.3 / 90.9 / 96.9 %.

Second interaction, from the attainability table: several targets are reachable only at a heavy count far from the setpoint — the `carboxylic_acid` ceiling collapses molecules to **7.7** heavy, `nitrile` to 17.4, single-carrier motifs to 3.0. **A Tier-T target that is attainable in isolation but not jointly with `target_n_heavy` is a spec error, not a tuning problem**, and is refused at compile time.

#### Public API

```python
FGName = str          # key into FG_VOCABULARY (17 curated SMARTS, §3 T-B)
RingSystem = str      # canonical SMILES of a ring skeleton, from the library vocabulary
JoinClass = str       # one of the 11 join classes, §3 T-C

@dataclass(frozen=True)
class FGConstraint:
    name: FGName
    mode: Literal["require", "forbid", "at_most", "at_least", "rate"]
    value: float | int | None = None      # count for at_most/at_least, matches/molecule for rate
    mechanism: Literal["auto", "seed", "ban", "tilt", "join", "reject"] = "auto"

@dataclass(frozen=True)
class MoleculeSpec:
    target_n_heavy: int
    n_heavy_window: tuple[int, int] | None = None
    target_n_atoms: int | None = None                       # total incl. H; see "the knob that is missing"
    allowed_ring_systems: frozenset[RingSystem] | None = None   # None = whole library vocabulary
    forbidden_ring_systems: frozenset[RingSystem] = frozenset()
    fragment_class_rates: Mapping[str, float] | None = None     # T-A IPF targets; len <= 10 enforced
    fg_constraints: tuple[FGConstraint, ...] = ()
    join_class_weights: Mapping[JoinClass, float] | None = None # 0.0 vetoes the class
    forbid_join_environments: frozenset[str] = frozenset({"het_gem"})   # T-C predicates; default on
    max_beta: float = 3.0
    min_effective_library: float = 60.0     # participation ratio of the REALISED draw
    min_seed_pool: int = 6
    max_attempts_per_molecule: float = 4.0
    seed: int | None = None

def compile_spec(spec: MoleculeSpec, library: FragmentLibrary) -> CompiledSpec: ...
def attainable_range(library: FragmentLibrary, stat: FGName | str, target_n_heavy: int,
                     *, n_draws: int = 400, rng: np.random.Generator) -> AttainableRange: ...
    # AttainableRange(floor, ceiling, heavy_at_floor, heavy_at_ceiling, n_carriers, attribution)

class RandomMoleculeGenerator:
    def __init__(self, library: FragmentLibrary, spec: MoleculeSpec,
                 rng: np.random.Generator) -> None: ...
    def sample(self, n: int) -> list[str]: ...          # SMILES; no rdkit-3D anywhere
    def diagnostics(self) -> GeneratorDiagnostics: ...  # yield, attempts/mol, effective library,
                                                        # achieved heavy mean/sd, realised FG rates
```

Errors — all subclasses of `ValueError`, all raised from `compile_spec` where possible, i.e. **before any molecule is drawn**:

```python
class SpecUnsatisfiableError(ValueError):
    statistic: str
    requested: float
    attainable: AttainableRange   # floor, ceiling, and the heavy count at each
    mechanism: str                # the mechanism that would have to deliver it
    reason: Literal["below_floor", "above_ceiling", "no_carrier",
                    "seed_pool_too_small", "join_reconstitutable",
                    "incompatible_with_size_setpoint", "class_dimension_too_large"]
```

Runtime failures are distinct and are never silent yield loss:
`RejectionBudgetExceeded(attempts_per_molecule, budget)`,
`EffectiveLibraryCollapse(participation_ratio, floor)` — checked on the realised draw, not the weight vector alone,
`ControlDidNotConverge(statistic, target, achieved, iterations)` — raised by the T-A IPF and by the T-B solver.

**The T-B solver is not a bisection.** Bisection assumes monotonicity, which the collapse regime violates. The solver evaluates the statistic on a β grid over `[-max_beta, +max_beta]`, **asserts monotonicity on the grid**, and interpolates; a non-monotone grid raises `ControlDidNotConverge` with the grid attached rather than returning an arbitrary root. It solves on the **realised** rate, never the nominal one. `np.exp` overflow is guarded by a `logsumexp` normaliser (§6).

### The knob that is missing, and it is the expensive one

`MoleculeSpec.target_n_heavy` is the setpoint, but the pipeline's hard gate is on **total atoms including hydrogen** (`parallel_synthesis.py:222`). Nothing controls the H count:

| target heavy | achieved heavy | total atoms | **sd(total) with heavy held exactly fixed** |
|---|---|---|---|
| 20 | 21.41 ± 4.39 | 44.1 ± 10.0 | 6.63 |
| 32 | 32.05 ± 3.01 | 63.6 ± 9.5 | 7.77 |
| 48 | 48.27 ± 2.97 | 96.0 ± 11.2 | **8.64** |

*Table: 800 accepted molecules per target, seed 0. The last column holds heavy-atom count exactly fixed and measures the remaining scatter in total atoms — i.e. the variance the setpoint cannot touch.*

At heavy exactly 48, total atoms still scatters with sd 8.64 — **more total-atom variance than the heavy-count controller contributes**. H/heavy spans 3.3× (fully aromatic to fully saturated). This is why `n_atoms` accounts for ~1000 of every ~1200 rejections and why a ±9-atom window is needed to reach 63 % acceptance.

**Fix: store `n_atoms_with_h` per fragment and project total atoms, not heavy atoms.** Should collapse the dominant rejection channel. *(Speculative on the magnitude; cheap to test.)*

---

## 4. A chemistry problem the summary statistics hid — now a Tier-T capability

Acetal and related content is enriched 12–61× over the source corpus and grows steeply with size:

| motif | NCI reference | heavy 20 | heavy 32 | heavy 48 | enrichment @48 |
|---|---|---|---|---|---|
| geminal ether / acetal | 4.7 % | 10.3 % | 22.0 % | **59.1 %** | 12.6× |
| N,O-acetal | 0.5 % | 8.7 % | 14.3 % | **30.6 %** | 58× |
| orthoester | 0.1 % | 0.8 % | 2.1 % | 5.9 % | 56× |
| aminal | 0.5 % | 5.7 % | 5.4 % | 6.6 % | 13× |

*Table: motif prevalence in accepted molecules (800 draws/target, full default rejection layer applied) against the same 1500 NCI molecules the fragments were cut from (CHNOF subset, n=954).*

**Mechanism — the original attribution was half right and the correction matters.** Label 3 does attach to oxygen in 143/143 dummy instances, and its compatibility set is all carbon-attaching labels. But the culprit pair is **label 13** = `[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]`, a ring carbon *already bonded to a ring heteroatom* — the anomeric carbon. Its attachment atoms are `C|het:O` 59 %, `C|het:OO` 24 %, `C|het:N` 18 %. Joining 13 to an O (label 3) makes a glycosidic acetal; to an N (label 5) an N,O-acetal. **79 % of join-made acetals and 100 % of join-made orthoesters come from `(3,13)`; 69 % of join-made N,O-acetals from `(5,13)`.** Blaming label 3 alone leads to the wrong fix: forbidding all label-3 joins costs the size setpoint 5 heavy atoms and nearly triples its sd (34.2 → 28.9, sd 3.6 → 9.7), and a narrow label-3-only guard *raises* the N,O-acetal rate from 19.2 % to **28.0 %**. **BRICS `reactionDefs` encode which bonds are retrosynthetically *cleavable*, not which recombinations are *isolable compounds*** — using the full table as a forward assembler is the source.

**Provenance is split, and both halves must be treated.** At target 48, of the molecules carrying a gem-ether, **73.2 % already carry it inside a library fragment** and only 26.8 % is join-created; **21 of 504 fragments** (29 of 734) are carriers. For the rest it reverses: N,O-acetal **98.7 %** join-created, orthoester 91.4 %, aminal 91.7 %. The original design attributed the whole phenomenon to the join table; for the largest channel that is wrong, and it is why a join guard alone cannot finish the job.

### Suppression: 0.00 %, and its true cost

*(library 504, 400 accepts/cell, seed 0, target heavy 48. "acetal family" = gem-ether/acetal `[CX4]([OX2])[OX2]`, N,O-acetal `[CX4]([OX2])[NX3]`, orthoester `[CX4]([OX2])([OX2])[OX2]`, aminal `[CX4]([NX3])[NX3]`.)*

| mechanism | acetal | N,O | ortho | aminal | yield | attempts/mol | ms/mol | heavy | eff. library |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 48.5 % | 19.2 % | 14.5 % | 3.0 % | 100 % | 1.00 | 2.00 | 47.15 ± 2.57 | 74.1 |
| join guard, **narrow** (label 3 only) | 28.7 % | **28.0 %** | 0.8 % | 2.8 % | 100 % | 1.00 | 2.15 | 47.34 ± 2.43 | 83.0 |
| join guard, **broad** (any het-gem) | 30.5 % | **0.0 %** | 2.0 % | 0.5 % | 96.9 % | 1.03 | 2.34 | 47.34 ± 2.50 | 85.3 |
| library filter only (drop 21 carriers) | 21.2 % | 14.8 % | 0.0 % | 2.2 % | 100 % | 1.00 | 2.08 | 47.20 ± 2.47 | 76.0 |
| post-hoc rejection, all four | 0.0 % | 0.0 % | 0.0 % | 0.0 % | **41.6 %** | **2.40** | 4.96 | 47.07 ± 2.53 | 79.2 |
| tilt β = −8 on all four | 4.2 % | 15.0 % | — | 16.5 % | 100 % | 1.00 | 1.03 | (target 32) | — |
| **library filter + broad join guard** | **0.00 %** | **0.00 %** | **0.00 %** | **0.00 %** | **96.9 %** | **1.03** | **2.43** | **47.20 ± 2.47** | **77.6** |

Same combination at targets 20 and 32: **0.00 % on all four**, yields 97.1 % / 95.9 %, heavy 19.28 ± 2.50 / 31.26 ± 2.41 — bias and sd indistinguishable from baseline, effective library 68.2 / 83.9. Independently reproduced with a class-IPF weight vector in place of the plain setpoint: acetal family 42.5 / 48.0 / 62.5 % → **0.0 %** at all three targets for **+0.26–0.35 ms/molecule**.

**Verdict: DIRECT (constructive), rejection rate 0, 3.1 % yield loss.** Post-hoc rejection buys the identical 0.00 % for a **58 % yield loss** and 2.4× the wall time; a β = −8 tilt — already past the collapse point — barely moves N,O-acetal (23.5 → 15.0 %) and does nothing to aminal (17.5 → 16.5 %), which is the T-0 floor result restated on the motif that matters most.

**The true cost is not the yield; it is the chemistry the guard takes with it** *(library 504, target 48, molecule-level presence, guarded vs baseline)*: **secondary amines −8.0 pts**, tertiary amines −3.5, aromatic N −3.0, phenol −1.2, ether −1.0, alcohol −1.0, benzene −1.0, aniline −0.7; ketone **+7.0**, amide +4.5, ester +1.8; mean aromatic rings 4.14 → 4.27. The −8.0 on secondary amines is intrinsic — the het-gem predicate blocks N onto a carbon already bearing N or O. If that loss is unacceptable for a downstream use, the measured alternative is an `ether` tilt at β = −6 plus a residual reject: acetal 0.0 % at 99.3 % yield, at the price of stripping ethers entirely. **State the trade in the spec; do not tune it silently.**

**Why this belongs in this document rather than a chemistry footnote:** the design's quality statistic was heteroatom fraction, measured at 0.258 / 0.245 / 0.260 against an NCI reference of 0.261. That statistic is *flat* across exactly the size range over which acetal content sextuples — necessarily so, since element counts are 100 % additive over fragments and therefore blind to every bond a join makes. A summary statistic reporting health while the underlying chemistry rots is this repository's signature failure, in a new place, and it is the measured basis for deleting the scalar heteroatom tilt in §3.

Related, and a free fix the original design missed: peroxides come from **two library fragments** (`[3*]OO`, `[3*]OO[3*]`), not from join artefacts — label 3 is not self-compatible, so an O–O join is unreachable. Removing those two fragments at build time takes the peroxide rate from 2.1–3.1 % to **0.00 %** at every target, confirmed on both library builds. The original design instead paid a permanent runtime rejection and sent future maintainers to the wrong layer.

---

## 5. Module layout and the seam

```
mxtaltools/dataset_utils/random_molecules/
    __init__.py              # explicit; do NOT inherit construction/'s namespace-package pattern
    fragment_library.py      # build recipe (§2), ring-system vocabulary, FG carrier sets,
                             #   join-class table with BOND ORDER (D1), attachment-environment table
    assembler.py
    controls.py              # Tier S setpoint, T-A IPF, T-B seed/ban/tilt, T-C join veto
    spec.py                  # MoleculeSpec, FGConstraint, compile_spec, SpecUnsatisfiableError
    generator.py
    smiles_source.py         # the drop-in for generate_smiles_dataset
    tests/
```

rdkit, numpy and stdlib only — **no torch, no `models/`, no GFN**. Verified importable while `parallel_synthesis` is broken.

**Output contract: `list[str]` of SMILES, and the package must never import `mxtaltools.dataset_utils.mol_building`.** The random conformer generator is expected to be replaced by a prior model from the conformer-GFlowNet work, so the SMILES→3D path is a component with a known replacement date; coupling to it would put a rewrite in this module's path for no benefit. Nothing in §3's control surface needs 3D — every Tier-T statistic is a graph property computed with `RDKit` 2D. **Add an import-surface test:** assert `mol_building` and any rdkit 3D entry point (`AllChem.EmbedMolecule`, `rdForceFieldHelpers`) are absent from `sys.modules` after importing `random_molecules` in a fresh interpreter. That test is the enforcement; a comment is not.

The drop-in must match the incumbent exactly. It is called positionally with three arguments at `parallel_synthesis.py:38` and `:94`, and with `seed=chunk_ind` at `generate_otf_crystal_chunk.py:30`:

```python
def generate_smiles_dataset(dataset_length, num_processes, smiles_dirs_path,
                            seed=1, spec=None, library=None) -> list[list[str]]:
```

Four binding requirements, each traced: **exactly `num_processes` chunks** (`modeller.py:1403` integrates only when the count matches — a different count deadlocks); **`str`, not `bytes`** (the incumbent's `.gz` branch emits bytes, which contaminate `MolData.smiles`); `smiles_dirs_path` must resolve because callers `os.chdir` into it first; and **honour `seed`**, which the incumbent does not — `np.random.seed(seed)` is commented out at `:154`.

---

## 6. Testing a random generator

The trap is that distributional tests either flake or assert nothing. Separate the two kinds:

**Per-sample invariants (∀, not statistics).** Validity, element set, connectivity, charge, size within the declared window. These hold for every sample or the generator is broken. Add from §2: **assert `n_heavy == GetNumHeavyAtoms()` per fragment**, and **assert the join bond order round-trips a known alkene** (`CC=CCc1ccccc1` decompose → reassemble → identical canonical SMILES). Either would have caught D1 or D2.

### The test tier for FG control specifically

Tier T's mechanism classes map onto test kinds, and mixing them is how a suppression test ends up asserting nothing.

**Per-sample invariants (DIRECT mechanisms only — these are ∀ claims because the mechanism is constructive):**
- every `forbid` / `at_most k` constraint holds on **every** emitted molecule, checked post-hoc even though it is enforced constructively. This is the only check that catches a hole in the join table or the carrier set.
- every ring system in every emitted molecule is in the library's ring-system vocabulary. This is the Tier-D claim of §3 T-D, and it is the test that fails the day someone adds a ring-forming move. Allow the canonicalisation escape explicitly: compare ring skeletons after a single stated canonicalisation, and pin the two known acridine forms in the test rather than widening the tolerance.
- `seed_size_aware` invariant (D3): for every accepted molecule, `n_heavy(seed) + Σ cap_heavy(open labels) ≥ target − window`.
- the import-surface assertion from §5.

**Distributional claims (TILT and SETPOINT mechanisms — ∃-in-expectation, so they need a stated test, a fixed seed, and a declared false-positive rate):**
- **knob monotonicity** on a declared β grid, per knob, with the grid stored in the test. Prefer this to absolute values, which are corpus-conditional (§8). The test must fail on a non-monotone grid rather than interpolate through it — that is the same assertion the solver makes at runtime.
- **ceilings, not 100 %**: a class-rate test asserts the realised rate reaches that class's *measured* ceiling (54–96 %, §3 T-B), never 1.0.
- **effective library floor computed on the realised draw**, recomputed against the library actually shipped. The previous guard's default was inherited from an 888-fragment library, measured a distribution the sampler never draws from, and was mathematically incapable of failing at the default spec.
- **Tier S reported alongside Tier T on every distributional test.** Budget: |bias| ≤ 1.5, sd ≤ 3.5. The tilt-collapse regime shows as −9.3 heavy atoms, so the size assertion is a cheap detector for a control failure that the FG assertion alone would pass.

**What "the acetal rate is suppressed" actually asserts — three clauses, and one of them alone is blind:**
1. **Exact zero, as an invariant.** `sum(mol.GetSubstructMatches(p) for p in ACETAL_FAMILY) == 0` for **every** molecule in a fixed-seed run of N = 2000 at each of targets 20/32/48. Because the mechanism is constructive (library filter + join predicate), the honest claim is 0, not "< 5 %". A threshold test here would pass on a broken guard.
2. **The mechanism is load-bearing** — re-introduce the defect and require a failure. With `forbid_join_environments` cleared *and* the 21 carrier fragments restored, the same fixed-seed run must show acetal-family prevalence **≥ 20 %** at target 32. Without this clause, clause 1 also passes on a library that happens to contain no acetal carriers, on a corpus swap, or on a generator that silently emits nothing.
3. **The cost is bounded** — yield ≥ 0.93, and heavy-atom bias/sd within (1.5, 3.5) of the guard-off run. Suppression achieved by rejection would satisfy clauses 1 and 2 at 41.6 % yield; this clause is what distinguishes the constructive mechanism from the expensive one.

Assert all three or the suppression test is blind.

**Three measurement hazards found while deriving §3; all are silent-wrong-answer shaped and all belong in the invariant list:**
- **M1 — deleting BRICS dummies to get a "bare fragment" fails on 8–12 fragments**, all label-9 N-heteroaromatics (`[9*]n1c2ccccc2c2ccccc21`, `[9*]n1ncc2c(O)ncnc21`, …): removing the dummy leaves a neutral 2-coordinate aromatic N with no H → `KekulizeException`. Any feature extractor wrapping that in `try/except: continue` gets a **silently all-zero feature row** for exactly those fragments — it contaminated an `aromatic_N` floor measurement (reported 0.160, true 0.000). **Normative capping rule, and it must be one rule: `carries(F, G)` is computed by setting each dummy's atomic number to 1, sanitizing, `RemoveHs` — 0/504 failures.** One probe used methyl-capping instead, which counts groups that only exist once joined to a carbon (`[3*]OC` reads as an ether under methyl-cap, as an alcohol under H-cap); its FG counts are therefore not comparable digit-for-digit with the floor table. Pin H-capping, and assert the carrier set size in the library build.
- **M2 — `mol.GetRingInfo()` on a temporary `Mol` is a use-after-free.** `strip(f.mol).GetRingInfo().NumRings()` returned **0 for all 504 fragments**; the identical computation with the mols materialised in a list returned **223**. No exception, no warning — it reports "no rings anywhere". Assert `sum(has_ring) > 0` in the library build; it is one line and it catches this class exactly.
- **M3 — a size projection that is not a fixed point of the policy it drives.** Two controllers produced +36 and −27 heavy-atom bias because the projection assumed each open slot would add `mean(terminal heavy)` while the capping policy then applied a size mask and drew *small* terminals. Fix: leave the cap draw unmasked so `E[cap] == cap_mean` by construction. Bias went to −0.17…−1.03. **Monotonicity testing would not have caught this** — it is an absolute-value defect, so the library build needs the explicit consistency assertion.

**Seed policy: explicit `np.random.Generator` from the start, never a global reseed.** The repo has an open determinism problem — 16 library `__init__`s call `torch.manual_seed` — and a new module must not add to it.

Two cautions the review surfaced. A "seeds are independent" test asserting zero set-intersection is a *collision* test, not an independence test; it passes 40/40 today but will start flaking the moment the spec narrows, and the natural response is exactly the loosening this suite should forbid. **Assert on the RNG stream instead.** And `np.exp` overflow is silent — `np.exp([1000,2,3])` gives `[inf, …]`, normalizing to `[nan, 0, 0]` — reachable at the tilt's β range; a guard that returns `None` converts it into an unexplained yield loss with no diagnostic.

---

## 7. Staging

**Stage 1 — library + assembler.** *Exit must now include the D1/D2 assertions above, the pinned build recipe of §2 (the count alone is not an exit criterion — 504, 730, 734, 1015 are all reachable from "BRICS-decompose NCI and CHNOF-filter"), the M1 capping rule, and the M2 ring assertion.* The original exit criterion ("returns 734 CHNOF fragments, `grow_one` returns valid unique molecules") is satisfied by the defective build. Stage 1 also now produces the tables Tier T consumes: ring-system vocabulary, FG carrier sets, the 11-class join table **with bond order**, and the precomputed attachment-environment array. Self-contained, no torch, **safe to stop indefinitely.**

**Stage 2 — size setpoint and rejection.** Delete `size_offset` and its calibration API (D2). **Fix D3 — score the seed** (without it, target 48 runs sd 6.56 with a 1st percentile of 23, and Stage 3 makes that worse by restricting the seed pool). *Stop: a complete, useful generator at ~1.0–2.1 ms/molecule.* **Best stopping point if the budget runs out** — and after D1/D2/D3, the first point at which stopping ships something correct.

**Stage 3 — Tier T, re-derived and re-scoped. The old Stage 3 is deleted, not amended.** Its numbers were the only ones in the design that could not be reproduced (the feature function `f(fragment)` was never defined; the obvious choice gave 0.372 against a claimed 0.455 at the design's own solved β = +5.34), its acceptance thresholds were hard-coded from those numbers, and the mechanism itself — a scalar heteroatom-fraction tilt with β by bisection — is now measured to be blind to joins (§4) and unsafe at the β it solved for (collapse regime, §3 T-B). Ship in this order, each sub-stage independently stoppable:

- **3a — T-D vocabulary control and the §4 suppression.** Ring-system allow/deny plus library filter plus the het-gem join predicate. Entirely constructive, exit is the three-clause acetal test in §6, and it removes the design's largest known chemistry defect. **If only one part of Tier T ships, ship this one.**
- **3b — T-C join-class weights.** The general form of 3a; unlocks the join-manufactured statistics that no fragment weight can reach (T-0 floor table). Exit: the join-class → FG yield table reproduces on the shipped library within its stated rates.
- **3c — T-B seed / ban / tilt for individual FGs.** Exit: monotonicity per knob on a declared grid; ceilings asserted rather than assumed; `min_seed_pool` and effective-library floors enforced; `SpecUnsatisfiableError` raised before any draw for every row of the attainability table that is out of range.
- **3d — T-A fragment-class IPF.** Last, because it is the only mechanism that costs the setpoint more than 1.5 heavy atoms (−2.4) and the only one with a measured divergence mode (per-signature classes: effective library 504 → 3). Exit: TV to target ≤ 0.05 with the class dimension capped at 10, plus the participation-ratio guard firing on a deliberately over-fine partition.

Recompute the effective-library-size floor against whichever build ships; the 888-derived default is invalid for both 504 and 734. Regenerate the attainability table (§3 T-0) as a build artifact rather than hard-coding it — every number in it is corpus-conditional.

**Stage 4 — the pipeline drop-in. First stage that modifies existing files; separate commit.** Carries two blocking repairs: delete the `sample_about_crystal` import at `parallel_synthesis.py:12`, and thread `do_mol_analysis=True` so the radius gate at `parallel_synthesis.py:226` stops raising `TypeError` on the first accepted sample. Budget a third: `parallel_synthesis.py:209-212` wraps `from_smiles` in a **bare `except: pass`**, so every generation, embedding and valence failure vanishes with no count and no message — any yield the generator reports is unverifiable at the consuming end.

**Stage 5 — measurement, not code, and its scope has shrunk.** The original Stage 5 measured generated `radius` — a 3D quantity that comes out of `mol_building.py`'s embedding path and becomes the autoencoder's `radial_normalization` buffer. **That measurement is no longer this module's to make.** The conformer generator is expected to be replaced by a prior model from the conformer-GFlowNet work, and §2/§5 pin the output contract at SMILES, so a Stage-5 number derived from the outgoing embedder would be a measurement of a component with a scheduled replacement, folded into a checkpoint identity.

Stage 5 therefore measures **only SMILES-level quantities**, all reproducible from a seeded `np.random.Generator` with no 3D step: heavy-atom and total-atom distributions, realised FG and ring-system rates against the corpus reference, effective library, uniqueness, yield and attempts/molecule per knob, throughput. The 3D-side measurement — radius distribution, embedding yield, MMFF strain — moves to whoever owns the conformer path, as a **consumer-side acceptance test on a delivered SMILES list**, not a stage of this design.

The caveat that motivated the original wording still stands and is handed over with it: `mol_building.py:243` calls `EmbedMolecule` with no `randomSeed` and `:98` uses the numpy global, so a fixed-seed SMILES list yields a **nondeterministic** 3D dataset and the reported `radius` max is a tail statistic over an unseeded sampler. **Thread `randomSeed`, or set the buffer from a quantile rather than a max** — in the conformer path, not here.

---

## 8. Scope honesty

What is solved is **heavy-atom count tracking**, **ring-system / larger-fragment content** (constructively, §3 T-D), and **the acetal family** (0.00 %, §4). What is *not* solved is producing chemically sound, embeddable, autoencoder-consumable large molecules. Two of the design's own measurements still say so: embedding yield falls to 86 % at heavy 48 (93 ms/molecule); generated radius at heavy 32 already exceeds the autoencoder's `radial_normalization` buffer, with reconstruction degrading badly past 30 atoms. Both are now consumer-side questions (§7 Stage 5), not this module's.

Three limits are structural rather than unfinished, and should not be re-litigated as tuning:
- **Join-manufactured content is not reachable from a fragment weight.** Ether, amide, ester, ketone, tertiary amine and alkene have non-zero suppression floors (0.35–0.82 matches/molecule with every carrier zeroed). Control for these lives at the join table or nowhere.
- **Any ring system absent from the library is unreachable at any β and any rejection budget** — assembly concatenates rings, it never forms one. Largest library ring is 7 atoms; macrocycles are out of scope by construction.
- **Total-atom (with-H) count remains uncontrolled**, sd 7.0–9.3 across targets 20–48, larger than the heavy-count controller's own contribution. Tier T moves its mean (production spec 32.8 / 53.6 / 82.5 vs 35.9 / 56.2 / 86.3 unconstrained) but not its scatter. Storing `n_atoms_with_h` per fragment and projecting total atoms is still the fix, and is still untested.

**The generator can hit a size target and now also a fragment-content target. Whether the molecules at that size are worth training on is a separate, open question.**

Every default here is conditional on the NCI-derived fragment corpus, and the conditioning is now larger than before: the size floor, the peroxide-free assumption, the acetal rate, the library-size floor, **the carrier sets, the attainability floors and ceilings, the class ceilings and the join-class yield table** all move with the corpus. Widening the corpus does *not* rescue a rare-feature knob — 1500 → 4999 NCI lines took F-carriers 4 → 13 and `≥2 fluoro` rejection only 247 → 46 attempts/molecule. **The library build path needs an assertion suite and a regenerated attainability artifact, not just a constructor.**

# CCDC obviation — CIF reader and Z′-any molecular extractor

**Status: Stages 0–2 APPROVED by the owner 2026-08-23, plus the CIF writer (§8). Stage 3 remains a separate decision.**
**Measured 2026-08-23. Corpus: `D:/crystal_datasets/CSD_dump`, 873 578 CIFs, random-shuffled with fixed seeds.**
**Interpreter: `C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe`.**

**Stated goal, in the owner's words: not just to obviate and speed up the CCDC workflows, but to gain "a workflow we trust and understand."** That is a stronger requirement than feature parity, and §1.5 makes it a design constraint rather than an aspiration.

This document is a **decision** plus the **observations** supporting it, per `AGENTS.md`. It was produced by an investigation pass and then corrected against an adversarial review that re-ran every load-bearing measurement at 2–10× the original sample size. Where the review overturned the original design, the corrected figure is given and the superseded one named, so nobody re-derives the retracted number.

**Read §1.5 first if you read only one section** — it is the part that serves the "trust and understand" goal rather than the parity goal.

---

## 0. The environment, verified

| Library | State |
|---|---|
| `ccdc` | **3.4.1, live and licensed** |
| `spglib` | 2.7.0 — imported at `crystal_ops.py:1873` (function-local), **not declared in `pyproject.toml`** |
| `ase` | 3.26.0 |
| `rdkit` | 2025.03.5 |
| `cctbx` | 2025.11 |
| `gemmi` | **not installed** |
| `pymatgen` | **not installed** |

*Table: parsing and chemistry libraries present in the project venv, tested by import, 2026-08-23.*

Two consequences. First, **CCDC being live is a gift, not an obstacle** — it means every claim below was checked against ground truth on hundreds of real structures, and the equivalence harness in §6 can actually run. Second, the two libraries most often reached for (`gemmi`, `pymatgen`) are absent, so proposing either is proposing a new hard dependency.

---

## 1. Scope

### 1.1 The finding that makes this tractable

Three measurements, and they are the spine of the whole design:

| # | Claim | Measurement |
|---|---|---|
| **M1** | **Z′ is derivable from the CIF alone, exactly.** `Z′ = _cell_formula_units_Z / len(_symmetry_equiv_pos_as_xyz)` | **0 mismatches in 1000** against `ccdc.Crystal.z_prime`. Operator counts also agree 1000/1000 |
| **M2** | **Molecular decomposition is derivable from the CIF alone.** Union-find over `_geom_bond_atom_site_label_1/2` reproduces `len(crystal.molecule.components)` | **194/200 (97.0 %)**. All 6 failures are metal/ionic/polymeric. **Zero organic failures** |
| **M3** | **No boundary reassembly is needed for CSD exports.** | **0/2000 molecules split across a cell boundary** (raw vs minimum-image distance). Bond symmetry codes are `1_555` in **3000/3000** files |

*Table: the three properties that determine whether a CCDC-free reader is a parsing problem or a perception problem. Verified against `ccdc` 3.4.1 as ground truth.*

**For the CSD corpus, the reader is not doing chemical perception at all — it is doing parsing.** Z′, the molecular decomposition, and geometrically whole molecules are all already in the file. That is the good news, and it is larger than expected.

*(Superseded: the original design justified M3 with "all bonds < 3.0 Å", which conflates long chemical bonds with boundary splits — 20/1000 files exceed 3.0 Å on I/metal bonds nowhere near a cell edge. The conclusion survives under the correct minimum-image test, and is stronger.)*

### 1.2 What gets replaced

| CCDC capability | Replacement | Evidence |
|---|---|---|
| Tokenizing, cell, atom sites | own tokenizer, ~250 lines | the needed CIF 1.1 subset is small and exactly testable |
| `symmetry_operators` | parse `_symmetry_equiv_pos_as_xyz` → 4×4 affine | operator-count agreement 1000/1000; cctbx set-equality cross-check |
| `z_prime`, `z_value` | **M1** | exact, 1000/1000 |
| `molecule.components` | **M2**, union-find over `_geom_bond` | 194/200, 0 organic failures |
| `packing()` (unit cell) | apply operators to component centroids, wrap centroid, rebuild atoms around it | reproduces CCDC to max 0.0000 Å; MXtalTools already implements this at `crystal_building/utils.py:646-714` |
| `all_atoms_have_sites` | missing-coordinate check | trivial |

### 1.3 What does not get replaced

One thing, stated plainly:

> **Bond-order, aromaticity and formal-charge assignment, and de-novo hydrogen placement, for a structure whose molecular identity is not otherwise known.**

The open stack's measured ceiling:

- **Heavy-atom skeleton — solved.** On CSD exports you don't even need perception; `_geom_bond` is exact for organics (M2).
- **Bond orders with hydrogens present — 91.1 % exact** (n=1132), 8.9 % silent disagreement with CCDC.
- **Bond orders from a heavy-atom-only structure — not solved, and worse than it looks.** 232/300 raise, but **61/300 return a molecule**, and **40 of those carry no radicals**. `BARBOL` yields `Fc1c(Cl)c(F)c(Cl)c(F)c1Cl` — a fully sanitized, radical-free, entirely plausible aromatic SMILES derived from a hydrogen-free skeleton. *A chemically wrong molecule that still looks like a molecule.* **This makes the H-present precondition a hard gate that must be enforced, not a formality.** *(Superseded: the original design said 0/300 and treated the refusal as a formality over a path that "would fail anyway". That was wrong and in the dangerous direction.)*
- **Bond orders + hydrogens with the molecule known (template SMILES) — solved, 250/250.**
- **Hydrogen placement — see §1.3.1. The original "0.7 % solved" figure conflated two different problems and is retracted.**

### 1.3.0 Bond orders are recoverable from GEOMETRY — ⚠ this supersedes much of §1.3

**Measured 2026-08-24, and it overturns the framing above.** §1.3 concluded that bond
orders cannot be inferred from a heavy-atom skeleton, on the evidence that
`rdDetermineBonds.DetermineBondOrders` fails there (232/300 raise, 61/300 return a
plausible wrong molecule). That evidence was misread: **`DetermineBondOrders` is a
valence/Kekulé search that ignores coordinates entirely.** Its failure says nothing about
whether the information is present in the structure.

It is present. C–C bond lengths over 150 organic CSD crystals, against CCDC's own
perceived orders:

| bond type | n | mean (Å) | std | p5–p95 |
|---|---|---|---|---|
| Single | 1944 | **1.508** | 0.040 | 1.425–1.555 |
| Aromatic | 1943 | **1.390** | 0.021 | 1.359–1.424 |
| Double | 152 | **1.356** | 0.035 | 1.314–1.416 |
| Triple | 7 | **1.196** | 0.008 | 1.184–1.203 |

*Table: C–C bond length by CCDC-perceived bond type, 150 random organic CSD crystals
(elements ≤ Cl), 2026-08-24. Separation: single vs double **2.9 pooled sd**, single vs
aromatic **2.6**, aromatic vs double **0.8**.*

**And CCDC is doing the same thing.** Reading a CIF, `crystal.molecule.bonds` returns
`Unknown` for **every** bond (71/71 on KUNTEV) — CCDC has no more bond-order data from the
file than this reader does. It calls `assign_bond_types()`, which perceives orders **from
geometry**. What looked like an irreplaceable CCDC capability is a bond-length and
planarity heuristic, and `assign_bond_types()` is therefore available as a calibration
oracle over the whole corpus.

**Consequence: a fourth chemistry strategy is viable**, and it breaks the H-free
circularity from the other side — geometry gives bond orders, bond orders give hydrogens:

    GeometryChemistry   bond lengths + ring planarity -> bond orders
                        -> AddHs(addCoords=True) for the H-free 6.3 %

**The one hard discrimination is aromatic vs double at 0.8 sd** — overlapping, and
chemically expected since an aromatic bond is intermediate. Bond length alone cannot
separate them; ring perception must (a bond in a planar 6-ring with all lengths ≈1.39 Å is
aromatic whatever its own length says). Any implementation should be validated on that
case specifically, not on the easy single/multiple split.

*Still true from §1.3: with hydrogens present, valence-based inference works at ~91 %.
The correction is only that H-free is not a dead end.*

### 1.3.1 Hydrogen placement, decomposed

The owner's question was: *"is there no RDKit equivalent? Or deterministic approximate function? I can't imagine what ccdc does internally is that sophisticated."* **Correct on both counts.** The original design's "de-novo H placement is not solved, 2/300" measured the *joint* problem — infer bond orders from a heavy-atom skeleton **and** place hydrogens — and attributed the failure to the wrong half. Separated:

**(i) Infer bond orders from a heavy-atom-only skeleton — genuinely hard and genuinely dangerous.** 61/300 return a plausible wrong molecule (§1.3). This is the real hard core.

**(ii) Place hydrogens given known bond orders — easy, deterministic, and already available.** `Chem.AddHs(mol, addCoords=True)`, measured over 10 drug-like molecules (embed → MMFF optimise → strip H → replace H, comparing against the optimised truth):

| molecule class | mean H displacement |
|---|---|
| rigid aromatics (acridine, chlorobenzophenone, benzoic acid, ethyl benzoate) | **0.015 – 0.058 Å** |
| with rotatable X–H (phenol/OH, amide NH, sulfonamide, sugar hydroxyls) | 0.22 – 0.55 Å (max 1.83 Å) |
| **overall, 10/10 succeeded** | **0.256 Å** |

*Table: hydrogen replacement accuracy using `AddHs(addCoords=True)` with bond orders known. "Truth" is the MMFF-optimised all-atom geometry; displacement is per-hydrogen nearest-neighbour distance, since H ordering differs.*

The split is chemically exactly what you would expect and it is the useful statement:

> **H positions determined by hybridization are placed essentially perfectly. H positions on a rotatable X–H bond — hydroxyl, amine, methyl — are a free torsional parameter, and RDKit sets them to a default rather than the crystallographic value.**

That residual is the only thing CCDC's `add_hydrogens` could plausibly be doing better, and for a crystal it would need to be doing it with hydrogen-bonding context. **For the packing-coefficient, RDF, eLJ and MLIP paths, a 0.26 Å mean H error on rotatable groups is almost certainly immaterial** — but it is a claim to measure, not assume: score the same crystal with deposited H versus replaced H and report the eLJ delta before relying on it.

### 1.3.2 Bug report: `protonate_mol` is broken

The owner recalled having an approximate function. It exists — `crystal_building/utils.py:840`, used by `dataset_utils/construction/protonate_crystal_dataset.py` — and it is **wrong in the silent way**.

It builds an `RWMol` by adding one atom per input atomic number and **never calls `AddBond`**. Every atom therefore has valence 0, so `AddHs` saturates each one independently:

```
benzoic acid heavy skeleton (9 atoms, C7O2)
  protonate_mol returns: 41 atoms, 32 hydrogens
  correct answer:                    6 hydrogens
```

It produces seven methanes and two waters at the heavy-atom positions, and returns well-formed `(atom_types, coords)` arrays that look like a protonated molecule. **Fix: pass the bond graph in.** With `_geom_bond` connectivity available from the CIF (M2), the correct version is the same three lines with the bonds added — which is precisely path (ii) above.

Hydrogens are present in **93.7 %** of the corpus (562/600), so the H-present path covers most of it. Note that both in-repo CIF fixtures (`examples/DAFMUV.cif`, `crystal_search/aaa.cif`) and `ACRDIN.cif` fall in the 6.3 % *without* — the fixtures are unrepresentative in exactly the direction that matters.

### 1.4 Scope decision

The honest scope line is **not** "CCDC does chemistry, we do geometry." It is:

> The reader is CCDC-free whenever the deposit carries hydrogens (93.7 %), the structure is organic (see below), and you accept ~91 % chemistry agreement with a guard that converts most of the residual into refusals. It is **not** CCDC-free for the H-free 6.3 %, nor for any pipeline whose default is `protonation_state='protonated'` — that is de-novo H placement on inferred bond orders, and has no open-stack equivalent.

**And a fact the original design hid in its denominator: 35.3 % of what the reader accepts is organometallic** (152/431 — Fe, Cu, Pd, Mo, Pt, Zr, Co, Cd, Ta, Al…). Feeding those skeletons to the chemistry backend raises on 138/400 and **silently returns two wrong molecules at formal charge 0** (`ISIJUS` as a Bi aryl, `ZERRAS` as an Si/Sn chain). True corpus yield is therefore **≈ 59 % pre-guard and ≈ 41 % post-guard**, not the "~63 % of organics" originally claimed.

So the choice is:

1. **Geometry-only obviation (Stages 0–2).** Fully achievable, no residual risk, covers every crystal-geometry, packing-coefficient, RDF, eLJ-without-charges and MLIP workflow. For GFN and CSP — where the molecule is always known — a template backend obviates CCDC *completely*.
2. **Bulk-ingestion obviation (add Stage 3).** Buys licence-free CSD ingestion at ~41 % corpus yield and a residual ~1.8 % silent-chemistry error rate.

**Decided 2026-08-23: (1). Stages 0–2 proceed; Stage 3 stays a separate decision, taken after Stage 4's equivalence harness produces numbers.**

Note that §1.3.1 improves option (1) beyond what was originally scoped. With bond orders known — which is the Stage 2 template case, and is also true for GFN and CSP where the molecule is always known — hydrogen placement is *also* obviated, at ~0.26 Å. The "protons are a real problem" caveat applies specifically to inferring bond orders *and* hydrogens together from a bare heavy-atom skeleton, which is the Stage 3 case.

### 1.5 The vocabulary is part of the deliverable

The owner's diagnosis: the existing featurizer is confused about `components`, `asymmetric unit`, and `molecule` in the CCDC namespace — and that confusion is inherited, not invented. A reader that reproduces CCDC's behaviour while leaving the vocabulary ambiguous fails the stated goal even if every number matches.

So **the reader defines its own terms explicitly, and every function signature uses them consistently.** Proposed, to be fixed in `errors.py`/`reader.py` docstrings and asserted in tests:

| Term | Definition in this reader | Not to be confused with |
|---|---|---|
| **site** | one `_atom_site` row: label, element, fractional coordinate | an atom in a built structure |
| **deposited set** | all sites in the file, exactly as written | the crystallographic asymmetric unit — the CSD exporter often writes *whole molecules* at special positions, so these differ whenever Z′ < 1 |
| **component** | one connected sub-graph of the deposited set under `_geom_bond` | CCDC's `components`, which is the same idea but computed by CCDC's own perception |
| **molecular component** / **ionic component** | a component, classified by net formal charge once chemistry is assigned | — |
| **Z′** | `_cell_formula_units_Z / len(symops)`, an exact rational | "number of components in the deposited set" — these coincide only when Z′ is a positive integer |
| **unit cell** | the deposited set expanded by all symmetry operators, centroid-wrapped | the P1 cell that `write_cif` currently emits |

### 1.5.1 Where the site / component / asymmetric-unit / molecule confusion comes from — MEASURED

The owner's question: *"is it problems in the files, CCDC, or with us?"*

**Answer: it is CCDC's vocabulary describing a real crystallographic distinction. Not the files, and not us.** CCDC exposes *two* objects both called some kind of "molecule", and they are genuinely different things. Measured over 300 random CSD entries, comparing three views of "what is in the file":

| comparison | agree |
|---|---|
| deposited set == `crystal.molecule` | **97.3 %** |
| deposited set == `crystal.asymmetric_unit_molecule` | **74.7 %** |
| CCDC's own two views agree with each other | **74.7 %** |

*Table: three views of the same crystal, over 300 random CSD entries, 2026-08-24. "Deposited set" is our reader's — the `_atom_site` rows verbatim. Agreement requires both atom count and component count to match.*

Broken down by Z′, the mechanism is unmistakable:

| | deposited set == `asymmetric_unit_molecule` |
|---|---|
| Z′ < 1 | **0 / 59 — never** |
| Z′ = 1 | 203 / 214 (94.9 %) |
| Z′ > 1 | 24 / 27 (88.9 %) |

*Table: the same 300 entries partitioned by Z′. The Z′<1 row is the whole story.*

So the four terms name four genuinely distinct things, and conflating any two is wrong for a measurable fraction of the corpus:

- **site** — one `_atom_site` row.
- **deposited set** — all of them, as written. What the depositor chose to list.
- **`crystal.molecule`** — the chemical molecules present. Matches the deposited set 97.3 % of the time, so it is *nearly* a synonym, which is exactly why it misleads.
- **`crystal.asymmetric_unit_molecule`** — the **crystallographic** asymmetric unit. At Z′<1 a molecule straddles a symmetry element, so the ASU is a *portion* of a molecule and this never equals the deposited set.
- **component** — a connectivity notion, orthogonal to all of the above.

Two refinements worth keeping:

- **The Z′<1 ratio is not a clean fraction.** `|AUM| / |deposited set| == Z′` in only **18/59 (30.5 %)** of cases, because atoms lying *on* the symmetry element are shared rather than halved. So "at Z′=1/2 the ASU is half the atoms" is wrong; you cannot recover the ASU by arithmetic.
- **A second, smaller mechanism operates at Z′=1** (11/214). CCDC's ASU drops 4–74 atoms with `has_disorder == False` — this is the *"extra (erroneous or dubious) atoms in silly places"* the incumbent featurizer's own comment at `featurization_utils.py:110-112` describes. Two of them (IPTCNI18 98→50, FOGDIO 148→74) drop exactly half, suggesting a special position that the declared Z′ does not reflect.

**Consequence for this reader:** it commits to the **deposited set** and says so in the type name. It never claims to produce an asymmetric unit, because at Z′<1 it does not have one, and deriving one requires deciding which atoms lie on the symmetry element — a separate computation this reader does not do. `CifCrystal.deposited_set_is_asymmetric_unit` returns that fact rather than leaving the caller to assume.

**A third term, found while doing Stage 0 and not previously written down anywhere:**

| Term | Definition | Measured |
|---|---|---|
| **centroid** | throughout the crystal code this means the **heavy-atom** centroid, not the all-atom centroid | `recenter_molecules()` zeroes the heavy centroid exactly (0.000000) and leaves the all-atom centroid at 0.69 Å; `pose_aunit()` places the heavy-atom centroid at `aunit_centroid` exactly (0.000000) while the all-atom centroid sits up to 0.14 fractional away |

*Table: which centroid the crystal code actually uses, measured on a 21-molecule test batch, 2026-08-23.*

The code is entirely self-consistent about this. **Nothing states it**, and `tests/test_data_classes_basics.py` asserted the all-atom centroid in two separate places — so both assertions were unsatisfiable, and neither had ever run. This is the vocabulary problem in miniature: a convention that is correct, uniform, undocumented, and therefore untestable by anyone who did not write it.

Two further consequences worth stating because they are the actual source of the confusion:

- **Z′ < 1 is 19.6 % of the corpus** (98/500; mostly 0.5), and there the deposited set is *not* an asymmetric unit — it is a whole molecule sitting on a special position, with the crystallographic ASU being half of it. Any code that treats "what is in the file" as "the asymmetric unit" is wrong for one structure in five.
- **`components` is a perception result, not a fact about the file.** M2 says the CIF's own `_geom_bond` loop reproduces it for organics, which is why this reader can be trusted where CCDC's answer is opaque — the bond list is data in the file, auditable line by line.

The test suite should assert the vocabulary: for each fixture, the expected site count, component count, and Z′ are written down separately, so a change that conflates two of them fails.

---

### 1.6 Cocrystals — deferred, but the data is captured now

**Not being built.** Cocrystal support needs a `MolCrystalData` refactor (its docstring
at `data_classes.py:707` says "exactly one molecule in asymmetric unit"), and that is a
separate project. What the reader does now is *capture* what such an implementation will
need, so it is not re-derived later.

`_chemical_formula_moiety` is present in **400/400** sampled CSD entries and is
multi-unit in **40 %** of them (27.2 % two moieties, 11.2 % three, 1.5 % four). It carries
three things a component decomposition cannot:

| | example | meaning |
|---|---|---|
| stoichiometry | `2(H2 O1)` | a dihydrate |
| formal charge | `C2 H5 N4 1+` | a **salt**, not a neutral cocrystal |
| species identity | `C7 H8 N4 O2,C6 H7 B1 O3` | the depositor's own statement |

*Table: what `_chemical_formula_moiety` encodes, from a 400-entry CSD sample, 2026-08-24.*

**The generalised invariant, verified on real cocrystals:**

> `n_components == sum(moiety coefficients) × Z'`

| identifier | components | Z′ | Σ coeff | predicted | moiety |
|---|---|---|---|---|---|
| MORFOP | 6 | 2 | 3 | **6 ✓** | `C13 H14 N6 O5 V1, 2(H2 O1)` |
| KIVYAS | 4 | 1 | 4 | **4 ✓** | `C16 H14 B2 N2 O10, 3(H2 O1)` |
| LALCOQ | 2 | 1 | 2 | **2 ✓** | `C2 H5 N4 1+, C5 H2 N1 O5 1-` (salt) |
| ULUREA | 2 | 1 | 2 | **2 ✓** | `C7 H8 N4 O2, C6 H7 B1 O3` (cocrystal) |
| FOWHEH | 3 | 1 | 3 | **3 ✓** | salt + chloroform |
| DIRSON | 5 | 2 | 2.0 | 4 ✗ | `…, 0.75(C4 H10 O1), 0.25(C6 H14)` — **fractional** |

*Table: the cocrystal generalisation checked against real structures. The reader's current
check is this formula with the sum hard-wired to 1, which is precisely why the incumbent
filter "always kills cocrystals" (`featurization_utils.py:304-306`, its own comment).*

**Captured:** `CifCrystal.formula_moiety` (raw), `.moieties` (parsed to species /
coefficient / charge), `.component_formulas()` (Hill formula per component — the bridge
from geometry to species; KIVYAS returns `['C16 H14 B2 N2 O10', 'H2 O1', 'H2 O1', 'H2 O1']`).

**The hard case, recorded so it is not met late:** DIRSON declares `0.75(C4 H10 O1)` and
`0.25(C6 H14)` — **fractional solvent occupancy**. That is disorder wearing a
stoichiometry coat, and no whole-molecule representation can hold it. `Moiety.is_fractional`
flags it. Any cocrystal work should decide its policy on this class before starting, not
on discovering it.

---

### 1.7 Chemistry backend — DECIDED 2026-08-24

**Approved: `TemplateChemistry` + `NoChemistry`. `GeomBondChemistry` DEFERRED.
`GeometryChemistry` REJECTED on measurement.**

| strategy | for | quality | silent-error mode |
|---|---|---|---|
| `TemplateChemistry` | caller supplies `{identifier: SMILES}` — GFN, CSP | 100 % | **none** |
| `NoChemistry` | geometry-only workflows | n/a | none; declares itself |
| `GeomBondChemistry` | H present, 93.7 % of corpus | ~91 % vs CCDC | **~1.84 % — DEFERRED** |
| `GeometryChemistry` | H-free, 6.3 % | **14 %** | **rejected** |

**Why `GeometryChemistry` was rejected.** §1.3.0 showed bond lengths separate by order at
2.6–2.9 pooled sd, and concluded the H-free case was tractable. **That conclusion was too
strong.** Built and measured on 250 held-out crystals with hydrogens removed:

    built a sanitizable molecule       48.0 %
    canonical SMILES matches CCDC      14.0 %
    AtomValenceException               126/250

Separable distributions do not give a molecule. The aromatic/double overlap (0.8 sd) is the
majority of organic bonding — 1292 of 3479 aromatics were called Double — and per-bond
errors compound multiplicatively, so 77.9 % per-bond became **15.6 % per-molecule**. A
production-grade perceiver needs bond angles, hybridisation from local geometry, ring-system
aromaticity propagation and formal-charge assignment: a research effort, not an afternoon.
*The harness and the free oracle (`assign_bond_types()` over 870k structures) now exist if
it is ever wanted.*

**Why `GeomBondChemistry` was deferred rather than built.** Three reasons, none of them
technical difficulty:

1. It introduces a **~1.84 % silent error** — wrong SMILES, fingerprint and partial charges,
   with no exception and no flag — into any dataset built with it.
2. The 91 % is **agreement with CCDC, not correctness**. `assign_bond_types()` is itself a
   geometry heuristic; both can be wrong together.
3. It buys **no throughput**: ~41 % post-guard yield against ~43 % for the incumbent CCDC
   path. And the guard's trade is poor — 83 wrong structures caught by rejecting 73 correct
   ones.

**The decided position has no silent-failure path at all.** Template is exact where it
applies; NoChemistry declares its own absence. Bulk CSD ingestion keeps CCDC, which the
owner holds a licence for. Revisit if licence-free bulk ingestion becomes a goal.

---

## 2. Module layout

New package. Nothing outside it imports `ccdc`.

```
mxtaltools/dataset_utils/construction/cif/
    __init__.py          # re-exports read_cif, CifReadError and subclasses
    errors.py            # every refusal is a named exception
    tokenizer.py         # CIF 1.1 -> blocks/loops. no chemistry, no numpy
    symmetry.py          # xyz-op strings <-> 4x4 affine; SG identification
    sites.py             # atom-site loop -> labels/elements/frac coords/occupancy
    topology.py          # bond graph -> components
    chemistry.py         # ChemistryBackend protocol + Geom/Template/No backends
    ccdc_backend.py      # CCDCChemistry; the ONLY module importing ccdc
    reader.py            # public entry: read_cif -> CifCrystal
    adapt.py             # CifCrystal -> (crystal_dict, molecule_dicts)  [the seam]
```

**Ship an explicit `__init__.py`.** The sibling `construction/` has none and works only as a PEP-420 namespace package — which is why a `pkgutil` module walk reaches 46 % of the package. Do not inherit that.

### Why a hand-rolled parser rather than ASE

**ASE is disqualified on measurement, and the reason is this repo's signature failure mode.** `ase/io/cif.py:378` is verbatim `elif no is not None: spacegroup = no` — it takes `_symmetry_Int_Tables_number` and never consults the H-M symbol when no symop loop is present. The acridine tree holds **8576 CIFs**; **87.3 % of sampled files carry IT number 1 with a non-P1 H-M symbol**, and ASE reads every one as P1, silently, with no diagnostic.

`cctbx` handled every file correctly and preserves occupancies where CCDC drops them, but it is a ~1 GB stack whose presence here is incidental. **Use it as a test-only cross-check, not a runtime dependency.**

### The seam

`featurize_cif_chunks.process_chunk` gains one keyword defaulting to today's behaviour:

```python
def process_chunk(chunk, chunk_ind, use_filenames_for_identifiers,
                  protonation_state, max_z_prime,
                  reader_backend: Literal['ccdc', 'native'] = 'ccdc'): ...
```

The `'native'` branch produces the same `crystal_dict` and re-enters the existing, unmodified downstream (`init_zp1_crystals` → `extract_zp1_pose_info` → `pose_aunit`/`build_unit_cell` → `crystal_rebuild_checks` → `instantiate_crystal`), all of which are pure torch/numpy.

---

## 3. Failure policy — what the reader must refuse

The repo's characteristic defect is a finite, plausible number with no diagnostic. Every case below raises a named exception rather than guessing. **The review found five of the original design's nine refusals never fire on the target corpus**, so they are marked here rather than sold as protection.

| Refusal | Fires on CSD_dump? | Note |
|---|---|---|
| **H-free structure reaching bond-order perception** | **enforce hard** | 61/300 return a plausible wrong molecule. This is the most important gate in the design |
| **Non-organic element** (allow-list: H,B,C,N,O,F,Si,P,S,Cl,Se,Br,I) | **35.3 % of accepts** | Absent from the original design. Two metals returned silently at charge 0 |
| **Polymer** via `_chemical_formula_moiety` matching `\)\s*n\b` | 3/431 accepts | The *working* discriminator |
| ~~Polymer via non-`1_555` bond code~~ | **0 / 3000 files** | **Welded shut.** All 12 CCDC-polymeric structures in a 1500 sample pass this gate. Keep only as protection for non-CSD input, and label it as such |
| **Any-atom formal charge not corroborated by moiety formula** | 83 caught / 73 false rejects | See §4 |
| ~~Net formal charge≠0~~ | **0 of 101 errors** | **Retracted.** Net charge is 0 in 1132/1132 cases; the failures are zwitterionic |
| Non-integer Z′ | 19.6 % of corpus | Real and load-bearing |
| ~~Special-position stabiliser (tol=0.05)~~ | 0 / 431 | Redundant with the integer-Z′ test, and uncalibratable — no positives exist to calibrate against |
| ~~Occupancy / disorder-group~~ | tag present in **0 / 1000** | The CSD exporter strips them. Must be a *refusal to assert*, not a `False` |
| ~~Operator closure under composition~~ | 0 / 2000 | Cheap; keep, but do not market it |

*Table: refusals ranked by measured firing rate on the target corpus. Struck-through rows fire zero times and must not be presented as protection — that is the `feedback_no_runtime_gate_on_a_retired_key` pattern.*

---

## 4. The chemistry guard, corrected

The original design's headline mitigation was "refuse when net formal charge ≠ 0 and the moiety formula does not corroborate it," claimed to convert 7 of 8 silent errors into refusals.

**Measured over 1132 organic components: it catches 0 of 101 disagreements.** Net formal charge is 0 in *every* case, because the failures are zwitterionic hypervalent S/P perceptions — `[O-][S@](#[N+]…)` where CCDC sees `S(=O)`. A neutral moiety string corroborates all of them.

Respecified as **any-atom** formal charge:

| | count |
|---|---|
| catches (wrong and refused) | 83 |
| false rejects (right and refused) | 73 |
| **silent wrong surviving** | **18 → 1.84 % of accepted** |

*Table: guard performance over 1132 organic H-bearing components, `_geom_bond` skeleton → `DetermineBondOrders(charge=0)`, canonical SMILES compared against CCDC's perception.*

Also retracted: the claim that a moiety-formula check is "an independent oracle." `DetermineBondOrders` never changes stoichiometry, so the perceived formula equals the input skeleton's *by construction*. It can only re-test charge. **The residual class is bond-order error at net-zero charge, and nothing in the design can see it.** That is the honest statement, and it is why datasets must record which backend produced them.

---

## 5. A blocking defect: the reader cannot read this package's own output

`ase_write_cif` (`common/ase_interface.py:170-179`, behind `write_cif`) emits the **CIF2 spellings**:

```
_space_group_IT_number          259/259 GFN-written CIFs
_space_group_name_H-M_alt       259/259
_space_group_symop_operation_xyz 259/259
_cell_formula_units_Z           ABSENT from all 259
```

The design as originally specified handles only the deprecated `_symmetry_*` spellings, so its flagship space-group contradiction check is blind on every file this package writes — including `examples/DAFMUV.cif`. And with `_cell_formula_units_Z` absent, M1 cannot compute Z′, so `read_cif(write_cif(x))` raises.

**Fix before Stage 1 ships:** accept both spellings, and decide explicitly what `read_cif` does with a Z-free ASE dump (derive Z from site count and operator count, or refuse loudly — either is fine, silence is not).

---

## 5b. The round-trip check is the reader's acceptance test

The featurizer already contains the right idea, and the reader should reuse it rather than
invent one: `crystal_rebuild_checks` (`featurize_cif_chunks.py:226`). **The deposited unit
cell is a ground truth that needs no external oracle** — not CCDC, not a fixture corpus,
just the file. Three layers:

1. **Reparameterization round trip** — pose → build → reparameterize, comparing
   `aunit_centroid`, `aunit_orientation`, `aunit_handedness`, `is_well_defined` against
   what went in (rtol 1e-2).
2. **Physical comparison against the deposited cell** — built vs deposited coordinates
   under minimum image, requiring `max nn-distance < 0.05 Å`, `mean < 0.01 Å`, **and
   `single_matches`: a bijection**. The bijection requirement is the load-bearing part; it
   rules out the degenerate case where many built atoms collapse onto one deposited atom
   while every individual distance looks small.
3. `validate_cell_params(check_crystal_system=True)`.

**The gap: the atom-type match is commented out** (`:261`). The bijection is positional
only, so a match that pairs a carbon with a nitrogen passes. This matters most exactly
where the plan predicts trouble — `z.repeat(sym_mult)` **tiles**, so at Z′>1 a
component-major deposited ordering against an image-major built ordering mismatches 50 %
of element slots while every position still finds a partner.

**Two findings from trying to measure that gap** (250 random CSD entries, 2026-08-23):

- **~1 % of crystals crash the pipeline, and the crash kills the whole chunk.**
  `crystal_building/utils.py:452` builds `unit_cell_atom_types` as
  `z[batch == i].repeat(sym_mult[i])`, assuming `n_ucell == n_aunit × sym_mult`. When that
  fails, `:455` raises `IndexError` — **uncaught by `process_chunk`**, so every crystal
  already processed in that chunk is lost. 1 of 100 fully-featurized entries.
- **Root cause on the observed case (HOBFAH, `C21 H25 O11 P1 Pt1 Ru3`, P2₁/n, Z′=1): a
  metal hydride counted inconsistently by the two legs.** The molecule leg's
  `remove_hydrogens()` strips all 25 hydrogens leaving 37 heavy atoms; the unit-cell leg
  retains the one Pt/Ru-bound hydride, giving 38 per image and 152 where 148 was expected.
  Note also that `remove_hydrogens()` mutates the molecule but **not** the crystal's
  `packing()`, which still returns all 62 atoms per image — so the two legs are deprotonated
  by different mechanisms and can disagree.

**Consequence for the reader:** its acceptance test is this round trip with the atom-type
check **enabled**, and the identity `n_ucell == n_aunit × sym_mult` becomes a checked
precondition that *refuses* rather than an assumption that raises `IndexError`.

## 6. Test plan

**Without a licence** (must be the bulk): tokenizer round-trips including `1.234(5)` esd stripping and `;`-delimited values; operator parsing against cctbx; Z′ arithmetic on fixtures with known Z′ ∈ {1, 2, 0.5}; union-find components on hand-built bond graphs; the ASE P1 trap as an explicit regression (**assert we do *not* reproduce it**); every refusal in §3 asserted to *raise*, with a matching negative test asserting it does not fire on a clean structure.

**With a licence** (`pytest.importorskip('ccdc')`): per-field agreement against the CCDC path at scale, publishing the table rather than asserting a threshold. Z′ and operator count should be exact; chemistry should be reported as a rate.

**Fixture discipline.** Both in-repo CIFs lack hydrogens, which is the 6.3 % case. **Add H-bearing fixtures, a Z′=2 case, a Z′=0.5 case, an organometallic, and a polymer** — otherwise the suite passes while blind to 93.7 % of the corpus.

**A hydrogen-stripping step is missing from the contract.** The incumbent's default (`protonation_state='deprotonated'`) removes hydrogens and stores heavy atoms only — real shipped output shows `nH=0` throughout. Any equivalence test asserting exact `num_atoms` fails on 93.7 % of structures unless the reader strips H after perception. That step must be in the algorithm, not assumed.

---

## 7. Staging

**Stage 0 — preconditions.** Replace `tests/test_crystal_workflow.py`, which is permanently red (missing drive colon at `:24`, undefined `cif_path` at `:25`) and is the only file in `tests/` importing `ccdc`. Declare `spglib` in `pyproject.toml`.

> ⚠ **Exit criterion, corrected 2026-08-23.** This stage originally exited on *"the suite is CCDC-free and green."* **That is an unbounded criterion** and it should not be restored. In a tree where whole paths have not executed in months, every fix exposes the next never-run branch: reaching a green suite took five repairs across `mol_methods.py`, `crystal_ops.py`, `crystal_opt_utils.py` and three test files, none of them CIF work. Those repairs are recorded in `refactor_plan.md` WS-0, which is where they belong.
>
> **Correct exit: `tests/test_crystal_workflow.py` collects, and runs or skips cleanly, and `spglib` is declared.** Nothing about the rest of the suite. Every later stage states its exit over the artifact being built — the parser, the writer, the fixtures — never over the state of the repository.

**Stage 1 — geometry-only reader, integer Z′, `NoChemistry`.** Produces geometrically correct `MolCrystalData` with `smiles=None`, `x=zeros`. **Verified end-to-end by the review**: such an object constructs, `mol2ucell` works, and `analyze(['reduction_en','elj'])` returns sane values. *Stop: geometry is CCDC-free, CCDC remains the default, nothing downstream changed.* **Genuinely abandonable.**

**Stage 2 — `TemplateChemistry`.** Caller supplies `{key: SMILES}`. *Stop: for any workflow that knows its molecules — GFN, CSP — CCDC is fully obviated.* Two cautions: the default `key='formula'` is ambiguous for 1.8 % of molecules, and **acridine collides with an unrelated alkyne under `C13H9N`** — key on identifier, not formula. And the 250/250 benchmark is circular if the templates came from CCDC's perception of the same structures; re-run it against independently sourced SMILES.

**Stage 3 — `GeomBondChemistry` + the corrected guard.** Only if §1.4 chooses bulk ingestion. Yield ≈ 41 % of corpus post-guard, residual ≈ 1.84 %.

**Stage 4 — CCDC as a pluggable backend + the equivalence harness.** *Stop: both readers coexist, divergence is quantified, switching the default is a decision backed by numbers.*

**Stage 5 — special positions and co-crystals. Do not start this inside the reader project.** It requires `sym_mult` to become per-component and the `z.repeat(sym_mult)` sites to become molecule-major gathers. **There are at least eight such sites, not four** (`ase_interface.py:134`, `crystal_building/utils.py:452`, `crystal_reduction.py:29`, `parallel_synthesis.py:345`, `crystal_ops.py:1894`, `AL_mace_utils.py:69`, `uma_utils.py:640`). Note `.repeat()` **tiles**; it does not interleave — the repo documents this correctly at `AL_mace_utils.py:580-583`. At Z′=1 the code is correct; the defect is Z′>1 block-tiling versus component-major, and it mismatches 50 % of element slots at Z′=2. This is a change to the *data model*.

---

## 8. The CIF writer — BUILT 2026-08-23

**Status: done.** `mxtaltools/common/cif_io.py` (new, depends only on numpy/torch + the package's own symmetry tables), reached via `MolCrystalData.write_cif(..., mode='asymmetric unit')`, which is now the default. Tests in `tests/cif/test_cif_writer.py`: 12 passing, **5/5 mutation kill rate**.

**Measured, incumbent vs new** — four crystals built in P-1, P2₁/c, P2₁2₁2₁ and Pbca, written, then read back with `ccdc` 3.4.1 as an independent reader:

| writer | space group CCDC recovers | operators | cell params |
|---|---|---|---|
| incumbent (`ase_write_cif`, `mode='unit cell'`) | **P1 for all four** | **1 for all four** | — |
| new (`mode='asymmetric unit'`) | P-1, P2₁/c, P2₁2₁2₁, Pbca — **all correct** | 2, 4, 4, 8 — **all correct** | max delta **4.8e-7** |

*Table: what survives a write→read cycle, verified against CCDC on four crystals spanning four space groups, 2026-08-23. Z and Z′ also round-trip exactly. Atom counts equal the asymmetric unit, not the expanded cell.*

**The strongest licence-free assertion**: every operator of every space group in `SYM_OPS` is rendered to a CIF `x,y,z` string and re-parsed by an *independently implemented* parser in the test, with exact equality — >1000 operators. A round trip through the writer's own inverse would pass for any self-consistent but wrong convention, so the test does not use one.

**Mutations caught** (each re-introduced, then reverted): operator sign flip (5 failures), dropped screw-axis translation (4), Cartesian coordinates written as fractional (1), symop loop omitted (4), space group misdeclared as P1 while writing another group's operators (4).

**Compatibility**: all eight existing `write_cif` call sites — seven in GFN eval, one in the COMPACK path — pass `mode` explicitly as `'unit cell'`, so none change behaviour. The P1 expansion is retained for exactly those callers.

**Defect found and fixed after the first version, 2026-08-23** — the writer emitted `SYM_OPS[sg_ind]` rather than the crystal's own `symmetry_operators`. It must emit what the *builder* applied: `crystal_building/utils.py:677` builds the cell from `mol_batch.symmetry_operators`, and the constructor stores the CIF's real operators, falling back to the standard table only when none are given (`crystal_ops.py:50-62`).

| operators vs `SYM_OPS[sg_ind]` | share of 400 random CSD entries |
|---|---|
| identical, same order | 62.5 % |
| same set, different order | 18.2 % |
| **different set** (alternate setting) | **19.2 %** |

*Table: how often a real CSD crystal's symmetry operators match the standard table, measured against `ccdc` 3.4.1, 2026-08-23. The different-set cases are settings such as `1/2-x,1/2+y,1/2-z` filed under IT number 14 — P2₁/n written as P2₁/c.*

For that 19.2 % the writer emitted operators that did not correspond to the coordinates, and the file still read back as a valid crystal. **Non-standard settings are not an edge case; they are more than a third of the corpus.**

**Why the test suite could not have caught it.** Every fixture was built as `MolCrystalData(sg_ind=...)` with no `symmetry_operators`, so all of them took the standard fallback — the standard path was the only path the suite could exercise. Twelve passing tests and a 5/5 mutation score, blind by construction of the fixtures. Coverage added: a hand-built P2₁/n-under-IT-14 case (licence-free), a refusal test for an operator count inconsistent with `sym_mult`, and a CCDC-gated test over 60 real CSD entries that **asserts its own sample contains non-standard settings** — otherwise it silently degrades back into the blind case. Reverting the writer to the standard table now fails 3 tests; before, it failed none.

**Still owed**, once the reader exists: the `read_cif(write_cif(x)) == x` assertion, which is what turns this from a writer test into the reader's test too.

### Original design notes

`MolCrystalData.write_cif` (`crystal_ops.py:1970`) delegates to `ase_write_cif` (`common/ase_interface.py:170`), which builds a bare ASE `Atoms` with a cell, calls `.write()`, then re-opens the file to string-replace `data_image0`. What it emits:

- **P1, always.** The `ase_crystal(..., spacegroup=...)` call exists at `ase_interface.py:154` but is on the `return_crystal` branch, which the writer path never takes. So all symmetry is expanded and **no space group is written at all** — matching the owner's "doesn't write space group info properly."
- **Lost:** `sg_ind`, symmetry operators, `z_prime`, every `aunit_*` parameter, `identifier`, `is_well_defined`, partial charges.
- Consumed by seven call sites across GFN eval and `crystal_analysis.py:1022` (the COMPACK path).

**This is the same problem as the reader, in the other direction, and it should be built as one thing.** A writer that emits space group + symmetry operators + the asymmetric unit rather than an expanded P1 cell is:

1. the natural output of the vocabulary in §1.5 — you can only write an asymmetric unit if you have a term for it;
2. **the reader's strongest test.** `read_cif(write_cif(crystal))` reproducing the crystal to numerical tolerance exercises tokenizer, symmetry, sites, components and Z′ in one assertion, needs no CCDC licence, and needs no fixture corpus;
3. the fix for a blocking defect already noted in §5 — the reader as originally specified could not read this package's own output.

**Design:**

```python
def write_cif(self, inds, path, mode='asymmetric unit'): ...
```

`mode='asymmetric unit'` becomes the new default: emit `_space_group_IT_number`, `_space_group_name_H-M_alt`, `_space_group_symop_operation_xyz`, `_cell_formula_units_Z`, and only the deposited set. `mode='unit cell'` and `'cluster'` keep today's P1 expansion for the visualisation and COMPACK paths that want it — **do not change what existing callers get by default without checking each one**; COMPACK in particular may depend on P1 expansion.

Drop the read-modify-write `data_image0` replacement by emitting the block header directly.

**Exit criterion:** `read_cif(write_cif(x)) == x` on cell parameters, space group number, operator set, site count, component count, Z′, and coordinates to 1e-5 — for a Z′=1, a Z′=2, and a Z′=0.5 fixture. Add a regression asserting the written file is **not** read as P1 by ASE (the trap in §2), since that is how the current output behaves.

---

## 9. What is unverified

- Whether `TemplateChemistry`'s 250/250 is circular (§7 Stage 2). **Blocks the Stage 2 claim.**
- Whether skeleton-isomorphic template collisions (tautomers, stereoisomers) survive the substructure match. The 1.8 % collision rate is measured; the split is not.
- The behaviour of the reader on non-CSD CIF sources (PDB-derived, computational output). Every refusal calibration here is CSD-conditional.
- Disorder handling is inert on *both* sides — CCDC reports `has_disorder == True` in 0/1000 files from this exporter — so parity is not an improvement, and neither reader is tested against a genuinely disordered structure.

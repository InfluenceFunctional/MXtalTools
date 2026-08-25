# Stage 1 review guide

**What to read, in what order, and what I would most like challenged.**
Nothing here is committed. Suite: 318 passed, 40 skipped, exit 0, ~24 s.

---

## 1. Changes to EXISTING files — 8 files, all small

Read these first; they are the only places existing behaviour changed.

| file | change | why |
|---|---|---|
| `mol_methods.py` | `scatter((z > 1).long(), …)` instead of `scatter(z > 1, …).long()` | a bool sum-scatter saturates at 1, so the heavy-atom count was `[1,1]` where truth was `[1,2]`. The batch branch of `radius_calculation` had therefore **never executed** |
| `crystal_ops.py` `_pad_tensor` | accepts a scalar; allocates on `val.device` | read `.dtype` off an `int` its own caller passes, so any crystal built with an int handedness failed. Default-device allocation would have thrown on CUDA |
| `crystal_ops.py` `box_analysis` | raises naming the missing field | was `TypeError: NoneType * Tensor`. Chose a raise over a default: `packing_coeff` is meaningless without `mol_volume`, and any default is a plausible-looking number |
| `crystal_building.py` `mol2ucell` | forwards `std_orientation` at Z′>1 | the Z′=1 branch and both `mol2cluster` branches forwarded it; this one silently substituted `True`. **Latent in production** (your generated `pos` is standard-oriented) but live for anything read from CIF |
| `crystal_opt_utils.py` | restored `sample_about_crystal` | deleted 2025-12-10, still imported. Made `mxtaltools.modeller` and `main.py` un-importable for 8.5 months |
| `pyproject.toml` | declared `spglib`; registered pytest markers | `spglib` was an undeclared import |
| `test_data_classes_basics.py` | 3 assertion fixes + a `device` fixture | asserted the **all-atom** centroid where the code zeroes the **heavy-atom** one — twice — and compared `(21,1)` against `(21,)`, broadcasting to a 21×21 comparison |
| `test_model_training.py` | removed module-scope invocation | ran a full training run at COLLECTION time |

**Worth knowing:** these are all outside CIF scope. You approved keeping them after I
over-reached; they are listed here so the diff is not a surprise.

---

## 2. NEW — the reader, 1446 lines

```
mxtaltools/dataset_utils/construction/cif/
    errors.py        named refusals; no bare raises
    tokenizer.py     CIF 1.1 -> blocks/loops. Lossless, chemistry-free
    symmetry.py      operator strings <-> 4x4; space group; centred settings
    sites.py         the deposited set
    topology.py      components via union-find over _geom_bond
    composition.py   moieties -- CAPTURED, UNUSED (see §4)
    reader.py        assembles Z' as an exact Fraction
    adapt.py         -> crystal_dict; builds the unit cell
mxtaltools/common/cif_io.py    the symmetry-bearing writer
```

**Measured against CCDC**, 300 random CSD files: space group **100 %**, symmetry
multiplicity **100 %**, Z′ **100 %**, sites 99.0 %, components 97.6 %. Built unit cell vs
`packing()`: **8.6e-15 Å**, bijective, elements checked. **9.0× faster** (1.63 vs 14.69
ms/file).

---

## 3. The judgement calls — what I would most like challenged

These are decisions, not measurements. Each could reasonably go the other way.

1. **No symop loop → refuse, never assume P1.** Inverts the ASE trap
   (`ase/io/cif.py:378` reads 87 % of your acridine tree as P1, silently). Cost: some
   bare-cell CIFs are rejected outright. *You agreed, but it is the most consequential
   default in the reader.*

2. **Defaults refuse.** `require_integer_z_prime=True`, `max_atomic_number=100`,
   `refuse_polymers=True`, `require_bonds=True`. Strict reads **81.2 %** of the corpus,
   permissive **100 %** — the gap is 17.8 % non-integer Z′ plus 1.0 % polymers. The reader
   *parses* everything; strictness is an explicit policy layer.

3. **The file's own operators always win** over `SYM_OPS[sg_ind]`. 37.5 % of the CSD
   differs. Substituting the table would describe a different crystal.

4. **The H-M symbol is recorded, not enforced.** `ZZZKEA02` says `C -1` under IT 2.
   Settings legitimately differ; the operators carry the truth.

5. **`Z'` is an exact `Fraction`**, not a float, so `1/3` never becomes `0.333`.

6. **`components.source == 'none'`** marks an unknown decomposition rather than a
   computed one, when `require_bonds=False`. An absence, never a guess.

---

## 4. Deliberate omissions and known gaps

- **`composition.py` is captured but unused.** Parses `_chemical_formula_moiety` into
  species/stoichiometry/charge for future cocrystal work (§1.6). You said you did not know
  what it does — **that is a fair reason to delete it**; the reconnaissance survives in the
  design doc either way.
- **Z′<1 is unsupported** — naive expansion over-generates (AJACOF: 256 vs 192) because
  operators map a component onto itself. The guard refuses loudly. You said ignore.
- **No chemistry.** No bond orders, no SMILES, no fingerprints, no partial charges. This is
  the next build.
- **`process_chunk` is untouched.** The native path is a correctness instrument, not a
  production switch.
- **`AGENTS.md` is still untracked.** `git add AGENTS.md` — until then §8's amendment
  cannot be written.

---

## 5. Where the design doc corrected itself

`cif_reader_design.md` carries five marked reversals, all from things measured after the
first draft. Worth reading the ⚠ marks specifically, since each one is a claim I got wrong
and then fixed:

| § | retracted | corrected to |
|---|---|---|
| 1.3.0 | "bond orders not inferable without H" | **geometry separates them at 2.6–2.9 sd**; `DetermineBondOrders` ignores coordinates, and CCDC's own `assign_bond_types()` is a geometry heuristic |
| 1.3.1 | "de-novo H placement not solved, 0.7 %" | conflated two problems; **with bond orders it is 0.256 Å** |
| S1 (lattice) | "every Z′>1 crystal scores two different geometries" | **latent** — your `pos` is standard-oriented, so the flag is a no-op |
| S8 (lattice) | "`reset_sg_info` silently substitutes operators" | **unreachable** — the guard is `hasattr` on a property that always exists |
| writer §8 | "Z′>1 one-way, 0/5 survive" | **refuted** — 5/5 round-trip; the probe measured re-ingestion filtering |

Three of those were caught by you noticing a number that did not match your experience.
The pattern worth watching for: **a method's failure reported as a fact about the problem.**

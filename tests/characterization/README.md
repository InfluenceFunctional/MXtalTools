# Stage-0 characterization suite: today's crystal building and analysis, pinned

Stage 0 of the change plan in `gfn_diffusion/energy_sampling/docs/design/crystal_units_audit.md`
(section 5). It changes no MXtalTools code. It pins what the code does today, bit for bit, so that
every later stage (unit map, batch-helper levels, element layout, per-unit reach, ...) changes
behaviour only where it means to, and visibly. It also states the known wrong behaviour as strict
expected failures that later stages must flip.

Scope: MXtalTools only. GFN code is not pinned or tested here (owner, 2026-10-22: GFN is fixed
separately from these outputs), so the audit's F2 case (acridine prior rows through GFN
`instantiate_crystals`) and the "GFN route on F3" xfail are omitted.

## Run

```
cd MXtalTools
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider tests/characterization/
```

CPU only, about 20 s. The tests read only `tests/characterization/data/`, never D:, and write no
files. Every test module runs under `charlib.deterministic()`: one thread,
`torch.use_deterministic_algorithms(True)`, float32 default dtype, restored afterwards.

## Files

| file | what it is |
|---|---|
| `charlib.py` | shared code: fixture -> crystal builders, the pin groups (`compute_group`), bit-exact encode / compare, the MLIP stubs, the float64 brute-force lattice sum, the geometric element truth, the refusal rule |
| `build_fixtures.py` | generator of `data/fixtures.pt` from D: (run once; `--check` rebuilds in memory and compares) |
| `build_goldens.py` | generator of `data/golden_<case>.pt` and `data/goldens_manifest.json` (`--check` recomputes and lists moved pins) |
| `test_characterization_pins.py` | every golden pin, one test per (case, pin group) |
| `test_characterization_oracle.py` | builder pair sets and eLJ against the brute-force lattice sum; the build against the stored crystal; the radial graph's boundary convention |
| `test_characterization_known.py` | strict xfails for known wrong behaviour, and their passing controls |
| `data/fixtures_manifest.json`, `data/goldens_manifest.json` | provenance; what each group and each golden entry is |

## Fixtures (`data/fixtures.pt`, plain tensors and strings)

- **F1** — 4 Z'=1 rows of `D:/crystal_datasets/csd_fd_v5` (status canonical), molecules without
  point symmetry (checked: no graph automorphism, nor the mirror, is realised by a rigid motion
  within 0.1 A): PYRAZB21 phenacetin (sg 14), YIRLEU (sg 2), OYIQIX (sg 9), HIFQUJ (sg 14). Rebuilt
  exactly as `crystal_fd.dataset.to_mol_crystal_data(mode='body')` builds them (asserted at build).
- **F3** — 2 Z'=2 rows of the r2 ingest store (status ok, standard setting, one compound),
  rebuilt through `MolCrystalData.from_arrays` with each row's own operators, float32. The two
  units of each row are different conformers. Measured three ways: the largest difference in
  interatomic distances between the units; the Kabsch RMSD after the best of every graph
  automorphism of the molecule, for a proper rotation; and the same for an improper one.
  - AHUMIZ (sg 2, 23 atoms per unit, handedness [1, -1]): distances differ by up to 2.47 A. RMSD
    1.22 A proper, 2.06 A improper, the best over 144 automorphisms.
  - AFEVOA (sg 14, 12 atoms per unit, handedness [-1, 1]): distances differ by up to 1.10 A. RMSD
    1.56 A proper, 0.40 A improper, over 2 automorphisms. So the two units are close to mirror
    images of each other, which makes F3 a per-unit handedness case too. Both units' `unit_smiles`
    carry the same `[C@@]`.
  - AFEVOA's units are stored out of canonical order, so canonicalisation moves atom blocks.
- **F3b** — today's identical-copy Z'=2 layout: `zp_doubling.double_zp1_crystal` of the two sg 14
  F1 rows (the molecule duplicated through the list constructor, as the crystal search does).
- **F4a / F4b** — a max_z_prime=2 batch of phenacetin (Z'=1) and AFEVOA (Z'=2, list constructor);
  the Z'=1 row's pad slot written by collate (F4a: 0, 0, 0) or by the constructor (F4b: 0, 1, 1).
- **AUDIT** — the hand-posed crystals of the audit probes (phenacetin P-1 and P2₁/c, pyridone
  P2₁/c, the probe1b phenacetin, the pyridone Z'=2 batch-mate, ethanol + methane, ethanol +
  dimethyl ether, pyridone + fluorobenzene), plus ethanol + propane (two_eth_meth's cell and slots,
  added 2026-10-22: every unit has >= 3 non-collinear heavy atoms, so its frame is defined, unlike
  methane's). Molecules are re-embedded by replaying each probe's own RDKit embedding sequence in a
  fresh process (from_smiles embeds unseeded, so a geometry depends on every embedding before it in
  the process); this reproduces the audit's pair counts exactly (10021/10022, 2301/2305,
  4993/4995, 654/655). The ethanol / methane / ether / propane molecules are RDKit
  `EmbedMolecule(randomSeed=3)` + MMFF.

## What is pinned (`test_characterization_pins.py`)

Cases: F1 and F3b at `std_orientation` True and False, F3 / F4a / F4b at True only (their atoms are
posed); cutoffs 6 and 10. Each case's group list is in `data/goldens_manifest.json` (`cases`), and
what each group pins in `group_descriptions`.

| group | cases | pinned |
|---|---|---|
| `box` | all | the constructed batch: per-row sums (radius, mass, mol_volume, num_atoms), symmetry, cell, `box_analysis` (T_fc, T_cf, cell_volume, packing_coeff, density), pose slots and atoms; for F3 (`from_arrays` runs no `box_analysis`) also `box_analysis` on a clone, where the #6 sum convention shows |
| `canonicalize` | all | `canonicalize_zp_aunits()`: slot order and the atom blocks / per-slot values moved with it |
| `latent` | main | `latent_params(gauge_fix_free_axes=True and False)` and `latent_to_cell_params` back |
| `ucell/std*` | all | `mol2ucell`: posed pos, unit_cell_pos / batch / mol_ind |
| `cluster/std*/cut*` | main | `mol2cluster` (pos, z, x, aux_ind, mol_ind, ptr, batch) and `construct_radial_graph`: intra and inter edge sets sorted, counts per crystal, inter distances |
| `analyze/std*/cut*` | all | `analyze(['reduction_en', 'elj'])` |
| `rdf/std*/cut*/<mode>` | main | `analyze(['rdf'])` for rdf_mode None, elementwise, envwise, atomwise: rdf, bins, channel labels |
| `mlip/std*` | main | MLIP input assembly without a model (below) |
| `reduction` | F1, F3 | `analyze(['reduction_en'])` on every row under 8 cell variants (own cell; a↔c and a↔b swapped; α or β 80°; angles 100/110/120° and 80/85/75°; lengths stretched) × the row's own space group and one of each crystal system (1, 2, 5, 14, 19, 92, 146, 169, 198). Every fixture cell is a reduced CSD cell, so the analyze pins only see the penalty's zero; these pin its walls |
| `helpers` | F1, F3, F3b, F4a | the batch helpers on an ordinary crystal batch (#3; stage 3 rewrites them): `subsample_new_batch([1, 0])` and `([0])`, `append_batch(clone)`, `batch_to_list` and its re-collation, every key with its `_slice_dict` / `_inc_dict` entries (aunit_* slots, T_fc, symmetry_operators, identifier, unit_smiles, unit_*, mol_ind, z_prime, max_z_prime, ...) |
| `zp_doubling` | F1, F3b | `double_zp1_crystal` of the two sg 14 F1 rows (params and doubled crystal); `_unit_cell_frac` of each F3b row (#15 site) |
| `stdcell` | F1, F3, F3b | `compute_standard_cell()` (spglib; its labels tile z, a #15 site; #14): cell, pose, pos, unit cell, operators, or the exception (F3b: ValueError, det P = 2) |
| `reparam` | F1, F3, F3b | `mol2ucell(std_orientation=True)` then `reparameterize_unit_cell()` (#18): its five outputs |
| `ingest` | F3 | the ingestion chain's structural steps (#9, #10): the F3 rows given as bare deposited sets (atoms wrapped into the cell, units interleaved atom by atom, unit 1 reversed; `charlib.f3_deposited_batch`) through `find_molecules` -> `parameterize_pose` -> `check_rebuild` (csd_v1 recipe arguments, bonds by distance): every key after each step |
| `exp6/std*/cut*` | F1, F3, F3b | `analyze(['exp6'])` (#25 typing; packaged 'fit' parameters, MMFF charges) |
| `split` | two-compound | `split_to_zp1_batch()`: the per-unit graphs the builder makes (#1, #11) |
| `elements/std*` | two-compound | per MLIP input builder and the ase unit-cell assembly: elements and positions per crystal, and the count of rows whose element is wrong against `charlib.element_truth` (#15) |

The **two-compound cases** (`two_eth_meth`, `two_eth_prop`, `two_eth_ether`, `two_pyr_flb`; groups
box, canonicalize, split, ucell, elements, analyze at std True and False) pin today's WRONG outputs:
a strict xfail only notices a flip to the right behaviour, so a change from one wrong value to
another (a [7, 7] split becoming [8, 6], 4 wrong elements becoming 6) would otherwise pass unseen.
Stages 2, 4 and 7 re-pin them deliberately when they change them on purpose.

**MLIP pins** (`mlip/std*`): `batch_to_fairchem_batch` (built unit cell, and force_rebuild), the
fairchem list path collated, `batch_to_mace_input_dict` and the collated `batch_to_mace_atomicdata`
(stub with r_max 6, elements 1-100); the ase 'unit cell' assembly (positions, numbers, cell); and
four routes through `analyze` with stub predictors: `['uma']`, `['mace']`, and the MLIP-on-cluster
branch `['elj', 'uma']` / `['elj', 'mace']` at cutoff 6 (`_resolve_aunit_z` /
`_mace_resolve_aunit_z` reading the aux_ind == 0 atoms, as calibrate_qm9_energy.py and the paper
evaluation use it). For each route: what `analyze` returns, the predictor energies it stores
(`*_pot` crystal leg, `*_gas_pot` gas leg), and every input the stub was handed, gas leg then crystal
leg, recorded even when the route raises. The UMA fields include `dataset` (the task name, which
selects UMA's output head) and `sid`.

The stubs return `charlib.stub_energy` of their input, a deterministic non-zero energy per graph
(sum over its atoms of 1e-3 (x + 2y + 3z) + 1e-2 Z, plus 1e-3 times its cell entries, plus 1e-4
times its edge count). So the lattice-energy arithmetic downstream of the model (#26: `E_cell /
(sym_mult z') - mean_k E_gas,k`, the per-unit gas leg through `split_to_zp1_batch`) is pinned
through the result. The UMA stub carries `inference_settings.external_graph_gen = True` and a
backbone cutoff of 6, as the real crystal predictor does by default (USE_UMA_EXTERNAL_GRAPH). So
the route pins the external neighbour list `attach_external_graph` hands UMA. Its source/target
convention against fairchem is tested separately by `tests/test_uma_external_graph.py`.

Exceptions are pinned too (by type): e.g. `analyze(['uma'])` on the F3 `from_arrays` batch raises
TypeError today (the UMA cell guard reads `packing_coeff`, which `from_arrays` does not set); the
gas-leg call made before the raise is pinned.

Small tensors are stored in full; tensors above 16 kB by the sha256 of their canonical bytes (NaN
given one bit pattern, -0.0 made +0.0, so equal bytes is `torch.equal` with NaN == NaN), with
shape, dtype, sum and the first values kept to make a mismatch readable. Lists of tensors or arrays
keep each element's type, dtype and shape.

**Environment.** Each golden file records the environment it was written under:
- `__environment__`: torch, numpy, torch_geometric, torch_scatter, torch_cluster, and every `MXT_*`
  variable except two inert ones.
- `__group_environments__`: the packages a group kind depends on beyond those. These are fairchem,
  mace and ase for `mlip` and `elements`; rdkit for `exp6`; spglib for `stdcell`; networkx, scipy
  and ase for `ingest`.

A pin test skips, naming the difference, when its own file's record differs from the current
environment, because a bit-for-bit pin does not transfer between environments. A digest of the
mxtaltools sources at generation is recorded in the manifest for information only.

### Not pinned here

- The ingest store's save / load round trip. It is covered by `tests/test_crystal_store.py`, which
  writes to pytest's tmp_path; this suite writes no files at test time.
- GFN (above).

## The brute-force oracle (`test_characterization_oracle.py`)

`charlib.oracle_pairs` takes an asymmetric unit and the crystal's own operators and cell. It
generates every symmetry image and every lattice translation that can bring an atom within the
cutoff (|Δf_i| ≤ cutoff·|row i of T_cf|), and lists every intermolecular pair in float64. It shares
no code with the cluster builder.

A pair counts when d < cutoff, strictly, the radial graph's convention:
`test_radial_graph_drops_points_at_exactly_r` pins that `asymmetric_radius_graph` /
`torch_cluster.radius` drop points at exactly r = 5, 6 and 10 in float32 and keep the next float
inside. A torch_cluster change that moves the boundary therefore shows there, by name. Pairs within
2e-4 A of the cutoff are boundary pairs, left out of both sides.

Two references:
- **The builder's own posed asymmetric unit** (`mol2ucell`). This tests that the cluster is
  COMPLETE. Per crystal it checks:
  - the pair set: per atom the count, the sorted distances within 1e-3 A, and the partner elements;
  - the eLJ re-summed in float64 over the builder's own pairs, within 2e-6 of Σ|e| (complete cases
    measured ≤ 6.4e-7);
  - the float32 `analyze` eLJ, within 2e-4 of Σ|e|. This is a gross-error check that passes even
    where a few far pairs are missing.
  A control checks the oracle against a cluster widened by 4 A.
- **The stored crystal** for the ingestion rows (F3, and AFEVOA as F4a's Z'=2 row). This tests that
  the build REPRODUCES the crystal the ingestion chain read:
  - each posed unit must be a symmetry / lattice image of its stored unit (split by the stored
    `mol_ind`), atom for atom in stored order, within 1e-4 A;
  - the builder's pair set must equal the oracle's built from the stored atoms.
  Without this, a later deliberate re-pin could lock in a wrong pose.

## Strict expected failures

Each states the right behaviour, fails today for the reason stated, and must flip at the stage
named. When it does, pytest reports XPASS(strict) as a failure: remove the marker in that change.

How a known failure is recognised:
- Every strict xfail expects `pytest.fail` and nothing else (`raises=pytest.fail.Exception`).
- The test calls `pytest.fail` when its own check finds today's wrong value.
- When the library stops at the known place, the test names both the exception type and the
  innermost function (`_known_crash`; e.g. AssertionError in `split_to_zp1_batch`, RuntimeError in
  torch's `einsum`).
- Any other exception, from the library or the harness, propagates and is reported as a failure. A
  regression cannot hide behind an xfail.

"Refused" means a deliberate refusal (`charlib.deliberate_refusal`): a ValueError, RuntimeError or
NotImplementedError (or a library subclass) raised by a `raise` statement in mxtaltools. A torch
shape error (RuntimeError from einsum / cat / view), an IndexError or a bare `assert` (which
`python -O` removes) is not a refusal, wherever it surfaces. `test_deliberate_refusal_rule` checks
the rule on one case of each.

| test | audit item | today | flips at |
|---|---|---|---|
| `test_two_compounds_do_not_trade_slots` | #19 | ethanol and dimethyl ether trade slots and atom blocks under `canonicalize_zp_aunits` | 7 |

The five rows scheduled for stage 3 (eight cases) flipped at stage 3 and left this table (below), the two oracle
rows scheduled for stage 5 (17 cases) flipped at stage 5, and the two #15 rows (21 cases) flipped at stage 4.

**Flipped at stage 4** (2026-10-23; one unit-cell element layout, `crystal_building/utils.py::unit_cell_gather_index`
and `unit_cell_z`; markers removed, each now an ordinary test that must keep passing). Exactly the 21 stage-4 xfails
flipped; no other xfail did:
- `test_mlip_elements_right_for_two_compounds` [2 crystals x 2 std x 5 builders] (#15): 4 of 36 / 8 of 96 unit-cell
  rows got the wrong element (fairchem vectorised and list, MACE dict and list, ase 'unit cell'): the unit cell is
  laid out [unit][image][atom] and the builders tiled the whole asymmetric unit image by image. Every reader now
  gathers elements in the unit cell's own layout.
- `test_zp_doubling_unit_cell_elements_right` [two_pyr_flb] (#15): `_unit_cell_frac` tiled unit 0's z over the whole
  unit cell.

**Flipped at stage 5** (2026-10-23; the per-unit all-atom cluster reach, `crystal_building/utils.py::molecule_reach`
and `MolCrystalBuilding._unit_reach_buffer`; `EXPECTED_MISSING_PAIRS` and `EXPECTED_ELJ64_OFF` emptied, so each case
is an ordinary test that must keep passing). Exactly the 17 stage-5 xfails flipped; no other xfail did:
- oracle `test_pair_set_matches_brute_force` [9 cases] (#16): pairs were missing (HIFQUJ, the real F1 row 3, 5 of
  10852 at 10 A, std both; phenacetin P-1 1 of 10022 at 10 A; phenacetin P2₁/c 4 of 2305 / 2331 at 6 A, std
  True / False; pyridone P2₁/c 2 at 10 A, std both, and 1 at 6 A, std False; probe1b phenacetin the 5.81 A H...H
  pair at 6 A): the reach test compared a heavy-atom radius about the heavy-atom centroid with all-atom centroids,
  and hydrogens reach past it. Every pair set now equals brute force.
- oracle `test_elj_float64_resum_matches_brute_force` [8 cases] (#16): the same cases except phenacetin P-1, whose
  one missing pair was worth 1e-8 of Σ\|e\|.

**Flipped at stage 3** (2026-10-23; the batch helpers' unit and edge levels; markers removed, each now an ordinary test
that must keep passing). Exactly the eight stage-3 xfails flipped; no other xfail did:
- `test_list_constructor_offsets_edge_index` (#7): two 6-atom molecules' `edge_index` were concatenated on dim 0 without
  an offset ([4, 5], both from index 0); the list constructor now joins every 'index' / 'face' key along its last axis,
  molecule k offset by the atoms before it ([2, 10]).
- `test_subsample_slices_a_per_unit_value` [F3-zp2-2, F4a-ragged-row1, F4a-ragged-row0] (#3): a per-unit value with a
  unit-pointer slice entry was carried whole by `subsample_new_batch` (4 of 4 values on F3, 3 of 3 on the ragged Z'
  [1, 2] F4a batch); it is now the `UNIT` level, sliced by the unit pointer of `z_prime`.
- `test_append_concatenates_a_per_unit_value` (#3): `append_batch` kept two equal per-unit values once (4 values for 8
  units) and refused different ones as a differing shared field; it now concatenates them per unit.
- `test_subsample_slices_an_index_tensor` [2 graphs, 3 graphs] (#3): a [2, E] index tensor was gathered by row on a
  2-graph batch ([2, 12] -> [1, 12]) and carried whole and un-offset on a 3-graph batch; an 'index' key is now the
  `EDGE` level: its rows' columns gathered and re-offset.
- `test_append_offsets_edge_index` (#3): the second batch's `edge_index` was not offset (on these 2-graph batches the
  [2, E] tensors were concatenated as rows, [4, 12]); it is now joined along its last axis, offset by the first
  batch's atom count.

Also flipped at stage 3, outside this suite: `tests/test_unit_map.py::test_collate_gives_a_map_less_row_its_derived_map`
(the stage-2 review blocker below), and `tests/test_zp_doubling.py::test_output_schema_matches_search_outputs` passes
again.

**Flipped at stage 1** (2026-10-22; markers removed, each now an ordinary test that must keep passing):
- `test_unequal_units_are_refused_or_scored_right` [eth_pyr21-std1] (#1): the 21-atom ethanol + 2-pyridone row
  stopped at the bare assert in `split_to_zp1_batch`; it now raises ValueError naming the row.
- `test_collate_refuses_to_truncate_a_zp2_row` (#4): `collate_data_list(max_z_prime=1)` truncated a Z'=2 row's
  slots silently; it now raises ValueError.
- `test_collate_leaves_its_inputs_unchanged` (section 7): collate widened the caller's Z'=1 row in place
  (aunit_centroid (1, 3) -> (1, 6), max_z_prime 1 -> 2); it now pads a shallow copy.
- `test_all_zp1_batch_at_width2_is_scored`, `test_all_zp1_batch_at_width2_equals_width1` (#12, #13): an all-Z'=1
  batch at max_z_prime 2 stopped at the bare assert in `join_zp1_cluster_batch`; the join now returns the split's
  clusters, and the scores equal the width-1 batch's bit for bit.
- `test_zp2_row_in_a_width1_batch_is_refused` [mol2ucell, analyze] (#12): a RuntimeError in torch `einsum`; now
  a ValueError from `mol2ucell` / `mol2cluster` naming the row.
- `test_zp1_row_cluster_in_mixed_batch_same_as_alone` [pyridone, F4a, F4b] (#13): pad slots widened the Z'=1
  row's cluster (phenacetin 9152 atoms, 6110 with aux_ind < 2, against 5928 (4056) alone; pyridone 3264 (2184)
  against 2256 (1368)); `zp_buffer` now reads each row's own slots only.
- `test_zp1_row_pairs_and_elj_in_mixed_batch_equal_alone` [pyridone] (#13, #16): 4995 pairs in the mixed batch
  against 4993 alone; now 4993 in both. 4995 is the true count: the row still misses 2 pairs to #16 (the oracle
  xfails `pyr_p21c-row0-*-cut10`), which stage 5 fixes.

**Flipped at stage 2** (2026-10-22; the atom -> unit map `unit_ind`; markers removed, each now an ordinary test that
must keep passing). Exactly the eleven stage-2 xfails flipped; no other xfail did:
- `test_unequal_units_are_refused_or_scored_right` [eth_meth-std0, eth_prop-std1, eth_prop-std0] and
  `test_unequal_units_split_by_their_atoms_or_refused` [eth_meth, eth_prop] (#1): ethanol + methane was split 7 + 7 and
  scored (eLJ -68.71 against -58.90, std False), ethanol + propane 10 + 10 (-124.86 vs -130.64 std True, -159.18 vs
  -163.42 std False). The list constructor now writes the map (9 + 5, 9 + 11) and `split_to_zp1_batch` refuses units
  of different sizes (ValueError naming the row) until stage 7.
- `test_one_molecule_row_relabelled_zp2_is_refused` [pyridone, phenacetin] (#2): a one-molecule row relabelled Z'=2 on
  the batch was cut into two half-molecule units (pyridone scored 412.4; phenacetin an IndexError). The constructor
  writes one unit for one molecule; the split refuses the row (unit 1 holds no atom). **Only rows that carry the map
  are protected.** A row without one (GFN's blank batch, filled by `set_mol_attrs`' batch path; `from_arrays` rows
  given no map; files written before stage 2) is read through the absence path, so relabelled Z'=2 it is still cut
  into two half-molecule units and scored (stage-2 review, 2026-10-23: GFN's `init_blank_crystal_batch` on a 6-atom
  molecule scored eLJ 75.40 with units of [3, 3] atoms, as before stage 2). Whether `set_mol_attrs`' batch path
  writes one unit per molecule graph, or GFN writes the map itself, is an owner decision; until then
  `tests/test_unit_map.py::test_blank_batch_route_relabelled_zp2_is_refused` is a strict xfail.
- `test_one_molecule_given_zp2_at_construction_is_refused` [pyridone, phenacetin] (#2): now refused by the constructor
  (`crystal_building.utils.refuse_unit_map_disagreeing_with_z_prime`).
- `test_list_constructor_sums_or_refuses_collated_rows` (#7): [1]-shaped `num_atoms` / `radius` / `mass` /
  `mol_volume` were concatenated (num_atoms [26, 26], packing halved); they are summed and stored 0-dim, `MolData`'s
  shape (until the stage-2 review the sum took molecule 0's shape, so such a row did not collate beside a default
  row), and a value that is not one per molecule is refused.
- `test_int_z_prime_row_collates_beside_a_default_row` (#7): an int z_prime is stored [1], the default's shape.

The stage-2 refusals, the map's writers and carriers, the derived unit level and the bit-identity of the absence path
are tested in `tests/test_unit_map.py` (every refusal also under `python -O` in a child interpreter).

**Gap left by stage 2, closed at stage 3 (2026-10-23).** `collate_data_list` (`mxtaltools/dataset_utils/utils.py`) of
rows with and without a map raised KeyError when the first row carried one, and dropped the map of the whole batch when
it did not (PyG takes the keys from the first row), so `tests/test_zp_doubling.py::test_output_schema_matches_search_
outputs` (doubled rows collated with native acridine Z'=2 rows) failed. Stage 3 applied the reviewed patch (gfn_diffusion
`energy_sampling/artifacts/fd_design_oct15/units_build_oct22/stage2/review_fix/make_collate_patch_tree.py`): each
map-less row gets `crystal_building.utils.derived_unit_ind`, the layout every reader gives it, on the shallow copy, in
any row order; the caller's rows are not modified. `tests/test_unit_map.py::test_collate_gives_a_map_less_row_its_
derived_map` passes, and `tests/test_batch_unit_levels.py` covers both orders.

The refusals themselves, and that they survive `python -O`, are tested in `tests/test_zprime_refusals.py` (a
child interpreter runs every one with `-O`), with the per-row `zp_buffer` (Z'=1 rows, and Z'>1 rows narrower than
their batch: synthetic, and the F3 rows beside F1 phenacetin at width 3), the all-Z'=1 identity join, the all-Z'=1
width-2 unit cell built directly and through each MLIP input builder's own rebuild branch, a
`std_orientation=False` build without `aunit_handedness`, and collate's no-mutation property.

Controls that pass today and must keep passing:
- the refusal rule (since stage 1 on the module function `aunit2ucell` called directly on an unsplit Z'=2 batch,
  torch's `einsum` RuntimeError, and `collate_data_list([])`, a bare assert: the stage-0 examples, and then the
  `build_unit_cell` method on that batch, became refusals);
- MLIP element assignment for one compound (F1, F3, F3b, F4a, five builders);
- `zp_doubling` elements on F3b;
- the unit-major `[unit][image][atom]` unit-cell layout at Z'=2, checked geometrically
  (`tests/test_unit_cell_layout.py` covers Z'=1);
- `batch_to_list` splitting a per-unit value, F3 and ragged F4a;
- the F4a / F4b pair-distance multisets and eLJ (within 8 float32 ulps; the bits are in the pins);
- the ingestion steps re-deriving F3's stored units, pose and handedness;
- the 43 of 52 oracle cases with no missing pair;
- the stored-crystal checks;
- the radial-graph boundary.

## The batch helpers' level contract (stage 3, 2026-10-23)

One graph per crystal; a per-atom unit counter (`unit_ind`); per-unit values as a third level. The batch helpers
(`subsample_new_batch`, `append_batch`, `rebuild_simple_slice_inc_`, `batch_to_list`, `collate_data_list`,
`MolCrystalIO.select_rows`, the ingest store's level map, `set_mol_attrs`' batch path, and `canonicalize_zp_aunits`
for per-unit values) give every stored value except `ptr` and `batch` exactly one level, and the helpers agree on it
(since the stage-3 review, 2026-10-23: `subsample_new_batch`, `append_batch` and `rebuild_simple_slice_inc_` took a
ragged value whose total equalled the atom or row count at that level, where `select_rows` and `batch_to_list` read
its entry). The authority is the code:
`MXtalBase`'s Notes and `MXtalBase._level_plan` in `mxtaltools/dataset_utils/data_classes.py`; pinned by
`tests/test_batch_unit_levels.py` (and `tests/test_batch_levels.py` for the atom and row levels).

| level | what it holds | sliced by | `subsample` / `select_rows` | `append_batch` | bookkeeping written |
|---|---|---|---|---|---|
| `atom` | one entry per atom, dim 0 (`pos`, `z`, `x`, `unit_ind`) | `ptr` | atoms of the selected rows | concatenated | entry `ptr`, increments 0 |
| `unit` | one entry per unit, dim 0, ragged: `sum(z_prime)` entries | the unit pointer `[0, z'_0, z'_0 + z'_1, ...]` of `z_prime` | units of the selected rows | concatenated | entry the unit pointer, increments 0 |
| `edge` | per row a segment of its own length (bonds, angles, torsions, ...) along the collation axis: the last axis for an 'index' / 'face' key, dim 0 otherwise | the per-row counts its slice entry records | segments of the selected rows; an 'index' / 'face' key's atom indices re-offset to the new atom offsets | concatenated, the other batch's index values offset by the first's atom count | entry the joined counts; increments the atom offsets for an index key (as collation writes them), else 0 |
| `row` | one entry per row (crystal), incl. lists and the padded per-unit slots `[B, max_z_prime, ...]` | `arange(B + 1)` | rows gathered | concatenated | entry `arange`, increments 0 |
| `batch` | constants: python values, dicts, 0-dim tensors, tables set on the batch | not sliced | carried whole | kept (validated equal) | none |

A value gets its level from the first rule that applies:
1. **the registry** `data_classes.KEY_LEVELS` (`declare_level(key, level)`, process-wide): the level is fixed by name, and
   a value that does not fit it is refused. Registered: `unit_ind` (atom), `max_z_prime` and the space-group lookup
   tables `asym_unit_lut`, `sym_mult_lut`, `free_axis_lut`, `asym_unit_dict` (batch: a 231-row batch had them gathered
   as rows), `build_conformer_tree`'s per-angle / per-torsion / per-closure values (edge), and the unit cell
   `unit_cell_pos`, `unit_cell_batch`, `unit_cell_mol_ind` (`unit_cell`: a level the helpers do not slice; every helper
   that moves values refuses it; rebuild the unit cell on the result);
2. **collation's naming contract**: an 'index' / 'face' tensor key (concatenated on its last axis and offset by the atom
   counts by `MXtalBase.__cat_dim__` / `__inc__`) is `edge`, whatever its length. Its per-row counts are its slice
   entry's, and then its increments must be the rows' atom offsets `ptr[:-1]` (what collation writes; the helpers
   re-offset its values as atom indices), else it is refused, as `select_rows` refuses it. Without a usable entry the
   counts are read from the values of a `[k, E]` tensor with `k >= 2` (each column belongs to the row of the atoms it
   indexes, columns grouped by row); a one-dimensional or `[1, E]` index value without an entry is refused, not read
   as atom indices. A key holding both 'index' and 'batch' is refused;
3. **the slice entry**, for every other tensor: an entry that describes it (`B + 1` entries from 0, non-decreasing,
   ending at its length) gives the level it records, whatever the value's length coincides with: `atom` when it is
   `ptr` and the length the atom count (tested first), `row` when it is `arange(B + 1)` and the length the row count,
   else `unit` when it is the unit pointer of `z_prime` and `edge` otherwise (`MXtalBase._entry_level`). A host-held
   entry is compared once per entry object, which needs no device sync; an entry held on a device (`add_graph_attr` /
   `add_node_attr` write them on the batch's device) is not read where the length is the atom or row count, so the
   hot path pays no host sync (`tests/test_batch_levels.py::test_unambiguous_batches_never_read_slice_entries` pins
   that `_recorded_level`, the device-entry reader, is not called on an ordinary batch; `tests/test_batch_unit_levels.
   py::test_a_device_resident_entry_is_not_read_where_the_length_decides` pins the device case);
4. **the length**, for a tensor without a usable entry: the atom count is `atom`, the row count `row`;
5. everything else is `batch`.

**How a new attribute declares its level.** Collate it from rows (collation writes the slice entry: a per-unit value
is `[z_prime, ...]` per row; a per-edge value `[E_row, ...]`), or attach it on a batch with `add_node_attr`,
`add_graph_attr` or `add_unit_attr` (a ragged value attached on a batch needs a host-held entry that describes it, as
collation and `add_unit_attr` write); name an atom-index tensor with 'index' (or 'face'), with increments `ptr[:-1]`
when its entry is set by hand, and give a value that is not an atom index no such name; register a key whose
bookkeeping cannot carry its level with `declare_level`. A value without a usable entry and without a registration is
read by its length: `batch` and carried whole when it is neither the atom nor the row count, and read per atom or per
row when its total coincides with one (exact only where every row's segments are its atoms or its one entry, e.g.
`z_prime` 1 on every row; give it an entry or register it otherwise). Refusals are `data_classes.BatchLevelError`
(a ValueError; `select_rows` raises ValueError), raised by `raise` statements and so kept under `python -O`.

**Accepted gap, pending the owner's decision (stage-3 review, 2026-10-23).** The stage-3 plan says nothing is silently
carried whole, but rule 5 carries whole a tensor of another length than the atom or row count that has no usable
entry and no registration (a table set on the batch, a per-unit value set without `add_unit_attr`, a value whose entry
went stale): `subsample_new_batch` and `append_batch` keep it as it is, while `batch_to_list` refuses a tensor whose
entry is stale. Refusing such values would re-pin `tests/test_batch_levels.py::test_values_without_a_usable_entry_
keep_the_length_rule` (lut [231] and a stale-entry `radius` carried whole) and `tests/test_batch_unit_levels.py::
test_add_unit_attr_and_declare_level` (`unit_free` carried whole). Until the owner decides, the asymmetry stands.

**Collation by a DataLoader.** `collate_data_list` and the loaders MXtalTools builds (`dataset_utils.utils.
MXtalDataLoader`: `get_dataloaders`, `quick_combine_dataloaders`, `update_dataloader_batch_size`) give a row without
the unit map its derived equal-block map when other rows carry one (`fill_unit_maps`). A plain `torch_geometric`
`DataLoader` / `Batch.from_data_list` does not: over rows with and without the map it raises KeyError (map first) or
drops the map (map-less first). Rows of mixed vintage go through one of the former.

## Which pins each stage is expected to move

A stage may move only the pin groups listed for it; any other moved pin is a regression to fix,
not to re-pin. "None" means the stage's proof is that these pins stay bit-identical.

| stage | pin groups expected to move | why |
|---|---|---|
| 1 refuse instead of guessing | F4a, F4b: `cluster`, `analyze`, `rdf`; `mlip` cluster routes; `ucell` too if the build stops sending the Z'=1 row through split / join | `zp_buffer` ignores pads, so every row with z_prime < max_z_prime builds the cluster it builds alone: Z'=1 rows, and Z'>1 rows narrower than their batch (no pinned case holds one; regeneration log); the build dispatches on each row's z_prime |
| 2 atom -> unit map | `helpers` of every case and `ingest` (new key); the groups of `two_eth_meth` and `two_eth_prop` that split or build (unequal units refused: equal blocks stay asserted at stage 2) | a new stored key; unequal units refused |
| 3 batch-helper levels | none (as planned). **Amendment, awaiting the owner's confirmation:** F3 `helpers` and `ingest` moved at stage 3 by a new key only (the unit map `unit_ind` that `from_arrays` and `find_molecules` now write; stage 2's writers, carried into stage 3 by its spec); records only added, each equal to the `mol_ind` record beside it, none removed or changed (change log, 2026-10-23). The stage-3 review fixes moved no pin | proof: F1, F3b, F4a `helpers` bit-identical; every record F3 held before reproduces |
| 4 one element layout | `elements` of `two_eth_ether` and `two_pyr_flb` (wrong rows -> 0). **Done 2026-10-23: exactly these moved (40 records; regeneration log)** | one compound: `mlip`, `zp_doubling`, `stdcell` bit-identical |
| 5 per-unit reach | `cluster`, `analyze`, `rdf`, `exp6` of every case, `mlip` cluster routes, the two-compound `analyze`. The Z'>1 clusters shrink (no summed radius); missing pairs appear. **Done 2026-10-23: exactly these moved (228 records; regeneration log)** | cell selection changes for every row |
| 6 GFN for one compound | none here (GFN is not pinned) | |
| 7 different compounds | every group of the two-compound cases | per-compound units, labels and order |

Before re-pinning cluster or energy groups (stage 5 above all), the oracle tests must pass for the
cases being re-pinned: a complete case's new eLJ is checked against the brute-force sum, not
accepted because it moved. Report the per-crystal energy differences in the change.

## Known faults no stage of the plan schedules

These are pinned as today's behaviour or not covered. The owner decides whether a stage takes them
on.
- **#14**: `compute_standard_cell` on an unsplit Z'>1 graph is silently wrong for `pos` (pinned in
  `stdcell` for F3). `get_fd_params` crashes at Z'>1, and the slot-0 helpers
  (`zp1_cell_parameters`, `standardize_aunit_orientation`, `noise_cell_parameters`, `do_embedding`,
  `sample_reasonable_random_parameters`, the FD helpers) raise or change shape at Z'>1. These are
  not covered here.
- **#20**: `canonicalize_free_axes` returns early for the whole batch when any row is Z'>1. It is
  pinned through the F4a / F4b `latent` group as today's behaviour.
- **#24**: atomwise RDF at Z'>1 for one compound: the channel meaning changes (528 channels against
  136). It is pinned in the F3 / F3b `rdf/.../atomwise` groups. Stage 7 covers only the
  multi-compound labels.
- **#27**: ellipsoid / GNN periodize treat a cluster as copies of the whole asymmetric unit. This
  is not on the eLJ route and is not covered.
- **The MLIP input builders' own rebuild branch** (`force_rebuild`, or no unit cell yet:
  `uma_utils.batch_to_ase_ucell_list`, `_resolve_aunit_z`, `batch_to_fairchem_atomicdata`;
  `AL_mace_utils._mace_resolve_aunit_z`, `batch_to_mace_atomicdata`) repeats `mol2ucell` but dispatches on
  `z_prime.amax()` and skips its slot guard. Since the stage-1 review fixes every width builds right or refuses
  there, except one case: a row with z_prime < 1 in a width-1 batch is still built as a Z'=1 crystal (`analyze`
  reaches these builders through `mol2ucell`, which refuses it). Calling `mol2ucell` from those branches would
  close it. Not covered here.

## Regenerating, deliberately

Goldens are the reference for "nothing changed". Regenerate them only in a change that alters
behaviour on purpose, and only after reading what moved:

```
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python tests/characterization/build_goldens.py --check        # lists every moved pin
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. python tests/characterization/build_goldens.py --regenerate [--case F1 ...] [--group cluster analyze ...]
```

`--case` and `--group` (group kinds) narrow a regeneration to what moved; every other pin keeps its
stored record. A partial regeneration is refused when a record it would keep was written under a
different environment; regenerate everything then.

The change that regenerates must say which pins moved and why, and justify each against the
per-stage table above. A pin that moves without an intended change is a regression to fix, not to
re-pin.

The fixtures need D:. They are regenerated only if a source row must change, or a fixture is
added. Run `build_fixtures.py --check` first: it separates added entries from changed ones, and
every existing entry must rebuild bit for bit. A changed fixture forces a golden regeneration,
reviewed together. Both generators refuse to overwrite without `--regenerate` and write only into
`data/`.

### Regeneration log

- 2026-10-23, stage 4: one unit-cell element layout (MXtalTools `crystal_building/utils.py`: `unit_cell_gather_index`,
  the vectorised index, and `unit_cell_z`, the per-crystal form by independent arithmetic; readers switched:
  `mlip_interfaces/uma_utils.py` (`batch_to_fairchem_batch`, `batch_to_fairchem_atomicdata`, `batch_to_ase_ucell_list`;
  `_tiled_gather_index` removed), `mlip_interfaces/AL_mace_utils.py` (`batch_to_mace_input_dict`,
  `batch_to_mace_atomicdata_hoisted`, `mxt_crystal_to_mace_atomicdata`, which takes a keyword-only `z_prime`),
  `common/ase_interface.py` ('unit cell' mode), `crystal_ops.py` (`compute_standard_cell`'s spglib numbers),
  `crystal_building/zp_doubling.py::_unit_cell_frac`, `crystal_search/standardize.py::unit_cell_fractional`; tests in
  `tests/test_unit_cell_elements.py`). A unit cell is laid out [unit][image][atom] at Z'>1 ([image][atom] at Z'=1),
  and every reader now gathers each row's element through the atom it is an image of.
  - The 21 #15 xfails flipped (table above); no other xfail did. `build_goldens.py --check` listed 40 moved records,
    exactly the `elements` pins of `two_eth_ether` and `two_pyr_flb` (5 builders x 2 std x {elements, n_wrong}),
    every `n_wrong` now 0; every one-compound `mlip`, `zp_doubling` and `stdcell` pin bit-identical, as the stage table
    requires. Re-pinned with `--regenerate --case two_eth_ether two_pyr_flb --group elements`; a record-by-record diff
    against the backup found exactly those 40 changed and none added or removed (gfn_diffusion
    `energy_sampling/artifacts/fd_design_oct15/units_build_oct22/stage4/golden_diff.py`).
  - An independent review (two reviewers, a refuter per finding; `.../stage4/review/`) found one reader still wrong,
    fixed here: the ASE 'unit cell' mode on a CLUSTER batch read the asymmetric unit as the row slice
    `z[ptr : ptr + num_atoms]`, which on a Z'>1 cluster is unit 0's sub-cluster (4 of 36 / 8 of 96 rows wrong on the
    two-compound fixtures; not a regression, visualisation and CIF export only); it now takes the graph's
    `aux_ind == 0` atoms. It also found the spglib numbers and `crystal_search/standardize.unit_cell_fractional`
    untested at two compounds; `tests/test_unit_cell_elements.py` now checks all three against the geometric truth,
    each with the old tiling as a control. No pin moved.

- 2026-10-23, stage 5: the per-unit all-atom cluster reach (MXtalTools `crystal_building/utils.py`: `molecule_reach`,
  `reference_molecule_reach`, `REACH_MARGIN`, `ucell2cluster`, `_pare_cluster_molwise`, `get_cart_translations(rho=)`;
  `dataset_utils/data_class_methods/crystal_building.py`: `MolCrystalBuilding._unit_reach_buffer` replaces
  `_zp_buffer`, computed after `build_unit_cell`; `crystal_building/image_pairs.py` keeps the same rule; tests in
  `tests/test_cluster_reach.py`). A molecule of a cluster is kept or flagged by its all-atom centroid's distance from
  the cluster's own molecule 0 (the first image in the unit cell, which sits at the clipped `aunit_centroid`),
  `rho = cutoff + zp_buffer + 2 * reach + 1e-3`, `reach` molecule 0's all-atom reach; a Z'>1 unit's cluster by
  `zp_buffer = max_u (R_u + |c_u - c_v|) - R_v` over its own row's units' molecules 0. Until today
  `2 * radius + 0.1` about the posed `pos`, `radius` the heavy-atom radius about the heavy-atom centroid (#16) and, at
  Z'>1, the row's summed radius plus the largest slot distance (#11).
  - Written test first: the first version of `tests/test_cluster_reach.py` failed 10 of 10 on the old code. An
    independent review (three reviewers, a refuter per finding; gfn_diffusion
    `energy_sampling/artifacts/fd_design_oct15/units_build_oct22/stage5/review/`) then found one real defect, fixed
    here: the first version centred the reach on the posed `pos`, which sits about the UNclipped `aunit_centroid`,
    while the unit cell (molecule 0) sits about the clipped one; at a component of 1.0 they are 1e-4 of that lattice
    vector apart, more than the 1e-3 margin for a vector over 10 A, and a pair 3e-4 A inside the cutoff was dropped
    (`test_a_centroid_at_the_clip_is_still_exact`). It also found the tests too coarse to see which centroid sets the
    reach; the synthetic molecule now carries an off-centre hydrogen and contacts 0.01 A inside the cutoff, and six
    in-memory mutants are each caught (`.../stage5/mutants/log_*.txt`: reach about `pos` 2 failures, about the
    heavy-atom centroid 2, 0.14 A short 12, slot-centroid buffer 1, `image_pairs` on the old rule 2; unpatched 94 of
    94 pass).
  - The 17 oracle xfails flipped (table above); no other xfail did. `ORACLE_SETS` gained F4b and the two-compound
    crystals `two_eth_ether` and `two_pyr_flb` (std True and False), whose pins this stage moves; their pair sets equal
    brute force.
  - `build_goldens.py --check` listed 228 moved records, every one in a group the stage table allows: `cluster` (F1,
    F3, F3b, F4a, F4b), `analyze` (those and both two-compound cases), `rdf`, `exp6` (F1, F3, F3b) and, in `mlip`, only
    `uma_cluster_route/result/elj` and `mace_cluster_route/result/elj` (every stub input and unit-cell route
    bit-identical). Re-pinned with `--regenerate --group cluster analyze rdf exp6 mlip`; a record-by-record diff
    against the backup found exactly those 228 changed and none added or removed (`.../stage5/golden_diff.py`).
  - Per crystal, before -> after (`.../stage5/logs/t14_compare_fixed.txt`, 56 crystal-settings): 9 gained exactly
    their missing pairs, none lost a pair. eLJ (raw) moved by the restored pairs: phenacetin P2₁/c -0.088 / -0.080
    (cut 6, std True / False), pyridone P2₁/c -0.018 (cut 6, std False) and -0.0037 (cut 10), probe1b -0.020 (cut 6),
    HIFQUJ -0.0087 (cut 10); elsewhere by at most 8.5e-4 on |eLJ| up to 1361, summation order.
  - Cost: Z'=1 clusters grow by the hydrogens' reach, by the molecule: summed over the fixtures' 36 Z'=1 settings, 18 %
    more cluster atoms; on 512 prior rows at cut 10, MIPCAS sg 2 7.4 % more window atoms (aux_ind < 2), qm9full 1.6 %
    fewer, acridine sg 14 38.5 % more (reach - radius = 1.01 A; wall time 1.06 -> 1.10 s, peak working set 1.68 ->
    1.81 GB, one CPU build each, the review's measurement). Z'>1 clusters shrink: 1.8 times fewer atoms over the
    fixtures' 20 Z'=2 settings (F3 AHUMIZ 20608 -> 11776 at cut 10).

- 2026-10-23, stage-3 review fixes (MXtalTools `data_classes.py`, `dataset_utils/utils.py`, `crystal_ops.py`; tests
  in `tests/test_batch_unit_levels.py`). No pin moved: `build_goldens.py --check` reproduces every golden bit for bit
  (gfn_diffusion `energy_sampling/artifacts/fd_design_oct15/units_build_oct22/stage3/review_fix/logs/
  fix2_5_goldens_check.txt`), and no strict xfail flipped. What changed, each with a test written first and seen
  failing (`.../review_fix/logs/failfirst_2.txt`):
  - A slice entry that describes a value decides its level before its length (rule 3 of the contract above). A
    collated per-edge or per-unit value whose total equalled the atom or row count was sliced per atom or per row by
    `subsample_new_batch`, `append_batch` (which then wrote `ptr` / `arange` as its entry for good),
    `rebuild_simple_slice_inc_` and `set_mol_attrs`' batch path, while `select_rows` and `batch_to_list` read the
    entry; e.g. 14 edges on 14 atoms: a draw of row 1 had 8 of its 9 values; z' [2, 0]: a draw of row 0 had 1 of its
    2 units. All helpers now agree. A device-held entry (`add_graph_attr` / `add_node_attr` on a CUDA batch) is still
    not read for a value of the atom or row count (no host sync on the hot path).
  - `append_batch` keeps a key registered `BATCH` whole (a 231-row batch's `sym_mult_lut` became [462]).
  - An 'index' / 'face' key whose entry describes it must hold its rows' atom offsets as increments (what collation
    writes), else `subsample_new_batch` / `append_batch` refuse it, as `select_rows` already did: a per-row value
    under such a name attached with `add_graph_attr` (zero increments) was shifted as atom indices ([9] became [-9]).
    A one-dimensional or [1, E] index value without an entry is refused rather than read as atom indices.
  - The loaders MXtalTools builds (`MXtalDataLoader`) fill the unit map of map-less rows beside rows that carry one,
    as `collate_data_list` does (`fill_unit_maps`, factored out of it); PyG's own `DataLoader` still does not.
  - Intended value change, recorded here (it came with stage 3 and the review confirmed it): a per-row or per-atom
    tensor whose name holds 'index' or 'face' (crystal search's per-row `dataset_index`, `build_conformer_tree`'s
    per-atom `tree_perm_index`) is the `edge` level, so `subsample_new_batch`, `append_batch` and the list
    constructor re-offset its values by the atom counts, as collation offsets them. Values differ from stage 2's (the
    review's probe on csd_fd_v5 rows with ids 100..103: `subsample([1, 2])` gave `dataset_index` [136, 182], now
    [101, 147]; `tests/test_batch_unit_levels.py::test_a_collated_per_row_index_value_round_trips`); they are the
    collation-consistent ones, so `batch_to_list` gives the rows' own ids back (stage 2 gave [136, 182], now
    [101, 102]). No live caller subsamples or appends `dataset_index` (the search goes collate -> `batch_to_list`,
    unchanged). Within the list constructor, molecule k's [1]-shaped `dataset_index` is offset by the atoms before it.
    Renaming crystal search's row id away from 'index' would take it out of the naming contract (a crystal_search
    change, not made here).
  - Bit-identity on ordinary batches: `subsample_new_batch`, `append_batch` and `rebuild_simple_slice_inc_` outputs
    (values and bookkeeping) are equal before and after these fixes on `mini_new_csd.pt` batches and on two r2 ingest
    store chunks (500 rows each; `.../review_fix/logs/perf_identity_compare.txt`); the CSD_V1 chain on the 16 CIF
    fixtures and the 156-CIF r2 sample stores the same rows bit for bit as the stage-3 run
    (`.../review_fix/chain/chain_compare_fix.txt`). The level plan costs 0.04-0.14 ms more per call on a collated
    batch (one host comparison per slice entry; 0.01 ms on a helper-made batch, whose entries share one object).

- 2026-10-23, stage 3 (the batch helpers' unit and edge levels, the level contract above; the stage-2 collate gap
  closed; the ingestion chain's map writers completed: MXtalTools `data_classes.py`, `dataset_utils/utils.py`,
  `crystal_ops.py`, `crystal_io.py`, `crystal_steps.py`, `crystal_chem.py`, `crystal_store.py`, comments in
  `crystal_building/utils.py`):
  - `build_goldens.py --check` before the regeneration listed 17 moved records, all "only in recomputed", all the new
    key `unit_ind` and its `_slice_dict` / `_inc_dict` entries, and nothing else (gfn_diffusion
    `energy_sampling/artifacts/fd_design_oct15/units_build_oct22/stage3/impl/logs/check_before_regen.txt`):
    - `helpers` of F3 (14 records: `subsample_10`, `subsample_0`, `append_clone`, `relist`, `batch_to_list/row0`,
      `row1`): `from_arrays` with a pose now writes its layout as the map (Z' equal contiguous blocks; -1 on a
      z_prime_mismatch row). Each new record equals the `mol_ind` record beside it bit for bit (values, slice entry
      `ptr`, zero increments). Justification: the stage-2 row of the table below lets `helpers` move by this new key;
      the stage-3 spec moved `from_arrays`' map writer from stage 2 into stage 3.
    - `ingest` of F3 (3 records, after `find_molecules`): `find_molecules` now writes the map (the units found, equal to
      `mol_ind`; -1 on rows that fail), as the stage-2 spec intended (it wrote none because `sanitize`, between it and
      `parameterize_pose`, could not fill it for the hydrogens it places; it fills it now). The records after
      `parameterize_pose` and `check_rebuild` did not move: the map they keep is the one stage 2 wrote there, bit for
      bit. Justification: as above (the stage-2 row lets `ingest` move by the new key).
  - Re-pinned with `--regenerate --case F3 --group helpers ingest`; `--check` after: "goldens reproduce bit for bit"
    (`.../stage3/impl/logs/check_after_regen.txt`). The manifest gained exactly the 17 entries.
  - Every other pin of every case reproduces bit for bit, the stage's proof: `helpers` of F1, F3b and F4a (atom and row
    values only, no index or ragged value) are bit-identical through the rewritten `subsample_new_batch`,
    `append_batch`, `batch_to_list` and the collate, as are every build, energy, RDF, MLIP, canonicalisation, latent,
    standard-cell, reparameterisation and zp_doubling pin.
  - The CSD_V1 chain (read -> ... -> features) on the 16 CIF fixtures and on 156 real CIFs (6 chunks of the r2 ingest
    store; 108 ok rows, 36 of them Z'=2): every row's every stored value bit-identical before and after the stage-3
    code, statuses equal, `unit_ind` equal to `mol_ind` on every ok row and -1 on every atom of the rows that fail at
    or before `find_molecules`. Against the r2 store rows themselves (written by the code of 2026-10-22 morning,
    before stages 1-3): statuses equal on 156 of 156, every common key equal on the rows' own pose slots except `pos`
    of ADOCUU (two placed hydrogens, 4.4e-16 A, one float64 ulp), which the pre-stage-3 code reproduces too, so it is
    not a stage-3 change (gfn_diffusion `.../stage3/chain/chain_compare.txt`).

- 2026-10-23, stage-2 review fixes (MXtalTools `crystal_building/utils.py`, `data_classes.py`, `crystal_ops.py`,
  `crystal_steps.py`, `crystal_io.py`; a comment in `crystal_rdf.py`): no pin moved (`build_goldens.py --check`: "goldens
  reproduce bit for bit"; no regeneration). Each change is pinned in `tests/test_unit_map.py`, refusals also under
  `python -O`; each new test failed on the tree before the fixes (21 of 48, in a shadow copy with the fixes undone):
  - `append_batch` gives a map-less side its derived map (it raised KeyError, or dropped the map without `validate`);
  - a map-less Z'>1 crystal row re-wrapped by the constructor keeps its equal blocks (it was given one unit, and its
    valid Z'=2 refused);
  - `parameterize_pose` writes the map from `find_molecules`' units (`mol_ind`): -1 (`NO_UNIT`) on rows whose units
    are not known as Z' equal blocks (it wrote one unit of all 46 atoms on EFUNUP, z_prime 0), `pose_failed` on an
    ok row whose `mol_ind` is not those blocks (it posed the blocks without reading `mol_ind`), and a map the batch
    already carries is checked and kept (it was refused as a second pass); `check_rebuild` builds from the rows' map;
    `check_validity`'s unit criteria run on a batch with the map and no `mol_ind`;
  - `from_arrays` takes a map in `atom_attrs` with a pose (it refused one), a row whose map is not its Z' equal blocks
    (the 6 + 8 atom case of #1) being a `z_prime_mismatch` row; without one it writes none, as before;
  - the map must be a one-dimensional integer (non-bool) tensor with one entry per atom, at construction and in every
    reader (a float map was truncated, an [N, 1] one broadcast to [N, N] by canonicalisation), comes out of
    canonicalisation long, and one row's `z_prime` must hold one value;
  - the list constructor stores its summed row scalars 0-dim, so a row built from collated molecules collates beside
    a default row.
  - Not fixed here: `collate_data_list` of rows with and without a map (above, "Known gap after stage 2").

- 2026-10-22, stage 2 (the atom -> unit map: a per-atom `unit_ind` on crystal rows, read by every reader through
  `crystal_building.utils.unit_layout`; MXtalTools `crystal_building/utils.py`, `data_classes.py`, `crystal_ops.py`,
  `crystal_building.py`, `crystal_io.py`, `crystal_steps.py`, `exp6_ff.py`, `crystal_rdf.py`, the old featurizer's
  `featurization_utils.py`):
  - `build_goldens.py --check` before the regeneration listed 210 moved records, all in groups the stage table lets
    stage 2 move, and nothing else (gfn_diffusion
    `energy_sampling/artifacts/fd_design_oct15/units_build_oct22/stage2/impl/logs/check_after_code_1.txt`):
    - `helpers` of F1, F3b and F4a (44 records, all "only in recomputed"): the new key `unit_ind` and its
      `_slice_dict` / `_inc_dict` entries on every subsampled, appended, split and re-collated batch (the
      constructor now writes the map: one molecule one unit, molecule k of a list unit k). Every record these groups
      held before reproduces bit for bit. F3's `helpers` did not move: `from_arrays` writes no map (below).
    - `ingest` of F3 (6 records, "only in recomputed"): `unit_ind` and its entries after `parameterize_pose` and
      `check_rebuild` (the step that poses the units writes the map: Z' equal contiguous blocks, here equal to
      `mol_ind`). `find_molecules` writes none (`sanitize`, between the two, refuses per-atom attributes it cannot
      fill for the hydrogens it places), and every other record of every step is bit-identical.
    - `two_eth_meth` and `two_eth_prop`: `split`, `ucell/std*`, `elements/std*` and `analyze/std*` (160 records): the
      [7, 7] / [10, 10] splits and the wrong energies, unit cells and elements built from them became one
      `raises` record each, ValueError from `split_to_zp1_batch` naming the row (units of [9, 5] / [9, 11] atoms are
      refused until stage 7). Their `box` and `canonicalize` pins did not move (the constructor writes an unequal
      map without refusing it; their slots are already in canonical order, so canonicalisation reads no unit).
  - Re-pinned with `--regenerate --case F1 F3b F4a --group helpers`, `--case F3 --group ingest` and `--case
    two_eth_meth two_eth_prop --group split ucell elements analyze`; `--check` after: "goldens reproduce bit for bit".
  - Every other pin of every case reproduces bit for bit, the stage's proof: the unit map equals the equal-block
    arithmetic for equal units, so split, cluster, energies, RDF (envwise labels read the map), exp6 typing (reads
    the map), MLIP assembly, canonicalisation, latents, the standard cell, reparameterisation and zp_doubling are
    unchanged, as are the two equal-count two-compound cases (`two_eth_ether`, `two_pyr_flb`).

- 2026-10-22, stage 1 (refuse instead of guessing; MXtalTools `crystal_building.py`, `crystal_building/utils.py`
  `_index_mols`, `dataset_utils/utils.py` `collate_data_list`):
  - Moved and re-pinned (`--regenerate --case F4a F4b --group cluster analyze rdf mlip`), 30 records per case,
    all on the Z'=1 phenacetin row 0 of the max_z_prime=2 batch, whose `zp_buffer` no longer counts its pad
    slot (stage 1 may move these groups; table below):
    - `cluster/std1/cut6`, `cut10`: the cluster shrinks from 10560 to 8480 atoms (cut 6) and from 17216 to 13992
      (cut 10); `pos`, `z`, `x`, `aux_ind`, `batch`, `ptr`, and `mol_ind` (numbered across the batch, so row 1's
      shift too) move with it; the inter and intra edge sets are renumbered, their counts (3926 / 20432 inter, 716
      / 882 intra) and the inter distances' multiset unchanged (`inter_dist_raw_order` keeps its sum).
    - `analyze/std1/cut6`, `cut10` `elj`: row 0 moves by one float32 ulp, -527.7417603 -> -527.7416992 (cut 6) and
      -637.1484375 -> -637.1484985 (cut 10), a summation-order change over the same pairs; the new value equals
      phenacetin scored alone, bit for bit. Row 1 (AFEVOA, Z'=2) is bit-identical (-320.9040833, -423.1654968).
    - `mlip/std1` `uma_cluster_route/result/elj`, `mace_cluster_route/result/elj`: the same row-0 eLJ at cut 6
      (these routes return the cluster eLJ); every stub input and every other `mlip` pin unchanged.
    - `rdf/std1/cut6`, `cut10` for `elementwise`, `envwise`, `atomwise`: summation order only. Rebuilt with the
      old all-slot buffer patched in, row 0 differs by at most 1e-7 relative per bin and now equals phenacetin
      alone bit for bit (elementwise, atomwise; envwise channels depend on the batch); row 1 is bit-identical.
      `rdf_mode` None, the bins and the channel labels unchanged.
  - Every other pin of every case reproduces bit for bit (`build_goldens.py --check` before the regeneration
    listed exactly the 60 records above, and nothing after it). The oracle tests pass for both F4a rows (pair
    sets and eLJ against the brute-force sum).
  - Moved on purpose by the same per-row buffer, in rows no pinned case holds (recorded 2026-10-22, from the
    stage-1 review's before/after comparison on r2 ingest-store rows: gfn_diffusion
    `energy_sampling/artifacts/fd_design_oct15/units_build_oct22/stage1/review_contract/cmp_report.txt`, reproduced
    in `.../stage1/review_fix/repro.txt`; raw eLJ, before minus after; after, every row equals its alone build bit
    for bit):
    - **Z'>1 rows narrower than their batch** (z_prime < max_z_prime, so they carry pad slots; the r2 store is
      written at width 4). Their buffer drops the pad distances, so they build the cluster they build alone.
      The stage-1 task's sentence "valid Z'>1 rows keep today's buffer bit for bit" holds only for rows that use
      every slot (`test_zp_buffer_of_full_rows_is_unchanged`; ZZZUXA02 below). The move follows the plan's
      stage-1 change ("`zp_buffer` ignores pad slots") and audit #13: a pad slot is not a unit, and counting it
      would keep these rows' scores batch-dependent. In a width-4 batch:
      - AADRIB (Z'=2): cluster 41760 -> 24080 atoms (std True, cut 6), eLJ by up to 1.34e-3 (std False, cut 10);
      - ABABEM (Z'=2): eLJ by up to 3.1e-4;
      - ABABUC (Z'=2): cluster 100640 -> 56480 atoms, eLJ by up to 4.9e-4;
      - BIDSUD (Z'=3): cluster 51910 -> 41006 atoms, eLJ by up to 1.2e-4;
      - ZZZUXA02 (Z'=4, every slot used): bit-identical.
      Pinned by `tests/test_zprime_refusals.py::test_zp2_row_in_a_wider_batch_builds_its_alone_cluster` and
      `::test_real_zp2_row_in_a_width3_batch_scores_as_alone` (the F3 rows beside F1 phenacetin at width 3).
    - **Z'=1 rows on real data move by more than summation order.** On F4a / F4b the pairs were the same. On
      real rows the pad-widened cluster had also held pairs within the cutoff that the heavy-atom reach test
      misses (#16); without the pad the row misses them, exactly as it does alone. In width-2 batches:
      - AABHTZ: -0.231 (std False, cut 6; cluster 7420 -> 2660 atoms) and -7.57e-3 (std True, cut 10);
      - ABABEO: -3.42e-2 (std False, cut 6);
      - the other three Z'=1 rows of the sample (AAPYPE, ABABAI, ABABAK): within 1e-3.
      Pair counts were not measured; the attribution to #16 is the mechanism, since a wider selection can
      only add molecules (`test_oracle_finds_what_a_widened_cluster_finds`). A GFN run that mixes Z' in one
      batch sees shifts of this size. Stage 5 (per-unit reach) restores the pairs; the pyridone test above
      (4993 against the true 4995 pairs) pins the mechanism on one row.

- 2026-10-22, stage-1 review fixes (MXtalTools `crystal_building.py`, `crystal_building/utils.py`
  `get_aunit_positions`, `common/geometry_utils.py` `batch_molecule_principal_axes_torch` and
  `scatter_compute_Ip`): no pin moved (`build_goldens.py --check`: every pin of every case reproduces bit for
  bit). Each change is pinned in `tests/test_zprime_refusals.py`, refusals also under `python -O`; each test
  failed before its change.
  - `mol2ucell` / `mol2cluster` refuse a row with z_prime < 1 at every width (`_refuse_z_prime_outside_slots`).
    At width 1 it had been posed and scored as a Z'=1 crystal, while the split refused it at width 2.
  - A batch without `aunit_handedness` splits without one. `std_orientation=False`, which never reads it,
    builds at width 2 exactly as with it (it had stopped in the split, TypeError); `std_orientation=True` is
    refused by `pose_aunit`, as at width 1.
  - `build_unit_cell` on a batch with more than one pose slot:
    - an all-Z'=1 batch is built from slot 0, the width-1 batch's unit cell bit for bit;
    - a row whose z_prime is not 1 is refused, naming the row (split Z'>1 rows first);
    - both had stopped in torch einsum. This is where the MLIP input builders' own rebuild branch lands: it
      dispatches on `z_prime.amax()`, so an all-Z'=1 batch at width 2 reached it unsplit. Every builder now hands
      the model the width-1 batch's input (`test_mlip_rebuild_branch_takes_an_all_zp1_batch_at_width2`); the
      remaining gap is under "Known faults".
  - Bare asserts on the std_orientation path are explicit raises:
    - `get_aunit_positions` without a handedness: ValueError;
    - `batch_molecule_principal_axes_torch`, atom counts that disagree with `batch`: ValueError (under `-O`
      the mismatch had passed silently);
    - `scatter_compute_Ip`, when the CPU eigh fails (e.g. a NaN coordinate): RuntimeError naming the graphs
      whose inertia tensor is not finite (under `-O` it had fallen through to an UnboundLocalError). On CUDA the
      CPU retry already raised torch's own error, a RuntimeError subclass.
  - The refusal-rule control's einsum example is now the module function `aunit2ucell` on unsplit F3, since the
    `build_unit_cell` method refuses that batch.

- 2026-10-22, review fixes (no MXtalTools code changed):
  - `mlip` of every case was re-pinned. The stubs now return non-zero energies; the UMA stub
    attaches the external graph; the UMA fields add `dataset` and `sid`; the cluster routes and the
    ase assembly were added; lists of arrays (MACE `cell`) are stored with their dtype and shape.
    Every other moved `mlip` pin was one of these.
  - F3 `box` gained `box_analysis_on_clone/*`.
  - New groups `reduction`, `helpers`, `zp_doubling`, `stdcell`, `reparam`, `ingest` and `exp6`.
  - New two-compound cases.
  - The fixture `AUDIT/two_eth_prop` was added; every other fixture entry rebuilt bit for bit.
  - `build_goldens.py --check` before the regeneration showed no other existing pin moved.

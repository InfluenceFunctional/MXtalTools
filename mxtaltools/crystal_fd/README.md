# crystal_fd: fundamental domains of Z' = 1 molecular crystals

**Status (2026-10-15).** Stage 0 of the design note `gfn_diffusion/energy_sampling/docs/design/fd_tooling.md`, owner
scope steps 1-3: (1) per-group tables, (2) core operations and the pose fold, (3) charts and the reference measure for
space groups 1 and 2. Since 2026-10-16 also the monoclinic cell canonicaliser and cell chart (sg 3-15) and an
importer from MolCrystalData rows (`io_mxt`: nonstandard settings re-expressed in the standard setting by `settings`,
owner decision D1; off-sheet Sohncke rows given their own enantiomer as reference, D2; Z' = 1 by default, D3).
Since 2026-10-17 also canonical datasets on disk (`dataset`: save, load, and a converter back to MolCrystalData
rows), with which the canonical CSD dataset (owner decision D6) was written: `D:/crystal_datasets/csd_fd_v1b`, which
records the code that wrote it; the earlier `csd_fd_v1` holds bitwise the same rows without the code digest and the
derived flags and is superseded. Since 2026-10-21 also the enantiomorph (`enantiomorph`; the importer's
`off_sheet='enantiomorph'`, owner decision of that date): an off-sheet Z' = 1 Sohncke row stored as its mirror-image
crystal with the unmirrored reference, as an alternative to D2; D2 stays the importer's default. Since 2026-10-21 also
a cell step for every group above monoclinic (`axis_order`, `canon`): the cell snapped to the group's exact metric and,
in sg 16-74, put in the group's canonical axis order, so `canonicalize` no longer returns `pose_only` rows. Since
2026-10-21 also charts and the reference measure for sg 16-230 (one coordinate per free cell parameter) and for sg 6-9
as an explicit double cover of the K cut.
Nothing outside this package and `tests/fd/` consumes it yet: no GFN adapter, no prior migration, no CIF/format
layer. The design note is the specification; where this code departs from it, the code and
its gates (below) are what is true now.

## What it does

A crystal is a `CrystalDescription`: space group `sg` (setting `SYM_OPS[sg]`), cell, and per asymmetric unit a
fractional centroid `t`, a proper rotation matrix `R` and a handedness `h = +-1`; the molecule `X` (a `MoleculeRef`,
centred on its heavy atoms as the builder centres it) sits at `T_fc t + R diag(h, 1, 1) X`.

The package turns any description of a crystal into one canonical description, and gives coordinates on the set of
canonical descriptions:

1. **Cell** (`ops.reduce_cell`, sg 1 and 2): the tri_niggli reduced basis, chosen from the lattice alone
   (`sym_utils.tri_niggli_reduction_penalty` at margin 0; ties within 1e-10 broken by the candidates' own Gram
   entries, never by the input basis). Sg 3-15 (since 2026-10-16): the basis at which
   `sym_utils.mono_reduction_penalty` at margin 0 is zero, among the ac-plane changes a' = p a + q c,
   c' = r a + s c, b' = (ps - qr) b that the group's setting class allows (`MONO_CLASS`: r even for 7, 13, 14; q even
   for 5, 8, 9, 12, 15; any for 3, 4, 6, 10, 11), each with the origin shift `standardize.setting_shift` gives it;
   alpha and gamma are set to exactly 90 deg (rows more than 1e-6 rad off are refused). The two bases of every
   monoclinic cell (N and N times the 2-fold about b) differ by an element of Gamma+, which the fold absorbs. Of the
   candidates tied within 1e-10 of a face the least violating (the one inside the domain) is taken, so the choice
   is the lattice's and the reduced cell lies in the closed domain; the key decides only within rounding of a face.
   Sg 16-230 (since 2026-10-21, `canon`): the cell is snapped to the group's exact metric (the mean of W^T G W over
   G's rotations; rows further than `metric_tol` from it are `cell_off_group`). Above monoclinic the only setting
   changes that are not isometries of the generic metric are orthorhombic axis permutations (survey of 2026-10-21):
   in sg 16-74 the cell is put in the lexicographically smallest (a, b, c) among the axis orders the group allows
   (`axis_order`: 11 groups allow all six, so a <= b <= c; 26 allow a <-> b with c fixed, so a <= b; Pbca (61) the
   three cyclic orders, so the shortest axis first and, on a tie, b <= c; 21 allow one), with that order's det +1
   signed permutation and quarter-grid origin shift. In sg 75-230 the metric-preserving setting changes are Gamma's
   linear parts, which the fold absorbs, so the snapped standard cell is canonical.
2. **Pose** (`domains.fold`, all 230 groups, Z' = 1): the image of `(t, R)` under Gamma+ (the det +1 part of the
   Euclidean normaliser, modulo Z^3 and free-axis shifts) that lies in ASU(Gamma+) x SO(3), with cctbx's tie-break cuts
   on faces and an orientation rule at special positions (in sg 6-9 this is the K cut in SO(3)). Free axes are gauged
   to the value `MolCrystalOps.canonicalize_free_axes` sets. The fold never builds atoms and takes no molecule: it
   works on the +1 sheet and never folds through molecular point symmetry.
3. **Charts** (every group; sg 6-9 only as a double cover): coordinates on the canonical set with exact maps and
   closed-form log-Jacobians, plus the trainer's latent read as a chart (log-Jacobian sg 1 and 2 only).
4. **Measure**: the log-density of Haar(dR) x (V d^k t) x mu_cell relative to a chart's coordinates, and image
   multiplicities between domains.

Notation: G the space group; Gamma its Euclidean normaliser N_E(G) modulo Z^3 and free shifts; Gamma+ the det +1 part;
K the subgroup of Gamma+ acting trivially on the centroid once free axes are projected out (nontrivial only in sg 6-9);
ASU(Gamma+) cctbx's asymmetric unit of Gamma+/K on the non-free coordinates.

## Owner decisions in force (2026-10-15)

- **The orientation chart for the flow is deferred.** Rotations are stored as matrices. `so3` converts exactly to and
  from four charts, each with its Haar log-density: `'hemisphere'` (the trainer's (theta, phi, r), exactly as
  MXtalTools builds it), `'quaternion'` (w >= 0), `'euler_zyz'` (alpha, cos beta, gamma) and `'axis_angle'` (the pi
  ball). No chart is the default; every call names one.
- **The cell measure is an explicit required argument**, never defaulted: `'latent_v1_lebesgue'` (sg 1-15: the
  measure the trainer's latent chart puts on the cell today; from orthorhombic up, see below) or
  `'metric_invariant'` (dG / V, basis-invariant).
- **From orthorhombic up (2026-10-21) the parameterization is designed for efficiency, not for today's trainer**
  ("We will change the trainer so it makes more physical sense. Averaging is probably dumb."): one cell coordinate per
  free cell parameter, and `'latent_v1_lebesgue'` means flat in each free log length (and free angle) in every family.
  The trainer's averaged length rows are not mirrored by any adapter. sg 6-9 enter only through
  `orientation_cover='double'`; orientation charts stay as they are (O3 deferred).

## API in brief

- `tables`: `group_table(sg, device=None) -> GroupTable` (G, Gamma, Gamma+, K, free axes, both asymmetric units
  with tie-break cuts, facet gluings, Fourier-Motzkin AR bounds, the trainer box with its coverage and `k_box`);
  `TABLE_HASH`; `verify_manifest`. The table `data/fd_tables_v1.json` is read once at import and refused
  (`TableMismatchError`) if SYM_OPS, NORMALIZER_OPS, CONTINUOUS_DIMS, ASYM_UNITS or the file differ from
  `data/MANIFEST.json`. Its digest reads CRLF as LF, and `data/.gitattributes` stops git converting it.
- `description`: `CrystalDescription` (`from_parts`, `from_poses`, `from_crystal_params` / `crystal_params` for
  MolCrystalData's layout, `cell_matrix`, `validate`; every field on `t`'s device), `MoleculeRef`.
- `so3`: `split_o3`, `random_rotations`, `to_chart` / `from_chart`, `haar_log_density`, `CHARTS`.
- `ops`: `apply`, `compose`, `inverse`, `cartesian_rotation`, `change_basis`, `reduce_cell(desc, tol=1e-10,
  angle_tol=1e-6, face_tol=FACE_TOL) -> CellReduction` (`desc, N, s, n_tied, violation, n_auto`; empty batches give
  empty results), `FACE_TOL` (1e-5, the special-metric flag's tolerance), `triclinic_basis`, `lattice_automorphisms`,
  `real_cells`, `check_cells`;
  monoclinic: `monoclinic_basis`, `mono_conditions` (one dimensionless condition per wall of the margin-0 zero set),
  `mono_bound` (the class's bound m on -cos beta), `mono_parity`, `mono_class`, `mono_setting_shifts`.
- `domains`: `fold -> FoldResult` (`desc, element, lattice, free_shift, slack, tie`), `unfold` (its exact inverse),
  `in_domain` (a row is in the domain iff it is its own fold), `in_asu` (the rules on one description, independent
  of `fold`; for validating a domain), `images` (`group='gamma_plus' | 'gamma'`, `within=None | 'g_asu' |
  'latent_v1_box'`), `to_sheet`, `gauge_free_axes`, `translation_section`. For Z' > 1 (stage Z2 of
  `energy_sampling/docs/design/zprime_fd.md`): `fold_unit(sg, cell, t, R, D=None, tol=1e-9) -> UnitFold` (`t, R,
  element, W, w, lattice, mirrored, slack, tie`) folds one unit per row into ASU(G) x SO(3) under all of G, an
  improper element mapping R to M R D (`D` default diag(1, 1, -1), the conformer seed-plane reflection) and setting
  `mirrored`, where the caller applies sigma to the unit's conformer; `unit_section`.
- `canon` (2026-10-20): `canonicalize(desc, *, sheet=1, tol=1e-9, cell_tol=1e-10, angle_tol=1e-6,
  face_tol=FACE_TOL, near_tol=FACE_TOL, ..., nudge=None) -> Canonicalization` runs the cell step (`reduce_cell` for
  sg 1-15; since 2026-10-21 the exact-metric snap and, sg 16-74, the axis order for sg 16-230), `to_sheet` and
  `fold` on any batch (space groups and Z' mixed) without raising for a row: rows with bad values, no real cell,
  Z' > 1, a monoclinic alpha or gamma off 90 deg or (sg >= 16) a cell off the group's metric are sorted out first,
  and a step that still raises is retried on halves of its rows down to the failing ones. Each row gets a status
  (`STATUS`, order fixed: `canonical`, `special_metric`, `off_sheet`, `pose_only`, `bad_values`, `unreal_cell`,
  `z_prime_gt1`, `mono_angle_off`, `cell_off_group`, `reduce_failed`, `sheet_failed`, `fold_failed`) and the cell
  and fold records; rows not canonicalised come back unchanged. Every space group now has a cell step, so a row that
  passes is `canonical`, `special_metric` (`n_auto` > 1) or `off_sheet` whatever its group; `pose_only` (sg >= 16
  until 2026-10-21) stays defined and is no longer produced. Records: `N`, `s` (the cell step's basis change and
  shift, `x_input = N x_canonical + s`: `reduce_cell`'s, or for sg 16-74 the applied signed axis permutation with
  its shift; identity for sg 75-230), `n_auto`, `n_tied`, `violation` (sg 16-74 in log lengths; 1, 1, 0 for sg
  75-230 and one-order groups), `cell_snap` (sg >= 16: the snap's size, largest change of a length, relative, or of an
  angle in rad; NaN for sg 1-15), the fold's fields. `near_face` is the CSD set's bitmask (`NEAR_CELL`,
  `NEAR_CENTROID`, `NEAR_FOLD_BAND`), `moved` marks rows whose canonical form differs from the input.
  `nudge=delta` (2026-10-21, owner decision; default None moves nothing and gives the same output as before):
  `canonical` rows with `NEAR_CELL` or `NEAR_CENTROID` are moved to the nearest point at least delta from every face,
  in `near_face`'s units: the cell-chart point (t and R kept; sg 16-74 the log lengths (log a, log b, log c) of
  `axis_order.OrthoCellChart`, whose face values are differences of log lengths, as the FREE monoclinic face
  log c - log a), then the fractional centroid in ASU(Gamma+) (cell and R kept, free axes at the gauge). An sg >= 16
  row whose fold chose the pose by the orientation rule (tie 2, a special position) first takes the pose that rule
  picks on the group's exact metric, applied with the row's own Q: an image of the row's pose (since the cell step
  snaps the cell, the canonical row's own pose to rounding). Special metrics, off-sheet rows and `NEAR_FOLD_BAND`
  alone are not moved. A moved row is kept only if canonicalising it again gives the same status, itself within
  `NUDGE_FIXED_TOL` (1e-9) and neither bit; otherwise, and where the domain is thinner than 2 delta, the unmoved row
  is kept with its flags. Fields `nudged`, `nudge_cell` (chart distance), `nudge_t` (largest fractional centroid
  change), `nudge_message` (why an eligible row was not moved); `near_face` and `moved` describe the moved row, the
  other records its unmoved canonical form. A moved row is another crystal: rescore its energy. What the nudge does
  not cover: Known limits.
  `fd_status(desc, *, sheet=1, tol=None) -> FDStatus` is the computed fundamental-domain check (no stored flag):
  `is_fd` iff the canonical form is decided (canonicalised with a cell step and Z' = 1: every space group since
  2026-10-21, sg 1-15 before), not a special metric, on the sheet, and equal to the row within `tol` (1e-9 for
  float64 rows, 1e-5 for float32); also `decided`, `cell_reduced` (the cell is its canonical cell), `pose_folded`
  (the pose against its own fold, in the row's basis), `on_sheet`, `special_metric`, `near_face`, `zp1`. It passes
  `nudge` on: the row is then compared with its nudged canonical form.
- `axis_order` (2026-10-21): `axis_orders(sg) -> AxisOrders(sg, kind, perm, N, s, faces, mirror)` for sg 16-74,
  derived at first use for all 59 groups from `standardize.setting_shift` (every det +1 signed axis permutation tried,
  the origin on the quarter-cell grid) and checked exactly with `settings.setting_matches`; of each admitted
  permutation the first sign choice is kept (all four are admitted in every group: the others differ by a diagonal
  2-fold of Gamma+); the classes must be the survey's (21 / 26 / 1 / 11 groups with 1 / 2 / 3 / 6 orders, Pbca the
  cyclic one), else `RuntimeError`. `classes()`, `choose(sg, cell, tol, face_tol) -> (pick, n_tied, n_auto, violation,
  face)` (the lexicographically smallest (log a, log b, log c) within `tol`; `n_auto` is the order of the subgroup
  generated by the allowed orders giving the chosen cell within `face_tol` in log lengths, so 1, 2, 3 or 6 even on
  a chain of near-ties), `OrthoCellChart` / `cell_chart`, `face_values`. Faces in
  y = (log a, log b, log c): six orders log b - log a and log c - log b; a <-> b log b - log a (both mirrors, so a
  cell within `face_tol` of one is a special metric); Pbca log b - log a and log c - log a (seams glued by the
  3-cycle, flagged `NEAR_CELL` and nudged; the cone point a = b = c is a special metric with `n_auto` 3).
- `charts`: the protocol is `dims` (kind `interval | circle | fixed | embedded`, period, AR position), `forward(y)`,
  `inverse(desc)`, `bounds(y_prefix, i)`. `CellChart('triclinic', length_box)`, `CellChart('monoclinic', length_box,
  sg=...)` ((log c, log a, log b, s), cos beta = -s m(a, c), s in [0, 1]; log|det| = log m - log sin beta relative to
  d(log a) d(log b) d(log c) d(beta); FREE groups carry a <= c in the length bounds; `unmap` puts coordinates within
  `CellChart.SLACK` = 1e-9 outside a bound on it, the rounding of a canonical cell on a face). Since 2026-10-21, from
  orthorhombic up (`cell_family(sg)`, `FAMILY_SG`; trigonal P and R groups, on hexagonal axes, are `'hexagonal'`), one
  coordinate per free log length, angles fixed (`HALF_PI`, gamma `HEX_GAMMA` = acos(-1/2)), b = a and c = a bit for
  bit, log|det| 0 relative to d(free log lengths): `CellChart('orthorhombic', box, sg=...)` (log a, log b, log c),
  the columns of `axis_order.OrthoCellChart`, with the class's canonical domain as AR bounds: none, a <= b, a <= b <= c,
  or Pbca's shortest-first wedge a <= b, a <= c; `faces`, `face_kinds` (mirror faces are walls) and `gluings`
  (`CellGluing`: Pbca's two seams glued by the 3-cycle, interval ends under O1 (a), meeting at the cone a = b = c);
  `'tetragonal'` and `'hexagonal'` (log a, log c); `'cubic'` (log a). `unmap` refuses cells further than
  `CellChart.FAMILY_TOL` = 1e-6 from the family metric. `PolytopeARChart(sg)` (u in
  [0, 1]^k; `t_bounds` for the bounds of the AR coordinates, `ar_coordinates(t)`; every facet pair of ASU(Gamma+)
  glued by a pure translation is a circle of period 1, listed in `seams` (`TranslationSeam`); in sg 88, 91, 92, 95,
  96, 98, 122, 141, 142, whose seam translation is (1/2, +-1/2, 0), the AR coordinates are s = M t with M =
  `basis` (rows (1, +-1, 0), (1, -+1, 0), (0, 0, 1); ASU(Gamma+) a box in s) and the log-Jacobian includes
  -`log_det_basis` = -log 2, so it stays relative to d^k t; every other group keeps the table's AR coordinates,
  bit for bit; a `stages` override is read in the AR coordinates, so in the sheared groups it is refused unless
  `stages_in='s'`: build it from `PolytopeARChart(sg).stages`, not `group_table(sg).ar`), `OrientationChart(name)`,
  `CrystalChart` (their product on the +1
  sheet, every group; `domain`, `multiplicity`, `log_multiplicity`; `CrystalChart.for_length_box(sg, (lo, hi),
  orientation)` with an explicit box, `CrystalChart.for_latent_v1(sg, radius, orientation)` with the trainer's length
  box, used above monoclinic as a box only; sg 6-9 refused for want of an SO(3)/K chart unless
  `orientation_cover='double'`: the chart then covers ASU(Gamma+) x all of SO(3), both K images, `multiplicity` 2,
  `domain` `'double_cover'`, and a density on it must be K-invariant relative to the reference measure: log p_y -
  `log_reference_density` equal at y and at the chart point of `k_image(desc)` (in `euler_zyz` p_y itself is equal;
  in `hemisphere` and `axis_angle` the two differ by the Haar ratio); augment training data with `k_image(desc)`),
  `LatentV1Chart(sg, radius)` (`describe` / `latents` for any
  group, run through the live MolCrystalOps maps unchanged; `forward` / `inverse` with log-Jacobian for sg 1 and 2;
  sg >= 16 refused for good with `LATENT_ABOVE_MONOCLINIC`: the trainer's latent there is to be replaced, and no
  averaged-row Jacobian is built; the live clamp is returned as a mask), `latent_v1_length_box`, `from_unit`.
- `measure`: `log_reference_density(desc, chart, cell_measure)` (any `CrystalChart`; `'metric_invariant'` triclinic
  only), `log_multiplicity(sg, src, dst)`, `images_per_crystal` (domains `'fundamental'`, `'double_cover'` (sg 6-9:
  2), `'latent_v1_box'` (refused also in the orthorhombic groups with more than one axis order: the trainer fixes
  none, so its box holds a cell-dependent number of orders), `'unit_cell'`), `cell_volume`.
- `checks`: `same_crystal(a, b, mol)` (largest atom deviation, Angstrom; float64 literal build from the table's G),
  `orbit_check(sg, n)`, `round_trip(desc, path, mol=None, cell_rtol=1e-8)`.
- `io_mxt`: `from_mol_crystal_data(rows_or_batch, sheet=1, pose_tol=1e-2, special_tol=0.1, op_tol=1e-6,
  allow_zp_gt1=False, nonstandard='reexpress', mirror_reference=None, metric_tol=1e-6, off_sheet=None) ->
  (CrystalDescription, ImportInfo)`:
  MolCrystalData rows (as `dataset_utils/construction` writes them) as descriptions, the row's `pos` aligned by
  `align_mol_batch_to_standard_axes` (handedness +1, GFN's admission frame) as the molecule, the stored centroid,
  rotation and handedness as the pose; `to_sheet(sheet)` per unit. A row whose stored operators are not
  `SYM_OPS[sg]` as a set (order free) is re-expressed in the standard setting by the (P, p) of
  `settings.find_setting` (owner decision D1; `ImportInfo.setting_P`, `setting_p`), or refused as
  `unresolved_setting`; `nonstandard='refuse'` restores the refusal (`operator_count_mismatch`,
  `nonstandard_setting`). A re-expressed row whose cell the rotations of `SYM_OPS[sg]` do not preserve within
  `metric_tol` (`ImportInfo.metric_dev`) is refused as `setting_metric_mismatch`; no change of setting can resolve
  it. On new_prot_csd.pt this refuses 6 rows deposited in a- or c-unique monoclinic settings with beta, which
  those operators require to be 90 deg, off 90 deg: five by 11 to 40 times the CIF's standard uncertainty
  (KAMKOY01 gives none), one (XENJEJ, beta = 90.001(2) deg) within it, refused only because `metric_tol` = 1e-6
  is far tighter than CIF precision. Snapping such rows is an owner choice (open). Standard rows are measured, not
  refused, so the rule is asymmetric (XATDEF, sg 173, metric_dev 2.3e-5, a = b within its CIF's uncertainty, is
  imported). A Z' = 1 Sohncke row left off the sheet is
  re-described with the mirror image of its aligned molecule as the reference at h = +1 (owner decision D2,
  `mirror_reference`; `ImportInfo.mirrored`: the row's `MoleculeRef` is then the mirrored molecule, which
  `align_mol_batch_to_standard_axes` at handedness +1 maps back to the unmirrored one;
  `ImportInfo.reference_handedness`, -1 where mirrored, is the handedness that leaves it fixed; see Known limits).
  `off_sheet` (`OFF_SHEET_MODES`) names the treatment: `'mirror_reference'` (D2; also `mirror_reference=True`, and
  the default when neither is given), `'keep'` (`mirror_reference=False`: left off the sheet) or `'enantiomorph'`
  (2026-10-21): the row is replaced by `enantiomorph.enantiomorph` of it, its mirror-image crystal at `h' = -h` with
  the unmirrored reference X (`reference_handedness` +1, `mirrored` False), flagged `ImportInfo.enantiomorph`;
  `ImportInfo.sg_source` is the space group before the map (sg elsewhere). Its atoms are the D2 row's inverted
  through the origin and shifted by the origin shift -p (fractional x -> -x - p; `enantiomorph_map(sg).p`, nonzero
  only in sg 80, 98, 210), and the reference X = X' diag(-1, 1, 1) bit for bit. `mirror_reference` with an
  `off_sheet` it contradicts raises.
  Refused as well: missing, non-finite or degenerate fields,
  molecules on special positions, rows the description does not rebuild, Z' > 1 rows whose units hold different
  molecules, and, unless `allow_zp_gt1`, every Z' > 1 row (`z_prime_gt1`: the CSD featurizer copies unit 0's
  conformer into every unit, so units 1 and above are not the deposited molecules; on new_prot_csd.pt they differ
  from the CIF by a median 0.05 and up to 2.5 Angstrom); `REASONS` lists the codes in check order, `ImportInfo` the
  per-row outcome. `operator_set_match` (`op_tol` suits exact fractions; 4-decimal translations need 1e-4),
  `is_sohncke`.
- `settings`: `find_setting(ops, sg, tol=1e-6) -> SettingTransform(P, p) | None` (`x_stored = P x_std + p`, det
  P > 0: integer between conventional cells of one volume, det 1/2 from a centred P1 setting to a primitive cell,
  det 2 from a primitive setting of a centred group; a bounded search over integer K with entries in [-2, 2] and
  origins on the 1/24 grid, then, only where that finds nothing, the origin solved exactly as linear congruences
  and the stored basis first reduced under a metric its rotations preserve; cached per operator set; measured
  2026-10-17 with cctbx's 530 Hall settings: all resolve plain, with a random origin shift on the 1/24 grid, and
  through a unimodular basis change with an entry of 3, each checked exactly), `setting_matches` (the finite test that the re-expressed group
  is `SYM_OPS[sg]`: operator set modulo Z^3 together with the stored lattice, and the new lattice inside the stored
  group's translations), `transform_ops`, `operator_sets_equal`, `re_express(desc, P, p, wrap=True)` (`ops.change_basis`
  for any det > 0 P: cell from P^T G P, t' = P^-1 (t - p) wrapped, R' = Q R).
- `enantiomorph` (2026-10-21): `enantiomorph(desc)` maps Z' = 1 rows of Sohncke groups (space groups mixed; other
  groups ValueError, Z' > 1 NotImplementedError) to their mirror-image crystal, inversion through the origin written
  in the standard setting: cell kept bit for bit, `t' = (-t - p) mod 1` (strictly in [0, 1)), `R' = R diag(1, -1,
  -1)`, `h' = -h`, the reference molecule unchanged, sg' from `enantiomorph_map(sg) -> EnantiomorphMap(sg, sg_out,
  P, p, gamma_index)`. The map is derived from the tables at first use (cached): the inverted operators (W, -w)
  are `SYM_OPS[sg_out]` after `settings.find_setting`'s (P, p); sg_out = sg where Gamma has an improper element
  (43 groups; the map is then Gamma's element (-I, -p)), else the ITA enantiomorphic partner (`PARTNER`, the 22
  groups 76/78, 91/95, 92/96, 144/145, 151/153, 152/154, 169/170, 171/172, 178/179, 180/181, 212/213, checked
  against the table). Current tables: P = I everywhere, p = 0 except sg 80 (0, 1/2, 0), 98 (0, 1/2, 1/4), 210 (1/4,
  1/4, 1/4); a P != I would go through `settings.re_express`. Applied twice it is the identity (t to rounding).
  Apply it in the standard setting and canonicalise after it: it does not keep ASU(Gamma+).
- `dataset` (schema `crystal_fd.dataset/1`): `save_dataset(path, desc, mols, identifier, reference_handedness=None,
  status=None, status_names=('canonical',), columns=None, column_docs=None, manifest=None, provenance=None)` writes a
  folder (`meta.json`, `data.pt` readable with `torch.load(weights_only=True)`, optional `manifest.pt` and
  `manifest.csv` naming every source row once with a code) under a temporary name, then renames it; it refuses an
  existing path and checks every molecule hash, column length, status code and the manifest's accounting.
  `load_dataset(path, status='canonical', verify=True, verify_molecules=False, allow_table_mismatch=False,
  code='warn') -> FDDataset` (`desc`, packed molecules with `molecule(i)` returning the saved `MoleculeRef` bit for
  bit, `reference_handedness`, `identifier`, `status`, `columns`, `select`, `to_mol_crystal_data`) checks the sha256 of
  `data.pt` and refuses a dataset written against another `TABLE_HASH`; `load_manifest`. Code identity: `meta.json`
  records `code_digest()` (sha256 of every `*.py` and `data/*` file of this package, and one per file) at
  `save_dataset`, because canonical forms depend on code outside the table (e.g. `reduce_cell`'s tie-breaks);
  `load_dataset` issues a `CodeMismatchWarning` naming the changed files when the package differs or the dataset
  records no digest (`code='raise'` refuses, `'ignore'` is silent; `code_mismatch`, `check_code`). Files added
  since, with every recorded file unchanged, neither warn nor refuse (since 2026-10-20). `csd_fd_v1b` now reports
  `added: canon.py` and `changed: dataset.py`; the only change to `dataset.py` is that rule, so its rows are still
  this code's canonical forms (re-checked 2026-10-20: `canon.fd_status` finds every sg 1-15 row in the domain).
  Since 2026-10-21 (`added: axis_order.py`, `changed: settings.py`, docstring only) that holds for its sg 1-15 rows
  alone (bit for bit the same canonicalisation): rows of sg >= 16 in a dataset written before then were stored with
  the imported cell (`pose_only`, `cell_canonical` False), not snapped or put in axis order, so they are not this
  code's canonical forms until re-canonicalised. The digest names
  bytes, so an edited comment trips it too; a writer that canonicalised in another process must check the digest
  itself (the CSD build records it per shard and refuses to assemble shards of other code). `to_mol_crystal_data(desc,
  mols, reference_handedness=None, mode='body', identifier=None, clip='raise', extra=None)` writes MolCrystalData rows
  with `aunit_handedness = h * reference_handedness` (MolCrystalData's handedness is relative to the molecule's aligned
  frame) and the attribute `reference_handedness`: `mode='body'` (`pos = X diag(h, 1, 1)`) builds as described with
  `std_orientation=False` and, through the alignment, `True`; `mode='literal'` (the posed asymmetric unit, as the CSD
  builders write `pos`) with `std_orientation=True` only. Centroids are wrapped first; rows `in_clip_band` (a wrapped
  coordinate in (1 - `BUILDER_CLIP`, 1), `BUILDER_CLIP` = 1e-4, where the live builder clips) are refused unless
  `clip='allow'`.
- `certify`: `tiling_certificate(sg) -> TilingCertificate` (`ok`, `volume`, `n_elements`, `overlaps`, `contacts`,
  `facets` / `uncovered`, `defects`, `shift_min` / `shift_max`), `certify_tiling(ProjectedDomain, shifts=None)`,
  `projected_domain(sg, entry=None)`, `group_defects`, `wall_defects`, `facet_coverage`, `pair_intersections`,
  `is_bounded`, `convex_volume`, and for ASU(G) under G `g_asu_domain(sg, entry=None)` and `g_asu_certificate(sg)`;
  `python -m mxtaltools.crystal_fd.certify` prints one row per group (ASU(Gamma+)). The module
  docstring derives why the certificate, K trivial and the free action on SO(3) make ASU(Gamma+) x SO(3) a
  fundamental domain.

A typical canonicalisation of any batch:

```python
from mxtaltools.crystal_fd import canon
c = canon.canonicalize(desc)         # never raises for a row; c.names() per row, c.desc canonical where c.done
in_fd = canon.fd_status(desc).is_fd  # computed: the row equals its decided canonical form
```

The same for triclinic or monoclinic rows only, step by step (each step raises for the whole batch on a bad row):

```python
from mxtaltools.crystal_fd import domains, ops
red = ops.reduce_cell(desc)          # raises on rows with no real cell
canon = domains.fold(red.desc).desc  # raises on non-finite rows
special = red.n_auto > 1             # special metrics: see Known limits
```

## Gates and how to run them

From the MXtalTools root, CPU only:

```
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/fd -q                  # every gate, slow tier included (~27 min)
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/fd -q -m "not slow"    # fast lane (~5 min)
python tests/fd/bench_fd.py --check                                   # gate 7; exit 1 if a budget is missed
```

and the GFN boundary, from `gfn_diffusion/energy_sampling`:

```
python -m pytest -q tests/crystal/test_fd_measure_contract.py
```

Tests marked `cctbx` skip when cctbx is not importable; `slow` marks the 230-group sweeps outside a representative set
and the 1e5-per-side histogram tests (which are gate 2's acceptance; the fast tier is a smoke lane).

- **Gate 0** (`test_gate0_tables.py`, all 230 groups): every table fact re-derived in exact rationals (closure,
  brute-force N_E, K, volumes, facets and gluings, AR bounds, the trainer box); manifest, pinned digest, CRLF copy,
  loader refusals; no cctbx or GFN module loaded at run time. Under `cctbx`: Gamma, both ASUs and every tie-break tree
  equal cctbx's, and regeneration is byte-identical.
- **Gate 0, tiling certificate** (`test_gate0_tiling_certificate.py`, all 230 groups, exact, about 2 s): the
  representatives of Gamma+/K are a group modulo Z^d (closure and distinctness over all n^2 products);
  vol x |Gamma+/K| = 1; every facet of ASU(Gamma+) on the non-free coordinates is covered exactly, without overlap,
  by its face-to-face contacts with images k.P + s; no examined image overlaps it; the dropped free-axis cuts are
  height-1 walls. This proves the images tile the centroid space without relying on the shift range being
  complete, so the centroid part of gate 1 holds by proof, not sampling (sg 6-9 tile by Gamma+/K; their K cut stays
  gate 1's). The fast vertex enumeration is checked against brute force on every examined pair (slow outside a
  representative set), the shift range against a brute force one shift beyond it, and the certificate rejects a
  translated domain, moved facets, G's asymmetric unit, a duplicated element, a translation nudged off the group, a
  wrong domain examined at shift 0 only, and a wall of the wrong height. Not covered: the cut trees' boundary choices
  and the orientation rule (gates 0 under `cctbx` and 1).
- **Gate 1** (`test_gate1_orbits.py`): one representative per orbit for generic points, exact special points and
  points 3e-10 to 5e-8 off them, in every group; facet mutations detected; K cut; cell canonicalisation invariant off,
  near and exactly on every tri_niggli face, with tie-break and old-rule controls; special metrics flagged.
- **Gate 2** (`test_gate2_measure.py`): volumes, and KS / chi-square of canonicalised physical draws against the
  chart density, per space group, cell measure and orientation chart, with mutation controls.
- **Gate 3** (`test_gate3_round_trips.py`): chart round trips to 1e-12; atoms against the float64 literal oracle; the
  live float32 builder within 1e-3 Angstrom of the oracle (a transposed-R control fails); coordinate kinds against
  the gluings; the trainer latent in float32.
- **Gate 4** (`test_gate4_energy.py`; GFN side in the boundary test): every Gamma+ image, fold and reduced fold has one
  eLJ; wrong gluings do not.
- **Gate 5** (`test_gate5_jacobians.py`): closed-form log-Jacobians against autograd and central differences.
- **Gate 7** (`bench_fd.py`): fold, cell canonicaliser and chart budgets.
- **Z' > 1, stages Z0 and Z2** (`energy_sampling/docs/design/zprime_fd.md` sections 8-9): `test_zprime_orbits.py`
  (gate G1 on synthetic units with a sigma-odd stand-in conformer: the anchor by `in_asu`, every other unit in
  ASU(G) from `g_asu`; exactly f descriptions per labelling, f = 2, or 1 in the 22 enantiomorphic-pair groups, with
  controls that break the count); `test_zprime_g_asu_certificate.py` (gate G3: `g_asu_certificate` holds in all 230
  groups); `test_zprime_unit_fold.py` (`fold_unit`: one representative per unit orbit, the `mirrored` parity, the
  orbit's only image in ASU(G), idempotence, special and band points, literal atoms of a chain conformer with sigma
  applied). The 230-group sweeps outside a representative set are `slow`.
- **Gate 8** (`test_gate8_latent.py`; GFN side in the boundary test): `LatentV1Chart` is the live latent map bit for
  bit; `CellChart`'s image is the live penalty's zero set; the reference density is GFN's `-jacobian_energy / T` plus
  the constant the documented latent ranges predict.
- `test_io_mxt.py` (local data; skips without `D:/crystal_datasets` or `MXT_CSD_DATA`): on the three mini CSD files,
  imported rows against a description fitted to their stored atoms (`same_crystal`), against the live builder's unit
  cell and against the raw CIF's heavy atoms (every imported row with its CIF; each below 5e-3 Angstrom; measured
  maxima 3.2e-4, 1.8e-5, 5.0e-4); every refusal reason from a mutated row (non-finite operators; special positions
  at an inversion centre and on a C2/c 2-fold axis, not on a 2_1 axis); reordered operators import; sheets; the real
  Z' = 2 rows of mini_new_prot_csd.pt carry unit 0's conformer in unit 1 (0.014 Angstrom off the CIF), a synthetic
  Z' = 2 row is refused by default and imported with `allow_zp_gt1`; float32 cells on a face are flagged by
  `reduce_cell`; no cctbx or GFN module loaded. These tests pin the importer before D1 and D2
  (`nonstandard='refuse', mirror_reference=False`) where they test its refusals and sheets.
- `test_io_settings.py` (D1 and D2): `find_setting` is (I, 0) on the standard setting of all 230 groups; every
  stored operator set of a nonstandard row of new_prot_csd.pt (`CLASSES`, generated from the full-file run) is
  resolved, re-expresses to `SYM_OPS[sg]` as a set, independently of operator order; per class, a float64 synthetic
  crystal carried into the stored setting by its literal atoms is, built with the class's own operators, the
  standard crystal to 1e-9 Angstrom, `re_express` returns it (`same_crystal` 1e-9), and a moved origin or doubled
  basis vector is refused; centred triclinic settings halve the cell; a primitive setting of C2/c (det 2); settings
  the grid pass misses (rhombohedral axes with an origin off the hexagonal grid, det 3; origins off every grid;
  unreduced bases, a unimodular change with an entry of 3, in 19 groups) resolve to the same crystal, the congruence
  solver solves or refuses, and the later passes leave every first-pass result unchanged; several
  settings of one monoclinic crystal reduce and fold to one canonical form; refusal codes and parameters; local data:
  the mini file against the importer before D1/D2 row by row, re-expressed rows against their CIFs, real rows of
  every class (a fixture written by the full-file run, `MXT_FD_SETTINGS_ROWS`) against the stored atoms and the
  CIFs, mirrored-reference rows (same atoms, h = +1, on the sheet; `mirror_reference=False` leaves them off);
  GFN's frame step (`MolData`, collate, `orient_molecule(mode='std')`) at `reference_handedness` leaves every
  reference of the fixture and the mini file in place, and returns X for a mirrored reference without it.
- `test_enantiomorph.py` (`enantiomorph`, the importer's `off_sheet='enantiomorph'`): the map of all 65 Sohncke groups
  (partner exactly where Gamma has no improper element, P = I, the three origin shifts, (P, p) passing
  `setting_matches`, (-I, -p) in Gamma, read-only); per group, random float64 crystals: the output's unit cell (the
  table's exact G) is the inversion of the input's up to a translation, element-aware within 1e-9 Angstrom, and the
  matching translations are exactly -p plus the lattice and centring translations; h flipped alone, p dropped (80, 98,
  210) and the source sg kept (partner groups) are each more than 0.1 Angstrom off under every translation; twice is
  the identity; the outputs of h = -1 rows canonicalise on the +1 sheet; the strict wrap on tiny and signed-zero
  centroids; the `re_express` path equals the direct one; refusals, padded slots, empty batches. Local data
  (`mini_new_prot_csd.pt`): the enantiomorph mode against D2 row by row (X = X' diag(-1, 1, 1) bit for bit, the
  unit cell the inversion of the D2 row's up to a translation, every other row bit for bit), parameter consistency,
  sheets None and -1, Z' = 2 rows (`allow_zp_gt1`) never mapped; synthetic float32 rows of partner and origin-shift
  groups give the inversion of the source crystal; a synthetic Z' = 2 row of sg 19 off the sheet, in one batch with a
  Z' = 1 row, is neither mapped nor D2-mirrored while the Z' = 1 row is; the docs name the origin-shift groups the
  map derives. Wrapper (`tests/test_crystal_recipes.py`): columns, references, docs and record
  against the D2 product and the direct calls, save round trip, `near_face='nudge'` together with it, (CCDC) the
  smiles stereo flip on HECVAP (P4_3 -> P4_1) and IYOHIP, and on a local csd_v1 chunk store the stereo rule: smiles
  is the posed unit's where h_input is +1 and its mirror image's where -1 on every row; the posed unit carries the
  deposited stereo on every Sohncke row but not on every row of other groups (WAMYUI, P-1), so `smiles_deposited`
  is compared through h_input on Sohncke rows only. Since 2026-10-21 the wrapper's products carry the sg >= 16 cell
  step (status `canonical`, the `cell_snap` column, `reduce_N` a det +1 signed axis permutation), and the fixture's
  two P212121 rows, re-described by the importer in each other allowed axis order, return to one row.
- `test_fd_dataset.py` (`dataset`): save then load returns every field bit for bit (descriptions, molecules and
  hashes, reference handedness, status, columns, manifest) on synthetic canonical rows of six groups with mirrored
  references and a set-aside status; existing paths, Z' > 1, a molecule off its hash, bad columns, status codes and
  manifests that miss or repeat a row are refused and leave nothing behind; a changed byte of `data.pt` and another
  table hash are refused on load; the recorded code digest is checked (a changed or missing digest warns and names the
  files, `code='raise'` refuses, `'ignore'` is silent; README files do not count); the converter's rows, built by the
  live builder (`mol2ucell`), are within 1e-3
  Angstrom of the float64 literal unit cell in sg 1, 2, 4, 14, 19 and 61 (mode 'body' with `std_orientation` False and
  True, 'literal' with True; aligned and mirrored references; h = +1 and -1), while a mirrored reference converted
  without its `reference_handedness` and 'literal' built with `std_orientation=False` are each more than 0.1 Angstrom
  off; clip-band rows are refused; local data (`MXT_FD_CSD_DATASET` or `D:/crystal_datasets/csd_fd_v1b`, skipped when
  absent): the CSD dataset's manifest accounts for every source row, every row is on the +1 sheet with
  `reference_handedness` -1 exactly where mirrored, 2,000 seeded canonical rows are their own fold (sg <= 15: their own
  reduced cell, n_auto <= 1), 150 build within 1e-3 Angstrom, and its derived flags are consistent (`cell_auto_tol`
  exactly on set-aside rows, `near_face` on every row within the stored tolerance of a face, `axis_order_k` 6 / 3 / 1
  for P212121 / Pbca / Pna21 and 1 outside orthorhombic groups, `coincide_group` numbering); no cctbx or GFN module
  loaded.
- `test_canon.py` (`canon`): one call on a batch mixing space groups, Z', both sheets, special metrics, padded unit
  slots (NaN, zero R, h = 0) and every bad-row kind, sorted-out rows first, gives the stated status per row (an sg
  19 row off the sheet is `off_sheet`); rows not canonicalised come back bit for bit, padding is untouched,
  canonicalised rows equal the cell step (reduce_cell; for sg >= 16 the snap and an axis order chosen by a plain
  lexicographic minimum), to_sheet and fold run on their first unit alone; scrambled rows of 16 groups return to
  one form and a second pass moves no non-special row; `fd_status` against stated expectations (a Gamma+ image, a
  folded pose in an unreduced basis, sheets +1, -1 and None, an off-sheet Sohncke row, a special metric, Z' = 2, sg
  19 rows decided and another axis order of them not in the domain, float32 rows and float32 values passed as
  float64); a row each step raises on is isolated and every other row's description and records equal an unpatched
  run; other exceptions propagate; bad arguments raise; near_face bits; the sg 1-15 rows of a batch come out bit for
  bit the same with and without sg >= 16 rows beside them (five modes, float64 and float32).
- `test_higher_sg_cell.py` (the sg 16-230 cell step; representative groups in the fast lane, all 59 orthorhombic
  and all 156 higher groups `slow`): the axis-order classes are the survey's, each kept (N, s) passes
  `setting_matches`, every permutation left out is refused with every sign choice; every allowed order and sign
  choice of a random crystal, at a random Gamma+ image, canonicalises to one row within 1e-12, ordered by the
  class's rule, with N and s the cell step, and that row's literal unit cell (table G) is the input's up to a proper
  isometry and a translation within 1e-9 Angstrom (a quarter-shifted origin, flipped h or an order without its
  shift are > 0.1 off); mirror faces (exact and within `face_tol`) are `special_metric` with `n_auto` 2 or 6 (6 on a
  chain of near-ties too, against a brute-force subgroup count) and never nudged, just outside `face_tol` canonical and unflagged, and with `near_tol` > `face_tol` flagged and
  nudged; Pbca seams are canonical with `NEAR_CELL`, one row from every cyclic order, nudged by the shortest
  log-length step to fixed points whose float32 re-descriptions return, while a row within the tie tolerance across
  a float32 rounding boundary lands on another cyclic order unless nudged; the cone is `special_metric` (`n_auto`
  3); in sg 75-230 every Gamma image canonicalises to one row (improper images of Sohncke groups to one off-sheet
  row), a cell 1e-7 off is snapped to the exact metric with `cell_snap` its size and atoms within a few snap sizes;
  float32 re-descriptions of every class return within 1e-5; the new step isolates a failing row; sg 6-9 keep their
  statuses.
- `test_nudge.py` (`canonicalize(nudge=...)`): rows 1e-6 from (or on) triclinic cell-chart faces (tied candidates
  included), sg 14 and sg 4 (FREE a <= c) monoclinic faces, and centroid faces in sg 2, 14 (a 2-fold axis of
  Gamma+ included) and 19 (`canonical` since 2026-10-21, `pose_only` before) are moved by delta less their distance (cell, or t, the rest kept), sit
  delta from every face, have `near_face` 0 and are their own canonical form; float32 re-descriptions (Gamma+
  element, lattice shift, near-reduced basis for sg 1-15) of the moved rows come back within 1e-5, while the unmoved
  rows on a face come back as other rows; sheets -1 and None are kept; every row not moved (a batch with every
  status, special metrics among them, and `NEAR_FOLD_BAND` alone) is bit-identical to nudge=None, dtypes included; a
  second nudge moves nothing; `fd_status` takes it; domains thinner than 2 delta and a move that fails the
  fixed-point check keep the unmoved row with a message and zero move sizes; sg 19 and 33 rows at special positions
  canonicalise (the snap) and nudge to one crystal from float64, float32 and angles 1e-7 rad off 90 deg, with a pose
  that is an image of the row's, while a fold of those cells as given picks other poses; the shortest-move solver
  against hand solutions; bad nudge values raise. Orthorhombic cell faces: in `test_higher_sg_cell.py`. The limits
  above are pinned: an on-face row's re-descriptions nudge to more than one crystal, the `near_tol` edge, and a
  c/a = 40 cell.
- `test_monoclinic_cell.py` (gates 1, 3, 4, 5 and 8 for monoclinic cells; sg 14, 15, 4, 9, 5, 7 in the fast lane,
  the other seven groups `slow` where swept): the class parity is exactly `setting_shift`'s setting group; the
  conditions are the live penalty's zero set; canonical cell and pose one from every allowed basis within 1e-9 in
  the interior and at 1e-7, 1e-4 and 1e-2 on both sides of every face; on faces the candidates tie, every row is
  flagged `n_auto = 2` and some get two canonical poses; across the tie band (3e-11 to 1e-8 from every face) every
  row is flagged from every description, cell and pose are one, and the canonical cell is on the zero set and inside
  the chart's bounds; float32 cells on every face (beta = 90 deg in both settings a source may report) are flagged
  and `face_tol` < `tol` is refused; alpha and gamma within `angle_tol` are set to 90 deg before the pose is
  re-expressed; empty batches; `n_auto` against a brute-force count (square and
  hexagonal ac nets included); cell then pose composition with literal atom checks (a frame rotation withheld, a
  change outside the class and a withheld origin shift are each > 0.1 Angstrom off); the search span has a margin; a
  parity check switched off is refused; the monoclinic chart's round trip, Jacobian (autograd and central
  differences, at the kink of m) and zero-set equality (a 1% pushed face breaks it; sg 10 included, whose trainer
  box has lo_a > lo_c so the FREE a <= c bound on log c binds); eLJ of canonical forms with
  controls; no cctbx or GFN module loaded.
- `test_higher_sg_charts.py` (charts and measure for sg 16-230 and the sg 6-9 double cover; 19, 61, 18, 33, 92, 88,
  148, 169, 198 and 9 in the fast lane, every group 6-9 and 16-230 `slow`): `cell_family` is `LATTICE_TYPE`; round
  trips within 1e-10 with exact cells; every draw is its own `canonicalize` form (on the cover: it or its K image)
  and every canonical form charts inside the bounds exactly when its cell is in the box; face rounding within `SLACK`
  and `FAMILY_TOL` refusals; log|det| against autograd and central differences through the natural coordinates;
  the reference density against a closed form, and its Monte Carlo integral against vol(ASU) x the integral of V
  over the cell domain for every orthorhombic class and family (a chart with its faces removed fails); Pbca's gluing
  re-describes a seam cell onto the other seam as one crystal; the double cover's multiplicity, K images (in the
  cover, same crystal, same density in Euler coordinates, exactly one in the fundamental domain, half the draws);
  `LatentV1Chart` refusals; local data (`csd_fd_v4`, read only): every canonical and special-metric row of sg 6-9
  and 16-230 charts inside its bounds (cell coordinates exactly) and maps back within 1e-10. Translation seams (all
  230 groups): one circle per facet pair glued by a pure translation (read off `GroupTable.facets`) and no other,
  facet points and their translates chart to coordinates equal modulo the period with equal log-Jacobians, and only
  the nine [1, +-1, 0] groups are sheared; in those the seam's ends are one crystal with one reference density and
  the interval ends are not.
- `test_gate4_higher_sg.py` (`slow`): crystals drawn from the new charts (19, 92, 146, 169, 198, 205: one group per
  Gamma+ point-group type above monoclinic; 29, 33, 61, 62), built with `dataset.to_mol_crystal_data`: every Gamma+
  image, every allowed axis order and (sg 9) the K image has the source's eLJ (relative 1e-3 or 5e-3 raw; measured
  2026-10-21: at most 5.5e-3 raw, relative 3e-4, on |eLJ| up to 1.1e3), while R turned 90 deg about x (median |dE|
  5.8-67 raw per group) and an axis order without its origin shift do not.
- `test_core_ops.py`, `test_fixtures.py`: SO(3) charts, affine actions, basis changes, images, sheets, guards, and the
  fixtures' own oracle.

**Regenerating the table** needs cctbx: `python -m mxtaltools.crystal_fd.generate_fd_tables` rewrites
`data/fd_tables_v1.json` and `data/MANIFEST.json` and refuses to write if any check fails. Review the diff, then edit
`PINNED_TABLE_SHA256` in `test_gate0_tables.py` by hand.

## Stage-0 scope

- Pose: `fold`, `unfold`, `in_domain`, `in_asu`, `images`, `to_sheet`, `gauge_free_axes` for all 230 groups at Z' = 1.
- Cell: `reduce_cell` and `CellChart` for triclinic (1, 2) and, pulled forward from stage 1, monoclinic (3-15) groups;
  since 2026-10-21 the cell step of `canonicalize` for sg 16-230 (exact-metric snap, orthorhombic axis order) with
  the log-length face chart of `axis_order`, and `CellChart` for the orthorhombic, tetragonal, hexagonal and cubic
  families.
- Charts and measure: sg 1 and 2; `CrystalChart` also takes monoclinic groups with trivial K (3-5, 10-15) and, since
  2026-10-21, every group 16-230 and sg 6-9 as the double cover, with `'latent_v1_lebesgue'` only (gate 2's
  histograms have not been run there; the measure is checked by its closed-form integral). `PolytopeARChart` and
  `LatentV1Chart.describe` / `latents` work in any group; `LatentV1Chart`'s log-Jacobian and `metric_invariant` are
  built for triclinic groups only.

## Known limits

- **Z' > 1** raises `NotImplementedError` everywhere except `fold_unit`, which folds single units into ASU(G); the
  anchored canonicaliser, the per-unit conformer in `CrystalDescription` and the charts are not built.
- **Special metrics.** A triclinic lattice with a proper automorphism (for example beta = gamma = 90 deg exactly, as
  CIFs often report) has `CellReduction.n_auto > 1`: the reduced cell is unique, but the canonical pose depends on the
  input basis, so two settings of one crystal can give two canonical descriptions. Count or set aside these rows;
  resolving them (the automorphisms as extra identifications in the fold) is a later stage. Monoclinic cells count
  automorphisms within the setting class modulo the 2-fold about b (generic 1). Every face of the monoclinic domain
  is a mirror whose reflection lies in the setting class, so a cell within about `face_tol` of any face (beta = 90
  deg among them) is flagged `n_auto >= 2`. The flag's tolerance `face_tol` (default `ops.FACE_TOL` = 1e-5, on the
  tie-break key) is separate from the search's `tol` = 1e-10 and much wider: every row the search could leave to
  candidate order is flagged from every input basis, and so is a cell stored in float32, as the CSD datasets store
  cells (a 90 deg angle is 4.4e-8 rad off; measured: every float32 face cell is flagged from 1e-6). A flagged row
  further than `tol` from the special metric (monoclinic: further than rounding from a face) still has one canonical
  form, so the flag over-marks rather than under-marks. The value is a choice (owner decision open).
  The flag does not cover faces that are not lattice mirrors: a triclinic cell on a Niggli face such as b = c (with
  the angles not swapped by a lattice automorphism) has `n_auto` = 1 and `n_tied` = 2, and `reduce_cell` breaks the
  tie by its Gram key, which is a function of the lattice in exact arithmetic; but a re-description rounded to float32
  puts the cell a rounding off the face, where the first criterion decides, so it can canonicalise to the other cell
  (beta and gamma swapped). Centroids within rounding of a face of ASU(Gamma+) behave alike (measured on the CSD
  dataset, 2026-10-17: about 30 triclinic rows within 1e-6 of a face and a few centroids within 1e-7 of one; spglib's
  Niggli cell differs from `reduce_cell`'s on the 12 triclinic rows with b = c exactly). The CSD build flags such rows
  (`near_face`, tolerance `FACE_TOL`) rather than setting them aside; `canonicalize(nudge=delta)` moves them off the
  face instead (another crystal, about delta times a cell length away). A monoclinic cell within about `face_tol`
  of a face is a special metric (every face is a mirror of the setting class) and is not moved. The flag is on the
  key (A, C, G_ac) / mean(A, B, C), so its edge in chart units depends on the cell: at the beta = 90 deg face (s = 0)
  it is s = `face_tol` mean(A, B, C) / (2 m a c). Measured in sg 14 (2026-10-21): 5.8e-6 and 7.7e-6 at s = 0 (4.2e-6
  and 6.0e-6 at s = 1) for a, b, c = 6, 3, 9 and 6, 7, 9; 1.7e-4 at both faces for 4, 3, 40, where no row is
  `canonical` with `NEAR_CELL`.
- **What the nudge does not cover** (measured 2026-10-21 on synthetic rows and the 135 nudged `csd_fd_v2` rows; no
  `csd_fd_v2` row reaches the third or fourth case). (1) It moves the canonical image the input rounded to: a
  crystal within rounding of a face canonicalises to either side and each side is moved inward, so two descriptions
  of it can nudge to two crystals about 2 delta apart, each its own fixed point (sg 2 rows on the a = b and v = 0
  faces: 15 and 10 of 20 float32 re-descriptions; 31 of the 135 `csd_fd_v2` rows in some of 10 float32 re-imports
  each, and 2 more by case 2). (2) `near_tol` is an edge: a row within rounding of it is moved from one description
  and not another, a jump of delta - `near_tol`; CIF coordinates on the 1e-5 grid sit exactly on the default (a
  centroid 1e-5 + 1e-13 from a face is not moved, its float32 cast is, so `fd_status(x32, nudge=...)` calls the
  stored row not in the domain). (3) The cell step is in
  chart units, whose float32 rounding grows about as (c/a)^2 for triclinic cells: a nudged row's float32
  re-descriptions in span-1 bases come back as another row from c/a about 30 (0 of 200 at c/a 20, 30 of 200 at 40;
  canonical triclinic CSD rows have c/a below 15). `near_tol` has the same blind spot. (4) `near_face` has no bit for
  the margin of the orientation rule (the K cut of sg 6-9), so a row near that cut is neither flagged nor protected
  (smallest margin among the 5,521 tie-2 rows of `csd_fd_v2` off a face: 2.2e-4).
- **The orientation rule on cells off the group's metric.** On a float32 cell (90 deg is 4.4e-8 rad off) or float64
  angles within `metric_tol` of the group's, Q = T W T^-1 is that far from a signed permutation, above `fold`'s
  `tol`: at a special position of Gamma+ (fold tie 2) `fold` decides the orientation rule by rounding, so a float32
  cast of a row can fold to a pose differing by a stabiliser element (the same crystal; 5-7 of 8 random poses at
  several special positions of sg 19 and 33). Since 2026-10-21 `canonicalize` snaps sg >= 16 cells to the exact
  metric before the fold, so it (and the nudge) decide on the exact metric; `fold` called alone still does not.
- **The sg 16-230 cell step** (2026-10-21). Mirror faces of the axis order (a = b, and b = c where all six orders
  are allowed) within `face_tol` are special metrics, like monoclinic faces; their tolerance is on log lengths,
  not on the Gram key of sg 1-15, so at the defaults (`face_tol` = `near_tol` = 1e-5) every orthorhombic row within
  `near_tol` of a mirror face is a special metric and none is nudged off one. Pbca's seams (a = b, a = c) are not
  mirrors: a row on one has one canonical form, chosen by the tie-break within `cell_tol`, and float32 rounding is
  monotone, so a re-description flips to another cyclic order only from a row whose two shortest axes are within
  `cell_tol` across a float32 rounding boundary (constructed in the tests; such rows are `NEAR_CELL` and the nudge
  moves them). The snap is a mean over the group's rotations computed elementwise, row by row (independent of the
  batch), but not bit-idempotent: snapping a snapped cell can move a length by an ulp. Orthorhombic `n_auto` counts
  allowed axis orders only (a cubic-like orthorhombic metric a = b = c gives 6 in the six-order groups, 3 in Pbca, 1
  in one-order groups), as the subgroup the orders within `face_tol` generate: a six-order cell with b - a and
  c - b within `face_tol` and c - a not has 6 (as a single-linkage merge of the lengths would give), the same Pbca
  cell 1 (canonical, `NEAR_CELL`: off the cone, the 3-cycle moves it by more than `face_tol`); groups 75-230 have
  `n_auto` 1: the survey's setting enumeration (integer changes with entries
  up to 2) finds every change that keeps their group among Gamma's linear parts whatever the metric, so an extra
  metric symmetry such as c = a in a tetragonal cell gives no second setting. Not measured on the CSD set by this
  code: the survey's counts (68,753
  orthorhombic rows in groups with more than one order, 53,751 already smallest-first, about 380 refcode families
  holding one crystal in two axis orders) are from the read-only survey of 2026-10-21, not a re-canonicalisation.
- **Charts from orthorhombic up** (2026-10-21). The chart's measure is flat in each free log length by owner
  decision; it is not the trainer's (the trainer fixes no orthorhombic axis order, averages a = b (a = b = c cubic)
  from independent rows above that, and overwrites its angle rows), so log Z on these charts is not comparable with
  GFN runs until the trainer changes. The sg 6-9 double cover is exact only for a density K-invariant relative to the
  reference measure (Haar on R; a Lebesgue density is K-invariant itself only in `euler_zyz`): a model trained
  without the K images puts unequal mass on the two descriptions of a crystal. `unmap` reads a for b and c, and
  refuses cells beyond `FAMILY_TOL` = 1e-6 of the family metric rather than snapping them (canonicalise first).
  In the nine sheared centroid groups (88, 91, 92, 95, 96, 98, 122, 141, 142) the centroid coordinates are
  (x + y, x - y, z) or (x - y, x + y, z), not (x, y, z): a model or prior built on the old u of these groups does not
  carry over.
- **The tolerance band.** Face tests use `tol` (default 1e-9, fractional unit-normal distance). Within about `tol` of
  a facet, edge or vertex, plane tests within a tolerance are not equivariant: `in_asu` can count 0 or 2 images of an
  orbit, and `fold` may report `tie = 3` (no candidate passed the cut trees; it then returns the candidate the
  orientation rule picks among those within `tol` of the largest slack: an image of the input within about `tol` of
  ASU(Gamma+)). `fold` stays a function of the orbit, and `in_domain` counts exactly one image per orbit, except where
  a slack lies within rounding (about 1e-16) of exactly +-`tol`. Face distances are fractional, so in 24 trigonal and
  hexagonal groups (147-150, 155, 162-167, 174-177, 182, 187-194) a band is 0.63-1.58 times as wide on one side of a
  gluing as on the other.
- **Non-finite rows and cells with no real metric** are refused by `fold`, `in_domain` and `reduce_cell` (naming the
  rows). `LatentV1Chart.describe` returns them, as the live map does.
- **Euler ZYZ** stores u = cos beta: a matrix round trip is good to about eps / sin(beta), about 1e-8 at the poles.
  The other three charts round-trip to about 1e-15 everywhere.
- **Building crystals** from descriptions: `dataset.to_mol_crystal_data` (wraps centroids to [0, 1) and refuses the
  builder's clip band, since the live builder clips centroids to [0, 1 - 1e-4] rather than wrapping). The test helper
  `tests/fd/_helpers.crystal_batch` builds with `std_orientation=False` and writes h without baking it into `pos`, so
  it builds an h = -1 description as the h = +1 crystal (every gate test uses h = +1).
- **`same_crystal`** is literal: both descriptions must share space group and cell (relative `cell_rtol`) and it does
  not remove rigid motions. `MoleculeRef.hash` identifies stored coordinates exactly; match molecules from different
  pipelines by coordinates.
- **Imports**: from MolCrystalData (`io_mxt`; nonstandard settings re-expressed one way into the standard setting,
  nothing writes one), or from parameters (`CrystalDescription.from_crystal_params`). CIF and other formats are
  the next stage. The (P, p) the search picks fixes the re-expressed axis order of an orthorhombic row, as the
  depositor's choice does for standard rows; until 2026-10-21 nothing undid it, so two settings of one orthorhombic
  crystal need not have given one canonical form (measured 2026-10-17 by enumerating every valid (P, p) of the search
  family on sampled re-expressed rows: 48 of 114 rows with sg > 15 reached more than one canonical cell, while
  alternatives that keep the importer's cell fold identically, and triclinic and monoclinic rows reach one canonical
  form from every alternative). Since 2026-10-21 `canonicalize`'s axis order removes that choice (every allowed order
  of a random crystal of each of the 59 groups canonicalises to one row: `test_higher_sg_cell.py`); the 2026-10-17
  measurement was not repeated on the importer's rows. The importer's handedness is
  relative to the aligned molecule, which is the same point set for a molecule and its mirror image: a Sohncke row
  with stored h = -1 is a crystal of the other enantiomer of that reference; by default (D2) a Z' = 1 such row is
  re-described with the mirrored molecule as its reference at h = +1, so a consumer must take the row's
  `MoleculeRef`, not the stored `pos` (with `mirror_reference=False`, or at Z' > 1, it stays off the +1 sheet,
  flagged). The mirrored reference must not be re-aligned at handedness +1: `align_mol_batch_to_standard_axes` (and
  GFN's `orient_molecule(mode='std')` without `target_handedness`) maps a molecule and its mirror image to one point
  set, so re-aligning X' returns X (GFN's frame step on the 115 mirrored rows of the importer fixture: within 8.5e-5
  Angstrom of X, median 12 Angstrom from X'). Built at h = +1 with the row's (t, R), X puts each molecule's mirror image in its place: in general neither
  the stored crystal nor its enantiomorph, and usually a clashing one (eLJ of the builder with `std_orientation=True`
  against the stored row's: median 887 and 1,080 for 28 re-expressed and 90 standard sampled mirrored rows, up to
  7.6e3, 2026-10-17). Passing `ImportInfo.reference_handedness` as the alignment's handedness leaves every reference
  fixed (tested: 543 imported rows of the fixture and the mini file, within 1.8e-4 Angstrom), so a dataset of
  imported rows must store that flag and a consumer that re-aligns must pass it. With `off_sheet='enantiomorph'` such
  a row holds X at h = +1 instead, but as the mirror-image crystal of the deposited one, in the partner space group
  for the 22 paired groups (`sg_source` keeps the deposited group): its energy is the deposited crystal's only for a
  mirror-invariant energy, and its `smiles` (canonicalize wrapper) is the enantiomer of the deposited unit's for a
  chiral molecule.
  D2 mirrors achiral molecules too (owner position: no folding by molecular point symmetry): their X' is X up to a
  proper rotation and an atom permutation, so one compound in a Sohncke group can carry two `MoleculeRef` hashes
  (sampled mirrored rows, 2026-10-17: 7 of 562 within 0.05 Angstrom of achiral, a lower bound). The aligned molecule
  is a function of the stored coordinates and of the arithmetic that aligns them: for a molecule with
  near-degenerate principal moments, a near-zero third moment or coplanar heavy atoms the frame is decided by
  rounding (BAZGOY03 flips by 180 deg under 26 of 50 rotations; QQQFVD03, coplanar, comes out reflected, which also
  flips h, through GFN's `MolData`, collate, `orient_molecule` path on bitwise the same float32 `pos`), so a
  consumer of saved descriptions must keep the saved molecule and its `reference_handedness`, not re-derive them.
  Equal canonical descriptions are not a duplicate test (observation, 2026-10-17: about 8% of pairs of CSD
  determinations of one crystal differ by more than 1 Angstrom in canonical form, with or without re-expression;
  inferred causes: centroids near a glued face, cells within a fraction of a degree of a special metric but outside
  `face_tol`, aligned frames that flip for near-achiral molecules, and different polymorphs). Z' > 1 rows are refused
  by default (see `io_mxt`); centred settings of a primitive group (I1, B1 stored as sg 1) are re-expressed in a
  primitive cell of half the volume.
- **cctbx** is needed only to regenerate the table and for the `cctbx` tests; its availability on the cluster is
  unchecked.

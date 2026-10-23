Crystal Building
================

A molecular crystal is a
:class:`~mxtaltools.dataset_utils.data_classes.MolCrystalData` object, and a
batch of crystals is a collated one
(:func:`~mxtaltools.dataset_utils.utils.collate_data_list`). Construction is
implemented by the methods of
:class:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding`
on top of :mod:`mxtaltools.crystal_building.utils`; cell parameters are handled
by :class:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps`.
Both classes are mixed into ``MolCrystalData``. The methods work on batches;
most also accept a single crystal, which they collate into a one-graph batch.


Crystal Parameterisation
------------------------

A crystal is described by:

- **Asymmetric-unit molecules** -- atomic numbers ``z`` and Cartesian positions
  ``pos`` (Angstrom). A crystal with Z' > 1 is one graph holding ``z_prime``
  contiguous blocks of atoms of equal size, one per molecule, so every
  asymmetric-unit molecule must have the same number of atoms.
- **Symmetry** -- the space group number ``sg_ind``, the symmetry operators
  ``symmetry_operators`` (4x4 affine matrices acting on fractional
  coordinates), their number ``sym_mult``, and ``z_prime``.
  :meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.reset_sg_info`
  assigns a space group with its standard operators.
- **Cell parameters**, per crystal:

  - ``cell_lengths`` -- a, b, c in Angstrom;
  - ``cell_angles`` -- alpha, beta, gamma in radians;
  - ``aunit_centroid`` -- fractional (unit-cell) coordinates of the centroid of
    each asymmetric-unit molecule, three columns per molecule
    (``3 * max_z_prime`` columns);
  - ``aunit_orientation`` -- one rotation vector (axis times angle in radians)
    per asymmetric-unit molecule, applied to the molecule in its standard
    orientation (below), ``3 * max_z_prime`` columns;
  - ``aunit_handedness`` -- +1 or -1 per asymmetric-unit molecule, selecting
    the standard orientation (below).

:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.full_cell_parameters`
returns ``[cell_lengths | cell_angles | aunit_centroid | aunit_orientation]``,
of width ``6 + 6 * max_z_prime``.
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.set_cell_parameters`
assigns such a vector and, unless told to skip it, runs
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.box_analysis`,
which sets the fractional-to-Cartesian transform ``T_fc``, its inverse
``T_cf``, ``cell_volume``, ``packing_coeff`` and ``density``, and raises
``ValueError`` when ``mol_volume`` or ``mass`` is unset.

**Standard orientation.**
:func:`~mxtaltools.crystal_building.utils.align_mol_batch_to_standard_axes`
centres each molecule on its heavy-atom centroid and aligns its principal
axes, computed over heavy atoms with unit masses, so that the axis of smallest
moment maps to h times x, the middle one to y and the largest to z, where h is
the handedness. Where h differs from the handedness of the molecule's own
principal-axis frame, the alignment includes a reflection, so the molecule is
mirrored.

**Latent parameters.**
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.latent_params`
and
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.latent_to_cell_params`
map the cell parameters to and from a vector of the same width and block
order in which each modelled range spans [-1, 1]; the ranges are listed in
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.latent_transform`.
Cell lengths are scaled by the space group's asymmetric-unit extent and the
molecular radius, centroids are expressed in asymmetric-unit-box coordinates,
and rotation vectors are converted to spherical coordinates.
``latent_params`` modifies the crystal first: it reorders the asymmetric units
by the distance of their fractional centroids from the origin and, by default
(``gauge_fix_free_axes=True``), moves each centroid to the centre of the
asymmetric-unit box along every fractional axis that all the space group's
rotations leave unchanged
(:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.canonicalize_free_axes`).
This gauge fixing is skipped for the whole batch when ``max_z_prime > 1`` or
any crystal has ``z_prime > 1``.
``latent_to_cell_params`` clamps the latents, assigns the cell parameters and
then, by default, forces the cell lengths and angles to agree with the crystal
system.

**Recovering parameters from a structure.**
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.reparameterize_unit_cell`
works on a built unit cell. Through
:func:`~mxtaltools.crystal_building.utils.parameterize_crystal_batch` it picks,
per crystal, the molecule whose heavy-atom centroid lies in the space group's
asymmetric-unit box and returns that molecule's fractional centroid, rotation
vector and handedness. It also returns, per crystal, whether exactly one
molecule's centroid lies in the box (``is_well_defined``); otherwise the
molecule whose wrapped fractional centroid has the smallest norm is chosen
among those in the box, or among all molecules if none is. The handedness is 0
where the principal-axis frame is degenerate. For ``max_z_prime == 1`` the
method also overwrites ``pos`` with the chosen molecules; it assigns no cell
parameters itself.


Construction
------------

Periodicity is applied to whole molecules: a molecule's heavy-atom centroid is
wrapped into the unit cell and all its atoms are placed around it, whether or
not they fall inside the cell. All crystals of a batch are built together with
tensor operations.

1. **Pose the asymmetric unit.**
   :meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.pose_aunit`
   sets ``pos`` to ``R p + T_fc c``, where ``p`` is the molecule centred on its
   heavy-atom centroid and, with ``std_orientation=True`` (the default), put in
   the standard orientation for its ``aunit_handedness``; ``R`` is the
   rotation matrix of ``aunit_orientation``; and ``c`` is ``aunit_centroid``.
2. **Build the unit cell.**
   :meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.build_unit_cell`
   (:func:`~mxtaltools.crystal_building.utils.aunit2ucell`) applies each
   symmetry operator to ``aunit_centroid`` (clipped to [0, 1 - 1e-4]) and wraps
   the result into [0, 1), except that a component exactly equal to 1.0 is
   kept. It places a copy of the molecule, centred on its heavy-atom centroid
   and transformed by the operator's 3x3 part in fractional coordinates, at
   each resulting centroid; only the posed coordinates relative to the
   centroid are used, not the absolute position of ``pos``. It
   sets ``unit_cell_pos`` (Cartesian, ``num_atoms * sym_mult`` atoms per
   crystal, ordered by symmetry operator), ``unit_cell_batch`` and
   ``unit_cell_mol_ind``.
3. **Build a cluster.**
   :meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.build_cluster`
   (:func:`~mxtaltools.crystal_building.utils.ucell2cluster`) returns a new
   batch of unit-cell copies translated by lattice vectors. The cluster centre
   is the mean atom position of the posed asymmetric unit. A lattice
   translation is kept when one of the translated cell's molecules has its
   mean atom position within ``2 * radius + cutoff + 0.1`` of that centre
   (:func:`~mxtaltools.crystal_building.utils.get_cart_translations`). Every
   such translation is found, whatever basis the cell is written in, and the
   untranslated cell is always kept; ``supercell_size`` is ignored. A crystal
   keeps at most ``MAX_CELLS_PER_CRYSTAL`` cells, the nearest ones, which only
   a cell much smaller than its molecule reaches.
4. **Flag distant molecules.** Molecules whose mean atom position is farther
   than ``cutoff + 2 * radius + 0.1`` from the cluster centre stay in the
   cluster and are marked ``aux_ind == 2``.

:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.mol2ucell`
runs steps 1-2 on the crystal in place.
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.mol2cluster`
runs steps 1-4 on a copy and returns the cluster; this is the structure
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_analysis.MolCrystalAnalysis.analyze`
builds for computes that need a cluster, before
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_analysis.MolCrystalAnalysis.construct_radial_graph`
builds the pair list.


Crystals with Z' > 1
--------------------

When ``max_z_prime > 1``, both ``mol2ucell`` and ``mol2cluster`` first split
every crystal in the batch into one graph per asymmetric-unit molecule
(:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.split_to_zp1_batch`),
each carrying its own block of ``aunit_centroid``, ``aunit_orientation`` and
``aunit_handedness``. All these graphs are posed and built together, and the
results are recombined per crystal:
``mol2ucell`` after the unit cell
(:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.join_zp1_ucell_batch`),
``mol2cluster`` after the cluster
(:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.join_zp1_cluster_batch`).
For the cluster, each crystal's ``cutoff`` is widened by the largest Cartesian
distance between any two of its ``max_z_prime`` centroid slots, including the
padding slots of crystals with ``z_prime < max_z_prime``.


Cluster Indices
---------------

A cluster batch carries two per-atom indices:

- ``aux_ind`` -- 0 for molecule 0 of each cluster (the image of the asymmetric
  unit under the first symmetry operator at zero translation), 1 for the other
  molecules, and 2 for the molecules flagged in step 4. For Z' > 1 every
  asymmetric-unit molecule has ``aux_ind == 0``.
  :meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_analysis.MolCrystalAnalysis.construct_radial_graph`
  pairs, within its ``cutoff``, atoms of the same ``aux_ind == 0`` molecule
  (``edge_index``), and
  ``aux_ind == 0`` atoms with ``aux_ind == 1`` atoms and, for Z' > 1, with
  ``aux_ind == 0`` atoms of other molecules (``edge_index_inter``);
  ``aux_ind == 2`` atoms enter no pair.
- ``mol_ind`` -- the molecule each atom belongs to. When ``max_z_prime == 1``
  it counts the molecules of each cluster from 0. When ``max_z_prime > 1`` the
  sub-cluster indices are offset so that molecules of different sub-clusters
  differ, and the numbering does not restart for each crystal.

On a cluster batch, ``num_atoms`` still counts the asymmetric-unit atoms.
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_building.MolCrystalBuilding.de_cluster`
reduces a cluster to its ``aux_ind == 0`` atoms.
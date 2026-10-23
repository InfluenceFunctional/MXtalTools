Dataset Creation
================

.. warning::

    **Status: not a verified workflow.** The dataset-construction scripts are
    not covered by an end-to-end test, and this page was checked by reading
    them, not by running them. Only the per-chunk ``.cif`` featurisation step
    has a test (below).

    No longer current: ``process_cifs_to_dataset.py``,
    ``collate_and_generate_dataset.py``, ``process_xyz_to_dataset.py`` and
    ``process_GEOM.py``, named by earlier versions of this page, are not in the
    repository; there is no GEOM-specific
    script and no on-disk (lmdb) dataset code; and the Blind Test grouping in
    :class:`~mxtaltools.dataset_utils.dataset_manager.DataManager` is marked in
    its own code as probably non-functional.

    The scripts described below are in
    ``mxtaltools/dataset_utils/construction/``. Each sets its input and output
    paths in its ``if __name__ == '__main__':`` block and has no command-line
    interface.

Datasets are lists of :class:`~mxtaltools.dataset_utils.data_classes.MolData`
or :class:`~mxtaltools.dataset_utils.data_classes.MolCrystalData` objects saved
with ``torch.save`` and loaded whole into memory by
:meth:`~mxtaltools.dataset_utils.dataset_manager.DataManager.load_training_dataset`.
The graph models featurise atoms from their atomic numbers when they run
(:meth:`~mxtaltools.models.graph_models.base_graph_model.BaseGraphModel.featurize_input_graph`),
optionally adding ``num_atoms``, ``radius`` or ``mol_volume`` as molecule
features.


Crystal Datasets from .cif Files
--------------------------------

These scripts need the CSD Python API (see :doc:`installation`).

1. ``dump_CSD.py`` searches the CSD by citation year (1921-2029) and writes
   every entry that has 3D coordinates, is not polymeric, has no disorder and
   has at least one atom as a ``.cif`` file.
2. ``featurize_cif_chunks.py`` (``process_cifs_to_chunks``) splits a directory
   of ``.cif`` files into chunks, processes the chunks in random order and saves
   the featurised crystals of each chunk with ``torch.save``. Structures are
   rejected by ``crystal_filter`` in ``featurization_utils.py``, including
   disordered and polymeric structures, missing atom sites, non-integer Z' or
   Z' above ``max_z_prime``, molecular components that do not match Z',
   invalid space-group information, failed packing, a unit cell with the wrong
   number of molecules or with symmetry images of unequal size, and molecules
   RDKit rejects. The existence check on each chunk's output file is disabled,
   so every run processes every chunk and parallel runs repeat each other's
   work. ``tests/test_crystal_workflow.py`` (pytest marker ``ccdc``; skipped
   without a CCDC licence and a local CIF corpus) runs ``process_chunk`` on 20
   CIF files and checks that crystals survive the read, build and rebuild
   path.
3. ``collate_dataset_chunks.py`` collates the chunks with
   :meth:`DataManager.process_new_dataset <mxtaltools.dataset_utils.dataset_manager.DataManager.process_new_dataset>`,
   which writes ``<name>.pt`` (the whole dataset), ``test_<name>.pt`` (up to
   the first 10000 samples, shuffled), ``mini_<name>.pt`` (up to the first 100,
   shuffled) and ``misc_data_for_<name>.npy`` (dataset statistics) to the
   datasets directory.

A CCDC-free CIF reader,
:func:`mxtaltools.dataset_utils.construction.cif.reader.read_cif`, is tested
in ``tests/cif/``; the scripts above do not use it.


Molecule Datasets from .xyz Files
---------------------------------

``featurize_xyzs.py`` (``process_xyzs_to_chunks``) reads every file in a
directory of ``.xyz`` files into
:class:`~mxtaltools.dataset_utils.data_classes.MolData` objects, saved in
chunks. When the chunk directory's path contains ``qm9`` (any case), it also
parses the QM9 properties from each file into ``y``. The atom parser
(``featurize_xyz_molecule`` in ``featurization_utils.py``) expects the QM9
file layout whatever the path: tab-separated atom lines and a SMILES string on
the third-to-last line. Files that raise ``ValueError`` are skipped, and a
chunk whose output file already exists is not reprocessed. The output file
name uses a module-level ``chunk_prefix`` that is defined only in the script's
``__main__`` block, so calling ``process_xyzs_to_chunks`` from other code
raises ``NameError``. The chunks are then collated with
``collate_dataset_chunks.py`` as above.


Filtering at Load Time
----------------------

Duplicate and polymorph filtering happens when a dataset is loaded for
training, in
:meth:`~mxtaltools.dataset_utils.dataset_manager.DataManager.load_dataset_for_modelling`,
not when it is built. The training entry point reads ``filter_conditions``,
``filter_polymorphs`` and ``filter_duplicate_molecules`` from the ``dataset``
section of the run config, which is filled from a dataset config in
``configs/dataset/`` (see :doc:`model_training`).

- ``filter_polymorphs`` groups samples whose identifiers are equal after the
  last two characters are dropped from identifiers ending in a digit (the CSD
  refcode suffix), and keeps one sample of each group at random.
- ``filter_duplicate_molecules`` groups samples whose stored molecular
  ``fingerprint`` (its first 2048 entries) is identical and keeps one of each
  group at random.
- ``filter_conditions`` is a list of ``[key, type, values]`` conditions
  (:meth:`~mxtaltools.dataset_utils.dataset_manager.DataManager.compute_filter`).
  The key ``atomic_number`` keeps samples whose atomic numbers all lie in
  ``values``; for other keys, type ``range`` keeps values in ``[min, max]``
  inclusive and type ``in`` keeps values equal to one of ``values``. The
  ``not_in`` branch does not assign its result, so it fails with
  ``UnboundLocalError``.


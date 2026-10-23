Examples
========

Example scripts are in the ``examples/`` directory of the repository. Each
loads a pre-trained checkpoint from ``checkpoints/`` or a small dataset from
``mini_datasets/`` through a relative path (``../checkpoints/...``), so it is
run from inside ``examples/`` in a clone (see :doc:`installation`).

.. warning::

    The example scripts are not covered by tests and were not run for this
    revision. Read against the current code:

    - ``crystal_density_prediction.py`` builds molecules with
      :meth:`~mxtaltools.dataset_utils.data_class_methods.mol_methods.MolDataMethods.from_smiles`
      without ``do_mol_analysis=True``, so ``mol_volume`` and ``mass`` are not
      set and ``mol_batch.mol_volume / packing_coeff_pred`` raises
      ``TypeError``. Its density factor
      1.6654 differs from the 1.66054 (amu per cubic Angstrom to g/cm\ :sup:`3`)
      used by
      :meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.box_analysis`.
    - ``molecule_autoencoder.py`` centres each molecule on its heavy-atom
      centroid (the
      :meth:`~mxtaltools.dataset_utils.data_class_methods.mol_methods.MolDataMethods.recenter_molecules`
      default), while ``Mo3ENet.encode`` asserts that the mean of all atom
      positions in the batch, hydrogens included, lies within 1e-3 of the
      origin.
    - ``crystal_analysis.py`` calls ``torch.load`` without
      ``weights_only=False``. From PyTorch 2.6 the default is
      ``weights_only=True``, which does not load pickled ``MolCrystalData``
      objects.


Crystal Density Prediction
--------------------------

Predict the crystal packing coefficient of molecules built from SMILES with a
pre-trained regressor (``checkpoints/cp_regressor.pt``, loaded by
:func:`~mxtaltools.common.training_utils.load_molecule_scalar_regressor`). The
model output is de-standardised with the model's ``target_std`` and
``target_mean`` buffers.

The packing coefficient is computed by
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.box_analysis`
as :math:`c_p = V_{mol} \, n_{sym} / V_{cell}`, where :math:`V_{mol}` is the
volume of the asymmetric-unit contents (``mol_volume``), :math:`n_{sym}` the
number of symmetry operators (``sym_mult``) and :math:`V_{cell}` the unit-cell
volume (``cell_volume``); equivalently :math:`c_p = V_{mol} / V_{aunit}` with
:math:`V_{aunit} = V_{cell} / n_{sym}`.

The script estimates uncertainty by Monte Carlo dropout:
:func:`~mxtaltools.common.training_utils.enable_dropout` puts the model's
dropout layers in training mode, and the mean and standard deviation of 50
predictions are taken.

Training data: not recorded in the repository or in the checkpoint's stored
config. That the model was trained on CSD data is unverified.

See ``examples/crystal_density_prediction.py``:

.. literalinclude:: ../../examples/crystal_density_prediction.py
    :language: python
    :start-after: if __name__ == '__main__':


Molecule Encoding
-----------------

Encode molecules into vector and scalar embeddings with a pre-trained Mo3ENet
autoencoder (``checkpoints/autoencoder.pt``, loaded by
:func:`~mxtaltools.common.training_utils.load_molecule_autoencoder`).
``model.encode`` returns the vector embedding and ``model.scalarizer`` maps it
to the scalar one; ``model.check_embedding_quality`` encodes and decodes the
batch and returns the graph reconstruction loss, the RMSD and the per-graph
``complete_graph_bools`` of the reconstruction. The atom types
the model accepts are those listed in the checkpoint's
``dataDims['allowed_atom_types']``.

Unverified: earlier documentation stated that the model was trained on
QM9-like molecules (up to 9 heavy atoms of H, C, N, O and F) and that
performance may degrade for fluorine-rich or highly symmetric molecules.

See ``examples/molecule_autoencoder.py``:

.. literalinclude:: ../../examples/molecule_autoencoder.py
    :language: python
    :start-after: if __name__ == '__main__':


Crystal Analysis & Scoring
--------------------------

The script loads ten crystals from ``mini_datasets/mini_CSD_dataset.pt`` and
makes a copy in randomly chosen space groups
(:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.reset_sg_info`)
with random cell parameters
(:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_ops.MolCrystalOps.sample_random_crystal_parameters`).
For both sets it builds a cluster and computes Lennard-Jones and electrostatic
energies with
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_analysis.MolCrystalAnalysis.build_and_analyze`,
the older fixed-request form of
:meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_analysis.MolCrystalAnalysis.analyze`,
and evaluates the crystal score model (``checkpoints/crystal_score.pt``, loaded
by :func:`~mxtaltools.common.training_utils.load_crystal_score_model`) on the
cluster. The model has three outputs; the script reads the first two as class
logits (:func:`~mxtaltools.models.utils.softmax_and_score`) and the third,
through a softplus, as a predicted RDF distance. See :doc:`crystal_building`
for how the cluster is built.

See ``examples/crystal_analysis.py``:

.. literalinclude:: ../../examples/crystal_analysis.py
    :language: python
    :start-after: if __name__ == '__main__':

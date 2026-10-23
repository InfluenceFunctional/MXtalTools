.. _about:

About
=====

MXtalTools is a Python library for machine learning on molecules and molecular crystals, built on `PyTorch <https://pytorch.org/>`_ and `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/>`_.

The library provides:

- **Crystal data objects** -- :class:`~mxtaltools.dataset_utils.data_classes.MolData` and :class:`~mxtaltools.dataset_utils.data_classes.MolCrystalData`, PyTorch Geometric data objects for molecules and molecular crystals, collated into batches with :func:`~mxtaltools.dataset_utils.utils.collate_data_list`.
- **Crystal building** -- Batched construction of the unit cell and of a finite cluster of periodic images from cell parameters and space-group symmetry operators, written in PyTorch so that gradients flow from the built coordinates to the cell parameters. See :doc:`crystal_building`.
- **Crystal analysis** -- :meth:`MolCrystalData.analyze <mxtaltools.dataset_utils.data_class_methods.crystal_analysis.MolCrystalAnalysis.analyze>` builds the structure a request needs and evaluates energies and other analyses on it, including Lennard-Jones-type potentials and the MACE and UMA interatomic potentials (see :doc:`installation` for their packages).
- **Crystal density prediction** -- Predict the crystal packing coefficient of a molecule with a pre-trained graph neural network (``checkpoints/cp_regressor.pt``, loaded by :func:`~mxtaltools.common.training_utils.load_molecule_scalar_regressor`).
- **Molecule autoencoder** -- Encode molecules into vector and scalar embeddings with a pre-trained Mo3ENet model (``checkpoints/autoencoder.pt``, loaded by :func:`~mxtaltools.common.training_utils.load_molecule_autoencoder`).
- **Crystal scoring** -- A pre-trained crystal score model (``checkpoints/crystal_score.pt``, loaded by :func:`~mxtaltools.common.training_utils.load_crystal_score_model`) with three outputs, which the example script reads as two class logits and a predicted RDF distance.
- **Crystal structure search** -- Optimise crystal parameters with torch autograd against a selectable target, including Lennard-Jones-type potentials and the MACE and UMA interatomic potentials (:mod:`mxtaltools.crystal_search.run_search`).
- **Dataset construction** -- Scripts that build molecule and crystal datasets from ``.xyz`` and ``.cif`` files; ``.cif`` processing uses the CSD Python API. See :doc:`dataset_creation`.
- **Model training** -- A config-driven training entry point, ``main.py``, for the graph neural network models. See :doc:`model_training`.


Reference
---------

If you use MXtalTools in a publication, please cite:

.. code-block:: bibtex

    @article{kilgour2026mxtaltools,
      title={MXtalTools: A Toolkit for Machine Learning on Molecular Crystals},
      author={Kilgour, Michael and Tuckerman, Mark E and Rogal, Jutta},
      journal={Journal of Chemical Information and Modeling},
      year={2026},
      publisher={American Chemical Society}
    }
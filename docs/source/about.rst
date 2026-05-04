.. _about:

About
=====

MXtalTools is a Python library for machine learning on molecules and molecular crystals, built on `PyTorch <https://pytorch.org/>`_ and `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/>`_.

The library provides:

- **Crystal building** -- Fast, differentiable construction of molecular crystal supercells from asymmetric unit parameters and space group symmetry.
- **Crystal density prediction** -- Predict crystal packing coefficients from molecular structure using a pre-trained graph neural network.
- **Molecule autoencoder** -- Encode molecules into equivariant vector and scalar representations using a pre-trained Mo3ENet model.
- **Crystal scoring** -- Evaluate crystal structures against CSD statistics using a trained classifier.
- **Crystal structure search** -- Optimize crystal packing parameters using machine-learned interatomic potentials and scoring models.
- **Dataset utilities** -- Tools for constructing molecular and crystal datasets from CSD, ``.cif``, and ``.xyz`` files.
- **Model training** -- Configurable training workflows for graph neural networks on molecular crystal tasks.


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
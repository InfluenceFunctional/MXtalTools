Examples
================

.. default-role:: math

MXtalTools has several analysis utilities for various molecular crystal tasks.

Crystal Density Prediction
------------------------------
Our model is trained on CSD data to predict the crystal packing coefficient, `c_{p = \frac{V_{mol}}{V_{aunit}}`, with `V_{mol}` the molecule volume, as estimated by our built-in utility, and `V_{aunit}` the volume of a crystal asymmetric unit, `V_{aunit}=\frac{V_{cell}}{Z}`.

While we achieve good accuracy and have some ability to estimate the prediction error via dropout variance, the model may not perform well for molecules unlike those in the CSD.

See here a working example you can run in examples/crystal_density_prediction.py.

.. literalinclude:: ../../examples/crystal_density_prediction.py
    :language: python
    :start-after: if __name__ == '__main__':



Molecule Encoding
------------------------------

This module uses a pretrained Mo3ENet to convert a molecule point cloud into equivariant vector and scalar representations.
To-date, Mo3ENet has been trained on QM9-like molecules, that is, molecules with 9-or-less heavy atoms and containing H, C, N, O, and F.
We tend to observe poorer performance on fluorine-rich or highly symmetric molecules.
We include in this example a utility to check the fidelity of each molecule representation.

For a runnable example, see examples/molecule_autoencoder.py.

.. literalinclude:: ../../examples/molecule_autoencoder.py
    :language: python
    :start-after: if __name__ == '__main__':



Crystal Analysis & Scoring
------------------------------

This module combines many utilities for the construction and analysis of molecular crystal data, including a score model trained on CSD data and applicable to molecules within the "CSD distribution".

For a runnable example, see examples/crystal_analysis.py

.. literalinclude:: ../../examples/crystal_analysis.py
    :language: python
    :start-after: if __name__ == '__main__':

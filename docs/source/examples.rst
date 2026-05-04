Examples
========

MXtalTools includes several analysis utilities for molecular crystal tasks.
Runnable example scripts are in the ``examples/`` directory.


Crystal Density Prediction
--------------------------

Predict the crystal packing coefficient from molecular SMILES using a pre-trained regressor.
The packing coefficient is defined as :math:`c_p = V_{mol} / V_{aunit}`, where :math:`V_{mol}` is the molecule volume and :math:`V_{aunit} = V_{cell} / Z` is the asymmetric unit volume.

The model is trained on CSD data and achieves good accuracy for CSD-like molecules.
Uncertainty can be estimated via dropout-based MC sampling.

See ``examples/crystal_density_prediction.py``:

.. literalinclude:: ../../examples/crystal_density_prediction.py
    :language: python
    :start-after: if __name__ == '__main__':


Molecule Encoding
-----------------

Encode molecules into equivariant vector and scalar representations using a pre-trained Mo3ENet autoencoder.
The model has been trained on QM9-like molecules (up to 9 heavy atoms, containing H, C, N, O, F).
Performance may degrade for fluorine-rich or highly symmetric molecules.

See ``examples/molecule_autoencoder.py``:

.. literalinclude:: ../../examples/molecule_autoencoder.py
    :language: python
    :start-after: if __name__ == '__main__':


Crystal Analysis & Scoring
--------------------------

Build and analyze molecular crystals, including Lennard-Jones potential evaluation,
radial distribution functions, and a CSD-trained crystal score model.

See ``examples/crystal_analysis.py``:

.. literalinclude:: ../../examples/crystal_analysis.py
    :language: python
    :start-after: if __name__ == '__main__':

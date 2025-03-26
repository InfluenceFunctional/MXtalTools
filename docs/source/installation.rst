.. _installation:

Installation Instructions
=========================

1. Download the code from this repository via

.. code-block:: bash

 git clone git@github.com:InfluenceFunctional/MXtalTools.git MXtalTools

Then, checkout the currently active branch

.. code-block:: bash

 git checkout equivariant_implement

2. Create and install pip environment. Note, you may need to adjust pytorch/pyg CUDA versions according to your local parameters.

.. code-block:: bash

 install_requirements.sh

3. If planning to do dataset construction, optionally also install rdkit (for single-molecule featurization) via

.. code-block:: bash

 pip install rdkit

and the Cambridge Structural Database Python API (for crystal featurization), which requires a separate license.

4. If you plan to do model training, or use our analysis tools, login to your weights and biases ("wandb") account, which is necessary for run monitoring and reporting in this repository. If just using functions or utilities from the package, this is probably unnecessary.

.. code-block:: bash

 wandb login

5. In configs/users create a *.yaml* file for yourself and edit the paths and wandb details to correspond to your preferences. When running the code, append the following to your command line prompt.

 python main.py --user YOUR_USERNAME

Installation Instructions
=========================

1. Download the code from this repository via

.. code-block:: bash

 git clone git@github.com:InfluenceFunctional/MXtalTools.git MXtalTools

2. Create and install pip environment. Note, you may need to adjust pytorch/pyg CUDA versions according to your local parameters.

.. code-block:: bash

 install_requirements.sh


3. If you plan to do any model training, or use our built-in analysis tools, login to your weights and biases ("wandb") account, which is necessary for run monitoring and reporting in this repository.
If just using tools, this is probably unnecessary.

.. code-block:: bash

 wandb login

4. In configs/users create a *.yaml* file for yourself and edit the paths and wandb details to correspond to your preferences.
When running the code, append the following to your command line prompt.
    --user YOUR_USERNAME


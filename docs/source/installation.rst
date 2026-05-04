.. _installation:

Installation
============

MXtalTools has two classes of dependency:

- **PyTorch + PyG** — must be installed manually because the correct wheels
  depend on your CUDA version.
- **Everything else** — handled automatically by ``pip install mxtaltools``.


Step 1 — Install PyTorch and PyTorch Geometric
-----------------------------------------------

Choose the commands for your CUDA version from the official guides:

- `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_
- `PyG installation guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_

For example, with **CUDA 11.8 and PyTorch 2.4**:

.. code-block:: bash

    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

    pip install torch-geometric==2.7.0


Step 2 — Install MXtalTools
-----------------------------

**Users:**

.. code-block:: bash

    pip install mxtaltools

**Developers** (editable install from a clone):

.. code-block:: bash

    git clone git@github.com:InfluenceFunctional/MXtalTools.git
    cd MXtalTools
    pip install -e .


Optional Dependencies
---------------------

**UMA model support** (``fairchem-core``) — only needed if you want to use the
UMA machine-learning interatomic potential:

.. code-block:: bash

    pip install mxtaltools[uma]
    # or, separately:
    pip install fairchem-core

**CSD Python API** — only needed for constructing crystal datasets from
``.cif`` files. Requires a valid licence from
`CCDC <https://www.ccdc.cam.ac.uk/>`_. Install separately following the CCDC
instructions.

**Weights & Biases** — required for model training. Already included in the
base install; activate with:

.. code-block:: bash

    wandb login

**User config** — for model training, create ``configs/users/YOUR_USERNAME.yaml``
with your paths and W&B settings, then pass ``--user YOUR_USERNAME`` on the
command line.

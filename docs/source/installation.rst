.. _installation:

Installation
============

The package metadata is ``pyproject.toml`` (built with Poetry; Python >= 3.9).
It deliberately does not declare PyTorch, PyTorch Geometric (PyG) or PyG's
compiled extensions, because the right wheels depend on your CUDA version:
install these first, by hand. The other required dependencies are declared
there and installed by ``pip``.

.. warning::

    These instructions were checked against ``pyproject.toml`` and the import
    statements of the current code, not by a tested installation. The
    repository pins no PyTorch or PyG version.


Step 1 -- Install PyTorch and PyTorch Geometric
-----------------------------------------------

Install ``torch`` for your CUDA version, then ``torch_geometric`` and the three
compiled extensions the library imports: ``torch_scatter`` (used throughout),
``torch_sparse`` (imported by :mod:`mxtaltools.dataset_utils.data_classes`) and
``torch_cluster`` (imported by :mod:`mxtaltools.models.functions.radial_graph`).
The library does not import ``pyg_lib`` or ``torch_spline_conv``.

- `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_
- `PyG installation guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_

Following the PyG guide, with ``${TORCH}`` the installed PyTorch version (for
example ``2.4.0``) and ``${CUDA}`` its build tag (for example ``cu118`` or
``cpu``):

.. code-block:: bash

    pip install torch_geometric
    pip install torch_scatter torch_sparse torch_cluster \
        -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html


Step 2 -- Install MXtalTools
----------------------------

From a clone of the repository (editable install):

.. code-block:: bash

    git clone https://github.com/InfluenceFunctional/MXtalTools.git
    cd MXtalTools
    pip install -e .

A released package can also be installed from PyPI, where ``update_dist.sh``
uploads it (``poetry build``, then ``twine upload``) under the name
``mxtaltools``. The repository does not record which commit a published
release was built from.

.. code-block:: bash

    pip install mxtaltools

The package excludes every ``*.pt`` file (``exclude`` in ``pyproject.toml``),
so the pre-trained checkpoints (``checkpoints/*.pt``) and small datasets
(``mini_datasets/*.pt``) used by the :doc:`examples` exist only in a clone;
``examples/`` is not in the package ``include`` list either.


Step 3 -- Install the MACE and UMA packages
-------------------------------------------

.. warning::

    MACE and UMA are selectable energy backends, but both packages are
    currently needed to import the core data classes:
    :mod:`mxtaltools.dataset_utils.data_classes` imports
    :mod:`mxtaltools.dataset_utils.data_class_methods.crystal_analysis`, which
    imports :mod:`mxtaltools.mlip_interfaces.AL_mace_utils` (which imports
    ``mace``) and :mod:`mxtaltools.mlip_interfaces.uma_utils` (which imports
    ``fairchem``) at module level. If either package is missing,
    ``from mxtaltools.dataset_utils.data_classes import MolData`` raises
    ``ModuleNotFoundError``.

- **MACE** -- package ``mace-torch`` (imported as ``mace``), not declared in
  ``pyproject.toml``: ``pip install mace-torch``. It serves the ``'mace'``,
  ``'mace_pot'`` and ``'mace_gas_pot'`` computes of
  :meth:`~mxtaltools.dataset_utils.data_class_methods.crystal_analysis.MolCrystalAnalysis.analyze`.
  The model file is supplied by the caller and loaded by
  :func:`~mxtaltools.mlip_interfaces.AL_mace_utils.load_mace_model`; none is in
  the repository. When ``cuequivariance``, ``cuequivariance_torch`` and
  ``cuequivariance_ops_torch`` are importable and CUDA is available,
  ``load_mace_model`` converts the model to cuEquivariance.
- **UMA** -- package ``fairchem-core``, declared as the ``uma`` extra:
  ``pip install "mxtaltools[uma]"``, ``pip install -e ".[uma]"`` from a clone,
  or ``pip install fairchem-core``. It serves the ``'uma'``, ``'uma_pot'`` and
  ``'uma_gas_pot'`` computes. The checkpoint is supplied by the caller and
  passed to fairchem's ``pretrained_mlip.load_predict_unit`` by
  :func:`~mxtaltools.mlip_interfaces.uma_utils.init_uma_crystal_predictor`.


Optional Dependencies
---------------------

**CSD Python API** (``ccdc``) -- requires a licence from
`CCDC <https://www.ccdc.cam.ac.uk/>`_; install it following the CCDC
instructions. It is not declared in ``pyproject.toml``. Within the package it
is imported only by the ``.cif`` dataset-construction scripts (``dump_CSD.py``,
``featurize_cif_chunks.py`` and ``cifs_to_molfiles.py`` in
``mxtaltools/dataset_utils/construction/``) and, when called, by
:func:`~mxtaltools.dataset_utils.data_class_methods.crystal_analysis.single_compack_run`
(COMPACK packing similarity). Tests that need it carry the ``ccdc`` pytest
marker.

**OVITO** (``ovito``) -- imported only by ``mxtaltools/common/ovito_utils.py``.

**Weights & Biases** -- ``wandb`` is a required dependency. The training entry
point ``main.py`` logs runs to Weights & Biases, so log in before training:

.. code-block:: bash

    wandb login

**User config** -- training also needs a user config,
``configs/users/<name>.yaml``, selected with ``--user <name>``; see
:doc:`model_training`.


Model Training
==============

.. warning::

    **Status:** MXtalTools has no canonical training configuration; the
    configs in ``configs/experiments/`` are task-specific or historical. The
    only automated check of this workflow is ``tests/test_model_training.py``
    (pytest marker ``slow``), which trains the autoencoder for two epochs from
    ``configs/test_configs/autoencoder.yaml`` with the user config
    ``configs/users/test_user.yaml`` and ``tests/datasets/mini_qm9_dataset.pt``.
    It was not run for this revision. The other modes have no test.

The entry point is ``main.py``, run from the repository root. It merges the
YAML configs described below into one namespace with
:func:`~mxtaltools.common.config_processing.process_main_config` and hands it
to ``mxtaltools.modeller.Modeller``:

.. code-block:: bash

    python main.py --user <name> --yaml_config <path/to/main_config.yaml>


Configs
-------

A run's settings come from, in order of precedence:

1. **Command-line arguments**, read as ``--key value`` pairs. ``--user`` and
   ``--yaml_config`` select the user and main configs; any other key replaces
   a top-level key of the main config, with the value as a string. Overrides
   are applied before the base configs are merged, so a key that is not
   written in the main config file itself (for example one that only a base
   config supplies) is silently ignored. Nested keys cannot be overridden, and
   the ``--key=value`` form is not supported.
2. **Main config** -- the file given by ``--yaml_config``; without it, the
   file named by ``paths: yaml_path`` in the user config.
3. **Base configs** -- keys missing from the main config are filled from the
   file named by ``base_config_path``, if set, and then from
   ``experiments/full_base.yaml``. Nested sections are merged key by key. Both
   paths are appended to the config directory (below) by string
   concatenation, which is why config values such as ``base_config_path``
   start with ``/``.
4. **Dataset config** -- ``dataset_yaml_path`` names a file in the ``dataset/``
   subdirectory of the config directory; its keys fill those still missing
   from the ``dataset`` section.

The **user config**, ``configs/users/<name>.yaml`` (relative to the working
directory), supplies the ``wandb`` section and the ``paths``. The main config's
``machine`` (``local`` or ``cluster``) selects the ``paths`` entries used; it
is read before the base configs are merged, so it must be set in the main
config file itself. The entries are:
``<machine>_config_path`` (the config directory),
``<machine>_dataset_dir_path``, ``<machine>_workdir_path``,
``<machine>_checkpoint_dir_path``, and ``<machine>_checkpoints_path``, which is
prefixed to every non-null ``model_paths`` entry. With ``machine: cluster``,
``save_checkpoints`` is forced to True. ``configs/users/test_user.yaml`` is a
minimal example for ``machine: local``.


Modes
-----

``mode`` selects the task. For ``autoencoder``, ``regression``,
``embedding_regression``, ``discriminator``, ``generator``, ``gan``,
``proxy_discriminator``, ``polymorph_classification`` and
``crystal_regression``, ``main.py`` trains (``Modeller.fit_models``) when
``max_epochs > 0`` and evaluates (``Modeller.evaluate_model``) when it is 0;
``embedding`` runs ``Modeller.ae_embedding_analysis``. The base configs in
``configs/experiments/base/`` set ``mode`` for their task.

Runs are logged to Weights & Biases (``wandb login`` first). When
``sweep_path`` is set, ``main.py`` creates or joins a W&B sweep; the sweep
project (``MXtalTools``) and entity (``mkilgour``) are fixed in ``main.py``.

A non-null ``model_paths.<model>`` reloads that model's checkpoint. The model
section of the config (``<model>.model``) is always replaced by the
checkpoint's; the optimizer section is replaced only when it is null or sets
``overwrite_on_reload: True``. The weights are loaded, but not the optimizer
state.


Autoencoder
-----------

``configs/experiments/base/autoencoder.yaml`` sets ``mode: autoencoder``,
``dataset_name: 'test_qm9_dataset.pt'`` and
``dataset_yaml_path: '/qm9_autoencoder.yaml'``, so a main config for an
autoencoder run starts from::

    base_config_path: /experiments/base/autoencoder.yaml

Datasets are read from the user config's ``<machine>_dataset_dir_path``.
``test_qm9_dataset.pt`` is the subset of up to 10000 samples that
:meth:`DataManager.process_new_dataset <mxtaltools.dataset_utils.dataset_manager.DataManager.process_new_dataset>`
writes next to the full ``qm9_dataset.pt`` (see :doc:`dataset_creation`).
Neither is in the repository, which holds only ``mini_qm9_dataset.pt`` (in
``mini_datasets/`` and ``tests/datasets/``).

Hydrogen handling is set by ``autoencoder.filter_protons``. When it is True,
hydrogen is dropped from the dataset's atom types, input batches are
deprotonated, and the model is built with ``protons_in_input=False``. The
autoencoder base config sets it to False and ``full_base.yaml`` to True.
``dataset.filter_protons`` and ``autoencoder.infer_protons`` appear in configs
but the training code does not read them.

To change any other setting, copy its key from the base config into the main
config and change the value there.

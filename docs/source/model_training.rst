Model Training
==============

Training is controlled by a set of ``.yaml`` configs.
Every run has a base config, and a main config which optionally overwrites entries in the base.
The main config can be specified on the command line:

.. code-block:: bash

    python main.py --user YOUR_USERNAME --yaml_config /PATH_TO_EXPERIMENT_CONFIGS/YOUR_EXPERIMENT.yaml

If no config is specified, the program uses ``dev.yaml`` by default.


Autoencoder
-----------

To train an autoencoder model on the default QM9 dataset, set your main config to::

    base_config_path: /experiments/base/autoencoder.yaml

For development, you may use the smaller 10k-sample dataset::

    dataset_name: 'test_qm9_molecules_dataset.pt'

Control hydrogen handling with::

    dataset:
      filter_protons: False

    autoencoder:
      infer_protons: False

To change any other settings for the dataset, batching, optimizer, or model, copy the relevant key from the base config into your main config and modify the value.
It will be overwritten at runtime.

When reloading a model checkpoint, model and optimizer values will automatically adopt those of the checkpoint, regardless of the config.

Results and telemetry are logged to Weights & Biases automatically.

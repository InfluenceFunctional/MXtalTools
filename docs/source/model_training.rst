XModel Training
=====

Training is controlled by a set of .yaml configs.
Every run has a base config, and a main config, which optionally overwrites entries in the base.
The main config can be specified on command line like so

.. code-block:: bash

 python main.py --user YOUR_USERNAME --yaml_config /PATH_TO_EXPERIMENT_CONFIGS/YOUR_EXPERIMENT.yaml

If no config is specified on the command line, the program will use 'dev.yaml' by default.

1. Autoencoder

To train an autoencoder model on the default qm9 dataset, edit your main config

 base_config_path: /experiments/base/autoencoder.yaml

For development purposes, you may use the smaller version of the standard dataset with only 10k samples,

 dataset_name: 'test_qm9_molecules_dataset.pt'.

Select whether you want hydrogens to be filtered altogether from the dataset and whether you want the model to infer the presence of hydrogens (requires hydrogens not be filtered in the first place).

 dataset:
   filter_protons: False

 autoencoder:
   infer_protons: False

To change any other settings for the dataset, batching, optimizer, or model, simply copy from the base method into your main config, and change the value. It will be overwritten at runtime. This is with the exception of when you are reloading a model checkpoint, where model and optimizer values will automatically adopt those of the checkpoint, regardless of whats in the config.

To run the code, simply run main.py using the python call above, or, if using a slurm-based scheduler, follow the template:

.. literalinclude:: ../../bash_sub.sh

The run will automatically log results & telemetry to wandb.

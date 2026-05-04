API Reference
=============

Common Utilities
----------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.common.adaptive_batching
    mxtaltools.common.ase_interface
    mxtaltools.common.clustering
    mxtaltools.common.config_processing
    mxtaltools.common.ellipsoid_ops
    mxtaltools.common.geometry_utils
    mxtaltools.common.instantiate_models
    mxtaltools.common.mol_classifier_utils
    mxtaltools.common.sym_utils
    mxtaltools.common.training_utils
    mxtaltools.common.utils

Constants
---------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.constants.asymmetric_units
    mxtaltools.constants.atom_properties
    mxtaltools.constants.classifier_constants
    mxtaltools.constants.csd_stats
    mxtaltools.constants.mol_classifier_constants
    mxtaltools.constants.space_group_feature_tensor
    mxtaltools.constants.space_group_info
    mxtaltools.constants.space_group_reference

Crystal Building
----------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.crystal_building.crystal_latent_transforms
    mxtaltools.crystal_building.random_crystal_sampling
    mxtaltools.crystal_building.utils

Crystal Search
--------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.crystal_search.crystal_opt_utils
    mxtaltools.crystal_search.run_search
    mxtaltools.crystal_search.utils

Dataset Utilities
-----------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.dataset_utils.data_classes
    mxtaltools.dataset_utils.dataset_manager
    mxtaltools.dataset_utils.mol_building
    mxtaltools.dataset_utils.utils

MLIP Interfaces
---------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.mlip_interfaces.mace_utils
    mxtaltools.mlip_interfaces.sp_calc_mk
    mxtaltools.mlip_interfaces.uma_utils

Models
------

Core model utilities:

.. autosummary::
    :toctree: _autosummary

    mxtaltools.models.autoencoder_utils
    mxtaltools.models.utils

Graph functions:

.. autosummary::
    :toctree: _autosummary

    mxtaltools.models.functions.minimum_image_neighbors
    mxtaltools.models.functions.radial_graph

Graph neural networks:

.. autosummary::
    :toctree: _autosummary

    mxtaltools.models.graph_models.base_graph_model
    mxtaltools.models.graph_models.graph_neural_network
    mxtaltools.models.graph_models.molecule_graph_model

Network modules:

.. autosummary::
    :toctree: _autosummary

    mxtaltools.models.modules.augmented_softmax_aggregator
    mxtaltools.models.modules.basis_functions
    mxtaltools.models.modules.components
    mxtaltools.models.modules.graph_convolution
    mxtaltools.models.modules.vector_LayerNorm

Task-specific models:

.. autosummary::
    :toctree: _autosummary

    mxtaltools.models.task_models.autoencoder_models
    mxtaltools.models.task_models.crystal_models
    mxtaltools.models.task_models.embedding_regression_models
    mxtaltools.models.task_models.generator_models
    mxtaltools.models.task_models.mol_classifier
    mxtaltools.models.task_models.regression_models

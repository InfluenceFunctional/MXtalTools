API Reference
=============

Modules are grouped by subject. Every module listed here is imported during the
documentation build, and the build fails if any of them cannot be imported.

Scripts (dataset construction, analysis and benchmarking), plotting and
reporting code, and the training entry point are not listed.


Crystal Data Objects
--------------------

:class:`~mxtaltools.dataset_utils.data_classes.MolData` and
:class:`~mxtaltools.dataset_utils.data_classes.MolCrystalData` hold molecules
and molecular crystals as PyTorch Geometric data objects. Most crystal
operations are methods on ``MolCrystalData``, defined in the
``data_class_methods`` modules and mixed into the class.

.. autosummary::
    :toctree: _autosummary

    mxtaltools.dataset_utils.data_classes
    mxtaltools.dataset_utils.utils
    mxtaltools.dataset_utils.data_class_methods.crystal_ops
    mxtaltools.dataset_utils.data_class_methods.crystal_building
    mxtaltools.dataset_utils.data_class_methods.crystal_analysis
    mxtaltools.dataset_utils.data_class_methods.mol_methods
    mxtaltools.dataset_utils.data_class_methods.ellipsoid_ops

Crystal Parameterisation and Construction
-----------------------------------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.crystal_building.crystal_latent_transforms
    mxtaltools.crystal_building.utils
    mxtaltools.crystal_building.random_crystal_sampling
    mxtaltools.crystal_building.zp_doubling
    mxtaltools.common.geometry_utils
    mxtaltools.common.sym_utils

Space Groups and Constants
--------------------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.constants.space_group_info
    mxtaltools.constants.asymmetric_units
    mxtaltools.constants.space_group_feature_tensor
    mxtaltools.constants.atom_properties
    mxtaltools.constants.csd_stats
    mxtaltools.constants.classifier_constants
    mxtaltools.constants.mol_classifier_constants

Graphs, Energies and Structure Comparison
-----------------------------------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.models.functions.radial_graph
    mxtaltools.models.functions.minimum_image_neighbors
    mxtaltools.analysis.vdw_analysis
    mxtaltools.analysis.crystal_rdf

Machine-Learned Interatomic Potentials
--------------------------------------

Optional backends: the MACE interfaces need ``mace-torch``, and the UMA
interface needs ``fairchem-core`` (``pip install mxtaltools[uma]``).

.. autosummary::
    :toctree: _autosummary

    mxtaltools.mlip_interfaces.AL_mace_utils
    mxtaltools.mlip_interfaces.pbc_neighbours
    mxtaltools.mlip_interfaces.mace_utils
    mxtaltools.mlip_interfaces.uma_utils
    mxtaltools.mlip_interfaces.sp_calc_mk

Crystal Structure Search
------------------------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.crystal_search.run_search
    mxtaltools.crystal_search.coordinator
    mxtaltools.crystal_search.run_state
    mxtaltools.crystal_search.standardize
    mxtaltools.crystal_search.crystal_opt_utils
    mxtaltools.crystal_search.utils

Conformers
----------

.. note::

    The conformer modules are being restructured. Their interfaces may change
    without notice.

.. autosummary::
    :toctree: _autosummary

    mxtaltools.conformers.topology
    mxtaltools.conformers.perception
    mxtaltools.conformers.geometry
    mxtaltools.conformers.builder
    mxtaltools.conformers.energy
    mxtaltools.conformers.prior
    mxtaltools.conformers.optimize
    mxtaltools.dataset_utils.data_class_methods.conformer_methods

Models
------

Graph models and network modules:

.. autosummary::
    :toctree: _autosummary

    mxtaltools.models.graph_models.base_graph_model
    mxtaltools.models.graph_models.graph_neural_network
    mxtaltools.models.graph_models.molecule_graph_model
    mxtaltools.models.modules.augmented_softmax_aggregator
    mxtaltools.models.modules.basis_functions
    mxtaltools.models.modules.components
    mxtaltools.models.modules.graph_convolution
    mxtaltools.models.modules.vector_LayerNorm

Task models and model loading:

.. autosummary::
    :toctree: _autosummary

    mxtaltools.models.task_models.autoencoder_models
    mxtaltools.models.task_models.crystal_models
    mxtaltools.models.task_models.embedding_regression_models
    mxtaltools.models.task_models.generator_models
    mxtaltools.models.task_models.mol_classifier
    mxtaltools.models.task_models.regression_models
    mxtaltools.models.utils
    mxtaltools.models.autoencoder_utils

Datasets
--------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.dataset_utils.mol_building
    mxtaltools.dataset_utils.dataset_manager
    mxtaltools.dataset_utils.construction.cif.reader
    mxtaltools.dataset_utils.construction.cif.tokenizer
    mxtaltools.dataset_utils.construction.cif.symmetry
    mxtaltools.dataset_utils.construction.cif.sites
    mxtaltools.dataset_utils.construction.cif.composition
    mxtaltools.dataset_utils.construction.cif.topology
    mxtaltools.dataset_utils.construction.cif.adapt
    mxtaltools.dataset_utils.construction.cif.errors

Utilities
---------

.. autosummary::
    :toctree: _autosummary

    mxtaltools.common.adaptive_batching
    mxtaltools.common.ase_interface
    mxtaltools.common.cif_io
    mxtaltools.common.clustering
    mxtaltools.common.config_processing
    mxtaltools.common.ellipsoid_ops
    mxtaltools.common.instantiate_models
    mxtaltools.common.mol_classifier_utils
    mxtaltools.common.training_utils
    mxtaltools.common.utils

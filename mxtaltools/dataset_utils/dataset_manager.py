"""Dataset manager for lists of MXtalTools molecule and crystal data objects.

Provides ``DataManager``, which collates saved chunk files into dataset files, loads a
saved dataset, filters, truncates and shuffles it, computes per-feature statistics and
crystal/molecule index maps, and assigns standardised regression targets; and
``qm9_targets_list``, the QM9 target names in the column order of ``y`` read by
``DataManager.get_target``.
"""
import gc
import glob
import os
import random
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import torch
from torch_geometric import nn as gnn
from tqdm import tqdm

from mxtaltools.common.geometry_utils import batch_compute_molecule_volume
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.atom_properties import VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD
from mxtaltools.dataset_utils.md_analysis.md_data_processing import generate_dataset_from_dumps
from mxtaltools.dataset_utils.utils import basic_stats, filter_graph_nodewise, collate_data_list
from mxtaltools.models.functions.minimum_image_neighbors import argwhere_minimum_image_convention_edges

qm9_targets_list = ["rotational_constant_a",
                    "rotational_constant_b",
                    "rotational_constant_c",
                    "dipole_moment",
                    "isotropic_polarizability",
                    "HOMO_energy",
                    "LUMO_energy",
                    "gap_energy",
                    "el_spatial_extent",
                    "zpv_energy",
                    "internal_energy_0",
                    "internal_energy_STP",
                    "enthalpy_STP",
                    "free_energy_STP",
                    "heat_capacity_STP"]


# noinspection PyAttributeOutsideInit
class DataManager:
    """Load, filter and summarise a dataset held in ``self.dataset``.

    ``self.dataset`` is normally a list of MXtalTools data objects (``MolData`` or
    ``MolCrystalData``); most methods read and replace it in place.

    Parameters
    ----------
    datasets_path : str or path-like
        Directory holding saved datasets and ``misc_data_for_*.npy`` files; stored as a
        ``Path``.
    device : str, optional
        Device of the ``asym_unit_dict`` tensors. Default ``'cpu'``.
    mode : str, optional
        Identifier-grouping mode read by ``get_identifier_duplicates``, which accepts
        ``'standard'`` (default) and ``'blind_test'``.
    chunks_path : str or path-like, optional
        Directory read by ``load_chunks``. If None,
        ``self.datasets_path.joinpath('/classifier_chunks/')`` is used; because that
        argument is an absolute path, the result is ``classifier_chunks`` at the
        filesystem root (on the same drive on Windows), not a subdirectory of
        ``datasets_path``.
    seed : int, optional
        Seed for the global NumPy and torch generators when ``config`` is None.
        Default 0.
    config : object, optional
        Attribute namespace. When given, ``config.seed`` replaces ``seed``;
        ``regression_target`` and ``sample_from_trajectory`` are copied if present in
        ``config.__dict__``. ``config.type`` is read when ``dataset_type`` is None,
        ``config.max_dataset_length`` by ``truncate_and_shuffle_dataset`` and
        ``config.dumps_dirs`` by ``molecule_cluster_dataset_processing``.
    dataset_type : str, optional
        ``'crystal'`` and ``'mol_cluster'`` select type-specific branches; other values
        take the generic path. If None, ``config.type`` is used, so ``config`` must then
        be given.
    do_crystal_indexing : bool, optional
        If True (default), ``extract_misc_stats_and_indices`` rebuilds the
        crystal/molecule index maps for crystal datasets.

    Notes
    -----
    Construction seeds the global NumPy and torch generators, stores every entry of
    ``ASYM_UNITS`` as a ``torch.Tensor`` (default floating dtype) on ``device`` in
    ``self.asym_unit_dict``, builds the atom-property lookup tensors
    (``init_atom_properties``) and sets ``self.times``, a dict of ``time()`` stamps
    written by the loading, filtering, truncation and target-assignment methods.
    ``self.regression_target`` is None unless taken from ``config``.
    """
    def __init__(self,
                 datasets_path,
                 device='cpu',
                 mode='standard',
                 chunks_path=None,
                 seed=0, config=None,
                 dataset_type=None,
                 do_crystal_indexing=True):
        self.datapoints = None
        self.datasets_path = Path(datasets_path)
        if chunks_path is not None:
            self.chunks_path = Path(chunks_path)
        else:
            self.chunks_path = self.datasets_path.joinpath('/classifier_chunks/')

        self.device = device  # cpu or cuda
        self.mode = mode  # standard or 'blind test'
        self.dataset_type = None
        self.config = config
        self.dataset_stats = None
        self.regression_target = None
        self.do_crystal_indexing = do_crystal_indexing

        if dataset_type is not None:
            self.dataset_type = dataset_type
        else:
            self.dataset_type = config.type

        if self.config is not None:
            self.regression_target = self.config.regression_target if 'regression_target' in self.config.__dict__.keys() else None
            self.sample_from_trajectory = self.config.sample_from_trajectory if 'sample_from_trajectory' in self.config.__dict__.keys() else None

        np.random.seed(seed=seed if config is None else config.seed)
        torch.manual_seed(seed=seed if config is None else config.seed)

        self.asym_unit_dict = ASYM_UNITS.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(device)

        self.init_atom_properties()

        self.times = {}

    def init_atom_properties(self):
        """Build per-element property lookup tensors, in place.

        Sets ``vdw_radii_tensor``, ``atom_weights_tensor``, ``electronegativity_tensor``,
        ``group_tensor`` and ``period_tensor`` from the values of ``VDW_RADII``,
        ``ATOM_WEIGHTS``, ``ELECTRONEGATIVITY``, ``GROUP`` and ``PERIOD`` in dict order;
        those dicts are keyed 0 to 99 in order, so position ``i`` holds the value for
        atomic number ``i``. ``extract_misc_stats_and_indices`` indexes them by ``z``;
        ``get_reduced_volume_fraction`` indexes ``vdw_radii_tensor`` by ``x``. They are
        created with torch defaults and not moved to ``device``.
        """
        self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()))
        self.atom_weights_tensor = torch.tensor(list(ATOM_WEIGHTS.values()))
        self.electronegativity_tensor = torch.tensor(list(ELECTRONEGATIVITY.values()))
        self.group_tensor = torch.tensor(list(GROUP.values()))
        self.period_tensor = torch.tensor(list(PERIOD.values()))

    def load_chunks(self,
                    chunks_patterns=None,
                    max_chunks=1e8,
                    subsamples_per_chunk=1e8):
        """Load saved chunk files into a new ``self.dataset`` list.

        Changes the process working directory to ``self.chunks_path`` (``os.chdir``)
        and reads chunk files from there.

        Parameters
        ----------
        chunks_patterns : list of str, optional
            Filename prefixes. In each pattern, backslashes and forward slashes are
            replaced by underscores, and files matching ``<pattern>*.pt`` or
            ``<pattern>*.pkl`` are collected. If None, every directory entry is a
            candidate.
        max_chunks : int, optional
            Maximum number of chunks to load. Default ``1e8``.
        subsamples_per_chunk : int, optional
            If smaller than a chunk's length, that many elements are drawn from the
            chunk without replacement by the global NumPy generator; otherwise the whole
            chunk is kept. Default ``1e8``.

        Notes
        -----
        The candidate list is shuffled with ``random.Random(1)`` (a fixed seed) before
        the first ``max_chunks`` are taken. Only names containing ``.pkl`` or ``.pt`` are
        loaded, with ``torch.load(..., weights_only=False)``. A loaded list is used as
        is; any other loaded object with ``is_batch`` True is split with
        ``to_data_list()``. A chunk raising ``EOFError`` is skipped with a printed
        message.
        """
        os.chdir(self.chunks_path)
        if chunks_patterns is None:
            chunks = os.listdir()
        else:
            chunks = []
            for pattern in chunks_patterns:
                pattern = pattern.replace('\\', '/').replace('/', '_')
                chunks.extend(glob.glob(f'{pattern}*.pt'))
                chunks.extend(glob.glob(f'{pattern}*.pkl'))

        print(f'Loading {len(chunks)}:{chunks} chunks from {chunks_patterns}')

        random.Random(1).shuffle(chunks)
        num_chunks = min([len(chunks), max_chunks])
        print(f'Collecting {num_chunks} dataset chunks')
        self.dataset = []
        for ind, chunk in enumerate(tqdm(chunks[:num_chunks])):
            if '.pkl' in chunk or '.pt' in chunk:
                try:
                    loaded_chunk = torch.load(chunk, weights_only=False)
                    if isinstance(loaded_chunk, list):
                        pass
                    elif loaded_chunk.is_batch:
                        loaded_chunk = loaded_chunk.to_data_list()
                    if subsamples_per_chunk < len(loaded_chunk):
                        samples_to_keep = np.random.choice(len(loaded_chunk), subsamples_per_chunk, replace=False)
                        self.dataset.extend([loaded_chunk[ind] for ind in samples_to_keep])
                    else:
                        self.dataset.extend(loaded_chunk)
                    del loaded_chunk
                except EOFError:
                    print(f"chunk {ind} failed to load - corrupted? EOFError")

    def load_dataset_for_modelling(self,
                                   dataset_name,
                                   override_length=None,
                                   filter_conditions=None,
                                   filter_polymorphs=False,
                                   filter_duplicate_molecules=False,
                                   filter_protons=False,
                                   conv_cutoff: Optional[float] = None,
                                   do_shuffle: bool = True,
                                   precompute_edges: bool = False,
                                   single_identifier=None
                                   ):
        """Load a dataset and prepare it and its statistics for modelling.

        Parameters
        ----------
        dataset_name : str
            File name under ``datasets_path``. Passed to
            ``molecule_cluster_dataset_processing`` for ``mol_cluster`` datasets and to
            ``load_training_dataset`` otherwise.
        override_length : int, optional
            Maximum dataset length, forwarded to ``truncate_and_shuffle_dataset``.
        filter_conditions : list, optional
            Conditions for ``get_dataset_filter_inds``; see ``compute_filter``.
        filter_polymorphs : bool, optional
            If True, keep one element per identifier group (``filter_polymorphs``).
            Default False.
        filter_duplicate_molecules : bool, optional
            If True, keep one element per unique molecular fingerprint
            (``filter_duplicate_molecules``). Default False.
        filter_protons : bool, optional
            If True, atomic number 1 is removed from ``present_atom_types``; the atoms
            themselves stay in the data. Default False.
        conv_cutoff : float, optional
            Edge cutoff for ``compute_edges``; used only when ``precompute_edges`` is
            True.
        do_shuffle : bool, optional
            Forwarded to ``truncate_and_shuffle_dataset``. Default True.
        precompute_edges : bool, optional
            If True, call ``compute_edges(conv_cutoff)``. Default False.
        single_identifier : str, optional
            If given, ``self.dataset`` is replaced by ``len(self.dataset)`` references to
            the first element whose ``identifier`` equals it (``list.index`` raises
            ``ValueError`` if none does).

        Notes
        -----
        Steps, in order: load; ``dataset_filtration``; for crystal datasets in which
        every ``z_prime`` equals 1, ``remove_zpg1_info``;
        ``truncate_and_shuffle_dataset``; recompute ``self.misc_dataset`` and
        ``self.dataset_stats`` on the truncated dataset with
        ``extract_misc_stats_and_indices``; ``assign_regression_targets``; set
        ``self.present_atom_types`` to the unique atomic numbers; set ``self.dataDims``
        from ``get_data_dimensions``; then the optional edge and single-identifier
        steps. Returns None.
        """

        if self.dataset_type == 'mol_cluster':
            self.molecule_cluster_dataset_processing(dataset_name)
        else:
            self.load_training_dataset(dataset_name)

        self.dataset_filtration(filter_conditions, filter_duplicate_molecules, filter_polymorphs)
        if self.dataset_type == 'crystal':
            if torch.all(torch.tensor([e.z_prime for e in self.dataset]) == 1):
                self.remove_zpg1_info()
        self.truncate_and_shuffle_dataset(override_length, do_shuffle=do_shuffle)
        self.misc_dataset = self.extract_misc_stats_and_indices(self.dataset)
        self.dataset_stats = self.misc_dataset['dataset_stats']
        self.assign_regression_targets()
        self.present_atom_types, _ = self.dataset_stats['atomic_number']['uniques']
        if filter_protons:
            if 1 in self.present_atom_types:
                self.present_atom_types = self.present_atom_types[self.present_atom_types != 1]

        self.dataDims = self.get_data_dimensions()

        if precompute_edges:
            self.compute_edges(conv_cutoff)

        if single_identifier is not None:  # replace dataset with copies of one element
            idents = [elem.identifier for elem in self.dataset]
            good_ind = idents.index(single_identifier)
            self.dataset = [self.dataset[good_ind] for _ in range(len(self.dataset))]

    def remove_zpg1_info(self):
        """Truncate Z'-indexed crystal attributes to their first molecule, in place.

        For every element of ``self.dataset``, keeps the first 3 columns of
        ``aunit_centroid`` and ``aunit_orientation`` and the first column of
        ``aunit_handedness``.
        """
        # filter Z'>1 information if not needed
        for elem in self.dataset:
            elem.aunit_centroid = elem.aunit_centroid[:, :3]
            elem.aunit_orientation = elem.aunit_orientation[:, :3]
            elem.aunit_handedness = elem.aunit_handedness[:, :1]

    def compute_edges(self,
                      conv_cutoff: float,
                      buffer: Optional[float] = 0.5):
        """Precompute graph edges for every element of ``self.dataset``, in place.

        Parameters
        ----------
        conv_cutoff : float
            Edge cutoff, in the units of the element positions.
        buffer : float, optional
            Added to ``conv_cutoff`` for datasets other than ``mol_cluster``.
            Default 0.5.

        Notes
        -----
        For ``mol_cluster`` datasets, calls
        ``molecule_cluster_edge_indexing(conv_cutoff)``; ``buffer`` is not used.
        Otherwise calls each element's
        ``construct_intra_radial_graph(cutoff=conv_cutoff + buffer)``, which sets
        ``edges_dict`` and ``edge_index`` on the element from pairs of its own atoms.
        """
        if self.dataset_type == 'mol_cluster':
            self.molecule_cluster_edge_indexing(conv_cutoff)
        else:
            # doesn't work yet for crystal datasets
            for ind in tqdm(range(len(self.dataset))):
                self.dataset[ind].construct_intra_radial_graph(
                    cutoff=conv_cutoff + buffer)  # add buffer to cutoff to allow for positional noising
                # todo need a custom collation function for these

    def molecule_cluster_dataset_processing(self, dataset_name):
        """Build ``self.dataset`` for a ``mol_cluster`` dataset from LAMMPS dump chunks.

        Parameters
        ----------
        dataset_name : str
            File name under ``datasets_path``; only its existence is checked.

        Notes
        -----
        If ``datasets_path / dataset_name`` does not exist and ``config`` has
        ``dumps_dirs``, converts the dumps in ``datasets_path / d`` for each ``d`` in
        ``config.dumps_dirs`` into chunks under ``datasets_path / 'classifier_chunks'``
        with ``generate_dataset_from_dumps(..., steps_per_save=1)``. Then calls
        ``process_new_dataset(new_dataset_name=None,
        chunks_patterns=config.dumps_dirs, save_dataset=False)``, converts
        ``self.dataset`` with ``to_data_list()`` and sets ``self.dataset_stats`` from
        ``self.misc_dataset``. As written, ``process_new_dataset`` with its default
        ``build_stats=True`` raises ``TypeError`` when it concatenates
        ``'misc_data_for_'`` with None, and ``load_chunks`` leaves ``self.dataset`` as a
        list, which has no ``to_data_list``.
        """
        # todo this almost certainly no longer works
        if not os.path.exists(self.datasets_path.joinpath(dataset_name)) and 'dumps_dirs' in self.config.__dict__.keys():
            # if it hasn't already been done, convert the relevant LAMMPS trajectories into trainable data objects
            generate_dataset_from_dumps([self.datasets_path.joinpath(dir_name) for dir_name in self.config.dumps_dirs],
                                        self.datasets_path.joinpath('classifier_chunks'),
                                        steps_per_save=1)

        self.process_new_dataset(new_dataset_name=None,
                                 chunks_patterns=[dump_dir for dump_dir in self.config.dumps_dirs],
                                 save_dataset=False,
                                 )
        self.dataset = self.dataset.to_data_list()  # process in list form
        self.dataset_stats = self.misc_dataset['dataset_stats']

    def molecule_cluster_edge_indexing(self, conv_cutoff):
        """Precompute minimum-image edges for each element of a ``mol_cluster`` dataset, in place.

        Parameters
        ----------
        conv_cutoff : float
            Maximum edge length, in the units of ``pos``.

        Notes
        -----
        Calls ``argwhere_minimum_image_convention_edges(1, elem.pos, elem.T_fc,
        conv_cutoff)`` for each element and sets its ``edge_index`` ([2, num_edges])
        and ``edge_attr`` (the [num_edges] minimum-image distances). The edges reflect
        the positions at call time and are not updated if the coordinates are later
        noised.
        """
        for ind in tqdm(range(len(self.dataset))):
            edges_dict = argwhere_minimum_image_convention_edges(1,
                                                                 self.dataset[ind].pos,
                                                                 self.dataset[ind].T_fc,
                                                                 conv_cutoff)
            self.dataset[ind].edge_index = edges_dict['edge_index']
            self.dataset[ind].edge_attr = edges_dict['dists']

    def truncate_and_shuffle_dataset(self, override_length=None, do_shuffle=True):
        """Truncate ``self.dataset`` to a maximum length, optionally in random order.

        Parameters
        ----------
        override_length : int, optional
            Maximum length. If None, ``config.max_dataset_length`` is used when
            ``config`` is given, otherwise ``1000000000000``.
        do_shuffle : bool, optional
            If True (default), keeps ``dataset_length`` indices drawn without
            replacement by the global NumPy generator, in random order. If False, keeps
            ``np.linspace(0, len(self.dataset) - 1, dataset_length)`` cast to int, in
            dataset order.

        Notes
        -----
        Sets ``self.max_dataset_length`` and ``self.dataset_length``
        (``min(len(self.dataset), max_dataset_length)``) and replaces ``self.dataset``.
        No train/test split is made here.
        """
        self.times['dataset_shuffle_start'] = time()
        # get dataset length & shuffle
        if override_length is not None:
            self.max_dataset_length = override_length
        elif self.config is not None:
            self.max_dataset_length = self.config.max_dataset_length
        else:
            self.max_dataset_length = 1000000000000

        self.dataset_length = min(len(self.dataset), self.max_dataset_length)
        if do_shuffle:
            inds_to_keep = list(np.random.choice(len(self.dataset), self.dataset_length, replace=False))
        else:
            inds_to_keep = np.linspace(0, len(self.dataset) - 1, min(len(self.dataset), self.dataset_length)).astype(
                int)

        self.dataset = [self.dataset[ind] for ind in inds_to_keep]
        self.times['dataset_shuffle_end'] = time()

    def assign_regression_targets(self):
        """Store the standardised targets from ``get_target`` in each element's ``y``, in place.

        For 1-D targets, element ``i`` gets ``y = targets[i]``. For targets with more
        than one dimension and ``targets.shape[1] > 1``, it gets ``targets[None, i]``
        (a leading dimension of 1). For targets with more than one dimension and
        ``targets.shape[1] == 1``, no ``y`` is assigned.
        """
        self.times['dataset_targets_assignment_start'] = time()
        targets = self.get_target()  # todo assign targets element-by-element in data list rather than as batch, omitting collation
        for ind in range(len(self.dataset)):
            if targets.ndim > 1:
                if targets.shape[1] > 1:
                    self.dataset[ind].y = targets[None, ind]
            else:
                self.dataset[ind].y = targets[ind]
        self.times['dataset_targets_assignment_end'] = time()

    def dataset_filtration(self, filter_conditions, filter_duplicate_molecules, filter_polymorphs):
        """Remove elements of ``self.dataset`` by condition, identifier group and duplicate molecule.

        Parameters
        ----------
        filter_conditions : list or None
            Conditions for ``get_dataset_filter_inds``; skipped if None.
        filter_duplicate_molecules : bool
            If True, removes the indices returned by ``filter_duplicate_molecules``.
        filter_polymorphs : bool
            If True, removes the indices returned by ``filter_polymorphs``.

        Notes
        -----
        The steps run in the order conditions, polymorphs, duplicate molecules; for
        crystal datasets ``rebuild_crystal_indices`` runs after each applied step.
        Removal counts are printed.
        """
        self.times['dataset_filtering_start'] = time()
        if filter_conditions is not None:
            bad_inds = self.get_dataset_filter_inds(filter_conditions)
            good_inds = [ind for ind in range(len(self.dataset)) if ind not in bad_inds]
            self.dataset = [self.dataset[ind] for ind in good_inds]
            print("Filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))
            if self.dataset_type == 'crystal':
                self.rebuild_crystal_indices()

        if filter_polymorphs:
            bad_inds = self.filter_polymorphs()
            good_inds = [ind for ind in range(len(self.dataset)) if ind not in bad_inds]
            self.dataset = [self.dataset[ind] for ind in good_inds]
            print("Polymorph filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))
            if self.dataset_type == 'crystal':
                self.rebuild_crystal_indices()

        if filter_duplicate_molecules:
            bad_inds = self.filter_duplicate_molecules()
            good_inds = [ind for ind in range(len(self.dataset)) if ind not in bad_inds]
            self.dataset = [self.dataset[ind] for ind in good_inds]
            print(
                "Duplicate molecule filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))
            if self.dataset_type == 'crystal':
                self.rebuild_crystal_indices()
        # if filter_protons:  # could be done in the above filtering, but a useful separate utility function in some cases
        #     self.filter_protons()
        self.times['dataset_filtering_end'] = time()

    def filter_protons(self):
        """Apply ``filter_graph_nodewise`` to ``self.dataset`` with the node mask ``self.dataset.x == 1``.

        The mask marks the nodes to keep. ``self.dataset`` must be a collated object with
        ``num_nodes`` and ``x``; a list dataset raises ``AttributeError``.
        ``filter_graph_nodewise`` calls ``data.subgraph``, which the MXtalTools data
        classes do not define. Not called by other methods of this class.
        """
        init_len = self.dataset.num_nodes
        keep_bools = self.dataset.x == 1
        self.dataset = filter_graph_nodewise(self.dataset, keep_bools)  # this is broken with our custom datatype
        print(f"Proton filter removed {init_len - self.dataset.num_nodes} atoms leaving {self.dataset.num_nodes}")

    def get_data_dimensions(self):
        """Collect feature names, standardisation vectors and dataset sizes into a dict.

        Also sets ``atom_keys``, ``molecule_keys``, ``lattice_keys``, ``lattice_means``,
        ``lattice_stds``, ``lattice_stats`` and ``length_covariance_matrix`` on
        ``self``. Requires ``dataset_stats``, ``dataset_length``, ``target_mean``,
        ``target_std`` and ``present_atom_types`` to be set.

        Returns
        -------
        dict
            ``'node_standardization_vector'``: array [6, 2] of (tight_mean, tight_std)
            for each of ``atom_keys``, with row 0 (atomic number) set to (0, 1);
            ``'graph_standardization_vector'``: array [3, 2] of (tight_mean, tight_std)
            for each of ``molecule_keys``; ``'standardization_dict'``:
            ``dataset_stats``; ``'dataset_length'``; ``'lattice_means'``,
            ``'lattice_stds'``, ``'lattice_stats'`` and ``'lattice_length_cov_mat'``;
            ``'regression_target'``, ``'target_mean'`` and ``'target_std'``;
            ``'num_atom_features'``, ``'atom_features'``, ``'num_molecule_features'``
            and ``'molecule_features'``; ``'allowed_atom_types'`` and
            ``'num_atom_types'`` from ``present_atom_types``. For ``mol_cluster``
            datasets, also ``'num_polymorphs'`` (the number of unique ``polymorph``
            values) and ``'num_topologies'`` (0).

        Notes
        -----
        For crystal datasets the lattice entries are the ``tight_mean`` and
        ``tight_std`` of the 12 ``lattice_keys`` statistics, those statistics dicts
        keyed by name, and ``torch.cov`` of the concatenated per-element
        ``cell_lengths`` rows. Otherwise they are 12 zeros, 12 values of 0.01, 12 empty
        lists and ``torch.ones((3, 3))``.
        """
        self.atom_keys = ['atomic_number', 'vdw_radii', 'atom_weight', 'electronegativity', 'group', 'period']
        self.molecule_keys = ['num_atoms', 'radius', 'mol_volume']
        self.lattice_keys = ['cell_a', 'cell_b', 'cell_c',
                             'cell_alpha', 'cell_beta', 'cell_gamma',
                             'aunit_x', 'aunit_y', 'aunit_z',
                             'aunit_rot_x', 'aunit_rot_y', 'aunit_rot_z']
        if self.dataset_type == 'crystal':
            self.lattice_means = [self.dataset_stats[feat]['tight_mean'] for feat in self.lattice_keys]
            self.lattice_stds = [self.dataset_stats[feat]['tight_std'] for feat in self.lattice_keys]
            self.lattice_stats = {key: self.dataset_stats[key] for key in self.lattice_keys}
            self.length_covariance_matrix = torch.cov(
                torch.cat([self.dataset[ind].cell_lengths for ind in range(len(self.dataset))], dim=0).T)
        else:
            self.lattice_means = [0 for _ in range(12)]
            self.lattice_stds = [0.01 for _ in range(12)]
            self.lattice_stats = [[] for _ in range(12)]
            self.length_covariance_matrix = torch.ones((3, 3))
        node_standardization_vector = np.asarray([[[self.dataset_stats[feat]['tight_mean'],
                                                    self.dataset_stats[feat]['tight_std']] for feat in
                                                   self.atom_keys]])[0]
        node_standardization_vector[0, :] = [0, 1]  # don't standardize atomic numbers - always first entry
        graph_standardization_vector = np.asarray([[[self.dataset_stats[feat]['tight_mean'],
                                                     self.dataset_stats[feat]['tight_std']] for feat in
                                                    self.molecule_keys]])[0]

        dim = {
            'node_standardization_vector': node_standardization_vector,
            'graph_standardization_vector': graph_standardization_vector,
            'standardization_dict': self.dataset_stats,
            'dataset_length': self.dataset_length,

            'lattice_means': self.lattice_means,
            'lattice_stds': self.lattice_stds,
            'lattice_stats': self.lattice_stats,
            'lattice_length_cov_mat': self.length_covariance_matrix,

            'regression_target': self.regression_target,
            'target_mean': self.target_mean,
            'target_std': self.target_std,

            'num_atom_features': len(self.atom_keys),
            'atom_features': self.atom_keys,

            'num_molecule_features': len(self.molecule_keys),
            'molecule_features': self.molecule_keys,

            'allowed_atom_types': self.present_atom_types,
            'num_atom_types': len(self.present_atom_types),
        }

        if self.dataset_type == 'mol_cluster': # todo probably non-functional
            dim['num_polymorphs'] = len(torch.unique(torch.cat([elem.polymorph for elem in self.dataset])))
            dim['num_topologies'] = 0

        return dim

    def get_target(self):
        """Compute standardised regression targets for ``self.dataset``.

        Sets ``self.target_mean`` and ``self.target_std``.

        Returns
        -------
        torch.Tensor
            ``(targets - target_mean) / target_std``, with ``targets`` built as described
            in Notes (``torch.tensor`` over per-element values, or ``torch.cat`` of
            per-element tensors along dim 0). If ``regression_target`` is None,
            ``target_mean`` and ``target_std`` are set to 0 and 1 and a tensor of
            ``len(self.dataset)`` zeros is returned.

        Raises
        ------
        AssertionError
            If ``regression_target`` is none of the named cases in Notes and reading it
            from the elements' ``_store`` or building the tensor raises any exception.

        Notes
        -----
        Targets by ``regression_target``: a name in ``qm9_targets_list`` reads that
        column of ``elem.y``; ``'crystal_packing_coefficient'`` reads
        ``packing_coeff``; ``'dipole'`` concatenates ``dipole``;
        ``'normed_aunit_volume'`` concatenates ``aunit_volume() / radius**3``; any other
        name reads ``elem._store[regression_target]``. Mean and std come from the
        targets clipped to the 5th and 95th percentiles of all their entries, over
        dim 0 when ``targets[0].ndim > 1`` and over all entries otherwise; the returned
        values are not clipped. A std whose maximum is below ``1e-4`` is replaced by ones.
        """
        if self.regression_target is not None:  # todo rewrite as data list method
            if self.regression_target in qm9_targets_list:
                target_ind = qm9_targets_list.index(self.regression_target)
                targets = torch.tensor([elem.y[:, target_ind] for elem in self.dataset])
            elif self.regression_target == 'crystal_packing_coefficient':
                targets = torch.tensor([elem.packing_coeff for elem in self.dataset])
            elif self.regression_target == 'dipole':
                targets = torch.cat([elem.dipole for elem in self.dataset])
            elif self.regression_target == 'normed_aunit_volume':
                targets = torch.cat([elem.aunit_volume()/elem.radius**3 for elem in self.dataset])
            else:
                try:
                    targets = torch.tensor([elem.__dict__['_store'][self.regression_target] for elem in self.dataset])
                except:
                    assert False, "Unrecognized regression target"

            clipped_targets = targets.clip(min=torch.quantile(targets, 0.05), max=torch.quantile(targets, 0.95))

            if targets[0].ndim > 1:
                self.target_mean = clipped_targets.mean(0)
                self.target_std = clipped_targets.std(0)
            else:
                self.target_mean = clipped_targets.mean()
                self.target_std = clipped_targets.std()

            if self.target_std.max() < 1e-4:
                self.target_std = torch.ones_like(self.target_std)

            std_targets = (targets - self.target_mean) / self.target_std

            return std_targets

        else:  # need have something just to fill the space
            self.target_mean, self.target_std = 0, 1
            return torch.tensor([0 for _ in range(len(self.dataset))])

    def get_reduced_volume_fraction(self):
        """Return ``reduced_volume`` over the summed van der Waals sphere volumes, per element.

        Returns
        -------
        torch.Tensor
            One entry per element along dim 0: ``elem.reduced_volume`` divided by the
            sum of ``4 / 3 * pi * r ** 3`` over the element's atoms, with ``r`` looked
            up in ``vdw_radii_tensor`` by ``elem.x``.
        """
        red_vol = torch.tensor([elem.reduced_volume for elem in self.dataset])
        atom_volumes = torch.tensor(
            [torch.sum(4 / 3 * torch.pi * self.vdw_radii_tensor[elem.x] ** 3) for elem in self.dataset])
        targets = red_vol / atom_volumes
        return targets

    def rebuild_crystal_indices(self):
        """Recompute ``crystal_to_mol_dict``, ``mol_to_crystal_dict`` and ``unique_molecules_dict`` from ``self.dataset``, in place."""
        # identify which molecules are in which crystals and vice-versa
        self.crystal_to_mol_dict, self.mol_to_crystal_dict = self.generate_mol2crystal_mapping()
        self.unique_molecules_dict = self.identify_unique_molecules_in_crystals()

    def load_training_dataset(self, dataset_name):
        """Load a saved dataset into ``self.dataset`` along with its miscellaneous data.

        Parameters
        ----------
        dataset_name : str
            File name under ``datasets_path``, ending in a 3-character suffix such as
            ``.pt``.

        Notes
        -----
        Loads with ``torch.load(..., weights_only=False)``; a loaded object whose
        lower-cased ``str(type(...))`` (module path included) contains ``'batch'`` is
        converted with ``to_data_list()``. Then loads ``misc_data_for_<stem>.npy`` from
        ``datasets_path``, where ``<stem>`` is ``dataset_name`` without its last 3
        characters and without everything up to and including its last ``'test_'``; if
        that file is absent, computes the dict with ``extract_misc_stats_and_indices``.
        The dict is stored as ``self.misc_dataset`` and each of its keys is also set as
        an attribute of ``self``. If ``dataset_name`` contains ``'blind_test'``,
        ``self.mode`` is set to ``'blind test'``. For crystal datasets whose name
        contains ``'test'`` or ``'mini'``, ``rebuild_crystal_indices`` is called.
        """
        self.times['dataset_loading_start'] = time()
        self.dataset = torch.load(self.datasets_path.joinpath(dataset_name), weights_only=False)

        if 'batch' in str(type(self.dataset)).lower():
            # if it's batched, revert to data list - this is slow, so if possible don't store datasets as batches but as data lists
            self.dataset = self.dataset.to_data_list()
            print("Dataset is in pre-collated format, which slows down initial loading!")

        """get miscellaneous data"""
        misc_data_path = self.datasets_path.joinpath('misc_data_for_' + dataset_name[:-3].split('test_')[-1] + '.npy')
        if os.path.exists(misc_data_path):
            self.misc_dataset = np.load(misc_data_path, allow_pickle=True).item()
        else:
            self.misc_dataset = self.extract_misc_stats_and_indices(self.dataset)

        self.times['dataset_loading_end'] = time()
        for key in self.misc_dataset.keys():
            setattr(self, key, self.misc_dataset[key])

        if 'blind_test' in dataset_name:
            self.mode = 'blind test'
            print("Switching to blind test indexing mode")  # TODO don't believe this does anything anymore

        if (('test' in dataset_name) or ('mini' in dataset_name)) and self.dataset_type == 'crystal':
            self.rebuild_crystal_indices()

    def process_new_dataset(self,
                            new_dataset_name: str = None,
                            test_dataset_size: int = 10000,
                            max_chunks: int = 1e8,
                            chunks_patterns: list = None,
                            samples_per_chunk=1e8,
                            build_stats: bool = True,
                            save_dataset=True):
        """Collate chunk files into ``self.dataset`` and optionally save it with its statistics.

        Parameters
        ----------
        new_dataset_name : str, optional
            Base file name of the outputs. Must be a string when ``build_stats`` or
            ``save_dataset`` is True.
        test_dataset_size : int, optional
            Maximum size of the ``test_`` subset. Default 10000.
        max_chunks : int, optional
            Forwarded to ``load_chunks``. Default ``1e8``.
        chunks_patterns : list of str, optional
            Forwarded to ``load_chunks``.
        samples_per_chunk : int, optional
            Forwarded to ``load_chunks`` as ``subsamples_per_chunk``. Default ``1e8``.
        build_stats : bool, optional
            If True (default), sets ``self.misc_dataset`` from
            ``extract_misc_stats_and_indices`` and writes it with ``np.save`` to
            ``datasets_path / ('misc_data_for_' + new_dataset_name)`` (``np.save``
            appends ``.npy``).
        save_dataset : bool, optional
            If True (default), writes ``mini_<name>.pt``, ``test_<name>.pt`` and
            ``<name>.pt`` under ``datasets_path``.

        Notes
        -----
        The ``mini_`` file holds the first ``min(len(self.dataset), 100)`` elements and
        the ``test_`` file the first ``min(len(self.dataset), test_dataset_size)``
        elements, each as a list permuted by the global NumPy generator. The full file
        holds ``self.dataset`` as loaded.
        """
        self.load_chunks(chunks_patterns=chunks_patterns,
                         max_chunks=max_chunks,
                         subsamples_per_chunk=samples_per_chunk)

        gc.collect()

        if isinstance(self.datasets_path, str):
            self.datasets_path = Path(self.datasets_path)

        if build_stats:
            print("building dataset statistics")
            self.misc_dataset = self.extract_misc_stats_and_indices(self.dataset)
            misc_dataset_path = self.datasets_path.joinpath('misc_data_for_' + new_dataset_name)
            np.save(misc_dataset_path, self.misc_dataset)

        if save_dataset:
            # dataset for unit testing
            mini_dataset_size = 100
            ints = list(
                np.random.choice(min(len(self.dataset), mini_dataset_size),
                                 min(len(self.dataset), mini_dataset_size),
                                 replace=False))

            test_dataset_path = self.datasets_path.joinpath('mini_' + new_dataset_name + '.pt')
            torch.save([self.dataset[ind] for ind in ints],
                       test_dataset_path)

            # dataset for functionality testing
            ints = list(
                np.random.choice(min(len(self.dataset), test_dataset_size),
                                 min(len(self.dataset), test_dataset_size),
                                 replace=False))

            test_dataset_path = self.datasets_path.joinpath('test_' + new_dataset_name + '.pt')
            torch.save([self.dataset[ind] for ind in ints],
                       test_dataset_path)

            # save full dataset
            train_dataset_path = self.datasets_path.joinpath(new_dataset_name + '.pt')
            torch.save(self.dataset,
                       train_dataset_path)
    

    def extract_misc_stats_and_indices(self, dataset):  #do this batchwise for memory efficiency
        """Compute per-feature statistics and, for crystal datasets, index maps.

        Parameters
        ----------
        dataset : list or collated data object
            Data objects; a list is collated with ``collate_data_list``.

        Returns
        -------
        dict
            ``'dataset_stats'``: dict of ``basic_stats`` results keyed by feature.
            Atom-level ``'atomic_number'`` (``z``) and ``'vdw_radii'``,
            ``'atom_weight'``, ``'electronegativity'``, ``'group'`` and ``'period'``
            (looked up by ``z``); molecule-level ``'num_atoms'``, ``'radius'`` and
            ``'mol_volume'``. Crystal datasets add ``'cell_a'``, ``'cell_b'``,
            ``'cell_c'`` and ``'cell_alpha'``, ``'cell_beta'``, ``'cell_gamma'`` (columns
            of ``cell_lengths`` and ``cell_angles``), ``'aunit_x'``, ``'aunit_y'``,
            ``'aunit_z'`` and ``'aunit_rot_x'``, ``'aunit_rot_y'``, ``'aunit_rot_z'``
            (first three columns of ``aunit_centroid`` and ``aunit_orientation``),
            ``'cell_volume'`` and ``'packing_coefficient'``. With
            ``do_crystal_indexing``, crystal datasets also get ``'crystal_to_mol_dict'``,
            ``'mol_to_crystal_dict'`` and ``'unique_molecules_dict'``.

        Notes
        -----
        The index maps are rebuilt by ``rebuild_crystal_indices`` from
        ``self.dataset``, not from the ``dataset`` argument, and are also set as
        attributes of ``self``.
        """
        if isinstance(dataset, list):
            data_batch = collate_data_list(dataset)
        else:
            data_batch = dataset

        misc_data_dict = {
            'dataset_stats': {
                'atomic_number': basic_stats(data_batch.z.long()),
                'vdw_radii': basic_stats(self.vdw_radii_tensor[data_batch.z.long()]),
                'atom_weight': basic_stats(self.atom_weights_tensor[data_batch.z.long()]),
                'electronegativity': basic_stats(self.electronegativity_tensor[data_batch.z.long()]),
                'group': basic_stats(self.group_tensor[data_batch.z.long()].long()),
                'period': basic_stats(self.period_tensor[data_batch.z.long()].long()),
                'num_atoms': basic_stats(data_batch.num_atoms.long()),
                'radius': basic_stats(data_batch.radius.float()),
                'mol_volume': basic_stats(data_batch.mol_volume.float()),
            }
        }

        if self.dataset_type == 'crystal':  # MK assumes Z'=1
            misc_data_dict['dataset_stats'].update({
                'cell_a': basic_stats(data_batch.cell_lengths[:, 0].float()),
                'cell_b': basic_stats(data_batch.cell_lengths[:, 1].float()),
                'cell_c': basic_stats(data_batch.cell_lengths[:, 2].float()),
                'cell_alpha': basic_stats(data_batch.cell_angles[:, 0].float()),
                'cell_beta': basic_stats(data_batch.cell_angles[:, 1].float()),
                'cell_gamma': basic_stats(data_batch.cell_angles[:, 2].float()),
                'aunit_x': basic_stats(data_batch.aunit_centroid[:, 0].float()),
                'aunit_y': basic_stats(data_batch.aunit_centroid[:, 1].float()),
                'aunit_z': basic_stats(data_batch.aunit_centroid[:, 2].float()),
                'aunit_rot_x': basic_stats(data_batch.aunit_orientation[:, 0].float()),
                'aunit_rot_y': basic_stats(data_batch.aunit_orientation[:, 1].float()),
                'aunit_rot_z': basic_stats(data_batch.aunit_orientation[:, 2].float()),
                'cell_volume': basic_stats(data_batch.cell_volume.float()),
                'packing_coefficient': basic_stats(data_batch.packing_coeff.float()),
            })

            if self.do_crystal_indexing:
                self.rebuild_crystal_indices()

                misc_data_dict.update({
                    'crystal_to_mol_dict': self.crystal_to_mol_dict,
                    'mol_to_crystal_dict': self.mol_to_crystal_dict,
                    'unique_molecules_dict': self.unique_molecules_dict
                })
        return misc_data_dict

    def generate_mol2crystal_mapping(self):
        """Map dataset identifiers to molecule indices and back.

        Some crystals have multiple molecules, and molecules are analysed with a
        separate indexing scheme; these dicts connect the identifier-wise and
        molecule-wise indexing. The current code assigns exactly one molecule index per
        element of ``self.dataset``, in dataset order, regardless of ``z_prime``.

        Returns
        -------
        crystal_to_mol_dict : dict
            ``str(identifier)`` -> one-element list holding that element's molecule
            index. If identifiers repeat, the last such element's index is kept.
        mol_to_crystal_dict : dict
            Molecule index -> ``str(identifier)``.
        """
        # print("Generating mol-to-crystal mapping")  # todo replace below with data-list based method
        mol_index = 0
        crystal_to_mol_dict = {}
        mol_to_crystal_dict = {}
        for index in range(len(self.dataset)):
            identifier = str(self.dataset[index].identifier)
            crystal_to_mol_dict[identifier] = []
            for _ in range(1):#int(self.dataset[index].z_prime)):  # assumes integer Z'
                crystal_to_mol_dict[identifier].append(mol_index)
                mol_to_crystal_dict[mol_index] = identifier
                mol_index += 1

        return crystal_to_mol_dict, mol_to_crystal_dict

    def identify_unique_molecules_in_crystals(self):
        """Identify all exactly unique molecules (up to molecular fingerprint) and list their dataset indices.

        At train time this can be used to repeat sampling of identical molecules.

        Returns
        -------
        dict or None
            Keys are the bytes of each unique fingerprint row; values are the lists of
            indices of the elements carrying it. Only the first 2048 entries of each
            element's ``fingerprint`` are used (one molecule per element). If
            ``self.dataset[0]`` has no ``fingerprint`` attribute, a message mentioning
            SMILES is printed and None is returned; no SMILES-based grouping is done.
        """
        if hasattr(self.dataset[0],'fingerprint'):
            # print("getting unique molecule fingerprints")
            fingerprints = []
            for z1 in range(len(self.dataset)):
                zp = int(1)#self.dataset[z1].z_prime)
                for ind in range(zp):
                    fingerprints.append(self.dataset[z1].fingerprint[2048 * ind:2048 * (ind + 1)])
            fps = np.stack(fingerprints)

            unique_fps, inverse_map = np.unique(fps, axis=0, return_inverse=True)
            molecules_in_crystals_dict = {unique.tobytes(): [] for unique in unique_fps}
            for ind, mapping in enumerate(inverse_map):  # we record the molecule inex for each unique molecular fingerprint
                molecules_in_crystals_dict[unique_fps[mapping].tobytes()].append(ind)
        else:
            print("Dataset missing fingerprint information, "
                  "using smiles (hopefully canonical!)")
            molecules_in_crystals_dict = None

        return molecules_in_crystals_dict

    def get_identifier_duplicates(self):
        """Group dataset indices by CSD identifier, or by blind test sample submission.

        In ``'standard'`` mode, groups by CSD identifier. CSD entries with numbers on
        the end are subsequent additions to the same crystal, often polymorphs or repeat
        measurements, so an identifier whose last character is a digit has its last two
        characters removed before grouping. In ``'blind_test'`` mode, groups by the
        first Roman numeral, searched from ``'XXX'`` down to ``'I'``, that is a
        substring of the identifier.

        Returns
        -------
        duplicate_lists : list of list of int
            Index groups with more than one member.
        duplicate_list_extension : list of int
            Concatenation of ``duplicate_lists``.
        duplicate_groups : dict
            Group key -> index list, for groups with more than one member; the lists
            are the same objects as in ``duplicate_lists``.

        Raises
        ------
        AssertionError
            If ``self.mode`` is neither ``'standard'`` nor ``'blind_test'``, including
            the value ``'blind test'`` that ``load_training_dataset`` sets.
        """
        print('Getting identifier duplicates')

        if self.mode == 'standard':  #
            crystal_to_identifier = {}
            for i in tqdm(range(len(self.dataset))):
                item = str(self.dataset[i].identifier)
                if item[-1].isdigit():
                    item = item[:-2]  # cut off 2 trailing digits, if any - always 2 in the CSD
                if item not in crystal_to_identifier.keys():
                    crystal_to_identifier[item] = []
                crystal_to_identifier[item].append(i)
        elif self.mode == 'blind_test':  # NOTE this is almost certainly deprecated / nonfunctional today
            blind_test_targets = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                                  'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
                                  'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX']

            crystal_to_identifier = {key: [] for key in blind_test_targets}
            for i in tqdm(range(len(self.dataset))):
                item = self.dataset[i].identifier
                for j in range(
                        len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
                    if blind_test_targets[-1 - j] in item:
                        crystal_to_identifier[blind_test_targets[-1 - j]].append(i)
                        break
        else:
            assert False, f"No such mode as {self.mode}"

        # delete identifiers with only one entry, despite having an index attached
        duplicate_lists = [crystal_to_identifier[key] for key in crystal_to_identifier.keys() if
                           len(crystal_to_identifier[key]) > 1]
        duplicate_list_extension = []
        for d_list in duplicate_lists:
            duplicate_list_extension.extend(d_list)

        duplicate_groups = {}
        for key in crystal_to_identifier.keys():
            if len(crystal_to_identifier[key]) > 1:
                duplicate_groups[key] = crystal_to_identifier[key]

        return duplicate_lists, duplicate_list_extension, duplicate_groups

    def get_dataset_filter_inds(self, filter_conditions):
        """Identify indices not passing certain filter conditions.

        Parameters
        ----------
        filter_conditions : list
            Conditions in the format ``[column_name, condition_type, [min, max] or
            [set]]`` with ``condition_type`` in ``['range', 'in', 'not_in']``; each is
            evaluated by ``compute_filter``.

        Returns
        -------
        numpy.ndarray
            Sorted unique indices of ``self.dataset`` failing at least one condition.
        """
        print('Filtering dataset starting from {} samples'.format(len(self.dataset)))
        bad_inds = []  # indices to be filtered

        for condition in filter_conditions:
            bad_inds.extend(self.compute_filter(condition))

        bad_inds = np.unique(bad_inds)  # remove redundant conditions

        return bad_inds

    def compute_filter(self, condition):
        """Apply one filter condition and return the indices of ``self.dataset`` that fail it.

        Parameters
        ----------
        condition : list
            ``[condition_key, condition_type, condition_range]``.

        Returns
        -------
        list of int or torch.Tensor
            Failing indices: a list for ``'atomic_number'``, a 1-D long tensor
            otherwise.

        Raises
        ------
        AssertionError
            For keys other than ``'atomic_number'``: if ``get_condition_values`` does
            not recognise the key, ``condition_type`` is not ``'range'``, ``'in'`` or
            ``'not_in'``, or a ``'range'`` condition does not have max > min.
        UnboundLocalError
            For ``condition_type == 'not_in'`` on keys other than ``'atomic_number'``:
            the matching indices are computed but not assigned, so the final print
            fails.

        Notes
        -----
        ``'atomic_number'`` ignores ``condition_type``: an element fails if any of its
        ``z`` values is not in ``condition_range``. Other keys are read with
        ``get_condition_values``, one value per element; ``'range'`` fails values
        outside ``[min, max]`` (bounds inclusive) and ``'in'`` fails values equal to
        none of ``condition_range``. No Z'-specific handling is applied. The
        ``'crystal_identifier'`` values are a list, on which the ``'range'`` and
        ``'in'`` comparisons raise ``TypeError``.
        """
        condition_key, condition_type, condition_range = condition
        if condition_key == 'atomic_number':
            # must search within molecules
            bad_inds = []
            for ind, elem in enumerate(self.dataset):
                if not set(elem.z.numpy()).issubset(condition_range):
                    bad_inds.append(ind)
        else:
            # molecule-wise
            values = self.get_condition_values(condition_key)

            if condition_type == 'range':
                # check fo the values to be in the range (inclusive)
                assert condition_range[1] > condition_range[0], "Range condition must be set low to high"
                bad_inds = torch.cat((
                    torch.argwhere(values > condition_range[1]),
                    torch.argwhere(values < condition_range[0])
                ))[:, 0]

            elif condition_type == 'in':
                # check for where the data is not equal to the explicitly enumerated range elements to be included
                # dataset entries are float, so switch conditions to float arrays
                bad_inds = torch.argwhere(
                    torch.logical_not(
                        torch.any(
                            torch.cat([values[..., None] == cond for cond in condition_range], dim=1),
                            dim=1))).flatten()

            elif condition_type == 'not_in':
                torch.argwhere(
                    torch.any(
                        torch.cat([values[..., None] == cond for cond in condition_range], dim=1),
                        dim=1)).flatten()
            else:
                assert False, f"{condition_type} is not an implemented dataset filter condition"

        print(f'{condition} filtered {len(bad_inds)} samples')

        return bad_inds

    def get_condition_values(self, condition_key):  # todo convert from batch back to data lists
        """Return one value per element of ``self.dataset`` for a filter key.

        Parameters
        ----------
        condition_key : str
            One of ``'crystal_z_prime'`` (``z_prime``),
            ``'asymmetric_unit_is_well_defined'`` (``is_well_defined``),
            ``'crystal_symmetry_operations_are_nonstandard'``
            (``nonstandard_symmetry``), ``'max_atomic_number'`` (``z.amax()``),
            ``'molecule_num_atoms'`` (``num_atoms``), ``'molecule_radius'``
            (``radius``), ``'crystal_space_group_number'`` (``sg_ind``),
            ``'crystal_identifier'`` (``identifier``), ``'reduced_volume_fraction'``
            (``get_reduced_volume_fraction``), ``'time_step'`` (``time_step``, built
            with ``torch.Tensor``, the default floating dtype) or
            ``'crystal_packing_coefficient'`` (``packing_coeff``).

        Returns
        -------
        torch.Tensor or list
            Tensor with one entry per element along dim 0, or a list of identifiers
            for ``'crystal_identifier'``.

        Raises
        ------
        AssertionError
            If ``condition_key`` is not one of the above.
        """
        if condition_key == 'crystal_z_prime':
            values = torch.tensor([elem.z_prime for elem in self.dataset])
        elif condition_key == 'asymmetric_unit_is_well_defined':
            values = torch.tensor([elem.is_well_defined for elem in self.dataset])
        elif condition_key == 'crystal_symmetry_operations_are_nonstandard':
            values = torch.tensor([elem.nonstandard_symmetry for elem in self.dataset])
        elif condition_key == 'max_atomic_number':
            values = torch.tensor([elem.z.amax() for elem in self.dataset])
        elif condition_key == 'molecule_num_atoms':
            values = torch.tensor([elem.num_atoms for elem in self.dataset])
        elif condition_key == 'molecule_radius':
            values = torch.tensor([elem.radius for elem in self.dataset])
        elif condition_key == 'crystal_space_group_number':
            values = torch.tensor([elem.sg_ind for elem in self.dataset])
        elif condition_key == 'crystal_identifier':
            values = [elem.identifier for elem in self.dataset]
        elif condition_key == 'reduced_volume_fraction':
            # ratio of asymmetric unit volume to the sum of atomwise volumes
            # a very coarse proxy for packing coefficient
            values = self.get_reduced_volume_fraction()
        elif condition_key == 'time_step':
            values = torch.Tensor([elem.time_step for elem in self.dataset])
        elif condition_key == 'crystal_packing_coefficient':
            values = torch.tensor([elem.packing_coeff for elem in self.dataset])
        else:
            assert False, f"{condition_key} is not implemented as a filterable item"

        return values

    def filter_polymorphs(self):
        """Pick one representative per identifier group and return the other indices.

        Groups come from ``get_identifier_duplicates`` (CSD identifier stems in
        ``'standard'`` mode). For each group with more than one member, one index is
        drawn by the global NumPy generator and kept; the others are returned. The
        group lists are modified in place.

        Returns
        -------
        list of int
            Indices of ``self.dataset`` to remove.
        """
        # use CSD identifiers to pick out polymorphs
        duplicate_lists, duplicate_list_extension, duplicate_groups = self.get_identifier_duplicates()  # issue - some of these are not isomorphic (i.e., different ionization), maybe ~2% from early tests

        # TODO consider other ways to select 'representative' polymorph
        # the representative structure will be randomly sampled from all available polymorphs
        # we will add all others to 'bad_inds', and filter them out at our leisure
        #print('Selecting representative structures from duplicate polymorphs')
        bad_inds = []
        for key in duplicate_groups.keys():
            inds = duplicate_groups[key]
            sampled_ind = np.random.choice(inds, size=1)
            inds.remove(sampled_ind)  # remove the good one
            bad_inds.extend(inds)  # delete unselected polymorphs from the dataset

        return bad_inds

    def filter_duplicate_molecules(self):
        """Find duplicate examples and pick one representative per molecule.

        For each entry of ``unique_molecules_dict`` with more than one molecule index,
        the molecules are mapped to identifiers through ``mol_to_crystal_dict`` and to
        dataset indices by identifier; one index is drawn by the global NumPy generator
        and kept, and the others are returned. An identifier shared by several elements
        maps to the last such index.

        Returns
        -------
        list of int
            Indices of ``self.dataset`` to remove.

        Notes
        -----
        Requires ``unique_molecules_dict`` and ``mol_to_crystal_dict``, set by
        ``rebuild_crystal_indices`` or from the loaded miscellaneous data; raises
        ``AttributeError`` if ``unique_molecules_dict`` is None.
        """
        # the representative structure will be randomly sampled from all available identical molecules
        # we will add all others to 'bad_inds', and filter them out at our leisure
        #print('Selecting representative structures from duplicate molecules')
        index_to_identifier_dict = {str(elem.identifier): ind for ind, elem in enumerate(self.dataset)}
        bad_inds = []
        for ind, (key, value) in enumerate(self.unique_molecules_dict.items()):
            if len(value) > 1:  # if there are multiple molecules
                mol_inds = self.unique_molecules_dict[key]  # identify their mol indices
                crystal_identifiers = [self.mol_to_crystal_dict[ind] for ind in
                                       mol_inds]  # identify their crystal identifiers
                crystal_inds = [index_to_identifier_dict[identifier] for identifier in crystal_identifiers]
                sampled_ind = np.random.choice(crystal_inds, size=1)
                crystal_inds.remove(sampled_ind)  # remove the good one
                bad_inds.extend(crystal_inds)  # delete unselected polymorphs from the dataset

        return bad_inds

    def __getitem__(self, idx):
        """Return ``self.dataset[idx]``."""
        return self.dataset[idx]

    def __len__(self):
        """Return ``len(self.dataset)``."""
        return len(self.dataset)


if __name__ == '__main__':
    miner = DataManager(device='cpu',
                        datasets_path=r"D:\crystal_datasets/",
                        chunks_path=r"D:\crystal_datasets/CSD_featurized_chunks/",
                        dataset_type='crystal')

    miner.load_dataset_for_modelling(dataset_name='test_CSD_dataset.pt',
                                     filter_conditions=[
                                         ['crystal_z_prime', 'in', [1]],
                                         # NOTE we can currently only process Z' == 1 in models
                                         ['crystal_symmetry_operations_are_nonstandard', 'in', [False]],
                                         ['max_atomic_number', 'range', [1, 100]],
                                         #['molecule_is_symmetric_top','in',[False]],
                                         #['molecule_is_spherical_top','in',[False]],
                                         #['crystal_packing_coefficient','range',[0.55, 0.85]],
                                         ['molecule_num_atoms', 'range', [3, 100]],
                                         ['molecule_radius', 'range', [1, 10]],
                                         ['asymmetric_unit_is_well_defined', 'in', [True]],

                                         #['crystal_identifier', 'not_in', ['OBEQUJ', 'OBEQOD', 'OBEQET', 'XATJOT', 'OBEQIX', 'KONTIQ','NACJAF', 'XAFPAY', 'XAFQON', 'XAFQIH', 'XAFPAY01', 'XAFPAY02', 'XAFPAY03', 'XAFPAY04','XAFQON','XAFQIH']],  # omit blind test 5 & 6 targets
                                         #['crystal_space_group_number','in',[2,14,19]]
                                     ])


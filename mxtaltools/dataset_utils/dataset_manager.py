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
        self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()))
        self.atom_weights_tensor = torch.tensor(list(ATOM_WEIGHTS.values()))
        self.electronegativity_tensor = torch.tensor(list(ELECTRONEGATIVITY.values()))
        self.group_tensor = torch.tensor(list(GROUP.values()))
        self.period_tensor = torch.tensor(list(PERIOD.values()))

    def load_chunks(self,
                    chunks_patterns=None,
                    max_chunks=1e8,
                    subsamples_per_chunk=1e8):
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
                    loaded_chunk = torch.load(chunk)
                    if loaded_chunk.is_batch:
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
        """

        Parameters
        ----------
        precompute_edges: bool
        do_shuffle
        conv_cutoff : float
        dataset_name
        override_length
        filter_conditions
        filter_polymorphs
        filter_duplicate_molecules
        filter_protons

        Returns
        -------

        """

        if self.dataset_type == 'mol_cluster':
            self.molecule_cluster_dataset_processing(dataset_name)
        else:
            self.load_training_dataset(dataset_name)

        self.dataset_filtration(filter_conditions, filter_duplicate_molecules, filter_polymorphs)
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

    def compute_edges(self,
                      conv_cutoff: float,
                      buffer: Optional[float] = 0.5):
        if self.dataset_type == 'mol_cluster':
            self.molecule_cluster_edge_indexing(conv_cutoff)
        else:
            # doesn't work yet for crystal datasets
            for ind in tqdm(range(len(self.dataset))):
                self.dataset[ind].construct_intra_radial_graph(
                    cutoff=conv_cutoff + buffer)  # add buffer to cutoff to allow for positional noising
                # todo need a custom collation function for these

    def molecule_cluster_dataset_processing(self, dataset_name):
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
        """prepopulate edge information - expensive to do repeatedly - will not work if we noise the coordinates"""
        for ind in tqdm(range(len(self.dataset))):
            edges_dict = argwhere_minimum_image_convention_edges(1,
                                                                 self.dataset[ind].pos,
                                                                 self.dataset[ind].T_fc,
                                                                 conv_cutoff)
            self.dataset[ind].edge_index = edges_dict['edge_index']
            self.dataset[ind].edge_attr = edges_dict['dists']

    def truncate_and_shuffle_dataset(self, override_length=None, do_shuffle=True):
        """defines train/test split as well as overall dataset size"""
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
        init_len = self.dataset.num_nodes
        keep_bools = self.dataset.x == 1
        self.dataset = filter_graph_nodewise(self.dataset, keep_bools)  # this is broken with our custom datatype
        print(f"Proton filter removed {init_len - self.dataset.num_nodes} atoms leaving {self.dataset.num_nodes}")

    def get_data_dimensions(self):
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
        red_vol = torch.tensor([elem.reduced_volume for elem in self.dataset])
        atom_volumes = torch.tensor(
            [torch.sum(4 / 3 * torch.pi * self.vdw_radii_tensor[elem.x] ** 3) for elem in self.dataset])
        targets = red_vol / atom_volumes
        return targets

    def rebuild_crystal_indices(self):
        # identify which molecules are in which crystals and vice-versa
        self.crystal_to_mol_dict, self.mol_to_crystal_dict = self.generate_mol2crystal_mapping()
        self.unique_molecules_dict = self.identify_unique_molecules_in_crystals()

    def load_training_dataset(self, dataset_name):
        self.times['dataset_loading_start'] = time()
        self.dataset = torch.load(self.datasets_path.joinpath(dataset_name))

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

        if 'test' in dataset_name and self.dataset_type == 'crystal':
            self.rebuild_crystal_indices()

    def process_new_dataset(self,
                            new_dataset_name: str = None,
                            test_dataset_size: int = 10000,
                            max_chunks: int = 1e8,
                            chunks_patterns: list = None,
                            samples_per_chunk=1e8,
                            build_stats: bool = True,
                            save_dataset=True):
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

        if self.dataset_type == 'crystal':
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
        """
        some crystals have multiple molecules, and we do batch analysis of molecules with a separate indexing scheme
        connect the crystal identifier-wise and mol-wise indexing with the following dicts
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
        """
        identify all exactly unique molecules (up to mol fingerprint)
        list their dataset indices in a dict

        at train time, we can use this to repeat sampling of identical molecules
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
                  "skipping unique molecule checks")
            molecules_in_crystals_dict = None

        return molecules_in_crystals_dict

    def get_identifier_duplicates(self):
        """
        by CSD identifier
        CSD entries with numbers on the end are subsequent additions to the same crystal
        often polymorphs or repeat  measurements

        option for grouping identifier by blind test sample submission
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
        elif self.mode == 'blind_test':  # todo test
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
        """
        identify indices not passing certain filter conditions
        conditions in the format [column_name, condition_type, [min, max] or [set]]
        condition_type in ['range','in','not_in']
        """
        print('Filtering dataset starting from {} samples'.format(len(self.dataset)))
        bad_inds = []  # indices to be filtered

        for condition in filter_conditions:
            bad_inds.extend(self.compute_filter(condition))

        bad_inds = np.unique(bad_inds)  # remove redundant conditions

        return bad_inds

    def compute_filter(self, condition):
        """
        apply given filters
        for atoms & molecules with potential Z'>1, need to adjust formatting a bit
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
        if condition_key == 'crystal_z_prime':
            values = torch.ones(len(self.dataset))
            # we're no longer doing this for now
            #values = torch.tensor([elem.z_prime for elem in self.dataset])
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
        """
        find duplicate examples and pick one representative per molecule
        :return:
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
        """
        find duplicate examples and pick one representative per molecule
        :return:
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
        return self.dataset[idx]

    def __len__(self):
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


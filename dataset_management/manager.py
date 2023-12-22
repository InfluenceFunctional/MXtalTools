import pandas as pd
import torch
from tqdm import tqdm
import os
import numpy as np

from common.utils import delete_from_dataframe, standardize_np
from constants.asymmetric_units import asym_unit_dict
from constants.space_group_info import SYM_OPS
from crystal_building.utils import build_unit_cell, batch_asymmetric_unit_pose_analysis_torch
from dataset_management.CrystalData import CrystalData
from dataset_management.utils import get_range_fraction, get_fraction
from constants.atom_properties import ELECTRONEGATIVITY, PERIOD, GROUP, VDW_RADII, SYMBOLS


class DataManager:
    def __init__(self, datasets_path, device='cpu', mode='standard', chunks_path=None, seed=0):
        self.datapoints = None
        self.datasets_path = datasets_path
        self.chunks_path = chunks_path
        self.device = device  # cpu or cuda
        self.mode = mode  # standard or 'blind test'

        np.random.seed(seed=seed)  # for certain random sampling ops

        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(device)

    def load_chunks(self):
        os.chdir(self.chunks_path)
        chunks = os.listdir()
        num_chunks = len(chunks)
        print(f'Collecting {num_chunks} dataset chunks')
        self.dataset = pd.concat([pd.read_pickle(chunk) for chunk in chunks], ignore_index=True)

    def load_dataset_for_modelling(self, config, dataset_name, misc_dataset_name, override_length=None,
                                   filter_conditions=None, filter_polymorphs=False, filter_duplicate_molecules=False):

        self.load_dataset_and_misc_data(dataset_name, misc_dataset_name)

        if filter_conditions is not None:
            bad_inds = self.get_dataset_filter_inds(filter_conditions)
            self.dataset = delete_from_dataframe(self.dataset, bad_inds)
            print("Filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))
            self.rebuild_indices()

        if filter_polymorphs:
            bad_inds = self.filter_polymorphs()
            self.dataset = delete_from_dataframe(self.dataset, bad_inds)
            print("Polymorph filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))
            self.rebuild_indices()

        if filter_duplicate_molecules:
            bad_inds = self.filter_duplicate_molecules()
            self.dataset = delete_from_dataframe(self.dataset, bad_inds)
            print("Duplicate molecule filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))
            self.rebuild_indices()

        self.generate_datapoints(config, override_length)

    def generate_datapoints(self, config, override_length):
        self.regression_target = config.regression_target
        self.dataset_seed = config.seed
        self.single_molecule_dataset_identifier = config.single_molecule_dataset_identifier
        np.random.seed(self.dataset_seed)

        if isinstance(self.dataset['atom_atomic_numbers'], list):  # todo write a function for this style of flexible concatenation
            self.allowed_atom_types = np.unique(
                np.concatenate([atoms for atoms_lists in self.dataset['atom_atomic_numbers'] for atoms in atoms_lists]))
        else:
            self.allowed_atom_types = np.unique(
                np.concatenate(self.dataset['atom_atomic_numbers']))

        if override_length is not None:
            self.max_dataset_length = override_length
        else:
            self.max_dataset_length = config.max_dataset_length

        self.dataset_length = min(len(self.dataset), self.max_dataset_length)

        if self.dataset_type == 'crystal':
            self.max_molecule_radius = np.amax(self.dataset['molecule_radius'])[0]
            self.min_num_atoms = np.amin(self.dataset['molecule_num_atoms'])[0]
            self.max_num_atoms = np.amax(self.dataset['molecule_num_atoms'])[0]
        elif self.dataset_type == 'molecule':
            self.max_molecule_radius = np.amax(self.dataset['molecule_radius'])
            self.min_num_atoms = np.amin(self.dataset['molecule_num_atoms'])
            self.max_num_atoms = np.amax(self.dataset['molecule_num_atoms'])

        # shuffle and cut up dataset before processing
        self.dataset = self.dataset.loc[np.random.choice(len(self.dataset), self.dataset_length, replace=False)]
        self.dataset = self.dataset.reset_index().drop(columns='index')  # reindexing is crucial here
        self.last_minute_featurization_and_one_hots()  # add a few odds & ends
        if config.save_dataset:
            self.dataset.to_pickle('training_dataset.pkl')

        '''identify keys to load & track'''
        self.atom_keys = config.atom_feature_keys
        self.molecule_keys = config.molecule_feature_keys
        if self.dataset_type == 'crystal':
            self.set_crystal_keys()
            self.set_crystal_generation_keys()
        else:
            self.crystal_keys = []
            self.lattice_keys = []
            self.crystal_generation_features = []
        self.set_tracking_keys()

        '''
        prep for modelling
        '''
        self.datapoints = self.generate_training_datapoints()

        if self.single_molecule_dataset_identifier is not None:  # make dataset a bunch of the same molecule
            identifiers = [item.csd_identifier for item in self.datapoints]
            index = identifiers.index(self.single_molecule_dataset_identifier)  # PIQTOY # VEJCES reasonably flat molecule # NICOAM03 from the paper fig
            new_datapoints = [self.datapoints[index] for i in range(self.dataset_length)]
            self.datapoints = new_datapoints

        self.dataDims = self.get_data_dimensions()

    def get_data_dimensions(self):
        dim = {
            'standardization_dict': self.standardization_dict,
            'dataset_length': self.dataset_length,

            'lattice_features': self.lattice_keys,
            'num_lattice_features': len(self.lattice_keys),
            'lattice_means': np.asarray([self.standardization_dict[key][0] for key in self.lattice_keys]),
            'lattice_stds': np.asarray([self.standardization_dict[key][1] for key in self.lattice_keys]),
            'lattice_cov_mat': self.covariance_matrix,

            'regression_target': self.regression_target,
            'target_mean': self.standardization_dict[self.regression_target][0],
            'target_std': self.standardization_dict[self.regression_target][1],

            'num_tracking_features': len(self.tracking_keys),
            'tracking_features': self.tracking_keys,

            'num_atom_features': len(self.atom_keys),
            'atom_features': self.atom_keys,

            'num_molecule_features': len(self.molecule_keys),
            'molecule_features': self.molecule_keys,

            'crystal_generation features': self.crystal_generation_features,
            'num crystal generation features': len(self.crystal_generation_features),

            'allowed_atom_types': self.allowed_atom_types,
            'num_atom_types': len(self.allowed_atom_types),
            'max_molecule_radius': self.max_molecule_radius,
            'min_molecule_num_atoms': self.min_num_atoms,
            'max_molecule_num_atoms': self.max_num_atoms,
        }

        return dim

    def set_crystal_generation_keys(self):
        # add symmetry features for generator
        self.crystal_generation_features = []
        space_group_features = [column for column in self.dataset.columns if 'sg_is' in column]
        crystal_system_features = [column for column in self.dataset.columns if 'crystal_system_is' in column]
        self.crystal_generation_features.extend(space_group_features)
        self.crystal_generation_features.extend(crystal_system_features)
        # self.crystal_generation_features.append('crystal_z_value')
        # self.crystal_generation_features.append('crystal_z_prime')
        self.crystal_generation_features.append('crystal_symmetry_multiplicity')
        # self.crystal_generation_features.append('crystal_packing_coefficient')
        # self.crystal_generation_features.append('crystal_cell_volume')
        # self.crystal_generation_features.append('crystal_reduced_volume')

    def set_tracking_keys(self):
        """
        set keys to be kept in tracking feature array
        will break if any of these are objects or strings
        """
        self.tracking_keys = []
        self.tracking_keys.extend(self.molecule_keys)
        if self.dataset_type == 'crystal':
            self.tracking_keys.extend(self.crystal_keys)
            self.tracking_keys.extend(self.lattice_keys)

        composition_keys = []
        for key in self.dataset.columns:
            if 'count' in key:
                composition_keys.append(key)
            if 'fraction' in key and 'fractional' not in key:
                composition_keys.append(key)
        self.tracking_keys.extend(composition_keys)

        self.tracking_keys.append('molecule_num_atoms')
        self.tracking_keys.append('molecule_volume')
        self.tracking_keys.append('molecule_mass')

        self.tracking_keys = list(set(self.tracking_keys))  # remove duplicates

    def set_crystal_keys(self):
        self.crystal_keys = ['crystal_space_group_number', 'crystal_space_group_setting',
                             'crystal_density', 'crystal_packing_coefficient',
                             'crystal_lattice_a', 'crystal_lattice_b', 'crystal_lattice_c',
                             'crystal_lattice_alpha', 'crystal_lattice_beta', 'crystal_lattice_gamma',
                             'crystal_z_value', 'crystal_z_prime', 'crystal_reduced_volume', 'crystal_cell_volume',
                             'crystal_symmetry_multiplicity', 'asymmetric_unit_is_well_defined',
                             ]

        # correct order here is crucially important
        self.lattice_keys = ['crystal_lattice_a', 'crystal_lattice_b', 'crystal_lattice_c',
                             'crystal_lattice_alpha', 'crystal_lattice_beta', 'crystal_lattice_gamma',
                             'asymmetric_unit_centroid_x', 'asymmetric_unit_centroid_y', 'asymmetric_unit_centroid_z',
                             'asymmetric_unit_rotvec_theta', 'asymmetric_unit_rotvec_phi', 'asymmetric_unit_rotvec_r'
                             ]

    def last_minute_featurization_and_one_hots(self):
        """
        add or update a few features including crystal feature one-hots
        #
        note we are surpressing a performancewarning from Pandas here.
        It's easier to do it this way and doesn't seem that slow.
        """

        if self.dataset_type == 'crystal':
            '''
            z_value
            '''
            for i in range(1, 32 + 1):
                self.dataset['crystal_z_is_{}'.format(i)] = self.dataset['crystal_z_value'] == i

            '''
            space group
            '''
            from constants.space_group_info import SPACE_GROUPS
            for i, symbol in enumerate(np.unique(list(SPACE_GROUPS.values()))):
                self.dataset['crystal_sg_is_' + symbol] = self.dataset['crystal_space_group_symbol'] == symbol

            '''
            crystal system
            '''
            from constants.space_group_info import LATTICE_TYPE
            # get dictionary for crystal system elements
            for i, system in enumerate(np.unique(list(LATTICE_TYPE.values()))):
                self.dataset['crystal_system_is_' + system] = self.dataset['crystal_system'] == system

            '''
            # set angle units to natural
            '''
            if (max(self.dataset['crystal_lattice_alpha']) > np.pi) or (max(self.dataset['crystal_lattice_beta']) > np.pi) or (max(self.dataset['crystal_lattice_gamma']) > np.pi):
                self.dataset['crystal_lattice_alpha'] = self.dataset['crystal_lattice_alpha'] * np.pi / 180
                self.dataset['crystal_lattice_beta'] = self.dataset['crystal_lattice_beta'] * np.pi / 180
                self.dataset['crystal_lattice_gamma'] = self.dataset['crystal_lattice_gamma'] * np.pi / 180

            '''
            check for heavy atoms
            '''
            znums = [10, 18, 36, 54]
            for znum in znums:
                self.dataset[f'molecule_atom_heavier_than_{znum}_fraction'] = np.asarray([get_range_fraction(atom_list, [znum, 200]) for atom_list in self.dataset['atom_atomic_numbers']])
        elif self.dataset_type == 'molecule':
            for anum in self.allowed_atom_types:
                self.dataset[f'molecule_{SYMBOLS[anum]}_fraction'] = np.asarray([
                    get_fraction(atom_list, anum) for atom_list in self.dataset['atom_atomic_numbers']
                    ])

    def get_regression_target(self):
        targets = self.dataset[self.regression_target]
        target_mean = self.standardization_dict[self.regression_target][0]
        target_std = self.standardization_dict[self.regression_target][1]

        return (targets - target_mean) / target_std

    def gather_tracking_features(self):
        """
        collect features of 'molecules' and append to atom-level data
        these must all be bools ints or floats - no strings will be processed
        """
        feature_array = np.zeros((self.dataset_length, len(self.tracking_keys)), dtype=float)
        for column_ind, key in enumerate(self.tracking_keys):
            feature_vector = np.asarray(self.dataset[key])

            if isinstance(feature_vector[0], list):  # feature vector is an array of lists
                feature_vector = np.concatenate(feature_vector)
            feature_array[:, column_ind] = feature_vector

        return feature_array

    def get_cell_features(self, ):
        """
        the 12 lattice parameters in correct order
        """
        key_dtype = []
        # featurize

        feature_array = np.zeros((self.dataset_length, 12), dtype=float)
        if self.dataset_type == 'crystal':
            for column_ind, key in enumerate(self.lattice_keys):
                feature_vector = np.asarray(self.dataset[key])

                if isinstance(feature_vector[0], list):  # feature vector is an array of lists
                    feature_vector = np.concatenate(feature_vector)

                key_dtype.append(feature_vector.dtype)

                feature_array[:, column_ind] = feature_vector

                assert np.sum(np.isnan(feature_vector)) == 0

        '''
        compute full covariance matrix, in raw basis
        '''
        if len(feature_array) == 1:  # error handling for if there_is_only one entry in the dataset, e.g., during CSP
            feature_array_with_normed_lengths = np.stack([feature_array for _ in range(10)])[:, 0, :]
        self.covariance_matrix = np.cov(feature_array, rowvar=False)  # we want the randn model to generate samples with normed lengths

        for i in range(len(self.covariance_matrix)):  # ensure it's well-conditioned
            self.covariance_matrix[i, i] = max((0.01, self.covariance_matrix[i, i]))

        return feature_array

    def concatenate_atom_features(self):
        """
        collect and normalize/standardize relevant atomic features
        must be bools ints or floats
        :param dataset:
        :return:
        """
        if self.dataset_type == 'crystal':
            atom_features_list = [np.zeros((len(self.dataset['atom_atomic_numbers'][i][0]), len(self.atom_keys))) for i in range(self.dataset_length)]
        elif self.dataset_type == 'molecule':
            atom_features_list = [np.zeros((len(self.dataset['atom_atomic_numbers'][i]), len(self.atom_keys))) for i in range(self.dataset_length)]

        for column_ind, key in enumerate(self.atom_keys):
            for i in range(self.dataset_length):
                if self.dataset_type == 'crystal':
                    feature_vector = np.asarray(self.dataset[key][i])[0]  # all atom features are lists-of-lists, for Z'=1 always just take the first element
                else:
                    feature_vector = np.asarray(self.dataset[key][i]) # all atom features are lists-of-lists, for Z'=1 always just take the first element

                if key == 'atom_atomic_numbers':
                    pass
                elif feature_vector.dtype == bool:
                    pass
                else:
                    feature_vector = standardize_np(feature_vector, known_mean=self.standardization_dict[key][0], known_std=self.standardization_dict[key][1])

                assert np.sum(np.isnan(feature_vector)) == 0
                atom_features_list[i][:, column_ind] = feature_vector

        return atom_features_list

    # todo add tests/ assertions for data construction

    def concatenate_molecule_features(self):
        """
        collect features of 'molecules' and append to atom-level data
        """

        # don't add molecule target if we are going to model it
        if self.regression_target in self.molecule_keys:
            self.molecule_keys.remove(self.regression_target)

        molecule_feature_array = np.zeros((self.dataset_length, len(self.molecule_keys)), dtype=float)
        for column_ind, key in enumerate(self.molecule_keys):
            feature_vector = np.asarray(self.dataset[key])

            if isinstance(feature_vector[0], list):  # feature vector is an array of lists
                feature_vector = np.concatenate(feature_vector)

            if feature_vector.dtype == bool:
                pass
            else:
                feature_vector = standardize_np(feature_vector, known_mean=self.standardization_dict[key][0], known_std=self.standardization_dict[key][1])

            molecule_feature_array[:, column_ind] = feature_vector

        assert np.sum(np.isnan(molecule_feature_array)) == 0
        return molecule_feature_array

    def generate_training_datapoints(self):
        tracking_features = self.gather_tracking_features()
        lattice_features = self.get_cell_features()
        targets = self.get_regression_target()
        molecule_features_array = self.concatenate_molecule_features()
        atom_features_list = self.concatenate_atom_features()

        combined_keys = self.atom_keys + self.molecule_keys + self.crystal_keys
        self.dataset.drop(columns=[key for key in combined_keys if key in self.dataset.columns], inplace=True)  # delete encoded columns to save on RAM
        self.dataset.drop(columns=[key for key in self.tracking_keys if key in self.dataset.columns], inplace=True)  # some of these are duplicates of above

        atom_coords = np.asarray(self.dataset['atom_coordinates'])
        if self.dataset_type == 'crystal':
            unit_cell_coords = self.dataset['crystal_unit_cell_coordinates']
            T_fc_list = self.dataset['crystal_fc_transform']
            crystal_identifier = self.dataset['crystal_identifier']
            asym_unit_handedness = self.dataset['asymmetric_unit_handedness']
            symmetry_ops = self.dataset['crystal_symmetry_operators']
        else:
            unit_cell_coords = torch.ones(len(self.dataset))
            T_fc_list = torch.ones(len(self.dataset))
            crystal_identifier = torch.ones(len(self.dataset))
            asym_unit_handedness = torch.ones(len(self.dataset))
            symmetry_ops = torch.ones(len(self.dataset))

        self.dataset = None
        return self.make_datapoints(atom_coords=atom_coords,
                                    atom_features_list=atom_features_list,
                                    mol_features=molecule_features_array,
                                    targets=targets,
                                    tracking_features=tracking_features,
                                    reference_cells=unit_cell_coords,
                                    lattice_features=lattice_features,
                                    T_fc_list=T_fc_list,
                                    identifiers=crystal_identifier,
                                    asymmetric_unit_handedness=asym_unit_handedness,
                                    crystal_symmetries=symmetry_ops)

    def make_datapoints(self, atom_coords, atom_features_list, mol_features,
                        targets, tracking_features, reference_cells, lattice_features,
                        T_fc_list, identifiers, asymmetric_unit_handedness, crystal_symmetries):
        """
        convert feature, target and tracking vectors into torch.geometric data objects
        :param atom_coords:
        :param smiles:
        :param atom_features_list:
        :param mol_features:
        :param targets:
        :param tracking_features:
        :return:
        """
        datapoints = []

        mult_ind = self.tracking_keys.index('crystal_symmetry_multiplicity') if self.dataset_type == 'crystal' else 0
        sg_ind_value_ind = self.tracking_keys.index('crystal_space_group_number') if self.dataset_type == 'crystal' else 0
        mol_size_ind = self.tracking_keys.index('molecule_num_atoms')
        mol_volume_ind = self.tracking_keys.index('molecule_volume')

        tracking_features = torch.Tensor(tracking_features)
        print("Generating crystal data objects")
        for i in tqdm(range(self.dataset_length)):
            datapoints.append(
                CrystalData(x=torch.Tensor(atom_features_list[i]),
                            pos=torch.Tensor(atom_coords[i])[0] if self.dataset_type == 'crystal' else torch.Tensor(atom_coords[i]),
                            y=targets[i],
                            mol_x=torch.Tensor(mol_features[i, None, :]),
                            tracking=tracking_features[i, None, :],
                            ref_cell_pos=np.asarray(reference_cells[i]),  # won't collate properly as a torch tensor - must leave as np array
                            mult=tracking_features[i, mult_ind].int(),
                            sg_ind=tracking_features[i, sg_ind_value_ind].int(),
                            cell_params=torch.Tensor(lattice_features[i, None, :]),
                            T_fc=torch.Tensor(T_fc_list[i])[None, ...],
                            mol_size=torch.Tensor(tracking_features[i, mol_size_ind]),
                            mol_volume=torch.Tensor(tracking_features[i, mol_volume_ind]),
                            csd_identifier=identifiers[i],
                            asym_unit_handedness=torch.Tensor(np.asarray(asymmetric_unit_handedness[i])),
                            symmetry_operators=crystal_symmetries[i]
                            ))

        return datapoints

    def rebuild_indices(self):
        self.dataset = self.dataset.reset_index().drop(columns='index')

        if self.dataset_type == 'crystal':
            self.crystal_to_mol_dict, self.mol_to_crystal_dict = \
                self.generate_mol2crystal_mapping()

            self.molecules_in_crystals_dict = \
                self.identify_unique_molecules_in_crystals()

    def load_dataset_and_misc_data(self, dataset_name, misc_dataset_name):
        self.dataset = pd.read_pickle(self.datasets_path + dataset_name)
        misc_data_dict = np.load(self.datasets_path + misc_dataset_name, allow_pickle=True).item()
        self.dataset_type = 'molecule' if 'qm9' in dataset_name.lower() else 'crystal'

        if 'blind_test' in dataset_name:
            self.mode = 'blind test'
            print("Switching to blind test indexing mode")

        if 'test' in dataset_name:
            self.rebuild_indices()

        self.standardization_dict = misc_data_dict['standardization_dict']

    def process_new_dataset(self, new_dataset_name):
        self.load_chunks()
        self.dataset_type = 'molecule' if 'qm9' in new_dataset_name.lower() else 'crystal'
        self.rebuild_indices()

        if 'qm9' in new_dataset_name.lower():
            self.mol_to_crystal_dict, self.crystal_to_mol_dict, self.molecules_in_crystals_dict = None, None, None
            pass
        else:
            self.asymmetric_unit_analysis()

        self.get_dataset_standardization_statistics()

        misc_data_dict = {
            'standardization_dict': self.standardization_dict
        }

        np.save(self.datasets_path + 'misc_data_for_' + new_dataset_name, misc_data_dict)
        ints = np.random.choice(min(len(self.dataset), 10000), min(len(self.dataset), 10000), replace=False)
        self.dataset.loc[ints].to_pickle(self.datasets_path + 'test_' + new_dataset_name + '.pkl')

        del misc_data_dict, self.mol_to_crystal_dict, self.crystal_to_mol_dict, self.standardization_dict, self.molecules_in_crystals_dict  # free up some memory
        self.dataset.to_pickle(self.datasets_path + new_dataset_name + '.pkl')

    def asymmetric_unit_analysis(self):
        """
        for each crystal
        for each molecule in Z' molecules
        1) build periodic lattice
        2) identify "canonical" asymmetric unit aka canonical conformer
        3) compute pose relative to standardized initial condition
        -: manage issues of symmetry
        -: note each Z' molecule will be independently featurized

        """
        print("Parameterizing Crystals")

        mol_position_list = []
        mol_orientation_list = []
        handedness_list = []
        well_defined_asym_unit = []
        canonical_conformer_coords_list = []

        # do the parameterization in batches of 1000
        chunk_size = 1000
        n_chunks = int(np.ceil(len(self.dataset) / chunk_size))
        for i in tqdm(range(n_chunks)):
            chunk = self.dataset.iloc[i * chunk_size:(i + 1) * chunk_size]
            # symmetry_multiplicity = torch.tensor([crystal['crystal_symmetry_multiplicity'] for ind, crystal in chunk.iterrows() for _ in range(int(crystal['crystal_z_prime']))], dtype=torch.int, device=self.device)
            # final_coords_list = [torch.tensor(crystal['atom_coordinates'][z_ind], dtype=torch.float32, device=self.device) for _, crystal in chunk.iterrows() for z_ind in range(int(crystal['crystal_z_prime']))]
            T_fc_list = torch.tensor(np.stack([crystal['crystal_fc_transform'] for ind, crystal in chunk.iterrows() for _ in range(int(crystal['crystal_z_prime']))]), dtype=torch.float32, device=self.device)
            # T_cf_list = torch.tensor(np.stack([crystal['crystal_cf_transform'] for ind, crystal in chunk.iterrows() for _ in range(int(crystal['crystal_z_prime']))]), dtype=torch.float32, device=self.device)
            # sym_ops_list = [torch.tensor(crystal['crystal_symmetry_operators'], dtype=torch.float32, device=self.device) for ind, crystal in chunk.iterrows() for _ in range(int(crystal['crystal_z_prime']))]

            # build unit cell for each molecule (separately for each Z')
            unit_cells_list = build_unit_cell(
                symmetry_multiplicity=torch.tensor([crystal['crystal_symmetry_multiplicity'] for ind, crystal in chunk.iterrows() for _ in range(int(crystal['crystal_z_prime']))], dtype=torch.int, device=self.device),
                final_coords_list=[torch.tensor(crystal['atom_coordinates'][z_ind], dtype=torch.float32, device=self.device) for _, crystal in chunk.iterrows() for z_ind in range(int(crystal['crystal_z_prime']))],
                T_fc_list=T_fc_list,
                T_cf_list=torch.tensor(np.stack([crystal['crystal_cf_transform'] for ind, crystal in chunk.iterrows() for _ in range(int(crystal['crystal_z_prime']))]), dtype=torch.float32, device=self.device),
                sym_ops_list=[torch.tensor(crystal['crystal_symmetry_operators'], dtype=torch.float32, device=self.device) for ind, crystal in chunk.iterrows() for _ in range(int(crystal['crystal_z_prime']))]
            )

            # analyze the cell
            mol_position_list_i, mol_orientation_list_i, handedness_list_i, well_defined_asym_unit_i, canonical_conformer_coords_list_i = \
                batch_asymmetric_unit_pose_analysis_torch(
                    unit_cells_list,
                    torch.tensor([crystal['crystal_space_group_number'] for ind, crystal in chunk.iterrows() for _ in range(int(crystal['crystal_z_prime']))], dtype=torch.int, device=self.device),
                    self.asym_unit_dict,
                    T_fc_list,
                    enforce_right_handedness=False,
                    rotation_basis='spherical',
                    return_asym_unit_coords=True
                )

            mol_position_list.extend(mol_position_list_i.cpu().detach().numpy())
            mol_orientation_list.extend(mol_orientation_list_i.cpu().detach().numpy())
            handedness_list.extend(handedness_list_i.cpu().detach().numpy())
            well_defined_asym_unit.extend(well_defined_asym_unit_i)
            canonical_conformer_coords_list.extend(canonical_conformer_coords_list_i)

        assert len(mol_position_list) == len(self.mol_to_crystal_dict), "Molecule indexing failure in chunking system"

        mol_position_list = np.stack(mol_position_list)
        mol_orientation_list = np.stack(mol_orientation_list)
        handedness_list = np.stack(handedness_list)

        # write results to the dataset
        (centroids_x_list, centroids_y_list, centroids_z_list,
         orientations_theta_list, orientations_phi_list, orientations_r_list,
         handednesses_list, validity_list, coordinates_list) = [[[] for _ in range(len(self.dataset))] for _ in range(9)]

        mol_to_ident_dict = {ident: ind for ind, ident in enumerate(self.dataset['crystal_identifier'])}

        for i, identifier in enumerate(tqdm(self.crystal_to_mol_dict.keys())):  # index molecules with their respective crystals
            df_index = mol_to_ident_dict[identifier]
            (centroids_x, centroids_y, centroids_z, orientations_theta,
             orientations_phi, orientations_r, handedness, well_defined, coords) = [], [], [], [], [], [], [], [], []

            for mol_ind in self.crystal_to_mol_dict[identifier]:
                centroids_x.append(mol_position_list[mol_ind, 0])
                centroids_y.append(mol_position_list[mol_ind, 1])
                centroids_z.append(mol_position_list[mol_ind, 2])

                orientations_theta.append(mol_orientation_list[mol_ind, 0])
                orientations_phi.append(mol_orientation_list[mol_ind, 1])
                orientations_r.append(mol_orientation_list[mol_ind, 2])

                handedness.append(handedness_list[mol_ind])
                well_defined.append(well_defined_asym_unit[mol_ind])
                coords.append(canonical_conformer_coords_list[mol_ind].cpu().detach().numpy())

            centroids_x_list[df_index] = centroids_x
            centroids_y_list[df_index] = centroids_y
            centroids_z_list[df_index] = centroids_z

            orientations_theta_list[df_index] = orientations_theta
            orientations_phi_list[df_index] = orientations_phi
            orientations_r_list[df_index] = orientations_r

            handednesses_list[df_index] = handedness
            validity_list[df_index] = well_defined
            coordinates_list[df_index] = coords

        self.dataset['asymmetric_unit_centroid_x'] = centroids_x_list
        self.dataset['asymmetric_unit_centroid_y'] = centroids_y_list
        self.dataset['asymmetric_unit_centroid_z'] = centroids_z_list

        self.dataset['asymmetric_unit_rotvec_theta'] = orientations_theta_list
        self.dataset['asymmetric_unit_rotvec_phi'] = orientations_phi_list
        self.dataset['asymmetric_unit_rotvec_r'] = orientations_r_list

        self.dataset['asymmetric_unit_handedness'] = handednesses_list
        self.dataset['asymmetric_unit_is_well_defined'] = validity_list
        self.dataset['atom_coordinates'] = coordinates_list

        # check that all the crystals have the correct number of Z'
        # for a small dataset, getting all of them right could be a fluke
        # for a large one, if the below checks out, very likely we haven't screwed up the indexing
        assert all([len(thing) == thing2 for thing, thing2 in zip(handednesses_list, self.dataset['crystal_z_prime'])]), "Error with asymmetric indexing and/or symmetry multiplicity"
        assert all([len(self.dataset['atom_coordinates'][ii][0]) == self.dataset['molecule_num_atoms'][ii][0] for ii in range(len(self.dataset))]), "Error with coordinates indexing"

        # identify whether crystal symmetry ops exactly agree with standards
        # this should be included in 'space group setting' but is sometimes missed
        rand_mat = np.logspace(-3, 3, 16).reshape(4, 4)
        sym_ops_agreement = np.zeros(len(self.dataset))
        for i in range(len(self.dataset)):
            sym_op = np.stack(self.dataset['crystal_symmetry_operators'].loc[i])
            std_op = np.stack(SYM_OPS[self.dataset['crystal_space_group_number'].loc[i]])

            if len(sym_op) == len(std_op):  # if the multiplicity is different, it is certainly nonstandard
                # sort these in a canonical way to extract a fingerprint
                # multiply them with a wacky matrix and order them by the sums

                sample_args = np.argsort((sym_op * rand_mat).sum((1, 2)))
                sym_args = np.argsort((std_op * rand_mat).sum((1, 2)))

                sym_ops_agreement[i] = np.prod(sym_op[sample_args] == std_op[sym_args])
            else:
                sym_ops_agreement[i] = 0

        self.dataset['crystal_symmetry_operations_are_standard'] = sym_ops_agreement

    def get_dataset_standardization_statistics(self):
        """
        get mean and std deviation for all int and float features
        for crystals, molecules, and atoms
        """
        print("Getting Dataset Statistics")
        std_dict = {}

        for column in tqdm(self.dataset.columns):
            values = None
            if column[:4] == 'atom':
                if isinstance(self.dataset[column], list):
                    values = np.concatenate([atoms for atoms_lists in self.dataset[column] for atoms in atoms_lists])
                else:
                    values = np.concatenate(self.dataset[column])
            elif column[:8] == 'molecule':
                if isinstance(self.dataset[column], list):
                    values = np.concatenate(self.dataset[column])
                else:
                    values = self.dataset[column]
            elif column[:7] == 'crystal':
                values = self.dataset[column]
            elif column[:15] == 'asymmetric_unit':
                values = np.concatenate(self.dataset[column])  # one for each Z' molecule

            if values is not None:
                if not isinstance(values[0], str):  # not some string - check here since it crashes issubdtype below
                    if values.ndim > 1:  # intentionally leaving out e.g., coordinates, principal axes
                        pass
                    elif values.dtype == bool:
                        pass
                    # this is clunky but np.issubdtype is too sensitive
                    elif ((values.dtype == np.float32) or (values.dtype == np.float64)
                          or (values.dtype == float) or (values.dtype == int)
                          or (values.dtype == np.int8) or (values.dtype == np.int16)
                          or (values.dtype == np.int32) or (values.dtype == np.int64)):
                        std_dict[column] = [values.mean(), values.std()]
                    else:
                        pass

        self.standardization_dict = std_dict

    def generate_mol2crystal_mapping(self):
        """
        some crystals have multiple molecules, and we do batch analysis of molecules with a separate indexing scheme
        connect the crystal identifier-wise and mol-wise indexing with the following dicts
        """
        # print("Generating mol-to-crystal mapping")
        mol_index = 0
        crystal_to_mol_dict = {}
        mol_to_crystal_dict = {}
        for index, identifier in enumerate(self.dataset['crystal_identifier']):
            crystal_to_mol_dict[identifier] = []
            for _ in range(int(self.dataset['crystal_z_prime'][index])):  # assumes integer Z'
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
        # print("getting unique molecule fingerprints")
        fps = np.concatenate(self.dataset['molecule_fingerprint'])
        unique_fps, inverse_map = np.unique(fps, axis=0, return_inverse=True)
        molecules_in_crystals_dict = {
            unique.tobytes(): [] for unique in unique_fps
        }
        for ind, mapping in enumerate(inverse_map):  # we record the molecule inex for each unique molecular fingerprint
            molecules_in_crystals_dict[unique_fps[mapping].tobytes()].append(ind)

        return molecules_in_crystals_dict

    def get_identifier_duplicates(self):
        """
        by CSD identifier
        CSD entries with numbers on the end are subsequent additions to the same crystal
        often polymorphs or repeat  measurements

        option for grouping identifier by blind test sample submission
        """
        print('getting identifier duplicates')

        if self.mode == 'standard':  #
            crystal_to_identifier = {}
            for i in tqdm(range(len(self.dataset['crystal_identifier']))):
                item = self.dataset['crystal_identifier'][i]
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
            for i in tqdm(range(len(self.dataset['crystal_identifier']))):
                item = self.dataset['crystal_identifier'][i]
                for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
                    if blind_test_targets[-1 - j] in item:
                        crystal_to_identifier[blind_test_targets[-1 - j]].append(i)
                        break
        else:
            assert False, f"No such mode as {self.mode}"

        # delete identifiers with only one entry, despite having an index attached
        duplicate_lists = [crystal_to_identifier[key] for key in crystal_to_identifier.keys() if len(crystal_to_identifier[key]) > 1]
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
        #
        # blind_test_identifiers = [
        #     'OBEQUJ', 'OBEQOD', 'OBEQET', 'XATJOT', 'OBEQIX', 'KONTIQ',
        #     'NACJAF', 'XAFPAY', 'XAFQON', 'XAFQIH', 'XAFPAY01', 'XAFPAY02', 'XAFPAY03', 'XAFPAY04',
        #     "COUMAR01", "COUMAR02", "COUMAR10", "COUMAR11", "COUMAR12", "COUMAR13",
        #     "COUMAR14", "COUMAR15", "COUMAR16", "COUMAR17",  # Z'!=1 or some other weird thing
        #     "COUMAR18", "COUMAR19"
        # ]

        # test_conditions = [
        #     ['molecule_mass', 'range', [0, 300]],
        #     ['crystal_space_group_setting', 'not_in', [2]],
        #     ['crystal_space_group_number', 'in', [1, 2, 14, 19]],
        #     ['atom_atomic_numbers', 'range', [0, 20]],
        #     ['crystal_is_organic', 'in', [True]],
        #     ['molecule_is_symmetric_top', 'not_in', [True]]
        # ]

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
        if condition_type == 'range':
            # check fo the values to be in the range (inclusive)
            assert condition_range[1] > condition_range[0], "Range condition must be set low to high"
            if 'crystal' in condition_key:
                bad_inds = np.concatenate((
                    np.argwhere(np.asarray(self.dataset[condition_key]) > condition_range[1]),
                    np.argwhere(np.asarray(self.dataset[condition_key]) < condition_range[0])
                ))[:, 0]
            elif condition_key[:4] == 'atom':
                max_values = np.asarray([np.amax(np.concatenate(atoms)) for atoms in self.dataset[condition_key]])
                min_values = np.asarray([np.amin(np.concatenate(atoms)) for atoms in self.dataset[condition_key]])
                bad_inds = np.concatenate((
                    np.argwhere(max_values > condition_range[1]),
                    np.argwhere(min_values < condition_range[0])
                ))[:, 0]
            elif 'molecule' in condition_key or 'asymmetric_unit' in condition_key:
                max_values = np.asarray([np.amax(mols) for mols in self.dataset[condition_key]])
                min_values = np.asarray([np.amin(mols) for mols in self.dataset[condition_key]])
                bad_inds = np.concatenate((
                    np.argwhere(max_values > condition_range[1]),
                    np.argwhere(min_values < condition_range[0])
                ))[:, 0]

        elif condition_type == 'in':
            # check for where the data is not equal to the explicitly enumerated range elements to be included
            if 'crystal' in condition_key:
                bad_inds = np.argwhere([
                    thing not in condition_range for thing in self.dataset[condition_key]
                ])[:, 0]
            elif 'molecule' in condition_key or 'asymmetric_unit' in condition_key:
                bad_inds = np.argwhere([
                    cond not in mol for mol in self.dataset[condition_key] for cond in condition_range
                ])[:, 0]
            elif condition_key[:4] == 'atom':
                bad_inds = np.concatenate([
                    np.argwhere([cond not in np.concatenate(atoms) for atoms in self.dataset[condition_key]])[:, 0]
                    for cond in condition_range]
                )

        elif condition_type == 'not_in':
            # check for where the data is equal to the explicitly enumerated elements to be excluded
            if 'crystal' in condition_key:
                bad_inds = np.argwhere([
                    thing in condition_range for thing in self.dataset[condition_key]
                ])[:, 0]
            elif 'molecule' in condition_key or 'asymmetric_unit' in condition_key:
                bad_inds = np.argwhere([
                    cond in mol for mol in self.dataset[condition_key] for cond in condition_range
                ])[:, 0]
            elif condition_key[:4] == 'atom':
                bad_inds = np.concatenate([
                    np.argwhere([cond in np.concatenate(atoms) for atoms in self.dataset[condition_key]])[:, 0]
                    for cond in condition_range]
                )
        else:
            assert False, f"{condition_type} is not an implemented dataset filter condition"

        print(f'{condition} filtered {len(bad_inds)} samples')

        return bad_inds

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
        print('selecting representative structures from duplicate polymorphs')
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
        print('selecting representative structures from duplicate molecules')
        index_to_identifier_dict = {ident: ind for ind, ident in enumerate(self.dataset['crystal_identifier'])}
        bad_inds = []
        for ind, (key, value) in enumerate(self.molecules_in_crystals_dict.items()):
            if len(value) > 1:  # if there are multiple molecules
                mol_inds = self.molecules_in_crystals_dict[key]  # identify their mol indices
                crystal_identifiers = [self.mol_to_crystal_dict[ind] for ind in mol_inds]  # identify their crystal identifiers
                crystal_inds = [index_to_identifier_dict[identifier] for identifier in crystal_identifiers]
                sampled_ind = np.random.choice(crystal_inds, size=1)
                crystal_inds.remove(sampled_ind)  # remove the good one
                bad_inds.extend(crystal_inds)  # delete unselected polymorphs from the dataset

        return bad_inds

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def __len__(self):
        return len(self.datapoints)


if __name__ == '__main__':
    # miner = DataManager(device='cpu', datasets_path=r"D:\crystal_datasets/", chunks_path=r"D:\crystal_datasets/featurized_chunks/")
    # miner.process_new_dataset('dataset')

    # miner = DataManager(device='cpu', datasets_path=r"D:\crystal_datasets/", chunks_path=r"D:\crystal_datasets/featurized_chunks/")
    # miner.process_new_dataset('dataset')

    miner = DataManager(device='cpu', datasets_path=r"D:\crystal_datasets/", chunks_path=r"D:\crystal_datasets/QM9_chunks/")
    miner.process_new_dataset('qm9_molecules_dataset')
    #
    # miner = DataManager(device='cpu', datasets_path=r"D:\crystal_datasets/", chunks_path=r"D:\crystal_datasets/acridin_chunks/")
    # miner.process_new_dataset('acridin_dataset')
    # '''filtering test'''
    # test_conditions = [
    #     ['molecule_mass', 'range', [0, 300]],
    #     ['crystal_space_group_setting', 'not_in', [2]],
    #     ['crystal_space_group_number', 'in', [1, 2, 14, 19]],
    #     ['atom_atomic_numbers', 'range', [0, 20]],
    #     ['crystal_is_organic', 'in', [True]],
    #     ['molecule_is_symmetric_top', 'not_in', [True]]
    # ]
    #
    # miner.load_dataset_for_modelling(dataset_name='new_dataset.pkl',
    #                                  filter_conditions=test_conditions, filter_polymorphs=True, filter_duplicate_molecules=True)
    #

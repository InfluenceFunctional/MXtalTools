import pandas as pd
import torch
import tqdm
import os
import numpy as np

from common.utils import delete_from_dataframe
from constants.asymmetric_units import asym_unit_dict
from constants.space_group_info import SYM_OPS
from crystal_building.utils import build_unit_cell, batch_asymmetric_unit_pose_analysis_torch


class DataManager():
    def __init__(self, datasets_path, device='cpu', chunks_path=None, seed=0):
        self.misc_data_dict = None
        self.standardization_dict = None
        self.crystal_to_mol_dict = None
        self.mol_to_crystal_dict = None
        self.molecules_in_crystals_dict = None
        self.dataset = None

        self.datasets_path = datasets_path
        self.chunks_path = chunks_path
        self.device = device

        np.random.seed(seed=seed)  # for certain random sampling ops

        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(device)

    def load_chunks(self):
        os.chdir(self.chunks_path)
        chunks = os.listdir()
        num_chunks = len(chunks)
        print(f'Collecting {num_chunks} dataset chunks')
        self.dataset = pd.concat([pd.read_pickle(chunk) for chunk in chunks[:25]], ignore_index=True)
        self.dataset = self.dataset.reset_index().drop(columns='index')

    def load_dataset_for_modelling(self, dataset_name,
                                   filter_conditions=None, filter_polymorphs=False, filter_duplicate_molecules=False):

        self.load_dataset_and_misc_data(dataset_name)

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

    def rebuild_indices(self):
        self.crystal_to_mol_dict, self.mol_to_crystal_dict = \
            self.generate_mol2crystal_mapping()

        self.molecules_in_crystals_dict = \
            self.identify_unique_molecules_in_crystals()

    def load_dataset_and_misc_data(self, dataset_name):
        self.dataset = pd.read_pickle(self.datasets_path + dataset_name)
        misc_data_dict = np.load(self.datasets_path + 'misc_data_for_' + dataset_name + '.npy', allow_pickle=True).item()

        self.crystal_to_mol_dict = misc_data_dict['crystal_to_mol_dict']
        self.mol_to_crystal_dict = misc_data_dict['mol_to_crystal_dict']
        self.molecules_in_crystals_dict = misc_data_dict['molecules_in_crystals_dict']
        self.standardization_dict = misc_data_dict['standardization_dict']

    def process_new_dataset(self, new_dataset_name):
        self.load_chunks()
        self.rebuild_indices()
        self.asymmetric_unit_analysis()
        self.get_dataset_standardization_statistics()

        misc_data_dict = {
            'crystal_to_mol_dict': self.crystal_to_mol_dict,
            'mol_to_crystal_dict': self.mol_to_crystal_dict,
            'molecules_in_crystals_dict': self.molecules_in_crystals_dict,
            'standardization_dict': self.standardization_dict
        }

        np.save(self.datasets_path + 'misc_data_for_' + new_dataset_name, misc_data_dict)
        self.dataset.to_pickle(self.datasets_path + new_dataset_name)
        ints = np.random.choice(min(len(self.dataset), 1000), min(len(self.dataset), 1000), replace=False)
        self.dataset.loc[ints].to_pickle(self.datasets_path + 'test_' + new_dataset_name)

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

        # symmetry_multiplicity = torch.tensor([crystal['crystal_symmetry_multiplicity'] for ind, crystal in self.dataset.iterrows() for _ in range(int(crystal['crystal_z_prime']))], dtype=torch.int, device=self.device)
        # final_coords_list = [torch.tensor(crystal['atom_coordinates'][z_ind], dtype=torch.float32, device=self.device) for ind, crystal in self.dataset.iterrows() for z_ind in range(int(crystal['crystal_z_prime']))]
        T_fc_list = torch.tensor(np.stack([crystal['crystal_fc_transform'] for ind, crystal in self.dataset.iterrows() for _ in range(int(crystal['crystal_z_prime']))]), dtype=torch.float32, device=self.device)
        # T_cf_list = torch.tensor(np.stack([crystal['crystal_cf_transform'] for ind, crystal in self.dataset.iterrows() for _ in range(int(crystal['crystal_z_prime']))]), dtype=torch.float32, device=self.device)
        # sym_ops_list = [torch.tensor(crystal['crystal_symmetry_operators'], dtype=torch.float32, device=self.device) for ind, crystal in self.dataset.iterrows() for _ in range(int(crystal['crystal_z_prime']))]

        # build unit cell for each molecule (separately for each Z')
        unit_cells_list = build_unit_cell(
            symmetry_multiplicity=torch.tensor([crystal['crystal_symmetry_multiplicity'] for ind, crystal in self.dataset.iterrows() for _ in range(int(crystal['crystal_z_prime']))], dtype=torch.int, device=self.device),
            final_coords_list=[torch.tensor(crystal['atom_coordinates'][z_ind], dtype=torch.float32, device=self.device) for ind, crystal in self.dataset.iterrows() for z_ind in range(int(crystal['crystal_z_prime']))],
            T_fc_list=T_fc_list,
            T_cf_list=torch.tensor(np.stack([crystal['crystal_cf_transform'] for ind, crystal in self.dataset.iterrows() for _ in range(int(crystal['crystal_z_prime']))]), dtype=torch.float32, device=self.device),
            sym_ops_list=[torch.tensor(crystal['crystal_symmetry_operators'], dtype=torch.float32, device=self.device) for ind, crystal in self.dataset.iterrows() for _ in range(int(crystal['crystal_z_prime']))]
        )

        # analyze the cell
        mol_position_list, mol_orientation_list, handedness_list, well_defined_asym_unit, canonical_conformer_coords_list = \
            batch_asymmetric_unit_pose_analysis_torch(
                unit_cells_list,
                torch.tensor([crystal['crystal_space_group_number'] for ind, crystal in self.dataset.iterrows() for _ in range(int(crystal['crystal_z_prime']))], dtype=torch.int, device=self.device),
                self.asym_unit_dict,
                T_fc_list,
                enforce_right_handedness=False,
                rotation_basis='spherical',
                return_asym_unit_coords=True
            )

        mol_position_list = mol_position_list.cpu().detach().numpy()
        mol_orientation_list = mol_orientation_list.cpu().detach().numpy()
        handedness_list = handedness_list.cpu().detach().numpy()

        # write results to the dataset
        (centroids_x_list, centroids_y_list, centroids_z_list,
         orientations_theta_list, orientations_phi_list, orientations_r_list,
         handednesses_list, validity_list, coordinates_list) = [[[] for _ in range(len(self.dataset))] for _ in range(9)]

        for i, identifier in enumerate(self.crystal_to_mol_dict.keys()):  # index molecules with their respective crystals
            df_index = self.dataset.loc[self.dataset['crystal_identifier'] == identifier].index[0]
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

        for column in tqdm.tqdm(self.dataset.columns):
            values = None
            if column[:4] == 'atom':
                values = np.concatenate([atoms for atoms_lists in self.dataset[column] for atoms in atoms_lists])
            elif column[:8] == 'molecule':
                values = np.concatenate(self.dataset[column])
            elif column[:7] == 'crystal':
                values = self.dataset[column]
            elif 'asymmetric_unit' in column:
                values = np.concatenate(self.dataset[column])  # one for each Z' molecule

            if values is not None:
                if not isinstance(values[0], str):  # not some string - check here since it crashes issubdtype below
                    if values.ndim > 1:  # intentionally leaving out e.g., coordinates, principal axes
                        pass
                    elif values.dtype == bool:
                        pass
                    # this is clunky but np.issubdtype is too sensitive
                    elif (values.dtype == float) or (values.dtype == int) or (values.dtype == np.int8) or (values.dtype == np.int16) or (values.dtype == np.int32) or (values.dtype == np.int64):
                        std_dict[column] = [values.mean(), values.std()]
                    else:
                        pass

    def generate_mol2crystal_mapping(self):
        """
        some crystals have multiple molecules, and we do batch analysis of molecules with a separate indexing scheme
        connect the crystal identifier-wise and mol-wise indexing with the following dicts
        """
        #print("Generating mol-to-crystal mapping")
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
        #print("getting unique molecule fingerprints")
        fps = np.concatenate(self.dataset['molecule_fingerprint'])
        unique_fps, inverse_map = np.unique(fps, axis=0, return_inverse=True)
        molecules_in_crystals_dict = {
            unique.tobytes(): [] for unique in unique_fps
        }
        for ind, map in enumerate(inverse_map):  # we record the molecule inex for each unique molecular fingerprint
            molecules_in_crystals_dict[unique_fps[map].tobytes()].append(ind)

        return molecules_in_crystals_dict

    def get_identifier_duplicates(self, mode='standard'):
        """
        by CSD identifier
        CSD entries with numbers on the end are subsequent additions to the same crystal
        often polymorphs or repeat  measurements

        option for grouping identifier by blind test sample submission
        """
        print('getting identifier duplicates')

        if mode == 'standard':
            all_identifiers = {}
            for i in tqdm.tqdm(range(len(self.dataset['crystal_identifier']))):
                item = self.dataset['crystal_identifier'][i]
                if item[-1].isdigit():
                    item = item[:-2]  # cut off trailing digits, if any
                if item not in all_identifiers.keys():
                    all_identifiers[item] = []
                all_identifiers[item].append(i)
        elif mode == 'blind_test':  # todo untested
            blind_test_targets = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                                  'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
                                  'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX']

            all_identifiers = {key: [] for key in blind_test_targets}
            for i in tqdm.tqdm(range(len(self.dataset['crystal_identifier']))):
                item = self.dataset['crystal_identifier'][i]
                for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
                    if blind_test_targets[-1 - j] in item:
                        all_identifiers[blind_test_targets[-1 - j]].append(i)
                        break

        # delete identifiers with only one entry, despite having an index attached
        duplicate_lists = [all_identifiers[key] for key in all_identifiers.keys() if len(all_identifiers[key]) > 1]
        duplicate_list_extension = []
        for d_list in duplicate_lists:
            duplicate_list_extension.extend(d_list)

        duplicate_groups = {}
        for key in all_identifiers.keys():
            if len(all_identifiers[key]) > 1:
                duplicate_groups[key] = all_identifiers[key]

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
                    np.argwhere(self.dataset[condition_key] > condition_range[1]),
                    np.argwhere(self.dataset[condition_key] < condition_range[0])
                ))[:, 0]
            elif 'atom' in condition_key:
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
            if 'crystal' in condition_key:  # todo filters need to interact
                bad_inds = np.argwhere([
                    thing not in condition_range for thing in self.dataset[condition_key]
                ])[:, 0]
            elif 'molecule' in condition_key or 'asymmetric_unit' in condition_key:
                bad_inds = np.argwhere([
                    cond not in mol for mol in self.dataset[condition_key] for cond in condition_range
                ])[:, 0]
            elif condition_key[:4] == 'atom':
                bad_inds = np.argwhere([
                    cond not in np.concatenate(atoms) for atoms in self.dataset[condition_key] for cond in condition_range
                ])[:, 0]

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
                bad_inds = np.argwhere([
                    cond in np.concatenate(atoms) for atoms in self.dataset[condition_key] for cond in condition_range
                ])[:, 0]
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
        bad_inds = []
        for key in self.molecules_in_crystals_dict.keys():
            mol_inds = self.molecules_in_crystals_dict[key]
            crystal_identifiers = [self.mol_to_crystal_dict[ind] for ind in mol_inds]
            crystal_inds = [self.dataset.loc[self.dataset['crystal_identifier'] == identifier].index[0] for identifier in crystal_identifiers]
            sampled_ind = np.random.choice(crystal_inds, size=1)
            crystal_inds.remove(sampled_ind)  # remove the good one
            bad_inds.extend(crystal_inds)  # delete unselected polymorphs from the dataset

        return bad_inds


if __name__ == '__main__':
    miner = DataManager(device='cuda', datasets_path=r"D:\crystal_datasets/", chunks_path=r"D:\crystal_datasets/featurized_chunks/")
    miner.process_new_dataset('new_dataset.pkl')

    test_conditions = [
        ['molecule_mass', 'range', [0, 300]],
        ['crystal_space_group_setting', 'not_in', [2]],
        ['crystal_space_group_number', 'in', [1, 2, 14, 19]],
        ['atom_atomic_numbers', 'range', [0, 20]],
        ['crystal_is_organic', 'in', [True]],
        ['molecule_is_symmetric_top', 'not_in', [True]]
    ]

    miner.load_dataset_for_modelling(dataset_name = 'new_dataset.pkl',
                                     filter_conditions=test_conditions, filter_polymorphs=True, filter_duplicate_molecules=True)

    aa = 1
import pandas as pd
from constants.asymmetric_units import asym_unit_dict
import tqdm
import os
import numpy as np
import torch
from scipy.spatial.distance import cdist


class DataManager():
    def __init__(self, datasets_path, device='cpu', chunks_path=None):
        self.crystal_to_mol_dict = None
        self.mol_to_crystal_dict = None
        self.molecules_in_crystals_dict = None
        self.dataset = None

        self.datasets_path = datasets_path
        self.chunks_path = chunks_path
        self.device = device

    def load_chunks(self):
        os.chdir(self.chunks_path)
        chunks = os.listdir()
        num_chunks = len(chunks)
        print(f'Collecting {num_chunks} dataset chunks')
        self.dataset = pd.concat([pd.read_pickle(chunk) for chunk in chunks], ignore_index=True)
        self.dataset = self.dataset.reset_index().drop(columns='index')

    def process_new_dataset(self, new_dataset_name):
        self.load_chunks()
        self.generate_mol2crystal_mapping()
        self.identify_unique_molecules_in_crystals()
        identifier_duplicate_lists, identifier_duplicate_list_extension, identifier_duplicate_groups = (
            self.get_identifier_duplicates())
        standardization_dict = self.get_dataset_standardization_statistics()

        np.save(self.datasets_path + 'stats_' + new_dataset_name)
        self.dataset.to_pickle(self.datasets_path + new_dataset_name)
        ints = np.random.choice(0, min(len(self.dataset), 1000), replace=False)
        self.dataset.loc[ints].to_pickle(self.datasets_path + '_test_' + new_dataset_name)
        aa = 0

    def get_dataset_standardization_statistics(self):
        """
        get mean and std deviation for all int and float features
        for crystals, molecules, and atoms
        """
        std_dict = {}

        for column in tqdm.tqdm(self.dataset.columns):
            print(column)
            value = None
            if column[:4] == 'atom':
                value = np.concatenate([atoms for atoms_lists in self.dataset[column] for atoms in atoms_lists])
            elif column[:8] == 'molecule':
                value = np.concatenate(self.dataset[column])
            elif column[:7] == 'crystal':
                value = self.dataset[column]

            if value is not None:
                if not isinstance(value[0], str):  # not some string - check here since it crashes issubdtype below
                    if value.ndim > 1:  # intentionally leaving out e.g., coordinates, principal axes
                        pass
                    elif value.dtype == bool:
                        pass
                    elif (value.dtype == float) or (value.dtype == int) or np.issubdtype(value, np.integer):
                        std_dict[column] = [value.mean(), value.std()]
                    else:
                        pass

        return std_dict

    def generate_mol2crystal_mapping(self):
        """
        some crystals have multiple molecules, and we do batch analysis of molecules with a separate indexing scheme
        connect the crystal identifier-wise and mol-wise indexing with the following dicts
        """
        mol_index = 0
        self.crystal_to_mol_dict = {}
        self.mol_to_crystal_dict = {}
        for index, identifier in enumerate(self.dataset['crystal_identifier']):
            self.crystal_to_mol_dict[identifier] = []
            for _ in range(int(self.dataset['crystal_z_prime'][index])):  # assumes integer Z'
                self.crystal_to_mol_dict[identifier].append(mol_index)
                self.mol_to_crystal_dict[mol_index] = identifier
                mol_index += 1

    def identify_unique_molecules_in_crystals(self):
        """
        identify all exactly unique molecules (up to mol fingerprint)
        list their dataset indices in a dict

        at train time, we can use this to repeat sampling of identical molecules
        """
        fps = np.concatenate(self.dataset['molecule_fingerprint'])
        unique_fps, inverse_map = np.unique(fps, axis=0, return_inverse=True)
        self.molecules_in_crystals_dict = {
            unique.tobytes(): [] for unique in unique_fps
        }
        for ind, map in enumerate(tqdm.tqdm(inverse_map)):
            self.molecules_in_crystals_dict[unique_fps[map].tobytes()].append(ind)

    #
    # def filter_dataset(self):
    #     print('Filtering dataset starting from {} samples'.format(len(self.dataset)))
    #     ## filtering out unwanted characteristics
    #     bad_inds = []
    #
    #     #
    #     # # exclude samples with extremely close atoms
    #     # n_bad_inds = len(bad_inds)
    #     # for j in range(len(self.dataset)):
    #     #     coords = self.dataset['crystal reference cell coords'][j]
    #     #     coords = coords.reshape(coords.shape[0] * coords.shape[1],3)
    #     #     distmat = torch.cdist(torch.Tensor(coords), torch.Tensor(coords), p=2) + torch.eye(len(coords))
    #     #     if torch.amin(distmat) < 0.1:
    #     #         #print('bad')
    #     #         bad_inds.append(j)
    #     # print('overlapping atoms caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # samples with bad CSD-generated reference cells
    #     n_bad_inds = len(bad_inds)
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal reference cell coords']) == 'error')[:, 0])  # missing coordinates
    #     bad_inds.extend(np.argwhere(np.asarray(np.isnan(self.dataset['crystal asymmetric unit centroid x'])))[:, 0])  # missing orientation features
    #     print('bad coordinates caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     if self.exclude_blind_test_targets:
    #         # CSD blind test 5 and 6 targets
    #         blind_test_identifiers = [
    #             'OBEQUJ', 'OBEQOD', 'OBEQET', 'XATJOT', 'OBEQIX', 'KONTIQ',
    #             'NACJAF', 'XAFPAY', 'XAFQON', 'XAFQIH', 'XAFPAY01', 'XAFPAY02', 'XAFPAY03', 'XAFPAY04',
    #             "COUMAR01", "COUMAR02", "COUMAR10", "COUMAR11", "COUMAR12", "COUMAR13",
    #             "COUMAR14", "COUMAR15", "COUMAR16", "COUMAR17",  # Z'!=1 or some other weird thing
    #             "COUMAR18", "COUMAR19"
    #         ]
    #         blind_test_identifiers.remove('XATJOT')  # multi-component
    #         blind_test_identifiers.remove('XAFQON')  # multi-component
    #         blind_test_identifiers.remove('KONTIQ')  # multi-component
    #
    #         # samples with bad CSD-generated reference cells
    #         n_bad_inds = len(bad_inds)
    #         for j in range(len(self.dataset)):
    #             item = self.dataset['identifier'][j]  # do it this way to remove the target, including any of its polymorphs
    #             if item[-1].isdigit():
    #                 item = item[:-2]  # cut off trailing digits, if any
    #             if item in blind_test_identifiers:
    #                 bad_inds.append(j)
    #         print('Blind test targets caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # filter for when the asymmetric unit definition is nonstandard (returns more than one centroid) # todo check for edge centroids
    #     n_bad_inds = len(bad_inds)
    #     for ii in range(len(self.dataset['atom coords'])):
    #         sg_ind = self.dataset['crystal spacegroup number'][ii]
    #         if str(sg_ind) in asym_unit_dict.keys():  # only do this check if this sg_ind is already encoded in the asym unit dict
    #             unit_cell_coords = self.dataset['crystal reference cell coords'][ii]
    #             T_cf = np.linalg.inv(self.dataset['crystal fc transform'][ii])
    #             asym_unit = asym_unit_dict[str(int(sg_ind))]  # will only work for units which we have written down the parameterization for
    #             # identify which of the Z asymmetric units is canonical
    #             centroids_cartesian = unit_cell_coords.mean(-2)
    #             centroids_fractional = np.inner(T_cf, centroids_cartesian).T
    #             centroids_fractional -= np.floor(centroids_fractional)
    #             if torch.is_tensor(asym_unit):
    #                 asym_unit = asym_unit.cpu().detach().numpy()
    #             canonical_conformer_index = find_coord_in_box_np(centroids_fractional, asym_unit)
    #             if len(canonical_conformer_index) != 1:
    #                 bad_inds.append(ii)
    #     print('Non uniform asymmetric unit caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # when the molecule is too long
    #     # cases where the csd has the wrong number of molecules
    #     n_bad_inds = len(bad_inds)
    #     # self.config.max_molecule_radius
    #     mol_radii = np.asarray([np.amax(np.linalg.norm(coords - coords.mean(0), axis=-1)) for coords in self.dataset['atom coords']])
    #     bad_inds.extend(np.argwhere(np.asarray(mol_radii) > self.max_molecule_radius)[:, 0])
    #     print('molecule too long filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # cases where the csd has the wrong number of molecules
    #     n_bad_inds = len(bad_inds)
    #     lens = [len(item) for item in self.dataset['crystal reference cell coords']]
    #     bad_inds.extend(np.argwhere(np.asarray(lens != self.dataset['crystal z value']))[:, 0])
    #     print('improper CSD Z value filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # # cases where the symmetry ops disagree with the Z value
    #     # n_bad_inds = len(bad_inds)
    #     # lens = [len(item) for item in self.dataset['crystal symmetries']]
    #     # bad_inds.extend(np.argwhere(np.asarray(lens != self.dataset['crystal z value']))[:, 0])
    #     # print('improper sym ops multiplicity filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # exclude samples with atoms on special positions # todo eventually relax this
    #     n_bad_inds = len(bad_inds)
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal atoms on special positions'].ne([[] for _ in range(len(self.dataset))])))[:, 0])
    #     print('special positions filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # Z prime
    #     n_bad_inds = len(bad_inds)
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z prime']) > self.max_z_prime)[:, 0])  # self.config.max_z_prime))
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z prime']) < self.min_z_prime)[:, 0])  # self.config.min_z_prime))
    #     print('z prime filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # Z value
    #     n_bad_inds = len(bad_inds)
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z value']) > self.max_z_value)[:, 0])  # self.config.max_z_value))
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z value']) < self.min_z_value)[:, 0])  # self.config.min_z_value))
    #     print('z value filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # molecule num atoms
    #     n_bad_inds = len(bad_inds)
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule num atoms']) > self.max_num_atoms)[:, 0])  # self.config.max_molecule_size))
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule num atoms']) < self.min_num_atoms)[:, 0])  # self.config.min_molecule_size))
    #     print('molecule num atoms filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # heaviest atom
    #     n_bad_inds = len(bad_inds)
    #     heaviest_atoms = np.asarray([max(atom_z) for atom_z in self.dataset['atom Z']])
    #     bad_inds.extend(np.argwhere(heaviest_atoms > self.max_atomic_number)[:, 0])  # self.config.max_atomic_number))
    #     print('max atom size filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # too diffuse or too dense
    #     n_bad_inds = len(bad_inds)
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal packing coefficient']) > self.max_packing_coefficient)[:, 0])  # self.config.max_molecule_size))
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal packing coefficient']) < self.min_packing_coefficient)[:, 0])  # self.config.min_molecule_size))
    #     print('crystal packing coefficient filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # erroneous densities
    #     n_bad_inds = len(bad_inds)
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal calculated density']) == 0)[:, 0])  # self.config.max_molecule_size))
    #     print('0 density filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # too hot or too cold
    #     n_bad_inds = len(bad_inds)
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal temperature']) > self.max_temperature)[:, 0])
    #     bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal temperature']) < self.min_temperature)[:, 0])
    #     print('crystal temperature filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # supported space groups
    #     if self.include_sgs is not None:
    #         n_bad_inds = len(bad_inds)
    #         bad_inds.extend(np.argwhere([self.dataset['crystal spacegroup symbol'][i] not in self.include_sgs for i in range(len(self.dataset['crystal spacegroup symbol']))])[:, 0])
    #         print('spacegroup filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     if self.exclude_crystal_systems is not None:
    #         n_bad_inds = len(bad_inds)
    #         bad_inds.extend(np.argwhere([self.dataset['crystal system'][i] in self.exclude_crystal_systems for i in range(len(self.dataset['crystal system']))])[:, 0])
    #         print('crystal system filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     if self.include_pgs is not None:
    #         # filter by point group
    #         n_bad_inds = len(bad_inds)
    #         bad_inds.extend(np.argwhere([self.dataset['crystal point group'][i] not in self.include_pgs for i in range(len(self.dataset['crystal point group']))])[:, 0])  # missing coordinates
    #         print('unwanted point groups caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # molecule organic
    #     if not self.include_organic:
    #         n_bad_inds = len(bad_inds)
    #         bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal is organic'] == 'True'))[:, 0])
    #         print('organic filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # molecule organometallic
    #     if not self.include_organometallic:
    #         n_bad_inds = len(bad_inds)
    #         bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal is organometallic'] == 'True'))[:, 0])
    #         print('organometallic filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # molecule has disorder
    #     if self.exclude_disordered_crystals:
    #         n_bad_inds = len(bad_inds)
    #         bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal has disorder']))[:, 0])
    #         print('disorder filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # missing r factor
    #     if self.exclude_missing_r_factor:
    #         n_bad_inds = len(bad_inds)
    #         bad_inds.extend(np.asarray([i for i in range(len(self.dataset['crystal r factor'])) if self.dataset['crystal r factor'][i] is None]))
    #         print('missing r factor filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     if self.exclude_nonstandard_settings:
    #         # nonstandard spacegroup setings have inconsistent lattice definitions
    #         n_bad_inds = len(bad_inds)
    #         settings = np.asarray([self.dataset['crystal spacegroup setting'][i] for i in range(len(self.dataset))])
    #         bad_inds.extend(np.argwhere(settings != 1)[:, 0])
    #         print('nonstandard spacegroup setting filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))
    #
    #     # collate bad indices
    #     bad_inds = np.unique(bad_inds)
    #
    #     # apply filtering
    #     self.dataset = delete_from_dataframe(self.dataset, bad_inds)
    #     print("Filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))
    #
    # def filter_polymorphs(self):
    #     '''
    #     find duplicate examples and pick one representative per molecule
    #     :return:
    #     '''
    #     # use CSD identifiers to pick out polymorphs
    #     duplicate_lists, duplicate_list_extension, duplicate_groups = self.get_identifier_duplicates()  # issue - some of these are not isomorphic (i.e., different ionization), maybe ~2% from early tests
    #
    #     # duplicate_groups_identifiers = {ident:[self.dataset['identifier'][n] for n in duplicate_groups[ident]] for ident in duplicate_groups.keys()}
    #     # duplicate_groups_packings = {ident:[self.dataset['crystal packing coefficient'][n] for n in duplicate_groups[ident]] for ident in duplicate_groups.keys()}
    #
    #     # todo delete blind test and any extra CSP samples from the training dataset
    #
    #     # now, the 'representative structure' is the highest resolution structure which as the same space group as the oldest structure
    #     # we will add all others to 'bad_inds', and filter them out at our leisure
    #     print('selecting representative structures from duplicate groups')
    #     bad_inds = []
    #     for key in duplicate_groups.keys():
    #         # print(key)
    #         inds = duplicate_groups[key]
    #         space_groups = [self.dataset['crystal spacegroup symbol'][ind] for ind in inds]
    #         oldest_structure = inds[np.argmin([self.dataset['crystal date'][ind] for ind in inds])]  # get the dataset index for the oldest structure
    #         oldest_structure_SG = self.dataset['crystal spacegroup symbol'][oldest_structure]
    #         inds_agree_with_oldest_SG = np.argwhere([sg == oldest_structure_SG for sg in space_groups])[:, 0]
    #         inds_agree_with_oldest_SG = [inds[ind] for ind in inds_agree_with_oldest_SG]
    #         agreeing_ind_with_best_arg_factor = inds[np.argmin([self.dataset['crystal r factor'][ind] for ind in inds_agree_with_oldest_SG])]
    #         inds.remove(agreeing_ind_with_best_arg_factor)  # remove the good one
    #         bad_inds.extend(inds)  # delete residues from the dataset
    #
    #     self.dataset = delete_from_dataframe(self.dataset, bad_inds)

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
        elif mode == 'blind_test':
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


if __name__ == '__main__':
    miner = DataManager(device='cuda', datasets_path=r"D:\crystal_datasets/", chunks_path=r"D:\crystal_datasets/featurized_chunks/")
    miner.process_new_dataset('new_dataset.pkl')

from common.geometry_calculations import compute_fractional_transform
from common.utils import *
import matplotlib.pyplot as plt
import pandas as pd

from constants.asymmetric_units import asym_unit_dict
from crystal_building.utils import *
from pyxtal import symmetry
import matplotlib.colors as colors
import tqdm
import os


class DataManager():
    def __init__(self, dataset_path, config=None, collect_chunks=False, database='csd'):
        self.config = config
        if config is None:
            self.max_z_prime = 1
            self.min_z_prime = 1
            self.max_z_value = 50
            self.min_z_value = 1
            self.max_num_atoms = 1000
            self.min_num_atoms = 5
            self.max_molecule_radius = 1000
            self.max_atomic_number = 87
            self.max_packing_coefficient = 0.85
            self.min_packing_coefficient = 0.55
            self.include_organic = True
            self.include_organometallic = True
            self.exclude_disordered_crystals = False
            self.exclude_missing_r_factor = False
            self.exclude_polymorphs = False
            self.exclude_nonstandard_settings = False
            self.max_temperature = 1000
            self.min_temperature = -10
            self.include_sgs = None
            self.include_pgs = None
            self.exclude_crystal_systems = None
            self.exclude_blind_test_targets = False
        else:
            self.max_z_prime = config.max_z_prime
            self.min_z_prime = config.min_z_prime
            self.max_z_value = config.max_z_value
            self.min_z_value = config.min_z_value
            self.max_num_atoms = config.max_num_atoms
            self.min_num_atoms = config.min_num_atoms
            self.max_molecule_radius = config.max_molecule_radius
            self.max_atomic_number = config.max_atomic_number
            self.max_packing_coefficient = 0.85
            self.min_packing_coefficient = config.min_packing_coefficient
            self.include_organic = config.include_organic
            self.include_organometallic = config.include_organometallic
            self.exclude_disordered_crystals = config.exclude_disordered_crystals
            self.exclude_polymorphs = config.exclude_polymorphs
            if self.exclude_polymorphs:
                self.exclude_missing_r_factor = True
            else:
                self.exclude_missing_r_factor = config.exclude_missing_r_factor
            self.exclude_nonstandard_settings = config.exclude_nonstandard_settings
            self.max_temperature = config.max_crystal_temperature
            self.min_temperature = config.min_crystal_temperature
            self.include_sgs = config.include_sgs
            self.include_pgs = config.include_pgs
            self.exclude_crystal_systems = config.exclude_crystal_systems
            self.exclude_blind_test_targets = config.exclude_blind_test_targets

        self.dataset_path = dataset_path
        self.collect_chunks = collect_chunks
        self.database = database

    def load_for_modelling(self, return_dataset=False, save_dataset=True):
        self.dataset = pd.read_pickle(self.dataset_path)
        self.dataset_keys = list(self.dataset.columns)
        self.filter_dataset()
        if self.exclude_polymorphs:
            self.filter_polymorphs()
        self.datasetPath = 'datasets/dataset'
        if save_dataset:
            self.dataset.to_pickle(self.datasetPath)
        if return_dataset:
            return self.dataset
        else:
            del self.dataset

    def process_new_dataset(self, dataset_name='new_dataset', test_mode=False):

        self.load_dataset(self.dataset_path, self.collect_chunks, test_mode=test_mode)
        self.numerize_dataset()
        self.curate_dataset()

        self.dataset.to_pickle('../../new_dataset')  # '../../' + dataset_name)
        self.dataset.loc[0:10000].to_pickle('../../test_new_dataset')

    def load_dataset(self, dataset_path, collect_chunks=False, test_mode=False):

        if collect_chunks:
            os.chdir(dataset_path + '/molecule_features')
            chunks = os.listdir()
            print('collecting chunks')

            if test_mode:
                nChunks = min(5, len(chunks))
            else:
                nChunks = len(chunks)

            data_chunks = []
            for i in tqdm.tqdm(range(nChunks)):
                data_chunks.append(pd.read_pickle(chunks[i]))
            self.dataset = pd.concat(data_chunks, ignore_index=True)
            del data_chunks

        else:
            self.dataset = pd.read_pickle(dataset_path)

        self.finished_numerization = False
        self.dataset_keys = list(self.dataset.columns)
        self.dataset_length = len(self.dataset)

    def curate_dataset(self):
        '''
        curate the dataset given conditions set in the config
        :return:
        '''
        # start by filtering polymorphs
        self.collate_spacegroups()
        self.filter_dataset()
        if self.exclude_polymorphs:
            self.filter_polymorphs()

    def collate_spacegroups(self):
        '''
        reassign spacegroup symbols using spacegroup numbers and settings
        set all spacegroup symbols to setting 1
        collect minority spacegroups into reasonable clusters
        :return:
        '''

        # main_symbol = {}
        # for i in tqdm.tqdm(range(self.dataset_length)):
        #     element = self.dataset['crystal spacegroup number'][i], self.dataset['crystal spacegroup setting'][i],
        #     if element[1] == 1:  # extract only the main or primary setting - if there are entries for which we are missing the primary setting, ignore them for now (assume marginal)
        #         if str(element[0]) not in main_symbol.keys():
        #             main_symbol[str(element[0])] = [self.dataset['crystal spacegroup symbol'][i]]
        #         else:
        #             main_symbol[str(element[0])].append(self.dataset['crystal spacegroup symbol'][i])
        #
        # # confirm they're all unique
        # sg_dict = {}
        # for key in main_symbol.keys():
        #     # if len(np.unique(main_symbol[key])) > 1: # the only groups where this doesn't work are with or without minus signs (chiral groups?)
        #     sg_dict[key] = np.unique(main_symbol[key])[0]
        print('Pre-generating spacegroup symmetries')

        self.space_groups = {}
        self.sg_numbers = {}
        for i in tqdm.tqdm(range(1, 231)):
            sym_group = symmetry.Group(i)
            self.space_groups[i] = sym_group.symbol
            self.sg_numbers[sym_group.symbol] = i

        # standardize SG symbols
        for i in tqdm.tqdm(range(self.dataset_length)):
            if self.dataset['crystal spacegroup number'][i] != 0:  # sometimes the sg number is broken
                self.dataset['crystal spacegroup symbol'][i] = self.space_groups[self.dataset['crystal spacegroup number'][i]]
            else:  # in which case, try reverse assigning the number, given the space group
                self.dataset['crystal spacegroup number'][i] = self.sg_numbers[self.dataset['crystal spacegroup symbol'][i]]

        #
        # self.sg_probabilities = {}
        # key = 'crystal spacegroup symbol'
        # self.dataset[key] = np.asarray(self.dataset[key])  # convert molecule and crystal features to numpy arrays for easy processing
        # unique_entries = np.unique(self.dataset[key])
        # print("One-hotting " + key + " with unique entries {}".format(unique_entries))
        # for entry in unique_entries:
        #     self.dataset[key + ' is ' + entry] = self.dataset[key] == entry
        #     self.sg_probabilities[entry] = np.average(self.dataset[key + ' is ' + entry])
        #     self.modellable_keys.append(key + ' is ' + entry)
        #
        # # identify majority and minority SGs
        # self.majority_sgs = []
        # self.minority_sgs = []
        # for key in self.sg_probabilities:
        #     if self.sg_probabilities[key] >= 0.01:
        #         self.majority_sgs.append(key)
        #     elif self.sg_probabilities[key] < 0.01:
        #         self.minority_sgs.append(key)
        #
        # self.dataset['crystal spacegroup is minority'] = np.asarray([spacegroup not in self.majority_sgs for spacegroup in self.dataset['crystal spacegroup symbol']])

    def filter_dataset(self):
        print('Filtering dataset starting from {} samples'.format(len(self.dataset)))
        ## filtering out unwanted characteristics
        bad_inds = []

        #
        # # exclude samples with extremely close atoms
        # n_bad_inds = len(bad_inds)
        # for j in range(len(self.dataset)):
        #     coords = self.dataset['crystal reference cell coords'][j]
        #     coords = coords.reshape(coords.shape[0] * coords.shape[1],3)
        #     distmat = torch.cdist(torch.Tensor(coords), torch.Tensor(coords), p=2) + torch.eye(len(coords))
        #     if torch.amin(distmat) < 0.1:
        #         #print('bad')
        #         bad_inds.append(j)
        # print('overlapping atoms caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # samples with bad CSD-generated reference cells
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal reference cell coords']) == 'error')[:, 0])  # missing coordinates
        bad_inds.extend(np.argwhere(np.asarray(np.isnan(self.dataset['crystal asymmetric unit centroid x'])))[:, 0])  # missing orientation features
        print('bad coordinates caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        if self.exclude_blind_test_targets:
            # CSD blind test 5 and 6 targets
            blind_test_identifiers = [
                'OBEQUJ', 'OBEQOD', 'OBEQET', 'XATJOT', 'OBEQIX', 'KONTIQ',
                'NACJAF', 'XAFPAY', 'XAFQON', 'XAFQIH', 'XAFPAY01', 'XAFPAY02', 'XAFPAY03', 'XAFPAY04',
                "COUMAR01", "COUMAR02", "COUMAR10", "COUMAR11", "COUMAR12", "COUMAR13",
                "COUMAR14", "COUMAR15", "COUMAR16", "COUMAR17",  # Z'!=1 or some other weird thing
                "COUMAR18", "COUMAR19"
            ]
            blind_test_identifiers.remove('XATJOT')  # multi-component
            blind_test_identifiers.remove('XAFQON')  # multi-component
            blind_test_identifiers.remove('KONTIQ')  # multi-component

            # samples with bad CSD-generated reference cells
            n_bad_inds = len(bad_inds)
            for j in range(len(self.dataset)):
                item = self.dataset['identifier'][j]  # do it this way to remove the target, including any of its polymorphs
                if item[-1].isdigit():
                    item = item[:-2]  # cut off trailing digits, if any
                if item in blind_test_identifiers:
                    bad_inds.append(j)
            print('Blind test targets caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # filter for when the asymmetric unit definition is nonstandard (returns more than one centroid) # todo check for edge centroids
        n_bad_inds = len(bad_inds)
        for ii in range(len(self.dataset['atom coords'])):
            sg_ind = self.dataset['crystal spacegroup number'][ii]
            if str(sg_ind) in asym_unit_dict.keys():  # only do this check if this sg_ind is already encoded in the asym unit dict
                unit_cell_coords = self.dataset['crystal reference cell coords'][ii]
                T_cf = np.linalg.inv(self.dataset['crystal fc transform'][ii])
                asym_unit = asym_unit_dict[str(int(sg_ind))]  # will only work for units which we have written down the parameterization for
                # identify which of the Z asymmetric units is canonical
                centroids_cartesian = unit_cell_coords.mean(-2)
                centroids_fractional = np.inner(T_cf, centroids_cartesian).T
                centroids_fractional -= np.floor(centroids_fractional)
                if torch.is_tensor(asym_unit):
                    asym_unit = asym_unit.cpu().detach().numpy()
                canonical_conformer_index = find_coord_in_box_np(centroids_fractional, asym_unit)
                if len(canonical_conformer_index) != 1:
                    bad_inds.append(ii)
        print('Non uniform asymmetric unit caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # when the molecule is too long
        # cases where the csd has the wrong number of molecules
        n_bad_inds = len(bad_inds)
        # self.config.max_molecule_radius
        mol_radii = np.asarray([np.amax(np.linalg.norm(coords - coords.mean(0), axis=-1)) for coords in self.dataset['atom coords']])
        bad_inds.extend(np.argwhere(np.asarray(mol_radii) > self.max_molecule_radius)[:, 0])
        print('molecule too long filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # cases where the csd has the wrong number of molecules
        n_bad_inds = len(bad_inds)
        lens = [len(item) for item in self.dataset['crystal reference cell coords']]
        bad_inds.extend(np.argwhere(np.asarray(lens != self.dataset['crystal z value']))[:, 0])
        print('improper CSD Z value filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # # cases where the symmetry ops disagree with the Z value
        # n_bad_inds = len(bad_inds)
        # lens = [len(item) for item in self.dataset['crystal symmetries']]
        # bad_inds.extend(np.argwhere(np.asarray(lens != self.dataset['crystal z value']))[:, 0])
        # print('improper sym ops multiplicity filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # exclude samples with atoms on special positions # todo eventually relax this
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal atoms on special positions'].ne([[] for _ in range(len(self.dataset))])))[:, 0])
        print('special positions filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # Z prime
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z prime']) > self.max_z_prime)[:, 0])  # self.config.max_z_prime))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z prime']) < self.min_z_prime)[:, 0])  # self.config.min_z_prime))
        print('z prime filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # Z value
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z value']) > self.max_z_value)[:, 0])  # self.config.max_z_value))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z value']) < self.min_z_value)[:, 0])  # self.config.min_z_value))
        print('z value filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # molecule num atoms
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule num atoms']) > self.max_num_atoms)[:, 0])  # self.config.max_molecule_size))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule num atoms']) < self.min_num_atoms)[:, 0])  # self.config.min_molecule_size))
        print('molecule num atoms filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # heaviest atom
        n_bad_inds = len(bad_inds)
        heaviest_atoms = np.asarray([max(atom_z) for atom_z in self.dataset['atom Z']])
        bad_inds.extend(np.argwhere(heaviest_atoms > self.max_atomic_number)[:, 0])  # self.config.max_atomic_number))
        print('max atom size filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # too diffuse or too dense
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal packing coefficient']) > self.max_packing_coefficient)[:, 0])  # self.config.max_molecule_size))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal packing coefficient']) < self.min_packing_coefficient)[:, 0])  # self.config.min_molecule_size))
        print('crystal packing coefficient filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # erroneous densities
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal calculated density']) == 0)[:, 0])  # self.config.max_molecule_size))
        print('0 density filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # too hot or too cold
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal temperature']) > self.max_temperature)[:, 0])
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal temperature']) < self.min_temperature)[:, 0])
        print('crystal temperature filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # supported space groups
        if self.include_sgs is not None:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere([self.dataset['crystal spacegroup symbol'][i] not in self.include_sgs for i in range(len(self.dataset['crystal spacegroup symbol']))])[:, 0])
            print('spacegroup filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        if self.exclude_crystal_systems is not None:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere([self.dataset['crystal system'][i] in self.exclude_crystal_systems for i in range(len(self.dataset['crystal system']))])[:, 0])
            print('crystal system filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        if self.include_pgs is not None:
            # filter by point group
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere([self.dataset['crystal point group'][i] not in self.include_pgs for i in range(len(self.dataset['crystal point group']))])[:, 0])  # missing coordinates
            print('unwanted point groups caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # molecule organic
        if not self.include_organic:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal is organic'] == 'True'))[:, 0])
            print('organic filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # molecule organometallic
        if not self.include_organometallic:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal is organometallic'] == 'True'))[:, 0])
            print('organometallic filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # molecule has disorder
        if self.exclude_disordered_crystals:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal has disorder']))[:, 0])
            print('disorder filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # missing r factor
        if self.exclude_missing_r_factor:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.asarray([i for i in range(len(self.dataset['crystal r factor'])) if self.dataset['crystal r factor'][i] is None]))
            print('missing r factor filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        if self.exclude_nonstandard_settings:
            # nonstandard spacegroup setings have inconsistent lattice definitions
            n_bad_inds = len(bad_inds)
            settings = np.asarray([self.dataset['crystal spacegroup setting'][i] for i in range(len(self.dataset))])
            bad_inds.extend(np.argwhere(settings != 1)[:, 0])
            print('nonstandard spacegroup setting filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # collate bad indices
        bad_inds = np.unique(bad_inds)

        # apply filtering
        self.dataset = delete_from_dataframe(self.dataset, bad_inds)
        print("Filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))

    def filter_polymorphs(self):
        '''
        find duplicate examples and pick one representative per molecule
        :return:
        '''
        # use CSD identifiers to pick out polymorphs
        duplicate_lists, duplicate_list_extension, duplicate_groups = self.get_identifier_duplicates()  # issue - some of these are not isomorphic (i.e., different ionization), maybe ~2% from early tests

        # duplicate_groups_identifiers = {ident:[self.dataset['identifier'][n] for n in duplicate_groups[ident]] for ident in duplicate_groups.keys()}
        # duplicate_groups_packings = {ident:[self.dataset['crystal packing coefficient'][n] for n in duplicate_groups[ident]] for ident in duplicate_groups.keys()}

        # todo delete blind test and any extra CSP samples from the training dataset

        # now, the 'representative structure' is the highest resolution structure which as the same space group as the oldest structure
        # we will add all others to 'bad_inds', and filter them out at our leisure
        print('selecting representative structures from duplicate groups')
        bad_inds = []
        for key in duplicate_groups.keys():
            # print(key)
            inds = duplicate_groups[key]
            space_groups = [self.dataset['crystal spacegroup symbol'][ind] for ind in inds]
            oldest_structure = inds[np.argmin([self.dataset['crystal date'][ind] for ind in inds])]  # get the dataset index for the oldest structure
            oldest_structure_SG = self.dataset['crystal spacegroup symbol'][oldest_structure]
            inds_agree_with_oldest_SG = np.argwhere([sg == oldest_structure_SG for sg in space_groups])[:, 0]
            inds_agree_with_oldest_SG = [inds[ind] for ind in inds_agree_with_oldest_SG]
            agreeing_ind_with_best_arg_factor = inds[np.argmin([self.dataset['crystal r factor'][ind] for ind in inds_agree_with_oldest_SG])]
            inds.remove(agreeing_ind_with_best_arg_factor)  # remove the good one
            bad_inds.extend(inds)  # delete residues from the dataset

        self.dataset = delete_from_dataframe(self.dataset, bad_inds)

    def get_identifier_duplicates(self):
        # by CSD identifier
        # CSD entries with numbers on the end are subsequent additions
        print('getting identifier duplicates')
        all_identifiers = {}
        if self.database.lower() == 'csd':
            for i in tqdm.tqdm(range(len(self.dataset['identifier']))):
                item = self.dataset['identifier'][i]
                if item[-1].isdigit():
                    item = item[:-2]  # cut off trailing digits, if any
                if item not in all_identifiers.keys():
                    all_identifiers[item] = []
                all_identifiers[item].append(i)
        elif self.database.lower() == 'cif':  # todo unfinished
            blind_test_targets = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                                  'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
                                  'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', ]
            all_identifiers = {key: [] for key in blind_test_targets}
            for i in tqdm.tqdm(range(len(self.dataset['identifier']))):
                item = self.dataset['identifier'][i]
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

    def numerize_dataset(self):
        '''
        convert dataset features such as strings into purely numerical vectors (not normed etc yet)
        :return:
        '''
        # if type(self.dataset['molecule point group']) is not str:
        #     self.dataset['molecule point group'] = [str(pg) for pg in self.dataset['molecule point group']]

        self.modellable_keys = []

        # manually add this one to get started
        self.dataset['molecule num atoms'] = np.asarray(self.dataset['molecule num atoms'])
        self.modellable_keys.append('molecule num atoms')

        for key in self.dataset_keys:
            if ('atom' not in key) and ('symmetry operators' not in key) \
                    and ('pressure' not in key) and ('chemical name' not in key) \
                    and ('disorder details' not in key) and ('polymorph' not in key) \
                    and ('identifier' not in key) and ('date' not in key) \
                    and ('smiles' not in key) and ('formula' not in key) \
                    and ('xyz' not in key) and ('organic' not in key) \
                    and ('polymeric' not in key) and ('organometallic' not in key) \
                    and ('powder' not in key) and ('spacegroup' not in key) \
                    and ('cell length' not in key) and ('angles' not in key) \
                    and ('radiation' not in key) and ('setting' not in key) \
                    and ('planes' not in key) and (('centroids') not in key) \
                    and ('axes' not in key):  # leave atom features as lists, as well as things we don't want to be categorized
                self.dataset[key] = np.asarray(self.dataset[key])  # convert molecule and crystal features to numpy arrays for easy processing

                if self.dataset[key].dtype == bool:  # keep bools
                    self.modellable_keys.append(key)
                    pass  # already effectively 'one-hot'

                elif self.dataset[key].dtype.type is np.str_:  # now, process categorial strings into one-hots
                    unique_entries = np.unique(self.dataset[key])
                    print("One-hotting " + key + " with unique entries {}".format(unique_entries))
                    for entry in unique_entries:
                        self.dataset[key + ' is ' + entry] = self.dataset[key] == entry
                        self.modellable_keys.append(key + ' is ' + entry)

                elif self.dataset[key].dtype == int:
                    self.modellable_keys.append(key)
                    pass
                elif self.dataset[key].dtype == float:
                    self.modellable_keys.append(key)
                    pass

        self.finished_numerization = True


if __name__ == '__main__':
    config = None
    # todo deprecate for new dataset construction method
    miner = DataManager(dataset_path='C:/Users\mikem\Desktop\CSP_runs\datasets/may_new_pull/mol_features', config=None, collect_chunks=True)
    miner.process_new_dataset(test_mode=False)

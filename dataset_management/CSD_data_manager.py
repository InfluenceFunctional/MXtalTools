from utils import *
import matplotlib.pyplot as plt
import tqdm
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Miner():
    def __init__(self, dataset_path, config=None, collect_chunks=False):
        self.config = config
        if config is None:
            self.max_z_prime = 1
            self.min_z_prime = 1
            self.max_z_value = 10
            self.min_z_value = 1
            self.max_num_atoms = 200
            self.min_num_atoms = 20
            self.max_atomic_number = 87
            self.max_packing_coefficient = 0.85
            self.min_packing_coefficient = 0.55
            self.include_organic = True
            self.include_organometallic=True
            self.exclude_disordered_crystals=True
            self.exclude_missing_r_factor=True
            self.exclude_polymorphs=True
            self.exclude_nonstandard_settings = True
            self.max_temperature = 1000
            self.min_temperature = 0
            self.include_sgs = None
        else:
            self.max_z_prime = config.max_z_prime
            self.min_z_prime = config.min_z_prime
            self.max_z_value = config.max_z_value
            self.min_z_value = config.min_z_value
            self.max_num_atoms = config.max_num_atoms
            self.min_num_atoms = config.min_num_atoms
            self.max_atomic_number = config.max_atomic_number
            self.max_packing_coefficient = 0.85
            self.min_packing_coefficient = config.min_atomic_number
            self.include_organic = config.include_organic
            self.include_organometallic= config.include_organometallic
            self.exclude_disordered_crystals=config.exclude_disordered_crystals
            self.exclude_polymorphs=config.exclude_polymorphs
            if self.exclude_polymorphs:
                self.exclude_missing_r_factor = True
            else:
                self.exclude_missing_r_factor = config.exclude_missing_r_factor
            self.exclude_nonstandard_settings = config.exclude_nonstandard_settings
            self.max_temperature = config.max_crystal_temperature
            self.min_temperature = config.min_crystal_temperature
            self.include_sgs = config.include_sgs

        self.dataset_path = dataset_path
        self.collect_chunks = collect_chunks

    def load_for_modelling(self):
        self.dataset = pd.read_pickle(self.dataset_path)
        self.dataset_keys = list(self.dataset.columns)
        self.filter_dataset()
        self.datasetPath = 'datasets/dataset'
        self.dataset.to_pickle(self.datasetPath)
        del (self.dataset)

    def process_new_dataset(self, test_mode=False):

        self.load_dataset(self.dataset_path, self.collect_chunks, test_mode=test_mode)
        self.numerize_dataset()
        self.curate_dataset()

        self.dataset.to_pickle('../../full_dataset')
        self.dataset.loc[0:1000].to_pickle('../../test_dataset')


    def load_dataset(self, dataset_path, collect_chunks=False,
                     post_electronegativity_fix=True,
                     post_symmetry_fix=True,
                     do_add_formula=False,
                     add_new_correlates=True,
                     add_unit_cells=False,
                     test_mode=False):

        if collect_chunks:
            os.chdir(dataset_path)
            chunks = os.listdir()
            print('collecting chunks')

            if test_mode:
                nChunks = min(5,len(chunks))
            else:
                nChunks = len(chunks)

            data_chunks = []
            for i in tqdm.tqdm(range(nChunks)):
                data_chunks.append(pd.read_pickle(chunks[i]))
            self.dataset = pd.concat(data_chunks,ignore_index=True)
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

    def collate_spacegroups(self, only_standard_settings = False):
        '''
        reassign spacegroup symbols using spacegroup numbers and settings
        set all spacegroup symbols to setting 1
        collect minority spacegroups into reasonable clusters
        :return:
        '''

        main_symbol = {}
        for i in tqdm.tqdm(range(self.dataset_length)):
            element = self.dataset['crystal spacegroup number'][i], self.dataset['crystal spacegroup setting'][i],
            if element[1] == 1:  # extract only the main or primary setting - if there are entries for which we are missing the primary setting, ignore them for now (assume marginal)
                if str(element[0]) not in main_symbol.keys():
                    main_symbol[str(element[0])] = [self.dataset['crystal spacegroup symbol'][i]]
                else:
                    main_symbol[str(element[0])].append(self.dataset['crystal spacegroup symbol'][i])

        # confirm they're all unique
        sg_dict = {}
        for key in main_symbol.keys():
            # if len(np.unique(main_symbol[key])) > 1: # the only groups where this doesn't work are with or without minus signs (chiral groups?)
            sg_dict[key] = np.unique(main_symbol[key])[0]

        # standardize SGs
        for i in tqdm.tqdm(range(self.dataset_length)):
            if str(self.dataset['crystal spacegroup number'][i]) in sg_dict.keys():
                self.dataset['crystal spacegroup symbol'][i] = sg_dict[str(self.dataset['crystal spacegroup number'][i])]

        self.sg_probabilities = {}
        key = 'crystal spacegroup symbol'
        self.dataset[key] = np.asarray(self.dataset[key])  # convert molecule and crystal features to numpy arrays for easy processing
        unique_entries = np.unique(self.dataset[key])
        print("One-hotting " + key + " with unique entries {}".format(unique_entries))
        for entry in unique_entries:
            self.dataset[key + ' is ' + entry] = self.dataset[key] == entry
            self.sg_probabilities[entry] = np.average(self.dataset[key + ' is ' + entry])
            self.modellable_keys.append(key + ' is ' + entry)

        # identify majority and minority SGs
        self.majority_sgs = []
        self.minority_sgs = []
        for key in self.sg_probabilities:
            if self.sg_probabilities[key] >= 0.01:
                self.majority_sgs.append(key)
            elif self.sg_probabilities[key] < 0.01:
                self.minority_sgs.append(key)

        self.dataset['crystal spacegroup is minority'] = np.asarray([spacegroup not in self.majority_sgs for spacegroup in self.dataset['crystal spacegroup symbol']])

    def filter_dataset(self):
        print('Filtering dataset')
        ## filtering out unwanted characteristics
        bad_inds = []

        # Z prime
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z prime']) > self.max_z_prime)[:, 0])  # self.config.max_z_prime))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z prime']) < self.min_z_prime)[:, 0])  # self.config.min_z_prime))

        # Z value
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z value']) > self.max_z_value)[:, 0])  # self.config.max_z_value))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal z value']) < self.min_z_value)[:, 0])  # self.config.min_z_value))

        # molecule num atoms
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule num atoms']) > self.max_num_atoms)[:, 0])  # self.config.max_molecule_size))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule num atoms']) < self.min_num_atoms)[:, 0])  # self.config.min_molecule_size))

        # heaviest atom
        heaviest_atoms = np.asarray([max(atom_z) for atom_z in self.dataset['atom Z']])
        bad_inds.extend(np.argwhere(heaviest_atoms > self.max_atomic_number)[:, 0])  # self.config.max_atomic_number))

        # too diffuse or too dense
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal packing coefficient']) > self.max_packing_coefficient)[:, 0])  # self.config.max_molecule_size))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal packing coefficient']) < self.min_packing_coefficient)[:, 0])  # self.config.min_molecule_size))

        # too hot or too cold
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal temperature']) > self.max_temperature)[:, 0])  # self.config.max_molecule_size))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal temperature']) < self.min_temperature)[:, 0])  # self.config.min_molecule_size))

        # supported space groups
        if self.include_sgs is not None:
            bad_inds.extend(np.argwhere([self.dataset['crystal spacegroup symbol'][i] not in self.include_sgs for i in range(len(self.dataset['crystal spacegroup symbol']))])[:,0])

        # molecule organic
        if not self.include_organic:
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule is organic']))[:, 0])

        # molecule organometallic
        if not self.include_organometallic:
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule is organometallic']))[:, 0])

        # molecule has disorder
        if self.exclude_disordered_crystals:
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal has disorder']))[:, 0])

        # missing r factor
        if self.exclude_missing_r_factor:
            bad_inds.extend(np.asarray([i for i in range(len(self.dataset['crystal r factor'])) if self.dataset['crystal r factor'][i] is None]))

        if self.exclude_nonstandard_settings:
            # nonstandard spacegroup setings have inconsistent lattice definitions
            settings = np.asarray([self.dataset['crystal spacegroup setting'][i] for i in range(len(self.dataset))])
            bad_inds.extend(np.argwhere(settings != 1)[:, 0])

        # collate bad indices
        bad_inds = np.unique(bad_inds)

        # apply filtering
        self.dataset = delete_from_dataframe(self.dataset,bad_inds)

        print("Filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))


    def filter_polymorphs(self):
        '''
        find duplicate examples and pick one representative per molecule
        :return:
        '''
        # check for polymorphs manually

        # add molecular formula to every molecule

        # check for duplicate molecular formulas

        # isomorph_groups, isomorph_groups_extension = self.get_isomorph_lists()

        duplicate_lists, duplicate_list_extension, duplicate_groups = self.get_CSD_identifier_duplicates()  # issue - some of these are not isomorphic (i.e., different ionization), maybe ~2% from early tests

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

        self.dataset = delete_from_dataframe(self.dataset,bad_inds)


    def get_CSD_identifier_duplicates(self):
        # by CSD identifier
        # CSD entries with numbers on the end are subsequent additions
        print('getting identifier duplicates')
        all_identifiers = {}
        for i in tqdm.tqdm(range(len(self.dataset['identifier']))):
            item = self.dataset['identifier'][i]
            if item[-1].isdigit():
                item = item[:-2]  # cut off trailing digits, if any
            if item not in all_identifiers.keys():
                all_identifiers[item] = []
            all_identifiers[item].append(i)

        # delete identifiers with only one entry, despite having an index attache
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
        if type(self.dataset['molecule point group']) is not str:
            self.dataset['molecule point group'] = [str(pg) for pg in self.dataset['molecule point group']]

        self.modellable_keys = []

        # manually add this one
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

                if self.dataset[key].dtype == bool:  # now, process categorial strings into one-hots
                    self.modellable_keys.append(key)
                    pass  # already effectively 'one-hot'

                elif self.dataset[key].dtype.type is np.str_:
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

    def correlations_analysis(self):
        '''
        evaluate pairwise correlations for molecule and crystal features
        :return:
        '''

        import seaborn as sb

        # make useful correlates
        correlates_list = []
        bad_keys = ['spacegroup', 'point group', 'cell lengths', 'angles', 'setting', 'powder', 'formula', 'radiation', 'polymeric', 'organic', 'organometallic', 'symmetry operators', 'xyz', 'crystal has', 'z prime', 'centring']
        for key in self.modellable_keys:
            key_good = True
            for bad_key in bad_keys:
                if bad_key in key:
                    key_good = False
                    break
            if key_good:
                if 'molecule' in key:
                    correlates_list.append(key)

        correlates_list.append('molecule point group is C1')

        for key in self.modellable_keys:
            key_good = True
            for bad_key in bad_keys:
                if bad_key in key:
                    key_good = False
                    break
            if key_good:
                if 'crystal' in key:
                    correlates_list.append(key)

        correlates_list.append('crystal spacegroup symbol is P21/c')
        correlates_list.append('crystal spacegroup symbol is P212121')
        correlates_list.append('crystal spacegroup symbol is P21')
        correlates_list.append('crystal spacegroup symbol is P-1')
        correlates_list.append('crystal spacegroup symbol is C2/c')
        correlates_list.append('crystal has glide')
        correlates_list.append('crystal has inversion')
        correlates_list.append('crystal has rotoinversion')
        correlates_list.append('crystal has mirror')
        correlates_list.append('crystal has rotation')
        correlates_list.append('crystal has screw')

        correlates_list.remove('crystal system is cubic')

        # setup correlates dict
        correlate_data = dict((k, self.dataset[k]) for k in (correlates_list))

        # visualize correlations
        df = pd.DataFrame.from_dict(correlate_data)
        C_mat = df.corr(method='pearson')
        plt.figure(1)
        plt.clf()
        sb.set(font_scale=.75)
        sb.heatmap(np.abs(C_mat - np.eye(len(C_mat), len(C_mat))), vmax=1.0, square=True)
        plt.tight_layout()

        # pull out biggest correlations
        correlations = np.triu(C_mat - np.eye(len(C_mat), len(C_mat)))
        # sort by absolute correlation
        sorted_correlation_inds = np.dstack(np.unravel_index(np.argsort(np.abs(correlations).ravel()), (len(C_mat), len(C_mat))))[0]
        corr_vals = [correlations[inds[0], inds[1]] for inds in sorted_correlation_inds]

        best_correlates = []
        for i in range(len(sorted_correlation_inds) - 50, len(sorted_correlation_inds)):
            best_correlates.append(correlates_list[sorted_correlation_inds[i, 0]] + ' + ' + correlates_list[sorted_correlation_inds[i, 1]] + ' ==> {:.3f}'.format(corr_vals[i]))

        # molecule - to - crystal correlations (shit we care about)
        mol_cryst_corrs = []
        mol_cryst_corrs_record = []
        for i in range(len(sorted_correlation_inds)):
            correlate_1 = correlates_list[sorted_correlation_inds[i, 0]]
            correlate_2 = correlates_list[sorted_correlation_inds[i, 1]]
            if 'organ' not in correlate_1 and 'organ' not in correlate_2:
                if ('crystal' in correlate_1 and 'molecule' in correlate_2) or ('crystal' in correlate_2 and 'molecule' in correlate_1):
                    mol_cryst_corrs.append(correlates_list[sorted_correlation_inds[i, 0]] + ' + ' + correlates_list[sorted_correlation_inds[i, 1]] + ' ==> {:.3f}'.format(corr_vals[i]))
                    mol_cryst_corrs_record.append([correlates_list[sorted_correlation_inds[i, 0]] + ' <==> ' + correlates_list[sorted_correlation_inds[i, 1]], corr_vals[i]])

        mol_cryst_to_plot = [record for record in mol_cryst_corrs_record if np.abs(record[1]) > 0.12]
        plt.figure(2)
        plt.clf()
        colors = ['b' if entry[1] > 0 else 'r' for entry in mol_cryst_to_plot]
        plt.barh([record[0] for record in mol_cryst_to_plot], [abs(record[1]) for record in mol_cryst_to_plot], color=colors)
        plt.title('Molecule-to-Crystal Correlations')
        plt.tight_layout()

    def one_hot_feature_to_dict(self, dict, key):
        import torch.nn.functional as F
        import torch

        array = torch.Tensor(dict[key]).long()
        classes = len(torch.unique(array))
        one_hot = F.one_hot(array - torch.amin(array), num_classes=classes)

        for i in range(classes):
            dict[key + ' #{}'.format(i)] = one_hot[:, i].numpy()

        return dict


if __name__ == '__main__':
    config = None
    miner = Miner(dataset_path='C:/Users\mikem\Desktop\CSP_runs\datasets/may_new_pull/mol_features', config=None, collect_chunks=True)
    miner.process_new_dataset(test_mode=True)

    # miner = Miner(config, collect_chunks=False, dataset_path='C:/Users\mikem\Desktop\CSP_runs\datasets/new_full_featurized.npy')
    # miner.basic_analysis()

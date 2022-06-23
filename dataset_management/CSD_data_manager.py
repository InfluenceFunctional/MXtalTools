from utils import *
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from dataset_management.random_crystal_builder import *
from pyxtal import symmetry
from nikos.coordinate_transformations import cell_vol, coor_trans_matrix
import matplotlib.colors as colors
from ase import Atoms
from ase.visualize import view
from ase.geometry.analysis import Analysis
from ase.ga.utilities import get_rdf

class Miner():
    def __init__(self, dataset_path, config=None, collect_chunks=False):
        self.config = config
        if config is None:
            self.max_z_prime = 1
            self.min_z_prime = 1
            self.max_z_value = 10
            self.min_z_value = 1
            self.max_num_atoms = 1000
            self.min_num_atoms = 5
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
        else:
            self.max_z_prime = config.max_z_prime
            self.min_z_prime = config.min_z_prime
            self.max_z_value = config.max_z_value
            self.min_z_value = config.min_z_value
            self.max_num_atoms = config.max_num_atoms
            self.min_num_atoms = config.min_num_atoms
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

        self.dataset_path = dataset_path
        self.collect_chunks = collect_chunks

    def load_for_modelling(self, return_dataset = False, save_dataset = True):
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
            del (self.dataset)

    def load_npy_for_modelling(self):
        self.dataset = np.load(self.dataset_path + '.npy', allow_pickle=True).item()
        self.dataset = pd.DataFrame.from_dict(self.dataset)
        self.dataset_keys = list(self.dataset.columns)
        self.filter_dataset()
        if self.exclude_polymorphs:
            self.filter_polymorphs()
        self.datasetPath = 'datasets/dataset'
        self.dataset.to_pickle(self.datasetPath)
        del (self.dataset)

    def process_new_dataset(self, test_mode=False):

        self.load_dataset(self.dataset_path, self.collect_chunks, test_mode=test_mode)
        self.numerize_dataset()
        self.curate_dataset()

        self.dataset.to_pickle('../../full_dataset')
        self.dataset.loc[0:10000].to_pickle('../../test_dataset')

    def load_dataset(self, dataset_path, collect_chunks=False, test_mode=False):

        if collect_chunks:
            os.chdir(dataset_path)
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
        print('Filtering dataset starting from {} samples'.format(len(self.dataset)))
        ## filtering out unwanted characteristics
        bad_inds = []

        # todo filter samples where space groups explicitly disagree with given crystal system

        # samples with bad CSD-generated reference cells
        if ('cell' in self.config.mode) or ('joint' in self.config.mode):
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal reference cell coords']) == 'error')[:,0])
            print('bad packing filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # cases where the csd has the wrong number of molecules
        n_bad_inds = len(bad_inds)
        lens = [len(item) for item in self.dataset['crystal reference cell coords']]
        bad_inds.extend(np.argwhere(np.asarray(lens != self.dataset['crystal z value']))[:,0])
        print('improper CSD Z value filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # exclude samples with atoms on special positions
        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal atoms on special positions'].ne([[] for _ in range(len(self.dataset))])))[:,0])
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
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal temperature']) > self.max_temperature)[:, 0])  # self.config.max_molecule_size))
        bad_inds.extend(np.argwhere(np.asarray(self.dataset['crystal temperature']) < self.min_temperature)[:, 0])  # self.config.min_molecule_size))
        print('crystal temperature filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # supported space groups
        if self.include_sgs is not None:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere([self.dataset['crystal spacegroup symbol'][i] not in self.include_sgs for i in range(len(self.dataset['crystal spacegroup symbol']))])[:, 0])
            print('spacegroup filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # molecule organic
        if not self.include_organic:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule is organic']))[:, 0])
            print('organic filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # molecule organometallic
        if not self.include_organometallic:
            n_bad_inds = len(bad_inds)
            bad_inds.extend(np.argwhere(np.asarray(self.dataset['molecule is organometallic']))[:, 0])
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

        self.dataset = delete_from_dataframe(self.dataset, bad_inds)

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

    def todays_analysis(self):

        self.dataset = pd.read_pickle(self.dataset_path)
        self.dataset_keys = list(self.dataset.columns)

        '''
        analysis to-do
        0.
        X confirm csd fractional transforms agree with my method
        1. 
        X get reference cell
        X get all centroids
        X consider nearest centroid as canonical
        2. 
        X compute inertial plane
        X compute orientation w.r.t. a and b axes
        rotation w.r.t. some canonical axis
        X plane overlaps with supercell axes out to 5x5
        3. 
        compute histograms of raw data and distances on CSD data
            distance - e.g., some kind of RMSD or COMPACK calc, now that we do have the CCDC api to-hand
        3a.
        repeat analysis for random cells
        3b. try to fit it all with a joint distribution, with auxiliary loss(es) [e.g., volume, orientation distance]
        4. 
        compare inertial and/or ring planes to 5x5 cell axes
        5. 
        '''
        print('Analyzing reference cells')
        self.sym_ops = {}
        self.point_groups = {}
        self.lattice_type = {}
        for i in tqdm.tqdm(range(1, 231)):
            sym_group = symmetry.Group(i)
            general_position_syms = sym_group.wyckoffs_organized[0][0]
            self.sym_ops[i] = [general_position_syms[i].affine_matrix for i in range(len(general_position_syms))]  # first 0 index is for general position, second index is superfluous, third index is the symmetry operation
            self.point_groups[i] = sym_group.point_group
            self.lattice_type[i] = sym_group.lattice_type

        # generate all combinations of 5x5 fractional vectors
        # Nikos says only take these [-n_max,n_max], n_1*n_2*n_3=0 and n_1|n_2|n_3=n_max
        supercell_ref_vectors_f = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    if (i != j != k):
                        supercell_ref_vectors_f.append([i, j, k])
        supercell_ref_vectors_f = np.asarray(supercell_ref_vectors_f)

        self.centroids_f = []
        self.supercell_angle_overlaps = []
        self.centroid_fractional_displacement = []
        self.centroid_cartesian_displacement = []
        self.inertial_angles = []
        self.inertial_axes = []
        self.orientation_angles = []
        good_inds = []
        rand_inds = []
        csd_inds = []
        tests = []
        ind = -1
        study_cells = True
        compare_cells = False
        diff_list = []
        for i in tqdm.tqdm(range(len(self.dataset))):
            if self.dataset['crystal reference cell coords'][i] != 'error':  # skip bad cells
                # confirm we have good syms
                z_value = int(self.dataset['crystal z value'][i])
                sg_number = self.dataset['crystal spacegroup number'][i]
                #sym_group = symmetry.Group(self.dataset['crystal spacegroup number'][i])
                if z_value != len(self.sym_ops[sg_number]):#sym_group[0]): # if there's something wrong with the symmetry
                    continue
                '''
                SETUP & BOILERPLATE
                '''
                # get cell params

                ind += 1
                good_inds.append(i)
                cell_lengths = np.asarray(self.dataset['crystal cell lengths'][i])
                cell_angles = np.asarray(self.dataset['crystal cell angles'][i])

                # get all the transforms
                T_fc = coor_trans_matrix('f_to_c', cell_lengths, cell_angles)
                T_cf = coor_trans_matrix('c_to_f', cell_lengths, cell_angles) #np.linalg.inv(T_fc)#

                atomic_numbers = np.asarray(self.dataset['atom Z'][i])
                heavy_atom_inds = np.argwhere(atomic_numbers > 1)[:, 0]
                masses = np.asarray(self.dataset['atom mass'][i])[heavy_atom_inds]
                atomic_numbers = atomic_numbers[heavy_atom_inds]

                if study_cells:
                    # atoms in cartesian coords
                    from_csd = np.random.randint(0,2,size=1)
                    if from_csd:
                        csd_inds.append(ind)

                        # reference cell coords
                        cell_coords_c = self.dataset['crystal reference cell coords'][i][:, :, :3]
                        cell_coords_f = self.dataset['crystal reference cell coords'][i][:, :, 3:]
                    else:
                        # random cell
                        rand_inds.append(ind)
                        # reference cell coords
                        coords_c = self.dataset['atom coords'][i][heavy_atom_inds]
                        random_coords = randomize_molecule_position_and_orientation(coords_c, masses, T_fc, T_cf, np.asarray(self.sym_ops[sg_number]),confirm_transform = False)
                        #tests.append(np.average(T_cf.dot(random_coords.T).T,axis=0))
                        cell_coords_c, cell_coords_f = build_random_crystal(T_cf, T_fc, random_coords, np.asarray(self.sym_ops[sg_number]), z_value)


                    self.get_cell_data(i, masses, T_cf, T_fc, cell_coords_c, cell_coords_f, z_value, supercell_ref_vectors_f)
                if compare_cells:
                    #check against CSD cell
                    #
                    # # get csd corrected for centroids outside the reference cell
                    # cell_coords_c = self.dataset['crystal reference cell coords'][i][:, :, :3]
                    # cell_coords_f = self.dataset['crystal reference cell coords'][i][:, :, 3:]
                    #
                    # # manually enforce our style of centroid
                    # flag = 0
                    # for j in range(z_value):
                    #     if (np.amin(CoG_f[j])) < 0 or (np.amax(CoG_f[j]) > 1):
                    #         flag = 1
                    #         # print('Sample found with molecule out of cell')
                    #         floor = np.floor(CoG_f[j])
                    #         new_fractional_coords = cell_coords_f[j] - floor
                    #         cell_coords_c[j] = T_fc.dot(new_fractional_coords.T).T
                    #         cell_coords_f[j] = new_fractional_coords
                    #
                    # # redo it
                    # if flag:
                    #     CoG_c = np.average(cell_coords_c, axis=1)
                    #     CoG_f = np.asarray([(T_cf.dot(CoG_c[n].T)).T for n in range(z_value)])
                    #
                    # coords_c = self.dataset['atom coords'][i][heavy_atom_inds]
                    # x, y, z = self.dataset['crystal reference cell centroid x'][i], self.dataset['crystal reference cell centroid y'][i], self.dataset['crystal reference cell centroid z'][i]
                    # #ang1, ang2, ang3 = self.dataset['crystal reference cell angle 1'][i], self.dataset['crystal reference cell angle 2'][i], self.dataset['crystal reference cell angle 3'][i]
                    # ang1, ang2, ang3 = orientation_angles
                    # random_coords = randomize_molecule_position_and_orientation(coords_c, masses, T_fc, confirm_transform=False, set_position=(x, y, z), set_rotation=(ang1, ang2, ang3))
                    # generated_c, _ = build_random_crystal(T_cf, T_fc, random_coords, np.asarray(self.sym_ops[sg_number]), z_value)
                    #
                    # mol1 = Atoms(numbers=np.tile(atomic_numbers, z_value), positions=cell_coords_c.reshape(z_value * len(atomic_numbers), 3), cell=np.concatenate((cell_lengths, cell_angles)), pbc=True)
                    # mol2 = Atoms(numbers=np.tile(atomic_numbers, z_value), positions=generated_c.reshape(z_value * len(atomic_numbers), 3), cell=np.concatenate((cell_lengths, cell_angles)), pbc=True)
                    # #a1 = Analysis(mol1)
                    # #a2 = Analysis(mol2)
                    # r_maxs = np.zeros(3)
                    # for j in range(3):
                    #     axb = np.cross(mol1.cell[(j + 1) % 3, :], mol1.cell[(j + 2) % 3, :])
                    #     h = mol1.get_volume() / np.linalg.norm(axb)
                    #     r_maxs[j] = h/2.01
                    #
                    # rdf1 = get_rdf(mol1,rmax=r_maxs.min(),nbins=50)[0]
                    # rdf2 = get_rdf(mol2,rmax=r_maxs.min(),nbins=50)[0]
                    # diff = np.sum(np.abs(rdf1-rdf2))
                    # if diff > 0.1:
                    #     aa = 1
                    # diff_list.append(diff)

                    # # check our ability to standardize consistently
                    coords_c = self.dataset['atom coords'][i][heavy_atom_inds]

                    angs = np.random.uniform(-np.pi, np.pi, 3)
                    angs[1] /= 2
                    pos = np.random.uniform(0, 1, 3)
                    error = rotate_invert_and_check(atomic_numbers, masses, coords_c, T_fc, T_cf, np.asarray(self.sym_ops[sg_number]), angs, pos)
                    # error = check_standardization(atomic_numbers, masses, coords_c, T_fc, T_cf, np.asarray(self.sym_ops[sg_number]), angs, pos)

                    # # check our ability to invert angles correctly
                    # cell_coords_c, _ = build_random_crystal(T_cf, T_fc, random_coords, np.asarray(self.sym_ops[sg_number]), z_value)
                    #
                    # # get all the centroids
                    # CoG_c = np.average(cell_coords_c, axis=1)
                    # CoG_f = np.asarray([(T_cf.dot(CoG_c[n].T)).T for n in range(z_value)])
                    #
                    # # take the fractional molecule centroid closest to the origin
                    # centroid_distance_from_origin_f = np.linalg.norm(CoG_f, axis=1)
                    # canonical_mol_ind = np.argmin(centroid_distance_from_origin_f)
                    #
                    # retrieved_centroid_f, orientation_angles = retrieve_alignment_parameters(masses, cell_coords_c[canonical_mol_ind], T_fc, T_cf)
                    #
                    # if np.linalg.norm(orientation_angles - angs) > 1e-6:
                    #     aa = 1
                    #view((mol1,mol2))


        if study_cells:
            results_dict = {
                'csd_centroids_f' : np.asarray(self.centroids_f)[csd_inds],
                #'csd_supercell_angles' : np.asarray(self.supercell_angle_overlaps)[csd_inds],
                'csd_centroid_fractional_displacement' : np.asarray(self.centroid_fractional_displacement)[csd_inds],
                'csd_centroid_cartesian_displacement' : np.asarray(self.centroid_cartesian_displacement)[csd_inds],
                #'csd_inertial_angles' : np.asarray(self.inertial_angles)[csd_inds],
                'csd_orientation_angles': np.asarray(self.orientation_angles)[csd_inds],
                'rand_centroids_f': np.asarray(self.centroids_f)[rand_inds],
                #'rand_supercell_angles': np.asarray(self.supercell_angle_overlaps)[rand_inds],
                'rand_centroid_fractional_displacement': np.asarray(self.centroid_fractional_displacement)[rand_inds],
                'rand_centroid_cartesian_displacement': np.asarray(self.centroid_cartesian_displacement)[rand_inds],
                #'rand_inertial_angles': np.asarray(self.inertial_angles)[rand_inds],
                'rand_orientation_angles': np.asarray(self.orientation_angles)[rand_inds],
            }
            np.save('cell_analysis',results_dict)
        if compare_cells:
            diffs = np.asarray(diff_list)
        debug_stop = 1
        if True:
            # look at results
            plt.figure(1)
            plt.clf()
            plt.title('Canonical molecule centroid fractional distance')
            plt.hist(results_dict['csd_centroid_fractional_displacement'], density=True, bins=100, alpha=0.5,label='csd')
            plt.hist(results_dict['rand_centroid_fractional_displacement'], density=True, bins=100, alpha=0.5,label='random')
            plt.legend()

            # # nearest alignment of inertial axis to one of a,b,c
            # plt.figure(2)
            # plt.clf()
            # plt.title('Closest overlap of principal inertial axis to a, b, or c')
            # plt.hist(np.amin(results_dict['csd_inertial_angles'],axis=1), density=True, bins=100, alpha=0.5,label='csd')
            # plt.hist(np.amin(results_dict['rand_inertial_angles'],axis=1), density=True, bins=100, alpha=0.5,label='random')
            # plt.legend()
            #
            # # nearest alignment of inertial axis to a 5x5 supercell direction
            # plt.figure(3)
            # plt.clf()
            # plt.title('Closest overlap of principal inertial axis to any crystallographic direction in 5x5')
            # plt.hist(np.amin(results_dict['csd_supercell_angles'],axis=1), density=True, bins=100, alpha=0.5,label='csd')
            # plt.hist(np.amin(results_dict['rand_supercell_angles'],axis=1), density=True, bins=100, alpha=0.5,label='random')
            # plt.legend()

            plt.figure(4)
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.title('CSD fractional x-y centroid')
            plt.hist2d(x=results_dict['csd_centroids_f'][:, 0], y=results_dict['csd_centroids_f'][:, 1], bins=100,norm=colors.LogNorm())
            plt.subplot(2, 2, 2)
            plt.title('Random fractional x-y centroid')
            plt.hist2d(x=results_dict['rand_centroids_f'][:, 0], y=results_dict['rand_centroids_f'][:, 1], bins=100,norm=colors.LogNorm())
            plt.subplot(2, 2, 3)
            plt.title('CSD fractional x-z centroid')
            plt.hist2d(x=results_dict['csd_centroids_f'][:, 0], y=results_dict['csd_centroids_f'][:, 2], bins=100,norm=colors.LogNorm())
            plt.subplot(2, 2, 4)
            plt.title('Random fractional x-z centroid')
            plt.hist2d(x=results_dict['rand_centroids_f'][:, 0], y=results_dict['rand_centroids_f'][:, 2], bins=100,norm=colors.LogNorm())
            plt.tight_layout()

            plt.figure(5)
            plt.clf()
            for i in range(3):
                plt.subplot(1,3,i+1)
                plt.title('angle {}'.format(i))
                plt.hist(results_dict['csd_orientation_angles'][:,i],density=True,bins=100)
                plt.hist(results_dict['rand_orientation_angles'][:,i],density=True,bins=100,alpha=0.35)

            plt.figure(6)
            plt.clf()
            plt.subplot(2, 3, 1)
            plt.title('CSD 0-1 angles')
            plt.hist2d(x=results_dict['csd_orientation_angles'][:, 0], y=results_dict['csd_orientation_angles'][:, 1], bins=100,norm=colors.LogNorm())
            plt.subplot(2, 3, 4)
            plt.title('Random 0-1 angles')
            plt.hist2d(x=results_dict['rand_orientation_angles'][:, 0], y=results_dict['rand_orientation_angles'][:, 1], bins=100,norm=colors.LogNorm())
            plt.subplot(2, 3, 2)
            plt.title('CSD 0-2 angles')
            plt.hist2d(x=results_dict['csd_orientation_angles'][:, 0], y=results_dict['csd_orientation_angles'][:, 2], bins=100,norm=colors.LogNorm())
            plt.subplot(2, 3, 5)
            plt.title('Random 0-2 angles')
            plt.hist2d(x=results_dict['rand_orientation_angles'][:, 0], y=results_dict['rand_orientation_angles'][:, 2], bins=100,norm=colors.LogNorm())
            plt.subplot(2, 3, 3)
            plt.title('CSD 1-2 angles')
            plt.hist2d(x=results_dict['csd_orientation_angles'][:, 1], y=results_dict['csd_orientation_angles'][:, 2], bins=100,norm=colors.LogNorm())
            plt.subplot(2, 3, 6)
            plt.title('Random 1-2 angles')
            plt.hist2d(x=results_dict['rand_orientation_angles'][:, 1], y=results_dict['rand_orientation_angles'][:, 2], bins=100,norm=colors.LogNorm())
            plt.tight_layout()



    def get_cell_data(self, i, masses, T_cf, T_fc,cell_coords_c, cell_coords_f, z_value, supercell_ref_vectors_f):

        # # confirm my fractional coords agree with CSD
        # my_cell_coords_f = np.zeros_like(cell_coords_f)
        # for n in range(len(my_cell_coords_f)):
        #     my_cell_coords_f[n] = (T_cf.dot(cell_coords_c[n].T)).T
        #
        # # sometimes, the CSD fractional and cartesian coordinates disagree with each other
        # conversion_error = np.average(np.abs(my_cell_coords_f - cell_coords_f))
        # if conversion_error > 1e-5:
        #     print("fractional conversion discrepancy of {:.3f} at {}, replacing cartesian coordinates by f_to_c transform".format(conversion_error, i))
        #     cell_coords_c = np.zeros_like(cell_coords_f)
        #     for n in range(len(my_cell_coords_f)):
        #         cell_coords_c[n] = (T_fc.dot(cell_coords_f[n].T)).T

        # get all lattice vectors in cartesian coords
        lattice_f = np.eye(3)
        lattice_c = (T_fc.dot(lattice_f.T)).T

        '''
        CENTROIDS
        '''
        # get all the centroids
        CoG_c = np.average(cell_coords_c, axis=1)
        CoG_f = np.asarray([(T_cf.dot(CoG_c[n].T)).T for n in range(z_value)])
        CoM_c = np.asarray([(cell_coords_c[n].T @ masses[:, None] / np.sum(masses)).T for n in range(z_value)])[:, 0, :]  # np.transpose(coords_c.T @ masses[:, None] / np.sum(masses)) # center of mass
        CoM_f = np.asarray([(T_cf.dot(CoM_c[n].T)).T for n in range(z_value)])

        # manually enforce our style of centroid
        flag = 0
        for i in range(z_value):
            if (np.amin(CoG_f[i])) < 0 or (np.amax(CoG_f[i])> 1):
                flag = 1
                #print('Sample found with molecule out of cell')
                floor = np.floor(CoG_f[i])
                new_fractional_coords = cell_coords_f[i] - floor
                cell_coords_c[i] = T_fc.dot(new_fractional_coords.T).T
                cell_coords_f[i] = new_fractional_coords

        # redo it
        if flag:
            CoG_c = np.average(cell_coords_c, axis=1)
            CoG_f = np.asarray([(T_cf.dot(CoG_c[n].T)).T for n in range(z_value)])
            CoM_c = np.asarray([(cell_coords_c[n].T @ masses[:, None] / np.sum(masses)).T for n in range(z_value)])[:, 0, :]  # np.transpose(coords_c.T @ masses[:, None] / np.sum(masses)) # center of mass
            CoM_f = np.asarray([(T_cf.dot(CoM_c[n].T)).T for n in range(z_value)])

        # take the fractional molecule centroid closest to the origin
        centroid_distance_from_origin_c = np.linalg.norm(CoG_c, axis=1)
        centroid_distance_from_origin_f = np.linalg.norm(CoG_f, axis=1)
        canonical_mol_ind = np.argmin(centroid_distance_from_origin_f)

        '''
        INERTIAL AXES
        # look for overlaps with lattice vectors
        '''
        # # get inertial axes
        # Ip_axes_c, Ip_moments_c, I_tensor_c = compute_principal_axes_np(masses, cell_coords_c[canonical_mol_ind])
        # # get angles w.r.t. a-axis and b-axis
        # inertial_angle1 = np.arccos(np.dot(lattice_c[0] / np.linalg.norm(lattice_c[0]), (Ip_axes_c[-1]) / np.linalg.norm(Ip_axes_c[-1]))) / np.pi * 180
        # inertial_angle2 = np.arccos(np.dot(lattice_c[1] / np.linalg.norm(lattice_c[1]), (Ip_axes_c[-1]) / np.linalg.norm(Ip_axes_c[-1]))) / np.pi * 180
        # inertial_angle3 = np.arccos(np.dot(lattice_c[2] / np.linalg.norm(lattice_c[2]), (Ip_axes_c[-1]) / np.linalg.norm(Ip_axes_c[-1]))) / np.pi * 180
        #
        # # get overlaps with supercell angles
        # supercell_ref_vectors_c = (T_fc.dot(supercell_ref_vectors_f.T)).T
        # supercell_vector_angles = np.zeros(len(supercell_ref_vectors_c))
        # for n in range(len(supercell_ref_vectors_c)):
        #     supercell_vector_angles[n] = np.arccos(np.dot(supercell_ref_vectors_c[n] / np.linalg.norm(supercell_ref_vectors_c[n]), (Ip_axes_c[-1]) / np.linalg.norm(Ip_axes_c[-1]))) / np.pi * 180


        '''
        ORIENTATION ANALYSIS
        Compute the rotations necessary to achieve the 'standard' orientation (I1 aligned with (1,1,1)-(0,0,0), 
        and I2 pointed along the perpendicular vector between (1,1,1)-(0,0,0) and (0,1,1) 
        '''
        _, orientation_angles = retrieve_alignment_parameters(masses, cell_coords_c[canonical_mol_ind], T_fc, T_cf)

        self.centroids_f.append(CoG_f[canonical_mol_ind])
        #self.supercell_angle_overlaps.append(supercell_vector_angles)
        self.centroid_fractional_displacement.append(centroid_distance_from_origin_f[canonical_mol_ind])
        self.centroid_cartesian_displacement.append(centroid_distance_from_origin_c[canonical_mol_ind])
        #self.inertial_angles.append([inertial_angle1, inertial_angle2, inertial_angle3])
        #self.inertial_axes.append(Ip_axes_c)
        self.orientation_angles.append(orientation_angles)


if __name__ == '__main__':
    config = None
    mode = 'analyze'

    if mode == 'gather':
        miner = Miner(dataset_path='C:/Users\mikem\Desktop\CSP_runs\datasets/may_new_pull/mol_features', config=None, collect_chunks=True)
        miner.process_new_dataset(test_mode=False)
    elif mode == 'analyze':
        miner = Miner(dataset_path='C:/Users\mikem\Desktop\CSP_runs\datasets/test_dataset', config=None, collect_chunks=False)
        miner.todays_analysis()

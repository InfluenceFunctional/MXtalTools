import torch
from tqdm import tqdm
import os
import numpy as np
from torch_geometric.loader.dataloader import Collater
from torch_scatter import scatter
from time import time

from mxtaltools.constants.asymmetric_units import asym_unit_dict
from mxtaltools.constants.atom_properties import VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD
from mxtaltools.dataset_management.utils import basic_stats, filter_graph_nodewise, \
    filter_batch_graphwise

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
                 dataset_type=None):
        self.datapoints = None
        self.datasets_path = datasets_path
        self.chunks_path = chunks_path
        self.device = device  # cpu or cuda
        self.mode = mode  # standard or 'blind test'
        self.dataset_type = None
        self.config = config
        self.dataset_stats = None
        if dataset_type is not None:
            self.dataset_type = dataset_type
        else:
            self.dataset_type = config.type

        if self.config is not None:
            self.regression_target = self.config.regression_target if 'regression_target' in self.config.__dict__.keys() else None
            self.sample_from_trajectory = self.config.sample_from_trajectory if 'sample_from_trajectory' in self.config.__dict__.keys() else None

        np.random.seed(seed=seed if config is None else config.seed)  # for certain random sampling ops
        torch.manual_seed(seed=seed if config is None else config.seed)
        self.collater = Collater(None, None)

        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(device)

        self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()))
        self.atom_weights_tensor = torch.tensor(list(ATOM_WEIGHTS.values()))
        self.electronegativity_tensor = torch.tensor(list(ELECTRONEGATIVITY.values()))
        self.group_tensor = torch.tensor(list(GROUP.values()))
        self.period_tensor = torch.tensor(list(PERIOD.values()))

        self.times = {}

    def load_chunks(self):
        os.chdir(self.chunks_path)
        chunks = os.listdir()
        num_chunks = len(chunks)
        print(f'Collecting {num_chunks} dataset chunks')
        self.dataset = []
        for chunk in tqdm(chunks):
            if '.pkl' in chunk:
                self.dataset.extend(torch.load(chunk))

    def load_dataset_for_modelling(self, dataset_name,
                                   override_length=None,
                                   filter_conditions=None,
                                   filter_polymorphs=False,
                                   filter_duplicate_molecules=False,
                                   filter_protons=False,
                                   override_dataset=None):
        """
        Loads a dataset for use in modelling, with last minute filtering, featurization and transformations.
        # TODO note currently only set up for Z'=1 crystal structures

        Args:
            dataset_name:
            override_length:
            filter_conditions:
            filter_polymorphs:
            filter_duplicate_molecules:
            filter_protons:  # todo deprecate this
            override_dataset:

        Returns:

        """

        if override_dataset is None:
            # todo rewrite molecule cluster dataloading
            # if not os.path.exists(self.datasets_path + dataset_name) and 'dumps_dirs' in config.__dict__.keys():
            #     # convert dump files to a combined dataframe if it doesn't already exist
            #     generate_dataset_from_dumps([self.datasets_path + dir_name for dir_name in config.dumps_dirs],
            #                                 self.datasets_path + dataset_name)

            self.load_training_dataset(dataset_name)
        else:  # for loading dataset on the fly
            self.dataset = override_dataset
            if self.dataset_type == 'crystal':
                self.rebuild_crystal_indices()
            # self.override_dataDims  # todo build a constant dataDims for the overall dataset (not just chunks) and ignore the one that comes out from get_dimension
            # self.override std dict  # todo may not be necessary?
            # self.override target    # todo add dummy value?, maybe just 'radius' but will need a mean and std just so the datapoint builder doesn't crash

        self.dataset_filtration(filter_conditions, filter_duplicate_molecules, filter_polymorphs)

        self.truncate_and_shuffle_dataset(override_length)

        self.get_cell_cov_mat()
        self.assign_targets()
        self.present_atom_types = torch.unique(torch.cat([elem.x for elem in self.dataset])).tolist()
        if filter_protons:
            if 1 in self.present_atom_types:
                self.present_atom_types.remove(1)
        self.dataDims = self.get_data_dimensions()

        for ind in range(len(self.dataset)):
            if self.dataset[ind].x.ndim == 1:
                self.dataset[ind].x = self.dataset[ind].x[:, None]

    def truncate_and_shuffle_dataset(self, override_length):
        """defines train/test split as well as overall dataset size"""
        self.times['dataset_shuffle_start'] = time()
        # get dataset length & shuffle
        self.max_dataset_length = override_length if override_length is not None else self.config.max_dataset_length
        self.dataset_length = min(len(self.dataset), self.max_dataset_length)
        inds_to_keep = list(np.random.choice(len(self.dataset), self.dataset_length, replace=False))
        self.dataset = [self.dataset[ind] for ind in inds_to_keep]
        self.times['dataset_shuffle_end'] = time()

    def assign_targets(self):
        self.times['dataset_targets_assignment_start'] = time()
        targets = self.get_target()  # todo assign targets element-by-element in data list rather than as batch, omitting collation
        for ind in range(len(self.dataset)):
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
        self.molecule_keys = ['num_atoms', 'radius']
        self.lattice_keys = ['cell_a', 'cell_b', 'cell_c',
                             'cell_alpha', 'cell_beta', 'cell_gamma',
                             'aunit_x', 'aunit_y', 'aunit_z',
                             'aunit_theta', 'aunit_phi', 'aunit_r']
        if self.dataset_type == 'crystal':
            self.lattice_means = [self.dataset_stats[feat]['tight_mean'] for feat in self.lattice_keys]
            self.lattice_stds = [self.dataset_stats[feat]['tight_std'] for feat in self.lattice_keys]
        else:
            self.lattice_means = [0 for _ in range(12)]
            self.lattice_stds = [0.01 for _ in range(12)]
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
            'lattice_cov_mat': self.covariance_matrix,

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

        if self.dataset_type == 'mol_cluster':
            dim['num_polymorphs'] = self.num_polymorphs  # todo reimplement
            dim['num_topologies'] = 0

        return dim

    def get_target(self):
        if self.regression_target is not None:  # todo rewrite as data list method
            if self.regression_target == 'crystal_reduced_volume_fraction':
                targets = self.get_reduced_volume_fraction()
            elif self.regression_target == 'crystal_reduced_volume':
                targets = torch.tensor([elem.reduced_volume for elem in self.dataset])
            elif self.regression_target in qm9_targets_list:
                target_ind = qm9_targets_list.index(self.regression_target)
                targets = torch.tensor([elem.y[:, target_ind] for elem in self.dataset])
            else:
                assert False, "Unrecognized regression target"

            clipped_targets = targets.clip(min=torch.quantile(targets, 0.05), max=torch.quantile(targets, 0.95))

            self.target_mean = clipped_targets.mean()
            self.target_std = clipped_targets.std()
            if self.target_std < 1e-4:
                self.target_std = 1

            return (targets - self.target_mean) / self.target_std

        else:  # need have something just to fill the space
            self.target_mean, self.target_std = 0, 1
            return [0 for _ in range(len(self.dataset))]

    def get_reduced_volume_fraction(self):
        red_vol = torch.tensor([elem.reduced_volume for elem in self.dataset])
        atom_volumes = torch.tensor(
            [torch.sum(4 / 3 * torch.pi * self.vdw_radii_tensor[elem.x] ** 3) for elem in self.dataset])
        targets = red_vol / atom_volumes
        return targets

    # target_list = [
    #     "molecule_rotational_constant_a",
    #     "molecule_rotational_constant_b",
    #     "molecule_rotational_constant_c",
    #     "molecule_dipole_moment",
    #     "molecule_isotropic_polarizability",
    #     "molecule_HOMO_energy",
    #     "molecule_LUMO_energy",
    #     "molecule_gap_energy",
    #     "molecule_el_spatial_extent",
    #     "molecule_zpv_energy",
    #     "molecule_internal_energy_0",
    #     "molecule_internal_energy_STP",
    #     "molecule_enthalpy_STP",
    #     "molecule_free_energy_STP",
    #     "molecule_heat_capacity_STP"]
    #
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots
    #
    # fig = make_subplots(rows=4, cols=4, subplot_titles=target_list)
    # for i, target in enumerate(target_list):
    #     targets = self.dataset[target]
    #     targets = np.clip(targets, a_min=np.quantile(targets, 0.001), a_max=np.quantile(targets, 0.999))
    #     fig.add_histogram(x=targets, nbinsx=100, histnorm='probability density', row=i % 4 + 1, col=i // 4 + 1)
    #
    # fig.show(renderer='browser')

    def get_cell_cov_mat(self):
        """
        compute full covariance matrix, in raw basis
        """
        cell_params = torch.zeros((len(self.dataset), 12), dtype=torch.float32)
        for ind in range(len(self.dataset)):
            cell_params[ind] = torch.cat(
                [self.dataset[ind].cell_lengths, self.dataset[ind].cell_angles, self.dataset[ind].pose_params0], dim=1)
        self.covariance_matrix = np.cov(cell_params, rowvar=False)

        for i in range(len(self.covariance_matrix)):  # ensure it's well-conditioned
            self.covariance_matrix[i, i] = max((0.01, self.covariance_matrix[i, i]))

    def rebuild_crystal_indices(self):
        # identify which molecules are in which crystals and vice-versa
        self.crystal_to_mol_dict, self.mol_to_crystal_dict = self.generate_mol2crystal_mapping()
        self.unique_molecules_dict = self.identify_unique_molecules_in_crystals()

    def load_training_dataset(self, dataset_name):
        self.times['dataset_loading_start'] = time()
        self.dataset = torch.load(self.datasets_path + dataset_name)

        if 'batch' in str(type(self.dataset)):  # if it's batched, revert to data list - this is slow, so if possible don't store datasets as batches
            self.dataset = self.dataset.to_data_list()
            print("Dataset is in pre-collated format, which slows down initial loading!")

        """get miscellaneous data"""
        self.misc_dataset = np.load(
            self.datasets_path + 'misc_data_for_' + dataset_name[:-3].split('test_')[-1] + '.npy',
            allow_pickle=True).item()
        self.times['dataset_loading_end'] = time()
        for key in self.misc_dataset.keys():
            setattr(self, key, self.misc_dataset[key])

        if 'blind_test' in dataset_name:
            self.mode = 'blind test'
            print("Switching to blind test indexing mode")

        if 'test' in dataset_name and self.dataset_type == 'crystal':
            self.rebuild_crystal_indices()

    def process_new_dataset(self, new_dataset_name, test_dataset_size: int = 10000):
        self.load_chunks()

        ints = list(
            np.random.choice(min(len(self.dataset), test_dataset_size),
                             min(len(self.dataset), test_dataset_size),
                             replace=False))
        torch.save(self.dataset[ints], self.datasets_path + 'test_' + new_dataset_name + '.pt')

        # save full dataset
        torch.save(self.dataset, self.datasets_path + new_dataset_name + '.pt')

        # collation here so we don't slow down saving above
        self.dataset = self.collater(self.dataset)
        misc_data_dict = self.extract_misc_stats_and_indices()
        # save smaller test dataset
        np.save(self.datasets_path + 'misc_data_for_' + new_dataset_name, misc_data_dict)

    def extract_misc_stats_and_indices(self):
        misc_data_dict = {
            'dataset_stats': {
                'atomic_number': basic_stats(self.dataset.x.float()),
                'vdw_radii': basic_stats(self.vdw_radii_tensor[self.dataset.x.long()]),
                'atom_weight': basic_stats(self.atom_weights_tensor[self.dataset.x.long()]),
                'electronegativity': basic_stats(self.electronegativity_tensor[self.dataset.x.long()]),
                'group': basic_stats(self.group_tensor[self.dataset.x.long()].float()),
                'period': basic_stats(self.period_tensor[self.dataset.x.long()].float()),
                'num_atoms': basic_stats(self.dataset.num_atoms.float()),
                'radius': basic_stats(self.dataset.radius.float()),
            }
        }
        if self.dataset_type == 'crystal':
            misc_data_dict['dataset_stats'].update({
                'cell_a': basic_stats(self.dataset.cell_lengths[:, 0].float()),
                'cell_b': basic_stats(self.dataset.cell_lengths[:, 1].float()),
                'cell_c': basic_stats(self.dataset.cell_lengths[:, 2].float()),
                'cell_alpha': basic_stats(self.dataset.cell_angles[:, 0].float()),
                'cell_beta': basic_stats(self.dataset.cell_angles[:, 1].float()),
                'cell_gamma': basic_stats(self.dataset.cell_angles[:, 2].float()),
                'aunit_x': basic_stats(self.dataset.pose_params0[:, 0].float()),
                'aunit_y': basic_stats(self.dataset.pose_params0[:, 1].float()),
                'aunit_z': basic_stats(self.dataset.pose_params0[:, 2].float()),
                'aunit_theta': basic_stats(self.dataset.pose_params0[:, 3].float()),
                'aunit_phi': basic_stats(self.dataset.pose_params0[:, 4].float()),
                'aunit_r': basic_stats(self.dataset.pose_params0[:, 5].float()),
                'z_prime': basic_stats(self.dataset.z_prime.float()),
                'cell_volume': basic_stats(self.dataset.cell_volume.float()),
                'reduced_volume': basic_stats(self.dataset.reduced_volume.float()),
            })
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
            for _ in range(int(self.dataset[index].z_prime)):  # assumes integer Z'
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
        fingerprints = []
        for z1 in range(len(self.dataset)):
            zp = int(self.dataset[z1].z_prime)
            for ind in range(zp):
                fingerprints.append(self.dataset[z1].fingerprint[2048 * ind:2048 * (ind + 1)])
        fps = np.stack(fingerprints)

        unique_fps, inverse_map = np.unique(fps, axis=0, return_inverse=True)
        molecules_in_crystals_dict = {unique.tobytes(): [] for unique in unique_fps}
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
            values = torch.tensor([elem.z_prime for elem in self.dataset])
        elif condition_key == 'asymmetric_unit_is_well_defined':
            values = torch.tensor([elem.is_well_defined for elem in self.dataset])
        elif condition_key == 'crystal_symmetry_operations_are_nonstandard':
            values = torch.tensor([elem.nonstandard_symmetry for elem in self.dataset])
        elif condition_key == 'max_atomic_number':
            # always take the largest value - this is what we are practically filtering
            values = torch.tensor([elem.x.amax() for elem in self.dataset])
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

    miner.load_dataset_for_modelling(dataset_name='dataset.pt',
                                     filter_conditions=[
                                         ['crystal_z_prime', 'in', [1]],
                                         # NOTE we can currently only process Z' == 1 in models
                                         ['crystal_symmetry_operations_are_nonstandard', 'in', [False]],
                                         ['max_atomic_number', 'range', [1, 100]],
                                         #['molecule_is_symmetric_top','in',[False]],
                                         #['molecule_is_spherical_top','in',[False]],
                                         #['crystal_packing_coefficient','range',[0.55, 0.85]],
                                         ['molecule_num_atoms', 'range', [3, 100]],
                                         ['molecule_radius', 'range', [1, 5]],
                                         ['asymmetric_unit_is_well_defined', 'in', [True]],
                                         ['reduced_volume_fraction', 'range', [0.75, 1.15]],

                                         #['crystal_identifier', 'not_in', ['OBEQUJ', 'OBEQOD', 'OBEQET', 'XATJOT', 'OBEQIX', 'KONTIQ','NACJAF', 'XAFPAY', 'XAFQON', 'XAFQIH', 'XAFPAY01', 'XAFPAY02', 'XAFPAY03', 'XAFPAY04','XAFQON','XAFQIH']],  # omit blind test 5 & 6 targets
                                         #['crystal_space_group_number','in',[2,14,19]]
                                     ])

import torch
import numpy as np
from utils import standardize
from dataset_management.CrystalData import CrystalData
from torch_geometric.data import Data
import sys
from torch_geometric.loader import DataLoader
import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
from pyxtal import symmetry
from dataset_management.dataset_manager import Miner


class BuildDataset:
    """
    build dataset object
    """

    def __init__(self, config, dataset_path=None, pg_dict=None, sg_dict=None, lattice_dict=None,
                 premade_dataset=None, replace_dataDims=None, override_length=None):
        self.target = config.target
        self.max_atomic_number = config.max_atomic_number
        self.atom_dict_size = {'atom z': self.max_atomic_number + 1}  # for embeddings
        self.dataset_seed = config.seeds.dataset
        self.max_temperature = config.max_crystal_temperature
        self.min_temperature = config.min_crystal_temperature
        self.max_num_atoms = config.max_num_atoms
        self.min_num_atoms = config.min_num_atoms
        self.min_z_value = config.min_z_value
        self.max_z_value = config.max_z_value
        self.min_packing_coefficient = config.min_packing_coefficient
        self.include_organic = config.include_organic
        self.include_organometallic = config.include_organometallic
        self.model_mode = config.mode
        self.conditional_modelling = config.generator.conditional_modelling
        self.include_sgs = config.include_sgs
        self.conditioning_mode = config.generator.conditioning_mode
        self.include_pgs = config.include_pgs
        self.replace_dataDims = replace_dataDims
        if override_length is not None:
            self.max_dataset_length = override_length
        else:
            self.max_dataset_length = config.dataset_length
        self.feature_richness = config.feature_richness

        self.set_keys()
        self.get_syms(pg_dict, sg_dict, lattice_dict)

        '''
        actually load the dataset
        '''

        if premade_dataset is None:
            if dataset_path is None:
                dataset = pd.read_pickle('datasets/dataset')
            else:
                dataset = pd.read_pickle(dataset_path)
        else:
            dataset = premade_dataset
        self.dataset_length = len(dataset)
        self.final_dataset_length = min(self.dataset_length, self.max_dataset_length)

        dataset = self.add_last_minute_features_quickly(dataset, config)  # add a few odds & ends

        '''
        prep for modelling
        '''

        lattice_features = self.get_cell_features(dataset)
        targets = self.get_targets(dataset)
        self.datapoints = self.generate_training_datapoints(dataset, lattice_features, targets, config)

        self.shuffle_datapoints()

    def set_keys(self):
        # define relevant features for analysis
        if self.feature_richness == 'full':
            self.atom_keys = ['atom Z',
                              'atom mass', 'atom is H bond acceptor',
                              'atom valence', 'atom vdW radius',  # 'atom is aromatic', # issue with aromaticity in test sets
                              'atom on a ring', 'atom degree', 'atom electronegativity']  # 'atom chirality', todo check chirality measure
            self.molecule_keys = ['molecule volume',
                                  'molecule mass', 'molecule num atoms', 'molecule volume', 'molecule point group is C1',
                                  'molecule num rings', 'molecule num donors', 'molecule num acceptors',
                                  'molecule num rotatable bonds', 'molecule planarity', 'molecule polarity',
                                  'molecule spherical defect', 'molecule eccentricity', 'molecule radius of gyration',
                                  'molecule principal moment 1', 'molecule principal moment 2', 'molecule principal moment 3',
                                  ]
        elif self.feature_richness == 'minimal':
            self.atom_keys = ['atom Z']
            self.molecule_keys = ['molecule volume']

        self.crystal_keys = ['crystal spacegroup symbol', 'crystal spacegroup number',
                             'crystal calculated density', 'crystal packing coefficient',
                             'crystal lattice centring', 'crystal system',
                             'crystal alpha', 'crystal beta', 'crystal gamma',
                             'crystal cell a', 'crystal cell b', 'crystal cell c',
                             'crystal z value', 'crystal z prime',  # 'crystal point group',
                             ]

        # for coupling NF models, must be an even number of these
        self.lattice_keys = ['crystal cell a', 'crystal cell b', 'crystal cell c',
                             'crystal alpha', 'crystal beta', 'crystal gamma',
                             'crystal asymmetric unit centroid x', 'crystal asymmetric unit centroid y', 'crystal asymmetric unit centroid z',
                             'crystal asymmetric unit rotvec 1', 'crystal asymmetric unit rotvec 2', 'crystal asymmetric unit rotvec 3',
                             ]

        if len(self.lattice_keys) % 2 != 0:  # coupling flow model requires an even number of dimensions
            print('For coupling flow, expect # latent dimensions to be even!')

        return

    def get_syms(self, pg_dict=None, sg_dict=None, lattice_dict=None):
        # get crystal symmetry factors
        if pg_dict is not None:
            self.pg_dict = pg_dict
            self.sg_dict = sg_dict
            self.lattice_dict = lattice_dict
        else:
            # get crystal symmetry factors
            self.pg_dict = {}
            self.sg_dict = {}
            self.lattice_dict = {}
            for i in range(1, 231):
                sym_group = symmetry.Group(i)
                self.pg_dict[i] = sym_group.point_group
                self.sg_dict[i] = sym_group.symbol
                self.lattice_dict[i] = sym_group.lattice_type

        if self.include_sgs is not None:
            print("Modelling with crystals from " + str(self.include_sgs))

    def add_last_minute_features_quickly(self, dataset, config):
        '''
        add some missing one-hot features
        '''

        '''
        z value
        '''
        for i in range(config.min_z_value + 1, config.max_z_value + 1):
            dataset['crystal z is {}'.format(i)] = dataset['crystal z value'] == i

        '''
        space group
        '''
        for i, symbol in enumerate(np.unique(list(self.sg_dict.values()))):
            dataset['crystal sg is ' + symbol] = dataset['crystal spacegroup symbol'] == symbol

        '''
        crystal system
        '''
        # get dictionary for crystal system elements
        for i, system in enumerate(np.unique(list(self.lattice_dict.values()))):
            dataset['crystal system is ' + system] = dataset['crystal system'] == system

        # '''
        # crystal point group
        # '''
        # # get crystal point groups and make an ordered dict
        # for i, group in enumerate(np.unique(list(self.pg_dict.values()))):
        #     dataset['crystal pg is ' + group] = dataset['crystal point group'] == group

        '''
        # set angle units to natural
        '''
        if (max(dataset['crystal alpha']) > np.pi) or (max(dataset['crystal beta']) > np.pi) or (max(dataset['crystal gamma']) > np.pi):
            dataset['crystal alpha'] = dataset['crystal alpha'] * np.pi / 180
            dataset['crystal beta'] = dataset['crystal beta'] * np.pi / 180
            dataset['crystal gamma'] = dataset['crystal gamma'] * np.pi / 180

        aa = 1
        '''
        recalculate crystal density
        '''
        from nikos.coordinate_transformations import cell_vol
        cell_volume = np.asarray([cell_vol(
            [dataset['crystal cell a'][i],dataset['crystal cell b'][i],dataset['crystal cell c'][i]],
            [dataset['crystal alpha'][i], dataset['crystal beta'][i], dataset['crystal gamma'][i]])
            for i in range(len(dataset))
        ])
        mass = dataset['molecule mass']
        Z = dataset['crystal z value']
        dataset['crystal density'] = mass * Z / cell_volume
        #density = mass * Z / cell_volume
        #density2 = (mass * Z) / (dataset['molecule volume'] * Z / dataset['crystal packing coefficient'])
        # '''
        # add functional group information
        # '''
        # if not 'molecule B fraction' in dataset.keys():
        #     from dataset_management.molecule_featurizer import get_fraction
        #     from mendeleev import element as element_table
        #
        #     for anum in range(1, 36):
        #         dataset[f'molecule {element_table(anum).symbol} fraction'] = np.asarray([get_fraction(atom_list, anum) for atom_list in dataset['atom Z']])

        return dataset

    def shuffle_datapoints(self):
        np.random.seed(self.dataset_seed)
        good_inds = np.random.choice(self.final_dataset_length, size=self.final_dataset_length, replace=False)
        self.dataset_length = len(good_inds)

        self.datapoints = [self.datapoints[i] for i in good_inds]

    def generate_training_data(self, atom_coords, smiles, atom_features_list, mol_features,
                               targets, tracking_features, reference_cells, lattice_features,
                               T_fc_list, identifiers, asymmetric_unit_handedness, crystal_symmetries):
        '''
        convert feature, target and tracking vectors into torch.geometric data objects
        :param atom_coords:
        :param smiles:
        :param atom_features_list:
        :param mol_features:
        :param targets:
        :param tracking_features:
        :return:
        '''
        datapoints = []

        z_value_ind = self.tracking_dict_keys.index('crystal z value')
        sg_ind_value_ind = self.tracking_dict_keys.index('crystal spacegroup number')
        mol_size_ind = self.tracking_dict_keys.index('molecule num atoms')
        mol_volume_ind = self.tracking_dict_keys.index('molecule volume')

        tracking_features = torch.Tensor(tracking_features)
        print("Generating final training datapoints")
        for i in tqdm.tqdm(range(self.final_dataset_length)):
            if targets[i].ndim == 1:
                target = torch.tensor(targets[i][np.newaxis, :])
            else:
                target = torch.tensor(targets[i])

            # append molecule features to atom features for each atom
            input_features = np.concatenate((atom_features_list[i], np.repeat(mol_features[i][np.newaxis, :], len(atom_features_list[i]), axis=0)), axis=1)

            input_features = torch.Tensor(input_features)
            assert torch.sum(torch.isnan(input_features)) == 0, "NaN in training input"
            # datapoints.append(Data(x=input_features.float(), pos=torch.Tensor(atom_coords[i]), y=[target, smiles[i], tracking_features[i], reference_cells[i]]))
            datapoints.append(CrystalData(x=input_features.float(),
                                          pos=torch.Tensor(atom_coords[i]),
                                          y=target,
                                          smiles=smiles[i],
                                          tracking=tracking_features[i, None, :],
                                          ref_cell_pos=reference_cells[i][:, :, :3],
                                          Z=tracking_features[i, z_value_ind].int(),
                                          sg_ind=tracking_features[i, sg_ind_value_ind].int(),
                                          cell_params=torch.Tensor(lattice_features[i, None, :]),
                                          T_fc=torch.Tensor(T_fc_list[i])[None, ...],
                                          mol_size=torch.Tensor(tracking_features[i, mol_size_ind]),
                                          mol_volume=torch.Tensor(tracking_features[i, mol_volume_ind]),
                                          csd_identifier=identifiers[i],
                                          asym_unit_handedness=torch.Tensor(np.asarray(asymmetric_unit_handedness[i])[None]),
                                          symmetry_operators=crystal_symmetries[i]
                                          ))

        return datapoints

    def concatenate_atom_features(self, dataset):
        """
        collect and normalize/standardize relevant atomic features
        :param dataset:
        :return:
        """

        keys_to_add = self.atom_keys
        print("Preparing atom-wise features")
        if self.replace_dataDims is not None:
            stds = self.replace_dataDims['atom stds']
            means = self.replace_dataDims['atom means']
        else:
            stds, means = {}, {}
        atom_features_list = [np.zeros((len(dataset['atom Z'][i]), len(keys_to_add))) for i in range(self.dataset_length)]

        for column, key in enumerate(keys_to_add):
            flat_feature = np.concatenate(dataset[key])
            if self.replace_dataDims is None:
                stds[key] = np.std(flat_feature)
                if stds[key] < 0.01:
                    stds[key] = 0.01  # make sure it well-conditioned
                means[key] = np.mean(flat_feature)

            for i in range(self.dataset_length):
                feature_vector = dataset[key][i]

                if type(feature_vector) is not np.ndarray:
                    feature_vector = np.asarray(feature_vector)

                if key == 'atom Z':
                    pass
                elif feature_vector.dtype == bool:
                    pass
                elif (feature_vector.dtype == float) or (np.issubdtype(feature_vector.dtype, np.floating)):
                    feature_vector = standardize(feature_vector, known_std=stds[key], known_mean=means[key])
                elif (feature_vector.dtype == int) or (np.issubdtype(feature_vector.dtype, np.integer)):
                    # if len(np.unique(feature_vector)) > 2:
                    feature_vector = standardize(feature_vector, known_std=stds[key], known_mean=means[key])
                    # else:
                    #     feature_vector = np.asarray(feature_vector == np.amax(feature_vector))  # turn it into a bool

                assert np.sum(np.isnan(feature_vector)) == 0
                atom_features_list[i][:, column] = feature_vector

        self.atom_means = means
        self.atom_stds = stds
        return atom_features_list

    def concatenate_molecule_features(self, dataset, mol_keys=True, extra_keys=None, add_lattice_overlaps=False):
        """
        collect features of 'molecules' and append to atom-level data
        """
        # normalize everything
        keys_to_add = []
        if mol_keys:
            keys_to_add.extend(self.molecule_keys)

        if extra_keys is not None:
            keys_to_add.extend(extra_keys)

        if False:  # add_lattice_overlaps: # todo not ready yet - also make sure only the discriminator sees this, and it gets properly updated when new cells are built
            lattice_overlap_keys = [key for key in dataset.columns if 'inertial overlap' in key]
            keys_to_add.extend(lattice_overlap_keys)
            # overlaps = np.asarray([dataset[key] for key in lattice_overlap_keys]).T

        print("Preparing molcule-wise features")
        if self.target in keys_to_add:  # don't add molecule target if we are going to model it
            keys_to_add.remove(self.target)

        if self.replace_dataDims is not None:
            stds = self.replace_dataDims['molecule stds']
            means = self.replace_dataDims['molecule means']
        else:
            stds, means = {}, {}

        molecule_feature_array = np.zeros((self.dataset_length, len(keys_to_add)), dtype=float)
        for column, key in enumerate(keys_to_add):
            feature_vector = dataset[key]
            if self.replace_dataDims is None:
                stds[key] = np.std(feature_vector)
                if stds[key] < 0.01:
                    stds[key] = 0.01  # make sure it's well conditioned
                means[key] = np.mean(feature_vector)

            if type(feature_vector) is not np.ndarray:
                feature_vector = np.asarray(feature_vector)

            if feature_vector.dtype == bool:
                pass
            elif key == 'crystal z value':
                pass  # don't normalize Z value, for now
            elif (feature_vector.dtype == float) or (np.issubdtype(feature_vector.dtype, np.floating)):
                feature_vector = standardize(feature_vector, known_mean=means[key], known_std=stds[key])
            elif (feature_vector.dtype == int) or (np.issubdtype(feature_vector.dtype, np.integer)):
                # if len(np.unique(feature_vector)) > 2:
                feature_vector = standardize(feature_vector, known_mean=means[key], known_std=stds[key])
                # else:
                #     feature_vector = np.asarray(feature_vector == np.amax(feature_vector))  # turn it into a bool

            molecule_feature_array[:, column] = feature_vector

        self.num_mol_features = len(keys_to_add)
        self.mol_keys = keys_to_add
        self.mol_means = means
        self.mol_stds = stds
        assert np.sum(np.isnan(molecule_feature_array)) == 0
        return molecule_feature_array

    def generate_training_datapoints(self, dataset, lattice_features, targets, config):
        tracking_features = self.gather_tracking_features(dataset)

        # add symmetry features for generator
        self.crystal_generation_features = []  # todo need an option to turn this off for certain models
        # point_group_features = [column for column in dataset.columns if 'pg is' in column]
        space_group_features = [column for column in dataset.columns if 'sg is' in column]
        crystal_system_features = [column for column in dataset.columns if 'crystal system is' in column]
        # self.crystal_generation_features.extend(point_group_features)
        self.crystal_generation_features.extend(space_group_features)
        self.crystal_generation_features.extend(crystal_system_features)
        self.crystal_generation_features.append('crystal z value')  # todo norm this

        molecule_features_array = self.concatenate_molecule_features(
            dataset, extra_keys=self.crystal_generation_features, add_lattice_overlaps=True)

        atom_features_list = self.concatenate_atom_features(dataset)

        if 'crystal symmetries' not in dataset.columns:
            dataset['crystal symmetries'] = [[] for _ in range(len(dataset))]
            print('No crystal symmetries in the dataset!')

        return self.generate_training_data(atom_coords=dataset['atom coords'],
                                           smiles=dataset['molecule smiles'],
                                           atom_features_list=atom_features_list,
                                           mol_features=molecule_features_array,
                                           targets=targets,
                                           tracking_features=tracking_features,
                                           reference_cells=dataset['crystal reference cell coords'],
                                           lattice_features=lattice_features,
                                           T_fc_list=dataset['crystal fc transform'],
                                           identifiers=dataset['identifier'],
                                           asymmetric_unit_handedness=dataset['crystal asymmetric unit handedness'],
                                           crystal_symmetries=dataset['crystal symmetries'])

    def get_cell_features(self, dataset):
        keys_to_add = self.lattice_keys
        key_dtype = []
        # featurize
        if self.replace_dataDims is not None:  # use mean & std from an external dataset
            stds = self.replace_dataDims['lattice stds']
            means = self.replace_dataDims['lattice means']
        else:
            stds, means = [], []

        feature_array = np.zeros((self.dataset_length, len(keys_to_add)), dtype=float)
        for column, key in enumerate(keys_to_add):
            feature_vector = dataset[key]
            if type(feature_vector) is not np.ndarray:
                feature_vector = np.asarray(feature_vector)

            if key == 'crystal z value':
                key_dtype.append('int32')
            else:
                key_dtype.append(feature_vector.dtype)

            if self.replace_dataDims is None:  # record mean & std for each feature
                mean = np.average(feature_vector)
                std = np.std(feature_vector)
                if (np.isnan(np.std(feature_vector))):
                    std = 1
                if (np.std(feature_vector) == 0):
                    std = 0.01
                means.append(mean)
                stds.append(std)

            # no need to standardize here
            feature_array[:, column] = feature_vector

            assert np.sum(np.isnan(feature_vector)) == 0

        '''
        compute full covariance matrix, in normalized basis
        '''
        # normalize the cell lengths against molecule volume & z value
        normed_cell_lengths = feature_array[:, :3] / (dataset['crystal z value'][:, None] ** (1 / 3)) / (dataset['molecule volume'][:, None] ** (1 / 3))
        feature_array_with_normed_lengths = feature_array.copy()
        feature_array_with_normed_lengths[:, :3] = normed_cell_lengths

        self.normed_lengths_means = np.mean(normed_cell_lengths, axis=0)
        self.normed_lengths_stds = np.std(normed_cell_lengths, axis=0)
        covariance_matrix = np.cov(feature_array_with_normed_lengths, rowvar=False)  # we want the randn model to generate samples with normed lengths

        for i in range(len(covariance_matrix)):  # ensure it's well-conditioned
            covariance_matrix[i, i] = max((0.01, covariance_matrix[i, i]))

        assert np.sum(np.isnan(stds)) == 0
        assert np.sum(np.asarray(stds) == 0) == 0

        self.lattice_means, self.lattice_stds, self.lattice_dtypes, self.covariance_matrix = \
            means, stds, key_dtype, covariance_matrix

        return feature_array

    def get_targets(self, dataset):
        if self.target == 'packing':
            targets = dataset['crystal packing coefficient']
        elif self.target == 'density':
            conversion = 1660  # 1 amu / cubic angstrom is 1.660 kg / m^3
            targets = dataset['molecule mass'] * dataset['crystal packing coefficient'] / dataset['molecule volume'] * conversion  # this is per-molecule, divide by Z to get the full crystal
        elif self.target == 'volume':
            targets = dataset['molecule volume'] / dataset['crystal packing coefficient']  # this is per-molecule, multiply by Z to get the full crystal value
        elif self.target == 'lattice vector':
            targets = dataset['crystal inertial overlap 2 to 0']
        else:
            print(f'{self.target} is not an implemented regression target!')
            sys.exit()

        self.target_std = targets.std()
        self.target_mean = targets.mean()

        return (targets - self.target_mean) / self.target_std

    def gather_tracking_features(self, dataset):
        """
        collect features of 'molecules' and append to atom-level data
        """
        # normalize everything
        keys_to_add = []
        keys_to_add.extend(['molecule volume', 'molecule mass', 'molecule num atoms', 'molecule point group is C1',
                            'molecule num rings', 'molecule num donors', 'molecule num acceptors',
                            'molecule num rotatable bonds', 'molecule planarity', 'molecule polarity',
                            'molecule spherical defect', 'molecule eccentricity', 'molecule radius of gyration',
                            'molecule principal moment 1', 'molecule principal moment 2', 'molecule principal moment 3',
                            'crystal r factor', 'crystal density'
                            ])

        keys_to_add.extend(self.crystal_keys)
        if 'crystal spacegroup symbol' in keys_to_add:
            keys_to_add.remove('crystal spacegroup symbol')  # we don't want to deal with strings
        if ('crystal system' in keys_to_add):
            keys_to_add.remove('crystal system')  # we don't want to deal with strings
        if ('crystal lattice centring' in keys_to_add):
            keys_to_add.remove('crystal lattice centring')  # we don't want to deal with strings
        if ('crystal point group' in keys_to_add):
            keys_to_add.remove('crystal point group')  # we don't want to deal with strings

        for key in dataset.keys():
            if ('molecule' in key) and ('fraction' in key):
                keys_to_add.append(key)
            if ('molecule has' in key):
                keys_to_add.append(key)

        print("Preparing molecule/crystal tracking features")
        if self.target in keys_to_add:  # don't add molecule target if we are going to model it
            keys_to_add.remove(self.target)

        feature_array = np.zeros((self.dataset_length, len(keys_to_add)), dtype=float)
        for column, key in enumerate(keys_to_add):
            if key == 'crystal r factor':
                feature_vector_i = np.asarray(dataset[key])
                feature_vector = np.zeros_like(feature_vector_i)
                for jj in range(len(feature_vector)):
                    if feature_vector_i[jj] != None:
                        if feature_vector_i[jj].lower() != 'none':
                            feature_vector[jj] = float(feature_vector_i[jj])

            else:
                feature_vector = dataset[key]
            if type(feature_vector) is not np.ndarray:
                feature_vector = np.asarray(feature_vector)

            if feature_vector.dtype == bool:
                pass
            elif feature_vector.dtype == float:
                feature_vector = feature_vector
            elif feature_vector.dtype == int:
                if len(np.unique(feature_vector)) > 2:
                    feature_vector = feature_vector
                else:
                    feature_vector = np.asarray(feature_vector == np.amax(feature_vector))  # turn it into a bool

            feature_array[:, column] = feature_vector

        self.n_tracking_features = len(keys_to_add)
        # store known info for training analysis
        self.tracking_dict_keys = keys_to_add

        return feature_array

    def get_dimension(self):
        dim = {
            'dataset length': len(self.datapoints),

            'lattice features': self.lattice_keys,
            'num lattice features': len(self.lattice_keys),
            'lattice means': self.lattice_means,
            'lattice stds': self.lattice_stds,
            'lattice cov mat': self.covariance_matrix,
            'lattice normed length means': self.normed_lengths_means,
            'lattice normed length stds': self.normed_lengths_stds,
            'lattice dtypes': self.lattice_dtypes,

            'target mean': self.target_mean,
            'target std': self.target_std,

            'num tracking features': self.n_tracking_features,
            'tracking features dict': self.tracking_dict_keys,

            'num atom features': len(self.atom_keys) + len(self.mol_keys),
            'num atomwise features': len(self.atom_keys),
            'atom means': self.atom_means,
            'atom stds': self.atom_stds,

            'num mol features': len(self.mol_keys),
            'molecule features': self.mol_keys,
            'molecule means': self.mol_means,
            'molecule stds': self.mol_stds,

            'atom embedding dict sizes': self.atom_dict_size,

            'conditional features': self.molecule_keys + self.crystal_generation_features,
            'num conditional features': len(self.crystal_generation_features + self.molecule_keys),

            'crystal generation features': self.crystal_generation_features,
            'num crystal generation features': len(self.crystal_generation_features),

            'space groups to search': self.include_sgs,
        }

        return dim

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def __len__(self):
        return len(self.datapoints)


def get_dataloaders(dataset_builder, config, override_batch_size=None):
    if override_batch_size is not None:
        batch_size = override_batch_size
    else:
        batch_size = config.max_batch_size
    train_size = int(0.8 * len(dataset_builder))  # split data into training and test sets
    test_size = len(dataset_builder) - train_size

    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size):
        train_dataset.append(dataset_builder[i])
    for i in range(test_size):
        test_dataset.append(dataset_builder[i])

    tr = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    te = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    return tr, te


def update_batch_size(loader, new_batch_size):
    return DataLoader(loader.dataset, batch_size=new_batch_size, shuffle=True, num_workers=0, pin_memory=False)


def delete_from_dataset(dataset, good_inds):
    print("Deleting unwanted entries")

    for key in dataset.keys():
        if type(dataset[key]) == list:
            dataset[key] = [dataset[key][i] for i in good_inds]
        elif type(dataset[key]) == np.ndarray:
            dataset[key] = dataset[key][np.asarray(good_inds)]

    return dataset


def get_extra_test_loader(config, paths, dataDims, pg_dict=None, sg_dict=None, lattice_dict=None):
    datasets = []
    for path in paths:
        miner = Miner(config=config, dataset_path=path, collect_chunks=False)
        miner.exclude_nonstandard_settings = False
        miner.exclude_crystal_systems = None
        miner.exclude_polymorphs = False
        miner.exclude_missing_r_factor = False
        miner.exclude_blind_test_targets = False
        dataset_i = miner.load_for_modelling(save_dataset=False, return_dataset=True)
        if config.test_mode:
            np.random.seed(config.seeds.dataset)
            randinds = np.random.choice(len(dataset_i), min(len(dataset_i), 500), replace=False)
            dataset_i = dataset_i.loc[randinds]
        datasets.append(dataset_i)
        del miner, dataset_i

    dataset = pd.concat(datasets)
    if 'level_0' in dataset.columns:  # housekeeping
        dataset = dataset.drop(columns='level_0')
    dataset = dataset.reset_index()

    dataset = dataset.drop('crystal symmetries', axis=1)  # can't mix nicely # todo delete this after next BT refeaturization

    extra_test_set_builder = BuildDataset(config, pg_dict=pg_dict,
                                          sg_dict=sg_dict,
                                          lattice_dict=lattice_dict,
                                          replace_dataDims=dataDims,
                                          override_length=len(dataset),
                                          premade_dataset=dataset)

    extra_test_loader = DataLoader(extra_test_set_builder.datapoints, batch_size=config.final_batch_size, shuffle=False, num_workers=0, pin_memory=False)
    del dataset, extra_test_set_builder
    return extra_test_loader

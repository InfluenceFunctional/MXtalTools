import torch
import numpy as np
from utils import standardize
from torch_geometric.data import Data
import sys
from torch_geometric.loader import DataLoader
import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
from pyxtal import symmetry


class BuildDataset:
    """
    build dataset object
    """

    def __init__(self, config, pg_dict=None, sg_dict=None, lattice_dict = None, premade_dataset=None):
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

        self.set_keys()

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

        '''
        actually load the dataset
        '''

        if premade_dataset is None:
            dataset = pd.read_pickle('datasets/dataset')
        else:
            dataset = premade_dataset
        self.dataset_length = len(dataset)
        self.final_dataset_length = min(self.dataset_length, config.dataset_length)

        dataset = self.add_last_minute_features_quickly(dataset, config)

        '''
        prep for modelling
        '''

        self.datapoints, self.means, self.stds, self.dtypes = self.get_joint_features(dataset)
        self.datapoints = self.generate_joint_training_datapoints(dataset, config)


        self.shuffle_datapoints()

    def set_keys(self):
        # define relevant features for analysis
        self.atom_keys = ['atom Z', 'atom mass', 'atom is H bond donor', 'atom is H bond acceptor',
                          'atom is aromatic', 'atom valence', 'atom vdW radius',
                          'atom on a ring', 'atom chirality', 'atom degree', 'atom electronegativity']

        self.crystal_keys = ['crystal spacegroup symbol', 'crystal spacegroup number',
                             'crystal calculated density', 'crystal packing coefficient',
                             'crystal lattice centring', 'crystal system',
                             'crystal alpha', 'crystal beta', 'crystal gamma',
                             'crystal cell a', 'crystal cell b', 'crystal cell c',
                             'crystal z value', 'crystal z prime', 'crystal point group',
                             ]
        self.molecule_keys = ['molecule point group is C1', 'molecule mass', 'molecule num atoms', 'molecule volume',
                              'molecule num rings', 'molecule n donors', 'molecule n acceptors',
                              'molecule n rotatable bonds', 'molecule planarity', 'molecule polarity',
                              'molecule spherical defect', 'molecule eccentricity', 'molecule radius of gyration',
                              'molecule principal moment 1', 'molecule principal moment 2', 'molecule principal moment 3',
                              ]
        # for coupling NF models, must be an even number of these
        self.lattice_keys = ['crystal cell a', 'crystal cell b', 'crystal cell c',
                             'crystal alpha', 'crystal beta', 'crystal gamma',
                             'crystal reference cell centroid x', 'crystal reference cell centroid y', 'crystal reference cell centroid z',
                             'crystal reference cell angle 1', 'crystal reference cell angle 2', 'crystal reference cell angle 3',
                             ]

        if len(self.lattice_keys) % 2 != 0:  # coupling flow model requires an even number of dimensions
            print('For coupling flow, expect # latent dimensions to be even!')

        return

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
        molecule symmetry
        '''

        dataset['molecule point group is C1'] = dataset['molecule point group'] == 'C1'

        '''
        space group
        '''
        if self.include_sgs is not None:
            self.include_sgs.append(config.generate_sgs) # make sure the searching group is always present

            for key in self.include_sgs:
                dataset['crystal sg is ' + key] = dataset['crystal spacegroup symbol'] == key

        else:
            self.include_sgs = list(np.unique(dataset['crystal spacegroup symbol']))
            for key in self.include_sgs:
                dataset['crystal sg is ' + key] = dataset['crystal spacegroup symbol'] == key
        '''
        crystal system
        '''
        # get dictionary for crystal system elements
        for i, system in enumerate(np.unique(list(self.lattice_dict.values()))):
            dataset['crystal system is ' + system] = dataset['crystal system'] == system

        '''
        crystal point group
        '''
        # get crystal point groups and make an ordered dict
        for i, group in enumerate(np.unique(list(self.pg_dict.values()))):
            dataset['crystal pg is ' + group] = dataset['crystal point group'] == group

        '''
        # set angle units to natural
        '''
        dataset['crystal alpha'] = dataset['crystal alpha'] * np.pi / 180
        dataset['crystal beta'] = dataset['crystal beta'] * np.pi / 180
        dataset['crystal gamma'] = dataset['crystal gamma'] * np.pi / 180

        return dataset

    def shuffle_datapoints(self):
        np.random.seed(self.dataset_seed)
        good_inds = np.random.choice(self.final_dataset_length, size=self.final_dataset_length, replace=False)
        self.dataset_length = len(good_inds)

        self.datapoints = [self.datapoints[i] for i in good_inds]

    def shuffle_final_dataset(self, atom_features_list, molecule_features_array, targets, smiles, coords, tracking_features):
        np.random.seed(self.dataset_seed)
        good_inds = np.random.choice(self.dataset_length, size=self.final_dataset_length, replace=False)
        self.dataset_length = len(good_inds)
        return [atom_features_list[i] for i in good_inds], molecule_features_array[good_inds], targets[good_inds], \
               [smiles[i] for i in good_inds], [coords[i] for i in good_inds], tracking_features[good_inds]

    def generate_training_data(self, atom_coords, smiles, atom_features_list, mol_features, targets, tracking_features, reference_cells=None):
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
            if 'cell' in self.model_mode:
                datapoints.append(Data(x=input_features.float(), pos=torch.Tensor(atom_coords[i]), y=[target, smiles[i], tracking_features[i], reference_cells[i]]))
            else:
                datapoints.append(Data(x=input_features.float(), pos=torch.Tensor(atom_coords[i]), y=[target, smiles[i], tracking_features[i]]))

        return datapoints

    def concatenate_atom_features(self, dataset):
        """
        collect and normalize/standardize relevant atomic features
        :param dataset:
        :return:
        """

        keys_to_add = self.atom_keys
        print("Preparing atom-wise features")
        stds, means = {}, {}
        atom_features_list = [np.zeros((len(dataset['atom Z'][i]), len(keys_to_add))) for i in range(self.dataset_length)]

        for column, key in enumerate(keys_to_add):
            flat_feature = np.concatenate(dataset[key])
            stds[key] = np.std(flat_feature)
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
                    if len(np.unique(feature_vector)) > 2:
                        feature_vector = standardize(feature_vector, known_std=stds[key], known_mean=means[key])
                    else:
                        feature_vector = np.asarray(feature_vector == np.amax(feature_vector))  # turn it into a bool

                assert np.sum(np.isnan(feature_vector)) == 0
                atom_features_list[i][:, column] = feature_vector

        return atom_features_list

    def concatenate_molecule_features(self, dataset, mol_keys=True, extra_keys=None):
        """
        collect features of 'molecules' and append to atom-level data
        """
        # normalize everything
        keys_to_add = []
        if mol_keys:
            keys_to_add.extend(self.molecule_keys)

        if extra_keys is not None:
            keys_to_add.extend(extra_keys)

        print("Preparing molcule-wise features")
        if self.target in keys_to_add:  # don't add molecule target if we are going to model it
            keys_to_add.remove(self.target)

        molecule_feature_array = np.zeros((self.dataset_length, len(keys_to_add)), dtype=float)
        for column, key in enumerate(keys_to_add):
            feature_vector = dataset[key]
            if type(feature_vector) is not np.ndarray:
                feature_vector = np.asarray(feature_vector)

            if feature_vector.dtype == bool:
                pass
            elif (feature_vector.dtype == float) or (np.issubdtype(feature_vector.dtype, np.floating)):
                feature_vector = standardize(feature_vector)
            elif (feature_vector.dtype == int) or (np.issubdtype(feature_vector.dtype, np.integer)):
                if len(np.unique(feature_vector)) > 2:
                    feature_vector = standardize(feature_vector)
                else:
                    feature_vector = np.asarray(feature_vector == np.amax(feature_vector))  # turn it into a bool

            molecule_feature_array[:, column] = feature_vector

        self.n_mol_features = len(keys_to_add)
        # store known info for training analysis
        self.mol_dict_keys = keys_to_add

        assert np.sum(np.isnan(molecule_feature_array)) == 0
        return molecule_feature_array

    def generate_joint_training_datapoints(self, dataset, config):
        tracking_features = self.gather_tracking_features(dataset)

        ref_coords = dataset['crystal reference cell coords']

        atom_features_list = self.concatenate_atom_features(dataset)

        # add crystal features for generator
        self.crystal_generation_features = []
        point_group_features = [column for column in dataset.columns if 'pg is' in column]
        space_group_features = [column for column in dataset.columns if 'sg is' in column]
        crystal_system_features = [column for column in dataset.columns if 'crystal system is' in column]
        self.crystal_generation_features.extend(point_group_features)
        self.crystal_generation_features.extend(space_group_features)
        self.crystal_generation_features.extend(crystal_system_features)
        self.crystal_generation_features.append('crystal z value') # todo norm this

        molecule_features_array = self.concatenate_molecule_features(
            dataset, extra_keys=self.crystal_generation_features)

        return self.generate_training_data(atom_coords=dataset['atom coords'],
                                           smiles=dataset['molecule smiles'],
                                           atom_features_list=atom_features_list,
                                           mol_features=molecule_features_array,
                                           targets=self.datapoints,
                                           tracking_features=tracking_features,
                                           reference_cells=ref_coords)

    def get_joint_features(self, dataset):
        keys_to_add = self.lattice_keys
        key_dtype = []
        # featurize
        means, stds = [], []
        feature_array = np.zeros((self.dataset_length, len(keys_to_add)), dtype=float)
        # raw_feature_array = np.zeros_like(feature_array)
        for column, key in enumerate(keys_to_add):
            feature_vector = dataset[key]
            if type(feature_vector) is not np.ndarray:
                feature_vector = np.asarray(feature_vector)

            if key == 'crystal z value':
                key_dtype.append('int32')
            else:
                key_dtype.append(feature_vector.dtype)

            # raw_feature_array[:, column] = feature_vector
            mean = np.average(feature_vector)
            std = np.std(feature_vector)
            if (np.isnan(np.std(feature_vector))):
                std = 1
            if (np.std(feature_vector) == 0):
                std = 0.01
            means.append(mean)
            stds.append(std)
            feature_vector = (feature_vector - means[-1]) / stds[-1]
            feature_array[:, column] = feature_vector

            assert np.sum(np.isnan(feature_vector)) == 0

        assert np.sum(np.isnan(stds)) == 0
        assert np.sum(np.asarray(stds) == 0) == 0

        return feature_array, means, stds, key_dtype

    def gather_tracking_features(self, dataset):
        """
        collect features of 'molecules' and append to atom-level data
        """
        # normalize everything
        keys_to_add = []
        keys_to_add.extend(self.molecule_keys)
        keys_to_add.extend(self.crystal_keys)
        if 'crystal spacegroup symbol' in keys_to_add:
            keys_to_add.remove('crystal spacegroup symbol')  # we don't want to deal with strings
        if ('crystal system' in keys_to_add):
            keys_to_add.remove('crystal system')  # we don't want to deal with strings
        if ('crystal lattice centring' in keys_to_add):
            keys_to_add.remove('crystal lattice centring')  # we don't want to deal with strings
        if ('crystal point group' in keys_to_add):
            keys_to_add.remove('crystal point group')  # we don't want to deal with strings

        print("Preparing molecule/crystal tracking features")
        if self.target in keys_to_add:  # don't add molecule target if we are going to model it
            keys_to_add.remove(self.target)

        feature_array = np.zeros((self.dataset_length, len(keys_to_add)), dtype=float)
        for column, key in enumerate(keys_to_add):
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
            'crystal features': self.lattice_keys,
            'n crystal features': len(self.lattice_keys),
            'dataset length': len(self.datapoints),
            'means': self.means,
            'stds': self.stds,
            'dtypes': self.dtypes,
            'n tracking features': self.n_tracking_features,
            'tracking features dict': self.tracking_dict_keys,
            'atom features': self.datapoints[0].x.shape[1],
            'n atomwise features': len(self.atom_keys),
            'n mol features': self.n_mol_features,
            'mol features': self.mol_dict_keys,
            'atom embedding dict sizes': self.atom_dict_size,
            'conditional features': self.molecule_keys + self.crystal_generation_features,
            'n conditional features': len(self.crystal_generation_features + self.molecule_keys),
            'n crystal generation features': len(self.crystal_generation_features),
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

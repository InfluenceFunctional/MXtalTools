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
    def __init__(self, config, pg_dict=None, premade_dataset = None):
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

        # define relevant features for analysis
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
                              # 'molecule H fraction', 'molecule C fraction', 'molecule N fraction', 'molecule O fraction',
                              # 'crystal is organic', 'crystal is organometallic', 'crystal temperature',
                              'molecule principal moment 1', 'molecule principal moment 2', 'molecule principal moment 3',
                              ]
        # for coupling NF models, must be an even number of these
        self.lattice_keys = ['crystal cell a', 'crystal cell b', 'crystal cell c',
                             'crystal alpha', 'crystal beta', 'crystal gamma',
                             'crystal reference cell centroid x', 'crystal reference cell centroid y', 'crystal reference cell centroid z',
                             'crystal reference cell angle 1', 'crystal reference cell angle 2', 'crystal reference cell angle 3',
                             ]
        # for key in self.include_sgs:
        #    self.lattice_keys.append('crystal spacegroup is ' + key)

        if len(self.lattice_keys) % 2 != 0:  # coupling flow model requires an even number of dimensions
            self.lattice_keys.append(self.lattice_keys[0])  # add a redundant dimension

        self.minimal_molecule_keys = ['molecule point group is C1', 'molecule mass', 'molecule num atoms', 'molecule volume']
        self.atom_keys = ['atom Z', 'atom mass', 'atom is H bond donor', 'atom is H bond acceptor',
                          'atom is aromatic', 'atom valence', 'atom vdW radius',
                          'atom on a ring', 'atom chirality', 'atom degree', 'atom electronegativity']

        if pg_dict is None:
            pg_dict = {}
            for i in tqdm.tqdm(range(1, 231)):
                sym_group = symmetry.Group(i)
                pg_dict[i] = sym_group.point_group

        if self.include_sgs is not None:
            print("Modelling within " + str(self.include_sgs))

        '''
        actually load the dataset
        '''

        if premade_dataset is None:
            dataset = pd.read_pickle('datasets/dataset')
        else:
            dataset = premade_dataset
        self.dataset_length = len(dataset)

        # add some missing binary features
        for i in range(config.min_z_value + 1, config.max_z_value + 1):
            dataset['crystal z is {}'.format(i)] = dataset['crystal z value'] == i
        dataset['molecule point group is C1'] = dataset['molecule point group'] == 'C1'
        dataset['crystal veracity'] = np.random.randint(0, 2, size=len(dataset['crystal z value'])).astype(bool)  # DUMMY VARIABLE

        if self.include_sgs is not None:
            for key in self.include_sgs:
                dataset['crystal spacegroup is ' + key] = dataset['crystal spacegroup symbol'] == key
        else:
            self.include_sgs = list(np.unique(dataset['crystal spacegroup symbol']))
            for key in self.include_sgs:
                dataset['crystal spacegroup is ' + key] = dataset['crystal spacegroup symbol'] == key
        self.final_dataset_length = min(self.dataset_length, config.dataset_length)

        # get dictionary for crystal system elements
        crystal_system_elems = np.unique(dataset['crystal system'])
        self.crystal_system_dict = {}
        for i, system in enumerate(crystal_system_elems):
            self.crystal_system_dict[i] = system
            self.crystal_system_dict[system] = i

        dataset['crystal system'] = [self.crystal_system_dict[system] for system in dataset['crystal system']]

        # get crystal point groups and make an ordered dict
        if pg_dict is not None:
            point_groups = [pg_dict[dataset['crystal spacegroup number'][i]] for i in range(len(dataset))]
            point_group_elems = np.unique(point_groups)
            self.point_group_dict = {}
            for i, group in enumerate(point_group_elems):
                self.point_group_dict[i] = group
                self.point_group_dict[group] = i

        dataset['crystal point group'] = [self.point_group_dict[pg] for pg in point_groups]

        # set angle units to natural
        # dataset['crystal reference cell angle 1'] = dataset['crystal reference cell angle 1'] * np.pi / 180
        # dataset['crystal reference cell angle 2'] = dataset['crystal reference cell angle 2'] * np.pi / 180
        # dataset['crystal reference cell angle 3'] = dataset['crystal reference cell angle 3'] * np.pi / 180
        #
        dataset['crystal alpha'] = dataset['crystal alpha'] * np.pi / 180
        dataset['crystal beta'] = dataset['crystal beta'] * np.pi / 180
        dataset['crystal gamma'] = dataset['crystal gamma'] * np.pi / 180

        if self.model_mode == 'joint modelling':
            self.datapoints, self.means, self.stds, self.dtypes = self.get_joint_features(dataset)
            self.datapoints = self.generate_joint_training_datapoints(dataset, config)
        elif self.model_mode == 'cell gan':
            self.datapoints, self.means, self.stds, self.dtypes = self.get_joint_features(dataset)
            self.datapoints = self.generate_joint_training_datapoints(dataset, config)
        else:
            molecule_features_array = self.concatenate_molecule_features(dataset)
            targets = self.get_targets(dataset)
            atom_features_list, hydrogen_inds = self.concatenate_atom_features(dataset)
            for i in range(len(dataset)):
                if len(hydrogen_inds[i]) > 0:
                    coords_i = [dataset['atom coords'][i][j] for j in range(len(dataset['atom coords'][i])) if j not in hydrogen_inds[i]]
                    dataset['atom coords'][i] = coords_i

            tracking_features = self.gather_tracking_features(dataset)
            self.datapoints = self.generate_training_data(dataset['atom coords'], dataset['identifier'], atom_features_list,
                                                          molecule_features_array, targets, tracking_features, dataset['crystal reference cell coords'])

        self.shuffle_datapoints()

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
        # removed 'atom partial charge' due to issues with organometallic dataset
        print("Preparing atom-wise features")
        stds, means = {}, {}
        hydrogen_inds = [np.argwhere(np.asarray(dataset['atom Z'][i]) == 1)[:, 0] for i in range(len(dataset))]  # identify hydrogens
        atom_features_list = [np.zeros((len(dataset['atom Z'][i]) - len(hydrogen_inds[i]), len(keys_to_add))) for i in range(self.dataset_length)]

        for column, key in enumerate(keys_to_add):
            flat_feature = np.concatenate(dataset[key])
            stds[key] = np.std(flat_feature)
            means[key] = np.mean(flat_feature)
            for i in range(self.dataset_length):
                feature_vector = dataset[key][i]
                if len(hydrogen_inds[i]) > 0:
                    feature_vector = [feature_vector[j] for j in range(len(feature_vector)) if j not in hydrogen_inds[i]]

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

                atom_features_list[i][:, column] = feature_vector

        return atom_features_list, hydrogen_inds

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

        return molecule_feature_array

    def generate_joint_training_datapoints(self, dataset, config):
        tracking_features = self.gather_tracking_features(dataset)

        if config.mode == 'cell gan':
            ref_coords = dataset['crystal reference cell coords']
        else:
            ref_coords = None

        atom_features_list, hydrogen_inds = self.concatenate_atom_features(dataset)
        for i in range(len(dataset)):
            if len(hydrogen_inds[i]) > 0:
                coords_i = [dataset['atom coords'][i][j] for j in range(len(dataset['atom coords'][i])) if j not in hydrogen_inds[i]]
                dataset['atom coords'][i] = coords_i

        molecule_features_array = self.concatenate_molecule_features(dataset, extra_keys=['crystal point group','crystal system'])

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
        raw_feature_array = np.zeros_like(feature_array)
        for column, key in enumerate(keys_to_add):
            feature_vector = dataset[key]
            if type(feature_vector) is not np.ndarray:
                feature_vector = np.asarray(feature_vector)

            if key == 'crystal z value':
                key_dtype.append('int32')
            else:
                key_dtype.append(feature_vector.dtype)

            raw_feature_array[:, column] = feature_vector
            means.append(np.average(feature_vector))
            stds.append(np.std(feature_vector))
            feature_vector = (feature_vector - means[-1]) / stds[-1]
            feature_array[:, column] = feature_vector

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

    def get_targets(self, dataset):
        """
        get training target classes
        maybe do some statistics
        etc.
        """
        print("Preparing training targets")
        if 'regression' in self.model_mode:
            print("training target is " + self.target)
            target_features = dataset[self.target]
            self.mean = np.mean(target_features)
            self.std = np.std(target_features)
            target_features = (target_features - self.mean) / self.std

        elif 'classification' in self.model_mode:
            keys_to_add = [self.target]
            self.target_features_keys = keys_to_add
            target_features, targets = self.collect_target_features(dataset, keys_to_add)

            print("training target is " + self.target)
            if self.target_features_dict[self.target]['dtype'] == 'bool':
                self.class_labels = ['False', 'True']
            else:
                self.class_labels = [str(self.target_features_dict[self.target][key]) for key in self.target_features_dict[self.target].keys() if type(key) == int]
            self.output_classes = len(self.class_labels)
            groups, counts = np.unique(targets, return_counts=True)

            self.class_weights = np.array(counts / np.sum(counts)).astype('float16')
            print([out for out in zip(self.class_labels, counts, self.class_weights)])

        else:
            print('Target not implemented for ' + self.model_mode)
            sys.exit()

        return target_features

    def collect_target_features(self, dataset, keys_to_add):
        target_features = np.zeros((self.dataset_length, len(keys_to_add)))

        self.target_features_dict = {}
        self.target_features_dict['target feature keys'] = keys_to_add

        for column, key in enumerate(keys_to_add):
            if type(dataset[key]) is not np.ndarray:
                feature = np.asarray(dataset[key])
            else:
                feature = np.asarray(dataset[key])

            if feature.dtype == bool:
                pass
                self.target_features_dict[key] = {'dtype': 'bool'}

            elif (feature.dtype == int) or (feature.dtype == float):
                print("Regression by classification is deprecated! Switch model mode to regression!")
                sys.exit()

            else:  # if it's a string, we have to finagle it a bit
                if 'minority' in feature:
                    pass
                else:
                    feature = self.collapse_minority_classes(feature, threshold=0.05)
                self.target_features_dict[key] = {'dtype': 'str'}

                # then, convert group titles to numbers, and record the dict
                uniques, counts = np.unique(feature, return_counts=True)
                for i, unique in enumerate(uniques):
                    self.target_features_dict[key][unique] = i
                    self.target_features_dict[key][i] = unique

                self.target_features_dict[key]['classes'] = len(uniques)

                feature = np.asarray([self.target_features_dict[key][feature[i]] for i in range(len(feature))])

            # record final value
            target_features[:, column] = feature

        return target_features, target_features[:, keys_to_add.index(self.target)]

    def collapse_minority_classes(self, feature, threshold=0.05):
        assert 0 <= threshold <= 1
        uniques, counts = np.unique(feature, return_counts=True)
        fractions = counts / sum(counts)
        keep_groups = uniques[np.argwhere(fractions > threshold)[:, 0]]

        return np.asarray([item if item in keep_groups else 'minority' for item in feature])

    def get_dimension(self):

        if self.model_mode == 'joint modelling':
            dim = {
                'crystal features': self.lattice_keys,
                'n crystal features': len(self.lattice_keys),
                'dataset length': len(self.datapoints),
                'means': self.means,
                'stds': self.stds,
                'dtypes': self.dtypes,
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
            }
            if self.conditional_modelling:
                dim['conditional features'] = self.molecule_keys
                dim['n conditional features'] = len(self.molecule_keys)
                if self.conditioning_mode == 'graph model':  #
                    dim['output classes'] = [2]  # placeholder - will be overwritten later
                    dim['atom features'] = self.datapoints[0].x.shape[1]
                    dim['mol features'] = self.n_mol_features
                    dim['atom embedding dict sizes'] = self.atom_dict_size


        elif 'regression' in self.model_mode:
            dim = {
                'atom features': self.datapoints[0].x.shape[1],
                'mol features': self.n_mol_features,
                'output classes': [1],
                'dataset length': len(self.datapoints),
                'atom embedding dict sizes': self.atom_dict_size,
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
                'mean': self.mean,
                'std': self.std,
            }
        elif 'classification' in self.model_mode:
            dim = {
                'atom features': self.datapoints[0].x.shape[1],
                'mol features': self.n_mol_features,
                'output classes': self.output_classes,
                'dataset length': len(self.datapoints),
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
                'class weights': self.class_weights,
                'class labels': self.class_labels,
            }
        elif self.model_mode == 'cell gan':
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
                'mol features': self.n_mol_features,
                'atom embedding dict sizes': self.atom_dict_size,
                'conditional features': ['crystal system', 'crystal point group'],
                'n conditional features': 2
            }
            if self.conditional_modelling:
                dim['conditional features'] += self.molecule_keys
                dim['n conditional features'] += len(self.molecule_keys)

        else:
            print(self.model_mode + ' is not a valid mode!')
            sys.exit()

        dim['crystal system dict'] = self.crystal_system_dict
        dim['point group dict'] = self.point_group_dict
        dim['point groups to search'] = self.include_pgs

        return dim

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def __len__(self):
        return len(self.datapoints)


def get_dataloaders(dataset_builder, config, override_batch_size=None):
    if override_batch_size is not None:
        batch_size = override_batch_size
    else:
        batch_size = config.initial_batch_size
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


def delete_from_dataset(dataset, good_inds):
    print("Deleting unwanted entries")

    for key in dataset.keys():
        if type(dataset[key]) == list:
            dataset[key] = [dataset[key][i] for i in good_inds]
        elif type(dataset[key]) == np.ndarray:
            dataset[key] = dataset[key][np.asarray(good_inds)]

    return dataset

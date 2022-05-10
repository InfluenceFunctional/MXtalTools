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


class BuildDataset():
    '''
    build dataset object
    '''
    def __init__(self, config):
        self.target = config.target
        self.duplicate_augmentation = config.duplicate_augmentation
        self.max_atomic_number = config.max_atomic_number
        self.atom_dict_size = {'atom z': self.max_atomic_number + 1} # for embeddings
        self.dataset_seed = config.dataset_seed
        self.max_temperature = config.max_crystal_temperature
        self.min_temperature = config.min_crystal_temperature
        self.max_num_atoms = config.max_num_atoms
        self.min_num_atoms = config.min_num_atoms
        self.min_z_value = config.min_z_value
        self.max_z_value = config.max_z_value
        self.min_packing_coefficient = config.min_packing_coefficient
        self.balance_target_frequency = config.balance_target_frequency
        self.group_target = config.group_target
        self.only_explicit_groups = config.only_explicit_groups
        self.multi_crystal_tasks = config.multi_crystal_tasks
        self.multi_molecule_tasks = config.multi_molecule_tasks
        self.include_organic = config.include_organic
        self.include_organometallic = config.include_organometallic
        self.concat_mol_to_atom_features = config.concat_mol_to_atom_features
        self.amount_of_features = config.amount_of_features
        self.regression_type_classes = config.regression_type_classes
        self.regression_quantile_cutoff = config.regression_quantile_cutoff
        self.model_type = config.mode
        self.conditional_modelling = config.conditional_modelling
        self.include_sgs = config.include_sgs
        if config.conditioning_mode == 'graph model':
            self.graph_conditioning = True
        else:
            self.graph_conditioning = False



        self.crystal_keys = ['crystal spacegroup symbol',
                             'crystal calculated density', 'crystal packing coefficient',
                             'crystal lattice centring', 'crystal system',
                             'crystal has 2-fold screw', 'crystal has 2-fold rotation', 'crystal has glide', 'crystal has inversion',
                             'crystal cell alpha','crystal cell beta','crystal cell gamma',
                             'crystal cell a','crystal cell b','crystal cell c'
                             ]
        self.molecule_keys = ['molecule point group is C1', 'molecule mass', 'molecule num atoms','molecule volume',
                              'molecule num rings', 'molecule n donors', 'molecule n acceptors',
                              'molecule n rotatable bonds', 'molecule planarity',
                              'molecule spherical defect', 'molecule eccentricity', 'molecule radius of gyration',
                              'molecule H fraction', 'molecule C fraction', 'molecule N fraction', 'molecule O fraction',
                              'molecule is organic', 'molecule is organometallic', 'crystal temperature','molecule polarity',
                              'molecule principal moment 1', 'molecule principal moment 2', 'molecule principal moment 3',
                              ]
        # for coupling NF models, must be an even number of these
        self.lattice_keys = ['crystal cell a', 'crystal cell b', 'crystal cell c',
                             'crystal cell alpha','crystal cell beta','crystal cell gamma',
                             'crystal calculated density','crystal packing coefficient',
                             'crystal z is 2','crystal z is 4','crystal z is 8',
                             'crystal has glide','crystal has 2-fold screw','crystal has inversion','crystal has 2-fold rotation',
                             'crystal spacegroup is P21/c','crystal spacegroup is P212121','crystal spacegroup is P21','crystal spacegroup is P-1']

        self.conditional_keys = ['molecule volume','molecule mass','molecule num atoms','molecule n donors','molecule n acceptors',
                                 'molecule num rings','molecule planarity','molecule spherical defect','molecule radius of gyration',
                                 ]
        self.minimal_molecule_keys = ['molecule point group is C1','molecule mass','molecule num atoms','molecule volume']
        self.atom_keys = ['atom Z','atom mass','atom is H bond donor','atom is H bond acceptor',
                          'atom is aromatic','atom valence','atom vdW radius',
                          'atom on a ring','atom chirality','atom degree','atom electronegativity']

        if self.include_sgs is not None:
            print("Modelling within " + str(self.include_sgs))
            if ('P21/c' in self.include_sgs) or ('P21' in self.include_sgs):
                self.lattice_keys.remove('crystal cell alpha')
                self.lattice_keys.remove('crystal cell gamma')
                self.conditional_keys.remove('crystal z value')
            if ('P-1' in self.include_sgs) or ('P1' in self.include_sgs):
                self.conditional_keys.remove('crystal z value')


        dataset = pd.read_pickle('datasets/dataset.npy')
        self.dataset_length = len(dataset)
        # add some missing binary features
        for i in range(config.min_z_value + 1, config.max_z_value + 1):
            dataset['crystal z is {}'.format(i)] = dataset['crystal z value'] == i
        for group in ['P21/c', 'P21', 'P212121', 'P-1']:
            dataset['crystal spacegroup is ' + group] = dataset['crystal spacegroup symbol'] == group
        if (len(self.lattice_keys) % 2) != 0:
            self.lattice_keys.remove('crystal calculated density')  # sacrificial dimension

        # get fully unfiltered dataset for the total CSD
        if config.mode == 'joint modelling':
            self.full_dataset_data, self.full_dataset_means, self.full_dataset_stds, self.full_dataset_dtypes =\
                self.crystal_feature_statistics(dataset)


        self.final_dataset_length = min(self.dataset_length,config.dataset_length)
        if self.model_type == 'joint modelling':
            self.datapoints, self.means, self.stds, self.dtypes = self.crystal_feature_statistics(dataset)
            if config.conditional_modelling:
                if config.conditioning_mode != 'graph model':
                    self.conditions, self.conditional_means, self.conditional_stds, self.conditional_dtypes =\
                        self.add_conditional_features(dataset,config)
                    self.datapoints = np.concatenate((self.datapoints, self.conditions), axis=1)

                elif config.conditioning_mode == 'graph model':
                    self.datapoints, self.conditional_means, self.conditional_stds, self.conditional_dtypes =\
                        self.add_conditional_features_for_graph(dataset,config)
                    # todo add shuffle function
        else:
            molecule_features_array = self.concatenate_molecule_features(dataset)
            targets = self.get_targets(dataset)

            if config.graph_model == 'classical':
                molecule_features_array, targets = self.shuffle_molecule_dataset(molecule_features_array, targets)
                classifier_scores = self.classical_classification(molecule_features_array, targets)
                print(classifier_scores)
            else:
                atom_features_list = self.concatenate_atom_features(dataset)
                tracking_features = self.gather_tracking_features(dataset)
                atom_features_list, molecule_features_array, targets, smiles, coords, tracking_features =\
                    self.shuffle_final_dataset(atom_features_list, molecule_features_array, targets, dataset['molecule smiles'], dataset['atom coords'], tracking_features)
                self.datapoints = self.generate_training_data(coords, smiles, atom_features_list, molecule_features_array, targets, tracking_features)


    def shuffle_conditional_dataset(self,datapoints, conditions):
        assert len(datapoints) == len(conditions)
        good_inds = np.random.choice(len(conditions),size=len(conditions),replace=False)[0:self.final_dataset_length]
        datapoints = datapoints[good_inds]
        conditions = [conditions[i] for i in good_inds]

        return datapoints, conditions


    def classical_classification(self, features, targets):

        # various classifiers
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        # from sklearn.svm import SVC
        # from sklearn.gaussian_process import GaussianProcessClassifier
        # from sklearn.gaussian_process.kernels import RBF
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        import sklearn.metrics as metrics

        names = [
            "Nearest Neighbors",
            #"Linear SVM",
            #"RBF SVM",
            #"Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]

        classifiers = [
            KNeighborsClassifier(3),
            #SVC(kernel="linear", C=0.025, probability = True),
            #SVC(gamma=2, C=1, probability = True),
            #GaussianProcessClassifier(1.0 * RBF(1.0),copy_X_train=False),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]
        # standardize inputs
        for j in range(features.shape[1]):
            features[:,j] = standardize(features[:,j])

        # do PCA
        use_pca = False

        if use_pca:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=8)
            pca.fit(features)
            X, y = pca.transform(features), targets[:,0]
        else:
            X, y = features, targets[:,0].astype(int)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # fit classifiers
        for name, clf in zip(names, classifiers):
            print('Fitting classifier ' + name)
            t0 = time.time()
            clf.fit(X_train, y_train)
            print("Fitting took {} seconds".format(int(time.time()-t0)))

        f1_score = {}
        roc_score = {}
        balanced_accuracy = {}
        for name, clf in zip(names, classifiers):
            proba = clf.predict_proba(X_test)
            predict = np.argmax(proba,axis=1)
            if len(np.unique(y_test)) > 2:
                f1_score[name] = metrics.f1_score(y_pred=predict, y_true=y_test, average='macro')
                roc_score[name] = metrics.roc_auc_score(y_score=proba, y_true=y_test, multi_class='ovo', average='macro')
                balanced_accuracy[name] = metrics.balanced_accuracy_score(y_true = y_test, y_pred = predict, adjusted=True)
            else:
                f1_score[name] = metrics.f1_score(y_pred=predict, y_true=y_test, average='macro')
                roc_score[name] = metrics.roc_auc_score(y_score=proba[:,1], y_true=y_test, average='macro')
                balanced_accuracy[name] = metrics.balanced_accuracy_score(y_true = y_test, y_pred = predict, adjusted=True)
        print("Finished evaluation")

        plot_results = False
        if plot_results:
            # plot results
            plt.figure(4)
            plt.clf()
            plt.barh([key for key in f1_score.keys()], [(f1_score[key] - 0.5) * 2 for key in f1_score.keys()],height=0.21)
            plt.barh([i - 0.25 for i in range(len(roc_score.keys()))], [2* (roc_score[key] - 0.5) for key in roc_score.keys()],height=0.21)
            plt.barh([i - 0.5 for i in range(len(balanced_accuracy.keys()))], [balanced_accuracy[key] for key in balanced_accuracy.keys()],height=0.21)
            plt.legend(('F1 Macro', 'Adj ROC AOC', 'Adj Bal acc'))
            plt.tight_layout()

        return {'f1 scores': f1_score, 'roc auc score': roc_score, 'balanced accuracy':balanced_accuracy}


    def shuffle_final_dataset(self, atom_features_list, molecule_features_array, targets, smiles, coords, tracking_features):
        np.random.seed(self.dataset_seed)
        good_inds = np.random.choice(self.dataset_length, size = self.final_dataset_length, replace = False)
        self.dataset_length = len(good_inds)
        return [atom_features_list[i] for i in good_inds], molecule_features_array[good_inds], targets[good_inds],\
               [smiles[i] for i in good_inds], [coords[i] for i in good_inds], tracking_features[good_inds]


    def shuffle_molecule_dataset(self, molecule_features_array, targets):
        np.random.seed(self.dataset_seed)
        good_inds = np.random.choice(self.dataset_length, size = self.final_dataset_length, replace = False)
        self.dataset_length = len(good_inds)
        return  molecule_features_array[good_inds], targets[good_inds]


    def generate_training_data(self, atom_coords, smiles, atom_features_list, mol_features, targets, tracking_features):
        # preprocess data
        datapoints = []

        print("Generating final training datapoints")
        for i in tqdm.tqdm(range(self.final_dataset_length)):
            if targets[i].ndim == 1:
                target = torch.tensor(targets[i][np.newaxis,:])
            else:
                target = torch.tensor(targets[i])

            # append molecule features to atom features for each atom
            input_features = np.concatenate((atom_features_list[i],np.repeat(mol_features[i][np.newaxis,:],len(atom_features_list[i]),axis=0)),axis=1)
            datapoints.append(Data(x=torch.Tensor(input_features).float(), pos=torch.Tensor(atom_coords[i]), y=[target, smiles[i], tracking_features[i]]))

        return datapoints


    def concatenate_atom_features(self, dataset):
        '''
        collect and normalize/standardize relevant atomic features
        :param dataset:
        :return:
        '''

        # # playing with pyxtal
        # from pyxtal import pyxtal, molecular_crystal, lattice, symmetry
        # from pymatgen.core import Molecule
        # from ase.build import make_supercell
        #
        # supercells = []
        # for i in tqdm.tqdm(range(5)):
        #     crystal = pyxtal(molecular=True)
        #     crystal.from_random(dim=3, numIons=[dataset['crystal z value'][i]], seed=self.dataset_seed,
        #                         species=[Molecule(species = dataset['atom Z'][i], coords = dataset['atom coords'][i])],
        #                         group=symmetry.Group(dataset['crystal spacegroup number and setting'][i][0]),
        #                         lattice=lattice.Lattice(ltype=dataset['crystal system'][i],
        #                                                 volume=dataset['molecule volume'][i] * dataset['crystal z value'][i] / 0.65,
        #                                                 matrix = dataset['crystal cell vectors'][i]
        #                                                 )
        #                         )
        #     supercells.append(make_supercell(crystal.to_ase(), np.eye(3) * 2))

        # crystal generation steps
        # generate unit cell
        # index [atom][molecule][prime cell]
        # generate XxX supercell
        # featurize appropriately
        # finished!

        keys_to_add = self.atom_keys
        # removed 'atom partial charge' due to issues with organometallic dataset
        print("Preparing atom-wise features")
        atom_features_list = [np.zeros((len(dataset['atom Z'][i]), len(keys_to_add))) for i in range(self.dataset_length)]

        for i in tqdm.tqdm(range(self.dataset_length)):
            for column, key in enumerate(keys_to_add):
                feature_vector = dataset[key][i]

                if type(feature_vector) is not np.ndarray:
                    feature_vector = np.asarray(feature_vector)

                if key == 'atom Z':
                    pass
                elif feature_vector.dtype == bool:
                    pass
                elif feature_vector.dtype == float:
                    feature_vector = standardize(feature_vector)
                elif feature_vector.dtype == int:
                    if len(np.unique(feature_vector)) > 2:
                        feature_vector = standardize(feature_vector)
                    else:
                        feature_vector = np.asarray(feature_vector == np.amax(feature_vector))  # turn it into a bool

                atom_features_list[i][:,column] = feature_vector

        return atom_features_list


    def concatenate_molecule_features(self, dataset,extra_keys = None):
        '''
        collect features of 'molecules' and append to atom-level data
        '''
        # normalize everything
        if self.amount_of_features.lower() == 'maximum':
            keys_to_add = self.molecule_keys
        elif self.amount_of_features.lower() == 'minimum':
            keys_to_add = self.minimal_molecule_keys
        else:
            print(self.amount_of_features + ' is not a valid featurization!')
            sys.exit()

        if extra_keys is not None:
            keys_to_add.extend(extra_keys)

        print("Preparing molcule-wise features")
        if self.target in keys_to_add: # don't add molecule target if we are going to model it
            keys_to_add.remove(self.target)

        molecule_feature_array = np.zeros((self.dataset_length,len(keys_to_add)),dtype=float)
        for column,key in enumerate(keys_to_add):
            feature_vector = dataset[key]
            if type(feature_vector) is not np.ndarray:
                feature_vector = np.asarray(feature_vector)

            if feature_vector.dtype == bool:
                pass
            elif feature_vector.dtype == float:
                feature_vector = standardize(feature_vector)
            elif feature_vector.dtype == int:
                if len(np.unique(feature_vector)) > 2:
                    feature_vector = standardize(feature_vector)
                else:
                    feature_vector = np.asarray(feature_vector == np.amax(feature_vector)) # turn it into a bool

            molecule_feature_array[:, column] = feature_vector

        self.n_mol_features = len(keys_to_add)
        # store known info for training analysis
        self.mol_dict_keys = keys_to_add

        return molecule_feature_array


    def add_conditional_features(self,dataset,config):
        if config.conditioning_mode == 'molecule features':
            keys_to_add = self.conditional_keys

            key_dtype = []
            # featurize
            means, stds = [], []
            feature_array = np.zeros((self.dataset_length, len(keys_to_add)), dtype=float)
            raw_feature_array = np.zeros_like(feature_array)
            for column, key in enumerate(keys_to_add):
                feature_vector = dataset[key]
                if type(feature_vector) is not np.ndarray:
                    feature_vector = np.asarray(feature_vector)

                key_dtype.append(feature_vector.dtype)

                raw_feature_array[:, column] = feature_vector
                means.append(np.average(feature_vector))
                stds.append(np.std(feature_vector))
                feature_vector = (feature_vector - means[-1]) / stds[-1]
                feature_array[:, column] = feature_vector
        else:
            print(config.conditioning_mode + ' is not a valid conditioning mode')
            sys.exit()

        return feature_array, means, stds, key_dtype


    def add_conditional_features_for_graph(self,dataset,config):
        if config.conditioning_mode == 'graph model':
            molecule_features_array = self.concatenate_molecule_features(dataset, extra_keys = self.conditional_keys)
            atom_features_list = self.concatenate_atom_features(dataset)
            tracking_features = self.gather_tracking_features(dataset)

            self.conditional_keys = np.arange(config.fc_depth) # the size of the conditioning dim is just the output from the graph model
            return self.generate_training_data(dataset['atom coords'], dataset['molecule smiles'], atom_features_list, molecule_features_array, self.datapoints, tracking_features),\
                   None, None, None

            #todo save standardization for future evaluation mode



    def crystal_feature_statistics(self,dataset):
        keys_to_add = self.lattice_keys
        key_dtype = []
        # featurize
        means, stds = [],[]
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


        # '''
        # toy dataset for debugging purposes
        # '''
        # from sklearn import datasets
        # feature_array = datasets.make_moons(n_samples=100000)[0]
        # feature_array += np.random.standard_normal(size=feature_array.shape) / 10
        # means = np.array((np.average(feature_array[:,0]),np.average(feature_array[:,1])))
        # stds = np.array((np.std(feature_array[:,0]),np.std(feature_array[:,1])))
        # feature_array[:,0] = (feature_array[:,0] - means[0]) / stds[0]
        # feature_array[:,1] = (feature_array[:,1] - means[1]) / stds[1]
        # key_dtype = [feature_array[:,0].dtype,feature_array[:,1].dtype]
        # self.lattice_keys = ['x','y']

        return feature_array, means, stds, key_dtype


    def gather_tracking_features(self, dataset):
        '''
        collect features of 'molecules' and append to atom-level data
        '''
        # normalize everything
        keys_to_add = []
        keys_to_add.extend(self.molecule_keys)
        keys_to_add.extend(self.crystal_keys)
        if 'crystal spacegroup symbol' in keys_to_add:
            keys_to_add.remove('crystal spacegroup symbol') # we don't want to deal with strings
        if ('crystal system' in keys_to_add):
            keys_to_add.remove('crystal system') # we don't want to deal with strings
        if ('crystal lattice centring' in keys_to_add):
            keys_to_add.remove('crystal lattice centring') # we don't want to deal with strings


        print("Preparing molecule/crystal tracking features")
        if self.target in keys_to_add: # don't add molecule target if we are going to model it
            keys_to_add.remove(self.target)

        feature_array = np.zeros((self.dataset_length,len(keys_to_add)),dtype=float)
        for column,key in enumerate(keys_to_add):
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
                    feature_vector = np.asarray(feature_vector == np.amax(feature_vector)) # turn it into a bool

            feature_array[:, column] = feature_vector

        ''' # plot statistics
        plt.clf()
        for i in range(12):
            plt.subplot(4,3,i+1)
            plt.title(keys_to_add[i])
            plt.hist(feature_array[:,i],density=True,bins=50)
            
        '''
        self.n_tracking_features = len(keys_to_add)
        # store known info for training analysis
        self.tracking_dict_keys = keys_to_add

        return feature_array


    def get_targets(self, dataset):
        '''
        get training target classes
        maybe do some statistics
        etc.
        '''
        print("Preparing training targets")
        if self.model_type == 'regression':
            print("training target is " + self.target)
            target_features = dataset[self.target]
            self.mean = np.mean(target_features)
            self.std = np.std(target_features)
            target_features = (target_features - self.mean) / self.std

        else:
            keys_to_add = []
            if self.multi_crystal_tasks:
                keys_to_add.extend(self.crystal_keys)
            if self.multi_molecule_tasks and (self.amount_of_features == 'minimum'): # add molecule features
                keys_to_add.extend(self.molecule_keys)
                for key in self.minimal_molecule_keys:
                    keys_to_add.remove(key) # don't want to train on the ones we are going to be feeding in the first place

            if self.target in keys_to_add:
                keys_to_add.remove(self.target)
            keys_to_add.insert(0, self.target)  # always put the main target in first position in the list

            self.target_features_keys = keys_to_add
            target_features, targets = self.collect_target_features(dataset, keys_to_add)

            print("training target is " + self.target)
            if self.target_features_dict[self.target]['dtype'] == 'bool':
                self.group_labels = ['False', 'True']
            else:
                self.group_labels = [self.target_features_dict[self.target][key] for key in self.target_features_dict[self.target].keys() if type(key) == int]
            self.output_classes = len(self.group_labels)
            groups, counts = np.unique(targets, return_counts=True)
            print([out for out in zip(groups, counts)])

            groups, counts = np.unique(targets,return_counts=True)
            print([out for out in zip(groups.astype(str), counts)])

            self.class_weights = self.reweighting(target_features)

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
                nbins = self.regression_type_classes
                ends_iles = self.regression_quantile_cutoff
                rrange = [np.quantile(feature, ends_iles), np.quantile(feature, 1-ends_iles)]
                feature = np.digitize(feature, np.linspace(rrange[0], rrange[1], nbins))
                self.target_features_dict[key] = {'dtype': 'str'}
                self.target_features_dict[key]['range'] = list(np.linspace(rrange[0],rrange[1],nbins).astype('float32'))
                print(key + ' discretization range is ' + str(self.target_features_dict[key]['range']))
                uniques, counts = np.unique(feature, return_counts=True)
                for i, unique in enumerate(uniques):
                    self.target_features_dict[key][str(i)] = unique
                    self.target_features_dict[key][int(unique)] = i
                self.target_features_dict[key]['classes'] = len(uniques)
                feature = np.asarray([self.target_features_dict[key][feature[i]] for i in range(len(feature))])

                #feature = standardize(feature)
                #self.target_features_dict[key] = {'dtype': 'int'}

            # elif feature.dtype == float: # quantize floats into classification problem
            #     rrange = [np.quantile(feature, 0.1), np.quantile(feature, 0.9)]
            #     feature = np.digitize(feature, np.linspace(rrange[0], rrange[1], 10))
            #     self.target_features_dict[key] = {'dtype': 'str'}
            #     uniques, counts = np.unique(feature, return_counts=True)
            #     for i, unique in enumerate(uniques):
            #         self.target_features_dict[key][str(i)] = unique
            #         self.target_features_dict[key][int(unique)] = i
            #     self.target_features_dict[key]['classes'] = len(uniques)
            #     feature = np.asarray([self.target_features_dict[key][feature[i]] for i in range(len(feature))])
            #
            #     #feature = standardize(feature)
            #     #self.target_features_dict[key] = {'dtype': 'float'}

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

        return target_features, target_features[:,keys_to_add.index(self.target)]


    def collapse_minority_classes(self, feature, threshold = 0.05):
        assert 0 <= threshold <= 1
        uniques, counts = np.unique(feature, return_counts=True)
        fractions = counts / sum(counts)
        keep_groups = uniques[np.argwhere(fractions > threshold)[:, 0]]
        return np.asarray([item if item in keep_groups else 'minority' for item in feature])


    def reweighting(self, targets):
        if targets.ndim > 1:
            targets = targets[:,0]
        classes, counts = np.unique(targets, return_counts=True)  # weight according to the first target
        if len(classes) > 100:
            print("too many classes! are you sure you didn't try to model a float?")
            sys.exit()
        self.normed_counts = counts/np.sum(counts)
        print('Class balance:')
        print(self.normed_counts * 100)
        print('Total counts = {}'.format(len(targets)))
        prop_weights = 1/self.normed_counts # weight is inverse to frequency
        prop_weights = np.clip(prop_weights,0,1e4)
        prop_weights = prop_weights / np.average(prop_weights) # normalize so learning-rate isn't way out of whack
        prop_weights = torch.Tensor(prop_weights)

        return prop_weights

    def get_full_dimension(self):
        dim = {
            'crystal features': self.lattice_keys,
            'n crystal features': len(self.lattice_keys),
            'dataset length': len(self.full_dataset_data),
            'means': self.full_dataset_means,
            'stds': self.full_dataset_stds,
            'dtypes' : self.full_dataset_dtypes,
        }
        return dim

    def get_dimension(self):
        if self.model_type == 'joint modelling':
            dim = {
                'crystal features' : self.lattice_keys,
                'n crystal features' : len(self.lattice_keys),
                'dataset length' : len(self.datapoints),
                'means' : self.means,
                'stds' : self.stds,
                'dtypes' : self.dtypes,
            }
            # todo fix this up for graph modelling
            if self.conditional_modelling:
                dim['conditional features'] =  self.conditional_keys
                dim['n conditional features'] =  len(self.conditional_keys)
                dim['conditional means'] =  self.conditional_means
                dim['conditional_stds'] =  self.conditional_stds
                dim['conditional dtypes'] =  self.conditional_dtypes
                if self.graph_conditioning: # todo - why is this all fucked up?
                    dim['output classes'] = [2] # placeholder
                    dim['atom features'] = self.datapoints[0].x.shape[1]
                    dim['mol features'] = self.n_mol_features
                    dim['atom embedding dict sizes'] = self.atom_dict_size
                    dim['prediction tasks'] = 1
                    dim['n tracking features'] = self.n_tracking_features
                    dim['tracking features dict'] = self.tracking_dict_keys
        elif self.model_type == 'regression':
            dim = {
                'atom features': self.datapoints[0].x.shape[1],
                'mol features': self.n_mol_features,
                'output classes': [1],
                'dataset length' : len(self.datapoints),
                'atom embedding dict sizes' : self.atom_dict_size,
                'prediction tasks': 1,
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
                'mean' : self.mean,
                'std' : self.std,
            }
        else:
            dim = {
                'atom features': self.datapoints[0].x.shape[1],
                'mol features': self.n_mol_features,
                'output classes': [self.output_classes],
                'dataset length' : len(self.datapoints),
                'atom embedding dict sizes' : self.atom_dict_size,
                'prediction tasks': 1,
                'n tracking features': self.n_tracking_features,
                'tracking features dict': self.tracking_dict_keys,
            }

            if self.multi_molecule_tasks or self.multi_crystal_tasks:
                dim['target features dict'] = self.target_features_dict
                dim['prediction tasks'] = len(self.target_features_keys)
                all_task_targets = []
                for key in self.target_features_dict['target feature keys']:
                    if self.target_features_dict[key]['dtype'] == 'bool':
                        all_task_targets.append(2)
                    elif self.target_features_dict[key]['dtype'] == 'int':
                        all_task_targets.append(1)
                    if self.target_features_dict[key]['dtype'] == 'float':
                        all_task_targets.append(1)
                    if self.target_features_dict[key]['dtype'] == 'str':
                        all_task_targets.append(self.target_features_dict[key]['classes'])
                dim['output classes'] = all_task_targets

        return dim


    def prep_for_embedding(self, dataset, key):
        unique_entries = np.arange(1,self.max_atomic_number + 1)#list(set(dataset[key]))
        dict = {}
        for i, symbol in enumerate(unique_entries):
            dict[symbol] = i

        standin = np.zeros((len(dataset[key])),dtype='uint32')
        for i in range(len(dataset[key])):
            standin[i] = dict[dataset[key][i]]

        if 'atom' in key:
            self.atom_dict_size[key] = len(dict)
        elif 'mol' in key:
            self.mol_dict_size[key] = len(dict)

        return standin

    def get_class_weights(self):
        return {'class weights': list(self.class_weights),
                'class probs': list(self.normed_counts),
                }

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def __len__(self):
        return len(self.datapoints)

    def get_full_dataset(self):
        return self.datapoints


def get_dataloaders(dataset_builder, config, override_batch_size = None):
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
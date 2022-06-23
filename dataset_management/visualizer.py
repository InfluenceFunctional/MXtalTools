import numpy as np
from new_CSD_datamining import miner
import argparse
import os
from utils import unstack_dataset
from ase import Atoms
import ase.visualize
import pandas as pd

if True:
    # get command line input
    parser = argparse.ArgumentParser()

    def add_bool_arg(parser, name, default=False):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action = 'store_true')
        group.add_argument('--no-' + name, dest=name, action = 'store_false')
        parser.set_defaults(**{name:default})


    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--machine', type=str, default='local')  # 'local' (local windows venv) or 'cluster' (linux env)
    add_bool_arg(parser, 'fingerprint_mode', default=False)
    parser.add_argument('--minimum_symmetry_element_frequency', type=float, default=0.01)
    parser.add_argument('--dataset_length', type=int, default=int(1e7))  # maximum number of items in the dataset before filtration

    if True:
        parser.add_argument('--max_groups', type=int, default=100000) # if == 2, this becomes a binary classifier, else number of groups to model is determined by min & max group examples below
        parser.add_argument('--group_target',type = str, default = 'P21/c') # if max_groups == 1, this is the group we model
        parser.add_argument('--max_group_examples', type=int, default=2000) # maximum number of examples for a particular group
        parser.add_argument('--min_group_examples', type=int, default=2000) # minimum number of examples for a particular group - obviated for binary classification
        parser.add_argument('--min_z_prime', type=int, default=0)  # maximum value of z prime to take into training set
        parser.add_argument('--max_z_prime', type=int, default=1)  # maximum value of z prime to take into training set
        parser.add_argument('--min_z_value', type=int, default=0)  # maximum of z to take into training set
        parser.add_argument('--max_z_value', type=int, default=4)  # maximum value of z to take into training set
        parser.add_argument('--max_molecule_size', type=int, default=100)  # maximum number of atoms for molecules in training set
        parser.add_argument('--max_molecule_volume', type=int, default=10000)  # maximum molecule volume
        parser.add_argument('--max_atomic_number', type=int, default=20)  # maximum atomic number for atoms in training set
        parser.add_argument('--max_rings', type=int, default=50)  # maximum number of rings for molecules in training set
        parser.add_argument('--max_components', type=int, default=1)  # maximum number of separate molecules per structure / entry
        parser.add_argument('--max_waters', type=int, default=0)  # maximum number of separate molecules per structure / entry
        add_bool_arg(parser, 'allow_organic', default=True)
        add_bool_arg(parser, 'allow_polymeric', default=False)
        add_bool_arg(parser, 'allow_organometallic', default=False)

    if False:
        parser.add_argument('--max_groups', type=int, default=100000)  # if == 2, this becomes a binary classifier, else number of groups to model is determined by min & max group examples below
        parser.add_argument('--group_target', type=str, default='P21/c')  # if max_groups == 1, this is the group we model
        parser.add_argument('--max_group_examples', type=int, default=100000)  # maximum number of examples for a particular group
        parser.add_argument('--min_group_examples', type=int, default=1)  # minimum number of examples for a particular group - obviated for binary classification
        parser.add_argument('--min_z_prime', type=int, default=0)  # maximum value of z prime to take into training set
        parser.add_argument('--max_z_prime', type=int, default=100)  # maximum value of z prime to take into training set
        parser.add_argument('--min_z_value', type=int, default=0)  # maximum of z to take into training set
        parser.add_argument('--max_z_value', type=int, default=100)  # maximum value of z to take into training set
        parser.add_argument('--max_molecule_size', type=int, default=1000)  # maximum number of atoms for molecules in training set
        parser.add_argument('--max_molecule_volume', type=int, default=10000)  # maximum molecule volume
        parser.add_argument('--max_atomic_number', type=int, default=100)  # maximum atomic number for atoms in training set
        parser.add_argument('--max_rings', type=int, default=100)  # maximum number of rings for molecules in training set
        parser.add_argument('--max_components', type=int, default=100)  # maximum number of separate molecules per structure / entry
        parser.add_argument('--max_waters', type=int, default=100)  # maximum number of separate molecules per structure / entry
        add_bool_arg(parser, 'allow_organic', default=True)
        add_bool_arg(parser, 'allow_polymeric', default=True)
        add_bool_arg(parser, 'allow_organometallic', default=True)


config = parser.parse_args()
if config.machine == 'local':
    config.workdir ='C:/Users\mikem\Desktop/CSP_runs'  # Working directory
elif config.machine == 'cluster':
    config.workdir = '/scratch/mk8347/csd_runs/'

config.seeds.dataset = config.seeds.dataset % 10

os.chdir(config.workdir)  # move to working dir

# load up the dataset
if True:  # for fast prototyping
    data = np.load('datasets/CSD_pull_sym_quick.npy', allow_pickle=True).item()
else: # full dataset
    data = np.load('datasets/CSD_pull_sym_spatial.npy', allow_pickle=True).item()  # dataset already exists
print('Loaded premade dataset')

# process it as if we were training
dataminer = miner(config,machine='local', override_data = data)
data, groups = dataminer.get_processed_dataset()
data = unstack_dataset(data)
# options for looking at particular samples

del(data['atom breakpoints'])
df = pd.DataFrame.from_dict(data)

def print_random_molecule(df, space_group):
    gdf = df.groupby('crystal space group symbol')
    group_indices = gdf.groups[space_group]
    i = np.random.randint(0,len(group_indices))
    z = df['atom symbols'][i]
    coords = df['atom coords'][i]
    amol = Atoms(z,positions=coords)
    ase.io.write('amol_{}.cif'.format(i),amol)
    print('Spherical defect = {:.3f} || Planar defect = {:.3f}'.format(df['mol spherical defect'][i],df['mol planar defect'][i]))
    print('Volume = {} || Num Atoms = {}'.format(df['mol volume'][i], df['mol num atoms'][i]))


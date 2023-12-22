import os
import pandas as pd
import tqdm
import warnings
from random import shuffle
from common.utils import chunkify
import numpy as np
from constants.atom_properties import NUMBERS

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

n_chunks = 10000  # too many chunks can cause problems e.g., if some have zero valid entries
chunks_path = r'D:/crystal_datasets/QCQM4_chunks/'  # where you would like processed dataset chunks to be stored before collation into final dataset

chunk_prefix = ''
sdfs_path = r'D:\crystal_datasets\Molecule_Datasets\pcqm4m-v2/'

os.chdir(sdfs_path)
sdfs_list = os.listdir()

if not os.path.exists(chunks_path):
    os.mkdir(chunks_path)

print(f"Breaking dataset into {n_chunks} chunks")

file_length = 9746078962
chunk_size = file_length // n_chunks
chunks_list = [chunk_size * n for n in range(n_chunks)]

chunk_inds = [n for n in range(n_chunks)]
shuffle(chunk_inds)


# todo something broken here when we get to large line inds
# break a single huge sdf file into chunks
for chunk_ind in chunk_inds:
    increment_df = None
    max_ind = chunks_list[chunk_ind] + chunk_size
    min_ind = chunks_list[chunk_ind]
    if not os.path.exists(chunks_path + chunk_prefix + f"_chunk_{chunk_ind}.pkl"):
        print(f"Starting chunk {chunk_ind}")
        with open(sdfs_list[0], "r") as f:
            in_atom_block = False
            for line_ind, line in tqdm.tqdm(enumerate(f)):
                if min_ind <= line_ind < max_ind:
                    if in_atom_block:
                        if atom_ind < num_atoms:
                            splitline = line.split()
                            atoms_array[atom_ind] = [NUMBERS[splitline[3]], float(splitline[0]), float(splitline[1]), float(splitline[2])]
                            atom_ind += 1
                        else:
                            molecule_dict['atom_coordinates'] = atoms_array[:, 1:]
                            molecule_dict['atom_atomic_numbers'] = atoms_array[:, 0]
                            molecule_dict['identifier'] = identifier
                            molecule_dict['molecule_num_atoms'] = num_atoms
                            molecule_dict['molecule_volume'] = np.random.uniform(0)  # explicit dummy value
                            molecule_dict['molecule_mass'] = np.random.uniform(0)  # explicit dummy value

                            new_df = pd.DataFrame()
                            for key in molecule_dict.keys():
                                new_df[key] = [molecule_dict[key]]

                            if increment_df is None:
                                increment_df = new_df
                            else:
                                increment_df = pd.concat([increment_df, new_df])

                            in_atom_block = False
                    elif '.xyz' == line[-5:-1]:
                        identifier = '_'.join(line.split('/')[-2:])[:-4]
                    elif '$$$$' in line:
                        new_mol = True
                    elif 'V2000' == line[-6:-1]:
                        num_atoms = int(line.split()[0])
                        in_atom_block = True
                        atoms_array = np.zeros((num_atoms, 4))
                        atom_ind = 0
                        molecule_dict = {}
                elif line_ind >= max_ind:
                    break

        if increment_df is not None:
            increment_df.to_pickle(chunks_path + f"{chunk_prefix}_chunk_{chunk_ind}.pkl")

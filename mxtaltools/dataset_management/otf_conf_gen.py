import gzip
import multiprocessing as mp
import os
from pathlib import Path
from random import shuffle

import numpy as np

from mxtaltools.common.utils import chunkify
from mxtaltools.dataset_management.data_manager import DataManager
from mxtaltools.dataset_management.dataset_generation.generate_dataset_from_smiles import process_smiles_list


def async_generate_random_conformer_dataset(dataset_length,
                                            smiles_source,
                                            dataset_name,
                                            workdir,
                                            allowed_atom_types: list,
                                            num_processes: int,
                                            pool,
                                            max_num_atoms: int,
                                            max_num_heavy_atoms: int,
                                            pare_to_size: int,
                                            max_radius: float,
                                            synchronize=True):
    dataset_path = Path(dataset_name)
    chunks_path = Path(workdir)
    smiles_path = Path(smiles_source)
    # get batch of smiles, and chunkify
    os.chdir(smiles_path)
    chunks = get_smiles_list(dataset_length, num_processes, smiles_path)
    os.chdir(chunks_path)

    # generate samples
    min_ind = len(os.listdir(chunks_path)) + 1  # always add one

    for ind, chunk in enumerate(chunks):
        # print(f'starting chunk {ind} with {len(chunk)} smiles')
        chunk_ind = min_ind + ind
        chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
        pool.apply_async(process_smiles_list,
                         args=(chunk, chunk_path, allowed_atom_types),
                         kwds={
                             'max_num_atoms': max_num_atoms,
                             'max_num_heavy_atoms': max_num_heavy_atoms,
                             'pare_to_size': pare_to_size,
                             'max_radius': max_radius,
                             'protonate': True,
                             'rotamers_per_sample': 1,
                             'allow_simple_hydrogen_rotations': False
                         })

    pool.close()
    # print(f"finished processing smiles chunks")
    if synchronize:
        pool.join()
        #
        # # generate temporary training dataset
        # miner = DataManager(device='cpu',
        #                     datasets_path=dataset_path,
        #                     chunks_path=chunks_path,
        #                     dataset_type='molecule',)
        # miner.process_new_dataset(new_dataset_name='temp_zinc_conf_dataset',
        #                           )
        #os.rmdir('chunks')
    else:
        return pool


def get_smiles_list(dataset_length, num_processes, smiles_dirs_path):
    h_dirs = os.listdir(smiles_dirs_path)
    h_dirs = [elem for elem in h_dirs if elem[0] == 'H']
    smiles_paths = []
    for dir in h_dirs:
        files = os.listdir(dir)
        smiles_paths.extend(
            [os.path.join(Path(dir), Path(file)) for file in files]
        )
    file_sizes = np.zeros(len(smiles_paths))
    for i, path in enumerate(smiles_paths):
        file_sizes[i] = os.path.getsize(path)
    paths_to_keep = []
    for i in range(len(file_sizes)):
        if file_sizes[i] > 0:
            paths_to_keep.append(i)
    smiles_paths = [smiles_paths[ind] for ind in paths_to_keep]
    file_sizes = file_sizes[paths_to_keep]
    # sample proportional to their size
    # select random set of smiles files
    smiles_list = []
    while len(smiles_list) < dataset_length:
        file_to_add = np.random.choice(
            range(len(smiles_paths)),
            size=1,
            replace=False,
            p=file_sizes / np.sum(file_sizes)
        )[0]
        filename = smiles_paths[file_to_add]
        if filename[-3:] == '.gz':
            with gzip.open(filename, 'r') as f:
                for line in f:
                    smiles_list.append(line.rstrip())
        elif filename[-4:] == '.txt':
            with open(filename, 'r') as f:
                for line in f:
                    smiles_list.append(line.rstrip())

    shuffle(smiles_list)
    chunks = chunkify(smiles_list[:dataset_length], num_processes)
    return chunks


if __name__ == '__main__':
    async_generate_random_conformer_dataset(dataset_length=1000,
                                            smiles_source=r'D:\crystal_datasets\zinc22',
                                            dataset_name=r'D:\crystal_datasets\zinc22')

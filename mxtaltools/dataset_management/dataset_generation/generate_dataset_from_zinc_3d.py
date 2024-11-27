import numpy as np
from tqdm import tqdm
import torch
import pickle
import multiprocessing as mp
import os
import rdkit.Chem as Chem
from pathlib import Path
import tarfile

from mxtaltools.dataset_management.CrystalData import CrystalData


def process_mol2_list(combo_path, chunk_ind, file_ind, chunks_dir):
    samples = []
    with tarfile.open(combo_path, 'r:gz') as f:
        entries = f.getmembers()
        for entry in entries:
            molfile = f.extractfile(entry).read()
            sample = process_mol2_block(molfile)
            if sample is not None:
                samples.append(sample)

    chunks_dir_path = Path(chunks_dir)
    with open(os.path.join(chunks_dir_path, f'{chunk_ind}_{file_ind}.pkl'), 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del samples


def process_mol2_block(molfile):
    try:
        mol = Chem.MolFromMol2Block(molfile)
        conf = mol.GetConformer()
    except:
        return None

    coords = np.array(conf.GetPositions())
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # molecule sizes filter
    if len(atom_types) < 6 or len(atom_types) > 100:
        return None

    # atom types filter
    if not set(atom_types).issubset([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]):
        return None

    sample = CrystalData(
        x=torch.tensor(atom_types, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float32),
        smiles=Chem.MolToSmiles(mol),
        identifier=mol.GetProp("_Name"),
        y=torch.zeros(1, dtype=torch.float32),
        require_crystal_features=False,
    )

    # molecule radius filter
    if sample.radius > 15:
        return None

    return sample.to_dict()


if __name__ == '__main__':
    parent_directory = r'D:\crystal_datasets\zinc22_3d'
    chunks_dir = os.path.join(Path(parent_directory), 'chunks')
    #parent_directory = r'/vast/mk8347/zinc22_3d'

    lmdb_database = 'zinc.lmdb'
    map_size = int(1e9)  # map size in bytes

    os.chdir(parent_directory)
    dirs = os.listdir()

    # if not os.path.exists(lmdb_database.split('.lmdb')[0] + '_keys.npy'):
    #     overall_index = int(0)
    # else:
    #     overall_index = np.load(lmdb_database.split('.lmdb')[0] + '_keys.npy', allow_pickle=True).item()

    keys_path = parent_directory + '/' + lmdb_database.split('.lmdb')[0] + '_keys'
    database_path = parent_directory + '/' + lmdb_database
    chunk_ind = - 1
    min_chunk = 0
    max_chunk = min(100000, len(dirs))
    tot_index = 0

    pool = mp.Pool(mp.cpu_count() - 1)

    with tqdm(total=max_chunk) as pbar:
        while chunk_ind < max_chunk - 1:
            pbar.update(1)
            chunk_ind += 1

            if not (max_chunk > chunk_ind >= min_chunk):
                continue

            if dirs[chunk_ind][0] == 'H':
                dirpath = Path(dirs[chunk_ind])
                for file_ind, file in enumerate(tqdm(os.listdir(dirpath))):
                    chunkpath = os.path.join(chunks_dir, fr'chunk_{chunk_ind}_{file_ind}.pkl')
                    if not os.path.exists(chunkpath):
                        filepath = Path(file)
                        combo_path = os.path.join(dirpath, filepath)
                        pool.apply_async(process_mol2_list, args=(combo_path, chunk_ind, file_ind, chunks_dir))

    pool.close()
    pool.join()

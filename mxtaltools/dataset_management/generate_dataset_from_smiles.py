import numpy as np
from tqdm import tqdm
import torch
import lmdb
import pickle

import os
import rdkit.Chem as Chem
from rdkit.Chem import AllChem

from mxtaltools.dataset_management.CrystalData import CrystalData


def write_lmdb(database, current_map_size, dict_to_write):
    try:
        with lmdb.open(database, map_size=current_map_size) as db:
            with db.begin(write=True) as txn:
                for key, value in dict_to_write.items():
                    txn.put(key.encode('ascii'), pickle.dumps(value))

        return current_map_size

    except lmdb.MapFullError:
        assert current_map_size < 2e11
        new_map_size = int(current_map_size * 1.25)
        print(f'Boosting map size from {current_map_size / 1e9:.1f}GB to {new_map_size / 1e9:.1f}GB')
        current_map_size = new_map_size
        return write_lmdb(database, current_map_size, dict_to_write)


if __name__ == '__main__':
    #parent_directory = r'D:\crystal_datasets\zinc22'
    parent_directory = r'/vast/mk8347/zinc'

    lmdb_database = 'zinc.lmdb'
    map_size = int(1e9)  # map size in bytes

    min_chunk = 0
    max_chunk = 5

    os.chdir(parent_directory)
    dirs = os.listdir()

    if not os.path.exists(lmdb_database.split('.lmdb')[0] + '_keys.npy'):
        overall_index = int(0)
    else:
        overall_index = np.load(lmdb_database.split('.lmdb')[0] + '_keys.npy', allow_pickle=True).item()

    keys_path = parent_directory + '/' + lmdb_database.split('.lmdb')[0] + '_keys'
    database_path = parent_directory + '/' + lmdb_database
    chunk_ind = - 1
    with tqdm(total=max_chunk) as pbar:
        while chunk_ind < max_chunk - 1:
            pbar.update(1)
            chunk_ind += 1

            if not (max_chunk > chunk_ind >= min_chunk):
                continue

            data_dict = {}
            samples = []
            os.chdir(dirs[chunk_ind])
            for file in os.listdir():
                data = open(file, 'r')
                for line in tqdm(data):
                    try:
                        mol = Chem.MolFromSmiles(line)
                        mol2 = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol2)
                        conf = mol2.GetConformer()
                    except ValueError as e:
                        continue

                    coords = np.array(conf.GetPositions())
                    atom_types = [atom.GetAtomicNum() for atom in mol2.GetAtoms()]

                    # molecule sizes filter
                    if len(atom_types) < 6 or len(atom_types) > 100:
                        continue

                    # atom types filter
                    if not set(atom_types).issubset([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]):
                        continue

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
                        continue

                    samples.append(sample.to_dict())

            if len(samples) > 0:
                sample_inds = [overall_index + cc_idx for cc_idx in range(1, len(samples) + 1)]
                data_dict.update({str(k): v for k, v in zip(sample_inds, samples)})
                overall_index += len(samples)

            np.save(keys_path, overall_index)

            map_size = write_lmdb(database_path, map_size, data_dict)

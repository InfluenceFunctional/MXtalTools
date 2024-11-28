import gzip
import multiprocessing as mp
import os
from pathlib import Path
from time import time

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

import numpy as np
import torch
from tqdm import tqdm

from mxtaltools.common.utils import chunkify
from mxtaltools.conformer_generation.conformer_generator import generate_random_conformers_from_smiles
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.dataset_management.data_manager import DataManager


def process_smiles_list(lines: list, chunk_path, allowed_atom_types, **conf_kwargs):
    samples = []
    for line in lines:
        sample = process_smiles(line, allowed_atom_types, to_dict=False, **conf_kwargs)
        if sample is not None:
            samples.append(sample)

    print(f"finished processing smiles list with {len(samples)} samples")
    torch.save(samples, chunk_path)
    del samples
    # with open(chunk_path, 'wb') as handle:
    #     pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_smiles(smile: str,
                   allowed_atom_types,
                   to_dict: bool = True,
                   max_radius: float = 15,
                   protonate: bool = True,
                   rotamers_per_sample: int = 1,
                   allow_simple_hydrogen_rotations: bool = False):
    if rotamers_per_sample > 1:
        assert False, "Multiple rotamers not implemented"
    coords, atom_types = generate_random_conformers_from_smiles(smile,
                                                                protonate=protonate,
                                                                max_rotamers_per_samples=rotamers_per_sample,
                                                                allow_simple_hydrogen_rotations=allow_simple_hydrogen_rotations)
    if coords is False:
        return None

    coords = coords[0]
    atom_types = atom_types[0]
    # molecule sizes filter
    if len(atom_types) < 6 or len(atom_types) > 100:
        return None

    # atom types filter
    if not set(atom_types).issubset(allowed_atom_types):  #[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]):
        return None

    sample = CrystalData(
        x=torch.tensor(atom_types, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float32),
        smiles=smile,
        identifier=smile,
        y=torch.zeros(1, dtype=torch.float32),
        require_crystal_features=False,
    )

    # molecule radius filter
    if sample.radius > max_radius:
        return None

    if to_dict:
        return sample.to_dict()
    else:
        return sample


if __name__ == '__main__':
    parent_directory = r'D:\crystal_datasets\zinc22'
    chunks_dir = os.path.join(Path(parent_directory), 'chunks')

    os.chdir(parent_directory)
    dirs = os.listdir()

    chunk_ind = - 1
    min_chunk = 0
    max_chunk = min(100000, len(dirs))

    pool = mp.Pool(mp.cpu_count() - 1)

    datapoint_counter = 0
    dataset_length = 100000
    t0 = time()
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

                        if combo_path[-3:] == '.gz':
                            with gzip.open(combo_path, 'r') as f:
                                lines = f.readlines()
                        elif combo_path[-4:] == '.smi':
                            with open(combo_path, 'r') as f:
                                lines = f.readlines()
                        else:
                            pass

                        chunks = chunkify(lines, int(np.ceil(len(lines) / 1000)))
                        del lines

                        for chunk_ind2, chunk in enumerate(chunks):
                            chunk_path = chunks_dir + f'/chunk_{chunk_ind}_{file_ind}_{chunk_ind2}.pkl'
                            # process_smiles_list(chunk, chunk_path)
                            if not os.path.exists(chunk_path):
                                pool.apply_async(process_smiles_list, args=(chunk, chunk_path, {}))
                                datapoint_counter += len(chunk)

                            if datapoint_counter >= dataset_length:
                                print('Hit required number of samples')
                                break

    pool.close()
    pool.join()

    print(time() - t0)

    miner = DataManager(device='cpu',
                        datasets_path=r"D:\crystal_datasets/",
                        chunks_path=chunks_dir,
                        dataset_type='molecule')
    miner.process_new_dataset(new_dataset_name='temp_zinc_conf_dataset')

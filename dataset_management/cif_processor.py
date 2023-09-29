import os
from ccdc import io
import pandas as pd
import tqdm
import warnings
from random import shuffle

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

from dataset_management.featurization_utils import extract_crystal_data, featurize_molecule, crystal_filter, chunkify

n_chunks = 1000
chunks_path = r'D:/crystal_datasets/featurized_chunks/'
cifs_path = r'D:/CSD_dump/'
os.chdir(cifs_path)
cifs_list = os.listdir()

'''
todo:
post-generation analysis:
    -: cell parameterization
    -: filtering
        -: mol fingerprint
        -: identifiers
'''
print(f"Breaking dataset into {n_chunks} chunks")
chunks_list = chunkify(cifs_list, n_chunks)
chunk_inds = [i for i in range(len(chunks_list))]
start_ind, stop_ind = 0, len(chunks_list)

shuffle(chunk_inds)  # optionally do it in random order
chunks_list = [chunks_list[ind] for ind in chunk_inds]

for chunk_ind, chunk in zip(chunk_inds, chunks_list[start_ind:stop_ind]):
    print(f"Starting chunk {chunk_ind}")
    increment_df = None
    if not os.path.exists(chunks_path + f"chunk_{chunk_ind}.pkl"):

        for ind, cif_path in enumerate(tqdm.tqdm(chunk)):

            reader = io.CrystalReader(cif_path, format='cif')

            for crystal in reader:  # one cif file may have many crystals in it
                passed_filter, unit_cell, rd_mols = crystal_filter(crystal)
                if passed_filter:  # filter various undesirable traits
                    crystal_dict, mol_volumes = extract_crystal_data(crystal, unit_cell)
                    molecules = []
                    for i_c, rd_mol in enumerate(rd_mols):
                        molecules.append(featurize_molecule(crystal, crystal_dict, rd_mol, mol_volumes[i_c], component_num=i_c))

                    crystal_keys = list(crystal_dict.keys())
                    for key in crystal_keys:
                        crystal_dict['crystal_' + key] = crystal_dict[key]
                        del crystal_dict[key]

                    for key in molecules[0].keys():
                        crystal_dict[key] = []
                        for molecule in molecules:
                            crystal_dict[key].append(molecule[key])

                    new_df = pd.DataFrame()
                    for key in crystal_dict.keys():
                        new_df[key] = [crystal_dict[key]]

                    if increment_df is None:
                        increment_df = new_df
                    else:
                        increment_df = pd.concat([increment_df, new_df])

        increment_df.to_pickle(chunks_path + f"chunk_{chunk_ind}.pkl")

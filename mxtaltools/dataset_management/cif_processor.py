import os
from ccdc import io
import pandas as pd
import tqdm
import warnings
from random import shuffle
from mxtaltools.dataset_management.featurization_utils import extract_crystal_data, featurize_molecule, crystal_filter, chunkify
import glob

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

n_chunks = 2  # too many chunks can cause problems e.g., if some have zero valid entries
use_filenames_for_identifiers = False  # for blind test submissions & other cases where the identifiers are in the filenames rather than in the .cif itself
filter_by_targets = False
chunks_path = r'D:/crystal_datasets/aspirin_chunks/'  # where you would like processed dataset chunks to be stored before collation into final dataset

chunk_prefix = ''
cifs_path = r'D:\crystal_datasets\aspirin_zzp_1/'
target_identifiers = None
# target_identifiers = ['OBEQUJ', 'OBEQOD', 'OBEQET', 'XATJOT', 'OBEQIX', 'KONTIQ',
#     'NACJAF', 'XAFPAY', 'XAFQON', 'XAFQIH', 'XAFPAY01', 'XAFPAY02', 'XAFPAY03', 'XAFPAY04']
# chunk_prefix = 'BT_5'
# cifs_path = r'D:\crystal_datasets\blind_test_3-6_cifs\blind_test_5\bk5106sup2\file_dump/'  # where are the cifs
# chunk_prefix = 'BT_6'
# cifs_path = r'D:\crystal_datasets\blind_test_3-6_cifs\blind_test_6\gp5080sup2'
os.chdir(cifs_path)
#cifs_list = os.listdir()
cifs_list = glob.glob(r'*/*.cif', recursive=True) + glob.glob('*.cif')  # plus any free dumps directly in this dir
if target_identifiers is not None and filter_by_targets:
    target_cifs = [cif for cif in cifs_list if cif.split('.cif')[0] in target_identifiers]
    cifs_list = target_cifs

if not os.path.exists(chunks_path):
    os.mkdir(chunks_path)

n_chunks = min(n_chunks, len(cifs_list))
print(f"Breaking dataset into {n_chunks} chunks")
chunks_list = chunkify(cifs_list, n_chunks)
chunk_inds = [i for i in range(len(chunks_list))]
start_ind, stop_ind = 0, len(chunks_list)

shuffle(chunk_inds)  # optionally do it in random order
chunks_list = [chunks_list[ind] for ind in chunk_inds]

for chunk_ind, chunk in zip(chunk_inds, chunks_list[start_ind:stop_ind]):  # todo consider adding indexing over multiple or nested directories
    increment_df = None
    if not os.path.exists(chunks_path + chunk_prefix + f"_chunk_{chunk_ind}.pkl"):
        print(f"Starting chunk {chunk_ind} with {len(chunk)} cifs")
        for ind, cif_path in enumerate(tqdm.tqdm(chunk)):
            reader = io.CrystalReader(cif_path, format='cif')
            if len(reader) > 1:
                print(f"Starting entry {ind} with {len(reader)} entries")
            for crystal_ind in range(len(reader)):  # one cif file may have many crystals in it
                try:
                    crystal = reader[crystal_ind]
                except RuntimeError:  # some crystals fail to load due to timeout in refine_bonds
                    continue  # skip this crystal

                passed_filter, unit_cell, rd_mols = crystal_filter(crystal)
                if passed_filter:  # filter various undesirable traits
                    if use_filenames_for_identifiers:  # filename includes BT target, group name, any built-in identifications, and an extra index for safety
                        identifier = cif_path.split('.cif')[0] + '_' + crystal.identifier + '_' + str(crystal_ind)
                    else:
                        identifier = crystal.identifier

                    crystal_dict, mol_volumes = extract_crystal_data(identifier, crystal, unit_cell)
                    molecules = []
                    for i_c, rd_mol in enumerate(rd_mols):  # one crystal may have Z prime molecules
                        molecules.append(featurize_molecule(crystal, rd_mol, mol_volumes[i_c], component_num=i_c))

                    # check for custom metrics
                    with open(cif_path, 'r') as f:
                        text = f.read()

                        if 'zzp' in text:
                            lines = text.split('\n')
                            for line_ind, line in enumerate(lines):
                                if 'zzp' in line:
                                    break
                            prop_line = lines[line_ind + 2]
                            crystal_dict['zzp_cost'] = prop_line.split()[0]
                            crystal_dict['contact_overlap_cost'] = prop_line.split()[-1]


                    # the rest is boilerplate for indexing and saving each crystal as a new DataFrame row
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

        if increment_df is not None:
            increment_df.to_pickle(chunks_path + f"{chunk_prefix}_chunk_{chunk_ind}.pkl")

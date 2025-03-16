import glob
import os
import warnings
from shutil import copyfile

import torch
import tqdm
from ccdc import io
from rdkit import Chem
from functools import reduce

from mxtaltools.dataset_utils.construction.featurization_utils import crystal_filter, \
    chunkify_path_list

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error


def process_dataset_chunks(n_chunks: int,
                           cifs_path: str,
                           chunks_path: str,
                           chunk_prefix: str,
                           target_identifiers: list = None,
                           filter_by_targets: bool = False):
    os.chdir(cifs_path)

    cifs_list = glob.glob(r'*/*.cif', recursive=True) + glob.glob('*.cif')  # plus any free dumps directly in this dir
    if target_identifiers is not None and filter_by_targets:
        target_cifs = [cif for cif in cifs_list if cif.split('.cif')[0] in target_identifiers]
        cifs_list = target_cifs

    if not os.path.exists(chunks_path):
        os.mkdir(chunks_path)

    chunk_inds, chunks_list, start_ind, stop_ind = chunkify_path_list(cifs_list, n_chunks)

    for chunk_ind, chunk in zip(chunk_inds, chunks_list[start_ind:stop_ind]):
        if not os.path.exists(chunks_path + chunk_prefix + f"_chunk_{chunk_ind}.pkl"):
            # if this chunk has not already been processed, process it
            chunk_data_list = process_chunk(chunk, chunk_ind)
            if len(chunk_data_list) > 0:
                for file in chunk_data_list:
                    copyfile(file, chunks_path + file)
            torch.save([], chunks_path + f"{chunk_prefix}_chunk_{chunk_ind}.pkl")


def generate_dataset_molfiles(n_chunks: int,
                              cifs_path: str,
                              chunks_path: str,
):
    os.chdir(cifs_path)

    cifs_list = glob.glob(r'*/*.cif', recursive=True) + glob.glob('*.cif')  # plus any free dumps directly in this dir

    if not os.path.exists(chunks_path):
        os.mkdir(chunks_path)

    chunk_inds, chunks_list, start_ind, stop_ind = chunkify_path_list(cifs_list, n_chunks)

    for chunk_ind, chunk in zip(chunk_inds, chunks_list[start_ind:stop_ind]):
        generate_molfiles(chunk, chunk_ind, chunks_path)


def generate_molfiles(chunk, chunk_ind, chunks_path: str):
    print(f"Starting chunk {chunk_ind} with {len(chunk)} cifs")

    for ind, cif_path in enumerate(tqdm.tqdm(chunk)):
        if cif_path[-4:] == '.cif':
            molfile_name = cif_path[:-4] + '.mol'
            if not os.path.exists(molfile_name):
                reader = io.CrystalReader(cif_path, format='cif')
                crystal = reader[0]
                rd_mols = []
                for component in crystal.molecule.components:
                    mol = Chem.MolFromMol2Block(component.to_string('mol2'),
                                                sanitize=True,
                                                removeHs=False)
                    rd_mols.append(mol)
                if len(rd_mols) > 1:
                    molfile = Chem.MolToMolBlock(reduce(Chem.CombineMols, rd_mols))
                else:
                    molfile = Chem.MolToMolBlock(rd_mols[0])
                with open(chunks_path + molfile_name, 'w') as f:
                    f.write(molfile)


def process_chunk(chunk, chunk_ind):
    print(f"Starting chunk {chunk_ind} with {len(chunk)} cifs")
    failed_parameterization_counter = 0
    failed_checks_counter = 0
    data_list = []
    for ind, cif_path in enumerate(tqdm.tqdm(chunk)):
        if cif_path[-4:] == '.cif':
            reader = io.CrystalReader(cif_path, format='cif')
            passed = True
            if len(reader) > 1:
                print(f"Starting entry {ind} with {len(reader)} entries")

            for crystal_ind in range(len(reader)):  # one cif file may have many crystals in it
                try:
                    crystal = reader[crystal_ind]
                except RuntimeError:  # some crystals fail to load due to timeout in refine_bonds
                    passed = False
                    continue  # skip this crystal
                passed_filter, unit_cell, rd_mols = crystal_filter(crystal,
                                                                   max_heavy_atoms=10000,
                                                                   protonation_state='protonated',
                                                                   max_atomic_number=10000)
                if not passed_filter:
                    passed = False

            if passed:
                data_list.append(cif_path)

    return data_list


if __name__ == '__main__':
    # process_dataset_chunks(n_chunks=1000,
    #                        #cifs_path='D:/crystal_datasets/CSD_cifs/',
    #                        cifs_path='D:/crystal_datasets/CSD_clean_w_protons/',
    #                        #chunks_path='D:/crystal_datasets/protonated_CSD_identifiers/',
    #                        chunks_path='D:/crystal_datasets/CSD_clean_w_protons/',
    #                        chunk_prefix='',
    #                        target_identifiers=None,
    #                        filter_by_targets=False)

    generate_dataset_molfiles(n_chunks=1000,
                              #cifs_path='D:/crystal_datasets/CSD_cifs/',
                              cifs_path='D:/crystal_datasets/CSD_clean_w_protons/',
                              #chunks_path='D:/crystal_datasets/protonated_CSD_identifiers/',
                              chunks_path='D:/crystal_datasets/CSD_clean_w_protons_molfiles/'
                              )

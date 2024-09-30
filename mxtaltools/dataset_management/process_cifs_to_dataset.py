import glob
import os
import warnings
from shutil import copyfile

import numpy as np
import torch
import tqdm
from ccdc import io

from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.dataset_management.featurization_utils import extract_crystal_data, featurize_molecule, crystal_filter, \
    chunkify_path_list, extract_custom_cif_data, rebuild_reparameterize_unit_cell
from mxtaltools.constants.space_group_info import SYM_OPS

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


def process_chunk(chunk, chunk_ind):
    print(f"Starting chunk {chunk_ind} with {len(chunk)} cifs")
    failed_parameterization_counter = 0
    failed_checks_counter = 0
    data_list = []
    for ind, cif_path in enumerate(tqdm.tqdm(chunk)):
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
    process_dataset_chunks(n_chunks=1000,
                           cifs_path='D:/crystal_datasets/CSD_cifs/',
                           chunks_path='D:/crystal_datasets/protonated_CSD_identifiers/',
                           chunk_prefix='',
                           target_identifiers=None,
                           filter_by_targets=False)

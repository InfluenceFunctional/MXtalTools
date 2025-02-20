import glob
import os
import warnings

import numpy as np
import torch
import tqdm
from ccdc import io

from mxtaltools.dataset_utils.CrystalData import CrystalData
from mxtaltools.dataset_utils.construction.featurization_utils import extract_crystal_data, featurize_molecule, crystal_filter, \
    chunkify_path_list, extract_custom_cif_data, rebuild_reparameterize_unit_cell
from mxtaltools.constants.space_group_info import SYM_OPS

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

"""
convert raw cif files into featurized chunks for later collation
"""

def process_dataset_chunks(n_chunks: int,
                           cifs_path: str,
                           chunks_path: str,
                           chunk_prefix: str,
                           use_filenames_for_identifiers: bool = False,
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
            chunk_data_list = process_chunk(chunk, chunk_ind, use_filenames_for_identifiers)
            if len(chunk_data_list) > 0:
                torch.save(chunk_data_list, chunks_path + f"{chunk_prefix}_chunk_{chunk_ind}.pkl")
            else:
                torch.save([], chunks_path + f"{chunk_prefix}_chunk_{chunk_ind}.pkl")


def process_chunk(chunk, chunk_ind, use_filenames_for_identifiers):
    print(f"Starting chunk {chunk_ind} with {len(chunk)} cifs")
    failed_parameterization_counter = 0
    failed_checks_counter = 0
    data_list = []
    for ind, cif_path in enumerate(tqdm.tqdm(chunk)):
        reader = io.CrystalReader(cif_path, format='cif')
        if len(reader) > 1:
            print(f"Starting entry {ind} with {len(reader)} entries")
        for crystal_ind in range(len(reader)):  # one cif file may have many crystals in it
            try:
                crystal = reader[crystal_ind]
            except RuntimeError:  # some crystals fail to load due to timeout in refine_bonds
                continue  # skip this crystal

            passed_filter, unit_cell, rd_mols = crystal_filter(crystal,
                                                               max_heavy_atoms=9,
                                                               protonation_state='protonated',
                                                               max_atomic_number=9)
            if not passed_filter:
                failed_checks_counter += 1
            else:
                if use_filenames_for_identifiers:  # filename includes BT target, group name, any built-in identifications, and an extra index for safety
                    identifier = cif_path.split('.cif')[0] + '_' + crystal.identifier + '_' + str(crystal_ind)
                else:
                    identifier = crystal.identifier

                crystal_dict = extract_crystal_data(identifier, crystal, unit_cell)
                molecules = []
                for i_c, rd_mol in enumerate(rd_mols):  # one crystal may have Z prime molecules
                    molecules.append(featurize_molecule(crystal, rd_mol, component_num=i_c,
                                                        protonation_state='protonated'))

                # check for custom metrics in the CIF text
                crystal_dict = extract_custom_cif_data(cif_path, crystal_dict)

                ### parameterize the crystal pose & symmetry parameters
                try:
                    sym_ops_are_standard = np.all(np.stack(SYM_OPS[crystal_dict['space_group_number']]) == np.stack(crystal_dict['symmetry_operators']))
                except ValueError:  # sometimes, there are not even the correct multiplicity of space group indexing
                    failed_parameterization_counter += 1
                    continue  # skip this sample

                sym_ops = np.stack(crystal_dict['symmetry_operators'])
                atomic_numbers = np.concatenate([mol['atom_atomic_numbers'] for mol in molecules])
                mol_ind = np.concatenate([np.ones(len(mol['atom_atomic_numbers']), dtype=np.int32) * ind for ind, mol in
                                          enumerate(molecules)])

                (pose_params_list, handedness_list, is_well_defined, reconstructed_unit_cell_coords_tensor,
                 reparameterized_aunit_coords_tensor,
                 parameterization_and_reconstruction_successful) = rebuild_reparameterize_unit_cell(molecules,
                                                                                                    crystal_dict)

                if parameterization_and_reconstruction_successful:  # if we can confidently analyze and rebuild this crystal
                    crystaldata = generate_crystaldata_sample(atomic_numbers, crystal_dict, handedness_list,
                                                              is_well_defined, mol_ind, molecules, pose_params_list,
                                                              reconstructed_unit_cell_coords_tensor,
                                                              reparameterized_aunit_coords_tensor, sym_ops,
                                                              sym_ops_are_standard)

                    data_list.append(crystaldata)
                else:
                    failed_parameterization_counter += 1

    print(f"Cell reconstruction failed {failed_parameterization_counter} times out of {len(chunk)}")
    print(f"Cell analysis failed {failed_checks_counter} times out of {len(chunk)}")

    return data_list


def generate_crystaldata_sample(atomic_numbers, crystal_dict, handedness_list, is_well_defined, mol_ind, molecules,
                                pose_params_list, reconstructed_unit_cell_coords_tensor,
                                reparameterized_aunit_coords_tensor, sym_ops, sym_ops_are_standard):

    crystaldata = CrystalData(  # CRYTODO
        x=torch.tensor(atomic_numbers, dtype=torch.long),
        pos=reparameterized_aunit_coords_tensor,
        mol_ind=torch.tensor(mol_ind, dtype=torch.int32),
        cell_lengths=torch.tensor(
            np.stack([
                crystal_dict['lattice_a'], crystal_dict['lattice_b'], crystal_dict['lattice_c']
            ]), dtype=torch.float32),
        cell_angles=torch.tensor(
            np.stack([
                crystal_dict['lattice_alpha'], crystal_dict['lattice_beta'],
                crystal_dict['lattice_gamma']
            ]), dtype=torch.float32),
        z_prime=int(crystal_dict['z_prime']),
        sg_ind=int(crystal_dict['space_group_number']),  # default to P1
        pose_parameters=[torch.tensor(pose_params_list[zp], dtype=torch.float32) for zp in
                         range(int(crystal_dict['z_prime']))],
        smiles='Z'.join([mol['molecule_smiles'] for mol in molecules]),  # separate molecules by 'Z'
        identifier=crystal_dict['identifier'],
        unit_cell_pos=reconstructed_unit_cell_coords_tensor,
        nonstandard_symmetry=not sym_ops_are_standard,
        symmetry_operators=sym_ops,
        aunit_handedness=[int(handedness_list[zp]) for zp in range(int(crystal_dict['z_prime']))],
        is_well_defined=is_well_defined,
        fingerprint=np.concatenate([np.array(mol['molecule_fingerprint']) for mol in molecules]),
        y=torch.ones(1),  # this dummy variable helps us later on - save us having to rebuild the dataset at runtime
    )
    return crystaldata


if __name__ == '__main__':
    # process_dataset_chunks(n_chunks=1000,
    #                        cifs_path='D:/crystal_datasets/CSD_cifs/',
    #                        chunks_path='D:/crystal_datasets/CSD_featurized_chunks/',
    #                        chunk_prefix='',
    #                        use_filenames_for_identifiers=False,
    #                        target_identifiers=None,
    #                        filter_by_targets=False)

    process_dataset_chunks(n_chunks=1000,
                           cifs_path='D:/crystal_datasets/CSD_cifs/',
                           chunks_path='D:/crystal_datasets/CSD_QM9_featurized_chunks/',
                           chunk_prefix='',
                           use_filenames_for_identifiers=False,
                           target_identifiers=None,
                           filter_by_targets=False)

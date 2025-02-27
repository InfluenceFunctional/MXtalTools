import glob
import os
import warnings

import numpy as np
import torch
import tqdm
from ccdc import io

from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.dataset_utils.construction.featurization_utils import extract_crystal_data, featurize_molecule, \
    crystal_filter, \
    chunkify_path_list, extract_custom_cif_data, rebuild_reparameterize_unit_cell
from mxtaltools.dataset_utils.data_classes import MolData, MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list

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
                                                               max_heavy_atoms=100,
                                                               protonation_state='deprotonated',
                                                               max_atomic_number=100)
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
                                                        protonation_state='deprotonated'))

                # check for custom metrics in the CIF text
                crystal_dict = extract_custom_cif_data(cif_path, crystal_dict)

                ### parameterize the crystal pose & symmetry parameters
                try:
                    sym_ops_are_standard = np.all(np.stack(SYM_OPS[crystal_dict['space_group_number']]) == np.stack(
                        crystal_dict['symmetry_operators']))
                except ValueError:  # sometimes, there are not even the correct multiplicity of space group indexing
                    failed_parameterization_counter += 1
                    continue  # skip this sample

                sym_ops = np.stack(crystal_dict['symmetry_operators'])
                atomic_numbers = np.concatenate([mol['atom_atomic_numbers'] for mol in molecules])

                mol = MolData(
                    z=torch.tensor(atomic_numbers, dtype=torch.long),
                    pos=torch.tensor(molecules[0]['atom_coordinates'], dtype=torch.float32),
                    x=torch.tensor(molecules[0]['atom_partial_charge'], dtype=torch.float32),
                    smiles=molecules[0]['molecule_smiles'],
                    skip_mol_analysis=False,
                    y=torch.ones(1),
                    graph_x=torch.ones(1),
                )

                crystal = MolCrystalData(
                    molecule=mol,
                    sg_ind=int(crystal_dict['space_group_number']),
                    cell_lengths=torch.tensor(
                        np.stack([
                            crystal_dict['lattice_a'], crystal_dict['lattice_b'], crystal_dict['lattice_c']
                        ]), dtype=torch.float32),
                    cell_angles=torch.tensor(
                        np.stack([
                            crystal_dict['lattice_alpha'], crystal_dict['lattice_beta'],
                            crystal_dict['lattice_gamma']
                        ]), dtype=torch.float32),
                    identifier=crystal_dict['identifier'],
                    unit_cell_pos=np.stack(crystal_dict['unit_cell_coordinates']),
                    nonstandard_symmetry=not sym_ops_are_standard,
                    symmetry_operators=sym_ops,
                    fingerprint=molecules[0]['molecule_fingerprint']
                )

                """
                extract pose information
                """
                crystal_batch = collate_data_list([crystal])
                aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = crystal_batch.reparameterize_unit_cell()

                crystal.aunit_centroid = aunit_centroid
                crystal.aunit_orientation = aunit_orientation
                crystal.aunit_handedness = int(aunit_handedness)
                crystal.is_well_defined = bool(is_well_defined)
                crystal.pos = pos

                """
                rebuild the crystal with these parameters to confirm correct reconstruction
                """
                crystal_batch = collate_data_list([crystal])
                crystal_batch.build_unit_cell()
                aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = crystal_batch.reparameterize_unit_cell()
                rebuild_successful = crystal_rebuild_checks(aunit_centroid, aunit_handedness, aunit_orientation,
                                                            crystal, crystal_batch, crystal_dict, is_well_defined, pos,
                                                            )

                if rebuild_successful:
                    data_list.append(crystal)
                else:
                    failed_parameterization_counter += 1

    print(f"Cell reconstruction failed {failed_parameterization_counter} times out of {len(chunk)}")
    print(f"Crystal filtered {failed_checks_counter} times out of {len(chunk)}")

    return data_list


def crystal_rebuild_checks(aunit_centroid, aunit_handedness, aunit_orientation, crystal, crystal_batch, crystal_dict,
                           is_well_defined, pos):
    rebuild_successful = False
    if torch.all(torch.isclose(aunit_centroid, crystal.aunit_centroid, atol=1e-2)) and \
            torch.all(torch.isclose(aunit_orientation, crystal.aunit_orientation, atol=1e-2)) and \
            int(aunit_handedness) == crystal.aunit_handedness and \
            bool(is_well_defined) == crystal.is_well_defined and \
            torch.all(torch.isclose(pos, crystal.pos, atol=1e-2)):

        uc_orig = torch.tensor(crystal_dict['unit_cell_coordinates'], dtype=torch.float32)
        uc_build = crystal_batch.unit_cell_pos[0]
        old_shape = uc_build.shape
        new_shape = (old_shape[0] * old_shape[1], 3)
        distmat = torch.cdist(uc_orig.reshape(new_shape), uc_build.reshape(new_shape))
        if torch.sum(distmat < 1e-2) == len(distmat):  # every atom matched
            try:
                crystal_batch.validate_cell_params(check_crystal_system=True)
                rebuild_successful = True
            except AssertionError:
                pass
    return rebuild_successful


'''
from mxtaltools.dataset_utils.utils import collate_data_list
crystal_batch = collate_data_list([crystal])
orientations, handedness = extract_aunit_orientation(
    crystal_batch,
    False,
    canonicalize_orientation=True
)
aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = crystal_batch.reparameterize_unit_cell()


'''


if __name__ == '__main__':
    process_dataset_chunks(n_chunks=1000,
                           cifs_path='D:/crystal_datasets/CSD_cifs/',
                           chunks_path='D:/crystal_datasets/CSD_featurized_chunks/',
                           chunk_prefix='',
                           use_filenames_for_identifiers=False,
                           target_identifiers=None,
                           filter_by_targets=False)



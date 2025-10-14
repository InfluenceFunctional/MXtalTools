"""
For processing crystal datasets from large lists of .cif files

"""
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
    chunkify_path_list, extract_custom_cif_data
from mxtaltools.dataset_utils.data_classes import MolData, MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

"""
convert raw cif files into featurized chunks for later collation
"""


def process_cifs_to_chunks(n_chunks: int,
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
        chunk_output_path = chunks_path + chunk_prefix + f"_chunk_{chunk_ind}.pkl"
        if not os.path.exists(chunk_output_path):
            # if this chunk has not already been processed, process it
            chunk_data_list = process_chunk(chunk, chunk_ind, use_filenames_for_identifiers)
            if len(chunk_data_list) > 0:
                torch.save(chunk_data_list, chunk_output_path)
            else:
                torch.save([], chunk_output_path)


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

                mols = []
                for m in molecules:
                    mol = MolData(
                        z=torch.tensor(m['atom_atomic_numbers'], dtype=torch.long),
                        pos=torch.tensor(m['atom_coordinates'], dtype=torch.float32),
                        x=torch.tensor(m['atom_partial_charge'], dtype=torch.float32),
                        fingerprint=torch.tensor(m['molecule_fingerprint'], dtype=torch.float32)[None, ...],
                        smiles=m['molecule_smiles'],
                        do_mol_analysis=True,
                    )
                    mols.append(mol)

                identical = cocrystal_check(crystal_dict, mols)
                if not identical:
                    continue  # MK I don't even want to deal with cocrystals right now

                z_prime = torch.tensor(int(crystal_dict['z_prime']), dtype=torch.long)
                # generate and parameterize separately the Z' equivalents
                crystals = []
                for mol in mols:
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
                        aunit_orientation=torch.zeros(3, dtype=torch.float32), # dummy
                        aunit_centroid=torch.zeros(3,dtype=torch.float32), # dummy
                        aunit_handedness=torch.zeros(1, dtype=torch.bool), # dummy
                        identifier=crystal_dict['identifier'],
                        nonstandard_symmetry=not sym_ops_are_standard,
                        symmetry_operators=sym_ops,
                        z_prime=torch.ones(1, dtype=torch.long),
                        do_box_analysis=True,
                        cocrystal = not identical,
                        max_z_prime = 1
                    )
                    crystals.append(crystal)

                """
                extract pose information (from each Z' structure)
                # """
                crystal_batch = collate_data_list(crystals, max_z_prime=1)

                molwise_ucell_coords = crystal_dict['unit_cell_coordinates']
                if len(crystals) > 1:
                    crystal_batch.unit_cell_pos = torch.tensor(np.concatenate(molwise_ucell_coords), dtype=torch.float32)
                    crystal_batch.unit_cell_batch = torch.arange(z_prime, dtype=torch.float32).repeat_interleave(crystal_batch.num_atoms * crystal_batch.sym_mult)

                else:
                    crystal_batch.unit_cell_pos = torch.tensor(np.concatenate(molwise_ucell_coords), dtype=torch.float32)
                    crystal_batch.unit_cell_batch = torch.zeros(len(crystal_batch.unit_cell_pos), dtype=torch.float32)

                aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = crystal_batch.reparameterize_unit_cell()

                """
                rebuild the crystal with these parameters to confirm correct reconstruction
                """
                for ind, crystal in enumerate(crystals):
                    crystal.aunit_centroid[:, :3] = aunit_centroid[ind][None, ...]
                    crystal.aunit_orientation[:, :3] = aunit_orientation[ind][None, ...]
                    crystal.aunit_handedness[:, :1] = aunit_handedness[ind]
                    crystal.is_well_defined = torch.tensor(is_well_defined, dtype=torch.bool)[ind]
                    crystal.pos = pos[crystal_batch.batch == ind]
                    crystal.box_analysis()

                rebuild_batch = collate_data_list(crystals, max_z_prime=1)
                rebuild_batch.pose_aunit()
                rebuild_batch.build_unit_cell()
                aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = rebuild_batch.reparameterize_unit_cell()
                rebuild_successful = crystal_rebuild_checks(aunit_centroid, aunit_handedness, aunit_orientation,
                                                            crystal, rebuild_batch, crystal_dict, is_well_defined, pos,
                                                            )
                if rebuild_successful:
                    if z_prime > 1:
                        # instantiate Z'>1 crystal object
                        mol = MolData(
                            z=rebuild_batch.z,
                            pos=rebuild_batch.pos,
                            x=rebuild_batch.x,
                            smiles='|'.join(rebuild_batch.smiles),
                            fingerprint=rebuild_batch.fingerprint.sum(0, keepdim=True),  # this will not work properly downstream, but we need it for Z'=1, for now
                            do_mol_analysis=False,  # combine manually below
                        )
                        mol.mass = torch.stack([m.mass for m in mols]).sum()
                        mol.mol_volume = torch.stack([m.mol_volume for m in mols]).sum()
                        mol.radius = torch.stack([m.radius for m in mols]).sum()

                        crystal = MolCrystalData(
                            molecule=mol,
                            sg_ind=rebuild_batch.sg_ind[0],
                            cell_lengths=rebuild_batch.cell_lengths[0],
                            cell_angles=rebuild_batch.cell_angles[0],
                            aunit_orientation=aunit_orientation.flatten(),
                            aunit_centroid=aunit_centroid.flatten(),
                            aunit_handedness=aunit_handedness.flatten(),
                            identifier=rebuild_batch.identifier[0],
                            nonstandard_symmetry=any(rebuild_batch.nonstandard_symmetry),
                            symmetry_operators=rebuild_batch.symmetry_operators[0],
                            z_prime=torch.ones(1, dtype=torch.long) * z_prime,
                            do_box_analysis=True,
                            cocrystal=not identical,
                            max_z_prime=6,
                            mol_ind=torch.arange(z_prime, dtype=torch.long).repeat_interleave(torch.stack([m.num_atoms for m in mols])),
                        )

                        data_list.append(crystal)
                    else:
                        mol = MolData(
                            z=rebuild_batch.z,
                            pos=rebuild_batch.pos,
                            x=rebuild_batch.x,
                            smiles=rebuild_batch.smiles[0],
                            fingerprint=rebuild_batch.fingerprint,
                            do_mol_analysis=True,  # combine manually below
                        )
                        crystal = MolCrystalData(
                            molecule=mol,
                            sg_ind=rebuild_batch.sg_ind[0],
                            cell_lengths=rebuild_batch.cell_lengths[0],
                            cell_angles=rebuild_batch.cell_angles[0],
                            aunit_orientation=aunit_orientation.flatten(),
                            aunit_centroid=aunit_centroid.flatten(),
                            aunit_handedness=aunit_handedness.flatten(),
                            identifier=rebuild_batch.identifier,
                            nonstandard_symmetry=any(rebuild_batch.nonstandard_symmetry),
                            symmetry_operators=rebuild_batch.symmetry_operators[0],
                            z_prime=torch.ones(1, dtype=torch.long),
                            do_box_analysis=True,
                            cocrystal=not identical,
                            max_z_prime=6,
                            mol_ind=torch.zeros(len(mol.z), dtype=torch.long),
                        )
                        data_list.append(crystal)
                else:
                    failed_parameterization_counter += 1

    print(f"Cell reconstruction failed {failed_parameterization_counter} times out of {len(chunk)}")
    print(f"Crystal filtered {failed_checks_counter} times out of {len(chunk)}")

    return data_list


def cocrystal_check(crystal_dict, mols):
    identical = True
    if crystal_dict['z_prime'] > 1:
        if not all([mol.num_atoms == mols[0].num_atoms for mol in mols]):
            identical = False

        # check for the same molecule
        if identical:
            fp = mols[0].smiles
            for m in mols[1:]:
                if m.smiles != fp: #not np.all(fp == m['molecule_fingerprint']):
                    identical = False
                    break

        if identical:
            # check if they are identical conformers (this is only a rough method)
            d1 = torch.cdist(mols[0].pos, mols[0].pos)
            f1 = d1.mean(1).sort(dim=0).values
            for mol in mols:
                f2 = torch.cdist(mol.pos, mol.pos).mean(1).sort(dim=0).values
                if not torch.allclose(f1, f2, atol=1e-1):
                    identical = False
    return identical


def crystal_rebuild_checks(aunit_centroid,
                           aunit_handedness,
                           aunit_orientation,
                           crystal,
                           crystal_batch,
                           crystal_dict,
                           is_well_defined,
                           pos):
    rebuild_successful = False
    # check reparameterization
    if not torch.all(crystal_batch.is_well_defined == torch.tensor(is_well_defined,dtype=torch.bool)):
        return False
    if not torch.all(torch.isclose(crystal_batch.aunit_centroid, aunit_centroid, rtol=1e-2)):
        return False
    if not torch.all(crystal_batch.aunit_handedness == aunit_handedness):
        return False
    if not torch.all(torch.isclose(crystal_batch.aunit_orientation, aunit_orientation, rtol=1e-02)):
        return False

    # compare full Z' unit cell positions and confirm they are all in agreement
    # TODO consider adding Z' by Z' check as well
    # extracted from CIF via CCDC
    uc_orig = torch.tensor(np.concatenate(crystal_dict['unit_cell_coordinates']), dtype=torch.float32)
    # rebuilt from parameters
    uc_build = crystal_batch.unit_cell_pos
    distmat = torch.cdist(uc_orig, uc_build)

    nearest_neighbors = np.argmin(distmat, axis=1)
    nn_dists = distmat[torch.arange(len(distmat)), nearest_neighbors]
    if (nn_dists.amax() < 5e-2) and (nn_dists.mean() < 1e-2):  # every atom matched
        try:
            crystal_batch.validate_cell_params(check_crystal_system=True)
            rebuild_successful = True
        except AssertionError:
            pass
    else:  # one failure mode is if the CCDC puts the molecules outside the box, because it computes centrids probably with protons when available, but we don't
        pass
    return rebuild_successful


if __name__ == '__main__':
    # full dataset processing
    # process_cifs_to_chunks(n_chunks=1000,
    #                        cifs_path='D:/crystal_datasets/CSD_dump/',
    #                        chunks_path='D:/crystal_datasets/CSD_featurized_chunks/',
    #                        chunk_prefix='',
    #                        use_filenames_for_identifiers=False,
    #                        target_identifiers=None,
    #                        filter_by_targets=False)
    #

    process_cifs_to_chunks(n_chunks=1,
                           cifs_path='D:/crystal_datasets/dafmuv/',
                           chunks_path='D:/crystal_datasets/',
                           chunk_prefix='',
                           use_filenames_for_identifiers=False,
                           target_identifiers=None,
                           filter_by_targets=False)



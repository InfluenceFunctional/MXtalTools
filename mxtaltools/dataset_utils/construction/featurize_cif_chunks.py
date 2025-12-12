"""
For processing crystal datasets from large lists of .cif files

"""
import glob
import os
import warnings
from collections import Counter

import numpy as np
import torch
import tqdm
from ccdc import io
from rdkit import Chem

from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.dataset_utils.construction.featurization_utils import extract_crystal_data, featurize_molecule, \
    crystal_filter, \
    chunkify_path_list, extract_custom_cif_data
from mxtaltools.dataset_utils.data_classes import MolData, MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

"""
convert raw cif files into featurized mxtaltools data chunks for later collation
"""


def process_cifs_to_chunks(n_chunks: int,
                           cifs_path: str,
                           chunks_path: str,
                           chunk_prefix: str,
                           use_filenames_for_identifiers: bool = False,
                           target_identifiers: list = None,
                           filter_by_targets: bool = False,
                           protonation_state: str = 'deprotonated',
                           max_z_prime: int = 6):
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
            chunk_data_list = process_chunk(chunk, chunk_ind,
                                            use_filenames_for_identifiers,
                                            protonation_state,
                                            max_z_prime)
            if len(chunk_data_list) > 0:
                torch.save(chunk_data_list, chunk_output_path)
            else:
                torch.save([], chunk_output_path)


def process_chunk(chunk, chunk_ind, use_filenames_for_identifiers, protonation_state, max_z_prime):
    print(f"Starting chunk {chunk_ind} with {len(chunk)} cifs")
    failed_parameterization_counter = 0
    failed_checks_counter = 0
    error_codes = []
    data_list = []
    for ind, cif_path in enumerate(tqdm.tqdm(chunk)):
        reader = io.CrystalReader(cif_path, format='cif')
        if len(reader) > 1:
            print(f"Starting entry {ind} with {len(reader)} entries")

        for crystal_ind in range(len(reader)):  # one cif file may have many crystals in it
            "instantiate ccdc crystal"
            try:
                csd_crystal = reader[crystal_ind]
                # Z and Z' do NOT copy into the reduced crystal for some reason, so we keep boty
                reduced_crystal = csd_crystal
                #reduced_crystal = csd_crystal.generate_reduced_crystal(csd_crystal)  # todo this breaks the symmetry settings - we still have to figure out how to do it nicely
                if protonation_state == 'protonated':
                    reduced_crystal.assign_bonds()
                    reduced_crystal.add_hydrogens()
                elif protonation_state == 'deprotonated':
                    reduced_crystal.assign_bonds()
                    reduced_crystal.remove_hydrogens()

            except RuntimeError:  # some crystals fail to load due to timeout in refine_bonds
                error_codes.append("Reader timeout")
                continue  # skip this crystal

            "crystal checks & rdkit mol"
            passed_filter, unit_cell, rd_mols, failure_mode = crystal_filter(
                csd_crystal,
                reduced_crystal,
                max_heavy_atoms=100,
                protonation_state=protonation_state,  # always deprotonate here
                max_atomic_number=100, max_z_prime=max_z_prime)
            if failure_mode is not None:
                error_codes.append(failure_mode)
                failed_checks_counter += 1
                continue

            if use_filenames_for_identifiers:  # filename includes BT target, group name, any built-in identifications, and an extra index for safety
                identifier = cif_path.split('.cif')[0] + '_' + csd_crystal.identifier + '_' + str(crystal_ind)
            else:
                identifier = csd_crystal.identifier

            # get crystal information
            crystal_dict = extract_crystal_data(identifier, csd_crystal, reduced_crystal, unit_cell)

            # get molecule information
            molecules = []
            for i_c, rd_mol in enumerate(rd_mols):  # one crystal may have Z' molecules
                molecules.append(featurize_molecule(reduced_crystal, rd_mol,
                                                    component_num=i_c,
                                                    protonation_state=protonation_state
                                                    # always deprotonate here - add protons in post-processing only
                                                    ))

            # check for custom metrics in the CIF text
            crystal_dict = extract_custom_cif_data(cif_path, crystal_dict)

            "check sym ops"
            try:
                sym_ops_are_standard = np.all(np.stack(SYM_OPS[crystal_dict['space_group_number']]) == np.stack(
                    crystal_dict['symmetry_operators']))
            except ValueError:  # sometimes, there are not even the correct multiplicity of space group indexing
                failed_checks_counter += 1
                error_codes.append('Incommensurate symmetry multiplicity')
                continue  # skip this sample

            sym_ops = np.stack(crystal_dict['symmetry_operators'])

            "instantiate molecules"
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

            if len(mols) > 1:
                if not cocrystal_check(mols):
                    error_codes.append("Non-identical cocrystals are not supported")
                    continue  # MK I can't deal with cocrystals right now

            """
            extract pose information (from each Z' structure)
            """
            # generate and parameterize separately the Z' equivalents
            z_prime = torch.tensor(int(crystal_dict['z_prime']), dtype=torch.long)
            zp1_crystals = init_zp1_crystals(
                crystal_dict, mols, sym_ops, sym_ops_are_standard)
            aunit_centroid, aunit_handedness, aunit_orientation, crystal_batch, is_well_defined, pos = extract_zp1_pose_info(
                crystal_dict, z_prime, zp1_crystals)

            """
            rebuild the crystal with these parameters to confirm correct reconstruction
            """
            for ind, zp1_crystal in enumerate(zp1_crystals):
                zp1_crystal.aunit_centroid[:, :3] = aunit_centroid[ind][None, ...]
                zp1_crystal.aunit_orientation[:, :3] = aunit_orientation[ind][None, ...]
                zp1_crystal.aunit_handedness[:, :1] = aunit_handedness[ind]
                zp1_crystal.is_well_defined = torch.tensor(is_well_defined, dtype=torch.bool)[ind]
                zp1_crystal.pos = pos[crystal_batch.batch == ind]
                zp1_crystal.box_analysis()

            rebuild_batch = collate_data_list(zp1_crystals, max_z_prime=1)
            rebuild_batch.pose_aunit()
            rebuild_batch.build_unit_cell()
            aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = rebuild_batch.reparameterize_unit_cell()
            rebuild_successful = crystal_rebuild_checks(aunit_centroid,
                                                        aunit_handedness,
                                                        aunit_orientation,
                                                        rebuild_batch,
                                                        crystal_dict,
                                                        is_well_defined,
                                                        )
            if not rebuild_successful:
                failed_parameterization_counter += 1
                error_codes.append("Failed reparameterization")
                continue

            if z_prime > 1:
                # instantiate Z'>1 crystal object
                mol = MolData(
                    z=rebuild_batch.z,
                    pos=rebuild_batch.pos,
                    x=rebuild_batch.x,
                    smiles='|'.join(rebuild_batch.smiles),
                    fingerprint=rebuild_batch.fingerprint.sum(0, keepdim=True),
                    do_mol_analysis=False,  # combine manually below
                )
                mol.mass = torch.stack([m.mass for m in mols]).sum()
                mol.mol_volume = torch.stack([m.mol_volume for m in mols]).sum()
                mol.radius = torch.stack([m.radius for m in mols]).sum()

                zp1_crystal = MolCrystalData(
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
                    cocrystal=False,
                    max_z_prime=max_z_prime,
                    mol_ind=torch.arange(z_prime, dtype=torch.long).repeat_interleave(
                        torch.stack([m.num_atoms for m in mols])),
                )

                data_list.append(zp1_crystal)
            else:
                mol = MolData(
                    z=rebuild_batch.z,
                    pos=rebuild_batch.pos,
                    x=rebuild_batch.x,
                    smiles=rebuild_batch.smiles[0],
                    fingerprint=rebuild_batch.fingerprint,
                    do_mol_analysis=True,  # combine manually below
                )
                zp1_crystal = MolCrystalData(
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
                    z_prime=torch.ones(1, dtype=torch.long),
                    do_box_analysis=True,
                    cocrystal=False,
                    max_z_prime=max_z_prime,
                    mol_ind=torch.zeros(len(mol.z), dtype=torch.long),
                )
                data_list.append(zp1_crystal)

    print(f"Cell reconstruction failed {failed_parameterization_counter} times out of {len(chunk)}")
    print(f"Crystal filtered {failed_checks_counter} times out of {len(chunk)}")
    print("Error codes:")
    print("\n".join(f"{c}: {n}" for c, n in Counter(error_codes).items()))

    return data_list


def init_zp1_crystals(crystal_dict, mols, sym_ops, sym_ops_are_standard):
    zp1_crystals = []
    for mol in mols:
        cell_lengths = torch.tensor(
            np.stack([
                crystal_dict['lattice_a'], crystal_dict['lattice_b'], crystal_dict['lattice_c']
            ]), dtype=torch.float32)
        cell_angles = torch.tensor(
            np.stack([
                crystal_dict['lattice_alpha'], crystal_dict['lattice_beta'],
                crystal_dict['lattice_gamma']
            ]), dtype=torch.float32)

        crystal = MolCrystalData(
            molecule=mol,
            sg_ind=int(crystal_dict['space_group_number']),
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
            aunit_orientation=torch.zeros(3, dtype=torch.float32),  # dummy
            aunit_centroid=torch.zeros(3, dtype=torch.float32),  # dummy
            aunit_handedness=torch.zeros(1, dtype=torch.bool),  # dummy
            identifier=crystal_dict['identifier'],
            nonstandard_symmetry=not sym_ops_are_standard,
            symmetry_operators=sym_ops,
            z_prime=torch.ones(1, dtype=torch.long),
            do_box_analysis=True,
            cocrystal=False,
            max_z_prime=1
        )
        zp1_crystals.append(crystal)
    return zp1_crystals


def extract_zp1_pose_info(crystal_dict, z_prime, zp1_crystals):
    crystal_batch = collate_data_list(zp1_crystals, max_z_prime=1)
    molwise_ucell_coords = crystal_dict['unit_cell_coordinates']
    if len(zp1_crystals) > 1:
        crystal_batch.unit_cell_pos = torch.tensor(np.concatenate(molwise_ucell_coords),
                                                   dtype=torch.float32)
        crystal_batch.unit_cell_batch = torch.arange(z_prime, dtype=torch.float32).repeat_interleave(
            crystal_batch.num_atoms * crystal_batch.sym_mult)

    else:
        crystal_batch.unit_cell_pos = torch.tensor(np.concatenate(molwise_ucell_coords),
                                                   dtype=torch.float32)
        crystal_batch.unit_cell_batch = torch.zeros(len(crystal_batch.unit_cell_pos), dtype=torch.float32)
    aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = crystal_batch.reparameterize_unit_cell()
    return aunit_centroid, aunit_handedness, aunit_orientation, crystal_batch, is_well_defined, pos


def cocrystal_check(mols):
    if not all([mol.num_atoms == mols[0].num_atoms for mol in mols]):
        return False

    if not torch.all(torch.stack([mol.z for mol in mols]).diff(dim=0) == 0):
        return False

    # check for the same molecule
    fp = mols[0].smiles
    for m in mols[1:]:
        if m.smiles != fp:  # not np.all(fp == m['molecule_fingerprint']):
            return False

    # check if they are identical conformers (this is only a rough method)
    d1 = torch.cdist(mols[0].pos, mols[0].pos)
    f1 = d1.mean(1).sort(dim=0).values
    for mol in mols:
        f2 = torch.cdist(mol.pos, mol.pos).mean(1).sort(dim=0).values
        if not torch.allclose(f1, f2, atol=1e-1):
            return False

    return True


def crystal_rebuild_checks(aunit_centroid,
                           aunit_handedness,
                           aunit_orientation,
                           crystal_batch,
                           crystal_dict,
                           is_well_defined,
                           ):
    rebuild_successful = False
    # check reparameterization
    if not torch.all(crystal_batch.is_well_defined == torch.tensor(is_well_defined, dtype=torch.bool)):
        return False
    if not torch.all(torch.isclose(crystal_batch.aunit_centroid, aunit_centroid, rtol=1e-2)):
        return False
    if not torch.all(crystal_batch.aunit_handedness.flatten() == aunit_handedness.flatten()):
        return False
    if not torch.all(torch.isclose(crystal_batch.aunit_orientation, aunit_orientation, rtol=1e-02)):
        return False

    # compare full Z' unit cell positions and confirm they are all in agreement
    # TODO consider adding Z' by Z' check as well
    # extracted from CIF via CCDC
    uc_orig = torch.tensor(np.concatenate(crystal_dict['unit_cell_coordinates']), dtype=torch.float32)
    uc_atoms = torch.tensor(np.concatenate(crystal_dict['unit_cell_atomic_numbers']), dtype=torch.long)
    # rebuilt from parameters
    uc_build = crystal_batch.unit_cell_pos
    distmat = torch.cdist(uc_orig, uc_build)

    nearest_neighbors = np.argmin(distmat, axis=1)
    nn_dists = distmat[torch.arange(len(distmat)), nearest_neighbors]
    matched_atom_types = uc_atoms == crystal_batch.z.repeat(crystal_batch.sym_mult[0])[nearest_neighbors]
    if (nn_dists.amax() < 5e-2) and (nn_dists.mean() < 1e-2) and torch.all(matched_atom_types):  # every atom matched
        try:
            crystal_batch.validate_cell_params(check_crystal_system=True)
            rebuild_successful = True
        except AssertionError:
            pass
    else:  # one failure mode is if the CCDC puts the molecules outside the box, because it computes centroids probably with protons when available, but we don't
        pass
    return rebuild_successful


""" visual comparison
from ase import Atoms
from ase.visualize import view
m1 = Atoms(positions=uc_orig.numpy(), symbols=np.concatenate(crystal_dict['unit_cell_atomic_numbers']))
m2 = Atoms(positions=uc_build.numpy(), symbols=crystal_batch.z.repeat(4))
view([m1, m2])
"""

if __name__ == '__main__':
    # full dataset processing
    # process_cifs_to_chunks(n_chunks=1000,
    #                        cifs_path='D:/crystal_datasets/CSD_dump/',
    #                        chunks_path='D:/crystal_datasets/CSD_featurized_chunks/',
    #                        chunk_prefix='',
    #                        use_filenames_for_identifiers=False,
    #                        target_identifiers=None,
    #                        filter_by_targets=False,
    #                        protonation_state='deprotonated')

    # process_cifs_to_chunks(n_chunks=1,
    #                        cifs_path='D:/crystal_datasets/dafmuv/',
    #                        chunks_path='D:/crystal_datasets/',
    #                        chunk_prefix='',
    #                        use_filenames_for_identifiers=False,
    #                        target_identifiers=None,
    #                        filter_by_targets=False,
    #                        protonation_state='deprotonated')


    # process_cifs_to_chunks(n_chunks=1,
    #                        cifs_path='D:/crystal_datasets/acridine/',
    #                        chunks_path='D:/crystal_datasets/acridine/',
    #                        chunk_prefix='',
    #                        use_filenames_for_identifiers=False,
    #                        target_identifiers=None,
    #                        filter_by_targets=False,
    #                        protonation_state='protonated')
    #

    # process_cifs_to_chunks(n_chunks=1,
    #                        cifs_path='D:/crystal_datasets/CSD_dump/',
    #                        chunks_path='D:/crystal_datasets/protonated_nicoam/',
    #                        chunk_prefix='',
    #                        use_filenames_for_identifiers=False,
    #                        target_identifiers=['NICOAM', 'NICOAM17'],
    #                        filter_by_targets=True,

    process_cifs_to_chunks(n_chunks=1,
                           cifs_path='D:/crystal_datasets/CSD_dump/',
                           chunks_path='D:/crystal_datasets/coumarin/',
                           chunk_prefix='',
                           use_filenames_for_identifiers=False,
                           target_identifiers=['COUMAR01'],
                           filter_by_targets=True,
                           protonation_state='protonated')


    process_cifs_to_chunks(n_chunks=1,
                           cifs_path='D:/crystal_datasets/CSD_dump/',
                           chunks_path='D:/crystal_datasets/xuldud/',
                           chunk_prefix='',
                           use_filenames_for_identifiers=False,
                           target_identifiers=['XULDUD'],
                           filter_by_targets=True,
                           protonation_state='protonated')

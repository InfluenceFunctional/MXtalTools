import glob
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path

from mxtaltools.common.mol_classifier_utils import convert_box_to_cell_params, convert_box_to_cell_vectors
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, ATOMIC_NUMBERS, VDW_RADII, ELECTRONEGATIVITY, GROUP, \
    PERIOD
from mxtaltools.constants.mol_classifier_constants import MOLECULE_NUM_ATOMS, structure2polymorph
import torch

from torch_scatter import scatter

"""
Utilities for processing LAMMPS outputs - dumps & screen files in the old e2emolmats format
"""


def flatten_dataframe(dataset1):
    """
    Flatten all elements of dataset1
    Speeds up I/O and makes compatible with new processing functions + polars integration
    Args:
        dataset1:

    Returns:

    """
    dataset2 = dataset1.copy()
    for column in dataset1.columns:
        try:
            vals = [np.concatenate(entry).flatten() for entry in dataset1[column]]
        except ValueError:
            vals = [np.array(entry).flatten() for entry in dataset1[column]]
        except TypeError:
            vals = [np.array(entry).flatten() for entry in dataset1[column]]

        dataset2[column] = vals
        del dataset1[column]  # for RAM purposed

    return dataset2


def process_thermo_data():
    f = open('screen.log', "r")
    text = f.read()
    lines = text.split('\n')
    f.close()
    hit_minimization = False
    skip = True
    results_dict = {'time_step': [],
                    'temp': [],
                    'E_pair': [],
                    'E_mol': [],
                    'E_tot': [],
                    'Press': [],
                    'Volume': [],
                    }

    if "Total wall time" not in text:  # skip analysis if the run crashed
        for key in results_dict.keys():
            results_dict[key] = np.zeros(1)
        return results_dict

    for ind, line in enumerate(lines):
        if 'ns/day' in line:
            text = line.split('ns/day')
            ns_per_day = float(text[0].split(' ')[1])
            results_dict['ns_per_day'] = ns_per_day

        if not hit_minimization:
            if 'Minimization stats' in line:
                hit_minimization = True
        elif skip:
            if "Step" in line:
                skip = False
                split_line = line.split(' ')
                volume_in_block = 'Volume' in split_line
                # print(ind)
        else:
            if "Loop" in line:
                skip = True

            if not skip:
                split_line = line.split(' ')
                entries = [float(entry) for entry in split_line if entry != '']
                for ind2, key in enumerate(results_dict.keys()):
                    if key != 'ns_per_day':
                        if key == 'Volume' and not volume_in_block:
                            results_dict[key].append(1)  # append constant volume when it is missing in screen
                        else:
                            results_dict[key].append(entries[ind2])

    for key in results_dict.keys():
        results_dict[key] = np.asarray(results_dict[key])

    # between fixes, the first and last step are repeated
    repeated_thermo_steps = np.argwhere(np.diff(results_dict['time_step']) == 0).flatten()
    for key in results_dict.keys():
        if results_dict[key].size > 1:
            if len(results_dict[key]) == len(results_dict['time_step']):
                results_dict[key] = np.delete(results_dict[key], repeated_thermo_steps)

    '''thermo outputs'''
    f = open('tmp.out', "r")
    text = f.read()
    lines = text.split('\n')
    f.close()

    frames = {}
    frame_data = []
    for ind, line in enumerate(lines):
        if line == '\n':
            pass
        elif len(line.split()) == 0:
            pass
        elif line[0] == '#':
            pass
        elif len(line.split()) == 2:
            if len(frame_data) > 0:
                frames[frame_num] = frame_data
            a, b = line.split()
            frame_num = int(a)
            n_mols = int(b)
            frame_data = np.zeros((n_mols, 3))
        else:
            mol_num, temp, kecom, internal = np.asarray(line.split()).astype(float)
            frame_data[int(mol_num) - 1] = temp, kecom, internal

    results_dict['thermo_trajectory'] = np.asarray(list(frames.values()))
    results_dict['thermo_time_step'] = np.asarray(list(frames.keys()))
    # averages over molecules
    results_dict['molwise_mean_temp'] = np.mean(results_dict['thermo_trajectory'][..., 0], axis=1)
    results_dict['molwise_mean_kecom'] = np.mean(results_dict['thermo_trajectory'][..., 1], axis=1)
    results_dict['molwise_mean_internal'] = np.mean(results_dict['thermo_trajectory'][..., 2], axis=1)

    return results_dict


def atom_mass_to_atom_num(atom_mass):
    weights_array = np.asarray(list(ATOM_WEIGHTS.values()))
    atom_index = np.argmin(np.abs(atom_mass - weights_array))
    return atom_index  # weights array indexes from 0


def process_dump(path):
    # pull atom indices straight out of the definition file
    if os.path.exists('new_system.data') or os.path.exists('system.data'):
        num2atomicnum = extract_atom_indexing()
    else:
        assert False, "Missing system data - cannot define atom types"

    file = open(path, 'r')
    lines = file.readlines()
    file.close()

    timestep = None
    n_atoms = None
    frame_outputs = {}  # todo is it really faster to do this as a bunch of dataframes? maybe just a dict or a pl.datadrame would be faster?
    for ind, line in enumerate(tqdm(lines, miniters=len(lines) // 10)):
        if "ITEM: ATOMS" in line:  # atoms header
            headers = line.split()[2:8]
            atom_data = np.zeros((n_atoms, len(headers)))
            for ind2 in range(n_atoms):
                newline = lines[1 + ind + ind2].split()
                try:  # if it's some index, translate it via information from system.data
                    atom_ind = int(newline[2])
                    newline[2] = num2atomicnum[atom_ind]
                except ValueError:
                    # atomic symbol
                    newline[2] = ATOMIC_NUMBERS[newline[2]]

                atom_data[ind2] = np.asarray(newline[:6]).astype(float)  # only want indices and positions

            frame_data = pd.DataFrame(atom_data, columns=headers)
            frame_data.attrs['cell_params'] = cell_params  # add attribute directly to dataframe
            frame_outputs[timestep] = frame_data
        elif "ITEM: TIMESTEP" in line:
            timestep = int(lines[ind + 1])
        elif "ITEM: NUMBER OF ATOMS" in line:
            n_atoms = int(lines[ind + 1])
        elif "ITEM: BOX BOUNDS" in line:
            cell_params = np.stack([
                np.asarray(lines[ind + 1].split()).astype(float),
                np.asarray(lines[ind + 2].split()).astype(float),
                np.asarray(lines[ind + 3].split()).astype(float)
            ])

        else:
            pass

    return frame_outputs


def extract_atom_indexing():
    if os.path.exists('new_system.data'):
        filepath = 'new_system.data'
    else:
        filepath = 'system.data'
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            if 'Mass' in line:
                mass_ind = ind
                break

        for ind, line in enumerate(lines):
            if 'Atoms' in line:
                atom_block_ind = ind
                break

        mass_block = lines[mass_ind + 1:atom_block_ind - 1]
        num2atomicnum = {}
        for line in mass_block:
            if line == '\n':
                pass
            else:
                index, atom_mass = line.split(' ')[0:2]
                num2atomicnum[int(index)] = int(atom_mass_to_atom_num(float(atom_mass)))
    return num2atomicnum


def mol_cluster_dataset_processing(df):
    #print("Cluster cleanup")
    t_fc_array = convert_box_to_cell_params(df['cell_params'])
    df['crystal_fc_transform'] = list(t_fc_array)
    df['polymorph_index'] = [structure2polymorph[sample] for sample in df['structure_identifier']]

    coords_list, atoms_list, density_list, mol_inds_lists, local_density_list, targets_list = [], [], [], [], [], []
    for t_ind in range(len(df)):
        T_fc = df.loc[t_ind]['crystal_fc_transform']
        molind2name_dict = df['molind2name_dict'][t_ind]

        # extract atom & molecule data
        ref_coords = df.loc[t_ind]['atom_coordinates'][0]
        atomic_numbers = np.asarray(df.loc[t_ind]['atom_atomic_numbers'][0], dtype=np.int64)
        num_molecules = len(molind2name_dict)
        mol_inds = np.asarray(df.loc[t_ind]['mol_ind'][0], dtype=np.int64)
        assert num_molecules == len(np.unique(mol_inds))

        # rough estimate of atoms per cubic angstrom
        density = len(atomic_numbers) / np.prod(np.diagonal(df.loc[t_ind]['crystal_fc_transform']))
        periodic = df.loc[t_ind]['cluster_type'] == 'supercell'

        if periodic:  # bring the full structure down to (0,0,0)
            ref_coords -= np.amin(ref_coords, axis=0)

        # default indexing has molecules out of order, but we need sequential data
        molwise_coords, molwise_atoms, mol_inds_list, targets = [], [], [], []
        for ind in np.unique(mol_inds):
            inds = mol_inds == ind
            molwise_coords.append(ref_coords[inds])
            molwise_atoms.append(atomic_numbers[inds])
            mol_inds_list.append(mol_inds[inds])  # also track mol inds, as these will be reordered
            targets.append(df['polymorph_index'].loc[t_ind])  # assumes whole sample is a single polymorph

        if periodic:
            molwise_coords = force_molecules_into_box(T_fc, molwise_coords, num_molecules)

        atom_atom_dists = cdist(np.concatenate(molwise_coords), np.concatenate(molwise_coords))
        atomwise_local_density = np.sum(atom_atom_dists < 6, axis=1)

        ''' look at surface structures
        from ase import Atoms
        from ase.visualize import view
        
        mol = Atoms(positions=np.concatenate(molwise_coords), numbers=np.asarray(7 - (mol_local_atomic_density > (mol_local_atomic_density.max() * 0.75))), cell=T_fc.T)
        view(mol)
        '''
        coords_list.append([np.concatenate(molwise_coords)])
        atoms_list.append([np.concatenate(molwise_atoms)])
        density_list.append([density])
        mol_inds_lists.append([np.concatenate(mol_inds_list)])
        local_density_list.append([atomwise_local_density])
        targets_list.append(targets)

    df['atom_coordinates'] = coords_list
    df['atom_atomic_numbers'] = atoms_list
    df['mol_ind'] = mol_inds_lists
    df['density'] = density_list
    df['local_density'] = local_density_list
    df['polymorph_index'] = targets_list

    vdw_radii = np.nan_to_num(list(VDW_RADII.values())).astype('float')
    atomic_masses = np.nan_to_num(list(ATOM_WEIGHTS.values()))
    electronegativities = np.nan_to_num(list(ELECTRONEGATIVITY.values()))
    atom_groups = np.nan_to_num(list(GROUP.values())).astype('float')
    atom_periods = np.nan_to_num(list(PERIOD.values())).astype('float')

    atom_feats_lists = [[] for _ in range(5)]
    for ind, atoms in enumerate(df['atom_atomic_numbers']):
        atom_feats_lists[0].append(atomic_masses[atoms])
        atom_feats_lists[1].append(vdw_radii[atoms])
        atom_feats_lists[2].append(electronegativities[atoms])
        atom_feats_lists[3].append(atom_groups[atoms])
        atom_feats_lists[4].append(atom_periods[atoms])

    df['atom_mass'] = atom_feats_lists[0]
    df['atom_vdW_radius'] = atom_feats_lists[1]
    df['atom_electronegativity'] = atom_feats_lists[2]
    df['atom_group'] = atom_feats_lists[3]
    df['atom_period'] = atom_feats_lists[4]

    return df


def force_molecules_into_box(T_fc, molwise_coords, num_molecules):
    # by default, we use ux, uy, uz (molecules do not fragment across boundaries, or respect periodicity)
    # so we have to force them back into the box
    # todo since we can have molecules of different sizes, we do this serially rather than in parallel

    # it could be parallelized with padding & torch, but we can do that later
    mol_centroids = np.zeros((num_molecules, 3))
    frac_mol_centroids = np.zeros_like(mol_centroids)
    for ind in range(num_molecules):
        mol_centroids[ind] = molwise_coords[ind].mean(0)
        frac_mol_centroids[ind] = mol_centroids[ind] @ np.linalg.inv(T_fc.T)
    adjustment_fractional_vector = -np.floor(frac_mol_centroids)
    adjustment_cart_vector = adjustment_fractional_vector @ T_fc.T
    for ind in range(num_molecules):
        molwise_coords[ind] += adjustment_cart_vector[ind]

    # check mols are in the box
    mol_centroids = np.zeros((num_molecules, 3))
    frac_mol_centroids = np.zeros_like(mol_centroids)
    for ind in range(num_molecules):
        mol_centroids[ind] = molwise_coords[ind].mean(0)
        frac_mol_centroids[ind] = mol_centroids[ind] @ np.linalg.inv(T_fc.T)
    assert -1e-5 <= frac_mol_centroids.min() and (
            1 + 1e-5) >= frac_mol_centroids.max(), "Molecules are not all in the box!"

    return molwise_coords


#
#
# def old_generate_dataset_from_dumps(dumps_dirs, dataset_path):
#     interim_path = dataset_path.split('.pkl')[0] + '_interim.pkl'
#     if os.path.exists(interim_path):
#         sample_df = pd.read_pickle(interim_path)
#     else:
#         sample_df = pd.DataFrame()
#     for dumps_dir in dumps_dirs:
#         os.chdir(dumps_dir)
#         dump_files = glob.glob(r'*/*.dump', recursive=True) + glob.glob(
#             '*.dump')  # plus any free dumps directly in this dir
#
#         if len(dump_files) == 0:
#             assert False, "No dump files in {}!".format(dumps_dir)
#
#         for path in tqdm(dump_files):  # todo make it so we skip over already-featurized dumps
#             os.chdir(dumps_dir)
#             print(f"Processing dump {path}")
#             p1 = Path(path)
#             path_parts = p1.parts
#             if len(path_parts) > 1:
#                 os.chdir(path_parts[0])
#                 path = path_parts[1]
#
#             run_config = np.load('run_config.npy', allow_pickle=True).item()
#
#             trajectory_dict = process_dump(path)
#             thermo_dict = process_thermo_data()
#
#             for ts, (time_step, vals) in enumerate(tqdm(trajectory_dict.items(), miniters=len(trajectory_dict) // 25)):
#                 if ts % 10 == 0:
#                     new_dict = {'atom_atomic_numbers': [vals['element'].astype(int)],
#                                 'mol_ind': [vals['mol']],
#                                 'atom_coordinates': [np.concatenate((
#                                     np.asarray(vals['xu'])[:, None],
#                                     np.asarray(vals['yu'])[:, None],
#                                     np.asarray(vals['zu'])[:, None]), axis=-1)],
#                                 'time_step': time_step,
#                                 'cell_params': vals.attrs['cell_params'],
#                                 'molecule_num_atoms': [MOLECULE_NUM_ATOMS[run_config['molind2name_dict'][val]] for val in
#                                                        vals['mol']],
#                                 'molecule_mass': [MOLECULE_MASSES[run_config['molind2name_dict'][val]] for val in
#                                                   vals['mol']],
#                                 }
#                     new_dict.update(run_config)
#                     new_dict['temperature'] = thermo_dict['temp'][np.argwhere(thermo_dict['time_step'] == time_step)]
#
#                     new_df = pd.DataFrame()
#                     for key in new_dict.keys():
#                         new_df[key] = [new_dict[key]]
#
#                     new_df = mol_cluster_dataset_processing(new_df)
#                     sample_df = pd.concat([sample_df, new_df])
#
#             sample_df.to_pickle(interim_path)
#
#     os.remove(interim_path)
#     sample_df.reset_index(drop=True, inplace=True)
#     sample_df = flatten_dataframe(sample_df)
#     sample_df.to_pickle(dataset_path)
#     return True


def generate_dataset_from_dumps(dumps_dirs: list,
                                chunks_path: str,
                                steps_per_save: int = 1):
    for dumps_dir in dumps_dirs:
        os.chdir(dumps_dir)
        dump_files = glob.glob(r'*/*.dump', recursive=True) + glob.glob('*.dump')

        if len(dump_files) == 0:
            assert False, "No dump files in {}!".format(dumps_dir)

        for path in tqdm(dump_files):
            os.chdir(dumps_dir)
            p1 = Path(path)
            path_parts = p1.parts
            if len(path_parts) > 1:
                os.chdir(path_parts[0])
                file_path = path_parts[1]
            else:
                file_path = path

            run_dir = os.getcwd()
            run_name = '_'.join(run_dir.replace('\\','/').split('/')[-2:])
            chunk_path = chunks_path + run_name + '.pt'

            if not os.path.exists(chunk_path):
                print(f"Processing dump {path}")

                run_config = np.load('run_config.npy', allow_pickle=True).item()

                trajectory_dict = process_dump(file_path)
                data_list = []
                for ts, (time_step, vals) in enumerate(tqdm(trajectory_dict.items(), miniters=len(trajectory_dict) // 10)):
                    if ts % steps_per_save == 0:  # each X time steps
                        (cell_angles, cell_lengths, cluster_coords, mol_centroids,
                         molecule_num_atoms, polymorph, molecule_types, cluster_type) = process_cluster(
                            run_config, vals)

                        datapoint = CrystalData(  # TODO update workflow with new data types
                            x=torch.tensor(vals['element'], dtype=torch.long),
                            mol_ind=torch.tensor(vals['mol'], dtype=torch.long),
                            pos=cluster_coords,
                            polymorph=polymorph,
                            centroid_pos=mol_centroids - mol_centroids.mean(),
                            time_step=time_step,
                            molecule_num_atoms=molecule_num_atoms,
                            mol_type_ind=molecule_types,
                            cell_lengths=cell_lengths,
                            cell_angles=cell_angles,
                            cluster_type=cluster_type,
                            sg_ind=1,
                            z_prime=0,
                            temperature=run_config['temperature'],
                        )

                        data_list.append(datapoint)

                torch.save(data_list, chunk_path)

    return True


def process_cluster_old_slow(run_config, vals):
    """order atoms according to molecule index"""
    unique_mols = np.unique(vals['mol'])
    assert len(unique_mols) == unique_mols.max(), 'Molecule indices are not continuous'
    mol_inds_tensor = torch.tensor(vals['mol'], dtype=torch.long)
    molwise_atom_index = torch.cat([
        torch.argwhere(mol_inds_tensor == unique).flatten() for unique in unique_mols
    ])
    unique_mol_names = np.unique(list(run_config['molind2name_dict'].values())).tolist()
    molecule_types = torch.tensor([unique_mol_names.index(run_config['molind2name_dict'][val]) for val in unique_mols])
    cluster_coords = torch.stack([
        torch.tensor(vals['xu'], dtype=torch.float32),
        torch.tensor(vals['yu'], dtype=torch.float32),
        torch.tensor(vals['zu'], dtype=torch.float32)]
    ).T[molwise_atom_index]
    T_fc, cell_angles, cell_lengths = convert_box_to_cell_vectors(vals.attrs['cell_params'][None, ...])
    T_fc, cell_angles, cell_lengths = torch.Tensor(T_fc[0]), torch.Tensor(cell_angles[0]), torch.Tensor(
        cell_lengths[0])
    polymorph_index = structure2polymorph[run_config['structure_identifier']]
    rough_density = len(cluster_coords) / torch.prod(torch.diagonal(T_fc))
    cluster_type = run_config['cluster_type']
    if cluster_type == 'supercell':  # if periodic
        cluster_coords -= cluster_coords.mean(0)
    num_mols = int(np.amax(unique_mols))
    molecule_num_atoms = torch.tensor([MOLECULE_NUM_ATOMS[run_config['molind2name_dict'][val]] for val
                                       in unique_mols], dtype=torch.long)
    assert torch.sum(molecule_num_atoms) == len(
        cluster_coords), "Wrong number of atoms for given molecules input"
    'get molecule centroids and coordinates'
    if len(torch.unique(molecule_num_atoms)) == 1:
        uniform_mol_size = True
        molwise_coords = cluster_coords.reshape(num_mols, molecule_num_atoms[0], 3)
        mol_centroids = molwise_coords.mean(1)
    else:
        uniform_mol_size = False
        counter = 0
        molwise_coords = []
        mol_centroids = torch.zeros(num_mols)
        for indi, mol_size in enumerate(molecule_num_atoms):
            molwise_coords.append(
                cluster_coords[counter:counter + int(mol_size)]
            )
            mol_centroids[indi] = molwise_coords[0].mean()
    'force molecules into unit cell'
    if cluster_type == 'supercell':  # periodic structure
        # could be parallelized for massive speed-up with torch rnn-style padding
        frac_mol_centroids = torch.zeros_like(mol_centroids)
        for ind in range(num_mols):
            frac_mol_centroids[ind] = mol_centroids[ind] @ torch.linalg.inv(T_fc.T)

        adjustment_fractional_vector = -torch.floor(frac_mol_centroids)
        adjustment_cart_vector = adjustment_fractional_vector @ T_fc.T
        for ind in range(num_mols):
            molwise_coords[ind] += adjustment_cart_vector[ind]

        # check mols are in the box - expensive!
        mol_centroids = torch.zeros((num_mols, 3))
        frac_mol_centroids = torch.zeros_like(mol_centroids)
        for ind in range(num_mols):
            mol_centroids[ind] = molwise_coords[ind].mean(0)
            frac_mol_centroids[ind] = mol_centroids[ind] @ torch.linalg.inv(T_fc.T)
        assert -1e-5 <= frac_mol_centroids.min() and (
                1 + 1e-5) >= frac_mol_centroids.max(), "Molecules are not all in the box!"
    if uniform_mol_size:
        cluster_coords = molwise_coords.reshape(num_mols * molecule_num_atoms[0], 3)
    else:
        cluster_coords = torch.stack(molwise_coords)

    polymorph = torch.tensor([polymorph_index for _ in range(num_mols)], dtype=torch.long)

    return cell_angles, cell_lengths, cluster_coords, mol_centroids, molecule_num_atoms, polymorph, molecule_types


def process_cluster(run_config, vals):
    """order atoms according to molecule index"""
    'box params'
    T_fc, cell_angles, cell_lengths = convert_box_to_cell_vectors(vals.attrs['cell_params'][None, ...])
    T_fc, cell_angles, cell_lengths = torch.Tensor(T_fc[0]), torch.Tensor(cell_angles[0]), torch.Tensor(
        cell_lengths[0])
    T_cf = torch.linalg.inv(T_fc)

    'molwise information'
    unique_mols = np.unique(vals['mol']).astype(int)
    num_mols = int(np.amax(unique_mols))
    molecule_num_atoms = torch.tensor([MOLECULE_NUM_ATOMS[run_config['molind2name_dict'][val]] for val
                                       in unique_mols], dtype=torch.long)
    assert len(unique_mols) == unique_mols.max(), 'Molecule indices are not continuous'
    unique_mol_names = np.unique(list(run_config['molind2name_dict'].values())).tolist()
    molecule_types = torch.tensor([unique_mol_names.index(run_config['molind2name_dict'][val]) for val in unique_mols])
    mol_inds_tensor = torch.tensor(vals['mol'], dtype=torch.long)

    'polymorph index'
    polymorph_index = structure2polymorph[run_config['structure_identifier']]
    polymorph = torch.tensor([polymorph_index for _ in range(num_mols)], dtype=torch.long)

    'extract coordinates'
    cluster_coords = torch.stack([  # note: atoms are molwise out of order, but that's ok, we have a batch index
        torch.tensor(vals['xu'], dtype=torch.float32),
        torch.tensor(vals['yu'], dtype=torch.float32),
        torch.tensor(vals['zu'], dtype=torch.float32)]
    ).T
    assert torch.sum(molecule_num_atoms) == len(
        cluster_coords), "Wrong number of atoms for given molecules input"

    cluster_type = run_config['cluster_type']

    'force molecules into unit cell, if relevant, and return centroids'
    mol_centroids = scatter(cluster_coords, mol_inds_tensor - 1, reduce='mean', dim=0)
    if cluster_type == 'supercell':  # periodic structure
        frac_mol_centroids = (T_cf @ mol_centroids.T).T

        if torch.abs(frac_mol_centroids - 0.5).max() > 0.5:
            adjustment_fractional_vector = -torch.floor(frac_mol_centroids)
            adjustment_cart_vector = adjustment_fractional_vector @ T_fc.T
            cluster_coords += adjustment_cart_vector[mol_inds_tensor - 1]
            mol_centroids = scatter(cluster_coords, mol_inds_tensor - 1, reduce='mean', dim=0)
            frac_mol_centroids = (T_cf @ mol_centroids.T).T

        assert float(torch.abs(frac_mol_centroids - 0.5).max()) <= 0.50001, "Molecules are not all in the box!"

    return cell_angles, cell_lengths, cluster_coords, mol_centroids, molecule_num_atoms, polymorph, molecule_types, cluster_type

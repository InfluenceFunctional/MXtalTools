import glob
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path

from mxtaltools.common.mol_classifier_utils import convert_box_to_cell_params
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, ATOMIC_NUMBERS, VDW_RADII, ELECTRONEGATIVITY, GROUP, \
    PERIOD
from mxtaltools.constants.mol_classifier_constants import MOLECULE_NUM_ATOMS, MOLECULE_MASSES, structure2polymorph
from mxtaltools.dataset_management.utils import flatten_dataframe


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
    if os.path.exists('new_system.data') or os.path.exists(
            'system.data'):  # pull atom indices straight out of the definition file
        num2atomicnum = extract_atom_indexing()
    else:
        assert False, "Missing system data - cannot define atom types"

    file = open(path, 'r')
    lines = file.readlines()
    file.close()

    timestep = None
    n_atoms = None
    frame_outputs = {}
    for ind, line in enumerate(tqdm(lines, miniters=len(lines) // 100)):
        if "ITEM: TIMESTEP" in line:
            timestep = int(lines[ind + 1])
        elif "ITEM: NUMBER OF ATOMS" in line:
            n_atoms = int(lines[ind + 1])
        elif "ITEM: BOX BOUNDS" in line:
            cell_params = np.stack([
                np.asarray(lines[ind + 1].split()).astype(float),
                np.asarray(lines[ind + 2].split()).astype(float),
                np.asarray(lines[ind + 3].split()).astype(float)
            ]
            )
        elif "ITEM: ATOMS" in line:  # atoms header
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


def generate_dataset_from_dumps(dumps_dirs, dataset_path):
    interim_path = dataset_path.split('.pkl')[0] + '_interim.pkl'
    if os.path.exists(interim_path):
        sample_df = pd.read_pickle(interim_path)
    else:
        sample_df = pd.DataFrame()
    for dumps_dir in dumps_dirs:
        os.chdir(dumps_dir)
        dump_files = glob.glob(r'*/*.dump', recursive=True) + glob.glob(
            '*.dump')  # plus any free dumps directly in this dir

        if len(dump_files) == 0:
            assert False, "No dump files in {}!".format(dumps_dir)

        for path in tqdm(dump_files):  # todo make it so we skip over already-featurized dumps
            os.chdir(dumps_dir)
            print(f"Processing dump {path}")
            p1 = Path(path)
            path_parts = p1.parts
            if len(path_parts) > 1:
                os.chdir(path_parts[0])
                path = path_parts[1]

            run_config = np.load('run_config.npy', allow_pickle=True).item()

            trajectory_dict = process_dump(path)
            thermo_dict = process_thermo_data()

            for ts, (time_step, vals) in enumerate(tqdm(trajectory_dict.items(), miniters=len(trajectory_dict) // 25)):
                if ts % 10 == 0:
                    new_dict = {'atom_atomic_numbers': [vals['element'].astype(int)],
                                'mol_ind': [vals['mol']],
                                'atom_coordinates': [np.concatenate((
                                    np.asarray(vals['xu'])[:, None],
                                    np.asarray(vals['yu'])[:, None],
                                    np.asarray(vals['zu'])[:, None]), axis=-1)],
                                'time_step': time_step,
                                'cell_params': vals.attrs['cell_params'],
                                'molecule_num_atoms': [MOLECULE_NUM_ATOMS[run_config['molind2name_dict'][val]] for val in
                                                       vals['mol']],
                                'molecule_mass': [MOLECULE_MASSES[run_config['molind2name_dict'][val]] for val in
                                                  vals['mol']],
                                }
                    new_dict.update(run_config)
                    new_dict['temperature'] = thermo_dict['temp'][np.argwhere(thermo_dict['time_step'] == time_step)]

                    new_df = pd.DataFrame()
                    for key in new_dict.keys():
                        new_df[key] = [new_dict[key]]

                    new_df = mol_cluster_dataset_processing(new_df)
                    sample_df = pd.concat([sample_df, new_df])

            sample_df.to_pickle(interim_path)

    os.remove(interim_path)
    sample_df.reset_index(drop=True, inplace=True)
    sample_df = flatten_dataframe(sample_df)
    sample_df.to_pickle(dataset_path)
    return True

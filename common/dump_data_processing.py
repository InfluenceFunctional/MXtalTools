import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants.classifier_constants import identifier2form, type2num
from constants.atom_properties import ATOM_WEIGHTS


def atom_mass_to_atom_num(atom_mass):
    weights_array = np.asarray(list(ATOM_WEIGHTS.values()))
    atom_index = np.argmin(np.abs(atom_mass - weights_array))
    return atom_index  # weights array indexes from 0


def process_dump(path):
    if os.path.exists('new_system.data') or os.path.exists(
            'system.data'):  # pull atom indices straight out of the definition file
        num2atomicnum = extract_atom_indexing()

    else:
        if path == 'traj_urea_interface.dump':
            num2atomicnum = {1: 1,
                             2: 8,
                             3: 6,
                             4: 7,
                             5: 7,
                             }

        else:
            from constants.classifier_constants import num2atomicnum

    file = open(path, 'r')
    lines = file.readlines()
    file.close()

    timestep = None
    n_atoms = None
    frame_outputs = {}
    for ind, line in enumerate(tqdm(lines)):
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
                try:
                    atom_ind = int(newline[2])
                    newline[2] = num2atomicnum[atom_ind]
                except ValueError:
                    newline[2] = int(type2num[newline[2]])
                    # newline[2] = num2atomicnum[atom_ind]

                atom_data[ind2] = np.asarray(newline[:6]).astype(float)  # only want indices and positions

            frame_data = pd.DataFrame(atom_data, columns=headers)
            frame_data.attrs['cell_params'] = cell_params  # add attribute directly to dataframe
            frame_outputs[timestep] = frame_data
            '''
            types, counts = np.unique(frame_data['element'], return_counts=True)
            print(counts / counts.min())
            '''
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


def generate_dataset_from_dumps(dumps_dirs, dataset_path):
    sample_df = pd.DataFrame()  # todo add capability to grow this object iteratively in case it crashes midway through generation
    for dumps_dir in dumps_dirs:
        os.chdir(dumps_dir)
        dump_files = glob.glob(r'*/*.dump', recursive=True) + glob.glob(
            '*.dump')  # plus any free dumps directly in this dir

        if len(dump_files) == 0:
            assert False

        for path in tqdm(dump_files):
            print(f"Processing dump {path}")
            form, gap_rate, temperature = process_LAMMPS_run_config(dumps_dir, path)

            trajectory_dict = process_dump(path)

            for ts, (times, vals) in enumerate(tqdm(trajectory_dict.items())):
                if 'xu' in vals.columns:
                    new_dict = {'atom_type': [vals['element'].astype(int)],
                                'mol_ind': [vals['mol']],
                                'coordinates': [np.concatenate((
                                    np.asarray(vals['xu'])[:, None],
                                    np.asarray(vals['yu'])[:, None],
                                    np.asarray(vals['zu'])[:, None]), axis=-1)],
                                'temperature': temperature,
                                'form': form,
                                'time_step': times,
                                'cell_params': vals.attrs['cell_params'],
                                'gap_rate': gap_rate,
                                }
                else:
                    new_dict = {'atom_type': [vals['element'].astype(int)],
                                'mol_ind': [vals['mol']],
                                'coordinates': [np.concatenate((
                                    np.asarray(vals['x'])[:, None],
                                    np.asarray(vals['y'])[:, None],
                                    np.asarray(vals['z'])[:, None]), axis=-1)],
                                'temperature': temperature,
                                'form': form,
                                'time_step': times,
                                'cell_params': vals.attrs['cell_params'],
                                'gap_rate': gap_rate,
                                }

                new_df = pd.DataFrame()
                for key in new_dict.keys():
                    new_df[key] = [new_dict[key]]

                if 'urea' in dumps_dir:
                    types, counts = np.unique(np.concatenate(new_df['atom_type']), return_counts=True)
                    ratios = counts / counts.min()
                    assert (all(ratios == [4, 1, 2, 1])), "Incorrect balance of atom types for urea"
                elif 'nic' in dumps_dir and 'defect' not in dumps_dir:
                    types, counts = np.unique(np.concatenate(new_df['atom_type']), return_counts=True)
                    ratios = counts / counts.min()
                    assert (all(ratios == [6, 6, 2, 1])), "Incorrect balance of atom types for nicotinamide"
                sample_df = pd.concat([sample_df, new_df])

    sample_df.to_pickle(dataset_path)

    return True


def process_LAMMPS_run_config(dumps_dir, path):
    if os.path.exists('run_config.npy'):
        run_config = np.load('run_config.npy', allow_pickle=True).item()
    elif os.path.exists(path.split('\\')[0] + '/' + 'run_config.npy'):
        run_config = np.load(path.split('\\')[0] + '/' + 'run_config.npy', allow_pickle=True).item()
    elif os.path.exists(path.split('/')[0] + '/' + 'run_config.npy'):
        run_config = np.load(path.split('/')[0] + '/' + 'run_config.npy', allow_pickle=True).item()
    elif 'urea' in dumps_dir:
        run_config = {'temperature': float(dumps_dir.split('T')[-1]),
                      'gap_rate': 0}
        if 'liq' in dumps_dir or 'interface' in dumps_dir:
            run_config['structure_identifier'] = 'UREA_Melt'
        else:
            run_config['structure_identifier'] = path.replace('\\', '/').split('/')[0]
    elif 'nicotinamide_liq' in dumps_dir or 'nic_liq' in dumps_dir:
        run_config = {'temperature': 350, 'gap_rate': 0, 'structure_identifier': 'NIC_Melt'}
    elif 'interface' in dumps_dir:
        run_config = {'temperature': 350, 'gap_rate': 0, 'structure_identifier': 'UREA_Melt'}
    else:
        assert False, "Trajectory directory is missing config file"
    if '/' in run_config['structure_identifier']:
        run_config['structure_identifier'] = run_config['structure_identifier'].split('/')[-1]
    temperature = run_config['temperature']
    form = identifier2form[run_config['structure_identifier']]
    gap_rate = run_config['gap_rate']
    return form, gap_rate, temperature

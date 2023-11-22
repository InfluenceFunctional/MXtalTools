import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from classify_lammps_trajs.NICOAM_constants import identifier2form, num2atomicnum, type2num


def process_dump(path):
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
            headers = line.split()[2:-3]
            atom_data = np.zeros((n_atoms, len(headers)))
            for ind2 in range(n_atoms):
                newline = lines[1 + ind + ind2].split()
                try:
                    atom_ind =int(newline[2])
                except ValueError:
                    atom_ind = int(type2num[newline[2]])

                newline[2] = num2atomicnum[atom_ind]

                atom_data[ind2] = np.asarray(newline[:-3]).astype(float)  # cut off velocity elements

            frame_data = pd.DataFrame(atom_data, columns=headers)
            frame_data.attrs['cell_params'] = cell_params  # add attribute directly to dataframe
            frame_outputs[timestep] = frame_data
        else:
            pass

    return frame_outputs


def generate_dataset_from_dumps(dumps_dirs, dataset_path):
    sample_df = pd.DataFrame()
    for dumps_dir in dumps_dirs:
        os.chdir(dumps_dir)
        dump_files = glob.glob(r'*/*.dump', recursive=True) + glob.glob('*.dump')  # plus any free dumps directly in this dir

        if len(dump_files) == 0:
            return False

        for path in tqdm(dump_files):
            print(f"Processing dump {path}")
            if os.path.exists('run_config.npy'):
                run_config = np.load('run_config.npy', allow_pickle=True).item()
            elif os.path.exists(path.split('\\')[0] + '/' + 'run_config.npy'):
                run_config = np.load(path.split('\\')[0] + '/' + 'run_config.npy', allow_pickle=True).item()
            elif os.path.exists(path.split('/')[0] + '/' + 'run_config.npy'):
                run_config = np.load(path.split('/')[0] + '/' + 'run_config.npy', allow_pickle=True).item()
            elif 'urea' in dumps_dir:
                run_config = {'temperature': float(dumps_dir.split('T')[-1]),
                              'gap_rate': 0}
                if 'liq' in dumps_dir:
                    run_config['structure_identifier'] = 'Melt'
                else:
                    run_config['structure_identifier'] = path.split('\\')[0]
            else:
                assert False, "Trajectory directory is missing config file"

            temperature = run_config['temperature']
            form = identifier2form[run_config['structure_identifier']]
            gap_rate = run_config['gap_rate']

            trajectory_dict = process_dump(path)

            for ts, (times, vals) in enumerate(tqdm(trajectory_dict.items())):
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

                sample_df = pd.concat([sample_df, new_df])

        sample_df.to_pickle(dataset_path)

    return None

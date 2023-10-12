import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def process_dump(path):
    file = open(path, 'r')
    lines = file.readlines()
    file.close()

    timestep = None  # todo add cell params extraction
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
                newline[2] = type2num[newline[2]]

                atom_data[ind2] = np.asarray(newline[:-3]).astype(float)  # cut off velocity elements

            frame_data = pd.DataFrame(atom_data, columns=headers)
            frame_data.attrs['cell_params'] = cell_params  # add attribute directly to dataframe
            frame_outputs[timestep] = frame_data
        else:
            pass

    return frame_outputs


def generate_dataset_from_dumps():
    dumps_dirs = [r'D:\crystal_datasets\bulk_crystal_trajs\bulk_crystal_trajs_100', r'D:\crystal_datasets\bulk_crystal_trajs\bulk_crystal_trajs_350']
    sample_df = pd.DataFrame()
    for dumps_dir in dumps_dirs:
        os.chdir(dumps_dir)
        dump_files = glob.glob(r'*/*.dump', recursive=True)

        for path in tqdm(dump_files):
            print(f"Processing dump {path}")
            temperature = int(dumps_dir.split('_')[-1])
            form = int(path.split('\\')[0])
            trajectory_dict = process_dump(path)

            for ts, (times, vals) in enumerate(tqdm(trajectory_dict.items())):
                new_dict = {'atom_type': [vals['id']],
                            'mol_ind': [vals['mol']],
                            'coordinates': [np.concatenate((
                                np.asarray(vals['x'][:, None]),
                                np.asarray(vals['y'][:, None]),
                                np.asarray(vals['z'][:, None])), axis=-1)],
                            'temperature': temperature,
                            'form': form,
                            'time_step': times,
                            'cell_params': vals.attrs['cell_params'],
                            }

                new_df = pd.DataFrame()
                for key in new_dict.keys():
                    new_df[key] = [new_dict[key]]

                sample_df = pd.concat([sample_df, new_df])

    sample_df.to_pickle("../nicotinamide_trajectory_dataset.pkl")


type2num = {
    'Ca1': 1,
    'Ca2': 2,
    'Ca': 3,
    'C': 4,
    'Nb': 5,
    'N': 6,
    'O': 7,
    'Hn': 8,
    'H4': 9,
    'Ha': 10,
}
num2atomicnum = {
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 7,
    6: 7,
    7: 8,
    8: 1,
    9: 1,
    10: 1,
}

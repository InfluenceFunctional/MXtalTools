import glob
import os

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from dataset_management.CrystalData import CrystalData
from dataset_management.utils import get_dataloaders


def process_dump(path):
    file = open(path, 'r')
    lines = file.readlines()
    file.close()

    timestep = None  # todo add cell params extraction
    n_atoms = None
    frame_outputs = {}
    for ind, line in enumerate(lines):
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
                new_dict = {'atom_type': [vals['element'].astype(int)],
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


def convert_box_to_cell_params(box_params):
    """
    ITEM: BOX BOUNDS xy xz yz
    xlo_bound xhi_bound xy
    ylo_bound yhi_bound xz
    zlo_bound zhi_bound yz

    a = xhi-xlo, 0, 0
    b = xy, yhi-ylo, 0
    c = xz, yz, zhi-zlo

    xlo = xlo_bound - MIN(0, xy, xz, xy+xz)
    xhi = xhi_bound - MAX(0, xy, xz, xy+xz)
    ylo = ylo_bound - MIN(0, yz)
    yhi = yhi_bound - MAX(0, yz)
    zlo = zlo_bound
    zhi = zhi_bound
    """
    xlo_bound = box_params[:, 0, 0]
    ylo_bound = box_params[:, 1, 0]
    zlo_bound = box_params[:, 2, 0]
    xhi_bound = box_params[:, 0, 1]
    yhi_bound = box_params[:, 1, 1]
    zhi_bound = box_params[:, 2, 1]
    xy = box_params[:, 0, 2]
    xz = box_params[:, 1, 2]
    yz = box_params[:, 2, 2]

    xlo = xlo_bound - np.stack((np.zeros_like(xy), xy, xz, xy + xz)).T.min(1)
    xhi = xhi_bound - np.stack((np.zeros_like(xy), xy, xz, xy + xz)).T.max(1)
    ylo = ylo_bound - np.stack((np.zeros_like(yz), yz)).T.min(1)
    yhi = yhi_bound - np.stack((np.zeros_like(yz), yz)).T.max(1)
    zlo = zlo_bound
    zhi = zhi_bound

    a = np.asarray([xhi - xlo, np.zeros_like(xhi), np.zeros_like(xhi)]).T
    b = np.asarray([xy, yhi - ylo, np.zeros_like(xy)]).T
    c = np.asarray([xz, yz, zhi - zlo]).T

    return a, b, c


def collect_to_traj_dataloaders(dataset_path):
    dataset = pd.read_pickle(dataset_path)
    dataset = dataset.reset_index().drop(columns='index')  # reindexing is crucial here

    forms = np.unique(dataset['form'])
    forms2tgt = {form: i for i, form in enumerate(forms)}
    targets = np.asarray(
        [forms2tgt[form] for form in dataset['form']]
    )

    a, b, c = convert_box_to_cell_params(np.stack(dataset['cell_params']))

    T_fc_list = np.zeros((len(a), 3, 3))
    for i in range(len(T_fc_list)):
        T_fc_list[i] = np.stack((a[i], b[i], c[i]))

    print('Generating training datapoints')
    datapoints = []
    for i in tqdm(range(len(dataset))):
        datapoints.append(
            CrystalData(
                x=torch.tensor(dataset.loc[i]['atom_type'][0], dtype=torch.long),
                pos=torch.Tensor(dataset.loc[i]['coordinates'][0]),
                y=torch.tensor(targets[i], dtype=torch.long),
                tracking=np.asarray(dataset.loc[i, ['temperature', 'time_step']]),
                T_fc=torch.Tensor(T_fc_list[i]),
                asym_unit_handedness=torch.ones(1),
                symmetry_operators=torch.ones(1),
            )
        )
    del dataset
    return get_dataloaders(datapoints, machine='local', batch_size=1, test_fraction=0.2)

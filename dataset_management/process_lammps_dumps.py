import os
import pandas as pd
import numpy as np
import torch
import warnings
from dataset_management.CrystalData import CrystalData
from dataset_management.lammps_traj_utils import generate_dataset_from_dumps, num2atomicnum, type2num

warnings.filterwarnings("ignore", category=FutureWarning)  # ignore numpy error

if __name__ == "__main__":
    dataset_path = r'D:\crystal_datasets\bulk_crystal_trajs\nicotinamide_trajectory_dataset.pkl'
    if not os.path.exists(dataset_path):
        generate_dataset_from_dumps()

    dataset = pd.read_pickle(dataset_path)
    dataset = dataset.reset_index().drop(columns='index')  # reindexing is crucial here

    conv_field = 10
    inside_cluster_radius = 20

    forms = np.unique(dataset['form'])
    forms2tgt = {form: i for i, form in enumerate(forms)}
    targets = np.asarray(
        [forms2tgt[form] for form in dataset['form']]
    )
    '''
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
    '''

    # T_fc_list = torch.cat((a,b,c),axis = 0)

    T_fc_list = 1

    datapoints = []
    for i in range(len(dataset)):
        datapoints.append(
            CrystalData(
                x=torch.Tensor(dataset.loc[i]['atom_type']),
                pos=torch.Tensor(dataset.loc[i]['coordinates']),
                y=targets[i],
                tracking=np.asarray(dataset.loc[i, ['temperature', 'time_step']]),
                T_fc=T_fc_list[i]
            )
        )

'''
Modelling strategy
x> get atomwise data and cell params
-> periodize to a predefined cutoff (min cluster size + conv field)
-> identify inside/outside cutoff atoms
-> identify surface atoms
-> convolve to atomwise loss
'''

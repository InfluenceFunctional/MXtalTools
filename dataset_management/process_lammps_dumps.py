import os

import warnings
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F

from dataset_management.lammps_traj_utils import generate_dataset_from_dumps, num2atomicnum, type2num, convert_box_to_cell_params, collect_to_traj_dataloaders
from dataset_management.utils import init_classifier

warnings.filterwarnings("ignore", category=FutureWarning)  # ignore numpy error

num_convs = 2
conv_cutoff = 6
inside_region_radius = 10
total_cluster_radius = inside_region_radius + num_convs * conv_cutoff

device = 'cuda'

if __name__ == "__main__":
    dataset_path = r'D:\crystal_datasets\bulk_crystal_trajs\nicotinamide_trajectory_dataset.pkl'
    if not os.path.exists(dataset_path):
        generate_dataset_from_dumps()

    train_loader, test_loader = collect_to_traj_dataloaders(dataset_path)

    classifier = init_classifier(conv_cutoff, num_convs)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    classifier.train(True)
    classifier.to(device)
    loss_record = []
    for epoch in range(10):
        for step, data in enumerate(tqdm(train_loader)):
            sample = data.to(device)

            '''identify inside and outside'''
            pos = sample.pos
            pos -= pos.mean(0)
            distmat = torch.cdist(torch.zeros((1, 3)).to(device), pos)[0, :]
            outside_inds = torch.argwhere((distmat < total_cluster_radius) * (distmat > inside_region_radius))[:, 0]
            inside_inds = torch.argwhere(distmat < inside_region_radius)[:, 0]

            ref_inds = torch.zeros(len(pos)).to(device)
            ref_inds[inside_inds] = 0
            ref_inds[outside_inds] = 1
            # covolution over these is wrong -
            # we actually just want to readout the middle but convolve everything
            # also need to add surfaces - maybe take clusters as-is and move centroid around?

            sample.aux_ind = ref_inds

            output = classifier(sample)

            loss = F.cross_entropy(output, sample.y.repeat(len(output)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record.append(loss.cpu().detach().numpy())

        print(np.mean(np.array(loss_record)))
        import plotly.graph_objects as go
        from scipy.ndimage.filters import gaussian_filter1d

        fig = go.Figure(go.Scattergl(y=gaussian_filter1d(np.asarray(loss_record), sigma=20))).show()

    aa = 1

'''
Modelling strategy
x> get atomwise data and cell params
-> periodize to a predefined cutoff (min cluster size + conv field)
-> identify inside/outside cutoff atoms
-> identify surface atoms
-> convolve to atomwise loss
'''

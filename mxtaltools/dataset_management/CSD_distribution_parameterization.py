"""
dataset cell parameter analysis
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader.dataloader import Collater

from mxtaltools.common.geometry_calculations import rotvec2sph, batch_molecule_vdW_volume, \
    batch_compute_normed_cell_vectors
from mxtaltools.constants.asymmetric_units import raw_asym_unit_dict
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.constants.space_group_info import LATTICE_TYPE

dataset_path = Path('D:/crystal_datasets/CSD_dataset.pt')
dataset = torch.load(dataset_path)
dataset = dataset.to_data_list()

# dataset_path = Path('D:/crystal_datasets/test_CSD_dataset.pt')
# dataset = torch.load(dataset_path)


def prep_good_dataset(dataset):
    collater = Collater(0, 0)
    # filter Z'=1
    good_dataset = [elem for elem in dataset if elem.z_prime == 1]
    print(f"Z' filter took {len(dataset) - len(good_dataset)}")
    dataset = good_dataset
    # filter diffuse
    red_vol = torch.tensor([elem.reduced_volume for elem in good_dataset])
    vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()))

    col_dataset = collater(dataset)

    molecule_volumes = batch_molecule_vdW_volume(col_dataset.x.flatten(),
                                                 col_dataset.pos,
                                                 col_dataset.batch,
                                                 col_dataset.num_graphs,
                                                 vdw_radii_tensor)

    packing_coeff = molecule_volumes / red_vol
    good_dataset = [elem for i, elem in enumerate(dataset) if 0.5 < packing_coeff[i] < 0.9]
    print(f"diffuse filter took {len(dataset) - len(good_dataset)}")
    dataset = good_dataset
    dataset = collater(dataset)

    molecule_volumes = batch_molecule_vdW_volume(dataset.x.flatten(),
                                                 dataset.pos,
                                                 dataset.batch,
                                                 dataset.num_graphs,
                                                 vdw_radii_tensor)

    packing_coeff = molecule_volumes / dataset.reduced_volume

    dataset.mol_volume = molecule_volumes
    dataset.packing_coeff = packing_coeff
    return dataset


dataset = prep_good_dataset(dataset)

'cell lengths'
normed_cell_vecs = batch_compute_normed_cell_vectors(dataset)
length_stds = normed_cell_vecs.std(0)
length_means = normed_cell_vecs.mean(0)
lengths_dist = normed_cell_vecs

shift = lambda x: F.softplus(x - 0.01, beta=5) + 0.01
shifted_normed_cell_lengths = shift(normed_cell_vecs)

s_length_stds = shifted_normed_cell_lengths.std(0)
s_length_means = shifted_normed_cell_lengths.mean(0)
s_cov_mat = torch.cov(shifted_normed_cell_lengths.T)

print(s_length_means)
print(s_length_stds)
print(s_cov_mat)

'cell angles'
lattices = [LATTICE_TYPE[int(dataset.sg_ind[ind])] for ind in range(len(dataset))]
css = np.unique(lattices)
angles = dataset.cell_angles
angles_stats = {}
for csi, cs in enumerate(css):
    good_inds = np.argwhere([lattice == cs for lattice in lattices]).flatten()
    angles_stats[cs] = [angles[good_inds, :].mean(0), angles[good_inds, :].std(0)]

angles_stds = angles.std(0)
angles_means = angles.mean(0)

print(angles_means)
print(angles_stds)
print(angles_stats)

'fractional positions (we take this as a uniform distribution'

positions = dataset.pose_params0[:, :3].clone()
for ind in range(len(positions)):
    try:
        positions[ind] = positions[ind] / torch.Tensor(raw_asym_unit_dict[str(int(dataset.sg_ind[ind]))])
    except:
        pass

pos_stds = positions.std(0)
pos_means = positions.mean(0)
pos_dist = positions

'mol orientations (these are just random)'
random_vectors = torch.randn(size=(10000, 3))
norms = random_vectors.norm(dim=1)
applied_norms = torch.rand(10000) * 2 * torch.pi
random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]
random_vectors[:, 2] = torch.abs(random_vectors[:, 2])
sph_vectors = rotvec2sph(random_vectors)

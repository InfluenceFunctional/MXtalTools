#!/usr/bin/env python

import numpy as np
from ase.io import write
from ase.visualize import view
from ase import Atoms
from pyxtal import symmetry
from crystal_building.utils_np import fractional_transform_np
from crystal_building.utils import build_unit_cell, ref_to_supercell
from constants.space_group_info import SYM_OPS
from crystal_building.coordinate_transformations import coor_trans_matrix

import torch

# Create the ZnPc skeleton (asymmetric cell first)
a = 19.274
b = 4.8538
c = 14.553

alpha = np.pi / 2
gamma = np.pi / 2
beta = 120.48 * np.pi / 180

T_fc, cell_vol = coor_trans_matrix('f_to_c', [a, b, c], [alpha, beta, gamma], return_vol=True)
T_cf = np.linalg.inv(T_fc)

misc = ['Zn', np.array([0, 0, 0]),
        'N', np.array([-783, -5176, 716]) * 1e-4,
        'N', np.array([345, -2027, 1345]) * 1e-4,
        'N', np.array([1608, 268, 2524]) * 1e-4,
        'N', np.array([992, 2242, 750]) * 1e-4,
        'C', np.array([-85, -4097, 1460]) * 1e-4,
        'C', np.array([331, -4985, 2564]) * 1e-4,
        'C', np.array([142, -6975, 3092]) * 1e-4,
        'C', np.array([673, -7286, 4178]) * 1e-4,
        'C', np.array([1370, -5688, 4721]) * 1e-4,
        'C', np.array([1563, -3708, 4191]) * 1e-4,
        'C', np.array([1030, -3392, 3101]) * 1e-4,
        'C', np.array([1028, -1540, 2308]) * 1e-4,
        'C', np.array([1590, 1980, 1798]) * 1e-4,
        'C', np.array([2228, 3946, 2018]) * 1e-4,
        'C', np.array([2969, 4514, 2926]) * 1e-4,
        'C', np.array([3445, 6538, 2846]) * 1e-4,
        'C', np.array([3194, 7969, 1897]) * 1e-4,
        'C', np.array([2455, 7426, 989]) * 1e-4,
        'C', np.array([1979, 5384, 1064]) * 1e-4,
        'C', np.array([1198, 4276, 286]) * 1e-4,
        ]

species = [specie for i, specie in enumerate(misc) if i % 2 == 0]
fractional_coords = np.asarray([specie for i, specie in enumerate(misc) if not i % 2 == 0])

symmetry_operations = SYM_OPS[14]  # I believe P21/a should be the same as P21/c (group 14) - can check here https://www.lpl.arizona.edu/PMRG/sites/lpl.arizona.edu.PMRG/files/ITC-Vol.A%20%282005%29%28ISBN%200792365909%29.pdf

# alternatively try the 2a special postiion
fractional_coords = np.concatenate((fractional_coords, -fractional_coords[1:]), axis=0)

symmetry_operations = [symmetry.Group(14).wyckoffs_organized[1][3][i].affine_matrix for i in range(2)]

for i in range(len(symmetry_operations)):  # for some reason these have no rotational component
    symmetry_operations[i][:3, :3] += np.eye(3)

# manual correction
symmetry_operations[1][0, 3] = 0.5
symmetry_operations[1][1, 3] = 0.5
symmetry_operations[1][2, 3] = 0

mol_species = species + species[1:]
crystal_species = mol_species * 2

unit_cell = build_unit_cell(torch.tensor([len(symmetry_operations)], dtype=int),
                            [torch.Tensor(fractional_transform_np(fractional_coords, T_fc))],
                            torch.Tensor(T_fc)[None, :, :],
                            torch.Tensor(T_cf)[None, :, :],
                            [torch.tensor(symmetry_operations, dtype=torch.float32)]
                            )[0].detach().numpy()

frac_centroids = fractional_transform_np(unit_cell.mean(1), T_cf)

crystal = Atoms(symbols=crystal_species, positions=unit_cell.reshape(82, 3), cell=T_fc.T)
view(crystal)

coords, atoms, inds, copies = ref_to_supercell([unit_cell], torch.Tensor(T_fc).permute(1, 0)[None, :, :], torch.Tensor(T_fc)[None, :, :],
                                               [torch.ones(41)], [2], supercell_scale=5, cutoff=3, pare_to_convolution_cluster=True
                                               )

supercell = Atoms(symbols=mol_species * copies, positions=coords[0].cpu().detach().numpy(), cell=T_fc.T)
view(supercell)

aa = 1

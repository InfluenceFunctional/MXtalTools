import torch
import numpy as np
from utils import coor_trans_matrix_torch
from dataset_management.random_crystal_builder import (randomize_molecule_position_and_orientation_torch,build_random_crystal_torch,ref_to_supercell_torch, clean_cell_output)


def parallel_differentiable_cell_builder(x,y,pos, num_graphs,batch, dataDims, cell_sample, sg_number_ind, atom_weights, sym_ops_list, override_position=None, override_orientation=None, override_cell_length=None, override_cell_angle=None):
    '''
    convert cell parameters to reference cell
    convert reference cell to 3x3 supercell
    all using differentiable torch functions
    '''
    volumes = []
    z_values = []
    sg_numbers = [int(y[2][i][sg_number_ind]) for i in range(num_graphs)]

    cell_lengths, cell_angles, rand_position, rand_rotation = cell_sample.split(3, 1)

    cell_lengths, cell_angles, rand_position, rand_rotation = clean_cell_output(
        cell_lengths, cell_angles, rand_position, rand_rotation, None, dataDims, enforce_crystal_system=False)

    if override_position is not None:
        rand_position = torch.tensor(override_position).to(rand_position.device)
    if override_orientation is not None:
        rand_rotation = torch.tensor(override_orientation).to(rand_rotation.device)
    if override_cell_length is not None:
        cell_lengths = torch.tensor(override_cell_length).to(rand_position.device)
    if override_cell_angle is not None:
        cell_angles = torch.tensor(override_cell_angle).to(rand_rotation.device)

    for i in range(num_graphs):
        atoms = x[batch == i]
        atomic_numbers = atoms[:, 0]
        # heavy_atom_inds = torch.argwhere(atomic_numbers > 1)[:, 0]
        # assert torch.sum(atomic_numbers == 1) == 0, 'hydrogens in supercell_dataset!'
        # atoms = atoms_i[heavy_atom_inds]
        coords = pos[batch == i, :]
        weights = torch.tensor([atom_weights[int(number)] for number in atomic_numbers]).to(coords.device)

        sym_ops = torch.tensor(sym_ops_list[sg_numbers[i]], dtype=coords.dtype).to(coords.device)
        z_value = len(sym_ops)  # number of molecules in the reference cell
        z_values.append(z_value)

        T_fc, vol = coor_trans_matrix_torch('f_to_c', cell_lengths[i], cell_angles[i], return_vol=True)
        T_fc = T_fc.to(coords.device)
        T_cf = torch.linalg.inv(T_fc)  # faster #coor_trans_matrix_torch('c_to_f', cell_lengths[i], cell_angles[i]).to(config.device)
        cell_vectors = torch.inner(T_fc, torch.eye(3).to(coords.device)).T  # T_fc.dot(torch.eye(3)).T
        volumes.append(vol)

        random_coords = randomize_molecule_position_and_orientation_torch(
            coords, weights, T_fc, sym_ops,
            set_position=rand_position[i], set_rotation=rand_rotation[i])

        reference_cell = build_random_crystal_torch(T_cf, T_fc, random_coords, sym_ops, z_value)

        supercell_atoms, supercell_coords = ref_to_supercell_torch(reference_cell, z_value, atoms, cell_vectors)

        supercell_batch = torch.ones(len(supercell_atoms)).int() * i

        # append supercell info to the data class #
        if i == 0:
            new_x = supercell_atoms
            new_coords = supercell_coords
            new_batch = supercell_batch
            new_ptr = torch.zeros(num_graphs)
        else:
            new_x = torch.cat((new_x, supercell_atoms), dim=0)
            new_coords = torch.cat((new_coords, supercell_coords), dim=0)
            new_batch = torch.cat((new_batch, supercell_batch))
            new_ptr[i] = new_ptr[-1] + len(new_x)


    return new_x, new_coords, new_batch, new_ptr
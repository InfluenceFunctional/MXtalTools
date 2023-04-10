import numpy as np
from models.utils import enforce_1d_bound
from common.geometry_calculations import compute_principal_axes_np, single_molecule_principal_axes, batch_molecule_principal_axes, compute_Ip_handedness
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import sys

# from pymatgen.symmetry import analyzer
# from pymatgen.core import (structure, lattice)

asym_unit_dict = {  # https://www.lpl.arizona.edu/PMRG/sites/lpl.arizona.edu.PMRG/files/ITC-Vol.A%20%282005%29%28ISBN%200792365909%29.pdf
    '1': [1, 1, 1],  # P1
    '2': [0.5, 1, 1],  # P-1
    '3': [1, 1, 0.5],  # P2
    '4': [1, 1, 0.5],  # P21
    '5': [0.5, 0.5, 1],  # C2
    '6': [1, 0.5, 1],  # Pm
    '7': [1, 0.5, 1],  # Pc
    '8': [1, 0.25, 1],  # Cm
    '9': [1, 0.25, 1],  # Cc
    '10': [0.5, 0.5, 1],  # P2/m
    '11': [1, 0.25, 1],  # P21/m
    '12': [0.5, 0.25, 1],  # C2/m
    '13': [0.5, 1, 0.5],  # P2/c
    '14': [1, 0.25, 1],  # P21/c
    '15': [0.5, 0.5, 0.5],  # C2/c
    '16': [0.5, 0.5, 1],  # P222
    '17': [0.5, 0.5, 1],  # P2221
    '18': [0.5, 0.5, 1],  # P21212
    '19': [0.5, 0.5, 1],  # P212121
    '20': [0.5, 0.5, 0.5],  # C2221
    '21': [0.25, 0.5, 1],  # C222
    '22': [0.25, 0.25, 1],  # F222
    '23': [0.5, 0.5, 0.5],  # I222
    '24': [0.5, 0.5, 0.5],  # I212121
    '25': [0.5, 0.5, 1],  # Pmm2
    '26': [0.5, 0.5, 1],  # Pmc21
    '27': [0.5, 0.5, 1],  # Pcc2
    '28': [0.25, 1, 1],  # Pma2
    '29': [0.25, 1, 1],  # Pca21
    '30': [0.5, 1, 0.5],  # Pnc2
    '31': [0.5, 0.5, 1],  # Pmn21
    '32': [0.5, 0.5, 1],  # Pba2
    '33': [0.5, 0.5, 1],  # Pna21
    '34': [0.5, 0.5, 1],  # Pnn2
    '35': [0.25, 0.5, 1],  # Cmm2
    '36': [0.5, 0.5, 0.5],  # Cmc21
    '37': [0.25, 0.5, 1],  # Ccc2
    '38': [0.5, 0.5, 0.5],  # Amm2
    '39': [0.5, 0.25, 1],  # Aem2
    '40': [0.25, 0.5, 1],  # Ama2
    '41': [0.5, 0.5, 0.5],  # Aea2
    '42': [0.25, 0.25, 1],  # Fmm2
    '43': [0.25, 0.25, 1],  # Fdd2
    '44': [0.5, 0.5, 0.5],  # Imm2
    '45': [0.5, 0.5, 0.5],  # Iba2
    '46': [0.25, 1, 0.5],  # Ima2
    '47': [0.5, 0.5, 0.5],  # Pmmm
    '48': [0.25, 0.5, 1],  # Pnnn
    '49': [0.5, 0.5, 0.5],  # Pccm
    '50': [0.5, 0.5, 0.5],  # Pban
    '51': [0.25, 0.5, 1],  # Pmma
    '52': [1, 0.25, 0.5],  # Pnna
    '53': [0.5, 1, 0.25],  # Pmna
    '54': [0.5, 0.5, 0.5],  # Pcca
    '55': [0.5, 0.5, 0.5],  # Pbam
    '56': [0.25, 1, 0.5],  # Pccn
    '57': [0.5, 1, 0.25],  # Pbcm
    '58': [0.5, 0.5, 0.5],  # Pnnm
    '59': [0.5, 0.5, 0.5],  # Pmmn
    '60': [0.5, 0.5, 0.5],  # Pbcn
    '61': [0.5, 0.5, 0.5],  # Pbca
    '62': [0.5, 0.25, 1],  # Pnma
    '63': [0.5, 0.5, 0.25],  # Cmcm
    '64': [0.25, 0.5, 0.5],  # Cmce
    '65': [0.25, 0.5, 0.5],  # Cmmm
    '66': [0.25, 0.5, 0.5],  # Cccm
    '67': [0.5, 0.25, 0.5],  # Cmme
    '68': [0.25, 0.5, 0.5],  # Ccce
    '69': [0.25, 0.25, 0.5],  # Fmmm
    '70': [0.125, 0.25, 1],  # Fddd
    '71': [0.25, 0.5, 0.5],  # Immm
    '72': [0.25, 0.5, 0.5],  # Ibam
    '73': [0.25, 0.5, 0.5],  # Ibca
    '74': [0.25, 0.25, 1],  # Imma
    '75': [0.5, 0.5, 1],  # P4
}



def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    todo this is not my original code, though I have rewritten it for torch
    Ripped out of torch3d source code
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        rmat = torch.Tensor(((one, zero, zero), (zero, cos, -sin), (zero, sin, cos)), device=angle.device)
    elif axis == "Y":
        rmat = torch.Tensor(((cos, zero, sin), (zero, one, zero), (-sin, zero, cos)), device=angle.device)
    elif axis == "Z":
        rmat = torch.Tensor(((cos, -sin, zero), (sin, cos, zero), (zero, zero, one)), device=angle.device)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return rmat
    #
    # if axis == "X":
    #     R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    # elif axis == "Y":
    #     R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    # elif axis == "Z":
    #     R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    # else:
    #     raise ValueError("letter must be either X, Y or Z.")
    #
    # return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_XYZ_rotation_matrix(angles):
    axes = ['X', 'Y', 'Z']
    compose_rotations = [axis_angle_rotation(axis, angle) for axis, angle in zip(axes, angles)]
    rotation_matrix = torch.matmul(torch.matmul(compose_rotations[0], compose_rotations[1]), compose_rotations[2])

    return rotation_matrix


def coor_trans_matrix(cell_lengths, cell_angles):
    '''
    compute f->c and c->f transforms as well as cell volume in a vectorized, differentiable way
    '''
    cos_a = torch.cos(cell_angles)
    sin_a = torch.sin(cell_angles)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[:, 0] ** 2 - cos_a[:, 1] ** 2 - cos_a[:, 2] ** 2 + 2.0 * cos_a[:, 0] * cos_a[:, 1] * cos_a[:, 2]

    vol = torch.sign(val) * torch.prod(cell_lengths, dim=1) * torch.sqrt(torch.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    T_fc_list = torch.zeros((len(cell_lengths), 3, 3), device=cell_lengths.device, dtype=cell_lengths.dtype)
    T_cf_list = torch.zeros((len(cell_lengths), 3, 3), device=cell_lengths.device, dtype=cell_lengths.dtype)

    ''' Converting from cartesian to fractional '''
    T_cf_list[:, 0, 0] = 1.0 / cell_lengths[:, 0]
    T_cf_list[:, 0, 1] = -cos_a[:, 2] / cell_lengths[:, 0] / sin_a[:, 2]
    T_cf_list[:, 0, 2] = cell_lengths[:, 1] * cell_lengths[:, 2] * (cos_a[:, 0] * cos_a[:, 2] - cos_a[:, 1]) / vol / sin_a[:, 2]
    T_cf_list[:, 1, 1] = 1.0 / cell_lengths[:, 1] / sin_a[:, 2]
    T_cf_list[:, 1, 2] = cell_lengths[:, 0] * cell_lengths[:, 2] * (cos_a[:, 1] * cos_a[:, 2] - cos_a[:, 0]) / vol / sin_a[:, 2]
    T_cf_list[:, 2, 2] = cell_lengths[:, 0] * cell_lengths[:, 1] * sin_a[:, 2] / vol
    ''' Converting from fractional to cartesian '''
    T_fc_list[:, 0, 0] = cell_lengths[:, 0]
    T_fc_list[:, 0, 1] = cell_lengths[:, 1] * cos_a[:, 2]
    T_fc_list[:, 0, 2] = cell_lengths[:, 2] * cos_a[:, 1]
    T_fc_list[:, 1, 1] = cell_lengths[:, 1] * sin_a[:, 2]
    T_fc_list[:, 1, 2] = cell_lengths[:, 2] * (cos_a[:, 0] - cos_a[:, 1] * cos_a[:, 2]) / sin_a[:, 2]
    T_fc_list[:, 2, 2] = vol / cell_lengths[:, 0] / cell_lengths[:, 1] / sin_a[:, 2]

    return T_fc_list, T_cf_list, torch.abs(vol)


def cell_vectors(T_fc_list):
    '''
    convert fractional vectors (1,1,1) into cartesian cell vectors (a,b,c)
    '''
    eyevec = torch.tile(torch.eye(3, device=T_fc_list.device), (len(T_fc_list), 1, 1))
    return torch.matmul(T_fc_list, eyevec).permute(0, 2, 1)


def ref_to_supercell(reference_cell_list, cell_vector_list, T_fc_list,
                     atoms_list, z_values, supercell_scale=5, cutoff=5,
                     sorted_fractional_translations=None):
    '''
    1) generate fractional translations for full supercell
    for each sample
    2) generate cartesian coordinates
    3) identify canonical conformer, inside inds, outside inds
    4) kick out molecules which are outside
    '''
    if sorted_fractional_translations is None:
        n_cells = (2 * supercell_scale + 1) ** 3
        fractional_translations = torch.zeros((n_cells, 3), device=T_fc_list.device)  # initialize the translations in fractional coords
        i = 0
        for xx in range(-supercell_scale, supercell_scale + 1):
            for yy in range(-supercell_scale, supercell_scale + 1):
                for zz in range(-supercell_scale, supercell_scale + 1):
                    fractional_translations[i] = torch.tensor((xx, yy, zz), device=T_fc_list.device)
                    i += 1
        sorted_fractional_translations = fractional_translations[torch.argsort(fractional_translations.abs().sum(1))]
    else:
        n_cells = len(sorted_fractional_translations)

    device = T_fc_list.device
    supercell_coords_list = []
    supercell_atoms_list = []
    ref_mol_inds_list = []
    copies = []
    z_values = [len(ref_list) for ref_list in reference_cell_list]  # todo delete once generator method is separated
    for i, (ref_cell, unit_cell_vectors, atoms, z_value) in enumerate(zip(reference_cell_list, cell_vector_list, atoms_list, z_values)):
        if type(ref_cell) == np.ndarray:
            ref_cell = torch.tensor(ref_cell, device=device)

        mol_n_atoms = len(atoms)
        supercell_coords = ref_cell.clone().reshape(z_value * ref_cell.shape[1], 3).tile(n_cells, 1)  # duplicate over XxXxX supercell
        cart_translations_i = torch.mul(unit_cell_vectors.tile(n_cells, 1), sorted_fractional_translations.reshape(n_cells * 3, 1))  # 3 cell vectors
        # cart_translations = torch.stack(cart_translations_i.split(3, dim=0), dim=0).sum(1)
        cart_translations = cart_translations_i.reshape(n_cells, 3, 3).sum(1)  # faster

        full_supercell_coords = supercell_coords + torch.repeat_interleave(cart_translations, ref_cell.shape[1] * ref_cell.shape[0], dim=0)  # add translations throughout

        # index atoms within the 'canonical' conformer, which is always indexed first
        # TODO canonical conformer is not indexed first in real cells - shouldn't actually impact morphology but would be nice to clean up
        in_mol_inds = torch.arange(mol_n_atoms)
        molwise_supercell_coords = full_supercell_coords.reshape(n_cells * z_value, mol_n_atoms, 3)
        ref_mol_centroid = molwise_supercell_coords[0].mean(0)  # first is always the canonical conformer
        all_mol_centroids = torch.mean(molwise_supercell_coords, dim=1)  # centroids for all molecules in the supercell
        mol_centroid_dists = torch.cdist(ref_mol_centroid[None, :], all_mol_centroids, p=2)[0]
        ref_mol_radius = torch.max(torch.cdist(ref_mol_centroid[None, :], full_supercell_coords[in_mol_inds]))

        successful_gconv = False
        extra_cutoff = 0
        while successful_gconv == False:
            # ignore atoms which are more than mol_radius + conv_cutoff + buffer
            convolve_mol_inds = torch.where((mol_centroid_dists <= (2 * ref_mol_radius + cutoff + extra_cutoff + 0.1)))[0]

            if len(convolve_mol_inds) <= mol_n_atoms:  # if the crystal is too diffuse / there are no molecules close enough to convolve with, we open the window and try again
                extra_cutoff += 0.5
            else:
                successful_gconv = True

        ref_mol_inds = torch.ones(len(convolve_mol_inds) * mol_n_atoms, dtype=int, device=device)  # only index molecules which will be kept
        ref_mol_inds[in_mol_inds] = 0

        assert len(convolve_mol_inds) > mol_n_atoms

        convolve_atom_inds = (torch.arange(mol_n_atoms, device=device)[:, None] + convolve_mol_inds * mol_n_atoms).T.reshape(len(convolve_mol_inds) * mol_n_atoms)  # looks complicated but it's fast

        supercell_coords_list.append(full_supercell_coords[convolve_atom_inds])  # only the relevant molecules are now kept
        supercell_atoms = atoms.repeat(len(convolve_mol_inds), 1)  # take the number of kept molecules only

        supercell_atoms_list.append(supercell_atoms)
        ref_mol_inds_list.append(ref_mol_inds)

        copies.append(len(convolve_mol_inds))

    n_copies = torch.tensor(copies, dtype=torch.int32)

    return supercell_coords_list, supercell_atoms_list, ref_mol_inds_list, n_copies


def update_supercell_data(supercell_data, supercell_atoms_list, supercell_coords_list, ref_mol_inds_list, reference_cell_list):
    device = supercell_data.x.device
    for i in range(supercell_data.num_graphs):
        if i == 0:
            new_batch = torch.ones(len(supercell_atoms_list[i]), device=device).int() * i
            new_ptr = torch.zeros(supercell_data.num_graphs + 1, device=device)
            new_ptr[1] = len(supercell_coords_list[0])
        else:
            new_batch = torch.cat((new_batch, torch.ones(len(supercell_atoms_list[i]), device=device).int() * i))
            new_ptr[i + 1] = new_ptr[i] + len(supercell_coords_list[i])

    # update dataloader with cell info
    supercell_data.x = torch.cat(supercell_atoms_list).type(dtype=torch.float32)
    supercell_data.pos = torch.cat(supercell_coords_list).type(dtype=torch.float32)
    supercell_data.batch = new_batch.type(dtype=torch.int64)
    supercell_data.ptr = new_ptr.type(dtype=torch.int64)
    supercell_data.aux_ind = torch.cat(ref_mol_inds_list).type(dtype=torch.int)
    supercell_data.ref_cell_pos = reference_cell_list

    return supercell_data


def clean_cell_output(cell_lengths, cell_angles, mol_position, mol_rotation, lattices, dataDims,
                      enforce_crystal_system=False, rotation_type='cartesian rotvec', return_transforms=False,
                      standardized_sample=True):
    '''
    assert physically meaningful parameters
    :param cell_lengths:
    :param cell_angles:
    :param mol_position:
    :param mol_rotation:
    :return:
    '''

    if standardized_sample:
        # de-standardize everything
        means = torch.Tensor(dataDims['lattice means']).to(cell_lengths.device)
        stds = torch.Tensor(dataDims['lattice stds']).to(cell_lengths.device)

        # soft clipping to ensure correct range with finite gradients
        cell_lengths = cell_lengths * stds[0:3] + means[0:3]
        cell_angles = cell_angles * stds[3:6] + means[3:6]
        mol_position = mol_position * stds[6:9] + means[6:9]
        mol_rotation = mol_rotation * stds[9:12] + means[9:12]

    # TODO write a bounding function instead of all this ad-hoc stuff

    cell_lengths = F.softplus(cell_lengths - 0.1) + 0.1  # enforces positive nonzero

    cell_angles = enforce_1d_bound(cell_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2, mode='soft')  # prevent too-skinny cells
    mol_position = enforce_1d_bound(mol_position, 0.5, 0.5, mode='soft')

    if (rotation_type == 'fractional rotvec') or return_transforms:
        pass  # T_fc_list, T_cf_list, generated_cell_volumes = coor_trans_matrix(cell_lengths, cell_angles)

    # if rotation_type == 'fractional rotvec':  # todo implement fractional rotation
    #     mol_rotation = torch.einsum('nij,nj->ni', (T_fc_list, mol_rotation))

    elif rotation_type == 'cartesian rotvec':
        pass

    norms = torch.linalg.norm(mol_rotation, dim=1)
    normed_norms = enforce_1d_bound(norms, torch.pi, torch.pi, mode='soft')
    mol_rotation = mol_rotation / norms[:, None] * normed_norms[:, None]  # renormalize

    for i in range(len(cell_lengths)):
        if enforce_crystal_system:
            lattice = lattices[i]

            # enforce agreement with crystal system
            if lattice.lower() == 'triclinic':
                pass
            elif lattice.lower() == 'monoclinic':  # fix alpha and gamma
                cell_angles[i, 0], cell_angles[i, 2] = torch.pi / 2, torch.pi / 2
            elif lattice.lower() == 'orthorhombic':  # fix all angles
                cell_angles[i] = torch.ones(3) * torch.pi / 2
            elif (lattice.lower() == 'tetragonal'):  # fix all angles and a & b vectors
                cell_angles[i] = torch.ones(3) * torch.pi / 2
                cell_lengths[i, 0], cell_lengths[i, 1] = torch.mean(cell_lengths[i, 0:2]) * torch.ones(2).to(cell_lengths.device)
            elif (lattice.lower() == 'hexagonal') or (lattice.lower() == 'trigonal') or (lattice.lower() == 'rhombohedral'):
                cell_lengths[i, 0], cell_lengths[i, 1] = torch.mean(cell_lengths[i, 0:2]) * torch.ones(2).to(cell_lengths.device)
                cell_angles[i, 0:2] = torch.pi / 2
                cell_angles[i, 2] = torch.pi * 2 / 3
            elif (lattice.lower() == 'cubic'):  # all angles 90 all lengths equal
                cell_lengths[i] = cell_lengths[i].mean() * torch.ones(3).to(cell_lengths.device)
                cell_angles[i] = torch.pi * torch.ones(3) / 2
            else:
                print(lattice + ' is not a valid crystal lattice!')
                sys.exit()
        else:
            # don't assume a crystal system, but snap angles close to 90, to assist in precise symmetry
            pass  # cell_angles[i, torch.abs(cell_angles[i] - torch.pi / 2) < 0.01] = torch.pi / 2

    if return_transforms:
        return cell_lengths, cell_angles, mol_position, mol_rotation, None, None, None  # T_fc_list, T_cf_list, generated_cell_volumes
    else:
        return cell_lengths, cell_angles, mol_position, mol_rotation


def compute_lattice_vector_overlap(coords_list, T_cf_list, normed_lattice_vectors=None):
    if normed_lattice_vectors is None:
        # initialize fractional lattice vectors - should be exactly identical to what's in molecule_featurizer.py
        supercell_scale = 2  # t
        n_cells = (2 * supercell_scale + 1) ** 3

        fractional_translations = np.zeros((n_cells, 3))  # initialize the translations in fractional coords
        i = 0
        for xx in range(-supercell_scale, supercell_scale + 1):
            for yy in range(-supercell_scale, supercell_scale + 1):
                for zz in range(-supercell_scale, supercell_scale + 1):
                    fractional_translations[i] = np.array((xx, yy, zz))
                    i += 1
        lattice_vectors = torch.Tensor(fractional_translations[np.argsort(np.abs(fractional_translations).sum(1))][1:])  # leave out the 0,0,0 element
        normed_lattice_vectors = lattice_vectors / torch.linalg.norm(lattice_vectors, axis=1)[:, None]

    # Ip_list = compute_principal_axes_list(coords_list)
    Ip_list, _, _ = batch_molecule_principal_axes(coords_list)

    # get mol axes in fractional basis
    vectors_f = torch.einsum('nij,nmj->nmi', (T_cf_list, Ip_list))

    # compute overlaps
    normed_vectors_f = vectors_f / torch.linalg.norm(vectors_f, axis=2)[:, :, None]
    return torch.einsum('ij,nmj->nmi', (normed_lattice_vectors, normed_vectors_f))


def get_cell_fractional_centroids(coords, T_cf):
    '''
    input is the cartesian coordinates and the c->f transformation matrix
    '''
    if isinstance(coords, np.ndarray):
        return np.einsum('nmj,ij->nmi', coords, T_cf).mean(1)
    elif torch.is_tensor(coords):
        return torch.einsum('nmj,ij->nmi', (coords, T_cf)).mean(1)


def c_f_transform(coords, T_cf):
    '''
    input is the cartesian coordinates and the c->f transformation matrix
    '''
    if coords.ndim == 2:  # option for extra dimension
        if isinstance(coords, np.ndarray):
            return np.einsum('nj,ij->ni', coords, T_cf)
        elif torch.is_tensor(coords):
            return torch.einsum('nj,ij->ni', (coords, T_cf))
    elif coords.ndim == 3:
        if isinstance(coords, np.ndarray):
            return np.einsum('nmj,ij->nmi', coords, T_cf)
        elif torch.is_tensor(coords):
            return torch.einsum('nmj,ij->nmi', (coords, T_cf))


def f_c_transform(coords, T_fc):
    '''
    input is the fractional coordinates and the f->c transformation matrix
    '''
    if coords.ndim == 2:  # option for extra dimension
        if isinstance(coords, np.ndarray):
            return np.einsum('nj,ij->ni', coords, T_fc)
        elif torch.is_tensor(coords):
            return torch.einsum('nj,ij->ni', (coords, T_fc))
    elif coords.ndim == 3:
        if isinstance(coords, np.ndarray):
            return np.einsum('nmj,ij->nmi', coords, T_fc)
        elif torch.is_tensor(coords):
            return torch.einsum('nmj,ij->nmi', (coords, T_fc))


#
# def cell_analysis(data, debug=False, return_final_coords = False, return_sym_ops = False):
#     '''
#     DEPRECATED
#     Parameters
#     ----------
#     data
#     debug
#     return_final_coords
#     return_sym_ops
#
#     Returns
#     -------
#
#     '''
#     cell_lengths = data.cell_params[:, 0:3]
#     cell_angles = data.cell_params[:, 3:6]
#
#     '''
#     get the centroids in the fractional basis
#     '''
#     canonical_centroids_list = []
#     canonical_centroids_inds = []
#     for i in range(data.num_graphs):
#         centroids_f = get_cell_fractional_centroids(torch.Tensor(data.ref_cell_pos[i]), torch.linalg.inv(data.T_fc[i]))
#         centroids_f -= torch.floor(centroids_f)
#         canonical_centroids_inds.append(torch.argmin(torch.linalg.norm(centroids_f, dim=1)))
#         canonical_centroids_list.append(centroids_f[canonical_centroids_inds[-1]])
#
#     canonical_frac_centroids = torch.stack(canonical_centroids_list)
#     canonical_centroids_inds = torch.stack(canonical_centroids_inds)
#
#     mol_orientations_list = []
#     final_coords_list = []
#     target_handedness = torch.zeros(data.num_graphs)
#     for i in range(data.num_graphs):
#         '''
#         allow each molecule to stick with its given handedness
#         '''
#         coords = torch.Tensor(data.ref_cell_pos[i][canonical_centroids_inds[i]],device=data.x.device)
#         coords -= coords.mean(0)
#         Ip_axes, _, _ = single_molecule_principal_axes(coords)
#         target_handedness[i] = compute_Ip_handedness(Ip_axes)
#         normed_alignment_target = torch.eye(3)
#         normed_alignment_target[0,0] = target_handedness[i] # adjust the target so that the molecule doesn't invert during the rotation
#
#         rot_matrix = torch.matmul(normed_alignment_target.T, torch.linalg.inv(Ip_axes.float()).T)
#         components = torch.Tensor(Rotation.from_matrix(rot_matrix.T).as_rotvec())  # CRITICAL we want the inverse transform here (transpose is the inverse for unitary rotation matrix)
#         mol_orientations_list.append(torch.Tensor(components))
#         if debug:  # confirm this rotation gets the desired orientation
#             std_coords = torch.inner(rot_matrix, coords).T
#             Ip_axes, _, _ = single_molecule_principal_axes(std_coords)
#             assert F.l1_loss(Ip_axes, normed_alignment_target, reduction='sum') < 0.5
#
#         final_coords_list.append(coords)  # will record if we had to invert
#
#     mol_orientations = torch.stack(mol_orientations_list)
#
#     if return_sym_ops:
#         sym_ops_list = []
#         for i in tqdm.tqdm(range(data.num_graphs)):
#             struc_lattice = lattice.Lattice(data.T_fc[i].T.type(dtype=torch.float16))
#             pymat_struc1 = structure.IStructure(species=data.x[data.batch == i, 0].repeat(data.Z[i]),
#                                                 coords=data.ref_cell_pos[i].reshape(int(data.Z[i] * len(data.pos[data.batch == i])), 3),
#                                                 lattice=struc_lattice, coords_are_cartesian=True)
#             sg_analyzer1 = analyzer.SpacegroupAnalyzer(pymat_struc1)
#             sym_ops = sg_analyzer1.get_symmetry_operations()
#             sym_ops_list.append([torch.Tensor(sym_ops[n].affine_matrix,device=data.x.device) for n in range(len(sym_ops))])
#
#     if return_final_coords:
#         if return_sym_ops:
#             return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), target_handedness, final_coords_list, sym_ops_list
#         else:
#             return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), target_handedness, final_coords_list
#     else:
#         if return_sym_ops:
#             return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), target_handedness, sym_ops_list
#         else:
#             return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), target_handedness


def find_coord_in_box(coords, box):
    return np.where((coords[:, 0] < box[0]) * (coords[:, 1] < box[1]) * (coords[:, 2] < box[2]))[0]


def unit_cell_analysis(unit_cell_coords, sg_ind, asym_unit_dict, T_cf, enforce_right_handedness=False):
    """

    Parameters
    ----------
    unit_cell_coords: coordinates for the full unit cell. Each list entry [Z, n_atoms, 3]
    sg_ind: space group index
    asym_unit_dict: dict which defines the asymmetric unit for each space group
    T_cf : list of cartesian-to-fractional matrix transforms
    enforce_right_handedness : DEPRECATED doesn't make sense given nature of our transform
    Returns : standardized cell parameters and canonical conformer handedness
    -------

    """
    if isinstance(unit_cell_coords, np.ndarray):
        asym_unit = np.asarray(asym_unit_dict[str(int(sg_ind))])  # will only work for units which we have written down the parameterization for
    else:
        unit_cell_coords = unit_cell_coords.cpu().detach().numpy()
        asym_unit = asym_unit_dict[str(int(sg_ind))].cpu().detach().numpy()
        T_cf = T_cf.cpu().detach().numpy()

    # identify which of the Z asymmetric units is canonical
    centroids_cartesian = unit_cell_coords.mean(-2)
    centroids_fractional = np.inner(T_cf, centroids_cartesian).T
    centroids_fractional -= np.floor(centroids_fractional)
    canonical_conformer_index = find_coord_in_box(centroids_fractional, asym_unit)

    canonical_conformer_coords = unit_cell_coords[canonical_conformer_index[0]] # we enforce in the filtering step that there must be exactly one centroid in the canonical asymmetric unit

    # next we need to compute the inverse of the rotation required to align the molecule with the cartesian axes
    Ip_axes, _, _ = compute_principal_axes_np(canonical_conformer_coords)
    handedness = compute_Ip_handedness(Ip_axes)

    '''
    we want the matrix which rotates from the standard to the native orientation
    native_orientation = rotation_matrix @ standard_orientation
    rotation_matrix = native_orientation @ inv(standard_orientation)

    standard_orientation = inv(rotation_matrix) @ native_orientation

    '''
    alignment = np.eye(3)
    if not enforce_right_handedness:
        alignment[0, 0] = handedness

    rotation_matrix = Ip_axes.T @ np.linalg.inv(alignment.T)

    ''' debugging 
    # check original transform on the Ip axes
    print(np.sum(np.abs(np.einsum('ij,nj->ni', rotation_matrix, alignment) - Ip_axes)) < 1e-5)
    
    # check the inverse transform
    centered_conformer = (canonical_conformer_coords - canonical_conformer_coords.mean(0))
    transformed_canonical_coords = np.einsum('ij,nj->ni', np.linalg.inv(rotation_matrix), centered_conformer)
    Ip_axes_transformed, _, _ = compute_principal_axes_np(transformed_canonical_coords)

    assert np.sum(np.abs(Ip_axes_transformed - alignment)) < 1e-5
    '''

    # test if we got it right
    if not enforce_right_handedness:
        assert np.linalg.det(rotation_matrix) > 0  # negative determinant is an improper rotation, which will not work

    mol_orientation = Rotation.from_matrix(rotation_matrix).as_rotvec()
    mol_position = centroids_fractional[canonical_conformer_index[0]]

    return mol_position, mol_orientation, handedness


def flip_I3(coords, Ip):
    '''
    flip the given coordinates such that the third principal inertial vector direction is swapped
    thus switching the handedness of the coordinates
    '''
    target_Ip = Ip.clone()
    target_Ip[0] = -target_Ip[0]
    rmat = target_Ip.T @ torch.linalg.inv(Ip).T
    flipped_coords = torch.inner(rmat, coords).T

    return flipped_coords


def invert_coords(coords):
    return -(coords - coords.mean(0)) + coords.mean(0)


def compute_principal_axes_list(coords_list, masses_list=None):
    Ip_axes_list = torch.zeros((len(coords_list), 3, 3), device=coords_list[0].device)
    if masses_list is None:
        for i, coords in enumerate(coords_list):
            Ip_axes_list[i], _, _ = single_molecule_principal_axes(coords)
    else:
        for i, (coords, masses) in enumerate(zip(coords_list, masses_list)):
            Ip_axes_list[i], _, _ = single_molecule_principal_axes(coords, masses)
    return Ip_axes_list


def align_crystaldata_to_principal_axes(data, handedness=None):
    """
    only works for geometric principal axes (all atoms mass = 1)
    """
    coords_list = [data.pos[data.ptr[i]:data.ptr[i + 1]] for i in range(data.num_graphs)]
    coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(data.num_graphs)]
    # principal_axes_list = compute_principal_axes_list(coords_list_centred, masses_list = None)
    principal_axes_list, _, _ = batch_molecule_principal_axes(coords_list_centred)  # much faster

    eye = torch.tile(torch.eye(3, device=data.x.device), (data.num_graphs, 1, 1))  # set as right-handed in general
    if handedness is not None:  # otherwise, custom
        eye[:, 0, 0] = handedness

    # rotation2 = torch.matmul(eye2.reshape(data.num_graphs, 3, 3), torch.linalg.inv(principal_axes_list.reshape(data.num_graphs, 3, 3))) # one step

    rotation_matrix_list = [torch.matmul(torch.linalg.inv(principal_axes_list[i]), eye[i]) for i in range(data.num_graphs)]

    data.pos = torch.cat([torch.einsum('ji, mj->mi', (rotation_matrix_list[i], coords_list_centred[i])) for i in range(data.num_graphs)])

    # for debugging
    # std_coords_list = [torch.einsum('ji, mj->mi', (rotation_matrix_list[i], coords_list_centred[i])) for i in range(data.num_graphs)]
    # principal_axes_list2, _, _ = batch_molecule_principal_axes(std_coords_list)  # much faster
    # print(torch.abs(principal_axes_list2 - eye).sum((1,2))) # should be close to zero
    return data


def random_crystaldata_alignment(data):
    coords_list = [data.pos[data.ptr[i]:data.ptr[i + 1]] for i in range(data.num_graphs)]
    coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(data.num_graphs)]

    rotation_matrix_list = torch.tensor(Rotation.random(num=data.num_graphs).as_matrix(), device=data.x.device, dtype=data.pos.dtype)
    data.pos = torch.cat([torch.einsum('ji, mj->mi', (rotation_matrix_list[i], coords_list_centred[i])) for i in range(data.num_graphs)])

    return data

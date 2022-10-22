import numpy as np
from utils import compute_principal_axes_np, compute_principal_axes_torch
from scipy.spatial.transform import Rotation
import torch
import time
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import sys
from pymatgen.symmetry import analyzer
from pymatgen.core import (structure, lattice)
import tqdm


def compute_Ip_handedness(Ip): # todo remove duplicate function
    if isinstance(Ip, np.ndarray):
        if Ip.ndim == 2:
            return np.sign(np.dot(Ip[0], np.cross(Ip[1], Ip[2])).sum())
        elif Ip.ndim == 3:
            return np.sign(np.dot(Ip[:, 0], np.cross(Ip[:, 1], Ip[:, 2])).sum())

    elif torch.is_tensor(Ip):
        if Ip.ndim == 2:
            return torch.sign(torch.mul(Ip[0], torch.cross(Ip[1], Ip[2])).sum()).float()
        elif Ip.ndim == 3:
            return torch.sign(torch.mul(Ip[:, 0], torch.cross(Ip[:, 1], Ip[:, 2], dim=1)).sum(1))


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
        rmat = torch.Tensor(((one, zero, zero), (zero, cos, -sin), (zero, sin, cos)))
    elif axis == "Y":
        rmat = torch.Tensor(((cos, zero, sin), (zero, one, zero), (-sin, zero, cos)))
    elif axis == "Z":
        rmat = torch.Tensor(((cos, -sin, zero), (sin, cos, zero), (zero, zero, one)))
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return rmat.to(angle.device)
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


def fast_differentiable_coor_trans_matrix(cell_lengths, cell_angles):
    '''
    compute f->c and c->f transforms as well as cell volume in a vectorized, differentiable way
    '''
    cos_a = torch.cos(cell_angles)
    sin_a = torch.sin(cell_angles)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[:, 0] ** 2 - cos_a[:, 1] ** 2 - cos_a[:, 2] ** 2 + 2.0 * cos_a[:, 0] * cos_a[:, 1] * cos_a[:, 2]

    vol = torch.sign(val) * torch.prod(cell_lengths, dim=1) * torch.sqrt(torch.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    T_fc_list = torch.zeros((len(cell_lengths), 3, 3)).to(cell_lengths.device)
    T_cf_list = torch.zeros((len(cell_lengths), 3, 3)).to(cell_lengths.device)

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


def fast_differentiable_cell_vectors(T_fc_list):
    '''
    convert fractional vectors (1,1,1) into cartesian cell vectors (a,b,c)
    '''
    eyevec = torch.tile(torch.eye(3).to(T_fc_list.device), (len(T_fc_list), 1, 1))
    return torch.matmul(T_fc_list, eyevec).permute(0, 2, 1)


def fast_differentiable_ref_to_supercell(reference_cell_list, cell_vector_list, T_fc_list, atoms_list, z_values, supercell_scale=2, cutoff=5, inside_mode='ref mol'):
    n_cells = (2 * supercell_scale + 1) ** 3
    fractional_translations = torch.zeros((n_cells, 3))  # initialize the translations in fractional coords
    i = 0
    for xx in range(-supercell_scale, supercell_scale + 1):
        for yy in range(-supercell_scale, supercell_scale + 1):
            for zz in range(-supercell_scale, supercell_scale + 1):
                fractional_translations[i] = torch.tensor((xx, yy, zz))
                i += 1
    sorted_fractional_translations = fractional_translations[torch.argsort(fractional_translations.abs().sum(1))].to(T_fc_list.device)

    supercell_coords_list = []
    supercell_atoms_list = []
    ref_mol_inds_list = []
    for i, (ref_cell, cell_vectors, atoms, z_value) in enumerate(zip(reference_cell_list, cell_vector_list, atoms_list, z_values)):
        ref_cell = torch.Tensor(ref_cell)
        supercell_coords = ref_cell.clone().reshape(z_value * ref_cell.shape[1], 3).tile(n_cells, 1)  # duplicate over XxXxX supercell
        cart_translations_i = torch.mul(cell_vectors.tile(n_cells, 1), sorted_fractional_translations.reshape(n_cells * 3, 1))  # 3 dimensions
        cart_translations = torch.stack(cart_translations_i.split(3, dim=0), dim=0).sum(1)

        supercell_coords_list.append(
            supercell_coords + torch.repeat_interleave(cart_translations, ref_cell.shape[1] * ref_cell.shape[0], dim=0)
        )

        supercell_atoms = atoms.repeat(n_cells * z_value, 1)
        if inside_mode == 'ref mol':
            # index atoms within the 'canonical' conformer, which is always indexed first
            in_mol_inds = torch.arange(len(atoms))
            ref_mol_inds = torch.ones(len(supercell_atoms), dtype=int).to(ref_cell.device)
            ref_mol_inds[in_mol_inds] = 0

        elif inside_mode == 'unit cell':
            # index atoms within the unit cell
            in_mol_inds = torch.arange(len(atoms) * z_value)
            ref_mol_inds = torch.ones(len(supercell_atoms), dtype=int).to(ref_cell.device)
            ref_mol_inds[in_mol_inds] = 0

        # also, note the atoms which are too far to ever appear in a convolution with this molecule, and generate an index to ignore them
        ref_mol_centroid = supercell_coords_list[-1][in_mol_inds].mean(0)
        centroid_dists = torch.cdist(ref_mol_centroid[None, :], supercell_coords_list[-1], p=2)
        ref_mol_max_dist = torch.max(centroid_dists[:,in_mol_inds])

        # ignore atoms which are more than mol_radius + conv_cutoff + buffer
        ignore_inds = torch.where((centroid_dists > (ref_mol_max_dist + cutoff + 0.5))[0])[0]
        ref_mol_inds[ignore_inds] = 2  # 2 is the index for completely ignoring these atoms in graph convolutions - haven't removed them entirely because it wrecks the crystal periodicity

        # if the crystal is too diffuse, we will have no intermolecular convolution inds - we need a failure mode which accounts for this
        if torch.sum(ref_mol_inds == 1) == 0:
            ref_mol_inds[ignore_inds] = 1 # un-ignore too-far interactions (just so the model doesn't crash)

        supercell_atoms_list.append(supercell_atoms)
        ref_mol_inds_list.append(ref_mol_inds)

    n_copies = torch.tensor(z_values) * n_cells

    return supercell_coords_list, supercell_atoms_list, ref_mol_inds_list, n_copies


def update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list):
    for i in range(supercell_data.num_graphs):
        if i == 0:
            new_batch = torch.ones(len(supercell_atoms_list[i])).int() * i
            new_ptr = torch.zeros(supercell_data.num_graphs + 1)
            new_ptr[1] = len(supercell_list[0])
        else:
            new_batch = torch.cat((new_batch, torch.ones(len(supercell_atoms_list[i])).int() * i))
            new_ptr[i + 1] = new_ptr[i] + len(supercell_list[i])

    # update dataloader with cell info
    supercell_data.x = torch.cat(supercell_atoms_list).type(dtype=torch.float32)
    supercell_data.pos = torch.cat(supercell_list).type(dtype=torch.float32)
    supercell_data.batch = new_batch.type(dtype=torch.int64)
    supercell_data.ptr = new_ptr.type(dtype=torch.int64)
    supercell_data.aux_ind = torch.cat(ref_mol_inds_list).type(dtype=torch.int)

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

    tanh_cutoff = 9

    cell_lengths = F.softplus(cell_lengths - 0.1) + 0.1  # enforces positive nonzero
    # mix of tanh and hardtanh allows us to get close to the limits, but still with a finite gradient outside the range
    norm1 = torch.pi / 2
    cell_angles = F.hardtanh((cell_angles - norm1) / norm1) * norm1 + norm1#(tanh_cutoff*(F.hardtanh((cell_angles - norm1) / norm1) * norm1 + norm1) + (F.tanh((cell_angles - norm1) / norm1) * norm1 + norm1)) / (tanh_cutoff + 1)  # squeeze to -pi/2...pi/2 then re-add pi/2 to make the range 0-pi
    norm2 = 0.5 # todo make sure this is 0-1 not -0.5 to 0.5
    mol_position = F.hardtanh((mol_position - norm2) / norm2) * norm2 + norm2#(tanh_cutoff*(F.hardtanh((mol_position - norm2) / norm2) * norm2 + norm2) + (F.tanh((mol_position - norm2) / norm2) * norm2 + norm2)) / (tanh_cutoff + 1)  # soft squeeze to -0.5 to 0.5, then re-add 0.5 to make the range 0-1

    if (rotation_type == 'fractional rotvec') or return_transforms:
        pass  # T_fc_list, T_cf_list, generated_cell_volumes = fast_differentiable_coor_trans_matrix(cell_lengths, cell_angles)

    # if rotation_type == 'fractional rotvec':  # todo implement fractional rotation
    #     mol_rotation = torch.einsum('nij,nj->ni', (T_fc_list, mol_rotation))

    elif rotation_type == 'cartesian rotvec':
        pass

    norms = torch.linalg.norm(mol_rotation, dim=1)
    norm3 = torch.pi
    normed_norms = F.hardtanh((norms - norm3) / norm3) * norm3 + norm3#(tanh_cutoff*(F.hardtanh((norms - norm3) / norm3) * norm3 + norm3) + (F.tanh((norms - norm3) / norm3) * norm3 + norm3)) / (tanh_cutoff + 1) #F.tanh(norms / torch.pi) * torch.pi  # the norm should be between -pi to pi
    mol_rotation = mol_rotation / norms[:, None] * normed_norms[:, None]  # renormalize

    # old - euler rotations
    # mol_rotation = F.tanh(mol_rotation / torch.pi) * torch.pi  # tanh from -pi to pi

    for i in range(len(cell_lengths)):
        if enforce_crystal_system:  # todo - untested # can alternately enforce this via auxiliary loss on the generator itself
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
                cell_lengths[i, 0], cell_lengths[i, 1] = torch.mean(cell_lengths[i, 0:2]) * torch.ones(2)
            elif (lattice.lower() == 'hexagonal') or (lattice.lower() == 'trigonal') or (lattice.lower() == 'rhombohedral'):
                cell_lengths[i, 0], cell_lengths[i, 1] = torch.mean(cell_lengths[i, 0:2]) * torch.ones(2)
                cell_angles[i,0:2] = torch.pi/2
                cell_angles[i,2] = torch.pi * 2/3
            elif (lattice.lower() == 'cubic'):  # all angles 90 all lengths equal
                cell_lengths[i] = cell_lengths[i].mean() * torch.ones(3)
                cell_angles[i] = torch.pi * torch.ones(3) / 2
            else:
                print(lattice + ' is not a valid crystal lattice!')
                sys.exit()
        else:
            # don't assume a crystal system, but snap angles close to 90, to assist in precise symmetry
            cell_angles[i, torch.abs(cell_angles[i] - torch.pi / 2) < 0.01] = torch.pi / 2

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

    Ip_list = compute_principal_axes_list(coords_list)

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
    if coords.ndim == 2: # option for extra dimension
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
    if coords.ndim == 2: # option for extra dimension
        if isinstance(coords, np.ndarray):
            return np.einsum('nj,ij->ni', coords, T_fc)
        elif torch.is_tensor(coords):
            return torch.einsum('nj,ij->ni', (coords, T_fc))
    elif coords.ndim == 3:
        if isinstance(coords, np.ndarray):
            return np.einsum('nmj,ij->nmi', coords, T_fc)
        elif torch.is_tensor(coords):
            return torch.einsum('nmj,ij->nmi', (coords, T_fc))


def cell_analysis(data, atom_weights, debug=False, return_final_coords = False, return_sym_ops = False):
    cell_lengths = data.cell_params[:, 0:3]
    cell_angles = data.cell_params[:, 3:6]

    '''
    get the centroids in the fractional basis
    '''
    canonical_centroids_list = []
    canonical_centroids_inds = []
    for i in range(data.num_graphs):
        centroids_f = get_cell_fractional_centroids(torch.Tensor(data.ref_cell_pos[i]), torch.linalg.inv(data.T_fc[i]))
        centroids_f -= torch.floor(centroids_f)
        canonical_centroids_inds.append(torch.argmin(torch.linalg.norm(centroids_f, dim=1)))
        canonical_centroids_list.append(centroids_f[canonical_centroids_inds[-1]])

    canonical_frac_centroids = torch.stack(canonical_centroids_list)
    canonical_centroids_inds = torch.stack(canonical_centroids_inds)

    mol_orientations_list = []
    final_coords_list = []
    target_handedness = torch.zeros(data.num_graphs)
    for i in range(data.num_graphs):
        '''
        allow each molecule to stick with its given handedness
        '''
        coords = torch.Tensor(data.ref_cell_pos[i][canonical_centroids_inds[i]]).to(data.x.device)
        coords -= coords.mean(0)
        Ip_axes, _, _ = compute_principal_axes_torch(coords)
        target_handedness[i] = compute_Ip_handedness(Ip_axes)
        normed_alignment_target = torch.eye(3)
        normed_alignment_target[0,0] = target_handedness[i] # adjust the target so that the molecule doesn't invert during the rotation

        rot_matrix = torch.matmul(normed_alignment_target.T, torch.linalg.inv(Ip_axes.float()).T)
        components = torch.Tensor(Rotation.from_matrix(rot_matrix.T).as_rotvec())  # CRITICAL we want the inverse transform here (transpose is the inverse for unitary rotation matrix)
        mol_orientations_list.append(torch.Tensor(components))
        if debug:  # confirm this rotation gets the desired orientation
            std_coords = torch.inner(rot_matrix, coords).T
            Ip_axes, _, _ = compute_principal_axes_torch(std_coords)
            assert F.l1_loss(Ip_axes, normed_alignment_target, reduction='sum') < 0.5

        final_coords_list.append(coords)  # will record if we had to invert

    mol_orientations = torch.stack(mol_orientations_list)

    if return_sym_ops:
        sym_ops_list = []
        for i in tqdm.tqdm(range(data.num_graphs)):
            struc_lattice = lattice.Lattice(data.T_fc[i].T.type(dtype=torch.float16))
            pymat_struc1 = structure.IStructure(species=data.x[data.batch == i, 0].repeat(data.Z[i]),
                                                coords=data.ref_cell_pos[i].reshape(int(data.Z[i] * len(data.pos[data.batch == i])), 3),
                                                lattice=struc_lattice, coords_are_cartesian=True)
            sg_analyzer1 = analyzer.SpacegroupAnalyzer(pymat_struc1)
            sym_ops = sg_analyzer1.get_symmetry_operations()
            sym_ops_list.append([torch.Tensor(sym_ops[n].affine_matrix).to(data.x.device) for n in range(len(sym_ops))])

    if return_final_coords:
        if return_sym_ops:
            return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), target_handedness, final_coords_list, sym_ops_list
        else:
            return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), target_handedness, final_coords_list
    else:
        if return_sym_ops:
            return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), target_handedness, sym_ops_list
        else:
            return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), target_handedness



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

def compute_principal_axes_list(coords_list, masses_list = None):
    Ip_axes_list = torch.zeros((len(coords_list), 3, 3)).to(coords_list[0].device)
    if masses_list is None:
        for i, coords in enumerate(coords_list):
            Ip_axes_list[i], _, _ = compute_principal_axes_torch(coords)
    else:
        for i, (coords,masses) in enumerate(zip(coords_list,masses_list)):
            Ip_axes_list[i], _, _ = compute_principal_axes_torch(coords,masses)
    return Ip_axes_list
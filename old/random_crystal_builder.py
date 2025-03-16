import numpy as np
from utils import compute_principal_axes_np, compute_principal_axes_torch
from scipy.spatial.transform import Rotation
import torch
import time
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import sys


def compute_Ip_handedness(Ip):
    return torch.sign(torch.mul(Ip[0], torch.cross(Ip[1], Ip[2])).sum()).float()


def build_random_crystal(T_cf, T_fc, coords, affine_ops, z_value):
    '''
    generate a random unit cell with appropriate general position point symmetries
    # ignores special positions
    '''

    # apply point symmetry to generate the reference cell
    coords_f = (T_cf.dot(coords.T)).T  # np.transpose(np.dot(T_cf, np.transpose(coords)))  # go to fractional coordinates
    points = coords_f.copy()
    affine_points = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)

    # pattern unit cell via symmetry ops
    cell_coords_f = np.zeros((z_value, len(coords_f), 3))
    cell_coords_c = np.zeros_like(cell_coords_f)
    centroids = np.zeros((z_value, 3))
    for zv in range(z_value):
        cell_coords_f[zv, :, :] = (affine_ops[zv].dot(affine_points.T)).T[:, :-1]  # np.inner(affine_points, affine_ops[zv])[..., :-1] # copied from pyxtal #sym_ops[zv].operate_multi(coords_f)
        cell_coords_f[zv, :, :] -= np.floor(np.average(cell_coords_f[zv, :, :], axis=0))  # ensures all copies in one unit cell
        centroids[zv] = np.average(cell_coords_f[zv], axis=0)
        cell_coords_c[zv, :, :] = (T_fc.dot(cell_coords_f[zv].T)).T  # np.transpose(np.dot(T_fc, np.transpose(cell_coords_f[zv, :, :])))

    # assert (np.amax(centroids < 1)) and (np.amin(centroids) > 0), "Molecules must be inside the unit cell!" # assert everyone is in the unit cell

    return cell_coords_c, cell_coords_f


def build_random_crystal_torch(T_cf, T_fc, coords, affine_ops, z_value):
    '''
    generate a random unit cell with appropriate general position point symmetries
    # ignores special positions
    '''

    # apply point symmetry to generate the reference cell
    coords_f = torch.inner(T_cf, coords).T  # (T_cf.dot(coords.T)).T #np.transpose(np.dot(T_cf, np.transpose(coords)))  # go to fractional coordinates
    affine_points = torch.cat((coords_f, torch.ones(coords_f.shape[:-1] + (1,)).to(coords.device)), dim=-1)

    # pattern unit cell via symmetry ops
    # cell_coords_f = []#torch.zeros((z_value, len(coords_f), 3)).to(coords.device)
    cell_coords_c = []  # torch.zeros_like(cell_coords_f)
    # centroids = []#torch.zeros((z_value, 3)).to(coords.device)
    for zv in range(z_value):
        val = torch.inner(affine_points, affine_ops[zv])[..., :-1]
        # cell_coords_f.append(val - torch.floor(val))
        # if the translated centroid is outside of the cell, shift the molecule
        centroid_f = val.mean(0)
        if (any(centroid_f < 0)) or (any(centroid_f > 1)):
            cell_coords_f = val - torch.floor(centroid_f)
        else:
            cell_coords_f = val
        # centroids.append(torch.mean(cell_coords_f[-1], dim=0))
        cell_coords_c.append(torch.inner(T_fc, cell_coords_f).T)

    # f_coord = torch.stack(cell_coords_f)
    c_coord = torch.stack(cell_coords_c)
    # f_cent = torch.stack(centroids)

    return c_coord  # , f_coord # don't actually need the f_coords in the output


def rotate_invert_and_check(atomic_numbers, masses, coords0, T_fc, T_cf, sym_ops, new_rotation, new_centroid_frac):
    '''
    initialize a standardized position
    '''
    coords = coords0.copy()

    # center coordinates on the center of mass
    CoM = coords.T.dot(masses) / np.sum(masses)
    coords -= CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(coords,masses)  # third row of the Ip_axes matrix is the principal moment axis

    # I1 alignment
    I1_alignment_vec = T_fc.dot(np.ones(3))
    I1_alignment_vec /= np.linalg.norm(I1_alignment_vec)

    # I2 alignment
    corner_point = np.array((0, 1, 1))
    t0 = T_fc.dot(corner_point).dot(I1_alignment_vec) / (I1_alignment_vec.dot(I1_alignment_vec))
    P0 = I1_alignment_vec * t0  # point nearest to (0,1,1)
    I2_alignment_vec = T_fc.dot(corner_point) - P0  # vector between two P0 and (0,1,1)
    I2_alignment_vec /= np.linalg.norm(I2_alignment_vec)

    # I3 alignment
    I3_alignment_vec = np.cross(I1_alignment_vec, I2_alignment_vec)

    # Apply standardization
    double_rot, rmsd = Rotation.align_vectors(b=Ip_axes[1:], a=np.array((I2_alignment_vec, I1_alignment_vec)), return_sensitivity=False)  # align our two vectors
    if rmsd > 0.1:  # if I3 is pointed in the wrong direction
        print('bad rmsd')
    coords = double_rot.apply(coords)
    coords -= coords.mean(0)  # set CoG at (0,0,0)
    if rmsd > 0.1:
        print('rmsd flag')

    standardized_coords = coords.copy()
    I_std, _, _ = compute_principal_axes_np(standardized_coords,masses)

    '''
    perform a rotation & translation
    '''

    # new_rotation = np.random.uniform(-np.pi, np.pi, size=3)
    # new_rotation[1] /= 2 # second rotation is from -pi/2 to pi/2 by convention
    # new_centroid_frac = np.random.uniform(0, 1, size=(3))

    # apply the random rotation
    rotation = Rotation.from_euler('XYZ', new_rotation)
    coords = rotation.apply(coords)

    # identify the true centroid
    affine_points = np.concatenate((new_centroid_frac, np.ones(new_centroid_frac.shape[:-1] + (1,))), axis=-1)
    centroids = np.zeros((len(sym_ops), 3))
    for zv in range(len(sym_ops)):
        centroids[zv, :] = (sym_ops[zv].dot(affine_points.T)).T[:-1]
        centroids[zv, :] -= np.floor(centroids[zv, :])

    centroid_distance_from_origin_f = np.linalg.norm(centroids, axis=1)
    new_centroid_frac = centroids[np.argmin(centroid_distance_from_origin_f)]

    # move to desired centroid
    coords = coords - np.average(coords, axis=0) + T_fc.dot(new_centroid_frac)

    '''
    re-standardize, from the rotated version
    '''

    re_std_coords = coords.copy()

    # center coordinates on the center of mass
    CoM = re_std_coords.T.dot(masses) / np.sum(masses)
    re_std_coords -= CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(re_std_coords, masses)  # third row of the Ip_axes matrix is the principal moment axis

    # Apply standardization
    double_rot2, rmsd = Rotation.align_vectors(b=Ip_axes[1:], a=np.array((I2_alignment_vec, I1_alignment_vec)), return_sensitivity=False)  # align our two vectors
    if rmsd > 0.1:
        print('rmsd flag')

    re_std_coords = double_rot2.apply(re_std_coords)
    re_std_coords -= re_std_coords.mean(0)  # set CoG at (0,0,0)
    I_re_std, _, _ = compute_principal_axes_np(re_std_coords, masses)

    '''
    invert the rotation
    '''

    # translation component
    centroid_c = np.average(coords, axis=0)
    re_coords = coords - centroid_c
    centroid_f = T_cf.dot(centroid_c)

    # rotation component - invert the rotation and extract the three Euler angles
    rmat = (re_coords[:3].T @ np.linalg.inv(re_std_coords[:3]).T)
    re_rotation = Rotation.from_matrix(rmat)
    angles = re_rotation.as_euler('XYZ')  # euler angles of the original rotation

    error = 0
    if np.sum(np.abs(re_std_coords - standardized_coords)) > 0.01:
        print('bad standardization')
        error += 1
    # if np.linalg.norm(angles - new_rotation) > 0.01:
    #     print('bad angles')
    #     error +=1
    if np.linalg.norm(centroid_f - new_centroid_frac) > 0.01:
        print('bad centroid')
        error += 1
    # if np.sum(np.abs(rotation.as_matrix() - rmat)) > 0.01:
    #     print('bad rmat')
    #     error += 1
    if error > 0:
        aa = 1
        from ase import Atoms
        from ase.visualize import view
        mol1 = Atoms(numbers=atomic_numbers, positions=standardized_coords, cell=I_std * 10)
        mol2 = Atoms(numbers=atomic_numbers, positions=re_std_coords, cell=I_re_std * 10)
        view((mol1, mol2))

    return error


def set_standard_position(masses, coords0, T_fc, alignment_check=False):
    coords = coords0.copy()
    # center coordinates on the center of mass
    CoM = coords.T.dot(masses) / np.sum(masses)
    coords -= CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(coords,masses)  # third row of the Ip_axes matrix is the principal moment axis

    I1_alignment_vec = T_fc.dot(np.ones(3))
    I1_alignment_vec /= np.linalg.norm(I1_alignment_vec)
    corner_point = np.array((0, 1, 1))
    t0 = T_fc.dot(corner_point).dot(I1_alignment_vec) / (I1_alignment_vec.dot(I1_alignment_vec))
    P0 = I1_alignment_vec * t0  # point nearest to (0,1,1)
    I2_alignment_vec = T_fc.dot(corner_point) - P0  # vector between two P0 and (0,1,1)
    I2_alignment_vec /= np.linalg.norm(I2_alignment_vec)

    double_rot, rmsd = Rotation.align_vectors(b=Ip_axes[1:], a=np.array((I2_alignment_vec, I1_alignment_vec)), return_sensitivity=False)  # align our two vectors
    coords = double_rot.apply(coords)
    if rmsd > 0.01:
        print('Bad rotation in standardization')

    if alignment_check:
        # confirm we are aligned
        CoM = coords.T.dot(masses) / np.sum(masses)
        coords -= CoM
        Ip_axes2, _, _ = compute_principal_axes_np(coords,masses)  # third row of the Ip_axes matrix is the principal moment axis

        if (np.linalg.norm(Ip_axes2[-1] - I1_alignment_vec) > 0.001) or (np.linalg.norm(Ip_axes2[-2] - I2_alignment_vec) > 0.001):
            print('failed alignment')
    '''
    molecule is now 'set' to a 'standard' orientation, with CoG at (0,0,0)
    '''
    return coords - np.average(coords, axis=0), I1_alignment_vec, I2_alignment_vec


def set_standard_position_torch(masses, coords, T_fc):
    # center coordinates on the center of mass
    CoM = torch.inner(coords.T, masses) / torch.sum(masses)  # coords.T.dot(masses) / torch.sum(masses)
    coords = coords - CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_torch(coords,masses)  # third row of the Ip_axes matrix is the principal moment axis

    I1_alignment_vec = torch.inner(T_fc, torch.ones(3).to(coords.device))  # T_fc.dot(np.ones(3))
    I1_alignment_vec = I1_alignment_vec / torch.linalg.norm(I1_alignment_vec)
    corner_point = torch.tensor((0, 1, 1)).to(coords.device).float()
    t0 = torch.inner(torch.inner(T_fc, corner_point), I1_alignment_vec) / torch.inner(I1_alignment_vec, I1_alignment_vec)
    # T_fc.dot(corner_point).dot(I1_alignment_vec)/(I1_alignment_vec.dot(I1_alignment_vec))
    P0 = I1_alignment_vec * t0  # point nearest to (0,1,1)
    I2_alignment_vec = torch.inner(T_fc, corner_point) - P0  # vector between two P0 and (0,1,1)
    I2_alignment_vec = I2_alignment_vec / torch.linalg.norm(I2_alignment_vec)

    # check if I3 is LH or RH to I1 and I2
    I3_direction = torch.sign(torch.dot(Ip_axes[0], torch.cross(Ip_axes[1], Ip_axes[2])))
    if I3_direction > 0:
        I3_alignment_vec = torch.cross(I2_alignment_vec, I1_alignment_vec)
    elif I3_direction < 0:
        I3_alignment_vec = -torch.cross(I2_alignment_vec, I1_alignment_vec)
    else:
        print('I3 is somehow perpendicular to itself! Bad!')
        I3_alignment_vec = torch.cross(I2_alignment_vec, I1_alignment_vec)

    alignment_target = torch.cat((I3_alignment_vec[None, :], I2_alignment_vec[None, :], I1_alignment_vec[None, :]), dim=0)
    rot_mat = alignment_target.T @ torch.linalg.inv(Ip_axes).T
    rot_coords = torch.inner(rot_mat, coords).T
    final_coords = rot_coords - torch.mean(rot_coords, dim=0), I1_alignment_vec, I2_alignment_vec

    return final_coords


def get_standard_rotation_torch(masses, coords, T_fc):
    # center coordinates on the center of mass
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_torch(coords,masses)

    # new standard position shortcut
    I1_alignment_vec = torch.inner(T_fc, torch.ones(3).to(coords.device))
    norm_I1_alignment_vec = I1_alignment_vec / torch.linalg.norm(I1_alignment_vec)

    corner_point = torch.tensor((0, 1, 1)).to(coords.device).float()
    t0 = torch.inner(torch.inner(T_fc, corner_point), I1_alignment_vec) / torch.inner(I1_alignment_vec, I1_alignment_vec)
    P0 = I1_alignment_vec * t0  # point nearest to (0,1,1)
    I2_alignment_vec = torch.inner(T_fc, corner_point) - P0
    norm_I2_alignment_vec = I2_alignment_vec / torch.linalg.norm(I2_alignment_vec)

    I3_direction = torch.sign(torch.dot(Ip_axes[0], torch.cross(Ip_axes[1], Ip_axes[2])))
    if I3_direction > 0:
        I3_alignment_vec = torch.cross(norm_I2_alignment_vec, norm_I1_alignment_vec)
    elif I3_direction < 0:
        I3_alignment_vec = -torch.cross(norm_I2_alignment_vec, norm_I1_alignment_vec)
    else:
        print('I3 is somehow perpendicular to itself! Bad!')
        I3_alignment_vec = torch.cross(norm_I2_alignment_vec, norm_I1_alignment_vec)

    alignment_target = torch.cat((I3_alignment_vec[None, :], norm_I2_alignment_vec[None, :], norm_I1_alignment_vec[None, :]), dim=0)
    std_mat = alignment_target.T @ torch.linalg.inv(Ip_axes).T

    return std_mat


# @nb.jit(nopython=True)
def randomize_molecule_position_and_orientation(coords, masses, T_fc, T_cf, sym_ops, set_position=None, set_rotation=None, confirm_transform=False, force_centroid=False):
    '''
    :param coords:
    :param masses:
    :param T_fc:
    :param set_position:
    :param set_orientation:
    :param set_rotation:
    :return:
    '''
    # coords0 = coords.copy()
    # random direction & rotation
    if set_rotation is not None:
        new_rotation = np.asarray(set_rotation, dtype=float)
    else:
        new_rotation = np.random.uniform(-np.pi, np.pi, size=3)
        new_rotation[1] /= 2  # second rotation is from -pi/2 to pi/2 by convention
    if set_position is not None:
        new_centroid_frac = np.asarray(set_position, dtype=float)
    else:
        new_centroid_frac = np.random.uniform(0, 1, size=(3))

    # we use the centroid closest to 0,0,0 as canonical
    # sometimes, the centroid we set up here not that one
    # therefore, we first find the canonical centroid, then apply rotations
    # otherwise, later rotation analysis will not work
    if not force_centroid:
        affine_points = np.concatenate((new_centroid_frac, np.ones(new_centroid_frac.shape[:-1] + (1,))), axis=-1)
        centroids = np.zeros((len(sym_ops), 3))
        for zv in range(len(sym_ops)):
            centroids[zv, :] = (sym_ops[zv].dot(affine_points.T)).T[:-1]
            centroids[zv, :] -= np.floor(centroids[zv, :])

        centroid_distance_from_origin_f = np.linalg.norm(centroids, axis=1)
        new_centroid_frac = centroids[np.argmin(centroid_distance_from_origin_f)]

    standardized_coords, normed_corner_vector1, alignment_vector1 = set_standard_position(masses, coords, T_fc, alignment_check=True)  # get standard position at CoG
    rotation = Rotation.from_euler('XYZ', new_rotation)  # apply rotation - not fancy but is invertible
    rotated_coords = rotation.apply(standardized_coords)
    final_coords = rotated_coords - np.average(rotated_coords, axis=0) + T_fc.dot(new_centroid_frac)  # 4. move centroid to the given coordinate

    if confirm_transform:
        # reverse the transform
        # '''
        points = final_coords.copy()

        inverse_std_coords, normed_corner_vector2, alignment_vector2 = set_standard_position(masses, points, T_fc, alignment_check=True)

        # translation component
        centroid_c = np.average(points, axis=0)
        points -= centroid_c
        centroid_check = T_cf.dot(centroid_c)

        # rotation component - invert the rotation and extract the three Euler angles
        rmat = (points[:3].T @ np.linalg.inv(inverse_std_coords[:3]).T)
        rot = Rotation.from_matrix(rmat)
        angle_check = rot.as_euler('XYZ')  # euler angles of the original rotation
        # '''
        # centroid_check, angle_check = retrieve_alignment_parameters(masses, coords, T_fc, T_cf)
        error = 0
        if (np.linalg.norm(centroid_check - new_centroid_frac) > 0.01) or (np.linalg.norm(angle_check - new_rotation) > 0.01):
            error = 1
            print('Error in analyzer!')

        return final_coords, centroid_check, angle_check, new_centroid_frac, new_rotation, error

    return final_coords


def randomize_molecule_position_and_orientation_torch(coords, masses, T_fc, sym_ops, set_position, set_rotation, return_rot=False):
    '''
    :param coords:
    :param masses:
    :param T_fc:
    :param set_position:
    :param set_orientation:
    :param set_rotation:
    :return:
    '''
    affine_points = torch.cat((set_position, torch.ones(set_position.shape[:-1] + (1,)).to(set_position.device)), dim=-1)
    centroids_i = []  # torch.zeros((len(sym_ops), 3)).to(set_position.device)
    for zv in range(len(sym_ops)):
        val = torch.inner(sym_ops[zv], affine_points.T).T[:-1]
        centroids_i.append(val - torch.floor(val))

    centroids = torch.stack(centroids_i).to(set_position.device)

    centroid_distance_from_origin_f = torch.linalg.norm(centroids, axis=1)
    new_centroid_frac = centroids[torch.argmin(centroid_distance_from_origin_f)]

    # CoM =   # coords.T.dot(masses) / torch.sum(masses)
    CoM_coords = coords - torch.inner(coords.T, masses) / torch.sum(masses)
    std_mat = get_standard_rotation_torch(masses, coords, T_fc)
    rot_mat = euler_XYZ_rotation_matrix(set_rotation)
    std_rot_mat = torch.matmul(rot_mat, std_mat)  # compose standardizing and subsequent rotation
    rotated_coords = torch.inner(std_rot_mat, CoM_coords).T
    #
    if return_rot:
        return (rotated_coords - rotated_coords.mean(0) + torch.inner(T_fc, new_centroid_frac), std_mat, rot_mat)
    else:
        return rotated_coords - rotated_coords.mean(0) + torch.inner(T_fc, new_centroid_frac)


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


def retrieve_alignment_parameters(masses, coords, T_fc, T_cf):  # todo deprecated
    '''
    :param masses:
    :param coords:
    :param T_fc:
    :param T_cf:
    :return:
    '''
    points = coords.copy()
    std_coords, normed_corner_vector, _ = set_standard_position(masses, points, T_fc, alignment_check=True)

    # translation component
    centroid_c = np.average(points, axis=0)
    points -= centroid_c
    centroid_f = T_cf.dot(centroid_c)

    # rotation component - invert the rotation and extract the three Euler angles
    rmat = (points[:3].T @ np.linalg.inv(std_coords[:3]).T)
    rot = Rotation.from_matrix(np.linalg.inv(rmat))  # inverse - since we ware backing out the transform
    angles = rot.as_euler('XYZ')  # euler angles of the original rotation

    return centroid_f, angles


def retrieve_new_orientation_parameters(masses, coords, T_fc, T_cf):
    '''
    for doing this one at a time
    :param masses:
    :param coords:
    :return:
    '''

    Ip_axes, _, _ = compute_principal_axes_torch(coords,masses)

    # the cartesian axes
    normed_alignment_target = torch.eye(3).float()

    # make sure the cartesian basis has the same handedness as the molecule principal inertial basis
    normed_alignment_target[0, 0] = compute_Ip_handedness(Ip_axes)

    rot_matrix = torch.matmul(normed_alignment_target.T, torch.linalg.inv(Ip_axes.float()).T)  # rotation matrix between target and current Ip axes
    components = Rotation.from_matrix(torch.linalg.inv(rot_matrix.T)).as_rotvec()  # invert the rot matrix, since we want the inverse transform
    frac_components = T_cf.dot(components)

    return components, frac_components, normed_alignment_target[0, 0]


def fast_differentiable_coor_trans_matrix(cell_lengths, cell_angles):
    '''
    compute f->c and c->f transforms as well as cell volume in a vectorized, differentiable way
    '''
    cos_a = torch.cos(cell_angles)
    sin_a = torch.sin(cell_angles)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[:, 0] ** 2 - cos_a[:, 1] ** 2 - cos_a[:, 2] ** 2 + 2.0 * cos_a[:, 0] * cos_a[:, 1] * cos_a[:, 2]

    # if torch.sum(val < 0) > 0:
    #     aa = 1 # todo what is this doing here

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


def fast_differentiable_standard_rotation_matrix(masses_list, coords_list, T_fc_list, return_euler_angles=False):
    '''
    determine the euler rotation required to rotate a molecule such that it's principal axes align with the cartesian (xyz) axes
    '''
    Ip_axes_list = torch.zeros((len(masses_list), 3, 3)).to(T_fc_list.device)
    for i, (masses, coords) in enumerate(zip(masses_list, coords_list)):
        Ip_axes_i, _, _ = compute_principal_axes_torch(coords,masses)
        Ip_axes_list[i] = Ip_axes_i

    # the cartesian axes
    normed_alignment_target_list = torch.stack([torch.eye(3) for n in range(len(coords_list))]).to(coords_list[0].device)

    # make sure the cartesian basis has the same handedness as the molecule principal inertial basis
    # I3_sign = torch.sign(torch.mul(Ip_axes_list[:, 0], torch.cross(Ip_axes_list[:, 1], Ip_axes_list[:, 2], dim=1)).sum(1))
    normed_alignment_target_list[:, 0, 0] = torch.sign(torch.mul(Ip_axes_list[:, 0], torch.cross(Ip_axes_list[:, 1], Ip_axes_list[:, 2], dim=1)).sum(1))

    if return_euler_angles:
        euler_matrices = torch.matmul(normed_alignment_target_list.permute(0, 2, 1), torch.linalg.inv(Ip_axes_list).permute(0, 2, 1))  # rotation matrix between target and current Ip axes
        angles = torch.stack([torch.Tensor(Rotation.from_matrix(euler_matrices[i]).as_euler('XYZ')) for i in range(len(euler_matrices))])
        return euler_matrices, angles
    else:
        return torch.matmul(normed_alignment_target_list.permute(0, 2, 1), torch.linalg.inv(Ip_axes_list).permute(0, 2, 1))  # rotation matrix between target and current Ip axes


def old_complex_fast_differentiable_standard_rotation_matrix(masses_list, coords_list, T_fc_list):
    Ip_axes_list = torch.zeros((len(masses_list), 3, 3)).to(T_fc_list.device)
    for i, (masses, coords) in enumerate(zip(masses_list, coords_list)):
        Ip_axes_i, _, _ = compute_principal_axes_torch(coords,masses)
        Ip_axes_list[i] = Ip_axes_i

    corner_point = torch.tensor((0, 1, 1)).to(T_fc_list.device).float()
    # top_corner_point = torch.ones(3).to(T_fc_list.device)
    # alignment_target_list = torch.zeros_like(Ip_axes_list)

    target1_list = torch.sum(T_fc_list, dim=2)
    normed_target1_list = torch.div(target1_list, torch.linalg.norm(target1_list, dim=1)[:, None])

    vec2_list = torch.inner(T_fc_list, corner_point)
    target2_list = vec2_list - torch.mul(normed_target1_list, torch.mul(vec2_list, normed_target1_list).sum(1)[:, None])

    I3_sign = torch.sign(torch.mul(Ip_axes_list[:, 0], torch.cross(Ip_axes_list[:, 1], Ip_axes_list[:, 2], dim=1)).sum(1))
    target3_list = torch.mul(torch.cross(target2_list, normed_target1_list, dim=1), I3_sign[:, None])

    alignment_target_list = torch.zeros_like(T_fc_list)  # torch.cat((target3_list[:,None,:], target2_list[:,None,:],normed_target1_list[:,None,:]), dim=1)
    alignment_target_list[:, 0, :] = target3_list
    alignment_target_list[:, 1, :] = target2_list
    alignment_target_list[:, 2, :] = normed_target1_list

    normed_alignment_target_list = torch.div(alignment_target_list, torch.linalg.norm(alignment_target_list, dim=2)[:, :, None])

    return torch.matmul(normed_alignment_target_list.permute(0, 2, 1), torch.linalg.inv(Ip_axes_list).permute(0, 2, 1))  # rotation matrix between target and current Ip axes


def fast_differentiable_applied_rotation_matrix(rotations_list, rotation='rotvec_quat'):
    '''
    rotations_list: Nx3 matrix of euler rotations
    return: Nx3x3 euler matrix for each rotation
    based of non-original code FYI
    '''

    if rotation.lower() == 'rotvec_quat':  # rotvec -> quaternion -> rotation matrix
        theta = torch.linalg.norm(rotations_list, dim=1)
        unit_vector_list = rotations_list / theta[:, None]
        q = torch.cat([torch.cos(theta / 2)[:, None], unit_vector_list * torch.sin(theta / 2)[:, None]], dim=1)
        # R = torch.Tensor((
        #     (1 - 2 * q[:, 2] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3], 2 * q[:, 1] * q[:, 3] + 2 * q[:, 0] * q[:, 2]),
        #     (2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1]),
        #     (2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2], 2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2)
        # ))

        R = torch.stack((torch.stack((1 - 2 * q[:, 2] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3], 2 * q[:, 1] * q[:, 3] + 2 * q[:, 0] * q[:, 2]), dim=1),
                         torch.stack((2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1]), dim=1),
                         torch.stack((2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2], 2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2), dim=1)),
                        dim=1)
        return R

    elif rotation.lower() == 'euler':
        cos_a = torch.cos(rotations_list)
        sin_a = torch.sin(rotations_list)
        one = torch.ones(len(cos_a)).to(rotations_list.device).float()
        zero = torch.zeros(len(cos_a)).to(rotations_list.device).float()

        # batch-define 3 Euler rotations
        m1 = torch.stack((torch.stack((one, zero, zero), dim=1),
                          torch.stack((zero, cos_a[:, 0], -sin_a[:, 0]), dim=1),
                          torch.stack((zero, sin_a[:, 0], cos_a[:, 0]), dim=1)),
                         dim=1)

        m2 = torch.stack((torch.stack((cos_a[:, 1], zero, sin_a[:, 1]), dim=1),
                          torch.stack((zero, one, zero), dim=1),
                          torch.stack((-sin_a[:, 1], zero, cos_a[:, 1]), dim=1)),
                         dim=1)

        m3 = torch.stack((torch.stack((cos_a[:, 2], -sin_a[:, 2], zero), dim=1),
                          torch.stack((sin_a[:, 2], cos_a[:, 2], zero), dim=1),
                          torch.stack((zero, zero, one), dim=1)),
                         dim=1)

        return torch.matmul(torch.matmul(m1, m2), m3)


def fast_differentiable_get_canonical_coords(mol_position, sym_ops_list, z_values):
    '''
    use point symmetry to determine which image is closest to (0,0,0)
    this is the 'canonical' conformer, to which we apply rotations
    '''
    # old method - new method disgrees ~1% of the time for unknown reasons, but it's much much faster
    # canonical_fractional_positions_list = torch.zeros((len(mol_position), 3)).to(mol_position.device)
    # for i, (set_position, sym_ops) in enumerate(zip(mol_position, sym_ops_list)):
    #     # affine_points = torch.cat((set_position, torch.ones(set_position.shape[:-1] + (1,)).to(set_position.device)), dim=-1)
    #     vals = torch.zeros((len(sym_ops), 3))
    #     for zv in range(len(sym_ops)):
    #         vals[zv] = torch.inner(sym_ops[zv], torch.cat((set_position, torch.ones(set_position.shape[:-1] + (1,)).to(set_position.device)), dim=-1).T).T[:-1]
    #     if any(vals.flatten() < 0) or any(vals.flatten() > 1):
    #         centroids = vals - torch.floor(vals)
    #     else:
    #         centroids = vals
    #     # canonical_ind = torch.argmin(torch.linalg.norm(centroids, dim=1))
    #     # canonical_fractional_positions_list[i] = centroids[canonical_ind]
    #     canonical_fractional_positions_list[i] = centroids[torch.argmin(torch.linalg.norm(centroids, dim=1))]

    z_values = torch.tensor(z_values)
    unique_z_values = torch.unique(z_values)
    z_inds = [torch.where(z_values == z)[0] for z in unique_z_values]

    canonical_fractional_positions_list = torch.zeros((len(mol_position), 3)).to(mol_position.device)

    # split it into z values and do each in parallel

    for i, (inds, z_value) in enumerate(zip(z_inds, unique_z_values)):
        vals = torch.zeros((z_value, len(inds), 3)).to(mol_position.device)
        # z_sym_ops = torch.stack([sym_ops_list[j] for j in inds])
        # affine_points = torch.cat((mol_position[inds],torch.ones(mol_position[inds].shape[:-1] + (1,)).to(mol_position.device)),dim=-1)
        for zv in range(z_value):
            vals[zv, :, :] = torch.einsum('nbh,nb->nh', (torch.stack([sym_ops_list[j] for j in inds])[:, zv],
                                                         torch.cat((mol_position[inds],
                                                                    torch.ones(mol_position[inds].shape[:-1] + (1,)).to(mol_position.device)), dim=-1)
                                                         )
                                          )[:, :-1]

        # if any(vals.flatten() < 0) or any(vals.flatten() > 1):
        centroids = vals - torch.floor(vals)
        # else:
        # centroids = vals

        canonical_fractional_positions_list[inds] = centroids[torch.argmin(torch.linalg.norm(centroids, dim=2), dim=0), torch.arange(len(inds))]

    return canonical_fractional_positions_list


def fast_differentiable_apply_rotations_and_translations(
        standardization_rotation_list, applied_rotation_list, coords_list, masses_list, T_fc_list, canonical_mol_position):
    '''
    go to standard position and then to the applied desired rotation in a single step
    starting from CoM coordinates (standardization is defined in CoM / intertial basis)
    '''
    final_coords_list = []
    rotations_list = torch.matmul(applied_rotation_list, standardization_rotation_list)  # list of rotations to apply - order is important since these do not commute

    for i, (rotation, coords, masses, T_fc, new_frac_pos) in enumerate(zip(rotations_list, coords_list, masses_list, T_fc_list, canonical_mol_position)):
        CoM_coords = coords - torch.inner(coords.T, masses) / torch.sum(masses)
        rotated_coords = torch.inner(rotation, CoM_coords).T
        final_coords_list.append(
            rotated_coords - rotated_coords.mean(0) + torch.inner(T_fc, new_frac_pos)
        )

    return final_coords_list


def fast_differentiable_apply_point_symmetry(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values):
    '''
    apply point symmetries to single molecules
    1. compute fractional and cartesian centroids, accounting for sym ops & remaining within the unit cell
    2. apply point symmetries within-molecule cartesian coordinates
    3. apply translations
    '''

    z_values = torch.tensor(z_values)
    unique_z_values = torch.unique(z_values)
    z_inds = [torch.where(z_values == z)[0] for z in unique_z_values]

    reference_cell_list_i = []

    for i, (inds, z_value) in enumerate(zip(z_inds, unique_z_values)):
        lens = torch.tensor([len(final_coords_list[ii]) for ii in inds])
        padded_coords_c = rnn.pad_sequence(final_coords_list, batch_first=True)[inds]
        centroids_c = torch.stack([padded_coords_c[ii, :lens[ii]].mean(0) for ii in range(len(padded_coords_c))])
        centroids_f = torch.einsum('nij,nj->ni', (T_cf_list[inds], centroids_c))
        #
        # padded_coords_f = torch.einsum('nij,nmj->nmi',
        #                                (T_cf_list[inds], padded_coords_c))
        #
        # centroids_f = torch.stack([padded_coords_f[ii, :lens[ii]].mean(0) for ii in range(len(padded_coords_f))])

        ref_cells = torch.zeros((z_value, len(inds), padded_coords_c.shape[1], 3)).to(final_coords_list[0].device)

        z_sym_ops = torch.stack([sym_ops_list[j] for j in inds])
        affine_centroids_f = torch.cat((centroids_f, torch.ones(centroids_f.shape[:-1] + (1,)).to(padded_coords_c.device)), dim=-1)

        for zv in range(z_value):
            centroids_f_z = torch.einsum('nij,nj->ni', (z_sym_ops[:, zv], affine_centroids_f))[..., :-1]  # rotate & translate centroids
            centroids_f_z_in_cell = centroids_f_z - torch.floor(centroids_f_z)  # keep centroids within unit cell

            rot_coords_c = torch.einsum('nmj,nij->nmi', (padded_coords_c - centroids_c[:, None, :], z_sym_ops[:, zv, :3, :3]))

            # the total translation in fractional coordinates
            trans_vecs_c = torch.einsum('nij,nj->ni', (T_fc_list[inds], centroids_f_z_in_cell))  # - centroids_f))

            # subtract initial centroid and add final position
            ref_cells[zv, :, :, :] = rot_coords_c + trans_vecs_c[:, None, :]

        # cells = [ref_cell[:,jj,:lens[jj],:] for jj in range(len(lens))]
        reference_cell_list_i.extend([ref_cells[:, jj, :lens[jj], :] for jj in range(len(lens))])

    sorted_z_inds = torch.argsort(torch.cat(z_inds))

    return [reference_cell_list_i[ind] for ind in sorted_z_inds]


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
        ref_mol_max_dist = torch.max(torch.cdist(ref_mol_centroid[None, :], supercell_coords_list[-1][in_mol_inds], p=2))

        # ignore atoms which are more than mol_radius + conv_cutoff
        ignore_inds = torch.where((torch.cdist(ref_mol_centroid[None, :], supercell_coords_list[-1], p=2) > (ref_mol_max_dist + cutoff))[0])[0]
        ref_mol_inds[ignore_inds] = 2  # 2 is the index for completely ignoring these atoms in graph convolutions - haven't removed them entirely because it wrecks the crystal periodicity

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

    cell_lengths = F.softplus(cell_lengths - 0.1) + 0.1  # enforces positive nonzero
    norm1 = torch.pi / 2
    cell_angles = F.tanh((cell_angles - norm1) / norm1) * norm1 + norm1  # squeeze to -pi/2...pi/2 then re-add pi/2 to make the range 0-pi
    mol_position = F.tanh((mol_position - 0.5) * 2) / 2 + 0.5  # soft squeeze to -0.5 to 0.5, then re-add 0.5 to make the range 0-1

    if (rotation_type == 'fractional rotvec') or return_transforms:
        pass  # T_fc_list, T_cf_list, generated_cell_volumes = fast_differentiable_coor_trans_matrix(cell_lengths, cell_angles)

    if rotation_type == 'fractional rotvec':  # todo implement
        mol_rotation = torch.einsum('nij,nj->ni', (T_fc_list, mol_rotation))

    elif rotation_type == 'cartesian rotvec':
        pass

    norms = torch.linalg.norm(mol_rotation, dim=1)
    normed_norms = F.tanh(norms / torch.pi) * torch.pi  # the norm should be between -pi to pi
    mol_rotation = mol_rotation / norms[:, None] * normed_norms[:, None]  # renormalize

    # old - euler rotations
    # mol_rotation = F.tanh(mol_rotation / torch.pi) * torch.pi  # tanh from -pi to pi

    for i in range(len(cell_lengths)):
        pass
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
        elif (lattice.lower() == 'hexagonal'):  # for rhombohedral, all angles and lengths equal, but not 90.
            # for truly hexagonal, alpha=90, gamma is 120, a=b!=c
            # todo implement 3&6 fold lattices
            print('hexagonal lattice is not yet implemented!')
            pass
        elif (lattice.lower() == 'cubic'):  # all angles 90 all lengths equal
            cell_lengths[i] = cell_lengths[i].mean() * torch.ones(3)
            cell_angles[i] = torch.pi * torch.ones(3) / 2
        else:
            print(lattice + ' is not a valid crystal lattice!')
            sys.exit()
    else:
        # todo we now need a symmetry analyzer to tell us what we're building
        # don't assume a crystal system, but snap angles close to 90, to assist in precise symmetry
        cell_angles[i, torch.abs(cell_angles[i] - torch.pi / 2) < 0.01] = torch.pi / 2

    if return_transforms:
        return cell_lengths, cell_angles, mol_position, mol_rotation, None, None, None  # T_fc_list, T_cf_list, generated_cell_volumes
    else:
        return cell_lengths, cell_angles, mol_position, mol_rotation


def compute_lattice_vector_overlap(masses_list, coords_list, T_cf_list, normed_lattice_vectors=None):
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

    ip_list = []
    for i in range(len(masses_list)):
        ip, _, _ = compute_principal_axes_torch(coords_list[i], masses_list[i])
        ip_list.append(ip)
    Ip_list = torch.stack(ip_list)

    # get mol axes in fractional basis
    vectors_f = torch.einsum('nij,nmj->nmi', (T_cf_list, Ip_list))

    # compute overlaps
    normed_vectors_f = vectors_f / torch.linalg.norm(vectors_f, axis=2)[:, :, None]
    return torch.einsum('ij,nmj->nmi', (normed_lattice_vectors, normed_vectors_f))


def get_fractional_centroids(coords, T_cf):
    return torch.einsum('nmj,ij->nmi', (coords, T_cf)).mean(1)


def new_cell_analysis(data, atom_weights, return_final_coords=False, debug=False):
    cell_lengths = data.cell_params[:, 0:3]
    cell_angles = data.cell_params[:, 3:6]

    canonical_centroids_list = []
    canonical_centroids_inds = []
    for i in range(data.num_graphs):
        coords_f = torch.stack([torch.inner(torch.linalg.inv(data.T_fc[i]), torch.Tensor(data.ref_cell_pos[i][n]).to(data.T_fc.device)).T for n in range(data.Z[i])])
        centroids_f = coords_f.mean(1)
        centroids_f -= torch.floor(centroids_f)
        canonical_centroids_inds.append(torch.argmin(torch.linalg.norm(centroids_f, dim=1)))
        canonical_centroids_list.append(centroids_f[canonical_centroids_inds[-1]])
        # centroids_c = torch.Tensor(data.ref_cell_pos[i]).mean(1)
        # centroids_cf = torch.inner(torch.linalg.inv(data.T_fc[i]),centroids_c).T

    canonical_frac_centroids = torch.stack(canonical_centroids_list)
    canonical_centroids_inds = torch.stack(canonical_centroids_inds)

    coords_list = []
    masses_list = []
    atoms_list = []
    for i in range(data.num_graphs):
        atoms_i = data.x[data.batch == i]
        atomic_numbers = atoms_i[:, 0]
        # heavy_atom_inds = torch.argwhere(atomic_numbers > 1)[:, 0]
        atoms_list.append(atoms_i)
        coords_list.append(data.pos[data.batch == i])
        masses_list.append(torch.tensor([atom_weights[int(number)] for number in atomic_numbers]).to(data.x.device))

    mol_orientations_list = []
    final_coords_list = []
    # target_handedness = torch.zeros(data.num_graphs)
    for i in range(data.num_graphs):
        '''
        enforce the standard position for all canonical molecules to be right-handed
        '''
        coords = torch.Tensor(data.ref_cell_pos[i][canonical_centroids_inds[i]]).to(data.x.device)
        Ip_axes, _, _ = compute_principal_axes_torch(coords, masses_list[i])
        target_handedness = compute_Ip_handedness(Ip_axes)
        if target_handedness == -1:  # switch handedness before the rotation, so that it's right handed all the way through
            coords = -(coords - coords.mean(0)) + coords.mean(0)
            Ip_axes, _, _ = compute_principal_axes_torch(coords, masses_list[i])  # recompute
            if debug:
                target_handedness = compute_Ip_handedness(Ip_axes)
                assert target_handedness == 1

        normed_alignment_target = torch.eye(3)  # always use the right-handed target

        rot_matrix = torch.matmul(normed_alignment_target.T, torch.linalg.inv(Ip_axes.float()).T)
        components = torch.Tensor(Rotation.from_matrix(rot_matrix.T).as_rotvec())  # CRITICAL we want the inverse transform here (transpose is the same for unitary rotation matrix)
        mol_orientations_list.append(torch.Tensor(components))
        if debug:  # confirm this rotation gets the desired orientation
            std_coords = torch.inner(rot_matrix, coords - coords.mean(0)).T
            Ip_axes, _, _ = compute_principal_axes_torch(std_coords,masses_list[i])
            assert F.l1_loss(Ip_axes, normed_alignment_target, reduction='sum') < 0.01

        final_coords_list.append(coords)  # will record if we had to invert

    mol_orientations = torch.stack(mol_orientations_list)

    if return_final_coords:
        return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1), final_coords_list  # [torch.Tensor(data.ref_cell_pos[i][canonical_centroids_inds[i]]) for i in range(data.num_graphs)]
    else:
        return torch.cat((cell_lengths, cell_angles, canonical_frac_centroids, mol_orientations), dim=1)


def flip_I3(coords, Ip):
    '''
    flip the given coordinates such that the third principal inertial vector direction is swapped
    '''
    target_Ip = Ip.clone()
    target_Ip[0] = -target_Ip[0]
    rmat = target_Ip.T @ torch.linalg.inv(Ip).T
    flipped_coords = torch.inner(rmat, coords).T

    return flipped_coords
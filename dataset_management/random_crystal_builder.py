import numpy as np
# import numba as nb
from nikos.rotations import rotation_matrix_from_vectors, euler_rotation, rodrigues_rotation
from utils import compute_principal_axes_np, compute_principal_axes_torch
from scipy.spatial.transform import Rotation
import torch
import time


def check_standardization(numbers, masses, coords0, T_fc, T_cf, sym_ops, new_rotation, new_centroid_frac):
    '''
    test the repeatability of our standardization algo
    :param masses:
    :param coords0:
    :param T_fc:
    :param T_cf:
    :param sym_ops:
    :param new_rotation:
    :param new_centroid_frac:
    :return:
    '''
    coords = coords0.copy()  # some coords
    coords -= coords.mean(0)
    rotation = Rotation.from_euler('XYZ', new_rotation)
    rot_coords = rotation.apply(coords)  # some rotated coords

    '''
    get alignment
    '''
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
    # I3_alignment_vec = np.cross(I1_alignment_vec,I2_alignment_vec)

    alignment_matrix = np.array((I2_alignment_vec, I1_alignment_vec))
    '''
    do standardization
    '''
    # center coordinates on the center of mass
    CoM = coords.T.dot(masses) / np.sum(masses)
    coords -= CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(masses, coords)

    # Apply standardization
    double_rot, rmsd = Rotation.align_vectors(b=Ip_axes[1:], a=alignment_matrix, return_sensitivity=False)  # align our two vectors
    if rmsd > 0.1:
        print('bad rmsd')

    coords = double_rot.apply(coords)
    I1_trans = double_rot.apply(Ip_axes)
    I1, _, _ = compute_principal_axes_np(masses, coords)
    coords -= coords.mean(0)  # set CoG at (0,0,0)

    # center coordinates on the center of mass
    CoM = rot_coords.T.dot(masses) / np.sum(masses)
    rot_coords -= CoM
    Ip_axes2, Ip_moments, I_tensor = compute_principal_axes_np(masses, rot_coords)

    # Apply standardization
    double_rot2, rmsd = Rotation.align_vectors(b=Ip_axes2[1:], a=alignment_matrix, return_sensitivity=False)  # align our two vectors
    if rmsd > 0.1:
        print('bad rmsd')

    rot_coords = double_rot2.apply(rot_coords)
    I2_trans = double_rot2.apply(Ip_axes)
    I2, _, _ = compute_principal_axes_np(masses, rot_coords)
    rot_coords -= rot_coords.mean(0)  # set CoG at (0,0,0)

    error = 0
    if np.sum(np.abs(coords - rot_coords)) > (0.1 * len(coords)):
        print('bad standardization')
        error += 1
        from ase import Atoms
        from ase.visualize import view
        mol1 = Atoms(numbers=numbers, positions=coords)
        mol2 = Atoms(numbers=numbers, positions=rot_coords)
        view((mol1, mol2))

    return error


def rotate_invert_and_check(atomic_numbers, masses, coords0, T_fc, T_cf, sym_ops, new_rotation, new_centroid_frac):
    '''
    initialize a standardized position
    '''
    coords = coords0.copy()

    # center coordinates on the center of mass
    CoM = coords.T.dot(masses) / np.sum(masses)
    coords -= CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(masses, coords)  # third row of the Ip_axes matrix is the principal moment axis

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
    I_std, _, _ = compute_principal_axes_np(masses, standardized_coords)

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
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(masses, re_std_coords)  # third row of the Ip_axes matrix is the principal moment axis

    # Apply standardization
    double_rot2, rmsd = Rotation.align_vectors(b=Ip_axes[1:], a=np.array((I2_alignment_vec, I1_alignment_vec)), return_sensitivity=False)  # align our two vectors
    if rmsd > 0.1:
        print('rmsd flag')

    re_std_coords = double_rot2.apply(re_std_coords)
    re_std_coords -= re_std_coords.mean(0)  # set CoG at (0,0,0)
    I_re_std, _, _ = compute_principal_axes_np(masses, re_std_coords)

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
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(masses, coords)  # third row of the Ip_axes matrix is the principal moment axis

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
        Ip_axes2, _, _ = compute_principal_axes_np(masses, coords)  # third row of the Ip_axes matrix is the principal moment axis

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
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_torch(masses, coords)  # third row of the Ip_axes matrix is the principal moment axis

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
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_torch(masses, coords)

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


def retrieve_alignment_parameters(masses, coords, T_fc, T_cf):
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
    rot = Rotation.from_matrix(rmat)
    angles = rot.as_euler('XYZ')  # euler angles of the original rotation

    return centroid_f, angles


'''
# test for rotation inversion
for i in range(10000):
    points = np.random.uniform(-1,1,size=(30,3))
    
    points -= np.average(points,axis=0)
    
    angles = np.random.uniform(-np.pi,np.pi,size=3)
    angles[1] /= 2
    
    rotation = Rotation.from_euler('xyz', angles)
    rot_points = rotation.apply(points)
    
    rmat = (rot_points[:3].T @ np.linalg.inv(points[:3]).T)
    new_rot = Rotation.from_matrix(rmat)
    new_angles = new_rot.as_euler('xyz')
    if np.linalg.norm(angles - new_angles).astype('float16') > 0.001:
        print(f'error at {i}')
    
'''


def old_randomize_molecule_position_and_orientation(coords, masses, T_fc, set_position=None, set_orientation=None, set_rotation=None):
    '''
    :param coords:
    :param masses:
    :param T_fc:
    :param set_position:
    :param set_orientation:
    :param set_rotation:
    :return:
    '''
    # random direction & rotation
    if set_orientation is not None:
        new_orientation = np.asarray(set_orientation, dtype=float)
    else:
        new_orientation = np.random.uniform(-1, 1, size=3)
    if set_rotation is not None:
        new_rotation = np.asarray(set_rotation, dtype=float)
    else:
        new_rotation = np.random.uniform(-1, 1, size=1)
    if set_position is not None:
        new_centroid_frac = np.asarray(set_position, dtype=float)
    else:
        new_centroid_frac = np.random.uniform(0, 1, size=(3))

    CoM = coords.T.dot(masses) / np.sum(masses)
    coords -= CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(masses, coords)  # third row of the Ip_axes matrix is the principal moment axis

    # align molecule principal axis to new orientation
    rot_mat = rotation_matrix_from_vectors(Ip_axes[-1], T_fc.dot(np.eye(3)[0] - new_orientation))  # define as difference from a vector
    coords = (rot_mat.dot((coords).T)).T  # apply rotation matrix (add and subtract CoM if not already done)    #coords = euler_rotation(rot_mat, coords)

    # rotate about the principal axis by theta
    coords = rodrigues_rotation(Ip_axes[-1], coords, np.prod((new_rotation, 180)))

    # move centroid to new location
    new_centroid_cart = T_fc.dot(new_centroid_frac)  # np.transpose(np.dot(T_fc, np.transpose(new_centroid_frac)))
    coords = coords - np.average(coords, axis=0) + new_centroid_cart

    return coords


# @nb.jit(nopython=True)
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


def ref_to_supercell(reference_cell, z_value, atoms, cell_vectors, supercell_size=1):
    # pattern molecule into reference cell, assuming consistent ordering between dataset (drawn from CSD) and CSD crystals
    coords = torch.tensor(reference_cell.reshape(z_value * reference_cell.shape[1], 3))  # assign reference cell coordinates to the coords array
    atoms = torch.tensor(np.tile(atoms, (z_value, 1)))  # simply copy the feature vectors
    # assert len(atoms) == len(coords) # assert everyone is the same size

    # look at the thing
    # amol = Atoms(numbers = atoms[:,0], positions = coords,cell=np.concatenate((cell_lengths,cell_angles)),pbc=True)
    # visualize.view(amol)
    # 5 tile the supercell
    # index the molecules as 'within main cell' vs 'outside'
    supercell_coords = coords.clone()
    for xx in range(-supercell_size, supercell_size + 1):
        for yy in range(-supercell_size, supercell_size + 1):
            for zz in range(-supercell_size, supercell_size + 1):
                if not all([xx == 0, yy == 0, zz == 0]):
                    supercell_coords = torch.cat((supercell_coords, coords + (cell_vectors[0] * xx + cell_vectors[1] * yy + cell_vectors[2] * zz)), dim=0)

    supercell_atoms = atoms.repeat((supercell_size * 2 + 1) ** 3, 1)
    supercell_atoms = torch.cat((supercell_atoms, torch.ones(len(supercell_atoms))[:, None]), dim=1)  # inside main unit cell
    supercell_atoms[len(atoms):, -1] = 0  # outside of main unit cell

    return supercell_atoms, supercell_coords


def ref_to_supercell_torch(reference_cell, z_value, atoms, cell_vectors, supercell_size=1):
    # pattern molecule into reference cell, assuming consistent ordering between dataset (drawn from CSD) and CSD crystals
    coords = reference_cell.reshape(z_value * reference_cell.shape[1], 3)  # assign reference cell coordinates to the coords array
    atoms = torch.tile(atoms, (z_value, 1))  # simply copy the feature vectors

    # tile the supercell
    # index the molecules as 'within main cell' vs 'outside'
    supercell_coords = coords.clone()
    for xx in range(-supercell_size, supercell_size + 1):
        for yy in range(-supercell_size, supercell_size + 1):
            for zz in range(-supercell_size, supercell_size + 1):
                if not all([xx == 0, yy == 0, zz == 0]):
                    supercell_coords = torch.cat((supercell_coords, coords + (cell_vectors[0] * xx + cell_vectors[1] * yy + cell_vectors[2] * zz)), dim=0)

    supercell_atoms = atoms.repeat((supercell_size * 2 + 1) ** 3, 1)
    ref_cell_inds = torch.ones(len(supercell_atoms)).to(reference_cell.device)[:, None]
    ref_cell_inds[len(atoms) * z_value:, 0] = 0
    supercell_atoms = torch.cat((supercell_atoms, ref_cell_inds), dim=1)  # inside main unit cell
    # supercell_atoms[len(atoms):, -1] = 0  # outside of main unit cell

    # assert len(supercell_coords) == len(supercell_atoms), 'different numbers of atoms & coordinates!'

    return supercell_atoms, supercell_coords


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


# look at the ref cell
'''
cell_vectors = np.transpose(np.dot(T_fc, np.transpose(np.eye(3)))
# single molecule
amol = Atoms(numbers=atoms[:, 0], positions=coords, cell=np.concatenate((cell_lengths,cell_angles)),pbc=True)
# unit cell
amol = Atoms(numbers=atoms[:, 0].repeat(z_value), positions=cell_coords_c.reshape(z_value * len(atoms), 3), cell=np.concatenate((cell_lengths,cell_angles)))
ase.visualize.view(amol)
'''

'''
need to copy effectively this code

def _get_coords_and_species(self, absolute=False, PBC=False, first=False, unitcell=False):
"""
Used to generate coords and species for get_coords_and_species

Args:
absolute: return absolute or relative coordinates 
PBC: whether or not to add coordinates in neighboring unit cells, 
first: whether or not to extract the information from only the first site
unitcell: whether or not to move the molecular center to the unit cell

Returns:
atomic coords: a numpy array of atomic coordinates in the site
species: a list of atomic species for the atomic coords
"""
coord0 = self.mol.cart_coords.dot(self.orientation.matrix.T)  #
wp_atomic_sites = []
wp_atomic_coords = None

for point_index, op2 in enumerate(self.wp.ops):
# Obtain the center in absolute coords
center_relative = op2.operate(self.position)
if unitcell:
    center_relative -= np.floor(center_relative)
center_absolute = np.dot(center_relative, self.lattice.matrix)

# Rotate the molecule (Euclidean metric)
#op2_m = self.wp.generators_m[point_index]

op2_m = self.wp.get_euclidean_generator(self.lattice.matrix, point_index)
rot = op2_m.affine_matrix[:3, :3].T
#if self.diag and self.wp.index > 0:
#    tau = op2.translation_vector
#else:
#    tau = op2_m.translation_vector
tmp = np.dot(coord0, rot) #+ tau

# Add absolute center to molecule
tmp += center_absolute
tmp = tmp.dot(self.lattice.inv_matrix)
if wp_atomic_coords is None:
    wp_atomic_coords = tmp
else:
    wp_atomic_coords = np.append(wp_atomic_coords, tmp, axis=0)
wp_atomic_sites.extend(self.symbols)

if first:
    break

if PBC:
# Filter PBC of wp_atomic_coords
wp_atomic_coords = filtered_coords(wp_atomic_coords, PBC=self.PBC)
# Add PBC copies of coords
m = create_matrix(PBC=self.PBC, omit=True)
# Move [0,0,0] PBC vector to first position in array
m2 = [[0, 0, 0]]
for v in m:
    m2.append(v)
new_coords = np.vstack([wp_atomic_coords + v for v in m2])
wp_atomic_coords = new_coords

if absolute:
wp_atomic_coords = wp_atomic_coords.dot(self.lattice.matrix)


'''


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

    return T_fc_list, T_cf_list, vol


def fast_differentiable_standard_rotation_matrix(masses_list, coords_list, T_fc_list):
    # t00 = time.time()
    Ip_axes_list = torch.zeros((len(masses_list), 3, 3)).to(T_fc_list.device)
    for i, (masses, coords) in enumerate(zip(masses_list, coords_list)):
        Ip_axes_i, _, _ = compute_principal_axes_torch(masses, coords)
        Ip_axes_list[i] = Ip_axes_i
    # t1 = time.time()
    # next define 3 vectors to create alignment matrix
    corner_point = torch.tensor((0, 1, 1)).to(T_fc_list.device).float()
    top_corner_point = torch.ones(3).to(T_fc_list.device)
    alignment_target_list = torch.zeros_like(Ip_axes_list)
    for i, (T_fc, coords) in enumerate(zip(T_fc_list, coords_list)):
        # alignment_target_list[i, 2, :] = torch.inner(T_fc, top_corner_point)
        #
        # t0 = torch.inner(torch.inner(T_fc, corner_point), alignment_target_list[i, 2, :]) / torch.inner(alignment_target_list[i, 2, :], alignment_target_list[i, 2, :])
        # P0 = alignment_target_list[i, 2, :] * t0  # point nearest to (0,1,1)

        vec = torch.inner(T_fc, top_corner_point)
        vec2 = torch.inner(T_fc, corner_point)
        target1 = vec / torch.linalg.norm(vec)
        target2 = vec2 - target1 * torch.inner(vec2, target1)

        # I3_direction = torch.sign(torch.dot(Ip_axes_list[i, 0], torch.cross(Ip_axes_list[i, 1], Ip_axes_list[i, 2])))
        if torch.sign(torch.dot(Ip_axes_list[i, 0], torch.cross(Ip_axes_list[i, 1], Ip_axes_list[i, 2]))) > 0:
            target3 = torch.cross(target2, target1)
        else:
            target3 = -torch.cross(target2, target1)

        alignment_target_list[i] = torch.cat((target3[None,:],target2[None,:],target1[None,:]),dim=0)

        #alignment_target_list[i, 2, :] = vec / torch.linalg.norm(vec)

        # P0 = alignment_target_list[i, 2, :] * torch.inner(torch.inner(T_fc, corner_point), alignment_target_list[i, 2, :])  # point nearest to (0,1,1)
        #alignment_target_list[i, 1, :] = torch.inner(T_fc, corner_point) - alignment_target_list[i, 2, :] * torch.inner(torch.inner(T_fc, corner_point), alignment_target_list[i, 2, :])

        # I3_direction = torch.sign(torch.dot(Ip_axes_list[i, 0], torch.cross(Ip_axes_list[i, 1], Ip_axes_list[i, 2])))
        # if torch.sign(torch.dot(Ip_axes_list[i, 0], torch.cross(Ip_axes_list[i, 1], Ip_axes_list[i, 2]))) > 0:
        #     alignment_target_list[i, 0, :] = torch.cross(alignment_target_list[i, 1, :], alignment_target_list[i, 2, :])
        # else:
        #     alignment_target_list[i, 0, :] = -torch.cross(alignment_target_list[i, 1, :], alignment_target_list[i, 2, :])
        # elif I3_direction < 0:
        #     alignment_target_list[i, 0, :] = -torch.cross(alignment_target_list[i, 1, :], alignment_target_list[i, 2, :])
        # else:
        #     print('I3 is somehow perpendicular to itself! Bad!')
        #     alignment_target_list[i, 0, :] = torch.cross(alignment_target_list[i, 1, :], alignment_target_list[i, 2, :])

    # t2 = time.time()
    normed_alignment_target_list = torch.div(alignment_target_list, torch.linalg.norm(alignment_target_list, dim=2)[:, :, None])

    # t3 = time.time()
    std_mat_list = torch.zeros_like(Ip_axes_list)
    for i, (alignment_target, Ip_axes) in enumerate(zip(normed_alignment_target_list, Ip_axes_list)):
        std_mat_list[i] = alignment_target.T @ torch.linalg.inv(Ip_axes).T
    # t4 = time.time()

    # tot_time = t4-t00
    # print(f'inertial took {t1 - t00:.1f} or {(t1 - t00) / tot_time:.2f} fraction')
    # print(f'alignment targets took {t2 - t1:.1f} or {(t2 - t1) / tot_time:.2f} fraction')
    # print(f'alignment norming took {t3 - t2:.1f} or {(t3 - t2) / tot_time:.2f} fraction')
    # print(f'std mat generation took {t4 - t3:.1f} or {(t4 - t3) / tot_time:.2f} fraction')

    return std_mat_list


def orig_fast_differentiable_standard_rotation_matrix(masses_list, coords_list, T_fc_list):
    t00 = time.time()
    Ip_axes_list = torch.zeros((len(masses_list), 3, 3)).to(T_fc_list.device)
    for i, (masses, coords) in enumerate(zip(masses_list, coords_list)):
        Ip_axes_i, _, _ = compute_principal_axes_torch(masses, coords)
        Ip_axes_list[i] = Ip_axes_i
    t1 = time.time()
    # next define 3 vectors to create alignment matrix
    corner_point = torch.tensor((0, 1, 1)).to(T_fc_list.device).float()
    top_corner_point = torch.ones(3).to(T_fc_list.device)
    alignment_target_list = torch.zeros_like(Ip_axes_list)
    for i, (T_fc, coords) in enumerate(zip(T_fc_list, coords_list)):
        alignment_target_list[i, 2, :] = torch.inner(T_fc, top_corner_point)

        t0 = torch.inner(torch.inner(T_fc, corner_point), alignment_target_list[i, 2, :]) / torch.inner(alignment_target_list[i, 2, :], alignment_target_list[i, 2, :])
        P0 = alignment_target_list[i, 2, :] * t0  # point nearest to (0,1,1)
        alignment_target_list[i, 1, :] = torch.inner(T_fc, corner_point) - P0

        I3_direction = torch.sign(torch.dot(Ip_axes_list[i, 0], torch.cross(Ip_axes_list[i, 1], Ip_axes_list[i, 2])))
        if I3_direction > 0:
            alignment_target_list[i, 0, :] = torch.cross(alignment_target_list[i, 1, :], alignment_target_list[i, 2, :])
        elif I3_direction < 0:
            alignment_target_list[i, 0, :] = -torch.cross(alignment_target_list[i, 1, :], alignment_target_list[i, 2, :])
        else:
            print('I3 is somehow perpendicular to itself! Bad!')
            alignment_target_list[i, 0, :] = torch.cross(alignment_target_list[i, 1, :], alignment_target_list[i, 2, :])

    t2 = time.time()
    normed_alignment_target_list = torch.div(alignment_target_list, torch.linalg.norm(alignment_target_list, dim=2)[:, :, None])

    t3 = time.time()
    std_mat_list = torch.zeros_like(Ip_axes_list)
    for i, (alignment_target, Ip_axes) in enumerate(zip(normed_alignment_target_list, Ip_axes_list)):
        std_mat_list[i] = alignment_target.T @ torch.linalg.inv(Ip_axes).T
    t4 = time.time()

    tot_time = t4 - t00
    print(f'inertial took {t1 - t00:.1f} or {(t1 - t00) / tot_time:.2f} fraction')
    print(f'alignment targets took {t2 - t1:.1f} or {(t2 - t1) / tot_time:.2f} fraction')
    print(f'alignment norming took {t3 - t2:.1f} or {(t3 - t2) / tot_time:.2f} fraction')
    print(f'std mat generation took {t4 - t3:.1f} or {(t4 - t3) / tot_time:.2f} fraction')

    return std_mat_list


def fast_differentiable_applied_rotation_matrix(rotations_list):
    '''
    based of non-original code FYI
    '''
    rotation_matrix_list = torch.zeros((len(rotations_list), 3, 3)).to(rotations_list.device)

    cos_a = torch.cos(rotations_list)
    sin_a = torch.sin(rotations_list)
    one = torch.ones(1).to(rotations_list.device)
    zero = torch.zeros(1).to(rotations_list.device)

    for i, angles in enumerate(rotations_list):
        rotation_matrix_list[i] = torch.matmul(
            torch.matmul(
                torch.Tensor(((one, zero, zero), (zero, cos_a[i, 0], -sin_a[i, 0]), (zero, sin_a[i, 0], cos_a[i, 0]))),
                torch.Tensor(((cos_a[i, 1], zero, sin_a[i, 1]), (zero, one, zero), (-sin_a[i, 1], zero, cos_a[i, 1])))
            ), torch.Tensor(((cos_a[i, 2], -sin_a[i, 2], zero), (sin_a[i, 2], cos_a[i, 2], zero), (zero, zero, one))),
        )

    return rotation_matrix_list


def fast_differentiable_get_canonical_coords(mol_position, sym_ops_list):
    '''
    use point symmetry to determine which image is closest to (0,0,0)
    this is the 'canonical' conformer, to which we apply rotations
    '''

    canonical_fractional_positions_list = torch.zeros((len(mol_position), 3)).to(mol_position.device)
    for i, (set_position, sym_ops) in enumerate(zip(mol_position, sym_ops_list)):
        # affine_points = torch.cat((set_position, torch.ones(set_position.shape[:-1] + (1,)).to(set_position.device)), dim=-1)
        vals = torch.zeros((len(sym_ops), 3))
        for zv in range(len(sym_ops)):
            vals[zv] = torch.inner(sym_ops[zv], torch.cat((set_position, torch.ones(set_position.shape[:-1] + (1,)).to(set_position.device)), dim=-1).T).T[:-1]
        if any(vals.flatten() < 0) or any(vals.flatten() > 1):
            centroids = vals - torch.floor(vals)
        else:
            centroids = vals
        # canonical_ind = torch.argmin(torch.linalg.norm(centroids, dim=1))
        # canonical_fractional_positions_list[i] = centroids[canonical_ind]
        canonical_fractional_positions_list[i] = centroids[torch.argmin(torch.linalg.norm(centroids, dim=1))]

    return canonical_fractional_positions_list


def prev_fast_differentiable_get_canonical_coords(mol_position, sym_ops_list):
    '''
    use point symmetry to determine which image is closest to (0,0,0)
    this is the 'canonical' conformer, to which we apply rotations
    '''

    canonical_fractional_positions_list = torch.zeros((len(mol_position), 3)).to(mol_position.device)
    for i, (set_position, sym_ops) in enumerate(zip(mol_position, sym_ops_list)):
        affine_points = torch.cat((set_position, torch.ones(set_position.shape[:-1] + (1,)).to(set_position.device)), dim=-1)
        vals = torch.zeros((len(sym_ops), 3))
        for zv in range(len(sym_ops)):
            vals[zv] = torch.inner(sym_ops[zv], affine_points.T).T[:-1]
        if any(vals < 0) or any(vals > 1):
            centroids = vals - torch.floor(vals)
        else:
            centroids = vals
        # canonical_ind = torch.argmin(torch.linalg.norm(centroids, dim=1))
        # canonical_fractional_positions_list[i] = centroids[canonical_ind]
        canonical_fractional_positions_list[i] = centroids[torch.argmin(torch.linalg.norm(centroids, dim=1))]

    return canonical_fractional_positions_list


def fast_differentiable_apply_rotations_and_translations(
        standardization_rotation_list, applied_rotation_list, coords_list, masses_list, T_fc_list, canonical_mol_position):
    '''
    go to standard position and then to the applied desired rotation in a single step
    starting from CoM coordinates (standardization is defined in CoM / intertial basis)
    '''
    final_coords_list = []
    rotations_list = torch.matmul(applied_rotation_list, standardization_rotation_list)  # list of rotations to apply

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
    '''

    reference_cell_list = []
    for i, (coords, sym_ops, T_cf, T_fc, z_value) in enumerate(zip(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values)):
        coords_f = torch.inner(T_cf, coords).T
        affine_points = torch.cat((coords_f, torch.ones(coords_f.shape[:-1] + (1,)).to(coords.device)), dim=-1)

        ref_cell = torch.zeros((z_value, len(coords_f), 3)).to(coords.device)
        for zv in range(z_value):
            dup_f = torch.inner(affine_points, sym_ops[zv])[..., :-1]
            centroid_f = dup_f.mean(0)
            if (any(centroid_f < 0)) or (any(centroid_f > 1)):
                image_f = dup_f - torch.floor(centroid_f)
            else:
                image_f = dup_f
            ref_cell[zv] = torch.inner(T_fc, image_f).T

        reference_cell_list.append(ref_cell)

    return reference_cell_list


def prev_fast_differentiable_apply_point_symmetry(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values):
    '''
    apply point symmetries to single molecules
    '''

    reference_cell_list = []
    for i, (coords, sym_ops, T_cf, T_fc, z_value) in enumerate(zip(final_coords_list, sym_ops_list, T_cf_list, T_fc_list, z_values)):
        coords_f = torch.inner(T_cf, coords).T
        affine_points = torch.cat((coords_f, torch.ones(coords_f.shape[:-1] + (1,)).to(coords.device)), dim=-1)

        ref_cell = torch.zeros((z_value, len(coords_f), 3)).to(coords.device)
        for zv in range(z_value):
            dup_f = torch.inner(affine_points, sym_ops[zv])[..., :-1]
            centroid_f = dup_f.mean(0)
            if (any(centroid_f < 0)) or (any(centroid_f > 1)):
                image_f = dup_f - torch.floor(centroid_f)
            else:
                image_f = dup_f
            ref_cell[zv] = torch.inner(T_fc, image_f).T

        reference_cell_list.append(ref_cell)

    return reference_cell_list

def fast_differentiable_cell_vectors(T_fc_list):
    '''
    convert fractional vectors (1,1,1) into cartesian cell vectors (a,b,c)
    '''
    eyevec = torch.tile(torch.eye(3).to(T_fc_list.device), (len(T_fc_list), 1, 1))
    return torch.matmul(T_fc_list, eyevec).permute(0, 2, 1)


def fast_differentiable_ref_to_supercell(reference_cell_list, cell_vector_list, T_fc_list, atoms_list, z_values):
    fractional_translations = torch.zeros((27, 3))  # initialize the translations in fractional coords
    i = 0
    for xx in range(-1, 2):
        for yy in range(-1, 2):
            for zz in range(-1, 2):
                fractional_translations[i] = torch.tensor((xx, yy, zz))
                i += 1
    sorted_fractional_translations = fractional_translations[torch.argsort(fractional_translations.abs().sum(1))].to(T_fc_list.device)

    supercell_list = []
    supercell_atoms_list = []
    for i, (ref_cell, cell_vectors, atoms, z_value) in enumerate(zip(reference_cell_list, cell_vector_list, atoms_list, z_values)):
        supercell_coords = ref_cell.clone().reshape(z_value * ref_cell.shape[1], 3).tile(27, 1)  # duplicate over 3x3x3 supercell
        cart_translations_i = torch.mul(cell_vectors.tile(27, 1), sorted_fractional_translations.reshape(81, 1))
        cart_translations = torch.stack(cart_translations_i.split(3, dim=0), dim=0).sum(1)

        supercell_list.append(
            supercell_coords + torch.repeat_interleave(cart_translations, ref_cell.shape[1] * ref_cell.shape[0], dim=0)
        )

        supercell_atoms = atoms.repeat(27 * z_value, 1)
        ref_cell_inds = torch.ones(len(supercell_atoms)).to(ref_cell.device)[:, None]
        ref_cell_inds[len(atoms) * z_value:, 0] = 0
        supercell_atoms_w_ind = torch.cat((supercell_atoms, ref_cell_inds), dim=1)  # inside main unit cell

        supercell_atoms_list.append(
            supercell_atoms_w_ind
        )

    return supercell_list, supercell_atoms_list

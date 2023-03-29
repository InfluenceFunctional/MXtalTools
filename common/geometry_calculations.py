import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_scatter import scatter


def compute_principal_axes_np(coords):
    points = coords - coords.mean(0)

    x, y, z = points.T
    Ixx = np.sum((y ** 2 + z ** 2))
    Iyy = np.sum((x ** 2 + z ** 2))
    Izz = np.sum((x ** 2 + y ** 2))
    Ixy = -np.sum(x * y)
    Iyz = -np.sum(y * z)
    Ixz = -np.sum(x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])  # inertial tensor
    Ipm, Ip = np.linalg.eig(I)  # principal inertial tensor
    Ipm, Ip = np.real(Ipm), np.real(Ip)
    sort_inds = np.argsort(Ipm)
    Ipm = Ipm[sort_inds]
    Ip = Ip.T[sort_inds]  # want eigenvectors to be sorted row-wise (rather than column-wise)

    # cardinal direction is vector from CoM to farthest atom
    dists = np.linalg.norm(points, axis=1)
    max_ind = np.argmax(dists)
    max_equivs = np.argwhere(np.round(dists, 8) == np.round(dists[max_ind], 8))[:, 0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(np.amin(max_equivs))
    direction = points[max_ind]
    direction = np.divide(direction, np.linalg.norm(direction))
    overlaps = Ip.dot(direction)  # check if the principal components point towards or away from the CoG

    Ip = (Ip.T * np.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction
    if np.any(np.abs(overlaps) < 1e-3):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
        # align the 'good' vectors
        fix_ind = np.argmin(np.abs(overlaps))  # vector with vanishing overlap
        if compute_Ip_handedness(Ip) < 0:  # make sure result is right handed
            Ip[fix_ind] = -Ip[fix_ind]

    return Ip, Ipm, I


def compute_inertial_tensor(x, y, z):
    Ixy = -torch.sum(x * y)
    Iyz = -torch.sum(y * z)
    Ixz = -torch.sum(x * z)
    # I = torch.tensor([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]],device=points.device)  # inertial tensor
    I = torch.tensor(
        [[torch.sum((y ** 2 + z ** 2)), Ixy, Ixz],
         [Ixy, torch.sum((x ** 2 + z ** 2)), Iyz],
         [Ixz, Iyz, torch.sum((x ** 2 + y ** 2))]], device=x.device)  # inertial tensor

    Ipm, Ip = torch.linalg.eig(I)  # principal inertial tensor

    return I, Ip, Ipm


def single_molecule_principal_axes(coords, masses=None, return_direction=False):
    if masses is not None:
        print('inertial tensor is purely geometric!')
    x, y, z = coords.T

    I, Ip, Ipm = compute_inertial_tensor(x, y, z)

    Ipm, Ip = torch.real(Ipm), torch.real(Ip)
    sort_inds = torch.argsort(Ipm)
    Ipm = Ipm[sort_inds]
    Ip = Ip.T[sort_inds]  # want eigenvectors to be sorted row-wise (rather than column-wise)

    # cardinal direction is vector from CoM to farthest atom
    dists = torch.linalg.norm(coords, axis=1)  # CoM is at 0,0,0
    max_ind = torch.argmax(dists)
    max_equivs = torch.where(dists == dists[max_ind])[0]  # torch.where(torch.round(dists, decimals=8) == torch.round(dists[max_ind], decimals=8))[0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(torch.amin(max_equivs))
    direction = coords[max_ind]
    # direction = direction / torch.linalg.norm(direction) # magnitude doesn't matter, only the sign
    overlaps = torch.inner(Ip, direction)  # Ip.dot(direction) # check if the principal components point towards or away from the CoG
    if any(overlaps == 0):  # exactly zero is invalid #
        overlaps[overlaps == 0] = 1e-9
    if any(torch.abs(overlaps) < 1e-8):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
        # align the 'good' vectors
        Ip = (Ip.T * torch.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction
        fix_ind = torch.argmin(torch.abs(overlaps))
        other_vectors = np.delete(np.arange(3), fix_ind)
        check_direction = torch.cross(Ip[other_vectors[0]], Ip[other_vectors[1]])
        # align the 'bad' vector
        Ip[fix_ind] = check_direction  # Ip[fix_ind] * torch.sign(torch.dot(check_direction, Ip[fix_ind]))
    else:
        Ip = (Ip.T * torch.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction

    if return_direction:
        return Ip, Ipm, I, direction
    else:
        return Ip, Ipm, I


def batch_molecule_principal_axes(coords_list):
    coords_list_centred = [coord - coord.mean(0) for coord in coords_list]
    all_coords = torch.cat(coords_list_centred)

    ptrs = [0]
    ptrs.extend([len(coord) for coord in coords_list])
    ptrs = torch.tensor(ptrs, dtype=torch.int, device=all_coords.device).cumsum(0)
    batch = torch.cat([(i - 1) * torch.ones(ptrs[i] - ptrs[i - 1], dtype=torch.int64, device=all_coords.device) for i in range(1, len(ptrs))])

    Ixy = -scatter(all_coords[:, 0] * all_coords[:, 1], batch)
    Iyz = -scatter(all_coords[:, 1] * all_coords[:, 2], batch)
    Ixz = -scatter(all_coords[:, 0] * all_coords[:, 2], batch)

    I = torch.cat(
        (torch.vstack((scatter(all_coords[:, 1] ** 2 + all_coords[:, 2] ** 2, batch), Ixy, Ixz))[:, None, :].T,
         torch.vstack((Ixy, scatter(all_coords[:, 0] ** 2 + all_coords[:, 2] ** 2, batch), Iyz))[:, None, :].T,
         torch.vstack((Ixz, Iyz, scatter(all_coords[:, 0] ** 2 + all_coords[:, 1] ** 2, batch)))[:, None, :].T
         ), dim=-2)  # inertial tensor

    Ipm, Ip = torch.linalg.eig(I)  # principal inertial tensor
    Ipm, Ip = torch.real(Ipm), torch.real(Ip)
    sort_inds = torch.argsort(Ipm, dim=1)
    Ipm = torch.stack([Ipm[i, sort_inds[i]] for i in range(len(sort_inds))])
    Ip = torch.stack([Ip[i].T[sort_inds[i]] for i in range(len(sort_inds))])  # want eigenvectors to be sorted row-wise (rather than column-wise)

    # cardinal direction is vector from CoM to farthest atom
    dists = torch.linalg.norm(all_coords, axis=1)  # CoM is at 0,0,0
    max_ind = torch.stack([torch.argmax(dists[batch == i]) + ptrs[i] for i in range(len(ptrs) - 1)])  # find furthest atom in each mol
    direction = all_coords[max_ind]
    overlaps = torch.einsum('nij,nj->ni', (Ip, direction))  # Ip.dot(direction) # check if the principal components point towards or away from the CoG

    Ip_fin = torch.zeros_like(Ip)
    for ii, Ip_i in enumerate(Ip):
        Ip_i = (Ip_i.T * torch.sign(overlaps[ii])).T  # if the vectors have negative overlap, flip the direction
        if any(torch.abs(overlaps[ii]) < 1e-3):  # if any overlaps are vanishing (up to 32 bit precision), determine the direction via the RHR (if two overlaps are vanishing, this will not work)
            # enforce right-handedness in the free vector
            fix_ind = torch.argmin(torch.abs(overlaps[ii]))  # vector with vanishing overlap
            if compute_Ip_handedness(Ip_i) < 0:  # make sure result is right handed
                Ip_i[fix_ind] = -Ip_i[fix_ind]

        Ip_fin[ii] = Ip_i

    return Ip_fin, Ipm, I


def coor_trans_matrix_torch(opt, v, a, return_vol=False):
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(a)
    sin_a = torch.sin(a)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = torch.sign(val) * v[0] * v[1] * v[2] * torch.sqrt(torch.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    m = torch.zeros((3, 3))
    if (opt == 'c_to_f'):
        ''' Converting from cartesian to fractional '''
        m[0, 0] = 1.0 / v[0]
        m[0, 1] = -cos_a[2] / v[0] / sin_a[2]
        m[0, 2] = v[1] * v[2] * (cos_a[0] * cos_a[2] - cos_a[1]) / vol / sin_a[2]
        m[1, 1] = 1.0 / v[1] / sin_a[2]
        m[1, 2] = v[0] * v[2] * (cos_a[1] * cos_a[2] - cos_a[0]) / vol / sin_a[2]
        m[2, 2] = v[0] * v[1] * sin_a[2] / vol
    elif (opt == 'f_to_c'):
        ''' Converting from fractional to cartesian '''
        m[0, 0] = v[0]
        m[0, 1] = v[1] * cos_a[2]
        m[0, 2] = v[2] * cos_a[1]
        m[1, 1] = v[1] * sin_a[2]
        m[1, 2] = v[2] * (cos_a[0] - cos_a[1] * cos_a[2]) / sin_a[2]
        m[2, 2] = vol / v[0] / v[1] / sin_a[2]

    # todo create m in a single-step
    if return_vol:
        return m, torch.abs(vol)
    else:
        return m


def cell_vol_torch(v, a):
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(a)  # in natural units

    ''' Calculate volume of the unit cell '''
    vol = v[0] * v[1] * v[2] * torch.sqrt(torch.abs(1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]))

    return vol


def invert_rotvec_handedness(rotvec):
    rot_mat = Rotation.from_rotvec(rotvec).as_matrix()
    return Rotation.from_matrix(-rot_mat).as_rotvec()  # negative of the rotation matrix gives the accurate rotation for opposite handed object


def compute_Ip_handedness(Ip):
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


def initialize_fractional_vectors(scale=2):
    # initialize fractional cell vectors
    supercell_scale = scale
    n_cells = (2 * supercell_scale + 1) ** 3

    fractional_translations = np.zeros((n_cells, 3))  # initialize the translations in fractional coords
    i = 0
    for xx in range(-supercell_scale, supercell_scale + 1):
        for yy in range(-supercell_scale, supercell_scale + 1):
            for zz in range(-supercell_scale, supercell_scale + 1):
                fractional_translations[i] = np.array((xx, yy, zz))
                i += 1
    sorted_fractional_translations = fractional_translations[np.argsort(np.abs(fractional_translations).sum(1))][1:]  # leave out the 0,0,0 element
    normed_fractional_translations = sorted_fractional_translations / np.linalg.norm(sorted_fractional_translations, axis=1)[:, None]

    return sorted_fractional_translations, normed_fractional_translations

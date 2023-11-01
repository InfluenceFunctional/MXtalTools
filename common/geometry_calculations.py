import sys

import numpy as np
import torch
from torch_scatter import scatter


def compute_principal_axes_np(coords):
    """
    compute the principal axes for a given set of particle coordinates, ignoring particle mass
    use our overlap rules to ensure a fixed direction for all axes under almost all circumstances
    """  # todo harmonize with torch version - currently disagrees ~0.5% of the time
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

    # cardinal direction is vector from CoM to the farthest atom
    dists = np.linalg.norm(points, axis=1)
    max_ind = np.argmax(dists)
    max_equivs = np.argwhere(np.round(dists, 8) == np.round(dists[max_ind], 8))[:, 0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(np.amin(max_equivs))
    direction = points[max_ind]
    direction = np.divide(direction, np.linalg.norm(direction))
    overlaps = Ip.dot(direction)  # check if the principal components point towards or away from the CoG
    signs = np.sign(overlaps)  # returns zero for zero overlap, but we want it to default to +1 in this case
    signs[signs == 0] = 1

    Ip = (Ip.T * signs).T  # if the vectors have negative overlap, flip the direction
    if np.any(np.abs(overlaps) < 1e-3):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
        # align the 'good' vectors
        fix_ind = np.argmin(np.abs(overlaps))  # vector with vanishing overlap
        if compute_Ip_handedness(Ip) < 0:  # make sure result is right handed
            Ip[fix_ind] = -Ip[fix_ind]

    return Ip, Ipm, I


def compute_inertial_tensor_torch(x: torch.tensor, y: torch.tensor, z: torch.tensor):
    """
    compute the inertial tensor for a series of x y z coordinates
    """
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


def single_molecule_principal_axes_torch(coords: torch.tensor, masses=None, return_direction=False):
    """
    compute the principal axes for a given set of particle coordinates, ignoring particle mass
    use our overlap rules to ensure a fixed direction for all axes under almost all circumstances
    optionally return the 'canonical direction' which should have positive overlap with all inertial principal axes
    """
    if masses is not None:
        print('Inertial tensor is purely geometric! Calculation will not account for varying masses')
    x, y, z = coords.T
    I, Ip, Ipm = compute_inertial_tensor_torch(x, y, z)

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


def batch_molecule_principal_axes_torch(coords_list: list):
    """
    rapidly compute principal axes for a list of coordinates in batch fashion
    """
    coords_list_centred = [coord - coord.mean(0) for coord in coords_list]
    all_coords = torch.cat(coords_list_centred)

    ptrs = [0]
    ptrs.extend([len(coord) for coord in coords_list])
    ptrs = torch.tensor(ptrs, dtype=torch.int, device=all_coords.device).cumsum(0)
    batch = torch.cat([(i - 1) * torch.ones(ptrs[i] - ptrs[i - 1], dtype=torch.int64, device=all_coords.device) for i in range(1, len(ptrs))])

    Ixy = -scatter(all_coords[:, 0] * all_coords[:, 1], batch)
    Iyz = -scatter(all_coords[:, 1] * all_coords[:, 2], batch)
    Ixz = -scatter(all_coords[:, 0] * all_coords[:, 2], batch)

    inertial_tensor = torch.cat(
        (torch.vstack((scatter(all_coords[:, 1] ** 2 + all_coords[:, 2] ** 2, batch), Ixy, Ixz))[:, None, :].permute(2, 1, 0),
         torch.vstack((Ixy, scatter(all_coords[:, 0] ** 2 + all_coords[:, 2] ** 2, batch), Iyz))[:, None, :].permute(2, 1, 0),
         torch.vstack((Ixz, Iyz, scatter(all_coords[:, 0] ** 2 + all_coords[:, 1] ** 2, batch)))[:, None, :].permute(2, 1, 0)
         ), dim=-2)  # inertial tensor

    Ipm, Ip = torch.linalg.eig(inertial_tensor)  # principal inertial tensor
    Ipm, Ip = torch.real(Ipm), torch.real(Ip)
    sort_inds = torch.argsort(Ipm, dim=1)
    Ipm = torch.stack([Ipm[i, sort_inds[i]] for i in range(len(sort_inds))])
    Ip = torch.stack([Ip[i].T[sort_inds[i]] for i in range(len(sort_inds))])  # want eigenvectors to be sorted row-wise (rather than column-wise)

    # cardinal direction is vector from CoM to farthest atom
    dists = torch.linalg.norm(all_coords, axis=1)  # CoM is at 0,0,0
    max_ind = torch.stack([torch.argmax(dists[batch == i]) + ptrs[i] for i in range(len(ptrs) - 1)])  # find furthest atom in each mol
    direction = all_coords[max_ind]
    overlaps = torch.einsum('nij,nj->ni', (Ip, direction))  # Ip.dot(direction) # check if the principal components point towards or away from the CoG
    signs = torch.sign(overlaps)  # we want any exactly zero overlaps to come with positive signs
    signs[signs == 0] = 1

    Ip_fin = torch.zeros_like(Ip)
    for ii, Ip_i in enumerate(Ip):
        Ip_i = (Ip_i.T * signs[ii]).T  # if the vectors have negative overlap, flip the direction
        if any(torch.abs(overlaps[ii]) < 1e-3):  # if any overlaps are vanishing (up to 32 bit precision), determine the direction via the RHR (if two overlaps are vanishing, this will not work)
            # enforce right-handedness in the free vector
            fix_ind = torch.argmin(torch.abs(overlaps[ii]))  # vector with vanishing overlap
            if compute_Ip_handedness(Ip_i) < 0:  # make sure result is right handed
                Ip_i[fix_ind] = -Ip_i[fix_ind]

        Ip_fin[ii] = Ip_i

    return Ip_fin, Ipm, inertial_tensor


def coor_trans_matrix_torch(opt: str, v: torch.tensor, a: torch.tensor, return_vol: bool = False):
    """
    Initially borrowed from Nikos
    Quickly convert from cell lengths and angles to fractional transform matrices fractional->cartesian or cartesian->fractional
    """
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
    elif opt == 'f_to_c':
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


def sph2rotvec(angles):
    """
    transform from axis-angle in polar coordinates to rotvec
    theta, phi, r -> xyz
    """
    if isinstance(angles, np.ndarray):
        if angles.ndim > 1:
            theta, phi, r = angles.T
            rotvec = r[:, None] * np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))).T
        else:
            theta, phi, r = angles
            rotvec = r * np.asarray((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))

        return rotvec

    elif torch.is_tensor(angles):
        if angles.ndim > 1:
            theta, phi, r = angles.T
            rotvec = r[:, None] * torch.stack((theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos())).T
        else:
            theta, phi, r = angles
            rotvec = r * torch.Tensor(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos())

        return rotvec

    else:
        print("Array type not supported! Must be np.ndarray or torch.tensor")
        return None


def rotvec2sph(rotvec):
    """
    transform rotation vector with axis rotvec/norm(rotvec) and angle ||rotvec||
    to spherical coordinates theta, phi and r ||rotvec||
    """
    if isinstance(rotvec, np.ndarray):
        r = np.linalg.norm(rotvec, axis=-1)
        if rotvec.ndim == 1:
            rotvec = rotvec[None, :]
            r = np.asarray(r)[None]

        unit_vector = rotvec / r[:, None]

        # convert unit vector to angles
        theta = np.arctan2(np.sqrt(unit_vector[:, 0] ** 2 + unit_vector[:, 1] ** 2), unit_vector[:, 2])
        phi = np.arctan2(unit_vector[:, 1], unit_vector[:, 0])
        if rotvec.ndim == 1:
            return np.concatenate((theta, phi, r), axis=-1)  # polar, azimuthal, applied rotation
        else:
            return np.concatenate((theta[:, None], phi[:, None], r[:, None]), axis=-1)  # polar, azimuthal, applied rotation

    elif torch.is_tensor(rotvec):
        r = torch.linalg.norm(rotvec, axis=-1)
        if rotvec.ndim == 1:
            rotvec = rotvec[None, :]
            r = torch.Tensor(r)[None]

        unit_vector = rotvec / r[:, None]

        # convert unit vector to angles
        theta = torch.arctan2(torch.sqrt(unit_vector[:, 0] ** 2 + unit_vector[:, 1] ** 2), unit_vector[:, 2])
        phi = torch.arctan2(unit_vector[:, 1], unit_vector[:, 0])
        if rotvec.ndim == 1:
            return torch.cat((theta, phi, r), dim=-1)  # polar, azimuthal, applied rotation
        else:
            return torch.cat((theta[:, None], phi[:, None], r[:, None]), dim=-1)  # polar, azimuthal, applied rotation

    else:
        print("Array type not supported! Must be np.ndarray or torch.tensor")
        return None


def compute_fractional_transform_torch(cell_lengths, cell_angles):
    """
    compute f->c and c->f transforms as well as cell volume in a vectorized, differentiable way
    """
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


def cell_vol_torch(v: torch.tensor, a: torch.tensor):
    """
    compute the volume of a parallelpiped given basis vector lengths and internal angles [a b c] [alpha beta gamma]
    """
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(a)  # in natural units

    ''' Calculate volume of the unit cell '''
    vol = v[0] * v[1] * v[2] * torch.sqrt(torch.abs(1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]))

    return vol


def compute_Ip_handedness(Ip):
    """
    determine the right or left handedness from the cross products of principal inertial axes
    np.array or torch.tensor input, single or multiple samples
    """
    if isinstance(Ip, np.ndarray):
        if Ip.ndim == 2:
            return np.sign(np.dot(Ip[0], np.cross(Ip[1], Ip[2])).sum())
        elif Ip.ndim == 3:
            return np.sign(np.dot(Ip[:, 0], np.cross(Ip[:, 1], Ip[:, 2], axis=1).T).sum(1))

    elif torch.is_tensor(Ip):
        if Ip.ndim == 2:
            return torch.sign(torch.mul(Ip[0], torch.cross(Ip[1], Ip[2])).sum()).float()
        elif Ip.ndim == 3:
            return torch.sign(torch.mul(Ip[:, 0], torch.cross(Ip[:, 1], Ip[:, 2], dim=1)).sum(1))

    else:
        print("Ip handedness calculation failed! Inputs were neither torch.tensor or numpy.array")
        sys.exit()


def cell_vol(v, a):
    """
    compute cell volume given v=[abc], a=[alpha,beta,gamma]
    with shapes [:,3]
    """
    """ Calculate cos and sin of cell angles """
    cos_a = np.cos(a)  # in natural units

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = v[0] * v[1] * v[2] * np.sqrt(np.abs(val))  # technically a signed quanitity

    return vol


def coor_trans_matrix(opt, v, a, return_vol=False):
    """ Calculate cos and sin of cell angles """  # todo test - enforce this agrees with the torch version
    if np.amax(a) > np.pi:
        print('Warning - large angles! Remember to convert to natural units!')

    cos_a = np.cos(a)
    sin_a = np.sin(a)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = np.sign(val) * v[0] * v[1] * v[2] * np.sqrt(np.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    m = np.zeros((3, 3), dtype=np.float_)
    if opt == 'c_to_f':
        ''' Converting from cartesian to fractional '''
        m[0, 0] = 1.0 / v[0]
        m[0, 1] = -cos_a[2] / v[0] / sin_a[2]
        m[0, 2] = v[1] * v[2] * (cos_a[0] * cos_a[2] - cos_a[1]) / vol / sin_a[2]
        m[1, 1] = 1.0 / v[1] / sin_a[2]
        m[1, 2] = v[0] * v[2] * (cos_a[1] * cos_a[2] - cos_a[0]) / vol / sin_a[2]
        m[2, 2] = v[0] * v[1] * sin_a[2] / vol
    elif opt == 'f_to_c':
        ''' Converting from fractional to cartesian '''
        m[0, 0] = v[0]
        m[0, 1] = v[1] * cos_a[2]
        m[0, 2] = v[2] * cos_a[1]
        m[1, 1] = v[1] * sin_a[2]
        m[1, 2] = v[2] * (cos_a[0] - cos_a[1] * cos_a[2]) / sin_a[2]
        m[2, 2] = vol / v[0] / v[1] / sin_a[2]

    if return_vol:
        return m, np.abs(vol)
    else:
        return m

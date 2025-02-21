import sys

import numpy as np
import torch
from torch import Tensor
from torch_scatter import scatter

from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.models.functions.asymmetric_radius_graph import radius


def compute_principal_axes_np(coords):
    """
    Compute the principal inertial axes for a given set of particle coordinates, ignoring particle mass.
    
    Use our overlap rules to ensure a fixed direction for all axes under almost all circumstances, excepting e.g., certain symmetric molecules.
    
    Parameters
    ----------
    coords

    Returns
    -------
    Ip : np.array(3,3)
        Principal inertial axes.
    Ipm : np.array(3)
        Principal inertial moments
    I : np.array(3)
        Inertial tensor in original frame
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
    max_equivs = np.argwhere(np.round(dists, 8) == np.round(dists[max_ind], 8))[:,
                 0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(np.amin(max_equivs))
    direction = points[max_ind]
    direction = np.divide(direction, np.linalg.norm(direction))
    overlaps = Ip.dot(direction)  # check if the principal components point towards or away from the CoG
    signs = np.sign(overlaps)  # returns zero for zero overlap, but we want it to default to +1 in this case
    signs[signs == 0] = 1

    Ip = (Ip.T * signs).T  # if the vectors have negative overlap, flip the direction
    if np.any(
            np.abs(
                overlaps) < 1e-3):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
        # align the 'good' vectors
        fix_ind = np.argmin(np.abs(overlaps))  # vector with vanishing overlap
        if compute_Ip_handedness(Ip) < 0:  # make sure result is right handed
            Ip[fix_ind] = -Ip[fix_ind]

    return Ip, Ipm, I


def compute_inertial_tensor_torch(x: torch.tensor, y: torch.tensor, z: torch.tensor):
    """
    Compute the principal inertial axes for a given set of particle coordinates, ignoring particle mass.

    Parameters
    ----------
    x : torch.tensor
    y : torch.tensor
    z : torch.tensor

    Returns
    -------
    Ip : np.array(3,3)
        Principal inertial axes.
    Ipm : np.array(3)
        Principal inertial moments
    I : np.array(3)
        Inertial tensor in original frame
    """  # todo harmonize with numpy version - currently disagrees ~0.5% of the time
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
    Compute the principal inertial axes for a given set of particle coordinates, ignoring particle mass.

    Use our overlap rules to ensure a fixed direction for all axes under almost all circumstances, excepting e.g., certain symmetric molecules.

    Parameters
    ----------
    coords : torch.tensor(n,3)
    masses : None
        not used
    return_direction : bool
        whether to add the direction between centroid and most distant coordinate to the output

    Returns
    -------
    Ip : torch.tensor(3,3)
        Principal inertial axes.
    Ipm : torch.tensor(3)
        Principal inertial moments
    I : torch.tensor(3)
        Inertial tensor in original frame

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
    max_equivs = torch.where(dists == dists[max_ind])[
        0]  # torch.where(torch.round(dists, decimals=8) == torch.round(dists[max_ind], decimals=8))[0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(torch.amin(max_equivs))
    direction = coords[max_ind]
    # direction = direction / torch.linalg.norm(direction) # magnitude doesn't matter, only the sign
    overlaps = torch.inner(Ip,
                           direction)  # Ip.dot(direction) # check if the principal components point towards or away from the CoG
    if any(overlaps == 0):  # exactly zero is invalid #
        overlaps[overlaps == 0] = 1e-9
    if any(torch.abs(
            overlaps) < 1e-8):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
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


def batch_molecule_principal_axes_torch(coords_list: list = None, skip_centring=False):
    """
    Parallel computation of principal inertial axes from a list of coordinate lists.

    Parameters
    ----------
    coords_list : list(torch.tensor(n,3))
    skip_centring : bool
        Whether to skip centering each point cloud - e.g., if the input is already centered

    Returns
    -------
    Ip_fin : list(torch.tensor(3,3))
    Ipm_fin : list(torch.tensor(3))
    I : list(torch.tensor(3,3))
    """
    if not skip_centring: # todo accelerate with scatter
        coords_list_centred = [coord - coord.mean(0) for coord in coords_list]
        all_coords = torch.cat(coords_list_centred)
    else:
        all_coords = torch.cat(coords_list)

    batch, ptrs = extract_batching_info(coords_list, all_coords.device) # todo pass batch info as an argument instead calculating here

    Ip, Ipm_fin, I = scatter_compute_Ip(all_coords, batch)

    # cardinal direction is vector from CoM to the farthest atom
    direction = get_cardinal_direction(all_coords, batch, ptrs)
    normed_direction = direction / torch.linalg.norm(direction, dim=1)[:, None]
    overlaps, signs = get_overlaps(Ip, normed_direction)

    Ip_fin = correct_Ip_directions(Ip, overlaps,
                                   signs)  # somehow, fails for mirror planes, on top of symmetric and spherical tops

    return Ip_fin, Ipm_fin, I

    '''  visualize clouds and axes for testing
    import plotly.graph_objects as go
    ind = 16
    x, y, z = all_coords[batch==ind].T.cpu().detach().numpy()
    fig = go.Figure(go.Scatter3d(x=x,y=y,z=z,mode='markers'))
    a, b, c = torch.stack([torch.zeros_like(direction[ind]), direction[ind]]).T.cpu().detach().numpy()
    fig.add_trace(go.Scatter3d(x=a, y=b, z=c))
    for i in range(3):
        a, b, c = torch.stack([torch.zeros_like(direction[ind]), Ip[ind, i]]).T.cpu().detach().numpy()
        fig.add_trace(go.Scatter3d(x=a, y=b, z=c))
    for i in range(3):
        a, b, c = torch.stack([torch.zeros_like(direction[ind]), Ip_fin[ind, i]]).T.cpu().detach().numpy()
        fig.add_trace(go.Scatter3d(x=a, y=b, z=c))
    fig.show(renderer='browser')
    '''


def correct_Ip_directions(Ip, overlaps, signs, overlap_threshold: float = 1e-5):
    """
    Enforce positive overlaps for given inertial principal axes with a given canonical direction, given their overlaps.

    Parameters
    ----------
    Ip : torch.tensor(3,3)
    overlaps : torch.tensor(3)
    signs : torch.tensor(3)
    overlap_threshold : float

    Returns
    -------
    Ip_fin: torch.tensor(3,3)
        Inertial principal axes with positive overlaps to the given canonical direction
    """
    Ip_fin = torch.zeros_like(Ip)
    for ii, Ip_i in enumerate(Ip):
        Ip_i = (Ip_i.T * signs[
            ii]).T  # if the vectors have negative overlap, flip the direction, happens if the cardinal direction is too close to an existing principal axisI
        if any(torch.abs(overlaps[
                             ii]) < overlap_threshold):  # if any overlaps are vanishing (up to 32 bit precision), determine the direction via the RHR (if two overlaps are vanishing, this will not work)
            # enforce right-handedness in the free vector
            fix_ind = torch.argmin(torch.abs(overlaps[ii]))  # vector with vanishing overlap
            if compute_Ip_handedness(Ip_i) < 0:  # make sure result is right-handed
                Ip_i[fix_ind] = -Ip_i[fix_ind]

        Ip_fin[ii] = Ip_i
    return Ip_fin


def get_overlaps(Ip, direction):
    """
    Compute overlaps and signs for given inertial principal axes with a given canonical direction

    Parameters
    ----------
    Ip : torch.tensor(3,3)
    direction : torch.tensor(3)

    Returns
    -------
    overlaps : torch.tensor(3)
    signs : torch.tensor(3)
    """
    overlaps = torch.einsum('nij,nj->ni', (
        Ip, direction))  # Ip.dot(direction) # check if the principal components point towards or away from the CoG
    signs = torch.sign(overlaps)  # we want any exactly zero overlaps to come with positive signs
    signs[signs == 0] = 1
    return overlaps, signs


def get_cardinal_direction(all_coords, batch, ptrs):
    """
    Compute cardinal direction for a list of sets of coordinates, defined as the vector from the centroid to the furthest coordinate.
    Output is not unique for certain symmetric inputs.

    Parameters
    ----------
    all_coords : torch.tensor(n,3)
    batch : torch.tensor(n)
    ptrs : torch.tensor(num_graphs+1)

    Returns
    -------
    direction : torch.tensor(num_graphs, 3)
    """
    dists = torch.linalg.norm(all_coords, axis=1)  # CoM is at 0,0,0
    max_ind = torch.stack(
        [torch.argmax(dists[batch == i]) + ptrs[i] for i in range(len(ptrs) - 1)])  # find the furthest atom in each mol
    direction = all_coords[max_ind]

    return direction


def scatter_compute_Ip(all_coords, batch):
    """
    Parallel function to compute inertial for a list of unequal sized sets of coordinates.

    Parameters
    ----------
    all_coords : torch.tensor(n,3)
    batch : torch.tensor(n)

    Returns
    -------
    Ip : torch.tensor(num_graphs, 3, 3)
    Ipm : torch.tensor(num_graphs, 3)
    I : torch.tensor(num_graphs, 3, 3)
    """
    Ixy = -scatter(all_coords[:, 0] * all_coords[:, 1],
                   batch, reduce='sum')
    Iyz = -scatter(all_coords[:, 1] * all_coords[:, 2],
                   batch, reduce='sum')
    Ixz = -scatter(all_coords[:, 0] * all_coords[:, 2],
                   batch, reduce='sum')
    Ixx = scatter(all_coords[:, 1] ** 2 + all_coords[:, 2] ** 2,
                  batch, reduce='sum')
    Iyy = scatter(all_coords[:, 0] ** 2 + all_coords[:, 2] ** 2,
                  batch, reduce='sum')
    Izz = scatter(all_coords[:, 0] ** 2 + all_coords[:, 1] ** 2,
                  batch, reduce='sum')

    inertial_tensor = torch.cat(
        (torch.vstack((Ixx, Ixy, Ixz))[:, None, :].permute(2, 1, 0),
         torch.vstack((Ixy, Iyy, Iyz))[:, None, :].permute(2, 1, 0),
         torch.vstack((Ixz, Iyz, Izz))[:, None, :].permute(2, 1, 0)
         ), dim=-2)  # inertial tensor

    Ipm, Ip = torch.linalg.eig(inertial_tensor)  # principal inertial tensor
    Ipm, Ip = torch.real(Ipm), torch.real(Ip)

    Ip = Ip.permute(0, 2, 1)  # switch to row-wise eigenvectors

    sort_inds = torch.argsort(Ipm, dim=1)
    Ipm = torch.stack([Ipm[i, sort_inds[i]] for i in range(len(sort_inds))])
    Ip = torch.stack([Ip[i][sort_inds[i]] for i in range(len(sort_inds))])  # sort also the eigenvectors

    return Ip, Ipm, inertial_tensor


def extract_batching_info(nodes_list, device='cpu'):
    """
    Extract batch and ptr info from a list of sets of coordinates.

    Parameters
    ----------
    nodes_list : list(torch.tensor(n,3)) with different n throughout
    devide : str

    Returns
    -------
    batch : torch.tensor(num_nodes)
    ptr : torch.tensor(num_graphs + 1)
    """
    ptrs = [0]
    ptrs.extend([len(coord) for coord in nodes_list])
    ptrs = torch.tensor(ptrs, dtype=torch.int, device=device).cumsum(0)
    batch = torch.cat(
        [(i - 1) * torch.ones(ptrs[i] - ptrs[i - 1], dtype=torch.int64, device=device) for i in range(1, len(ptrs))])
    return batch, ptrs


def sph2rotvec(angles):
    """
    Transform from axis-angle in polar coordinates to rotation vector

    Parameters
    ----------
    angles : (nx3)
        theta, phi, r

    Returns
    -------
    rotvec : (nx3)
        x, y, z

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
    to spherical coordinates theta, phi and r=||rotvec||

    Parameters
    ----------
    rotvec : (nx3)
        x, y, z

    Returns
    -------
    angles : (nx3)
        theta, phi, r

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
            return np.concatenate((theta[:, None], phi[:, None], r[:, None]),
                                  axis=-1)  # polar, azimuthal, applied rotation

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

    Parameters
    ----------
    cell_lengths : torch.tensor(nx3)
        a, b, c
    cell_angles : torch.tensor(nx3)
        alpha, beta, gamma

    Returns
    -------
        fc_transform : torch.tensor(n,3,3)
        cf_transform : torch.tensor(n,3,3)
        cell_volumes : torch.tensor(n)
    """
    cos_a = torch.cos(cell_angles)
    sin_a = torch.sin(cell_angles)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[:, 0] ** 2 - cos_a[:, 1] ** 2 - cos_a[:, 2] ** 2 + 2.0 * cos_a[:, 0] * cos_a[:, 1] * cos_a[:, 2]

    vol = torch.sign(val) * torch.prod(cell_lengths, dim=1) * torch.sqrt(
        torch.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    T_fc_list = torch.zeros((len(cell_lengths), 3, 3), device=cell_lengths.device, dtype=cell_lengths.dtype)
    T_cf_list = torch.zeros((len(cell_lengths), 3, 3), device=cell_lengths.device, dtype=cell_lengths.dtype)

    ''' Converting from cartesian to fractional '''
    T_cf_list[:, 0, 0] = 1.0 / cell_lengths[:, 0]
    T_cf_list[:, 0, 1] = -cos_a[:, 2] / cell_lengths[:, 0] / sin_a[:, 2]
    T_cf_list[:, 0, 2] = cell_lengths[:, 1] * cell_lengths[:, 2] * (
            cos_a[:, 0] * cos_a[:, 2] - cos_a[:, 1]) / vol / sin_a[:, 2]
    T_cf_list[:, 1, 1] = 1.0 / cell_lengths[:, 1] / sin_a[:, 2]
    T_cf_list[:, 1, 2] = cell_lengths[:, 0] * cell_lengths[:, 2] * (
            cos_a[:, 1] * cos_a[:, 2] - cos_a[:, 0]) / vol / sin_a[:, 2]
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
    compute the volume of a parallelpiped given basis vector lengths and internal angles
    Parameters
    ----------
    v : torch.tensor(3)
        [a b c]
    a : torch.tensor(3)
        [alpha beta gamma]

    Returns
    -------
    cell_volume : float

    """
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(a)  # in natural units

    ''' Calculate volume of the unit cell '''
    vol = v[0] * v[1] * v[2] * torch.sqrt(
        torch.abs(1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]))

    return vol


def compute_Ip_handedness(Ip):
    """
    determine the right or left handedness from the cross products of principal inertial axes
    np.array or torch.tensor input, single or multiple samples


    Parameters
    ----------
    Ip : (opt n, 3, 3)
        principal inertial tensor

    Returns
    -------
    handedness : (n)
        +/- 1, the handedness of the cross products of principal inertial axes
    """
    if isinstance(Ip, np.ndarray):
        if Ip.ndim == 2:
            return np.sign(np.dot(Ip[0], np.cross(Ip[1], Ip[2])).sum())
        elif Ip.ndim == 3:
            return np.sign(np.dot(Ip[:, 0], np.cross(Ip[:, 1], Ip[:, 2], axis=1).T).sum(1))

    elif torch.is_tensor(Ip):
        if Ip.ndim == 2:
            return torch.sign(torch.mul(Ip[0], torch.cross(Ip[1], Ip[2], dim=0)).sum()).float()
        elif Ip.ndim == 3:
            return torch.sign(torch.mul(Ip[:, 0], torch.cross(Ip[:, 1], Ip[:, 2], dim=1)).sum(1))

    else:
        print("Ip handedness calculation failed! Inputs were neither torch.tensor or numpy.array")
        sys.exit()


def cell_vol_np(v, a):
    """
    compute the volume of a parallelpiped given basis vector lengths and internal angles
    Parameters
    ----------
    v : np.array(3)
        [a b c]
    a : np.array(3)
        [alpha beta gamma]

    Returns
    -------
    cell_volume : float

    """
    """ Calculate cos and sin of cell angles """
    cos_a = np.cos(a)  # in natural units

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = v[0] * v[1] * v[2] * np.sqrt(np.abs(val))  # technically a signed quanitity

    return vol


def coor_trans_matrix_np(opt, v, a, return_vol=False):
    """
    compute f->c and c->f transforms as well as cell volume in a vectorized, differentiable way

    Parameters
    ----------
    opt : str 'c_to_f' or 'f_to_c'
        which direction to transform between fractional and cartesian
    v : np.array(3)
        a, b, c
    a : np.array(3)
        alpha, beta, gamma
    return_vol : bool, optional
        return the absolute value of the cell volume

    Returns
    -------
        transform : np.array(3,3)
        cell_volumes : float
    """
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


def mol_batch_vdW_volume(mol_batch):
    """
    wrapper for batch_compute_vdW_volume
    """
    return batch_molecule_vdW_volume(
        mol_batch.z,
        mol_batch.pos,
        mol_batch.batch,
        mol_batch.num_graphs,
        torch.FloatTensor(list(VDW_RADII.values()),
                          device=mol_batch.z.device))


def batch_molecule_vdW_volume(
        atom_types_in: torch.LongTensor,
        pos: torch.FloatTensor,
        batch: torch.LongTensor,
        num_graphs: int,
        vdw_radii_tensor: torch.FloatTensor
):
    if atom_types_in.ndim > 1:
        atom_types = atom_types_in[:, 0]
    else:
        atom_types = atom_types_in.clone()

    atom_volumes = 4 / 3 * torch.pi * vdw_radii_tensor[atom_types] ** 3
    raw_vdw_volumes = scatter(atom_volumes, batch, dim=0, dim_size=num_graphs, reduce='sum')
    bonds_i, bonds_j = radius(pos, pos,
                              r=2 * vdw_radii_tensor.max(),
                              batch_x=batch,
                              batch_y=batch,
                              max_num_neighbors=6)
    mask = ~(bonds_i >= bonds_j)  # eliminate duplicates
    bonds_i, bonds_j = bonds_i[mask], bonds_j[mask]
    bond_lengths = torch.linalg.norm(pos[bonds_i] - pos[bonds_j], dim=1)
    radii_i, radii_j = vdw_radii_tensor[atom_types[bonds_i]], vdw_radii_tensor[atom_types[bonds_j]]
    # https://mathworld.wolfram.com/Sphere-SphereIntersection.html
    sphere_overlaps = (torch.pi * (radii_i + radii_j - bond_lengths) ** 2 *
                       (bond_lengths ** 2 + 2 * bond_lengths * radii_j - 3 * radii_j ** 2
                        + 2 * bond_lengths * radii_i + 6 * radii_j * radii_i - 3 * radii_i ** 2) / (12 * bond_lengths))
    sphere_overlaps[bond_lengths > (radii_i + radii_j)] = 0
    molwise_sphere_overlaps = scatter(sphere_overlaps, batch[bonds_i], dim=0, dim_size=num_graphs,
                                      reduce='sum')
    corrected_mol_volume = raw_vdw_volumes - molwise_sphere_overlaps
    return corrected_mol_volume


def grid_compute_molecule_volume(atom_types, pos, vdw_radii_tensor, eps):
    """
    brute force grid approach to computing vdW volume for a single molecule
    Parameters
    ----------
    atom_types
    pos
    vdw_radii_tensor

    Returns
    -------

    """
    convergence_history = []
    dx = 0.1
    converged = False
    ind = -1
    max_iters = 10
    while converged is False and ind < max_iters:
        dx *= 0.75
        xmin, ymin, zmin = (pos.amin(0) - vdw_radii_tensor.amax())
        xmax, ymax, zmax = (pos.amax(0) + vdw_radii_tensor.amax())
        num_x = int((xmax - xmin) / dx)
        num_y = int((ymax - ymin) / dx)
        num_z = int((zmax - zmin) / dx)
        grid = torch.meshgrid(torch.linspace(xmin, xmax, num_x),
                              torch.linspace(ymin, ymax, num_y),
                              torch.linspace(zmin, zmax, num_z),
                              indexing='xy'
                              )
        grid = torch.stack(grid)
        grid_flat = grid.reshape(3, num_x * num_y * num_z).T

        edges = radius(x=grid_flat, y=pos, r=vdw_radii_tensor.amax(),
                       max_num_neighbors=int(1e12))

        dists = torch.linalg.norm(pos[edges[0]] - grid_flat[edges[1]], dim=1)
        close_enough = dists <= vdw_radii_tensor[atom_types[edges[0]]]

        overlapped_points = len(edges[1, close_enough].unique())
        box_volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        overlapping_fraction = overlapped_points / len(grid_flat)
        occupied_volume = box_volume * overlapping_fraction
        convergence_history.append(float(occupied_volume))
        ind += 1
        if ind > 1:
            conv = abs(convergence_history[-2] - convergence_history[-1]) / convergence_history[1]
            if conv < eps:
                converged = True

            print(ind)
            print(conv)

    return occupied_volume


def grid_compute_molecule_volume_pointwise(atom_types, pos, vdw_radii_tensor, eps):
    """
    brute force grid approach to computing vdW volume for a single molecule
    Parameters
    ----------
    atom_types
    pos
    vdw_radii_tensor

    Returns
    -------

    """
    convergence_history = []
    dx = 0.1
    converged = False
    ind = -1
    max_iters = 10
    while converged is False and ind < max_iters:
        dx *= 0.75
        xmin, ymin, zmin = (pos.amin(0) - vdw_radii_tensor.amax())
        xmax, ymax, zmax = (pos.amax(0) + vdw_radii_tensor.amax())
        num_x = int((xmax - xmin) / dx)
        num_y = int((ymax - ymin) / dx)
        num_z = int((zmax - zmin) / dx)
        grid = torch.meshgrid(torch.linspace(xmin, xmax, num_x),
                              torch.linspace(ymin, ymax, num_y),
                              torch.linspace(zmin, zmax, num_z),
                              indexing='xy'
                              )
        grid = torch.stack(grid)
        grid_flat = grid.reshape(3, num_x * num_y * num_z).T

        edges = radius(x=grid_flat, y=pos, r=vdw_radii_tensor.amax(),
                       max_num_neighbors=int(1e12))

        dists = torch.linalg.norm(pos[edges[0]] - grid_flat[edges[1]], dim=1)
        close_enough = dists <= vdw_radii_tensor[atom_types[edges[0]]]

        overlapped_points = len(edges[1, close_enough].unique())
        box_volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        overlapping_fraction = overlapped_points / len(grid_flat)
        occupied_volume = box_volume * overlapping_fraction
        convergence_history.append(float(occupied_volume))
        ind += 1
        if ind > 1:
            conv = abs(convergence_history[-2] - convergence_history[-1]) / convergence_history[1]
            if conv < eps:
                converged = True

            print(ind)
            print(conv)

    return occupied_volume


def batch_compute_normed_cell_vectors(data):
    return data.cell_lengths / torch.pow(data.sym_mult[:, None] * data.mol_volume[:, None], 1 / 3)


def batch_denorm_cell_vectors(normed_cell_vecs, data):
    return normed_cell_vecs * torch.pow(data.sym_mult[:, None] * data.mol_volume[:, None], 1 / 3)


def norm_circular_components(components: torch.tensor):
    """
    Use Pythagoras to norm the sum of squares to the unit circle.
    Parameters
    ----------
    components : torch.tensor(n, 2)

    Returns
    -------
    normed_components : torch.tensor(n, 2)
    """

    return components / torch.sqrt(torch.sum(components ** 2, dim=-1))[:, None]


def components2angle(components: torch.tensor, norm_components=True):
    """
    Take two non-normalized components[n, 2] representing sin(angle) and cos(angle), compute the resulting angle,
    following     https://ai.stackexchange.com/questions/38045/how-can-i-encode-angle-data-to-train-neural-networks

    Optionally norm the sum of squares - doesn't appear to do much though.

    Parameters
    ----------
    components : torch.tensor(n, 2)
    norm_components : bool, optional

    Returns
    -------
    angles : torch.tensor(n, 2)
    """

    if norm_components:
        normed_components = norm_circular_components(components)
        angles = torch.atan2(normed_components[:, 0], normed_components[:, 1])
    else:
        angles = torch.atan2(components[:, 0], components[:, 1])

    return angles


def angle2components(angle: torch.tensor):
    """
    Tecompose an angle input into sin(angle) and cos(angle)

    Parameters
    ----------
    angle : torch.tensor(n)

    Returns
    -------
    sin(angle), cos(angle) : torch.tensor, torch.tensor
    """

    return torch.cat((torch.sin(angle)[:, None], torch.cos(angle)[:, None]), dim=1)


def enforce_crystal_system(lattice_lengths, lattice_angles, sg_inds, symmetries_dict):
    """
    enforce physical bounds on cell parameters
    https://en.wikipedia.org/wiki/Crystal_system
    """  # todo double check these limits

    lattices = [symmetries_dict['lattice_type'][int(sg_inds[n])] for n in range(len(sg_inds))]

    pi_tensor = torch.ones_like(lattice_lengths[0, 0]) * torch.pi

    fixed_lengths = torch.zeros_like(lattice_lengths)
    fixed_angles = torch.zeros_like(lattice_angles)

    for i in range(len(lattice_lengths)):
        lengths = lattice_lengths[i]
        angles = lattice_angles[i]
        lattice = lattices[i]
        # enforce agreement with crystal system
        if lattice.lower() == 'triclinic':  # anything goes
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = angles * 1

        elif lattice.lower() == 'monoclinic':  # fix alpha and gamma to pi/2
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, angles[1], pi_tensor.clone() / 2,
            ), dim=- 1)
        elif lattice.lower() == 'orthorhombic':  # fix all angles at pi/2
            fixed_lengths[i] = lengths * 1
            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)
        elif lattice.lower() == 'tetragonal':  # fix all angles pi/2 and take the mean of a & b vectors
            mean_tensor = torch.mean(lengths[0:2])
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, lengths[2] * 1,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)

        elif lattice.lower() == 'hexagonal':
            # mean of ab, c is free
            # alpha beta are pi/2, gamma is 2pi/3

            mean_tensor = torch.mean(lengths[0:2])
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, lengths[2] * 1,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() * 2 / 3,
            ), dim=- 1)

        # elif lattice.lower()  == 'trigonal':

        elif lattice.lower() == 'rhombohedral':
            # mean of abc vector lengths
            # mean of all angles

            mean_tensor = torch.mean(lengths)
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, mean_tensor,
            ), dim=- 1)

            mean_angle = torch.mean(angles)
            fixed_angles[i] = torch.stack((
                mean_angle, mean_angle, mean_angle,
            ), dim=- 1)

        elif lattice.lower() == 'cubic':  # all angles 90 all lengths equal
            mean_tensor = torch.mean(lengths)
            fixed_lengths[i] = torch.stack((
                mean_tensor, mean_tensor, mean_tensor,
            ), dim=- 1)

            fixed_angles[i] = torch.stack((
                pi_tensor.clone() / 2, pi_tensor.clone() / 2, pi_tensor.clone() / 2,
            ), dim=- 1)
        else:
            print(lattice + ' is not a valid crystal lattice!')
            sys.exit()

    return fixed_lengths, fixed_angles


def cell_parameters_to_box_vectors(opt: str,
                                   cell_lengths: torch.tensor,
                                   cell_angles: torch.tensor,
                                   return_vol: bool = False):
    """  # TODO I believe this is a duplicate function
    Initially borrowed from Nikos
    Quickly convert from cell lengths and angles to fractional transform matrices fractional->cartesian or cartesian->fractional
    """
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(cell_angles)
    sin_a = torch.sin(cell_angles)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = torch.sign(val) * cell_lengths[0] * cell_lengths[1] * cell_lengths[2] * torch.sqrt(
        torch.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    m = torch.zeros((3, 3))
    if opt == 'c_to_f':
        ''' Converting from cartesian to fractional '''
        m[0, 0] = 1.0 / cell_lengths[0]
        m[0, 1] = -cos_a[2] / cell_lengths[0] / sin_a[2]
        m[0, 2] = cell_lengths[1] * cell_lengths[2] * (cos_a[0] * cos_a[2] - cos_a[1]) / vol / sin_a[2]
        m[1, 1] = 1.0 / cell_lengths[1] / sin_a[2]
        m[1, 2] = cell_lengths[0] * cell_lengths[2] * (cos_a[1] * cos_a[2] - cos_a[0]) / vol / sin_a[2]
        m[2, 2] = cell_lengths[0] * cell_lengths[1] * sin_a[2] / vol
    elif opt == 'f_to_c':
        ''' Converting from fractional to cartesian '''
        m[0, 0] = cell_lengths[0]
        m[0, 1] = cell_lengths[1] * cos_a[2]
        m[0, 2] = cell_lengths[2] * cos_a[1]
        m[1, 1] = cell_lengths[1] * sin_a[2]
        m[1, 2] = cell_lengths[2] * (cos_a[0] - cos_a[1] * cos_a[2]) / sin_a[2]
        m[2, 2] = vol / cell_lengths[0] / cell_lengths[1] / sin_a[2]

    # todo create m in a single-step
    if return_vol:
        return m, torch.abs(vol)
    else:
        return m


def compute_mol_radius(coords: torch.FloatTensor) -> Tensor:
    """
    compute centroid for each molecule
    then the distance of all atoms to the centroid
    most distant atom defines the 'radius'
    """
    centroid = coords.mean(0)
    return torch.amax(torch.linalg.norm(coords - centroid, dim=-1))


def batch_compute_mol_radius(coords: torch.FloatTensor,
                             batch: torch.LongTensor,
                             num_graphs: int,
                             nodes_per_graph: torch.LongTensor) -> Tensor:
    """
    compute centroid for each molecule
    then the distance of all atoms to the centroid
    most distant atom defines the 'radius'
    """
    centroids = get_batch_centroids(coords, batch, num_graphs)
    dists = torch.linalg.norm(coords - centroids.repeat_interleave(nodes_per_graph, 0), dim=-1)
    return scatter(dists, batch, dim=0, dim_size=num_graphs, reduce='max')


def get_batch_centroids(coords: torch.FloatTensor,
                        batch: torch.LongTensor,
                        num_graphs: int,
                        ) -> Tensor:
    return scatter(coords, batch, dim=0, dim_size=num_graphs, reduce='mean')


def batch_compute_mol_mass(z: torch.LongTensor,
                           batch: torch.LongTensor,
                           masses_tensor: torch.FloatTensor,
                           num_graphs: int) -> Tensor:
    return scatter(masses_tensor[z], batch, dim=0, dim_size=num_graphs, reduce='sum')


def compute_mol_mass(z: torch.LongTensor,
                     masses_tensor: torch.FloatTensor) -> Tensor:
    return torch.sum(masses_tensor[z])

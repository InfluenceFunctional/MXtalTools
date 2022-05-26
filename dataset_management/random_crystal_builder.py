import numpy as np
#import numba as nb
from nikos.rotations import rotation_matrix_from_vectors, euler_rotation, rodriguez_rotation
from utils import compute_principal_axes_np


#@nb.jit(nopython=True)
def randomize_molecule_position_and_orientation(coords, weights, T_fc, set_position=None, set_rotation = None):
    '''
    :param coords:
    :param weights:
    :param T_fc:
    :param set_position:
    :param set_orientation:
    :param set_rotation:
    :return:
    '''
    # random direction & rotation
    if set_rotation is not None:
        new_rotation = np.asarray(set_rotation,dtype=float)
    else:
        new_rotation = np.random.uniform(-1, 1, size=3)
    if set_position is not None:
        new_centroid_frac = np.asarray(set_position,dtype=float)
    else:
        new_centroid_frac = np.random.uniform(0, 1, size=(3))

    # center coordinates on the center of mass
    CoM = coords.T.dot(weights) / np.sum(weights)
    coords -= CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(weights, coords, np.zeros(3)) # third row of the Ip_axes matrix is the principal moment axis

    #1. align I1 to (1,1,1)
    normed_corner_vector = T_fc.dot(np.ones(3))
    normed_corner_vector /= np.linalg.norm(normed_corner_vector)
    rot_mat = rotation_matrix_from_vectors(Ip_axes[-1], normed_corner_vector) # align I1 to (1,1,1)
    coords = (rot_mat.dot(coords.T)).T # apply rotation matrix (add and subtract CoM if not already done)    #coords = euler_rotation(rot_mat, coords)
    Ip_axes2, Ip_moments2, I_tensor2 = compute_principal_axes_np(weights, coords, np.zeros(3)) # third row of the Ip_axes matrix is the principal moment axis
    # Ip_axes2 = rot_mat.dot(Ip_axes.T).T or this way - not exact but pretty close

    #2. rotate I2 to align with the perpendicular vector between (1,1,1)-(0,0,0) and (0,1,1)
    #a) find the point on (1,1,1)-(0,0,0) closest to (0,1,1)
    corner_point = np.array((0,1,1))
    t0 = T_fc.dot(corner_point).dot(normed_corner_vector)/(normed_corner_vector.dot(normed_corner_vector))
    P0 = normed_corner_vector * t0 # point nearest to (0,1,1)
    rot_alignment_vec = T_fc.dot(corner_point) - P0 # vector between two points
    #b) get angle between I2 and the corner vector
    I2_alignment_angle = np.arccos(np.dot(rot_alignment_vec, Ip_axes2[1])/np.linalg.norm(rot_alignment_vec))
    #c) execute the rotation about I1 - now I1 is aligned to (1,1,1)-(0,0,0) and I2 is pointed from this line, to the corner (0,1,1)
    coords = rodriguez_rotation(Ip_axes2[-1], coords, np.prod((I2_alignment_angle,180)))
    #Ip_axes3, Ip_moments3, I_tensor3 = compute_principal_axes_np(weights, aligned_coords, CoM) # third row of the Ip_axes matrix is the principal moment axis

    '''
    molecule is now 'set' to a 'standard' position
    '''

    #3. rotate to align with an arbitrary vector in the fractional basis
    new_direction = T_fc.dot(new_rotation) # new vector is A*a + B*b + C*c in the frac basis
    new_direction /= np.linalg.norm(new_direction)
    new_rot_mat = rotation_matrix_from_vectors(normed_corner_vector, new_direction) # align molecule (which is pointed at (1,1,1)) to new random direction
    coords = (new_rot_mat.dot(coords.T)).T

    #4. move centroid to the given coordinate
    coords = coords - np.average(coords, axis=0) + T_fc.dot(new_centroid_frac)
    return coords


def old_randomize_molecule_position_and_orientation(coords, weights, T_fc, set_position=None, set_orientation=None, set_rotation = None):
    '''
    # todo define rodriguez rotation against some reference axis
    :param coords:
    :param weights:
    :param T_fc:
    :param set_position:
    :param set_orientation:
    :param set_rotation:
    :return:
    '''
    # random direction & rotation
    if set_orientation is not None:
        new_orientation = np.asarray(set_orientation,dtype=float)
    else:
        new_orientation = np.random.uniform(-1, 1, size=3)
    if set_rotation is not None:
        new_rotation = np.asarray(set_rotation,dtype=float)
    else:
        new_rotation = np.random.uniform(-1, 1, size=1)
    if set_position is not None:
        new_centroid_frac = np.asarray(set_position,dtype=float)
    else:
        new_centroid_frac = np.random.uniform(0, 1, size=(3))


    CoM = coords.T.dot(weights) / np.sum(weights)
    coords -= CoM
    Ip_axes, Ip_moments, I_tensor = compute_principal_axes_np(weights, coords, CoM) # third row of the Ip_axes matrix is the principal moment axis

    # align molecule principal axis to new orientation
    rot_mat = rotation_matrix_from_vectors(Ip_axes[-1], T_fc.dot(np.eye(3)[0] - new_orientation)) # define as difference from a vector
    coords = (rot_mat.dot((coords).T)).T # apply rotation matrix (add and subtract CoM if not already done)    #coords = euler_rotation(rot_mat, coords)

    # rotate about the principal axis by theta
    # todo would be nice if this was defined against some standard axis, but it's not obvious to me
    coords = rodriguez_rotation(Ip_axes[-1], coords, np.prod((new_rotation, 180)))

    # move centroid to new location
    new_centroid_cart = T_fc.dot(new_centroid_frac) #np.transpose(np.dot(T_fc, np.transpose(new_centroid_frac)))
    coords = coords - np.average(coords, axis=0) + new_centroid_cart

    return coords

#@nb.jit(nopython=True)
def build_random_crystal(T_cf, T_fc, coords, affine_ops, z_value):
    '''
    generate a random unit cell with appropriate general position point symmetries
    # ignores special positions
    '''

    # apply point symmetry to generate the reference cell
    coords_f = (T_cf.dot(coords.T)).T #np.transpose(np.dot(T_cf, np.transpose(coords)))  # go to fractional coordinates
    points = coords_f.copy()
    affine_points = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)

    # pattern unit cell via symmetry ops
    cell_coords_f = np.zeros((z_value, len(coords_f), 3))
    cell_coords_c = np.zeros_like(cell_coords_f)
    centroids = np.zeros((z_value, 3))
    for zv in range(z_value):
        cell_coords_f[zv, :, :] = (affine_ops[zv].dot(affine_points.T)).T[:,:-1]#np.inner(affine_points, affine_ops[zv])[..., :-1] # copied from pyxtal #sym_ops[zv].operate_multi(coords_f)
        cell_coords_f[zv, :, :] -= np.floor(np.average(cell_coords_f[zv, :, :], axis=0))  # ensures all copies in one unit cell
        centroids[zv] = np.average(cell_coords_f[zv],axis=0)
        cell_coords_c[zv, :, :] = (T_fc.dot(cell_coords_f[zv].T)).T #np.transpose(np.dot(T_fc, np.transpose(cell_coords_f[zv, :, :])))

    #assert (np.amax(centroids < 1)) and (np.amin(centroids) > 0), "Molecules must be inside the unit cell!" # assert everyone is in the unit cell

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

    return cell_coords_c, cell_coords_f
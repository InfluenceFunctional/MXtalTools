from ase import Atoms
from ase.visualize import view
import numpy as np
from scipy.spatial.transform import Rotation
from utils import compute_principal_axes_np
import ase.build as build
from ase.collections import g2
from scipy.stats import ortho_group

# posarray = np.array([[1.28477144, -1.26507129, -0.55453128],
#                      [0.35840929, -0.66224127, -0.05693115],
#                      [-0.56880893, -1.29130714, 0.822312],
#                      [-0.18914199, -3.58097673, 1.25617464],
#                      [-0.94028036, 1.53608994, 0.27086882],
#                      [0.04613271, 0.71944163, -0.26227612],
#                      [0.79324279, 1.49776457, -1.10591248],
#                      [-0.77020557, 2.80056112, -0.26714171],
#                      [-0.31828734, -2.50219261, 1.01585411],
#                      [0.30416796, 2.74793178, -1.11841683]])
# symbols = 'OCN2C2NCNC'
# mol = Atoms(symbols=symbols, positions=posarray)  # build.molecule('methylenecyclopropane')

# posarray = np.array([[0.05050699, -0.66897398, -0.82410899],
#                      [-1.09735445, -0.32528872, 0.15859318],
#                      [-0.82184683, -1.02614946, 1.5195327],
#                      [0.60591604, -1.57188107, 1.32039934],
#                      [0.99576161, -1.43483634, 0.05023969],
#                      [-1.06336063, 1.22608374, 0.18484812],
#                      [0.26445399, 1.64446489, 0.78096727],
#                      [1.21354359, 1.3780486, -0.10958408],
#                      [0.57329296, 0.77174978, -1.3422823],
#                      [-0.8333958, 1.5222743, -1.32411768],
#                      [-1.9706379, -2.0304721, 1.89061818],
#                      [-3.33092968, -1.31135086, 1.78347236],
#                      [-1.88604061, -2.50875731, 3.347458],
#                      [-1.95764026, -3.2375259, 0.95337325],
#                      [1.57755419, -2.04265651, 2.41668246],
#                      [1.32550468, -1.28684349, 3.73055775],
#                      [1.52070211, -3.56334011, 2.63740532],
#                      [3.02522127, -1.71282601, 2.00218563],
#                      [1.4602457, 0.68708107, -2.61935031],
#                      [0.64108012, 0.2643696, -3.84944712],
#                      [2.569508, -0.36847404, -2.44120534],
#                      [2.19375061, 2.01057804, -2.89109372],
#                      [-1.12557526, 3.00590043, -1.73256028],
#                      [-2.57344921, 3.31721809, -1.29248679],
#                      [-0.22407024, 4.07076726, -1.0904315],
#                      [-1.132741, 3.1908401, -3.25966512]])
# symbols = 'C26'

posarray = np.array([[-3.47213355e-01, -2.13093782e+00, 2.95222219e+00],
                     [3.47213355e-01, 2.13093782e+00, -2.95222219e+00],
                     [1.31889463e-16, 2.07807343e-16, -7.44658615e-17],
                     [2.00761030e-01, -1.92048213e+00, 1.35669053e+00],
                     [-2.00761030e-01, 1.92048213e+00, -1.35669053e+00]])
symbols = 'C2HgS2'
mol = Atoms(symbols=symbols, positions=posarray)  # build.molecule('methylenecyclopropane')

coords = mol.positions
numbers = mol.get_atomic_numbers()
masses = mol.get_masses()

for i in range(1000):
    rand_rot = np.random.uniform(-0.05 * np.pi, 0.05 * np.pi, size=3)

    # rotate to a shared target
    # init two positions
    # position 1
    coords -= coords.T.dot(masses) / masses.sum()
    Ip1, _, _, D1 = compute_principal_axes_np(masses, coords, return_direction=True)
    mol1 = Atoms(symbols=symbols + 'Cl', positions=np.concatenate((coords, D1[None, :] * 5)), cell=Ip1 * 10)  # [10,10,10,90,90,90])

    # position 2
    prep_rot = Rotation.from_euler('XYZ', rand_rot)
    rot_coords = prep_rot.apply(coords)  # np.array((coords[:,0],coords[:,2],coords[:,1])).T
    rot_coords -= rot_coords.T.dot(masses) / masses.sum()
    I2_trans = prep_rot.apply(Ip1)
    Ip2, _, _, D2 = compute_principal_axes_np(masses, rot_coords, return_direction=True)
    mol2 = Atoms(symbols=symbols + 'Cl', positions=np.concatenate((rot_coords, D2[None, :] * 5)), cell=Ip2 * 10)  # [10,10,10,90,90,90])

    # alignment matrix
    # alignment_matrix = np.array(((0,1,1),(1,1,0)))#,(1,0,1)))
    alignment_matrix = ortho_group.rvs(dim=3)

    rot1, rmsd1 = Rotation.align_vectors(a=alignment_matrix[1:], b=Ip1[1:])
    rot2, rmsd2 = Rotation.align_vectors(a=alignment_matrix[1:], b=Ip2[1:])
    if rmsd1 > 0.1:
        print('big rmsd1 error of {}'.format(rmsd1))
    if rmsd2 > 0.1:
        print('big rmsd2 error of {}'.format(rmsd2))

    # rot1,rmsd1 = Rotation.align_vectors(a=alignment_matrix, b=Ip1)
    # rot2,rmsd2 = Rotation.align_vectors(a=alignment_matrix, b=Ip2)

    coords_std = rot1.apply(coords)
    rot_coords_std = rot2.apply(rot_coords)
    Ip1_std, _, _, D1_std = compute_principal_axes_np(masses, coords_std, return_direction=True)
    Ip2_std, _, _, D2_std = compute_principal_axes_np(masses, rot_coords_std, return_direction=True)
    I1_std = rot1.apply(Ip1)
    I2_std = rot2.apply(Ip2)

    mol1_std = Atoms(symbols=symbols + 'Cl', positions=np.concatenate((coords_std, D1_std[None, :] * 5)), cell=Ip1_std * 10)  # )[10,10,10,90,90,90])
    mol2_std = Atoms(symbols=symbols + 'Cl', positions=np.concatenate((rot_coords_std, D2_std[None, :] * 5)), cell=Ip2_std * 10)  # [10,10,10,90,90,90])

    error = np.sum(np.abs(rot_coords_std - coords_std))
    print("Error was {:.3f}".format(error))
    if error > 1:
        points = (coords - coords.T.dot(masses) / np.sum(masses))
        x, y, z = points.T
        Ixx = np.sum(masses * (y ** 2 + z ** 2))
        Iyy = np.sum(masses * (x ** 2 + z ** 2))
        Izz = np.sum(masses * (x ** 2 + y ** 2))
        Ixy = -np.sum(masses * x * y)
        Iyz = -np.sum(masses * y * z)
        Ixz = -np.sum(masses * x * z)
        I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])  # inertial tensor
        Ipm, Ip = np.linalg.eig(I)  # principal inertial tensor
        Ipm, Ip = np.real(Ipm), np.real(Ip)
        sort_inds = np.argsort(Ipm)
        Ipm = Ipm[sort_inds]
        Ip = Ip.T[sort_inds]  # want eigenvectors to be sorted row-wise (rather than column-wise)
        Ip1_orig = Ip.copy()

        # cardinal direction is vector from CoM to farthest atom
        dists = np.linalg.norm(points, axis=1)
        max_ind = np.argmax(dists)
        max_equivs = np.argwhere(np.round(dists,8) == np.round(dists[max_ind],8))[:,0] # if there are multiple equidistant atoms - pick the one with the lowest index
        max_ind = int(np.amin(max_equivs))
        direction = points[max_ind]
        direction /= np.linalg.norm(direction)
        overlaps = Ip.dot(direction)  # check if the principal components point towards or away from the CoG
        if any(overlaps == 0):  # exactly zero is invalid
            overlaps[overlaps == 0] = 1
        if any(overlaps < 1e-8):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
            # align the 'good' vectors
            Ip = (Ip.T * np.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction
            fix_ind = np.argmin(np.abs(overlaps))
            other_vectors = np.delete(np.arange(3), fix_ind)
            check_direction = np.cross(Ip[other_vectors[0]], Ip[other_vectors[1]])
            # align the 'bad' vector
            Ip[fix_ind] *= np.sign(np.dot(check_direction, Ip[fix_ind]))
        else:
            Ip = (Ip.T * np.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction

        test_Ip1 = Ip  # if the vectors have negative overlap, flip the direction
        direction1 = direction.copy()
        overlaps1 = overlaps.copy()

        points = (rot_coords - rot_coords.T.dot(masses) / np.sum(masses))
        x, y, z = points.T
        Ixx = np.sum(masses * (y ** 2 + z ** 2))
        Iyy = np.sum(masses * (x ** 2 + z ** 2))
        Izz = np.sum(masses * (x ** 2 + y ** 2))
        Ixy = -np.sum(masses * x * y)
        Iyz = -np.sum(masses * y * z)
        Ixz = -np.sum(masses * x * z)
        I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])  # inertial tensor
        Ipm, Ip = np.linalg.eig(I)  # principal inertial tensor
        Ipm, Ip = np.real(Ipm), np.real(Ip)
        sort_inds = np.argsort(Ipm)
        Ipm = Ipm[sort_inds]
        Ip = Ip.T[sort_inds] # want eigenvectors to be sorted row-wise (rather than column-wise)
        Ip2_orig = Ip.copy()

        # cardinal direction is vector from CoM to farthest atom
        dists = np.linalg.norm(points, axis=1)
        max_ind = np.argmax(dists)
        max_equivs = np.argwhere(np.round(dists,8) == np.round(dists[max_ind],8))[:,0] # if there are multiple equidistant atoms - pick the one with the lowest index
        max_ind = int(np.amin(max_equivs))
        direction = points[max_ind]
        direction /= np.linalg.norm(direction)
        overlaps = Ip.dot(direction)  # check if the principal components point towards or away from the CoG
        if any(overlaps == 0):  # exactly zero is invalid
            overlaps[overlaps == 0] = 1
        if any(overlaps < 1e-8):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
            # align the 'good' vectors
            Ip = (Ip.T * np.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction
            fix_ind = np.argmin(np.abs(overlaps))
            other_vectors = np.delete(np.arange(3), fix_ind)
            check_direction = np.cross(Ip[other_vectors[0]], Ip[other_vectors[1]])
            # align the 'bad' vector
            Ip[fix_ind] *= np.sign(np.dot(check_direction, Ip[fix_ind]))
        else:
            Ip = (Ip.T * np.sign(overlaps)).T  # if the vectors have negative overlap, flip the direction

        test_Ip2 = Ip
        direction2 = direction.copy()
        overlaps2 = overlaps.copy()

        break

''' # functional - rotate back and forth
# center on CoM
coords -= coords.T.dot(masses) / masses.sum()
Ip, Ipm, I = compute_principal_axes_np(masses, coords)
mol1 = Atoms(symbols='OCN2C2NCNC', positions=coords,cell=[10,10,10,90,90,90])

rot = Rotation.from_euler('XYZ', rand_rot)
rot_coords = rot.apply(coords)  # np.array((coords[:,0],coords[:,2],coords[:,1])).T
rot_coords -= rot_coords.T.dot(masses) / masses.sum()
I2_trans = rot.apply(Ip)
Ip2, Ipm2, I2 = compute_principal_axes_np(masses, rot_coords)
mol2 = Atoms(symbols='OCN2C2NCNC', positions=rot_coords,cell=[10,10,10,90,90,90])

rot2 = Rotation.from_euler('ZYX', -rand_rot[-1::-1])
rot_coords2 = rot2.apply(rot_coords)  # np.array((coords[:,0],coords[:,2],coords[:,1])).T
rot_coords2 -= rot_coords.T.dot(masses) / masses.sum()
I3_trans = rot2.apply(Ip2)
Ip3, Ipm3, I3 = compute_principal_axes_np(masses, rot_coords2)
mol3 = Atoms(symbols='OCN2C2NCNC', positions=rot_coords2,cell=[10,10,10,90,90,90])

error = np.sum(np.abs(rot_coords2 - coords))
print("Error was {:.3f}".format(error))
'''

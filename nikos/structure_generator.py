import numba as nb
import numpy as np

from calculate_molecular_positions import calculate_molecular_positions
from calculate_molecular_rotations import calculate_molecular_rotations
from coordinate_transformations import coor_trans, coor_trans_matrix
from generate_unit_cell import generate_unit_cell
from identify_close_contacts import identify_close_contacts
from perpendicular_vectors import perpendicular_vectors
from plane_vector_transformation import plane_vector_transformation
from rotations import rotation_matrix_from_vectors, euler_rotation, rodriguez_rotation
from symmetry_operations import symmetry_operations
    
def structure_generator(sg,
                        random_cell_geometry,
                        atom_sfc,
                        ring_vector,
                        zzp_pairs):
    ''' Initialize unit cells '''
    unit_cells = []
    
    ''' Get the space group symmetry features '''
    rs, ts = symmetry_operations(sg)
    z_sg = len(rs)
    
    ''' Set the cell vectors and cell angles '''
    cell_vectors = np.array(random_cell_geometry[:3])
    cell_angles = np.array(random_cell_geometry[3:])
    
    ''' Calculate the transformation matrix from cartesian to fractional coordinates and vice versa '''
    T_cf = coor_trans_matrix('c_to_f',cell_vectors,cell_angles)
    T_fc = coor_trans_matrix('f_to_c',cell_vectors,cell_angles)
    
    ''' Generate a random plane in fractional coordinates to place the ring(s) of the molecules '''
    k = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)]) 
    k = k / np.linalg.norm(k)
    
    ''' Check if it is consistent with the database '''
    ''' k must form an angle (95% CI) in the range:
        72 - 97.63 with a vector n_u n_v n_w = 0, n_u | n_v | n_w = n_max = 1
		83.62 - 96.38 with a vector n_u n_v n_w = 0, n_u | n_v | n_w = n_max = 2
  		86.65 - 93.33 with a vector n_u n_v n_w = 0, n_u | n_v | n_w = n_max = 3
  		86.99 - 92.99 with a vector n_u n_v n_w = 0, n_u | n_v | n_w = n_max = 4
  		88.47 - 91.52 with a vector n_u n_v n_w = 0, n_u | n_v | n_w = n_max = 5
    '''
    angle_intervals = np.array([[81.72,97.63],
                                [83.62,96.38],
                                [86.65,93.33],
                                [86.99,92.99],
                                [88.47,91.52]])
    
    proceed_probability = 1.0
    for n in range(5):
        ''' Generate possible nearly perpendicular vectors '''
        perp_vectors = perpendicular_vectors(n + 1)
        
        ''' Loop over all nearly perpendicular vectors and calculate the minimum angle '''
        products = []
        for vec in perp_vectors:
            ''' Calculate the angle between k and vec ''' 
            cos_theta = np.dot(vec,k) / np.linalg.norm(vec)
            theta = np.arccos(cos_theta) * 180.0 / np.pi
            
            products.append([abs(cos_theta),theta])
            
        products.sort(key = lambda x:x[0])
        min_angle = products[0][1]
        
        ''' Check if the minimum angle is within the specified interval. If yes, the acceptance probability is 95% '''
        if min_angle >= angle_intervals[n,0] and min_angle <= angle_intervals[n,1]:
            proceed_probability *= 0.95
        else:
            proceed_probability *= 0.05
    
    rand_n = np.random.uniform(0,1)
    if rand_n > proceed_probability:
        print('Structure discarded on filter step 1: The normal plane vector is not acceptable')
        return [np.zeros(12).tolist()]
    
    ''' Convert the normal ring plane to cartesian coordinates '''
    k = plane_vector_transformation('f_to_c',k,cell_vectors,cell_angles)
    
    '''  Calculate the rotation matrix so that one ring plane vector becomes parallel to random plane vector '''
    Rmat = rotation_matrix_from_vectors(ring_vector,k)
    
    ''' Rotate the reference molecule so that the normal ring plane vector becomes parallel to random plane vector k '''
    v = euler_rotation(Rmat,atom_sfc) 
    
    dzzp_angles = calculate_molecular_rotations(zzp_pairs,k,v,T_cf,cell_vectors,cell_angles,0.01)
    
    if dzzp_angles == []:
        print('Structure discarded on filter step 2: No molecular rotation is consistent to the ZZPs')
        return [np.zeros(12).tolist()]
    
    ''' Loop over all possible dzzp_angles '''
    for angle in dzzp_angles:
        ''' Rotate the molecule about the plane vector to get the bond vectors '''
        bv = rodriguez_rotation(k,v,angle)
        
        ''' Convert to fractional coordinates '''
        bv_f = coor_trans('c_to_f',bv,cell_vectors,cell_angles)
        
        ''' Get the possible molecular positions based on the high charge atoms '''
        r_cm_list = calculate_molecular_positions(high_charge_atoms,bv_f)

        if r_cm_list == []:
            continue
        else:
            r_cm_list = np.array(r_cm_list,dtype=np.float_)

        ''' Loop over possible molecular positions '''
        for r_cm in r_cm_list:
            ''' Move the molecule to position '''
            r_f = bv_f + r_cm
            
            ''' Calculate the molecular positions, atomic positions and bond vectors for all molecules in the unit cell '''
            r_mol_uc, r_at_uc, bv_at_uc = generate_unit_cell(z_sg,rs,ts,r_cm,n_atoms,r_f,atom_mass)
            
            ''' Identify close contacts '''
            close_contacts, species_pairs, overlap, too_close = identify_close_contacts(z_sg,n_atoms,r_at_uc,cell_vectors,cell_angles,atom_name,atom_vdw)
            
            if too_close:
                continue
            else:
                unit_cells.append([cell_vectors[0],
                                   cell_vectors[1],
                                   cell_vectors[2],
                                   cell_angles[0],
                                   cell_angles[1],
                                   cell_angles[2],
                                   k[0],
                                   k[1],
                                   k[2],
                                   angle,
                                   r_cm[0],
                                   r_cm[1],
                                   r_cm[2]])
    
    return np.array(unit_cells)
 
''' Set the space group '''
sg = 'Pca21'

''' Set the number of atoms '''
n_atoms = 17

''' Set the atomic species '''
atom_name = np.array(['O','O','C','C','C','C','C','C','C','C','C','H','H','H','H','H','H']).reshape(n_atoms,1)

''' Set the atomic vdW radii'''
atom_vdw = np.array([1.52,1.52,1.70,1.70,1.70,1.70,1.70,1.70,1.70,1.70,1.70,1.20,1.20,1.20,1.20,1.20,1.20]).reshape(n_atoms,1)

''' The atomic coordinates in the reference system '''
atom_sfc = np.array([[-1.0024,   1.0330,  -0.0084],
                     [-3.1773,   0.7271,   0.0015],
                     [-2.0918,   0.2055,  -0.0022],
                     [-1.8446,  -1.2239,   0.0049],
                     [-0.5794,  -1.7139,   0.0048],
                     [ 1.8677,  -1.2501,  -0.0069],
                     [ 2.8932,  -0.3298,  -0.0042],
                     [ 2.6097,   1.0275,   0.0099],
                     [ 1.3095,   1.4842,   0.0095],
                     [ 0.2808,   0.5421,  -0.0047],
                     [ 0.5333,  -0.8241,  -0.0010],
                     [-2.6828,  -1.7773,   0.0171],
                     [-0.3838,  -2.6962,   0.0064],
                     [ 2.0631,  -2.1366,  -0.0046],
                     [ 3.7395,  -0.7100,  -0.0062],
                     [ 3.1963,   1.7495,  -0.0010],
                     [ 1.0892,   2.4476,  -0.0238]])

''' Set the mass of the atoms '''
atom_mass = np.array([15.999,15.999,12.0107,12.0107,12.0107,12.0107,12.0107,12.0107,12.0107,12.0107,12.0107,1.0089,1.0089,1.0089,1.0089,1.0089,1.0089])

''' This is the random cell geometry with values from the database '''
random_cell_geometry = np.array([15.503,5.666,7.918,90.0,90.0,90.0])

''' This is the normal ring plane in cartesian coordinates in your reference coordinate system '''
ring_vector = np.array([0,0,1.0])

''' Set the high charge atoms, the most probable close contact pairs, and the pairs that are most likely to be found at distance 0.25k in fractional coordinates '''
high_charge_atoms = np.array([0,2])
close_contact_pairs = np.array([])
zzp_pairs = np.array([[0,5],[0,6]])

unit_cells = structure_generator(sg,
                                 random_cell_geometry,
                                 atom_sfc,
                                 ring_vector,
                                 zzp_pairs)
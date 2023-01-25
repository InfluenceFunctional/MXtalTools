import numba as nb
import numpy as np

'''
correlates ZZP distances with atom-pair distances? and angles?
pass for now
'''

@nb.jit(nopython=True)
def calculate_molecular_rotations(zzp_pairs,k,v,T_cf,cell_vectors,cell_angles,tol):
    ''' Set the distance between the ZZPs in fractional coordinates '''
    d_zzp_list = [-1.00,-0.75,-0.50,-0.25,0.25,0.50,0.75,1.00]
    
    ''' Initialize angles list. It is not an empty list because numba cannot work well with empty lists. 
        This value will be discarded later '''
    dzzp_angles = [1000.0]

    ''' Loop over all atomic pairs '''
    for at1, at2 in zzp_pairs:
        ''' Initialize coefficients and the norm '''
        A = np.zeros(3,dtype=np.float_)
        B = np.zeros(3,dtype=np.float_)
        C = np.zeros(3,dtype=np.float_)
        N = np.zeros(3,dtype=np.float_)
            
        ''' Calculate the separation between the atoms of the pair '''
        dx = v[at1,0] - v[at2,0]
        dy = v[at1,1] - v[at2,1]
        dz = v[at1,2] - v[at2,2]
        
        ''' Calculate the dot product of the plane vector with the position vector between the two atoms '''
        k_dv = np.dot(k,v[at1] - v[at2])
        
        ''' Identify the rotation angle(s) for which the distances between atomic pairs
            corresponds to distances between the ZZPs by solving equations
            da = (A[0] - C[0]) cos_theta[0] + B[0] sin_theta[0] + C[0]
            db = (A[1] - C[1]) cos_theta[1] + B[1] sin_theta[1] + C[1]
            dc = (A[2] - C[2]) cos_theta[2] + B[2] sin_theta[2] + C[2] '''
        C[0] =  (T_cf[0,0] * k[0] + T_cf[0,1] * k[1] + T_cf[0,2] * k[2]) * k_dv
        B[0] = -(T_cf[0,0] * (k[1] * dz - k[2] * dy) + T_cf[0,1] * (k[2] * dx - k[0] * dz) + T_cf[0,2] * (k[0] * dy - k[1] * dx))
        A[0] =   T_cf[0,0] * dx + T_cf[0,1] * dy + T_cf[0,2] * dz - C[0]
        
        C[1] =  (T_cf[1,1] * k[1] + T_cf[1,2] * k[2]) * k_dv
        B[1] = -(T_cf[1,1] * (k[2] * dx - k[0] * dz) + T_cf[1,2] * (k[0] * dy - k[1] * dx))
        A[1] =   T_cf[1,1] * dy + T_cf[1,2] * dz - C[1]
        
        C[2] =  T_cf[2,2] * k[2] * k_dv
        B[2] = -T_cf[2,2] * (k[0] * dy - k[1] * dx)
        A[2] =  T_cf[2,2] * dz - C[2]
        
        N = np.sqrt(A**2 + B**2)
        
        if N[0] * N[1] * N[2] == 0.0:
            continue
        
        phi = np.arctan2(B,A) * 180.0 / np.pi
        
        for d_zzp in d_zzp_list:
            if abs((d_zzp - C[0]) / N[0]) <= 1.0:
                angle = np.arccos((d_zzp - C[0]) / N[0]) * 180.0 / np.pi
                
                dzzp_angles.append(( angle - phi[0]) % 360.0)
                dzzp_angles.append((-angle - phi[0]) % 360.0)
                
            if abs((d_zzp - C[1]) / N[1]) <= 1.0:
                angle = np.arccos((d_zzp - C[1]) / N[1]) * 180.0 / np.pi
                
                dzzp_angles.append(( angle - phi[1]) % 360.0)
                dzzp_angles.append((-angle - phi[1]) % 360.0)
                
            if abs((d_zzp - C[2]) / N[2]) <= 1.0:
                angle = np.arccos((d_zzp - C[2]) / N[2]) * 180.0 / np.pi
                
                dzzp_angles.append(( angle - phi[2]) % 360.0)
                dzzp_angles.append((-angle - phi[2]) % 360.0)
        
    ''' Sort and cluster angles '''
    dzzp_angles = np.array(dzzp_angles)
    dzzp_angles = np.sort(dzzp_angles)
        
    n_angles = 1
    centroid = dzzp_angles[0]
    dzzp_angles_cen = [1000.0]
    for i in range(1,len(dzzp_angles)):
        if dzzp_angles[i] - dzzp_angles[i-1] > 2:
            dzzp_angles_cen.append(centroid / n_angles)
            n_angles = 1
            centroid = dzzp_angles[i]
        else:
            n_angles += 1
            centroid += dzzp_angles[i]
            
    return dzzp_angles_cen[1:]
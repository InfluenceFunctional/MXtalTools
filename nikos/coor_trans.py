import numba as nb
import numpy as np

@nb.jit(nopython=True)
def coor_trans(opt,pos,v,a):
    
    ''' Calculate cos and sin of cell angles '''
    cos_a = np.cos(a*np.pi/180.0)
    sin_a = np.sin(a*np.pi/180.0)
    
    ''' Calculate volume of the unit cell '''
    vol = v[0]*v[1]*v[2]*np.sqrt(1.0 - cos_a[0]**2 - cos_a[1]**2 - cos_a[2]**2 + 2.0*cos_a[0]*cos_a[1]*cos_a[2])
    
    ''' Setting the transformation matrix '''
    m = np.zeros((3,3),dtype=np.float_)
    if (opt == 'c_to_f'):
        ''' Converting from cartesian to fractional '''
        m[0,0] =  1.0/v[0]
        m[0,1] = -cos_a[2]/v[0]/sin_a[2]
        m[0,2] =  v[1]*v[2]*(cos_a[0]*cos_a[2]-cos_a[1])/vol/sin_a[2]
        m[1,1] =  1.0/v[1]/sin_a[2]
        m[1,2] =  v[0]*v[2]*(cos_a[1]*cos_a[2]-cos_a[0])/vol/sin_a[2]
        m[2,2] =  v[0]*v[1]*sin_a[2]/vol
    elif (opt == 'f_to_c'):
        ''' Converting from fractional to cartesian '''
        m[0,0] =  v[0]
        m[0,1] =  v[1]*cos_a[2]
        m[0,2] =  v[2]*cos_a[1]
        m[1,1] =  v[1]*sin_a[2]
        m[1,2] =  v[2]*(cos_a[0]-cos_a[1]*cos_a[2])/sin_a[2]
        m[2,2] =  vol/v[0]/v[1]/sin_a[2]

    p = np.zeros_like(pos)
    
    if pos.ndim == 1:
        p[0] = m[0,0] * pos[0] + m[0,1] * pos[1] + m[0,2] * pos[2]
        p[1] = m[1,0] * pos[0] + m[1,1] * pos[1] + m[1,2] * pos[2]
        p[2] = m[2,0] * pos[0] + m[2,1] * pos[1] + m[2,2] * pos[2]
        
    elif pos.ndim == 2:
        n_atoms = p.shape[0]
        for at in range(n_atoms):
            p[at,0] = m[0,0] * pos[at,0] + m[0,1] * pos[at,1] + m[0,2] * pos[at,2]
            p[at,1] = m[1,0] * pos[at,0] + m[1,1] * pos[at,1] + m[1,2] * pos[at,2]
            p[at,2] = m[2,0] * pos[at,0] + m[2,1] * pos[at,1] + m[2,2] * pos[at,2]
            
    else:
        n_mols = p.shape[0]
        n_atoms = p.shape[1]
        for mol in range(n_mols):
            for at in range(n_atoms):
                p[mol,at,0] = m[0,0] * pos[mol,at,0] + m[0,1] * pos[mol,at,1] + m[0,2] * pos[mol,at,2]
                p[mol,at,1] = m[1,0] * pos[mol,at,0] + m[1,1] * pos[mol,at,1] + m[1,2] * pos[mol,at,2]
                p[mol,at,2] = m[2,0] * pos[mol,at,0] + m[2,1] * pos[mol,at,1] + m[2,2] * pos[mol,at,2]

    return p

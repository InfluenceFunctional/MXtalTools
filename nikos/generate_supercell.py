import numba as nb
import numpy as np

@nb.jit(nopython=True)
def generate_supercell(z,n_atoms,r):
    ''' Initialize atomic positions in the supercell '''
    r_sc = np.zeros((27*z,n_atoms,3),dtype=np.float_)
    
    ''' Set the positions of the atoms in the reference cell '''
    mol_index = 0
    for mol in range(z):
        for at in range(n_atoms):
            r_sc[mol_index,at,:] = r[mol_index,at,:]
            
        mol_index += 1
        
    ''' Set the atomic positions in the supercell '''
    for ix in [-1,0,1]:
        for iy in [-1,0,1]:
            for iz in [-1,0,1]:
                # Exclude the reference cell
                if ix == iy == iz == 0:
                    continue
                
                else:
                    for mol in range(z):
                        for at in range(n_atoms):
                            r_sc[mol_index,at,:] = r[mol,at,:] + np.array([ix,iy,iz]) 
                        mol_index += 1
             
    return r_sc
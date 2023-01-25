import numpy as np

def generate_unit_cell(z_sg,rs,ts,r_cm,n_atoms,pos,mass):
    ''' Initialize molecular and atomic positions '''
    r_mol = np.zeros((z_sg,3),dtype=np.float_)
    r_at  = np.zeros((z_sg,n_atoms,3),dtype=np.float_)
    bv_at = np.zeros((z_sg,n_atoms,3),dtype=np.float_)
    
    ''' Calculate molecular positions for all molecules in the unit cell '''
    for mol in range(z_sg):
        r_mol[mol] = r_cm * rs[mol] + ts[mol]
        
    ''' Calculate the bond vectors for all atoms in the unit cell'''
    for mol in range(z_sg):
        for at in range(n_atoms):
            bv_at[mol,at] = pos[at] * rs[mol] + ts[mol] - r_mol[mol]
            
    ''' Apply PDB to the molecular positions '''
    r_mol = r_mol % 1.0
    
    ''' Calculate the atomic positions for all atoms in the unit cell '''
    for mol in range(z_sg):
        for at in range(n_atoms):
            r_at[mol,at] = bv_at[mol,at] + r_mol[mol]
            
    return r_mol, r_at, bv_at
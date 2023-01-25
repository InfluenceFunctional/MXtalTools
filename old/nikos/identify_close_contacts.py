import numba as nb
import numpy as np

from crystal_building.coordinate_transformations import coor_trans
from generate_supercell import generate_supercell

@nb.jit(nopython=True)
def identify_close_contacts(z,n_atoms,r,cell_vec,cell_ang,atom_name,atom_vdw):
    ''' Generate the supercell '''
    r_sc = generate_supercell(z,n_atoms,r)
    for i in range(27*z):
        r_sc[i] = coor_trans('f_to_c',r_sc[i],cell_vec,cell_ang)
        
    ''' Identify close contacts based on the vdw radius of the atoms '''    
    close_contacts = []
    species_pairs = []
    overlap = []
    too_close = False
    
    # Search for close contacts within the molecules of the unit cell
    for i in range(z-1):
        for j in range(i+1,z):
            for a in range(n_atoms):
                for b in range(n_atoms):
                    x_ab = r_sc[i,a,0] - r_sc[j,b,0]
                    y_ab = r_sc[i,a,1] - r_sc[j,b,1]
                    z_ab = r_sc[i,a,2] - r_sc[j,b,2]
                    
                    d_vdw = atom_vdw[a,0] + atom_vdw[b,0]
                        
                    if x_ab > d_vdw or x_ab < -d_vdw:
                        continue
                    if y_ab > d_vdw or y_ab < -d_vdw:
                        continue
                    if z_ab > d_vdw or z_ab < -d_vdw:
                        continue
                    
                    d_ab_sq  = x_ab * x_ab + y_ab * y_ab + z_ab * z_ab
                    d_vdw_sq = d_vdw * d_vdw
                    
                    if d_ab_sq <= d_vdw_sq:
                        d_ab = np.sqrt(d_ab_sq)
                        close_contacts.append([i,j,a,b])
                        species_pairs.append([atom_name[a,0],atom_name[b,0]])
                        overlap.append([d_ab,d_vdw - d_ab])
                        
                        if d_vdw - d_ab > min(atom_vdw[a,0],atom_vdw[b,0]):
                            too_close = True
                        
                    if too_close:
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            break
    
    if not too_close:
        for i in range(z):
            for j in range(z,27*z):
                for a in range(n_atoms):
                    for b in range(n_atoms):
                        x_ab = r_sc[i,a,0] - r_sc[j,b,0]
                        y_ab = r_sc[i,a,1] - r_sc[j,b,1]
                        z_ab = r_sc[i,a,2] - r_sc[j,b,2]
                        
                        d_vdw = atom_vdw[a,0] + atom_vdw[b,0]
                            
                        if x_ab > d_vdw or x_ab < -d_vdw:
                            continue
                        if y_ab > d_vdw or y_ab < -d_vdw:
                            continue
                        if z_ab > d_vdw or z_ab < -d_vdw:
                            continue
                        
                        d_ab_sq  = x_ab * x_ab + y_ab * y_ab + z_ab * z_ab
                        d_vdw_sq = d_vdw * d_vdw
                        
                        if d_ab_sq <= d_vdw_sq:
                            d_ab = np.sqrt(d_ab_sq)
                            close_contacts.append([i,j,a,b])
                            species_pairs.append([atom_name[a,0],atom_name[b,0]])
                            overlap.append([d_ab,d_vdw - d_ab])
                            
                            if d_vdw - d_ab > min(atom_vdw[a,0],atom_vdw[b,0]):
                                too_close = True
                            
                        if too_close:
                            break
                    if too_close:
                        break
                if too_close:
                    break
            if too_close:
                break
    
    return close_contacts, species_pairs, overlap, too_close
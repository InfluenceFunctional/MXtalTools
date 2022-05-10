###############################################################################
# You can use this finction to calculate the partial atomic charges.
# Since you have the cartesian atomic coordinates, you need to generate a cif 
# file for the molecule. In general, cif files contains the fractional 
# coordinates. However, in case where a=b=c=1.0 and alpha=beta=gamma=90.0, 
# the fractional and cartesian coordinates are identical. So you need to place
# the molecule in a unit cell with a=b=c=1.0 and alpha=beta=gamma=90.0.
# You can use any space group for the cif file, it does not matter when 
# all you want to do is to calculate the fractional atomic charges

import numpy as np

from coor_trans import coor_trans
from plane_vector_transformation import plane_vector_transformation

from ccdc.descriptors import MolecularDescriptors as MD
from ccdc.io import CrystalReader

''' Enter the cif filename to get crystal data '''
cif_fname = 'Coumarin_I_C9H6O2_CCDC_1130207.cif'

''' Read crystal data '''
csd_crystal = CrystalReader(cif_fname)[0]

# Move molecule inside unit cell 
csd_crystal.centre_molecule()

''' Get crystal properties from the cif file '''
# This properties are useful when you read an actual cif file.
# In case you set manually the cell geometry this will just return the values
# you entered.
# For the calculation of molecular properties you do not need them, but I have 
# them here in case you need for other applications
sg = csd_crystal.spacegroup_symbol
z_sg = csd_crystal.z_value
z_prime = csd_crystal.z_prime

cell_a = round(csd_crystal.cell_lengths[0],4)
cell_b = round(csd_crystal.cell_lengths[1],4)
cell_c = round(csd_crystal.cell_lengths[2],4)

cell_alpha = round(csd_crystal.cell_angles[0],4)
cell_beta = round(csd_crystal.cell_angles[1],4)
cell_gamma = round(csd_crystal.cell_angles[2],4)

cell_vectors = np.array([cell_a,cell_b,cell_c])
cell_angles = np.array([cell_alpha,cell_beta,cell_gamma])

cell_vol = round(csd_crystal.cell_volume,4)
cell_den = round(csd_crystal.calculated_density,4)

''' Get molecular properties '''
# These properties depend on the molecule and they are independent on the cell
# geometry, so you can use them
csd_mol = csd_crystal.molecule

''' Get the molecular id, formula, volume '''
mol_id = csd_mol.identifier
mol_formula = csd_mol.formula
mol_vol = csd_mol.molecular_volume

''' Assign partial atomic charges '''
mol_charges = csd_mol.assign_partial_charges()
        
''' Get atomic properties ''' 
n_atoms = len(csd_mol.atoms)

atom_charge = []
atom_label = []
atom_mass = []
atom_name = []
atom_vdw = []
for at in csd_mol.atoms:
    atom_charge.append(round(at.partial_charge,4))
    atom_label.append(at.label)
    atom_mass.append(at.atomic_weight)
    atom_name.append(at.atomic_symbol)
    atom_vdw.append(at.vdw_radius)
            
''' Check if each atom is a metal, a hydrogen bond donor/acceptor and whether or not it is in a ring system '''
atom_is_metal = []
atom_is_donor = []
atom_is_acceptor = []
atom_is_cyclic = []
for at in csd_mol.atoms:
    atom_is_metal.append(at.is_metal)
    atom_is_donor.append(at.is_donor)
    atom_is_acceptor.append(at.is_acceptor)
    atom_is_cyclic.append(at.is_cyclic)
    
''' Get the molecular rings '''
mol_rings = csd_mol.rings

''' Get the centroids of the molecules in fractional and cartesian coordiantes '''
centroid = MD.atom_centroid(*tuple(a for a in csd_mol.atoms))
mol_cen_c = np.array([round(centroid[0],4),round(centroid[1],4),round(centroid[2],4)])
mol_cen_f = coor_trans('c_to_f',mol_cen_c,cell_vectors,cell_angles)

''' Get the centroids and normal planes of the rings of the rings '''
# When we calculate the normal ring plane in fractional coordinates, we cannot
# apply a simple coordinate transformation from cartesian to fractional 
# coordinates. This is why we use the plane_vector_transformation function
# that actually transforms the cartesian normal plane to a vactor that is 
# perpendicular to the ring of the plane on fractional coordinates.
ring_centroids_c = []
ring_centroids_f = []
ring_planes_c = []
ring_planes_f = []
for ring in mol_rings:
    centroid = MD.ring_centroid(ring)
    centroid_c = np.array([centroid[0],centroid[1],centroid[2]])
    centroid_f = coor_trans('c_to_f',centroid_c,cell_vectors,cell_angles)
    ring_centroids_c.append(centroid_c)
    ring_centroids_f.append(centroid_f)
    
    plane = MD.ring_plane(ring).normal
    plane_c = np.array([plane[0],plane[1],plane[2]])
    plane_f = plane_vector_transformation('c_to_f',plane_c,cell_vectors,cell_angles)
    ring_planes_c.append(plane_c)
    ring_planes_f.append(plane_f)
    
ring_centroids_c = np.array(ring_centroids_c)
ring_centroids_f = np.array(ring_centroids_f)

ring_planes_c = np.array(ring_planes_c)
ring_planes_f = np.array(ring_planes_f)

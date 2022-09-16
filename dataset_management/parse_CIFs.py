from ccdc import io
import numpy as np

path = 'C:/Users/mikem/OneDrive/NYU/CSD/blind_tests/blind_test_6/gp5080sup2/Group_01_Chadha_Singh/XXII-1_Chadha_Singh.cif'
crystal_reader = io.CrystalReader(path,format='cif')

crystals = []
coords = []
atoms = []
z_values = []
for crystal in crystal_reader:
    crystals.append(crystal)
    #cell = crystals[-1].packing(((0, 0, 0), (1, 1, 1)))
    #coords.append(np.stack([cell.atoms[n].coordinates for n in range(len(cell.atoms))]))
    #atoms.append(np.stack([cell.atoms[n].atomic_number for n in range(len(cell.atoms))]))


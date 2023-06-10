# import pytest
from crystal_building.builder import SupercellBuilder
from dataset_management.utils import load_test_dataset
import numpy as np
import torch

'''
test module for crystal builder

1) does not crash
2) can analyze and exactly recreate a given crystal in all spacegroups
3) correctly executes all subtasks
'''

'''load test dataset'''
test_dataset_path = r'C:\Users\mikem\OneDrive\NYU\CSD\MCryGAN\tests/dataset_for_tests'
test_crystals, dataDims, symmetry_info = load_test_dataset(test_dataset_path)

'''initialize supercell builder'''
supercell_builder = SupercellBuilder(symmetry_info, dataDims, supercell_size=1, device='cpu')

'''Build reference supercell from unit cell sample, then analyze and rebuild it from scratch'''
reference_supercells = supercell_builder.unit_cell_to_supercell(
    test_crystals, graph_convolution_cutoff=6, supercell_size=1, pare_to_convolution_cluster=False)


# todo fix parameterization - it's a bit janky

rebuilt_supercells, _, _ = supercell_builder.build_supercells(
    test_crystals, test_crystals.cell_params, standardized_sample=False,
    align_molecules=True, target_handedness=reference_supercells.asym_unit_handedness,
    rescale_asymmetric_unit=False, graph_convolution_cutoff=6, supercell_size=1, pare_to_convolution_cluster=False)

'''absolute coords'''
assert torch.mean(torch.abs(rebuilt_supercells.pos - reference_supercells.pos)) < 1e-3

'''rebuild clusters and compare RDFs'''


'''
from models.utils import ase_mol_from_crystaldata
from ase.visualize import view
mol1 = ase_mol_from_crystaldata(reference_supercells,1)
mol2 = ase_mol_from_crystaldata(rebuilt_supercells,1)
view([mol1,mol2])
'''

aa = 0

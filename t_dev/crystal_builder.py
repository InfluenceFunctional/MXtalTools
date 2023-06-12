# import pytest
from common.utils import compute_rdf_distance
from crystal_building.builder import SupercellBuilder
from dataset_management.utils import load_test_dataset
from models.crystal_rdf import crystal_rdf
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
supercell_size = 5

'''initialize supercell builder'''
supercell_builder = SupercellBuilder(symmetry_info, dataDims, supercell_size=supercell_size, device='cpu')


def test_cell_parameterization_and_reconstruction():
    """
    Build reference supercell from unit cell sample,
    then analyze and rebuild it from scratch
    one crystal per space group in test dataset
    """
    reference_supercells = supercell_builder.unit_cell_to_supercell(
        test_crystals, graph_convolution_cutoff=6,
        supercell_size=supercell_size, pare_to_convolution_cluster=True)

    rebuilt_supercells, _, _ = supercell_builder.build_supercells(
        test_crystals, test_crystals.cell_params,
        skip_cell_cleaning=True, standardized_sample=False,
        align_molecules=True, target_handedness=reference_supercells.asym_unit_handedness,
        rescale_asymmetric_unit=False, graph_convolution_cutoff=6,
        supercell_size=supercell_size, pare_to_convolution_cluster=True)

    '''high symmetry molecules may 'veer' different ways, so this assertion may fail'''  # todo force parameterization into the +z half of the sphere
    #assert torch.mean(torch.abs(reference_supercells.cell_params - rebuilt_supercells.cell_params)) < 1e-3

    '''
    compare RDFs - should uniquely characterize the material
    should ideally do this atomwise but the indexing is not perfect
    '''
    rdf_range, rdf_bins = [0, 10], 100
    reference_rdf, rr, _ = crystal_rdf(reference_supercells, rrange=rdf_range, bins=rdf_bins, elementwise=True,
                                       raw_density=True, atomwise=False, mode='all', cpu_detach=True)
    rebuilt_rdf, _, _ = crystal_rdf(rebuilt_supercells, rrange=rdf_range, bins=rdf_bins, elementwise=True,
                                    raw_density=True, atomwise=False, mode='all', cpu_detach=True)

    '''compute earth mover's distance between RDFs'''
    rdf_dists = np.zeros(reference_supercells.num_graphs)
    for i in range(reference_supercells.num_graphs):
        rdf_dists[i] = compute_rdf_distance(reference_rdf[i], rebuilt_rdf[i], rr)

    assert all(rdf_dists < 1e-3)  # RDFs should be nearly identical

    '''  # optionally look at some cells
    from models.utils import ase_mol_from_crystaldata
    from ase.visualize import view
    crystal_number = 0
    mol1 = ase_mol_from_crystaldata(reference_supercells, crystal_number)
    mol2 = ase_mol_from_crystaldata(rebuilt_supercells ,crystal_number)
    view([mol1,mol2])
    '''


def test_distorted_cell_reconstruction():
    """
    Build reference supercell from unit cell sample,
    then analyze and rebuilt it from scratch, with some distortion
    one crystal per space group in test dataset
    """
    reference_supercells = supercell_builder.unit_cell_to_supercell(
        test_crystals, graph_convolution_cutoff=6,
        supercell_size=supercell_size, pare_to_convolution_cluster=True)

    distorted_params = test_crystals.cell_params + 0.1
    rebuilt_supercells, _, _ = supercell_builder.build_supercells(
        test_crystals, distorted_params,
        skip_cell_cleaning=False, standardized_sample=False,
        align_molecules=True, target_handedness=reference_supercells.asym_unit_handedness,
        rescale_asymmetric_unit=True, graph_convolution_cutoff=6,
        supercell_size=supercell_size, pare_to_convolution_cluster=True)

    '''
    compare RDFs - should uniquely characterize the material
    should ideally do this atomwise but the indexing is not perfect
    '''
    rdf_range, rdf_bins = [0, 10], 100
    reference_rdf, rr, _ = crystal_rdf(reference_supercells, rrange=rdf_range, bins=rdf_bins, elementwise=True,
                                       raw_density=True, atomwise=False, mode='all', cpu_detach=True)
    rebuilt_rdf, _, _ = crystal_rdf(rebuilt_supercells, rrange=rdf_range, bins=rdf_bins, elementwise=True,
                                    raw_density=True, atomwise=False, mode='all', cpu_detach=True)

    '''compute earth mover's distance between RDFs'''
    rdf_dists = np.zeros(reference_supercells.num_graphs)
    for i in range(reference_supercells.num_graphs):
        rdf_dists[i] = compute_rdf_distance(reference_rdf[i], rebuilt_rdf[i], rr)

    assert all(rdf_dists > 1e-1)  # RDFs should be significantly different


test_cell_parameterization_and_reconstruction()
test_distorted_cell_reconstruction()

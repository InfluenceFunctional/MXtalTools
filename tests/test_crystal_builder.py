from common.utils import compute_rdf_distance
from crystal_building.utils import batch_asymmetric_unit_pose_analysis_torch
from models.crystal_rdf import crystal_rdf
from crystal_modeller import Modeller
import numpy as np
import torch
from common.config_processing import get_config

'''
test module for crystal builder

1) does not crash
2) can analyze and exactly recreate a given crystal in all spacegroups
3) correctly executes all subtasks
'''

'''load test dataset'''
config_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/test_configs/crystal_building.yaml'
user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/users/mkilgour.yaml'
config = get_config(user_yaml_path=user_path, main_yaml_path=config_path)
modeller = Modeller(config)
_, data_loader, _ = modeller.load_dataset_and_dataloaders(override_test_fraction=1)
modeller.misc_pre_training_items()
supercell_builder = modeller.supercell_builder
test_crystals = next(iter(data_loader))

supercell_size = 5
rotation_basis = 'spherical'


class TestClass:

    @staticmethod
    def test_cell_parameterization_and_reconstruction():
        """
        Build reference supercell from unit cell sample,
        then analyze and rebuild it from scratch
        one crystal per space group in test dataset
        """
        '''
        build reference cells
        '''
        reference_supercells = supercell_builder.prebuilt_unit_cell_to_supercell(
            test_crystals, graph_convolution_cutoff=6,
            supercell_size=supercell_size, pare_to_convolution_cluster=True)

        '''
        pose analysis
        '''
        position, rotation, handedness, well_defined_asym_unit, canonical_coords_list = \
            batch_asymmetric_unit_pose_analysis_torch(
                [torch.Tensor(test_crystals.ref_cell_pos[ii]) for ii in range(test_crystals.num_graphs)],
                torch.Tensor(test_crystals.sg_ind),
                supercell_builder.asym_unit_dict,
                torch.Tensor(test_crystals.T_fc),
                enforce_right_handedness=False,
                return_asym_unit_coords=True,
                rotation_basis=rotation_basis)

        updated_params = test_crystals.cell_params.clone()
        updated_params[:, 9:12] = rotation  # overwrite to canonical parameters
        # supercell_data.asym_unit_handedness = mol_handedness

        rebuilt_supercells, _ = supercell_builder.build_supercells(
            test_crystals, updated_params,
            align_to_standardized_orientation=True, target_handedness=reference_supercells.asym_unit_handedness,
            graph_convolution_cutoff=6,
            supercell_size=supercell_size, pare_to_convolution_cluster=True)

        '''high symmetry molecules may 'veer' different ways, so this assertion may fail'''
        assert torch.mean(torch.abs(reference_supercells.cell_params - rebuilt_supercells.cell_params)) < 1e-3

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

        assert all(rdf_dists < 1e-2)  # RDFs should be nearly identical

        '''  # optionally look at some cells
        from models.utils import ase_mol_from_crystaldata
        from ase.visualize import view
        crystal_number = 0
        mol1 = ase_mol_from_crystaldata(reference_supercells, crystal_number)
        mol2 = ase_mol_from_crystaldata(rebuilt_supercells ,crystal_number)
        view([mol1,mol2])
        '''

    @staticmethod
    def test_distorted_cell_reconstruction():
        """
        Build reference supercell from unit cell sample,
        then analyze and rebuilt it from scratch, with some distortion
        one crystal per space group in test dataset
        """
        reference_supercells = supercell_builder.prebuilt_unit_cell_to_supercell(
            test_crystals, graph_convolution_cutoff=6,
            supercell_size=supercell_size, pare_to_convolution_cluster=True)

        distorted_params = test_crystals.cell_params / 2
        rebuilt_supercells, _ = supercell_builder.build_supercells(
            test_crystals, distorted_params,
            align_to_standardized_orientation=True, target_handedness=reference_supercells.asym_unit_handedness,
            graph_convolution_cutoff=6,
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

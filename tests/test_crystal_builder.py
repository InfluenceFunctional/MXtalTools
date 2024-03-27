from torch_geometric.loader.dataloader import Collater

from mxtaltools.common.utils import compute_rdf_distance, init_sym_info
from mxtaltools.crystal_building.utils import batch_asymmetric_unit_pose_analysis_torch, get_intra_mol_dists, clean_cell_params
from mxtaltools.models.crystal_rdf import crystal_rdf
from mxtaltools.modeller import Modeller
import numpy as np
import torch
from mxtaltools.common.config_processing import process_main_config
from tqdm import tqdm

'''
test module for crystal builder

1) does not crash
2) can analyze and exactly recreate a given crystal in all spacegroups
3) correctly executes all subtasks
'''

'''load test dataset'''
config_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/test_configs/crystal_building.yaml'
user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/users/mkilgour.yaml'
config = process_main_config(user_yaml_path=user_path, main_yaml_path=config_path)
modeller = Modeller(config)
_, data_loader, _ = modeller.load_dataset_and_dataloaders(override_test_fraction=1)
modeller.init_gaussian_generator()
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
            molecule_data=test_crystals,
            cell_parameters=updated_params,
            align_to_standardized_orientation=True,
            target_handedness=reference_supercells.asym_unit_handedness,
            graph_convolution_cutoff=6,
            supercell_size=supercell_size,
            pare_to_convolution_cluster=True)

        '''high symmetry molecules may 'veer' different ways, so this assertion may fail'''
        assert torch.mean(torch.abs(reference_supercells.cell_params - rebuilt_supercells.cell_params)) < 1e-3

        '''confirm that molecules were not distorted in original cell'''
        for ind in range(reference_supercells.num_graphs):
            dists = get_intra_mol_dists(reference_supercells, ind)  # assumes molecules are indexed sequentially in blocks

            dmat = torch.zeros((len(dists), len(dists)))
            for i in range(len(dists)):
                for j in range(len(dists)):
                    if j > i:
                        dmat[i, j] = torch.mean(torch.abs(dists[i] - dists[j]))

            assert dmat.max() < 1e-3, f"Too-large distortion in sample {ind + 1}"

        '''confirm that molecules were not distorted in cell building'''
        for ind in range(rebuilt_supercells.num_graphs):
            dists = get_intra_mol_dists(rebuilt_supercells, ind)  # assumes molecules are indexed sequentially in blocks

            dmat = torch.zeros((len(dists), len(dists)))
            for i in range(len(dists)):
                for j in range(len(dists)):
                    if j > i:
                        dmat[i, j] = torch.mean(torch.abs(dists[i] - dists[j]))

            assert dmat.max() < 1e-3, f"Too-large distortion in sample {ind + 1}"

        """build one molecule in each space group and confirm molecules are not warped"""
        test_molecule = test_crystals[0]
        test_molecule.pos = torch.randn_like(test_molecule.pos)
        collater = Collater(None, None)
        mol_batch = collater([test_molecule for _ in range(230)])
        mol_batch.sg_ind = torch.arange(1, 231)
        for i in range(230):
            mol_batch.symmetry_operators[i] = supercell_builder.sym_ops[i + 1]
            mol_batch.mult[i] = len(mol_batch.symmetry_operators[i])

        all_params = torch.ones((230, 12), dtype=torch.float32, device=supercell_builder.device) / 2
        all_params[:, 3:6] = torch.pi / 2  # valid in most SGs
        all_params[:, 0:3] *= 20

        symmetries_dict = init_sym_info()
        final_samples = clean_cell_params(all_params, mol_batch.sg_ind, modeller.lattice_means, modeller.lattice_stds,
                                          symmetries_dict, supercell_builder.asym_unit_dict,
                                          rescale_asymmetric_unit=False, destandardize=False, mode='hard')

        all_sg_supercells, _ = supercell_builder.build_supercells(
            molecule_data=mol_batch,
            cell_parameters=final_samples,
            align_to_standardized_orientation=False,
            graph_convolution_cutoff=6,
            supercell_size=1,
            pare_to_convolution_cluster=True)

        dmaxes = torch.zeros(230)
        for ind in tqdm(range(all_sg_supercells.num_graphs)):
            dists = get_intra_mol_dists(all_sg_supercells, ind)  # assumes molecules are indexed sequentially in blocks

            dmat = torch.zeros((len(dists), len(dists)))
            for i in range(len(dists)):
                for j in range(len(dists)):
                    if j > i:
                        dmat[i, j] = torch.mean(torch.abs(dists[i] - dists[j]))
            dmaxes[ind] = dmat.amax()

        assert dmaxes.amax() < 1e-3

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

        assert all(rdf_dists < 1e-1)  # RDFs should be nearly identical

        '''  # optionally look at some cells
        from mxtaltools.models.utils import ase_mol_from_crystaldata
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

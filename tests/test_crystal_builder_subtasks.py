from mxtaltools.common.config_processing import process_main_config
from mxtaltools.modeller import Modeller
from mxtaltools.crystal_building.utils import (rotvec2rotmat, aunit2unit_cell, descale_asymmetric_unit,
                                               align_molecules_to_principal_axes, batch_asymmetric_unit_pose_analysis_torch)
from scipy.spatial.transform import Rotation
from mxtaltools.common.geometry_utils import sph2rotvec, rotvec2sph, batch_molecule_principal_axes_torch
import numpy as np
import torch

'''
run tests on subtasks of the supercell builder
'''
'''load test dataset'''
config_path = r'/configs/test_configs/crystal_building.yaml'
user_path = r'/configs/Users/mikem.yaml'
config = process_main_config(user_yaml_path=user_path, main_yaml_path=config_path)
modeller = Modeller(config)
_, data_loader, _ = modeller.load_dataset_and_dataloaders(override_test_fraction=1)
modeller.init_gaussian_generator()
supercell_builder = modeller.crystal_builder
test_crystals = next(iter(data_loader))

supercell_size = 5
rotation_basis = 'spherical'


class TestClass:
    def test_rotvec2rotmat(self):
        """
        confirm transformation from rotvec to rotation matrix in cartesian and spherical base
        """
        '''check cartesian mode'''
        rotations = [Rotation.random() for _ in range(5)]
        rotvecs = torch.stack([torch.Tensor(rotation.as_rotvec()) for rotation in rotations])

        rotvecs2 = sph2rotvec(rotvec2sph(rotvecs))
        assert (rotvecs - rotvecs2).abs().mean() < 1e-4

        rotmats = rotvec2rotmat(rotvecs, basis='cartesian')
        check_rotmats = torch.stack([torch.Tensor(rotation.as_matrix()) for rotation in rotations])
        assert (rotmats - check_rotmats).abs().mean() < 1e-4

        '''check spherical mode'''
        rotations = [Rotation.random() for _ in range(5)]
        rotvecs = rotvec2sph(torch.stack([torch.Tensor(rotation.as_rotvec()) for rotation in rotations]))
        rotmats = rotvec2rotmat(rotvecs, basis='spherical')
        check_rotmats = torch.stack([torch.Tensor(rotation.as_matrix()) for rotation in rotations])
        assert (rotmats - check_rotmats).abs().mean() < 1e-4

        return None

    # todo doesn't currently work - have to set the pos argument as the canonical conformer which is not necessarily true
    def WIP_build_unit_cell(self):
        test_unit_cells = \
            aunit2unit_cell(test_crystals.sym_mult.clone(),
                            [test_crystals.pos[test_crystals.batch == ii] for ii in range(test_crystals.num_graphs)],
                            test_crystals.T_fc.clone(),
                            torch.linalg.inv(test_crystals.T_fc),
                            [torch.Tensor(test_crystals.symmetry_operators[ii]) for ii in range(test_crystals.num_graphs)]
                            )

        disagreements = torch.stack([(test_unit_cells[ii] - test_crystals.unit_cell_pos[ii]).abs().sum() for ii in range(test_crystals.num_graphs)])
        assert disagreements.mean() < 1e-4
        return None

    # todo define an assertion - right now the function itself is the best check unless we do it manually for each SG
    def WIP_scale_asymmetric_unit(self):
        space_groups = torch.tensor(np.asarray(list(supercell_builder.asym_unit_dict.keys())).astype(int))
        centroid_coords = torch.Tensor(np.random.uniform(0, 1, size=(len(space_groups), 3)))
        scaled_centroids = descale_asymmetric_unit(supercell_builder.asym_unit_dict, mol_position=centroid_coords, sg_inds=space_groups)
        return None

    # todo this check may fail for high symmetry molecules - need either to get rid of them or find a way to deal with them
    def test_align_crystaldata_to_principal_axes(self):
        '''
        align some crystaldata to cartesian axes in natural handedness
        then check that this is what happened
        '''

        aligned_test_crystals = align_molecules_to_principal_axes(test_crystals.clone(),
                                                                  handedness=test_crystals.aunit_handedness)
        aligned_principal_axes, _, _ = \
            batch_molecule_principal_axes_torch(
                [aligned_test_crystals.pos[test_crystals.batch == ii] for ii in range(test_crystals.num_graphs)])

        alignment_check = torch.eye(3).tile(aligned_test_crystals.num_graphs, 1, 1)
        alignment_check[:, 0, 0] = test_crystals.aunit_handedness

        assert torch.mean(torch.abs(alignment_check - aligned_principal_axes)) < 1e-4
        return None

    # todo this is redundant, as this same function is used to define these parameters in dataset construction
    def WIP_batch_asymmetric_unit_pose_analysis(self):
        positions, orientations, handedness, well_defined_asym_unit, canonical_conformer_coords = (
            batch_asymmetric_unit_pose_analysis_torch(
                unit_cell_coords_list=[torch.Tensor(test_crystals.unit_cell_pos[ii]) for ii in range(test_crystals.num_graphs)],
                sg_ind_list=test_crystals.sg_ind,
                asym_unit_dict=supercell_builder.asym_unit_dict,
                T_fc_list=test_crystals.T_fc,
                enforce_right_handedness=False,
                rotation_basis='cartesian',
                return_asym_unit_coords=True
            ))

        '''confirm cell params agree with dataset construction'''
        assert (positions - test_crystals.cell_params[:, 6:9]).abs().mean() < 1e-4
        assert (orientations - test_crystals.cell_params[:, 9:12]).abs().mean() < 1e-4
        assert (handedness - test_crystals.aunit_handedness).abs().mean() < 1e-4

        return None

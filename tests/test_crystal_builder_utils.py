"""
test module for crystal builder utilities
"""

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

aa = 1
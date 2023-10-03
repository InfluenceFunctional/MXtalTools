"""
integral test of full pipeline functionality
"""
from common.config_processing import get_config
from crystal_modeller import Modeller
import os

# ====================================
'''
parse arguments from config and command line and generate config namespace
'''

os.chdir('../')  # go up to main source directory
source_dir = os.getcwd()


class TestClass:
    @staticmethod
    def test_GAN():
        os.chdir(source_dir)
        config_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/test_configs/gan.yaml'
        user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/users/mkilgour.yaml'
        config = get_config(user_yaml_path=user_path, main_yaml_path=config_path)
        modeller = Modeller(config)
        modeller.train_crystal_models()

    @staticmethod
    def test_regressor():
        os.chdir(source_dir)
        config_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/test_configs/regressor.yaml'
        user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/users/mkilgour.yaml'
        config = get_config(user_yaml_path=user_path, main_yaml_path=config_path)
        modeller = Modeller(config)
        modeller.train_crystal_models()

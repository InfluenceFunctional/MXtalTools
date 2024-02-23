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


def test_model_training(config_path):
    os.chdir(source_dir)
    user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/users/mkilgour.yaml'
    config = get_config(user_yaml_path=user_path, main_yaml_path=source_dir + config_path)
    modeller = Modeller(config)
    modeller.train_crystal_models()


class TestClass:
    @staticmethod
    def test_all_models():
        for config_path in [r'/configs/test_configs/discriminator.yaml',
                            r'/configs/test_configs/regressor.yaml',
                            r'/configs/test_configs/autoencoder.yaml']:
            test_model_training(config_path)

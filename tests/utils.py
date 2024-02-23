import os

from common.config_processing import get_config
from crystal_modeller import Modeller
from tests.test_model_training import source_dir


def test_model_training(config_path):
    os.chdir(source_dir)
    user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/users/mkilgour.yaml'
    config = get_config(user_yaml_path=user_path, main_yaml_path=source_dir + config_path)
    modeller = Modeller(config)
    modeller.train_crystal_models()

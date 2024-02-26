"""
integral test of full pipeline functionality

NOTE: tends to take at least several minutes
"""
import os

from tests.utils import test_model_training

# ====================================

os.chdir('../')  # go up to main source directory
source_dir = os.getcwd()

class TestClass:
    @staticmethod
    def test_all_models():
        for config_path in [r'/configs/test_configs/discriminator.yaml',
                            r'/configs/test_configs/regressor.yaml',
                            r'/configs/test_configs/autoencoder.yaml',
                            r'/configs/test_configs/embedding_regressor.yaml']:
            test_model_training(config_path)

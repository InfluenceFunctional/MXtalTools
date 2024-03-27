"""
integral test of full pipeline functionality

NOTE: tends to take at least several minutes
"""
import os

from tests.utils import train_model

# ====================================

os.chdir('../')  # go up to main source directory
source_dir = os.getcwd()

class TestClass:
    @staticmethod
    def test_all_models():
        for config_path in [r'/configs/test_configs/discriminator.yaml',
                            r'/configs/test_configs/regressor.yaml',
                            r'/configs/test_configs/autoencoder.yaml',
                            #r'/configs/test_configs/embedding_regressor.yaml',  # need new benchmark checkpoint
                            r'/configs/test_configs/generator.yaml'
                            ]:
            os.chdir(source_dir)
            train_model(config_path)

"""
integral test of full pipeline functionality

NOTE: tends to take at least several minutes
"""
import os

import pytest

from tests.utils import train_model
from pathlib import Path

# ====================================


source_dir = Path(__file__).resolve().parent.parent
test_user_path = source_dir / "configs" / "users" / "test_user.yaml"
test_config_path = source_dir / "configs" / "test_configs"
configs = [
    # test_config_path / 'regressor.yaml',
    # test_config_path / 'discriminator.yaml',
    test_config_path / 'autoencoder.yaml'
]


@pytest.mark.slow
class TestClass:
    @staticmethod
    def test_all_models():
        for config_path in configs:
            os.chdir(source_dir)
            train_model(config_path, test_user_path)

# NOTE: this used to end with `tc = TestClass(); tc.test_all_models()` at module
# scope, so a FULL TRAINING RUN executed at COLLECTION time -- any failure became
# a collection error that aborted the entire pytest session, and `pytest
# --collect-only` trained a model.  pytest discovers the class by name.


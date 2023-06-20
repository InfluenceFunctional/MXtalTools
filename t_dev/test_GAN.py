"""
integral test of full pipeline functionality
GAN mode with all options "on"
"""

"""import statements"""
import pytest
import argparse, warnings
from common.config_processing import add_args, process_config, get_config
from crystal_modeller import Modeller

warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # ignore w&b error
warnings.filterwarnings("ignore", category=FutureWarning)


# ====================================
if __name__ == '__main__':
    '''
    parse arguments from config and command line and generate config namespace
    '''
    import os
    os.chdir('../')
    parser = argparse.ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)

    args = parser.parse_args()

    args.yaml_config = r'configs/test_configs/GAN_test.yaml'

    config = get_config(args, override_args, args2config)
    config = process_config(config)

    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))

    '''
    run the code in selected mode
    '''
    predictor = Modeller(config)
    predictor.train_crystal_models()
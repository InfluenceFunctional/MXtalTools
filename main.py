"""import statements"""
import argparse, warnings
from common.config_processing import get_config
from crystal_modeller import Modeller
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # ignore w&b error
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


# ====================================
if __name__ == '__main__':
    '''
    parse arguments from config and command line and generate config namespace
    '''
    parser = argparse.ArgumentParser()
    _, override_args = parser.parse_known_args()

    config = get_config(override_args)

    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))

    '''
    run the code in selected mode
    '''
    predictor = Modeller(config)
    if config.mode == 'gan' or config.mode == 'regression':
        predictor.train_crystal_models()

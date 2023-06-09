"""import statements"""
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
    parser = argparse.ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)

    args = parser.parse_args()

    config = get_config(args, override_args, args2config)
    config = process_config(config)

    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))

    '''
    run the code in selected mode
    '''
    predictor = Modeller(config)
    if config.mode == 'figures':
        predictor.nov_22_figures()  # figures like those from the Nov/2022 paper
    elif config.mode == 'sampling':
        predictor.model_sampling()
    else:
        predictor.train_crystal_models()

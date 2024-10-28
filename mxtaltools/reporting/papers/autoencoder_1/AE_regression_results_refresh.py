"""
refresh model results for Autoencoder paper
"""
import os
import yaml

from mxtaltools.common.config_processing import load_yaml, process_main_config
from mxtaltools.modeller import Modeller
from model_paths import er_ae_path, er_paths, targets

ae_path = er_ae_path
mxt_path = r'C:\\Users\\mikem\\PycharmProjects\\Python_Codes\\MXtalTools'

max_dataset_length = 10000

for er_path, target in zip(er_paths, targets):
    os.chdir(mxt_path)

    config = load_yaml('configs/analyses/er_analysis.yaml')
    config['model_paths']['autoencoder'] = str(ae_path)
    config['model_paths']['embedding_regressor'] = str(er_path)
    config['dataset']['max_dataset_length'] = max_dataset_length
    config['dataset']['regression_target'] = target[0]
    config['embedding_regressor']['prediction_type'] = target[1]
    config['embedding_regressor']['num_targets'] = target[2]
    config['dataset_name'] = 'test_' + target[3]

    with open('configs/analyses/analysis.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    override_args = ['--user', 'mkilgour', '--yaml_config', mxt_path + r'/configs/analyses/analysis.yaml']
    config = process_main_config(override_args, append_model_paths=False)

    predictor = Modeller(config)
    predictor.embedding_regressor_analysis()

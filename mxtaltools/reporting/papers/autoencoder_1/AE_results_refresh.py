"""
refresh model results for Autoencoder paper
"""
import os
import yaml
from model_paths import ae_paths

from mxtaltools.common.config_processing import load_yaml, process_main_config
from mxtaltools.modeller import Modeller

filter_protons = [True, False]#[False, True, False]
infer_protons = [False, False] #[False, False, True]
max_dataset_length = 1000

mxt_path = r'C:\\Users\\mikem\\PycharmProjects\\Python_Codes\\MXtalTools'

for path, filter, infer in zip(ae_paths, filter_protons, infer_protons):
    os.chdir(mxt_path)

    config = load_yaml('configs/analyses/ae_analysis.yaml')
    config['model_paths']['autoencoder'] = str(path)
    config['dataset']['filter_protons'] = filter
    config['autoencoder']['filter_protons'] = filter
    config['autoencoder']['infer_protons'] = infer
    config['dataset']['max_dataset_length'] = max_dataset_length

    with open('configs/analyses/analysis.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    override_args = ['--user', 'mkilgour', '--yaml_config', mxt_path + r'/configs/analyses/analysis.yaml']
    config = process_main_config(override_args, append_model_paths=False)

    predictor = Modeller(config)
    predictor.ae_analysis()

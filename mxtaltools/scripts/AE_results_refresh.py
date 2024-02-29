"""
refresh model results for Autoencoder paper
"""
import os
import yaml

from mxtaltools.common.config_processing import load_yaml, get_config
from mxtaltools.modeller import Modeller

ae_paths = [r'/cluster/best_autoencoder_autoencoder_tests_qm9_test23_4_26-02-22-29-57',
            r'/cluster/best_autoencoder_autoencoder_tests_qm9_test23_7_27-02-14-34-41',
            r'/cluster/best_autoencoder_autoencoder_tests_qm9_test23_8_27-02-15-35-51'
            ]

filter_protons = [True, False, False]
infer_protons = [False, False, True]
max_dataset_length = 10000000

os.chdir(r'C:\Users\mikem\OneDrive\NYU\CSD\MCryGAN')

for path, filter, infer in zip(ae_paths, filter_protons, infer_protons):
    os.chdir(r'C:\Users\mikem\OneDrive\NYU\CSD\MCryGAN')

    config = load_yaml('configs/analyses/ae_analysis.yaml')
    config['model_paths']['autoencoder'] = path
    config['dataset']['filter_protons'] = filter
    config['autoencoder']['infer_protons'] = infer
    config['dataset']['max_dataset_length'] = max_dataset_length

    with open('configs/analyses/analysis.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    override_args = ['--user', 'mkilgour', '--yaml_config', r'C:\Users\mikem\OneDrive\NYU\CSD\MCryGAN\configs/analyses/analysis.yaml']
    config = get_config(override_args)

    predictor = Modeller(config)
    predictor.autoencoder_embedding_analysis()

"""
refresh model results for Autoencoder paper
"""
import os
import yaml

from mxtaltools.common.config_processing import load_yaml, process_main_config
from mxtaltools.modeller import Modeller
from model_paths import proxy_ae_path, proxy_model_path

proxy_paths = proxy_model_path
mxt_path = r'C:\\Users\\mikem\\PycharmProjects\\Python_Codes\\MXtalTools'

max_dataset_length = 10000000

for er_path in proxy_paths:
    os.chdir(mxt_path)

    config = load_yaml('configs/analyses/pd_analysis.yaml')
    config['dataset_name'] = 'test_qm9_dataset.pt'  # test_qm9_dataset.pt
    config['model_paths']['autoencoder'] = str(proxy_ae_path)
    config['model_paths']['proxy_discriminator'] = str(er_path)
    config['dataset']['max_dataset_length'] = max_dataset_length

    with open('configs/analyses/analysis.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    override_args = ['--user', 'mkilgour', '--yaml_config', mxt_path + r'/configs/analyses/analysis.yaml']
    config = process_main_config(override_args, append_model_paths=False)

    predictor = Modeller(config)
    predictor.proxy_discriminator_analysis(samples_per_molecule=10)

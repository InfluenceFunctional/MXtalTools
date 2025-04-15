"""
refresh model results for Autoencoder paper
"""
import os
import yaml
from pathlib import Path
from mxtaltools.common.config_processing import load_yaml, process_main_config
from mxtaltools.modeller import Modeller
import torch
from mxtaltools.common.utils import namespace2dict
from mxtaltools.reporting.papers.autoencoder_1.AE_figures import proxy_discriminator_figure

mxt_path = r'C:\\Users\\mikem\\PycharmProjects\\Python_Codes\\MXtalTools'

max_dataset_length = 10000000

pd_checkpoint_dir = r"D:\crystal_datasets\proxy_checkpoints"
proxy_paths = os.listdir(pd_checkpoint_dir)
proxy_paths = [elem for elem in proxy_paths if 'proxy_discriminator_tests_zinc_test7' in elem]
proxy_paths = [str(Path(pd_checkpoint_dir).joinpath(Path(elem))) for elem in proxy_paths]
ae_path = r'C:\Users\mikem\PycharmProjects\Python_Codes\MXtalTools\checkpoints\autoencoder.pt'
for er_path in proxy_paths:
    os.chdir(mxt_path)

    model_config = torch.load(er_path)['config']

    config = load_yaml('configs/analyses/pd_analysis.yaml')
    config['dataset_name'] = 'eval_pd_dataset_sg1.pt'  # test_qm9_dataset.pt
    config['model_paths']['autoencoder'] = ae_path
    config['model_paths']['proxy_discriminator'] = er_path
    config['dataset']['max_dataset_length'] = max_dataset_length
    config['proxy_discriminator'] = namespace2dict(model_config)  # overwrite PD settings

    with open('configs/analyses/analysis.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    override_args = ['--user', 'mkilgour', '--yaml_config', mxt_path + r'/configs/analyses/analysis.yaml']
    config = process_main_config(override_args, append_model_paths=False)

    predictor = Modeller(config)
    predictor.pd_evaluation()
    del predictor

"""collect results and make fig"""
fig5 = proxy_discriminator_figure()


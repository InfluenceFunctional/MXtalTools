"""
refresh model results for Autoencoder paper
"""
import os
import yaml

from mxtaltools.common.config_processing import load_yaml, get_config
from mxtaltools.modeller import Modeller

ae_path = r'/cluster/best_autoencoder_autoencoder_tests_qm9_test23_7_27-02-14-34-41'

regressor_paths = os.listdir(r'C:\Users\mikem\crystals\CSP_runs\models\cluster\ae_regressor')

targets = ["molecule_rotational_constant_a",  # 0
           "molecule_rotational_constant_b",  # 1
           "molecule_rotational_constant_c",  # 2
           "molecule_dipole_moment",  # 3
           "molecule_isotropic_polarizability",  # 4
           "molecule_HOMO_energy",  # 5
           "molecule_LUMO_energy",  # 6
           "molecule_gap_energy",  # 7
           "molecule_el_spatial_extent",  # 8
           "molecule_zpv_energy",  # 9
           "molecule_internal_energy_0",  # 10
           "molecule_internal_energy_STP",  # 11
           "molecule_enthalpy_STP",  # 12
           "molecule_free_energy_STP",  # 13
           "molecule_heat_capacity_STP",  # 14
           ]

max_dataset_length = 10000000

for path, target in zip(regressor_paths, targets):
    os.chdir(r'C:\Users\mikem\OneDrive\NYU\CSD\MCryGAN')

    config = load_yaml('configs/analyses/er_analysis.yaml')
    config['model_paths']['autoencoder'] = ae_path
    config['model_paths']['embedding_regressor'] = '/cluster/ae_regressor/' + path
    config['dataset']['filter_protons'] = False
    config['dataset']['max_dataset_length'] = max_dataset_length
    config['dataset']['regression_target'] = target

    with open('configs/analyses/analysis.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    override_args = ['--user', 'mkilgour', '--yaml_config', r'C:\Users\mikem\OneDrive\NYU\CSD\MCryGAN\configs/analyses/analysis.yaml']
    config = get_config(override_args)

    predictor = Modeller(config)
    predictor.embedding_regressor_analysis()

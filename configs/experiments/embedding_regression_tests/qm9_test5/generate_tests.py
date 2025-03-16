from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
Regress over all scalar QM9 Properties
"""

search_space = {
    'dataset': {'regression_target': [
        "molecule_rotational_constant_a",  # 0
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
    ]},
    'models': [
        'best_autoencoder_autoencoder_tests_qm9_test21_18_13-02-10-13-43',
        'best_autoencoder_autoencoder_tests_qm9_test21_20_13-02-10-13-47',
    ]
}
n_runs = len(search_space['dataset']['regression_target']) * 2
for ii in range(n_runs):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ii)
    config['dataset']['regression_target'] = search_space['dataset']['regression_target'][ii//2]
    if ii % 2 == 0:
        config['model_paths']['autoencoder'] = search_space['models'][0]
    else:
        config['model_paths']['autoencoder'] = search_space['models'][1]

    with open(str(ii) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

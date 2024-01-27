from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
Regress over all scalar QM9 Properties
"""

search_space = {
    'dataset': {'regression_target': [
        "molecule_rotational_constant_a",
        "molecule_rotational_constant_b",
        "molecule_rotational_constant_c",
        "molecule_dipole_moment",
        "molecule_isotropic_polarizability",
        "molecule_HOMO_energy",
        "molecule_LUMO_energy",
        "molecule_gap_energy",
        "molecule_el_spatial_extent",
        "molecule_zpv_energy",
        "molecule_internal_energy_0",
        "molecule_internal_energy_STP",
        "molecule_enthalpy_STP",
        "molecule_free_energy_STP",
        "molecule_heat_capacity_STP",
    ]}

}
np.random.seed(1)
n_runs = len(search_space['dataset']['regression_target'])
for ii in range(n_runs):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ii)
    config['dataset']['regression_target'] = search_space['dataset']['regression_target'][ii]

    with open(str(ii) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

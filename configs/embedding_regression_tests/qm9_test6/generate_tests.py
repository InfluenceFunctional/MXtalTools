from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('../../experiments/base/embedding_regressor.yaml')

models = [r'/best_autoencoder_autoencoder_tests_qm9_test23_4_26-02-22-29-57',  # no protons
          r'/best_autoencoder_autoencoder_tests_qm9_test23_7_27-02-14-34-41'  # protons
          ]

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
# model, filter_protons
# depth, num_layers, dropout, norm_mode


config_list = [
    [  # 0: no protons baseline
        models[0], True, targets[7],
        64, 12, 0.1, 'layer'
    ],
    [  # 1: no protons large
        models[0], True, targets[7],
        512, 12, 0.1, 'layer'
    ],
    [  # 2: protons baseline
        models[1], False, targets[7],
        64, 12, 0.1, 'layer'
    ],
    [  # 3: protons large
        models[1], False, targets[7],
        512, 12, 0.1, 'layer'
    ],
]

ind = 0
for ix1 in range(len(config_list)):

    config = copy(base_config)
    config['machine'] = 'cluster'
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    config['dataset']['filter_protons'] = config_list[ix1][1]
    config['model_paths']['autoencoder'] = config_list[ix1][0]
    config['dataset']['regression_target'] = config_list[ix1][2]
    config['embedding_regressor']['model']['depth'] = config_list[ix1][3]
    config['embedding_regressor']['model']['num_layers'] = config_list[ix1][4]
    config['embedding_regressor']['model']['dropout'] = config_list[ix1][5]
    config['embedding_regressor']['model']['norm_mode'] = config_list[ix1][6]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

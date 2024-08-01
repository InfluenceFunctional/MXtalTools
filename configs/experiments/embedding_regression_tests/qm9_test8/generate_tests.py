from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

# test our standard embedding regressor with unfrozen and untrained autoencoder

base_config = load_yaml('../../base/embedding_regressor.yaml')

models = [r'/best_autoencoder_autoencoder_tests_qm9_test23_4_26-02-22-29-57_fixed',  # no protons
          r'/best_autoencoder_experiments_autoencoder_tests_qm9_test24_25_11-05-19-15-19_fixed'  # protons
          ]

targets = ["rotational_constant_a",  # 0
           "rotational_constant_b",  # 1
           "rotational_constant_c",  # 2
           "dipole_moment",  # 3
           "isotropic_polarizability",  # 4
           "HOMO_energy",  # 5
           "LUMO_energy",  # 6
           "gap_energy",  # 7
           "el_spatial_extent",  # 8
           "zpv_energy",  # 9
           "internal_energy_0",  # 10
           "internal_energy_STP",  # 11
           "enthalpy_STP",  # 12
           "free_energy_STP",  # 13
           "heat_capacity_STP",  # 14
           ]
# model, filter_protons
# depth, num_layers, dropout, norm_mode
# max_dataset_length


config_list = [
    [  # 0: no protons baseline - worst
        models[0], True, targets[7],
        64, 12, 0.1, 'layer', 10000000
    ],
    [  # 1: no protons large - second best
        models[0], True, targets[7],
        512, 12, 0.1, 'layer', 10000000
    ],
    [  # 2: protons baseline - second best
        models[1], False, targets[7],
        64, 12, 0.1, 'layer', 10000000
    ],
    [  # 3: protons large - by far the best
        models[1], False, targets[7],
        512, 12, 0.1, 'layer', 10000000
    ],
    [  # 3
        models[1], False, targets[7],
        256, 12, 0.1, 'layer', 10000000
    ],
    [  # 5
        models[1], False, targets[7],
        512, 12, 0.25, 'layer', 10000000
    ],
    [  # 6 - best test
        models[1], False, targets[7],
        512, 4, 0.1, 'layer', 10000000
    ],
    [  # 7
        models[1], False, targets[7],
        512, 12, 0.1, 'batch', 10000000
    ],
    [  # 8
        models[1], False, targets[7],
        512, 12, 0.1, 'layer', 10000000
    ],
    [  # 9
        models[1], False, targets[7],
        128, 20, 0.1, 'layer', 10000000
    ],
    [  # 10
        models[1], False, targets[7],
        1024, 4, 0.1, 'layer', 10000000
    ],
]

for ind in range(len(targets)):  # 11-25 - main production runs
    bb = copy(config_list[6])
    bb[2] = targets[ind]
    config_list.append(bb)

ind = 0
for ix1 in range(len(config_list)):
    config = copy(base_config)
    config['machine'] = 'cluster'
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    config['dataset']['filter_protons'] = config_list[ix1][1]
    config['model_paths']['autoencoder'] = config_list[ix1][0]
    config['dataset']['regression_target'] = config_list[ix1][2]
    config['embedding_regressor']['freeze_encoder'] = False
    config['embedding_regressor']['model']['depth'] = config_list[ix1][3]
    config['embedding_regressor']['model']['num_layers'] = config_list[ix1][4]
    config['embedding_regressor']['model']['dropout'] = config_list[ix1][5]
    config['embedding_regressor']['model']['norm_mode'] = config_list[ix1][6]
    config['dataset']['max_dataset_length'] = config_list[ix1][7]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

models = [
    r'best_autoencoder_experiments_autoencoder_tests_qm9_test30_0_28-10-15-51-07'  # model with protons
    ]

targets = [
    ["rotational_constant_a", 'scalar', 1, 'qm9_dataset.pt'],  # 0
    ["rotational_constant_b", 'scalar', 1, 'qm9_dataset.pt'],  # 1
    ["rotational_constant_c", 'scalar', 1, 'qm9_dataset.pt'],  # 2
    ["dipole_moment", 'scalar', 1, 'qm9_dataset.pt'],  # 3
    ["isotropic_polarizability", 'scalar', 1, 'qm9_dataset.pt'],  # 4
    ["HOMO_energy", 'scalar', 1, 'qm9_dataset.pt'],  # 5
    ["LUMO_energy", 'scalar', 1, 'qm9_dataset.pt'],  # 6
    ["gap_energy", 'scalar', 1, 'qm9_dataset.pt'],  # 7
    ["el_spatial_extent", 'scalar', 1, 'qm9_dataset.pt'],  # 8
    ["zpv_energy", 'scalar', 1, 'qm9_dataset.pt'],  # 9
    ["internal_energy_0", 'scalar', 1, 'qm9_dataset.pt'],  # 10
    ["internal_energy_STP", 'scalar', 1, 'qm9_dataset.pt'],  # 11
    ["enthalpy_STP", 'scalar', 1, 'qm9_dataset.pt'],  # 12
    ["free_energy_STP", 'scalar', 1, 'qm9_dataset.pt'],  # 13
    ["heat_capacity_STP", 'scalar', 1, 'qm9_dataset.pt'],  # 14
    ["dipole", 'vector', 1, 'qm9s_dataset.pt'],  # 15
    ["polar", '2-tensor', 16, 'qm9s_dataset.pt'],  # 16
    ["quadrupole", '2-tensor', 16, 'qm9s_dataset.pt'],  # 17
    ["octapole", '3-tensor', 16, 'qm9s_dataset.pt'],  # 18
    ["hyperpolar", '3-tensor', 16, 'qm9s_dataset.pt'],  # 19
]
# model, filter_protons
# depth, num_layers, dropout, norm_mode
# max_dataset_length, prediction_type, num_outputs, dataset_name

config_list = [
    [  # 0 - baseline
        models[0], False, targets[0],
        256, 4, 0.1, None, 10000000,
        'scalar', 1, 'qm9_dataset.pt'
    ],
]

for ind in range(len(targets)):  # 0-19 main production runs
    bb = copy(config_list[0])
    bb[2] = targets[ind][0]
    bb[8] = targets[ind][1]
    bb[9] = targets[ind][2]
    bb[10] = targets[ind][3]

    config_list.append(bb)

config_list = config_list[1:]
ind = 0
for ix1 in range(
        len(config_list)):  # note for later use - 'depth' here is not a config which is used! we were fooling ourselves!!
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
    config['dataset']['max_dataset_length'] = config_list[ix1][7]
    config['embedding_regressor']['prediction_type'] = config_list[ix1][8]
    config['embedding_regressor']['num_targets'] = config_list[ix1][9]
    config['dataset_name'] = config_list[ix1][10]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

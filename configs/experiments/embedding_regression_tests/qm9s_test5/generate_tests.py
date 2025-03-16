from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

models = [
    r'best_autoencoder_experiments_autoencoder_tests_qm9_test30_0_28-10-15-51-07'  # model with protons
    ]

# model, filter_protons
# depth, num_layers, dropout, norm_mode
# max_dataset_length, prediction_type, num_outputs, dataset_name


config_list = [
    [  # 0 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0, None, 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 1 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0, None, 10000000,
        '3-tensor', 32, 'qm9s_dataset.pt',
        2000
    ],
    [  # 2 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0, None, 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        500
    ],
    [  # 3 - baseline
        models[0], False, 'hyperpolar',
        256, 4, 0, None, 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 4 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0.25, None, 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 5 - baseline
        models[0], False, 'hyperpolar',
        256, 4, 0.25, None, 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 6 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0, 'batch', 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 7 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0.25, 'batch', 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 8 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0, 'layer', 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 9 - baseline
        models[0], False, 'hyperpolar',
        512, 2, 0, None, 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 10 - baseline
        models[0], False, 'hyperpolar',
        512, 4, .5, 'batch', 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        2000
    ],
    [  # 11 - baseline
        models[0], False, 'hyperpolar',
        512, 4, .5, 'batch', 10000000,
        '3-tensor', 16, 'qm9s_dataset.pt',
        500
    ],
    [  # 12 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0, None, 10000000,
        '3-tensor', 32, 'qm9s_dataset.pt',
        10000
    ],
    [  # 13 - baseline
        models[0], False, 'hyperpolar',
        128, 4, 0, None, 10000000,
        '3-tensor', 64, 'qm9s_dataset.pt',
        2000
    ],
    [  # 14 - baseline
        models[0], False, 'hyperpolar',
        128, 12, 0, None, 10000000,
        '3-tensor', 32, 'qm9s_dataset.pt',
        2000
    ],
    [  # 15 - baseline
        models[0], False, 'hyperpolar',
        128, 12, 0, None, 10000000,
        '3-tensor', 64, 'qm9s_dataset.pt',
        10000
    ],
]


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
    config['dataset']['max_batch_size'] = config_list[ix1][11]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

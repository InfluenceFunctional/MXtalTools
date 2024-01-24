from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
big random attack for variational autoencoder
"""

search_space = {
    'model': {
        'num_graph_convolutions': [0, 1, 4],
        'embedding_depth': [64, 128, 256, 512],
        'bottleneck_dim': [32, 64, 128, 256],
        'nodewise_fc_layers': [1, 2, 4, 8],
        'num_decoder_layers': [1, 2, 4, 8],
        'decoder_ramp_depth': [False, True],
        'num_decoder_points': [64, 128, 256, 512],

    },
    'optimizer': {
        'weight_decay': [0.01, 0.05, 0.1],
        'encoder_init_lr': [5e-4, 1e-4, 1e-3],
        'decoder_init_lr': [5e-4, 1e-4, 1e-3],
    },
    'KLD_weight': [0.01, 0.1, 0.5],

}
np.random.seed(1)
ind = 0
n_runs = 200
for ii in range(n_runs):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    config['autoencoder']['KLD_weight'] = search_space['KLD_weight'][np.random.randint(0, len(search_space['KLD_weight']))]

    for subgroup in ['model', 'optimizer']:
        for key in search_space[subgroup].keys():
            options = search_space[subgroup][key]
            config['autoencoder'][subgroup][key] = options[np.random.randint(0, len(options))]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

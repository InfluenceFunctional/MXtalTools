from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
Results - these were all competitive - convergint to about .94 rather quickly with minimal overfitting
"""

search_space = {
    'model': {
        'num_graph_convolutions': [1],
        'embedding_depth': [128, 256],
        'bottleneck_dim': [256],
        'nodewise_fc_layers': [8],
        'num_decoder_layers': [2, 4],
        'decoder_ramp_depth': [True],
        'num_decoder_points': [128],
        'decoder_norm_mode': ['batch', 'layer', None],
        'variational_encoder': [True, False],
    },
    'optimizer': {
        'weight_decay': [0.1],
        'encoder_init_lr': [5e-4],
        'decoder_init_lr': [5e-4],
    },
    'KLD_weight': [0.001],
    'dataset': {'filter_protons': [True]},
    'autoencoder_positional_noise': [0],
}
# norm mode, decoder layers, decoder points, weight decay, kld, filter protons, positional noise
config_list = [
    ['batch', 4, 128, .1, 0.001, True, 0, 5e-4, 5e-4],  # 0
    ['layer', 4, 128, .1, 0.001, True, 0, 5e-4, 5e-4],  # 1
    [None, 4, 128, .1, 0.001, True, 0, 5e-5, 5e-5],  # 2  #NaN
    ['batch', 8, 128, .1, 0.001, True, 0, 5e-4, 5e-4],  # 3  # immediate NaN 2nd try
    ['batch', 4, 256, .1, 0.001, True, 0, 5e-5, 5e-5],  # 4  # NaN
    ['batch', 4, 128, .75, 0.001, True, 0, 5e-4, 5e-4],  # 5
    ['batch', 4, 128, .1, 0.1, True, 0, 5e-5, 5e-5],  # 6 NaN
    ['batch', 4, 128, .1, 0.001, False, 0, 5e-5, 5e-5],  # 7  # NaN
    ['batch', 4, 128, .1, 0.001, True, 0.5, 5e-4, 5e-4],  # 8
]

np.random.seed(1)
ind = 0
for ix1 in range(len(config_list)):

    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    config['autoencoder']['model']['decoder_norm_mode'] = config_list[ix1][0]
    config['autoencoder']['model']['num_decoder_layers'] = config_list[ix1][1]
    config['autoencoder']['model']['num_decoder_points'] = config_list[ix1][2]
    config['autoencoder']['optimizer']['weight_decay'] = config_list[ix1][3]
    config['autoencoder']['KLD_weight'] = config_list[ix1][4]
    config['dataset']['filter_protons'] = config_list[ix1][5]
    config['autoencoder_positional_noise'] = config_list[ix1][6]
    config['autoencoder']['optimizer']['encoder_init_lr'] = config_list[ix1][7]
    config['autoencoder']['optimizer']['decoder_init_lr'] = config_list[ix1][8]


    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

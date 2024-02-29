from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('../../experiments/base/autoencoder.yaml')

# filter_protons, infer protons, variational, ramp depth
# decoder layers, conv layers, nodewise layers, decoder norm
# decoder points, embedding dim, bottleneck dim, dropout
# positional noise, weight decay, overlap_eps, max_batch
# lr shrink lambda, batch growth increment, min_lr, max_lr, KLD_threshold
config_list = [
    [  # 0: baseline config
        True, False, True, True,
        1, 1, 4, 'layer',
        512, 512, 512, 0.1,
        0, 0.05, 0.001, 300,
        0.99, 0.05, 1e-6, 2e-4, 0.97
    ],
    [  # 1: shallow - high KLD bad loss
        True, False, True, True,
        1, 1, 1, 'layer',
        512, 512, 512, 0.1,
        0, 0.05, 0.001, 300,
        0.99, 0.05, 1e-6, 2e-4, 0.97
    ],
    [  # 2: deep with heavy bottleneck - OK
        True, False, True, True,
        4, 1, 4, 'layer',
        512, 512, 128, 0.1,
        0, 0.05, 0.001, 300,
        0.99, 0.05, 1e-6, 2e-4, 0.97
    ],
    [  # 3: deeper with heavy bottleneck - worst
        True, False, True, True,
        8, 1, 4, 'layer',
        512, 512, 128, 0.1,
        0, 0.05, 0.001, 300,
        0.99, 0.05, 1e-6, 2e-4, 0.97
    ],
    [  # 4: deep and narrow - good KLD, decent loss but best train loss - best overfit
        True, False, True, True,
        8, 1, 4, None,
        512, 128, 128, 0,
        0, 0.05, 0.001, 300,
        0.99, 0.05, 1e-6, 2e-4, 0.97
    ],
]

ind = 0
for ix1 in range(len(config_list)):

    config = copy(base_config)
    config['machine'] = 'cluster'
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    config['dataset']['filter_protons'] = config_list[ix1][0]
    config['autoencoder']['infer_protons'] = config_list[ix1][1]
    config['autoencoder']['model']['variational_encoder'] = config_list[ix1][2]
    config['autoencoder']['model']['decoder_ramp_depth'] = config_list[ix1][3]

    config['autoencoder']['model']['num_decoder_layers'] = config_list[ix1][4]
    config['autoencoder']['model']['num_graph_convolutions'] = config_list[ix1][5]
    config['autoencoder']['model']['nodewise_fc_layers'] = config_list[ix1][6]
    config['autoencoder']['model']['decoder_norm_mode'] = config_list[ix1][7]

    config['autoencoder']['model']['num_decoder_points'] = config_list[ix1][8]
    config['autoencoder']['model']['embedding_depth'] = config_list[ix1][9]
    config['autoencoder']['model']['bottleneck_dim'] = config_list[ix1][10]
    config['autoencoder']['model']['graph_node_dropout'] = config_list[ix1][11]
    config['autoencoder']['model']['decoder_dropout_probability'] = config_list[ix1][11]

    config['positional_noise']['autoencoder'] = config_list[ix1][12]
    config['autoencoder']['optimizer']['weight_decay'] = config_list[ix1][13]
    config['autoencoder']['overlap_eps']['test'] = config_list[ix1][14]
    config['max_batch_size'] = config_list[ix1][15]

    config['autoencoder']['optimizer']['lr_shrink_lambda'] = config_list[ix1][16]
    config['batch_growth_increment'] = config_list[ix1][17]
    config['autoencoder']['optimizer']['min_lr'] = config_list[ix1][18]
    config['autoencoder']['optimizer']['max_lr'] = config_list[ix1][19]
    config['autoencoder']['KLD_threshold'] = config_list[ix1][20]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

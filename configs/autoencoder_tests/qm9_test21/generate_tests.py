from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

# decoder layers, decoder points, weight decay, filter protons, positional noise, dropout, embedding_dim, bottleneck_dim, num_convs, num_nodewise
config_list = [
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4],  # 0
    [4, 256, 0.05, False, 0, 0, 256, 256, 1, 4],  # 1
    [4, 256, 0.05, True, 0, 0, 256, 256, 0, 4],  # 2
    [4, 256, 0.05, False, 0, 0, 256, 256, 0, 4],  # 3
    [4, 256, 0.05, True, 0, 0, 256, 256, 2, 2],  # 4
    [4, 256, 0.05, False, 0, 0, 256, 256, 2, 2],  # 5
]

np.random.seed(1)
ind = 0
for ix1 in range(len(config_list)):

    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    config['autoencoder']['model']['num_decoder_layers'] = config_list[ix1][0]
    config['autoencoder']['model']['num_decoder_points'] = config_list[ix1][1]
    config['autoencoder']['optimizer']['weight_decay'] = config_list[ix1][2]
    config['dataset']['filter_protons'] = config_list[ix1][3]
    config['autoencoder_positional_noise'] = config_list[ix1][4]
    config['autoencoder']['model']['graph_node_dropout'] = config_list[ix1][5]
    config['autoencoder']['model']['decoder_dropout_probability'] = config_list[ix1][5]
    config['autoencoder']['model']['embedding_depth'] = config_list[ix1][6]
    config['autoencoder']['model']['bottleneck_dim'] = config_list[ix1][7]
    config['autoencoder']['model']['num_graph_convolutions'] = config_list[ix1][8]
    config['autoencoder']['model']['nodewise_fc_layers'] = config_list[ix1][9]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

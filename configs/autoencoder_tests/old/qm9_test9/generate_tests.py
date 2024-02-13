from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
- testing new equivariant model again



"""

configs = [
    [2, 171, 171, 4, 4, 0.1, 0.995, 0.25, 3],  # 0
    [2, 171, 171, 4, 4, 0.1, 0.999, 0.25, 3],  # 1
    [1, 171, 171, 4, 4, 0.1, 0.995, 0.25, 3],  # 2
    [2, 342, 171, 4, 4, 0.1, 0.995, 0.25, 3],  # 3
    [2, 171, 57, 4, 4, 0.1, 0.995, 0.25, 3],  # 4
    [2, 171, 171, 2, 4, 0.1, 0.995, 0.25, 3],  # 5
    [2, 171, 171, 4, 2, 0.1, 0.995, 0.25, 3],  # 6
    [2, 171, 171, 4, 4, 0.01, 0.995, 0.25, 3],  # 7
    [2, 171, 171, 4, 4, 1, 0.995, 0.25, 3],  # 8
    [2, 171, 171, 4, 4, 0.1, 0.995, 0, 3],  # 9
    [2, 171, 171, 4, 4, 0.1, 0.995, 0.25, 9],  # 10
]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['autoencoder']['model']['num_graph_convolutions'] = configs[ii][0]
    config['autoencoder']['model']['embedding_depth'] = configs[ii][1]
    config['autoencoder']['model']['graph_message_depth'] = configs[ii][2]
    config['autoencoder']['model']['nodewise_fc_layers'] = configs[ii][3]
    config['autoencoder']['model']['num_decoder_layers'] = configs[ii][4]
    config['autoencoder']['optimizer']['weight_decay'] = configs[ii][5]
    config['autoencoder']['optimizer']['lr_shrink_lambda'] = configs[ii][6]
    config['autoencoder_positional_noise'] = configs[ii][7]
    config['autoencoder']['model']['num_attention_heads'] = configs[ii][8]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

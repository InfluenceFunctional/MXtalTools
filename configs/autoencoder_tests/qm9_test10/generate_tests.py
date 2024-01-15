from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
- testing new equivariant model again



"""

configs = [
    [1, 171, 171, 1, 1, 1],  # 0
    [0, 513, 171, 1, 1, 1],  # 1
    [0, 513, 171, 4, 4, 1],  # 2
    [2, 513, 171, 1, 4, 9],  # 3
    [4, 513, 171, 1, 4, 9],  # 4
    [4, 513, 171, 1, 8, 9],  # 5
    [4, 513, 513, 1, 8, 27],  # 6
    [8, 513, 513, 1, 8, 27],  # 7
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
    config['autoencoder']['model']['num_attention_heads'] = configs[ii][5]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

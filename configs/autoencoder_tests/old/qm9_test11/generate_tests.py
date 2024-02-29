from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
- testing new equivariant model again


Seeing large model problems here
-: significant overfitting
-: NaN outputs
"""

configs = [
    [0, 171, 171, 8, 1, 9, 0],  # 0  decent convergence without overfitting
    [0, 171, 171, 8, 8, 9, 0.5],  # 1  great convergence with some overfitting before crashing
    [4, 171, 171, 1, 4, 9, 0],  # 2  2nd best convergence minimal overfitting
    [4, 57, 57, 1, 4, 3, 0],  # 3 slow and gradual convergence
    [8, 171, 171, 1, 8, 9, 0],  # 4 crashed early OK performance
    [4, 513, 171, 1, 4, 9, 0],  # 5  very slow, not really converging
    [0, 57, 57, 8, 8, 9, 0.5],  # 6
    [0, 57, 57, 8, 8, 9, 0],  # 7
    [0, 57, 57, 12, 12, 9, 0],  # 8  NaN in types crash
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
    config['autoencoder_positional_noise'] = configs[ii][6]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

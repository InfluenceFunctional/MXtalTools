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
    [1, 171, 171, 1, 1, 1, 1e-4],  # 0
    [0, 513, 171, 1, 1, 1, 1e-4],  # 1
    [0, 513, 171, 4, 4, 1, 1e-4],  # 2
    [2, 513, 171, 1, 4, 9, 1e-5],  # 3  wacky explosion
    [4, 513, 171, 1, 4, 9, 1e-5],  # 4 CUDA error
    [4, 513, 171, 1, 8, 9, 1e-5],  # 5 NaN out
    [4, 513, 513, 1, 8, 27, 1e-5],  # 6 NaN out
    [8, 513, 513, 1, 8, 27, 1e-5],  # 7 NaN out
    [1, 171, 171, 4, 4, 9, 1e-4],  # 8
    [2, 171, 171, 4, 4, 9, 1e-4],  # 9
    [3, 171, 171, 4, 4, 9, 1e-4],  # 10
    [4, 171, 171, 4, 4, 9, 1e-4],  # 11

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
    config['autoencoder']['optimizer']['encoder_init_lr'] = configs[ii][6]


    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

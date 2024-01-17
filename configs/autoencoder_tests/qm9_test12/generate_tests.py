from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
- testing new equivariant model again
looking for fast training with minimal overfitting and good stability

1) train small no-conv nets faster
2) is there a goldilocks conv net
3) keep mucking about with conv options
"""

configs = [
    [0, 56, 56, 8, 8, 9, 0, 1e-3, 512, False],  # 0

    [0, 76, 76, 8, 8, 9, 0, 1e-4, 512, False],  # 1
    [0, 128, 128, 8, 8, 9, 0, 1e-4, 512, False],  # 2
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 512, False],  # 3
    [0, 513, 513, 8, 1, 9, 0, 1e-4, 512, False],  # 4

    [0, 128, 128, 8, 8, 9, 0, 1e-3, 512, False],  # 5

    [0, 171, 171, 1, 8, 9, 0, 1e-4, 512, False],  # 6
    [0, 342, 342, 1, 8, 9, 0, 1e-4, 512, False],  # 7
    [0, 513, 513, 1, 8, 9, 0, 1e-4, 512, False],  # 8

    [4, 171, 171, 1, 4, 9, 0, 1e-4, 256, False],  # 9
    [4, 171, 171, 1, 4, 9, 0, 1e-4, 768, False],  # 10

    [4, 171, 171, 1, 4, 9, 0, 1e-4, 512, True],  # 11
    [0, 128, 128, 8, 8, 9, 0, 1e-4, 512, True],  # 12

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
    config['autoencoder']['optimizer']['encoder_init_lr'] = configs[ii][7]
    config['autoencoder']['model']['num_decoder_points'] = configs[ii][8]
    config['autoencoder']['model']['decoder_ramp_depth'] = configs[ii][9]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

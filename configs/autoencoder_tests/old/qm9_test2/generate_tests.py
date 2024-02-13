from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
1) test performance on converging the QM9
2) check positional noise & embedding depth
"""

configs = [
    [0, 513, 2],  # not good
    [.5, 513, 2],  # not good
    [1, 513, 2],  # not good
    [0, 1026, 2],  # good start but crashed
    [0, 2049, 2],  # crashed early
    [0, 513, 4],  # best by far
    [0, 1026, 1],  # not converging - went hard into a particular type, maya suddenly drop
]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['autoencoder_positional_noise'] = configs[ii][0]
    config['autoencoder']['model']['embedding_depth'] = configs[ii][1]
    config['autoencoder']['model']['nodewise_fc_layers'] = configs[ii][2]
    config['autoencoder']['model']['num_decoder_layers'] = configs[ii][2]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

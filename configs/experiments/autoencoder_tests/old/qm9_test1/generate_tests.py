from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
1) test performance on converging the QM9
2) check positional noise & embedding depth
"""

configs = [
    [0, 64],  # bad
    [0, 128],  # not as bad
    [0, 256],  # intermediate quality
    [0, 512],  # excellent
    [.1, 64],  # bad
    [.1, 512],  # excellent
    [.5, 64],  # bad
    [.5, 512],  # excellent
    [1, 64],  # bad
    [1, 512]  # excellent
]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['autoencoder_positional_noise'] = configs[ii][0]
    config['autoencoder']['model']['embedding_depth'] = configs[ii][1]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

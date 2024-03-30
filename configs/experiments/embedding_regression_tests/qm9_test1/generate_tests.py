from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
Converge regression model based on molecule regressor
"""

configs = [
    [1, 256],  # awful
    [4, 256],  # middle/bad
    [12, 256],  # pretty good
    [1, 512],  # awful
    [4, 512],  # middle/bad
    [12, 512],  # middle/bad
    [16, 256],  # third best
    [4, 1024],  # pretty good
    [24, 256],  # second best
    [40, 256],  # best
]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['embedding_regressor']['model']['num_layers'] = configs[ii][0]
    config['embedding_regressor']['model']['depth'] = configs[ii][1]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
Converge regression model based on molecule regressor
"""

configs = [
    [0, None],  # third best ever
    [0.1, None],  # middling
    [0.5, None],  # bad
    [0, 'batch'],  # second best ever
    [0, 'layer'],  # best ever
    [0.5, 'batch'],  # bad
]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['embedding_regressor']['model']['dropout'] = configs[ii][0]
    config['embedding_regressor']['model']['norm_mode'] = configs[ii][1]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

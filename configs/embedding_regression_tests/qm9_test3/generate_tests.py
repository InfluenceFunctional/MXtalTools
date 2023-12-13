from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
Converge regression model based on molecule regressor
"""

configs = [
    [60, 256],
    [80, 256],
    [40, 512],
    [60, 512],
    [20, 1024],
    [40, 1024],
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

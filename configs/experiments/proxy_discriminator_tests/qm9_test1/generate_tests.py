from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

"""
Proxy discriminator params
"""

configs = [
    [2, 256],  #
    [4, 256],  #
    [8, 256],  #
    [2, 512],  #
]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['proxy_discriminator']['model']['num_layers'] = configs[ii][0]
    config['proxy_discriminator']['model']['depth'] = configs[ii][1]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

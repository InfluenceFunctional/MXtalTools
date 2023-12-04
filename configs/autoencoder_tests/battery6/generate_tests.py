from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
goal of this battery is to try to converge some random point clouds
"""

random_fraction = [0, 0.33, 0.66, 1]
decoder_depth = [2, 4, 8]

ind = 0
for i1 in range(len(random_fraction)):
    for i2 in range(len(decoder_depth)):
        config = copy(base_config)
        config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
        config['autoencoder']['random_fraction'] = random_fraction[i1]
        config['autoencoder']['num_decoder_layers'] = decoder_depth[i2]

        with open(str(ind) + '.yaml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        ind += 1

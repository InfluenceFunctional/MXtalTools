from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
- testing new equivariant model again



"""

configs = [
    [1, 171, 4, 4],
    [2, 171, 4, 4],
    [3, 171, 4, 4],
    [2, 342, 4, 4],
    [2, 171, 2, 4],
    [2, 171, 4, 2],
    [2, 171, 8, 4],
    [2, 171, 4, 8],

]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['autoencoder']['model']['num_graph_convolutions'] = configs[ii][0]
    config['autoencoder']['model']['embedding_depth'] = configs[ii][1]
    config['autoencoder']['model']['nodewise_fc_layers'] = configs[ii][2]
    config['autoencoder']['model']['num_decoder_layers'] = configs[ii][3]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

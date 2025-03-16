from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
from previous tests:
- deep decoder with lots of points
- shallow encoder with lots of FCs
- deep but not too-deep embedding

open question: role of optimizer?
result: possibly stochasticity plays a big role here
"""

configs = [  # none better than test3 best
    [2, 513, 4, 4, 1024],  # tied with 1 early but crashed
    [2, 513, 4, 4, 2048],  # best here
    [1, 513, 8, 4, 512],  # meh
    [1, 513, 4, 8, 512],  # meh
    [1, 513, 8, 8, 512],  # meh
    [1, 513, 8, 8, 1024],  # meh
]


ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['autoencoder']['model']['num_graph_convolutions'] = configs[ii][0]
    config['autoencoder']['model']['embedding_depth'] = configs[ii][1]
    config['autoencoder']['model']['nodewise_fc_layers'] = configs[ii][2]
    config['autoencoder']['model']['num_decoder_layers'] = configs[ii][3]
    config['autoencoder']['model']['num_decoder_points'] = configs[ii][4]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

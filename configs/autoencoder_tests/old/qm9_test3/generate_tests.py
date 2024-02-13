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
    [1, 513, 4, 4, 256],  # OK
    [4, 513, 4, 4, 256],  # trash
    [2, 513, 2, 4, 256],  # OK
    [2, 513, 4, 2, 256],  # not great loss but somehow amazing node matching? Really confusing
    [2, 513, 4, 4, 512],  # fasted by far to converge, though still hard, and not great nodewise
    [2, 513, 6, 4, 256],  # OK
    [2, 513, 4, 6, 256],  # OK
]

# correlates from test 2 and 3
# num decoder points is huge
# more GC's is bad
# but more GC nodewise is good
# more decoder layers is good
# very deep embedding causes hideously slow training & non-convergence in general
# either need 'deep but not too deep' embedding, or some way to make it behave

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

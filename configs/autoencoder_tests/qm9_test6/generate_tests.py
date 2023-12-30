from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
- trying optimization & regularization changes on fixed architecture
"""

configs = [
    ['adamw', 0.001, 0.0001, 0.9, 0.999, 0.01, 'max', 'leaky relu', 3],
    ['adamw', 0.001, 0.0001, 0.9, 0.999, 0.1, 'max', 'leaky relu', 3],
    ['adamw', 0.001, 0.0001, 0.9, 0.999, 0.5, 'max', 'leaky relu', 3],
]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['autoencoder']['optimizer']['optimizer'] = configs[ii][0]
    config['autoencoder']['optimizer']['decoder_init_lr'] = configs[ii][1]
    config['autoencoder']['optimizer']['encoder_init_lr'] = configs[ii][2]
    config['autoencoder']['optimizer']['beta1'] = configs[ii][3]
    config['autoencoder']['optimizer']['beta2'] = configs[ii][4]
    config['autoencoder']['optimizer']['weight_decay'] = configs[ii][5]
    config['autoencoder']['model']['graph_aggregator'] = configs[ii][6]
    config['autoencoder']['model']['activation'] = configs[ii][7]
    config['autoencoder']['model']['num_attention_heads'] = configs[ii][8]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

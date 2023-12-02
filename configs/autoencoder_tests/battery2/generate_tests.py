from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')


"""
goal of this battery is to try to converge some random point clouds
"""

depths_list = [2, 4]
widths_list = [512, 1024]

ind = 0
for i1 in range(len(depths_list)):
    for i2 in range(len(widths_list)):
        config = copy(base_config)
        config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
        config['autoencoder']['model']['embedding_depth'] = widths_list[i2]
        config['autoencoder']['model']['num_decoder_layers'] = depths_list[i1]
        config['autoencoder']['model']['num_graph_convolutions'] = depths_list[i1]
        config['autoencoder']['model']['nodewise_fc_layers'] = depths_list[i1]

        config['dataset']['filter_conditions'] = \
            [['crystal_z_prime', 'in', [1]],
             ['molecule_num_atoms', 'range', [3, 15]],
             ['molecule_radius', 'range', [0.1, 5]],
             ['atom_atomic_numbers', 'range', [6, 9]]
             ]
        with open(str(ind) + '.yaml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        ind += 1


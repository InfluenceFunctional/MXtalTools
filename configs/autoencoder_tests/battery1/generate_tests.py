from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')


"""
goal of this battery is to test training on successively random point clouds with increasing number of points
"""
np.random.seed(12345)
seeds_list = np.arange(1, 1000)

num_atoms_list = [5, 7, 10, 12]

for ind in range(4):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['dataset']['filter_conditions'] = [['crystal_z_prime', 'in', [1]],
                                              ['molecule_num_atoms', 'range', [3, num_atoms_list[ind]]],
                                              ['molecule_radius', 'range', [0.1, 5]],
                                              ['atom_atomic_numbers', 'range', [6, 8]]
                                              ]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

from copy import copy

import yaml

from mxtaltools.common.config_processing import load_yaml

base_config = load_yaml('base.yaml')

num_configs = 1000

"""
sequentially numbered sweep configs
"""
for ix1 in range(num_configs):
    config = copy(base_config)
    config['run_name'] = f"run_{ix1}"
    config['opt_seed'] = ix1
    config['mol_seed'] = ix1

    with open(str(ix1) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
import os
from itertools import product

import numpy as np
import yaml
from pathlib import Path
from copy import deepcopy


def load_yaml(path):
    path = Path(path)
    with path.open('r') as f:
        return yaml.safe_load(f), path.parent  # return both content and its directory


def overwrite_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


if __name__ == "__main__":
    base_path = 'mipcas.yaml'
    base, spec_dir = load_yaml(base_path)

    ind = 0
    for enfunc in ['elj', 'uma']:
        for seed_ind in np.arange(10):

            config = deepcopy(base)
            config['opt_seed'] = seed_ind
            for opt in config['opt']:
                opt['optim_target'] = enfunc
            config['run_name'] = f'mipcas_{enfunc}_{seed_ind}'

            config_path = f'mipcas_{enfunc}_{ind}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            ind += 1

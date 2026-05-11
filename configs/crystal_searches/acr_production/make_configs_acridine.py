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
    base_path = 'acridine.yaml'
    base, spec_dir = load_yaml(base_path)

    n_samples = 40000
    n_parallel = 16
    bsz = 1000
    zps = []
    sgs = []
    zps.append(1) # Form II & XII
    sgs.append(14)
    zps.append(2) # Form III & VII
    sgs.append(14)
    zps.append(2) # Form VI
    sgs.append(9)
    zps.append(3) # Form IV
    sgs.append(19)

    # zps.append(2) # Form V
    # sgs.append(14)


    for zp, sg in zip(zps, sgs):
        ind = 0
        for seed_ind in np.arange(ind, ind + n_parallel):
            seed_ind = int(seed_ind)
            config = deepcopy(base)
            config['opt_seed'] = seed_ind
            config['sgs_to_search'] = [sg]
            config['zp_to_search'] = [zp]
            config['run_name'] = f'may_acridine_sg{sg}_zp{zp}_{seed_ind}'
            config['batch_size'] = bsz
            config['num_samples'] = int(n_samples/n_parallel)

            config_path = f'{sg}_{zp}_{ind}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            ind += 1

import os
from itertools import product

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
    base_path = 'coumarin.yaml'
    base, spec_dir = load_yaml(base_path)

    ind = 0
    for sg in [4, 14, 29, 19]:  # Pca21, P21, P212121, P21/c
        if sg == 19:
            for zp in [1, 2, 3]:
                config = deepcopy(base)
                config['sgs_to_search'] = [sg]
                config['zp_to_search'] = [zp]
                config['run_name'] = f'coumarin_sg{sg}_zp{zp}'

                config_path = f'coumarin_{ind}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                ind += 1


        else:
            for zp in [1]:
                config = deepcopy(base)
                config['sgs_to_search'] = [sg]
                config['zp_to_search'] = [zp]
                config['run_name'] = f'coumarin_sg{sg}_zp{zp}'

                config_path = f'coumarin_{ind}.yaml'
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                ind += 1

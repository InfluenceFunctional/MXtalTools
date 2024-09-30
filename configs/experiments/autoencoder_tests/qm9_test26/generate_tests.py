from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

config_list = [
    {
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False
                }
    },  # 0 - baseline
    {
        'dataset': {'filter_protons': False},
        'autoencoder': {
            'infer_protons': False
        }
    },  # 0 - baseline w protons
    {
        'dataset': {'filter_protons': False},
        'autoencoder': {
            'infer_protons': True
        }
    },  # 0 - baseline inferring protons
]


def overwrite_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            assert k in d1.keys()
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


ind = 0
for ix1 in range(len(config_list)):
    config = copy(base_config)
    config['machine'] = 'cluster'
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    run_config = config_list[ix1]
    overwrite_nested_dict(config, run_config)

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import os

base_config = load_yaml('base.yaml')

config_list = [
    {
        'min_batch_size': 25,
        'batch_growth_increment': 0.3,
        'max_batch_size': 500,
        'generator': {
            'samples_per_iter': 20,
            'step_size': 0.025,
            'optimizer': {
                'init_lr': 1e-5,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.2,
                'lr_shrink_lambda': 0.97,
            },
            'model': {
                'hidden_dim': 256,
                'dropout': 0,
                'norm': None,
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 0 - baseline: trying to capture the lightning in a bottle from last year
    {
        'min_batch_size': 25,
        'batch_growth_increment': 0.3,
        'max_batch_size': 500,
        'generator': {
            'samples_per_iter': 20,
            'step_size': 0.025,
            'optimizer': {
                'init_lr': 1e-5,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.2,
                'lr_shrink_lambda': 0.97,
            },
            'model': {
                'hidden_dim': 256,
                'dropout': 0,
                'norm': 'layer',
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 1 - layernorm
    {
        'min_batch_size': 25,
        'batch_growth_increment': 0.5,
        'max_batch_size': 10000,
        'generator': {
            'samples_per_iter': 3,
            'step_size': 0.025,
            'optimizer': {
                'init_lr': 1e-5,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.2,
                'lr_shrink_lambda': 0.97,
            },
            'model': {
                'hidden_dim': 256,
                'dropout': 0,
                'norm': None,
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 2 - larger batch
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

    # automate tagging
    run_name = os.path.basename(os.getcwd())
    config['logger']['run_name'] = run_name
    config['logger']['experiment_tag'] = run_name

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import os

base_config = load_yaml('base.yaml')

config_list = [
    {
        'min_batch_size': 500,
        'max_batch_size': 10000,
        'generator': {
            'samples_per_iter': 5,
            'mean_step_size': 2,
            'init_vdw_loss_factor': 0.001,
            'optimizer': {
                'init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.1,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'hidden_dim': 1024,
                'dropout': 0,
                'norm': None,
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 0 - baseline
    {
        'min_batch_size': 500,
        'max_batch_size': 10000,
        'generator': {
            'samples_per_iter': 5,
            'mean_step_size': 2,
            'init_vdw_loss_factor': 0.001,
            'optimizer': {
                'init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.1,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'hidden_dim': 1024,
                'dropout': 0,
                'norm': 'layer',
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 1- baseline with norm
    {
        'min_batch_size': 500,
        'max_batch_size': 10000,
        'generator': {
            'samples_per_iter': 5,
            'mean_step_size': 2,
            'init_vdw_loss_factor': 0.001,
            'optimizer': {
                'init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.1,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'hidden_dim': 1024,
                'dropout': 0,
                'norm': None,
                'num_layers': 8,
                'vector_norm': None,
            }
        }
    },  # 2 - baseline double length
    {
        'min_batch_size': 500,
        'max_batch_size': 10000,
        'generator': {
            'samples_per_iter': 5,
            'mean_step_size': 2,
            'init_vdw_loss_factor': 0.001,
            'optimizer': {
                'init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.1,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'hidden_dim': 512,
                'dropout': 0,
                'norm': None,
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 3 - baseline half depth
    {
        'min_batch_size': 500,
        'max_batch_size': 10000,
        'generator': {
            'samples_per_iter': 5,
            'mean_step_size': 0.5,
            'init_vdw_loss_factor': 0.001,
            'optimizer': {
                'init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.1,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'hidden_dim': 1024,
                'dropout': 0,
                'norm': None,
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 4 - baseline small steps
    {
        'min_batch_size': 500,
        'max_batch_size': 10000,
        'generator': {
            'samples_per_iter': 5,
            'mean_step_size': 4,
            'init_vdw_loss_factor': 0.001,
            'optimizer': {
                'init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.1,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'hidden_dim': 1024,
                'dropout': 0,
                'norm': None,
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 5 - baseline big steps
    {
        'min_batch_size': 500,
        'max_batch_size': 10000,
        'generator': {
            'samples_per_iter': 5,
            'mean_step_size': 4,
            'init_vdw_loss_factor': 0.001,
            'optimizer': {
                'init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.25,
                'lr_shrink_lambda': 0.95,
            },
            'model': {
                'hidden_dim': 1024,
                'dropout': 0,
                'norm': None,
                'num_layers': 4,
                'vector_norm': None,
            }
        }
    },  # 6 - 5 with faster LR schedule
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

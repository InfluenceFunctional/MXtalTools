from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

SG_list = [1, 2, 3, 4, 5,
           6, 7, 8, 9, 10,
           11, 12, 13, 14, 15,
           16, 17, 18, 19, 20]
configs_list = [
    # 0 - baseline
    {'max_batch_size': 1000,
     'generate_sgs': ['P21/c'],
     'generator':
         {'model': {'hidden_dim': 2048,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 2,
                    },
          'optimizer': {'max_lr': 5e-4}
          }},
    # 1 - narrower
    {'max_batch_size': 1000,
     'generate_sgs': ['P21/c'],
     'generator':
         {'model': {'hidden_dim': 1024,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 4,
                    },
          'optimizer': {'max_lr': 5e-4}
          }},
    # 2 - single-step
    {'max_batch_size': 1000,
     'generate_sgs': ['P21/c'],
     'generator':
         {'model': {'hidden_dim': 1024,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 4,
                    },
          'optimizer': {'max_lr': 5e-4},
          'samples_per_iter': 1
          }},
    # 3 - double-step
    {'max_batch_size': 1000,
     'generate_sgs': ['P21/c'],
     'generator':
         {'model': {'hidden_dim': 1024,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 4,
                    },
          'optimizer': {'max_lr': 5e-4},
          'samples_per_iter': 2
          }},
]


def overwrite_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            assert k in d1.keys()
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


"""
sequentially numbered sweep configs
"""
for ix1 in range(len(configs_list)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ix1)

    run_config = configs_list[ix1]
    overwrite_nested_dict(config, run_config)

    with open(str(ix1) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

configs_list = [
    {'generate_sgs': ['P-1'],
     'generator':
         {'model': {'hidden_dim': 512,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 8,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1
          }},
    {'max_batch_size': 200,
     'generate_sgs': ['P-1'],
     'generator':
         {'model': {'hidden_dim': 512,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 8,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1
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

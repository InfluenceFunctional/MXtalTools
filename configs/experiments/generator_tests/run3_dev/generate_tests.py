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
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 256,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 4,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-3}
          }},
    # 1 - baseline with batch norm
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 256,
                    'dropout': 0,
                    'norm': 'batch',
                    'num_layers': 4,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-3}
          }},
    # 2 - baseline with layer norm
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 256,
                    'dropout': 0,
                    'norm': 'layer',
                    'num_layers': 4,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-3}
          }},
    # 3 - very deep baseline
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 256,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 20,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 4e-3}
          }},
    # 4 - baseline with higher max lr
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 256,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 4,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 4e-3}
          }},
    # 5 - baseline with dropout
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 256,
                    'dropout': 0.25,
                    'norm': None,
                    'num_layers': 4,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-3}
          }},
    # 6 - short and fat
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 1024,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 2,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-3}
          }},
    # 7 - just big
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 512,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 20,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-3}

          }},
    # 8 - big with norm  # stalled out
    {'max_batch_size': 100000,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 512,
                    'dropout': 0,
                    'norm': 'layer',
                    'num_layers': 20,
                    'vector_norm': 'vector layer'},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-3}
          }},
    # 9 - baseline with small batch
    {'max_batch_size': 500,
     'generate_sgs': SG_list,
     'generator':
         {'model': {'hidden_dim': 256,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 4,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 3,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-3}
          }},
    # 10 - fat with 1 SG, lower LR, big variation scale
    {'max_batch_size': 10000,
     'generate_sgs': ['P21/c'],
     'generator':
         {'model': {'hidden_dim': 1024,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 4,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 8,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-4}
          }},
    # 11 - 10, but even fatter
    {'max_batch_size': 10000,
     'generate_sgs': ['P21/c'],
     'generator':
         {'model': {'hidden_dim': 2048,
                    'dropout': 0,
                    'norm': None,
                    'num_layers': 4,
                    'vector_norm': None},
          'prior_loss_coefficient': 1,
          'prior_coefficient_threshold': 0.01,
          'variation_scale': 8,
          'vdw_loss_coefficient': 1,
          'optimizer': {'max_lr': 1e-4}
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

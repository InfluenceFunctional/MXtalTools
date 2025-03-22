from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy, deepcopy
import os

base_config = load_yaml('base.yaml')

"""
Questions need answers:
- scan over embeddings (ae, ellipsoid, principal moments)
- scan over problem types (LJ, LJ + some ES, LJ + heavy ES, MACE)

To-Do:
- get nice converging parameters
- run above tests


Notes:
lr too high
"""
# convergence tests
config_list = [
    {
        'dataset': {
            'max_dataset_length': 100000000
        },
        'positional_noise': {'autoencoder': 0.001},
        'proxy_discriminator': {
            'embedding_type': 'autoencoder',
            'electrostatic_scaling_factor': 0,
            'train_on_mace': False,
            'optimizer': {
                'init_lr': 5e-5,
                'max_lr': 1e-4,
                'min_lr': 5e-6,
                'weight_decay': 0.005,
                'lr_growth_lambda': 1.01,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'hidden_dim': 1024,
                'dropout': 0,
                'norm': None,
                'num_layers': 40,
            }}
    },  # 0 - AE, LJ, 1024x40
]
# training curve
best_config = deepcopy(config_list[0])
config_list = [best_config]
for hidden_dim in [512, 1024]:
    for layers in [16, 32]:
        for norm in [None, 'layer']:
            for dropout in [0, 0.25, 0.5]:
                config_i = deepcopy(best_config)
                config_i['proxy_discriminator']['model']['num_layers'] = layers
                config_i['proxy_discriminator']['model']['hidden_dim'] = hidden_dim
                config_i['proxy_discriminator']['model']['dropout'] = dropout
                config_i['proxy_discriminator']['model']['norm'] = norm
                config_list.append(config_i)

# production tests
best_config = deepcopy(config_list[12])
for embedding in ['mol_volume', None]:
    for esf in [0, 1000, 100000, 'mace']:
        config_i = deepcopy(best_config)
        config_i['proxy_discriminator']['embedding_type'] = embedding
        if esf == 'mace':
            config_i['proxy_discriminator']['electrostatic_scaling_factor'] = 0
            config_i['proxy_discriminator']['train_on_mace'] = True
        else:
            config_i['proxy_discriminator']['electrostatic_scaling_factor'] = esf
            config_i['proxy_discriminator']['train_on_mace'] = False

        config_list.append(config_i)


"""
Proxy discriminator params
"""


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

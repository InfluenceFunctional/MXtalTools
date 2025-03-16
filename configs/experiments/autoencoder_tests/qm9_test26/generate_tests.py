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
    },  # 0 - baseline - awful
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 2e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.99,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 128,
                        'embedding_dim': 256,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.05,
                        'cutoff': 6,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 4,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}
    },  # 1 - narrower model - awful
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 1},
    },  # 2 - 0 with noise
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': .5},
    },  # 3 - 0 with noise
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': .1},
    },  # 4 - 0 with noise
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 2e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.99,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 6,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 1024,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 5 - fast overfit attempt
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

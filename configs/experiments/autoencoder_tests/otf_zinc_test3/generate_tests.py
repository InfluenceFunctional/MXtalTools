from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

config_list = [
    {
        'dataset': {'otf_build_size': 1000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'nearest_node_threshold': 0.5,
            'affine_scale_factor': None,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 0.01,
            'clumping_loss_coefficient': 0.01,
            'nearest_component_loss_coefficient': 0.1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 5e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9985,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'mlp',
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 0 - baseline mlp
    {
        'dataset': {'otf_build_size': 1000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'nearest_node_threshold': 0.5,
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 0.01,
            'clumping_loss_coefficient': 0.01,
            'nearest_component_loss_coefficient': 0.1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 5e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9985,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'mlp',
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 1 - baseline mlp with noise
    {
        'dataset': {'otf_build_size': 1000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'nearest_node_threshold': 0.5,
            'affine_scale_factor': None,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 0.01,
            'clumping_loss_coefficient': 0.01,
            'nearest_component_loss_coefficient': 0.1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 5e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9985,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 64,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 2 - baseline gnn
    {
        'dataset': {'otf_build_size': 1000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'nearest_node_threshold': 0.5,
            'affine_scale_factor': None,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 0.01,
            'clumping_loss_coefficient': 0.01,
            'nearest_component_loss_coefficient': 0.1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 5e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9985,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 64,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'mlp',
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 3 - big mlp
    {
        'dataset': {'otf_build_size': 1000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'nearest_node_threshold': 0.5,
            'affine_scale_factor': None,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 0.01,
            'clumping_loss_coefficient': 0.01,
            'nearest_component_loss_coefficient': 1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 5e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9985,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'mlp',
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 4 - baseline mlp with hard component loss
    {
        'dataset': {'otf_build_size': 1000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'nearest_node_threshold': 0.5,
            'affine_scale_factor': None,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 0.01,
            'clumping_loss_coefficient': 0.01,
            'nearest_component_loss_coefficient': 1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 5e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9985,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'mlp',
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 256
                }}}
    },  # 4 - baseline mlp with hard component loss and lots of nodes

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

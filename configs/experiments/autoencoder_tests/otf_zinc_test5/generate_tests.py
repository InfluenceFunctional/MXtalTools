from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

config_list = [
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
                        'hidden_dim': 32,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 0 - baseline gnn with noise - about as good as baseline, only a bit less overfit
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
                        'num_convs': 4,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 32,
                        'num_layers': 8,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 1 - deep gnn with noise - crashed out very early
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
                        'dropout': 0.25,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 32,
                        'num_layers': 4,
                        'dropout': 0.25,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 2 - baseline gnn with noise and dropout - slow and steady convergence
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
                        'num_convs': 4,
                        'fcs_per_gc': 2,
                        'dropout': 0.25,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 32,
                        'num_layers': 8,
                        'dropout': 0.25,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 3 - deep gnn with noise and dropout - crashed out very early
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
                'bottleneck_dim': 64,
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
                        'hidden_dim': 32,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 4 - baseline gnn with noise and 64 embedding - pretty solid, slow and steady  RESTART
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
                'bottleneck_dim': 64,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 4,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 32,
                        'num_layers': 8,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 5 - deep gnn with noise and 64 embedding - levelled out very early
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
                'bottleneck_dim': 64,
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
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 32,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 6 - big encoder with noise and 64 embedding - levelled out midway, bad
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0.1},
        'autoencoder': {
            'affine_scale_factor': 1,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
                'bottleneck_dim': 64,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.25,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 32,
                        'num_layers': 4,
                        'dropout': 0.25,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 7 - baseline gnn with noise and dropout and 64 embedding - maybe levelled out or just slow RESTART

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

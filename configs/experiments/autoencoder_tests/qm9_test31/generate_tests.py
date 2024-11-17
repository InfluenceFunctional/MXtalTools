from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

config_list = [
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 1,
            'clumping_loss_coefficient': 1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 0 - new baseline
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 1,
            'clumping_loss_coefficient': 0.01,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 1 - weak clump
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 0.01,
            'clumping_loss_coefficient': 1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 2 - weak nearest node
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 1,
            'clumping_loss_coefficient': 1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 64,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 3 - 64 bottleneck
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 1,
            'clumping_loss_coefficient': 1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 64
                }}}
    },  # 4 - 64 num nodes
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 1,
            'clumping_loss_coefficient': 1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 64,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 64
                }}}
    },  # 5 - 65 bottleneck and nodes
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': 1,
            'clumping_loss_coefficient': 1,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}
    },  # 6 - 256 bottleneck and nodes
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': .0001,
            'clumping_loss_coefficient': .0001,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 7 - baseline with weak losses
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': .0001,
            'clumping_loss_coefficient': .0001,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 64,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 8 - 64 bottleneck weak
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': .0001,
            'clumping_loss_coefficient': .0001,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 64
                }}}
    },  # 9 - 64 num nodes weak
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': .0001,
            'clumping_loss_coefficient': .0001,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 64,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 64
                }}}
    },  # 10 - 65 bottleneck and nodes weak
    {
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'nearest_node_loss_coefficient': .0001,
            'clumping_loss_coefficient': .0001,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.9975,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 4,
                        'dropout': 0.5,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}
    },  # 11 - 256 bottleneck and nodes, weak
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

from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

config_list = [
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
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
    },  # 0 - baseline gnn - good though had a big NaN crash
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
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
                        'norm': 'graph layer',
                        'vector_norm': 'graph vector layer',
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 32,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': 'layer',
                        'vector_norm': 'vector layer',
                    },
                    'num_nodes': 64
                }}}
    },  # 1 - baseline gnn with norms - big overfit and bad performance
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
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
    },  # 2 - baseline gnn with big encoder - best model, big overfit - TO CONTINUE
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
    },  # 3 - baseline gnn with noise - less overfit, also excellent - comparable to #2 - TO CONTINUE
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
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
                        'hidden_dim': 64,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 4 - baseline gnn with big decoder - moderate overfit moderate performance
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
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
                        'hidden_dim': 64,
                        'num_layers': 4,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None,
                    },
                    'num_nodes': 64
                }}}
    },  # 5 - baseline gnn with big encoder and decoder - big overfit, good performance
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
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
                        'norm': 'graph layer',
                        'vector_norm': 'graph vector layer',
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 32,
                        'num_layers': 4,
                        'dropout': 0.25,
                        'norm': 'layer',
                        'vector_norm': 'vector layer',
                    },
                    'num_nodes': 64
                }}}
    },  # 6 - baseline gnn with norm and dropout - big overfit, good PM overlap but OK RMSD
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
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
                        'node_dim': 512,
                        'message_dim': 64,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.25,
                        'cutoff': 3,
                        'norm': 'graph layer',
                        'vector_norm': 'graph vector layer',
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 64,
                        'num_layers': 4,
                        'dropout': 0.25,
                        'norm': 'layer',
                        'vector_norm': 'vector layer',
                    },
                    'num_nodes': 64
                }}}
    },  # 7 - baseline gnn with big encoder and decoder and norm and dropout - decent overfit decent performance
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
            'filter_protons': False,
            'infer_protons': False,
            'sigma_threshold': 0.15,
            'init_sigma': 2,
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
    },  # 8 - baseline gnn with big component factor - decent overfit decent performance, best component loss but only slightly
    {
        'dataset': {'otf_build_size': 10000},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'affine_scale_factor': None,
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
    },  # 9 - deep gnn - low overfit pretty good performance
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
                        'node_dim': 512,
                        'message_dim': 64,
                        'embedding_dim': 512,
                        'num_convs': 4,
                        'fcs_per_gc': 2,
                        'dropout': 0.25,
                        'cutoff': 3,
                        'norm': 'graph layer',
                        'vector_norm': 'graph vector layer',
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 64,
                        'num_layers': 8,
                        'dropout': 0.25,
                        'norm': 'layer',
                        'vector_norm': 'vector layer',
                    },
                    'num_nodes': 64
                }}}
    },  # 10 - very large GNN with norming, dropout, and noise  - low overfit but kindof bad
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
                        'num_convs': 4,
                        'fcs_per_gc': 2,
                        'dropout': 0.25,
                        'cutoff': 3,
                        'norm': 'graph layer',
                        'vector_norm': 'graph vector layer',
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 64,
                        'num_layers': 8,
                        'dropout': 0.25,
                        'norm': 'layer',
                        'vector_norm': 'vector layer',
                    },
                    'num_nodes': 64
                }}}
    },  # 11 - very large GNN with norming, dropout, and noise, and 64 bottleneck - low overfit but kindof bad
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
                'bottleneck_dim': 128,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 64,
                        'embedding_dim': 512,
                        'num_convs': 4,
                        'fcs_per_gc': 2,
                        'dropout': 0.25,
                        'cutoff': 3,
                        'norm': 'graph layer',
                        'vector_norm': 'graph vector layer',
                    }},
                'decoder': {
                    'model_type': 'gnn',
                    'fc': {
                        'hidden_dim': 64,
                        'num_layers': 8,
                        'dropout': 0.25,
                        'norm': 'layer',
                        'vector_norm': 'vector layer',
                    },
                    'num_nodes': 64
                }}}
    },  # 11 - very large GNN with norming, dropout, and noise, and 128 bottleneck- low overfit but kindof bad
]

'''
upshots
norming - bad? weird
dropout - good
deep gnn - maybe good
big encoder - good
noise - good
small bottleneck - slower to train
'''

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

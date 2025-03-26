from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

config_list = [
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'dropout': 0,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}
    },  # 0 - baseline
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 1 - short & fat model
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 2 - baseline w single layer decoder
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'dropout': 0,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}
    },  # 3 - baseline with 512 outs
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 4 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None,
                        'vector_norm': None,
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None,
                        'vector_norm': None},
                    'num_nodes': 256
                }}}
    },  # 5 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 512,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 6 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 7 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 8,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 8 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 8,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 9 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 10 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 11 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}

    },  # 12 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': 'graph layer'
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 13 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 6,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 14 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 2,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 15 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0.25,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 16 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
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
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 14,
                        'dropout': 0.25,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 17 - small and bare-bones  # failed
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.99,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 18 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.975,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 19 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.1,
                'lr_shrink_lambda': 0.99,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 20 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0,
                'lr_growth_lambda': 1.1,
                'lr_shrink_lambda': 0.99,
            },
            'model': {
                'bottleneck_dim': 256,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 64,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 1,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 21 - small and bare-bones
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.99,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 256,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 8,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 512
                }}}
    },  # 22 - new large baseline
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-4,
                'min_lr': 1e-6,
                'weight_decay': 0,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.99,
            },
            'model': {
                'bottleneck_dim': 362,
                'encoder': {
                    'graph': {
                        'node_dim': 362,
                        'message_dim': 362//2,
                        'embedding_dim': 362,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 362,
                        'num_layers': 8,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 362
                }}}
    },  # 23 - slightly smaller big baseline
    {
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 1e-4,
                'decoder_init_lr': 1e-4,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.99,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 256,
                        'message_dim': 128,
                        'embedding_dim': 256,
                        'num_convs': 1,
                        'fcs_per_gc': 1,
                        'dropout': 0,
                        'cutoff': 3,
                        'norm': None
                    }},
                'decoder': {
                    'fc': {
                        'hidden_dim': 256,
                        'num_layers': 8,
                        'dropout': 0,
                        'norm': None},
                    'num_nodes': 256
                }}}
    },  # 24 - big bottleneck and deep decoder
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

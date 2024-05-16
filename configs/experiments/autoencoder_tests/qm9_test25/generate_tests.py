from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

config_list = [
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0.25},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 0: big and noisy  # better train than 18 but worse test
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 2},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0.25},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 1: big and noisy seed  # same as 0
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0.25},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 2: big and noisy protons
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 2},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0.25},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 3: big and noisy protons seed
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0.25},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': True,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 4: big and noisy infer  # horrible
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 2},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0.25},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': True,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 5: big and noisy infer seed  # horrible

    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0.25},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 1024
                }}}},  # 6: big and noisy 1024 output  # bit worse than 0 looks like?
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0.25},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 7,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 1024
                }}}},  # 7: big and noisy 1024 half cutoff  # also bit worse than 0, very similar to 6

    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 8: big no noise
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 9: big no noise protons  # NaN after 250 epochs
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': True,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 10: big no noise protons inferred  # converged

    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'batch_growth_increment': 0.25,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
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
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 11: 24-18 with slower lr decay
    {
        'min_batch_size': 10,
        'max_batch_size': 300,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 5e-4,
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
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 12: 24-13 redo  - crashed, rerunning
    {
        'min_batch_size': 10,
        'max_batch_size': 300,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 5e-4,
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
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 13: 24-13 redo with 512 outputs  - rather good

    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'batch_growth_increment': 0.25,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 5e-4},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 14: LITERALLY just 24-18
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'batch_growth_increment': 0.25,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 5e-4},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 15: LITERALLY just 24-18 with protons
    {
        'min_batch_size': 50,
        'max_batch_size': 10000,
        'batch_growth_increment': 0.25,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 5e-4},
            'infer_protons': True,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 16: LITERALLY just 24-18 protons inferred

    {
        'min_batch_size': 10,
        'max_batch_size': 300,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 5e-4,
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
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 17: 24-13 redo with 512 outputs with protons
    {
        'min_batch_size': 10,
        'max_batch_size': 300,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': False},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 1e-3},
            'infer_protons': True,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 5e-4,
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
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 18: 24-13 redo with 512 outputs protons inferred
    # batch size cap should be set at a hard limit - e.g., 2522 from 24_18
    # otherwise, on different machines, you end up with different max batch sizes
    # which significantly impacts training

    {
        'min_batch_size': 50,
        'max_batch_size': 2522,
        'batch_growth_increment': 0.25,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0},
        'autoencoder': {
            'overlap_eps': {'test': 5e-4},
            'infer_protons': False,
            'type_distance_scaling': 2,
            'optimizer': {
                'init_lr': 5e-5,
                'encoder_init_lr': 5e-5,
                'decoder_init_lr': 5e-5,
                'max_lr': 1e-3,
                'min_lr': 1e-6,
                'weight_decay': 0.05,
                'lr_growth_lambda': 1.05,
                'lr_shrink_lambda': 0.995,
            },
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 512
                }}}},  # 19: LITERALLY just 24-18 with fixed batch sizing

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

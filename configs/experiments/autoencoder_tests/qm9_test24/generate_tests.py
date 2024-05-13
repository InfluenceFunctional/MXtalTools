from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

config_list = [
    {
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 0: base config # crashed and restarted and crashed again
    {
        'seeds': {'model': 2},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 1: base config new seed
    {
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 2: 4 fcs 1 conv  # crashed
    {
        'seeds': {'model': 2},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 3: 4 fcs 1 conv new seed  # maybe marginally the best
    {
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.051,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 4: less dropout  # crashed OOM KILL
    {
        'seeds': {'model': 2},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 5: less dropout new seed  # maybe marginally the best?
    {
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 256,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 6: thick message
    {
        'seeds': {'model': 2},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 256,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 7: thick message new seed
    {
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.1,
                        'cutoff': 7,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 8: short cutoff
    {
        'seeds': {'model': 2},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.1,
                        'cutoff': 7,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 9: short cutoff new seed  # crashed
    # all above had issues with crashes, insufficient usage
    # also, pretty comparable performance

    # next stage
    # lower max batch OR large max batch with high LR
    # gaussian radials
    # 1/4:512/256 -> 1:512 -> 256
    # try also 5e-4 overlap eps
    # try some positional noise

    # also, we re-added the vector renormalization factor which
    # may significantly impact training, hopefully for the best

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
                }}}},  # 10: new baseline - faster convergence than prior baseline
    {
        'min_batch_size': 10,
        'max_batch_size': 2000,
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
                }}}},  # 11: large batch
    {
        'min_batch_size': 10,
        'max_batch_size': 2000,
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
                }}}},  # 12: large batch high max lr - smooth but a bit worse than 10
    {
        'min_batch_size': 10,
        'max_batch_size': 300,
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'positional_noise': {'autoencoder': 0.1},
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
                }}}},  # 13: base with noise - similar, maybe slightly better than 10
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
                        'num_convs': 2,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 2,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 14: big with noise - all weirdly killed, including OOM
    {
        'min_batch_size': 10,
        'max_batch_size': 300,
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
                }}}},  # 15 lower overlap eps - kindof normal performance
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
                        'radial_embedding': 'bessel',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 16 bessel radial - almost identical to 10

    # JOBS ON A100 are the ones being cancelled - low usage I think

    # ok let's just blast this thing as hard as possible - 10k batch higher lr
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
                    'num_nodes': 256
                }}}},  # 17: slower. Good train/tebut worse rmsds
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
                }}}},  # 18: 512 output nodes - New best/baseline
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
            'type_distance_scaling': 1,
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
                    'num_nodes': 256
                }}}},  # 19: CUDA assert after 703 epochs then OOM kill after 200 - was converging OK before cutoff mid-run
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
                        'hidden_dim': 364,
                        'num_layers': 2,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 20: converged at 2k steps and second run cancelled - slightly better than baseline
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
                'bottleneck_dim': 1024,
                'encoder': {
                    'graph': {
                        'node_dim': 1024,
                        'message_dim': 128,
                        'embedding_dim': 1024,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 1024,
                        'num_layers': 4,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 21: 1024 embedding - CUDA error then OOM kill, both early
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
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.05,
                        'cutoff': 14,
                        'radial_embedding': 'gaussian',
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 2,
                        'dropout': 0.05,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},  # 22: 2 gcs 2 decoder layers - converges at a very high loss

    # we need also some protonated and inferred proton runs
    # one run with higher noise
    # also, slower LR decay
    {
        'min_batch_size': 10,
        'max_batch_size': 300,
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
                }}}},  # 23: 13 with more noise - almost identical... maybe more noise?

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
                }}}},  # 24: 18 with protons - mean to do 13 but didn't
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
                }}}},  # 25: 18 with protons - almost identical to 24
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
                }}}},  # 26: 13 inferred protons - same issue as 23/24
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
                }}}},  # 27: 18 inferred protons


    '''
    - extend convergence - keep cutting off training at 2k steps
    - local test very large noise
    - these OOM issues are getting ridiculous - investigate
    - see further notes in test 25
    '''
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

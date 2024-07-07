from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')
configs_list = [
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.05,
             'lr_shrink_lambda': 0.9999,
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
                     'dropout': 0.25,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 0: Baseline - big model with dropout
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.05,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.05,
             'lr_shrink_lambda': 0.9999,
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
                     'dropout': 0.25,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 1: Loose sigma
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.1,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.05,
             'lr_shrink_lambda': 0.9999,
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
                     'dropout': 0.25,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 2: Really loose sigma
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.05,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.05,
             'lr_shrink_lambda': 0.9999,
         },
         'model': {
             'bottleneck_dim': 1024,
             'encoder': {
                 'graph': {
                     'node_dim': 1024,
                     'message_dim': 128,
                     'embedding_dim': 1024,
                     'num_convs': 2,
                     'fcs_per_gc': 2,
                     'dropout': 0.25,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 1024,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 3: Loose sigma, double depth
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
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
                     'dropout': 0.25,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 4: Baseline with more aggressive LR
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
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
                     'dropout': 0.5,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.5,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 5: Baseline with more aggressive LR + double dropout
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
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
                     'dropout': 0.25,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 256,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 6: Smaller model with more aggressive LR
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 5e-4,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
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
                     'dropout': 0.25,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 7: Baseline with more aggressive LR
    {'min_batch_size': 10,
     'max_batch_size': 1000,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 5e-4,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
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
                     'dropout': 0.5,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.5,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 8: Baseline with more aggressive LR + double dropout
    {'min_batch_size': 10,
     'max_batch_size': 100,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 5e-5,
             'decoder_init_lr': 5e-5,
             'max_lr': 5e-4,
             'min_lr': 1e-6,
             'weight_decay': 0.05,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
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
                     'dropout': 0.5,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.5,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 9: Baseline with more aggressive LR + double dropout, low max batch
    {'min_batch_size': 10,
     'max_batch_size': 132,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 4e-5,
             'decoder_init_lr': 2e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.003,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.99,
         },
         'model': {
             'bottleneck_dim': 512,
             'encoder': {
                 'graph': {
                     'node_dim': 1024,
                     'message_dim': 128,
                     'embedding_dim': 512,
                     'num_convs': 4,
                     'fcs_per_gc': 1,
                     'dropout': 0.25,
                     'cutoff': 5,
                     'radial_embedding': 'bessel',
                     'norm': None,
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 1024,
                     'num_layers': 3,
                     'dropout': 0.5,
                     'norm': None,
                     'vector_norm': None, },
                 'num_nodes': 1024,
                 'ramp_depth': True,
             }}}},  # 10: close copy of sweep 104
    {'min_batch_size': 10,
     'max_batch_size': 132,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 4e-5,
             'decoder_init_lr': 2e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.003,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.9975,
         },
         'model': {
             'bottleneck_dim': 512,
             'encoder': {
                 'graph': {
                     'node_dim': 512,
                     'message_dim': 128,
                     'embedding_dim': 512,
                     'num_convs': 4,
                     'fcs_per_gc': 1,
                     'dropout': 0.25,
                     'cutoff': 6,
                     'radial_embedding': 'bessel',
                     'norm': None,
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 4,
                     'dropout': 0.5,
                     'norm': None,
                     'vector_norm': None, },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 11: close copy of sweep 104 - lower LR lambda slightly smaller
    {'min_batch_size': 10,
     'max_batch_size': 132,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 4e-5,
             'decoder_init_lr': 2e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.003,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
         },
         'model': {
             'bottleneck_dim': 256,
             'encoder': {
                 'graph': {
                     'node_dim': 256,
                     'message_dim': 128,
                     'embedding_dim': 256,
                     'num_convs': 4,
                     'fcs_per_gc': 1,
                     'dropout': 0.25,
                     'cutoff': 6,
                     'radial_embedding': 'bessel',
                     'norm': None,
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 256,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': None,
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 12: close copy of sweep 104, half size, minor changes, lower LR lambda slightly smaller
    {'min_batch_size': 10,
     'max_batch_size': 100,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 4e-5,
             'decoder_init_lr': 2e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.003,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
         },
         'model': {
             'bottleneck_dim': 512,
             'encoder': {
                 'graph': {
                     'node_dim': 512,
                     'message_dim': 128,
                     'embedding_dim': 512,
                     'num_convs': 1,
                     'fcs_per_gc': 8,
                     'dropout': 0.25,
                     'cutoff': 6,
                     'radial_embedding': 'bessel',
                     'norm': 'graph layer',
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 512,
                     'num_layers': 8,
                     'dropout': 0.25,
                     'norm': 'layer',
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 13: Mike's model 8x512
    {'min_batch_size': 10,
         'max_batch_size': 100,
         'seeds': {'model': 1},
         'positional_noise': {'autoencoder': 0},
         'autoencoder': {
             'overlap_eps': {'test': 1e-3},
             'infer_protons': False,
             'filter_protons': True,
             'sigma_threshold': 0.01,
             'type_distance_scaling': 2,
             'optimizer': {
                 'init_lr': 5e-5,
                 'encoder_init_lr': 4e-5,
                 'decoder_init_lr': 2e-5,
                 'max_lr': 1e-3,
                 'min_lr': 1e-6,
                 'weight_decay': 0.003,
                 'lr_growth_lambda': 1.01,
                 'lr_shrink_lambda': 0.999,
             },
             'model': {
                 'bottleneck_dim': 512,
                 'encoder': {
                     'graph': {
                         'node_dim': 512,
                         'message_dim': 128,
                         'embedding_dim': 512,
                         'num_convs': 1,
                         'fcs_per_gc': 2,
                         'dropout': 0.25,
                         'cutoff': 6,
                         'radial_embedding': 'bessel',
                         'norm': 'graph layer',
                         'vector_norm': 'graph vector layer',
                     }},
                 'decoder': {
                     'fc': {
                         'hidden_dim': 512,
                         'num_layers': 2,
                         'dropout': 0.25,
                         'norm': 'layer',
                         'vector_norm': 'vector layer', },
                     'num_nodes': 512,
                     'ramp_depth': True,
                 }}}},  # 14: Mike's model 2x512
    {'min_batch_size': 10,
         'max_batch_size': 100,
         'seeds': {'model': 1},
         'positional_noise': {'autoencoder': 0},
         'autoencoder': {
             'overlap_eps': {'test': 1e-3},
             'infer_protons': False,
             'filter_protons': True,
             'sigma_threshold': 0.01,
             'type_distance_scaling': 2,
             'optimizer': {
                 'init_lr': 5e-5,
                 'encoder_init_lr': 4e-5,
                 'decoder_init_lr': 2e-5,
                 'max_lr': 1e-3,
                 'min_lr': 1e-6,
                 'weight_decay': 0.003,
                 'lr_growth_lambda': 1.01,
                 'lr_shrink_lambda': 0.999,
             },
             'model': {
                 'bottleneck_dim': 724,
                 'encoder': {
                     'graph': {
                         'node_dim': 724,
                         'message_dim': 128,
                         'embedding_dim': 724,
                         'num_convs': 1,
                         'fcs_per_gc': 8,
                         'dropout': 0.25,
                         'cutoff': 6,
                         'radial_embedding': 'bessel',
                         'norm': 'graph layer',
                         'vector_norm': 'graph vector layer',
                     }},
                 'decoder': {
                     'fc': {
                         'hidden_dim': 724,
                         'num_layers': 8,
                         'dropout': 0.25,
                         'norm': 'layer',
                         'vector_norm': 'vector layer', },
                     'num_nodes': 512,
                     'ramp_depth': True,
                 }}}},  # 15: Mike's model 8x724
    {'min_batch_size': 10,
         'max_batch_size': 100,
         'seeds': {'model': 1},
         'positional_noise': {'autoencoder': 0},
         'autoencoder': {
             'overlap_eps': {'test': 1e-3},
             'infer_protons': False,
             'filter_protons': True,
             'sigma_threshold': 0.01,
             'type_distance_scaling': 2,
             'optimizer': {
                 'init_lr': 5e-5,
                 'encoder_init_lr': 4e-5,
                 'decoder_init_lr': 2e-5,
                 'max_lr': 1e-3,
                 'min_lr': 1e-6,
                 'weight_decay': 0.003,
                 'lr_growth_lambda': 1.01,
                 'lr_shrink_lambda': 0.999,
             },
             'model': {
                 'bottleneck_dim': 512,
                 'encoder': {
                     'graph': {
                         'node_dim': 512,
                         'message_dim': 128,
                         'embedding_dim': 512,
                         'num_convs': 1,
                         'fcs_per_gc': 8,
                         'dropout': 0.25,
                         'cutoff': 6,
                         'radial_embedding': 'bessel',
                         'norm': 'graph layer',
                         'vector_norm': 'graph vector layer',
                     }},
                 'decoder': {
                     'fc': {
                         'hidden_dim': 512,
                         'num_layers': 8,
                         'dropout': 0.25,
                         'norm': 'layer',
                         'vector_norm': 'vector layer', },
                     'num_nodes': 1024,
                     'ramp_depth': True,
                 }}}},  # 16: Mike's model 8x512 -> 1024
    {'min_batch_size': 10,
     'max_batch_size': 132,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 4e-5,
             'decoder_init_lr': 2e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.003,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
         },
         'model': {
             'bottleneck_dim': 256,
             'encoder': {
                 'graph': {
                     'node_dim': 256,
                     'message_dim': 128,
                     'embedding_dim': 256,
                     'num_convs': 8,
                     'fcs_per_gc': 1,
                     'dropout': 0.25,
                     'cutoff': 6,
                     'radial_embedding': 'bessel',
                     'norm': None,
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 256,
                     'num_layers': 4,
                     'dropout': 0.25,
                     'norm': None,
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 17: 12 + more convs
    {'min_batch_size': 10,
     'max_batch_size': 132,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 4e-5,
             'decoder_init_lr': 2e-5,
             'max_lr': 1e-3,
             'min_lr': 1e-6,
             'weight_decay': 0.003,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.999,
         },
         'model': {
             'bottleneck_dim': 256,
             'encoder': {
                 'graph': {
                     'node_dim': 256,
                     'message_dim': 128,
                     'embedding_dim': 256,
                     'num_convs': 4,
                     'fcs_per_gc': 1,
                     'dropout': 0.25,
                     'cutoff': 6,
                     'radial_embedding': 'bessel',
                     'norm': None,
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 256,
                     'num_layers': 12,
                     'dropout': 0.25,
                     'norm': None,
                     'vector_norm': 'vector layer', },
                 'num_nodes': 512,
                 'ramp_depth': True,
             }}}},  # 18: 12 with huge decoder
    {'min_batch_size': 10,
     'max_batch_size': 132,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'test': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.01,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 0.00004205932928066357,
             'decoder_init_lr': 0.00001939383815016441,
             'max_lr': 0.0009520396832912248,
             'min_lr': 1e-6,
             'weight_decay': 0.0031908810182467765,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.99,
             'beta1': 0.9080372575922618,
             'beta2': 0.99810670687236,
         },
         'model': {
             'bottleneck_dim': 639,
             'encoder': {
                 'graph': {
                     'node_dim': 948,
                     'message_dim': 109,
                     'embedding_dim': 492,
                     'num_convs': 4,
                     'fcs_per_gc': 1,
                     'dropout': 0.2422100129086513,
                     'cutoff': 5.575466887418605,
                     'radial_embedding': 'bessel',
                     'norm': None,
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 998,
                     'num_layers': 3,
                     'dropout': 0.4824555940509227,
                     'norm': None,
                     'vector_norm': None, },
                 'num_nodes': 839,
                 'ramp_depth': True,
             }}}},  # 19: actual copy of sweep 104

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

from copy import copy
import numpy as np

dev = {'run_name': 'dev',
       'experiment_tag': ['dev'],
       'device': 'cuda',
       'seed': 1234,
       'training_iterations': int(1e8),
       'min_num_training_steps': 50000,
       'convergence_history': 1000,
       'convergence_eps': 1e-3,
       'self_overlap_eps': 1e-3,

       # loss settings
       'independent_node_weights': False,  # made node weight a learnable function - doesn't really train
       'node_weight_temperature': 100,
       'sigma': 0.25,  # larger increases overlaps
       'sigma_lambda': 0.99,
       'sigma_threshold': 0.01,
       'type_distance_scaling': 0.5,  # larger decreases overlaps
       'overlap_type': 'gaussian',
       'log_reconstruction': False,
       'do_training': True,
       'cart_dimension': 3,
       'train_reconstruction_loss': True,
       'train_constraining_loss': True,
       'train_nodewise_type_loss': False,
       'train_centroids_loss': False,
       'train_type_confidence_loss': False,
       'train_num_points_loss': False,
       'train_encoding_type_loss': False,

       # dataset & dataloading
       'batch_size_min': 2,
       'batch_size_max': 500,
       'batch_size_increment': 1,
       'max_num_points': 5,
       'min_num_points': 3,
       'points_spread': 1,  # max cloud radius
       'max_point_types': 3,

       # lrs
       'encoder_lr': 1e-4,
       'decoder_lr': 1e-3,
       'lr_lambda': 0.99,
       'lr_timescale': 500,

       # encoder
       'encoder_aggregator': 'max',
       'encoder_num_layers': 4,
       'encoder_num_nodewise_fcs': 4,
       'encoder_embedding_depth': 512,
       'encoder_graph_norm': None,
       'encoder_dropout': 0,

       # decoder
       'decoder_num_layers': 4,
       'decoder_embedding_depth': 512,
       'decoder_fc_norm': 'batch',
       'decoder_dropout': 0,
       'num_decoder_points': 256,

       # paths
       'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
       # 'run_directory': '/scratch/mk8347/csd_runs/',
       'save_directory': 'D:/crystals_extra/',
       # 'save_directory': '/scratch/mk8347/csd_runs/',
       'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
       }

configs = []

base_config = {'run_name': 'dev',
               'experiment_tag': ['battery_7'],
               'device': 'cuda',
               'seed': 1234,
               'training_iterations': int(1e8),
               'min_num_training_steps': 50000,
               'convergence_history': 1000,
               'convergence_eps': 1e-3,

               # loss settings
               'sigma': 0.25,  # larger increases overlaps
               'sigma_lambda': 0.9,
               'sigma_threshold': 0.025,
               'type_distance_scaling': 0.5,  # larger decreases overlaps
               'overlap_type': 'gaussian',
               'log_reconstruction': False,
               'do_training': True,
               'cart_dimension': 3,
               'train_reconstruction_loss': True,
               'train_constraining_loss': True,
               'train_nodewise_type_loss': False,
               'train_centroids_loss': False,
               'train_type_confidence_loss': False,
               'train_num_points_loss': False,
               'train_encoding_type_loss': False,

               # dataset & dataloading
               'batch_size_min': 10,
               'batch_size_max': 1000,
               'batch_size_increment': 1,
               'max_num_points': 20,
               'min_num_points': 2,
               'points_spread': 1,  # max cloud radius
               'max_point_types': 3,

               # lrs
               'encoder_lr': 1e-4,
               'decoder_lr': 1e-3,
               'lr_lambda': 0.985,
               'lr_timescale': 2000,

               # encoder
               'encoder_aggregator': 'max',
               'encoder_num_layers': 1,
               'encoder_embedding_depth': 1024,
               'encoder_num_nodewise_fcs': 2,
               'encoder_graph_norm': 'graph layer',
               'encoder_dropout': 0,

               # decoder
               'decoder_num_layers': 2,
               'decoder_embedding_depth': 1024,
               'decoder_fc_norm': 'layer',
               'decoder_dropout': 0,
               'num_decoder_points': 100,

               # paths
               # 'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
               'run_directory': '/scratch/mk8347/csd_runs/',
               # 'save_directory': 'D:/crystals_extra/',
               'save_directory': '/scratch/mk8347/csd_runs/',
               'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
               }

search_space = {
    'embedding_depth': [512, 1024],
    'num_decoder_points': [128, 256, 512],
    'encoder_num_layers': [1, 4],
    'decoder_num_layers': [4, 8, 12],
    'encoder_num_nodewise_fcs': [4, 8],
    'encoder_graph_norm': [None, 'graph layer'],
    'decoder_fc_norm': [None, 'layer', 'batch'],
    'encoder_lr': [1e-3, 1e-4],
    'decoder_lr': [1e-3],
}

np.random.seed(0)
num_tests = 1000
randints = np.stack(
    [np.concatenate(
        [np.random.randint(0, len(values), size=1) for values in search_space.values()]
    ) for _ in range(num_tests)]
)

for i in range(num_tests):
    new_config = copy(base_config)
    new_config['run_name'] = f"{new_config['experiment_tag'][0]}_{i}"

    for ind, (key, values) in enumerate(search_space.items()):
        if key == 'embedding_depth':
            new_config['decoder_embedding_depth'] = values[randints[i, ind]]
            new_config['encoder_embedding_depth'] = values[randints[i, ind]]
        else:
            new_config[key] = values[randints[i, ind]]

    configs.append(new_config)

aa = 1

# configs = []
# base_config = {'run_name': 'base',
#                'experiment_tag': ['block_1'],
#                'training_iterations': 1000000,
#                'min_num_training_steps': 5000,
#                'do_training': True,
#                'train_nodewise_type_loss': True,
#                'train_reconstruction_loss': True,
#                'train_type_confidence_loss': True,
#                'train_num_points_loss': True,
#                'train_encoding_type_loss': True,
#                'train_centroids_loss': False,
#                'batch_size_min': 2,
#                'batch_size_max': 1000,
#                'batch_size_increment': 1,
#                'mean_num_points': 10,
#                'num_points_spread': 1,  # more like a sigma
#                'max_num_points': 2,
#                'min_num_points': 2,
#                'points_spread': 1,
#                'point_types_max': 5,
#                'device': 'cuda',
#                'seed': 1234,
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-4,
#                'lr_lambda': 0.975,
#                'lr_timescale': 500,
#                'encoder_aggregator': 'combo',
#                'encoder_num_layers': 2,
#                'encoder_num_fc_layers': 2,
#                'encoder_embedding_depth': 128,
#                'encoder_num_nodewise_fcs': 1,
#                'encoder_fc_norm': None,
#                'encoder_graph_norm': None,
#                'encoder_message_norm': None,
#                'encoder_dropout': 0,
#                'decoder_num_layers': 2,
#                'decoder_embedding_depth': 128,
#                'decoder_num_nodewise_fcs': 1,
#                'decoder_graph_norm': None,
#                'decoder_message_norm': None,
#                'decoder_dropout': 0,
#                'sigma': 1,
#                'sigma_lambda': 0.95,
#                'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                #'run_directory': '/scratch/mk8347/csd_runs/',
#                'save_directory': 'D:/crystals_extra/',
#                #'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,
#                }
#
# search_space = {
#     'encoder_embedding_depth': [64, 128, 256, 512, 1024],
#     'decoder_embedding_depth': [64, 128, 256, 512, 1024],
#     'encoder_dropout': [0, 0.1, 0.25],
#     'decoder_dropout': [0, 0.1, 0.25],
#     'encoder_graph_norm': [None, 'graph layer'],
#     'decoder_graph_norm': [None, 'graph layer'],
#     'encoder_fc_norm': [None, 'layer'],
#     'encoder_num_layers': [1, 2, 3, 4],
#     'decoder_num_layers': [1, 2, 3, 4],
#     'encoder_num_fc_layers': [1, 2, 4, 8],
#     'encoder_num_nodewise_fcs': [1, 2, 4, 8],
#     'decoder_num_nodewise_fcs': [1, 2, 4, 8],
#     'encoder_lr': [1e-3, 1e-4, 1e-5],
#     'decoder_lr': [1e-3, 1e-4, 1e-5],
#     'encoder_aggregator': ['max', 'sum', 'combo'],
# }
#
#
# np.random.seed(0)
# num_tests = 1000
# randints = np.stack(
#     [np.concatenate(
#         [np.random.randint(0, len(values), size=1) for values in search_space.values()]
#     ) for _ in range(num_tests)]
# )
#
# for i in range(num_tests):
#     new_config = copy(base_config)
#     new_config['run_name'] = f'block_1_test_{i}'
#
#     for ind, (key, values) in enumerate(search_space.items()):
#         new_config[key] = values[randints[i, ind]]
#
#     configs.append(new_config)

#
# configs = []
# base_config = {'run_name': 'base',
#                'experiment_tag': ['block_2'],
#                'training_iterations': 1000000,
#                'min_num_training_steps': 5000,
#                'do_training': True,
#                'train_nodewise_type_loss': True,
#                'train_reconstruction_loss': True,
#                'train_type_confidence_loss': True,
#                'train_num_points_loss': True,
#                'train_encoding_type_loss': True,
#                'train_centroids_loss': True,
#                'batch_size_min': 2,
#                'batch_size_max': 1000,
#                'batch_size_increment': 1,
#                'mean_num_points': 10,
#                'num_points_spread': 1,  # more like a sigma
#                'max_num_points': 2,
#                'min_num_points': 2,
#                'points_spread': 1,
#                'point_types_max': 5,
#                'device': 'cuda',
#                'seed': 1234,
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-4,
#                'lr_lambda': 0.975,
#                'lr_timescale': 500,
#                'encoder_aggregator': 'combo',
#                'encoder_num_layers': 2,
#                'encoder_num_fc_layers': 2,
#                'encoder_embedding_depth': 128,
#                'encoder_num_nodewise_fcs': 1,
#                'encoder_fc_norm': None,
#                'encoder_graph_norm': None,
#                'encoder_message_norm': None,
#                'encoder_dropout': 0,
#                'decoder_num_layers': 2,
#                'decoder_embedding_depth': 128,
#                'decoder_num_nodewise_fcs': 1,
#                'decoder_graph_norm': None,
#                'decoder_message_norm': None,
#                'decoder_dropout': 0,
#                'sigma': 0.25,
#                'overlap_type': 'gaussian',
#                'sigma_lambda': 0.95,
#                #'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                'run_directory': '/scratch/mk8347/csd_runs/',
#                #'save_directory': 'D:/crystals_extra/',
#                'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,
#                }
#
# search_space = {
#     'encoder_embedding_depth': [64, 128, 256, 512, 1024],
#     'decoder_embedding_depth': [64, 128, 256, 512, 1024],
#     'encoder_dropout': [0, 0.1],
#     'decoder_dropout': [0, 0.1],
#     'encoder_graph_norm': [None, 'graph layer'],
#     'decoder_graph_norm': [None, 'graph layer'],
#     'encoder_fc_norm': [None, 'layer'],
#     'encoder_num_layers': [1, 2, 3, 4],
#     'decoder_num_layers': [1, 2, 3, 4],
#     'encoder_num_fc_layers': [1, 2, 4, 8],
#     'encoder_num_nodewise_fcs': [1, 2, 4, 8],
#     'decoder_num_nodewise_fcs': [1, 2, 4, 8],
#     'encoder_lr': [1e-3, 1e-4, 1e-5],
#     'decoder_lr': [1e-3, 1e-4, 1e-5],
#     'encoder_aggregator': ['max', 'sum', 'combo'],
#     'sigma': [0.1, 0.5, 1],
#     'overlap_type': ['gaussian', 'inverse'],
# }

#
#
# configs = []
# base_config = {'run_name': 'base',
#                'experiment_tag': ['block_3'],
#                'training_iterations': 1000000,
#                'min_num_training_steps': 5000,
#                'do_training': True,
#                'train_nodewise_type_loss': True,
#                'train_reconstruction_loss': False,
#                'train_type_confidence_loss': True,
#                'train_num_points_loss': False,
#                'train_encoding_type_loss': False,
#                'train_centroids_loss': False,
#                'batch_size_min': 2,
#                'batch_size_max': 1000,
#                'batch_size_increment': 1,
#                'mean_num_points': 10,
#                'num_points_spread': 1,  # more like a sigma
#                'max_num_points': 2,
#                'min_num_points': 2,
#                'points_spread': 1,
#                'point_types_max': 5,
#                'device': 'cuda',
#                'seed': 1234,
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-4,
#                'lr_lambda': 0.975,
#                'lr_timescale': 500,
#                'encoder_aggregator': 'combo',
#                'encoder_num_layers': 2,
#                'encoder_num_fc_layers': 2,
#                'encoder_embedding_depth': 128,
#                'encoder_num_nodewise_fcs': 1,
#                'encoder_fc_norm': None,
#                'encoder_graph_norm': None,
#                'encoder_message_norm': None,
#                'encoder_dropout': 0,
#                'decoder_num_layers': 2,
#                'decoder_embedding_depth': 128,
#                'decoder_num_nodewise_fcs': 1,
#                'decoder_graph_norm': None,
#                'decoder_message_norm': None,
#                'decoder_dropout': 0,
#                'sigma': 0.25,
#                'overlap_type': 'gaussian',
#                'sigma_lambda': 0.95,
#                'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                #'run_directory': '/scratch/mk8347/csd_runs/',
#                'save_directory': 'D:/crystals_extra/',
#                #'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,
#                }
#
# search_space = {
#     'encoder_embedding_depth': [128, 256, 512],
#     'decoder_embedding_depth': [128, 256, 512],
#     'encoder_dropout': [0, 0.1],
#     'decoder_dropout': [0, 0.1],
#     'encoder_graph_norm': [None, 'graph layer'],
#     'decoder_graph_norm': [None, 'graph layer'],
#     'encoder_fc_norm': [None, 'layer'],
#     'encoder_num_layers': [1, 2, 4],
#     'decoder_num_layers': [1, 2, 4],
#     'encoder_num_fc_layers': [1, 2, 8],
#     'encoder_num_nodewise_fcs': [1, 4],
#     'decoder_num_nodewise_fcs': [1, 4],
#     'encoder_lr': [1e-3, 1e-4, 1e-5],
#     'decoder_lr': [1e-3, 1e-4, 1e-5],
#     'encoder_aggregator': ['max', 'combo'],
#     'sigma': [0.01],
#     'overlap_type': ['gaussian'],
# }


#
#
# base_config = {'run_name': 'base',
#                'experiment_tag': ['battery_1'],
#                'device': 'cuda',
#                'seed': 1234,
#                'training_iterations': int(1e8),
#                'min_num_training_steps': 10000,
#                'convergence_history': 50,
#                'convergence_eps': 1e-3,
#
#                # loss settings
#                'sigma': 0.1,  # larger increases overlaps
#                'sigma_lambda': 0.95,
#                'sigma_threshold': 0.1,
#                'type_distance_scaling': 0.1,  # larger decreases overlaps
#                'overlap_type': 'gaussian',
#                'log_reconstruction': False,
#                'do_training': True,
#                'cart_dimension': 3,
#                'train_nodewise_type_loss': True,
#                'train_reconstruction_loss': True,
#                'train_centroids_loss': True,
#                'train_type_confidence_loss': False,
#                'train_num_points_loss': False,
#                'train_encoding_type_loss': False,
#
#                # dataset & dataloading
#                'batch_size_min': 2,
#                'batch_size_max': 400,
#                'num_decoder_points': 100,
#                'batch_size_increment': 1,
#                'mean_num_points': 5,
#                'num_points_spread': 1,  # more like a sigma
#                'max_num_points': 10,
#                'min_num_points': 2,
#                'points_spread': 1,  # max value
#                'max_point_types': 5,
#
#                # lrs
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-4,
#                'lr_lambda': 0.975,
#                'lr_timescale': 500,
#
#                # encoder
#                'encoder_aggregator': 'combo',
#                'encoder_num_layers': 1,
#                'encoder_num_fc_layers': 4,
#                'encoder_embedding_depth': 128,
#                'encoder_num_nodewise_fcs': 4,
#                'encoder_fc_norm': None,
#                'encoder_graph_norm': None,
#                'encoder_dropout': 0,
#
#                # decoder
#                'decoder_num_layers': 1,
#                'decoder_embedding_depth': 32,
#                'decoder_fc_norm': None,
#                'decoder_dropout': 0,
#
#                # paths
#                #'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                'run_directory': '/scratch/mk8347/csd_runs/',
#                #'save_directory': 'D:/crystals_extra/',
#                'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
#                }
#
# search_space = {
#     'encoder_embedding_depth': [256, 512],
#     'decoder_embedding_depth': [16, 32, 64],
#     'num_decoder_points': [50, 100, 200],
#     'encoder_dropout': [0, 0.1],
#     'decoder_dropout': [0, 0.1],
#     'encoder_graph_norm': [None, 'graph layer'],
#     'decoder_graph_norm': [None, 'layer'],
#     'encoder_fc_norm': [None, 'layer'],
#     'encoder_num_layers': [1, 2],
#     'decoder_num_layers': [1, 2],
#     'encoder_num_fc_layers': [1, 2, 4],
#     'encoder_num_nodewise_fcs': [1, 2, 4],
#     'encoder_lr': [1e-4, 1e-5],
#     'decoder_lr': [1e-4, 1e-5],
#     'encoder_aggregator': ['max', 'combo'],
#     'overlap_type': ['gaussian'],
#     'train_typewise_node_loss': [True, False],
# }
#
# np.random.seed(0)
# num_tests = 1000
# randints = np.stack(
#     [np.concatenate(
#         [np.random.randint(0, len(values), size=1) for values in search_space.values()]
#     ) for _ in range(num_tests)]
# )
#
# for i in range(num_tests):
#     new_config = copy(base_config)
#     new_config['run_name'] = f"{new_config['experiment_tag'][0]}_{i}"
#
#     for ind, (key, values) in enumerate(search_space.items()):
#         new_config[key] = values[randints[i, ind]]
#
#     configs.append(new_config)

#
# base_config = {'run_name': 'base',
#                'experiment_tag': ['battery_2'],
#                'device': 'cuda',
#                'seed': 1234,
#                'training_iterations': int(1e8),
#                'min_num_training_steps': 10000,
#                'convergence_history': 50,
#                'convergence_eps': 1e-3,
#
#                # loss settings
#                'sigma': 0.1,  # larger increases overlaps
#                'sigma_lambda': 0.95,
#                'sigma_threshold': 0.1,
#                'type_distance_scaling': 0.25,  # larger decreases overlaps
#                'overlap_type': 'gaussian',
#                'log_reconstruction': False,
#                'do_training': True,
#                'cart_dimension': 3,
#                'train_nodewise_type_loss': True,
#                'train_reconstruction_loss': True,
#                'train_centroids_loss': True,
#                'train_type_confidence_loss': False,
#                'train_num_points_loss': False,
#                'train_encoding_type_loss': False,
#
#                # dataset & dataloading
#                'batch_size_min': 2,
#                'batch_size_max': 400,
#                'num_decoder_points': 100,
#                'batch_size_increment': 1,
#                'mean_num_points': 5,
#                'num_points_spread': 1,  # more like a sigma
#                'max_num_points': 10,
#                'min_num_points': 2,
#                'points_spread': 1,  # max value
#                'max_point_types': 5,
#
#                # lrs
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-4,
#                'lr_lambda': 0.975,
#                'lr_timescale': 500,
#
#                # encoder
#                'encoder_aggregator': 'combo',
#                'encoder_num_layers': 1,
#                'encoder_num_fc_layers': 4,
#                'encoder_embedding_depth': 128,
#                'encoder_num_nodewise_fcs': 4,
#                'encoder_fc_norm': None,
#                'encoder_graph_norm': None,
#                'encoder_dropout': 0,
#
#                # decoder
#                'decoder_num_layers': 1,
#                'decoder_embedding_depth': 32,
#                'decoder_fc_norm': None,
#                'decoder_dropout': 0,
#
#                # paths
#                #'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                'run_directory': '/scratch/mk8347/csd_runs/',
#                #'save_directory': 'D:/crystals_extra/',
#                'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
#                }
#
# search_space = {
#     'encoder_embedding_depth': [256, 512],
#     'decoder_embedding_depth': [16, 32, 64],
#     'num_decoder_points': [50, 100, 200],
#     'encoder_dropout': [0, 0.1],
#     'decoder_dropout': [0, 0.1],
#     'encoder_graph_norm': [None, 'graph layer'],
#     'decoder_graph_norm': [None, 'layer'],  #TODO FIX
#     'encoder_fc_norm': [None, 'layer'],
#     'encoder_num_layers': [1, 2],
#     'decoder_num_layers': [1, 2],
#     'encoder_num_fc_layers': [1, 2, 4],
#     'encoder_num_nodewise_fcs': [1, 2, 4],
#     'encoder_lr': [1e-4, 1e-5],
#     'decoder_lr': [1e-4, 1e-5],
#     'encoder_aggregator': ['max', 'combo'],
#     'overlap_type': ['gaussian'],
#     'train_typewise_node_loss': [True, False],
# }

#
#
# base_config = {'run_name': 'base',
#                'experiment_tag': ['battery_3'],
#                'device': 'cuda',
#                'seed': 1234,
#                'training_iterations': int(1e8),
#                'min_num_training_steps': 10000,
#                'convergence_history': 200,
#                'convergence_eps': 1e-3,
#
#                # loss settings
#                'sigma': 0.2,  # larger increases overlaps
#                'sigma_lambda': 0.95,
#                'sigma_threshold': 0.1,
#                'type_distance_scaling': 2,  # larger decreases overlaps
#                'overlap_type': 'gaussian',
#                'log_reconstruction': False,
#                'do_training': True,
#                'cart_dimension': 3,
#                'train_nodewise_type_loss': True,
#                'train_reconstruction_loss': True,
#                'train_centroids_loss': True,
#                'train_type_confidence_loss': False,
#                'train_num_points_loss': False,
#                'train_encoding_type_loss': False,
#
#                # dataset & dataloading
#                'batch_size_min': 2,
#                'batch_size_max': 200,
#                'num_decoder_points': 100,
#                'batch_size_increment': 1,
#                'mean_num_points': 5,
#                'num_points_spread': 1,  # more like a sigma
#                'max_num_points': 10,
#                'min_num_points': 2,
#                'points_spread': 1,  # max value
#                'max_point_types': 5,
#
#                # lrs
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-4,
#                'lr_lambda': 0.975,
#                'lr_timescale': 500,
#
#                # encoder
#                'encoder_aggregator': 'max',
#                'encoder_num_layers': 1,
#                'encoder_num_fc_layers': 4,
#                'encoder_embedding_depth': 256,
#                'encoder_num_nodewise_fcs': 4,
#                'encoder_fc_norm': 'layer',
#                'encoder_graph_norm': 'graph layer',
#                'encoder_dropout': 0,
#
#                # decoder
#                'decoder_num_layers': 1,
#                'decoder_embedding_depth': 128,
#                'decoder_fc_norm': 'layer',
#                'decoder_dropout': 0,
#
#                # paths
#                # 'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                'run_directory': '/scratch/mk8347/csd_runs/',
#                # 'save_directory': 'D:/crystals_extra/',
#                'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
#                }
#
# search_space = {
#     'encoder_embedding_depth': [256, 512, 1024],
#     'decoder_embedding_depth': [32, 64, 128],
#     'num_decoder_points': [64, 128, 256],
#     'encoder_dropout': [0.1, 0.25],
#     'decoder_dropout': [0.1, 0.25],
#     'encoder_num_layers': [1, 2, 4],
#     'decoder_num_layers': [1, 2, 4],
#     'encoder_num_fc_layers': [2, 4],
#     'encoder_num_nodewise_fcs': [2, 4],
#     'encoder_lr': [5e-4, 1e-4, 1e-5],
#     'decoder_lr': [5e-4, 1e-4, 1e-5],
# }
#
#
#
# base_config = {'run_name': 'base',
#                'experiment_tag': ['battery_4'],
#                'device': 'cuda',
#                'seed': 1234,
#                'training_iterations': int(1e8),
#                'min_num_training_steps': 10000,
#                'convergence_history': 200,
#                'convergence_eps': 1e-3,
#
#                # loss settings
#                'sigma': 0.2,  # larger increases overlaps
#                'sigma_lambda': 0.95,
#                'sigma_threshold': 0.1,
#                'type_distance_scaling': 2,  # larger decreases overlaps
#                'overlap_type': 'gaussian',
#                'log_reconstruction': False,
#                'do_training': True,
#                'cart_dimension': 3,
#                'train_nodewise_type_loss': False,
#                'train_reconstruction_loss': True,
#                'train_centroids_loss': True,
#                'train_type_confidence_loss': False,
#                'train_num_points_loss': False,
#                'train_encoding_type_loss': False,
#
#                # dataset & dataloading
#                'batch_size_min': 2,
#                'batch_size_max': 200,
#                'num_decoder_points': 100,
#                'batch_size_increment': 1,
#                'mean_num_points': 5,
#                'num_points_spread': 1,  # more like a sigma
#                'max_num_points': 10,
#                'min_num_points': 2,
#                'points_spread': 1,  # max value
#                'max_point_types': 5,
#
#                # lrs
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-4,
#                'lr_lambda': 0.975,
#                'lr_timescale': 500,
#
#                # encoder
#                'encoder_aggregator': 'max',
#                'encoder_num_layers': 1,
#                'encoder_num_fc_layers': 4,
#                'encoder_embedding_depth': 256,
#                'encoder_num_nodewise_fcs': 4,
#                'encoder_fc_norm': 'layer',
#                'encoder_graph_norm': 'graph layer',
#                'encoder_dropout': 0,
#
#                # decoder
#                'decoder_num_layers': 1,
#                'decoder_embedding_depth': 128,
#                'decoder_fc_norm': 'layer',
#                'decoder_dropout': 0,
#
#                # paths
#                # 'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                'run_directory': '/scratch/mk8347/csd_runs/',
#                # 'save_directory': 'D:/crystals_extra/',
#                'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
#                }
#
# search_space = {
#     'encoder_embedding_depth': [256, 512, 1024],
#     'decoder_embedding_depth': [32, 64, 128],
#     'num_decoder_points': [64, 128, 256],
#     'encoder_dropout': [0.1, 0.25],
#     'decoder_dropout': [0.1, 0.25],
#     'encoder_num_layers': [1, 2, 4],
#     'decoder_num_layers': [1, 2, 4],
#     'encoder_num_fc_layers': [2, 4],
#     'encoder_num_nodewise_fcs': [2, 4],
#     'encoder_lr': [5e-4, 1e-4, 1e-5],
#     'decoder_lr': [5e-4, 1e-4, 1e-5],
# }

#
#
# base_config = {'run_name': 'dev',
#                'experiment_tag': ['battery_5'],
#                'device': 'cuda',
#                'seed': 1234,
#                'training_iterations': int(1e8),
#                'min_num_training_steps': 50000,
#                'convergence_history': 1000,
#                'convergence_eps': 1e-3,
#
#                # loss settings
#                'sigma': 0.25,  # larger increases overlaps
#                'sigma_lambda': 0.9,
#                'sigma_threshold': 0.025,
#                'type_distance_scaling': 0.5,  # larger decreases overlaps
#                'overlap_type': 'gaussian',
#                'log_reconstruction': False,
#                'do_training': True,
#                'cart_dimension': 3,
#                'train_reconstruction_loss': True,
#                'train_constraining_loss': True,
#                'train_nodewise_type_loss': False,
#                'train_centroids_loss': False,
#                'train_type_confidence_loss': False,
#                'train_num_points_loss': False,
#                'train_encoding_type_loss': False,
#
#                # dataset & dataloading
#                'batch_size_min': 2,
#                'batch_size_max': 400,
#                'batch_size_increment': 1,
#                'max_num_points': 4,
#                'min_num_points': 2,
#                'points_spread': 1,  # max cloud radius
#                'max_point_types': 3,
#
#                # lrs
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-3,
#                'lr_lambda': 0.975,
#                'lr_timescale': 2000,
#
#                # encoder
#                'encoder_aggregator': 'max',
#                'encoder_num_layers': 1,
#                'encoder_embedding_depth': 1024,
#                'encoder_num_nodewise_fcs': 2,
#                'encoder_graph_norm': 'graph layer',
#                'encoder_dropout': 0,
#
#                # decoder
#                'decoder_num_layers': 2,
#                'decoder_embedding_depth': 1024,
#                'decoder_fc_norm': 'layer',
#                'decoder_dropout': 0,
#                'num_decoder_points': 100,
#
#                # paths
#                # 'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                'run_directory': '/scratch/mk8347/csd_runs/',
#                # 'save_directory': 'D:/crystals_extra/',
#                'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
#                }
#
# search_space = {
#     'embedding_depth': [256, 512, 1024],
#     'num_decoder_points': [64, 128, 256],
#     'encoder_num_layers': [1, 4],
#     'decoder_num_layers': [1, 4],
#     'encoder_num_nodewise_fcs': [2, 4],
#     'encoder_graph_norm': [None, 'graph layer'],
#     'decoder_fc_norm': [None, 'layer', 'batch'],
#     'encoder_lr': [1e-3, 1e-4, 1e-5],
#     'decoder_lr': [1e-3, 1e-4, 1e-5],
# }
#
#
#
# base_config = {'run_name': 'dev',
#                'experiment_tag': ['battery_6'],
#                'device': 'cuda',
#                'seed': 1234,
#                'training_iterations': int(1e8),
#                'min_num_training_steps': 50000,
#                'convergence_history': 1000,
#                'convergence_eps': 1e-3,
#
#                # loss settings
#                'sigma': 0.25,  # larger increases overlaps
#                'sigma_lambda': 0.9,
#                'sigma_threshold': 0.025,
#                'type_distance_scaling': 0.5,  # larger decreases overlaps
#                'overlap_type': 'gaussian',
#                'log_reconstruction': False,
#                'do_training': True,
#                'cart_dimension': 3,
#                'train_reconstruction_loss': True,
#                'train_constraining_loss': True,
#                'train_nodewise_type_loss': False,
#                'train_centroids_loss': False,
#                'train_type_confidence_loss': False,
#                'train_num_points_loss': False,
#                'train_encoding_type_loss': False,
#
#                # dataset & dataloading
#                'batch_size_min': 2,
#                'batch_size_max': 400,
#                'batch_size_increment': 1,
#                'max_num_points': 4,
#                'min_num_points': 2,
#                'points_spread': 1,  # max cloud radius
#                'max_point_types': 3,
#
#                # lrs
#                'encoder_lr': 1e-4,
#                'decoder_lr': 1e-3,
#                'lr_lambda': 0.985,
#                'lr_timescale': 2000,
#
#                # encoder
#                'encoder_aggregator': 'max',
#                'encoder_num_layers': 1,
#                'encoder_embedding_depth': 1024,
#                'encoder_num_nodewise_fcs': 2,
#                'encoder_graph_norm': 'graph layer',
#                'encoder_dropout': 0,
#
#                # decoder
#                'decoder_num_layers': 2,
#                'decoder_embedding_depth': 1024,
#                'decoder_fc_norm': 'layer',
#                'decoder_dropout': 0,
#                'num_decoder_points': 100,
#
#                # paths
#                # 'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
#                'run_directory': '/scratch/mk8347/csd_runs/',
#                # 'save_directory': 'D:/crystals_extra/',
#                'save_directory': '/scratch/mk8347/csd_runs/',
#                'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
#                }
# search_space = {
#     'embedding_depth': [512, 1024],
#     'num_decoder_points': [128, 256, 512],
#     'encoder_num_layers': [1, 4],
#     'decoder_num_layers': [4, 8, 12],
#     'encoder_num_nodewise_fcs': [4, 8],
#     'encoder_graph_norm': [None, 'graph layer'],
#     'decoder_fc_norm': [None, 'layer', 'batch'],
#     'encoder_lr': [1e-3, 1e-4],
#     'decoder_lr': [1e-3],
# }

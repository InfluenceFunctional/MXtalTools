from copy import copy
import numpy as np

dev = {'run_name': 'dev',
       'experiment_tag': ['dev'],
       'training_iterations': 1000000,
       'min_num_training_steps': 2000,
       'do_training': True,
       'train_nodewise_type_loss': True,
       'train_reconstruction_loss': True,
       'train_type_confidence_loss': True,
       'train_num_points_loss': True,
       'train_encoding_type_loss': True,
       'train_centroids_loss': True,
       'batch_size_min': 2,
       'batch_size_max': 1000,
       'batch_size_increment': 1,
       'mean_num_points': 10,
       'num_points_spread': 1,  # more like a sigma
       'max_num_points': 2,
       'min_num_points': 2,
       'points_spread': 1,
       'point_types_max': 5,
       'device': 'cuda',
       'seed': 12345,
       'encoder_lr': 1e-4,
       'decoder_lr': 1e-4,
       'lr_lambda': 0.975,
       'lr_timescale': 500,
       'encoder_aggregator': 'combo',
       'encoder_num_layers': 2,
       'encoder_num_fc_layers': 2,
       'encoder_embedding_depth': 512,
       'encoder_num_nodewise_fcs': 1,
       'encoder_fc_norm': None,
       'encoder_graph_norm': None,
       'encoder_message_norm': None,
       'encoder_dropout': 0,
       'decoder_num_layers': 2,
       'decoder_embedding_depth': 512,
       'decoder_num_nodewise_fcs': 1,
       'decoder_graph_norm': None,
       'decoder_message_norm': None,
       'decoder_dropout': 0,
       'sigma': 1,
       'overlap_type': 'inverse',
       'sigma_lambda': 0.95,
       'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
       # 'run_directory': '/scratch/mk8347/csd_runs/',
       'save_directory': 'D:/crystals_extra/',
       # 'save_directory': '/scratch/mk8347/csd_runs/',
       'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
       }

configs = []
base_config = {'run_name': 'base',
               'experiment_tag': ['block_2'],
               'training_iterations': 1000000,
               'min_num_training_steps': 5000,
               'do_training': True,
               'train_nodewise_type_loss': True,
               'train_reconstruction_loss': True,
               'train_type_confidence_loss': True,
               'train_num_points_loss': True,
               'train_encoding_type_loss': True,
               'train_centroids_loss': True,
               'batch_size_min': 2,
               'batch_size_max': 1000,
               'batch_size_increment': 1,
               'mean_num_points': 10,
               'num_points_spread': 1,  # more like a sigma
               'max_num_points': 2,
               'min_num_points': 2,
               'points_spread': 1,
               'point_types_max': 5,
               'device': 'cuda',
               'seed': 1234,
               'encoder_lr': 1e-4,
               'decoder_lr': 1e-4,
               'lr_lambda': 0.975,
               'lr_timescale': 500,
               'encoder_aggregator': 'combo',
               'encoder_num_layers': 2,
               'encoder_num_fc_layers': 2,
               'encoder_embedding_depth': 128,
               'encoder_num_nodewise_fcs': 1,
               'encoder_fc_norm': None,
               'encoder_graph_norm': None,
               'encoder_message_norm': None,
               'encoder_dropout': 0,
               'decoder_num_layers': 2,
               'decoder_embedding_depth': 128,
               'decoder_num_nodewise_fcs': 1,
               'decoder_graph_norm': None,
               'decoder_message_norm': None,
               'decoder_dropout': 0,
               'sigma': 0.25,
               'overlap_type': 'gaussian',
               'sigma_lambda': 0.95,
               #'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
               'run_directory': '/scratch/mk8347/csd_runs/',
               #'save_directory': 'D:/crystals_extra/',
               'save_directory': '/scratch/mk8347/csd_runs/',
               'checkpoint_path': None,
               }

search_space = {
    'encoder_embedding_depth': [64, 128, 256, 512, 1024],
    'decoder_embedding_depth': [64, 128, 256, 512, 1024],
    'encoder_dropout': [0, 0.1],
    'decoder_dropout': [0, 0.1],
    'encoder_graph_norm': [None, 'graph layer'],
    'decoder_graph_norm': [None, 'graph layer'],
    'encoder_fc_norm': [None, 'layer'],
    'encoder_num_layers': [1, 2, 3, 4],
    'decoder_num_layers': [1, 2, 3, 4],
    'encoder_num_fc_layers': [1, 2, 4, 8],
    'encoder_num_nodewise_fcs': [1, 2, 4, 8],
    'decoder_num_nodewise_fcs': [1, 2, 4, 8],
    'encoder_lr': [1e-3, 1e-4, 1e-5],
    'decoder_lr': [1e-3, 1e-4, 1e-5],
    'encoder_aggregator': ['max', 'sum', 'combo'],
    'sigma': [0.1, 0.5, 1],
    'overlap_type': ['gaussian', 'inverse'],
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
        new_config[key] = values[randints[i, ind]]

    configs.append(new_config)

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
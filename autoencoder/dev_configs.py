from copy import copy

dev = {'run_name': 'dev',
       'training_iterations': 1000000,
       'do_training': True,
       'batch_size_min': 10,
       'batch_size_max': 5000,
       'batch_size_increment': 5,
       'mean_num_points': 10,
       'num_points_spread': 2,
       'points_spread': 1,
       'point_types_max': 2,
       'device': 'cuda',
       'seed': 12345,
       'learning_rate': 1e-4,
       'lr_lambda': 0.975,
       'lr_timescale': 500,
       'encoder_num_layers': 2,
       'encoder_embedding_depth': 256,
       'encoder_num_nodewise_fcs': 1,
       'encoder_fc_norm': None,
       'encoder_graph_norm': None,
       'encoder_message_norm': None,
       'encoder_dropout': 0,
       'decoder_num_layers': 2,
       'decoder_embedding_depth': 256,
       'decoder_num_nodewise_fcs': 1,
       'decoder_graph_norm': None,
       'decoder_message_norm': None,
       'decoder_dropout': 0,
       'sigma': 0.05,
       'sigma_lambda': 0.95,
       'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
       # 'run_directory': '/scratch/mk8347/csd_runs/',
       'save_directory': 'D:/crystals_extra/',
       # 'save_directory': '/scratch/mk8347/csd_runs/',
       'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
       }

configs = []
base_config = {'run_name': 'base',
               'training_iterations': 1000000,
               'do_training': True,
               'batch_size_min': 10,
               'batch_size_max': 5000,
               'batch_size_increment': 5,
               'mean_num_points': 10,
               'num_points_spread': 2,
               'points_spread': 1,
               'point_types_max': 2,
               'device': 'cuda',
               'seed': 12345,
               'learning_rate': 1e-4,
               'lr_lambda': 0.975,
               'lr_timescale': 500,
               'encoder_num_layers': 2,
               'encoder_embedding_depth': 256,
               'encoder_num_nodewise_fcs': 1,
               'encoder_fc_norm': None,
               'encoder_graph_norm': None,
               'encoder_message_norm': None,
               'encoder_dropout': 0,
               'decoder_num_layers': 2,
               'decoder_embedding_depth': 256,
               'decoder_num_nodewise_fcs': 1,
               'decoder_graph_norm': None,
               'decoder_message_norm': None,
               'decoder_dropout': 0,
               'sigma': 0.05,
               'sigma_lambda': 0.95,
               #'run_directory': r'C:\Users\mikem\crystals\CSP_runs',
               'run_directory': '/scratch/mk8347/csd_runs/',
               #'save_directory': 'D:/crystals_extra/',
               'save_directory': '/scratch/mk8347/csd_runs/',
               'checkpoint_path': None,  # r'C:\Users\mikem\crystals\CSP_runs\models\cluster/test1_autoencoder_ckpt_11_12',
               }

ind = 1
for e_embed in [128, 512]:
    for e_convs in [1, 2]:
        for d_embed in [128, 512]:
            for d_convs in [1, 2]:
                configs.append(copy(base_config))
                configs[-1]['run_name'] = f'test{ind}'
                configs[-1]['encoder_embedding_depth'] = e_embed
                configs[-1]['decoder_embedding_depth'] = d_embed
                configs[-1]['encoder_num_layers'] = e_convs
                configs[-1]['decoder_num_layers'] = d_convs

                ind += 1

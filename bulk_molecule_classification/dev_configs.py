from copy import copy

defect_clusters_6_pure_nic_runs = [0, 1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 150, 151,
                                   152, 153, 154, 155, 156, 157, 158, 159, 2, 200, 201, 202, 203, 204,
                                   205, 206, 207, 208, 209, 250, 251, 252, 253, 254, 255, 256, 257, 258,
                                   259, 3, 4, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 7, 8, 9]
defect_clusters_5_rerun_pure_nic_runs = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110,
                                         111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 13, 14, 15, 16, 17,
                                         18, 19, 2, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                                         210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 3, 4, 5, 6, 7, 8, 9]

dev = {'run_name': 'dev_nic',
       'convergence_history': 50,
       'num_convs': 1,
       'embedding_depth': 256,
       'message_depth': 128,
       'dropout': 0.25,
       'graph_norm': 'graph layer',
       'fc_norm': 'layer',
       'num_fcs': 2,
       'num_epochs': 1000,
       'dataset_size': 500,
       'conv_cutoff': 6,
       'batch_size': 5,
       'reporting_frequency': 1,
       'train_model': True,
       'trajs_to_analyze_list': None,  # [f'D:/crystals_extra/classifier_training/crystal_in_melt_test7/{ind}/' for ind in range(54)],
       # [f'D:/crystals_extra/classifier_training/paper_nic_clusters2/{ind}/' for ind in range(12)],
       # [f'D:/crystals_extra/defect_clusters_6/{num}/' for num in defect_clusters_6_pure_nic_runs] +
       # [f'D:/crystals_extra/defect_clusters_5_rerun/{num}/' for num in defect_clusters_5_rerun_pure_nic_runs],
       'do_classifier_evaluation': False,
       'classifier_path': None,  # 'C:/Users/mikem/crystals/classifier_runs/dev_nic_1conv_best_classifier_checkpoint',
       'learning_rate': 1e-4,
       'datasets_path': r'D:/crystals_extra/classifier_training/traj_pickles/',
       'dumps_path': r'D:/crystals_extra/classifier_training/',
       'dumps_dirs': ['new_small_nic_liq_T350', 'nicotinamide_liq', 'bulk_trajs3', 'new_small_bulk'],  # ['new_small_bulk'], #
       'training_temps': [100, 350],
       'dataset_name': 'new_nic_full',  # 'nicotinamide_trajectories_dataset_100_350_vf', # 'small_nic_test', #
       'runs_path': r'C:/Users/mikem/crystals/classifier_runs/',
       'results_path': r'D:\crystals_extra\classifier_training\results/',
       'device': 'cuda',
       'seed': 1}
#
# dev = {'run_name': 'dev_urea',
#        'convergence_history': 50,
#        'num_convs': 1,
#        'embedding_depth': 256,
#        'message_depth': 128,
#        'dropout': 0.25,
#        'graph_norm': 'graph layer',
#        'fc_norm': 'layer',
#        'num_fcs': 2,
#        'num_epochs': 1000,
#        'dataset_size': 200,
#        'conv_cutoff': 6,
#        'batch_size': 5,
#        'reporting_frequency': 1,
#        'train_model': True,
#        'trajs_to_analyze_list': None, #['D:/crystals_extra/classifier_training/urea_melt_interface_T200'],
#        'do_classifier_evaluation': True,
#        'classifier_path': None, #'C:/Users/mikem/crystals/classifier_runs/dev_urea2_best_classifier_checkpoint',
#        'learning_rate': 1e-4,
#        'datasets_path': r'D:/crystals_extra/classifier_training/traj_pickles/',
#        'dumps_path': r'D:/crystals_extra/classifier_training/',
#        'dumps_dirs': ['new_small_urea_liq_T350', 'daisuke_small_ureas/T100', 'daisuke_small_ureas/T200', 'urea_liq_T350', 'urea_bulk_trajs/T100', 'urea_bulk_trajs/T200'], #['urea_liq_T350', 'urea_bulk_trajs/T100', 'urea_bulk_trajs/T200'],
#        'training_temps': [100, 200, 350],
#        'dataset_name': 'new_urea_full', #'urea_trajectories_dataset_100_200_350',
#        'runs_path': r'C:/Users/mikem/crystals/classifier_runs/',
#        'results_path': r'D:\crystals_extra\classifier_training\results/',
#        'device': 'cuda',
#        'seed': 1}

configs = []
base_config = {'run_name': 'dev',
               'convergence_history': 50,
               'num_convs': 1,
               'embedding_depth': 256,
               'message_depth': 128,
               'dropout': 0.25,
               'graph_norm': 'graph layer',
               'fc_norm': 'layer',
               'num_fcs': 2,
               'num_epochs': 1000,
               'dataset_size': 500,
               'conv_cutoff': 6,
               'batch_size': 5,
               'reporting_frequency': 1,
               'train_model': True,
               'trajs_to_analyze_list': None,
               'do_classifier_evaluation': False,
               'classifier_path': None,
               'learning_rate': 1e-4,
               'datasets_path': r'/vast/mk8347/molecule_clusters/traj_pickles/',
               'dumps_path': r'/vast/mk8347/molecule_clusters/',
               'training_temps': [100, 350],
               'dataset_name': 'new_nic_full',
               'dumps_dirs': None,  # ['melt_trajs2', 'melt_trajs2'],  # ['urea_bulk_trajs/T100', 'urea_bulk_trajs/T250', 'urea_bulk_trajs/liqT700'],
               'runs_path': r'/vast/mk8347/molecule_clusters/classifier_ckpts/',
               'results_path': r'/vast/mk8347/molecule_clusters/results/',
               'device': 'cuda',
               'seed': 1}

config_list = [
    [1, 256, 128],
    [2, 256, 128],
    [1, 64, 32],
    [2, 64, 32],
]

for i in range(len(config_list)):
    for si in range(4):
        for mol in range(0, 2):
            configs.append(copy(base_config))
            configs[-1]['seed'] = si
            configs[-1]['num_convs'] = config_list[i][0]
            configs[-1]['embedding_depth'] = config_list[i][1]
            configs[-1]['message_depth'] = config_list[i][2]

            if mol == 1:
                configs[-1]['run_name'] = f'urea test{i}'
                configs[-1]['dataset_name'] = 'new_urea_full'
            elif mol == 0:
                configs[-1]['run_name'] = f'nic test{i}'
                configs[-1]['dataset_name'] = 'new_nic_full'

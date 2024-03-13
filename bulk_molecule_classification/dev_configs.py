from copy import copy

dev = {'run_name': 'dev_nic',
       'dataset_temperature': 'cold',
       'convergence_history': 50,
       'num_convs': 1,
       'embedding_depth': 256,
       'message_depth': 128,
       'dropout': 0.5,
       'graph_norm': 'layer',
       'fc_norm': 'layer',
       'num_fcs': 2,
       'num_epochs': 1000,
       'dataset_size': 1000,
       'conv_cutoff': 6,
       'batch_size': 5,
       'reporting_frequency': 1,
       'train_model': True,
       'trajs_to_analyze_list': None,
       # [f'D:/crystals_extra/classifier_training/paper_nic_clusters2/{ind}/' for ind in [1, 2, 7]],
       # [f'D:/crystals_extra/classifier_training/crystal_in_melt_test10/{ind}/' for ind in range(0, 48)],
       # [f'D:/crystals_extra/classifier_training/paper_nic_clusters2/{ind}/' for ind in range(12)],
       # [f'D:/crystals_extra/defect_clusters_6/{num}/' for num in defect_clusters_6_pure_nic_runs] +
       # [f'D:/crystals_extra/defect_clusters_5_rerun/{num}/' for num in defect_clusters_5_rerun_pure_nic_runs],
       'do_classifier_evaluation': False,
       'classifier_path': None, #'C:/Users/mikem/crystals/classifier_runs/nic_March12_test0_0_best_classifier_checkpoint',
       # 'C:/Users/mikem/crystals/classifier_runs/nic_Feb2_test2_1_best_classifier_checkpoint',  # test1 for cold test2 for hot
       # 'C:/Users/mikem/crystals/classifier_runs/dev_nic_best_hot_classifier_checkpoint', #'C:/Users/mikem/crystals/classifier_runs/nic_test0_3_best_classifier_checkpoint',
       'learning_rate': 1e-4,
       'datasets_path': r'D:/crystals_extra/classifier_training/traj_pickles/',
       'dumps_path': r'D:/crystals_extra/classifier_training/',
       'dumps_dirs': ['new_small_nic_liq_T350', 'nicotinamide_liq', 'new_nic_bulk_small', 'new_nic_bulk_big'],
       # ['new_small_nic_liq_T350', 'nicotinamide_liq', 'bulk_trajs3', 'new_small_bulk'],  # ['new_small_bulk'], #
       'training_temps': [100, 350],
       'dataset_name': 'new_nic_full_redo_check',  # 'nicotinamide_trajectories_dataset_100_350_vf', # 'small_nic_test', #
       'runs_path': r'C:/Users/mikem/crystals/classifier_runs/',
       'results_path': r'D:\crystals_extra\classifier_training\results/',
       'device': 'cuda',
       'seed': 1}
#
#
# dev = {'run_name': 'dev_urea',
#        'dataset_temperature': 'cold',
#        'convergence_history': 50,
#        'num_convs': 1,
#        'embedding_depth': 256,
#        'message_depth': 128,
#        'dropout': 0.5,
#        'graph_norm': 'layer',
#        'fc_norm': 'layer',
#        'num_fcs': 2,
#        'num_epochs': 1000,
#        'dataset_size': 1000,
#        'conv_cutoff': 6,
#        'batch_size': 5,
#        'reporting_frequency': 1,
#        'train_model': False,
#        'trajs_to_analyze_list': None, #['D:/crystals_extra/classifier_training/urea_melt_interface_T200'],
#        'do_classifier_evaluation': True,
#        'classifier_path': None, #'C:/Users/mikem/crystals/classifier_runs/urea_Feb2_test1_1_best_classifier_checkpoint',
#        'learning_rate': 1e-4,
#        'datasets_path': r'D:/crystals_extra/classifier_training/traj_pickles/',
#        'dumps_path': r'D:/crystals_extra/classifier_training/',
#        'dumps_dirs': ['new_small_urea_liq_T350', 'daisuke_small_ureas/T100', 'daisuke_small_ureas/T200', 'urea_liq_T350', 'urea_bulk_trajs/T100', 'urea_bulk_trajs/T200'], #['urea_liq_T350', 'urea_bulk_trajs/T100', 'urea_bulk_trajs/T200'],
#        'training_temps': [100, 200, 350],
#        'dataset_name': 'new_urea_full_mar12_redo_test', #'urea_trajectories_dataset_100_200_350',
#        'runs_path': r'C:/Users/mikem/crystals/classifier_runs/',
#        'results_path': r'D:\crystals_extra\classifier_training\results/',
#        'device': 'cuda',
#        'seed': 1}

configs = []
base_config = {'run_name': 'dev',
               'dataset_temperature': 'cold',
               'convergence_history': 200,
               'num_convs': 1,
               'embedding_depth': 256,
               'message_depth': 128,
               'dropout': 0.5,
               'graph_norm': 'layer',
               'fc_norm': 'layer',
               'num_fcs': 2,
               'num_epochs': 1000,
               'dataset_size': 5000,
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
               'dataset_name': 'new_nic_full_redo',
               'dumps_dirs': None,
               # ['melt_trajs2', 'melt_trajs2'],  # ['urea_bulk_trajs/T100', 'urea_bulk_trajs/T250', 'urea_bulk_trajs/liqT700'],
               'runs_path': r'/vast/mk8347/molecule_clusters/classifier_ckpts/',
               'results_path': r'/vast/mk8347/molecule_clusters/results/',
               'device': 'cuda',
               'seed': 1}

# config_list = [
#     [1, 256, 128, 0, None, None, 2, 1000],  # 0
#     [1, 256, 128, .25, None, None, 2, 1000],  # 1
#     [1, 256, 128, .5, None, None, 2, 1000],  # 2
#     [1, 256, 128, 0, 'layer', None, 2, 1000],  # 3
#     [1, 256, 128, 0, None, 'layer', 2, 1000],  # 4
#     [1, 256, 128, 0, 'layer', 'layer', 2, 1000],  # 5
#     [1, 256, 128, 0.5, 'layer', 'layer', 2, 1000],  # 6
#     [1, 512, 256, 0.25, 'layer', 'layer', 2, 1000],  # 7
#     [1, 256, 256, 0.25, 'layer', 'layer', 2, 1000],  # 8
#     [1, 128, 64, 0.25, 'layer', 'layer', 2, 1000],  # 9
#     [1, 256, 128, 0.25, 'layer', 'batch', 2, 1000],  # 10
#     [1, 256, 128, 0.25, 'layer', 'layer', 1, 1000],  # 11
#     [1, 256, 128, 0.25, 'layer', 'layer', 4, 1000],  # 12
#     [1, 256, 128, 0.25, 'layer', 'layer', 2, 5000],  # 13
# ]
config_list = [
    # [1, 256, 128, 0.25, 'layer', 'layer', 2, 1000, 'cold'],  # 0
    [1, 256, 128, 0.5, 'layer', 'layer', 2, 1000, 'cold'],  # 1
    # [1, 256, 128, 0.25, 'layer', 'layer', 2, 1000, 'hot'],  # 2
    [1, 256, 128, 0.5, 'layer', 'layer', 2, 1000, 'hot'],  # 3
]

test_name = 'March13_2'
for device in ['cuda']:  # , 'cpu']:
    for i in range(len(config_list)):
        for si in range(2):
            for mol in range(2):
                configs.append(copy(base_config))
                configs[-1]['device'] = device
                configs[-1]['seed'] = si
                configs[-1]['num_convs'] = config_list[i][0]
                configs[-1]['embedding_depth'] = config_list[i][1]
                configs[-1]['dropout'] = config_list[i][3]
                configs[-1]['graph_norm'] = config_list[i][4]
                configs[-1]['fc_norm'] = config_list[i][5]
                configs[-1]['num_fcs'] = config_list[i][6]
                configs[-1]['dataset_size'] = config_list[i][7]
                configs[-1]['dataset_temperature'] = config_list[i][8]

                if mol == 1:
                    configs[-1]['run_name'] = f'urea_{test_name}_test{i}_{si}'
                    configs[-1]['dataset_name'] = 'new_urea_full_mar12_redo'
                    configs[-1]['dumps_dirs'] = ['new_small_urea_liq_T350', 'daisuke_small_ureas/T100',
                                                 'daisuke_small_ureas/T200', 'urea_liq_T350', 'urea_bulk_trajs/T100',
                                                 'urea_bulk_trajs/T200']
                    configs[-1]['training_temps'] = [100, 200, 350]
                elif mol == 0:
                    configs[-1]['run_name'] = f'nic_{test_name}_test{i}_{si}'
                    configs[-1]['dataset_name'] = 'new_nic_full_redo_check'
                    configs[-1]['dumps_dirs'] = ['new_small_nic_liq_T350', 'nicotinamide_liq', 'new_nic_bulk_small', 'new_nic_bulk_big']
                    configs[-1]['training_temps'] = [100, 350]
aa = 1
#
# configs = []
# base_config = {'run_name': 'cluster_traj_eval',
#                'convergence_history': 200,
#                'num_convs': 1,
#                'embedding_depth': 256,
#                'message_depth': 128,
#                'dropout': 0.25,
#                'graph_norm': 'layer',
#                'fc_norm': 'layer',
#                'num_fcs': 2,
#                'num_epochs': 1000,
#                'dataset_size': 1000,
#                'conv_cutoff': 6,
#                'batch_size': 5,
#                'reporting_frequency': 1,
#                'train_model': False,
#                'trajs_to_analyze_list': None,
#                'do_classifier_evaluation': False,
#                'classifier_path': '/vast/mk8347/molecule_clusters/classifier_ckpts/nic_Feb2_test2_1_best_classifier_checkpoint',
#                'learning_rate': 1e-4,
#                'datasets_path': r'/vast/mk8347/molecule_clusters/traj_pickles/',
#                'dumps_path': r'/vast/mk8347/molecule_clusters/',
#                'training_temps': [100, 350],
#                'dataset_name': 'new_nic_full',
#                'dumps_dirs': None,
#                # ['melt_trajs2', 'melt_trajs2'],  # ['urea_bulk_trajs/T100', 'urea_bulk_trajs/T250', 'urea_bulk_trajs/liqT700'],
#                'runs_path': r'/vast/mk8347/molecule_clusters/classifier_ckpts/',
#                'results_path': r'/vast/mk8347/molecule_clusters/results/',
#                'device': 'cpu',
#                'seed': 1}
#
# # for evaluation
# for ind in range(0, 60):
#     configs.append(copy(base_config))
#     configs[-1]['trajs_to_analyze_list'] = [f'/vast/mk8347/molecule_clusters/crystal_in_melt_test9/{ind}/']

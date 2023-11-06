from copy import copy

defect_clusters_6_pure_nic_runs = [0, 1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 150, 151,
                                   152, 153, 154, 155, 156, 157, 158, 159, 2, 200, 201, 202, 203, 204,
                                   205, 206, 207, 208, 209, 250, 251, 252, 253, 254, 255, 256, 257, 258,
                                   259, 3, 4, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 7, 8, 9]
defect_clusters_5_rerun_pure_nic_runs = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110,
                                         111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 13, 14, 15, 16, 17,
                                         18, 19, 2, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                                         210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 3, 4, 5, 6, 7, 8, 9]

dev = {'run_name': 'dev',
       'convergence_history': 25,
       'num_convs': 1,
       'embedding_depth': 256,
       'message_depth': 128,
       'dropout': 0.25,
       'graph_norm': 'graph layer',
       'fc_norm': 'layer',
       'num_fcs': 4,
       'num_epochs': 50,
       'dataset_size': 250,
       'conv_cutoff': 6,
       'batch_size': 5,
       'reporting_frequency': 1,
       'train_model': False,
       'trajs_to_analyze_list': [f'D:/crystals_extra/defect_clusters_6/{num}/' for num in defect_clusters_6_pure_nic_runs[:25]],  # + [f'D:/crystals_extra/classifier_training/melt_trajs2/{num}/' for num in range(2)] +
       # [f'D:/crystals_extra/defect_clusters_5_rerun/{num}/' for num in defect_clusters_5_rerun_pure_nic_runs[:25]],
       'do_classifier_evaluation': True,
       'classifier_path': 'C:/Users/mikem/crystals/classifier_runs/dev_best_classifier_checkpoint',
       # 'C:/Users/mikem/crystals/classifier_runs/test3_best_classifier_checkpoint',  # r'C:/Users/mikem/crystals/classifier_runs/test6_best_classifier_checkpoint',
       'learning_rate': 1e-4,
       'datasets_path': r'D:/crystals_extra/classifier_training/',
       'dumps_path': r'D:/crystals_extra/classifier_training/',
       'runs_path': r'C:/Users/mikem/crystals/classifier_runs/',
       'device': 'cuda'}

config1 = {'run_name': 'eval1',
           'convergence_history': 50,
           'num_convs': 1,
           'embedding_depth': 256,
           'message_depth': 128,
           'dropout': 0.25,
           'graph_norm': 'graph layer',
           'fc_norm': 'layer',
           'num_fcs': 2,
           'num_epochs': 1000,
           'dataset_size': 100000,
           'conv_cutoff': 6,
           'batch_size': 1,
           'reporting_frequency': 1,
           'train_model': False,
           'do_classifier_evaluation': True,
           'classifier_path': r'/vast/mk8347/molecule_clusters/classifier_ckpts/test1_best_classifier_checkpoint',
           'trajs_to_analyze_list': [f'/vast/mk8347/molecule_clusters/defect_clusters_6/{num}/' for num in defect_clusters_6_pure_nic_runs[:30]],  # + [f'D:/crystals_extra/classifier_training/melt_trajs2/{num}/' for num in range(2)] +
           # [f'D:/crystals_extra/defect_clusters_5_rerun/{num}/' for num in defect_clusters_5_rerun_pure_nic_runs[:25]],
           'learning_rate': 1e-4,
           'datasets_path': r'/vast/mk8347/molecule_clusters/bulk_trajs1/',
           'dumps_path': r'/vast/mk8347/molecule_clusters/',
           'runs_path': r'/vast/mk8347/molecule_clusters/classifier_ckpts/',
           'device': 'cpu'}

config2 = copy(config1)
config2['run_name'] = 'eval2'
config2['trajs_to_analyze_list'] = [f'/vast/mk8347/molecule_clusters/defect_clusters_6/{num}/' for num in defect_clusters_6_pure_nic_runs[30:]]

config3 = copy(config1)
config3['run_name'] = 'eval3'
config3['trajs_to_analyze_list'] = [f'/vast/mk8347/molecule_clusters/defect_clusters_5_rerun/{num}/' for num in defect_clusters_5_rerun_pure_nic_runs[30:]]

config4 = copy(config1)
config4['run_name'] = 'eval4'
config4['trajs_to_analyze_list'] = [f'/vast/mk8347/molecule_clusters/defect_clusters_5_rerun/{num}/' for num in defect_clusters_5_rerun_pure_nic_runs[30:]]

config5 = copy(config1)
config5['run_name'] = 'eval5'
config5['trajs_to_analyze_list'] = [f'/vast/mk8347/molecule_clusters/melt_trajs2/{num}/' for num in range(4)]

config6 = copy(config1)
config6['run_name'] = 'eval6'
config6['trajs_to_analyze_list'] = [f'D:/crystals_extra/classifier_training/melt_trajs2/{num}/' for num in range(4, 9)]

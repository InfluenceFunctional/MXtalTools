from copy import copy

defect_clusters_6_pure_nic_runs = [0, 1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 2, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 3, 4, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 7, 8, 9]
defect_clusters_5_rerun_pure_nic_runs = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 13, 14, 15, 16, 17, 18, 19, 2, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 3, 4, 5, 6, 7, 8, 9]

dev = {'run_name': 'dev',
       'num_convs': 2,
       'embedding_depth': 128,
       'message_depth': 64,
       'dropout': 0.25,
       'graph_norm': 'graph layer',
       'fc_norm': 'layer',
       'num_fcs': 2,
       'num_epochs': 1000,
       'dataset_size': 1000,
       'conv_cutoff': 6,
       'batch_size': 1,
       'reporting_frequency': 1,
       'train_model': False,
       'trajs_to_analyze_list': [f'D:/crystals_extra/defect_clusters_6/{num}/' for num in defect_clusters_6_pure_nic_runs[:25]] +
                                [f'D:/crystals_extra/defect_clusters_5_reruns/{num}/' for num in defect_clusters_5_rerun_pure_nic_runs[:25]],
       'do_classifier_evaluation': True,
       'classifier_path': r'C:\Users\mikem\crystals\classifier_runs/test6_best_classifier_checkpoint',
       'learning_rate': 1e-4,
       'datasets_path': r'C:/Users/mikem/crystals/clusters/cluster_structures/bulk_trajs1/',
       'dumps_path': r'C:\Users\mikem\crystals\clusters\cluster_structures/',
       'runs_path': r'C:\Users\mikem\crystals\classifier_runs',
       'device': 'cuda'}

config1 = {'run_name': 'test1',
           'num_convs': 2,
           'embedding_depth': 128,
           'message_depth': 64,
           'dropout': 0.25,
           'graph_norm': 'graph layer',
           'fc_norm': 'layer',
           'num_fcs': 2,
           'num_epochs': 1000,
           'dataset_size': 10000,
           'conv_cutoff': 6,
           'batch_size': 1,
           'reporting_frequency': 5,
           'train_model': True,
           'classifier_path': None,
           'learning_rate': 1e-3,
           'datasets_path': r'/vast/mk8347/molecule_clusters/bulk_trajs1/',
           'dumps_path': r'/vast/mk8347/molecule_clusters/',
           'runs_path': r'/vast/mk8347/molecule_clusters/classifier_ckpts/',
           'device': 'cpu'}

config2 = copy(config1)
config2['run_name'] = 'test2'
config2['batch_size'] = 5

config3 = copy(config1)
config3['run_name'] = 'test3'
config3['embedding_depth'] = 256
config3['fc_depth'] = 256

config4 = copy(config1)
config4['run_name'] = 'test4'
config4['dropout'] = 0
config4['fc_norm'] = None
config4['graph_norm'] = None

config5 = copy(config1)
config5['run_name'] = 'test5'
config5['num_convs'] = 4

config6 = copy(config1)
config6['run_name'] = 'test6'
config6['learning_rate'] = 1e-4

config7 = copy(config1)
config7['run_name'] = 'test7'
config7['device'] = 'cuda'

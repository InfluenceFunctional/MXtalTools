from copy import copy

dev = {'run_name': 'dev',
       'num_convs': 2,
       'embedding_depth': 128,
       'message_depth': 64,
       'dropout': 0.25,
       'graph_norm': 'graph layer',
       'fc_norm': 'layer',
       'num_fcs': 2,
       'num_epochs': 1000,
       'dataset_size': 100,
       'conv_cutoff': 6,
       'batch_size': 1,
       'reporting_frequency': 5,
       'train_model': True,
       'classifier_path': None,
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

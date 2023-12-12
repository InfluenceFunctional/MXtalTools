from copy import copy

defect_clusters_6_pure_nic_runs = [0, 1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 150, 151,
                                   152, 153, 154, 155, 156, 157, 158, 159, 2, 200, 201, 202, 203, 204,
                                   205, 206, 207, 208, 209, 250, 251, 252, 253, 254, 255, 256, 257, 258,
                                   259, 3, 4, 5, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 7, 8, 9]
defect_clusters_5_rerun_pure_nic_runs = [0, 1, 10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110,
                                         111, 112, 113, 114, 115, 116, 117, 118, 119, 12, 13, 14, 15, 16, 17,
                                         18, 19, 2, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                                         210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 3, 4, 5, 6, 7, 8, 9]
#
# dev = {'run_name': 'dev_nic',
#        'convergence_history': 50,
#        'num_convs': 1,
#        'embedding_depth': 256,
#        'message_depth': 128,
#        'dropout': 0.25,
#        'graph_norm': 'graph layer',
#        'fc_norm': 'layer',
#        'num_fcs': 2,
#        'num_epochs': 1000,
#        'dataset_size': 10000,
#        'conv_cutoff': 6,
#        'batch_size': 5,
#        'reporting_frequency': 1,
#        'train_model': False,
#        'trajs_to_analyze_list': ['D:/crystals_extra/classifier_training/crystals_in_melt_test3']
#        # ["D:/crystals_extra/classifier_training/melt_crystal_trajs/melt_test7"], #[f'D:/crystals_extra/classifier_training/melt_trajs2/{num}/' for num in range(9)],# +  # FORWARD SLASHES ONLY
#        # [f'D:/crystals_extra/defect_clusters_6/{num}/' for num in defect_clusters_6_pure_nic_runs] +
#        # [f'D:/crystals_extra/defect_clusters_5_rerun/{num}/' for num in defect_clusters_5_rerun_pure_nic_runs],
#        'do_classifier_evaluation': False,
#        'classifier_path': 'C:/Users/mikem/crystals/classifier_runs/dev_nic_1conv_best_classifier_checkpoint',
#        'learning_rate': 1e-4,
#        'datasets_path': r'D:/crystals_extra/classifier_training/',
#        'dumps_path': r'D:/crystals_extra/classifier_training/',
#        'dumps_dirs': ['nicotinamide_liq', 'bulk_trajs3'],
#        'training_temps': [100, 350],
#        'dataset_name': 'nicotinamide_trajectories_dataset_100_350_vf',
#        'runs_path': r'C:/Users/mikem/crystals/classifier_runs/',
#        'device': 'cuda',
#        'seed': 1}

dev = {'run_name': 'dev_urea',
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
       'train_model': False,
       'trajs_to_analyze_list': ['D:/crystals_extra/classifier_training/urea_melt_interface_T200'],
       'do_classifier_evaluation': False,
       'classifier_path': 'C:/Users/mikem/crystals/classifier_runs/dev_urea1_best_classifier_checkpoint',
       'learning_rate': 1e-4,
       'datasets_path': r'D:/crystals_extra/classifier_training/',
       'dumps_path': r'D:/crystals_extra/classifier_training/',
       'dumps_dirs': ['urea_liq_T350', 'urea_bulk_trajs/T100', 'urea_bulk_trajs/T200'],
       'training_temps': [100, 200, 350],
       'dataset_name': 'urea_trajectories_dataset_100_200_350',
       'runs_path': r'C:/Users/mikem/crystals/classifier_runs/',
       'device': 'cuda',
       'seed': 1}

configs = []
base_config = {'run_name': 'dev',
               'num_forms': 10,
               'num_topologies': 2,
               'mol_num_atoms': 15,  # 8 for Urea 15 for Nicotinamide
               'convergence_history': 50,
               'num_convs': 2,
               'embedding_depth': 256,
               'message_depth': 128,
               'dropout': 0.25,
               'graph_norm': 'graph layer',
               'fc_norm': 'layer',
               'num_fcs': 2,
               'num_epochs': 1000,
               'dataset_size': 100,
               'conv_cutoff': 6,
               'batch_size': 5,
               'reporting_frequency': 1,
               'train_model': True,
               'trajs_to_analyze_list': None,
               'do_classifier_evaluation': False,
               'classifier_path': None,
               'learning_rate': 1e-4,
               'datasets_path': r'/vast/mk8347/molecule_clusters/bulk_trajs1/',
               'dumps_path': r'/vast/mk8347/molecule_clusters/',
               'dumps_dirs': None,  # ['melt_trajs2', 'melt_trajs2'],  # ['urea_bulk_trajs/T100', 'urea_bulk_trajs/T250', 'urea_bulk_trajs/liqT700'],
               'dataset_name': None,  # 'nicotinamide_trajectories_dataset',
               'runs_path': r'/vast/mk8347/molecule_clusters/classifier_ckpts/',
               'device': 'cuda',
               'seed': 1}

for i in range(4):
    for mol in range(0, 2):
        configs.append(copy(base_config))
        configs[-1]['run_name'] = f'mol {mol} test{i}'
        configs[-1]['seed'] = i
        if mol == 1:
            configs[-1]['num_forms'] = 7
            configs[-1]['mol_num_atoms'] = 8
            configs[-1]['dumps_dirs'] = ['urea_bulk_trajs/T100', 'urea_bulk_trajs/T200', 'urea_bulk_trajs/liqT700']
            configs[-1]['dataset_name'] = 'urea_trajectories_dataset_100_200_700'
        elif mol == 0:
            configs[-1]['num_forms'] = 10
            configs[-1]['mol_num_atoms'] = 15
            configs[-1]['dumps_dirs'] = ['bulk_trajs3']
            configs[-1]['dataset_name'] = 'nic_trajectories_dataset_100_350_800'

'''
configs = []
base_config = {'run_name': 'test1',
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
               'train_model': True,
               'do_classifier_evaluation': False,
               'classifier_path': None,  # r'/vast/mk8347/molecule_clusters/classifier_ckpts/test1_best_classifier_checkpoint',
               'trajs_to_analyze_list': None,
               'learning_rate': 1e-4,
               'datasets_path': r'/vast/mk8347/molecule_clusters/bulk_trajs1/',
               'dumps_path': r'/vast/mk8347/molecule_clusters/',
               'runs_path': r'/vast/mk8347/molecule_clusters/classifier_ckpts/',
               'device': 'cuda',
               'seed': 1
               }

ind = 1
for i in [1, 2, 4, 6]:
    for j in [1, 2, 3, 4, 5]:
        configs.append(copy(base_config))
        configs[-1]['run_name'] = f'test{ind}'
        configs[-1]['num_convs'] = i
        configs[-1]['seed'] = j
        ind += 1
'''

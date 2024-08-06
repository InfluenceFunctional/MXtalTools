from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

# test our standard embedding regressor with unfrozen and untrained autoencoder

base_config = load_yaml('../../base/embedding_regressor.yaml')

models = [r'/best_autoencoder_autoencoder_tests_qm9_test23_4_26-02-22-29-57_fixed',  # no protons
          r'/best_autoencoder_experiments_autoencoder_tests_qm9_test24_25_11-05-19-15-19_fixed'  # protons
          ]

config_list = []
for features in [32, 64, 128, 256, 512]:
    for layers in [1, 2, 4, 8, 16]:
        config_list.append([features, layers])

ind = 0
for ix1 in range(len(config_list)):
    config = copy(base_config)
    config['machine'] = 'cluster'
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    config['dataset']['filter_protons'] = False
    config['model_paths']['autoencoder'] = models[1]
    config['dataset']['regression_target'] = 'gap_energy'
    config['embedding_regressor']['freeze_encoder'] = True
    config['embedding_regressor']['model']['hidden_dim'] = config_list[ix1][0]
    config['embedding_regressor']['model']['num_layers'] = config_list[ix1][1]
    config['embedding_regressor']['model']['dropout'] = 0.1
    config['embedding_regressor']['model']['norm_mode'] = 'layer'
    config['dataset']['max_dataset_length'] = 10000000

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

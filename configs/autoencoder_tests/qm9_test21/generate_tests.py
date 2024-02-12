from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

# decoder layers, decoder points, weight decay, filter protons, positional noise, dropout, embedding_dim, bottleneck_dim, num_convs, num_nodewise,
# ramp_depth, lr shrink lambda, max batch, min_lr, batch_growth_increment
config_list = [
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .999, 10000, 5e-5, 0.5],  # 0 - converged
    [4, 256, 0.05, False, 0, 0, 256, 256, 1, 4, True, .999, 10000, 5e-5, 0.5],  # 1 - converged too early
    [4, 256, 0.05, True, 0, 0, 256, 256, 0, 4, True, .999, 10000, 5e-5, 0.5],  # 2  cancelled - flat
    [4, 256, 0.05, False, 0, 0, 256, 256, 0, 4, True, .999, 10000, 5e-5, 0.5],  # 3  cancelled - flat

    [4, 256, 0.05, True, 0, 0, 256, 256, 2, 2, True, .999, 10000, 5e-5, 0.5],  # 4 - converged - ~worse than 0
    [4, 256, 0.05, False, 0, 0, 256, 256, 2, 2, True, .999, 10000, 5e-5, 0.5],  # 5 - converged too early - ~worse than 1

    [4, 256, 0.1, True, 0, 0, 256, 256, 1, 4, True, .999, 10000, 5e-5, 0.5],  # 6 - ~
    [8, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .999, 10000, 5e-5, 0.5],  # 7 - ~
    [4, 256, 0.05, True, 0, 0, 512, 512, 1, 4, True, .999, 10000, 5e-5, 0.5],  # 8 - fast at first, then unstable, then OK
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, False, .999, 10000, 5e-5, 0.5],  # 9 - ~

    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 10000, 5e-5, 0.5],  # 10 - better than prior runs
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 500, 5e-5, 0.5],  # 11 - rapid training but levels out around .96
    [4, 256, 0.05, True, 0, 0, 512, 128, 1, 4, True, .99, 10000, 5e-5, 0.5],  # 12 - same as 10
    [4, 512, 0.05, False, 0, 0, 512, 512, 1, 4, True, .99, 10000, 5e-5, 0.5],  # 13 - ok at first then very rapid convergence, eventual overfit starting at ~90%
    # seems like low batch size and also good LR decay are important
    # try 1) slower batch increase, 2) low batch max, 3) lower LR min
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 500, 1e-5, 0.5],  # 14
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 500, 1e-5, 0.05],  # 15
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 5000, 1e-5, 0.05],  # 16
    [4, 512, 0.05, True, 0, 0, 512, 512, 1, 4, True, .99, 500, 1e-5, 0.5],  # 17

]

np.random.seed(1)
ind = 0
for ix1 in range(len(config_list)):

    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    config['autoencoder']['model']['num_decoder_layers'] = config_list[ix1][0]
    config['autoencoder']['model']['num_decoder_points'] = config_list[ix1][1]
    config['autoencoder']['optimizer']['weight_decay'] = config_list[ix1][2]
    config['dataset']['filter_protons'] = config_list[ix1][3]
    config['autoencoder_positional_noise'] = config_list[ix1][4]
    config['autoencoder']['model']['graph_node_dropout'] = config_list[ix1][5]
    config['autoencoder']['model']['decoder_dropout_probability'] = config_list[ix1][5]
    config['autoencoder']['model']['embedding_depth'] = config_list[ix1][6]
    config['autoencoder']['model']['bottleneck_dim'] = config_list[ix1][7]
    config['autoencoder']['model']['num_graph_convolutions'] = config_list[ix1][8]
    config['autoencoder']['model']['nodewise_fc_layers'] = config_list[ix1][9]
    config['autoencoder']['model']['decoder_ramp_depth'] = config_list[ix1][10]
    config['autoencoder']['optimizer']['lr_shrink_lambda'] = config_list[ix1][11]
    config['max_batch_size'] = config_list[ix1][12]
    config['autoencoder']['optimizer']['min_lr'] = config_list[ix1][13]
    config['batch_growth_increment'] = config_list[ix1][14]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
- testing new equivariant model again
looking for fast training with minimal overfitting and good stability

1) iterate on recent bests

Best examples:
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 3 converged to 96% in 6k steps, then noisy
    [4, 171, 171, 1, 4, 9, 0, 1e-4, 256, False, 'equivariant softmax'],  # 9 CUDA assert after 419, was the fastest yet seen

"""

configs = [
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 512, False, 'equivariant combo'],  # 0
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 256, False, 'equivariant softmax'],  # 1
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 342, False, 'equivariant softmax'],  # 2

    [0, 128, 128, 8, 8, 9, 0, 5e-5, 256, False, 'equivariant softmax'],  # 3

    [4, 171, 171, 1, 4, 9, 0, 1e-4, 256, False, 'equivariant softmax'],  # 4

    [0, 342, 342, 8, 1, 9, 0, 1e-4, 512, True, 'equivariant combo'],  # 5
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 256, True, 'equivariant softmax'],  # 6
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 342, True, 'equivariant softmax'],  # 7

    [0, 128, 128, 8, 8, 9, 0, 5e-5, 256, True, 'equivariant softmax'],  # 8

    [4, 171, 171, 1, 4, 9, 0, 1e-4, 256, True, 'equivariant softmax'],  # 9

]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['autoencoder']['model']['num_graph_convolutions'] = configs[ii][0]
    config['autoencoder']['model']['embedding_depth'] = configs[ii][1]
    config['autoencoder']['model']['graph_message_depth'] = configs[ii][2]
    config['autoencoder']['model']['nodewise_fc_layers'] = configs[ii][3]
    config['autoencoder']['model']['num_decoder_layers'] = configs[ii][4]
    config['autoencoder']['model']['num_attention_heads'] = configs[ii][5]
    config['autoencoder_positional_noise'] = configs[ii][6]
    config['autoencoder']['optimizer']['encoder_init_lr'] = configs[ii][7]
    config['autoencoder']['model']['num_decoder_points'] = configs[ii][8]
    config['autoencoder']['model']['decoder_ramp_depth'] = configs[ii][9]
    config['autoencoder']['model']['graph_aggregator'] = configs[ii][10]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

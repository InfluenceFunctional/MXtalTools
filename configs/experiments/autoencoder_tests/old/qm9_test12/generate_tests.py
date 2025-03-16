from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

"""
- testing new equivariant model again
looking for fast training with minimal overfitting and good stability

1) train small no-conv nets faster
2) is there a goldilocks conv net
3) keep mucking about with conv options

Findings:
NaN issues everywhere
Combo - slow, unstable, and overfits
Convolutions - Highly unclear
Big decoders - yield overfitting?
Possibly less decoder points could train faster

Best examples:
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 3 converged to 96% in 6k steps, then noisy
    [4, 171, 171, 1, 4, 9, 0, 1e-4, 256, False, 'equivariant softmax'],  # 9 CUDA assert after 419, was the fastest yet seen

"""

configs = [
    [0, 56, 56, 8, 8, 9, 0, 1e-3, 512, False, 'equivariant softmax'],  # 0 consistent but slow

    [0, 76, 76, 8, 8, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 1 quite slow, type NaN at 974 epochs
    [0, 128, 128, 8, 8, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 2 fast at first, type NaN after 98 epochs
    [0, 342, 342, 8, 1, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 3 converged to 96% in 6k steps, then noisy
    [0, 513, 513, 8, 1, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 4 similar to 3 so far, but overfitting

    [0, 128, 128, 8, 8, 9, 0, 1e-3, 512, False, 'equivariant softmax'],  # 5 type NaN after 111, was looking ok

    [0, 171, 171, 1, 8, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 6 overfitting NaN after 200
    [0, 342, 342, 1, 8, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 7 Nan after 104, was looking ok but overfitting
    [0, 513, 513, 1, 8, 9, 0, 1e-4, 512, False, 'equivariant softmax'],  # 8 NaN in step 2

    [4, 171, 171, 1, 4, 9, 0, 1e-4, 256, False, 'equivariant softmax'],  # 9 CUDA assert after 419, was the fastest yet seen
    [4, 171, 171, 1, 4, 9, 0, 1e-4, 768, False, 'equivariant softmax'],  # 10 very unstable and encoder NaN

    [4, 171, 171, 1, 4, 9, 0, 1e-4, 512, True, 'equivariant softmax'],  # 11 epoch 2 type NaN
    [0, 128, 128, 8, 8, 9, 0, 1e-4, 512, True, 'equivariant softmax'],  # 12 step 0 type NaN

    [0, 76, 76, 8, 8, 9, 0, 1e-4, 512, False, 'equivariant combo'],  # 13  very slow convergence + overfit somehow
    [0, 128, 128, 8, 8, 9, 0, 1e-4, 512, False, 'equivariant combo'],  # 14 very slow convergence + overfit somehow
    [4, 171, 171, 1, 4, 9, 0, 1e-4, 512, True, 'equivariant combo'],  # 15 encoder NaN on step 3
    [0, 171, 171, 1, 8, 9, 0, 1e-4, 512, False, 'equivariant combo'],  # 16 slow. encoder NaN at epoch 100
    [4, 171, 171, 1, 4, 9, 0, 1e-4, 256, False, 'equivariant combo'],  # 17 OK, slow but not crazy clow

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

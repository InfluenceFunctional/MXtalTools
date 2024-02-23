from common.config_processing import load_yaml
import yaml
from copy import copy
import numpy as np

base_config = load_yaml('base.yaml')

# decoder layers, decoder points, weight decay, filter protons, positional noise, dropout, embedding_dim, bottleneck_dim, num_convs, num_nodewise,
# ramp_depth, lr shrink lambda, max batch, min_lr, batch_growth_increment, max_lr, variational, guess protons, KLD_threshold, decoder norm mode, overlap_eps
config_list = [
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 0 - converged
    [4, 256, 0.05, False, 0, 0, 256, 256, 1, 4, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 1 - converged too early
    [4, 256, 0.05, True, 0, 0, 256, 256, 0, 4, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 2  cancelled - flat
    [4, 256, 0.05, False, 0, 0, 256, 256, 0, 4, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 3  cancelled - flat

    [4, 256, 0.05, True, 0, 0, 256, 256, 2, 2, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 4 - converged - ~worse than 0
    [4, 256, 0.05, False, 0, 0, 256, 256, 2, 2, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 5 - converged too early - ~worse than 1

    [4, 256, 0.1, True, 0, 0, 256, 256, 1, 4, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 6 - ~
    [8, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 7 - ~
    [4, 256, 0.05, True, 0, 0, 512, 512, 1, 4, True, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 8 - fast at first, then unstable, then OK
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, False, .999, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 9 - ~

    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 10 - better than prior runs
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 500, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 11 - rapid training but levels out around .96
    [4, 256, 0.05, True, 0, 0, 512, 128, 1, 4, True, .99, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 12 - same as 10
    [4, 512, 0.05, False, 0, 0, 512, 512, 1, 4, True, .99, 10000, 5e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 13 - ok at first then very rapid convergence, eventual overfit starting at ~90%
    # seems like low batch size and also good LR decay are important
    # try 1) slower batch increase, 2) low batch max, 3) lower LR min
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 500, 1e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 14 - 2nd best
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 500, 1e-5, 0.05, 5e-4, False, False, 0.95, 'layer', 0.001],  # 15 - new best
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 5000, 1e-5, 0.05, 5e-4, False, False, 0.95, 'layer', 0.001],  # 16 - slightly better than 11
    [4, 512, 0.05, True, 0, 0, 512, 512, 1, 4, True, .99, 500, 1e-5, 0.5, 5e-4, False, False, 0.95, 'layer', 0.001],  # 17 - tied 2nd best
    # LR peak is perhaps too high
    # regardless of differences in configs, GPU/random noise gave them different final batch sizes, and the results track them well
    # small batches, even lower LR minimum and maximum
    # also we are ready to train with protons, proton replacement, and variational annealing
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 18 - converges firmly to 0.975
    [4, 256, 0.05, False, 0, 0, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 19 - overfitting around 0.9
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, True, False, 0.95, 'layer', 0.001],  # 20 - pretty beautiful convergence tbh
    [4, 256, 0.05, False, 0, 0, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, True, 0.95, 'layer', 0.001],  # 21 - overfitting around 0.84
    # need to do better checkpointing on variational
    # also would be nice if protonated models didn't overfit
    [4, 256, 0.05, True, 0, 0, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, True, False, 0.97, 'layer', 0.001],  # 22
    [4, 256, 0.05, False, 0, 0, 128, 128, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 23 - overfits but at lower fidelity
    [4, 256, 0.05, False, 0, 0, 128, 128, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, True, 0.95, 'layer', 0.001],  # 24 - overfits but at lower fidelity
    # try to fit proton model
    [4, 256, 0.05, False, 0, 0.1, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 25 - slightly better convergence without overfit
    [4, 256, 0.05, False, 0.1, 0, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 26 - approx 19
    [4, 256, 0.05, False, 0, 0.1, 512, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 27 - slightly worse convergence but without overfit
    [4, 256, 0.05, False, 0, 0, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'batch', 0.001],  # 28 - meh
    # a little dropout appears to fix overfit
    # but still want it to converge quickly / deeply
    # try size / norm
    [2, 256, 0.05, False, 0, 0.1, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 29 - faster but levelled out near 25, where 25 is still decreasing
    [8, 256, 0.05, False, 0, 0.1, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 30 - same as 31
    [4, 256, 0.05, False, 0, 0.1, 256, 256, 1, 2, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 31 - worse
    # less decoders
    # deeper
    [1, 256, 0.05, False, 0, 0.1, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 32 - this one probably the best of the batch
    [2, 256, 0.05, False, 0, 0.1, 256, 256, 1, 8, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 33 - also good
    [2, 256, 0.05, False, 0, 0.1, 256, 256, 2, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 34 - bit less good
    [2, 256, 0.05, False, 0, 0.1, 512, 512, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 35 - bit less good
    # all 4 of these are improvements on 29, levelling out in similar range from 0.05-0.06
    [1, 256, 0.05, False, 0, 0.1, 256, 256, 1, 8, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 36 # normal convergence but to almost 96
    [1, 512, 0.05, False, 0, 0.1, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 37 # bit better than prior works
    [1, 256, 0.05, False, 0, 0.1, 256, 256, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 38 # slightly worse than prior
    [1, 256, 0.05, False, 0, 0.1, 512, 512, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 39  # very fast early convergence to 96+
    # possible LR peak is still too high
    # depth + shallow decoder appears quite effective
    [1, 256, 0.05, False, 0, 0.1, 512, 512, 1, 8, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 40 - worse about 94
    [1, 512, 0.05, False, 0, 0.1, 512, 512, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 41 - about as good as 39
    [1, 256, 0.05, False, 0, 0.1, 512, 512, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, True, 0.95, 'layer', 0.001],  # 42 - not great - around 0.8 with some overfit
    [1, 256, 0.05, True, 0, 0.1, 512, 512, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 43 - ultra fast convergence below prior best
    # possible that LR minimum could go even lower
    # proton prediction is hard - maybe even more regularization
    # not clear about where architecture should go
    [1, 256, 0.05, False, 0, 0.1, 512, 512, 1, 4, True, .99, 300, 1e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 44
    [1, 256, 0.05, True, 0, 0.1, 512, 512, 1, 4, True, .99, 300, 1e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 45 - cancelled / restarted
    [1, 256, 0.05, False, 0, 0.1, 1024, 512, 1, 4, True, .99, 300, 1e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.001],  # 46
    [1, 256, 0.05, True, 0, 0.1, 512, 512, 1, 4, True, .99, 300, 5e-6, 0.05, 2e-4, False, False, 0.95, 'layer', 0.0005],  # 47
    # weirdly - reconstruction loss is overfitting a bit but other losses show test as slightly better (I think due to dropout)
    # it's also not showing up in the overall loss
    # maybe sigma could go even lower?
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
    config['autoencoder']['optimizer']['max_lr'] = config_list[ix1][15]
    config['autoencoder']['model']['variational_encoder'] = config_list[ix1][16]
    config['autoencoder']['infer_protons'] = config_list[ix1][17]
    config['autoencoder']['KLD_threshold'] = config_list[ix1][18]
    config['autoencoder']['model']['decoder_norm_mode'] = config_list[ix1][19]
    config['autoencoder']['overlap_eps']['test'] = config_list[ix1][20]


    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')
configs_list = [
    {'min_batch_size': 10,
     'max_batch_size': 132,
     'seeds': {'model': 1},
     'positional_noise': {'autoencoder': 0},
     'autoencoder': {
         'overlap_eps': {'train': 1e-3},
         'infer_protons': False,
         'filter_protons': True,
         'sigma_threshold': 0.05,
         'type_distance_scaling': 2,
         'optimizer': {
             'init_lr': 5e-5,
             'encoder_init_lr': 4e-4,
             'decoder_init_lr': 2e-5,
             'max_lr': 1e-4,
             'min_lr': 1e-6,
             'weight_decay': 3e-3,
             'lr_growth_lambda': 1.01,
             'lr_shrink_lambda': 0.99,
             'beta1': 0.908,
             'beta2': 0.998,
         },
         'model': {
             'bottleneck_dim': 648,
             'encoder': {
                 'graph': {
                     'node_dim': 1024,
                     'message_dim': 128,
                     'embedding_dim': 512,
                     'num_convs': 3,
                     'fcs_per_gc': 2,
                     'dropout': 0.05,
                     'cutoff': 6,
                     'radial_embedding': 'bessel',
                     'norm': None,
                     'vector_norm': 'graph vector layer',
                 }},
             'decoder': {
                 'fc': {
                     'hidden_dim': 1024,
                     'num_layers': 4,
                     'dropout': 0.05,
                     'norm': None,
                     'vector_norm': None, },
                 'num_nodes': 1024,
                 'ramp_depth': True,
             }}}},  # 0: actual copy of sweep 104 - bit deeper

]


def overwrite_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            assert k in d1.keys()
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1
"""
sequentially numbered sweep configs
"""
for ix1 in range(len(configs_list)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ix1)

    run_config = configs_list[ix1]
    overwrite_nested_dict(config, run_config)

    with open(str(ix1) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

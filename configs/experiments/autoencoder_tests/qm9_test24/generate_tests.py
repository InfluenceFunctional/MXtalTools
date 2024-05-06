from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('../../base/autoencoder.yaml')

config_list = [
    {  # base config
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},
    {  # base config new seed
        'seeds': {'model': 2},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 2,
                        'fcs_per_gc': 2,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},
    {  # 4 fcs 1 conv
        'seeds': {'model': 1},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},
    {  # 4 fcs 1 conv new seed
        'seeds': {'model': 2},
        'dataset': {'filter_protons': True},
        'autoencoder': {
            'infer_protons': False,
            'type_distance_scaling': 2,
            'model': {
                'bottleneck_dim': 512,
                'encoder': {
                    'graph': {
                        'node_dim': 512,
                        'message_dim': 128,
                        'embedding_dim': 512,
                        'num_convs': 1,
                        'fcs_per_gc': 4,
                        'dropout': 0.1,
                        'cutoff': 14,
                        'norm': 'graph layer'}},
                'decoder': {
                    'fc': {
                        'hidden_dim': 512,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'norm': 'layer'},
                    'num_nodes': 256
                }}}},
]


def write_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            assert k in d1.keys()
            d1[k] = write_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


ind = 0
for ix1 in range(len(config_list)):
    config = copy(base_config)
    config['machine'] = 'cluster'
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)

    run_config = config_list[ix1]
    write_nested_dict(config, run_config)

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

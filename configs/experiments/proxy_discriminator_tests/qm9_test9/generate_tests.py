from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

"""
Proxy discriminator params
"""
configs = [
    {
        'num_layers': 4,
        'depth': 256,
        'dropout': 0,
        'norm': None,
        'max_batch_size': 20000,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 0 - baseline
    {
        'num_layers': 8,
        'depth': 256,
        'dropout': 0,
        'norm': None,
        'max_batch_size': 20000,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 1 - more layers
    {
        'num_layers': 4,
        'depth': 256,
        'dropout': 0,
        'norm': 'layer',
        'max_batch_size': 20000,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 2 - layer norm
    {
        'num_layers': 4,
        'depth': 256,
        'dropout': 0.25,
        'norm': None,
        'max_batch_size': 20000,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 3 - dropout
    {
        'num_layers': 4,
        'depth': 256,
        'dropout': 0,
        'norm': None,
        'max_batch_size': 200,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 4 - small max batch
    {
        'num_layers': 4,
        'depth': 256,
        'dropout': 0,
        'norm': None,
        'max_batch_size': 2000,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 5 - medium batch size
    {
        'num_layers': 4,
        'depth': 256,
        'dropout': 0,
        'norm': None,
        'max_batch_size': 20000,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.1,
        'vdw_turnover_potential': 10,
    },  # 6 - big weight decay
    {
        'num_layers': 4,
        'depth': 256,
        'dropout': 0,
        'norm': None,
        'max_batch_size': 20000,
        'embedding_type': 'autoencoder',
        'resample_each': 20,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 7 - resample less
    {
        'num_layers': 4,
        'depth': 256,
        'dropout': 0,
        'norm': None,
        'max_batch_size': 20000,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.999975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 8 - very slow LR annealing
    {
        'num_layers': 4,
        'depth': 512,
        'dropout': 0,
        'norm': None,
        'max_batch_size': 20000,
        'embedding_type': 'autoencoder',
        'resample_each': 10,
        'buffer_size': 100000,
        'max_lr': 1e-3,
        'init_lr': 1e-4,
        'lr_growth_lambda': 1.01,
        'lr_shrink_lambda': 0.99975,
        'weight_decay': 0.001,
        'vdw_turnover_potential': 10,
    },  # 9 - more width
]

# upshots:
# low weight decay, not too slow LR annealing, moderate batch size, big model, infrequent resampling

ind = 0
for ii, config_i in enumerate(configs):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['proxy_discriminator']['model']['num_layers'] = config_i['num_layers']
    config['proxy_discriminator']['model']['depth'] = config_i['depth']
    config['max_batch_size'] = config_i['max_batch_size']
    config['proxy_discriminator']['model']['norm'] = config_i['norm']
    config['proxy_discriminator']['model']['dropout'] = config_i['dropout']
    config['proxy_discriminator']['embedding_type'] = config_i['embedding_type']
    config['dataset']['resample_each'] = config_i['resample_each']
    config['dataset']['buffer_size'] = config_i['buffer_size']
    config['proxy_discriminator']['optimizer']['max_lr'] = config_i['max_lr']
    config['proxy_discriminator']['optimizer']['init_lr'] = config_i['init_lr']
    config['proxy_discriminator']['optimizer']['lr_growth_lambda'] = config_i['lr_growth_lambda']
    config['proxy_discriminator']['optimizer']['lr_shrink_lambda'] = config_i['lr_shrink_lambda']
    config['proxy_discriminator']['optimizer']['weight_decay'] = config_i['weight_decay']
    config['proxy_discriminator']['vdw_turnover_potential'] = config_i['vdw_turnover_potential']

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

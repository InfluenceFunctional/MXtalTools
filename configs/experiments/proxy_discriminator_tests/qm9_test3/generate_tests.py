from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

"""
Proxy discriminator params
"""
# layers, depth, max batch, norm, vector norm, dropout
configs = [
    [4, 1024, 100, None, None, 0],  # 0
    [4, 1024, 500, None, None, 0],  # 1
    [4, 1024, 1000, None, None, 0],  # 2
    [4, 1024, 2000, None, None, 0],  # 3 - converged fastest/best
    [4, 1024, 4000, None, None, 0],  # 4

    [4, 2048, 2000, None, None, 0],  # 5
    [8, 1024, 2000, None, None, 0],  # 6

]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['proxy_discriminator']['model']['num_layers'] = configs[ii][0]
    config['proxy_discriminator']['model']['depth'] = configs[ii][1]
    config['max_batch_size'] = configs[ii][2]
    config['proxy_discriminator']['model']['norm'] = configs[ii][3]
    config['proxy_discriminator']['model']['vector_norm'] = configs[ii][4]
    config['proxy_discriminator']['model']['dropout'] = configs[ii][5]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

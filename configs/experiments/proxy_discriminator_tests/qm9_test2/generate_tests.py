from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy

base_config = load_yaml('base.yaml')

"""
Proxy discriminator params
"""

configs = [
    # max batch size -- failed - didn't assign the right config
    [4, 256, 200, 'layer', 'vector layer', 0],  #
    [4, 256, 500, 'layer', 'vector layer', 0],  #
    [4, 256, 1000, 'layer', 'vector layer', 0],  #
    [4, 256, 2000, 'layer', 'vector layer', 0],  #
    [4, 256, 5000, 'layer', 'vector layer', 0],  #
    # simple model & size
    [4, 256, 2000, None, None, 0],  #
    [4, 512, 2000, None, None, 0],  #
    [4, 1024, 2000, None, None, 0],  #
    [8, 256, 2000, None, None, 0],  #
    [16, 256, 2000, None, None, 0],  #
    [24, 256, 2000, None, None, 0],  #
    # big & regularized
    [16, 512, 2000, 'layer', 'vector layer', 0.25],  #
]

ind = 0
for ii in range(len(configs)):
    config = copy(base_config)
    config['logger']['run_name'] = config['logger']['run_name'] + '_' + str(ind)
    config['proxy_discriminator']['model']['num_layers'] = configs[ii][0]
    config['proxy_discriminator']['model']['depth'] = configs[ii][1]
    config['proxy_discriminator']['max_batch_size'] = configs[ii][2]
    config['proxy_discriminator']['model']['norm'] = configs[ii][3]
    config['proxy_discriminator']['model']['vector_norm'] = configs[ii][4]
    config['proxy_discriminator']['model']['dropout'] = configs[ii][5]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

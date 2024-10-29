from mxtaltools.common.config_processing import load_yaml
import yaml
from copy import copy
from mxtaltools.constants.space_group_info import SPACE_GROUPS

base_config = load_yaml('base.yaml')

"""
Proxy discriminator params
"""
# layers, depth, max batch, norm, vector norm, dropout
c1 = [4, 1024, 2000, None, None, 0]  # 0
configs = [c1 for _ in range(20)]
sgs_list = list(SPACE_GROUPS.values())

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
    config['generate_sgs'] = sgs_list[ii]

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1

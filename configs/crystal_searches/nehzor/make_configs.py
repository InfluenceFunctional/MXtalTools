import os
from itertools import product

import numpy as np
import yaml
from pathlib import Path
from copy import deepcopy


def load_yaml(path):
    path = Path(path)
    with path.open('r') as f:
        return yaml.safe_load(f), path.parent  # return both content and its directory


def overwrite_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


if __name__ == "__main__":
    base_path = 'base.yaml'
    base, spec_dir = load_yaml(base_path)

    schedules = {
        'elj_uma': [
            {'optim_target': 'elj', 'compression_factor': 1.0, 'init_lr': 0.05, 'max_num_steps': 300},
            {'optim_target': 'uma', 'compression_factor': 0.0, 'init_lr': 0.05, 'max_num_steps': 200},
        ],
        'elj_elj_uma': [
            {'optim_target': 'elj', 'compression_factor': 2.0, 'init_lr': 0.05, 'max_num_steps': 300},
            {'optim_target': 'elj', 'compression_factor': 0.0, 'init_lr': 0.02, 'max_num_steps': 100},
            {'optim_target': 'uma', 'compression_factor': 0.0, 'init_lr': 0.005, 'max_num_steps': 200},
        ],
        'uma_highcomp': [
            {'optim_target': 'uma', 'compression_factor': 5.0, 'init_lr': 0.05, 'max_num_steps': 300},
            {'optim_target': 'uma', 'compression_factor': 0.0, 'init_lr': 0.05, 'max_num_steps': 200}
        ],
        'elj_uma_tgt':[
            {'optim_target': 'elj', 'compression_factor': 0.0, 'init_lr': 0.05, 'max_num_steps': 300, 'target_packing_coeff': 0.73},
            {'optim_target': 'uma', 'compression_factor': 0.0, 'init_lr': 0.05, 'max_num_steps': 200, 'target_packing_coeff': 0.73},
        ],
    }

    # defaults shared across all opt stages (override per-stage above)
    opt_defaults = {
        'enforce_reduced': True,
        'convergence_eps': 0.00001,
        'optimizer_func': 'rprop',
        'anneal_lr': False,
        'grad_norm_clip': 0.1,
        'show_tqdm': True,
        'cutoff': 10,
        'target_packing_coeff': None,
    }

    ind = 40
    n_seeds = 5  # per schedule

    for sched_name, stages in schedules.items():
        for seed_ind in range(n_seeds):
            config = deepcopy(base)
            config['opt_seed'] = int(seed_ind)
            config['run_name'] = f'nehzor_{sched_name}_{seed_ind}'

            config['opt'] = []
            for stage in stages:
                opt_stage = deepcopy(opt_defaults)
                opt_stage.update(stage)
                config['opt'].append(opt_stage)

            config_path = f'nehzor_{ind}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            ind += 1
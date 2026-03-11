"""
A script for loading a batch of molecules and optimizing them against a given property via torch autograd
"""
import gc
import os
from pathlib import Path
from time import sleep

import torch
from tqdm import tqdm

from mxtaltools.common.config_processing import load_yaml, dict2namespace
from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.crystal_search.utils import get_initial_state, init_samples_to_optim, parse_args, parse_opt_config, \
    recover_opt_state, process_target
from mxtaltools.dataset_utils.utils import collate_data_list

if __name__ == '__main__':
    args = parse_args()  # call config with "python run_search.py --config /path/to/config.yaml
    source_dir = Path(__file__).resolve().parent.parent.parent
    if args.config is None:
        config_path = source_dir / 'configs' / 'crystal_searches' / 'base.yaml'
    else:
        config_path = Path(args.config)

    config = dict2namespace(load_yaml(config_path))

    device = config.device

    if device == 'cuda':
        # prevents from dipping into windows virtual vram which is super slow
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)

    if config.target_path is not None:
        target, config = process_target(config)
    else:
        target = None

    if config.init_sample_method == 'data':
        samples_to_optim = torch.load(config.dataset_path, weights_only=False)
        index_block = torch.arange(config.mol_seed * config.num_samples, (config.mol_seed + 1) * config.num_samples)
        samples_to_optim = [samples_to_optim[ind] for ind in index_block]
    else:
        samples_to_optim = init_samples_to_optim(config, target=target)

    out_path = Path(config.out_dir + f"/{config.run_name}.pt")  # where to save outputs
    num_samples = len(samples_to_optim)
    print(f"Starting optimization of {num_samples} crystal samples")

    opt_outs = []
    num_opts = len(config.opt)
    batch_idx = -1
    finished = False
    cursor = 0
    pbar = tqdm(total=num_samples, unit="samples")
    prev_best_samples = None
    while not finished:
        try:
            batch_idx += 1
            crystal_batch = collate_data_list(samples_to_optim[cursor:cursor + config.batch_size]).to(device)

            if (prev_best_samples is None) or (
                    prev_best_samples is not None and len(prev_best_samples) < crystal_batch.num_graphs):
                crystal_batch = get_initial_state(config, crystal_batch, device, batch_idx)
                # target_params = target.full_cell_parameters()[0]
                # crystal_batch.cell_lengths[:, 0] = target_params[2]
                # crystal_batch.cell_lengths[:, 1] = target_params[1]
                # crystal_batch.cell_lengths[:, 2] = target_params[0]
                # crystal_batch.cell_angles[:, 1] = target_params[4]
                # crystal_batch.box_analysis()
            else:
                crystal_batch = recover_opt_state(crystal_batch, config, device, batch_idx, prev_best_samples)

            for opt_ind, opt_config in enumerate(config.opt):
                # do optimization in N stages
                opt_config = parse_opt_config(opt_config, config, device, target)

                'do opt'
                opt_out, opt_record = crystal_batch.optimize_crystal_parameters(return_record = True, **opt_config)
                if config.save_trajs:
                    opt_record.update({'base_crystal': samples_to_optim[0]})
                    torch.save(opt_record, Path(str(out_path).replace('.pt', f'_traj{batch_idx}_{opt_ind}.pt')))

                crystal_batch = collate_data_list(opt_out).to(device)

                if 'uma_predictor' in opt_config.keys():
                    del opt_config['uma_predictor']
                if 'score_model' in opt_config.keys():
                    del opt_config['score_model']

            crystal_batch.box_analysis()
            opt_outs.extend(crystal_batch.cpu().detach().batch_to_list())

            torch.save(opt_outs, out_path)

            cursor += config.batch_size
            prev_best_samples = None
            pbar.update(min(config.batch_size, num_samples - cursor))  # safe final update
            if cursor >= len(samples_to_optim):
                finished = True
            else:
                if config.grow_batch_size:
                    config.batch_size = int(config.batch_size * 1.2)  # keep pushing the batch size between sets
                    print(f"Boosting batch size to {config.batch_size}")

            del crystal_batch


        except (RuntimeError, ValueError) as e:
            if is_cuda_oom(e):
                if config.batch_size == 1:
                    assert False, "Cascading bsz error"
                config.batch_size = max(int(config.batch_size * 0.9), 1)
                del crystal_batch
                if 'uma_predictor' in opt_config.keys():
                    del opt_config['uma_predictor']
                if 'score_model' in opt_config.keys():
                    del opt_config['score_model']
                print(f"OOM error: dropping batch size to {config.batch_size}")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                sleep(0.1)
                if os.path.exists('opt_intermediates.pt'):
                    prev_best_samples = torch.load('opt_intermediates.pt', weights_only=False)
            else:
                raise e

    print(f"Sampling complete! Optimized a total of {len(opt_outs)} crystal samples.")

    # batch = collate_data_list(opt_outs)
    # batch.plot_batch_cell_params(space='real', quantiles=[0.1, 0.5], split_by_sg=True)
    #
    # batch.plot_batch_density_funnel(split_by_sg=True)

    aa = 1
"""


batch = collate_data_list(opt_outs)
batch.plot_batch_cell_params(space='real', quantiles=[0.1, 0.5])

batch.plot_batch_density_funnel()


"""

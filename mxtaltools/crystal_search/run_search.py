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
from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.crystal_search.utils import get_initial_state, init_samples_to_optim, parse_args
from mxtaltools.dataset_utils.utils import collate_data_list


def init_score_model(opt_config, device, config):
    # load up score model if it's needed
    score_model = load_crystal_score_model(config.score_model_checkpoint, device).to(device)
    score_model.eval()
    opt_config['score_model'] = score_model


if __name__ == '__main__':
    args = parse_args()  # call config with "python run_search.py --config /path/to/config.yaml
    source_dir = Path(__file__).resolve().parent.parent.parent
    if args.config is None:
        config_path = source_dir / 'configs' / 'crystal_searches' / 'base.yaml'
        #config_path = source_dir / 'configs' / 'crystal_searches' / 'acridine.yaml'
    else:
        config_path = Path(args.config)

    config = dict2namespace(load_yaml(config_path))

    device = config.device
    if device == 'cuda':
        # prevents from dipping into windows virtual vram which is super slow
        torch.cuda.set_per_process_memory_fraction(0.99, device=0)

    samples_to_optim = init_samples_to_optim(config)

    out_path = Path(config.out_dir + f"/{config.run_name}.pt")  # where to save outputs
    num_samples = len(samples_to_optim)
    print(f"Starting optimization of {num_samples} crystal samples")

    opt_outs = []
    num_opts = len(config.opt)
    batch_idx = 0
    finished = False
    cursor = 0
    pbar = tqdm(total=num_samples, unit="samples")
    prev_best_samples = None
    while not finished:
        try:
            crystal_batch = collate_data_list(samples_to_optim[cursor:cursor + config.batch_size]).to(device)

            if prev_best_samples is None:
                crystal_batch = get_initial_state(config, crystal_batch, device)
            else:
                # if we oomed out in the last iter, recover the system state
                crystal_batch.set_cell_parameters(
                    prev_best_samples[:crystal_batch.num_graphs].to(crystal_batch.device)
                )

            for opt_ind, opt_config in enumerate(config.opt):
                # do optimization in N stages
                if opt_config['optim_target'] in ['rdf_score', 'classification_score']:
                    init_score_model(opt_config, device, config)

                'do opt'
                opt_out = crystal_batch.optimize_crystal_parameters(**opt_config)

                crystal_batch = collate_data_list(opt_out).to(device)

            crystal_batch.box_analysis()
            opt_outs.extend(crystal_batch.cpu().batch_to_list())
            torch.save(opt_outs, out_path)

            cursor += config.batch_size
            prev_best_samples = None
            pbar.update(min(config.batch_size, num_samples - cursor))  # safe final update
            if cursor >= len(samples_to_optim):
                finished = True
            else:
                if config.grow_batch_size:
                    config.batch_size = int(config.batch_size * 1.2) # keep pushing the batch size between sets
                    print(f"Boosting batch size to {config.batch_size}")

        except (RuntimeError, ValueError) as e:
            if is_cuda_oom(e):
                config.batch_size = max(int(config.batch_size * 0.9), 1)
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
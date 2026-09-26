"""
A script for loading a batch of molecules and optimizing them against a given property via torch autograd
"""
import gc
import json
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


def reject_retired_keys(config):
    """The latent-space umbrella repulsion is removed; a config still carrying its keys
    would otherwise run without it and read as if it had been applied."""
    found = ['umbrella_path'] if hasattr(config, 'umbrella_path') else []
    for i, opt in enumerate(config.opt):
        opt = opt if isinstance(opt, dict) else vars(opt)
        found += [f'opt[{i}].{k}' for k in opt if k.startswith('umbrella')]
    if found:
        raise ValueError(f"retired crystal-search keys (umbrella repulsion was removed): {found}")


def run_files(out_path):
    """Per-run side files next to the output. Named after the run so two searches sharing a working directory can
    never read (or delete) each other's -- the May acridine chunks carry copies across arms, consistent with one
    shared opt_intermediates.pt."""
    stem = str(out_path)[:-3] if str(out_path).endswith('.pt') else str(out_path)
    return stem + '_opt_intermediates.pt', stem + '_progress.json'


def discard_intermediates(path):
    """The intermediates file belongs to ONE batch position (cursor): a retry of that same cursor may resume from it,
    nothing else may. It is removed at the start of a run and after every completed batch. Before, a later batch
    whose retry followed a step-0 OOM (which writes no file) reloaded a stale file and relaxed an EARLIER batch's
    walkers -- 46-55% of the relaxations in the acr_wrap_sep26 random-start jobs repeated earlier starts."""
    if os.path.exists(path):
        os.remove(path)


def crystal_search(config):
    reject_retired_keys(config)
    device = config.device

    if device == 'cuda':
        # prevents from dipping into windows virtual vram which is super slow
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)

    if config.target_path is not None:
        target, config = process_target(config)
    else:
        target = None

    samples_to_optim = init_samples_to_optim(config, target=target)
    data_mode = config.init_sample_method == 'data'
    if data_mode:
        # outputs carry the index of the seed they came from: the enforce_reduced filter drops rows, so position in
        # the output list is not the seed's position
        for i, sample in enumerate(samples_to_optim):
            sample.dataset_index = torch.tensor([i], dtype=torch.long)

    out_path = Path(config.out_dir + f"/{config.run_name}.pt")  # where to save outputs
    intermediates, progress = run_files(out_path)
    num_samples = len(samples_to_optim)
    print(f"Starting optimization of {num_samples} crystal samples")

    if os.path.exists(out_path) and not config.force_restart_run:
        opt_outs = torch.load(out_path, weights_only=False)
        if os.path.exists(progress):
            # the cursor, not len(opt_outs): enforce_reduced drops rows, and a cursor recomputed from the output
            # length re-relaxed completed samples; batch_idx continues so random starts are never redrawn
            with open(progress) as fh:
                state = json.load(fh)
            cursor, batch_idx = int(state['cursor']), int(state['batch_idx'])
        else:  # output written before progress files existed
            cursor = len(opt_outs)
            batch_idx = (cursor // config.batch_size) - 1  # so batch_idx+=1 lands on the right value
        if cursor >= num_samples:
            print(f"{config.run_name}: all {num_samples} samples already done")
            return opt_outs
    else:
        opt_outs = []
        cursor = 0
        batch_idx = -1

    num_opts = len(config.opt)
    finished = False
    pbar = tqdm(total=num_samples, unit="samples")
    prev_best_samples = None
    discard_intermediates(intermediates)  # a file left by an earlier (crashed) attempt of this run is not current

    while not finished:
        crystal_batch, opt_config = None, {}
        try:
            batch_idx += 1
            crystal_batch = collate_data_list(samples_to_optim[cursor:cursor + config.batch_size]).to(device)

            if (prev_best_samples is None) or (
                    prev_best_samples is not None and len(prev_best_samples) < crystal_batch.num_graphs):
                crystal_batch = get_initial_state(config, crystal_batch, device, batch_idx)
            else:
                crystal_batch = recover_opt_state(crystal_batch, config, device, batch_idx, prev_best_samples)

            for opt_ind, opt_config in enumerate(config.opt):
                # do optimization in N stages
                opt_config = parse_opt_config(opt_config, config, device, target)
                opt_config['intermediates_path'] = intermediates

                'do opt'
                opt_out, opt_record = crystal_batch.optimize_crystal_parameters(return_record=True, **opt_config)

                if config.save_trajs:
                    opt_record.update({'base_crystal': samples_to_optim[0]})
                    torch.save(opt_record, Path(str(out_path).replace('.pt', f'_traj{batch_idx}_{opt_ind}.pt')))

                if 'predictor' in opt_config.keys():
                    del opt_config['predictor']
                if 'score_model' in opt_config.keys():
                    del opt_config['score_model']

                if len(opt_out) == 0:  # the enforce_reduced filter dropped every row of this batch
                    crystal_batch = None
                    break
                crystal_batch = collate_data_list(opt_out).to(device)

            if crystal_batch is not None:
                crystal_batch.box_analysis()
                opt_outs.extend(crystal_batch.cpu().detach().batch_to_list())

            torch.save(opt_outs, out_path)

            cursor += config.batch_size
            with open(progress, 'w') as fh:
                json.dump({'cursor': cursor, 'batch_idx': batch_idx, 'num_samples': num_samples,
                           'n_out': len(opt_outs)}, fh)
            prev_best_samples = None
            discard_intermediates(intermediates)  # this cursor is done; its saved states must not seed the next one
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
                del crystal_batch  # may be None: the OOM can hit while the batch is still being built
                if 'predictor' in opt_config.keys():
                    del opt_config['predictor']
                if 'score_model' in opt_config.keys():
                    del opt_config['score_model']
                print(f"OOM error: dropping batch size to {config.batch_size}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                sleep(0.1)
                # only a file written by an attempt at THIS cursor can exist here (discard_intermediates). Seeded
                # (data-mode) runs never resume from it: after a stage drops rows, its row order no longer matches
                # the cursor's seeds, so a seed slot would resume another seed's state -- redo the seeds instead.
                if os.path.exists(intermediates) and not data_mode:
                    prev_best_samples = torch.load(intermediates, weights_only=False)
            else:
                raise e


    return opt_outs

    print(f"Sampling complete! Optimized a total of {len(opt_outs)} crystal samples.")

    # batch = collate_data_list(opt_outs)
    # batch.plot_batch_cell_params(space='real', quantiles=[0.1, 0.5], split_by_sg=True)
    #
    # batch.plot_batch_density_funnel(split_by_sg=True)

    aa = 1

if __name__ == '__main__':
    args = parse_args()  # call config with "python run_search.py --config /path/to/config.yaml
    source_dir = Path(__file__).resolve().parent.parent.parent
    if args.config is None:
        config_path = source_dir / 'configs' / 'crystal_searches' / 'base.yaml'
    else:
        config_path = Path(args.config)

    config = dict2namespace(load_yaml(config_path))

    crystal_search(config)

"""


batch = collate_data_list(opt_outs)
batch.plot_batch_cell_params(space='real', quantiles=[0.1, 0.5])

batch.plot_batch_density_funnel()


"""

"""
A script for loading a batch of molecules and optimizing them against a given property via torch autograd
"""
import argparse
import gc
import os
from pathlib import Path
from random import shuffle
from time import sleep

import numpy as np
import torch
from tqdm import tqdm


from mxtaltools.common.config_processing import load_yaml, dict2namespace
from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list


def get_initial_state(config, crystal_batch):
    # sample initial parameters
    if config.init_target_cp == 'std':
        target_cp = (torch.randn(crystal_batch.num_graphs, device=device) * 0.0447 + 0.6226).clip(min=0.45, max=0.95)
    elif config.init_target_cp is not None:
        target_cp = config.init_target_cp
    else:
        target_cp = None

    if config.init_sample_method == 'reasonable':
        crystal_batch.sample_reasonable_random_parameters(
            target_packing_coeff=target_cp,
            tolerance=5,
            max_attempts=50,
            sample_niggli=config.init_sample_reduced,
            seed=config.opt_seed,
        )
    elif config.init_sample_method == 'random':
        if config.init_sample_reduced:
            crystal_batch.sample_random_reduced_crystal_parameters(
                target_packing_coeff=target_cp,
                seed=config.opt_seed,
            )
        else:
            crystal_batch.sample_random_crystal_parameters(
                target_packing_coeff=target_cp,
                seed=config.opt_seed,
            )
    else:
        assert False
    return crystal_batch

def init_samples_to_optim(config):
    """
    Load and select molecules to optimize
    """
    mol_list = torch.load(config.mol_path, weights_only=False)
    if not isinstance(mol_list, list):
        mol_list = [mol_list]
    if config.sampling_mode == 'all':
        mols_to_optim = mol_list
    elif config.sampling_mode == 'random':
        rng = np.random.RandomState(config.mol_seed)
        inds = rng.randint(0, len(mol_list), config.mols_to_sample)
        mols_to_optim = [mol_list[ind] for ind in inds]
    else:
        assert False, "Sampling mode must be 'all' or 'random"
    """
    Initialize full set of crystals to optimize
    """
    max_zp = max(config.zp_to_search)
    samples_to_optim = []
    ones3 = torch.ones(3, device='cpu')
    ones1 = torch.ones(1, device='cpu')
    print("Initializing crystals to optimize")
    for mol in mols_to_optim:
        for sg in config.sgs_to_search:
            for s_ind in range(config.num_samples):
                for zp in config.zp_to_search:
                    opt_sample = MolCrystalData(
                        molecule=[mol.clone() for _ in range(zp)] if zp > 1 else mol.clone(),  # duplicate molecules here
                        sg_ind=sg,
                        aunit_handedness=ones1,
                        cell_lengths=ones3,
                        cell_angles=ones3,
                        aunit_centroid=ones3,
                        aunit_orientation=ones3,
                        skip_box_analysis=True,
                        max_z_prime=max_zp,
                        z_prime=zp,
                        do_box_analysis=True,  # need this just to instantiate the tensors
                    )
                    samples_to_optim.append(opt_sample)
    shuffle(samples_to_optim)
    return samples_to_optim

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize a batch of molecules against a given property via torch autograd"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If not provided, defaults to configs/crystal_searches/base.yaml",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()  # call config with "python run_search.py --config /path/to/config.yaml
    source_dir = Path(__file__).resolve().parent.parent.parent
    if args.config is None:
        #config_path = source_dir / 'configs' / 'crystal_searches' / 'base.yaml'
        config_path = source_dir / 'configs' / 'crystal_searches' / 'acridine.yaml'
    else:
        config_path = Path(args.config)

    config = dict2namespace(load_yaml(config_path))

    device = config.device
    if device == 'cuda':
        # prevents from dipping into windows virtual vram
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
                crystal_batch = get_initial_state(config, crystal_batch)
            else:
                # if we oomed out in the last iter, recover the system state
                crystal_batch.set_cell_parameters(
                    prev_best_samples[:crystal_batch.num_graphs].to(crystal_batch.device)
                )

            for opt_ind, opt_config in enumerate(config.opt):
                # do optimization in N stages
                if opt_config['optim_target'] in ['rdf_score', 'classification_score']:
                    # load up score model if it's needed
                    score_model = load_crystal_score_model(config.score_model_checkpoint, device).to(device)
                    score_model.eval()
                    opt_config['score_model'] = score_model
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
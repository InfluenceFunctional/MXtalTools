"""Generate one crystal-search config per molecule chunk, for parallel anchor generation.

Each chunk file produced by prep_qm9_anchor_mols.py --chunks holds a DISJOINT set of
standardized molecules, so the jobs partition the work with no duplicate structures and no
coordination between them. Every config uses sampling_mode 'all' -- the chunk file already
IS this job's molecule set, so run_search does no sampling of its own.

Settings are the house ELJ recipe (two rprop stages, compress then relax), copied from
mipcas_elj_*.yaml, which is already SG2/Z'=1.

Usage:
    python make_chunks.py --chunks 20 --num-samples 50 \\
        --mol-dir D:\\crystal_datasets\\conditional\\priors \\
        --mol-stem qm9_cluster_mols --out-dir D:\\crystal_datasets\\conditional\\anchors
"""
import argparse
from pathlib import Path

import yaml


def opt_stage(target, compression, lr, steps):
    return {
        'optim_target': target,
        'enforce_reduced': True,
        'compression_factor': compression,
        'cutoff': 10,
        'init_lr': lr,
        'convergence_eps': 0.00001,
        'optimizer_func': 'rprop',
        'anneal_lr': False,
        'grad_norm_clip': 0.1,
        'show_tqdm': False,          # per-chunk logs; a progress bar per job is noise
        'max_num_steps': steps,
        'target_packing_coeff': None,
        'umbrella': False,
    }


def build(args, k):
    # as_posix, not str: a cluster config set is generated on Windows, and a Path join there
    # turns /scratch/... into backslashes that no longer resolve on the target machine
    mol_path = (Path(args.mol_dir) / f"{args.mol_stem}_chunk{k}.pt").as_posix()
    return {
        'device': 'cuda',
        'mol_path': mol_path,
        'dataset_path': None,
        'target_path': None,
        'target_identifier': None,
        'umbrella_path': None,
        'out_dir': Path(args.out_dir).as_posix(),
        'run_name': f"{args.tag}_chunk{k}",
        'save_trajs': False,
        'uma_predictor_path': None,
        'mace_predictor_path': None,
        'score_model_checkpoint': None,
        'force_restart_run': False,   # resume a partially finished chunk from its cursor
        'init_sample_method': 'random',
        'init_reduced': True,
        'init_sample_reduced': True,
        'init_target_cp': 'wide',
        'mol_seed': 0,
        'opt_seed': k,                # distinct random inits per chunk
        'sampling_mode': 'all',
        'mols_to_sample': 0,          # unused under 'all'
        'num_samples': args.num_samples,
        'sgs_to_search': [args.sg],
        'zp_to_search': [args.zp],
        'batch_size': args.batch_size,
        'grow_batch_size': False,
        'opt': [
            opt_stage('elj', 1.0, 0.05, 450),
            opt_stage('elj', 0.0, 0.01, 50),
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--chunks', type=int, required=True)
    p.add_argument('--num-samples', type=int, default=50,
                   help='random inits per molecule; the bk11 verdict is that depth beyond '
                        'an estimable group buys nothing, and 50 is what the local set used')
    p.add_argument('--mol-dir', required=True)
    p.add_argument('--mol-stem', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--tag', default='qm9cluster')
    p.add_argument('--sg', type=int, default=2)
    p.add_argument('--zp', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1000,
                   help='structures per GPU batch; 1000 peaked at 8.4 GB on a 16 GB card')
    p.add_argument('--dest', type=Path, default=Path(__file__).parent)
    args = p.parse_args()

    written = []
    for k in range(args.chunks):
        cfg = build(args, k)
        path = args.dest / f"{args.tag}_chunk{k}.yaml"
        with path.open('w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
        written.append(path)

    # every chunk must read a DIFFERENT molecule file and write a DIFFERENT output, or
    # jobs silently overwrite each other's results
    mols = {yaml.safe_load(p.read_text(encoding='utf-8'))['mol_path'] for p in written}
    runs = {yaml.safe_load(p.read_text(encoding='utf-8'))['run_name'] for p in written}
    assert len(mols) == len(written), 'chunk configs share a mol_path'
    assert len(runs) == len(written), 'chunk configs share a run_name'

    print(f"wrote {len(written)} configs to {args.dest}")
    print(f"  molecules: {args.mol_stem}_chunk[0..{args.chunks - 1}].pt")
    print(f"  each: {args.num_samples} inits/molecule, SG{args.sg} Z'={args.zp}, "
          f"batch {args.batch_size}, elj two-stage rprop")
    print(f"  outputs: {args.out_dir}/{args.tag}_chunk[k].pt")
    print()
    print("run one chunk with:")
    print(f"  python mxtaltools/crystal_search/run_search.py --config "
          f"{written[0]}")


if __name__ == '__main__':
    main()

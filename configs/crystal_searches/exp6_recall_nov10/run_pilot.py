"""exp-6 recall test, stage B pilot: one molecule, one kind of job (README.md).

    python run_pilot.py NAME reference      the UMA-s 1.1 random-start search (configs/NAME_uma_search.yaml)
    python run_pilot.py NAME cheap          for exp6, then elj: the random-start search (configs/NAME_<e>_search.yaml),
                                            the PICKS lowest distinct minima of it (by that energy) written to
                                            <out_dir>/NAME_<e>_picks.pt, and UMA-s 1.1 relaxing them as seeds
                                            (configs/NAME_uma_seeds_from_<e>.yaml)

Every step is bounded: a search stops at num_samples random starts (or sooner by its Good-Turing stop rule), a
seeded run relaxes at most PICKS seeds. A step whose outputs exist is resumed by run_search (force_restart_run
false), and a picks file that exists is kept.

--smoke DIR --local-panel DIR --uma PATH: a tiny local run of the same chain (6 UMA-s starts, 24 cheap starts, 4 picks,
4 slots, GPU capped at --gpu-fraction) writing to DIR, for checking the chain before the cluster job.
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PICKS = 300  # the cheap arm's MLIP budget per energy (docs/design/exp6_recall_test.md in gfn_diffusion)


def load(name, smoke):
    from mxtaltools.crystal_search.run_search import load_yaml
    c = load_yaml(os.path.join(HERE, 'configs', name + '.yaml'))
    if smoke is not None:
        c['out_dir'] = smoke['out']
        c['mol_path'] = os.path.join(smoke['panel'], os.path.basename(c['mol_path']))
        if c.get('uma_predictor_path'):
            c['uma_predictor_path'] = smoke['uma']
        uma = any(str(o['optim_target']).lower() == 'uma' for o in c['opt'])
        if c['num_samples']:
            c['num_samples'] = 6 if uma else 24
        c['batch_size'] = c['stream']['slots'] = 4 if uma else 24
        if c['stream'].get('seeds'):
            c['stream']['seeds'] = os.path.join(smoke['out'], os.path.basename(c['stream']['seeds']))
    return c


def run(name, smoke):
    from mxtaltools.crystal_search import run_search
    c = load(name, smoke)
    t0 = time.time()
    print(f'STEP {name} start {time.strftime("%Y-%m-%dT%H:%M:%S")}', flush=True)
    run_search.crystal_search(run_search.dict2namespace(c))
    print(f'STEP {name} done {time.time() - t0:.0f} s', flush=True)
    return c


def pick(c, path, n):
    """The n lowest-energy rows of a search's one-row-per-minimum file, as a list, written to path."""
    if os.path.exists(path):
        print(f'picks {path} exist; kept', flush=True)
        return
    minima = torch.load(os.path.join(c['out_dir'], c['run_name'] + '_minima.pt'), weights_only=False)
    minima = list(minima)
    if not minima:
        raise RuntimeError(f"{c['run_name']}: no minima to pick from")
    minima.sort(key=lambda r: float(r.minimum_f))
    keep = minima[:n]
    torch.save(keep, path + '.tmp')
    os.replace(path + '.tmp', path)
    print(f"picks {path}: {len(keep)} of {len(minima)} minima, E/kT {float(keep[0].minimum_f):.3f} to "
          f"{float(keep[-1].minimum_f):.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('kind', choices=['reference', 'cheap'])
    p.add_argument('--smoke', default=None)
    p.add_argument('--local-panel', default=None)
    p.add_argument('--uma', default=None)
    p.add_argument('--gpu-fraction', type=float, default=None)
    a = p.parse_args()
    if a.gpu_fraction is not None:
        orig = torch.cuda.set_per_process_memory_fraction
        torch.cuda.set_per_process_memory_fraction = lambda f, device=None: orig(a.gpu_fraction, device=0)
    smoke = None if a.smoke is None else dict(out=a.smoke, panel=a.local_panel, uma=a.uma)
    if smoke is not None:
        os.makedirs(a.smoke, exist_ok=True)
    if a.kind == 'reference':
        run(f'{a.name}_uma_search', smoke)
        return
    for e in ('exp6', 'elj'):
        c = run(f'{a.name}_{e}_search', smoke)
        pick(c, os.path.join(c['out_dir'], f'{a.name}_{e}_picks.pt'), 4 if smoke else PICKS)
        run(f'{a.name}_uma_seeds_from_{e}', smoke)


if __name__ == '__main__':
    sys.exit(main())

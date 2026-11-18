"""acr_mlip_nov17: one model's chain (README.md).

    python run.py MODEL                     checks, seed files, then <MODEL>_known and <MODEL>_search (configs/)
    python run.py MODEL --smoke DIR         a tiny local run of the same chain on the dev-box files (models.py 'local',
                                            D:/crystal_datasets/acridine): 4 pool seeds, 4 random starts, 4 slots,
                                            stages capped at 40 steps; on the CPU when CUDA is hidden
                                            (CUDA_VISIBLE_DEVICES=-1), else GPU memory capped at --gpu-fraction;
                                            outputs and seed files in DIR

Checks before any search: the conformer's SHA1 is the one models.py names, the model file exists, and for a model with
a pinned fairchem the imported version is it. The seed files are built once (run.py's own atomic write; an existing
file whose manifest names the same inputs is kept): each seed's cell parameters and handedness put on the model's
conformer (coordinator.rebuild_crystals), the seed's origin in seeds_<MODEL>_<set>.json (index j = start index -(j + 1)).
Each search resumes from its committed outputs when resubmitted (force_restart_run false).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as M  # noqa: E402

SMOKE = dict(n_pool=4, n_random=4, slots=4, max_steps=40)


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def checks(model, mol_path, model_path):
    m = M.MODELS[model]
    assert sha1(mol_path) == m['mol_sha1'], f'{mol_path} is not {model}\'s conformer (SHA1 {sha1(mol_path)})'
    assert os.path.exists(model_path), f'no model file {model_path}'
    if m['fairchem']:
        from importlib.metadata import version
        assert version('fairchem-core') == m['fairchem'], \
            f"fairchem-core {version('fairchem-core')} imported, {m['fairchem']} needed (PYTHONPATH first: {m['pyextra']})"
    print(f'checks ok: {model} conformer {m["mol_sha1"][:12]}, model {model_path}', flush=True)


def build_seeds(model, which, mol_path, out_dir, n=None):
    """seeds_<model>_<which>.pt in out_dir (which: 'known' = ACRDIN04, ACRDIN12; 'pool' = models.SEED_POOL's first n
    rows, all when n is None); kept when its manifest names the same inputs. Returns its path."""
    from mxtaltools.crystal_search import coordinator as co
    path = os.path.join(out_dir, f'seeds_{model}_{which}.pt')
    src_file = M.KNOWN if which == 'known' else M.SEED_POOL
    inputs = dict(source=sha1(src_file), mol=sha1(mol_path), n=n)
    manifest = path[:-3] + '.json'
    if os.path.exists(path) and os.path.exists(manifest) and json.load(open(manifest))['inputs'] == inputs:
        print(f'seeds {path} exist; kept', flush=True)
        return path
    src = torch.load(src_file, weights_only=False)
    params, hand = src['params'].float(), src['hand'].float()
    origin = list(src['identifier']) if which == 'known' else [f'{s}#{i}' for i, s in enumerate(src['source'])]
    if n is not None:
        params, hand, origin = params[:n], hand[:n], origin[:n]
    t0 = time.time()
    rows = co.rebuild_crystals(SimpleNamespace(mol_path=mol_path, sg=M.SG, z_prime=1), params, hand)
    torch.save(rows, path + '.tmp')
    os.replace(path + '.tmp', path)
    with open(manifest + '.tmp', 'w') as fh:
        json.dump(dict(inputs=inputs, origin=origin, note='origin[j] is the seed of start index -(j + 1)'), fh)
    os.replace(manifest + '.tmp', manifest)
    print(f'seeds {path}: {len(rows)} rows ({time.time() - t0:.0f} s)', flush=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', choices=list(M.MODELS))
    p.add_argument('--smoke', default=None, help='output folder of a tiny local run')
    p.add_argument('--gpu-fraction', type=float, default=0.4)
    a = p.parse_args()
    m = M.MODELS[a.model]
    smoke = a.smoke is not None
    out_dir = a.smoke if smoke else M.OUT
    mol_path = f"D:/crystal_datasets/acridine/{m['mol']}" if smoke else f"{M.DATA}/{m['mol']}"
    model_path = m['local'] if smoke else m['path']
    os.makedirs(out_dir, exist_ok=True)
    if smoke and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    checks(a.model, mol_path, model_path)
    seeds = dict(known=build_seeds(a.model, 'known', mol_path, out_dir),
                 pool=build_seeds(a.model, 'pool', mol_path, out_dir, n=SMOKE['n_pool'] if smoke else None))

    from mxtaltools.crystal_search import run_search
    for kind in ('known', 'search'):
        run = f'{a.model}_{kind}'
        c = yaml.safe_load(open(os.path.join(M.HERE, 'configs', run + '.yaml')))
        if smoke:
            c.update(out_dir=out_dir, mol_path=mol_path, batch_size=SMOKE['slots'],
                     num_samples=min(c['num_samples'], SMOKE['n_random']),
                     device='cuda' if torch.cuda.is_available() else 'cpu')
            for st in c['opt']:  # the chain, not the physics: short descents
                st['max_num_steps'] = min(int(st['max_num_steps']), SMOKE['max_steps'])
            c['stream'].update(slots=SMOKE['slots'], seeds=seeds['known' if kind == 'known' else 'pool'])
            c['mace_predictor_path' if m['kind'] == 'mace' else 'uma_predictor_path'] = model_path
        t0 = time.time()
        print(f'STEP {run} start {time.strftime("%Y-%m-%dT%H:%M:%S")}', flush=True)
        run_search.crystal_search(run_search.dict2namespace(c))
        print(f'STEP {run} done {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()

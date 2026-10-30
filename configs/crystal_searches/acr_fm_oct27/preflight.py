"""acr_fm_oct27 preflight of one model's campaign (CPU job, submit_preflight.sbatch; the stream arrays start only after it
succeeds). acr_zp1_sep30's preflight.py, per model.

    python preflight.py <campaign directory> MODEL

Checks, in order, and exits non-zero at the first failure:
  1. imports the jobs need (mxtaltools, spglib, and mace or fairchem at the version campaign.MODELS asks for);
  2. the molecule file is the model's conformer (campaign.MODELS[MODEL]['mol_sha1']) and the checkpoint on the cluster
     has the campaign's energy_model_id (the frozen coord.yaml);
  3. seeds: seed_pool.pt and ../acr_zp1_sep30/known_forms.pt, rebuilt from their cell parameters with the model's
     conformer (coordinator.rebuild_crystals), dealt round-robin into seeds/seeds_<k>.pt, k < campaign.TASKS[MODEL]
     ['seeded']; skipped when seeds/manifest.json already records the same inputs;
  4. the coordinator's per-row path on 64 seeds: standardize_cells (spglib) and atomwise RDFs;
  5. one crystal energy of 2 seeds under the model, on CPU (finite);
  6. host memory per start (RSS growth while task 0's seeds are held, per seed).
On success writes <campaign>/preflight_ok (JSON: the checks' numbers); launch.sh submits the arrays after this job.
"""
import hashlib
import json
import os
import sys
import time
from argparse import Namespace
from types import SimpleNamespace

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign as C  # noqa: E402


def rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # peak, Linux: KiB
    except ImportError:  # a local (Windows) dry run
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(camp, model):
    t0 = time.time()
    out = {}
    m = C.MODELS[model]
    n_tasks = C.TASKS[model]['seeded']
    import spglib  # noqa: F401
    from importlib.metadata import version
    if m['kind'] == 'mace':
        import mace  # noqa: F401
    else:
        import fairchem.core  # noqa: F401
        assert not m['fairchem'] or version('fairchem-core') == m['fairchem'], \
            f"fairchem-core {version('fairchem-core')} imported, {m['fairchem']} needed (PYEXTRA {m['pyextra']})"
    from mxtaltools.crystal_search import coordinator as co
    from mxtaltools.crystal_search.standardize import standardize_cells
    from mxtaltools.crystal_search.utils import parse_opt_config
    from mxtaltools.dataset_utils.utils import collate_data_list
    out['versions'] = {p: version(p) for p in ('torch', 'fairchem-core', 'mace-torch')}
    print(f"1. imports ok ({out['versions']})", flush=True)

    coord = yaml.safe_load(open(os.path.join(camp, 'coord.yaml')))
    seeded = yaml.safe_load(open(os.path.join(camp, 'streams', 'seeded.yaml')))
    assert coord['mol_path'] == C.mol_path(model) and sha1(coord['mol_path']) == m['mol_sha1'], \
        f"{coord['mol_path']} is not {model}'s conformer"
    paths = Namespace(mace_predictor_path=seeded['mace_predictor_path'], uma_predictor_path=seeded['uma_predictor_path'])
    mid = co.energy_model_id({'optim_target': m['kind']}, paths)
    assert mid == coord['energy_model_id'], f'cluster model {mid} != campaign {coord["energy_model_id"]}'
    print(f'2. molecule and model ok ({mid})', flush=True)

    cfg = SimpleNamespace(mol_path=coord['mol_path'], sg=14, z_prime=1)
    sdir = os.path.join(camp, 'seeds')
    os.makedirs(sdir, exist_ok=True)
    manifest = os.path.join(sdir, 'manifest.json')
    inputs = dict(pool=sha1(os.path.join(HERE, 'seed_pool.pt')), known=sha1(str(C.KNOWN)), mol=m['mol_sha1'],
                  n_tasks=n_tasks)
    rss0 = rss_gb()
    if os.path.exists(manifest) and json.load(open(manifest)).get('inputs') == inputs and \
            all(os.path.exists(os.path.join(sdir, f'seeds_{k}.pt')) for k in range(n_tasks)):
        seeds = torch.load(os.path.join(sdir, 'seeds_0.pt'), weights_only=False)
        print(f'3. seeds already built ({json.load(open(manifest))["counts"]})', flush=True)
    else:
        pool = torch.load(os.path.join(HERE, 'seed_pool.pt'), weights_only=False)
        known = torch.load(str(C.KNOWN), weights_only=False)
        n_prior = len(pool['params'])
        params = torch.cat([pool['params'], known['params'].float()])
        hand = torch.cat([pool['hand'], known['hand'].float()])
        counts = []
        for k in range(n_tasks):  # one task's share at a time: the whole list held at once is ~2x the template's
            idx = torch.arange(k, len(params), n_tasks)
            part = co.rebuild_crystals(cfg, params[idx], hand[idx])
            for i, c in zip(idx.tolist(), part):  # provenance: the pool row's source and index, or a known form's name
                c.identifier = f"{pool['source'][i]}#{i}" if i < n_prior else known['identifier'][i - n_prior]
            if k == 0:
                seeds = part
            tmp = os.path.join(sdir, f'.seeds_{k}.pt.tmp')
            torch.save(part, tmp)
            os.replace(tmp, os.path.join(sdir, f'seeds_{k}.pt'))
            counts.append(len(part))
        json.dump(dict(inputs=inputs, counts=counts, n_prior=n_prior, known=known['identifier']), open(manifest, 'w'))
        print(f'3. seeds built: {n_prior} pooled states + {len(known["identifier"])} known forms -> {counts}',
              flush=True)
    out['seed_counts'] = json.load(open(manifest))['counts']
    out['host_gb_per_1000_seeds'] = 1000 * max(rss_gb() - rss0, 0.0) / max(len(seeds), 1)

    probe = [c.clone() for c in seeds[:64]]
    std, info = standardize_cells(collate_data_list(probe), on_failure='flag')
    assert bool(info['ok'].all()), f'{int((~info["ok"]).sum())} of 64 seeds had no reduced cell'
    r = co.compute_rdfs(std.batch_to_list(), 32, coord['rdf_mode'])
    assert torch.isfinite(r).all()
    print(f'4. standardize + {coord["rdf_mode"]} RDF ok on 64 seeds', flush=True)

    pred = parse_opt_config(dict(optim_target=m['kind']), paths, 'cpu', None)['predictor']
    b = collate_data_list([c.clone() for c in seeds[:2]])
    with torch.no_grad():
        e = (b.compute_crystal_mace(pred, std_orientation=True) if m['kind'] == 'mace'
             else b.compute_crystal_uma(pred, std_orientation=True))
    assert torch.isfinite(e).all(), e
    out['crystal_eV'] = [float(x) for x in e.flatten()]
    print(f'5. {model} crystal energies {out["crystal_eV"]} eV', flush=True)
    print(f'6. host memory: {out["host_gb_per_1000_seeds"]:.3f} GB per 1000 held seeds (peak RSS {rss_gb():.2f} GB)')
    out.update(time_s=time.time() - t0, peak_rss_gb=rss_gb(), inputs=inputs)
    json.dump(out, open(os.path.join(camp, 'preflight_ok'), 'w'), indent=1)
    print('preflight ok', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

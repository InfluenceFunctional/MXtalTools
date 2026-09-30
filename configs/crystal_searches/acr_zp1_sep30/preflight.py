"""acr_zp1_sep30 preflight (CPU job, submit_preflight.sbatch; the stream arrays start only after it succeeds).

    python preflight.py <campaign directory>

Checks, in order, and exits non-zero at the first failure:
  1. imports the jobs need (mxtaltools, mace, spglib) inside the container;
  2. the molecule file is the universal conformer (MOL_SHA1) and the MACE file on the cluster has the campaign's
     energy_model_id (the frozen coord.yaml);
  3. seeds: seed_pool.pt (earlier sg14 Z'=1 search end states, filtered and thinned locally) and known_forms.pt, rebuilt
     from their cell parameters with the campaign molecule (coordinator.rebuild_crystals), dealt round-robin into
     seeds/seeds_<k>.pt, k < N_TASKS (the seeded stream's array tasks); skipped when seeds/manifest.json already records
     the same inputs;
  4. the coordinator's per-row path on 64 seeds: standardize_cells (spglib) and atomwise RDFs;
  5. one MACE crystal energy of 2 seeds on CPU (finite);
  6. host memory per start (RSS growth while the seeds are held, per seed).
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
MOL_SHA1 = 'a112cf197b9ca06f0857eded426277c7db5e7dba'  # acr_newmodel_conformer.pt as written 2026-09-30
N_TASKS = 8  # make_campaign.TASKS['seeded']


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


def main(camp):
    t0 = time.time()
    out = {}
    import mace  # noqa: F401
    import spglib  # noqa: F401
    from mxtaltools.crystal_search import coordinator as co
    from mxtaltools.crystal_search.standardize import standardize_cells
    from mxtaltools.crystal_search.utils import parse_opt_config
    from mxtaltools.dataset_utils.utils import collate_data_list
    print('1. imports ok', flush=True)

    coord = yaml.safe_load(open(os.path.join(camp, 'coord.yaml')))
    seeded = yaml.safe_load(open(os.path.join(camp, 'streams', 'seeded.yaml')))
    assert sha1(coord['mol_path']) == MOL_SHA1, f"{coord['mol_path']} is not the universal conformer"
    mid = co.energy_model_id({'optim_target': 'mace'}, Namespace(mace_predictor_path=seeded['mace_predictor_path']))
    assert mid == coord['energy_model_id'], f'cluster model {mid} != campaign {coord["energy_model_id"]}'
    print(f'2. molecule and model ok ({mid})', flush=True)

    cfg = SimpleNamespace(mol_path=coord['mol_path'], sg=14, z_prime=1)
    sdir = os.path.join(camp, 'seeds')
    os.makedirs(sdir, exist_ok=True)
    manifest = os.path.join(sdir, 'manifest.json')
    inputs = dict(pool=sha1(os.path.join(HERE, 'seed_pool.pt')), known=sha1(os.path.join(HERE, 'known_forms.pt')),
                  mol=MOL_SHA1, n_tasks=N_TASKS)
    rss0 = rss_gb()
    if os.path.exists(manifest) and json.load(open(manifest)).get('inputs') == inputs and \
            all(os.path.exists(os.path.join(sdir, f'seeds_{k}.pt')) for k in range(N_TASKS)):
        seeds = torch.load(os.path.join(sdir, 'seeds_0.pt'), weights_only=False)
        print(f'3. seeds already built ({json.load(open(manifest))["counts"]})', flush=True)
    else:
        pool = torch.load(os.path.join(HERE, 'seed_pool.pt'), weights_only=False)
        known = torch.load(os.path.join(HERE, 'known_forms.pt'), weights_only=False)
        n_prior = len(pool['params'])
        params = torch.cat([pool['params'], known['params']])
        hand = torch.cat([pool['hand'], known['hand']])
        seeds = co.rebuild_crystals(cfg, params, hand)
        for i, c in enumerate(seeds):  # provenance: the pool row's source and index, or a known form's name
            c.identifier = f"{pool['source'][i]}#{i}" if i < n_prior else known['identifier'][i - n_prior]
        counts = []
        for k in range(N_TASKS):
            part = seeds[k::N_TASKS]
            tmp = os.path.join(sdir, f'.seeds_{k}.pt.tmp')
            torch.save(part, tmp)
            os.replace(tmp, os.path.join(sdir, f'seeds_{k}.pt'))
            counts.append(len(part))
        json.dump(dict(inputs=inputs, counts=counts, n_prior=n_prior, known=known['identifier']), open(manifest, 'w'))
        print(f'3. seeds built: {n_prior} pooled search states + {len(known["identifier"])} known forms -> {counts}',
              flush=True)
    out['seed_counts'] = json.load(open(manifest))['counts']
    out['host_gb_per_1000_seeds'] = 1000 * max(rss_gb() - rss0, 0.0) / max(len(seeds), 1)

    probe = [c.clone() for c in seeds[:64]]
    std, info = standardize_cells(collate_data_list(probe), on_failure='flag')
    assert bool(info['ok'].all()), f'{int((~info["ok"]).sum())} of 64 seeds had no reduced cell'
    r = co.compute_rdfs(std.batch_to_list(), 32, coord['rdf_mode'])
    assert torch.isfinite(r).all()
    print(f'4. standardize + {coord["rdf_mode"]} RDF ok on 64 seeds', flush=True)

    pred = parse_opt_config(dict(optim_target='mace'), Namespace(mace_predictor_path=seeded['mace_predictor_path']),
                            'cpu', None)['predictor']
    b = collate_data_list([c.clone() for c in seeds[:2]])
    with torch.no_grad():
        e = b.compute_crystal_mace(pred, std_orientation=True)
    assert torch.isfinite(e).all(), e
    out['mace_crystal_eV'] = [float(x) for x in e.flatten()]
    print(f'5. MACE crystal energies {out["mace_crystal_eV"]} eV', flush=True)
    print(f'6. host memory: {out["host_gb_per_1000_seeds"]:.3f} GB per 1000 held seeds (peak RSS {rss_gb():.2f} GB)')
    out.update(time_s=time.time() - t0, peak_rss_gb=rss_gb(), inputs=inputs)
    json.dump(out, open(os.path.join(camp, 'preflight_ok'), 'w'), indent=1)
    print('preflight ok', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])

"""
Build the seed set for the `aug21seed` block: perturbed copies of known basins.

The seeded arms answer a question random starts cannot: are the missed forms
REACHABLE and HOLDABLE by this optimizer at all? Each seed is a known structure
displaced by a controlled amount in latent space. After the search runs, the
fraction that returns to its anchor, as a function of displacement, is the basin's
radius of attraction under the production schedule.

Read the outcome like this:

  returns from large displacement  -- a wide, robust basin. The previous search
                                      missing it is then genuinely surprising and
                                      points at coverage.
  returns only from tiny displacement -- narrow basin; random starts would rarely
                                      land it, and more samples is the fix.
  does NOT return even from zero displacement -- the optimizer cannot hold the
                                      structure it was handed. No amount of
                                      sampling would ever have reported it, and
                                      the schedule itself has to change.

Displacement uses `log_noise_latent_parameters`, the same operator
`new_calibrate_prior_noise` uses to calibrate the prior's thermal radius, so the
magnitudes are in the units the rest of the pipeline already speaks.

Anchors are the known forms for the target SG/Z' PLUS Nikos' accepted structures
there, since his are the proposals we most want to know the status of.

    python make_seeds.py --sg 14 --zp 2
"""
import argparse
import json
import os

import torch

from mxtaltools.dataset_utils.utils import collate_data_list

ROOT = r"D:\crystal_datasets\acridine"
#: zero is deliberate and is the most informative single row: it tests whether
#: the optimizer holds the structure with no displacement at all.
#: The displacement ladder IS the exploration -- it is what stops every seed
#: relaxing back to the same place, and it is why the seeded arms keep the
#: PRODUCTION optimizer. Heating the learning rate instead would confound "narrow
#: basin" with "optimizer too hot to settle", and would stop measuring the search
#: we are actually trying to explain. If everything returns even from the top
#: rung, extend the ladder rather than the learning rate.
LOG_NOISE = [None, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
COPIES_PER_LEVEL = 40


def anchors_for(sg, zp, nikos_path):
    """Known forms in this SG/Z', plus Nikos' accepted structures there."""
    out = []
    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    #: std_opt_, NOT std_ -- it carries opt_acridine_conformer, the same molecule
    #: the searches use. The std_ file carries an older conformer (aromatic C-C
    #: 1.3668 vs 1.4027 A) and would seed a different molecule entirely.
    for i, e in enumerate(poly.batch_to_list()):
        if (int(poly.sg_ind[i]), int(poly.z_prime[i])) == (sg, zp):
            out.append((poly.identifier[i], e))

    if nikos_path and os.path.exists(nikos_path):
        blob = torch.load(nikos_path, weights_only=False, map_location='cpu')
        lv = blob['l1'] if isinstance(blob, dict) and 'l1' in blob else None
        if lv is not None:
            for i, e in enumerate(lv.batch_to_list()):
                if (int(lv.sg_ind[i]), int(lv.z_prime[i])) == (sg, zp):
                    out.append((f'nikos_{lv.identifier[i]}', e))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sg', type=int, default=14)
    ap.add_argument('--zp', type=int, default=2)
    ap.add_argument('--nikos', default=os.path.join(
        ROOT, 'nikos_comparison', 'nikos_levels.pt'),
        help="his structures, already reprojected onto our conformer (L1)")
    ap.add_argument('--out', default=None)
    ap.add_argument('--shards', type=int, default=6,
                    help="split into one file per search arm. `init_samples_to_optim` "
                         "clips num_samples to the seed count and then indexes "
                         "arange(mol_seed*n, (mol_seed+1)*n), so a single file with "
                         "mol_seed>0 runs off the end -- sharding keeps mol_seed at 0.")
    cli = ap.parse_args()

    anchors = anchors_for(cli.sg, cli.zp, cli.nikos)
    if not anchors:
        raise SystemExit(f"no anchors for sg={cli.sg} Z'={cli.zp}")
    print(f"anchors for sg={cli.sg} Z'={cli.zp}:")
    for name, _ in anchors:
        print(f"   {name}")

    seeds, meta = [], []
    for name, crystal in anchors:
        for lvl in LOG_NOISE:
            n = 1 if lvl is None else COPIES_PER_LEVEL
            batch = collate_data_list([crystal.clone() for _ in range(n)],
                                      exclude_keys=['rdf', 'elj', 'mace'])
            batch.aunit_handedness = batch.aunit_handedness.abs()
            if lvl is not None:
                batch.log_noise_latent_parameters(lvl, lvl)
            for e in batch.batch_to_list():
                seeds.append(e)
                meta.append({'anchor': name, 'log_noise': lvl})

    stem = cli.out or os.path.join(ROOT, f'seeds_sg{cli.sg}_zp{cli.zp}')
    stem = stem[:-3] if stem.endswith('.pt') else stem
    torch.save(meta, f'{stem}_meta.pt')

    n = len(seeds)
    per = -(-n // cli.shards)          # ceil, so the shards cover everything
    print(f"\n{len(anchors)} anchors x "
          f"(1 undisplaced + {len(LOG_NOISE) - 1} levels x {COPIES_PER_LEVEL})"
          f" = {n} seeds -> {cli.shards} shards of <= {per}")
    written = 0
    for k in range(cli.shards):
        part = seeds[k * per:(k + 1) * per]
        if not part:
            print(f"   shard {k}: EMPTY -- reduce --shards")
            continue
        torch.save(part, f'{stem}_{k}.pt')
        written += len(part)
        print(f"   shard {k}: {len(part):4d} seeds -> "
              f"{os.path.basename(stem)}_{k}.pt")
    assert written == n, f"sharding lost seeds: {written} of {n}"
    #: sidecar so make_configs_acridine.py can count seeds without importing
    #: mxtaltools to unpickle the shards
    counts = [len(seeds[k * per:(k + 1) * per]) for k in range(cli.shards)]
    with open(f'{stem}_counts.json', 'w') as fh:
        json.dump(counts, fh)
    print(f"      shard sizes -> {os.path.basename(stem)}_counts.json")
    print(f"      provenance (anchor + displacement) -> "
          f"{os.path.basename(stem)}_meta.pt")
    print(f"\nCopy the shards to the cluster path named in the aug21seed configs.")
    print(f"If you change --shards, set SEED_SHARDS in make_configs_acridine.py "
          f"to match and regenerate.")


if __name__ == '__main__':
    main()

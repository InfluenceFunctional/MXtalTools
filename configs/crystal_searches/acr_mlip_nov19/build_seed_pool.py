"""seed_pool.pt for acr_mlip_nov19: the minima of acr_mlip_nov17's three searches, then acr_fm_oct27's seed pool.

    python build_seed_pool.py [--nov17 D:/crystal_datasets/acridine/campaigns/acr_mlip_nov17] [--write]

Order (the searches relax seeds in file order, and the seed stop rule can end the seeds before the last):
1. acr_mlip_nov17's search minima under MACE-OFF24, UMA-s 1.1 and UMA-s 1.2, interleaved one per model in turn, each
   model's own minima in ascending energy (its <run>_minima.pt; every minimum, the molecule's C2 copies included);
2. acr_fm_oct27's seed_pool.pt (21,882 states of the earlier acridine landscapes), in its own order.

params are each state's cell lengths, cell angles, asymmetric-unit centroid and orientation, hand its handedness, as
run.py's rebuild expects (coordinator.rebuild_crystals). source names each row's origin: '<nov17 run>#<start index>' or
the earlier pool's own source. No deduplication: a state that relaxes to a known minimum under a model is a recapture.
"""
import argparse
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OLD_POOL = HERE.parent / 'acr_fm_oct27' / 'seed_pool.pt'
RUNS = (('maceoff24m_search', 'mace'), ('umas1p1_search', 'uma'), ('umas1p2_search', 'uma'))


def minima_params(path, key):
    rows = torch.load(path, weights_only=False)
    rows = sorted(rows, key=lambda r: float(r[key]))
    params = torch.stack([torch.cat([r.cell_lengths[0], r.cell_angles[0], r.aunit_centroid[0], r.aunit_orientation[0]])
                          for r in rows]).float()
    hand = torch.stack([r.aunit_handedness.reshape(1) for r in rows]).float()
    starts = [int(r.start_index) for r in rows]
    return params, hand, starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nov17', default='D:/crystal_datasets/acridine/campaigns/acr_mlip_nov17')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    per = []
    for run, key in RUNS:
        p, h, s = minima_params(os.path.join(a.nov17, f'{run}_minima.pt'), key)
        per.append((run, p, h, s))
        print(f'{run}: {len(p)} minima')
    params, hand, source = [], [], []
    for i in range(max(len(p) for _, p, _, _ in per)):  # one per model in turn
        for run, p, h, s in per:
            if i < len(p):
                params.append(p[i]), hand.append(h[i]), source.append(f'acr_mlip_nov17/{run}#{s[i]}')
    old = torch.load(OLD_POOL, weights_only=False)
    params = torch.cat([torch.stack(params), old['params'].float()])
    hand = torch.cat([torch.stack(hand), old['hand'].float()])
    source += [f'acr_fm_oct27/{s}' for s in old['source']]
    assert params.shape == (len(source), 12) and hand.shape == (len(source), 1)
    assert set(hand.flatten().tolist()) <= {-1.0, 1.0}, 'handedness must be +-1'
    n_new = len(source) - len(old['source'])
    print(f'seed pool: {len(source)} rows = {n_new} nov17 minima + {len(old["source"])} acr_fm_oct27 states')
    if a.write:
        out = HERE / 'seed_pool.pt'
        torch.save(dict(params=params, hand=hand, source=source,
                        note='acr_mlip_nov17 search minima (interleaved by model, ascending energy), then '
                             'acr_fm_oct27/seed_pool.pt; build_seed_pool.py'), out)
        print('wrote', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    main()

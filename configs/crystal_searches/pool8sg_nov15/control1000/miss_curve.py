"""The miss of an N-start search, as a function of N, from the control1000 searches.

    python miss_curve.py SEARCH_DIR [--orders 200] [--seed 0]

For each molecule, each start's lowest minimum reached (the lowest minima.minimum_f among the minimum ids its trajectory
states carry; +inf for a start that reaches none). The starts are independent draws, so the first N of a random ordering
of a molecule's 1,000 starts is an N-start search: the MISS of that search is the lowest minimum it reaches minus the
lowest of all 1,000 (kT_s). Over --orders random orderings per molecule, pooled over the group's molecules, prints per
(group, N) the quantiles of the miss and the share of searches that reach the 1,000-start lowest, as TSV.

The reference is the 1,000-start lowest, not the true one: as N nears 1,000 a subset shares most of the reference's
starts and the miss goes to 0 by construction. Read N well below 1,000 (the grid stops at 500).

PESSIMISTIC AT SMALL N. The full search's cascade drops a start whose energy, past a step, is still well above its
condition's lowest finished energy (stream_search: early_stop), unrelaxed and with no minimum. Against the 1,000-start
running minimum that drops many starts, which here reach nothing; an N-start search cascades against its own, higher,
minimum and would relax some of them. The effect fades as N grows (the grid starts at 5).
"""
import argparse
import glob
import os
import re

import torch

GRID = (5, 10, 20, 50, 100, 150, 200, 300, 400, 500)
QUANTILES = (0.5, 0.75, 0.9, 0.95)


def start_best(slim):
    """[conditions, num_samples] each start's lowest minimum reached (kT_s), and [conditions] the lowest of all."""
    traj, mins = slim['traj'], slim['minima']
    n = int(slim['config']['num_samples'])
    n_cond = len(slim['keys'])
    order = torch.argsort(mins['minimum_id'])
    ids, f = mins['minimum_id'][order], mins['minimum_f'][order].double()
    mid = traj['minimum_id'].long()
    has = mid >= 0
    pos = torch.searchsorted(ids, mid[has]).clamp(max=ids.numel() - 1)
    assert bool((ids[pos] == mid[has]).all()), 'a trajectory minimum id is not in the minima table'
    f_state = torch.full((mid.numel(),), float('inf'), dtype=torch.float64)
    f_state[has] = f[pos]
    best = torch.full((n_cond * n,), float('inf'), dtype=torch.float64)
    best = best.scatter_reduce(0, traj['start_index'].long(), f_state, reduce='amin')
    best = best.view(n_cond, n)
    lowest = torch.full((n_cond,), float('inf'), dtype=torch.float64)
    lowest = lowest.scatter_reduce(0, mins['condition'].long(), f, reduce='amin')
    return best, lowest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('search_dir')
    ap.add_argument('--orders', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    g_rand = torch.Generator().manual_seed(a.seed)
    miss = {}
    for path in sorted(glob.glob(os.path.join(a.search_dir, 'sg*', '*_slim.pt'))):
        g = int(re.search(r'sg(\d+)', os.path.basename(os.path.dirname(path))).group(1))
        best, lowest = start_best(torch.load(path, map_location='cpu', weights_only=False))
        n = best.shape[1]
        grid = [k for k in GRID if k <= n]
        for c in range(best.shape[0]):
            if not torch.isfinite(lowest[c]):
                continue
            perm = torch.argsort(torch.rand(a.orders, n, generator=g_rand), dim=1)
            run = torch.cummin(best[c][perm], dim=1).values            # [orders, n]: lowest after k starts
            m = run[:, [k - 1 for k in grid]] - lowest[c]              # [orders, grid]
            miss.setdefault(g, []).append(m)
    if not miss:
        raise SystemExit(f'no sg*/*_slim.pt under {a.search_dir}')
    print('group\tmolecules\tN\t' + '\t'.join(f'miss_p{int(q * 100)}' for q in QUANTILES) + '\tfound_frac')
    for g in sorted(miss):
        m = torch.stack(miss[g])                                       # [molecules, orders, grid]
        for i, k in enumerate(GRID[:m.shape[-1]]):
            col = m[:, :, i].flatten()
            col = torch.where(torch.isfinite(col), col, torch.full_like(col, 1e9))
            qs = torch.quantile(col, torch.tensor(QUANTILES, dtype=col.dtype))
            print(f'{g}\t{m.shape[0]}\t{k}\t' + '\t'.join(f'{float(v):.3f}' for v in qs)
                  + f'\t{float((col <= 1e-9).double().mean()):.3f}')


if __name__ == '__main__':
    main()

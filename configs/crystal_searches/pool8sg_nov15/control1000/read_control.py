"""Read the control1000 searches: what do 100 starts per molecule miss, against 1,000?

    python read_control.py SEARCH_DIR [--first 100] [--window 5] [--per_molecule]

SEARCH_DIR holds sg<g>/pool8ctl<g>_c<k>_slim.pt (README.md). In each slim, start j of condition c is start_index
c x num_samples + j (the stream's numbering), so a condition's first N starts are, statistically, an N-start search.
A start REACHES a minimum when any of its trajectory states carries that minimum's id (traj.minimum_id). Energies are
the slim's own unit, kT_s (minima.minimum_f / traj.f_kT).

Per (group, molecule):
  minima          minima of the full 1,000-start search
  found_best      the 1,000-start lowest minimum is reached by the first N starts
  gap_kT          lowest minimum the first N reach, above the 1,000-start lowest
  window          minima within --window kT_s of the 1,000-start lowest
  window_reached  share of those the first N reach
Prints one row per group (molecules, share found_best, median and 90th percentile gap_kT, median window and
window_reached) and, with --per_molecule, every molecule.
"""
import argparse
import glob
import os
import re
import statistics

import torch


def per_condition(slim, first, window):
    traj, mins = slim['traj'], slim['minima']
    n = int(slim['config']['num_samples'])
    reached = (traj['minimum_id'] >= 0) & (traj['start_index'] % n < first)
    out = []
    for c in torch.unique(mins['condition']).tolist():
        sel = mins['condition'] == c
        ids, f = mins['minimum_id'][sel], mins['minimum_f'][sel].double()
        if ids.numel() == 0:
            continue
        hit = set(traj['minimum_id'][reached & (traj['condition'] == c)].long().tolist())
        got = torch.tensor([int(i) in hit for i in ids.tolist()])
        best = float(f.min())
        win = f <= best + window
        out.append({'condition': c, 'key': slim['keys'][c] if c < len(slim['keys']) else c,
                    'minima': int(ids.numel()), 'reached': int(got.sum()),
                    'found_best': bool(got[f.argmin()]),
                    'gap_kT': float(f[got].min()) - best if bool(got.any()) else float('inf'),
                    'window': int(win.sum()), 'window_reached': float(got[win].float().mean())})
    return out


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('search_dir')
    ap.add_argument('--first', type=int, default=100)
    ap.add_argument('--window', type=float, default=5.0)
    ap.add_argument('--per_molecule', action='store_true')
    a = ap.parse_args()
    rows = {}
    for path in sorted(glob.glob(os.path.join(a.search_dir, 'sg*', '*_slim.pt'))):
        g = int(re.search(r'sg(\d+)', os.path.basename(os.path.dirname(path))).group(1))
        slim = torch.load(path, map_location='cpu', weights_only=False)
        n = int(slim['config']['num_samples'])
        if n < a.first:
            raise SystemExit(f'{path}: {n} starts per molecule, fewer than --first {a.first}')
        for r in per_condition(slim, a.first, a.window):
            r['file'] = os.path.basename(path)
            rows.setdefault(g, []).append(r)
    if not rows:
        raise SystemExit(f'no sg*/*_slim.pt under {a.search_dir}')
    print(f'First {a.first} starts against the full search; window {a.window:g} kT_s; energies in kT_s.')
    print(f"{'group':>6} {'molecules':>9} {'found best':>10} {'gap p50':>8} {'gap p90':>8} "
          f"{'minima p50':>10} {'window p50':>10} {'reached p50':>11}")
    for g in sorted(rows):
        rs = rows[g]
        gaps = [r['gap_kT'] for r in rs]
        print(f"{g:>6} {len(rs):>9} {sum(r['found_best'] for r in rs) / len(rs):>10.0%} "
              f"{statistics.median(gaps):>8.2f} {quantile(gaps, 0.9):>8.2f} "
              f"{statistics.median(r['minima'] for r in rs):>10.0f} {statistics.median(r['window'] for r in rs):>10.0f} "
              f"{statistics.median(r['window_reached'] for r in rs):>11.0%}")
    if a.per_molecule:
        for g in sorted(rows):
            for r in rows[g]:
                print(f"sg{g} {r['file']} {str(r['key'])[:40]:40s} minima {r['minima']:4d} reached {r['reached']:4d} "
                      f"best {'Y' if r['found_best'] else 'n'} gap {r['gap_kT']:5.2f} window {r['window']:4d} "
                      f"reached {r['window_reached']:4.0%}")


if __name__ == '__main__':
    main()

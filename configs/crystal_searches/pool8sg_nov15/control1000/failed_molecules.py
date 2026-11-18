"""Which molecules did the control1000 searches fail on, and how did their starts end?

    python failed_molecules.py SEARCH_DIR [--first 100] [--packing 0.55 0.95] [--window 5] [--list]

SEARCH_DIR holds sg<g>/pool8ctl<g>_c<k>_slim.pt (README.md). A minimum's packing coefficient is that of its lowest
recorded trajectory state carrying its id (band_from_slim.packing_of: molecule volume x symmetry multiplicity / cell
volume). A molecule FAILS a search when no minimum in the packing window lies within --window kT_s of the lowest
minimum that search reached ('no in-window minimum near the lowest'), the case that leaves the band builder few or no
rows; and fails outright when it reached no in-window minimum at all. Read for the first --first starts and for all
of them: a failure that disappears with more starts is the search; one that persists is the energy or the settings.

Each start's ending, from its trajectory (start j of condition c is start_index c x num_samples + j):
  converged   some state carries a minimum id
  step_cap    no minimum, and its last state is at the last stage's step limit (max_num_steps - 1)
  cascade     no minimum, and it ended in the last stage before the limit (the early_stop cascade)
  other       no minimum, and its last state is in an earlier stage
Disconnected starts are redrawn before relaxing and are not in the trajectories: the file's own counts
(effort.starts_disconnected / starts_redrawn) are printed per file.

Prints per group the failure shares at --first and at all starts and the share whose lowest minimum is out of window;
with --list, every molecule that fails either search: its lowest minimum (kT_s, packing), the gap to its lowest
in-window minimum, its in-window minima, and its starts' endings.
"""
import argparse
import glob
import importlib.util
import os
import re

import torch

_spec = importlib.util.spec_from_file_location(
    'band_from_slim', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pool8_nov09',
                                   'band_from_slim.py'))
bfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bfs)


def minimum_packing(traj, pc):
    """{minimum id: packing of its lowest recorded state}"""
    mid = traj['minimum_id'].long()
    has = torch.nonzero(mid >= 0).flatten()
    order = has[torch.argsort(traj['f_kT'][has].double())]
    out = {}
    for i, m in zip(order.tolist(), mid[order].tolist()):
        out.setdefault(m, float(pc[i]))
    return out


def endings(traj, n_starts, stages):
    """[n_starts] 0 converged, 1 step_cap, 2 cascade, 3 other (by start_index)"""
    si = traj['start_index'].long()
    conv = torch.zeros(n_starts, dtype=torch.bool)
    conv[si[traj['minimum_id'] >= 0]] = True
    key = traj['stage'].long() * 100000 + traj['step'].long()
    last = torch.full((n_starts,), -1, dtype=torch.long).scatter_reduce(0, si, key, reduce='amax')
    stage, step = last // 100000, last % 100000
    final = len(stages) - 1
    cap = int(stages[final].get('max_num_steps', 500)) - 1
    out = torch.full((n_starts,), 3, dtype=torch.long)
    out[(stage == final) & (step < cap)] = 2
    out[(stage == final) & (step >= cap)] = 1
    out[conv] = 0
    return out


def judge(f, inwin, lo_window):
    """(fails near, fails outright, lowest out of window, gap to lowest in-window) over the minima given"""
    if f.numel() == 0:
        return True, True, True, float('inf')
    lo = float(f.min())
    gap = float(f[inwin].min()) - lo if bool(inwin.any()) else float('inf')
    return gap > lo_window, not bool(inwin.any()), not bool(inwin[f.argmin()]), gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('search_dir')
    ap.add_argument('--first', type=int, default=100)
    ap.add_argument('--packing', type=float, nargs=2, default=(0.55, 0.95))
    ap.add_argument('--window', type=float, default=5.0)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    groups, failed = {}, []
    for path in sorted(glob.glob(os.path.join(a.search_dir, 'sg*', '*_slim.pt'))):
        g = int(re.search(r'sg(\d+)', os.path.basename(os.path.dirname(path))).group(1))
        slim = torch.load(path, map_location='cpu', weights_only=False)
        keys = [tuple(k) for k in slim['keys']]
        traj = dict(slim['traj'])
        n = int(slim['config']['num_samples'])
        pc = bfs.packing_of(traj, keys, bfs.templates_of(slim))
        mpack = minimum_packing(traj, pc)
        ends = endings(traj, len(keys) * n, list(slim['config']['opt']))
        eff = slim.get('effort') or {}
        print(f"{os.path.basename(path)}: starts disconnected {eff.get('starts_disconnected')}, redrawn "
              f"{eff.get('starts_redrawn')}, kept disconnected {eff.get('starts_kept_disconnected')}")
        mins = slim['minima']
        si, mid = traj['start_index'].long(), traj['minimum_id'].long()
        reached_first = set(mid[(mid >= 0) & (si % n < a.first)].tolist())
        for c, key in enumerate(keys):
            sel = mins['condition'] == c
            ids, f = mins['minimum_id'][sel].long(), mins['minimum_f'][sel].double()
            p = torch.tensor([mpack.get(int(i), float('nan')) for i in ids.tolist()], dtype=torch.float64)
            inwin = (p >= a.packing[0]) & (p <= a.packing[1])
            got = torch.tensor([int(i) in reached_first for i in ids.tolist()], dtype=torch.bool)
            full = judge(f, inwin, a.window)
            first = judge(f[got], inwin[got], a.window)
            e = ends[c * n:(c + 1) * n]
            row = dict(group=g, key=key[0], file=os.path.basename(path), minima=int(ids.numel()),
                       inwin=int(inwin.sum()), lo=float(f.min()) if f.numel() else float('nan'),
                       lo_pack=float(p[f.argmin()]) if f.numel() else float('nan'), full=full, first=first,
                       ends=[float((e == k).double().mean()) for k in range(4)])
            groups.setdefault(g, []).append(row)
            if full[0] or first[0]:
                failed.append(row)
    if not groups:
        raise SystemExit(f'no sg*/*_slim.pt under {a.search_dir}')
    print(f'\nPacking window {a.packing[0]}-{a.packing[1]}; window {a.window:g} kT_s; fail = no in-window minimum '
          f'within the window of the lowest; first {a.first} starts vs all.')
    print(f"{'group':>6} {'molecules':>9} {'fail@' + str(a.first):>9} {'fail@all':>9} {'none@' + str(a.first):>9} "
          f"{'none@all':>9} {'lowest out@all':>14} {'converged':>9} {'step_cap':>8} {'cascade':>8} {'other':>6}")
    for g in sorted(groups):
        rs = groups[g]
        sh = lambda fn: sum(fn(r) for r in rs) / len(rs)
        ends = [sum(r['ends'][k] for r in rs) / len(rs) for k in range(4)]
        print(f"{g:>6} {len(rs):>9} {sh(lambda r: r['first'][0]):>9.0%} {sh(lambda r: r['full'][0]):>9.0%} "
              f"{sh(lambda r: r['first'][1]):>9.0%} {sh(lambda r: r['full'][1]):>9.0%} {sh(lambda r: r['full'][2]):>14.0%} "
              f"{ends[0]:>9.0%} {ends[1]:>8.0%} {ends[2]:>8.0%} {ends[3]:>6.0%}")
    if a.list and failed:
        print('\nFailed molecules (either search): lowest minimum kT_s and packing; gap to the lowest in-window '
              'minimum (kT_s, all starts); in-window minima / all; endings converged/step_cap/cascade/other')
        for r in sorted(failed, key=lambda r: (r['group'], r['key'])):
            print(f"sg{r['group']} {r['key']:28s} lowest {r['lo']:8.2f} pack {r['lo_pack']:.3f} gap {r['full'][3]:6.2f} "
                  f"in-window {r['inwin']}/{r['minima']} fail@first {'Y' if r['first'][0] else 'n'} "
                  f"fail@all {'Y' if r['full'][0] else 'n'} ends " + '/'.join(f'{x:.0%}' for x in r['ends']))


if __name__ == '__main__':
    main()

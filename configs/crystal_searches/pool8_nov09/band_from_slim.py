"""Band sets from one pool8 task's <run>_slim.pt (slim_outputs.py, format 'stream_slim/1'): the prior constructor's
first step, run per task.

    python band_from_slim.py SLIM.pt OUT_DIR --split pool8_mols.tsv [--bands 0:1:8,1:3:12,3:5:12] [--net_A 0.1]
        [--packing 0.55 0.95] [--starts N] [--device cpu]

For each split of --split (a TSV with columns key and split, the pool's train / heldout_random / heldout_scaffold) that
has conditions in this task, writes OUT_DIR/<split>/<run>_band/ with ``stream_search.write_band``, the writer the
streaming search itself calls with ``basin_set: true``: the Z' = 1 states within the top of --bands of their
condition's lowest minimum (E/kT under the search's envelope), thinned along each descent at --net_A A, then picked per
condition by ``minima.banded_farthest_points``: --bands lists lo:hi:budget (E/kT above the lowest minimum), and within
each band the lowest state comes first, then the state farthest from every one kept so far (owner 2026-11-09: 32 rows
per molecule, 8 in 0-1 kT, 12 in 1-3, 12 in 3-5); each pick is re-analysed at the search's hard cutoff. --bands none
keeps write_band's --net_A ball net within each descent group under --band_kT instead. Templates are rebuilt from the file's molecules as the
search built them (``utils.init_samples_to_optim``, one per molecule); the minima library is the file's minima.

Before banding:
  * --packing [lo hi] keeps the states whose packing coefficient (molecule volume x symmetry multiplicity / cell
    volume, from the state's cell parameters) lies in the window (default 0.55 0.95, molecular_crystal.py's
    PACKING_COEFF_WINDOW in GFN). A condition's lowest minimum is the reference whether or not it is in the window.
    Dropping a state shortens the path-length thinning's step across the gap to the direct distance (a slight
    under-keep along that segment); the ball net that follows sets the final spacing.
  * --starts N keeps the states of each condition's first N starts (start j of condition c is start_index
    c x num_samples + j, the stream's numbering; a depth-versus-breadth subset: the first N starts approximate an
    N-start search; the minima library, and so the band's reference minimum, stay the full search's).

A split with no state in the band writes no folder. Writes OUT_DIR/<split>/<run>_band.json per split (write_band's
record: states in, in band, path-thinned, net, the importer's refusals; wall time), and when every split is written,
OUT_DIR/done/<run>.json (the task is done; a rerun of a done task does nothing). A rerun skips a split whose record
exists and redoes one whose folder exists without it (a task stopped mid-write).
"""
import argparse
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

import torch

FORMAT_IN = 'stream_slim/1'


def read_split(path):
    """{key: split} from a TSV with a header naming 'key' and 'split'."""
    with open(path) as fh:
        head = fh.readline().rstrip('\n').split('\t')
        ki, si = head.index('key'), head.index('split')
        return {p[ki]: p[si] for p in (ln.rstrip('\n').split('\t') for ln in fh) if len(p) > max(ki, si)}


def templates_of(slim):
    """{(identifier, sg, Z'): crystal row} as the search built them: init_samples_to_optim, one per molecule."""
    from mxtaltools.common.config_processing import dict2namespace
    from mxtaltools.crystal_search import minima as mn
    from mxtaltools.crystal_search.utils import init_samples_to_optim
    cfg = dict2namespace(dict(slim['config']))
    out = {}
    for m in slim['molecules']:
        for r in init_samples_to_optim(cfg, target=m, num_samples=1):
            out.setdefault(mn.row_key(r), r)
    missing = [k for k in slim['keys'] if tuple(k) not in out]
    if missing:
        raise ValueError(f'{len(missing)} conditions have no template (e.g. {missing[:2]})')
    return out


def packing_of(traj, keys, templates):
    """[n] packing coefficient of each state: molecule volume x symmetry multiplicity / cell volume."""
    from mxtaltools.crystal_fd.measure import cell_volume
    per = torch.tensor([float(templates[tuple(k)].mol_volume) * float(templates[tuple(k)].sym_mult) for k in keys],
                       dtype=torch.float64)
    return per[traj['condition'].long()] / cell_volume(traj['params'][:, :6].double())


def subset(traj, mask):
    """The trajectory dict's per-state entries at ``mask``; the rest as they are."""
    n = len(traj['f_kT'])
    return {k: (v[mask] if torch.is_tensor(v) and v.shape[:1] == (n,) else v) for k, v in traj.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('slim')
    ap.add_argument('out_dir')
    ap.add_argument('--split', required=True)
    ap.add_argument('--bands', default='0:1:8,1:3:12,3:5:12',
                    help="lo:hi:budget,... (E/kT above the lowest minimum), or 'none' for the ball net")
    ap.add_argument('--band_kT', type=float, default=None, help="with --bands none: the band (default 15)")
    ap.add_argument('--net_A', type=float, default=0.1)
    ap.add_argument('--packing', type=float, nargs=2, default=(0.55, 0.95))
    ap.add_argument('--starts', type=int, default=None)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args(argv)
    from mxtaltools.crystal_search.stream_search import write_band
    if a.bands == 'none':
        bands, band_kT = None, 15.0 if a.band_kT is None else a.band_kT
    else:
        if a.band_kT is not None:
            raise SystemExit('--band_kT is the top of --bands; give it with --bands none only')
        bands = [(float(lo), float(hi), int(b)) for lo, hi, b in (t.split(':') for t in a.bands.split(','))]
        band_kT = max(hi for _, hi, _ in bands)

    t0 = time.time()
    run = os.path.basename(a.slim)[:-len('_slim.pt')]
    done = os.path.join(a.out_dir, 'done', f'{run}.json')
    if os.path.exists(done):
        print(f'{run}: done before ({done})')
        return json.load(open(done))
    slim = torch.load(a.slim, weights_only=False)
    if slim.get('format') != FORMAT_IN:
        raise ValueError(f"{a.slim}: format {slim.get('format')!r}, not {FORMAT_IN}")
    keys = [tuple(k) for k in slim['keys']]
    split_of = read_split(a.split)
    unknown = sorted({k[0] for k in keys} - set(split_of))
    if unknown:
        raise ValueError(f'{len(unknown)} conditions are not in {a.split} (e.g. {unknown[:3]})')
    stages = slim['config']['opt']
    last = stages[-1]
    key = str(last['optim_target']).lower()
    templates = templates_of(slim)

    traj = dict(slim['traj'])
    traj['keys'] = keys
    traj['handedness'] = traj['handedness'].float()
    n_all = len(traj['f_kT'])
    pc = packing_of(traj, keys, templates)
    keep = (pc >= a.packing[0]) & (pc <= a.packing[1])
    if a.starts is not None:   # start j of condition c is start_index c x num_samples + j
        j = traj['start_index'].long() - traj['condition'].long() * int(slim['config']['num_samples'])
        if int(j.min()) < 0 or int(j.max()) >= int(slim['config']['num_samples']):
            raise ValueError('start_index is not c x num_samples + j: --starts cannot rank starts per condition')
        keep &= j < a.starts
    mn_ = slim['minima']
    library = SimpleNamespace(key=[keys[int(c)] for c in mn_['condition'].tolist()], f=mn_['minimum_f'].tolist())

    cond_split = [split_of[k[0]] for k in keys]
    state_split = torch.tensor([sorted(set(cond_split)).index(s) for s in cond_split])[traj['condition'].long()]
    records = {}
    for si, split in enumerate(sorted(set(cond_split))):
        d = os.path.join(a.out_dir, split)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f'{run}_band')
        rec_path = os.path.join(d, f'{run}_band.json')
        if os.path.exists(rec_path):
            records[split] = json.load(open(rec_path))
            continue
        if os.path.exists(path):   # written by a run stopped before its record
            shutil.rmtree(path)
        sel = keep & (state_split == si)
        t1 = time.time()
        _, rec, evals = write_band(
            subset(traj, sel), templates, path, library=library, kT=float(slim['kT']), key=key,
            band_kT=band_kT, net_A=a.net_A, energy=slim['energy'], search=slim['search'], identity=slim['identity'],
            device=a.device, predictor=None, cutoff=float(last.get('cutoff') or 10),
            supercell_size=int(last.get('supercell_size', 10)), energy_chunk=256, bands=bands)
        rec = dict(rec, split=split, conditions=sum(s == split for s in cond_split), states_all=int(
            (state_split == si).sum()), states_in_window=int(sel.sum()), packing=list(a.packing), starts=a.starts,
            bands=bands, evals=int(evals), wall_s=round(time.time() - t1, 1), slim=os.path.abspath(a.slim),
            mxt_commit_search=slim.get('mxt_commit'))
        with open(rec_path + '.tmp', 'w') as fh:
            json.dump(rec, fh, indent=1, default=str)
        os.replace(rec_path + '.tmp', rec_path)
        records[split] = rec
        print(f"{run} {split}: {rec['conditions']} conditions, {rec['states_all']:,} states, "
              f"{rec['states_in_window']:,} in the packing window, {rec.get('in_band', 0):,} in band, "
              f"{rec.get('path_thinned', 0):,} path-thinned, net {rec.get('net', 0):,}; {rec['wall_s']} s", flush=True)
    print(f'{run}: {n_all:,} states, {len(keys)} conditions, {time.time() - t0:.0f} s', flush=True)
    os.makedirs(os.path.dirname(done), exist_ok=True)
    with open(done + '.tmp', 'w') as fh:
        json.dump(records, fh, indent=1, default=str)
    os.replace(done + '.tmp', done)
    return records


if __name__ == '__main__':
    sys.exit(0 if main() is not None else 1)

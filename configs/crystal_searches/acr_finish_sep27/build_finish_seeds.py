"""
Finish-pass seeds for acr_finish_sep27: a snapshot of acr_proposals_sep27 end states, rebuilt as run_search data-mode
seeds. Cluster-side; submit.sbatch calls ensure() before a finish job.

SOURCES lists the sep27 wrap runs and how many of their output rows to take: the first FIRST_ROWS rows of a
random-start run (rows are only ever appended, so every snapshot taken once the run has that many rows is the same
set, and the two finish jobs continue identical states), and every row of a seeded run, which must be complete.
A source run with fewer rows than needed fails loudly: resubmit once it has them.

Each end state is re-expressed by its full cell parameters and handedness and rebuilt with the search's own conformer
(acr_proposals_sep27/build_seeds.build; its round trip reproduces eLJ to 7e-7 relative). The compact file beside the
dataset keeps, per row, the source run, its row in that run's output, the recorded MACE energy, and the source's
dataset_index when it has one; the finish outputs' dataset_index is the row of that file.

    python build_finish_seeds.py wrap <opt_outs dir> <mol_path> <out.pt>
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'acr_proposals_sep27'))
from build_seeds import build  # noqa: E402

FIRST_ROWS = 1000
SOURCES = {
    'wrap': [('sep27rwrap_acridine_sg14_zp2_0', FIRST_ROWS), ('sep27rwrap_acridine_sg14_zp2_1', FIRST_ROWS),
             ('sep27seedwrap_acridine_sg14_zp2_0', None), ('sep27seedwrap_acridine_sg14_zp2_1', None)],
}
SEEDED_ROWS = 282  # rows per complete seeded run (its seed shard; no rows were dropped)


def expected_rows(which):
    return sum(n if n is not None else SEEDED_ROWS for _, n in SOURCES[which])


def _load(path, tries=10):
    """The random-start runs are still writing their output; a read can meet a half-written file."""
    for k in range(tries):
        try:
            return torch.load(path, weights_only=False, map_location='cpu')
        except Exception as e:  # noqa: BLE001 -- truncated zip, EOF: retry after the writer finishes
            if k == tries - 1:
                raise
            print(f'retrying read of {path}: {type(e).__name__}')
            time.sleep(30)


def snapshot(which, opt_outs):
    rows, meta = [], []
    for run, n in SOURCES[which]:
        out = _load(os.path.join(opt_outs, f'{run}.pt'))
        if n is None:
            prog = json.load(open(os.path.join(opt_outs, f'{run}_progress.json')))
            if prog['cursor'] < prog['num_samples'] or len(out) != SEEDED_ROWS:
                raise SystemExit(f'{run}: seeded run incomplete or not {SEEDED_ROWS} rows ({len(out)} rows, '
                                 f'cursor {prog["cursor"]} of {prog["num_samples"]})')
            take = out
        else:
            if len(out) < n:
                raise SystemExit(f'{run}: {len(out)} rows so far, the finish pass needs its first {n}; resubmit later')
            take = out[:n]
        for i, c in enumerate(take):
            rows.append(c)
            meta.append(dict(source=run, row=i, mace=float(c.mace),
                             source_dataset_index=int(c.dataset_index) if 'dataset_index' in c.keys() else -1))
    return rows, meta


def ensure(which, opt_outs, mol_path, out_path):
    """Build out_path unless it already holds the expected number of rows."""
    want = expected_rows(which)
    if os.path.exists(out_path) and len(_load(out_path)) == want:
        return
    rows, meta = snapshot(which, opt_outs)
    assert len(rows) == want, (len(rows), want)
    from mxtaltools.dataset_utils.utils import collate_data_list
    params, hand = [], []
    for k in range(0, len(rows), 500):
        b = collate_data_list(rows[k:k + 500])
        params.append(b.full_cell_parameters().detach().float())
        hand.append(b.aunit_handedness.detach().float().reshape(-1, 2))
    compact = os.path.splitext(out_path)[0] + '_compact.pth'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f'.{os.getpid()}.tmp'  # both finish jobs may build at once: same rows, own temp files
    torch.save({'params': torch.cat(params), 'handedness': torch.cat(hand), 'meta': meta}, compact + tmp)
    os.replace(compact + tmp, compact)
    build(compact, mol_path, out_path + tmp)
    os.replace(out_path + tmp, out_path)
    print(f'built {want} finish seeds ({which}) -> {out_path}')


if __name__ == '__main__':
    ensure(*sys.argv[1:5])

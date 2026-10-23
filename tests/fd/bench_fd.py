"""
Named benchmark, gate 7 of fd_tooling.md section 4: the fundamental-domain canonicalisers and charts on CPU.

    python tests/fd/bench_fd.py [--rows 100000] [--chart-rows 1000] [--reps 5] [--threads 4] [--check]

Times, each as the median of ``--reps`` calls after one warm-up, float64 unless named:
- the pose canonicaliser ``domains.fold`` at sg 2 (|Gamma+| = 8), sg 61 (|Gamma+| = 32) and sg 227
  (|Gamma+| = 192), random poses;
- the triclinic cell canonicaliser ``ops.triclinic_basis`` on scrambled reduced cells;
- chart forward + inverse + log-Jacobian per 1000 rows: the sg 2 domain chart (``CrystalChart``) with each
  orientation chart, and the trainer latent chart (``LatentV1Chart``, float32 as the trainer runs it).
Budgets (gate 7): fold >= 1e5 rows/s at |Gamma+| <= 32 and >= 5e4 at 192; cell <= 15 min per 1.2e6 rows; charts
<= 4 ms per 1000. Not collected by pytest (the name does not start with ``test_``). With ``--check`` it exits 1 if
any value misses its budget (and names it); otherwise it only prints. Run with CUDA_VISIBLE_DEVICES=-1: the GPU is
never used. On a shared CPU the numbers are contended.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mxtaltools.constants.space_group_info import LATTICE_TYPE  # noqa: E402
from mxtaltools.crystal_fd import charts, checks, domains, ops, so3  # noqa: E402
from mxtaltools.crystal_fd.description import CrystalDescription  # noqa: E402
from mxtaltools.crystal_fd.tables import group_table  # noqa: E402

F64 = torch.float64


def median_time(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def poses(sg, n, g):
    return CrystalDescription.from_parts(sg, checks.random_cells(LATTICE_TYPE[sg], n, g),
                                         torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--rows', type=int, default=100_000)
    ap.add_argument('--cell-rows', type=int, default=20_000)
    ap.add_argument('--chart-rows', type=int, default=1000)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--check', action='store_true', help='exit 1 if any value misses its budget')
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    g = torch.Generator().manual_seed(0)
    rows = []

    for sg, floor in ((2, 1e5), (61, 1e5), (227, 5e4)):
        d = poses(sg, a.rows, g)
        t = median_time(lambda: domains.fold(d), a.reps)
        rows.append((f'fold, sg {sg} (|Gamma+| = {len(group_table(sg).gamma_plus)})', 'rows/s', a.rows / t,
                     ('>=', floor)))

    cells = checks.random_cells('triclinic', a.cell_rows, g)
    t = median_time(lambda: ops.triclinic_basis(cells), max(1, a.reps // 2))
    rate = a.cell_rows / t
    rows.append(('triclinic_basis (cell canonicaliser)', 'rows/s', rate, ('>=', 1.2e6 / 900)))
    rows.append(('triclinic_basis, 1.2e6 rows', 'min', 1.2e6 / rate / 60, ('<=', 15.0)))

    n = a.chart_rows
    for name in ('hemisphere', 'euler_zyz', 'axis_angle'):
        ch = charts.CrystalChart.for_latent_v1(2, 1.7561, name)
        y, _ = charts.from_unit(ch, torch.rand(n, ch.ndim, generator=g, dtype=F64))

        def run():
            m = ch.forward(y)
            ch.inverse(m.desc)
        rows.append((f'CrystalChart sg 2 / {name}: forward + inverse + logdet', 'ms per 1000 rows',
                     1e3 * median_time(run, a.reps) * 1000 / n, ('<=', 4.0)))
    lv = charts.LatentV1Chart(2, 1.7561)
    x = (-0.98 + 1.96 * torch.rand(n, 12, generator=g)).float()

    def run_latent():
        m = lv.forward(x)
        lv.inverse(m.desc)
    rows.append(('LatentV1Chart sg 2 (float32): forward + inverse + logdet', 'ms per 1000 rows',
                 1e3 * median_time(run_latent, a.reps) * 1000 / n, ('<=', 4.0)))

    print(f'\nGate 7 timings on CPU ({a.threads} threads, torch {torch.__version__}); each value is the median of '
          f'{a.reps} calls after a warm-up, on {a.rows} random poses (fold), {a.cell_rows} random triclinic cells '
          f'(cell) and {n} chart rows. Compare each value with its budget (fd_tooling.md gate 7); a shared CPU '
          f'inflates times.\n')
    w = max(len(r[0]) for r in rows)
    print(f'{"operation":<{w}}  {"value":>12}  {"unit":<18}  {"budget":<12}  met')
    missed = []
    for name, unit, val, (op, lim) in rows:
        met = val >= lim if op == '>=' else val <= lim
        missed += [] if met else [name]
        print(f'{name:<{w}}  {val:>12.4g}  {unit:<18}  {op + " " + f"{lim:.4g}":<12}  {"yes" if met else "NO"}')
    if missed:
        print(f'\n{len(missed)} budget(s) missed: {missed}')
    return 1 if (a.check and missed) else 0


if __name__ == '__main__':
    sys.exit(main())

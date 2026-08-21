"""
Turn the benchmark logs and outputs into a throughput comparison.

Counts what is ON DISK, not what was requested. `run_search` saves after every
batch, so an arm killed at the wall limit still leaves everything it finished --
and on the CPU side that is the expected case, not a failure. An arm that did
600 of 1000 samples in 12 hours is a perfectly good throughput measurement; only
an arm with ZERO output tells us nothing.

    python summarize_bench.py --logs /path/to/slurm-*.out
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import torch

OUT_DIR = '/scratch/mk8347/data/crystal_datasets/acridine/opt_outs'
BENCH_LINE = re.compile(
    r'BENCH device=(?P<dev>\w+) arm=(?P<arm>\d+).*?seconds=(?P<sec>\d+)')
RUN_NAME = re.compile(r'bench(?P<dev>gpu|cpu)_acridine_sg\d+_zp\d+_b(?P<batch>\d+)_(?P<arm>\d+)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--logs', default='slurm-*.out')
    ap.add_argument('--out-dir', default=OUT_DIR)
    cli = ap.parse_args()

    seconds = {}
    for path in glob.glob(cli.logs):
        for line in open(path, errors='ignore'):
            m = BENCH_LINE.search(line)
            if m:
                seconds[(m['dev'], int(m['arm']))] = int(m['sec'])
    if not seconds:
        raise SystemExit(f"no BENCH lines in {cli.logs} -- did the jobs run?")

    rows = []
    for path in sorted(glob.glob(os.path.join(cli.out_dir, 'bench*_acridine_*.pt'))):
        m = RUN_NAME.search(os.path.basename(path))
        if not m:
            continue
        dev, batch, arm = m['dev'], int(m['batch']), int(m['arm'])
        try:
            done = len(torch.load(path, weights_only=False, map_location='cpu'))
        except Exception as e:
            print(f"   unreadable: {os.path.basename(path)} ({type(e).__name__})")
            continue
        sec = seconds.get((dev, arm))
        if sec is None:
            print(f"   no timing for {dev} arm {arm} -- skipped")
            continue
        rows.append((dev, batch, arm, done, sec, done / sec if sec else 0.0))

    if not rows:
        raise SystemExit("no arm produced output")

    print(f"{'device':7s} {'batch':>6} {'arm':>4} {'done':>6} {'seconds':>8} "
          f"{'samples/s':>10} {'samples/hr':>11}")
    for dev, batch, arm, done, sec, rate in sorted(rows):
        print(f"{dev:7s} {batch:6d} {arm:4d} {done:6d} {sec:8d} "
              f"{rate:10.3f} {rate * 3600:11.0f}")

    by_dev = defaultdict(list)
    by_cpu_batch = defaultdict(list)
    for dev, batch, arm, done, sec, rate in rows:
        by_dev[dev].append(rate)
        if dev == 'cpu':
            by_cpu_batch[batch].append(rate)

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float('nan')

    print()
    for dev in ('gpu', 'cpu'):
        if by_dev[dev]:
            r = med(by_dev[dev])
            print(f"{dev}: median {r:.3f} samples/s ({r * 3600:.0f}/hr) "
                  f"over {len(by_dev[dev])} arms")

    if by_cpu_batch:
        print("\nCPU throughput by batch size (the operating point this finds):")
        for batch in sorted(by_cpu_batch):
            r = med(by_cpu_batch[batch])
            print(f"   batch {batch:4d}: {r:7.3f} samples/s ({r * 3600:6.0f}/hr)")
        best = max(by_cpu_batch, key=lambda b: med(by_cpu_batch[b]))
        print(f"   best CPU batch size: {best}")

    if by_dev['gpu'] and by_dev['cpu']:
        g, c = med(by_dev['gpu']), med(by_dev['cpu'])
        print(f"\nGPU is {g / c:.1f}x the CPU throughput "
              f"(median of medians, best CPU batch aside)")
        #: what it costs to do one production combo on each
        for name, r in (('GPU', g), ('CPU', c)):
            hrs = 100_000 / r / 3600
            print(f"   100,000 samples on {name}: {hrs:,.1f} core-hours "
                  f"({hrs / 24:,.1f} days on one job)")
        print("   NOTE acr_production asked for 100k per combo and landed 55-86k; "
              "budget against what arrives.")


if __name__ == '__main__':
    main()

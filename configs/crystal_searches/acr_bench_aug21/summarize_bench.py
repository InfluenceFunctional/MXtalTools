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

OUT_DIR = '/scratch/mk8347/data/crystal_datasets/acridine/opt_outs'
BENCH_LINE = re.compile(
    r'BENCH device=(?P<dev>\w+) arm=(?P<arm>\d+).*?seconds=(?P<sec>\d+)')
RUN_NAME = re.compile(r'bench(?P<dev>gpu|cpu)_acridine_sg\d+_zp\d+_b(?P<batch>\d+)_(?P<arm>\d+)')
#: run_search drives `tqdm(total=num_samples, unit="samples")`, so the log records
#: progress independently of the output file. That is the fallback when the .pt
#: cannot be unpickled -- counting a pickled list requires the classes that wrote
#: it, so without mxtaltools on the path every file reads as unreadable and the
#: whole run would look like it produced nothing.
TQDM_LINE = re.compile(r'(?P<done>\d+)/(?P<total>\d+)\s*\[[^\]]*\]\s*$')


def count_from_log(path):
    """Highest sample count tqdm reported, or None."""
    best = None
    for chunk in open(path, errors='ignore'):
        for piece in chunk.replace('\r', '\n').split('\n'):
            m = TQDM_LINE.search(piece.strip())
            if m:
                best = max(best or 0, int(m['done']))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--logs', default='slurm-*.out')
    ap.add_argument('--out-dir', default=OUT_DIR)
    cli = ap.parse_args()

    seconds, log_counts = {}, {}
    for path in glob.glob(cli.logs):
        key = None
        for line in open(path, errors='ignore'):
            m = BENCH_LINE.search(line)
            if m:
                key = (m['dev'], int(m['arm']))
                seconds[key] = int(m['sec'])
        if key is not None:
            n = count_from_log(path)
            if n is not None:
                log_counts[key] = n
    if not seconds:
        raise SystemExit(f"no BENCH lines in {cli.logs} -- did the jobs run?")

    #: prefer the output file (authoritative), fall back to the log's tqdm
    rows, unreadable = [], 0
    seen = set()
    for path in sorted(glob.glob(os.path.join(cli.out_dir, 'bench*_acridine_*.pt'))):
        m = RUN_NAME.search(os.path.basename(path))
        if not m:
            continue
        dev, batch, arm = m['dev'], int(m['batch']), int(m['arm'])
        sec = seconds.get((dev, arm))
        if sec is None:
            print(f"   no timing for {dev} arm {arm} -- skipped")
            continue
        try:
            import torch
            done = len(torch.load(path, weights_only=False, map_location='cpu'))
            src = 'file'
        except Exception:
            done = log_counts.get((dev, arm))
            src = 'log'
            if done is None:
                unreadable += 1
                continue
        seen.add((dev, arm))
        rows.append((dev, batch, arm, done, sec, done / sec if sec else 0.0, src))

    #: arms whose output file is missing entirely still have a log
    for (dev, arm), sec in seconds.items():
        if (dev, arm) in seen or (dev, arm) not in log_counts:
            continue
        rows.append((dev, -1, arm, log_counts[(dev, arm)], sec,
                     log_counts[(dev, arm)] / sec if sec else 0.0, 'log'))
        print(f"   {dev} arm {arm}: no output file, counted from the log")

    if unreadable:
        print(f"   {unreadable} output file(s) unreadable AND unlogged -- run this "
              f"inside the container with PYTHONPATH set to unpickle them")
    if not rows:
        raise SystemExit("no arm produced a countable result")

    print(f"{'device':7s} {'batch':>6} {'arm':>4} {'done':>6} {'seconds':>8} "
          f"{'samples/s':>10} {'samples/hr':>11}  src")
    for dev, batch, arm, done, sec, rate, src in sorted(rows):
        #: batch -1 means the output file was missing and the row came from the
        #: log, which carries no batch size
        b = '?' if batch < 0 else str(batch)
        print(f"{dev:7s} {b:>6} {arm:4d} {done:6d} {sec:8d} "
              f"{rate:10.3f} {rate * 3600:11.0f}  {src}")

    by_dev = defaultdict(list)
    by_cpu_batch = defaultdict(list)
    for dev, batch, arm, done, sec, rate, src in rows:
        by_dev[dev].append(rate)
        if dev == 'cpu' and batch > 0:
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

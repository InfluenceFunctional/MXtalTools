"""
Status of every qm9_full_sep29 task, from the cluster's own files: INDEX.tsv (what each array index must produce), the
progress file each run writes beside its output, and the newest job log of each array index. Standard library only, so
the login node's python3 runs it:

    python3 check_battery.py [--out-dir DIR] [--log-dir DIR]

A task is done when its progress file says every start was relaxed and written (cursor = num_samples = n_out) and that
count is the one INDEX.tsv expects; a different count means the run used another configuration. Rates come only from
logs that show relaxations (a finished task resubmitted exits at once).
"""
import argparse
import collections
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def newest_logs(log_dir):
    """array index -> the log of the newest job that ran it (a later job id is a later submission)"""
    best = {}
    for p in log_dir.glob('qm9full-*_*.out'):
        m = re.fullmatch(r'qm9full-(\d+)_(\d+)\.out', p.name)
        if m and (int(m.group(2)) not in best or int(m.group(1)) > best[int(m.group(2))][0]):
            best[int(m.group(2))] = (int(m.group(1)), p)
    return {task: p for task, (_, p) in best.items()}


def read_log(p):
    text = p.read_text(errors='replace')
    start = re.search(r' start=(\S+)', text)
    end = re.search(r'status=(\d+) end=(\S+)', text)
    gpu = re.search(r'^(NVIDIA[^,\n]*)', text, re.M)
    wall = None
    if start and end:
        wall = (datetime.fromisoformat(end.group(2)) - datetime.fromisoformat(start.group(1))).total_seconds()
    return dict(missing_mol='molecule file missing' in text, worked='Starting optimization' in text,
                status=int(end.group(1)) if end else None, wall=wall, gpu=gpu.group(1) if gpu else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, default=None, help="default: out_dir of tasks/0.yaml")
    ap.add_argument('--log-dir', type=Path, default=Path('/scratch/mk8347/logs'))
    args = ap.parse_args()
    header, *lines = (HERE / 'INDEX.tsv').read_text().splitlines()
    tasks = [dict(zip(header.split('\t'), line.split('\t'))) for line in lines]
    out_dir = args.out_dir or Path(re.search(r'(?m)^out_dir: (\S+)', (HERE / 'tasks' / '0.yaml').read_text()).group(1))
    logs = newest_logs(args.log_dir)

    state, done_relax, want_relax = {}, 0, 0
    statuses, gpus, work_relax, work_wall, walls = collections.Counter(), collections.Counter(), 0, 0.0, []
    for t in tasks:
        i, want = int(t['task']), int(t['relaxations'])
        want_relax += want
        prog = out_dir / f"{t['run_name']}_progress.json"
        if not prog.exists():
            state[i] = 'not started'
        else:
            p = json.loads(prog.read_text())
            if p['num_samples'] != want:
                state[i] = f"other configuration (ran {p['num_samples']}, INDEX.tsv expects {want})"
            elif p['cursor'] >= p['num_samples'] and p['n_out'] == want:
                state[i] = 'done'
                done_relax += want
            else:
                state[i] = f"incomplete ({p['cursor']} of {p['num_samples']})"
        if i in logs:
            info = read_log(logs[i])
            if info['status'] is not None:
                statuses[info['status']] += 1
            if info['missing_mol']:
                statuses['exited at the molecule-file check'] += 1
                state[i] += '; newest log: molecule file missing'
            if state[i] == 'done' and info['worked'] and info['wall']:
                work_relax += want
                work_wall += info['wall']
                walls.append(info['wall'])
                gpus[info['gpu']] += 1

    kinds = collections.Counter(s.split(' (')[0].split(';')[0] for s in state.values())
    size = sum(f.stat().st_size for f in out_dir.glob('qm9full_c*.pt'))
    print(f'qm9_full_sep29: {len(tasks)} tasks in INDEX.tsv; outputs in {out_dir}')
    print('  ' + ', '.join(f'{k}: {v}' for k, v in sorted(kinds.items())))
    print(f'  crystals in finished runs: {done_relax:,} of {want_relax:,}; output .pt files {size / 1e9:.2f} GB')
    print(f'  newest log per task: exit statuses {dict(statuses)}; tasks with no log: {len(tasks) - len(logs)}')
    if walls:
        walls.sort()
        print(f'  finished tasks with a log: {len(walls)}; wall time median {walls[len(walls) // 2] / 60:.1f} min, '
              f'longest {walls[-1] / 60:.1f} min; {work_relax / work_wall:.1f} relaxations per s pooled; '
              f'GPUs {dict(gpus)}')
    bad = [(i, s) for i, s in state.items() if s != 'done']
    for i, s in bad[:40]:
        print(f"  task {i} ({tasks[i]['run_name']}): {s}")
    if len(bad) > 40:
        print(f'  ... {len(bad) - 40} more not done')


if __name__ == '__main__':
    main()

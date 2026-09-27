"""
Run-state safety for crystal_search: atomic writes, one writer per run, and clean stops between batches.

Writes. The output list and the progress file are each replaced atomically (written and fsynced to a temporary file
beside the target, then renamed over it), so a job killed mid-save, or a node that crashes, leaves the previous complete
file rather than a truncated one that fails every restart. The output is written before the progress file; a kill
between the two leaves one uncommitted batch at the end of the output list, which committed_outputs drops on resume so
the batch is redone rather than appended twice. On Windows a rename fails while another process holds the target open
(e.g. someone loading the outputs); the rename is retried for up to REPLACE_BUDGET_S and then raises.

Lease. RunLease allows one writer per run name. The owner id names the process: for a SLURM job the job id, its restart
count, host and pid, else host and pid. It is written to <stem>_owner.json, renewed before every save and kept fresh by a
heartbeat thread between saves (a batch may run longer than the staleness window). Another owner is refused while the
lease is fresh (by file modification times on the shared filesystem, never by comparing two machines' clocks) and may
take it over once stale; the same SLURM job after a requeue (a higher restart count) takes it over at once. A writer that
finds the lease taken over raises RunLeaseLost before writing anything. Nothing waits on another process: a start
either proceeds or raises.

Stops. StopRequest turns SIGUSR1 (e.g. #SBATCH --signal=USR1@900, sent before the walltime) and the presence of stop
files into a reason to leave the batch loop cleanly, with every completed batch saved; SIGTERM (SLURM's walltime or
scancel) unwinds the process with SystemExit so the lease is released on the way out.
"""
import json
import os
import signal
import socket
import sys
import threading
import time

import torch

REPLACE_BUDGET_S = 30.0


class RunLeaseLost(Exception):
    """Another process owns this run now; this one must stop without writing."""


def _replace(tmp, path, budget_s=REPLACE_BUDGET_S):
    """os.replace, retried on Windows while another process holds the target open; raises after budget_s."""
    delay, t_end = 0.05, time.monotonic() + budget_s
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if sys.platform != 'win32' or time.monotonic() > t_end:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 1.0)


def atomic_torch_save(obj, path):
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'wb') as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        _replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_json_dump(obj, path):
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w') as fh:
            json.dump(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        _replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_json(path):
    """The parsed file, or None if it is missing or unreadable (a file from before atomic writes may be partial)."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def committed_outputs(opt_outs, state):
    """The prefix of the output list that the progress state vouches for (state['n_out'] rows)."""
    n_out = state.get('n_out')
    if n_out is None:
        return opt_outs
    if len(opt_outs) < n_out:
        raise RuntimeError(f'output list has {len(opt_outs)} rows but the progress file records {n_out}: '
                           f'the output file is older than its progress file')
    return opt_outs[:n_out]


def owner_record():
    host, pid = socket.gethostname(), os.getpid()
    job = os.environ.get('SLURM_JOB_ID')
    if job:
        restart = int(os.environ.get('SLURM_RESTART_COUNT', '0') or 0)
        return dict(owner=f'slurm:{job}:{restart}:{host}:{pid}', job=job, restart=restart, host=host, pid=pid)
    return dict(owner=f'{host}:{pid}', job=None, restart=None, host=host, pid=pid)


def owner_id():
    return owner_record()['owner']


class RunLease:
    def __init__(self, stem, stale_after_s=1800.0, settle_s=2.0, heartbeat_s=None):
        self.path = f'{stem}_owner.json'
        self.record = owner_record()
        self.owner = self.record['owner']
        self.stale_after_s = float(stale_after_s)
        self.settle_s = float(settle_s)
        self.heartbeat_s = float(heartbeat_s) if heartbeat_s is not None else max(0.5, min(60.0, stale_after_s / 4))
        self._stop = threading.Event()
        self._thread = None

    def _age_s(self):
        """Seconds since the lease was last written or touched, on the filesystem's clock (a probe file written now)."""
        probe = f'{self.path}.{os.getpid()}.probe'
        with open(probe, 'w'):
            pass
        try:
            return os.path.getmtime(probe) - os.path.getmtime(self.path)
        finally:
            os.remove(probe)

    def acquire(self):
        held = read_json(self.path)
        if held is not None and held.get('owner') != self.owner:
            requeue = (self.record['job'] is not None and held.get('job') == self.record['job']
                       and (held.get('restart') or 0) < self.record['restart'])
            if requeue:
                print(f"taking over {self.path} from {held.get('owner')} (the same SLURM job, requeued)")
            else:
                age = self._age_s()
                if age < self.stale_after_s:
                    raise RunLeaseLost(f"{self.path}: run is held by {held.get('owner')} (renewed {age:.0f} s ago; "
                                       f"stale after {self.stale_after_s:.0f} s). Refusing to start a second writer.")
                print(f"taking over {self.path} from {held.get('owner')} (not renewed for {age:.0f} s)")
        self.renew()
        time.sleep(self.settle_s)  # two starters taking over one stale lease: the later write wins, the other leaves
        self.check()
        self._start_heartbeat()

    def renew(self):
        atomic_json_dump(self.record, self.path)

    def check(self):
        held = read_json(self.path)
        if held is None or held.get('owner') != self.owner:
            raise RunLeaseLost(f"{self.path}: lease now held by {None if held is None else held.get('owner')}, "
                               f"not {self.owner}; stopping without writing")
        self.renew()

    def _start_heartbeat(self):
        """Keep the lease fresh between saves: touch it every heartbeat_s while it still names this owner; stop as
        soon as it does not (the next check() then raises before any write)."""
        if self._thread is not None:
            return

        def beat():
            while not self._stop.wait(self.heartbeat_s):
                held = read_json(self.path)
                if held is None or held.get('owner') != self.owner:
                    return
                try:
                    os.utime(self.path)
                except OSError:
                    pass
        self._thread = threading.Thread(target=beat, name='run-lease-heartbeat', daemon=True)
        self._thread.start()

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        held = read_json(self.path)
        if held is not None and held.get('owner') == self.owner:
            os.remove(self.path)


class StopRequest:
    def __init__(self, stop_files=()):
        self.stop_files = [str(f) for f in stop_files]
        self.signalled = None
        self._previous = {}
        handlers = [(getattr(signal, 'SIGUSR1', None), self._handler), (signal.SIGTERM, self._terminate)]
        for sig, handler in handlers:
            if sig is None:  # no SIGUSR1 on Windows
                continue
            try:
                self._previous[sig] = signal.signal(sig, handler)
            except ValueError:  # not the main thread: signals cannot be caught here; stop files still work
                pass

    def _handler(self, signum, frame):
        self.signalled = signum

    def _terminate(self, signum, frame):
        raise SystemExit(128 + int(signum))  # unwind: the caller's finally releases the lease; saves are atomic

    def reason(self):
        if self.signalled is not None:
            return f'signal {self.signalled}'
        for f in self.stop_files:
            if os.path.exists(f):
                return f'stop file {f}'
        return None

    def close(self):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}

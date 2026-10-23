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
    """torch.save obj to path atomically: written, flushed and fsynced to <path>.<pid>.tmp, then renamed over path by
    _replace; the temporary file is removed if it still exists afterwards (the save or the rename raised)."""
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
    """json.dump obj to path atomically, in text mode, by the same temporary-file, fsync and rename steps as
    atomic_torch_save."""
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
    """The prefix of the output list that the progress state vouches for (state['n_out'] rows); opt_outs unchanged if
    state has no 'n_out' or it is None. Raises RuntimeError if opt_outs is shorter than n_out."""
    n_out = state.get('n_out')
    if n_out is None:
        return opt_outs
    if len(opt_outs) < n_out:
        raise RuntimeError(f'output list has {len(opt_outs)} rows but the progress file records {n_out}: '
                           f'the output file is older than its progress file')
    return opt_outs[:n_out]


def owner_record():
    """This process's lease record, a dict with owner, job, restart, host and pid. With SLURM_JOB_ID non-empty, owner is
    'slurm:<job>:<restart>:<host>:<pid>', job is that string and restart is the int SLURM_RESTART_COUNT (0 if unset
    or empty); otherwise owner is '<host>:<pid>' and job and restart are None."""
    host, pid = socket.gethostname(), os.getpid()
    job = os.environ.get('SLURM_JOB_ID')
    if job:
        restart = int(os.environ.get('SLURM_RESTART_COUNT', '0') or 0)
        return dict(owner=f'slurm:{job}:{restart}:{host}:{pid}', job=job, restart=restart, host=host, pid=pid)
    return dict(owner=f'{host}:{pid}', job=None, restart=None, host=host, pid=pid)


def owner_id():
    """The owner string of owner_record()."""
    return owner_record()['owner']


class RunLease:
    """One-writer lease for a run, held in the JSON file <stem>_owner.json and recording owner_record() (see the
    module docstring). stale_after_s is the lease age, in seconds of file modification time, from which another owner
    may take it over; settle_s is the pause in acquire() between writing the lease and re-checking it; heartbeat_s is
    the interval at which the heartbeat thread touches the lease, by default stale_after_s / 4 clipped to [0.5, 60] s.
    Constructing a lease does not touch the filesystem."""
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
        """Take the lease or raise RunLeaseLost. A lease file naming another owner is taken over at once if it is the
        same SLURM job with a lower restart count, taken over if its age (_age_s) is at least stale_after_s, and
        otherwise refused with RunLeaseLost; a missing or unreadable file, or one naming this owner, is taken without
        an age check. The record is then written, settle_s is slept, check() confirms (and renews) ownership, and the
        heartbeat thread is started."""
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
        """Write this process's record to the lease file with atomic_json_dump, without checking the current owner."""
        atomic_json_dump(self.record, self.path)

    def check(self):
        """Raise RunLeaseLost unless the lease file names this owner (a missing or unreadable file also raises);
        otherwise renew it."""
        held = read_json(self.path)
        if held is None or held.get('owner') != self.owner:
            raise RunLeaseLost(f"{self.path}: lease now held by {None if held is None else held.get('owner')}, "
                               f"not {self.owner}; stopping without writing")
        self.renew()

    def _start_heartbeat(self):
        """Keep the lease fresh between saves: touch it (os.utime; OSError ignored) every heartbeat_s while it still
        names this owner. The thread stops only when a read names another owner (the next check() then raises before
        any write) or release() sets the stop event; a read that finds the file missing or unreadable (e.g. while
        renew() replaces it) skips that touch and the thread keeps beating, leaving ownership to check(). Starts at
        most one daemon thread per lease; a stopped thread is not restarted."""
        if self._thread is not None:
            return

        def beat():
            while not self._stop.wait(self.heartbeat_s):
                held = read_json(self.path)
                if held is None:  # unreadable right now (e.g. mid-replace): keep beating; check() judges ownership
                    continue
                if held.get('owner') != self.owner:
                    return
                try:
                    os.utime(self.path)
                except OSError:
                    pass
        self._thread = threading.Thread(target=beat, name='run-lease-heartbeat', daemon=True)
        self._thread.start()

    def release(self):
        """Stop the heartbeat thread (joined for up to 5 s), then delete the lease file if it still names this
        owner."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        held = read_json(self.path)
        if held is not None and held.get('owner') == self.owner:
            os.remove(self.path)


class StopRequest:
    """Reasons to leave the batch loop (see the module docstring). Construction installs signal handlers: SIGUSR1,
    where the platform has it, records the signal number in signalled; SIGTERM raises SystemExit(128 + signum). The
    previous handlers are kept for close(). Outside the main thread signal.signal raises ValueError, which is
    ignored, so no handler is installed and only stop_files (stored as strings) apply."""
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
        """Record signum for reason()."""
        self.signalled = signum

    def _terminate(self, signum, frame):
        """Raise SystemExit(128 + signum)."""
        raise SystemExit(128 + int(signum))  # unwind: the caller's finally releases the lease; saves are atomic

    def reason(self):
        """'signal <signum>' once SIGUSR1 has been received, else 'stop file <path>' for the first of stop_files that
        exists, else None."""
        if self.signalled is not None:
            return f'signal {self.signalled}'
        for f in self.stop_files:
            if os.path.exists(f):
                return f'stop file {f}'
        return None

    def close(self):
        """Restore the signal handlers replaced at construction."""
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous = {}

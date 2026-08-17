"""
Shared pytest options, and the suite-wide GPU pre-flight.

`pytest_addoption` is only honoured in a conftest, not in a test module -- defining
it in the test file makes every test in that file ERROR on an unknown option rather
than skip, which is worse than not having the option at all.
"""
import os

import pytest

#: Free VRAM below which any GPU-touching test refuses to run.
#:
#: WHY THIS IS SESSION-WIDE AND NOT PER-FILE. A second CUDA consumer BSODs this box --
#: three times in 24 h, which is why energy_sampling/gpu_guard.py exists. That guard
#: protects train.py's entrypoint only. test_uma_gpu_real_batches.py carries its own
#: copy of the check, and that worked right up until the next GPU-touching thing was
#: written WITHOUT it and took the machine down (2026-08-14, an ad-hoc MACE timing
#: script that copied that file's batch construction and left its guard behind).
#:
#: A convention every new file has to remember is not a guard. This one is autouse:
#: a test opts IN to the GPU by requesting the `gpu` fixture, and gets the pre-flight
#: whether or not its author thought about it.
MIN_FREE_MB = 6000

SKIP_ENV = 'MXT_SKIP_GPU_PREFLIGHT'      # set to 1 only if you know the card fits both


def pytest_addoption(parser):
    parser.addoption(
        '--uma-checkpoint', action='store', default=None,
        help='Path to the UMA checkpoint. Without it (or $UMA_CHECKPOINT) the GPU '
             'MLIP tests in test_uma_gpu_real_batches.py skip, so CPU CI stays green.')


def gpu_preflight():
    """(ok, reason). Prefers the real guard -- it knows about other TRAINING
    PROCESSES, not merely free bytes -- and falls back to a free-VRAM floor when the
    GFN repo is not on the path, since this suite must also run standalone."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False, 'no GPU'
    if os.environ.get(SKIP_ENV, '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return True, f'{SKIP_ENV} set -- pre-flight bypassed'
    try:
        import gpu_guard
        ok, text = gpu_guard.describe()
        if not ok:
            return False, text
    except ImportError:
        pass                              # not on the GFN path; VRAM floor still applies
    except Exception as e:
        return False, f'gpu_guard raised: {e}'
    free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
    if free_mb < MIN_FREE_MB:
        return False, f'{free_mb:.0f} MiB free, need {MIN_FREE_MB}'
    return True, f'{free_mb:.0f} MiB free'


@pytest.fixture(scope='session')
def gpu():
    """Request this fixture from any test that touches CUDA. Skips -- never fails --
    so CPU CI and a busy card both stay green rather than red-for-the-wrong-reason."""
    ok, reason = gpu_preflight()
    if not ok:
        pytest.skip(f'GPU pre-flight refused: {reason}')
    return 'cuda'

"""
The retained-autograd-graph trap in offline bulk scoring, and the guard on it.

WHY THIS EXISTS. fairchem's _run_inference picks `torch.no_grad() if direct_forces
else nullcontext()` -- inverted from the obvious reading -- and our crystal
predictor sets direct_forces=False. So a scoring scan that differentiates nothing
still builds a graph per crystal, and `analyze(assign_outputs=True)` stores the
grad-carrying energies onto the batch, keeping those activations reachable.
Measured 2026-08-19 on real chunks: 100-250 MB retained PER CRYSTAL, zero under
no_grad. It cost a 122k-row job a cascading OOM at batch size 1.

It fails INVISIBLY, which is the real problem: the OOM handler shrinks the batch
and the shrunken size is sticky in `state`, so the symptom is a scan that quietly
runs slower forever rather than one that stops.
"""
import warnings

import pytest
import torch

from mxtaltools.common import adaptive_batching
from mxtaltools.common.adaptive_batching import adaptive_batched_analysis


class _Tiny:
    """The smallest thing the helper will drive: it only needs batch_to_list,
    analyze and .to()/.cpu()."""

    def __init__(self, rows):
        self.rows = rows

    def batch_to_list(self):
        return list(self.rows)

    def analyze(self, analyses, assign_outputs=True, **kwargs):
        return None

    def to(self, *_a, **_k):
        return self

    def cpu(self):
        return self


@pytest.fixture(autouse=True)
def _reset_warn_flag():
    """The warning is once-per-process by design, so a test that ran earlier would
    otherwise consume it and leave this one asserting nothing."""
    adaptive_batching._GRAD_WARNED = False
    yield
    adaptive_batching._GRAD_WARNED = False


def _collate(monkeypatch):
    monkeypatch.setattr(adaptive_batching, 'collate_data_list',
                        lambda rows, **kw: _Tiny(rows))


def test_warns_when_grad_is_enabled(monkeypatch):
    """The guard must fire on the configuration that silently costs 100-250 MB a
    row -- otherwise the next offline script repeats the same job-killing bug."""
    _collate(monkeypatch)
    with torch.enable_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        adaptive_batched_analysis(_Tiny([1, 2, 3]), 'noop', {}, device='cpu')
    messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
    assert any('no_grad' in m for m in messages), \
        f'no grad warning raised; got {messages}'


def test_silent_under_no_grad(monkeypatch):
    """The correct usage must not nag -- a warning that fires on every call is one
    everybody filters out, including on the run where it matters."""
    _collate(monkeypatch)
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        adaptive_batched_analysis(_Tiny([1, 2, 3]), 'noop', {}, device='cpu')
    assert not [w for w in caught if w.category is RuntimeWarning]


def test_warns_only_once(monkeypatch):
    """A bulk scan calls this hundreds of times; a per-call warning is noise."""
    _collate(monkeypatch)
    with torch.enable_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for _ in range(3):
            adaptive_batched_analysis(_Tiny([1]), 'noop', {}, device='cpu')
    assert len([w for w in caught if w.category is RuntimeWarning]) == 1


def test_cascading_oom_message_names_grad(monkeypatch):
    """The failure must name its own likeliest cause. One crystal does not exhaust
    a card; a few hundred retained graphs do, and the bare message read as 'this
    crystal is too big' -- which sent the first investigation the wrong way."""
    _collate(monkeypatch)

    def _always_oom(self, analyses, assign_outputs=True, **kwargs):
        raise RuntimeError('CUDA out of memory. Tried to allocate 2.00 MiB')

    monkeypatch.setattr(_Tiny, 'analyze', _always_oom)
    with torch.enable_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(RuntimeError, match='GRAD WAS ENABLED'):
            adaptive_batched_analysis(_Tiny([1, 2]), 'noop', {},
                                      initial_batch_size=1, oom_sleep=0,
                                      device='cpu')


# ------------------------------------------------------------------ the real thing

@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_grad_enabled_retains_memory_no_grad_does_not(gpu):
    """
    THE MEASUREMENT THE GUARD IS ABOUT, on real tensors rather than a stub.

    Re-introduces the bug and requires it to show: scoring under enable_grad must
    retain memory across iterations, and under no_grad must not. If a future change
    makes fairchem-style retention harmless, the first assertion fails and this
    test should be retired deliberately -- not silently kept as decoration.
    """
    lin = torch.nn.Linear(512, 512).to(gpu)

    def scan(grad_on, keep):
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        ctx = torch.enable_grad() if grad_on else torch.no_grad()
        with ctx:
            for _ in range(8):
                x = torch.randn(256, 512, device=gpu, requires_grad=grad_on)
                # stored like analyze(assign_outputs=True) stores energies: the
                # result stays reachable, so its graph does too
                keep.append(lin(x).sum())
        return torch.cuda.memory_allocated() - base

    kept_grad, kept_nograd = [], []
    grew_with_grad = scan(True, kept_grad)
    grew_without = scan(False, kept_nograd)

    assert grew_with_grad > 0, ('grad-enabled scoring retained nothing -- the trap '
                                'this guard exists for no longer reproduces')
    assert grew_without < grew_with_grad / 4, (
        f'no_grad retained {grew_without} B against {grew_with_grad} B with grad; '
        f'no_grad is supposed to make the retention go away')

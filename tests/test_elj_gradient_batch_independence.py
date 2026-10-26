"""A crystal's eLJ gradient with respect to its cell parameters does not depend on its batch-mates, on CPU, with
several threads (the setting a batched crystal search runs in).

The eLJ route gathers rows with repeated indices (each unit-cell atom into every image of it, each cluster atom into
every pair it is in, each crystal's cell into each of its atoms), so the backward sums several gradients into one row.
torch's CPU backward of ``x[index]`` adds those with parallel atomics once the gathered tensor holds 32768 elements
and more than one thread runs: the float32 sum order, and so the last bits of a gradient, changed from call to call
and with the batch, while the energy did not. A gradient component whose exact value is zero (float64: the length of
an axis of 73-148 A along which the crystal makes no contact, and the centroid along it) then changed sign between
batches, and the search's rprop, which steps by the sign, diverged (NEHZOR sg 14, 2026-10-26: 8 of 32 rows by step 1). The route now gathers through
``geometry_utils.gather_rows`` (``index_select`` on CPU, whose backward adds in index order).

Pinned here, on the four Z'=1 CSD rows of the characterization fixture F1 at a 10 A cutoff, at 4 threads with
deterministic algorithms off: each row's energy and gradient are bitwise the same alone, in the batch, in the batch
reversed, and in the batch again. The test asserts that its gathers cross the parallel threshold, so it cannot pass by
not exercising the path. CPU only, a few seconds.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

from mxtaltools.common.geometry_utils import gather_rows

REPO = Path(__file__).resolve().parents[1]
#: torch's CPU parallel grain: x[index]'s backward runs in parallel from this many gathered elements (torch 2.8)
PARALLEL_GRAIN = 32768
THREADS = 4
CUTOFF = 10.0


def _charlib():
    spec = importlib.util.spec_from_file_location('charlib_for_elj_gradient_batch_independence',
                                                  str(REPO / 'tests' / 'characterization' / 'charlib.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CL = _charlib()


@pytest.fixture(autouse=True, scope='module')
def _parallel_cpu():
    """THREADS threads, deterministic algorithms off (with them on, torch adds serially and hides the fault), float32;
    the previous settings come back on exit."""
    prev = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(), torch.get_default_dtype())
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(False)
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_num_threads(prev[0])
        torch.use_deterministic_algorithms(prev[1], warn_only=prev[2])
        torch.set_default_dtype(prev[3])


def _elj_and_grad(rows):
    """eLJ [n] and its gradient with respect to the cell parameters [n, 12] of a batch of ``rows``, the cell carrying
    its gradient into T_fc (box analysis on), as the crystal search differentiates it; and the number of intermolecular
    pairs."""
    batch = CL.collate(rows)
    p = batch.full_cell_parameters().detach().clone().requires_grad_(True)
    x = batch.clone().detach()
    x.set_cell_parameters(p)
    out, cluster = x.analyze(computes=['elj'], cutoff=CUTOFF, return_cluster=True)
    e = out['elj']
    g, = torch.autograd.grad(e.sum(), p)
    return e.detach(), g, cluster.edges_dict['intermolecular_dist'].shape[0]


def test_gather_rows_is_the_gather_with_a_reproducible_cpu_backward():
    gen = torch.Generator().manual_seed(0)
    n = PARALLEL_GRAIN // 3 + 1000                                      # above the grain, many repeats per row
    x0 = torch.randn(16, 3, generator=gen)
    index = torch.randint(0, 16, (n,), generator=gen)
    up = torch.randn(n, 3, generator=gen) * 100
    assert torch.equal(gather_rows(x0, index), x0[index])
    grads = []
    for _ in range(10):
        x = x0.clone().requires_grad_(True)
        (gather_rows(x, index) * up).sum().backward()
        grads.append(x.grad)
    assert all(torch.equal(grads[0], gr) for gr in grads[1:])


def test_each_rows_elj_gradient_is_the_same_alone_and_in_any_batch():
    rows = [CL.f1_row(r) for r in CL.fixtures()['F1']]
    assert torch.get_num_threads() == THREADS
    alone = [_elj_and_grad([r]) for r in rows]
    e_all, g_all, n_pairs = _elj_and_grad(rows)
    assert 3 * n_pairs >= PARALLEL_GRAIN, (f'the batch has {n_pairs} intermolecular pairs; its pair gather holds '
                                           f'{3 * n_pairs} elements, below the {PARALLEL_GRAIN} at which torch adds '
                                           f'in parallel, so this test does not exercise the path')
    e_rev, g_rev, _ = _elj_and_grad(rows[::-1])
    e_again, g_again, _ = _elj_and_grad(rows)
    for i, (e1, g1, _) in enumerate(alone):
        j = len(rows) - 1 - i
        for name, e, g in (('batch', e_all[i], g_all[i]), ('reversed batch', e_rev[j], g_rev[j]),
                           ('batch again', e_again[i], g_again[i])):
            assert torch.equal(e, e1[0]), f'row {i}: eLJ in the {name} differs from the row alone'
            assert torch.equal(g, g1[0]), (f'row {i}: eLJ gradient in the {name} differs from the row alone by '
                                           f'{float((g - g1[0]).abs().max()):.3g} (|grad| {float(g1[0].norm()):.3g})')

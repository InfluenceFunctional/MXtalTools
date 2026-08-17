"""
GPU: the hoisted MACE builder must give IDENTICAL ENERGIES to the list path.

WHY SEPARATE FROM test_mace_atomicdata_vectorisation.py. That file compares the INPUT
batch on CPU with no inference. It cannot catch a field the model reads that the
comparison does not cover, and it never launches a CUDA kernel. This one closes both
by running the real predictor and comparing the numbers the sampler trains on.

TWO HARD CONSTRAINTS ON THIS BOX, both discovered the expensive way (2026-08-14):

  BATCH SIZE. The acridine MACE model OOMs above ~5 crystals. MAX_GRAPHS is 4 and
  every test here is parametrised at or under it. This is NOT a tuning knob.

  KERNEL DURATION. The forward costs ~425 ms at 2 graphs and is close to fixed --
  which puts 8 graphs at ~1.7 s and 16 at ~3.4 s, across the 2 s Windows WDDM
  watchdog. Two runs of an earlier sweep BSOD'd the machine that way, with the card
  verified idle both times: it is a TDR timeout, not memory and not co-tenancy, so
  no pre-flight can prevent it. The only protection is staying small.

The suite therefore skips cleanly with no GPU, no checkpoint, or a busy card (the
`gpu` fixture in conftest.py runs the pre-flight), and is inert in CPU CI.

RUN:
    MACE_CHECKPOINT=/path/to/model.model pytest tests/test_mace_gpu_real_batches.py -q
"""
import os

import pytest
import torch

from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
MODEL_ENV = 'MACE_CHECKPOINT'

#: Never raise this without re-reading the module docstring. Both limits above bite.
MAX_GRAPHS = 4


@pytest.fixture(scope='module')
def model(gpu):
    path = os.environ.get(MODEL_ENV)
    if not path or not os.path.exists(path):
        pytest.skip(f'set {MODEL_ENV} to a MACE checkpoint to run the GPU tests')
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    return load_mace_model(path, gpu, torch.float32)


@pytest.fixture(scope='module')
def crystals(model):
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    allowed = set(int(z) for z in model.atomic_numbers)
    good = []
    for c in torch.load(DATASET, weights_only=False, map_location='cpu'):
        try:
            if not set(int(z) for z in c.z).issubset(allowed):
                continue
            p = collate_data_list([c.clone()])
            p.pose_aunit(std_orientation=False)
            p.build_unit_cell()
            good.append(c)
        except Exception:
            pass
    if not good:
        pytest.skip('no fixture crystal is buildable and in this model\'s table')
    return good


def _built(crystals, n, device):
    assert n <= MAX_GRAPHS, f'{n} graphs exceeds MAX_GRAPHS -- see the module docstring'
    pool = (crystals * (n // len(crystals) + 1))[:n]
    b = collate_data_list([c.clone() for c in pool]).to(device)
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    return b


def _energy(batch, model, hoisted):
    """One crystal-leg energy through the chosen builder, with the flag flipped at
    the module level -- the same switch production reads, so this exercises the
    shipping selection logic rather than calling the builders directly."""
    from mxtaltools.mlip_interfaces import AL_mace_utils as M
    prev = M.USE_HOISTED_MACE_ATOMICDATA
    M.USE_HOISTED_MACE_ATOMICDATA = hoisted
    try:
        return M.compute_crystal_mace_on_mxt_batch(
            batch, model, std_orientation=False, pbc=True).detach().float().cpu()
    finally:
        M.USE_HOISTED_MACE_ATOMICDATA = prev


@pytest.mark.parametrize('n', [1, 2])
def test_inputs_are_bit_identical_on_gpu(crystals, model, gpu, n):
    """The CPU file proves this on CPU; a device-dependent construction bug would
    only show here."""
    from mxtaltools.mlip_interfaces.AL_mace_utils import (
        verify_mace_atomicdata_equivalence)
    assert verify_mace_atomicdata_equivalence(_built(crystals, n, gpu), False,
                                              model, False)


def test_energies_match_within_model_nondeterminism(crystals, model, gpu):
    """
    A CONTROL COMPARISON, not torch.equal.

    MACE on GPU is not bit-reproducible against itself (tf32 and reduction order), so
    a bitwise assertion here would test the GPU's arithmetic, not this code. The
    honest bar: the cross-path spread must not exceed the SAME-path spread by more
    than a small factor. A construction bug puts the cross-path delta orders of
    magnitude above the control, not inside it.
    """
    b = _built(crystals, 2, gpu)
    a1 = _energy(b, model, hoisted=False)
    a2 = _energy(b, model, hoisted=False)          # the control
    h1 = _energy(b, model, hoisted=True)

    control = (a1 - a2).abs().max().item()
    cross = (a1 - h1).abs().max().item()
    scale = a1.abs().max().item()
    assert torch.isfinite(h1).all(), 'hoisted path produced non-finite energies'
    # absolute floor so a perfectly deterministic control (0.0) does not make the
    # bar unsatisfiable
    bar = max(10 * control, 1e-4 * max(scale, 1.0))
    assert cross <= bar, (
        f'cross-path |d| {cross:.3e} exceeds {bar:.3e} '
        f'(same-path control {control:.3e}, energy scale {scale:.3e}) -- '
        f'that is a construction difference, not nondeterminism')


def _energy_with(batch, model, hoisted, batched_nl):
    """As _energy, but also flipping the batched-neighbour-list switch."""
    from mxtaltools.mlip_interfaces import AL_mace_utils as M
    prev = M.USE_BATCHED_MACE_NEIGHBOURS
    M.USE_BATCHED_MACE_NEIGHBOURS = batched_nl
    try:
        return _energy(batch, model, hoisted)
    finally:
        M.USE_BATCHED_MACE_NEIGHBOURS = prev


@pytest.mark.parametrize('n', [1, 2])
def test_energies_match_with_batched_neighbours(crystals, model, gpu, n):
    """
    THE GATE ON MXT_BATCHED_MACE_NEIGHBOURS. Until this passes, that flag ships off.

    The CPU suite already proves the batched neighbour list yields the same EDGE SET
    as matscipy, on synthetic cells and on real crystals. That is necessary and not
    sufficient: it says nothing about whether the tensors handed to the model produce
    the same number. Only a real forward does.

    Same control comparison as test_energies_match_within_model_nondeterminism, and
    for the same reason -- MACE is not bit-reproducible against itself on GPU, so the
    bar is that the cross-path spread stays within the same-path spread, not equality.

    NOTE the edge ORDER genuinely differs between the two paths (matscipy's cell-list
    order vs the radius search's). MACE sums over edges, so this perturbs float
    summation order only. If this test fails by a wide margin it is a construction
    bug; if it fails by a hair, suspect the control bar rather than the neighbours.
    """
    b = _built(crystals, n, gpu)
    a1 = _energy_with(b, model, hoisted=True, batched_nl=False)
    a2 = _energy_with(b, model, hoisted=True, batched_nl=False)     # control
    h1 = _energy_with(b, model, hoisted=True, batched_nl=True)

    control = (a1 - a2).abs().max().item()
    cross = (a1 - h1).abs().max().item()
    scale = a1.abs().max().item()
    assert torch.isfinite(h1).all(), 'batched neighbours produced non-finite energies'
    bar = max(10 * control, 1e-4 * max(scale, 1.0))
    assert cross <= bar, (
        f'cross-path |d| {cross:.3e} exceeds {bar:.3e} (same-path control '
        f'{control:.3e}, scale {scale:.3e}) -- the batched neighbour list changes '
        f'the energy by more than summation order explains')


def test_batched_neighbour_count_matches(crystals, model, gpu):
    """
    A DIAGNOSTIC to run alongside the energy test: equal edge COUNTS per graph.

    If the energy test fails, this says immediately whether the neighbour list is
    wrong (counts differ) or whether the energy moved for some other reason (counts
    match). Cheap, and it turns a one-bit failure into a direction.
    """
    from mxtaltools.mlip_interfaces import AL_mace_utils as M
    b = _built(crystals, 2, gpu)
    prev = M.USE_BATCHED_MACE_NEIGHBOURS
    try:
        M.USE_BATCHED_MACE_NEIGHBOURS = False
        ref = M.batch_to_mace_atomicdata_hoisted(b, False, model, False)
        M.USE_BATCHED_MACE_NEIGHBOURS = True
        new = M.batch_to_mace_atomicdata_hoisted(b, False, model, False)
    finally:
        M.USE_BATCHED_MACE_NEIGHBOURS = prev
    for g, (a, c) in enumerate(zip(ref, new)):
        assert a.edge_index.shape == c.edge_index.shape, (
            f'graph {g}: {a.edge_index.shape[1]} edges from matscipy vs '
            f'{c.edge_index.shape[1]} from the batched list')


def test_hoisted_is_not_slower(crystals, model, gpu):
    """Sanity, not a benchmark: 2 graphs is far too small to show the host-side win
    (the fixed ~425 ms forward dominates), so this only asserts it did not REGRESS."""
    import time
    b = _built(crystals, 2, gpu)
    for hoisted in (False, True):
        _energy(b, model, hoisted)                 # warm
    t = {}
    for hoisted in (False, True):
        t0 = time.perf_counter()
        _energy(b, model, hoisted)
        torch.cuda.synchronize()
        t[hoisted] = time.perf_counter() - t0
    assert t[True] <= t[False] * 1.5, (
        f'hoisted {t[True]*1e3:.0f} ms vs list {t[False]*1e3:.0f} ms at 2 graphs')

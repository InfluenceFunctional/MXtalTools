"""
GPU tests: the vectorised AtomicData builder must give IDENTICAL UMA ENERGIES to
the list path, on batches that look like real training traffic.

WHY THIS FILE EXISTS SEPARATELY FROM test_uma_atomicdata_vectorisation.py. That file
compares the INPUT batch, on CPU, with no checkpoint. It cannot catch a field the
model reads that the comparison does not cover, and it never exercises a CUDA kernel.
This one closes both by running the actual predictor and comparing the numbers the
sampler would train on.

THE THREE BATCH FAMILIES, chosen because they stress different things:

  reference   -- real crystals from the CSD test set. Physical cells, sane densities,
                 the regime the energies are calibrated against.
  perturbed   -- the same crystals with small latent perturbations. Still physical,
                 but no longer exactly the dataset geometry: this is what most of a
                 training batch looks like once the policy is near the data.
  random      -- crystals from uniform random latents. These are TRASH by design:
                 clashing atoms, degenerate cells, oblique boxes. They exercise the
                 max_cp density guard, fairchem's large-box/PBC fallback, and the
                 numerically nastiest energies -- exactly where a construction bug
                 would be most likely to show up and least likely to be noticed.

THE BAR, and why it is not bitwise. Bitwise equality IS the bar for the input
tensors, and it holds on GPU (test_inputs_are_bit_identical_on_gpu). It is NOT a
usable bar for the energies, because measurement showed UMA is not bit-reproducible
against itself:

    same path, run twice   4.79e-3 eV     <- the control
    list path, run twice   2.86e-3 eV
    vectorised vs list     4.15e-3 eV     <- SMALLER than the control
    energy scale           ~837 eV

Re-running one path disagrees with itself by MORE than the two paths disagree with
each other, so a `torch.equal` assertion here tests the GPU's reduction order, not
this code. The honest test is therefore a CONTROL COMPARISON: the cross-path spread
must not exceed the same-path spread by more than a small factor. That is a real
falsifiable claim -- a construction bug would put the cross-path delta orders of
magnitude above the control, not inside it.

TF32 is the source: with tf32=False the same-path spread drops to 5.3e-5 eV and the
cross-path to 4.6e-5, both ~100x tighter. That is a property of the shipping
configuration worth knowing on its own -- it is a noise floor under every reward
(~0.1 kJ/mol after the x96.485/sym_mult conversion), and it is what the hosted
gas-phase reference removes one copy of.

RUN:
    pytest tests/test_uma_gpu_real_batches.py -q \
        --uma-checkpoint /scratch/mk8347/models/uma/esen_s.pt

Skips cleanly with no GPU or no checkpoint, so it is inert in CPU CI.
"""
import os

import pytest
import torch

from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')


#: --uma-checkpoint is declared in tests/conftest.py -- pytest_addoption is ignored
#: in a test module, and defining it here made every test ERROR instead of skip.


#: Refuse to start if another job holds the card. These tests load a full MLIP and
#: run real batches, and on this box a second CUDA consumer BSODs the machine --
#: three times in 24 h, which is why energy_sampling/gpu_guard.py exists. That guard
#: protects train.py's entrypoint; nothing protected pytest, and this suite duly
#: took the box down. DEFAULT-DENY, same policy as the guard: anything on the GPU
#: means do not add to it.
MIN_FREE_MB = 6000


def _gpu_is_free():
    """(ok, reason). Prefers the real guard; falls back to free VRAM if the GFN repo
    is not importable, since this file must also work standalone in mxtaltools."""
    try:
        import gpu_guard  # noqa: F401 -- energy_sampling/, only on the GFN PYTHONPATH
        others = gpu_guard.training_processes()
        if others:
            return False, f'another training run holds the GPU: {others[0][1][:70]}'
        mem = gpu_guard.gpu_memory()
        if mem and mem[1] < MIN_FREE_MB:
            return False, f'only {mem[1]} MB free (need {MIN_FREE_MB}), util {mem[3]}%'
        return True, ''
    except ImportError:
        free, total = torch.cuda.mem_get_info(0)
        free_mb = free // (1024 * 1024)
        if free_mb < MIN_FREE_MB:
            return False, f'only {free_mb} MB free (need {MIN_FREE_MB})'
        return True, ''


@pytest.fixture(scope='module')
def device():
    # device_count as well as is_available: under CUDA_VISIBLE_DEVICES="" torch can
    # still report is_available() True while exposing zero devices, and the tests
    # then fail at torch.load instead of skipping
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip('no GPU')
    ok, reason = _gpu_is_free()
    if not ok:
        pytest.skip(f'GPU is in use -- refusing to co-tenant ({reason}). '
                    f'Set GFN_ALLOW_GPU_SHARING=1 only if you know the card fits both.')
    return 'cuda'


@pytest.fixture(scope='module')
def checkpoint_path(request):
    path = request.config.getoption('--uma-checkpoint') or os.environ.get('UMA_CHECKPOINT')
    if not path or not os.path.exists(path):
        pytest.skip('pass --uma-checkpoint (or set UMA_CHECKPOINT) to run the GPU tests')
    return path


@pytest.fixture(scope='module')
def predictor(checkpoint_path, device):
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    return init_uma_crystal_predictor(checkpoint_path, device=device)


@pytest.fixture(scope='module')
def crystals(device):
    """
    Buildable entries only.

    5 of the 100 (indices 34/44/51/52/59, all Z'=2) raise inside mxtaltools'
    aunit2ucell -- `einsum(): subscript j has size 7 ... does not broadcast with
    previously seen size 4` -- when their unit cell is built. PRE-EXISTING and
    unrelated to anything here (it is the fixture, before this code runs), but a
    batch large enough to include one takes the whole test down, which is how the
    64-crystal timing test failed. Filtered rather than worked around, and worth
    reporting upstream on its own.
    """
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    # load to CPU and let each family .to(device): map_location=device would fail
    # before the skip can fire on a box with no usable GPU
    raw = torch.load(DATASET, weights_only=False, map_location='cpu')
    good = []
    for c in raw:
        try:
            probe = collate_data_list([c.clone()])
            probe.pose_aunit(std_orientation=False)
            probe.build_unit_cell()
            good.append(c)
        except Exception:
            pass
    if len(good) < 12:
        pytest.skip(f'only {len(good)} buildable crystals in the fixture')
    return good


# ----------------------------------------------------------------- batch families

def _reference(crystals, n, device):
    """Real CSD crystals, untouched."""
    b = collate_data_list(crystals[:n]).to(device)
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    return b


def _perturbed(crystals, n, device, scale=0.02, seed=0):
    """The same crystals, small latent perturbation. Still physical."""
    torch.manual_seed(seed)
    b = collate_data_list(crystals[:n]).to(device)
    b.cell_lengths = b.cell_lengths * (1 + scale * torch.randn_like(b.cell_lengths))
    b.cell_angles = b.cell_angles + scale * torch.randn_like(b.cell_angles)
    b.aunit_centroid = (b.aunit_centroid
                        + scale * torch.randn_like(b.aunit_centroid)).clip(0.01, 0.99)
    b.box_analysis()
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    return b


def _random(crystals, n, device, seed=0):
    """Uniform random latents -- physically meaningless cells, on purpose."""
    torch.manual_seed(seed + 1)
    b = collate_data_list(crystals[:n]).to(device)
    b.cell_lengths = torch.rand_like(b.cell_lengths) * 6 * b.radius[:, None] + 3
    b.cell_angles = torch.rand_like(b.cell_angles) * (torch.pi * 0.6) + torch.pi * 0.2
    b.aunit_centroid = torch.rand_like(b.aunit_centroid).clip(0.01, 0.99)
    b.aunit_orientation = torch.randn_like(b.aunit_orientation)
    b.box_analysis()
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    return b


FAMILIES = {'reference': _reference, 'perturbed': _perturbed, 'random': _random}


def _energy_both_ways(batch, predictor):
    """Run the predictor through each construction path and return both energies."""
    from mxtaltools.mlip_interfaces import uma_utils as U

    saved = U.USE_VECTORISED_ATOMICDATA
    try:
        U.USE_VECTORISED_ATOMICDATA = True
        vec = U.compute_crystal_uma_on_mxt_batch(batch.clone(), False, predictor).detach()
        U.USE_VECTORISED_ATOMICDATA = False
        lst = U.compute_crystal_uma_on_mxt_batch(batch.clone(), False, predictor).detach()
    finally:
        U.USE_VECTORISED_ATOMICDATA = saved
    return vec, lst


@pytest.mark.parametrize('family', list(FAMILIES))
def test_inputs_are_bit_identical_on_gpu(family, crystals, predictor, device):
    """Bitwise, on device. This is where an exact bar belongs -- construction is a
    reindexing, so anything but equality is a bug."""
    from mxtaltools.mlip_interfaces import uma_utils as U
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    batch = FAMILIES[family](crystals, 12, device)
    vec = U.batch_to_fairchem_batch(batch, False)
    lst = atomicdata_list_to_batch(U.batch_to_fairchem_atomicdata(batch, False))
    for field in ('pos', 'atomic_numbers', 'cell', 'pbc', 'natoms', 'batch'):
        a, b = getattr(vec, field), getattr(lst, field)
        assert a.dtype == b.dtype and torch.equal(a, b), f'{family}: {field}'


@pytest.mark.parametrize('family', list(FAMILIES))
def test_energy_delta_is_within_model_nondeterminism(family, crystals, predictor, device):
    """
    THE HEADLINE CLAIM, stated against a control.

    UMA does not reproduce itself bitwise on GPU, so "vectorised == list" cannot be
    asserted exactly. What CAN be asserted is that swapping construction paths adds
    no more variation than re-running one path does. The control is measured in the
    same test on the same batch, so it cannot drift away from the claim.
    """
    batch = FAMILIES[family](crystals, 12, device)
    a1, a2 = _energy_both_ways(batch, predictor)[0], _energy_both_ways(batch, predictor)[0]
    control = (a1 - a2).abs().max().item()          # same path, twice
    vec, lst = _energy_both_ways(batch, predictor)
    cross = (vec - lst).abs().max().item()          # across paths
    scale = vec.abs().mean().item()
    print(f'\n{family}: control {control:.3e}  cross-path {cross:.3e}  '
          f'scale {scale:.1f} eV  (rel {cross / scale:.2e})')
    # generous factor: the control is itself a noisy 12-sample max. A construction
    # bug would be orders of magnitude out, not a factor of 4.
    assert cross <= max(control * 4.0, 1e-6 * scale), (
        f'{family}: cross-path delta {cross:.3e} eV exceeds model nondeterminism '
        f'(control {control:.3e}) -- this is a real difference, not float noise')


@pytest.mark.parametrize('family', list(FAMILIES))
def test_energies_are_finite(family, crystals, predictor, device):
    """Trash cells are allowed to be high-energy; they are NOT allowed to be NaN.
    A non-finite reward propagates into log_reward and poisons the TB loss."""
    batch = FAMILIES[family](crystals, 12, device)
    vec, _ = _energy_both_ways(batch, predictor)
    assert torch.isfinite(vec).all(), f'{family}: {(~torch.isfinite(vec)).sum()} non-finite'


def test_random_cells_are_actually_worse(crystals, predictor, device):
    """Sanity on the FIXTURE, not the code: if random latents scored like reference
    crystals, the 'trash' family would be testing nothing. Guards against a builder
    that silently ignores the perturbation."""
    ref, _ = _energy_both_ways(_reference(crystals, 12, device), predictor)
    rnd, _ = _energy_both_ways(_random(crystals, 12, device), predictor)
    assert rnd.mean() > ref.mean(), (
        f'random cells scored {rnd.mean():.2f} vs reference {ref.mean():.2f} -- '
        f'the random fixture is not producing bad crystals')


def test_lattice_energy_matches_with_hosted_gas_reference(crystals, predictor, device):
    """
    The gas-phase hosting, end to end: a pre-set uma_gas_pot must reproduce the
    lattice energy the full two-leg path computes, to within UMA's own
    rotation-invariance error.

    Reports the drift in kJ/mol rather than asserting a tight bound, because that
    drift IS the quantity of interest -- it is the per-sample noise hosting removes
    from every reward, and its size is what decides whether hosting is a speedup or
    a correctness fix. The assertion is deliberately loose (1 kJ/mol against lattice
    energies of order 100); tighten it once the number is known on a real molecule set.
    """
    batch = _reference(crystals, 8, device)
    full = batch.clone().compute_lattice_uma(predictor, std_orientation=False).detach()

    hosted = batch.clone()
    gas = hosted.clone().compute_lattice_gas_phase_uma(predictor).detach()
    hosted.add_graph_attr(gas, 'uma_gas_pot')
    got = hosted.compute_lattice_uma(predictor, std_orientation=False).detach()

    drift = (got - full).abs()
    print(f'\nhosted-gas drift: mean {drift.mean():.4f} max {drift.max():.4f} kJ/mol')
    assert drift.max() < 1.0, (
        f'hosted gas reference drifts {drift.max():.3f} kJ/mol -- larger than expected '
        f'for a rotation-invariant quantity; run gas_reference_audit before enabling')


def test_tf32_is_the_nondeterminism_source(crystals, predictor, checkpoint_path, device):
    """
    Records a property of the SHIPPING CONFIG, not of this change: UMA energies are
    not reproducible run-to-run, and tf32 is why.

    This is a noise floor under every reward (x96.485/sym_mult puts ~5e-3 eV at
    ~0.1 kJ/mol against lattice energies of order 100), so it is worth a standing
    measurement rather than folklore. Not an assertion about which is 'right' --
    tf32=True is a deliberate speed choice -- just a pin on the size of it, so a
    future change that makes it worse is visible.
    """
    from mxtaltools.mlip_interfaces import uma_utils as U

    batch = _reference(crystals, 12, device)
    a1, a2 = _energy_both_ways(batch, predictor)[0], _energy_both_ways(batch, predictor)[0]
    tf32_spread = (a1 - a2).abs().max().item()

    settings = U.crystal_inference_settings()
    settings.tf32 = False
    fp32 = U._build_uma_crystal_predictor(checkpoint_path, device, settings)
    b1, b2 = _energy_both_ways(batch, fp32)[0], _energy_both_ways(batch, fp32)[0]
    fp32_spread = (b1 - b2).abs().max().item()

    print(f'\nrun-to-run spread: tf32 {tf32_spread:.3e} eV   fp32 {fp32_spread:.3e} eV '
          f'({tf32_spread / max(fp32_spread, 1e-12):.0f}x)')
    assert fp32_spread < tf32_spread, 'expected fp32 to be tighter than tf32'


@pytest.mark.parametrize('family', ['reference', 'random'])
def test_vectorised_is_faster_on_gpu(family, crystals, predictor, device):
    """Not a correctness test -- a regression guard on the whole point of the change.
    Times construction only (no predictor), with a CUDA sync so the number is real."""
    import time
    from mxtaltools.mlip_interfaces import uma_utils as U
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

    batch = FAMILIES[family](crystals, 64, device)

    def timed(fn):
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 3

    t_vec = timed(lambda: U.batch_to_fairchem_batch(batch, False))
    t_lst = timed(lambda: atomicdata_list_to_batch(
        U.batch_to_fairchem_atomicdata(batch, False)))
    print(f'\n{family}: list {t_lst*1e3:.1f} ms  vectorised {t_vec*1e3:.2f} ms  '
          f'({t_lst/t_vec:.0f}x)')
    assert t_vec < t_lst, 'vectorised construction is not faster on GPU'

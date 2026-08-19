"""
Gates on USE_UMA_EXTERNAL_GRAPH: our neighbour list handed to fairchem must be the
graph fairchem would have built -- where fairchem's own graph is right -- and the
energies and grads through a real UMA forward must sit inside the nondeterminism
floor.

WHAT IS AND IS NOT CLAIMED. The external path is exact; fairchem's internal
radius_graph_pbc_v2 is not, in two documented ways (see USE_UMA_EXTERNAL_GRAPH in
uma_utils.py): it assumes atoms are wrapped inside the cell (unit_cell_pos is not),
and it truncates at max_neighbors=300 (degenerate cells reach ~2710). So the honest
gates are:

  * CONVENTION, on wrapped positions where both are exact: identical edge sets,
    as {(source, target, offset)} -- the mapping between our (i, j, S) and
    fairchem's [source, target] + cell_offsets is exactly what this pins.
  * CONTAINMENT, on real unwrapped positions: fairchem's edges are a subset of
    ours. Every edge it finds is real, so anything outside our set is OUR bug;
    anything of ours it misses is its wrapping assumption, reported not hidden.
  * ENERGY and GRADS through the forward, on physical cells (where the edge sets
    agree): control comparison against same-path repeat spread, the established
    bar for UMA on GPU (test_uma_gpu_real_batches.py explains why torch.equal
    would test the GPU, not the code).

RUN:
    pytest tests/test_uma_external_graph.py -q \
        --uma-checkpoint D:/crystal_datasets/esen_s.pt
"""
import os

import pytest
import torch

from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
MIN_FREE_MB = 6000
CUTOFF = 6.0     # CPU convention tests only; the GPU tests read the model's own


@pytest.fixture(scope='module')
def crystals():
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
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
    if len(good) < 8:
        pytest.skip(f'only {len(good)} buildable crystals in the fixture')
    return good


def _built(crystals, n, device='cpu'):
    b = collate_data_list([c.clone() for c in crystals[:n]]).to(device)
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    return b


def _fairchem_data(uma_batch):
    """The minimal duck-typed object generate_graph reads."""
    class _D:
        pass
    d = _D()
    d.pos = uma_batch.pos
    d.cell = uma_batch.cell
    d.natoms = uma_batch.natoms
    d.batch = uma_batch.batch
    d.pbc = uma_batch.pbc
    return d


def _fairchem_edge_set(uma_batch, cutoff, max_neighbors=100000):
    """{(source, target, Sx, Sy, Sz)} from fairchem's own generator, untruncated by
    default so the comparison is against its geometry, not its neighbour cap."""
    from fairchem.core.graph.compute import generate_graph
    g = generate_graph(_fairchem_data(uma_batch), cutoff=cutoff,
                       max_neighbors=max_neighbors,
                       enforce_max_neighbors_strictly=False,
                       radius_pbc_version=2,
                       pbc=torch.ones(len(uma_batch.natoms), 3, dtype=torch.bool))
    ei = g['edge_index'].cpu().numpy()
    off = g['cell_offsets'].cpu().numpy().round().astype(int)
    return {(int(ei[0, k]), int(ei[1, k]), *map(int, off[k]))
            for k in range(ei.shape[1])}


def _our_edge_set(uma_batch, cutoff, pbc=True):
    from mxtaltools.mlip_interfaces.uma_utils import attach_external_graph
    b = uma_batch.clone() if hasattr(uma_batch, 'clone') else uma_batch
    attach_external_graph(b, cutoff, pbc)
    ei = b.edge_index.cpu().numpy()
    off = b.cell_offsets.cpu().numpy().round().astype(int)
    return {(int(ei[0, k]), int(ei[1, k]), *map(int, off[k]))
            for k in range(ei.shape[1])}


def _uma_batch(batch):
    from mxtaltools.mlip_interfaces.uma_utils import batch_to_fairchem_batch
    return batch_to_fairchem_batch(batch, False, True, False)


@pytest.mark.parametrize('n', [1, 4, 9])
def test_convention_matches_fairchem_on_wrapped_positions(crystals, n):
    """
    On positions wrapped into the cell -- the case fairchem's generator documents
    itself correct for -- the two edge sets must be IDENTICAL, offsets included.
    This is the whole (source, target, offset-sign) convention mapping in one
    assertion: any transposition or sign error moves every periodic edge.
    """
    from mxtaltools.mlip_interfaces.pbc_neighbours import _wrap

    ub = _uma_batch(_built(crystals, n))
    pos_w, _ = _wrap(ub.pos, ub.batch, ub.cell)
    ub.pos = pos_w

    theirs = _fairchem_edge_set(ub, CUTOFF)
    ours = _our_edge_set(ub, CUTOFF)
    assert ours == theirs, (
        f'{len(ours - theirs)} edges only ours, {len(theirs - ours)} only fairchem, '
        f'of {len(theirs)} -- the convention mapping is wrong')


@pytest.mark.parametrize('n', [4, 9])
def test_fairchem_edges_are_a_subset_on_real_positions(crystals, n):
    """
    On REAL (unwrapped) unit cells: every edge fairchem finds must be in our set --
    an edge of theirs we lack is our bug, full stop. Edges of ours they lack are
    their wrapping assumption biting on real data; counted and printed so the size
    of the correctness change is on the record, not asserted away.
    """
    ub = _uma_batch(_built(crystals, n))
    theirs = _fairchem_edge_set(ub, CUTOFF)
    ours = _our_edge_set(ub, CUTOFF)
    missing_from_ours = theirs - ours
    assert not missing_from_ours, (
        f'{len(missing_from_ours)} fairchem edges are absent from our list -- '
        f'that direction is unambiguously our bug')
    extra = ours - theirs
    print(f'\n[external-graph] n={n}: fairchem missed {len(extra)} of {len(ours)} '
          f'edges on unwrapped positions ({100 * len(extra) / max(len(ours), 1):.2f}%)')


# ------------------------------------------------------------------ GPU energy gate

def _gpu_ok():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False, 'no GPU'
    free, _ = torch.cuda.mem_get_info(0)
    if free // (1024 * 1024) < MIN_FREE_MB:
        return False, f'only {free // (1024 * 1024)} MB free'
    return True, ''


@pytest.fixture(scope='module')
def device():
    ok, reason = _gpu_ok()
    if not ok:
        pytest.skip(reason)
    return 'cuda'


@pytest.fixture(scope='module')
def checkpoint_path(request):
    path = request.config.getoption('--uma-checkpoint') or os.environ.get('UMA_CHECKPOINT')
    if not path or not os.path.exists(path):
        pytest.skip('pass --uma-checkpoint (or set UMA_CHECKPOINT) to run the GPU tests')
    return path


@pytest.fixture(scope='module')
def predictors(checkpoint_path, device):
    """(internal otf, external graph) predictors from the same checkpoint."""
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    from mxtaltools.mlip_interfaces.uma_utils import crystal_inference_settings
    internal = init_uma_crystal_predictor(
        checkpoint_path, device=device,
        inference_settings=crystal_inference_settings(external_graph_gen=False))
    external = init_uma_crystal_predictor(
        checkpoint_path, device=device,
        inference_settings=crystal_inference_settings(external_graph_gen=True))
    return internal, external


def _energy(batch, predictor, keep_graph=False):
    from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
    e = compute_crystal_uma_on_mxt_batch(batch, False, predictor)
    return e if keep_graph else e.detach().float().cpu()


def _wrap_ucell(b):
    """Wrap unit_cell_pos into the cell, per graph. A per-atom lattice translation:
    the crystal is unchanged, but fairchem's internal generator -- which assumes
    wrapped positions -- becomes exact, so on a wrapped batch BOTH paths are exact
    and directly comparable."""
    from mxtaltools.mlip_interfaces.pbc_neighbours import _wrap
    cells = b.T_fc.transpose(1, 2)
    pos_w, _ = _wrap(b.unit_cell_pos.detach(), b.unit_cell_batch, cells)
    b.unit_cell_pos = pos_w
    return b


def test_energies_match_where_both_graphs_are_exact(crystals, predictors, device):
    """
    THE GATE, on wrapped positions -- the case where the edge sets are proven
    identical above, so the energies must sit inside the same-path nondeterminism
    spread. On UNWRAPPED cells the two legitimately differ (fairchem drops 0.4-0.6%
    of edges there; see test_external_graph_is_wrap_invariant, which pins which
    side of that difference is correct).
    """
    internal, external = predictors
    b = _wrap_ucell(_built(crystals, 8, device))
    a1 = _energy(b.clone(), internal)
    a2 = _energy(b.clone(), internal)          # the control
    x1 = _energy(b.clone(), external)

    control = (a1 - a2).abs().max().item()
    cross = (a1 - x1).abs().max().item()
    scale = a1.abs().max().item()
    assert torch.isfinite(x1).all(), 'external graph produced non-finite energies'
    bar = max(10 * control, 1e-4 * max(scale, 1.0))
    assert cross <= bar, (
        f'external-graph |d| {cross:.3e} exceeds {bar:.3e} (same-path control '
        f'{control:.3e}, scale {scale:.3e}) -- the supplied graph changes the '
        f'energy by more than nondeterminism explains')


def test_external_graph_is_wrap_invariant(crystals, predictors, device):
    """
    THE DISCRIMINATOR. Wrapping atoms by lattice vectors is a symmetry of the
    crystal: a correct energy cannot change under it. The external path must be
    invariant (its builder wraps internally and corrects the shifts exactly).
    The internal path's wrapped-vs-unwrapped delta is the measured size of the
    edge-dropping bug the external graph fixes -- printed for the record, since
    it is fairchem's defect, not this code's, and need not be asserted.
    """
    internal, external = predictors
    raw = _built(crystals, 8, device)
    wrapped = _wrap_ucell(raw.clone())

    x_raw = _energy(raw.clone(), external)
    x_wrap = _energy(wrapped.clone(), external)
    i_ctrl = (_energy(wrapped.clone(), internal)
              - _energy(wrapped.clone(), internal)).abs().max().item()
    i_raw = _energy(raw.clone(), internal)
    i_wrap = _energy(wrapped.clone(), internal)

    scale = x_wrap.abs().max().item()
    bar = max(10 * i_ctrl, 1e-4 * max(scale, 1.0))
    x_delta = (x_raw - x_wrap).abs().max().item()
    i_delta = (i_raw - i_wrap).abs().max().item()
    print(f'\n[wrap-invariance] external delta {x_delta:.3e}, internal delta '
          f'{i_delta:.3e}, nondeterminism control {i_ctrl:.3e}, scale {scale:.3e}')
    assert x_delta <= bar, (
        f'external path is not wrap-invariant: |d| {x_delta:.3e} > {bar:.3e} -- '
        f'that would be OUR bug, not fairchem\'s')


def test_grads_match_where_both_graphs_are_exact(crystals, predictors, device):
    """
    The sampler trains on gradients. The external branch recomputes distances
    differentiably from pos and cell, so grads flow by construction -- this pins
    that they flow to the SAME values, inside the same-path backward spread.
    Wrapped positions, same reason as the energy gate.
    """
    internal, external = predictors

    def grads(pred):
        b = _wrap_ucell(_built(crystals, 4, device))
        b.unit_cell_pos.requires_grad_(True)
        b.T_fc.requires_grad_(True)
        e = _energy(b, pred, keep_graph=True)
        g_pos, g_cell = torch.autograd.grad(
            e.sum(), (b.unit_cell_pos, b.T_fc), allow_unused=True)
        assert g_pos is not None, 'no gradient reached unit_cell_pos'
        assert g_cell is not None, 'no gradient reached T_fc'
        return g_pos.detach().float().cpu(), g_cell.detach().float().cpu()

    r1p, r1c = grads(internal)
    r2p, r2c = grads(internal)                 # the control
    xp, xc = grads(external)

    for name, r1, r2, nw in (('unit_cell_pos', r1p, r2p, xp),
                             ('T_fc', r1c, r2c, xc)):
        control = (r1 - r2).abs().max().item()
        cross = (r1 - nw).abs().max().item()
        scale = r1.abs().max().item()
        assert torch.isfinite(nw).all(), f'non-finite grad on {name}'
        bar = max(10 * control, 1e-4 * max(scale, 1.0))
        assert cross <= bar, (
            f'grad on {name} differs by {cross:.3e} > {bar:.3e} '
            f'(same-path control {control:.3e}, scale {scale:.3e})')


def test_gas_leg_runs_through_external_predictor(crystals, predictors, device):
    """
    The isolated-molecule leg reaches the CRYSTAL predictor with pbc=False
    (crystal_analysis.compute_lattice_gas_phase_uma), and with external_graph_gen
    the model asserts edges are present on EVERY call -- so the pbc=False leg must
    get them too, through the same attach. Finite energies inside the control bar.
    """
    from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
    internal, external = predictors

    def gas(pred):
        b = _built(crystals, 4, device)
        b.reset_sg_info(sg_ind=1)
        b.box_analysis()
        return compute_crystal_uma_on_mxt_batch(
            b, False, pred, pbc=False, force_rebuild=True).detach().float().cpu()

    a1, a2, x1 = gas(internal), gas(internal), gas(external)
    control = (a1 - a2).abs().max().item()
    cross = (a1 - x1).abs().max().item()
    scale = a1.abs().max().item()
    assert torch.isfinite(x1).all()
    bar = max(10 * control, 1e-4 * max(scale, 1.0))
    assert cross <= bar, (
        f'gas-leg |d| {cross:.3e} exceeds {bar:.3e} (control {control:.3e}, '
        f'scale {scale:.3e})')

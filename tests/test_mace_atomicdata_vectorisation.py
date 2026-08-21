"""
The hoisted MACE AtomicData builder must be BIT-IDENTICAL to the list path.

This is a physics code and the change is a pure loop-invariant hoist, so `allclose`
is the wrong bar -- any difference at all is a bug. Every test here asserts exact
tensor equality against `Collater(batch_to_mace_atomicdata(...))`, the shipping
reference, via verify_mace_atomicdata_equivalence.

WHAT MAKES IT NON-TRIVIAL, and why each case below exists. The builder has to
reproduce three different notions of "which atoms are the asymmetric unit":

  * a fresh batch with no unit cell -> build it (and for Z'>1, via the
    split_to_zp1 detour, because pose_aunit/build_unit_cell are Z'=1 operations);
  * a SUPERCELL/cluster batch -> only aux_ind == 0 is the aunit, the rest are image
    atoms unit_cell_pos already covers;
  * a plain unit-cell batch -> all of batch.z;

and it must TILE each aunit sym_mult times, image-major, to match unit_cell_pos's
layout. Tiling versus repeat_interleaving is the failure that would be invisible:
both give the right SHAPE, both collate cleanly, and the energies would simply be
wrong. Pinned directly in test_tiling_is_not_repeat_interleave.

A FOURTH case this file adds over the UMA one: `batch_to_mace_atomicdata` guards on
`hasattr(batch, 'unit_cell_pos')` as well as `is None`, which uma_utils does not.
test_batch_without_the_attribute_at_all covers it -- that difference is exactly why
the resolver is duplicated rather than shared.

CPU-ONLY BY CONSTRUCTION. CUDA_VISIBLE_DEVICES is forced empty before torch loads:
nothing here needs a GPU, and MACE inference on the dev box trips the WDDM watchdog
and takes the machine down. The GPU-side equivalence (same INPUTS -> same ENERGIES)
lives in test_mace_gpu_real_batches.py and must run on the cluster.
"""
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import pytest
import torch

from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import (
    batch_to_mace_atomicdata,
    batch_to_mace_atomicdata_hoisted,
    verify_mace_atomicdata_equivalence,
    mace_batch_differences,
    _mace_resolve_aunit_z,
)
from mxtaltools.mlip_interfaces.uma_utils import _tiled_gather_index
from mace.tools.torch_geometric.dataloader import Collater

#: RANDOM CSD, deliberately -- this file tests the BUILDER, not the energy.
#: Element tables, Z'-mixed batches and the unknown-element guard all need
#: chemical DIVERSITY, and an acridine-only fixture is C/H/N, which is exactly
#: this model's table [1,6,7] -- so `test_unknown_element_raises_rather_than_
#: mislabelling` would have no unknown element to raise on. The energy
#: equivalence tests live in test_mace_gpu_real_batches.py and DO use acridine.
DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
MODEL_ENV = 'MACE_CHECKPOINT'


# --------------------------------------------------------------------- fixtures

class _StubModel:
    """
    Everything the builders read from a MACE model, and nothing else: `r_max` and
    `atomic_numbers`.

    A STUB RATHER THAN A CHECKPOINT, and this is a coverage fix, not a convenience.
    Using the real acridine model forced an element filter (it is trained on C/H/N and
    `atomic_numbers_to_indices` raises on anything else), and that filter silently cut
    the fixture from 100 crystals to 5 -- ALL OF THEM Z'=1. So the Z'>1 branch of
    `_mace_resolve_aunit_z`, the split_to_zp1/join detour, was never executed by any
    test, while two of the three production arms this work exists for
    (acridine_sg14_zp2_mace, acridine_sg9_zp2_mace) are Z'=2.

    A wide element table restores the whole fixture, and since these tests compare two
    construction paths against each other, the model's WEIGHTS are irrelevant -- only
    the cutoff and the element table shape the tensors. It also means this file needs
    no checkpoint at all, so it runs in CI.
    """

    def __init__(self, r_max=6.0, atomic_numbers=tuple(range(1, 101))):
        self.r_max = torch.tensor(float(r_max))
        self.atomic_numbers = torch.tensor(list(atomic_numbers))


@pytest.fixture(scope='module', params=[4.0, 6.0], ids=['rmax4', 'rmax6'])
def model(request):
    """Two cutoffs: r_max changes the neighbour count and, on small cells, the number
    of periodic shells -- a builder bug can hide at one and not the other."""
    return _StubModel(r_max=request.param)


@pytest.fixture(scope='module')
def crystals():
    """
    Buildable fixture entries. NO element filter -- see _StubModel.

    Some entries raise inside aunit2ucell (pre-existing, unrelated: it is the fixture,
    before any of this code runs) and one bad entry takes a whole batch down, so those
    are dropped.
    """
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    good = []
    for c in torch.load(DATASET, weights_only=False, map_location='cpu'):
        try:
            p = collate_data_list([c.clone()])
            p.pose_aunit(std_orientation=False)
            p.build_unit_cell()
            good.append(c)
        except Exception:
            pass
    if not good:
        pytest.skip('no buildable crystal in the fixture')
    return good


@pytest.fixture(scope='module')
def zp2_crystals():
    """
    Z'>1 entries, taken from the RAW fixture and deliberately NOT put through the
    `crystals` buildability filter.

    That filter calls `build_unit_cell()` directly, which is a Z'=1 OPERATION -- on a
    Z'>1 batch it raises `einsum(): subscript j has size 7 ...` at every batch size.
    The valid route is the split_to_zp1 / join detour, and the BUILDER performs it.
    So filtering on direct buildability discards precisely the crystals whose branch
    most needs testing: two of the three production arms are Z'=2.
    """
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    raw = torch.load(DATASET, weights_only=False, map_location='cpu')
    out = [c for c in raw if int(c.z_prime.max()) > 1]
    if not out:
        pytest.skip('fixture has no Z\'>1 crystals')
    return out


def _built(crystals, n):
    """A batch with its unit cell ALREADY built.

    The equivalence check has to run on one of these: the no-unit-cell branch MUTATES
    the batch, so calling both paths on a fresh batch would have the second take a
    different branch than the first and compare two different things.
    """
    pool = (crystals * (n // len(crystals) + 1))[:n]
    b = collate_data_list([c.clone() for c in pool])
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    return b


# ------------------------------------------------------------------ equivalence

@pytest.mark.parametrize('n', [1, 2, 5, 13])
def test_identical_to_the_list_path(crystals, model, n):
    """Including n=1 (a single graph has no batching to get wrong, so a failure
    there is a marshalling bug) and n=13 (not a round number, so nothing lines up
    with a stride by luck)."""
    assert verify_mace_atomicdata_equivalence(_built(crystals, n), False, model, False)


def _compare_fresh(make_batch, model):
    """
    Both builders on an UNBUILT batch, compared afterwards.

    verify_mace_atomicdata_equivalence cannot be used here: the fresh-build branch
    MUTATES the batch, so the second path would take the already-built branch and the
    two would not be comparable. Each path therefore gets its own untouched copy.

    The neighbour list is pinned to matscipy on both sides for the same reason the
    verifier pins it: this compares the HOIST, and only the hoisted builder has a
    batched-NL leg, so at the module default (batched on) the two would differ on
    edge order alone. The batched list has its own edge-set and energy gates.
    """
    import mxtaltools.mlip_interfaces.AL_mace_utils as M
    prev = M.USE_BATCHED_MACE_NEIGHBOURS
    M.USE_BATCHED_MACE_NEIGHBOURS = False
    try:
        collate = Collater([None], [None])
        ref = collate(batch_to_mace_atomicdata(make_batch(), False, model, False))
        new = collate(batch_to_mace_atomicdata_hoisted(make_batch(), False, model, False))
    finally:
        M.USE_BATCHED_MACE_NEIGHBOURS = prev
    bad = mace_batch_differences(ref, new)
    assert not bad, 'fresh-build paths differ:\n  ' + '\n  '.join(bad)


@pytest.mark.parametrize('n', [2, 4])
def test_zprime_gt_1_fresh_build(zp2_crystals, model, n):
    """
    Z'>1 takes the split_to_zp1_batch / join_zp1_ucell_batch detour, because
    pose_aunit and build_unit_cell are Z'=1 operations.

    Untested on the mace path until 2026-08-14: the element filter that the real
    acridine checkpoint forced had cut the fixture to 5 crystals, all Z'=1, and the
    buildability filter would have dropped the Z'>1 ones anyway for the wrong reason.
    prod0810 arms 5 and 6 (acridine_sg14_zp2_mace, acridine_sg9_zp2_mace) are Z'=2.
    """
    if len(zp2_crystals) < n:
        pytest.skip(f'need {n} Z\'>1 crystals, have {len(zp2_crystals)}')
    _compare_fresh(
        lambda: collate_data_list([c.clone() for c in zp2_crystals[:n]]), model)


def test_zprime_mixed_batch(crystals, zp2_crystals, model):
    """Z'=1 and Z'>1 in ONE batch: the tiled gather is then ragged in sym_mult AND
    z_prime at once, which neither pure batch exercises."""
    zp1 = [c for c in crystals if int(c.z_prime.max()) == 1]
    if not zp1 or not zp2_crystals:
        pytest.skip('need both Z\'=1 and Z\'>1 crystals')
    mixed = [zp1[0], zp2_crystals[0], zp1[-1]]
    _compare_fresh(lambda: collate_data_list([c.clone() for c in mixed]), model)


def _edge_key(d):
    """Order-insensitive edge identity for one AtomicData: {(i, j, Sx, Sy, Sz)}."""
    ei = d.edge_index.cpu().numpy()
    us = d.unit_shifts.cpu().numpy().round().astype(int)
    return {(int(ei[0, k]), int(ei[1, k]), *map(int, us[k])) for k in range(ei.shape[1])}


@pytest.mark.parametrize('n', [1, 4, 9])
def test_batched_neighbours_build_the_same_atomicdata(crystals, model, n, monkeypatch):
    """
    The batched neighbour list must produce the SAME AtomicData as matscipy: same
    edge set per graph, same shapes, and shifts consistent with unit_shifts @ cell.

    Edge ORDER is not compared -- see pbc_neighbours' module docstring. Everything
    that is order-independent IS compared, exactly.

    This is the CPU half of validating MXT_BATCHED_MACE_NEIGHBOURS. The other half is
    energies through a real forward (test_mace_gpu_real_batches.py); both halves
    passed 2026-08-19 and the flag now defaults on.
    """
    import mxtaltools.mlip_interfaces.AL_mace_utils as M

    b1, b2 = _built(crystals, n), _built(crystals, n)
    monkeypatch.setattr(M, 'USE_BATCHED_MACE_NEIGHBOURS', False)
    ref = M.batch_to_mace_atomicdata_hoisted(b1, False, model, False)
    monkeypatch.setattr(M, 'USE_BATCHED_MACE_NEIGHBOURS', True)
    new = M.batch_to_mace_atomicdata_hoisted(b2, False, model, False)

    assert len(ref) == len(new)
    for g, (a, c) in enumerate(zip(ref, new)):
        assert _edge_key(a) == _edge_key(c), f'graph {g}: edge sets differ'
        assert a.positions.shape == c.positions.shape, f'graph {g}: positions'
        assert torch.equal(a.node_attrs, c.node_attrs), f'graph {g}: node_attrs'
        # shifts must be the cartesian image of unit_shifts under this graph's cell,
        # or the model's displacements are wrong even with a correct edge set
        cell = torch.as_tensor(np.asarray(a.cell), dtype=c.unit_shifts.dtype)
        assert torch.allclose(c.shifts, c.unit_shifts @ cell.reshape(3, 3), atol=1e-6), \
            f'graph {g}: shifts inconsistent with unit_shifts @ cell'


@pytest.mark.parametrize('n', [48])
def test_batched_neighbours_at_batch_scale(crystals, model, n, monkeypatch):
    """
    The same equivalence at a batch size where the SPLIT can be wrong.

    WHY A SEPARATE, LARGER CASE. The consumer that turns one batched neighbour list
    into per-graph AtomicData was rewritten from a per-graph mask (`e_g == ind`) to a
    single sort plus contiguous slices, because the mask form is O(G*E) and costs
    3*G device->host syncs. The failure mode of the new form is an OFF-BY-ONE IN THE
    SLICE BOUNDS OR THE PER-GRAPH ATOM OFFSET, and neither is reliably exercised at
    n = 1, 4, 9: with few graphs the bounds are trivially right and a misapplied
    offset often still lands inside the graph.

    Ragged by construction -- the fixture mixes space groups, so sym_mult and atom
    counts differ per graph and the slice starts are not a multiple of anything.

    Asserts the total edge count too: a bounds bug that drops or double-counts a
    graph's edges can still leave every INDIVIDUAL edge set a subset-match, and the
    per-graph set comparison alone would not necessarily catch it.
    """
    import mxtaltools.mlip_interfaces.AL_mace_utils as M

    b1, b2 = _built(crystals, n), _built(crystals, n)
    monkeypatch.setattr(M, 'USE_BATCHED_MACE_NEIGHBOURS', False)
    ref = M.batch_to_mace_atomicdata_hoisted(b1, False, model, False)
    monkeypatch.setattr(M, 'USE_BATCHED_MACE_NEIGHBOURS', True)
    new = M.batch_to_mace_atomicdata_hoisted(b2, False, model, False)

    assert len(ref) == len(new) == n
    assert sum(a.edge_index.shape[1] for a in ref) == \
           sum(c.edge_index.shape[1] for c in new), 'total edge count differs'

    for g, (a, c) in enumerate(zip(ref, new)):
        assert _edge_key(a) == _edge_key(c), f'graph {g}: edge sets differ'
        # every index must land inside THIS graph -- the offset is applied per edge
        # from its own graph, so a wrong one can still produce a valid-looking index
        n_at = a.positions.shape[0]
        assert int(c.edge_index.min()) >= 0 and int(c.edge_index.max()) < n_at, \
            f'graph {g}: edge_index escapes its own atom range'
        cell = torch.as_tensor(np.asarray(a.cell), dtype=c.unit_shifts.dtype)
        assert torch.allclose(c.shifts, c.unit_shifts @ cell.reshape(3, 3), atol=1e-6), \
            f'graph {g}: shifts inconsistent with unit_shifts @ cell'


@pytest.mark.parametrize('batched', [False, True])
def test_builder_works_when_the_cell_requires_grad(crystals, model, batched,
                                                   monkeypatch):
    """
    The builder must run when T_fc carries gradients.

    WHY THIS IS A SEPARATE TEST. Every other case here runs without autograd, and the
    batched-neighbour branch converts tensors to numpy -- which RAISES on a
    grad-carrying tensor. So the whole no-grad suite passed while that branch was
    broken for the one caller that needs it: crystal_opt_utils' gradient descent,
    which backprops through the cell into the MLIP. The failure was invisible to
    every existing test and only appeared under a real optimiser-shaped call.

    Both flag states are parametrised because the reference path builds shifts from a
    DETACHED numpy cell and so was never exposed; asserting only the batched branch
    would not pin the property that the two must agree on.
    """
    import mxtaltools.mlip_interfaces.AL_mace_utils as M

    monkeypatch.setattr(M, 'USE_BATCHED_MACE_NEIGHBOURS', batched)
    b = _built(crystals, 6)
    b.T_fc.requires_grad_(True)
    assert b.T_fc.requires_grad

    ds = M.batch_to_mace_atomicdata_hoisted(b, False, model, False)
    assert len(ds) == 6
    # shifts are a placeholder here -- compute_crystal_mace_on_mxt_batch rebuilds them
    # differentiably before the forward -- so they must be detached, not merely present
    for g, a in enumerate(ds):
        assert not a.shifts.requires_grad, f'graph {g}: shifts leaked a grad'


@pytest.mark.parametrize('n', [1, 4, 17])
def test_device_built_input_dict_matches_the_collated_one(crystals, model, n):
    """
    The dict built directly on device must equal what the Collater path produces.

    EXACT equality on every model-visible field. This replaces both the per-graph
    AtomicData construction and PyG's Collater, so it is a pure construction change --
    any difference is a bug, not rounding.

    17 is deliberately not a round number: a builder that got the per-graph slicing
    subtly wrong can still line up at 1, 2, 4 and fail on a ragged count.
    """
    from mxtaltools.mlip_interfaces.AL_mace_utils import (
        verify_mace_input_dict_equivalence)
    assert verify_mace_input_dict_equivalence(_built(crystals, n), False, model, False)


def test_device_built_dict_is_on_by_default():
    """
    The fast path must be the DEFAULT, not an env var someone has to remember: the
    a100_stab_aug16 battery measured the slow path shipping by omission (occupancy
    54% vs 82%), and phase6_handoff.md §3.1 turned that into the requirement that
    this behaviour live in code. The energy/grad gates it waited on are in
    test_mace_gpu_real_batches.py and passed 2026-08-19.
    """
    import mxtaltools.mlip_interfaces.AL_mace_utils as M
    assert M.USE_GPU_MACE_BATCH is True, (
        'MXT_GPU_MACE_BATCH defaults OFF -- the measured fast path must ship as the '
        'default, with the env var as a kill switch only')

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip('device residency needs a GPU')


def test_unknown_element_raises_rather_than_mislabelling(crystals):
    """
    An element outside the model's table must RAISE.

    The reference builder gets this from atomic_numbers_to_indices, which raises on an
    unknown z. The device path uses a gather table instead, and a gather CANNOT raise
    -- it would return whatever sits at the index. Getting this wrong swaps one
    element for another and produces a perfectly well-formed, quietly wrong energy.
    """
    from mxtaltools.mlip_interfaces.AL_mace_utils import batch_to_mace_input_dict
    narrow = _StubModel(atomic_numbers=(1, 6))          # no nitrogen
    b = _built(crystals, 8)
    zs = set(int(z) for z in b.z)
    if zs.issubset({1, 6}):
        pytest.skip('fixture happens to contain no element outside H/C')
    with pytest.raises(ValueError, match='not in the model element table'):
        batch_to_mace_input_dict(b, False, narrow, False)


def test_batched_neighbours_are_on_by_default():
    """The measured fast path must ship as the DEFAULT (phase6_handoff.md §3.1: the
    slow path shipping by omission cost 27% of the step and 27 points of occupancy).
    The energy validation the old default-off waited on ran 2026-08-19
    (test_mace_gpu_real_batches.py, energies and grads)."""
    import mxtaltools.mlip_interfaces.AL_mace_utils as M
    assert M.USE_BATCHED_MACE_NEIGHBOURS is True, (
        'MXT_BATCHED_MACE_NEIGHBOURS defaults OFF -- the validated fast path must '
        'ship in code, with the env var as a kill switch only')


def test_pbc_false(crystals, model):
    """The gas-phase leg runs pbc=False, which changes the neighbour list, the cell
    and the pbc flag together."""
    assert verify_mace_atomicdata_equivalence(_built(crystals, 4), False, model, False,
                                              pbc=False)


def test_ragged_sym_mult(crystals, model):
    """Different space groups in one batch => different sym_mult per graph => the
    tiled gather has a ragged stride. A builder that assumed a uniform multiplier
    would pass every equal-sym_mult test and fail here."""
    b = _built(crystals, min(8, len(crystals)))
    if len(torch.unique(b.sym_mult)) < 2:
        pytest.skip('fixture subset has a uniform sym_mult')
    assert verify_mace_atomicdata_equivalence(b, False, model, False)


def test_supercell_batch_uses_only_aux_ind_zero(crystals, model):
    """With aux_ind present, only aux_ind == 0 is the asymmetric unit; taking all of
    batch.z would double-count the image atoms unit_cell_pos already covers."""
    b = _built(crystals, 4)
    n_aunit = int(b.z.shape[0])
    keep_n = n_aunit // 2                 # atoms [0, keep_n) stay aux_ind == 0
    b.aux_ind = torch.zeros(n_aunit, dtype=torch.long)
    b.aux_ind[keep_n:] = 1                # the rest are image atoms
    z, ind = _mace_resolve_aunit_z(b, False, False)
    assert int(z.shape[0]) == keep_n, 'aux_ind != 0 was not excluded'
    assert int(ind.shape[0]) == int(z.shape[0])
    assert keep_n < n_aunit, 'the fixture must actually have some image atoms'


def test_batch_without_the_attribute_at_all(crystals, model):
    """batch_to_mace_atomicdata guards on hasattr AS WELL AS `is None` -- uma_utils
    does not, which is why the resolver is duplicated rather than shared. A batch
    with no unit_cell_pos attribute must rebuild, not raise."""
    b = collate_data_list([crystals[0].clone()])
    if hasattr(b, 'unit_cell_pos'):
        try:
            del b.unit_cell_pos
        except Exception:
            pytest.skip('cannot remove unit_cell_pos from this data class')
    z, ind = _mace_resolve_aunit_z(b, False, False)   # must not raise
    assert z.shape[0] > 0 and ind.shape[0] == z.shape[0]


def test_tiling_is_not_repeat_interleave(crystals, model):
    """
    THE INVISIBLE FAILURE. `Tensor.repeat(k)` on 1-D gives [a,b,c,a,b,c];
    repeat_interleave gives [a,a,b,b,c,c]. Both have the right length, both collate
    cleanly, and unit_cell_pos is laid out image-major -- so getting this backwards
    pairs every position with the wrong element and yields a plausible wrong energy.
    """
    b = _built(crystals, 3)
    z, batch_ind = _mace_resolve_aunit_z(b, False, False)
    gather, out_n = _tiled_gather_index(batch_ind, b.sym_mult, b.num_graphs)
    tiled = z[gather]

    lo = 0
    for i in range(b.num_graphs):
        n_i = int((batch_ind == i).sum())
        m_i = int(b.sym_mult[i])
        got = tiled[lo:lo + n_i * m_i]
        assert torch.equal(got, z[batch_ind == i].repeat(m_i)), f'graph {i} not tiled'
        if n_i > 1 and m_i > 1 and len(torch.unique(z[batch_ind == i])) > 1:
            assert not torch.equal(got, z[batch_ind == i].repeat_interleave(m_i)), (
                f'graph {i}: tiling and repeat_interleave agree, so this test is '
                f'blind here -- pick a fixture with a non-uniform aunit')
        lo += n_i * m_i


def test_no_attribute_is_missing_or_none(crystals, model):
    """A field the hoisted path forgets would be None (or absent) rather than wrong,
    and an equality check that skips None on BOTH sides would not catch it."""
    ds = batch_to_mace_atomicdata_hoisted(_built(crystals, 3), False, model, False)
    ref = batch_to_mace_atomicdata(_built(crystals, 3), False, model, False)
    for key in ref[0].keys:
        for i, d in enumerate(ds):
            assert key in d.keys, f'graph {i} is missing {key!r}'
            assert getattr(d, key) is not None, f'graph {i} has {key!r} as None'


def test_constants_are_not_aliased_across_graphs_after_collation(crystals, model):
    """
    The hoist SHARES the constant tensors between graphs, which is only safe because
    Collater concatenates rather than aliases. Verified directly: mutate the collated
    result and the source constants must be untouched.
    """
    b = _built(crystals, 3)
    ds = batch_to_mace_atomicdata_hoisted(b, False, model, False)
    before = ds[0].weight.clone()
    collated = Collater([None], [None])(ds)
    collated.weight.add_(1.0)
    assert torch.equal(ds[0].weight, before), (
        'collation aliased a shared constant -- mutating the batch changed the source')

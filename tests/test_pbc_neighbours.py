"""
The batched PBC neighbour list must find the SAME EDGE SET as matscipy.

THE BAR, and why it is not array equality. matscipy emits edges in its own cell-list
order; no reimplementation reproduces that sequence, and MACE sums over edges so the
order only perturbs float summation -- the same class as the tf32 noise already in
every energy. So the assertion is on the SET of (i, j, Sx, Sy, Sz) tuples, exact, and
it is a real bar: a dropped edge, a spurious edge, a wrong shift or an off-by-one in
the grid all break it, while a reordering does not.

Energy equivalence is the acceptance criterion (test_mace_gpu_real_batches.py), but
it cannot be the ONLY one: a single dropped long edge moves the energy by less than
the model's own nondeterminism, so an energy-only suite would pass while quietly
truncating the neighbour list. The set comparison is what makes the energy test
interpretable when it does drift.

CPU-only. No GPU, no checkpoint -- pure geometry.
"""
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import pytest
import torch

from mace.data import get_neighborhood
import mxtaltools.mlip_interfaces.pbc_neighbours as nb
from mxtaltools.mlip_interfaces.pbc_neighbours import (
    batched_pbc_neighbour_list, lattice_shift_range, get_neighborhood_batched_numpy)


def _edge_set(edge_index, unit_shifts):
    """Order-insensitive comparison key."""
    ei = np.asarray(edge_index)
    us = np.asarray(unit_shifts).astype(np.int64)
    return {(int(ei[0, k]), int(ei[1, k]), int(us[k, 0]), int(us[k, 1]), int(us[k, 2]))
            for k in range(ei.shape[1])}


def _reference(positions, cutoff, cell, pbc=(True, True, True)):
    # get_neighborhood MUTATES the cell it is given; hand it a copy so the fixture
    # is not altered underneath the comparison
    ei, _, us, _ = get_neighborhood(positions=np.array(positions, dtype=float),
                                    cutoff=cutoff, pbc=list(pbc),
                                    cell=np.array(cell, dtype=float))
    return _edge_set(ei, us)


def _mine(positions, cutoff, cell, pbc=(True, True, True)):
    ei, _, us, _ = get_neighborhood_batched_numpy(
        np.array(positions, dtype=float), cutoff, list(pbc), np.array(cell, dtype=float))
    return _edge_set(ei, us)


# ------------------------------------------------------------------- geometries

def _cubic(a=6.0):
    return np.diag([a, a, a]).astype(float)


def _triclinic():
    """Sheared on purpose: this is where cutoff/|a_i| (the WRONG range formula)
    under-counts, and it under-counts silently."""
    return np.array([[7.0, 0.0, 0.0],
                     [3.2, 6.1, 0.0],
                     [2.7, 1.9, 5.4]], dtype=float)


def _thin_slab():
    """One very short axis, so the shift range along it must exceed 1."""
    return np.array([[3.1, 0.0, 0.0],
                     [0.0, 9.0, 0.0],
                     [0.0, 0.0, 9.0]], dtype=float)


def _rand_pos(n, cell, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)) @ cell


def _unwrapped_pos(n, cell, seed, lo=-0.6, hi=1.6):
    """
    Positions that SPILL OUT of the cell, i.e. fractional coordinates outside [0, 1).

    THIS IS THE REALISTIC CASE AND IT IS WHAT THE ORIGINAL FIXTURES MISSED.
    `unit_cell_pos` is never wrapped -- molecules straddle cell boundaries -- so the
    fractional span exceeds one cell width and a pair can need one more shell than
    `ceil(cutoff / d_perp)` allows. With `rand(n,3) @ cell` the span is under 1 and
    ceil() hides the whole effect, which is why a 43-test suite passed while real CSD
    crystals lost 46 edges out of 2704 on one graph.
    """
    rng = np.random.default_rng(seed)
    return (lo + (hi - lo) * rng.random((n, 3))) @ cell


CELLS = {'cubic': _cubic(), 'triclinic': _triclinic(), 'thin': _thin_slab()}


# ----------------------------------------------------------------- equivalence

#: 6.5 is here ON PURPOSE and must not be dropped to make the suite faster. The
#: triclinic fixture has d_perp_1 = 5.97 against |a_1| = 7.0, so the correct and the
#: naive (cutoff/|a_i|) shift ranges AGREE for every cutoff below ~6 and diverge
#: above it. With only 3.0 and 5.0 in this list the whole equivalence sweep passed
#: against a deliberately broken range formula -- 30 of 31 tests green while the
#: neighbour list silently dropped edges. Keep a cutoff that crosses the boundary.
CUTOFFS = [3.0, 5.0, 6.5]


@pytest.mark.parametrize('name', sorted(CELLS))
@pytest.mark.parametrize('n', [1, 2, 8, 30])
@pytest.mark.parametrize('cutoff', CUTOFFS)
def test_edge_set_matches_matscipy(name, n, cutoff):
    cell = CELLS[name]
    pos = _rand_pos(n, cell, seed=hash((name, n)) % 2**31)
    assert _mine(pos, cutoff, cell) == _reference(pos, cutoff, cell)


@pytest.mark.parametrize('name', sorted(CELLS))
@pytest.mark.parametrize('n', [4, 20])
@pytest.mark.parametrize('cutoff', CUTOFFS)
def test_edge_set_matches_with_unwrapped_positions(name, n, cutoff):
    """The regression that matters: atoms outside the cell. See _unwrapped_pos."""
    cell = CELLS[name]
    pos = _unwrapped_pos(n, cell, seed=hash((name, n, cutoff)) % 2**31)
    assert _mine(pos, cutoff, cell) == _reference(pos, cutoff, cell)


def test_shift_range_grows_with_the_fractional_spread():
    """Direct check on the term itself: a batch whose atoms span two cell widths must
    demand a strictly larger range than the same cell with atoms inside it."""
    from mxtaltools.mlip_interfaces.pbc_neighbours import fractional_spread
    cell = torch.tensor(_cubic(6.0), dtype=torch.float64).unsqueeze(0)
    inside = torch.tensor(_rand_pos(10, _cubic(6.0), 1), dtype=torch.float64)
    outside = torch.tensor(_unwrapped_pos(10, _cubic(6.0), 1), dtype=torch.float64)
    b = torch.zeros(10, dtype=torch.long)
    r_in = lattice_shift_range(cell, 5.0, fractional_spread(inside, b, cell, 1))
    r_out = lattice_shift_range(cell, 5.0, fractional_spread(outside, b, cell, 1))
    assert (r_out >= r_in).all()
    assert (r_out > r_in).any(), 'the spread term is not doing anything here'


def test_shift_range_cap_never_binds_on_physical_cells():
    """THE DIRECTION THAT MATTERS. The cap is a memory bound, and it must never be
    the thing deciding a real crystal's edge set. Measured on the acridine prior,
    physical cells want n_i <= 3 at a 6 A cutoff; every fixture here must sit well
    under MAX_SHIFT_RANGE with the counter untouched."""
    from mxtaltools.mlip_interfaces.pbc_neighbours import (
        MAX_SHIFT_RANGE, _SHIFT_CAP_CALLS, drain_neighbour_path_counts)
    drain_neighbour_path_counts()                       # zero the counters
    for name, cell in (('cubic', _cubic(6.0)), ('triclinic', _triclinic())):
        c = torch.tensor(cell, dtype=torch.float64).unsqueeze(0)
        rng = lattice_shift_range(c, 6.0, torch.ones_like(c[:, :, 0]))
        assert int(rng.max()) < MAX_SHIFT_RANGE, (
            f'{name} wants {int(rng.max())} shifts, at or above the cap '
            f'{MAX_SHIFT_RANGE} -- the cap would be deciding a PHYSICAL edge set')
    assert _SHIFT_CAP_CALLS['capped'] == 0


def test_shift_range_cap_binds_on_a_degenerate_cell_and_says_so():
    """The other direction, without which the test above passes on a cap that does
    nothing. A flattened cell has interplanar spacing -> 0, so the requested range
    is unbounded; it must be refused, COUNTED, and reported."""
    from mxtaltools.mlip_interfaces.pbc_neighbours import (
        MAX_SHIFT_RANGE, _SHIFT_CAP_CALLS, drain_neighbour_path_counts)
    drain_neighbour_path_counts()
    cell = torch.tensor(_cubic(6.0), dtype=torch.float64).unsqueeze(0).clone()
    cell[0, 2] = cell[0, 2] * 1e-3                      # squash the third axis
    rng = lattice_shift_range(cell, 6.0, torch.ones_like(cell[:, :, 0]))
    assert int(rng.max()) == MAX_SHIFT_RANGE, 'the cap did not bind'
    assert _SHIFT_CAP_CALLS['capped'] == 1
    assert _SHIFT_CAP_CALLS['max_requested'] > MAX_SHIFT_RANGE, (
        'the refused range must be recorded, or a capped call is '
        'indistinguishable from a call that never needed capping')
    out = drain_neighbour_path_counts()
    if out:                                             # only if a list was built
        assert out['energy/nl_shift_capped_frac'] > 0


def test_one_degenerate_cell_does_not_set_the_grid_for_the_whole_batch():
    """WHY THE CAP EXISTS AT ALL, as an assertion rather than a comment.

    `batched_pbc_neighbour_list` ghost-expands every graph on `.max(dim=0)` of the
    per-graph ranges, so without a ceiling ONE degenerate crystal drives K -- and
    therefore the memory -- for every sane crystal beside it. Measured 2026-08-20
    at 128 acridine graphs: squashing a single cell x0.001 took K from 245 to
    50,029 and OOM'd a 16 GB card, while the edge count rose only 2.5x."""
    from mxtaltools.mlip_interfaces.pbc_neighbours import MAX_SHIFT_RANGE
    good = torch.tensor(_cubic(6.0), dtype=torch.float64)
    cells = good.unsqueeze(0).repeat(8, 1, 1).clone()
    cells[0, 2] = cells[0, 2] * 1e-4                    # one bad cell in eight
    rng = lattice_shift_range(cells, 6.0, torch.ones_like(cells[:, :, 0]))
    shared = rng.max(dim=0).values                      # what the batch actually uses
    K = int(torch.prod(2 * shared + 1))
    assert int(shared.max()) <= MAX_SHIFT_RANGE
    assert K <= (2 * MAX_SHIFT_RANGE + 1) ** 3, f'K={K} is not bounded by the cap'


def test_single_atom_still_has_periodic_self_edges(n=1):
    """A one-atom cell has NO ordinary neighbours but does have self-images across
    the boundary. Dropping i == j unconditionally -- the obvious way to write the
    self-edge filter -- would return nothing here and be invisible in a bulk test."""
    cell = _cubic(4.0)
    pos = np.zeros((1, 3))
    mine, ref = _mine(pos, 5.0, cell), _reference(pos, 5.0, cell)
    assert mine == ref
    assert len(mine) > 0, 'periodic self-images were dropped'


def test_true_self_edge_is_excluded():
    """i == j with S == 0 must NOT appear -- that is an atom with itself."""
    cell = _cubic(6.0)
    pos = _rand_pos(6, cell, seed=5)
    for (i, j, sx, sy, sz) in _mine(pos, 4.0, cell):
        assert not (i == j and sx == 0 and sy == 0 and sz == 0)


def test_cutoff_larger_than_the_cell_needs_multiple_shells():
    """Cutoff well over the shortest axis: the range must be >1 along it. This is the
    case a `range = 1` assumption gets wrong, and it gets it wrong by dropping edges,
    which no shape or dtype check would notice."""
    cell = _thin_slab()
    pos = _rand_pos(5, cell, seed=11)
    assert _mine(pos, 7.0, cell) == _reference(pos, 7.0, cell)


def test_non_periodic_matches_and_reproduces_the_cell_rewrite():
    """pbc=False: the reference REWRITES the cell in place and the caller uses the
    returned one. Both the edges and that side effect have to match."""
    cell = _cubic(5.0)
    pos = _rand_pos(7, cell, seed=3)
    ref_cell = np.array(cell, dtype=float)
    ref_ei, _, ref_us, ref_cell_out = get_neighborhood(
        positions=pos, cutoff=4.0, pbc=[False] * 3, cell=ref_cell)
    my_ei, _, my_us, my_cell_out = get_neighborhood_batched_numpy(
        pos, 4.0, [False] * 3, np.array(cell, dtype=float))
    assert _edge_set(my_ei, my_us) == _edge_set(ref_ei, ref_us)
    assert np.allclose(my_cell_out, ref_cell_out), 'the cell rewrite was not reproduced'


# ------------------------------------------------------------------ the batching

def test_batched_equals_per_graph():
    """The whole point: many graphs at once must equal the same graphs one at a time,
    including graphs with DIFFERENT cells and different atom counts in one call."""
    specs = [(6, _cubic(6.0), 1), (11, _triclinic(), 2), (4, _thin_slab(), 3),
             (1, _cubic(4.0), 4)]
    cutoff = 4.5
    poss = [_rand_pos(n, c, s) for n, c, s in specs]

    pos = torch.tensor(np.concatenate(poss), dtype=torch.float64)
    batch_ind = torch.tensor(sum([[g] * len(p) for g, p in enumerate(poss)], []),
                             dtype=torch.long)
    cells = torch.tensor(np.stack([c for _, c, _ in specs]), dtype=torch.float64)

    ei, us, gid = batched_pbc_neighbour_list(pos, batch_ind, cells, cutoff)

    starts = np.cumsum([0] + [len(p) for p in poss])
    for g, p in enumerate(poss):
        m = (gid == g).numpy()
        got = _edge_set(ei[:, m].numpy() - starts[g], us[m].numpy())
        assert got == _reference(p, cutoff, specs[g][1]), f'graph {g} differs'


def test_shift_range_uses_interplanar_spacing_not_vector_length():
    """
    The formula that fails silently. For a sheared cell the interplanar spacing is
    SMALLER than the lattice vector length, so cutoff/|a_i| under-counts and drops
    edges. Asserted directly, because the consequence is a slightly wrong energy
    rather than an error.
    """
    cell = torch.tensor(_triclinic(), dtype=torch.float64).unsqueeze(0)
    cutoff = 6.0
    proper = lattice_shift_range(cell, cutoff)[0]
    naive = torch.ceil(cutoff / cell[0].norm(dim=-1)).long()
    assert (proper >= naive).all(), 'interplanar range must never be smaller'
    assert (proper > naive).any(), (
        'this fixture no longer distinguishes the two formulas -- pick a more '
        'sheared cell or the test is blind')


def _dense_cell_batch():
    """One small cell packed hard enough that the neighbour count is large -- the
    regime where the cap actually matters."""
    cell = _cubic(7.0)
    rng = np.random.default_rng(3)
    pos = rng.random((160, 3)) @ cell
    return (torch.tensor(pos, dtype=torch.float64),
            torch.zeros(160, dtype=torch.long),
            torch.tensor(cell, dtype=torch.float64).unsqueeze(0))


def test_radius_cap_starts_far_below_the_ghost_count():
    """
    The starting cap must not be derived from the shift count.

    max_num_neighbors sizes torch_cluster's output buffer, so a cap chosen as k*64
    costs memory proportional to how oversized it is. Measured true maxima are ~141
    neighbours on physical cells and ~2710 on degenerate ones, against a k*64 of
    21952 and 118976 -- which cost 18 GiB, 33 GiB, and an outright OOM. This pins the
    start to a value chosen from those measurements rather than from k.
    """
    from mxtaltools.mlip_interfaces import pbc_neighbours as pn
    assert pn.RADIUS_CAP_START <= 8192, (
        f'RADIUS_CAP_START={pn.RADIUS_CAP_START} is back in the regime that OOM\'d; '
        f'it must be sized from measured neighbour counts, not from the shift grid')
    assert pn.RADIUS_CAP_START > 2710, (
        'RADIUS_CAP_START must clear the measured worst case on degenerate cells, '
        'or every such batch pays a wasted radius search before escalating')


def test_cap_escalation_gives_the_same_edges(monkeypatch):
    """
    A cap too small to hold the true neighbour count must ESCALATE and still return
    the identical edge set -- never a truncated one.

    Driven by forcing an absurdly small start, because on well-behaved fixtures the
    real start never overflows and the retry path would otherwise be dead code that
    only runs in production. Truncation here would be silent, so 'it did not raise'
    is not evidence: the edge SET is compared against the unforced result.
    """
    from mxtaltools.mlip_interfaces import pbc_neighbours as pn
    pytest.importorskip('torch_cluster')
    pos, b, cell = _dense_cell_batch()

    ei_ref, us_ref, _ = pn.batched_pbc_neighbour_list(pos, b, cell, 6.0)
    ref = _edge_set(ei_ref, us_ref)

    monkeypatch.setattr(pn, 'RADIUS_CAP_START', 8)
    ei_esc, us_esc, _ = pn.batched_pbc_neighbour_list(pos, b, cell, 6.0)
    esc = _edge_set(ei_esc, us_esc)

    assert len(ref) > 0, 'fixture produced no edges -- it cannot exercise the cap'
    assert esc == ref, (
        f'escalated result differs: {len(ref - esc)} missing, {len(esc - ref)} '
        f'spurious -- the cap TRUNCATED instead of retrying')


def test_empty_result_is_well_formed():
    """Atoms far apart in a huge box: no edges. Shapes must still be right, or the
    downstream concat/collate fails on exactly the batch that had nothing to say."""
    cell = _cubic(200.0)
    pos = np.array([[0., 0., 0.], [90., 90., 90.]])
    ei, us, gid = batched_pbc_neighbour_list(
        torch.tensor(pos), torch.zeros(2, dtype=torch.long),
        torch.tensor(cell).unsqueeze(0), 3.0)
    assert ei.shape == (2, 0) and us.shape == (0, 3) and gid.shape == (0,)


# --------------------------------------------------------------- edge cap ---
# The per-node edge cap is the module's ONE deliberate approximation, so these
# test it in both directions: that it is inert when it should be, that it fires
# when it should, and that what it keeps is the SHORTEST edges rather than
# whatever the search happened to return first.

def _degenerate_batch(squash, n_graphs=4, n_atoms=24, cutoff=6.0):
    """A batch whose FIRST cell is squashed and whose others are physical."""
    base = torch.tensor([[11.0, 0, 0], [0, 6.0, 0], [0, 0, 14.0]])
    cells = base.repeat(n_graphs, 1, 1).clone()
    cells[0, 2, 2] *= squash
    torch.manual_seed(0)
    pos = torch.rand(n_graphs * n_atoms, 3) @ base
    gi = torch.arange(n_graphs).repeat_interleave(n_atoms)
    return pos, gi, cells, cutoff


def test_edge_cap_is_inert_when_k_exceeds_degree():
    """THE SAFETY PROPERTY. With the cap above the true max degree the edge set
    must be IDENTICAL -- a cap that perturbs a structure it is not binding on
    would silently change energies everywhere, which is the F-047 failure."""
    pos, gi, cells, cut = _degenerate_batch(1.0)
    nb._EDGE_CAP_COUNTS.update(calls=0, capped=0, max_degree=0,
                               edges_in=0, edges_out=0)
    ref = nb.batched_pbc_neighbour_list(pos, gi, cells, cut, pbc=True)
    max_deg = nb._EDGE_CAP_COUNTS['max_degree']
    assert max_deg > 0, 'no edges built; the fixture proves nothing'
    got = nb._apply_edge_cap(ref[0][0], ref[0][1], ref[1], ref[2],
                             pos, cells, max_deg + 1)
    assert torch.equal(got[0], ref[0][0]) and torch.equal(got[1], ref[0][1])
    assert nb._EDGE_CAP_COUNTS['capped'] == 0, 'cap fired when it should not have'


def test_edge_cap_binds_on_a_degenerate_cell_and_bounds_the_total():
    """The cap must actually FIRE, and the bound must be the promised
    n_nodes * K -- a cap that is never exercised is not a cap."""
    pos, gi, cells, cut = _degenerate_batch(0.01)
    nb._EDGE_CAP_COUNTS.update(calls=0, capped=0, max_degree=0,
                               edges_in=0, edges_out=0)
    uncapped = nb.batched_pbc_neighbour_list(pos, gi, cells, cut, pbc=True)
    max_deg = nb._EDGE_CAP_COUNTS['max_degree']
    assert max_deg > 50, f'fixture is not degenerate enough (max degree {max_deg})'
    K = 8
    i, j, s, g = nb._apply_edge_cap(uncapped[0][0], uncapped[0][1], uncapped[1],
                                    uncapped[2], pos, cells, K)
    assert i.numel() < uncapped[0].shape[1], 'cap discarded nothing'
    assert int(torch.bincount(i).max()) <= K, 'a node kept more than K edges'
    assert i.numel() <= pos.shape[0] * K, 'total edges exceed the n_nodes * K bound'


def test_edge_cap_keeps_the_shortest_edges():
    """WHICH edges survive is the whole safety argument -- MACE's radial basis
    decays to zero at r_max, so dropping the LONGEST is what makes the error
    tolerable. Keeping an arbitrary K (what torch_cluster's own
    max_num_neighbors does) would not be defensible."""
    pos, gi, cells, cut = _degenerate_batch(0.05)
    full = nb.batched_pbc_neighbour_list(pos, gi, cells, cut, pbc=True)
    i_f, j_f, s_f, g_f = full[0][0], full[0][1], full[1], full[2]

    def dists(i, j, s, g):
        off = torch.einsum('ec,ecd->ed', s.to(cells.dtype), cells[g])
        return (pos[j] + off - pos[i]).norm(dim=1)

    K = 5
    i, j, s, g = nb._apply_edge_cap(i_f, j_f, s_f, g_f, pos, cells, K)
    d_all, d_keep = dists(i_f, j_f, s_f, g_f), dists(i, j, s, g)
    node = int(torch.bincount(i_f).argmax())          # the busiest node
    ref = d_all[i_f == node].sort().values[:K]
    assert torch.allclose(d_keep[i == node].sort().values, ref), \
        'the kept edges are not the K shortest at that node'


def test_edge_cap_never_orphans_a_node():
    """Per-NODE rather than per-graph, precisely so this cannot happen: an
    isolated node receives no messages, so its features are garbage while it
    still contributes to the energy sum."""
    pos, gi, cells, cut = _degenerate_batch(0.01)
    full = nb.batched_pbc_neighbour_list(pos, gi, cells, cut, pbc=True)
    had = torch.bincount(full[0][0], minlength=pos.shape[0]) > 0
    i, _, _, _ = nb._apply_edge_cap(full[0][0], full[0][1], full[1], full[2],
                                    pos, cells, 1)
    still = torch.bincount(i, minlength=pos.shape[0]) > 0
    assert torch.equal(had, still), 'a node that had edges lost all of them'

"""
Batched, on-device PBC neighbour lists for MLIP inputs.

WHY THIS EXISTS. MACE's `mace.data.get_neighborhood` is a matscipy/ASE cell-list in
numpy, called ONCE PER CRYSTAL on the host, and every millisecond of it is time the
GPU spends idle -- which on a usage-policed cluster is not merely slow, it is the
number jobs get cancelled on. Its share of the AtomicData build is roughly
three-quarters and RISES as the rest of the build is optimised, so treat any pinned
percentage here as stale by construction and re-measure; `energy/mace_nl_frac_of_build`
reports it per run.

THE ALGORITHM, and why it is not a cell list. A cell list wins when N is large. Here
a unit cell is ~10^2 atoms (acridine sg14 Z'=1 is 92), so exact all-pairs per lattice
shift is both simpler and faster on a GPU, and -- more importantly -- it is EXACT.
There is no neighbour cap to overflow, which is the failure mode that would otherwise
be silent: `asymmetric_radius_graph`'s `max_num_neighbors=32` default would quietly
drop edges in a dense organic crystal and yield a plausible wrong energy.

    for each integer lattice shift S in the enumerated grid:
        d_ij = |pos_j + S @ cell - pos_i|      (all i,j within each graph)
        keep d_ij < cutoff, dropping i==j only when S == 0

WORK: sum_g n_g^2 * K. At 1000 crystals of 92 atoms with K=27 that is ~230M distance
evaluations -- nothing for a GPU, done in K passes of ~100 MB each rather than one
2.7 GB tensor.

THE SHIFT GRID IS THE PART THAT FAILS SILENTLY. Too small a range drops long edges
and nothing raises; the energy just moves a little. The range per axis is
ceil(cutoff / d_perp), where d_perp is the interplanar spacing V / |a_j x a_k| -- NOT
cutoff / |a_i|, which is wrong for oblique cells and wrong in the unsafe direction
(it under-counts exactly when the cell is most sheared). Pinned in the tests against
matscipy on triclinic cells.

EDGE ORDER IS NOT PRESERVED, deliberately. matscipy emits in its own cell-list order;
any other algorithm produces the same SET in a different sequence. MACE sums over
edges, so a reordering perturbs only float summation order -- the same class as the
tf32 noise already present. The tests therefore assert an identical edge SET (exact,
order-insensitive) and energies inside a same-path control, rather than pretending to
a bitwise array match that no reimplementation could hold.
"""
import os

import numpy as np
import torch

#: Starting max_num_neighbors for the radius-search fast path.
#:
#: Chosen from MEASURED true maxima, not from the shift count: ~141 neighbours on
#: physical CSD cells at a 6 A cutoff, ~2710 on the degenerate cells a policy emits
#: before it learns. 4096 clears both, and overflow escalates rather than truncating,
#: so this is a performance choice with a correctness backstop -- not a silent bound.
RADIUS_CAP_START = 4096

#: Which branch actually ran, counted per call.
#:
#: THE POINT IS THE CLUSTER. `_pairs_by_radius_search` returns None and falls back to
#: the O(sum n^2 * K) all-pairs kernel when torch_cluster is not importable, and that
#: fallback is SILENT: the answer stays correct and only the speed collapses, so it
#: presents as "the GPU is slow" rather than as a missing dependency. torch_cluster
#: imports on the dev box; nothing has verified it on the cluster. Surfaced as a
#: metric so a run reports which path it took instead of leaving it to be inferred
#: from a disappointing throughput number.
_PATH_CALLS = {'radius': 0, 'allpairs': 0}

#: Per-axis ceiling on the periodic shift range, and the reason it has to exist.
#:
#: `lattice_shift_range` divides the cutoff by the interplanar spacing, and that
#: spacing goes to ZERO as a cell flattens -- so `need` is unbounded from above.
#: It is deliberately clamped only from BELOW there (a degenerate cell must not
#: silently produce no periodic edges), which leaves the top open.
#:
#: That would be survivable if each graph paid for its own geometry. It does not:
#: `batched_pbc_neighbour_list` takes `.max(dim=0)` over the batch and ghost-
#: expands EVERY graph on the worst cell's grid, so one degenerate crystal sets
#: the memory for all of them. MEASURED 2026-08-20, 128 acridine graphs / 2944
#: atoms / 6 A cutoff, squashing ONE cell's third axis:
#:
#:     physical        grid [3,3,2]     K=245       peak  254 MiB
#:     x0.01           grid [3,3,52]    K=5,145     peak 1725 MiB
#:     x0.003          grid [3,3,171]   K=16,807    peak 3825 MiB
#:     x0.001          grid [3,3,510]   K=50,029    OOM (16 GB card)
#:
#: Edge COUNT rose only 2.5x across that range while memory rose 15x, so it is
#: not the answer getting bigger -- it is ghost expansion the 127 sane cells did
#: not need. An untrained policy emits exactly these cells, which is why the
#: failure is an early-training transient that vanishes once geometry is sane.
#:
#: 8 rather than the measured physical maximum of 3: this is a MEMORY BOUND, not
#: a modelling choice, and it must never be the thing deciding a physical cell's
#: edge set. 8 bounds K at 17^3 = 4913 while leaving 2.7x headroom over what real
#: acridine needs at this cutoff. A larger cutoff or a long thin cell legitimately
#: needs more, which is why binding it is COUNTED and reported rather than
#: silently applied -- see `_SHIFT_CAP_CALLS`.
#:
#: NOT a cap on `max_num_neighbors`: that truncates the neighbour list of a cell
#: whose geometry is fine, which is the silent-wrong-energy failure this module
#: exists to avoid and the one F-047 removed from the UMA path. Capping SHIFTS
#: drops distant images of a cell that is already geometrically meaningless.
#: Overridable so the cap is a SINGLE-KEY TOGGLE rather than a code revision:
#: without this, comparing capped against uncapped means comparing two builds,
#: and a cross-version A/B cannot isolate the cap from anything else that moved.
#: A very large value (e.g. 100000) restores the old unbounded behaviour for the
#: control arm. Matches how the other MLIP path switches are expressed
#: (MXT_BATCHED_MACE_NEIGHBOURS, MXT_UMA_EXTERNAL_GRAPH), so a battery can put it
#: in an <arm>.env beside the config.
MAX_SHIFT_RANGE = int(os.environ.get('MXT_MAX_SHIFT_RANGE', '8'))

#: Calls in which the cap actually bound, and the largest range it refused. A cap
#: that fires without saying so is the same defect as a flag that reports success
#: without executing.
_SHIFT_CAP_CALLS = {'calls': 0, 'capped': 0, 'max_requested': 0}


def drain_neighbour_path_counts():
    """Pop the per-branch call counts. {} when the list was never built."""
    total = _PATH_CALLS['radius'] + _PATH_CALLS['allpairs']
    if not total:
        return {}
    out = {'energy/nl_radius_calls': _PATH_CALLS['radius'],
           'energy/nl_allpairs_calls': _PATH_CALLS['allpairs'],
           # >0 means at least one cell in the batch was degenerate enough to ask
           # for a shift grid past MAX_SHIFT_RANGE. Its periodic images are
           # incomplete from that point -- acceptable only because such a cell is
           # geometrically meaningless and the bounding term rejects it anyway.
           'energy/nl_shift_capped_frac':
               _SHIFT_CAP_CALLS['capped'] / max(_SHIFT_CAP_CALLS['calls'], 1),
           'energy/nl_shift_max_requested': _SHIFT_CAP_CALLS['max_requested'],
           # 1.0 = fast path throughout; anything less means the fallback is running
           # and the neighbour list is costing orders of magnitude more than it should
           'energy/nl_fastpath_frac': _PATH_CALLS['radius'] / total}
    _PATH_CALLS['radius'] = _PATH_CALLS['allpairs'] = 0
    _SHIFT_CAP_CALLS.update(calls=0, capped=0, max_requested=0)
    return out


def lattice_shift_range(cells: torch.Tensor, cutoff: float,
                        frac_spread: torch.Tensor = None) -> torch.Tensor:
    """
    Per-axis integer shift range needed to cover `cutoff`, for each graph.

    cells: [G, 3, 3], ROW-vector lattice (row i is lattice vector a_i), matching what
    get_neighborhood is handed (`T_fc.transpose(-2, -1)`).
    frac_spread: [G, 3], per-axis span of the FRACTIONAL coordinates
    (f_max - f_min). Omit ONLY if every atom is known to lie inside one cell.

    Returns [G, 3] non-negative ints: axis i needs shifts in [-n_i, +n_i].

    TWO TERMS, AND THE SECOND ONE IS NOT OPTIONAL HERE.

    d_perp_i = V / |a_j x a_k| is the true interplanar spacing. Using cutoff/|a_i|
    instead under-counts on sheared cells -- silently, by dropping long edges.

    The spread term is the one that actually bit. `ceil(cutoff / d_perp)` is the
    textbook range and it assumes every atom sits INSIDE the cell. `unit_cell_pos` is
    not wrapped: molecules straddle cell boundaries, so fractional coordinates span
    more than one cell width and a pair can need one more shell than the geometry
    alone implies. For a pair to interact we need |df_k + S_k| * d_perp_k <= cutoff
    with |df_k| <= spread_k, hence n_k = ceil(cutoff / d_perp_k + spread_k).

    Measured: on real CSD crystals this was 46 missing edges out of 2704 on one graph
    -- no error, no shape change, just a slightly wrong energy. It survived a 43-test
    suite because those fixtures generated positions as `rand(n,3) @ cell`, i.e.
    fractional coordinates in [0, 1), where ceil() absorbs the whole spread term.
    """
    a1, a2, a3 = cells[:, 0], cells[:, 1], cells[:, 2]
    cross = torch.stack((torch.cross(a2, a3, dim=-1),
                         torch.cross(a3, a1, dim=-1),
                         torch.cross(a1, a2, dim=-1)), dim=1)      # [G,3,3]
    volume = (a1 * torch.cross(a2, a3, dim=-1)).sum(-1).abs()      # [G]
    areas = cross.norm(dim=-1)                                     # [G,3]
    # a degenerate cell would divide by ~0; clamp so the range comes out large
    # (conservative, more shifts) rather than zero (silently no periodic edges)
    d_perp = volume.unsqueeze(-1) / areas.clamp_min(1e-12)
    need = cutoff / d_perp.clamp_min(1e-12)
    if frac_spread is not None:
        need = need + frac_spread
    rng = torch.ceil(need).long().clamp_min(0)
    # CEILING, and it is a memory bound rather than a modelling choice -- see
    # MAX_SHIFT_RANGE. Counted, never silent: a capped call has an incomplete
    # periodic image set, and a reader must be able to tell that from a call that
    # simply had sane geometry.
    _SHIFT_CAP_CALLS['calls'] += 1
    biggest = int(rng.max()) if rng.numel() else 0
    if biggest > MAX_SHIFT_RANGE:
        _SHIFT_CAP_CALLS['capped'] += 1
        _SHIFT_CAP_CALLS['max_requested'] = max(
            _SHIFT_CAP_CALLS['max_requested'], biggest)
        rng = rng.clamp_max(MAX_SHIFT_RANGE)
    return rng


def fractional_spread(pos: torch.Tensor, batch_ind: torch.Tensor,
                      cells: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Per-graph, per-axis span of fractional coordinates: [G, 3]. Row-vector
    lattice, so f = r @ A^-1."""
    inv = torch.linalg.inv(cells)                                  # [G,3,3]
    frac = torch.einsum('nd,ndk->nk', pos, inv[batch_ind])         # [N,3]
    big = torch.full((num_graphs, 3), -float('inf'), dtype=pos.dtype, device=pos.device)
    small = torch.full((num_graphs, 3), float('inf'), dtype=pos.dtype, device=pos.device)
    f_max = big.index_reduce(0, batch_ind, frac, 'amax', include_self=True)
    f_min = small.index_reduce(0, batch_ind, frac, 'amin', include_self=True)
    spread = (f_max - f_min)
    return torch.where(torch.isfinite(spread), spread, torch.zeros_like(spread))


def _shift_grid(n_max: torch.Tensor, device) -> torch.Tensor:
    """All integer shifts in the box [-n, +n] per axis. [K, 3], K = prod(2n+1)."""
    axes = [torch.arange(-int(n), int(n) + 1, device=device, dtype=torch.long)
            for n in n_max]
    return torch.cartesian_prod(*axes).reshape(-1, 3)


def _wrap(pos, batch_ind, cells):
    """
    Fractional-wrap into [0, 1), returning (wrapped cartesian, integer wrap vector).

    WHY WRAP AT ALL. `unit_cell_pos` is unwrapped: measured fractional spread on real
    CSD crystals reaches 2.4 cell widths, which forces the shift grid out to [4,3,2]
    (K=315). Wrapped, the spread is 1 by construction and the grid is [3,2,2] (K=175).

    The shift correction is exact, not an approximation. For an edge found between
    WRAPPED positions with shift S_w, the shift valid for the UNWRAPPED positions the
    model is actually handed is

        S_true = S_w + w_i - w_j

    since r^u = r^w + w @ A, so |r_j^u + S_t A - r_i^u| = |r_j^w + (w_j + S_t - w_i) A - r_i^w|.
    """
    inv = torch.linalg.inv(cells)
    frac = torch.einsum('nd,ndk->nk', pos, inv[batch_ind])
    w = torch.floor(frac)
    frac_w = frac - w
    pos_w = torch.einsum('nd,ndk->nk', frac_w, cells[batch_ind])
    return pos_w, w.long()


def _pairs_by_radius_search(pos_w, batch_ind, cells, shifts, cutoff, num_graphs):
    """
    Ghost-image expansion + one radius search, or None if torch_cluster is absent.

    O(N*K) rather than the O(sum_g n_g^2 * K) of the all-pairs fallback: at 92 atoms
    and K=175 that is 16k ghosts per graph instead of 2.7M pair evaluations, ~92x
    less work, and it is why this path exists at all.

    max_num_neighbors is CHECKED, not trusted. torch_cluster silently truncates at the
    cap -- that is the failure this whole module is written to avoid, and a cap that
    quietly bites would produce a plausible wrong energy with no error anywhere.

    THE CAP IS ESCALATED, NOT SET TO k*64. torch_cluster sizes its output buffer from
    max_num_neighbors, so an oversized cap costs memory in proportion to how wrong it
    is, and k*64 is wrong by a lot: measured true maxima are ~141 neighbours on
    physical CSD cells and ~2710 on the degenerate cells a policy emits early in
    training, against a k*64 of 21952 and 118976 respectively. That cost the whole
    card -- 18 GiB on 384 physical graphs, 33 GiB on 128 degenerate ones, and an
    outright OOM at 256 degenerate. The starting cap below covers both regimes with
    headroom; overflow escalates rather than failing, and only a cap beyond the old
    k*64 ceiling raises.

    Escalation is safe precisely BECAUSE the count is checked: a too-small cap is
    detected and retried, never accepted. Verified to produce identical edge sets to
    the k*64 cap on physical and degenerate cells alike.
    """
    try:
        from torch_cluster import radius
    except ImportError:
        return None

    n_atoms = pos_w.shape[0]
    k = shifts.shape[0]
    # offset[n, k] = shifts[k] @ cell[graph(n)]. Distinct subscripts for the SHIFT
    # index and the LATTICE index -- reusing 'k' for both silently broadcasts.
    offsets = torch.einsum('kd,ndc->nkc', shifts.to(cells.dtype), cells[batch_ind])
    ghost_pos = (pos_w.unsqueeze(1) + offsets).reshape(-1, 3)     # [N*K, 3]
    ghost_parent = torch.arange(n_atoms, device=pos_w.device).repeat_interleave(k)
    ghost_shift = shifts.repeat(n_atoms, 1)
    ghost_batch = batch_ind.repeat_interleave(k)

    # strictly increasing, and never above the number of ghosts that exist: a cap
    # larger than the candidate set buys nothing and costs buffer. sorted(set(...))
    # so a small shift grid cannot produce a ladder that steps DOWN.
    ceiling = min(int(k * 64), max(n_atoms * k, 1))
    ladder = sorted({min(RADIUS_CAP_START, ceiling),
                     min(RADIUS_CAP_START * 4, ceiling), ceiling})
    for cap in ladder:
        # radius(x, y, r, batch_x, batch_y): for every point in y, the points in x
        # within r. y = the real atoms, x = the ghosts.
        assign = radius(ghost_pos, pos_w, float(cutoff), ghost_batch, batch_ind,
                        max_num_neighbors=cap)
        i_idx, g_idx = assign[0], assign[1]
        counts = torch.bincount(i_idx, minlength=n_atoms)
        if int(counts.max()) < cap:
            return i_idx, ghost_parent[g_idx], ghost_shift[g_idx]
        del assign, i_idx, g_idx, counts      # the retry needs the memory back
    raise RuntimeError(
        f'radius search hit max_num_neighbors={ceiling} even after escalation; the '
        f'neighbour list is TRUNCATED and every energy downstream would be quietly '
        f'wrong')


def batched_pbc_neighbour_list(pos: torch.Tensor,
                               batch_ind: torch.Tensor,
                               cells: torch.Tensor,
                               cutoff: float,
                               pbc: bool = True):
    """
    Neighbour lists for every graph at once, on `pos.device`.

    pos        [N, 3]      unit-cell atom positions, graphs concatenated
    batch_ind  [N]         graph id per atom, SORTED (PyG layout)
    cells      [G, 3, 3]   row-vector lattice per graph
    cutoff     float
    pbc        bool        False = no periodic images; only the S == 0 block runs

    Returns (edge_index [2, E] with GLOBAL atom indices, unit_shifts [E, 3] int,
             graph_id [E]). Convention matches mace.data.get_neighborhood:
    edge (i, j, S) means the displacement is pos[j] - pos[i] + S @ cell, so the
    condition is |pos[j] + S @ cell - pos[i]| < cutoff, and i == j is kept only when
    S != 0 (a true self-edge across a boundary is a real interaction).
    """
    device = pos.device
    num_graphs = cells.shape[0]
    counts = torch.bincount(batch_ind, minlength=num_graphs)
    n_max = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts

    # ---- fast path: wrap, ghost-expand, one radius search ----
    if pbc:
        pos_w, wrap_v = _wrap(pos, batch_ind, cells)
        # spread is 1 by construction after wrapping, so the grid is the geometric
        # range plus one shell, not the raw (up to 2.4 cell widths) data spread
        rng = lattice_shift_range(cells, cutoff,
                                  torch.ones_like(cells[:, :, 0])).max(dim=0).values
        fast = _pairs_by_radius_search(pos_w, batch_ind, cells,
                                       _shift_grid(rng, device), cutoff, num_graphs)
        _PATH_CALLS['radius' if fast is not None else 'allpairs'] += 1
        if fast is not None:
            i_g, j_g, s_w = fast
            # back to shifts valid for the UNWRAPPED positions the model receives
            s_true = s_w + wrap_v[i_g] - wrap_v[j_g]
            keep = ~((i_g == j_g) & (s_true == 0).all(dim=1))
            i_g, j_g, s_true = i_g[keep], j_g[keep], s_true[keep]
            return (torch.stack((i_g, j_g)), s_true, batch_ind[i_g])

    # --- pad to [G, n_max, 3] so every shift pass is one batched op ---
    slot = torch.arange(pos.shape[0], device=device) - starts[batch_ind]
    pos_pad = pos.new_zeros((num_graphs, n_max, 3))
    pos_pad[batch_ind, slot] = pos
    valid = torch.zeros((num_graphs, n_max), dtype=torch.bool, device=device)
    valid[batch_ind, slot] = True
    pair_valid = valid[:, :, None] & valid[:, None, :]              # [G, n, n]

    if pbc:
        # spread included: unit_cell_pos is NOT wrapped into the cell, and without
        # this term real crystals silently lose their longest edges
        spread = fractional_spread(pos, batch_ind, cells, num_graphs)
        shifts = _shift_grid(
            lattice_shift_range(cells, cutoff, spread).max(dim=0).values, device)
    else:
        shifts = torch.zeros((1, 3), dtype=torch.long, device=device)

    cut2 = float(cutoff) ** 2
    eye = torch.eye(n_max, dtype=torch.bool, device=device).unsqueeze(0)

    out_i, out_j, out_s, out_g = [], [], [], []
    for s in shifts:
        # offset per graph: S @ cell   (row-vector lattice, so this is a row combo)
        off = (s.to(cells.dtype).unsqueeze(0) @ cells).squeeze(1)   # [G, 3]
        d = pos_pad[:, None, :, :] + off[:, None, None, :] - pos_pad[:, :, None, :]
        keep = (d.pow(2).sum(-1) < cut2) & pair_valid
        if not bool(s.any()):
            keep = keep & ~eye          # drop TRUE self-edges only (S == 0)
        if not keep.any():
            continue
        g, i, j = keep.nonzero(as_tuple=True)
        out_i.append(starts[g] + i)     # back to global atom indices
        out_j.append(starts[g] + j)
        out_s.append(s.unsqueeze(0).expand(g.shape[0], 3))
        out_g.append(g)

    if not out_i:
        empty = torch.zeros((2, 0), dtype=torch.long, device=device)
        return (empty, torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device))

    return (torch.stack((torch.cat(out_i), torch.cat(out_j))),
            torch.cat(out_s), torch.cat(out_g))


def get_neighborhood_batched_numpy(positions, cutoff, pbc, cell):
    """
    Drop-in for mace.data.get_neighborhood on ONE graph, backed by the batched kernel.

    Exists for the equivalence tests: it lets the new path be compared against the
    reference one graph at a time, with the reference's own quirks reproduced --

      * the non-periodic branch REWRITES `cell` in place and the caller uses the
        RETURNED cell, so that side effect is preserved deliberately;
      * `shifts = unit_shifts @ cell` uses the (possibly rewritten) cell.

    Production should call batched_pbc_neighbour_list directly on the whole batch;
    this wrapper throws away the batching that is the entire point.
    """
    cell = np.array(cell, dtype=float)
    if cell is None or not cell.any():
        cell = np.identity(3, dtype=float)
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    for ax in range(3):
        if not pbc[ax]:
            cell[ax, :] = max_positions * 5 * cutoff * identity[ax, :]

    pos_t = torch.as_tensor(positions, dtype=torch.float64)
    cells_t = torch.as_tensor(cell, dtype=torch.float64).unsqueeze(0)
    b = torch.zeros(pos_t.shape[0], dtype=torch.long)
    edge_index, unit_shifts, _ = batched_pbc_neighbour_list(
        pos_t, b, cells_t, cutoff, pbc=all(pbc))

    unit_shifts_np = unit_shifts.cpu().numpy().astype(float)
    return (edge_index.cpu().numpy(), unit_shifts_np @ cell, unit_shifts_np, cell)

"""Radius-graph edge construction for batched point sets.

``build_radial_graph`` returns edge indices either for a plain within-graph
radius graph or, when per-node labels ``aux_ind`` are given, for the
inside/outside split used on molecular clusters: intramolecular edges among
the inside nodes, and intermolecular edges that end on inside nodes.
``asymmetric_radius_graph`` builds the directed edges for the second case.

The module-level name ``radius`` is ``torch_cluster.radius``: the
``from torch_cluster import radius`` statement that follows the local
``radius`` definition rebinds the name, so ``radius`` imported from this
module is the torch_cluster function.
"""
from typing import Optional

import torch
from torch_geometric import nn as gnn


def radius(x: torch.Tensor, y: torch.Tensor, r: float,
           batch_x: Optional[torch.Tensor] = None,
           batch_y: Optional[torch.Tensor] = None,
           max_num_neighbors: int = 32,
           num_workers: int = 1) -> torch.Tensor:
    r"""Find, for each point in `y`, the points in `x` within distance `r`.

    This definition is shadowed. The module later executes
    ``from torch_cluster import radius``, which rebinds the module-level name
    ``radius`` to ``torch_cluster.radius``; importing ``radius`` from this
    module yields the torch_cluster function, not this one. If this body is
    reached through another reference, it reshapes 1-D inputs to [N, 1],
    makes both inputs contiguous and forwards every argument positionally to
    the module-level ``radius`` (``torch_cluster.radius``).

    Parameters
    ----------
    x : torch.Tensor
        [N, F] points searched over; a 1-D tensor is treated as [N, 1].
    y : torch.Tensor
        [M, F] query points; a 1-D tensor is treated as [M, 1].
    r : float
        Search radius, in the units of `x` and `y`.
    batch_x : torch.Tensor, optional
        [N] graph index of each point in `x`; must be sorted. None places all
        points in one graph.
    batch_y : torch.Tensor, optional
        [M] graph index of each point in `y`; must be sorted. None places all
        points in one graph.
    max_num_neighbors : int, optional
        Maximum number of points of `x` returned per point of `y`; when more
        are in range, torch_cluster chooses which are kept. Default 32.
    num_workers : int, optional
        Worker count forwarded to torch_cluster. Default 1.

    Returns
    -------
    torch.Tensor
        [2, num_edges] long tensor; row 0 indexes `y`, row 1 indexes `x`.
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    # ptr_x: Optional[torch.Tensor] = None
    # if batch_x is not None:
    #     assert x.size(0) == batch_x.numel()
    #     batch_size = int(batch_x.max()) + 1
    #
    #     deg = x.new_zeros(batch_size, dtype=torch.long)
    #     deg.scatter_add_(0, batch_x, torch.ones_like(batch_x))
    #
    #     ptr_x = deg.new_zeros(batch_size + 1)
    #     torch.cumsum(deg, 0, out=ptr_x[1:])
    #
    # ptr_y: Optional[torch.Tensor] = None
    # if batch_y is not None:
    #     assert y.size(0) == batch_y.numel()
    #     batch_size = int(batch_y.max()) + 1
    #
    #     deg = y.new_zeros(batch_size, dtype=torch.long)
    #     deg.scatter_add_(0, batch_y, torch.ones_like(batch_y))
    #
    #     ptr_y = deg.new_zeros(batch_size + 1)
    #     torch.cumsum(deg, 0, out=ptr_y[1:])
    return radius(
        x,
        y,
        r,
        batch_x,
        batch_y,
        max_num_neighbors,
        num_workers
    )

    # update usage of torch_cluster.radius
    # return torch.ops.torch_cluster.radius(x,
    #                                       y,
    #                                       ptr_x,
    #                                       ptr_y,
    #                                       r,
    #                                       max_num_neighbors,
    #                                       num_workers)


from torch_cluster import radius


# @torch.jit.script
def asymmetric_radius_graph(x: torch.Tensor,
                            r: float,
                            inside_inds: torch.Tensor,
                            convolve_inds: torch.Tensor,
                            batch: torch.Tensor,
                            loop: bool = False,
                            max_num_neighbors: int = 32,
                            flow: str = 'source_to_target',
                            num_workers: int = 1) -> torch.Tensor:
    r"""Directed radius-graph edges from `convolve_inds` nodes to `inside_inds` nodes.

    For each node in ``x[inside_inds]``, ``torch_cluster.radius`` finds the
    nodes of ``x[convolve_inds]`` within distance `r` (restricted to the same
    graph when `batch` is given); both ends are then mapped back to indices
    into `x`.

    Parameters
    ----------
    x : torch.Tensor
        [num_nodes, F] node positions; distances are Euclidean over the last
        dimension.
    r : float
        Radius, in the units of `x`.
    inside_inds : torch.Tensor
        Long indices into `x` of the query (target) nodes.
    convolve_inds : torch.Tensor or None
        Long indices into `x` of the candidate neighbour (source) nodes. None
        uses every node, ``torch.arange(len(x))``.
    batch : torch.Tensor or None
        [num_nodes] graph index of each node; ``batch[inside_inds]`` and
        ``batch[convolve_inds]`` must be sorted. None treats all nodes as one
        graph.
    loop : bool, optional
        If False (default), edges whose two ends are the same node of `x` are
        removed after the search.
    max_num_neighbors : int, optional
        Neighbour cap per query node passed to ``torch_cluster.radius``;
        ``max_num_neighbors + 1`` is passed when `loop` is False. Default 32.
    flow : {'source_to_target', 'target_to_source'}, optional
        Row order of the output. Default 'source_to_target'.
    num_workers : int, optional
        Forwarded to ``torch_cluster.radius``. Default 1.

    Returns
    -------
    torch.Tensor
        [2, num_edges] long tensor of indices into `x`. With
        ``flow='source_to_target'`` row 0 holds the `convolve_inds` node
        (source) and row 1 the `inside_inds` node (target);
        ``'target_to_source'`` swaps the rows.

    Raises
    ------
    AssertionError
        If `flow` is not one of the two accepted strings.

    Notes
    -----
    Each (target, source) pair found gives one directed edge; when
    `inside_inds` and `convolve_inds` are the same set, each pair appears in
    both directions. The extra neighbour allowed when `loop` is False is only
    removed again if it is a self-match, so a query node can keep up to
    ``max_num_neighbors + 1`` edges.
    """
    if convolve_inds is None:  # indexes of items within x to convolve against y
        convolve_inds = torch.arange(len(x))

    assert flow in ['source_to_target', 'target_to_source']
    if batch is not None:
        edge_index = radius(x[convolve_inds], x[inside_inds], r, batch[convolve_inds], batch[inside_inds],
                            max_num_neighbors if loop else max_num_neighbors + 1,
                            num_workers)
    else:
        edge_index = radius(x[convolve_inds], x[inside_inds], r, None, None,
                            max_num_neighbors if loop else max_num_neighbors + 1,
                            num_workers)

    target, source = edge_index[0], edge_index[1]

    # edge_index[1] = inside_inds[edge_index[1, :]] # reindex
    target = inside_inds[target]  # contains correct indexes
    source = convolve_inds[source]

    if flow == 'source_to_target':
        row, col = source, target
    else:
        row, col = target, source

    if not loop:  # now properly deletes self-loops
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


def build_radial_graph(pos: torch.FloatTensor,
                       batch: torch.LongTensor,
                       ptr: torch.LongTensor,
                       cutoff: float,
                       max_num_neighbors: int,
                       aux_ind: torch.LongTensor=None,
                       mol_ind: torch.LongTensor=None,
                       ):
    r"""Build radius-graph edge indices for a batch of point sets.

    Without `aux_ind`, returns a plain radius graph within each graph of
    `batch`. With `aux_ind`, edges are built only towards the
    ``aux_ind == 0`` ("inside") nodes: intramolecular edges among inside
    nodes, and intermolecular edges from ``aux_ind == 1`` ("outside") nodes
    and from inside nodes of other molecules. Nodes with ``aux_ind == 2``
    appear in no edge.

    Parameters
    ----------
    pos : torch.Tensor
        [num_nodes, 3] node positions.
    batch : torch.Tensor
        [num_nodes] graph index of each node; must be sorted
        (non-decreasing).
    ptr : torch.Tensor
        [num_graphs + 1] graph offsets. Only ``len(ptr) - 1`` (the number of
        graphs) is used, and only when `aux_ind` is given.
    cutoff : float
        Edge radius, in the units of `pos`.
    max_num_neighbors : int
        Neighbour cap per query node, passed to the underlying radius search.
    aux_ind : torch.Tensor, optional
        [num_nodes] node label with values in {0, 1, 2}: 0 = inside,
        1 = outside neighbour candidate, 2 = excluded. dtype int64, int32 or
        bool.
    mol_ind : torch.Tensor, optional
        [num_nodes] molecule index of each node. Required when `aux_ind` is
        given; ignored otherwise.

    Returns
    -------
    dict
        Without `aux_ind`: ``{'edge_index': [2, num_edges]}`` from
        ``torch_geometric.nn.radius_graph`` (``flow='source_to_target'``, no
        self-loops).

        With `aux_ind`:

        - ``'edge_index'``: [2, num_intra_edges] edges between inside nodes
          with equal `mol_ind`, in both directions, without self-loops.
        - ``'edge_index_inter'``: [2, num_inter_edges]; row 0 is the source
          (an outside node, or an inside node of a different molecule) and
          row 1 the inside target. Outside-to-inside pairs appear once;
          pairs of inside nodes in different molecules appear in both
          directions. Each inside node therefore has, exactly once as a
          source (up to the neighbour cap), every ``aux_ind == 1`` node
          within `cutoff` (whatever its `mol_ind`) and every
          ``aux_ind == 0`` node within `cutoff` with a different `mol_ind`.
        - ``'inside_inds'``: [num_inside] long indices of the
          ``aux_ind == 0`` nodes.
        - ``'outside_inds'``: [num_outside] long indices of the
          ``aux_ind == 1`` nodes.
        - ``'inside_batch'``: [num_inside] ``batch[inside_inds]``.
        - ``'n_repeats'``: Python list of ``len(ptr) - 1`` ints, each graph's
          total node count (all labels) integer-divided by its inside node
          count.

    Raises
    ------
    AssertionError
        If `aux_ind` has another dtype or contains values outside {0, 1, 2}.
    TypeError
        If `aux_ind` is given and `mol_ind` is None; `mol_ind` is indexed
        unconditionally on that path.

    Notes
    -----
    With `aux_ind`, the inside-inside and outside-inside searches are two
    separate ``asymmetric_radius_graph`` calls, each applying
    `max_num_neighbors` on its own.
    """
    if aux_ind is not None:  # there is an 'inside' 'outside' distinction
        assert aux_ind.dtype in (torch.int64, torch.int32, torch.bool)
        assert (aux_ind == 0).sum() + (aux_ind == 1).sum() + (aux_ind == 2).sum() == aux_ind.numel()

        inside_bool = aux_ind == 0
        outside_bool = aux_ind == 1
        inside_inds = torch.where(inside_bool)[0]
        # atoms which are not in the asymmetric unit but which we will convolve - pre-excluding many from outside the cutoff
        outside_inds = torch.where(outside_bool)[0]
        inside_batch = batch[inside_inds]  # get the feature vectors we want to repeat

        # counts of atoms per molecule in full batch
        batch_counts = torch.bincount(batch, minlength=len(ptr) - 1)

        # counts of atoms per molecule in the inside region
        inside_counts = torch.bincount(inside_batch, minlength=len(ptr) - 1)

        n_repeats = (batch_counts // inside_counts).tolist()

        # within aunit edges - includes intermolecular edges when Z'>1
        edge_index = asymmetric_radius_graph(pos, batch=batch, r=cutoff,
                                             # intramolecular interactions - stack over range 3 convolutions
                                             max_num_neighbors=max_num_neighbors, flow='source_to_target',
                                             inside_inds=inside_inds, convolve_inds=inside_inds)

        # intermolecular edges
        edge_index_inter = asymmetric_radius_graph(pos, batch=batch, r=cutoff,
                                                   # extra radius for intermolecular graph convolution
                                                   max_num_neighbors=max_num_neighbors, flow='source_to_target',
                                                   inside_inds=inside_inds, convolve_inds=outside_inds)


        # disaggregate within-aunit intermolecular edges from genuine intramolecular edges
        # necessary for Z'>1 structures
        intra_edge = mol_ind[edge_index[0]] == mol_ind[edge_index[1]]
        if torch.any(~intra_edge):
            # if there are any within aunit intra edges
            edge_index_inter = torch.cat([edge_index_inter, edge_index[:, ~intra_edge]], dim=1)
            edge_index = edge_index[:, intra_edge]

        return {'edge_index': edge_index,
                'edge_index_inter': edge_index_inter,
                'inside_inds': inside_inds,
                'outside_inds': outside_inds,
                'inside_batch': inside_batch,
                'n_repeats': n_repeats}

    else:

        edge_index = gnn.radius_graph(pos,
                                      r=cutoff,
                                      batch=batch,
                                      max_num_neighbors=max_num_neighbors,
                                      flow='source_to_target')  # note - requires batch be monotonically increasing

        return {'edge_index': edge_index}

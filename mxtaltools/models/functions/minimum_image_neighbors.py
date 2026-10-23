"""Minimum-image neighbour edges for a single periodic cell."""
import torch


def argwhere_minimum_image_convention_edges(num_graphs, pos, T_fc, cutoff):
    """Edges between atom pairs of one periodic cell within `cutoff` under the minimum-image convention.

    Parameters
    ----------
    num_graphs : int
        Must be 1; only a single cell is supported.
    pos : torch.Tensor
        [num_nodes, 3] Cartesian positions.
    T_fc : torch.Tensor
        [3, 3] fractional-to-Cartesian matrix, Cartesian = ``T_fc @ frac``
        for column vectors. For a 3-D tensor, ``T_fc[0]`` is used.
    cutoff : float
        Maximum edge length, in the units of `pos`.

    Returns
    -------
    dict
        ``'edge_index'``: [2, num_edges] long tensor of node pairs (i, j),
        with both orderings of each pair present. ``'dists'``: [num_edges]
        minimum-image distance of each edge.

    Raises
    ------
    AssertionError
        If `num_graphs` is not 1.

    Notes
    -----
    Positions are converted to fractional coordinates and wrapped into
    [0, 1); each component of every fractional displacement is shifted by
    its nearest integer and the result is mapped back to Cartesian. Each
    ordered pair therefore contributes exactly one image, the one selected by
    per-component rounding, which is the shortest image in an orthogonal cell
    but not necessarily in an oblique one. Edges satisfy
    ``0 < distance <= cutoff``, so self-pairs and coincident atoms are
    excluded. Dense [num_nodes, num_nodes] displacement and distance
    matrices are built.
    """
    assert num_graphs == 1  # this only works one at a time
    # restrict particles individually to box
    if T_fc.ndim == 3:
        T_fc = T_fc[0, ...]
    frac_coords = pos @ torch.linalg.inv(T_fc.T)
    frac_coords -= torch.floor(frac_coords)
    # B.9 in Tuckerman
    # convert to fractional
    # get pointwise differences
    # subtract nearest integer
    # transform back to cartesian
    fdistmats = torch.stack([
        frac_coords[:, ind, None] - frac_coords[None, :, ind]
        for ind in range(3)])
    fdistmats -= torch.round(fdistmats)
    distmats = fdistmats.permute((1, 2, 0)) @ T_fc.T
    norms = torch.linalg.norm(distmats, dim=-1)
    a, b = torch.where((norms > 0) * (norms <= cutoff))  # faster but still pretty slow
    edge_index = torch.cat((a[None, :], b[None, :]), dim=0)
    dist = norms[edge_index[0], edge_index[1]]

    return {'edge_index': edge_index, 'dists': dist}

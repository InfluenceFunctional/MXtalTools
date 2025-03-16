import torch


def argwhere_minimum_image_convention_edges(num_graphs, pos, T_fc, cutoff):
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

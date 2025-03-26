"""
for a single graph
"""

import torch
import numpy as np
import torch_geometric.nn as gnn

periodic = True
device = 'cpu'

# particle positions
pos = torch.Tensor(np.load(r'C:\Users\mikem\crystals\classifier_runs\single_cluster.npy', allow_pickle=True)).to(device)
batch = torch.ones(len(pos)).to(device)
cutoff = 6

# transpose of box vectors
T_fc = torch.tensor([[30.1598, 0.0000, -6.3270],
                     [0.0000, 21.4887, 0.0000],
                     [0.0000, 0.0000, 29.8945]], device=device)

# what we want is the edge indices and distances

if periodic:  # get radial embeddings periodically using minimum image convention

    # we sometimes use unwrapped molecule coords
    # so we first have to force all atoms into the cell
    frac_coords = pos @ torch.linalg.inv(T_fc.T)
    frac_coords -= torch.floor(frac_coords)

    # B.9 in Tuckerman
    # convert to fractional
    # get pointwise differences
    # subtract nearest integer
    # transform back to cartesian
    # I tried this earlier in cartesian form with the LAMMPS method but couldn't quite get it to work perfectly
    # in any case this is pretty fast if you have enough RAM/VRAM even for thousands of atoms

    # get parwise distances on a, b, c fractional axes
    fractional_distmats = torch.stack([
        frac_coords[:, ind, None] - frac_coords[None, :, ind]
        for ind in range(3)])

    fractional_distmats -= torch.round(fractional_distmats)  # this is the MIC

    # convert back to cartesian, get the distances and edges under the cutoff
    cartesian_distmats = fractional_distmats.permute((1, 2, 0)) @ T_fc.T
    norms = torch.linalg.norm(cartesian_distmats, dim=-1)
    a, b = torch.where((norms > 0) * (norms <= cutoff))  # faster but still pretty slow
    edge_index = torch.cat((a[None, :], b[None, :]), dim=0)
    dist = norms[edge_index[0], edge_index[1]]

else:  # just get the radial graph the normal way
    edge_index = gnn.radius_graph(pos, r=cutoff, batch=batch,
                                  max_num_neighbors=100,
                                  flow='source_to_target')  # note - requires batch be monotonically increasing
    i, j = edge_index  # i->j source-to-target
    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()


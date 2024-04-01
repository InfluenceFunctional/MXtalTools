import torch
import torch.nn as nn

from models.GraphNeuralNetwork import EmbeddingBlock, GCBlock
from models.basis_functions import BesselBasisLayer, GaussianEmbedding
from models.components import MLP, construct_radial_graph
from models.global_aggregation import global_aggregation


class MoleculeClassifier(nn.Module):
    def __init__(self,
                 input_node_depth: int,
                 node_embedding_depth: int,
                 nodewise_fc_layers: int,
                 message_depth: int,
                 convolution_type: str,
                 graph_embedding_depth: int,
                 num_fcs: int,
                 fc_norm,
                 num_blocks: int,
                 num_radial: int,
                 output_dimension: int,
                 num_embedding_types=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 embedding_hidden_dimension=5,
                 message_norm=None,
                 message_dropout=0,
                 nodewise_norm=None,
                 nodewise_dropout=0,
                 radial_embedding='bessel',
                 attention_heads=1,
                 seed=1,
                 ):
        super(MoleculeClassifier, self).__init__()

        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)

        self.atom_embedding = EmbeddingBlock(node_embedding_depth,
                                             num_embedding_types,
                                             input_node_depth,
                                             embedding_hidden_dimension)

        self.interaction_blocks = torch.nn.ModuleList([
            GCBlock(message_depth,
                    node_embedding_depth,
                    convolution_type,
                    num_radial,
                    norm=message_norm,
                    dropout=message_dropout,
                    heads=attention_heads,
                    )
            for _ in range(num_blocks)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            MLP(
                layers=nodewise_fc_layers,
                filters=node_embedding_depth,
                input_dim=node_embedding_depth,
                output_dim=node_embedding_depth,
                activation=activation,
                norm=nodewise_norm,
                dropout=nodewise_dropout,
            )
            for _ in range(num_blocks)
        ])

        self.global_pool = global_aggregation('molwise', graph_embedding_depth)

        self.gnn_mlp = MLP(layers=num_fcs,
                           filters=node_embedding_depth,
                           norm=fc_norm,
                           dropout=nodewise_dropout,
                           input_dim=graph_embedding_depth,
                           output_dim=output_dimension,
                           conditioning_dim=0,
                           seed=seed
                           )

    def get_geom_embedding(self, edge_index, pos):
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)

    def forward(self, data, return_latent=False, return_embedding=False):

        x = self.atom_embedding(data.x)  # embed atomic numbers & compute initial atom-wise feature vector
        batch = data.batch

        if data.periodic[0]:  # get radial embeddings periodically using minimum image convention
            edge_index, dist, rbf = self.periodize_box(data)

        else:  # just get the radial graph the normal way
            edges_dict = construct_radial_graph(data.pos, data.batch, data.ptr, self.cutoff, self.max_num_neighbors)
            edge_index = edges_dict['edge_index']
            dist, rbf = self.get_geom_embedding(edge_index, data.pos)

        for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
            x = convolution(x, rbf, edge_index, batch)  # graph convolution
            x = fc(x, batch=batch)  # feature-wise 1D convolution

        x = self.global_pool(x, batch, cluster=data.mol_ind, output_dim=data.num_graphs)

        if not return_embedding:
            return self.gnn_mlp(x, return_latent=return_latent)
        else:
            return self.gnn_mlp(x, return_latent=return_latent), x

        """
        >>> norm safety test
        self.gnn_mlp.eval()
        x0 = x.clone()
        y0 = self.gnn_mlp(x0, return_latent=return_latent)
        x1 = x0
        x1[0, :] += 10
        y1 = self.gnn_mlp(x1, return_latent=return_latent)
        print(torch.sum(torch.abs(y0[1:]-y1[1:])))
        import plotly.graph_objects as go
        fig = go.Figure(go.Heatmap(z=(y0-y1).abs().cpu().detach().numpy())).show()
        
        >>> norm inside FC safety
        self.eval()
        x0 = x.clone()
        x1 = x.clone()
        x1[0, :] += 10
        for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
            x0 = fc(x0, batch=batch)  # feature-wise 1D convolution, residual is already inside
        
        for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
            x1 = fc(x1, batch=batch)  # feature-wise 1D convolution, residual is already inside
        
        far_inds = torch.arange(len(x))[1:]
        print(torch.sum(torch.abs(x0[far_inds] - x1[far_inds])))
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Scatter(y=(x0[far_inds] - x1[far_inds]).abs().sum(1).cpu().detach().numpy())).show()

        """

    def periodize_box(self, data):
        assert data.num_graphs == 1  # this only works one at a time
        # restrict particles individually to box
        frac_coords = data.pos @ torch.linalg.inv(data.T_fc.T)
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
        distmats = fdistmats.permute((1, 2, 0)) @ data.T_fc.T
        norms = torch.linalg.norm(distmats, dim=-1)
        a, b = torch.where((norms > 0) * (norms <= self.cutoff))  # faster but still pretty slow
        edge_index = torch.cat((a[None, :], b[None, :]), dim=0)
        dist = norms[edge_index[0], edge_index[1]]
        rbf = self.rbf(dist)

        return edge_index, dist, rbf

    '''
    from torch_nl import compute_neighborlist
    import time
    pbc = torch.tensor([True, True, True]).to('cuda')
    cell = data.T_fc.T
    t0 = time.time()
    a, b = torch.where((norms > 0) * (norms <= self.cutoff))  # faster but still pretty slow
    edge_index = torch.cat((a[None, :], b[None, :]), dim=0)
    t1 = time.time()
    edge_index2, batch_mapping, shifts_idx = compute_neighborlist(
        self.cutoff, data.pos, cell, pbc, data.batch, False
    )
    t2 = time.time()
    print(t1-t0)
    print(t2-t1)
    '''
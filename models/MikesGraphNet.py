from math import sqrt, pi as PI
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_sparse import SparseTensor
import torch_geometric.nn as gnn
from models.asymmetric_radius_graph import asymmetric_radius_graph


class MikesGraphNet(torch.nn.Module):
    def __init__(self, hidden_channels: int,
                 graph_convolution_filters: int,
                 graph_convolution: str,
                 out_channels: int,
                 num_blocks: int,
                 num_spherical: int,
                 num_radial: int,
                 atom_embedding_dims,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 embedding_hidden_dimension=5,
                 norm=None,
                 dropout=0,
                 radial_embedding='bessel',
                 spherical_embedding=True,
                 num_atom_features=1,
                 attention_heads=1,
                 crystal_mode=False,
                 ):
        super(MikesGraphNet, self).__init__()

        self.num_blocks = num_blocks
        self.spherical_embedding = spherical_embedding
        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.crystal_mode = crystal_mode

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)

        if spherical_embedding:
            self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)

        self.atom_embeddings = EmbeddingBlock(hidden_channels, atom_embedding_dims, num_atom_features, embedding_hidden_dimension, activation)

        self.interaction_blocks = torch.nn.ModuleList([
            GCBlock(graph_convolution_filters,
                    hidden_channels,
                    radial_dim=num_radial,
                    spherical_dim=int(num_radial * num_spherical),
                    spherical=spherical_embedding,
                    convolution_mode=graph_convolution,
                    norm=norm,
                    dropout=dropout,
                    heads=attention_heads,
                    )
            for _ in range(num_blocks)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            FCBlock(hidden_channels, norm, dropout, activation)
            for _ in range(num_blocks)
        ])

        self.global_blocks = torch.nn.ModuleList([
            nn.Identity()
            for _ in range(num_blocks)
        ])
        # self.global_blocks = torch.nn.ModuleList([
        #     GlobalBlock(hidden_channels, graph_convolution_filters, norm, dropout, activation)
        #     for _ in range(num_blocks)
        # ])

        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, z, pos, batch=None, return_dists = False):
        """"""
        if self.crystal_mode: # allow incoming edges from outside the central crystal but exclude outgoing edges
            atom_inds = z[:,-1].clone() # pull the atom-wise index per-molecule
            z = z[:,:-1] # then delete it, since it's just an index, not used for modelling
            inside_inds = torch.where(atom_inds == 1)[0] # todo I think this might be slow - maybe feed it?
            edge_index = asymmetric_radius_graph(pos,pos[inside_inds],batch_x=batch, batch_y = batch[inside_inds],r=self.cutoff,
                                                 max_num_neighbors=self.max_num_neighbors, flow='source_to_target',inside_inds=inside_inds)
        else:
            edge_index = gnn.radius_graph(pos, r=self.cutoff, batch=batch,
                                          max_num_neighbors=self.max_num_neighbors, flow='source_to_target')

        if self.spherical_embedding:
            i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = \
                triplets(edge_index, num_nodes=z.size(0))
        else:
            i, j = edge_index

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        rbf = self.rbf(dist)

        # get spherical embedding
        if self.spherical_embedding:
            # Calculate angles.
            pos_i = pos[idx_i]
            pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
            angle = torch.atan2(b, a)

            sbf = self.sbf(dist, angle, idx_kj)

        # graph model starts here
        x = self.atom_embeddings(z)  # embed atomic numbers

        for n, (convolution, fc, global_agg) in enumerate(zip(self.interaction_blocks, self.fc_blocks, self.global_blocks)):
            if self.spherical_embedding:
                x = x + convolution(x, rbf, edge_index, sbf=sbf, idx_kj=idx_kj, idx_ji=idx_ji)  # graph convolution with angular features
            else:
                x = x + convolution(x, rbf, edge_index) # nodes up to max(j) are updated, others ignored


            x = x + fc(x)  # feature-wise 1D convolution

            #x = x + global_agg(x, batch)  # aggregate global information to all nodes # CURRENTLY IDENTITY - DEPRECATED,

            if self.crystal_mode: # copy the in-cell feature vectors to all corresponding outside atoms (relatively simple due to consistent structure of centralcell:supercells
                keep_cell_inds = torch.where(atom_inds == 1)[0] # find the indices for the reference cells
                n_repeats = len(x) // len(keep_cell_inds) # find the ratio of total atoms to within-ref-cell atoms # should always be exactly 27
                keep_cell_batch = batch[keep_cell_inds] # get the feature vectors we want to repeat
                # might be able to do this in one step with something like torch.interleave
                # else do it as a for loop if it's fast enough
                if n < (self.num_blocks - 1):
                    for i in range(batch[-1]):
                        unit_cell_inds = keep_cell_inds[keep_cell_batch == i]
                        x[unit_cell_inds[0]:unit_cell_inds[0] + len(unit_cell_inds) * n_repeats, :] = x[unit_cell_inds].repeat(n_repeats,1) # copy the first unit cell to all periodic images
                else: # on the final convolutional block, do not broadcast the reference cell
                    x = x[keep_cell_inds] # reduce the output to only the nodes of the reference cell - outer nodes are just copies anyway, so this gets the same information with less memory / compute overhead


        if return_dists:
            if self.crystal_mode:
                return self.output_layer(x), dist, keep_cell_inds
            else:
                return self.output_layer(x), dist
        else:
            if self.crystal_mode:
                return self.output_layer(x), keep_cell_inds
            else:
                return self.output_layer(x)


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()
        import sympy as sym
        from torch_geometric.nn.models.dimenet_utils import (bessel_basis,
                                                             real_sph_harm)

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        # self.freq = torch.arange(1,self.freq.numel() + 1).mul(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class GaussianEmbedding(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianEmbedding, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_atom_features, embedding_dimension, activation='gelu'):
        super(EmbeddingBlock, self).__init__()
        self.num_embeddings = 1
        for key in embedding_size.keys():
            self.embeddings = nn.Embedding(int(embedding_size[key]), embedding_dimension)
        self.linear = nn.Linear(embedding_dimension + num_atom_features - self.num_embeddings, hidden_channels)
        self.activation = Activation(activation, hidden_channels)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        embedding = self.embeddings(x[:, 0].long())
        concat_vec = torch.cat([embedding, x[:, self.num_embeddings:]], dim=-1)

        return self.activation(self.linear(concat_vec))


class GCBlock(torch.nn.Module):
    def __init__(self, graph_convolution_filters, hidden_channels, radial_dim, convolution_mode, spherical_dim=None, spherical=False, norm=None, dropout=0, heads=1):
        super(GCBlock, self).__init__()
        self.norm = Normalization(norm, graph_convolution_filters)
        self.node_to_message = nn.Linear(hidden_channels, graph_convolution_filters)
        self.message_to_node = nn.Linear(graph_convolution_filters, hidden_channels,bias=False) # don't want to send spurious messages, though it probably doesn't matter anyway
        self.radial_to_message = nn.Linear(radial_dim, graph_convolution_filters)

        if spherical:  # need more linear layers to aggregate angular information to radial
            assert spherical_dim is not None, "Spherical information must have a dimension != 0 for spherical message aggregation"
            self.spherical_to_message = nn.Linear(spherical_dim, graph_convolution_filters)
            self.radial_spherical_aggregation = torch.add  # could also do dot

        if convolution_mode == 'self attention':
            self.GConv = gnn.GATv2Conv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=graph_convolution_filters,
            )
        elif convolution_mode == 'full message passing':
            self.GConv = MPConv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters,
                edge_dim=graph_convolution_filters,
            )

    def forward(self, x, rbf, edge_index, sbf=None, idx_kj=None, idx_ji=None):
        # convert local information into edge weights
        if sbf is None:  # no angular information
            edge_attr = self.radial_to_message(rbf)
        else:
            # aggregate spherical messages to radial
            # rbf = self.radial_to_message(rbf)
            # sbf = self.spherical_message(sbf)
            edge_attr = self.radial_spherical_aggregation(self.radial_to_message(rbf)[idx_kj], self.spherical_to_message(sbf))  # combine radial and spherical info in triplet space
            edge_attr = scatter(edge_attr, idx_ji, dim=0)  # collect triplets back down to pair space

        # convolve
        x = self.norm(self.node_to_message(x))
        x = self.GConv(x, edge_index, edge_attr)

        return self.message_to_node(x)


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(MPConv, self).__init__()
        self.linear1 = nn.Linear(in_channels * 2 + edge_dim, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index
        m = self.linear2(F.gelu(self.norm(self.linear1(torch.cat((x[i], x[j], edge_attr), dim=-1)))))
        #m = self.linear1(torch.cat((x[i], x[j], edge_attr), dim=-1))

        return scatter(m, j, dim=0, dim_size=len(x)) # send directional messages from i to j, enforcing the size of the output dimension


class FCBlock(torch.nn.Module):
    '''
    fully-connected block, following the original transformer architecture with norm first
    '''

    def __init__(self, hidden_channels, norm, dropout, activation):
        super(FCBlock, self).__init__()
        self.norm = Normalization(norm, hidden_channels)
        self.activation = Activation(activation, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))


class GlobalBlock(torch.nn.Module):
    '''
    fully-connected block, following the original transformer architecture with norm first
    '''

    def __init__(self, hidden_channels, graph_convolution_filters, norm, dropout, activation):
        super(GlobalBlock, self).__init__()
        self.global_pool = gnn.GlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(graph_convolution_filters, graph_convolution_filters), nn.GELU(), nn.Linear(graph_convolution_filters, 1)),
        )
        self.norm = Normalization(norm, hidden_channels)
        self.activation = Activation(activation, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(hidden_channels, graph_convolution_filters)
        self.linear2 = nn.Linear(graph_convolution_filters + graph_convolution_filters, hidden_channels)

    def forward(self, x, batch):
        x = self.dropout(self.activation(self.linear1(self.norm(x))))  # scale to graph conv dimension
        global_embedding = self.global_pool(x, batch)  # attention aggregation

        return self.linear2(torch.cat((x, global_embedding[batch]), dim=-1))  # expand global to local basis, concatenate and forward


class kernelActivation(nn.Module):  # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis))  # positive and negative values for Dirichlet Kernel
        gamma = 1 / (6 * (self.dict[-1] - self.dict[-2]) ** 2)  # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma', torch.ones(1) * gamma)  #

        # self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1, 1), groups=int(channels), bias=False)

        # nn.init.normal(self.linear.weight.data, std=0.1)

    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x) == 2:
            x = x.reshape(2, self.channels, 1)

        return torch.exp(-self.gamma * (x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1)  # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1)  # apply linear coefficients and sum

        return x


def triplets(edge_index, num_nodes):
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'gelu':
            self.activation = F.gelu
        elif activation_func == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)

    def forward(self, input):
        return self.activation(input)


class Normalization(nn.Module):
    def __init__(self, norm, filters, *args, **kwargs):
        super().__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(filters)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(filters)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            print(norm + " is not a valid normalization")
            sys.exit()

    def forward(self, input):
        return self.norm(input)

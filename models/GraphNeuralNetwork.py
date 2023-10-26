from math import pi as PI

from models.basis_functions import GaussianEmbedding, BesselBasisLayer
from models.components import Normalization, Activation
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
import torch_geometric.nn as gnn
from models.components import MLP


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self,
                 input_node_depth: int,
                 node_embedding_depth: int,
                 nodewise_fc_layers: int,
                 message_depth: int,
                 convolution_type: str,
                 graph_embedding_depth: int,
                 num_blocks: int,
                 num_radial: int,
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
                 periodize_inside_nodes=False,
                 outside_convolution_type='none',
                 ):
        super(GraphNeuralNetwork, self).__init__()

        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.periodize_inside_nodes = periodize_inside_nodes
        self.outside_convolution_type = outside_convolution_type

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)

        self.atom_embedding = EmbeddingBlock(node_embedding_depth, num_embedding_types, input_node_depth, embedding_hidden_dimension)

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

        if node_embedding_depth != graph_embedding_depth:
            self.output_layer = nn.Linear(node_embedding_depth, graph_embedding_depth)
        else:
            self.output_layer = nn.Identity()

    def get_geom_embedding(self, edge_index, pos):
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)

    def forward(self, z, pos, batch, ptr, edges_dict: dict):
        """
        # todo write docstring
        """
        # graph model starts here

        x = self.atom_embedding(z)  # embed atomic numbers & compute initial atom-wise feature vector #

        if self.outside_convolution_type == 'none':  # assumes input with inside-outside structure, and enforces periodicity after each convolution
            edge_index = edges_dict['edge_index']
            dist, rbf = self.get_geom_embedding(edge_index, pos)
            assert not self.periodize_inside_nodes, "Cannot periodize to outside nodes if there are no outside nodes"
        elif self.outside_convolution_type == 'all_layers':
            edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
            edge_index = torch.cat((edge_index, edge_index_inter), dim=1)  # all edges counted in one big batch
            dist, rbf = self.get_geom_embedding(edge_index, pos)
        elif self.outside_convolution_type == 'last_layer':
            edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
            dist, rbf = self.get_geom_embedding(edge_index, pos)
            dist_inter, rbf_inter = self.get_geom_embedding(torch.cat((edge_index, edge_index_inter), dim=1), pos)
        else:
            assert False, "Must select a valid treatment of inside vs outside nodes"

        for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
            if n == (len(self.interaction_blocks) - 1) and self.outside_convolution_type == 'last_layer':
                x = convolution(x, rbf_inter, torch.cat((edge_index, edge_index_inter), dim=1), batch)  # return only the results of the intermolecular convolution, omitting intramolecular features
            else:
                x = convolution(x, rbf, edge_index, batch)  # graph convolution - residual is already inside the conv operator

            if not self.periodize_inside_nodes:
                x = fc(x, batch=batch)  # feature-wise 1D convolution, residual is already inside
            else:
                x[inside_inds] = x[inside_inds] + fc(x[inside_inds], batch=batch[inside_inds])  # update only the inside inds

                for ii in range(len(ptr) - 1):  # enforce periodicity for each crystal, assuming invariant node features
                    x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)  # copy the first unit cell to all periodic images

                if n == len(self.interaction_blocks) - 1:
                    x = x[inside_inds]  # reduce to inside image

        return self.output_layer(x)


'''
import networkx as nx
import matplotlib.pyplot as plt
intra_edges = (edge_index[:, edge_index[0, :] < ptr[1]].cpu().detach().numpy().T)
inter_edges = (edge_index_inter[:, edge_index_inter[0, :] < ptr[1]].cpu().detach().numpy().T)
plt.clf()
G = nx.Graph()
G = G.to_directed()
G.add_weighted_edges_from(np.concatenate((intra_edges, np.ones(len(intra_edges))[:, None] * 2), axis=1))
G.add_weighted_edges_from(np.concatenate((inter_edges, np.ones(len(inter_edges))[:, None] * 0.25), axis=1))
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
node_weights = np.concatenate((np.ones(9)*2, np.ones(len(G.nodes)-9)))
nx.draw_kamada_kawai(G, arrows=True, node_size=node_weights * 100, edge_color=weights, linewidths = 1, width=weights, 
edge_cmap=plt.cm.RdYlGn, node_color = node_weights, cmap=plt.cm.RdYlGn)
'''


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_atom_types, num_atom_features, atom_type_embedding_dimension):
        super(EmbeddingBlock, self).__init__()
        self.embeddings = nn.Embedding(num_atom_types + 1, atom_type_embedding_dimension)
        self.linear = nn.Linear(atom_type_embedding_dimension + num_atom_features - 1, hidden_channels)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        embedding = self.embeddings(x[:, 0].long())  # always embed the first dimension only (by convention, atomic number)
        concat_vec = torch.cat([embedding, x[:, 1:]], dim=-1)

        return self.linear(concat_vec)


class GCBlock(torch.nn.Module):
    def __init__(self,
                 message_depth,
                 node_embedding_depth,
                 convolution_type,
                 radial_dim,
                 norm=None,
                 dropout=0,
                 heads=1):
        super(GCBlock, self).__init__()
        if norm == 'graph':
            message_norm = 'layer'
        else:
            message_norm = norm

        self.norm1 = Normalization(message_norm, message_depth)
        self.norm2 = Normalization(message_norm, message_depth)
        self.node_to_message = nn.Linear(node_embedding_depth, message_depth, bias=False)
        self.message_to_node = nn.Linear(message_depth, node_embedding_depth, bias=False)  # don't want to send spurious messages, though it probably doesn't matter anyway
        self.radial_to_message = nn.Linear(radial_dim, message_depth, bias=False)

        if convolution_type == 'GATv2':
            self.GConv = gnn.GATv2Conv(
                in_channels=message_depth,
                out_channels=message_depth // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=message_depth,
            )
        elif convolution_type == 'TransformerConv':
            self.GConv = gnn.TransformerConv(
                in_channels=message_depth,
                out_channels=message_depth // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=message_depth,
                beta=True,
            )

    def compute_edge_attributes(self, rbf):
        edge_attr = self.radial_to_message(rbf)

        return edge_attr

    def forward(self, x, rbf, edge_index, batch):
        # generate messages
        x = self.norm1(self.node_to_message(x), batch)

        # generate edge embeddings
        edge_attr = self.norm2(self.compute_edge_attributes(rbf), batch[edge_index[0]])

        # convolve
        x = self.GConv(x, edge_index, edge_attr)

        # reshape to node-wise
        return self.message_to_node(x)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(gnn.MessagePassing):
    '''
    ~~the graph convolution used in the popular SchNet~~
    '''

    def __init__(self, in_channels, out_channels, num_filters, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)  # de-weight distant nodes
        # W = self.nn(edge_attr) * C.view(-1, 1)
        W = edge_attr * C.view(-1, 1)  # in my method, edge_attr are pre-featurized

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)

        return x

    def message(self, x_j, W):
        return x_j * W


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, dropout=0, norm=None, activation='leaky relu'):
        super(MPConv, self).__init__()

        self.MLP = MLP(layers=4,
                       filters=out_channels,
                       input_dim=in_channels * 2 + edge_dim,
                       dropout=dropout,
                       norm=norm,
                       output_dim=out_channels,
                       activation=activation,
                       )

    def forward(self, x, edge_index, edge_attr):
        m = self.MLP(torch.cat((x[edge_index[0]], x[edge_index[1]], edge_attr), dim=-1))

        return scatter(m, edge_index[1], dim=0, dim_size=len(x))  # send directional messages from i to j, enforcing the size of the output dimension


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
        x = self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))
        return x


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

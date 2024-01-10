import torch
from torch import nn as nn
from torch_geometric import nn as gnn
from torch_scatter import scatter

from models.components import Normalization, MLP, EMLP
import e3nn.o3 as o3


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, node_embedding_depth, num_input_classes, num_scalar_input_features, type_embedding_dimension):
        super(EmbeddingBlock, self).__init__()

        self.embeddings = nn.Embedding(num_input_classes + 1, type_embedding_dimension)
        self.linear = nn.Linear(type_embedding_dimension + num_scalar_input_features - 1, node_embedding_depth)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        embedding = self.embeddings(x[:, 0].long())  # always embed the first dimension only (by convention, atomic number)

        concat_vec = torch.cat([embedding, x[:, 1:]], dim=-1)

        return self.linear(concat_vec)


class EquivariantEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, num_classes, embedding_dimension):
        super(EquivariantEmbeddingBlock, self).__init__()

        self.embeddings = nn.Embedding(num_classes + 1, embedding_dimension)
        self.linear = o3.Linear(irreps_in, irreps_out)

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
                 radial_dim,
                 norm=None,
                 dropout=0,
                 heads=1,
                 equivariant=False,
                 irreps=None):
        super(GCBlock, self).__init__()
        if norm == 'graph':
            message_norm = 'layer'
        else:
            message_norm = norm
        # todo add equivariant norming
        self.message_norm = Normalization(message_norm, message_depth) if equivariant is False else None
        self.edge_attribute_norm = Normalization(message_norm, message_depth) if equivariant is False else None

        self.embed_edge = nn.Linear(radial_dim, message_depth, bias=False)
        self.equivariant = equivariant
        if equivariant:
            self.node_to_message = o3.Linear(irreps, irreps)
            self.message_to_node = o3.Linear(irreps, irreps)
            self.GConv = EquivariantMessagePassing(irreps, message_depth)
        else:
            self.node_to_message = nn.Linear(node_embedding_depth, message_depth, bias=False)
            self.message_to_node = nn.Linear(message_depth, node_embedding_depth, bias=False)  # don't want to send spurious messages, though it probably doesn't matter anyway
            assert message_depth % heads == 0
            self.GConv = gnn.TransformerConv(
                in_channels=message_depth,
                out_channels=message_depth // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=message_depth,
                beta=True,
            )

    def embed_edge_attrs(self, edge_attr):
        return self.embed_edge(edge_attr)

    def forward(self, x, edge_attr, edge_index, batch):
        # generate messages
        x = self.node_to_message(x)
        if self.message_norm is not None:  # todo make new identity class which can accept extra args
            x = self.message_norm(x, batch)

        # generate edge embeddings
        edge_emb = self.embed_edge_attrs(edge_attr)

        if self.edge_attribute_norm is not None:
            self.edge_attribute_norm(edge_emb, batch[edge_index[0]])  # todo confirm that this makes sense like at all

        # convolve
        x = self.GConv(x, edge_index, edge_emb)  # todo confirm correct indexing with transformerconv

        # reshape to node dimension
        return self.message_to_node(x)


'''equivariance testing

from e3nn.util.test import equivariance_error

self.irreps_node = o3.Irreps('129x0e + 128x1o')

err1 = equivariance_error(self.node_to_message,
                         args_in=[x],
                         irreps_in=self.irreps_node,
                         irreps_out=self.irreps_node,
                         ntrials=5
                         )

'''


class EquivariantMessagePassing(nn.Module):
    def __init__(self, irreps, num_edge_attr):
        super(EquivariantMessagePassing, self).__init__()
        self.convolution = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps,
            irreps_in2=o3.Irreps(f'{num_edge_attr}x0e'),
            irreps_out=irreps,
            internal_weights=True,
        )  # todo add support for nonscalar edge attributes

    def forward(self, x, edge_index, edge_attrs):
        messages = self.convolution(x[edge_index[0]], edge_attrs)

        return scatter(messages, edge_index[1], dim=0, dim_size=len(x), reduce='mean')  # simple mean aggregation - radial attention may be possible


class MPConv(torch.nn.Module):  # todo refactor and add a transformer version
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


class FC_Block(torch.nn.Module):
    def __init__(self,
                 nodewise_fc_layers,
                 node_embedding_depth,
                 activation,
                 nodewise_norm,
                 nodewise_dropout,
                 equivariant=False,
                 irreps=None):
        super(FC_Block, self).__init__()

        if equivariant:
            self.model = EMLP(
                layers=nodewise_fc_layers,
                irreps_in=irreps,
                irreps_out=irreps,
                irreps_hidden=irreps,
                norm=nodewise_norm,
                activation=activation,
            )
        else:
            self.model = MLP(layers=nodewise_fc_layers,
                             filters=node_embedding_depth,
                             input_dim=node_embedding_depth,
                             output_dim=node_embedding_depth,
                             activation=activation,
                             norm=nodewise_norm,
                             dropout=nodewise_dropout)

    def forward(self, x, conditions=None, return_latent=False, batch=None):
        return self.model(x, conditions=conditions, return_latent=return_latent, batch=batch)

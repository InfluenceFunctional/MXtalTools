import torch
from torch import nn as nn
from torch_geometric import nn as gnn
from torch_scatter import scatter

from mxtaltools.models.components import MLP
from mxtaltools.models.equivariant_TransformerConv import EquiVTransformerConv


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, init_node_embedding_dim, num_input_classes, num_scalar_input_features, atom_type_embedding_dim):
        super(EmbeddingBlock, self).__init__()

        self.embeddings = nn.Embedding(num_input_classes + 1, atom_type_embedding_dim)
        self.linear = nn.Linear(atom_type_embedding_dim + num_scalar_input_features - 1, init_node_embedding_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        embedding = self.embeddings(
            x[:, 0].long())  # always embed the first dimension only (by convention, atomic number)

        concat_vec = torch.cat([embedding, x[:, 1:]], dim=-1)

        return self.linear(concat_vec)


class GCBlock(torch.nn.Module):
    def __init__(self,
                 message_depth,
                 node_embedding_depth,
                 radial_dim,
                 dropout=0,
                 heads=1,
                 equivariant=False):
        super(GCBlock, self).__init__()
        self.embed_edge = nn.Linear(radial_dim, radial_dim, bias=False)
        self.equivariant = equivariant
        if equivariant:
            self.V_GConv = EquiVTransformerConv(
                in_channels=message_depth,
                out_channels=message_depth // heads,
                heads=heads,
                edge_dim=radial_dim,
            )
            self.vec_to_message = nn.Linear(node_embedding_depth, message_depth, bias=False)
            self.message_to_vec = nn.Linear(message_depth, node_embedding_depth, bias=False)

        self.node_to_message = nn.Linear(node_embedding_depth, message_depth, bias=False)
        self.message_to_node = nn.Linear(message_depth, node_embedding_depth, bias=False)

        assert message_depth % heads == 0
        self.GConv = gnn.TransformerConv(
            in_channels=message_depth,
            out_channels=message_depth // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=radial_dim,
            beta=True,
        )

    def embed_edge_attrs(self, edge_attr):
        return self.embed_edge(edge_attr)

    def forward(self, x, v, edge_attr, edge_index):
        # convolve
        edge_embedding = self.embed_edge_attrs(edge_attr)
        x, (_, alpha) = self.GConv(
            self.node_to_message(x), edge_index, edge_embedding,
            return_attention_weights=True)

        if self.equivariant:
            v = self.V_GConv(self.vec_to_message(v), alpha, edge_index, edge_embedding)

        if v is not None:
            return self.message_to_node(x), self.message_to_vec(v)
        else:
            # reshape to node dimension
            return self.message_to_node(x)


class FCBlock(torch.nn.Module):
    def __init__(self,
                 nodewise_fc_layers,
                 node_embedding_depth,
                 activation,
                 nodewise_norm,
                 nodewise_dropout,
                 equivariant=False,
                 vector_norm=None,
                 ):
        super(FCBlock, self).__init__()
        self.equivariant = equivariant

        self.model = MLP(layers=nodewise_fc_layers,
                         filters=node_embedding_depth,
                         input_dim=node_embedding_depth,
                         output_dim=node_embedding_depth,
                         conditioning_dim=node_embedding_depth if equivariant else 0,
                         activation=activation,
                         norm=nodewise_norm,
                         dropout=nodewise_dropout,
                         equivariant=equivariant,
                         vector_norm=vector_norm)

    def forward(self, x, v=None, return_latent=False, batch=None):
        return self.model(x,
                          v=v,
                          conditions=torch.linalg.norm(v, dim=1) if v is not None else None,
                          return_latent=return_latent,
                          batch=batch)


class OutputBlock(torch.nn.Module):
    def __init__(self, node_dim, embedding_dim, equivariant_graph):

        super().__init__()
        self.equivariant_graph = equivariant_graph

        if self.equivariant_graph:
            if node_dim != embedding_dim:
                self.v_output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
            else:
                self.v_output_layer = nn.Identity()

        if node_dim != embedding_dim:
            self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def forward(self, x, v):
        if v is not None:
            return self.output_layer(x), self.v_output_layer(v)
        else:
            return self.output_layer(x)

from typing import Optional

import torch
from torch import Tensor
from torch import nn as nn
from torch_geometric import nn as gnn

from mxtaltools.models.modules.components import EMLP
from mxtaltools.models.modules.equivariant_TransformerConv import EquiVTransformerConv


class GCBlock(torch.nn.Module):
    def __init__(self,
                 message_depth,
                 node_embedding_depth,
                 radial_dim,
                 dropout=0,
                 heads=1,
                 add_vector_channel=False):
        super(GCBlock, self).__init__()
        self.embed_edge = nn.Linear(radial_dim, radial_dim, bias=False)
        self.equivariant = add_vector_channel
        if add_vector_channel:
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
        """
        Embed edge attributes

        Parameters
        ----------
        edge_attr : torch.tensor[n_edges, n_features]
            1d attributes for each edge, typically an embedding over radial basis functions

        Returns
        -------
        torch.tensor[n_edges, message_depth]
        """
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

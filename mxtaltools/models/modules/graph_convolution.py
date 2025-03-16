import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (
    Adj,
    OptTensor,
)

from mxtaltools.models.modules.augmented_softmax_aggregator import AugSoftmaxAggregation, VectorAugSoftmaxAggregation
from mxtaltools.models.modules.components import Normalization, Activation


class MConv(MessagePassing):
    """
    Message passing layer with optional vector channel.
    Aggregation done via softmax operator.
    Message embedding via linear operator.
    """

    def __init__(
            self,
            message_dim,
            node_dim,
            edge_embedding_dim,
            norm=None,
            activation_fn='gelu',
    ):
        super().__init__(aggr=AugSoftmaxAggregation(temperature=1,
                                                    learn=True,
                                                    bias=0.1,
                                                    channels=message_dim))

        self.in_channels = node_dim
        self.out_channels = node_dim
        self.edge_dim = edge_embedding_dim
        self.message_dim = message_dim

        '''initialize scalar transforms'''
        self.edge2message = nn.Linear(edge_embedding_dim, message_dim, bias=False)
        self.source_node2message = nn.Linear(node_dim, message_dim, bias=False)
        self.tgt_node2message = nn.Linear(node_dim, message_dim, bias=False)
        self.generate_message = nn.Linear(int(3 * message_dim), message_dim, bias=False)

        self.norm = Normalization(norm, message_dim)
        self.activation = Activation(activation_fn, message_dim)
        self.message2node = nn.Linear(message_dim, node_dim, bias=False)

        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            edge_attr: Tensor,
    ) -> Tensor:
        r"""
        Runs the forward pass of the module.
        """

        out = self.propagate(edge_index=edge_index,
                             x=x,
                             edge_attr=edge_attr,
                             num_nodes=x.size(0))

        return x + self.message2node(out)

    def message(self,
                x_i: Tensor,
                x_j: Tensor,
                edge_attr: Tensor) -> Tensor:
        edge_attr = self.edge2message(edge_attr)
        msg_i = self.source_node2message(x_i)
        msg_j = self.tgt_node2message(x_j)
        return self.activation(
            self.norm(
                self.generate_message(
                    torch.cat([msg_i, msg_j, edge_attr], dim=-1))))


class v_MConv(MessagePassing):
    """
    Message passing layer with optional vector channel.
    Aggregation done via softmax operator.
    Message embedding via linear operator.
    """

    def __init__(
            self,
            message_depth,
            node_depth,
            edge_embedding_dim,
            norm=None,
    ):
        super().__init__(aggr=VectorAugSoftmaxAggregation(temperature=1,
                                                          learn=True,
                                                          bias=0.1,
                                                          channels=message_depth),
                         node_dim=0)

        self.in_channels = node_depth
        self.out_channels = node_depth
        self.edge_dim = edge_embedding_dim
        self.message_dim = message_depth

        '''initialize scalar transforms'''
        self.edge2message = nn.Linear(edge_embedding_dim, message_depth, bias=False)
        self.source_node2message = nn.Linear(node_depth, message_depth, bias=False)
        self.tgt_node2message = nn.Linear(node_depth, message_depth, bias=False)
        self.norm = Normalization(norm, message_depth)
        self.update2node = nn.Linear(message_depth, node_depth, bias=False)

        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            edge_attr: Tensor,
    ) -> Tensor:
        r"""
        Runs the forward pass of the module.
        """

        out = self.propagate(edge_index=edge_index,
                             x=x,
                             edge_attr=edge_attr,
                             num_nodes=x.size(0))

        return x + self.update2node(out)

    def message(self,
                x_i: Tensor,
                x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:
        edge_attr = self.edge2message(edge_attr)
        msg_i = self.source_node2message(x_i)
        msg_j = self.tgt_node2message(x_j)

        out = (msg_i + msg_j) * edge_attr[:, None, :]  # switch to gating - addition is not allowed

        return self.norm(out)

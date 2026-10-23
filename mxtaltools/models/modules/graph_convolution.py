"""Residual message-passing layers for scalar node features (:class:`MConv`) and vector
node features (:class:`v_MConv`), aggregating messages with the augmented softmax
aggregators."""

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
    """Residual message-passing layer on scalar node features.

    Message embedding via linear operators: each edge message is built from bias-free
    linear embeddings of the two endpoint node features and of the edge features,
    concatenated, mixed by a bias-free linear layer, then normalized and activated.
    Aggregation is done via a learnable :class:`AugSoftmaxAggregation`
    (``channels=message_dim``, initial temperature 1, initial bias 0.1) at
    ``edge_index[1]``; the aggregate is projected back to ``node_dim`` and added to the
    input features.

    Parameters
    ----------
    message_dim : int
        Width of the messages.
    node_dim : int
        Width of the node features (input and output).
    edge_embedding_dim : int
        Width of the edge features.
    norm : str or None, optional
        Message normalization name, passed to :class:`Normalization`. Default None
        (identity).
    activation_fn : str, optional
        Message activation name, passed to :class:`Activation`. Default ``'gelu'``.
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
        r"""Run one round of message passing with a residual update.

        Parameters
        ----------
        x : Tensor
            Node features, [num_nodes, node_dim].
        edge_index : Adj
            Edge indices, [2, num_edges]; messages are aggregated at
            ``edge_index[1]``.
        edge_attr : Tensor
            Edge features, [num_edges, edge_embedding_dim].

        Returns
        -------
        Tensor
            ``x + message2node(aggregated messages)``, [num_nodes, node_dim]. Nodes
            with no incoming edges get a zero aggregate and are returned unchanged.
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
        """Build one message per edge.

        ``x_i`` is embedded by ``source_node2message`` and ``x_j`` by
        ``tgt_node2message``.

        Parameters
        ----------
        x_i : Tensor
            Features of the node at ``edge_index[1]`` of each edge,
            [num_edges, node_dim].
        x_j : Tensor
            Features of the node at ``edge_index[0]`` of each edge,
            [num_edges, node_dim].
        edge_attr : Tensor
            Edge features, [num_edges, edge_embedding_dim].

        Returns
        -------
        Tensor
            ``activation(norm(generate_message(cat[source_node2message(x_i),
            tgt_node2message(x_j), edge2message(edge_attr)])))``,
            [num_edges, message_dim]. The normalization is called without a batch
            vector.
        """
        edge_attr = self.edge2message(edge_attr)
        msg_i = self.source_node2message(x_i)
        msg_j = self.tgt_node2message(x_j)
        return self.activation(
            self.norm(
                self.generate_message(
                    torch.cat([msg_i, msg_j, edge_attr], dim=-1))))


class v_MConv(MessagePassing):
    """Residual message-passing layer on vector node features.

    Node features are vectors, [num_nodes, 3, node_depth]. Message embedding via
    linear operators: each edge message is
    ``norm((source_node2message(x_i) + tgt_node2message(x_j)) * edge2message(edge_attr))``,
    i.e. bias-free linear maps over the channel dimension of the two endpoint vectors,
    summed, then scaled per channel by the embedded scalar edge features. No
    activation is applied. Aggregation is done along dimension 0 via a learnable
    :class:`VectorAugSoftmaxAggregation` (``channels=message_depth``, initial
    temperature 1, initial bias 0.1) at ``edge_index[1]``; the aggregate is projected
    back to ``node_depth`` and added to the input vectors.

    Parameters
    ----------
    message_depth : int
        Number of vector channels in the messages. A value of 1 makes the aggregator
        raise IndexError in :meth:`forward` (see :class:`VectorAugSoftmaxAggregation`).
    node_depth : int
        Number of vector channels of the node features (input and output).
    edge_embedding_dim : int
        Width of the scalar edge features.
    norm : str or None, optional
        Message normalization name, passed to :class:`Normalization`. Default None
        (identity).
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
        r"""Run one round of vector message passing with a residual update.

        Parameters
        ----------
        x : Tensor
            Node vectors, [num_nodes, 3, node_depth].
        edge_index : Adj
            Edge indices, [2, num_edges]; messages are aggregated at
            ``edge_index[1]``.
        edge_attr : Tensor
            Scalar edge features, [num_edges, edge_embedding_dim].

        Returns
        -------
        Tensor
            ``x + update2node(aggregated messages)``, [num_nodes, 3, node_depth].
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
        """Build one vector message per edge.

        Parameters
        ----------
        x_i : Tensor
            Vectors of the node at ``edge_index[1]`` of each edge,
            [num_edges, 3, node_depth].
        x_j : Tensor
            Vectors of the node at ``edge_index[0]`` of each edge,
            [num_edges, 3, node_depth].
        edge_attr : OptTensor
            Edge features, [num_edges, edge_embedding_dim]. Required: it is passed to
            ``edge2message`` unconditionally.

        Returns
        -------
        Tensor
            The sum ``source_node2message(x_i) + tgt_node2message(x_j)`` multiplied by
            ``edge2message(edge_attr)[:, None, :]``, then passed through ``norm``,
            [num_edges, 3, message_depth]. The normalization is called without a
            batch vector.
        """
        edge_attr = self.edge2message(edge_attr)
        msg_i = self.source_node2message(x_i)
        msg_j = self.tgt_node2message(x_j)

        out = (msg_i + msg_j) * edge_attr[:, None, :]  # switch to gating - addition is not allowed

        return self.norm(out)

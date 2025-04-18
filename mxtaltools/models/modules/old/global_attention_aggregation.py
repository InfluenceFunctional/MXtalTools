from typing import Optional

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax


class AttentionalAggregation_w_alpha(Aggregation):
    r"""The soft attention aggregation layer from the `"Graph Matching Networks
    for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \cdot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPs.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]` (for
            node-level gating) or :obj:`[1, out_channels]` (for feature-level
            gating), *e.g.*, defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn: torch.nn.Module,
                 nn: Optional[torch.nn.Module] = None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2, return_alpha=False) -> (Tensor, Tensor):

        self.assert_two_dimensional_input(x, dim)

        if isinstance(self.gate_nn, torch_geometric.nn.MLP):
            gate = self.gate_nn(x, index, dim_size)
        else:
            gate = self.gate_nn(x)

        if isinstance(self.nn, torch_geometric.nn.MLP):
            x = self.nn(x, index, dim_size)
        elif self.nn is not None:
            x = self.nn(x)

        gate = softmax(gate, index, ptr, dim_size, dim)
        if return_alpha:
            return self.reduce(gate * x, index, ptr, dim_size, dim), gate
        else:
            return self.reduce(gate * x, index, ptr, dim_size, dim)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'nn={self.nn})')

import typing
from typing import Optional, Tuple, Union
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
)

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


class EquiVTransformerConv(MessagePassing):
    r"""
    Modified TransformerConv with pre-computed attention weights
    """
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            edge_dim: Optional[int] = None,

            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_value = Linear(in_channels[0], heads * out_channels, bias=False)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        self.lin_skip = Linear(in_channels[1], heads * out_channels,
                               bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()

    def forward(  # noqa: F811
            self,
            x: Union[Tensor, PairTensor],
            alpha: Tensor,
            edge_index: Adj,
            edge_attr: OptTensor = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        H, C = self.heads, self.out_channels

        value = self.lin_value(x).view(len(x), 3, H, C)

        out = self.propagate(edge_index, alpha=alpha, value=value,
                             edge_attr=edge_attr)

        out = out.view(len(out), 3, self.heads * self.out_channels)

        x_r = self.lin_skip(x)
        out = out + x_r  # vector pointwise multiplication is not allowed

        return out

    def message(self, alpha: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)

        self._alpha = alpha

        out = value_j
        if edge_attr is not None:
            out = out * edge_attr[:, None, :, :]  # switch to gating - addition is not allowed

        out = out * alpha[:, None, :, None]
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


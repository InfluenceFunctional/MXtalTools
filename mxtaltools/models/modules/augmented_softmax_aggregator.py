from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax


class AugSoftmaxAggregation(Aggregation):
    r"""The softmax aggregation operator based on a temperature term, as
    described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper.

    Modified with learnable bias term
    """

    def __init__(self,
                 temperature: float = 1.0,
                 learn: bool = True,
                 semi_grad: bool = False,
                 channels: int = 1,
                 bias: float = 0.1):
        super().__init__()

        if learn and semi_grad:
            raise ValueError(
                f"Cannot enable 'semi_grad' in '{self.__class__.__name__}' in "
                f"case the temperature term 't' is learnable")

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_termperature = temperature
        self._init_bias = bias
        self.learn = learn
        self.semi_grad = semi_grad
        self.channels = channels

        self.t = Parameter(torch.empty(channels)) if learn else temperature
        self.b = Parameter(torch.empty(channels)) if learn else bias
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.t, Tensor):
            self.t.data.fill_(self._init_termperature)
        if isinstance(self.b, Tensor):
            self.b.data.fill_(self._init_bias)

    def forward(self, x: Tensor,
                index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        t = self.t
        b = self.b
        if self.channels != 1:
            self.assert_two_dimensional_input(x, dim)
            assert isinstance(t, Tensor)
            t = t.view(-1, self.channels)
            b = b.view(-1, self.channels)

        alpha = x
        if not isinstance(t, (int, float)) or t != 1:
            alpha = x * t

        if not self.learn and self.semi_grad:
            with torch.no_grad():
                alpha = softmax(alpha, index, ptr, dim_size, dim)
        else:
            alpha = softmax(alpha, index, ptr, dim_size, dim)
        return self.reduce(x * (alpha + b), index, ptr, dim_size, dim, reduce='sum')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')


class VectorAugSoftmaxAggregation(Aggregation):
    """
    adjusted to weigh by vector length rather than raw value
    """

    def __init__(self,
                 temperature: float = 1.0,
                 learn: bool = True,
                 semi_grad: bool = False,
                 channels: int = 1,
                 bias: float = 0.1):
        super().__init__()

        if learn and semi_grad:
            raise ValueError(
                f"Cannot enable 'semi_grad' in '{self.__class__.__name__}' in "
                f"case the temperature term 't' is learnable")

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_termperature = temperature
        self._init_bias = bias
        self.learn = learn
        self.semi_grad = semi_grad
        self.channels = channels

        self.t = Parameter(torch.empty(channels)) if learn else temperature
        self.b = Parameter(torch.empty(channels)) if learn else bias
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.t, Tensor):
            self.t.data.fill_(self._init_termperature)
        if isinstance(self.b, Tensor):
            self.b.data.fill_(self._init_bias)

    def forward(self, x: Tensor,
                index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None,
                dim: int = 0,
                cart_dim: int = 1) -> Tensor:

        t = self.t
        b = self.b
        if self.channels != 1:
            t = t.view(-1, self.channels)
            b = b.view(-1, self.channels)

        alpha = x
        if not isinstance(t, (int, float)) or t != 1:
            alpha = torch.linalg.norm(x, dim=cart_dim) * t  # go via vector length

        if not self.learn and self.semi_grad:
            with torch.no_grad():
                alpha = softmax(alpha, index, ptr, dim_size, dim)
        else:
            alpha = softmax(alpha, index, ptr, dim_size, dim)
        return self.reduce(x * (alpha[:, None, :] + b[None, :, :]), index, ptr, dim_size, dim, reduce='sum')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')

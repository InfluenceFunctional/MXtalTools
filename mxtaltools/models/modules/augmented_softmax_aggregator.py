"""Softmax aggregation operators with a temperature and an additive bias on the weights,
for scalar features (:class:`AugSoftmaxAggregation`) and for vector features weighted by
vector length (:class:`VectorAugSoftmaxAggregation`)."""

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax


class AugSoftmaxAggregation(Aggregation):
    r"""The softmax aggregation operator based on a temperature term, as
    described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper, modified with a bias term added to
    the softmax weights (learnable when ``learn`` is True).

    For each group of rows sharing an ``index`` (or ``ptr`` segment) along ``dim``,
    the output is ``sum_i x_i * (softmax_i(t * x_i) + b)``, computed elementwise per
    feature column.

    Parameters
    ----------
    temperature : float, optional
        Initial value of the temperature ``t``. Default 1.0.
    learn : bool, optional
        If True, ``t`` and ``b`` are :class:`torch.nn.Parameter` of shape [channels],
        filled with ``temperature`` and ``bias``. If False, they are the Python floats
        ``temperature`` and ``bias``. Default True.
    semi_grad : bool, optional
        Allowed only with ``learn=False``. If True, the softmax weights are computed
        under :func:`torch.no_grad`, so gradients reach ``x`` only through the factor
        ``x`` in ``x * (weights + b)``. Default False.
    channels : int, optional
        Number of feature columns with their own ``t`` and ``b``. Default 1.
    bias : float, optional
        Initial value of the bias ``b``. Default 0.1.

    Raises
    ------
    ValueError
        If ``learn`` and ``semi_grad`` are both True, or if ``learn`` is False and
        ``channels`` is not 1.
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
        """Refill ``t`` and ``b`` with their initial values when they are tensors
        (``learn=True``); no-op otherwise."""
        if isinstance(self.t, Tensor):
            self.t.data.fill_(self._init_termperature)
        if isinstance(self.b, Tensor):
            self.b.data.fill_(self._init_bias)

    def forward(self, x: Tensor,
                index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        """Aggregate ``x`` over the groups given by ``index`` or ``ptr``.

        Parameters
        ----------
        x : Tensor
            Features to aggregate, e.g. [num_rows, channels]. When ``channels`` is not
            1, ``x`` must be two-dimensional and ``dim`` must be 0 or -2.
        index : Tensor, optional
            Group index of each row of ``x`` along ``dim``.
        ptr : Tensor, optional
            CSR segment pointers, used in place of ``index``.
        dim_size : int, optional
            Number of output groups.
        dim : int, optional
            Dimension to aggregate over. Default -2.

        Returns
        -------
        Tensor
            ``sum_i x_i * (softmax_i(t * x_i) + b)`` per group, with ``dim`` of size
            ``dim_size``.

        Raises
        ------
        ValueError
            From :meth:`assert_two_dimensional_input`, when ``channels`` is not 1 and
            ``x`` is not two-dimensional or ``dim`` is not 0 or -2.
        """

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
    """Variant of :class:`AugSoftmaxAggregation` that weighs by vector length rather
    than raw value.

    For vector features ``x`` of shape [num_rows, 3, channels], the softmax weights
    are computed per channel from ``t * ||x_i||`` (norm over ``cart_dim``), and the
    output per group is ``sum_i x_i * (softmax_i(t * ||x_i||) + b)``, the same weight
    applied to every Cartesian component of a vector.

    Parameters
    ----------
    temperature : float, optional
        Initial value of the temperature ``t``. Default 1.0.
    learn : bool, optional
        If True, ``t`` and ``b`` are :class:`torch.nn.Parameter` of shape [channels],
        filled with ``temperature`` and ``bias``. If False, they are Python floats.
        Default True.
    semi_grad : bool, optional
        Allowed only with ``learn=False``. If True, the softmax weights are computed
        under :func:`torch.no_grad`. Default False.
    channels : int, optional
        Number of vector channels with their own ``t`` and ``b``. Default 1.
    bias : float, optional
        Initial value of the bias ``b``. Default 0.1.

    Raises
    ------
    ValueError
        If ``learn`` and ``semi_grad`` are both True, or if ``learn`` is False and
        ``channels`` is not 1.

    Notes
    -----
    :meth:`forward` indexes ``b`` as ``b[None, :, :]``, which requires ``b`` to be the
    2-D view made when ``channels`` is not 1. Construction with the default
    ``channels=1`` succeeds, but :meth:`forward` then raises IndexError
    (``learn=True``, 1-D ``b``) or TypeError (``learn=False``, float ``b``).
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
        """Refill ``t`` and ``b`` with their initial values when they are tensors
        (``learn=True``); no-op otherwise."""
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
        """Aggregate vector features ``x`` over the groups given by ``index`` or ``ptr``.

        Parameters
        ----------
        x : Tensor
            Vector features, [num_rows, 3, channels].
        index : Tensor, optional
            Group index of each row of ``x`` along ``dim``, [num_rows].
        ptr : Tensor, optional
            CSR segment pointers, used in place of ``index``.
        dim_size : int, optional
            Number of output groups.
        dim : int, optional
            Dimension to aggregate over. Default 0.
        cart_dim : int, optional
            Dimension over which vector norms are taken. Default 1. The weights are
            broadcast back onto ``x`` along axis 1 regardless of this value.

        Returns
        -------
        Tensor
            ``sum_i x_i * (softmax_i(t * ||x_i||) + b)`` per group,
            [dim_size, 3, channels] for the default ``dim`` and ``cart_dim``.
        """

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

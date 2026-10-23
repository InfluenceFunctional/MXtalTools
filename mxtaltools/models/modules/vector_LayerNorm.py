"""Norm-based scaling layer for vector features of shape [n, 3, channels]."""

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import ones
from torch_geometric.utils import scatter


class VectorLayerNorm(torch.nn.Module):
    r"""Simplified graphwise layernorm operating on the norms of vectors only,
    based on the torch_geometric ``LayerNorm``.

    Inputs are not centred: each vector is divided by a per-channel norm statistic
    and, if ``affine``, multiplied by a per-channel weight. The statistic depends on
    ``mode``:

    - ``'graph'``: the mean vector norm over the rows of each graph (all rows when no
      ``batch`` is given), computed separately for each channel.
    - ``'node'``: each vector's own norm, so every row and channel is scaled to length
      ``norm / (norm + eps)`` (close to 1 unless the norm is small) before the weight
      is applied.

    Parameters
    ----------
    in_channels : int
        Number of vector channels (last dimension of the input).
    eps : float, optional
        Added to the norm statistic before dividing. Default 1e-5.
    affine : bool, optional
        If True, a learnable ``weight`` of shape [in_channels], initialised to ones,
        multiplies the output; if False, ``weight`` is None. Default True.
    mode : str, optional
        ``'graph'`` or ``'node'``. Default ``'graph'``. Any other value is accepted at
        construction and raises ValueError in :meth:`forward`.
    """

    def __init__(
            self,
            in_channels: int,
            eps: float = 1e-5,
            affine: bool = True,
            mode: str = 'graph',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine
        self.mode = mode

        if affine:
            self.weight = Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module: fills ``weight`` with ones
        (no-op when ``affine`` is False)."""
        ones(self.weight)

    def forward(self,
                v: Tensor,
                batch: Optional[torch.LongTensor] = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Scale vectors ``v`` by the norm statistic selected by ``mode``.

        Parameters
        ----------
        v : Tensor
            The source tensor, vector [n, 3, k] with ``k = in_channels``; norms are
            taken over dimension 1.
        batch : torch.LongTensor, optional
            The batch vector :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which
            assigns each row of ``v`` to a specific example, [n]. Used only in
            ``'graph'`` mode; if None, all rows are treated as one example.
            Default None.
        batch_size : int, optional
            The number of examples :math:`B`. Computed as ``int(batch.max()) + 1``
            if not given. Used only in ``'graph'`` mode with ``batch``. Default None.

        Returns
        -------
        Tensor
            Scaled vectors, same shape as ``v``.

        Raises
        ------
        ValueError
            If ``mode`` is neither ``'graph'`` nor ``'node'``.
        """
        if self.mode == 'graph':
            if batch is None:  # assumes whole input is single graph
                norm = torch.linalg.norm(v, dim=1).mean(0)
                out = v / (norm + self.eps)[None, None, :]

            else:  # take norms graph-wise
                if batch_size is None:
                    batch_size = int(batch.max()) + 1

                norm = torch.linalg.norm(v, dim=1)

                mean = scatter(norm, batch, dim=0, dim_size=batch_size, reduce='mean')

                out = v / (mean.index_select(0, batch) + self.eps)[:, None, :]

            if self.weight is not None:
                out = out * self.weight

            return out

        if self.mode == 'node':  # separate norms node-by-node
            norm = torch.linalg.norm(v, dim=1)
            out = v / (norm + self.eps)[:, None, :]

            if self.weight is not None:
                out = out * self.weight

            return out

        raise ValueError(f"Unknown normalization mode: {self.mode}")

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'affine={self.affine}, mode={self.mode})')

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.inits import ones
from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree, scatter


class VectorLayerNorm(torch.nn.Module):
    r"""  # TODO confirm layer vs batch norm behavior
    Simplified graphwise layernorm operating on the norms of vectors only
    based on torch gnn layernorm
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
        r"""Resets all learnable parameters of the module."""
        ones(self.weight)

    def forward(self, v: Tensor, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""
        Args:
            v (torch.Tensor): The source tensor, vector [nx3xk].
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
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

    '''
    graphwise vs full-set test
    norm = torch.linalg.norm(v, dim=1).mean(0)
    out1 = v / (norm + self.eps)[None, None, :]
    
    if batch_size is None:
        batch_size = int(batch.max()) + 1
    
    norm = torch.linalg.norm(v, dim=1)
    
    mean = scatter(norm, batch, dim=0, dim_size=batch_size, reduce='mean')
    
    out2 = v / (mean.index_select(0, batch) + self.eps)[:, None, :]
    
    means1= torch.stack([torch.linalg.norm(out1[batch==ind],dim=1).mean() for ind in range(50)])
    means2= torch.stack([torch.linalg.norm(out2[batch==ind],dim=1).mean() for ind in range(50)])

    
    '''

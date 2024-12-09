import sys
from argparse import Namespace
from math import pi as PI
from typing import Optional
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn as nn
from torch.nn import Parameter
from torch_geometric import nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax

from CrystalData import CrystalData
from utils import (
    VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD)
from utils import collate_decoded_data, swarm_vs_tgt_fig, ae_reconstruction_loss


# noinspection PyAttributeOutsideInit
class Normalization(nn.Module):
    r"""
    Wrapper module for several normalization options

    Args:
        norm (str): type of normalization function
        filters (int): feature depth of objects to be normalized
    """

    def __init__(self,
                 norm: str,
                 filters: int):
        super().__init__()
        self.norm_type = norm
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(filters)
        elif norm == 'graph layer':
            self.norm = gnn.LayerNorm(filters)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(filters)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(filters)  # not tested
        elif norm == 'graph':
            self.norm = gnn.GraphNorm(filters)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            print(norm + " is not a valid normalization")
            sys.exit()

    def forward(self,
                x: torch.Tensor,
                batch: Optional[torch.LongTensor] = None):

        if batch is not None and self.norm_type != 'batch' and self.norm_type != 'layer' and self.norm_type is not None:
            return self.norm(x, batch)

        return self.norm(x)


class Activation(nn.Module):
    r"""
    Wrapper module for several activation options

    Args:
        activation_func (str): type of activation function
        filters (int): feature depth of objects to be normalized
    """

    def __init__(self,
                 activation_func: str,
                 filters: int):
        super().__init__()
        if activation_func is not None:
            if activation_func.lower() == 'relu':
                self.activation = F.relu
            elif activation_func.lower() == 'gelu':
                self.activation = F.gelu
            elif activation_func.lower() == 'leaky relu':
                self.activation = F.leaky_relu
            elif activation_func.lower() == 'tanh':
                self.activation = F.tanh
            elif activation_func.lower() == 'sigmoid':
                self.activation = F.sigmoid
        elif activation_func is None:
            self.activation = nn.Identity()

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return self.activation(x)


def radius(x: torch.Tensor, y: torch.Tensor, r: float,
           batch_x: Optional[torch.Tensor] = None,
           batch_y: Optional[torch.Tensor] = None,
           max_num_neighbors: int = 32,
           num_workers: int = 1) -> torch.Tensor:
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)

    .. code-block:: python

        import torch
        from torch_cluster import radius

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    ptr_x: Optional[torch.Tensor] = None
    if batch_x is not None:
        assert x.size(0) == batch_x.numel()
        batch_size = int(batch_x.max()) + 1

        deg = x.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch_x, torch.ones_like(batch_x))

        ptr_x = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr_x[1:])

    ptr_y: Optional[torch.Tensor] = None
    if batch_y is not None:
        assert y.size(0) == batch_y.numel()
        batch_size = int(batch_y.max()) + 1

        deg = y.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch_y, torch.ones_like(batch_y))

        ptr_y = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr_y[1:])

    return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
                                          max_num_neighbors, num_workers)


# @torch.jit.script
def asymmetric_radius_graph(x: torch.Tensor,
                            r: float,
                            inside_inds: torch.Tensor,
                            convolve_inds: torch.Tensor,
                            batch: torch.Tensor,
                            loop: bool = False,
                            max_num_neighbors: int = 32, flow: str = 'source_to_target',
                            num_workers: int = 1) -> torch.Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        inside_inds (Tensor): original indices for the nodes in the y subgraph

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """
    if convolve_inds is None:  # indexes of items within x to convolve against y
        convolve_inds = torch.arange(len(x))

    assert flow in ['source_to_target', 'target_to_source']
    if batch is not None:
        edge_index = radius(x[convolve_inds], x[inside_inds], r, batch[convolve_inds], batch[inside_inds],
                            max_num_neighbors if loop else max_num_neighbors + 1,
                            num_workers)
    else:
        edge_index = radius(x[convolve_inds], x[inside_inds], r, None, None,
                            max_num_neighbors if loop else max_num_neighbors + 1,
                            num_workers)

    target, source = edge_index[0], edge_index[1]

    # edge_index[1] = inside_inds[edge_index[1, :]] # reindex
    target = inside_inds[target]  # contains correct indexes
    source = convolve_inds[source]

    if flow == 'source_to_target':
        row, col = source, target
    else:
        row, col = target, source

    if not loop:  # now properly deletes self-loops
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


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


class BesselBasisLayer(torch.nn.Module):  # NOTE borrowed from DimeNet implementation
    def __init__(self,
                 num_radial: int,
                 cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        super(BesselBasisLayer, self).__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff))
        self.envelope = Envelope(envelope_exponent)
        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self,
                dist: torch.Tensor
                ) -> torch.Tensor:
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class Envelope(torch.nn.Module):
    def __init__(self,
                 exponent: float):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class GaussianEmbedding(torch.nn.Module):
    def __init__(self,
                 start: float = 0.0,
                 stop: float = 5.0,
                 num_gaussians: int = 50):
        super(GaussianEmbedding, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

        self.register_buffer('offset', offset)
        self.register_buffer('coeff', torch.tensor([coeff], dtype=torch.float32))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def construct_radial_graph(pos: torch.FloatTensor,
                           batch: torch.LongTensor,
                           ptr: torch.LongTensor,
                           cutoff: float,
                           max_num_neighbors: int,
                           aux_ind=None,
                           mol_ind=None):
    r"""
    Construct edge indices over a radial graph.
    Optionally, compute intra (within ref_mol_inds) and inter (between ref_mol_inds and outside inds) edges.
    Args:
        pos: node positions
        batch: index of graph to which each node belongs
        ptr: edges of batch
        cutoff: maximum edge length
        max_num_neighbors: maximum number of neighbors per node
        aux_ind: optional auxiliary index for identifying "inside" and "outside" nodes
        mol_ind: optional index for the identity of the molecule a given atom is inside, for when there are multiple molecules per asymmetric unit, or in a cluster of molecules

    Returns:
        dict: dictionary of edge information
    """
    if aux_ind is not None:  # there is an 'inside' 'outside' distinction
        inside_bool = aux_ind == 0
        outside_bool = aux_ind == 1
        inside_inds = torch.where(inside_bool)[0]
        # atoms which are not in the asymmetric unit but which we will convolve - pre-excluding many from outside the cutoff
        outside_inds = torch.where(outside_bool)[0]
        inside_batch = batch[inside_inds]  # get the feature vectors we want to repeat
        n_repeats = [int(torch.sum(batch == ii) / torch.sum(inside_batch == ii)) for ii in
                     range(len(ptr) - 1)]  # number of molecules in convolution region

        # intramolecular edges
        edge_index = asymmetric_radius_graph(pos, batch=batch, r=cutoff,
                                             # intramolecular interactions - stack over range 3 convolutions
                                             max_num_neighbors=max_num_neighbors, flow='source_to_target',
                                             inside_inds=inside_inds, convolve_inds=inside_inds)

        # intermolecular edges
        edge_index_inter = asymmetric_radius_graph(pos, batch=batch, r=cutoff,
                                                   # extra radius for intermolecular graph convolution
                                                   max_num_neighbors=max_num_neighbors, flow='source_to_target',
                                                   inside_inds=inside_inds, convolve_inds=outside_inds)

        # for zp>1 systems, we also need to generate intermolecular edges within the asymmetric unit
        if mol_ind is not None:
            # for each inside molecule, get its edges to the Z'-1 other 'inside' symmetry units
            unique_mol_inds = torch.unique(mol_ind)
            if len(unique_mol_inds) > 1:
                for zp in unique_mol_inds:
                    inside_nodes = torch.where(inside_bool * (mol_ind == zp))[0]
                    outside_nodes = torch.where(inside_bool * (mol_ind != zp))[0]

                    # intramolecular edges
                    edge_index_inter = torch.cat([edge_index_inter,
                                                  asymmetric_radius_graph(
                                                      pos, batch=batch, r=cutoff,
                                                      max_num_neighbors=max_num_neighbors, flow='source_to_target',
                                                      inside_inds=inside_nodes, convolve_inds=outside_nodes)],
                                                 dim=1)

        return {'edge_index': edge_index,
                'edge_index_inter': edge_index_inter,
                'inside_inds': inside_inds,
                'outside_inds': outside_inds,
                'inside_batch': inside_batch,
                'n_repeats': n_repeats}

    else:

        edge_index = gnn.radius_graph(pos, r=cutoff, batch=batch,
                                      max_num_neighbors=max_num_neighbors,
                                      flow='source_to_target')  # note - requires batch be monotonically increasing

        return {'edge_index': edge_index}


# noinspection PyAttributeOutsideInit
class scalarMLP(nn.Module):  # todo simplify and smooth out +1's and other custom methods for a general depth controller
    r"""
    Flexible multi-layer perceptron module, with several options.

    Args:
        layers (int): number of fully-connected layers
        filters (int): feature depth with FC layers
        input_dim (int): feature depth of inputs
        output_dim (int): feature depth of outputs
        activation (str): activation function
        seed (int): random seed
        dropout (float): dropout probability
        ramp_depth (bool): whether to ramp the feature depth exponentially from input_dim to output_dim through the network
    """

    def __init__(self,
                 layers: int,
                 filters: int,
                 input_dim: int,
                 output_dim: int,
                 activation: str = 'gelu',
                 seed: int = 0,
                 dropout: float = 0,
                 norm: 'str' = None,
                 norm_after_linear: bool = True,
                 bias: bool = True,
                 ramp_depth: bool = False,
                 ):
        super(scalarMLP, self).__init__()
        # initialize constants and layers
        self.n_layers = layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.norm_mode = norm
        self.dropout_p = dropout
        self.activation = activation
        self.bias = bias
        self.norm_after_linear = norm_after_linear

        self.ramp_depth = ramp_depth

        torch.manual_seed(seed)
        self.init_scalar_filters(filters)
        self.init_scalar_transforms()

    def init_scalar_filters(self, filters):
        """
        returns a list with layers + 1 integer elements
        """
        if self.ramp_depth:  # smoothly ramp feature depth across layers
            # linear scaling
            # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
            # log scaling for consistent growth ratio
            p = np.log(self.output_dim) / np.log(self.input_dim)
            n_filters = [self.input_dim] + [int(self.input_dim ** (1 + (p - 1) * (i / self.n_layers))) for i in
                                            range(self.n_layers)]
        else:
            n_filters = [self.input_dim] + [filters for _ in range(self.n_layers)]
        self.s_filters_in = n_filters[:-1]
        self.s_filters_out = n_filters[1:]

        if n_filters.count(n_filters[0]) != len(
                n_filters):  # if they are not all the same, we need residue adjustments
            self.same_depth = False
            self.residue_adjust = torch.nn.ModuleList([
                nn.Linear(self.s_filters_in[i], self.s_filters_out[i], bias=False)
                for i in range(self.n_layers)
            ])
        else:
            self.same_depth = True

    def init_scalar_transforms(self):
        """scalar MLP layers"""

        '''working layers'''
        self.fc_layers = torch.nn.ModuleList([
            nn.Linear(self.s_filters_in[i],
                      self.s_filters_out[i], bias=self.bias)
            for i in range(self.n_layers)
        ])
        self.fc_activations = torch.nn.ModuleList([
            Activation(self.activation, self.s_filters_out[i])
            for i in range(self.n_layers)
        ])
        if self.norm_after_linear:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode, self.s_filters_out[i])
                for i in range(self.n_layers)
            ])
        else:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode,
                              self.s_filters_in[i]
                              )
                for i in range(self.n_layers)
            ])
        self.fc_dropouts = torch.nn.ModuleList([
            nn.Dropout(p=self.dropout_p)
            for _ in range(self.n_layers)
        ])

        '''output layer'''
        if self.output_dim != self.s_filters_out[-1]:
            self.output_layer = nn.Linear(self.s_filters_out[-1], self.output_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                return_latent: bool = False,
                batch: Optional[torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """initialize to correct feature dimension"""

        for i, (norm, linear, activation, dropout) in enumerate(
                zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts)):

            'get residue'
            if self.same_depth:
                res_x = x.clone()
            else:
                res_x = self.residue_adjust[i](x)

            'linear layer'
            if self.norm_after_linear:
                x = res_x + dropout(activation(norm(linear(x), batch=batch)))
            else:
                x = res_x + dropout(activation(linear(norm(x, batch=batch))))

        if return_latent:
            return self.output_layer(x), x
        else:
            return self.output_layer(x)

class VectorActivation(nn.Module):
    r"""
    Modified implementation of the vector activation function from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_layers.py

    Generates an axis as a learned linear combination of input v, then the normalized overlaps of all the components of v.

    Applies an activation function on the vector overlaps, such that, e.g., for ReLU activation, vectors with negative overlap are rotated to be perpendicular to the learned axis (zero overlap) and vectors with positive overlap are untouched.

    Args:
        hidden_dim (int): feature depth of input/output vectors, :math:`(k\times 3)`
        act_func (str): activation function to apply to the normalized vector overlaps with the learned axis
    """

    def __init__(self,
                 hidden_dim: int,
                 act_func: str):
        super(VectorActivation, self).__init__()

        self.embedding = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = Activation(act_func, hidden_dim)

    def forward(self,
                v: torch.Tensor
                ) -> torch.Tensor:
        direction = self.embedding(v)[..., -1]
        normed_direction = direction / (torch.linalg.norm(direction, dim=1, keepdim=True) + 1e-5)

        projection = torch.einsum('nik,ni->nk', v, normed_direction).clip(max=0)
        correction = -self.activation(projection[..., None]) * normed_direction[:, None, :]

        activated_output = v + correction.permute(0, 2, 1)

        return activated_output

# noinspection PyAttributeOutsideInit
class vectorMLP(scalarMLP):
    r"""
    scalarMLP model with l=1 vector track added with o3 equivariance
    """

    def __init__(self,
                 layers: int,
                 filters: int,
                 input_dim: int,
                 output_dim: int,
                 vector_input_dim: int,
                 vector_output_dim: int,
                 activation: str = 'gelu',
                 seed: int = 0,
                 dropout: float = 0,
                 norm: str = None,
                 norm_after_linear: bool = True,
                 bias: bool = True,
                 vector_norm: str = None,
                 ramp_depth: bool = False,
                 v_to_s_combination: str = 'sum'):
        super(scalarMLP, self).__init__()
        # initialize constants and layers
        self.n_layers = layers
        self.output_dim = output_dim
        self.v_output_dim = vector_output_dim
        self.v_input_dim = vector_input_dim
        self.input_dim = input_dim
        self.norm_mode = norm
        self.dropout_p = dropout
        self.activation = activation
        self.bias = bias
        self.norm_after_linear = norm_after_linear
        self.v_norm_mode = vector_norm
        self.ramp_depth = ramp_depth
        self.v_to_s_combination = v_to_s_combination

        torch.manual_seed(seed)

        # addition of two normally distributed 3-vectors increases the norm by roughly this factor on average
        # divide this out to combat vector elongation & poor gradient flow
        self.vector_addition_rescaling_factor = 1.6

        self.init_scalar_filters(filters)
        self.init_vector_filters(filters)

        self.init_scalar_transforms()
        self.init_vector_transforms()

    def init_vector_filters(self, filters):
        if self.ramp_depth:  # smoothly ramp feature depth across layers
            # linear scaling
            # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
            # exp scaling for consistent growth ratio
            p = np.log(self.v_output_dim) / np.log(self.input_dim)
            v_n_filters = [self.v_input_dim] + [int(self.v_input_dim ** (1 + (p - 1) * (i / (self.n_layers)))) for i in
                           range(self.n_layers)]
        else:
            v_n_filters = [self.v_input_dim] + [filters for _ in range(self.n_layers)]
        self.v_filters_in = v_n_filters[:-1]
        self.v_filters_out = v_n_filters[1:]

        # if they are not all the same, we need residue adjustments
        if v_n_filters.count(v_n_filters[0]) != len(v_n_filters):
            self.v_same_depth = False
            self.v_residue_adjust = torch.nn.ModuleList([
                nn.Linear(self.v_filters_in[i], self.v_filters_out[i], bias=False)
                for i in range(self.n_layers)
            ])
        else:
            self.v_same_depth = True

    def init_vector_transforms(self):
        """vector MLP layers"""
        '''working layers'''
        self.v_fc_layers = torch.nn.ModuleList([
            nn.Linear(self.v_filters_in[i], self.v_filters_out[i], bias=False)
            for i in range(self.n_layers)
        ])
        self.s_to_v_gating_layers = torch.nn.ModuleList([
            nn.Linear(self.s_filters_out[i], self.v_filters_out[i], bias=False)
            for i in range(self.n_layers)
        ])
        self.s_to_v_activations = torch.nn.ModuleList(
            [  # use tanh as gating function rather than standard activation which is unbound
                Activation(self.activation, self.v_filters_out[i])
                # positive outputs only to maintain equivariance (no vectors flipped)
                for i in range(self.n_layers)
            ])
        self.v_fc_norms = torch.nn.ModuleList([
            Normalization(self.v_norm_mode, self.v_filters_out[i])
            for i in range(self.n_layers)
        ])
        self.vector_to_scalar = torch.nn.ModuleList([
            Scalarizer(self.v_filters_in[i], 3, self.norm_mode, self.activation, self.dropout_p,
                       output_dim=self.s_filters_in[i])
            for i in range(self.n_layers)
        ])
        self.scalar_to_vector_norm = torch.nn.ModuleList([
            Normalization(self.norm_mode, self.v_filters_out[i])
            for i in range(self.n_layers)
        ])
        self.vector_activation = torch.nn.ModuleList([
            VectorActivation(self.v_filters_out[i], self.activation)
            for i in range(self.n_layers)
        ])

        '''output layer'''
        if self.v_output_dim != self.v_filters_out[-1]:
            self.v_output_layer = nn.Linear(self.v_filters_out[-1], self.v_output_dim, bias=False)
        else:
            self.v_output_layer = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                return_latent: bool = False,
                batch: Optional[torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        for i, (s_norm, s_linear, s_act, s_dropout,
                v_norm, v_linear, v_act,
                s2v_norm, s2v_linear, s2v_act,
                v2s_linear) in enumerate(
            zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts,
                self.v_fc_norms, self.v_fc_layers, self.vector_activation,
                self.scalar_to_vector_norm, self.s_to_v_gating_layers, self.s_to_v_activations,
                self.vector_to_scalar)):
            res_x, res_v = self.get_residues(i, x, v)

            if torch.sum(torch.isnan(x)) != 0:
                assert False, "NaN values in EMLP scalars"

            if torch.sum(torch.isnan(v)) != 0:
                assert False, "NaN values in EMLP vectors"

            'scalar forward'
            if self.v_to_s_combination == 'concatenate':
                # concatenate vector lengths to scalar values
                x = torch.cat([x, v2s_linear(v)], dim=-1)
            elif self.v_to_s_combination == 'sum':
                x = x + v2s_linear(v)
            else:
                assert False, f'{self.v_to_s_combination} not implemented'

            if self.norm_after_linear:
                x = res_x + s_dropout(s_act(s_norm(s_linear(x), batch=batch)))
            else:
                x = res_x + s_dropout(s_act(s_linear(s_norm(x, batch=batch))))

            'vector forward'  # A(FC(x)) * FC(N(v))   # rescaling factor keeps norm from exploding
            s2v_gating = s2v_act(s2v_norm(s2v_linear(x))[:, None, :])
            v = v_act(v_norm(v_linear(v), batch=batch))
            v = (res_v + s2v_gating * v) / self.vector_addition_rescaling_factor

        if torch.sum(torch.isnan(x)) != 0:
            assert False, "NaN values in EMLP scalars"
        if torch.sum(torch.isnan(v)) != 0:
            assert False, "NaN values in EMLP vectors"

        if return_latent:
            return self.output_layer(x), self.v_output_layer(v), x
        else:
            return self.output_layer(x), self.v_output_layer(v)

    def get_residues(self,
                     i: int,
                     x: torch.Tensor,
                     v: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.same_depth:
            x = x.clone()
        else:
            x = self.residue_adjust[i](x)

        if self.v_same_depth:
            v = v.clone()
        else:
            v = self.v_residue_adjust[i](v)

        return x, v


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
            edge_index,
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


class EmbeddingBlock(torch.nn.Module):
    def __init__(self,
                 init_node_embedding_dim: int,
                 num_input_classes: int,
                 num_scalar_input_features: int,
                 atom_type_embedding_dim: int):
        super(EmbeddingBlock, self).__init__()

        self.embeddings = nn.Embedding(num_input_classes + 1, atom_type_embedding_dim)
        self.linear = nn.Linear(atom_type_embedding_dim + num_scalar_input_features - 1, init_node_embedding_dim)

    def forward(self,
                x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        # always embed the first dimension only (by convention, atomic number)
        embedding = self.embeddings(x[:, 0].long())

        return self.linear(torch.cat([embedding, x[:, 1:]], dim=-1))


class BaseGraphModel(torch.nn.Module):
    def __init__(self):
        super(BaseGraphModel, self).__init__()
        self.atom_feats = 0
        self.mol_feats = 0
        self.n_mol_feats = 0
        self.n_atom_feats = 0

    def get_data_stats(self,
                       atom_features: list,
                       molecule_features: list,
                       node_standardization_tensor: OptTensor = None,
                       graph_standardization_tensor: OptTensor = None
                       ):

        if node_standardization_tensor is None:
            node_standardization_tensor = torch.ones((len(atom_features), 2), dtype=torch.float32)
            node_standardization_tensor[:, 0] = 0
        if graph_standardization_tensor is None:
            graph_standardization_tensor = torch.ones((len(molecule_features), 2), dtype=torch.float32)
            graph_standardization_tensor[:, 0] = 0

        self.n_atom_feats = len(atom_features)
        self.n_mol_feats = len(molecule_features)
        self.atom_feats = atom_features
        self.mol_feats = molecule_features

        # generate atom property embeddings
        atom_embeddings_list = [torch.arange(len(VDW_RADII))]  # start with raw atomic number
        if 'vdw_radii' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(VDW_RADII.values())))
        if 'atom_weight' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(ATOM_WEIGHTS.values())))
        if 'electronegativity' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(ELECTRONEGATIVITY.values())))
        if 'group' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(GROUP.values())))
        if 'period' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(PERIOD.values())))

        assert len(atom_embeddings_list) == self.n_atom_feats

        self.register_buffer('atom_properties_tensor', torch.stack(atom_embeddings_list).T)

        if not torch.is_tensor(node_standardization_tensor):
            node_standardization_tensor = torch.tensor(node_standardization_tensor, dtype=torch.float32)
        if not torch.is_tensor(graph_standardization_tensor):
            graph_standardization_tensor = torch.tensor(graph_standardization_tensor, dtype=torch.float32)

        # store atom standardizations
        self.register_buffer('node_standardization_tensor', node_standardization_tensor)
        if self.n_mol_feats != 0:
            self.register_buffer('graph_standardization_tensor', graph_standardization_tensor)

    def featurize_input_graph(self,
                              data: CrystalData
                              ) -> CrystalData:
        if data.x.ndim > 1:
            data.x = data.x[:, 0]

        data.x = self.atom_properties_tensor[data.x.long()]

        if self.n_mol_feats > 0:
            mol_x_list = []
            if 'num_atoms' in self.mol_feats:
                mol_x_list.append(data.num_atoms)
            if 'radius' in self.mol_feats:
                mol_x_list.append(data.radius)
            if 'mol_volume' in self.mol_feats:
                mol_x_list.append(data.mol_volume)
            data.mol_x = torch.stack(mol_x_list).T

        return data

    def standardize(self,
                    data: CrystalData
                    ) -> CrystalData:

        data.x = (data.x - self.node_standardization_tensor[:, 0]) / self.node_standardization_tensor[:, 1]

        if self.n_mol_feats > 0:
            data.mol_x = (
                    (data.mol_x - self.graph_standardization_tensor[:, 0]) / self.graph_standardization_tensor[:, 1])

        return data

    def forward(self,
                data: CrystalData,
                return_dists: bool = False,
                return_latent: bool = False
                ):
        # featurize atom properties on the fly
        data = self.featurize_input_graph(data)

        # standardize on the fly from model-attached statistics
        data = self.standardize(data)

        return self.model(data.x,
                          data.pos,
                          data.batch,
                          data.ptr,
                          data.mol_x,
                          data.num_graphs,
                          edge_index=data.edge_index,
                          return_dists=return_dists,
                          return_latent=return_latent)

    def compile_self(self, dynamic=True, fullgraph=False):
        self.model = torch.compile(self.model, dynamic=dynamic, fullgraph=fullgraph)



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


# noinspection PyAttributeOutsideInit
class VectorMoleculeGraphModel(nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_pos_to_node_dim: bool = False,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(VectorMoleculeGraphModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_pos_to_node_dim = concat_pos_to_node_dim
        self.concat_mol_to_node_dim = concat_mol_to_node_dim

        if self.concat_pos_to_node_dim:
            input_node_dim += 1  # radial dimension - vector features explicitly added later

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats

        self.graph_net = VectorGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        self.v_global_pool = VectorAugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = vectorMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     vector_input_dim=fc_config.hidden_dim,
                                     v_to_s_combination='sum',
                                     vector_norm=fc_config.vector_norm,
                                     vector_output_dim=fc_config.hidden_dim,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
            self.v_output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()
            self.v_output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                batch: torch.LongTensor,
                ptr: torch.LongTensor,
                num_graphs: int,
                mol_x: Optional[torch.Tensor] = None,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:

        if len(self.graph_net.interaction_blocks) > 0 or return_dists:
            if edges_dict is None:  # option to rebuild radial graph
                edges_dict = construct_radial_graph(
                    pos,
                    batch,
                    ptr,
                    self.convolution_cutoff,
                    self.max_num_neighbors,
                )
            else:
                edges_dict = None

        x, v = self.append_init_node_features(x, pos, ptr, mol_x)
        x, v = self.graph_net(x,
                              v,
                              pos,
                              batch,
                              edges_dict)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation
        x = self.global_pool(x,
                             batch,
                             dim_size=num_graphs)
        v = self.v_global_pool(v,
                               batch,
                               dim_size=num_graphs,
                               dim=0,
                               cart_dim=1)

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                x = torch.cat([x, self.mol_fc(mol_x)], dim=-1)
            x, v = self.gnn_mlp(x, v)

        x_out, v_out = self.output_fc(x), self.v_output_fc(v)

        extra_outputs = self.collect_extra_outputs(x,
                                                   pos,
                                                   batch,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return x_out, v_out, extra_outputs
        else:
            return x_out, v_out

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              pos: torch.Tensor,
                              batch: torch.LongTensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self,
                                  x: torch.Tensor,
                                  pos: torch.Tensor,
                                  ptr: torch.LongTensor,
                                  mol_x: Optional[torch.Tensor] = None,
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x[:, None]

        # append radial position as scalar feature
        # and 3 vector dimensions (unit vectors from centroid)
        rad = torch.linalg.norm(pos, dim=1)
        if self.concat_pos_to_node_dim:
            x = torch.cat((x, rad[:, None]), dim=-1)  # radii
        # v = pos / (rad[:, None] + 1e-5)  # normed directions
        v = pos[..., None]  # set dimension as [n,3,k]
        # richer embedding as 3 component vectors rather than one single vector
        #v = pos[:, :, None] * torch.eye(3, device=pos.device, dtype=torch.float32).repeat(len(pos), 1, 1)

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x, v


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
            edge_index,
            edge_attr,
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


class VectorGNN(torch.nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 node_dim: int,
                 fcs_per_gc: int,
                 message_dim: int,
                 embedding_dim: int,
                 num_convs: int,
                 num_radial: int,
                 num_input_classes=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 atom_type_embedding_dim: int = 5,
                 norm: Optional[str] = None,
                 vector_norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 override_cutoff: Optional[float] = None,
                 v_embedding_dim: Optional[int] = None,
                 v_input_node_dim: Optional[int] = None,
                 ):
        super(VectorGNN, self).__init__()

        self.max_num_neighbors = max_num_neighbors

        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        if atom_type_embedding_dim == 0:
            self.init_node_embedding = nn.Identity()
        else:
            self.init_node_embedding = EmbeddingBlock(node_dim,
                                                      num_input_classes,
                                                      input_node_dim,
                                                      atom_type_embedding_dim)
        if v_input_node_dim is None:
            v_input_node_dim = 1
        self.init_vector_embedding = self.init_vector_embedding = nn.Linear(v_input_node_dim, node_dim, bias=False)

        self.zeroth_fc_block = vectorMLP(layers=fcs_per_gc,
                                         filters=node_dim,
                                         input_dim=node_dim,
                                         output_dim=node_dim,
                                         activation=activation,
                                         norm=norm,
                                         dropout=dropout,
                                         vector_input_dim=node_dim,
                                         vector_output_dim=node_dim,
                                         vector_norm=vector_norm)

        self.interaction_blocks = torch.nn.ModuleList([
            MConv(
                message_dim=message_dim,
                node_dim=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
                activation_fn=activation)
            for _ in range(num_convs)
        ])
        self.vector_interaction_blocks = torch.nn.ModuleList([
            v_MConv(
                message_depth=message_dim,
                node_depth=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
            )
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            vectorMLP(layers=fcs_per_gc,
                      filters=node_dim,
                      input_dim=node_dim,
                      output_dim=node_dim,
                      activation=activation,
                      norm=norm,
                      dropout=dropout,
                      vector_norm=vector_norm,
                      vector_input_dim=node_dim,
                      vector_output_dim=node_dim)
            for _ in range(num_convs)
        ])

        if node_dim != embedding_dim:
            self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

        if v_embedding_dim is None:
            v_embedding_dim = embedding_dim

        if node_dim != v_embedding_dim:
            self.v_output_layer = nn.Linear(node_dim, v_embedding_dim, bias=False)
        else:
            self.v_output_layer = nn.Identity()

    def radial_embedding(self,
                         edge_index: torch.LongTensor,
                         pos: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.LongTensor,
                edges_dict: dict
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.init_node_embedding(x)
        v = self.init_vector_embedding(v)
        x, v = self.zeroth_fc_block(x=x, v=v, batch=batch)

        if len(self.interaction_blocks) > 0:
            dist, rbf = self.radial_embedding(edges_dict['edge_index'], pos)
            for n, (convolution, vector_convolution, fc) in enumerate(
                    zip(self.interaction_blocks, self.vector_interaction_blocks, self.fc_blocks)):
                x = convolution(x, edges_dict['edge_index'], rbf)
                v = vector_convolution(v, edges_dict['edge_index'], rbf)
                x, v = fc(x=x, v=v, batch=batch)

        return self.output_layer(x), self.v_output_layer(v)



class ScalarGNN(torch.nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 node_dim: int,
                 fcs_per_gc: int,
                 message_dim: int,
                 embedding_dim: int,
                 num_convs: int,
                 num_radial: int,
                 num_input_classes=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 atom_type_embedding_dim: int = 5,
                 norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 override_cutoff: Optional[float] = None
                 ):
        super(ScalarGNN, self).__init__()

        self.max_num_neighbors = max_num_neighbors

        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        self.init_node_embedding = EmbeddingBlock(node_dim,
                                                  num_input_classes,
                                                  input_node_dim,
                                                  atom_type_embedding_dim)

        self.zeroth_fc_block = scalarMLP(layers=fcs_per_gc,
                                         filters=node_dim,
                                         input_dim=node_dim,
                                         output_dim=node_dim,
                                         activation=activation,
                                         norm=norm,
                                         dropout=dropout)

        self.interaction_blocks = torch.nn.ModuleList([
            MConv(
                message_dim=message_dim,
                node_dim=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
                activation_fn=activation)
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            scalarMLP(layers=fcs_per_gc,
                      filters=node_dim,
                      input_dim=node_dim,
                      output_dim=node_dim,
                      activation=activation,
                      norm=norm,
                      dropout=dropout)
            for _ in range(num_convs)
        ])

        if node_dim != embedding_dim:
            self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def radial_embedding(self,
                         edge_index,
                         pos: torch.Tensor,
                         dist: Optional[torch.Tensor] = None,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute elements for radial & spherical embeddings
        """
        if dist is None:
            i, j = edge_index  # i->j source-to-target
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self,
                z: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.LongTensor,
                edge_index: torch.LongTensor,
                dist: Optional[torch.Tensor] = None
                ) -> torch.Tensor:

        x = self.init_node_embedding(z)
        x = self.zeroth_fc_block(x=x, batch=batch)

        if len(self.interaction_blocks) > 0:
            dist, rbf = self.radial_embedding(edge_index, pos, dist)
            for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
                x = convolution(x, edge_index, rbf)
                x = fc(x, batch=batch)

        return self.output_layer(x)



# noinspection PyAttributeOutsideInit
class ScalarMoleculeGraphModel(nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_pos_to_node_dim: bool = False,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(ScalarMoleculeGraphModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_pos_to_node_dim = concat_pos_to_node_dim
        self.concat_mol_to_node_dim = concat_mol_to_node_dim

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats

        self.graph_net = ScalarGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = scalarMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                batch: torch.LongTensor,
                ptr: torch.LongTensor,
                mol_x: Union[torch.Tensor],
                num_graphs: int,
                edge_index: Optional[torch.LongTensor] = None,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False,
                force_edges_rebuild: bool = False,
                ) -> Tuple[torch.Tensor, Optional[dict]]:

        if edge_index is not None and not force_edges_rebuild:
            edge_index = edge_index
        else:
            # option to rebuild radial graph
            edges_dict = construct_radial_graph(
                pos,
                batch,
                ptr,
                self.convolution_cutoff,
                self.max_num_neighbors,
            )
            edge_index = edges_dict['edge_index']

        x = self.append_init_node_features(x, pos, ptr, mol_x)
        x = self.graph_net(x,
                           pos,
                           batch,
                           edge_index)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation
        x = self.global_pool(x,
                             batch,
                             dim_size=num_graphs)

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                x = torch.cat([x, self.mol_fc(mol_x)], dim=-1)
            gmlp_out = self.gnn_mlp(x)

            x = gmlp_out

        output = self.output_fc(x)

        extra_outputs = self.collect_extra_outputs(x,
                                                   pos,
                                                   batch,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              pos: torch.Tensor,
                              batch: torch.LongTensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self, x, pos, ptr, mol_x):
        if x.ndim == 1:
            x = x[:, None]

        # simply append node coordinates, PointNet style
        if self.concat_pos_to_node_dim:
            x = torch.cat((x, pos), dim=-1)

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x


class MoleculeScalarRegressor(BaseGraphModel):
    def __init__(self,
                 config: Namespace,
                 atom_features: list,
                 molecule_features: list,
                 node_standardization_tensor: Optional[torch.Tensor] = None,
                 graph_standardization_tensor: Optional[torch.Tensor] = None,
                 target_standardization_tensor: Optional[torch.Tensor] = None,
                 seed: int = 0
                 ):
        """
        wrapper for molecule model, with appropriate I/O
        """
        super(MoleculeScalarRegressor, self).__init__()
        torch.manual_seed(seed)
        # TODO save target mean and std inside this class
        self.get_data_stats(atom_features,
                            molecule_features,
                            node_standardization_tensor,
                            graph_standardization_tensor)

        if target_standardization_tensor is not None:
            self.register_buffer('target_mean', target_standardization_tensor[0])
            self.register_buffer('target_std', target_standardization_tensor[1])
        else:
            self.register_buffer('target_mean', torch.ones(1)[0])
            self.register_buffer('target_std', torch.ones(1)[0])

        self.model = ScalarMoleculeGraphModel(
            input_node_dim=self.n_atom_feats,
            num_mol_feats=self.n_mol_feats,
            output_dim=1,
            seed=seed,
            concat_mol_to_node_dim=True,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
        )

    # uses default forward method inherited from base class


class Scalarizer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 embedding_dim: int,
                 norm_mode: str,
                 act_func: str,
                 dropout: float = 0,
                 output_dim: int = None,
                 ):
        """
        Generate a learned invariant representation of dimension :math:`(k)` from a list of vectors of dimension :math:`v=(k x 3)`.

        Generate m vectors as a linear combination of the k vectors of v, take their normalized dot products with the components of v, concatenate to the norms of v, and linearly combine to the so-called scalarized representation of v.

        Args:
            hidden_dim (int): feature depth of input and output
            embedding_dim (int): number of vectors to use for dot-product projection
            norm_mode (str): type of normalization to use
            act_func (str): type of activation to use
            dropout (float): dropout probability
        """
        super(Scalarizer, self).__init__()
        if output_dim is None:
            output_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        reduced_dim = max(hidden_dim // 4, 1)
        #self.linear = nn.Linear(int(hidden_dim * (1 + embedding_dim)), hidden_dim, bias=True)
        self.embedding = nn.Linear(reduced_dim, embedding_dim, bias=False)
        self.dim_red = nn.Linear(hidden_dim, reduced_dim, bias=False)
        self.linear = nn.Linear(reduced_dim * 4, output_dim, bias=True)
        self.norm = Normalization(norm_mode, output_dim)
        self.activation = Activation(act_func, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                v: torch.Tensor
                ) -> torch.Tensor:
        eps = 1e-3
        v_red = self.dim_red(v)
        norm = torch.linalg.norm(v_red, dim=1)
        normed_v_red = v_red / (norm[:, None, :] + eps)

        directions = self.embedding(v_red)
        normed_directions = directions / (torch.linalg.norm(directions, dim=1)[:, None, :] + eps)

        projections = torch.einsum('nik,nij->njk', normed_v_red, normed_directions)

        v2 = torch.cat([norm, projections.flatten(1)], dim=1)

        return self.dropout(self.activation(self.norm(self.linear(v2))))


# noinspection PyAttributeOutsideInit
class Mo3ENet(BaseGraphModel):
    def __init__(self, seed, config,
                 num_atom_types: int,
                 atom_embedding_vector: torch.tensor,
                 radial_normalization: float,
                 infer_protons: bool,
                 protons_in_input: bool,
                 ):
        super(Mo3ENet, self).__init__()
        """
        3D o3 equivariant multi-type point cloud autoencoder model
        Mo3ENet
        """

        torch.manual_seed(seed)

        self.cartesian_dimension = 3
        self.num_classes = num_atom_types
        self.output_depth = self.num_classes + self.cartesian_dimension + 1
        self.num_decoder_nodes = config.decoder.num_nodes
        self.bottleneck_dim = config.bottleneck_dim
        self.decoder_type = config.decoder.model_type
        # todo add type distance scaling and num atom types and node weight temperature
        self.register_buffer('atom_embedding_vector', atom_embedding_vector)
        self.register_buffer('radial_normalization', torch.tensor(radial_normalization, dtype=torch.float32))
        self.register_buffer('protons_in_input', torch.tensor(protons_in_input, dtype=torch.bool))
        self.register_buffer('inferring_protons', torch.tensor(infer_protons, dtype=torch.bool))
        self.register_buffer('convolution_cutoff', config.encoder.graph.cutoff / self.radial_normalization)

        self.encoder = Mo3ENetEncoder(seed,
                                      config.encoder,
                                      config.bottleneck_dim,
                                      override_cutoff=self.convolution_cutoff)
        if self.decoder_type == 'mlp':
            self.decoder = Mo3ENetDecoder(seed,
                                          config.decoder,
                                          config.bottleneck_dim,
                                          self.output_depth, self.num_decoder_nodes)
        elif self.decoder_type == 'gnn':
            self.decoder = Mo3ENetGraphDecoder(config.decoder,
                                               config.bottleneck_dim,
                                               self.output_depth,
                                               self.num_decoder_nodes,
                                               )
        else:
            assert False, "Unknown decoder type" + str(self.decoder_type)
        self.scalarizer = Scalarizer(config.bottleneck_dim,
                                     self.cartesian_dimension,
                                     None, None, 0)

    def forward(self,
                data: CrystalData,
                return_latent: bool = False,
                return_dists: bool = False,
                ):
        encoding = self.encode(data)
        if torch.sum(torch.isnan(encoding)) != 0:
            assert False, "NaN values in encoding"
        decoding = self.decode(encoding)
        if torch.sum(torch.isnan(decoding)) != 0:
            assert False, "NaN values in decoding"
        if return_latent:
            return decoding, encoding
        else:
            return decoding

    def encode(self,
               data,
               override_centering: bool = False):
        # normalize radii
        if not override_centering:
            assert torch.linalg.norm(data.pos.mean(0)) < 1e-3, "Encoder trained only for centered molecules!"
        data.pos /= self.radial_normalization
        _, encoding = self.encoder(data)

        return encoding

    def decode(self, encoding):
        """encoding nx3xk"""
        s = self.scalarizer(encoding)
        if torch.sum(torch.isnan(s)) > 0:
            assert False, "NaN values in scalarized encoding"
        scalar_decoding, vector_decoding = self.decoder(s, v=encoding)

        '''combine vector and scalar features to n*nodes x m'''
        # de-normalize predicted node positions and rearrange to correct format
        # from n_graphs, x (num_nodes * scalar feats), v (num_nodes * scalar_feats)
        if self.decoder_type == 'mlp':
            decoding = torch.cat([
                vector_decoding.permute(0, 2, 1).reshape(len(vector_decoding) * self.num_decoder_nodes,
                                                         3) * self.radial_normalization,
                scalar_decoding.reshape(len(scalar_decoding) * self.num_decoder_nodes, self.output_depth - 3)],
                dim=-1)
        elif self.decoder_type == 'gnn':
            decoding = torch.cat(
                [
                    vector_decoding[:, :, 0] * self.radial_normalization,
                    scalar_decoding
                ],
                dim=1
            )
        else:
            assert False, "Unknown decoder type" + str(self.decoder_type)

        return decoding

    ''' equivariance testing
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    v = encoding.clone()

    'initialize rotations'
    rotations = torch.tensor(
        R.random(len(v)).as_matrix() *
        np.random.choice((-1, 1), replace=True, size=len(v))[:, None, None],
        dtype=torch.float,
        device=v.device)
    'rotate input'
    r_v = torch.einsum('ij, njk -> nik', rotations[0], v)

    'get output'
    s1, out1 = self.decoder(self.scalarizer(v), v=v)
    s2, out2 = self.decoder(self.scalarizer(r_v), v=r_v)

    'rotated output'
    r_out1 = torch.einsum('ij, njk -> nik', rotations[0], out1)

    print(torch.mean(torch.abs(r_out1 - out2) / out2.abs()))
    print(torch.mean(torch.abs(s1 - s2) / s2.abs()))

    '''

    def compile_self(self, dynamic=True, fullgraph=False):
        self.encoder = torch.compile(self.encoder, dynamic=dynamic, fullgraph=fullgraph)
        self.decoder = torch.compile(self.decoder, dynamic=dynamic, fullgraph=fullgraph)
        self.scalarizer = torch.compile(self.scalarizer, dynamic=dynamic, fullgraph=fullgraph)

    def check_embedding_quality(self, data,
                                sigma=0.35,
                                type_distance_scaling=2,
                                # todo next two should be properties of the model
                                node_weight_temperature=1,
                                num_atom_types=5,
                                visualize=False,
                                ):
        encoding = self.encode(data.clone())
        decoding = self.decode(encoding)

        data.x = self.atom_embedding_vector[data.x].flatten()
        decoded_data, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
            collate_decoded_data(data,
                                 decoding,
                                 self.num_decoder_nodes,
                                 node_weight_temperature,
                                 data.x.device))

        (nodewise_reconstruction_loss,  # todo adjust with new losses
         nodewise_type_loss,
         reconstruction_loss,
         self_likelihoods,
         ) = ae_reconstruction_loss(data,
                                    decoded_data,
                                    nodewise_weights,
                                    nodewise_weights_tensor,
                                    num_atom_types,
                                    type_distance_scaling,
                                    sigma)

        rmsds = torch.zeros(data.num_graphs)
        max_dists = torch.zeros_like(rmsds)
        tot_overlaps = torch.zeros_like(rmsds)
        match_successful = torch.zeros_like(rmsds)
        # for ind in range(data.num_graphs):
        #     rmsds[ind], max_dists[ind], tot_overlaps[ind], match_successful[ind], fig2 = scaffolded_decoder_clustering(
        #         ind,
        #         data,
        #         decoded_data,
        #         num_atom_types,
        #         return_fig=True)
        if visualize:
            for ind in range(data.num_graphs):
                swarm_vs_tgt_fig(data, decoded_data, num_atom_types, graph_ind=ind).show()

        return reconstruction_loss, rmsds, max_dists, tot_overlaps, match_successful


class Mo3ENetDecoder(nn.Module):
    def __init__(self, seed, config, bottleneck_dim, output_depth, num_nodes):
        super(Mo3ENetDecoder, self).__init__()
        self.model = vectorMLP(
            seed=seed,
            layers=config.fc.num_layers,
            filters=config.fc.hidden_dim,
            input_dim=bottleneck_dim,
            vector_input_dim=bottleneck_dim,
            vector_output_dim=num_nodes,
            output_dim=(output_depth - 3) * num_nodes,
            activation=config.activation,
            norm=config.fc.norm,
            dropout=config.fc.dropout,
            vector_norm=config.fc.vector_norm,
            ramp_depth=config.ramp_depth,
        )

    def forward(self, x, v):
        return self.model(x, v)


class Mo3ENetGraphDecoder(nn.Module):
    def __init__(self, config, bottleneck_dim, output_depth, num_nodes):
        super(Mo3ENetGraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = config.fc.hidden_dim
        self.model = VectorGNN(
            input_node_dim=config.fc.hidden_dim,
            node_dim=config.fc.hidden_dim,
            fcs_per_gc=1,
            message_dim=config.fc.hidden_dim // 4,
            embedding_dim=output_depth - 3,
            num_convs=config.fc.num_layers,
            num_radial=32,
            num_input_classes=101,
            cutoff=2,
            max_num_neighbors=32,
            envelope_exponent=5,
            activation='gelu',
            atom_type_embedding_dim=0,
            norm=('graph ' + config.fc.norm) if config.fc.norm is not None else None,
            vector_norm=('graph ' + config.fc.vector_norm) if config.fc.vector_norm is not None else None,
            dropout=config.fc.dropout,
            radial_embedding='gaussian',
            override_cutoff=None,
            v_embedding_dim=1,
            v_input_node_dim=config.fc.hidden_dim,
        )
        self.s_to_nodes = nn.Linear(bottleneck_dim, config.fc.hidden_dim * num_nodes, bias=False)
        self.v_to_nodes = nn.Linear(bottleneck_dim, config.fc.hidden_dim * num_nodes, bias=False)
        self.v_to_pos = nn.Linear(bottleneck_dim, num_nodes, bias=False)

    def forward(self, x, v):
        num_graphs = len(x)

        # all combinations of edges within each graph
        edges = []
        edges_i = torch.combinations(torch.arange(self.num_nodes), r=2, with_replacement=False).to(x.device)

        for ind in range(num_graphs):
            batch_ind = ind * self.num_nodes
            edges.append(
                batch_ind + torch.cat([edges_i, torch.fliplr(edges_i)], dim=0)
            )
        batch = torch.arange(num_graphs, device=x.device).repeat_interleave(self.num_nodes)
        edges = torch.cat(edges, dim=0)
        edges_dict = {'edge_index': edges.T}

        x = self.s_to_nodes(x).reshape(num_graphs * self.num_nodes, self.hidden_dim)
        directions = self.v_to_pos(v).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, 3, 1)[..., 0]
        pos = directions / (1e-4 + torch.linalg.norm(directions, dim=1))[:, None]
        v = self.v_to_nodes(v).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, self.hidden_dim, 3).permute(0, 2,
                                                                                                                 1)

        return self.model(x, v, pos, batch, edges_dict)

    ''' equivariance test
    def v_to_node(v, num_graphs):
        v2 = self.v_to_nodes(v).reshape(num_graphs, 3, self.hidden_dim, self.num_nodes)
        v2 = v2.permute(0, 3, 1, 2).flatten(0, 1)
        return v2

    from scipy.spatial.transform import Rotation as R
    import numpy as np

    'initialize rotations'
    rotations = torch.tensor(
        R.random(num_graphs).as_matrix() *
        np.random.choice((-1, 1), replace=True, size=num_graphs)[:, None, None],
        dtype=torch.float,
        device=x.device)

    'rotate input'
    r_v = torch.einsum('ij, njk -> nik', rotations[0], v)

    'get output'
    out1 = v_to_node(v, num_graphs)
    out2 = v_to_node(r_v, num_graphs)

    'rotated output'
    r_out1 = torch.einsum('ij, njk -> nik', rotations[0], out1)

    print(torch.mean(torch.abs(r_out1 - out2)/out2.abs()))

    import plotly.graph_objects as go

    fig = go.Figure(go.Histogram(x=((out2 - r_out1)/out2.abs()).flatten().abs().log10().cpu().detach().numpy(), nbinsx=100)).show()

    def v_to_node(v, num_graphs):
    v2 = self.v_to_nodes(v).reshape(num_graphs, 3, self.hidden_dim, self.num_nodes)
    v2 = v2.permute(0, 3, 1, 2).flatten(0, 1)
    return v2

    # ---- graph model --- 
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    'initialize rotations'
    rotations = torch.tensor(
        R.random(num_graphs).as_matrix() *
        np.random.choice((-1, 1), replace=True, size=num_graphs)[:, None, None],
        dtype=torch.float,
        device=x.device)

    'rotate input'
    r_v = torch.einsum('ij, njk -> nik', rotations[0], v)
    r_pos = torch.einsum('ij, nj -> ni', rotations[0], pos)

    'get output'
    s1, out1 = self.model(x, v, pos, batch, edges_dict)
    s2, out2 = self.model(x, r_v, r_pos, batch, edges_dict)

    'rotated output'
    r_out1 = torch.einsum('ij, njk -> nik', rotations[0], out1)

    print(torch.mean(torch.abs(r_out1 - out2)/out2.abs()))

    # final equivariance test
    if not hasattr(self, 'v0'):
    self.x0 = x.clone()
    self.v0 = v.clone()

from scipy.spatial.transform import Rotation as R
import numpy as np

num_graphs = len(x)

# all combinations of edges within each graph
edges = []
edges_i = torch.combinations(torch.arange(self.num_nodes), r=2, with_replacement=False).to(x.device)

for ind in range(num_graphs):
    batch_ind = ind * self.num_nodes
    edges.append(
        batch_ind + torch.cat([edges_i, torch.fliplr(edges_i)], dim=0)
    )
edges = torch.cat(edges, dim=0)
edges_dict = {'edge_index': edges.T}
batch = torch.arange(num_graphs, device=x.device).repeat_interleave(self.num_nodes)

'initialize rotations'
rotations = torch.tensor(
    R.random(num_graphs).as_matrix() *
    np.random.choice((-1, 1), replace=True, size=num_graphs)[:, None, None],
    dtype=torch.float,
    device=x.device)

x = self.x0.clone()
v = self.v0.clone()

def rotate_object(rotations, thing, batch, num_graphs):
    return torch.cat(
    [torch.einsum('ij, njk->nik', rotations[ind], thing[batch == ind])
     for ind in range(num_graphs)])

rv = rotate_object(rotations, v, torch.arange(num_graphs, device=x.device), num_graphs)

xf = self.s_to_nodes(x).reshape(num_graphs * self.num_nodes, self.hidden_dim)
pos = self.v_to_pos(v).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, 3, 1)[..., 0]
vf = self.v_to_nodes(v).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, self.hidden_dim, 3).permute(0, 2, 1)
rpos = self.v_to_pos(rv).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, 3, 1)[..., 0]

posr = rotate_object(rotations, pos[:, :, None], batch, num_graphs)[..., 0]

vfr = rotate_object(rotations, vf, batch, num_graphs)
rvf = self.v_to_nodes(rv).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, self.hidden_dim, 3).permute(0, 2, 1)

xo, yo = self.model(xf, vf, pos, batch, edges_dict)
rxo, ryo = self.model(xf, rvf, rpos, batch, edges_dict)

yor = rotate_object(rotations, yo, batch, num_graphs)

print(((vfr-rvf).abs()/rvf.abs()).mean())
print(((yor-ryo).abs()/ryo.abs()).mean())
print(((rpos-posr).abs()/rpos.abs()).mean())
    '''


class Mo3ENetEncoder(nn.Module):
    def __init__(self, seed, config, bottleneck_dim, override_cutoff=None):
        super(Mo3ENetEncoder, self).__init__()
        self.model = VectorMoleculeGraphModel(
            input_node_dim=1,
            num_mol_feats=0,
            output_dim=bottleneck_dim,
            seed=seed,
            concat_pos_to_node_dim=True,
            concat_mol_to_node_dim=False,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
            override_cutoff=override_cutoff,
        )

    def forward(self, data):
        return self.model(data.x,
                          data.pos,
                          data.batch,
                          data.ptr,
                          num_graphs=data.num_graphs,
                          )

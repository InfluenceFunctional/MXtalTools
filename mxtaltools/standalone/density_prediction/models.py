import sys
from argparse import Namespace
from math import pi as PI
from typing import Optional
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric import nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax

from mxtaltools.standalone.density_prediction.CrystalData import CrystalData
from mxtaltools.standalone.density_prediction.utils import (
    VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD)


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

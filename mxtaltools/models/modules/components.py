import sys
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric import nn as gnn

from mxtaltools.models.functions.asymmetric_radius_graph import asymmetric_radius_graph
from mxtaltools.models.modules.vector_LayerNorm import VectorLayerNorm


class Scalarizer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 embedding_dim: int,
                 norm_mode: str,
                 act_func: str,
                 dropout: float = 0
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
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.linear = nn.Linear(int(hidden_dim * (1 + embedding_dim)), hidden_dim, bias=True)
        self.norm = Normalization(norm_mode, hidden_dim)
        self.activation = Activation(act_func, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                v: torch.Tensor
                ) -> torch.Tensor:
        norm = torch.linalg.norm(v, dim=1)
        normed_v = v / (norm[:, None, :] + 1e-5)

        directions = self.embedding(v)
        normed_directions = directions / (torch.linalg.norm(directions, dim=1, keepdim=True) + 1e-5)

        projections = torch.einsum('nik,nij->njk', normed_v, normed_directions)

        v2 = torch.cat([norm, projections.reshape(v.shape[0], self.embedding_dim * self.hidden_dim)], dim=1)

        return self.dropout(self.activation(self.norm(self.linear(v2))))


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

    '''
    # tests # todo write equivariance check
    # assert projection.max() <= 1
    # assert projeciton.min() >= -1
    # assert torch.einsum('nik,ni->nk', activated_output, normed_direction).min() >= 1e-3, "Vector Activation Failed"

    import numpy as np
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_histogram(x=projection.flatten(),nbinsx=100)
    fig.add_histogram(x=projection2.flatten(), nbinsx=100)
    fig.show(renderer='browser')
    '''


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
        elif norm == 'graph vector layer':
            self.norm = VectorLayerNorm(filters, mode='graph')
        elif norm == 'vector layer':
            self.norm = VectorLayerNorm(filters, mode='node')
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
            elif activation_func.lower() == 'kernel':  # rather expensive
                self.activation = KernelActivation(n_basis=10, span=4, filters=filters)
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


class KernelActivation(nn.Module):
    r"""
    Function for learning an activation function for every node in a given layer, as a linear combination of basis functions over a given span.
    Uses nn.Conv1d groups option for efficient evaluation.

    Args:
        n_basis (int): number of basis functions
        span (float): span over which to define localized basis functions
        filters (int): feature depth of inputs to be activated
    """

    def __init__(self,
                 n_basis: int,
                 span: float,
                 filters: int):
        super(KernelActivation, self).__init__()

        self.channels, self.n_basis = filters, n_basis
        # define the space of basis functions
        self.register_buffer('dict',
                             torch.linspace(-span, span, n_basis))  # positive and negative values for Dirichlet Kernel
        gamma = 1 / (6 * (self.dict[-1] - self.dict[
            -2]) ** 2)  # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma', torch.ones(1) * gamma)  #

        # self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(filters * n_basis, filters, kernel_size=(1, 1), groups=int(filters), bias=False)

        # nn.init.normal(self.linear.weight.data, std=0.1)

    def kernel(self,
               x: torch.Tensor
               ) -> torch.Tensor:
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x) == 2:
            x = x.reshape(2, self.channels, 1)

        return torch.exp(-self.gamma * (x - self.dict) ** 2)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1)  # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3],
                      x.shape[4])  # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1)  # apply linear coefficients and sum

        return x


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
        conditioning_dim (int): dimension of optional conditioning vector for initial layer
        conditioning_mode: 'concat_to_first' conditioning is done by concatenating conditioning vector to first layer input. There is currently no other option.
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
                 conditioning_dim: int = 0,
                 norm: 'str' = None,
                 norm_after_linear: bool = True,
                 bias: bool = True,
                 conditioning_mode: str = 'concat_to_first',
                 ramp_depth: bool = False,
                 ):
        super(scalarMLP, self).__init__()
        # initialize constants and layers
        self.n_layers = layers
        self.conditioning_mode = conditioning_mode  # todo write a proper all_layer conditioning mode
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim

        self.input_dim = input_dim + conditioning_dim
        self.norm_mode = norm
        self.dropout_p = dropout
        self.activation = activation
        self.bias = bias
        self.norm_after_linear = norm_after_linear

        self.ramp_depth = ramp_depth

        torch.manual_seed(seed)
        self.init_scalar_filters(filters, layers)
        self.init_scalar_transforms()

    def init_scalar_filters(self, filters, layers):
        if isinstance(filters, list):
            self.n_filters = filters
            residue_filters = [self.input_dim] + self.n_filters

        elif self.ramp_depth:  # smoothly ramp feature depth across layers
            # linear scaling
            # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
            # log scaling for consistent growth ratio
            p = np.log(self.output_dim) / np.log(self.input_dim)
            self.n_filters = [int(self.input_dim ** (1 + (p - 1) * (i / (self.n_layers)))) for i in
                              range(self.n_layers)]
            residue_filters = [self.input_dim] + self.n_filters
            self.same_depth = False
        else:
            self.n_filters = [filters for _ in range(layers)]

        if self.n_filters.count(self.n_filters[0]) != len(
                self.n_filters):  # if they are not all the same, we need residue adjustments
            self.same_depth = False
            self.residue_adjust = torch.nn.ModuleList([
                nn.Linear(residue_filters[i], residue_filters[i + 1], bias=False)
                for i in range(self.n_layers)
            ])
        else:
            self.same_depth = True

    def init_scalar_transforms(self):
        """scalar MLP layers"""

        '''input layer'''
        if self.input_dim != self.n_filters[0]:
            self.init_layer = nn.Linear(self.input_dim, self.n_filters[0])  # set appropriate sizing
        else:
            self.init_layer = nn.Identity()

        '''working layers'''
        self.fc_layers = torch.nn.ModuleList([
            nn.Linear(self.n_filters[i],
                      self.n_filters[i], bias=self.bias)
            for i in range(self.n_layers)
        ])
        self.fc_activations = torch.nn.ModuleList([
            Activation(self.activation, self.n_filters[i])
            for i in range(self.n_layers)
        ])
        if self.norm_after_linear:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode, self.n_filters[i])
                for i in range(self.n_layers)
            ])
        else:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode,
                              self.n_filters[i]
                              )
                for i in range(self.n_layers)
            ])
        self.fc_dropouts = torch.nn.ModuleList([
            nn.Dropout(p=self.dropout_p)
            for _ in range(self.n_layers)
        ])

        '''output layer'''
        if self.output_dim != self.n_filters[-1]:
            self.output_layer = nn.Linear(self.n_filters[-1], self.output_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                conditions: Optional[torch.Tensor] = None,
                return_latent: bool = False,
                batch: Optional[torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if conditions is not None:
            x = torch.cat((x, conditions), dim=-1)

        'initialize to correct feature dimension'
        x = self.init_layer(x)

        for i, (norm, linear, activation, dropout) in enumerate(
                zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts)):

            'get residue'
            if self.same_depth:
                x = x.clone()
            else:
                x = self.residue_adjust[i](x)

            'linear layer'
            if self.norm_after_linear:
                x = x + dropout(activation(norm(linear(x), batch=batch)))
            else:
                x = x + dropout(activation(linear(norm(x, batch=batch))))

        if return_latent:
            return self.output_layer(x), x
        else:
            return self.output_layer(x)


# noinspection PyAttributeOutsideInit
class vectorMLP(nn.Module):  # todo simplify and smooth out +1's and other custom methods for a general depth controller
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
                 conditioning_dim: int = 0,
                 norm: str = None,
                 norm_after_linear: bool = True,
                 bias: bool = True,
                 vector_norm: str = None,
                 conditioning_mode: str = 'concat_to_first',
                 ramp_depth: bool = False,
                 v_to_s_combination: str = 'concatenate'):
        super(vectorMLP, self).__init__()
        # initialize constants and layers
        self.n_layers = layers
        self.conditioning_mode = conditioning_mode
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim
        self.v_output_dim = vector_output_dim
        self.v_input_dim = vector_input_dim
        self.input_dim = input_dim + conditioning_dim
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

        self.init_scalar_filters(filters, layers)
        self.init_vector_filters(filters, layers)

        self.init_scalar_transforms()
        self.init_vector_transforms()

    def init_scalar_filters(self, filters, layers):
        if isinstance(filters, list):
            self.n_filters = filters
            residue_filters = [self.input_dim] + self.n_filters

        elif self.ramp_depth:  # smoothly ramp feature depth across layers
            # linear scaling
            # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
            # log scaling for consistent growth ratio
            p = np.log(self.output_dim) / np.log(self.input_dim)
            self.n_filters = [int(self.input_dim ** (1 + (p - 1) * (i / (self.n_layers)))) for i in
                              range(self.n_layers)]
            residue_filters = [self.input_dim] + self.n_filters
            self.same_depth = False
        else:
            self.n_filters = [filters for _ in range(layers)]
            self.same_depth = True

        if self.n_filters.count(self.n_filters[0]) != len(
                self.n_filters):  # if they are not all the same, we need residue adjustments
            self.same_depth = False
            self.residue_adjust = torch.nn.ModuleList([
                nn.Linear(residue_filters[i], residue_filters[i + 1], bias=False)
                for i in range(self.n_layers)
            ])
        else:
            self.same_depth = True

    def init_vector_filters(self, filters, layers):
        if isinstance(filters, list):
            self.v_n_filters = filters
            residue_filters = [self.v_input_dim] + self.v_n_filters

        elif self.ramp_depth:  # smoothly ramp feature depth across layers
            # linear scaling
            # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
            # exp scaling for consistent growth ratio
            p = np.log(self.v_output_dim) / np.log(self.input_dim)
            self.v_n_filters = [int(self.v_input_dim ** (1 + (p - 1) * (i / (self.n_layers)))) for i in
                                range(self.n_layers)]
            residue_filters = [self.v_input_dim] + self.v_n_filters
        else:
            self.v_n_filters = [filters for _ in range(layers)]

        # if they are not all the same, we need residue adjustments
        if self.n_filters.count(self.n_filters[0]) != len(self.n_filters):
            self.v_same_depth = False
            residue_filters[0] -= self.conditioning_dim
            self.v_residue_adjust = torch.nn.ModuleList([
                nn.Linear(residue_filters[i], residue_filters[i + 1], bias=False)
                for i in range(self.n_layers)
            ])
        else:
            self.v_same_depth = True

    def init_scalar_transforms(self):
        """scalar MLP layers"""

        '''input layer'''
        if self.input_dim != self.n_filters[0]:
            self.init_layer = nn.Linear(self.input_dim, self.n_filters[0])  # set appropriate sizing
        else:
            self.init_layer = nn.Identity()

        '''working layers'''
        self.fc_layers = torch.nn.ModuleList([
            nn.Linear(self.n_filters[i] + self.v_n_filters[i] if self.v_to_s_combination == 'concatenate' else 0,
                      self.n_filters[i], bias=self.bias)
            for i in range(self.n_layers)
        ])
        self.fc_activations = torch.nn.ModuleList([
            Activation(self.activation, self.n_filters[i])
            for i in range(self.n_layers)
        ])
        if self.norm_after_linear:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode, self.n_filters[i])
                for i in range(self.n_layers)
            ])
        else:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode,
                              self.n_filters[i] + self.v_n_filters[i] if self.v_to_s_combination == 'concatenate' else 0
                              )
                for i in range(self.n_layers)
            ])
        self.fc_dropouts = torch.nn.ModuleList([
            nn.Dropout(p=self.dropout_p)
            for _ in range(self.n_layers)
        ])

        '''output layer'''
        if self.output_dim != self.n_filters[-1]:
            self.output_layer = nn.Linear(self.n_filters[-1], self.output_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def init_vector_transforms(self):
        """vector MLP layers"""
        '''input layer'''
        if self.v_input_dim != self.n_filters[0]:
            self.v_init_layer = nn.Linear(self.v_input_dim - self.conditioning_dim, self.v_n_filters[0], bias=False)
        else:
            self.v_init_layer = nn.Identity()

        '''working layers'''
        self.v_fc_layers = torch.nn.ModuleList([
            nn.Linear(self.v_n_filters[i], self.v_n_filters[i], bias=False)
            for i in range(self.n_layers)
        ])
        self.s_to_v_gating_layers = torch.nn.ModuleList([
            nn.Linear(self.n_filters[i], self.v_n_filters[i], bias=False)
            for i in range(self.n_layers)
        ])
        self.s_to_v_activations = torch.nn.ModuleList(
            [  # use tanh as gating function rather than standard activation which is unbound
                Activation(self.activation, self.v_n_filters[i])
                # positive outputs only to maintain equivariance (no vectors flipped)
                for i in range(self.n_layers)
            ])
        self.v_fc_norms = torch.nn.ModuleList([
            Normalization(self.v_norm_mode, self.v_n_filters[i])
            for i in range(self.n_layers)
        ])
        self.vector_to_scalar = torch.nn.ModuleList([
            Scalarizer(self.v_n_filters[i], 3, self.norm_mode, self.activation, self.dropout_p)
            for i in range(self.n_layers)
        ])
        self.scalar_to_vector_norm = torch.nn.ModuleList([
            Normalization(self.norm_mode, self.v_n_filters[i])
            for i in range(self.n_layers)
        ])
        self.vector_activation = torch.nn.ModuleList([
            VectorActivation(self.v_n_filters[i], self.activation)
            for i in range(self.n_layers)
        ])

        '''output layer'''
        if self.v_output_dim != self.n_filters[-1]:
            self.v_output_layer = nn.Linear(self.v_n_filters[-1], self.v_output_dim, bias=False)
        else:
            self.v_output_layer = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                conditions: Optional[torch.Tensor] = None,
                return_latent: bool = False,
                batch: Optional[torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if conditions is not None:
            x = torch.cat((x, conditions), dim=-1)

        # get the right feature depth
        x = self.init_layer(x)
        v = self.v_init_layer(v)

        for i, (norm, linear, activation, dropout, v_norm, v_linear, v_act, s2v_norm, s2v_linear, s2v_act,
                v2s_linear) in enumerate(
                zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts,
                    self.v_fc_norms, self.v_fc_layers, self.vector_activation,
                    self.scalar_to_vector_norm, self.s_to_v_gating_layers, self.s_to_v_activations,
                    self.vector_to_scalar)):
            res_x, res_v = self.get_residues(i, x, v)

            'scalar forward'
            if self.v_to_s_combination == 'concatenate':
                # concatenate vector lengths to scalar values
                x = torch.cat([res_x, v2s_linear(res_v)], dim=-1)
            elif self.v_to_s_combination == 'sum':
                x = res_x + v2s_linear(res_v)
            else:
                assert False, f'{self.v_to_s_combination} not implemented'

            if self.norm_after_linear:
                x = res_x + dropout(activation(norm(linear(x), batch=batch)))
            else:
                x = res_x + dropout(activation(linear(norm(x, batch=batch))))

            'vector forward'
            gating_factor = s2v_act(s2v_norm(s2v_linear(res_x))[:, None, :])
            vector_mix = v_norm(v_linear(res_v), batch=batch)
            vector_mix = v_act(vector_mix)

            # A(FC(x)) * FC(N(v))   # rescaling factor keeps norm from exploding
            v = (res_v + gating_factor * vector_mix) / self.vector_addition_rescaling_factor

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

# old - deprecated
# class GlobalAggregation(nn.Module):  # TODO upgrade/replace with new PyG aggregation module
#     r"""
#     Wrapper for several types of global aggregation functions
#
#     Args:
#         agg_func (str): aggregation function
#         filters (int): feature depth of input/output
#     """
#
#     def __init__(self,
#                  agg_func: str,
#                  filters: int):
#         super(GlobalAggregation, self).__init__()
#         self.agg_func = agg_func
#         if agg_func == 'mean':
#             self.agg = gnn.global_mean_pool
#         elif agg_func == 'sum':
#             self.agg = gnn.global_add_pool
#         elif agg_func == 'max':
#             self.agg = gnn.global_max_pool
#         elif agg_func == 'attention':
#             self.agg = gnn.GlobalAttention(
#                 nn.Sequential(nn.Linear(filters, filters), nn.LeakyReLU(), nn.Linear(filters, 1)))
#         elif agg_func == 'set2set':
#             self.agg = gnn.Set2Set(in_channels=filters, processing_steps=4)
#             self.agg_fc = nn.Linear(filters * 2, filters)  # condense to correct number of filters
#         elif agg_func == 'simple combo':
#             self.agg_list1 = [gnn.global_max_pool, gnn.global_mean_pool,
#                               gnn.global_add_pool]  # simple aggregation functions
#             self.agg_fc = EMLP(
#                 layers=1,
#                 filters=filters,
#                 input_dim=filters * (len(self.agg_list1)),
#                 output_dim=filters,
#                 norm=None,
#                 dropout=0)  # condense to correct number of filters
#         elif agg_func == 'mean sum':
#             pass
#         elif agg_func == 'combo':
#             self.agg_list1 = [gnn.global_max_pool, gnn.global_mean_pool,
#                               gnn.global_add_pool]  # simple aggregation functions
#             self.agg_list2 = nn.ModuleList([gnn.GlobalAttention(
#                 EMLP(input_dim=filters,
#                      output_dim=1,
#                      layers=1,
#                      filters=filters,
#                      activation='leaky relu',
#                      norm=None),
#             )])  # aggregation functions requiring parameters
#             self.agg_fc = EMLP(
#                 layers=1,
#                 filters=filters,
#                 input_dim=filters * (len(self.agg_list1) + 1),
#                 output_dim=filters,
#                 norm=None,
#                 dropout=0)  # condense to correct number of filters
#         elif agg_func == 'molwise':
#             self.agg = gnn.pool.max_pool_x
#         elif agg_func is None:
#             self.agg = nn.Identity()
#
#         if agg_func == 'equivariant max':
#             print("WARNING Equivariant max pooling is mostly but not 100% equivariant, e.g., in degenerate cases")
#
#     def forward(self, x: torch.Tensor,
#                 batch: torch.Tensor,
#                 cluster: torch.Tensor = None,
#                 output_dim: int = None,
#                 v: Optional[torch.Tensor] = None):
#         if self.agg_func == 'set2set':
#             x = self.agg(x, batch, size=output_dim)
#             return self.agg_fc(x)
#         elif self.agg_func == 'combo':
#             output1 = [agg(x, batch, size=output_dim) for agg in self.agg_list1]
#             output2 = [agg(x, batch, size=output_dim) for agg in self.agg_list2]
#             # output3 = [agg(x, batch, 3, size = output_dim) for agg in self.agg_list3]
#             return self.agg_fc(torch.cat((output1 + output2), dim=1))
#         elif self.agg_func == 'simple combo':
#             output1 = [agg(x, batch, size=output_dim) for agg in self.agg_list1]
#             return self.agg_fc(torch.cat(output1, dim=1))
#         elif self.agg_func is None:
#             return x  # do nothing
#         elif self.agg_func == 'molwise':
#             if cluster.ndim > 1:
#                 cluster = cluster[0,
#                           :]  # usually a single cluster - if in the future a batch of clusters, reindex molecules accordingly
#             return self.agg(cluster=cluster, batch=batch, x=x)[0]
#         elif self.agg_func == 'mean sum':
#             return (scatter(x, batch, dim_size=output_dim, dim=0, reduce='mean') +
#                     scatter(x, batch, dim_size=output_dim, dim=0, reduce='sum'))
#         elif self.agg_func == 'softmax':
#             weights = scatter_softmax(x, batch, dim=0)
#             return scatter(weights * x, batch, dim_size=output_dim, dim=0, reduce='sum')
#         elif self.agg_func == 'equivariant softmax':
#             weights = scatter_softmax(torch.linalg.norm(v, dim=1), batch, dim=0)
#             return (scatter(weights * x, batch, dim_size=output_dim, dim=0, reduce='sum'),
#                     scatter(weights[:, None, :] * v, batch, dim=0, dim_size=output_dim, reduce='sum'))
#         elif self.agg_func == 'equivariant combo':
#             scalar_agg, alpha = self.agg(x, batch, dim_size=output_dim, return_alpha=True)
#             agg1 = scatter(alpha[:, 0, None, None] * v, batch, dim=0, dim_size=output_dim,
#                            reduce='sum')  # use the same attention weights for vector aggregation
#             agg2 = scatter(v, batch, dim_size=output_dim, dim=0, reduce='mean')
#             agg3 = scatter(v, batch, dim_size=output_dim, dim=0, reduce='sum')
#
#             return scalar_agg, self.agg_fc(
#                 self.agg_norm(
#                     torch.cat([agg1, agg2, agg3], dim=-1),
#                     batch=torch.arange(len(agg1), device=agg1.device, dtype=torch.long)))  # return num_graphsx3xk
#         elif self.agg_func == 'equivariant attention':
#             scalar_agg, alpha = self.agg(x, batch, dim_size=output_dim, return_alpha=True)
#             vector_agg = scatter(alpha[:, 0, None, None] * v, batch, dim=0, dim_size=output_dim,
#                                  reduce='sum')  # use the same attention weights for vector aggregation
#             return scalar_agg, vector_agg
#         else:
#             return self.agg(x, batch, size=output_dim)


# old - replaced by separate scalar and vector models
# # noinspection PyAttributeOutsideInit
# class EMLP(nn.Module):
#     r"""
#     Flexible multi-layer perceptron module, with several options.
#
#     Features an equivariance option which adds a second feature track for vectors. Vector operations are equivariant w.r.t., O(3) operations on the inputs.
#
#     Args:
#         layers (int): number of fully-connected layers
#         filters (int): feature depth with FC layers
#         input_dim (int): feature depth of inputs
#         output_dim (int): feature depth of outputs
#         activation (str): activation function
#         seed (int): random seed
#         dropout (float): dropout probability
#         conditioning_dim (int): dimension of optional conditioning vector for initial layer
#         conditioning_mode: 'concat_to_first' conditioning is done by concatenating conditioning vector to first layer input. There is currently no other option.
#         add_vector_channel (bool): adds a second track for vector feature inputs and outputs, :math:`(batch, 3, k)`, which transform equivariantly
#         vector_output_dim (int): dimension of vector outputs
#         vector_norm (bool): whether to apply normalization to vector norms. Only graph layernorm and layernorm implemented.
#         ramp_depth (bool): whether to ramp the feature depth exponentially from input_dim to output_dim through the network
#     """
#
#     def __init__(self,
#                  layers: int,
#                  filters: int,
#                  input_dim: int,
#                  output_dim: int,
#                  activation: str = 'gelu',
#                  seed: int = 0,
#                  dropout: float = 0,
#                  conditioning_dim: int = 0,
#                  norm: 'str' = None,
#                  norm_after_linear: bool = True,
#                  bias: bool = True,
#                  conditioning_mode: str = 'concat_to_first',
#                  add_vector_channel: bool = False,
#                  vector_output_dim: Optional[int] = None,
#                  vector_norm: Optional[bool] = None,
#                  ramp_depth: bool = False,
#                  vector_input_dim: Optional[int] = None,
#                  v_to_s_combination: Optional[str] = 'concatenate'):
#         super(EMLP, self).__init__()
#         # initialize constants and layers
#         self.n_layers = layers
#         self.conditioning_mode = conditioning_mode  # todo write a proper all_layer conditioning mode
#         self.conditioning_dim = conditioning_dim
#         self.output_dim = output_dim
#         self.v_output_dim = vector_output_dim if vector_output_dim is not None else output_dim
#         self.v_input_dim = vector_input_dim if vector_input_dim is not None else input_dim
#         self.input_dim = input_dim + conditioning_dim
#         self.norm_mode = norm
#         self.dropout_p = dropout
#         self.activation = activation
#         self.bias = bias
#         self.norm_after_linear = norm_after_linear
#         self.equivariant = add_vector_channel
#         self.v_norm_mode = vector_norm
#         self.ramp_depth = ramp_depth
#         self.v_to_s_combination = v_to_s_combination
#         if self.v_norm_mode:
#             assert self.equivariant
#
#         torch.manual_seed(seed)
#         # addition of two normally distributed 3-vectors increases the norm by roughly this factor on average
#         # divide this out to combat vector elongation & poor gradient flow
#         self.vector_addition_rescaling_factor = 1.6
#         self.init_filters(filters, layers)
#         self.init_scalar_transforms()
#         if add_vector_channel:
#             self.init_vector_transforms()
#
#     def init_filters(self, filters, layers):
#         if isinstance(filters, list):
#             self.n_filters = filters
#             residue_filters = [self.input_dim] + self.n_filters
#
#         elif self.ramp_depth:  # smoothly ramp feature depth across layers
#             # linear scaling
#             # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
#             # log scaling for consistent growth ratio
#             p = np.log(self.output_dim) / np.log(self.input_dim)
#             self.n_filters = [int(self.input_dim ** (1 + (p - 1) * (i / (self.n_layers)))) for i in
#                               range(self.n_layers)]
#             residue_filters = [self.input_dim] + self.n_filters
#             self.same_depth = False
#         else:
#             self.n_filters = [filters for _ in range(layers)]
#
#         if self.n_filters.count(self.n_filters[0]) != len(
#                 self.n_filters):  # if they are not all the same, we need residue adjustments
#             self.same_depth = False
#             self.residue_adjust = torch.nn.ModuleList([
#                 nn.Linear(residue_filters[i], residue_filters[i + 1], bias=False)
#                 for i in range(self.n_layers)
#             ])
#         else:
#             self.same_depth = True
#
#         if self.equivariant:
#             if isinstance(filters, list):
#
#                 self.v_n_filters = filters
#                 residue_filters = [self.v_input_dim] + self.v_n_filters
#
#             elif self.ramp_depth:  # smoothly ramp feature depth across layers
#                 # linear scaling
#                 # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
#
#                 # exp scaling for consistent growth ratio
#                 p = np.log(self.v_output_dim) / np.log(self.input_dim)
#                 self.v_n_filters = [int(self.v_input_dim ** (1 + (p - 1) * (i / (self.n_layers)))) for i in
#                                     range(self.n_layers)]
#                 residue_filters = [self.v_input_dim] + self.v_n_filters
#             else:
#                 self.v_n_filters = [filters for _ in range(layers)]
#
#             if self.n_filters.count(self.n_filters[0]) != len(
#                     self.n_filters):  # if they are not all the same, we need residue adjustments
#                 residue_filters[0] -= self.conditioning_dim
#                 self.v_residue_adjust = torch.nn.ModuleList([
#                     nn.Linear(residue_filters[i], residue_filters[i + 1], bias=False)
#                     for i in range(self.n_layers)
#                 ])
#
#     def init_scalar_transforms(self):
#         """scalar MLP layers"""
#
#         '''input layer'''
#         if self.input_dim != self.n_filters[0]:
#             self.init_layer = nn.Linear(self.input_dim, self.n_filters[0])  # set appropriate sizing
#         else:
#             self.init_layer = nn.Identity()
#
#         '''working layers'''
#         self.fc_layers = torch.nn.ModuleList([
#             nn.Linear(self.n_filters[i] + (self.v_n_filters[i] if
#                                            (self.equivariant and self.v_to_s_combination == 'concatenate')
#                                            else 0),
#                       self.n_filters[i], bias=self.bias)
#             for i in range(self.n_layers)
#         ])
#         self.fc_activations = torch.nn.ModuleList([
#             Activation(self.activation, self.n_filters[i])
#             for i in range(self.n_layers)
#         ])
#         if self.norm_after_linear:
#             self.fc_norms = torch.nn.ModuleList([
#                 Normalization(self.norm_mode, self.n_filters[i])
#                 for i in range(self.n_layers)
#             ])
#         else:
#             self.fc_norms = torch.nn.ModuleList([
#                 Normalization(self.norm_mode,
#                               self.n_filters[i] + (self.v_n_filters[i] if
#                                                    (self.equivariant and self.v_to_s_combination == 'concatenate')
#                                                    else 0)
#                               )
#                 for i in range(self.n_layers)
#             ])
#         self.fc_dropouts = torch.nn.ModuleList([
#             nn.Dropout(p=self.dropout_p)
#             for _ in range(self.n_layers)
#         ])
#
#         '''output layer'''
#         if self.output_dim != self.n_filters[-1]:
#             self.output_layer = nn.Linear(self.n_filters[-1], self.output_dim, bias=False)
#         else:
#             self.output_layer = nn.Identity()
#
#     def init_vector_transforms(self):
#         """vector MLP layers"""
#         '''input layer'''
#         if self.v_input_dim != self.n_filters[0]:
#             self.v_init_layer = nn.Linear(self.v_input_dim - self.conditioning_dim, self.v_n_filters[0], bias=False)
#         else:
#             self.v_init_layer = nn.Identity()
#
#         '''working layers'''
#         self.v_fc_layers = torch.nn.ModuleList([
#             nn.Linear(self.v_n_filters[i], self.v_n_filters[i], bias=False)
#             for i in range(self.n_layers)
#         ])
#         self.s_to_v_gating_layers = torch.nn.ModuleList([
#             nn.Linear(self.n_filters[i], self.v_n_filters[i], bias=False)
#             for i in range(self.n_layers)
#         ])
#         self.s_to_v_activations = torch.nn.ModuleList(
#             [  # use tanh as gating function rather than standard activation which is unbound
#                 Activation(self.activation, self.v_n_filters[i])
#                 # positive outputs only to maintain equivariance (no vectors flipped)
#                 for i in range(self.n_layers)
#             ])
#         self.v_fc_norms = torch.nn.ModuleList([
#             Normalization(self.v_norm_mode, self.v_n_filters[i])
#             for i in range(self.n_layers)
#         ])
#         self.vector_to_scalar = torch.nn.ModuleList([
#             Scalarizer(self.v_n_filters[i], 3, self.norm_mode, self.activation, self.dropout_p)
#             for i in range(self.n_layers)
#         ])
#         self.scalar_to_vector_norm = torch.nn.ModuleList([
#             Normalization(self.norm_mode, self.v_n_filters[i])
#             for i in range(self.n_layers)
#         ])
#         self.vector_activation = torch.nn.ModuleList([
#             VectorActivation(self.v_n_filters[i], self.activation)
#             for i in range(self.n_layers)
#         ])
#
#         '''output layer'''
#         if self.v_output_dim != self.n_filters[-1]:
#             self.v_output_layer = nn.Linear(self.v_n_filters[-1], self.v_output_dim, bias=False)
#         else:
#             self.v_output_layer = nn.Identity()
#
#     def forward(self, x, v=None, conditions=None, return_latent=False, batch=None):
#         if conditions is not None:
#             x = torch.cat((x, conditions), dim=-1)
#
#         x = self.init_layer(x)  # get the right feature depth
#         if v is not None:
#             v = self.v_init_layer(v)
#
#         for i, (norm, linear, activation, dropout) in enumerate(
#                 zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts)):
#             x, v = self.get_residues(i, x, v)
#
#             x = self.scalar_forward(i, activation, batch, dropout, linear, norm, x, v)
#
#             if self.equivariant:
#                 v = self.vector_forward(i, x, v, batch)
#
#         if not self.equivariant:
#             if return_latent:
#                 return self.output_layer(x), x
#             else:
#                 return self.output_layer(x)
#         else:
#             if return_latent:
#                 return self.output_layer(x), self.v_output_layer(v), x
#             else:
#                 return self.output_layer(x), self.v_output_layer(v)
#
#     def get_residues(self, i, x, v):
#         if self.same_depth:
#             x = x.clone()
#         else:
#             x = self.residue_adjust[i](x)
#         if self.equivariant:
#             if self.same_depth:
#                 v = v.clone()
#             else:
#                 v = self.v_residue_adjust[i](v)
#         else:
#             v = None
#
#         return x, v
#
#     def scalar_forward(self, i, activation, batch, dropout, linear, norm, x, v):
#         res = x.clone()
#         if v is not None:
#             if self.v_to_s_combination == 'concatenate':
#                 # concatenate vector lengths to scalar values
#                 x = torch.cat([x, self.vector_to_scalar[i](v)],
#                               dim=-1)
#             elif self.v_to_s_combination == 'sum':
#                 x = x + self.vector_to_scalar[i](v)
#             else:
#                 assert False, f'{self.v_to_s_combination} not implemented'
#
#         if self.norm_after_linear:
#             x = res + dropout(activation(norm(linear(x), batch=batch)))
#         else:
#             x = res + dropout(activation(linear(norm(x, batch=batch))))
#
#         return x
#
#     def vector_forward(self, i, x, v, batch):
#         gating_factor = self.s_to_v_activations[i](
#             self.scalar_to_vector_norm[i](
#                 self.s_to_v_gating_layers[i](x))[:, None, :]
#         )
#         vector_mix = self.v_fc_norms[i](self.v_fc_layers[i](v), batch=batch)
#         vector_mix = self.vector_activation[i](vector_mix)
#
#         # A(FC(x)) * FC(N(v))   # rescaling factor keeps norm from exploding
#         return (v + gating_factor * vector_mix) / self.vector_addition_rescaling_factor

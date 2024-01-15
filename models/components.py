import sys

import torch
from torch import nn
import torch_geometric.nn as gnn
from torch.nn import functional as F
from e3nn import o3
import e3nn.nn as enn

from models.asymmetric_radius_graph import asymmetric_radius_graph


class MLP(nn.Module):
    def __init__(self, layers, filters, input_dim, output_dim,
                 activation='gelu', seed=0, dropout=0, conditioning_dim=0,
                 norm=None, bias=True, norm_after_linear=False,
                 conditioning_mode='concat_to_first',
                 equivariant=False,
                 residue_v_to_s=False,
                 vector_output_dim=None):
        super(MLP, self).__init__()
        # initialize constants and layers

        self.n_layers = layers
        self.conditioning_mode = conditioning_mode  # todo write a proper all_layer conditioning mode
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim
        self.v_output_dim = vector_output_dim if vector_output_dim is not None else output_dim
        self.input_dim = input_dim + conditioning_dim
        self.norm_mode = norm
        self.dropout_p = dropout
        self.activation = activation
        self.bias = bias
        self.norm_after_linear = norm_after_linear
        self.equivariant = equivariant
        self.residue_v_to_s = residue_v_to_s
        if residue_v_to_s:
            assert self.equivariant

        torch.manual_seed(seed)

        self.init_filters(filters, layers)

        self.init_scalar_transforms()

        if equivariant:
            self.init_vector_transforms()

    def init_filters(self, filters, layers):
        if isinstance(filters, list):
            self.n_filters = filters
        else:
            self.n_filters = [filters for _ in range(layers + 1)]
            self.same_depth = True
        if self.n_filters.count(self.n_filters[0]) != len(self.n_filters):  # if they are not all the same, we need residue adjustments
            self.same_depth = False
            self.residue_adjust = torch.nn.ModuleList([
                nn.Linear(self.n_filters[i], self.n_filters[i + 1], bias=False)
                for i in range(self.n_layers)
            ])
            if self.equivariant:
                self.v_residue_adjust = torch.nn.ModuleList([
                    nn.Linear(self.n_filters[i], self.n_filters[i + 1], bias=False)
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
            nn.Linear(self.n_filters[i] + (self.n_filters[i] if self.equivariant else 0),
                      self.n_filters[i + 1], bias=self.bias)
            for i in range(self.n_layers)
        ])
        self.fc_activations = torch.nn.ModuleList([
            Activation(self.activation, self.n_filters[i + 1])
            for i in range(self.n_layers)
        ])
        if self.norm_after_linear:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode, self.n_filters[i + 1])
                for i in range(self.n_layers)
            ])
        else:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode,
                              self.n_filters[i] + (self.n_filters[i] if self.equivariant else 0))
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
        if self.input_dim != self.n_filters[0]:
            self.v_init_layer = nn.Linear(self.input_dim//2, self.n_filters[0], bias=False)
        else:
            self.v_init_layer = nn.Identity()

        '''working layers'''
        self.v_fc_layers = torch.nn.ModuleList([
            nn.Linear(self.n_filters[i + 1], self.n_filters[i + 1], bias=False)
            for i in range(self.n_layers)
        ])
        self.s_to_v_gating_layers = torch.nn.ModuleList([
            nn.Linear(self.n_filters[i + 1], self.n_filters[i + 1], bias=True)
            for i in range(self.n_layers)
        ])
        self.s_to_v_activations = torch.nn.ModuleList([
            Activation(self.activation, self.n_filters[i + 1])
            for i in range(self.n_layers)
        ])

        '''output layer'''
        if self.v_output_dim != self.n_filters[-1]:
            self.v_output_layer = nn.Linear(self.n_filters[-1], self.v_output_dim, bias=False)
        else:
            self.v_output_layer = nn.Identity()

    def forward(self, x, v=None, conditions=None, return_latent=False, batch=None):
        if conditions is not None:
            x = torch.cat((x, conditions), dim=-1)

        x = self.init_layer(x)  # get the right feature depth
        if v is not None:
            v = self.v_init_layer(v)

        for i, (norm, linear, activation, dropout) in enumerate(zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts)):
            res, v_res = self.get_residues(i, v, x)

            x = self.scalar_forward(activation, batch, dropout, i, linear, norm, x, v, res)

            if self.equivariant:
                v = self.vector_forward(i, v, x, v_res)

        if not self.equivariant:
            if return_latent:
                return self.output_layer(x), x
            else:
                return self.output_layer(x)
        else:
            if return_latent:
                return self.output_layer(x), self.v_output_layer(v), x
            else:
                return self.output_layer(x), self.v_output_layer(v)

    def get_residues(self, i, v, x):
        if self.same_depth:
            res = x.clone()
        else:
            res = self.residue_adjust[i](x)
        if self.equivariant:
            if self.same_depth:
                v_res = v.clone()
            else:
                v_res = self.v_residue_adjust[i](v)
        else:
            v_res = None

        return res, v_res

    def vector_forward(self, i, v, x, v_res):
        v = v_res + self.s_to_v_activations[i](self.s_to_v_gating_layers[i](x)[:, None, :]) * self.v_fc_layers[i](v)  # A(FC(x)) * FC(v)
        return v

    def scalar_forward(self, activation, batch, dropout, i, linear, norm, x, v, res):
        if v is not None:  # concatenate vector lengths to scalar values
            x = torch.cat([x, torch.linalg.norm(v, dim=1)], dim=-1)

        if self.norm_after_linear:
            x = res + dropout(activation(norm(linear(x), batch=batch)))
        else:
            x = res + dropout(activation(linear(norm(x, batch=batch))))

        return x


'''
equivariance test
>> linear scaling layer
from scipy.spatial.transform import Rotation as R

rmat = torch.tensor(R.random().as_matrix(),device=x.device, dtype=torch.float32)

v1 = v.clone()
rotv1 = torch.einsum('ij, njk->nik', rmat, v1)

y1 = v1 + F.tanh(self.s_to_v_gating_layers[i](x)[:, None, :]) * self.v_fc_layers[i](v1)
y2 = rotv1 + F.tanh(self.s_to_v_gating_layers[i](x)[:, None, :]) * self.v_fc_layers[i](rotv1)

roty1 = torch.einsum('ij, njk->nik', rmat, y1)

print(torch.mean(torch.abs(y2 - roty1)))

'''


class EMLP(nn.Module):  # equivariant MLP  # todo deprecate
    def __init__(self, layers, irreps_hidden, irreps_in, irreps_out,
                 activation='gelu', seed=0,
                 norm=None):
        super(EMLP, self).__init__()
        # initialize constants and layers

        self.n_layers = layers
        self.irreps_out = irreps_out
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.activation = activation

        torch.manual_seed(seed)

        self.fc_layers = torch.nn.ModuleList([
            o3.Linear(irreps_hidden, irreps_hidden)
            for i in range(self.n_layers)
        ])

        activations = [Activation(activation, 1) if '0e' in str(irrep) else None for irrep in irreps_hidden]
        self.fc_activations = torch.nn.ModuleList([
            enn.Activation(irreps_in=irreps_hidden,
                           acts=activations)
            for _ in range(self.n_layers)
        ])

        if self.irreps_in != self.irreps_hidden:
            self.init_layer = o3.Linear(self.irreps_in, self.irreps_hidden)  # set appropriate sizing
        else:
            self.init_layer = nn.Identity()

        if self.irreps_in != self.irreps_hidden:
            self.output_layer = o3.Linear(self.irreps_hidden, self.irreps_out)
        else:
            self.output_layer = nn.Identity()

    def forward(self, x, conditions=None, return_latent=False, batch=None):

        x = self.init_layer(x)  # get the right feature depth

        for i, (linear, activation) in enumerate(zip(self.fc_layers, self.fc_activations)):
            x = x + activation(linear(x))  # residue -- always same hidden dimension

        if return_latent:
            return self.output_layer(x), x
        else:
            return self.output_layer(x)


class Normalization(nn.Module):
    def __init__(self, norm, filters, *args, **kwargs):
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

    def forward(self, input, batch=None):
        if batch is not None and self.norm_type != 'batch' and self.norm_type is not None:
            return self.norm(input, batch)

        return self.norm(input)


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func.lower() == 'relu':
            self.activation = F.relu
        elif activation_func.lower() == 'gelu':
            self.activation = F.gelu
        elif activation_func.lower() == 'kernel':  # rather expensive
            self.activation = kernelActivation(n_basis=10, span=4, channels=filters)
        elif activation_func.lower() == 'leaky relu':
            self.activation = F.leaky_relu

    def forward(self, input):
        return self.activation(input)


class kernelActivation(nn.Module):  # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis))  # positive and negative values for Dirichlet Kernel
        gamma = 1 / (6 * (self.dict[-1] - self.dict[-2]) ** 2)  # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma', torch.ones(1) * gamma)  #

        # self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1, 1), groups=int(channels), bias=False)

        # nn.init.normal(self.linear.weight.data, std=0.1)

    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x) == 2:
            x = x.reshape(2, self.channels, 1)

        return torch.exp(-self.gamma * (x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1)  # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1)  # apply linear coefficients and sum

        return x


def construct_radial_graph(pos, batch, ptr, cutoff, max_num_neighbors, aux_ind=None):
    """
    construct edge indices over a radial graph
    optionally, compute intra (within ref_mol_inds) and inter (between ref_mol_inds and outside inds) edges
    """
    if aux_ind is not None:
        inside_inds = torch.where(aux_ind == 0)[0]
        outside_inds = torch.where(aux_ind == 1)[0]  # atoms which are not in the asymmetric unit but which we will convolve - pre-excluding many from outside the cutoff
        inside_batch = batch[inside_inds]  # get the feature vectors we want to repeat
        n_repeats = [int(torch.sum(batch == ii) / torch.sum(inside_batch == ii)) for ii in range(len(ptr) - 1)]  # number of molecules in convolution region

        # intramolecular edges
        edge_index = asymmetric_radius_graph(pos, batch=batch, r=cutoff,  # intramolecular interactions - stack over range 3 convolutions
                                             max_num_neighbors=max_num_neighbors, flow='source_to_target',
                                             inside_inds=inside_inds, convolve_inds=inside_inds)

        # intermolecular edges
        edge_index_inter = asymmetric_radius_graph(pos, batch=batch, r=cutoff,  # extra radius for intermolecular graph convolution
                                                   max_num_neighbors=max_num_neighbors, flow='source_to_target',
                                                   inside_inds=inside_inds, convolve_inds=outside_inds)

        return {'edge_index': edge_index, 'edge_index_inter': edge_index_inter, 'inside_inds': inside_inds,
                'outside_inds': outside_inds, 'inside_batch': inside_batch, 'n_repeats': n_repeats}

    else:

        edge_index = gnn.radius_graph(pos, r=cutoff, batch=batch,
                                      max_num_neighbors=max_num_neighbors, flow='source_to_target')  # note - requires batch be monotonically increasing

        return {'edge_index': edge_index}

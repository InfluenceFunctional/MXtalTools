'''Import statements'''
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric
import torch_geometric.nn as gnn
from models.DimeNetCustom import CustomDimeNet
from models.CustomSchNet import CustomSchNet
from models.MikesGraphNet import MikesGraphNet, FCBlock
import sys
from nflib.flows import *
from nflib.nets import *
from nflib.spline_flows import *
from torch.distributions import MultivariateNormal, Uniform
import itertools
from sklearn.decomposition import PCA
from models.asymmetric_radius_graph import asymmetric_radius_graph
from utils import single_point_compute_rdf_torch, parallel_compute_rdf_torch


class molecule_graph_model(nn.Module):
    def __init__(self, dataDims, seed,
                 num_atom_feats,
                 num_mol_feats,
                 output_dimension,
                 activation,
                 num_fc_layers,
                 fc_depth,
                 fc_dropout_probability,
                 fc_norm_mode,
                 graph_model,
                 graph_filters,
                 graph_convolutional_layers,
                 concat_mol_to_atom_features,
                 pooling,
                 graph_norm,
                 num_spherical,
                 num_radial,
                 graph_convolution,
                 num_attention_heads,
                 add_spherical_basis,
                 add_torsional_basis,
                 atom_embedding_size,
                 radial_function,
                 max_num_neighbors,
                 convolution_cutoff,
                 return_latent=False, crystal_mode=False, crystal_convolution_type=None, device='cuda'):
        super(molecule_graph_model, self).__init__()
        # initialize constants and layers
        self.device = device
        self.return_latent = return_latent
        self.activation = activation
        self.num_fc_layers = num_fc_layers
        self.fc_depth = fc_depth
        self.fc_dropout_probability = fc_dropout_probability
        self.fc_norm_mode = fc_norm_mode
        self.graph_model = graph_model
        self.graph_convolution = graph_convolution
        self.output_classes = output_dimension
        self.graph_convolution_layers = graph_convolutional_layers
        self.graph_filters = graph_filters
        self.graph_norm = graph_norm
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_attention_heads = num_attention_heads
        self.add_spherical_basis = add_spherical_basis
        self.add_torsional_basis = add_torsional_basis
        self.n_mol_feats = num_mol_feats  # dataDims['num mol features']
        self.n_atom_feats = num_atom_feats  # dataDims['num atom features']
        self.radial_function = radial_function
        self.max_num_neighbors = max_num_neighbors
        self.graph_convolution_cutoff = convolution_cutoff
        if not concat_mol_to_atom_features:  # if we are not adding molwise feats to atoms, subtract the dimension
            self.n_atom_feats -= self.n_mol_feats
        self.pooling = pooling
        self.fc_norm_mode = fc_norm_mode
        self.embedding_dim = atom_embedding_size
        self.crystal_mode = crystal_mode
        self.crystal_convolution_type = crystal_convolution_type

        torch.manual_seed(seed)

        if self.graph_model is not None:
            if self.graph_model == 'mike':  # mike's net - others currently deprecated. For future, implement new convolutions in the mikenet class
                self.graph_net = MikesGraphNet(
                    crystal_mode=crystal_mode,
                    crystal_convolution_type=self.crystal_convolution_type,
                    graph_convolution_filters=self.graph_filters,
                    graph_convolution=self.graph_convolution,
                    out_channels=self.fc_depth,
                    hidden_channels=self.embedding_dim,
                    num_blocks=self.graph_convolution_layers,
                    num_radial=self.num_radial,
                    num_spherical=self.num_spherical,
                    max_num_neighbors=self.max_num_neighbors,
                    cutoff=self.graph_convolution_cutoff,
                    activation='gelu',
                    embedding_hidden_dimension=self.embedding_dim,
                    num_atom_features=self.n_atom_feats,
                    norm=self.graph_norm,
                    dropout=self.fc_dropout_probability,
                    spherical_embedding=self.add_spherical_basis,
                    torsional_embedding=self.add_torsional_basis,
                    radial_embedding=self.radial_function,
                    atom_embedding_dims=dataDims['atom embedding dict sizes'],
                    attention_heads=self.num_attention_heads,
                )
            else:
                print(self.graph_model + ' is not a valid graph model!!')
                sys.exit()
        else:
            self.graph_net = nn.Identity()
            self.graph_filters = 0  # no accounting for dime inputs or outputs
            self.pools = nn.Identity()

        # initialize global pooling operation
        if self.graph_model is not None:
            self.global_pool = global_aggregation(self.pooling, self.fc_depth)

        # molecule features FC layer
        if self.n_mol_feats != 0:
            self.mol_fc = nn.Linear(self.n_mol_feats, self.n_mol_feats)

        # FC model to post-process graph fingerprint
        self.gnn_mlp = general_MLP(layers=self.num_fc_layers,
                                   filters=self.fc_depth,
                                   norm=self.fc_norm_mode,
                                   dropout=self.fc_dropout_probability,
                                   input_dim=self.fc_depth,
                                   output_dim=self.fc_depth,
                                   conditioning_dim=self.n_mol_feats,
                                   seed=seed
                                   )

        self.output_fc = nn.Linear(self.fc_depth, self.output_classes, bias=False)

    def forward(self, data, return_latent=False, return_dists=False):
        extra_outputs = {}
        if self.graph_model is not None:
            x, dists_dict = self.graph_net(data.x[:, :self.n_atom_feats], data.pos, data.batch, ptr=data.ptr, ref_mol_inds=data.aux_ind, return_dists=return_dists)  # get atoms encoding
            if self.crystal_mode:  # model only outputs ref mol atoms - many fewer
                x = self.global_pool(x, data.batch[torch.where(data.aux_ind == 0)[0]])
            else:
                x = self.global_pool(x, data.batch)  # aggregate atoms to molecule

        mol_feats = self.mol_fc(data.x[data.ptr[:-1], -self.n_mol_feats:])  # molecule features are repeated, only need one per molecule (hence data.ptr)

        if self.graph_model is not None:
            x = self.gnn_mlp(x, conditions=mol_feats)  # mix graph fingerprint with molecule-scale features
        else:
            x = self.gnn_mlp(mol_feats)

        output = self.output_fc(x)

        if return_dists:
            if self.graph_model is not None:
                extra_outputs['dists dict'] = dists_dict
            else:
                extra_outputs['dists dict'] = None
        if return_latent:
            extra_outputs['latent'] = output.cpu().detach().numpy()

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output


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

        # y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        # for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func.lower() == 'relu':
            self.activation = F.relu
        elif activation_func.lower() == 'gelu':
            self.activation = F.gelu
        elif activation_func.lower() == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)
        elif activation_func.lower() == 'leaky relu':
            self.activation = F.leaky_relu

    def forward(self, input):
        return self.activation(input)


class Normalization(nn.Module):
    def __init__(self, norm, filters, *args, **kwargs):
        super().__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(filters)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(filters)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            print(norm + " is not a valid normalization")
            sys.exit()

    def forward(self, input):
        return self.norm(input)


class independent_gaussian_model(nn.Module):
    def __init__(self, input_dim, means, stds, normed_length_means, normed_length_stds, cov_mat=None):
        super(independent_gaussian_model, self).__init__()

        self.input_dim = input_dim
        fixed_norms = torch.Tensor(means)
        fixed_norms[:3] = torch.Tensor(normed_length_means)
        fixed_stds = torch.Tensor(stds)
        fixed_stds[:3] = torch.Tensor(normed_length_stds)

        self.register_buffer('means', torch.Tensor(means))
        self.register_buffer('stds', torch.Tensor(stds))
        self.register_buffer('fixed_norms', torch.Tensor(fixed_norms))
        self.register_buffer('fixed_stds', torch.Tensor(fixed_stds))

        if cov_mat is not None:
            pass
        else:
            cov_mat = torch.diag(torch.Tensor(fixed_stds).pow(2))

        fixed_means = means.copy()
        fixed_means[:3] = normed_length_means
        self.prior = MultivariateNormal(fixed_norms, torch.Tensor(cov_mat))  # apply standardization
        self.dummy_params = nn.Parameter(torch.ones(100))

    def forward(self, data, num_samples):
        '''
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        '''
        # conditions are unused - dummy
        # denormalize sample before standardizing
        samples = self.prior.sample((num_samples,)).to(data.x.device)
        samples[:, :3] = samples[:, :3] * (data.Z[:, None] ** (1 / 3)) * (data.mol_volume[:, None] ** (1 / 3))
        return (samples - self.means.to(samples.device)) / self.stds.to(samples.device)  # we want samples in standardized basis

    def backward(self, samples):
        return samples * self.stds + self.means

    def score(self, samples):
        return self.prior.log_prob(samples)


class general_MLP(nn.Module):
    def __init__(self, layers, filters, input_dim, output_dim, activation='gelu', seed=0, dropout=0, conditioning_dim=0, norm=None):
        super(general_MLP, self).__init__()
        # initialize constants and layers
        self.n_layers = layers
        self.n_filters = filters
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim
        self.input_dim = input_dim + conditioning_dim
        self.norm_mode = norm
        self.dropout_p = dropout
        self.activation = activation

        torch.manual_seed(seed)

        self.fc_layers = torch.nn.ModuleList([
            nn.Linear(self.n_filters, self.n_filters)
            for _ in range(self.n_layers)
        ])

        self.fc_norms = torch.nn.ModuleList([
            Normalization(self.norm_mode, self.n_filters)
            for _ in range(self.n_layers)
        ])

        self.fc_dropouts = torch.nn.ModuleList([
            nn.Dropout(p=self.dropout_p)
            for _ in range(self.n_layers)
        ])

        self.fc_activations = torch.nn.ModuleList([
            Activation(activation, self.n_filters)
            for _ in range(self.n_layers)
        ])

        self.init_layer = nn.Linear(self.input_dim, self.n_filters)  # set appropriate sizing
        self.output_layer = nn.Linear(self.n_filters, self.output_dim, bias=False)

    def forward(self, x, conditions=None, return_latent=False):
        if type(x) == torch_geometric.data.batch.CrystalDataBatch:  # extract conditions from trailing atomic features
            # if x.num_graphs == 1:
            #     x = x.x[:, -self.input_dim:]
            # else:
            x = gnn.global_max_pool(x.x, x.batch)[:, -self.input_dim:]  # x.x[x.ptr[:-1]][:, -self.input_dim:]

        if conditions is not None:
            # if type(conditions) == torch_geometric.data.batch.DataBatch: # extract conditions from trailing atomic features
            #     if len(x) == 1:
            #         conditions = conditions.x[:,-self.conditioning_dim:]
            #     else:
            #         conditions = conditions.x[conditions.ptr[:-1]][:,-self.conditioning_dim:]

            x = torch.cat((x, conditions), dim=1)

        x = self.init_layer(x)

        for norm, linear, activation, dropout in zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts):
            x = x + dropout(activation(linear(norm(x))))  # residue

        if return_latent:
            return self.output_layer(x), x
        else:
            return self.output_layer(x)


class global_aggregation(nn.Module):
    '''
    wrapper for several types of global aggregation functions
    '''

    def __init__(self, agg_func, filters):
        super(global_aggregation, self).__init__()
        self.agg_func = agg_func
        if agg_func == 'mean':
            self.agg = gnn.global_mean_pool
        elif agg_func == 'sum':
            self.agg = gnn.global_add_pool
        elif agg_func == 'attention':
            self.agg = gnn.GlobalAttention(nn.Sequential(nn.Linear(filters, filters), nn.LeakyReLU(), nn.Linear(filters, 1)))
        elif agg_func == 'set2set':
            self.agg = gnn.Set2Set(in_channels=filters, processing_steps=4)
            self.agg_fc = nn.Linear(filters * 2, filters)  # condense to correct number of filters
        elif agg_func == 'combo':
            self.agg_list1 = [gnn.global_max_pool, gnn.global_mean_pool, gnn.global_add_pool]  # simple aggregation functions
            self.agg_list2 = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.Linear(filters, filters), nn.LeakyReLU(), nn.Linear(filters, 1)))])  # aggregation functions requiring parameters
            self.agg_fc = nn.Linear(filters * (len(self.agg_list1) + len(self.agg_list2)), filters)  # condense to correct number of filters

    def forward(self, x, batch):
        if self.agg_func == 'set2set':
            x = self.agg(x, batch)
            return self.agg_fc(x)
        elif self.agg_func == 'combo':
            output1 = [agg(x, batch) for agg in self.agg_list1]
            output2 = [agg(x, batch) for agg in self.agg_list2]
            return self.agg_fc(torch.cat((output1 + output2), dim=1))
        else:
            return self.agg(x, batch)


def crystal_rdf(crystaldata, rrange=[0, 10], bins=100, intermolecular=False, elementwise=False, raw_density=False, atomwise=False):
    '''
    compute the RDF for all the supercells in a CrystalData object
    without respect for atom type
    '''
    if crystaldata.aux_ind is not None:
        in_inds = torch.where(crystaldata.aux_ind == 0)[0]
        if intermolecular:
            out_inds = torch.where(crystaldata.aux_ind == 1)[0].to(crystaldata.pos.device)
        else:
            out_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
    else:
        in_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
        out_inds = in_inds

    edges = asymmetric_radius_graph(crystaldata.pos,
                                    batch=crystaldata.batch,
                                    inside_inds=in_inds,
                                    convolve_inds=out_inds,
                                    r=max(rrange), max_num_neighbors=500, flow='source_to_target')

    crystal_number = crystaldata.batch[edges[0]]

    dists = (crystaldata.pos[edges[0]] - crystaldata.pos[edges[1]]).pow(2).sum(dim=-1).sqrt()



    assert not (elementwise and atomwise)

    if elementwise:
        if raw_density:
            density = torch.ones(crystaldata.num_graphs).to(dists.device)
        else:
            density = None
        relevant_elements = [5, 6, 7, 8, 9, 15, 16, 17, 35]
        element_symbols = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
        elements = [crystaldata.x[edges[0], 0], crystaldata.x[edges[1], 0]]
        rdfs_dict = {}
        rdfs_array = torch.zeros((crystaldata.num_graphs, int((len(relevant_elements) ** 2 + len(relevant_elements)) / 2), bins))
        ind = 0
        for i, element1 in enumerate(relevant_elements):
            for j, element2 in enumerate(relevant_elements):
                if j >= i:
                    rdfs_dict[ind] = element_symbols[element1] + ' to ' + element_symbols[element2]
                    rdfs_array[:, ind], rr = parallel_compute_rdf_torch([dists[(crystal_number == n) * (elements[0] == element1) * (elements[1] == element2)] for n in range(crystaldata.num_graphs)],
                                                                        rrange=rrange, bins=bins, density=density)
                    ind += 1
        return rdfs_array, rr, rdfs_dict
    elif atomwise:  # todo finish & test
        # generate atomwise indices
        rdfs_array_list = []
        rdfs_dict_list = []
        all_atom_inds = []
        for i in range(crystaldata.num_graphs):
            all_atom_inds.append(torch.arange(crystaldata.mol_size[i]).tile(int((crystaldata.batch == i).sum() // crystaldata.mol_size[i])))
        atom_inds = torch.cat(all_atom_inds)
        atoms = [atom_inds[edges[0]].to(crystaldata.x.device), atom_inds[edges[1]].to(crystaldata.x.device)]
        ind = 0
        for n in range(crystaldata.num_graphs):
            atom = []
            for i in range(int(crystaldata.mol_size[n])):
                for j in range(int(crystaldata.mol_size[n])):
                    if j >= i:
                        atom.append([i, j])
            atom = torch.Tensor(atom)
            rdfs_dict_list.append(atom)

            if raw_density:
                density = torch.ones(len(atom)).to(dists.device)
            else:
                density = None

            rdfs_array, rr = parallel_compute_rdf_torch([dists[(crystal_number == n) * (atoms[0] == atom[m, 0]) * (atoms[1] == atom[m, 1])]
                                                         for m in range(len(atom))],
                                                        rrange=rrange, bins=bins, density=density)
            ind += 1

            rdfs_array_list.append(rdfs_array)

        return rdfs_array_list, rr, rdfs_dict_list

    else:
        if raw_density:
            density = torch.ones(crystaldata.num_graphs).to(dists.device)
        else:
            density = None
        return parallel_compute_rdf_torch([dists[crystal_number == n] for n in range(crystaldata.num_graphs)], rrange=rrange, bins=bins, density=density)

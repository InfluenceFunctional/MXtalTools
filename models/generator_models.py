import sys

import torch
from torch import nn as nn
from torch.distributions import MultivariateNormal, Uniform

from crystal_building.utils import clean_cell_params
from models.components import MLP
from models.base_models import molecule_graph_model
from constants.asymmetric_units import asym_unit_dict


class crystal_generator(nn.Module):
    def __init__(self, seed, device, config, dataDims, sym_info):
        super(crystal_generator, self).__init__()

        self.device = device
        self.symmetries_dict = sym_info
        self.lattice_means = torch.tensor(dataDims['lattice_means'], dtype=torch.float32, device=device)
        self.lattice_stds = torch.tensor(dataDims['lattice_stds'], dtype=torch.float32, device=device)
        self.norm_lattice_lengths = False

        # initialize asymmetric unit dict
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(self.device)

        '''set random prior'''
        self.latent_dim = config.prior_dimension
        if config.prior == 'multivariate normal':
            self.prior = MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim))
        elif config.prior.lower() == 'uniform':
            self.prior = Uniform(low=0, high=1)
        else:
            print(config.prior + ' is not an implemented prior!!')
            sys.exit()

        '''conditioning model'''
        self.num_crystal_features = dataDims['num crystal generation features']
        torch.manual_seed(config.seeds.model)

        self.conditioner = molecule_graph_model(
            dataDims=dataDims,
            atom_type_embedding_dims=config.conditioner.init_atom_embedding_dim,
            seed=seed,
            num_atom_feats=self.atom_input_feats,  # we will add directly the normed coordinates to the node features
            num_mol_feats=self.num_mol_feats,
            output_dimension=config.conditioner.output_dim,  # starting size for decoder model
            activation=config.conditioner.activation,
            num_fc_layers=config.conditioner.num_fc_layers,
            fc_depth=config.conditioner.fc_depth,
            fc_dropout_probability=config.conditioner.fc_dropout_probability,
            fc_norm_mode=config.conditioner.fc_norm_mode,
            graph_message_depth=config.conditioner.graph_filters,
            graph_convolutional_layers=config.conditioner.graph_convolution_layers,
            concat_mol_to_atom_features=config.conditioner.concat_mol_features,
            graph_aggregator=config.conditioner.pooling,
            graph_norm=config.conditioner.graph_norm,
            num_spherical=config.conditioner.num_spherical,
            num_radial=config.conditioner.num_radial,
            graph_convolution_type=config.conditioner.graph_convolution,
            num_attention_heads=config.conditioner.num_attention_heads,
            add_spherical_basis=config.conditioner.add_spherical_basis,
            add_torsional_basis=config.conditioner.add_torsional_basis,
            graph_node_dims=config.conditioner.atom_embedding_size,
            radial_function=config.conditioner.radial_function,
            max_num_neighbors=config.conditioner.max_num_neighbors,
            convolution_cutoff=config.conditioner.graph_convolution_cutoff,
            positional_embedding=config.conditioner.positional_embedding,
            max_molecule_size=config.max_molecule_radius,
            crystal_mode=False,
            crystal_convolution_type=None,
        )

        '''
        generator model
        '''
        self.model = MLP(layers=config.num_fc_layers,
                         filters=config.fc_depth,
                         norm=config.fc_norm_mode,
                         dropout=config.fc_dropout_probability,
                         input_dim=self.latent_dim,
                         output_dim=dataDims['num lattice features'] + 3,  # 3 extra dimensions for angle decoder
                         conditioning_dim=config.conditioner.output_dim + self.num_crystal_features,  # include crystal information for the generator
                         seed=config.seeds.model
                         )

    def sample_latent(self, n_samples):
        # return torch.ones((n_samples,12)).to(self.device) # when we don't actually want any noise (test purposes)
        return self.prior.sample((n_samples,)).to(self.device)

    def forward(self, n_samples, z=None, conditions=None, return_latent=False, return_condition=False, return_prior=False):
        if z is None:  # sample random numbers from prior distribution
            z = self.sample_latent(n_samples)

        normed_coords = conditions.pos / self.conditioner.max_molecule_size  # norm coords by maximum molecule radius
        crystal_information = conditions.x[:, -self.num_crystal_features:]

        if self.skinny_inputs:
            conditions.x = torch.cat((conditions.x[:, 0, None], normed_coords), dim=-1)  # take only the atomic number for atomwise features
        else:
            conditions.x = torch.cat((conditions.x[:, :-self.num_crystal_features], normed_coords), dim=-1)  # concatenate to input features, leaving out crystal info from conditioner

        conditions_encoding = self.conditioner(conditions)
        conditions_encoding = torch.cat((conditions_encoding, crystal_information[conditions.ptr[:-1]]), dim=-1)
        if return_latent:
            samples, latent = self.model(z, conditions=conditions_encoding, return_latent=return_latent)
        else:
            samples = self.model(z, conditions=conditions_encoding, return_latent=return_latent)

        clean_samples = clean_cell_params(samples, conditions.sg_ind, self.lattice_means, self.lattice_stds,
                                          self.symmetries_dict, self.asym_unit_dict, destandardize=True, mode='soft')

        if any((return_condition, return_prior, return_latent)):
            output = [clean_samples]
            if return_latent:
                output.append(latent)
            if return_prior:
                output.append(z)
            if return_condition:
                output.append(conditions_encoding)
            return output
        else:
            return clean_samples


class independent_gaussian_model(nn.Module):
    def __init__(self, input_dim, means, stds,  sym_info, device, cov_mat=None):
        super(independent_gaussian_model, self).__init__()

        self.device = device
        self.input_dim = input_dim
        means = torch.Tensor(means)
        stds = torch.Tensor(stds)

        self.register_buffer('means', torch.Tensor(means))
        self.register_buffer('stds', torch.Tensor(stds))
        self.register_buffer('fixed_norms', torch.Tensor(means))
        self.register_buffer('fixed_stds', torch.Tensor(stds))

        self.symmetries_dict = sym_info
        # initialize asymmetric unit dict
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key])#.to(self.device)

        if cov_mat is not None:
            pass
        else:
            cov_mat = torch.diag(torch.Tensor(stds).pow(2))

        try:
            self.prior = MultivariateNormal(means, torch.Tensor(cov_mat))  # apply standardization
        except ValueError:  # for some datasets (e.g., all tetragonal space groups) the covariance matrix is ill conditioned, so we throw away off diagonals (mostly unimportant)
            self.prior = MultivariateNormal(loc=means, covariance_matrix=torch.eye(12, dtype=torch.float32) * torch.Tensor(cov_mat).diagonal())

    def forward(self, num_samples, data):
        """
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        """
        samples = self.prior.sample((num_samples,)).to(data.x.device)  # samples in the destandardied 'real' basis
        final_samples = clean_cell_params(samples, data.sg_ind, self.means, self.stds,
                                          self.symmetries_dict, self.asym_unit_dict,
                                          rescale_asymmetric_unit=False, destandardize=False, mode='hard')

        return final_samples

    def backward(self, samples):
        return samples * self.stds + self.means

    def score(self, samples):
        return self.prior.log_prob(samples)

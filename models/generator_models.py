import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import nn as nn
from torch.distributions import MultivariateNormal, Uniform

from common.utils import components2angle
from crystal_building.utils import scale_asymmetric_unit, clean_cell_params
from models.components import MLP
from models.base_models import molecule_graph_model
from models.utils import enforce_1d_bound, decode_to_sph_rotvec, clean_generator_output, enforce_crystal_system
from constants.asymmetric_units import asym_unit_dict


class crystal_generator(nn.Module):
    def __init__(self, config, conditioner_config, generator_config, dataDims, sym_info):
        super(crystal_generator, self).__init__()

        self.device = config.device
        self.symmetries_dict = sym_info
        self.lattice_means = torch.tensor(dataDims['lattice means'], dtype=torch.float32, device=config.device)
        self.lattice_stds = torch.tensor(dataDims['lattice stds'], dtype=torch.float32, device=config.device)
        self.norm_lattice_lengths = False

        # initialize asymmetric unit dict
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(self.device)

        '''set random prior'''
        self.latent_dim = generator_config.prior_dimension
        if generator_config.prior == 'multivariate normal':
            self.prior = MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim))
        elif generator_config.prior.lower() == 'uniform':
            self.prior = Uniform(low=0, high=1)
        else:
            print(generator_config.prior + ' is not an implemented prior!!')
            sys.exit()

        '''conditioning model'''
        self.num_crystal_features = config.dataDims['num crystal generation features']
        torch.manual_seed(config.seeds.model)

        if conditioner_config.skinny_atomwise_features:
            self.skinny_inputs = True
            self.atom_input_feats = 1 + 3  # take first dim (atomic number) and three for coordinates
            self.num_mol_feats = 0
        else:
            self.skinny_inputs = False
            self.atom_input_feats = dataDims['num atom features'] + 3 - self.num_crystal_features
            self.num_mol_feats = dataDims['num mol features'] - self.num_crystal_features

        self.conditioner = molecule_graph_model(
            dataDims=dataDims,
            atom_embedding_dims=conditioner_config.init_atom_embedding_dim,
            seed=config.seeds.model,
            num_atom_feats=self.atom_input_feats,  # we will add directly the normed coordinates to the node features
            num_mol_feats=self.num_mol_feats,
            output_dimension=conditioner_config.output_dim,  # starting size for decoder model
            activation=conditioner_config.activation,
            num_fc_layers=conditioner_config.num_fc_layers,
            fc_depth=conditioner_config.fc_depth,
            fc_dropout_probability=conditioner_config.fc_dropout_probability,
            fc_norm_mode=conditioner_config.fc_norm_mode,
            graph_filters=conditioner_config.graph_filters,
            graph_convolutional_layers=conditioner_config.graph_convolution_layers,
            concat_mol_to_atom_features=conditioner_config.concat_mol_features,
            pooling=conditioner_config.pooling,
            graph_norm=conditioner_config.graph_norm,
            num_spherical=conditioner_config.num_spherical,
            num_radial=conditioner_config.num_radial,
            graph_convolution=conditioner_config.graph_convolution,
            num_attention_heads=conditioner_config.num_attention_heads,
            add_spherical_basis=conditioner_config.add_spherical_basis,
            add_torsional_basis=conditioner_config.add_torsional_basis,
            graph_embedding_size=conditioner_config.atom_embedding_size,
            radial_function=conditioner_config.radial_function,
            max_num_neighbors=conditioner_config.max_num_neighbors,
            convolution_cutoff=conditioner_config.graph_convolution_cutoff,
            positional_embedding=conditioner_config.positional_embedding,
            max_molecule_size=config.max_molecule_radius,
            crystal_mode=False,
            crystal_convolution_type=None,
        )

        '''
        generator model
        '''
        self.model = MLP(layers=generator_config.num_fc_layers,
                         filters=generator_config.fc_depth,
                         norm=generator_config.fc_norm_mode,
                         dropout=generator_config.fc_dropout_probability,
                         input_dim=self.latent_dim,
                         output_dim=dataDims['num lattice features'] + 3,  # 3 extra dimensions for angle decoder
                         conditioning_dim=conditioner_config.output_dim + self.num_crystal_features,  # include crystal information for the generator
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
    def __init__(self, input_dim, means, stds, normed_length_means, normed_length_stds, sym_info, device, cov_mat=None):
        super(independent_gaussian_model, self).__init__()

        self.device = device
        self.input_dim = input_dim
        fixed_norms = torch.Tensor(means)
        fixed_norms[:3] = torch.Tensor(normed_length_means)
        fixed_stds = torch.Tensor(stds)
        fixed_stds[:3] = torch.Tensor(normed_length_stds)

        self.register_buffer('means', torch.Tensor(means))
        self.register_buffer('stds', torch.Tensor(stds))
        self.register_buffer('fixed_norms', torch.Tensor(fixed_norms))
        self.register_buffer('fixed_stds', torch.Tensor(fixed_stds))

        self.symmetries_dict = sym_info
        # initialize asymmetric unit dict
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key])#.to(self.device)

        if cov_mat is not None:
            pass
        else:
            cov_mat = torch.diag(torch.Tensor(fixed_stds).pow(2))

        fixed_means = means.copy()
        fixed_means[:3] = normed_length_means
        try:
            self.prior = MultivariateNormal(fixed_norms, torch.Tensor(cov_mat))  # apply standardization
        except ValueError:  # for some datasets (e.g., all tetragonal space groups) the covariance matrix is ill conditioned, so we throw away off diagonals (mostly unimportant)
            self.prior = MultivariateNormal(loc=fixed_norms, covariance_matrix=torch.eye(12, dtype=torch.float32) * torch.Tensor(cov_mat).diagonal())
        self.dummy_params = nn.Parameter(torch.ones(100))

    def forward(self, num_samples, data):
        """
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        """
        # unconditional sampling from the raw data distribution
        sg_inds = data.sg_ind

        samples = self.prior.sample((num_samples,)).to(data.x.device)  # samples in the destandardied 'real' basis
        samples[:, :3] = samples[:, :3] * (data.Z[:, None] ** (1 / 3)) * (data.mol_volume[:, None] ** (1 / 3))  # denorm lattice vectors

        final_samples = clean_cell_params(samples, sg_inds, self.means, self.stds,
                                          self.symmetries_dict, self.asym_unit_dict,
                                          rescale_asymmetric_unit=False, destandardize=False, mode='hard')

        return final_samples

    def backward(self, samples):
        return samples * self.stds + self.means

    def score(self, samples):
        return self.prior.log_prob(samples)

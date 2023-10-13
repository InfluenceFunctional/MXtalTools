import sys

import torch
from torch import nn as nn
from torch.distributions import MultivariateNormal, Uniform

from constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from crystal_building.utils import clean_cell_params
from models.components import MLP
from models.base_models import molecule_graph_model
from constants.asymmetric_units import asym_unit_dict
from models.utils import clean_generator_output


class crystal_generator(nn.Module):
    def __init__(self, seed, device, config, dataDims, sym_info):
        super(crystal_generator, self).__init__()

        self.device = device
        self.symmetries_dict = sym_info
        self.lattice_means = torch.tensor(dataDims['lattice_means'], dtype=torch.float32, device=device)
        self.lattice_stds = torch.tensor(dataDims['lattice_stds'], dtype=torch.float32, device=device)
        self.radial_norm_factor = config.radial_norm_factor

        # initialize asymmetric unit dict
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(self.device)

        self.register_buffer('SG_FEATURE_TENSOR', SG_FEATURE_TENSOR.clone())

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
        torch.manual_seed(seed)

        self.conditioner = molecule_graph_model(
            num_atom_feats=dataDims['num_atom_features'],
            num_mol_feats=dataDims['num_molecule_features'],
            output_dimension=config.conditioner.graph_embedding_depth,
            seed=seed,
            graph_convolution_type=config.conditioner.graph_convolution_type,
            graph_aggregator=config.conditioner.graph_aggregator,
            concat_pos_to_atom_features=True,
            concat_mol_to_atom_features=config.conditioner.concat_mol_to_atom_features,
            concat_crystal_to_atom_features=True,
            activation=config.conditioner.activation,
            num_fc_layers=config.conditioner.num_fc_layers,
            fc_depth=config.conditioner.fc_depth,
            fc_norm_mode=config.conditioner.fc_norm_mode,
            fc_dropout_probability=config.conditioner.fc_dropout_probability,
            graph_node_norm=config.conditioner.graph_node_norm,
            graph_node_dropout=config.conditioner.graph_node_dropout,
            graph_message_norm=config.conditioner.graph_message_norm,
            graph_message_dropout=config.conditioner.graph_message_dropout,
            num_attention_heads=config.conditioner.num_attention_heads,
            graph_message_depth=config.conditioner.graph_message_depth,
            graph_node_dims=config.conditioner.graph_node_dims,
            num_graph_convolutions=config.conditioner.num_graph_convolutions,
            graph_embedding_depth=config.conditioner.graph_embedding_depth,
            nodewise_fc_layers=config.conditioner.nodewise_fc_layers,
            num_radial=config.conditioner.num_radial,
            radial_function=config.conditioner.radial_function,
            max_num_neighbors=config.conditioner.max_num_neighbors,
            convolution_cutoff=config.conditioner.convolution_cutoff,
            atom_type_embedding_dims=config.conditioner.atom_type_embedding_dims,
            periodic_structure=False,
            outside_convolution_type='none'
        )

        '''
        generator model
        '''
        self.model = MLP(layers=config.num_fc_layers,
                         filters=config.fc_depth,
                         norm=config.fc_norm_mode,
                         dropout=config.fc_dropout_probability,
                         input_dim=self.latent_dim,
                         output_dim=12 + 3,  # 3 extra dimensions for angle decoder
                         conditioning_dim=config.conditioner.graph_embedding_depth + SG_FEATURE_TENSOR.shape[1] + 1,  # include crystal information for the generator and the target packing coeff
                         seed=seed,
                         conditioning_mode=config.conditioning_mode,
                         )

    def sample_latent(self, n_samples):
        # return torch.ones((n_samples,12)).to(self.device) # when we don't actually want any noise (test purposes)
        return self.prior.sample((n_samples,)).to(self.device)

    def forward(self, n_samples, molecule_data, z=None, return_condition=False, return_prior=False, return_raw_samples=False, target_packing=0):
        if z is None:  # sample random numbers from prior distribution
            z = self.sample_latent(n_samples)

        molecule_data.pos = molecule_data.pos / self.radial_norm_factor
        molecule_encoding = self.conditioner(molecule_data)
        molecule_encoding = torch.cat((molecule_encoding,
                                       torch.tensor(self.SG_FEATURE_TENSOR[molecule_data.sg_ind], dtype=torch.float32, device=molecule_data.x.device),
                                       target_packing[:, None]), dim=-1)

        samples = self.model(z, conditions=molecule_encoding)

        clean_samples = clean_cell_params(samples, molecule_data.sg_ind, self.lattice_means, self.lattice_stds,
                                          self.symmetries_dict, self.asym_unit_dict, destandardize=True, mode='soft')

        if any((return_condition, return_prior, return_raw_samples)):
            output = [clean_samples]
            if return_prior:
                output.append(z)
            if return_condition:
                output.append(molecule_encoding)
            if return_raw_samples:
                output.append(
                    torch.cat(
                        clean_generator_output(samples, self.lattice_means, self.lattice_stds,
                                               destandardize=True, mode=None),
                        dim=-1))  # destandardize but don't clean up or normalize fractional outputs
            return output
        else:
            return clean_samples


class independent_gaussian_model(nn.Module):
    def __init__(self, input_dim, means, stds, sym_info, device, cov_mat=None):
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
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key])  # .to(self.device)

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

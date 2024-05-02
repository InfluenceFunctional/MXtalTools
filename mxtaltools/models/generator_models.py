import sys

import torch
from torch import nn as nn
from torch.distributions import MultivariateNormal, Uniform

from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.crystal_building.utils import clean_cell_params
from mxtaltools.models.base_graph_model import BaseGraphModel
from mxtaltools.models.components import MLP
from mxtaltools.models.molecule_graph_model import MoleculeGraphModel
from mxtaltools.constants.asymmetric_units import asym_unit_dict
from mxtaltools.models.utils import clean_generator_output


class CrystalGenerator(BaseGraphModel):
    def __init__(self, seed, device, config, sym_info,
                 atom_features: list,
                 molecule_features: list,
                 node_standardization_tensor: torch.tensor,
                 graph_standardization_tensor: torch.tensor,
                 lattice_means: torch.tensor,
                 lattice_stds: torch.tensor,
                 ):
        super(CrystalGenerator, self).__init__()

        self.device = device
        torch.manual_seed(seed)
        self.get_data_stats(atom_features,
                            molecule_features,
                            node_standardization_tensor,
                            graph_standardization_tensor)


        self.symmetries_dict = sym_info
        self.lattice_means = torch.tensor(lattice_means, dtype=torch.float32, device=device)
        self.lattice_stds = torch.tensor(lattice_stds, dtype=torch.float32, device=device)
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

        # equivalent to equivariant point encoder model
        self.conditioner = MoleculeGraphModel(
            input_node_dim=self.num_atom_feats,
            num_mol_feats=self.num_mol_feats,
            output_dim=config.conditioner.graph_embedding_depth,
            seed=seed,
            equivariant=True,
            graph_aggregator=config.conditioner.graph_aggregator,
            concat_pos_to_node_dim=True,
            concat_mol_to_node_dim=False,
            concat_crystal_to_node_dim=False,
            activation=config.conditioner.activation,
            fc_config=config.conditioner.fc,
            graph_config=config.conditioner.graph,
            periodize_inside_nodes=False,
            outside_convolution_type='none',
            vector_norm='graph vector layer' if config.conditioner.graph_node_norm == 'graph layer' else None,
        )

        '''
        generator model
        '''
        self.model = MLP(
            layers=config.generator.num_layers,
            filters=config.generator.hidden_dim,
            input_dim=self.latent_dim + SG_FEATURE_TENSOR.shape[1] + 1,
            # include crystal information for the generator and the target packing coeff
            output_dim=12 + 3,  # 3 extra dimensions for angle decoder
            conditioning_dim=0,
            activation='gelu',
            conditioning_mode=None,
            norm=config.generator.norm,
            dropout=config.generator.dropout,
            equivariant=True,
            vector_output_dim=3,  # opt for rotvec output
            vector_input_dim=config.conditioner.graph_embedding_depth,
            vector_norm=config.generator.norm,
            ramp_depth=False,
            v_to_s_combination='sum'
        )

    def sample_latent(self, n_samples):
        # return torch.ones((n_samples,12)).to(self.device) # when we don't actually want any noise (test purposes)
        return self.prior.sample((n_samples,)).to(self.device)

    def forward(self, n_samples, molecule_data, z=None, return_condition=False, return_prior=False,
                return_raw_samples=False, target_packing=0, skip_standardization=False):

        if not skip_standardization:
            molecule_data = self.standardize(molecule_data)

        if z is None:  # sample random numbers from prior distribution
            z = self.sample_latent(n_samples)

        molecule_data.pos = molecule_data.pos / self.radial_norm_factor
        _, molecule_encoding = self.conditioner(molecule_data)

        scalar_input = torch.cat((
            z,
            torch.tensor(self.SG_FEATURE_TENSOR[molecule_data.sg_ind],
                         dtype=torch.float32,
                         device=molecule_data.x.device),
            target_packing[:, None]), dim=-1)

        # conditioning goes to vector track, noise and crystal information to scalar track
        samples, _ = self.model(scalar_input, v=molecule_encoding)  # omit vector outputs for now

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
                        clean_generator_output(samples=samples,
                                               lattice_means=self.lattice_means,
                                               lattice_stds=self.lattice_stds,
                                               destandardize=True, mode=None),
                        dim=-1))  # destandardize but don't clean up or normalize fractional outputs
            return output
        else:
            return clean_samples


class IndependentGaussianGenerator(nn.Module):
    def __init__(self, input_dim, means, stds, sym_info, device, cov_mat=None):
        super(IndependentGaussianGenerator, self).__init__()

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
            self.prior = MultivariateNormal(loc=means,
                                            covariance_matrix=torch.eye(12, dtype=torch.float32) * torch.Tensor(
                                                cov_mat).diagonal())

    def forward(self, num_samples, data=None, sg_ind=None):
        """
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        """
        if data is None:
            pass
        else:
            sg_ind = data.sg_ind

        samples = self.prior.sample((num_samples,))  # samples in the destandardied 'real' basis
        final_samples = clean_cell_params(samples, sg_ind, self.means, self.stds,
                                          self.symmetries_dict, self.asym_unit_dict,
                                          rescale_asymmetric_unit=True, destandardize=False, mode='soft').to(
            self.device)

        return final_samples

    def backward(self, samples):
        return samples * self.stds + self.means

    def score(self, samples):
        return self.prior.log_prob(samples)

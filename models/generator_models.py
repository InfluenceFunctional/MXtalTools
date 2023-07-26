import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import MultivariateNormal, Uniform

from common.utils import components2angle
from models.components import MLP
from models.base_models import molecule_graph_model
from models.utils import enforce_1d_bound


class crystal_generator(nn.Module):
    def __init__(self, config, dataDims):
        super(crystal_generator, self).__init__()

        self.device = config.device

        self.lattice_means = torch.tensor(dataDims['lattice means'], dtype=torch.float32, device=config.device)
        self.lattice_stds = torch.tensor(dataDims['lattice stds'], dtype=torch.float32, device=config.device)
        self.norm_lattice_lengths = False

        '''set random prior'''
        self.latent_dim = config.generator.prior_dimension
        if config.generator.prior == 'multivariate normal':
            self.prior = MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim))
        elif config.generator.prior.lower() == 'uniform':
            self.prior = Uniform(low=0, high=1)
        else:
            print(config.generator.prior + ' is not an implemented prior!!')
            sys.exit()

        '''conditioning model'''
        self.num_crystal_features = config.dataDims['num crystal generation features']
        torch.manual_seed(config.seeds.model)

        if config.generator.conditioner.skinny_atomwise_features:
            self.skinny_inputs = True
            self.atom_input_feats = 1 + 3  # take first dim (atomic number) and three for coordinates
            self.num_mol_feats = 0
        else:
            self.skinny_inputs = False
            self.atom_input_feats = dataDims['num atom features'] + 3 - self.num_crystal_features
            self.num_mol_feats = dataDims['num mol features'] - self.num_crystal_features

        self.conditioner = molecule_graph_model(
            dataDims=dataDims,
            atom_embedding_dims=config.generator.conditioner.init_atom_embedding_dim,
            seed=config.seeds.model,
            num_atom_feats=self.atom_input_feats,  # we will add directly the normed coordinates to the node features
            num_mol_feats=self.num_mol_feats,
            output_dimension=config.generator.conditioner.output_dim,  # starting size for decoder model
            activation=config.generator.conditioner.activation,
            num_fc_layers=config.generator.conditioner.num_fc_layers,
            fc_depth=config.generator.conditioner.fc_depth,
            fc_dropout_probability=config.generator.conditioner.fc_dropout_probability,
            fc_norm_mode=config.generator.conditioner.fc_norm_mode,
            graph_filters=config.generator.conditioner.graph_filters,
            graph_convolutional_layers=config.generator.conditioner.graph_convolution_layers,
            concat_mol_to_atom_features=config.generator.conditioner.concat_mol_features,
            pooling=config.generator.conditioner.pooling,
            graph_norm=config.generator.conditioner.graph_norm,
            num_spherical=config.generator.conditioner.num_spherical,
            num_radial=config.generator.conditioner.num_radial,
            graph_convolution=config.generator.conditioner.graph_convolution,
            num_attention_heads=config.generator.conditioner.num_attention_heads,
            add_spherical_basis=config.generator.conditioner.add_spherical_basis,
            add_torsional_basis=config.generator.conditioner.add_torsional_basis,
            graph_embedding_size=config.generator.conditioner.atom_embedding_size,
            radial_function=config.generator.conditioner.radial_function,
            max_num_neighbors=config.generator.conditioner.max_num_neighbors,
            convolution_cutoff=config.generator.conditioner.graph_convolution_cutoff,
            positional_embedding=config.generator.conditioner.positional_embedding,
            max_molecule_size=config.max_molecule_radius,
            crystal_mode=False,
            crystal_convolution_type=None,
        )

        '''
        generator model
        '''
        self.model = MLP(layers=config.generator.num_fc_layers,
                         filters=config.generator.fc_depth,
                         norm=config.generator.fc_norm_mode,
                         dropout=config.generator.fc_dropout_probability,
                         input_dim=self.latent_dim,
                         output_dim=dataDims['num lattice features'] + 3,  # 3 extra dimensions for angle decoder
                         conditioning_dim=config.generator.conditioner.output_dim + self.num_crystal_features,  # include crystal information for the generator
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

        '''separate components'''
        lattice_lengths = samples[:, :3]
        lattice_angles = samples[:, 3:6]
        mol_positions = samples[:, 6:9]
        mol_orientations = samples[:, 9:]

        '''destandardize & decode angles - initial prediction was done in standardized basis'''
        real_lattice_lengths = lattice_lengths * self.lattice_stds[:3] + self.lattice_means[:3]
        real_lattice_angles = lattice_angles * self.lattice_stds[3:6] + self.lattice_means[3:6]  # not bothering to encode as an angle
        real_mol_positions = mol_positions * self.lattice_stds[6:9] + self.lattice_means[6:9]

        theta_encoding = F.sigmoid(mol_orientations[:, 0:2])  # restrict to positive quadrant
        real_orientation_theta = components2angle(theta_encoding)
        # phi_encoding = torch.cat((mol_orientations[:, 2, None], F.sigmoid(mol_orientations[:, 3, None])),dim=-1) # restrict to positive angles
        real_orientation_phi = components2angle(mol_orientations[:, 2:4])  # unrestricted
        real_orientation_r = components2angle(mol_orientations[:, 4:6])  # unrestricted

        '''enforce physical bounds on cell parameters'''
        clean_lattice_lengths = F.softplus(real_lattice_lengths - 0.1) + 0.1  # enforces positive nonzero
        clean_lattice_angles = enforce_1d_bound(real_lattice_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2, mode='soft')  # range from (0,pi) with 20% limit to prevent too-skinny cells
        clean_mol_positions = enforce_1d_bound(real_mol_positions, 0.5, 0.5, mode='soft')  # enforce fractional centroids between 0 and 1
        clean_mol_orientations = torch.cat((  # this one is already bounded
            real_orientation_theta[:,None],
            real_orientation_phi[:,None],
            real_orientation_r[:,None]
        ), dim=-1)

        model_samples = torch.cat((
            clean_lattice_lengths,
            clean_lattice_angles,
            clean_mol_positions,
            clean_mol_orientations,
        ), dim=-1)

        # into generator model
        if any((return_condition, return_prior, return_latent)):
            output = [model_samples]
            if return_latent:
                output.append(latent)
            if return_prior:
                output.append(z)
            if return_condition:
                output.append(conditions_encoding)
            return output
        else:
            return model_samples

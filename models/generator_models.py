import torch
import torch.nn as nn
import sys
from torch.distributions import MultivariateNormal, Uniform
from models.torch_models import molecule_graph_model
from models.model_components import general_MLP


class crystal_generator(nn.Module):
    def __init__(self, config, dataDims):
        super(crystal_generator, self).__init__()

        self.device = config.device
        self.latent_dim = config.generator.prior_dimension
        if config.generator.prior == 'multivariate normal':
            self.prior = MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim))
        elif config.generator.prior.lower() == 'uniform':
            self.prior = Uniform(low=0, high=1)
        else:
            print(config.generator.prior + ' is not an implemented prior!!')
            sys.exit()

        '''
        conditioning model
        '''
        self.crystal_features_to_ignore = config.dataDims['num crystal generation features']

        self.conditioner = molecule_graph_model(
            dataDims,
            seed=config.seeds.model,
            num_atom_feats=dataDims['num atom features'] + 3 - self.crystal_features_to_ignore,  # we will add directly the normed coordinates to the node features
            num_mol_feats=dataDims['num mol features'] - self.crystal_features_to_ignore,
            output_dimension=config.conditioner.fc_depth,
            activation=config.conditioner.activation,
            num_fc_layers=config.conditioner.num_fc_layers,
            fc_depth=config.conditioner.fc_depth,
            fc_dropout_probability=config.conditioner.fc_dropout_probability,
            fc_norm_mode=config.conditioner.fc_norm_mode,
            graph_model=config.conditioner.graph_model,
            graph_filters=config.conditioner.graph_filters,
            graph_convolutional_layers=config.conditioner.graph_convolution_layers,
            concat_mol_to_atom_features=True,
            pooling=config.conditioner.pooling,
            graph_norm=config.conditioner.graph_norm,
            num_spherical=config.conditioner.num_spherical,
            num_radial=config.conditioner.num_radial,
            graph_convolution=config.conditioner.graph_convolution,
            num_attention_heads=config.conditioner.num_attention_heads,
            add_spherical_basis=config.conditioner.add_spherical_basis,
            add_torsional_basis=config.conditioner.add_torsional_basis,
            graph_embedding_size=config.conditioner.atom_embedding_size,
            radial_function=config.conditioner.radial_function,
            max_num_neighbors=config.conditioner.max_num_neighbors,
            convolution_cutoff=config.conditioner.graph_convolution_cutoff,
            positional_embedding=config.conditioner.positional_embedding,
            max_molecule_size=config.max_molecule_radius,
        )

        '''
        generator model
        '''
        self.model = general_MLP(layers=config.generator.num_fc_layers,
                                 filters=config.generator.fc_depth,
                                 norm=config.generator.fc_norm_mode,
                                 dropout=config.generator.fc_dropout_probability,
                                 input_dim=self.latent_dim,
                                 output_dim=dataDims['num lattice features'],
                                 conditioning_dim=config.conditioner.fc_depth,
                                 seed=config.seeds.model
                                 )

    def sample_latent(self, n_samples):
        # return torch.ones((n_samples,12)).to(self.device) # when we don't actually want any noise (test purposes)
        return self.prior.sample((n_samples,)).to(self.device)

    def forward(self, n_samples, z=None, conditions=None, return_latent=False, return_condition=False, return_prior=False):
        if z is None:  # sample random numbers from simple prior
            z = self.sample_latent(n_samples)

        if conditions is not None:  # conditions here is a crystal data object
            normed_coords = conditions.pos / self.conditioner.max_molecule_size  # norm coords by maximum molecule radius
            conditions.x = torch.cat((conditions.x[:, :-self.crystal_features_to_ignore], normed_coords), dim=-1)  # concatenate to input features
            conditions_encoding = self.conditioner(conditions)
        else:
            conditions_encoding = None

        # run through model
        if any((return_condition, return_prior, return_latent)):
            output = [self.model(z, conditions=conditions_encoding, return_latent=return_latent)]
            if return_prior:
                output.append(z)
            if return_condition:
                output.append(conditions_encoding)
            return output
        else:
            return self.model(z, conditions=conditions_encoding, return_latent=return_latent)

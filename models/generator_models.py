import torch
import torch.nn as nn
import sys
from torch.distributions import MultivariateNormal, Uniform
from models.torch_models import molecule_graph_model, independent_gaussian_model
from models.model_components import general_MLP
from crystal_building.crystal_builder_tools import align_crystaldata_to_principal_axes


class crystal_generator(nn.Module):
    def __init__(self, config, dataDims):
        super(crystal_generator, self).__init__()

        self.device = config.device
        self.conditioning_mode = config.generator.conditioning_mode
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
        if self.conditioning_mode == 'graph model':  # molecular graph model
            self.conditioner = molecule_graph_model(
                dataDims,
                seed=config.seeds.model,
                num_atom_feats=dataDims['num atom features'],
                num_mol_feats=dataDims['num mol features'],
                output_dimension=config.generator.fc_depth,
                activation=config.generator.conditioner_activation,
                num_fc_layers=config.generator.conditioner_num_fc_layers,
                fc_depth=config.generator.conditioner_fc_depth,
                fc_dropout_probability=config.generator.conditioner_fc_dropout_probability,
                fc_norm_mode=config.generator.conditioner_fc_norm_mode,
                graph_model=config.generator.graph_model,
                graph_filters=config.generator.graph_filters,
                graph_convolutional_layers=config.generator.graph_convolution_layers,
                concat_mol_to_atom_features=True,
                pooling=config.generator.pooling,
                graph_norm=config.generator.graph_norm,
                num_spherical=config.generator.num_spherical,
                num_radial=config.generator.num_radial,
                graph_convolution=config.generator.graph_convolution,
                num_attention_heads=config.generator.num_attention_heads,
                add_spherical_basis=config.generator.add_spherical_basis,
                add_torsional_basis=config.generator.add_torsional_basis,
                atom_embedding_size=config.generator.atom_embedding_size,
                radial_function=config.generator.radial_function,
                max_num_neighbors=config.generator.max_num_neighbors,
                convolution_cutoff=config.generator.graph_convolution_cutoff,
            )
        elif self.conditioning_mode == 'molecule features':
            self.conditioner = general_MLP(layers=config.generator.conditioner_num_fc_layers,
                                           filters=config.generator.conditioner_fc_depth,
                                           norm=config.generator.conditioner_fc_norm_mode,
                                           dropout=config.generator.conditioner_fc_dropout_probability,
                                           input_dim=dataDims['num conditional features'],
                                           output_dim=config.generator.fc_depth,
                                           conditioning_dim=0,
                                           seed=config.seeds.model
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
                                 conditioning_dim=config.generator.fc_depth,
                                 seed=config.seeds.model
                                 )


    def sample_latent(self, n_samples):
        # return torch.ones((n_samples,12)).to(self.device) # when we don't actually want any noise (test purposes)
        return self.prior.sample((n_samples,)).to(self.device)

    def forward(self, n_samples, z=None, conditions=None, return_latent=False, return_condition=False, return_prior=False):
        if z is None:  # sample random numbers from simple prior
            z = self.sample_latent(n_samples)

        if conditions is not None:
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


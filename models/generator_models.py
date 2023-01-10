import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from nflib.flows import Invertible1x1Conv
from nflib.spline_flows import NSF_CL
from torch.distributions import MultivariateNormal, Uniform
import itertools
from models.torch_models import molecule_graph_model, independent_gaussian_model
from models.model_components import general_MLP, Normalization


class crystal_generator(nn.Module):
    def __init__(self, config, dataDims):
        super(crystal_generator, self).__init__()

        self.device = config.device
        self.generator_model_type = config.generator.model_type

        if config.generator.prior == 'multivariate normal':
            self.prior = MultivariateNormal(torch.zeros(dataDims['num lattice features']), torch.eye(dataDims['num lattice features']))
        elif config.generator.prior.lower() == 'uniform':
            self.prior = Uniform(low=0, high=1)
        else:
            print(config.generator.prior + ' is not an implemented prior!!')
            sys.exit()
        '''
        conditioning model
        '''
        if config.generator.conditioning_mode == 'graph model':  # molecular graph model
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
        elif config.generator.conditioning_mode == 'molecule features':
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
        if self.generator_model_type.lower() == 'mlp':  # simple MLP
            self.model = general_MLP(layers=config.generator.num_fc_layers,
                                     filters=config.generator.fc_depth,
                                     norm=config.generator.fc_norm_mode,
                                     dropout=config.generator.fc_dropout_probability,
                                     input_dim=dataDims['num lattice features'],
                                     output_dim=dataDims['num lattice features'],
                                     conditioning_dim=config.generator.fc_depth,
                                     seed=config.seeds.model
                                     )

        elif self.generator_model_type.lower() == 'fit normal':
            assert config.generator.prior.lower() == 'multivariate normal'
            self.model = independent_gaussian_model(config, dataDims, dataDims['lattice means'], dataDims['lattice stds'])
        else:
            print(self.generator_model_type + ' is not an implemented generator model!')
            sys.exit()

    def sample_latent(self, n_samples):
        # return torch.ones((n_samples,12)).to(self.device) # when we don't actually want any noise (test purposes)
        return self.prior.sample((n_samples,)).to(self.device)

    def forward(self, n_samples, z=None, conditions=None, return_latent=False, return_condition=False, return_prior=False):
        if z is None:  # sample random numbers from simple prior
            z = self.sample_latent(n_samples)
            # z = torch.zeros_like(z0)

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





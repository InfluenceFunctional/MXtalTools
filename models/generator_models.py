import torch
import torch.nn as nn
import sys
from utils import get_strides
from torch.distributions import MultivariateNormal, Uniform
from models.torch_models import molecule_graph_model, PointCloudDecoder
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
            self.crystal_features_to_ignore = config.dataDims['num crystal generation features']

            self.conditioner = molecule_graph_model(
                dataDims,
                seed=config.seeds.model,
                num_atom_feats=dataDims['num atom features'] + 3 - self.crystal_features_to_ignore,  # we will add directly the normed coordinates to the node features
                num_mol_feats=dataDims['num mol features'] - self.crystal_features_to_ignore,
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
                graph_embedding_size=config.generator.atom_embedding_size,
                radial_function=config.generator.radial_function,
                max_num_neighbors=config.generator.max_num_neighbors,
                convolution_cutoff=config.generator.graph_convolution_cutoff,
                positional_embedding = config.generator.positional_embedding,
                max_molecule_size=config.max_molecule_radius,
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

        if conditions is not None: # conditions here is a crystal data object
            normed_coords = conditions.pos / self.conditioner.max_molecule_size # norm coords by maximum molecule radius
            conditions.x = torch.cat((conditions.x[:,-self.crystal_features_to_ignore],normed_coords),dim=-1) # concatenate to input features
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




class molecule_autoencoder(nn.Module):
    def __init__(self, config, dataDims):
        super(molecule_autoencoder, self).__init__()

        self.device = config.device
        torch.manual_seed(config.seeds.model)

        '''
        conditioning model
        '''
        self.crystal_features_to_ignore = config.dataDims['num crystal generation features']
        conv_embedding_dim = config.generator.decoder_embedding_dim
        self.conditioner = molecule_graph_model(
            dataDims=dataDims,
            atom_embedding_dims = len(config.conditioner_classes) + 1,
            seed=config.seeds.model,
            num_atom_feats=dataDims['num atom features'] + 3 - self.crystal_features_to_ignore, # we will add directly the normed coordinates to the node features
            num_mol_feats=dataDims['num mol features'] - self.crystal_features_to_ignore,
            output_dimension=conv_embedding_dim * 3 * 3 * 3, # starting size for decoder model
            activation=config.generator.conditioner_activation,
            num_fc_layers=config.generator.conditioner_num_fc_layers,
            fc_depth=config.generator.conditioner_fc_depth,
            fc_dropout_probability=config.generator.conditioner_fc_dropout_probability,
            fc_norm_mode=config.generator.conditioner_fc_norm_mode,
            graph_model=config.generator.graph_model,
            graph_filters=config.generator.graph_filters,
            graph_convolutional_layers=config.generator.graph_convolution_layers,
            concat_mol_to_atom_features=False,
            pooling=config.generator.pooling,
            graph_norm=config.generator.graph_norm,
            num_spherical=config.generator.num_spherical,
            num_radial=config.generator.num_radial,
            graph_convolution=config.generator.graph_convolution,
            num_attention_heads=config.generator.num_attention_heads,
            add_spherical_basis=config.generator.add_spherical_basis,
            add_torsional_basis=config.generator.add_torsional_basis,
            graph_embedding_size=config.generator.atom_embedding_size,
            radial_function=config.generator.radial_function,
            max_num_neighbors=config.generator.max_num_neighbors,
            convolution_cutoff=config.generator.graph_convolution_cutoff,
            positional_embedding = config.generator.positional_embedding,
            max_molecule_size=1,
            crystal_mode=False,
            crystal_convolution_type= None,
            skip_mlp = False
        )


        '''
        generator model
        common atom types
        '''
        # stride 4 adds 3N - 1
        # stride 3 adds 2N
        # stride 2 adds N+1
        # stride 1 adds 2
        # n_target_bins = int((config.max_molecule_radius) * 2 / config.generator.autoencoder_resolution) + 1
        # strides = [2,2,2] # that brings it to 30 3-7-15-31,
        # current_size = 29
        # if n_target_bins < current_size:
        #     strides = [2,2]
        #     current_size = 13
        # if n_target_bins < current_size:
        #     strides = [2]
        #     current_size = 5
        #
        # diff = n_target_bins - current_size
        # for _ in range(diff//2): # must be an even number of bins in this approach
        #     strides += [1] # pad up to the required layers

        n_target_bins = int((config.max_molecule_radius) * 2 / config.generator.autoencoder_resolution)
        strides, final_image_size = get_strides(n_target_bins) # automatically find the right number of strides within 4-5 steps (minimizes overall stack depth)

        self.decoder = PointCloudDecoder(input_filters = conv_embedding_dim,
                                         n_classes = len(config.conditioner_classes) + 1,
                                         strides = strides)

        self.mlp = general_MLP(input_dim = conv_embedding_dim * 3 * 3 * 3,
                               layers = 2,
                               output_dim = 1,
                               filters = config.generator.conditioner_fc_depth,
                               norm = config.generator.conditioner_fc_norm_mode,
                               dropout = config.generator.conditioner_fc_dropout_probability,
                               activation = 'leaky relu')



    def forward(self, data):
        normed_coords = data.pos / self.conditioner.max_molecule_size # norm coords by maximum molecule radius
        data.x = torch.cat((data.x[:,:-self.crystal_features_to_ignore],normed_coords),dim=-1) # concatenate position to input features

        conditions_encoding = self.conditioner(data)
        return self.decoder(conditions_encoding), self.mlp(conditions_encoding) # return decoder and regression target

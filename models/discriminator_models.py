import torch.nn as nn
from models.base_models import molecule_graph_model
import torch


class crystal_discriminator(nn.Module):
    def __init__(self, config, discrim_config, dataDims):
        '''
        wrapper for molecule model, with appropriate I/O
        '''
        torch.manual_seed(config.seeds.model)

        super(crystal_discriminator, self).__init__()
        self.model = molecule_graph_model(
            dataDims=dataDims,
            seed=config.seeds.model,
            num_atom_feats=dataDims['num_atom_features'] - dataDims['num crystal generation features'],
            num_mol_feats=dataDims['num_mol_features'] - dataDims['num crystal generation features'],
            output_dimension=2,  # 'yes' and 'no'
            activation=discrim_config.activation,
            num_fc_layers=discrim_config.num_fc_layers,
            fc_depth=discrim_config.fc_depth,
            fc_dropout_probability=discrim_config.fc_dropout_probability,
            fc_norm_mode=discrim_config.fc_norm_mode,
            graph_filters=discrim_config.graph_filters,
            graph_convolutional_layers=discrim_config.graph_convolution_layers,
            concat_mol_to_atom_features=True,
            pooling=discrim_config.pooling,
            graph_norm=discrim_config.graph_norm,
            num_spherical=discrim_config.num_spherical,
            num_radial=discrim_config.num_radial,
            graph_convolution=discrim_config.graph_convolution,
            num_attention_heads=discrim_config.num_attention_heads,
            add_spherical_basis=discrim_config.add_spherical_basis,
            add_torsional_basis=discrim_config.add_torsional_basis,
            graph_embedding_size=discrim_config.atom_embedding_size,
            radial_function=discrim_config.radial_function,
            max_num_neighbors=discrim_config.max_num_neighbors,
            convolution_cutoff=discrim_config.graph_convolution_cutoff,
            crystal_mode=True,
            device=config.device,
            crystal_convolution_type=discrim_config.crystal_convolution_type,
            max_molecule_size=config.max_molecule_radius,
        )
        self.crystal_features_to_ignore = config.dataDims['num crystal generation features']

    def forward(self, data, return_dists=False, return_latent=False):
        data.x = data.x[:, :-self.crystal_features_to_ignore]  # leave out the trailing N features, which give information on the crystal lattice
        return self.model(data, return_dists=return_dists, return_latent=return_latent)

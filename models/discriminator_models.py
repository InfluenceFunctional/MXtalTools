import torch.nn as nn
from models.torch_models import molecule_graph_model


class crystal_discriminator(nn.Module):
    def __init__(self, config, dataDims):
        '''
        wrapper for molecule model, with appropriate I/O
        '''
        super(crystal_discriminator, self).__init__()
        self.model = molecule_graph_model(
            dataDims=dataDims,
            seed=config.seeds.model,
            num_atom_feats=dataDims['num atom features'] - dataDims['num crystal generation features'],
            num_mol_feats=dataDims['num mol features'] - dataDims['num crystal generation features'],
            output_dimension=2, # 'yes' and 'no'
            activation=config.discriminator.activation,
            num_fc_layers=config.discriminator.num_fc_layers,
            fc_depth=config.discriminator.fc_depth,
            fc_dropout_probability=config.discriminator.fc_dropout_probability,
            fc_norm_mode=config.discriminator.fc_norm_mode,
            graph_model=config.discriminator.graph_model,
            graph_filters=config.discriminator.graph_filters,
            graph_convolutional_layers=config.discriminator.graph_convolution_layers,
            concat_mol_to_atom_features=True,
            pooling=config.discriminator.pooling,
            graph_norm=config.discriminator.graph_norm,
            num_spherical=config.discriminator.num_spherical,
            num_radial=config.discriminator.num_radial,
            graph_convolution=config.discriminator.graph_convolution,
            num_attention_heads=config.discriminator.num_attention_heads,
            add_spherical_basis=config.discriminator.add_spherical_basis,
            add_torsional_basis=config.discriminator.add_torsional_basis,
            atom_embedding_size=config.discriminator.atom_embedding_size,
            radial_function=config.discriminator.radial_function,
            max_num_neighbors=config.discriminator.max_num_neighbors,
            convolution_cutoff=config.discriminator.graph_convolution_cutoff,
            crystal_mode=True,
            device=config.device,
            crystal_convolution_type = config.discriminator.crystal_convolution_type,
            max_molecule_size=config.max_molecule_radius,
        )
        self.crystal_features_to_ignore = config.dataDims['num crystal generation features']

    def forward(self, data, return_dists=False, return_latent=False):
        data.x = data.x[:,:-self.crystal_features_to_ignore] # leave out the trailing N features, which give information on the crystal lattice
        return self.model(data, return_dists=return_dists, return_latent=return_latent)

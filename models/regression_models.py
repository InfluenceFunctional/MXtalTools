import torch.nn as nn
from models.base_models import molecule_graph_model


class molecule_regressor(nn.Module):
    def __init__(self, config, regressor_config, dataDims):
        '''
        wrapper for molecule model, with appropriate I/O
        '''
        super(molecule_regressor, self).__init__()
        self.model = molecule_graph_model(
            dataDims,
            seed=config.seeds.model,
            num_atom_feats=dataDims['num atom features'] - dataDims['num crystal generation features'],
            num_mol_feats=dataDims['num mol features'] - dataDims['num crystal generation features'],
            output_dimension=1,  # single-target regression
            activation=regressor_config.activation,
            num_fc_layers=regressor_config.num_fc_layers,
            fc_depth=regressor_config.fc_depth,
            fc_dropout_probability=regressor_config.fc_dropout_probability,
            fc_norm_mode=regressor_config.fc_norm_mode,
            graph_filters=regressor_config.graph_filters,
            graph_convolutional_layers=regressor_config.graph_convolution_layers,
            concat_mol_to_atom_features=True,
            pooling=regressor_config.pooling,
            graph_norm=regressor_config.graph_norm,
            num_spherical=regressor_config.num_spherical,
            num_radial=regressor_config.num_radial,
            graph_convolution=regressor_config.graph_convolution,
            num_attention_heads=regressor_config.num_attention_heads,
            add_spherical_basis=regressor_config.add_spherical_basis,
            add_torsional_basis=regressor_config.add_torsional_basis,
            graph_embedding_size=regressor_config.atom_embedding_size,
            radial_function=regressor_config.radial_function,
            max_num_neighbors=regressor_config.max_num_neighbors,
            convolution_cutoff=regressor_config.graph_convolution_cutoff,
            device=config.device,
            max_molecule_size=config.max_molecule_radius,
        )
        self.crystal_features_to_ignore = config.dataDims['num crystal generation features']

    def forward(self, data, return_dists=False, return_latent=False):
        data.x = data.x[:, :-self.crystal_features_to_ignore]  # leave out the trailing N features, which give information on the crystal lattice and packing coefficient
        return self.model(data, return_dists=return_dists, return_latent=return_latent)

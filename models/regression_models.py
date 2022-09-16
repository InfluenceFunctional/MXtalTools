import torch.nn as nn
from models.torch_models import molecule_graph_model


class molecule_regressor(nn.Module):
    def __init__(self, config, dataDims):
        '''
        wrapper for molecule model, with appropriate I/O
        '''
        super(molecule_regressor, self).__init__()
        self.model = molecule_graph_model(
            dataDims,
            seed=config.seeds.model,
            output_dimension=1,  # single-target regression
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
            add_radial_basis=config.generator.add_radial_basis,
            atom_embedding_size=config.generator.atom_embedding_size,
            radial_function=config.generator.radial_function,
            max_num_neighbors=config.generator.max_num_neighbors,
            convolution_cutoff=config.generator.graph_convolution_cutoff,
            device=config.device,
        )
        self.crystal_features_to_ignore = config.dataDims['num crystal generation features']

    def forward(self, data, return_dists=False, return_latent=False):
        #data.x = data.x[:, :-self.crystal_features_to_ignore]  # leave out the trailing N features, which give information on the crystal lattice
        return self.model(data, return_dists=return_dists, return_latent=return_latent)

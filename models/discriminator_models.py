import torch.nn as nn
from models.base_models import molecule_graph_model
import torch


class crystal_discriminator(nn.Module):
    def __init__(self, seed, config, dataDims):
        '''
        wrapper for molecule model, with appropriate I/O
        '''
        torch.manual_seed(config.seeds.model)

        super(crystal_discriminator, self).__init__()
        self.model = molecule_graph_model(
            dataDims=dataDims,
            seed=seed,
            num_atom_feats=dataDims['num_atom_features'] - dataDims['num crystal generation features'],
            num_mol_feats=dataDims['num_molecule_features'] - dataDims['num crystal generation features'],
            output_dimension=2,  # 'yes' and 'no'
            activation=config.activation,
            num_fc_layers=config.num_fc_layers,
            fc_depth=config.fc_depth,
            fc_dropout_probability=config.fc_dropout_probability,
            fc_norm_mode=config.fc_norm_mode,
            graph_message_depth=config.graph_filters,
            graph_convolutional_layers=config.graph_convolution_layers,
            concat_mol_to_atom_features=True,
            graph_aggregator=config.pooling,
            graph_norm=config.graph_norm,
            num_spherical=config.num_spherical,
            num_radial=config.num_radial,
            graph_convolution_type=config.graph_convolution,
            num_attention_heads=config.num_attention_heads,
            add_spherical_basis=config.add_spherical_basis,
            add_torsional_basis=config.add_torsional_basis,
            graph_node_dims=config.atom_embedding_size,
            radial_function=config.radial_function,
            max_num_neighbors=config.max_num_neighbors,
            convolution_cutoff=config.graph_convolution_cutoff,
            crystal_mode=True,
            device=config.device,
            crystal_convolution_type=config.crystal_convolution_type,
            max_molecule_size=config.max_molecule_radius,
        )

    def forward(self, data, return_dists=False, return_latent=False):
        return self.model(data, return_dists=return_dists, return_latent=return_latent)
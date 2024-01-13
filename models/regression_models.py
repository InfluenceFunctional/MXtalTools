import torch.nn as nn
from models.base_models import molecule_graph_model


class molecule_regressor(nn.Module):
    def __init__(self, seed, config, dataDims):
        """
        wrapper for molecule model, with appropriate I/O
        """
        super(molecule_regressor, self).__init__()
        self.model = molecule_graph_model(
            num_atom_feats=dataDims['num_atom_features'],
            num_mol_feats=dataDims['num_molecule_features'],
            output_dimension=1,
            seed=seed,
            graph_aggregator=config.graph_aggregator,
            concat_pos_to_atom_features=False,
            concat_mol_to_atom_features=config.concat_mol_to_atom_features,
            concat_crystal_to_atom_features=False,
            activation=config.activation,
            num_fc_layers=config.num_fc_layers,
            fc_depth=config.fc_depth,
            fc_norm_mode=config.fc_norm_mode,
            fc_dropout_probability=config.fc_dropout_probability,
            graph_node_norm=config.graph_node_norm,
            graph_node_dropout=config.graph_node_dropout,
            graph_message_dropout=config.graph_message_dropout,
            num_attention_heads=config.num_attention_heads,
            graph_message_depth=config.graph_message_depth,
            graph_node_dims=config.graph_node_dims,
            num_graph_convolutions=config.num_graph_convolutions,
            graph_embedding_depth=config.graph_embedding_depth,
            nodewise_fc_layers=config.nodewise_fc_layers,
            num_radial=config.num_radial,
            radial_function=config.radial_function,
            max_num_neighbors=config.max_num_neighbors,
            convolution_cutoff=config.convolution_cutoff,
            atom_type_embedding_dims=config.atom_type_embedding_dims,
            periodic_structure=False,
            outside_convolution_type='none'
        )

    def forward(self, data, return_dists=False, return_latent=False):
        return self.model(data, return_dists=return_dists, return_latent=return_latent)

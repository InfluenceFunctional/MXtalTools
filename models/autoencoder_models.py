import torch
import torch.nn as nn

from models.base_models import molecule_graph_model
from models.components import MLP


class point_autoencoder(nn.Module):
    def __init__(self, seed, config, dataDims):
        super(point_autoencoder, self).__init__()
        '''conditioning model'''

        self.num_classes = dataDims['num_atom_types']
        self.output_depth = self.num_classes + 3 + 1
        self.num_nodes = config.num_decoder_points

        self.encoder = molecule_graph_model(
            num_atom_feats=1,
            num_mol_feats=0,
            output_dimension=config.embedding_depth,
            seed=seed,
            graph_convolution_type='TransformerConv',
            graph_aggregator=config.graph_aggregator,
            concat_pos_to_atom_features=True,
            concat_mol_to_atom_features=False,
            concat_crystal_to_atom_features=False,
            activation='gelu',
            num_fc_layers=0,
            fc_depth=0,
            fc_norm_mode=None,
            fc_dropout_probability=None,
            graph_node_norm=config.graph_node_norm,
            graph_node_dropout=config.graph_node_dropout,
            graph_message_norm=None,
            graph_message_dropout=config.graph_message_dropout,
            num_attention_heads=config.num_attention_heads,
            graph_message_depth=config.embedding_depth,
            graph_node_dims=config.embedding_depth,
            num_graph_convolutions=config.num_graph_convolutions,
            graph_embedding_depth=config.embedding_depth,
            nodewise_fc_layers=config.nodewise_fc_layers,
            num_radial=config.num_radial,
            radial_function=config.radial_function,
            max_num_neighbors=config.max_num_neighbors,
            convolution_cutoff=config.convolution_cutoff,
            atom_type_embedding_dims=config.atom_type_embedding_dims,
            periodic_structure=False,
            outside_convolution_type='none',
            cartesian_dimension=3,
        )

        # graph size model
        self.num_atoms_prediction = MLP(layers=2,
                                        filters=32,
                                        norm=None,
                                        dropout=0,
                                        input_dim=config.embedding_depth,
                                        output_dim=1,
                                        conditioning_dim=0,
                                        seed=seed,
                                        conditioning_mode=None,
                                        )

        self.composition_prediction = MLP(layers=2,
                                          filters=32,
                                          norm=None,
                                          dropout=0,
                                          input_dim=config.embedding_depth,
                                          output_dim=self.num_classes,
                                          conditioning_dim=0,
                                          seed=seed,
                                          conditioning_mode=None,
                                          )
        # swarm decoder
        self.decoder = MLP(
            layers=config.num_decoder_layers,
            filters=config.embedding_depth,
            input_dim=config.embedding_depth,
            output_dim=self.output_depth * self.num_nodes,
            activation='gelu',
            norm=config.decoder_norm_mode,
            dropout=config.decoder_dropout_probability,
        )

    def forward(self, data):
        """
        initialize nodes on randn with uniform embedding
        decode
        """
        encoding = self.encoder(data)

        return self.decoder(encoding).reshape(self.num_nodes * data.num_graphs, self.output_depth)

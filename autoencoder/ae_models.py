import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.graphgym.models import gnn

from models.GraphNeuralNetwork import GCBlock
from models.base_models import molecule_graph_model
from models.basis_functions import BesselBasisLayer, GaussianEmbedding
from models.components import MLP


class point_cloud_encoder(nn.Module):
    def __init__(self, cart_dimension, aggregator, embedding_depth, num_layers, num_fc_layers, num_nodewise_fcs, fc_norm, graph_norm, dropout, cutoff, seed, device, num_classes):
        super(point_cloud_encoder, self).__init__()

        self.device = device
        '''conditioning model'''
        torch.manual_seed(seed)

        self.conditioner = molecule_graph_model(
            num_atom_feats=1,
            num_mol_feats=0,
            output_dimension=embedding_depth,
            seed=seed,
            graph_convolution_type='TransformerConv',
            graph_aggregator=aggregator,
            concat_pos_to_atom_features=True,
            concat_mol_to_atom_features=False,
            concat_crystal_to_atom_features=False,
            activation='gelu',
            num_fc_layers=num_fc_layers,
            fc_depth=embedding_depth,
            fc_norm_mode=fc_norm,
            fc_dropout_probability=dropout,
            graph_node_norm=graph_norm,
            graph_node_dropout=dropout,
            graph_message_norm=None,
            graph_message_dropout=dropout,
            num_attention_heads=4,
            graph_message_depth=embedding_depth,
            graph_node_dims=embedding_depth,
            num_graph_convolutions=num_layers,
            graph_embedding_depth=embedding_depth,
            nodewise_fc_layers=num_nodewise_fcs,
            num_radial=50,
            radial_function='gaussian',
            max_num_neighbors=100,
            convolution_cutoff=cutoff,
            atom_type_embedding_dims=5,
            periodic_structure=False,
            outside_convolution_type='none',
            cartesian_dimension=cart_dimension,
        )

        # graph size model
        self.num_atoms_prediction = MLP(layers=1,
                                        filters=32,
                                        norm=None,
                                        dropout=0,
                                        input_dim=embedding_depth,
                                        output_dim=1,
                                        conditioning_dim=0,
                                        seed=seed,
                                        conditioning_mode=None,
                                        )

        self.composition_prediction = MLP(layers=1,
                                          filters=32,
                                          norm=None,
                                          dropout=0,
                                          input_dim=embedding_depth,
                                          output_dim=num_classes,
                                          conditioning_dim=0,
                                          seed=seed,
                                          conditioning_mode=None,
                                          )

    def forward(self, data):
        encoding = self.conditioner(data)
        num_atoms_prediction = self.num_atoms_prediction(encoding)
        composition_prediction = self.composition_prediction(encoding)

        return encoding, num_atoms_prediction, composition_prediction


class fc_decoder(nn.Module):
    def __init__(self, num_nodes, cart_dimension, input_depth, embedding_depth, num_layers, fc_norm, dropout, max_ntypes, seed, device):
        super(fc_decoder, self).__init__()
        self.cart_dimension = cart_dimension
        self.device = device
        self.output_depth = max_ntypes + cart_dimension
        torch.manual_seed(seed)
        self.num_nodes = num_nodes
        self.embedding_depth = embedding_depth

        self.MLP = MLP(
            layers=num_layers,
            filters=embedding_depth,
            input_dim=input_depth,
            output_dim=self.output_depth * self.num_nodes,
            activation='gelu',
            norm=fc_norm,
            dropout=dropout,
        )

    def forward(self, encoding, data, num_points):
        """
        initialize nodes on randn with uniform embedding
        decode
        """

        x = self.MLP(encoding)

        return x.reshape(self.num_nodes * data.num_graphs, self.output_depth)

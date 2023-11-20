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


class point_cloud_decoder(nn.Module):
    def __init__(self, cart_dimension, input_depth, embedding_depth, num_layers, num_nodewise_fcs, graph_norm, message_norm, dropout, cutoff, max_ntypes, seed, device):
        super(point_cloud_decoder, self).__init__()
        self.cart_dimension = cart_dimension
        self.device = device
        self.max_num_neighbors = 100
        self.cutoff = cutoff
        output_depth = max_ntypes
        torch.manual_seed(seed)

        radial_embedding = 'gaussian'
        num_radial = 50
        envelope_exponent: int = 5

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)

        if input_depth != embedding_depth:
            self.init_layer = nn.Linear(input_depth, embedding_depth)
        else:
            self.init_layer = nn.Identity()

        grid_radius = 1
        self.num_gridpoints = (2 * grid_radius + 1) ** cart_dimension
        self.grid_positions = torch.zeros((self.num_gridpoints, cart_dimension))  # initialize the translations in fractional coords
        i = 0
        if cart_dimension == 3:
            for xx in range(-grid_radius, grid_radius + 1):
                for yy in range(-grid_radius, grid_radius + 1):
                    for zz in range(-grid_radius, grid_radius + 1):
                        self.grid_positions[i] = torch.tensor((xx, yy, zz))
                        i += 1
        elif cart_dimension == 2:
            for xx in range(-grid_radius, grid_radius + 1):
                for yy in range(-grid_radius, grid_radius + 1):
                    self.grid_positions[i] = torch.tensor((xx, yy))
                    i += 1
        elif cart_dimension == 1:
            for xx in range(-grid_radius, grid_radius + 1):
                self.grid_positions[i] = torch.tensor((xx))
                i += 1

        self.upscale = nn.Linear(input_depth, input_depth * self.num_gridpoints)
        self.init_norm1 = nn.Identity()  # nn.LayerNorm(input_depth)
        self.init_norm2 = nn.Identity()  # nn.LayerNorm(embedding_depth)

        self.interaction_blocks = torch.nn.ModuleList([
            GCBlock(embedding_depth,
                    embedding_depth,
                    'TransformerConv',
                    num_radial,
                    norm=message_norm,
                    dropout=dropout,
                    heads=4,
                    )
            for _ in range(num_layers)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            MLP(
                layers=num_nodewise_fcs,
                filters=embedding_depth,
                input_dim=embedding_depth,
                output_dim=embedding_depth,
                activation='gelu',
                norm=graph_norm,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embedding_depth, output_depth)

    def get_geom_embedding(self, edge_index, pos):
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)

    def forward(self, encoding, data, graph_sizes):
        """
        initialize nodes on randn with uniform embedding
        decode
        """

        batch = data.batch
        data.pos = torch.randn_like(data.pos)
        grid = F.gelu(self.init_norm1(self.upscale(encoding).reshape(data.num_graphs, encoding.shape[1], self.num_gridpoints).permute(0, 2, 1)))

        grid = grid.cpu()
        data.x = torch.cat([gnn.knn_interpolate(grid[ind], self.grid_positions.cpu(), data.pos[data.batch == ind].cpu())
                            for ind in range(data.num_graphs)]).to(encoding.device)

        x = F.gelu(self.init_norm2(self.init_layer(data.x)))
        for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
            edge_index = gnn.radius_graph(data.pos, r=self.cutoff, batch=batch,
                                          max_num_neighbors=self.max_num_neighbors, flow='source_to_target')
            dist, rbf = self.get_geom_embedding(edge_index, data.pos)

            res = x.clone()
            x = convolution(x, rbf, edge_index, batch)
            x = fc(x, batch=batch)
            x = res + x

            data.pos += x[:, :self.cart_dimension]  # update positions with convolution results
            x[:, -self.cart_dimension:] = data.pos  # update position in note embedding

        return torch.cat((data.pos, self.output_layer(x)), dim=1)


class fc_decoder(nn.Module):
    def __init__(self, num_nodes, cart_dimension, input_depth, embedding_depth, num_layers, fc_norm, dropout, cutoff, max_ntypes, seed, device):
        super(fc_decoder, self).__init__()
        self.cart_dimension = cart_dimension
        self.device = device
        self.max_num_neighbors = 100
        self.cutoff = cutoff
        output_depth = max_ntypes + cart_dimension
        torch.manual_seed(seed)
        self.num_nodes = num_nodes
        self.embedding_depth = embedding_depth

        self.init_layer = nn.Linear(input_depth, embedding_depth * num_nodes)

        self.fc_blocks = torch.nn.ModuleList([
            MLP(
                layers=num_layers,
                filters=embedding_depth,
                input_dim=embedding_depth,
                output_dim=embedding_depth,
                activation='gelu',
                norm=fc_norm,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embedding_depth, output_depth)

    def forward(self, encoding, data, num_points):
        """
        initialize nodes on randn with uniform embedding
        decode
        """

        x = self.init_layer(encoding).reshape(data.num_graphs * self.num_nodes, self.embedding_depth)

        for block in self.fc_blocks:
            x = block(x)

        return self.output_layer(x)

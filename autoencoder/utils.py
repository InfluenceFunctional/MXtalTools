import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.models import gnn

from models.GraphNeuralNetwork import GCBlock
from models.base_models import molecule_graph_model
from models.basis_functions import GaussianEmbedding, BesselBasisLayer
from models.components import MLP
import torch_geometric.nn as gnn


def get_reconstruction_likelihood(data, decoded_data, sigma, dist_to_self=False):
    """
    compute the overlap of 3D gaussians centered on points in the target data
    with those in the predicted data. Each gaussian in the target should have an overlap totalling 1.

    do this independently for each class

    scale predicted points gaussian heights by their confidence in each class

    sigma must be significantly smaller than inter-particle distances in the target data
    """
    target_types = data.x[:, 0]

    if dist_to_self:
        target_probs = F.one_hot(data.x[:, 0], num_classes=5)[:, target_types].diag()
        dists = torch.cdist(data.pos, data.pos, p=2)  # n_targets x n_guesses

    else:
        dists = torch.cdist(data.pos, decoded_data.pos, p=2)  # n_targets x n_guesses
        target_probs = decoded_data.x[:, target_types].diag()

    overlap = torch.exp(-dists ** 2 / sigma)

    # scale all overlaps by the predicted confidence in each particle type
    scaled_overlap = overlap * target_probs

    # did this for all graphs combined, now split into graphwise components
    # todo accelerate with scatter
    return torch.cat([
        scaled_overlap[data.batch == ind][:, data.batch == ind].sum(1) for ind in range(data.num_graphs)
    ])


class point_cloud_encoder(nn.Module):
    def __init__(self, aggregator, embedding_depth, num_layers, num_fc_layers, num_nodewise_fcs, fc_norm, graph_norm, message_norm, dropout, cutoff, seed, device, num_classes):
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
            graph_message_norm=message_norm,
            graph_message_dropout=dropout,
            num_attention_heads=num_layers,
            graph_message_depth=embedding_depth // 2,
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
            outside_convolution_type='none'
        )

        # graph size model
        self.num_atoms_prediction = MLP(layers=1,
                                        filters=32,
                                        norm=None,
                                        dropout=0,
                                        input_dim=embedding_depth,
                                        output_dim=1,  # 3 extra dimensions for angle decoder
                                        conditioning_dim=0,  # include crystal information for the generator and the target packing coeff
                                        seed=seed,
                                        conditioning_mode=None,
                                        )

        self.composition_prediction = MLP(layers=1,
                                          filters=32,
                                          norm=None,
                                          dropout=0,
                                          input_dim=embedding_depth,
                                          output_dim=num_classes,  # 3 extra dimensions for angle decoder
                                          conditioning_dim=0,  # include crystal information for the generator and the target packing coeff
                                          seed=seed,
                                          conditioning_mode=None,
                                          )

    def forward(self, data):
        encoding = self.conditioner(data)
        num_atoms_prediction = self.num_atoms_prediction(encoding)
        composition_prediction = self.composition_prediction(encoding)

        return encoding, num_atoms_prediction, composition_prediction


class point_cloud_decoder(nn.Module):
    def __init__(self, input_depth, embedding_depth, num_layers, num_nodewise_fcs, graph_norm, message_norm, dropout, cutoff, max_ntypes, seed, device):
        super(point_cloud_decoder, self).__init__()

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

        self.upscale = nn.Linear(input_depth, input_depth * 27)
        self.init_norm1 = nn.LayerNorm(input_depth)
        self.init_norm2 = nn.LayerNorm(embedding_depth)

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

        self.grid_positions = torch.zeros((27, 3))  # initialize the translations in fractional coords
        i = 0
        for xx in range(-1, 1 + 1):
            for yy in range(-1, 1 + 1):
                for zz in range(-1, 1 + 1):
                    self.grid_positions[i] = torch.tensor((xx, yy, zz))
                    i += 1

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
        grid = F.gelu(self.init_norm1(self.upscale(encoding).reshape(data.num_graphs, encoding.shape[1], 27).permute(0, 2, 1)))

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

            data.pos += x[:, :3]  # update positions with convolution results
            x[:, -3:] = data.pos  # update position in note embedding

        return torch.cat((data.pos, self.output_layer(x)), dim=1)


def load_checkpoint(path, encoder, decoder, optimizer):
    checkpoint = torch.load(path)
    if list(checkpoint['encoder_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['encoder_state_dict']):
            checkpoint['encoder_state_dict'][i[7:]] = checkpoint['encoder_state_dict'].pop(i)
    if list(checkpoint['decoder_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['decoder_state_dict']):
            checkpoint['decoder_state_dict'][i[7:]] = checkpoint['decoder_state_dict'].pop(i)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return encoder, decoder, optimizer
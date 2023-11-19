import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.models import gnn

from autoencoder.reporting import update_losses
from models.GraphNeuralNetwork import GCBlock
from models.base_models import molecule_graph_model
from models.basis_functions import GaussianEmbedding, BesselBasisLayer
from models.components import MLP
import torch_geometric.nn as gnn
from torch_scatter import scatter


def compute_loss(losses, config, working_sigma, num_points_prediction, composition_prediction, decoding, data, point_num_rands):

    graph_weights = point_num_rands / config.num_fc_nodes
    nodewise_weights = graph_weights.repeat(config.num_fc_nodes)

    decoded_data = data.clone()
    decoded_data.pos = decoding[:, :config.cart_dimension]
    decoded_data.x = F.softmax(decoding[:, config.cart_dimension:], dim=1) * torch.tensor(nodewise_weights, dtype=torch.float32, device=config.device)[:, None]
    decoded_data.batch = torch.arange(data.num_graphs).repeat_interleave(config.num_fc_nodes).to(config.device)

    true_nodes = torch.cat([F.one_hot(data.x[data.batch == i, 0], num_classes=config.max_point_types).float() for i in range(data.num_graphs)])
    per_graph_true_types = torch.stack([true_nodes[data.batch == i].float().mean(0) for i in range(data.num_graphs)])
    per_graph_pred_types = torch.stack([decoded_data.x[decoded_data.batch == i].sum(0) for i in range(data.num_graphs)]) / torch.tensor(point_num_rands, dtype=torch.float32, device=config.device)[:, None]

    decoder_likelihoods = get_reconstruction_likelihood(data, decoded_data, working_sigma, overlap_type=config.overlap_type, num_classes=config.max_point_types, log_scale=config.log_reconstruction)
    self_likelihoods = get_reconstruction_likelihood(data, data, working_sigma, overlap_type=config.overlap_type, num_classes=config.max_point_types, log_scale=config.log_reconstruction, dist_to_self=True)  # if sigma is too large, these can be > 1

    encoding_type_loss = F.binary_cross_entropy_with_logits(composition_prediction, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)  # subtract out minimum
    num_points_loss = F.mse_loss(torch.Tensor(point_num_rands).to(config.device), num_points_prediction[:, 0])

    nodewise_type_loss = F.binary_cross_entropy(per_graph_pred_types, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)
    # type_confidence_loss = torch.prod(decoded_data.x, dim=1).mean()  # probably better but sometimes unstable
    reconstruction_loss = torch.mean(scatter(F.smooth_l1_loss(decoder_likelihoods, self_likelihoods, reduction='none'), data.batch, reduce='mean'))  # overlaps should all be exactly 1

    centroid_dists = torch.linalg.norm(data.pos, dim=1)
    centroid_dists_means = torch.stack([centroid_dists[data.batch == i].mean() for i in range(data.num_graphs)])
    centroid_dists_stds = torch.stack([centroid_dists[data.batch == i].std() for i in range(data.num_graphs)])

    decoded_centroid_dists = torch.linalg.norm(decoded_data.pos, dim=1)
    decoded_centroid_dists_means = torch.stack([decoded_centroid_dists[decoded_data.batch == i].mean() for i in range(data.num_graphs)])
    decoded_centroid_dists_stds = torch.stack([decoded_centroid_dists[decoded_data.batch == i].std() for i in range(data.num_graphs)])

    centroid_dist_loss = F.smooth_l1_loss(decoded_centroid_dists_means, centroid_dists_means)
    centroid_std_loss = F.smooth_l1_loss(decoded_centroid_dists_stds, centroid_dists_stds)

    type_confidence_loss = torch.mean(-torch.log(torch.amax(decoded_data.x, dim=1)))

    loss_list = []
    if config.train_nodewise_type_loss:
        loss_list.append(nodewise_type_loss)
    if config.train_reconstruction_loss:
        loss_list.append(reconstruction_loss)
    if config.train_type_confidence_loss:
        loss_list.append(type_confidence_loss)
    if config.train_num_points_loss:
        loss_list.append(num_points_loss)
    if config.train_encoding_type_loss:
        loss_list.append(encoding_type_loss)
    if config.train_centroids_loss:
        loss_list.append(centroid_dist_loss)
        loss_list.append(centroid_std_loss)

    loss = torch.sum(torch.stack(loss_list))

    losses = update_losses(losses, num_points_loss, reconstruction_loss, encoding_type_loss,
                           working_sigma, type_confidence_loss, loss, nodewise_type_loss,
                           centroid_dist_loss, centroid_std_loss)

    return loss, losses, decoded_data, nodewise_weights


def get_reconstruction_likelihood(data, decoded_data, sigma, overlap_type, num_classes, dist_to_self=False, log_scale=False):
    """
    compute the overlap of ND gaussians centered on points in the target data
    with those in the predicted data. Each gaussian in the target should have an overlap totalling 1.

    do this independently for each class

    scale predicted points gaussian heights by their confidence in each class

    sigma must be significantly smaller than inter-particle distances in the target data
    """
    target_types = data.x[:, 0]

    if dist_to_self:
        dists = torch.cdist(data.pos, data.pos, p=2)  # n_targets x n_guesses
        target_probs = F.one_hot(data.x[:, 0], num_classes=num_classes)[:, target_types].diag()

    else:
        dists = torch.cdist(data.pos, decoded_data.pos, p=2)  # n_targets x n_guesses
        # target_probs = decoded_data.x[:, target_types].diag()
        target_probs = decoded_data.x[:, target_types]

    if overlap_type == 'gaussian':
        overlap = torch.exp(-(dists / sigma) ** 2)
    elif overlap_type == 'inverse':
        overlap = 1 / (dists / sigma + 1)
    elif overlap_type == 'exponential':
        overlap = torch.exp(-dists / sigma)
    else:
        assert False, f"{overlap_type} is not an implemented overlap function"

    # scale all overlaps by the predicted confidence in each particle type
    scaled_overlap = overlap * target_probs.T

    # did this for all graphs combined, now split into graphwise components
    # todo accelerate with scatter

    nodewise_overlap = torch.cat([
        scaled_overlap[data.batch == ind][:, decoded_data.batch == ind].sum(1) for ind in range(data.num_graphs)
    ])

    if log_scale:
        return torch.log(nodewise_overlap)
    else:
        return nodewise_overlap


class point_cloud_encoder(nn.Module):
    def __init__(self, cart_dimension, aggregator, embedding_depth, num_layers, num_fc_layers, num_nodewise_fcs, fc_norm, graph_norm, message_norm, dropout, cutoff, seed, device, num_classes):
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
    def __init__(self, num_nodes, cart_dimension, input_depth, embedding_depth, num_layers, num_nodewise_fcs, graph_norm, message_norm, dropout, cutoff, max_ntypes, seed, device):
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

    def forward(self, encoding, data, num_points):
        """
        initialize nodes on randn with uniform embedding
        decode
        """

        x = self.init_layer(encoding).reshape(data.num_graphs * self.num_nodes, self.embedding_depth)

        for block in self.fc_blocks:
            x = block(x)

        return self.output_layer(x)


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

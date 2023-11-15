from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.models import gnn

from models.GraphNeuralNetwork import EmbeddingBlock, GCBlock
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


def get_reconstruction_likelihood_discrete(smoother, data, decoded_data, num_particle_types, num_bins=100, hist_range=[-1, 1]):
    """
    compare predicted types & coordinates to 3D gaussian fields
    """
    real_particle_coords = data.pos
    real_particle_probs = F.one_hot(data.x[:, 0], num_particle_types).float()

    decoded_particle_coords = decoded_data.pos
    decoded_particle_probs = decoded_data.x

    smoothed_target = get_smoothed_density(real_particle_coords, real_particle_probs, smoother, hist_range, num_particle_types, data, num_bins)
    smoothed_guess = get_smoothed_density(decoded_particle_coords, decoded_particle_probs, smoother, hist_range, num_particle_types, decoded_data, num_bins)
    hist_overlap = torch.sum(torch.min(smoothed_target, smoothed_guess), dim=(1, 2, 3, 4)) / data.mol_size

    if torch.round(smoothed_target.sum()) != len(data.x) or torch.round(smoothed_guess.sum()) != len(data.x):
        print("Warning, density grid is either insufficiently large or insufficiently dense for given configs")

    return (-torch.log(hist_overlap.clip(min=1e-6))).mean()

    # return F.binary_cross_entropy(smoothed_guess, smoothed_target)


def get_smoothed_density(coords, probs, smoother, hist_range, num_particle_types, data, num_bins):
    particle_discrete_indices = torch.bucketize(coords, torch.linspace(hist_range[0], hist_range[1], num_bins + 1, device='cuda')) - 1

    # target distribution in discretized classwise space
    target = torch.zeros((data.num_graphs, num_bins, num_bins, num_bins, num_particle_types), dtype=torch.float32, device='cuda')
    for ii in range(data.num_graphs):
        target[
        ii,
        particle_discrete_indices[data.batch == ii, 0],
        particle_discrete_indices[data.batch == ii, 1],
        particle_discrete_indices[data.batch == ii, 2], :] = probs[data.batch == ii]

    # assert torch.abs(target.sum() - probs.sum()) < 1e-3, "Multiple Particles in a single bin, increase bin density!"

    # target = F.one_hot(target, num_classes=num_particle_types + 1).float()
    target = torch.permute(target, (0, 4, 1, 2, 3))

    pad = smoother.weight.shape[-1]  # // 2
    smoothed_target = torch.stack([F.pad(smoother(target[i]), (pad, pad, pad, pad, pad, pad), mode='constant') for i in range(data.num_graphs)])

    # assert torch.abs(smoothed_target.sum() - probs.sum()) < 1e-2

    return smoothed_target

    # viz target
    # fig = go.Figure()
    # sample_true = target[0].argmax(0).cpu().detach().numpy()
    # x, y, z = sample_true.nonzero()
    # fig.add_trace(go.Scatter3d(
    #     x=x, y=y, z=z,
    #     mode='markers',
    #     showlegend=True,
    #     marker=dict(
    #         size=10,
    #         color=sample_true[x, y, z],
    #         colorscale='Jet',
    #         cmin=0,
    #         opacity=0.5
    #     )))
    # fig.update_layout(showlegend=True)
    # fig.show()


def get_reconstruction_likelihood_old(data, decoding, sigma: float = 1):
    """
    compare predicted types & coordinates to 3D gaussian fields
    """
    cov_mat = torch.eye(3, dtype=torch.float, device=data.x.device) * sigma
    cov_det = torch.prod(torch.diag(cov_mat))  # true for diagonal covariance matrix
    cov_inv = torch.eye(3, dtype=torch.float, device=data.x.device) / sigma  # true for diagonal

    norm = (2 * torch.pi * torch.ones(1, dtype=torch.float, device=data.x.device)) ** (-3 / 2)
    norm2 = 1 / (torch.sqrt(cov_det)) * norm

    types_raw = decoding[:, 3:]
    types_prod = F.softmax(types_raw, dim=-1)
    pos_pred = decoding[:, :3]

    # for testing - this is the true data, should have a higher probability
    # types_prod = F.one_hot(data.x[:, 0], num_classes=5)
    # pos_pred = data.pos

    # gaussians centered on true locations - batch of all at once
    # below is the 3D gaussian for unit covariance
    point_density = lambda x, points: norm2 * torch.exp(-0.5 * torch.sum((x - points) ** 2, dim=1) * cov_inv[0, 0])  # valid for unit covariance

    graphwise_probs = [[] for _ in range(data.num_graphs)]
    graph_inds = []
    type_inds = []
    for graph_ind in range(data.num_graphs):
        graph_inds.append(data.batch == graph_ind)

    for type_ind in range(types_raw.shape[1]):
        type_inds.append(data.x[:, 0] == type_ind)

    for graph_ind in range(data.num_graphs):  # todo could no doubt be sped up
        for type_ind in range(types_raw.shape[1]):
            relevant_inds = graph_inds[graph_ind] * type_inds[type_ind]
            predicted_points = pos_pred[relevant_inds]
            predicted_types_probs = types_prod[relevant_inds, type_ind]

            true_points = data.pos[relevant_inds]
            true_probs = data.x[relevant_inds, 0]

            num_inds = torch.sum(relevant_inds)
            relevant_likelihoods = []
            for atom_ind in range(num_inds):  # todo should we also punish it explicitly for probability outisde where we want it?
                relevant_likelihoods.extend(point_density(predicted_points[atom_ind], true_points) * predicted_types_probs[atom_ind])

            graphwise_probs[graph_ind].extend(relevant_likelihoods)

        graphwise_probs[graph_ind] = torch.stack(graphwise_probs[graph_ind])

    combined_scores = torch.cat(graphwise_probs)

    return combined_scores


def test_get_reconstruction_loss_old(data, decoding):
    score1 = get_reconstruction_likelihood_old(data, decoding, sigma=0.1)
    true_encoding = torch.cat((
        data.pos, F.one_hot(data.x[:, 0], num_classes=decoding.shape[1] - 3)
    ), dim=1)
    score2 = get_reconstruction_likelihood_old(data, true_encoding, sigma=0.1)

    assert score2.mean() / score1.mean() > 10


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


class fc_decoder(nn.Module):
    def __init__(self, input_depth, embedding_depth, num_layers, num_nodewise_fcs, graph_norm, message_norm, dropout, cutoff, max_ntypes, seed, device):
        super(fc_decoder, self).__init__()

        self.device = device
        self.max_num_nodes = 10
        self.output_dim = max_ntypes + 3

        self.fc = MLP(
                layers=num_layers,
                filters=embedding_depth * self.max_num_nodes,
                input_dim=input_depth * self.max_num_nodes,
                output_dim=embedding_depth * self.max_num_nodes,
                activation='gelu',
                norm=graph_norm,
                dropout=dropout,
            )

        self.output_layer = nn.Linear(embedding_depth * self.max_num_nodes, (self.output_dim) * self.max_num_nodes)

    def forward(self, encoding, data, points_per_graph):
        """
        initialize nodes on randn with uniform embedding
        decode
        """

        x = encoding.repeat_interleave(self.max_num_nodes, 1)

        x = self.fc(x)

        x = self.output_layer(x)

        out = torch.cat([torch.stack([x[j, self.output_dim * i: self.output_dim * (i + 1)] for i in range(points_per_graph[j])]) for j in range(data.num_graphs)])
        return out


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
        self.init_norm2 = nn.LayerNorm(input_depth)

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

            x = convolution(x, rbf, edge_index, batch)
            x = fc(x, batch=batch)

            data.pos += x[:, :3]  # update positions with convolution results
            x[:, -3:] = data.pos  # update position in note embedding

        return torch.cat((data.pos, self.output_layer(x)), dim=1)


class old_point_cloud_decoder(nn.Module):
    def __init__(self, input_depth, embedding_depth, num_layers, num_nodewise_fcs, graph_norm, message_norm, dropout, cutoff, max_ntypes, seed, device):
        super(point_cloud_decoder, self).__init__()

        self.device = device

        self.conditioner = molecule_graph_model(
            num_atom_feats=input_depth,
            num_mol_feats=0,
            output_dimension=3 + max_ntypes,
            seed=seed,
            graph_convolution_type='TransformerConv',
            graph_aggregator=None,
            concat_pos_to_atom_features=True,
            concat_mol_to_atom_features=False,
            concat_crystal_to_atom_features=False,
            activation='gelu',
            num_fc_layers=0,
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
            atom_type_embedding_dims=1,
            periodic_structure=False,
            outside_convolution_type='none'
        )

    def forward(self, encoding, data, graph_sizes):
        """
        initialize nodes on randn with uniform embedding
        decode
        """
        data.pos = torch.randn_like(data.pos)
        data.x = encoding.repeat_interleave(torch.tensor(graph_sizes, dtype=torch.long, device=encoding.device), dim=0)
        data.x[:, 0] = 1  # to fool the encoder / so it won't throw wacky errors at us

        return self.conditioner(data)


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
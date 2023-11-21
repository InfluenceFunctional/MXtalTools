import torch
from torch import nn as nn
from torch.nn import functional as F

from autoencoder.ae_models import point_cloud_decoder
from models.base_models import molecule_graph_model

from models.components import MLP


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

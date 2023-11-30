import torch
import torch.nn.functional as F

from autoencoder.reporting import update_losses
from torch_scatter import scatter
from models.utils import compute_gaussian_overlap, split_reconstruction_likelihood


def compute_loss(wandb, step, losses, config, working_sigma, num_points_prediction, composition_prediction, decoding, data, point_num_rands):
    point_num_rands_tensor = torch.tensor(point_num_rands, dtype=torch.float32, device=config.device)[:, None]

    decoded_data = data.clone()
    decoded_data.pos = decoding[:, :config.cart_dimension]
    decoded_data.batch = torch.arange(data.num_graphs).repeat_interleave(config.num_decoder_points).to(config.device)
    if config.independent_node_weights:
        nodewise_weights_tensor = (torch.exp(decoding[:, -1]/config.node_weight_temperature) /
                                   scatter(torch.exp(decoding[:, -1]/config.node_weight_temperature), decoded_data.batch
                                           ).repeat_interleave(config.num_decoder_points))  # fast graph-wise softmax with high temperature
    else:
        graph_weights = point_num_rands / config.num_decoder_points
        nodewise_weights = graph_weights.repeat(config.num_decoder_points)  # should be repeat_interleave?
        nodewise_weights_tensor = torch.tensor(nodewise_weights, dtype=torch.float32, device=config.device)

    # low T causes instability in backprop
    decoded_data.x = F.softmax(decoding[:, config.cart_dimension:-1], dim=1) * nodewise_weights_tensor[:, None]

    true_nodes = F.one_hot(data.x[:, 0], num_classes=config.max_point_types).float()
    per_graph_true_types = scatter(true_nodes, data.batch[:, None], dim=0, reduce='mean')
    per_graph_pred_types = scatter(decoded_data.x, decoded_data.batch[:, None], dim=0, reduce='sum') / point_num_rands_tensor

    decoder_likelihoods = compute_gaussian_overlap(true_nodes, data, decoded_data, working_sigma, nodewise_weights=nodewise_weights_tensor,
                                                   overlap_type=config.overlap_type, log_scale=config.log_reconstruction,
                                                   type_distance_scaling=config.type_distance_scaling)

    self_likelihoods = compute_gaussian_overlap(true_nodes, data, data, working_sigma, nodewise_weights=torch.ones(data.num_nodes, dtype=torch.float32, device=config.device),
                                                overlap_type=config.overlap_type, log_scale=config.log_reconstruction,
                                                type_distance_scaling=config.type_distance_scaling, dist_to_self=True)  # if sigma is too large, these can be > 1, so we map to the overlap of the true density with itself

    encoding_type_loss = F.binary_cross_entropy_with_logits(composition_prediction, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)  # subtract out minimum
    num_points_loss = F.mse_loss(torch.Tensor(point_num_rands).to(config.device), num_points_prediction[:, 0])

    nodewise_type_loss = F.binary_cross_entropy(per_graph_pred_types, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)

    # todo check this reduces to a number
    reconstruction_loss = torch.mean(scatter(F.smooth_l1_loss(decoder_likelihoods, self_likelihoods, reduction='none'), data.batch, reduce='mean'))  # overlaps should all be exactly 1

    true_dists = torch.linalg.norm(data.pos, dim=1)
    mean_true_dist = scatter(true_dists, data.batch, dim=0, reduce='mean')

    decoded_dists = torch.linalg.norm(decoded_data.pos, dim=1)
    mean_decoded_dist = scatter(decoded_dists, decoded_data.batch, dim=0, reduce='mean')
    mean_dist_loss = F.smooth_l1_loss(mean_decoded_dist, mean_true_dist)

    constraining_loss = F.relu(decoded_dists - config.points_spread).mean()  # keep decoder points within the working volume

    loss_list = []
    if config.train_nodewise_type_loss:
        loss_list.append(nodewise_type_loss)
    if config.train_reconstruction_loss:
        loss_list.append(reconstruction_loss)
    if config.train_num_points_loss:
        loss_list.append(num_points_loss)
    if config.train_encoding_type_loss:
        loss_list.append(encoding_type_loss)
    if config.train_centroids_loss:
        loss_list.append(mean_dist_loss)
    if config.train_constraining_loss:
        loss_list.append(constraining_loss)

    loss = torch.sum(torch.stack(loss_list))

    losses = update_losses(losses, num_points_loss, reconstruction_loss, encoding_type_loss,
                           working_sigma, loss, nodewise_type_loss,
                           mean_dist_loss, constraining_loss, self_likelihoods)

    if step % 500 == 0:
        coord_overlap, type_overlap = split_reconstruction_likelihood(
            data, decoded_data, working_sigma, nodewise_weights=nodewise_weights_tensor,
            overlap_type=config.overlap_type, num_classes=config.max_point_types,
            type_distance_scaling=config.type_distance_scaling)

        self_coord_overlap, self_type_overlap = split_reconstruction_likelihood(
            data, data, working_sigma, nodewise_weights=torch.ones_like(data.x),
            overlap_type=config.overlap_type, num_classes=config.max_point_types,
            type_distance_scaling=config.type_distance_scaling, dist_to_self=True)

        wandb.log({"positions_wise_overlap": (coord_overlap / self_coord_overlap).mean().cpu().detach().numpy(),
                   "typewise_overlap": (type_overlap / self_type_overlap).mean().cpu().detach().numpy()})

    return loss, losses, decoded_data, nodewise_weights_tensor.cpu().detach().numpy(), (decoder_likelihoods / self_likelihoods).mean().cpu().detach().numpy()


def get_reconstruction_likelihood(data, decoded_data, sigma, overlap_type, num_classes, dist_to_self=False, log_scale=False):
    """
    compute the overlap of ND gaussians centered on points in the target data
    with those in the predicted data. Each gaussian in the target should have an overlap totalling 1.

    do this independently for each class

    scale predicted points gaussian heights by their confidence in each class

    sigma must be significantly smaller than inter-particle distances in the target data

    currently punishes distance issues much more severely than atom type issues
    """
    target_types = data.x[:, 0]

    if dist_to_self:
        dists = torch.cdist(data.pos, data.pos, p=2)  # n_targets x n_guesses
        target_probs = F.one_hot(data.x[:, 0], num_classes=num_classes)[:, target_types].float()

        # random_probs = F.softmax(torch.randn(len(data.x), num_classes, dtype=torch.float32, device=data.x.device))[:, target_types]

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

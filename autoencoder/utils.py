import torch
import torch.nn.functional as F

from autoencoder.reporting import update_losses
from torch_scatter import scatter
from math import sqrt


def compute_loss(wandb, step, losses, config, working_sigma, num_points_prediction, composition_prediction, decoding, data, point_num_rands):

    graph_weights = point_num_rands / config.num_fc_nodes
    nodewise_weights = graph_weights.repeat(config.num_fc_nodes)
    nodewise_weights_tensor = torch.tensor(nodewise_weights, dtype=torch.float32, device=config.device)

    decoded_data = data.clone()
    decoded_data.pos = decoding[:, :config.cart_dimension]
    decoded_data.x = F.softmax(decoding[:, config.cart_dimension:], dim=1) * nodewise_weights_tensor[:, None]
    decoded_data.batch = torch.arange(data.num_graphs).repeat_interleave(config.num_fc_nodes).to(config.device)

    true_nodes = torch.cat([F.one_hot(data.x[data.batch == i, 0], num_classes=config.max_point_types).float() for i in range(data.num_graphs)])
    per_graph_true_types = torch.stack([true_nodes[data.batch == i].float().mean(0) for i in range(data.num_graphs)])
    per_graph_pred_types = torch.stack([decoded_data.x[decoded_data.batch == i].sum(0) for i in range(data.num_graphs)]) / torch.tensor(point_num_rands, dtype=torch.float32, device=config.device)[:, None]

    decoder_likelihoods = high_dim_reconstruction_likelihood(data, decoded_data, working_sigma, nodewise_weights=nodewise_weights_tensor,
                                                             overlap_type=config.overlap_type, num_classes=config.max_point_types,
                                                             log_scale=config.log_reconstruction, type_distance_scaling=config.type_distance_scaling)
    self_likelihoods = high_dim_reconstruction_likelihood(data, data, working_sigma, nodewise_weights=torch.ones_like(data.x),
                                                          overlap_type=config.overlap_type, num_classes=config.max_point_types,
                                                          log_scale=config.log_reconstruction, type_distance_scaling=config.type_distance_scaling,
                                                          dist_to_self=True)  # if sigma is too large, these can be > 1

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
        loss_list.append(centroid_dist_loss)
        loss_list.append(centroid_std_loss)

    loss = torch.sum(torch.stack(loss_list))

    losses = update_losses(losses, num_points_loss, reconstruction_loss, encoding_type_loss,
                           working_sigma, loss, nodewise_type_loss,
                           centroid_dist_loss, centroid_std_loss)

    if step % 100 == 0:
        coord_overlap, type_overlap = split_reconstruction_likelihood(data, decoded_data, working_sigma, nodewise_weights=nodewise_weights_tensor,
                                                                      overlap_type=config.overlap_type, num_classes=config.max_point_types,
                                                                      type_distance_scaling=config.type_distance_scaling)
        self_coord_overlap, self_type_overlap = split_reconstruction_likelihood(data, data, working_sigma, nodewise_weights=torch.ones_like(data.x),
                                                                                overlap_type=config.overlap_type, num_classes=config.max_point_types,
                                                                                type_distance_scaling=config.type_distance_scaling, dist_to_self=True)
        wandb.log({"positions_wise_overlap": (coord_overlap / self_coord_overlap).mean().cpu().detach().numpy(),
                   "typewise_overlap": (type_overlap / self_type_overlap).mean().cpu().detach().numpy()})

    return loss, losses, decoded_data, nodewise_weights, (decoder_likelihoods/self_likelihoods).mean().cpu().detach().numpy()


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


def high_dim_reconstruction_likelihood(data, decoded_data, sigma, overlap_type, nodewise_weights, num_classes,
                                       dist_to_self=False, log_scale=False, type_distance_scaling=0.1):
    """
    same as previous version
    except atom type differences are treated as high dimensional distances
    """
    # num_types_scale_factor = sqrt(num_classes)  # should help rescale typewise distances to be more comparable in higher dimension
    ref_types = F.one_hot(data.x[:, 0], num_classes=num_classes) * type_distance_scaling  # / num_types_scale_factor  # rescale typewise distances to be smaller
    ref_points = torch.cat((data.pos, ref_types), dim=1)

    if dist_to_self:
        pred_points = ref_points
    else:
        pred_types = decoded_data.x / nodewise_weights[:, None] * type_distance_scaling  # / num_types_scale_factor
        pred_points = torch.cat((decoded_data.pos, pred_types), dim=1)  # assume input x has already been normalized

    dists = torch.cdist(ref_points, pred_points, p=2)  # n_targets x n_guesses

    if overlap_type == 'gaussian':
        overlap = torch.exp(-(dists / sigma) ** 2)
    elif overlap_type == 'inverse':
        overlap = 1 / (dists / sigma + 1)
    elif overlap_type == 'exponential':
        overlap = torch.exp(-dists / sigma)
    else:
        assert False, f"{overlap_type} is not an implemented overlap function"

    # scale all overlaps by the predicted confidence in each particle type
    scaled_overlap = overlap * nodewise_weights.T

    # did this for all graphs combined, now split into graphwise components
    # todo accelerate with scatter

    nodewise_overlap = torch.cat([
        scaled_overlap[data.batch == ind][:, decoded_data.batch == ind].sum(1) for ind in range(data.num_graphs)
    ])

    if log_scale:
        return torch.log(nodewise_overlap)
    else:
        return nodewise_overlap


def split_reconstruction_likelihood(data, decoded_data, sigma, overlap_type, nodewise_weights, num_classes,
                                    dist_to_self=False, type_distance_scaling=0.1):
    """
    same as previous version
    except atom type differences are treated as high dimensional distances
    """
    # num_types_scale_factor = sqrt(num_classes)  # should help rescale typewise distances to be more comparable in higher dimension
    ref_types = F.one_hot(data.x[:, 0], num_classes=num_classes) * type_distance_scaling  # / num_types_scale_factor  # rescale typewise distances to be smaller

    if dist_to_self:
        pred_types = ref_types
    else:
        pred_types = decoded_data.x / nodewise_weights[:, None] * type_distance_scaling  # / num_types_scale_factor

    d1 = torch.cdist(data.pos, decoded_data.pos, p=2)  # n_targets x n_guesses
    d2 = torch.cdist(ref_types, pred_types, p=2)

    if overlap_type == 'gaussian':
        o1 = torch.exp(-(d1 / sigma) ** 2)
        o2 = torch.exp(-(d2 / sigma) ** 2)
    elif overlap_type == 'inverse':
        o1 = 1 / (d1 / sigma + 1)
        o2 = 1 / (d2 / sigma + 1)

    elif overlap_type == 'exponential':
        o1 = torch.exp(-d1 / sigma)
        o2 = torch.exp(-d2 / sigma)

    else:
        assert False, f"{overlap_type} is not an implemented overlap function"

    # scale all overlaps by the predicted confidence in each particle type
    so1 = o1 * nodewise_weights.T
    so2 = o2 * nodewise_weights.T

    no1 = torch.cat([
        so1[data.batch == ind][:, decoded_data.batch == ind].sum(1) for ind in range(data.num_graphs)
    ])

    no2 = torch.cat([
        so2[data.batch == ind][:, decoded_data.batch == ind].sum(1) for ind in range(data.num_graphs)
    ])

    return no1, no2


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

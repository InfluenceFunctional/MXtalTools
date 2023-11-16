import os
from argparse import Namespace
import warnings

from autoencoder.reporting import log_rmsd_loss, log_losses, save_checkpoint, update_losses

warnings.filterwarnings("ignore", category=UserWarning)  # ignore numpy error
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR
from tqdm import tqdm
from dataset_management.CrystalData import CrystalData
from torch_geometric.loader.dataloader import Collater
from torch.optim import Adam
import argparse
from torch_scatter import scatter
import wandb
from autoencoder.utils import (
    point_cloud_encoder, point_cloud_decoder, get_reconstruction_likelihood, load_checkpoint)
from autoencoder.configs import dev, configs
from models.utils import check_convergence

os.environ["WANDB_START_METHOD"] = 'thread'

parser = argparse.ArgumentParser()
args = parser.parse_known_args()[1]

if '--config' in args:
    config = configs[int(args[1])]
else:
    config = dev

config = configs[26]

config = Namespace(**config)
batch_size = config.batch_size_min
working_sigma = config.sigma
mean_num_points = config.mean_num_points
os.chdir(config.run_directory)

converged = False

with (wandb.init(
        config=config.__dict__,
        project='PointAutoencoder',
        entity='mkilgour',
        tags=config.experiment_tag
)):
    wandb.run.name = config.run_name
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    encoder = point_cloud_encoder(aggregator=config.encoder_aggregator,
                                  embedding_depth=config.encoder_embedding_depth,
                                  num_fc_layers=config.encoder_num_fc_layers,
                                  num_layers=config.encoder_num_layers,
                                  num_nodewise_fcs=config.encoder_num_nodewise_fcs,
                                  fc_norm=config.encoder_fc_norm,
                                  graph_norm=config.encoder_graph_norm,
                                  message_norm=config.encoder_message_norm,
                                  dropout=config.encoder_dropout,
                                  cutoff=config.points_spread * 2,
                                  seed=config.seed, device=config.device,
                                  num_classes=config.point_types_max).to(config.device)

    decoder = point_cloud_decoder(input_depth=config.encoder_embedding_depth,
                                  embedding_depth=config.decoder_embedding_depth,
                                  num_layers=config.decoder_num_layers,
                                  num_nodewise_fcs=config.decoder_num_nodewise_fcs,
                                  graph_norm=config.decoder_graph_norm,
                                  message_norm=config.decoder_message_norm,
                                  dropout=config.decoder_dropout,
                                  cutoff=config.points_spread * 2,
                                  max_ntypes=config.point_types_max,
                                  seed=config.seed, device=config.device).to(config.device)

    collater = Collater(0, 0)
    optimizer = Adam([{'params': encoder.parameters(), 'lr': config.encoder_lr},
                      {'params': decoder.parameters(), 'lr': config.decoder_lr}], lr=(config.encoder_lr + config.decoder_lr) / 2)

    if config.checkpoint_path is not None:
        encoder, decoder, optimizer = load_checkpoint(config.checkpoint_path, encoder, decoder, optimizer)

    if not config.do_training:
        encoder.eval()
        decoder.eval()

    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: config.lr_lambda)
    wandb.watch((encoder, decoder), log_graph=True, log_freq=100)

    losses = {'reconstruction_loss': [],
              'num_points_loss': [],
              'overall_type_loss': [],
              'scaled_reconstruction_loss': [],
              'type_confidence_loss': [],
              'combined_loss': [],
              'nodewise_type_loss': [],
              'adjusted_nodewise_type_loss': [],
              'centroid_mean_loss': [],
              'centroid_std_loss': [],
              }

    for step in tqdm(range(config.training_iterations)):
        if converged:
            break

        if step % 10 == 0 and batch_size < config.batch_size_max:
            batch_size += config.batch_size_increment

        point_num_rands = np.clip(
            np.round((np.random.randn(batch_size) * config.num_points_spread + mean_num_points)),
            a_min=config.min_num_points, a_max=config.max_num_points).astype(int)

        coords_list = [torch.rand(rand, 3) * config.points_spread for rand in point_num_rands]
        centered_coords_list = [coords - coords.mean(0) for coords in coords_list]
        types_list = [torch.randint(config.point_types_max, size=(rand,)) for rand in point_num_rands]

        data = collater([CrystalData(
            x=types_list[n][:, None],
            pos=centered_coords_list[n],
            mol_size=torch.tensor(point_num_rands[n], dtype=torch.long),
        ) for n in range(batch_size)]).to(config.device)

        encoding, num_points_prediction, composition_prediction = encoder(data.clone())
        decoding = decoder(encoding, data.clone(), point_num_rands)

        decoded_data = data.clone()
        decoded_data.pos = decoding[:, :3]
        decoded_data.x = F.softmax(decoding[:, 3:], dim=1)

        true_nodes = torch.cat([F.one_hot(data.x[data.batch == i, 0], num_classes=config.point_types_max).float() for i in range(data.num_graphs)])
        per_graph_true_types = torch.stack([true_nodes[data.batch == i].float().mean(0) for i in range(data.num_graphs)])
        per_graph_pred_types = torch.stack([decoded_data.x[data.batch == i].mean(0) for i in range(data.num_graphs)])

        decoder_likelihoods = get_reconstruction_likelihood(data, decoded_data, working_sigma, overlap_type=config.overlap_type)
        self_likelihoods = get_reconstruction_likelihood(data, decoded_data, working_sigma, overlap_type=config.overlap_type, dist_to_self=True)  # if sigma is too large, these can be > 1

        encoding_type_loss = F.binary_cross_entropy_with_logits(composition_prediction, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)  # subtract out minimum
        num_points_loss = F.mse_loss(torch.Tensor(point_num_rands).to(config.device), num_points_prediction[:, 0])

        nodewise_type_loss = F.binary_cross_entropy(per_graph_pred_types, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)
        type_confidence_loss = torch.mean(-torch.log(torch.amax(decoded_data.x, dim=1)))
        # type_confidence_loss = torch.prod(decoded_data.x, dim=1).mean()  # probably better but sometimes unstable
        reconstruction_loss = 10 * torch.mean(scatter(F.smooth_l1_loss(decoder_likelihoods, self_likelihoods, reduction='none'), data.batch, reduce='mean'))  # overlaps should all be exactly 1

        centroid_dists = torch.linalg.norm(data.pos, dim=1)
        centroid_dists_means = torch.stack([centroid_dists[data.batch == i].mean() for i in range(data.num_graphs)])
        centroid_dists_stds = torch.stack([centroid_dists[data.batch == i].std() for i in range(data.num_graphs)])

        decoded_centroid_dists = torch.linalg.norm(decoded_data.pos, dim=1)
        decoded_centroid_dists_means = torch.stack([decoded_centroid_dists[data.batch == i].mean() for i in range(data.num_graphs)])
        decoded_centroid_dists_stds = torch.stack([decoded_centroid_dists[data.batch == i].std() for i in range(data.num_graphs)])

        centroid_dist_loss = F.smooth_l1_loss(decoded_centroid_dists_means, centroid_dists_means)
        centroid_std_loss = F.smooth_l1_loss(decoded_centroid_dists_stds, centroid_dists_stds)

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

        optimizer.zero_grad()
        if config.do_training:
            loss.backward()
            optimizer.step()

        losses = update_losses(losses, num_points_loss, reconstruction_loss, encoding_type_loss,
                               working_sigma, type_confidence_loss, loss, nodewise_type_loss,
                               centroid_dist_loss, centroid_std_loss)

        if step % 10 == 0:
            log_losses(wandb, losses, step, optimizer, data, batch_size, working_sigma, decoded_data, mean_num_points)
            save_checkpoint(encoder, decoder, optimizer, config, step, losses)

        if step % 25 == 0:
            log_rmsd_loss(wandb, data, decoded_data)

        if step % 50 == 0:
            if np.mean(losses['reconstruction_loss'][-50:]) < 1:
                if working_sigma > 0.01:  # make the problem harder
                    working_sigma *= config.sigma_lambda
                elif working_sigma <= 0.01:
                    mean_num_points += 1  # make the problem harder

            if step > config.min_num_training_steps:
                converged1 = check_convergence(np.asarray(losses['scaled_reconstruction_loss']), 50, 1e-3)
                converged2 = check_convergence(np.asarray(losses['combined_loss']), 50, 1e-3)
                converged = converged1 and converged2

        if step % config.lr_timescale == 0 and step != 0 and optimizer.param_groups[0]['lr'] > 1e-5:
            scheduler.step()

aa = 1

# import plotly.graph_objects as go
# best_match_ind = np.argwhere(rmsds == np.amin(rmsds[np.isfinite(rmsds)]))[:, 0][0]
# fig = go.Figure()
# x, y, z = data.pos[data.batch == best_match_ind].T.cpu().detach().numpy()
# fig.add_trace(go.Scatter3d(
#     x=x, y=y, z=z,
#     mode='markers',
#     showlegend=True,
#     marker=dict(
#         color='red',
#         opacity=0.5
#     )))
# x, y, z = decoded_data.pos[data.batch == best_match_ind].T.cpu().detach().numpy()
# fig.add_trace(go.Scatter3d(
#     x=x, y=y, z=z,
#     mode='markers',
#     showlegend=True,
#     marker=dict(
#         color='blue',
#         opacity=0.5
#     )))
# fig.update_layout(showlegend=True)
# fig.show()


# reconstruction includes 1) radial graph 2) overall orientation 3) chirality
# real_rdf, rr, _ = new_crystal_rdf(data, rrange=[0, config.points_spread * 2], bins=2000, raw_density=True, elementwise=True, cpu_detach=False,
#                                   atomic_numbers_override=torch.arange(config.point_types_max, dtype=torch.long, device=config.device))
# decoded_rdf, rr, _ = new_crystal_rdf(decoded_data, rrange=[0, config.points_spread * 2], bins=2000, raw_density=True, elementwise=True, cpu_detach=False,
#                                      atomic_numbers_override=torch.arange(config.point_types_max, dtype=torch.long, device=config.device))
#
# rdf_dists = torch.zeros(data.num_graphs, device=config.device, dtype=torch.float32)
# for i in range(data.num_graphs):
#     rdf_dists[i] = compute_rdf_distance(real_rdf[i], decoded_rdf[i], rr) / point_num_rands[i]  # divides out the trivial size correlation
#
# real_Ip, _, _ = batch_molecule_principal_axes_torch([data.pos[data.batch == i] for i in range(data.num_graphs)])
# decoded_Ip, _, _ = batch_molecule_principal_axes_torch([decoded_data.pos[data.batch == i] for i in range(data.num_graphs)])
#
# real_Ip_handedness = compute_Ip_handedness(real_Ip)
# decoded_Ip_handedness = compute_Ip_handedness(decoded_Ip)

# reconstruction_loss = torch.log10(1 + rdf_dists).mean() + F.smooth_l1_loss(decoded_Ip, real_Ip)  # could be a more appropriate loss function here
# F.smooth_l1_loss(decoded_Ip_handedness, real_Ip_handedness)

import os
from argparse import Namespace
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # ignore numpy error
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
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

config = Namespace(**config)
batch_size = config.batch_size_min
working_sigma = config.sigma
mean_num_points = config.mean_num_points
os.chdir(config.run_directory)

converged = False

with (wandb.init(
        config=config.__dict__,
        project='PointAutoencoder',
        entity='mkilgour'
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
              }
    # todo add convergence flag

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

        decoder_likelihoods = get_reconstruction_likelihood(data, decoded_data, working_sigma)
        self_likelihoods = get_reconstruction_likelihood(data, decoded_data, working_sigma, dist_to_self=True)  # if sigma is too large, these can be > 1

        encoding_type_loss = F.binary_cross_entropy_with_logits(composition_prediction, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)  # subtract out minimum
        num_points_loss = F.mse_loss(torch.Tensor(point_num_rands).to(config.device), num_points_prediction[:, 0])

        nodewise_type_loss = F.binary_cross_entropy(per_graph_pred_types, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)
        type_confidence_loss = torch.mean(-torch.log(torch.amax(decoded_data.x, dim=1)))
        # type_confidence_loss = torch.prod(decoded_data.x, dim=1).mean()  # probably better but sometimes unstable
        reconstruction_loss = 10 * torch.mean(scatter(F.smooth_l1_loss(decoder_likelihoods, self_likelihoods, reduction='none'), data.batch, reduce='mean'))  # overlaps should all be exactly 1

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

        loss = torch.sum(torch.stack(loss_list))

        optimizer.zero_grad()
        if config.do_training:
            loss.backward()
            optimizer.step()

        losses['num_points_loss'].append(num_points_loss.mean().cpu().detach().numpy())
        losses['reconstruction_loss'].append(reconstruction_loss.cpu().detach().numpy())
        losses['overall_type_loss'].append(encoding_type_loss.cpu().detach().numpy())
        losses['scaled_reconstruction_loss'].append(reconstruction_loss.cpu().detach().numpy() * working_sigma)  # as sigma decreases, credit the reconstruction loss
        losses['type_confidence_loss'].append(type_confidence_loss.cpu().detach().numpy())
        losses['combined_loss'].append(loss.cpu().detach().numpy())
        losses['nodewise_type_loss'].append(nodewise_type_loss.cpu().detach().numpy())

        if step % 10 == 0:
            losses_dict = {ltype: np.mean(lval[-10:]) for ltype, lval in losses.items()}
            best_losses_dict = {'best_' + ltype: np.amin(lval) for ltype, lval in losses.items()}

            wandb.log(losses_dict)
            wandb.log(best_losses_dict)

            wandb.log({'batch_size': batch_size,
                       'encoder_learning_rate': optimizer.param_groups[0]['lr'],
                       'decoder_learning_rate': optimizer.param_groups[1]['lr'],
                       'step': step,
                       'mean_type_confidence': np.mean(np.amax(decoded_data.x.cpu().detach().numpy(), axis=1)),
                       'sigma': working_sigma,
                       'mean_num_points_setting': mean_num_points,
                       'mean_num_points_actual': data.num_nodes / data.num_graphs,
                       })

            if losses['scaled_reconstruction_loss'][-1] == np.amin(losses['scaled_reconstruction_loss']):
                torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           config.save_directory + config.run_name + '_' + 'autoencoder_ckpt' + '_' + str(step))

        if step % 50 == 0:
            if np.mean(losses['reconstruction_loss'][-50:]) < 1:
                if working_sigma > 0.01:  # make the problem harder
                    working_sigma *= config.sigma_lambda
                elif working_sigma <= 0.01:
                    mean_num_points += 1  # make the problem harder

            if step > config.min_num_training_steps:
                converged = check_convergence(np.asarray(losses['scaled_reconstruction_loss']),
                                              50,
                                              1e-3)

        if step % 25 == 0:
            rmsds = np.zeros(data.num_graphs)
            general_rmsds = np.zeros_like(rmsds)
            for g_ind in range(data.num_graphs):
                inds = data.batch == g_ind

                ref_coords = data.pos[inds]
                ref_types = data.x[inds, 0]

                pred_coords = decoded_data.pos[inds]
                pred_types = torch.argmax(decoded_data.x[inds], dim=1)

                dists = cdist(ref_coords.cpu().detach().numpy(), pred_coords.cpu().detach().numpy())
                assignment = linear_sum_assignment(dists)
                general_rmsds[g_ind] = dists[assignment].sum() / len(ref_coords)

                a, b = torch.unique(ref_types, return_counts=True)
                c, d = torch.unique(pred_types, return_counts=True)
                if len(a) == len(c):
                    if all(a == c) and all(b == d):
                        typewise_rmsds = np.zeros(len(a))

                        for it, type_ind in enumerate(a):
                            ref_type_pos = ref_coords[ref_types == type_ind]
                            pred_type_pos = pred_coords[ref_types == type_ind]

                            dists = cdist(ref_type_pos.cpu().detach().numpy(), pred_type_pos.cpu().detach().numpy())
                            assignment = linear_sum_assignment(dists)
                            typewise_rmsds[it] = dists[assignment].sum() / len(ref_type_pos)

                        rmsds[g_ind] = np.mean(typewise_rmsds)
                    else:
                        rmsds[g_ind] = np.nan
                else:
                    rmsds[g_ind] = np.nan

            wandb.log({'Matching Clouds Fraction': np.mean(np.isfinite(rmsds)),
                       'Matching Clouds RMSDs': np.mean(rmsds[np.isfinite(rmsds)]),
                       'All Points RMSDs': np.mean(general_rmsds)})

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

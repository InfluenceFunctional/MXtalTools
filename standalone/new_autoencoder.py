import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR
from tqdm import tqdm

os.environ["WANDB_START_METHOD"] = 'thread'

from autoencoder_crystaldata import CrystalData
from torch_geometric.loader.dataloader import Collater
from torch.optim import Adam
import wandb

from standalone.new_autoencoder_utils import (
    point_cloud_encoder, point_cloud_decoder, get_reconstruction_likelihood)

config = {'training_iterations': 1000000,
          'batch_size_min': 2,
          'batch_size_max': 1000,
          'mean_num_points': 5,
          'num_points_spread': 1,
          'points_spread': 1,
          'point_types_max': 2,
          'device': 'cuda',
          'seed': 12345,
          'learning_rate': 1e-4,
          'lr_lambda': 0.95,
          'lr_timescale': 500,
          'embedding_depth': 512,
          'num_layers': 4,
          'sigma': 0.05,
          }

os.chdir(r'C:\Users\mikem\crystals\CSP_runs')

config = Namespace(**config)

with wandb.init(
        config=config.__dict__,
        project='MXtalTools',
        entity='mkilgour'
):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    encoder = point_cloud_encoder(embedding_depth=config.embedding_depth,
                                  num_layers=config.num_layers,
                                  cutoff=config.points_spread,
                                  seed=config.seed, device=config.device,
                                  num_classes=config.point_types_max).to(config.device)
    decoder = point_cloud_decoder(embedding_depth=config.embedding_depth,
                                  num_layers=config.num_layers,
                                  cutoff=config.points_spread * 2,
                                  max_ntypes=config.point_types_max,
                                  seed=config.seed, device=config.device).to(config.device)
    collater = Collater(0, 0)
    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config.learning_rate)
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: config.lr_lambda)

    wandb.watch((encoder, decoder), log_graph=True, log_freq=100)

    losses = {'reconstruction_loss': [],
              'num_points_loss': [],
              'overall_type_loss': [],
              }

    batch_size = config.batch_size_min

    for step in tqdm(range(config.training_iterations)):

        if step % 10 == 0 and batch_size < config.batch_size_max:
            batch_size += 1

        point_num_rands = np.clip(np.round((np.random.randn(batch_size) + config.mean_num_points) * config.num_points_spread), a_min=1, a_max=1000).astype(int)

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

        num_points_loss = F.smooth_l1_loss(torch.Tensor(point_num_rands).to(config.device), num_points_prediction[:, 0])
        per_graph_true_types = torch.stack([F.one_hot(data.x[data.batch == i, 0], num_classes=config.point_types_max).float().mean(0) for i in range(data.num_graphs)])

        reconstruction_loss = get_reconstruction_likelihood(data, decoded_data, config.sigma)

        overall_type_loss = F.binary_cross_entropy_with_logits(composition_prediction, per_graph_true_types) - F.binary_cross_entropy(per_graph_true_types, per_graph_true_types)  # subtract out minimum

        loss = reconstruction_loss + num_points_loss + overall_type_loss
        loss = loss.clip(max=10)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #
        # fig = go.Figure()
        # x, y, z = data.pos[data.batch == 0].cpu().detach().numpy()
        # fig.add_trace(go.Scatter3d(
        #     x=x, y=y, z=z,
        #     mode='markers',
        #     showlegend=True,
        #     marker=dict(
        #         color='red',
        #         opacity=0.5
        #     )))
        # x, y, z = decoded_data.pos[data.batch == 0].cpu().detach().numpy()
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

        losses['num_points_loss'].append(num_points_loss.mean().cpu().detach().numpy())
        losses['reconstruction_loss'].append(reconstruction_loss.cpu().detach().numpy())
        losses['overall_type_loss'].append(overall_type_loss.cpu().detach().numpy())

        if step % 10 == 0:
            wandb.log({'reconstruction_loss': np.mean(losses['reconstruction_loss'][-10:]),
                       'num_atoms_loss': np.mean(losses['num_points_loss'][-10:]),
                       'overall_type_loss': np.mean(losses['overall_type_loss'][-10:]),
                       'combined_loss': np.mean(losses['reconstruction_loss'][-10:]) + np.mean(losses['num_points_loss'][-10:]) + np.mean(losses['overall_type_loss'][-10:]),
                       'batch_size': batch_size,
                       'learning_rate': optimizer.param_groups[0]['lr'],
                       'step': step,
                       'best_reconstruction_loss': np.amin(losses['reconstruction_loss']),
                       })
            if losses['reconstruction_loss'][-1] == np.amin(losses['reconstruction_loss']):
                torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           'D:/crystals_extra/autoencoder_ckpt')

        if step % config.lr_timescale == 0 and step != 0 and optimizer.param_groups[0]['lr'] > 1e-5:
            scheduler.step()

    aa = 1

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

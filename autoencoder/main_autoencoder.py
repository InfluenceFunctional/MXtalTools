import os
from argparse import Namespace
import warnings
from autoencoder.reporting import log_losses, save_checkpoint, overlap_plot
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiplicativeLR
from tqdm import tqdm
from dataset_management.CrystalData import CrystalData
from torch_geometric.loader.dataloader import Collater
from torch.optim import Adam
import argparse
import wandb
from autoencoder.utils import (
    point_cloud_encoder, load_checkpoint, fc_decoder, compute_loss)
from autoencoder.configs import dev, configs
from models.utils import check_convergence

warnings.filterwarnings("ignore", category=UserWarning)
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
        entity='mkilgour',
        tags=config.experiment_tag
)):
    wandb.run.name = config.run_name
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    encoder = point_cloud_encoder(cart_dimension=config.cart_dimension,
                                  aggregator=config.encoder_aggregator,
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
                                  num_classes=config.max_point_types).to(config.device)

    decoder = fc_decoder(num_nodes=config.num_fc_nodes,
                         cart_dimension=config.cart_dimension,
                         input_depth=config.encoder_embedding_depth,
                         embedding_depth=config.decoder_embedding_depth,
                         num_layers=config.decoder_num_layers,
                         num_nodewise_fcs=config.decoder_num_nodewise_fcs,
                         graph_norm=config.decoder_graph_norm,
                         message_norm=config.decoder_message_norm,
                         dropout=config.decoder_dropout,
                         cutoff=config.points_spread * 2,
                         max_ntypes=config.max_point_types,
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

        coords_list = [torch.rand(rand, config.cart_dimension) * config.points_spread for rand in point_num_rands]
        centered_coords_list = [coords - coords.mean(0) for coords in coords_list]
        types_list = [torch.randint(config.max_point_types, size=(rand,)) for rand in point_num_rands]

        data = collater([CrystalData(
            x=types_list[n][:, None],
            pos=centered_coords_list[n],
            mol_size=torch.tensor(point_num_rands[n], dtype=torch.long),
        ) for n in range(batch_size)]).to(config.device)

        encoding, num_points_prediction, composition_prediction = encoder(data.clone())
        decoding = decoder(encoding, data.clone(), point_num_rands)

        loss, losses, decoded_data, nodewise_weights = compute_loss(
            losses, config, working_sigma, num_points_prediction, composition_prediction, decoding, data, point_num_rands)

        optimizer.zero_grad()
        if config.do_training:
            loss.backward()
            optimizer.step()

        if step % 10 == 0:
            log_losses(wandb, losses, step, optimizer, data, batch_size, working_sigma, decoded_data, mean_num_points)

        if step % 50 == 0:
            save_checkpoint(encoder, decoder, optimizer, config, step, losses)

        if step % 100 == 0:
            if np.mean(losses['reconstruction_loss'][-100:]) < 0.15:
                if working_sigma > 0.0001:  # make the problem harder
                    working_sigma *= config.sigma_lambda
                # elif working_sigma <= 0.01:
                #     mean_num_points += 1  # make the problem harder

            if step > config.min_num_training_steps:
                converged1 = check_convergence(np.asarray(losses['scaled_reconstruction_loss']), config.convergence_history, config.convergence_eps)
                converged2 = check_convergence(np.asarray(losses['combined_loss']), config.convergence_history, config.convergence_eps)
                converged = converged1 and converged2

            overlap_plot(wandb, data, decoded_data, working_sigma, config, nodewise_weights)

        if step % config.lr_timescale == 0 and step != 0 and optimizer.param_groups[0]['lr'] > 1e-5:
            scheduler.step()

aa = 1

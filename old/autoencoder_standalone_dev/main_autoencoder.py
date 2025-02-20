import os
from argparse import Namespace
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from autoencoder_standalone_dev.reporting import log_losses, save_checkpoint
from reporting.online import overlap_plot
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiplicativeLR
from tqdm import tqdm
from dataset_management.CrystalData import CrystalData
from torch_geometric.loader.dataloader import Collater
from torch.optim import Adam
import argparse
import wandb
from autoencoder_standalone_dev.utils import (
    load_checkpoint, compute_loss)
from autoencoder_standalone_dev.ae_models import point_cloud_encoder, fc_decoder
from autoencoder_standalone_dev.configs import dev, configs
from models.utils import check_convergence
from datetime import datetime

os.environ["WANDB_START_METHOD"] = 'thread'
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime

parser = argparse.ArgumentParser()
args = parser.parse_known_args()[1]

if '--config' in args:
    config = configs[int(args[1])]
else:
    config = dev

config = Namespace(**config)
batch_size = config.batch_size_min
working_sigma = config.sigma
working_min_points = config.min_num_points
working_max_points = config.max_num_points
os.chdir(config.run_directory)
hit_max_batch = False

converged = False

with (wandb.init(
        config=config.__dict__,
        project='PointAutoencoder',
        entity='mkilgour',
        tags=config.experiment_tag
)):
    wandb.run.name = config.run_name + '_' + datetime.today().strftime("%d-%m-%H-%M-%S")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    encoder = point_cloud_encoder(
        cart_dimension=config.cart_dimension,
        aggregator=config.encoder_aggregator,
        embedding_depth=config.encoder_embedding_depth,
        num_fc_layers=0,
        num_layers=config.encoder_num_layers,
        num_nodewise_fcs=config.encoder_num_nodewise_fcs,
        fc_norm=None,
        graph_norm=config.encoder_graph_norm,
        dropout=config.encoder_dropout,
        cutoff=config.points_spread * 2,
        seed=config.seed, device=config.device,
        num_classes=config.max_point_types).to(config.device)

    decoder = fc_decoder(
        num_nodes=config.num_decoder_points,
        cart_dimension=config.cart_dimension,
        input_depth=config.encoder_embedding_depth,
        embedding_depth=config.decoder_embedding_depth,
        num_layers=config.decoder_num_layers,
        fc_norm=config.decoder_fc_norm,
        dropout=config.decoder_dropout,
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

    sigma_record = []
    losses = {'reconstruction_loss': [],
              'num_points_loss': [],
              'overall_type_loss': [],
              'scaled_reconstruction_loss': [],
              'combined_loss': [],
              'nodewise_type_loss': [],
              'centroid_mean_loss': [],
              'constraining_loss': [],
              'mean_self_overlap': [],
              }

    for step in tqdm(range(config.training_iterations), miniters=100):
        if converged:
            break
        try:
            if step % 10 == 0 and batch_size < config.batch_size_max and not hit_max_batch:
                batch_size += config.batch_size_increment

            point_num_rands = np.random.randint(low=working_min_points, high=working_max_points + 1, size=batch_size)

            # truly random point clouds within a sphere of fixed maximum volume
            vectors = torch.rand(point_num_rands.sum(), config.cart_dimension, dtype=torch.float32, device=config.device)
            norms = torch.linalg.norm(vectors, dim=1)[:, None]
            lengths = torch.rand(point_num_rands.sum(), 1, dtype=torch.float32, device=config.device)
            coords_list = (vectors / norms * lengths).split(point_num_rands.tolist())

            # centered_coords_list = [coords - coords.mean(0) for coords in coords_list]
            types_list = torch.randint(config.max_point_types, size=(point_num_rands.sum(),), device=config.device).split(point_num_rands.tolist())

            data = collater([CrystalData(  # CRYTODO
                x=types_list[n][:, None],
                pos=coords_list[n],
                mol_size=torch.tensor(point_num_rands[n], dtype=torch.long, device=config.device),
            ) for n in range(batch_size)]).to(config.device)  # todo this is now one of the slower steps

            encoding, num_points_prediction, composition_prediction = encoder(data.clone())
            decoding = decoder(encoding, data.clone(), point_num_rands)

            sigma_record.append(working_sigma)
            loss, losses, decoded_data, nodewise_weights, mean_sample_likelihood = compute_loss(
                wandb, step,
                losses, config, working_sigma, num_points_prediction,
                composition_prediction, decoding, data, point_num_rands)

            optimizer.zero_grad()
            if config.do_training:
                loss.backward()
                optimizer.step()

            if step % 100 == 0:
                log_losses(
                    wandb, losses, step, optimizer, data, batch_size,
                    working_sigma, mean_sample_likelihood,
                    working_min_points, working_max_points)

                save_checkpoint(encoder, decoder, optimizer, config, step, losses)

                if np.mean(losses['reconstruction_loss'][-100:]) < config.sigma_threshold:
                    if np.abs(1 - np.mean(losses['mean_self_overlap'][-100:])) > config.self_overlap_eps:  # if points are insufficiently well separated, make the problem harder
                        working_sigma *= config.sigma_lambda

                if step > config.min_num_training_steps:
                    converged1 = check_convergence(np.asarray(losses['scaled_reconstruction_loss']), config.convergence_history, config.convergence_eps)
                    converged2 = check_convergence(np.asarray(losses['combined_loss']), config.convergence_history, config.convergence_eps)
                    converged3 = check_convergence(np.asarray(sigma_record), config.convergence_history, config.convergence_eps)
                    converged = converged1 and converged2 and converged3

            if step % 1000 == 0:
                overlap_plot(wandb, data, decoded_data, working_sigma, config.max_point_types, config.cart_dimension, nodewise_weights)

            if step % config.lr_timescale == 0 and step != 0 and optimizer.param_groups[0]['lr'] > 1e-5:
                scheduler.step()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "nonzero is not supported for tensors with more than INT_MAX elements" in str(e):
                batch_size = batch_size - max(1, int(batch_size * 0.1))
                hit_max_batch = True
                print('Hit Max Batch Size!')
            else:
                raise e

aa = 1

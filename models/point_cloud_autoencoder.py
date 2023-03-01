'''
can a model recreate the point cloud positions given an encoding?
'''
import torch
from torch import backends, optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
from models.model_components import *
import tqdm
import plotly.graph_objects as go
from models.basis_functions import GaussianEmbedding, BesselBasisLayer
from models.positional_encodings import PosEncoding3D, PosEncoding2D
from torch_scatter import scatter
import time
from e3nn import o3
from torch.optim import lr_scheduler
import wandb
from models.global_aggregation import global_aggregation
from scipy.stats import wasserstein_distance
from utils import make_grid, torch_emd
import itertools
from torch_geometric.transforms import spherical, polar
import torch_geometric.nn as gnn
from plotly.subplots import make_subplots
from models.torch_models import molecule_graph_model
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime

n_runs = 1
multipliers = np.asarray((1, 2, 4, 8, 16, 32))

backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

for run in range(n_runs):
    randints = np.random.randint(0, len(multipliers), size=8)

    avg_num_particles_per_sample = 10
    cart_dim = 3
    n_bins = 6
    n_gridpoints = 1
    convergence_criteria = 1e-3  # minimum improvement in last history_length epochs
    n_particle_types = 5

    model_type = 'mike' # 'mike' 'e3' WIP
    gconv_type = 'none' # 'TransformerConv' 'none'
    initial_transform = None
    embedding_type = 'pos'  # 'pos' 'rad' 'sph' 'polar'
    batch_size = 500  # int(10 * multipliers[randints[0]])
    init_lr = 1e-3  # 1e-6 * multipliers[randints[1]]
    lr_lambda = 0.999  # (1 - 0.001 * multipliers[randints[2]])
    n_layers = 1 # int(max(1, 1 * multipliers[randints[3]]))
    n_filters = 1024  # int(64 * multipliers[randints[4]])
    encoder_layers = 2  # int(max(1, 1 * multipliers[randints[5]]))
    encoder_filters = 1024#[64,128,256,1024,2048,4096] # n_layers +1 entries # int(256 * multipliers[randints[6]])
    encoding_output_depth = encoder_filters // (n_gridpoints ** cart_dim)  # int(256 * multipliers[randints[7]])
    norm_type_1 = 'graph' # 'graph' 'layer' 'batch' if np.random.randint(0, 2) == 1 else None
    norm_type_2 = 'layer'  # 'batch' if np.random.randint(0, 2) == 1 else None
    pooling = 'max'  # 'combo' 'max' 'mean' 'attention'

    n_epochs = int(1e7 / batch_size)
    history_length = int(1e5 / batch_size)
    target_shape = [batch_size] + [n_bins for n in range(cart_dim)]

    config = {
        'embedding_type': embedding_type,
        'pooling': pooling,
        'avg_num_particles_per_sample': avg_num_particles_per_sample,
        'cart_dim': cart_dim,
        'batch_size': batch_size,
        'n_bins': n_bins,
        'n_epochs': n_epochs,
        'target_shape': target_shape,
        'convergence_criteria': convergence_criteria,
        'n_gridpoints': n_gridpoints,
        'history_length': history_length,
        'init_lr': init_lr,
        'r_lambda': lr_lambda,
        'n_layers': n_layers,
        'n_filters': n_filters,
        'encoder_layers': encoder_layers,
        'encoder_filters': encoder_filters,
        'encoding_output_depth': encoding_output_depth,
        'norm_type_1': norm_type_1,
        'norm_type_2': norm_type_2,
    }

    '''
    init model
    '''

    class graph_encoder(torch.nn.Module):
        def __init__(self):
            super(graph_encoder, self).__init__()
            # grid of points over which to sample
            gridpoint_lim = 1 - 1 / n_gridpoints
            grid = make_grid(gridpoint_lim, n_gridpoints, cart_dim)
            shift_vector = torch.repeat_interleave(grid,batch_size * avg_num_particles_per_sample,dim=0)

            self.register_buffer("grid", grid)
            self.register_buffer("shift_vector", shift_vector)

            self.num_radial = 50
            self.sph_od_list = [i for i in range(11)]
            self.num_spherical = int(torch.sum(torch.Tensor(self.sph_od_list) * 2 + 1))
            self.encoding_dim = encoding_output_depth
            self.init_transform = initial_transform

            if embedding_type == 'pos' or embedding_type == 'polar':
                embedding_dim = cart_dim
            elif embedding_type == 'rad':
                embedding_dim = self.num_radial
            elif embedding_type == 'sph':
                embedding_dim = self.num_radial + self.num_spherical

            self.mlp = general_MLP(layers=n_layers, filters=n_filters,
                                   input_dim=self.encoding_dim * len(grid),  # self.num_radial * cart_dim, #avg_num_particles_per_sample*cart_dim,
                                   output_dim=n_particle_types * n_bins ** cart_dim,
                                   dropout=0,
                                   norm=norm_type_1,
                                   activation='leaky relu',
                                   norm_after_linear=True,
                                   )

            self.mlp2 = general_MLP(layers=encoder_layers, filters=encoder_filters,
                                    input_dim=embedding_dim + 1,
                                    output_dim=self.encoding_dim,
                                    activation='leaky relu',
                                    norm=norm_type_2,
                                    dropout=0,
                                    norm_after_linear=True)

            cutoff = 1  # / n_gridpoints
            # self.radial_basis = BesselBasisLayer(num_radial=self.num_radial, cutoff=cutoff)
            self.radial_basis = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=self.num_radial)
            # self.pos_encoding = PosEncoding2D(self.num_radial, 1)
            self.global_aggregation = global_aggregation(pooling, filters=self.encoding_dim * len(grid))


        def transform(self, pos, batch):
            if self.init_transform is None:
                init_embedding = pos

            return init_embedding

        def encode(self, pos, batch):
            # return pos.reshape(len(pos), avg_num_particles_per_sample*cart_dim) # just give it the positions directly
            # encoding = self.pos_encoding(pos)
            # return scatter(encoding,batch,dim=0)

            type = pos[:,0].tile(len(self.grid),1) #atom type
            pos_i = pos[:,1:].tile(len(self.grid),1) #atom position
            batch_i = batch.tile(len(self.grid))
            pos_i = pos_i - self.shift_vector # shifted coordinates

            if embedding_type == 'pos':
                embedding = pos_i
            elif embedding_type == 'polar':
                embedding = torch.zeros_like(pos_i)
                rho = torch.norm(pos_i, p=2, dim=-1).view(-1, 1)
                # rho = rho / rho.max()

                theta = torch.atan2(pos_i[..., 1], pos_i[..., 0]).view(-1, 1)
                theta = theta + (theta < 0).type_as(theta) * (2 * torch.pi)
                theta = theta / (2 * torch.pi)

                embedding[:, 0] = rho[:, 0]
                embedding[:, 1] = theta[:, 0]

                if cart_dim == 3:
                    phi = torch.acos(pos_i[..., 2] / rho.view(-1)).view(-1, 1)
                    phi = phi / torch.pi
                    embedding[:, 2] = phi[:, 0]
            elif embedding_type == 'rad':
                dists = torch.linalg.norm(pos_i, dim=-1)
                embedding = self.radial_basis(dists)
            elif embedding_type == 'sph':
                dists = torch.linalg.norm(pos_i, dim=-1)
                rbf = self.radial_basis(dists)
                if cart_dim == 2:
                    sbf = o3.spherical_harmonics(self.sph_od_list, x=torch.cat((pos_i, torch.zeros_like(pos_i)), dim=-1)[:, :3], normalize=True, normalization='component')
                elif cart_dim == 3:
                    sbf = o3.spherical_harmonics(self.sph_od_list, x=pos_i, normalize=True, normalization='component')
                embedding = torch.cat((rbf, sbf), dim=-1)

            encoding = self.mlp2(torch.cat((type,embedding),dim=-1),batch=batch_i)
            encoding = torch.hstack(encoding.split(batch_size * avg_num_particles_per_sample, dim=0))
            graph_output = self.global_aggregation(encoding, pos=None, batch=batch, output_dim = batch_size)

            if len(graph_output) != batch_size:
                assert False

            return graph_output

        def forward(self, pos, batch):
            init_embedding = self.transform(pos,batch)
            encoding = self.encode(init_embedding, batch)
            output = self.mlp(encoding)
            return output

    grid_index = torch.tensor(list(itertools.product([n for n in range(n_bins)], repeat=cart_dim))).cuda()
    grid_index -= n_bins // 2  # center basis vectors on the origin
    converged = False
    with wandb.init(project='shape_encoding', entity='mkilgour', config=config):
        wandb.log(config)
        torch.random.manual_seed(0)
        if model_type == 'encoder':
            model = graph_encoder()
        elif model_type == 'mike':
            model = molecule_graph_model(
                dataDims = None,
                seed = 0,
                num_atom_feats = cart_dim + 1,
                num_mol_feats =0,
                output_dimension = (n_particle_types + 1) * n_bins ** cart_dim,
                activation = 'leaky relu',
                num_fc_layers = n_layers,
                fc_depth = n_filters,
                fc_dropout_probability = 0,
                fc_norm_mode = norm_type_1,
                graph_model = 'mike',
                graph_filters = encoder_filters//4,
                graph_convolutional_layers = encoder_layers,
                concat_mol_to_atom_features = False,
                pooling = pooling,
                graph_norm = norm_type_2,
                num_spherical = 6,
                num_radial = 50,
                graph_convolution = gconv_type,
                num_attention_heads = 4,
                add_spherical_basis = False,
                add_torsional_basis = False,
                atom_embedding_size = 512,
                radial_function = 'gaussian',
                max_num_neighbors = 100,
                convolution_cutoff = 2,
                max_molecule_size = 1,
                return_latent=False,
                crystal_mode=False,
                crystal_convolution_type=None,
                positional_embedding='sph3',
                atom_embedding_dims = n_particle_types,
                device='cuda',
            )


        optimizer = optim.Adam(model.parameters(), lr=init_lr)
        scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_lambda)

        '''
        training loop
        '''
        model.cuda()
        model.train()
        loss_record = []
        for epoch in tqdm.tqdm(range(n_epochs), miniters=n_epochs // 100, mininterval=30):
            epoch_loss = []
            # just make a random one every epoch
            batch = torch.randint(low=0, high=batch_size, size=(batch_size * avg_num_particles_per_sample, 1),device='cuda')[:, 0]  # batch index
            batch = batch[torch.argsort(batch)]
            particle_coords = torch.rand(batch_size * avg_num_particles_per_sample, cart_dim,device='cuda') * 2 - 1  # particle positions with mean of avg_num_particles_per_sample particles per sample
            particle_types = torch.randint(low=1,high=n_particle_types+1, size = (batch_size * avg_num_particles_per_sample,1),device='cuda')[:,0]
            buckets = torch.bucketize(particle_coords, torch.linspace(-1, 1, n_bins + 1, device='cuda')) - 1

            if cart_dim == 2:
                target = torch.zeros((batch_size, n_bins, n_bins), dtype=torch.long, device='cuda')
                for ii in range(batch_size):
                    target[ii, buckets[batch == ii, 0], buckets[batch == ii, 1]] = particle_types[batch == ii]
            elif cart_dim == 3:
                target = torch.zeros((batch_size, n_bins, n_bins, n_bins), dtype=torch.long, device='cuda')
                for ii in range(batch_size):
                    target[ii, buckets[batch == ii, 0], buckets[batch == ii, 1],buckets[batch == ii, 2]] = particle_types[batch == ii]

            # if cart_dim == 2:
            #     target = torch.stack([torch.histogramdd(particle_coords[batch == ii], range=[-1, 1, -1, 1], bins=n_bins)[0] for ii in range(batch_size)]).cuda()
            # elif cart_dim == 3:
            #     target = torch.stack([torch.histogramdd(particle_coords[batch == ii], range=[-1, 1, -1, 1, -1, 1], bins=n_bins)[0] for ii in range(batch_size)]).cuda()
            #target = target.clip(min=0, max=1)

            if model_type == 'encoder':
                output = model(torch.cat((particle_types, particle_coords), dim=-1), batch)
            elif model_type == 'mike':
                output = model(x=torch.cat((particle_types[:,None], particle_coords), dim=-1),
                               pos=particle_coords,
                               batch=batch,
                               num_graphs = batch_size)

            if cart_dim == 2:
                output = output.reshape(batch_size, 1+n_particle_types, n_bins, n_bins)
            elif cart_dim == 3:
                output = output.reshape(batch_size, 1+n_particle_types, n_bins, n_bins, n_bins)
            #bce_loss = F.binary_cross_entropy_with_logits(output, target, reduction='none').mean()
            bce_loss = F.cross_entropy(output,target)

            loss = bce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach())

            lr = optimizer.param_groups[0]['lr']
            if lr > 1e-6:
                scheduler.step()

            if epoch % 200 == 0:
                if cart_dim == 2:
                    fig = make_subplots(rows=2,cols=4)
                    for img_i in range(4):
                        fig.add_trace(go.Heatmap(z=output[img_i].argmax(0).cpu().detach().numpy()),
                                      row = 1, col = img_i+1)
                        fig.add_trace(go.Heatmap(z=target[img_i].cpu().detach().numpy(),),
                                      row = 2, col = img_i+1)


                elif cart_dim == 3:
                    fig = make_subplots(
                        rows=2, cols=2,
                        specs=[[{'type': 'scene'}, {'type': 'scene'}],
                               [{'type': 'scene'}, {'type': 'scene'}]])

                    for img_i in range(2):
                        for img_j in range(2):
                            sample_guess = output[img_i * 2 + img_j].argmax(0).cpu().detach().numpy()
                            sample_true = target[img_i * 2 + img_j].cpu().detach().numpy()

                            X, Y, Z = (sample_guess + 1).nonzero()
                            fig.add_trace(go.Volume(
                                x=X.flatten(),
                                y=Y.flatten(),
                                z=Z.flatten(),
                                value=sample_guess.flatten(),
                                isomin=0.001,
                                isomax=1,
                                opacity=0.05,  # needs to be small to see through all surfaces
                                surface_count=50,  # needs to be a large number for good volume rendering
                                colorscale='Jet',
                                cmin=0,
                                showlegend=True
                            ), row=img_i + 1, col=img_j + 1)

                            x, y, z = sample_true.nonzero()
                            fig.add_trace(go.Scatter3d(
                                x=x, y=y, z=z,
                                mode='markers',
                                showlegend=True,
                                marker=dict(
                                    size=10,
                                    color=sample_true[x, y, z],
                                    colorscale='Jet',
                                    cmin=0,
                                    opacity=0.5
                                )), row=img_i + 1, col=img_j + 1)
                            fig.update_layout(showlegend=True)

                layout = go.Layout(
                    margin=go.layout.Margin(
                        l=0,  # left margin
                        r=0,  # right margin
                        b=0,  # bottom margin
                        t=40,  # top margin
                    )
                )
                fig.layout.margin = layout.margin
                # fig.show()
                wandb.log({"samples": fig})

            if epoch % 10 == 0:
                print(f'epoch {epoch} loss {torch.log10(loss.detach()).mean():.3f}')
                loss_record_tensor = torch.tensor(loss_record)
                best_loss = loss_record_tensor.min()
                best_loss_epoch = int(torch.argmin(loss_record_tensor))
                # ema = torch.exp(torch.arange(-min(epoch,history_length), 1) / 10)
                # ema /= ema.sum()
                # trailing_loss = (loss_record_tensor[-history_length:] * ema[-history_length:]).sum()
                trailing_best = loss_record_tensor[-history_length:].min()
                if epoch > history_length:
                    best_prior = loss_record_tensor[:-history_length].min()
                else:
                    best_prior = loss_record_tensor[0]
                convergence_rate = ((best_prior - trailing_best).abs() / best_prior)

                if (epoch > (2 * history_length)) and (convergence_rate < 1e-2):
                    # if (epoch - best_loss_epoch) > history_length:
                    if (convergence_rate < convergence_criteria) or (trailing_best > best_prior):  # converged or diverged
                        converged = True
                        print("Converged!")

                wandb.log({'loss': loss,
                           'lr': lr,
                           'best_loss': best_loss,
                           'epoch': epoch,
                           'convergence_rate': convergence_rate,
                           'loss_slope': loss_record_tensor[-1] / epoch,
                           'bce_loss': bce_loss.mean().cpu().detach().numpy(),
                           # 'trailing_loss': trailing_loss}
                           })

                if converged:
                    break

        loss_record = torch.stack(loss_record).cpu().detach().numpy()

    a = 1
'''
    plt.figure(1)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.loglog(loss_record, '-')
    plt.ylim(1e-2, 1)
    plt.subplot(1, 2, 2)
    plt.semilogy(loss_record, '-')
    plt.ylim(1e-2, 1)
    plt.tight_layout()

    plt.figure(2)
    plt.clf()
    if cart_dim == 2:
        for ii in np.arange(1, 40, 2):
            plt.subplot(5, 8, ii)
            plt.imshow(torch.sigmoid((output[ii])).cpu().detach().numpy())
            plt.subplot(5, 8, ii + 1)
            plt.imshow(target[ii].cpu().detach().numpy())
        plt.tight_layout()
    elif cart_dim == 3:
        fig = plt.figure(2)
        plt.clf()
        sample = torch.sigmoid(output).cpu().detach().numpy()
        goal = target.cpu().detach().numpy()
        ax = fig.add_subplot(221, projection='3d')
        x, y, z = sample[0].nonzero()
        ax.scatter(x, y, z, c=(sample[0]), alpha=1)
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.scatter(x, y, z, c=goal[0], alpha=1)
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.scatter(x, y, z, c=(sample[1]), alpha=1)
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.scatter(x, y, z, c=goal[1], alpha=1)
        plt.show()
        plt.tight_layout()
'''

'''old
 # 1D sliced wasserstein distance (earth mover's distance)
            n_dirs = 50
            dir = torch.randn((n_dirs, cart_dim), device='cuda')
            dir /= torch.linalg.norm(dir, dim=-1)[:, None]

            #rand_intercepts = torch.random(n_dirs)
            target_weights = target.reshape(batch_size,n_bins**cart_dim) # tested - same indexing as target_grid
            target_vectors = grid_index[:, None, :] * target_weights.T[:, :, None]
            target_overlap = torch.einsum('mnj,kj->nkm', (target_vectors, dir))

            output_weights = torch.sigmoid(output.reshape(batch_size,n_bins**cart_dim)) # tested - same indexing as target_grid
            output_vectors = grid_index[:, None, :] * output_weights.T[:, :, None]
            output_overlap = torch.einsum('mnj,kj->nkm', (output_vectors, dir))

            emd_loss = torch_emd(target_overlap, output_overlap).mean(1) / (n_bins ** cart_dim) # averge over slices
'''

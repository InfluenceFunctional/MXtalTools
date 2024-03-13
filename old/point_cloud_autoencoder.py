'''
can a model recreate the point cloud positions given an encoding?
'''
from torch import backends, optim
from models.components import *
import tqdm
import plotly.graph_objects as go
from torch.optim import lr_scheduler
import wandb
import itertools
from plotly.subplots import make_subplots
from models.base_models import molecule_graph_model
from models.generator_models import PointCloudDecoder
import os
from dataset_management.utils import update_dataloader_batch_size

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # slows down runtime

backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

workdir = 'C:/Users\mikem\Desktop/CSP_runs/point_cloud_encodings'
os.chdir(workdir)  # move to working dir

model_name = 'encoder_1'
avg_num_particles_per_sample = 10  # initial number
max_num_particles = 10
cartesian_dimension = 3
n_bins = 20
n_gridpoints = 1
convergence_criteria = 1e-7  # minimum improvement in last history_length epochs
n_particle_types = 1

gconv_type = 'none'  # 'TransformerConv' 'none'
embedding_type = 'pos'  # 'pos' 'rad' 'sph' 'polar'
batch_size = 10
batch_size_growth_lambda = 1.0005
max_batch_size = 200
init_lr = 1e-4
lr_growth_lambda = 1.001
lr_shrink_lambda = 0.9995
min_lr = 5e-4 #1e-5
max_lr = 1e-3 #1e-2
weight_decay = 0.01
n_layers = 4
n_filters = 512
fc_dropout = 0.5
encoder_layers = 1
encoder_filters = 256
encoding_output_depth = encoder_filters // (n_gridpoints ** cartesian_dimension)
norm_type_1 = None # 'graph' 'layer' 'batch' if np.random.randint(0, 2) == 1 else None
norm_type_2 = None  # 'batch' if np.random.randint(0, 2) == 1 else None
pooling = 'max'  # 'combo' 'max' 'mean' 'attention'
conv_embedding_dim = 128
resolution_scaling = [5, 2]  # [2,
decoder_mode = 'conv_transpose'  # 'mlp' 'conv_transpose' 'upscale' WIP
gauss_sigma = 0 #1
gauss_kernel_size = 5

n_epochs = int(1e5 / batch_size)
history_length = 100  # int(1e4 / batch_size)

config = {
    'fc_dropout': fc_dropout,
    'decoder_mode': decoder_mode,
    'resolution_scaling': resolution_scaling,
    'weight_decay': weight_decay,
    'init_lr': init_lr,
    'max_lr': max_lr,
    'min_lr': min_lr,
    'embedding_type': embedding_type,
    'pooling': pooling,
    'avg_num_particles_per_sample': avg_num_particles_per_sample,
    'cartesian_dimension': cartesian_dimension,
    'batch_size': batch_size,
    'n_bins': n_bins,
    'n_epochs': n_epochs,
    'convergence_criteria': convergence_criteria,
    'n_gridpoints': n_gridpoints,
    'history_length': history_length,
    'lr_shrink_lambda': lr_shrink_lambda,
    'lr_growth_lambda': lr_growth_lambda,
    'n_layers': n_layers,
    'n_filters': n_filters,
    'encoder_layers': encoder_layers,
    'encoder_filters': encoder_filters,
    'encoding_output_depth': encoding_output_depth,
    'norm_type_1': norm_type_1,
    'norm_type_2': norm_type_2,
    'conv_embedding_dim': conv_embedding_dim,
}

'''
init model
'''

def make_gaussian_kernel(sigma, kernel_size):
    ks = int(sigma * kernel_size)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


class molecule_autoencoder(nn.Module):
    def __init__(self):
        super(molecule_autoencoder, self).__init__()

        self.device = 'cuda'
        seed = 1
        torch.manual_seed(seed)

        '''
        conditioning model
        '''
        if True:
            self.conditioner = molecule_graph_model(
                dataDims=None,
                seed=seed,
                atom_embedding_dims=n_particle_types + 1,
                num_atom_feats=cartesian_dimension + 1,
                num_mol_feats=0,
                output_dimension=conv_embedding_dim * 3 ** cartesian_dimension,
                activation='gelu',
                num_fc_layers=n_layers,
                fc_depth=n_filters,
                fc_dropout_probability=fc_dropout,
                fc_norm_mode=norm_type_1,
                graph_filters=encoder_filters // 4,
                graph_convolutional_layers=encoder_layers,
                concat_mol_to_atom_features=False,
                pooling=pooling,
                graph_norm=norm_type_2,
                num_spherical=6,
                num_radial=50,
                graph_convolution=gconv_type,
                num_attention_heads=4,
                add_spherical_basis=False,
                add_torsional_basis=False,
                graph_embedding_size=512,
                radial_function='gaussian',
                max_num_neighbors=100,
                convolution_cutoff=2,
                max_molecule_size=1,
                return_latent=False,
                crystal_mode=False,
                crystal_convolution_type=None,
                positional_embedding='sph3',
                device='cuda',
                skip_mlp=False,
            )

        '''
        generator model
        common atom types
        '''
        if decoder_mode == 'conv_transpose':
            n_target_bins = n_bins  # / 0.5) + 1 # make up for odd in stride
            strides = [2, 2, 2]  # that brings it to 30 3-7-15-31, -2 for final conv
            current_size = 29
            if n_target_bins < current_size:
                strides = [2, 2]
                current_size = 13
            if n_target_bins < current_size:
                strides = [2]
                current_size = 5

            diff = n_target_bins - current_size
            for _ in range(diff // 2):  # must be an even number of bins in this approach
                strides += [1]  # pad up to the required layers
            resolution_scaling_input = strides
        else:
            resolution_scaling_input = resolution_scaling


        self.decoder = PointCloudDecoder(input_filters=conv_embedding_dim,
                                         n_classes=n_particle_types + 1,
                                         strides=resolution_scaling_input,
                                         mode=decoder_mode)

    def forward(self, x, pos, batch, num_graphs):
        conditions_encoding = self.conditioner(x=x, pos=pos, batch=batch, num_graphs=num_graphs)

        return self.decoder(conditions_encoding)


grid_index = torch.tensor(list(itertools.product([n for n in range(n_bins)], repeat=cartesian_dimension))).cuda()
grid_index -= n_bins // 2  # center basis vectors on the origin
converged = False
hit_max_lr = False
with wandb.init(project='shape_encoding', entity='mkilgour', config=config):
    wandb.log(config)
    torch.random.manual_seed(0)
    model = molecule_autoencoder()

    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=init_lr)#, weight_decay=weight_decay)
    scheduler1 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_shrink_lambda)
    scheduler2 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: lr_growth_lambda)

    wandb.watch((model), log_graph=True, log_freq=100)

    '''
    training loop
    '''

    model.cuda()
    model.train()
    loss_record = []
    for epoch in tqdm.tqdm(range(n_epochs), miniters=n_epochs // 100, mininterval=30):
        epoch_loss = []
        if batch_size < max_batch_size:
            batch_size += max(1, int(batch_size * (-1 + batch_size_growth_lambda)))
        # just make a random one every epoch
        if (epoch % 400 == 0) and (epoch > 400):
            if avg_num_particles_per_sample < max_num_particles:
                avg_num_particles_per_sample += 1  # add one particle per hundred epochs
        batch = torch.randint(low=0, high=batch_size, size=(batch_size * avg_num_particles_per_sample, 1), device='cuda')[:, 0]  # batch index
        batch = batch[torch.argsort(batch)]
        particle_coords = torch.rand(batch_size * avg_num_particles_per_sample, cartesian_dimension, device='cuda') * 2 - 1  # particle positions with mean of avg_num_particles_per_sample particles per sample
        particle_types = torch.randint(low=1, high=n_particle_types + 1, size=(batch_size * avg_num_particles_per_sample, 1), device='cuda')[:, 0]

        output = model(x=torch.cat((particle_types[:, None], particle_coords), dim=-1),
                       pos=particle_coords,
                       batch=batch,
                       num_graphs=batch_size)


        n_bins = output.shape[-1]
        buckets = torch.bucketize(particle_coords, torch.linspace(-1, 1, n_bins + 1, device='cuda')) - 1

        if cartesian_dimension == 2:
            target = torch.zeros((batch_size, n_bins, n_bins), dtype=torch.long, device='cuda')
            for ii in range(batch_size):
                target[ii, buckets[batch == ii, 0], buckets[batch == ii, 1]] = particle_types[batch == ii]
        elif cartesian_dimension == 3:
            target = torch.zeros((batch_size, n_bins, n_bins, n_bins), dtype=torch.long, device='cuda')
            for ii in range(batch_size):
                target[ii, buckets[batch == ii, 0], buckets[batch == ii, 1], buckets[batch == ii, 2]] = particle_types[batch == ii]

        target = F.one_hot(target, num_classes = n_particle_types + 1).float()
        target = torch.permute(target, (0,4,1,2,3))

        # gaussian_kernel = make_gaussian_kernel(gauss_sigma, gauss_kernel_size).cuda()
        # k3d = torch.einsum('i,j,k->ijk', gaussian_kernel, gaussian_kernel, gaussian_kernel)
        # k3d = k3d / k3d.sum()
        # for cn in range(1, 1+ n_particle_types):
        #     target[:,cn][:,None,:,:,:] = F.conv3d(target[:, cn, ...][:, None, :, :, :], k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(gaussian_kernel) // 2)
        #
        # target = target / torch.linalg.norm(target,dim=1)[:,None,:,:,:]

        bce_loss = F.cross_entropy(output, target) #/ (torch.sum(target > 0) / len(target.flatten()))
        loss = bce_loss

        '''
        fig = go.Figure()
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
            )))
        fig.update_layout(showlegend=True)
        fig.show(renderer='browser')
        '''

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.detach())

        lr = optimizer.param_groups[0]['lr']
        if lr >= max_lr:
            hit_max_lr = True
        if not hit_max_lr:
            scheduler2.step()
        else:
            if lr > min_lr:
                scheduler1.step()

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
                if (convergence_rate < convergence_criteria) or (trailing_best > best_prior):  # converged or diverged
                    aa = 1
                    # converged = True
                    # print("Converged!")

            wandb.log({'loss': loss * (torch.sum(target > 0).cpu().detach().numpy() / len(target.flatten())),
                       'lr': lr,
                       'best_loss': best_loss * (torch.sum(target > 0).cpu().detach().numpy() / len(target.flatten())),
                       'epoch': epoch,
                       'convergence_rate': convergence_rate,
                       'bce_loss': bce_loss.mean().cpu().detach().numpy(),# * (torch.sum(target > 0).cpu().detach().numpy() / len(target.flatten())),
                       'batch_size': batch_size,
                       })

            if converged:
                break


        if epoch % 200 == 0:
            if loss_record_tensor[-1] == loss_record_tensor.min():
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'config': config},
                           model_name)

            if cartesian_dimension == 2:
                fig = make_subplots(rows=2, cols=4)
                for img_i in range(4):
                    if n_particle_types == 1:
                        output_data = output[img_i, 1].sigmoid()
                    else:
                        output_data = output[img_i].argmax(0)

                    fig.add_trace(go.Heatmap(z=output_data.cpu().detach().numpy()),
                                  row=1, col=img_i + 1)
                    fig.add_trace(go.Heatmap(z=target[img_i].cpu().detach().numpy(), ),
                                  row=2, col=img_i + 1)


            elif cartesian_dimension == 3:
                fig = make_subplots(
                    rows=2, cols=2,
                    specs=[[{'type': 'scene'}, {'type': 'scene'}],
                           [{'type': 'scene'}, {'type': 'scene'}]])

                for img_i in range(2):
                    for img_j in range(2):
                        if n_particle_types == 1:
                            sample_guess = output[img_i * 2 + img_j, 1].sigmoid().cpu().detach().numpy()
                        else:
                            sample_guess = output[img_i * 2 + img_j].argmax(0).cpu().detach().numpy()
                        sample_true = target[img_i * 2 + img_j].argmax(0).cpu().detach().numpy()

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
            # fig.show(renderer='browser')
            wandb.log({"samples": fig})


    loss_record = torch.stack(loss_record).cpu().detach().numpy()

assert False
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
if cartesian_dimension == 2:
    for ii in np.arange(1, 40, 2):
        plt.subplot(5, 8, ii)
        plt.imshow(torch.sigmoid((output[ii])).cpu().detach().numpy())
        plt.subplot(5, 8, ii + 1)
        plt.imshow(target[ii].cpu().detach().numpy())
    plt.tight_layout()
elif cartesian_dimension == 3:
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
        dir = torch.randn((n_dirs, cartesian_dimension), device='cuda')
        dir /= torch.linalg.norm(dir, dim=-1)[:, None]

        #rand_intercepts = torch.random(n_dirs)
        target_weights = target.reshape(batch_size,n_bins**cartesian_dimension) # tested - same indexing as target_grid
        target_vectors = grid_index[:, None, :] * target_weights.T[:, :, None]
        target_overlap = torch.einsum('mnj,kj->nkm', (target_vectors, dir))

        output_weights = torch.sigmoid(output.reshape(batch_size,n_bins**cartesian_dimension)) # tested - same indexing as target_grid
        output_vectors = grid_index[:, None, :] * output_weights.T[:, :, None]
        output_overlap = torch.einsum('mnj,kj->nkm', (output_vectors, dir))

        emd_loss = torch_emd(target_overlap, output_overlap).mean(1) / (n_bins ** cartesian_dimension) # averge over slices
'''

#
# class graph_encoder(torch.nn.Module):
#     def __init__(self):
#         super(graph_encoder, self).__init__()
#         # grid of points over which to sample
#         gridpoint_lim = 1 - 1 / n_gridpoints
#         grid = make_grid(gridpoint_lim, n_gridpoints, cartesian_dimension)
#         shift_vector = torch.repeat_interleave(grid, batch_size * avg_num_particles_per_sample, dim=0)
#
#         self.register_buffer("grid", grid)
#         self.register_buffer("shift_vector", shift_vector)
#
#         self.num_radial = 50
#         self.sph_od_list = [i for i in range(11)]
#         self.num_spherical = int(torch.sum(torch.Tensor(self.sph_od_list) * 2 + 1))
#         self.encoding_dim = encoding_output_depth
#         self.init_transform = initial_transform
#
#         if embedding_type == 'pos' or embedding_type == 'polar':
#             embedding_dim = cartesian_dimension
#         elif embedding_type == 'rad':
#             embedding_dim = self.num_radial
#         elif embedding_type == 'sph':
#             embedding_dim = self.num_radial + self.num_spherical
#
#         if decoder_type == 'mlp':
#             self.output_dim = (n_particle_types + 1) * n_bins ** cartesian_dimension
#         elif decoder_type == 'conv':
#             initial_conv_dim = 128
#             if cartesian_dimension == 2:
#                 self.output_dim = 3 * 3 * initial_conv_dim
#                 image_dim = self.output_dim // 3 // 3
#             elif cartesian_dimension == 3:
#                 self.output_dim = 3 * 3 * 3 * initial_conv_dim
#                 image_dim = self.output_dim // 3 // 3 // 3
#
#             self.decoder = ConvDecoder(image_dim, n_particle_types)
#
#         self.mlp = general_MLP(layers=n_layers, filters=n_filters,
#                                input_dim=self.encoding_dim * len(grid),  # self.num_radial * cartesian_dimension, #avg_num_particles_per_sample*cartesian_dimension,
#                                output_dim=self.output_dim,  # n_particle_types * n_bins ** cartesian_dimension,
#                                dropout=0,
#                                norm=norm_type_1,
#                                activation='gelu',
#                                norm_after_linear=True,
#                                )
#
#         self.mlp2 = general_MLP(layers=encoder_layers, filters=encoder_filters,
#                                 input_dim=embedding_dim + 1,
#                                 output_dim=self.encoding_dim,
#                                 activation='gelu',
#                                 norm=norm_type_2,
#                                 dropout=0,
#                                 norm_after_linear=True)
#
#         cutoff = 1  # / n_gridpoints
#         # self.radial_basis = BesselBasisLayer(num_radial=self.num_radial, cutoff=cutoff)
#         self.radial_basis = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=self.num_radial)
#         # self.pos_encoding = PosEncoding2D(self.num_radial, 1)
#         self.global_aggregation = global_aggregation(pooling, filters=self.encoding_dim * len(grid))
#
#     def transform(self, pos, batch):
#         if self.init_transform is None:
#             init_embedding = pos
#
#         return init_embedding
#
#     def encode(self, pos, batch):
#         # return pos.reshape(len(pos), avg_num_particles_per_sample*cartesian_dimension) # just give it the positions directly
#         # encoding = self.pos_encoding(pos)
#         # return scatter(encoding,batch,dim=0)
#
#         type = pos[:, 0].tile(len(self.grid), 1)  # atom type
#         pos_i = pos[:, 1:].tile(len(self.grid), 1)  # atom position
#         batch_i = batch.tile(len(self.grid))
#         pos_i = pos_i - self.shift_vector  # shifted coordinates
#
#         if embedding_type == 'pos':
#             embedding = pos_i
#         elif embedding_type == 'polar':
#             embedding = torch.zeros_like(pos_i)
#             rho = torch.norm(pos_i, p=2, dim=-1).view(-1, 1)
#             # rho = rho / rho.max()
#
#             theta = torch.atan2(pos_i[..., 1], pos_i[..., 0]).view(-1, 1)
#             theta = theta + (theta < 0).type_as(theta) * (2 * torch.pi)
#             theta = theta / (2 * torch.pi)
#
#             embedding[:, 0] = rho[:, 0]
#             embedding[:, 1] = theta[:, 0]
#
#             if cartesian_dimension == 3:
#                 phi = torch.acos(pos_i[..., 2] / rho.view(-1)).view(-1, 1)
#                 phi = phi / torch.pi
#                 embedding[:, 2] = phi[:, 0]
#         elif embedding_type == 'rad':
#             dists = torch.linalg.norm(pos_i, dim=-1)
#             embedding = self.radial_basis(dists)
#         elif embedding_type == 'sph':
#             dists = torch.linalg.norm(pos_i, dim=-1)
#             rbf = self.radial_basis(dists)
#             if cartesian_dimension == 2:
#                 sbf = o3.spherical_harmonics(self.sph_od_list, x=torch.cat((pos_i, torch.zeros_like(pos_i)), dim=-1)[:, :3], normalize=True, normalization='component')
#             elif cartesian_dimension == 3:
#                 sbf = o3.spherical_harmonics(self.sph_od_list, x=pos_i, normalize=True, normalization='component')
#             embedding = torch.cat((rbf, sbf), dim=-1)
#
#         encoding = self.mlp2(torch.cat((type[0, :, None], embedding), dim=-1), batch=batch_i)
#         encoding = torch.hstack(encoding.split(batch_size * avg_num_particles_per_sample, dim=0))
#         graph_output = self.global_aggregation(encoding, pos=None, batch=batch, output_dim=batch_size)
#
#         if len(graph_output) != batch_size:
#             assert False
#
#         return graph_output
#
#     def forward(self, pos, batch):
#         init_embedding = self.transform(pos, batch)
#         encoding = self.encode(init_embedding, batch)
#         output = self.mlp(encoding)
#         if decoder_type == 'mlp':
#             return output
#         elif decoder_type == 'conv':
#             return self.decoder(output)
#
#
#
#
# class ConvDecoder(torch.nn.Module):
#     def __init__(self, image_dim, n_particle_types):
#         super(ConvDecoder, self).__init__()
#         self.num_blocks = 5
#
#         image_depths = torch.linspace(image_dim, n_particle_types + 1, self.num_blocks + 1).long()
#
#         if cartesian_dimension == 2:
#             conv = nn.Conv2d
#             bn = nn.BatchNorm2d
#             self.unflatten = nn.Unflatten(dim=1, unflattened_size=(image_dim, 3, 3))
#
#         elif cartesian_dimension == 3:
#             conv = nn.Conv3d
#             bn = nn.BatchNorm3d
#             self.unflatten = nn.Unflatten(dim=1, unflattened_size=(image_dim, 3, 3,3))
#
#         # self.upsample_blocks = torch.nn.ModuleList([
#         #     nn.Upsample(scale_factor=1.5)
#         #     for n in range(self.num_blocks)
#         # ])
#
#         # self.conv_blocks = torch.nn.ModuleList([
#         #     conv(in_channels=image_depths[n], out_channels=image_depths[n + 1], kernel_size=3, stride=1, padding=1)
#         #     for n in range(self.num_blocks)
#         # ])
#         # self.bn_blocks = torch.nn.ModuleList([
#         #     bn(image_depths[n + 1])
#         #     for n in range(self.num_blocks)
#         # ])
#         # self.conv_blocks2 = torch.nn.ModuleList([
#         #     conv(in_channels=image_depths[-1], out_channels=image_depths[-1], kernel_size=3, stride=1, padding=1)
#         #     for n in range(self.num_blocks)
#         # ])
#         # self.bn_blocks2 = torch.nn.ModuleList([
#         #     bn(image_depths[-1])
#         #     for n in range(self.num_blocks)
#         # ])
#
#         self.final_conv = conv(in_channels=image_depths[-1], out_channels=n_particle_types + 1, kernel_size=3, stride=1,padding=0)
#
#         strides = [2,2,1,1,1]
#         self.upsample_blocks = torch.nn.ModuleList([
#             nn.Identity()
#             for n in range(self.num_blocks)
#         ])
#
#         self.conv_blocks = torch.nn.ModuleList([
#             nn.ConvTranspose2d(in_channels=image_depths[n], out_channels=image_depths[n+1], kernel_size=3, stride=strides[n], output_padding=0)
#             for n in range(self.num_blocks)
#         ])
#         self.bn_blocks = torch.nn.ModuleList([
#             nn.BatchNorm2d(image_depths[n+1])
#             for n in range(self.num_blocks)
#         ])
#
#     def forward(self, x):
#         x = self.unflatten(x)
#         for i, (us, conv, bn) in enumerate(zip(self.upsample_blocks, self.conv_blocks, self.bn_blocks)):
#             x = F.leaky_relu_(bn(conv(us(x))))
#
#         # for i, (conv, bn) in enumerate(zip(self.conv_blocks2, self.bn_blocks2)):
#         #     x = x + F.leaky_relu_(bn(conv(x)))
#
#         x = self.final_conv(x)
#
#         return x
#

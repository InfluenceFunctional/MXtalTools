'''Import statements'''
from models.MikesGraphNet import MikesGraphNet
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.global_aggregation import global_aggregation
from models.components import MLP
from torch.distributions import MultivariateNormal
import numpy as np
from ase import Atoms
from models.asymmetric_radius_graph import asymmetric_radius_graph


class molecule_graph_model(nn.Module):
    def __init__(self, dataDims, seed,
                 num_atom_feats,
                 num_mol_feats,
                 output_dimension,
                 activation,
                 num_fc_layers,
                 fc_depth,
                 fc_dropout_probability,
                 fc_norm_mode,
                 graph_filters,
                 graph_convolutional_layers,
                 concat_mol_to_atom_features,
                 pooling,
                 graph_norm,
                 num_spherical,
                 num_radial,
                 graph_convolution,
                 num_attention_heads,
                 add_spherical_basis,
                 add_torsional_basis,
                 graph_embedding_size,
                 radial_function,
                 max_num_neighbors,
                 convolution_cutoff,
                 max_molecule_size,
                 return_latent=False,
                 crystal_mode=False,
                 crystal_convolution_type=None,
                 positional_embedding='sph',
                 atom_embedding_dims=5,
                 device='cuda'):

        super(molecule_graph_model, self).__init__()
        # initialize constants and layers
        self.device = device
        self.return_latent = return_latent
        self.activation = activation
        self.num_fc_layers = num_fc_layers
        self.fc_depth = fc_depth
        self.fc_dropout_probability = fc_dropout_probability
        self.fc_norm_mode = fc_norm_mode
        self.graph_convolution = graph_convolution
        self.output_classes = output_dimension
        self.graph_convolution_layers = graph_convolutional_layers
        self.graph_filters = graph_filters
        self.graph_norm = graph_norm
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_attention_heads = num_attention_heads
        self.add_spherical_basis = add_spherical_basis
        self.add_torsional_basis = add_torsional_basis
        self.n_mol_feats = num_mol_feats  # dataDims['num mol features']
        self.n_atom_feats = num_atom_feats  # dataDims['num atom features']
        self.radial_function = radial_function
        self.max_num_neighbors = max_num_neighbors
        self.graph_convolution_cutoff = convolution_cutoff
        if not concat_mol_to_atom_features:  # if we are not adding molwise feats to atoms, subtract the dimension
            self.n_atom_feats -= self.n_mol_feats
        self.pooling = pooling
        self.fc_norm_mode = fc_norm_mode
        self.graph_embedding_size = graph_embedding_size
        self.crystal_mode = crystal_mode
        self.crystal_convolution_type = crystal_convolution_type
        self.max_molecule_size = max_molecule_size
        self.atom_embedding_dims = atom_embedding_dims  # todo clean this up

        if dataDims is None:
            self.num_atom_types = 101
        else:
            self.num_atom_types = list(dataDims['atom embedding dict sizes'].values())[0] + 1

        torch.manual_seed(seed)

        self.graph_net = MikesGraphNet(
            crystal_mode=crystal_mode,
            crystal_convolution_type=self.crystal_convolution_type,
            graph_convolution_filters=self.graph_filters,
            graph_convolution=self.graph_convolution,
            out_channels=self.fc_depth,
            hidden_channels=self.graph_embedding_size,
            num_blocks=self.graph_convolution_layers,
            num_radial=self.num_radial,
            num_spherical=self.num_spherical,
            max_num_neighbors=self.max_num_neighbors,
            cutoff=self.graph_convolution_cutoff,
            activation='gelu',
            embedding_hidden_dimension=self.atom_embedding_dims,
            num_atom_features=self.n_atom_feats,
            norm=self.graph_norm,
            dropout=self.fc_dropout_probability,
            spherical_embedding=self.add_spherical_basis,
            torsional_embedding=self.add_torsional_basis,
            radial_embedding=self.radial_function,
            num_atom_types=self.num_atom_types,
            attention_heads=self.num_attention_heads,
        )

        # initialize global pooling operation
        self.global_pool = global_aggregation(self.pooling, self.fc_depth,
                                              geometric_embedding=positional_embedding,
                                              num_radial=num_radial,
                                              spherical_order=num_spherical,
                                              radial_embedding=radial_function,
                                              max_molecule_size=max_molecule_size)

        # molecule features FC layer
        if self.n_mol_feats != 0:
            self.mol_fc = nn.Linear(self.n_mol_feats, self.n_mol_feats)

        # FC model to post-process graph fingerprint
        if self.num_fc_layers > 0:
            self.gnn_mlp = MLP(layers=self.num_fc_layers,
                               filters=self.fc_depth,
                               norm=self.fc_norm_mode,
                               dropout=self.fc_dropout_probability,
                               input_dim=self.fc_depth,
                               output_dim=self.fc_depth,
                               conditioning_dim=self.n_mol_feats,
                               seed=seed
                               )
        else:
            self.gnn_mlp = nn.Identity()

        if self.fc_depth != self.output_classes:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(self.fc_depth, self.output_classes, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self, data=None, x=None, pos=None, batch=None, ptr=None, aux_ind=None, num_graphs=None, return_latent=False, return_dists=False):
        if data is not None:
            x = data.x
            pos = data.pos
            batch = data.batch
            aux_ind = data.aux_ind
            ptr = data.ptr
            num_graphs = data.num_graphs

        extra_outputs = {}
        if self.n_mol_feats > 0:
            mol_feats = self.mol_fc(x[ptr[:-1], -self.n_mol_feats:])  # molecule features are repeated, only need one per molecule (hence data.ptr)
        else:
            mol_feats = None

        x, dists_dict = self.graph_net(x[:, :self.n_atom_feats], pos, batch, ptr=ptr, ref_mol_inds=aux_ind, return_dists=return_dists)  # get atoms encoding

        if self.crystal_mode:  # model only outputs ref mol atoms - many fewer
            x = self.global_pool(x, pos, batch[torch.where(aux_ind == 0)[0]], output_dim=num_graphs)
        else:
            x = self.global_pool(x, pos, batch, output_dim=num_graphs)  # aggregate atoms to molecule

        if self.num_fc_layers > 0:
            x = self.gnn_mlp(x, conditions=mol_feats)  # mix graph fingerprint with molecule-scale features

        output = self.output_fc(x)

        if return_dists:
            extra_outputs['dists dict'] = dists_dict
        if return_latent:
            extra_outputs['latent'] = output.cpu().detach().numpy()

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output


class independent_gaussian_model(nn.Module):
    def __init__(self, input_dim, means, stds, normed_length_means, normed_length_stds, cov_mat=None):
        super(independent_gaussian_model, self).__init__()

        self.input_dim = input_dim
        fixed_norms = torch.Tensor(means)
        fixed_norms[:3] = torch.Tensor(normed_length_means)
        fixed_stds = torch.Tensor(stds)
        fixed_stds[:3] = torch.Tensor(normed_length_stds)

        self.register_buffer('means', torch.Tensor(means))
        self.register_buffer('stds', torch.Tensor(stds))
        self.register_buffer('fixed_norms', torch.Tensor(fixed_norms))
        self.register_buffer('fixed_stds', torch.Tensor(fixed_stds))

        if cov_mat is not None:
            pass
        else:
            cov_mat = torch.diag(torch.Tensor(fixed_stds).pow(2))

        fixed_means = means.copy()
        fixed_means[:3] = normed_length_means
        self.prior = MultivariateNormal(fixed_norms, torch.Tensor(cov_mat))  # apply standardization
        self.dummy_params = nn.Parameter(torch.ones(100))

    def forward(self, num_samples, data):
        '''
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        '''
        # conditions are unused - dummy
        # denormalize sample before standardizing
        samples = self.prior.sample((num_samples,)).to(data.x.device)
        samples[:, :3] = samples[:, :3] * (data.Z[:, None] ** (1 / 3)) * (data.mol_volume[:, None] ** (1 / 3))
        return (samples - self.means.to(samples.device)) / self.stds.to(samples.device)  # we want samples in standardized basis

    def backward(self, samples):
        return samples * self.stds + self.means

    def score(self, samples):
        return self.prior.log_prob(samples)


class PointCloudDecoder(nn.Module):
    def __init__(self, input_filters, n_classes, strides, init_image_size):
        super(PointCloudDecoder, self).__init__()
        '''
        model to deconvolve a 1D vector to an NxNxN array of voxel classwise probabilities
        '''
        self.strides = strides
        self.num_blocks = len(strides)

        img_size = [init_image_size]
        for i, stride in enumerate(self.strides):
            if stride == 2:
                img_size += [img_size[i] + img_size[i] + 1]
            elif stride == 1:
                img_size += [img_size[i] + 2]
            elif stride == 3:
                img_size += [img_size[i] + 2*img_size[i]]
            elif stride == 4:
                img_size += [img_size[i] + 3*img_size[i] - 1]

        n_voxels = torch.Tensor(img_size) ** 3
        self.image_depths = torch.maximum(torch.ceil(n_voxels[0] * input_filters / n_voxels),torch.ones_like(n_voxels)*64).long()
        #total_layer_size = n_voxels * self.image_depths
        #self.image_depths = torch.linspace(input_filters, n_classes, self.num_blocks + 1).long()
        #self.image_depths = torch.linspace(input_filters, input_filters, self.num_blocks + 1).long()


        conv = nn.ConvTranspose3d
        bn = nn.LayerNorm  # nn.InstanceNorm3d #nn.BatchNorm3d

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(input_filters, init_image_size, init_image_size, init_image_size))

        # stride 4 adds 3N - 1
        # stride 3 adds 2N
        # stride 2 adds N+1
        # stride 1 adds 2
        self.conv_blocks = torch.nn.ModuleList([
            conv(in_channels=self.image_depths[n], out_channels=self.image_depths[n + 1], kernel_size=3, stride=strides[n], output_padding=0)
            for n in range(self.num_blocks)
        ])

        # for layer norm
        self.bn_blocks = torch.nn.ModuleList([
            bn([img_size[n + 1], img_size[n + 1], img_size[n + 1]])
            for n in range(self.num_blocks)
        ])
        self.final_conv = nn.Conv3d(in_channels=self.image_depths[-1], out_channels=n_classes, kernel_size=(1,1,1), padding=0)
        #self.final_conv = nn.Conv3d(in_channels=self.image_depths[-1], out_channels=n_classes, kernel_size=(3,3,3), padding=0)


    def forward(self, x):
        x = self.unflatten(x)
        for bn, conv in zip(self.bn_blocks, self.conv_blocks):  # upscale and deconvolve
            x = F.gelu(bn(conv(x)))

        return self.final_conv(x)  # process to output


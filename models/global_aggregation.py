import torch
import torch.nn as nn
from models.model_components import general_MLP
from e3nn.o3 import spherical_harmonics
from models.basis_functions import BesselBasisLayer, GaussianEmbedding
from torch_geometric import nn as gnn


class global_aggregation(nn.Module):
    '''
    wrapper for several types of global aggregation functions
    '''

    def __init__(self, agg_func, filters):
        super(global_aggregation, self).__init__()
        self.agg_func = agg_func
        if agg_func == 'mean':
            self.agg = gnn.global_mean_pool
        elif agg_func == 'sum':
            self.agg = gnn.global_add_pool
        elif agg_func == 'attention':
            self.agg = gnn.GlobalAttention(nn.Sequential(nn.Linear(filters, filters), nn.LeakyReLU(), nn.Linear(filters, 1)))
        elif agg_func == 'set2set':
            self.agg = gnn.Set2Set(in_channels=filters, processing_steps=4)
            self.agg_fc = nn.Linear(filters * 2, filters)  # condense to correct number of filters
        elif agg_func == 'combo':
            self.agg_list1 = [gnn.global_max_pool, gnn.global_mean_pool, gnn.global_add_pool]  # simple aggregation functions
            self.agg_list2 = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.Linear(filters, filters), nn.LeakyReLU(), nn.Linear(filters, 1)))])  # aggregation functions requiring parameters
            self.agg_fc = nn.Linear(filters * (len(self.agg_list1) + len(self.agg_list2)), filters)  # condense to correct number of filters
        elif agg_func == 'geometric':  # global aggregation via geometry-involved pooling
            self.agg = SphGeoPooling(in_channels=filters,num_radial = 50, spherical_order=11)

    def forward(self, x, pos, batch):
        if self.agg_func == 'set2set':
            x = self.agg(x, batch)
            return self.agg_fc(x)
        elif self.agg_func == 'combo':
            output1 = [agg(x, batch) for agg in self.agg_list1]
            output2 = [agg(x, batch) for agg in self.agg_list2]
            return self.agg_fc(torch.cat((output1 + output2), dim=1))
        elif self.agg_func == 'geometric':
            return self.agg(x, pos, batch)
        else:
            return self.agg(x, batch)


class SphGeoPooling(nn.Module):  # a global aggregation function using spherical harmonics
    def __init__(self, in_channels, num_radial=50, spherical_order=11, cutoff=10,
                 activation='leaky relu', dropout=0, norm=None):
        super(SphGeoPooling, self).__init__()

        # radial and spherical basis layers
        self.spherical_order = spherical_order
        self.sph_od_list = [i for i in range(spherical_order)]
        self.num_spherical = int(torch.sum(torch.Tensor(self.sph_od_list) * 2 + 1))
        self.radial_basis = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff)
        # self.radial_basis = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)

        # message generation
        self.mlp = general_MLP(layers=8, filters=in_channels,
                               input_dim=in_channels + num_radial + self.num_spherical,
                               output_dim=in_channels,
                               activation=activation,
                               norm=norm,
                               dropout=dropout)

        # message aggregation
        self.global_pool = global_aggregation('combo', in_channels)

    def forward(self, x, pos, batch):
        '''
        assume positions are pre-centred on the molecule centroids
        '''

        '''
        generate edge embedding
        '''
        num_graphs = int(batch[-1] + 1)
        dists = torch.linalg.norm(pos, dim=-1)  # centroids are at (0,0,0)
        rbf = self.radial_basis(dists)
        sbf = torch.cat([spherical_harmonics(self.sph_od_list, x=pos[batch == ii], normalize=True, normalization='component') for ii in range(num_graphs)])

        '''
        do node aggregation
        '''
        self.messages = self.mlp(torch.cat((x, rbf, sbf), dim=-1))

        # aggregation
        return self.global_pool(self.messages, pos, batch)

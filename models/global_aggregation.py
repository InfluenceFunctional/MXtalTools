import torch
import torch.nn as nn
from models.model_components import general_MLP
from e3nn.o3 import spherical_harmonics
from models.basis_functions import BesselBasisLayer, GaussianEmbedding
from torch_geometric import nn as gnn
from models.positional_encodings import PosEncoding3D

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
        #self.spherical_order = spherical_order
        #self.sph_od_list = [i for i in range(spherical_order)]
        #self.num_spherical = int(torch.sum(torch.Tensor(self.sph_od_list) * 2 + 1))
        #self.radial_basis = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff)
        # self.radial_basis = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)
        # fourier basis

        self.pos_encoding = PosEncoding3D(in_channels // 3, cutoff)

        # message generation
        self.mlp = general_MLP(layers=4, filters=in_channels,
                               input_dim=in_channels,# + num_radial + self.num_spherical,
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
        # spherical harmonic embedding
        num_graphs = int(batch[-1] + 1)
        #dists = torch.linalg.norm(pos, dim=-1)  # centroids are at (0,0,0)
        #rbf = self.radial_basis(dists)
        #sbf = spherical_harmonics(self.sph_od_list, x=pos, normalize=True, normalization='component')
        #messages = self.mlp(torch.cat((x, rbf, sbf), dim=-1))

        tot_emb = self.pos_encoding(pos)
        #messages = self.mlp(torch.cat((x, tot_emb), dim=-1))
        messages = self.mlp(x + tot_emb)

        '''
        do node aggregation
        '''

        # aggregation
        return self.global_pool(messages, pos, batch)


''' embedding test
import matplotlib.pyplot as plt
d1 = torch.cdist(pos,pos,p=2).cpu().detach().numpy()
d2 = torch.cdist(tot_emb,tot_emb,p=2).cpu().detach().numpy()
plt.clf()
plt.scatter(d1.flatten()[:10000],d2.flatten()[:10000])


xx = torch.rand(100,3).to('cuda')*20
dists = torch.linalg.norm(xx, dim=-1)  # centroids are at (0,0,0)
rbf = self.radial_basis(dists)
sbf = spherical_harmonics(self.sph_od_list, x=xx, normalize=True, normalization='component')
reps = torch.cat((rbf,sbf),dim=-1)
x,y,z = xx.T
channels = 25
inv_freq = 1 / ((xx.max()-xx.min()) *10000 ** (torch.arange(0, channels, 2,device='cuda').float() / channels))
x_emb = get_emb(torch.einsum('i,j->ij',(x,inv_freq)))
y_emb = get_emb(torch.einsum('i,j->ij',(y,inv_freq)))
z_emb = get_emb(torch.einsum('i,j->ij',(z,inv_freq)))
tot_emb = torch.cat((x_emb,y_emb,z_emb),dim=-1)
d1 = torch.cdist(xx,xx,p=2).cpu().detach().numpy()
d2 = torch.cdist(reps,reps,p=2).cpu().detach().numpy()
d3 = torch.cdist(tot_emb, tot_emb, p=2).cpu().detach().numpy()
plt.clf()
plt.subplot(1,2,1)
plt.scatter(d1.flatten(),d3.flatten(),alpha=0.25)
plt.subplot(2,2,2)
plt.imshow(reps.cpu().detach().numpy())
plt.subplot(2,2,4)
plt.imshow(tot_emb.cpu().detach().numpy())

'''


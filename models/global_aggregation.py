import torch
import torch.nn as nn
from models.components import MLP
from torch_geometric import nn as gnn
from torch_scatter import scatter


class global_aggregation(nn.Module):
    """
    wrapper for several types of global aggregation functions
    NOTE - I believe PyG might have a new built-in method which does exactly this
    """

    def __init__(self, agg_func, depth):  # todo rewrite this with new pyg aggr class and/or custom functions (e.g., scatter)
        super(global_aggregation, self).__init__()
        self.agg_func = agg_func
        if agg_func == 'mean':
            self.agg = gnn.global_mean_pool
        elif agg_func == 'sum':
            self.agg = gnn.global_add_pool
        elif agg_func == 'max':
            self.agg = gnn.global_max_pool
        elif self.agg_func == '1o max':
            pass
        elif agg_func == 'attention':
            self.agg = gnn.GlobalAttention(nn.Sequential(nn.Linear(depth, depth), nn.LeakyReLU(), nn.Linear(depth, 1)))
        elif agg_func == 'set2set':
            self.agg = gnn.Set2Set(in_channels=depth, processing_steps=4)
            self.agg_fc = nn.Linear(depth * 2, depth)  # condense to correct number of filters
        elif agg_func == 'simple combo':
            self.agg_list1 = [gnn.global_max_pool, gnn.global_mean_pool, gnn.global_add_pool]  # simple aggregation functions
            self.agg_fc = MLP(
                layers=1,
                filters=depth,
                input_dim=depth * (len(self.agg_list1)),
                output_dim=depth,
                norm=None,
                dropout=0)  # condense to correct number of filters
        elif agg_func == 'mean sum':  # todo add a max aggregator which picks max by vector length (equivariant!)
            pass
        elif agg_func == 'combo':
            self.agg_list1 = [gnn.global_max_pool, gnn.global_mean_pool, gnn.global_add_pool]  # simple aggregation functions
            self.agg_list2 = nn.ModuleList([gnn.GlobalAttention(
                MLP(input_dim=depth,
                    output_dim=1,
                    layers=1,
                    filters=depth,
                    activation='leaky relu',
                    norm=None),
            )])  # aggregation functions requiring parameters
            self.agg_fc = MLP(
                layers=1,
                filters=depth,
                input_dim=depth * (len(self.agg_list1) + 1),
                output_dim=depth,
                norm=None,
                dropout=0)  # condense to correct number of filters
        elif agg_func == 'molwise':
            self.agg = gnn.pool.max_pool_x
        elif agg_func is None:
            self.agg = nn.Identity()

    def forward(self, x, batch, cluster=None, output_dim=None, v=None):
        if self.agg_func == 'set2set':
            x = self.agg(x, batch, size=output_dim)
            return self.agg_fc(x)
        elif self.agg_func == 'combo':
            output1 = [agg(x, batch, size=output_dim) for agg in self.agg_list1]
            output2 = [agg(x, batch, size=output_dim) for agg in self.agg_list2]
            # output3 = [agg(x, batch, 3, size = output_dim) for agg in self.agg_list3]
            return self.agg_fc(torch.cat((output1 + output2), dim=1))
        elif self.agg_func == 'simple combo':
            output1 = [agg(x, batch, size=output_dim) for agg in self.agg_list1]
            return self.agg_fc(torch.cat(output1, dim=1))
        elif self.agg_func is None:
            return x  # do nothing
        elif self.agg_func == 'molwise':
            return self.agg(cluster=cluster, batch=batch, x=x)[0]
        elif self.agg_func == 'mean sum':
            return (scatter(x, batch, dim_size=output_dim, dim=0, reduce='mean') +
                    scatter(x, batch, dim_size=output_dim, dim=0, reduce='sum'))
        elif self.agg_func == 'equivariant max':  # todo equivariant attention aggregation
            # assume the input is nx3xk dimensional
            agg = torch.stack([v[batch == bind][x[batch == bind].argmax(dim=0), :, torch.arange(v.shape[-1])] for bind in range(batch[-1] + 1)])
            return scatter(x, batch, dim_size=output_dim, dim=0, reduce='max'), agg
        else:
            return self.agg(x, batch, size=output_dim)

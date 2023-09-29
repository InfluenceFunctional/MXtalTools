import torch
import torch.nn as nn
from models.components import MLP
from torch_geometric import nn as gnn


class global_aggregation(nn.Module):
    """
    wrapper for several types of global aggregation functions
    NOTE - I believe PyG might have a new built-in method which does exactly this
    """

    def __init__(self, agg_func, depth):
        super(global_aggregation, self).__init__()
        self.agg_func = agg_func
        if agg_func == 'mean':
            self.agg = gnn.global_mean_pool
        elif agg_func == 'sum':
            self.agg = gnn.global_add_pool
        elif agg_func == 'max':
            self.agg = gnn.global_max_pool
        elif agg_func == 'attention':
            self.agg = gnn.GlobalAttention(nn.Sequential(nn.Linear(depth, depth), nn.LeakyReLU(), nn.Linear(depth, 1)))
        elif agg_func == 'set2set':
            self.agg = gnn.Set2Set(in_channels=depth, processing_steps=4)
            self.agg_fc = nn.Linear(depth * 2, depth)  # condense to correct number of filters
        elif agg_func == 'combo':
            self.agg_list1 = [gnn.global_max_pool, gnn.global_mean_pool, gnn.global_add_pool]  # simple aggregation functions
            # self.agg_list3 = [gnn.global_sort_pool]
            # self.agg_list2 = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.Linear(filters, filters), nn.LeakyReLU(), nn.Linear(filters, 1)))])  # aggregation functions requiring parameters
            self.agg_list2 = nn.ModuleList([gnn.GlobalAttention(
                MLP(input_dim=depth,
                    output_dim=1,
                    layers=4,
                    filters=depth,
                    activation='leaky relu',
                    norm=None),
                # nn.Sequential(nn.Linear(filters, filters), nn.LeakyReLU(), nn.Linear(filters, 1))
            )])  # aggregation functions requiring parameters
            self.agg_fc = MLP(
                layers=1,
                filters=depth,
                input_dim=depth * (len(self.agg_list1) + 1),
                output_dim=depth,
                norm=None,
                dropout=0)  # condense to correct number of filters

    def forward(self, x, pos, batch, output_dim=None):
        if self.agg_func == 'set2set':
            x = self.agg(x, batch, size=output_dim)
            return self.agg_fc(x)
        elif self.agg_func == 'combo':
            output1 = [agg(x, batch, size=output_dim) for agg in self.agg_list1]
            output2 = [agg(x, batch, size=output_dim) for agg in self.agg_list2]
            # output3 = [agg(x, batch, 3, size = output_dim) for agg in self.agg_list3]
            return self.agg_fc(torch.cat((output1 + output2), dim=1))
        else:
            return self.agg(x, batch, size=output_dim)



from models.basis_functions import GaussianEmbedding, BesselBasisLayer
import torch
import torch.nn as nn

from models.gnn_blocks import EmbeddingBlock, GCBlock, FC_Block, EquivariantEmbeddingBlock
import e3nn.o3 as o3


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self,
                 input_node_depth: int,
                 node_embedding_depth: int,
                 nodewise_fc_layers: int,
                 message_depth: int,
                 convolution_type: str,
                 graph_embedding_depth: int,
                 num_blocks: int,
                 num_radial: int,
                 num_embedding_types=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 embedding_hidden_dimension=5,
                 message_norm=None,
                 message_dropout=0,
                 nodewise_norm=None,
                 nodewise_dropout=0,
                 radial_embedding='bessel',
                 attention_heads=1,
                 periodize_inside_nodes=False,
                 outside_convolution_type='none',
                 equivariant_graph=False,
                 sh_order=None,
                 ):
        super(GraphNeuralNetwork, self).__init__()

        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.periodize_inside_nodes = periodize_inside_nodes
        self.outside_convolution_type = outside_convolution_type
        self.equivariant_graph = equivariant_graph
        if self.equivariant_graph:
            self.sh_order = sh_order
            self.irreps_sh = o3.Irreps.spherical_harmonics(self.sh_order)
            self.sh_dim = self.irreps_sh.dim
            self.irreps_node = o3.Irreps('129x0e + 128x1o')
            self.irreps_out = o3.Irreps('171x1o')
        else:
            self.irreps_sh = None
            self.sh_dim = 0
            self.irreps_node = None

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)

        if self.equivariant_graph:
            irreps_in = o3.Irreps(f"{input_node_depth - 1 + embedding_hidden_dimension}x0e") + self.irreps_sh
            self.atom_embedding = EquivariantEmbeddingBlock(
                irreps_in,
                self.irreps_node,
                num_embedding_types,
                embedding_hidden_dimension
            )
        else:
            self.atom_embedding = EmbeddingBlock(node_embedding_depth,
                                                 num_embedding_types,
                                                 input_node_depth,
                                                 embedding_hidden_dimension,
                                                 )

        self.interaction_blocks = torch.nn.ModuleList([
            GCBlock(message_depth,
                    node_embedding_depth,
                    num_radial,
                    norm=message_norm,
                    dropout=message_dropout,
                    heads=attention_heads,
                    equivariant=equivariant_graph,
                    irreps=self.irreps_node,
                    )
            for _ in range(num_blocks)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            FC_Block(
                nodewise_fc_layers,
                node_embedding_depth,
                activation,
                nodewise_norm,
                nodewise_dropout,
                equivariant=self.equivariant_graph,
                irreps=self.irreps_node
            )
            for _ in range(num_blocks)
        ])

        if self.equivariant_graph and self.irreps_node != self.irreps_out:
            self.output_layer = o3.Linear(self.irreps_node, self.irreps_out)
        elif not self.equivariant_graph and node_embedding_depth != graph_embedding_depth:
            self.output_layer = nn.Linear(node_embedding_depth, graph_embedding_depth)
        else:
            self.output_layer = nn.Identity()

    def get_geom_embedding(self, edge_index, pos):
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)

    def forward(self, z, pos, batch, ptr, edges_dict: dict):
        """
        # todo write docstring
        """
        # graph model starts here

        x = self.atom_embedding(z)  # embed atomic numbers & compute initial atom-wise feature vector #

        if self.outside_convolution_type == 'none':  # assumes input with inside-outside structure, and enforces periodicity after each convolution
            edge_index = edges_dict['edge_index']
            dist, rbf = self.get_geom_embedding(edge_index, pos)
            assert not self.periodize_inside_nodes, "Cannot periodize to outside nodes if there are no outside nodes"
        elif self.outside_convolution_type == 'all_layers':
            edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
            edge_index = torch.cat((edge_index, edge_index_inter), dim=1)  # all edges counted in one big batch
            dist, rbf = self.get_geom_embedding(edge_index, pos)
        elif self.outside_convolution_type == 'last_layer':
            edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
            dist, rbf = self.get_geom_embedding(edge_index, pos)
            dist_inter, rbf_inter = self.get_geom_embedding(torch.cat((edge_index, edge_index_inter), dim=1), pos)
        else:
            assert False, "Must select a valid treatment of inside vs outside nodes"

        for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
            if n == (len(self.interaction_blocks) - 1) and self.outside_convolution_type == 'last_layer':
                x = convolution(x, rbf_inter, torch.cat((edge_index, edge_index_inter), dim=1), batch)  # return only the results of the intermolecular convolution, omitting intramolecular features
            else:
                x = x + convolution(x, rbf, edge_index, batch)  # graph convolution  # todo sort out residual

            if not self.periodize_inside_nodes:
                x = fc(x, batch=batch)  # feature-wise 1D convolution, residual is already inside
            else:  # todo note the below has an extra residue
                x[inside_inds] = x[inside_inds] + fc(x[inside_inds], batch=batch[inside_inds])  # update only the inside inds

                for ii in range(len(ptr) - 1):  # enforce periodicity for each crystal, assuming invariant node features
                    x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)  # copy the first unit cell to all periodic images

                if n == len(self.interaction_blocks) - 1:
                    x = x[inside_inds]  # reduce to inside image

        return self.output_layer(x)


'''
import networkx as nx
import matplotlib.pyplot as plt
intra_edges = (edge_index[:, edge_index[0, :] < ptr[1]].cpu().detach().numpy().T)
inter_edges = (edge_index_inter[:, edge_index_inter[0, :] < ptr[1]].cpu().detach().numpy().T)
plt.clf()
G = nx.Graph()
G = G.to_directed()
G.add_weighted_edges_from(np.concatenate((intra_edges, np.ones(len(intra_edges))[:, None] * 2), axis=1))
G.add_weighted_edges_from(np.concatenate((inter_edges, np.ones(len(inter_edges))[:, None] * 0.25), axis=1))
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
node_weights = np.concatenate((np.ones(9)*2, np.ones(len(G.nodes)-9)))
nx.draw_kamada_kawai(G, arrows=True, node_size=node_weights * 100, edge_color=weights, linewidths = 1, width=weights, 
edge_cmap=plt.cm.RdYlGn, node_color = node_weights, cmap=plt.cm.RdYlGn)
'''

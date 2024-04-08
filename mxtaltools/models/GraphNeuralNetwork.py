from mxtaltools.models.basis_functions import GaussianEmbedding, BesselBasisLayer
import torch
import torch.nn as nn

from mxtaltools.models.gnn_blocks import EmbeddingBlock, GC_Block, FC_Block
from mxtaltools.models.utils import get_model_nans


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self,
                 input_node_depth: int,
                 node_embedding_depth: int,
                 nodewise_fc_layers: int,
                 message_depth: int,
                 graph_embedding_depth: int,
                 num_blocks: int,
                 num_radial: int,
                 num_embedding_types=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 embedding_hidden_dimension=5,
                 message_dropout=0,
                 nodewise_norm=None,
                 nodewise_dropout=0,
                 radial_embedding='bessel',
                 attention_heads=1,
                 periodize_inside_nodes=False,
                 outside_convolution_type='none',
                 equivariant_graph=False,
                 vector_norm=None,
                 skip_embedding=False
                 ):
        super(GraphNeuralNetwork, self).__init__()

        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.periodize_inside_nodes = periodize_inside_nodes
        self.outside_convolution_type = outside_convolution_type
        self.equivariant_graph = equivariant_graph

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)

        if not skip_embedding:
            self.init_node_embedding = EmbeddingBlock(node_embedding_depth,
                                                      num_embedding_types,
                                                      input_node_depth,
                                                      embedding_hidden_dimension,
                                                      )
        else:
            self.init_node_embedding = nn.Identity()

        if self.equivariant_graph:
            self.init_vector_embedding = nn.Linear(1, node_embedding_depth, bias=False)

        self.zeroth_fc_block = FC_Block(
                nodewise_fc_layers,
                node_embedding_depth,
                activation,
                nodewise_norm,
                nodewise_dropout,
                equivariant=self.equivariant_graph,
                vector_norm=vector_norm)

        self.interaction_blocks = torch.nn.ModuleList([
            GC_Block(message_depth,
                     node_embedding_depth,
                     num_radial,
                     dropout=message_dropout,
                     heads=attention_heads,
                     equivariant=equivariant_graph,
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
                vector_norm=vector_norm,
            )
            for _ in range(num_blocks)
        ])

        if self.equivariant_graph:  # todo abstract out output block
            if node_embedding_depth != graph_embedding_depth:
                self.v_output_layer = nn.Linear(node_embedding_depth, graph_embedding_depth, bias=False)
            else:
                self.v_output_layer = nn.Identity()
        if node_embedding_depth != graph_embedding_depth:
            self.output_layer = nn.Linear(node_embedding_depth, graph_embedding_depth, bias=False)
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
        if self.equivariant_graph:
            x, v = z[:, :-3], z[:, -3:]
            v = self.init_vector_embedding(v[:, :, None])
            x = self.init_node_embedding(x)  # embed atomic numbers & compute initial atom-wise feature vector
            x, v = self.zeroth_fc_block(x=x, v=v, batch=batch)
        else:
            x, v = self.init_node_embedding(z), None  # embed atomic numbers & compute initial atom-wise feature vector
            x = self.zeroth_fc_block(x=x, v=v, batch=batch)

        assert torch.sum(torch.isnan(x)) == 0, f"NaN in gnn init output {get_model_nans(self.init_node_embedding)}"

        (edge_index, edge_index_inter,
         inside_batch, inside_inds,
         n_repeats,
         rbf, rbf_inter) = self.get_edge_info(edges_dict, pos)

        for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
            if v is not None:
                res = x.clone()
                vres = v.clone()
                x, v = convolution(x, v, rbf, edge_index, batch)
                x = x + res
                v = 0.7076 * (v + vres)
            else:
                if n == (len(self.interaction_blocks) - 1) and self.outside_convolution_type == 'last_layer':
                    x = convolution(x, v, rbf_inter, torch.cat((edge_index, edge_index_inter), dim=1), batch)  # return only the results of the intermolecular convolution, omitting intramolecular features
                else:
                    x = x + convolution(x, v, rbf, edge_index, batch)  # graph convolution

            assert torch.sum(torch.isnan(x)) == 0, f"NaN in conv output {get_model_nans(self.interaction_blocks)}"

            if not self.periodize_inside_nodes:
                if self.equivariant_graph:
                    xres = x.clone()
                    vres = v.clone()
                    x, v = fc(x, v=v, batch=batch)
                    x = x + xres
                    v = 0.7076 * (v + vres)
                else:
                    x = x + fc(x, v=v, batch=batch)  # feature-wise 1D convolution, residual is already inside

                assert torch.sum(torch.isnan(x)) == 0, f"NaN in fc_block output {get_model_nans(self.fc_blocks)}"

            else:
                assert v is None, "Vector embeddings not yet set up for periodic graphs"
                x[inside_inds] = x[inside_inds] + fc(x[inside_inds], batch=batch[inside_inds])  # update only the inside inds
                x = self.periodize_molecular_crystal(inside_batch, inside_inds, n, n_repeats, ptr, x)

        if v is not None:
            return self.output_layer(x), self.v_output_layer(v)
        else:
            return self.output_layer(x)

    def periodize_molecular_crystal(self, inside_batch, inside_inds, n, n_repeats, ptr, x):
        for ii in range(len(ptr) - 1):  # enforce periodicity for each crystal, assuming invariant node features
            x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)  # copy the first unit cell to all periodic images
        if n == len(self.interaction_blocks) - 1:
            x = x[inside_inds]  # reduce to inside image
        return x

    def get_edge_info(self, edges_dict, pos):
        if self.outside_convolution_type == 'none':  # assumes input with inside-outside structure, and enforces periodicity after each convolution
            edge_index = edges_dict['edge_index']
            if 'dists' in edges_dict.keys():  # previously generated distances - e.g., for periodic MIC
                dist = edges_dict['dists']
                rbf = self.rbf(dist)
            else:
                dist, rbf = self.get_geom_embedding(edge_index, pos)
            edge_index_inter, inside_batch, inside_inds, n_repeats, rbf_inter = None, None, None, None, None
            assert not self.periodize_inside_nodes, "Cannot periodize to outside nodes if there are no outside nodes"
        elif self.outside_convolution_type == 'all_layers':
            edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
            edge_index = torch.cat((edge_index, edge_index_inter), dim=1)  # all edges counted in one big batch
            dist, rbf = self.get_geom_embedding(edge_index, pos)
            rbf_inter = None
        elif self.outside_convolution_type == 'last_layer':
            edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
            dist, rbf = self.get_geom_embedding(edge_index, pos)
            dist_inter, rbf_inter = self.get_geom_embedding(torch.cat((edge_index, edge_index_inter), dim=1), pos)
        else:
            assert False, "Must select a valid treatment of inside vs outside nodes"
        return edge_index, edge_index_inter, inside_batch, inside_inds, n_repeats, rbf, rbf_inter


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

from mxtaltools.models.basis_functions import GaussianEmbedding, BesselBasisLayer
import torch
import torch.nn as nn

from mxtaltools.models.gnn_blocks import EmbeddingBlock, GCBlock, FCBlock, OutputBlock
from mxtaltools.models.utils import get_model_nans


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 node_dim: int,
                 fcs_per_gc: int,
                 message_dim: int,
                 embedding_dim: int,
                 num_convs: int,
                 num_radial: int,
                 num_input_classes=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 atom_type_embedding_dim=5,
                 norm=None,
                 dropout=0,
                 radial_embedding='bessel',
                 num_attention_heads=1,

                 periodize_inside_nodes=False,
                 outside_convolution_type='none',
                 equivariant=False,
                 vector_norm=None,
                 ):
        super(GraphNeuralNetwork, self).__init__()

        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.periodize_inside_nodes = periodize_inside_nodes
        self.outside_convolution_type = outside_convolution_type
        self.equivariant = equivariant

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)

        self.init_node_embedding = EmbeddingBlock(node_dim,
                                                  num_input_classes,
                                                  input_node_dim,
                                                  atom_type_embedding_dim)

        if self.equivariant:
            self.init_vector_embedding = nn.Linear(1, node_dim, bias=False)

        self.zeroth_fc_block = FCBlock(
            fcs_per_gc,
            node_dim,
            activation,
            norm,
            dropout,
            equivariant=self.equivariant,
            vector_norm=vector_norm)

        self.interaction_blocks = torch.nn.ModuleList([
            GCBlock(message_dim,
                    node_dim,
                    num_radial,
                    heads=num_attention_heads,
                    equivariant=equivariant,
                    )
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            FCBlock(
                fcs_per_gc,
                node_dim,
                activation,
                norm,
                dropout,
                equivariant=self.equivariant,
                vector_norm=vector_norm,
            )
            for _ in range(num_convs)
        ])

        self.output_block = OutputBlock(node_dim, embedding_dim, equivariant)

    def radial_embedding(self, edge_index, pos):
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self, z, pos, batch, ptr, edges_dict: dict):

        x, v = self.init_node_embeddings(z)

        if self.equivariant:
            x, v = self.zeroth_fc_block(x=x, v=v, batch=batch)
        else:
            x = self.zeroth_fc_block(x=x, v=v, batch=batch)

        (edge_index, edge_index_inter,
         inside_batch, inside_inds,
         n_repeats,
         rbf, rbf_inter) = self.get_edges(edges_dict, pos)

        for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
            if v is not None:
                x_res, v_res = x.clone(), v.clone()
                x, v = convolution(x, v, rbf, edge_index)
                x = x + x_res
                v = v + v_res
            else:
                x = x + convolution(x, v, rbf, edge_index)

            if not self.periodize_inside_nodes:  # inside/outside periodic convolution
                if self.equivariant:
                    x_res, v_res = x.clone(), v.clone()
                    x, v = fc(x, v=v, batch=batch)
                    x = x + x_res
                    v = v + v_res
                else:
                    x = x + fc(x, v=v, batch=batch)

                #assert torch.sum(torch.isnan(x)) == 0, f"NaN in fc_block output {get_model_nans(self.fc_blocks)}"

            else:
                assert v is None, "Vector embeddings not set up for periodic molecular crystal graph convolutions"
                # update only the inside inds
                x[inside_inds] = (x[inside_inds] + fc(x[inside_inds], batch=batch[inside_inds]))

                # then broadcast to all symmetry images
                x = self.periodize_molecular_crystal(inside_batch, inside_inds, n, n_repeats, ptr, x)

        return self.output_block(x, v)

    def init_node_embeddings(self, z):
        if self.equivariant:
            x, v = z[:, :-3], z[:, -3:]  # vector features are trailing 3 dimensions of node input
            v = self.init_vector_embedding(v[:, :, None])  # [n_nodes, 3] -> [n_nodes, 3, n_dim]
            x = self.init_node_embedding(x)  # embed atomic numbers & compute initial atom-wise feature vector
        else:
            x, v = self.init_node_embedding(z), None  # embed atomic numbers & compute initial atom-wise feature vector
        return x, v

    def periodize_molecular_crystal(self, inside_batch, inside_inds, n, n_repeats, ptr, x):
        for ii in range(len(ptr) - 1):  # enforce periodicity for each crystal, assuming invariant node features
            # copy the first asymmtric unit to all periodic images (safe since all are SE(3) invariant)
            x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)

        if n == len(self.interaction_blocks) - 1:
            x = x[inside_inds]  # reduce to inside image on the final convolution

        return x

    def get_edges(self, edges_dict, pos):
        if self.outside_convolution_type == 'none':
            # no inside/outside distinctions
            edge_index = edges_dict['edge_index']
            if 'dists' in edges_dict.keys():  # previously generated distances - e.g., for periodic MIC
                dist = edges_dict['dists']
                rbf = self.rbf(dist)
            else:
                dist, rbf = self.radial_embedding(edge_index, pos)
            edge_index_inter, inside_batch, inside_inds, n_repeats, rbf_inter = None, None, None, None, None
            assert not self.periodize_inside_nodes, "Cannot periodize to outside nodes if there are no outside nodes"

        elif self.outside_convolution_type == 'all_layers':
            # assumes input with inside-outside structure, and enforces periodicity after each convolution

            edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
            edge_index = torch.cat((edge_index, edge_index_inter), dim=1)  # all edges counted in one big batch
            dist, rbf = self.radial_embedding(edge_index, pos)
            rbf_inter = None

        elif self.outside_convolution_type == 'last_layer':
            assert False, "Last layer outside convolution is deprecated"
            # edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
            # dist, rbf = self.radial_embedding(edge_index, pos)
            # dist_inter, rbf_inter = self.radial_embedding(torch.cat((edge_index, edge_index_inter), dim=1), pos)

            # re-integrate this in forward method to bring it back
            # if n == (len(self.interaction_blocks) - 1) and self.outside_convolution_type == 'last_layer':
            #     # return only the results of the intermolecular convolution, omitting intramolecular features
            #     x = convolution(x, v,
            #                     rbf_inter,
            #                     torch.cat((edge_index, edge_index_inter), dim=1),
            #                     batch)

        else:
            assert False, "Must select a valid treatment of inside vs outside nodes"

        return edge_index, edge_index_inter, inside_batch, inside_inds, n_repeats, rbf, rbf_inter

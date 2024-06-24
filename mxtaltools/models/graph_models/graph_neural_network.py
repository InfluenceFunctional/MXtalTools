from typing import Optional, Tuple

from mxtaltools.models.modules.basis_functions import GaussianEmbedding, BesselBasisLayer
import torch
import torch.nn as nn

from mxtaltools.models.modules.components import scalarMLP, vectorMLP
from mxtaltools.models.graph_models.gnn_blocks import GCBlock, FCBlock, OutputBlock
from mxtaltools.models.modules.graph_convolution import MConv, v_MConv
from mxtaltools.models.modules.node_embedding_layer import EmbeddingBlock


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
                 norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 num_attention_heads: int = 1,

                 periodize_inside_nodes: bool = False,
                 outside_convolution_type: str = 'none',
                 add_vector_track: bool = False,
                 vector_norm: Optional[str] = None,
                 override_cutoff: Optional[float] = None
                 ):
        super(GraphNeuralNetwork, self).__init__()

        self.max_num_neighbors = max_num_neighbors
        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.periodize_inside_nodes = periodize_inside_nodes
        self.outside_convolution_type = outside_convolution_type
        self.add_vector_track = add_vector_track
        self.vector_addition_rescaling_factor = 1.6

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        self.init_node_embedding = EmbeddingBlock(node_dim,
                                                  num_input_classes,
                                                  input_node_dim,
                                                  atom_type_embedding_dim)

        if self.add_vector_track:
            self.init_vector_embedding = nn.Linear(1, node_dim, bias=False)

        self.zeroth_fc_block = FCBlock(
            fcs_per_gc,
            node_dim,
            activation,
            norm,
            dropout,
            equivariant=self.add_vector_track,
            vector_norm=vector_norm)

        self.interaction_blocks = torch.nn.ModuleList([
            GCBlock(message_dim,
                    node_dim,
                    num_radial,
                    heads=num_attention_heads,
                    add_vector_channel=add_vector_track,
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
                equivariant=self.add_vector_track,
                vector_norm=vector_norm,
            )
            for _ in range(num_convs)
        ])

        self.output_block = OutputBlock(node_dim, embedding_dim, add_vector_track)

    def radial_embedding(self, edge_index, pos):
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self, z, pos, batch, ptr, edges_dict: dict):

        x, v = self.init_node_embeddings(z)

        if self.add_vector_track:
            x, v = self.zeroth_fc_block(x=x, v=v, batch=batch)
        else:
            x = self.zeroth_fc_block(x=x, v=v, batch=batch)

        if len(self.interaction_blocks) > 0:
            (edge_index, edge_index_inter,
             inside_batch, inside_inds,
             n_repeats,
             rbf, rbf_inter) = self.get_edges(edges_dict, pos)

            for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
                if v is not None:
                    x_res, v_res = x.clone(), v.clone()
                    x, v = convolution(x, v, rbf, edge_index)
                    x = x + x_res
                    v = (v + v_res) / self.vector_addition_rescaling_factor
                else:
                    x = x + convolution(x, v, rbf, edge_index)

                if not self.periodize_inside_nodes:  # inside/outside periodic convolution
                    if self.add_vector_track:
                        x_res, v_res = x.clone(), v.clone()
                        x, v = fc(x, v=v, batch=batch)
                        x = x + x_res
                        v = (v + v_res) / self.vector_addition_rescaling_factor
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
        if self.add_vector_track:
            x, v = z[:, :-3], z[:, -3:]  # vector features are trailing 3 dimensions of node input
            v = self.init_vector_embedding(v[:, :, None])  # [n_nodes, 3] -> [n_nodes, 3, n_dim]
            x = self.init_node_embedding(x)  # embed atomic numbers & compute initial atom-wise feature vector
        else:
            x, v = self.init_node_embedding(z), None  # embed atomic numbers & compute initial atom-wise feature vector
        return x, v

    def periodize_molecular_crystal(self, inside_batch, inside_inds, n, n_repeats, ptr, x):
        for ii in range(len(ptr) - 1):  # enforce periodicity for each crystal, assuming invariant node features
            # copy the first asymmetric unit to all periodic images (safe since all are SE(3) invariant)
            x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)

        if n == len(self.interaction_blocks) - 1:
            x = x[inside_inds]  # reduce to inside image on the final convolution

        return x

    def get_edges(self,
                  edges_dict: dict,
                  pos: torch.Tensor):
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


class ScalarGNN(torch.nn.Module):
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
                 atom_type_embedding_dim: int = 5,
                 norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 override_cutoff: Optional[float] = None
                 ):
        super(ScalarGNN, self).__init__()

        self.max_num_neighbors = max_num_neighbors

        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        self.init_node_embedding = EmbeddingBlock(node_dim,
                                                  num_input_classes,
                                                  input_node_dim,
                                                  atom_type_embedding_dim)

        self.zeroth_fc_block = scalarMLP(layers=fcs_per_gc,
                                         filters=node_dim,
                                         input_dim=node_dim,
                                         output_dim=node_dim,
                                         conditioning_dim=0,
                                         activation=activation,
                                         norm=norm,
                                         dropout=dropout)

        self.interaction_blocks = torch.nn.ModuleList([
            MConv(
                message_dim=message_dim,
                node_dim=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
                activation_fn=activation)
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            scalarMLP(layers=fcs_per_gc,
                      filters=node_dim,
                      input_dim=node_dim,
                      output_dim=node_dim,
                      conditioning_dim=0,
                      activation=activation,
                      norm=norm,
                      dropout=dropout)
            for _ in range(num_convs)
        ])

        if node_dim != embedding_dim:
            self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def radial_embedding(self,
                         edge_index: torch.LongTensor,
                         pos: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self,
                z: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.LongTensor,
                edges_dict: dict
                ) -> torch.Tensor:

        x = self.init_node_embedding(z)
        x = self.zeroth_fc_block(x=x, batch=batch)

        if len(self.interaction_blocks) > 0:
            dist, rbf = self.radial_embedding(edges_dict['edge_index'], pos)
            for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
                x = convolution(x, edges_dict['edge_index'], rbf)
                x = fc(x, batch=batch)

        return self.output_layer(x)


class VectorGNN(torch.nn.Module):
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
                 atom_type_embedding_dim: int = 5,
                 norm: Optional[str] = None,
                 vector_norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 override_cutoff: Optional[float] = None
                 ):
        super(VectorGNN, self).__init__()

        self.max_num_neighbors = max_num_neighbors

        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        self.init_node_embedding = EmbeddingBlock(node_dim,
                                                  num_input_classes,
                                                  input_node_dim,
                                                  atom_type_embedding_dim)
        self.init_vector_embedding = self.init_vector_embedding = nn.Linear(1, node_dim, bias=False)

        self.zeroth_fc_block = vectorMLP(layers=fcs_per_gc,
                                         filters=node_dim,
                                         input_dim=node_dim,
                                         output_dim=node_dim,
                                         conditioning_dim=0,
                                         activation=activation,
                                         norm=norm,
                                         dropout=dropout,
                                         vector_input_dim=node_dim,
                                         vector_output_dim=node_dim,
                                         vector_norm=vector_norm)

        self.interaction_blocks = torch.nn.ModuleList([
            MConv(
                message_dim=message_dim,
                node_dim=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
                activation_fn=activation)
            for _ in range(num_convs)
        ])
        self.vector_interaction_blocks = torch.nn.ModuleList([
            v_MConv(
                message_depth=message_dim,
                node_depth=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
            )
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            vectorMLP(layers=fcs_per_gc,
                      filters=node_dim,
                      input_dim=node_dim,
                      output_dim=node_dim,
                      conditioning_dim=0,
                      activation=activation,
                      norm=norm,
                      dropout=dropout,
                      vector_norm=vector_norm,
                      vector_input_dim=node_dim,
                      vector_output_dim=node_dim)
            for _ in range(num_convs)
        ])

        if node_dim != embedding_dim:
            self.output_layer, self.v_output_layer = nn.Linear(node_dim, embedding_dim, bias=False), nn.Linear(node_dim,
                                                                                                               embedding_dim,
                                                                                                               bias=False)
        else:
            self.output_layer, self.v_output_layer = nn.Identity(), nn.Identity()

    def radial_embedding(self,
                         edge_index: torch.LongTensor,
                         pos: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.LongTensor,
                edges_dict: dict
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.init_node_embedding(x)
        v = self.init_vector_embedding(v)
        x, v = self.zeroth_fc_block(x=x, v=v, batch=batch)

        if len(self.interaction_blocks) > 0:
            dist, rbf = self.radial_embedding(edges_dict['edge_index'], pos)
            for n, (convolution, vector_convolution, fc) in enumerate(
                    zip(self.interaction_blocks, self.vector_interaction_blocks, self.fc_blocks)):
                x = convolution(x, edges_dict['edge_index'], rbf)
                v = vector_convolution(v, edges_dict['edge_index'], rbf)
                x, v = fc(x=x, v=v, batch=batch)

        return self.output_layer(x), self.v_output_layer(v)


class MolCrystalScalarGNN(torch.nn.Module):
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
                 atom_type_embedding_dim: int = 5,
                 norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 override_cutoff: Optional[float] = None
                 ):
        super(MolCrystalScalarGNN, self).__init__()

        self.max_num_neighbors = max_num_neighbors

        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        self.init_node_embedding = EmbeddingBlock(node_dim,
                                                  num_input_classes,
                                                  input_node_dim,
                                                  atom_type_embedding_dim)

        self.zeroth_fc_block = scalarMLP(layers=fcs_per_gc,
                                         filters=node_dim,
                                         input_dim=node_dim,
                                         output_dim=node_dim,
                                         conditioning_dim=0,
                                         activation=activation,
                                         norm=norm,
                                         dropout=dropout)

        self.interaction_blocks = torch.nn.ModuleList([
            MConv(
                message_dim=message_dim,
                node_dim=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
                activation_fn=activation)
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            scalarMLP(layers=fcs_per_gc,
                      filters=node_dim,
                      input_dim=node_dim,
                      output_dim=node_dim,
                      conditioning_dim=0,
                      activation=activation,
                      norm=norm,
                      dropout=dropout)
            for _ in range(num_convs)
        ])

        if node_dim != embedding_dim:
            self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def radial_embedding(self,
                         edge_index: torch.LongTensor,
                         pos: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute elements for radial & spherical embeddings
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self,
                z: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.LongTensor,
                aux_ind: torch.Tensor,
                ptr: torch.LongTensor,
                edges_dict: dict
                ) -> torch.Tensor:

        x = self.init_node_embedding(z)
        x = self.zeroth_fc_block(x=x, batch=batch)

        # assumes input with inside-outside structure, and enforces periodicity after each convolution
        edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
        edge_index = torch.cat((edge_index, edge_index_inter), dim=1)  # all edges counted in one big batch

        if len(self.interaction_blocks) > 0:
            dist, rbf = self.radial_embedding(edge_index, pos)

            for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
                x = convolution(x, edge_index, rbf)

                # manually periodize inside nodes to outside nodes after each convolution
                x[inside_inds] = fc(x[inside_inds], batch=batch[inside_inds])

                # then broadcast node features to all symmetry images
                x = self.periodize_molecular_crystal(inside_batch, inside_inds, n, n_repeats, ptr, x, aux_ind)

        return self.output_layer(x)

    def periodize_molecular_crystal(self, inside_batch, inside_inds, n, n_repeats, ptr, x, aux_ind):
        for ii in range(len(ptr) - 1):  # enforce periodicity for each crystal, assuming invariant node features
            # copy the first asymmetric unit to all periodic images (safe since all are SE(3) invariant)
            # todo check if this could be done faster with repeat_interleave
            x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)

        x[:, -1] = aux_ind  # manually re-indicate inside/outside structure

        if n == len(self.interaction_blocks) - 1:
            x = x[inside_inds]  # reduce to inside image on the final convolution

        return x

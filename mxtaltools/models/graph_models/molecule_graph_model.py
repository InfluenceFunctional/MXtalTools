from argparse import Namespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from mxtaltools.models.graph_models.graph_neural_network import ScalarGNN, VectorGNN, \
    MolCrystalScalarGNN
from mxtaltools.models.modules.augmented_softmax_aggregator import AugSoftmaxAggregation, VectorAugSoftmaxAggregation
from mxtaltools.models.modules.components import scalarMLP, vectorMLP
from mxtaltools.models.functions.radial_graph import build_radial_graph
from mxtaltools.models.functions.minimum_image_neighbors import argwhere_minimum_image_convention_edges


# noinspection PyAttributeOutsideInit
class ScalarMoleculeGraphModel(nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_pos_to_node_dim: bool = False,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(ScalarMoleculeGraphModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_pos_to_node_dim = concat_pos_to_node_dim
        self.concat_mol_to_node_dim = concat_mol_to_node_dim

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats

        self.graph_net = ScalarGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = scalarMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                batch: torch.LongTensor,
                ptr: torch.LongTensor,
                mol_x: Union[torch.Tensor],
                num_graphs: int,
                edge_index: Optional[torch.LongTensor] = None,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False,
                ) -> Tuple[torch.Tensor, Optional[dict]]:

        x = self.append_init_node_features(x, pos, ptr, mol_x)
        x = self.graph_net(x,
                           pos,
                           batch,
                           edge_index)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation
        x = self.global_pool(x,
                             batch,
                             dim_size=num_graphs)

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                x = torch.cat([x, self.mol_fc(mol_x)], dim=-1)
            gmlp_out = self.gnn_mlp(x)

            x = gmlp_out

        output = self.output_fc(x)

        extra_outputs = self.collect_extra_outputs(x,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self, x, pos, ptr, mol_x):
        if x.ndim == 1:
            x = x[:, None]

        # simply append node coordinates, PointNet style
        if self.concat_pos_to_node_dim:
            x = torch.cat((x, pos), dim=-1)

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x


# noinspection PyAttributeOutsideInit
class VectorMoleculeGraphModel(nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_pos_to_node_dim: bool = False,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(VectorMoleculeGraphModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_pos_to_node_dim = concat_pos_to_node_dim
        self.concat_mol_to_node_dim = concat_mol_to_node_dim

        if self.concat_pos_to_node_dim:
            input_node_dim += 1  # radial dimension - vector features explicitly added later

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats

        self.graph_net = VectorGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        self.v_global_pool = VectorAugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = vectorMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     vector_input_dim=fc_config.hidden_dim,
                                     v_to_s_combination='sum',
                                     vector_norm=fc_config.vector_norm,
                                     vector_output_dim=fc_config.hidden_dim,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
            self.v_output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()
            self.v_output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                batch: torch.LongTensor,
                ptr: torch.LongTensor,
                num_graphs: int,
                mol_x: Optional[torch.Tensor] = None,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:

        if len(self.graph_net.interaction_blocks) > 0 or return_dists:
            # todo clean up options around prebuilt radial graphs
            if edges_dict is None:  # option to rebuild radial graph
                edges_dict = build_radial_graph(
                    pos,
                    batch,
                    ptr,
                    self.convolution_cutoff,
                    self.max_num_neighbors,
                )
            else:
                edges_dict = None

        x, v = self.append_init_node_features(x, pos, ptr, mol_x)
        x, v = self.graph_net(x,
                              v,
                              pos,
                              batch,
                              edges_dict)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation
        x = self.global_pool(x,
                             batch,
                             dim_size=num_graphs)
        v = self.v_global_pool(v,
                               batch,
                               dim_size=num_graphs,
                               dim=0,
                               cart_dim=1)

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                x = torch.cat([x, self.mol_fc(mol_x)], dim=-1)
            x, v = self.gnn_mlp(x, v)

        x_out, v_out = self.output_fc(x), self.v_output_fc(v)

        extra_outputs = self.collect_extra_outputs(x,
                                                   pos,
                                                   batch,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return x_out, v_out, extra_outputs
        else:
            return x_out, v_out

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              pos: torch.Tensor,
                              batch: torch.LongTensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self,
                                  x: torch.Tensor,
                                  pos: torch.Tensor,
                                  ptr: torch.LongTensor,
                                  mol_x: Optional[torch.Tensor] = None,
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 1:
            x = x[:, None]

        # append radial position as scalar feature
        # and 3 vector dimensions (unit vectors from centroid)
        rad = torch.linalg.norm(pos, dim=1)
        if self.concat_pos_to_node_dim:
            x = torch.cat((x, rad[:, None]), dim=-1)  # radii
        # v = pos / (rad[:, None] + 1e-5)  # normed directions
        v = pos[..., None]  # set dimension as [n,3,k]
        # richer embedding as 3 component vectors rather than one single vector
        #v = pos[:, :, None] * torch.eye(3, device=pos.device, dtype=torch.float32).repeat(len(pos), 1, 1)

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x, v


# noinspection PyAttributeOutsideInit
class MolecularCrystalGraphModel(nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_mol_ind_to_node_dim: bool = False,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(MolecularCrystalGraphModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_mol_to_node_dim = concat_mol_to_node_dim
        self.concat_mol_ind_to_node_dim = concat_mol_ind_to_node_dim

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats
        if concat_mol_ind_to_node_dim:
            input_node_dim += 2  # aux_ind and mol_ind will be appended

        self.graph_net = MolCrystalScalarGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = scalarMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                batch: torch.LongTensor,
                ptr: torch.LongTensor,
                mol_x: Union[torch.Tensor],
                num_graphs: int,
                aux_ind: torch.Tensor,
                mol_ind: torch.Tensor,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False
                ) -> Tuple[torch.Tensor, Optional[dict]]:

        if len(self.graph_net.interaction_blocks) > 0 or return_dists:
            if edges_dict is None:  # option to rebuild radial graph
                edges_dict = build_radial_graph(
                    pos,
                    batch,
                    ptr,
                    self.convolution_cutoff,
                    self.max_num_neighbors,
                    aux_ind=aux_ind,
                    mol_ind=mol_ind,
                )
            else:
                edges_dict = None

        x = self.append_init_node_features(x, ptr, mol_x, aux_ind, mol_ind)
        g = self.graph_net(x,
                           pos,
                           batch,
                           aux_ind,
                           ptr,
                           edges_dict)  # get graph encoding

        if return_embedding:
            embedding = g.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation, only over 'inside' nodes
        g = self.global_pool(g,
                             edges_dict['inside_batch'],
                             dim_size=num_graphs)

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                g = torch.cat([g, self.mol_fc(mol_x)], dim=-1)
            g = self.gnn_mlp(g)

        output = self.output_fc(g)

        extra_outputs = self.collect_extra_outputs(x,
                                                   pos,
                                                   batch,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              pos: torch.Tensor,
                              batch: torch.LongTensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict
            if 'edge_index_inter' in edges_dict.keys():
                extra_outputs['dists_dict']['intermolecular_dist'] = (
                    (pos[edges_dict['edge_index_inter'][0]] - pos[edges_dict['edge_index_inter'][1]]).pow(2).sum(
                        dim=-1).sqrt()
                )

                extra_outputs['dists_dict']['intermolecular_dist_batch'] = batch[edges_dict['edge_index_inter'][0]]

                extra_outputs['dists_dict']['intermolecular_dist_atoms'] = \
                    [x[edges_dict['edge_index_inter'][0], 0].long(),
                     x[edges_dict['edge_index_inter'][1], 0].long()]

                extra_outputs['dists_dict']['intermolecular_dist_inds'] = edges_dict['edge_index_inter']

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self, x, ptr, mol_x, aux_ind, mol_ind):
        if x.ndim == 1:
            x = x[:, None]

        if self.concat_mol_ind_to_node_dim:
            x = torch.cat((x, mol_ind[:, None], aux_ind[:, None]), dim=-1)

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x


# noinspection PyAttributeOutsideInit
class MoleculeClusterModel(nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(MoleculeClusterModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_mol_to_node_dim = concat_mol_to_node_dim

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats

        self.graph_net = ScalarGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = scalarMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                ptr: torch.LongTensor,
                mol_x: Union[torch.Tensor],
                num_graphs: int,
                mol_ind: torch.Tensor,
                T_fc: torch.Tensor,
                edge_index: Optional[torch.LongTensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False
                ) -> Tuple[torch.Tensor, Optional[dict]]:

        assert ptr[-1] == len(x), "Only one cluster sample per batch is allowed for this model"

        if len(self.graph_net.interaction_blocks) > 0 or return_dists:
            if edge_index is not None:
                pass
            elif edges_dict is None:  # option to rebuild radial graph
                edges_dict = argwhere_minimum_image_convention_edges(
                    num_graphs, pos, T_fc, self.convolution_cutoff)
                edge_index = edges_dict['edge_index']
                edge_attr = edges_dict['dists']
            else:
                edge_index = edges_dict['edge_index']
                edge_attr = edges_dict['dists']

        x = self.append_init_node_features(x, ptr, mol_x)
        x = self.graph_net(x,
                           pos,
                           mol_ind,
                           edge_index,
                           edge_attr)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation
        x = self.global_pool(x,
                             index=mol_ind - mol_ind.min(),
                             dim_size=int(mol_ind.max()))

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                x = torch.cat([x, self.mol_fc(mol_x)], dim=-1)
            x = self.gnn_mlp(x)

        extra_outputs = self.collect_extra_outputs(x,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return self.output_fc(x), extra_outputs
        else:
            return self.output_fc(x)

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self, x, ptr, mol_x):
        if x.ndim == 1:
            x = x[:, None]

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x

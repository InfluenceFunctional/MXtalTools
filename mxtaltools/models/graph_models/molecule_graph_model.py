from argparse import Namespace
from typing import Optional, Tuple, Union

from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.modules.augmented_softmax_aggregator import AugSoftmaxAggregation, VectorAugSoftmaxAggregation
from mxtaltools.models.graph_models.graph_neural_network import GraphNeuralNetwork, ScalarGNN, VectorGNN, \
    MolCrystalScalarGNN

import torch
import torch.nn as nn

from mxtaltools.models.modules.components import construct_radial_graph, GlobalAggregation, EMLP, scalarMLP, vectorMLP

from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.models.utils import argwhere_minimum_image_convention_edges


# noinspection PyAttributeOutsideInit
class MoleculeGraphModel(nn.Module):
    def __init__(self,
                 input_node_dim,
                 output_dim,
                 graph_aggregator,
                 fc_config,
                 graph_config,
                 activation='gelu',
                 num_mol_feats=0,
                 concat_pos_to_node_dim=False,
                 concat_mol_to_node_dim=False,
                 concat_crystal_to_node_dim=False,
                 concat_aux_ind_to_node_dim=False,
                 concat_mol_ind_to_node_dim=False,
                 seed=5,
                 periodize_inside_nodes=False,
                 outside_convolution_type='none',
                 vector_norm=None,
                 equivariant=False,
                 override_cutoff=None
                 ):

        super(MoleculeGraphModel, self).__init__()

        torch.manual_seed(seed)
        self.equivariant = equivariant
        self.periodic_structure = periodize_inside_nodes

        self.concat_pos_to_node_dim = concat_pos_to_node_dim
        self.concat_mol_to_node_dim = concat_mol_to_node_dim
        self.concat_crystal_to_node_dim = concat_crystal_to_node_dim
        self.concat_aux_ind_to_node_dim = concat_aux_ind_to_node_dim
        self.concat_mol_ind_to_node_dim = concat_mol_ind_to_node_dim

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        self.register_buffer('SG_FEATURE_TENSOR', SG_FEATURE_TENSOR.clone())  # store space group information

        input_node_dim = self.adjust_input_dim(concat_aux_ind_to_node_dim, concat_crystal_to_node_dim,
                                               concat_mol_ind_to_node_dim, concat_mol_to_node_dim,
                                               concat_pos_to_node_dim, input_node_dim, num_mol_feats)

        self.graph_net = GraphNeuralNetwork(
            activation=activation,
            input_node_dim=input_node_dim,
            periodize_inside_nodes=periodize_inside_nodes,
            outside_convolution_type=outside_convolution_type,
            add_vector_track=equivariant,
            vector_norm=vector_norm,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = GlobalAggregation(graph_aggregator, graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = EMLP(layers=fc_config.num_layers,
                                filters=fc_config.hidden_dim,
                                norm=fc_config.norm,
                                dropout=fc_config.dropout,
                                input_dim=graph_config.embedding_dim,
                                output_dim=fc_config.hidden_dim,
                                conditioning_dim=num_mol_feats,
                                seed=seed,
                                add_vector_channel=self.equivariant,
                                vector_output_dim=graph_config.embedding_dim,
                                vector_norm='vector layer' if vector_norm is not None else None,
                                )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        self.init_output_layer(graph_output_dim, output_dim)

    def adjust_input_dim(self, concat_aux_ind_to_node_dim, concat_crystal_to_node_dim,
                         concat_mol_ind_to_atom_featuers, concat_mol_to_node_dim, concat_pos_to_node_dim,
                         input_node_dim, num_mol_feats):
        if concat_pos_to_node_dim:
            if self.equivariant:
                input_node_dim += 1  # radial dimension - vector features explicitly added later
            else:
                input_node_dim += 3  # cartesian dimension always 3
        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats
        if concat_crystal_to_node_dim:
            input_node_dim += SG_FEATURE_TENSOR.shape[1]
        if concat_aux_ind_to_node_dim:
            input_node_dim += 1
        if concat_mol_ind_to_atom_featuers:
            input_node_dim += 1
        return input_node_dim

    def init_output_layer(self, graph_embedding_dim, output_dim):
        """initialize output reshaping layers"""
        if graph_embedding_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_embedding_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()
        if self.equivariant:
            if graph_embedding_dim != output_dim:  # only want this if we have to change the dimension
                self.v_output_fc = nn.Linear(graph_embedding_dim, output_dim, bias=False)
            else:
                self.v_output_fc = nn.Identity()

    def forward(self,
                data: CrystalData,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False):

        if len(self.graph_net.interaction_blocks) > 0 or return_dists:
            if edges_dict is None:  # option to rebuild radial graph
                if hasattr(data, 'periodic'):
                    if all(data.periodic):  # todo only currently works for batches containing a single graph
                        assert data.num_graphs == 1, "MIC Periodic graphs not supported for more than one graph per data object"
                        edges_dict = argwhere_minimum_image_convention_edges(
                            data.num_graphs, data.pos, data.T_fc, self.convolution_cutoff)
                else:
                    edges_dict = construct_radial_graph(data.pos,
                                                        data.batch,
                                                        data.ptr,
                                                        self.convolution_cutoff,
                                                        self.max_num_neighbors,
                                                        aux_ind=data.aux_ind,
                                                        )
            else:
                edges_dict = None

        if self.graph_net.outside_convolution_type != 'none':
            agg_batch = edges_dict['inside_batch']
        else:
            agg_batch = data.batch

        x = data.x  # already cloned before it comes into this function
        x = self.append_init_node_features(data, x)
        x = self.graph_net(x,
                           data.pos,
                           data.batch,
                           data.ptr,
                           edges_dict)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        if self.equivariant:
            x, v = x
            x, v = self.global_pool(x,
                                    agg_batch,
                                    v=v,
                                    cluster=data.mol_ind if self.global_pool.agg_func == 'molwise' else None,
                                    output_dim=data.num_graphs)  # aggregate atoms to molecule / graph representation
        else:
            v = None
            x = self.global_pool(x,
                                 agg_batch,
                                 v=v,
                                 cluster=data.mol_ind if self.global_pool.agg_func == 'molwise' else None,
                                 output_dim=data.num_graphs)  # aggregate atoms to molecule / graph representation

        if self.num_fc_layers > 0:
            gmlp_out = self.gnn_mlp(x,
                                    v=v,
                                    conditions=self.mol_fc(data.mol_x) if self.mol_fc is not None else None
                                    # add molecule-wise features
                                    )
            if self.equivariant:
                x, v = gmlp_out
            else:
                x = gmlp_out

        output = (self.output_fc(x), self.v_output_fc(v)) if self.equivariant else self.output_fc(x)

        extra_outputs = self.collect_extra_outputs(data, edges_dict,
                                                   return_dists, return_latent, return_embedding,
                                                   x, embedding)

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output

    @staticmethod
    def collect_extra_outputs(data, edges_dict, return_dists, return_latent, return_embedding, x, embedding):
        extra_outputs = {}
        if return_dists:
            extra_outputs['dists_dict'] = edges_dict
            if 'edge_index_inter' in edges_dict.keys():
                extra_outputs['dists_dict']['intermolecular_dist'] = ((
                                                                              data.pos[
                                                                                  edges_dict['edge_index_inter'][0]] -
                                                                              data.pos[
                                                                                  edges_dict['edge_index_inter'][1]])
                                                                      .pow(2).sum(dim=-1).sqrt())
                extra_outputs['dists_dict']['intermolecular_dist_batch'] = data.batch[edges_dict['edge_index_inter'][0]]
                extra_outputs['dists_dict']['intermolecular_dist_atoms'] = [
                    data.x[edges_dict['edge_index_inter'][0], 0].long(),
                    data.x[edges_dict['edge_index_inter'][1], 0].long()]
                extra_outputs['dists_dict']['intermolecular_dist_inds'] = edges_dict['edge_index_inter']
        if return_latent:
            extra_outputs['final_activation'] = x.detach()
        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()
        return extra_outputs

    def append_init_node_features(self, data, x):
        if x.ndim == 1:
            x = x[:, None]
        if self.concat_pos_to_node_dim:
            if self.equivariant:
                # append radial position as scalar feature
                # and 3 vector dimensions (unit vectors from centroid)
                rad = torch.linalg.norm(data.pos, dim=1)
                x = torch.cat((x, rad[:, None], data.pos / (rad[:, None] + 1e-5)),
                              dim=-1)  # radii and normed directions
            else:
                x = torch.cat((x, data.pos), dim=-1)  # simply append node coordinates, PointNet style

        if self.concat_mol_to_node_dim:  # add molwise information
            nodes_per_graph = torch.diff(data.ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(data.mol_x, nodes_per_graph, 0)),
                          dim=-1)

        if self.concat_crystal_to_node_dim:  # tell the sample what space group it's in
            nodes_per_graph = torch.diff(data.ptr)
            crystal_features = torch.tensor(self.SG_FEATURE_TENSOR[data.sg_ind], dtype=torch.float32,
                                            device=data.x.device)
            x = torch.cat((x,
                           torch.repeat_interleave(crystal_features, nodes_per_graph, 0)),
                          dim=-1)

        if self.concat_aux_ind_to_node_dim:  # tell the node which symmetry image it's on
            x = torch.cat((x, data.aux_ind[:, None]), dim=-1)

        if self.concat_mol_ind_to_node_dim:  # tell the node which molecule it's on
            x = torch.cat((x, data.mol_ind[:, None]), dim=-1)

        return x


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
                                     input_dim=graph_config.embedding_dim,
                                     output_dim=fc_config.hidden_dim,
                                     conditioning_dim=num_mol_feats,
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
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False
                ) -> Tuple[torch.Tensor, Optional[dict]]:

        if len(self.graph_net.interaction_blocks) > 0 or return_dists:
            if edges_dict is None:  # option to rebuild radial graph
                edges_dict = construct_radial_graph(
                    pos,
                    batch,
                    ptr,
                    self.convolution_cutoff,
                    self.max_num_neighbors,
                )
            else:
                edges_dict = None

        x = self.append_init_node_features(x, pos, ptr, mol_x)
        x = self.graph_net(x,
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

        if self.num_fc_layers > 0:
            gmlp_out = self.gnn_mlp(x,
                                    conditions=self.mol_fc(mol_x) if self.mol_fc is not None else None
                                    )
            x = gmlp_out

        output = self.output_fc(x)

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
                                     input_dim=graph_config.embedding_dim,
                                     output_dim=fc_config.hidden_dim,
                                     conditioning_dim=num_mol_feats,
                                     seed=seed,
                                     vector_input_dim=fc_config.hidden_dim,
                                     v_to_s_combination='concatenate',
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
            if edges_dict is None:  # option to rebuild radial graph
                edges_dict = construct_radial_graph(
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
            x, v = self.gnn_mlp(x, v,
                                conditions=self.mol_fc(mol_x) if self.mol_fc is not None else None
                                )

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
        v = pos / (rad[:, None] + 1e-5)  # normed directions
        v = v[..., None]  # set dimension as [n,3,k]

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
                                     input_dim=graph_config.embedding_dim,
                                     output_dim=fc_config.hidden_dim,
                                     conditioning_dim=num_mol_feats,
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
                edges_dict = construct_radial_graph(
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
            g = self.gnn_mlp(g,
                                    conditions=self.mol_fc(mol_x) if self.mol_fc is not None else None
                                    )

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

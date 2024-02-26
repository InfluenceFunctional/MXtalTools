from models.GraphNeuralNetwork import GraphNeuralNetwork

import torch
import torch.nn as nn

from models.components import MLP, construct_radial_graph, GlobalAggregation

from constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from models.utils import get_model_nans


class MoleculeGraphModel(nn.Module):
    def __init__(self, num_atom_feats,
                 output_dimension,
                 graph_aggregator,
                 equivariant_graph=False,
                 concat_pos_to_atom_features=False,
                 concat_mol_to_atom_features=False,
                 concat_crystal_to_atom_features=False,
                 concat_cell_params_to_atom_features=False,
                 activation='gelu',
                 num_mol_feats=0,
                 num_fc_layers=4,
                 fc_depth=256,
                 fc_dropout_probability=0,
                 fc_norm_mode=None,
                 graph_node_norm=None,
                 graph_node_dropout=0,
                 graph_message_dropout=0,
                 num_radial=32,
                 num_attention_heads=4,
                 graph_message_depth=64,
                 graph_node_dims=128,
                 num_graph_convolutions=1,
                 graph_embedding_depth=256,
                 nodewise_fc_layers=1,
                 radial_function='gaussian',
                 max_num_neighbors=100,
                 convolution_cutoff=6,
                 atom_type_embedding_dims=5,
                 seed=5,
                 periodic_structure=False,
                 outside_convolution_type='none',
                 cartesian_dimension=3,
                 vector_norm=None,
                 ):

        super(MoleculeGraphModel, self).__init__()

        torch.manual_seed(seed)
        self.equivariant_graph = equivariant_graph
        self.periodic_structure = periodic_structure
        self.concat_pos_to_atom_features = concat_pos_to_atom_features
        self.concat_mol_to_atom_features = concat_mol_to_atom_features
        self.concat_crystal_to_atom_features = concat_crystal_to_atom_features
        self.concat_cell_params_to_atom_features = concat_cell_params_to_atom_features
        self.convolution_cutoff, self.max_num_neighbors = convolution_cutoff, max_num_neighbors
        self.num_fc_layers = num_fc_layers

        self.register_buffer('SG_FEATURE_TENSOR', SG_FEATURE_TENSOR.clone())

        input_node_depth = num_atom_feats
        if concat_pos_to_atom_features:
            if self.equivariant_graph:
                input_node_depth += 1  # radial dimension - spherical harmonics explicitly added later
            else:
                input_node_depth += cartesian_dimension

        if concat_mol_to_atom_features:
            input_node_depth += num_mol_feats
        if concat_crystal_to_atom_features:
            input_node_depth += SG_FEATURE_TENSOR.shape[1]
        if concat_cell_params_to_atom_features:
            input_node_depth += 12

        self.graph_net = GraphNeuralNetwork(
            input_node_depth=input_node_depth,
            node_embedding_depth=graph_node_dims,
            graph_embedding_depth=graph_embedding_depth,
            nodewise_fc_layers=nodewise_fc_layers,
            message_depth=graph_message_depth,
            num_blocks=num_graph_convolutions,
            num_radial=num_radial,
            max_num_neighbors=max_num_neighbors,
            activation=activation,
            embedding_hidden_dimension=atom_type_embedding_dims,
            cutoff=convolution_cutoff,
            message_dropout=graph_message_dropout,
            nodewise_norm=graph_node_norm,
            nodewise_dropout=graph_node_dropout,
            radial_embedding=radial_function,
            attention_heads=num_attention_heads,
            periodize_inside_nodes=periodic_structure,
            outside_convolution_type=outside_convolution_type,
            equivariant_graph=equivariant_graph,
            vector_norm=vector_norm,
        )

        # initialize global pooling operation
        self.global_pool = GlobalAggregation(graph_aggregator, graph_embedding_depth)

        # molecule features FC layer
        if num_mol_feats != 0:
            self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats)
        else:
            self.mol_fc = None

        """Optional built-in FC model to post-process graph fingerprint"""
        if num_fc_layers > 0:
            self.gnn_mlp = MLP(layers=num_fc_layers,
                               filters=fc_depth,
                               norm=fc_norm_mode,
                               dropout=fc_dropout_probability,
                               input_dim=graph_embedding_depth,
                               output_dim=fc_depth,
                               conditioning_dim=num_mol_feats,
                               seed=seed,
                               equivariant=self.equivariant_graph,
                               vector_output_dim=graph_embedding_depth,
                               vector_norm='vector layer' if vector_norm is not None else None,
                               )
            gnn_output_depth = fc_depth
        else:
            gnn_output_depth = graph_embedding_depth

        """initialize output reshaping layers"""
        if gnn_output_depth != output_dimension:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(gnn_output_depth, output_dimension, bias=False)
        else:
            self.output_fc = nn.Identity()

        if self.equivariant_graph:
            if gnn_output_depth != output_dimension:  # only want this if we have to change the dimension
                self.v_output_fc = nn.Linear(gnn_output_depth, output_dimension, bias=False)
            else:
                self.v_output_fc = nn.Identity()

    def forward(self, data, edges_dict=None, return_latent=False, return_dists=False):
        if edges_dict is None:  # option to rebuild radial graph
            edges_dict = construct_radial_graph(data.pos,
                                                data.batch,
                                                data.ptr,
                                                self.convolution_cutoff,
                                                self.max_num_neighbors,
                                                aux_ind=data.aux_ind)

        if self.graph_net.outside_convolution_type != 'none':
            agg_batch = edges_dict['inside_batch']
        else:
            agg_batch = data.batch

        x = data.x  # already cloned before it comes into this function
        x = self.append_init_node_features(data, x)
        x = self.graph_net(x, data.pos,
                           data.batch, data.ptr,
                           edges_dict)  # get graph encoding
        if self.equivariant_graph:
            x, v = x
            # assert torch.sum(torch.isnan(x)) == 0, f"NaN in gnn scalar output {get_model_nans(self.graph_net)}"

            x, v = self.global_pool(x,
                                    agg_batch,
                                    v=v,
                                    cluster=data.mol_ind if self.global_pool.agg_func == 'molwise' else None,
                                    output_dim=data.num_graphs)  # aggregate atoms to molecule / graph representation
        else:
            v = None
            # torch.sum(torch.isnan(v)) == 0, f"NaN in gnn vector output {get_model_nans(self.graph_net)}"

            x = self.global_pool(x,
                                 agg_batch,
                                 v=v,
                                 cluster=data.mol_ind if self.global_pool.agg_func == 'molwise' else None,
                                 output_dim=data.num_graphs)  # aggregate atoms to molecule / graph representation

        # assert torch.sum(torch.isnan(x)) == 0, f"NaN in gnn pool output {get_model_nans(self.graph_net)}"

        if v is not None:
            assert torch.sum(torch.isnan(v)) == 0, f"NaN in gnn pool vector output {get_model_nans(self.graph_net)}"

        if self.num_fc_layers > 0:
            gmlp_out = self.gnn_mlp(x, v=v,
                                    conditions=self.mol_fc(data.mol_x) if self.mol_fc is not None else None  # add graph-wise features
                                    )
            if self.equivariant_graph:
                x, v = gmlp_out
            else:
                x = gmlp_out

        output = (self.output_fc(x), self.v_output_fc(v)) if self.equivariant_graph else self.output_fc(x)

        # if self.equivariant_graph:
        # assert torch.sum(torch.isnan(output[0])) == 0, f"NaN in gnn pool output {get_model_nans(self.graph_net)}"
        # assert torch.sum(torch.isnan(output[1])) == 0, f"NaN in gnn pool vector output {get_model_nans(self.graph_net)}"
        # else:
        # assert torch.sum(torch.isnan(output)) == 0, f"NaN in gnn pool output {get_model_nans(self.graph_net)}"

        extra_outputs = self.collect_extra_outputs(data, edges_dict, return_dists, return_latent, x)

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output

    def collect_extra_outputs(self, data, edges_dict, return_dists, return_latent, x):
        extra_outputs = {}
        if return_dists:
            extra_outputs['dists_dict'] = edges_dict
            if 'edge_index_inter' in edges_dict.keys():
                extra_outputs['dists_dict']['intermolecular_dist'] = (data.pos[edges_dict['edge_index_inter'][0]] - data.pos[edges_dict['edge_index_inter'][1]]).pow(2).sum(dim=-1).sqrt()
                extra_outputs['dists_dict']['intermolecular_dist_batch'] = data.batch[edges_dict['edge_index_inter'][0]]
                extra_outputs['dists_dict']['intermolecular_dist_atoms'] = [data.x[edges_dict['edge_index_inter'][0], 0].long(), data.x[edges_dict['edge_index_inter'][1], 0].long()]
                extra_outputs['dists_dict']['intermolecular_dist_inds'] = edges_dict['edge_index_inter']
        if return_latent:
            extra_outputs['final_activation'] = x.cpu().detach().numpy()
        return extra_outputs

    def append_init_node_features(self, data, x):
        if self.concat_pos_to_atom_features:
            if self.equivariant_graph:
                # rad = torch.linalg.norm(data.pos, dim=1)
                # centroid = torch.zeros((1, 3), device=data.pos.device, dtype=data.pos.dtype)
                # sh = o3.spherical_harmonics(
                #     l=[ind for ind in range(0, self.sh_order + 1)],
                #     x=data.pos - centroid,
                #     normalize=True,
                #     normalization='component'
                # )
                # x = torch.cat((x, rad[:, None], sh), dim=-1)

                rad = torch.linalg.norm(data.pos, dim=1)
                x = torch.cat((x, rad[:, None], data.pos / rad[:, None]), dim=-1)  # radii and normed directions
            else:
                x = torch.cat((x, data.pos), dim=-1)

        if self.concat_mol_to_atom_features:
            nodes_per_graph = torch.diff(data.ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(data.mol_x, nodes_per_graph, 0)),
                          dim=-1)
        if self.concat_crystal_to_atom_features:
            nodes_per_graph = torch.diff(data.ptr)
            crystal_features = torch.tensor(self.SG_FEATURE_TENSOR[data.sg_ind], dtype=torch.float32, device=data.x.device)
            x = torch.cat((x,
                           torch.repeat_interleave(crystal_features, nodes_per_graph, 0)),
                          dim=-1)
        if self.concat_cell_params_to_atom_features:
            nodes_per_graph = torch.diff(data.ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(data.cell_params, nodes_per_graph, 0)),
                          dim=-1)
        return x

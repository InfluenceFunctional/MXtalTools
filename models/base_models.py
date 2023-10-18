from models.GraphNeuralNetwork import GraphNeuralNetwork

import torch
import torch.nn as nn

from models.global_aggregation import global_aggregation
from models.components import MLP, construct_radial_graph

from constants.space_group_feature_tensor import SG_FEATURE_TENSOR


class molecule_graph_model(nn.Module):
    def __init__(self, num_atom_feats,
                 output_dimension,
                 graph_convolution_type,
                 graph_aggregator,
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
                 graph_message_norm=None,
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
                 ):

        super(molecule_graph_model, self).__init__()

        torch.manual_seed(seed)
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
            input_node_depth += 3
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
            convolution_type=graph_convolution_type,
            num_blocks=num_graph_convolutions,
            num_radial=num_radial,
            max_num_neighbors=max_num_neighbors,
            activation=activation,
            embedding_hidden_dimension=atom_type_embedding_dims,
            cutoff=convolution_cutoff,
            message_norm=graph_message_norm,
            message_dropout=graph_message_dropout,
            nodewise_norm=graph_node_norm,
            nodewise_dropout=graph_node_dropout,
            radial_embedding=radial_function,
            attention_heads=num_attention_heads,
            periodize_inside_nodes=periodic_structure,
            outside_convolution_type=outside_convolution_type,
        )

        # initialize global pooling operation
        self.global_pool = global_aggregation(graph_aggregator, graph_embedding_depth)

        # molecule features FC layer
        if num_mol_feats != 0:
            self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats)
        else:
            self.mol_fc = None

        # FC model to post-process graph fingerprint
        if num_fc_layers > 0:
            self.gnn_mlp = MLP(layers=num_fc_layers,
                               filters=fc_depth,
                               norm=fc_norm_mode,
                               dropout=fc_dropout_probability,
                               input_dim=graph_embedding_depth,
                               output_dim=fc_depth,
                               conditioning_dim=num_mol_feats,
                               seed=seed
                               )

            if fc_depth != output_dimension:  # only want this if we have to change the dimension
                self.output_fc = nn.Linear(fc_depth, output_dimension, bias=False)
            else:
                self.output_fc = nn.Identity()
        else:
            self.gnn_mlp = nn.Identity()
            if graph_embedding_depth != output_dimension:  # only want this if we have to change the dimension
                self.output_fc = nn.Linear(graph_embedding_depth, output_dimension, bias=False)
            else:
                self.output_fc = nn.Identity()

    def forward(self, data, edges_dict=None, return_latent=False, return_dists=False):

        if edges_dict is None:  # option to pass pre-prepared radial graph
            edges_dict = construct_radial_graph(data.pos, data.batch, data.ptr, self.convolution_cutoff, self.max_num_neighbors, aux_ind=data.aux_ind)

        if self.graph_net.outside_convolution_type != 'none':
            agg_batch = edges_dict['inside_batch']
        else:
            agg_batch = data.batch

        x = data.x  # already cloned before it comes into this function
        if self.concat_pos_to_atom_features:
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

        x = self.graph_net(x, data.pos, data.batch, data.ptr, edges_dict)  # get graph encoding
        if self.global_pool.agg_func == 'molwise':
            x = self.global_pool(x, agg_batch, cluster=data.mol_ind, output_dim=data.num_graphs)  # aggregate atoms to molecules
        else:
            x = self.global_pool(x, agg_batch, output_dim=data.num_graphs)  # aggregate atoms to molecule

        if self.mol_fc is not None:
            mol_feats = self.mol_fc(data.mol_x)  # molecule features are repeated, only need one per molecule (hence data.ptr)
        else:
            mol_feats = None

        if self.num_fc_layers > 0:
            x = self.gnn_mlp(x, conditions=mol_feats)  # mix graph fingerprint with molecule-scale features

        output = self.output_fc(x)

        extra_outputs = {}
        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.cpu().detach().numpy()

        assert torch.sum(torch.isnan(output)) == 0
        assert torch.sum(torch.isfinite(output)) == len(output.flatten())

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output

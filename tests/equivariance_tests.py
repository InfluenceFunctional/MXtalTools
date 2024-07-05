"""
equivariance tests for our equivariant model components

MLP
    -: linear
    -: layernorm
    -: vector activation
    -: end-to-end

GNN
    -: aggregation
    -: convolution
    -: end-to-end
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.nn import Linear

from mxtaltools.models.modules.augmented_softmax_aggregator import VectorAugSoftmaxAggregation
from mxtaltools.models.modules.components import VectorActivation
from mxtaltools.models.modules.components import vectorMLP
from mxtaltools.models.modules.graph_convolution import v_MConv
from mxtaltools.models.modules.vector_LayerNorm import VectorLayerNorm
from tests.utils import check_tensor_similarity, is_module_equivariant

device = 'cpu'
num_samples = 100
num_graphs = 5
feature_depth = 12

rotation_matrix = torch.tensor(R.random().as_matrix(), device=device, dtype=torch.float32)


# TODO update with new graph convolution and end-to-end model

@torch.no_grad()
def test_VectorActivation():
    module = VectorActivation(feature_depth, 'gelu')
    vector_batch = torch.randn(num_samples, 3, feature_depth)
    rotated_output, output_from_rotated = is_module_equivariant(vector_batch, rotation_matrix, module)

    check_tensor_similarity(output_from_rotated, rotated_output)


@torch.no_grad()
def test_VectorLinear():
    module = Linear(feature_depth, feature_depth, bias=False)
    vector_batch = torch.randn(num_samples, 3, feature_depth)
    rotated_output, output_from_rotated = is_module_equivariant(vector_batch, rotation_matrix, module)

    check_tensor_similarity(output_from_rotated, rotated_output)


@torch.no_grad()
def test_VectorLayerNorm():
    for mode in ['graph', 'node']:
        module = VectorLayerNorm(feature_depth, mode=mode)
        vector_batch = torch.randn(num_samples, 3, feature_depth)
        batch_inds = torch.tensor(
            np.random.choice(np.arange(num_graphs), size=num_samples), device=device, dtype=torch.long
        )
        rotated_output, output_from_rotated = is_module_equivariant(vector_batch, rotation_matrix, module,
                                                                    batch=batch_inds)
        check_tensor_similarity(output_from_rotated, rotated_output)


# @torch.no_grad()
# def test_EMLP():
#     module = EMLP(layers=1, input_dim=feature_depth, filters=feature_depth, output_dim=feature_depth, activation='gelu',
#                   dropout=0, norm='layer', add_vector_channel=True, vector_output_dim=feature_depth,
#                   vector_norm='vector layer', ramp_depth=True)
#
#     vector_batch = torch.randn(num_samples, 3, feature_depth)
#     '''forward pass'''
#     rotated_vector_batch = torch.einsum('ij, njk -> nik', rotation_matrix, vector_batch)
#     output = module.vector_forward(i=0, x=vector_batch.norm(dim=1), v=vector_batch, batch=None)
#     output_from_rotated = module.vector_forward(i=0, x=vector_batch.norm(dim=1), v=rotated_vector_batch, batch=None)
#     rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)
#     check_tensor_similarity(output_from_rotated, rotated_output)
#
#     '''full model'''
#     rotated_output, output_from_rotated = is_module_equivariant(x=vector_batch.norm(dim=1), v=vector_batch, rotation_matrix=rotation_matrix, module=module)
#     check_tensor_similarity(output_from_rotated, rotated_output)

@torch.no_grad()
def test_vectorMLP():
    module = vectorMLP(layers=1,
                       input_dim=feature_depth,
                       filters=feature_depth,
                       output_dim=feature_depth,
                       activation='gelu',
                       dropout=0,
                       norm='layer',
                       vector_input_dim=feature_depth,
                       vector_output_dim=feature_depth,
                       vector_norm='vector layer',
                       ramp_depth=True)
    scalar_batch = torch.randn(num_samples, feature_depth)
    vector_batch = torch.randn(num_samples, 3, feature_depth)
    '''forward pass'''
    rotated_vector_batch = torch.einsum('ij, njk -> nik', rotation_matrix, vector_batch)
    _, output = module(x=scalar_batch, v=vector_batch, batch=None)
    _, output_from_rotated = module(x=scalar_batch, v=rotated_vector_batch, batch=None)
    rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)
    check_tensor_similarity(output_from_rotated, rotated_output)

    '''full model'''
    rotated_output, output_from_rotated = is_module_equivariant(x=scalar_batch, v=vector_batch,
                                                                rotation_matrix=rotation_matrix, module=module)
    check_tensor_similarity(output_from_rotated, rotated_output)


#
# @torch.no_grad()
# def test_aggregators():
#     for aggregator in ['equivariant softmax', 'equivariant combo', 'equivariant attention']:
#         module = GlobalAggregation(aggregator, feature_depth)  # todo replace with up-to-date aggregators
#         vector_batch = torch.randn(num_samples, 3, feature_depth)
#         batch_inds = torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples), device=device, dtype=torch.long)
#         rotated_output, output_from_rotated = is_module_equivariant(x=vector_batch.norm(dim=1), v=vector_batch, rotation_matrix=rotation_matrix, module=module, batch=batch_inds)
#
#         check_tensor_similarity(output_from_rotated, rotated_output)
#
@torch.no_grad()
def test_softmax_aggregator():
    module = VectorAugSoftmaxAggregation(channels=feature_depth)  # todo replace with up-to-date aggregators
    vector_batch = torch.randn(num_samples, 3, feature_depth)
    rotated_vector_batch = torch.einsum('ij, njk -> nik', rotation_matrix, vector_batch)

    batch_inds = torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples), device=device, dtype=torch.long)
    output = module(vector_batch, index=batch_inds, dim=0)
    output_from_rotated = module(rotated_vector_batch, index=batch_inds, dim=0)
    rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)
    check_tensor_similarity(output_from_rotated, rotated_output)


#
# @torch.no_grad()
# def test_graph_convolution():
#     module = GCBlock(message_depth=feature_depth, node_embedding_depth=feature_depth, radial_dim=10,
#                      dropout=0, heads=4, add_vector_channel=True)
#     vector_batch = torch.randn(num_samples, 3, feature_depth)
#     batch_inds = torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples), device=device, dtype=torch.long)
#     edge_index = torch.stack([
#         torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples * 10), device=device, dtype=torch.long)
#         for _ in range(2)]
#     )
#     edge_attr = torch.randn(edge_index.shape[1], 10)
#     rotated_vector_batch = torch.einsum('ij, njk -> nik', rotation_matrix, vector_batch)
#     _, output = module(x=vector_batch.norm(dim=1), v=vector_batch, edge_attr=edge_attr, edge_index=edge_index, batch=batch_inds)
#     _, output_from_rotated = module(x=vector_batch.norm(dim=1), v=rotated_vector_batch, edge_attr=edge_attr, edge_index=edge_index, batch=batch_inds)
#     rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)
#
#     check_tensor_similarity(output_from_rotated, rotated_output)


@torch.no_grad()
def test_graph_convolution():
    module = v_MConv(message_depth=feature_depth,
                     node_depth=feature_depth,
                     edge_embedding_dim=10)

    vector_batch = torch.randn(num_samples, 3, feature_depth)
    #batch_inds = torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples), device=device, dtype=torch.long)
    edge_index = torch.stack([
        torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples * 10), device=device, dtype=torch.long)
        for _ in range(2)]
    )
    edge_attr = torch.randn(edge_index.shape[1], 10)
    rotated_vector_batch = torch.einsum('ij, njk -> nik', rotation_matrix, vector_batch)
    output = module(vector_batch, edge_attr=edge_attr, edge_index=edge_index)
    output_from_rotated = module(rotated_vector_batch, edge_attr=edge_attr, edge_index=edge_index)
    rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)

    check_tensor_similarity(output_from_rotated, rotated_output)

#
# @torch.no_grad()
# def test_equivariant_graph():
#     module = MoleculeGraphModel(input_node_dim=1,
#                                 num_mol_feats=0,
#                                 output_dim=feature_depth,
#                                 equivariant=True,
#                                 graph_aggregator='equivariant softmax',
#                                 concat_pos_to_node_dim=True,
#                                 concat_mol_to_node_dim=False,
#                                 concat_crystal_to_node_dim=False,
#                                 activation='gelu',
#                                 fc_num_layers=0,
#                                 fc_hidden_dim=0,
#                                 fc_norm=None,
#                                 fc_dropout=None,
#                                 graph_norm='graph layer',
#                                 graph_dropout=0,
#                                 graph_message_dropout=0,
#                                 graph_message_dim=feature_depth,
#                                 graph_node_dim=feature_depth,
#                                 graph_num_convs=2,
#                                 graph_embedding_dim=feature_depth,
#                                 graph_fcs_per_gc=2,
#                                 graph_num_radial=10,
#                                 graph_radial_function='bessel',
#                                 graph_max_num_neighbors=100,
#                                 graph_convolution_cutoff=2,
#                                 graph_atom_type_embedding_dim=5,
#                                 periodize_inside_nodes=False,
#                                 outside_convolution_type='none',
#                                 cartesian_dimension=3,
#                                 vector_norm='graph vector layer',
#                                 )
#
#     datapoints = []
#     rot_datapoints = []
#     for ind in range(num_graphs):
#         num_nodes = np.random.randint(3, 10, size=1)[0]
#         vector_batch = torch.randn(num_nodes, 3)
#         atom_feats = torch.tensor(np.random.randint(0, 4, num_nodes), device=device, dtype=torch.float32)[:, None]
#
#         datapoints.append(CrystalData(x=atom_feats,
#                                       pos=vector_batch,
#                                       y=torch.ones(1),
#                                       mol_x=None,
#                                       tracking=None,
#                                       unit_cell_pos=None,  # won't collate properly as a torch tensor - must leave as np array
#                                       mult=None,
#                                       sg_ind=None,
#                                       cell_params=None,
#                                       T_fc=None,
#                                       mol_size=None,
#                                       mol_volume=None,
#                                       csd_identifier=None,
#                                       asym_unit_handedness=None,
#                                       symmetry_operators=None))
#         rotated_vector_batch = torch.einsum('ij, nj -> ni', rotation_matrix, vector_batch)
#
#         rot_datapoints.append(CrystalData(x=atom_feats,
#                                           pos=rotated_vector_batch,
#                                           y=torch.ones(1),
#                                           mol_x=None,
#                                           tracking=None,
#                                           unit_cell_pos=None,  # won't collate properly as a torch tensor - must leave as np array
#                                           mult=None,
#                                           sg_ind=None,
#                                           cell_params=None,
#                                           T_fc=None,
#                                           mol_size=None,
#                                           mol_volume=None,
#                                           csd_identifier=None,
#                                           asym_unit_handedness=None,
#                                           symmetry_operators=None))
#
#     collater = Collater(0, 0)
#     data = collater(datapoints)
#     rot_data = collater(rot_datapoints)
#
#     _, output = module(data)
#     _, output_from_rotated = module(rot_data)
#     rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)
#
#     check_tensor_similarity(output_from_rotated, rotated_output)

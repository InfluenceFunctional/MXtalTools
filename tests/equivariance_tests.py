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
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
from torch_geometric.loader.dataloader import Collater

from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.base_models import MoleculeGraphModel
from mxtaltools.models.components import MLP, GlobalAggregation
from mxtaltools.models.gnn_blocks import GC_Block
from mxtaltools.models.vector_LayerNorm import VectorLayerNorm
from tests.utils import check_tensor_similarity, is_module_equivariant
from torch.nn import Linear
from mxtaltools.models.components import VectorActivation

device = 'cpu'
num_samples = 100
num_graphs = 5
feature_depth = 12

rotation_matrix = torch.tensor(R.random().as_matrix(), device=device, dtype=torch.float32)


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
        rotated_output, output_from_rotated = is_module_equivariant(vector_batch, rotation_matrix, module, batch=batch_inds)
        check_tensor_similarity(output_from_rotated, rotated_output)


@torch.no_grad()
def test_EMLP():
    module = MLP(layers=1, input_dim=feature_depth, filters=feature_depth, output_dim=feature_depth, activation='gelu',
                 dropout=0, norm='layer', equivariant=True, vector_output_dim=feature_depth,
                 vector_norm='vector layer', ramp_depth=True)

    vector_batch = torch.randn(num_samples, 3, feature_depth)
    '''forward pass'''
    rotated_vector_batch = torch.einsum('ij, njk -> nik', rotation_matrix, vector_batch)
    output = module.vector_forward(i=0, x=vector_batch.norm(dim=1), v=vector_batch, batch=None)
    output_from_rotated = module.vector_forward(i=0, x=vector_batch.norm(dim=1), v=rotated_vector_batch, batch=None)
    rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)
    check_tensor_similarity(output_from_rotated, rotated_output)

    '''full model'''
    rotated_output, output_from_rotated = is_module_equivariant(x=vector_batch.norm(dim=1), v=vector_batch, rotation_matrix=rotation_matrix, module=module)
    check_tensor_similarity(output_from_rotated, rotated_output)


@torch.no_grad()
def test_aggregators():
    for aggregator in ['equivariant softmax', 'equivariant combo', 'equivariant attention']:
        module = GlobalAggregation(aggregator, feature_depth)
        vector_batch = torch.randn(num_samples, 3, feature_depth)
        batch_inds = torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples), device=device, dtype=torch.long)
        rotated_output, output_from_rotated = is_module_equivariant(x=vector_batch.norm(dim=1), v=vector_batch, rotation_matrix=rotation_matrix, module=module, batch=batch_inds)

        check_tensor_similarity(output_from_rotated, rotated_output)


@torch.no_grad()
def test_graph_convolution():
    module = GC_Block(message_depth=feature_depth, node_embedding_depth=feature_depth, radial_dim=10,
                      dropout=0, heads=4, equivariant=True)
    vector_batch = torch.randn(num_samples, 3, feature_depth)
    batch_inds = torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples), device=device, dtype=torch.long)
    edge_index = torch.stack([
        torch.tensor(np.random.choice(np.arange(num_graphs), size=num_samples * 10), device=device, dtype=torch.long)
        for _ in range(2)]
    )
    edge_attr = torch.randn(edge_index.shape[1], 10)
    rotated_vector_batch = torch.einsum('ij, njk -> nik', rotation_matrix, vector_batch)
    _, output = module(x=vector_batch.norm(dim=1), v=vector_batch, edge_attr=edge_attr, edge_index=edge_index, batch=batch_inds)
    _, output_from_rotated = module(x=vector_batch.norm(dim=1), v=rotated_vector_batch, edge_attr=edge_attr, edge_index=edge_index, batch=batch_inds)
    rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)

    check_tensor_similarity(output_from_rotated, rotated_output)

@torch.no_grad()
def test_equivariant_graph():
    module = MoleculeGraphModel(num_atom_feats=1,
                                num_mol_feats=0,
                                output_dimension=feature_depth,
                                equivariant_graph=True,
                                graph_aggregator='equivariant softmax',
                                concat_pos_to_atom_features=True,
                                concat_mol_to_atom_features=False,
                                concat_crystal_to_atom_features=False,
                                activation='gelu',
                                num_fc_layers=0,
                                fc_depth=0,
                                fc_norm_mode=None,
                                fc_dropout_probability=None,
                                graph_node_norm='graph layer',
                                graph_node_dropout=0,
                                graph_message_dropout=0,
                                graph_message_depth=feature_depth,
                                graph_node_dims=feature_depth,
                                num_graph_convolutions=2,
                                graph_embedding_depth=feature_depth,
                                nodewise_fc_layers=2,
                                num_radial=10,
                                radial_function='bessel',
                                max_num_neighbors=100,
                                convolution_cutoff=2,
                                atom_type_embedding_dims=5,
                                periodic_structure=False,
                                outside_convolution_type='none',
                                cartesian_dimension=3,
                                vector_norm='graph vector layer',
                                )

    datapoints = []
    rot_datapoints = []
    for ind in range(num_graphs):
        num_nodes = np.random.randint(3, 10, size=1)[0]
        vector_batch = torch.randn(num_nodes, 3)
        atom_feats = torch.tensor(np.random.randint(0, 4, num_nodes), device=device, dtype=torch.float32)[:, None]

        datapoints.append(CrystalData(x=atom_feats,
                                      pos=vector_batch,
                                      y=torch.ones(1),
                                      mol_x=None,
                                      tracking=None,
                                      ref_cell_pos=None,  # won't collate properly as a torch tensor - must leave as np array
                                      mult=None,
                                      sg_ind=None,
                                      cell_params=None,
                                      T_fc=None,
                                      mol_size=None,
                                      mol_volume=None,
                                      csd_identifier=None,
                                      asym_unit_handedness=None,
                                      symmetry_operators=None))
        rotated_vector_batch = torch.einsum('ij, nj -> ni', rotation_matrix, vector_batch)

        rot_datapoints.append(CrystalData(x=atom_feats,
                                          pos=rotated_vector_batch,
                                          y=torch.ones(1),
                                          mol_x=None,
                                          tracking=None,
                                          ref_cell_pos=None,  # won't collate properly as a torch tensor - must leave as np array
                                          mult=None,
                                          sg_ind=None,
                                          cell_params=None,
                                          T_fc=None,
                                          mol_size=None,
                                          mol_volume=None,
                                          csd_identifier=None,
                                          asym_unit_handedness=None,
                                          symmetry_operators=None))

    collater = Collater(0, 0)
    data = collater(datapoints)
    rot_data = collater(rot_datapoints)

    _, output = module(data)
    _, output_from_rotated = module(rot_data)
    rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)

    check_tensor_similarity(output_from_rotated, rotated_output)
import torch
from torch_geometric.typing import OptTensor

from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.molecule_graph_model import MolecularCrystalGraphModel


class MolecularCrystalModel(BaseGraphModel):
    def __init__(self, seed, config,
                 atom_features: list,
                 molecule_features: list,
                 output_dim: int,
                 node_standardization_tensor: OptTensor = None,
                 graph_standardization_tensor: OptTensor = None):
        """
        wrapper for molecule model, with appropriate I/O
        """
        super(MolecularCrystalModel, self).__init__()

        torch.manual_seed(seed)
        self.get_data_stats(atom_features,
                            molecule_features,
                            node_standardization_tensor,
                            graph_standardization_tensor)

        self.model = MolecularCrystalGraphModel(
            input_node_dim=self.n_atom_feats,
            num_mol_feats=self.n_mol_feats,
            output_dim=output_dim,
            seed=seed,
            concat_mol_to_node_dim=True,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
        )

    def forward(self, data, return_dists=False, return_latent=False):
        """overwrites base method"""
        # on the fly atom property embeddings
        data = self.featurize_input_graph(data)
        # on the fly input standardization
        data = self.standardize(data)

        return self.model(data.x,
                          data.pos,
                          data.batch,
                          data.ptr,
                          data.mol_x,
                          data.num_graphs,
                          data.aux_ind,
                          data.mol_ind,
                          return_dists=return_dists, return_latent=return_latent)


# class HierarchicalCrystalDiscriminator(nn.Module):
#     def __init__(self, seed, config, n_atom_types, input_depth):
#         '''
#         wrapper for molecule model, with appropriate I/O
#         '''
#         torch.manual_seed(seed)
#         super(HierarchicalCrystalDiscriminator, self).__init__()
#         self.conditioner = molecule_encoder
#         self.gnn = point_graph
#   self.crystal_graph=PeriodicCrystalGraph()
#     def symmetrize(self):
#     def forward(self, data,):
#         # if crystalline, encode the data object, then generate pattern it according to the crystal symmetry
#         # then evaluate material graph
#         # if it's some bulk, thing, just encode all the separate molecules and evaluate the resulting material graph
#         return self.model(data, return_dists=return_dists, return_latent=return_latent)
#


# deprecated
#
# class MolCrystal(BaseGraphModel):
#     def __init__(self, seed, config,
#                  atom_features: list,
#                  molecule_features: list,
#                  node_standardization_tensor: OptTensor = None,
#                  graph_standardization_tensor: OptTensor = None):
#         """
#         wrapper for molecule model, with appropriate I/O
#         """
#         super(MolCrystal, self).__init__()
#
#         torch.manual_seed(seed)
#         self.get_data_stats(atom_features,
#                             molecule_features,
#                             node_standardization_tensor,
#                             graph_standardization_tensor)
#
#         self.model = MoleculeGraphModel(
#             input_node_dim=self.n_atom_feats,
#             num_mol_feats=self.n_mol_feats,
#             output_dim=2 + 1,  # 2 for classification and 1 for distance regression
#             seed=seed,
#             graph_aggregator=config.graph_aggregator,
#             concat_pos_to_node_dim=False,
#             concat_mol_to_node_dim=True,
#             concat_crystal_to_node_dim=False,
#             activation=config.activation,
#             fc_config=config.fc,
#             graph_config=config.graph,
#             periodize_inside_nodes=True,
#             outside_convolution_type=config.periodic_convolution_type
#         )
#
#     def forward(self, data, return_dists=False, return_latent=False):
#         """overwrites base method"""
#         # on the fly atom property embeddings
#         data = self.featurize_input_graph(data)
#         # on the fly input standardization
#         data = self.standardize(data)
#
#         outputs = self.model(data, return_dists=return_dists, return_latent=return_latent)
#
#         if isinstance(outputs, tuple):  # if we have extra outputs, pick out the actual model output and adjust
#             model_outputs, extra_outputs = outputs
#             return (torch.cat([model_outputs[:, :2], F.softplus(model_outputs[:, -1, None])], dim=1), extra_outputs)
#
#         else:
#             return torch.cat([outputs[:, 0], F.softplus(outputs[:, 1])], dim=1)

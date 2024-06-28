from typing import Tuple, Optional

import torch

from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.molecule_graph_model import MoleculeClusterModel


class MoleculeClusterClassifier(BaseGraphModel):
    def __init__(self,
                 seed,
                 config,
                 output_dim,
                 atom_features: list,
                 molecule_features: list,
                 node_standardization_tensor: torch.Tensor,
                 graph_standardization_tensor: torch.Tensor,
                 ):
        super(MoleculeClusterClassifier, self).__init__()

        torch.manual_seed(seed)
        self.get_data_stats(atom_features,
                            molecule_features,
                            node_standardization_tensor,
                            graph_standardization_tensor)

        self.model = MoleculeClusterModel(
            input_node_dim=self.n_atom_feats,
            num_mol_feats=0,
            output_dim=output_dim,
            seed=seed,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
        )

    def forward(self,
                data: CrystalData,
                return_dists: bool = False,
                return_latent: bool = False,
                return_embedding: bool = False,
                ) -> Tuple[torch.Tensor, Optional[dict]]:
        # featurize atom properties on the fly
        data = self.featurize_input_graph(data)

        # standardize on the fly from model-attached statistics
        data = self.standardize(data)

        return self.model(data.x,
                          data.pos,
                          data.ptr,
                          data.mol_x,
                          data.num_graphs,
                          data.mol_ind,
                          data.T_fc,
                          data.edge_index,
                          data.edge_attr,
                          return_dists=return_dists,
                          return_latent=return_latent,
                          return_embedding=return_embedding)


# deprecated
# class PolymorphClassifier(BaseGraphModel):
#     def __init__(self, seed, config,
#                  dataDims: dict,
#                  atom_features: list,
#                  molecule_features: list,
#                  node_standardization_tensor: torch.tensor,
#                  graph_standardization_tensor: torch.tensor,
#                  ):
#         super(PolymorphClassifier, self).__init__()
#
#         torch.manual_seed(seed)
#         self.get_data_stats(atom_features,
#                             molecule_features,
#                             node_standardization_tensor,
#                             graph_standardization_tensor)
#
#         self.model = MoleculeGraphModel(
#             input_node_dim=self.n_atom_feats,
#             num_mol_feats=0,
#             output_dim=dataDims['num_polymorphs'] + dataDims['num_topologies'],
#             seed=seed,
#             graph_aggregator='molwise',
#             activation=config.activation,
#             fc_config=config.fc,
#             graph_config=config.graph,
#             outside_convolution_type='none'
#         )
#
#     def forward(self, data, return_dists=False, return_latent=False, return_embedding=False,
#                 skip_standardization=False):
#         if not skip_standardization:
#             data = self.standardize(data)
#
#         return self.model(data,
#                           return_dists=return_dists,
#                           return_latent=return_latent,
#                           return_embedding=return_embedding)
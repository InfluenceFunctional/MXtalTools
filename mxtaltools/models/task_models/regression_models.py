from argparse import Namespace
from typing import Optional

import torch

from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.molecule_graph_model import ScalarMoleculeGraphModel


class MoleculeScalarRegressor(BaseGraphModel):
    def __init__(self,
                 config: Namespace,
                 atom_features: list,
                 molecule_features: list,
                 node_standardization_tensor: Optional[torch.Tensor] = None,
                 graph_standardization_tensor: Optional[torch.Tensor] = None,
                 seed: int = 0
                 ):
        """
        wrapper for molecule model, with appropriate I/O
        """
        super(MoleculeScalarRegressor, self).__init__()
        torch.manual_seed(seed)

        self.get_data_stats(atom_features,
                            molecule_features,
                            node_standardization_tensor,
                            graph_standardization_tensor)

        self.model = ScalarMoleculeGraphModel(
            input_node_dim=self.n_atom_feats,
            num_mol_feats=self.n_mol_feats,
            output_dim=1,
            seed=seed,
            concat_mol_to_node_dim=True,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
        )

    # uses default forward method inherited from base class


# deprecated
# class MoleculeRegressor(BaseGraphModel):
#     def __init__(self,
#                  seed,
#                  config,
#                  atom_features: list,
#                  molecule_features: list,
#                  node_standardization_tensor: OptTensor = None,
#                  graph_standardization_tensor: OptTensor = None
#                  ):
#         """
#         wrapper for molecule model, with appropriate I/O
#         """
#         super(MoleculeRegressor, self).__init__()
#         torch.manual_seed(seed)
#
#         self.get_data_stats(atom_features,
#                             molecule_features,
#                             node_standardization_tensor,
#                             graph_standardization_tensor)
#
#         self.model = MoleculeGraphModel(
#             input_node_dim=self.n_atom_feats,
#             num_mol_feats=self.n_mol_feats,
#             output_dim=1,
#             seed=seed,
#             graph_aggregator=config.graph_aggregator,
#             concat_mol_to_node_dim=True,
#             activation=config.activation,
#             fc_config=config.fc,
#             graph_config=config.graph,
#             periodize_inside_nodes=False,
#             outside_convolution_type='none'
#         )
#
#     # uses default forward method inherited from base class

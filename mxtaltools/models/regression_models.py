import torch

from mxtaltools.models.base_graph_model import BaseGraphModel
from mxtaltools.models.molecule_graph_model import MoleculeGraphModel


class MoleculeRegressor(BaseGraphModel):
    def __init__(self,
                 seed,
                 config,
                 dataDims: dict = None,
                 num_atom_features: int = None,
                 num_molecule_features: int = None,
                 node_standardization_tensor: torch.tensor = None,
                 graph_standardization_tensor: torch.tensor = None):
        """
        wrapper for molecule model, with appropriate I/O
        """
        super(MoleculeRegressor, self).__init__()
        torch.manual_seed(seed)

        self.get_data_stats(dataDims,
                            graph_standardization_tensor,
                            node_standardization_tensor,
                            num_atom_features,
                            num_molecule_features)

        self.model = MoleculeGraphModel(
            input_node_dim=self.n_atom_feats,
            num_mol_feats=self.n_mol_feats,
            output_dim=1,
            seed=seed,
            graph_aggregator=config.graph_aggregator,
            concat_mol_to_node_dim=True,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
            periodize_inside_nodes=False,
            outside_convolution_type='none'
        )

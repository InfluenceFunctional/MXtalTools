
import torch.nn.functional as F

from mxtaltools.models.base_graph_model import BaseGraphModel
from mxtaltools.models.molecule_graph_model import MoleculeGraphModel
import torch


class MolCrystal(BaseGraphModel):
    def __init__(self, seed, config, dataDims=None,
                 num_atom_features=None, num_molecule_features=None,
                 node_standardization_tensor: torch.tensor = None,
                 graph_standardization_tensor: torch.tensor = None):
        """
        wrapper for molecule model, with appropriate I/O
        """
        super(MolCrystal, self).__init__()

        torch.manual_seed(seed)
        self.get_data_stats(dataDims,
                            graph_standardization_tensor,
                            node_standardization_tensor,
                            num_atom_features,
                            num_molecule_features)

        self.model = MoleculeGraphModel(
            input_node_dim=self.num_atom_feats,
            num_mol_feats=self.num_mol_feats,
            output_dim=2 + 1,  # 2 for classification and 1 for distance regression
            seed=seed,
            graph_aggregator=config.graph_aggregator,
            concat_pos_to_node_dim=False,
            concat_mol_to_node_dim=True,
            concat_crystal_to_node_dim=False,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
            periodize_inside_nodes=True,
            outside_convolution_type=config.periodic_convolution_type
        )

    def forward(self, data, return_dists=False, return_latent=False, skip_standardization=False):
        """overwrites base method"""
        if not skip_standardization:
            data = self.standardize(data)

        outputs = self.model(data, return_dists=return_dists, return_latent=return_latent)

        if len(outputs) > 1:  # if we have extra outputs, pick out the actual model output and adjust
            model_outputs = outputs[0]
            rescaled_distance = F.softplus(model_outputs[:, -1])  # set distance prediction as strictly positive
            outputs = (torch.cat([model_outputs[:, :2], rescaled_distance[:, None]], dim=1), outputs[1])
            return outputs

        else:
            model_outputs = outputs
            model_outputs[:, -1] = F.softplus(model_outputs[:, -1])  # set distance prediction as strictly positive
            return model_outputs

# class HierarchicalCrystalDiscriminator(nn.Module):
#     def __init__(self, seed, config, n_atom_types, input_depth):
#         '''
#         wrapper for molecule model, with appropriate I/O
#         '''
#         torch.manual_seed(seed)
#         super(HierarchicalCrystalDiscriminator, self).__init__()
#         self.conditioner = molecule_encoder
#         self.gnn = point_graph
#
#     def forward(self, data,):
#         # if crystalline, encode the data object, then generate pattern it according to the crystal symmetry
#         # then evaluate material graph
#         # if it's some bulk, thing, just encode all the separate molecules and evaluate the resulting material graph
#         return self.model(data, return_dists=return_dists, return_latent=return_latent)

#

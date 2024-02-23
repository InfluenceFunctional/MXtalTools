import torch.nn as nn

import torch.nn.functional as F
from models.base_models import molecule_graph_model
import torch


class CrystalDiscriminator(nn.Module):
    def __init__(self, seed, config, dataDims=None, num_atom_features=None, num_molecule_features=None):
        '''
        wrapper for molecule model, with appropriate I/O
        '''
        torch.manual_seed(seed)
        if dataDims is not None:
            n_atom_feats = dataDims['num_atom_features']
            n_mol_feats = dataDims['num_molecule_features']
        else:
            n_atom_feats = num_atom_features
            n_mol_feats = num_molecule_features

        super(CrystalDiscriminator, self).__init__()
        self.model = molecule_graph_model(
            num_atom_feats=n_atom_feats,
            num_mol_feats=n_mol_feats,
            output_dimension=2 + 1,  # 2 for classification and 1 for distance regression
            seed=seed,
            graph_aggregator=config.graph_aggregator,
            concat_pos_to_atom_features=False,
            concat_mol_to_atom_features=config.concat_mol_to_atom_features,
            concat_crystal_to_atom_features=False,
            activation=config.activation,
            num_fc_layers=config.num_fc_layers,
            fc_depth=config.fc_depth,
            fc_norm_mode=config.fc_norm_mode,
            fc_dropout_probability=config.fc_dropout_probability,
            graph_node_norm=config.graph_node_norm,
            graph_node_dropout=config.graph_node_dropout,
            graph_message_dropout=config.graph_message_dropout,
            num_attention_heads=config.num_attention_heads,
            graph_message_depth=config.graph_message_depth,
            graph_node_dims=config.graph_node_dims,
            num_graph_convolutions=config.num_graph_convolutions,
            graph_embedding_depth=config.graph_embedding_depth,
            nodewise_fc_layers=config.nodewise_fc_layers,
            num_radial=config.num_radial,
            radial_function=config.radial_function,
            max_num_neighbors=config.max_num_neighbors,
            convolution_cutoff=config.convolution_cutoff,
            atom_type_embedding_dims=config.atom_type_embedding_dims,
            periodic_structure=True,
            outside_convolution_type=config.periodic_convolution_type
        )

    def forward(self, data, return_dists=False, return_latent=False):
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

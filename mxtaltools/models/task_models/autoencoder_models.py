import torch
from torch import nn as nn

from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.molecule_graph_model import VectorMoleculeGraphModel
from mxtaltools.models.modules.components import Scalarizer, vectorMLP


# noinspection PyAttributeOutsideInit
class Mo3ENet(BaseGraphModel):
    def __init__(self, seed, config,
                 num_atom_types: int,
                 atom_embedding_vector: torch.tensor,
                 radial_normalization: float,
                 infer_protons: bool,
                 protons_in_input: bool,
                 ):
        super(Mo3ENet, self).__init__()
        """
        3D o3 equivariant multi-type point cloud autoencoder model
        Mo3ENet
        """

        torch.manual_seed(seed)

        self.cartesian_dimension = 3
        self.num_classes = num_atom_types
        self.output_depth = self.num_classes + self.cartesian_dimension + 1
        self.num_decoder_nodes = config.decoder.num_nodes
        self.bottleneck_dim = config.bottleneck_dim

        self.register_buffer('atom_embedding_vector', atom_embedding_vector)
        self.register_buffer('radial_normalization', torch.tensor(radial_normalization, dtype=torch.float32))
        self.register_buffer('protons_in_input', torch.tensor(protons_in_input, dtype=torch.bool))
        self.register_buffer('inferring_protons', torch.tensor(infer_protons, dtype=torch.bool))
        self.register_buffer('convolution_cutoff', config.encoder.graph.cutoff / self.radial_normalization)

        self.encoder = Mo3ENetEncoder(seed,
                                      config.encoder,
                                      config.bottleneck_dim,
                                      override_cutoff=self.convolution_cutoff)
        self.decoder = Mo3ENetDecoder(seed,
                                      config.decoder,
                                      config.bottleneck_dim,
                                      self.output_depth, self.num_decoder_nodes)
        self.scalarizer = Scalarizer(config.bottleneck_dim,
                                     self.cartesian_dimension,
                                     None, None, 0)

    def forward(self,
                data: CrystalData,
                return_latent: bool = False,
                return_dists: bool = False,
                ):
        encoding = self.encode(data)
        decoding = self.decode(encoding)

        if return_latent:
            return decoding, encoding
        else:
            return decoding

    def encode(self, data):
        # normalize radii
        data.pos /= self.radial_normalization
        _, encoding = self.encoder(data)

        return encoding

    def decode(self, encoding):
        """encoding nx3xk"""
        scalar_decoding, vector_decoding = self.decoder(self.scalarizer(encoding), v=encoding)

        '''combine vector and scalar features to n*nodes x m'''
        # de-normalize predicted node positions and rearrange to correct format
        decoding = torch.cat([
            vector_decoding.permute(0, 2, 1).reshape(len(vector_decoding) * self.num_decoder_nodes, 3) * self.radial_normalization,
            scalar_decoding.reshape(len(scalar_decoding) * self.num_decoder_nodes, self.output_depth - 3)],
            dim=-1)

        return decoding


class Mo3ENetDecoder(nn.Module):
    def __init__(self, seed, config, bottleneck_dim, output_depth, num_nodes):
        super(Mo3ENetDecoder, self).__init__()
        self.model = vectorMLP(
            seed=seed,
            layers=config.fc.num_layers,
            filters=config.fc.hidden_dim,
            input_dim=bottleneck_dim,
            vector_input_dim=bottleneck_dim,
            vector_output_dim=num_nodes,
            output_dim=(output_depth - 3) * num_nodes,
            conditioning_dim=0,
            activation=config.activation,
            norm=config.fc.norm,
            dropout=config.fc.dropout,
            vector_norm=config.fc.vector_norm,
            ramp_depth=config.ramp_depth,
        )

    def forward(self, x, v):
        return self.model(x, v)


class Mo3ENetEncoder(nn.Module):
    def __init__(self, seed, config, bottleneck_dim, override_cutoff=None):
        super(Mo3ENetEncoder, self).__init__()
        self.model = VectorMoleculeGraphModel(
            input_node_dim=1,
            num_mol_feats=0,
            output_dim=bottleneck_dim,
            seed=seed,
            concat_pos_to_node_dim=True,
            concat_mol_to_node_dim=False,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
            override_cutoff=override_cutoff,
        )

    def forward(self, data):
        return self.model(data.x,
                          data.pos,
                          data.batch,
                          data.ptr,
                          num_graphs=data.num_graphs,
                          )

# model from original draft of Mo3ENet paper
# # noinspection PyAttributeOutsideInit
# class PointAutoencoder(BaseGraphModel):
#     def __init__(self, seed, config,
#                  num_atom_types: int,
#                  atom_embedding_vector: torch.tensor,
#                  radial_normalization: float,
#                  infer_protons: bool,
#                  protons_in_input: bool,
#                  ):
#         super(PointAutoencoder, self).__init__()
#         """
#         3D o3 equivariant multi-type point cloud autoencoder model
#         Mo3ENet
#         """
#
#         torch.manual_seed(seed)
#
#         self.cartesian_dimension = 3
#         self.num_classes = num_atom_types
#         self.output_depth = self.num_classes + self.cartesian_dimension + 1
#         self.num_decoder_nodes = config.decoder.num_nodes
#         self.bottleneck_dim = config.bottleneck_dim
#
#         self.register_buffer('atom_embedding_vector', atom_embedding_vector)
#         self.register_buffer('radial_normalization', torch.tensor(radial_normalization, dtype=torch.float32))
#         self.register_buffer('protons_in_input', torch.tensor(protons_in_input, dtype=torch.bool))
#         self.register_buffer('inferring_protons', torch.tensor(infer_protons, dtype=torch.bool))
#         self.register_buffer('convolution_cutoff', config.encoder.graph.cutoff / self.radial_normalization)
#
#         self.encoder = PointEncoder(seed, config.encoder, config.bottleneck_dim,
#                                     override_cutoff=self.convolution_cutoff)
#         self.decoder = PointDecoder(seed, config.decoder, config.bottleneck_dim,
#                                     self.output_depth, self.num_decoder_nodes)
#         self.scalarizer = Scalarizer(config.bottleneck_dim, self.cartesian_dimension, None, None, 0)
#
#     def forward(self, data, return_encoding=False, **kwargs):
#         encoding = self.encode(data)
#         decoding = self.decode(encoding)
#
#         # de-normalize predicted node positions and rearrange to correct format
#         decoding_fin = torch.cat(
#             [decoding[:, :self.cartesian_dimension] * self.radial_normalization,
#              decoding[:, self.cartesian_dimension:]], dim=1)
#         if return_encoding:
#             return decoding_fin, encoding
#         else:
#             return decoding_fin
#
#     def encode(self, data):
#         # # subtract mean
#         # centroids = scatter(data.pos, data.batch, reduce='mean', dim=0)
#         # data.pos -= torch.repeat_interleave(centroids, data.num_atoms, dim=0, output_size=data.num_nodes)
#         # normalize radii
#         data.pos /= self.radial_normalization
#         _, encoding = self.encoder(data)
#
#         # assert torch.sum(torch.isnan(encoding)) == 0, f"NaN in encoder output {get_model_nans(self.encoder)}"
#         return encoding
#
#     def decode(self, encoding):
#         """encoding nx3xk"""
#         decoding = self.decoder(self.scalarizer(encoding),
#                                 #x=torch.zeros_like(encoding[:, 0, :]),  #
#                                 v=encoding)  # scalar input comes through scalarizer in first layer vector_to_scalar
#
#         scalar_decoding, vector_decoding = decoding
#         '''combine vector and scalar features to n*nodes x m'''
#         decoding = torch.cat([
#             vector_decoding.permute(0, 2, 1).reshape(len(vector_decoding) * self.num_decoder_nodes, 3),
#             scalar_decoding.reshape(len(scalar_decoding) * self.num_decoder_nodes, self.output_depth - 3)],
#             dim=-1)
#
#         assert torch.sum(torch.isnan(encoding)) == 0, f"NaN in decoder output with {get_model_nans(self.decoder)}"
#
#         return decoding
#
#
# class PointDecoder(nn.Module):
#     def __init__(self, seed, config, bottleneck_dim, output_depth, num_nodes):
#         super(PointDecoder, self).__init__()
#         self.model = EMLP(
#             seed=seed,
#             layers=config.fc.num_layers,
#             filters=config.fc.hidden_dim,
#             input_dim=bottleneck_dim,
#             output_dim=(output_depth - 3) * num_nodes,
#             conditioning_dim=0,
#             activation=config.activation,
#             norm=config.fc.norm,
#             dropout=config.fc.dropout,
#             add_vector_channel=True,
#             vector_output_dim=num_nodes,
#             vector_norm=config.vector_norm,
#             ramp_depth=config.ramp_depth,
#         )
#
#     def forward(self, x, v):
#         return self.model(x, v)
#
#
# class PointEncoder(nn.Module):
#     def __init__(self, seed, config, bottleneck_dim, override_cutoff=None):
#         super(PointEncoder, self).__init__()
#         self.model = MoleculeGraphModel(
#             input_node_dim=1,
#             num_mol_feats=0,
#             output_dim=bottleneck_dim,
#             seed=seed,
#             equivariant=True,
#             graph_aggregator=config.graph_aggregator,
#             concat_pos_to_node_dim=True,
#             concat_mol_to_node_dim=False,
#             concat_crystal_to_node_dim=False,
#             activation=config.activation,
#             fc_config=config.fc,
#             graph_config=config.graph,
#             periodize_inside_nodes=False,
#             outside_convolution_type='none',
#             vector_norm=config.vector_norm,
#             override_cutoff=override_cutoff,
#         )
#
#     def forward(self, data):
#         return self.model(data)

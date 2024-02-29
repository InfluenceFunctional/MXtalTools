from torch import nn as nn

from mxtaltools.models.base_models import MoleculeGraphModel
from mxtaltools.models.components import MLP, Scalarizer
from mxtaltools.models.utils import get_model_nans

import torch


class PointAutoencoder(nn.Module):
    def __init__(self, seed, config, num_atom_types):
        super(PointAutoencoder, self).__init__()
        """
        3D o3 equivariant multi-type point cloud autoencoder model
        """
        cartesian_dimension = 3
        self.num_classes = num_atom_types
        self.output_depth = self.num_classes + cartesian_dimension + 1
        self.num_nodes = config.num_decoder_points

        self.fully_equivariant = True
        self.variational = config.variational_encoder
        self.bottleneck_dim = config.bottleneck_dim

        self.encoder = PointEncoder(seed, config)

        self.scalarizer = Scalarizer(config.bottleneck_dim, 3, None, None, 0)

        self.decoder = MLP(
            layers=config.num_decoder_layers,
            filters=config.embedding_depth,
            input_dim=config.bottleneck_dim,
            output_dim=(self.output_depth - 3) * self.num_nodes,
            conditioning_dim=0,
            activation='gelu',
            norm=config.decoder_norm_mode,
            dropout=config.decoder_dropout_probability,
            equivariant=True,
            vector_output_dim=self.num_nodes,
            vector_norm=config.decoder_vector_norm,
            ramp_depth=config.decoder_ramp_depth,
        )

    def forward(self, data, return_encoding=False):
        encoding = self.encode(data)
        if return_encoding:
            return self.decode(encoding), encoding
        else:
            return self.decode(encoding)

    def encode(self, data, z=None):

        if self.variational:
            encoding = self.variational_sampling(data, z)
        else:
            _, encoding = self.encoder(data)

        assert torch.sum(torch.isnan(encoding)) == 0, f"NaN in encoder output {get_model_nans(self.encoder)}"

        return encoding

    def variational_sampling(self, data, z):
        """
        here we enforce regularization only against the norms of the embedding vectors
        the directions may not be / in practice are not randomly distributed, so generation based on random directions will not work
        would require somehow to regularize also over directions (maybe over dot products) as well
        though this is also impossible in principle for flat molecules, since they cannot mix vectors out of plane
        """
        x, v = self.encoder(data)
        mu = torch.linalg.norm(v, dim=1)
        log_sigma = x.clip(max=1)  # if this becomes large, we get Inf in next step
        sigma = torch.exp(0.5 * log_sigma)

        if z is None:
            z = torch.randn((len(sigma), 3, sigma.shape[-1]), dtype=v.dtype, device=v.device)

        stochastic_weight = torch.linalg.norm(z * sigma[:, None, :] + mu[:, None, :], dim=1)  # parameterized distribution
        encoding = stochastic_weight[:, None, :] * v / (torch.linalg.norm(v, dim=1)[:, None, :] + 1e-3)  # rescale vector length by learned distribution
        self.kld = (sigma ** 2 + mu ** 2 - log_sigma - 0.5)  # KL divergence of embedded distribution
        return encoding

    def decode(self, encoding):
        """encoding nx3xk"""
        decoding = self.decoder(x=self.scalarizer(encoding),
                                v=encoding)

        scalar_decoding, vector_decoding = decoding
        '''combine vector and scalar features to n*nodes x m'''
        decoding = torch.cat([
            vector_decoding.permute(0, 2, 1).reshape(len(vector_decoding) * self.num_nodes, 3),
            scalar_decoding.reshape(len(scalar_decoding) * self.num_nodes, self.output_depth - 3)],
            dim=-1)

        assert torch.sum(torch.isnan(encoding)) == 0, f"NaN in decoder output with {get_model_nans(self.decoder)}"

        return decoding


class PointEncoder(nn.Module):
    def __init__(self, seed, config):
        super(PointEncoder, self).__init__()
        self.model = MoleculeGraphModel(
            num_atom_feats=1,
            num_mol_feats=0,
            output_dimension=config.bottleneck_dim,
            seed=seed,
            equivariant_graph=True,
            graph_aggregator=config.graph_aggregator,
            concat_pos_to_atom_features=True,
            concat_mol_to_atom_features=False,
            concat_crystal_to_atom_features=False,
            activation=config.activation,
            num_fc_layers=0,
            fc_depth=0,
            fc_norm_mode=None,
            fc_dropout_probability=None,
            graph_node_norm=config.graph_node_norm,
            graph_node_dropout=config.graph_node_dropout,
            graph_message_dropout=config.graph_message_dropout,
            num_attention_heads=config.num_attention_heads,
            graph_message_depth=config.graph_message_depth,
            graph_node_dims=config.embedding_depth,
            num_graph_convolutions=config.num_graph_convolutions,
            graph_embedding_depth=config.embedding_depth,
            nodewise_fc_layers=config.nodewise_fc_layers,
            num_radial=config.num_radial,
            radial_function=config.radial_function,
            max_num_neighbors=config.max_num_neighbors,
            convolution_cutoff=config.convolution_cutoff,
            atom_type_embedding_dims=config.atom_type_embedding_dims,
            periodic_structure=False,
            outside_convolution_type='none',
            cartesian_dimension=3,
            vector_norm=config.encoder_vector_norm,
        )

    def forward(self, data):
        return self.model(data)

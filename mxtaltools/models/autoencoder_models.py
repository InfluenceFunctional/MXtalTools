from torch import nn as nn

from mxtaltools.models.base_graph_model import BaseGraphModel
from mxtaltools.models.molecule_graph_model import MoleculeGraphModel
from mxtaltools.models.components import Scalarizer, MLP
from mxtaltools.models.utils import get_model_nans

import torch


# noinspection PyAttributeOutsideInit
class PointAutoencoder(BaseGraphModel):
    def __init__(self, seed, config, num_atom_types,
                 dataDims=None,
                 num_atom_features=None, num_molecule_features=None,
                 node_standardization_tensor: torch.tensor = None,
                 graph_standardization_tensor: torch.tensor = None
                 ):
        super(PointAutoencoder, self).__init__()
        """
        3D o3 equivariant multi-type point cloud autoencoder model
        Mo3ENN
        """

        torch.manual_seed(seed)
        self.get_data_stats(dataDims,
                            graph_standardization_tensor,
                            node_standardization_tensor,
                            num_atom_features,
                            num_molecule_features)

        cartesian_dimension = 3
        self.num_classes = num_atom_types
        self.output_depth = self.num_classes + cartesian_dimension + 1
        self.num_nodes = config.num_decoder_points

        self.fully_equivariant = True
        self.variational = config.variational_encoder
        self.bottleneck_dim = config.bottleneck_dim

        self.encoder = PointEncoder(seed, config)

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

        self.scalarizer = Scalarizer(config.bottleneck_dim, 3, None, None, 0)
        self.decoder.vector_to_scalar[0] = self.scalarizer

    def forward(self, data,
                return_encoding=False,
                skip_standardization=False):

        if not skip_standardization:
            self.standardize(data)

        encoding = self.encode(data)
        if return_encoding:
            return self.decode(encoding), encoding
        else:
            return self.decode(encoding)

    def encode(self, data, z=None, skip_standardization=False):
        if not skip_standardization:
            self.standardize(data)

        if self.variational:
            encoding = self.variational_sampling(data, z)
        else:
            _, encoding = self.encoder(data)

        assert torch.sum(torch.isnan(encoding)) == 0, f"NaN in encoder output {get_model_nans(self.encoder)}"

        return encoding

    def variational_sampling(self, data, z):
        """  # TODO reconsider / rewrite mathematical issues here
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

        stochastic_weight = torch.linalg.norm(z * sigma[:, None, :] + mu[:, None, :],
                                              dim=1)  # parameterized distribution
        encoding = stochastic_weight[:, None, :] * v / (
                    torch.linalg.norm(v, dim=1)[:, None, :] + 1e-3)  # rescale vector length by learned distribution
        self.kld = (sigma ** 2 + mu ** 2 - log_sigma - 0.5)  # KL divergence of embedded distribution
        return encoding

    def decode(self, encoding):
        """encoding nx3xk"""
        decoding = self.decoder(x=torch.zeros_like(encoding[:, 0, :]),  # self.scalarizer(encoding),
                                v=encoding)  # scalar input comes through scalarizer in first layer vector_to_scalar

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
            input_node_dim=1,
            num_mol_feats=0,
            output_dim=config.bottleneck_dim,
            seed=seed,
            equivariant=True,
            graph_aggregator=config.graph_aggregator,
            concat_pos_to_node_dim=True,
            concat_mol_to_node_dim=False,
            concat_crystal_to_node_dim=False,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
            periodize_inside_nodes=False,
            outside_convolution_type='none',
            vector_norm=config.encoder_vector_norm,
        )

    def forward(self, data):
        return self.model(data)

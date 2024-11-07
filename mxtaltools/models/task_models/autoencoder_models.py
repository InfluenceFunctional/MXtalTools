import torch
from torch import nn as nn

from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.molecule_graph_model import VectorMoleculeGraphModel
from mxtaltools.models.modules.components import Scalarizer, vectorMLP
from mxtaltools.models.utils import collate_decoded_data, ae_reconstruction_loss
from mxtaltools.reporting.ae_reporting import scaffolded_decoder_clustering, swarm_vs_tgt_fig


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
        # todo add type distance scaling and num atom types and node weight temperature
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

    def encode(self,
               data,
               override_centering: bool = False):
        # normalize radii
        if not override_centering:
            assert torch.linalg.norm(data.pos.mean(0)) < 1e-3, "Encoder trained only for centered molecules!"
        data.pos /= self.radial_normalization
        _, encoding = self.encoder(data)

        return encoding

    def decode(self, encoding):
        """encoding nx3xk"""
        scalar_decoding, vector_decoding = self.decoder(self.scalarizer(encoding), v=encoding)

        '''combine vector and scalar features to n*nodes x m'''
        # de-normalize predicted node positions and rearrange to correct format
        decoding = torch.cat([
            vector_decoding.permute(0, 2, 1).reshape(len(vector_decoding) * self.num_decoder_nodes,
                                                     3) * self.radial_normalization,
            scalar_decoding.reshape(len(scalar_decoding) * self.num_decoder_nodes, self.output_depth - 3)],
            dim=-1)

        return decoding

    def compile_self(self, dynamic=True, fullgraph=False):
        self.encoder = torch.compile(self.encoder, dynamic=dynamic, fullgraph=fullgraph)
        self.decoder = torch.compile(self.decoder, dynamic=dynamic, fullgraph=fullgraph)
        self.scalarizer = torch.compile(self.scalarizer, dynamic=dynamic, fullgraph=fullgraph)

    def check_embedding_quality(self, data,
                                sigma=0.35,
                                type_distance_scaling=2,
                                # todo next two should be properties of the model
                                node_weight_temperature=1,
                                num_atom_types=5,
                                visualize=False,
                                ):
        encoding = self.encode(data.clone())
        decoding = self.decode(encoding)

        data.x = self.atom_embedding_vector[data.x].flatten()
        decoded_data, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
            collate_decoded_data(data,
                                 decoding,
                                 self.num_decoder_nodes,
                                 node_weight_temperature,
                                 data.x.device))

        (nodewise_reconstruction_loss,
         nodewise_type_loss,
         reconstruction_loss,
         self_likelihoods,
         ) = ae_reconstruction_loss(data,
                                    decoded_data,
                                    nodewise_weights,
                                    num_atom_types,
                                    type_distance_scaling,
                                    sigma)

        rmsds = torch.zeros(data.num_graphs)
        max_dists = torch.zeros_like(rmsds)
        tot_overlaps = torch.zeros_like(rmsds)
        match_successful = torch.zeros_like(rmsds)
        for ind in range(data.num_graphs):
            rmsds[ind], max_dists[ind], tot_overlaps[ind], match_successful[ind], fig2 = scaffolded_decoder_clustering(
                ind,
                data,
                decoded_data,
                num_atom_types,
                return_fig=True)
        if visualize:
            for ind in range(data.num_graphs):
                swarm_vs_tgt_fig(data, decoded_data, num_atom_types, graph_ind=ind).show()

        return reconstruction_loss, rmsds, max_dists, tot_overlaps, match_successful


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

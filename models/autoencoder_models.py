import torch
import torch.nn as nn

from models.base_models import molecule_graph_model
from models.basis_functions import BesselBasisLayer, GaussianEmbedding
from models.components import MLP
import e3nn
from torch_scatter import scatter


class point_autoencoder(nn.Module):
    def __init__(self, seed, config, dataDims):
        super(point_autoencoder, self).__init__()
        '''conditioning model'''

        self.num_classes = dataDims['num_atom_types']
        self.output_depth = self.num_classes + 3 + 1
        self.num_nodes = config.num_decoder_points

        if config.encoder_type.lower() == 'equivariant':
            self.encoder = equivariant_encoder(config)
        elif self.encoder_type.lower() == 'variant':
            self.encoder = point_encoder(seed, config)

        if config.decoder_type.lower() == 'equivariant':
            self.decoder = equivariant_decoder(config)
        elif config.decoder_type.lower() == 'variant':  # unstructured swarm decoder
            self.decoder = MLP(
                layers=config.num_decoder_layers,
                filters=config.embedding_depth,
                input_dim=config.embedding_depth,
                output_dim=self.output_depth * self.num_nodes,
                activation='gelu',
                norm=config.decoder_norm_mode,
                dropout=config.decoder_dropout_probability,
            )

    def forward(self, data):
        encoding = self.encoder(data)

        return self.decoder(encoding).reshape(self.num_nodes * data.num_graphs, self.output_depth)

    def encode(self, data):
        """
        pass only the encoding
        """
        return self.encoder(data)


class equivariant_decoder(nn.Module):
    def __init__(self, config):
        super(equivariant_decoder, self).__init__()

        self.encoding_irreps_out = e3nn.o3.Irreps(f"{config.embedding_depth // 3:.0f}x1o")
        self.decoding_radial_irreps = e3nn.o3.Irreps(f"{config.embedding_depth // 3} x0e")
        self.decoding_irreps_out = e3nn.o3.Irreps(f"{config.num_decoder_points:.0f} x1o + {config.num_decoder_points * (self.num_classes + 1):.0f} x0e")

        self.tp2 = e3nn.o3.FullyConnectedTensorProduct(
            irreps_in1=self.decoding_radial_irreps,
            irreps_in2=self.encoding_irreps_out,
            irreps_out=self.decoding_irreps_out,
            internal_weights=True,
        )

        self.decoding_radial = MLP(layers=config.num_decoder_layers,
                                   filters=config.embedding_depth,
                                   input_dim=self.encoding_irreps_out.num_irreps,
                                   output_dim=self.decoding_radial_irreps.num_irreps,
                                   norm=config.decoder_norm_mode,
                                   dropout=config.decoder_dropout_probability,
                                   activation=config.activation)

    def forward(self, encoding, num_graphs):
        r_dec = self.decoding_radial(
            torch.linalg.norm(encoding.reshape(num_graphs, self.encoding_irreps_out.num_irreps, 3), dim=-1)  # to invariant representation
        )

        decoding = self.tp2(r_dec, encoding)

        return decoding


class equivariant_encoder(nn.Module):
    def __init__(self, config):
        super(equivariant_encoder, self).__init__()

        sh_order = config.sh_order
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(sh_order)
        self.encoding_irreps_out = e3nn.o3.Irreps(f"{config.embedding_depth // 3:.0f}x1o")
        self.encoding_radial_irreps = e3nn.o3.Irreps(f"{config.encoder_radial_depth:.0f} x0e")

        if config.radial_function == 'bessel':
            self.rbf = BesselBasisLayer(config.num_radial, 1, 5)
        elif config.radial_function == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=1, num_gaussians=config.num_radial)

        self.tp1 = e3nn.o3.FullyConnectedTensorProduct(
            irreps_in1=self.encoding_radial_irreps,
            irreps_in2=self.irreps_sh,
            irreps_out=self.encoding_irreps_out,
            internal_weights=True,
        )

        self.rbf_emb = MLP(layers=1, filters=config.encoder_radial_depth,
                           input_dim=config.num_radial,
                           output_dim=config.encoder_radial_depth,
                           norm=config.graph_node_norm,
                           dropout=config.graph_node_dropout,
                           activation=config.activation)
        self.node_emb = nn.Embedding(100, config.atom_type_embedding_dims)
        self.radial_message = MLP(layers=config.nodewise_fc_layers,
                                  filters=config.encoder_radial_depth,
                                  input_dim=config.encoder_radial_depth + config.atom_type_embedding_dims,
                                  output_dim=config.encoder_radial_depth,
                                  norm=config.graph_node_norm,
                                  dropout=config.graph_node_dropout,
                                  activation=config.activation)

    def forward(self, data):

        pos = data.pos
        z = data.x[:, 0].long()

        centroid = torch.zeros((1, 3), dtype=torch.float32, device=pos.device)

        e_x = e3nn.o3.spherical_harmonics(
            l=self.irreps_sh,
            x=pos - centroid,
            normalize=False,
            normalization='component'
        )

        dist = torch.linalg.norm(pos, dim=1)
        rbf = self.rbf(dist)
        r_emb = self.rbf_emb(rbf)
        node_emb = self.node_emb(z)
        radial_message = self.radial_message(torch.cat([r_emb, node_emb], dim=-1))

        c_emb = self.tp1(
            radial_message,
            e_x)

        # graph aggregation by sum and mean
        encoding = (scatter(c_emb, data.batch, dim_size=data.num_graphs, dim=0, reduce='mean') +
                    scatter(c_emb, data.batch, dim_size=data.num_graphs, dim=0, reduce='sum'))

        return encoding


class point_encoder(nn.Module):
    def __init__(self, seed, config):
        super(point_encoder, self).__init__()
        self.model = molecule_graph_model(
            num_atom_feats=1,
            num_mol_feats=0,
            output_dimension=config.embedding_depth,
            seed=seed,
            graph_convolution_type='TransformerConv',
            graph_aggregator=config.graph_aggregator,
            concat_pos_to_atom_features=True,
            concat_mol_to_atom_features=False,
            concat_crystal_to_atom_features=False,
            activation='gelu',
            num_fc_layers=0,
            fc_depth=0,
            fc_norm_mode=None,
            fc_dropout_probability=None,
            graph_node_norm=config.graph_node_norm,
            graph_node_dropout=config.graph_node_dropout,
            graph_message_norm=None,
            graph_message_dropout=config.graph_message_dropout,
            num_attention_heads=config.num_attention_heads,
            graph_message_depth=config.embedding_depth,
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
        )

    def forward(self, data):
        return self.model(data)

import torch
from e3nn import o3 as o3, nn as enn
from torch import nn as nn
from torch.nn import functional as F
from torch_scatter import scatter

from mxtaltools.models.basis_functions import BesselBasisLayer, GaussianEmbedding
from mxtaltools.models.components import MLP


class equivariant_decoder(nn.Module):
    def __init__(self, config):
        super(equivariant_decoder, self).__init__()

        self.encoding_irreps_out = o3.Irreps(f"{config.embedding_depth // 3:.0f}x1o")
        self.decoding_radial_irreps = o3.Irreps(f"{config.embedding_depth // 3} x0e")
        self.decoding_irreps_out = o3.Irreps(f"{config.num_decoder_points:.0f} x1o + {config.num_decoder_points * (self.num_classes + 1):.0f} x0e")

        self.tp2 = o3.FullyConnectedTensorProduct(
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
        self.irreps_sh = o3.Irreps.spherical_harmonics(sh_order)
        self.irreps_mid = o3.Irreps('64x0e + 64x1e + 64x1o + 32x2e + 32x2o')
        self.encoding_irreps_out = o3.Irreps(f"{config.embedding_depth // 3:.0f}x1o")
        self.encoding_radial_irreps = o3.Irreps(f"{config.encoder_radial_depth:.0f} x0e")

        '''initial node embedding'''
        self.node_emb = nn.Embedding(100, config.atom_type_embedding_dims)

        '''edge embedding'''
        if config.radial_function == 'bessel':
            self.rbf = BesselBasisLayer(config.num_radial, 1, 5)
        elif config.radial_function == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=1, num_gaussians=config.num_radial)
        self.rbf_emb = MLP(layers=1,
                           filters=config.encoder_radial_depth,
                           input_dim=config.num_radial,
                           output_dim=config.encoder_radial_depth,
                           norm=config.graph_node_norm,
                           dropout=config.graph_node_dropout,
                           activation=config.activation)
        self.radial_message = MLP(layers=config.nodewise_fc_layers,
                                  filters=config.encoder_radial_depth,
                                  input_dim=config.encoder_radial_depth + config.atom_type_embedding_dims,
                                  output_dim=config.encoder_radial_depth,
                                  norm=config.graph_node_norm,
                                  dropout=config.graph_node_dropout,
                                  activation=config.activation)

        '''equivariant aggregation op'''
        self.tp1 = o3.FullyConnectedTensorProduct(
            irreps_in1=self.encoding_radial_irreps,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_mid,
            internal_weights=True,
        )

        self.act1 = enn.Activation(irreps_in=self.irreps_mid,
                                   acts=[F.gelu, None, None, None, None])

        self.tl1 = o3.Linear(irreps_in=self.irreps_mid,
                             irreps_out=self.irreps_mid,
                             internal_weights=True)

        self.act2 = enn.Activation(irreps_in=self.irreps_mid,
                                   acts=[F.gelu, None, None, None, None])

        self.tl2 = o3.Linear(irreps_in=self.irreps_mid,
                             irreps_out=self.encoding_irreps_out,
                             internal_weights=True)

    def forward(self, data):

        pos = data.pos
        z = data.x[:, 0].long()

        centroid = torch.zeros((1, 3), dtype=torch.float32, device=pos.device)

        e_x = o3.spherical_harmonics(
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

        c_emb = self.act1(self.tp1(
            radial_message, e_x))

        # graph aggregation by sum and mean
        c_emb = (scatter(c_emb, data.batch, dim_size=data.num_graphs, dim=0, reduce='mean') +
                 scatter(c_emb, data.batch, dim_size=data.num_graphs, dim=0, reduce='sum'))

        c_emb = self.act2(self.tl1(c_emb))

        encoding = self.tl2(c_emb)

        return encoding



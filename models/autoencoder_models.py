import torch.nn as nn
from torch import nn as nn

from models.base_models import molecule_graph_model
from models.components import MLP

import torch


class point_autoencoder(nn.Module):
    def __init__(self, seed, config, dataDims):
        super(point_autoencoder, self).__init__()
        '''conditioning model'''
        cartesian_dimension = 3
        self.num_classes = dataDims['num_atom_types']
        self.output_depth = self.num_classes + cartesian_dimension + 1
        self.num_nodes = config.num_decoder_points

        self.equivariant_encoder = config.encoder_type == 'equivariant'
        self.equivariant_decoder = config.decoder_type == 'equivariant'
        self.fully_equivariant = self.equivariant_encoder and self.equivariant_decoder
        self.variational = config.variational_encoder
        self.bottleneck_dim = config.bottleneck_dim

        self.encoder = point_encoder(seed, config)

        self.decoder = MLP(
            layers=config.num_decoder_layers,  # todo deprecate non equivariant encoder I/O
            filters=config.embedding_depth if self.equivariant_decoder else config.embedding_depth * 3,
            input_dim=config.bottleneck_dim if self.equivariant_encoder else config.bottleneck_dim * 3,
            output_dim=(self.output_depth - 3 if self.equivariant_decoder else 0) * self.num_nodes,
            conditioning_dim=0,
            activation='gelu',
            norm=config.decoder_norm_mode,
            dropout=config.decoder_dropout_probability,
            equivariant=config.decoder_type == 'equivariant',
            vector_output_dim=self.num_nodes,
            vector_norm=config.decoder_vector_norm,
            ramp_depth=config.decoder_ramp_depth,
        )

    def forward(self, data):
        encoding = self.encode(data)

        return self.decode(encoding)

    def encode(self, data, z=None):
        """
        pass only the encoding
        """
        encoding = self.encoder(data)
        if self.equivariant_encoder:
            x, v = encoding  # extract vector component
        else:
            x = encoding

        if self.variational:  # this appears to break equivariance but it's only because the lengths are changing
            assert self.equivariant_encoder, "Variational autoencoder only implemented for equivariant encoder"
            mu = torch.linalg.norm(v, dim=1)
            log_sigma = x
            sigma = torch.exp(0.5 * log_sigma)
            if z is None:
                z = torch.randn_like(sigma)
            stochastic_weight = z*sigma + mu  # parameterized distribution
            encoding = stochastic_weight[:, None, :] * v  # rescale vector length by learned distribution
            self.kld = (sigma ** 2 + mu ** 2 - log_sigma - 0.5)  # KL divergence of embedded distribution
        else:
            if self.equivariant_encoder:
                encoding = v
            else:
                encoding = x

        assert torch.sum(torch.isnan(encoding)) == 0, "NaN in encoder output"

        return encoding

    '''
    encoder equivariance test
    from scipy.spatial.transform import Rotation as R

    rmat = torch.tensor(R.random().as_matrix(), device=data.x.device, dtype=torch.float32)
    
    d1 = data.clone()
    encoding = self.encoder(d1)
    rotpos = torch.einsum('ij, nj->ni', rmat, d1.pos)
    rotencoding = torch.einsum('ij, njk->nik', rmat, encoding)
    d2 = d1.clone()
    d2.pos = rotpos
    
    encoding2 = self.encoder(d2)
    
    print(torch.mean(torch.abs(encoding2 - rotencoding)))
    print(torch.amax(torch.abs(encoding2 - rotencoding)))
    '''

    def decode(self, encoding):
        if self.decoder.equivariant:
            '''encoding nx3xk'''
            norm = torch.linalg.norm(encoding, dim=1)
            decoding = self.decoder(x=norm,
                                    v=encoding)

            scalar_decoding, vector_decoding = decoding
            '''combine vector and scalar features to n*nodes x m'''
            decoding = torch.cat([
                vector_decoding.permute(0, 2, 1).reshape(len(vector_decoding) * self.num_nodes, 3),
                scalar_decoding.reshape(len(scalar_decoding) * self.num_nodes, self.output_depth - 3)],
                dim=-1)
        else:
            '''encoding nxk'''
            decoding = self.decoder(encoding)
            '''decoding n x nodes*m to n*nodes x m'''
            decoding = decoding.reshape(self.num_nodes * len(encoding), self.output_depth)

        assert torch.sum(torch.isnan(decoding)) == 0, "NaN in decoder output"

        return decoding

    '''
    >>> vector encoding equivariance test
    
    
    
    >>> vector decoding equivariance test
    from scipy.spatial.transform import Rotation as R
    
    rmat = torch.tensor(R.random().as_matrix(),device=encoding.device, dtype=torch.float32)
    
    v1 = encoding.clone()
    rotv1 = torch.einsum('ij, nkj->nki', rmat, v1)
    
    decoding = self.decoder(x=torch.linalg.norm(v1, dim=-1),
                            v=v1.permute(0,2,1))
    y1 = decoding[1]
    
    decoding = self.decoder(x=torch.linalg.norm(rotv1, dim=-1),
                            v=rotv1.permute(0,2,1))
    y2 = decoding[1]
    
    roty1 = torch.einsum('ij, njk->nik', rmat, y1)
    
    print(torch.mean(torch.abs(y2 - roty1)))
    
    >>> scalar decoding invariance test
    from scipy.spatial.transform import Rotation as R
    
    rmat = torch.tensor(R.random().as_matrix(),device=encoding.device, dtype=torch.float32)
    
    v1 = encoding.clone()
    rotv1 = torch.einsum('ij, nkj->nki', rmat, v1)
    
    decoding = self.decoder(x=torch.linalg.norm(v1, dim=-1),
                            v=v1.permute(0,2,1))
    y1 = decoding[0]
    
    decoding = self.decoder(x=torch.linalg.norm(rotv1, dim=-1),
                            v=rotv1.permute(0,2,1))
    y2 = decoding[0]
    torch.mean(torch.abs(y1-y2))
    
    >>> including reshaping and recombination
    from scipy.spatial.transform import Rotation as R
    
    rmat = torch.tensor(R.random().as_matrix(), device=encoding.device, dtype=torch.float32)
    
    v1 = encoding.clone()
    rotv1 = torch.einsum('ij, nkj->nki', rmat, v1)
    
    decoding = self.decoder(x=torch.linalg.norm(v1, dim=-1),
                            v=v1.permute(0, 2, 1))
    
    scalar_decoding, vector_decoding = decoding
    vector_decoding = vector_decoding.permute(0,2,1).reshape(len(vector_decoding) * self.num_nodes, 3)
    scalar_decoding = scalar_decoding.reshape(len(scalar_decoding) * self.num_nodes, self.output_depth - 3)
    y1 = torch.cat([vector_decoding, scalar_decoding], dim=-1)[:, :3]
    
    decoding = self.decoder(x=torch.linalg.norm(rotv1, dim=-1),
                            v=rotv1.permute(0, 2, 1))
    
    scalar_decoding, vector_decoding = decoding
    vector_decoding = vector_decoding.permute(0,2,1).reshape(len(vector_decoding) * self.num_nodes, 3)
    scalar_decoding = scalar_decoding.reshape(len(scalar_decoding) * self.num_nodes, self.output_depth - 3)
    y2 = torch.cat([vector_decoding, scalar_decoding], dim=-1)[:, :3]
    
    roty1 = torch.einsum('ij, nj->ni', rmat, y1)
    
    print(torch.mean(torch.abs(roty1 - y2))) 
    '''


class point_encoder(nn.Module):
    def __init__(self, seed, config):
        super(point_encoder, self).__init__()
        self.model = molecule_graph_model(
            num_atom_feats=1,
            num_mol_feats=0,
            output_dimension=config.bottleneck_dim,
            seed=seed,
            equivariant_graph=True if config.encoder_type == 'equivariant' else False,
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


"""
>>> cross product equivariance test
from scipy.spatial.transform import Rotation as R

rmat = torch.tensor(R.random().as_matrix(), device='cpu', dtype=torch.float32)

v1 = torch.randn(10,3)
encoding = torch.cross(v1[:5],v1[5:])
v2 = torch.einsum('ij, nj->ni', rmat, v1)
rotencoding = torch.einsum('ij, nj->ni', rmat, encoding)
encoding2 = torch.cross(v2[:5],v2[5:])

print(torch.mean(torch.abs(encoding2 - rotencoding)))
print(torch.amax(torch.abs(encoding2 - rotencoding)))

rmat = -rmat

v1 = torch.randn(10,3)
encoding = torch.cross(v1[:5],v1[5:])
v2 = torch.einsum('ij, nj->ni', rmat, v1)
rotencoding = torch.einsum('ij, nj->ni', rmat, encoding)
encoding2 = torch.cross(v2[:5],v2[5:])

print(torch.mean(torch.abs(encoding2 - rotencoding)))
print(torch.amax(torch.abs(encoding2 - rotencoding)))

"""

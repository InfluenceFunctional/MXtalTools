import torch
from torch import nn as nn
from torch_geometric import nn as gnn
from torch_scatter import scatter

from models.components import Normalization, MLP
from models.equivariant_TransformerConv import EquiVTransformerConv


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, node_embedding_depth, num_input_classes, num_scalar_input_features, type_embedding_dimension):
        super(EmbeddingBlock, self).__init__()

        self.embeddings = nn.Embedding(num_input_classes + 1, type_embedding_dimension)
        self.linear = nn.Linear(type_embedding_dimension + num_scalar_input_features - 1, node_embedding_depth)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        embedding = self.embeddings(x[:, 0].long())  # always embed the first dimension only (by convention, atomic number)

        concat_vec = torch.cat([embedding, x[:, 1:]], dim=-1)

        return self.linear(concat_vec)


class GC_Block(torch.nn.Module):
    def __init__(self,
                 message_depth,
                 node_embedding_depth,
                 radial_dim,
                 dropout=0,
                 heads=1,
                 equivariant=False):
        super(GC_Block, self).__init__()
        self.embed_edge = nn.Linear(radial_dim, radial_dim, bias=False)
        self.equivariant = equivariant
        if equivariant:
            self.V_GConv = EquiVTransformerConv(
                in_channels=message_depth,
                out_channels=message_depth // heads,
                heads=heads,
                edge_dim=radial_dim,
            )
            self.vec_to_message = nn.Linear(node_embedding_depth, message_depth, bias=False)
            self.message_to_vec = nn.Linear(message_depth, node_embedding_depth, bias=False)

        self.node_to_message = nn.Linear(node_embedding_depth, message_depth, bias=False)
        self.message_to_node = nn.Linear(message_depth, node_embedding_depth, bias=False)  # don't want to send spurious messages, though it probably doesn't matter anyway
        assert message_depth % heads == 0
        self.GConv = gnn.TransformerConv(
            in_channels=message_depth,
            out_channels=message_depth // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=radial_dim,
            beta=True,
        )

    def embed_edge_attrs(self, edge_attr):
        return self.embed_edge(edge_attr)

    def forward(self, x, v, edge_attr, edge_index, batch):
        # convolve
        edge_embedding = self.embed_edge_attrs(edge_attr)
        x, (_, alpha) = self.GConv(
            self.node_to_message(x), edge_index, edge_embedding,
            return_attention_weights=True)

        if self.equivariant:
            v = self.V_GConv(self.vec_to_message(v), alpha, edge_index, edge_embedding)

        if v is not None:
            return self.message_to_node(x), self.message_to_vec(v)
        else:
            # reshape to node dimension
            return self.message_to_node(x)


'''
equivariance test 

> linear layer

from scipy.spatial.transform import Rotation as R
rmat = torch.tensor(R.random().as_matrix(),device=x.device, dtype=torch.float32)
embedding = self.vec_to_message(v).permute(0,2,1)
rotv = torch.einsum('ij, nkj -> nki', rmat, v.permute(0,2,1)).permute(0,2,1)
rotembedding = torch.einsum('ij, nkj -> nki', rmat, embedding)

rotembedding2 = self.vec_to_message(rotv).permute(0,2,1)
print(torch.mean(torch.abs(rotembedding - rotembedding2))/torch.mean(torch.abs(rotembedding)))


> graph convolution
from scipy.spatial.transform import Rotation as R
rmat = torch.tensor(R.random().as_matrix(),device=x.device, dtype=torch.float32)
embedding = self.V_GConv(self.vec_to_message(v), alpha, edge_index, edge_embedding)
rotv = torch.einsum('ij, njk -> nik', rmat, v)
rotembedding = torch.einsum('ij, njk -> nik', rmat, embedding)

rotembedding2 = self.V_GConv(self.vec_to_message(rotv), alpha, edge_index, edge_embedding)
print(torch.mean(torch.abs(rotembedding - rotembedding2))/torch.mean(torch.abs(rotembedding)))
'''


# TODO deprecate
# class EquivariantMessagePassing(nn.Module):
#     def __init__(self, irreps, num_edge_attr):
#         super(EquivariantMessagePassing, self).__init__()
#         self.convolution = o3.FullyConnectedTensorProduct(
#             irreps_in1=irreps,
#             irreps_in2=o3.Irreps(f'{num_edge_attr}x0e'),
#             irreps_out=irreps,
#             internal_weights=True,
#         )
#
#     def forward(self, x, edge_index, edge_attrs):
#         messages = self.convolution(x[edge_index[0]], edge_attrs)
#
#         return scatter(messages, edge_index[1], dim=0, dim_size=len(x), reduce='mean')  # simple mean aggregation - radial attention may be possible


class MPConv(torch.nn.Module):  # todo refactor and add a transformer version
    def __init__(self, in_channels, out_channels, edge_dim, dropout=0, norm=None, activation='leaky relu'):
        super(MPConv, self).__init__()

        self.MLP = MLP(layers=4,
                       filters=out_channels,
                       input_dim=in_channels * 2 + edge_dim,
                       dropout=dropout,
                       norm=norm,
                       output_dim=out_channels,
                       activation=activation,
                       )

    def forward(self, x, edge_index, edge_attr):
        m = self.MLP(torch.cat((x[edge_index[0]], x[edge_index[1]], edge_attr), dim=-1))

        return scatter(m, edge_index[1], dim=0, dim_size=len(x))  # send directional messages from i to j, enforcing the size of the output dimension


class FC_Block(torch.nn.Module):
    def __init__(self,
                 nodewise_fc_layers,
                 node_embedding_depth,
                 activation,
                 nodewise_norm,
                 nodewise_dropout,
                 equivariant=False,
                 vector_norm=False,
                 ):
        super(FC_Block, self).__init__()
        self.equivariant = equivariant

        self.model = MLP(layers=nodewise_fc_layers,
                         filters=node_embedding_depth,
                         input_dim=node_embedding_depth,
                         output_dim=node_embedding_depth,
                         conditioning_dim=node_embedding_depth if equivariant else 0,
                         activation=activation,
                         norm=nodewise_norm,
                         dropout=nodewise_dropout,
                         equivariant=equivariant,
                         vector_norm=vector_norm)

    def forward(self, x, v=None, return_latent=False, batch=None):
        return self.model(x,
                          v=v,
                          conditions=torch.linalg.norm(v, dim=1) if v is not None else None,
                          return_latent=return_latent,
                          batch=batch)


'''
equivariance test

>>> FC block
from scipy.spatial.transform import Rotation as R
rmat = torch.tensor(R.random().as_matrix(),device=x.device, dtype=torch.float32)
_, embedding = self.model(x,
                          v=v,
                          conditions=torch.linalg.norm(v, dim=1) if v is not None else None,
                          return_latent=return_latent,
                          batch=batch)
rotv = torch.einsum('ij, njk -> nik', rmat, v)
rotembedding = torch.einsum('ij, njk -> nik', rmat, embedding)

_, rotembedding2 = self.model(x,
                          v=rotv,
                          conditions=torch.linalg.norm(v, dim=1) if v is not None else None,
                          return_latent=return_latent,
                          batch=batch)
print(torch.mean(torch.abs(rotembedding - rotembedding2))/torch.mean(torch.abs(rotembedding)))
'''

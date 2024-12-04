import torch
from torch import nn as nn

from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.graph_neural_network import VectorGNN
from mxtaltools.models.graph_models.molecule_graph_model import VectorMoleculeGraphModel
from mxtaltools.models.modules.components import Scalarizer, vectorMLP
from mxtaltools.models.utils import collate_decoded_data, ae_reconstruction_loss
from mxtaltools.reporting.ae_reporting import swarm_vs_tgt_fig


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
        self.decoder_type = config.decoder.model_type
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
        if self.decoder_type == 'mlp':
            self.decoder = Mo3ENetDecoder(seed,
                                          config.decoder,
                                          config.bottleneck_dim,
                                          self.output_depth, self.num_decoder_nodes)
        elif self.decoder_type == 'gnn':
            self.decoder = Mo3ENetGraphDecoder(config.decoder,
                                               config.bottleneck_dim,
                                               self.output_depth,
                                               self.num_decoder_nodes,
                                               )
        else:
            assert False, "Unknown decoder type" + str(self.decoder_type)
        self.scalarizer = Scalarizer(config.bottleneck_dim,
                                     self.cartesian_dimension,
                                     None, None, 0)

    def forward(self,
                data: CrystalData,
                return_latent: bool = False,
                return_dists: bool = False,
                ):
        encoding = self.encode(data)
        if torch.sum(torch.isnan(encoding)) != 0:
            assert False, "NaN values in encoding"
        decoding = self.decode(encoding)
        if torch.sum(torch.isnan(decoding)) != 0:
            assert False, "NaN values in decoding"
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
        s = self.scalarizer(encoding)
        if torch.sum(torch.isnan(s)) > 0:
            assert False, "NaN values in scalarized encoding"
        scalar_decoding, vector_decoding = self.decoder(s, v=encoding)

        '''combine vector and scalar features to n*nodes x m'''
        # de-normalize predicted node positions and rearrange to correct format
        # from n_graphs, x (num_nodes * scalar feats), v (num_nodes * scalar_feats)
        if self.decoder_type == 'mlp':
            decoding = torch.cat([
                vector_decoding.permute(0, 2, 1).reshape(len(vector_decoding) * self.num_decoder_nodes,
                                                         3) * self.radial_normalization,
                scalar_decoding.reshape(len(scalar_decoding) * self.num_decoder_nodes, self.output_depth - 3)],
                dim=-1)
        elif self.decoder_type == 'gnn':
            decoding = torch.cat(
                [
                    vector_decoding[:, :, 0] * self.radial_normalization,
                    scalar_decoding
                ],
                dim=1
            )
        else:
            assert False, "Unknown decoder type" + str(self.decoder_type)

        return decoding

    ''' equivariance testing
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    
    v = encoding.clone()
    
    'initialize rotations'
    rotations = torch.tensor(
        R.random(len(v)).as_matrix() *
        np.random.choice((-1, 1), replace=True, size=len(v))[:, None, None],
        dtype=torch.float,
        device=v.device)
    'rotate input'
    r_v = torch.einsum('ij, njk -> nik', rotations[0], v)
    
    'get output'
    s1, out1 = self.decoder(self.scalarizer(v), v=v)
    s2, out2 = self.decoder(self.scalarizer(r_v), v=r_v)
    
    'rotated output'
    r_out1 = torch.einsum('ij, njk -> nik', rotations[0], out1)
    
    print(torch.mean(torch.abs(r_out1 - out2) / out2.abs()))
    print(torch.mean(torch.abs(s1 - s2) / s2.abs()))

    '''

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

        (nodewise_reconstruction_loss,  # todo adjust with new losses
         nodewise_type_loss,
         reconstruction_loss,
         self_likelihoods,
         ) = ae_reconstruction_loss(data,
                                    decoded_data,
                                    nodewise_weights,
                                    nodewise_weights_tensor,
                                    num_atom_types,
                                    type_distance_scaling,
                                    sigma)

        rmsds = torch.zeros(data.num_graphs)
        max_dists = torch.zeros_like(rmsds)
        tot_overlaps = torch.zeros_like(rmsds)
        match_successful = torch.zeros_like(rmsds)
        # for ind in range(data.num_graphs):
        #     rmsds[ind], max_dists[ind], tot_overlaps[ind], match_successful[ind], fig2 = scaffolded_decoder_clustering(
        #         ind,
        #         data,
        #         decoded_data,
        #         num_atom_types,
        #         return_fig=True)
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


class Mo3ENetGraphDecoder(nn.Module):
    def __init__(self, config, bottleneck_dim, output_depth, num_nodes):
        super(Mo3ENetGraphDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = config.fc.hidden_dim
        self.model = VectorGNN(
            input_node_dim=config.fc.hidden_dim,
            node_dim=config.fc.hidden_dim,
            fcs_per_gc=1,
            message_dim=config.fc.hidden_dim // 4,
            embedding_dim=output_depth - 3,
            num_convs=config.fc.num_layers,
            num_radial=32,
            num_input_classes=101,
            cutoff=2,
            max_num_neighbors=32,
            envelope_exponent=5,
            activation='gelu',
            atom_type_embedding_dim=0,
            norm=('graph ' + config.fc.norm) if config.fc.norm is not None else None,
            vector_norm=('graph ' + config.fc.vector_norm) if config.fc.vector_norm is not None else None,
            dropout=config.fc.dropout,
            radial_embedding='gaussian',
            override_cutoff=None,
            v_embedding_dim=1,
            v_input_node_dim=config.fc.hidden_dim,
        )
        self.s_to_nodes = nn.Linear(bottleneck_dim, config.fc.hidden_dim * num_nodes, bias=False)
        self.v_to_nodes = nn.Linear(bottleneck_dim, config.fc.hidden_dim * num_nodes, bias=False)
        self.v_to_pos = nn.Linear(bottleneck_dim, num_nodes, bias=False)

    def forward(self, x, v):
        num_graphs = len(x)

        # all combinations of edges within each graph
        edges = []
        edges_i = torch.combinations(torch.arange(self.num_nodes), r=2, with_replacement=False).to(x.device)

        for ind in range(num_graphs):
            batch_ind = ind * self.num_nodes
            edges.append(
                batch_ind + torch.cat([edges_i, torch.fliplr(edges_i)], dim=0)
            )
        batch = torch.arange(num_graphs, device=x.device).repeat_interleave(self.num_nodes)
        edges = torch.cat(edges, dim=0)
        edges_dict = {'edge_index': edges.T}

        x = self.s_to_nodes(x).reshape(num_graphs * self.num_nodes, self.hidden_dim)
        directions = self.v_to_pos(v).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, 3, 1)[..., 0]
        pos = directions / (1e-4 + torch.linalg.norm(directions, dim=1))[:, None]
        v = self.v_to_nodes(v).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, self.hidden_dim, 3).permute(0, 2, 1)

        return self.model(x, v, pos, batch, edges_dict)

    ''' equivariance test
    def v_to_node(v, num_graphs):
        v2 = self.v_to_nodes(v).reshape(num_graphs, 3, self.hidden_dim, self.num_nodes)
        v2 = v2.permute(0, 3, 1, 2).flatten(0, 1)
        return v2
    
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    
    'initialize rotations'
    rotations = torch.tensor(
        R.random(num_graphs).as_matrix() *
        np.random.choice((-1, 1), replace=True, size=num_graphs)[:, None, None],
        dtype=torch.float,
        device=x.device)
    
    'rotate input'
    r_v = torch.einsum('ij, njk -> nik', rotations[0], v)
    
    'get output'
    out1 = v_to_node(v, num_graphs)
    out2 = v_to_node(r_v, num_graphs)
    
    'rotated output'
    r_out1 = torch.einsum('ij, njk -> nik', rotations[0], out1)
    
    print(torch.mean(torch.abs(r_out1 - out2)/out2.abs()))

    import plotly.graph_objects as go
    
    fig = go.Figure(go.Histogram(x=((out2 - r_out1)/out2.abs()).flatten().abs().log10().cpu().detach().numpy(), nbinsx=100)).show()

    def v_to_node(v, num_graphs):
    v2 = self.v_to_nodes(v).reshape(num_graphs, 3, self.hidden_dim, self.num_nodes)
    v2 = v2.permute(0, 3, 1, 2).flatten(0, 1)
    return v2
    
    # ---- graph model --- 
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    
    'initialize rotations'
    rotations = torch.tensor(
        R.random(num_graphs).as_matrix() *
        np.random.choice((-1, 1), replace=True, size=num_graphs)[:, None, None],
        dtype=torch.float,
        device=x.device)
    
    'rotate input'
    r_v = torch.einsum('ij, njk -> nik', rotations[0], v)
    r_pos = torch.einsum('ij, nj -> ni', rotations[0], pos)
    
    'get output'
    s1, out1 = self.model(x, v, pos, batch, edges_dict)
    s2, out2 = self.model(x, r_v, r_pos, batch, edges_dict)
    
    'rotated output'
    r_out1 = torch.einsum('ij, njk -> nik', rotations[0], out1)
    
    print(torch.mean(torch.abs(r_out1 - out2)/out2.abs()))

    # final equivariance test
    if not hasattr(self, 'v0'):
    self.x0 = x.clone()
    self.v0 = v.clone()

from scipy.spatial.transform import Rotation as R
import numpy as np

num_graphs = len(x)

# all combinations of edges within each graph
edges = []
edges_i = torch.combinations(torch.arange(self.num_nodes), r=2, with_replacement=False).to(x.device)

for ind in range(num_graphs):
    batch_ind = ind * self.num_nodes
    edges.append(
        batch_ind + torch.cat([edges_i, torch.fliplr(edges_i)], dim=0)
    )
edges = torch.cat(edges, dim=0)
edges_dict = {'edge_index': edges.T}
batch = torch.arange(num_graphs, device=x.device).repeat_interleave(self.num_nodes)

'initialize rotations'
rotations = torch.tensor(
    R.random(num_graphs).as_matrix() *
    np.random.choice((-1, 1), replace=True, size=num_graphs)[:, None, None],
    dtype=torch.float,
    device=x.device)

x = self.x0.clone()
v = self.v0.clone()

def rotate_object(rotations, thing, batch, num_graphs):
    return torch.cat(
    [torch.einsum('ij, njk->nik', rotations[ind], thing[batch == ind])
     for ind in range(num_graphs)])

rv = rotate_object(rotations, v, torch.arange(num_graphs, device=x.device), num_graphs)

xf = self.s_to_nodes(x).reshape(num_graphs * self.num_nodes, self.hidden_dim)
pos = self.v_to_pos(v).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, 3, 1)[..., 0]
vf = self.v_to_nodes(v).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, self.hidden_dim, 3).permute(0, 2, 1)
rpos = self.v_to_pos(rv).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, 3, 1)[..., 0]

posr = rotate_object(rotations, pos[:, :, None], batch, num_graphs)[..., 0]

vfr = rotate_object(rotations, vf, batch, num_graphs)
rvf = self.v_to_nodes(rv).permute(0, 2, 1).reshape(num_graphs * self.num_nodes, self.hidden_dim, 3).permute(0, 2, 1)

xo, yo = self.model(xf, vf, pos, batch, edges_dict)
rxo, ryo = self.model(xf, rvf, rpos, batch, edges_dict)

yor = rotate_object(rotations, yo, batch, num_graphs)

print(((vfr-rvf).abs()/rvf.abs()).mean())
print(((yor-ryo).abs()/ryo.abs()).mean())
print(((rpos-posr).abs()/rpos.abs()).mean())
    '''


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

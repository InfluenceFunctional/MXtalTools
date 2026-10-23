"""Mo3ENet, a molecular point-cloud autoencoder with scalar and Cartesian-vector channels,
and its encoder, MLP decoder and graph decoder modules."""
import torch
from torch import nn as nn
import torch.nn.functional as F

from mxtaltools.models.autoencoder_utils import decoding2mol_batch, ae_reconstruction_loss, batch_rmsd
from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.graph_neural_network import VectorGNN
from mxtaltools.models.graph_models.molecule_graph_model import VectorMoleculeGraphModel
from mxtaltools.models.modules.components import Scalarizer, vectorMLP
from mxtaltools.reporting.ae_reporting import swarm_vs_tgt_fig


# noinspection PyAttributeOutsideInit
class Mo3ENet(BaseGraphModel):
    """Molecular point-cloud autoencoder (Mo3ENet).

    :class:`Mo3ENetEncoder` maps a batch of molecules to a vector encoding
    [num_graphs, 3, bottleneck_dim]. A :class:`Scalarizer` derives scalars
    [num_graphs, bottleneck_dim] from it, and the decoder (:class:`Mo3ENetDecoder` for
    ``decoder_type == 'mlp'``, :class:`Mo3ENetGraphDecoder` for ``'gnn'``) maps both to
    ``num_decoder_nodes`` points per graph, each with a position, ``num_atom_types`` type
    logits and a weight logit (see :meth:`decode`).

    Parameters
    ----------
    seed : int
        Passed to ``torch.manual_seed`` at construction, to the encoder and to the MLP
        decoder.
    config : Namespace
        Reads ``bottleneck_dim``, ``decoder.num_nodes``, ``decoder.model_type``
        (``'mlp'`` when absent) and ``encoder.graph.cutoff``; the ``encoder`` and
        ``decoder`` sections are passed to the submodules.
    num_atom_types : int
        Number of type classes; ``output_depth = num_atom_types + 4``.
    atom_embedding_vector : torch.Tensor
        Registered as a buffer. :meth:`check_embedding_quality` indexes it by atomic
        number to obtain type class indices.
    radial_normalization : float
        Input positions are divided by it before encoding and decoded positions are
        multiplied by it. Stored as a float32 buffer.
    protons_in_input : bool
        Stored as a bool buffer; not read in this module.

    Notes
    -----
    Further buffers: ``inferring_protons`` (False; not read in this module) and
    ``convolution_cutoff = config.encoder.graph.cutoff / radial_normalization``, passed to
    the encoder as ``override_cutoff``.

    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator. An unknown ``decoder.model_type`` raises AssertionError.
    """
    def __init__(self,
                 seed,
                 config,
                 num_atom_types: int,
                 atom_embedding_vector: torch.Tensor,
                 radial_normalization: float,
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
        if not hasattr(config.decoder, 'model_type'):
            self.decoder_type = 'mlp'  # old model
        else:
            self.decoder_type = config.decoder.model_type
        # todo add type distance scaling and num atom types and node weight temperature
        self.register_buffer('atom_embedding_vector', atom_embedding_vector)
        self.register_buffer('radial_normalization', torch.tensor(radial_normalization, dtype=torch.float32))
        self.register_buffer('protons_in_input', torch.tensor(protons_in_input, dtype=torch.bool))
        self.register_buffer('inferring_protons', torch.tensor(False, dtype=torch.bool))
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

    def forward(self, mol_batch,
                return_latent: bool = False,
                return_dists: bool = False,
                **kwargs):
        """Encode and decode a batch of molecules.

        Calls :meth:`encode` (with its centering check; ``mol_batch.pos`` is divided in
        place by ``radial_normalization``) and then :meth:`decode`, printing a message
        when the encoding or the decoding contains NaN.

        Parameters
        ----------
        mol_batch : MolData
            Molecules to encode; modified in place as above.
        return_latent : bool, optional
            Also return the encoding. Default False.
        return_dists : bool, optional
            Not used. Default False.
        ``**kwargs``
            Accepted and not used.

        Returns
        -------
        torch.Tensor or tuple
            ``decoding`` [num_graphs * num_decoder_nodes, output_depth] (see
            :meth:`decode`); with ``return_latent``, ``(decoding, encoding)`` where
            ``encoding`` is [num_graphs, 3, bottleneck_dim].
        """
        encoding = self.encode(mol_batch)
        if torch.sum(torch.isnan(encoding)) != 0:
            print("NaN values in encoding")
        decoding = self.decode(encoding)
        if torch.sum(torch.isnan(decoding)) != 0:
            print("NaN values in decoding")
        if return_latent:
            return decoding, encoding
        else:
            return decoding

    def encode(self,
               mol_batch,
               override_centering: bool = False):
        """Encode a batch of molecules into vector encodings.

        Divides ``mol_batch.pos`` in place by ``radial_normalization`` (the caller's batch
        is modified), then runs :class:`Mo3ENetEncoder` and returns its vector output.

        Parameters
        ----------
        mol_batch : MolData
            Molecules; ``z``, ``pos``, ``batch``, ``ptr`` and ``num_graphs`` are read.
        override_centering : bool, optional
            Skip the centering check. Default False.

        Returns
        -------
        torch.Tensor
            [num_graphs, 3, bottleneck_dim] vector encoding.

        Raises
        ------
        AssertionError
            Unless ``override_centering``, when the norm of the mean position over all
            nodes of the batch (not per graph) is not below 1e-3.
        """
        # normalize radii
        if not override_centering:
            assert torch.linalg.norm(mol_batch.pos.mean(0)) < 1e-3, "Encoder trained only for centered molecules!"
        mol_batch.pos /= self.radial_normalization
        _, encoding = self.encoder(mol_batch)

        return encoding

    def decode(self, encoding):
        """Decode vector encodings into per-graph point swarms.

        The decoder receives the :class:`Scalarizer` output [num_graphs, bottleneck_dim]
        as scalars and ``encoding`` as vectors. A message is printed when the scalarized
        encoding contains NaN.

        Parameters
        ----------
        encoding : torch.Tensor
            [num_graphs, 3, bottleneck_dim] vector encoding.

        Returns
        -------
        torch.Tensor
            [num_graphs * num_decoder_nodes, output_depth], rows grouped by graph:
            columns 0:3 are positions multiplied by ``radial_normalization``, columns
            ``3:3 + num_atom_types`` are type logits and the last column is a weight logit
            (see :func:`~mxtaltools.models.autoencoder_utils.decoding2mol_batch`).

        Raises
        ------
        AssertionError
            If ``decoder_type`` is neither ``'mlp'`` nor ``'gnn'``.
        """
        s = self.scalarizer(encoding)
        if torch.sum(torch.isnan(s)) > 0:
            print("NaN values in scalarized encoding")
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

    def compile_self(self, dynamic=True, fullgraph=False):
        """Replace ``encoder``, ``decoder`` and ``scalarizer`` with ``torch.compile`` wrappers, forwarding ``dynamic`` and ``fullgraph``."""
        self.encoder = torch.compile(self.encoder, dynamic=dynamic, fullgraph=fullgraph)
        self.decoder = torch.compile(self.decoder, dynamic=dynamic, fullgraph=fullgraph)
        self.scalarizer = torch.compile(self.scalarizer, dynamic=dynamic, fullgraph=fullgraph)

    def check_embedding_quality(self, mol_batch,
                                sigma=0.35,
                                type_distance_scaling=2,
                                # todo next two should be properties of the model
                                node_weight_temperature=1,
                                num_atom_types=5,
                                visualize=False,
                                ):
        """Compute the reconstruction loss and matched-point distances for a batch.

        Encodes a clone of ``mol_batch`` (with the centering check of :meth:`encode`) and
        decodes it, then sets ``mol_batch.x`` in place to
        ``atom_embedding_vector[mol_batch.z].flatten()`` (type class indices).
        :func:`~mxtaltools.models.autoencoder_utils.decoding2mol_batch` builds the decoded
        swarm and sets ``mol_batch.aux_ind`` in place,
        :func:`~mxtaltools.models.autoencoder_utils.ae_reconstruction_loss` gives the
        reconstruction loss, and :func:`~mxtaltools.models.autoencoder_utils.batch_rmsd`,
        called with its own default cutoff, threshold and ``type_distance_scaling`` (not
        this method's ``type_distance_scaling``), gives the matched distances.

        Parameters
        ----------
        mol_batch : MolData
            Molecules; ``x`` and ``aux_ind`` are overwritten in place, the other tensor
            attributes are replaced by detached tensors with the same values (see
            :func:`~mxtaltools.models.autoencoder_utils.init_decoded_data`), and the
            values of ``pos`` are not changed.
        sigma : float, optional
            Gaussian width for the reconstruction loss. Default 0.35.
        type_distance_scaling : float, optional
            Type-channel scaling for the reconstruction loss. Default 2.
        node_weight_temperature : float, optional
            Softmax temperature of the decoded weight logits. Default 1.
        num_atom_types : int, optional
            Number of type classes for the one-hot encodings, independent of the model's
            ``num_classes``. Default 5.
        visualize : bool, optional
            If True, show
            :func:`~mxtaltools.reporting.ae_reporting.swarm_vs_tgt_fig` for every graph
            (atom types ``[1, 6, 7, 8, 9]``) with ``renderer='browser'``. Default False.

        Returns
        -------
        graph_reconstruction_loss : torch.Tensor
            [num_graphs] per-graph reconstruction loss.
        rmsd : torch.Tensor
            [num_graphs] per-graph mean matched joint-space distance from
            :func:`~mxtaltools.models.autoencoder_utils.batch_rmsd`; NaN for graphs with
            an unmatched atom.
        complete_graph_bools : torch.Tensor
            [num_graphs] bool, True when every atom of the graph is matched.
        """
        encoding = self.encode(mol_batch.clone())
        decoding = self.decode(encoding)

        mol_batch.x = self.atom_embedding_vector[mol_batch.z].flatten()
        decoded_mol_batch, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
            decoding2mol_batch(mol_batch,
                               decoding,
                               self.num_decoder_nodes,
                               node_weight_temperature,
                               mol_batch.x.device))

        (nodewise_reconstruction_loss, nodewise_type_loss,
         graph_reconstruction_loss, self_likelihoods,
         nearest_node_loss, graph_clumping_loss,
         nearest_component_dist, nearest_component_loss) = ae_reconstruction_loss(mol_batch,
                                                                                  decoded_mol_batch,
                                                                                  nodewise_weights,
                                                                                  nodewise_weights_tensor,
                                                                                  num_atom_types,
                                                                                  type_distance_scaling,
                                                                                  sigma,
                                                                                  )


        true_node_one_hot = F.one_hot(mol_batch.x.flatten().long(), num_classes=num_atom_types).float()

        (rmsd, pred_dists, complete_graph_bools, particle_matched_bools,
         pred_particle_points, pred_particle_weights)=(
            batch_rmsd(mol_batch,
                       decoded_mol_batch,
                       true_node_one_hot))
        if visualize:
            for ind in range(mol_batch.num_graphs):
                swarm_vs_tgt_fig(mol_batch, decoded_mol_batch, [1, 6, 7, 8, 9], graph_ind=ind).show(renderer='browser')

        return graph_reconstruction_loss, rmsd, complete_graph_bools


class Mo3ENetDecoder(nn.Module):
    """MLP decoder of :class:`Mo3ENet`: a :class:`vectorMLP` from the scalar and vector bottleneck to per-node outputs.

    Parameters
    ----------
    seed : int
        Passed to :class:`vectorMLP`.
    config : Namespace
        Reads ``fc.num_layers``, ``fc.hidden_dim``, ``fc.norm``, ``fc.dropout``,
        ``fc.vector_norm``, ``activation`` and ``ramp_depth``.
    bottleneck_dim : int
        Width of the input scalars and channel count of the input vectors.
    output_depth : int
        Per-node output width including the 3 position columns; the scalar output has
        ``(output_depth - 3) * num_nodes`` columns.
    num_nodes : int
        Nodes per graph; the channel count of the vector output.
    """
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
        """Return ``(scalars, vectors)``, [num_graphs, (output_depth - 3) * num_nodes] and [num_graphs, 3, num_nodes], from ``x`` [num_graphs, bottleneck_dim] and ``v`` [num_graphs, 3, bottleneck_dim]."""
        return self.model(x, v)


class Mo3ENetGraphDecoder(nn.Module):
    """Graph decoder of :class:`Mo3ENet`: expands each bottleneck into ``num_nodes`` nodes on a fully connected graph and runs a :class:`VectorGNN`.

    Bias-free linear layers map the scalar bottleneck to ``hidden_dim`` scalar features
    per node (``s_to_nodes``), and the vector bottleneck to ``hidden_dim`` vector channels
    per node (``v_to_nodes``) and to one direction per node (``v_to_pos``). The
    :class:`VectorGNN` has ``config.fc.num_layers`` convolutions with one MLP layer each,
    node width ``hidden_dim``, message width ``hidden_dim // 4``, a 32-function Gaussian
    radial embedding with cutoff 2, ``'gelu'`` activation, no atom-type embedding, scalar
    output width ``output_depth - 3`` and one vector output channel. ``config.fc.norm`` and
    ``config.fc.vector_norm`` are passed with a ``'graph '`` prefix (None stays None).

    Parameters
    ----------
    config : Namespace
        Reads ``fc.hidden_dim``, ``fc.num_layers``, ``fc.norm``, ``fc.vector_norm`` and
        ``fc.dropout``.
    bottleneck_dim : int
        Width of the input scalars and channel count of the input vectors.
    output_depth : int
        Per-node output width including the 3 position columns.
    num_nodes : int
        Nodes per graph.
    """
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
        """Decode bottlenecks into per-node scalar and vector outputs.

        Node ``m`` of graph ``g`` is row ``g * num_nodes + m``. Its input position to the
        :class:`VectorGNN` is ``d / (0.1 + norm(d))``, where ``d`` is its ``v_to_pos``
        direction, so every input position has norm below 1; the returned vectors are
        separate outputs of the :class:`VectorGNN`. Edges join every ordered pair of
        distinct nodes within each graph and are passed to the :class:`VectorGNN` as
        ``{'edge_index': ...}``.

        Parameters
        ----------
        x : torch.Tensor
            [num_graphs, bottleneck_dim] scalar bottleneck.
        v : torch.Tensor
            [num_graphs, 3, bottleneck_dim] vector bottleneck.

        Returns
        -------
        tuple of torch.Tensor
            ``(scalars, vectors)`` from the :class:`VectorGNN`,
            [num_graphs * num_nodes, output_depth - 3] and [num_graphs * num_nodes, 3, 1].
        """
        eps = 1e-1
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
        pos = directions / (eps + torch.linalg.norm(directions, dim=1))[:, None]
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
    """Encoder of :class:`Mo3ENet`: a :class:`VectorMoleculeGraphModel` on atomic numbers and positions.

    Node inputs are the atomic number ``z`` plus the norm of each position
    (``concat_pos_to_node_dim=True``); there are no molecule-level features.

    Parameters
    ----------
    seed : int
        Passed to :class:`VectorMoleculeGraphModel`.
    config : Namespace
        ``activation``, ``fc`` and ``graph`` are passed as ``activation``, ``fc_config``
        and ``graph_config``.
    bottleneck_dim : int
        Output width of both the scalar and the vector track.
    override_cutoff : float or torch.Tensor, optional
        Graph cutoff used instead of ``config.graph.cutoff``. Default None.
    """
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

    def forward(self, mol_batch):
        """Return ``(x_out, v_out)``, [num_graphs, bottleneck_dim] and [num_graphs, 3, bottleneck_dim], for ``mol_batch`` (``z``, ``pos``, ``batch``, ``ptr`` and ``num_graphs`` are read; any radial graph is built inside the model from ``pos``)."""
        return self.model(mol_batch.z,
                          mol_batch.pos,
                          mol_batch.batch,
                          mol_batch.ptr,
                          num_graphs=mol_batch.num_graphs,
                          )

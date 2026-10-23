"""Message-passing graph network trunks producing per-node embeddings: :class:`ScalarGNN`
(scalar node features), :class:`VectorGNN` (scalar and Cartesian-vector node features)
and :class:`MolCrystalScalarGNN` (scalar features over an inside/outside molecular-crystal
cluster graph)."""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from mxtaltools.models.modules.basis_functions import GaussianEmbedding, BesselBasisLayer
from mxtaltools.models.modules.components import scalarMLP, vectorMLP, EmbeddingBlock
from mxtaltools.models.modules.graph_convolution import MConv, v_MConv


class ScalarGNN(torch.nn.Module):
    """Message-passing network mapping node features and positions to per-node embeddings.

    Node features are embedded by an :class:`EmbeddingBlock` (column 0 as an integer
    type index, the remaining columns concatenated, then a linear layer to
    ``node_dim``) and passed through a :class:`scalarMLP`. Each of ``num_convs``
    stages then applies an :class:`MConv` over the edges, with a radial basis
    expansion of the edge lengths as edge attributes, followed by a
    :class:`scalarMLP`. A bias-free linear layer maps ``node_dim`` to
    ``embedding_dim`` (an identity when they are equal).

    Parameters
    ----------
    input_node_dim : int
        Width of the input node features, including column 0.
    node_dim : int
        Hidden node width.
    fcs_per_gc : int
        Number of blocks in each :class:`scalarMLP`.
    message_dim : int
        Message width of each :class:`MConv`.
    embedding_dim : int
        Output width.
    num_convs : int
        Number of convolution stages. With 0, no edges or distances are used.
    num_radial : int
        Number of radial basis functions; also the edge-attribute width of each
        :class:`MConv`.
    num_input_classes : int, optional
        Passed to :class:`EmbeddingBlock`, whose table has ``num_input_classes + 1``
        rows. Default 101.
    cutoff : float, optional
        Stored as the float32 buffer ``cutoff`` and passed to the radial basis.
        Default 5.0.
    max_num_neighbors : int, optional
        Stored as ``max_num_neighbors``; not read by this module. Default 32.
    envelope_exponent : int, optional
        Envelope exponent of :class:`BesselBasisLayer`; used only with
        ``radial_embedding='bessel'``. Default 5.
    activation : str, optional
        Activation name for the :class:`scalarMLP` blocks and the :class:`MConv`
        messages. Default ``'gelu'``.
    atom_type_embedding_dim : int, optional
        Width of the type embedding in :class:`EmbeddingBlock`. Default 5.
    norm : str or None, optional
        Normalization name for the :class:`scalarMLP` blocks; the :class:`MConv`
        layers are built with ``norm=None``. Default None.
    dropout : float, optional
        Dropout probability in the :class:`scalarMLP` blocks. Default 0.
    radial_embedding : str, optional
        ``'bessel'`` (:class:`BesselBasisLayer` with ``cutoff``) or ``'gaussian'``
        (:class:`GaussianEmbedding` with ``num_radial`` centres evenly spaced from 0
        to ``cutoff``). Any other value leaves ``rbf`` unset, so :meth:`forward`
        raises AttributeError when ``num_convs > 0``. Default ``'bessel'``.
    override_cutoff : float, optional
        If given, used instead of ``cutoff``.
    """
    def __init__(self,
                 input_node_dim: int,
                 node_dim: int,
                 fcs_per_gc: int,
                 message_dim: int,
                 embedding_dim: int,
                 num_convs: int,
                 num_radial: int,
                 num_input_classes=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 atom_type_embedding_dim: int = 5,
                 norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 override_cutoff: Optional[float] = None
                 ):
        super(ScalarGNN, self).__init__()

        self.max_num_neighbors = max_num_neighbors

        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        self.init_node_embedding = EmbeddingBlock(node_dim,
                                                  num_input_classes,
                                                  input_node_dim,
                                                  atom_type_embedding_dim)

        self.zeroth_fc_block = scalarMLP(layers=fcs_per_gc,
                                         filters=node_dim,
                                         input_dim=node_dim,
                                         output_dim=node_dim,
                                         activation=activation,
                                         norm=norm,
                                         dropout=dropout)

        self.interaction_blocks = torch.nn.ModuleList([
            MConv(
                message_dim=message_dim,
                node_dim=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
                activation_fn=activation)
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            scalarMLP(layers=fcs_per_gc,
                      filters=node_dim,
                      input_dim=node_dim,
                      output_dim=node_dim,
                      activation=activation,
                      norm=norm,
                      dropout=dropout)
            for _ in range(num_convs)
        ])

        if node_dim != embedding_dim:
            self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def radial_embedding(self,
                         edge_index,
                         pos: torch.Tensor,
                         dist: Optional[torch.Tensor] = None,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the edge lengths and their radial basis expansion.

        Parameters
        ----------
        edge_index : torch.LongTensor
            Edges, [2, num_edges]; read only when ``dist`` is None.
        pos : torch.Tensor
            Node positions, [num_nodes, 3]; read only when ``dist`` is None.
        dist : torch.Tensor, optional
            Precomputed edge lengths, [num_edges]; returned unchanged.

        Returns
        -------
        tuple of torch.Tensor
            ``(dist, rbf)``: the Euclidean distances between ``pos[edge_index[0]]`` and
            ``pos[edge_index[1]]``, [num_edges], in the units of ``pos``, and
            ``self.rbf(dist)``, [num_edges, num_radial].
        """
        if dist is None:
            i, j = edge_index  # i->j source-to-target
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self,
                z: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.LongTensor,
                edge_index: torch.LongTensor,
                dist: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Embed the nodes of a batch of graphs.

        Parameters
        ----------
        z : torch.Tensor
            Node features, [num_nodes, input_node_dim] or [num_nodes] (treated as one
            column); column 0 is cast to long and embedded as the type index.
        pos : torch.Tensor
            Node positions, [num_nodes, 3].
        batch : torch.LongTensor
            Graph index of each node, [num_nodes]; forwarded to the
            :class:`scalarMLP` normalizations.
        edge_index : torch.LongTensor
            Edges, [2, num_edges]; read only when ``num_convs > 0``.
        dist : torch.Tensor, optional
            Precomputed edge lengths, [num_edges]; when None they are computed from
            ``pos``.

        Returns
        -------
        torch.Tensor
            Node embeddings, [num_nodes, embedding_dim].
        """

        x = self.init_node_embedding(z)
        x = self.zeroth_fc_block(x=x, batch=batch)

        if len(self.interaction_blocks) > 0:
            dist, rbf = self.radial_embedding(edge_index, pos, dist)
            for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
                x = convolution(x, edge_index, rbf)
                x = fc(x, batch=batch)

        return self.output_layer(x)


class VectorGNN(torch.nn.Module):
    """Message-passing network over scalar and Cartesian-vector node features.

    Scalars ``x`` are embedded by an :class:`EmbeddingBlock` (an identity when
    ``atom_type_embedding_dim`` is 0) and vectors ``v`` by a bias-free linear layer
    over their channel axis, from ``v_input_node_dim`` to ``node_dim`` channels; a
    :class:`vectorMLP` then updates both. Each of ``num_convs`` stages applies an
    :class:`MConv` to ``x`` and a :class:`v_MConv` to ``v``, both with a radial basis
    expansion of the edge lengths as edge attributes, followed by a
    :class:`vectorMLP`. Bias-free linear output layers map the scalar track to
    ``embedding_dim`` and the vector track to ``v_embedding_dim`` channels
    (identities when the width already equals ``node_dim``).

    Parameters
    ----------
    input_node_dim : int
        Width of the input scalars, including column 0; not used when
        ``atom_type_embedding_dim`` is 0.
    node_dim : int
        Hidden width of both tracks. When ``atom_type_embedding_dim`` is 0, the input
        scalars must already have this width.
    fcs_per_gc : int
        Number of blocks in each :class:`vectorMLP`.
    message_dim : int
        Message width of each :class:`MConv` and :class:`v_MConv`.
    embedding_dim : int
        Output width of the scalar track.
    num_convs : int
        Number of convolution stages. With 0, no edges or distances are used.
    num_radial : int
        Number of radial basis functions; also the edge-attribute width of each
        convolution.
    num_input_classes : int, optional
        Passed to :class:`EmbeddingBlock`, whose table has ``num_input_classes + 1``
        rows. Default 101.
    cutoff : float, optional
        Stored as the float32 buffer ``cutoff`` and passed to the radial basis.
        Default 5.0.
    max_num_neighbors : int, optional
        Stored as ``max_num_neighbors``; not read by this module. Default 32.
    envelope_exponent : int, optional
        Envelope exponent of :class:`BesselBasisLayer`; used only with
        ``radial_embedding='bessel'``. Default 5.
    activation : str, optional
        Activation name for the :class:`vectorMLP` layers and the :class:`MConv`
        messages. Default ``'gelu'``.
    atom_type_embedding_dim : int, optional
        Width of the type embedding in :class:`EmbeddingBlock`; 0 replaces the
        embedding with an identity. Default 5.
    norm : str or None, optional
        Scalar normalization name for the :class:`vectorMLP` layers; the
        convolutions are built with ``norm=None``. Default None.
    vector_norm : str or None, optional
        Vector normalization name for the :class:`vectorMLP` layers. Default None.
    dropout : float, optional
        Dropout probability, passed to the :class:`vectorMLP` layers. Default 0.
    radial_embedding : str, optional
        ``'bessel'`` or ``'gaussian'``, as in :class:`ScalarGNN`; any other value
        leaves ``rbf`` unset, so :meth:`forward` raises AttributeError when
        ``num_convs > 0``. Default ``'bessel'``.
    override_cutoff : float or torch.Tensor, optional
        If given, used instead of ``cutoff``; a tensor is cloned and detached.
    v_embedding_dim : int, optional
        Output channel count of the vector track. Defaults to ``embedding_dim``.
    v_input_node_dim : int, optional
        Input channel count of ``v``. Defaults to 1.
    """
    def __init__(self,
                 input_node_dim: int,
                 node_dim: int,
                 fcs_per_gc: int,
                 message_dim: int,
                 embedding_dim: int,
                 num_convs: int,
                 num_radial: int,
                 num_input_classes=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 atom_type_embedding_dim: int = 5,
                 norm: Optional[str] = None,
                 vector_norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 override_cutoff: Optional[float] = None,
                 v_embedding_dim: Optional[int] = None,
                 v_input_node_dim: Optional[int] = None,
                 ):
        super(VectorGNN, self).__init__()

        self.max_num_neighbors = max_num_neighbors

        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32) if not torch.is_tensor(override_cutoff) else override_cutoff.clone().detach())

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        if atom_type_embedding_dim == 0:
            self.init_node_embedding = nn.Identity()
        else:
            self.init_node_embedding = EmbeddingBlock(node_dim,
                                                      num_input_classes,
                                                      input_node_dim,
                                                      atom_type_embedding_dim)
        if v_input_node_dim is None:
            v_input_node_dim = 1
        self.init_vector_embedding = self.init_vector_embedding = nn.Linear(v_input_node_dim, node_dim, bias=False)

        self.zeroth_fc_block = vectorMLP(layers=fcs_per_gc,
                                         filters=node_dim,
                                         input_dim=node_dim,
                                         output_dim=node_dim,
                                         activation=activation,
                                         norm=norm,
                                         dropout=dropout,
                                         vector_input_dim=node_dim,
                                         vector_output_dim=node_dim,
                                         vector_norm=vector_norm)

        self.interaction_blocks = torch.nn.ModuleList([
            MConv(
                message_dim=message_dim,
                node_dim=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
                activation_fn=activation)
            for _ in range(num_convs)
        ])
        self.vector_interaction_blocks = torch.nn.ModuleList([
            v_MConv(
                message_depth=message_dim,
                node_depth=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
            )
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            vectorMLP(layers=fcs_per_gc,
                      filters=node_dim,
                      input_dim=node_dim,
                      output_dim=node_dim,
                      activation=activation,
                      norm=norm,
                      dropout=dropout,
                      vector_norm=vector_norm,
                      vector_input_dim=node_dim,
                      vector_output_dim=node_dim)
            for _ in range(num_convs)
        ])

        if node_dim != embedding_dim:
            self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

        if v_embedding_dim is None:
            v_embedding_dim = embedding_dim

        if node_dim != v_embedding_dim:
            self.v_output_layer = nn.Linear(node_dim, v_embedding_dim, bias=False)
        else:
            self.v_output_layer = nn.Identity()

    def radial_embedding(self,
                         edge_index: torch.LongTensor,
                         pos: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the edge lengths and their radial basis expansion.

        Parameters
        ----------
        edge_index : torch.LongTensor
            Edges, [2, num_edges].
        pos : torch.Tensor
            Node positions, [num_nodes, 3].

        Returns
        -------
        tuple of torch.Tensor
            ``(dist, rbf)``: the Euclidean distances between ``pos[edge_index[0]]`` and
            ``pos[edge_index[1]]``, [num_edges], in the units of ``pos``, and
            ``self.rbf(dist)``, [num_edges, num_radial].
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.LongTensor,
                edges_dict: dict
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed the scalar and vector node features of a batch of graphs.

        Parameters
        ----------
        x : torch.Tensor
            Node scalars, [num_nodes, input_node_dim] or [num_nodes] (column 0 is cast
            to long and embedded as the type index); [num_nodes, node_dim] when
            ``atom_type_embedding_dim`` is 0.
        v : torch.Tensor
            Node vectors, [num_nodes, 3, v_input_node_dim].
        pos : torch.Tensor
            Node positions, [num_nodes, 3].
        batch : torch.LongTensor
            Graph index of each node, [num_nodes]; forwarded to the
            :class:`vectorMLP` layers.
        edges_dict : dict
            ``edges_dict['edge_index']``, [2, num_edges], is read only when
            ``num_convs > 0``; otherwise ``edges_dict`` is not read.

        Returns
        -------
        tuple of torch.Tensor
            ``(x_out, v_out)`` with ``x_out`` [num_nodes, embedding_dim] and ``v_out``
            [num_nodes, 3, v_embedding_dim].
        """

        x = self.init_node_embedding(x)
        v = self.init_vector_embedding(v)
        x, v = self.zeroth_fc_block(x=x, v=v, batch=batch)

        if len(self.interaction_blocks) > 0:
            dist, rbf = self.radial_embedding(edges_dict['edge_index'], pos)
            for n, (convolution, vector_convolution, fc) in enumerate(
                    zip(self.interaction_blocks, self.vector_interaction_blocks, self.fc_blocks)):
                x = convolution(x, edges_dict['edge_index'], rbf)
                v = vector_convolution(v, edges_dict['edge_index'], rbf)
                x, v = fc(x=x, v=v, batch=batch)

        return self.output_layer(x), self.v_output_layer(v)


class MolCrystalScalarGNN(torch.nn.Module):
    """Scalar message-passing network over an inside/outside molecular-crystal cluster graph.

    Layers are built as in :class:`ScalarGNN`. :meth:`forward` convolves over the
    intramolecular and intermolecular edges together, applies each stage's
    :class:`scalarMLP` to the inside nodes only, and after every stage copies the
    inside-node features to all nodes of their graph with
    :meth:`periodize_molecular_crystal`.

    Parameters
    ----------
    input_node_dim : int
        Width of the input node features, including column 0.
    node_dim : int
        Hidden node width.
    fcs_per_gc : int
        Number of blocks in each :class:`scalarMLP`.
    message_dim : int
        Message width of each :class:`MConv`.
    embedding_dim : int
        Output width.
    num_convs : int
        Number of convolution stages.
    num_radial : int
        Number of radial basis functions; also the edge-attribute width of each
        :class:`MConv`.
    num_input_classes : int, optional
        Passed to :class:`EmbeddingBlock`, whose table has ``num_input_classes + 1``
        rows. Default 101.
    cutoff : float, optional
        Stored as the float32 buffer ``cutoff`` and passed to the radial basis.
        Default 5.0.
    max_num_neighbors : int, optional
        Stored as ``max_num_neighbors``; not read by this module. Default 32.
    envelope_exponent : int, optional
        Envelope exponent of :class:`BesselBasisLayer`; used only with
        ``radial_embedding='bessel'``. Default 5.
    activation : str, optional
        Activation name for the :class:`scalarMLP` blocks and the :class:`MConv`
        messages. Default ``'gelu'``.
    atom_type_embedding_dim : int, optional
        Width of the type embedding in :class:`EmbeddingBlock`. Default 5.
    norm : str or None, optional
        Normalization name for the :class:`scalarMLP` blocks; the :class:`MConv`
        layers are built with ``norm=None``. Default None.
    dropout : float, optional
        Dropout probability in the :class:`scalarMLP` blocks. Default 0.
    radial_embedding : str, optional
        ``'bessel'`` or ``'gaussian'``, as in :class:`ScalarGNN`; any other value
        leaves ``rbf`` unset, so :meth:`forward` raises AttributeError when
        ``num_convs > 0``. Default ``'bessel'``.
    override_cutoff : float, optional
        If given, used instead of ``cutoff``.
    """
    def __init__(self,
                 input_node_dim: int,
                 node_dim: int,
                 fcs_per_gc: int,
                 message_dim: int,
                 embedding_dim: int,
                 num_convs: int,
                 num_radial: int,
                 num_input_classes=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 atom_type_embedding_dim: int = 5,
                 norm: Optional[str] = None,
                 dropout: float = 0,
                 radial_embedding: str = 'bessel',
                 override_cutoff: Optional[float] = None
                 ):
        super(MolCrystalScalarGNN, self).__init__()

        self.max_num_neighbors = max_num_neighbors

        if override_cutoff is None:
            self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        else:
            self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)

        self.init_node_embedding = EmbeddingBlock(node_dim,
                                                  num_input_classes,
                                                  input_node_dim,
                                                  atom_type_embedding_dim)

        self.zeroth_fc_block = scalarMLP(layers=fcs_per_gc,
                                         filters=node_dim,
                                         input_dim=node_dim,
                                         output_dim=node_dim,
                                         activation=activation,
                                         norm=norm,
                                         dropout=dropout)

        self.interaction_blocks = torch.nn.ModuleList([
            MConv(
                message_dim=message_dim,
                node_dim=node_dim,
                edge_embedding_dim=num_radial,
                norm=None,
                activation_fn=activation)
            for _ in range(num_convs)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            scalarMLP(layers=fcs_per_gc,
                      filters=node_dim,
                      input_dim=node_dim,
                      output_dim=node_dim,
                      activation=activation,
                      norm=norm,
                      dropout=dropout)
            for _ in range(num_convs)
        ])

        if node_dim != embedding_dim:
            self.output_layer = nn.Linear(node_dim, embedding_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def radial_embedding(self,
                         edge_index: torch.LongTensor,
                         pos: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the edge lengths and their radial basis expansion.

        Parameters
        ----------
        edge_index : torch.LongTensor
            Edges, [2, num_edges].
        pos : torch.Tensor
            Node positions, [num_nodes, 3].

        Returns
        -------
        tuple of torch.Tensor
            ``(dist, rbf)``: the Euclidean distances between ``pos[edge_index[0]]`` and
            ``pos[edge_index[1]]``, [num_edges], in the units of ``pos``, and
            ``self.rbf(dist)``, [num_edges, num_radial].
        """
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        return dist, self.rbf(dist)  # apply radial basis functions

    def forward(self,
                z: torch.Tensor,
                pos: torch.Tensor,
                batch: torch.LongTensor,
                aux_ind: torch.Tensor,
                ptr: torch.LongTensor,
                edges_dict: dict
                ) -> torch.Tensor:
        """Embed a batch of molecular-crystal cluster graphs, reducing to the inside nodes.

        ``edges_dict['edge_index']`` and ``edges_dict['edge_index_inter']`` are
        concatenated along dim 1 and their lengths computed from ``pos``. In each
        stage the :class:`MConv` is applied to all nodes over all of these edges, the
        stage's :class:`scalarMLP` output replaces the rows ``inside_inds``, and
        :meth:`periodize_molecular_crystal` copies the inside rows to every node of
        their graph and writes ``aux_ind`` into the last feature column.

        Parameters
        ----------
        z : torch.Tensor
            Node features, [num_nodes, input_node_dim] or [num_nodes] (treated as one
            column); column 0 is cast to long and embedded as the type index.
        pos : torch.Tensor
            Node positions, [num_nodes, 3].
        batch : torch.LongTensor
            Graph index of each node, [num_nodes]. The zeroth :class:`scalarMLP`
            receives ``batch``; the stage blocks receive ``batch[inside_inds]``.
        aux_ind : torch.Tensor
            Node region label, [num_nodes], written into the last feature column of
            every node after each stage.
        ptr : torch.LongTensor
            Node offsets of the graphs, [num_graphs + 1].
        edges_dict : dict
            Edge dict in the form returned by ``build_radial_graph`` with ``aux_ind``.
            ``'edge_index'``, ``'edge_index_inter'``, ``'inside_inds'``,
            ``'outside_inds'``, ``'inside_batch'`` and ``'n_repeats'`` are looked up
            even when ``num_convs`` is 0; ``'outside_inds'`` is not otherwise used.

        Returns
        -------
        torch.Tensor
            With ``num_convs > 0``, embeddings of the inside nodes in the order of
            ``inside_inds``, [num_inside, embedding_dim]; with ``num_convs`` 0,
            embeddings of all nodes, [num_nodes, embedding_dim].
        """

        x = self.init_node_embedding(z)
        x = self.zeroth_fc_block(x=x, batch=batch)

        # assumes input with inside-outside structure, and enforces periodicity after each convolution
        edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = (
            edges_dict['edge_index'], edges_dict['edge_index_inter'], edges_dict['inside_inds'],
            edges_dict['outside_inds'], edges_dict['inside_batch'], edges_dict['n_repeats']
        )
        edge_index = torch.cat((edge_index, edge_index_inter), dim=1)  # all edges counted in one big batch

        if len(self.interaction_blocks) > 0:
            dist, rbf = self.radial_embedding(edge_index, pos)

            for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
                x = convolution(x, edge_index, rbf)

                # manually periodize inside nodes to outside nodes after each convolution
                x[inside_inds] = fc(x[inside_inds], batch=batch[inside_inds])

                # then broadcast node features to all symmetry images
                x = self.periodize_molecular_crystal(inside_batch, inside_inds, n, n_repeats, ptr, x, aux_ind)

        return self.output_layer(x)

    def periodize_molecular_crystal(self, inside_batch, inside_inds, n, n_repeats, ptr, x, aux_ind):
        """Copy inside-node features to every node of their graph and rewrite the last column.

        For each graph ``ii``, rows ``ptr[ii]:ptr[ii + 1]`` of ``x`` are overwritten
        with that graph's inside rows (``x[inside_inds[inside_batch == ii]]``) tiled
        ``n_repeats[ii]`` times along dim 0. The tiled block must have as many rows as
        the graph has nodes, or a single row, which is broadcast to all of them;
        otherwise the assignment raises RuntimeError. Each node receives the features
        of its matching inside node only when the
        graph's nodes are laid out as consecutive copies of its inside nodes in the
        same order. The last column of ``x`` is then set to ``aux_ind``.

        Parameters
        ----------
        inside_batch : torch.LongTensor
            Graph index of each inside node, [num_inside].
        inside_inds : torch.LongTensor
            Indices of the inside nodes, [num_inside].
        n : int
            Index of the current convolution stage.
        n_repeats : list of int
            Number of copies per graph, length ``num_graphs``.
        ptr : torch.LongTensor
            Node offsets of the graphs, [num_graphs + 1].
        x : torch.Tensor
            Node features, [num_nodes, node_dim]; modified in place.
        aux_ind : torch.Tensor
            Node region label, [num_nodes].

        Returns
        -------
        torch.Tensor
            ``x``, [num_nodes, node_dim]; on the last stage
            (``n == len(self.interaction_blocks) - 1``), ``x[inside_inds]``,
            [num_inside, node_dim].
        """
        for ii in range(len(ptr) - 1):  # enforce periodicity for each crystal, assuming invariant node features
            # copy the first asymmetric unit to all periodic images (safe since all are SE(3) invariant)
            # assumes the crystals are ordered specifically in this way
            # todo check if this could be done faster with repeat_interleave
            x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)

        x[:, -1] = aux_ind  # manually re-indicate inside/outside structure

        if n == len(self.interaction_blocks) - 1:
            x = x[inside_inds]  # reduce to inside image on the final convolution

        return x

# old method - has since been broken up into separate models
# class GraphNeuralNetwork(torch.nn.Module):
#     def __init__(self,
#                  input_node_dim: int,
#                  node_dim: int,
#                  fcs_per_gc: int,
#                  message_dim: int,
#                  embedding_dim: int,
#                  num_convs: int,
#                  num_radial: int,
#                  num_input_classes=101,
#                  cutoff: float = 5.0,
#                  max_num_neighbors: int = 32,
#                  envelope_exponent: int = 5,
#                  activation='gelu',
#                  atom_type_embedding_dim=5,
#                  norm: Optional[str] = None,
#                  dropout: float = 0,
#                  radial_embedding: str = 'bessel',
#                  num_attention_heads: int = 1,
#
#                  periodize_inside_nodes: bool = False,
#                  outside_convolution_type: str = 'none',
#                  add_vector_track: bool = False,
#                  vector_norm: Optional[str] = None,
#                  override_cutoff: Optional[float] = None
#                  ):
#         super(GraphNeuralNetwork, self).__init__()
#
#         self.max_num_neighbors = max_num_neighbors
#         if override_cutoff is None:
#             self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
#         else:
#             self.register_buffer('cutoff', torch.tensor(override_cutoff, dtype=torch.float32))
#
#         self.periodize_inside_nodes = periodize_inside_nodes
#         self.outside_convolution_type = outside_convolution_type
#         self.add_vector_track = add_vector_track
#         self.vector_addition_rescaling_factor = 1.6
#
#         if radial_embedding == 'bessel':
#             self.rbf = BesselBasisLayer(num_radial, self.cutoff, envelope_exponent)
#         elif radial_embedding == 'gaussian':
#             self.rbf = GaussianEmbedding(start=0.0, stop=self.cutoff, num_gaussians=num_radial)
#
#         self.init_node_embedding = EmbeddingBlock(node_dim,
#                                                   num_input_classes,
#                                                   input_node_dim,
#                                                   atom_type_embedding_dim)
#
#         if self.add_vector_track:
#             self.init_vector_embedding = nn.Linear(1, node_dim, bias=False)
#
#         self.zeroth_fc_block = FCBlock(
#             fcs_per_gc,
#             node_dim,
#             activation,
#             norm,
#             dropout,
#             equivariant=self.add_vector_track,
#             vector_norm=vector_norm)
#
#         self.interaction_blocks = torch.nn.ModuleList([
#             GCBlock(message_dim,
#                     node_dim,
#                     num_radial,
#                     heads=num_attention_heads,
#                     add_vector_channel=add_vector_track,
#                     )
#             for _ in range(num_convs)
#         ])
#
#         self.fc_blocks = torch.nn.ModuleList([
#             FCBlock(
#                 fcs_per_gc,
#                 node_dim,
#                 activation,
#                 norm,
#                 dropout,
#                 equivariant=self.add_vector_track,
#                 vector_norm=vector_norm,
#             )
#             for _ in range(num_convs)
#         ])
#
#         self.output_block = OutputBlock(node_dim, embedding_dim, add_vector_track)
#
#     def radial_embedding(self, edge_index, pos):
#         """
#         compute elements for radial & spherical embeddings
#         """
#         i, j = edge_index  # i->j source-to-target
#         dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
#
#         return dist, self.rbf(dist)  # apply radial basis functions
#
#     def forward(self, z, pos, batch, ptr, edges_dict: dict):
#
#         x, v = self.init_node_embeddings(z)
#
#         if self.add_vector_track:
#             x, v = self.zeroth_fc_block(x=x, v=v, batch=batch)
#         else:
#             x = self.zeroth_fc_block(x=x, v=v, batch=batch)
#
#         if len(self.interaction_blocks) > 0:
#             (edge_index, edge_index_inter,
#              inside_batch, inside_inds,
#              n_repeats,
#              rbf, rbf_inter) = self.get_edges(edges_dict, pos)
#
#             for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
#                 if v is not None:
#                     x_res, v_res = x.clone(), v.clone()
#                     x, v = convolution(x, v, rbf, edge_index)
#                     x = x + x_res
#                     v = (v + v_res) / self.vector_addition_rescaling_factor
#                 else:
#                     x = x + convolution(x, v, rbf, edge_index)
#
#                 if not self.periodize_inside_nodes:  # inside/outside periodic convolution
#                     if self.add_vector_track:
#                         x_res, v_res = x.clone(), v.clone()
#                         x, v = fc(x, v=v, batch=batch)
#                         x = x + x_res
#                         v = (v + v_res) / self.vector_addition_rescaling_factor
#                     else:
#                         x = x + fc(x, v=v, batch=batch)
#
#                     #assert torch.sum(torch.isnan(x)) == 0, f"NaN in fc_block output {get_model_nans(self.fc_blocks)}"
#
#                 else:
#                     assert v is None, "Vector embeddings not set up for periodic molecular crystal graph convolutions"
#                     # update only the inside inds
#                     x[inside_inds] = (x[inside_inds] + fc(x[inside_inds], batch=batch[inside_inds]))
#
#                     # then broadcast to all symmetry images
#                     x = self.periodize_molecular_crystal(inside_batch, inside_inds, n, n_repeats, ptr, x)
#
#         return self.output_block(x, v)
#
#     def init_node_embeddings(self, z):
#         if self.add_vector_track:
#             x, v = z[:, :-3], z[:, -3:]  # vector features are trailing 3 dimensions of node input
#             v = self.init_vector_embedding(v[:, :, None])  # [n_nodes, 3] -> [n_nodes, 3, n_dim]
#             x = self.init_node_embedding(x)  # embed atomic numbers & compute initial atom-wise feature vector
#         else:
#             x, v = self.init_node_embedding(z), None  # embed atomic numbers & compute initial atom-wise feature vector
#         return x, v
#
#     def periodize_molecular_crystal(self, inside_batch, inside_inds, n, n_repeats, ptr, x):
#         for ii in range(len(ptr) - 1):  # enforce periodicity for each crystal, assuming invariant node features
#             # copy the first asymmetric unit to all periodic images (safe since all are SE(3) invariant)
#             x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)
#
#         if n == len(self.interaction_blocks) - 1:
#             x = x[inside_inds]  # reduce to inside image on the final convolution
#
#         return x
#
#     def get_edges(self,
#                   edges_dict: dict,
#                   pos: torch.Tensor):
#         if self.outside_convolution_type == 'none':
#             # no inside/outside distinctions
#             edge_index = edges_dict['edge_index']
#             if 'dists' in edges_dict.keys():  # previously generated distances - e.g., for periodic MIC
#                 dist = edges_dict['dists']
#                 rbf = self.rbf(dist)
#             else:
#                 dist, rbf = self.radial_embedding(edge_index, pos)
#             edge_index_inter, inside_batch, inside_inds, n_repeats, rbf_inter = None, None, None, None, None
#             assert not self.periodize_inside_nodes, "Cannot periodize to outside nodes if there are no outside nodes"
#
#         elif self.outside_convolution_type == 'all_layers':
#             # assumes input with inside-outside structure, and enforces periodicity after each convolution
#
#             edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
#             edge_index = torch.cat((edge_index, edge_index_inter), dim=1)  # all edges counted in one big batch
#             dist, rbf = self.radial_embedding(edge_index, pos)
#             rbf_inter = None
#
#         elif self.outside_convolution_type == 'last_layer':
#             assert False, "Last layer outside convolution is deprecated"
#             # edge_index, edge_index_inter, inside_inds, outside_inds, inside_batch, n_repeats = list(edges_dict.values())
#             # dist, rbf = self.radial_embedding(edge_index, pos)
#             # dist_inter, rbf_inter = self.radial_embedding(torch.cat((edge_index, edge_index_inter), dim=1), pos)
#
#             # re-integrate this in forward method to bring it back
#             # if n == (len(self.interaction_blocks) - 1) and self.outside_convolution_type == 'last_layer':
#             #     # return only the results of the intermolecular convolution, omitting intramolecular features
#             #     x = convolution(x, v,
#             #                     rbf_inter,
#             #                     torch.cat((edge_index, edge_index_inter), dim=1),
#             #                     batch)
#
#         else:
#             assert False, "Must select a valid treatment of inside vs outside nodes"
#
#         return edge_index, edge_index_inter, inside_batch, inside_inds, n_repeats, rbf, rbf_inter

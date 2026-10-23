"""Graph neural network models that map batches of molecules, molecular crystals or
molecular clusters to pooled outputs.

Each model embeds node features with a graph network from
:mod:`mxtaltools.models.graph_models.graph_neural_network`, pools nodes with a learnable
augmented-softmax aggregation, optionally applies an MLP from
:mod:`mxtaltools.models.modules.components`, and projects to ``output_dim``.
"""

from argparse import Namespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from mxtaltools.models.graph_models.graph_neural_network import ScalarGNN, VectorGNN, \
    MolCrystalScalarGNN
from mxtaltools.models.modules.augmented_softmax_aggregator import AugSoftmaxAggregation, VectorAugSoftmaxAggregation
from mxtaltools.models.modules.components import scalarMLP, vectorMLP
from mxtaltools.models.functions.radial_graph import build_radial_graph
from mxtaltools.models.functions.minimum_image_neighbors import argwhere_minimum_image_convention_edges


# noinspection PyAttributeOutsideInit
class ScalarMoleculeGraphModel(nn.Module):
    """Scalar graph neural network mapping a batch of molecular graphs to one output row per graph.

    Node features are optionally extended with positions and per-graph features, encoded
    by a :class:`ScalarGNN`, pooled per graph with a learnable
    :class:`AugSoftmaxAggregation`, optionally passed through a :class:`scalarMLP`, and
    projected to ``output_dim`` by a bias-free linear layer (identity when the widths
    already match).

    Parameters
    ----------
    input_node_dim : int
        Width of the node features given to the graph network, not counting the
        ``num_mol_feats`` columns added when ``concat_mol_to_node_dim`` is True (those are
        added here). It is not increased for ``concat_pos_to_node_dim``, so in that case it
        must already count the three position columns. Column 0 of the node features is
        read as an integer atom type by the graph network's embedding block.
    output_dim : int
        Width of the per-graph output.
    fc_config : Namespace
        Post-pooling MLP settings: ``num_layers`` (0 disables the MLP), ``hidden_dim``,
        ``norm`` and ``dropout``.
    graph_config : Namespace
        Graph network settings. ``cutoff`` and ``max_num_neighbors`` are read here, every
        attribute is forwarded as a keyword argument to :class:`ScalarGNN`, and
        ``embedding_dim`` also sets the pooling and MLP input width.
    activation : str, optional
        Activation name forwarded to :class:`ScalarGNN`. Default ``'gelu'``.
    num_mol_feats : int, optional
        Width of the per-graph features ``mol_x``. When nonzero, ``mol_fc =
        Linear(num_mol_feats, num_mol_feats)`` is built and its output is concatenated to
        the pooled embedding before the MLP. Default 0.
    concat_pos_to_node_dim : bool, optional
        Concatenate ``pos`` to the node features before the graph network. Default False.
    concat_mol_to_node_dim : bool, optional
        Repeat ``mol_x`` over the nodes of each graph and concatenate it to the node
        features. Default False.
    seed : int, optional
        Passed to ``torch.manual_seed`` at construction and to the MLP. Default 5.
    override_cutoff : float, optional
        If given, used instead of ``graph_config.cutoff`` for the ``convolution_cutoff``
        buffer and by the graph network.

    Notes
    -----
    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator.
    """

    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_pos_to_node_dim: bool = False,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(ScalarMoleculeGraphModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_pos_to_node_dim = concat_pos_to_node_dim
        self.concat_mol_to_node_dim = concat_mol_to_node_dim

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats

        self.graph_net = ScalarGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = scalarMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                batch: torch.LongTensor,
                ptr: torch.LongTensor,
                mol_x: Union[torch.Tensor],
                num_graphs: int,
                edge_index: Optional[torch.LongTensor] = None,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False,
                ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Encode a batch of graphs.

        This model does not build edges: ``edge_index`` is passed straight to the graph
        network, which needs it whenever it has convolution layers, and distances are
        computed there from ``pos``.

        Parameters
        ----------
        x : torch.Tensor
            Node features, [num_nodes, num_features] or [num_nodes] (treated as one column).
        pos : torch.FloatTensor
            Node positions, [num_nodes, 3].
        batch : torch.LongTensor
            Graph index of each node, [num_nodes].
        ptr : torch.LongTensor
            Node offsets of the graphs, [num_graphs + 1]; read only when
            ``concat_mol_to_node_dim`` is True.
        mol_x : torch.Tensor
            Per-graph features, [num_graphs, num_mol_feats]; read when
            ``concat_mol_to_node_dim`` is True, or when both ``mol_fc`` and the MLP exist.
        num_graphs : int
            Number of graphs; the length of the pooled output.
        edge_index : torch.LongTensor, optional
            Edges, [2, num_edges].
        edges_dict : dict, optional
            Not read; returned under ``'dists_dict'`` when ``return_dists`` is True.
        return_latent : bool, optional
            Return the per-graph activation entering ``output_fc`` under
            ``'final_activation'``.
        return_dists : bool, optional
            Return ``edges_dict`` under ``'dists_dict'``.
        return_embedding : bool, optional
            Return the node embeddings from the graph network under ``'graph_embedding'``.

        Returns
        -------
        torch.Tensor or tuple
            The output, [num_graphs, output_dim]. If any ``return_*`` flag is True,
            ``(output, extra_outputs)`` with the dict from :meth:`collect_extra_outputs`.
        """

        x = self.append_init_node_features(x, pos, ptr, mol_x)
        x = self.graph_net(x,
                           pos,
                           batch,
                           edge_index)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation
        x = self.global_pool(x,
                             batch,
                             dim_size=num_graphs)

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                x = torch.cat([x, self.mol_fc(mol_x)], dim=-1)
            gmlp_out = self.gnn_mlp(x)

            x = gmlp_out

        output = self.output_fc(x)

        extra_outputs = self.collect_extra_outputs(x,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        """Collect the optional outputs of :meth:`forward` into a dict.

        Keys present only when their flag is True: ``'dists_dict'`` (``edges_dict``, as
        given), ``'final_activation'`` (``x.detach()``) and ``'graph_embedding'``
        (``embedding.detach()``). Empty dict when no flag is set.
        """
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self, x, pos, ptr, mol_x):
        """Return node features with optional positions and per-graph features appended.

        A 1-D ``x`` becomes one column. ``pos`` [num_nodes, 3] is appended when
        ``concat_pos_to_node_dim`` is True, then ``mol_x`` repeated by
        ``torch.diff(ptr)`` nodes per graph when ``concat_mol_to_node_dim`` is True.
        The inputs are not modified in place.
        """
        if x.ndim == 1:
            x = x[:, None]

        # simply append node coordinates, PointNet style
        if self.concat_pos_to_node_dim:
            x = torch.cat((x, pos), dim=-1)

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x


# noinspection PyAttributeOutsideInit
class VectorMoleculeGraphModel(nn.Module):
    """Graph neural network with scalar and Cartesian-vector channels, mapping a batch of
    molecular graphs to per-graph scalar and vector outputs.

    Each node starts from its scalar features and a single vector channel equal to its
    position. A :class:`VectorGNN` updates both tracks over a radial graph,
    :class:`AugSoftmaxAggregation` and :class:`VectorAugSoftmaxAggregation` pool them per
    graph, an optional :class:`vectorMLP` processes the pooled pair, and bias-free linear
    layers ``output_fc`` and ``v_output_fc`` project both tracks to ``output_dim``
    channels (identities when the widths already match).

    Parameters
    ----------
    input_node_dim : int
        Width of the node features ``x`` as passed to :meth:`forward`. The width given to
        the graph network adds 1 when ``concat_pos_to_node_dim`` is True and
        ``num_mol_feats`` when ``concat_mol_to_node_dim`` is True. Unless
        ``graph_config.atom_type_embedding_dim`` is 0, column 0 is read as an integer atom
        type by the graph network's embedding block.
    output_dim : int
        Channel count of both outputs.
    fc_config : Namespace
        Post-pooling :class:`vectorMLP` settings: ``num_layers`` (0 disables the MLP),
        ``hidden_dim``, ``norm``, ``vector_norm`` and ``dropout``.
    graph_config : Namespace
        Graph network settings. ``cutoff`` and ``max_num_neighbors`` are read here, every
        attribute is forwarded as a keyword argument to :class:`VectorGNN`, and
        ``embedding_dim`` also sets the pooling width.
    activation : str, optional
        Activation name forwarded to :class:`VectorGNN`. Default ``'gelu'``.
    num_mol_feats : int, optional
        Width of the per-graph features ``mol_x``. When nonzero, ``mol_fc =
        Linear(num_mol_feats, num_mol_feats)`` is built and, if the MLP exists, its output
        is concatenated to the pooled scalar embedding before the MLP. Default 0.
    concat_pos_to_node_dim : bool, optional
        Append the Euclidean norm of each node position as one scalar column. Default
        False.
    concat_mol_to_node_dim : bool, optional
        Repeat ``mol_x`` over the nodes of each graph and append it to the scalar node
        features. Default False.
    seed : int, optional
        Passed to ``torch.manual_seed`` at construction and to the MLP. Default 5.
    override_cutoff : float or torch.Tensor, optional
        If given, used instead of ``graph_config.cutoff`` for the ``convolution_cutoff``
        buffer (which sets the radius of graphs built in :meth:`forward`) and by the graph
        network.

    Notes
    -----
    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator.

    The initial vector input has one channel, so ``graph_config.v_input_node_dim`` must be
    1, None or absent. The graph network's vector output has
    ``graph_config.v_embedding_dim`` channels (``graph_config.embedding_dim`` when that is
    absent or None), while the vector pooling is built with ``graph_config.embedding_dim``
    channels, so the two must be equal. When ``fc_config.num_layers > 0`` the MLP's vector
    track is built for ``fc_config.hidden_dim`` input channels, which must then equal the
    pooled vector width as well.
    """

    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_pos_to_node_dim: bool = False,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(VectorMoleculeGraphModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_pos_to_node_dim = concat_pos_to_node_dim
        self.concat_mol_to_node_dim = concat_mol_to_node_dim

        if self.concat_pos_to_node_dim:
            input_node_dim += 1  # radial dimension - vector features explicitly added later

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32) if not torch.is_tensor(override_cutoff) else override_cutoff.clone().detach())

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats

        self.graph_net = VectorGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        self.v_global_pool = VectorAugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = vectorMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     vector_input_dim=fc_config.hidden_dim,
                                     v_to_s_combination='sum',
                                     vector_norm=fc_config.vector_norm,
                                     vector_output_dim=fc_config.hidden_dim,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
            self.v_output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()
            self.v_output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                batch: torch.LongTensor,
                ptr: torch.LongTensor,
                num_graphs: int,
                mol_x: Optional[torch.Tensor] = None,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """Encode a batch of graphs into per-graph scalar and vector outputs.

        When the graph network has convolution layers or ``return_dists`` is True:

        - if ``edges_dict`` is None, a radial graph is built with
          :func:`build_radial_graph` from ``pos``, ``batch``, ``ptr``,
          ``convolution_cutoff`` and ``max_num_neighbors``;
        - otherwise ``edges_dict`` is replaced by None, so a supplied edge dict is
          discarded and the graph network receives None.

        Otherwise ``edges_dict`` is passed through unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Node features, [num_nodes, input_node_dim] or [num_nodes] (treated as one
            column).
        pos : torch.FloatTensor
            Node positions, [num_nodes, 3]; the initial vector channel, the source of the
            appended norm column, and the input to edge construction.
        batch : torch.LongTensor
            Graph index of each node, [num_nodes].
        ptr : torch.LongTensor
            Node offsets of the graphs, [num_graphs + 1].
        num_graphs : int
            Number of graphs; the length of the pooled outputs.
        mol_x : torch.Tensor, optional
            Per-graph features, [num_graphs, num_mol_feats]. Needed when
            ``concat_mol_to_node_dim`` is True, or when ``num_mol_feats > 0`` and the MLP
            exists.
        edges_dict : dict, optional
            Edge dict; see above.
        return_latent : bool, optional
            Return the pooled scalar activation entering ``output_fc`` (after the MLP when
            present) under ``'final_activation'``.
        return_dists : bool, optional
            Return the edge dict used in this call under ``'dists_dict'``.
        return_embedding : bool, optional
            Return the scalar node embeddings from the graph network under
            ``'graph_embedding'``.

        Returns
        -------
        tuple
            ``(x_out, v_out)`` with ``x_out`` [num_graphs, output_dim] and ``v_out``
            [num_graphs, 3, output_dim]. If any ``return_*`` flag is True,
            ``(x_out, v_out, extra_outputs)`` with the dict from
            :meth:`collect_extra_outputs`.
        """

        if len(self.graph_net.interaction_blocks) > 0 or return_dists:
            # todo clean up options around prebuilt radial graphs
            if edges_dict is None:  # option to rebuild radial graph
                edges_dict = build_radial_graph(
                    pos,
                    batch,
                    ptr,
                    self.convolution_cutoff,
                    self.max_num_neighbors,
                )
            else:
                edges_dict = None

        x, v = self.append_init_node_features(x, pos, ptr, mol_x)
        x, v = self.graph_net(x,
                              v,
                              pos,
                              batch,
                              edges_dict)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation
        x = self.global_pool(x,
                             batch,
                             dim_size=num_graphs)
        v = self.v_global_pool(v,
                               batch,
                               dim_size=num_graphs,
                               dim=0,
                               cart_dim=1)

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                x = torch.cat([x, self.mol_fc(mol_x)], dim=-1)
            x, v = self.gnn_mlp(x, v)

        x_out, v_out = self.output_fc(x), self.v_output_fc(v)

        extra_outputs = self.collect_extra_outputs(x,
                                                   pos,
                                                   batch,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return x_out, v_out, extra_outputs
        else:
            return x_out, v_out

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              pos: torch.Tensor,
                              batch: torch.LongTensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        """Collect the optional outputs of :meth:`forward` into a dict.

        Keys present only when their flag is True: ``'dists_dict'`` (``edges_dict``, as
        given), ``'final_activation'`` (``x.detach()``) and ``'graph_embedding'``
        (``embedding.detach()``). ``pos`` and ``batch`` are not read. Empty dict when no
        flag is set.
        """
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self,
                                  x: torch.Tensor,
                                  pos: torch.Tensor,
                                  ptr: torch.LongTensor,
                                  mol_x: Optional[torch.Tensor] = None,
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the initial scalar and vector node inputs.

        Returns
        -------
        x : torch.Tensor
            [num_nodes, width]: ``x`` (1-D input becomes one column), then
            ``torch.linalg.norm(pos, dim=1)`` when ``concat_pos_to_node_dim`` is True, then
            ``mol_x`` repeated by ``torch.diff(ptr)`` nodes per graph when
            ``concat_mol_to_node_dim`` is True.
        v : torch.Tensor
            [num_nodes, 3, 1]: ``pos`` with a trailing channel dimension (a view of
            ``pos``).
        """
        if x.ndim == 1:
            x = x[:, None]

        # append radial position as scalar feature
        # and 3 vector dimensions (unit vectors from centroid)
        rad = torch.linalg.norm(pos, dim=1)
        if self.concat_pos_to_node_dim:
            x = torch.cat((x, rad[:, None]), dim=-1)  # radii
        # v = pos / (rad[:, None] + 1e-5)  # normed directions
        v = pos[..., None]  # set dimension as [n,3,k]
        # richer embedding as 3 component vectors rather than one single vector
        #v = pos[:, :, None] * torch.eye(3, device=pos.device, dtype=torch.float32).repeat(len(pos), 1, 1)

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x, v


# noinspection PyAttributeOutsideInit
class MolecularCrystalGraphModel(nn.Module):
    """Scalar graph neural network over molecular-crystal graphs with an inside/outside
    node structure, producing one output row per crystal.

    Node features are encoded by a :class:`MolCrystalScalarGNN`, which convolves over
    intra- and intermolecular edges and, after each convolution, copies the features of the
    inside nodes onto every image of the crystal. The inside nodes are pooled per crystal
    with a learnable :class:`AugSoftmaxAggregation`, optionally passed through a
    :class:`scalarMLP`, and projected to ``output_dim`` by a bias-free linear layer
    (identity when the widths already match).

    Parameters
    ----------
    input_node_dim : int
        Width of the node features ``x`` as passed to :meth:`forward`. The width given to
        the graph network adds ``num_mol_feats`` when ``concat_mol_to_node_dim`` is True
        and 2 when ``concat_mol_ind_to_node_dim`` is True. Column 0 is read as an integer
        atom type by the graph network's embedding block.
    output_dim : int
        Width of the per-crystal output.
    fc_config : Namespace
        Post-pooling MLP settings: ``num_layers`` (0 disables the MLP), ``hidden_dim``,
        ``norm`` and ``dropout``.
    graph_config : Namespace
        Graph network settings. ``cutoff`` and ``max_num_neighbors`` are read here, every
        attribute is forwarded as a keyword argument to :class:`MolCrystalScalarGNN`, and
        ``embedding_dim`` also sets the pooling and MLP input width.
    activation : str, optional
        Activation name forwarded to the graph network. Default ``'gelu'``.
    num_mol_feats : int, optional
        Width of the per-crystal features ``mol_x``. When nonzero, ``mol_fc =
        Linear(num_mol_feats, num_mol_feats)`` is built and its output is concatenated to
        the pooled embedding before the MLP. Default 0.
    concat_mol_ind_to_node_dim : bool, optional
        Append ``mol_ind`` and ``aux_ind`` as two node-feature columns. Default False.
    concat_mol_to_node_dim : bool, optional
        Repeat ``mol_x`` over the nodes of each crystal and append it to the node
        features. Default False.
    seed : int, optional
        Passed to ``torch.manual_seed`` at construction and to the MLP. Default 5.
    override_cutoff : float, optional
        If given, used instead of ``graph_config.cutoff`` for the ``convolution_cutoff``
        buffer and by the graph network.

    Notes
    -----
    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator.
    """

    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_mol_ind_to_node_dim: bool = False,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(MolecularCrystalGraphModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_mol_to_node_dim = concat_mol_to_node_dim
        self.concat_mol_ind_to_node_dim = concat_mol_ind_to_node_dim

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats
        if concat_mol_ind_to_node_dim:
            input_node_dim += 2  # aux_ind and mol_ind will be appended

        self.graph_net = MolCrystalScalarGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = scalarMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                batch: torch.LongTensor,
                ptr: torch.LongTensor,
                mol_x: Union[torch.Tensor],
                num_graphs: int,
                aux_ind: torch.LongTensor,
                mol_ind: torch.LongTensor,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False,
                force_edges_rebuild: bool = False,
                ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Encode a batch of molecular crystals.

        When the graph network has convolution layers, ``return_dists`` is True or
        ``force_edges_rebuild`` is True, and ``edges_dict`` is None or
        ``force_edges_rebuild`` is True, the edges are built with
        :func:`build_radial_graph` from ``pos``, ``batch``, ``ptr``,
        ``convolution_cutoff``, ``max_num_neighbors``, ``aux_ind`` and ``mol_ind``. The
        graph network and the pooling always index ``edges_dict`` (``'edge_index'``,
        ``'edge_index_inter'``, ``'inside_inds'``, ``'outside_inds'``, ``'inside_batch'``,
        ``'n_repeats'``), so an edge dict must be supplied or built.

        Parameters
        ----------
        x : torch.Tensor
            Node features, [num_nodes, input_node_dim] or [num_nodes] (treated as one
            column).
        pos : torch.FloatTensor
            Node positions, [num_nodes, 3].
        batch : torch.LongTensor
            Crystal index of each node, [num_nodes].
        ptr : torch.LongTensor
            Node offsets of the crystals, [num_graphs + 1].
        mol_x : torch.Tensor
            Per-crystal features, [num_graphs, num_mol_feats].
        num_graphs : int
            Number of crystals; the length of the pooled output.
        aux_ind : torch.LongTensor
            Node region label, [num_nodes]: 0 for inside nodes, 1 for outside neighbour
            candidates, 2 for nodes in no edge (see :func:`build_radial_graph`).
        mol_ind : torch.LongTensor
            Molecule index of each node, [num_nodes]; separates intra- from
            intermolecular edges.
        edges_dict : dict, optional
            Prebuilt edge dict from :func:`build_radial_graph` with ``aux_ind`` given.
        return_latent : bool, optional
            Return, under ``'final_activation'``, the node features given to the graph
            network (``x`` after :meth:`append_init_node_features`), detached.
        return_dists : bool, optional
            Return the edge dict under ``'dists_dict'``; see
            :meth:`collect_extra_outputs`.
        return_embedding : bool, optional
            Return the node embeddings from the graph network under
            ``'graph_embedding'``.
        force_edges_rebuild : bool, optional
            Rebuild the edges even when ``edges_dict`` is given.

        Returns
        -------
        torch.Tensor or tuple
            The output, [num_graphs, output_dim]. If any ``return_*`` flag is True,
            ``(output, extra_outputs)``.
        """

        if len(self.graph_net.interaction_blocks) > 0 or return_dists or force_edges_rebuild:
            if edges_dict is None or force_edges_rebuild:  # option to rebuild radial graph
                edges_dict = build_radial_graph(
                    pos,
                    batch,
                    ptr,
                    self.convolution_cutoff,
                    self.max_num_neighbors,
                    aux_ind=aux_ind,
                    mol_ind=mol_ind,
                )

        x = self.append_init_node_features(x, ptr, mol_x, aux_ind, mol_ind)
        g = self.graph_net(x,
                           pos,
                           batch,
                           aux_ind,
                           ptr,
                           edges_dict)  # get graph encoding

        if return_embedding:
            embedding = g.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation, only over 'inside' nodes
        g = self.global_pool(g,
                             edges_dict['inside_batch'],
                             dim_size=num_graphs)

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                g = torch.cat([g, self.mol_fc(mol_x)], dim=-1)
            g = self.gnn_mlp(g)

        output = self.output_fc(g)

        extra_outputs = self.collect_extra_outputs(x,
                                                   pos,
                                                   batch,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              pos: torch.Tensor,
                              batch: torch.LongTensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        """Collect the optional outputs of :meth:`forward` into a dict.

        With ``return_dists``, ``'dists_dict'`` is ``edges_dict`` itself; if it contains
        ``'edge_index_inter'``, these keys are written into that same dict (mutating the
        ``edges_dict`` passed in):``'intermolecular_dist'`` (Euclidean length of each
        intermolecular edge from ``pos``, [num_inter_edges]),
        ``'intermolecular_dist_batch'`` (``batch`` of each edge's first node),
        ``'intermolecular_dist_atoms'`` (a list of two tensors: column 0 of ``x`` at each
        end, cast to long) and ``'intermolecular_dist_inds'`` (the edge index). With
        ``return_latent``, ``'final_activation'`` is ``x.detach()``; with
        ``return_embedding``, ``'graph_embedding'`` is ``embedding.detach()``.
        """
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict
            if 'edge_index_inter' in edges_dict.keys():
                extra_outputs['dists_dict']['intermolecular_dist'] = (
                    (pos[edges_dict['edge_index_inter'][0]] - pos[edges_dict['edge_index_inter'][1]]).pow(2).sum(
                        dim=-1).sqrt()
                )

                extra_outputs['dists_dict']['intermolecular_dist_batch'] = batch[edges_dict['edge_index_inter'][0]]

                extra_outputs['dists_dict']['intermolecular_dist_atoms'] = \
                    [x[edges_dict['edge_index_inter'][0], 0].long(),
                     x[edges_dict['edge_index_inter'][1], 0].long()]

                extra_outputs['dists_dict']['intermolecular_dist_inds'] = edges_dict['edge_index_inter']

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self, x, ptr, mol_x, aux_ind, mol_ind):
        """Return node features with optional index columns and per-crystal features appended.

        A 1-D ``x`` becomes one column. ``mol_ind`` and ``aux_ind`` are appended as two
        columns when ``concat_mol_ind_to_node_dim`` is True, then ``mol_x`` repeated by
        ``torch.diff(ptr)`` nodes per crystal when ``concat_mol_to_node_dim`` is True.
        The inputs are not modified in place.
        """
        if x.ndim == 1:
            x = x[:, None]

        if self.concat_mol_ind_to_node_dim:
            x = torch.cat((x, mol_ind[:, None], aux_ind[:, None]), dim=-1)

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x


# noinspection PyAttributeOutsideInit
class MoleculeClusterModel(nn.Module):
    """Scalar graph neural network over a periodic cluster of molecules, producing one
    output row per molecule.

    Nodes are encoded by a :class:`ScalarGNN` over given edges or over
    minimum-image-convention edges built from the cell, pooled per molecule by ``mol_ind``
    with a learnable :class:`AugSoftmaxAggregation`, optionally passed through a
    :class:`scalarMLP`, and projected to ``output_dim`` by a bias-free linear layer
    (identity when the widths already match).

    Parameters
    ----------
    input_node_dim : int
        Width of the node features ``x`` as passed to :meth:`forward`; the width given to
        the graph network adds ``num_mol_feats`` when ``concat_mol_to_node_dim`` is True.
        Column 0 is read as an integer atom type by the graph network's embedding block.
    output_dim : int
        Width of the per-molecule output.
    fc_config : Namespace
        Post-pooling MLP settings: ``num_layers`` (0 disables the MLP), ``hidden_dim``,
        ``norm`` and ``dropout``.
    graph_config : Namespace
        Graph network settings. ``cutoff`` and ``max_num_neighbors`` are read here, every
        attribute is forwarded as a keyword argument to :class:`ScalarGNN`, and
        ``embedding_dim`` also sets the pooling and MLP input width.
    activation : str, optional
        Activation name forwarded to :class:`ScalarGNN`. Default ``'gelu'``.
    num_mol_feats : int, optional
        Width of ``mol_x``. When nonzero, ``mol_fc = Linear(num_mol_feats,
        num_mol_feats)`` is built and its output is concatenated to the pooled embedding
        before the MLP. Default 0.
    concat_mol_to_node_dim : bool, optional
        Repeat ``mol_x`` by ``torch.diff(ptr)`` nodes per graph and append it to the node
        features. Default False.
    seed : int, optional
        Passed to ``torch.manual_seed`` at construction and to the MLP. Default 5.
    override_cutoff : float, optional
        If given, used instead of ``graph_config.cutoff`` for the ``convolution_cutoff``
        buffer (the radius of minimum-image edges built in :meth:`forward`) and by the
        graph network.

    Notes
    -----
    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator.
    """

    def __init__(self,
                 input_node_dim: int,
                 output_dim: int,
                 fc_config: Namespace,
                 graph_config: Namespace,
                 activation: str = 'gelu',
                 num_mol_feats: int = 0,
                 concat_mol_to_node_dim: bool = False,
                 seed: int = 5,
                 override_cutoff=None
                 ):

        super(MoleculeClusterModel, self).__init__()

        torch.manual_seed(seed)

        self.concat_mol_to_node_dim = concat_mol_to_node_dim

        if override_cutoff is None:
            self.register_buffer('convolution_cutoff', torch.tensor(graph_config.cutoff, dtype=torch.float32))
        else:
            self.register_buffer('convolution_cutoff', torch.tensor(override_cutoff, dtype=torch.float32))

        self.max_num_neighbors = graph_config.max_num_neighbors
        self.num_fc_layers = fc_config.num_layers

        if concat_mol_to_node_dim:
            input_node_dim += num_mol_feats

        self.graph_net = ScalarGNN(
            activation=activation,
            input_node_dim=input_node_dim,
            override_cutoff=override_cutoff,
            **graph_config.__dict__
        )

        # initialize global pooling operation
        self.global_pool = AugSoftmaxAggregation(
            temperature=1,
            learn=True,
            bias=0.1,
            channels=graph_config.embedding_dim)

        # molecule features FC layer
        self.mol_fc = nn.Linear(num_mol_feats, num_mol_feats) if num_mol_feats != 0 else None

        """Optional MLP model to post-process graph embedding"""
        if fc_config.num_layers > 0:
            self.gnn_mlp = scalarMLP(layers=fc_config.num_layers,
                                     filters=fc_config.hidden_dim,
                                     norm=fc_config.norm,
                                     dropout=fc_config.dropout,
                                     input_dim=graph_config.embedding_dim + num_mol_feats,
                                     output_dim=fc_config.hidden_dim,
                                     seed=seed,
                                     )
            graph_output_dim = fc_config.hidden_dim
        else:
            graph_output_dim = graph_config.embedding_dim

        """initialize output reshaping layers"""
        if graph_output_dim != output_dim:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(graph_output_dim, output_dim, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pos: torch.FloatTensor,
                ptr: torch.LongTensor,
                mol_x: Union[torch.Tensor],
                num_graphs: int,
                mol_ind: torch.Tensor,
                T_fc: torch.Tensor,
                edge_index: Optional[torch.LongTensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                edges_dict: Optional[dict] = None,
                return_latent: bool = False,
                return_dists: bool = False,
                return_embedding: bool = False
                ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Encode the molecules of a cluster.

        When the graph network has convolution layers or ``return_dists`` is True, edges
        are chosen in this order: ``edge_index`` with ``edge_attr`` if ``edge_index`` is
        given; else, if ``edges_dict`` is None, edges from
        :func:`argwhere_minimum_image_convention_edges` (``num_graphs``, ``pos``,
        ``T_fc``, ``convolution_cutoff``), which asserts ``num_graphs == 1``; else
        ``edges_dict['edge_index']`` and ``edges_dict['dists']``. Distances are computed
        from ``pos`` by the graph network when ``edge_attr`` is None.

        Nodes are pooled with index ``mol_ind - mol_ind.min()`` and
        ``dim_size=int(mol_ind.max())``.

        Parameters
        ----------
        x : torch.Tensor
            Node features, [num_nodes, input_node_dim] or [num_nodes] (treated as one
            column).
        pos : torch.FloatTensor
            Cartesian node positions, [num_nodes, 3].
        ptr : torch.LongTensor
            Node offsets of the graphs, [num_graphs + 1].
        mol_x : torch.Tensor
            Per-graph features; read when ``concat_mol_to_node_dim`` is True, or when
            both ``mol_fc`` and the MLP exist.
        num_graphs : int
            Number of graphs in the batch.
        mol_ind : torch.Tensor
            Molecule index of each node, [num_nodes]; the pooling index and the batch
            vector for the graph network's normalizations.
        T_fc : torch.Tensor
            Fractional-to-Cartesian cell matrix, [3, 3] or [1, 3, 3]; read only when
            minimum-image edges are built.
        edge_index : torch.LongTensor, optional
            Edges, [2, num_edges].
        edge_attr : torch.Tensor, optional
            Edge lengths, [num_edges], passed to the graph network as distances.
        edges_dict : dict, optional
            Dict with ``'edge_index'`` and ``'dists'``.
        return_latent : bool, optional
            Return the per-molecule activation entering ``output_fc`` under
            ``'final_activation'``.
        return_dists : bool, optional
            Return ``edges_dict`` (the given or built dict; None when ``edge_index`` was
            given without one) under ``'dists_dict'``.
        return_embedding : bool, optional
            Return the node embeddings from the graph network under
            ``'graph_embedding'``.

        Returns
        -------
        torch.Tensor or tuple
            The output, [int(mol_ind.max()), output_dim]. If any ``return_*`` flag is True,
            ``(output, extra_outputs)``.

        Raises
        ------
        AssertionError
            If ``ptr[-1] != len(x)``.
        """

        assert ptr[-1] == len(x), "Only one cluster sample per batch is allowed for this model"

        if len(self.graph_net.interaction_blocks) > 0 or return_dists:
            if edge_index is not None:
                pass
            elif edges_dict is None:  # option to rebuild radial graph
                edges_dict = argwhere_minimum_image_convention_edges(
                    num_graphs, pos, T_fc, self.convolution_cutoff)
                edge_index = edges_dict['edge_index']
                edge_attr = edges_dict['dists']
            else:
                edge_index = edges_dict['edge_index']
                edge_attr = edges_dict['dists']

        x = self.append_init_node_features(x, ptr, mol_x)
        x = self.graph_net(x,
                           pos,
                           mol_ind,
                           edge_index,
                           edge_attr)  # get graph encoding

        if return_embedding:
            embedding = x.clone()
        else:
            embedding = None

        # aggregate atoms to molecule / graph representation
        x = self.global_pool(x,
                             index=mol_ind - mol_ind.min(),
                             dim_size=int(mol_ind.max()))

        if self.num_fc_layers > 0:
            if self.mol_fc is not None:
                x = torch.cat([x, self.mol_fc(mol_x)], dim=-1)
            x = self.gnn_mlp(x)

        extra_outputs = self.collect_extra_outputs(x,
                                                   edges_dict,
                                                   return_dists,
                                                   return_latent,
                                                   return_embedding,
                                                   embedding)

        if len(extra_outputs) > 0:
            return self.output_fc(x), extra_outputs
        else:
            return self.output_fc(x)

    @staticmethod
    def collect_extra_outputs(x: torch.Tensor,
                              edges_dict: dict,
                              return_dists: bool,
                              return_latent: bool,
                              return_embedding: bool,
                              embedding: Union[torch.Tensor, None]) -> dict:
        """Collect the optional outputs of :meth:`forward` into a dict.

        Keys present only when their flag is True: ``'dists_dict'`` (``edges_dict``, as
        given), ``'final_activation'`` (``x.detach()``) and ``'graph_embedding'``
        (``embedding.detach()``). Empty dict when no flag is set.
        """
        extra_outputs = {}

        if return_dists:
            extra_outputs['dists_dict'] = edges_dict

        if return_latent:
            extra_outputs['final_activation'] = x.detach()

        if return_embedding:
            extra_outputs['graph_embedding'] = embedding.detach()

        return extra_outputs

    def append_init_node_features(self, x, ptr, mol_x):
        """Return node features with per-graph features optionally appended.

        A 1-D ``x`` becomes one column; ``mol_x`` repeated by ``torch.diff(ptr)`` nodes per
        graph is appended when ``concat_mol_to_node_dim`` is True. The inputs are not
        modified in place.
        """
        if x.ndim == 1:
            x = x[:, None]

        # add molwise information to input node features
        if self.concat_mol_to_node_dim:
            nodes_per_graph = torch.diff(ptr)
            x = torch.cat((x,
                           torch.repeat_interleave(mol_x, nodes_per_graph, 0)),
                          dim=-1)

        return x

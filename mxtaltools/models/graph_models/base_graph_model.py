"""Base module for graph task models: element-property node featurization, feature
standardization from stored buffers, and a forward pass into a wrapped graph model."""
import torch
from torch_geometric.typing import OptTensor

from mxtaltools.constants.atom_properties import VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD


class BaseGraphModel(torch.nn.Module):
    """Base module that featurizes and standardizes graph inputs and calls a wrapped model.

    The wrapped network is expected in ``self.model``, which this class does not set;
    :meth:`forward` and :meth:`compile_self` use it. Construction sets ``atom_feats``,
    ``mol_feats``, ``n_atom_feats`` and ``n_mol_feats`` to 0; :meth:`get_data_stats`
    sets them and registers the buffers the other methods read.
    """
    def __init__(self):
        super(BaseGraphModel, self).__init__()
        self.atom_feats = 0
        self.mol_feats = 0
        self.n_mol_feats = 0
        self.n_atom_feats = 0

    def get_data_stats(self,
                       atom_features: list,
                       molecule_features: list,
                       node_standardization_tensor: OptTensor = None,
                       graph_standardization_tensor: OptTensor = None
                       ):
        """Record the feature lists and register the featurization and standardization buffers.

        Sets ``atom_feats``, ``mol_feats``, ``n_atom_feats`` and ``n_mol_feats``, and
        registers ``atom_properties_tensor``, [len(VDW_RADII), n_atom_feats]: row ``i``
        holds ``i`` followed by the ``i``-th value of each selected table from
        ``mxtaltools.constants.atom_properties``, always in the order ``'vdw_radii'``,
        ``'atom_weight'``, ``'electronegativity'``, ``'group'``, ``'period'``, whatever
        their order in ``atom_features``. Also registers ``node_standardization_tensor``
        and, when ``molecule_features`` is non-empty, ``graph_standardization_tensor``.

        Parameters
        ----------
        atom_features : list
            Node feature names. Its length sets ``n_atom_feats`` and must equal one (the
            leading index column) plus the number of the five property names above that
            it contains.
        molecule_features : list
            Graph feature names; its length sets ``n_mol_feats``. See
            :meth:`featurize_input_graph` for the names that produce columns.
        node_standardization_tensor : torch.Tensor or array-like, optional
            [n_atom_feats, 2]: :meth:`standardize` subtracts column 0 from each node
            feature and divides by column 1. A non-tensor is converted to float32.
            Default: column 0 all 0, column 1 all 1.
        graph_standardization_tensor : torch.Tensor or array-like, optional
            [n_mol_feats, 2], used the same way for the graph features. Registered only
            when ``molecule_features`` is non-empty. Same conversion and default.

        Raises
        ------
        AssertionError
            If the number of columns built for ``atom_properties_tensor`` differs from
            ``len(atom_features)``.
        """

        if node_standardization_tensor is None:
            node_standardization_tensor = torch.ones((len(atom_features), 2), dtype=torch.float32)
            node_standardization_tensor[:, 0] = 0
        if graph_standardization_tensor is None:
            graph_standardization_tensor = torch.ones((len(molecule_features), 2), dtype=torch.float32)
            graph_standardization_tensor[:, 0] = 0

        self.n_atom_feats = len(atom_features)
        self.n_mol_feats = len(molecule_features)
        self.atom_feats = atom_features
        self.mol_feats = molecule_features

        # generate atom property embeddings
        atom_embeddings_list = [torch.arange(len(VDW_RADII))]  # start with raw atomic number
        if 'vdw_radii' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(VDW_RADII.values())))
        if 'atom_weight' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(ATOM_WEIGHTS.values())))
        if 'electronegativity' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(ELECTRONEGATIVITY.values())))
        if 'group' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(GROUP.values())))
        if 'period' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(PERIOD.values())))

        assert len(atom_embeddings_list) == self.n_atom_feats

        self.register_buffer('atom_properties_tensor', torch.stack(atom_embeddings_list).T)

        if not torch.is_tensor(node_standardization_tensor):
            node_standardization_tensor = torch.tensor(node_standardization_tensor, dtype=torch.float32)
        if not torch.is_tensor(graph_standardization_tensor):
            graph_standardization_tensor = torch.tensor(graph_standardization_tensor, dtype=torch.float32)

        # store atom standardizations
        self.register_buffer('node_standardization_tensor', node_standardization_tensor)
        if self.n_mol_feats != 0:
            self.register_buffer('graph_standardization_tensor', graph_standardization_tensor)

    def featurize_input_graph(self,
                              data
                              ):
        """Set the node and graph input features of ``data``.

        Assigns ``data.x = atom_properties_tensor[data.z.long()]``, [num_nodes,
        n_atom_feats]. When ``n_mol_feats > 0``, also assigns ``data.mol_x``,
        [num_graphs, k], stacking ``data.num_atoms``, ``data.radius`` and
        ``data.mol_volume`` in that order, each only if its name is in ``mol_feats``;
        ``k`` is the number of these names present, and other names add no column.
        If none is present, ``torch.stack`` of the empty list raises RuntimeError.

        Parameters
        ----------
        data : MolData or MolCrystalData
            Collated graphs with ``z`` and, when used, the graph properties above.
            Modified in place.

        Returns
        -------
        MolData or MolCrystalData
            ``data`` itself.
        """

        data.x = self.atom_properties_tensor[data.z.long()]

        if self.n_mol_feats > 0:
            mol_x_list = []
            if 'num_atoms' in self.mol_feats:
                mol_x_list.append(data.num_atoms)
            if 'radius' in self.mol_feats:
                mol_x_list.append(data.radius)
            if 'mol_volume' in self.mol_feats:
                mol_x_list.append(data.mol_volume)
            data.mol_x = torch.stack(mol_x_list).T

        return data

    def standardize(self,
                    data
                    ):
        """Standardize ``data.x`` and, when ``n_mol_feats > 0``, ``data.mol_x``.

        Each column has column 0 of ``node_standardization_tensor`` (of
        ``graph_standardization_tensor`` for ``mol_x``) subtracted and is divided by
        column 1. Every column of ``x`` is transformed, including the leading
        atomic-number column.

        Parameters
        ----------
        data : MolData or MolCrystalData
            Graphs with ``x`` and, when used, ``mol_x``, as set by
            :meth:`featurize_input_graph`. The attributes are reassigned on ``data``.

        Returns
        -------
        MolData or MolCrystalData
            ``data`` itself.
        """

        data.x = (data.x - self.node_standardization_tensor[:, 0]) / self.node_standardization_tensor[:, 1]

        if self.n_mol_feats > 0:
            data.mol_x = (
                    (data.mol_x - self.graph_standardization_tensor[:, 0]) / self.graph_standardization_tensor[:, 1])

        return data

    def forward(self,
                data_batch,
                return_dists: bool = False,
                return_latent: bool = False,
                force_edges_rebuild: bool = False,
                ):
        """Featurize and standardize ``data_batch``, build edges if needed, and call ``self.model``.

        Runs :meth:`featurize_input_graph` and :meth:`standardize`. When
        ``data_batch.edge_index`` is None or ``force_edges_rebuild`` is True, edges are
        built with cutoff ``float(self.model.convolution_cutoff)``: by
        ``construct_intra_radial_graph`` if the class name of ``data_batch`` contains
        ``'crystal'`` (case-insensitive), otherwise by ``construct_radial_graph``.

        Parameters
        ----------
        data_batch : MolData or MolCrystalData
            Collated graphs with ``z``, ``pos``, ``batch``, ``ptr``, ``num_graphs`` and
            ``edge_index``. ``mol_x`` is read whatever ``n_mol_feats`` is; with
            ``n_mol_feats`` 0 it is not assigned here, so it must already be set
            (a missing attribute raises AttributeError). Modified in place: ``x``,
            ``mol_x`` and, when edges are built, ``edges_dict`` and ``edge_index``.
        return_dists : bool, optional
            Forwarded to ``self.model``. Default False.
        return_latent : bool, optional
            Forwarded to ``self.model``. Default False.
        force_edges_rebuild : bool, optional
            Build edges even when ``data_batch.edge_index`` is set. Default False.

        Returns
        -------
        object
            The return value of ``self.model(x, pos, batch, ptr, mol_x, num_graphs,
            edge_index=edge_index, return_dists=return_dists,
            return_latent=return_latent)``, with the arguments taken from ``data_batch``.

        Notes
        -----
        ``construct_radial_graph`` is provided only by the ``MolCrystalAnalysis``
        mixin of :class:`MolCrystalData`, whose class name contains ``'crystal'``, so
        on a :class:`MolData` batch the non-crystal branch raises AttributeError. On
        a crystal batch that already has ``aux_ind`` set (a cluster),
        ``construct_intra_radial_graph`` raises RuntimeError.
        """
        # featurize atom properties on the fly
        data_batch = self.featurize_input_graph(data_batch)

        # standardize on the fly from model-attached statistics
        data_batch = self.standardize(data_batch)

        # get radial graph
        if data_batch.edge_index is None or force_edges_rebuild:
            if 'crystal' in data_batch.__class__.__name__.lower():
                data_batch.construct_intra_radial_graph(float(self.model.convolution_cutoff))
            else:
                data_batch.construct_radial_graph(float(self.model.convolution_cutoff))

        return self.model(data_batch.x,
                          data_batch.pos,
                          data_batch.batch,
                          data_batch.ptr,
                          data_batch.mol_x,
                          data_batch.num_graphs,
                          edge_index=data_batch.edge_index,
                          return_dists=return_dists,
                          return_latent=return_latent)

    def compile_self(self, dynamic=True, fullgraph=False):
        """Replace ``self.model`` with ``torch.compile(self.model, dynamic=dynamic, fullgraph=fullgraph)``."""
        self.model = torch.compile(self.model, dynamic=dynamic, fullgraph=fullgraph)

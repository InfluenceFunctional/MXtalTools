"""Graph-model wrapper that featurizes and standardizes a molecular-crystal batch and
encodes it with a :class:`MolecularCrystalGraphModel`, one output row per crystal."""
import torch
from torch_geometric.typing import OptTensor

from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.molecule_graph_model import MolecularCrystalGraphModel


class MolecularCrystalModel(BaseGraphModel):
    def __init__(self, seed, config,
                 atom_features: list,
                 molecule_features: list,
                 output_dim: int,
                 node_standardization_tensor: OptTensor = None,
                 graph_standardization_tensor: OptTensor = None):
        """Wrapper for a :class:`MolecularCrystalGraphModel`, with input featurization and
        standardization.

        Calls ``torch.manual_seed(seed)``, which reseeds torch's global random number
        generator, then :meth:`BaseGraphModel.get_data_stats`, which registers the
        atom-property lookup table and the standardization buffers. ``self.model`` is a
        :class:`MolecularCrystalGraphModel` with ``input_node_dim=len(atom_features)``,
        ``num_mol_feats=len(molecule_features)`` and ``concat_mol_to_node_dim=True``.

        Parameters
        ----------
        seed : int
            Seed for ``torch.manual_seed``; also passed to the graph model.
        config : Namespace
            Model settings. ``config.activation``, ``config.fc`` and ``config.graph`` are
            passed to the graph model as ``activation``, ``fc_config`` and
            ``graph_config``.
        atom_features : list
            Atom feature names. The node features are the atomic number followed by those
            of ``'vdw_radii'``, ``'atom_weight'``, ``'electronegativity'``, ``'group'`` and
            ``'period'`` that appear in the list, in that fixed order.
        molecule_features : list
            Per-crystal feature names; ``'num_atoms'``, ``'radius'`` and ``'mol_volume'``
            are read from the batch attributes of the same name.
        output_dim : int
            Width of the per-crystal output.
        node_standardization_tensor : torch.Tensor, optional
            (mean, std) of each node-feature column, [len(atom_features), 2]. Default:
            mean 0 and std 1 for every column.
        graph_standardization_tensor : torch.Tensor, optional
            (mean, std) of each molecule-feature column, [len(molecule_features), 2];
            registered only when ``molecule_features`` is nonempty. Default: mean 0 and
            std 1 for every column.

        Raises
        ------
        AssertionError
            From :meth:`BaseGraphModel.get_data_stats`, if ``len(atom_features)`` is not
            one plus the number of recognized property names it contains.
        """
        super(MolecularCrystalModel, self).__init__()

        torch.manual_seed(seed)
        self.get_data_stats(atom_features,
                            molecule_features,
                            node_standardization_tensor,
                            graph_standardization_tensor)

        self.model = MolecularCrystalGraphModel(
            input_node_dim=self.n_atom_feats,
            num_mol_feats=self.n_mol_feats,
            output_dim=output_dim,
            seed=seed,
            concat_mol_to_node_dim=True,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
        )

    def forward(self, crystal_batch, return_dists=False, return_latent=False, force_edges_rebuild=False):
        """Featurize, standardize and encode a batch of molecular crystals.

        Overrides :meth:`BaseGraphModel.forward`. ``crystal_batch.x`` is overwritten with
        the standardized node features looked up from ``crystal_batch.z`` and, when
        molecule features are configured, ``crystal_batch.mol_x`` with the standardized
        per-crystal features. The batch's ``x``, ``pos``, ``batch``, ``ptr``, ``mol_x``,
        ``num_graphs``, ``aux_ind``, ``mol_ind`` and, if the attribute exists,
        ``edges_dict`` (else None) are passed to :meth:`MolecularCrystalGraphModel.forward`,
        which builds the edge dict with :func:`build_radial_graph` when it is None or
        ``force_edges_rebuild`` is True, provided the graph network has convolution layers
        or ``return_dists`` or ``force_edges_rebuild`` is True; an edge dict built there is
        not stored on ``crystal_batch``.

        Parameters
        ----------
        crystal_batch : MolCrystalData
            Crystal cluster batch carrying the attributes listed above (``z`` in place of
            ``x``) and the molecule-feature attributes named at construction. Modified in
            place as described above.
        return_dists : bool, optional
            Return the edge dict under ``'dists_dict'``. When it holds
            ``'edge_index_inter'``, intermolecular-distance keys are written into that
            dict, which is ``crystal_batch.edges_dict`` itself when that was supplied.
        return_latent : bool, optional
            Return the node features entering the graph network under
            ``'final_activation'``.
        force_edges_rebuild : bool, optional
            Rebuild the radial graph even when ``crystal_batch.edges_dict`` exists.

        Returns
        -------
        torch.Tensor or tuple
            The output, [num_graphs, output_dim]; ``(output, extra_outputs)`` when
            ``return_dists`` or ``return_latent`` is True. See
            :meth:`MolecularCrystalGraphModel.forward`.
        """
        # on the fly atom property embeddings
        crystal_batch = self.featurize_input_graph(crystal_batch)
        # on the fly input standardization
        crystal_batch = self.standardize(crystal_batch)

        return self.model(crystal_batch.x,
                          crystal_batch.pos,
                          crystal_batch.batch,
                          crystal_batch.ptr,
                          crystal_batch.mol_x,
                          crystal_batch.num_graphs,
                          crystal_batch.aux_ind,
                          crystal_batch.mol_ind,
                          edges_dict=crystal_batch.edges_dict if hasattr(crystal_batch, 'edges_dict') else None,
                          return_dists=return_dists,
                          return_latent=return_latent,
                          force_edges_rebuild=force_edges_rebuild,)


# class HierarchicalCrystalDiscriminator(nn.Module):
#     def __init__(self, seed, config, n_atom_types, input_depth):
#         '''
#         wrapper for molecule model, with appropriate I/O
#         '''
#         torch.manual_seed(seed)
#         super(HierarchicalCrystalDiscriminator, self).__init__()
#         self.conditioner = molecule_encoder
#         self.gnn = point_graph
#   self.crystal_graph=PeriodicCrystalGraph()
#     def symmetrize(self):
#     def forward(self, data,):
#         # if crystalline, encode the data object, then generate pattern it according to the crystal symmetry
#         # then evaluate material graph
#         # if it's some bulk, thing, just encode all the separate molecules and evaluate the resulting material graph
#         return self.model(data, return_dists=return_dists, return_latent=return_latent)
#


# deprecated
#
# class MolCrystal(BaseGraphModel):
#     def __init__(self, seed, config,
#                  atom_features: list,
#                  molecule_features: list,
#                  node_standardization_tensor: OptTensor = None,
#                  graph_standardization_tensor: OptTensor = None):
#         """
#         wrapper for molecule model, with appropriate I/O
#         """
#         super(MolCrystal, self).__init__()
#
#         torch.manual_seed(seed)
#         self.get_data_stats(atom_features,
#                             molecule_features,
#                             node_standardization_tensor,
#                             graph_standardization_tensor)
#
#         self.model = MoleculeGraphModel(
#             input_node_dim=self.n_atom_feats,
#             num_mol_feats=self.n_mol_feats,
#             output_dim=2 + 1,  # 2 for classification and 1 for distance regression
#             seed=seed,
#             graph_aggregator=config.graph_aggregator,
#             concat_pos_to_node_dim=False,
#             concat_mol_to_node_dim=True,
#             concat_crystal_to_node_dim=False,
#             activation=config.activation,
#             fc_config=config.fc,
#             graph_config=config.graph,
#             periodize_inside_nodes=True,
#             outside_convolution_type=config.periodic_convolution_type
#         )
#
#     def forward(self, data, return_dists=False, return_latent=False):
#         """overwrites base method"""
#         # on the fly atom property embeddings
#         data = self.featurize_input_graph(data)
#         # on the fly input standardization
#         data = self.standardize(data)
#
#         outputs = self.model(data, return_dists=return_dists, return_latent=return_latent)
#
#         if isinstance(outputs, tuple):  # if we have extra outputs, pick out the actual model output and adjust
#             model_outputs, extra_outputs = outputs
#             return (torch.cat([model_outputs[:, :2], F.softplus(model_outputs[:, -1, None])], dim=1), extra_outputs)
#
#         else:
#             return torch.cat([outputs[:, 0], F.softplus(outputs[:, 1])], dim=1)

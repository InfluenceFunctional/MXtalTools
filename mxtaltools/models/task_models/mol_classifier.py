"""Graph-model wrapper that featurizes and standardizes a molecular cluster batch and
classifies its molecules with a :class:`MoleculeClusterModel`, pooling nodes by molecule index."""
from typing import Tuple, Optional

import torch

from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.molecule_graph_model import MoleculeClusterModel


class MoleculeClusterClassifier(BaseGraphModel):
    """Per-molecule classifier over a molecular cluster.

    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator, then :meth:`BaseGraphModel.get_data_stats`, which registers the
    atom-property lookup table and the standardization buffers. ``self.model`` is a
    :class:`MoleculeClusterModel` with ``input_node_dim=len(atom_features)`` and
    ``num_mol_feats=0``, so the molecule features are computed and standardized in
    :meth:`forward` but not read by the model.

    Parameters
    ----------
    seed : int
        Seed for ``torch.manual_seed``; also passed to the cluster model.
    config : Namespace
        Model settings. ``config.activation``, ``config.fc`` and ``config.graph`` are
        passed to the cluster model as ``activation``, ``fc_config`` and
        ``graph_config``.
    output_dim : int
        Width of the per-molecule output.
    atom_features : list
        Atom feature names. The node features are the atomic number followed by those of
        ``'vdw_radii'``, ``'atom_weight'``, ``'electronegativity'``, ``'group'`` and
        ``'period'`` that appear in the list, in that fixed order.
    molecule_features : list
        Per-graph feature names; ``'num_atoms'``, ``'radius'`` and ``'mol_volume'`` are
        read from the batch attributes of the same name.
    node_standardization_tensor : torch.Tensor
        (mean, std) of each node-feature column, [len(atom_features), 2]. None gives mean
        0 and std 1 for every column.
    graph_standardization_tensor : torch.Tensor
        (mean, std) of each molecule-feature column, [len(molecule_features), 2];
        registered only when ``molecule_features`` is nonempty. None gives mean 0 and
        std 1 for every column.

    Raises
    ------
    AssertionError
        From :meth:`BaseGraphModel.get_data_stats`, if ``len(atom_features)`` is not one
        plus the number of recognized property names it contains.
    """

    def __init__(self,
                 seed,
                 config,
                 output_dim,
                 atom_features: list,
                 molecule_features: list,
                 node_standardization_tensor: torch.Tensor,
                 graph_standardization_tensor: torch.Tensor,
                 ):
        super(MoleculeClusterClassifier, self).__init__()

        torch.manual_seed(seed)
        self.get_data_stats(atom_features,
                            molecule_features,
                            node_standardization_tensor,
                            graph_standardization_tensor)

        self.model = MoleculeClusterModel(
            input_node_dim=self.n_atom_feats,
            num_mol_feats=0,
            output_dim=output_dim,
            seed=seed,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
        )

    def forward(self,
                data_batch,
                return_dists: bool = False,
                return_latent: bool = False,
                return_embedding: bool = False,
                ) -> Tuple[torch.Tensor, Optional[dict]]:
        """Featurize, standardize and classify the molecules of a cluster.

        ``data_batch.x`` is overwritten with the standardized node features looked up
        from ``data_batch.z`` and, when molecule features are configured,
        ``data_batch.mol_x`` with the standardized per-graph features. The batch's ``x``,
        ``pos``, ``ptr``, ``mol_x``, ``num_graphs``, ``mol_ind``, ``T_fc``,
        ``edge_index`` and ``edge_attr`` are passed to
        :meth:`MoleculeClusterModel.forward`. When that model needs edges, it uses
        ``edge_index`` and ``edge_attr`` if ``edge_index`` is not None and otherwise
        builds minimum-image edges from ``pos`` and ``T_fc`` (asserting
        ``num_graphs == 1``).

        Parameters
        ----------
        data_batch : MolCrystalData
            Cluster batch carrying the attributes listed above (``z`` in place of ``x``)
            and the molecule-feature attributes named at construction. Modified in place
            as described above.
        return_dists : bool, optional
            Return the edge dict under ``'dists_dict'`` (None when ``edge_index`` was
            given).
        return_latent : bool, optional
            Return the per-molecule activation entering the output layer under
            ``'final_activation'``.
        return_embedding : bool, optional
            Return the node embeddings from the graph network under
            ``'graph_embedding'``.

        Returns
        -------
        torch.Tensor or tuple
            The per-molecule output, [int(mol_ind.max()), output_dim], pooled with index
            ``mol_ind - mol_ind.min()``; this is one row per molecule when ``mol_ind``
            runs contiguously from 1. ``(output, extra_outputs)`` when any ``return_*``
            flag is True. See :meth:`MoleculeClusterModel.forward`.
        """
        # featurize atom properties on the fly
        data_batch = self.featurize_input_graph(data_batch)

        # standardize on the fly from model-attached statistics
        data_batch = self.standardize(data_batch)

        return self.model(data_batch.x,
                          data_batch.pos,
                          data_batch.ptr,
                          data_batch.mol_x,
                          data_batch.num_graphs,
                          data_batch.mol_ind,
                          data_batch.T_fc,
                          data_batch.edge_index,
                          data_batch.edge_attr,
                          return_dists=return_dists,
                          return_latent=return_latent,
                          return_embedding=return_embedding)


# deprecated
# class PolymorphClassifier(BaseGraphModel):
#     def __init__(self, seed, config,
#                  dataDims: dict,
#                  atom_features: list,
#                  molecule_features: list,
#                  node_standardization_tensor: torch.tensor,
#                  graph_standardization_tensor: torch.tensor,
#                  ):
#         super(PolymorphClassifier, self).__init__()
#
#         torch.manual_seed(seed)
#         self.get_data_stats(atom_features,
#                             molecule_features,
#                             node_standardization_tensor,
#                             graph_standardization_tensor)
#
#         self.model = MoleculeGraphModel(
#             input_node_dim=self.n_atom_feats,
#             num_mol_feats=0,
#             output_dim=dataDims['num_polymorphs'] + dataDims['num_topologies'],
#             seed=seed,
#             graph_aggregator='molwise',
#             activation=config.activation,
#             fc_config=config.fc,
#             graph_config=config.graph,
#             outside_convolution_type='none'
#         )
#
#     def forward(self, data, return_dists=False, return_latent=False, return_embedding=False,
#                 skip_standardization=False):
#         if not skip_standardization:
#             data = self.standardize(data)
#
#         return self.model(data,
#                           return_dists=return_dists,
#                           return_latent=return_latent,
#                           return_embedding=return_embedding)

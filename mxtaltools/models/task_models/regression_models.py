"""Graph-model wrapper that featurizes and standardizes a molecule batch and regresses one
scalar per graph with a :class:`ScalarMoleculeGraphModel`."""
from argparse import Namespace
from typing import Optional

import torch

from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.graph_models.molecule_graph_model import ScalarMoleculeGraphModel


class MoleculeScalarRegressor(BaseGraphModel):
    def __init__(self,
                 config: Namespace,
                 atom_features: list,
                 molecule_features: list,
                 node_standardization_tensor: Optional[torch.Tensor] = None,
                 graph_standardization_tensor: Optional[torch.Tensor] = None,
                 target_standardization_tensor: Optional[torch.Tensor] = None,
                 seed: int = 0
                 ):
        """Wrapper for a :class:`ScalarMoleculeGraphModel` with one output, with input
        featurization and standardization.

        Calls ``torch.manual_seed(seed)``, which reseeds torch's global random number
        generator, then :meth:`BaseGraphModel.get_data_stats`, which registers the
        atom-property lookup table and the standardization buffers. ``self.model`` is a
        :class:`ScalarMoleculeGraphModel` with ``input_node_dim=len(atom_features)``,
        ``num_mol_feats=len(molecule_features)``, ``output_dim=1`` and
        ``concat_mol_to_node_dim=True``.

        Parameters
        ----------
        config : Namespace
            Model settings. ``config.activation``, ``config.fc`` and ``config.graph`` are
            passed to the graph model as ``activation``, ``fc_config`` and
            ``graph_config``.
        atom_features : list
            Atom feature names. The node features are the atomic number followed by those
            of ``'vdw_radii'``, ``'atom_weight'``, ``'electronegativity'``, ``'group'`` and
            ``'period'`` that appear in the list, in that fixed order.
        molecule_features : list
            Per-graph feature names; ``'num_atoms'``, ``'radius'`` and ``'mol_volume'`` are
            read from the batch attributes of the same name.
        node_standardization_tensor : torch.Tensor, optional
            (mean, std) of each node-feature column, [len(atom_features), 2]. Default:
            mean 0 and std 1 for every column.
        graph_standardization_tensor : torch.Tensor, optional
            (mean, std) of each molecule-feature column, [len(molecule_features), 2];
            registered only when ``molecule_features`` is nonempty. Default: mean 0 and
            std 1 for every column.
        target_standardization_tensor : torch.Tensor, optional
            Entries ``[0]`` and ``[1]`` are registered as the ``target_mean`` and
            ``target_std`` buffers, which this class does not read. If None, both buffers
            are the 0-dim tensor 1.0 (the mean default is 1, not 0).
        seed : int, optional
            Seed for ``torch.manual_seed``; also passed to the graph model. Default 0.

        Raises
        ------
        AssertionError
            From :meth:`BaseGraphModel.get_data_stats`, if ``len(atom_features)`` is not
            one plus the number of recognized property names it contains.

        Notes
        -----
        ``forward`` is inherited from :meth:`BaseGraphModel.forward`: it overwrites the
        batch's ``x`` (and ``mol_x`` when molecule features are configured) with the
        standardized features; when ``edge_index`` is None or ``force_edges_rebuild`` is
        True it calls ``construct_intra_radial_graph`` on batches whose class name
        contains ``'crystal'`` and ``construct_radial_graph`` (which :class:`MolData` does
        not define) otherwise, with ``self.model.convolution_cutoff``; and it returns the
        output of
        :meth:`ScalarMoleculeGraphModel.forward`, [num_graphs, 1], or
        ``(output, extra_outputs)`` when ``return_dists`` or ``return_latent`` is True.
        No edge dict is passed to the graph model, so ``'dists_dict'`` is None.
        """
        super(MoleculeScalarRegressor, self).__init__()
        torch.manual_seed(seed)
        self.get_data_stats(atom_features,
                            molecule_features,
                            node_standardization_tensor,
                            graph_standardization_tensor)

        if target_standardization_tensor is not None:
            self.register_buffer('target_mean', target_standardization_tensor[0])
            self.register_buffer('target_std', target_standardization_tensor[1])
        else:
            self.register_buffer('target_mean', torch.ones(1)[0])
            self.register_buffer('target_std', torch.ones(1)[0])

        self.model = ScalarMoleculeGraphModel(
            input_node_dim=self.n_atom_feats,
            num_mol_feats=self.n_mol_feats,
            output_dim=1,
            seed=seed,
            concat_mol_to_node_dim=True,
            activation=config.activation,
            fc_config=config.fc,
            graph_config=config.graph,
        )

    # uses default forward method inherited from base class

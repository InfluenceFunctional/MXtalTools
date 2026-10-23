"""MLP regression heads applied to precomputed molecule embeddings: an equivariant head on
scalar and vector embeddings, and an invariant head on scalar embeddings."""
from typing import Optional

import torch

from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.modules.components import vectorMLP, scalarMLP


class EquivariantEmbeddingRegressor(BaseGraphModel):
    """Single-property prediction head for pretrained embeddings, with a scalar and a
    vector output track.

    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator. ``self.model`` is a :class:`vectorMLP` with scalar input width
    ``config.bottleneck_dim + conditions_dim``, vector input channels
    ``config.bottleneck_dim``, and ``num_targets`` scalar outputs and vector output
    channels. No featurization or standardization buffers are set up.

    Parameters
    ----------
    seed : int
        Seed for ``torch.manual_seed``; also passed to the MLP.
    config : Namespace
        MLP settings: ``num_layers``, ``hidden_dim``, ``norm``, ``dropout``,
        ``bottleneck_dim`` and ``vector_norm``.
    num_targets : int, optional
        Width of the scalar output and channel count of the vector output. Default 1.
    conditions_dim : int, optional
        Extra scalar input columns beyond ``config.bottleneck_dim``. Default 0.
    prediction_type : str, optional
        Stored as ``self.prediction_type``; not read by this class. Default
        ``'scalar'``.
    """

    def __init__(self, seed, config,
                 num_targets: int = 1,
                 conditions_dim: int = 0,
                 prediction_type: str = 'scalar'):
        super(EquivariantEmbeddingRegressor, self).__init__()

        torch.manual_seed(seed)

        self.prediction_type = prediction_type
        self.output_dim = int(1 * num_targets)

        # regression model
        self.model = vectorMLP(layers=config.num_layers,
                               filters=config.hidden_dim,
                               norm=config.norm,
                               dropout=config.dropout,
                               input_dim=config.bottleneck_dim + conditions_dim,
                               output_dim=self.output_dim,
                               vector_input_dim=config.bottleneck_dim,
                               vector_output_dim=self.output_dim,
                               seed=seed,
                               vector_norm=config.vector_norm
                               )

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the equivariant MLP to scalar and vector embeddings.

        No standardization is done; the inputs (raw outputs of an autoencoder model) are
        passed to :meth:`vectorMLP.forward` as given.

        Parameters
        ----------
        x : torch.Tensor
            Scalar inputs, [num_samples, bottleneck_dim + conditions_dim].
        v : torch.Tensor
            Vector inputs, [num_samples, 3, bottleneck_dim].

        Returns
        -------
        tuple of torch.Tensor
            ``(x_out, v_out)``: scalar outputs [num_samples, num_targets] and vector
            outputs [num_samples, 3, num_targets].
        """

        x, v = self.model(x=x, v=v)

        return x, v


class InvariantEmbeddingRegressor(BaseGraphModel):
    """Single-property prediction head for pretrained embeddings, scalar inputs and
    outputs only.

    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator. ``self.model`` is a :class:`scalarMLP` from
    ``config.bottleneck_dim + conditions_dim`` input columns to ``num_targets`` outputs.
    The buffers ``target_mean`` and ``target_std`` are registered but not read by this
    class.

    Parameters
    ----------
    seed : int
        Seed for ``torch.manual_seed``; also passed to the MLP.
    config : Namespace
        MLP settings: ``num_layers``, ``hidden_dim``, ``norm``, ``dropout`` and
        ``bottleneck_dim``.
    num_targets : int, optional
        Width of the output. Default 1.
    conditions_dim : int, optional
        Extra input columns beyond ``config.bottleneck_dim``. Default 0.
    target_standardization_tensor : torch.Tensor, optional
        Entries ``[0]`` and ``[1]`` are registered as the ``target_mean`` and
        ``target_std`` buffers. If None, both buffers are the 0-dim tensor 1.0 (the mean
        default is 1, not 0).
    """

    def __init__(self, seed, config,
                 num_targets: int = 1,
                 conditions_dim: int = 0,
                 target_standardization_tensor: Optional[torch.Tensor] = None,
                 ):
        super(InvariantEmbeddingRegressor, self).__init__()

        torch.manual_seed(seed)

        self.output_dim = int(1 * num_targets)

        if target_standardization_tensor is not None:
            self.register_buffer('target_mean', target_standardization_tensor[0])
            self.register_buffer('target_std', target_standardization_tensor[1])
        else:
            self.register_buffer('target_mean', torch.ones(1)[0])
            self.register_buffer('target_std', torch.ones(1)[0])

        # regression model
        self.model = scalarMLP(layers=config.num_layers,
                               filters=config.hidden_dim,
                               norm=config.norm,
                               dropout=config.dropout,
                               input_dim=config.bottleneck_dim + conditions_dim,
                               output_dim=self.output_dim,
                               seed=seed,
                               )

    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        """Apply the MLP to scalar embeddings.

        No standardization is done; the inputs (raw outputs of an autoencoder model) are
        passed to :meth:`scalarMLP.forward` as given, and the output is not
        de-standardized with ``target_mean`` or ``target_std``.

        Parameters
        ----------
        x : torch.Tensor
            Inputs, [num_samples, bottleneck_dim + conditions_dim].

        Returns
        -------
        torch.Tensor
            Outputs, [num_samples, num_targets].
        """

        x = self.model(x=x)

        return x

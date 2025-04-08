from typing import Optional

import torch

from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.modules.components import vectorMLP, scalarMLP


class EquivariantEmbeddingRegressor(BaseGraphModel):
    """single property prediction head for pretrained embeddings"""

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
        """no need to do standardization, inputs are raw outputs from autoencoder model
        """

        x, v = self.model(x=x, v=v)

        return x, v


class InvariantEmbeddingRegressor(BaseGraphModel):
    """single property prediction head for pretrained embeddings"""

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
        """no need to do standardization, inputs are raw outputs from autoencoder model
        """

        x = self.model(x=x)

        return x

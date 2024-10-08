import torch

from mxtaltools.models.graph_models.base_graph_model import BaseGraphModel
from mxtaltools.models.modules.components import vectorMLP


class EmbeddingRegressor(BaseGraphModel):
    """single property prediction head for pretrained embeddings"""

    def __init__(self, seed, config, num_targets: int = 1, conditions_dim: int = 0):
        super(EmbeddingRegressor, self).__init__()

        torch.manual_seed(seed)

        self.prediction_type = 'scalar'
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
                v: torch.Tensor) -> torch.Tensor:
        """no need to do standardization, inputs are raw outputs from autoencoder model
        """

        x, v = self.model(x=x, v=v)

        return x

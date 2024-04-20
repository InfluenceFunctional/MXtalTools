import torch

from mxtaltools.models.base_graph_model import BaseGraphModel
from mxtaltools.models.components import MLP


class EmbeddingRegressor(BaseGraphModel):
    """single property prediction head for pretrained embeddings"""

    def __init__(self, seed, config, num_targets: int = 1):
        super(EmbeddingRegressor, self).__init__()

        torch.manual_seed(seed)

        self.equivariant = config.equivariant
        self.prediction_type = 'scalar'
        self.output_dim = int(1 * num_targets)

        # regression model
        self.model = MLP(layers=config.num_layers,
                         filters=config.hidden_dim,
                         norm=config.norm,
                         dropout=config.dropout,
                         input_dim=config.bottleneck_dim,
                         output_dim=self.output_dim,
                         conditioning_dim=0,
                         seed=seed,
                         conditioning_mode=None,
                         equivariant=config.equivariant,
                         vector_norm=config.vector_norm if config.equivariant else None,
                         )

    def forward(self, x, v=None):
        """no need to do standardization, inputs are raw outputs from autoencoder model
        """

        if self.equivariant:
            x, v = self.model(x=x,
                              v=v,
                              )
        else:
            x = self.model(x)

        return x

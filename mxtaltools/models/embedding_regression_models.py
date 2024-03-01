import torch
import torch.nn as nn

from mxtaltools.models.components import MLP


class embedding_regressor(nn.Module):
    """single property prediction head for pretrained embeddings"""

    def __init__(self, seed, config, num_targets):
        super(embedding_regressor, self).__init__()

        self.equivariant = config.equivariant
        self.prediction_type = 'scalar'

        if self.prediction_type == 'scalar':
            self.output_dim = int(1 * num_targets)
        # elif prediction_type == 'vector':
        #     if self.equivariant:
        #         self.output_dim = int(1 * num_targets)
        #     else:
        #         self.output_dim = int(3 * num_targets)

        # graph size model
        self.model = MLP(layers=config.num_layers,
                         filters=config.depth,
                         norm=config.norm_mode,
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

        if self.equivariant:
            x, v = self.model(x=x,
                              v=v,
                              )
        else:
            x = self.model(x)

        #if self.prediction_type == 'scalar':
        return x
        # elif self.prediction_type == 'vector':  # todo rewrite
        #     if self.equivariant:
        #         return v.permute(0, 2, 1)
        #     else:
        #         return x.reshape(len(x), x.shape[-1] // 3, 3)

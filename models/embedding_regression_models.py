import torch
import torch.nn as nn

from models.components import MLP


class embedding_regressor(nn.Module):
    def __init__(self, seed, config):
        super(embedding_regressor, self).__init__()

        # graph size model
        self.model = MLP(layers=config.num_layers,
                         filters=config.depth,
                         norm=config.norm_mode,
                         dropout=config.dropout,
                         input_dim=config.embedding_depth,
                         output_dim=1,
                         conditioning_dim=0,
                         seed=seed,
                         conditioning_mode=None,
                         )

    def forward(self, embedding):
        return self.model(embedding)

import torch
import torch.nn as nn

from models.components import MLP


class embedding_regressor(nn.Module):
    """single property prediction head for pretrained embeddings"""

    def __init__(self, seed, config, prediction_type, embedding_type, num_targets):
        super(embedding_regressor, self).__init__()

        self.equivariant = config.equivariant
        self.embedding_type = embedding_type
        self.prediction_type = prediction_type

        if prediction_type == 'scalar':
            self.output_dim = int(1 * num_targets)
        elif prediction_type == 'vector':
            if self.equivariant:
                self.output_dim = int(1 * num_targets)
            else:
                self.output_dim = int(3 * num_targets)

        # graph size model
        self.model = MLP(layers=config.num_layers,
                         filters=config.depth,
                         norm=config.norm_mode,
                         dropout=config.dropout,
                         input_dim=config.embedding_depth,
                         output_dim=self.output_dim,
                         conditioning_dim=0,
                         seed=seed,
                         conditioning_mode=None,
                         equivariant=config.equivariant,
                         vector_norm=config.vector_norm if config.equivariant else False,
                         residue_v_to_s=True if config.equivariant else False,
                         )

    def forward(self, embedding):

        if self.equivariant:
            if self.embedding_type == 'equivariant':
                x, v = self.model(x=torch.linalg.norm(embedding, dim=1),
                                  v=embedding,
                                  )
            else:
                assert False, "Cannot do equivariant property prediction with non-equivariant embedding"
        else:
            x = self.model(torch.linalg.norm(embedding, dim=1))

        if self.prediction_type == 'scalar':
            return x
        elif self.prediction_type == 'vector':
            if self.equivariant:
                return v.permute(0, 2, 1)
            else:
                return x.reshape(len(x), x.shape[-1] // 3, 3)

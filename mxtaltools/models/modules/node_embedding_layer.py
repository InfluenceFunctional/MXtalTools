import torch
from torch import nn as nn, Tensor


class EmbeddingBlock(torch.nn.Module):
    def __init__(self,
                 init_node_embedding_dim: int,
                 num_input_classes: int,
                 num_scalar_input_features: int,
                 atom_type_embedding_dim: int):
        super(EmbeddingBlock, self).__init__()

        self.embeddings = nn.Embedding(num_input_classes + 1, atom_type_embedding_dim)
        self.linear = nn.Linear(atom_type_embedding_dim + num_scalar_input_features - 1, init_node_embedding_dim)

    def forward(self,
                x: Tensor) -> Tensor:

        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        # always embed the first dimension only (by convention, atomic number)
        embedding = self.embeddings(x[:, 0].long())

        return self.linear(torch.cat([embedding, x[:, 1:]], dim=-1))

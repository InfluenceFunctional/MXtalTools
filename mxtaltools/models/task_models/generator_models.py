import os

import numpy as np
import torch
from torch import nn as nn

from mxtaltools.common.geometry_utils import enforce_crystal_system
from mxtaltools.common.utils import softplus_shift
from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.models.modules.components import scalarMLP
from mxtaltools.models.utils import enforce_1d_bound


class CrystalGenerator(nn.Module):
    def __init__(self,
                 seed: int,
                 config,
                 embedding_dim: int,
                 sym_info: dict,
                 z_prime=1):
        super(CrystalGenerator, self).__init__()

        torch.manual_seed(seed)
        self.symmetries_dict = sym_info
        self.register_buffer('SG_FEATURE_TENSOR', SG_FEATURE_TENSOR.clone())  # store space group information

        self.model = scalarMLP(layers=config.num_layers,
                               filters=config.hidden_dim,
                               norm=config.norm,
                               dropout=config.dropout,
                               # scalar embedding, prior, target deviation,
                               # sg information, vector embedding,
                               input_dim=embedding_dim + 12 + 1 + 237 + embedding_dim * 3,
                               output_dim=6 + z_prime * 6,
                               )

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                sg_ind_list: torch.LongTensor,
                prior: torch.Tensor,
                step_size: torch.Tensor,
                ) -> torch.Tensor:
        x_w_sg = torch.cat([x, self.SG_FEATURE_TENSOR[sg_ind_list], prior, step_size], dim=1)
        x_w_v = torch.cat([x_w_sg, v.reshape(v.shape[0], v.shape[1] * v.shape[2])], dim=1)
        delta = self.model(x=x_w_v)
        raw_sample = prior + delta * step_size

        return raw_sample



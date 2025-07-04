import torch
import torch.nn.functional as F
from torch import nn as nn

from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.models.modules.components import scalarMLP


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
                               input_dim=embedding_dim + 12 + 4 + 237 + embedding_dim * 3,
                               output_dim=6 + z_prime * 6 + 2,
                               )

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                sg_ind_list: torch.LongTensor,
                prior: torch.Tensor,
                max_length_step: torch.Tensor,
                max_angle_step: torch.Tensor,
                max_position_step: torch.Tensor,
                max_orientation_step: torch.Tensor,
                ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Step format is x1 = x0 + max_step_size * unit_delta * sigmoid(||delta||)
        Yielding an adjustable actual step size on 0-max_step_size
        """
        x_w_sg = torch.cat([
            x,
            self.SG_FEATURE_TENSOR[sg_ind_list],
            prior,
            max_length_step,
            max_angle_step,
            max_position_step,
            max_orientation_step],
           dim=1)
        x_w_v = torch.cat([x_w_sg, v.reshape(v.shape[0], v.shape[1] * v.shape[2])], dim=1)
        outputs = self.model(x=x_w_v)

        return outputs[:, :12], outputs[:, 12], outputs[:, 13]



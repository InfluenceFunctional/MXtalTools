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
                               output_dim=6 + z_prime * 6,
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
                ) -> torch.Tensor:
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
        length_delta, angle_delta, position_delta, orientation_delta = self.model(x=x_w_v).split(3, dim=1)

        eps = 1e-4  # account for possible instability here
        length_delta_norm = length_delta.norm(dim=1, keepdim=True).clamp(min=eps)
        angle_delta_norm = angle_delta.norm(dim=1, keepdim=True).clamp(min=eps)
        position_delta_norm = position_delta.norm(dim=1, keepdim=True).clamp(min=eps)
        orientation_delta_norm = orientation_delta.norm(dim=1, keepdim=True).clamp(min=eps)

        normed_length_delta = length_delta / length_delta_norm
        normed_angle_delta = angle_delta / angle_delta_norm
        normed_position_delta = position_delta / position_delta_norm
        normed_orientation_delta = orientation_delta / orientation_delta_norm

        step = torch.cat([
            max_length_step * F.sigmoid(length_delta_norm) * normed_length_delta,
            max_angle_step * F.sigmoid(angle_delta_norm) * normed_angle_delta,
            max_position_step * F.sigmoid(position_delta_norm) * normed_position_delta,
            max_orientation_step * F.sigmoid(orientation_delta_norm) * normed_orientation_delta,
        ], dim=-1)

        return step



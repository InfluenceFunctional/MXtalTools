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

        path = os.path.join(os.path.dirname(__file__), '../../constants/prior_stds.npy')
        self.register_buffer('stds', torch.tensor(np.load(path, allow_pickle=True), dtype=torch.float32))

        path = os.path.join(os.path.dirname(__file__), '../../constants/prior_means.npy')
        self.register_buffer('means', torch.tensor(np.load(path, allow_pickle=True), dtype=torch.float32))

        self.register_buffer('SG_FEATURE_TENSOR', SG_FEATURE_TENSOR.clone())  # store space group information

        # generator model
        # self.model = vectorMLP(layers=config.num_layers,
        #                        filters=config.hidden_dim,
        #                        norm=config.norm,
        #                        dropout=config.dropout,
        #                        # embedding, prior, target deviation, sg information, prior scaling
        #                        input_dim=embedding_dim + 9 + 1 + 237 + 12,
        #                        output_dim=6 + z_prime * 3,
        #                        vector_input_dim=embedding_dim + z_prime + 3,
        #                        vector_output_dim=z_prime,
        #                        seed=seed,
        #                        vector_norm=config.vector_norm
        #                        )
        self.model = scalarMLP(layers=config.num_layers,
                               filters=config.hidden_dim,
                               norm=config.norm,
                               dropout=config.dropout,
                               # scalar embedding, prior, target deviation,
                               # sg information, prior scaling, vector embedding,
                               # reference vector
                               input_dim=embedding_dim + 12 + 1 + 237 + 12 + embedding_dim * 3 + 9,
                               output_dim=6 + z_prime * 6,
                               )

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                sg_ind_list: torch.LongTensor,
                prior: torch.Tensor,
                ) -> torch.Tensor:
        x_w_sg = torch.cat([x, self.SG_FEATURE_TENSOR[sg_ind_list]], dim=1)

        #x, v = self.model(x=x_w_sg, v=v)
        #raw_sample = torch.cat([x, v[:, :, 0]], dim=-1) * self.stds[sg_ind_list] + self.means[sg_ind_list]

        x_w_v = torch.cat([x_w_sg, v.reshape(v.shape[0], v.shape[1] * v.shape[2])], dim=1)
        delta = self.model(x=x_w_v)
        raw_sample = prior + delta

        sample = self.cleanup_sample(raw_sample, sg_ind_list)

        return sample

    def cleanup_sample(self, raw_sample, sg_ind_list):
        # force outputs into physical ranges
        # cell lengths have to be positive nonzero
        cell_lengths = softplus_shift(raw_sample[:, :3])
        # range from (0,pi) with 20% padding to prevent too-skinny cells
        cell_angles = enforce_1d_bound(raw_sample[:, 3:6], x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2,
                                       mode='hard')
        # positions must be on 0-1
        mol_positions = enforce_1d_bound(raw_sample[:, 6:9], x_span=0.5, x_center=0.5, mode='hard')
        # for now, just enforce vector norm
        rotvec = raw_sample[:, 9:12]
        norm = torch.linalg.norm(rotvec, dim=1)
        new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode='hard')  # MUST be nonzero
        new_rotvec = rotvec / norm[:, None] * new_norm[:, None]
        # invert_inds = torch.argwhere(new_rotvec[:, 2] < 0)
        # new_rotvec[invert_inds] = -new_rotvec[invert_inds]  # z direction always positive
        # force cells to conform to crystal system
        cell_lengths, cell_angles = enforce_crystal_system(cell_lengths, cell_angles, sg_ind_list,
                                                           self.symmetries_dict)
        sample = torch.cat((cell_lengths, cell_angles, mol_positions, new_rotvec), dim=-1)
        return sample


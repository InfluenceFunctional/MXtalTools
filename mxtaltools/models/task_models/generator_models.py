"""MLP model that maps molecule embeddings, space-group features, a prior sample and
step-size inputs to raw outputs returned in three parts."""
import torch
import torch.nn.functional as F
from torch import nn as nn

from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.models.modules.components import scalarMLP


class CrystalGenerator(nn.Module):
    """Scalar MLP over molecule embeddings, space-group features, a prior sample and four
    step-size inputs.

    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator, and registers a clone of ``SG_FEATURE_TENSOR`` [231, 237] as the
    buffer ``SG_FEATURE_TENSOR``. ``self.model`` is a :class:`scalarMLP` with input width
    ``embedding_dim + 12 + 4 + 237 + 3 * embedding_dim`` and output width
    ``6 + 6 * z_prime + 2``. ``seed`` is not passed to the MLP, whose constructor calls
    ``torch.manual_seed(0)`` (its default) before building its layers.

    Parameters
    ----------
    seed : int
        Seed for ``torch.manual_seed``.
    config : Namespace
        MLP settings: ``num_layers``, ``hidden_dim``, ``norm`` and ``dropout``.
    embedding_dim : int
        Width of the scalar embedding and channel count of the vector embedding.
    sym_info : dict
        Stored as ``self.symmetries_dict``; not read by this class.
    z_prime : int, optional
        Sets the MLP output width, ``6 + 6 * z_prime + 2``. Default 1.
    """

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
        """Apply the MLP and split its output into three parts.

        The MLP input is the column concatenation of ``x``,
        ``SG_FEATURE_TENSOR[sg_ind_list]``, ``prior``, ``max_length_step``,
        ``max_angle_step``, ``max_position_step``, ``max_orientation_step`` and ``v``
        reshaped to ``[num_samples, v.shape[1] * v.shape[2]]``; its total width must
        equal the constructed input width. The outputs are returned raw: no step is
        formed, and the step-size tensors enter only as input features.

        Parameters
        ----------
        x : torch.Tensor
            Scalar embeddings, [num_samples, embedding_dim].
        v : torch.Tensor
            Vector embeddings, [num_samples, 3, embedding_dim].
        sg_ind_list : torch.LongTensor
            Space-group number of each sample, [num_samples]; indexes the rows of
            ``SG_FEATURE_TENSOR`` (1-230; row 0 is all zeros).
        prior : torch.Tensor
            Prior sample, 2-D with ``num_samples`` rows.
        max_length_step : torch.Tensor
            Step-size input, 2-D with ``num_samples`` rows.
        max_angle_step : torch.Tensor
            Step-size input, 2-D with ``num_samples`` rows.
        max_position_step : torch.Tensor
            Step-size input, 2-D with ``num_samples`` rows.
        max_orientation_step : torch.Tensor
            Step-size input, 2-D with ``num_samples`` rows.

        Returns
        -------
        tuple of torch.Tensor
            ``(outputs[:, :12], outputs[:, 12], outputs[:, 13])``: [num_samples, 12],
            [num_samples] and [num_samples]. Output columns beyond 13 (present when
            ``z_prime > 1``) are not returned.

        Notes
        -----
        The constructed input width reserves 16 columns (``12 + 4``) for ``prior`` and
        the four step-size tensors together.
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



"""Hard-coded summary statistics of six-component unit-cell parameter vectors.

All three names are float32 tensors created at import time on torch's default
device (no ``device`` argument is given). The module does not state the
normalisation of the length components or the unit of entries 3-5, whose
means lie near pi/2.

cell_means : torch.Tensor
    [6] per-component means. Entries 0-2 are the components covered by
    ``cell_lengths_cov_mat``; entries 3-5 follow.
cell_stds : torch.Tensor
    [6] per-component standard deviations, in the same order as
    ``cell_means``; ``cell_stds[:3] ** 2`` matches the diagonal of
    ``cell_lengths_cov_mat`` to the stored precision.
cell_lengths_cov_mat : torch.Tensor
    [3, 3] symmetric covariance matrix of entries 0-2.
"""
import torch
# hardcoded CSD stats
cell_means = torch.tensor(
    [1.0411, 1.1640, 1.4564,
     1.5619, 1.5691, 1.5509],  # use triclinic
    dtype=torch.float32)
cell_stds = torch.tensor(
    [0.3846, 0.4280, 0.4864,
     0.2363, 0.2046, 0.2624],
    dtype=torch.float32)

cell_lengths_cov_mat = torch.tensor([
    [0.1479, -0.0651, -0.0670],
    [-0.0651, 0.1832, -0.1050],
    [-0.0670, -0.1050, 0.2366]],
    dtype=torch.float32)

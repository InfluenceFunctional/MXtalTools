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

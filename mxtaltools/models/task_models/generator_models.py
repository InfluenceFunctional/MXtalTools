import torch
import os
import numpy as np
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Uniform, LogNormal

from mxtaltools.constants.asymmetric_units import asym_unit_dict
from mxtaltools.models.modules.components import vectorMLP
from mxtaltools.models.utils import enforce_crystal_system, enforce_1d_bound, clean_cell_params


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

        # generator model
        self.model = vectorMLP(layers=config.num_layers,
                               filters=config.hidden_dim,
                               norm=config.norm,
                               dropout=config.dropout,
                               input_dim=embedding_dim + 9 * z_prime + 2,
                               output_dim=6 + z_prime * 3,
                               vector_input_dim=embedding_dim + z_prime + 3,
                               vector_output_dim=z_prime,
                               conditioning_dim=0,
                               seed=seed,
                               vector_norm=config.vector_norm
                               )

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                sg_ind_list: torch.LongTensor,
                return_raw_sample=False) -> torch.Tensor:

        x, v = self.model(x=x, v=v)

        raw_sample = torch.cat([x, v[:, :, 0]], dim=-1) * self.stds[sg_ind_list] + self.means[sg_ind_list]
        if return_raw_sample:
            sample = raw_sample
        else:
            # cleanup outputs

            # cell lengths have to be positive nonzero
            cell_lengths = F.softplus(raw_sample[:, :3] - 0.1) + 0.1
            # range from (0,pi) with 20% padding to prevent too-skinny cells
            cell_angles = enforce_1d_bound(raw_sample[:, 3:6], x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2,
                                           mode='soft')
            # positions must be on 0-1
            mol_positions = enforce_1d_bound(raw_sample[:, 6:9], x_span=0.5, x_center=0.5, mode='soft')
            # for now, just enforce vector norm
            rotvec = raw_sample[:, 9:12]
            norm = torch.linalg.norm(rotvec, dim=1)
            new_norm = enforce_1d_bound(norm, x_span=0.99 * torch.pi, x_center=torch.pi, mode='soft')  # MUST be nonzero
            new_rotvec = rotvec / norm[:, None] * new_norm[:, None]
            new_rotvec[:, 2] = F.softplus(new_rotvec[:, 2] - 0.01) + 0.01  # z direction always positive

            # force cells to conform to crystal system
            cell_lengths, cell_angles = enforce_crystal_system(cell_lengths, cell_angles, sg_ind_list,
                                                               self.symmetries_dict)
            sample = torch.cat((cell_lengths, cell_angles, mol_positions, new_rotvec), dim=-1)

        return sample


class IndependentGaussianGenerator(nn.Module):
    def __init__(self, input_dim, means, stds, sym_info, device, cov_mat=None):
        super(IndependentGaussianGenerator, self).__init__()

        self.device = device
        self.input_dim = input_dim
        means = torch.Tensor(means)
        stds = torch.Tensor(stds)

        self.register_buffer('means', torch.Tensor(means))
        self.register_buffer('stds', torch.Tensor(stds))
        self.register_buffer('fixed_norms', torch.Tensor(means))
        self.register_buffer('fixed_stds', torch.Tensor(stds))

        self.symmetries_dict = sym_info
        # initialize asymmetric unit dict
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key])  # .to(self.device)

        if cov_mat is not None:
            pass
        else:
            cov_mat = torch.diag(torch.Tensor(stds).pow(2))

        try:
            self.prior = MultivariateNormal(means, torch.Tensor(cov_mat))  # apply standardization
        except ValueError:  # for some datasets (e.g., all tetragonal space groups) the covariance matrix is ill conditioned, so we throw away off diagonals (mostly unimportant)
            self.prior = MultivariateNormal(loc=means,
                                            covariance_matrix=torch.eye(12, dtype=torch.float32) * torch.Tensor(
                                                cov_mat).diagonal())

    def forward(self, num_samples, data=None, sg_ind=None):
        """
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        """
        if data is None:
            pass
        else:
            sg_ind = data.sg_ind

        samples = self.prior.sample((num_samples,))  # samples in the destandardied 'real' basis
        final_samples = clean_cell_params(samples, sg_ind, self.means, self.stds,
                                          self.symmetries_dict, self.asym_unit_dict,
                                          rescale_asymmetric_unit=True, destandardize=False, mode='soft').to(
            self.device)

        return final_samples

    def backward(self, samples):
        return samples * self.stds + self.means

    def score(self, samples):
        return self.prior.log_prob(samples)


class GeneratorPrior(nn.Module):
    """
    angle means and stds
    {'cubic': [tensor([1.5708, 1.5708, 1.5708]), tensor([0., 0., 0.])],
     'hexagonal': [tensor([1.5708, 1.5708, 2.0944]), tensor([0., 0., 0.])],
     'monoclinic': [tensor([1.5708, 1.7693, 1.5713]), tensor([0.0020, 0.1465, 0.0163])],
     'orthorhombic': [tensor([1.5708, 1.5708, 1.5708]), tensor([0.0001, 0.0001, 0.0000])],
     'tetragonal': [tensor([1.5708, 1.5708, 1.5708]), tensor([0., 0., 0.])],
     'triclinic': [tensor([1.5619, 1.5691, 1.5509]), tensor([0.2363, 0.2046, 0.2624])]} <- use this one for now

    length_stds
    tensor([0.5163, 0.5930, 0.6284])
    length_means
    tensor([1.2740, 1.4319, 1.7752])
    """

    def __init__(self, sym_info, device):
        super(GeneratorPrior, self).__init__()

        self.device = device
        self.symmetries_dict = sym_info
        path = os.path.join(os.path.dirname(__file__), '../../constants/prior_norm_factors.npy')
        self.norm_factors = torch.tensor(np.load(path, allow_pickle=True), dtype=torch.float32, device=device)
        # initialize asymmetric unit dict
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key])  # .to(self.device)

        cell_means = torch.tensor([1.2740, 1.4319, 1.7752, 1.5619, 1.5691, 1.5509], dtype=torch.float32)
        cell_stds = torch.tensor([0.5163, 0.5930, 0.6284, 0.2363, 0.2046, 0.2624], dtype=torch.float32)

        self.lengths_prior = MultivariateNormal(torch.log(cell_means[:3]), torch.eye(3) * cell_stds[:3])  # apply standardization
        self.angles_prior = MultivariateNormal(cell_means[3:], torch.eye(3) * cell_stds[3:])  # apply standardization

        self.pose_prior = Uniform(0, 1)

    def sample_poses(self, num_samples, z_prime=1):
        """
        prior samples are xyz (fractional positions defined on 0-1)

        theta, phi, r (orientation angles defined on [0,pi/2], [-pi,pi], [0, 2*pi]
        also, the theta parameter has to be rescaled as it's not actually uniform

        To leverage equivariance of prediction model, we will sample instead directly
        the rotation vector, (ijk), conditioned to generate the appropriate statistics

        Parameters
        ----------
        num_samples

        Returns
        -------

        """
        positions = self.pose_prior.sample((num_samples, 3))

        # random directions on the sphere, getting naturally the correct distribution of theta, phi
        random_vectors = torch.randn(size=(num_samples, 3))

        # set norms uniformly between 0-2pi
        norms = random_vectors.norm(dim=1)
        applied_norms = (torch.rand(num_samples) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
        random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]

        # restrict theta to upper half-sphere (positive z)
        random_vectors[:, 2] = torch.abs(random_vectors[:, 2])

        # # rescale
        # samples[:, 4] = samples[:, 4] * 2 * torch.pi - torch.pi
        # samples[:, 5] *= 2 * torch.pi
        #
        # # theta is special - we approximate it by
        # theta = -torch.abs(torch.randn(num_samples)) / (torch.pi / 2) + torch.pi / 2
        # #theta = theta.clip(min=0, max=torch.pi/2)

        return torch.cat((positions, random_vectors), dim=1)

    def sample_cell_vectors(self, num_samples):
        return torch.cat([torch.exp(self.lengths_prior.sample((num_samples,))),
                          self.angles_prior.sample((num_samples,))],dim=1)

    def forward(self, num_samples, sg_ind_list):
        """
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        """
        cell_samples = self.sample_cell_vectors(num_samples)
        cell_lengths, cell_angles = cell_samples[:, 0:3], cell_samples[:, 3:6]
        pose_params = self.sample_poses(num_samples)

        # enforce 'hard' bounds
        # harshly enforces positive nonzero
        cell_lengths = torch.maximum(F.relu(cell_lengths), 0.1 * torch.ones_like(cell_lengths))
        # range from (0,pi) with 20% padding to prevent too-skinny cells
        cell_angles = enforce_1d_bound(cell_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2, mode='hard')

        # force cells to conform to crystal system
        cell_lengths, cell_angles = enforce_crystal_system(cell_lengths, cell_angles, sg_ind_list, self.symmetries_dict)

        return torch.cat([cell_lengths, cell_angles, pose_params], dim=1)

    def generate_norming_factors(self):
        norms = np.zeros((231, 12))
        stds = np.zeros_like(norms)
        means = np.zeros_like(stds)
        for ind in range(1, 231):
            p1 = self.generator_prior(10000, ind * torch.ones(10000))
            p2 = self.generator_prior(10000, ind * torch.ones(10000))

            d1 = torch.abs(p1 - p2)
            scale = d1.mean(0)

            norms[ind] = scale.detach().numpy()
            stds[ind] = p1.std(0).detach().numpy()
            means[ind] = p1.mean(0).detach().numpy()

        np.save('prior_norm_factors', norms)
        np.save('prior_stds', stds)
        np.save('prior_means', means)

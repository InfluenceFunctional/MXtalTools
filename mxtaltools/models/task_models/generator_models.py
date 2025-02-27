import os

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import MultivariateNormal, Uniform

from mxtaltools.common.utils import softplus_shift
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.space_group_feature_tensor import SG_FEATURE_TENSOR
from mxtaltools.models.modules.components import scalarMLP
from mxtaltools.models.utils import enforce_1d_bound, clean_cell_params
from mxtaltools.common.geometry_utils import enforce_crystal_system, sample_random_valid_rotvecs, \
    enforce_crystal_system2


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


class CSDPrior(nn.Module):
    """
    angle means and stds
    {'cubic': [tensor([1.5708, 1.5708, 1.5708]), tensor([0., 0., 0.])],
     'hexagonal': [tensor([1.5708, 1.5708, 2.0944]), tensor([0., 0., 0.])],
     'monoclinic': [tensor([1.5708, 1.7693, 1.5713]), tensor([0.0020, 0.1465, 0.0163])],
     'orthorhombic': [tensor([1.5708, 1.5708, 1.5708]), tensor([0.0001, 0.0001, 0.0000])],
     'tetragonal': [tensor([1.5708, 1.5708, 1.5708]), tensor([0., 0., 0.])],
     'triclinic': [tensor([1.5619, 1.5691, 1.5509]), tensor([0.2363, 0.2046, 0.2624])]} <- use this one for now
    """

    def __init__(self, sym_info, device, cell_means, cell_stds, lengths_cov_mat):
        super(CSDPrior, self).__init__()

        self.device = device
        self.symmetries_dict = sym_info
        path = os.path.join(os.path.dirname(__file__), '../../constants/prior_norm_factors.npy')
        self.norm_factors = torch.tensor(np.load(path, allow_pickle=True), dtype=torch.float32, device=device)
        # initialize asymmetric unit dict
        self.asym_unit_dict = ASYM_UNITS.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key])  # .to(self.device)

        #cell_means = torch.tensor(cell_means, dtype=torch.float32, device='cpu')
        #cell_stds = torch.tensor(cell_stds, dtype=torch.float32, device='cpu')
        #lengths_cov_mat = torch.tensor(lengths_cov_mat, dtype=torch.float32, device='cpu')
        #print("Using hardcoded CSD statistics for prior!")
        cell_means = torch.tensor(
            [1.0411, 1.1640, 1.4564,
             1.5619, 1.5691, 1.5509],  # use triclinic
            dtype=torch.float32)
        cell_stds = torch.tensor(
            [0.3846, 0.4280, 0.4864,
             0.2363, 0.2046, 0.2624],
            dtype=torch.float32)

        lengths_cov_mat = torch.tensor([
            [0.1479, -0.0651, -0.0670],
            [-0.0651, 0.1832, -0.1050],
            [-0.0670, -0.1050, 0.2366]],
            dtype=torch.float32)

        self.lengths_prior = MultivariateNormal(cell_means[:3], lengths_cov_mat)  # apply standardization
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
        return torch.cat([self.lengths_prior.sample((num_samples,)),
                          self.angles_prior.sample((num_samples,))], dim=1)

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
        cell_lengths = softplus_shift(cell_lengths)  # very gently enforce positive
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

            norms[ind] = scale.detach().numpy()  # todo normalize this across DoF as well
            stds[ind] = p1.std(0).detach().numpy()
            means[ind] = p1.mean(0).detach().numpy()

        np.save('prior_norm_factors', norms)
        np.save('prior_stds', stds)
        np.save('prior_means', means)


class NewCSDPrior(nn.Module):
    """
    sample from the general distribution of the CSD for Z'=1
    """
    def __init__(self, sym_info, device):
        super(NewCSDPrior, self).__init__()

        from mxtaltools.constants.csd_stats import cell_means, cell_stds, cell_lengths_cov_mat
        self.device = device
        self.symmetries_dict = sym_info
        path = os.path.join(os.path.dirname(__file__), '../../constants/prior_norm_factors.npy')
        self.norm_factors = torch.tensor(np.load(path, allow_pickle=True), dtype=torch.float32, device=device)

        # initialize asymmetric unit dict
        self.asym_unit_dict = ASYM_UNITS.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key])  # .to(self.device)

        self.lengths_prior = MultivariateNormal(cell_means[:3], cell_lengths_cov_mat)  # apply standardization
        self.angles_prior = MultivariateNormal(cell_means[3:], torch.eye(3) * cell_stds[3:])  # apply standardization
        self.pose_prior = Uniform(0, 1)

    def sample_poses(self, num_samples):
        """
        prior samples are xyz (fractional positions defined on 0-1)
        combined with r*(ijk) the rotation matrix defining the orientation of the molecule

        To leverage equivariance of prediction model, we will sample instead directly
        the rotation vector, (ijk), conditioned to generate the appropriate statistics

        Parameters
        ----------
        num_samples

        Returns
        -------

        """
        positions = self.pose_prior.sample((num_samples, 3))
        rotvecs = sample_random_valid_rotvecs(num_samples)

        return torch.cat((positions, rotvecs), dim=1)

    def sample_cell_vectors(self, num_samples):
        """
        cell lengths are sampled in the normed basis
        """
        return torch.cat([self.lengths_prior.sample((num_samples,)),
                          self.angles_prior.sample((num_samples,))], dim=1)

    def forward(self, num_samples, sg_ind_list):
        """
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        """
        cell_samples = self.sample_cell_vectors(num_samples)
        cell_lengths, cell_angles = cell_samples[:, 0:3], cell_samples[:, 3:6]
        pose_params = self.sample_poses(num_samples)

        # very gently enforce positive cell lengths
        cell_lengths = softplus_shift(cell_lengths)
        # range from (0,pi) with 20% padding to prevent too-skinny cells
        cell_angles = enforce_1d_bound(cell_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2, mode='hard')

        # force cells to conform to crystal system
        lattices = [self.symmetries_dict['lattice_type'][int(sg_ind_list[n])] for n in range(len(sg_ind_list))]

        cell_lengths, cell_angles = enforce_crystal_system2(cell_lengths, cell_angles, lattices)

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

            norms[ind] = scale.detach().numpy()  # todo normalize this across DoF as well
            stds[ind] = p1.std(0).detach().numpy()
            means[ind] = p1.mean(0).detach().numpy()

        np.save('prior_norm_factors', norms)
        np.save('prior_stds', stds)
        np.save('prior_means', means)


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
        self.asym_unit_dict = ASYM_UNITS.copy()
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

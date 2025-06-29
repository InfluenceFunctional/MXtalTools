import torch
from torch import nn as nn

from mxtaltools.common.geometry_utils import sph2rotvec, rotvec2sph


class AunitTransform(nn.Module):
    def __init__(self, asym_unit_dict):
        super().__init__()
        self.asym_unit_dict = asym_unit_dict

    def forward(self, cell_parameters, sg_inds):
        cell_lengths, cell_angles, cell_centroids, cell_orientations = cell_parameters.split(3, dim=1)
        auvs = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in sg_inds])
        # aunit_lengths = cell_lengths * auvs
        aunit_centroids = cell_centroids / auvs

        return torch.cat([
            cell_lengths,
            cell_angles,
            aunit_centroids,
            cell_orientations
        ], dim=1)

    def inverse(self, cell_parameters, sg_inds):
        cell_lengths, cell_angles, aunit_centroids, cell_orientations = cell_parameters.split(3, dim=1)
        auvs = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in sg_inds])
        #cell_lengths = aunit_lengths / auvs
        cell_centroids = aunit_centroids * auvs

        return torch.cat([
            cell_lengths,
            cell_angles,
            cell_centroids,
            cell_orientations
        ], dim=1)


class NiggliTransform(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        self.eps = 1e-6

    def forward(self, aunit_params, mol_radii):
        """
        Reduce physical aunit parameters to a niggli-normed basis
        """
        a, b, c, al, be, ga, u, v, w, x, y, z = aunit_params.split(1, dim=1)

        c_normed = c / 2 / mol_radii.clip(min=1.0)[:, None]

        # rescale a and b
        a_scale = (a / b)
        b_scale = (b / c)
        # get maximum cosine values
        al_cos_max, be_cos_max, ga_cos_max = self.get_max_cos(a, b, c)

        al_cos = torch.cos(al).clip(min=0, max=1)
        be_cos = torch.cos(be).clip(min=0, max=1)
        ga_cos = torch.cos(ga).clip(min=0, max=1)

        # scale actual cosines against their maxima
        al_scaled = (al_cos / al_cos_max)
        be_scaled = (be_cos / be_cos_max)
        ga_scaled = (ga_cos / ga_cos_max)

        return torch.cat([
            a_scale, b_scale, c_normed,
            al_scaled, be_scaled, ga_scaled,
            u, v, w, x, y, z
        ], dim=1)

    def get_max_cos(self, a, b, c):
        al_cos_max = (b / 2 / c)
        be_cos_max = (a / 2 / c)
        ga_cos_max = (a / 2 / b)
        return al_cos_max, be_cos_max, ga_cos_max

    def inverse(self, niggli_params, mol_radii):
        (a_scale, b_scale, c_normed,
         al_scaled, be_scaled, ga_scaled,
         u, v, w,
         x, y, z) = niggli_params.split(1, dim=1)
        # denormalize c
        c = c_normed * 2 * mol_radii[:, None]
        # descale a and b
        b = b_scale * c
        a = a_scale * b

        # descale the cosines
        al_cos_max, be_cos_max, ga_cos_max = self.get_max_cos(a, b, c)

        al_cos = al_scaled * al_cos_max
        be_cos = be_scaled * be_cos_max
        ga_cos = ga_scaled * ga_cos_max

        # retrieve the angles
        # for acute niggli cells the minimum cos value is 0
        al = torch.arccos(al_cos.clip(self.eps, 1 - self.eps))
        be = torch.arccos(be_cos.clip(self.eps, 1 - self.eps))
        ga = torch.arccos(ga_cos.clip(self.eps, 1 - self.eps))

        return torch.cat(
            [a, b, c,
             al, be, ga,
             u, v, w,
             x, y, z],
            dim=1)


class BoundedTransform(nn.Module):
    def __init__(self, min_val, max_val, slope=1.0, eps=1e-6, bias: float = 0.0):
        super().__init__()
        self.register_buffer('min_val', torch.tensor(min_val))
        self.register_buffer('max_val', torch.tensor(max_val))
        self.slope = slope
        self.eps = eps
        self.bias = bias

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Maps latent (standard normal-like) to bounded physical space."""
        sig = torch.sigmoid((latent + self.bias) * self.slope)
        return self.min_val + (self.max_val - self.min_val) * sig

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Maps from bounded physical space to latent."""
        x = (value - self.min_val) / (self.max_val - self.min_val)
        x = x.clamp(self.eps, 1 - self.eps)  # avoid exploding logits
        return torch.log(x / (1 - x)) / self.slope - self.bias


class LogNormalTransform(nn.Module):
    def __init__(self,
                 mean_log: float = 0.4,
                 std_log: float = 0.35,
                 eps: float = 1e-6
                 ):
        super().__init__()
        self.register_buffer('mean_log', torch.tensor(mean_log))
        self.register_buffer('std_log', torch.tensor(std_log))
        self.eps = eps

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Maps latent to log-normal physical variable"""
        return torch.exp((latent * self.std_log + self.mean_log).clip(min=-5, max=20))

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Maps log-normal physical value to latent"""
        return (torch.log(value.clip(min=self.eps)) - self.mean_log) / self.std_log


class RotationTransform(nn.Module):
    def __init__(self,
                 max_angle: float = 2 * torch.pi,
                 slope: float = 1.0,
                 eps: float = 1e-6
                 ):
        super().__init__()
        self.max_angle = max_angle
        self.eps = eps
        self.register_buffer('normal_mean', torch.tensor(0.0))
        self.register_buffer('normal_std', torch.tensor(1.0))
        self.normal = torch.distributions.Normal(self.normal_mean, self.normal_std)
        self.polar_dist = torch.distributions.HalfNormal(scale=2.0)

    def azimuth_to_std_normal(self, phi):
        # Map from [-π, π] → [0, 1]
        u = (phi + torch.pi) / (2 * torch.pi)
        u = u.clamp(self.eps, 1 - self.eps)
        return self.normal.icdf(u)

    def std_normal_to_azimuth(self, z):
        u = self.normal.cdf(z)
        return u * (2 * torch.pi) - torch.pi  # maps back to [-π, π]

    def polar_to_std_normal(self, theta):
        u = self.polar_dist.cdf(theta.clamp(0, torch.pi / 2))
        u = u.clamp(self.eps, 1 - self.eps)
        return self.normal.icdf(u)

    def std_normal_to_polar(self, z):
        u = torch.distributions.Normal(0, 1).cdf(z)
        return self.polar_dist.icdf(u)

    def rotation_to_std_normal(self, r):
        return r - torch.pi  # assuming std=1

    def std_normal_to_rotation(self, z):
        return z + torch.pi

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Maps latent vector (ℝ³) to rotation vector (SO(3))
        using canonical z > 0 encoding and norm bounding.
        """
        std_theta, std_phi, std_r = latent.split(1, dim=1)

        theta = self.std_normal_to_polar(std_theta)
        phi = self.std_normal_to_azimuth(std_phi)
        r = self.std_normal_to_rotation(std_r)

        rotvec = sph2rotvec(torch.cat([theta, phi, r], dim=1))

        return rotvec

    def inverse(self, rotvec: torch.Tensor) -> torch.Tensor:
        """
        Maps rotation vector back to latent space using inverse angle norm and canonical z decoding.
        """
        # convert to spherical coordinates
        # polar, azimuthal, turn
        theta, phi, r = rotvec2sph(rotvec).split(1, dim=1)

        std_theta = self.polar_to_std_normal(theta)
        std_phi = self.azimuth_to_std_normal(phi)
        std_r = self.rotation_to_std_normal(r)

        return torch.cat([std_theta, std_phi, std_r], dim=1)


class StdNormalTransform(nn.Module):
    def __init__(self,
                 length_slope: float = 1.0,
                 angle_slope: float = 1.0,
                 centroid_slope: float = 1.0,
                 orientation_slope: float = 1.0,
                 c_log_mean: float = 0.24,
                 c_log_std: float = 0.3618,
                 ):
        super().__init__()
        self.eps = 1e-6

        self.transforms = nn.ModuleDict({
            'A': BoundedTransform(0.0, 1.0, slope=length_slope, bias=1.15),
            'B': BoundedTransform(0.0, 1.0, slope=length_slope, bias=1.15),
            'C': LogNormalTransform(c_log_mean, c_log_std),

            'cos_alpha': BoundedTransform(0.0, 1.0, slope=angle_slope),
            'cos_beta': BoundedTransform(0.0, 1.0, slope=angle_slope),
            'cos_gamma': BoundedTransform(0.0, 1.0, slope=angle_slope),

            'centroid_u': BoundedTransform(0.0, 1.0, slope=centroid_slope),
            'centroid_v': BoundedTransform(0.0, 1.0, slope=centroid_slope),
            'centroid_w': BoundedTransform(0.0, 1.0, slope=centroid_slope),
        })
        self.rotation_transform = RotationTransform(
            max_angle=2 * torch.pi,
            slope=orientation_slope
        )

    def forward(self, niggli_params):
        """
        Convert niggli parameters to standard normal basisq
        """
        params = torch.stack(
            [self.transforms[key].inverse(niggli_params[:, ind])
             for ind, key in enumerate(self.transforms.keys())]).T
        return torch.cat(
            [params, self.rotation_transform.inverse(niggli_params[:, 9:])],
            dim=1
        )

    def inverse(self, std_params):
        params = torch.stack(
            [self.transforms[key](std_params[:, ind])
             for ind, key in enumerate(self.transforms.keys())]).T
        return torch.cat(
            [params, self.rotation_transform(std_params[:, 9:])],
            dim=1
        )


class CompositeTransform(nn.Module):
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, sg_inds, mol_radii):
        for transform in self.transforms:
            if 'aunit' in str(transform).lower():
                x = transform.forward(x, sg_inds)
            elif 'niggli' in str(transform).lower():
                x = transform.forward(x, mol_radii)
            else:
                x = transform.forward(x)
        return x

    def inverse(self, x, sg_inds, mol_radii):
        for transform in reversed(self.transforms):
            if 'aunit' in str(transform).lower():
                x = transform.inverse(x, sg_inds)
            elif 'niggli' in str(transform).lower():
                x = transform.inverse(x, mol_radii)
            else:
                x = transform.inverse(x)
        return x

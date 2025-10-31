import math

import numpy as np
import torch
from torch import nn as nn

from mxtaltools.common.geometry_utils import sph2cart_rotvec, cart2sph_rotvec


class AunitTransform(nn.Module):
    def __init__(self, asym_unit_dict, max_z_prime: int = 1):
        super().__init__()
        self.asym_unit_dict = asym_unit_dict
        self.max_z_prime = max_z_prime

    def forward(self, cell_parameters, sg_inds):
        box_params, aunit_params = torch.tensor_split(cell_parameters, [6], dim=1)
        aunit_centroids, cell_orientations = aunit_params.split(3 * self.max_z_prime, dim=1)

        auvs = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in sg_inds]).to(aunit_centroids.device)
        # this is the parallel way to do it over multiple Z'
        aunit_centroids = (
                aunit_centroids
                .reshape(-1, self.max_z_prime, 3)  # [n, max_z', 3]
                / auvs.unsqueeze(1)  # [n, 1, 3] → broadcast over Z′
        ).reshape(-1, 3 * self.max_z_prime)  # back to [n, 3 * max_z']

        return torch.cat([
            box_params,
            aunit_centroids,
            cell_orientations
        ], dim=1)

    def inverse(self, cell_parameters, sg_inds):
        box_params, aunit_params = torch.tensor_split(cell_parameters, [6], dim=1)
        aunit_centroids, cell_orientations = aunit_params.split(3 * self.max_z_prime, dim=1)
        auvs = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in sg_inds]).to(aunit_centroids.device)
        # cell_lengths = aunit_lengths * auvs
        aunit_centroids = (
                aunit_centroids
                .reshape(-1, self.max_z_prime, 3)  # [n, max_z', 3]
                * auvs.unsqueeze(1)  # [n, 1, 3] → broadcast over Z′
        ).reshape(-1, 3 * self.max_z_prime)  # back to [n, 3 * max_z']

        return torch.cat([
            box_params,
            aunit_centroids,
            cell_orientations
        ], dim=1)


def get_max_cos(a, b, c, eps=1e-6):
    al_cos_max = b / 2 / c.clamp(min=eps)
    be_cos_max = a / 2 / c.clamp(min=eps)
    ga_cos_max = a / 2 / b.clamp(min=eps)
    return al_cos_max, be_cos_max, ga_cos_max


class NiggliTransform(nn.Module):  # todo small roundtrip disagreement in here
    def __init__(self,
                 ):
        super().__init__()
        self.eps = 1e-6

    def forward(self, params, mol_radii):
        """
        Reduce physical aunit parameters to a niggli-normed basis
        """
        box_params = params[:, :6]
        aunit_params = params[:, 6:]
        a, b, c, al, be, ga = box_params.split(1, dim=1)

        c_normed = c / 2 / mol_radii.clip(min=1.0)[:, None]

        # rescale a and b
        a_scale = a / b
        b_scale = b / c
        # get maximum cosine values
        al_cos_max, be_cos_max, ga_cos_max = get_max_cos(a, b, c)

        # for general cells, angle between -pi/2 and pi/2
        al_cos = torch.cos(al).clip(min=-1, max=1)
        be_cos = torch.cos(be).clip(min=-1, max=1)
        ga_cos = torch.cos(ga).clip(min=-1, max=1)

        # scale actual cosines against their maxima (positive or negative)
        al_scaled = al_cos / al_cos_max
        be_scaled = be_cos / be_cos_max
        ga_scaled = ga_cos / ga_cos_max

        return torch.cat([
            a_scale, b_scale, c_normed,
            al_scaled, be_scaled, ga_scaled,
            aunit_params],
            dim=1)

    """
    samples = torch.cat([
            a_scale, b_scale, c_normed,
            al_scaled, be_scaled, ga_scaled,
            u, v, w, x, y, z
        ], dim=1)
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=4, cols=3)
    for ind in range(12):
        ss = samples[:, ind]
        if ind in [0, 1, 3, 4, 5]:
            ss=ss.clip(min=-1, max=1)
        fig.add_histogram(x=ss, nbinsx=50, row=ind // 3 + 1, col = ind % 3 + 1)
    fig.show()
    
    """

    def inverse(self, niggli_params, mol_radii):
        """
        Niggli params to cell params
        :param niggli_params:
        :param mol_radii:
        :return:
        """
        box_params = niggli_params[:, :6]
        aunit_params = niggli_params[:, 6:]
        (a_scale, b_scale, c_normed,
         al_scaled, be_scaled, ga_scaled) = box_params.split(1, dim=1)
        # denormalize c
        c = c_normed * 2 * mol_radii[:, None]
        # descale a and b
        b = b_scale * c
        a = a_scale * b

        # descale the cosines
        al_cos_max, be_cos_max, ga_cos_max = get_max_cos(a, b, c)

        al_cos = al_scaled * al_cos_max
        be_cos = be_scaled * be_cos_max
        ga_cos = ga_scaled * ga_cos_max

        # retrieve the angles
        # for acute niggli cells the minimum cos value is 0
        al = torch.arccos(al_cos.clip(-1 + self.eps, 1 - self.eps))
        be = torch.arccos(be_cos.clip(-1 + self.eps, 1 - self.eps))
        ga = torch.arccos(ga_cos.clip(-1 + self.eps, 1 - self.eps))

        return torch.cat(
            [a, b, c,
             al, be, ga,
             aunit_params],
            dim=1)


class BoundedTransform(nn.Module):
    def __init__(self, min_val, max_val, slope=1.0, eps=1e-6, bias: float = 0.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
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


class SquashingTransform(nn.Module):
    def __init__(self,
                 min_val,
                 max_val,
                 eps=1e-2,
                 threshold: float = 5.0,
                 softness: float = 5.0,
                 ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps
        self.threshold = threshold
        self.softness = softness
        self.sat_level = max_val

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Ultra-smooth soft clipper with gradual derivative transition using PyTorch

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
        threshold : float
            Point where transition becomes noticeable (not abrupt)
        sat_level : float
            Maximum output level (saturation point)
        softness : float
            Controls how gradual the transition is (higher = more gradual)

        Returns:
        --------
        torch.Tensor
            Soft-clipped output with very smooth derivative
        """
        # Use a smooth tanh-like function but with better center linearity
        # Scale input to control where transition begins
        x_scaled = latent / self.threshold

        # Use a modified rational sigmoid that:
        # 1. Has slope ≈ 1 at center
        # 2. Saturates at ±sat_level
        # 3. Has smooth derivative everywhere

        # Higher-order sigmoid: x / (1 + |x|^softness)^(1/softness)
        abs_x_scaled = torch.abs(x_scaled)
        sign_x = torch.sign(x_scaled)

        # Smooth transition factor
        denom = torch.pow(1 + torch.pow(abs_x_scaled, self.softness), 1.0 / self.softness)

        # Apply saturation scaling
        squashed = self.sat_level * sign_x * abs_x_scaled / denom

        return squashed.clip(min=self.min_val,
                             max=self.max_val)  # clipping shouldn't be necessary but one can't be too careful

    def inverse(self, squashed: torch.Tensor) -> torch.Tensor:
        """
        Analytical inverse of the smooth soft clip function

        Parameters:
        -----------
        y : torch.Tensor
            Output from the soft clip function (input to inverse)
        threshold : float
            Same threshold parameter used in forward function
        sat_level : float
            Same saturation level used in forward function
        softness : float
            Same softness parameter used in forward function

        Returns:
        --------
        torch.Tensor
            Original input x that would produce output y
        """
        # Clamp y to valid range to avoid numerical issues
        y_clamped = torch.clamp(squashed, self.min_val + self.eps, self.max_val - self.eps)

        # Normalize y by saturation level
        y_norm = y_clamped / self.sat_level
        abs_y_norm = torch.abs(y_norm)
        sign_y = torch.sign(y_norm)

        y_power = torch.pow(abs_y_norm, self.softness)
        u = y_power / (1 - y_power + 1e-8)  # Small epsilon for numerical stability

        # Convert back: t = u^(1/softness)
        t = torch.pow(u, 1.0 / self.softness)

        # Scale back to original domain
        stretched = sign_y * t * self.threshold

        return stretched


class LogNormalTransform(nn.Module):
    def __init__(self,
                 mean_log: float = 0.5,
                 std_log: float = 0.35,
                 eps: float = 1e-6,
                 exp_min: float = None,
                 exp_max: float = None,
                 ):
        super().__init__()
        self.mean_log = mean_log
        self.std_log = std_log
        self.eps = eps
        self.exp_min = exp_min
        self.exp_max = exp_max

    def inverse(self, latent: torch.Tensor) -> torch.Tensor:
        """Maps latent to log-normal physical variable"""
        return torch.exp(
            (latent * self.std_log + self.mean_log).clip(min=np.log(self.exp_min), max=np.log(self.exp_max)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Maps log-normal physical value to latent"""
        return (torch.log(value.clip(min=self.eps)) - self.mean_log) / self.std_log


class RotationTransform(nn.Module):
    def __init__(self,
                 eps: float = 1e-6,
                 mode: str = 'linear',  # linear or wrapped
                 ):
        super().__init__()
        self.eps = eps
        self.register_buffer('normal_mean', torch.tensor(0.0))
        self.register_buffer('normal_std', torch.tensor(1.0))
        self.normal = torch.distributions.Normal(self.normal_mean, self.normal_std)
        self.polar_dist = torch.distributions.HalfNormal(scale=2.0)
        self.mode = mode

    def wrap_to_pi(self, x):
        # (-pi, pi]
        return (x + math.pi) % (2 * math.pi) - math.pi

    def azimuth_to_std_normal(self, phi):
        # Map from [-π, π] → [0, 1]
        u = (phi + torch.pi) / (2 * torch.pi)
        u = u.clamp(self.eps, 1 - self.eps)
        return self.normal.icdf(u)

    def std_normal_to_azimuth(self, z):
        u = self.normal.cdf(z)
        return u * (2 * torch.pi) - torch.pi  # maps back to [-π, π]

    def azimuth_to_uniform(self, phi):
        # latent space and azimuthal angle are identical
        # uniform distributions on [-pi, pi]
        return self.wrap_to_pi(phi)

    def uniform_to_azimuth(self, z):
        # latent space should be uniform on [-pi, pi]
        # wrap for safety
        return self.wrap_to_pi(z)

    def polar_to_std_normal(self, theta):
        u = theta / (torch.pi / 2)
        u = u.clip(self.eps, 1 - self.eps)
        u = torch.log(u) - torch.log(1 - u) - torch.pi / 4  # inverse sigmoid
        return u

    def std_normal_to_polar(self, z):
        return (torch.pi / 2) * torch.sigmoid(z + torch.pi / 4)

    def std_normal_to_rotation(self, z):
        # z ∈ (−∞, ∞) → r ∈ [0, 2π]
        if self.mode == 'linear':
            return 2 * torch.pi * torch.sigmoid(z)
        elif self.mode == 'wrapped':
            # the rotation is naturally std normal with mean pi
            return self.wrap_to_pi(z) + torch.pi

    def rotation_to_std_normal(self, r):
        # r ∈ [0, 2π] → z ∈ (−∞, ∞)
        if self.mode == 'linear':
            u = r / (2 * torch.pi)
            u = u.clamp(self.eps, 1 - self.eps)  # numerical stability
            return torch.log(u) - torch.log(1 - u)
        elif self.mode == 'wrapped':
            # since r is naturally bounded on [0, 2pi], we shouldn't have to manually wrap here
            return r - torch.pi

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Maps latent vector (ℝ³) to rotation vector (SO(3))
        using canonical z > 0 encoding and norm bounding.
        """
        std_theta, std_phi, std_r = latent.split(1, dim=1)

        theta = self.std_normal_to_polar(std_theta)  # .clip(min=0, max=torch.pi / 2)
        if self.mode == 'wrapped':
            phi = self.uniform_to_azimuth(std_phi)  # .clip(min=-torch.pi, max=torch.pi)
            r = self.std_normal_to_rotation(std_r).clip(min=0.01,
                                                        max=torch.pi * 2 - 0.01)  # cannot be allowed to touch extrema exactly
        elif self.mode == 'linear':
            phi = self.std_normal_to_azimuth(std_phi)  # .clip(min=-torch.pi, max=torch.pi)
            r = self.std_normal_to_rotation(std_r).clip(min=0.01,
                                                        max=torch.pi * 2 - 0.01)  # cannot be allowed to touch extrema exactly

        rotvec = sph2cart_rotvec(torch.cat([theta, phi, r], dim=1))

        return rotvec

    def inverse(self, rotvec: torch.Tensor) -> torch.Tensor:
        """
        Maps rotation vector back to latent space using inverse angle norm and canonical z decoding.
        """
        # convert to spherical coordinates
        # polar, azimuthal, turn
        theta, phi, r = cart2sph_rotvec(rotvec).split(1, dim=1)

        std_theta = self.polar_to_std_normal(theta)
        if self.mode == 'linear':
            std_phi = self.azimuth_to_std_normal(phi)
            std_r = self.rotation_to_std_normal(r)
        elif self.mode == 'wrapped':
            std_phi = self.azimuth_to_uniform(phi)
            std_r = self.rotation_to_std_normal(r)

        return torch.cat([std_theta, std_phi, std_r], dim=1)


class ProbitTransform(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer('normal_mean', torch.tensor(0.0))
        self.register_buffer('normal_std', torch.tensor(1.0))
        self.normal = torch.distributions.Normal(self.normal_mean, self.normal_std)

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid extreme values
        u = u.clamp(self.eps, 1 - self.eps)
        return self.normal.icdf(u)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.normal.cdf(z)


#
# class StdNormalTransform(nn.Module):
#     def __init__(self,
#                  length_slope: float = 1.0,
#                  angle_slope: float = 1.0,
#                  c_log_mean: float = 0.24,
#                  c_log_std: float = 0.24,  # 0.3618,
#                  rot_mode: str = 'linear',
#                  ):
#         super().__init__()
#         self.eps = 1e-6
#
#         self.transforms = nn.ModuleDict({  # todo parallelize/accelerate these
#             'A': BoundedTransform(0.0, 1.0, slope=length_slope, bias=1.15),
#             'B': BoundedTransform(0.0, 1.0, slope=length_slope, bias=1.15),
#             'C': LogNormalTransform(c_log_mean, c_log_std, exp_min=0.01, exp_max=8),
#
#             'cos_alpha': BoundedTransform(-1.0, 1.0, slope=angle_slope),
#             'cos_beta': BoundedTransform(-1.0, 1.0, slope=angle_slope),
#             'cos_gamma': BoundedTransform(-1.0, 1.0, slope=angle_slope),
#
#             'centroid_u': ProbitTransform(),
#             'centroid_v': ProbitTransform(),
#             'centroid_w': ProbitTransform(),
#         })
#         self.rotation_transform = RotationTransform(
#             mode=rot_mode,
#         )
#
#     def forward(self, niggli_params):
#         """
#         Convert niggli parameters to standard normal basisq
#         """
#         params = torch.stack(
#             [self.transforms[key].inverse(niggli_params[:, ind])
#              for ind, key in enumerate(self.transforms.keys())]).T
#         return torch.cat(
#             [params, self.rotation_transform.inverse(niggli_params[:, 9:])],
#             dim=1
#         )
#
#     def inverse(self, std_params):
#         params = torch.stack(
#             [self.transforms[key](std_params[:, ind])
#              for ind, key in enumerate(self.transforms.keys())]).T
#         return torch.cat(
#             [params, self.rotation_transform(std_params[:, 9:])],
#             dim=1
#         )


class UnitToPMUnit(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return u * 0.5 + 0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z - 0.5) * 2


class ThetaTransform(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return (1 / 4) * torch.pi * u + torch.pi / 4

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z - torch.pi / 4) / torch.pi * 4


class PhiTransform(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return u * torch.pi

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z / torch.pi


class RTransform(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return u * torch.pi + torch.pi

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z - torch.pi) / torch.pi


class UnitTransform(nn.Module):
    def __init__(self, max_z_prime: int = 1):
        super().__init__()
        self.max_z_prime = max_z_prime
        self.transforms = [
            UnitToPMUnit(),  # a/b
            UnitToPMUnit(),  # b/c
            LogNormalTransform(std_log=2,
                               mean_log=0.25,
                               exp_min=0.01, exp_max=4),  # c (normed)
            IdentityTransform(),  # alpha
            IdentityTransform(),  # beta
            IdentityTransform(),  # gamma
        ]
        for zp in range(max_z_prime):
            self.transforms.extend([
                UnitToPMUnit(),  # u
                UnitToPMUnit(),  # w
                UnitToPMUnit(),  # v
            ])
        for zp in range(max_z_prime):
            self.transforms.extend([
                ThetaTransform(),  # theta
                PhiTransform(),  # phi
                RTransform(),  # r
            ])

    def forward(self, niggli_params):
        """
        Convert niggli parameters to uniform latent basis
        """
        box_params, aunit_params = torch.tensor_split(niggli_params, [6], dim=1)
        aunit_centroids, cell_orientations = aunit_params.split(3 * self.max_z_prime, dim=1)
        sph_rotvec = cart2sph_rotvec(cell_orientations.reshape(len(cell_orientations) * self.max_z_prime, 3)).reshape(
            len(cell_orientations), self.max_z_prime * 3)
        rot_params = torch.cat([box_params, aunit_centroids, sph_rotvec], dim=1)

        params = torch.stack(
            [transform.forward(rot_params[:, ind])
             for ind, transform in enumerate(self.transforms)]).T

        return params

    def inverse(self, std_params):
        """
        Convert uniform latent basis to niggli space
        :param std_params:
        :return:
        """
        params = torch.stack(
            [transform.inverse(std_params[:, ind])
             for ind, transform in enumerate(self.transforms)]).T

        box_params, aunit_params = torch.tensor_split(params, [6], dim=1)
        aunit_centroids, cell_orientations = aunit_params.split(3 * self.max_z_prime, dim=1)
        sph_rotvec = sph2cart_rotvec(cell_orientations.reshape(len(cell_orientations) * self.max_z_prime, 3)).reshape(
            len(cell_orientations), self.max_z_prime * 3)
        rot_params = torch.cat([box_params, aunit_centroids, sph_rotvec], dim=1)

        return rot_params


class IdentityTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return z

    def inverse(self, z):
        return z


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


def enforce_niggli_plane(cell_lengths, cell_angles, mode, eps=1e-6):
    """
    enforces the condition
    ab * ga_cos + ac * be_cos + bc * al_cos >= 0
    'mirror' mode is for random samples, and symmetrically puts them on the correct side of the plane
    'shift' mode shifts offending samples to the nearest boundary of the good plane

    """
    a, b, c = cell_lengths.split(1, dim=1)
    al, be, ga = cell_angles.split(1, dim=1)

    ab, ac, al_cos, bc, be_cos, ga_cos, overlap = compute_niggli_overlap(a, al, b, be, c, ga)
    if torch.any(overlap < 0):
        if mode == 'mirror':
            # Symmetric reflection for bad samples
            bad_inds = torch.argwhere(overlap.flatten() < 0).flatten()
            al[bad_inds] = torch.pi - al[bad_inds]
            be[bad_inds] = torch.pi - be[bad_inds]
            ga[bad_inds] = torch.pi - ga[bad_inds]

        elif mode == 'shift':
            # Project offending points to the nearest point on the zero overlap plane
            """
            for ab*cos(gamma) + ac*cos(beta) + bc*cos(alpha)=Ax+By+Cz=overlap as the equation of the plane
            project from arbitrary point r=(xyz) to the zero-plane
            The plane normal is N=(ABC). The overlap is N * r.
            The corrective vector is v=-(N * r)/(N*N)*N
            
            Issue with this approach is due to roundtrip cos/arccos float problems, it doesn't perfectly work often in one shot.
            So we iterate and add a small positive factor.
            
            **should** be idempotent, or pretty close
            """
            boost = 1e-3
            for _ in range(20):
                r = torch.cat([ga_cos, be_cos, al_cos], dim=1)  # (n, 3)
                N = torch.cat([ab, ac, bc], dim=1)  # (n, 3)

                # Compute scalar projection
                dot = (N * r).sum(dim=1, keepdim=True)
                norm2 = (N ** 2).sum(dim=1, keepdim=True)

                # Correction vector
                shift = -(dot / (norm2 + eps)) * N + boost * N / N.norm(dim=1, keepdim=True)

                # Only apply when overlap < 0
                mask = (overlap < 0).float()
                fixed_r = r + mask * shift

                # Convert back to angles
                ga, be, al = torch.arccos(fixed_r.clip(-1 + eps, 1 - eps)).split(1, dim=1)
                ga_cos, be_cos, al_cos = ga.cos(), be.cos(), al.cos()

                """"""
                overlap = ab * ga.cos() + ac * be.cos() + bc * al.cos()
                if torch.all(overlap >= 0):
                    break

        else:
            raise ValueError(f"Unknown mode '{mode}': use 'mirror' or 'shift'")

    ab, ac, al_cos, bc, be_cos, ga_cos, overlap = compute_niggli_overlap(a, al, b, be, c, ga)
    # assert torch.all(overlap >= 0), "Niggli plane enforcement failed!!"
    if torch.any(overlap < 0):
        print(f"Niggli enforcement failed with overlap of {overlap.amin():3g}")

    return torch.cat([al, be, ga], dim=1)


def compute_niggli_overlap(a, al, b, be, c, ga):
    ab = a * b
    ac = a * c
    bc = b * c
    al_cos = torch.cos(al)
    be_cos = torch.cos(be)
    ga_cos = torch.cos(ga)
    overlap = ab * ga_cos + ac * be_cos + bc * al_cos
    return ab, ac, al_cos, bc, be_cos, ga_cos, overlap

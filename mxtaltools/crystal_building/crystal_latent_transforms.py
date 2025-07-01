import numpy as np
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
        aunit_centroids = cell_centroids / auvs.to(cell_centroids.device)

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
                 mean_log: float = 0.4,
                 std_log: float = 0.35,
                 eps: float = 1e-6,
                 exp_min: float = None,
                 exp_max: float = None,
                 ):
        super().__init__()
        self.register_buffer('mean_log', torch.tensor(mean_log))
        self.register_buffer('std_log', torch.tensor(std_log))
        self.eps = eps
        self.exp_min = exp_min
        self.exp_max = exp_max

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Maps latent to log-normal physical variable"""
        return torch.exp((latent * self.std_log + self.mean_log).clip(min=np.log(self.exp_min), max=np.log(self.exp_max)))

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Maps log-normal physical value to latent"""
        return (torch.log(value.clip(min=self.eps)) - self.mean_log) / self.std_log


class RotationTransform(nn.Module):
    def __init__(self,
                 max_angle: float = 2 * torch.pi,
                 eps: float = 1e-6
                 ):
        super().__init__()
        self.max_angle = max_angle
        self.eps = eps
        self.register_buffer('normal_mean', torch.tensor(0.0))
        self.register_buffer('normal_std', torch.tensor(1.0))
        self.normal = torch.distributions.Normal(self.normal_mean, self.normal_std)
        self.polar_dist = torch.distributions.HalfNormal(scale=2.0)
        self.bound = SquashingTransform(min_val=-6, max_val=6)

    def azimuth_to_std_normal(self, phi):
        # Map from [-π, π] → [0, 1]
        u = (phi + torch.pi) / (2 * torch.pi)
        u = u.clamp(self.eps, 1 - self.eps)
        return self.normal.icdf(u)

    def std_normal_to_azimuth(self, z):
        u = self.normal.cdf(z)
        return u * (2 * torch.pi) - torch.pi  # maps back to [-π, π]

    def polar_to_std_normal(self, theta):
        u = theta / (torch.pi / 2)
        u = u.clip(self.eps, 1 - self.eps)
        u = torch.log(u) - torch.log(1 - u) - torch.pi / 4  # inverse sigmoid
        return u

    def std_normal_to_polar(self, z):
        return (torch.pi / 2) * torch.sigmoid(z + torch.pi / 4)

    def rotation_to_std_normal(self, r):
        return self.bound(r - torch.pi)  # assuming std=1 and enforcing the bound

    def std_normal_to_rotation(self, z):
        return self.bound.inverse(z) + torch.pi

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Maps latent vector (ℝ³) to rotation vector (SO(3))
        using canonical z > 0 encoding and norm bounding.
        """
        std_theta, std_phi, std_r = latent.split(1, dim=1)

        theta = self.std_normal_to_polar(std_theta)#.clip(min=0, max=torch.pi / 2)
        phi = self.std_normal_to_azimuth(std_phi)#.clip(min=-torch.pi, max=torch.pi)
        r = self.std_normal_to_rotation(std_r).clip(min=0.01, max=torch.pi * 2 - 0.01)  # cannot be allowed to touch extrema exactly

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
                 c_log_mean: float = 0.24,
                 c_log_std: float = 0.25,  #0.3618,
                 ):
        super().__init__()
        self.eps = 1e-6

        self.transforms = nn.ModuleDict({
            'A': BoundedTransform(0.0, 1.0, slope=length_slope, bias=1.15),
            'B': BoundedTransform(0.0, 1.0, slope=length_slope, bias=1.15),
            'C': LogNormalTransform(c_log_mean, c_log_std, exp_min=0.01, exp_max=8),

            'cos_alpha': BoundedTransform(0.0, 1.0, slope=angle_slope),
            'cos_beta': BoundedTransform(0.0, 1.0, slope=angle_slope),
            'cos_gamma': BoundedTransform(0.0, 1.0, slope=angle_slope),

            'centroid_u': BoundedTransform(0.0, 1.0, slope=centroid_slope),
            'centroid_v': BoundedTransform(0.0, 1.0, slope=centroid_slope),
            'centroid_w': BoundedTransform(0.0, 1.0, slope=centroid_slope),
        })
        self.rotation_transform = RotationTransform(
            max_angle=2 * torch.pi,
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

#
# import torch
# import torch.nn as nn
# from typing import Callable, Optional
#
# def compute_log_det_jacobian(
#     transform_fn: Callable[[torch.Tensor], torch.Tensor],
#     x: torch.Tensor,
#     create_graph: bool = False
# ) -> torch.Tensor:
#     """
#     Compute log determinant of Jacobian for arbitrary transform function.
#
#     Args:
#         transform_fn: Function that takes [batch, dim] and returns [batch, dim]
#         x: Input tensor of shape [batch, dim]
#         create_graph: Whether to create computational graph for higher-order derivatives
#
#     Returns:
#         log_det: Log determinant for each batch element, shape [batch]
#     """
#     batch_size, dim = x.shape
#     x = x.requires_grad_(True)
#     y = transform_fn(x)
#
#     log_dets = []
#
#     for batch_idx in range(batch_size):
#         jacobian_row = []
#
#         for output_dim in range(dim):
#             # Create one-hot gradient for this output dimension
#             grad_outputs = torch.zeros_like(y)
#             grad_outputs[batch_idx, output_dim] = 1.0
#
#             # Compute gradients
#             grads = torch.autograd.grad(
#                 outputs=y,
#                 inputs=x,
#                 grad_outputs=grad_outputs,
#                 retain_graph=True,
#                 create_graph=create_graph,
#                 only_inputs=True
#             )[0]
#
#             # Extract row of Jacobian for this batch element
#             jacobian_row.append(grads[batch_idx])
#
#         # Stack to form Jacobian matrix [dim, dim]
#         jacobian = torch.stack(jacobian_row, dim=0)
#
#         # Compute log determinant
#         log_det = torch.logdet(jacobian)
#         log_dets.append(log_det)
#
#     return torch.stack(log_dets)
#
# def compute_log_det_jacobian_diagonal(
#     transform_fn: Callable[[torch.Tensor], torch.Tensor],
#     x: torch.Tensor,
#     create_graph: bool = False
# ) -> torch.Tensor:
#     """
#     Fast diagonal approximation of log determinant of Jacobian.
#     Assumes off-diagonal elements are negligible.
#
#     Args:
#         transform_fn: Function that takes [batch, dim] and returns [batch, dim]
#         x: Input tensor of shape [batch, dim]
#         create_graph: Whether to create computational graph
#
#     Returns:
#         log_det: Approximate log determinant for each batch element, shape [batch]
#     """
#     batch_size, dim = x.shape
#     x = x.requires_grad_(True)
#     y = transform_fn(x)
#
#     log_det = torch.zeros(batch_size, device=x.device)
#
#     for i in range(dim):
#         # Compute diagonal element of Jacobian
#         grad_outputs = torch.zeros_like(y)
#         grad_outputs[:, i] = 1.0
#
#         grads = torch.autograd.grad(
#             outputs=y,
#             inputs=x,
#             grad_outputs=grad_outputs,
#             retain_graph=True,
#             create_graph=create_graph,
#             only_inputs=True
#         )[0]
#
#         # Add log of diagonal element
#         diagonal_element = grads[:, i]
#         log_det += torch.log(torch.abs(diagonal_element) + 1e-8)
#
#     return log_det
#
# class JacobianWrapper(nn.Module):
#     """
#     Wrapper that adds Jacobian computation to any transform.
#     """
#
#     def __init__(self, transform: nn.Module):
#         super().__init__()
#         self.transform = transform
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.transform(x)
#
#     def inverse(self, y: torch.Tensor) -> torch.Tensor:
#         if hasattr(self.transform, 'inverse'):
#             return self.transform.inverse(y)
#         else:
#             raise NotImplementedError("Transform does not have inverse method")
#
#     def log_det_jacobian_forward(self, x: torch.Tensor, diagonal_approx: bool = False) -> torch.Tensor:
#         """Compute log det of forward Jacobian."""
#         if diagonal_approx:
#             return compute_log_det_jacobian_diagonal(self.transform, x)
#         else:
#             return compute_log_det_jacobian(self.transform, x)
#
#     def log_det_jacobian_inverse(self, y: torch.Tensor, diagonal_approx: bool = False) -> torch.Tensor:
#         """Compute log det of inverse Jacobian."""
#         if not hasattr(self.transform, 'inverse'):
#             raise NotImplementedError("Transform does not have inverse method")
#
#         if diagonal_approx:
#             return compute_log_det_jacobian_diagonal(self.transform.inverse, y)
#         else:
#             return compute_log_det_jacobian(self.transform.inverse, y)
#
# # Simple functional interface
# def add_jacobian_computation(transform_fn: Callable[[torch.Tensor], torch.Tensor]):
#     """
#     Decorator to add Jacobian computation methods to a function.
#
#     Usage:
#         @add_jacobian_computation
#         def my_transform(x):
#             return torch.tanh(x * 2)
#
#         x = torch.randn(4, 3)
#         y = my_transform(x)
#         log_det = my_transform.log_det_jacobian(x)
#     """
#     def log_det_jacobian(x: torch.Tensor, diagonal_approx: bool = False) -> torch.Tensor:
#         if diagonal_approx:
#             return compute_log_det_jacobian_diagonal(transform_fn, x)
#         else:
#             return compute_log_det_jacobian(transform_fn, x)
#
#     # Attach method to function
#     transform_fn.log_det_jacobian = log_det_jacobian
#     return transform_fn
#
# # Example usage and testing
# if __name__ == "__main__":
#     # Example 1: Simple function
#     @add_jacobian_computation
#     def simple_transform(x):
#         return torch.tanh(x * 2) + x**2
#
#     # Example 2: Your complex transforms
#     class SquashingTransform(nn.Module):
#         def __init__(self, min_val, max_val, threshold=5.0, softness=5.0):
#             super().__init__()
#             self.min_val = min_val
#             self.max_val = max_val
#             self.threshold = threshold
#             self.softness = softness
#             self.sat_level = max_val
#
#         def forward(self, latent):
#             x_scaled = latent / self.threshold
#             abs_x_scaled = torch.abs(x_scaled)
#             sign_x = torch.sign(x_scaled)
#             denom = torch.pow(1 + torch.pow(abs_x_scaled, self.softness), 1.0 / self.softness)
#             squashed = self.sat_level * sign_x * abs_x_scaled / denom
#             return squashed.clip(min=self.min_val, max=self.max_val)
#
#     # Test data
#     x = torch.randn(4, 3)
#
#     print("Testing simple function:")
#     y = simple_transform(x)
#     log_det_exact = simple_transform.log_det_jacobian(x, diagonal_approx=False)
#     log_det_diag = simple_transform.log_det_jacobian(x, diagonal_approx=True)
#     print(f"Output shape: {y.shape}")
#     print(f"Log det (exact): {log_det_exact}")
#     print(f"Log det (diagonal): {log_det_diag}")
#     print(f"Difference: {(log_det_exact - log_det_diag).abs().max():.6f}")
#
#     print("\nTesting complex transform with wrapper:")
#     squash = SquashingTransform(min_val=-5, max_val=5)
#     wrapped_squash = JacobianWrapper(squash)
#
#     y2 = wrapped_squash(x)
#     log_det_exact2 = wrapped_squash.log_det_jacobian_forward(x, diagonal_approx=False)
#     log_det_diag2 = wrapped_squash.log_det_jacobian_forward(x, diagonal_approx=True)
#     print(f"Output shape: {y2.shape}")
#     print(f"Log det (exact): {log_det_exact2}")
#     print(f"Log det (diagonal): {log_det_diag2}")
#     print(f"Difference: {(log_det_exact2 - log_det_diag2).abs().max():.6f}")
#
#     # Performance comparison
#     import time
#
#     print("\nPerformance comparison (100 iterations):")
#
#     # Exact method
#     start = time.time()
#     for _ in range(100):
#         _ = compute_log_det_jacobian(squash, x)
#     exact_time = time.time() - start
#
#     # Diagonal approximation
#     start = time.time()
#     for _ in range(100):
#         _ = compute_log_det_jacobian_diagonal(squash, x)
#     diag_time = time.time() - start
#
#     print(f"Exact method: {exact_time:.4f}s")
#     print(f"Diagonal approx: {diag_time:.4f}s")
#     print(f"Speedup: {exact_time/diag_time:.2f}x")
#
#     # Validation against finite differences
#     print("\nValidation against finite differences:")
#     eps = 1e-5
#     batch_size, dim = x.shape
#
#     # Compute finite difference Jacobian for first batch element
#     x_test = x[0:1].clone()
#     y_base = squash(x_test)
#     jacobian_fd = torch.zeros(dim, dim)
#
#     for i in range(dim):
#         x_plus = x_test.clone()
#         x_minus = x_test.clone()
#         x_plus[0, i] += eps
#         x_minus[0, i] -= eps
#
#         y_plus = squash(x_plus)
#         y_minus = squash(x_minus)
#
#         jacobian_fd[:, i] = ((y_plus - y_minus) / (2 * eps)).squeeze()
#
#     log_det_fd = torch.logdet(jacobian_fd)
#     log_det_auto = compute_log_det_jacobian(squash, x_test)[0]
#
#     print(f"Finite difference log det: {log_det_fd:.6f}")
#     print(f"Automatic log det: {log_det_auto:.6f}")
#     print(f"Error: {(log_det_fd - log_det_auto).abs():.8f}")

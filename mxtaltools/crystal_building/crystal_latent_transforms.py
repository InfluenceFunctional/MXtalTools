"""
Invertible transforms between crystal cell parameters and normalized bases.

Each transform is a ``torch.nn.Module`` with a ``forward`` and an ``inverse``
method acting on batched tensors, one sample per row. Full cell parameters use
the column layout ``[a, b, c, alpha, beta, gamma, centroids, orientations]``,
with angles in radians, ``3 * max_z_prime`` centroid columns and
``3 * max_z_prime`` orientation columns (rotation vectors). Which direction is
called ``forward`` differs between classes and is stated on each.

The latent basis of ``MolCrystalData.latent_params`` and
``MolCrystalData.latent_to_cell_params`` is implemented in
``mxtaltools/dataset_utils/data_class_methods/crystal_ops.py`` and does not
use these classes.
"""
import math

import numpy as np
import torch
from torch import nn as nn

from mxtaltools.common.geometry_utils import sph2cart_rotvec, cart2sph_rotvec


class AunitTransform(nn.Module):
    """
    Rescale centroids between unit-cell and asymmetric-unit fractional coordinates.

    Only the centroid columns change; the six box parameters and the
    orientation columns pass through unchanged.

    Parameters
    ----------
    asym_unit_dict : dict
        Maps the space-group number as a string (``str(int(sg_ind))``) to a
        tensor [3]: the fraction of each unit-cell axis spanned by the
        asymmetric unit, as in ``ASYM_UNITS``. Values must be tensors; they
        are combined with ``torch.stack``.
    max_z_prime : int, optional
        Number of asymmetric-unit slots per row. Default 1.
    """
    def __init__(self, asym_unit_dict, max_z_prime: int = 1):
        super().__init__()
        self.asym_unit_dict = asym_unit_dict
        self.max_z_prime = max_z_prime

    def forward(self, cell_parameters, sg_inds):
        """
        Divide each centroid triple by its row's asymmetric-unit extents.

        Parameters
        ----------
        cell_parameters : torch.Tensor
            [n, 6 + 6 * max_z_prime]: box parameters, 3 * max_z_prime centroid
            columns ((x, y, z) per slot, unit-cell fractional), then
            3 * max_z_prime orientation columns.
        sg_inds : iterable
            [n] space-group numbers, looked up in ``asym_unit_dict`` as
            ``str(int(ind))``.

        Returns
        -------
        torch.Tensor
            New [n, 6 + 6 * max_z_prime] tensor with the centroids divided
            elementwise by the extents (asymmetric-unit fractional); other
            columns unchanged. The extents are moved to the input's device.
        """
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
        """
        Multiply each centroid triple by its row's asymmetric-unit extents.

        Inverse of :meth:`forward`; same arguments and layout.

        Parameters
        ----------
        cell_parameters : torch.Tensor
            [n, 6 + 6 * max_z_prime] with asymmetric-unit fractional centroids.
        sg_inds : iterable
            [n] space-group numbers.

        Returns
        -------
        torch.Tensor
            New [n, 6 + 6 * max_z_prime] tensor with unit-cell fractional
            centroids; other columns unchanged.
        """
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
    """
    Return the cosine bounds ``b / (2c)``, ``a / (2c)`` and ``a / (2b)``.

    These are the Niggli-condition bounds on ``|cos(alpha)|``,
    ``|cos(beta)|`` and ``|cos(gamma)|`` for a cell with a <= b <= c.

    Parameters
    ----------
    a, b, c : torch.Tensor
        Cell lengths, broadcast-compatible.
    eps : float, optional
        Lower clamp on the denominators ``c`` and ``b``. Default 1e-6.

    Returns
    -------
    tuple of torch.Tensor
        ``(al_cos_max, be_cos_max, ga_cos_max)``.
    """
    al_cos_max = b / 2 / c.clamp(min=eps)
    be_cos_max = a / 2 / c.clamp(min=eps)
    ga_cos_max = a / 2 / b.clamp(min=eps)
    return al_cos_max, be_cos_max, ga_cos_max


class NiggliTransform(nn.Module):  # todo small roundtrip disagreement in here
    """
    Map box parameters to and from a length-ratio / scaled-cosine basis.

    :meth:`forward` replaces (a, b, c, alpha, beta, gamma) with
    (a / b, b / c, c / (2 * radius), cos(alpha) / (b / 2c),
    cos(beta) / (a / 2c), cos(gamma) / (a / 2b)), the cosine bounds coming
    from :func:`get_max_cos`. Columns after the first six pass through
    unchanged. Angles are in radians. :meth:`inverse` undoes the map.

    Notes
    -----
    :meth:`forward` clamps ``mol_radii`` below at 1.0 before normalizing c;
    :meth:`inverse` uses ``mol_radii`` unclamped. Rows with a radius below
    1.0 therefore come back with a, b and c multiplied by the radius; the
    angles are unaffected.
    """
    def __init__(self,
                 ):
        super().__init__()
        self.eps = 1e-6

    def forward(self, params, mol_radii):
        """
        Map cell parameters to the length-ratio / scaled-cosine basis.

        Parameters
        ----------
        params : torch.Tensor
            [n, 6 + k]: (a, b, c, alpha, beta, gamma) with angles in radians,
            then k columns passed through unchanged.
        mol_radii : torch.Tensor
            [n] molecular radii, in the length unit of a, b, c; clamped below
            at 1.0.

        Returns
        -------
        torch.Tensor
            New [n, 6 + k] tensor: (a / b, b / c, c / (2 * max(mol_radii, 1)),
            cos(alpha) / (b / 2c), cos(beta) / (a / 2c), cos(gamma) / (a / 2b)),
            then the passed-through columns.
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
        Map the length-ratio / scaled-cosine basis back to cell parameters.

        Parameters
        ----------
        niggli_params : torch.Tensor
            [n, 6 + k] in the layout returned by :meth:`forward`.
        mol_radii : torch.Tensor
            [n] molecular radii, used without the 1.0 lower clamp applied in
            :meth:`forward`.

        Returns
        -------
        torch.Tensor
            New [n, 6 + k] tensor (a, b, c, alpha, beta, gamma, ...) with
            c = c_normed * 2 * mol_radii, b = (b / c) * c, a = (a / b) * b,
            and each angle the arccos of its scaled cosine times its bound
            from :func:`get_max_cos`, the cosine clamped to
            [-1 + eps, 1 - eps] (eps = 1e-6). Angles in radians; the other
            columns unchanged.
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
    """
    Scaled logistic map between the real line and (min_val, max_val).

    :meth:`forward` maps latent to bounded values; :meth:`inverse` maps
    bounded values to latent.

    Parameters
    ----------
    min_val, max_val : float
        Interval bounds.
    slope : float, optional
        Multiplier inside the sigmoid, applied after the bias. Default 1.0.
    eps : float, optional
        Clamp margin on the normalized value in :meth:`inverse`. Default 1e-6.
    bias : float, optional
        Offset added to the latent before the slope. Default 0.0.
    """
    def __init__(self, min_val, max_val, slope=1.0, eps=1e-6, bias: float = 0.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.slope = slope
        self.eps = eps
        self.bias = bias

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Map latent (standard normal-like) values into (min_val, max_val).

        Returns ``min_val + (max_val - min_val) * sigmoid((latent + bias) * slope)``.
        """
        sig = torch.sigmoid((latent + self.bias) * self.slope)
        return self.min_val + (self.max_val - self.min_val) * sig

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """
        Map values in [min_val, max_val] to latent; inverse of :meth:`forward`.

        Returns ``logit(x) / slope - bias`` with
        ``x = clamp((value - min_val) / (max_val - min_val), eps, 1 - eps)``.
        """
        x = (value - self.min_val) / (self.max_val - self.min_val)
        x = x.clamp(self.eps, 1 - self.eps)  # avoid exploding logits
        return torch.log(x / (1 - x)) / self.slope - self.bias


class SquashingTransform(nn.Module):
    """
    Odd soft-clipping map from the real line towards (-max_val, max_val).

    :meth:`forward` computes ``max_val * s / (1 + |s| ** softness) ** (1 / softness)``
    with ``s = latent / threshold`` and clips the result to
    [min_val, max_val]; :meth:`inverse` is its analytic inverse.
    ``max_val`` is stored as ``sat_level``.

    Parameters
    ----------
    min_val : float
        Lower clip on the :meth:`forward` output.
    max_val : float
        Saturation level and upper clip on the :meth:`forward` output.
    eps : float, optional
        :meth:`inverse` clamps its input to [min_val + eps, max_val - eps].
        Default 1e-2.
    threshold : float, optional
        Input scale; the slope at 0 is ``max_val / threshold``. Default 5.0.
    softness : float, optional
        Exponent of the saturation. Larger values keep the map closer to
        linear up to ``|latent| ~ threshold`` and saturate more abruptly
        beyond it. Default 5.0.
    """
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
        Soft-clip ``latent`` towards (-sat_level, sat_level).

        Uses the attributes ``threshold``, ``sat_level`` (= ``max_val``) and
        ``softness`` set at construction.

        Parameters
        ----------
        latent : torch.Tensor
            Input of any shape.

        Returns
        -------
        torch.Tensor
            ``sat_level * sign(s) * |s| / (1 + |s| ** softness) ** (1 / softness)``
            with ``s = latent / threshold``, clipped to [min_val, max_val].
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
        Analytic inverse of :meth:`forward`.

        Parameters
        ----------
        squashed : torch.Tensor
            Values to invert, of any shape; first clamped to
            [min_val + eps, max_val - eps].

        Returns
        -------
        torch.Tensor
            ``sign(y) * threshold * (p / (1 - p + 1e-8)) ** (1 / softness)``
            with ``y = clamped / sat_level`` and ``p = |y| ** softness``.
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
    """
    Standardized log of a positive variable, and a clipped exponential back.

    :meth:`forward` maps physical values to latent; :meth:`inverse` maps
    latent to physical values.

    Parameters
    ----------
    mean_log : float, optional
        Subtracted from the log value. Default 0.5.
    std_log : float, optional
        Divides the centred log value. Default 0.35.
    eps : float, optional
        Lower clamp on the value before the log in :meth:`forward`.
        Default 1e-6.
    exp_min, exp_max : float, optional
        Output range of :meth:`inverse`, applied as a clip in log space.
        Default None; :meth:`inverse` calls ``np.log`` on both, which raises
        ``TypeError`` unless both are set.
    """
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
        """
        Map latent to the log-normal physical variable.

        Returns ``exp(clip(latent * std_log + mean_log, log(exp_min), log(exp_max)))``.
        """
        return torch.exp(
            (latent * self.std_log + self.mean_log).clip(min=np.log(self.exp_min), max=np.log(self.exp_max)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        Map the log-normal physical value to latent.

        Returns ``(log(max(value, eps)) - mean_log) / std_log``.
        """
        return (torch.log(value.clip(min=self.eps)) - self.mean_log) / self.std_log


class RotationTransform(nn.Module):
    """
    Map between a 3-column latent and rotation vectors with positive z.

    Rotation vectors are handled in spherical form (theta, phi, r): polar
    angle and azimuth of the rotation axis, and rotation angle (the vector
    norm), all in radians. The latent columns are (theta, phi, r), each
    mapped to the real line. :meth:`forward` maps latent to rotation vector;
    :meth:`inverse` maps rotation vector to latent.

    Parameters
    ----------
    eps : float, optional
        Clamp margin for the [0, 1]-valued intermediates. Default 1e-6.
    mode : str, optional
        ``'linear'`` (default): phi = 2 pi * Phi(latent) - pi, with Phi the
        standard normal CDF, and r = 2 pi * sigmoid(latent). ``'wrapped'``:
        phi is the latent wrapped into [-pi, pi), and
        r = wrap_to_pi(latent) + pi. Any other value leaves the phi and r
        terms unassigned, so :meth:`forward` and :meth:`inverse` raise
        ``UnboundLocalError``.

    Notes
    -----
    In both modes theta = (pi / 2) * sigmoid(latent + pi / 4), which lies in
    (0, pi / 2), so the rotation axis has positive z.
    """
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
        """Wrap angles in radians into [-pi, pi)."""
        # (-pi, pi]
        return (x + math.pi) % (2 * math.pi) - math.pi

    def azimuth_to_std_normal(self, phi):
        """
        Map phi in [-pi, pi] to the real line.

        Returns the standard normal inverse CDF of ``(phi + pi) / (2 pi)``,
        the argument clamped to [eps, 1 - eps].
        """
        # Map from [-π, π] → [0, 1]
        u = (phi + torch.pi) / (2 * torch.pi)
        u = u.clamp(self.eps, 1 - self.eps)
        return self.normal.icdf(u)

    def std_normal_to_azimuth(self, z):
        """Map real ``z`` to ``2 pi * Phi(z) - pi`` in [-pi, pi], with Phi the standard normal CDF."""
        u = self.normal.cdf(z)
        return u * (2 * torch.pi) - torch.pi  # maps back to [-π, π]

    def azimuth_to_uniform(self, phi):
        """Wrap phi into [-pi, pi); the ``'wrapped'``-mode azimuth latent."""
        # latent space and azimuthal angle are identical
        # uniform distributions on [-pi, pi]
        return self.wrap_to_pi(phi)

    def uniform_to_azimuth(self, z):
        """Wrap the ``'wrapped'``-mode azimuth latent into [-pi, pi)."""
        # latent space should be uniform on [-pi, pi]
        # wrap for safety
        return self.wrap_to_pi(z)

    def polar_to_std_normal(self, theta):
        """
        Map theta in [0, pi / 2] to the real line; inverse of :meth:`std_normal_to_polar`.

        Returns ``logit(clip(theta / (pi / 2), eps, 1 - eps)) - pi / 4``.
        """
        u = theta / (torch.pi / 2)
        u = u.clip(self.eps, 1 - self.eps)
        u = torch.log(u) - torch.log(1 - u) - torch.pi / 4  # inverse sigmoid
        return u

    def std_normal_to_polar(self, z):
        """Map real ``z`` to ``(pi / 2) * sigmoid(z + pi / 4)``, in (0, pi / 2)."""
        return (torch.pi / 2) * torch.sigmoid(z + torch.pi / 4)

    def std_normal_to_rotation(self, z):
        """
        Map the rotation-angle latent to r in [0, 2 pi].

        ``'linear'``: ``2 pi * sigmoid(z)``. ``'wrapped'``:
        ``wrap_to_pi(z) + pi``. Any other mode returns None.
        """
        # z ∈ (−∞, ∞) → r ∈ [0, 2π]
        if self.mode == 'linear':
            return 2 * torch.pi * torch.sigmoid(z)
        elif self.mode == 'wrapped':
            # the rotation is naturally std normal with mean pi
            return self.wrap_to_pi(z) + torch.pi

    def rotation_to_std_normal(self, r):
        """
        Map r in [0, 2 pi] to the rotation-angle latent.

        ``'linear'``: ``logit(clamp(r / (2 pi), eps, 1 - eps))``.
        ``'wrapped'``: ``r - pi``, without wrapping. Any other mode returns
        None.
        """
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
        Map latent vectors (R^3) to rotation vectors (SO(3)).

        Uses the canonical z > 0 encoding and norm bounding described on the
        class.

        Parameters
        ----------
        latent : torch.Tensor
            [n, 3] latent columns (theta, phi, r).

        Returns
        -------
        torch.Tensor
            [n, 3] rotation vectors ``r * (sin(theta) cos(phi),
            sin(theta) sin(phi), cos(theta))``, with the rotation angle r
            clipped to [0.01, 2 pi - 0.01] radians.
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
        Map rotation vectors back to latent space; inverse of :meth:`forward`.

        Parameters
        ----------
        rotvec : torch.Tensor
            [n, 3] rotation vectors, converted to (theta, phi, r) with
            ``cart2sph_rotvec``.

        Returns
        -------
        torch.Tensor
            [n, 3] latent columns (theta, phi, r). Axes with theta > pi / 2
            (negative z) reach the ``1 - eps`` clamp in
            :meth:`polar_to_std_normal`.
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
    """
    Standard normal CDF (:meth:`forward`) and inverse CDF (:meth:`inverse`).

    Parameters
    ----------
    eps : float, optional
        :meth:`inverse` clamps its input to [eps, 1 - eps]. Default 1e-6.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer('normal_mean', torch.tensor(0.0))
        self.register_buffer('normal_std', torch.tensor(1.0))
        self.normal = torch.distributions.Normal(self.normal_mean, self.normal_std)

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """Map values in [0, 1], clamped to [eps, 1 - eps], through the standard normal inverse CDF."""
        # Clamp to avoid extreme values
        u = u.clamp(self.eps, 1 - self.eps)
        return self.normal.icdf(u)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map real values to [0, 1] through the standard normal CDF."""
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
    """Affine map [0, 1] -> [-1, 1] (:meth:`forward`) and back; ``eps`` is accepted and unused."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """Return ``u * 0.5 + 0.5``."""
        return u * 0.5 + 0.5

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``(z - 0.5) * 2``."""
        return (z - 0.5) * 2


class ThetaTransform(nn.Module):
    """Affine map of a polar angle [0, pi / 2] -> [-1, 1] (:meth:`forward`) and back; ``eps`` is unused."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """Return ``(pi / 4) * u + pi / 4``."""
        return (1 / 4) * torch.pi * u + torch.pi / 4

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``(z - pi / 4) * 4 / pi``."""
        return (z - torch.pi / 4) / torch.pi * 4


class PhiTransform(nn.Module):
    """Affine map of an azimuth [-pi, pi] -> [-1, 1] (:meth:`forward`) and back; ``eps`` is unused."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """Return ``u * pi``."""
        return u * torch.pi

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``z / pi``."""
        return z / torch.pi


class RTransform(nn.Module):
    """Affine map of a rotation angle [0, 2 pi] -> [-1, 1] (:meth:`forward`) and back; ``eps`` is unused."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        """Return ``u * pi + pi``."""
        return u * torch.pi + torch.pi

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return ``(z - pi) / pi``."""
        return (z - torch.pi) / torch.pi


class UnitTransform(nn.Module):
    """
    Column-wise map between reduced cell parameters and a latent basis.

    Works on [n, 6 + 6 * max_z_prime] tensors laid out as (a / b, b / c,
    normalized c, three angle columns, centroids, orientation rotation
    vectors). Each orientation rotation vector is converted to spherical
    (theta, phi, r) with ``cart2sph_rotvec``, then one scalar transform is
    applied per column. In the :meth:`forward` direction:

    - columns 0 and 1: :class:`UnitToPMUnit`, ``2 x - 1``;
    - column 2: :class:`LogNormalTransform` with ``mean_log=0.25``,
      ``std_log=2``, ``exp_min=0.01``, ``exp_max=4``;
    - columns 3 to 5: :class:`IdentityTransform`;
    - the 3 * max_z_prime centroid columns: :class:`UnitToPMUnit`;
    - each orientation's (theta, phi, r): :class:`ThetaTransform`,
      :class:`PhiTransform`, :class:`RTransform`.

    The column transforms are held in a plain list, not registered as
    submodules.

    Parameters
    ----------
    max_z_prime : int, optional
        Number of asymmetric-unit slots per row. Default 1.
    """
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
        Convert reduced cell parameters to the latent basis.

        Parameters
        ----------
        niggli_params : torch.Tensor
            [n, 6 + 6 * max_z_prime]; the last 3 * max_z_prime columns are
            rotation vectors.

        Returns
        -------
        torch.Tensor
            New [n, 6 + 6 * max_z_prime] tensor, each column mapped by its
            transform's ``forward``. The maps are affine except column 2
            (standardized log) and columns 3 to 5 (identity); inputs in
            [0, 1] for the ratio and centroid columns, and theta in
            [0, pi / 2], phi in [-pi, pi], r in [0, 2 pi] for the
            orientations, land in [-1, 1].
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
        Convert latent parameters back to the input basis of :meth:`forward`.

        Parameters
        ----------
        std_params : torch.Tensor
            [n, 6 + 6 * max_z_prime] latent parameters.

        Returns
        -------
        torch.Tensor
            New [n, 6 + 6 * max_z_prime] tensor, each column mapped by its
            transform's ``inverse`` (column 2 lands in [0.01, 4]), with each
            orientation converted from spherical (theta, phi, r) back to a
            rotation vector by ``sph2cart_rotvec``.
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
    """Pass-through transform; :meth:`forward` and :meth:`inverse` return the input object."""
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """Return ``z`` unchanged."""
        return z

    def inverse(self, z):
        """Return ``z`` unchanged."""
        return z


class CompositeTransform(nn.Module):
    """
    Chain transforms, choosing each call's extra argument from the transform's repr.

    For each transform, ``str(transform).lower()`` is tested: if it
    contains ``'aunit'`` the call receives ``sg_inds``; otherwise, if it
    contains ``'niggli'``, it receives ``mol_radii``; otherwise it receives
    only the tensor.

    Parameters
    ----------
    transforms : list of nn.Module
        Transforms with ``forward`` and ``inverse``; stored as an
        ``nn.ModuleList``.
    """
    def __init__(self, transforms: list[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, sg_inds, mol_radii):
        """
        Apply each transform's ``forward`` in list order.

        Parameters
        ----------
        x : torch.Tensor
            Input to the first transform.
        sg_inds : iterable
            Passed to transforms whose repr contains ``'aunit'``.
        mol_radii : torch.Tensor
            Passed to transforms whose repr contains ``'niggli'``.

        Returns
        -------
        torch.Tensor
            Output of the last transform.
        """
        for transform in self.transforms:
            if 'aunit' in str(transform).lower():
                x = transform.forward(x, sg_inds)
            elif 'niggli' in str(transform).lower():
                x = transform.forward(x, mol_radii)
            else:
                x = transform.forward(x)

        return x

    def inverse(self, x, sg_inds, mol_radii):
        """
        Apply each transform's ``inverse`` in reverse list order.

        Arguments are dispatched as in :meth:`forward`.

        Parameters
        ----------
        x : torch.Tensor
            Input to the last transform's ``inverse``.
        sg_inds : iterable
            Passed to transforms whose repr contains ``'aunit'``.
        mol_radii : torch.Tensor
            Passed to transforms whose repr contains ``'niggli'``.

        Returns
        -------
        torch.Tensor
            Output of the first transform's ``inverse``.
        """
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
    Move cell angles onto the side of the plane where the overlap is non-negative.

    Enforces ``ab * ga_cos + ac * be_cos + bc * al_cos >= 0`` (the overlap
    from :func:`compute_niggli_overlap`, equal to a.b + a.c + b.c for the
    cell vectors). Nothing is changed when no row violates it. In
    ``'mirror'`` mode only violating rows are changed; in ``'shift'`` mode
    every row is passed back through arccos, as described under ``mode``.

    Parameters
    ----------
    cell_lengths : torch.Tensor
        [n, 3] cell lengths (a, b, c). Not modified.
    cell_angles : torch.Tensor
        [n, 3] cell angles (alpha, beta, gamma) in radians. In ``'mirror'``
        mode the violating rows of this tensor are overwritten in place.
    mode : str
        ``'mirror'`` (for random samples): replace every angle of a
        violating row by ``pi - angle``, which negates the three cosines and
        so the overlap, putting the row symmetrically on the correct side.
        ``'shift'``: shift violating rows to the nearest boundary of the
        allowed half-space. For up to 20 iterations, the cosine vector
        (ga_cos, be_cos, al_cos) of each still-violating row is projected
        onto the plane with normal (ab, ac, bc) and moved a further 1e-3
        along the unit normal; all rows are then converted back with arccos
        of the cosines clamped to [-1 + eps, 1 - eps]. The loop stops once
        no row violates the condition.
    eps : float, optional
        Clamp margin for arccos and guard on the squared normal length in
        ``'shift'`` mode. Default 1e-6.

    Returns
    -------
    torch.Tensor
        New [n, 3] tensor of angles (alpha, beta, gamma) in radians.

    Raises
    ------
    ValueError
        If ``mode`` is neither ``'mirror'`` nor ``'shift'`` and at least
        one row violates the condition; the mode is not checked otherwise.

    Notes
    -----
    If any row still violates the condition at the end, the minimum overlap
    is printed; no error is raised.
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
    """
    Compute length products, angle cosines and the overlap a.b + a.c + b.c.

    Note the positional order ``(a, al, b, be, c, ga)``.

    Parameters
    ----------
    a, b, c : torch.Tensor
        Cell lengths.
    al, be, ga : torch.Tensor
        Cell angles alpha, beta, gamma in radians, broadcast-compatible with
        the lengths.

    Returns
    -------
    tuple of torch.Tensor
        ``(ab, ac, al_cos, bc, be_cos, ga_cos, overlap)`` with
        ``ab = a * b``, ``ac = a * c``, ``bc = b * c``, the three cosines,
        and ``overlap = ab * ga_cos + ac * be_cos + bc * al_cos``.
    """
    ab = a * b
    ac = a * c
    bc = b * c
    al_cos = torch.cos(al)
    be_cos = torch.cos(be)
    ga_cos = torch.cos(ga)
    overlap = ab * ga_cos + ac * be_cos + bc * al_cos
    return ab, ac, al_cos, bc, be_cos, ga_cos, overlap

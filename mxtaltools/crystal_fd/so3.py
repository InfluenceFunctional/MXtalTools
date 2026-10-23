"""Rotations as matrices, exact conversions to orientation charts, and Haar densities.

Rotations are stored as [n, 3, 3] float64 matrices; every chart below is a
conversion to and from a matrix, and no chart is the default.

Charts
------
``hemisphere``
    The trainer's (theta, phi, r): rotation vector ``v = r (sin theta cos phi,
    sin theta sin phi, cos theta)`` on the +z hemisphere, theta in [0, pi/2],
    phi in [-pi, pi], r in [0, 2 pi]. This is MXtalTools' convention:
    ``rotmat2rotvec_stable`` (angle in [0, pi]) then ``canonicalize_rotvec``
    (a vector with v_z < 0 becomes ``-v / |v| (2 pi - |v|)``), read by
    ``cart2sph_rotvec``; the latent rows are ``sph_rotvec2lat`` of it.
``quaternion``
    Unit quaternion (w, x, y, z) with w >= 0 (at w = 0, the first nonzero of
    x, y, z is positive); R = R(q) = R(-q).
``euler_zyz``
    (alpha, u, gamma) with u = cos(beta) and R = Rz(alpha) Ry(beta) Rz(gamma);
    alpha, gamma in [0, 2 pi), u in [-1, 1].
``axis_angle``
    The ball vector ``v = omega n`` with omega in [0, pi] (the quaternion's
    angle); at omega = pi the representative is the quaternion's.

``haar_log_density(chart, coords)`` is the log-density of the normalised Haar
measure relative to the chart's Lebesgue measure (for ``quaternion``, relative to
the surface measure of the w >= 0 hemisphere of S^3).
"""
import math

import torch

CHARTS = ('hemisphere', 'quaternion', 'euler_zyz', 'axis_angle')
TWO_PI = 2 * math.pi


# ---------------------------------------------------------------------------
# O(3) helpers
# ---------------------------------------------------------------------------

def nearest_rotation(M: torch.Tensor) -> torch.Tensor:
    """Polar projection of [n, 3, 3] matrices with positive determinant onto SO(3) (SVD)."""
    u, _, vh = torch.linalg.svd(M)
    d = torch.sign(torch.linalg.det(u @ vh))
    u = u.clone()
    u[:, :, 2] = u[:, :, 2] * d[:, None]
    return u @ vh


def split_o3(P: torch.Tensor, project: bool = True):
    """Split poses P [n, 3, 3] (orthogonal up to rounding) into (R [n, 3, 3] proper, h [n] long) with P = R diag(h, 1, 1).

    ``h = sign(det P)``; with ``project`` the proper part is projected onto SO(3)
    (polar decomposition), as ``zp_doubling.proper_rotvecs`` does.
    """
    det = torch.linalg.det(P)
    if bool((det == 0).any()):
        raise ValueError('singular pose')
    h = torch.sign(det)
    flip = torch.ones(P.shape[0], 3, dtype=P.dtype, device=P.device)
    flip[:, 0] = h
    R = P * flip[:, None, :]
    if project:
        R = nearest_rotation(R)
    return R, h.to(torch.long)


def random_rotations(n: int, generator=None, dtype=torch.float64, device=None) -> torch.Tensor:
    """[n, 3, 3] Haar-random rotations (normalised Gaussian quaternions)."""
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64, device=device)
    return quaternion_to_matrix(q / q.norm(dim=1, keepdim=True)).to(dtype)


def _rz(a):
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*a.shape, 3, 3)


def _ry_cs(c, s):
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(*c.shape, 3, 3)


# ---------------------------------------------------------------------------
# quaternions
# ---------------------------------------------------------------------------

def _canonical_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalise and pick the w >= 0 representative (at w = 0: first nonzero of x, y, z positive)."""
    q = q / q.norm(dim=1, keepdim=True)
    v = q[:, 1:]
    first = torch.where(v[:, 0] != 0, v[:, 0], torch.where(v[:, 1] != 0, v[:, 1], v[:, 2]))
    neg = (q[:, 0] < 0) | ((q[:, 0] == 0) & (first < 0))
    return torch.where(neg[:, None], -q, q)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """[n, 3, 3] rotations -> [n, 4] unit quaternions (w, x, y, z), canonical hemisphere.

    Shepperd's method (the largest of w^2, x^2, y^2, z^2 from the diagonal), as
    ``rotmat2rotvec_stable``: no division by a small number at any angle.
    """
    R = R.to(torch.float64)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    cand = torch.stack([m00 + m11 + m22, m00, m11, m22], dim=1)
    k = cand.argmax(dim=1)
    q = torch.empty(len(R), 4, dtype=torch.float64, device=R.device)
    r = R
    s0 = torch.sqrt((1 + m00 + m11 + m22).clamp(min=0)) * 2
    s1 = torch.sqrt((1 + m00 - m11 - m22).clamp(min=0)) * 2
    s2 = torch.sqrt((1 - m00 + m11 - m22).clamp(min=0)) * 2
    s3 = torch.sqrt((1 - m00 - m11 + m22).clamp(min=0)) * 2
    one = torch.ones_like(s0)
    s0, s1, s2, s3 = (torch.where(k == i, s, one) for i, s in enumerate((s0, s1, s2, s3)))
    c0 = torch.stack([s0 / 4, (r[:, 2, 1] - r[:, 1, 2]) / s0, (r[:, 0, 2] - r[:, 2, 0]) / s0,
                      (r[:, 1, 0] - r[:, 0, 1]) / s0], 1)
    c1 = torch.stack([(r[:, 2, 1] - r[:, 1, 2]) / s1, s1 / 4, (r[:, 0, 1] + r[:, 1, 0]) / s1,
                      (r[:, 0, 2] + r[:, 2, 0]) / s1], 1)
    c2 = torch.stack([(r[:, 0, 2] - r[:, 2, 0]) / s2, (r[:, 0, 1] + r[:, 1, 0]) / s2, s2 / 4,
                      (r[:, 1, 2] + r[:, 2, 1]) / s2], 1)
    c3 = torch.stack([(r[:, 1, 0] - r[:, 0, 1]) / s3, (r[:, 0, 2] + r[:, 2, 0]) / s3,
                      (r[:, 1, 2] + r[:, 2, 1]) / s3, s3 / 4], 1)
    q = torch.where((k == 0)[:, None], c0, torch.where((k == 1)[:, None], c1, torch.where((k == 2)[:, None], c2, c3)))
    return _canonical_quaternion(q)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """[n, 4] quaternions (w, x, y, z), any norm > 0 -> [n, 3, 3] rotations."""
    q = q.to(torch.float64)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# rotation vectors: the axis-angle ball and the trainer's hemisphere
# ---------------------------------------------------------------------------

def rotvec_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """[n, 3] rotation vectors of any norm -> [n, 3, 3]; exact at |v| = 0 (sin(x/2)/x by sinc)."""
    v = v.to(torch.float64)
    ang = v.norm(dim=1)
    half_sinc = 0.5 * torch.sinc(ang / TWO_PI)  # sin(ang / 2) / ang
    return quaternion_to_matrix(torch.cat([torch.cos(ang / 2)[:, None], v * half_sinc[:, None]], 1))


def matrix_to_rotvec(R: torch.Tensor, hemisphere: bool = False) -> torch.Tensor:
    """[n, 3, 3] -> [n, 3] rotation vectors.

    ``hemisphere=False``: the ball vector, angle in [0, pi] (``axis_angle``).
    ``hemisphere=True``: MXtalTools' convention, a vector with negative z
    replaced by ``-v / |v| (2 pi - |v|)`` (``canonicalize_rotvec``).
    """
    q = matrix_to_quaternion(R)
    vec = q[:, 1:]
    vn = vec.norm(dim=1)
    ang = 2 * torch.atan2(vn, q[:, 0])
    scale = torch.where(vn > 0, ang / vn.clamp(min=1e-300), torch.full_like(vn, 2.0))
    v = vec * scale[:, None]
    if hemisphere:
        flip = v[:, 2] < 0
        safe = ang.clamp(min=1e-300)
        v = torch.where(flip[:, None], -v / safe[:, None] * (TWO_PI - ang)[:, None], v)
    return v


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """[n, 3, 3] -> [n, 3] ball vectors ``omega n``, omega in [0, pi]."""
    return matrix_to_rotvec(R, hemisphere=False)


def axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    return rotvec_to_matrix(v)


def matrix_to_hemisphere(R: torch.Tensor) -> torch.Tensor:
    """[n, 3, 3] -> [n, 3] (theta, phi, r) in the trainer's convention.

    theta = atan2(sqrt(vx^2 + vy^2), vz), phi = atan2(vy, vx), r = |v| of the
    hemisphere rotation vector. At r = 0 the axis is undefined and (0, 0, 0) is
    returned.
    """
    v = matrix_to_rotvec(R, hemisphere=True)
    r = v.norm(dim=1)
    theta = torch.atan2(torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2), v[:, 2])
    phi = torch.atan2(v[:, 1], v[:, 0])
    return torch.stack([theta, phi, r], 1)


def hemisphere_to_matrix(sph: torch.Tensor) -> torch.Tensor:
    """[n, 3] (theta, phi, r) -> [n, 3, 3]; the axis comes from the angles, so r = 0 and 2 pi are exact."""
    sph = sph.to(torch.float64)
    theta, phi, r = sph.unbind(1)
    n = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)], 1)
    return quaternion_to_matrix(torch.cat([torch.cos(r / 2)[:, None], torch.sin(r / 2)[:, None] * n], 1))


# ---------------------------------------------------------------------------
# Euler ZYZ
# ---------------------------------------------------------------------------

def euler_zyz_to_matrix(e: torch.Tensor) -> torch.Tensor:
    """[n, 3] (alpha, u = cos beta, gamma) -> R = Rz(alpha) Ry(beta) Rz(gamma), beta in [0, pi]."""
    e = e.to(torch.float64)
    a, u, g = e.unbind(1)
    u = u.clamp(-1, 1)
    s = torch.sqrt((1 - u * u).clamp(min=0))
    return _rz(a) @ _ry_cs(u, s) @ _rz(g)


def matrix_to_euler_zyz(R: torch.Tensor) -> torch.Tensor:
    """[n, 3, 3] -> [n, 3] (alpha, u, gamma), alpha and gamma in [0, 2 pi).

    alpha + gamma and alpha - gamma are read from the well-conditioned sums
    (R00 + R11, R10 - R01) = (1 + u)(cos, sin)(alpha + gamma) and
    (R11 - R00, -(R10 + R01)) = (1 - u)(cos, sin)(alpha - gamma); the branch
    (alpha, gamma) vs (alpha + pi, gamma + pi) is fixed by sin(beta) >= 0. At
    the poles only alpha + gamma (u = 1) or alpha - gamma (u = -1) is defined
    and the split is the formula's. The matrix round trip is limited by u = cos
    beta, which float64 cannot resolve for beta below about 1.5e-8: its error is
    about eps / sin(beta) (1e-10 at beta = 1e-6) and about 1e-8 at the poles.
    """
    R = R.to(torch.float64)
    u = R[:, 2, 2].clamp(-1, 1)
    sm = torch.atan2(R[:, 1, 0] - R[:, 0, 1], R[:, 0, 0] + R[:, 1, 1])
    df = torch.atan2(-(R[:, 1, 0] + R[:, 0, 1]), R[:, 1, 1] - R[:, 0, 0])
    a = (sm + df) / 2
    g = (sm - df) / 2
    wrong = torch.cos(a) * R[:, 0, 2] + torch.sin(a) * R[:, 1, 2] < 0
    a = torch.where(wrong, a + math.pi, a)
    g = torch.where(wrong, g + math.pi, g)
    return torch.stack([torch.remainder(a, TWO_PI), u, torch.remainder(g, TWO_PI)], 1)


# ---------------------------------------------------------------------------
# conversions by name, and Haar densities
# ---------------------------------------------------------------------------

_TO = {'hemisphere': matrix_to_hemisphere, 'quaternion': matrix_to_quaternion,
       'euler_zyz': matrix_to_euler_zyz, 'axis_angle': matrix_to_axis_angle}
_FROM = {'hemisphere': hemisphere_to_matrix, 'quaternion': quaternion_to_matrix,
         'euler_zyz': euler_zyz_to_matrix, 'axis_angle': axis_angle_to_matrix}


def to_chart(chart: str, R: torch.Tensor) -> torch.Tensor:
    """Coordinates of rotations [n, 3, 3] in ``chart`` (one of ``CHARTS``)."""
    return _TO[_check(chart)](R)


def from_chart(chart: str, coords: torch.Tensor) -> torch.Tensor:
    """Rotations [n, 3, 3] from coordinates in ``chart``."""
    return _FROM[_check(chart)](coords)


def _check(chart):
    if chart not in CHARTS:
        raise ValueError(f'unknown orientation chart {chart!r}; expected one of {CHARTS}')
    return chart


def haar_log_density(chart: str, coords: torch.Tensor) -> torch.Tensor:
    """Log-density [n] of the normalised Haar measure relative to the chart's Lebesgue measure.

    - ``hemisphere`` (theta, phi, r): ``log(sin^2(r/2) sin(theta)) - log(2 pi^2)``;
      -inf at r = 0, 2 pi and theta = 0.
    - ``euler_zyz`` (alpha, u, gamma): ``-log(8 pi^2)``.
    - ``axis_angle`` v: ``2 log(sin(|v|/2) / |v|) - log(2 pi^2)``, finite at v = 0.
    - ``quaternion``: ``-log(pi^2)`` relative to the surface measure of the
      w >= 0 hemisphere of S^3.
    """
    coords = coords.to(torch.float64)
    _check(chart)
    if chart == 'hemisphere':
        theta, r = coords[:, 0], coords[:, 2]
        return 2 * torch.log(torch.sin(r / 2).abs()) + torch.log(torch.sin(theta).abs()) - math.log(2 * math.pi ** 2)
    if chart == 'euler_zyz':
        return torch.full((len(coords),), -math.log(8 * math.pi ** 2), dtype=torch.float64, device=coords.device)
    if chart == 'axis_angle':
        ang = coords.norm(dim=1)
        return 2 * torch.log(0.5 * torch.sinc(ang / TWO_PI)) - math.log(2 * math.pi ** 2)
    return torch.full((len(coords),), -math.log(math.pi ** 2), dtype=torch.float64, device=coords.device)

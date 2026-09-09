"""
Batched, differentiable primitives for internal-coordinate <-> Cartesian conversion.

Everything here operates on flat ``[M, 3]`` position tensors and is topology-agnostic;
see ``topology.py`` for tree construction and ``builder.py`` for the batched driver.

Conventions (fixed here, relied on everywhere else)
---------------------------------------------------
For a new atom ``n`` placed against reference atoms ``(a, b, c)``:

    r     = |c - n|                bond length,  c is bonded to n
    theta = angle(b, c, n)         in (0, pi)
    phi   = dihedral(a, b, c, n)   in (-pi, pi], IUPAC sign convention

``place_nerf`` and ``dihedral``/``bond_angle`` are exact inverses of one another
(verified to machine precision in the test suite).
"""

import torch

# axis constants, materialised lazily per dtype/device
_EX = (1.0, 0.0, 0.0)
_EY = (0.0, 1.0, 0.0)
_EZ = (0.0, 0.0, 1.0)


def _unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _axis(vec, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(vec, dtype=ref.dtype, device=ref.device)


def bond_length(pc: torch.Tensor, pn: torch.Tensor) -> torch.Tensor:
    """|c - n| for each row."""
    return torch.linalg.norm(pn - pc, dim=-1)


def bond_angle(pb: torch.Tensor, pc: torch.Tensor, pn: torch.Tensor) -> torch.Tensor:
    """Angle b-c-n in (0, pi).

    Uses atan2(|u x v|, u.v) rather than acos(u.v), which is ill-conditioned near
    0 and pi -- exactly where linear centres (alkynes, nitriles, azides) live.
    """
    u = pb - pc
    v = pn - pc
    return torch.atan2(torch.linalg.norm(torch.linalg.cross(u, v, dim=-1), dim=-1),
                       (u * v).sum(-1))


def dihedral(pa: torch.Tensor, pb: torch.Tensor, pc: torch.Tensor, pn: torch.Tensor) -> torch.Tensor:
    """Proper dihedral a-b-c-n in (-pi, pi], IUPAC sign convention.

    Matches RDKit ``rdMolTransforms.GetDihedralRad``.
    """
    b0 = pa - pb
    b1 = _unit(pc - pb)
    b2 = pn - pc
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    x = (v * w).sum(-1)
    y = (torch.linalg.cross(b1, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


def wilson_angle(pi: torch.Tensor, pj: torch.Tensor, pk: torch.Tensor,
                 pn: torch.Tensor) -> torch.Tensor:
    """Out-of-plane angle of bond j->i from the plane (j, k, n). ``j`` is CENTRAL.

    In (-pi/2, pi/2]: zero when i lies in the k-j-n plane, +-pi/2 when the bond is
    normal to it. Every energy term that uses this squares it, so the sign convention
    is not load-bearing here.
    """
    a = pi - pj
    n = _unit(torch.linalg.cross(pk - pj, pn - pj, dim=-1))
    s = (a * n).sum(-1) / a.norm(dim=-1).clamp_min(1e-12)
    return torch.asin(s.clamp(-1.0, 1.0))


def place_nerf(pa: torch.Tensor, pb: torch.Tensor, pc: torch.Tensor,
               r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Natural Extension Reference Frame placement.

    Returns the position of the new atom given three placed reference atoms and
    ``(r, theta, phi)`` as defined in the module docstring. Fully batched over the
    leading dimension; this is the only kernel used for atoms at tree depth >= 3.
    """
    bc = _unit(pc - pb)
    n = _unit(torch.linalg.cross(pb - pa, bc, dim=-1))
    m2 = torch.linalg.cross(n, bc, dim=-1)

    sin_t = torch.sin(theta)
    d = ((-r * torch.cos(theta)).unsqueeze(-1) * bc
         + (r * sin_t * torch.cos(phi)).unsqueeze(-1) * m2
         + (r * sin_t * torch.sin(phi)).unsqueeze(-1) * n)
    return pc + d


def _sinc(rho2: torch.Tensor, rho: torch.Tensor, small: torch.Tensor) -> torch.Tensor:
    """``sin(rho) / rho``, finite and differentiable through rho = 0.

    The series is not an optimisation. `sin(rho)/rho` is 0/0 at the origin, and autograd
    reaches the same singularity one derivative earlier -- d(rho)/du = u/rho is NaN at rho = 0
    even where the composed value is perfectly finite. Both branches of a `where` are
    evaluated, so the guard has to be inside the arithmetic rather than around it.
    """
    return torch.where(small, 1.0 - rho2 / 6.0 + rho2 * rho2 / 120.0, torch.sin(rho) / rho)


def _cosc(rho2: torch.Tensor, rho2_safe: torch.Tensor, rho: torch.Tensor,
          small: torch.Tensor) -> torch.Tensor:
    """``(1 - cos(rho)) / rho^2``, finite and differentiable through rho = 0.

    Used as ``cos(rho) = 1 - rho2 * cosc``, which keeps the whole placement a function of
    ``rho2 = u^2 + v^2`` and never of ``rho`` alone -- so no derivative ever divides by the
    radius.

    DIVIDES BY THE CLAMPED ``rho2_safe``, NOT ``rho2``. Both branches of a `torch.where` are
    evaluated AND differentiated, so a 0/0 in the branch that is discarded still returns NaN
    through the mask -- 0 * NaN is NaN. Measured: with the raw ``rho2`` here the placement's
    VALUE was exact to 4e-16 everywhere while its gradient at u = v = 0 was NaN, which is the
    quiet half of this failure and the half that would have reached the policy.
    """
    return torch.where(small, 0.5 - rho2 / 24.0 + rho2 * rho2 / 720.0,
                       (1.0 - torch.cos(rho)) / rho2_safe)


def place_nerf_transverse(pa: torch.Tensor, pb: torch.Tensor, pc: torch.Tensor,
                          r: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                          small_eps: float = 1e-8) -> torch.Tensor:
    """NeRF placement in TRANSVERSE bending coordinates, regular at a linear centre.

    ``(theta, phi)`` is a polar chart on the bend: with ``rho = pi - theta`` the bend
    magnitude and ``phi`` the bending plane, ``rho = 0`` is the pole, where ``phi`` is
    undefined and ``sin(theta) d theta d phi`` collapses. A LINEAR EQUILIBRIUM SITS EXACTLY
    THERE, which is why an alkyne or nitrile breaks an atom-tree chart -- the motion is
    perfectly physical, the coordinates are not.

    ``u = rho cos(phi)``, ``v = rho sin(phi)`` are the Cartesian components of the same
    bend. They are regular at the pole (both zero, no undefined direction) and carry the
    same two degrees of freedom, so nothing is added or lost. Substituting into the standard
    placement and using ``-cos(theta) = cos(rho)``, ``sin(theta) = sin(rho)``:

        d = r cos(rho) bc  +  r u sinc(rho) m2  +  r v sinc(rho) n

    which is evaluated DIRECTLY in ``(u, v)``. Recovering ``rho`` and ``phi`` by
    ``atan2(v, u)`` and calling the ordinary placement would put the pole straight back into
    the computational graph -- the values would agree and the gradients would not.

    The measure follows from the same substitution and is the reason this is the right
    change rather than a reparameterisation of convenience:

        sin(theta) d theta d phi = (sin(rho) / rho) du dv

    so the divergent ``log sin(theta)`` becomes ``log sinc(rho)``, which is smooth and
    vanishes at the pole. See :func:`log_jacobian`.
    """
    bc = _unit(pc - pb)
    n = _unit(torch.linalg.cross(pb - pa, bc, dim=-1))
    m2 = torch.linalg.cross(n, bc, dim=-1)

    rho2 = u * u + v * v
    # clamped so the unselected branch of each `where` is finite in VALUE AND IN GRADIENT;
    # `small` decides which branch is used, and the clamp only ever bites inside the region
    # `small` discards. Every division below uses the clamped forms.
    rho2_safe = rho2.clamp_min(small_eps * small_eps)
    rho = torch.sqrt(rho2_safe)
    small = rho2 < small_eps
    sinc = _sinc(rho2, rho, small)
    cos_rho = 1.0 - rho2 * _cosc(rho2, rho2_safe, rho, small)

    d = ((r * cos_rho).unsqueeze(-1) * bc
         + (r * u * sinc).unsqueeze(-1) * m2
         + (r * v * sinc).unsqueeze(-1) * n)
    return pc + d


def log_sinc(u: torch.Tensor, v: torch.Tensor, small_eps: float = 1e-8) -> torch.Tensor:
    """``log(sin(rho) / rho)`` for ``rho = hypot(u, v)`` -- the transverse measure term.

    THE WHOLE POINT OF THE CHART, in one line. The polar volume element contributes
    ``log sin(theta)``, which diverges to ``-inf`` at the linear geometry: an infinitely
    repulsive wall placed exactly where an alkyne or nitrile wants to sit, and a coordinate
    artifact rather than physics. The transverse element contributes ``log sinc(rho)``, which
    is smooth and equal to ZERO there. Nothing is being softened -- the two densities are the
    same density in different coordinates; only one of them is expressible.

    The singularity is not removed, it is MOVED, and that is the trade being made: sinc
    vanishes at ``rho = pi``, which is ``theta = 0`` -- the placed atom folded back onto the
    b-c axis, on top of its own grandparent. A vanishing measure there is correct, because
    that geometry is sterically forbidden, whereas the linear one is a real equilibrium.

    Valid on the OPEN DISC ``rho < pi``. Outside it the chart is not injective and ``sin rho``
    turns negative; this returns ``-inf`` there rather than NaN, so an out-of-domain row is a
    zero-probability row a caller can see and count, not a poisoned batch. Keeping the sampler
    inside the disc is the scaling's job, not this function's.
    """
    rho2 = u * u + v * v
    rho2_safe = rho2.clamp_min(small_eps * small_eps)
    rho = torch.sqrt(rho2_safe)
    s = _sinc(rho2, rho, rho2 < small_eps)
    # clamped INSIDE the log: `where` differentiates the branch it discards, so a log of a
    # negative sinc would return NaN through the mask even for rows the mask rejects.
    lg = torch.log(s.clamp_min(torch.finfo(s.dtype).tiny))
    return torch.where(rho2 < torch.pi ** 2, lg,
                       torch.full_like(lg, -float('inf')))


def measure_transverse(pa: torch.Tensor, pb: torch.Tensor, pc: torch.Tensor,
                       pn: torch.Tensor, small_eps: float = 1e-8):
    """Cartesian -> ``(u, v)``, the exact inverse of :func:`place_nerf_transverse`.

    NOT ``transverse_from_polar(bond_angle(...), dihedral(...))``. That route measures the
    azimuth with ``atan2`` on a frame that is degenerate precisely at the geometry this chart
    exists to represent, so the answer would be numerical noise multiplied by a bend of zero
    size -- correct in the limit, meaningless in practice.

    Projecting onto the placement frame instead, with ``w = (n - c)/r``:

        cos(rho) = w . bc,   p = w . m2,   q = w . n_hat,   hypot(p, q) = sin(rho)

    so ``rho = atan2(hypot(p, q), w . bc)`` and ``(u, v) = (p, q) / sinc(rho)``. This
    ``atan2`` is regular at the pole -- it is the POLAR one, reading ``atan2(0, 1) = 0``,
    where the azimuthal ``atan2(v, u)`` is the singular one. The division by ``sinc`` is the
    safe series near zero and tends to ``(u, v) = (p, q)``, as it must.
    """
    bc = _unit(pc - pb)
    nh = _unit(torch.linalg.cross(pb - pa, bc, dim=-1))
    m2 = torch.linalg.cross(nh, bc, dim=-1)
    w = _unit(pn - pc)

    p = (w * m2).sum(-1)
    q = (w * nh).sum(-1)
    s2 = p * p + q * q
    sin_rho = torch.sqrt(s2.clamp_min(small_eps * small_eps))
    rho = torch.atan2(sin_rho, (w * bc).sum(-1))
    rho2 = rho * rho
    sinc = _sinc(rho2, rho.clamp_min(small_eps), rho2 < small_eps)
    return p / sinc, q / sinc


def transverse_from_polar(theta: torch.Tensor, phi: torch.Tensor):
    """``(theta, phi) -> (u, v)``. For converting a stored reference, NOT for the hot path."""
    rho = torch.pi - theta
    return rho * torch.cos(phi), rho * torch.sin(phi)


def polar_from_transverse(u: torch.Tensor, v: torch.Tensor):
    """``(u, v) -> (theta, phi)``, the inverse of :func:`transverse_from_polar`.

    SINGULAR AT u = v = 0 by construction -- that is the pole the transverse pair exists to
    avoid. Provided for reporting and tests only; never call it inside a placement or a
    density, or the singularity returns through the back door.
    """
    rho = torch.sqrt(u * u + v * v)
    return torch.pi - rho, torch.atan2(v, u)


def place_seed_second(pc: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Second atom of a molecule: distance ``r`` from the root along +x."""
    return pc + r.unsqueeze(-1) * _axis(_EX, r)


def place_seed_third(pb: torch.Tensor, pc: torch.Tensor,
                     r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Third atom: ``r`` from ``c`` at angle ``theta`` to ``b-c``, in a canonical plane.

    The dihedral is undefined at this point (only two atoms are placed), so the
    remaining rotational freedom is spent fixing the molecular frame -- this is
    where 3 of the 6 external DoF get consumed.
    """
    bc = _unit(pc - pb)
    ez = _axis(_EZ, r).expand_as(bc)
    ey = _axis(_EY, r).expand_as(bc)
    cross_z = torch.linalg.cross(bc, ez, dim=-1)
    # fall back to ey when bc is (anti)parallel to ez
    degenerate = (torch.linalg.norm(cross_z, dim=-1, keepdim=True) < 1e-6)
    perp = _unit(torch.where(degenerate, torch.linalg.cross(bc, ey, dim=-1), cross_z))

    return pc + ((-r * torch.cos(theta)).unsqueeze(-1) * bc
                 + (r * torch.sin(theta)).unsqueeze(-1) * perp)

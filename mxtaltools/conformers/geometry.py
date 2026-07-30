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

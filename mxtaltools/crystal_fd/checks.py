"""Checks shipped with the package: literal crystal comparison, orbit counts and chart round trips.

``same_crystal`` builds every molecule of each Z' = 1 row literally, in float64,
from the table's exact operations of G (not from the live float32 builder), and
returns the largest atom deviation in Angstrom.
"""
from typing import NamedTuple, Optional

import torch

from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import so3
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
from mxtaltools.crystal_fd.domains import _PAD, Section, in_asu, translation_section
from mxtaltools.crystal_fd.ops import cartesian_rotation
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
_NEIGHBOURS = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3).to(F64)


def unit_cell_fractional(desc: CrystalDescription, mol: MoleculeRef) -> torch.Tensor:
    """[B, |G|, n_atoms, 3] float64 fractional coordinates of every molecule of each Z' = 1 row.

    Molecule g sits at centroid W_g t + w_g wrapped to [0, 1)^3, with fractional
    offsets W_g T^-1 P X (P = R diag(h, 1, 1), X the centred body coordinates); every
    row must have the same space group.
    """
    d = desc.double()
    if bool((d.z_prime != 1).any()):
        raise NotImplementedError("Z' > 1")
    sg = int(d.sg[0])
    if bool((d.sg != sg).any()):
        raise ValueError('every row must have the same space group')
    tab = group_table(sg, d.device)
    W, w = tab.g_W, tab.g_w
    T = d.cell_matrix()
    X = mol.pos.to(device=d.device, dtype=F64)
    off = torch.linalg.solve(T, d.pose[:, 0] @ X.T).transpose(1, 2)            # [B, n, 3]
    cen = torch.einsum('gij,bj->bgi', W, d.t[:, 0]) + w
    cen = cen - torch.floor(cen)
    return cen[:, :, None] + torch.einsum('gij,bnj->bgni', W, off)


def same_crystal(a: CrystalDescription, b: CrystalDescription, mol: MoleculeRef, cell_rtol: float = 1e-8,
                 max_elements: int = 2 ** 24) -> torch.Tensor:
    """[B] largest distance (Angstrom) from an atom of either crystal to the same atom of its best-matching molecule.

    Atoms are compared in order within a molecule, each difference at its
    minimum lattice image (rounded, then the 27 neighbouring images; exact for
    reduced cells); molecules are matched to their nearest counterpart, both ways.
    Row i of ``a`` and of ``b`` must share space group and cell (relative
    ``cell_rtol``): re-express one in the other's basis first (``ops.change_basis``).
    Differences are measured in the cell of ``a``. At most about ``max_elements``
    float64 values are held at once (chunked over rows, then over molecules of ``a``).
    """
    a, b = a.double(), b.double()
    if a.num_rows != b.num_rows or not torch.equal(a.sg, b.sg):
        raise ValueError('rows must pair up with equal space groups')
    if ((a.cell - b.cell).abs() > cell_rtol * a.cell.abs().clamp(min=1)).any():
        raise ValueError('cells differ: re-express one description in the other basis first (ops.change_basis)')
    out = torch.empty(a.num_rows, dtype=F64, device=a.device)
    nb = _NEIGHBOURS.to(a.device)
    for sg in torch.unique(a.sg).tolist():
        rows = torch.nonzero(a.sg == sg)[:, 0]
        k = len(group_table(sg).g_W)
        per_mol = k * len(mol.z) * 27 * 3                                       # values per molecule of a row of a
        chunk = max(1, max_elements // (k * per_mol))
        kc = max(1, min(k, max_elements // per_mol))
        for lo in range(0, len(rows), chunk):
            r = rows[lo:lo + chunk]
            fa, fb = unit_cell_fractional(a[r], mol), unit_cell_fractional(b[r], mol)
            T = a[r].cell_matrix()
            dist = torch.empty(len(r), k, k, dtype=F64, device=a.device)        # [b, k, l]
            for j in range(0, k, kc):
                df = fa[:, j:j + kc, None] - fb[:, None]                        # [b, kc, l, n, 3]
                df = df - torch.round(df)
                cart = torch.einsum('bij,bklnsj->bklnsi', T, df[..., None, :] + nb)
                dist[:, j:j + kc] = cart.norm(dim=-1).amin(-1).amax(-1)
            out[r] = torch.maximum(dist.amin(2).amax(1), dist.amin(1).amax(1))
    return out


# ---------------------------------------------------------------------------
# orbit counts
# ---------------------------------------------------------------------------

def random_cells(lattice_type: str, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """[n, 6] random cells of a family: lengths uniform in [4, 16] Angstrom, angles per family (radians).

    Triclinic angles are uniform in [65, 115] degrees, redrawn until the cell has
    positive volume; monoclinic beta in [95, 125]; trigonal groups use the hexagonal family.
    """
    L = 4 + 12 * torch.rand(n, 3, generator=generator, dtype=F64)
    deg = torch.full((n, 3), 90.0, dtype=F64)
    if lattice_type == 'triclinic':
        deg = 65 + 50 * torch.rand(n, 3, generator=generator, dtype=F64)
        bad = torch.ones(n, dtype=torch.bool)
        for _ in range(100):  # redraw rows until the Gram determinant is positive; bounded
            deg[bad] = 65 + 50 * torch.rand(int(bad.sum()), 3, generator=generator, dtype=F64)
            ca, cb, cg = torch.cos(torch.deg2rad(deg)).unbind(1)
            bad = 1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg <= 1e-3
            if not bool(bad.any()):
                break
        else:
            raise RuntimeError('triclinic cell sampling did not reach positive volumes in 100 rounds')
    elif lattice_type == 'monoclinic':
        deg[:, 1] = 95 + 30 * torch.rand(n, generator=generator, dtype=F64)
    elif lattice_type == 'hexagonal':
        deg[:, 2] = 120.0
    elif lattice_type not in ('orthorhombic', 'tetragonal', 'cubic'):
        raise ValueError(lattice_type)
    if lattice_type in ('tetragonal', 'hexagonal'):
        L[:, 1] = L[:, 0]
    if lattice_type == 'cubic':
        L[:, 1] = L[:, 2] = L[:, 0]
    return torch.cat([L, torch.deg2rad(deg)], 1)


class OrbitCount(NamedTuple):
    """Result of ``orbit_check``: rows checked, rows whose orbit has other than one in-domain image, and the count range."""
    n: int
    n_not_one: int
    min_count: int
    max_count: int
    counts: torch.Tensor


def orbit_check(sg: int, n: int = 10_000, section: Optional[Section] = None, seed: int = 0, tol: float = 1e-9,
                chunk: int = 512) -> OrbitCount:
    """Count the in-domain images of the Gamma+ orbits of ``n`` random Z' = 1 poses (a fundamental domain gives 1).

    Poses: t uniform in [0, 1)^3, Haar R, random cells of the family. Every Gamma+
    element is applied with every lattice shift that reaches the domain's bounding
    box (free coordinates at the gauge), and ``domains.in_asu`` (the rules, independent
    of ``fold``) counts the images inside.
    """
    sec = translation_section(sg) if section is None else section
    g = torch.Generator().manual_seed(seed)
    cell = random_cells(LATTICE_TYPE[sg], n, g)
    t = torch.rand(n, 3, generator=g, dtype=F64)
    R = so3.random_rotations(n, g)
    counts = torch.empty(n, dtype=torch.long)
    m, S = sec.order, sec.shifts.shape[0]
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        b = cell[sl].shape[0]
        T = CrystalDescription.from_parts(sg, cell[sl], t[sl], R[sl]).cell_matrix()
        y = torch.einsum('mij,bj->bmi', sec.W, t[sl]) + sec.w
        y = torch.where(sec.free, sec.gauge, y)
        nlo = torch.where(sec.free, torch.zeros_like(y), torch.ceil(sec.region.lo - _PAD - y))
        x = y[:, :, None] + nlo[:, :, None] + sec.shifts.to(F64)                    # [b, m, S, 3]
        x = torch.where(sec.free, sec.gauge, x)
        Q = cartesian_rotation(T[:, None].expand(b, m, 3, 3).reshape(-1, 3, 3),
                               sec.W[None].expand(b, m, 3, 3).reshape(-1, 3, 3)).reshape(b, m, 3, 3)
        Rimg = (Q @ R[sl][:, None])[:, :, None].expand(b, m, S, 3, 3)
        cand = CrystalDescription.from_parts(sg, cell[sl][:, None, None].expand(b, m, S, 6).reshape(-1, 6),
                                             x.reshape(-1, 3), Rimg.reshape(-1, 3, 3))
        inside, _ = in_asu(cand, sec, tol)
        counts[sl] = inside.reshape(b, m * S).sum(1)
    return OrbitCount(n, int((counts != 1).sum()), int(counts.min()), int(counts.max()), counts)


# ---------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------

class RoundTrip(NamedTuple):
    """Result of ``round_trip``: the returned descriptions and the largest deviation of each part.

    ``cell`` (Angstrom and radians), ``t`` (fractional), ``R`` (matrix entries),
    ``atoms`` (Angstrom; None without a molecule).
    """
    desc: CrystalDescription
    cell: float
    t: float
    R: float
    atoms: Optional[float]


def round_trip(desc: CrystalDescription, path, mol: Optional[MoleculeRef] = None,
               cell_rtol: float = 1e-8) -> RoundTrip:
    """desc -> ``path`` -> desc, and how far it moved.

    ``path`` is a chart (``inverse`` then ``forward``) or a callable desc -> desc.
    With ``mol``, atoms are compared by ``same_crystal`` (``cell_rtol``: allowed
    relative cell change; a float32 path such as ``LatentV1Chart`` needs about 1e-5).
    """
    d = desc.double()
    if callable(path) and not hasattr(path, 'inverse'):
        back = path(d).double()
    else:
        y, _ = path.inverse(d)
        back = path.forward(y).desc.double()
    atoms = None if mol is None else float(same_crystal(d, back, mol, cell_rtol=cell_rtol).max())
    return RoundTrip(back, float((back.cell - d.cell).abs().max()), float((back.t - d.t).abs().max()),
                     float((back.R - d.R).abs().max()), atoms)

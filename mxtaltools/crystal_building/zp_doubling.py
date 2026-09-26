"""
Z'=1 -> Z'=2 re-description of a P2_1/c (sg 14, standard setting) crystal.

A Z'=1 sg14 crystal with cell (a, b, c, beta) is exactly a Z'=2 sg14 crystal on an index-2 supercell. To keep all
four SYM_OPS[14] operators in the same setting, the supercell must keep b (the 2_1 screw squares to b) and c (the
c-glide squares to c), which leaves one sublattice, {n a + m b + l c : n even}. With A = 2a, molecule 0 sits at
M^-1 f and molecule 1 at M^-1 (f + (1, 0, 0)) with the same pose (M = diag(2, 1, 1)); every other basis of that
sublattice keeping the SYM_OPS form (a' = pA + qc, b' = det b, c' = rA + sc, r even, s odd, det = ps - qr = +-1)
describes the same crystal. By default the basis is chosen with zero monoclinic reduction penalty -- the cell the
search's `enforce_reduced` keeps -- preferring the identity, then det = +1, then the shortest a' + c'.

Conventions (mxtaltools): a along x, b in the xy plane (batch_compute_fractional_transform), molecule atoms =
R(rotvec) diag(h, 1, 1) X_std + T_fc f. A new a' direction rotates the Cartesian frame by U = T_fc(new) V^-1 (V: new
lattice vectors in the old frame), so poses become U R diag(h, 1, 1). When b' = -b the doubled centres land at
negative y; both molecules are then re-expressed through the inversion (f -> -f, pose -> -pose, handedness flips),
as MolCrystalOps._transform_aunit_params would.

The result is one exact Z'=2 description per Z'=1 crystal: same energy per molecule, same packing, and -- for the
12 lowest acridine Z'=1 families checked on 2026-09-25 -- a local minimum under symmetry breaking too. It is not a
generator of genuinely Z'=2 packings.
"""
import itertools
from typing import Optional, Union

import torch

from mxtaltools.common.geometry_utils import batch_compute_fractional_transform, rotvec2rotmat, rotmat2rotvec
from mxtaltools.common.sym_utils import cell_reduction_penalty
from mxtaltools.crystal_building.utils import canonicalize_rotvec

SG_DOUBLED = 14
#: half-lattice origin shifts of the parent that keep SYM_OPS[14]'s form (they move the origin between inversion
#: centres); used only to move the doubled centres away from the periodic x / z faces of the asymmetric unit
ORIGINS = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.5), (0.5, 0.0, 0.5))
_CANDIDATES = {}


def proper_rotvecs(poses: torch.Tensor):
    """poses [n, 3, 3] (orthogonal up to rounding, either determinant) -> (canonical rotvecs [n, 3], handedness [n]),
    with pose = R(rotvec) diag(handedness, 1, 1). The proper part is first projected onto the nearest rotation (polar
    decomposition): rotmat2rotvec reads the axis from the antisymmetric part, which amplifies a non-orthogonality of
    ~1e-7 (e.g. from float32 cell angles) by 1/sin(angle) near a rotation by pi. Rotvecs are then put on
    canonicalize_rotvec's +z hemisphere, the form the latent transforms expect."""
    h = torch.sign(torch.linalg.det(poses))
    flip = torch.ones(len(poses), 3, dtype=poses.dtype, device=poses.device)
    flip[:, 0] = h
    rot = poses * flip[:, None, :]  # pose @ diag(h, 1, 1)
    u, _, vh = torch.linalg.svd(rot)
    rot = u @ vh
    return canonicalize_rotvec(rotmat2rotvec(rot)), h


def _t_fc(lengths, angles):
    t_fc, _, _ = batch_compute_fractional_transform(torch.as_tensor(lengths, dtype=torch.float64)[None],
                                                    torch.as_tensor(angles, dtype=torch.float64)[None])
    return t_fc[0]


def _basis_candidates(q_max: int = 120, p_max: int = 5, r_max: int = 8):
    """(p, q, r, s, det) with r even, s odd, det = ps - qr = +-1: the ac-plane basis changes of the doubled lattice
    that keep SYM_OPS[14]'s form (the MONO_GLIDE setting group of sym_utils). The range covers parents far from
    reduced (May search chunks needed |q| up to 41); a parent beyond it raises in double_zp1_params."""
    key = (q_max, p_max, r_max)
    if key not in _CANDIDATES:
        rows = [(p, q, r, s, p * s - q * r)
                for p, q, r, s in itertools.product(range(-p_max, p_max + 1), range(-q_max, q_max + 1),
                                                    range(-r_max, r_max + 1), range(-p_max, p_max + 1))
                if r % 2 == 0 and s % 2 == 1 and abs(p * s - q * r) == 1]
        _CANDIDATES[key] = torch.tensor(rows, dtype=torch.float64)
    return _CANDIDATES[key]


def _cell_of(v):
    """lengths and angles (alpha, beta, gamma) of the lattice whose vectors are the columns of v; alpha and gamma are
    snapped to pi/2 when within 1e-6 (stored monoclinic cells carry float32(pi/2))."""
    lengths = torch.linalg.norm(v, dim=0)
    ang = lambda i, j: torch.arccos((v[:, i] @ v[:, j]) / lengths[i] / lengths[j])
    angles = torch.stack([ang(1, 2), ang(0, 2), ang(0, 1)])
    for i in (0, 2):
        if abs(float(angles[i]) - torch.pi / 2) < 1e-6:
            angles[i] = torch.pi / 2
    return lengths, angles


def reduced_doubling_basis(lengths, angles) -> Optional[torch.Tensor]:
    """3x3 integer basis change P (columns in the (2a, b, c) basis) giving a zero-reduction-penalty doubled cell, or
    None if no candidate within the searched range has zero penalty."""
    v0 = _t_fc(lengths, angles) @ torch.diag(torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64))
    p, q, r, s, det = _basis_candidates().T
    a_new = p[:, None] * v0[:, 0] + q[:, None] * v0[:, 2]
    c_new = r[:, None] * v0[:, 0] + s[:, None] * v0[:, 2]
    la, lc = torch.linalg.norm(a_new, dim=1), torch.linalg.norm(c_new, dim=1)
    beta = torch.arccos(((a_new * c_new).sum(1) / la / lc).clamp(-1, 1))
    lb = torch.full_like(la, float(torch.linalg.norm(v0[:, 1])))
    half_pi = torch.full_like(beta, torch.pi / 2)
    pen = cell_reduction_penalty(torch.stack([half_pi, beta, half_pi], 1).float(), torch.stack([la, lb, lc], 1).float(),
                                 torch.full((len(la),), SG_DOUBLED), 0.0)
    ok = torch.nonzero(pen <= 1e-12).flatten()
    if len(ok) == 0:
        return None
    identity = (p == 1) & (q == 0) & (r == 0) & (s == 1)
    # preference: identity first, then det = +1, then the shortest a' + c'
    key = sorted(ok.tolist(), key=lambda k: (not bool(identity[k]), float(det[k]) != 1.0, float(la[k] + lc[k])))
    k = key[0]
    return torch.tensor([[p[k], 0.0, r[k]], [0.0, det[k], 0.0], [q[k], 0.0, s[k]]], dtype=torch.float64)


def double_zp1_params(lengths, angles, centroid, rotvec, handedness: float, reduce: bool = True,
                      origin=(0.0, 0.0, 0.0), basis: Optional[torch.Tensor] = None) -> dict:
    """Z'=1 sg14 parameters (lengths, angles in radians, fractional centroid [3], rotvec [3], handedness) -> the Z'=2
    parameters of the same crystal: dict(lengths [3], angles [3], centroid [2, 3] in [0, 1), rotvec [2, 3],
    handedness [2], M = the doubled lattice vectors in the parent basis (columns), origin, reduction_penalty)."""
    lengths, angles, centroid, rotvec = (torch.as_tensor(x, dtype=torch.float64).reshape(-1)
                                         for x in (lengths, angles, centroid, rotvec))
    origin_t = torch.as_tensor(origin, dtype=torch.float64)
    m0 = torch.diag(torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64))
    if basis is None:
        basis = reduced_doubling_basis(lengths, angles) if reduce else torch.eye(3, dtype=torch.float64)
        if basis is None:
            raise RuntimeError('no zero-reduction-penalty basis found for the doubled cell')
    m = m0 @ basis
    v = _t_fc(lengths, angles) @ m
    new_lengths, new_angles = _cell_of(v)
    u = _t_fc(new_lengths, new_angles) @ torch.linalg.inv(v)
    if not (torch.allclose(u @ u.T, torch.eye(3, dtype=torch.float64), atol=1e-6) and torch.linalg.det(u) > 0):
        raise RuntimeError('the doubled cell is not a proper rotation of the parent lattice')
    m_inv = torch.linalg.inv(m)
    f = centroid - origin_t
    cen = torch.stack([m_inv @ f, m_inv @ (f + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))])
    pose = u @ rotvec2rotmat(rotvec[None])[0] @ torch.diag(torch.tensor([float(handedness), 1.0, 1.0], dtype=torch.float64))
    if basis[1, 1] < 0:  # b' = -b: re-express both molecules through the inversion so the centres land at y >= 0
        cen, pose = -cen, -pose
    cen = cen - torch.floor(cen)
    rv, h = proper_rotvecs(pose[None])
    rv_new, h_new = rv[0], float(h[0])
    pen = cell_reduction_penalty(new_angles[None].float(), new_lengths[None].float(), torch.tensor([SG_DOUBLED]), 0.0)
    return dict(lengths=new_lengths, angles=new_angles, centroid=cen, rotvec=torch.stack([rv_new, rv_new]),
                handedness=torch.tensor([h_new, h_new], dtype=torch.float64), M=m, origin=origin_t,
                reduction_penalty=float(pen[0]))


def _face_margin(centroid):
    """smallest distance (fractional) of any doubled centre from a periodic x or z face."""
    g = centroid[:, [0, 2]]
    return float(torch.minimum(g, 1 - g).min())


def _molecule_of(crystal):
    """the crystal's molecule as MolData. mass, mol_volume and radius are made 0-d: set_mol_attrs SUMS 0-d tensors
    over the Z' molecules (the search's Z'>1 convention) but concatenates anything else, which would leave
    per-molecule values and halve packing_coeff and density."""
    from mxtaltools.dataset_utils.data_classes import MolData
    kw = {k: getattr(crystal, k) for k in ('x', 'smiles', 'identifier')
          if k in crystal.keys() and getattr(crystal, k) is not None}
    for k in ('mol_volume', 'mass', 'radius'):
        if k in crystal.keys() and getattr(crystal, k) is not None:
            kw[k] = torch.as_tensor(getattr(crystal, k)).reshape(()).clone()
    return MolData(z=crystal.z.clone(), pos=crystal.pos.clone(), **kw)


def _unit_cell_frac(crystal):
    """(fractional unit-cell atoms wrapped into [0, 1), element of each, T_fc) via mol2ucell(std_orientation=True),
    the geometry every energy route scores."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    b = collate_data_list([crystal.clone()])
    with torch.no_grad():
        b.mol2ucell(std_orientation=True)
    t_fc = b.T_fc[0].double()
    f = b.unit_cell_pos.double() @ torch.linalg.inv(t_fc).T
    z_mol = crystal.z[:int(crystal.num_atoms) // int(crystal.z_prime)]
    z = z_mol.repeat(f.shape[0] // len(z_mol))
    return f - torch.floor(f), z, t_fc


def doubling_deviation(parent, doubled, m, origin):
    """(largest distance in Angstrom between matched atoms, whether the matching is one-to-one). The parent unit cell
    and its copy shifted by the parent a are mapped into the doubled basis (f_D = M^-1 (f_P - origin)) and matched,
    element by element and modulo the DOUBLED lattice, to the doubled unit cell. Requiring every mapped parent atom to
    be the nearest match of exactly one doubled atom rules out e.g. molecule 1 stacked onto molecule 0, which a
    match modulo the parent lattice cannot see."""
    fp, zp, _ = _unit_cell_frac(parent)
    fd, zd, t_fc_d = _unit_cell_frac(doubled)
    if len(fd) != 2 * len(fp):
        return float('inf'), False
    shift = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    mapped = (torch.cat([fp, fp + shift]) - origin) @ torch.linalg.inv(m).T
    zmap = torch.cat([zp, zp])
    worst, one_to_one = 0.0, True
    for e in torch.unique(zd):
        a, b = fd[zd == e], mapped[zmap == e]
        if len(a) != len(b):
            return float('inf'), False
        d = a[:, None] - b[None]
        d = d - torch.round(d)
        dist = torch.linalg.norm(d @ t_fc_d.T, dim=-1)
        nearest = dist.min(dim=1)
        worst = max(worst, float(nearest.values.max()))
        one_to_one = one_to_one and bool((torch.bincount(nearest.indices, minlength=len(b)) == 1).all())
    return worst, one_to_one


def double_zp1_crystal(crystal, reduce: bool = True, origin: Union[str, tuple] = 'best', check: bool = True,
                       tol: float = 1e-3):
    """Z'=2 MolCrystalData describing the Z'=1 sg14 crystal `crystal` (built the way the search builds Z'=2
    crystals: the molecule duplicated). origin='best' picks the parent origin shift keeping the doubled centres
    farthest from the periodic x / z faces. check=True rebuilds both unit cells and raises unless the doubled atoms
    match the parent's, and its a-shifted copy's, one-to-one within `tol` Angstrom (doubling_deviation). Rotvecs are
    canonical (+z hemisphere) and the attribute shapes are those of search outputs.
    Energies are not copied; per molecule they equal the parent's."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    if int(crystal.sg_ind) != SG_DOUBLED or int(crystal.z_prime) != 1:
        raise ValueError(f"double_zp1_crystal needs a Z'=1 sg{SG_DOUBLED} crystal, got sg{int(crystal.sg_ind)} "
                         f"Z'={int(crystal.z_prime)}")
    if bool(getattr(crystal, 'nonstandard_symmetry', False)):
        raise ValueError('double_zp1_crystal needs the standard SYM_OPS[14] setting')
    args = (crystal.cell_lengths.double().reshape(3), crystal.cell_angles.double().reshape(3),
            crystal.aunit_centroid.double().reshape(-1)[:3], crystal.aunit_orientation.double().reshape(-1)[:3],
            float(crystal.aunit_handedness.reshape(-1)[0]))
    if origin == 'best':
        out = max((double_zp1_params(*args, reduce=reduce, origin=o) for o in ORIGINS),
                  key=lambda d: _face_margin(d['centroid']))
    else:
        out = double_zp1_params(*args, reduce=reduce, origin=origin)
    mol = _molecule_of(crystal)
    doubled = MolCrystalData(
        molecule=[mol, mol.clone()], sg_ind=SG_DOUBLED, z_prime=2, max_z_prime=2,
        cell_lengths=out['lengths'].float(), cell_angles=out['angles'].float(),
        aunit_centroid=out['centroid'].reshape(-1).float(), aunit_orientation=out['rotvec'].reshape(-1).float(),
        aunit_handedness=out['handedness'].float(), identifier=getattr(crystal, 'identifier', None),
        do_box_analysis=True)
    # round trip through a batch: the attribute shapes of search outputs (run_search saves batch_to_list items)
    from mxtaltools.dataset_utils.utils import collate_data_list
    doubled = collate_data_list([doubled]).batch_to_list()[0]
    if check:
        worst, one_to_one = doubling_deviation(crystal, doubled, out['M'], out['origin'])
        if worst > tol or not one_to_one:
            raise RuntimeError(f'doubled crystal does not reproduce its parent: max atom deviation {worst:.2e} A '
                               f'(tol {tol}), one-to-one atom matching: {one_to_one}')
    return doubled, out

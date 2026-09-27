"""
Re-express crystals in their reduced cell, lattice-only (no symmetry search).

The reduced cell of a lattice is the cell at which sym_utils.cell_reduction_penalty is zero at margin 0, among the cells
that describe the crystal with the same operator list SYM_OPS[sg]: the space group's SETTING GROUP of basis changes N (new
basis vectors as the columns of N, in old fractional coordinates) with an origin shift o where the group needs one,
x_new = N^-1 (x_old - o). o is taken from the quarter-cell grid, zero preferred: Cc and C2/c need it (c' = c +- a moves
the glide plane by b/4, so without it about 30% of their lattices cannot reach the reduced cell); P2_1/c does not (c' =
c +- a turns the c-glide into an n-glide, which no shift undoes). At margin 0 that cell is unique per lattice for
monoclinic groups up to the sign of the basis (tests/test_monoclinic_reduction_walls.py) and for triclinic ones in the
tri_niggli convention; higher systems' penalties fix only the crystal-system metric, so any cell with that metric is
reduced and N = I.

Only crystals whose own operator list is SYM_OPS[sg] as a set are handled (a P2_1/n or P2_1/a description of sg 14 is
not: its basis changes are a different group, and the asymmetric-unit box is the standard setting's). standardize_cells
flags such rows and apply_basis_change refuses them.

Why not MolCrystalOps.compute_standard_cell (spglib): on no-wall acridine search outputs it refused 0.15% of crystals
(pseudo-symmetry within 1e-3 A makes spglib report a supercell or a supergroup) and returned a different crystal for
~4e-4 of them (its reparameterisation converts orientations with rotmat2rotvec, which replaces rotations near pi or the
identity by pi about (1, 1, 1)); it also moves the origin of cells that are already reduced. This route never asks for
the crystal's symmetry: it only changes the basis of the stored lattice, within the setting group, and transforms the
stored parameters directly -- centres x_new = N^-1 (x - o), poses rotated by the change of Cartesian frame
Q = T_fc_new N^-1 T_cf_old, converted by rotmat2rotvec_stable -- then puts each molecule's image back in its
asymmetric-unit box with crystal_opt_utils.canonicalize_aunit. same_crystal_deviation checks the result atom by atom.

Choosing N: the stored lattice is first reduced without regard to the setting group (Gauss-Lagrange in the ac-plane for
monoclinic groups, b being the unique axis; Niggli for triclinic ones), then every small correction M (entries within
+-3; signed axis permutations for triclinic) is tried: N = N0 M must keep SYM_OPS[sg] (with some grid shift) and have
det +1 (MXtalTools bases are right-handed). The cell with zero margin-0 penalty is taken: the identity when the stored cell already is
reduced (its parameters are then left exactly as they are), else the one closest to the identity. A crystal with no
zero-penalty candidate raises (or is flagged with on_failure='flag') -- nothing is dropped silently.
"""
import itertools

import numpy as np
import torch

from mxtaltools.common.geometry_utils import batch_compute_fractional_transform, rotvec2rotmat
from mxtaltools.common.sym_utils import cell_reduction_penalty
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_building.zp_doubling import proper_rotvecs

PENALTY_ZERO = 1e-10  # float32 inputs: a cell on a wall evaluates to ~1e-12, not 0
# origin shifts tried by setting_shift, fewest nonzero components first (so o = 0 whenever it works)
_SHIFTS = np.array(sorted(itertools.product((0.0, 0.25, 0.5, 0.75), repeat=3),
                          key=lambda o: (sum(v > 0 for v in o), o)), dtype=np.float64)
_OPS_CACHE = {}
_SHIFT_CACHE = {}
_SETTING_CACHE = {}


def _ops(sg):
    """SYM_OPS[sg] as (W [k,3,3], w [k,3] mod 1), float64."""
    if sg not in _OPS_CACHE:
        o = np.asarray(SYM_OPS[int(sg)], dtype=np.float64)
        _OPS_CACHE[sg] = (o[:, :3, :3], np.mod(o[:, :3, 3], 1.0))
    return _OPS_CACHE[sg]


def _all_in(W, w, Wt, wt, tol):
    """[m] bool: every operator (Wt[k], wt[m, k]) is one of the (W[j], w[j]); translations mod 1."""
    mW = np.abs(Wt[:, None] - W[None]).reshape(len(Wt), len(W), -1).max(-1) < tol  # [k, j]
    d = np.abs(np.mod(wt, 1.0)[:, :, None, :] - w[None, None])  # [m, k, j, 3]
    d = np.minimum(d, 1 - d).max(-1)
    return (mW[None] & (d < tol)).any(-1).all(-1)


def setting_shift(N, sg, tol=1e-6):
    """The origin shift o [3] (old fractional coordinates, from the quarter-cell grid, zero preferred) with which the
    basis change N (integer, det +-1) maps SYM_OPS[sg] onto itself -- x_new = N^-1 (x - o), so the operator (W, w)
    becomes (N^-1 W N, N^-1 (W o + w - o)) -- or None if N is not in the setting group."""
    N = np.asarray(N, dtype=np.float64)
    key = (int(sg), np.round(N).astype(np.int64).tobytes())
    if key not in _SHIFT_CACHE:
        Ninv = np.linalg.inv(N)
        W, w = _ops(sg)
        Wn = Ninv[None] @ W @ N[None]
        wn = np.einsum('ij,mkj->mki', Ninv,
                       np.einsum('kij,mj->mki', W, _SHIFTS) + w[None] - _SHIFTS[:, None, :])  # [m, k, 3]
        ok = _all_in(W, w, Wn, wn, tol)
        _SHIFT_CACHE[key] = _SHIFTS[int(np.argmax(ok))].copy() if ok.any() else None
    o = _SHIFT_CACHE[key]
    return None if o is None else o.copy()


def keeps_operators(N, sg, tol=1e-6):
    """True if the basis change N (integer, det +-1) maps SYM_OPS[sg] onto itself, with an origin shift if needed."""
    return setting_shift(N, sg, tol) is not None


def standard_setting(ops, sg, tol=1e-6):
    """True if a crystal's operator list `ops` [k, 4, 4] is SYM_OPS[sg] as a set (any order; translations mod 1).
    None (no stored list) counts as standard."""
    if ops is None:
        return True
    o = ops.detach().cpu().double().numpy() if torch.is_tensor(ops) else np.asarray(ops, dtype=np.float64)
    key = (int(sg), np.round(o * 1e6).astype(np.int64).tobytes())
    if key not in _SETTING_CACHE:
        W, w = _ops(sg)
        _SETTING_CACHE[key] = (o.ndim == 3 and len(o) == len(W)
                               and bool(_all_in(W, w, o[:, :3, :3], o[None, :, :3, 3], tol)[0]))
    return _SETTING_CACHE[key]


def metric(lengths, angles):
    a, b, c = lengths
    al, be, ga = angles
    return np.array([[a * a, a * b * np.cos(ga), a * c * np.cos(be)],
                     [a * b * np.cos(ga), b * b, b * c * np.cos(al)],
                     [a * c * np.cos(be), b * c * np.cos(al), c * c]])


def cell_from_metric(G):
    L = np.sqrt(np.diag(G))
    al = np.arccos(np.clip(G[1, 2] / (L[1] * L[2]), -1, 1))
    be = np.arccos(np.clip(G[0, 2] / (L[0] * L[2]), -1, 1))
    ga = np.arccos(np.clip(G[0, 1] / (L[0] * L[1]), -1, 1))
    return L, np.array([al, be, ga])


def _gauss_reduce_ac(G):
    """Gauss-Lagrange reduction of the (a, c) plane of metric G. Returns the 3x3 integer N0 (b column untouched)."""
    M = np.eye(2, dtype=np.int64)  # columns: u, v in (a, c) coordinates
    g = np.array([[G[0, 0], G[0, 2]], [G[0, 2], G[2, 2]]])

    def dot(x, y):
        return float(x @ g @ y)
    for _ in range(10000):  # terminates: the norms decrease strictly until the loop breaks
        u, v = M[:, 0], M[:, 1]
        if dot(u, u) > dot(v, v):
            M = M[:, ::-1].copy()
            u, v = M[:, 0], M[:, 1]
        m = int(np.round(dot(u, v) / dot(u, u)))
        if m == 0:
            break
        M[:, 1] = v - m * u
    else:
        raise RuntimeError('Gauss reduction did not terminate')
    N0 = np.eye(3, dtype=np.int64)
    N0[0, 0], N0[2, 0], N0[0, 2], N0[2, 2] = M[0, 0], M[1, 0], M[0, 1], M[1, 1]
    N0[1, 1] = int(round(np.linalg.det(M)))  # b' = det b keeps the basis right-handed (det N0 = +1)
    return N0


def _niggli_N0(G):
    import spglib
    lat = np.linalg.cholesky(G)  # rows are lattice vectors with metric G (lat @ lat.T = G)
    red = spglib.niggli_reduce(lat, eps=1e-8)
    if red is None:
        raise RuntimeError('spglib.niggli_reduce failed')
    N0 = np.linalg.solve(lat.T, np.asarray(red).T)  # red.T = lat.T @ N0
    N0r = np.round(N0)
    if np.abs(N0 - N0r).max() > 1e-6 or abs(abs(np.linalg.det(N0r)) - 1) > 1e-9:
        raise RuntimeError('Niggli basis is not a unimodular change of the stored basis')
    return N0r.astype(np.int64)


def _small_corrections(system):
    if system == 'monoclinic':
        out = []
        for p, q, r, s in itertools.product(range(-3, 4), repeat=4):
            d = p * s - q * r
            if abs(d) == 1:
                out.append(np.array([[p, 0, r], [0, d, 0], [q, 0, s]], dtype=np.int64))  # b' = det b: det3 = +1
        return out
    if system == 'triclinic':
        out = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                P = np.zeros((3, 3), dtype=np.int64)
                for i, j in enumerate(perm):
                    P[j, i] = signs[i]
                if round(np.linalg.det(P)) == 1:
                    out.append(P)
        return out
    return [np.eye(3, dtype=np.int64)]


def _system(sg):
    sg = int(sg)
    if sg <= 2:
        return 'triclinic'
    if sg <= 15:
        return 'monoclinic'
    return 'higher'


def _penalty0(sg, L, A):
    return float(cell_reduction_penalty(torch.tensor(A[None], dtype=torch.float64),
                                        torch.tensor(L[None], dtype=torch.float64),
                                        torch.tensor([int(sg)]), margin=0.0)[0])


def _system_angles(sg, angles):
    """Monoclinic cells have alpha = gamma = 90 exactly; float32 storage leaves them 4e-8 rad off, which a strongly
    skewed basis amplifies into a spurious penalty."""
    angles = np.asarray(angles, dtype=np.float64).copy()
    if _system(sg) == 'monoclinic':
        angles[0] = angles[2] = np.pi / 2
    return angles


def choose_basis(sg, lengths, angles):
    """(N [3,3] int, new lengths, new angles, n_zero_candidates) for one crystal; N is None if no candidate cell has
    zero margin-0 penalty."""
    sg = int(sg)
    lengths = np.asarray(lengths, dtype=np.float64)
    angles = _system_angles(sg, angles)
    if _penalty0(sg, lengths, angles) <= PENALTY_ZERO:
        return np.eye(3, dtype=np.int64), lengths, angles, 1
    G = metric(lengths, angles)
    system = _system(sg)
    if system == 'monoclinic':
        N0 = _gauss_reduce_ac(G)
    elif system == 'triclinic':
        N0 = _niggli_N0(G)
    else:
        return None, lengths, angles, 0
    cands = [N0 @ M for M in _small_corrections(system)]
    cands = [N for N in cands if round(np.linalg.det(N)) == 1 and keeps_operators(N, sg)]
    if not cands:
        return None, lengths, angles, 0
    cells = [cell_from_metric(N.T @ G @ N) for N in cands]
    pen = cell_reduction_penalty(torch.tensor(np.stack([c[1] for c in cells]), dtype=torch.float64),
                                 torch.tensor(np.stack([c[0] for c in cells]), dtype=torch.float64),
                                 torch.full((len(cells),), sg), margin=0.0).numpy()  # one call for every candidate
    best, n_zero = None, 0
    for N, (Ln, An), p in zip(cands, cells, pen):
        if p > PENALTY_ZERO:
            continue
        n_zero += 1
        key = (int(np.abs(N - np.eye(3, dtype=np.int64)).sum()), tuple(N.flatten()))
        if best is None or key < best[0]:
            best = (key, N, Ln, An)
    if best is None:
        return None, lengths, angles, 0
    return best[1], best[2], best[3], n_zero


def apply_basis_change(batch, Ns, rows=None):
    """Re-describe crystals in the basis N (new basis vectors as columns, in old fractional coordinates; each N must
    keep SYM_OPS[sg], with the origin shift o = setting_shift(N, sg), and have det +1; each changed crystal's own
    operators must be SYM_OPS[sg] -- all checked). The crystal is unchanged: new cell from the metric N^T G N, centres
    x_new = N^-1 (x - o), poses rotated by the change of Cartesian frame Q = T_fc_new N^-1 T_cf_old, then every
    molecule's image back in its asymmetric-unit box (canonicalize_aunit). Rows not in `rows` (default: those with
    N != I) keep their parameters exactly. Returns the new batch (a clone)."""
    from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit
    out = batch.clone()
    n = out.num_graphs
    Ns = np.asarray(Ns, dtype=np.int64).reshape(n, 3, 3)
    eye = np.eye(3, dtype=np.int64)
    if rows is None:
        rows = ~(Ns == eye[None]).all(axis=(1, 2))
    rows = np.asarray(rows, dtype=bool)
    if not rows.any():
        return out
    sgs = out.sg_ind.reshape(-1).cpu().numpy()
    ops = out.symmetry_operators
    origins = np.zeros((n, 3))
    for i in np.nonzero(rows)[0]:
        if ops is not None and not standard_setting(ops[i], sgs[i]):
            raise ValueError(f'crystal {i}: its operators are not SYM_OPS[{sgs[i]}] as a set (a nonstandard setting); '
                             f'describe it in the standard setting first')
        o = setting_shift(Ns[i], sgs[i]) if round(np.linalg.det(Ns[i])) == 1 else None
        if o is None:
            raise ValueError(f'crystal {i}: basis change {Ns[i].tolist()} is not in the setting group of sg {sgs[i]} '
                             f'(or not right-handed)')
        origins[i] = o
    L = out.cell_lengths.double().cpu().numpy()
    A = out.cell_angles.double().cpu().numpy()
    newL, newA = L.copy(), A.copy()
    for i in np.nonzero(rows)[0]:
        newL[i], newA[i] = cell_from_metric(Ns[i].T @ metric(L[i], _system_angles(sgs[i], A[i])) @ Ns[i])

    idx = torch.as_tensor(np.nonzero(rows)[0])
    dev = out.cell_lengths.device
    f64 = torch.float64
    Ninv = torch.linalg.inv(torch.as_tensor(Ns[rows], dtype=f64, device=dev))
    t_fc_old, _, _ = batch_compute_fractional_transform(torch.as_tensor(L[rows], dtype=f64, device=dev),
                                                        torch.as_tensor(A[rows], dtype=f64, device=dev))
    t_fc_new, _, _ = batch_compute_fractional_transform(torch.as_tensor(newL[rows], dtype=f64, device=dev),
                                                        torch.as_tensor(newA[rows], dtype=f64, device=dev))
    Q = t_fc_new @ Ninv @ torch.linalg.inv(t_fc_old)  # old Cartesian frame -> new Cartesian frame (proper rotation)
    O = torch.as_tensor(origins[rows], dtype=f64, device=dev)  # a pure translation: poses do not see it

    centroid = out.aunit_centroid.clone()
    orientation = out.aunit_orientation.clone()
    handedness = out.aunit_handedness.clone()
    zp = out.z_prime.reshape(-1).cpu()
    for k in range(out.max_z_prime):
        sl = slice(3 * k, 3 * k + 3)
        present = zp[idx] > k  # padding slots of lower-Z' crystals may hold anything (NaN): never transform them
        if not present.any():
            continue
        sel = idx[present]
        p_dev = present.to(dev)
        f_new = torch.einsum('nij,nj->ni', Ninv[p_dev], centroid[sel, sl].to(f64) - O[p_dev])
        f_new = f_new - torch.floor(f_new)
        h_old = handedness[sel, k].to(f64)
        pose = rotvec2rotmat(orientation[sel, sl].to(f64))
        pose = pose * torch.stack([h_old, torch.ones_like(h_old), torch.ones_like(h_old)], 1)[:, None, :]
        rv, h = proper_rotvecs(Q[p_dev] @ pose)
        centroid[sel, sl] = f_new.to(centroid.dtype)
        orientation[sel, sl] = rv.to(orientation.dtype)
        handedness[sel, k] = h.to(handedness.dtype)

    keep = torch.as_tensor(~rows, device=dev)
    kept = (out.aunit_centroid[keep].clone(), out.aunit_orientation[keep].clone(), out.aunit_handedness[keep].clone())
    out.cell_lengths = torch.as_tensor(newL, dtype=out.cell_lengths.dtype, device=dev)
    out.cell_angles = torch.as_tensor(newA, dtype=out.cell_angles.dtype, device=dev)
    out.aunit_centroid = centroid
    out.aunit_orientation = orientation
    out.aunit_handedness = handedness
    out.box_analysis()
    canonicalize_aunit(out)  # every molecule's image into its asymmetric-unit box, canonical rotvecs
    _snap_edge_sliver(out, torch.as_tensor(rows, device=dev))
    # canonicalize_aunit re-chooses images for every row; the others keep their parameters exactly
    out.aunit_centroid[keep], out.aunit_orientation[keep], out.aunit_handedness[keep] = kept
    return out


def standardize_cells(batch, on_failure='raise'):
    """Return (standardised clone of the batch, info). info: N [n,3,3] int (identity where unchanged), origin [n,3]
    (the shift o of x_new = N^-1 (x - o); zero where unchanged), changed [n] bool, ok [n] bool, nonstandard [n] bool
    (operators not SYM_OPS[sg]: not handled, ok False), n_zero [n] (zero-penalty candidate cells found: 2 is the
    +-basis pair, which describe one cell). Crystals already reduced keep their parameters exactly. on_failure='raise'
    raises ValueError listing the crystals not handled; 'flag' leaves them unchanged with ok False."""
    n = batch.num_graphs
    L = batch.cell_lengths.double().cpu().numpy()
    A = batch.cell_angles.double().cpu().numpy()
    sgs = batch.sg_ind.reshape(-1).cpu().numpy()
    ops = batch.symmetry_operators
    Ns = np.tile(np.eye(3, dtype=np.int64), (n, 1, 1))
    origin = np.zeros((n, 3))
    ok = np.ones(n, dtype=bool)
    nonstandard = np.zeros(n, dtype=bool)
    n_zero = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if ops is not None and not standard_setting(ops[i], sgs[i]):
            nonstandard[i] = True
            ok[i] = False
            continue
        N, _, _, nz = choose_basis(sgs[i], L[i], A[i])
        n_zero[i] = nz
        if N is None:
            ok[i] = False
            continue
        Ns[i] = N
        origin[i] = setting_shift(N, sgs[i])
    if not ok.all() and on_failure == 'raise':
        msg = []
        for bad, why in ((np.nonzero(~ok & ~nonstandard)[0], 'no zero-penalty cell in the setting group'),
                         (np.nonzero(nonstandard)[0], 'operators not SYM_OPS[sg] (nonstandard setting)')):
            if len(bad):
                msg.append(f'{why} for {len(bad)} crystal(s): indices {bad[:20].tolist()}, '
                           f'space groups {sorted(set(sgs[bad].tolist()))}')
        raise ValueError('standardize_cells: ' + '; '.join(msg))
    changed = ok & ~(Ns == np.eye(3, dtype=np.int64)[None]).all(axis=(1, 2))
    return apply_basis_change(batch, Ns, rows=changed), dict(N=Ns, origin=origin, changed=changed, ok=ok,
                                                             nonstandard=nonstandard, n_zero=n_zero)


def _snap_edge_sliver(batch, rows):
    """The crystal builders clip a centre coordinate to [0, CELL_EDGE]; one in (CELL_EDGE, 1) is built up to 1e-4 of a
    cell length away from where it is (canonicalize_aunit can pick such an image). On an axis the asymmetric-unit box
    spans whole (box extent 1, so 1 is 0 by a lattice translation) the value is moved to whichever of 0 and CELL_EDGE is
    nearer, halving the worst case and making a centre that sits at a face essentially exact."""
    from mxtaltools.crystal_search.crystal_opt_utils import CELL_EDGE
    box = torch.stack([batch.asym_unit_dict[str(int(sg))] for sg in batch.sg_ind]).to(batch.aunit_centroid.device)
    cen = batch.aunit_centroid.clone()
    for k in range(batch.max_z_prime):
        sl = slice(3 * k, 3 * k + 3)
        c = cen[:, sl]
        snap = rows[:, None] & (box >= 1 - 1e-9) & (c > CELL_EDGE) & ((1 - c) < (c - CELL_EDGE))
        cen[:, sl] = torch.where(snap, torch.zeros_like(c), c)
    batch.aunit_centroid = cen


def unit_cell_fractional(batch):
    """[(fractional coordinates [m,3] float64, atomic numbers [m]) per crystal] of the unit cell mxtaltools builds."""
    b = batch.clone()
    b.mol2ucell()
    out = []
    for i in range(b.num_graphs):
        T = b.T_fc[i].double().cpu()
        pos = b.unit_cell_pos[b.unit_cell_batch == i].double().cpu()
        frac = torch.linalg.solve(T, pos.T).T.numpy()
        zmol = b.z[b.batch == i].cpu().numpy()
        out.append((frac, np.tile(zmol, len(frac) // len(zmol))))
    return out


def same_crystal_deviation(before, after, N, origin=None):
    """Max over atoms (both directions) of the distance, in Angstrom, from each atom of `before`, mapped into `after`'s
    basis by x_new = N^-1 (x - origin) (origin [n,3], default zero), to the nearest same-element atom image of `after`.
    [n] float64."""
    ua, ub = unit_cell_fractional(before), unit_cell_fractional(after)
    origin = np.zeros((len(ua), 3)) if origin is None else np.asarray(origin, dtype=np.float64).reshape(len(ua), 3)
    T_new = after.T_fc.double().cpu().numpy()
    dev = np.zeros(len(ua))
    for i, ((xa, za), (xb, zb)) in enumerate(zip(ua, ub)):
        if len(xa) != len(xb):
            dev[i] = np.inf
            continue
        y = (xa - origin[i]) @ np.linalg.inv(np.asarray(N[i], dtype=np.float64)).T
        worst = 0.0
        for el in np.unique(za):
            d = y[za == el][:, None, :] - xb[zb == el][None, :, :]
            d -= np.round(d)
            dist = np.linalg.norm(d @ T_new[i].T, axis=-1)
            worst = max(worst, dist.min(1).max(), dist.min(0).max())
        dev[i] = worst
    return dev

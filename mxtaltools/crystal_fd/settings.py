"""Nonstandard settings re-expressed in the standard setting ``SYM_OPS[sg]``, one way.

Owner decision D1 (2026-10-16): the CSD importer re-expresses a row deposited in a nonstandard setting (P2_1/n,
P2_1/a, I2/a, Pn, Pcab, another origin choice, a centred setting of P1, ...) in the standard setting; nothing writes a
nonstandard setting.

A setting change is a pair (P, p): new coordinates x_n of the stored fractional coordinates x_s are
``x_s = P x_n + p`` (the columns of P are the new basis vectors in the stored basis; p is the new origin in stored
coordinates), as ``ops.change_basis`` writes it with (N, s). A stored operator (W, w) becomes
``(P^-1 W P, P^-1 (W p + w - p))``. P is rational with det P > 0 (the new basis stays right-handed, so no crystal is
mirrored): integer with det 1 between two conventional cells of one volume (P2_1/n -> P2_1/c, I2/a -> C2/c, Pcab ->
Pbca), det 1/2 from a centred P1 setting (I1, B1) to a primitive cell, in which case the unit cell halves and Z with
it, while Z' stays 1; det 2 from a primitive setting of a centred group (none in the 2025-2026 CSD files).

What equality means. With G_s the stored group (stored operators plus Z^3 translations) and G_std the standard one
(``SYM_OPS[sg]`` plus Z^3), (P, p) is accepted when P^-1 G_s P = G_std as infinite groups. The conjugate is
T + P^-1 Z^3, T the transformed operators. ``setting_matches`` checks the equality finitely: (i) T combined with the
translations P^-1 Z^3, reduced modulo Z^3 with duplicates dropped (``transform_ops``), equals ``SYM_OPS[sg]`` as a set
(rotation parts equal, translations equal modulo 1, within ``tol``); (ii) every new basis vector P e_i is a
translation of G_s (it is congruent modulo Z^3 to a stored pure translation), i.e. Z^3 lies in the conjugate. Both
inclusions follow: an element tau + P^-1 n of the conjugate lies in SYM_OPS + Z^3 by (i); an element s of G_std is
congruent to some tau + P^-1 n by (i), so s = tau + P^-1 n + m with m in Z^3, which lies in the conjugate by (ii).
(iii), every P^-1 e_i a translation of G_std, follows from (i) and is checked as well. The conjugate of the group the
stored list generates is then G_std, so the stored list need not be closed.

How (P, p) is found (``find_setting``), a bounded finite search in up to four passes; every loop runs over a finite
set, so the search always ends:

1. The point-group orders must agree (operators per pure translation).
2. Candidate P, in order: P = K for integer K with det +1 and entries in {-1, 0, 1} (``_k_set``: identity first, then
   by the sum of absolute entries, the number of negative entries, descending entries), then with entries in
   [-2, 2]; this integer family is skipped when the stored and standard groups have different numbers of pure
   translations (a centred setting of a primitive group, or the reverse). Then P = B_s K B_std^-1 over the same K,
   with B_s, B_std short bases of the two translation lattices (Hermite form, then pairwise size reduction). Only P
   whose conjugation maps the stored rotation set onto the standard one are kept.
3. For each kept P, the origin: p = P q. Pass 1 takes q on the grid (1/24) Z^3 in [0, 1)^3 (every origin shift
   between ITA settings is a multiple of 1/8, 1/6 or 1/12 of the standard cell), zero first, then by the
   denominators used; the candidate q are filtered operator by operator: (P^-1 W P - I) q + P^-1 w must be congruent
   modulo 1 to a standard translation of that rotation. Pass 2, run over every kept P only when pass 1 found
   nothing, solves the same conditions exactly (``_origin_exact``): they are linear congruences in q modulo the
   standard translation lattice, solved by integer row reduction (``_solve_mod1``), so an origin off the grid (a
   rhombohedral-axes setting shifted on its own cell's grid maps off the hexagonal one) is found too.
4. Passes 3 and 4, only when passes 1 and 2 found nothing: the stored basis is first reduced under a metric its
   rotations preserve (``_invariant_basis``: sum of W^T W over the stored rotations, pairwise size reduction), which
   brings a cell given in an unreduced basis (a unimodular change with entries beyond [-2, 2]) back within the span
   of step 2; passes 1 and 2 then run on the operators in that basis (N0), and the result is composed,
   P = N0 P', p = N0 p'.
5. The first (P, p) that passes ``setting_matches`` on the stored operators is returned; None if none does (the
   importer's ``unresolved_setting``). Passes 2-4 run only where pass 1 fails, so they never change a result pass 1
   gives.

The result depends on the stored operators as a set only (not their order). It is not unique: any element of the
affine normaliser of G_std composed with it is another. For triclinic and monoclinic groups ``ops.reduce_cell`` and
``domains.fold`` remove the choice; for orthorhombic groups the choice fixes the re-expressed axis order, as the
depositor's choice does for rows already in the standard setting, and since 2026-10-21 ``canon.canonicalize``'s
axis order (``axis_order``) and fold remove it; in groups 75-230 the alternatives differ by Gamma, which the fold
absorbs.

``re_express`` applies (P, p) to a description: the cell from the metric P^T G P, ``t' = P^-1 (t - p)`` wrapped to
[0, 1), ``R' = Q R`` with ``Q = T_fc' P^-1 T_fc^-1`` (the frame rotation between the two cells' Cartesian
conventions, projected onto SO(3)), exactly as ``ops.change_basis`` does for integer N; handedness is unchanged.
"""
import itertools
from typing import NamedTuple, Optional

import numpy as np
import torch

from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.ops import _greedy_reduce, cell_from_metric_tensor, cell_matrices, metric_tensor

F64 = torch.float64

#: origin candidates are q = k / GRID_DENOMINATOR per coordinate, k = 0..23 (``_grid`` gives their order)
GRID_DENOMINATOR = 24
#: largest entry of the integer matrices K searched (``find_setting`` tries 1 first, then this)
MAX_SPAN = 2

_CACHE = {}
_K_SETS = {}
_GRID = None


class SettingTransform(NamedTuple):
    """A change of setting ``x_s = P x_n + p``: P [3, 3] and p [3], float64 numpy arrays (rational entries)."""
    P: np.ndarray
    p: np.ndarray


# ---------------------------------------------------------------------------
# operator sets
# ---------------------------------------------------------------------------

def _as_ops(ops) -> np.ndarray:
    o = ops.detach().cpu().double().numpy() if torch.is_tensor(ops) else np.asarray(ops, dtype=np.float64)
    return o.reshape(-1, 4, 4)


def std_ops(sg: int) -> np.ndarray:
    """[n, 4, 4] float64 ``SYM_OPS[sg]``."""
    return np.stack([np.asarray(o, dtype=np.float64) for o in SYM_OPS[int(sg)]])


def _split(ops: np.ndarray, tol: float):
    """(W [n, 3, 3] int64, w [n, 3] in [0, 1)) of affine 4x4 operators, or None if a rotation part is not integer or
    the last row is not (0, 0, 0, 1) within ``tol``."""
    W = np.round(ops[:, :3, :3])
    if np.abs(ops[:, :3, :3] - W).max() > tol or np.abs(ops[:, 3] - np.array([0, 0, 0, 1.0])).max() > tol:
        return None
    w = ops[:, :3, 3] - np.floor(ops[:, :3, 3])
    w = np.where(np.abs(w - 1) <= tol, 0.0, w)
    return W.astype(np.int64), w


def _mod1_close(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """|a - b| modulo 1 <= tol in every component (broadcast; last axis is the vector)."""
    d = a - b
    return (np.abs(d - np.round(d)) <= tol).all(-1)


def _dedupe(W: np.ndarray, w: np.ndarray, tol: float):
    """Operators unique modulo Z^3 (first occurrence kept)."""
    keep = []
    for i in range(len(W)):
        if not any((W[i] == W[j]).all() and _mod1_close(w[i], w[j], tol) for j in keep):
            keep.append(i)
    return W[keep], w[keep]


def _pure_translations(W: np.ndarray, w: np.ndarray, tol: float) -> np.ndarray:
    """[k, 3] translations (mod 1, unique) of the operators whose rotation part is the identity (0 included)."""
    eye = (W == np.eye(3, dtype=np.int64)).all((-1, -2))
    _, t = _dedupe(W[eye], w[eye], tol)
    if not len(t) or not any(_mod1_close(x, np.zeros(3), tol) for x in t):
        t = np.concatenate([np.zeros((1, 3)), t])
    return t


def _lattice_images(Pi: np.ndarray, tol: float, limit: int = 64) -> np.ndarray:
    """[k, 3] the translations P^-1 Z^3 modulo Z^3 (the stored lattice in the new coordinates; {0} when P^-1 is
    integer), by closure under addition; at most ``limit`` (raises beyond: no setting change of a real crystal has
    an index that large)."""
    gens = Pi.T                                                 # rows: P^-1 e_i
    out = [np.zeros(3)]
    frontier = [np.zeros(3)]
    while frontier:
        new = []
        for x in frontier:
            for g in gens:
                y = x + g
                y = y - np.floor(y)
                y = np.where(np.abs(y - 1) <= tol, 0.0, y)
                if not any(_mod1_close(y, z, tol) for z in out):
                    out.append(y)
                    new.append(y)
                    if len(out) > limit:
                        raise ValueError(f'P^-1 Z^3 has more than {limit} classes modulo Z^3')
        frontier = new
    return np.stack(out)


def transform_ops(ops, P, p, tol: float = 1e-6) -> np.ndarray:
    """[m, 4, 4] float64: the group of the operators ``ops`` [n, 4, 4] in the coordinates x_n of ``x_s = P x_n + p``,
    as operators modulo Z^3.

    (W, w) -> (P^-1 W P, P^-1 (W p + w - p)), each combined with every translation of the stored lattice in the
    new coordinates (P^-1 Z^3 modulo Z^3: only {0} unless the new cell is larger, as for a primitive setting of a
    centred group); translations reduced to [0, 1) and duplicates modulo Z^3 dropped (a centred setting re-expressed
    primitive has fewer distinct operators). Rotation parts are left in float (integers when P keeps the group's
    lattice).
    """
    o = _as_ops(ops)
    P = np.asarray(P, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    Pi = np.linalg.inv(P)
    W = o[:, :3, :3]
    w = o[:, :3, 3]
    Wn = Pi[None] @ W @ P[None]
    wn = np.einsum('ij,nj->ni', Pi, np.einsum('nij,j->ni', W, p) + w - p)
    L = _lattice_images(Pi, tol)
    Wn = np.repeat(Wn, len(L), 0)
    wn = (wn[:, None] + L[None]).reshape(-1, 3)
    wn = wn - np.floor(wn)
    wn = np.where(np.abs(wn - 1) <= tol, 0.0, wn)
    keep = []
    for i in range(len(Wn)):
        if not any(np.abs(Wn[i] - Wn[j]).max() <= tol and _mod1_close(wn[i], wn[j], tol) for j in keep):
            keep.append(i)
    out = np.zeros((len(keep), 4, 4))
    out[:, :3, :3] = Wn[keep]
    out[:, :3, 3] = wn[keep]
    out[:, 3, 3] = 1.0
    return out


def operator_sets_equal(a, b, tol: float = 1e-6) -> bool:
    """True when the operator lists ``a`` and ``b`` [n, 4, 4] are one set: rotation parts equal within ``tol`` and
    translations equal modulo 1 within ``tol``, matched as a bijection (duplicates modulo Z^3 not allowed)."""
    a, b = _as_ops(a), _as_ops(b)
    if len(a) != len(b):
        return False
    dW = np.abs(a[:, None, :3, :3] - b[None, :, :3, :3]).max((-1, -2)) <= tol
    m = dW & _mod1_close(a[:, None, :3, 3], b[None, :, :3, 3], tol)
    return bool((m.sum(0) == 1).all() and (m.sum(1) == 1).all())


def setting_matches(ops, sg: int, P, p, tol: float = 1e-6) -> bool:
    """True when ``x_s = P x_n + p`` re-expresses the group of the stored operators ``ops`` as ``SYM_OPS[sg]``.

    The conditions of the module docstring: the transformed group (``transform_ops``) equals ``SYM_OPS[sg]`` as a
    set modulo Z^3; every new basis vector P e_i is a translation of the stored group; every stored basis vector
    P^-1 e_i is a translation of the standard group. ``det P > 0`` is also required.
    """
    o = _as_ops(ops)
    P = np.asarray(P, dtype=np.float64)
    if not np.isfinite(o).all() or not np.isfinite(P).all() or not np.linalg.det(P) > 1e-9:
        return False
    s = _split(o, tol)
    std = std_ops(sg)
    ss = _split(std, tol)
    if s is None:
        return False
    if not operator_sets_equal(transform_ops(o, P, p, tol), std, tol):
        return False
    Ts = _pure_translations(*s, tol)
    Tstd = _pure_translations(*ss, tol)
    cols = P.T                                                  # rows: P e_i in stored coordinates
    if not _mod1_close(cols[:, None], Ts[None], tol).any(1).all():
        return False
    back = np.linalg.inv(P).T                                   # rows: P^-1 e_i in standard coordinates
    return bool(_mod1_close(back[:, None], Tstd[None], tol).any(1).all())


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def _k_set(span: int) -> np.ndarray:
    """[k, 3, 3] int64: every integer matrix with entries in [-span, span] and det +1, identity first, then by the sum
    of absolute entries, then by the number of negative entries, then by descending entries (row-major), so that
    P2_1/n -> P2_1/c takes c' = a + c rather than its negative and axis permutations come before their sign flips."""
    if span not in _K_SETS:
        vals = np.arange(-span, span + 1, dtype=np.int64)
        m = np.stack(np.meshgrid(*[vals] * 9, indexing='ij'), -1).reshape(-1, 3, 3)
        det = np.round(np.linalg.det(m.astype(np.float64))).astype(np.int64)
        m = m[det == 1]
        eye = np.eye(3, dtype=np.int64)
        m = m[~(m == eye).all((1, 2))]
        flat = m.reshape(len(m), 9)
        order = np.lexsort(tuple((-flat).T[::-1]) + ((flat < 0).sum(1), np.abs(flat).sum(1)))
        _K_SETS[span] = np.concatenate([eye[None], m[order]])
    return _K_SETS[span]


def _grid() -> np.ndarray:
    """[24^3, 3] origin candidates q = k / 24, zero first, then by the largest denominator rank, its sum, then
    lexicographically. Ranks: denominators 1, 2, 4, 8, 3, 6, 12, 24 in that order."""
    global _GRID
    if _GRID is None:
        D = GRID_DENOMINATOR
        pref = [1, 2, 4, 8, 3, 6, 12, 24]
        rank = np.array([next(i for i, d in enumerate(pref) if (k * d) % D == 0) for k in range(D)])
        k = np.array(list(itertools.product(range(D), repeat=3)), dtype=np.int64)
        r = rank[k]
        order = np.lexsort((k[:, 2], k[:, 1], k[:, 0], r.sum(1), r.max(1)))
        _GRID = k[order].astype(np.float64) / D
    return _GRID


def _lattice_basis(T: np.ndarray, tol: float) -> Optional[np.ndarray]:
    """[3, 3] (columns) a short basis, det > 0, of the lattice Z^3 + the translations ``T`` [k, 3]; None when a
    translation is not a fraction with denominator at most 48."""
    D = next((d for d in range(1, 49) if np.abs(T * d - np.round(T * d)).max() <= tol * d), None)
    if D is None:
        return None
    A = np.concatenate([np.eye(3, dtype=np.int64) * D, np.round(T * D).astype(np.int64)])
    r = 0
    for c in range(3):                                          # row Hermite form by Euclid on each column
        while True:
            nz = [i for i in range(r, len(A)) if A[i, c] != 0]
            piv = min(nz, key=lambda i: abs(A[i, c]))
            A[[r, piv]] = A[[piv, r]]
            clean = True
            for i in range(r + 1, len(A)):
                if A[i, c]:
                    A[i] -= (A[i, c] // A[r, c]) * A[r]
                    clean = clean and A[i, c] == 0
            if clean:
                break
        r += 1
    B = A[:3].T.astype(np.float64) / D
    if np.linalg.det(B) < 0:
        B[:, 0] = -B[:, 0]
    N, _ = _greedy_reduce(torch.as_tensor(B.T @ B, dtype=F64)[None])
    return B @ N[0].numpy().astype(np.float64)


def _rot_keys(W: np.ndarray) -> np.ndarray:
    """[...] int64 keys of integer matrices [..., 3, 3] with entries clipped to [-8, 8] (rotation parts lie in
    [-1, 1]; a clipped entry is never a rotation's)."""
    c = np.clip(W, -8, 8).astype(np.int64).reshape(*W.shape[:-2], 9) + 8
    return (c * (17 ** np.arange(9, dtype=np.int64))).sum(-1)


def _rotation_valid(P: np.ndarray, Ws: np.ndarray, Wstd: np.ndarray, tol: float, chunk: int = 4096) -> np.ndarray:
    """[k] bool: P [k, 3, 3] whose conjugation maps every stored rotation Ws [a, 3, 3] onto a standard one."""
    out = np.zeros(len(P), dtype=bool)
    Wf = Ws.astype(np.float64)
    std_keys = _rot_keys(Wstd)
    for lo in range(0, len(P), chunk):
        Pc = P[lo:lo + chunk]
        Pi = np.linalg.inv(Pc)
        Wn = Pi[:, None] @ Wf[None] @ Pc[:, None]               # [k, a, 3, 3]
        Wr = np.round(Wn)
        integer = (np.abs(Wn - Wr) <= tol).all((-1, -2)) & (np.abs(Wr) <= 1).all((-1, -2))
        hit = np.isin(_rot_keys(Wr), std_keys)
        out[lo:lo + chunk] = (integer & hit).all(1)
    return out


def _origin(P: np.ndarray, W: np.ndarray, w: np.ndarray, std_by_rot: dict, tol: float) -> Optional[np.ndarray]:
    """The first grid q (``_grid`` order) for which every stored operator maps into the standard group, or None."""
    Pi = np.linalg.inv(P)
    q = _grid()
    for Wi, wi in zip(W, w):
        Wn = np.round(Pi @ Wi @ P).astype(np.int64)
        A = std_by_rot.get(Wn.tobytes())
        if A is None:
            return None
        v = q @ (Wn - np.eye(3)).T + Pi @ wi                    # [m, 3]
        q = q[_mod1_close(v[:, None], A[None], tol).any(1)]
        if not len(q):
            return None
    return q[0]


def _solve_mod1(M: np.ndarray, b: np.ndarray, tol: float) -> Optional[np.ndarray]:
    """A y [k] with ``M y = b`` modulo 1 in every row (M [m, k] integer, b [m] real), or None when there is none.

    Integer row reduction to echelon form (Euclid on each column; the row operations are unimodular, so the solution
    set modulo 1 is unchanged), with b carried along modulo 1. Rows that end all zero must have b = 0 modulo 1 within
    ``tol``. Back substitution with the non-pivot unknowns at 0 then satisfies every pivot row exactly, hence every
    row modulo 1. Terminates: within a column each pass either clears the entries below the pivot or strictly lowers
    the smallest nonzero |entry| (the remainders are smaller than the pivot).
    """
    M = np.array(M, dtype=np.int64)
    b = np.asarray(b, dtype=np.float64) % 1.0
    m, k = M.shape
    r = 0
    piv = []
    for c in range(k):
        if r == m:
            break
        while True:
            nz = np.nonzero(M[r:, c])[0] + r
            if not len(nz):
                break
            i = nz[np.argmin(np.abs(M[nz, c]))]
            M[[r, i]] = M[[i, r]]
            b[[r, i]] = b[[i, r]]
            below = np.nonzero(M[r + 1:, c])[0] + r + 1
            if not len(below):
                break
            f = M[below, c] // M[r, c]
            M[below] = M[below] - f[:, None] * M[r][None]
            b[below] = (b[below] - f * b[r]) % 1.0
        if M[r, c] != 0:
            piv.append((r, c))
            r += 1
    rest = b[r:]
    if len(rest) and float(np.abs(rest - np.round(rest)).max()) > tol:
        return None
    y = np.zeros(k)
    for i, c in reversed(piv):
        y[c] = (b[i] - M[i, c + 1:] @ y[c + 1:]) / M[i, c]
    return y


def _origin_exact(P: np.ndarray, W: np.ndarray, w: np.ndarray, std_by_rot: dict, Bstd: np.ndarray,
                  tol: float) -> Optional[np.ndarray]:
    """The origin q (standard coordinates, in [0, 1)^3) solved exactly, or None.

    Every stored operator (W, w) must map into the standard group: ``(P^-1 W P - I) q + P^-1 w - a`` must lie in the
    standard translation lattice L_std (Z^3 plus the standard centring vectors, basis ``Bstd``), a being any one
    standard translation of the rotation P^-1 W P (the others differ from it by elements of L_std). In y = Bstd^-1 q
    these are linear congruences modulo Z^3 with integer coefficients ``Bstd^-1 (P^-1 W P - I) Bstd`` (P^-1 W P
    preserves L_std), solved by ``_solve_mod1``; one solution is returned (free coordinates of a polar group at 0).
    """
    Pi = np.linalg.inv(P)
    Bi = np.linalg.inv(Bstd)
    Ms, bs = [], []
    for Wi, wi in zip(W, w):
        Wn = np.round(Pi @ Wi @ P).astype(np.int64)
        A = std_by_rot.get(Wn.tobytes())
        if A is None:
            return None
        Mf = Bi @ (Wn - np.eye(3)) @ Bstd
        Mi = np.round(Mf)
        if np.abs(Mf - Mi).max() > 1e-6:
            return None
        Ms.append(Mi.astype(np.int64))
        bs.append(Bi @ (A[0] - Pi @ wi))
    y = _solve_mod1(np.concatenate(Ms), np.concatenate(bs), tol)
    if y is None:
        return None
    q = Bstd @ y
    q = q - np.floor(q)
    return np.where(np.abs(q - 1) <= tol, 0.0, q)


def _invariant_basis(W: np.ndarray) -> np.ndarray:
    """[3, 3] int64 N0 with det +1: the stored basis reduced (pairwise size reduction, ``ops._greedy_reduce``) under
    the metric ``sum_W W^T W`` over the stored rotation parts W [n, 3, 3], which every one of them preserves (they
    form a finite group), so that an axis of the group and the plane it acts on are orthogonal under it. Columns:
    the reduced basis vectors in stored coordinates."""
    rot = np.unique(np.asarray(W, dtype=np.int64).reshape(len(W), 9), axis=0).reshape(-1, 3, 3).astype(np.float64)
    G = np.einsum('kji,kjl->il', rot, rot)
    N, _ = _greedy_reduce(torch.as_tensor(G, dtype=F64)[None])
    N = N[0].numpy().astype(np.int64)
    if round(float(np.linalg.det(N.astype(np.float64)))) < 0:
        N[:, 0] = -N[:, 0]
    return N


def _key(ops: np.ndarray, sg: int, tol: float):
    s = _split(ops, tol)
    if s is None:
        return None
    W, w = s
    q = np.round(w * 1e6).astype(np.int64) % 1_000_000
    rows = np.concatenate([W.reshape(len(W), 9), q], 1)
    rows = rows[np.lexsort(rows.T[::-1])]
    return int(sg), float(tol), rows.tobytes()


def find_setting(ops, sg: int, tol: float = 1e-6) -> Optional[SettingTransform]:
    """(P, p) with ``x_s = P x_n + p`` re-expressing the stored operators ``ops`` [n, 4, 4] as ``SYM_OPS[sg]``, or None.

    The search of the module docstring; the result passes ``setting_matches`` and is cached per (sg, operator set).
    The standard setting itself gives (I, 0).
    """
    o = _as_ops(ops)
    sg = int(sg)
    if not np.isfinite(o).all() or not 1 <= sg <= 230:
        return None
    key = _key(o, sg, tol)
    if key is None:
        return None
    if key not in _CACHE:
        _CACHE[key] = _search(o, sg, tol)
    res = _CACHE[key]
    return None if res is None else SettingTransform(res.P.copy(), res.p.copy())


def _search(o: np.ndarray, sg: int, tol: float) -> Optional[SettingTransform]:
    """Passes 1-4 of the module docstring, in order; the first (P, p) that passes ``setting_matches`` on ``o``."""
    for origin in ('grid', 'exact'):
        res = _search_pass(o, sg, tol, origin)
        if res is not None:
            return res
    N0 = _invariant_basis(_split(o, tol)[0])
    if (N0 == np.eye(3, dtype=np.int64)).all():
        return None
    o0 = transform_ops(o, N0.astype(np.float64), np.zeros(3), tol)
    for origin in ('grid', 'exact'):
        res = _search_pass(o0, sg, tol, origin)
        if res is None:
            continue
        P = N0.astype(np.float64) @ res.P
        p = N0.astype(np.float64) @ res.p
        p = p - np.floor(p)
        p = np.where(np.abs(p - 1) <= tol, 0.0, p)
        if setting_matches(o, sg, P, p, tol):
            return SettingTransform(P, p)
    return None


def _search_pass(o: np.ndarray, sg: int, tol: float, origin: str) -> Optional[SettingTransform]:
    """One pass over the candidate P of step 2, with the origin from the grid (``origin='grid'``, ``_origin``) or
    solved exactly (``'exact'``, ``_origin_exact``)."""
    W, w = _split(o, tol)
    W, w = _dedupe(W, w, tol)
    std = std_ops(sg)
    Wstd, wstd = _split(std, tol)
    Wstd, wstd = _dedupe(Wstd, wstd, tol)
    Ts, Tstd = _pure_translations(W, w, tol), _pure_translations(Wstd, wstd, tol)
    if len(W) * len(Tstd) != len(Wstd) * len(Ts):               # point-group orders differ
        return None
    rot_s = np.unique(W.reshape(len(W), 9), axis=0).reshape(-1, 3, 3)
    rot_std = np.unique(Wstd.reshape(len(Wstd), 9), axis=0).reshape(-1, 3, 3)
    if len(rot_s) != len(rot_std):
        return None
    std_by_rot = {}
    for Wi, wi in zip(Wstd, wstd):
        k = Wi.tobytes()
        std_by_rot[k] = np.concatenate([std_by_rot[k], wi[None]]) if k in std_by_rot else wi[None]
    families = []
    if len(Ts) == len(Tstd):
        families.append(lambda K: K.astype(np.float64))
    Bs, Bstd = _lattice_basis(Ts, tol), _lattice_basis(Tstd, tol)
    if Bs is not None and Bstd is not None:
        Bstd_inv = np.linalg.inv(Bstd)
        families.append(lambda K: Bs[None] @ K.astype(np.float64) @ Bstd_inv[None])
    if origin == 'exact' and Bstd is None:
        return None
    for make in families:
        for span in range(1, MAX_SPAN + 1):
            K = _k_set(span)
            if span > 1:
                K = K[np.abs(K).max((1, 2)) > 1]                # the span-1 matrices were tried already
            P = make(K)
            ok = np.nonzero(_rotation_valid(P, rot_s, rot_std, tol))[0]
            for i in ok:
                if origin == 'grid':
                    q = _origin(P[i], W, w, std_by_rot, tol)
                else:
                    q = _origin_exact(P[i], W, w, std_by_rot, Bstd, tol)
                if q is None:
                    continue
                snap = np.round(P[i] * 48) / 48                 # B_s K B_std^-1 is rational; drop its rounding
                Pi = np.where(np.abs(P[i] - snap) <= 1e-9, snap, P[i])
                p = Pi @ q
                p = p - np.floor(p)
                if setting_matches(o, sg, Pi, p, tol):
                    return SettingTransform(Pi, p)
    return None


# ---------------------------------------------------------------------------
# descriptions
# ---------------------------------------------------------------------------

def re_express(desc: CrystalDescription, P, p, wrap: bool = True) -> CrystalDescription:
    """Re-describe each row in the coordinates of ``x_s = P x_n + p``; the crystal does not move.

    ``P`` [3, 3] or [B, 3, 3] real with det > 0 (rational in practice: integer, or det 1/2 for a centred P1 setting),
    ``p`` [3] or [B, 3]. The cell is read off the metric P^T G P, centroids become ``P^-1 (t - p)`` (wrapped to
    [0, 1) with ``wrap``), rotations ``Q R`` with ``Q = T_fc' P^-1 T_fc^-1`` projected onto SO(3); handedness is
    unchanged. For an integer P this is ``ops.change_basis(desc, P, p)`` (then the wrap). The caller supplies the
    space group of the new coordinates (the description keeps ``desc.sg``): use (P, p) from ``find_setting``.
    """
    d = desc.double()
    B = d.num_rows
    Pf = torch.as_tensor(np.asarray(P) if not torch.is_tensor(P) else P, dtype=F64, device=d.device).expand(B, 3, 3)
    pf = torch.as_tensor(np.asarray(p) if not torch.is_tensor(p) else p, dtype=F64, device=d.device).expand(B, 3)
    if B and not bool((torch.linalg.det(Pf) > 1e-9).all()):
        raise ValueError('every setting change must have det P > 0 (a left-handed basis would mirror the crystal)')
    cell = cell_from_metric_tensor(Pf.transpose(1, 2) @ metric_tensor(d.cell) @ Pf)
    T_old = d.cell_matrix()
    T_new = cell_matrices(cell)
    Pinv = torch.linalg.inv(Pf)
    Q = T_new @ Pinv @ torch.linalg.inv(T_old)                  # old Cartesian frame -> new; proper
    u, _, vh = torch.linalg.svd(Q)
    Q = u @ vh
    t = torch.einsum('bij,bzj->bzi', Pinv, d.t - pf[:, None])
    if wrap:
        t = t - torch.floor(t)
    R = Q[:, None] @ d.R
    m = d.unit_mask
    return d.replace(cell=cell, t=torch.where(m[..., None], t, d.t), R=torch.where(m[..., None, None], R, d.R))

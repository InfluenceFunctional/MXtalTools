"""Affine operations on crystal descriptions, basis changes, and triclinic cell reduction.

An affine element ``(W, w)`` acts on fractional coordinates as ``x -> W x + w``
and on poses by the Cartesian matrix ``Q = T_fc W T_fc^-1``: ``P -> Q P``. For
an element of the Euclidean normaliser of the cell's family ``Q`` is orthogonal;
it is projected onto O(3) (polar decomposition) to absorb rounding of the cell.
A det -1 element flips handedness.

A basis change ``N`` (integer, det +1, new basis vectors as the columns of N in
old fractional coordinates) with origin shift ``s`` re-describes the same
crystal: cell from the metric ``N^T G N``, ``t' = N^-1 (t - s)`` and
``P' = Q P`` with ``Q = T_fc' N^-1 T_fc^-1`` (the change of Cartesian frame), as
``crystal_search.standardize.apply_basis_change`` does on a batch.

Triclinic reduction (``reduce_cell``) picks, per lattice, the cell of the
tri_niggli convention (``sym_utils.tri_niggli_reduction_penalty`` at margin 0).
Ties within ``tol`` of a face are broken by the candidates' own cell parameters,
never by the input basis, so the reduced cell depends on the lattice alone. A
lattice with a proper automorphism (a special metric, e.g. beta = gamma = 90 deg
exactly) has several reduced bases with one cell; the pose in that cell then
depends on the input basis. ``CellReduction.n_auto`` counts them (1 generic).

Monoclinic reduction (``reduce_cell`` on sg 3-15, b-unique, cell choice 1, the
SYM_OPS setting) picks the cell at which ``sym_utils.mono_reduction_penalty`` at
margin 0 is zero, among the bases its setting class allows (``monoclinic_basis``).
The basis changes are a' = p a + q c, c' = r a + s c, b' = (ps - qr) b with
ps - qr = +-1 (so det N = +1), restricted by the class of ``sym_utils.MONO_CLASS``:
r even (GLIDE: 7, 13, 14), q even (CENTRED: 5, 8, 9, 12, 15), any (FREE: 3, 4, 6,
10, 11); each carries the origin shift ``standardize.setting_shift`` gives it. Every
monoclinic cell has two such bases, N and N times the 2-fold about b; that 2-fold
is in Gamma+ in every monoclinic group, so ``domains.fold`` absorbs the choice. The
faces of the monoclinic domain are mirrors: a cell on one has a further
automorphism within the setting class, flagged by ``n_auto`` > 1. Candidates within
``tol`` of the zero set tie; of them the least violating are kept (for a cell not
within rounding of a face, exactly the pair inside the domain, so the choice does
not depend on the input basis and the reduced cell lies in the closed domain), and
only then the tie-break key decides.

The special-metric flag ``n_auto`` is tested with its own tolerance ``face_tol``
(default ``FACE_TOL`` = 1e-5, dimensionless, on the tie-break key), much wider than
the search's ``tol`` = 1e-10: a lattice the search could resolve by candidate order
from some input basis is flagged from every basis, and so is a cell stored in float32
(relative rounding about 6e-8: a 90 deg angle is stored 4.4e-8 rad off) whose source
reported it on a face.
"""
import itertools
from typing import NamedTuple

import numpy as np
import torch

from mxtaltools.common.sym_utils import MONO_CENTRED, MONO_CLASS, MONO_FREE, MONO_GLIDE
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.so3 import split_o3
from mxtaltools.crystal_search.standardize import setting_shift

F64 = torch.float64

#: default tolerance of the special-metric flag ``CellReduction.n_auto`` (``reduce_cell(face_tol=...)``): two bases
#: whose tie-break keys (Gram entries / mean squared length) agree within it count as automorphic. A choice, with a
#: margin of about 10 over float32 storage of a cell its source reported on a face or special metric (measured
#: 2026-10-16 on 2000 cells per face of sg 14, 15 and 4: the face condition is up to 4.5e-7 off after rounding, and
#: every row is flagged from face_tol 1e-6; a float32 hexagonal triclinic metric is flagged at 1e-6, not at 3e-7),
#: and 1e5 times the search's ``tol``, so any row the search could leave to candidate order is flagged from every
#: input basis. A flagged row further than ``tol`` from a special metric (monoclinic: further than rounding from a
#: face) still has one canonical form from every basis.
FACE_TOL = 1e-5


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def cell_matrices(cell: torch.Tensor) -> torch.Tensor:
    """[B, 6] cells -> [B, 3, 3] float64 T_fc (MXtalTools frame: a along x, b in the xy plane)."""
    from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
    c = torch.as_tensor(cell).to(F64)
    return batch_compute_fractional_transform(c[:, :3], c[:, 3:])[0]


def metric_tensor(cell: torch.Tensor) -> torch.Tensor:
    """[B, 6] -> [B, 3, 3] Gram matrices G_ij = a_i . a_j, float64."""
    c = torch.as_tensor(cell).to(F64)
    a, b, cc = c[:, 0], c[:, 1], c[:, 2]
    ca, cb, cg = torch.cos(c[:, 3]), torch.cos(c[:, 4]), torch.cos(c[:, 5])
    return torch.stack([a * a, a * b * cg, a * cc * cb,
                        a * b * cg, b * b, b * cc * ca,
                        a * cc * cb, b * cc * ca, cc * cc], 1).reshape(-1, 3, 3)


def cell_from_metric_tensor(G: torch.Tensor) -> torch.Tensor:
    """[B, 3, 3] Gram matrices -> [B, 6] cells (cosines clipped to [-1, 1])."""
    G = G.to(F64)
    L = torch.sqrt(torch.diagonal(G, dim1=1, dim2=2))
    al = torch.arccos((G[:, 1, 2] / (L[:, 1] * L[:, 2])).clamp(-1, 1))
    be = torch.arccos((G[:, 0, 2] / (L[:, 0] * L[:, 2])).clamp(-1, 1))
    ga = torch.arccos((G[:, 0, 1] / (L[:, 0] * L[:, 1])).clamp(-1, 1))
    return torch.cat([L, torch.stack([al, be, ga], 1)], 1)


# ---------------------------------------------------------------------------
# affine elements
# ---------------------------------------------------------------------------

def compose(a, b):
    """``a o b`` for affine elements (W, w), batched by broadcasting: x -> Wa (Wb x + wb) + wa."""
    Wa, wa = (torch.as_tensor(v, dtype=F64) for v in a)
    Wb, wb = (torch.as_tensor(v, dtype=F64) for v in b)
    return Wa @ Wb, (Wa @ wb[..., None])[..., 0] + wa


def inverse(op):
    """Inverse of an affine element (W, w): (W^-1, -W^-1 w)."""
    W, w = (torch.as_tensor(v, dtype=F64) for v in op)
    Wi = torch.linalg.inv(W)
    return Wi, -(Wi @ w[..., None])[..., 0]


def cartesian_rotation(T: torch.Tensor, W: torch.Tensor, project: bool = True) -> torch.Tensor:
    """Q = T W T^-1 for T [B, 3, 3] and W [B, 3, 3] or [3, 3]; projected onto O(3) with ``project``."""
    T = T.to(F64)
    W = torch.as_tensor(W, dtype=F64, device=T.device)
    Q = T @ W @ torch.linalg.inv(T)
    if project:
        u, _, vh = torch.linalg.svd(Q)
        Q = u @ vh
    return Q


def apply(desc: CrystalDescription, W, w, wrap: bool = False) -> CrystalDescription:
    """Apply the affine element (W, w) to every asymmetric unit of every row.

    ``W`` [3, 3] or [B, 3, 3], ``w`` [3] or [B, 3] (fractional). Centroids become
    ``W t + w`` (wrapped to [0, 1) with ``wrap``), poses ``Q P``; the result's
    (R, h) is the O(3) split of Q P, so a det -1 element flips h. The cell is
    unchanged: the element must be an isometry of the row's metric (an element
    of the Euclidean normaliser of its family). Result is float64.
    """
    d = desc.double()
    B, Z = d.t.shape[:2]
    W = torch.as_tensor(W, dtype=F64, device=d.device).expand(B, 3, 3)
    w = torch.as_tensor(w, dtype=F64, device=d.device).expand(B, 3)
    Q = cartesian_rotation(d.cell_matrix(), W)
    t = torch.einsum('bij,bzj->bzi', W, d.t) + w[:, None]
    if wrap:
        t = t - torch.floor(t)
    P = Q[:, None] @ d.pose
    R, h = split_o3(P.reshape(-1, 3, 3))
    m = d.unit_mask
    t = torch.where(m[..., None], t, d.t)
    R = torch.where(m[..., None, None], R.reshape(B, Z, 3, 3), d.R)
    h = torch.where(m, h.reshape(B, Z), d.h)
    return d.replace(t=t, R=R, h=h)


# ---------------------------------------------------------------------------
# basis changes
# ---------------------------------------------------------------------------

def change_basis(desc: CrystalDescription, N, s=None) -> CrystalDescription:
    """Re-describe each row in the basis ``N`` with origin shift ``s``; the crystal does not move.

    Parameters
    ----------
    desc : CrystalDescription
    N : array-like
        [3, 3] or [B, 3, 3] integer basis changes, det +1, each keeping
        ``SYM_OPS[sg]`` (with ``s``).
    s : array-like, optional
        [3] or [B, 3] origin shifts (old fractional coordinates); default
        ``standardize.setting_shift(N, sg)`` per row.

    Raises
    ------
    ValueError
        If a row's N is not integer with det +1 or is outside its setting group.
    """
    d = desc.double()
    B, Z = d.t.shape[:2]
    Nf = torch.as_tensor(N, dtype=F64, device=d.device).expand(B, 3, 3)
    Ni = torch.round(Nf).to(torch.long)
    if (Nf - Ni).abs().amax() > 1e-9 or bool((torch.round(torch.linalg.det(Nf)) != 1).any()):
        raise ValueError('every basis change must be an integer matrix with det +1')
    if s is None:
        shifts = []
        for i in range(B):
            o = setting_shift(Ni[i].cpu().numpy(), int(d.sg[i]))
            if o is None:
                raise ValueError(f'row {i}: basis change {Ni[i].tolist()} is not in the setting group of '
                                 f'sg {int(d.sg[i])}')
            shifts.append(o)
        s = torch.as_tensor(np.stack(shifts), dtype=F64, device=d.device)
    s = torch.as_tensor(s, dtype=F64, device=d.device).expand(B, 3)
    cell = cell_from_metric_tensor(Nf.transpose(1, 2) @ metric_tensor(d.cell) @ Nf)
    T_old = d.cell_matrix()
    T_new = cell_matrices(cell)
    Ninv = torch.linalg.inv(Nf)
    Q = T_new @ Ninv @ torch.linalg.inv(T_old)  # old Cartesian frame -> new; proper
    u, _, vh = torch.linalg.svd(Q)
    Q = u @ vh
    t = torch.einsum('bij,bzj->bzi', Ninv, d.t - s[:, None])
    R = Q[:, None] @ d.R
    m = d.unit_mask
    return d.replace(cell=cell,
                     t=torch.where(m[..., None], t, d.t),
                     R=torch.where(m[..., None, None], R, d.R))


# ---------------------------------------------------------------------------
# triclinic reduction
# ---------------------------------------------------------------------------

def _neighbourhood() -> torch.Tensor:
    """Every 3x3 matrix with entries in {-1, 0, 1} and det +1 (3480), identity first."""
    allm = np.array(list(itertools.product((-1, 0, 1), repeat=9)), dtype=np.int64).reshape(-1, 3, 3)
    det = np.round(np.linalg.det(allm.astype(float))).astype(np.int64)
    m = allm[det == 1]
    eye = np.eye(3, dtype=np.int64)
    m = np.concatenate([eye[None], m[~(m == eye).all(axis=(1, 2))]])
    return torch.as_tensor(m)


_NEIGHBOURHOOD = None


def tri_niggli_conditions(G: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] Gram matrices -> [..., 8] dimensionless conditions, all >= 0 exactly on the tri_niggli zero set.

    With A, B, C the squared lengths and xi = 2 b.c, eta = 2 a.c, zeta = 2 a.b:
    1 - a/b, 1 - b/c, 1 - |xi|/B, 1 - |eta|/A, 1 - |zeta|/A, -eta/A, -zeta/A and
    (A + B + xi + eta + zeta) / (A + B): the walls of
    ``sym_utils.tri_niggli_reduction_penalty`` at margin 0, one per wall.
    """
    A, Bq, C = G[..., 0, 0], G[..., 1, 1], G[..., 2, 2]
    xi, eta, zeta = 2 * G[..., 1, 2], 2 * G[..., 0, 2], 2 * G[..., 0, 1]
    a, b, c = torch.sqrt(A), torch.sqrt(Bq), torch.sqrt(C)
    return torch.stack([1 - a / b, 1 - b / c, 1 - xi.abs() / Bq, 1 - eta.abs() / A, 1 - zeta.abs() / A,
                        -eta / A, -zeta / A, (A + Bq + xi + eta + zeta) / (A + Bq)], -1)


def _greedy_reduce(G: torch.Tensor, max_iter: int = 1000):
    """Pairwise size reduction b_i -= round(b_i.b_j / b_j.b_j) b_j while it shortens b_i (relative 1e-12).

    Returns (N [B, 3, 3] long, G' = N^T G N). Terminates: every accepted step
    shortens a basis vector by a relative 1e-12, and lattice vectors have a
    positive minimum; ``max_iter`` passes bound it and raise.
    """
    G = G.clone()
    B = G.shape[0]
    N = torch.eye(3, dtype=torch.long, device=G.device).repeat(B, 1, 1)
    pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    for _ in range(max_iter):
        changed = torch.zeros(B, dtype=torch.bool, device=G.device)
        for i, j in pairs:
            m = torch.round(G[:, i, j] / G[:, j, j])
            new = G[:, i, i] - 2 * m * G[:, i, j] + m * m * G[:, j, j]
            ok = (m != 0) & (new < G[:, i, i] * (1 - 1e-12))
            if not bool(ok.any()):
                continue
            E = torch.eye(3, dtype=F64, device=G.device).repeat(B, 1, 1)
            E[:, j, i] = torch.where(ok, -m, torch.zeros_like(m))
            G = E.transpose(1, 2) @ G @ E
            N = _imatmul(N, E.round().to(torch.long))
            changed |= ok
        if not bool(changed.any()):
            return N, G
    raise RuntimeError(f'pairwise reduction did not converge in {max_iter} passes')


def _lex_select(keys: torch.Tensor, valid: torch.Tensor, tol: float, largest: bool) -> torch.Tensor:
    """Per row, the index of the lexicographic min (or max) of ``keys`` [B, k, d] among ``valid`` [B, k].

    Component by component, every candidate within ``tol`` of the extreme stays
    in; the first survivor in candidate order is returned.
    """
    keep = valid.clone()
    sign = -1.0 if largest else 1.0
    for c in range(keys.shape[-1]):
        v = sign * keys[..., c]
        best = torch.where(keep, v, torch.full_like(v, float('inf'))).amin(1, keepdim=True)
        keep &= v <= best + tol
    return keep.to(torch.int8).argmax(1)


class CellReduction(NamedTuple):
    """Result of ``reduce_cell``: the re-described rows, the basis change and origin shift applied, and diagnostics.

    ``n_auto`` > 1 marks a special metric: the reduced cell is unique, but the
    pose in it depends on which of the ``n_auto`` automorphic bases was taken, so
    two settings of one crystal can give two canonical descriptions. Automorphisms
    are counted with ``reduce_cell``'s ``face_tol`` (default ``FACE_TOL`` = 1e-5 on the
    tie-break key), so a row within about ``face_tol`` of a special metric is
    flagged, from every input basis; a flagged row further than ``tol`` from the
    special metric (monoclinic: further than rounding from a face) still has one
    canonical form from every basis. Monoclinic rows
    count automorphisms within the setting class modulo {identity, 2-fold about b}
    (the 2-fold is in Gamma+, which the fold absorbs), and candidates likewise in
    pairs, so 1 is generic in both fields; a monoclinic cell within about
    ``face_tol`` of any face of the domain has ``n_auto`` >= 2.
    """
    desc: CrystalDescription
    N: torch.Tensor           #: [B, 3, 3] long, input basis -> reduced basis
    s: torch.Tensor           #: [B, 3] origin shift
    n_tied: torch.Tensor      #: [B] candidates within tol of the zero set (1 off faces)
    violation: torch.Tensor   #: [B] largest condition violation of the chosen cell (<= tol)
    n_auto: torch.Tensor      #: [B] proper automorphisms of the reduced metric within face_tol (1 generic)


def _gram_keys(G: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] -> [..., 6] (A, B, C, xi, eta, zeta) / mean(A, B, C): the triclinic tie-break key."""
    scale = torch.diagonal(G, dim1=-2, dim2=-1).mean(-1, keepdim=True)
    return torch.stack([G[..., 0, 0], G[..., 1, 1], G[..., 2, 2],
                        2 * G[..., 1, 2], 2 * G[..., 0, 2], 2 * G[..., 0, 1]], -1) / scale


def lattice_automorphisms(cell: torch.Tensor, tol: float = 1e-10, chunk: int = 128) -> torch.Tensor:
    """[B] long: the proper basis changes M (entries in {-1, 0, 1}, det +1) with M^T G M = G, the identity included.

    Equality is on the tie-break key (``_gram_keys`` within ``tol``), so a count above 1
    means bases the triclinic tie-break cannot tell apart at that tolerance;
    ``reduce_cell`` passes its ``face_tol``. Exact for a reduced cell, whose
    automorphisms lie in that neighbourhood (1 for a generic lattice; 2 at
    beta = gamma = 90 deg; 24 for a cubic metric).
    """
    global _NEIGHBOURHOOD
    if _NEIGHBOURHOOD is None:
        _NEIGHBOURHOOD = _neighbourhood()
    G = metric_tensor(cell)
    Mf = _NEIGHBOURHOOD.to(device=G.device, dtype=F64)
    out = []
    for lo in range(0, len(G), chunk):
        g = G[lo:lo + chunk]
        Gc = torch.einsum('kji,bjl,klm->bkim', Mf, g, Mf)
        same = ((_gram_keys(Gc) - _gram_keys(g)[:, None]).abs() <= tol).all(-1)
        out.append(same.sum(1))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=G.device)


def real_cells(cell: torch.Tensor, rel_det: float = 1e-12) -> torch.Tensor:
    """[B] bool: [B, 6] cells with finite positive lengths, finite angles and Gram det > ``rel_det`` (abc)^2."""
    c = torch.as_tensor(cell).to(F64)
    ok = torch.isfinite(c).all(1) & (c[:, :3] > 0).all(1)
    cs = torch.cos(torch.where(ok[:, None], c[:, 3:], torch.ones_like(c[:, 3:])))
    return ok & (1 - (cs ** 2).sum(1) + 2 * cs.prod(1) > rel_det)          # det G / (abc)^2


def check_cells(cell: torch.Tensor, rel_det: float = 1e-12) -> torch.Tensor:
    """Raise ValueError naming the rows of [B, 6] cells that are not ``real_cells``; return the cells as float64."""
    c = torch.as_tensor(cell).to(F64)
    ok = real_cells(c, rel_det)
    if not bool(ok.all()):
        bad = torch.nonzero(~ok)[:, 0].tolist()
        raise ValueError(f'{len(bad)} rows have no real cell (non-finite, non-positive length or Gram determinant '
                         f'<= {rel_det} (abc)^2): rows {bad[:10]}{" ..." if len(bad) > 10 else ""}')
    return c


def triclinic_basis(cell: torch.Tensor, tol: float = 1e-10, chunk: int = 256):
    """The tri_niggli reduced basis of each lattice, chosen independently of the input basis.

    Candidates are ``N0 M`` for the pairwise-reduced basis N0 and every M with
    entries in {-1, 0, 1} and det +1. Those whose largest condition violation
    (``tri_niggli_conditions``) is at most ``tol`` are tied; among them the
    lexicographically smallest (A, B, C, xi, eta, zeta) / mean(A, B, C) is
    taken, each component compared within ``tol``. Off faces exactly one
    candidate is valid. A lattice with no valid candidate gets the least
    violating one, and its violation is returned.

    Returns
    -------
    (N [B, 3, 3] long, cell [B, 6], n_tied [B] long, violation [B])
    """
    global _NEIGHBOURHOOD
    if _NEIGHBOURHOOD is None:
        _NEIGHBOURHOOD = _neighbourhood()
    G = metric_tensor(cell)
    if len(G) == 0:
        return (torch.zeros(0, 3, 3, dtype=torch.long, device=G.device), torch.zeros(0, 6, dtype=F64, device=G.device),
                torch.zeros(0, dtype=torch.long, device=G.device), torch.zeros(0, dtype=F64, device=G.device))
    N0, G0 = _greedy_reduce(G)
    M = _NEIGHBOURHOOD.to(G.device)
    Mf = M.to(F64)
    out_N, out_tied, out_v = [], [], []
    for lo in range(0, len(G0), chunk):
        g = G0[lo:lo + chunk]
        Gc = torch.einsum('kji,bjl,klm->bkim', Mf, g, Mf)                  # [b, k, 3, 3]
        viol = (-tri_niggli_conditions(Gc)).clamp(min=0).amax(-1)         # [b, k]
        vmin = viol.amin(1, keepdim=True)
        valid = viol <= torch.maximum(vmin, torch.full_like(vmin, tol))
        keys = _gram_keys(Gc)
        pick = _lex_select(keys, valid, tol, largest=False)
        out_N.append(M[pick])
        out_tied.append(valid.sum(1))
        out_v.append(viol.gather(1, pick[:, None])[:, 0])
    Nsel = _imatmul(N0, torch.cat(out_N))
    cell_new = cell_from_metric_tensor(Nsel.to(F64).transpose(1, 2) @ G @ Nsel.to(F64))
    return Nsel, cell_new, torch.cat(out_tied), torch.cat(out_v)


def reduce_cell(desc: CrystalDescription, tol: float = 1e-10, angle_tol: float = 1e-6,
                face_tol: float = FACE_TOL) -> CellReduction:
    """Re-describe triclinic (sg 1, 2) and monoclinic (sg 3-15) rows in their reduced cell.

    Triclinic rows: ``triclinic_basis``. Monoclinic rows: ``monoclinic_basis``, with
    the origin shift ``standardize.setting_shift`` gives the chosen basis change; alpha
    and gamma are first set to exactly pi/2 (rows more than ``angle_tol`` rad off are
    refused), and the reduced cell has them at exactly pi/2. Rows of the two systems
    may be mixed; the triclinic rows are computed exactly as on their own. An empty
    batch gives an empty result.

    Parameters
    ----------
    tol : float
        The search's tie tolerance on the dimensionless face conditions and keys.
    angle_tol : float
        Rad; monoclinic rows whose alpha or gamma is further than this from pi/2 are refused.
    face_tol : float
        Tolerance of the special-metric flag ``n_auto`` (``FACE_TOL``); at least ``tol``.

    Raises
    ------
    NotImplementedError
        For any other space group.
    ValueError
        For rows with no real cell (``check_cells``), monoclinic rows whose alpha or
        gamma is more than ``angle_tol`` from pi/2, or ``face_tol`` < ``tol``.
    RuntimeError
        If the search leaves a row more than ``tol`` off the reduced set, or a chosen
        monoclinic basis change is outside its setting group (never expected).
    """
    if not face_tol >= tol:
        raise ValueError(f'face_tol ({face_tol}) must be at least tol ({tol}): the flag must cover every tie')
    sg = desc.sg
    tri = (sg == 1) | (sg == 2)
    mono = (sg >= 3) & (sg <= 15)
    if not bool((tri | mono).all()):
        raise NotImplementedError('reduce_cell handles triclinic (1, 2) and monoclinic (3-15) space groups only')
    if desc.num_rows == 0:
        d = desc.double()
        dev = d.device
        return CellReduction(d, torch.zeros(0, 3, 3, dtype=torch.long, device=dev), torch.zeros(0, 3, dtype=F64, device=dev),
                             torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=F64, device=dev),
                             torch.zeros(0, dtype=torch.long, device=dev))
    check_cells(desc.cell)
    off = mono & ((desc.cell[:, [3, 5]].to(F64) - torch.pi / 2).abs() > angle_tol).any(1)
    if bool(off.any()):
        bad = torch.nonzero(off)[:, 0].tolist()
        raise ValueError(f'{len(bad)} monoclinic rows have alpha or gamma more than {angle_tol} rad from 90 deg: '
                         f'rows {bad[:10]}{" ..." if len(bad) > 10 else ""}')
    if bool(tri.all()):
        return _reduce_triclinic(desc, tol, face_tol)
    if bool(mono.all()):
        return _reduce_monoclinic(desc, tol, face_tol)
    parts, rows = [], []
    for m, fn in ((tri, lambda d: _reduce_triclinic(d, tol, face_tol)),
                  (mono, lambda d: _reduce_monoclinic(d, tol, face_tol))):
        r = torch.nonzero(m)[:, 0]
        parts.append(fn(desc[r]))
        rows.append(r)
    order = torch.argsort(torch.cat(rows))
    return CellReduction(CrystalDescription.cat([p.desc for p in parts])[order],
                         *(torch.cat([getattr(p, f) for p in parts])[order] for f in CellReduction._fields[1:]))


def _reduce_triclinic(desc: CrystalDescription, tol: float, face_tol: float) -> CellReduction:
    N, cell, n_tied, viol = triclinic_basis(desc.cell, tol=tol)
    if bool((viol > tol).any()):
        bad = torch.nonzero(viol > tol)[:, 0].tolist()
        raise RuntimeError(f'reduce_cell: rows {bad[:10]} found no basis within {tol} of the reduced set '
                           f'(largest violation {float(viol.max()):.3g})')
    s = torch.zeros(desc.num_rows, 3, dtype=F64, device=desc.device)  # every det +1 change keeps P1 and P-1
    return CellReduction(change_basis(desc, N, s), N, s, n_tied, viol, lattice_automorphisms(cell, face_tol))


# ---------------------------------------------------------------------------
# monoclinic reduction
# ---------------------------------------------------------------------------
#
# A 2x2 integer matrix M (columns: the new a and c in old (a, c) coordinates) is the ac-plane part of the 3x3
# basis change N = [[p, 0, r], [0, d, 0], [q, 0, s]] with M = [[p, r], [q, s]] and d = det M = ps - qr = +-1, so
# a' = p a + q c, c' = r a + s c, b' = d b and det N = d^2 = +1.

#: correction entries searched around the Gauss-reduced ac basis (``monoclinic_basis``). The chosen correction had
#: entries within 1 in every case checked (tests/fd/test_monoclinic_cell.py: span 1 and span 3 choose the same
#: cells); 3 leaves a margin
MONO_SPAN = 3


def _imatmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Product of small integer matrices (long), through float64 (exact here; CUDA has no integer matmul)."""
    return torch.round(a.to(F64) @ b.to(F64)).to(torch.long)


def _square_neighbourhood(span: int) -> torch.Tensor:
    """[k, 2, 2] long: every integer matrix with entries in [-span, span] and det +-1, identity first."""
    m = np.array(list(itertools.product(range(-span, span + 1), repeat=4)), dtype=np.int64).reshape(-1, 2, 2)
    m = m[np.abs(m[:, 0, 0] * m[:, 1, 1] - m[:, 0, 1] * m[:, 1, 0]) == 1]
    eye = np.eye(2, dtype=np.int64)
    return torch.as_tensor(np.concatenate([eye[None], m[~(m == eye).all(axis=(1, 2))]]))


_MONO_CANDIDATES = None
_MONO_AUTOS = None


def _mono_sets(device):
    """(candidate corrections with entries in [-MONO_SPAN, MONO_SPAN], automorphism search set with entries in
    {-1, 0, 1}), both [k, 2, 2] long on ``device``."""
    global _MONO_CANDIDATES, _MONO_AUTOS
    if _MONO_CANDIDATES is None:
        _MONO_CANDIDATES = _square_neighbourhood(MONO_SPAN)
        _MONO_AUTOS = _square_neighbourhood(1)
    return _MONO_CANDIDATES.to(device), _MONO_AUTOS.to(device)


def mono_class(sg) -> torch.Tensor:
    """[B] long setting class (``sym_utils.MONO_GLIDE`` / ``MONO_CENTRED`` / ``MONO_FREE``) of space groups ``sg``
    [B]; -1 outside 3-15."""
    sg = torch.as_tensor(sg).long()
    return MONO_CLASS.to(sg.device)[sg]


def mono_parity(M: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    """[...] bool: the ac-plane changes M [..., 2, 2] (long) that the setting class ``cls`` (broadcastable) allows.

    GLIDE: r = M[0, 1] even (c' = r a + s c keeps the c-glide a c-glide); CENTRED:
    q = M[1, 0] even (a' = p a + q c keeps the C-centring); FREE: any. These are
    exactly the changes ``standardize.setting_shift`` accepts (with some origin
    shift), checked over every change with entries in [-3, 3] for each of sg 3-15 by
    tests/fd/test_monoclinic_cell.py.
    """
    r_even = torch.remainder(M[..., 0, 1], 2) == 0
    q_even = torch.remainder(M[..., 1, 0], 2) == 0
    return torch.where(cls == MONO_GLIDE, r_even, torch.where(cls == MONO_CENTRED, q_even, torch.ones_like(r_even)))


def mono_bound(a: torch.Tensor, c: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
    """The class's bound m on -cos(beta) (margin 0, ``sym_utils.mono_reduction_penalty``): GLIDE min(a/c, c/2a),
    CENTRED min(c/a, a/2c), FREE a/2c. At most 1/sqrt(2) for GLIDE and CENTRED (the two terms multiply to 1/2) and
    1/2 for FREE on its domain (a <= c), so beta <= 135 deg; the penalty's clamp of -m to [-1, 0] never binds there."""
    return torch.where(cls == MONO_GLIDE, torch.minimum(a / c, c / (2 * a)),
                       torch.where(cls == MONO_CENTRED, torch.minimum(c / a, a / (2 * c)), a / (2 * c)))


def _mono_conditions(A, C, Gac, cls):
    """[..., 4] (W1, W2, W3, order) from the ac Gram entries; inactive conditions are 1."""
    a, c = torch.sqrt(A), torch.sqrt(C)
    cb = Gac / (a * c)
    one = torch.ones_like(cb)
    glide, centred, free = cls == MONO_GLIDE, cls == MONO_CENTRED, cls == MONO_FREE
    w2 = torch.where(glide, 1 + c / a * cb, torch.where(centred, 1 + a / c * cb, one))
    w3 = torch.where(glide, 1 + 2 * a / c * cb, 1 + 2 * c / a * cb)
    order = torch.where(free, 1 - a / c, one)
    return torch.stack([-cb, w2, w3, order], -1)


def mono_conditions(G: torch.Tensor, sg) -> torch.Tensor:
    """[..., 3, 3] Gram matrices (alpha = gamma = 90 deg), sg broadcastable -> [..., 4] dimensionless conditions
    (W1, W2, W3, order), all >= 0 exactly on the margin-0 zero set of ``sym_utils.mono_reduction_penalty``.

    With cos(beta) = G_ac / (a c): W1 = -cos(beta); GLIDE W2 = 1 + (c/a) cos(beta),
    W3 = 1 + (2a/c) cos(beta); CENTRED W2 = 1 + (a/c) cos(beta), W3 = 1 + (2c/a) cos(beta);
    FREE W3 = 1 + (2c/a) cos(beta), order = 1 - a/c. A condition a class does not have
    is 1. They are the penalty's walls written one per face (cos(beta) >= -m is
    ``W2 and W3`` for GLIDE and CENTRED).
    """
    return _mono_conditions(G[..., 0, 0], G[..., 2, 2], G[..., 0, 2], mono_class(sg))


def _gauss_reduce_2d(g: torch.Tensor, max_iter: int = 1000):
    """Lagrange-Gauss reduction of [B, 2, 2] Gram matrices: (M [B, 2, 2] long, M^T g M) with |u| <= |v| and
    |2 u.v| <= |u|^2 for the columns u, v of the new basis (up to rounding).

    Terminates: a reduction step is taken only when it shortens v by a relative 1e-12,
    a swap only when it puts the shorter vector first, and lattice vectors have a
    positive minimum length; ``max_iter`` passes bound it and raise.
    """
    g = g.clone()
    B = g.shape[0]
    M = torch.eye(2, dtype=torch.long, device=g.device).repeat(B, 1, 1)
    swap = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=g.device)
    for _ in range(max_iter):
        sw = g[:, 0, 0] > g[:, 1, 1]
        if bool(sw.any()):
            M = torch.where(sw[:, None, None], _imatmul(M, swap), M)
            g = torch.where(sw[:, None, None], g.flip(1).flip(2), g)
        m = torch.round(g[:, 0, 1] / g[:, 0, 0])
        new = g[:, 1, 1] - 2 * m * g[:, 0, 1] + m * m * g[:, 0, 0]
        ok = (m != 0) & (new < g[:, 1, 1] * (1 - 1e-12))
        if not bool(ok.any()):
            return M, g
        E = torch.eye(2, dtype=F64, device=g.device).repeat(B, 1, 1)
        E[:, 0, 1] = torch.where(ok, -m, torch.zeros_like(m))
        g = E.transpose(1, 2) @ g @ E
        M = _imatmul(M, E)
    raise RuntimeError(f'Gauss reduction did not converge in {max_iter} passes')


def _ac_gram(cell: torch.Tensor) -> torch.Tensor:
    """[B, 6] monoclinic cells -> [B, 2, 2] Gram matrices of (a, c), float64."""
    c = torch.as_tensor(cell, dtype=F64)
    gac = c[:, 0] * c[:, 2] * torch.cos(c[:, 4])
    return torch.stack([c[:, 0] ** 2, gac, gac, c[:, 2] ** 2], 1).reshape(-1, 2, 2)


def _mono_keys(g: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
    """[..., 2, 2] ac Gram, [...] b^2 -> [..., 3] (A, C, G_ac) / mean(A, B, C): the monoclinic tie-break key."""
    scale = (g[..., 0, 0] + g[..., 1, 1] + B2) / 3
    return torch.stack([g[..., 0, 0], g[..., 1, 1], g[..., 0, 1]], -1) / scale[..., None]


def _embed_ac(M: torch.Tensor) -> torch.Tensor:
    """[B, 2, 2] long ac changes -> [B, 3, 3] long N = [[p, 0, r], [0, det M, 0], [q, 0, s]] (det N = +1)."""
    N = torch.zeros(M.shape[0], 3, 3, dtype=torch.long, device=M.device)
    N[:, 0, 0], N[:, 0, 2], N[:, 2, 0], N[:, 2, 2] = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
    N[:, 1, 1] = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    return N


def _int_inverse(M: torch.Tensor) -> torch.Tensor:
    """Inverse of [..., 2, 2] long matrices with det +-1 (exact: adj(M) / det M)."""
    d = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    adj = torch.stack([M[..., 1, 1], -M[..., 0, 1], -M[..., 1, 0], M[..., 0, 0]], -1).reshape(M.shape)
    return adj * d[..., None, None]


def monoclinic_basis(cell: torch.Tensor, sg, tol: float = 1e-10, chunk: int = 4096, face_tol: float = FACE_TOL):
    """The margin-0 reduced basis of each monoclinic lattice within its setting class, chosen independently of the
    input basis.

    Candidates are ``M0 K``: M0 Lagrange-Gauss reduces the (a, c) plane, K runs over
    every 2x2 integer matrix with entries in [-MONO_SPAN, MONO_SPAN] and det +-1, and
    only products the class allows (``mono_parity``) count. Those whose largest
    condition violation (``mono_conditions``) is at most ``tol`` are tied (``n_tied``).
    Of them the least violating are kept: for a cell further than rounding from a
    face that is exactly the pair inside the domain (violation 0), so the choice does
    not depend on the input basis and the cell lies in the closed domain. Among those
    the lexicographically smallest (A, C, G_ac) / mean(A, B, C) is taken, each
    component compared within ``tol``; several pairs reach this step only within
    rounding of a face, which ``n_auto`` flags, so any deterministic rule would do
    there. Candidates come in pairs (M, -M) with one cell (-M is the 2-fold about b);
    of the chosen pair the one whose (trace, p, r, q) has a positive first nonzero
    entry is returned, so a reduced input keeps its basis. A lattice with no valid
    candidate gets the least violating one, and its violation is returned.

    Parameters
    ----------
    cell : [B, 6]
        Monoclinic cells; alpha and gamma are not read (taken as exactly pi/2).
    sg : int or [B]
        Space groups, 3-15.
    face_tol : float
        Tolerance on the tie-break key for counting automorphisms (``n_auto``); ``FACE_TOL`` by default.

    Returns
    -------
    (N [B, 3, 3] long, cell [B, 6] with alpha = gamma = pi/2 exactly, n_tied [B] long (candidate pairs within
    ``tol``; 1 off faces), violation [B], n_auto [B] long (automorphisms of the reduced metric within the setting
    class, modulo +-I, keys equal within ``face_tol``; 1 generic, 2 within about ``face_tol`` of one face))
    """
    c = torch.as_tensor(cell, dtype=F64)
    B = c.shape[0]
    dev = c.device
    if B == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return (torch.zeros(0, 3, 3, dtype=torch.long, device=dev), torch.zeros(0, 6, dtype=F64, device=dev), z,
                torch.zeros(0, dtype=F64, device=dev), z.clone())
    cls = mono_class(torch.as_tensor(sg, device=dev).reshape(-1).expand(B))
    if bool((cls < 0).any()):
        raise ValueError('monoclinic_basis takes space groups 3-15 only')
    g = _ac_gram(c)
    B2 = c[:, 1] ** 2
    M0, _ = _gauss_reduce_2d(g)
    K, AUT = _mono_sets(dev)
    out_M, out_tied, out_v, out_auto = [], [], [], []
    for lo in range(0, B, chunk):
        sl = slice(lo, lo + chunk)
        Mc = _imatmul(M0[sl, None], K[None])                                # [b, k, 2, 2] long
        Mf = Mc.to(F64)
        gc = Mf.transpose(-1, -2) @ g[sl, None] @ Mf                        # [b, k, 2, 2]
        cl = cls[sl, None]
        cond = _mono_conditions(gc[..., 0, 0], gc[..., 1, 1], gc[..., 0, 1], cl)
        viol = (-cond).clamp(min=0).amax(-1)
        viol = torch.where(mono_parity(Mc, cl), viol, torch.full_like(viol, float('inf')))
        vmin = viol.amin(1, keepdim=True)
        valid = viol <= torch.maximum(vmin, torch.full_like(vmin, tol))
        # of the tied candidates the least violating (the pair inside the domain), then the key
        pick = _lex_select(_mono_keys(gc, B2[sl, None]), valid & (viol <= vmin), tol, largest=False)
        Msel = Mc[torch.arange(len(pick), device=dev), pick]
        # of the pair (M, -M), the one with a positive leading (trace, p, r, q)
        lead = torch.stack([Msel[:, 0, 0] + Msel[:, 1, 1], Msel[:, 0, 0], Msel[:, 0, 1], Msel[:, 1, 0]], 1)
        first = lead.gather(1, (lead != 0).to(torch.int8).argmax(1, keepdim=True))[:, 0]
        Msel = torch.where((first < 0)[:, None, None], -Msel, Msel)
        out_M.append(Msel)
        out_tied.append(valid.sum(1) // 2)
        out_v.append(viol.gather(1, pick[:, None])[:, 0])
        # automorphisms of the reduced metric: K*^-1 A0 K* for A0 an automorphism of the Gauss-reduced metric (whose
        # automorphisms have entries in {-1, 0, 1}), counted where the class allows them, modulo +-I
        Ks = K[pick]
        g0 = M0[sl].to(F64).transpose(1, 2) @ g[sl] @ M0[sl].to(F64)
        Af = AUT.to(F64)
        ga = Af.transpose(-1, -2)[None] @ g0[:, None] @ Af[None]             # [b, a, 2, 2]
        same = ((_mono_keys(ga, B2[sl, None]) - _mono_keys(g0, B2[sl])[:, None]).abs() <= face_tol).all(-1)
        conj = _imatmul(_imatmul(_int_inverse(Ks)[:, None], AUT[None]), Ks[:, None])   # [b, a, 2, 2] long
        out_auto.append((same & mono_parity(conj, cl)).sum(1) // 2)
    Msel = torch.cat(out_M)
    gf = Msel.to(F64).transpose(1, 2) @ g @ Msel.to(F64)
    a_, c_ = torch.sqrt(gf[:, 0, 0]), torch.sqrt(gf[:, 1, 1])
    beta = torch.arccos((gf[:, 0, 1] / (a_ * c_)).clamp(-1, 1))
    half = torch.full_like(beta, torch.pi / 2)
    cell_new = torch.stack([a_, c[:, 1], c_, half, beta, half], 1)
    viol = (-_mono_conditions(gf[:, 0, 0], gf[:, 1, 1], gf[:, 0, 1], cls)).clamp(min=0).amax(-1)
    return _embed_ac(Msel), cell_new, torch.cat(out_tied), viol, torch.cat(out_auto)


def mono_setting_shifts(N: torch.Tensor, sg) -> torch.Tensor:
    """[B, 3] float64 origin shifts ``standardize.setting_shift(N_i, sg_i)`` (one call per distinct (sg, N)).

    Raises RuntimeError naming the rows whose N is outside its setting group.
    """
    N = torch.as_tensor(N).long()
    B = N.shape[0]
    sg = torch.as_tensor(sg, device=N.device).long().reshape(-1).expand(B)
    keys, inv = torch.unique(torch.cat([sg[:, None], N.reshape(B, 9)], 1), dim=0, return_inverse=True)
    shifts, bad = [], []
    for k in keys.cpu().numpy():
        o = setting_shift(k[1:].reshape(3, 3), int(k[0]))
        bad.append(o is None)
        shifts.append(np.zeros(3) if o is None else o)
    bad_rows = torch.nonzero(torch.as_tensor(bad, device=N.device)[inv])[:, 0].tolist()
    if bad_rows:
        raise RuntimeError(f'rows {bad_rows[:10]}: the chosen basis change is outside the setting group')
    return torch.as_tensor(np.stack(shifts), dtype=F64, device=N.device)[inv] if B else \
        torch.zeros(0, 3, dtype=F64, device=N.device)


def _reduce_monoclinic(desc: CrystalDescription, tol: float, face_tol: float) -> CellReduction:
    """Monoclinic rows whose alpha and gamma ``reduce_cell`` has checked; they are set to exactly pi/2 here."""
    d = desc.double()
    cell = d.cell.clone()
    cell[:, 3] = cell[:, 5] = torch.pi / 2
    d = d.replace(cell=cell)
    N, cell_new, n_tied, viol, n_auto = monoclinic_basis(cell, d.sg, tol=tol, face_tol=face_tol)
    if bool((viol > tol).any()):
        bad = torch.nonzero(viol > tol)[:, 0].tolist()
        raise RuntimeError(f'reduce_cell: rows {bad[:10]} found no basis within {tol} of the reduced set '
                           f'(largest violation {float(viol.max()):.3g})')
    s = mono_setting_shifts(N, d.sg)
    out = change_basis(d, N, s)
    return CellReduction(out.replace(cell=cell_new), N, s, n_tied, viol, n_auto)

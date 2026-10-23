"""Canonicalisation of whole batches with an outcome per row, and the computed fundamental-domain status check.

``canonicalize`` runs the cell step (``ops.reduce_cell`` for sg 1-15; for sg 16-230 the snap to the group's exact
metric and, in sg 16-74, the orthorhombic axis order of ``axis_order``), ``domains.to_sheet`` and ``domains.fold`` on
a batch that may mix space groups and Z', and may hold rows those steps refuse. Rows the steps cannot take are sorted
out first; a step that still raises (``RuntimeError`` or ``ValueError``) is retried on halves of its rows until the
failing rows stand alone, so one row never stops the batch. Each row gets a status code (``STATUS``) and, where a
step failed, a message. A row that is not canonicalised is returned as it came in (cast to float64). Every step works
on the first asymmetric unit only (every row it takes is Z' = 1), so padded unit slots never reach it.

The cell step for sg >= 16 (since 2026-10-21). The cell is snapped to the group's exact metric (``_exact_metric_cell``:
the mean of W^T G W over the rotations W of G; rows further than ``metric_tol`` from it are ``cell_off_group``), and
the size of that snap is recorded (``cell_snap``). Above the monoclinic groups the only setting changes that are not
isometries of the generic metric are orthorhombic axis permutations; in sg 16-74 the cell is put in the group's
canonical axis order (``axis_order``: a <= b <= c where all six orders are allowed, a <= b where a <-> b is, the
shortest axis first in Pbca), applied with its origin shift through ``ops.change_basis`` and recorded in ``N`` and
``s`` as ``reduce_cell``'s basis change is. In sg 75-230 the metric-preserving setting changes are Gamma's linear
parts, which the fold absorbs, so the snapped cell is canonical. Two allowed orders giving one cell within
``face_tol`` (in log lengths: a mirror face of the k = 6 and k = 2 classes, or Pbca's cone point a = b = c) make a
special metric (``n_auto`` > 1), as for sg 1-15.

``fd_status`` is the computed check (there is no stored flag): a row is in the fundamental domain (``is_fd``) when it
was canonicalised (any space group, since 2026-10-21) with Z' = 1, is not a special metric, is on the requested sheet,
and equals its canonical form within ``tol``.

Canonical is not duplicate-free: near a face of the cell chart or of the pose domain (``near_face``), a re-description
of one crystal rounded at float32 level can canonicalise to a different row. A special metric (``n_auto`` > 1) has a
canonical cell but a pose that depends on the input basis, and its canonical form is not a fixed point: canonicalising
it again can return the pose of an automorphic basis (for monoclinic cells on a face of the reduced domain this
alternates between two poses).

``canonicalize(..., nudge=delta)`` moves each ``canonical`` row with the ``NEAR_CELL`` or ``NEAR_CENTROID`` bit to the
nearest point at least ``delta`` from every face (cell-chart units for the cell, with t and R kept: sg 1-15 the
``charts.CellChart`` coordinates, sg 16-74 the log lengths of ``axis_order.OrthoCellChart``; fractional for the
centroid, with the cell and R kept and free axes at the gauge). An sg >= 16 row whose fold chose the pose by the
orientation rule (tie 2, a special position of Gamma+) first takes the pose that rule picks on the group's exact
metric (``_exact_metric_pose``; since the cell step snaps the cell, this is the canonical row's own pose up to
rounding). The moved row is a different crystal: its energy must be rescored. It is kept only if canonicalising it
again gives the same status, the row itself within ``NUDGE_FIXED_TOL``, and neither bit; otherwise the unmoved row
is kept with its flags and a reason.

What the nudge does not do (measured 2026-10-21):

- It moves the canonical image the input rounded to. A crystal within rounding of a face canonicalises to either side,
  and each side is moved inward, so two descriptions of it can nudge to two crystals about 2 delta apart, each its
  own fixed point (sg 2 rows on the a = b and v = 0 faces: 15 and 10 of 20 float32 re-descriptions).
- ``near_tol`` is an edge: a row within rounding of it is moved from one description and not from another, a jump of
  delta - near_tol. CIF coordinates on the 1e-5 grid sit exactly on the default.
- The cell step is in chart units, whose float32 rounding grows about as (c/a)^2 for triclinic cells: a nudged row's
  float32 re-descriptions in span-1 bases come back as another row from c/a about 30 (0 of 200 at c/a 20, 30 of 200
  at c/a 40).
- ``near_face`` has no bit for the margin of the orientation rule (the K cut of sg 6-9), so a row near that cut is
  neither flagged nor protected.
"""
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import math

import torch

from mxtaltools.crystal_fd import axis_order, domains, ops
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64

#: status codes of ``canonicalize``, by index. The first four are rows that were canonicalised; the others are
#: returned unchanged.
STATUS = (
    'canonical',       # cell step done (sg 1-15: reduced; sg >= 16: snapped to the group's exact metric, sg 16-74 in
                       # canonical axis order), on the sheet, pose folded; the canonical form is unique
    'special_metric',  # on the sheet, cell step done and folded, but n_auto > 1: the pose depends on the input basis
    'off_sheet',       # cell step done and folded on its own sheet: a Sohncke row cannot reach the requested sheet
                       # through G (its special-metric flag is still in n_auto)
    'pose_only',       # no longer produced (until 2026-10-21: sg >= 16 rows, pose folded, cell kept as given); the
                       # code is kept so that the codes stay stable
    'bad_values',      # sg outside 1-230, Z' < 1, a non-finite cell, centroid or rotation, an R that is not a proper
                       # rotation within rot_tol, or h not +-1 (in a used unit)
    'unreal_cell',     # no real cell (ops.real_cells)
    'z_prime_gt1',     # Z' > 1: not canonicalised (fd_tooling.md decision D3)
    'mono_angle_off',  # sg 3-15: alpha or gamma further than angle_tol from 90 degrees
    'cell_off_group',  # sg >= 16: the group's rotations do not preserve the cell's metric within metric_tol, so its
                       # normaliser elements are not isometries of it and the fold would change the crystal
    'reduce_failed',   # the cell step (ops.reduce_cell; sg >= 16 the snap and axis order) raised on this row alone
    'sheet_failed',    # domains.to_sheet raised on this row alone (message)
    'fold_failed',     # domains.fold raised on this row alone (message)
)
_CODE = {name: i for i, name in enumerate(STATUS)}
_DONE = (_CODE['canonical'], _CODE['special_metric'], _CODE['off_sheet'], _CODE['pose_only'])

#: bits of ``Canonicalization.near_face``
NEAR_CELL = 1         # the canonical cell within near_tol of a face of its cell chart (sg 1-15 the reduced cell's
                      # chart, sg 16-74 the axis-order chart in log lengths), or tied cell-step candidates
NEAR_CENTROID = 2     # the canonical centroid within near_tol (fractional) of a face of ASU(Gamma+)
NEAR_FOLD_BAND = 4    # fold took its tolerance-band fallback (FoldResult.tie == 3)

#: a nudged row must canonicalise to itself within this (cell lengths relative, angles in rad, t fractional, R entries)
NUDGE_FIXED_TOL = 1e-9

# a length box wide enough that only the cell chart's shape bounds count in the face distance
_WIDE_BOX = (torch.ones(3, dtype=F64), torch.full((3,), 1e4, dtype=F64))
_CELL_CHARTS: Dict = {}


@dataclass(frozen=True, eq=False)
class Canonicalization:
    """Result of ``canonicalize``; every tensor is per row, on the batch's device.

    Attributes
    ----------
    desc : CrystalDescription
        Float64. Canonicalised rows (``done``) in their canonical form (``nudged`` rows: moved off the face); every
        other row exactly as it came in.
    status : torch.Tensor
        [B] int8 index into ``STATUS``; ``names()`` gives the strings.
    message : list of str
        Per row: the error of a failed step, '' otherwise.
    N, s : torch.Tensor
        [B, 3, 3] long, [B, 3]: the cell step's basis change and origin shift, ``x_input = N x_canonical + s``:
        ``reduce_cell``'s (sg 1-15), the orthorhombic axis order's (sg 16-74: a signed axis permutation, det +1, with
        its quarter-grid shift; the identity where the cell is already in canonical order or the group allows one
        order); identity and zero for sg 75-230 and where no cell step.
    n_auto, n_tied : torch.Tensor
        [B] long, -1 where no cell step (``ops.CellReduction``; sg 16-74 ``axis_order.choose``: the order of the
        subgroup generated by the allowed orders giving the chosen cell within ``face_tol`` in log lengths (1, 2, 3
        or 6), and orders in the closed domain within ``cell_tol``; 1 for sg 75-230).
    violation : torch.Tensor
        [B], NaN where no cell step (0 for sg 75-230 and the one-order orthorhombic groups).
    cell_snap : torch.Tensor
        [B] float64: sg >= 16, the size of the snap of the input cell to the group's exact metric (largest change of a
        length, relative, or of an angle, rad); NaN where no snap (sg 1-15, rows not canonicalised).
    on_sheet : torch.Tensor
        [B] bool: a canonicalised row that ended on the requested sheet (every canonicalised row for ``sheet=None``).
    sheet_moved : torch.Tensor
        [B] bool: h was changed through an improper element of G (``domains.to_sheet``).
    element, lattice, free_shift, slack, tie : torch.Tensor
        ``domains.FoldResult``'s fields: element -1, slack NaN and tie -1 where not folded. Nudged rows keep the
        records of their unmoved canonical form (``N``, ``s``, ``n_auto``, ``n_tied``, ``violation`` and
        ``cell_snap`` too).
    near_face : torch.Tensor
        [B] int8 bitmask of ``NEAR_CELL``, ``NEAR_CENTROID``, ``NEAR_FOLD_BAND``; 0 where not canonicalised; for
        ``nudged`` rows, that of the moved row.
    moved : torch.Tensor
        [B] bool: the canonical form (``nudged`` rows: the moved row) differs from the input (cell lengths relative,
        angles, t, R, h) by more than ``moved_tol``.
    nudged : torch.Tensor
        [B] bool: ``desc`` holds the row moved off a face (``canonicalize``'s ``nudge``); False for ``nudge=None``.
    nudge_cell : torch.Tensor
        [B] float64: length of the cell's move in cell-chart coordinates (the chart of ``near_face``: sg 16-74 the
        log lengths (log a, log b, log c)); 0 if none.
    nudge_t : torch.Tensor
        [B] float64: largest absolute change of a fractional centroid coordinate; 0 if none.
    nudge_message : list of str
        Per row: why a row with the ``NEAR_CELL`` or ``NEAR_CENTROID`` bit was not nudged; '' otherwise.
    """
    desc: CrystalDescription
    status: torch.Tensor
    message: List[str]
    N: torch.Tensor
    s: torch.Tensor
    n_auto: torch.Tensor
    n_tied: torch.Tensor
    violation: torch.Tensor
    cell_snap: torch.Tensor
    on_sheet: torch.Tensor
    sheet_moved: torch.Tensor
    element: torch.Tensor
    lattice: torch.Tensor
    free_shift: torch.Tensor
    slack: torch.Tensor
    tie: torch.Tensor
    near_face: torch.Tensor
    moved: torch.Tensor
    nudged: torch.Tensor
    nudge_cell: torch.Tensor
    nudge_t: torch.Tensor
    nudge_message: List[str]

    def names(self) -> List[str]:
        """Per-row status names."""
        return [STATUS[int(c)] for c in self.status.tolist()]

    @property
    def done(self) -> torch.Tensor:
        """[B] bool: the row was canonicalised (its ``desc`` row is a canonical form, of its own sheet)."""
        return _isin(self.status, _DONE)


@dataclass(frozen=True, eq=False)
class FDStatus:
    """Result of ``fd_status``; every field is a [B] bool tensor except ``status``.

    Attributes
    ----------
    is_fd : decided, not a special metric, on the sheet, and equal to its canonical form within ``tol``.
    decided : canonicalised with a cell step (every space group since 2026-10-21; a ``pose_only`` row, no longer
        produced, is not decided) and Z' = 1; special metrics and off-sheet rows included (they are decided not to be
        in the domain).
    cell_reduced : decided, and the row's cell equals its canonical cell (sg 1-15 reduced; sg >= 16 snapped to the
        exact metric and, sg 16-74, in canonical axis order) within ``tol``.
    pose_folded : the row's pose equals the fold of the row as given (in its own basis) within ``tol``; False for rows
        fold cannot take.
    on_sheet : Z' = 1 and h equals the requested sheet (any h for ``sheet=None``).
    special_metric : the reduced cell has ``n_auto`` > 1.
    near_face : the canonical form is near a face (``Canonicalization.near_face`` != 0).
    zp1 : Z' = 1.
    status : [B] int8 ``canonicalize`` status codes (``STATUS``).
    """
    is_fd: torch.Tensor
    decided: torch.Tensor
    cell_reduced: torch.Tensor
    pose_folded: torch.Tensor
    on_sheet: torch.Tensor
    special_metric: torch.Tensor
    near_face: torch.Tensor
    zp1: torch.Tensor
    status: torch.Tensor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _isin(codes: torch.Tensor, allowed) -> torch.Tensor:
    return torch.isin(codes.to(torch.long), torch.tensor(allowed, dtype=torch.long, device=codes.device))


def _isolate(fn: Callable, d: CrystalDescription) -> Tuple[list, Dict[int, str]]:
    """Run ``fn`` on the rows of ``d``; where it raises a ``RuntimeError`` or ``ValueError``, split the rows in half
    and retry, down to single rows. Any other exception propagates (a programming error is not a row's problem).

    Returns ([(row indices, result), ...], {row: message} for rows that raise on their own). Every split halves a
    set of at least two rows and a single row is never split, so this ends after at most 2 B - 1 calls.
    """
    done, failed = [], {}
    if d.num_rows == 0:
        return done, failed
    stack = [torch.arange(d.num_rows, device=d.device)]
    while stack:
        idx = stack.pop()
        try:
            done.append((idx, fn(d[idx])))
        except (RuntimeError, ValueError) as e:      # includes NotImplementedError and torch's LinAlgError
            if len(idx) == 1:
                failed[int(idx[0])] = f'{type(e).__name__}: {str(e)[:300]}'
            else:
                half = len(idx) // 2
                stack += [idx[half:], idx[:half]]
    return done, failed


def _unit0(d: CrystalDescription) -> CrystalDescription:
    """The first asymmetric unit of every row, as a Z' = 1 batch (Zmax 1)."""
    return CrystalDescription(sg=d.sg, cell=d.cell, z_prime=torch.ones_like(d.z_prime), unit_mask=d.unit_mask[:, :1],
                              t=d.t[:, :1], R=d.R[:, :1], h=d.h[:, :1])


def _cell_chart(sg: int):
    """The cell chart of space group ``sg`` whose bounds ``near_face`` measures, cached: sg 1-15 a
    ``charts.CellChart`` with a wide length box, sg 16-74 ``axis_order.OrthoCellChart`` (log lengths)."""
    if axis_order.ORTHORHOMBIC[0] <= sg <= axis_order.ORTHORHOMBIC[-1]:
        return axis_order.cell_chart(sg)
    from mxtaltools.crystal_fd.charts import CellChart
    key = 'triclinic' if sg <= 2 else int(sg)
    if key not in _CELL_CHARTS:
        _CELL_CHARTS[key] = CellChart('triclinic', _WIDE_BOX) if sg <= 2 else CellChart('monoclinic', _WIDE_BOX, sg=sg)
    return _CELL_CHARTS[key]


def _cell_faces(chart) -> Tuple[torch.Tensor, torch.Tensor]:
    """(A [k, n], o [k]) float64: the face values are y A^T + o (sg 1-15 the columns of ``_cell_face_distance``, in
    order; sg 16-74 those of ``axis_order.face_values``)."""
    if chart.family == 'orthorhombic':  # differences of log lengths (axis_order.AxisOrders.faces)
        return chart.faces, torch.zeros(chart.faces.shape[0], dtype=F64)
    if chart.family == 'triclinic':     # 1 - r2, 1 - r1, s_beta + 1, -s_beta, s_gamma + 1, -s_gamma, v, 1 - v
        A = torch.zeros(8, 6, dtype=F64)
        for i, (col, sign) in enumerate(((1, -1), (2, -1), (3, 1), (3, -1), (4, 1), (4, -1), (5, 1), (5, -1))):
            A[i, col] = sign
        return A, torch.tensor([1., 1., 1., 0., 1., 0., 0., 1.], dtype=F64)
    rows, o = [[0., 0., 0., 1.], [0., 0., 0., -1.]], [0., 1.]          # s, 1 - s
    if chart.mono_class == 2:                                          # FREE: log c - log a
        rows.append([1., -1., 0., 0.])
        o.append(0.)
    return torch.tensor(rows, dtype=F64), torch.tensor(o, dtype=F64)


def _cell_face_distance(sg: int, cell: torch.Tensor) -> torch.Tensor:
    """[b] distance, in cell-chart units, of reduced cells of one space group (1-15) to the chart's shape bounds. The
    face distance of sg 16-74 comes from the axis-order step (``axis_order.choose``, through ``_high_cell``)."""
    if not 1 <= sg <= 15:
        raise ValueError(f'_cell_face_distance takes space groups 1-15, not {sg} (sg 16-74: axis_order.choose)')
    chart = _cell_chart(sg)
    y, _ = chart.unmap(cell.detach().to('cpu', F64))
    if sg <= 2:     # r2 <= 1, r1 <= 1, s_beta and s_gamma in [-1, 0], v in [0, 1]
        dist = torch.stack([1 - y[:, 1], 1 - y[:, 2], y[:, 3] + 1, -y[:, 3], y[:, 4] + 1, -y[:, 4],
                            y[:, 5], 1 - y[:, 5]], 1)
    else:           # s in [0, 1], and a <= c for the FREE class
        cols = [y[:, 3], 1 - y[:, 3]]
        if chart.mono_class == 2:
            cols.append(y[:, 0] - y[:, 1])
        dist = torch.stack(cols, 1)
    return dist.amin(1).to(cell.device)


def _bad_values(d: CrystalDescription, rot_tol: float) -> torch.Tensor:
    """[B] bool: sg outside 1-230, Z' < 1, a non-finite cell, or a used unit with a non-finite centroid or rotation,
    an R that is not a proper rotation within ``rot_tol``, or h not +-1."""
    m = d.unit_mask
    t_ok = (torch.isfinite(d.t).all(-1) | ~m).all(1)
    R_fin = torch.isfinite(d.R).all(-1).all(-1)
    R = torch.where(R_fin[..., None, None], d.R, torch.eye(3, dtype=d.R.dtype, device=d.device))
    eye = torch.eye(3, dtype=R.dtype, device=d.device)
    orth = (R @ R.transpose(-1, -2) - eye).abs().amax((-1, -2)) <= rot_tol
    proper = (torch.linalg.det(R) - 1).abs() <= rot_tol
    R_ok = ((R_fin & orth & proper) | ~m).all(1)
    h_ok = ((d.h.abs() == 1) | ~m).all(1)
    sg_ok = (d.sg >= 1) & (d.sg <= 230) & (d.z_prime >= 1)
    return ~(sg_ok & torch.isfinite(d.cell).all(1) & t_ok & R_ok & h_ok)


def _metric_deviation(sg: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """[b]: max over the rotations W of G and entries (i, j) of |(W^T G W - G)_ij| / sqrt(G_ii G_jj), G the cell's
    metric (``io_mxt.metric_deviation``'s measure); 0 for a cell the group's rotations preserve exactly."""
    G = ops.metric_tensor(cell)
    n = torch.sqrt(torch.diagonal(G, dim1=1, dim2=2))
    out = torch.zeros(len(G), dtype=F64, device=cell.device)
    for s in torch.unique(sg).tolist():
        m = sg == s
        W = group_table(int(s), cell.device).g_W.to(F64)
        g = G[m]
        Gw = torch.einsum('wji,bjk,wkl->bwil', W, g, W)
        out[m] = ((Gw - g[:, None]).abs() / (n[m][:, None, :, None] * n[m][:, None, None, :])).amax((1, 2, 3))
    return out


def _differs(a: CrystalDescription, b: CrystalDescription, tol: float) -> torch.Tensor:
    """[B] bool: unit 0 of two batches differs in cell (lengths relative), t, R or h by more than ``tol``."""
    rel = ((a.cell[:, :3] - b.cell[:, :3]).abs() / b.cell[:, :3].abs()).amax(1)
    ang = (a.cell[:, 3:] - b.cell[:, 3:]).abs().amax(1)
    dt = (a.t[:, 0] - b.t[:, 0]).abs().amax(1)
    dR = (a.R[:, 0] - b.R[:, 0]).abs().amax(-1).amax(-1)
    return (rel > tol) | (ang > tol) | (dt > tol) | (dR > tol) | (a.h[:, 0] != b.h[:, 0])


def _dtype_tol(desc: CrystalDescription) -> float:
    """The comparison tolerance a description's dtype supports: 1e-9 for float64, 1e-5 otherwise."""
    return 1e-9 if desc.cell.dtype == torch.float64 else 1e-5


def _check_tol(**tols):
    for name, v in tols.items():
        if v is not None and not (math.isfinite(v) and v > 0):
            raise ValueError(f'{name} must be finite and positive, not {v}')


# ---------------------------------------------------------------------------
# nudge
# ---------------------------------------------------------------------------

def _min_norm_push(A: torch.Tensor, b: torch.Tensor, eps: float = 1e-13) -> Optional[torch.Tensor]:
    """The shortest x with A x >= b (float64, a few rows and columns), or None when no x satisfies them all.

    Rows with b <= 0 hold at x = 0. The rows in play start as those with b > 0; in each round the answer is the
    shortest, among linearly independent subsets S of them (at most n rows), of the minimum-norm solution of
    A_S x = b_S that satisfies every row in play (the shortest feasible x is one of these: by Caratheodory its active
    rows contain an independent subset that spans it). Rows that answer breaks join the play and the round repeats;
    the set grows every round, so there are at most k rounds. Feasibility is tested within ``eps``.
    """
    k, n = A.shape
    pool = [i for i in range(k) if float(b[i]) > 0]
    if not pool:
        return torch.zeros(n, dtype=F64)
    for _ in range(k):
        best = None
        for r in range(1, min(n, len(pool)) + 1):
            for S in itertools.combinations(pool, r):
                AS = A[list(S)]
                M = AS @ AS.T
                ev = torch.linalg.eigvalsh(M)
                if not float(ev[0]) > 1e-12 * float(ev[-1]):                     # dependent rows (or a zero row)
                    continue
                x = AS.T @ torch.linalg.solve(M, b[list(S)])
                if bool((A[pool] @ x >= b[pool] - eps).all()) and (best is None or x.norm() < best.norm()):
                    best = x
        if best is None:
            return None
        broken = [i for i in range(k) if i not in pool and float(A[i] @ best) < float(b[i]) - eps]
        if not broken:
            return best
        pool = sorted(set(pool) | set(broken))
    return None


def _exact_metric_cell(sg: int, cell: torch.Tensor) -> torch.Tensor:
    """[b, 6] float64: the cells whose metric is the mean of W^T G W over the distinct rotations W of space group ``sg``
    (G the metric of ``cell``), which those rotations preserve to rounding.

    Computed with elementwise products and sums in a fixed order (no matmul or einsum, whose rounding can depend on the
    batch's shape), so a row's result does not depend on the other rows it is computed with. The cell step for
    sg >= 16 snaps every cell with it; for an orthorhombic group every rotation is diagonal, so the off-diagonal mean
    is exactly 0 (angles exactly pi/2) and each length is the mean of copies of its own square.
    """
    G = ops.metric_tensor(cell)
    W = torch.unique(group_table(int(sg), cell.device).g_W.to(F64).reshape(-1, 9), dim=0).reshape(-1, 3, 3)
    tot = torch.zeros_like(G)
    for Wq in W:
        term = torch.zeros_like(G)
        for j in range(3):
            for k in range(3):
                term = term + Wq[j, :, None] * G[:, j, k, None, None] * Wq[k, None, :]
        tot = tot + term
    return ops.cell_from_metric_tensor(tot / W.shape[0])


class _HighCell(NamedTuple):
    """Result of ``_high_cell``, laid out as ``ops.CellReduction`` plus the snap size and the face distance."""
    desc: CrystalDescription
    N: torch.Tensor           #: [b, 3, 3] long, input basis -> canonical axis order (identity outside 16-74)
    s: torch.Tensor           #: [b, 3] origin shift
    n_tied: torch.Tensor      #: [b] orders in the closed domain within tol (1 off faces and outside 16-74)
    violation: torch.Tensor   #: [b] largest face violation of the chosen order (0 outside 16-74)
    n_auto: torch.Tensor      #: [b] allowed orders giving the chosen cell within face_tol (1 generic)
    snap: torch.Tensor        #: [b] size of the snap to the exact metric (lengths relative, angles rad)
    face: torch.Tensor        #: [b] smallest face value of the canonical cell (log units); NaN without faces


def _high_cell(x: CrystalDescription, tol: float, face_tol: float) -> _HighCell:
    """The cell step of sg 16-230 on Z' = 1 rows (module docstring): every cell snapped to its group's exact metric
    (``_exact_metric_cell``), then, in sg 16-74, put in the group's canonical axis order (``axis_order.choose`` on the
    snapped lengths) through ``ops.change_basis`` with that order's (N, s); the new cell is the snapped cell's lengths
    and angles permuted. Rows whose order is the identity are kept as snapped (t and R bit for bit)."""
    d = x.double()
    B, dev = d.num_rows, d.device
    snapped = d.cell.clone()
    for sg in torch.unique(d.sg).tolist():
        m = d.sg == sg
        snapped[m] = _exact_metric_cell(int(sg), d.cell[m])
    snap = torch.maximum(((snapped[:, :3] - d.cell[:, :3]).abs() / d.cell[:, :3]).amax(1),
                         (snapped[:, 3:] - d.cell[:, 3:]).abs().amax(1))
    N = torch.eye(3, dtype=torch.long, device=dev).repeat(B, 1, 1)
    s = torch.zeros(B, 3, dtype=F64, device=dev)
    perm = torch.arange(3, device=dev).repeat(B, 1)
    n_tied = torch.ones(B, dtype=torch.long, device=dev)
    n_auto = torch.ones(B, dtype=torch.long, device=dev)
    viol = torch.zeros(B, dtype=F64, device=dev)
    face = torch.full((B,), float('nan'), dtype=F64, device=dev)
    for sg in torch.unique(d.sg).tolist():
        if sg not in axis_order.ORTHORHOMBIC:
            continue
        m = torch.nonzero(d.sg == sg)[:, 0]
        orders = axis_order.axis_orders(sg)
        pick, n_tied[m], n_auto[m], viol[m], face[m] = axis_order.choose(sg, snapped[m], tol, face_tol)
        N[m], s[m], perm[m] = orders.N.to(dev)[pick], orders.s.to(dev)[pick], orders.perm.to(dev)[pick]
    out = d.replace(cell=snapped)
    move = torch.nonzero((N != torch.eye(3, dtype=torch.long, device=dev)).any(2).any(1))[:, 0]
    if len(move):
        moved = ops.change_basis(out[move], N[move], s[move])
        cell, t, R = snapped.clone(), out.t.clone(), out.R.clone()
        cell[move] = torch.cat([snapped[move, :3].gather(1, perm[move]), snapped[move, 3:].gather(1, perm[move])], 1)
        t[move], R[move] = moved.t, moved.R
        out = out.replace(cell=cell, t=t, R=R)
    return _HighCell(out, N, s, n_tied, viol, n_auto, snap, face)


def _exact_metric_pose(row: CrystalDescription, tol: float) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(t [3], R [3, 3]) of a canonical row (sg >= 16) whose fold chose its pose by the orientation rule (tie 2).

    ``domains.fold`` compares the candidate poses Q R with Q = T W T^-1 from the row's cell. Off the group's metric
    (a float32 cell: about 4e-8) Q is that far from a signed permutation, so poses equal on the exact metric differ
    by rounding and rounding decides. Here the rule is run on ``_exact_metric_cell``, and the element it picks is
    applied with the row's own Q, so the pose is an image of the row's. None when that fold does not tie at the
    row's centroid.
    """
    sg = int(row.sg[0])
    x = _unit0(row)
    fr = domains.fold(x.replace(cell=_exact_metric_cell(sg, x.cell)), tol=tol)
    t0 = x.t[0, 0]
    if int(fr.tie[0]) != 2 or float((fr.desc.t[0, 0] - t0).abs().max()) > tol:
        return None
    e = int(fr.element[0])
    if e == 0:                                                   # the identity: the pose already chosen
        return t0, x.R[0, 0]
    W = domains.translation_section(sg, x.device).W[e]
    return fr.desc.t[0, 0], ops.cartesian_rotation(x.cell_matrix(), W[None])[0] @ x.R[0, 0]


def _nudge_rows(canon: CrystalDescription, d: CrystalDescription, code: torch.Tensor, near: torch.Tensor,
                moved: torch.Tensor, tie: torch.Tensor, nudge: float, moved_tol: float, tol: float, recanon: Callable):
    """Move the canonical (and pose_only, no longer produced) rows with ``NEAR_CELL`` or ``NEAR_CENTROID`` ``nudge``
    off their faces.

    Cell (``NEAR_CELL``; sg 1-15, and sg 16-74 in the log lengths of ``axis_order.OrthoCellChart``): the shortest move
    of the cell-chart point ``y`` with every face value of ``_cell_faces`` at least ``nudge``, mapped back to a cell;
    t and R kept. Centroid (``NEAR_CENTROID``): the shortest
    move of t (free coordinates fixed at the gauge) with every top-level cut of ASU(Gamma+) at least ``nudge`` (unit
    normals, fractional); the cell kept, and R kept except that an sg >= 16 row with fold tie 2 first takes
    ``_exact_metric_pose``. A row is kept moved only if ``recanon`` (canonicalize, nudge=None) gives it the same
    status, itself within ``NUDGE_FIXED_TOL`` and no ``NEAR_CELL`` / ``NEAR_CENTROID`` bit.

    Returns (desc, near, moved, nudged, nudge_cell, nudge_t, nudge_message), every row not moved as it came in.
    """
    B, dev = canon.num_rows, canon.device
    nudged = torch.zeros(B, dtype=torch.bool, device=dev)
    move_cell = torch.zeros(B, dtype=F64, device=dev)
    move_t = torch.zeros(B, dtype=F64, device=dev)
    message = [''] * B
    bits = near.to(torch.long) & (NEAR_CELL | NEAR_CENTROID)
    rows = torch.nonzero(_isin(code, (_CODE['canonical'], _CODE['pose_only'])) & (bits != 0))[:, 0].tolist()
    if not rows:
        return canon, near, moved, nudged, move_cell, move_t, message
    cell, t, R = canon.cell.clone(), canon.t.clone(), canon.R.clone()
    tried = []
    for i in rows:
        sg, kind = int(canon.sg[i]), int(bits[i])
        new_cell, dy, dt, new_R = None, None, None, None
        if kind & NEAR_CELL:
            chart = _cell_chart(sg)
            A, o = _cell_faces(chart)
            y, _ = chart.unmap(canon.cell[i:i + 1].detach().to('cpu', F64))
            dy = _min_norm_push(A, nudge - (y[0] @ A.T + o))
            if dy is None:
                message[i] = f'nudge: no point of the cell chart is {nudge:g} from every face (chart thinner than 2 nudge)'
                continue
            if bool((dy != 0).any()):
                new_cell = chart.map(y + dy)[0][0].to(dev)
        if kind & NEAR_CENTROID:
            t0 = canon.t[i, 0]
            if sg >= 16 and int(tie[i]) == 2:
                pose = _exact_metric_pose(canon[i:i + 1], tol)
                if pose is None:
                    message[i] = "nudge: the fold on the group's exact metric does not tie at the canonical centroid"
                    continue
                t0, new_R = pose
            sec = domains.translation_section(sg)
            x = torch.where(sec.free, sec.gauge, t0.detach().to('cpu', F64))
            A = sec.region.normals
            free = sec.free[None].expand_as(A)
            dt = _min_norm_push(torch.where(free, torch.zeros_like(A), A), nudge - (x @ A.T + sec.region.offsets))
            if dt is None:
                message[i] = (f'nudge: no point of ASU(Gamma+) is {nudge:g} from every face with free coordinates at the '
                              'gauge (domain thinner than 2 nudge)')
                continue
            if bool((dt != 0).any()):
                t[i, 0] = t0 + dt.to(dev)
                move_t[i] = dt.abs().max()
        if new_cell is not None:
            cell[i] = new_cell
            move_cell[i] = dy.norm()
        if new_R is not None:
            R[i, 0] = new_R
        tried.append(i)
    if not tried:
        return canon, near, moved, nudged, torch.zeros_like(move_cell), torch.zeros_like(move_t), message
    idx = torch.tensor(tried, dtype=torch.long, device=dev)
    x = canon.replace(cell=cell, t=t, R=R)[idx]
    again = recanon(x)
    away = _differs(again.desc, x, NUDGE_FIXED_TOL)
    out_cell, out_t, out_R = canon.cell.clone(), canon.t.clone(), canon.R.clone()
    near, moved = near.clone(), moved.clone()
    for j, i in enumerate(tried):
        if int(again.status[j]) != int(code[i]):
            message[i] = f'nudge: the nudged row canonicalises to status {STATUS[int(again.status[j])]}'
        elif bool(away[j]):
            message[i] = f'nudge: the nudged row is not its own canonical form within {NUDGE_FIXED_TOL:g}'
        elif int(again.near_face[j]) & (NEAR_CELL | NEAR_CENTROID):
            message[i] = f'nudge: the nudged row is still near a face (near_face {int(again.near_face[j])})'
        else:
            out_cell[i], out_t[i], out_R[i] = x.cell[j], x.t[j], x.R[j]
            near[i] = again.near_face[j]
            nudged[i] = True
            continue
        move_cell[i], move_t[i] = 0.0, 0.0
    out = canon.replace(cell=out_cell, t=out_t, R=out_R)
    if bool(nudged.any()):
        moved[nudged] = _differs(out[nudged], d[nudged], moved_tol)
    return out, near, moved, nudged, move_cell, move_t, message


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------

def canonicalize(desc: CrystalDescription, *, sheet: Optional[int] = 1, tol: float = 1e-9, cell_tol: float = 1e-10,
                 angle_tol: float = 1e-6, face_tol: float = ops.FACE_TOL, near_tol: float = ops.FACE_TOL,
                 rot_tol: float = 1e-6, metric_tol: float = 1e-4,
                 moved_tol: Optional[float] = None, nudge: Optional[float] = None) -> Canonicalization:
    """Canonical descriptions of a batch, with a status per row; no row-level problem raises.

    Steps, each on the rows still in play: sort out rows with bad values, no real cell, Z' > 1, a monoclinic alpha or
    gamma off 90 degrees, or (sg >= 16) a cell the group does not preserve; ``ops.reduce_cell`` on sg 1-15; on sg
    16-230 the snap to the exact metric and, sg 16-74, the canonical axis order (``_high_cell``, module docstring);
    ``domains.to_sheet(sheet)`` (skipped for ``sheet=None``); ``domains.fold``. A step that raises is retried on halves
    of its rows (``_isolate``), so only the rows that fail on their own are marked failed, and those are returned
    unchanged. Bad arguments raise.

    Parameters
    ----------
    desc : CrystalDescription
        Any batch: space groups and Z' may be mixed, float32 or float64.
    sheet : int or None
        Handedness to move rows to: +1, the fundamental domain's sheet, or -1; None keeps every row's handedness.
        Only improper elements of G are used (the same crystal), so rows of Sohncke groups stay on their sheet
        (``on_sheet`` False). How such a row should be re-described (a mirrored reference, decision D2, or the
        enantiomorph) is not this function's choice.
    tol : float
        ``domains.fold``'s face tolerance.
    cell_tol, angle_tol, face_tol : float
        ``ops.reduce_cell``'s ``tol``, ``angle_tol`` and ``face_tol`` (``face_tol`` sets the special-metric flag). For
        sg 16-74 ``cell_tol`` is the axis order's tie tolerance and ``face_tol`` its automorphism tolerance, both on
        log lengths (``axis_order.choose``).
    near_tol : float
        Distance (cell-chart units for the cell, fractional for the centroid) within which a canonical form is
        ``near_face``; the default is the CSD dataset's. Sg 16-74: log-length differences, the unit of ``face_tol``
        there, so at the defaults every cell within ``near_tol`` of a mirror face of the axis order is a special
        metric.
    rot_tol : float
        A used R further than this from a proper rotation (entries of R R^T - I, det R - 1) is ``bad_values``
        (``CrystalDescription.validate``'s default).
    metric_tol : float
        sg >= 16: a cell whose metric the group's rotations change by more than this (relative,
        ``io_mxt.metric_deviation``'s measure) is ``cell_off_group``.
    moved_tol : float, optional
        Tolerance of ``moved``; default 1e-9 for float64 input and 1e-5 for float32 input (its rounding; monoclinic
        alpha and gamma are set to exactly 90 degrees). Float32-origin values passed as float64 need it set.
    nudge : float, optional
        None: no row is moved. A step delta (> ``near_tol``): ``canonical`` rows with the ``NEAR_CELL`` or
        ``NEAR_CENTROID`` bit are moved to the nearest point at least delta from every face, in the units of
        ``near_face`` (module docstring; ``nudged``, ``nudge_cell``, ``nudge_t``, ``nudge_message``); for sg 16-74
        the cell's faces are log-length differences (Pbca's seams a = b and a = c, and mirror faces outside
        ``face_tol`` when ``near_tol`` exceeds it). Special metrics, off-sheet rows and rows with ``NEAR_FOLD_BAND``
        alone are not moved. A moved row is another crystal: rescore its energy.
    """
    if sheet not in (1, -1, None):
        raise ValueError('sheet must be +1, -1 or None')
    if isinstance(nudge, bool):
        raise ValueError(f'nudge must be None or a step, not {nudge}')
    _check_tol(tol=tol, cell_tol=cell_tol, angle_tol=angle_tol, face_tol=face_tol, near_tol=near_tol, rot_tol=rot_tol,
               metric_tol=metric_tol, moved_tol=moved_tol, nudge=nudge)
    if not face_tol >= cell_tol:
        raise ValueError(f'face_tol ({face_tol}) must be at least cell_tol ({cell_tol})')
    if nudge is not None and not nudge > near_tol:
        raise ValueError(f'nudge ({nudge}) must exceed near_tol ({near_tol}): a nudged row would still be near a face')
    moved_tol = _dtype_tol(desc) if moved_tol is None else moved_tol
    d = desc.double()
    B, dev = d.num_rows, d.device
    code = torch.full((B,), -1, dtype=torch.int8, device=dev)                     # -1: still in play
    message = [''] * B

    def mark(mask, name):
        code[mask & (code < 0)] = _CODE[name]

    # ---- rows the steps cannot take, in precedence order
    mark(_bad_values(d, rot_tol), 'bad_values')
    mark(~ops.real_cells(torch.nan_to_num(d.cell, nan=0.0)).to(dev), 'unreal_cell')
    mark(d.z_prime != 1, 'z_prime_gt1')
    mono = (d.sg >= 3) & (d.sg <= 15)
    mark(mono & ((d.cell[:, [3, 5]] - torch.pi / 2).abs() > angle_tol).any(1), 'mono_angle_off')
    high = torch.nonzero((code < 0) & (d.sg >= 16))[:, 0]
    if len(high):
        off = torch.zeros(B, dtype=torch.bool, device=dev)
        off[high] = _metric_deviation(d.sg[high], d.cell[high]) > metric_tol
        mark(off, 'cell_off_group')

    # every step works on the first unit only: every row it takes is Z' = 1
    x = _unit0(d)
    cell, t, R, h = x.cell.clone(), x.t.clone(), x.R.clone(), x.h.clone()
    rec = dict(N=torch.eye(3, dtype=torch.long, device=dev).repeat(B, 1, 1),
               s=torch.zeros(B, 3, dtype=F64, device=dev),
               n_auto=torch.full((B,), -1, dtype=torch.long, device=dev),
               n_tied=torch.full((B,), -1, dtype=torch.long, device=dev),
               violation=torch.full((B,), float('nan'), dtype=F64, device=dev),
               cell_snap=torch.full((B,), float('nan'), dtype=F64, device=dev),
               on_sheet=torch.ones(B, dtype=torch.bool, device=dev),
               sheet_moved=torch.zeros(B, dtype=torch.bool, device=dev),
               element=torch.full((B,), -1, dtype=torch.long, device=dev),
               lattice=torch.zeros(B, 3, dtype=torch.long, device=dev),
               free_shift=torch.zeros(B, 3, dtype=F64, device=dev),
               slack=torch.full((B,), float('nan'), dtype=F64, device=dev),
               tie=torch.full((B,), -1, dtype=torch.int8, device=dev))
    cell_face = torch.full((B,), float('nan'), dtype=F64, device=dev)

    def fail(rows, failed, name, step):
        for j, msg in failed.items():
            code[rows[j]] = _CODE[name]
            message[int(rows[j])] = f'{step}: {msg}'

    # ---- cell: sg 1-15
    rows = torch.nonzero((code < 0) & (d.sg <= 15))[:, 0]
    parts, failed = _isolate(lambda y: ops.reduce_cell(y, tol=cell_tol, angle_tol=angle_tol, face_tol=face_tol),
                             x[rows])
    fail(rows, failed, 'reduce_failed', 'reduce_cell')
    for idx, red in parts:
        g = rows[idx]
        cell[g], t[g], R[g], h[g] = red.desc.cell, red.desc.t, red.desc.R, red.desc.h
        rec['N'][g], rec['s'][g] = red.N.to(dev), red.s.to(dev, F64)
        rec['n_auto'][g], rec['n_tied'][g] = red.n_auto.to(dev), red.n_tied.to(dev)
        rec['violation'][g] = red.violation.to(dev, F64)
        for sg in torch.unique(d.sg[g]).tolist():
            m = d.sg[g] == sg
            cell_face[g[m]] = _cell_face_distance(int(sg), red.desc.cell[m])

    # ---- cell: sg 16-230 (snap to the exact metric; sg 16-74 the canonical axis order)
    rows = torch.nonzero((code < 0) & (d.sg >= 16))[:, 0]
    for sg in torch.unique(d.sg[rows]).tolist():        # derived outside _isolate: tables off the survey raise here
        if sg in axis_order.ORTHORHOMBIC:
            axis_order.axis_orders(sg)
    parts, failed = _isolate(lambda y: _high_cell(y, cell_tol, face_tol), x[rows])
    fail(rows, failed, 'reduce_failed', 'cell step (sg >= 16)')
    for idx, hc in parts:
        g = rows[idx]
        cell[g], t[g], R[g], h[g] = hc.desc.cell, hc.desc.t, hc.desc.R, hc.desc.h
        rec['N'][g], rec['s'][g] = hc.N.to(dev), hc.s.to(dev, F64)
        rec['n_auto'][g], rec['n_tied'][g] = hc.n_auto.to(dev), hc.n_tied.to(dev)
        rec['violation'][g], rec['cell_snap'][g] = hc.violation.to(dev, F64), hc.snap.to(dev, F64)
        cell_face[g] = hc.face.to(dev, F64)

    # ---- sheet
    if sheet is not None:
        work = x.replace(cell=cell, t=t, R=R, h=h)
        rows = torch.nonzero(code < 0)[:, 0]
        parts, failed = _isolate(lambda y: domains.to_sheet(y, sheet), work[rows])
        fail(rows, failed, 'sheet_failed', 'to_sheet')
        for idx, (moved_d, ok) in parts:
            g = rows[idx]
            rec['on_sheet'][g] = ok.to(dev)
            rec['sheet_moved'][g] = moved_d.h[:, 0] != h[g, 0]
            t[g], R[g], h[g] = moved_d.t, moved_d.R, moved_d.h

    # ---- pose: every group
    work = x.replace(cell=cell, t=t, R=R, h=h)
    rows = torch.nonzero(code < 0)[:, 0]
    parts, failed = _isolate(lambda y: domains.fold(y, tol=tol), work[rows])
    fail(rows, failed, 'fold_failed', 'fold')
    for idx, fr in parts:
        g = rows[idx]
        t[g], R[g] = fr.desc.t, fr.desc.R
        rec['element'][g], rec['lattice'][g] = fr.element.to(dev), fr.lattice.to(dev)
        rec['free_shift'][g], rec['slack'][g], rec['tie'][g] = fr.free_shift.to(dev), fr.slack.to(dev), fr.tie.to(dev)

    # ---- outcome (every row still in play had a cell step: pose_only is no longer produced)
    pending = code < 0
    mark(pending & ~rec['on_sheet'], 'off_sheet')
    mark(pending & (rec['n_auto'] > 1), 'special_metric')
    mark(pending, 'canonical')
    done = _isin(code, _DONE)
    keep = ~done
    if bool(keep.any()):                         # rows not canonicalised: empty records
        rec['N'][keep] = torch.eye(3, dtype=torch.long, device=dev)
        rec['s'][keep] = 0.0
        rec['n_auto'][keep] = -1
        rec['n_tied'][keep] = -1
        rec['violation'][keep] = float('nan')
        rec['cell_snap'][keep] = float('nan')
        rec['on_sheet'][keep] = False
        rec['sheet_moved'][keep] = False
        rec['element'][keep] = -1
        rec['lattice'][keep] = 0
        rec['free_shift'][keep] = 0.0
        rec['slack'][keep] = float('nan')
        rec['tie'][keep] = -1
        cell_face[keep] = float('nan')
    # the canonical first unit goes back into the full rows of canonicalised rows; every other row is untouched
    out_cell, out_t, out_R, out_h = d.cell.clone(), d.t.clone(), d.R.clone(), d.h.clone()
    out_cell[done] = cell[done]
    out_t[done, 0], out_R[done, 0], out_h[done, 0] = t[done, 0], R[done, 0], h[done, 0]
    canon = d.replace(cell=out_cell, t=out_t, R=out_R, h=out_h)

    near = torch.zeros(B, dtype=torch.int8, device=dev)
    near_cell = (cell_face <= near_tol) | (rec['n_tied'] > 1)       # NaN (no faces) and -1 (no cell step) count out
    near |= (done & near_cell).to(torch.int8) * NEAR_CELL
    near |= (done & (rec['slack'] <= near_tol)).to(torch.int8) * NEAR_CENTROID
    near |= (done & (rec['tie'] == 3)).to(torch.int8) * NEAR_FOLD_BAND
    moved = torch.zeros(B, dtype=torch.bool, device=dev)
    if bool(done.any()):
        moved[done] = _differs(canon[done], d[done], moved_tol)
    nudged = torch.zeros(B, dtype=torch.bool, device=dev)
    nudge_cell = torch.zeros(B, dtype=F64, device=dev)
    nudge_t = torch.zeros(B, dtype=F64, device=dev)
    nudge_message = [''] * B
    if nudge is not None:
        def recanon(y):
            return canonicalize(y, sheet=sheet, tol=tol, cell_tol=cell_tol, angle_tol=angle_tol, face_tol=face_tol,
                                near_tol=near_tol, rot_tol=rot_tol, metric_tol=metric_tol)
        canon, near, moved, nudged, nudge_cell, nudge_t, nudge_message = _nudge_rows(
            canon, d, code, near, moved, rec['tie'], float(nudge), moved_tol, tol, recanon)
    return Canonicalization(desc=canon, status=code, message=message, near_face=near, moved=moved, nudged=nudged,
                            nudge_cell=nudge_cell, nudge_t=nudge_t, nudge_message=nudge_message, **rec)


# ---------------------------------------------------------------------------
# the status check
# ---------------------------------------------------------------------------

def fd_status(desc: CrystalDescription, *, sheet: Optional[int] = 1, tol: Optional[float] = None,
              fold_tol: float = 1e-9, **canon_kwargs) -> FDStatus:
    """Whether each row is in the fundamental domain, computed (``canonicalize`` plus a comparison), never stored.

    Parameters
    ----------
    sheet : int or None
        The fundamental domain's sheet (+1). None accepts either handedness, so both descriptions of a non-Sohncke
        crystal (one per sheet) pass: ``is_fd`` then no longer picks one description per crystal.
    tol : float, optional
        Comparison tolerance (cell lengths relative, angles in rad, t fractional, R entries). Default 1e-9 for float64
        descriptions and 1e-5 for float32 ones (their rounding; a float32 row within that of a face can canonicalise
        to its partner, see ``near_face``). Values stored in float32 but passed as float64 need it set: monoclinic
        alpha and gamma are about 4e-8 rad off 90 degrees in float32.
    fold_tol : float
        ``domains.fold``'s face tolerance.
    canon_kwargs
        Further ``canonicalize`` arguments (``cell_tol``, ``angle_tol``, ``face_tol``, ``near_tol``, ``rot_tol``,
        ``metric_tol``, ``nudge``). With ``nudge`` the canonical form compared against is the nudged one, so a row
        still on a face that the nudge would move is not ``is_fd``.

    Cost: one ``canonicalize`` and one more fold. ``reduce_cell`` dominates; on 2 CPU threads (2026-10-20) the whole
    canonical CSD set (338,770 rows) took 70 s, and re-describing P-1 rows in random bases about 1,100 rows/s.
    """
    tol = _dtype_tol(desc) if tol is None else tol
    _check_tol(tol=tol, fold_tol=fold_tol)
    c = canonicalize(desc, sheet=sheet, tol=fold_tol, moved_tol=tol, **canon_kwargs)
    d = desc.double()
    B, dev = d.num_rows, d.device
    zp1 = d.z_prime == 1
    done = c.done
    decided = done & zp1 & (c.status.long() != _CODE['pose_only'])
    same = torch.zeros(B, dtype=torch.bool, device=dev)
    cell_same = torch.zeros(B, dtype=torch.bool, device=dev)
    if bool(done.any()):
        a, b = c.desc[done], d[done]
        same[done] = ~_differs(a, b, tol)
        rel = ((a.cell[:, :3] - b.cell[:, :3]).abs() / b.cell[:, :3].abs()).amax(1)
        cell_same[done] = (rel <= tol) & ((a.cell[:, 3:] - b.cell[:, 3:]).abs().amax(1) <= tol)
    # the pose as given against its own fold (in the row's own basis), for every row fold can take
    pose_folded = torch.zeros(B, dtype=torch.bool, device=dev)
    takes = ~_isin(c.status, (_CODE['bad_values'], _CODE['unreal_cell'], _CODE['z_prime_gt1']))
    rows = torch.nonzero(takes)[:, 0]
    x = _unit0(d)
    parts, _ = _isolate(lambda y: domains.fold(y, tol=fold_tol), x[rows])
    for idx, fr in parts:
        g = rows[idx]
        pose_folded[g] = ((fr.desc.t[:, 0] - d.t[g, 0]).abs().amax(1) <= tol) & \
            ((fr.desc.R[:, 0] - d.R[g, 0]).abs().amax(-1).amax(-1) <= tol)
    on_sheet = zp1 & ((d.h[:, 0] == sheet) if sheet is not None else torch.ones(B, dtype=torch.bool, device=dev))
    special = c.n_auto > 1
    is_fd = decided & ~special & on_sheet & same
    return FDStatus(is_fd=is_fd, decided=decided, cell_reduced=decided & cell_same, pose_folded=pose_folded,
                    on_sheet=on_sheet, special_metric=special, near_face=c.near_face != 0, zp1=zp1, status=c.status)

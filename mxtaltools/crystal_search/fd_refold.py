"""Re-describe Z' = 1 crystal-search rows as the same crystal in crystal_fd's canonical form, read off the lattice
MXtalTools actually builds. Used by ``crystal_opt_utils.gradient_descent_optimization(fd_refold=True)`` (wrap mode):
during the walk for rows whose lattice is in trouble (``refold_triggers``), and at the end of a stage for every
returned row (``fold_batch``, through ``crystal_opt_utils.fd_fold_aunit``).

Why (2026-10-26 fd_stress evidence, full campaign stages from FD starts). ``clean_cell_parameters`` bounds each cell
angle to [0.2 pi, 0.8 pi] by a hardtanh, so a walker that wants to shear further gets zero gradient and sticks on the
clamp, although the same lattice has a reduced description well inside the range. And a triclinic walker can jump
across F = 0 (F below) in one rprop step, after which the stated angles describe no lattice at all.

The built lattice (``built_lattice``). ``batch_compute_fractional_transform`` builds T_fc from the stated lengths and
angles with the signed volume sign(F) abc sqrt|F|, F = 1 - cos^2 alpha - cos^2 beta - cos^2 gamma
+ 2 cos alpha cos beta cos gamma (= (V / abc)^2 for a real cell). T_fc is upper triangular: a along x, b in the xy
plane with T_11 = b sin gamma > 0, and T_22 = vol / (ab sin gamma), of the sign of F. For F > 0 it is the
upper-triangular, positive-diagonal factor of the stated metric, so the stated parameters are the lattice's. For
F < 0 no lattice has the stated angles, but T_fc is still invertible (det T_fc = vol < 0): the builder makes a REAL
crystal on a LEFT-HANDED basis (a, b, c_built), whose c length and alpha, beta are not the stated ones (row 18 of the
P1 sweep: stated c 10.37 A, alpha 36.0, beta 64.6 deg; built 11.93 A, 45.3, 68.1 deg). Its energy is that real
crystal's; only the label is wrong (crystal_fd refuses it as ``unreal_cell``).

Same crystal under the basis flip. The live builder (``MolCrystalOps``, as ``canonicalize_aunit`` documents) places
the image of the asymmetric unit under the fractional operation (W, w) of G at centre T (W f + w) and pose
T W T^-1 P, P = R(rotvec) diag(h, 1, 1), for any invertible T; the crystal is that set modulo the lattice T Z^3.
With D = diag(1, 1, -1) (D^2 = I) put T' = T D, f' = D f, W' = D W D, w' = D w, n' = D n. Then
T' (W' f' + w' + n') = T (W f + w + n) and T' W' T'^-1 = T W T^-1, so (T', f', P, {(W', w')}) builds the same atoms,
with the Cartesian pose P unchanged. Where det T < 0 the flip is applied; it is needed only in triclinic groups (a
monoclinic cell has F = sin^2 beta > 0, orthorhombic and higher F > 0 fixed, and MXtalTools builds the R groups on
hexagonal axes), where {(D W D, D w)} = {(I, 0)} or {(I, 0), (-I, 0)} is G itself; a flipped row of another group is
refused (``degenerate_lattice``). T' = (a, b, -c_built) is upper triangular with a positive diagonal (T'_22 = -T_22
> 0), so it IS the matrix ``batch_compute_fractional_transform`` builds from the lengths and angles of its own metric
T'^T T' (the upper-triangular, positive-diagonal factor of a positive definite metric is unique): writing those
lengths and angles with f' and the unchanged rotation vector and handedness re-describes the built crystal exactly,
in the same Cartesian frame. Where det T > 0 the same reading returns the stated parameters (to rounding).

Canonical form. The re-read description goes to ``crystal_fd.canon.canonicalize`` (the cell step: tri_niggli
reduction for sg 1-2, the monoclinic reduced basis for 3-15, the exact-metric snap and axis order above; then
``to_sheet`` when ``sheet`` is given, and the pose fold), which changes only the description of a crystal (gated in
tests/fd), and centres are wrapped into [0, 1) by lattice translation (the builder clips a centre outside
[0, 1 - 1e-4] instead of wrapping it; canonical t lies outside [0, 1) in 84 groups). Rotation vectors are crystal_fd's
(+z hemisphere, ``CrystalDescription.crystal_params``). A canonical cell lies well inside the optimiser's ranges in
the families that trigger a refold: tri_niggli angles in [60, 120] deg with (V / abc)^2 >= 1/2 (Minkowski's bound in 3D,
attained by the face-centred cubic primitive cell), monoclinic beta in [90, 135] deg (``ops.mono_bound``).

A result is used only if writing it into the optimiser's raw parameters leaves the next ``clean_cell_parameters``
nothing to change (``in_clean_ranges``: every length at or above ``length_pad``, every angle inside the clamp, every
rotation-vector norm inside [0.001 pi, 1.999 pi], each within ``CLEAN_TOL``); otherwise the cleaned row would be
another crystal, so the row is left as it was (``outside_clean_range``; e.g. a reduced cell with a lattice vector
shorter than 3 A). A wrapped canonical centre that falls in the builder's clip band (1 - 1e-4, 1) is moved by the
builder by at most 1e-4 fractional, as every centre there is.

Only Z' = 1. Rows whose stored symmetry operators are not ``SYM_OPS[sg]`` as a set are refused
(``nonstandard_setting``): crystal_fd works in the standard setting.
"""
import inspect
import math
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch

F64 = torch.float64

# ---------------------------------------------------------------------------------------------------------------
# WORKING ASSUMPTIONS (2026-10-26, scope: gradient_descent_optimization(fd_refold=True); revisit when a run shows
# refolds on rows that were not stuck, or rows still sticking on the clamp or crossing F = 0 between refolds)
# ---------------------------------------------------------------------------------------------------------------
#: a row is refolded when a cleaned cell angle lies within this of the clamp bound pi/2 +- (pi/2) angle_pad
#: (default bound 0.2 pi / 0.8 pi, so the trigger is <= 0.225 pi = 40.5 deg or >= 0.775 pi = 139.5 deg). Only free
#: angles can get there (enforce_crystal_system fixes the others at 90 or 120 deg). Below the 9 deg between the
#: largest canonical monoclinic beta (135 deg) and the clamp (144 deg), so a refolded row never re-triggers at once.
REFOLD_ANGLE_MARGIN = 0.025 * math.pi
#: ... or when F = (V / abc)^2 of the cleaned angles is at or below this, F <= 0 included (the builder then builds the
#: left-handed lattice described above). A canonical triclinic cell has F >= 1/2 and a monoclinic cell inside the
#: clamp F = sin^2 beta >= sin^2(0.8 pi) = 0.345, so neither re-triggers at once.
REFOLD_MIN_VOLUME_FACTOR = 0.1
#: per row and stage, at most this many refold attempts (applied or refused): the bound on re-descriptions; after it
#: the row walks as without the option
REFOLD_MAX_ATTEMPTS = 8

#: clean_cell_parameters bounds each rotation-vector norm to pi +- 0.999 pi (hard-coded there, not an argument)
ROTVEC_NORM_SPAN = 0.999 * math.pi
#: tolerance of ``in_clean_ranges``: a value at most this past a bound is moved by clean by at most this
CLEAN_TOL = 1e-6

#: per-row outcome codes, by index
REFOLD_STATUS = (
    'applied',              # the canonical re-description of the built crystal is in params
    'fd_refused',           # crystal_fd did not canonicalise the re-read row (fd_status says why)
    'nonstandard_setting',  # the row's symmetry operators are not SYM_OPS[sg] as a set
    'degenerate_lattice',   # the built T_fc is singular or not finite, or left-handed outside sg 1-2
    'outside_clean_range',  # the canonical form would be changed by clean_cell_parameters (another crystal)
)
APPLIED = 0
_CODE = {name: i for i, name in enumerate(REFOLD_STATUS)}


def clean_defaults():
    """(length_pad, angle_pad): ``MolCrystalData.clean_cell_parameters``' defaults, which the optimiser's per-step
    clean uses (read from its signature, so they follow that method)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    p = inspect.signature(MolCrystalData.clean_cell_parameters).parameters
    return float(p['length_pad'].default), float(p['angle_pad'].default)


def volume_factor(cell_angles: torch.Tensor) -> torch.Tensor:
    """[B] F = 1 - cos^2 alpha - cos^2 beta - cos^2 gamma + 2 cos alpha cos beta cos gamma (float64): (V / abc)^2 for a
    real cell, <= 0 for an angle triple no lattice has."""
    c = torch.cos(cell_angles.to(F64))
    return 1 - (c ** 2).sum(-1) + 2 * c.prod(-1)


def refold_triggers(cell_angles: torch.Tensor, angle_pad: Optional[float] = None,
                    margin: Optional[float] = None, min_volume_factor: Optional[float] = None) -> torch.Tensor:
    """[B] bool: a cleaned cell angle within ``margin`` (default REFOLD_ANGLE_MARGIN) of the clamp bound
    pi/2 +- (pi/2) ``angle_pad`` (default clean's), or F at or below ``min_volume_factor`` (default
    REFOLD_MIN_VOLUME_FACTOR). Working assumptions: module constants."""
    angle_pad = clean_defaults()[1] if angle_pad is None else float(angle_pad)
    margin = REFOLD_ANGLE_MARGIN if margin is None else float(margin)
    min_vf = REFOLD_MIN_VOLUME_FACTOR if min_volume_factor is None else float(min_volume_factor)
    A = cell_angles.detach().to(F64)
    near_clamp = ((A - math.pi / 2).abs() >= math.pi / 2 * angle_pad - margin).any(-1)
    return near_clamp | (volume_factor(A) <= min_vf)


def built_lattice(cell_lengths: torch.Tensor, cell_angles: torch.Tensor, aunit_centroid: torch.Tensor):
    """The lattice the live builder makes from these Z' = 1 parameters, read off its T_fc (module docstring).

    Returns float64 (lengths [B, 3], angles [B, 3], centroid [B, 3], flipped [B] bool, ok [B] bool): where T_fc's
    basis is left-handed (det < 0, i.e. F < 0) the c column and the centre's z are negated (``flipped``); lengths and
    angles are those of the (flipped) basis' metric, so ``batch_compute_fractional_transform`` of them is that basis.
    ``ok`` is False where T_fc is singular or not finite (no lattice to read)."""
    from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
    L, A, f = cell_lengths.detach().to(F64), cell_angles.detach().to(F64), aunit_centroid.detach().to(F64)
    T = batch_compute_fractional_transform(L, A)[0]
    det = torch.linalg.det(T)
    ok = torch.isfinite(T).all(-1).all(-1) & torch.isfinite(det) & (det.abs() > 1e-12 * L.prod(-1).abs())
    flipped = ok & (det < 0)
    D = torch.tensor([1.0, 1.0, -1.0], dtype=F64, device=T.device)
    T = torch.where(flipped[:, None, None], T * D, T)  # negate column c
    f = torch.where(flipped[:, None], f * D, f)
    G = T.transpose(1, 2) @ T
    lengths = torch.diagonal(G, dim1=1, dim2=2).clamp(min=0).sqrt()
    safe = torch.where(ok[:, None], lengths, torch.ones_like(lengths))

    def angle(i, j, k):
        return torch.arccos((G[:, i, j] / (safe[:, i] * safe[:, j])).clamp(-1, 1))
    angles = torch.stack([angle(1, 2, 0), angle(0, 2, 1), angle(0, 1, 2)], -1)
    return lengths, angles, f, flipped, ok


def in_clean_ranges(lengths: torch.Tensor, angles: torch.Tensor, rotvec: torch.Tensor,
                    length_pad: float, angle_pad: float, tol: float = CLEAN_TOL) -> torch.Tensor:
    """[B] bool: ``clean_cell_parameters(mode='hard')`` at these pads would move no value by more than ``tol``
    (lengths >= length_pad, |angle - pi/2| <= (pi/2) angle_pad, rotation-vector norm within pi +- ROTVEC_NORM_SPAN)."""
    L, A = lengths.to(F64), angles.to(F64)
    norm = rotvec.to(F64).norm(dim=-1)
    return ((L >= length_pad - tol).all(-1)
            & ((A - math.pi / 2).abs() <= math.pi / 2 * angle_pad + tol).all(-1)
            & ((norm - math.pi).abs() <= ROTVEC_NORM_SPAN + tol)
            & torch.isfinite(L).all(-1) & torch.isfinite(A).all(-1) & torch.isfinite(norm))


def wrap_unit(t: torch.Tensor) -> torch.Tensor:
    """Fractional centres lattice-translated into [0, 1) (a value that rounds to 1 after the floor goes to 0)."""
    w = t - torch.floor(t)
    return torch.where(w >= 1, w - 1, w)


def standard_setting(sg: Sequence[int], symmetry_operators) -> torch.Tensor:
    """[B] bool: row i's operators are ``SYM_OPS[sg[i]]`` as a set (``crystal_fd.io_mxt.operator_set_match``); True
    for every row when ``symmetry_operators`` is None."""
    if symmetry_operators is None:
        return torch.ones(len(sg), dtype=torch.bool)
    from mxtaltools.crystal_fd.io_mxt import operator_set_match
    return torch.tensor([bool(operator_set_match(np.asarray(o), int(s))[0]) for s, o in zip(sg, symmetry_operators)],
                        dtype=torch.bool)


class RefoldResult(NamedTuple):
    """Per row (B rows, float64 on the CPU).

    params : [B, 12] lengths, angles, centroid (in [0, 1)), rotation vector (+z hemisphere) of the canonical
        re-description where ``status`` is ``applied``; the input row (float64) elsewhere.
    handedness : [B] float64, the canonical handedness where applied (``sheet=None`` keeps it), the input elsewhere.
    status : [B] int8 index into REFOLD_STATUS.
    fd_status : [B] int8 index into ``crystal_fd.canon.STATUS``, -1 where canonicalize was not run.
    flipped : [B] bool, the built basis was left-handed (F < 0) and was read with c negated.
    """
    params: torch.Tensor
    handedness: torch.Tensor
    status: torch.Tensor
    fd_status: torch.Tensor
    flipped: torch.Tensor

    @property
    def applied(self) -> torch.Tensor:
        return self.status == APPLIED


def refold_params(sg, cell_lengths, cell_angles, aunit_centroid, aunit_orientation, aunit_handedness,
                  symmetry_operators=None, sheet: Optional[int] = 1, length_pad: Optional[float] = None,
                  angle_pad: Optional[float] = None) -> RefoldResult:
    """The canonical re-description of the crystal the live builder makes from each Z' = 1 row (module docstring).

    Parameters
    ----------
    sg : [B] space-group numbers (standard setting ``SYM_OPS[sg]``).
    cell_lengths, cell_angles, aunit_centroid, aunit_orientation : [B, 3] each, the values the builder uses (cleaned;
        centres as built: wrapped into the cell and clipped).
    aunit_handedness : [B] or [B, 1], +-1.
    symmetry_operators : optional per-row operator arrays; rows not in the standard setting are refused.
    sheet : crystal_fd ``canonicalize``'s sheet: +1 (the fundamental domain's; may change h through an improper
        element of G, the same crystal) or None (keep each row's handedness).
    length_pad, angle_pad : the clean ranges results must lie in (default clean_cell_parameters' defaults).
    """
    from mxtaltools.crystal_fd import canon
    from mxtaltools.crystal_fd.description import CrystalDescription
    from mxtaltools.crystal_fd.ops import real_cells
    dl, da = clean_defaults()
    length_pad = dl if length_pad is None else float(length_pad)
    angle_pad = da if angle_pad is None else float(angle_pad)
    sg = torch.as_tensor(sg).reshape(-1).long().cpu()
    B = len(sg)
    L0 = torch.as_tensor(cell_lengths).detach().to(F64).cpu().reshape(B, 3)
    A0 = torch.as_tensor(cell_angles).detach().to(F64).cpu().reshape(B, 3)
    c0 = torch.as_tensor(aunit_centroid).detach().to(F64).cpu().reshape(B, 3)
    r0 = torch.as_tensor(aunit_orientation).detach().to(F64).cpu().reshape(B, 3)
    h0 = torch.as_tensor(aunit_handedness).detach().to(F64).cpu().reshape(B)
    params = torch.cat([L0, A0, c0, r0], 1)
    hand = h0.clone()
    status = torch.full((B,), APPLIED, dtype=torch.int8)
    fd_status = torch.full((B,), -1, dtype=torch.int8)
    if B == 0:
        return RefoldResult(params, hand, status, fd_status, torch.zeros(0, dtype=torch.bool))

    status[~standard_setting(sg.tolist(), symmetry_operators)] = _CODE['nonstandard_setting']
    L1, A1, f1, flipped, ok = built_lattice(L0, A0, c0)
    bad_lattice = ~ok | (flipped & (sg > 2)) | ~real_cells(torch.cat([L1, A1], 1).nan_to_num(0.0))
    status[(status == APPLIED) & bad_lattice] = _CODE['degenerate_lattice']

    rows = torch.nonzero(status == APPLIED).flatten()
    if len(rows):
        desc = CrystalDescription.from_crystal_params(sg[rows], L1[rows], A1[rows], f1[rows], r0[rows], h0[rows])
        c = canon.canonicalize(desc, sheet=sheet)
        fd_status[rows] = c.status.to(torch.int8)
        L, A, t, rv, h = c.desc.crystal_params()
        t = wrap_unit(t)
        good = c.done & in_clean_ranges(L, A, rv, length_pad, angle_pad)
        status[rows[~c.done]] = _CODE['fd_refused']
        status[rows[c.done & ~good]] = _CODE['outside_clean_range']
        g = rows[good]
        params[g] = torch.cat([L, A, t, rv], 1)[good]
        hand[g] = h.reshape(-1).to(F64)[good]
    return RefoldResult(params, hand, status, fd_status, flipped)


def check_zp1(crystal_batch):
    """Raise ValueError unless every row of the batch is Z' = 1 at max_z_prime 1."""
    zp = crystal_batch.z_prime.reshape(-1)
    if int(crystal_batch.max_z_prime) != 1 or bool((zp != 1).any()):
        raise ValueError(f"fd_refold handles Z' = 1 batches only (max_z_prime {int(crystal_batch.max_z_prime)}, "
                         f"z_prime {sorted(set(zp.tolist()))})")


def batch_refold(crystal_batch, rows: Optional[torch.Tensor] = None, sheet: Optional[int] = 1) -> RefoldResult:
    """``refold_params`` of rows ``rows`` (default all) of a Z' = 1 MolCrystalData batch, from the values the builder
    uses: cleaned cell, ``aunit_centroid`` as assigned (the caller wraps it), orientation and handedness."""
    check_zp1(crystal_batch)
    n = crystal_batch.num_graphs
    rows = torch.arange(n) if rows is None else torch.as_tensor(rows).reshape(-1).long().cpu()
    ops = [crystal_batch.symmetry_operators[i] for i in rows.tolist()]
    return refold_params(crystal_batch.sg_ind.reshape(-1).cpu()[rows],
                         crystal_batch.cell_lengths.detach().cpu()[rows],
                         crystal_batch.cell_angles.detach().cpu()[rows],
                         crystal_batch.aunit_centroid.detach().cpu()[rows],
                         crystal_batch.aunit_orientation.detach().cpu()[rows],
                         crystal_batch.aunit_handedness.detach().cpu().reshape(n, -1)[rows, 0],
                         symmetry_operators=ops, sheet=sheet)


def fold_batch(crystal_batch, sheet: Optional[int] = 1) -> torch.Tensor:
    """In place: every Z' = 1 row re-described by its crystal_fd canonical form, read off the built lattice; returns
    [num_graphs] int8 REFOLD_STATUS codes.

    Centres are wrapped into [0, 1) first (lattice translation), then ``batch_refold`` (sheet +1 by default: rotation
    vectors on the +z hemisphere, handedness possibly changed through an improper element of G). Rows it does not
    apply keep their cell, their wrapped centre and their rotation, with the rotation vector moved to the +z
    hemisphere (``canonicalize_orientation``); a ``flipped`` row among them keeps its stated (unreal) angles. Never
    raises for a row. Replaces cell_lengths, cell_angles, aunit_centroid, aunit_orientation and aunit_handedness with
    new tensors (dtype and device kept) and reruns ``box_analysis``."""
    check_zp1(crystal_batch)
    crystal_batch.aunit_centroid = wrap_unit(crystal_batch.aunit_centroid.detach().clone())
    res = batch_refold(crystal_batch, sheet=sheet)
    crystal_batch.aunit_orientation = crystal_batch.aunit_orientation.detach().clone()
    crystal_batch.canonicalize_orientation()  # the fallback rows' +z hemisphere; applied rows are overwritten below
    ok = res.applied.to(crystal_batch.cell_lengths.device)
    if bool(ok.any()):
        dev = crystal_batch.cell_lengths.device
        p = res.params.to(dev)

        def put(cur, new):
            out = cur.detach().clone()
            out[ok] = new[ok].to(out.dtype)
            return out
        crystal_batch.cell_lengths = put(crystal_batch.cell_lengths, p[:, 0:3])
        crystal_batch.cell_angles = put(crystal_batch.cell_angles, p[:, 3:6])
        crystal_batch.aunit_centroid = put(crystal_batch.aunit_centroid, p[:, 6:9])
        crystal_batch.aunit_orientation = put(crystal_batch.aunit_orientation, p[:, 9:12])
        h = crystal_batch.aunit_handedness
        crystal_batch.aunit_handedness = put(h, res.handedness.to(dev).reshape(h.shape))
    crystal_batch.box_analysis()
    return res.status

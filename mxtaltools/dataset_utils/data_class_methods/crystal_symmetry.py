"""Space-group detection from the atoms: ``find_symmetry(batch, ...)``, an optional function.

The owner's "general approach" to structures saved with less symmetry than they have
(``docs/design/crystal_ingestion.md`` §7a, Q5, in the GFN repository: P2_1/c saved in P1 is "a
common case"): expand each row to its full cell (P1), detect the space group from the atoms
with spglib at an explicit tolerance (heavy atoms first, then check the hydrogens),
re-express the detected group in the standard setting ``SYM_OPS[sg]``, verify by rebuilding
the cell from that description, and record agreement or disagreement with the declared group.

Not a chain step (yet): it is not mixed into ``MolCrystalData``, adds no code to the ingestion
status table and no ``done`` bit, and is in no recipe. It takes a batch from
``MolCrystalData.read`` / ``from_arrays``, or a stored or later-stage batch (whose ``pos``
holds atoms the operators expand into the cell: the deposited set, unwrapped units, or the
posed asymmetric unit), and returns a separate per-row record; the batch is never modified.
Every row gets a status (``SYMMETRY_STATUS``) and a message, never an exception; bad
arguments raise.

Conventions
-----------
- A setting change is ``x_in = P x_std + p`` (``crystal_fd.settings.SettingTransform``): the
  columns of P are the standard basis vectors in the input basis, p the standard origin in
  input coordinates; spglib's ``transformation_matrix`` and ``origin_shift`` are the inverse
  (``x_std = P_spg x_in + p_spg``).
- "Operators per cell" count the operators of a group modulo the lattice of the INPUT cell,
  pure translations included: a P1 row in a cell twice the true one has 2 detected operators.
- Distances are Angstrom, under periodic images (the nearest image found by rounding
  fractional differences; exact for distances small against the cell, an upper bound beyond).
"""
import warnings
from typing import Optional

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_fd.ops import cell_from_metric_tensor
from mxtaltools.crystal_fd.settings import find_setting, transform_ops

__all__ = ['find_symmetry', 'SYMMETRY_STATUS', 'RELATIONS']

#: Per-row status codes of ``find_symmetry`` (the record's ``status`` holds the index), in the
#: order the checks run; the first problem names the row. Codes are only ever appended.
SYMMETRY_STATUS = ('ok', 'not_run', 'no_atoms', 'superposed_atoms', 'no_heavy_atoms', 'detection_failed',
                   'setting_failed', 'rebuild_mismatch', 'hydrogen_mismatch', 'find_symmetry_failed')
_STATUS = {name: i for i, name in enumerate(SYMMETRY_STATUS)}

#: The detected group against the declared one (the record's ``relation`` holds the index, -1
#: where not determined), as groups of operators on the input cell (module docstring):
#: ``equal`` the same group; ``higher`` the declared group is a proper subgroup of the detected
#: one (the file carries less symmetry than its atoms); ``lower`` the detected group is a proper
#: subgroup of the declared one; ``different`` neither contains the other.
RELATIONS = ('equal', 'higher', 'lower', 'different')
_RELATION = {name: i for i, name in enumerate(RELATIONS)}

#: Rational entries of a setting matrix are multiples of 1/48 (``crystal_fd.settings`` snaps the
#: same way); an entry within ``_SNAP_TOL`` of one is set to it, removing float noise.
_SNAP_DENOMINATOR = 48
_SNAP_TOL = 1e-6
#: Operator comparisons on exact (file or SYM_OPS) operators: rotation entries and fractional
#: translations equal within this.
_OP_TOL = 1e-6
#: Nearest candidates examined per atom in the fractional k-d tree before taking the Cartesian
#: nearest (the fractional and Cartesian orders agree at small distances).
_NEIGHBOURS = 8

_F64 = torch.float64
_STD_OPS = {}


class _Refusal(Exception):
    """Internal: one row stops at a check; carries the status name, a message and what was found so far."""

    def __init__(self, code: str, message: str, info: dict):
        assert code in _STATUS, code
        self.code = code
        self.info = info
        super().__init__(message)


def find_symmetry(batch, *, symprec: float, angle_tolerance: float, rebuild_tol: float, h_tol: float,
                  merge_tol: float, rows=None) -> dict:
    """Detect each row's space group from its atoms and compare it with the declared one.

    Per row: (1) expand the row's atoms by its own operators to the full cell, merging images
    of one element that coincide (an atom on a special position of the declared group); (2) run
    spglib on the heavy atoms (Z > 1) of that cell at ``symprec``; (3) find the change of
    setting to ``SYM_OPS[sg]`` (spglib's at the Hall number whose operators are ``SYM_OPS[sg]``,
    ``crystal_ops.spglib_hall_number``; else ``crystal_fd.settings.find_setting`` on the
    detected operators) and the standard cell (metric ``P^T G P``); (4) rebuild: carry
    ``SYM_OPS[sg]`` into the input coordinates by that setting change
    (``crystal_fd.settings.transform_ops``) and send every atom of the cell through every
    operator, measuring how far its image lies from the nearest atom of its element; the
    heavy atoms must match within ``rebuild_tol`` and the hydrogens within ``h_tol``, each
    operator sending distinct atoms to distinct atoms. Then the standard-setting description
    (standard cell, one atom per orbit, ``SYM_OPS[sg]``) rebuilds the input cell within those
    residuals. (5) Compare the operators of step (4), the group the rebuild checks, with the
    declared ones (``RELATIONS``). spglib's own operator list can be a proper subgroup of the
    group it names (seen at loose ``symprec``: a list of pure translations named Cc); the
    record then follows ``SYM_OPS[sg]`` in the input cell, and the message gives spglib's count.

    Parameters
    ----------
    batch : MolCrystalDataBatch
        A batch holding ``ptr``, ``z``, ``pos``, ``cell_lengths``, ``cell_angles``, ``sg_ind``,
        ``symmetry_operators`` and ``status``: from ``read`` / ``from_arrays``, or any later
        stage (its ``pos`` expanded by its operators must give the cell; for a batch through
        ``sanitize`` the hydrogens are the ones it holds, e.g. regenerated). Not modified.
    symprec : float
        Angstrom, > 0: spglib's distance tolerance for the detection on the heavy atoms. Also
        the tolerance for matching a detected operator with a declared one: rotation parts
        equal, translations differing by at most ``symprec`` Angstrom modulo the lattice.
    angle_tolerance : float
        Degrees, > 0, spglib's angle tolerance for the lattice; or exactly -1 for spglib's own
        rule (a tolerance derived from ``symprec``: an angle change times the cell lengths it
        joins must stay below ``symprec``, about 0.006 degrees for 10 A edges at 1e-3 A). It
        decides only whether the cell's metric admits a rotation; the rebuild of step (4)
        works in fractional coordinates and does not test the metric, so a cell whose angles
        are a few thousandths of a degree off 90 can come out ``lower`` at -1 and ``equal`` at
        an explicit 0.05.
    rebuild_tol : float
        Angstrom, > 0: largest accepted distance of a heavy atom's image under a standard
        operator from the nearest heavy atom of its element.
    h_tol : float
        Angstrom, > 0: the same for hydrogens (atomic number 1), against the heavy-atom group.
    merge_tol : float
        Angstrom, > 0: images (under the declared operators) of atoms of one element closer
        than this are linked, and each connected set of linked images is one atom, kept at its
        first image: an atom on a special position of the declared group, or a molecule
        written whole on one. Linking is pair by pair, so a chain (e.g. six images on a small
        ring around a 6-fold axis) can join images further apart than ``merge_tol``;
        ``merge_span`` records the largest distance within one merged set. Images of atoms of
        different elements closer than this refuse the row (``superposed_atoms``).
    rows : None, bool mask [B] or integer indices
        The rows to run on; None: the rows whose ``status`` is 0 (``ok`` in every status
        table). Other rows get ``not_run``.

    Returns
    -------
    dict
        Per row (B rows), on the CPU:

        - ``status`` [B] int8, index into ``SYMMETRY_STATUS``; ``message`` list[B] of str
          (declared and detected group, relation, residuals; the problem where there is one);
        - ``sg_declared`` [B] long (the batch's ``sg_ind``), ``sg_detected`` [B] long (0 where
          not detected), ``hm_detected`` list[B] (spglib's international symbol, '' where
          not detected);
        - ``relation`` [B] int8, index into ``RELATIONS`` (-1 where not determined);
        - ``n_ops_declared``, ``n_ops_detected`` [B] long, operators per input cell (0 where
          not known); ``index`` [B] float64, their ratio (the subgroup index for ``higher``;
          NaN where not known). Z' in the detected group is the declared Z' divided by it (a
          fraction when a unit sits on a special position of the detected group).
          ``n_ops_detected``, ``index`` and ``relation`` describe ``SYM_OPS[sg]`` carried into
          the input cell (the operators the rebuild sends the atoms through) wherever a setting
          was found, and spglib's operator list on ``setting_failed`` rows;
        - ``P`` [B, 3, 3], ``p`` [B, 3] float64: ``x_in = P x_std + p`` (NaN where not found);
          ``std_cell`` [B, 6] float64, the standard cell (Angstrom, radians);
        - ``rebuild_dev`` [B] float64 (heavy atoms) and ``h_dev`` [B] float64 (hydrogens; 0
          for a row without hydrogens): the largest image-to-atom distance of step (4), NaN
          where not reached;
        - ``n_cell_atoms`` [B] long (atoms of the expanded input cell), ``n_asym_heavy``,
          ``n_asym_h`` [B] long (orbits of the detected group: heavy atoms and hydrogens of
          its asymmetric unit; -1 where not determined);
        - ``merge_span`` [B] float64: Angstrom, the largest distance between two images merged
          into one atom in step (1) (0 where none merged; NaN where not reached). Above
          ``merge_tol`` only through a chain; above ``symprec`` the message says so, since a
          kept image then lies off its special position by more than the detection allows
          and can turn the declared group ``lower`` by the merge alone;
        - ``sg_all_atoms`` [B] long: spglib on all atoms at ``symprec``, run only on
          ``hydrogen_mismatch`` rows (0 elsewhere);
        - ``asym_frac`` list[B] of float64 arrays [k, 3]: one atom per orbit (the first in the
          expanded cell's order, which starts with the declared first operator's images), in
          standard-setting fractional coordinates wrapped to [0, 1); ``asym_z`` list[B] of
          int64 arrays [k]; None on rows whose status is not ``ok``;
        - ``status_names`` (``SYMMETRY_STATUS``), ``relation_names`` (``RELATIONS``), ``args``
          (the tolerances as run).

    Raises
    ------
    ValueError
        A tolerance that is not a finite positive number (``angle_tolerance`` may be -1), a
        batch without the fields above, or bad ``rows``.

    Notes
    -----
    Status codes (``SYMMETRY_STATUS``), the first that applies naming the row:

    ``not_run``
        the row is not selected (by default: its status is not ``ok``).
    ``no_atoms``
        the row holds no atoms.
    ``superposed_atoms``
        images of atoms of different elements lie within ``merge_tol`` of each other (e.g.
        substitutional disorder written as two sites).
    ``no_heavy_atoms``
        the cell holds only hydrogens.
    ``detection_failed``
        spglib finds no group on the heavy atoms (its reason is kept, e.g. "too close distance
        between atoms"; the message says so where spglib gives none).
    ``setting_failed``
        no change of setting to ``SYM_OPS[sg]`` was found, or it is not right-handed.
    ``rebuild_mismatch``
        a heavy atom's image lies further than ``rebuild_tol`` from every heavy atom of its
        element, or an operator sends two heavy atoms onto one.
    ``hydrogen_mismatch``
        the heavy-atom group does not hold for the hydrogens within ``h_tol`` (or sends two
        onto one); ``sg_all_atoms`` then says what spglib finds on all atoms.
    ``find_symmetry_failed``
        processing the row raised a ``RuntimeError`` or ``ValueError`` (message kept); other
        rows are unaffected.

    ``relation``, the detected group and the setting are reported wherever they were reached,
    including on ``rebuild_mismatch`` and ``hydrogen_mismatch`` rows (then not verified, or
    verified on the heavy atoms only). On a ``rebuild_mismatch`` row they describe the group
    the rebuild refused, ``SYM_OPS[sg]``, which can be larger than spglib's own list.
    """
    symprec = _positive('symprec', symprec)
    rebuild_tol = _positive('rebuild_tol', rebuild_tol)
    h_tol = _positive('h_tol', h_tol)
    merge_tol = _positive('merge_tol', merge_tol)
    if isinstance(angle_tolerance, bool) or not isinstance(angle_tolerance, (int, float, np.integer, np.floating)) \
            or not np.isfinite(angle_tolerance) or not (angle_tolerance > 0 or angle_tolerance == -1):
        raise ValueError(f'angle_tolerance must be a finite number > 0 (degrees) or -1 (spglib\'s rule), '
                         f'got {angle_tolerance!r}')
    angle_tolerance = float(angle_tolerance)
    missing = [k for k in ('ptr', 'z', 'pos', 'cell_lengths', 'cell_angles', 'sg_ind', 'symmetry_operators', 'status')
               if k not in batch._store]
    if missing:
        raise ValueError(f'find_symmetry needs a batch with {missing}')

    ptr = batch.ptr.detach().to('cpu').long()
    B = len(ptr) - 1
    run = _selected_rows(batch, rows, B)
    lengths = batch.cell_lengths.detach().to('cpu', _F64)
    angles = batch.cell_angles.detach().to('cpu', _F64)
    T_fc_all, T_cf_all, _ = batch_compute_fractional_transform(lengths, angles)
    z_all = batch.z.detach().to('cpu').long().numpy()
    pos_all = batch.pos.detach().to('cpu', _F64).numpy()
    sg_decl = batch.sg_ind.detach().to('cpu').long()
    args = dict(symprec=symprec, angle_tolerance=angle_tolerance, rebuild_tol=rebuild_tol, h_tol=h_tol,
                merge_tol=merge_tol)

    out = _empty_record(B, sg_decl)
    out['args'] = args
    for b in range(B):
        if not run[b]:
            out['status'][b] = _STATUS['not_run']
            out['message'][b] = 'not selected' if rows is not None else 'row status is not ok'
            continue
        s, e = int(ptr[b]), int(ptr[b + 1])
        T_fc, T_cf = T_fc_all[b].numpy(), T_cf_all[b].numpy()
        frac = pos_all[s:e] @ T_cf.T
        ops = batch.symmetry_operators[b]
        ops = np.asarray(ops.detach().cpu().numpy() if torch.is_tensor(ops) else ops, dtype=np.float64)
        try:
            res = _symmetry_row(z_all[s:e], frac, T_fc, ops, int(sg_decl[b]), **args)
            code, message = 'ok', res.pop('message')
        except _Refusal as r:
            res, code, message = r.info, r.code, str(r)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as err:
            res, code, message = {}, 'find_symmetry_failed', f'{type(err).__name__}: {str(err)[:300]}'
        out['status'][b] = _STATUS[code]
        out['message'][b] = message
        for key, value in res.items():
            if isinstance(out[key], list):
                out[key][b] = value
            else:
                out[key][b] = torch.as_tensor(value, dtype=out[key].dtype)
    return out


###############################################################################
# helpers
###############################################################################

def _positive(name, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite number > 0, got {value!r}')
    return float(value)


def _selected_rows(batch, rows, B) -> np.ndarray:
    """[B] bool: the rows to run on."""
    if rows is None:
        return (batch.status.detach().to('cpu') == 0).numpy()
    t = torch.as_tensor(rows).detach().to('cpu').reshape(-1)
    if t.dtype == torch.bool:
        if len(t) != B:
            raise ValueError(f'a row mask needs {B} entries, got {len(t)}')
        return t.numpy().copy()
    if t.is_floating_point() or t.is_complex():
        raise ValueError(f'rows must be a bool mask or integer indices, got {t.dtype}')
    t = t.long()
    if bool((t < 0).any()) or bool((t >= B).any()):
        raise ValueError(f'row indices must lie in [0, {B}), got {t.tolist()[:10]}')
    mask = np.zeros(B, dtype=bool)
    mask[t.numpy()] = True
    return mask


def _empty_record(B, sg_decl) -> dict:
    nan = float('nan')
    return dict(status=torch.zeros(B, dtype=torch.int8), message=[''] * B, sg_declared=sg_decl.clone(),
                sg_detected=torch.zeros(B, dtype=torch.long), hm_detected=[''] * B,
                relation=torch.full((B,), -1, dtype=torch.int8),
                n_ops_declared=torch.zeros(B, dtype=torch.long), n_ops_detected=torch.zeros(B, dtype=torch.long),
                index=torch.full((B,), nan, dtype=_F64), P=torch.full((B, 3, 3), nan, dtype=_F64),
                p=torch.full((B, 3), nan, dtype=_F64), std_cell=torch.full((B, 6), nan, dtype=_F64),
                rebuild_dev=torch.full((B,), nan, dtype=_F64), h_dev=torch.full((B,), nan, dtype=_F64),
                n_cell_atoms=torch.zeros(B, dtype=torch.long), n_asym_heavy=torch.full((B,), -1, dtype=torch.long),
                n_asym_h=torch.full((B,), -1, dtype=torch.long), merge_span=torch.full((B,), nan, dtype=_F64),
                sg_all_atoms=torch.zeros(B, dtype=torch.long),
                asym_frac=[None] * B, asym_z=[None] * B, status_names=SYMMETRY_STATUS, relation_names=RELATIONS)


def _std_ops(sg: int) -> np.ndarray:
    if sg not in _STD_OPS:
        _STD_OPS[sg] = np.stack([np.asarray(o, dtype=np.float64) for o in SYM_OPS[sg]])
    return _STD_OPS[sg]


def _snap(M: np.ndarray) -> np.ndarray:
    """Entries within ``_SNAP_TOL`` of a multiple of 1/48 set to it."""
    s = np.round(M * _SNAP_DENOMINATOR) / _SNAP_DENOMINATOR
    return np.where(np.abs(M - s) <= _SNAP_TOL, s, M)


def _wrap(x: np.ndarray) -> np.ndarray:
    """Fractional coordinates in [0, 1) (``x - floor(x)`` can round to 1.0)."""
    x = x - np.floor(x)
    x[x >= 1.0] = 0.0
    return x


def _min_image_dist(d: np.ndarray, T_fc: np.ndarray) -> np.ndarray:
    """Cartesian length of fractional differences ``d`` [..., 3] at their rounded (nearest) image."""
    d = d - np.round(d)
    return np.linalg.norm(d @ T_fc.T, axis=-1)


def _dedupe_ops(ops: np.ndarray) -> np.ndarray:
    """Operators unique modulo Z^3 (first kept): rotation parts and translations mod 1 within ``_OP_TOL``."""
    keep = []
    for i in range(len(ops)):
        if not any(np.abs(ops[i, :3, :3] - ops[j, :3, :3]).max() <= _OP_TOL
                   and np.abs((ops[i, :3, 3] - ops[j, :3, 3]) - np.round(ops[i, :3, 3] - ops[j, :3, 3])).max()
                   <= _OP_TOL for j in keep):
            keep.append(i)
    return ops[keep]


def _contains(big: np.ndarray, small: np.ndarray, T_fc: np.ndarray, tol: float) -> bool:
    """Every operator of ``small`` [k, 4, 4] matches one of ``big`` [m, 4, 4]: rotation parts equal (rounded),
    translations within ``tol`` Angstrom modulo the lattice."""
    rot_b, rot_s = np.round(big[:, :3, :3]), np.round(small[:, :3, :3])
    same_rot = (rot_s[:, None] == rot_b[None]).all((-1, -2))                  # [k, m]
    d = _min_image_dist(small[:, None, :3, 3] - big[None, :, :3, 3], T_fc)    # [k, m]
    return bool((same_rot & (d <= tol)).any(1).all())


def _relation(group: np.ndarray, decl: np.ndarray, T_fc: np.ndarray, tol: float) -> str:
    """The ``RELATIONS`` name of the operators ``group`` against the declared ``decl`` (both per input cell, unique
    modulo the lattice), operators matched by ``_contains`` at ``tol`` Angstrom."""
    decl_in_group = _contains(group, decl, T_fc, tol)
    group_in_decl = _contains(decl, group, T_fc, tol)
    if decl_in_group and len(group) == len(decl):
        return 'equal'
    if decl_in_group and len(group) > len(decl):
        return 'higher'
    if group_in_decl and len(group) < len(decl):
        return 'lower'
    return 'different'


def _spglib_dataset(lattice, frac, numbers, symprec, angle_tolerance, hall_number):
    """``(dataset, '')`` or ``(None, message)``, the message carrying spglib's reason.

    spglib 2.7 hands its reason only to an exception: under its default (old) error handling the call returns None
    and ``get_error_message`` returns '' (it never stores the reason). So the call opts into spglib's documented
    exception handling (``spglib.error.OLD_ERROR_HANDLING = False``, restored after the call; module state, so not
    thread-safe), and a ``SpglibError`` becomes the message. Where spglib still returns None without a reason (the
    environment variable ``SPGLIB_OLD_ERROR_HANDLING`` forces the old handling), the message says so."""
    import spglib
    err_mod = getattr(spglib, 'error', None)
    error_cls = getattr(err_mod, 'SpglibError', None)
    old_handling = getattr(err_mod, 'OLD_ERROR_HANDLING', None)
    with warnings.catch_warnings():
        # spglib 2.7 warns on every call under its old error handling, and get_error_message is deprecated
        warnings.simplefilter('ignore', DeprecationWarning)
        try:
            if old_handling is not None:
                err_mod.OLD_ERROR_HANDLING = False
            ds = spglib.get_symmetry_dataset((lattice, frac, numbers), symprec=symprec,
                                             angle_tolerance=angle_tolerance, hall_number=hall_number)
        except Exception as err:                  # noqa: BLE001 -- re-raised unless spglib's own
            if error_cls is not None and isinstance(err, error_cls):
                reason = ' '.join(str(err).split())
                return None, f'spglib: {reason}' if reason else 'spglib raised without a reason'
            raise
        finally:
            if old_handling is not None:
                err_mod.OLD_ERROR_HANDLING = old_handling
        if ds is None:
            get = getattr(spglib, 'get_error_message', None)
            reason = (get() or '').strip() if get is not None else ''
            if reason and reason != 'no error':
                return None, f'spglib: {reason}'
            return None, 'spglib returned no dataset and gave no reason'
    return ds, ''


###############################################################################
# one row
###############################################################################

def _symmetry_row(z, frac, T_fc, ops, sg_decl, *, symprec, angle_tolerance, rebuild_tol, h_tol, merge_tol) -> dict:
    """One row's record entries (``find_symmetry``); raises ``_Refusal`` at the first failing check."""
    info = {}
    if len(z) == 0:
        raise _Refusal('no_atoms', 'the row holds no atoms', info)

    # (1) the full cell under the declared operators
    X, Zx, site, clash, span = _expand(frac, z, ops, T_fc, merge_tol)
    info.update(n_cell_atoms=len(X), merge_span=span)
    merged = '' if span <= symprec else \
        (f'; images merged into one atom span up to {span:.2g} A (merge_tol {merge_tol:g} A), above symprec: a kept '
         f'image can lie off its special position by more than the detection allows')
    if clash is not None:
        n_pairs, a, c = clash
        raise _Refusal('superposed_atoms', f'{n_pairs} pair(s) of images of atoms of different elements lie within '
                                           f'merge_tol = {merge_tol:g} A (e.g. input atoms {a} and {c})', info)
    ops_decl = _dedupe_ops(ops)
    info['n_ops_declared'] = len(ops_decl)

    # (2) detection on the heavy atoms
    heavy = Zx > 1
    if not heavy.any():
        raise _Refusal('no_heavy_atoms', 'the cell holds only hydrogens', info)
    lattice = T_fc.T.copy()                       # rows: a, b, c (Cartesian)
    ds, err = _spglib_dataset(lattice, X[heavy], Zx[heavy], symprec, angle_tolerance, 0)
    if ds is None:
        raise _Refusal('detection_failed', f'no group found on the heavy atoms at symprec {symprec:g} A '
                                           f'({err}){merged}', info)
    sg = int(ds.number)
    ops_det = np.zeros((len(ds.rotations), 4, 4))
    ops_det[:, :3, :3] = ds.rotations
    ops_det[:, :3, 3] = ds.translations
    ops_det[:, 3, 3] = 1.0
    info.update(sg_detected=sg, hm_detected=str(ds.international))

    def compare(group, source):
        """(5) the operators ``group`` (per input cell) against the declared ones: record entries and summary."""
        relation = _relation(group, ops_decl, T_fc, symprec)
        index = len(group) / len(ops_decl)
        info.update(n_ops_detected=len(group), index=index, relation=_RELATION[relation])
        return (f"declared sg {sg_decl} ({len(ops_decl)} operators per cell) -> detected {ds.international} "
                f"(sg {sg}, {len(group)}{source}): {relation}" + (f' (index {index:g})' if relation == 'higher' else '')
                + merged)

    # (5, provisional) spglib's operator list: what a row without a setting reports
    summary = compare(ops_det, ", spglib's operator list")

    # (3) the standard setting
    P_spg, p_spg, how = _standard_setting(ds, lattice, X[heavy], Zx[heavy], ops_det, sg, symprec, angle_tolerance,
                                          T_fc)
    if P_spg is None:
        raise _Refusal('setting_failed', f'{summary}; no change of setting to SYM_OPS[{sg}] ({how})', info)
    P = _snap(np.linalg.inv(P_spg))
    p = _wrap(-(P @ p_spg))
    if not np.linalg.det(P) > 1e-9:
        raise _Refusal('setting_failed', f'{summary}; the setting change is not right-handed (det P = '
                                         f'{np.linalg.det(P):.3g})', info)
    G = T_fc.T @ T_fc
    std_cell = cell_from_metric_tensor(torch.as_tensor(P.T @ G @ P, dtype=_F64)[None])[0].numpy()
    info.update(P=P, p=p, std_cell=std_cell)

    # (4) the rebuild: SYM_OPS[sg] carried into the input coordinates, applied to every atom; (5) its relation to the
    # declared group (the counts, index and relation of the record are those of the operators the rebuild checks)
    ops_in = transform_ops(_std_ops(sg), P_spg, p_spg, tol=_OP_TOL)
    summary = compare(ops_in, '')
    if len(ops_in) != len(ops_det):
        summary += (f"; spglib's operator list has {len(ops_det)} operators per input cell, SYM_OPS[{sg}] carried "
                    f"into it {len(ops_in)}: the counts and relation are SYM_OPS[{sg}]'s")
    dev, arg = _image_match(X, Zx, ops_in, T_fc)
    hyd = ~heavy
    rebuild_dev = float(dev[:, heavy].max())
    h_dev = float(dev[:, hyd].max()) if hyd.any() else 0.0
    info.update(rebuild_dev=rebuild_dev, h_dev=h_dev)
    summary += f'; rebuild {rebuild_dev:.2g} A (heavy), {h_dev:.2g} A (H)'
    if rebuild_dev > rebuild_tol:
        raise _Refusal('rebuild_mismatch', f'{summary}: heavy atoms above rebuild_tol = {rebuild_tol:g} A', info)
    if not _permutes(arg, heavy):
        raise _Refusal('rebuild_mismatch', f'{summary}: an operator sends two heavy atoms onto one', info)
    comp = _orbits(arg)
    info['n_asym_heavy'] = int(len(np.unique(comp[heavy])))
    if hyd.any() and (h_dev > h_tol or not _permutes(arg, hyd)):
        ds_all, err_all = _spglib_dataset(lattice, X, Zx, symprec, angle_tolerance, 0)
        info['sg_all_atoms'] = int(ds_all.number) if ds_all is not None else 0
        what = f'up to {h_dev:.3g} A off, above h_tol = {h_tol:g} A' if h_dev > h_tol else \
            'an operator sends two hydrogens onto one'
        raise _Refusal('hydrogen_mismatch', f'{summary}: the heavy-atom group does not hold for the hydrogens ({what}); '
                                            f'spglib on all atoms: ' +
                       (f'{ds_all.international} (sg {ds_all.number})' if ds_all is not None else err_all), info)
    info['n_asym_h'] = int(len(np.unique(comp[hyd]))) if hyd.any() else 0

    # one atom per orbit, in standard-setting coordinates
    _, first = np.unique(comp, return_index=True)
    reps = np.sort(first)
    info['asym_frac'] = _wrap(X[reps] @ P_spg.T + p_spg)
    info['asym_z'] = Zx[reps].astype(np.int64)
    info['message'] = summary + (f' [setting from {how}]' if how != 'spglib' else '')
    return info


def _expand(frac, z, ops, T_fc, merge_tol):
    """The cell under ``ops``: ``(X [n, 3] in [0, 1), Z [n], site [n], clash, span)``.

    Images of the input atoms under every operator, in operator order (atoms in input order within each). Images
    of one element within ``merge_tol`` of each other are linked pair by pair, and each connected set is one atom,
    kept at its first image: an atom on a special position of the declared group, or a molecule written whole on one
    (the file lists the atoms the site symmetry relates). A chain of links can join images further apart than
    ``merge_tol``: ``span`` is the largest distance (Angstrom) between two images of one set (0.0 when none merge).
    ``clash`` is None, or ``(pairs, input atom, input atom)`` when images of atoms of different elements lie within
    ``merge_tol``.
    """
    img = _wrap(np.einsum('kij,nj->kni', ops[:, :3, :3], frac) + ops[:, None, :3, 3]).reshape(-1, 3)   # [m n, 3]
    m, n = len(ops), len(frac)
    site = np.tile(np.arange(n), m)
    zz = z[site]
    pairs = _close_pairs(img, T_fc, merge_tol)
    clash = None
    span = 0.0
    if len(pairs):
        mixed = zz[pairs[:, 0]] != zz[pairs[:, 1]]
        if mixed.any():
            a, c = pairs[mixed][0]
            clash = (int(mixed.sum()), int(site[a]), int(site[c]))
        same = pairs[~mixed]
        graph = coo_matrix((np.ones(len(same)), (same[:, 0], same[:, 1])), shape=(m * n, m * n))
        _, labels = connected_components(graph, directed=False)
        _, first, counts = np.unique(labels, return_index=True, return_counts=True)
        keep = np.sort(first)
        span = _merged_span(img, labels, counts, T_fc)
    else:
        keep = np.arange(m * n)
    return img[keep], zz[keep], site[keep], clash, span


def _merged_span(img, labels, counts, T_fc) -> float:
    """The largest distance (Angstrom) between two images with one label, over labels held by more than one image.
    ``labels`` [N] take the values 0 .. len(counts) - 1, ``counts`` their multiplicities."""
    order = np.argsort(labels, kind='stable')
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    span = 0.0
    for k in np.unique(counts[counts > 1]):
        members = order[starts[counts == k][:, None] + np.arange(k)[None]]                    # [sets, k]
        Y = img[members]
        span = max(span, float(_min_image_dist(Y[:, :, None] - Y[:, None], T_fc).max()))
    return span


def _close_pairs(X, T_fc, tol) -> np.ndarray:
    """[q, 2] index pairs of atoms closer than ``tol`` Angstrom under periodic images."""
    if len(X) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    r_frac = tol * np.linalg.norm(np.linalg.inv(T_fc), 2)      # |df| <= |T_cf| |dx|
    pairs = cKDTree(X, boxsize=1.0).query_pairs(r_frac, output_type='ndarray')
    if not len(pairs):
        return pairs.reshape(0, 2).astype(np.int64)
    d = _min_image_dist(X[pairs[:, 0]] - X[pairs[:, 1]], T_fc)
    return pairs[d < tol].astype(np.int64)


def _standard_setting(ds, lattice, Xh, Zh, ops_det, sg, symprec, angle_tolerance, T_fc):
    """``(P_spg, p_spg, how)`` with ``x_std = P_spg x_in + p_spg`` in the setting of ``SYM_OPS[sg]``, or
    ``(None, None, why)``. spglib's at the Hall number of ``SYM_OPS[sg]`` first; else ``find_setting`` on the
    detected operators, with translations compared to ``symprec`` over the shortest cell length."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_ops import spglib_hall_number
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)     # spglib 2.7's error-handling notice, per call
        hall = spglib_hall_number(sg)
    ds_std, why = (ds, '') if int(ds.hall_number) == hall else \
        _spglib_dataset(lattice, Xh, Zh, symprec, angle_tolerance, hall)
    if ds_std is not None and int(ds_std.number) == sg:
        P_spg = _snap(np.asarray(ds_std.transformation_matrix, dtype=np.float64))
        if np.linalg.det(P_spg) > 1e-9:
            return P_spg, np.asarray(ds_std.origin_shift, dtype=np.float64), 'spglib'
        why = f'spglib\'s transformation at Hall number {hall} has det {np.linalg.det(P_spg):.3g}'
    elif ds_std is not None:
        why = f'spglib at Hall number {hall} finds sg {int(ds_std.number)}'
    tol = symprec / float(np.linalg.norm(T_fc, axis=0).min())
    st = find_setting(ops_det, sg, tol=tol)
    if st is None:
        return None, None, f'{why}; find_setting at tol {tol:.3g} found none'
    P_spg = _snap(np.linalg.inv(st.P))
    return P_spg, -(P_spg @ st.p), f'find_setting ({why})'


def _image_match(X, Zx, ops_in, T_fc):
    """``(dev [k, n], arg [k, n])``: for every operator and atom, the distance from the atom's image to the nearest
    atom of its element, and that atom's index."""
    k, n = len(ops_in), len(X)
    dev = np.empty((k, n))
    arg = np.empty((k, n), dtype=np.int64)
    trees = {}
    for e in np.unique(Zx):
        idx = np.nonzero(Zx == e)[0]
        trees[e] = (idx, cKDTree(X[idx], boxsize=1.0), min(_NEIGHBOURS, len(idx)))
    for i, op in enumerate(ops_in):
        img = _wrap(X @ op[:3, :3].T + op[:3, 3])
        for e, (idx, tree, kk) in trees.items():
            _, j = tree.query(img[idx], k=list(range(1, kk + 1)))                         # [n_e, kk]
            d = _min_image_dist(img[idx][:, None] - X[idx][j], T_fc)                      # [n_e, kk]
            best = d.argmin(1)
            rows = np.arange(len(idx))
            dev[i, idx] = d[rows, best]
            arg[i, idx] = idx[j[rows, best]]
    return dev, arg


def _permutes(arg, mask) -> bool:
    """Every operator sends the atoms of ``mask`` to distinct atoms."""
    sub = arg[:, mask]
    return all(len(np.unique(row)) == len(row) for row in sub)


def _orbits(arg) -> np.ndarray:
    """[n] component label of every atom under the matches ``arg`` (edges atom -> its image's match)."""
    k, n = arg.shape
    src = np.broadcast_to(np.arange(n), (k, n)).reshape(-1)
    graph = coo_matrix((np.ones(k * n), (src, arg.reshape(-1))), shape=(n, n))
    _, labels = connected_components(graph, directed=True, connection='weak')
    return labels

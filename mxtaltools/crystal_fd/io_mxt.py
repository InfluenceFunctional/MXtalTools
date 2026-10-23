"""Import MolCrystalData rows as CrystalDescriptions in the standard setting, one way, with refusal reasons.

What a row means. A MolCrystalData row (as the dataset builders under
``dataset_utils/construction`` write it, and as the live builder reads it with
``std_orientation=True``) places asymmetric unit ``k`` at
``T_fc t_k + R_k diag(h_k, 1, 1) X_k``:

- ``t_k`` is ``aunit_centroid`` (fractional heavy-atom centroid), ``R_k`` the
  rotation of the rotation vector ``aunit_orientation`` and ``h_k`` is
  ``aunit_handedness``;
- ``X_k`` is the unit's stored ``pos`` aligned by
  ``align_mol_batch_to_standard_axes`` with handedness +1: recentred on its
  heavy atoms and rotated so its heavy-atom principal axes lie along (x, y, z).
  That alignment is ``Ip`` itself, so it mirrors the molecule when the stored
  copy's principal frame is left-handed, and ``X`` is the same point set for a
  molecule and its mirror image (``Ip`` is reflection-covariant). It is the
  frame GFN's buffer admission builds (``orient_molecule(mode='std')``), and the
  frame in which the trainer builds every molecule at handedness +1.

So the import is the identity on the stored parameters of a row stored in the
standard setting, with ``X`` (unit 0) as the row's ``MoleculeRef``; ``h = -1`` says
the crystal holds the mirror image of ``X`` at a proper rotation.

Nonstandard settings (owner decision D1, 2026-10-16). A row whose stored operators
are not ``SYM_OPS[sg]`` as a set (P2_1/n, P2_1/a, I2/a, Pn, Pcab, another origin
choice, a centred setting of P1 such as I1 or B1, ...) is re-expressed in the
standard setting by the change of setting (P, p) that ``settings.find_setting``
derives from the stored operators (``x_stored = P x_new + p``, det P > 0, P
integer or, for a centred P1 setting, det 1/2): cell, centroid (wrapped to
[0, 1)) and rotation by ``settings.re_express``, as ``ops.change_basis`` re-expresses
a pose; handedness and molecule are unchanged. A centred P1 row re-expressed
primitive has half the cell and half the Z, still Z' = 1. The pose check below runs
on the stored parameters, the special-position check on the re-expressed ones.
``ImportInfo`` records (P, p) per imported row. A row no (P, p) resolves is refused
as ``unresolved_setting``. With ``nonstandard='refuse'`` such rows are refused as
before (``operator_count_mismatch``, ``nonstandard_setting``).

Handedness. ``to_sheet`` then moves rows to the requested sheet through an improper
element of G where G has one. A Z' = 1 row of a Sohncke group (G proper) left off the
sheet holds the other enantiomer of ``X``; molecular point symmetry is never used to
move it. By default (owner decision D2, 2026-10-16; ``mirror_reference``) such a row
is re-described with its own enantiomer as the reference molecule: ``X' = diag(-1, 1,
1) X`` at ``h' = -h``, the same atoms (``R diag(h', 1, 1) X' = R diag(h, 1, 1) X``), so
the row's ``MoleculeRef`` is the mirror image of the aligned stored ``pos`` and
``ImportInfo.mirrored`` marks it. That reference must not be re-aligned at handedness
+1: ``align_mol_batch_to_standard_axes`` with handedness None or +1 (as the builder
does with ``std_orientation=True`` at h = +1, and GFN's ``orient_molecule(mode='std')``
without ``target_handedness``) maps it back to ``X``, a molecule and its mirror image
aligning to one point set. Built at h = +1 with the row's (t, R), ``X`` puts the mirror
image of each molecule in its place, at the same centroid and rotation: in general
neither the stored crystal nor its enantiomorph (whose molecules sit at the inverted
centroids), and usually a clashing one (derived; measured 2026-10-17 on 118 sampled
mirrored rows of new_prot_csd.pt: eLJ of that build differs from the stored row's by a
median of 887 (28 re-expressed rows) and 1,080 (90 standard rows), up to 7.6e3
MXtalTools units). ``ImportInfo.reference_handedness`` (-1 where
mirrored, +1 elsewhere) is the handedness argument under which that alignment leaves
each row's reference fixed; a consumer that re-aligns references must pass it, and a
saved dataset must store it with the molecule. With ``mirror_reference=False``, or for
Z' > 1 rows, the row stays off the sheet, flagged.

``off_sheet`` names the treatment of such a row: ``'mirror_reference'`` (D2, the default),
``'keep'`` (``mirror_reference=False``) or ``'enantiomorph'`` (owner decision 2026-10-21): the
row is replaced by its mirror-image crystal, ``enantiomorph.enantiomorph`` (inversion through
the origin, written in the standard setting of its own group or, for the 22 groups of the 11
enantiomorphic pairs (76/78, 91/95, 92/96, 144/145, 151/153, 152/154, 169/170, 171/172,
178/179, 180/181, 212/213), of the partner group): reference ``X`` (not mirrored), ``h' = -h``
(on the sheet), ``reference_handedness`` +1, ``mirrored`` False, ``ImportInfo.enantiomorph``
True and ``ImportInfo.sg_source`` the space group before the map. That row is not the deposited
crystal but its mirror image (energies are required to be mirror-invariant), and it is not
canonical: canonicalise after the import. Its atoms are the D2 row's inverted through the origin
and shifted by the origin shift -p (fractional x -> -x - p; ``enantiomorph.enantiomorph_map(sg).p``,
nonzero only in sg 80, 98, 210), and ``X = X' diag(-1, 1, 1)`` bit for bit. Not applied to Z' > 1
rows or with ``sheet=None``.

Checks, cheapest first; a row is refused for the first that fails (``REASONS``
order). Rows that fail a cheap check skip the later, batched ones.

1. fields present (``missing_field``), space group in 1-230
   (``invalid_space_group``), integer Z' with equal unit blocks
   (``invalid_z_prime``), ``cocrystal`` or a disorder flag set if the row
   carries one (``cocrystal``, ``disorder``);
2. setting: the stored ``symmetry_operators`` equal ``SYM_OPS[sg]`` as a set
   (rotation parts equal, translations equal modulo 1, within ``op_tol``; the order
   may differ), or ``settings.find_setting`` resolves them (``unresolved_setting``
   otherwise). With ``nonstandard='refuse'``: ``operator_count_mismatch`` (a
   different number of operators) or ``nonstandard_setting`` (same number, another
   set). Non-finite operators are ``nonfinite`` and skip this comparison;
3. ``nonfinite`` values (cell, centroids, rotation vectors, ``pos``, operators), a
   cell with no real metric (``degenerate_cell``), handedness other than +-1
   (``degenerate_handedness``: 0 is what the builder stores for a degenerate
   principal-axis basis), a unit with no heavy atom (``no_heavy_atoms``);
4. batched: a re-expressed row whose cell the rotations of ``SYM_OPS[sg]`` do not
   preserve within ``metric_tol`` (``setting_metric_mismatch``: ``metric_dev``, the
   largest ``|(W^T G W - G)_ij| / sqrt(G_ii G_jj)``, above ``metric_tol``; e.g. an
   a-unique 'P 21/c 1 1' setting deposited with beta, not alpha, off 90 deg;
   ``ops.reduce_cell`` would refuse it at its ``angle_tol``). No change of setting can
   resolve such a row: ``W^T G W - G`` becomes ``P^T (W^T G W - G) P`` under any (P, p).
   ``metric_tol`` = 1e-6 is far tighter than CIF precision, so the refusal is not always
   a contradiction: of the 6 rows it refuses in new_prot_csd.pt, five have beta 11 to 40
   times its CIF standard uncertainty from 90 deg (KAMKOY01 gives none) and one,
   XENJEJ (beta = 90.001(2) deg), is within it. Snapping such rows is an owner choice
   (open). Standard rows are measured (``ImportInfo.metric_dev``) but not refused for
   it, as before D1, so the rule is asymmetric: XATDEF (sg 173, a and b equal within
   the CIF's uncertainty, metric_dev 2.3e-5) is imported. A molecule on a special
   position (``special_position``: two
   molecules of the unit cell, G images of the units in the standard setting
   (re-expressed rows: after the re-expression), with centroids within
   ``special_tol`` Angstrom); the description does not rebuild the stored
   coordinates (``pose_mismatch``: the literal asymmetric unit
   ``T_fc t + R diag(h, 1, 1) X`` of the stored parameters, in the stored cell,
   differs from the stored ``pos`` by more than ``pose_tol`` Angstrom modulo the
   lattice); Z' > 1 units whose aligned molecules differ (``zp_units_differ``: the
   description holds one molecule per row);
5. Z' > 1 (``z_prime_gt1``), unless ``allow_zp_gt1``: a Z' > 1 row is not the
   deposited crystal. The CSD featurizer (``featurize_cif_chunks`` with
   ``featurization_utils.overwrite_conformer_to_zp``) writes unit 0's conformer into
   every unit, gated only loosely (nearest-atom distance 0.5 Angstrom, eLJ within
   1%), so units 1 and above differ from the CIF: on new_prot_csd.pt by a median of
   0.05 Angstrom and up to 2.5 Angstrom (heavy atoms; 2026-10-16). Older files keep
   each unit's own conformer and fail ``zp_units_differ`` instead. With
   ``allow_zp_gt1`` such rows are imported as that one-conformer idealisation and
   flagged (``ImportInfo.zp_gt1``); the rest of the package raises
   ``NotImplementedError`` on them. Rows with Z' = 1 are imported fully.

The molecule ``X`` is a function of the stored coordinates and of the arithmetic that
aligns them, not of the molecule: the principal-axis frame of a molecule with
near-degenerate moments, a near-zero third moment or coplanar heavy atoms is decided by
rounding. BAZGOY03 (7 heavy atoms, third moments 2e-4 and 0): 26 of 50 rotated copies
align 180 deg apart, which ``pose_mismatch`` catches when the stored pose was computed in
the other frame. QQQFVD03 (coplanar heavy atoms): GFN's path (``MolData``, collate,
``orient_molecule(mode='std')``) on bitwise the same float32 ``pos`` returns
``diag(1, 1, -1) X``, a reflected frame (hydrogens 1.8 Angstrom off) that would also flip
h (2026-10-17). So the same function on the same stored ``pos`` through another code path
is not guaranteed to reproduce ``X``: a consumer of saved descriptions must keep the saved
``X`` (with its ``reference_handedness``), not re-derive it. Re-aligning the saved
reference with its ``reference_handedness`` returns it (tests/fd/test_io_settings.py).
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_fd import settings, so3
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
from mxtaltools.crystal_fd.domains import to_sheet
from mxtaltools.crystal_fd.enantiomorph import enantiomorph
from mxtaltools.crystal_fd.ops import cell_matrices, real_cells
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64

#: refusal codes, in the order the checks run; a refused row carries the first that applies.
#: ``operator_count_mismatch`` and ``nonstandard_setting`` occur only with ``nonstandard='refuse'``,
#: ``unresolved_setting`` only with ``nonstandard='reexpress'`` (the default)
REASONS = ('missing_field', 'invalid_space_group', 'invalid_z_prime', 'cocrystal', 'disorder',
           'operator_count_mismatch', 'nonstandard_setting', 'unresolved_setting', 'nonfinite', 'degenerate_cell',
           'degenerate_handedness', 'no_heavy_atoms', 'setting_metric_mismatch', 'special_position', 'pose_mismatch',
           'zp_units_differ', 'z_prime_gt1')
#: how ``from_mol_crystal_data`` treats rows whose stored operators are not ``SYM_OPS[sg]`` as a set
NONSTANDARD_MODES = ('reexpress', 'refuse')
#: how ``from_mol_crystal_data`` treats a Z' = 1 Sohncke row ``to_sheet`` leaves off the requested sheet
OFF_SHEET_MODES = ('mirror_reference', 'enantiomorph', 'keep')
#: fields a row must carry
REQUIRED = ('sg_ind', 'z_prime', 'cell_lengths', 'cell_angles', 'aunit_centroid', 'aunit_orientation',
            'aunit_handedness', 'pos', 'z', 'symmetry_operators')
#: row attributes read as disorder flags when present (none of the 2025-2026 CSD files carries one)
DISORDER_KEYS = ('has_disorder', 'disorder', 'is_disordered')

_NEIGHBOURS = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3).to(F64)
_STD_CACHE = {}
_SOHNCKE = {}


@dataclass(frozen=True, eq=False)
class ImportInfo:
    """Per-row outcome of ``from_mol_crystal_data``.

    Input-row fields ([B]): ``imported`` bool; ``reason`` (str, or None when imported);
    ``reason_mask`` [B, len(REASONS)] bool, every check that ran and failed (a row
    failing a cheap check is not given the batched ones); ``identifier`` (list);
    ``sg`` long (-1 when unreadable); ``z_prime`` long (0 when unreadable);
    ``stored_nonstandard`` bool (the row's ``nonstandard_symmetry`` flag, False when
    absent); ``ops_reordered`` bool (stored operators equal ``SYM_OPS[sg]`` as a set but
    not in order); ``setting_reexpressed`` bool (stored operators not ``SYM_OPS[sg]`` as a
    set, resolved by ``settings.find_setting``; whether or not a later check refused the
    row); ``well_defined`` bool (the row's ``is_well_defined``, True when
    absent); ``pose_dev`` float64 (largest atom deviation of the literal asymmetric
    units from the stored ``pos``, Angstrom, in the stored setting; NaN where not computed);
    ``min_centroid_sep`` float64 (closest pair of molecular centroids in the unit cell,
    over G images and units and their lattice neighbours, Angstrom, in the standard setting;
    NaN where not computed, +inf where the unit cell holds one molecule, i.e. P1 at Z' = 1);
    ``metric_dev`` float64 (largest ``|(W^T G W - G)_ij| / sqrt(G_ii G_jj)`` over the rotations W
    of ``SYM_OPS[sg]``, the cell's departure from the group's metric constraints in the standard
    setting, dimensionless (about rad off the diagonal); NaN where not computed).

    Imported-row fields ([N], in input order): ``row`` long (input row); ``mols``
    (list of MoleculeRef: unit 0's aligned molecule, or its mirror image where
    ``mirrored``); ``zp_gt1`` bool (only with ``allow_zp_gt1``: the featurizer's
    one-conformer idealisation, not the CIF); ``h_input`` [N, Zmax] long (stored
    handedness, 0 in empty slots); ``on_sheet`` bool (every unit on the requested sheet
    after ``to_sheet`` and the mirrored reference or enantiomorph; True when no sheet was
    requested); ``sohncke`` bool (G has no improper element); ``setting_P`` [N, 3, 3] float64 and
    ``setting_p`` [N, 3] float64 (the change of setting applied, ``x_stored = P x + p``;
    identity and zero for rows already in the standard setting; for an ``enantiomorph`` row, that
    of the source crystal, before the map); ``mirrored`` bool (D2: the reference molecule is
    ``diag(-1, 1, 1) X`` and h was flipped); ``enantiomorph`` bool (``off_sheet='enantiomorph'``:
    the row is the mirror image of the imported crystal, reference ``X``); ``sg_source`` long (the
    space group before the enantiomorph map; the description's ``sg`` on every other row).

    Derived: ``reference_handedness`` [N] long, -1 where ``mirrored`` and +1 elsewhere.
    """
    imported: torch.Tensor
    reason: list
    reason_mask: torch.Tensor
    identifier: list
    sg: torch.Tensor
    z_prime: torch.Tensor
    stored_nonstandard: torch.Tensor
    ops_reordered: torch.Tensor
    setting_reexpressed: torch.Tensor
    well_defined: torch.Tensor
    pose_dev: torch.Tensor
    min_centroid_sep: torch.Tensor
    metric_dev: torch.Tensor
    row: torch.Tensor
    mols: list
    zp_gt1: torch.Tensor
    h_input: torch.Tensor
    on_sheet: torch.Tensor
    sohncke: torch.Tensor
    setting_P: torch.Tensor
    setting_p: torch.Tensor
    mirrored: torch.Tensor
    enantiomorph: torch.Tensor
    sg_source: torch.Tensor

    @property
    def n_rows(self) -> int:
        return len(self.reason)

    @property
    def reference_handedness(self) -> torch.Tensor:
        """[N] long, per imported row: the handedness (``align_mol_batch_to_standard_axes``'s ``handedness`` column
        0; GFN's ``orient_molecule(mode='std', target_handedness=...)``) under which aligning the row's reference
        molecule leaves it where it is: -1 for a D2 mirrored reference ``X' = diag(-1, 1, 1) X``, +1 for ``X``
        (enantiomorph rows included).
        Aligning ``X'`` at +1 (or None) returns ``X``, and building that at h = +1 builds a different crystal
        (module docstring, Handedness); a dataset of imported rows must store this flag with the molecule."""
        return torch.where(self.mirrored, -1, 1).to(torch.long)

    def reason_counts(self) -> dict:
        """{reason: rows refused for it first}, over every code in REASONS."""
        out = {r: 0 for r in REASONS}
        for r in self.reason:
            if r is not None:
                out[r] += 1
        return out


# ---------------------------------------------------------------------------
# per-row checks
# ---------------------------------------------------------------------------

def _std_ops(sg: int) -> np.ndarray:
    if sg not in _STD_CACHE:
        _STD_CACHE[sg] = np.stack([np.asarray(o, dtype=np.float64) for o in SYM_OPS[sg]])
    return _STD_CACHE[sg]


def operator_set_match(ops, sg: int, tol: float = 1e-6):
    """(same_set, same_order) for affine operators ``ops`` [n, 4, 4] against ``SYM_OPS[sg]``.

    Two operators match when their rotation parts are equal integer matrices and their
    translations agree modulo 1, within ``tol``; the sets are equal when the match is a
    bijection (SYM_OPS has no two operators equal modulo the lattice).
    """
    ops = np.asarray(ops, dtype=np.float64).reshape(-1, 4, 4)
    std = _std_ops(sg)
    if len(ops) != len(std):
        return False, False
    W = np.round(ops[:, :3, :3])
    if np.abs(ops[:, :3, :3] - W).max() > tol or np.abs(ops[:, 3] - np.array([0, 0, 0, 1.0])).max() > tol:
        return False, False
    dW = np.abs(W[:, None] - std[None, :, :3, :3]).max((-1, -2)) <= tol
    dw = ops[:, None, :3, 3] - std[None, :, :3, 3]
    dw = np.abs(dw - np.round(dw)).max(-1) <= tol
    match = dW & dw
    same = bool((match.sum(1) == 1).all() and (match.sum(0) == 1).all())
    return same, bool(same and np.diag(match).all())


def is_sohncke(sg: int) -> bool:
    """True when G (SYM_OPS[sg]) has no improper element: its crystals hold one hand of a chiral molecule."""
    sg = int(sg)
    if sg not in _SOHNCKE:
        _SOHNCKE[sg] = bool((torch.round(torch.linalg.det(group_table(sg).g_W)) > 0).all())
    return _SOHNCKE[sg]


def _scalar(v) -> float:
    return float(torch.as_tensor(v).reshape(-1)[0])


#: optional fields read when present
OPTIONAL = ('nonstandard_symmetry', 'is_well_defined', 'cocrystal', 'identifier') + DISORDER_KEYS


def _pick(v, i: int, B: int):
    """Row ``i`` of a collated batch attribute: list entries and leading-dimension tensor rows; anything else as is."""
    if isinstance(v, (list, tuple)):
        return v[i] if len(v) == B else v
    if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == B:
        return v[i]
    if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == B:
        return v[i]
    return v


def _row_fields(batch) -> list:
    """Per input row, a dict of the fields the importer reads (None where a field is absent).

    A list holds MolCrystalData rows. A collated batch (anything with ``ptr`` and ``num_graphs``) is read graph by
    graph without ``to_data_list``: per-graph values by ``_pick``, ``pos`` and ``z`` by ``ptr``.
    """
    names = REQUIRED + OPTIONAL
    if isinstance(batch, (list, tuple)):
        out = []
        for r in batch:
            keys = set(r.keys())
            out.append({k: (getattr(r, k) if k in keys else None) for k in names})
        return out
    B = int(batch.num_graphs)
    ptr = batch.ptr
    keys = set(batch.keys())
    out = []
    for i in range(B):
        a, b = int(ptr[i]), int(ptr[i + 1])
        f = {}
        for k in names:
            if k not in keys:
                f[k] = None
            elif k in ('pos', 'z'):
                f[k] = getattr(batch, k)[a:b]
            else:
                f[k] = _pick(getattr(batch, k), i, B)
        out.append(f)
    return out


def _row_checks(f: dict, op_tol: float, nonstandard: str = 'reexpress'):
    """(first failing reason or None, set of failing reasons, sg, z', stored nonstandard flag, reordered, well defined,
    setting transform or None)."""
    fails = []
    sg, zp, reordered, transform = -1, 0, False, None
    nonstd = bool(f['nonstandard_symmetry']) if f['nonstandard_symmetry'] is not None else False
    well = bool(f['is_well_defined']) if f['is_well_defined'] is not None else True
    if any(f[k] is None for k in REQUIRED):
        return 'missing_field', {'missing_field'}, sg, zp, nonstd, reordered, well, transform
    sgv = _scalar(f['sg_ind'])
    sg = int(sgv) if np.isfinite(sgv) else -1
    if not (np.isfinite(sgv) and sgv == sg and 1 <= sg <= 230):
        fails.append('invalid_space_group')
    zpv = _scalar(f['z_prime'])
    zp = int(zpv) if np.isfinite(zpv) else 0
    n_atoms = int(torch.as_tensor(f['z']).numel())
    if (not np.isfinite(zpv) or zpv != zp or zp < 1 or n_atoms % zp
            or torch.as_tensor(f['aunit_centroid']).numel() < 3 * zp
            or torch.as_tensor(f['aunit_orientation']).numel() < 3 * zp
            or torch.as_tensor(f['aunit_handedness']).numel() < zp
            or torch.as_tensor(f['pos']).reshape(-1, 3).shape[0] != n_atoms):
        fails.append('invalid_z_prime')
    if f['cocrystal'] is not None and bool(f['cocrystal']):
        fails.append('cocrystal')
    if any(f[k] is not None and bool(f[k]) for k in DISORDER_KEYS):
        fails.append('disorder')
    ops = np.asarray(f['symmetry_operators'], dtype=np.float64)
    if not np.isfinite(ops).all():
        fails.append('nonfinite')
    elif 'invalid_space_group' not in fails:
        if ops.size % 16 or len(ops.reshape(-1, 4, 4)) != len(SYM_OPS[sg]):
            same, in_order, kind = False, False, 'operator_count_mismatch'
        else:
            same, in_order = operator_set_match(ops, sg, op_tol)
            kind = 'nonstandard_setting'
        reordered = same and not in_order
        if not same:
            if nonstandard == 'refuse':
                fails.append(kind)
            else:
                transform = settings.find_setting(ops, sg, op_tol) if ops.size % 16 == 0 else None
                if transform is None:
                    fails.append('unresolved_setting')
    if 'invalid_z_prime' not in fails:
        cell = torch.cat([torch.as_tensor(f['cell_lengths']).reshape(-1), torch.as_tensor(f['cell_angles']).reshape(-1)])
        vals = [cell, torch.as_tensor(f['aunit_centroid']).reshape(-1)[:3 * zp],
                torch.as_tensor(f['aunit_orientation']).reshape(-1)[:3 * zp], torch.as_tensor(f['pos']).reshape(-1)]
        if not all(bool(torch.isfinite(v.double()).all()) for v in vals) or cell.numel() != 6:
            fails.append('nonfinite')
        elif not bool(real_cells(cell.double()[None])[0]):
            fails.append('degenerate_cell')
        h = torch.as_tensor(f['aunit_handedness']).reshape(-1)[:zp].double()
        if not bool((h.abs() == 1).all()):
            fails.append('degenerate_handedness')
        heavy = (torch.as_tensor(f['z']).reshape(zp, -1) > 1).any(1)
        if not bool(heavy.all()):
            fails.append('no_heavy_atoms')
    first = next((r for r in REASONS if r in fails), None)
    return first, set(fails), sg, zp, nonstd, reordered, well, transform


# ---------------------------------------------------------------------------
# batched checks
# ---------------------------------------------------------------------------

def _min_centroid_separation(sg: torch.Tensor, cell: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[b] smallest Cartesian distance (Angstrom) between two distinct molecules of each unit cell.

    Molecules are the G images of every unit (centroids W_g t_k + w_g); the identity
    image of a unit is excluded from its own comparison. Distances are minimum images,
    rounded then searched over the 27 neighbouring cells.
    """
    out = torch.full((len(sg),), float('inf'), dtype=F64)
    T_all = cell_matrices(cell)
    nb = _NEIGHBOURS
    Z = t.shape[1]
    for s in torch.unique(sg).tolist():
        idx = torch.nonzero(sg == s)[:, 0]
        tab = group_table(s)
        W, w = tab.g_W, tab.g_w
        nG = W.shape[0]
        y = torch.einsum('gij,bkj->bkgi', W, t[idx]) + w                    # [b, Z, G, 3] images of every unit
        d = y[:, None, :, :, :] - t[idx][:, :, None, None, :]               # [b, k, l, G, 3]: image (l, g) - unit k
        d = d - torch.round(d)
        cart = torch.einsum('bij,bklgnj->bklgni', T_all[idx], d[..., None, :] + nb)
        dist = cart.norm(dim=-1).amin(-1)                                   # [b, k, l, G]
        self_img = torch.zeros(Z, Z, nG, dtype=torch.bool)
        self_img[torch.arange(Z), torch.arange(Z), 0] = True                # g = 0 is the identity (table order)
        m = mask[idx]
        valid = m[:, :, None, None] & m[:, None, :, None] & ~self_img
        dist = torch.where(valid, dist, torch.full_like(dist, float('inf')))
        out[idx] = dist.reshape(len(idx), -1).amin(1)
    return out


class _Units:
    """The attributes ``align_mol_batch_to_standard_axes`` reads, for molecules given as flat tensors.

    ``recenter_molecules`` is ``MolData.recenter_molecules`` for a batch (heavy-atom centroids, ``center_batch``),
    so the alignment is the live builder's, computed on the stored coordinates in their stored dtype.
    """
    is_batch = True

    def __init__(self, pos: torch.Tensor, z: torch.Tensor, batch: torch.Tensor, num_graphs: int):
        self.pos, self.z, self.batch, self.num_graphs = pos, z, batch, num_graphs
        self.num_atoms = torch.bincount(batch, minlength=num_graphs)
        self.device = pos.device

    def recenter_molecules(self, center_on_heavy_atoms: bool = True):
        from mxtaltools.common.geometry_utils import center_batch
        self.pos = center_batch(self.pos, self.batch, self.num_graphs, self.num_atoms, center_on_heavy_atoms, self.z)


def _batched(fs: list, pose_tol: float, transforms: Optional[list] = None):
    """Literal descriptions of rows that passed the per-row checks (field dicts), with the batched checks.

    Unit k of a row is the k-th of ``z'`` equal contiguous atom blocks of ``pos`` (as ``split_to_zp1_batch``
    reads it), with parameter block k of the centroid, rotation vector and handedness. ``transforms`` (one
    ``settings.SettingTransform`` or None per row): rows with one are re-expressed in the standard setting
    (``settings.re_express``) after the pose check, which reads the stored setting, and before the special-position
    check, which reads the standard one.

    Returns (t [b, Z, 3], R [b, Z, 3, 3], h [b, Z], cell [b, 6], sg [b], z' [b], X list of [n, 3] float64
    (unit 0, aligned), z list, pose_dev [b], min_sep [b], units_differ [b] bool), the poses and cells in the
    standard setting.
    """
    from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes
    zps = [int(_scalar(f['z_prime'])) for f in fs]
    Zm = max(zps)
    b = len(fs)
    pos_l, z_l, ub_l, t_l, rv_l, h_l, cell_l, sg_l = [], [], [], [], [], [], [], []
    U = 0
    for f, zp in zip(fs, zps):
        pos = torch.as_tensor(f['pos']).reshape(-1, 3)
        z = torch.as_tensor(f['z']).reshape(-1).long()
        n = len(z) // zp
        pos_l.append(pos)
        z_l.append(z)
        ub_l.append(torch.arange(U, U + zp).repeat_interleave(n))
        U += zp
        t_l.append(torch.as_tensor(f['aunit_centroid']).reshape(-1)[:3 * zp].double().reshape(zp, 3))
        rv_l.append(torch.as_tensor(f['aunit_orientation']).reshape(-1)[:3 * zp].double().reshape(zp, 3))
        h_l.append(torch.as_tensor(f['aunit_handedness']).reshape(-1)[:zp].double().round().long())
        cell_l.append(torch.cat([torch.as_tensor(f['cell_lengths']).reshape(3),
                                 torch.as_tensor(f['cell_angles']).reshape(3)]).double())
        sg_l.append(int(_scalar(f['sg_ind'])))
    dtypes = {p.dtype for p in pos_l}
    pos_all = torch.cat([p.to(torch.float32) for p in pos_l]) if len(dtypes) > 1 else torch.cat(pos_l)
    stored = pos_all.double()
    z_all = torch.cat(z_l)
    ub = torch.cat(ub_l)
    units = _Units(pos_all.clone(), z_all, ub, U)
    align_mol_batch_to_standard_axes(units, handedness=None)                 # the live builder's frame, in place
    X = units.pos.double()
    zp_t = torch.tensor(zps)
    unit_row = torch.arange(b).repeat_interleave(zp_t)
    t_u = torch.cat(t_l)
    R_u = so3.rotvec_to_matrix(torch.cat(rv_l))
    h_u = torch.cat(h_l)
    cell = torch.stack(cell_l)
    T_u = cell_matrices(cell)[unit_row]
    flip = torch.ones(U, 3, dtype=F64)
    flip[:, 0] = h_u.double()
    P_u = R_u * flip[:, None, :]
    lit = torch.einsum('nij,nj->ni', P_u[ub], X) + torch.einsum('nij,nj->ni', T_u[ub], t_u[ub])
    fd = torch.linalg.solve(T_u[ub], (lit - stored)[..., None])[..., 0]     # fractional differences
    nu = torch.bincount(ub, minlength=U).clamp(min=1).double()
    shift = torch.round(torch.zeros(U, 3, dtype=F64).index_add_(0, ub, fd) / nu[:, None])
    dev_atom = torch.einsum('nij,nj->ni', T_u[ub], fd - shift[ub]).norm(dim=1)
    dev_u = torch.zeros(U, dtype=F64).scatter_reduce_(0, ub, dev_atom, 'amax', include_self=True)

    unit_slot = torch.arange(U) - (torch.cumsum(zp_t, 0) - zp_t)[unit_row]
    t = torch.zeros(b, Zm, 3, dtype=F64)
    R = torch.eye(3, dtype=F64).repeat(b, Zm, 1, 1)
    h = torch.ones(b, Zm, dtype=torch.long)
    t[unit_row, unit_slot] = t_u
    R[unit_row, unit_slot] = R_u
    h[unit_row, unit_slot] = h_u
    pose_dev = torch.zeros(b, dtype=F64).scatter_reduce_(0, unit_row, dev_u, 'amax', include_self=True)
    sg = torch.tensor(sg_l, dtype=torch.long)
    mask = torch.arange(Zm)[None] < zp_t[:, None]
    moved = [i for i, tr in enumerate(transforms or []) if tr is not None]
    if moved:
        ii = torch.tensor(moved)
        sub = CrystalDescription.from_parts(sg[ii], cell[ii], t[ii], R[ii], h[ii], zp_t[ii])
        new = settings.re_express(sub, torch.as_tensor(np.stack([transforms[i].P for i in moved])),
                                  torch.as_tensor(np.stack([transforms[i].p for i in moved])), wrap=True)
        t[ii], R[ii], cell[ii] = new.t, new.R, new.cell
    min_sep = _min_centroid_separation(sg, cell, t, mask)
    metric_dev = metric_deviation(sg, cell)

    Xs, zs, differ = [], [], torch.zeros(b, dtype=torch.bool)
    starts = torch.cumsum(torch.bincount(ub, minlength=U), 0) - torch.bincount(ub, minlength=U)
    counts = torch.bincount(ub, minlength=U)
    first_unit = torch.cumsum(zp_t, 0) - zp_t
    for i in range(b):
        u0 = int(first_unit[i])
        s0, n0 = int(starts[u0]), int(counts[u0])
        X0 = X[s0:s0 + n0]
        z0 = z_all[s0:s0 + n0]
        for k in range(1, zps[i]):
            sk, nk = int(starts[u0 + k]), int(counts[u0 + k])
            if nk != n0 or not torch.equal(z_all[sk:sk + nk], z0) or \
                    float((X[sk:sk + nk] - X0).norm(dim=1).max()) > pose_tol:
                differ[i] = True
        Xs.append(X0)
        zs.append(z0)
    return t, R, h, cell, sg, zp_t, Xs, zs, pose_dev, min_sep, differ, metric_dev


def metric_deviation(sg: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """[b] float64: max over the rotations W of ``SYM_OPS[sg]`` and entries (i, j) of
    ``|(W^T G W - G)_ij| / sqrt(G_ii G_jj)``, G the metric of ``cell`` [b, 6] (standard setting); 0 for a cell the
    group's rotations preserve exactly (float32 storage of a CSD cell: about 1e-7)."""
    from mxtaltools.crystal_fd.ops import metric_tensor
    G = metric_tensor(cell)
    out = torch.zeros(len(G), dtype=F64)
    n = torch.sqrt(torch.diagonal(G, dim1=1, dim2=2))
    for s in torch.unique(sg).tolist():
        m = sg == s
        W = group_table(s).g_W
        g = G[m]
        Gw = torch.einsum('wji,bjk,wkl->bwil', W, g, W)
        out[m] = ((Gw - g[:, None]).abs() / (n[m][:, None, :, None] * n[m][:, None, None, :])).amax((1, 2, 3))
    return out


# ---------------------------------------------------------------------------
# the importer
# ---------------------------------------------------------------------------

def _off_sheet_mode(mirror_reference: Optional[bool], off_sheet: Optional[str]) -> str:
    """The ``OFF_SHEET_MODES`` entry ``from_mol_crystal_data`` runs: ``off_sheet`` when given, else from
    ``mirror_reference`` (True or None: ``'mirror_reference'``; False: ``'keep'``). Both given must agree:
    ``mirror_reference`` True with ``'mirror_reference'`` only, False with ``'keep'`` or ``'enantiomorph'``."""
    if mirror_reference is not None and not isinstance(mirror_reference, bool):
        raise ValueError('mirror_reference must be True, False or None')
    if off_sheet is None:
        return 'keep' if mirror_reference is False else 'mirror_reference'
    if not isinstance(off_sheet, str) or off_sheet not in OFF_SHEET_MODES:
        raise ValueError(f'off_sheet must be one of {OFF_SHEET_MODES} or None, got {off_sheet!r}')
    if mirror_reference is not None and mirror_reference != (off_sheet == 'mirror_reference'):
        raise ValueError(f'mirror_reference={mirror_reference} contradicts off_sheet={off_sheet!r}; pass one of them')
    return off_sheet


def from_mol_crystal_data(batch, sheet: Optional[int] = 1, pose_tol: float = 1e-2, special_tol: float = 0.1,
                          op_tol: float = 1e-6, allow_zp_gt1: bool = False, nonstandard: str = 'reexpress',
                          mirror_reference: Optional[bool] = None, metric_tol: float = 1e-6,
                          off_sheet: Optional[str] = None):
    """Import MolCrystalData rows as descriptions in the standard setting ``SYM_OPS[sg]``.

    Parameters
    ----------
    batch : list of MolCrystalData, or a collated MolCrystalData batch (read graph by graph through ``ptr``)
    sheet : int or None
        Move imported rows to this handedness with ``domains.to_sheet`` (per unit for
        Z' > 1); None keeps the stored handedness (and ``off_sheet`` does nothing).
        Rows that cannot move (Sohncke groups) keep it and are flagged ``on_sheet = False``,
        unless ``off_sheet`` (``mirror_reference``) re-describes them.
    pose_tol : float
        Angstrom. Largest allowed deviation of the literal asymmetric units from the stored
        ``pos``, and between the aligned molecules of the units of a Z' > 1 row. Float32
        storage gives about 1e-5; rotation vectors stored near angle 0 or pi (where the
        builder's arccos loses precision) up to a few 1e-4; the dataset builder accepted a
        rebuild within 5e-2 of the CIF atoms.
    special_tol : float
        Angstrom. Two molecular centroids of one unit cell closer than this mark a molecule
        on a special position. On a special position the images coincide (float32 storage:
        about 1e-5); molecules that only wrap around each other come closer than 1 Angstrom
        without touching (new_prot_csd.pt: the closest pair is 0.495 Angstrom, DEWSOO, whose
        closest heavy-atom contact between the two is 3.38 Angstrom).
    op_tol : float
        Tolerance on operator entries for the setting comparison. The default suits exact
        fractions (as the CSD featurizer stores them); a source that writes translations as
        decimals needs a tolerance above their rounding (4 decimals: 0.3333 for 1/3, so 1e-4).
    allow_zp_gt1 : bool
        Import Z' > 1 rows (flagged ``zp_gt1``) instead of refusing them as ``z_prime_gt1``.
        Their units 1 and above carry the featurizer's copy of unit 0's conformer, not the
        deposited molecules (module docstring, check 5).
    nonstandard : str
        ``'reexpress'`` (default, owner decision D1): rows whose stored operators are not
        ``SYM_OPS[sg]`` as a set are re-expressed in the standard setting by the (P, p) of
        ``settings.find_setting``, or refused as ``unresolved_setting``. ``'refuse'``: refused
        as ``operator_count_mismatch`` or ``nonstandard_setting`` (the importer before D1).
    mirror_reference : bool or None
        True (the behaviour when neither this nor ``off_sheet`` is given; owner decision D2): a
        Z' = 1 row of a Sohncke group left off the requested sheet by ``to_sheet`` is
        re-described with the mirror image of its aligned molecule as the reference,
        ``X' = diag(-1, 1, 1) X`` at ``h' = -h`` (the same atoms), and flagged
        ``ImportInfo.mirrored``. False leaves such rows off the sheet, flagged
        ``on_sheet = False``. Not applied to Z' > 1 rows. None: as ``off_sheet`` says.
    metric_tol : float
        Largest ``ImportInfo.metric_dev`` of a re-expressed row (``setting_metric_mismatch``
        above it). The default is ``ops.reduce_cell``'s ``angle_tol``; float32 storage
        amplified by a setting change gives up to about 1e-6 (new_prot_csd.pt, 2026-10-16:
        every re-expressed row but 6 is at or below 3.7e-7; the 6, rows deposited in an a- or
        c-unique monoclinic setting with beta rather than the unique-axis angle off 90 deg,
        are at 3.5e-5 to 2.2e-2, XENJEJ's 3.5e-5 within its CIF's standard uncertainty;
        module docstring, check 4).
    off_sheet : str or None
        ``OFF_SHEET_MODES``: ``'mirror_reference'`` (as ``mirror_reference=True``), ``'keep'``
        (as ``mirror_reference=False``) or ``'enantiomorph'``: such a row is replaced by its
        mirror-image crystal (``enantiomorph.enantiomorph``; reference ``X``, ``h' = -h``, sg the
        partner in the 11 enantiomorphic pairs), flagged ``ImportInfo.enantiomorph``, with
        ``ImportInfo.sg_source`` the space group before the map (module docstring, Handedness).
        None: as ``mirror_reference`` says. ``mirror_reference`` given with an ``off_sheet`` it
        contradicts (True with another mode, False with ``'mirror_reference'``) raises.

    Returns
    -------
    (CrystalDescription of the imported rows, float64, Zmax = their largest Z' (1 when none),
    ImportInfo)
    """
    rows = _row_fields(batch)
    if sheet not in (None, 1, -1):
        raise ValueError('sheet must be +1, -1 or None')
    if nonstandard not in NONSTANDARD_MODES:
        raise ValueError(f'nonstandard must be one of {NONSTANDARD_MODES}')
    mode = _off_sheet_mode(mirror_reference, off_sheet)
    B = len(rows)
    reason: List[Optional[str]] = [None] * B
    rmask = torch.zeros(B, len(REASONS), dtype=torch.bool)
    sg_all = torch.full((B,), -1, dtype=torch.long)
    zp_all = torch.zeros(B, dtype=torch.long)
    nonstd = torch.zeros(B, dtype=torch.bool)
    reord = torch.zeros(B, dtype=torch.bool)
    reexp = torch.zeros(B, dtype=torch.bool)
    well = torch.ones(B, dtype=torch.bool)
    ident, transforms = [], [None] * B
    for i, r in enumerate(rows):
        first, fails, sg, zp, ns, ro, wd, tr = _row_checks(r, op_tol, nonstandard)
        reason[i] = first
        for f in fails:
            rmask[i, REASONS.index(f)] = True
        sg_all[i], zp_all[i], nonstd[i], reord[i], well[i] = sg, zp, ns, ro, wd
        transforms[i] = tr
        reexp[i] = tr is not None
        idv = r['identifier']
        ident.append(idv[0] if isinstance(idv, (list, tuple)) and len(idv) else idv)
    pose_dev = torch.full((B,), float('nan'), dtype=F64)
    min_sep = torch.full((B,), float('nan'), dtype=F64)
    metric_dev = torch.full((B,), float('nan'), dtype=F64)
    passed = [i for i in range(B) if reason[i] is None]
    flags = (nonstd, reord, reexp, well, metric_dev)

    empty = CrystalDescription.from_parts(torch.zeros(0, dtype=torch.long), torch.zeros(0, 6, dtype=F64),
                                          torch.zeros(0, 1, 3, dtype=F64), torch.zeros(0, 1, 3, 3, dtype=F64),
                                          torch.zeros(0, 1, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    if not passed:
        return empty, _info(reason, rmask, ident, sg_all, zp_all, flags, pose_dev, min_sep, *_no_rows())
    t, R, h, cell, sg, zp, Xs, zs, dev, sep, differ, mdev = _batched([rows[i] for i in passed], pose_tol,
                                                                     [transforms[i] for i in passed])
    pidx = torch.tensor(passed)
    pose_dev[pidx] = dev
    min_sep[pidx] = sep
    metric_dev[pidx] = mdev
    bad = {'setting_metric_mismatch': reexp[pidx] & ~(mdev <= metric_tol),
           'special_position': sep < special_tol, 'pose_mismatch': ~(dev <= pose_tol), 'zp_units_differ': differ,
           'z_prime_gt1': (zp > 1) & (not allow_zp_gt1)}
    keep = torch.ones(len(passed), dtype=torch.bool)
    for name in ('setting_metric_mismatch', 'special_position', 'pose_mismatch', 'zp_units_differ', 'z_prime_gt1'):
        k = REASONS.index(name)
        rmask[pidx, k] = bad[name]
        for j in torch.nonzero(bad[name] & keep)[:, 0].tolist():
            reason[passed[j]] = name
        keep &= ~bad[name]
    sel = torch.nonzero(keep)[:, 0]
    if len(sel) == 0:
        return empty, _info(reason, rmask, ident, sg_all, zp_all, flags, pose_dev, min_sep, *_no_rows())
    Zm = int(zp[sel].max())
    desc = CrystalDescription.from_parts(sg[sel], cell[sel], t[sel, :Zm], R[sel, :Zm], h[sel, :Zm], zp[sel])
    h_in = torch.where(desc.unit_mask, desc.h, torch.zeros_like(desc.h))
    on_sheet = torch.ones(len(sel), dtype=torch.bool)
    if sheet is not None:
        desc, on_sheet = _to_sheet_units(desc, sheet)
    sohncke = torch.tensor([is_sohncke(s) for s in desc.sg.tolist()], dtype=torch.bool)
    mirrored = torch.zeros(len(sel), dtype=torch.bool)
    enant = torch.zeros(len(sel), dtype=torch.bool)
    sg_source = desc.sg.clone()
    if sheet is not None and mode == 'mirror_reference':
        # D2: the row's own enantiomer as the reference, X' = diag(-1, 1, 1) X at h' = -h (the same atoms)
        mirrored = ~on_sheet & sohncke & (desc.z_prime == 1)
        if bool(mirrored.any()):
            hh = desc.h.clone()
            hh[mirrored, 0] = -hh[mirrored, 0]
            desc = desc.replace(h=hh)
            on_sheet = on_sheet | mirrored
    elif sheet is not None and mode == 'enantiomorph':
        # owner decision 2026-10-21: the mirror-image crystal, reference X, h' = -h, sg' possibly the partner
        enant = ~on_sheet & sohncke & (desc.z_prime == 1)
        if bool(enant.any()):
            idx = torch.nonzero(enant)[:, 0]
            e = enantiomorph(desc[idx])
            desc = desc.replace(sg=desc.sg.index_put((idx,), e.sg), cell=desc.cell.index_put((idx,), e.cell),
                                t=desc.t.index_put((idx,), e.t), R=desc.R.index_put((idx,), e.R),
                                h=desc.h.index_put((idx,), e.h))
            on_sheet = on_sheet | enant
    flip = torch.tensor([-1.0, 1.0, 1.0], dtype=F64)
    mols = [MoleculeRef.from_coords(Xs[j] * flip if bool(mirrored[k]) else Xs[j], zs[j])
            for k, j in enumerate(sel.tolist())]
    eye = torch.eye(3, dtype=F64)
    sP = torch.stack([eye if transforms[passed[j]] is None else torch.as_tensor(transforms[passed[j]].P, dtype=F64)
                      for j in sel.tolist()])
    sp = torch.stack([torch.zeros(3, dtype=F64) if transforms[passed[j]] is None
                      else torch.as_tensor(transforms[passed[j]].p, dtype=F64) for j in sel.tolist()])
    return desc, _info(reason, rmask, ident, sg_all, zp_all, flags, pose_dev, min_sep,
                       pidx[sel], mols, h_in, on_sheet, sohncke, sP, sp, mirrored, enant, sg_source)


def _no_rows():
    """The imported-row fields of ``_info`` for no imported row."""
    return (torch.zeros(0, dtype=torch.long), [], torch.zeros(0, 1, dtype=torch.long), torch.zeros(0, dtype=torch.bool),
            torch.zeros(0, dtype=torch.bool), torch.zeros(0, 3, 3, dtype=F64), torch.zeros(0, 3, dtype=F64),
            torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.long))


def _to_sheet_units(desc: CrystalDescription, sheet: int):
    """``domains.to_sheet`` applied to every unit (each unit's molecule may be replaced by any of its G images)."""
    B, Z = desc.t.shape[:2]
    m = desc.unit_mask
    flat = CrystalDescription.from_parts(desc.sg[:, None].expand(B, Z)[m], desc.cell[:, None].expand(B, Z, 6)[m],
                                         desc.t[m], desc.R[m], desc.h[m])
    moved, ok = to_sheet(flat, sheet)
    t, R, h = desc.t.clone(), desc.R.clone(), desc.h.clone()
    t[m], R[m], h[m] = moved.t[:, 0], moved.R[:, 0], moved.h[:, 0]
    ok_rows = torch.ones(B, dtype=torch.bool)
    ok_full = torch.ones(B, Z, dtype=torch.bool)
    ok_full[m] = ok
    ok_rows &= ok_full.all(1)
    return desc.replace(t=t, R=R, h=h), ok_rows


def _info(reason, rmask, ident, sg, zp, flags, pose_dev, min_sep, row, mols, h_in, on_sheet, sohncke, sP, sp,
          mirrored, enant, sg_source):
    nonstd, reord, reexp, well, metric_dev = flags
    imported = torch.tensor([r is None for r in reason], dtype=torch.bool)
    return ImportInfo(imported=imported, reason=list(reason), reason_mask=rmask, identifier=ident, sg=sg,
                      z_prime=zp, stored_nonstandard=nonstd, ops_reordered=reord, setting_reexpressed=reexp,
                      well_defined=well, pose_dev=pose_dev, min_centroid_sep=min_sep, metric_dev=metric_dev,
                      row=row, mols=mols,
                      zp_gt1=(zp[row] > 1) if len(row) else torch.zeros(0, dtype=torch.bool),
                      h_input=h_in, on_sheet=on_sheet, sohncke=sohncke, setting_P=sP, setting_p=sp,
                      mirrored=mirrored, enantiomorph=enant, sg_source=sg_source)

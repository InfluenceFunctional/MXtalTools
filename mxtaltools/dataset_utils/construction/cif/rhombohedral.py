"""R space groups filed on rhombohedral axes, re-expressed on hexagonal axes at read.

`SYM_OPS` lists the seven R groups (146, 148, 155, 160, 161, 166, 167) on hexagonal axes in
the obverse setting: |G| operators, a third of them composed with each centring translation
(0, 0, 0), (2/3, 1/3, 1/3), (1/3, 2/3, 2/3). A file on rhombohedral axes describes the same
crystal on the primitive cell (a = b = c, alpha = beta = gamma) with |G| / 3 operators, a count
that is neither the table's nor a centred multiple of it, so `space_group_from_block` refuses
it. In the CSD dump that is 467 of 873,578 CIFs, every one with a ``*`` Hall symbol.

`reexpress_rhombohedral_axes` turns such a block into a copy that states the hexagonal
description, so every later stage of both readers (`read_cif_block`, `MolCrystalData.read`)
runs on it unchanged:

    a_H = a_R - b_R,  b_H = b_R - c_R,  c_H = a_R + b_R + c_R     (P below, ITA Table 5.1.3.1)
    x_H = P^-1 x_R                                               (the same points)
    operators SYM_OPS[n] in table order; Z and the cell volume times 3

It re-expresses only when the file's cell is rhombohedral, its Hall symbol (where present) is
the rhombohedral-axes one, and its operators, carried onto the new axes and composed with the
three centring translations, are the set ``SYM_OPS[n]`` (equal rotations, translations equal
modulo 1 to 1e-6). These are the checks against the file. Two more checks follow, and they
guard this module's arithmetic, not the file: every image of every deposited site under the
file's operators coincides with one of its images under ``SYM_OPS[n]`` on the new axes and
back (to 1e-5 Angstrom), and the new cell's metric is the old one carried by P. Once the
operators match, both hold by construction from ``_P``, ``_P_INV3``, ``_CENTRING`` and the
cell formulae (up to SYM_OPS's 8-decimal 2/3), so they fail only on a defect in the transform
itself (a wrong matrix or cell formula; the tests inject one). Values are written into the
copy with ``repr``, so they parse back to the same floats; the operators parse back
bit-identical to ``SYM_OPS[n]``.

Operators that cannot be read (including a term that overflows a float) leave the block as it
is: the regular read meets the same failure and refuses the row as it did before this step.
"""

from typing import Optional, Tuple

import numpy as np

from mxtaltools.constants.space_group_info import SPACE_GROUPS, SYM_OPS

from .errors import CifInconsistentSymmetryError, CifReadError
from .sites import _FRACT_TAGS, _LABEL_TAG
from .symmetry import _HM_TAGS, _NUM_TAGS, _OP_TAGS, parse_symop, symops_from_block
from .tokenizer import CifBlock, CifLoop, as_float, as_int, is_missing, strip_esd

__all__ = ['reexpress_rhombohedral_axes', 'HALL_SYMBOLS']

#: Hall symbols of the R groups: (rhombohedral axes, hexagonal axes obverse), as spglib 2.7 names them.
HALL_SYMBOLS = {146: ('P 3*', 'R 3'), 148: ('-P 3*', '-R 3'), 155: ('P 3* 2', 'R 3 2"'),
                160: ('P 3* -2', 'R 3 -2"'), 161: ('P 3* -2n', 'R 3 -2"c'),
                166: ('-P 3* 2', '-R 3 2"'), 167: ('-P 3* 2n', '-R 3 2"c')}

#: Columns are the hexagonal axes in the rhombohedral basis (a_H = a_R - b_R, ...); x_R = P x_H.
_P = np.array([[1., 0., 1.], [-1., 1., 1.], [0., -1., 1.]])
#: 3 P^-1, an integer matrix: x_H = (_P_INV3 @ x_R) / 3.
_P_INV3 = np.array([[2., -1., -1.], [1., 1., -2.], [1., 1., 1.]])
#: The obverse centring translations on hexagonal axes (the images of the rhombohedral lattice).
_CENTRING = np.array([[0., 0., 0.], [2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]])

_CELL_TAGS = ('_cell_length_a', '_cell_length_b', '_cell_length_c',
              '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')
_Z_TAG = '_cell_formula_units_z'           # CifBlock keys are lower-cased
_VOLUME_TAG = '_cell_volume'
_HALL_TAGS = ('_space_group_name_hall', '_symmetry_space_group_name_hall')
_CELL_SETTING_TAG = '_symmetry_cell_setting'
_ANISO_PREFIX = '_atom_site_aniso_'

_OP_TOL = 1e-6                              # translations modulo 1, as the readers compare operators
_REBUILD_TOL = 1e-5                         # Angstrom; SYM_OPS holds 2/3 to 8 decimals (3e-9 fractional)
_METRIC_TOL = 1e-9                          # relative to a_R ** 2

_OP_TEXT = {}


def reexpress_rhombohedral_axes(block: CifBlock,
                                identifier: Optional[str] = None) -> Tuple[CifBlock, Optional[str]]:
    """An R group filed on rhombohedral axes, as a block on hexagonal axes (obverse).

    Parameters
    ----------
    block : CifBlock
        One parsed data block; never modified.
    identifier : str, optional
        Used in the note and in error messages only.

    Returns
    -------
    (CifBlock, str or None)
        ``(block, None)``, the same object, unless the block declares an R group (IT number,
        else the H-M symbol, as `space_group_from_block` reads them) and lists a third of
        ``len(SYM_OPS[n])`` operators. ``(block, None)`` also when the operators, the cell,
        Z or the fractional coordinates cannot be read: the regular read then refuses the
        block as it did before this step existed.

        Otherwise a new block with the same name, which shares the unchanged loops, and a
        one-line note saying what was re-expressed. In the new block: the cell lengths and
        angles (a_H = b_H = 2 a sin(alpha / 2), c_H = a sqrt(3 (1 + 2 cos alpha)), angles
        90, 90, 120 degrees); ``_cell_volume`` and ``_cell_formula_units_Z`` times 3 where
        present; the atom-site loop's fractional coordinates; the operator loop, replaced
        by one column, under the file's operator tag, holding ``SYM_OPS[n]`` in table
        order; the Hall symbol where present; ``_symmetry_cell_setting``, where present,
        ``trigonal``. The ``_atom_site_aniso_*`` loop (displacement tensors on the old axes;
        no reader interprets them) and any other operator loop are dropped. Other tags and
        loops are kept as written: the H-M symbol (the same for both axes), and the
        ``_geom_*`` site-symmetry codes, which still name the file's operators (both
        readers drop a bond to a symmetry image).

    Raises
    ------
    CifInconsistentSymmetryError
        The block lists the rhombohedral operator count but: a Hall symbol is present and
        is not the rhombohedral-axes one for ``n``; the cell is not rhombohedral (a, b, c
        or alpha, beta, gamma not equal as read, or 1 + 2 cos alpha <= 0); the operators
        on the new axes are not ``SYM_OPS[n]`` (another origin or setting); or the rebuild
        or metric check fails (a defect in this module's transform; see the module notes).
    """
    number = _number(block)
    if number not in HALL_SYMBOLS:
        return block, None
    try:
        ops, op_tag = symops_from_block(block, identifier)
    except (CifReadError, ValueError, OverflowError):  # e.g. '+1e400': the regular read refuses it as before
        return block, None
    op_loop = block.loop_with(op_tag)
    if 3 * len(ops) != len(SYM_OPS[number]) or op_loop is None:
        return block, None

    cell = _cell(block)
    z_value = _z_value(block)
    site_loop, frac_r = _fractional(block)
    if cell is None or z_value is False or frac_r is None:
        return block, None

    name = f'{number} ({SPACE_GROUPS[number]})'
    hall_tag, hall = _first_present(block, _HALL_TAGS)
    if hall is not None and _squash(hall) != _squash(HALL_SYMBOLS[number][0]):
        raise CifInconsistentSymmetryError(
            f'space group {name} lists {len(ops)} operators, the count on rhombohedral axes, but its Hall '
            f'symbol {hall!r} is not the rhombohedral-axes symbol {HALL_SYMBOLS[number][0]!r}', identifier)
    a, b, c, alpha, beta, gamma = cell
    cos_alpha = np.cos(np.radians(alpha))
    if not (a == b == c and alpha == beta == gamma) or 1 + 2 * cos_alpha <= 0:
        raise CifInconsistentSymmetryError(
            f'space group {name} lists {len(ops)} operators, the count on rhombohedral axes, but the cell '
            f'{list(cell)} is not rhombohedral (a = b = c, alpha = beta = gamma < 120 degrees)', identifier)

    ops_r = np.stack(ops)
    ops_h = np.stack([np.asarray(o, dtype=np.float64) for o in SYM_OPS[number]])
    problem = _operator_mismatch(ops_r, ops_h)
    if problem is not None:
        raise CifInconsistentSymmetryError(
            f'space group {name} is on rhombohedral axes, but its {len(ops)} operators carried onto '
            f'hexagonal axes (obverse) are not SYM_OPS[{number}]: {problem}', identifier)

    a_h, c_h = _hexagonal_lengths(a, alpha)
    frac_h = _hexagonal_frac(frac_r)
    metric_dev = _metric_deviation(a, alpha, a_h, c_h)
    rebuild_dev = _rebuild_deviation(frac_r, ops_r, frac_h, ops_h, _metric(a, a, a, alpha, alpha, alpha))
    if not (metric_dev <= _METRIC_TOL and rebuild_dev <= _REBUILD_TOL):
        raise CifInconsistentSymmetryError(
            f'space group {name}: the hexagonal-axes description does not rebuild the rhombohedral one '
            f'(metric deviation {metric_dev:.2e} relative, image deviation {rebuild_dev:.2e} A)', identifier)

    out = CifBlock(block.name)
    out.tags = dict(block.tags)
    for tag, value in zip(_CELL_TAGS, (a_h, a_h, c_h, 90.0, 90.0, 120.0)):
        out.tags[tag] = repr(float(value))
    volume = _volume(block)
    if volume is not None:
        out.tags[_VOLUME_TAG] = repr(3 * volume)
    if z_value is not None:
        out.tags[_Z_TAG] = str(3 * z_value)
    if hall_tag is not None:
        out.tags[hall_tag] = HALL_SYMBOLS[number][1]
    if _CELL_SETTING_TAG in out.tags:
        out.tags[_CELL_SETTING_TAG] = 'trigonal'

    written_op_tag = next(t for t in op_loop.tags if t.lower() == op_tag.lower())
    fract_cols = [[t.lower() for t in site_loop.tags].index(f) for f in _FRACT_TAGS]
    for loop in block.loops:
        if loop is op_loop:
            out.loops.append(CifLoop([written_op_tag], [[s] for s in _standard_op_text(number)]))
        elif loop is site_loop:
            rows = []
            for row, x in zip(loop.rows, frac_h):
                row = list(row)
                for col, v in zip(fract_cols, x):
                    row[col] = repr(float(v))
                rows.append(row)
            out.loops.append(CifLoop(list(loop.tags), rows))
        elif any(t.lower().startswith(_ANISO_PREFIX) or t.lower() in _OP_TAGS for t in loop.tags):
            continue                        # old-axes displacement tensors; a second operator loop
        else:
            out.loops.append(loop)

    z_text = 'absent' if z_value is None else f'{z_value} -> {3 * z_value}'
    note = (f're-expressed from rhombohedral axes (a = b = c = {a!r} A, alpha = beta = gamma = {alpha!r} deg, '
            f'{len(ops)} operators, Hall {hall!r}) to hexagonal axes, obverse (a_H = a - b, b_H = b - c, '
            f'c_H = a + b + c; x_H = P^-1 x; operators SYM_OPS[{number}], {len(ops_h)}; Z {z_text}); '
            f'images rebuilt to {rebuild_dev:.1e} A')
    return out, note


def _number(block) -> Optional[int]:
    """The IT number as `space_group_from_block` reads it, or None when it cannot be read."""
    _, raw = block.first_of(*_NUM_TAGS)
    if not is_missing(raw):
        try:
            return as_int(raw)
        except (CifReadError, ValueError, OverflowError):
            return None
    _, symbol = block.first_of(*_HM_TAGS)
    if symbol is None:
        return None
    symbol = symbol.strip().strip("'\"")
    if is_missing(symbol):
        return None
    squashed = symbol.replace(' ', '')
    matches = [k for k, v in SPACE_GROUPS.items() if v.replace(' ', '') == squashed]
    return matches[0] if matches else None


def _cell(block):
    """(a, b, c, alpha, beta, gamma) as read (Angstrom, degrees), or None when any is absent or unreadable."""
    values = []
    for tag in _CELL_TAGS:
        try:
            v = as_float(block.get(tag))
        except CifReadError:
            return None
        if v is None or not np.isfinite(v) or v <= 0:
            return None
        values.append(v)
    return tuple(values)


def _z_value(block):
    """``_cell_formula_units_Z``: a positive int, None when absent, False when unreadable or not positive."""
    raw = block.get(_Z_TAG)
    if raw is None or is_missing(raw):
        return None
    try:
        z = as_int(raw)
    except (CifReadError, ValueError, OverflowError):
        return False
    return z if z > 0 else False


def _volume(block):
    """``_cell_volume`` as a float, or None when absent or unreadable."""
    try:
        v = as_float(block.get(_VOLUME_TAG))
    except CifReadError:
        return None
    return v if v is not None and np.isfinite(v) else None


def _fractional(block):
    """(atom-site loop, [n, 3] float64), the loop `sites_from_block` reads; (None, None) when unreadable."""
    loop = block.loop_with(_LABEL_TAG)
    if loop is None:
        for t in _FRACT_TAGS:
            loop = block.loop_with(t)
            if loop is not None:
                break
    if loop is None or len(loop) == 0 or any(t not in loop for t in _FRACT_TAGS):
        return None, None
    frac = np.empty((len(loop), 3), dtype=np.float64)
    for j, t in enumerate(_FRACT_TAGS):
        for i, v in enumerate(loop.column(t)):
            if is_missing(v):
                return None, None
            try:
                frac[i, j] = float(strip_esd(v))
            except ValueError:
                return None, None
    if not np.isfinite(frac).all():
        return None, None
    return loop, frac


def _first_present(block, tags):
    """(lower-cased tag, value with quotes stripped) of the first present, non-missing tag, else (None, None)."""
    for t in tags:
        v = block.get(t)
        if v is not None and not is_missing(v):
            return t.lower(), v.strip().strip("'\"")
    return None, None


def _squash(symbol: str) -> str:
    return symbol.replace(' ', '').lower()


def _operator_mismatch(ops_r: np.ndarray, ops_h: np.ndarray) -> Optional[str]:
    """None when ``ops_r`` [m, 4, 4] on hexagonal axes, times the centring, is the set ``ops_h`` [3m, 4, 4]."""
    rot = np.einsum('ij,mjk,kl->mil', _P_INV3, ops_r[:, :3, :3], _P) / 3
    if np.abs(rot - np.round(rot)).max() > 1e-9:
        return 'their rotations are not integer matrices on the new axes'
    rot = np.round(rot)
    trans = ops_r[:, :3, 3] @ _P_INV3.T / 3
    cand_rot = np.repeat(rot, 3, axis=0)
    cand_t = (trans[:, None, :] + _CENTRING[None]).reshape(-1, 3)
    same_rot = np.abs(cand_rot[:, None] - ops_h[None, :, :3, :3]).max((-1, -2)) <= 1e-9
    dt = cand_t[:, None, :] - ops_h[None, :, :3, 3]
    match = same_rot & (np.abs(dt - np.round(dt)).max(-1) <= _OP_TOL)
    if (match.sum(1) == 1).all() and (match.sum(0) == 1).all():
        return None
    return f'{int((match.sum(0) == 0).sum())} of the table operators have no counterpart'


def _hexagonal_lengths(a, alpha):
    """(a_H, c_H) in Angstrom of the hexagonal cell of the rhombohedral cell (a, alpha in degrees)."""
    return 2 * a * np.sin(np.radians(alpha) / 2), a * np.sqrt(3 * (1 + 2 * np.cos(np.radians(alpha))))


def _hexagonal_frac(frac_r: np.ndarray) -> np.ndarray:
    """[n, 3] fractional coordinates on hexagonal axes of [n, 3] on rhombohedral axes: x_H = P^-1 x_R."""
    return frac_r @ _P_INV3.T / 3


def _metric(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Metric tensor (Gram matrix of the cell vectors), lengths in Angstrom, angles in degrees."""
    ca, cb, cg = np.cos(np.radians([alpha, beta, gamma]))
    return np.array([[a * a, a * b * cg, a * c * cb],
                     [a * b * cg, b * b, b * c * ca],
                     [a * c * cb, b * c * ca, c * c]])


def _metric_deviation(a, alpha, a_h, c_h) -> float:
    """Largest entry of |G_H - P^T G_R P|, relative to a ** 2."""
    g_r = _metric(a, a, a, alpha, alpha, alpha)
    g_h = _metric(a_h, a_h, c_h, 90.0, 90.0, 120.0)
    return float(np.abs(g_h - _P.T @ g_r @ _P).max() / (a * a))


def _rebuild_deviation(frac_r, ops_r, frac_h, ops_h, g_r) -> float:
    """Two-way largest distance (Angstrom) between each site's images in the two descriptions.

    The images on hexagonal axes are carried back to the rhombohedral basis (x_R = P x_H);
    distances are taken modulo the rhombohedral lattice (fractional differences rounded),
    each image to the nearest image of the same site in the other description.
    """
    img_r = np.einsum('mij,nj->nmi', ops_r[:, :3, :3], frac_r) + ops_r[None, :, :3, 3]
    img_h = np.einsum('mij,nj->nmi', ops_h[:, :3, :3], frac_h) + ops_h[None, :, :3, 3]
    back = img_h @ _P.T
    d = back[:, :, None, :] - img_r[:, None, :, :]
    d = d - np.round(d)
    dist = np.sqrt(np.maximum(np.einsum('...i,ij,...j->...', d, g_r, d), 0.0))
    return float(max(dist.min(axis=2).max(), dist.min(axis=1).max()))


def _standard_op_text(number: int):
    """``SYM_OPS[number]`` as operator strings that `parse_symop` reads back bit-identical."""
    text = _OP_TEXT.get(number)
    if text is None:
        text = []
        for op in SYM_OPS[number]:
            op = np.asarray(op, dtype=np.float64)
            s = _op_text(op)
            if not np.array_equal(parse_symop(s), op):
                raise RuntimeError(f'operator text {s!r} does not parse back to SYM_OPS[{number}]: a defect here')
            text.append(s)
        _OP_TEXT[number] = text
    return text


def _op_text(op: np.ndarray) -> str:
    """One [4, 4] operator as ``'x-y,x,z+0.66666667'``: coefficients and translations written with ``repr``."""
    rows = []
    for i in range(3):
        parts = []
        for j, axis in enumerate('xyz'):
            k = float(op[i, j])
            if k == 1:
                parts.append('+' + axis)
            elif k == -1:
                parts.append('-' + axis)
            elif k != 0:
                parts.append(('+' if k > 0 else '-') + repr(abs(k)) + '*' + axis)
        t = float(op[i, 3])
        if t != 0:
            parts.append(('+' if t > 0 else '-') + repr(abs(t)))
        rows.append(''.join(parts).lstrip('+'))
    return ','.join(rows)

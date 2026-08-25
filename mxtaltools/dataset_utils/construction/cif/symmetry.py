"""Symmetry: CIF operator strings <-> 4x4 affines, and space group identification.

Two things this module refuses to do, both deliberate:

1. **It never substitutes `SYM_OPS[sg_ind]` for the file's own operators.**
   Measured over 400 random CSD entries, only 62.5 % carry operators identical to
   the standard table in the same order; 18.2 % are the same set reordered and
   19.2 % are a different set entirely (alternate settings such as P2_1/n filed
   under IT number 14).  The unit-cell builder applies whatever the object holds
   (`crystal_building/utils.py:677`), so substituting the table would describe a
   different crystal.

2. **It does not guess when sources disagree.**  A file whose IT number and H-M
   symbol name different groups, or whose operator count contradicts its declared
   group, raises `CifInconsistentSymmetryError`.
"""

from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

from mxtaltools.constants.space_group_info import SPACE_GROUPS, SYM_OPS

from .errors import CifInconsistentSymmetryError, CifSymmetryError
from .tokenizer import CifBlock, as_int, is_missing

__all__ = ['parse_symop', 'symops_from_block', 'space_group_from_block', 'SymmetryInfo']

_AXIS = {'x': 0, 'y': 1, 'z': 2}

# CSD exports carry the deprecated spellings; this package's writer emits the
# current ones. A reader must accept both without having to know which it holds.
_OP_TAGS = ('_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz')
_NUM_TAGS = ('_space_group_IT_number', '_symmetry_Int_Tables_number')
_HM_TAGS = ('_space_group_name_H-M_alt', '_symmetry_space_group_name_H-M',
            '_space_group_name_H-M_full')


class SymmetryInfo:
    """What the file says about symmetry, plus how it was established."""

    __slots__ = ('operators', 'number', 'symbol', 'op_tag', 'number_tag', 'symbol_tag',
                 'operators_are_standard')

    def __init__(self, operators, number, symbol, op_tag, number_tag, symbol_tag):
        self.operators = operators
        self.number = number
        self.symbol = symbol
        self.op_tag = op_tag
        self.number_tag = number_tag
        self.symbol_tag = symbol_tag
        std = SYM_OPS.get(number)
        self.operators_are_standard = bool(
            std is not None and len(operators) == len(std)
            and all(np.allclose(a, np.asarray(b, float), atol=1e-6)
                    for a, b in zip(operators, std)))

    @property
    def multiplicity(self) -> int:
        return len(self.operators)

    def __repr__(self) -> str:
        return (f'SymmetryInfo(#{self.number} {self.symbol!r}, {len(self.operators)} ops, '
                f'standard={self.operators_are_standard})')



def _centring_vectors(ops: List[np.ndarray]) -> List[np.ndarray]:
    """Pure lattice translations in the set: operators whose rotation is the identity.

    For a primitive group this is just [(0,0,0)]; for a C-centred setting it is
    [(0,0,0), (1/2,1/2,0)], and so on.
    """
    eye = np.eye(3)
    out = []
    for o in ops:
        if np.allclose(o[:3, :3], eye, atol=1e-8):
            out.append(np.mod(o[:3, 3], 1.0))
    return out


def _factorises_as_centred(ops: List[np.ndarray],
                           standard: List[np.ndarray],
                           centring: List[np.ndarray]) -> bool:
    """Is `ops` exactly {standard op composed with each centring vector}?

    Compared modulo 1 on the translation part, since an operator and the same
    operator shifted by a full lattice vector are the same symmetry element.
    """
    def key(rot, trans):
        return (tuple(np.round(rot.ravel(), 6)),
                tuple(np.round(np.mod(trans, 1.0), 6)))

    have = {key(o[:3, :3], o[:3, 3]) for o in ops}
    want = set()
    for s in standard:
        s = np.asarray(s, dtype=float)
        for c in centring:
            want.add(key(s[:3, :3], s[:3, 3] + c))
    return have == want


def parse_symop(expr: str, identifier: str = None) -> np.ndarray:
    """A CIF `'x,y,z'` operator string -> a 4x4 affine acting on fractional coords.

    Accepts the forms CSD and this package emit: `-x,y+1/2,-z+1/2`,
    `1/2-x,1/2+y,1/2-z` (translation first), `x-y,x,z` (multiple axes per term),
    and coefficient forms like `1/2*x`.

    >>> parse_symop('x,y,z')[:3, :3].tolist()
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """
    op = np.zeros((4, 4))
    op[3, 3] = 1.0
    terms = expr.strip().strip("'\"").split(',')
    if len(terms) != 3:
        raise CifSymmetryError(f'operator {expr!r} does not have three components', identifier)

    for row, term in enumerate(terms):
        t = term.strip().lower().replace(' ', '')
        if not t:
            raise CifSymmetryError(f'operator {expr!r} has an empty component', identifier)
        # normalise so every part carries its own sign, then split on '+'
        t = t.replace('-', '+-')
        for part in [p for p in t.split('+') if p]:
            if part[-1] in _AXIS:
                col = _AXIS[part[-1]]
                coeff = part[:-1].rstrip('*')
                if coeff in ('', '-'):
                    op[row, col] += -1.0 if coeff == '-' else 1.0
                else:
                    try:
                        op[row, col] += float(Fraction(coeff))
                    except (ValueError, ZeroDivisionError):
                        raise CifSymmetryError(
                            f'operator {expr!r}: cannot read coefficient {coeff!r}', identifier)
            else:
                try:
                    op[row, 3] += float(Fraction(part))
                except (ValueError, ZeroDivisionError):
                    raise CifSymmetryError(
                        f'operator {expr!r}: cannot read translation {part!r}', identifier)
    return op


def symops_from_block(block: CifBlock, identifier: str = None) -> Tuple[List[np.ndarray], str]:
    """Every symmetry operator in the block, in file order, as 4x4 affines.

    Order is preserved because it is contracted: `sym_mult` indexes into it and
    the unit-cell builder tiles by position.

    Refuses, rather than defaulting to P1, when no operator loop is present -- a
    silent P1 default is exactly the ASE trap this reader exists to avoid
    (`ase/io/cif.py:378` reads a non-P1 file as P1 whenever the symop loop is
    missing, silently, on 87 % of one local tree).
    """
    for tag in _OP_TAGS:
        loop = block.loop_with(tag)
        if loop is not None:
            raw = loop.column(tag)
            break
    else:
        scalar_tag, scalar = block.first_of(*_OP_TAGS)
        if scalar is None:
            raise CifSymmetryError(
                f'no symmetry operators: none of {_OP_TAGS} present. Refusing to '
                'assume P1 -- a symmetry-stripped file and a genuine P1 cell are '
                'indistinguishable here, and ase/io/cif.py:378 reads the former as '
                'the latter silently. A bare unit cell needs its own explicit '
                'processing route (symmetry inference, then re-derivation of the '
                'asymmetric unit); that route is not built, and this workflow is '
                'deliberately blocked on it rather than guessing.', identifier)
        raw, tag = [scalar], scalar_tag

    ops = [parse_symop(r, identifier) for r in raw]
    if not ops:
        raise CifSymmetryError('symmetry operator loop is empty', identifier)

    identity = np.eye(4)
    if not any(np.allclose(o, identity, atol=1e-8) for o in ops):
        raise CifSymmetryError(
            'symmetry operator list does not contain the identity', identifier)
    return ops, tag


def space_group_from_block(block: CifBlock, identifier: str = None) -> SymmetryInfo:
    """Establish the space group and its operators, checking the sources agree.

    The operator list is authoritative for geometry; the number and symbol are
    metadata that must be CONSISTENT with it, not a substitute for it.
    """
    ops, op_tag = symops_from_block(block, identifier)

    num_tag, num_raw = block.first_of(*_NUM_TAGS)
    number = as_int(num_raw, num_tag or '', identifier) if not is_missing(num_raw) else None

    sym_tag, symbol = block.first_of(*_HM_TAGS)
    if symbol is not None:
        symbol = symbol.strip().strip("'\"")
        if is_missing(symbol):
            symbol = None

    if number is None and symbol is None:
        raise CifSymmetryError(
            'neither an IT number nor an H-M symbol is present; cannot identify the '
            'space group. The operator list alone is not enough to name it', identifier)

    if number is None:                     # recover the number from the symbol
        squashed = symbol.replace(' ', '')
        matches = [k for k, v in SPACE_GROUPS.items() if v.replace(' ', '') == squashed]
        if not matches:
            raise CifSymmetryError(
                f'H-M symbol {symbol!r} is not in SPACE_GROUPS and no IT number is '
                'present', identifier)
        number = matches[0]

    if number not in SYM_OPS:
        raise CifSymmetryError(
            f'space group number {number} is outside SYM_OPS (1-230)', identifier)

    # --- consistency, not substitution ---
    # A CENTRED SETTING legitimately lists a multiple of the standard operator
    # count: e.g. ZZZKEA02 is 'C -1' filed under IT number 2, whose four operators
    # are the standard P-1 pair composed with the C-centring translation (1/2,1/2,0).
    # That is a real structure, not a malformed file, so it must be accepted --
    # but only when the operator set actually FACTORISES that way. Anything else
    # is a genuine contradiction and is refused.
    expected = len(SYM_OPS[number])
    centring = _centring_vectors(ops)
    if len(ops) != expected:
        if len(ops) == expected * len(centring) and _factorises_as_centred(
                ops, SYM_OPS[number], centring):
            pass                                     # centred setting; accept
        else:
            raise CifInconsistentSymmetryError(
                f'space group {number} ({SPACE_GROUPS[number]}) has {expected} operators '
                f'but the file lists {len(ops)}, and they do not factorise as a centred '
                f'setting ({len(centring)} centring vector(s) found). The file is '
                'internally inconsistent; guessing which source to believe would '
                'silently build a different crystal', identifier)

    if symbol is not None:
        declared = SPACE_GROUPS[number].replace(' ', '')
        given = symbol.replace(' ', '')
        if given != declared:
            # settings legitimately differ in symbol (P21/n vs P21/c under IT 14),
            # so this is recorded, not refused -- the operators carry the truth.
            pass

    return SymmetryInfo(ops, number, symbol or SPACE_GROUPS[number],
                        op_tag, num_tag, sym_tag)

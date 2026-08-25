"""The atom-site loop: labels, elements, fractional coordinates, occupancy.

Vocabulary note (design §1.5): this returns the **deposited set** -- every
`_atom_site` row exactly as written.  That is NOT necessarily the
crystallographic asymmetric unit: at Z' < 1, which is 19.6 % of the CSD, the
exporter writes a whole molecule sitting on a special position and the true ASU
is a fraction of it.  Nothing here assumes otherwise.
"""

from typing import List, Optional

import numpy as np

from .errors import CifMissingDataError, CifParseError, CifUnsupportedError
from .tokenizer import CifBlock, as_float, is_missing, strip_esd

__all__ = ['SiteSet', 'sites_from_block', 'element_from_symbol']

_LABEL_TAG = '_atom_site_label'
_SYMBOL_TAG = '_atom_site_type_symbol'
_FRACT_TAGS = ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')
_OCC_TAG = '_atom_site_occupancy'

_SYMBOLS = None


def _symbol_table():
    global _SYMBOLS
    if _SYMBOLS is None:
        from ase.data import chemical_symbols
        _SYMBOLS = {s.upper(): i for i, s in enumerate(chemical_symbols) if s}
    return _SYMBOLS


def element_from_symbol(raw: str, identifier: str = None) -> int:
    """`'C'`, `'Cl'`, `'Fe3+'`, `'O-'` -> atomic number.

    CIF `_atom_site_type_symbol` may carry an oxidation state suffix, and CSD
    writes both `Fe3+` and `Fe`.  The leading alphabetic run is the element.
    Raises rather than defaulting to carbon -- a mis-typed element is precisely
    the silent wrong answer this reader exists to avoid.
    """
    s = raw.strip().strip("'\"")
    alpha = ''
    for ch in s:
        if ch.isalpha():
            alpha += ch
        else:
            break
    if not alpha:
        raise CifParseError(f'atom site type symbol {raw!r} has no element', identifier)
    table = _symbol_table()
    key = alpha.capitalize().upper()
    if key in table:
        return table[key]
    # 'D' (deuterium) is a legal CIF symbol and is hydrogen
    if key == 'D':
        return 1
    raise CifParseError(f'unknown element symbol {alpha!r} (from {raw!r})', identifier)


class SiteSet:
    """The deposited set: parallel arrays, one entry per `_atom_site` row."""

    __slots__ = ('labels', 'z', 'frac', 'occupancy', 'has_occupancy')

    def __init__(self, labels, z, frac, occupancy, has_occupancy):
        self.labels = labels                    # list[str], the loop's own labels
        self.z = z                              # (n,) int   atomic numbers
        self.frac = frac                        # (n, 3) float fractional coordinates
        self.occupancy = occupancy              # (n,) float, all 1.0 when absent
        self.has_occupancy = has_occupancy      # was the tag actually present?

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_heavy(self) -> int:
        return int((self.z > 1).sum())

    def __repr__(self) -> str:
        return f'SiteSet({len(self)} sites, {self.n_heavy} heavy)'


def sites_from_block(block: CifBlock,
                     identifier: str = None,
                     max_atomic_number: Optional[int] = None) -> SiteSet:
    """Read the atom-site loop into a `SiteSet`.

    Refusals, all named rather than silent:
      - no atom-site loop, or no fractional coordinates -> `CifMissingDataError`
      - a site with a missing coordinate -> `CifMissingDataError` (NOT dropped:
        silently dropping a site changes the composition)
      - duplicate site labels -> `CifParseError`, since `_geom_bond` refers to
        sites BY LABEL and a duplicate makes the bond graph ambiguous
      - partial occupancy -> `CifUnsupportedError(reason='partial-occupancy')`,
        because this reader has no disorder model

    Occupancy note: the CSD exporter strips `_atom_site_occupancy` entirely --
    measured 0/1000 files carry it -- so its ABSENCE means "not stated", never
    "fully occupied". `has_occupancy` records which, so a caller can tell a
    checked full occupancy from an unchecked one.
    """
    loop = block.loop_with(_LABEL_TAG)
    if loop is None:
        for t in _FRACT_TAGS:
            loop = block.loop_with(t)
            if loop is not None:
                break
    if loop is None:
        raise CifMissingDataError(f'no atom-site loop ({_LABEL_TAG} absent)', identifier)

    for t in _FRACT_TAGS:
        if t not in loop:
            raise CifMissingDataError(
                f'atom-site loop has no {t}; only fractional coordinates are supported',
                identifier)

    n = len(loop)
    if n == 0:
        raise CifMissingDataError('atom-site loop is empty', identifier)

    # NOTE: do NOT strip quotes here. The tokenizer already removed real quoting,
    # and a TRAILING APOSTROPHE IS PART OF THE LABEL: the CSD writes S1 / S1' for
    # corresponding atoms in the two symmetry-independent molecules of a Z'=2
    # structure. Stripping it collapses them into a false duplicate.
    labels = ([v.strip() for v in loop.column(_LABEL_TAG)]
              if _LABEL_TAG in loop else [f'site{i}' for i in range(n)])
    if len(set(labels)) != len(labels):
        dupes = sorted({l for l in labels if labels.count(l) > 1})[:5]
        raise CifParseError(
            f'duplicate atom site labels {dupes}; _geom_bond refers to sites by '
            'label, so the bond graph would be ambiguous', identifier)

    if _SYMBOL_TAG in loop:
        z = np.array([element_from_symbol(v, identifier) for v in loop.column(_SYMBOL_TAG)],
                     dtype=np.int64)
    else:
        # fall back to the leading alphabetic run of the label ('C12' -> C)
        z = np.array([element_from_symbol(l, identifier) for l in labels], dtype=np.int64)

    frac = np.empty((n, 3), dtype=np.float64)
    for j, t in enumerate(_FRACT_TAGS):
        col = loop.column(t)
        for i, v in enumerate(col):
            if is_missing(v):
                raise CifMissingDataError(
                    f'site {labels[i]!r} has no {t}; refusing to drop it, since that '
                    'would silently change the composition', identifier)
            frac[i, j] = float(strip_esd(v))

    has_occ = _OCC_TAG in loop
    if has_occ:
        occ = np.array([1.0 if is_missing(v) else float(strip_esd(v))
                        for v in loop.column(_OCC_TAG)], dtype=np.float64)
        partial = np.abs(occ - 1.0) > 1e-3
        if partial.any():
            bad = [labels[i] for i in np.where(partial)[0][:5]]
            raise CifUnsupportedError(
                f'{int(partial.sum())} site(s) carry partial occupancy (e.g. {bad}); '
                'this reader has no disorder model',
                reason='partial-occupancy', identifier=identifier)
    else:
        occ = np.ones(n, dtype=np.float64)

    if max_atomic_number is not None and z.max() > max_atomic_number:
        offenders = sorted(set(z[z > max_atomic_number].tolist()))
        raise CifUnsupportedError(
            f'atomic numbers {offenders} exceed max_atomic_number={max_atomic_number}',
            reason='atomic-number', identifier=identifier)

    return SiteSet(labels, z, frac, occ, has_occ)

"""Composition: `_chemical_formula_moiety` and per-component formulas.

**Nothing in the reader acts on this yet.** It is captured because cocrystal
support is a stated future requirement that needs a `MolCrystalData` refactor,
and the information a cocrystal implementation will need is in the file *now*.
Discarding it and re-parsing later would mean re-deriving what the depositor
already stated.

What the tag gives, measured over 400 random CSD entries -- present in **100 %**
of them, and multi-unit in 40 %::

    60.0 %   one moiety      'C21 H28 N2 O2'
    27.2 %   two moieties    'C7 H8 N4 O2,C6 H7 B1 O3'          <- a cocrystal
    11.2 %   three
     1.5 %   four

It encodes three things a component decomposition cannot::

    stoichiometry   '2(H2 O1)'      a dihydrate
    formal charge   'C2 H5 N4 1+'   a salt, not a neutral cocrystal
    identity        the depositor's own statement of which species are present

The cocrystal generalisation of the reader's current `components == Z'` check is
therefore ``components == sum(stoichiometric coefficients) * Z'`` -- verified by
hand on MORFOP (1 complex + 2 waters, Z'=2 -> 6 components) and KIVYAS
(1 + 3 waters, Z'=1 -> 4).  The current check is that formula with the sum
hard-wired to 1, which is exactly why it "always kills cocrystals" as the
incumbent filter's own comment at `featurization_utils.py:304-306` records.

Known hard case, recorded so it is not discovered late: DIRSON declares
``0.75(C4 H10 O1), 0.25(C6 H14)`` -- FRACTIONAL solvent occupancy. That is
disorder wearing a stoichiometry coat, and no whole-molecule representation can
hold it.
"""

import re
from typing import Dict, List, NamedTuple, Optional

import numpy as np

__all__ = ['Moiety', 'parse_formula_moiety', 'hill_formula', 'component_formulas']

# '2(H2 O1)' | '0.75(C4 H10 O1)' | 'C2 H5 N4 1+' | 'C21 H28 N2 O2'
_COEFF_GROUP = re.compile(r'^\s*([0-9]*\.?[0-9]+)\s*\((.+)\)\s*$')
_CHARGE = re.compile(r'\s([0-9]*)([+-])\s*$')
_ELEMENT = re.compile(r'([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)')


class Moiety(NamedTuple):
    """One comma-separated unit of `_chemical_formula_moiety`.

    `count` is the leading coefficient (1.0 when absent), `elements` maps element
    symbol to its count in one unit, `charge` is the signed trailing formal charge
    (0 when unstated), and `raw` is the whitespace-stripped unit text as written,
    coefficient and charge included.
    """

    count: float                 # stoichiometric coefficient; 0.75 does occur
    elements: Dict[str, float]   # element -> count within ONE unit
    charge: int                  # formal charge, 0 when unstated
    raw: str

    @property
    def is_fractional(self) -> bool:
        """A non-integer coefficient means disorder, not stoichiometry.

        True when `count` differs from its nearest integer by more than 1e-9.
        """
        return abs(self.count - round(self.count)) > 1e-9


def parse_formula_moiety(text: Optional[str]) -> List[Moiety]:
    """`'C7 H8 N4 O2,2(H2 O1)'` -> two `Moiety` records.

    Returns [] for a missing tag (None or ''). This is captured metadata, and a
    malformed moiety string is not meant to fail a read that is otherwise sound:
    malformed units are skipped or partially parsed rather than refused, so
    callers that need it to be well-formed should check the result. One kind of
    input does raise: an element symbol followed, optionally after whitespace,
    by a bare '.' with no digits (e.g. 'C. H4') reaches ``float('.')`` and
    raises ValueError.

    Units are split on commas outside parentheses. A unit of the form
    ``n(...)`` takes `n` as its coefficient; a trailing whitespace-separated
    ``k+`` / ``k-`` (``k`` defaulting to 1) is the charge; element tokens are an
    uppercase letter plus optional lowercase letter and an optional count
    (default 1), summed when a symbol repeats. A unit with no element token is
    dropped.
    """
    if not text:
        return []
    out: List[Moiety] = []
    for unit in _split_units(text):
        raw = unit.strip()
        if not raw:
            continue
        count = 1.0
        m = _COEFF_GROUP.match(raw)
        body = raw
        if m:
            count = float(m.group(1))
            body = m.group(2)
        charge = 0
        cm = _CHARGE.search(body)
        if cm:
            magnitude = int(cm.group(1)) if cm.group(1) else 1
            charge = magnitude if cm.group(2) == '+' else -magnitude
            body = body[:cm.start()]
        elements: Dict[str, float] = {}
        for sym, num in _ELEMENT.findall(body):
            if not sym:
                continue
            elements[sym] = elements.get(sym, 0.0) + (float(num) if num else 1.0)
        if elements:
            out.append(Moiety(count, elements, charge, raw))
    return out


def _split_units(text: str) -> List[str]:
    """Split on commas that are NOT inside parentheses."""
    units, depth, buf = [], 0, []
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        if ch == ',' and depth == 0:
            units.append(''.join(buf)); buf = []
        else:
            buf.append(ch)
    if buf:
        units.append(''.join(buf))
    return units


def hill_formula(z: np.ndarray) -> str:
    """Atomic numbers -> a Hill-notation formula string ('C7 H6 O2').

    Hill order (C, H, then alphabetical) so two components of the same species
    produce the same string and can be grouped by equality. C and then H come
    first whenever present, so H leads when there is no C ('H1 Cl1'); the other
    symbols follow in sorted order. A count of 1 is written explicitly
    ('H2 O1'). Symbols come from ``ase.data.chemical_symbols``.
    """
    from ase.data import chemical_symbols
    counts: Dict[str, int] = {}
    for zi in z:
        s = chemical_symbols[int(zi)]
        counts[s] = counts.get(s, 0) + 1
    ordered = []
    for s in ('C', 'H'):
        if s in counts:
            ordered.append((s, counts.pop(s)))
    ordered += sorted(counts.items())
    return ' '.join(f'{s}{n}' for s, n in ordered)


def component_formulas(z: np.ndarray, component_index: np.ndarray) -> List[str]:
    """Hill formula per component, in component-index order.

    The bridge a cocrystal implementation needs: it turns "component 3" into
    "H2 O1", so components can be matched against the declared moieties.

    `z` and `component_index` are parallel [n_sites] arrays. Returns a list of
    length ``component_index.max() + 1`` ([] for empty input); an index with no
    sites yields ''.
    """
    n = int(component_index.max()) + 1 if len(component_index) else 0
    return [hill_formula(z[component_index == k]) for k in range(n)]

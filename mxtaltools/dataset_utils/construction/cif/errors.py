"""Every refusal this reader makes is one of these.

The design principle, from `docs/design/cif_reader_design.md` §3: this repository's
characteristic defect is a finite, plausible number with no diagnostic.  So the
reader raises a NAMED exception rather than guessing, and every refusal carries
the identifier and the value that triggered it.

A bare `CifReadError` is never raised -- always a subclass, so a caller can
distinguish "this file is malformed" from "this crystal is outside what the
reader supports".

Current exceptions to the above: a non-numeric coordinate or occupancy value in
the atom-site loop escapes `sites_from_block` as a plain ValueError rather than
one of these, and the `CifParseError` refusals raised by `parse_cif` carry no
identifier.
"""

__all__ = [
    'CifReadError',
    'CifParseError', 'CifMissingDataError',
    'CifSymmetryError', 'CifInconsistentSymmetryError',
    'CifZPrimeError', 'CifTopologyError', 'CifUnsupportedError',
]


class CifReadError(Exception):
    """Base for every refusal. Never raised directly.

    `message` is prefixed with ``[<identifier>]`` and a space when `identifier`
    is truthy; `identifier` is kept as an attribute (None when not given).
    """

    def __init__(self, message: str, identifier: str = None):
        self.identifier = identifier
        super().__init__(f'[{identifier}] {message}' if identifier else message)


class CifParseError(CifReadError):
    """The file is not well-formed CIF: unterminated block, ragged loop, bad number.

    Also raised for duplicate atom-site labels and for an atom-site type symbol
    or label that names no known element.
    """


class CifMissingDataError(CifReadError):
    """A tag the reader needs is absent. Names the tag.

    Also raised when a needed value is present but unusable: a non-positive cell
    length, a cell angle outside (0, 180) degrees, an empty atom-site loop, or a
    site with a missing fractional coordinate.
    """


class CifSymmetryError(CifReadError):
    """Symmetry could not be established: no operators, unparseable operator."""


class CifInconsistentSymmetryError(CifReadError):
    """Symmetry sources contradict each other.

    Currently raised by `space_group_from_block` in one case: the operator count
    of the IT number (declared, or recovered from the H-M symbol) disagrees with
    the number of listed operators and the listed set does not factorise as that
    group's operators composed with centring translations. An H-M symbol that
    differs from the IT number's canonical
    symbol is accepted, not refused.

    Deliberately distinct from `CifSymmetryError`: this one means the file is
    internally inconsistent, and guessing which source to believe is exactly the
    silent-wrong-answer failure the reader exists to avoid.
    """


class CifZPrimeError(CifReadError):
    """Z' cannot be derived: `_cell_formula_units_Z` is absent or non-positive."""


class CifTopologyError(CifReadError):
    """Component decomposition failed or contradicts the declared composition."""


class CifUnsupportedError(CifReadError):
    """Well-formed and self-consistent, but outside what this reader handles.

    Carries a `reason` so callers can filter by category rather than by message
    text. The reasons currently raised in this package are 'polymer'
    (`components_from_block`), 'non-integer-zprime' (`read_cif_block`),
    'partial-occupancy' and 'atomic-number' (`sites_from_block`). The message is
    suffixed with ``(reason=<reason>)`` before the `CifReadError` identifier
    prefix is applied.
    """

    def __init__(self, message: str, reason: str, identifier: str = None):
        self.reason = reason
        super().__init__(f'{message} (reason={reason})', identifier)

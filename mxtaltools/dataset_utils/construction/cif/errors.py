"""Every refusal this reader makes is one of these.

The design principle, from `docs/design/cif_reader_design.md` §3: this repository's
characteristic defect is a finite, plausible number with no diagnostic.  So the
reader raises a NAMED exception rather than guessing, and every refusal carries
the identifier and the value that triggered it.

A bare `CifReadError` is never raised -- always a subclass, so a caller can
distinguish "this file is malformed" from "this crystal is outside what the
reader supports".
"""

__all__ = [
    'CifReadError',
    'CifParseError', 'CifMissingDataError',
    'CifSymmetryError', 'CifInconsistentSymmetryError',
    'CifZPrimeError', 'CifTopologyError', 'CifUnsupportedError',
]


class CifReadError(Exception):
    """Base for every refusal. Never raised directly."""

    def __init__(self, message: str, identifier: str = None):
        self.identifier = identifier
        super().__init__(f'[{identifier}] {message}' if identifier else message)


class CifParseError(CifReadError):
    """The file is not well-formed CIF: unterminated block, ragged loop, bad number."""


class CifMissingDataError(CifReadError):
    """A tag the reader needs is absent. Names the tag."""


class CifSymmetryError(CifReadError):
    """Symmetry could not be established: no operators, unparseable operator."""


class CifInconsistentSymmetryError(CifReadError):
    """Symmetry sources contradict each other.

    E.g. the declared IT number's operator count disagrees with the listed
    operators, or the H-M symbol and the IT number name different groups.
    Deliberately distinct from `CifSymmetryError`: this one means the file is
    internally inconsistent, and guessing which source to believe is exactly the
    silent-wrong-answer failure the reader exists to avoid.
    """


class CifZPrimeError(CifReadError):
    """Z' is absent, non-positive, or inconsistent with the deposited set."""


class CifTopologyError(CifReadError):
    """Component decomposition failed or contradicts the declared composition."""


class CifUnsupportedError(CifReadError):
    """Well-formed and self-consistent, but outside what this reader handles.

    Carries a `reason` so callers can filter by category rather than by message
    text -- e.g. 'polymer', 'non-integer-zprime', 'non-organic-element'.
    """

    def __init__(self, message: str, reason: str, identifier: str = None):
        self.reason = reason
        super().__init__(f'{message} (reason={reason})', identifier)

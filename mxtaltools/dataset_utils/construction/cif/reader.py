"""The public entry point: a CIF file -> `CifCrystal` records.

Geometry only.  No bond orders, no aromaticity, no formal charges, no hydrogen
placement -- those are a chemistry backend's job (design §1.3), and doing them
here from a heavy-atom skeleton is the one thing the open stack genuinely cannot
do safely (61/300 return a plausible WRONG molecule).

What this does produce is enough for the crystal-geometry, packing-coefficient,
RDF, eLJ-without-charges and MLIP paths.
"""

from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .errors import (CifMissingDataError, CifTopologyError, CifUnsupportedError,
                     CifZPrimeError)
from .sites import SiteSet, sites_from_block
from .symmetry import SymmetryInfo, space_group_from_block
from .tokenizer import CifBlock, as_float, as_int, parse_cif
from .topology import ComponentSet, components_from_block

__all__ = ['CifCrystal', 'read_cif', 'read_cif_block']

_CELL_LENGTH_TAGS = ('_cell_length_a', '_cell_length_b', '_cell_length_c')
_CELL_ANGLE_TAGS = ('_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')
_Z_TAGS = ('_cell_formula_units_Z', '_cell_formula_units_z')


class CifCrystal:
    """One crystal, as the file describes it.

    Terms are the design's (§1.5) and are used strictly:

    ``sites``       the DEPOSITED SET -- every `_atom_site` row as written. At
                    Z' < 1 this is NOT the crystallographic asymmetric unit; the
                    exporter has written a whole molecule on a special position.
    ``components``  connected sub-graphs of the deposited set under `_geom_bond`.
    ``z_prime``     an exact `Fraction`, `Z / len(symmetry operators)`. Kept exact
                    rather than float so that 1/3 does not become 0.333...
    ``cell_angles`` RADIANS, matching `MolCrystalData`. The file states degrees.
    """

    __slots__ = ('identifier', 'sites', 'symmetry', 'components',
                 'cell_lengths', 'cell_angles', 'z_value', 'z_prime', 'block_name',
                 'source_path', 'formula_moiety')

    def __init__(self, identifier, sites, symmetry, components,
                 cell_lengths, cell_angles, z_value, z_prime, block_name, source_path,
                 formula_moiety=None):
        self.identifier = identifier
        self.sites = sites
        self.symmetry = symmetry
        self.components = components
        self.cell_lengths = cell_lengths          # (3,) float, Angstrom
        self.cell_angles = cell_angles            # (3,) float, RADIANS
        self.z_value = z_value
        self.z_prime = z_prime                    # Fraction
        self.block_name = block_name
        self.source_path = source_path
        # captured, not acted on: cocrystal support is a stated future requirement
        # and this is where the file declares species, stoichiometry and charge.
        # See composition.py.
        self.formula_moiety = formula_moiety

    @property
    def moieties(self):
        """Parsed `_chemical_formula_moiety`: species, stoichiometry, charge.

        Present in 100 % of a 400-entry CSD sample, multi-unit in 40 %. Nothing
        in the reader consumes it yet; it is the input a cocrystal implementation
        will need, and re-deriving it later would be redoing what the depositor
        already stated.
        """
        from .composition import parse_formula_moiety
        return parse_formula_moiety(self.formula_moiety)

    def component_formulas(self):
        """Hill formula per component -- the bridge from geometry to species."""
        from .composition import component_formulas
        return component_formulas(self.sites.z, self.components.component_index)

    @property
    def sym_mult(self) -> int:
        """Number of symmetry operators. The unit cell holds this many images."""
        return self.symmetry.multiplicity

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def deposited_set_is_asymmetric_unit(self) -> bool:
        """True only when Z' >= 1.

        At Z' < 1 the deposited set spans more than one asymmetric unit, so any
        code treating "what is in the file" as "the ASU" is wrong -- for 19.6 %
        of the CSD.
        """
        return self.z_prime >= 1

    def __repr__(self) -> str:
        return (f'CifCrystal({self.identifier!r}, #{self.symmetry.number} '
                f'{self.symmetry.symbol}, {self.n_sites} sites, '
                f"{self.components.n_components} components, Z'={self.z_prime})")


def _cell_from_block(block: CifBlock, identifier: str):
    lengths = []
    for t in _CELL_LENGTH_TAGS:
        v = as_float(block.get(t), t, identifier)
        if v is None:
            raise CifMissingDataError(f'{t} is absent', identifier)
        if v <= 0:
            raise CifMissingDataError(f'{t} is {v}, which is not a length', identifier)
        lengths.append(v)

    angles = []
    for t in _CELL_ANGLE_TAGS:
        v = as_float(block.get(t), t, identifier)
        if v is None:
            raise CifMissingDataError(f'{t} is absent', identifier)
        if not (0 < v < 180):
            raise CifMissingDataError(
                f'{t} is {v} degrees, outside (0, 180)', identifier)
        angles.append(v)

    return (np.array(lengths, dtype=np.float64),
            np.radians(np.array(angles, dtype=np.float64)))


def read_cif_block(block: CifBlock,
                   identifier: Optional[str] = None,
                   source_path: Optional[str] = None,
                   refuse_polymers: bool = True,
                   max_atomic_number: Optional[int] = 100,
                   require_integer_z_prime: bool = True,
                   require_bonds: bool = True) -> CifCrystal:
    """Assemble one `CifCrystal` from one parsed data block."""
    # The block name is the identity: `data_EFUNUP` for a CSD export, and this
    # package's writer emits `data_{identifier}` for the same reason.
    # `_chemical_name_common` is a DESCRIPTION ("cytosinium 4-nitrobenzoate
    # cytosine monohydrate") and makes a poor key, so it is only a fallback.
    ident = identifier or block.name or block.get('_chemical_name_common')

    symmetry = space_group_from_block(block, ident)
    sites = sites_from_block(block, ident, max_atomic_number=max_atomic_number)
    cell_lengths, cell_angles = _cell_from_block(block, ident)
    components = components_from_block(block, sites, ident,
                                       refuse_polymers=refuse_polymers,
                                       require_bonds=require_bonds)

    z_tag, z_raw = block.first_of(*_Z_TAGS)
    z_value = as_int(z_raw, z_tag or '_cell_formula_units_Z', ident)
    if z_value is None:
        raise CifZPrimeError(
            'no _cell_formula_units_Z, so Z\' cannot be derived. It is '
            'Z / len(symmetry operators) and there is no substitute', ident)
    if z_value <= 0:
        raise CifZPrimeError(f'_cell_formula_units_Z is {z_value}', ident)

    # Exact rational: 1/3 must not become 0.3333.
    z_prime = Fraction(z_value, symmetry.multiplicity)

    # An UNKNOWN decomposition is only safe when there is genuinely one component.
    # The unit cell is built by wrapping each component's centroid so molecules move
    # rigidly; with every site in one component a multi-molecule deposited set wraps
    # as a single lump. Measured on EFUNUP (4 components): the built cell differs by
    # 3.359 A, silently. Z' > 1 guarantees at least Z' components, and a multi-unit
    # moiety formula declares more than one species, so both are detectable here
    # without the bond loop that is missing.
    if components.source == 'none':
        moiety = block.get('_chemical_formula_moiety')
        n_moieties = len([u for u in (moiety or '').split(',') if u.strip()])
        if z_prime > 1 or n_moieties > 1:
            raise CifTopologyError(
                f"component decomposition is unknown (no {'_geom_bond'} loop) but this "
                f"crystal cannot be single-component: Z'={z_prime}, "
                f'{n_moieties} moiety unit(s). Building the unit cell would wrap every '
                'site as one rigid body and silently produce a different crystal',
                ident)

    if require_integer_z_prime and z_prime.denominator != 1:
        raise CifUnsupportedError(
            f"Z' is {z_prime} (Z={z_value} / {symmetry.multiplicity} operators); "
            'a molecule sits on a special position and the deposited set is not '
            'an asymmetric unit',
            reason='non-integer-zprime', identifier=ident)

    return CifCrystal(ident, sites, symmetry, components,
                      cell_lengths, cell_angles, z_value, z_prime,
                      block.name, source_path,
                      formula_moiety=block.get('_chemical_formula_moiety'))


def read_cif(source: Union[str, Path],
             refuse_polymers: bool = True,
             max_atomic_number: Optional[int] = 100,
             require_integer_z_prime: bool = True,
             require_bonds: bool = True) -> List[CifCrystal]:
    """Read every crystal in a CIF file (or CIF text).

    A `.cif` may hold several `data_` blocks; all are returned, in file order.

    Parameters
    ----------
    source : path to a .cif, or the CIF text itself
    refuse_polymers : raise on a moiety formula declaring a polymer
    max_atomic_number : raise if any site exceeds it (None = no limit)
    require_integer_z_prime : raise when a molecule sits on a special position
    require_bonds : raise when the file has no `_geom_bond` loop. False reads the
        deposited set with its component decomposition marked UNKNOWN
        (`components.source == 'none'`), never guessed from distances.

    DEFAULTS REFUSE. Every flag above defaults to the strict setting: the caller
    must explicitly ask to accept a polymer, a heavy element, or a non-integer Z'.
    There is enough genuine nonsense in the CSD that a permissive default would
    quietly admit structures no workflow here should treat as valid.

    Raises
    ------
    CifReadError subclasses only. Every refusal is named; see `errors.py`.
    """
    text, path = _load(source)
    blocks = parse_cif(text)
    return [read_cif_block(b, source_path=path,
                           refuse_polymers=refuse_polymers,
                           max_atomic_number=max_atomic_number,
                           require_integer_z_prime=require_integer_z_prime,
                           require_bonds=require_bonds)
            for b in blocks]


def _load(source):
    if isinstance(source, Path):
        return source.read_text(encoding='utf-8', errors='ignore'), str(source)
    if isinstance(source, str) and '\n' not in source and source.strip().endswith('.cif'):
        p = Path(source)
        return p.read_text(encoding='utf-8', errors='ignore'), str(p)
    return source, None

"""CCDC-free CIF reading for MXtalTools.

See `docs/design/cif_reader_design.md`. Vocabulary is defined in §1.5 and used
consistently throughout this package:

    site           one `_atom_site` row
    deposited set  every site in the file, exactly as written -- NOT necessarily
                   the crystallographic asymmetric unit (at Z'<1, ~20% of the CSD,
                   the exporter writes a whole molecule on a special position)
    component      one connected sub-graph of the deposited set under `_geom_bond`
    Z'             _cell_formula_units_Z / len(symmetry operators), an exact rational
    unit cell      the deposited set expanded by all operators, centroid-wrapped
"""

from .errors import (CifReadError, CifParseError, CifMissingDataError,
                     CifSymmetryError, CifInconsistentSymmetryError,
                     CifZPrimeError, CifTopologyError, CifUnsupportedError)
from .tokenizer import CifBlock, CifLoop, parse_cif, strip_esd, is_missing, as_float, as_int
from .sites import SiteSet, sites_from_block, element_from_symbol
from .symmetry import SymmetryInfo, space_group_from_block, parse_symop
from .topology import ComponentSet, components_from_block
from .reader import CifCrystal, read_cif, read_cif_block
from .composition import Moiety, parse_formula_moiety, hill_formula, component_formulas

__all__ = [
    'CifReadError', 'CifParseError', 'CifMissingDataError', 'CifSymmetryError',
    'CifInconsistentSymmetryError', 'CifZPrimeError', 'CifTopologyError',
    'CifUnsupportedError',
    'CifBlock', 'CifLoop', 'parse_cif', 'strip_esd', 'is_missing', 'as_float', 'as_int',
    'SiteSet', 'sites_from_block', 'element_from_symbol',
    'SymmetryInfo', 'space_group_from_block', 'parse_symop',
    'ComponentSet', 'components_from_block',
    'CifCrystal', 'read_cif', 'read_cif_block',
    'Moiety', 'parse_formula_moiety', 'hill_formula', 'component_formulas',
]

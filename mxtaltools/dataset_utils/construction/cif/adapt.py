"""The seam: `CifCrystal` -> the `crystal_dict` the existing pipeline consumes.

Byte-compatible with what `featurization_utils.extract_crystal_data` emits, so
`init_zp1_crystals` -> `extract_zp1_pose_info` -> `pose_aunit`/`build_unit_cell`
-> `crystal_rebuild_checks` -> `instantiate_crystal` all run unmodified.

The one piece of real geometry here is `build_unit_cell_components`: CCDC hands
over `unit_cell.components` ready-made, so a CCDC-free reader has to construct
them.  The construction is the design's: apply every operator to the COMPONENT
CENTROID, wrap that centroid into the cell, and carry the component's atoms
rigidly with it.  Wrapping atoms individually would tear molecules apart across
the boundary.
"""

from typing import Dict, List, Optional

import numpy as np

from mxtaltools.common.geometry_utils import coor_trans_matrix_np
from mxtaltools.constants.space_group_info import SPACE_GROUPS

from .reader import CifCrystal

__all__ = ['build_unit_cell_components', 'to_crystal_dict']


def build_unit_cell_components(crystal: CifCrystal) -> Dict[str, list]:
    """Expand the deposited set by every symmetry operator.

    Returns
    -------
    dict with 'coordinates' and 'atomic_numbers', each a list of length
    `sym_mult * n_components` -- one entry per molecule in the unit cell, matching
    the order `extract_crystal_data` gets from `unit_cell.components`.

    Ordering is OPERATOR-MAJOR then component: [op0 comp0, op0 comp1, op1 comp0, ...].
    Stated because it is contracted -- `crystal_rebuild_checks` matches atoms
    positionally, and `z.repeat(sym_mult)` downstream TILES rather than
    interleaves, so the two orderings must be reconciled deliberately, never by
    coincidence.
    """
    T_fc, _ = coor_trans_matrix_np('f_to_c', crystal.cell_lengths,
                                   crystal.cell_angles, return_vol=True)
    frac = crystal.sites.frac
    comp_index = crystal.components.component_index
    n_comp = crystal.components.n_components

    coords: List[np.ndarray] = []
    numbers: List[np.ndarray] = []

    for op in crystal.symmetry.operators:
        R, t = op[:3, :3], op[:3, 3]
        image = frac @ R.T + t                       # fractional, unwrapped
        for k in range(n_comp):
            mask = comp_index == k
            block = image[mask]
            centroid = block.mean(axis=0)
            # wrap the CENTROID, move the whole component with it: wrapping atoms
            # individually would split a molecule straddling the cell boundary
            block = block - np.floor(centroid)
            coords.append(block @ T_fc.T)
            numbers.append(crystal.sites.z[mask].copy())

    return {'coordinates': coords, 'atomic_numbers': numbers}


def to_crystal_dict(crystal: CifCrystal,
                    identifier: Optional[str] = None) -> Dict:
    """Render a `CifCrystal` in the incumbent `crystal_dict` schema.

    Every key `extract_crystal_data` produces is present with the same dtype and
    units, so nothing downstream needs to know which reader produced it.  Two
    keys are added, both prefixed so they cannot collide with the incumbent's:

      ``native_operators_are_standard``  whether the file's operators match
          SYM_OPS[sg] exactly, in order. False for ~37.5 % of the CSD and NOT an
          error; recorded so a dataset can be audited after the fact.
      ``native_reader``  marks provenance, so a dataset built by this reader is
          distinguishable from one built by CCDC.
    """
    ident = identifier or crystal.identifier
    sym = crystal.symmetry

    d: Dict = {}
    d['identifier'] = ident
    d['z_prime'] = float(crystal.z_prime)
    d['z_value'] = int(crystal.z_value)
    d['symmetry_operators'] = [np.asarray(o, dtype=float) for o in sym.operators]
    d['symmetry_multiplicity'] = len(sym.operators)

    d['space_group_number'] = int(sym.number)
    # The incumbent takes `spacegroup_number_and_setting` from CCDC. The setting is
    # extracted there and never consumed (featurization_utils.py:94), so 0 is a
    # faithful placeholder rather than a guess at a value nothing reads.
    d['space_group_setting'] = 0
    # CANONICAL symbol, matching the incumbent's `SPACE_GROUPS[number]`. The file's
    # own symbol may differ legitimately (ZZZKEA02 says 'C -1' under number 2) and
    # is kept separately rather than written here, because the incumbent asserts
    # this value is in SPACE_GROUPS.values().
    d['space_group_symbol'] = SPACE_GROUPS[int(sym.number)]
    d['native_file_space_group_symbol'] = sym.symbol

    a, b, c = [float(v) for v in crystal.cell_lengths]
    al, be, ga = [float(v) for v in crystal.cell_angles]        # already radians
    d['lattice_a'], d['lattice_b'], d['lattice_c'] = a, b, c
    d['lattice_alpha'], d['lattice_beta'], d['lattice_gamma'] = al, be, ga

    d['fc_transform'], d['cell_volume'] = coor_trans_matrix_np(
        'f_to_c', crystal.cell_lengths, crystal.cell_angles, return_vol=True)

    cell = build_unit_cell_components(crystal)
    d['unit_cell_coordinates'] = cell['coordinates']
    d['unit_cell_atomic_numbers'] = cell['atomic_numbers']

    d['native_operators_are_standard'] = bool(sym.operators_are_standard)
    d['native_reader'] = True

    # the incumbent's own invariant, asserted here so a mismatch is loud at the
    # seam rather than deep inside the builder
    expected = int(round(d['symmetry_multiplicity'] * d['z_prime']))
    if len(d['unit_cell_coordinates']) != expected:
        raise ValueError(
            f'[{ident}] built {len(d["unit_cell_coordinates"])} unit-cell components '
            f'but symmetry_multiplicity * z_prime = {expected}')
    return d

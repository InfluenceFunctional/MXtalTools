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

    Each 4x4 operator in ``crystal.symmetry.operators`` is applied to the
    fractional site coordinates ``crystal.sites.frac``; for each component the
    image is shifted by the integer floor of its unweighted fractional centroid,
    so the centroid lands in [0, 1) and the component moves rigidly, and is then
    converted to Cartesian with the cell's fractional-to-Cartesian matrix.

    Parameters
    ----------
    crystal : CifCrystal
        Source crystal; read only.

    Returns
    -------
    dict
        ``'coordinates'``: list of [n_atoms_k, 3] float Cartesian arrays, in the
        units of ``crystal.cell_lengths``. ``'atomic_numbers'``: list of
        [n_atoms_k] copies of the component's ``crystal.sites.z``. Each list has
        length ``len(operators) * n_components`` -- one entry per molecule in
        the unit cell, matching the order `extract_crystal_data` gets from
        `unit_cell.components`.

    Notes
    -----
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
    units, so nothing downstream needs to know which reader produced it.  Three
    keys are added, all prefixed so they cannot collide with the incumbent's:

    - ``native_file_space_group_symbol``: ``crystal.symmetry.symbol``, the
      file's own H-M symbol, or the canonical symbol when the file gives none;
      ``space_group_symbol`` holds the canonical ``SPACE_GROUPS[number]``.
    - ``native_operators_are_standard``: whether the file's operators match
      SYM_OPS[sg] element-wise, in order and with the same count, under
      ``np.allclose(atol=1e-6)``. False for ~37.5 % of the CSD and NOT an
      error; recorded so a dataset can be audited after the fact.
    - ``native_reader``: always True; marks provenance, so a dataset built by
      this reader is distinguishable from one built by CCDC.

    Parameters
    ----------
    crystal : CifCrystal
        Source crystal; read only.
    identifier : str, optional
        Value for ``'identifier'`` and for error messages; falls back to
        ``crystal.identifier`` when not given or empty.

    Returns
    -------
    dict
        ``identifier``; ``space_group_symbol`` (canonical) and the three
        ``native_*`` keys above; scalar cell entries ``lattice_a/b/c`` (from
        ``crystal.cell_lengths``) and ``lattice_alpha/beta/gamma`` in radians;
        ``fc_transform`` and
        ``cell_volume`` from `coor_trans_matrix_np`; ``z_prime`` as float;
        ``z_value``, ``symmetry_multiplicity`` and ``space_group_number`` as int;
        ``space_group_setting`` fixed at 0; ``symmetry_operators`` as a list of
        4x4 float arrays; ``unit_cell_coordinates`` and
        ``unit_cell_atomic_numbers`` from `build_unit_cell_components`.

    Raises
    ------
    ValueError
        If the number of unit-cell components differs from
        ``round(symmetry_multiplicity * z_prime)``, i.e. the deposited set's
        component count differs from Z'.
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

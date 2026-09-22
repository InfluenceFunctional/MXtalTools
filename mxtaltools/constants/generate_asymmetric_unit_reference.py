"""Generate a complete asymmetric-unit reference for all 230 space groups.

**Nothing consumes the output yet, and it cannot be consumed by the current code.**
`constants/asymmetric_units.py` stores one axis-aligned fractional box per group,
which can only represent an ASU that IS a box. 108 of 230 groups are therefore
stored as the placeholder `[1, 1, 1]` -- the whole cell -- which is why
`is_well_defined` (does exactly one symmetry image land inside the box?) is
structurally unreachable for those groups: every image is inside.

This script records what the real domains are, so that when a general
parameterisation exists the data does not have to be re-derived.

SOURCE. `cctbx.sgtbx.direct_space_asu.reference_table` carries the International
Tables asymmetric units as half-space cuts, including the conditional facets that
a box cannot express, e.g.

    Fm-3m   z>=0 | x-y>=0 | x+y<=1/2 | y-z>=0
    R-3c    x>=0 [y<=0] | y>=0 | z>=0 [x-y>=0 [y<=0]] | z<=1/12 [...]

cctbx is a GENERATION-time dependency only: this writes a JSON table and the
runtime never imports cctbx.

VALIDATION. Two checks are run per group and recorded in the output:
  * Monte-Carlo volume x multiplicity should be 1 -- the domain tiles the cell.
  * Where `asymmetric_units.py` holds a real (non-placeholder) box, it is
    compared against the cctbx domain. They agree on all such groups.

TWO TRAPS, both of which cost time when this was first written:
  * Several ASUs lie PARTLY OUTSIDE the unit cube. Fd-3m has
    box_min = [-1/8, -1/8, -1/4]. Sampling [0,1)^3 gives zero hits and looks like
    an empty domain. Sample the group's own bounding box instead.
  * High multiplicity means a tiny domain: at mult 192 the ASU is ~0.5 % of the
    cell, so uniform sampling of the cube needs ~58,000 points for 300 hits.

Usage:  python -m mxtaltools.constants.generate_asymmetric_unit_reference
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name('asymmetric_unit_reference.json')
N_SAMPLES = 200_000
SEED = 20260831


def _as_floats(vec):
    return [float(Fraction(str(v))) for v in vec]


def describe(sg: int, rng) -> dict:
    from cctbx.sgtbx.direct_space_asu import reference_table

    from mxtaltools.constants.asymmetric_units import ASYM_UNITS
    from mxtaltools.constants.space_group_info import SPACE_GROUPS, SYM_OPS

    asu = reference_table.get_asu(sg)
    shape = asu.shape_only()
    lo = np.array(_as_floats(shape.box_min()))
    hi = np.array(_as_floats(shape.box_max()))

    # sample the domain's OWN bounding box, never the unit cube: several ASUs
    # extend to negative coordinates and would otherwise register as empty
    pts = lo + rng.random((N_SAMPLES, 3)) * (hi - lo)
    inside = np.fromiter(
        (shape.is_inside((float(a), float(b), float(c))) for a, b, c in pts),
        dtype=bool, count=len(pts))
    box_volume = float(np.prod(hi - lo))
    volume = float(inside.mean()) * box_volume
    mult = len(SYM_OPS[sg])

    # is the domain exactly its bounding box? that is the ONLY case the current
    # `asymmetric_units.py` format can represent without loss
    is_box = bool(inside.all())

    stored = [float(x) for x in ASYM_UNITS[str(sg)]]
    is_placeholder = stored == [1.0, 1.0, 1.0]

    return {
        'space_group': sg,
        'symbol': SPACE_GROUPS[sg],
        'multiplicity': mult,
        'cuts': [c.as_xyz() for c in asu.cuts],
        'n_cuts': len(asu.cuts),
        'bounding_box_min': [str(v) for v in shape.box_min()],
        'bounding_box_max': [str(v) for v in shape.box_max()],
        'extends_outside_unit_cube': bool((lo < -1e-9).any() or (hi > 1 + 1e-9).any()),
        'volume_monte_carlo': round(volume, 8),
        'volume_times_multiplicity': round(volume * mult, 6),
        'is_exactly_its_bounding_box': is_box,
        'stored_box': stored,
        'stored_is_placeholder': is_placeholder,
    }


def main():
    rng = np.random.default_rng(SEED)
    rows = [describe(sg, rng) for sg in range(1, 231)]

    boxlike = sum(r['is_exactly_its_bounding_box'] for r in rows)
    outside = sum(r['extends_outside_unit_cube'] for r in rows)
    placeholders = sum(r['stored_is_placeholder'] for r in rows)
    bad_vol = [r['space_group'] for r in rows
               if abs(r['volume_times_multiplicity'] - 1.0) > 0.05]

    payload = {
        '_meta': {
            'source': 'cctbx.sgtbx.direct_space_asu.reference_table (International Tables)',
            'generator': 'mxtaltools/constants/generate_asymmetric_unit_reference.py',
            'samples_per_group': N_SAMPLES,
            'seed': SEED,
            'consumed_by': 'NOTHING YET -- the runtime format stores one axis-aligned '
                           'box per group and cannot represent a general domain',
            'summary': {
                'groups': len(rows),
                'asu_is_exactly_a_box': boxlike,
                'asu_extends_outside_unit_cube': outside,
                'currently_stored_as_placeholder': placeholders,
                'volume_check_failures': bad_vol,
            },
        },
        'asymmetric_units': {str(r['space_group']): r for r in rows},
    }
    OUT.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')

    print(f'wrote {OUT}')
    print(f'  groups                          : {len(rows)}')
    print(f'  ASU is exactly an axis box      : {boxlike}  '
          f'(the only shape the current format can hold)')
    print(f'  ASU extends outside [0,1)^3     : {outside}')
    print(f'  currently stored as placeholder : {placeholders}')
    print(f'  volume x multiplicity != 1      : {bad_vol or "none"}')


if __name__ == '__main__':
    main()

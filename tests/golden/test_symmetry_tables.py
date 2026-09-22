"""F -- the symmetry tables: 17,020 lines of literal data with no tests at all.

`constants/space_group_info.py` holds 230 space groups and 4,425 symmetry
operators as Python literals, and `constants/asymmetric_units.py` holds a box per
group. Everything downstream trusts them: the unit-cell builder tiles by operator
count, `sym_mult` indexes into the list, and the asymmetric-unit box decides which
symmetry image is "canonical". A wrong entry produces a plausible crystal.

These are mathematical invariants, so they need no reference values -- they hold
or the table is wrong. That makes them cheap, exhaustive and permanent.

A NOTE ON THE CLOSURE TEST, because it caught me first.
`np.mod(-1e-16, 1.0)` returns 0.9999999999999999, so a translation of zero reads
as one. Comparing raw `mod` results reported 7 groups as non-closed -- 146, 148,
155, 160, 161, 166, 167, all rhombohedral in the hexagonal setting, where
translations of exactly 1 arise. Rounding BEFORE the mod, all 230 close. The
tables were never wrong; the comparison was.
"""

import itertools

import numpy as np
import pytest

from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.space_group_info import LATTICE_TYPE, SPACE_GROUPS, SYM_OPS

pytestmark = pytest.mark.golden


def _key(op):
    """Identity of a symmetry operator, modulo whole lattice translations.

    Rounds BEFORE the mod: see the module docstring. Getting this backwards
    reports every rhombohedral group as broken.
    """
    op = np.asarray(op, dtype=float)
    trans = np.mod(np.round(op[:3, 3], 6), 1.0)
    return (tuple(np.round(op[:3, :3].ravel(), 6)), tuple(np.round(trans, 6)))


def test_table_shape():
    """230 space groups, and the operator count is what nothing else records."""
    assert len(SYM_OPS) == 230
    assert set(SYM_OPS) == set(range(1, 231))
    total = sum(len(v) for v in SYM_OPS.values())
    assert total == 4425, f'{total} operators; the table changed size'


@pytest.mark.parametrize('sg', sorted(SYM_OPS))
def test_operators_are_proper_affine(sg):
    """Every operator is a 4x4 affine with |det(R)| == 1.

    A symmetry operation is a rigid motion: it may reflect (det -1) but never
    scale. A determinant off 1 means the operator would stretch the crystal.
    """
    for i, op in enumerate(SYM_OPS[sg]):
        o = np.asarray(op, dtype=float)
        assert o.shape == (4, 4), f'sg {sg} op {i} has shape {o.shape}'
        det = np.linalg.det(o[:3, :3])
        assert abs(abs(det) - 1.0) < 1e-8, (
            f'sg {sg} ({SPACE_GROUPS[sg]}) op {i} has |det| = {abs(det):.9f}, '
            'so it scales rather than moves')
        assert np.allclose(o[3], [0, 0, 0, 1], atol=1e-12), (
            f'sg {sg} op {i} has a non-affine bottom row')


@pytest.mark.parametrize('sg', sorted(SYM_OPS))
def test_identity_present_and_first(sg):
    """The identity must be present, and at index 0.

    Position matters, not just membership: `aunit2ucell` treats image 0 as the
    asymmetric unit as deposited, so an identity anywhere else silently relabels
    which image is the reference.
    """
    ops = [np.asarray(o, dtype=float) for o in SYM_OPS[sg]]
    assert any(np.allclose(o, np.eye(4), atol=1e-8) for o in ops), (
        f'sg {sg} ({SPACE_GROUPS[sg]}) has no identity operator')
    assert np.allclose(ops[0], np.eye(4), atol=1e-8), (
        f'sg {sg}: the identity is not at index 0')


@pytest.mark.parametrize('sg', sorted(SYM_OPS))
def test_operators_are_distinct(sg):
    """No duplicates, modulo lattice translation.

    A duplicate would inflate `sym_mult`, so the builder would place two molecules
    on top of each other and the cell would be over-full.
    """
    keys = [_key(o) for o in SYM_OPS[sg]]
    assert len(set(keys)) == len(keys), (
        f'sg {sg} ({SPACE_GROUPS[sg]}) lists a duplicate operator')


@pytest.mark.parametrize('sg', sorted(SYM_OPS))
def test_operator_set_is_closed_under_composition(sg):
    """Composing any two operators gives another member. This is what makes it a GROUP.

    If the set were not closed, applying operators to build a unit cell could
    generate positions the group does not contain -- atoms outside the crystal's
    own symmetry.
    """
    ops = [np.asarray(o, dtype=float) for o in SYM_OPS[sg]]
    have = {_key(o) for o in ops}
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            assert _key(a @ b) in have, (
                f'sg {sg} ({SPACE_GROUPS[sg]}): op{i} composed with op{j} is not '
                'in the operator set, so the table is not a group')


@pytest.mark.parametrize('sg', sorted(SYM_OPS))
def test_every_operator_has_an_inverse(sg):
    """A group needs inverses. Equivalent to closure for a finite set, checked
    directly so a failure names the offending operator rather than a product."""
    ops = [np.asarray(o, dtype=float) for o in SYM_OPS[sg]]
    have = {_key(o) for o in ops}
    for i, o in enumerate(ops):
        assert _key(np.linalg.inv(o)) in have, (
            f'sg {sg} ({SPACE_GROUPS[sg]}) op {i} has no inverse in the set')


# --------------------------------------------------------------------------
# asymmetric-unit boxes
# --------------------------------------------------------------------------

PLACEHOLDER = [sg for sg, box in ASYM_UNITS.items()
               if [float(x) for x in box] == [1.0, 1.0, 1.0]]
REAL = [sg for sg in ASYM_UNITS if sg not in PLACEHOLDER]


def test_placeholder_count_is_recorded():
    """108 of 230 boxes are [1,1,1] -- the whole cell, a PLACEHOLDER for groups
    whose asymmetric unit is not a parallelepiped.

    This is why `is_well_defined` (which asks whether exactly one symmetry image
    lands inside the box) can never be True for those groups: every image is
    inside. Pinning the count makes any change to that population visible.
    """
    assert len(PLACEHOLDER) == 108, (
        f'{len(PLACEHOLDER)} placeholder boxes, was 108. The set of space groups '
        'that can never satisfy is_well_defined has changed.')
    assert len(REAL) == 122


@pytest.mark.parametrize('sg', sorted(int(s) for s in REAL))
def test_asymmetric_unit_volume_times_multiplicity_is_one(sg):
    """A genuine asymmetric unit tiles the cell exactly `sym_mult` times.

    So box volume * sym_mult == 1. This is the one arithmetic check on 122 boxes
    that nothing else validates -- and it holds exactly for all of them today.
    """
    box = [float(x) for x in ASYM_UNITS[sg if sg in ASYM_UNITS else str(sg)]]
    mult = len(SYM_OPS[sg])
    product = box[0] * box[1] * box[2] * mult
    assert product == pytest.approx(1.0, abs=1e-6), (
        f'sg {sg} ({SPACE_GROUPS[sg]}): box {box} x {mult} images = {product:.6f}, '
        'not 1 -- the box is not a fundamental domain of this group')


def test_lattice_type_never_says_rhombohedral():
    """Records a live quirk rather than asserting it is right.

    `LATTICE_TYPE` maps NO space group to 'rhombohedral' -- the seven R-groups
    (146, 148, 155, 160, 161, 166, 167) are all labelled 'hexagonal', which is
    correct for the hexagonal SETTING they are tabulated in. Any code branching
    on a 'rhombohedral' label is therefore dead. Pinned so that a future change
    making it reachable is a deliberate, visible event.
    """
    assert not any(v == 'rhombohedral' for v in LATTICE_TYPE.values())
    r_groups = [146, 148, 155, 160, 161, 166, 167]
    assert {LATTICE_TYPE[sg] for sg in r_groups} == {'hexagonal'}

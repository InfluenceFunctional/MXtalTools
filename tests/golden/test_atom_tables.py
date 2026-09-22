"""A -- the atom property tables, and the positional-indexing assumption on them.

`crystal_analysis.py:107` builds `torch.tensor(list(VDW_RADII.values()))` and then
indexes it BY ATOMIC NUMBER. That is correct only because the dict's insertion
order happens to be 0, 1, 2, ... 99. Reordering the literal, dropping the `0:`
sentinel, or appending an element out of order shifts every entry and changes
every eLJ energy -- silently, since the result is still a plausible number.

Measured: at least 13 call sites index a table this way, including the model's
own input featurisation. Zero tests cover it today.
"""

import numpy as np
import pytest
import torch

from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII

pytestmark = pytest.mark.golden

TABLES = {'ATOM_WEIGHTS': ATOM_WEIGHTS, 'VDW_RADII': VDW_RADII}


@pytest.mark.parametrize('name', sorted(TABLES))
def test_keys_are_contiguous_from_zero(name):
    """The whole positional-indexing idiom rests on this one property."""
    keys = list(TABLES[name].keys())
    assert keys == list(range(len(keys))), (
        f'{name} keys are not 0..{len(keys)-1} in order, so '
        f'`tensor(list({name}.values()))[z]` no longer selects element z')


@pytest.mark.parametrize('name', sorted(TABLES))
def test_positional_lookup_equals_keyed_lookup(name):
    """`values()[z] == table[z]` for every z. This is the idiom, asserted directly."""
    table = TABLES[name]
    values = list(table.values())
    bad = [z for z in table if values[z] != table[z]]
    assert not bad, f'{name}: positional and keyed lookup disagree for z={bad[:8]}'


def test_vdw_radii_anchor_values(golden):
    """Anchor the VALUES too.

    The contiguity test above still passes under a wholesale table swap -- e.g.
    Bondi radii replaced with Alvarez -- which would rescale every eLJ energy.
    These anchors do not.
    """
    z = [0, 1, 6, 7, 8, 17, 53]
    got = [float(VDW_RADII[i]) for i in z]
    ref = golden('atom_tables', 'A2_vdw_radii_anchors', got)
    assert got == pytest.approx(ref, abs=0.0), f'VDW_RADII changed at z={z}'


def test_atom_weight_anchor_values(golden):
    z = [0, 1, 6, 8, 82]
    got = [float(ATOM_WEIGHTS[i]) for i in z]
    ref = golden('atom_tables', 'A2_atom_weight_anchors', got)
    assert got == pytest.approx(ref, abs=0.0), f'ATOM_WEIGHTS changed at z={z}'


def test_tensor_cast_preserves_indexing():
    """The cast that the call sites actually perform, checked end to end."""
    for name, table in TABLES.items():
        t = torch.tensor(list(table.values()), dtype=torch.float32)
        assert len(t) == len(table)
        for z in (1, 6, 7, 8, 17, 53):
            # RELATIVE tolerance: the tables span 1.0 to 207.2, so a float32 cast
            # of 126.90447 lands 2.4e-6 away in absolute terms. That is precision,
            # not a defect -- an absolute tolerance here fails on iodine and passes
            # on hydrogen, which is exactly backwards.
            assert float(t[z]) == pytest.approx(float(table[z]), rel=1e-6), (
                f'{name}: tensor[{z}] != table[{z}] after cast')

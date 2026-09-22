"""D -- the latent <-> physical cell map, pinned in the INVERSE direction.

THE POINT OF THIS FILE, in one measurement.

`au_range` and `ang_range` are duplicated byte-for-byte at `crystal_ops.py:384/387`
and again at `:428/431`, with no shared constant. Hoisting them to one constant and
widening the lower bound from 0.075 to 0.05 -- an entirely reasonable-looking edit --
gives this:

    decoded cell lengths   5.704, 9.073, 23.001   ->   4.753, 6.831, 23.001
    a*b*c                  1190.4                 ->   746.8   (37% smaller)
    roundtrip residual     1.788e-07              ->   1.788e-07   (IDENTICAL)

The same stored latent decodes to a different crystal, and the roundtrip is
bit-for-bit unchanged, because both directions read the same constants and the
change cancels. Every stored prior, replay buffer row and checkpoint policy would
silently mean something else, with no test able to notice.

So this file pins the INVERSE direction absolutely: a fixed latent must decode to
a fixed physical cell. That is the only assertion a duplicated-constant change
cannot pass.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.golden

# a fixed latent, spanning the interior of [-1, 1] on every coordinate
LATENT = [0.1, -0.4, 0.7, 0.2, -0.6, 0.55, -0.3, -0.2, 0.6, 0.15, -0.5, 0.35]


@pytest.fixture(scope='module')
def crystal():
    """A literal crystal -- no dataset, no RNG, no I/O, CPU only.

    `radius`, `mass` and `mol_volume` are supplied explicitly rather than computed,
    so the fixture cannot drift when the molecule-scalar code changes: this file is
    testing the latent map, not the scalars.
    """
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list

    mol = MolData(
        z=torch.tensor([6, 6, 8, 7]),
        pos=torch.tensor([[0., 0., 0.], [1.5, 0., 0.], [0., 1.4, 0.], [0., 0., 1.3]]),
        radius=5.0, mass=50.0, mol_volume=80.0, identifier='golden_latent')
    c = MolCrystalData(
        molecule=mol, sg_ind=14,
        cell_lengths=torch.tensor([10., 10., 10.]),
        cell_angles=torch.tensor([np.pi / 2] * 3),
        aunit_centroid=torch.tensor([.3, .3, .3]),
        aunit_orientation=torch.tensor([.4, .2, 1.1]),
        aunit_handedness=1, do_box_analysis=True)
    return collate_data_list([c])


def test_latent_decodes_to_a_fixed_physical_cell(golden, crystal):
    """A fixed latent -> fixed cell lengths and angles. The load-bearing pin.

    Catches any change to `au_range` / `ang_range`, including one applied
    consistently to BOTH duplicated copies -- which the roundtrip cannot see.
    """
    out = crystal.inv_latent_transform(torch.tensor([LATENT]))
    lengths = [float(v) for v in out[0, :3]]
    angles = [float(v) for v in out[0, 3:6]]

    ref_l = golden('latent_map', 'D1_cell_lengths', lengths)
    ref_a = golden('latent_map', 'D1_cell_angles', angles)
    rtol, atol = golden.tol('latent_map', 'D1_cell_lengths', 1e-6, 1e-6)
    assert lengths == pytest.approx(ref_l, rel=rtol, abs=atol)
    assert angles == pytest.approx(ref_a, rel=rtol, abs=atol)


def test_decoded_cell_volume_is_pinned(golden, crystal):
    """The scalar a*b*c, pinned separately.

    A single memorable number that moves under any length-range change. The
    reference widening (0.075 -> 0.05) takes it from 1190.4 to 746.8.
    """
    out = crystal.inv_latent_transform(torch.tensor([LATENT]))
    vol = float(out[0, 0] * out[0, 1] * out[0, 2])
    ref = golden('latent_map', 'D2_abc_product', vol)
    assert vol == pytest.approx(ref, rel=1e-6)


def test_decoded_angles_lie_inside_the_declared_range(crystal):
    """`ang_range` is [0.2*pi, 0.8*pi]. An interior latent must decode inside it.

    Not a golden value -- a property. It fails if the angle range is narrowed
    below the decoded values, which a pinned number alone would not explain.
    """
    out = crystal.inv_latent_transform(torch.tensor([LATENT]))
    angles = out[0, 3:6]
    assert bool((angles > 0.2 * np.pi).all()), f'angle below 0.2*pi: {angles}'
    assert bool((angles < 0.8 * np.pi).all()), f'angle above 0.8*pi: {angles}'


def test_roundtrip_is_exact_but_proves_nothing_about_the_cell(crystal):
    """The roundtrip holds -- and this test documents that it is NOT enough.

    Kept deliberately, next to the pins above, so the distinction is visible in
    one file: `latent_transform(inv_latent_transform(x)) == x` is satisfied for
    ANY consistent choice of constants, including wrong ones. It is a necessary
    check and a worthless one on its own.
    """
    lat = torch.tensor([LATENT])
    out = crystal.inv_latent_transform(lat)
    back = crystal.latent_transform(out)
    assert float((back - lat).abs().max()) < 1e-5, 'the map stopped being invertible'


def test_range_constants_are_still_duplicated_not_shared():
    """Guard the assumption the pins above are protecting.

    `au_range` and `ang_range` appear TWICE in crystal_ops.py -- once in each
    direction of the map -- with no shared constant. If someone hoists them, that
    is an improvement, but the pins above become the only thing standing between a
    value change and a silently different crystal. This test records the current
    state so a hoist is a deliberate, visible event rather than a quiet one.
    """
    import inspect

    from mxtaltools.dataset_utils.data_class_methods import crystal_ops
    src = inspect.getsource(crystal_ops)
    n_au = src.count('[[0.075, 0.075, 0.1], [3, 3, 4]]')
    n_ang = src.count('[0.2 * torch.pi, 0.8 * torch.pi]')
    assert n_au == 2 and n_ang == 2, (
        f'au_range literal appears {n_au}x and ang_range {n_ang}x; expected 2 each. '
        'If they were hoisted to a shared constant, update this test -- and note '
        'that the golden pins above are now the ONLY guard on their values.')

"""
The layout contract of `build_unit_cell`, checked geometrically rather than assumed.

WHY THIS FILE EXISTS. `unit_cell_pos` is consumed as IMAGE-MAJOR: all atoms of
symmetry image 0, then all of image 1, and so on. Every consumer re-derives the
element assignment from that assumption by tiling the asymmetric unit's `z` array --
production through `_tiled_gather_index`, and the ground-truth fixtures in
`test_uma_vs_stock_fairchem.py` / `test_mace_vs_stock.py` independently through
`np.tile`.

Two independent re-derivations of the SAME assumption are not independent evidence.
If the builder emitted atom-major order instead, both sides would pair every position
with the wrong element identically, agree with each other exactly, and score a
chemically nonsensical structure. Both L1 ground-truth gates would stay green. That is
a common-mode hole, and it is the case those gates explicitly cannot cover (see
`docs/mlip_validation.md` §6 and element 10 of the method doc).

THE CHECK. Do not compare element labels -- that is the assumption under test.
Compare GEOMETRY: every symmetry image is an isometry of the asymmetric unit, so each
contiguous n_i-row block of `unit_cell_pos` must have exactly the asymmetric unit's
intra-block pairwise distance matrix, entry for entry. Distances are preserved by
rotations, translations, inversions and mirrors alike, so this holds for every space
group operator without special-casing. It is exact, CPU-only, needs no model, and an
atom-major block is not a rigid copy of anything, so a layout inversion fails
immediately.

WHAT IT DOES NOT PIN. Elementwise equality of the distance matrix fixes the ordering
up to an automorphism of that matrix -- two chemically equivalent atoms at identical
distances from everything else (e.g. two hydrogens on the same carbon) could swap
without detection. A genuine layout inversion is not such an automorphism, so this
closes the hole it was written for; it is not a general permutation check.

CPU-only, no GPU, no checkpoint.
"""
import os

import pytest
import torch

from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
N_CRYSTALS = 12
TOL = 1e-3          # Angstrom, on distances of order 1-20 A


@pytest.fixture(scope='module')
def crystals():
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    good = []
    for c in torch.load(DATASET, weights_only=False, map_location='cpu'):
        try:
            b = collate_data_list([c.clone()])
            b.pose_aunit(std_orientation=False)
            b.build_unit_cell()
            if int(b.sym_mult[0]) > 1:      # Z=1 cannot exhibit a layout inversion
                good.append(c)
        except Exception:
            pass
    if len(good) < 4:
        pytest.skip(f'only {len(good)} buildable multi-image crystals in the fixture')
    return good[:N_CRYSTALS]


def _built(crystals):
    b = collate_data_list([c.clone() for c in crystals])
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    return b


def _pdist(x):
    return torch.cdist(x.double(), x.double())


def _image_blocks(batch, i):
    """The n_i-row blocks of crystal i's unit cell, under the image-major reading."""
    ucell = batch.unit_cell_pos[batch.unit_cell_batch == i]
    n_i = int((batch.batch == i).sum())
    mult = int(batch.sym_mult[i])
    assert ucell.shape[0] == n_i * mult, (
        f'crystal {i}: {ucell.shape[0]} unit-cell rows against n_i*sym_mult = '
        f'{n_i * mult} -- the batch is inconsistent before layout is even in question')
    return [ucell[k * n_i:(k + 1) * n_i] for k in range(mult)], n_i, mult


def test_every_image_block_is_a_rigid_copy_of_the_aunit(crystals):
    """
    THE CONTRACT. Under the image-major reading each block is one symmetry image, and
    a symmetry image is an isometry, so its internal distance matrix must equal the
    asymmetric unit's exactly. Checked entry for entry, which pins the atom ORDER
    inside the block as well as the block boundaries.
    """
    batch = _built(crystals)
    for i in range(batch.num_graphs):
        blocks, n_i, mult = _image_blocks(batch, i)
        ref = _pdist(blocks[0])
        for k, blk in enumerate(blocks[1:], start=1):
            dev = (_pdist(blk) - ref).abs().max().item()
            assert dev < TOL, (
                f'crystal {i}, image {k} of {mult}: intra-block distances differ from '
                f'image 0 by up to {dev:.4f} A. Under the image-major layout every '
                f'block is an isometry of the same asymmetric unit, so this means '
                f'`unit_cell_pos` is NOT laid out image-major -- and every consumer '
                f'that tiles the aunit element array is mislabelling atoms.')


def test_block_zero_matches_the_posed_asymmetric_unit(crystals):
    """
    Anchors the blocks to the actual aunit rather than only to each other. Without
    this, a builder that emitted `sym_mult` copies of some OTHER rigid body would
    satisfy the test above.
    """
    batch = _built(crystals)
    for i in range(batch.num_graphs):
        blocks, n_i, mult = _image_blocks(batch, i)
        aunit = batch.pos[batch.batch == i]
        dev = (_pdist(blocks[0]) - _pdist(aunit)).abs().max().item()
        assert dev < TOL, (
            f'crystal {i}: block 0 is not the posed asymmetric unit (distances differ '
            f'by up to {dev:.4f} A)')


def test_an_atom_major_layout_would_be_caught(crystals):
    """
    NEGATIVE CONTROL, and the reason the two tests above can be believed.

    Re-index one crystal's unit cell atom-major -- exactly the inversion this file
    exists to detect -- and require the rigid-copy check to reject it. Without this,
    a check that passes tells us nothing about whether it could ever fail.
    """
    batch = _built(crystals)
    i = 0
    blocks, n_i, mult = _image_blocks(batch, i)
    ucell = batch.unit_cell_pos[batch.unit_cell_batch == i]

    # image-major [img, atom] -> atom-major [atom, img], the plausible wrong layout
    inverted = ucell.view(mult, n_i, 3).transpose(0, 1).reshape(mult * n_i, 3)
    bad = [inverted[k * n_i:(k + 1) * n_i] for k in range(mult)]

    ref = _pdist(bad[0])
    worst = max((_pdist(b) - ref).abs().max().item() for b in bad[1:])
    assert worst > TOL, (
        f'an atom-major relabelling of crystal {i} still satisfies the rigid-copy '
        f'check (worst deviation {worst:.6f} A). The check cannot detect the layout '
        f'inversion it was written for, and both L1 gates remain blind to it.')


def test_the_fixture_has_enough_symmetry_to_be_meaningful(crystals):
    """Guards the fixture: at Z=1 there is only one image and no layout to invert, so
    a fixture that drifted to Z=1 crystals would make every test above vacuous."""
    batch = _built(crystals)
    mults = [int(m) for m in batch.sym_mult]
    assert min(mults) > 1, f'fixture contains Z=1 crystals: sym_mult = {mults}'
    assert max(mults) >= 4, (
        f'fixture symmetry is weak (max sym_mult {max(mults)}); a layout inversion is '
        f'easiest to miss at low multiplicity')

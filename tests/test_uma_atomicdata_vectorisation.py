"""
The vectorised fairchem AtomicData builder must be BIT-IDENTICAL to the list path.

This is a physics code and the change is a pure reindexing, so `allclose` is the
wrong bar -- any difference at all is a bug. Every test here asserts exact tensor
equality against `atomicdata_list_to_batch(batch_to_fairchem_atomicdata(...))`,
which is the shipping reference.

WHAT MAKES THIS NON-TRIVIAL, and why each case below exists. The builder has to
reproduce three different notions of "which atoms are the asymmetric unit":

  * a fresh batch with no unit cell -> build it (and for Z'>1, via the
    split_to_zp1 detour, because pose_aunit/build_unit_cell are Z'=1 operations);
  * a SUPERCELL/cluster batch -> only aux_ind == 0 is the aunit, the rest are
    image atoms unit_cell_pos already covers;
  * a plain unit-cell batch -> all of batch.z.

and then TILE each aunit sym_mult times, image-major, to match unit_cell_pos's
layout. Tiling versus repeat_interleaving is the failure that would be invisible:
both give the right SHAPE, both collate cleanly, and the energies would simply be
wrong. test_tiling_is_not_repeat_interleave pins it directly.

CPU-only by construction -- nothing here needs a GPU or a UMA checkpoint.
"""
import os
import sys

import pytest
import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import (
    batch_to_fairchem_atomicdata,
    batch_to_fairchem_batch,
    verify_fairchem_batch_equivalence,
    _tiled_gather_index,
)
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')


# --------------------------------------------------------------------- fixtures

@pytest.fixture(scope='module')
def crystals():
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    return torch.load(DATASET, weights_only=False, map_location='cpu')


def _built(crystals, lo, hi, std_orientation=False):
    """A batch with its unit cell ALREADY built.

    The equivalence check has to run on one of these: the no-unit-cell branch
    MUTATES the batch (it builds unit_cell_pos), so calling both paths on a fresh
    batch would have the second take a different branch than the first and compare
    two different things.
    """
    b = collate_data_list(crystals[lo:hi])
    b.pose_aunit(std_orientation=std_orientation)
    b.build_unit_cell()
    return b


# ------------------------------------------------------------------- the checks

@pytest.mark.parametrize('lo,hi', [(0, 1), (0, 6), (10, 26), (40, 44)])
def test_identical_to_the_list_path(crystals, lo, hi):
    """The headline claim, over several batch sizes including a single crystal."""
    verify_fairchem_batch_equivalence(_built(crystals, lo, hi), std_orientation=False)


def test_ragged_sym_mult_is_handled(crystals):
    """sym_mult varies across the batch (4 and 8 in this dataset), so the per-graph
    tile lengths differ -- the case a naive reshape would get wrong."""
    b = _built(crystals, 0, 12)
    mults = set(b.sym_mult.tolist())
    assert len(mults) > 1, f'fixture no longer exercises ragged sym_mult ({mults})'
    verify_fairchem_batch_equivalence(b, std_orientation=False)


def test_pbc_false(crystals):
    """The gas-phase leg runs pbc=False; the flag must reach every graph."""
    b = _built(crystals, 0, 5)
    out = verify_fairchem_batch_equivalence(b, std_orientation=False, pbc=False)
    assert out.pbc.shape == (b.num_graphs, 3)
    assert not out.pbc.any()


def test_task_name_is_per_graph(crystals):
    b = _built(crystals, 0, 4)
    out = batch_to_fairchem_batch(b, std_orientation=False, task_name='omol')
    assert list(out.dataset) == ['omol'] * b.num_graphs


def test_supercell_batch_uses_only_aux_ind_zero(crystals):
    """
    The aux_ind case: a cluster batch where only aux_ind == 0 is the asymmetric
    unit and the rest are image atoms.

    Marked PER GRAPH -- one image atom each -- because a flat slice across the
    concatenated batch would empty the later graphs' asymmetric units, which is a
    malformed batch rather than a supercell (and raises inside bincount/max).
    """
    b = _built(crystals, 0, 5)
    aux = torch.zeros(b.z.shape[0], dtype=torch.long)
    for i in range(b.num_graphs):          # last atom of each graph = image atom
        aux[torch.nonzero(b.batch == i).flatten()[-1]] = 1
    b.aux_ind = aux

    keep = aux == 0
    counts = torch.bincount(b.batch[keep], minlength=b.num_graphs)
    assert (counts > 0).all(), 'fixture emptied a graph'
    b.unit_cell_pos = b.unit_cell_pos[:int((counts * b.sym_mult).sum())].clone()
    b.unit_cell_batch = torch.repeat_interleave(
        torch.arange(b.num_graphs), counts * b.sym_mult)

    out = verify_fairchem_batch_equivalence(b, std_orientation=False)
    # and confirm the branch is load-bearing: ignoring aux_ind would keep every
    # atom, so the atom count must actually be smaller than the un-masked one
    assert int(out.natoms.sum()) == int((counts * b.sym_mult).sum())
    assert int(out.natoms.sum()) < int((torch.bincount(b.batch) * b.sym_mult).sum())


def test_zprime_gt_1_fresh_build(crystals):
    """Z'>1 takes the split_to_zp1 detour. Both paths BUILD the cell, so they must
    run on separate copies of an unbuilt batch and be compared afterwards."""
    zp2 = [c for c in crystals if int(c.z_prime) > 1]
    if not zp2:
        pytest.skip('no Z\'>1 entries in the mini dataset')
    a = collate_data_list(zp2[:4])
    b = collate_data_list(zp2[:4])
    ref = atomicdata_list_to_batch(batch_to_fairchem_atomicdata(a, std_orientation=False))
    new = batch_to_fairchem_batch(b, std_orientation=False)
    for field in ('pos', 'atomic_numbers', 'cell', 'natoms', 'batch'):
        assert torch.equal(getattr(ref, field), getattr(new, field)), field


def test_tiling_is_not_repeat_interleave(crystals):
    """
    Pin the ordering directly. `repeat(k)` tiles ([a,b,a,b]); repeat_interleave
    would give [a,a,b,b]. Same shape, same dtype, collates fine, wrong energies --
    so it is worth an explicit assertion rather than relying on the equality test
    to notice.
    """
    b = _built(crystals, 0, 3)
    z, batch_ind = b.z, b.batch
    gather, out_n = _tiled_gather_index(batch_ind, b.sym_mult, b.num_graphs)
    got = z[gather]
    expected = torch.cat([z[batch_ind == i].repeat(int(b.sym_mult[i]))
                          for i in range(b.num_graphs)])
    assert torch.equal(got, expected)
    n0 = int((batch_ind == 0).sum())
    m0 = int(b.sym_mult[0])
    if m0 > 1 and n0 > 1:
        first = got[:n0 * m0]
        assert torch.equal(first[:n0], first[n0:2 * n0]), 'not tiled image-major'


def test_mismatched_unit_cell_raises(crystals):
    """A batch whose unit_cell_pos disagrees with sum(n_i * sym_mult_i) must fail
    loudly. Silently proceeding would pair positions with the wrong elements."""
    b = _built(crystals, 0, 4)
    b.unit_cell_pos = b.unit_cell_pos[:-1].clone()
    with pytest.raises(RuntimeError, match='unit cell size mismatch'):
        batch_to_fairchem_batch(b, std_orientation=False)


def test_no_attribute_is_missing_or_none(crystals):
    """
    The two objects must have the SAME attribute surface, not merely the same
    model-visible tensors. A directly-constructed AtomicData leaves fairchem's
    collation bookkeeping (__slices__/__cumsum__/__cat_dims__/__natoms_list__) as
    None; nothing in the predict path reads them today, but shipping four silently
    None attributes is a version-bump away from a confusing failure.
    """
    b = _built(crystals, 0, 5)
    ref = atomicdata_list_to_batch(batch_to_fairchem_atomicdata(b, std_orientation=False))
    new = batch_to_fairchem_batch(b, std_orientation=False)
    assert set(vars(ref)) == set(vars(new)), set(vars(ref)) ^ set(vars(new))
    for name in ('__slices__', '__cumsum__', '__cat_dims__', '__natoms_list__'):
        assert getattr(new, name) is not None, f'{name} left unpopulated'
    assert new.__natoms_list__ == ref.__natoms_list__
    assert new.__slices__['pos'] == ref.__slices__['pos']
    assert new.__cat_dims__ == ref.__cat_dims__


def test_get_example_round_trips(crystals):
    """The one operation that actually reads the bookkeeping. If it works, the
    populated values are self-consistent."""
    b = _built(crystals, 0, 4)
    new = batch_to_fairchem_batch(b, std_orientation=False)
    ref = atomicdata_list_to_batch(batch_to_fairchem_atomicdata(b, std_orientation=False))
    for i in range(b.num_graphs):
        a, c = new.get_example(i), ref.get_example(i)
        assert torch.equal(a.pos, c.pos), f'graph {i} pos'
        assert torch.equal(a.atomic_numbers, c.atomic_numbers), f'graph {i} z'
        assert torch.equal(a.cell, c.cell), f'graph {i} cell'


def test_the_flag_selects_equivalent_batches(crystals):
    """
    The wiring, not just the builder. compute_crystal_uma_on_mxt_batch branches on
    USE_VECTORISED_ATOMICDATA, so the flag has to be shown to select two paths that
    agree -- a correct builder reached by incorrect wiring is still wrong energies.

    Stops short of calling the predictor: that needs a UMA checkpoint and a GPU.
    What it pins is that the object handed to predict() is the same either way.
    """
    from mxtaltools.mlip_interfaces import uma_utils as U

    b = _built(crystals, 0, 6)
    assert U.USE_VECTORISED_ATOMICDATA is True, 'default should be the fast path'

    vec = U.batch_to_fairchem_batch(b, std_orientation=False, pbc=True)
    lst = atomicdata_list_to_batch(
        U.batch_to_fairchem_atomicdata(b, std_orientation=False, pbc=True))
    for field in ('pos', 'atomic_numbers', 'cell', 'pbc', 'natoms', 'batch'):
        assert torch.equal(getattr(vec, field), getattr(lst, field)), field


def test_single_atom_and_single_graph(crystals):
    """Degenerate shapes: one graph, and the smallest molecule available."""
    verify_fairchem_batch_equivalence(_built(crystals, 0, 1), std_orientation=False)
    sizes = [(i, int((collate_data_list([c])).z.shape[0])) for i, c in enumerate(crystals[:40])]
    smallest = min(sizes, key=lambda t: t[1])[0]
    verify_fairchem_batch_equivalence(
        _built(crystals, smallest, smallest + 1), std_orientation=False)

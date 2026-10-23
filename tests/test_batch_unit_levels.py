"""Stage 3 of the Z'>1 units plan: the batch helpers' five levels (gfn_diffusion
energy_sampling/docs/design/crystal_units_audit.md, section 5 stage 3; audit items #3, #4, #7, #8).

What is pinned here, on mixed-Z' crystal batches (Z' 1, 2 and 3 at max_z_prime 3) that carry a value at every level:

- ATOM (``pos``, ``z``, ``x``, the unit map ``unit_ind``), UNIT (``unit_score``: [z', 2] per row, collated to
  [sum z', 2] with the unit pointer as its slice entry), EDGE (``edge_index`` [2, E] atom indices, offset per row by
  collation; ``edge_attr`` [E, 3] per-edge values, not offset; rows of 0, 5, 7, 10, 14 and 15 edges), ROW (cell,
  padded per-unit pose slots, ``identifier``) and BATCH (``max_z_prime``, a 231-entry table set on the batch);
- ``subsample_new_batch(idx)`` equals collating the selected rows, every value and every slice / increment entry
  (repeats, reversed order, one row, a mask, an empty selection), and ``batch_to_list`` of the result gives the
  selected rows back;
- ``append_batch`` equals collating both row lists, in both orders, including a side whose rows are all Z'=1 (its unit
  pointer is its row pointer), an empty side, and sides with and without the unit map;
- ``collate_data_list`` of rows with and without the unit map, in both orders;
- ``batch_to_list`` -> collate round trip;
- ``MolCrystalIO.select_rows`` on a ``from_arrays`` batch carrying unit and edge values equals
  ``subsample_new_batch``; the ingest store saves and reloads them bit for bit and names their levels;
- the list constructor's edge offsets (audit #7) and its refusal of merges it cannot do;
- ``add_unit_attr``, ``declare_level`` and the registry (a lookup table on a 231-row batch stays whole);
- an index tensor without a slice entry: its per-row counts read from its values;
- the stage-3 review: a slice entry that describes a value decides its level before a coincidental length (per-edge
  and per-unit values whose totals equal the atom or row count, through every helper, the store and the blank-batch
  route of ``set_mol_attrs``), a device-held entry is not read where the length decides, ``append_batch`` keeps a
  registered constant whole, an 'index' key whose increments are not atom offsets is refused, and MXtalTools'
  DataLoaders fill the unit map of mixed rows;
- every new refusal: a ``BatchLevelError`` / ``ValueError`` raised by a ``raise`` statement in mxtaltools, and the same
  under ``python -O`` (a child interpreter runs every one with ``-O``).

CPU only; synthetic molecules without point symmetry (the shapes of tests/test_unit_map.py), a triclinic cell.
"""
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]

_ZA = torch.tensor([6, 6, 8, 7, 6, 1])
_PA = torch.tensor([[0., 0, 0], [1.5, 0, 0], [2.2, 1.2, 0], [-0.7, 1.2, 0.3], [2.1, -1.3, 0.8], [-0.5, -0.9, -0.4]])
_PA2 = _PA + torch.tensor([[0., 0, 0], [0.05, -0.02, 0.01], [0.1, 0.05, -0.08], [-0.04, 0.07, 0.02],
                           [0.03, 0.02, -0.05], [0.0, 0.04, 0.03]])
_ZB = torch.tensor([6, 6, 6, 6, 8, 8, 1, 1])
_PB = torch.tensor([[0., 0, 0], [1.4, 0, 0], [2.1, 1.2, 0], [1.4, 2.4, 0.2], [3.5, 1.2, 0.1], [0.0, 2.4, -0.3],
                    [-0.5, -0.9, 0], [3.9, 0.3, 0]])
_MOLS = {'A': (_ZA, _PA), 'A2': (_ZA, _PA2), 'B': (_ZB, _PB)}
_LENGTHS = torch.tensor([10.5, 11.0, 12.5])
_ANGLES = torch.tensor([1.62, 1.74, 1.52])
_CENTROIDS = (0.12, 0.20, 0.31, 0.40, 0.70, 0.15, 0.62, 0.55, 0.71)
_ORIENTATIONS = (-1.2, 0.5, 0.4, 0.3, 0.2, 1.0, 0.7, -0.6, 0.9)
WIDTH = 3

#: the rows: (units, edges per molecule?) -> Z' 1, 2, 3, 1, 2, 1; edges 5, 10, 15, 7, 14, 0
_ROWS = ((('A',), True), (('A', 'A2'), True), (('A2', 'A', 'A2'), True), (('B',), True), (('B', 'B'), True),
         (('A2',), False))


def _chain(n):
    return torch.stack([torch.arange(n - 1), torch.arange(1, n)])


def _mol(name, edges=True):
    from mxtaltools.dataset_utils.data_classes import MolData
    z, pos = _MOLS[name]
    n = len(z)
    return MolData(z=z.clone(), pos=pos.clone(), x=torch.arange(n, dtype=torch.float32)[:, None] + 100 * len(name),
                   edge_index=_chain(n) if edges else torch.zeros(2, 0, dtype=torch.long), do_mol_analysis=True)


def row(k, names=None, edges=None, extras=True, centroids=_CENTROIDS):
    """Crystal row ``k`` of ``_ROWS`` through the constructor (the list constructor for Z' > 1), max_z_prime 3, with a
    per-unit ``unit_score`` [z', 2] and a per-edge ``edge_attr`` [E, 3] of values unique to the row."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    names = _ROWS[k][0] if names is None else names
    edges = _ROWS[k][1] if edges is None else edges
    zp = len(names)
    mols = [_mol(n, edges) for n in names]
    r = MolCrystalData(molecule=mols if zp > 1 else mols[0], sg_ind=2, z_prime=torch.tensor([zp]),
                       max_z_prime=WIDTH, cell_lengths=_LENGTHS.clone() + k, cell_angles=_ANGLES.clone(),
                       aunit_centroid=torch.tensor(centroids[:3 * zp]),
                       aunit_orientation=torch.tensor(_ORIENTATIONS[:3 * zp]),
                       aunit_handedness=torch.tensor([1, -1, 1][:zp]), identifier=f'ROW{k}', do_box_analysis=True)
    if extras:
        r.unit_score = torch.arange(2 * zp, dtype=torch.float32).reshape(zp, 2) + 100 * k
        E = r.edge_index.shape[1]
        r.edge_attr = torch.arange(3 * E, dtype=torch.float32).reshape(E, 3) + 1000 * k
    return r


def rows(ks=range(len(_ROWS)), **kw):
    return [row(k, **kw) for k in ks]


def collate(rs, max_z_prime=WIDTH):
    from mxtaltools.dataset_utils.utils import collate_data_list
    return collate_data_list([r.clone() for r in rs], max_z_prime=max_z_prime)


def _without_map(b):
    c = b.clone()
    c._store.pop('unit_ind', None)
    if c.__dict__.get('_slice_dict') is not None:
        c._slice_dict.pop('unit_ind', None)
        c._inc_dict.pop('unit_ind', None)
    return c


def _same(a, b):
    """Equal bit for bit: type, dtype, shape and values (NaN equal to NaN); containers recursively."""
    if torch.is_tensor(a):
        if not (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape):
            return False
        if a.is_floating_point():
            return bool(((a == b) | (a.isnan() & b.isnan())).all())
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and \
            bool(np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc'))
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def assert_same_batch(a, b, skip=()):
    """Every stored value, ``_num_graphs`` and every slice / increment entry equal (increments of a list: None from
    collation, zeros from the helpers' rebuild; both mean no offset). A batch-level python value (``max_z_prime``)
    has an entry after collation, which collates the rows' ints before ``collate_data_list`` sets the batch's, and none
    after a helper: its entries are not compared (``batch_to_list`` drops it before splitting)."""
    assert a.num_graphs == b.num_graphs
    assert set(a._store.keys()) == set(b._store.keys()), set(a._store.keys()) ^ set(b._store.keys())
    for k in a._store.keys():
        if k not in skip:
            assert _same(a._store[k], b._store[k]), k
    constants = {k for k, v in a._store.items() if not torch.is_tensor(v) and not isinstance(v, (list, tuple))}
    sa, sb = set(a._slice_dict) - constants, set(b._slice_dict) - constants
    assert sa == sb, sa ^ sb
    for k in sa:
        assert torch.equal(a._slice_dict[k].cpu().long(), b._slice_dict[k].cpu().long()), f'_slice_dict[{k}]'
        ia, ib = a._inc_dict.get(k), b._inc_dict.get(k)
        if torch.is_tensor(a._store[k]):
            assert torch.equal(ia.cpu().long(), ib.cpu().long()), f'_inc_dict[{k}]'
        else:
            for i in (ia, ib):
                assert i is None or not bool(i.any()), f'_inc_dict[{k}]'


def assert_rows_equal(got_rows, want_rows):
    """``batch_to_list`` elements against the rows they came from (as collation and separation give them)."""
    assert len(got_rows) == len(want_rows)
    for g, w in zip(got_rows, want_rows):
        assert set(g.keys()) == set(w.keys()), set(g.keys()) ^ set(w.keys())
        for k in w.keys():
            assert _same(g[k], w[k]), k


# ------------------------------------------------------------------------------------------------
# the levels
# ------------------------------------------------------------------------------------------------

def test_levels_of_a_mixed_batch():
    from mxtaltools.dataset_utils.data_classes import ATOM, UNIT, EDGE, ROW, BATCH
    b = collate(rows())
    b.lut = torch.arange(231)
    assert b.z_prime.tolist() == [1, 2, 3, 1, 2, 1] and b.max_z_prime == 3
    plan = b._level_plan(b._ptr_cpu(), int(b.ptr[-1]), b.num_graphs)
    levels = {k: lv for k, (lv, _) in plan.items()}
    for k in ('pos', 'z', 'x', 'unit_ind'):
        assert levels[k] == ATOM, k
    for k in ('cell_lengths', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness', 'identifier', 'z_prime',
              'num_atoms', 'radius', 'symmetry_operators'):
        assert levels[k] == ROW, k
    assert levels['unit_score'] == UNIT and torch.equal(plan['unit_score'][1], torch.tensor([0, 1, 3, 6, 7, 9, 10]))
    assert levels['edge_index'] == EDGE and levels['edge_attr'] == EDGE
    assert plan['edge_index'][1].tolist() == [0, 5, 15, 30, 37, 51, 51] == plan['edge_attr'][1].tolist()
    assert levels['max_z_prime'] == BATCH and levels['lut'] == BATCH
    # collation recorded the unit pointer for the per-unit value and atom offsets as the index key's increments
    assert torch.equal(b._slice_dict['unit_score'], torch.tensor([0, 1, 3, 6, 7, 9, 10]))
    assert torch.equal(b._inc_dict['edge_index'], b.ptr[:-1]) and not bool(b._inc_dict['edge_attr'].any())


# ------------------------------------------------------------------------------------------------
# subsample_new_batch: the selected rows, as collating them gives them
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('idx', [[2, 0], [1], [5, 2, 2, 0, 4], list(range(6))[::-1], 'mask', 'tensor'],
                         ids=['2-0', 'one', 'repeats', 'reversed', 'mask', 'tensor'])
def test_subsample_equals_collating_the_selected_rows(idx):
    rs = rows()
    b = collate(rs)
    if idx == 'mask':
        sel, idx = [1, 2, 4], torch.tensor([False, True, True, False, True, False])
    elif idx == 'tensor':
        sel, idx = [4, 1], torch.tensor([4, 1])
    else:
        sel = idx
    sub = b.subsample_new_batch(idx)
    ref = collate([rs[i] for i in sel])
    assert_same_batch(sub, ref)
    assert_rows_equal(sub.batch_to_list(), ref.batch_to_list())
    # the per-unit values and edges of the selected rows, in their own numbering
    for got, i in zip(sub.batch_to_list(), sel):
        assert torch.equal(got.unit_score, rs[i].unit_score) and torch.equal(got.edge_attr, rs[i].edge_attr)
        assert torch.equal(got.edge_index, rs[i].edge_index)
    # and a second draw from the draw
    again = sub.subsample_new_batch([len(sel) - 1])
    assert_same_batch(again, collate([rs[sel[-1]]]))


def test_subsample_of_nothing():
    b = collate(rows())
    empty = b.subsample_new_batch([])
    assert empty.num_graphs == 0 and empty.pos.shape == (0, 3) and empty.unit_score.shape == (0, 2)
    assert empty.edge_index.shape == (2, 0) and empty.edge_attr.shape == (0, 3) and empty.identifier == []
    for k in ('unit_score', 'edge_index', 'edge_attr', 'pos', 'cell_lengths'):
        assert empty._slice_dict[k].tolist() == [0], k
    assert empty.batch_to_list() == []
    # appending it changes nothing; appending to it gives the other batch
    assert_same_batch(b.append_batch(empty), b)
    assert_same_batch(empty.append_batch(b), b)


def test_an_index_tensor_on_two_rows_and_on_two_atoms():
    """Audit #3: a [2, E] index tensor on a 2-row batch was gathered by row ([2, 12] -> [1, 12]), and one on a 2-atom
    batch would be gathered by atom; it is per edge whatever the counts."""
    from mxtaltools.dataset_utils.data_classes import MolData
    from mxtaltools.dataset_utils.utils import collate_data_list
    two_rows = collate([row(0), row(3)])
    assert two_rows.edge_index.shape == (2, 12)
    one = two_rows.subsample_new_batch([1])
    assert torch.equal(one.edge_index, _chain(8)) and one._slice_dict['edge_index'].tolist() == [0, 7]
    m = MolData(z=torch.tensor([6, 8]), pos=torch.tensor([[0., 0, 0], [1.2, 0, 0]]),
                edge_index=torch.tensor([[0, 1], [1, 0]]))
    two_atoms = collate_data_list([m])
    assert int(two_atoms.ptr[-1]) == 2 == two_atoms.edge_index.shape[0]
    rep = two_atoms.subsample_new_batch([0, 0, 0])
    assert torch.equal(rep.edge_index, torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]]))
    assert [e.edge_index.tolist() for e in rep.to_data_list()] == [[[0, 1], [1, 0]]] * 3


def test_an_index_tensor_without_an_entry_is_cut_by_its_values():
    """An index tensor set on a batch after collation has no slice entry: each column belongs to the row of the atoms it
    indexes (columns grouped by row); otherwise it is refused (``_REFUSALS``)."""
    rs = rows()
    b = collate(rs)
    ei = b.edge_index.clone()
    del b.edge_index
    b._slice_dict.pop('edge_index'), b._inc_dict.pop('edge_index')
    b.bond_index = ei                                       # no entry
    sub = b.subsample_new_batch([4, 1])
    assert torch.equal(sub.bond_index, collate([rs[4], rs[1]]).edge_index)
    assert sub._slice_dict['bond_index'].tolist() == [0, 14, 24]


# ------------------------------------------------------------------------------------------------
# append_batch: both row lists, as collating them gives them
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('split', [(range(0, 3), range(3, 6)), ((0, 3, 5), (1, 2, 4)), ((1,), (0, 2, 3, 4, 5))],
                         ids=['halves', 'all-zp1-side', 'one-row-side'])
def test_append_equals_collating_both_lists(split):
    rs = rows()
    first, second = ([rs[i] for i in s] for s in split)
    for a, c in ((first, second), (second, first)):
        joined = collate(a).append_batch(collate(c))
        ref = collate(a + c)
        assert_same_batch(joined, ref)
        assert_rows_equal(joined.batch_to_list(), ref.batch_to_list())
        # and a draw from the join
        assert_same_batch(joined.subsample_new_batch([len(a), 0]), collate([c[0], a[0]]))


def test_append_with_and_without_the_unit_map():
    """A side without the map (an older file) gets its derived map (stage-2 review), and the ragged values still join."""
    rs = rows()
    a, c = collate(rs[:3]), _without_map(collate(rs[3:]))
    for x, y, want in ((a, c, rs), (c, a, rs[3:] + rs[:3])):
        joined = x.append_batch(y)
        ref = collate(want)
        assert_same_batch(joined, ref)


def test_append_refuses_a_ragged_value_held_otherwise():
    """A per-unit value one side holds per unit and the other as a constant (no entry, another length): refused, not
    kept once as shared metadata (audit #3)."""
    from mxtaltools.dataset_utils.data_classes import BatchLevelError
    a, c = collate(rows(range(1, 3))), collate(rows(range(3, 5)))
    c._slice_dict.pop('unit_score'), c._inc_dict.pop('unit_score')
    c.unit_score = torch.zeros(5, 2)
    for x, y in ((a, c), (c, a)):
        with pytest.raises(BatchLevelError, match="'unit_score' is ragged"):
            x.append_batch(y)


# ------------------------------------------------------------------------------------------------
# collate and batch_to_list
# ------------------------------------------------------------------------------------------------

def test_collate_rows_with_and_without_the_map_in_both_orders():
    from mxtaltools.crystal_building.utils import derived_unit_ind
    rs = rows()
    old = [_without_map(r) for r in rs[1:3]]
    for order in ([rs[0]] + old + [rs[3]], old + [rs[0], rs[3]]):
        b = collate(order)
        want = torch.cat([r.unit_ind if 'unit_ind' in r._store else derived_unit_ind(r) for r in order])
        assert _same(b.unit_ind, want) and torch.equal(b._slice_dict['unit_ind'], b.ptr)
        assert all('unit_ind' not in r._store for r in old)        # the inputs are not given it
        # and the batch reads as the map-less batch reads (the absence path)
        s, ns = b.split_to_zp1_batch(), _without_map(b).split_to_zp1_batch()
        for k in ('pos', 'z', 'batch', 'ptr', 'num_atoms', 'aunit_centroid'):
            assert _same(s[k], ns[k]), k


def test_batch_to_list_round_trip():
    rs = rows()
    b = collate(rs)
    back = b.batch_to_list()
    for got, r in zip(back, rs):
        for k in ('unit_score', 'edge_attr', 'edge_index', 'unit_ind', 'pos', 'aunit_centroid'):
            assert torch.equal(got[k].reshape(-1), r[k].reshape(-1).to(got[k].dtype)), k
    assert_same_batch(collate(back), b)


def test_batch_to_list_refuses_a_stale_entry():
    from mxtaltools.dataset_utils.data_classes import BatchLevelError
    b = collate(rows())
    b.unit_score = b.unit_score[:-1]                     # one unit's value gone, the entry left as it was
    with pytest.raises(BatchLevelError, match="slice entry of 'unit_score'"):
        b.batch_to_list()


# ------------------------------------------------------------------------------------------------
# select_rows and the store, on a from_arrays batch
# ------------------------------------------------------------------------------------------------

def _from_arrays_with_ragged_values():
    """A posed from_arrays batch of 4 rows (Z' 1, 2, 1, 3; 6-atom units), with a per-unit value (add_unit_attr) and
    edge values written as collation writes them."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    zps = [1, 2, 1, 3]
    xyz = torch.cat([torch.cat([_PA + 3.0 * u for u in range(z)]) for z in zps])
    z = torch.cat([_ZA.repeat(zp) for zp in zps])
    counts = torch.tensor([6 * zp for zp in zps])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    B = len(zps)
    b = MolCrystalData.from_arrays(torch.cat([_LENGTHS, _ANGLES])[None].repeat(B, 1), xyz, z, ptr=ptr, sg=2,
                                   setting='standard', z_prime=torch.tensor(zps),
                                   aunit_centroid=torch.tensor([_CENTROIDS] * B),
                                   aunit_orientation=torch.tensor([_ORIENTATIONS] * B),
                                   aunit_handedness=torch.tensor([[1, -1, 1]] * B), identifier=list('pqrs'))
    b.add_unit_attr(torch.arange(14, dtype=torch.float32).reshape(7, 2), 'unit_score')
    e = [_chain(int(n)) for n in counts]
    b['edge_index'] = torch.cat([ei + int(ptr[i]) for i, ei in enumerate(e)], dim=1)
    ecount = torch.tensor([ei.shape[1] for ei in e])
    b._slice_dict['edge_index'] = torch.cat([torch.zeros(1, dtype=torch.long), ecount.cumsum(0)])
    b._inc_dict['edge_index'] = ptr[:-1].clone()
    b['edge_attr'] = torch.arange(int(ecount.sum()), dtype=torch.float32)
    b._slice_dict['edge_attr'] = b._slice_dict['edge_index'].clone()
    b._inc_dict['edge_attr'] = torch.zeros(B, dtype=torch.long)
    return b


@pytest.mark.parametrize('idx', [[3, 0], [1, 1], [], [0, 1, 2, 3]], ids=['3-0', 'repeat', 'none', 'all'])
def test_select_rows_equals_subsample(idx):
    b = _from_arrays_with_ragged_values()
    chosen, sub = b.select_rows(idx), b.subsample_new_batch(idx)
    assert_same_batch(chosen, sub)
    if idx:
        assert chosen.unit_score.tolist() == [v for i in idx for v in
                                              b.unit_score[int(b.unit_ptr[i]):int(b.unit_ptr[i + 1])].tolist()]
        for got, i in zip(chosen.to_data_list(), idx):
            assert torch.equal(got.edge_index, _chain(6 * int(b.z_prime[i])))


def test_store_keeps_unit_and_edge_values(tmp_path):
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    b = _from_arrays_with_ragged_values()
    meta = b.save(tmp_path / 'store')
    back = MolCrystalData.load(tmp_path / 'store')
    assert list(back._store.keys()) == list(b._store.keys())
    assert_same_batch(back, b)
    levels = {k: v['level'] for k, v in meta['levels'].items()}
    assert levels['unit_score'] == 'unit' and levels['edge_index'] == 'edge' and levels['edge_attr'] == 'edge'
    assert levels['unit_ind'] == 'atom' and levels['cell_lengths'] == 'row'
    assert_same_batch(back.select_rows([2, 3]), b.subsample_new_batch([2, 3]))


def test_from_arrays_unit_map_and_select_rows_refusals():
    b = _from_arrays_with_ragged_values()
    blocks = [torch.arange(zp).repeat_interleave(6) for zp in (1, 2, 1, 3)]     # from_arrays writes its layout
    assert torch.equal(b.unit_ind, torch.cat(blocks)) and torch.equal(b.unit_ind, b.mol_ind)
    stale = b.clone()
    stale._inc_dict['edge_index'] = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="increments are not its rows' atom offsets"):
        stale.select_rows([1])


# ------------------------------------------------------------------------------------------------
# the list constructor (audit #7)
# ------------------------------------------------------------------------------------------------

def _crystal_with_trees(names):
    """A crystal of the molecules ``names`` (each with edge_index and tree_bond_index), list constructor for several."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    mols = [_mol(n) for n in names]
    for m in mols:
        m.tree_bond_index = m.edge_index.flip(0).clone()
    zp = len(mols)
    return MolCrystalData(molecule=mols if zp > 1 else mols[0], sg_ind=2, z_prime=torch.tensor([zp]),
                          max_z_prime=3, cell_lengths=_LENGTHS.clone(), cell_angles=_ANGLES.clone(),
                          aunit_centroid=torch.tensor(_CENTROIDS[:3 * zp]),
                          aunit_orientation=torch.tensor(_ORIENTATIONS[:3 * zp]),
                          aunit_handedness=torch.tensor([1, -1, 1][:zp]), do_box_analysis=True)


def test_list_constructor_offsets_index_keys_and_collates():
    """Audit #7: the list constructor joins every 'index' key along its last axis, molecule k offset by the atoms
    before it (edge_index and tree_bond_index alike); the crystal then collates, draws and splits with its rows."""
    c = _crystal_with_trees(('A', 'B', 'A2'))
    want = torch.cat([_chain(6), _chain(8) + 6, _chain(6) + 14], dim=1)
    assert torch.equal(c.edge_index, want) and torch.equal(c.tree_bond_index, want.flip(0))
    assert torch.equal(c.edge_index, row(0, names=('A', 'B', 'A2'), extras=False).edge_index)
    assert c.unit_ind.tolist() == [0] * 6 + [1] * 8 + [2] * 6
    one = _crystal_with_trees(('B',))
    b = collate([one, c])
    assert torch.equal(b.edge_index[:, 7:], want + 8)
    assert torch.equal(b.subsample_new_batch([1]).tree_bond_index, want.flip(0))
    assert torch.equal(b.subsample_new_batch([1, 0]).edge_index, torch.cat([want, _chain(8) + 20], dim=1))
    assert torch.equal(b.batch_to_list()[1].edge_index, want)


def test_set_mol_attrs_batch_path_keeps_ragged_bookkeeping():
    """GFN's init_blank_crystal_batch route copies a molecule batch's values onto a crystal batch: an index tensor and
    a per-edge value keep the source's slice / increment entries (an index tensor of a 2-graph batch was given a
    per-row entry by its length), so the crystal batch draws and splits them by row."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    from mxtaltools.dataset_utils.utils import collate_data_list
    mols = [_mol('A'), _mol('B')]
    for m, n in zip(mols, (5, 7)):
        m.bond_attr = torch.arange(float(n))
    src = collate_data_list(mols)
    blank = collate_data_list([MolCrystalData(max_z_prime=1)], max_z_prime=1)
    blank.__dict__['_num_graphs'] = src.num_graphs
    blank['device'] = src.device
    blank.set_mol_attrs(src.clone())
    del blank['device']
    for k in ('edge_index', 'bond_attr'):
        assert torch.equal(blank._slice_dict[k], src._slice_dict[k]), k
        assert torch.equal(blank._inc_dict[k], src._inc_dict[k]), k
    assert torch.equal(blank.subsample_new_batch([1]).edge_index, _chain(8))
    assert blank.subsample_new_batch([1]).bond_attr.tolist() == list(range(7))


def test_canonicalization_moves_a_per_unit_value_with_its_unit():
    """canonicalize_zp_aunits reorders a moving crystal's units: a ragged per-unit value follows them (unit j of the
    canonical order takes the value of the unit that moved into slot j), as the atom blocks and the padded slots do."""
    rev = (0.62, 0.55, 0.71, 0.40, 0.70, 0.15, 0.12, 0.20, 0.31)        # slot order not canonical (norms descend)
    b = collate([row(0), row(2, centroids=rev), row(1, centroids=rev)])
    b.x = torch.arange(len(b.z), dtype=torch.float32)[:, None]              # a tag per atom
    from mxtaltools.dataset_utils.data_class_methods.crystal_ops import _canonical_unit_order
    order = _canonical_unit_order(b)
    assert order[1].tolist() == [2, 1, 0] and order[2, :2].tolist() == [1, 0]
    before = b.unit_score.clone()
    c = b.clone()
    c.canonicalize_zp_aunits()
    uptr = [0, 1, 4, 6]
    want = torch.cat([before[0:1], before[1:4][[2, 1, 0]], before[4:6][[1, 0]]])
    assert torch.equal(c.unit_score, want)
    # consistent with the atoms: the first atom of each new block is the first atom of the old block it came from
    for r, (s, e) in enumerate(zip(uptr[:-1], uptr[1:])):
        a0 = int(b.ptr[r])
        n_u = (int(b.ptr[r + 1]) - a0) // (e - s)
        for j in range(e - s):
            old = int(order[r, j]) if e - s > 1 else 0
            assert float(c.x[a0 + j * n_u, 0]) == float(b.x[a0 + old * n_u, 0])
    assert torch.equal(c._slice_dict['unit_score'], torch.tensor(uptr))


# ------------------------------------------------------------------------------------------------
# add_unit_attr, declare_level and the registry
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def declared():
    """Keys declared by a test, removed from the process-wide registry after it."""
    from mxtaltools.dataset_utils.data_classes import KEY_LEVELS
    added = []
    yield added
    for k in added:
        KEY_LEVELS.pop(k, None)


def test_add_unit_attr_and_declare_level(declared):
    from mxtaltools.dataset_utils.data_classes import UNIT, EDGE, BATCH, BatchLevelError, declare_level
    rs = rows(extras=False)
    b = collate(rs)
    b.add_unit_attr(torch.arange(10.0), 'unit_energy')
    assert b.subsample_new_batch([2, 1]).unit_energy.tolist() == [3.0, 4.0, 5.0, 1.0, 2.0]
    assert [r.unit_energy.tolist() for r in b.batch_to_list()] == [[0.], [1., 2.], [3., 4., 5.], [6.], [7., 8.], [9.]]
    # a per-unit value set without an entry is batch-level; declared, it is sliced by the unit pointer of z_prime.
    # (Carrying it whole is the accepted gap of the level contract's rule 5, pending the owner's decision on refusing
    # entry-less tensors of another length; tests/characterization/README.md, "The batch helpers' level contract".)
    b.unit_free = torch.arange(10.0)
    assert b.subsample_new_batch([1]).unit_free.tolist() == list(range(10))
    declare_level('unit_free', UNIT)
    declared.append('unit_free')
    assert b.subsample_new_batch([1]).unit_free.tolist() == [1.0, 2.0]
    # a declared constant whose length is the row count stays whole; undeclared, the length rule gathers it per row
    big = collate(rows([0], extras=False)).subsample_new_batch([0] * 231)
    big.my_table, big.sym_mult_lut = torch.arange(231), torch.arange(231)
    sub = big.subsample_new_batch([5])
    assert sub.sym_mult_lut.tolist() == list(range(231)) and sub.my_table.tolist() == [5]
    declare_level('my_table', BATCH)
    declared.append('my_table')
    assert big.subsample_new_batch([5]).my_table.tolist() == list(range(231))
    # a per-edge value whose total equals the atom count is cut by the counts its entry records, declared or not (the
    # stage-3 review, 2026-10-23: it was cut per atom by its length, e.coincide[6:18], until it was declared)
    e = collate(rows(range(3)))
    total = int(e.ptr[-1])
    e['coincide'] = torch.arange(float(total))
    e._slice_dict['coincide'] = torch.tensor([0, 10, 20, total])
    e._inc_dict['coincide'] = torch.zeros(3, dtype=torch.long)
    assert e.subsample_new_batch([1]).coincide.tolist() == list(range(10, 20))           # by its entry
    declare_level('coincide', EDGE)
    declared.append('coincide')
    assert e.subsample_new_batch([1]).coincide.tolist() == list(range(10, 20))
    with pytest.raises(BatchLevelError, match="registered at level 'edge'"):
        declare_level('coincide', UNIT)


def test_registered_unit_cell_is_refused_by_the_moving_helpers():
    from mxtaltools.dataset_utils.data_classes import BatchLevelError
    b = collate(rows(range(2), extras=False))
    b.mol2ucell()
    for call in (lambda: b.subsample_new_batch([1]), lambda: b.append_batch(b)):
        with pytest.raises(BatchLevelError, match='per unit-cell atom'):
            call()
    b.rebuild_simple_slice_inc_()                            # bookkeeping only: no entry for the unit cell
    assert 'unit_cell_pos' not in b._slice_dict


# ------------------------------------------------------------------------------------------------
# the stage-3 review (2026-10-23): a slice entry that describes a value decides its level before a coincidental
# length; append_batch keeps registered constants whole; an index key's increments must be its rows' atom offsets;
# MXtalTools' DataLoaders fill the unit map of rows that lack it
# ------------------------------------------------------------------------------------------------

def _mol_edges(n, extra, k):
    """A molecule of ``n`` carbons on a chain (``n - 1`` edges, plus two ring-closing edges when ``extra``), with a
    per-edge ``edge_attr`` [E] unique to ``k``."""
    from mxtaltools.dataset_utils.data_classes import MolData
    ei = _chain(n)
    if extra:
        ei = torch.cat([ei, torch.tensor([[0, 2], [n - 1, n - 3]])], 1)
    g = torch.Generator().manual_seed(k)
    return MolData(z=torch.full((n,), 6), pos=torch.randn(n, 3, generator=g), edge_index=ei,
                   edge_attr=torch.arange(ei.shape[1], dtype=torch.float32) + 100 * k)


def _collate_mols(ms):
    from mxtaltools.dataset_utils.utils import collate_data_list
    return collate_data_list([m.clone() for m in ms])


def test_a_per_edge_value_whose_total_is_the_atom_count_is_cut_by_its_entry():
    """Two molecules of 6 and 8 atoms with 5 and 9 edges: 14 edges on 14 atoms. The collated entry [0, 5, 14] decides
    (EDGE), not the length (it was ATOM: a draw of row 1 gave 8 of its 9 values)."""
    from mxtaltools.dataset_utils.data_classes import EDGE
    ms = [_mol_edges(6, False, 0), _mol_edges(8, True, 1)]
    b = _collate_mols(ms)
    assert int(b.ptr[-1]) == b.edge_attr.shape[0] == 14 and b._slice_dict['edge_attr'].tolist() == [0, 5, 14]
    level, pointer = b._level_plan(b._ptr_cpu(), 14, 2)['edge_attr']
    assert level == EDGE and pointer.tolist() == [0, 5, 14]
    for idx in ([1], [1, 0], [0, 1, 1]):
        sub = b.subsample_new_batch(idx)
        assert_same_batch(sub, _collate_mols([ms[i] for i in idx]))
        for got, i in zip(sub.to_data_list(), idx):
            assert torch.equal(got.edge_attr, ms[i].edge_attr) and torch.equal(got.edge_index, ms[i].edge_index)
    # the bookkeeping survives a rebuild in place (it was rewritten [0, 6, 14] by the length)
    c = b.clone().rebuild_simple_slice_inc_()
    assert c._slice_dict['edge_attr'].tolist() == [0, 5, 14]
    assert [e.edge_attr.shape[0] for e in c.to_data_list()] == [5, 9]
    # a 3-row batch (19 atoms, 18 edges) whose draw of rows 0, 1 has as many edges as atoms, and a draw of that draw
    ms3 = ms + [_mol_edges(5, False, 2)]
    d = _collate_mols(ms3).subsample_new_batch([0, 1])
    assert int(d.ptr[-1]) == d.edge_attr.shape[0] == 14
    assert torch.equal(d.subsample_new_batch([1]).edge_attr, ms[1].edge_attr)
    # append: two such batches join their own counts, as collating the four rows does
    j = _collate_mols(ms).append_batch(_collate_mols(ms[::-1]))
    assert_same_batch(j, _collate_mols(ms + ms[::-1]))


def test_a_per_edge_value_whose_total_is_the_row_count_is_cut_by_its_entry():
    """Rows of 0, 1 and 2 edges: 3 edges on 3 rows (it was ROW: a draw of row 2 gave one value)."""
    from mxtaltools.dataset_utils.data_classes import MolData
    ms = []
    for k, (n, E) in enumerate(((4, 0), (5, 1), (6, 2))):
        ei = _chain(n)[:, :E]
        ms.append(MolData(z=torch.full((n,), 6), pos=torch.randn(n, 3), edge_index=ei,
                          edge_attr=torch.arange(E, dtype=torch.float32) + 100 * k))
    b = _collate_mols(ms)
    assert b.num_graphs == b.edge_attr.shape[0] == 3
    for idx in ([2], [0], [2, 0, 1]):
        assert_same_batch(b.subsample_new_batch(idx), _collate_mols([ms[i] for i in idx]))
    assert_same_batch(b.append_batch(b), _collate_mols(ms + ms))


def test_a_per_edge_crystal_value_whose_total_is_the_row_count():
    """Two crystal rows holding 0 and 2 contacts: 2 values on 2 rows (a draw of row 0 gave [5.])."""
    r0, r1 = row(0, extras=False), row(3, extras=False)
    r0.contact_attr = torch.zeros(0)
    r1.contact_attr = torch.tensor([5., 6.])
    b = collate([r0, r1])
    assert b._slice_dict['contact_attr'].tolist() == [0, 0, 2]
    assert b.subsample_new_batch([0]).contact_attr.tolist() == []
    assert b.subsample_new_batch([1]).contact_attr.tolist() == [5., 6.]
    assert [e.contact_attr.tolist() for e in b.append_batch(b).batch_to_list()] == [[], [5., 6.], [], [5., 6.]]


def _from_arrays_zps(zps):
    """A posed from_arrays batch of 6-atom units; a row labelled z' 0 holds one unit's 6 atoms (z_prime_mismatch)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    natoms = [6 * max(z, 1) for z in zps]
    xyz = torch.cat([torch.cat([_PA + 3.0 * u for u in range(n // 6)]) for n in natoms])
    z = torch.cat([_ZA.repeat(n // 6) for n in natoms])
    counts = torch.tensor(natoms)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    B = len(zps)
    return MolCrystalData.from_arrays(torch.cat([_LENGTHS, _ANGLES])[None].repeat(B, 1), xyz, z, ptr=ptr, sg=2,
                                      setting='standard', z_prime=torch.tensor(zps),
                                      aunit_centroid=torch.tensor([_CENTROIDS] * B),
                                      aunit_orientation=torch.tensor([_ORIENTATIONS] * B),
                                      aunit_handedness=torch.tensor([[1, -1, 1]] * B),
                                      identifier=[f'r{i}' for i in range(B)])


def test_a_per_unit_value_whose_unit_total_is_the_row_count():
    """z' [2, 0] (ingest batches hold failed rows at z' 0): 2 units on 2 rows. add_unit_attr's entry [0, 2, 2] decides
    (UNIT); it was ROW by the length, and append relabelled it per row for good."""
    from mxtaltools.dataset_utils.data_classes import UNIT
    b = _from_arrays_zps([2, 0])
    assert b.z_prime.tolist() == [2, 0] and b.num_graphs == 2
    vals = torch.tensor([[10., 11.], [20., 21.]])
    b.add_unit_attr(vals, 'unit_score')
    assert b._level_plan(b._ptr_cpu(), int(b.ptr[-1]), 2)['unit_score'][0] == UNIT
    for draw in (b.subsample_new_batch([0]), b.select_rows([0])):
        assert torch.equal(draw.unit_score, vals) and draw._slice_dict['unit_score'].tolist() == [0, 2]
    assert b.subsample_new_batch([1]).unit_score.shape == (0, 2)
    assert_same_batch(b.subsample_new_batch([1, 0]), b.select_rows([1, 0]))
    j = b.append_batch(b)
    assert j._slice_dict['unit_score'].tolist() == [0, 2, 2, 4, 4]
    assert [e.unit_score.shape[0] for e in j.to_data_list()] == [2, 0, 2, 0]
    assert torch.equal(j.subsample_new_batch([2]).unit_score, vals)
    b3 = _from_arrays_zps([3, 0, 0])
    b3.add_unit_attr(torch.arange(3.), 'unit_e')
    assert b3.subsample_new_batch([0]).unit_e.tolist() == [0., 1., 2.] == b3.select_rows([0]).unit_e.tolist()


def test_select_rows_and_subsample_agree_when_the_edge_total_is_the_atom_count(tmp_path):
    """A from_arrays batch of two 6-atom rows with 5 and 7 edges (12 on 12 atoms), as is, and stored and reloaded."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    fb = _from_arrays_zps([1, 1])
    fb['edge_attr'] = torch.arange(12.)
    fb._slice_dict['edge_attr'] = torch.tensor([0, 5, 12])
    fb._inc_dict['edge_attr'] = torch.zeros(2, dtype=torch.long)
    meta = fb.save(tmp_path / 'store')
    assert meta['levels']['edge_attr']['level'] == 'edge'
    back = MolCrystalData.load(tmp_path / 'store')
    for b in (fb, back):
        assert b.subsample_new_batch([1]).edge_attr.tolist() == list(range(5, 12)) == \
            b.select_rows([1]).edge_attr.tolist()
        assert_same_batch(b.subsample_new_batch([1, 0]), b.select_rows([1, 0]))


def test_set_mol_attrs_batch_path_keeps_a_coincident_per_edge_entry():
    """GFN's blank-batch route: a per-edge value whose total equals the source's atom count keeps the source's entry
    (it was given the atom pointer, and a draw of row 1 had 8 of its 9 values)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    from mxtaltools.dataset_utils.utils import collate_data_list
    ms = [_mol('A'), _mol('B')]
    ms[1].edge_index = torch.cat([ms[1].edge_index, torch.tensor([[0, 2], [7, 5]])], 1)
    for m, k in zip(ms, (0, 1)):
        m.bond_attr = torch.arange(float(m.edge_index.shape[1])) + 100 * k
    src = collate_data_list(ms)
    assert int(src.ptr[-1]) == src.bond_attr.shape[0] == 14
    blank = collate_data_list([MolCrystalData(max_z_prime=1)], max_z_prime=1)
    blank.__dict__['_num_graphs'] = src.num_graphs
    blank['device'] = src.device
    blank.set_mol_attrs(src.clone())
    del blank['device']
    assert torch.equal(blank._slice_dict['bond_attr'], src._slice_dict['bond_attr'])
    assert torch.equal(blank.subsample_new_batch([1]).bond_attr, ms[1].bond_attr)
    # the atom and row values keep the routes they had (add_node_attr / add_graph_attr entries)
    assert torch.equal(blank._slice_dict['pos'], src.ptr) and torch.equal(blank._slice_dict['num_atoms'], torch.arange(3))


def test_a_device_resident_entry_is_not_read_where_the_length_decides():
    """add_graph_attr / add_node_attr write their entries on the batch's device; the level of an atom- or row-length
    value is then taken from its length without reading the entry (no host sync on the hot path). A 'meta' tensor
    stands in for a device entry: reading it would raise."""
    from mxtaltools.dataset_utils.data_classes import ATOM, ROW
    b = collate(rows(range(3), extras=False))
    n, B = int(b.ptr[-1]), b.num_graphs
    b.atom_tag, b.row_tag = torch.arange(float(n)), torch.arange(float(B))
    b._slice_dict['atom_tag'] = torch.empty(B + 1, dtype=torch.long, device='meta')
    b._slice_dict['row_tag'] = torch.empty(B + 1, dtype=torch.long, device='meta')
    plan = b._level_plan(b._ptr_cpu(), n, B)
    assert plan['atom_tag'][0] == ATOM and plan['row_tag'][0] == ROW


def test_append_keeps_a_registered_constant_whole():
    """sym_mult_lut is registered BATCH: two 231-row batches keep it [231] (it was concatenated to [462] by the row
    count, where subsample kept it whole), and a differing one is refused under validate."""
    big = collate(rows([0], extras=False)).subsample_new_batch([0] * 231)
    big.sym_mult_lut = torch.arange(231)
    assert big.subsample_new_batch(list(range(231))).sym_mult_lut is big.sym_mult_lut
    j = big.append_batch(big)
    assert j.num_graphs == 462 and torch.equal(j.sym_mult_lut, big.sym_mult_lut) and 'sym_mult_lut' not in j._slice_dict
    other = big.clone()
    other.sym_mult_lut = torch.arange(231).flip(0)
    with pytest.raises(ValueError, match="'sym_mult_lut' differs"):
        big.append_batch(other)
    assert torch.equal(big.append_batch(other, validate=False).sym_mult_lut, big.sym_mult_lut)


def test_a_collated_per_row_index_value_round_trips():
    """A per-row value under an 'index' name, collated from rows (crystal search's dataset_index): collation offsets it
    by the atom counts and records those offsets as its increments; every helper re-offsets it consistently, so the
    rows split back to their own ids (before stage 3, subsample gave [136, 182] for ids [101, 102])."""
    rs = rows(range(3), extras=False)
    for i, r in enumerate(rs):
        r.dataset_index = torch.tensor([40 + i])
    b = collate(rs)
    assert torch.equal(b._inc_dict['dataset_index'], b.ptr[:-1])
    ids = lambda x: [int(e.dataset_index) for e in x.batch_to_list()]       # noqa: E731
    assert ids(b.subsample_new_batch([2, 0])) == [42, 40]
    assert ids(b.append_batch(b.subsample_new_batch([1]))) == [40, 41, 42, 41]


def test_an_index_key_whose_increments_are_not_atom_offsets_is_refused():
    """An 'index' value attached with add_graph_attr (entry arange, increments 0), or row-local values with an explicit
    entry and zero increments, is not shifted as atom indices (it was: [9] became [-9]); refused, as select_rows
    refuses it (``_select_rows_index_without_offsets``). A one-dimensional 'index' value without an entry is refused
    rather than read as atom indices."""
    from mxtaltools.dataset_utils.data_classes import BatchLevelError
    b = collate(rows(range(3), extras=False))
    b.add_graph_attr(torch.tensor([7, 8, 9]), 'parent_index')
    for call in (lambda: b.subsample_new_batch([2]), lambda: b.append_batch(b)):
        with pytest.raises(BatchLevelError, match="'parent_index'.*increments"):
            call()
    m = _collate_mols([_mol_edges(3, False, 0), _mol_edges(4, False, 1), _mol_edges(5, False, 2)])
    m.local_index = torch.cat([_chain(n) for n in (3, 4, 5)], 1)
    m._slice_dict['local_index'] = torch.tensor([0, 2, 5, 9])
    m._inc_dict['local_index'] = torch.zeros(3, dtype=torch.long)
    with pytest.raises(BatchLevelError, match="'local_index'.*increments"):
        m.subsample_new_batch([2])
    c = collate(rows(range(3), extras=False))
    c.parent_index = torch.tensor([2, 1, 0])                 # no entry, one dimension
    with pytest.raises(BatchLevelError, match="'parent_index'.*per-row counts are unknown"):
        c.subsample_new_batch([1])


def test_mxtaltools_loaders_fill_the_unit_map_of_mixed_rows():
    """get_dataloaders / update_dataloader_batch_size / quick_combine_dataloaders collate rows with and without the
    unit map as collate_data_list does, in either order (PyG's own collation raised KeyError, or dropped the map); rows
    that all carry it, or none, collate as PyG collates them."""
    from torch_geometric.data import Batch
    from mxtaltools.crystal_building.utils import derived_unit_ind
    from mxtaltools.dataset_utils.utils import (get_dataloaders, update_dataloader_batch_size,
                                                quick_combine_dataloaders)
    rs = rows(range(3), extras=False)
    old = _without_map(rs[1])
    by_id = {r.identifier: r for r in (rs[0], old, rs[2])}
    want_map = lambda b: torch.cat([by_id[i].unit_ind if 'unit_ind' in by_id[i]._store       # noqa: E731
                                    else derived_unit_ind(by_id[i]) for i in b.identifier])
    for order in ([rs[0], old, rs[2]], [old, rs[0], rs[2]]):
        tr, _ = get_dataloaders(list(order), machine='local', batch_size=3, test_fraction=0, shuffle=False)
        b = next(iter(tr))
        assert b.identifier == [r.identifier for r in order] and torch.equal(b.unit_ind, want_map(b))
        assert torch.equal(b.unit_ind, collate(order).unit_ind)
        b2 = next(iter(update_dataloader_batch_size(tr, 3)))
        assert torch.equal(b2.unit_ind, want_map(b2))
        b3 = next(iter(quick_combine_dataloaders([order[0].clone()], tr, 4, 4)))
        assert b3.num_graphs == 4 and torch.equal(b3.unit_ind, want_map(b3))
    assert 'unit_ind' not in old._store                       # the caller's row is not given one
    for same in (rs, [_without_map(r) for r in rs]):
        tr, _ = get_dataloaders(list(same), machine='local', batch_size=3, test_fraction=0, shuffle=False)
        assert_same_batch(next(iter(tr)), Batch.from_data_list(list(same)))


# ------------------------------------------------------------------------------------------------
# refusals: what each one is, and that it holds under python -O
# ------------------------------------------------------------------------------------------------

def _stale_unit_map():
    b = collate(rows(range(2), extras=False))
    b.unit_ind = b.unit_ind[:-1]
    b.subsample_new_batch([0])


def _unit_cell_subsample():
    b = collate(rows(range(2), extras=False))
    b.mol2ucell()
    b.subsample_new_batch([1])


def _unit_cell_append():
    b = collate(rows(range(2), extras=False))
    b.mol2ucell()
    b.append_batch(b)


def _unit_cell_select_rows():
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    xyz, z = torch.cat([_PA, _PB]), torch.cat([_ZA, _ZB])
    b = MolCrystalData.from_arrays(torch.cat([_LENGTHS, _ANGLES])[None].repeat(2, 1), xyz, z,
                                   ptr=torch.tensor([0, 6, 14]), sg=2, setting='standard', z_prime=torch.tensor([1, 1]),
                                   aunit_centroid=torch.tensor([_CENTROIDS[:3]] * 2),
                                   aunit_orientation=torch.tensor([_ORIENTATIONS[:3]] * 2),
                                   aunit_handedness=torch.tensor([1, -1]))
    b.mol2ucell()
    b.select_rows([1])


def _index_not_grouped():
    b = collate(rows(range(3)))
    ei = b.edge_index.flip(1).clone()                       # columns in reverse row order
    del b.edge_index
    b._slice_dict.pop('edge_index'), b._inc_dict.pop('edge_index')
    b.bond_index = ei
    b.subsample_new_batch([1])


def _registered_edge_without_entry():
    b = collate(rows(range(2)))
    b.tree_torsion_is_proper = torch.ones(7, dtype=torch.bool)
    b.subsample_new_batch([1])


def _batch_and_index_in_one_name():
    b = collate(rows(range(2), extras=False))
    b.batch_index = b.edge_index.clone()
    b.subsample_new_batch([0])


def _append_ragged_against_constant():
    a, c = collate(rows(range(1, 3))), collate(rows(range(3, 5)))
    c._slice_dict.pop('unit_score'), c._inc_dict.pop('unit_score')
    c.unit_score = torch.zeros(5, 2)
    a.append_batch(c)


def _append_ragged_other_trailing():
    a, c = collate(rows(range(1, 3))), collate(rows(range(3, 5)))
    c.unit_score = torch.zeros(3, 4)
    a.append_batch(c)


def _add_unit_attr_wrong_length():
    collate(rows(range(2), extras=False)).add_unit_attr(torch.zeros(4), 'unit_energy')


def _add_unit_attr_without_z_prime():
    from mxtaltools.dataset_utils.utils import collate_data_list
    collate_data_list([_mol('A'), _mol('B')]).add_unit_attr(torch.zeros(2), 'unit_energy')


def _batch_to_list_stale_entry():
    b = collate(rows())
    b.unit_score = b.unit_score[:-1]
    b.batch_to_list()


def _list_constructor_key_sets():
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    a, c = _mol('A'), _mol('B')
    c.extra = torch.zeros(8)
    MolCrystalData(molecule=[a, c], sg_ind=2, z_prime=torch.tensor([2]), max_z_prime=2)


def _list_constructor_index_shapes():
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    a, c = _mol('A'), _mol('B')
    c.edge_index = torch.zeros(3, 2, dtype=torch.long)
    MolCrystalData(molecule=[a, c], sg_ind=2, z_prime=torch.tensor([2]), max_z_prime=2)


def _declare_conflicting_level():
    from mxtaltools.dataset_utils.data_classes import declare_level
    declare_level('unit_ind', 'row')


def _select_rows_index_without_offsets():
    b = _from_arrays_with_ragged_values()
    b._inc_dict['edge_index'] = torch.zeros(4, dtype=torch.long)
    b.select_rows([1])


def _subsample_index_without_offsets():
    b = collate(rows(range(3), extras=False))
    b.add_graph_attr(torch.tensor([7, 8, 9]), 'parent_index')
    b.subsample_new_batch([2])


def _append_index_without_offsets():
    b = _from_arrays_with_ragged_values()
    b._inc_dict['edge_index'] = torch.zeros(4, dtype=torch.long)
    b.append_batch(b)


def _index_one_dim_without_entry():
    b = collate(rows(range(3), extras=False))
    b.parent_index = torch.tensor([2, 1, 0])
    b.subsample_new_batch([1])


#: name -> (call, innermost mxtaltools function that must raise, words the message must hold)
REFUSALS = {
    'stale_unit_map': (_stale_unit_map, '_declared_level', ("'unit_ind' is registered at level 'atom'", 'stale')),
    'unit_cell_subsample': (_unit_cell_subsample, 'subsample_new_batch', ('subsample_new_batch', 'unit-cell atom')),
    'unit_cell_append': (_unit_cell_append, 'append_batch', ('append_batch', 'unit-cell atom')),
    'unit_cell_select_rows': (_unit_cell_select_rows, 'select_rows', ('unit-cell atom',)),
    'index_not_grouped': (_index_not_grouped, '_edge_pointer', ("'bond_index'", 'per-row counts are unknown',
                                                                'grouped by row')),
    'registered_edge_without_entry': (_registered_edge_without_entry, '_edge_pointer',
                                      ("'tree_torsion_is_proper'", 'no slice entry describes it')),
    'batch_and_index_in_one_name': (_batch_and_index_in_one_name, '_edge_pointer', ("'batch_index'", 'rename it')),
    'append_ragged_against_constant': (_append_ragged_against_constant, '_join_segments',
                                       ("'unit_score' is ragged", 'batch')),
    'append_ragged_other_trailing': (_append_ragged_other_trailing, '_join_segments',
                                     ("'unit_score' is (5, 2) in self and (3, 4) in other",)),
    'add_unit_attr_wrong_length': (_add_unit_attr_wrong_length, 'add_unit_attr', ('[3, ...] (sum of z_prime)',)),
    'add_unit_attr_without_z_prime': (_add_unit_attr_without_z_prime, 'add_unit_attr', ('no integer z_prime',)),
    'batch_to_list_stale_entry': (_batch_to_list_stale_entry, 'batch_to_list', ("slice entry of 'unit_score'",)),
    'list_constructor_key_sets': (_list_constructor_key_sets, 'set_mol_attrs', ('molecule 1', "['extra']")),
    'list_constructor_index_shapes': (_list_constructor_index_shapes, 'set_mol_attrs',
                                      ("index key 'edge_index'", 'last axis')),
    'declare_conflicting_level': (_declare_conflicting_level, 'declare_level', ("registered at level 'atom'",)),
    'select_rows_index_without_offsets': (_select_rows_index_without_offsets, 'select_rows',
                                          ("increments are not its rows' atom offsets",)),
    'subsample_index_without_offsets': (_subsample_index_without_offsets, '_edge_pointer',
                                        ("'parent_index'", "increments are not its rows' atom offsets")),
    'append_index_without_offsets': (_append_index_without_offsets, '_edge_pointer',
                                     ("'edge_index'", "increments are not its rows' atom offsets")),
    'index_one_dim_without_entry': (_index_one_dim_without_entry, '_edge_pointer',
                                    ("'parent_index'", 'per-row counts are unknown')),
}


def _asserts_active() -> bool:
    try:
        assert False
    except AssertionError:
        return True
    return False


def run_refusals() -> dict:
    """Run every refusal; per case the exception type, the innermost frame's function and file, the line, and the
    message. Run here and, by ``under_O``, in a child interpreter started with ``-O``."""
    torch.set_num_threads(1)
    out = {'optimize': sys.flags.optimize, 'asserts_active': _asserts_active()}
    for name, (call, *_) in REFUSALS.items():
        try:
            call()
            out[name] = {'raised': None}
        except Exception as e:  # recorded, judged by the tests
            f = traceback.extract_tb(e.__traceback__)[-1]
            out[name] = {'raised': type(e).__name__, 'is_value_error': isinstance(e, ValueError), 'function': f.name,
                         'file': Path(f.filename).as_posix(), 'line': (f.line or '').strip(), 'message': str(e)}
    return out


def _judge(name, rec):
    _, where, words = REFUSALS[name]
    assert rec['raised'] is not None and rec['is_value_error'], rec
    assert rec['function'] == where, rec
    assert '/mxtaltools/' in rec['file'], rec
    assert rec['line'].startswith('raise'), rec          # a raise statement, not an accidental crash
    for w in words:
        assert w in rec['message'], (w, rec['message'])


@pytest.fixture(scope='module')
def here():
    return run_refusals()


_CHILD = (
    "import importlib.util, json\n"
    "spec = importlib.util.spec_from_file_location('batch_unit_levels_child', {path!r})\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "print('RESULT ' + json.dumps(mod.run_refusals()))\n"
)


@pytest.fixture(scope='module')
def under_O():
    """``run_refusals()`` in a child interpreter started with ``python -O`` (asserts stripped), CPU only, writing no
    bytecode."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run([sys.executable, '-O', '-c', _CHILD.format(path=str(HERE))], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=900)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT ')]
    assert proc.returncode == 0 and lines, proc.stderr[-4000:]
    return json.loads(lines[-1][len('RESULT '):])


@pytest.mark.parametrize('name', list(REFUSALS))
def test_refusal(here, name):
    _judge(name, here[name])


def test_refusals_hold_under_python_O(under_O):
    assert under_O['optimize'] == 1 and not under_O['asserts_active']
    for name in REFUSALS:
        _judge(name, under_O[name])

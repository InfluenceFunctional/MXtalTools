"""Batch helpers read the levels collation records: subsample_new_batch, rebuild_simple_slice_inc_,
append_batch's bookkeeping and batch_to_list.

CPU only. Where a value's leading dimension leaves its level ambiguous (a one-graph or
one-node batch; as many atoms as rows, which ``read`` reaches with failed rows), the
helpers take the level from ``_slice_dict``; elsewhere, and for a value without an
entry, they keep the length rule. Pinned here: on ordinary collated batches of
``mini_new_csd.pt`` the results equal a transcription of the length-rule code these
helpers had before (``_old_subsample``, ``_old_rebuild``); on the ambiguous batches they
equal building the selected rows directly, where the length rule did not.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data.storage import GlobalStorage

from mxtaltools.dataset_utils.data_class_methods.crystal_io import READ_STATUS
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MXtalBase
from mxtaltools.dataset_utils.utils import collate_data_list

FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
MINI_CSD = Path(__file__).parent / 'datasets' / 'mini_new_csd.pt'
OK = READ_STATUS.index('ok')
#: collation offsets this old-schema per-atom field per row; the helpers never re-offset values
OFFSET_BY_COLLATE = 'aunit_batch'


# --------------------------------------------------------------------------------------
# the length rule as the helpers implemented it before this change (reference for "unchanged")
# --------------------------------------------------------------------------------------

def _old_rebuild(b):
    """``rebuild_simple_slice_inc_`` before: every level from the value's length."""
    ptr = b.ptr.long()
    n, g = int(ptr[-1]), b.num_graphs
    graph_slices, graph_inc = torch.arange(g + 1), torch.zeros(g, dtype=torch.long)
    slice_dict, inc_dict = {}, {}
    for key, val in b._store.items():
        if key in ('ptr', 'batch'):
            continue
        if torch.is_tensor(val):
            if val.ndim == 0 or val.size(0) == 1:
                continue
            if val.size(0) == n:
                slice_dict[key], inc_dict[key] = ptr, graph_inc
            elif val.size(0) == g:
                slice_dict[key], inc_dict[key] = graph_slices, graph_inc
        elif isinstance(val, (list, tuple)) and len(val) == g:
            slice_dict[key], inc_dict[key] = graph_slices, graph_inc
    b.__dict__['_slice_dict'], b.__dict__['_inc_dict'] = slice_dict, inc_dict
    return b


def _old_subsample(b, idx):
    """``subsample_new_batch`` before: gathers by length, then the old rebuild on the result."""
    idx = torch.as_tensor(idx, dtype=torch.long).reshape(-1)
    ptr = b.ptr.long()
    n, g, k = int(ptr[-1]), b.num_graphs, len(idx)
    lengths = ptr[idx + 1] - ptr[idx]
    new_ptr = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    seg = torch.repeat_interleave(torch.arange(k), lengths)
    node_src = ptr[idx][seg] + torch.arange(int(new_ptr[-1])) - new_ptr[seg]
    out = type(b).__new__(type(b))
    out.__dict__['_store'] = GlobalStorage(_parent=out)
    for key, val in b._store.items():
        if key in ('ptr', 'batch'):
            continue
        if torch.is_tensor(val):
            if val.ndim == 0 or val.size(0) == 1:
                out[key] = val
            elif val.size(0) == n:
                out[key] = val[node_src]
            elif val.size(0) == g:
                out[key] = val[idx]
            else:
                out[key] = val
        elif isinstance(val, list) and len(val) == g:
            out[key] = [val[i] for i in idx.tolist()]
        elif isinstance(val, tuple) and len(val) == g:
            out[key] = tuple(val[i] for i in idx.tolist())
        else:
            out[key] = val
    out['batch'], out['ptr'] = seg, new_ptr
    out.__dict__['_num_graphs'] = k
    return _old_rebuild(out)


# --------------------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------------------

def _same(a, b) -> bool:
    """Exactly equal: python type, dtype, shape and bits; containers recursively."""
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return a.dtype == b.dtype and a.shape == b.shape and a.numpy().tobytes() == b.numpy().tobytes()
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def _assert_same_batch(a, b, skip=(), inc=True):
    """Stored values, row count and bookkeeping equal (``skip``: keys whose values may differ).

    ``inc=False`` compares increments only for tensors, and only that they are zero: a
    rebuild writes zero tensors for lists where collation and ``select_rows`` write None,
    and both mean no offset.
    """
    assert type(a) is type(b) and a.num_graphs == b.num_graphs
    assert set(a._store.keys()) == set(b._store.keys())
    for k in a._store.keys():
        if k not in skip:
            assert _same(a._store[k], b._store[k]), k
    assert set(a._slice_dict) == set(b._slice_dict)
    for k in a._slice_dict:
        assert torch.equal(a._slice_dict[k], b._slice_dict[k]), f'_slice_dict[{k}]'
        if k in skip:
            continue
        if inc:
            assert _same(a._inc_dict[k], b._inc_dict[k]), f'_inc_dict[{k}]'
        elif torch.is_tensor(a._store[k]):
            assert not bool(a._inc_dict[k].any()) and not bool(b._inc_dict[k].any()), f'_inc_dict[{k}]'


def _rows():
    if not MINI_CSD.exists():
        pytest.skip(f'{MINI_CSD} absent')
    return torch.load(MINI_CSD, weights_only=False)


def _zp(r):
    return int(r.z_prime.reshape(-1)[0])


def _collate(rows):
    return collate_data_list([r.clone() for r in rows])


# ======================================================================================
# ordinary collated batches: unchanged
# ======================================================================================

def test_ordinary_collated_batches_behave_as_before():
    rows = _rows()
    zp2 = [r for r in rows if _zp(r) == 2]
    assert len(zp2) >= 2
    batches = [_collate(rows[:10]), _collate(zp2), _collate(rows)]
    g = torch.Generator().manual_seed(0)
    for batch in batches:
        B, N = batch.num_graphs, int(batch.ptr[-1])
        assert B > 1 and N > 1 and N != B
        selections = [torch.randint(0, B, (7,), generator=g), torch.arange(B).flip(0), [1, 1, 0],
                      torch.arange(B) % 2 == 0]
        for idx in selections:
            ref_idx = torch.nonzero(idx).reshape(-1) if torch.is_tensor(idx) and idx.dtype == torch.bool else idx
            new, old = batch.subsample_new_batch(idx), _old_subsample(batch, ref_idx)
            _assert_same_batch(new, old)
            # rows split from either the same way
            for a, b in zip(new.batch_to_list(), old.batch_to_list()):
                assert set(a.keys()) == set(b.keys()) and all(_same(a[k], b[k]) for k in a.keys())
        # a rebuild in place, and the bookkeeping of an append, as the length rule wrote them
        rebuilt = batch.clone().rebuild_simple_slice_inc_()
        _assert_same_batch(rebuilt, _old_rebuild(batch.clone()))
        appended = batch.append_batch(batch.subsample_new_batch([0, 1]))
        _assert_same_batch(appended, _old_rebuild(appended.clone()))


def test_unambiguous_batches_never_read_slice_entries(monkeypatch):
    """The entry is read only where the length is ambiguous: no per-key work on ordinary batches."""
    rows = _rows()
    batch = _collate(rows[:10])

    def forbidden(*args, **kwargs):
        raise AssertionError('a slice entry was read on an unambiguous batch')

    monkeypatch.setattr(MXtalBase, '_recorded_level', forbidden)
    sub = batch.subsample_new_batch([0, 3, 3, 9])
    sub.rebuild_simple_slice_inc_()
    batch.append_batch(sub)


# ======================================================================================
# ambiguous lengths: decided by the recorded levels
# ======================================================================================

@pytest.mark.parametrize('zp', [1, 2])
def test_one_crystal_batch(zp):
    """A one-graph batch's per-row tensors have length 1, which the length rule files as shared."""
    rows = _rows()
    r = next(x for x in rows if _zp(x) == zp)
    one = _collate([r])
    twice = one.subsample_new_batch([0, 0])
    ref = _collate([r, r])
    assert twice.num_graphs == 2 and twice.cell_lengths.shape == (2, 3)
    assert twice.aunit_centroid.shape == (2, 3 * zp) and len(twice.symmetry_operators) == 2
    _assert_same_batch(twice, ref, skip={OFFSET_BY_COLLATE}, inc=False)
    assert torch.equal(twice[OFFSET_BY_COLLATE], one[OFFSET_BY_COLLATE].repeat(2))     # gathered, not re-offset

    # the length rule copied every per-row tensor through: a 2-row batch with 1-row cells
    old = _old_subsample(one, [0, 0])
    assert old.cell_lengths.shape == (1, 3) and 'cell_lengths' not in old._slice_dict

    # the same crystals, row by row
    a, b = twice.clone(), ref.clone()
    a.mol2ucell()
    b.mol2ucell()
    assert torch.equal(a.unit_cell_batch, b.unit_cell_batch) and torch.equal(a.unit_cell_pos, b.unit_cell_pos)

    # a rebuild keeps the recorded levels, and the rows split back out
    kept = one.clone().rebuild_simple_slice_inc_()
    assert set(kept._slice_dict) == set(one._slice_dict)
    for elem in twice.batch_to_list():
        alone = one.batch_to_list()[0]
        assert set(elem.keys()) == set(alone.keys()) and all(_same(elem[k], alone[k]) for k in alone.keys())


def test_one_row_drawn_from_a_larger_batch_keeps_its_levels():
    """A one-row draw from an ordinary batch: the result records the source's levels, so it splits and redraws."""
    rows = [r for r in _rows() if _zp(r) == 1][:6]
    batch = _collate(rows)
    one = batch.subsample_new_batch([3])
    assert 'cell_lengths' in one._slice_dict and 'is_well_defined' in one._slice_dict
    assert 'cell_lengths' not in _old_subsample(batch, [3])._slice_dict          # what the length rule recorded
    elem, alone = one.batch_to_list()[0], _collate([rows[3]]).batch_to_list()[0]
    assert set(elem.keys()) == set(alone.keys())
    for k in alone.keys():
        if k != OFFSET_BY_COLLATE:
            assert _same(elem[k], alone[k]), k
    again = one.subsample_new_batch([0, 0, 0])
    assert again.cell_lengths.shape == (3, 3) and again.identifier == [rows[3].identifier] * 3


def _atoms_equal_rows_read_batch():
    """24 failed rows (no atoms) and NUKCOL (25 atoms): 25 atoms, 25 rows."""
    batch = MolCrystalData.read([f'missing/ROW{i:02d}.cif' for i in range(24)] + [FIXTURES / 'NUKCOL.cif'])
    assert batch.num_graphs == int(batch.ptr[-1]) == 25
    return batch


def test_read_batch_with_as_many_atoms_as_rows():
    batch = _atoms_equal_rows_read_batch()
    alone = MolCrystalData.read(FIXTURES / 'NUKCOL.cif')
    for idx in (batch.status == OK, [24], [24, 3, 24], torch.arange(25).flip(0)):
        new, chosen = batch.subsample_new_batch(idx), batch.select_rows(idx)
        # values and slices as select_rows (which reads the levels) gives them; increments are
        # written as zeros for lists here (the rebuild's convention) and None by select_rows
        _assert_same_batch(new, chosen, inc=False)
    good = batch.subsample_new_batch(batch.status == OK)
    for k in alone._store.keys():
        assert _same(good[k], alone[k]), k
    # the length rule gathered per-row fields by atom index
    old = _old_subsample(batch, [24])
    assert old.cell_lengths.shape == (25, 3) and old.identifier == ['NUKCOL']

    # a rebuild in place keeps the levels the reader declared
    rebuilt = batch.clone().rebuild_simple_slice_inc_()
    for k, s in batch._slice_dict.items():
        assert torch.equal(rebuilt._slice_dict[k], s), k
    assert torch.equal(_old_rebuild(batch.clone())._slice_dict['cell_lengths'], batch.ptr)   # it filed them per atom


def test_append_keeps_levels_when_atoms_equal_rows():
    batch = _atoms_equal_rows_read_batch()
    both = batch.append_batch(batch)
    assert both.num_graphs == int(both.ptr[-1]) == 50
    assert torch.equal(both._slice_dict['cell_lengths'], torch.arange(51))
    assert torch.equal(both._slice_dict['pos'], both.ptr) and torch.equal(both._slice_dict['occupancy'], both.ptr)
    assert torch.equal(_old_rebuild(both.clone())._slice_dict['cell_lengths'], both.ptr)    # what it was
    good = both.subsample_new_batch(both.status == OK)
    alone = MolCrystalData.read(FIXTURES / 'NUKCOL.cif')
    assert good.identifier == ['NUKCOL', 'NUKCOL'] and torch.equal(good.pos, torch.cat([alone.pos, alone.pos]))
    assert torch.equal(good.cell_lengths, alone.cell_lengths.repeat(2, 1))


def test_posed_rows_with_as_many_atoms_as_rows_split_into_rows():
    """from_arrays rows (Z' = 1 and 2 units, rows without atoms) with N == B: draw, then split into rows."""
    unit = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    xyz = torch.cat([unit, unit + 3.0, unit - 2.0])                     # rows of 2 (Z' 1), 0, 0, 4 (Z' 2), 0, 0
    z = torch.tensor([6, 6, 6, 6, 7, 7])
    ptr = torch.tensor([0, 2, 2, 2, 6, 6, 6])
    B = 6
    batch = MolCrystalData.from_arrays(
        torch.tensor([[8.0, 9.0, 10.0, 1.5708, 1.8, 1.5708]]).repeat(B, 1), xyz, z, ptr=ptr, sg=2,
        setting='standard', z_prime=torch.tensor([1, 1, 1, 2, 1, 1]),
        aunit_centroid=torch.rand(B, 6), aunit_orientation=torch.rand(B, 6),
        aunit_handedness=torch.ones(B, 2), is_well_defined=[True] * B, identifier=list('abcdef'))
    assert batch.num_graphs == int(batch.ptr[-1]) == B
    drawn = batch.subsample_new_batch([3, 0, 3])
    assert drawn.identifier == ['d', 'a', 'd'] and drawn.cell_lengths.shape == (3, 3)
    assert torch.equal(drawn.pos, torch.cat([xyz[2:6], xyz[0:2], xyz[2:6]]))
    assert torch.equal(drawn.aunit_centroid, batch.aunit_centroid[[3, 0, 3]])
    for elem, i in zip(batch.batch_to_list(), range(B)):
        s, e = int(ptr[i]), int(ptr[i + 1])
        assert elem.identifier == 'abcdef'[i] and torch.equal(elem.pos, xyz[s:e])
        assert torch.equal(elem.cell_lengths, batch.cell_lengths[i:i + 1])
    split = drawn.batch_to_list()
    assert [e.identifier for e in split] == ['d', 'a', 'd'] and torch.equal(split[1].pos, xyz[0:2])


def test_one_atom_batch():
    """A one-atom batch's per-atom tensors have length 1."""
    batch = MolCrystalData.from_arrays(torch.tensor([[8.0, 9.0, 10.0, 1.5708, 1.8, 1.5708]] * 2),
                                       torch.tensor([[0.5, 0.5, 0.5]]), torch.tensor([8]),
                                       ptr=torch.tensor([0, 1, 1]), sg=2, setting='standard')
    drawn = batch.subsample_new_batch([0, 0, 1])
    assert drawn.pos.shape == (2, 3) and drawn.z.tolist() == [8, 8] and drawn.num_atoms.tolist() == [1, 1, 0]
    assert _old_subsample(batch, [0, 0, 1]).pos.shape == (1, 3)                   # copied through before


CELL = torch.tensor([[8.0, 9.0, 10.0, 1.5708, 1.8, 1.5708]])


def _bare(cell, xyz, z, ptr, identifier):
    return MolCrystalData.from_arrays(cell, torch.as_tensor(xyz, dtype=torch.float32), torch.as_tensor(z),
                                      ptr=torch.tensor(ptr), sg=2, setting='standard', identifier=identifier)


def test_append_to_a_batch_with_one_atom_in_every_row():
    """One atom per row (ptr == arange) cannot tell per-atom from per-row: the other batch's levels decide."""
    one = _bare(CELL, [[0.1, 0.2, 0.3]], [8], [0, 1], ['A'])
    nb = _bare(torch.cat([CELL * 2, CELL * 3]), [[1.0, 1, 1], [2.0, 2, 2]], [6, 7], [0, 0, 2], ['B0', 'B1'])  # N == B
    for joined, order in ((one.append_batch(nb), ['A', 'B0', 'B1']), (nb.append_batch(one), ['B0', 'B1', 'A'])):
        assert joined.num_graphs == int(joined.ptr[-1]) == 3
        assert torch.equal(joined._slice_dict['cell_lengths'], torch.arange(4))
        assert torch.equal(joined._slice_dict['num_atoms'], torch.arange(4))
        assert torch.equal(joined._slice_dict['pos'], joined.ptr)
        i = order.index('B1')
        for drawn in (joined.subsample_new_batch([i]), joined.select_rows([i])):
            assert drawn.identifier == ['B1'] and torch.equal(drawn.cell_lengths, nb.cell_lengths[1:])
            assert drawn.num_atoms.tolist() == [2] and torch.equal(drawn.pos, nb.pos)
        split = joined.to_data_list()
        assert [e.identifier for e in split] == order
        assert torch.equal(split[i].cell_lengths, nb.cell_lengths[1:]) and torch.equal(split[i].pos, nb.pos)
    # the length rule filed the result's per-row tensors per atom, as did taking the one-atom side's level
    assert torch.equal(_old_rebuild(one.append_batch(nb))._slice_dict['cell_lengths'], torch.tensor([0, 1, 1, 3]))

    # two such batches: the result has one atom in every row too, and either gather is the same
    both = one.append_batch(one)
    drawn = both.subsample_new_batch([1])
    assert drawn.identifier == ['A'] and torch.equal(drawn.pos, one.pos) and torch.equal(drawn.cell_lengths, one.cell_lengths)


def test_append_takes_the_level_only_one_batch_records():
    """A one-row batch files a length-1 value set without an entry as shared; the other batch's entry decides."""
    two = _bare(CELL, [[0.0, 0, 0], [1.0, 1, 1]], [6, 6], [0, 2], ['P'])
    empty = _bare(CELL * 2, torch.zeros(0, 3), torch.zeros(0, dtype=torch.long), [0, 0], ['Q'])
    two.score = torch.tensor([5.0])                                   # no entry: shared by the length rule
    empty.add_graph_attr(torch.tensor([7.0]), 'score')                # recorded per row
    for joined, order in ((two.append_batch(empty), [5.0, 7.0]), (empty.append_batch(two), [7.0, 5.0])):
        assert joined.num_graphs == int(joined.ptr[-1]) == 2              # as many atoms as rows
        assert torch.equal(joined._slice_dict['score'], torch.arange(3))
        drawn = joined.subsample_new_batch([1, 0])
        assert drawn.score.tolist() == order[::-1]
    # without the recorded side, the result's length rule filed it per atom
    assert torch.equal(_old_rebuild(two.append_batch(empty))._slice_dict['score'], torch.tensor([0, 2, 2]))


def test_values_without_a_usable_entry_keep_the_length_rule():
    rows = [r for r in _rows() if _zp(r) == 1][:4]
    batch = _collate(rows)
    batch.per_row_unrecorded = torch.arange(4.0)          # set after collation: no entry, length decides
    batch.lut = torch.arange(231)                         # shared table: no entry, other length
    batch.cell_volume = torch.ones(4, 2)                  # same length as its entry: entry still fits
    sub = batch.subsample_new_batch([2, 0])
    assert sub.per_row_unrecorded.tolist() == [2.0, 0.0] and 'per_row_unrecorded' in sub._slice_dict
    assert sub.lut is batch.lut and 'lut' not in sub._slice_dict

    one = _collate(rows[:1])
    one.extra = torch.tensor([7.0])                       # one-graph batch, no entry: shared, as before
    one.radius = torch.tensor([1.0, 2.0, 3.0])            # entry left from collation no longer fits: length rule
    drawn = one.subsample_new_batch([0, 0])
    assert drawn.extra is one.extra and 'extra' not in drawn._slice_dict
    assert drawn.radius is one.radius and 'radius' not in drawn._slice_dict
    assert drawn.cell_lengths.shape == (2, 3)

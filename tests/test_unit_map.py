"""Stage 2 of the Z'>1 units plan: the atom -> unit map (gfn_diffusion energy_sampling/docs/design/crystal_units_audit.md,
section 5 stage 2; items #1, #2, #7, #8, #9, #19, #25 and section 7).

What is pinned here:

- the map: ``unit_ind``, per atom of a crystal ROW, the row-local unit 0 .. z_prime - 1 (long), read by every reader
  through ``crystal_building.utils.unit_layout``; derived, never stored: ``num_units``, ``unit_ptr``, ``unit_batch``,
  ``unit_atom_counts``, also on mixed-Z' batches;
- absence is explicit: a row without ``unit_ind`` has ``z_prime`` equal contiguous blocks, today's layout, and with
  a map equal to it every reader's output (split, cluster, eLJ, canonicalisation, exp6 typing, envwise RDF labels) is
  bit-identical to the same batch without the map;
- the writers: the list constructor (molecule k -> unit k), a single molecule (one unit), a map-less Z'>1 crystal row
  re-wrapped (its equal blocks), an explicit ``unit_ind``, the old featurizer's ``instantiate_crystal``,
  ``zp_doubling``, ``from_arrays`` with a pose (its layout, -1 on z_prime_mismatch rows, or the map given; since
  stage 3), ``find_molecules`` (the units found, -1 on rows that fail; since stage 3), ``sanitize`` (fills it for the
  hydrogens it places; tests/test_crystal_chem.py and the chain runs), ``parameterize_pose`` (a carried map checked
  and kept; on a batch written before stage 3, from find_molecules' units, ``mol_ind``, -1 where the units are not
  known), the split (each unit-graph one unit), the clusters (none), and ``canonicalize_zp_aunits`` (moved with the
  atom blocks and relabelled to the new slot order, long);
- the map travels with its atoms through collate (a map-less row beside rows with one gets its derived map, in any
  order; stage 3), ``batch_to_list``, ``subsample_new_batch``, ``append_batch`` (a map-less side gets its derived
  map), ``select_rows``, ``check_rebuild``'s work batch and the ingest store;
- known gap, as a strict xfail: GFN's blank-batch route relabelled Z'=2 (an owner decision);
- refusals, each a ValueError raised by a ``raise`` statement in mxtaltools and still raised under ``python -O`` (a
  child interpreter runs every one with ``-O``): unequal units (the ethanol + methane case, here 6 + 8 atoms),
  non-contiguous units, values outside 0 .. z_prime - 1, a unit without atoms (a one-molecule row relabelled Z'=2,
  at the build and at construction), a Z'=2 row relabelled Z'=1 at width 1, a row without a map whose atoms do not
  divide by z_prime, a stale map of the wrong length, the list constructor's merges it cannot do, and (stage-2
  review) a map that is not a one-dimensional integer tensor with one entry per atom, a [2]-entry z_prime on one row,
  a map that cannot be derived for append_batch, and from_arrays' map refusals;
- the list-constructor fixes of audit #7: [1]-shaped row scalars summed and stored 0-dim, an int z_prime stored [1].

The crystals are synthetic (small molecules without point symmetry, a triclinic cell) except the ingestion and store
tests, which use the CIF fixtures of tests/cif/fixtures (IDOVUT is Z'=2).
"""
import copy
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
FIXTURES = HERE.parent / 'cif' / 'fixtures'
FIND = dict(bonds='file', bond_scale=(1.15, 1.35), iso_cap=5000, special_tol=0.1)
POSE = dict(handedness='measured', degenerate_tol=1e-4, handedness_tol=1e-3)

# small molecules without point symmetry (>= 3 non-collinear heavy atoms each): A (6 atoms), A2 (A's atoms, perturbed:
# the same molecule, another conformer), B (8 atoms; A + B = 14 atoms, divisible by 2, the ethanol + methane case)
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
_CUTOFF = 6.0
#: three slots' centroids and orientations; slot order (0.62 > 0.12 by norm) is NOT canonical for the first two
_CENTROIDS = (0.62, 0.55, 0.71, 0.12, 0.20, 0.31, 0.40, 0.70, 0.15)
_ORIENTATIONS = (-1.2, 0.5, 0.4, 0.3, 0.2, 1.0, 0.7, -0.6, 0.9)


def _mol(name, **kw):
    from mxtaltools.dataset_utils.data_classes import MolData
    z, pos = _MOLS[name]
    return MolData(z=z.clone(), pos=pos.clone(), x=torch.arange(len(z), dtype=torch.float32)[:, None] + 100 *
                   len(name), do_mol_analysis=True, **kw)


def row(names=('A',), identifier='ROW', max_z_prime=None, z_prime=None, centroids=_CENTROIDS,
        orientations=_ORIENTATIONS, **kw):
    """A crystal through the constructor: one MolData for one name, the list constructor for several."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    zp = len(names)
    mols = [_mol(n) for n in names]
    return MolCrystalData(molecule=mols if zp > 1 else mols[0], sg_ind=2,
                          z_prime=torch.tensor([zp]) if z_prime is None else z_prime,
                          max_z_prime=max_z_prime or zp, cell_lengths=_LENGTHS.clone(), cell_angles=_ANGLES.clone(),
                          aunit_centroid=torch.tensor(centroids[:3 * zp]),
                          aunit_orientation=torch.tensor(orientations[:3 * zp]),
                          aunit_handedness=torch.tensor([1, -1, 1][:zp]), identifier=identifier,
                          do_box_analysis=True, **kw)


def _collate(rows, max_z_prime=None):
    from mxtaltools.dataset_utils.utils import collate_data_list
    return collate_data_list(rows, max_z_prime=max_z_prime)


def _equal_blocks(counts, zps):
    return torch.cat([torch.arange(z).repeat_interleave(n // z) for n, z in zip(counts, zps)])


def _without_map(b):
    c = b.clone()
    c._store.pop('unit_ind', None)
    return c


def _same(a, b):
    """Equal bit for bit: type, dtype, shape and values (NaN equal to NaN)."""
    import numpy as np
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
    return a == b


# ------------------------------------------------------------------------------------------------
# the writers
# ------------------------------------------------------------------------------------------------

def test_list_constructor_writes_molecule_k_as_unit_k():
    r = row(('A', 'B', 'A2'), max_z_prime=3)
    assert r.unit_ind.dtype == torch.long
    assert r.unit_ind.tolist() == [0] * 6 + [1] * 8 + [2] * 6
    assert int(r.num_atoms) == 20 and r.z_prime.tolist() == [3]


def test_single_molecule_is_one_unit_and_an_explicit_map_overrides():
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    r = row(('A',))
    assert r.unit_ind.tolist() == [0] * 6
    # a merged two-unit molecule needs its map (as the old featurizer passes it)
    merged = _collate([row(('A', 'A2'))]).batch_to_list()[0]
    from mxtaltools.dataset_utils.data_classes import MolData
    m = MolData(z=merged.z.clone(), pos=merged.pos.clone(), x=merged.x.clone(), do_mol_analysis=True)
    kw = dict(sg_ind=2, z_prime=torch.tensor([2]), max_z_prime=2, cell_lengths=_LENGTHS.clone(),
              cell_angles=_ANGLES.clone(), aunit_centroid=torch.tensor(_CENTROIDS[:6]),
              aunit_orientation=torch.tensor(_ORIENTATIONS[:6]), aunit_handedness=torch.tensor([1, 1]))
    with pytest.raises(ValueError, match='unit'):
        MolCrystalData(molecule=m, **kw)
    ok = MolCrystalData(molecule=m, unit_ind=torch.arange(2).repeat_interleave(6), **kw)
    assert ok.unit_ind.tolist() == [0] * 6 + [1] * 6
    # a single crystal row passed as the molecule keeps its own map
    again = MolCrystalData(molecule=merged, **kw)
    assert again.unit_ind.tolist() == [0] * 6 + [1] * 6
    # a map-less Z'=2 crystal row (an older file) passed as the molecule keeps the layout every reader gives it, its
    # equal blocks (stage-2 review: it was given one unit, and its valid Z'=2 refused)
    old = _without_map(merged)
    assert MolCrystalData(molecule=old, **kw).unit_ind.tolist() == [0] * 6 + [1] * 6
    bare = MolCrystalData(molecule=old)             # no space group: the row's z_prime is copied
    assert bare.z_prime.tolist() == [2] and bare.unit_ind.tolist() == [0] * 6 + [1] * 6
    # a molecule's z_prime is not read (GFN writes z_prime onto molecule batches): it is one unit, so refused at 2
    m2 = MolData(z=merged.z.clone(), pos=merged.pos.clone(), x=merged.x.clone(), do_mol_analysis=True)
    m2.z_prime = torch.tensor([2])
    with pytest.raises(ValueError, match='unit\\(s\\) \\[1\\]'):
        MolCrystalData(molecule=m2, **kw)
    # a map-less row whose units cannot be read gets none: the readers refuse it as they did
    odd = _odd_row_batch().batch_to_list()[0]
    assert int(odd.z_prime) == 2 and len(odd.z) == 11
    assert 'unit_ind' not in MolCrystalData(molecule=odd)._store


def test_old_featurizer_writes_the_map():
    from mxtaltools.dataset_utils.construction.featurization_utils import instantiate_crystal
    from mxtaltools.dataset_utils.data_classes import MolData
    mols = [_mol('A'), _mol('A')]
    rebuild = SimpleNamespace(
        z=torch.cat([_ZA, _ZA]), pos=torch.cat([_PA, _PA2]), x=torch.zeros(12, 1), smiles=['CCO', 'CCO'],
        fingerprint=torch.zeros(2, 4), sg_ind=torch.tensor([2, 2]), cell_lengths=_LENGTHS[None].repeat(2, 1),
        cell_angles=_ANGLES[None].repeat(2, 1), identifier=['FEAT', 'FEAT'], nonstandard_symmetry=[False, False],
        symmetry_operators=[None, None])
    from mxtaltools.constants.space_group_info import SYM_OPS
    import numpy as np
    rebuild.symmetry_operators = [np.stack(SYM_OPS[2])] * 2
    c = instantiate_crystal(torch.tensor(_CENTROIDS[:6]), torch.tensor([1, 1]), torch.tensor(_ORIENTATIONS[:6]), 2,
                            mols, rebuild, 2)
    assert c.unit_ind.tolist() == [0] * 6 + [1] * 6 and c.mol_ind.tolist() == c.unit_ind.tolist()
    one = instantiate_crystal(torch.tensor(_CENTROIDS[:3]), torch.tensor([1]), torch.tensor(_ORIENTATIONS[:3]), 1,
                              mols[:1], SimpleNamespace(**dict(vars(rebuild), z=_ZA, pos=_PA, x=torch.zeros(6, 1),
                                                               smiles=['CCO'], fingerprint=torch.zeros(1, 4))), 1)
    assert one.unit_ind.tolist() == [0] * 6


def test_zp_doubling_writes_the_map():
    from mxtaltools.crystal_building.zp_doubling import double_zp1_crystal
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    zp1 = MolCrystalData(molecule=_mol('B'), sg_ind=14, cell_lengths=torch.tensor([9.0, 7.5, 11.0]),
                         cell_angles=torch.tensor([torch.pi / 2, 1.85, torch.pi / 2]),
                         aunit_centroid=torch.tensor([0.21, 0.13, 0.27]), aunit_orientation=torch.tensor([0.3, 0.2, 1.0]),
                         aunit_handedness=torch.tensor([1]), identifier='DBL', do_box_analysis=True)
    doubled, _ = double_zp1_crystal(zp1, reduce=False, origin=(0.0, 0.0, 0.0), check=False)
    assert doubled.unit_ind.tolist() == [0] * 8 + [1] * 8 and doubled.z_prime.tolist() == [2]


def _from_arrays(atom_units=(6, 6), zp=2, atom_map=None, posed=True, **extra):
    """A from_arrays batch of one row: molecule A (6 atoms) and A2 (6) or B (8) as its units."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    parts = {6: (_ZA, _PA), 8: (_ZB, _PB)}
    xyz = torch.cat([parts[n][1] + 3.0 * k for k, n in enumerate(atom_units)])
    z = torch.cat([parts[n][0] for n in atom_units])
    kw = dict(ptr=torch.tensor([0, len(z)]), sg=2, setting='standard')
    if posed:
        kw.update(z_prime=torch.tensor([zp]), aunit_centroid=torch.tensor([_CENTROIDS[:3 * zp]]),
                  aunit_orientation=torch.tensor([_ORIENTATIONS[:3 * zp]]),
                  aunit_handedness=torch.tensor([[1, -1, 1][:zp]]))
    if atom_map is not None:
        extra['atom_attrs'] = {'unit_ind': atom_map}
    return MolCrystalData.from_arrays(torch.cat([_LENGTHS, _ANGLES])[None], xyz, z, **kw, **extra)


def test_from_arrays_writes_its_layout_as_the_map_or_the_given_one():
    """Since stage 3 (2026-10-23) a posed from_arrays batch writes its pose's layout as the map (Z' equal contiguous
    blocks, equal to mol_ind; NO_UNIT on a z_prime_mismatch row), so its rows carry the map like constructor-built
    ones; until then it wrote none (the readers derived the same layout). Since the stage-2 review it takes a map in
    atom_attrs with a pose: stored per atom, checked against Z' per row (a map that is not Z' equal contiguous blocks,
    e.g. the 6 + 8 atom units of audit #1, is a z_prime_mismatch row, which the readers refuse)."""
    from mxtaltools.crystal_building.utils import NO_UNIT, derived_unit_ind
    b = _from_arrays()
    assert _same(b.unit_ind, torch.tensor([0] * 6 + [1] * 6)) and torch.equal(b.unit_ind, b.mol_ind)
    assert torch.equal(b._slice_dict['unit_ind'], b.ptr.cpu()) and not bool(b._inc_dict['unit_ind'].any())
    # the layout every reader gave the batch without it (the absence path), written out
    assert _same(b.unit_ind, derived_unit_ind(_without_map(b)))
    assert torch.equal(b.unit_batch, torch.tensor([0] * 6 + [1] * 6))
    # the same equal-block map given: the same batch, bit for bit
    m = _from_arrays(atom_map=torch.tensor([0] * 6 + [1] * 6))
    assert m.status.tolist() == [0] and set(m._store.keys()) == set(b._store.keys())
    for k in b._store.keys():
        assert _same(m[k], b[k]), k
    s, ns = m.split_to_zp1_batch(), _without_map(b).split_to_zp1_batch()
    for k in ('pos', 'z', 'batch', 'ptr', 'num_atoms', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness'):
        assert _same(s[k], ns[k]), k
    # a numpy map is taken too
    import numpy as np
    assert _same(_from_arrays(atom_map=np.array([0] * 6 + [1] * 6)).unit_ind, m.unit_ind)
    # a row whose atoms do not split into its Z' (8 atoms at Z'=3) is a z_prime_mismatch row: NO_UNIT, which every
    # reader refuses, as it refused the row without a map
    from mxtaltools.crystal_building.utils import unit_layout
    from mxtaltools.dataset_utils.data_class_methods.crystal_io import READ_STATUS
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    mismatch = READ_STATUS.index('z_prime_mismatch')
    two_rows = MolCrystalData.from_arrays(
        torch.cat([_LENGTHS, _ANGLES])[None].repeat(2, 1), torch.cat([_PA, _PA + 3.0, _PB + 6.0]),
        torch.cat([_ZA, _ZA, _ZB]), ptr=torch.tensor([0, 12, 20]), sg=2, setting='standard',
        z_prime=torch.tensor([2, 3]), aunit_centroid=torch.tensor([_CENTROIDS] * 2),
        aunit_orientation=torch.tensor([_ORIENTATIONS] * 2), aunit_handedness=torch.tensor([[1, -1, 1]] * 2))
    assert two_rows.status.tolist() == [0, mismatch]
    assert two_rows.unit_ind.tolist() == [0] * 6 + [1] * 6 + [NO_UNIT] * 8
    for rows in (two_rows, _without_map(two_rows)):
        with pytest.raises(ValueError, match='row 1'):
            unit_layout(rows, 'a reader')
    # 6 + 8 atoms at Z'=2: without a map from_arrays writes 7 + 7 blocks (nothing can tell); with its map the row is a
    # z_prime_mismatch row, its map kept, and every reader refuses it
    blind = _from_arrays((6, 8))
    assert blind.status.tolist() == [0] and blind.split_to_zp1_batch().num_atoms.tolist() == [7, 7]
    assert blind.unit_ind.tolist() == [0] * 7 + [1] * 7
    told = _from_arrays((6, 8), atom_map=torch.tensor([0] * 6 + [1] * 8))
    from mxtaltools.dataset_utils.data_class_methods.crystal_io import READ_STATUS
    mismatch = READ_STATUS.index('z_prime_mismatch')
    assert told.status.tolist() == [mismatch] and 'units of [6, 8] atoms' in told.message[0]
    assert told.unit_ind.tolist() == [0] * 6 + [1] * 8 and told.mol_ind.tolist() == [0] * 14
    from mxtaltools.crystal_building.utils import unit_layout
    with pytest.raises(ValueError, match='row 0.*\\[6, 8\\]'):
        unit_layout(told, 'a reader')
    with pytest.raises(ValueError, match='row 0'):     # its slots were zeroed (a z_prime_mismatch row): one slot
        told.split_to_zp1_batch()
    with pytest.raises(ValueError):
        told.analyze(['elj'], cutoff=_CUTOFF)
    # interleaved units of equal size are not the blocks posed either
    inter = _from_arrays(atom_map=torch.arange(12) % 2)
    assert inter.status.tolist() == [mismatch] and 'contiguous' in inter.message[0]
    # refused: per crystal, on a bare batch, a float / bool / 2-D / wrong-length map
    with pytest.raises(ValueError, match='pass it in atom_attrs'):
        _from_arrays(unit_ind=torch.zeros(1, dtype=torch.long))      # per crystal (attrs)
    with pytest.raises(ValueError, match='taken only with a pose'):
        _from_arrays(posed=False, atom_map=torch.zeros(12, dtype=torch.long))
    for bad, words in ((torch.zeros(12), 'dtype'), (torch.zeros(12, dtype=torch.bool), 'dtype'),
                       (torch.zeros(12, 1, dtype=torch.long), 'shape'), (torch.zeros(11, dtype=torch.long), '11')):
        with pytest.raises(ValueError, match=words):
            _from_arrays(atom_map=bad)


def test_ingestion_writes_the_map_at_find_molecules_and_the_store_keeps_it(tmp_path):
    """Since stage 3 (2026-10-23) find_molecules writes the map (the units it found, equal to mol_ind), and
    parameterize_pose and check_rebuild keep it (in stage 2 parameterize_pose wrote it: sanitize, between the two,
    could not fill it for the hydrogens it places; it fills it now, tests/test_crystal_chem.py)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    b0 = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in ('IDOVUT', 'KUNTEV', 'NUKCOL')], dtype=torch.float64)
    b1 = b0.find_molecules(**FIND)
    assert _same(b1.unit_ind, _equal_blocks(b1.num_atoms.tolist(), b1.z_prime.tolist()))
    assert torch.equal(b1.unit_ind, b1.mol_ind) and torch.equal(b1._slice_dict['unit_ind'], b1.ptr.cpu())
    b2 = b1.parameterize_pose(**POSE)
    assert _same(b2.unit_ind, b1.unit_ind)      # kept
    b3 = b2.check_rebuild()
    assert [int(s) for s in b3.status] == [0, 0, 0] and b3.z_prime.tolist() == [2, 1, 1]
    want = _equal_blocks(b3.num_atoms.tolist(), b3.z_prime.tolist())
    assert _same(b3.unit_ind, want)
    assert torch.equal(b3.unit_ind, b3.mol_ind)
    assert torch.equal(b3._slice_dict['unit_ind'], b3.ptr.cpu())
    # a batch written before stage 3 (no map after find_molecules) gets the same map at parameterize_pose
    old = _without_map(b1)
    old._slice_dict.pop('unit_ind'), old._inc_dict.pop('unit_ind')
    assert _same(old.parameterize_pose(**POSE).unit_ind, b2.unit_ind)
    with pytest.raises(ValueError, match='runs once'):      # the map is find_molecules' output: a second pass refused
        b1.find_molecules(**FIND)
    with pytest.raises(ValueError, match='runs once'):
        b2.parameterize_pose(**POSE)
    # select_rows splits it per atom
    sel = b3.select_rows([2, 0])
    assert _same(sel.unit_ind, torch.cat([want[int(b3.ptr[2]):int(b3.ptr[3])], want[:int(b3.ptr[1])]]))
    # the store keeps it bit for bit, with its slice entry
    b3.save(tmp_path / 'store', steps_run=b3.steps_run)
    back = MolCrystalData.load(tmp_path / 'store')
    assert _same(back.unit_ind, b3.unit_ind) and torch.equal(back._slice_dict['unit_ind'], b3._slice_dict['unit_ind'])
    # the Z'=2 row splits the same from the stored map and from the absence layout
    a, c = back.select_rows([0]).split_to_zp1_batch(), _without_map(back.select_rows([0])).split_to_zp1_batch()
    for k in ('pos', 'z', 'batch', 'ptr', 'num_atoms', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness'):
        assert _same(a[k], c[k]), k
    assert a.num_atoms.tolist() == [int(b3.num_atoms[0]) // 2] * 2


def _found(names=('IDOVUT', 'KUNTEV', 'EFUNUP')):
    """Fixture CIFs through find_molecules: IDOVUT Z'=2 and KUNTEV Z'=1 ok, EFUNUP refused (multi_component, Z' 0)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    return MolCrystalData.read([FIXTURES / f'{n}.cif' for n in names], dtype=torch.float64).find_molecules(**FIND)


def _carry_map(b, values):
    b = b.clone()
    b['unit_ind'] = values
    b._slice_dict['unit_ind'] = b.ptr.detach().cpu().clone()
    b._inc_dict['unit_ind'] = torch.zeros(b.num_graphs, dtype=torch.long)
    return b


def _before_stage3(b):
    """A find_molecules output as written before stage 3: no unit map (nor its bookkeeping)."""
    b = _without_map(b)
    b._slice_dict.pop('unit_ind', None), b._inc_dict.pop('unit_ind', None)
    return b


@pytest.mark.parametrize('written', ['stage3', 'before_stage3'])
def test_the_map_is_find_molecules_units(tmp_path, written):
    """The map's source is the units find_molecules found (mol_ind): since stage 3 find_molecules writes it and
    parameterize_pose keeps it; on a batch written before, parameterize_pose writes it from mol_ind (stage-2 review).
    Either way a row whose units are not known as Z' equal blocks (EFUNUP: Z' 0, four components in mol_ind) gets
    NO_UNIT (-1), not one unit of all its atoms, so its map contradicts neither its z_prime nor its mol_ind, and every
    reader refuses it as without a map. The store keeps it."""
    from mxtaltools.crystal_building.utils import unit_layout, NO_UNIT
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    b1 = _found()
    assert b1.z_prime.tolist() == [2, 1, 0] and int(b1.status[2]) != 0
    if written == 'before_stage3':
        b1 = _before_stage3(b1)
    p = b1.parameterize_pose(**POSE)
    assert [int(s) for s in p.status[:2]] == [0, 0]
    ptr = p.ptr.tolist()
    want = _equal_blocks([ptr[1] - ptr[0], ptr[2] - ptr[1]], [2, 1])
    for b in ((p, b1) if written == 'stage3' else (p,)):
        assert _same(b.unit_ind[:ptr[2]], want) and torch.equal(b.unit_ind[:ptr[2]], b.mol_ind[:ptr[2]])
        assert b.unit_ind[ptr[2]:].tolist() == [NO_UNIT] * (ptr[3] - ptr[2])
    for b in (p, _without_map(p)):     # refused with and without the map
        with pytest.raises(ValueError, match='row 2'):
            unit_layout(b, 'a reader', rows=[2])
        unit_layout(b, 'a reader', rows=[0, 1])
    p.save(tmp_path / 'store', steps_run=p.steps_run)
    back = MolCrystalData.load(tmp_path / 'store')
    assert _same(back.unit_ind, p.unit_ind) and torch.equal(back._slice_dict['unit_ind'], p._slice_dict['unit_ind'])


def test_parameterize_pose_refuses_a_row_whose_units_are_not_the_blocks_it_poses():
    """An ok row whose units (mol_ind on a batch written before stage 3; the carried map since, which
    test_parameterize_pose_checks_and_keeps_a_map_the_batch_carries covers) are not Z' equal contiguous blocks in unit
    order gets pose_failed naming them (the step poses those blocks; before the stage-2 review it posed them without
    reading mol_ind), and NO_UNIT; other rows unchanged."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_steps import STATUS
    from mxtaltools.crystal_building.utils import NO_UNIT
    b1 = _before_stage3(_found())
    p = b1.parameterize_pose(**POSE)
    s, e = int(b1.ptr[0]), int(b1.ptr[1])
    for name, relabel in (('reversed', lambda u: u.flip(0)), ('interleaved', lambda u: torch.arange(len(u)) % 2)):
        t = b1.clone()
        mi = t.mol_ind.clone()
        mi[s:e] = relabel(mi[s:e])
        t['mol_ind'] = mi
        q = t.parameterize_pose(**POSE)
        assert STATUS[int(q.status[0])] == 'pose_failed', name
        assert 'mol_ind' in q.message[0] and 'equal contiguous blocks' in q.message[0], name
        assert q.unit_ind[s:e].tolist() == [NO_UNIT] * (e - s), name
        assert int(q.status[1]) == 0 and torch.equal(q.unit_ind[e:], p.unit_ind[e:]), name
        assert torch.equal(q.pos[e:], p.pos[e:]), name
        assert torch.equal(q.aunit_centroid[1], p.aunit_centroid[1, :q.aunit_centroid.shape[1]]), name


def test_parameterize_pose_checks_and_keeps_a_map_the_batch_carries():
    """unit_ind is no longer one of parameterize_pose's own keys (a carried map, e.g. a later find_molecules', was
    refused as a second pass): it is checked on the ok rows and kept as it is; the pose is the same bit for bit."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_steps import STATUS
    b1 = _found()
    p = b1.parameterize_pose(**POSE)
    carried = b1.mol_ind.clone()
    q = _carry_map(b1, carried).parameterize_pose(**POSE)
    assert torch.equal(q.unit_ind, carried)                 # kept: EFUNUP's row keeps the carried values
    for k in p._store.keys():
        if k != 'unit_ind':
            assert _same(q[k], p[k]), k
    with pytest.raises(ValueError, match='runs once'):      # the DONE_POSE bit still refuses a second pass
        q.parameterize_pose(**POSE)
    s, e = int(b1.ptr[0]), int(b1.ptr[1])
    bad = carried.clone()
    bad[s:e] = bad[s:e].flip(0)
    r = _carry_map(b1, bad).parameterize_pose(**POSE)
    assert STATUS[int(r.status[0])] == 'pose_failed' and 'unit_ind' in r.message[0]
    assert int(r.status[1]) == 0 and torch.equal(r.unit_ind, bad)
    with pytest.raises(ValueError, match='cannot be read'):  # a map that is not one integer entry per atom
        _carry_map(b1, carried.float()).parameterize_pose(**POSE)


def test_check_rebuild_builds_from_the_rows_map(monkeypatch):
    """check_rebuild hands the rows' map to the batch it builds (it built from the absence layout; equal at this
    stage, so the result is the same, but a later stage's unequal map would have been dropped)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    p = _found(('IDOVUT', 'KUNTEV')).parameterize_pose(**POSE)
    seen = []
    real = MolCrystalData.from_arrays.__func__

    def spy(cls, *args, **kwargs):
        out = real(cls, *args, **kwargs)
        seen.append(out)
        return out

    monkeypatch.setattr(MolCrystalData, 'from_arrays', classmethod(spy))
    r = p.check_rebuild()
    assert [int(s) for s in r.status] == [0, 0] and seen
    assert all('unit_ind' in w._store for w in seen)
    assert torch.equal(torch.cat([w.unit_ind for w in seen]), p.unit_ind)


def test_check_validity_reads_units_from_the_map_without_mol_ind():
    """check_validity's unit criteria run on a batch carrying the map and no mol_ind (they were refused: 'it needs
    units'), with the same outcome as the same batch with mol_ind and no map."""
    with_map = _from_arrays(atom_map=torch.tensor([0] * 6 + [1] * 6))
    with_map._store.pop('mol_ind')
    with_map._slice_dict.pop('mol_ind'), with_map._inc_dict.pop('mol_ind')
    no_map = _from_arrays()
    no_map._store.pop('unit_ind')        # from_arrays writes the map since stage 3: a batch without one, as before
    no_map._slice_dict.pop('unit_ind'), no_map._inc_dict.pop('unit_ind')
    for b in (with_map, no_map):
        assert ('unit_ind' in b._store) != ('mol_ind' in b._store)
    for crit in (dict(max_heavy_atoms=5), dict(max_heavy_atoms=50), dict(heavy_nn_range=(0.5, 3.0))):
        a, c = with_map.check_validity(**crit), no_map.check_validity(**crit)
        assert a.status.tolist() == c.status.tolist() and a.message == c.message, crit
        assert torch.equal(a.validity_checked, c.validity_checked), crit
    assert with_map.check_validity(max_heavy_atoms=4).status.tolist() != [0]     # 5 heavy atoms per unit


# ------------------------------------------------------------------------------------------------
# derived helpers and the absence path
# ------------------------------------------------------------------------------------------------

def _mixed(with_map=True):
    """Rows of Z' 1, 2 and 3 at max_z_prime 3 (A; A + A2; A2 + A + A2), with or without stored maps."""
    b = _collate([row(('A',), 'ONE', max_z_prime=3), row(('A', 'A2'), 'TWO', max_z_prime=3),
                  row(('A2', 'A', 'A2'), 'THREE', max_z_prime=3)], max_z_prime=3)
    return b if with_map else _without_map(b)


@pytest.mark.parametrize('with_map', [True, False])
def test_unit_ptr_unit_batch_on_a_mixed_zp_batch(with_map):
    from mxtaltools.crystal_building.utils import unit_layout
    b = _mixed(with_map)
    assert ('unit_ind' in b._store) == with_map
    assert b.num_units.tolist() == [1, 2, 3]
    assert b.unit_ptr.tolist() == [0, 1, 3, 6]
    assert b.unit_atom_counts.tolist() == [6] * 6
    assert b.unit_batch.tolist() == [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6
    lay = unit_layout(b)
    assert lay.stored == with_map and lay.unit_ind.tolist() == [0] * 6 + [0] * 6 + [1] * 6 + [0] * 6 + [1] * 6 + [2] * 6
    assert lay.atoms_per_unit.tolist() == [6, 6, 6]
    assert all(t.dtype == torch.long for t in (b.unit_ptr, b.unit_batch, b.unit_atom_counts, b.num_units))


def test_absence_path_is_the_equal_block_layout_bit_for_bit():
    """A row without a map gets today's layout; with a map equal to it, every reader's output is bit-identical."""
    from mxtaltools.crystal_building.utils import unit_layout, equal_block_unit_ind
    from mxtaltools.analysis.exp6_ff import aunit_molecule_index
    b, nb = _mixed(True), _mixed(False)
    assert torch.equal(unit_layout(nb).unit_ind, b.unit_ind)
    assert torch.equal(equal_block_unit_ind(torch.tensor([6, 12, 18]), torch.tensor([1, 2, 3])), b.unit_ind)
    # exp6 typing's molecule index: today's formula (z_prime contiguous equal blocks, numbered batch-wide)
    ptr = b.ptr
    local = torch.arange(len(b.batch)) - ptr[:-1][b.batch]
    per = ((ptr[1:] - ptr[:-1]) // b.z_prime)[b.batch]
    old = torch.cat([torch.zeros(1, dtype=torch.long), b.z_prime.cumsum(0)[:-1]])[b.batch] + local // per
    assert torch.equal(aunit_molecule_index(b), old) and torch.equal(aunit_molecule_index(nb), old)
    # the split
    s, ns = b.split_to_zp1_batch(), nb.split_to_zp1_batch()
    for k in ('pos', 'z', 'x', 'batch', 'ptr', 'num_atoms', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness',
              'radius', 'mass', 'mol_volume', 'z_prime'):
        assert _same(s[k], ns[k]), k
    assert s.unit_ind.tolist() == [0] * 36 and ns.unit_ind.tolist() == [0] * 36   # each unit-graph is one unit
    # the cluster and the energy
    for std in (True, False):
        r1, c1 = b.clone().analyze(['elj'], cutoff=_CUTOFF, std_orientation=std, return_cluster=True)
        r2, c2 = nb.clone().analyze(['elj'], cutoff=_CUTOFF, std_orientation=std, return_cluster=True)
        assert torch.equal(r1['elj'], r2['elj'])
        for k in ('pos', 'z', 'mol_ind', 'aux_ind', 'batch', 'ptr'):
            assert torch.equal(c1[k], c2[k]), k
        assert 'unit_ind' not in c1._store and 'unit_ind' not in c2._store      # clusters carry no row map
    # envwise RDF labels
    rdf1 = b.clone().analyze(['rdf'], cutoff=_CUTOFF, rdf_mode='envwise')['rdf']
    rdf2 = nb.clone().analyze(['rdf'], cutoff=_CUTOFF, rdf_mode='envwise')['rdf']
    assert torch.equal(rdf1[0], rdf2[0]) and rdf1[2] == rdf2[2]


def test_a_width1_batch_keeps_its_cluster_and_energy_with_a_map():
    b = _collate([row(('A',), 'P'), row(('B',), 'Q')])
    assert b.unit_ind.tolist() == [0] * 14
    r1 = b.clone().analyze(['elj'], cutoff=_CUTOFF)['elj']
    r2 = _without_map(b).analyze(['elj'], cutoff=_CUTOFF)['elj']
    assert torch.equal(r1, r2)


# ------------------------------------------------------------------------------------------------
# the map travels with its atoms
# ------------------------------------------------------------------------------------------------

def test_collate_list_subsample_append_carry_the_map():
    b = _mixed(True)
    full = b.unit_ind.clone()
    blocks = [full[int(b.ptr[i]):int(b.ptr[i + 1])] for i in range(3)]
    assert torch.equal(b._slice_dict['unit_ind'], b.ptr) and not bool(b._inc_dict['unit_ind'].any())
    rows = b.batch_to_list()
    assert [r.unit_ind.tolist() for r in rows] == [x.tolist() for x in blocks]
    again = _collate(rows, max_z_prime=3)
    assert torch.equal(again.unit_ind, full)
    sub = b.subsample_new_batch([2, 0])
    assert torch.equal(sub.unit_ind, torch.cat([blocks[2], blocks[0]]))
    assert sub.unit_batch.tolist() == [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6
    one = b.subsample_new_batch([1])
    assert torch.equal(one.unit_ind, blocks[1]) and one.unit_ptr.tolist() == [0, 2]
    joined = b.append_batch(sub)
    assert torch.equal(joined.unit_ind, torch.cat([full, blocks[2], blocks[0]]))
    assert joined.unit_ptr.tolist() == [0, 1, 3, 6, 9, 10]
    for r, blk in zip(joined.batch_to_list(), blocks + [blocks[2], blocks[0]]):
        assert torch.equal(r.unit_ind, blk)


def _odd_row_batch():
    """A map-less batch of one Z'=2 row of 11 atoms: its units cannot be read (11 is not divisible by 2)."""
    b = _collate([row(('A', 'A2'), 'ODD')])
    b._store.pop('unit_ind')
    b.pos, b.z, b.x = b.pos[:-1], b.z[:-1], b.x[:-1]
    b.batch, b.ptr, b.num_atoms = b.batch[:-1], torch.tensor([0, 11]), torch.tensor([11])
    b._slice_dict['pos'] = b._slice_dict['z'] = b._slice_dict['x'] = torch.tensor([0, 11])
    return b


@pytest.mark.parametrize('validate', [True, False])
def test_append_batch_gives_a_map_less_side_its_derived_map(validate):
    """Stage-2 review (blocker): rows with a map (constructor-built since stage 2) appended to rows without one (older
    files, GFN's blank batch, from_arrays) raised KeyError (validate) or dropped the map for the whole result. The
    map-less side now gets the layout every reader gives it (crystal_building.utils.derived_unit_ind), so the result
    reads exactly as the two sides read apart, in either order."""
    from mxtaltools.crystal_building.utils import derived_unit_ind, NO_UNIT
    new = _collate([row(('A',), 'NEW1', max_z_prime=2), row(('A', 'A2'), 'NEW2')], max_z_prime=2)
    old = _without_map(_collate([row(('A2', 'A'), 'OLD2'), row(('B',), 'OLD1', max_z_prime=2)], max_z_prime=2))
    assert _same(derived_unit_ind(old), _equal_blocks([12, 8], [2, 1]))
    for first, second in ((new, old), (old, new)):
        j = first.append_batch(second, validate=validate)
        want = torch.cat([first.unit_ind if 'unit_ind' in first._store else derived_unit_ind(first),
                          second.unit_ind if 'unit_ind' in second._store else derived_unit_ind(second)])
        assert _same(j.unit_ind, want)
        assert torch.equal(j._slice_dict['unit_ind'].cpu(), j.ptr.cpu())
        sizes = [n // z for n, z in zip(j.num_atoms.tolist(), j.z_prime.tolist()) for _ in range(z)]
        assert j.unit_batch.tolist() == torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes)).tolist()
        assert j.num_units.tolist() == j.z_prime.tolist()
        # it reads as the map-less join reads (the absence path): split, bit for bit
        bare = _without_map(first).append_batch(_without_map(second), validate=validate)
        s, ns = j.split_to_zp1_batch(), bare.split_to_zp1_batch()
        for k in ('pos', 'z', 'x', 'batch', 'ptr', 'num_atoms', 'aunit_centroid', 'aunit_orientation'):
            assert _same(s[k], ns[k]), k
    # a map-less row whose units cannot be read gets NO_UNIT, and is refused with and without it
    odd = _odd_row_batch()
    j = new.append_batch(odd, validate=validate)
    assert j.unit_ind[-11:].tolist() == [NO_UNIT] * 11
    for b in (j, _without_map(new).append_batch(odd, validate=validate)):
        with pytest.raises(ValueError, match='row 2 \\(ODD\\)'):
            b.split_to_zp1_batch()
    # no map can be laid over atoms the row counts do not cover
    stale = _without_map(_collate([row(('A',), 'STALE', max_z_prime=2)], max_z_prime=2))
    stale.num_atoms = torch.tensor([5])
    with pytest.raises(ValueError, match='sum to 5, but 6 atoms'):
        new.append_batch(stale, validate=validate)


def _collate_mixed_presence():
    """Old (map-less) and new rows collated in both orders: the result must carry the map, the derived layout on the
    map-less rows."""
    from mxtaltools.crystal_building.utils import derived_unit_ind
    new, old = row(('A', 'A2'), 'NEW'), _without_map(row(('A2', 'A'), 'OLD'))
    for pair in ([new, old], [old, new]):
        b = _collate([r.clone() for r in pair])
        want = torch.cat([r.unit_ind if 'unit_ind' in r._store else derived_unit_ind(r) for r in pair])
        assert 'unit_ind' in b._store and _same(b.unit_ind, want)


def test_collate_gives_a_map_less_row_its_derived_map():
    """Flipped at stage 3 (2026-10-23; the stage-2 review blocker): collate_data_list raised KeyError when the first
    row carried unit_ind and a later one did not, and dropped the map when the first did not (PyG takes the keys from
    the first row); it now gives each map-less row crystal_building.utils.derived_unit_ind on a shallow copy, so
    tests/test_zp_doubling.py::test_output_schema_matches_search_outputs passes again. The inputs are not modified."""
    _collate_mixed_presence()
    from mxtaltools.crystal_building.utils import NO_UNIT
    new, old = row(('A', 'A2'), 'NEW'), _without_map(row(('A2', 'A'), 'OLD'))
    b = _collate([new, old])
    assert 'unit_ind' not in old._store                       # the caller's row is not given the map
    assert torch.equal(b._slice_dict['unit_ind'], b.ptr) and not bool(b._inc_dict['unit_ind'].any())
    # a map-less row whose units cannot be read gets NO_UNIT, refused as without it
    odd = _odd_row_batch().batch_to_list()[0]
    mixed = _collate([_collate([new]).batch_to_list()[0], odd])     # rows of one shape convention (batch_to_list)
    assert mixed.unit_ind[-11:].tolist() == [NO_UNIT] * 11
    with pytest.raises(ValueError, match='row 1 \\(ODD\\)'):
        mixed.split_to_zp1_batch()
    # rows that all carry it, or none, collate as before
    assert 'unit_ind' not in _collate([_without_map(row(('A',), 'P')), _without_map(row(('B',), 'Q'))])._store


def test_set_mol_attrs_batch_path_copies_the_map_or_leaves_it_absent():
    """GFN's init_blank_crystal_batch route: a crystal batch filled from a molecule batch by set_mol_attrs."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    for src, has in ((_mixed(True), True), (_mixed(False), False)):
        blank = _collate([MolCrystalData(max_z_prime=3)], max_z_prime=3)
        blank.__dict__['_num_graphs'] = src.num_graphs
        blank['device'] = src.device
        blank.set_mol_attrs(src.clone())
        del blank['device']
        assert ('unit_ind' in blank._store) == has
        if has:
            assert torch.equal(blank.unit_ind, src.unit_ind)


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                   reason="stage-2 review, OWNER DECISION: a crystal batch filled from a molecule batch by set_mol_attrs "
                          "(GFN's init_blank_crystal_batch route) carries no unit map, so the audit #2 refusal does not "
                          "reach it: relabelled Z'=2 (as GFN condition_samples does), its one-molecule row is cut into "
                          "two half-molecule units and scored. Flips if set_mol_attrs' batch path writes one unit per "
                          "molecule graph (option a); not if GFN writes the map itself (option b)")
def test_blank_batch_route_relabelled_zp2_is_refused():
    crystal = _collate([row(('A',), 'BLANK', max_z_prime=2)], max_z_prime=2)
    crystal._store.pop('unit_ind')          # the crystal part of a blank batch, before its molecules are copied in
    crystal.set_mol_attrs(_collate([_mol('A')]))
    crystal.aunit_centroid = torch.tensor([_CENTROIDS[:6]])
    crystal.aunit_orientation = torch.tensor([_ORIENTATIONS[:6]])
    crystal.aunit_handedness = torch.tensor([[1, 1]])
    crystal.z_prime = torch.tensor([2])
    try:
        elj = crystal.analyze(['elj'], cutoff=_CUTOFF)['elj']
    except ValueError as e:
        assert 'row 0' in str(e)
        return
    pytest.fail(f"a one-molecule row relabelled Z'=2 on the blank-batch route was scored: eLJ {elj.tolist()}")


# ------------------------------------------------------------------------------------------------
# canonicalisation moves and relabels the map
# ------------------------------------------------------------------------------------------------

def test_canonicalize_moves_and_relabels_the_map():
    b = _collate([row(('A',), 'STAY', max_z_prime=3), row(('A2', 'A'), 'MOVE2', max_z_prime=3,
                                                            centroids=(0.62, 0.55, 0.71, 0.12, 0.20, 0.31)),
                  row(('A', 'A2', 'A'), 'MOVE3', max_z_prime=3, centroids=(0.62, 0.55, 0.71, 0.40, 0.70, 0.15,
                                                                           0.12, 0.20, 0.31))], max_z_prime=3)
    b.x = torch.arange(len(b.z), dtype=torch.float32)[:, None]      # a unique tag per atom
    tag_before = b.x[:, 0].clone()
    old_unit = b.unit_ind.clone()
    from mxtaltools.dataset_utils.data_class_methods.crystal_ops import _canonical_unit_order
    order = _canonical_unit_order(b)
    c = b.clone()
    c.canonicalize_zp_aunits()
    # the moved atoms keep their per-atom identity (x tags) and their new label is their unit's new slot
    perm = torch.tensor([int(torch.nonzero(tag_before == t)[0]) for t in c.x[:, 0]])
    assert torch.equal(c.x[:, 0], tag_before[perm])
    inv = torch.argsort(order, dim=1)
    rows = b.batch
    assert torch.equal(c.unit_ind, inv[rows[perm], old_unit[perm]])
    # and the result is again z_prime contiguous blocks, labelled in order
    assert torch.equal(c.unit_ind, _equal_blocks([6, 12, 18], [1, 2, 3]))
    # row 1 (A2, A) moved: block 0 now holds A's atoms (old atoms 12..17), block 1 A2's (old 6..11); row 2 reversed
    assert c.x[6:18, 0].tolist() == list(range(12, 18)) + list(range(6, 12))
    assert c.x[18:36, 0].tolist() == list(range(30, 36)) + list(range(24, 30)) + list(range(18, 24))
    assert bool((order[1:, :2] != torch.arange(2)).any())       # the two Z'>1 rows did move
    # a stored map of another integer dtype comes out long, the map's dtype, relabelled the same (stage-2 review: it
    # kept int32); a float map is refused (unit_layout), not cast
    i32 = b.clone()
    i32['unit_ind'] = i32.unit_ind.int()
    i32.canonicalize_zp_aunits()
    assert _same(i32.unit_ind, c.unit_ind)
    flt = b.clone()
    flt['unit_ind'] = flt.unit_ind.float()
    with pytest.raises(ValueError, match='dtype'):
        flt.canonicalize_zp_aunits()
    # without a map, the same move, and no map is added
    nc = _without_map(b)
    nc.canonicalize_zp_aunits()
    assert 'unit_ind' not in nc._store
    for k in ('pos', 'z', 'x', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness'):
        assert torch.equal(c[k], nc[k]), k


def test_canonicalize_refuses_a_moving_crystal_whose_units_cannot_be_read():
    b = _collate([row(('B', 'A'), 'UNEQUAL', max_z_prime=2)], max_z_prime=2)   # 8 + 6, slots out of order
    with pytest.raises(ValueError, match='row 0 \\(UNEQUAL\\).*units of \\[8, 6\\] atoms'):
        b.clone().canonicalize_zp_aunits()
    # the same row with its slots already in canonical order does not move and is not checked
    ok = _collate([row(('B', 'A'), 'UNEQUAL', max_z_prime=2, centroids=(0.12, 0.20, 0.31, 0.62, 0.55, 0.71))])
    ok.canonicalize_zp_aunits()
    assert ok.unit_ind.tolist() == [0] * 8 + [1] * 6


# ------------------------------------------------------------------------------------------------
# the list constructor's merges (audit #7)
# ------------------------------------------------------------------------------------------------

def test_list_constructor_sums_collated_row_scalars_and_refuses_the_rest():
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    m = _mol('A')
    rows = _collate([m.clone(), _mol('A2')]).to_data_list()
    assert rows[0].num_atoms.shape == (1,)
    kw = dict(sg_ind=2, z_prime=torch.tensor([2]), max_z_prime=2, cell_lengths=_LENGTHS.clone(),
              cell_angles=_ANGLES.clone(), aunit_centroid=torch.tensor(_CENTROIDS[:6]),
              aunit_orientation=torch.tensor(_ORIENTATIONS[:6]), aunit_handedness=torch.tensor([1, 1]),
              do_box_analysis=True)
    c = MolCrystalData(molecule=rows, **kw)
    # stored 0-dim, MolData's shape, whatever the inputs' shapes (since the stage-2 review: it took molecule 0's)
    assert c.num_atoms.shape == () and int(c.num_atoms) == 12
    for k in ('radius', 'mass', 'mol_volume'):
        assert getattr(c, k).shape == ()
        assert torch.equal(getattr(c, k), rows[0][k].reshape(()) + rows[1][k].reshape(())), k
    fresh = MolCrystalData(molecule=[m.clone(), _mol('A2')], **kw)        # 0-dim inputs: summed as before
    assert fresh.num_atoms.shape == () and int(fresh.num_atoms) == 12
    assert torch.equal(fresh.mass, c.mass) and c.packing_coeff.numel() == 1
    # either order of fresh and collated molecules, the same shapes
    for mols in ([m.clone(), rows[1].clone()], [rows[0].clone(), _mol('A2')]):
        mixed = MolCrystalData(molecule=mols, **kw)
        assert all(mixed[k].shape == () for k in ('num_atoms', 'radius', 'mass', 'mol_volume'))
    # so a row built from collated molecules collates beside a default row, in either order (it did not, audit #7)
    c.identifier = 'FROM_COLLATED'
    one = row(('A',), 'DEFAULT', max_z_prime=2)
    for pair in ([one.clone(), c.clone()], [c.clone(), one.clone()]):
        b = _collate(pair, max_z_prime=2)
        assert sorted(b.num_atoms.tolist()) == [6, 12] and b.mass.shape == (2,)
    # a value that is not one per molecule cannot be summed
    bad = [rows[0].clone(), rows[1].clone()]
    bad[1].num_atoms = torch.tensor([6, 6])
    with pytest.raises(ValueError, match="'num_atoms' must be one value per listed molecule"):
        MolCrystalData(molecule=bad, **kw)
    # a listed molecule that is itself several units
    two = _collate([row(('A', 'A2'))]).batch_to_list()[0]
    with pytest.raises(ValueError, match='each listed molecule is one unit'):
        MolCrystalData(molecule=[two, _mol('A')], **dict(kw, z_prime=torch.tensor([3]), max_z_prime=3,
                                                       aunit_centroid=torch.tensor(_CENTROIDS),
                                                       aunit_orientation=torch.tensor(_ORIENTATIONS),
                                                       aunit_handedness=torch.tensor([1, 1, 1])))


def test_int_z_prime_is_stored_like_the_default():
    r2 = row(('A', 'A2'), 'INT', z_prime=2)
    assert r2.z_prime.shape == (1,) and r2.z_prime.dtype == torch.long and int(r2.z_prime) == 2
    r1 = row(('A',), 'DEFAULT', max_z_prime=2)
    b = _collate([r1, r2], max_z_prime=2)
    assert b.z_prime.tolist() == [1, 2]
    t0 = row(('A', 'A2'), 'TENSOR0', z_prime=torch.tensor(2))     # a tensor keeps its shape
    assert t0.z_prime.shape == ()


# ------------------------------------------------------------------------------------------------
# refusals: what each one is, and that it holds under python -O
# ------------------------------------------------------------------------------------------------

def _split_unequal():
    _collate([row(('A',), 'FIRST', max_z_prime=2), row(('A', 'B'), 'ETH_METH_LIKE')], max_z_prime=2).split_to_zp1_batch()


def _analyze_unequal():
    _collate([row(('A', 'B'), 'ETH_METH_LIKE')]).analyze(['elj'], cutoff=_CUTOFF)


def _split_not_contiguous():
    b = _collate([row(('A', 'A2'), 'INTERLEAVED')])
    b.unit_ind = torch.arange(12) % 2                 # equal sizes, units interleaved atom by atom
    b.split_to_zp1_batch()


def _split_value_outside():
    b = _collate([row(('A', 'A2'), 'OUTSIDE')])
    b.unit_ind = torch.tensor([0] * 6 + [2] * 6)
    b.split_to_zp1_batch()


def _relabel_one_molecule_zp2():
    b = _collate([row(('A',), 'RELABEL', max_z_prime=2)], max_z_prime=2)
    b.aunit_centroid = torch.tensor([_CENTROIDS[:6]])
    b.aunit_orientation = torch.tensor([_ORIENTATIONS[:6]])
    b.aunit_handedness = torch.tensor([[1, 1]])
    b.z_prime = torch.tensor([2])
    b.analyze(['elj'], cutoff=_CUTOFF)


def _construct_one_molecule_zp2():
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    MolCrystalData(molecule=_mol('A'), sg_ind=2, z_prime=torch.tensor([2]), max_z_prime=2,
                   cell_lengths=_LENGTHS.clone(), cell_angles=_ANGLES.clone(),
                   aunit_centroid=torch.tensor(_CENTROIDS[:6]), aunit_orientation=torch.tensor(_ORIENTATIONS[:6]),
                   aunit_handedness=torch.tensor([1, 1]), identifier='GIVEN2')


def _construct_two_molecules_zp1():
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    MolCrystalData(molecule=[_mol('A'), _mol('A2')], sg_ind=2, max_z_prime=2,
                   cell_lengths=_LENGTHS.clone(), cell_angles=_ANGLES.clone(),
                   aunit_centroid=torch.tensor(_CENTROIDS[:6]), aunit_orientation=torch.tensor(_ORIENTATIONS[:6]),
                   aunit_handedness=torch.tensor([1, 1]), identifier='LIST1')


def _relabel_zp2_row_zp1_at_width1(route):
    b = _collate([row(('A',), 'OK'), row(('A', 'A2'), 'WAS2', max_z_prime=2)], max_z_prime=2)
    b.z_prime = torch.tensor([1, 1])
    b.aunit_centroid, b.aunit_orientation = b.aunit_centroid[:, :3], b.aunit_orientation[:, :3]
    b.aunit_handedness = b.aunit_handedness[:, :1]
    b.max_z_prime = 1
    if route == 'mol2ucell':
        b.mol2ucell()
    else:
        b.mol2cluster(cutoff=_CUTOFF)


def _split_no_map_not_divisible():
    b = _collate([row(('A', 'A2'), 'ODD')])
    b._store.pop('unit_ind')
    b.pos, b.z, b.x = b.pos[:-1], b.z[:-1], b.x[:-1]
    b.batch, b.ptr, b.num_atoms = b.batch[:-1], torch.tensor([0, 11]), torch.tensor([11])
    b.split_to_zp1_batch()


def _split_stale_map():
    b = _collate([row(('A', 'A2'), 'STALE')])
    b.pos, b.z, b.x = b.pos[:-1], b.z[:-1], b.x[:-1]            # unit_ind not trimmed with the atoms
    b.batch, b.ptr, b.num_atoms = b.batch[:-1], torch.tensor([0, 11]), torch.tensor([11])
    b.split_to_zp1_batch()


def _layout_unequal():
    from mxtaltools.crystal_building.utils import unit_layout
    unit_layout(_collate([row(('A', 'B'), 'ETH_METH_LIKE')]), 'a reader')


def _exp6_typing_unequal():
    from mxtaltools.analysis.exp6_ff import aunit_molecule_index
    aunit_molecule_index(_collate([row(('A', 'B'), 'ETH_METH_LIKE')]))


def _unit_batch_property_unequal():
    _collate([row(('A', 'B'), 'ETH_METH_LIKE')]).unit_batch


def _list_constructor_cannot_sum():
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    a, b = _mol('A'), _mol('A2')
    b.mass = torch.tensor([1.0, 2.0])
    MolCrystalData(molecule=[a, b], sg_ind=2, z_prime=torch.tensor([2]), max_z_prime=2)


def _construct_with_map(unit_ind, z_prime=1):
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    MolCrystalData(molecule=_mol('A'), sg_ind=2, z_prime=torch.tensor(z_prime).reshape(-1), max_z_prime=1,
                   cell_lengths=_LENGTHS.clone(), cell_angles=_ANGLES.clone(),
                   aunit_centroid=torch.tensor(_CENTROIDS[:3]), aunit_orientation=torch.tensor(_ORIENTATIONS[:3]),
                   aunit_handedness=torch.tensor([1]), identifier='MAPPED', unit_ind=unit_ind)


def _split_float_map():
    b = _collate([row(('A', 'A2'), 'FLOAT')])
    b.unit_ind = b.unit_ind.float() + 0.25         # .long() would have read it as the blocks
    b.split_to_zp1_batch()


def _append_stale_counts():
    new = _collate([row(('A',), 'NEW')])
    stale = _without_map(_collate([row(('A',), 'STALE')]))
    stale.num_atoms = torch.tensor([5])
    new.append_batch(stale)


#: name -> (call, innermost mxtaltools function that must raise, words the message must hold)
REFUSALS = {
    # stage-2 review (2026-10-23): the map's dtype and shape, z_prime's entries, a map that cannot be derived, and
    # from_arrays' map
    'construct_float_map': (lambda: _construct_with_map(torch.tensor([0., .7, .2, .9, 0., .5])),
                            'refuse_unit_map_disagreeing_with_z_prime',
                            ("MolCrystalData('MAPPED')", 'not a unit map', 'dtype is torch.float32')),
    'construct_bool_map': (lambda: _construct_with_map(torch.zeros(6, dtype=torch.bool)),
                           'refuse_unit_map_disagreeing_with_z_prime', ('not a unit map', 'dtype is torch.bool')),
    'construct_column_map': (lambda: _construct_with_map(torch.zeros(6, 1, dtype=torch.long)),
                             'refuse_unit_map_disagreeing_with_z_prime', ('not a unit map', 'shape is (6, 1)')),
    'construct_short_map': (lambda: _construct_with_map(torch.zeros(5, dtype=torch.long)),
                            'refuse_unit_map_disagreeing_with_z_prime', ('5 entries for 6 atoms',)),
    'construct_two_z_primes': (lambda: _construct_with_map(torch.zeros(6, dtype=torch.long), z_prime=[2, 2]),
                               'refuse_unit_map_disagreeing_with_z_prime', ('z_prime holds 2 values',)),
    'split_float_map': (_split_float_map, 'split_to_zp1_batch', ('cannot be read as a unit map', 'dtype')),
    'append_stale_counts': (_append_stale_counts, 'derived_unit_ind', ('append_batch', 'sum to 5, but 6 atoms')),
    'from_arrays_map_without_pose': (lambda: _from_arrays(posed=False, atom_map=torch.zeros(12, dtype=torch.long)),
                                     'from_arrays', ('taken only with a pose',)),
    'from_arrays_float_map': (lambda: _from_arrays(atom_map=torch.zeros(12)), 'from_arrays',
                              ('not a unit map', 'dtype')),
    'split_unequal_units': (_split_unequal, 'split_to_zp1_batch',
                            ('row 1 (ETH_METH_LIKE)', '14 atoms in units of [6, 8] atoms', 'z_prime 2')),
    'analyze_unequal_units': (_analyze_unequal, 'split_to_zp1_batch', ('row 0 (ETH_METH_LIKE)', '[6, 8]')),
    'split_not_contiguous': (_split_not_contiguous, 'split_to_zp1_batch', ('row 0 (INTERLEAVED)', 'contiguous')),
    'split_value_outside': (_split_value_outside, 'split_to_zp1_batch', ('row 0 (OUTSIDE)', 'values [2] outside 0..1')),
    'relabel_one_molecule_zp2': (_relabel_one_molecule_zp2, 'split_to_zp1_batch',
                                 ('row 0 (RELABEL)', 'unit(s) [1]', 'hold no atom')),
    'construct_one_molecule_zp2': (_construct_one_molecule_zp2, 'refuse_unit_map_disagreeing_with_z_prime',
                                   ("MolCrystalData('GIVEN2')", 'z_prime 2', 'unit(s) [1]')),
    'construct_two_molecules_zp1': (_construct_two_molecules_zp1, 'refuse_unit_map_disagreeing_with_z_prime',
                                    ("MolCrystalData('LIST1')", 'z_prime 1', 'unit 1')),
    'relabel_zp2_row_zp1_mol2ucell': (lambda: _relabel_zp2_row_zp1_at_width1('mol2ucell'),
                                      '_refuse_z_prime_outside_slots', ('mol2ucell', 'row 1 (WAS2)', 'unit map')),
    'relabel_zp2_row_zp1_mol2cluster': (lambda: _relabel_zp2_row_zp1_at_width1('mol2cluster'),
                                        '_refuse_z_prime_outside_slots', ('mol2cluster', 'row 1 (WAS2)', 'unit map')),
    'split_no_map_not_divisible': (_split_no_map_not_divisible, 'split_to_zp1_batch',
                                   ('row 0 (ODD)', '11 atoms, not divisible by z_prime 2')),
    'split_stale_map': (_split_stale_map, 'split_to_zp1_batch', ('unit_ind holds 12 entries', '11 atoms')),
    'layout_unequal': (_layout_unequal, 'unit_layout', ('a reader', 'row 0 (ETH_METH_LIKE)', '[6, 8]')),
    'exp6_typing_unequal': (_exp6_typing_unequal, 'unit_layout', ('exp6 atom typing', '[6, 8]')),
    'unit_batch_property_unequal': (_unit_batch_property_unequal, 'unit_layout', ('unit_batch', '[6, 8]')),
    'list_constructor_cannot_sum': (_list_constructor_cannot_sum, 'set_mol_attrs',
                                    ("'mass' must be one value per listed molecule",)),
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
    "spec = importlib.util.spec_from_file_location('unit_map_child', {path!r})\n"
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

"""canonicalize_zp_aunits moves each asymmetric unit's atom block with its pose slot.

Up to git HEAD 1a0c138b it reordered only the pose slots (``aunit_centroid``, ``aunit_orientation``,
``aunit_handedness``). A Z' > 1 row whose units hold different conformers (the pre-copy CSD files, and the
ingestion chain of ``MolCrystalSteps``) then rebuilt a different crystal: atom block k was posed by the unit that
had moved into slot k. Checked here, CPU only:

- Z' = 1 batches: every stored attribute bit-identical to the slots-only function (the HEAD body is
  ``_slots_only`` below), on all Z' = 1 rows of tests/datasets/mini_new_csd.pt;
- Z' > 1 rows with distinct per-unit conformers (mini_new_csd.pt; D:/crystal_datasets/mini_new_new_csd.pt when
  present; synthetic Z' = 2 and 3 rows): ``mol2ucell`` builds the same crystal before and after (element-aware,
  under periodic images, within ``TOL``), the slots-only function builds a different one on some row, and
  ``latent_params`` -> ``latent_to_cell_params`` rebuilds the crystal;
- rows whose units are copies (post-copy files): the same crystal as the slots-only function;
- the other per-atom, per-atom-list, bond and per-slot attributes move with the atoms; the single-crystal path; the
  order agrees with ``canonicalize_aunit_order``; the refusals.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mxtaltools.crystal_building.utils import canonicalize_aunit_order
from mxtaltools.dataset_utils.data_class_methods.crystal_ops import _canonical_unit_order
from mxtaltools.dataset_utils.utils import collate_data_list

MINI = Path(__file__).resolve().parent / 'datasets' / 'mini_new_csd.pt'
MINI_NEW_NEW = Path('D:/crystal_datasets/mini_new_new_csd.pt')  # pre-copy: every Z' > 1 unit keeps its conformer
MINI_PROT = Path('D:/crystal_datasets/mini_new_prot_csd.pt')  # post-copy: units are copies of unit 0
TOL = 1e-4  # Angstrom
TEETH = 1e-2  # Angstrom: the slots-only function must miss by more than this on some reordered row


def _load(path):
    if not path.exists():
        pytest.skip(f'{path} not present')
    return torch.load(path, weights_only=False, map_location='cpu')


@pytest.fixture(scope='module')
def mini_rows():
    return _load(MINI)


def _slots_only(c):
    """The body of MolCrystalOps.canonicalize_zp_aunits at git HEAD 1a0c138b: the pose slots only."""
    if c.is_batch:
        c.aunit_centroid, c.aunit_orientation, c.aunit_handedness = canonicalize_aunit_order(c)
    else:
        c.aunit_centroid, c.aunit_orientation, c.aunit_handedness = canonicalize_aunit_order(
            collate_data_list([c], max_z_prime=c.max_z_prime))


def _prepared(row, max_z_prime=3):
    """A clone with ``max_z_prime`` pose slots (stored slot columns truncated), without the legacy per-atom
    ``aunit_batch`` (absent from the synthetic rows), so rows of different files collate together."""
    r = row.clone()
    r._store.pop('aunit_batch', None)
    r.aunit_centroid = r.aunit_centroid[:, :3 * max_z_prime]
    r.aunit_orientation = r.aunit_orientation[:, :3 * max_z_prime]
    r.aunit_handedness = r.aunit_handedness[:, :max_z_prime]
    r.max_z_prime = max_z_prime
    return r


def _batch(rows, max_z_prime=None):
    b = collate_data_list([r.clone() for r in rows], max_z_prime=max_z_prime)
    b.box_analysis()
    return b


def _cells(batch):
    """Per crystal: unit-cell positions (float64), element labels in the builder's layout, T_fc (float64).

    ``mol2ucell`` writes each crystal unit-major, and each unit image-major, so unit k's labels are its own atom
    block of ``z`` repeated ``sym_mult`` times; each set takes its labels from its own ``z``.
    """
    b = batch.clone()
    b.mol2ucell(std_orientation=True)
    out = []
    for g in range(b.num_graphs):
        zp = int(b.z_prime[g])
        n = int(b.num_atoms[g]) // zp
        zb = b.z[b.ptr[g]:b.ptr[g + 1]]
        labels = torch.cat([zb[k * n:(k + 1) * n].repeat(int(b.sym_mult[g])) for k in range(zp)])
        out.append((b.unit_cell_pos[b.unit_cell_batch == g].double(), labels, b.T_fc[g].double()))
    return out


def _crystal_dev(a, b):
    """Largest distance (Angstrom) from an atom of either set to the nearest same-element atom of the other, over
    periodic images (minimum image of the fractional difference)."""
    (pa, la, T), (pb, lb, _) = a, b
    if len(pa) != len(pb) or not torch.equal(torch.sort(la).values, torch.sort(lb).values):
        return float('inf')
    fa, fb = torch.linalg.solve(T, pa.T).T, torch.linalg.solve(T, pb.T).T
    d = fa[:, None] - fb[None]
    d = d - torch.round(d)
    dist = (d @ T.T).norm(dim=-1)
    dist = torch.where(la[:, None] == lb[None], dist, torch.full_like(dist, float('inf')))
    return max(float(dist.min(1).values.max()), float(dist.min(0).values.max()))


def _reordered(batch):
    """[num_graphs] bool: crystals with z_prime > 1 whose used slots change order."""
    order = _canonical_unit_order(batch)
    slot = torch.arange(batch.max_z_prime)
    used = slot[None] < batch.z_prime[:, None]
    return (batch.z_prime > 1) & ((order != slot[None]) & used).any(1)


def _swap_units(row):
    """The same crystal with its units in reverse order: atom blocks and pose slots reversed together."""
    r = row.clone()
    zp = int(r.z_prime)
    n = len(r.z) // zp
    rev = torch.cat([torch.arange(k * n, (k + 1) * n) for k in reversed(range(zp))])
    r.pos, r.z, r.x = r.pos[rev], r.z[rev], r.x[rev]
    k = r.aunit_handedness.shape[1]
    slots = list(reversed(range(zp))) + list(range(zp, k))
    cols = torch.tensor([3 * s + i for s in slots for i in range(3)])
    r.aunit_centroid = r.aunit_centroid[:, cols]
    r.aunit_orientation = r.aunit_orientation[:, cols]
    r.aunit_handedness = r.aunit_handedness[:, torch.tensor(slots)]
    return r


def _zp1_template(mini_rows):
    """A Z' = 1 row of the mini file with at least 8 heavy atoms (non-degenerate principal axes)."""
    for r in mini_rows:
        if int(r.z_prime) == 1 and int((r.z > 1).sum()) >= 8 and int(r.sg_ind) == 14:
            return r
    raise AssertionError('no Z\' = 1 sg 14 template row with 8 heavy atoms in mini_new_csd.pt')


def _synthetic(template, z_primes, seed, copies=False, max_z_prime=3):
    """Rows of z_prime units built from ``template``'s molecule, each unit its own conformer (heavy-atom noise of
    0.1 A, so the internal geometries differ) unless ``copies``, posed from uniform latents in (-0.9, 0.9) with a
    random handedness per slot. Slot order is whatever the draw gives."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for zp in z_primes:
        r = _prepared(template, max_z_prime)
        n = len(r.z)
        blocks = [r.pos + (0 if copies else 0.1 * torch.randn(n, 3, generator=gen)) for _ in range(zp)]
        if copies:
            blocks = [blocks[0]] * zp
        r.pos = torch.cat(blocks)
        r.z, r.x = r.z.repeat(zp), r.x.repeat(zp)
        r.mol_ind = torch.arange(zp).repeat_interleave(n)
        r.num_atoms = torch.tensor(n * zp)
        r.z_prime = torch.tensor([zp])
        for key in ('radius', 'mass', 'mol_volume'):
            r[key] = r[key] * zp
        r.aunit_centroid = torch.zeros(1, 3 * max_z_prime)
        r.aunit_orientation = torch.zeros(1, 3 * max_z_prime)
        r.aunit_handedness = torch.ones(1, max_z_prime, dtype=torch.long)
        r.aunit_handedness[0, :zp] = torch.randint(0, 2, (zp,), generator=gen) * 2 - 1
        r.max_z_prime = max_z_prime
        r.identifier = f'synthetic_zp{zp}_{len(rows)}'
        rows.append(r)
    b = collate_data_list([r.clone() for r in rows], max_z_prime=max_z_prime)
    b.box_analysis()
    b.latent_to_cell_params(0.9 * (2 * torch.rand(b.num_graphs, 6 + 6 * max_z_prime, generator=gen) - 1))
    return b.batch_to_list()


def _distinct_sources(mini_rows):
    """name -> Z' > 1 rows with distinct per-unit conformers, each also unit-swapped, so at least one of its two
    descriptions is reordered."""
    real = [_prepared(r) for r in mini_rows if int(r.z_prime) > 1]
    sources = {'mini_new_csd': real + [_swap_units(r) for r in real]}
    if MINI_NEW_NEW.exists():
        rows = [_prepared(r) for r in torch.load(MINI_NEW_NEW, weights_only=False, map_location='cpu')
                if int(r.z_prime) > 1]
        sources['mini_new_new_csd'] = rows + [_swap_units(r) for r in rows]
    synthetic = _synthetic(_zp1_template(mini_rows), [2, 3, 3, 2, 3, 2, 3, 3], seed=0)
    sources['synthetic'] = synthetic + [_swap_units(r) for r in synthetic]
    return sources


def _assert_same_store(a, b):
    assert set(a.keys()) == set(b.keys())
    for key in a.keys():
        x, y = a[key], b[key]
        if torch.is_tensor(x):
            assert torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape, key
            assert torch.equal(x, y), key
        elif isinstance(x, np.ndarray):
            assert np.array_equal(x, y), key
        elif isinstance(x, list):
            assert len(x) == len(y), key
            for u, v in zip(x, y):
                assert (np.array_equal(u, v) if isinstance(u, np.ndarray) else u == v), key
        else:
            assert x == y, key


@pytest.mark.parametrize('max_z_prime', [1, 3])
def test_zp1_batches_are_bit_identical_to_the_slots_only_function(mini_rows, max_z_prime):
    rows = [r for r in mini_rows if int(r.z_prime) == 1]
    assert len(rows) == 95
    new = collate_data_list([r.clone() for r in rows], max_z_prime=max_z_prime)
    old = new.clone()
    pos_before = new.pos
    new.canonicalize_zp_aunits()
    _slots_only(old)
    _assert_same_store(new, old)
    assert new.pos is pos_before  # no atom attribute is touched on a Z' = 1 batch


def test_zp1_single_crystals_are_bit_identical_to_the_slots_only_function(mini_rows):
    rows = [r for r in mini_rows if int(r.z_prime) == 1][:20]
    for r in rows:
        new, old = r.clone(), r.clone()
        new.max_z_prime = old.max_z_prime = 1
        new.canonicalize_zp_aunits()
        _slots_only(old)
        _assert_same_store(new, old)


def test_distinct_conformer_rows_rebuild_the_same_crystal(mini_rows):
    for name, rows in _distinct_sources(mini_rows).items():
        batch = _batch(rows, max_z_prime=3)
        reordered = _reordered(batch)
        before = _cells(batch)
        fixed = batch.clone()
        fixed.canonicalize_zp_aunits()
        slots_only = batch.clone()
        _slots_only(slots_only)
        after, wrong = _cells(fixed), _cells(slots_only)
        dev = [_crystal_dev(x, y) for x, y in zip(before, after)]
        miss = [_crystal_dev(x, y) for x, y in zip(before, wrong)]
        assert max(dev) < TOL, (name, dev)
        assert int(reordered.sum()) >= len(rows) // 2, name  # the check ran on reordered rows
        assert max(miss) > TEETH, (name, miss)  # and would have caught the slots-only function
        assert all(m < TOL for m, r in zip(miss, reordered.tolist()) if not r)


def test_mixed_batch_moves_only_the_reordered_crystals(mini_rows):
    template = _zp1_template(mini_rows)
    rows = _synthetic(template, [1, 1, 3, 2, 1, 3, 2, 1, 1], seed=1)
    batch = _batch(rows, max_z_prime=3)
    reordered = _reordered(batch)
    assert reordered.any() and not reordered.all()
    fixed = batch.clone()
    fixed.canonicalize_zp_aunits()
    assert max(_crystal_dev(x, y) for x, y in zip(_cells(batch), _cells(fixed))) < TOL
    for g in range(batch.num_graphs):
        s, e = int(batch.ptr[g]), int(batch.ptr[g + 1])
        if not bool(reordered[g]):
            assert torch.equal(fixed.pos[s:e], batch.pos[s:e]) and torch.equal(fixed.z[s:e], batch.z[s:e])


def test_latent_round_trip_rebuilds_the_crystal(mini_rows):
    for name, rows in _distinct_sources(mini_rows).items():
        batch = _batch(rows, max_z_prime=3)
        before = _cells(batch)
        canon = batch.clone()
        lat = canon.latent_params()  # canonicalizes in place
        rebuilt = canon.clone()
        rebuilt.latent_to_cell_params(lat)
        k = batch.max_z_prime
        slot_of_col = torch.cat([torch.full((6,), -1), torch.arange(k).repeat_interleave(3).repeat(2)])
        used_col = slot_of_col[None] < batch.z_prime[:, None]  # cell columns and the used slots' columns
        inside = (((lat > -0.99) & (lat < 1 - 1e-4)) | ~used_col).all(1)  # no used entry clamped on decoding
        assert int(inside.sum()) >= len(rows) // 2, name
        dev = [_crystal_dev(x, y) for x, y, ok in zip(before, _cells(rebuilt), inside.tolist()) if ok]
        assert max(dev) < TOL, (name, dev)


def test_copy_rows_build_the_same_crystal_as_the_slots_only_function(mini_rows):
    template = _zp1_template(mini_rows)
    copies = _synthetic(template, [2, 3, 3, 2, 3], seed=2, copies=True)
    exact = []
    for r in mini_rows:  # as overwrite_conformer_to_zp: every unit becomes unit 0's atoms
        if int(r.z_prime) == 1:
            continue
        c = _prepared(r)
        n = len(c.z) // int(c.z_prime)
        c.pos, c.z, c.x = c.pos[:n].repeat(int(c.z_prime), 1), c.z[:n].repeat(int(c.z_prime)), c.x[:n].repeat(
            int(c.z_prime))
        exact += [c, _swap_units(c)]
    for rows in (copies, exact):
        batch = _batch(rows, max_z_prime=3)
        assert _reordered(batch).any()
        fixed, old = batch.clone(), batch.clone()
        fixed.canonicalize_zp_aunits()
        _slots_only(old)
        assert torch.equal(fixed.pos, old.pos)  # identical blocks: moving them changes nothing
        a, b = fixed.clone(), old.clone()
        a.mol2ucell(std_orientation=True)
        b.mol2ucell(std_orientation=True)
        assert torch.equal(a.unit_cell_pos, b.unit_cell_pos)


def test_post_copy_file_rows_build_the_same_crystal_as_the_slots_only_function():
    rows = [_prepared(r) for r in _load(MINI_PROT) if int(r.z_prime) > 1]
    batch = _batch(rows + [_swap_units(r) for r in rows], max_z_prime=3)
    assert _reordered(batch).any()
    fixed, old = batch.clone(), batch.clone()
    fixed.canonicalize_zp_aunits()
    _slots_only(old)
    dev = [_crystal_dev(x, y) for x, y in zip(_cells(fixed), _cells(old))]
    assert max(dev) < TOL, dev


def test_every_attribute_laid_out_unit_by_unit_moves_with_its_slot(mini_rows):
    rows = _synthetic(_zp1_template(mini_rows), [3, 2, 3, 3], seed=3)
    reordered = _reordered(_batch(rows, max_z_prime=3)).tolist()
    rows = [r if x else _swap_units(r) for r, x in zip(rows, reordered)]  # every row reordered
    for i, r in enumerate(rows):
        n_atoms, zp = len(r.z), int(r.z_prime)
        n = n_atoms // zp
        r.deposit_site = torch.arange(n_atoms) + 1000 * i
        r.atom_labels = [f'r{i}a{j}' for j in range(n_atoms)]
        r.file_bonds = np.stack([np.arange(n_atoms - 1), np.arange(1, n_atoms)]).astype(np.int64)
        r.unit_radius = torch.full((1, 3), float('nan'))
        r.unit_radius[0, :zp] = torch.arange(zp) + 10.0 * i
        r.aunit_op = torch.full((1, 3), -1, dtype=torch.long)
        r.aunit_op[0, :zp] = torch.arange(zp) + 3
    batch = _batch(rows, max_z_prime=3)
    counts = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
    batch.mol_bonds = [torch.stack([torch.arange(0, c - 1, 2), torch.arange(1, c, 2)]) for c in counts]
    order = _canonical_unit_order(batch)
    assert _reordered(batch).all()
    new = batch.clone()
    new.canonicalize_zp_aunits()
    for g in range(batch.num_graphs):
        zp = int(batch.z_prime[g])
        s, e = int(batch.ptr[g]), int(batch.ptr[g + 1])
        n = (e - s) // zp
        for j in range(zp):
            src = slice(s + int(order[g, j]) * n, s + (int(order[g, j]) + 1) * n)
            dst = slice(s + j * n, s + (j + 1) * n)
            for key in ('pos', 'z', 'x', 'deposit_site'):
                assert torch.equal(new[key][dst], batch[key][src]), key
            assert new.atom_labels[g][j * n:(j + 1) * n] == batch.atom_labels[g][int(order[g, j]) * n:
                                                                                 (int(order[g, j]) + 1) * n]
            assert float(new.unit_radius[g, j]) == float(batch.unit_radius[g, int(order[g, j])])
            assert int(new.aunit_op[g, j]) == int(batch.aunit_op[g, int(order[g, j])])
        assert torch.isnan(new.unit_radius[g, zp:]).all() and (new.aunit_op[g, zp:] == -1).all()
        assert torch.equal(new.mol_ind[s:e], torch.arange(zp).repeat_interleave(n))  # numbers units by position
        site_old, site_new = batch.deposit_site[s:e], new.deposit_site[s:e]
        for key in ('file_bonds', 'mol_bonds'):
            old_pairs = {tuple(sorted(int(site_old[i]) for i in pair)) for pair in np.asarray(batch[key][g]).T}
            new_pairs = {tuple(sorted(int(site_new[i]) for i in pair)) for pair in np.asarray(new[key][g]).T}
            assert old_pairs == new_pairs, key  # renumbered: the same atoms stay bonded
        assert type(new.file_bonds[g]) is np.ndarray and torch.is_tensor(new.mol_bonds[g])


def test_a_single_crystal_moves_as_in_a_batch(mini_rows):
    rows = _synthetic(_zp1_template(mini_rows), [3, 2, 3], seed=4)
    batch = _batch(rows, max_z_prime=3)
    batch.canonicalize_zp_aunits()
    for g, r in enumerate(rows):
        single = r.clone()
        single.canonicalize_zp_aunits()
        s, e = int(batch.ptr[g]), int(batch.ptr[g + 1])
        for key in ('pos', 'z', 'x', 'mol_ind'):
            assert torch.equal(single[key], batch[key][s:e]), key
        for key in ('aunit_centroid', 'aunit_orientation', 'aunit_handedness'):
            assert torch.equal(single[key][0], batch[key][g]), key


def test_the_order_agrees_with_canonicalize_aunit_order():
    gen = torch.Generator().manual_seed(5)
    B, k = 256, 4
    batch = SimpleNamespace(num_graphs=B, max_z_prime=k, device=torch.device('cpu'),
                            z_prime=torch.randint(0, k + 1, (B,), generator=gen),
                            aunit_centroid=torch.rand(B, 3 * k, generator=gen),
                            aunit_orientation=torch.rand(B, 3 * k, generator=gen),
                            aunit_handedness=torch.randint(0, 2, (B, k), generator=gen) * 2 - 1)
    order = _canonical_unit_order(batch)
    centroid, orientation, handedness = canonicalize_aunit_order(batch)
    used = torch.arange(k)[None] < batch.z_prime[:, None]
    gathered = torch.gather(batch.aunit_centroid.reshape(B, k, 3), 1, order[..., None].expand(-1, -1, 3))
    assert torch.equal(gathered[used], centroid.reshape(B, k, 3)[used])
    gathered = torch.gather(batch.aunit_orientation.reshape(B, k, 3), 1, order[..., None].expand(-1, -1, 3))
    assert torch.equal(gathered[used], orientation.reshape(B, k, 3)[used])
    assert torch.equal(torch.gather(batch.aunit_handedness, 1, order), handedness)
    assert (order[used] < batch.z_prime[:, None].expand(-1, k)[used]).all()  # used slots sort first


def test_refuses_a_crystal_it_cannot_split_into_unit_blocks(mini_rows):
    r = _synthetic(_zp1_template(mini_rows), [2], seed=6)[0]
    r.aunit_centroid[0, :6] = torch.tensor([0.4, 0.4, 0.4, 0.1, 0.1, 0.1])  # slot 1 sorts first
    odd = r.clone()
    odd.pos, odd.z, odd.x = odd.pos[:-1], odd.z[:-1], odd.x[:-1]
    odd.num_atoms = torch.tensor(len(odd.z))
    with pytest.raises(ValueError, match='not divisible by z_prime'):
        collate_data_list([odd], max_z_prime=3).canonicalize_zp_aunits()
    behind = r.clone()
    behind.aunit_centroid[0, :3] = torch.tensor([1.1, 1.1, 1.1])  # a used slot sorts behind the unused slot 2
    with pytest.raises(ValueError, match='sorting behind an unused one'):
        collate_data_list([behind], max_z_prime=3).canonicalize_zp_aunits()

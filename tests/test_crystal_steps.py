"""Contract tests for the structural full-chain steps (crystal_steps.py).

CPU only. find_molecules, parameterize_pose, check_rebuild and check_validity on the 16
CIF fixtures and on constructed rows (Z' = 2 with a shuffled, perturbed second unit; a
molecule on an inversion centre written whole and as a half, with distance and file bonds;
chains; split and mixed deposited sets; bond loops that omit bonds or are malformed;
rotor-rich units whose isomorphism search reaches its bounds; an exactly symmetric
heavy-atom set; relabelled posed blocks; pose flags; the moiety composition), plus the step
contract itself: a new batch every time, pass-through rows bit-identical (also on a second
pass), one status per failing row, isolation of a raising row in every step, ``done`` bits,
``steps_run`` and argument refusals. The comparison with the old CCDC featurizer runs only
where the CSD Python API and its licence are available.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mxtaltools.common.geometry_utils import coor_trans_matrix_np
from mxtaltools.crystal_building.utils import extract_aunit_orientation
from mxtaltools.dataset_utils.data_class_methods import crystal_steps as cs
from mxtaltools.dataset_utils.data_class_methods.crystal_io import DONE_READ
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import (DONE_FIND_MOLECULES, DONE_POSE, DONE_REBUILD,
                                                                       DONE_VALIDITY, STATUS, VALIDITY_CRITERIA)
from mxtaltools.dataset_utils.data_classes import MolCrystalData

FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
FIND = dict(bonds='file', bond_scale=(1.15, 1.35), iso_cap=5000, special_tol=0.1)
FIND_D = dict(FIND, bonds='distance')
POSE = dict(handedness='measured', degenerate_tol=1e-4, handedness_tol=1e-3)
F64 = torch.float64

#: The fixtures' outcome after check_rebuild (file bonds), by identifier: status and Z'.
EXPECTED = {'AJACOF': ('special_position', 0), 'EFUNUP': ('multi_component', 0), 'MBTPHC': ('multi_component', 0),
            'IDOVUT': ('ok', 2)}


_STATUS_OF = {name: i for i, name in enumerate(STATUS)}


def _status(batch):
    return [STATUS[int(s)] for s in batch.status.tolist()]


def _fixtures(dtype=F64):
    return MolCrystalData.read(sorted(FIXTURES.glob('*.cif')), dtype=dtype)


def _chain(batch, find=FIND):
    b1 = batch.find_molecules(**find)
    b2 = b1.parameterize_pose(**POSE)
    return b1, b2, b2.check_rebuild()


def _row_sl(batch, i):
    return slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))


def _exact_image_dev(batch, i):
    """Per unit of row i: its pos block against the exact operator image of its deposited atoms (Angstrom)."""
    sl = _row_sl(batch, i)
    zp = int(batch.z_prime[i])
    pos, dep = batch.pos[sl].double(), batch.deposit_frac[sl].double()
    n_u = len(pos) // zp
    T = torch.as_tensor(coor_trans_matrix_np('f_to_c', batch.cell_lengths[i].double().numpy(),
                                             batch.cell_angles[i].double().numpy()))
    out = []
    for u in range(zp):
        us = slice(u * n_u, (u + 1) * n_u)
        op = torch.as_tensor(batch.symmetry_operators[i][int(batch.aunit_op[i, u])])
        img = dep[us] @ op[:3, :3].T + op[:3, 3]
        fr = pos[us] @ torch.linalg.inv(T).T
        out.append(float(((img + torch.round((fr - img).mean(0)) - fr) @ T.T).norm(dim=1).max()))
    return out


def _snapshot(batch):
    """A deep copy of everything a step could mutate, for before/after comparison."""
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in batch._store.items()}, \
        {k: v.clone() for k, v in batch._slice_dict.items() if torch.is_tensor(v)}


def _same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and \
            torch.equal(torch.nan_to_num(a, nan=1e300), torch.nan_to_num(b, nan=1e300))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _assert_unchanged(batch, snap):
    store, slices = snap
    assert list(batch._store.keys()) == list(store.keys())
    for k, v in store.items():
        assert _same(batch._store[k], v), k
    for k, v in slices.items():
        assert torch.equal(batch._slice_dict[k], v), k


def _p1(cell_deg, frac, z, *, sg=1, attrs=None, dtype=F64):
    """A one-row bare batch from fractional coordinates and a cell in degrees."""
    cell = torch.tensor([list(cell_deg[:3]) + list(np.radians(cell_deg[3:]))], dtype=F64)
    T = torch.as_tensor(coor_trans_matrix_np('f_to_c', np.array(cell_deg[:3], float), np.radians(cell_deg[3:])))
    xyz = torch.as_tensor(np.asarray(frac, float)) @ T.T
    return MolCrystalData.from_arrays(cell, xyz, torch.as_tensor(z), ptr=[0, len(z)], sg=sg, setting='standard',
                                      dtype=dtype, **(attrs or {}))


# a trans-dichloroethene-like molecule centred on the origin (Cartesian, Angstrom): C, Cl, H and their inverses
_HALF = np.array([[0.665, 0.2, 0.0], [1.6, 1.5, 0.3], [1.2, -0.75, -0.1]])
_HALF_Z = [6, 17, 1]
_CUBE = (12.0, 12.0, 12.0, 90.0, 90.0, 90.0)


def _cart_to_frac(cart, cell=_CUBE):
    T = coor_trans_matrix_np('f_to_c', np.array(cell[:3], float), np.radians(cell[3:]))
    return np.asarray(cart) @ np.linalg.inv(T).T


# ======================================================================================
# the fixtures
# ======================================================================================

def test_fixture_chain_statuses_poses_and_exact_images():
    b0 = _fixtures()
    b1, b2, b3 = _chain(b0)
    for i, ident in enumerate(b3.identifier):
        status, zp = EXPECTED.get(ident, ('ok', 1))
        assert (_status(b3)[i], int(b3.z_prime[i])) == (status, zp), ident
        if status != 'ok':
            continue
        # each unit's block is the exact image of its own deposited atoms (no copy of unit 0)
        assert max(_exact_image_dev(b3, i)) < 1e-12, ident
        # the float32 builder reproduces the deposited set expanded by the file's operators
        assert float(b3.rebuild_max_dev[i]) < 1e-4, ident
        assert not bool(b3.label_flip[i]), ident
    ok = b3.status == 0
    assert int(ok.sum()) == 13
    assert torch.equal(b3.done[ok], torch.full_like(b3.done[ok], DONE_READ | DONE_FIND_MOLECULES | DONE_POSE |
                                                    DONE_REBUILD))
    assert [s['step'] for s in b3.steps_run] == ['find_molecules', 'parameterize_pose', 'check_rebuild']
    assert b3.steps_run[0]['args'] == dict(FIND, bond_scale=[1.15, 1.35])
    assert b3.max_z_prime == 2 and b3.aunit_centroid.shape == (16, 6) and b3.aunit_handedness.dtype == torch.long


def test_fixture_file_and_distance_bonds_agree():
    b0 = _fixtures()
    f, d = b0.find_molecules(**FIND), b0.find_molecules(**FIND_D)
    assert _status(f) == _status(d)
    assert torch.equal(f.z_prime, d.z_prime) and torch.equal(f.pos, d.pos) and torch.equal(f.mol_ind, d.mol_ind)


def test_zp2_fixture_keeps_each_units_own_conformer():
    b0 = MolCrystalData.read(FIXTURES / 'IDOVUT.cif', dtype=F64)
    b1, b2, b3 = _chain(b0)
    assert _status(b3) == ['ok'] and int(b3.z_prime[0]) == 2
    n = int(b3.num_atoms[0]) // 2
    heavy = b3.z[:n] > 1
    assert torch.equal(b3.z[:n], b3.z[n:])
    d0, d1 = torch.cdist(b3.pos[:n][heavy], b3.pos[:n][heavy]), torch.cdist(b3.pos[n:][heavy], b3.pos[n:][heavy])
    assert float((d0 - d1).abs().max()) > 1e-3          # two conformers, index-matched
    # the units' internal distances are the deposited components', through the stored permutation
    dep = b0.pos[b1.deposit_site]
    for u in (0, 1):
        sl = slice(u * n, (u + 1) * n)
        torch.testing.assert_close(torch.cdist(b3.pos[sl], b3.pos[sl]), torch.cdist(dep[sl], dep[sl]),
                                   rtol=0, atol=1e-10)
    assert torch.equal(b3.mol_ind, torch.arange(2).repeat_interleave(n))
    assert b3.unit_radius.shape == (1, 2) and torch.isclose(b3.radius, b3.unit_radius.sum(1)).all()


def test_find_molecules_reorders_every_per_atom_field_consistently():
    b0 = _fixtures()
    b1 = b0.find_molecules(**FIND)
    for i in range(b1.num_graphs):
        sl = _row_sl(b1, i)
        site = b1.deposit_site[sl]
        assert torch.equal(torch.sort(site).values, torch.arange(len(site)))
        assert torch.equal(b1.z[sl], b0.z[sl][site])
        assert torch.equal(b1.occupancy[sl], b0.occupancy[sl][site])
        assert b1.atom_labels[i] == [b0.atom_labels[i][j] for j in site.tolist()]
        # positions move only by lattice vectors
        T_cf = torch.linalg.inv(b1.T_fc[i])
        df = (b1.pos[sl] - b0.pos[sl][site]) @ T_cf.T
        assert float((df - torch.round(df)).abs().max()) < 1e-9
        if _status(b1)[i] == 'ok' and b0.file_bonds[i] is not None:
            # file bonds renumbered onto the new order name the same atom pairs
            old = {tuple(sorted((b0.atom_labels[i][a], b0.atom_labels[i][b]))) for a, b in b0.file_bonds[i].T}
            new = {tuple(sorted((b1.atom_labels[i][a], b1.atom_labels[i][b]))) for a, b in b1.file_bonds[i].T}
            assert old == new
            assert b1.mol_bonds[i].shape[0] == 2


# ======================================================================================
# constructed rows
# ======================================================================================

def _kuntev_zp2(shuffle_seed=0, noise=0.02):
    """P1 with KUNTEV's molecule and its inversion image (shuffled, perturbed): Z' = 2, two conformers."""
    k = MolCrystalData.read(FIXTURES / 'KUNTEV.cif', dtype=F64)
    T = k.T_fc[0]
    frac = k.pos @ torch.linalg.inv(T).T
    g = torch.Generator().manual_seed(shuffle_seed)
    perm = torch.randperm(len(frac), generator=g)
    second = (-frac @ T.T + noise * torch.randn(len(frac), 3, generator=g, dtype=F64)) @ torch.linalg.inv(T).T
    cell = torch.cat([k.cell_lengths, k.cell_angles], 1)
    xyz = torch.cat([frac, second[perm]]) @ T.T
    z = torch.cat([k.z, k.z[perm]])
    return MolCrystalData.from_arrays(cell, xyz, z, ptr=[0, len(z)], sg=1, setting='standard', dtype=F64), perm


def test_zp2_constructed_units_reindexed_onto_unit_0():
    b0, perm = _kuntev_zp2()
    b1, b2, b3 = _chain(b0, FIND_D)
    assert _status(b3) == ['ok'] and int(b3.z_prime[0]) == 2
    n = len(perm)
    # unit 1's atom j is the image of unit 0's atom j: the shuffle is undone
    assert torch.equal(b1.deposit_site[:n], torch.arange(n))
    assert torch.equal(b1.deposit_site[n:] - n, torch.argsort(perm))
    assert float(b3.rebuild_max_dev[0]) < 1e-4
    # each unit keeps its own (perturbed) conformer
    assert max(_exact_image_dev(b3, 0)) < 1e-12
    edges = {tuple(e) for e in b1.mol_bonds[0].T.tolist()}
    assert all((a + n, b + n) in edges for a, b in edges if b < n)


def test_a_molecule_wrapped_atom_by_atom_is_made_whole():
    whole = np.concatenate([_HALF, -_HALF])                 # centred on a cell corner
    frac = _cart_to_frac(whole)
    wrapped = frac - np.floor(frac)                         # each atom wrapped on its own: the molecule is torn
    zz = _HALF_Z * 2
    bonds = np.array([[0, 3], [0, 1], [0, 2], [3, 4], [3, 5]]).T
    for find, attrs in ((FIND_D, None), (FIND, dict(file_bonds=[bonds]))):
        b0 = _p1(_CUBE, wrapped, zz, attrs=attrs)
        b1 = b0.find_molecules(**find)
        assert _status(b1) == ['ok'] and 'moved by a lattice vector' in b1.message[0]
        d = torch.cdist(b1.pos, b1.pos)
        torch.testing.assert_close(d, torch.cdist(torch.as_tensor(whole)[b1.deposit_site],
                                                  torch.as_tensor(whole)[b1.deposit_site]), rtol=0, atol=1e-10)
        # the deposited coordinates are kept as given; the unit moved by lattice vectors only
        torch.testing.assert_close(b1.deposit_frac, torch.as_tensor(wrapped)[b1.deposit_site], rtol=0, atol=1e-12)
        df = b1.pos @ torch.linalg.inv(b1.T_fc[0]).T - b1.deposit_frac
        assert float((df - torch.round(df)).abs().max()) < 1e-12 and float(df.abs().max()) > 0.5


def test_special_position_and_z_prime_fractional():
    whole = np.concatenate([_HALF, -_HALF])
    zz = _HALF_Z * 2
    half_bonds = np.array([[0, 0], [1, 2]])                         # C-Cl, C-H
    whole_bonds = np.array([[0, 0, 0, 3, 3], [1, 2, 3, 4, 5]])     # and the C=C joining the halves
    for find in (FIND_D, FIND):
        attrs = (lambda bonds: dict(file_bonds=[bonds])) if find['bonds'] == 'file' else (lambda bonds: None)
        on_centre = _p1(_CUBE, _cart_to_frac(whole), zz, sg=2, attrs=attrs(whole_bonds))
        half = _p1(_CUBE, _cart_to_frac(_HALF), _HALF_Z, sg=2, attrs=attrs(half_bonds))
        p1_whole = _p1(_CUBE, _cart_to_frac(whole), zz, sg=1, attrs=attrs(whole_bonds))
        assert _status(on_centre.find_molecules(**find)) == ['special_position'], find
        r = half.find_molecules(**find)
        assert _status(r) == ['z_prime_fractional'] and 'order 2' in r.message[0], find
        assert _status(p1_whole.find_molecules(**find)) == ['ok'], find


def test_unlisted_bond_with_file_bonds():
    # one Pd dimer whose bond loop omits the Pd-Pd bond joining its two identical halves (2.8 A): not Z' = 2
    half = np.array([[0.0, 0.0, 0.0], [0.4, 2.0, 0.3], [-0.6, -0.4, 2.2], [1.3, 2.9, 0.9], [-1.2, 2.6, -0.4],
                     [2.3, 3.6, 0.2]])
    hb = [(0, 1), (0, 2), (1, 3), (1, 4), (3, 5)]
    cart = np.concatenate([half, half @ np.diag([-1.0, -1.0, 1.0]) + np.array([2.8, 0.0, 0.0])]) + 10
    cell = (25.0, 26.0, 27.0, 90.0, 90.0, 90.0)
    listed = np.array(hb + [(a + 6, b + 6) for a, b in hb]).T
    for sg, z_value in ((1, -1), (2, -1), (2, 2)):
        row = _p1(cell, _cart_to_frac(cart, cell), [46, 7, 17, 6, 6, 8] * 2, sg=sg,
                  attrs=dict(file_bonds=[listed], z_value=[z_value]))
        r = row.find_molecules(**FIND)
        assert _status(r) == ['unlisted_bond'] and 'atoms 0 and 6' in r.message[0], (sg, z_value)
        assert int(r.n_components[0]) == 2
    # with the bond listed: one molecule
    joined = _p1(cell, _cart_to_frac(cart, cell), [46, 7, 17, 6, 6, 8] * 2,
                 attrs=dict(file_bonds=[np.concatenate([listed, [[0], [6]]], 1)]))
    r = joined.find_molecules(**FIND)
    assert _status(r) == ['ok'] and int(r.z_prime[0]) == 1
    # a bond loop without X-H bonds (SHELXL without BOND $H): named, and distance bonds recover the molecule
    b0 = MolCrystalData.read(FIXTURES / 'KUNTEV.cif', dtype=F64)
    fb, z = b0.file_bonds[0], b0.z.numpy()
    b0.file_bonds = [fb[:, (z[fb[0]] > 1) & (z[fb[1]] > 1)]]
    r = b0.find_molecules(**FIND)
    assert _status(r) == ['unlisted_bond'] and 'no bond to hydrogen' in r.message[0]
    assert _status(b0.find_molecules(**FIND_D)) == ['ok']


def test_malformed_file_bonds_fail_their_row_only():
    names = ('KUNTEV', 'NUKCOL', 'CEKZOH')
    good = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in names], dtype=F64)
    ref = good.find_molecules(**FIND)
    n1 = int(good.num_atoms[1])
    fb = good.file_bonds[1]
    bad_entries = dict(out_of_range=np.concatenate([fb, [[0], [n1]]], 1), negative=np.concatenate([fb, [[0], [-1]]], 1),
                       floats=fb.astype(np.float64), self_bond=np.concatenate([fb, [[3], [3]]], 1), shape=fb.T.copy(),
                       ragged=[[0, 1], [2]])
    for label, entry in bad_entries.items():
        for find in (FIND, FIND_D):
            b = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in names], dtype=F64)
            b.file_bonds = [good.file_bonds[0], entry, good.file_bonds[2]]
            r = b.find_molecules(**find)
            assert _status(r)[1] == 'find_molecules_failed' and 'file_bonds' in r.message[1], (label, find['bonds'])
            other = r.select_rows([0, 2])
            expect = (ref if find is FIND else good.find_molecules(**FIND_D)).select_rows([0, 2])
            for k in ('status', 'pos', 'z', 'mol_ind', 'deposit_site', 'z_prime'):
                assert _same(other[k], expect[k]), (label, k)
            # the failing row keeps its entry as given
            assert _same(r.file_bonds[1], entry) if not isinstance(entry, list) else r.file_bonds[1] == entry


def test_polymer_statuses():
    chain = _p1((3.0, 10.0, 10.0, 90.0, 90.0, 90.0), [[0.0, 0.0, 0.0], [0.5, 0.05, 0.0]], [6, 6])
    r = chain.find_molecules(**FIND_D)
    assert _status(r) == ['polymer'] and 'lattice translate' in r.message[0]
    declared = _p1(_CUBE, _cart_to_frac(np.concatenate([_HALF, -_HALF]) + 6), _HALF_Z * 2,
                   attrs=dict(moiety=['(C2 H2 Cl2)n']))
    assert _status(declared.find_molecules(**FIND_D)) == ['polymer']
    # a self-map generating an infinite chain (a 2_1 screw) against a finite one (an inversion)
    screw, inversion = np.eye(4), np.diag([-1.0, -1.0, -1.0, 1.0])
    screw[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    screw[1, 3] = 0.5
    assert cs._closure_order([screw]) is None
    assert cs._closure_order([inversion]) == 2


def test_multi_component_mismatch_no_bonds_and_split():
    mol = np.concatenate([_HALF, -_HALF]) + 6
    water = np.array([[2.0, 2.0, 2.0], [2.95, 2.0, 2.0], [1.75, 2.9, 2.0]])
    mixed = _p1(_CUBE, _cart_to_frac(np.concatenate([mol, water])), _HALF_Z * 2 + [8, 1, 1])
    r = mixed.find_molecules(**FIND_D)
    assert _status(r) == ['multi_component'] and int(r.n_components[0]) == 2
    assert torch.equal(torch.unique(r.mol_ind), torch.tensor([0, 1]))
    wrong_z = _p1(_CUBE, _cart_to_frac(mol), _HALF_Z * 2, attrs=dict(z_value=[2]))
    assert _status(wrong_z.find_molecules(**FIND_D)) == ['z_prime_mismatch']
    atoms = _p1(_CUBE, [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]], [18, 18])
    assert _status(atoms.find_molecules(**FIND_D)) == ['no_bonds']
    no_loop = MolCrystalData.read(FIXTURES / 'KUNTEV.cif', dtype=F64)
    no_loop.file_bonds = [None]
    assert _status(no_loop.find_molecules(**FIND)) == ['no_bonds']
    # a chain C-C-C-Cl in P-1 deposited with its last two atoms replaced by their inversion image
    chain = np.array([[3.0, 3.0, 3.0], [4.5, 3.0, 3.0], [5.0, 4.4, 3.0], [6.5, 4.6, 3.5]])
    split = np.concatenate([chain[:2], -chain[2:]])
    r = _p1(_CUBE, _cart_to_frac(split), [6, 6, 6, 17], sg=2).find_molecules(**FIND_D)
    assert _status(r) == ['split_molecule']
    assert _status(_p1(_CUBE, _cart_to_frac(chain), [6, 6, 6, 17], sg=2).find_molecules(**FIND_D)) == ['ok']


def _benzene(offset):
    ang = np.arange(6) * np.pi / 3
    ring = np.stack([1.39 * np.cos(ang), 1.39 * np.sin(ang), np.zeros(6)], 1)
    hyd = ring * (2.47 / 1.39)
    return np.concatenate([ring, hyd]) + offset, [6] * 6 + [1] * 6


def test_unit_order_at_the_cap_and_degenerate_axes():
    a, za = _benzene(np.array([3.0, 3.0, 3.0]))
    b, zb = _benzene(np.array([8.0, 8.0, 8.0]))
    two = _p1(_CUBE, _cart_to_frac(np.concatenate([a, b])), za + zb)
    # one isomorphism allowed: the search among the geometrically nearest atoms finds an exact order (benzene
    # has 12 automorphisms that are also its symmetries, so any of them is exact; ties prefer a proper fit)
    r = two.find_molecules(**dict(FIND_D, iso_cap=1))
    assert _status(r) == ['ok']
    p = r.pos.numpy()
    rms, _ = cs._kabsch(p[12:], p[:12], improper=False)
    assert rms < 1e-9
    # a bound too small to reach a first isomorphism of KUNTEV's 39 heavy atoms (20 candidate pairs)
    k2, _ = _kuntev_zp2()
    r = k2.find_molecules(**dict(FIND_D, iso_cap=1))
    assert _status(r) == ['unit_order_failed'] and 'iso_cap' in r.message[0]
    found = two.find_molecules(**FIND_D)
    assert _status(found) == ['ok'] and int(found.z_prime[0]) == 2
    # a regular hexagon has two equal principal moments: its axes are undetermined
    posed = found.parameterize_pose(**POSE)
    assert _status(posed) == ['degenerate_axes']
    small = _p1(_CUBE, _cart_to_frac(_HALF + 6), _HALF_Z).find_molecules(**FIND_D)
    assert _status(small.parameterize_pose(**POSE)) == ['degenerate_axes']        # two heavy atoms


def _tbu_chain(k, cl_pos):
    """Heavy atoms of a zigzag chain of k C, each carrying a tert-butyl, ending in a phenyl ring with a Cl at ring
    position ``cl_pos`` (2 ortho, 3 meta): (cart, z, bonds [E, 2]). The rotors make the automorphism group 6^k."""
    cart, z, bonds = [], [], []

    def add(p, e):
        cart.append(np.asarray(p, float))
        z.append(e)
        return len(cart) - 1

    prev = None
    for i in range(k):
        c = add([1.27 * i, 0.8 * (i % 2), 0.0], 6)
        if prev is not None:
            bonds.append((prev, c))
        sgn = 1 if i % 2 else -1
        q = add([1.27 * i, 0.8 * (i % 2) + 1.5 * sgn, 0.0], 6)
        bonds.append((c, q))
        for t in range(3):
            ang = 2 * np.pi * t / 3 + 0.3 * i
            bonds.append((q, add([1.27 * i + 0.5 * np.cos(ang), 0.8 * (i % 2) + 2.0 * sgn,
                                  1.4 * np.sin(ang) + 0.2 * t], 6)))
        prev = c
    centre = np.array([1.27 * k + 1.6, 0.0, 0.0])
    ring = [add(centre + 1.39 * np.array([np.cos(np.pi + np.pi * j / 3), np.sin(np.pi + np.pi * j / 3), 0.0]), 6)
            for j in range(6)]
    bonds += [(ring[j], ring[(j + 1) % 6]) for j in range(6)] + [(prev, ring[0])]
    j = cl_pos - 1
    bonds.append((ring[j], add(centre + 3.1 * np.array([np.cos(np.pi + np.pi * j / 3), np.sin(np.pi + np.pi * j / 3),
                                                         0.0]), 17)))
    return np.array(cart), np.array(z), np.array(bonds)


def _triarm(k):
    """Heavy atoms of a molecule with three arms (graph symmetry over the arms) of k tert-butyl-carrying C each, the
    arms in different conformations: (cart, z, bonds [E, 2])."""
    cart, z, bonds = [[0.0, 0.0, 0.0]], [6], []
    for arm in range(3):
        ang = 2 * np.pi * arm / 3
        d = np.array([np.cos(ang), np.sin(ang), 0.25 * (arm - 1)])
        perp = np.array([-np.sin(ang), np.cos(ang), 0.3 * arm])
        unit = perp / np.linalg.norm(perp)
        prev = 0
        for i in range(k):
            p = (i + 1) * 1.5 * d + 0.4 * (i % 2) * perp + 0.2 * arm * np.array([0, 0, 1.0]) * (i + 1)
            cart.append(p)
            z.append(6)
            c = len(cart) - 1
            bonds.append((prev, c))
            q = p + 1.5 * unit
            cart.append(q)
            z.append(6)
            bonds.append((c, len(cart) - 1))
            qi = len(cart) - 1
            for t in range(3):
                a2 = 2 * np.pi * t / 3 + 0.7 * arm + 0.3 * i
                cart.append(q + 0.9 * unit + 1.2 * np.array([np.cos(a2), np.sin(a2), 0.5 * t]))
                z.append(6)
                bonds.append((qi, len(cart) - 1))
            prev = c
        cart.append((k + 1) * 1.5 * d + 0.1 * arm)
        z.append(8)
        bonds.append((prev, len(cart) - 1))
    return np.array(cart), np.array(z), np.array(bonds)


def _two_units(u0, u1, seed):
    """A P1 row (40 A cube, file bonds) of unit u0 and unit u1 rotated, shuffled and noised by 0.01 A; and the
    shuffle (new index -> old index of u1)."""
    (c0, z0, b0), (c1, z1, b1) = u0, u1
    g = np.random.default_rng(seed)
    q, _ = np.linalg.qr(g.normal(size=(3, 3)))
    c1 = (c1 - c1.mean(0)) @ (q * np.sign(np.linalg.det(q))).T + g.normal(scale=0.01, size=c1.shape)
    perm = g.permutation(len(z1))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    cart = np.concatenate([c0 - c0.mean(0) + 8.0, c1[perm] + 26.0])
    bonds = np.concatenate([b0, inv[b1] + len(z0)]).T.copy()
    cube = (40.0, 40.0, 40.0, 90.0, 90.0, 90.0)
    row = _p1(cube, _cart_to_frac(cart, cube), np.concatenate([z0, z1[perm]]), attrs=dict(file_bonds=[bonds]))
    return row, perm


def test_isomorphism_search_is_bounded():
    import time
    # units of equal composition and degree sequence, different graphs (Cl ortho / meta): before the bound, the
    # search took 6.3 s at k = 6 and about 6.5x more per added rotor, whatever iso_cap
    row, _ = _two_units(_tbu_chain(8, 2), _tbu_chain(8, 3), seed=0)
    t = time.time()
    r = row.find_molecules(**FIND)
    assert _status(r) == ['multi_component'] and time.time() - t < 2.0
    # the search itself, without the hash test in front: the work bound stops it (41.7 s at k = 7 before)
    import networkx as nx
    graphs = []
    for u in (_tbu_chain(7, 2), _tbu_chain(7, 3)):
        g = nx.Graph()
        g.add_nodes_from((i, dict(e=int(e), id=i)) for i, e in enumerate(u[1]))
        g.add_edges_from(map(tuple, u[2]))
        graphs.append(g)
    t = time.time()
    best, exhausted = cs._best_isomorphism(graphs[0], graphs[1], _tbu_chain(7, 2)[0], _tbu_chain(7, 3)[0], 500,
                                           None)
    assert best is None and not exhausted and time.time() - t < 3.0


@pytest.mark.parametrize('unit', ['chain8', 'triarm2'])
def test_rotor_rich_units_are_reindexed_geometrically(unit):
    # 6^8 and 6^6 x 3! automorphisms: past iso_cap, the order comes from the geometric fit (seeded by the units'
    # principal axes), and it is the true correspondence
    u = _tbu_chain(8, 2) if unit == 'chain8' else _triarm(2)
    n = len(u[1])
    for seed in (1, 2):
        row, perm = _two_units(u, u, seed)
        r = row.find_molecules(**FIND)
        assert _status(r) == ['ok'] and int(r.z_prime[0]) == 2, r.message[0]
        assert np.array_equal(perm[r.deposit_site[n:].numpy() - n], np.arange(n))


def test_orientation_equals_extract_aunit_orientation_in_float32():
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(40, 3, generator=g) * 2
    z = torch.full((40,), 6)
    counts = torch.tensor([10, 10, 10, 10])
    batch = torch.arange(4).repeat_interleave(counts)
    from types import SimpleNamespace
    ns = SimpleNamespace(pos=pos, batch=batch, num_graphs=4, num_atoms=counts, z=z, device=torch.device('cpu'))
    rv_ref, h_ref = extract_aunit_orientation(ns, False, canonicalize_orientation=True)
    rv, h, _, _ = cs._orientations(pos, z, counts, 4)
    assert torch.equal(rv, rv_ref) and torch.equal(h.to(h_ref.dtype), h_ref)


# ======================================================================================
# check_rebuild
# ======================================================================================

def _posed_fixtures(names=('KUNTEV', 'NUKCOL', 'CEKZOH')):
    b0 = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in names], dtype=F64)
    return b0.find_molecules(**FIND).parameterize_pose(**POSE)


def test_rebuild_detects_a_moved_pose_and_swapped_elements():
    posed = _posed_fixtures()
    assert _status(posed.check_rebuild()) == ['ok'] * 3
    moved = cs._copy_batch(posed)
    moved.aunit_centroid[1, 0] += 0.01
    r = moved.check_rebuild()
    assert _status(r) == ['ok', 'rebuild_mismatch', 'ok']
    assert float(r.rebuild_max_dev[1]) > 0.05 and float(r.rebuild_max_dev[0]) < 1e-4
    # two heavy atoms of different elements exchange places: the geometry still matches as a set
    swapped = cs._copy_batch(posed)
    sl = _row_sl(swapped, 2)
    zz = swapped.z[sl]
    i = int(torch.nonzero(zz == 6)[0])
    j = int(torch.nonzero((zz > 1) & (zz != 6))[0])
    p = swapped.pos[sl].clone()
    p[[i, j]] = p[[j, i]]
    swapped.pos[sl] = p
    assert _status(swapped.check_rebuild()) == ['ok', 'ok', 'element_mismatch']
    assert _status(swapped.check_rebuild(match_elements=False)) == ['ok', 'ok', 'ok']


def test_label_flip_when_the_pose_uses_another_image():
    posed = _posed_fixtures(('KUNTEV',))
    T_fc, T_cf = cs._transforms64(posed)
    ops = torch.as_tensor(posed.symmetry_operators[0])
    assert np.allclose(posed.symmetry_operators[0][0], np.eye(4))
    # images of the posed unit under the file's operators: 0 is the unit itself, 1 its inversion image
    images, cents = cs._unit_images(posed.pos.double(), posed.z, T_fc[0], T_cf[0], ops)
    img = images[1]
    rv, h, _, _ = cs._orientations(img, posed.z, torch.tensor([len(img)]), 1)
    cell = torch.cat([posed.cell_lengths, posed.cell_angles], 1)
    flipped = MolCrystalData.from_arrays(cell, img, posed.z, ptr=posed.ptr, sg=2, ops=list(posed.symmetry_operators),
                                         z_prime=[1], aunit_centroid=cents[1][None], aunit_orientation=rv,
                                         aunit_handedness=h[:, None], dtype=F64)
    r = flipped.check_rebuild()
    assert _status(r) == ['ok'] and bool(r.label_flip[0]) and 'label_flip' in r.message[0]
    same = posed.check_rebuild()
    assert not bool(same.label_flip[0])


def test_label_flip_when_atoms_sit_on_another_atoms_image():
    # two hydrogens of the posed block exchange places: the built cell still matches the deposited set as a set
    # (hydrogens do not enter the frame), but each sits on the image of the other's deposited atom
    posed = _posed_fixtures(('KUNTEV',))
    h = torch.nonzero(posed.z == 1).reshape(-1)
    i, j = int(h[0]), int(h[-1])
    swapped = cs._copy_batch(posed)
    swapped.pos[[i, j]] = posed.pos[[j, i]]
    r = swapped.check_rebuild()
    n_ops = len(posed.symmetry_operators[0])                 # each swapped atom, under every operator
    assert _status(r) == ['ok'] and bool(r.label_flip[0]) and \
        f'{2 * n_ops} built atom(s) sit on an image of another' in r.message[0]
    assert float(r.rebuild_max_dev[0]) < 1e-4
    # one hydrogen moved onto another: two built atoms share a reference atom
    stacked = cs._copy_batch(posed)
    stacked.pos[i] = posed.pos[j]
    assert _status(stacked.check_rebuild()) == ['unmatched_atoms']


@pytest.mark.parametrize('sg', [1, 2])
def test_a_symmetric_heavy_atom_set_rebuilds_from_the_float32_frame(sg):
    # an exactly centrosymmetric, planar heavy-atom set on a general position: the farthest heavy atom is a tie
    # and the normal has no overlap with it, so the axis signs follow rounding; the stored frame is the float32
    # one the builder computes, and the row rebuilds (it failed at 0.43 A with a float64 frame)
    whole = np.concatenate([_HALF, -_HALF]) + np.array([3.1, 2.7, 4.2])
    posed = _p1(_CUBE, _cart_to_frac(whole), _HALF_Z * 2, sg=sg).find_molecules(**FIND_D).parameterize_pose(**POSE)
    assert _status(posed) == ['ok'] and bool(posed.handedness_unstable[0])
    r = posed.check_rebuild()
    assert _status(r) == ['ok'] and float(r.rebuild_max_dev[0]) < 1e-4
    assert max(_exact_image_dev(r, 0)) < 1e-12


def test_pose_flags():
    # an asymmetric chain O-C-N-Cl in P-1, heavy-atom centroid on the box face x = 1/2 (box [1/2, 1, 1]) or inside
    mol = np.array([[0.0, 0.0, 0.0], [1.47, 0.0, 0.0], [-0.5, 1.3, 0.2], [2.0, 1.4, -0.6]])
    cell = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
    flags = {}
    for x in (0.5, 0.3):
        cart = mol - mol.mean(0) + np.array([x, 0.3, 0.6]) * 20.0
        posed = _p1(cell, _cart_to_frac(cart, cell), [6, 7, 8, 17], sg=2).find_molecules(**FIND_D) \
            .parameterize_pose(**POSE)
        assert _status(posed) == ['ok']
        flags[x] = (bool(posed.asu_ambiguous[0]), bool(posed.is_well_defined[0]), bool(posed.handedness_unstable[0]))
        assert _status(posed.check_rebuild()) == ['ok']
    assert flags[0.5][0] and flags[0.3] == (False, True, False)
    # on the fixtures (handedness_tol 1e-3): three ok rows have no single image in the box, three have unstable signs
    b3 = _chain(_fixtures())[2]
    ok = (b3.status == 0).tolist()
    ids = b3.identifier

    def named(flag):
        return sorted(ids[i] for i in range(len(ids)) if ok[i] and bool(flag[i]))

    assert named(~b3.is_well_defined) == named(b3.asu_ambiguous) == ['DELDEG', 'MIHTEE52', 'ZZZKEA02']
    assert named(b3.handedness_unstable) == ['IDOVUT', 'IYOHIP', 'MIHTEE52']


# ======================================================================================
# check_validity
# ======================================================================================

def test_validity_criteria():
    posed = _posed_fixtures()                    # KUNTEV (S), NUKCOL (C N O), CEKZOH
    r = posed.check_validity(max_atomic_number=8)
    assert _status(r)[0] == 'atomic_number_above_max'
    assert int(r.validity_checked[0]) == 1 << VALIDITY_CRITERIA.index('max_atomic_number')
    assert _status(posed.check_validity(max_heavy_atoms=5)) == ['heavy_atoms_above_max'] * 3
    assert _status(posed.check_validity(max_heavy_atoms=100, heavy_nn_range=(0.9, 3.0))) == ['ok'] * 3
    assert _status(posed.check_validity(heavy_nn_range=(1.6, 3.0))) == ['heavy_nn_out_of_range'] * 3
    assert _status(posed.check_validity(cell_constraints=1e-6, min_contact=0.5)) == ['ok'] * 3
    both = posed.check_validity(max_atomic_number=8, max_heavy_atoms=10)
    assert _status(both)[0] == 'atomic_number_above_max' and 'max_heavy_atoms' in both.message[0]
    assert int(both.done[0]) & DONE_VALIDITY
    # superposed atoms in a bare batch (a molecule written whole on an inversion centre)
    on_centre = _p1(_CUBE, _cart_to_frac(np.concatenate([_HALF, -_HALF])), _HALF_Z * 2, sg=2)
    assert _status(on_centre.check_validity(min_contact=0.5)) == ['atoms_too_close']
    # a monoclinic operator set on a cell whose alpha is not 90 degrees
    bent = _p1((10.0, 11.0, 12.0, 91.0, 100.0, 90.0), _cart_to_frac(_HALF + 5, (10.0, 11.0, 12.0, 91.0, 100.0, 90.0)),
               _HALF_Z, sg=14)
    assert _status(bent.check_validity(cell_constraints=1e-6)) == ['cell_constraints_violated']
    disordered = _p1(_CUBE, _cart_to_frac(_HALF + 6), _HALF_Z, attrs=dict(has_disorder=[True]))
    assert _status(disordered.check_validity(disorder=True)) == ['disordered']
    with pytest.raises(ValueError, match='has_disorder'):
        posed.check_validity(disorder=True)
    with pytest.raises(ValueError, match='units'):
        on_centre.check_validity(heavy_nn_range=(0.9, 3.0))
    with pytest.raises(ValueError, match='occupancy'):
        on_centre.check_validity(occupancy=1e-3)


def test_validity_composition_against_the_moiety():
    posed = _posed_fixtures()                    # moieties: KUNTEV C32 H26 N2 O4 S1, NUKCOL C20 H12 O5 (no H
    # deposited), CEKZOH C7 H10 N2 O2; Z equal to the operator count in all three
    r = posed.check_validity(composition='all')
    assert _status(r) == ['ok', 'composition_mismatch', 'ok'] and 'H 0 (expected 12)' in r.message[1]
    assert int(r.validity_checked[0]) == 1 << VALIDITY_CRITERIA.index('composition')
    assert _status(posed.check_validity(composition='heavy')) == ['ok'] * 3
    # a missing solvent molecule (KUNTEV) fails both; a Z that implies two molecules (CEKZOH) fails both
    b = cs._copy_batch(posed)
    b.moiety = [b.moiety[0] + ',C5 H5 N1', b.moiety[1], b.moiety[2]]
    b.z_value = torch.tensor([2, 4, 16])
    for mode in ('heavy', 'all'):
        assert _status(b.check_validity(composition=mode)) == ['composition_mismatch', 'ok' if mode == 'heavy' else
                                                               'composition_mismatch', 'composition_mismatch']
    # rows without a moiety or a Z are not checked (their bit stays clear); a batch with none raises
    b.moiety = [None, b.moiety[1], 'C. H4']     # the last does not parse
    r = b.check_validity(composition='heavy')
    assert _status(r) == ['ok', 'ok', 'ok'] and r.validity_checked.tolist() == [0, 1 << 7, 0]
    on_centre = _p1(_CUBE, _cart_to_frac(np.concatenate([_HALF, -_HALF])), _HALF_Z * 2, sg=2)
    with pytest.raises(ValueError, match='moiety'):
        on_centre.check_validity(composition='all')
    with pytest.raises(ValueError):
        posed.check_validity(composition='some')


def test_min_contact_passes_one_cutoff(monkeypatch):
    # the unit cell's atoms are searched against one scalar cutoff, never an [n, n] matrix (quadratic memory)
    posed = _posed_fixtures(('KUNTEV',))
    seen = []
    real = cs._contacts

    def spy(xa, xb, T_fc, cutoff, **kw):
        seen.append(np.ndim(cutoff))
        return real(xa, xb, T_fc, cutoff, **kw)

    monkeypatch.setattr(cs, '_contacts', spy)
    assert _status(posed.check_validity(min_contact=0.5)) == ['ok'] and seen == [0]
    # a scalar cutoff finds exactly the contacts of the equal matrix
    g = np.random.default_rng(0)
    x = g.uniform(size=(60, 3))
    T = coor_trans_matrix_np('f_to_c', np.array([7.0, 8.0, 9.0]), np.radians([80.0, 95.0, 100.0]))
    a = real(x, x, T, 2.0, exclude_self=True)
    b = real(x, x, T, np.full((60, 60), 2.0), exclude_self=True)
    assert all(np.array_equal(u, v) for u, v in zip(a, b)) and len(a[0]) > 0


def test_validity_occupancy_from_a_read_batch():
    # a site 5e-4 off 1 reads as ok (read refuses only sites off by more than 1e-3) and fails a stricter criterion
    frac = _cart_to_frac(np.concatenate([_HALF, -_HALF]) + 6)
    lines = ['data_occ', '_symmetry_Int_Tables_number 1', "_symmetry_space_group_name_H-M 'P 1'",
             'loop_', '_symmetry_equiv_pos_as_xyz', "'x,y,z'"]
    lines += [f'{t} 12.0' for t in ('_cell_length_a', '_cell_length_b', '_cell_length_c')]
    lines += [f'{t} 90.0' for t in ('_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma')]
    lines += ['loop_', '_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y',
              '_atom_site_fract_z', '_atom_site_occupancy']
    for i, (zz, f) in enumerate(zip(_HALF_Z * 2, frac)):
        sym = {6: 'C', 17: 'Cl', 1: 'H'}[zz]
        lines.append(f'{sym}{i} {sym} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f} {0.9995 if i == 1 else 1.0}')
    b = MolCrystalData.read('\n'.join(lines) + '\n', dtype=F64)
    assert _status(b) == ['ok'] and bool(b.has_occupancy[0])
    assert _status(b.check_validity(occupancy=1e-3)) == ['ok']
    assert _status(b.check_validity(occupancy=1e-4)) == ['occupancy_not_one']


# ======================================================================================
# the step contract
# ======================================================================================

def _mixed_batch():
    """Fixtures plus failing read rows (a missing file and malformed text)."""
    return MolCrystalData.read(sorted(FIXTURES.glob('*.cif'))[:6] + ['no_such_file.cif', 'data_x\n_cell_length_a ?\n'],
                               dtype=F64)


def test_steps_return_new_batches_and_leave_inputs_and_failed_rows_untouched():
    b0 = _mixed_batch()
    snaps = [_snapshot(b0)]
    b1 = b0.find_molecules(**FIND)
    snaps.append(_snapshot(b1))
    b2 = b1.parameterize_pose(**POSE)
    snaps.append(_snapshot(b2))
    b3 = b2.check_rebuild()
    snaps.append(_snapshot(b3))
    b4 = b3.check_validity(max_atomic_number=100, min_contact=0.5)
    for batch, snap in zip((b0, b1, b2, b3), snaps):
        _assert_unchanged(batch, snap)
    # rows that were not ok on entry to a step leave it bit-identical in every attribute they had
    for before, after in ((b0, b1), (b1, b2), (b2, b3), (b3, b4)):
        skipped = torch.nonzero(before.status != 0).reshape(-1)
        sb, sa = before.select_rows(skipped), after.select_rows(skipped)
        for k in sb.keys():
            if k in ('steps_run',):
                continue
            assert _same(sb[k], sa[k]), k
    assert _status(b0)[-2] == 'unreadable_file' and _status(b0)[-1] != 'ok'
    assert torch.equal(b4.done[-2:], torch.full((2,), DONE_READ, dtype=b4.done.dtype))
    # select_rows splits every new attribute at its level
    ok = b4.select_rows(b4.status == 0)
    assert ok.num_graphs == int((b4.status == 0).sum()) and ok.steps_run == b4.steps_run
    assert len(ok.mol_bonds) == ok.num_graphs and len(ok.deposit_frac) == len(ok.pos)


def test_a_row_that_raises_is_isolated(monkeypatch):
    b0 = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in ('KUNTEV', 'NUKCOL', 'CEKZOH', 'XEWCUY')], dtype=F64)
    real = cs._find_row
    calls = []

    def flaky(z, *args, **kwargs):
        calls.append(len(z))
        if len(z) == 25:                         # NUKCOL
            raise ValueError('boom')
        return real(z, *args, **kwargs)

    monkeypatch.setattr(cs, '_find_row', flaky)
    r = b0.find_molecules(**FIND)
    assert _status(r) == ['ok', 'find_molecules_failed', 'ok', 'ok'] and 'boom' in r.message[1]
    assert len(calls) <= 2 * 4 * 4                # halving: bounded re-runs

    def broken(z, *args, **kwargs):
        raise KeyError('a programming error')

    monkeypatch.setattr(cs, '_find_row', broken)
    with pytest.raises(KeyError):
        b0.find_molecules(**FIND)


@pytest.mark.parametrize('step', ['pose', 'rebuild', 'validity'])
def test_a_row_that_raises_in_a_later_step_is_isolated(monkeypatch, step):
    b1 = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in ('KUNTEV', 'NUKCOL', 'CEKZOH', 'XEWCUY')],
                             dtype=F64).find_molecules(**FIND)
    b2 = b1.parameterize_pose(**POSE)
    b3 = b2.check_rebuild()
    clean = dict(pose=b2, rebuild=b3, validity=b3.check_validity(max_atomic_number=100))[step]
    target = 1
    if step == 'pose':
        real = cs._pose_rows

        def flaky(idx, *args):
            if target in idx:
                raise ValueError('boom')
            return real(idx, *args)

        monkeypatch.setattr(cs, '_pose_rows', flaky)
        r = b1.parameterize_pose(**POSE)
    elif step == 'rebuild':
        real = cs._rebuild_rows

        def flaky(batch, idx, *args):
            if target in idx:
                raise RuntimeError('boom')
            return real(batch, idx, *args)

        monkeypatch.setattr(cs, '_rebuild_rows', flaky)
        r = b2.check_rebuild()
    else:
        real = cs._validity_row

        def flaky(batch, b, *args):
            if b == target:
                raise ValueError('boom')
            return real(batch, b, *args)

        monkeypatch.setattr(cs, '_validity_row', flaky)
        r = b3.check_validity(max_atomic_number=100)
    code = dict(pose='pose_failed', rebuild='rebuild_failed', validity='validity_failed')[step]
    expect = _status(clean)
    expect[target] = code
    assert _status(r) == expect and 'boom' in r.message[target]
    assert int(r.done[target]) & dict(pose=DONE_POSE, rebuild=DONE_REBUILD, validity=DONE_VALIDITY)[step]
    others = [0, 2, 3]
    for k in clean.keys():
        if k not in ('steps_run', 'max_z_prime'):
            assert _same(clean.select_rows(others)[k], r.select_rows(others)[k]), k


def test_second_passes_refuse_or_keep_the_rows_they_pass_through():
    b0 = _fixtures()
    b1 = b0.find_molecules(**FIND)
    failed1 = b1.select_rows(b1.status != 0)
    for batch in (b1, failed1):                  # every row through it, ok or not
        with pytest.raises(ValueError, match='runs once'):
            batch.find_molecules(**FIND)
    b2 = b1.parameterize_pose(**POSE)
    for batch in (b2, b2.select_rows(b2.status != 0)):
        with pytest.raises(ValueError, match='runs once'):
            batch.parameterize_pose(**POSE)
    # check_rebuild and check_validity may run again: they re-check the ok rows and keep the others bit-identical
    first = b2.check_rebuild()
    strict = b2.check_rebuild(max_dev=float(first.rebuild_max_dev[first.status == 0].median()), mean_dev=1.0)
    again = strict.check_rebuild()
    skipped = torch.nonzero(strict.status != 0).reshape(-1)
    assert int((strict.status == _STATUS_OF['rebuild_mismatch']).sum()) > 0
    sb, sa = strict.select_rows(skipped), again.select_rows(skipped)
    for k in sb.keys():
        if k != 'steps_run':
            assert _same(sb[k], sa[k]), k
    v1 = again.check_validity(max_heavy_atoms=30)
    v2 = v1.check_validity(max_atomic_number=100)
    skipped = torch.nonzero(v1.status != 0).reshape(-1)
    for k in v1.keys():
        if k != 'steps_run':
            assert _same(v1.select_rows(skipped)[k], v2.select_rows(skipped)[k]), k
    ok = torch.nonzero(v2.status == 0).reshape(-1)
    bits = (1 << VALIDITY_CRITERIA.index('max_heavy_atoms')) | (1 << VALIDITY_CRITERIA.index('max_atomic_number'))
    assert bool((v2.validity_checked[ok] == bits).all())


def test_store_keeps_rows_that_failed_a_step(tmp_path):
    b3 = _chain(_fixtures())[2]
    assert set(_status(b3)) - {'ok'}
    b3.save(tmp_path / 'store', steps_run=b3.steps_run)
    back = MolCrystalData.load(tmp_path / 'store')
    for k in b3.keys():
        assert _same(b3[k], back[k]), k


def test_bad_arguments_raise():
    b0 = MolCrystalData.read(FIXTURES / 'KUNTEV.cif', dtype=F64)
    for bad in (dict(bonds='both'), dict(bond_scale=0), dict(bond_scale=(1.0, 1.0, 1.0)), dict(iso_cap=0),
                dict(iso_cap=True), dict(special_tol=-1.0), dict(special_tol=float('nan'))):
        with pytest.raises(ValueError):
            b0.find_molecules(**dict(FIND, **bad))
    with pytest.raises(ValueError, match='find_molecules'):
        b0.parameterize_pose(**POSE)
    b1 = b0.find_molecules(**FIND)
    with pytest.raises(ValueError, match='already'):
        b1.find_molecules(**FIND)
    for bad in (dict(handedness='+1'), dict(degenerate_tol=float('nan')), dict(handedness_tol=-1)):
        with pytest.raises(ValueError):
            b1.parameterize_pose(**dict(POSE, **bad))
    with pytest.raises(ValueError, match='posed'):
        b1.check_rebuild()
    b2 = b1.parameterize_pose(**POSE)
    with pytest.raises(ValueError, match='bare'):
        b2.find_molecules(**FIND)
    for bad in (dict(max_dev=0), dict(mean_dev=-1), dict(match_elements=1)):
        with pytest.raises(ValueError):
            b2.check_rebuild(**bad)
    for bad in (dict(max_atomic_number=0), dict(heavy_nn_range=(3.0, 1.0)), dict(heavy_nn_range=1.0),
                dict(min_contact=0), dict(disorder='yes'), dict(cell_constraints=-1)):
        with pytest.raises(ValueError):
            b2.check_validity(**bad)
    with pytest.raises(ValueError, match='file_bonds'):
        _p1(_CUBE, _cart_to_frac(_HALF + 6), _HALF_Z).find_molecules(**FIND)


def test_store_round_trip_of_a_stepped_batch(tmp_path):
    posed = _posed_fixtures().check_rebuild().check_validity(max_atomic_number=100)
    assert _status(posed) == ['ok'] * 3
    posed.save(tmp_path / 'store', steps_run=posed.steps_run)
    back = MolCrystalData.load(tmp_path / 'store')
    for k in posed.keys():
        assert _same(posed[k], back[k]), k


# ======================================================================================
# against the old featurizer (CCDC)
# ======================================================================================

def test_matches_the_old_featurizer_on_fixtures():
    pytest.importorskip('ccdc')
    try:
        import mxtaltools.dataset_utils.construction.featurize_cif_chunks as fcc
    except Exception as e:                       # the licence is checked on import
        pytest.skip(f'CSD Python API unavailable: {e}')
    # every fixture the old featurizer accepts at Z' = 1 (the old one regenerates H: heavy atoms only)
    names = ('CEKZOH', 'DELDEG', 'HECVAP', 'IYOHIP', 'KICDUX', 'KUNTEV', 'MIHTEE52', 'NUKCOL', 'UGOMAF', 'XANCIA',
             'XEWCUY')
    import contextlib
    import io
    from mxtaltools.common.geometry_utils import rotvec2rotmat
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        old = [fcc.process_chunk([str(FIXTURES / f'{n}.cif')], 0, False, 'protonated', 6) for n in names]
    new = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in names], dtype=F64)
    new = new.find_molecules(**FIND).parameterize_pose(**POSE).check_rebuild()
    for i, rows in enumerate(old):
        assert len(rows) == 1 and _status(new)[i] == 'ok' and not bool(new.label_flip[i]), names[i]
        o = rows[0]
        sl = _row_sl(new, i)
        hn, ho = new.z[sl] > 1, o.z > 1
        # the heavy atoms keep the CIF's order in both: compare index by index
        assert torch.equal(new.z[sl][hn], o.z[ho]), names[i]
        dev = float((new.pos[sl][hn] - o.pos[ho].double()).norm(dim=1).max())
        # the old rows carry the float32 rounding of their own pose, which grows near rotations by pi
        # (old float32 arccos): XANCIA, |rotvec| 3.130, is 1.9e-4 A off; the new block is the exact image of
        # the CIF's atoms (test_fixture_chain_statuses_poses_and_exact_images)
        near_pi = abs(float(o.aunit_orientation[0, :3].double().norm()) - np.pi) < 0.02
        assert dev < (5e-4 if near_pi else 1e-4), (names[i], dev)
        dc = new.aunit_centroid[i, :3] - o.aunit_centroid[0, :3].double()
        assert float((new.T_fc[i] @ (dc - torch.round(dc))).norm()) < 1e-4, names[i]
        assert int(new.aunit_handedness[i, 0]) == int(o.aunit_handedness.reshape(-1)[0]), names[i]
        R1, R2 = (rotvec2rotmat(v[None].double())[0] for v in (new.aunit_orientation[i, :3],
                                                                 o.aunit_orientation[0, :3]))
        angle = float(torch.rad2deg(torch.arccos((((R1.T @ R2).trace() - 1) / 2).clamp(-1, 1))))
        assert angle < 1e-2, (names[i], angle)

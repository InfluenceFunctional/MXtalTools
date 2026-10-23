"""Contract tests for the chemistry full-chain steps (crystal_chem.py): sanitize and features.

CPU only. Two kinds of input:

- a constructed Z' = 2 ethanol crystal in P1, written as a CIF, with a FAKE CCDC backend
  (``crystal_chem._backends`` patched) that returns chosen components, hydrogens and bonds.
  It needs no licence and reaches every sanitize status and the hydrogen re-indexing of
  units 1+; its mol2 text is not mol2, so RDKit cannot read it and the CCDC bond types
  are kept (``features`` then rebuilds the molecule from them); a Z' = 1 borylferrocene crystal
  goes the same way for the batch-level fallback flag and note;
- single units as ``sanitize`` stores those RDKit cannot read (organometallic and boron units),
  fed to ``features``' row function: the fallback tiers, the per-problem reach, the bits and
  charges against an independent partial sanitization;
- a constructed Z' = 1 ethanol crystal in R-3 filed on rhombohedral axes, which ``read``
  re-expresses on hexagonal axes while CCDC reads the file's own axes: with a fake backend
  (coordinates on those axes, one atom a rhombohedral lattice vector away) and with the real
  API where it imports;
- the 16 CIF fixtures with the real CSD Python API, where it imports (skipped otherwise):
  the chain end to end, the hydrogen choices, the protonation refusal, the comparison with
  the old featurizer (SMILES, fingerprint, charges, hydrogens) and the store round trip.

Plus the step contract: new batch, input unchanged, pass-through rows bit-identical, one
status per failing row, isolation, ``done`` bits, ``steps_run``, argument refusals.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mxtaltools.dataset_utils.data_class_methods import crystal_chem as cc
from mxtaltools.dataset_utils.data_class_methods import crystal_steps
from mxtaltools.dataset_utils.data_class_methods.crystal_chem import (BOND_ORDERS, DONE_FEATURES, FEATURES_FALLBACK,
                                                                      LEWIS_ELEMENTS, OLD_FINGERPRINT, STATUS)
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import (DONE_FIND_MOLECULES, DONE_POSE, DONE_REBUILD,
                                                                       DONE_SANITIZE, DONE_VALIDITY)
from mxtaltools.dataset_utils.data_classes import MolCrystalData

FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
FIND = dict(bonds='file', bond_scale=(1.15, 1.35), iso_cap=5000, special_tol=0.1)
FIND_D = dict(FIND, bonds='distance')
POSE = dict(handedness='measured', degenerate_tol=1e-4, handedness_tol=1e-3)
REGEN = dict(chemistry='ccdc', hydrogens='regenerate', on_protonation_change='flag')
F64 = torch.float64


def _status(batch):
    return [STATUS[int(s)] for s in batch.status.tolist()]


def _row_sl(batch, i):
    return slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))


def _snapshot(batch):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in batch._store.items()}


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
    assert list(batch._store.keys()) == list(snap.keys())
    for k, v in snap.items():
        assert _same(batch._store[k], v), k


def _assert_row_passed_through(inp, out, i):
    """Row i of ``out`` equals row i of ``inp`` on every attribute ``inp`` holds (atoms by slice, rows by index)."""
    a, b = inp.select_rows([i]), out.select_rows([i])
    for k in a.keys():
        if k in ('steps_run',):
            continue
        assert _same(a[k], b[k]), k


def _have_ccdc():
    backend, why = cc._backends()
    return backend is not None


needs_ccdc = pytest.mark.skipif(not _have_ccdc(), reason='CSD Python API unavailable or unlicensed')


# ======================================================================================
# a constructed Z' = 2 crystal and a fake CCDC backend
# ======================================================================================

_A = 20.0                                       # cubic P1 cell edge (Angstrom)
_HEAVY = np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.03, 1.33, 0.0]])          # C1, C2, O1
_HEAVY_Z = [6, 6, 8]
_H = np.array([[-0.36, 1.03, 0.0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89],    # on C1
               [1.88, -0.51, 0.89], [1.88, -0.51, -0.89],                           # on C2
               [2.99, 1.30, 0.0]])                                                  # on O1
_H_PARENT = [0, 0, 0, 1, 1, 2]


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _units(far_first=False):
    """Two ethanol units (Cartesian): unit 0 placed as given, unit 1 rotated, moved and slightly bent (its own
    conformer). ``far_first``: unit 0 (listed first) far from the origin, so the canonical unit order
    (``canonicalize_zp_aunits``, by centroid norm) swaps the two."""
    R0, R1 = _rot([0.3, -0.5, 0.8], 37), _rot([1, 1, 0], 70)    # generic: no frame axis set by rounding
    h1 = _HEAVY.copy()
    h1[2] += [0.0, 0.0, 0.08]                   # unit 1's O moved: another conformer
    H1 = _H.copy()
    H1[5] += [0.0, 0.0, 0.08]
    off0, off1 = np.array([4.0, 4.0, 4.0]), np.array([13.0, 12.0, 11.0])
    if far_first:
        off0, off1 = off1, off0
    return (_HEAVY @ R0.T + off0, _H @ R0.T + off0), (h1 @ R1.T + off1, H1 @ R1.T + off1)


def _write_cif(path, name, with_oh=True, far_first=False, heavy_z=None):
    """The constructed crystal as a P1 CIF (no bond loop: find_molecules with bonds='distance'); ``heavy_z``
    replaces the heavy atoms' elements (C1, C2, O1)."""
    (h0, H0), (h1, H1) = _units(far_first)
    sym = {1: 'H', 5: 'B', 6: 'C', 8: 'O'}
    sites = []
    n_h = 0
    for u, (hv, hh) in enumerate(((h0, H0), (h1, H1))):
        for k, (z, p) in enumerate(zip(heavy_z or _HEAVY_Z, hv)):
            sites.append((f'{sym[z]}{k + 1}{"AB"[u]}', sym[z], p / _A))
        for k, p in enumerate(hh):
            if not with_oh and _H_PARENT[k] == 2:
                continue
            n_h += 1
            sites.append((f'H{n_h}', 'H', p / _A))
    lines = [f'data_{name}', f'_cell_length_a {_A}', f'_cell_length_b {_A}', f'_cell_length_c {_A}',
             '_cell_angle_alpha 90', '_cell_angle_beta 90', '_cell_angle_gamma 90',
             "_symmetry_space_group_name_H-M 'P 1'", '_symmetry_Int_Tables_number 1', '_cell_formula_units_Z 2',
             'loop_', '_symmetry_equiv_pos_as_xyz', 'x,y,z',
             'loop_', '_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y',
             '_atom_site_fract_z']
    lines += [f'{lab} {s} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}' for lab, s, f in sites]
    path.write_text('\n'.join(lines) + '\n')
    return path


class _FakeType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class _FakeAtom:
    def __init__(self, index, z, cart, label, charge=0, frac=None):
        self.index, self.atomic_number, self.label, self.formal_charge = index, z, label, charge
        self.coordinates = tuple(cart)
        self.fractional_coordinates = tuple(np.asarray(cart) / _A) if frac is None else tuple(frac)


class _FakeBond:
    def __init__(self, a, b, kind='Single'):
        self.atoms, self.bond_type = (a, b), _FakeType(kind)


class _FakeComponent:
    def __init__(self, atoms, bonds):
        self.atoms, self.bonds = atoms, bonds

    def to_string(self, fmt):
        return 'not a mol2 block\n'             # RDKit refuses: the CCDC bond types are kept


class _FakeCrystal:
    """``molecule.components`` built by ``build(mode)``; mode follows the hydrogen calls."""

    def __init__(self, identifier, build):
        self.identifier, self._build, self._mode = identifier, build, 'as_read'

    def assign_bonds(self):
        pass

    def add_hydrogens(self, mode='all', add_sites=True):
        self._mode = 'regenerate' if mode == 'all' else 'add_missing'

    def remove_hydrogens(self):
        self._mode = 'remove'

    @property
    def molecule(self):
        return type('FakeMolecule', (), {'components': self._build(self._mode)})()


class _FakeIO:
    def __init__(self, crystals):
        self._crystals = crystals

    def CrystalReader(self, path, format='cif'):
        crystal = self._crystals[Path(path).stem]
        return type('FakeReader', (), {'__len__': lambda s: 1, '__getitem__': lambda s, i: crystal,
                                       'close': lambda s: None})()


def _component(heavy, hyd, parents, labels_h, heavy_labels, h_order=None, extra_bonds=(), skip_h_bond=(),
               heavy_charges=(0, 0, 0), heavy_z=None, kinds=('Single', 'Single')):
    """A fake component: heavy atoms then hydrogens (in ``h_order``), bonds C1-C2 and C2-O1 (CCDC bond types
    ``kinds``), single bonds H-parent; ``heavy_z`` replaces the heavy atoms' elements."""
    atoms = [_FakeAtom(i, z, p, lab, q) for i, (z, p, lab, q) in
             enumerate(zip(heavy_z or _HEAVY_Z, heavy, heavy_labels, heavy_charges))]
    order = list(range(len(hyd))) if h_order is None else list(h_order)
    for k in order:
        atoms.append(_FakeAtom(len(atoms), 1, hyd[k], labels_h[k]))
    bonds = [_FakeBond(atoms[0], atoms[1], kinds[0]), _FakeBond(atoms[1], atoms[2], kinds[1])]
    for slot, k in enumerate(order):
        if k in skip_h_bond:
            continue
        bonds.append(_FakeBond(atoms[3 + slot], atoms[parents[k]]))
    for i, j in extra_bonds:
        bonds.append(_FakeBond(atoms[i], atoms[j]))
    return _FakeComponent(atoms, bonds)


def _builder(*, shift=0.03, drop_oh=(), shuffle_unit1=True, merge=False, heavy_offset=None, unbonded_h=False,
             as_read_moved=False, kept_h=(), heavy_charges=(0, 0, 0), far_first=False, heavy_z=None,
             kinds=('Single', 'Single')):
    """A ``build(mode)`` for the constructed crystal: regenerated hydrogens moved ``shift`` along their bond (unit
    1's listed in shuffled order), the O-H hydrogen left out of the units in ``drop_oh``, both units in one
    component (``merge``), a heavy atom moved (``heavy_offset``), a hydrogen without a bond (``unbonded_h``),
    unit 0's hydrogens ``kept_h`` left on their deposited sites by every hydrogen call (as CCDC keeps a hydrogen
    bonded to two atoms), formal charges on (C1, C2, O1) of both units (``heavy_charges``), other heavy-atom
    elements (``heavy_z``, as the CIF's) and CCDC bond types of C1-C2 and C2-O1 (``kinds``)."""
    (h0, H0), (h1, H1) = _units(far_first)

    def build(mode):
        comps = []
        for u, (hv, hh) in enumerate(((h0, H0), (h1, H1))):
            hv = hv.copy()
            if heavy_offset is not None and u == 0:
                hv[1] += heavy_offset
            keep = [k for k in range(6) if not (_H_PARENT[k] == 2 and u in drop_oh)]
            if mode == 'remove':
                keep = [k for k in kept_h if u == 0]
            if mode in ('regenerate', 'add_missing') or (mode == 'as_read' and as_read_moved):
                hyd = np.array([hh[k] + shift * (hh[k] - hv[_H_PARENT[k]]) / np.linalg.norm(hh[k] - hv[_H_PARENT[k]])
                                for k in range(6)])
            else:
                hyd = hh.copy()
            if mode == 'add_missing':
                hyd = hh.copy()                 # deposited ones kept; the O-H one (absent from the CIF) is new
                hyd[5] = hh[5] + 0.02
            if u == 0:
                for k in kept_h:
                    hyd[k] = hh[k]
            order = keep[::-1] if (u == 1 and shuffle_unit1) else keep
            labels_h = [f'H{u}{k}' for k in range(6)]
            comps.append(_component(hv, hyd, _H_PARENT, labels_h, [f'C1{"AB"[u]}', f'C2{"AB"[u]}', f'O1{"AB"[u]}'],
                                    h_order=order, skip_h_bond=(0,) if unbonded_h and u == 0 else (),
                                    heavy_charges=heavy_charges, heavy_z=heavy_z, kinds=kinds))
        if merge:
            a = comps[0].atoms + comps[1].atoms
            for i, atom in enumerate(a):
                atom.index = i
            comps = [_FakeComponent(a, comps[0].bonds + comps[1].bonds)]
        return comps

    return build


@pytest.fixture
def fake(monkeypatch):
    """Install a fake backend: ``fake({'ETOH': build, ...})``."""
    def install(crystals):
        io = _FakeIO({name: _FakeCrystal(name, build) for name, build in crystals.items()})
        monkeypatch.setattr(cc, '_backends', lambda: (io, ''))
        return io
    return install


def _etoh(tmp_path, name='ETOH', **kw):
    batch = MolCrystalData.read([_write_cif(tmp_path / f'{name}.cif', name, **kw)], dtype=F64)
    found = batch.find_molecules(**FIND_D)
    assert _status(found) == ['ok'] and int(found.z_prime[0]) == 2
    return found


# ======================================================================================
# sanitize with the fake backend
# ======================================================================================

def test_regenerate_reindexes_unit_hydrogens_and_keeps_heavy_atoms(tmp_path, fake):
    found = _etoh(tmp_path)
    fake({'ETOH': _builder()})
    snap = _snapshot(found)
    out = found.sanitize(**REGEN)
    _assert_unchanged(found, snap)
    assert out is not found and _status(out) == ['ok']
    assert int(out.done[0]) & DONE_SANITIZE and out.steps_run[-1] == dict(step='sanitize', args=REGEN)
    z, pos, mi = out.z, out.pos.double(), out.mol_ind
    n_u = len(z) // 2
    assert n_u == 9 and torch.equal(z[:n_u], z[n_u:]) and z[:3].tolist() == _HEAVY_Z and (z[3:n_u] == 1).all()
    assert mi.tolist() == [0] * 9 + [1] * 9
    # heavy atoms: exactly find_molecules' coordinates, in its order
    hin = found.z > 1
    assert torch.equal(out.pos[out.z > 1], found.pos[hin])
    # hydrogens: all placed by the backend, none deposited
    H = (z == 1)
    assert out.h_source[H].tolist() == [1] * 12 and out.h_source[~H].tolist() == [0] * 6
    assert (out.deposit_site[H] == -1).all() and (out.occupancy[H] == 1).all()
    assert len(out.removed_deposit_site[0]) == 12 and out.removed_deposit_frac[0].shape == (12, 3)
    assert sorted(out.removed_deposit_site[0].tolist()) == sorted(found.deposit_site[found.z == 1].tolist())
    # one atom order: unit 1's hydrogen j has unit 0's hydrogen j's parent and sits on its image after a fit
    bonds = out.mol_bonds[0]
    parent = {}
    for i, j in bonds.T.tolist():
        for h, p in ((i, j), (j, i)):
            if int(z[h]) == 1 and int(z[p]) > 1:
                parent[h] = p
    for j in range(3, n_u):
        assert parent[j] == parent[n_u + j] - n_u
    rms, apply = crystal_steps._kabsch(pos[n_u:n_u + 3].numpy(), pos[:3].numpy(), False)
    fitted = apply(pos[n_u + 3:].numpy())
    assert np.linalg.norm(fitted - pos[3:n_u].numpy(), axis=1).max() < 0.15
    # hydrogens at the backend's parent-to-hydrogen vector: 0.03 A beyond the deposited ones
    dep_H = found.pos[found.z == 1].double().numpy()
    d = np.linalg.norm(out.pos[H].double().numpy()[:, None] - dep_H[None], axis=-1).min(1)
    assert np.allclose(d, 0.03, atol=1e-6)
    # chemistry: CCDC's single bonds (RDKit could not read the fake mol2), charges 0, no SMILES
    assert bool(out.has_chemistry[0]) and out.unit_smiles[0] == [None, None]
    assert set(BOND_ORDERS[c] for c in out.mol_bond_order[0].tolist()) == {'single'}
    assert bonds.shape == (2, 2 * 8) and (bonds[0] < bonds[1]).all()
    assert (out.formal_charge == 0).all()
    assert 'RDKit cannot read unit(s) [0, 1]' in out.message[0]
    # protonation unchanged; per-unit properties recomputed with the new atoms
    assert bool(out.protonation_checked[0]) and not bool(out.protonation_changed[0])
    assert (out.h_count_change == 0).all()
    assert torch.allclose(out.unit_mass[0].double(), found.unit_mass[0].double(), atol=1e-4)
    assert torch.equal(out.unit_radius, found.unit_radius)          # heavy-atom radius: heavy atoms did not move
    assert int(out.num_atoms[0]) == 18 and out.ptr.tolist() == [0, 18]


def test_fake_chain_through_pose_rebuild_and_features(tmp_path, fake):
    found = _etoh(tmp_path)
    fake({'ETOH': _builder()})
    posed = found.sanitize(**REGEN).parameterize_pose(**POSE).check_rebuild()
    assert _status(posed) == ['ok'] and float(posed.rebuild_max_dev[0]) < 1e-5
    snap = _snapshot(posed)
    feat = posed.features(fingerprint=OLD_FINGERPRINT, charges='gasteiger')
    _assert_unchanged(posed, snap)
    assert _status(feat) == ['ok'] and int(feat.done[0]) & DONE_FEATURES
    # sanitize stored no SMILES (RDKit could not read the fake mol2): features fills them from the rebuilt molecule
    assert feat.unit_smiles[0] == ['[H]OC([H])([H])C([H])([H])[H]'] * 2
    assert feat.unit_fingerprint.shape == (1, 2, 256) and feat.unit_fingerprint.dtype == torch.uint8
    assert torch.equal(feat.unit_fingerprint[0, 0], feat.unit_fingerprint[0, 1])
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    want = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol)
    assert np.array_equal(np.unpackbits(feat.unit_fingerprint[0, 0].numpy()), want)
    x = feat.x
    assert torch.isfinite(x).all() and not bool(feat.charges_nonfinite[0])
    assert torch.allclose(x[:9], x[9:]) and abs(float(x[:9].sum())) < 1e-6
    assert feat.features_fallback.tolist() == [0] and 'features_fallback' not in feat.message[0]
    assert feat.steps_run[-1] == dict(step='features', args=dict(fingerprint=OLD_FINGERPRINT, charges='gasteiger'))
    with pytest.raises(ValueError, match='runs once'):
        feat.features(fingerprint=None, charges=None)


def test_protonation_change_is_flagged_or_refused(tmp_path, fake):
    found = _etoh(tmp_path)
    fake({'ETOH': _builder(drop_oh=(0, 1))})
    flagged = found.sanitize(**REGEN)
    assert _status(flagged) == ['ok'] and bool(flagged.protonation_changed[0])
    o = flagged.z == 8
    assert flagged.h_count_change[o].tolist() == [-1, -1] and (flagged.h_count_change[~o] == 0).all()
    assert 'protonation_changed' in flagged.message[0] and 'O3A -1' in flagged.message[0]
    assert int(flagged.num_atoms[0]) == 16
    refused = found.sanitize(**dict(REGEN, on_protonation_change='refuse'))
    assert _status(refused) == ['hydrogen_count_mismatch'] and int(refused.done[0]) & DONE_SANITIZE
    # the refused row keeps its atoms; the per-atom detail stays on its heavy atoms
    for k in ('z', 'pos', 'mol_ind', 'deposit_frac', 'deposit_site'):
        assert torch.equal(refused[k], found[k]), k
    assert refused.h_count_change[found.z == 8].tolist() == [-1, -1] and bool(refused.protonation_changed[0])
    assert refused.mol_bonds[0] is found.mol_bonds[0] or np.array_equal(refused.mol_bonds[0], found.mol_bonds[0])
    assert refused.unit_smiles[0] is None and refused.mol_bond_order[0] is None and not bool(refused.has_chemistry[0])


@pytest.mark.parametrize('build, code, text', [
    (dict(drop_oh=(1,)), 'hydrogens_failed', 'one atom order cannot hold both'),
    (dict(merge=True), 'component_mismatch', 'spans units'),
    (dict(heavy_offset=np.array([0.05, 0.0, 0.0])), 'chemistry_failed', 'cannot be mapped'),
    (dict(unbonded_h=True), 'hydrogens_failed', 'no heavy-atom neighbour'),
])
def test_sanitize_refusals(tmp_path, fake, build, code, text):
    found = _etoh(tmp_path)
    fake({'ETOH': _builder(**build)})
    snap = _snapshot(found)
    out = found.sanitize(**REGEN)
    _assert_unchanged(found, snap)
    assert _status(out) == [code] and text in out.message[0]
    assert int(out.done[0]) & DONE_SANITIZE
    for k in ('z', 'pos', 'mol_ind', 'deposit_frac'):
        assert torch.equal(out[k], found[k]), k


def test_as_read_and_add_missing_keep_deposited_hydrogens(tmp_path, fake):
    found = _etoh(tmp_path)
    fake({'ETOH': _builder()})
    out = found.sanitize(chemistry='ccdc', hydrogens='as_read', on_protonation_change='refuse')
    assert _status(out) == ['ok'] and (out.h_source == 0).all() and not bool(out.protonation_changed[0])
    # every deposited atom kept, at its exact coordinates (hydrogens moved behind the heavy atoms of their unit)
    order = out.deposit_site.tolist()
    assert sorted(order) == sorted(found.deposit_site.tolist()) and out.removed_deposit_site[0] is None
    by_site = {int(s): i for i, s in enumerate(found.deposit_site.tolist())}
    assert torch.equal(out.pos, found.pos[[by_site[s] for s in order]])
    # CCDC's hydrogens off the deposited sites, with as_read: refused
    fake({'ETOH': _builder(as_read_moved=True)})
    moved = found.sanitize(chemistry='ccdc', hydrogens='as_read', on_protonation_change='flag')
    assert _status(moved) == ['chemistry_failed'] and 'not deposited hydrogens' in moved.message[0]
    # a CIF without the O-H hydrogens: add_missing keeps the ten deposited ones and adds the two
    found2 = _etoh(tmp_path, 'ETOHNOH', with_oh=False)
    fake({'ETOHNOH': _builder()})
    added = found2.sanitize(chemistry='ccdc', hydrogens='add_missing', on_protonation_change='flag')
    assert _status(added) == ['ok'] and int(added.num_atoms[0]) == 18
    H = added.z == 1
    assert int((added.h_source[H] == 0).sum()) == 10 and int((added.h_source[H] == 1).sum()) == 2
    o = added.z == 8
    assert added.h_count_change[o].tolist() == [1, 1] and bool(added.protonation_changed[0])
    assert (added.deposit_site[added.h_source == 1] == -1).all()


def test_remove_with_ccdc_and_backend_unavailable(tmp_path, fake, monkeypatch):
    found = _etoh(tmp_path)
    fake({'ETOH': _builder()})
    out = found.sanitize(chemistry='ccdc', hydrogens='remove', on_protonation_change='refuse')
    assert _status(out) == ['ok'] and (out.z > 1).all() and int(out.num_atoms[0]) == 6
    assert bool(out.has_chemistry[0]) and not bool(out.protonation_checked[0])
    assert len(out.removed_deposit_site[0]) == 12
    monkeypatch.setattr(cc, '_backends', lambda: (None, 'no licence here'))
    gone = found.sanitize(**REGEN)
    assert _status(gone) == ['backend_unavailable'] and gone.message[0] == 'no licence here'
    assert int(gone.done[0]) & DONE_SANITIZE
    for k in ('z', 'pos', 'mol_ind', 'deposit_frac'):
        assert torch.equal(gone[k], found[k]), k


def test_a_hydrogen_the_backend_keeps_stays_deposited(tmp_path, fake):
    """CCDC's add_hydrogens(mode='all') and remove_hydrogens keep a hydrogen bonded to two atoms: on its deposited
    site it is that deposited atom in every mode, and with hydrogens='remove' it refuses the row."""
    found = _etoh(tmp_path)
    target = torch.as_tensor(_units()[0][1][5])                    # unit 0's O-H hydrogen, as deposited
    k = int((found.pos.double() - target).norm(dim=1).argmin())
    assert float((found.pos[k].double() - target).norm()) < 1e-5 and int(found.z[k]) == 1
    site = int(found.deposit_site[k])
    fake({'ETOH': _builder(kept_h=(5,))})
    out = found.sanitize(**REGEN)
    assert _status(out) == ['ok']
    at = torch.nonzero(out.deposit_site == site).flatten()
    assert len(at) == 1
    j = int(at[0])
    assert int(out.h_source[j]) == 0 and torch.equal(out.pos[j], found.pos[k])
    assert torch.equal(out.deposit_frac[j], found.deposit_frac[k]) and out.atom_labels[0][j] == found.atom_labels[0][k]
    H = out.z == 1
    assert int((out.h_source[H] == 1).sum()) == 11 and int(out.mol_ind[j]) == 0
    removed = out.removed_deposit_site[0].tolist()
    assert len(removed) == 11 and site not in removed and '11 deposited hydrogen(s)' in out.message[0]
    assert not bool(out.protonation_changed[0])
    # a hydrogen CCDC places near a deposited site (5e-3 A off) is not that atom: identity is the same coordinates
    fake({'ETOH': _builder(shift=0.005)})
    near = found.sanitize(**REGEN)
    H = near.z == 1
    assert (near.h_source[H] == 1).all() and (near.deposit_site[H] == -1).all()
    assert len(near.removed_deposit_site[0]) == 12
    # with 'remove' the hydrogen left behind refuses the row; its atoms stay as they were
    fake({'ETOH': _builder(kept_h=(5,))})
    gone = found.sanitize(chemistry='ccdc', hydrogens='remove', on_protonation_change='flag')
    assert _status(gone) == ['hydrogens_failed'] and "remove_hydrogens left 1 hydrogen(s), e.g. H05" in gone.message[0]
    for key in ('z', 'pos', 'mol_ind', 'deposit_site'):
        assert torch.equal(gone[key], found[key]), key


def test_charge_imbalance_is_flagged(tmp_path, fake):
    found = _etoh(tmp_path)
    fake({'ETOH': _builder()})
    neutral = found.sanitize(**REGEN)
    assert not bool(neutral.charge_imbalance[0]) and 'charge_imbalance' not in neutral.message[0]
    fake({'ETOH': _builder(heavy_charges=(1, 0, -1))})              # a zwitterion: neutral, not flagged
    zwitterion = found.sanitize(**REGEN)
    assert _status(zwitterion) == ['ok'] and not bool(zwitterion.charge_imbalance[0])
    assert zwitterion.formal_charge[zwitterion.z == 8].tolist() == [-1, -1]
    fake({'ETOH': _builder(heavy_charges=(0, 0, -1))})
    charged = found.sanitize(**REGEN)
    assert _status(charged) == ['ok'] and bool(charged.charge_imbalance[0])
    assert 'charge_imbalance' in charged.message[0] and 'net formal charges are [-1, -1]' in charged.message[0]
    # no flag without chemistry, nor on a refused row
    none = found.sanitize(chemistry='none', hydrogens='as_read', on_protonation_change='flag')
    assert not bool(none.charge_imbalance[0])
    fake({'ETOH': _builder(heavy_charges=(0, 0, -1), merge=True)})
    refused = found.sanitize(**REGEN)
    assert _status(refused) == ['component_mismatch'] and not bool(refused.charge_imbalance[0])


def _featurized_far_first(tmp_path, fake):
    """The constructed crystal with unit 0 far from the origin, through features, with sentinel per-unit chemistry
    (the two ethanol units' own fingerprints and SMILES coincide); and a copy after canonicalize_zp_aunits."""
    found = _etoh(tmp_path, 'ETOHFAR', far_first=True)
    fake({'ETOHFAR': _builder(far_first=True)})
    feat = found.sanitize(**REGEN).parameterize_pose(**POSE).features(fingerprint=OLD_FINGERPRINT, charges=None)
    assert _status(feat) == ['ok']
    feat.unit_fingerprint[0, 0] = 1
    feat.unit_fingerprint[0, 1] = 2
    feat.unit_smiles[0] = ['unit0', 'unit1']
    canon = feat.select_rows([0])
    canon.canonicalize_zp_aunits()
    return feat, canon


def test_canonicalize_moves_the_featurized_units(tmp_path, fake):
    feat, canon = _featurized_far_first(tmp_path, fake)
    n_u = int(feat.num_atoms[0]) // 2
    assert torch.equal(canon.pos[:n_u], feat.pos[n_u:]) and torch.equal(canon.pos[n_u:], feat.pos[:n_u])
    assert torch.equal(canon.h_source, feat.h_source) and torch.equal(canon.formal_charge, feat.formal_charge)


def test_canonicalize_moves_unit_chemistry_with_its_atoms(tmp_path, fake):
    feat, canon = _featurized_far_first(tmp_path, fake)
    assert canon.unit_smiles[0] == ['unit1', 'unit0']
    assert (canon.unit_fingerprint[0, 0] == 2).all() and (canon.unit_fingerprint[0, 1] == 1).all()


def test_rows_without_a_source_file_and_isolation(tmp_path, fake, monkeypatch):
    paths = [_write_cif(tmp_path / f'{n}.cif', n) for n in ('ETOHA', 'ETOHB', 'ETOHC')]
    text = paths[2].read_text()
    batch = MolCrystalData.read([paths[0], paths[1], text], dtype=F64)
    found = batch.find_molecules(**FIND_D)
    assert _status(found) == ['ok'] * 3 and found.source_path[2] is None
    fake({'ETOHA': _builder(), 'ETOHB': _builder()})
    real = cc._ccdc_row

    def flaky(backend, row, T_fc, path, block, *args):
        if block == 'ETOHA':
            raise RuntimeError('boom')
        return real(backend, row, T_fc, path, block, *args)

    monkeypatch.setattr(cc, '_ccdc_row', flaky)
    out = found.sanitize(**REGEN)
    assert _status(out) == ['chemistry_failed', 'ok', 'chemistry_failed']
    assert 'RuntimeError: boom' in out.message[0] and 'no source file' in out.message[2]
    for i in (0, 2):
        for k in ('z', 'pos', 'mol_ind'):
            assert torch.equal(out[k][_row_sl(out, i)], found[k][_row_sl(found, i)]), k
    assert (out.done & DONE_SANITIZE).bool().all()
    # a file gone since the read
    monkeypatch.setattr(cc, '_ccdc_row', real)
    paths[1].unlink()
    gone = found.select_rows([1]).sanitize(**REGEN)
    assert _status(gone) == ['chemistry_failed'] and 'not there any more' in gone.message[0]


def test_pass_through_rows_are_bit_identical(tmp_path, fake):
    good = _write_cif(tmp_path / 'ETOH.cif', 'ETOH')
    batch = MolCrystalData.read([FIXTURES / 'AJACOF.cif', good, FIXTURES / 'EFUNUP.cif'], dtype=F64)
    found = batch.find_molecules(**FIND_D)
    assert _status(found)[1] == 'ok' and _status(found)[0] != 'ok' and _status(found)[2] != 'ok'
    fake({'ETOH': _builder()})
    out = found.sanitize(**REGEN)
    for i in (0, 2):
        _assert_row_passed_through(found, out, i)
        assert int(out.done[i]) == int(found.done[i])
        assert int(out.h_source[_row_sl(out, i)].abs().sum()) == 0


# ======================================================================================
# features: the partial-sanitization fallback for units outside RDKit's valence model
# ======================================================================================

def _partial_fingerprint(batch, b, unit, ops):
    """Morgan bits (OLD_FINGERPRINT) of row ``b``'s unit ``unit``, built here from the stored chemistry (not by
    crystal_chem) and sanitized with ``ops`` after ``UpdatePropertyCache(strict=False)``; and whether RDKit's full
    sanitization refuses the same molecule."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    n_u = int(batch.num_atoms[b]) // int(batch.z_prime[b])
    s = int(batch.ptr[b]) + unit * n_u
    kind = {'single': Chem.BondType.SINGLE, 'double': Chem.BondType.DOUBLE}
    m = Chem.RWMol()
    for zz in batch.z[s:s + n_u].tolist():
        atom = Chem.Atom(int(zz))
        atom.SetNoImplicit(True)
        m.AddAtom(atom)
    lo = unit * n_u
    for (i, j), c in zip(np.asarray(batch.mol_bonds[b]).reshape(2, -1).T.tolist(), batch.mol_bond_order[b].tolist()):
        if lo <= i < lo + n_u and lo <= j < lo + n_u:
            m.AddBond(i - lo, j - lo, kind[BOND_ORDERS[c]])
    strict_refused = False
    try:
        Chem.SanitizeMol(Chem.Mol(m.GetMol()))
    except Exception:
        strict_refused = True
    mol = m.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, ops)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    return gen.GetFingerprintAsNumPy(mol), strict_refused


def test_features_fallback_for_a_unit_outside_the_valence_model(tmp_path, fake):
    """A unit holding boron whose stored chemistry RDKit's valence check refuses (C2=B: C2 carries five bonds):
    featurized with every sanitization step but the valence check, flagged 'valence' with the reason, no SMILES.
    The row beside it, sanitized in full, is bit-identical to that row featurized alone. Before the fallback the
    boron row was features_failed."""
    assert FEATURES_FALLBACK == ('none', 'valence', 'kekule') and 5 not in LEWIS_ELEMENTS
    assert LEWIS_ELEMENTS == frozenset((1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53))
    paths = [_write_cif(tmp_path / 'ETOH.cif', 'ETOH'), _write_cif(tmp_path / 'ETBH.cif', 'ETBH', heavy_z=[6, 6, 5])]
    found = MolCrystalData.read(paths, dtype=F64).find_molecules(**FIND_D)
    assert _status(found) == ['ok', 'ok'] and found.z_prime.tolist() == [2, 2]
    fake({'ETOH': _builder(), 'ETBH': _builder(heavy_z=[6, 6, 5], kinds=('Single', 'Double'))})
    posed = found.sanitize(**REGEN).parameterize_pose(**POSE).check_rebuild()
    assert _status(posed) == ['ok', 'ok'] and posed.unit_smiles[1] == [None, None]
    assert 'no SMILES' in posed.message[1] and (posed.z[_row_sl(posed, 1)] == 5).sum() == 2
    snap = _snapshot(posed)
    feat = posed.features(fingerprint=OLD_FINGERPRINT, charges='gasteiger')
    _assert_unchanged(posed, snap)
    assert _status(feat) == ['ok', 'ok'] and (feat.done & DONE_FEATURES).bool().all()
    assert feat.features_fallback.dtype == torch.int8
    assert feat.features_fallback.tolist() == [0, FEATURES_FALLBACK.index('valence')]
    msg = feat.message[1]
    assert "features_fallback: unit 0 'valence', unit 1 'valence'" in msg and 'AtomValenceException' in msg
    assert 'no SMILES for them' in msg and feat.unit_smiles[1] == [None, None]
    assert msg.startswith(posed.message[1])                     # sanitize's note kept, the fallback's appended
    # the bits are RDKit's partial sanitization of the stored chemistry, which its full sanitization refuses
    from rdkit import Chem
    S = Chem.SanitizeFlags
    want, strict_refused = _partial_fingerprint(feat, 1, 0, S.SANITIZE_ALL ^ S.SANITIZE_PROPERTIES)
    assert strict_refused and want.sum() > 0
    assert np.array_equal(np.unpackbits(feat.unit_fingerprint[1, 0].numpy()), want)
    assert torch.equal(feat.unit_fingerprint[1, 0], feat.unit_fingerprint[1, 1])
    sl = _row_sl(feat, 1)
    assert bool(feat.charges_nonfinite[1]) == bool((~torch.isfinite(feat.x[sl])).any())
    # the row sanitized in full: as featurized alone (the fallback never touches it)
    alone = posed.select_rows([0]).features(fingerprint=OLD_FINGERPRINT, charges='gasteiger')
    first = feat.select_rows([0])
    for k in ('status', 'message', 'unit_smiles', 'unit_fingerprint', 'x', 'charges_nonfinite', 'features_fallback'):
        assert _same(first[k], alone[k]), k


def test_features_refuses_a_lewis_unit_whose_stored_chemistry_does_not_sanitize(tmp_path, fake):
    """Every atom in LEWIS_ELEMENTS: stored chemistry RDKit refuses (C2=O, C2 with five bonds) is no Lewis
    structure, so the row stays features_failed with the reason, and nothing is written for it."""
    found = _etoh(tmp_path)
    fake({'ETOH': _builder(kinds=('Single', 'Double'))})
    posed = found.sanitize(**REGEN).parameterize_pose(**POSE)
    assert _status(posed) == ['ok'] and posed.unit_smiles[0] == [None, None]
    feat = posed.features(fingerprint=OLD_FINGERPRINT, charges='gasteiger')
    assert _status(feat) == ['features_failed'] and int(feat.done[0]) & DONE_FEATURES
    msg = feat.message[0]
    assert msg.startswith('unit 0: RDKit cannot rebuild the stored chemistry: AtomValenceException') and \
        'no fallback' in msg
    assert feat.features_fallback.tolist() == [0] and torch.equal(feat.x, posed.x)
    assert not feat.unit_fingerprint.any() and feat.unit_smiles[0] == [None, None]


def _stored(atoms, bonds):
    """``(z, formal charges (0), pos, bonds [2, E] with i < j, codes)`` from ``[(Z, xyz)]`` and ``[(i, j, order)]``
    as ``sanitize`` stores a unit RDKit cannot read (CCDC's bond types, charges 0)."""
    z = np.array([a[0] for a in atoms], dtype=np.int64)
    pos = np.array([a[1] for a in atoms], dtype=np.float64)
    e = np.array([(min(i, j), max(i, j)) for i, j, _ in bonds], dtype=np.int64).T.copy()
    codes = np.array([BOND_ORDERS.index(k) for _, _, k in bonds], dtype=np.int8)
    return z, np.zeros(len(z), dtype=np.int64), pos, e, codes


def _cyclopentadienyls(metal=True):
    """Ferrocene as CCDC types it: two aromatic C5H5 rings, each carbon bound to Fe by a pi bond; without the
    metal, one C5H5 ring alone (aromatic, no charge)."""
    atoms, bonds = ([(26, (0.0, 0.0, 0.0))], []) if metal else ([], [])
    for zc, phase in ((1.65, 0.0), (-1.65, np.pi / 5)) if metal else ((0.0, 0.0),):
        c0 = len(atoms)
        angles = [phase + 2 * np.pi * k / 5 for k in range(5)]
        atoms += [(6, (1.21 * np.cos(t), 1.21 * np.sin(t), zc)) for t in angles]
        atoms += [(1, (2.28 * np.cos(t), 2.28 * np.sin(t), zc)) for t in angles]
        for k in range(5):
            bonds += [(c0 + k, c0 + (k + 1) % 5, 'aromatic'), (c0 + k, c0 + 5 + k, 'single')]
            if metal:
                bonds.append((0, c0 + k, 'pi'))
    return atoms, bonds


def _iron_pentacarbonyl():
    """Fe(CO)5 as CCDC types it: Fe-C single, C#O triple, no charges (each O carries three bonds)."""
    atoms, bonds = [(26, (0.0, 0.0, 0.0))], []
    for d in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-0.5, 0.866, 0), (-0.5, -0.866, 0)):
        d = np.asarray(d, dtype=float) / np.linalg.norm(d)
        c = len(atoms)
        atoms += [(6, tuple(1.80 * d)), (8, tuple(2.95 * d))]
        bonds += [(0, c, 'single'), (c, c + 1, 'triple')]
    return atoms, bonds


def _aromatic_chain():
    """Fe-CH2-CH3 with the C-C bond typed aromatic: an aromatic bond in no ring."""
    atoms = [(26, (0.0, 0.0, 0.0)), (6, (2.0, 0.0, 0.0)), (6, (3.5, 0.0, 0.0)), (1, (2.3, 1.0, 0.0)),
             (1, (2.3, -1.0, 0.0)), (1, (3.9, 1.0, 0.0)), (1, (3.9, -0.5, 0.9)), (1, (3.9, -0.5, -0.9))]
    bonds = [(0, 1, 'single'), (1, 2, 'aromatic'), (1, 3, 'single'), (1, 4, 'single'), (2, 5, 'single'),
             (2, 6, 'single'), (2, 7, 'single')]
    return atoms, bonds


def test_features_fallback_tiers_on_organometallic_chemistry():
    """Units as sanitize stores them when RDKit cannot read CCDC's perception. Ferrocene (aromatic Cp rings that
    have no Kekule form without charges): 'kekule', the stored aromatic flags kept; Fe(CO)5 (O with three bonds):
    'valence'. Both get bits, no SMILES, and charges flagged non-finite (Gasteiger has no iron). Refused: a C5H5
    ring alone (every atom in LEWIS_ELEMENTS) and an aromatic bond in no ring (even beside a metal)."""
    gen = cc._morgan(OLD_FINGERPRINT)

    def run(atoms, bonds):
        z, q, pos, e, codes = _stored(atoms, bonds)
        return cc._features_row(z, q, pos, e, codes, 1, None, gen, 'gasteiger')

    for (atoms, bonds), tier, error in ((_cyclopentadienyls(), 'kekule', 'KekulizeException'),
                                        (_iron_pentacarbonyl(), 'valence', 'AtomValenceException')):
        z, q, pos, e, codes = _stored(atoms, bonds)
        with pytest.raises(Exception, match='kekulize|valence'):
            cc._rdkit_molecule(z, q, e, codes, pos)             # RDKit's full sanitization refuses it
        res = run(atoms, bonds)
        assert res['fallback'] == {0: FEATURES_FALLBACK.index(tier)} and error in res['fallback_reason']
        assert res['smiles'] == [None] and np.unpackbits(res['bits'][0]).sum() > 0
        assert res['nonfinite'] and len(res['charges']) == len(atoms)
    mol = cc._rdkit_molecule(*[_stored(*_cyclopentadienyls())[i] for i in (0, 1, 3, 4, 2)], fallback=2)
    assert sum(b.GetIsAromatic() for b in mol.GetBonds()) == 10      # the ten ring bonds stay aromatic
    with pytest.raises(cc._Refusal, match='no fallback') as refused:
        run(*_cyclopentadienyls(metal=False))
    assert refused.value.code == 'features_failed' and 'KekulizeException' in str(refused.value)
    with pytest.raises(cc._Refusal, match='aromatic bond\\(s\\) are in no ring') as refused:
        run(*_aromatic_chain())
    assert refused.value.code == 'features_failed'


def _borylferrocene():
    """Ferrocene with ring A's first hydrogen replaced by BH3 (boron with four bonds, no charge): RDKit's full
    sanitization refuses the boron's valence, the sanitization without the valence check the Cp rings' Kekule form;
    both problems lie at the boron or on rings bound to the iron."""
    atoms, bonds = _cyclopentadienyls()
    assert atoms[6][0] == 1 and (1, 6, 'single') in bonds          # ring A's first hydrogen, on carbon 1
    b = np.array([2.79, 0.0, 1.65])
    atoms[6] = (5, tuple(b))
    for d in ((0.333, 0.943, 0.0), (0.333, -0.471, 0.816), (0.333, -0.471, -0.816)):
        atoms.append((1, tuple(b + 1.19 * np.asarray(d))))
        bonds.append((6, len(atoms) - 1, 'single'))
    return atoms, bonds


def _graph(elements, bonds):
    """``(atoms, bonds)`` for ``_stored`` from elements and bonds, at fixed generic coordinates (the features of
    these units do not depend on them)."""
    pos = np.random.default_rng(len(elements)).normal(size=(len(elements), 3)) * 2.0
    return [(z, tuple(p)) for z, p in zip(elements, pos)], bonds


def _ring(c0, kind='aromatic'):
    """The five bonds of a C5 ring on atoms c0..c0 + 4."""
    return [(c0 + k, c0 + (k + 1) % 5, kind) for k in range(5)]


def _cp_boryl(spacers=0):
    """A C5H4 ring typed aromatic (no Kekule form) carrying BH2, through ``spacers`` CH2 groups: the ring is
    bonded to the boron (0) or ``spacers + 1`` bonds from it."""
    el, bonds = [6] * 5, _ring(0)
    for k in range(1, 5):
        el.append(1)
        bonds.append((k, len(el) - 1, 'single'))
    prev = 0
    for _ in range(spacers):
        el.append(6)
        c = len(el) - 1
        bonds.append((prev, c, 'single'))
        for _ in range(2):
            el.append(1)
            bonds.append((c, len(el) - 1, 'single'))
        prev = c
    el.append(5)
    bo = len(el) - 1
    bonds.append((prev, bo, 'single'))
    for _ in range(2):
        el.append(1)
        bonds.append((bo, len(el) - 1, 'single'))
    return _graph(el, bonds)


def _boryl_ammonium(spacers):
    """H2B-(CH2)k-N(CH3)3 with no charge (N with four bonds): the over-valent nitrogen ``spacers + 1`` bonds from
    the boron."""
    el, bonds = [5, 1, 1], [(0, 1, 'single'), (0, 2, 'single')]
    prev = 0
    for _ in range(spacers):
        el += [6, 1, 1]
        c = len(el) - 3
        bonds += [(prev, c, 'single'), (c, c + 1, 'single'), (c, c + 2, 'single')]
        prev = c
    el.append(7)
    n = len(el) - 1
    bonds.append((prev, n, 'single'))
    for _ in range(3):
        el += [6, 1, 1, 1]
        c = len(el) - 4
        bonds += [(n, c, 'single'), (c, c + 1, 'single'), (c, c + 2, 'single'), (c, c + 3, 'single')]
    return _graph(el, bonds)


def _ferrocene_with_remote_ring():
    """Ferrocene whose ring A carries -CH2-CH2- to a C5H4 ring typed aromatic (no Kekule form) bound to nothing
    else: three unkekulizable systems, the two bound to the iron listed first, the third 4 bonds from it."""
    atoms, bonds = _cyclopentadienyls()
    el = [a[0] for a in atoms]
    assert el[6] == 1 and (1, 6, 'single') in bonds
    el[6] = 6                                       # ring A's first hydrogen becomes CH2
    ca = 6
    el += [1, 1, 6, 1, 1]
    cb = len(el) - 3
    bonds += [(ca, len(el) - 5, 'single'), (ca, len(el) - 4, 'single'), (ca, cb, 'single'),
              (cb, cb + 1, 'single'), (cb, cb + 2, 'single')]
    r0 = len(el)
    el += [6] * 5
    bonds += _ring(r0) + [(cb, r0, 'single')]
    for k in range(1, 5):
        el.append(1)
        bonds.append((r0 + k, len(el) - 1, 'single'))
    return _graph(el, bonds)


@pytest.mark.parametrize('unit, outcome, text', [
    (_cp_boryl(0), 'kekule', None),
    (_cp_boryl(1), 'refused', 'the aromatic system of atoms [0, 1, 2, 3, 4] has no Kekule form and is 2 bonds'),
    (_cp_boryl(2), 'refused', 'the aromatic system of atoms [0, 1, 2, 3, 4] has no Kekule form and is 3 bonds'),
    (_boryl_ammonium(0), 'valence', None),
    (_boryl_ammonium(1), 'valence', None),
    (_boryl_ammonium(2), 'refused', 'AtomValenceException at atom 9 (N), 3 bonds'),
    (_ferrocene_with_remote_ring(), 'refused', 'has no Kekule form and is 4 bonds'),
], ids=['ring_on_boron', 'ring_2_bonds_from_boron', 'ring_3_bonds_from_boron', 'n_on_boron', 'n_2_bonds_from_boron',
        'n_3_bonds_from_boron', 'third_ring_4_bonds'])
def test_features_fallback_only_for_problems_near_an_element_outside_the_valence_model(unit, outcome, text):
    """The fallback is decided per problem RDKit names, not per unit: an over-valent atom more than 2 bonds from
    every atom outside LEWIS_ELEMENTS, or an unkekulizable aromatic system not bonded to one, is the stored
    chemistry's own defect (as in a unit of LEWIS_ELEMENTS alone), so the unit is refused even though it holds
    boron or iron; the same defect on or beside the boron gets the fallback. RDKit names one unkekulizable system
    per pass: a remote ring listed after two rings bound to the iron is still found."""
    z, q, pos, e, codes = _stored(*unit)
    with pytest.raises(Exception):
        cc._rdkit_molecule(z, q, e, codes, pos)                 # RDKit's full sanitization refuses every one
    gen = cc._morgan(OLD_FINGERPRINT)
    if outcome == 'refused':
        with pytest.raises(cc._Refusal) as refused:
            cc._features_row(z, q, pos, e, codes, 1, None, gen, 'gasteiger')
        msg = str(refused.value)
        assert refused.value.code == 'features_failed' and 'no fallback: ' in msg and text in msg, msg
        assert 'outside LEWIS_ELEMENTS' in msg
    else:
        res = cc._features_row(z, q, pos, e, codes, 1, None, gen, 'gasteiger')
        assert res['fallback'] == {0: FEATURES_FALLBACK.index(outcome)}


def _independent_features(atoms, bonds, ops):
    """Morgan bits (OLD_FINGERPRINT) and Gasteiger charges of a unit built here from ``[(Z, xyz)]`` and
    ``[(i, j, kind)]`` (not by crystal_chem): every hydrogen explicit, aromatic bonds and their atoms flagged, a
    'pi' bond unspecified; ``UpdatePropertyCache(strict=False)``, then ``SanitizeMol(mol, ops)``."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    kind = {'single': Chem.BondType.SINGLE, 'double': Chem.BondType.DOUBLE, 'triple': Chem.BondType.TRIPLE,
            'aromatic': Chem.BondType.AROMATIC, 'pi': Chem.BondType.UNSPECIFIED}
    m = Chem.RWMol()
    for zz, _ in atoms:
        atom = Chem.Atom(int(zz))
        atom.SetNoImplicit(True)
        m.AddAtom(atom)
    for i, j, k in bonds:
        m.AddBond(i, j, kind[k])
        if k == 'aromatic':
            m.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
            m.GetAtomWithIdx(i).SetIsAromatic(True)
            m.GetAtomWithIdx(j).SetIsAromatic(True)
    mol = m.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol, ops)
    bits = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False) \
        .GetFingerprintAsNumPy(mol)
    AllChem.ComputeGasteigerCharges(mol)
    return bits, np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])


@pytest.mark.parametrize('unit, finite', [(_cyclopentadienyls(), False), (_cp_boryl(0), True)],
                         ids=['ferrocene', 'boryl_cp'])
def test_kekule_fallback_features_equal_an_independent_partial_sanitization(unit, finite):
    """Bits and charges of a 'kekule' unit equal those of the unit built here and sanitized with the documented
    steps: everything but the valence check, kekulization and aromaticity perception. The boryl ring has no metal,
    so its Gasteiger charges are finite and the comparison covers the hybridization step they depend on."""
    from rdkit import Chem
    S = Chem.SanitizeFlags
    want_bits, want_q = _independent_features(*unit, S.SANITIZE_ALL ^ S.SANITIZE_PROPERTIES ^ S.SANITIZE_KEKULIZE ^
                                              S.SANITIZE_SETAROMATICITY)
    z, q, pos, e, codes = _stored(*unit)
    res = cc._features_row(z, q, pos, e, codes, 1, None, cc._morgan(OLD_FINGERPRINT), 'gasteiger')
    assert res['fallback'] == {0: FEATURES_FALLBACK.index('kekule')}
    assert want_bits.sum() > 0 and np.array_equal(np.unpackbits(res['bits'][0]), want_bits)
    assert bool(np.isfinite(want_q).all()) == finite and res['nonfinite'] == (not finite)
    assert np.array_equal(np.isnan(res['charges']), np.isnan(want_q))
    assert np.array_equal(res['charges'][~np.isnan(want_q)], want_q[~np.isnan(want_q)])


_SYMBOL = {1: 'H', 5: 'B', 6: 'C', 8: 'O', 26: 'Fe'}


def _write_unit_cif(path, name, atoms, offset):
    """One molecule (``[(Z, xyz)]``, Cartesian, Angstrom) moved by ``offset`` into the cubic P1 cell, as a CIF."""
    lines = [f'data_{name}', f'_cell_length_a {_A}', f'_cell_length_b {_A}', f'_cell_length_c {_A}',
             '_cell_angle_alpha 90', '_cell_angle_beta 90', '_cell_angle_gamma 90',
             "_symmetry_space_group_name_H-M 'P 1'", '_symmetry_Int_Tables_number 1', '_cell_formula_units_Z 1',
             'loop_', '_symmetry_equiv_pos_as_xyz', 'x,y,z',
             'loop_', '_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y',
             '_atom_site_fract_z']
    for k, (z, p) in enumerate(atoms):
        f = (np.asarray(p) + offset) / _A
        lines.append(f'{_SYMBOL[z]}{k} {_SYMBOL[z]} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}')
    path.write_text('\n'.join(lines) + '\n')
    return path


def _unit_builder(atoms, bonds, offset):
    """A ``build(mode)`` returning the molecule as one CCDC component, every atom (hydrogens included) on its CIF
    coordinates in every mode, with CCDC's bond types for ``bonds``."""
    kinds = {'single': 'Single', 'double': 'Double', 'triple': 'Triple', 'aromatic': 'Aromatic', 'pi': 'Pi'}

    def build(mode):
        fa = [_FakeAtom(i, z, np.asarray(p) + offset, f'{_SYMBOL[z]}{i}') for i, (z, p) in enumerate(atoms)]
        return [_FakeComponent(fa, [_FakeBond(fa[i], fa[j], kinds[k]) for i, j, k in bonds])]

    return build


def test_features_writes_the_kekule_flag_and_both_refusals(tmp_path, fake):
    """Through the batch method: a borylferrocene crystal (Z' = 1) beside the ethanol one. Its unit needs the
    'kekule' fallback, so features_fallback holds 2 (the furthest tier), and the note names RDKit's refusal of the
    full sanitization (the boron's valence) and of the sanitization without the valence check (the Cp rings'
    Kekule form): the second is why the stored aromatic flags were kept."""
    atoms, bonds = _borylferrocene()
    offset = np.array([7.0, 8.0, 9.0])
    paths = [_write_cif(tmp_path / 'ETOH.cif', 'ETOH'),
             _write_unit_cif(tmp_path / 'BFC.cif', 'BFC', atoms, offset)]
    found = MolCrystalData.read(paths, dtype=F64).find_molecules(**FIND_D)
    assert _status(found) == ['ok', 'ok'] and found.z_prime.tolist() == [2, 1]
    assert int(found.num_atoms[1]) == len(atoms)
    fake({'ETOH': _builder(), 'BFC': _unit_builder(atoms, bonds, offset)})
    san = found.sanitize(chemistry='ccdc', hydrogens='as_read', on_protonation_change='flag')
    assert _status(san) == ['ok', 'ok'] and san.unit_smiles[1] == [None]
    posed = san.parameterize_pose(**POSE)
    assert _status(posed) == ['ok', 'ok']
    feat = posed.features(fingerprint=OLD_FINGERPRINT, charges='gasteiger')
    assert _status(feat) == ['ok', 'ok'] and feat.features_fallback.dtype == torch.int8
    assert feat.features_fallback.tolist() == [0, FEATURES_FALLBACK.index('kekule')]
    msg = feat.message[1]
    full, _, tier1 = msg.partition('; and the sanitization of unit 0 without the valence check: ')
    assert "features_fallback: unit 0 'kekule' (RDKit refuses the full sanitization of unit 0: " \
           "AtomValenceException: Explicit valence for atom" in full and ' B, 4, ' in full
    assert tier1.startswith("KekulizeException: Can't kekulize mol.  Unkekulized atoms: ") and \
        '); no SMILES for them' in tier1
    assert feat.unit_smiles[1] == [None] and np.unpackbits(feat.unit_fingerprint[1, 0].numpy()).sum() > 0
    assert bool(feat.charges_nonfinite[1]) and 'features_fallback' not in feat.message[0]


# ======================================================================================
# chemistry='none'
# ======================================================================================

def test_none_remove_and_as_read(tmp_path):
    found = _etoh(tmp_path)
    snap = _snapshot(found)
    out = found.sanitize(chemistry='none', hydrogens='remove', on_protonation_change='flag')
    _assert_unchanged(found, snap)
    assert _status(out) == ['ok'] and (out.z > 1).all() and int(out.num_atoms[0]) == 6
    assert torch.equal(out.pos, found.pos[found.z > 1]) and torch.equal(out.mol_ind, found.mol_ind[found.z > 1])
    assert out.mol_bonds[0].shape == (2, 4) and int(out.mol_bonds[0].max()) < 6
    assert len(out.removed_deposit_site[0]) == 12 and not bool(out.has_chemistry[0])
    assert 'formal_charge' not in out.keys() and 'unit_smiles' not in out.keys() and 'mol_bond_order' not in out.keys()
    assert float(out.unit_mass[0, 0]) < float(found.unit_mass[0, 0])
    assert out.atom_labels[0] == [lab for lab, z in zip(found.atom_labels[0], found.z.tolist()) if z > 1]
    kept = found.sanitize(chemistry='none', hydrogens='as_read', on_protonation_change='flag')
    for k in found.keys():
        if k not in ('done', 'steps_run', 'status', 'message'):
            assert _same(kept[k], found[k]), k
    assert (kept.h_source == 0).all() and not bool(kept.has_chemistry[0])
    # no chemistry: the pose runs, features refuses the row
    posed = kept.parameterize_pose(**POSE)
    feat = posed.features(fingerprint=OLD_FINGERPRINT, charges='gasteiger')
    assert _status(feat) == ['features_failed'] and 'no chemistry' in feat.message[0]
    assert int(feat.done[0]) & DONE_FEATURES and torch.equal(feat.x, posed.x)


# ======================================================================================
# argument and batch refusals
# ======================================================================================

def test_argument_refusals(tmp_path):
    found = _etoh(tmp_path)
    for bad in (dict(REGEN, chemistry='rdkit'), dict(REGEN, chemistry='template'), dict(REGEN, hydrogens='all'),
                dict(REGEN, on_protonation_change='ignore'), dict(REGEN, chemistry='none'),
                dict(REGEN, chemistry='none', hydrogens='add_missing')):
        with pytest.raises(ValueError):
            found.sanitize(**bad)
    with pytest.raises(TypeError):
        found.sanitize(chemistry='none', hydrogens='remove')
    bare = MolCrystalData.read([_write_cif(tmp_path / 'X.cif', 'X')], dtype=F64)
    with pytest.raises(ValueError, match='find_molecules'):
        bare.sanitize(chemistry='none', hydrogens='remove', on_protonation_change='flag')
    once = found.sanitize(chemistry='none', hydrogens='remove', on_protonation_change='flag')
    with pytest.raises(ValueError, match='runs once'):
        once.sanitize(chemistry='none', hydrogens='remove', on_protonation_change='flag')
    posed = found.parameterize_pose(**POSE)
    with pytest.raises(ValueError, match='before parameterize_pose'):
        posed.sanitize(chemistry='none', hydrogens='remove', on_protonation_change='flag')
    with pytest.raises(ValueError, match='posed batch'):
        found.features(fingerprint=None, charges=None)
    for fp in (dict(radius=2, n_bits=2048), dict(OLD_FINGERPRINT, n_bits=100), dict(OLD_FINGERPRINT, radius=-1),
               dict(OLD_FINGERPRINT, chirality=1), dict(OLD_FINGERPRINT, n_bits=True), [2, 2048, False]):
        with pytest.raises(ValueError):
            posed.features(fingerprint=fp, charges=None)
    with pytest.raises(ValueError):
        posed.features(fingerprint=None, charges='mmff')
    with pytest.raises(ValueError):
        cc.MolCrystalChem.sanitize(object(), **REGEN)                       # not a batch
    # a batch from from_arrays has no source file for CCDC
    cell = torch.tensor([[_A, _A, _A, np.pi / 2, np.pi / 2, np.pi / 2]], dtype=F64)
    arr = MolCrystalData.from_arrays(cell, found.deposit_frac.double() * _A, found.z, ptr=[0, len(found.z)], sg=1,
                                     setting='standard', dtype=F64).find_molecules(**FIND_D)
    with pytest.raises(ValueError, match='source file'):
        arr.sanitize(**REGEN)


def test_status_table_extends_the_steps_table():
    assert STATUS[:len(crystal_steps.STATUS)] == crystal_steps.STATUS
    assert len(set(STATUS)) == len(STATUS) and DONE_FEATURES == 64
    assert OLD_FINGERPRINT == dict(radius=2, n_bits=2048, chirality=False)
    from mxtaltools.dataset_utils.data_class_methods.crystal_store import status_table
    assert status_table()[:len(STATUS)] == STATUS


def test_pose_runs_after_a_sanitize_refusal(tmp_path, fake):
    found = _etoh(tmp_path)
    fake({'ETOH': _builder(merge=True)})
    out = found.sanitize(**REGEN)
    assert _status(out) == ['component_mismatch']
    out.parameterize_pose(**POSE)


# ======================================================================================
# an R group filed on rhombohedral axes: the row is on hexagonal axes, CCDC on the file's
# ======================================================================================

_RH_A, _RH_ALPHA = 9.0, 75.0                    # the file's rhombohedral cell (Angstrom, degrees)
#: R-3 on rhombohedral axes, 6 operators; ``read`` re-expresses the file on hexagonal axes (SYM_OPS[148], 18)
_RH_OPS = ('x,y,z', 'z,x,y', 'y,z,x', '-x,-y,-z', '-z,-x,-y', '-y,-z,-x')
_RH_CENTROID = np.array([0.13, 0.31, 0.62])     # the molecule's centroid, rhombohedral fractional
_ETOH_SMILES = '[H]OC([H])([H])C([H])([H])[H]'


def _rh_lattice():
    """The rhombohedral lattice vectors as rows (a along x, b in the xy plane), Angstrom."""
    a, t = _RH_A, np.radians(_RH_ALPHA)
    cy = a * (np.cos(t) - np.cos(t) ** 2) / np.sin(t)
    return np.array([[a, 0, 0], [a * np.cos(t), a * np.sin(t), 0],
                     [a * np.cos(t), cy, np.sqrt(a * a - (a * np.cos(t)) ** 2 - cy * cy)]])


def _rh_molecule():
    """(heavy [3, 3], hydrogens [6, 3]): ethanol, Cartesian on ``_rh_lattice``'s axes, centroid at ``_RH_CENTROID``."""
    R = _rot([0.3, -0.5, 0.8], 37)
    heavy, hyd = _HEAVY @ R.T, _H @ R.T
    shift = _RH_CENTROID @ _rh_lattice() - np.vstack([heavy, hyd]).mean(0)
    return heavy + shift, hyd + shift


def _write_rh_cif(path, name):
    """The crystal as a CIF on rhombohedral axes (no bond loop: find_molecules with bonds='distance')."""
    heavy, hyd = _rh_molecule()
    inv = np.linalg.inv(_rh_lattice())
    sites = [(lab, el, p @ inv) for lab, el, p in zip(('C1', 'C2', 'O1'), ('C', 'C', 'O'), heavy)]
    sites += [(f'H{k + 1}', 'H', p @ inv) for k, p in enumerate(hyd)]
    lines = [f'data_{name}', f'_cell_length_a {_RH_A}', f'_cell_length_b {_RH_A}', f'_cell_length_c {_RH_A}',
             f'_cell_angle_alpha {_RH_ALPHA}', f'_cell_angle_beta {_RH_ALPHA}', f'_cell_angle_gamma {_RH_ALPHA}',
             "_symmetry_space_group_name_H-M 'R -3'", '_symmetry_Int_Tables_number 148',
             "_space_group_name_Hall '-P 3*'", '_cell_formula_units_Z 6', 'loop_', '_symmetry_equiv_pos_as_xyz']
    lines += list(_RH_OPS)
    lines += ['loop_', '_atom_site_label', '_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y',
              '_atom_site_fract_z']
    lines += [f'{lab} {el} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}' for lab, el, f in sites]
    path.write_text('\n'.join(lines) + '\n')
    return path


class _FakeRhCrystal(_FakeCrystal):
    """A fake crystal that states its cell (CCDC's ``cell_lengths``, ``cell_angles``)."""

    def __init__(self, identifier, build, a, alpha):
        super().__init__(identifier, build)
        self.cell_lengths, self.cell_angles = (a, a, a), (alpha, alpha, alpha)


def _rh_builder(shift=0.03, kept_h=()):
    """``build(mode)`` of the crystal as CCDC reads the file: fractional coordinates on its rhombohedral axes, O1
    and its hydrogen listed one rhombohedral cell along a (a centring translation, 2/3 1/3 1/3, on the hexagonal
    axes), the hydrogens moved ``shift`` A along their bonds (regenerated) except those in ``kept_h``, left on
    their deposited sites; CCDC's hydrogens labelled HC1.. (the deposited ones are H1..)."""
    heavy, hyd = _rh_molecule()
    lat = _rh_lattice()
    inv = np.linalg.inv(lat)

    def build(mode):
        hv, hh = heavy.copy(), hyd.copy()
        hv[2] += lat[0]
        hh[5] += lat[0]
        hh = np.array([hh[k] if k in kept_h else
                       hh[k] + shift * (hh[k] - hv[_H_PARENT[k]]) / np.linalg.norm(hh[k] - hv[_H_PARENT[k]])
                       for k in range(6)])
        atoms = [_FakeAtom(i, z, p, lab, frac=p @ inv) for i, (z, p, lab) in enumerate(zip(_HEAVY_Z, hv,
                                                                                          ('C1', 'C2', 'O1')))]
        atoms += [_FakeAtom(3 + k, 1, p, f'HC{k + 1}', frac=p @ inv) for k, p in enumerate(hh)]
        bonds = [_FakeBond(atoms[0], atoms[1]), _FakeBond(atoms[1], atoms[2])]
        bonds += [_FakeBond(atoms[3 + k], atoms[_H_PARENT[k]]) for k in range(6)]
        return [_FakeComponent(atoms, bonds)]

    return build


def _rh_found(tmp_path):
    read = MolCrystalData.read([_write_rh_cif(tmp_path / 'RHETOH.cif', 'RHETOH')], dtype=F64)
    assert read.message[0].startswith('[RHETOH] re-expressed from rhombohedral axes') and int(read.sg_ind[0]) == 148
    found = read.find_molecules(**FIND_D)
    assert _status(found) == ['ok'] and int(found.z_prime[0]) == 1
    return found


def test_rhombohedral_axes_coordinates_are_mapped_on_the_files_axes(tmp_path, monkeypatch):
    """CCDC reads a re-expressed row's file on its rhombohedral axes; sanitize compares there (modulo the
    rhombohedral lattice) and brings the hydrogens back onto the row's hexagonal axes. Before this, every such row
    was chemistry_failed ("cannot be mapped")."""
    found = _rh_found(tmp_path)
    crystal = _FakeRhCrystal('RHETOH', _rh_builder(), _RH_A, _RH_ALPHA)
    monkeypatch.setattr(cc, '_backends', lambda: (_FakeIO({'RHETOH': crystal}), ''))
    snap = _snapshot(found)
    out = found.sanitize(**REGEN)
    _assert_unchanged(found, snap)
    assert _status(out) == ['ok'], out.message[0]
    z = out.z
    assert z.tolist() == _HEAVY_Z + [1] * 6
    assert torch.equal(out.pos[z > 1], found.pos[found.z > 1])        # heavy atoms: find_molecules' coordinates
    H = z == 1
    assert out.h_source[H].tolist() == [1] * 6 and (out.deposit_site[H] == -1).all()
    # hydrogens at CCDC's parent-to-hydrogen vectors, carried onto the row's axes: 0.03 A beyond the deposited ones
    dep_H = found.pos[found.z == 1].double().numpy()
    d = np.linalg.norm(out.pos[H].double().numpy()[:, None] - dep_H[None], axis=-1).min(1)
    assert np.allclose(d, 0.03, atol=1e-6)
    # and their deposited-frame coordinates on the row's hexagonal axes: the same vectors from the parents'
    T_fc = crystal_steps._transforms64(out)[0][0].numpy()
    parent = {}
    for i, j in out.mol_bonds[0].T.tolist():
        for h, p in ((i, j), (j, i)):
            if int(z[h]) == 1 and int(z[p]) > 1:
                parent[h] = p
    hs = sorted(parent)
    ps = [parent[h] for h in hs]
    dep = out.deposit_frac.double().numpy()
    pos = out.pos.double().numpy()
    assert np.abs((dep[hs] - dep[ps]) @ T_fc.T - (pos[hs] - pos[ps])).max() < 1e-9
    assert [int(z[p]) for p in ps] == [6, 6, 6, 6, 6, 8]
    assert not bool(out.protonation_changed[0])
    # the placed hydrogens' deposited coordinates rebuild the posed crystal on the row's axes
    posed = out.parameterize_pose(**POSE).check_rebuild()
    assert _status(posed) == ['ok'] and float(posed.rebuild_max_dev[0]) < 1e-5
    # a rhombohedral cell that is not the row's (a 0.1 A longer): the frames are not converted, the map fails
    other = _FakeRhCrystal('RHETOH', _rh_builder(), _RH_A + 0.1, _RH_ALPHA)
    monkeypatch.setattr(cc, '_backends', lambda: (_FakeIO({'RHETOH': other}), ''))
    refused = found.sanitize(**REGEN)
    assert _status(refused) == ['chemistry_failed'] and 'cannot be mapped' in refused.message[0]


def test_rhombohedral_axes_hydrogens_are_matched_on_the_files_axes(tmp_path, monkeypatch):
    """A hydrogen CCDC keeps on its deposited site is that deposited atom on a re-expressed row too: the match runs
    on CCDC's rhombohedral axes (modulo that lattice), here for the O1 hydrogen listed one rhombohedral cell away.
    Matched on the row's hexagonal axes, no hydrogen is found: 'regenerate' re-places it and 'as_read' refuses."""
    found = _rh_found(tmp_path)
    labels = list(found.atom_labels[0])
    k = labels.index('H6')                                             # O1's hydrogen, as deposited
    site = int(found.deposit_site[k])

    def install(build):
        crystal = _FakeRhCrystal('RHETOH', build, _RH_A, _RH_ALPHA)
        monkeypatch.setattr(cc, '_backends', lambda: (_FakeIO({'RHETOH': crystal}), ''))

    # regenerate, CCDC keeping O1's hydrogen on its deposited site and placing the other five
    install(_rh_builder(kept_h=(5,)))
    out = found.sanitize(**REGEN)
    assert _status(out) == ['ok'], out.message[0]
    at = torch.nonzero(out.deposit_site == site).flatten()
    assert len(at) == 1
    j = int(at[0])
    assert int(out.h_source[j]) == 0 and out.atom_labels[0][j] == 'H6'
    assert torch.equal(out.pos[j], found.pos[k]) and torch.equal(out.deposit_frac[j], found.deposit_frac[k])
    H = out.z == 1
    placed = H & (out.h_source == 1)
    assert int(placed.sum()) == 5 and (out.deposit_site[placed] == -1).all()
    assert sorted(out.atom_labels[0][i] for i in torch.nonzero(placed).flatten().tolist()) == \
        [f'HC{i}' for i in range(1, 6)]
    removed = out.removed_deposit_site[0].tolist()
    assert len(removed) == 5 and site not in removed and not bool(out.protonation_changed[0])
    # as_read, every CCDC hydrogen on its deposited site: all six are the deposited atoms
    install(_rh_builder(shift=0.0))
    kept = found.sanitize(chemistry='ccdc', hydrogens='as_read', on_protonation_change='refuse')
    assert _status(kept) == ['ok'], kept.message[0]
    assert (kept.h_source == 0).all() and kept.removed_deposit_site[0] is None
    order = kept.deposit_site.tolist()
    assert sorted(order) == sorted(found.deposit_site.tolist())
    by_site = {int(s): i for i, s in enumerate(found.deposit_site.tolist())}
    assert torch.equal(kept.pos, found.pos[[by_site[s] for s in order]])
    assert list(kept.atom_labels[0]) == [labels[by_site[s]] for s in order]


# ======================================================================================
# the real CSD Python API on the fixtures
# ======================================================================================

@pytest.fixture(scope='module')
def fixture_chain():
    if not _have_ccdc():
        pytest.skip('CSD Python API unavailable or unlicensed')
    read = MolCrystalData.read(sorted(FIXTURES.glob('*.cif')), dtype=F64)
    found = read.find_molecules(**FIND)
    san = found.sanitize(**REGEN)
    posed = san.parameterize_pose(**POSE).check_rebuild().check_validity(max_atomic_number=100)
    feat = posed.features(fingerprint=OLD_FINGERPRINT, charges='gasteiger')
    return found, san, posed, feat


@needs_ccdc
def test_fixture_chain_end_to_end(fixture_chain):
    found, san, posed, feat = fixture_chain
    names = list(feat.identifier)
    st = dict(zip(names, _status(feat)))
    assert st['AJACOF'] == 'special_position' and st['EFUNUP'] == 'multi_component' and \
        st['MBTPHC'] == 'multi_component'
    ok = [i for i, s in enumerate(_status(feat)) if s == 'ok']
    assert len(ok) == 13
    full = 1 | DONE_FIND_MOLECULES | DONE_SANITIZE | DONE_POSE | DONE_REBUILD | DONE_VALIDITY | DONE_FEATURES
    assert all(int(feat.done[i]) == full for i in ok)
    for i in ok:
        sl_f, sl_s = _row_sl(found, i), _row_sl(san, i)
        # heavy atoms: find_molecules' coordinates exactly, before the pose
        assert torch.equal(san.pos[sl_s][san.z[sl_s] > 1], found.pos[sl_f][found.z[sl_f] > 1]), names[i]
        assert float(feat.rebuild_max_dev[i]) < 1e-4, names[i]
        H = feat.z[_row_sl(feat, i)] == 1
        assert (feat.h_source[_row_sl(feat, i)][H] == 1).all() and (feat.h_source[_row_sl(feat, i)][~H] == 0).all()
        assert feat.unit_smiles[i] is not None and all(isinstance(s, str) for s in feat.unit_smiles[i])
    # hydrogens regenerated: NUKCOL's file holds none (25 heavy atoms, 12 hydrogens placed), not compared
    k = names.index('NUKCOL')
    assert int(found.num_atoms[k]) == 25 and int(san.num_atoms[k]) == 37 and not bool(san.protonation_checked[k])
    # broken protonation is visible: KICDUX loses its B-H, ZZZKEA02 gains an N-H
    assert sorted(names[i] for i in range(len(names)) if bool(san.protonation_changed[i])) == ['KICDUX', 'ZZZKEA02']
    # and leaves each a charged unit (net +1): the cheap check that also covers files without hydrogens
    assert sorted(names[i] for i in range(len(names)) if bool(san.charge_imbalance[i])) == ['KICDUX', 'ZZZKEA02']
    kb = names.index('KICDUX')
    sl = _row_sl(san, kb)
    assert san.h_count_change[sl][san.z[sl] == 5].tolist() == [-1]
    # Z' = 2: equal atom order, each unit its own conformer
    j = names.index('IDOVUT')
    sl = _row_sl(feat, j)
    z, p = feat.z[sl], feat.pos[sl].double().numpy()
    n_u = len(z) // 2
    assert int(feat.z_prime[j]) == 2 and torch.equal(z[:n_u], z[n_u:])
    h = (z[:n_u] > 1).numpy()
    rms = min(crystal_steps._kabsch(p[n_u:][h], p[:n_u][h], imp)[0] for imp in (False, True))
    assert rms > 5e-3
    assert feat.unit_smiles[j][0] == feat.unit_smiles[j][1]
    # charges: UGOMAF (Sn) gives non-finite Gasteiger charges, kept as NaN and flagged; others finite
    u = names.index('UGOMAF')
    assert bool(feat.charges_nonfinite[u]) and torch.isnan(feat.x[_row_sl(feat, u)]).any()
    for i in ok:
        if i != u:
            assert not bool(feat.charges_nonfinite[i]) and torch.isfinite(feat.x[_row_sl(feat, i)]).all(), names[i]
    # rows refused before sanitize pass through it untouched
    for i in range(len(names)):
        if i not in ok:
            assert int(san.done[i]) == int(found.done[i])


@needs_ccdc
def test_fixture_hydrogen_choices():
    read = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in ('CEKZOH', 'NUKCOL', 'ZZZKEA02', 'KICDUX')], dtype=F64)
    found = read.find_molecules(**FIND)
    refused = found.sanitize(**dict(REGEN, on_protonation_change='refuse'))
    assert _status(refused) == ['ok', 'ok', 'hydrogen_count_mismatch', 'hydrogen_count_mismatch']
    for i in (2, 3):
        for k in ('z', 'pos'):
            assert torch.equal(refused[k][_row_sl(refused, i)], found[k][_row_sl(found, i)])
    as_read = found.sanitize(chemistry='ccdc', hydrogens='as_read', on_protonation_change='flag')
    assert _status(as_read)[:2] == ['ok', 'ok'] and (as_read.h_source == 0).all()
    sl = _row_sl(as_read, 0)
    by_site = {int(s): i for i, s in enumerate(found.deposit_site[_row_sl(found, 0)].tolist())}
    order = [by_site[int(s)] for s in as_read.deposit_site[sl].tolist()]
    assert torch.equal(as_read.pos[sl], found.pos[_row_sl(found, 0)][order])
    missing = found.sanitize(chemistry='ccdc', hydrogens='add_missing', on_protonation_change='flag')
    nk = _row_sl(missing, 1)                    # NUKCOL: no deposited hydrogen, all twelve placed
    assert int((missing.z[nk] == 1).sum()) == 12 and (missing.h_source[nk][missing.z[nk] == 1] == 1).all()
    ck = _row_sl(missing, 0)                    # CEKZOH: its ten deposited hydrogens kept
    assert int(((missing.z[ck] == 1) & (missing.h_source[ck] == 0)).sum()) == 10
    removed = found.sanitize(chemistry='ccdc', hydrogens='remove', on_protonation_change='flag')
    assert (removed.z > 1).all() and _status(removed)[:2] == ['ok', 'ok']
    # every heavy atom read as carrying no hydrogen: the old featurizer's deprotonated-mode SMILES, radicals and all
    assert removed.unit_smiles[0] == ['[C][C]OC(=O)/C(C#N)=C(/[C])[N-2]']
    assert bool(removed.charge_imbalance[0]) and 'net formal charges are [-2]' in removed.message[0]


@needs_ccdc
def test_fixtures_match_the_old_featurizer():
    try:
        import mxtaltools.dataset_utils.construction.featurize_cif_chunks as fcc
    except Exception as e:                       # the licence is checked on import
        pytest.skip(f'old featurizer unavailable: {e}')
    import contextlib
    import io
    from scipy.optimize import linear_sum_assignment
    names = ('CEKZOH', 'DELDEG', 'HECVAP', 'IYOHIP', 'KICDUX', 'KUNTEV', 'MIHTEE52', 'NUKCOL', 'UGOMAF', 'XANCIA',
             'XEWCUY')
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        old = [fcc.process_chunk([str(FIXTURES / f'{n}.cif')], 0, False, 'protonated', 6) for n in names]
    new = MolCrystalData.read([FIXTURES / f'{n}.cif' for n in names], dtype=F64).find_molecules(**FIND)
    new = new.sanitize(**REGEN).parameterize_pose(**POSE).check_rebuild()
    new = new.features(fingerprint=OLD_FINGERPRINT, charges='gasteiger')
    for i, rows in enumerate(old):
        assert len(rows) == 1 and _status(new)[i] == 'ok', names[i]
        o, sl = rows[0], _row_sl(new, i)
        zn, zo = new.z[sl], o.z
        assert int((zn == 1).sum()) == int((zo == 1).sum()), names[i]
        assert new.unit_smiles[i] == [o.smiles], names[i]
        assert np.array_equal(np.unpackbits(new.unit_fingerprint[i, 0].numpy()),
                              o.fingerprint.reshape(-1).numpy().astype(np.uint8)), names[i]
        # atoms: heavy index by index (both keep the CIF's heavy order), hydrogens by assignment
        pn, po = new.pos[sl].double().numpy(), o.pos.double().numpy()
        hn, ho = (zn > 1).numpy(), (zo > 1).numpy()
        Hn, Ho = np.nonzero(~hn)[0], np.nonzero(~ho)[0]
        cost = np.linalg.norm(pn[Hn][:, None] - po[Ho][None], axis=-1)
        rr, cc_ = linear_sum_assignment(cost)
        assert float(np.linalg.norm(pn[hn] - po[ho], axis=1).max()) < 5e-4, names[i]
        assert float(cost[rr, cc_].max()) < 5e-4, names[i]
        xn = new.x[sl].double().numpy()
        xo = o.x.double().numpy()
        if bool(new.charges_nonfinite[i]):
            assert (xo == 0).all(), names[i]             # the old one zeroed them silently
            continue
        assert np.abs(xn[hn] - xo[ho]).max() < 1e-6, names[i]
        assert np.abs(xn[Hn[rr]] - xo[Ho[cc_]]).max() < 1e-6, names[i]


@needs_ccdc
def test_store_round_trip_of_a_featurized_batch(fixture_chain, tmp_path):
    _, _, _, feat = fixture_chain
    feat.save(tmp_path / 'store', steps_run=feat.steps_run)
    back = MolCrystalData.load(tmp_path / 'store')
    for k in feat.keys():
        assert _same(feat[k], back[k]), k


@needs_ccdc
def test_real_ccdc_on_a_rhombohedral_axes_file(tmp_path):
    """The real API reads the file on its rhombohedral axes (the premise of the frame change) and the row passes
    sanitize, pose and rebuild on its hexagonal axes."""
    found = _rh_found(tmp_path)
    crystal = cc._ccdc_crystal(cc._backends()[0], found.source_path[0], found.block_name[0])
    assert np.allclose(tuple(crystal.cell_lengths), [_RH_A] * 3) and np.allclose(tuple(crystal.cell_angles),
                                                                                 [_RH_ALPHA] * 3)
    assert cc._ccdc_on_rhombohedral_axes(crystal, crystal_steps._transforms64(found)[0][0].numpy())
    out = found.sanitize(**REGEN)
    assert _status(out) == ['ok'], out.message[0]
    assert out.unit_smiles[0] == [_ETOH_SMILES] and torch.equal(out.pos[out.z > 1], found.pos[found.z > 1])
    assert int((out.z == 1).sum()) == 6 and (out.h_source[out.z == 1] == 1).all()
    posed = out.parameterize_pose(**POSE).check_rebuild()
    assert _status(posed) == ['ok'] and float(posed.rebuild_max_dev[0]) < 1e-4

"""MolCrystalData -> CrystalDescription import (``crystal_fd.io_mxt``), on the local mini CSD files.

1. Round trip against the original atoms. Every imported row of ``mini_new_prot_csd.pt``, ``mini_new_new_csd.pt``
   and ``mini_new_csd.pt`` (the 100-row samples of the full files) is compared, by ``checks.same_crystal``, with a
   second description fitted directly to the row's stored atoms (``pos``, the crystal's own molecule): an O(3)
   Kabsch fit of the imported molecule onto them and their heavy-atom centroid, no stored orientation or
   handedness used. The imported description is the one after ``to_sheet(+1)``, so the check also covers the
   improper element that moved it.
2. The literal unit cell (``checks.unit_cell_fractional``, the table's G) against the live builder's unit cell
   built from the row itself (``pose_aunit()`` with the row's stored operators, then ``build_unit_cell()``),
   atom sets compared per element at minimum image.
3. Where the raw CIF is on disk (``D:/crystal_datasets/CSD_dump``), the literal unit cell's heavy atoms against the
   CIF's, expanded by the CIF's own operators (the CCDC-free reader of ``dataset_utils/construction/cif``).
   Every imported row of the three files whose CIF is on disk.
4. Refusal reasons: one importable row mutated once per reason gets that reason (non-finite operators are
   ``nonfinite``; a molecule on the 2-fold axis of C2/c is ``special_position``, one on a 2_1 screw axis of P2_1/c
   is not); reordered operators still import; the mini files' nonstandard settings are refused.
5. Handedness: non-Sohncke rows end on the +1 sheet, Sohncke rows with stored h = -1 stay off it and are flagged;
   ``sheet=None`` keeps the stored handedness.
6. Z' = 2. Real rows of mini_new_prot_csd.pt: unit 0's stored atoms are the CIF's, units 1 and above are not (the
   featurizer copied unit 0's conformer into them; 0.014 Angstrom measured here, up to 2.5 Angstrom in the full
   file). Synthetic (from a P-1 row: a second unit at t + (1/2, 1/2, 1/2)): refused as ``z_prime_gt1`` by
   default; with ``allow_zp_gt1`` imported with the flag, per-unit sheet move, literal cell against the live
   builder; a second unit whose atoms are in another order is refused.
7. Imported cells on a face, stored in float32 (beta = 90 deg is 4.4e-8 rad off), are flagged by
   ``ops.reduce_cell`` (n_auto >= 2).
8. Importing and running ``io_mxt`` loads neither cctbx nor a GFN module.

Measured on 2026-10-16 over the 236 imported rows of the three files (every one has its CIF on disk); the
thresholds below have at least 10x margin:
- same_crystal against the fit to the stored atoms: median 2e-6, max 3.2e-4 Angstrom (HEWBEU; the stored rotation
  vector reproduces the stored atoms only to that level);
- literal against live unit cell: median 3e-6, max 1.8e-5 Angstrom (the description is what the builder builds);
- literal heavy atoms against the CIF: median 4.5e-6, max 5.0e-4 Angstrom.

The mini files are local data: every test here skips when they are absent.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mxtaltools.crystal_fd import checks, io_mxt
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.ops import cell_matrices

DATA = Path(os.environ.get('MXT_CSD_DATA', 'D:/crystal_datasets'))
MINIS = ('mini_new_prot_csd.pt', 'mini_new_new_csd.pt', 'mini_new_csd.pt')
CIF_DIR = DATA / 'CSD_dump'
F64 = torch.float64
NB = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3).to(F64)

#: Angstrom: tolerances for the checks above (measured maxima in the module docstring)
ATOM_TOL = 5e-3
CIF_TOL = 5e-3
#: the importer as these tests were written for it, before owner decisions D1 (nonstandard settings re-expressed)
#: and D2 (off-sheet Sohncke rows mirrored), 2026-10-16; where a test pins it, the defaults are covered by
#: test_io_settings.py
LEGACY = dict(nonstandard='refuse', mirror_reference=False)


def _load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f'{path} not on this machine')
    return torch.load(path, weights_only=False)


@pytest.fixture(scope='module', params=MINIS)
def imported(request):
    data = _load(request.param)
    desc, info = io_mxt.from_mol_crystal_data(data, **LEGACY)
    return request.param, data, desc, info


@pytest.fixture(scope='module')
def prot():
    return _load('mini_new_prot_csd.pt')


def _fit_to_atoms(row, mol, sg) -> CrystalDescription:
    """A description fitted to the stored atoms alone: O(3) Kabsch of the molecule onto them, heavy-atom centroid."""
    Y = torch.as_tensor(row.pos, dtype=F64)
    z = torch.as_tensor(row.z)
    c = Y[z > 1].mean(0)
    X = mol.pos
    U, _, Vt = torch.linalg.svd(X.T @ (Y - c))
    P = Vt.T @ U.T                                                   # P X ~ Y - c, det +-1
    cell = torch.cat([torch.as_tensor(row.cell_lengths, dtype=F64).reshape(3),
                      torch.as_tensor(row.cell_angles, dtype=F64).reshape(3)])[None]
    t = torch.linalg.solve(cell_matrices(cell)[0], c)
    return CrystalDescription.from_poses(sg, cell, t[None], P[None])


def _set_dev(fa, za, fb, zb, T) -> float:
    """Largest distance (Angstrom) from an atom of either set to the nearest atom of the same element in the other."""
    out = 0.0
    for (f1, z1, f2, z2) in ((fa, za, fb, zb), (fb, zb, fa, za)):
        for el in torch.unique(z1).tolist():
            a, b = f1[z1 == el], f2[z2 == el]
            if len(b) == 0:
                return float('inf')
            d = a[:, None] - b[None]
            d = d - torch.round(d)
            cart = torch.einsum('ij,abnj->abni', T, d[:, :, None] + NB)
            out = max(out, float(cart.norm(dim=-1).amin(-1).amin(1).max()))
    return out


def _literal_cell(desc_row, mol):
    """(fractional [n, 3] wrapped atoms, z [n]) of one Z' = 1 description's unit cell, float64."""
    f = checks.unit_cell_fractional(desc_row, mol)[0]               # [G, n, 3]
    f = f - torch.floor(f)
    return f.reshape(-1, 3), mol.z.repeat(f.shape[0])


# ---------------------------------------------------------------------------
# 1-3: round trips
# ---------------------------------------------------------------------------

def test_imports_something(imported):
    name, data, desc, info = imported
    assert info.n_rows == len(data) and desc.num_rows == int(info.imported.sum()) >= 50, name
    desc.validate()
    assert torch.equal(desc.sg, info.sg[info.row]) and len(info.mols) == desc.num_rows
    refused = [r for r in info.reason if r is not None]
    assert set(refused) <= {'nonstandard_setting', 'operator_count_mismatch', 'zp_units_differ', 'z_prime_gt1'},         set(refused)


def test_collated_batch_equals_list(prot):
    """A collated batch is read graph by graph (no to_data_list, which these batches do not support)."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    rows = [r for r in prot if int(r.z_prime.reshape(-1)[0]) == 1][:60]
    d1, i1 = io_mxt.from_mol_crystal_data(rows)
    d2, i2 = io_mxt.from_mol_crystal_data(collate_data_list(rows, max_z_prime=1))
    assert i1.reason == i2.reason and i1.identifier == i2.identifier
    for f in ('sg', 'cell', 't', 'R', 'h', 'z_prime'):
        assert torch.equal(getattr(d1, f), getattr(d2, f)), f
    assert all(torch.equal(a.pos, b.pos) for a, b in zip(i1.mols, i2.mols))


def test_round_trip_same_crystal(imported):
    """Imported description (after to_sheet) against a description fitted to the stored atoms alone."""
    name, data, desc, info = imported
    worst = 0.0
    for j, i in enumerate(info.row.tolist()):
        fit = _fit_to_atoms(data[i], info.mols[j], int(desc.sg[j]))
        worst = max(worst, float(checks.same_crystal(desc[j], fit, info.mols[j], cell_rtol=1e-6)[0]))
    assert worst < ATOM_TOL, (name, worst)


def test_round_trip_live_unit_cell(imported):
    """Literal unit cell (table G) against the live builder's (the row's own stored operators)."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    name, data, desc, info = imported
    worst = 0.0
    for j, i in enumerate(info.row.tolist()):
        b = collate_data_list([data[i].clone()], max_z_prime=1)
        b.pose_aunit()
        b.build_unit_cell()
        T = desc[j].cell_matrix()[0]
        f_live = torch.linalg.solve(T, b.unit_cell_pos.double().T).T
        z_live = b.z.repeat(int(b.sym_mult[0]))
        f_lit, z_lit = _literal_cell(desc[j], info.mols[j])
        assert len(f_lit) == len(f_live)
        worst = max(worst, _set_dev(f_lit, z_lit, f_live, z_live, T))
    assert worst < ATOM_TOL, (name, worst)


def test_round_trip_against_cif(imported):
    """Heavy atoms of the literal unit cell against the raw CIF expanded by its own operators, every imported row."""
    if not CIF_DIR.exists():
        pytest.skip(f'{CIF_DIR} not on this machine')
    from mxtaltools.dataset_utils.construction.cif import read_cif
    name, data, desc, info = imported
    n, worst = 0, 0.0
    for j in range(desc.num_rows):
        path = CIF_DIR / f'{info.identifier[info.row[j]]}.cif'
        if not path.exists():
            continue
        cif = read_cif(path, require_bonds=False)[0]
        cell = torch.cat([torch.as_tensor(cif.cell_lengths), torch.as_tensor(cif.cell_angles)]).to(F64)
        assert torch.allclose(cell, desc.cell[j], rtol=1e-5), info.identifier[info.row[j]]
        heavy = torch.as_tensor(cif.sites.z) > 1
        f0 = torch.as_tensor(cif.sites.frac, dtype=F64)[heavy]
        ops = torch.as_tensor(np.stack(cif.symmetry.operators), dtype=F64)
        f_cif = (torch.einsum('gij,nj->gni', ops[:, :3, :3], f0) + ops[:, None, :3, 3]).reshape(-1, 3)
        z_cif = torch.as_tensor(cif.sites.z)[heavy].repeat(len(ops))
        f_lit, z_lit = _literal_cell(desc[j], info.mols[j])
        keep = z_lit > 1
        worst = max(worst, _set_dev(f_lit[keep], z_lit[keep], f_cif - torch.floor(f_cif), z_cif,
                                    desc[j].cell_matrix()[0]))
        n += 1
    if n == 0:
        pytest.skip('no CIF of an imported row on disk')
    assert n >= 50 and worst < CIF_TOL, (name, n, worst)


# ---------------------------------------------------------------------------
# 4: refusals
# ---------------------------------------------------------------------------

def _standard_row(rows, sg=None):
    _, info = io_mxt.from_mol_crystal_data(rows, **LEGACY)
    for i in info.row.tolist():
        if sg is None or int(info.sg[i]) == sg:
            return rows[i]
    pytest.skip(f'no importable row of sg {sg}')


def _mutated(row, **fields):
    r = row.clone()
    for k, v in fields.items():
        setattr(r, k, v)
    return r


def _reposed(row, t_new):
    """``row`` with unit 0 moved to the fractional centroid ``t_new``, orientation kept: the stored atoms follow."""
    r = row.clone()
    cell = torch.cat([torch.as_tensor(row.cell_lengths).reshape(3), torch.as_tensor(row.cell_angles).reshape(3)])
    T = cell_matrices(cell.double()[None])[0]
    n = len(row.z) // int(row.z_prime.reshape(-1)[0])
    c = row.aunit_centroid.clone()
    shift = (T @ (torch.as_tensor(t_new, dtype=F64) - c[0, :3].double())).float()
    c[0, :3] = torch.as_tensor(t_new, dtype=torch.float32)
    p = row.pos.clone()
    p[:n] = p[:n] + shift
    r.aunit_centroid, r.pos = c, p
    return r


def test_refusal_reasons(prot):
    row = _standard_row(prot, sg=2)
    ops = np.asarray(row.symmetry_operators)
    c = row.aunit_centroid.clone()
    c_nan = c.clone()
    c_nan[0, 0] = float('nan')
    c_inv = c.clone()
    c_inv[0, :3] = 0.0                                               # on the inversion centre of P-1
    h0 = row.aunit_handedness.clone()
    h0[0, 0] = 0
    rv = row.aunit_orientation.clone()
    rv[0, :3] = rv[0, :3] * 1.05                                     # a different rotation: atoms move
    shifted = ops.copy()
    shifted[1, 0, 3] += 0.25                                         # another operator set of the same size
    c2c = _standard_row(prot, sg=15)
    cases = [
        ('missing_field', _mutated(row, pos=None)),
        ('invalid_space_group', _mutated(row, sg_ind=torch.tensor(0))),
        ('invalid_z_prime', _mutated(row, z_prime=torch.tensor([0]))),
        ('cocrystal', _mutated(row, cocrystal=True)),
        ('disorder', _mutated(row, has_disorder=True)),
        ('operator_count_mismatch', _mutated(row, symmetry_operators=ops[:1])),
        ('nonstandard_setting', _mutated(row, symmetry_operators=shifted)),
        ('nonfinite', _mutated(row, aunit_centroid=c_nan)),
        ('nonfinite', _mutated(row, symmetry_operators=np.full_like(ops, np.nan))),   # not a setting question
        ('degenerate_cell', _mutated(row, cell_angles=torch.tensor([[0.2, 0.2, 3.0]]))),
        ('degenerate_handedness', _mutated(row, aunit_handedness=h0)),
        ('no_heavy_atoms', _mutated(row, z=torch.ones_like(row.z))),
        ('special_position', _mutated(row, aunit_centroid=c_inv)),
        ('special_position', _reposed(c2c, [0.0, 0.3, 0.25])),     # on a 2-fold axis of C2/c
        ('pose_mismatch', _mutated(row, aunit_orientation=rv)),
        ('z_prime_gt1', _zp2(row)),
    ]
    screw = _reposed(_standard_row(prot, sg=14), [0.0, 0.3, 0.25])   # on a 2_1 axis of P2_1/c: no fixed point
    rows = [row, screw] + [r for _, r in cases]
    desc, info = io_mxt.from_mol_crystal_data(rows, **LEGACY)
    assert info.reason[:2] == [None, None], info.reason[:2]
    assert info.reason[2:] == [k for k, _ in cases], list(zip([k for k, _ in cases], info.reason[2:]))
    assert info.row.tolist() == [0, 1] and desc.num_rows == 2
    assert set(info.reason_counts()) == set(io_mxt.REASONS)
    assert {k for k, _ in cases} == set(io_mxt.REASONS) - {'zp_units_differ', 'unresolved_setting',
                                                           'setting_metric_mismatch'}  # test_zp2; test_io_settings


def test_reordered_operators_import(prot):
    row = _standard_row(prot, sg=14)
    ops = np.asarray(row.symmetry_operators)
    r = _mutated(row, symmetry_operators=ops[::-1].copy())
    desc, info = io_mxt.from_mol_crystal_data([row, r])
    assert info.imported.all() and info.ops_reordered.tolist() == [False, True]
    d = float(checks.same_crystal(desc[0], desc[1], info.mols[0])[0])
    assert d < 1e-9, d


def test_nonstandard_rows_refused(prot):
    _, info = io_mxt.from_mol_crystal_data(prot, **LEGACY)
    ns = [r for r in info.reason if r == 'nonstandard_setting']
    assert len(ns) >= 5
    # every refused setting carries the builder's own nonstandard flag; flagged rows may still import (reordered)
    for i, r in enumerate(info.reason):
        if r == 'nonstandard_setting':
            assert bool(info.stored_nonstandard[i])
        if info.ops_reordered[i]:
            assert r is None and bool(info.stored_nonstandard[i])


# ---------------------------------------------------------------------------
# 5: handedness
# ---------------------------------------------------------------------------

def test_sheet(prot):
    desc, info = io_mxt.from_mol_crystal_data(prot, **LEGACY)
    keep, info_k = io_mxt.from_mol_crystal_data(prot, sheet=None, **LEGACY)
    assert torch.equal(keep.h, info_k.h_input) and torch.equal(info.h_input, info_k.h_input)
    h_in = info.h_input[:, 0]
    assert (h_in == -1).any() and (h_in == 1).any()
    off = info.sohncke & (h_in == -1)
    assert torch.equal(~info.on_sheet, off)
    assert bool((desc.h[info.on_sheet, 0] == 1).all()) and bool((desc.h[off, 0] == -1).all())
    moved = info.on_sheet & (h_in == -1)
    assert moved.any()
    # a moved row is the same crystal up to the cell's float32 rounding: the stored angle 90 deg is 1.5707963705 rad
    # (4.4e-8 rad off), so T W T^-1 of a mirror is orthogonal only to that level and its O(3) projection moves
    # atoms by about 1e-7 relative (4e-7 Angstrom measured)
    for j in torch.nonzero(moved)[:, 0].tolist():
        d = float(checks.same_crystal(desc[j], keep[j], info.mols[j])[0])
        assert d < 1e-5, d


# ---------------------------------------------------------------------------
# 6: Z' = 2
# ---------------------------------------------------------------------------

def _zp2(row, permute: bool = False):
    """A Z' = 2 row of a P-1 crystal: unit 2 is unit 1 moved by (1/2, 1/2, 1/2).

    ``permute``: unit 2's atoms in reverse order (the same point set, so its stored pose is still exact, but the
    units no longer hold one molecule with one atom order).
    """
    r = row.clone()
    T = torch.as_tensor(row.T_fc, dtype=torch.float32).reshape(3, 3)
    shift = T @ torch.tensor([0.5, 0.5, 0.5])
    n = int(row.num_atoms)
    order = torch.arange(n - 1, -1, -1) if permute else torch.arange(n)
    r.pos = torch.cat([row.pos, (row.pos + shift)[order]])
    r.z = torch.cat([row.z, row.z[order]])
    r.x = torch.cat([row.x, row.x[order]])
    r.mol_ind = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    r.num_atoms = torch.tensor(2 * n)
    r.z_prime = torch.tensor([2])
    c, o, h = row.aunit_centroid.clone(), row.aunit_orientation.clone(), row.aunit_handedness.clone()
    c[0, 3:6] = torch.remainder(c[0, :3] + 0.5, 1.0)
    o[0, 3:6] = o[0, :3]
    h[0, 1] = h[0, 0]
    r.aunit_centroid, r.aunit_orientation, r.aunit_handedness = c, o, h
    return r


def test_zp2(prot):
    from mxtaltools.dataset_utils.utils import collate_data_list
    _, info0 = io_mxt.from_mol_crystal_data(prot, sheet=None)
    cands = [prot[i] for i in info0.row.tolist() if int(info0.sg[i]) == 2]
    row = next((r for r in cands if int(r.aunit_handedness.reshape(-1)[0]) == -1), cands[0])
    rows = [row, _zp2(row), _zp2(row, permute=True)]
    _, info_d = io_mxt.from_mol_crystal_data(rows)                    # default: Z' > 1 is not imported
    assert info_d.reason == [None, 'z_prime_gt1', 'zp_units_differ'], info_d.reason
    desc, info = io_mxt.from_mol_crystal_data(rows, allow_zp_gt1=True)
    assert info.reason == [None, None, 'zp_units_differ'], info.reason
    assert info.zp_gt1.tolist() == [False, True] and desc.max_z_prime == 2
    assert desc.z_prime.tolist() == [1, 2] and bool((desc.h[1] == 1).all()) and bool(info.on_sheet.all())
    # the Z' = 2 description's units, each a Z' = 1 crystal, against the live builder of the split row
    b = collate_data_list([_zp2(row)], max_z_prime=2).split_to_zp1_batch()
    b.pose_aunit()
    b.build_unit_cell()
    mol = info.mols[1]
    for k in range(2):
        unit = CrystalDescription.from_parts(desc.sg[1:2], desc.cell[1:2], desc.t[1:2, k], desc.R[1:2, k],
                                             desc.h[1:2, k])
        T = unit.cell_matrix()[0]
        live = b.unit_cell_pos[b.unit_cell_batch == k].double()
        f_live = torch.linalg.solve(T, live.T).T
        f_lit, z_lit = _literal_cell(unit, mol)
        assert _set_dev(f_lit, z_lit, f_live, mol.z.repeat(2), T) < ATOM_TOL


def test_real_zp_gt1_rows_carry_one_conformer(prot):
    """The Z' > 1 rows of mini_new_prot_csd.pt against their CIFs, each unit's stored heavy atoms expanded by the row's
    own operators: unit 0 is the CIF's, units 1 and above are not (the featurizer wrote unit 0's conformer into them).
    This is why the importer refuses Z' > 1 by default (``z_prime_gt1``)."""
    if not CIF_DIR.exists():
        pytest.skip(f'{CIF_DIR} not on this machine')
    from mxtaltools.dataset_utils.construction.cif import read_cif
    rows = [r for r in prot if int(r.z_prime.reshape(-1)[0]) > 1 and (CIF_DIR / f'{r.identifier}.cif').exists()]
    if not rows:
        pytest.skip("no Z' > 1 row with its CIF")
    for r in rows:
        zp = int(r.z_prime.reshape(-1)[0])
        cell = torch.cat([torch.as_tensor(r.cell_lengths).reshape(3), torch.as_tensor(r.cell_angles).reshape(3)])
        T = cell_matrices(cell.double()[None])[0]
        ops = torch.as_tensor(np.asarray(r.symmetry_operators), dtype=F64).reshape(-1, 4, 4)
        cif = read_cif(CIF_DIR / f'{r.identifier}.cif', require_bonds=False, require_integer_z_prime=False)[0]
        heavy = torch.as_tensor(cif.sites.z) > 1
        f0 = torch.as_tensor(cif.sites.frac, dtype=F64)[heavy]
        cops = torch.as_tensor(np.stack(cif.symmetry.operators), dtype=F64)
        f_cif = (torch.einsum('gij,nj->gni', cops[:, :3, :3], f0) + cops[:, None, :3, 3]).reshape(-1, 3)
        z_cif = torch.as_tensor(cif.sites.z)[heavy].repeat(len(cops))
        n = len(r.z) // zp
        dev = []
        for k in range(zp):
            z = r.z[k * n:(k + 1) * n]
            f = torch.linalg.solve(T, r.pos[k * n:(k + 1) * n].double().T).T[z > 1]
            fu = (torch.einsum('gij,nj->gni', ops[:, :3, :3], f) + ops[:, None, :3, 3]).reshape(-1, 3)
            zu = z[z > 1].repeat(len(ops))
            out = 0.0                                                # one way: unit k's atoms to the nearest CIF atom
            for el in torch.unique(zu).tolist():
                d = fu[zu == el][:, None] - f_cif[z_cif == el][None]
                d = d - torch.round(d)
                out = max(out, float(torch.einsum('ij,abnj->abni', T, d[:, :, None] + NB).norm(dim=-1)
                                     .amin(-1).amin(1).max()))
            dev.append(out)
        assert dev[0] < CIF_TOL and min(dev[1:]) > CIF_TOL, (r.identifier, dev)
    _, info = io_mxt.from_mol_crystal_data(rows, allow_zp_gt1=True, **LEGACY)
    assert all(x is not None for x in info.reason)                   # these mini rows: nonstandard settings


def test_float32_faces_are_flagged_after_import():
    """Imported monoclinic Z' = 1 rows whose beta is within float32 rounding of 90 deg (stored 4.4e-8 rad off), and
    triclinic rows with two such angles, are special metrics to ``ops.reduce_cell`` (n_auto >= 2)."""
    from mxtaltools.crystal_fd import ops
    desc, info = io_mxt.from_mol_crystal_data(_load('mini_new_csd.pt'))
    d = desc[torch.nonzero((desc.z_prime == 1) & (desc.sg <= 15))[:, 0]]
    red = ops.reduce_cell(d)
    near = (d.cell[:, 3:] - torch.pi / 2).abs() < 1e-6
    mono = d.sg >= 3
    special = (mono & near[:, 1]) | (~mono & (near.sum(1) >= 2))
    assert int((mono & near[:, 1]).sum()) >= 1, 'no monoclinic row at beta = 90 deg in this file'
    assert bool((red.n_auto[special] >= 2).all()), red.n_auto[special].tolist()


# ---------------------------------------------------------------------------
# 8: import isolation
# ---------------------------------------------------------------------------

_FRESH = '''
import sys, torch
from mxtaltools.crystal_fd import io_mxt
assert io_mxt.operator_set_match(io_mxt._std_ops(14)[::-1], 14) == (True, False)
bad = [k for k in sys.modules if k.split('.')[0] in ('cctbx', 'libtbx', 'scitbx', 'boost_adaptbx')]
assert not bad, bad
gfn = sys.argv[1]
files = [str(getattr(v, '__file__', None) or '').replace(chr(92), '/').lower() for v in list(sys.modules.values())]
loaded = [f for f in files if f.startswith(gfn)]
assert not loaded, loaded
'''


def test_io_mxt_imports_no_cctbx_or_gfn():
    root = Path(io_mxt.__file__).parents[2]
    gfn = str(root.parent / 'gfn_diffusion').replace(chr(92), '/').lower()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONPATH=str(root))
    r = subprocess.run([sys.executable, '-c', _FRESH, gfn], env=env, capture_output=True, text=True, timeout=300,
                       cwd=str(root))
    assert r.returncode == 0, r.stderr[-2000:]

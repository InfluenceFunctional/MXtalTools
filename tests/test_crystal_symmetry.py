"""Contract tests for find_symmetry (crystal_symmetry.py): space groups detected from the atoms.

CPU only. The owner's case (a P2_1/c crystal saved in P1, in a nonstandard basis with a shifted
origin); the 16 CIF fixtures as declared (every detected group equals the declared one,
including a centred P1 setting and a molecule written whole on a special position) and
expanded to P1 (each declared group found again); a supercell (more symmetry with the same
space-group number), a primitive cell of a centred group, the tolerance deciding, hydrogens
that break the heavy-atom group, a declared group the merged atoms no longer hold (lower), a
chain of merged images (merge_span), the angle tolerance deciding; the record's counts and
relation following the group the rebuild checks when spglib's own list is short; the rebuild
refusing a wrong origin and two atoms sent onto one; 'different' (a lattice that drops the
declared group; operators sharing rotations but not translations); every status, spglib's
reason on a failure, an exception other than RuntimeError / ValueError propagating; the
record's setting change checked independently of the code under test (a literal rebuild from
its standard-setting description, and ``crystal_fd.settings.setting_matches``); a later-stage
batch; input untouched; arguments.
"""
import copy
import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pytest
import spglib
import torch

from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_fd.settings import setting_matches
from mxtaltools.dataset_utils.data_class_methods import crystal_symmetry as csym
from mxtaltools.dataset_utils.data_class_methods.crystal_symmetry import RELATIONS, SYMMETRY_STATUS, find_symmetry
from mxtaltools.dataset_utils.data_classes import MolCrystalData

F64 = torch.float64
FIXTURES = Path(__file__).parent / 'cif' / 'fixtures'
TOL = dict(symprec=1e-3, angle_tolerance=-1, rebuild_tol=1e-3, h_tol=1e-3, merge_tol=1e-2)
FIND = dict(bonds='file', bond_scale=(1.15, 1.35), iso_cap=5000, special_tol=0.1)
POSE = dict(handedness='measured', degenerate_tol=1e-4, handedness_tol=1e-3)
DEG = np.pi / 180


# ======================================================================================
# helpers (independent of the module under test)
# ======================================================================================

def _T(cell6):
    """T_fc [3, 3] of a cell (Angstrom, radians)."""
    c = torch.as_tensor(np.asarray(cell6, dtype=float))[None]
    return batch_compute_fractional_transform(c[:, :3], c[:, 3:])[0][0].numpy()


def _cell_from_basis(cell6, N):
    """Lengths and angles of the cell whose basis vectors are the columns of N in the basis of ``cell6``."""
    T = _T(cell6)
    G = N.T @ (T.T @ T) @ N
    L = np.sqrt(np.diag(G))
    return np.array([*L, np.arccos(G[1, 2] / (L[1] * L[2])), np.arccos(G[0, 2] / (L[0] * L[2])),
                     np.arccos(G[0, 1] / (L[0] * L[1]))])


def _dist(d, T):
    d = d - np.round(d)
    return np.linalg.norm(d @ T.T, axis=-1)


def _full_cell(frac, z, ops, T, tol=1e-2):
    """The cell under ``ops`` (fractional in [0, 1)), images of one element within ``tol`` A kept once."""
    img = np.einsum('kij,nj->kni', ops[:, :3, :3], frac) + ops[:, None, :3, 3]
    img = (img - np.floor(img)).reshape(-1, 3)
    zz = np.tile(np.asarray(z), len(ops))
    keep = []
    for a in range(len(img)):
        if keep and ((_dist(img[keep] - img[a], T) < tol) & (zz[keep] == zz[a])).any():
            continue
        keep.append(a)
    return img[keep], zz[keep]


def _synthetic(sg, cell6, z, seed, min_dist=1.0):
    """Random atoms (elements ``z``) in general positions of ``SYM_OPS[sg]``: ``(asym frac, cell frac, cell z)``,
    no two atoms of the cell closer than ``min_dist`` A."""
    rng = np.random.default_rng(seed)
    T, ops = _T(cell6), np.stack(SYM_OPS[sg])
    for _ in range(2000):
        frac = rng.random((len(z), 3))
        X, Z = _full_cell(frac, z, ops, T, tol=1e-9)
        d = _dist(X[:, None] - X[None], T)
        np.fill_diagonal(d, np.inf)
        if len(X) == len(z) * len(ops) and d.min() >= min_dist:
            return frac, X, Z
    raise AssertionError('no well-separated atom set found')


def _batch(rows, sg=1, ops=None):
    """A bare batch from ``[(cell6, frac, z), ...]``: P1 (``setting='standard'``), or ``sg`` with ``ops``."""
    cell = torch.tensor(np.stack([np.asarray(r[0], float) for r in rows]), dtype=F64)
    xyz = torch.as_tensor(np.concatenate([np.asarray(r[1], float).reshape(-1, 3) @ _T(r[0]).T for r in rows]))
    z = torch.as_tensor(np.concatenate([np.asarray(r[2], dtype=np.int64) for r in rows]))
    ptr = torch.as_tensor(np.concatenate([[0], np.cumsum([len(r[2]) for r in rows])]))
    if ops is None:
        return MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=sg, setting='standard', dtype=F64)
    return MolCrystalData.from_arrays(cell, xyz, z, ptr=ptr, sg=sg, ops=ops, dtype=F64)


def _fixtures(dtype=F64):
    return MolCrystalData.read(sorted(FIXTURES.glob('*.cif')), dtype=dtype)


def _fixture_cells(b):
    """{identifier: (cell6, cell frac, cell z, file operators, sg)} for the read-ok fixtures, expanded by their own
    operators (whole molecules on special positions written once)."""
    out = {}
    for i, ident in enumerate(b.identifier):
        if int(b.status[i]) != 0:
            continue
        s, e = int(b.ptr[i]), int(b.ptr[i + 1])
        cell6 = np.concatenate([b.cell_lengths[i].double().numpy(), b.cell_angles[i].double().numpy()])
        T = _T(cell6)
        frac = b.pos[s:e].double().numpy() @ np.linalg.inv(T).T
        ops = np.asarray(b.symmetry_operators[i], dtype=float)
        X, Z = _full_cell(frac, b.z[s:e].numpy(), ops, T)
        out[ident] = (cell6, X, Z, ops, int(b.sg_ind[i]))
    return out


def _status(rec, i):
    return SYMMETRY_STATUS[int(rec['status'][i])]


def _relation(rec, i):
    r = int(rec['relation'][i])
    return RELATIONS[r] if r >= 0 else None


def _rebuild_dev(rec, i, cell6, X, Z):
    """The record's standard-setting description rebuilt by hand and matched to the cell ``X``, ``Z``.

    One atom per orbit (``asym_frac``, ``asym_z``) under ``SYM_OPS[sg]`` and the standard lattice, carried to the
    input cell by ``x_in = P x_std + p``: the largest distance from a rebuilt atom to the cell's nearest atom of its
    element and from a cell atom to the nearest rebuilt one (a two-way Hausdorff distance, Angstrom)."""
    sg, P, p = int(rec['sg_detected'][i]), rec['P'][i].numpy(), rec['p'][i].numpy()
    A, Az = rec['asym_frac'][i], rec['asym_z'][i]
    ops = np.stack(SYM_OPS[sg])
    imgs = (np.einsum('kij,nj->kni', ops[:, :3, :3], A) + ops[:, None, :3, 3]).reshape(-1, 3)
    iz = np.tile(Az, len(ops))
    if np.linalg.det(P) < 1 - 1e-9:                # the standard cell is the smaller: its translations inside ours
        shifts = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)
        imgs = (imgs[:, None] + shifts[None]).reshape(-1, 3)
        iz = np.repeat(iz, len(shifts))
    built = imgs @ P.T + p
    T = _T(cell6)
    worst = 0.0
    for e in np.unique(Z):
        d = _dist(built[iz == e][:, None] - X[Z == e][None], T)
        worst = max(worst, float(d.min(1).max()), float(d.min(0).max()))
    assert set(np.unique(iz)) == set(np.unique(Z))
    return worst


# ======================================================================================
# the owner's case
# ======================================================================================

def test_p21c_saved_in_p1_in_a_nonstandard_basis_with_a_shifted_origin_is_found():
    """P2_1/c written in P1 (identity only), in the basis a, b, a + c with the origin moved: detected as
    P2_1/c, four operators per cell against one, rebuilt from its standard-setting description."""
    cell6 = np.array([10.0, 12.0, 14.0, 90 * DEG, 105 * DEG, 90 * DEG])
    z = [6] * 8 + [7, 8] + [1] * 4
    _, X, Z = _synthetic(14, cell6, z, seed=1)
    N = np.array([[1.0, 0, 1], [0, 1, 0], [0, 0, 1]])        # x_old = N x_new + s
    s = np.array([0.137, 0.291, 0.083])
    Xn = (X - s) @ np.linalg.inv(N).T
    Xn -= np.floor(Xn)
    cell_n = _cell_from_basis(cell6, N)
    rec = find_symmetry(_batch([(cell_n, Xn, Z)]), **TOL)
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert (int(rec['sg_declared'][0]), int(rec['sg_detected'][0])) == (1, 14)
    assert _relation(rec, 0) == 'higher'
    assert (int(rec['n_ops_declared'][0]), int(rec['n_ops_detected'][0]), float(rec['index'][0])) == (1, 4, 4.0)
    assert (int(rec['n_asym_heavy'][0]), int(rec['n_asym_h'][0])) == (10, 4)
    assert float(rec['rebuild_dev'][0]) < 1e-9 and float(rec['h_dev'][0]) < 1e-9
    assert abs(np.linalg.det(rec['P'][0].numpy()) - 1) < 1e-9
    assert _rebuild_dev(rec, 0, cell_n, Xn, Z) < 1e-9
    # the standard cell is the input cell in the new basis: same volume
    V = lambda c: abs(np.linalg.det(_T(c)))
    assert abs(V(rec['std_cell'][0].numpy()) - V(cell_n)) < 1e-6
    assert 'P2_1/c' in rec['hm_detected'][0]


# ======================================================================================
# the fixtures
# ======================================================================================

@pytest.mark.parametrize('dtype', [F64, torch.float32])
def test_fixtures_as_declared_detect_their_own_group(dtype):
    b = _fixtures(dtype)
    rec = find_symmetry(b, **TOL)
    ok = [i for i in range(b.num_graphs) if int(b.status[i]) == 0]
    assert len(ok) == 16
    for i in ok:
        ident = b.identifier[i]
        assert _status(rec, i) == 'ok', (ident, rec['message'][i])
        assert int(rec['sg_detected'][i]) == int(b.sg_ind[i]), ident
        assert _relation(rec, i) == 'equal', ident
        assert float(rec['index'][i]) == 1.0
        assert float(rec['rebuild_dev'][i]) < (1e-6 if dtype == F64 else 1e-4), ident
        # the file's group re-expressed by the record's setting change is SYM_OPS[sg]
        ops = np.asarray(b.symmetry_operators[i], dtype=float)
        assert setting_matches(ops, int(b.sg_ind[i]), rec['P'][i].numpy(), rec['p'][i].numpy(), tol=1e-5), ident
        V_in = float(b.cell_volume[i])
        V_std = abs(np.linalg.det(_T(rec['std_cell'][i].numpy())))
        assert V_std == pytest.approx(np.linalg.det(rec['P'][i].numpy()) * V_in, rel=1e-6), ident
    by_id = {b.identifier[i]: i for i in ok}
    # a centred P1 setting: four operators per cell, the standard P-1 cell is half the volume
    i = by_id['ZZZKEA02']
    assert int(rec['n_ops_declared'][i]) == 4 and np.linalg.det(rec['P'][i].numpy()) == pytest.approx(0.5)
    # a molecule written whole on a 2-fold axis (find_molecules: special_position): the coinciding images merge
    i = by_id['AJACOF']
    assert int(rec['n_cell_atoms'][i]) < 8 * int(b.num_atoms[i])


def test_fixtures_saved_in_p1_detect_the_declared_group():
    b = _fixtures()
    declared = find_symmetry(b, **TOL)
    cells = _fixture_cells(b)
    names = sorted(cells)
    p1 = _batch([cells[n][:3] for n in names])
    rec = find_symmetry(p1, **TOL)
    for j, name in enumerate(names):
        cell6, X, Z, ops, sg = cells[name]
        i = b.identifier.index(name)
        assert _status(rec, j) == 'ok', (name, rec['message'][j])
        assert int(rec['sg_detected'][j]) == sg, name
        assert _relation(rec, j) == 'higher', name
        assert int(rec['n_ops_detected'][j]) == len(ops) and float(rec['index'][j]) == len(ops), name
        # the same asymmetric unit as the declared description
        assert int(rec['n_asym_heavy'][j]) == int(declared['n_asym_heavy'][i]), name
        assert int(rec['n_asym_h'][j]) == int(declared['n_asym_h'][i]), name
        assert _rebuild_dev(rec, j, cell6, X, Z) < 1e-6, name


# ======================================================================================
# cells and tolerances
# ======================================================================================

def test_a_supercell_has_more_symmetry_under_the_same_number():
    """P1 written in a cell doubled along b: still sg 1, but two operators per cell (a lattice translation)."""
    cell6 = np.array([8.0, 9.0, 10.0, 80 * DEG, 95 * DEG, 100 * DEG])
    z = [6] * 6 + [8] + [1] * 3
    _, X, Z = _synthetic(1, cell6, z, seed=2)
    X2 = np.concatenate([X * [1, 0.5, 1], X * [1, 0.5, 1] + [0, 0.5, 0]])
    cell2 = cell6 * np.array([1, 2, 1, 1, 1, 1])
    rec = find_symmetry(_batch([(cell2, X2, np.tile(Z, 2))]), **TOL)
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert (int(rec['sg_detected'][0]), _relation(rec, 0), int(rec['n_ops_detected'][0])) == (1, 'higher', 2)
    assert np.linalg.det(rec['P'][0].numpy()) == pytest.approx(0.5)
    assert int(rec['n_asym_heavy'][0]) == 7
    assert _rebuild_dev(rec, 0, cell2, X2, np.tile(Z, 2)) < 1e-9


def test_a_primitive_cell_of_a_centred_group():
    """C2/c (CEKZOH, 8 operators) written in P1 in its primitive cell: sg 15, the standard cell twice ours."""
    cells = _fixture_cells(_fixtures())
    cell6, X, Z, ops, sg = cells['CEKZOH']
    assert sg == 15
    Pp = np.array([[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1.0]])           # columns (a + b) / 2, (-a + b) / 2, c
    Xp = X @ np.linalg.inv(Pp).T
    Xp -= np.floor(Xp)
    cell_p = _cell_from_basis(cell6, Pp)
    Xp, Zp = _full_cell(Xp, Z, np.eye(4)[None], _T(cell_p), tol=1e-6)   # the two centring copies are one atom
    assert len(Xp) == len(X) // 2
    rec = find_symmetry(_batch([(cell_p, Xp, Zp)]), **TOL)
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert (int(rec['sg_detected'][0]), _relation(rec, 0), float(rec['index'][0])) == (15, 'higher', 4.0)
    assert np.linalg.det(rec['P'][0].numpy()) == pytest.approx(2.0)
    assert _rebuild_dev(rec, 0, cell_p, Xp, Zp) < 1e-9


def _noisy_p1_of_pbar1(sigma, seed=3):
    cell6 = np.array([9.0, 10.0, 11.0, 70 * DEG, 80 * DEG, 85 * DEG])
    z = [6] * 6 + [7] + [1] * 3
    _, X, Z = _synthetic(2, cell6, z, seed=seed, min_dist=1.2)
    rng = np.random.default_rng(seed)
    T = _T(cell6)
    Xn = X + rng.normal(0, sigma, X.shape) @ np.linalg.inv(T).T
    return cell6, Xn - np.floor(Xn), Z


def test_the_tolerance_decides():
    cell6, X, Z = _noisy_p1_of_pbar1(3e-3)
    b = _batch([(cell6, X, Z)])
    tight = find_symmetry(b, **TOL)
    assert _status(tight, 0) == 'ok' and (int(tight['sg_detected'][0]), _relation(tight, 0)) == (1, 'equal')
    loose = dict(TOL, symprec=5e-2, rebuild_tol=5e-2, h_tol=5e-2)
    rec = find_symmetry(b, **loose)
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert (int(rec['sg_detected'][0]), _relation(rec, 0)) == (2, 'higher')
    assert 1e-3 < float(rec['rebuild_dev'][0]) < 5e-2
    # the same detection, verified at a tighter rebuild tolerance: refused, the detection still reported
    rec = find_symmetry(b, **dict(loose, rebuild_tol=1e-3))
    assert _status(rec, 0) == 'rebuild_mismatch'
    assert (int(rec['sg_detected'][0]), _relation(rec, 0)) == (2, 'higher')
    assert rec['asym_frac'][0] is None


def test_hydrogens_that_break_the_heavy_atom_group():
    cell6, X, Z = _noisy_p1_of_pbar1(0.0)
    half = len(Z) // 2                             # the second half: the inversion images
    Xh = X.copy()
    moved = np.zeros(len(Z), dtype=bool)
    moved[half:] = Z[half:] == 1
    Xh[moved] += np.array([0.25, 0, 0]) @ np.linalg.inv(_T(cell6)).T
    b = _batch([(cell6, Xh, Z)])
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'hydrogen_mismatch', rec['message'][0]
    assert (int(rec['sg_detected'][0]), _relation(rec, 0)) == (2, 'higher')
    assert float(rec['rebuild_dev'][0]) < 1e-9 and 0.2 < float(rec['h_dev'][0]) < 0.3
    assert int(rec['sg_all_atoms'][0]) == 1
    assert int(rec['n_asym_h'][0]) == -1 and rec['asym_frac'][0] is None
    rec = find_symmetry(b, **dict(TOL, h_tol=0.5))
    assert _status(rec, 0) == 'ok' and int(rec['n_asym_h'][0]) == 3 and int(rec['sg_all_atoms'][0]) == 0


def test_a_declared_group_the_merged_atoms_do_not_hold_is_lower():
    """P-1 declared, a Cl 0.003 A from the inversion centre: its two images (0.006 A apart) merge at
    merge_tol 0.01 A into one atom off the centre, so at symprec 1e-3 the atoms hold P1 only."""
    cell6 = np.array([9.0, 10.0, 11.0, 70 * DEG, 80 * DEG, 85 * DEG])
    frac, _, _ = _synthetic(2, cell6, [6] * 5, seed=4, min_dist=1.5)
    frac = frac * 0.5 + 0.25                       # keep the carbons away from the centre at the origin
    cl = np.array([[0.003, 0, 0]]) @ np.linalg.inv(_T(cell6)).T
    b = _batch([(cell6, np.concatenate([frac, cl]), [6] * 5 + [17])], sg=2, ops=[np.stack(SYM_OPS[2])])
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert int(rec['n_cell_atoms'][0]) == 11
    assert (int(rec['sg_detected'][0]), _relation(rec, 0), float(rec['index'][0])) == (1, 'lower', 0.5)
    # the merge is in the record: the two images span 0.006 A, above symprec, and the message says so
    assert float(rec['merge_span'][0]) == pytest.approx(0.006, abs=1e-9)
    assert 'span up to' in rec['message'][0]
    # a looser detection holds the declared group again
    rec = find_symmetry(b, **dict(TOL, symprec=2e-2, rebuild_tol=2e-2))
    assert (int(rec['sg_detected'][0]), _relation(rec, 0)) == (2, 'equal')
    assert 'span up to' not in rec['message'][0]


def test_a_chain_of_merged_images_is_recorded():
    """P6 declared, a carbon 0.006 A off the 6-fold axis: its six images lie on a hexagon of side 0.006 A, so
    neighbours link at merge_tol 0.01 A and the chain makes one atom of images up to 0.012 A apart (opposite
    corners), more than merge_tol. merge_span records that; at symprec 1e-3 the kept image breaks the 6-fold."""
    cell6 = np.array([10.0, 10.0, 8.0, 90 * DEG, 90 * DEG, 120 * DEG])
    frac, _, _ = _synthetic(168, cell6, [6, 6, 7], seed=6, min_dist=1.5)   # images 1.5 A apart: off every axis
    near_axis = np.array([[0.006 / 10.0, 0.0, 0.1]])
    ops = np.stack(SYM_OPS[168])
    b = _batch([(cell6, np.concatenate([frac, near_axis]), [6, 6, 7, 6])], sg=168, ops=[ops])
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert int(rec['n_cell_atoms'][0]) == 6 * 3 + 1
    assert float(rec['merge_span'][0]) == pytest.approx(0.012, abs=1e-9)
    assert float(rec['merge_span'][0]) > TOL['merge_tol']
    assert _relation(rec, 0) == 'lower' and 'span up to 0.012 A' in rec['message'][0]
    # nothing merged: span 0
    rec = find_symmetry(b, **dict(TOL, merge_tol=1e-3))
    assert float(rec['merge_span'][0]) == 0.0 and int(rec['n_cell_atoms'][0]) == 6 * 4


def test_the_angle_tolerance_reaches_spglib():
    """P2_1/c atoms, exact in fractional coordinates, on a cell with gamma = 90.01 degrees: under spglib's derived
    angle rule at symprec 1e-3 A the metric fails the 2-fold (lower: P-1); at an explicit 0.05 degrees it holds
    (equal). The rebuild works in fractional coordinates and passes both ways."""
    cell6 = np.array([10.0, 12.0, 14.0, 90 * DEG, 105 * DEG, 90.01 * DEG])
    asym = np.array([[0.11, 0.21, 0.31], [0.37, 0.12, 0.08], [0.25, 0.41, 0.17], [0.05, 0.33, 0.44]])
    b = _batch([(cell6, asym, [6, 6, 7, 8])], sg=14, ops=[np.stack(SYM_OPS[14])])
    rec = find_symmetry(b, **dict(TOL, angle_tolerance=-1))
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert (int(rec['sg_detected'][0]), _relation(rec, 0)) == (2, 'lower')
    rec = find_symmetry(b, **dict(TOL, angle_tolerance=0.05))
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert (int(rec['sg_detected'][0]), _relation(rec, 0)) == (14, 'equal')
    assert float(rec['rebuild_dev'][0]) < 1e-9


# ======================================================================================
# statuses and the contract
# ======================================================================================

def _status_batch():
    """Rows: 0 a P1 copy of a P-1 set (ok, higher); 1 a C and an O on one site (superposed); 2 hydrogens only;
    3 no atoms; 4 a bad element (read status not ok)."""
    cell6, X, Z = _noisy_p1_of_pbar1(0.0)
    sup = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
    hh = np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])
    return _batch([(cell6, X, Z), (cell6, sup, [6, 8, 6]), (cell6, hh, [1, 1]), (cell6, np.zeros((0, 3)), []),
                   (cell6, X[:3], [0, 6, 6])])


def test_every_status():
    b = _status_batch()
    assert [int(s) for s in b.status] == [0, 0, 0, 0, 6]   # bad_element on the last row
    rec = find_symmetry(b, **TOL)
    assert [_status(rec, i) for i in range(5)] == ['ok', 'superposed_atoms', 'no_heavy_atoms', 'no_atoms', 'not_run']
    assert _relation(rec, 0) == 'higher' and _relation(rec, 1) is None
    assert 'different elements' in rec['message'][1]
    # selected explicitly, a row is run whatever its status; unselected rows are not
    rec = find_symmetry(b, **TOL, rows=[1])
    assert [_status(rec, i) for i in range(5)] == ['not_run', 'superposed_atoms', 'not_run', 'not_run', 'not_run']


def test_detection_and_setting_failures(monkeypatch):
    b = _status_batch().select_rows([0])
    monkeypatch.setattr(csym, '_spglib_dataset', lambda *a, **k: (None, 'spglib: forced'))
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'detection_failed' and 'forced' in rec['message'][0]
    monkeypatch.undo()
    monkeypatch.setattr(csym, '_standard_setting', lambda *a, **k: (None, None, 'forced'))
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'setting_failed' and 'forced' in rec['message'][0]
    assert (int(rec['sg_detected'][0]), _relation(rec, 0)) == (2, 'higher')     # reported though not verified
    assert bool(torch.isnan(rec['P'][0]).all())


def test_spglib_failure_on_close_atoms_is_a_status(monkeypatch):
    """Two carbons 0.005 A apart (kept apart at merge_tol 1e-3): spglib refuses them at symprec 1e-2, and its reason
    is in the message; spglib's error-handling flag is as it was. Where spglib gives no reason, the message says so."""
    cell6 = np.array([9.0, 10.0, 11.0, 90 * DEG, 90 * DEG, 90 * DEG])
    frac = np.array([[0.1, 0.1, 0.1], [0.1 + 0.005 / 9, 0.1, 0.1], [0.5, 0.4, 0.3]])
    b = _batch([(cell6, frac, [6, 6, 8])])
    flag = getattr(spglib.error, 'OLD_ERROR_HANDLING', None)
    rec = find_symmetry(b, **dict(TOL, symprec=1e-2, merge_tol=1e-3))
    assert _status(rec, 0) == 'detection_failed', rec['message'][0]
    assert 'too close' in rec['message'][0], rec['message'][0]
    assert getattr(spglib.error, 'OLD_ERROR_HANDLING', None) is flag
    monkeypatch.setattr(spglib, 'get_symmetry_dataset', lambda *a, **k: None)
    rec = find_symmetry(b, **dict(TOL, symprec=1e-2, merge_tol=1e-3))
    assert _status(rec, 0) == 'detection_failed'
    assert 'spglib returned no dataset and gave no reason' in rec['message'][0], rec['message'][0]


def _p21c_declared(seed=7):
    """A P2_1/c row in the standard setting with its own operators (4 atoms + 2 H in the asymmetric unit)."""
    cell6 = np.array([10.0, 12.0, 14.0, 90 * DEG, 105 * DEG, 90 * DEG])
    z = [6, 6, 7, 8, 1, 1]
    asym, X, Z = _synthetic(14, cell6, z, seed=seed)
    return cell6, asym, z, X, Z


def test_counts_and_relation_follow_the_group_the_rebuild_checks(monkeypatch):
    """spglib's operator list can be a proper subgroup of the group it names (CSD TEGBEP at 0.05 A: three pure
    translations named Cc). Here spglib's list is cut to its translations (identity only) on the owner's case: the
    record still counts SYM_OPS[14] carried into the input cell (four operators, index 4, higher), the operators the
    rebuild sends every atom through, and the message gives spglib's count."""
    cell6 = np.array([10.0, 12.0, 14.0, 90 * DEG, 105 * DEG, 90 * DEG])
    _, X, Z = _synthetic(14, cell6, [6] * 8 + [7, 8] + [1] * 4, seed=1)
    b = _batch([(cell6, X, Z)])
    real = csym._spglib_dataset

    def translations_only(lattice, frac, numbers, symprec, angle_tolerance, hall_number):
        ds, err = real(lattice, frac, numbers, symprec, angle_tolerance, hall_number)
        if ds is None or hall_number != 0:
            return ds, err
        R, t = np.asarray(ds.rotations), np.asarray(ds.translations)
        keep = (R == np.eye(3, dtype=R.dtype)).all((1, 2))
        return dataclasses.replace(ds, rotations=R[keep], translations=t[keep]), err

    monkeypatch.setattr(csym, '_spglib_dataset', translations_only)
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert int(rec['sg_detected'][0]) == 14
    assert (int(rec['n_ops_detected'][0]), float(rec['index'][0]), _relation(rec, 0)) == (4, 4.0, 'higher')
    assert "spglib's operator list has 1 operators per input cell" in rec['message'][0]
    assert _rebuild_dev(rec, 0, cell6, X, Z) < 1e-9


def test_the_rebuild_sends_the_atoms_through_the_counted_operators(monkeypatch):
    """The number of operators the rebuild applies is n_ops_detected, on every ok fixture row and on P1 copies."""
    b = _fixtures()
    cells = _fixture_cells(b)
    names = sorted(cells)
    seen = []
    real = csym._image_match

    def counting(X, Zx, ops_in, T_fc):
        seen.append(len(ops_in))
        return real(X, Zx, ops_in, T_fc)

    monkeypatch.setattr(csym, '_image_match', counting)
    for batch in (b, _batch([cells[n][:3] for n in names])):
        seen.clear()
        rec = find_symmetry(batch, **TOL)
        ok = [i for i in range(batch.num_graphs) if _status(rec, i) == 'ok']
        assert len(ok) == 16 and len(seen) == 16
        assert seen == [int(rec['n_ops_detected'][i]) for i in ok]


def test_a_wrong_origin_is_refused_by_the_rebuild(monkeypatch):
    """The setting change handed to the rebuild with its origin moved 0.01 along a (0.1 A): the operators carried
    into the input cell no longer hold (the inversion moves atoms by 0.2 A): rebuild_mismatch, the moved origin
    reported. Those operators share their rotations with the declared ones but not all translations: 'different'."""
    cell6, asym, z, X, Z = _p21c_declared()
    b = _batch([(cell6, asym, z)], sg=14, ops=[np.stack(SYM_OPS[14])])
    plain = find_symmetry(b, **TOL)
    assert _status(plain, 0) == 'ok' and _relation(plain, 0) == 'equal', plain['message'][0]
    real = csym._standard_setting

    def moved(*a, **k):
        P_spg, p_spg, how = real(*a, **k)
        return P_spg, p_spg + np.array([0.01, 0.0, 0.0]), how

    monkeypatch.setattr(csym, '_standard_setting', moved)
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'rebuild_mismatch', rec['message'][0]
    assert 0.1 < float(rec['rebuild_dev'][0]) < 0.3
    assert _relation(rec, 0) == 'different' and int(rec['n_ops_detected'][0]) == 4
    d = rec['p'][0].numpy() - (plain['p'][0].numpy() - [0.01, 0, 0])
    assert np.abs(d - np.round(d)).max() < 1e-9
    assert rec['asym_frac'][0] is None


def test_two_heavy_atoms_sent_onto_one_are_refused(monkeypatch):
    """An exact P-1 carbon set plus one carbon 0.003 A from one of them (kept apart at merge_tol 1e-3). spglib
    refuses such a pair outright, so the detection here is handed the set without the extra carbon; the rebuild
    sees all atoms: the inversion sends the pair onto one carbon, within rebuild_tol (0.01 A) but not a permutation."""
    cell6 = np.array([9.0, 10.0, 11.0, 70 * DEG, 80 * DEG, 85 * DEG])
    _, X, Z = _synthetic(2, cell6, [6] * 5, seed=8, min_dist=1.5)
    extra = X[:1] + np.array([0.003, 0, 0]) @ np.linalg.inv(_T(cell6)).T
    b = _batch([(cell6, np.concatenate([X, extra]), np.concatenate([Z, [6]]))])
    real = csym._spglib_dataset
    monkeypatch.setattr(csym, '_spglib_dataset',
                        lambda lattice, frac, numbers, *a: real(lattice, frac[:-1], numbers[:-1], *a))
    tol = dict(TOL, rebuild_tol=1e-2, merge_tol=1e-3)
    rec = find_symmetry(b, **tol)
    assert _status(rec, 0) == 'rebuild_mismatch', rec['message'][0]
    assert 'two heavy atoms onto one' in rec['message'][0]
    assert int(rec['sg_detected'][0]) == 2 and float(rec['rebuild_dev'][0]) == pytest.approx(0.003, abs=1e-9)
    monkeypatch.undo()
    rec = find_symmetry(b, **tol)                       # spglib on all atoms: no inversion, P1 holds
    assert _status(rec, 0) == 'ok' and int(rec['sg_detected'][0]) == 1, rec['message'][0]


def test_two_hydrogens_sent_onto_one_are_refused():
    """An exact P-1 carbon set; hydrogens at h, -h and -h + 0.05 A. The inversion sends the last two onto the one at
    h, within h_tol (0.1 A) but not a permutation: hydrogen_mismatch, and spglib on all atoms finds P1."""
    cell6 = np.array([9.0, 10.0, 11.0, 70 * DEG, 80 * DEG, 85 * DEG])
    _, X, Z = _synthetic(2, cell6, [6] * 5, seed=9, min_dist=1.5)
    h = np.array([0.3, 0.3, 0.3])
    dx = np.array([0.05, 0, 0]) @ np.linalg.inv(_T(cell6)).T
    frac = np.concatenate([X, [h, -h, -h + dx]])
    b = _batch([(cell6, frac - np.floor(frac), np.concatenate([Z, [1, 1, 1]]))])
    rec = find_symmetry(b, **dict(TOL, h_tol=0.1))
    assert _status(rec, 0) == 'hydrogen_mismatch', rec['message'][0]
    assert 'two hydrogens onto one' in rec['message'][0]
    assert int(rec['sg_detected'][0]) == 2 and float(rec['h_dev'][0]) == pytest.approx(0.05, abs=1e-6)
    assert int(rec['sg_all_atoms'][0]) == 1


def test_a_lattice_that_drops_the_declared_group_gives_different():
    """P-1 atoms on a b-unique monoclinic cell (beta 100 degrees), declared with a c-unique 2_1 (-x, -y, z + 1/2).
    The expanded cell holds the 2_1 in fractional coordinates, but the metric does not admit a 2-fold along c:
    detected P-1 {1, -1} against declared {1, 2_1}, neither containing the other."""
    cell6 = np.array([9.0, 10.0, 11.0, 90 * DEG, 100 * DEG, 90 * DEG])
    rng = np.random.default_rng(5)
    half = rng.random((6, 3)) * 0.4 + 0.05
    frac = np.concatenate([half, -half])
    screw_c = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1.0]])
    b = _batch([(cell6, frac, [6, 6, 6, 7, 8, 6] * 2)], sg=4, ops=[np.stack([np.eye(4), screw_c])])
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'ok', rec['message'][0]
    assert int(rec['sg_detected'][0]) == 2 and _relation(rec, 0) == 'different'
    assert (int(rec['n_ops_declared'][0]), int(rec['n_ops_detected'][0])) == (2, 2)


def test_relations_compare_translations_as_well_as_rotations():
    """The relation helper on operator lists: a 2-fold and a 2_1 screw along b share their rotation but not their
    translation (different); a 2-fold through x = 0 and one through x = 0.01 (0.2 A apart in translation) differ at
    symprec 1e-3 A; subgroup pairs are higher / lower, and unequal pairs with no containment are different."""
    T = _T(np.array([10.0, 12.0, 14.0, 90 * DEG, 105 * DEG, 90 * DEG]))

    def op(R, t):
        o = np.eye(4)
        o[:3, :3], o[:3, 3] = R, t
        return o

    E, two_y, inv, mir_y = (np.eye(3), np.diag([-1.0, 1, -1]), -np.eye(3), np.diag([1.0, -1, 1]))
    p2 = np.stack([op(E, 0), op(two_y, 0)])
    p21 = np.stack([op(E, 0), op(two_y, [0, 0.5, 0.5])])          # the 2_1 of SYM_OPS[14] (axis at z = 1/4)
    p2_shift = np.stack([op(E, 0), op(two_y, [0.02, 0, 0])])
    p21c = np.stack(SYM_OPS[14]).astype(float)
    p_1 = np.stack([op(E, 0), op(inv, 0)])
    p2m = np.stack([op(E, 0), op(two_y, 0), op(inv, 0), op(mir_y, 0)])
    rel = lambda g, d: csym._relation(g, d, T, 1e-3)
    assert rel(p2, p21) == 'different' and rel(p21, p2) == 'different'
    assert rel(p2_shift, p2) == 'different' and rel(p2, p2) == 'equal'
    assert rel(p21c, p21) == 'higher' and rel(p21, p21c) == 'lower'
    assert rel(p2m, p21) == 'different' and rel(p2, p21c) == 'different'
    assert rel(p_1, p21c) == 'lower' and rel(p2m, p2) == 'higher'


def test_other_exceptions_propagate(monkeypatch):
    """Only RuntimeError, ValueError and LinAlgError become find_symmetry_failed; anything else is a bug and
    raises."""
    b = _status_batch()

    def boom(*a, **k):
        raise TypeError('a bug')

    monkeypatch.setattr(csym, '_image_match', boom)
    with pytest.raises(TypeError, match='a bug'):
        find_symmetry(b, **TOL)


def test_a_row_that_raises_is_isolated(monkeypatch):
    b = _status_batch()
    plain = find_symmetry(b, **TOL)
    real = csym._image_match

    def boom(X, Zx, ops_in, T_fc):
        if len(X) > 10:                            # row 0 only: the other rows stop before the rebuild
            raise ValueError('boom')
        return real(X, Zx, ops_in, T_fc)

    monkeypatch.setattr(csym, '_image_match', boom)
    rec = find_symmetry(b, **TOL)
    assert _status(rec, 0) == 'find_symmetry_failed' and 'boom' in rec['message'][0]
    for i in range(1, 5):
        assert _status(rec, i) == _status(plain, i) and rec['message'][i] == plain['message'][i]
    # a raising row next to an ok one: the ok row's record is unchanged
    two = _batch([(_noisy_p1_of_pbar1(0.0)[0], np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]]), [6, 8])] +
                 [_noisy_p1_of_pbar1(0.0)])
    monkeypatch.undo()
    plain = find_symmetry(two, **TOL)
    monkeypatch.setattr(csym, '_image_match', boom)
    rec = find_symmetry(two, **TOL)
    assert (_status(rec, 0), _status(rec, 1)) == (_status(plain, 0), 'find_symmetry_failed')
    assert rec['message'][0] == plain['message'][0] and int(rec['sg_detected'][0]) == int(plain['sg_detected'][0])


def test_the_input_is_not_modified():
    b = _fixtures()
    store = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in b._store.items()}
    slices = {k: v.clone() for k, v in b._slice_dict.items()}
    find_symmetry(b, **TOL)
    find_symmetry(b, **TOL, rows=torch.ones(b.num_graphs, dtype=torch.bool))
    assert list(b._store.keys()) == list(store)
    for k, v in store.items():
        w = b._store[k]
        if torch.is_tensor(v):
            assert torch.equal(torch.nan_to_num(w, nan=7.0), torch.nan_to_num(v, nan=7.0)), k
        elif k == 'symmetry_operators':
            assert all(np.array_equal(x, y) for x, y in zip(w, v)), k
        elif k == 'file_bonds':
            assert all((x is None and y is None) or np.array_equal(x, y) for x, y in zip(w, v)), k
        else:
            assert w == v, k
    for k, v in slices.items():
        assert torch.equal(b._slice_dict[k], v), k


def test_a_posed_batch_gives_the_same_groups():
    """After find_molecules and parameterize_pose, pos holds each unit's image in the asymmetric unit; expanded by
    the same operators it is the same cell."""
    b0 = _fixtures()
    posed = b0.find_molecules(**FIND).parameterize_pose(**POSE)
    bare, rec = find_symmetry(b0, **TOL), find_symmetry(posed, **TOL)
    ran = [i for i in range(b0.num_graphs) if int(posed.status[i]) == 0]
    assert len(ran) >= 12
    for i in ran:
        assert _status(rec, i) == 'ok', (posed.identifier[i], rec['message'][i])
        for key in ('sg_detected', 'relation', 'n_ops_detected', 'n_cell_atoms', 'n_asym_heavy', 'n_asym_h'):
            assert int(rec[key][i]) == int(bare[key][i]), (posed.identifier[i], key)
    assert all(_status(rec, i) == 'not_run' for i in range(b0.num_graphs) if i not in ran)


def test_record_layout():
    b = _status_batch()
    rec = find_symmetry(b, **TOL)
    B = b.num_graphs
    assert rec['status_names'] == SYMMETRY_STATUS and rec['relation_names'] == RELATIONS
    assert rec['args'] == dict(symprec=1e-3, angle_tolerance=-1.0, rebuild_tol=1e-3, h_tol=1e-3, merge_tol=1e-2)
    shapes = dict(status=(B,), sg_declared=(B,), sg_detected=(B,), relation=(B,), n_ops_declared=(B,),
                  n_ops_detected=(B,), index=(B,), P=(B, 3, 3), p=(B, 3), std_cell=(B, 6), rebuild_dev=(B,),
                  h_dev=(B,), n_cell_atoms=(B,), n_asym_heavy=(B,), n_asym_h=(B,), merge_span=(B,),
                  sg_all_atoms=(B,))
    for key, shape in shapes.items():
        assert torch.is_tensor(rec[key]) and tuple(rec[key].shape) == shape, key
    for key in ('message', 'hm_detected', 'asym_frac', 'asym_z'):
        assert isinstance(rec[key], list) and len(rec[key]) == B, key
    assert rec['asym_frac'][0].shape == (len(rec['asym_z'][0]), 3)
    assert ((rec['asym_frac'][0] >= 0) & (rec['asym_frac'][0] < 1)).all()
    assert int(rec['n_asym_heavy'][0]) + int(rec['n_asym_h'][0]) == len(rec['asym_z'][0])


def test_bad_arguments_raise():
    b = _status_batch()
    with pytest.raises(TypeError):
        find_symmetry(b, symprec=1e-3)                                  # every tolerance is required
    for key in ('symprec', 'rebuild_tol', 'h_tol', 'merge_tol'):
        for bad in (0, -1e-3, float('nan'), float('inf'), True, '1e-3', None):
            with pytest.raises(ValueError):
                find_symmetry(b, **dict(TOL, **{key: bad}))
    for bad in (0, -2, float('nan'), True, None):
        with pytest.raises(ValueError):
            find_symmetry(b, **dict(TOL, angle_tolerance=bad))
    find_symmetry(b, **dict(TOL, angle_tolerance=5.0))
    for bad in (torch.ones(3, dtype=torch.bool), [7], [-1], [0.5]):
        with pytest.raises(ValueError):
            find_symmetry(b, **TOL, rows=bad)
    sub = b.select_rows([0])
    del sub['symmetry_operators']
    with pytest.raises(ValueError):
        find_symmetry(sub, **TOL)

"""
crystal_search.fd_refold and gradient_descent_optimization(fd_refold=...).

Contracts.
1. built_lattice reads the lattice the live builder makes: where the stated angles have F <= 0 the builder's T_fc is
   a left-handed basis; the re-read parameters (c -> -c, centre z -> -z) build the SAME atoms in the SAME Cartesian
   frame (live builder unit cells compared atom for atom, no rigid motion allowed).
2. refold_params keeps the crystal: the live builder's unit cell of the refolded row is the input's up to a proper
   rigid motion (all molecules, modulo the lattice; the lattices map onto each other by an integer unimodular
   matrix) within ATOM_TOL, and the eLJ agrees within max(ELJ_RTOL |E|, ELJ_ATOL), on normal triclinic rows, rows on
   the angle clamp, rows with F < 0, monoclinic rows on the clamp, and rows of the 12 groups where canonicalize_aunit
   raises. Control: the refolded row with its molecule turned 90 degrees about its body x axis is more than 0.1 A
   off (a flipped handedness is not a control in P-1: for a molecule close to a 2-fold axis it lands near the
   inversion image, 0.06 A off on KEQQON).
   Local data (skipped without D:/crystal_datasets): the P1 endpoints 18 and 24 of the 2026-10-26 optim_c1_rows run
   (F < 0) with their own molecule; eLJ against the energy that run stored.
3. A refold result is never itself a trigger (the termination argument), is inside the clean ranges with centres in
   [0, 1); results the next clean would change are refused and the row returned as it was.
4. fold_batch (the end of a stage) never raises, also where canonicalize_aunit raises (control: it does there).
5. gradient_descent_optimization: fd_refold=False is the default and gives exactly what omitting it gives (records,
   outputs, no new keys); fd_refold=True refolds triggered rows before step 0 is evaluated, with step 0's energy equal
   to the unrefolded crystal's (the off run's step 0); records and outputs carry the counts; the attempt cap holds;
   the refolded rows' optimizer state is dropped and the others' kept; wrong uses raise.

CPU only. The bit-identity of the off path against the code before this option was checked once against the
pre-change module (2026-10-26, outside this file); here it is pinned against omitting the argument.
"""
import math
from pathlib import Path

import pytest
import torch

from mxtaltools.crystal_search import crystal_opt_utils as cou
from mxtaltools.crystal_search import fd_refold as fr

F64 = torch.float64
HERE = Path(__file__).resolve().parent
CSD = HERE / 'datasets' / 'mini_new_csd.pt'
LOCAL_ROWS = Path('D:/crystal_datasets/fd_stress_oct26/optim_c1_rows/groups/sg_001_rows.pt')
LOCAL_MOL = Path('D:/crystal_datasets/fd_stress_oct26/c1_all/molecule.pt')
ATOM_TOL = 1e-3       # Angstrom; float32 builds agree to ~1e-5
ELJ_RTOL, ELJ_ATOL = 1e-3, 5e-3  # gate 4's (tests/fd/test_gate4_energy.py); float32 noise measured here up to
#                                  4.8e-4 relative (|eLJ| 28, sg 70, a 27 A, atoms 1.1e-4 A apart), typically < 3e-5
CONTROL = 0.1         # Angstrom: a different crystal is at least this far off
RAISING = (48, 50, 59, 68, 70, 85, 86, 126, 130, 133, 137, 141)
MOL_VOLUME, PACKING = 200.0, 0.5  # Angstrom^3 (KEQQON is about this size), the cell sizing of _case
NB = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3).to(F64)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def mol():
    """(z, pos) of KEQQON (13 atoms) from the mini CSD set."""
    if not CSD.exists():
        pytest.skip(f'{CSD} not present')
    rows = torch.load(CSD, weights_only=False)
    c = next(c for c in rows if c.identifier[0] == 'KEQQON')
    return c.z.clone(), c.pos.clone().float()


def _rows(mol, sg, params, hand):
    """Collated Z' = 1 MolCrystalData rows with these parameters (sg an int or one per row)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list
    z, pos = mol
    params = torch.as_tensor(params).to(torch.float32).reshape(-1, 12)
    hand = torch.as_tensor(hand).to(torch.float32).reshape(-1)
    sgs = [int(sg)] * len(params) if isinstance(sg, int) else [int(s) for s in sg]
    out = []
    for i in range(len(params)):
        m = MolData(z=z.clone(), pos=pos.clone(), x=torch.zeros(len(z), 1), do_mol_analysis=True)
        out.append(MolCrystalData(molecule=m, sg_ind=sgs[i], cell_lengths=params[i, :3].clone(),
                                  cell_angles=params[i, 3:6].clone(), aunit_centroid=params[i, 6:9].clone(),
                                  aunit_orientation=params[i, 9:12].clone(), aunit_handedness=hand[i:i + 1].clone(),
                                  do_box_analysis=True))
    return collate_data_list(out)


def _unit_cells(b):
    """Per row (T_fc float64, Cartesian molecules [k, n, 3] float64) from the live builder (mol2ucell, std
    orientation, as analyze builds)."""
    b = b.clone()
    with torch.no_grad():
        b.mol2ucell(std_orientation=True)
    pos, ub = b.unit_cell_pos.double(), b.unit_cell_batch
    return [(b.T_fc[i].double(), pos[ub == i].reshape(int(b.sym_mult[i]), int(b.num_atoms[i]), 3))
            for i in range(b.num_graphs)]


def _kabsch(P, Q):
    """Proper rotation R minimising sum |R P_i - Q_i| for centred [n, 3] P, Q (atoms in correspondence)."""
    U, _, Vt = torch.linalg.svd(P.T @ Q)
    d = float(torch.sign(torch.linalg.det(Vt.T @ U.T)))
    return Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=F64)) @ U.T


def _congruence(uc_a, uc_b):
    """Distance (Angstrom) between crystal a and crystal b moved onto it by the best proper rigid motion.

    The motion is the Kabsch rotation Q of one molecule of b onto molecule 0 of a (atoms by index: a description
    change never relabels atoms), with the translation between their centroids. Two parts, the larger returned:
    atoms, over every atom of b the distance to the same atom of its best-matching molecule of a modulo a's lattice
    (nearest image); lattice, Q must carry b's lattice onto a's: M = T_a^-1 Q T_b rounded to the integer matrix N
    with |det N| = 1 (else inf), and the largest column norm of Q T_b - T_a N. With one molecule per cell (P1) the
    atom part is zero for ANY orientation and only the lattice part sees a turned molecule. The best over b's
    molecules is returned."""
    Ta, A = uc_a
    Tb, Bm = uc_b
    k, n, _ = A.shape
    Tai = torch.linalg.inv(Ta)
    fa = (A.reshape(-1, 3) @ Tai.T).reshape(k, n, 3)
    a0, ca0 = A[0], A[0].mean(0)
    best = float('inf')
    for j in range(Bm.shape[0]):
        bj = Bm[j]
        Q = _kabsch(bj - bj.mean(0), a0 - ca0)
        s = ca0 - Q @ bj.mean(0)
        N = torch.round(Tai @ Q @ Tb)
        if abs(abs(float(torch.linalg.det(N))) - 1) > 1e-6:
            continue
        lat = float((Q @ Tb - Ta @ N).norm(dim=0).max())
        fy = ((Bm.reshape(-1, 3) @ Q.T + s) @ Tai.T).reshape(-1, n, 3)
        df = fy[:, None] - fa[None]
        df = df - torch.round(df)
        dist = torch.einsum('ij,bknsj->bknsi', Ta, df[..., None, :] + NB).norm(dim=-1).amin(-1)  # [kb, ka, n]
        best = min(best, max(lat, float(dist.amax(-1).amin(1).amax())))
    return best


def _elj(b):
    with torch.no_grad():
        return b.analyze(['elj'], cutoff=10, supercell_size=10)['elj'].double().reshape(-1)


def _elj_close(e, ref):
    return (e - ref).abs() <= torch.maximum(torch.full_like(ref, ELJ_ATOL), ELJ_RTOL * ref.abs())


def _random_rotvecs(n, g):
    v = torch.randn(n, 3, generator=g, dtype=F64)
    return v / v.norm(dim=1, keepdim=True) * (0.2 + 2.6 * torch.rand(n, 1, generator=g, dtype=F64))


def _triclinic_angles(n, g, kind):
    """[n, 3] angles inside the clamp: 'normal' (70-110 deg), 'clamp' (one angle exactly on 0.2 pi or 0.8 pi,
    F > 0.1), 'unreal' (F < 0, angles 40-140 deg). Rejection, bounded."""
    out = []
    for _ in range(2000):
        if kind == 'normal':
            a = torch.deg2rad(70 + 40 * torch.rand(3, generator=g, dtype=F64))
        else:
            a = torch.deg2rad(40 + 100 * torch.rand(3, generator=g, dtype=F64))
            if kind == 'clamp':
                i = int(torch.randint(3, (1,), generator=g))
                a[i] = (0.2 if bool(torch.rand(1, generator=g) < 0.5) else 0.8) * math.pi
        F = float(fr.volume_factor(a[None])[0])
        if (kind == 'unreal' and F < 0) or (kind != 'unreal' and F > 0.1):
            out.append(a)
        if len(out) == n:
            return torch.stack(out)
    raise RuntimeError(f'could not draw {n} {kind} angle triples')


def _case(kind, n=6, seed=0):
    """(sg list, params [n, 12] float64, hand [n]) for one row family. Cell lengths around the edge at which
    sym_mult molecules of MOL_VOLUME fill the cell to a packing coefficient of PACKING (times 0.9-1.2, and
    F^(-1/6) for skewed cells): denser cells overflow the pair search's per-atom neighbour cap (1000), which
    truncates differently in different descriptions of one crystal."""
    g = torch.Generator().manual_seed(seed)
    from mxtaltools.common.geometry_utils import enforce_crystal_system
    from mxtaltools.constants.space_group_info import SYM_OPS
    if kind in ('normal', 'clamp', 'unreal'):
        sg = [1 + (i % 2) for i in range(n)]
        A = _triclinic_angles(n, g, kind)
    elif kind == 'mono_clamp':
        sg = [(4, 14, 15, 7, 9, 5)[i % 6] for i in range(n)]
        beta = torch.where(torch.arange(n) % 2 == 0, 0.2 * math.pi, 0.8 * math.pi).to(F64)
        A = torch.stack([torch.full((n,), math.pi / 2, dtype=F64), beta, torch.full((n,), math.pi / 2, dtype=F64)], 1)
    elif kind == 'raising':
        sg = [RAISING[i % len(RAISING)] for i in range(n)]
        A = torch.full((n, 3), math.pi / 2, dtype=F64)
    else:
        raise ValueError(kind)
    edge = torch.tensor([(len(SYM_OPS[s]) * MOL_VOLUME / PACKING) ** (1 / 3) for s in sg], dtype=F64)
    skew = fr.volume_factor(A).abs().clamp(min=0.25) ** (-1 / 6)
    L = (edge * skew)[:, None] * (0.9 + 0.3 * torch.rand(n, 3, generator=g, dtype=F64))
    L, A = enforce_crystal_system(L, A, torch.tensor(sg))
    t = torch.rand(n, 3, generator=g, dtype=F64)
    rv = _random_rotvecs(n, g)
    hand = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(F64)
    return sg, torch.cat([L, A, t, rv], 1), hand


# ---------------------------------------------------------------------------
# 1. the built lattice
# ---------------------------------------------------------------------------

def test_built_lattice_rereads_a_left_handed_basis_in_the_same_frame(mol):
    sg, p, h = _case('unreal', n=8, seed=1)
    p = p.float().double()  # the values a float32 row holds
    from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
    L1, A1, f1, flipped, ok = fr.built_lattice(p[:, :3], p[:, 3:6], p[:, 6:9])
    assert bool(ok.all()) and bool(flipped.all())
    T = batch_compute_fractional_transform(p[:, :3], p[:, 3:6])[0]
    assert bool((torch.linalg.det(T) < 0).all())
    D = torch.tensor([1.0, 1.0, -1.0], dtype=F64)
    T1 = batch_compute_fractional_transform(L1, A1)[0]
    assert torch.allclose(T1, T * D, atol=1e-9), 'the re-read cell must build the flipped basis itself'
    assert torch.allclose(f1, p[:, 6:9] * D)
    assert bool((fr.volume_factor(A1) > 0).all())
    # the live builder: identical atoms in the identical frame (no rigid motion allowed)
    b0 = _rows(mol, sg, p, h)
    q = torch.cat([L1, A1, fr.wrap_unit(f1), p[:, 9:12]], 1)
    b1 = _rows(mol, sg, q, h)
    for (Ta, A), (Tb, B) in zip(_unit_cells(b0), _unit_cells(b1)):
        assert torch.allclose(Ta * D, Tb, atol=1e-4)
        fa, fb = A.reshape(-1, 3) @ torch.linalg.inv(Ta).T, B.reshape(-1, 3) @ torch.linalg.inv(Ta).T
        df = fa - fb
        df = df - torch.round(df)
        assert float((df @ Ta.T).norm(dim=-1).max()) < ATOM_TOL
    assert bool(_elj_close(_elj(b1), _elj(b0)).all())


def test_built_lattice_returns_a_real_cell_unchanged():
    sg, p, h = _case('normal', n=8, seed=2)
    L1, A1, f1, flipped, ok = fr.built_lattice(p[:, :3], p[:, 3:6], p[:, 6:9])
    assert bool(ok.all()) and not bool(flipped.any())
    assert torch.allclose(L1, p[:, :3], atol=1e-10) and torch.allclose(A1, p[:, 3:6], atol=1e-10)
    assert torch.equal(f1, p[:, 6:9])


# ---------------------------------------------------------------------------
# 2. the refold keeps the crystal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['normal', 'clamp', 'unreal', 'mono_clamp', 'raising'])
def test_refold_keeps_the_crystal(mol, kind):
    n = 12 if kind == 'raising' else 6
    sg, p, h = _case(kind, n=n, seed=10 + len(kind))
    b0 = _rows(mol, sg, p, h)
    res = fr.batch_refold(b0)
    names = [fr.REFOLD_STATUS[int(s)] for s in res.status]
    assert names == ['applied'] * n, names
    assert bool((res.flipped == (kind == 'unreal')).all())
    q = res.params
    assert bool(((q[:, 6:9] >= 0) & (q[:, 6:9] < 1)).all()), 'centres must be wrapped into [0, 1)'
    assert bool((q[:, 11] >= 0).all()), 'rotation vectors on the +z hemisphere'
    lp, ap = fr.clean_defaults()
    assert bool(fr.in_clean_ranges(q[:, :3], q[:, 3:6], q[:, 9:12], lp, ap).all())
    b1 = _rows(mol, sg, q, res.handedness)
    for i, (ua, ub) in enumerate(zip(_unit_cells(b0), _unit_cells(b1))):
        dev = _congruence(ua, ub)
        assert dev < ATOM_TOL, (kind, i, sg[i], dev)
    e0, e1 = _elj(b0), _elj(b1)
    assert bool(_elj_close(e1, e0).all()), (e0, e1)
    # control: the molecule turned 90 degrees about its body x axis is another crystal, which the check must see
    from mxtaltools.crystal_fd.so3 import matrix_to_rotvec, rotvec_to_matrix
    rx = rotvec_to_matrix(torch.tensor([[math.pi / 2, 0.0, 0.0]], dtype=F64))
    qc = q.clone()
    qc[:, 9:12] = matrix_to_rotvec(rotvec_to_matrix(q[:, 9:12]) @ rx, hemisphere=True)
    bc = _rows(mol, sg, qc, res.handedness)
    devs = [_congruence(ua, ub) for ua, ub in zip(_unit_cells(b0), _unit_cells(bc))]
    assert min(devs) > CONTROL, devs


def test_refold_moves_clamped_rows_inside_and_never_retriggers(mol):
    """The termination argument's premise, on many rows of every triggering family: an applied refold's angles are
    not a trigger (tri_niggli angles in [60, 120] deg with F >= 1/2; monoclinic beta in [90, 135] deg)."""
    params, sgs, hands = [], [], []
    for kind, seed in (('clamp', 3), ('unreal', 4), ('mono_clamp', 5)):
        sg, p, h = _case(kind, n=60, seed=seed)
        params.append(p)
        sgs += sg
        hands.append(h)
    p, h = torch.cat(params), torch.cat(hands)
    assert bool(fr.refold_triggers(p.float()[:, 3:6]).all()), 'every case must start on a trigger'
    res = fr.refold_params(sgs, p[:, :3].float(), p[:, 3:6].float(), p[:, 6:9].float(), p[:, 9:12].float(), h)
    ok = res.applied
    assert int(ok.sum()) >= 0.9 * len(p), [fr.REFOLD_STATUS[int(s)] for s in res.status]
    assert not bool(fr.refold_triggers(res.params[ok, 3:6].float()).any())
    tri = torch.tensor(sgs) <= 2
    A = torch.rad2deg(res.params[:, 3:6])
    assert bool(((A[ok & tri] >= 60 - 1e-6) & (A[ok & tri] <= 120 + 1e-6)).all())
    assert bool((fr.volume_factor(res.params[ok & tri, 3:6]) >= 0.5 - 1e-9).all())
    mono = ok & ~tri
    assert bool(((A[mono, 1] >= 90 - 1e-6) & (A[mono, 1] <= 135 + 1e-6)).all())
    # refused rows come back as they were
    assert torch.equal(res.params[~ok].float(), p[~ok].float())


def test_results_the_next_clean_would_change_are_refused(mol):
    """a = b = 3 A at gamma = 0.2 pi: the reduced cell has a lattice vector of 1.85 A, which clean's 3 A floor would
    lift, so the refold is refused and the row (and, at the end of a stage, its wrapped centre) is kept."""
    p = torch.tensor([[3.0, 3.0, 8.0, math.pi / 2, math.pi / 2, 0.2 * math.pi, 0.3, 0.6, 1.2, 0.5, -0.4, -1.1]],
                     dtype=F64)
    res = fr.refold_params([1], p[:, :3], p[:, 3:6], p[:, 6:9], p[:, 9:12], torch.ones(1))
    assert fr.REFOLD_STATUS[int(res.status[0])] == 'outside_clean_range'
    assert torch.equal(res.params, p)
    b = _rows(mol, 1, p, torch.ones(1))
    status = cou.fd_fold_aunit(b)
    assert fr.REFOLD_STATUS[int(status[0])] == 'outside_clean_range'
    assert torch.allclose(b.cell_lengths.double(), p[:, :3]) and torch.allclose(b.cell_angles.double(), p[:, 3:6])
    assert torch.allclose(b.aunit_centroid.double(), fr.wrap_unit(p[:, 6:9]), atol=1e-7)
    assert float(b.aunit_orientation[0, 2]) >= 0


def test_nonstandard_setting_is_refused(mol):
    import numpy as np
    sg, p, h = _case('mono_clamp', n=6, seed=6)
    b = _rows(mol, sg, p, h)
    ops = [np.array(o, copy=True) for o in b.symmetry_operators]
    k = sg.index(14)
    ops[k][1, :3, 3] = ops[k][1, :3, 3] + np.array([0.0, 0.0, 0.5])  # the c glide's translation moved: not SYM_OPS[14]
    b.symmetry_operators = ops
    res = fr.batch_refold(b)
    assert fr.REFOLD_STATUS[int(res.status[k])] == 'nonstandard_setting'
    assert all(fr.REFOLD_STATUS[int(s)] == 'applied' for i, s in enumerate(res.status) if i != k)


@pytest.mark.skipif(not (LOCAL_ROWS.exists() and LOCAL_MOL.exists()), reason='local fd_stress data not present')
def test_local_p1_endpoints_with_f_below_zero():
    d = torch.load(LOCAL_ROWS, weights_only=False)
    m = torch.load(LOCAL_MOL, weights_only=False)['mol']
    idx = [18, 24]
    p, h = d['end'][idx], d['end_hand'][idx]
    assert bool((fr.volume_factor(p[:, 3:6]) < 0).all()), 'these rows were picked for F < 0'
    mol = (m.z.clone(), m.pos.clone().float())
    b0 = _rows(mol, 1, p, h)
    res = fr.batch_refold(b0)
    assert bool(res.applied.all()) and bool(res.flipped.all())
    # row 18 as traced on 2026-10-26: stated c 10.37 A, alpha 36.0 deg; built 11.93 A, 45.3 deg (flipped: 134.7)
    L1, A1, *_ = fr.built_lattice(b0.cell_lengths[:1], b0.cell_angles[:1], b0.aunit_centroid[:1])
    assert abs(float(L1[0, 2]) - 11.93) < 0.01 and abs(float(torch.rad2deg(A1[0, 0])) - (180 - 45.3)) < 0.1
    b1 = _rows(mol, 1, res.params, res.handedness)
    for ua, ub in zip(_unit_cells(b0), _unit_cells(b1)):
        dev = _congruence(ua, ub)
        assert dev < ATOM_TOL, dev
    assert bool(_elj_close(_elj(b1), d['elj_end'][idx].double()).all())


# ---------------------------------------------------------------------------
# 3-4. triggers and the end-of-stage fold
# ---------------------------------------------------------------------------

def test_triggers():
    lo, hi = 0.2 * math.pi, 0.8 * math.pi
    A = torch.tensor([[math.pi / 2] * 3,                      # cube: no
                      [math.pi / 2, hi, math.pi / 2],          # monoclinic on the clamp: yes
                      [math.pi / 2, hi - 0.03 * math.pi, math.pi / 2],  # 5.4 deg inside: no
                      [math.pi / 2, hi - 0.02 * math.pi, math.pi / 2],  # 3.6 deg inside: yes
                      [lo + 0.01, math.pi / 2, math.pi / 2],    # near the low bound: yes
                      [math.pi / 2, math.pi / 2, 2 * math.pi / 3],  # hexagonal gamma: no
                      [math.radians(130.7), math.radians(112.7), math.radians(113.2)],  # F = 0.071: yes
                      [math.radians(36.0), math.radians(64.6), math.radians(107.9)]])   # F < 0 (and clamp): yes
    assert fr.refold_triggers(A).tolist() == [False, True, False, True, True, False, True, True]


def test_fold_batch_never_raises_where_canonicalize_aunit_does(mol):
    sg, p, h = _case('raising', n=36, seed=7)
    raised = set()
    for i in range(len(sg)):  # control: the old end-of-stage call raises in each of these groups
        x = _rows(mol, sg[i:i + 1], p[i:i + 1], h[i:i + 1])
        try:
            cou.canonicalize_aunit(x)
        except ValueError:
            raised.add(sg[i])
    assert raised == set(RAISING), sorted(set(RAISING) - raised)
    b0 = _rows(mol, sg, p, h)
    b = b0.clone()
    status = cou.fd_fold_aunit(b)
    assert bool((status == fr.APPLIED).all()), status
    c = b.aunit_centroid.double()
    assert bool(((c >= 0) & (c < 1)).all()) and bool((b.aunit_orientation[:, 2] >= 0).all())
    assert bool(_elj_close(_elj(b), _elj(b0)).all())
    # the folded rows are crystal_fd canonical (+1 sheet where G has an improper element)
    from mxtaltools.crystal_fd import canon
    from mxtaltools.crystal_fd.description import CrystalDescription
    d = CrystalDescription.from_crystal_params(b.sg_ind, b.cell_lengths, b.cell_angles, b.aunit_centroid,
                                               b.aunit_orientation, b.aunit_handedness)
    assert bool(canon.fd_status(d).decided.all())


# ---------------------------------------------------------------------------
# 5. the optimiser
# ---------------------------------------------------------------------------

def _start_batch(mol):
    """sg 1 rows: two on the clamp (raw angle past it), two with F < 0, two normal."""
    _, pc, hc = _case('clamp', n=2, seed=20)
    _, pu, hu = _case('unreal', n=2, seed=21)
    _, pn, hn = _case('normal', n=2, seed=22)
    p = torch.cat([pc, pu, pn])
    p[0, 3:6] = torch.where((p[0, 3:6] - 0.8 * math.pi).abs() < 1e-9, p[0, 3:6] + 0.05, p[0, 3:6])  # stuck past
    p[0, 3:6] = torch.where((p[0, 3:6] - 0.2 * math.pi).abs() < 1e-9, p[0, 3:6] - 0.05, p[0, 3:6])
    return _rows(mol, 1, p, torch.cat([hc, hu, hn])), torch.tensor([True, True, True, True, False, False])


def _opt(batch, steps=4, **kw):
    kw.setdefault('centroid_boundary', 'wrap')
    return cou.gradient_descent_optimization(batch.full_cell_parameters(), batch, optimizer_func='rprop',
                                             init_lr=0.01, max_num_steps=steps, optim_target='elj', cutoff=10,
                                             convergence_eps=-1.0, show_tqdm=False, **kw)


@pytest.fixture
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)  # CPU scatter sums are thread-order dependent; bit comparisons need one thread
    yield
    torch.set_num_threads(n)


def test_off_is_the_default_and_identical(mol, one_thread):
    import inspect
    assert inspect.signature(cou.gradient_descent_optimization).parameters['fd_refold'].default is False
    b, _ = _start_batch(mol)
    out_a, rec_a = _opt(b.clone())
    out_b, rec_b = _opt(b.clone(), fd_refold=False)
    assert set(rec_a) == set(rec_b) and 'fd_refold' not in rec_a
    for k in rec_a:
        assert torch.equal(rec_a[k], rec_b[k]), k
    for x, y in zip(out_a, out_b):
        assert torch.equal(x.full_cell_parameters(), y.full_cell_parameters())
        assert torch.equal(x.aunit_handedness, y.aunit_handedness) and torch.equal(x.elj, y.elj)
        assert 'fd_refolds' not in x.keys() and 'fd_fold_status' not in x.keys()


def test_on_refolds_before_step_0_and_keeps_its_energy(mol):
    b, trig = _start_batch(mol)
    _, rec_off = _opt(b.clone(), steps=2)
    out, rec = _opt(b.clone(), steps=6, fd_refold=True)
    assert rec['fd_refold'].shape == rec['loss'].shape
    assert rec['fd_refold'][0].tolist() == trig.tolist(), rec['fd_refold'][0]
    assert torch.equal(rec['fd_refold'].sum(0), torch.tensor([int(o.fd_refolds) for o in out]))
    assert bool((rec['fd_refold'] <= rec['fd_refold_tried']).all())
    # step 0 evaluates the refolded description of the same crystal: the energy of the unrefolded start
    assert bool(_elj_close(rec['elj'][0].double(), rec_off['elj'][0].double()).all()), (rec['elj'][0], rec_off['elj'][0])
    assert not torch.equal(rec['params'][0][trig], rec_off['params'][0][trig])
    A = rec['params'][0, trig, 3:6].double()
    assert not bool(fr.refold_triggers(A).any()), 'step 0 of a refolded row is off every trigger'
    assert all(int(o.fd_fold_status) == fr.APPLIED for o in out)
    from mxtaltools.dataset_utils.utils import collate_data_list
    c = collate_data_list(out)
    assert bool(((c.aunit_centroid >= 0) & (c.aunit_centroid < 1)).all()) and bool((c.aunit_orientation[:, 2] >= 0).all())
    # the returned energy is the returned row's
    assert bool(_elj_close(_elj(_rows(mol, 1, c.full_cell_parameters(), c.aunit_handedness)), c.elj.double()).all())


def test_attempt_cap_bounds_refolds_per_row(mol, monkeypatch):
    monkeypatch.setattr(fr, 'REFOLD_MAX_ATTEMPTS', 2)
    monkeypatch.setattr(fr, 'refold_triggers', lambda angles, *a, **k: torch.ones(len(angles), dtype=torch.bool))
    b, _ = _start_batch(mol)
    out, rec = _opt(b, steps=7, fd_refold=True)
    assert rec['loss'].shape[0] == 6, 'the step cap ends the stage'
    assert rec['fd_refold_tried'].sum(0).tolist() == [2] * 6
    assert rec['fd_refold_tried'][:2].all() and not rec['fd_refold_tried'][2:].any()


def test_refold_carries_only_the_refolded_rows_rprop_state(mol):
    b, trig = _start_batch(mol)
    b.clean_cell_parameters(mode='hard', canonicalize_orientations=False)  # the step's cleaned batch
    pm = cou.CrystalParams(b.full_cell_parameters().detach())
    opt = cou.init_opt(0.01, 'rprop', pm)
    for _ in range(3):  # adapt the step sizes away from lr
        opt.zero_grad()
        loss = sum((p ** 3).sum() for p in pm.parameters())
        loss.backward()
        opt.step()
    state0 = {i: {k: v.clone() for k, v in opt.state[p].items() if torch.is_tensor(v)} for i, p in enumerate(pm.params)}
    before = [p.detach().clone() for p in pm.params]
    attempts = torch.zeros(6, dtype=torch.long)
    applied, tried = cou._fd_refold_rows(b, pm, opt, None, attempts)
    assert applied.tolist() == trig.tolist() and tried.tolist() == trig.tolist()
    assert attempts.tolist() == trig.long().tolist()
    for i, p in enumerate(pm.params):
        st, s0 = opt.state[p], state0[i]
        assert torch.equal(p.detach(), before[i]) != bool(trig[i])
        if not trig[i]:  # untouched
            assert all(torch.equal(st[k], s0[k]) for k in s0)
            continue
        assert not bool(st['prev'].any()), 'previous gradient forgotten'
        ss, ss0 = st['step_size'].view(-1), s0['step_size'].view(-1)
        for lo, hi in cou.REFOLD_STATE_BLOCKS:  # each block at the median of its learned sizes, never reset to lr
            assert bool((ss[lo:hi] == ss0[lo:hi].median()).all())


def test_carry_drops_state_of_other_optimizers():
    p = torch.nn.Parameter(torch.ones(12))
    opt = torch.optim.Adam([p], lr=0.1)
    p.sum().backward()
    opt.step()
    cou._carry_optimizer_state(opt, p)
    assert p not in opt.state or len(opt.state[p]) == 0


def test_convergence_record_removes_the_refold_jump(mol, monkeypatch):
    """convergence='params': a refold at step k shifts the row's convergence-record history by the jump: the record read by
    check_convergence has the same step-to-step changes as records['params'] everywhere except at k, where the jump
    is gone; records['params'] keeps the true descriptions."""
    seen = {}
    real = cou.check_convergence

    def spy(record, s_ind, *a, **k):
        seen['rec'] = record[:s_ind].clone()
        return real(record, s_ind, *a, **k)
    monkeypatch.setattr(cou, 'check_convergence', spy)
    calls = {'n': 0}

    def fire_at_step_3(angles, *a, **k):  # every row refolds at step 3 only, mid-walk
        calls['n'] += 1
        return torch.full((len(angles),), calls['n'] == 4, dtype=torch.bool)
    monkeypatch.setattr(fr, 'refold_triggers', fire_at_step_3)
    b, _ = _start_batch(mol)
    out, rec = _opt(b, steps=52, fd_refold=True, convergence='params')
    assert rec['fd_refold'][3].any() and not rec['fd_refold'][:3].any() and not rec['fd_refold'][4:].any()
    conv = seen['rec']
    T = conv.shape[0]
    params = rec['params'][:T]
    steps_refolded = rec['fd_refold'][:T]
    assert bool(steps_refolded.any())
    d_conv, d_par = conv.diff(dim=0), params.diff(dim=0)
    jump_at = steps_refolded[1:]  # [T-1, n]: the difference into a refolded step
    assert torch.allclose(d_conv[~jump_at], d_par[~jump_at], atol=1e-4)
    big = (d_par - d_conv).abs().amax(-1)[jump_at]
    assert bool((big > 1e-3).any()), 'the true record jumps where the convergence record does not'
    assert torch.equal(conv[-1], params[-1]), 'the latest step is the true description'


def test_wrong_uses_raise(mol):
    b, _ = _start_batch(mol)
    with pytest.raises(ValueError, match='wrap'):
        _opt(b.clone(), centroid_boundary='clamp', fd_refold=True)
    zp2 = b.clone()
    zp2.z_prime = torch.full_like(zp2.z_prime, 2)
    with pytest.raises(ValueError, match="Z' = 1"):
        _opt(zp2, fd_refold=True)


def test_run_search_refuses_fd_refold_where_the_box_is_needed():
    from types import SimpleNamespace
    from mxtaltools.crystal_search.run_search import check_fd_refold_stages
    wrap = dict(centroid_boundary='wrap')
    check_fd_refold_stages([dict(wrap, fd_refold=True), dict(wrap)], None)  # the campaign's wrap -> wrap: fine
    check_fd_refold_stages([dict(fd_refold=False), dict()], {'dir': 'x'})  # option off: untouched
    with pytest.raises(ValueError, match='clamp-mode'):
        check_fd_refold_stages([dict(wrap, fd_refold=True), dict()], None)
    with pytest.raises(ValueError, match='clamp-mode'):
        check_fd_refold_stages([SimpleNamespace(**wrap, fd_refold=True), SimpleNamespace(centroid_boundary='clamp')], None)
    with pytest.raises(ValueError, match='campaign'):
        check_fd_refold_stages([dict(wrap, fd_refold=True)], {'dir': 'x'})

"""
crystal_search/standardize.py: lattice-only re-standardisation to the reduced cell, and
common/geometry_utils.py::rotmat2rotvec_stable, which it relies on.

Contracts.
1. rotmat2rotvec_stable inverts rotvec2rotmat for every rotation, including those at and near the identity and a rotation
   by pi (rotmat2rotvec replaces those by pi about (1, 1, 1)).
2. keeps_operators is the setting group: for sg 14 (P 1 21/c 1) an ac-plane basis change keeps SYM_OPS[14] exactly when
   its r entry (c' = r a + s c) is even, with no origin shift; for Cc and C2/c (sg 9, 15) every change with even q does,
   an odd r needing the origin shift b/4 (setting_shift); any unimodular change keeps P1 and P-1.
3. apply_basis_change re-describes a crystal without moving an atom, and refuses a basis change outside the setting group
   and a crystal whose own operators are not SYM_OPS[sg] (a P21/n or P21/a description of sg 14).
4. A crystal re-described in a skewed setting-group basis standardises to the same reduced cell as the original, with
   zero margin-0 penalty and every atom where it was (up to the returned origin shift): acridine sg14 Z'=2, and CSD
   crystals in monoclinic (all setting classes present), triclinic and orthorhombic groups; a molecule oriented within
   1e-5 rad of a rotation by pi survives. choose_basis finds a reduced cell for every random monoclinic lattice.
5. A crystal that is already reduced keeps its parameters exactly.
6. A crystal with no zero-penalty cell, or in a nonstandard setting, is reported, never passed through silently.

CPU only.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mxtaltools.common.geometry_utils import rotmat2rotvec_stable
from mxtaltools.crystal_search.standardize import apply_basis_change, choose_basis, keeps_operators, \
    same_crystal_deviation, setting_shift, standardize_cells
from mxtaltools.dataset_utils.utils import collate_data_list

HERE = Path(__file__).resolve().parent
ACRIDINE = HERE / 'datasets' / 'mini_acridine.pt'
CSD = HERE.parent / 'mini_datasets' / 'mini_new_csd.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']
ATOM_TOL = 1e-4  # Angstrom; observed <= 2e-5 (float32 parameters)


def _strip(c):
    c = c.clone()
    for k in DROP:
        if k in c.keys():
            delattr(c, k)
    return c


def _load(path, keep):
    if not path.exists():
        pytest.skip(f'{path} not present')
    return [_strip(c) for c in torch.load(path, weights_only=False) if keep(c)]


def _acridine():
    return _load(ACRIDINE, lambda c: int(c.sg_ind) == 14 and int(c.z_prime) == 2)


def _csd(sgs):
    return _load(CSD, lambda c: int(c.sg_ind) in sgs and not bool(c.nonstandard_symmetry))


def _acridine_cc():
    return _load(ACRIDINE, lambda c: int(c.sg_ind) == 9)


def _random_setting_matrix(sg, rng, lengths, angles, n_factors=2):
    """A product of random small basis changes that keep SYM_OPS[sg], det +1, whose skewed cell the search itself
    could hold (every angle within clean_cell_parameters' 36-144 deg, no length over 4x the longest original)."""
    from mxtaltools.crystal_search.standardize import cell_from_metric, metric
    if sg > 15:
        return np.eye(3, dtype=np.int64)  # higher systems: the penalty fixes only the metric, nothing to skew within it
    G = metric(np.asarray(lengths, float), np.asarray(angles, float))
    for attempt in range(20000):
        N = np.eye(3, dtype=np.int64)
        for _ in range(n_factors if attempt < 10000 else 1):  # some cells admit only a single-factor skew
            if sg <= 2:
                M = rng.integers(-1, 2, size=(3, 3))
            else:
                p, q, r, s = rng.integers(-2, 3, size=4)
                M = np.array([[p, 0, r], [0, p * s - q * r, 0], [q, 0, s]])
            if round(np.linalg.det(M)) != 1 or not keeps_operators(M, sg):
                break
            N = N @ M
        else:
            if (N == np.eye(3, dtype=np.int64)).all():
                continue
            L, A = cell_from_metric(N.T @ G @ N)
            if np.degrees(A).min() > 37 and np.degrees(A).max() < 143 and L.max() < 4 * max(lengths):
                return N
    raise RuntimeError('no admissible skew found')


# ---------------------------------------------------------------------------
# 1-2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('where', ['near_pi', 'near_identity', 'random'])
def test_rotmat2rotvec_stable_inverts_every_rotation(dtype, where):
    rng = np.random.default_rng(0)
    axis = rng.normal(size=(600, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    eps = np.repeat([0.0, 1e-9, 1e-7, 1e-5, 3e-4, 1e-3], 100)
    angle = {'near_pi': math.pi - eps, 'near_identity': eps, 'random': rng.uniform(0, math.pi, 600)}[where]
    R = torch.tensor(Rotation.from_rotvec(axis * angle[:, None]).as_matrix(), dtype=dtype)
    rv = rotmat2rotvec_stable(R)
    assert rv.dtype == dtype
    back = Rotation.from_rotvec(rv.double().numpy()).as_matrix()
    err = np.abs(back - R.double().numpy()).max()
    assert err < (1e-6 if dtype == torch.float32 else 1e-12), f'max matrix error {err:.2e}'


def test_setting_group_of_sg14_is_the_even_r_rule_and_triclinic_keeps_everything():
    for p in range(-2, 3):
        for q in range(-2, 3):
            for r in range(-2, 3):
                for s in range(-2, 3):
                    d = p * s - q * r
                    if abs(d) != 1:
                        continue
                    N = np.array([[p, 0, r], [0, d, 0], [q, 0, s]])
                    assert keeps_operators(N, 14) == (r % 2 == 0), (p, q, r, s)
                    if r % 2 == 0:
                        assert not setting_shift(N, 14).any()
                    assert keeps_operators(N, 2) and keeps_operators(N, 1)
                    for sg in (9, 15):  # centred: q even; an odd r moves the glide plane by b/4
                        assert keeps_operators(N, sg) == (q % 2 == 0), (sg, p, q, r, s)
                        if q % 2 == 0:
                            o = setting_shift(N, sg)
                            assert (o[1] == 0.25) == (r % 2 == 1), (sg, p, q, r, s, o)


@pytest.mark.parametrize('sg', list(range(3, 16)))
def test_choose_basis_reaches_a_reduced_cell_for_every_monoclinic_lattice(sg):
    rng = np.random.default_rng(sg)
    for _ in range(100):
        a, b, c = rng.uniform(4, 25, 3)
        beta = np.radians(rng.uniform(90, 135))
        N, L, A, nz = choose_basis(sg, [a, b, c], [np.pi / 2, beta, np.pi / 2])
        assert N is not None, (sg, a, b, c, np.degrees(beta))


# ---------------------------------------------------------------------------
# 3-5
# ---------------------------------------------------------------------------

def _check_round_trip(crystals, seed=0):
    rng = np.random.default_rng(seed)
    orig = collate_data_list(crystals)
    sgs = orig.sg_ind.reshape(-1).numpy()
    ref, ref_info = standardize_cells(orig)
    Ns = np.stack([_random_setting_matrix(int(g), rng, orig.cell_lengths[i].double().numpy(),
                                          orig.cell_angles[i].double().numpy()) for i, g in enumerate(sgs)])
    Os = np.stack([setting_shift(N, int(g)) for N, g in zip(Ns, sgs)])
    skewed = apply_basis_change(orig, Ns)
    assert same_crystal_deviation(orig, skewed, Ns, Os).max() < ATOM_TOL, 'the skew itself moved atoms'
    std, info = standardize_cells(skewed)
    total = Ns @ info['N']
    total_o = Os + np.einsum('nij,nj->ni', Ns, info['origin'])  # x_std = (N1 N2)^-1 (x - o1 - N1 o2)
    dev = same_crystal_deviation(orig, std, total, total_o)
    assert dev.max() < ATOM_TOL, f'standardisation moved atoms by up to {dev.max():.2e} A'
    assert float(std.compute_cell_reduction_penalty().max()) < 1e-6, 'output not in the reduced domain'
    assert torch.allclose(std.cell_lengths.double(), ref.cell_lengths.double(), atol=1e-4), \
        'a skewed description must reduce to the same cell as the original'
    assert torch.allclose(std.cell_angles.double(), ref.cell_angles.double(), atol=1e-5)
    return info


def test_acridine_zp2_round_trip():
    crystals = _acridine()
    info = _check_round_trip(crystals * 4)
    assert info['changed'].any(), 'the skews should have needed a basis change'


def test_acridine_cc_round_trip_needs_the_origin_shift():
    crystals = _acridine_cc()
    if not crystals:
        pytest.skip('no Cc crystal in mini_acridine')
    info = _check_round_trip(crystals * 6, seed=9)
    assert info['changed'].any()
    # the review's case: c' = a + c (r odd) is a valid description only with the shift b/4
    orig = collate_data_list(crystals[:1])
    N = np.array([[[1, 0, 1], [0, 1, 0], [0, 0, 1]]])
    assert setting_shift(N[0], 9).tolist() == [0.0, 0.25, 0.0]
    skewed = apply_basis_change(orig, N)
    assert same_crystal_deviation(orig, skewed, N, [[0.0, 0.25, 0.0]]).max() < ATOM_TOL
    std, info = standardize_cells(skewed)
    assert info['ok'].all() and info['changed'].all()
    total_o = np.array([[0.0, 0.25, 0.0]]) + np.einsum('nij,nj->ni', N, info['origin'])
    assert same_crystal_deviation(orig, std, N @ info['N'], total_o).max() < ATOM_TOL
    assert float(std.compute_cell_reduction_penalty().max()) < 1e-6


@pytest.mark.parametrize('sgs', [(14,), (4,), (9,), (1, 2), (18, 19, 33, 60, 61)],
                         ids=['P21c_glide', 'P21_free', 'Cc_centred', 'triclinic', 'orthorhombic'])
def test_csd_round_trip(sgs):
    crystals = _csd(sgs)
    if not crystals:
        pytest.skip(f'no crystal in {sgs}')
    _check_round_trip(crystals * 2, seed=int(sgs[0]))


def test_orientation_near_pi_survives_standardisation():
    c = _acridine()[0].clone()
    ori = c.aunit_orientation.clone().reshape(-1)
    ax = np.array([0.3, 0.5, 0.81])
    ax /= np.linalg.norm(ax)
    ori[0:3] = torch.tensor(ax * (math.pi - 1e-5), dtype=ori.dtype)  # molecule 0 a hair short of a half-turn
    c.aunit_orientation = ori.reshape(c.aunit_orientation.shape)
    _check_round_trip([c] * 3, seed=5)


@pytest.mark.parametrize('source', ['acridine', 'csd'])
def test_already_reduced_crystals_keep_their_parameters_exactly(source):
    b = collate_data_list(_acridine() if source == 'acridine' else _csd((4, 14, 19, 61)))
    assert float(b.compute_cell_reduction_penalty().max()) < 1e-10, 'setup: these are reduced'
    s, info = standardize_cells(b)
    assert not info['changed'].any()
    assert torch.equal(s.full_cell_parameters(), b.full_cell_parameters())
    assert torch.equal(s.aunit_handedness, b.aunit_handedness)


def test_apply_basis_change_refuses_a_change_outside_the_setting_group():
    b = collate_data_list(_acridine()[:1])
    with pytest.raises(ValueError, match='not in the setting group'):
        apply_basis_change(b, np.array([[[1, 0, 1], [0, -1, 0], [0, 0, 1]]]))  # c' = a + c: r odd


def _nonstandard_sg14():
    from mxtaltools.crystal_search.standardize import standard_setting
    return _load(CSD, lambda c: int(c.sg_ind) == 14 and not standard_setting(c.symmetry_operators, 14))


def test_a_nonstandard_setting_is_refused_not_rebuilt_as_another_crystal():
    crystals = _nonstandard_sg14()
    if not crystals:
        pytest.skip('no P21/n or P21/a crystal in mini_new_csd')
    b = collate_data_list(crystals[:1])
    with pytest.raises(ValueError, match='nonstandard setting'):  # a' = a + c keeps SYM_OPS[14], not P21/n
        apply_basis_change(b, np.array([[[1, 0, 0], [0, 1, 0], [1, 0, 1]]]))
    b2 = collate_data_list([crystals[0], _csd((14,))[0]])
    with pytest.raises(ValueError, match='nonstandard setting'):
        standardize_cells(b2)
    s, info = standardize_cells(b2, on_failure='flag')
    assert info['nonstandard'].tolist() == [True, False] and info['ok'].tolist() == [False, True]
    assert torch.equal(s.full_cell_parameters()[0], b2.full_cell_parameters()[0])


# ---------------------------------------------------------------------------
# 6
# ---------------------------------------------------------------------------

def test_a_crystal_with_no_reduced_cell_is_reported():
    c = _csd((19,))[0].clone()
    ang = c.cell_angles.clone()
    ang.reshape(-1)[1] = ang.reshape(-1)[1] + 0.2  # an orthorhombic cell with beta != 90: no cell in its domain
    c.cell_angles = ang
    b = collate_data_list([c, _csd((19,))[0]])
    with pytest.raises(ValueError, match='no zero-penalty cell'):
        standardize_cells(b)
    s, info = standardize_cells(b, on_failure='flag')
    assert info['ok'].tolist() == [False, True]
    assert torch.equal(s.full_cell_parameters()[0], b.full_cell_parameters()[0]), 'a failed crystal is left as it was'

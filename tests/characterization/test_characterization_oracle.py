"""Stage 0: the cluster builder and eLJ against an independent float64 brute-force lattice sum.

The oracle (``charlib.oracle_pairs``) takes an asymmetric unit and the crystal's own operators and
cell, generates every symmetry image and every lattice translation that can bring an atom within
the cutoff, and lists every intermolecular pair closer than the cutoff, in float64. It shares no
code with ``mol2cluster`` / ``build_cluster`` / ``construct_radial_graph``.

Conventions:
- a pair counts when d < cutoff, strictly: the radial graph's convention (torch_cluster.radius
  drops points at exactly r; pinned by ``test_radial_graph_drops_points_at_exactly_r``);
- pairs with |d - cutoff| < ``charlib.ORACLE_BAND`` (2e-4 A) are boundary pairs: the float32
  builder may put them on either side, so they are left out of both sets and both energies;
- sorted per-atom distances must agree within ``charlib.ORACLE_DIST_TOL`` (1e-3 A; the builder
  poses and images in float32), partner elements exactly.

Three checks per crystal, cutoff and std_orientation, with the builder's own posed asymmetric unit
(``mol2ucell``) as the oracle's reference (so they test that the cluster is COMPLETE):
1. pair set: the builder's (target atom, partner) pairs equal the oracle's, in every case. Until
   stage 5 (2026-10-23) this was a strict xfail on 9 cases, where the builder missed pairs (audit
   #16: the reach test compared a heavy-atom radius with all-atom centroids, so a hydrogen beyond
   the radius could be cut); the all-atom, per-unit reach flipped every one of them
   (``EXPECTED_MISSING_PAIRS`` is now empty; the cases and their counts are in the README's
   regeneration log).
2. eLJ, float64 re-sum: the library's pair function over the builder's own pairs, in float64, equals
   the oracle's sum within 2e-6 of sum|e| (complete cases measured <= 6.4e-7; float32 positions).
   Xfail only where the missing pairs carry enough energy to show (a single 10 A pair does not);
   none since stage 5.
3. eLJ as analyze() returns it (float32), boundary pairs removed: within 2e-4 of sum|e|. Passes even
   where a few far pairs are missing; it catches gross errors (radii, scaling, double counting).

And, for the ingestion rows (F3, and AFEVOA as F4a's Z'=2 row), with the STORED crystal atoms as the
reference, so that they test that the build REPRODUCES the crystal: each posed unit must be a
symmetry / lattice image of its stored unit, atom for atom; and the builder's pair set must equal
the oracle's built from the stored atoms.

Every check calls ``pytest.fail`` on its own failure, and the strict xfails expect exactly that
(``raises=pytest.fail.Exception``), so an exception from the library or the harness is reported as
a failure, never taken for the known one.
"""
import numpy as np
import pytest
import torch

import charlib

FAILED = pytest.fail.Exception   # what pytest.fail raises

#: (crystal set, row) -> rows built by the builder in one batch
ORACLE_SETS = {
    'F1': ((True, False), 4),
    'F3': ((True,), 2),
    'F3b': ((True, False), 2),
    'F4a': ((True,), 2),
    'F4b': ((True,), 2),
    # two compounds of equal atom count per row (ethanol + dimethyl ether; pyridone + fluorobenzene): units of
    # different elements and reaches (added at stage 5, whose re-pin moved their analyze pins)
    'two_eth_ether': ((True, False), 1),
    'two_pyr_flb': ((True, False), 1),
    'phen_p1': ((True, False), 1),
    'phen_p21c': ((True, False), 1),
    'pyr_p21c': ((True, False), 1),
    'phen_probe1b': ((True, False), 1),
    'pyr_zp2': ((True, False), 1),
}

_R16 = ('audit #16: the cluster reach test uses the heavy-atom radius against all-atom centroids, so '
        'hydrogens reaching past it can be cut; flips at stage 5 (per-unit all-atom reach)')

#: (set, row, std_orientation, cutoff): the builder misses pairs. Empty since stage 5 (2026-10-23), which flipped
#: the 9 cases measured on 2026-10-22: HIFQUJ (F1 row 3, sg 14) 5 of 10852 at 10 A, std True and False; phenacetin
#: P-1 1 of 10022 at 10 A; phenacetin P21/c 4 of 2305 (std True) and 4 of 2331 (std False) at 6 A; pyridone P21/c
#: 2 of 4995 at 10 A (std True), 1 of 966 at 6 A and 2 of 4955 at 10 A (std False); phenacetin probe1b the 5.81 A
#: H...H pair at 6 A. A case added here is a regression to fix, not a known failure.
EXPECTED_MISSING_PAIRS = {}

#: the subset whose missing pairs move the float64 eLJ beyond 2e-6 of sum|e| (before stage 5: 8 of the 9 above;
#: phenacetin P-1's one pair was worth 1.1e-8 of sum|e|). Complete cases measured <= 6.4e-7.
EXPECTED_ELJ64_OFF = {}

ELJ64_REL_TOL = 2e-6
ELJ32_REL_TOL = 2e-4

PARAMS = [(name, g, std, cutoff) for name, (stds, n) in ORACLE_SETS.items() for g in range(n)
          for std in stds for cutoff in charlib.CUTOFFS]


def _id(p):
    name, g, std, cutoff = p
    return f'{name}-row{g}-std{int(std)}-cut{int(cutoff)}'


def _batch(name):
    if name in charlib.CASES:
        return charlib.build_case(name)
    return charlib.collate([charlib.synthetic_row(charlib.fixtures()['AUDIT'][name])])


_CACHE = {}


def _measure(name, std, cutoff):
    """Builder and oracle results for every crystal of a set (cached: deterministic, read-only)."""
    key = (name, std, cutoff)
    if key not in _CACHE:
        b = _batch(name)
        posed = b.clone()
        posed.mol2ucell(std_orientation=std)
        res, cl = b.clone().analyze(['elj'], cutoff=cutoff, std_orientation=std, return_cluster=True)
        rows = []
        for g in range(b.num_graphs):
            units, zs = charlib.crystal_units(posed, g)
            oracle = charlib.oracle_pairs(units, zs, posed.T_fc[g], posed.symmetry_operators[g], cutoff)
            builder = charlib.builder_pairs(cl, g, torch.cat(units))
            rows.append({'oracle': oracle, 'builder': builder, 'z': torch.cat(zs), 'elj32': float(res['elj'][g]),
                         'posed': posed})
        _CACHE[key] = rows
    return _CACHE[key]


def _marks(table, p):
    return [pytest.mark.xfail(strict=True, raises=FAILED, reason=f'{table[p]}; {_R16}')] if p in table else []


def _check_pair_sets(cmp):
    if cmp['missing'] or cmp['extra']:
        pytest.fail(f'builder has {cmp["n_builder"]} pairs, brute force {cmp["n_oracle"]}: {cmp["missing"]} missing, '
                    f'{cmp["extra"]} extra; per target atom (index, oracle - builder, unmatched oracle distances): '
                    f'{cmp["count_mismatch_atoms"][:6]}')
    if cmp['distance_mismatch']:
        pytest.fail(f'sorted distances disagree beyond tol: {cmp["distance_mismatch"][:6]}')
    if cmp['element_mismatch']:
        pytest.fail(f'partner elements disagree at target atoms {cmp["element_mismatch"][:6]}')


@pytest.mark.parametrize('name,g,std,cutoff', [pytest.param(*p, marks=_marks(EXPECTED_MISSING_PAIRS, p), id=_id(p))
                                               for p in PARAMS])
def test_pair_set_matches_brute_force(name, g, std, cutoff):
    r = _measure(name, std, cutoff)[g]
    _check_pair_sets(charlib.compare_pair_sets(r['builder'], r['oracle'], cutoff))


@pytest.mark.parametrize('name,g,std,cutoff', [pytest.param(*p, marks=_marks(EXPECTED_ELJ64_OFF, p), id=_id(p))
                                               for p in PARAMS])
def test_elj_float64_resum_matches_brute_force(name, g, std, cutoff):
    r = _measure(name, std, cutoff)[g]
    e_b, _ = charlib.interior_energy(r['builder']['targets'], r['z'], cutoff, library=True)
    e_o, scale = charlib.interior_energy(r['oracle']['targets'], r['z'], cutoff)
    if abs(e_b - e_o) > ELJ64_REL_TOL * scale:
        pytest.fail(f'eLJ over the builder pairs {e_b:.6f}, brute force {e_o:.6f}: |diff| {abs(e_b - e_o):.3e} '
                    f'> {ELJ64_REL_TOL} x sum|e| {scale:.1f}')


@pytest.mark.parametrize('name,g,std,cutoff', [pytest.param(*p, id=_id(p)) for p in PARAMS])
def test_analyze_elj_matches_brute_force_at_float32(name, g, std, cutoff):
    r = _measure(name, std, cutoff)[g]
    e_route = r['elj32'] - charlib.boundary_energy(r['builder']['targets'], r['z'], cutoff)
    e_o, scale = charlib.interior_energy(r['oracle']['targets'], r['z'], cutoff)
    if abs(e_route - e_o) > ELJ32_REL_TOL * scale:
        pytest.fail(f"analyze(['elj']) {r['elj32']:.5f} (boundary pairs removed: {e_route:.5f}), brute force {e_o:.5f}")


def test_oracle_finds_what_a_widened_cluster_finds():
    """Control on the oracle itself: on the audit's phenacetin P-1 case the cluster built with a 4 A
    wider selection (which can only add molecules) finds exactly the oracle's 10022 pairs at 10 A."""
    b = _batch('phen_p1')
    c = b.clone()
    c.pose_aunit(std_orientation=True)
    c.build_unit_cell()
    cl = c.build_cluster(cutoff=10.0, zp_buffer=4.0)
    cl.construct_radial_graph(cutoff=10.0)
    units, zs = charlib.crystal_units(c, 0)
    oracle = charlib.oracle_pairs(units, zs, c.T_fc[0], c.symmetry_operators[0], 10.0)
    cmp = charlib.compare_pair_sets(charlib.builder_pairs(cl, 0, torch.cat(units)), oracle, 10.0)
    assert cmp['n_oracle'] == 10022 and cmp['missing'] == 0 and cmp['extra'] == 0, cmp


# ------------------------------------------------------------------------------------------------
# the stored crystal as the reference: does the build reproduce the crystal the ingestion chain read?
# ------------------------------------------------------------------------------------------------

#: (set, row in the built batch, F3 fixture record): rows whose atoms are the ingestion chain's stored crystal
STORED_ROWS = [('F3', 0, 0), ('F3', 1, 1), ('F4a', 1, 1)]


def _stored_units(rec):
    """The stored (ingestion) asymmetric unit of an F3 record, split by its stored ``mol_ind``, float64."""
    mi = rec['mol_ind']
    units = [rec['pos'][mi == u].double() for u in range(int(rec['z_prime']))]
    zs = [rec['z'][mi == u] for u in range(int(rec['z_prime']))]
    # the stored units are contiguous blocks in mol_ind order (find_molecules' layout), as the builder assumes
    assert torch.equal(mi, torch.arange(len(units)).repeat_interleave(torch.tensor([len(u) for u in units])))
    return units, zs


def _image_deviation(stored, posed, T_fc, ops):
    """min over operators k and lattice vectors n of max_atom |op_k(stored) + n - posed| (order-matched, A)."""
    T_fc = T_fc.double()
    T_cf = torch.linalg.inv(T_fc)
    ops = torch.as_tensor(np.asarray(ops), dtype=torch.float64)
    fs, fp = stored @ T_cf.T, posed.double() @ T_cf.T
    best = float('inf')
    for k in range(len(ops)):
        img = fs @ ops[k, :3, :3].T + ops[k, :3, 3]
        n = torch.round((fp - img).mean(0))
        best = min(best, float(((img + n - fp) @ T_fc.T).norm(dim=1).max()))
    return best


@pytest.mark.parametrize('name,g,i', STORED_ROWS, ids=[f'{n}-row{g}' for n, g, _ in STORED_ROWS])
def test_posed_units_are_images_of_stored_units(name, g, i):
    """mol2ucell(std_orientation=True) poses each unit from its slot; the result must be the stored unit itself
    or one of its symmetry / lattice images, atom for atom in stored order, within 1e-4 A (float32 build)."""
    rec = charlib.fixtures()['F3'][i]
    b = charlib.build_case(name)
    b.mol2ucell(std_orientation=True)
    posed, _ = charlib.crystal_units(b, g)
    stored, _ = _stored_units(rec)
    devs = [_image_deviation(s, p, b.T_fc[g], b.symmetry_operators[g]) for s, p in zip(stored, posed)]
    if max(devs) >= 1e-4:
        pytest.fail(f'{rec["identifier"]}: posed unit vs nearest image of its stored unit, max per-atom deviation '
                    f'{[f"{d:.2e}" for d in devs]} A')


_STORED_PARAMS = [(n, g, i, c) for n, g, i in STORED_ROWS for c in charlib.CUTOFFS]


@pytest.mark.parametrize('name,g,i,cutoff', _STORED_PARAMS,
                         ids=[f'{n}-row{g}-cut{int(c)}' for n, g, i, c in _STORED_PARAMS])
def test_pair_set_matches_brute_force_from_stored_atoms(name, g, i, cutoff):
    """The builder's pairs equal those of the crystal the ingestion chain stored (oracle built from the stored
    atoms, not from the build). Targets are matched by unit and atom index, which the image test above
    establishes (each atom's environment is the same in every image of its unit)."""
    rec = charlib.fixtures()['F3'][i]
    r = _measure(name, True, cutoff)[g]
    units, zs = _stored_units(rec)
    oracle = charlib.oracle_pairs(units, zs, r['posed'].T_fc[g], rec['ops'], cutoff)
    _check_pair_sets(charlib.compare_pair_sets(r['builder'], oracle, cutoff))


# ------------------------------------------------------------------------------------------------
# the radial graph's boundary convention, which the oracle's d < cutoff copies
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('r', [5.0, 6.0, 10.0])
def test_radial_graph_drops_points_at_exactly_r(r):
    """asymmetric_radius_graph (torch_cluster.radius underneath) keeps d < r and drops d == r exactly, in
    float32; the oracle's strict d < cutoff and its 2e-4 A boundary band rest on this. A change of
    torch_cluster or of the radial graph that moves the boundary shows here, by name."""
    from torch_cluster import radius
    from mxtaltools.models.functions.radial_graph import asymmetric_radius_graph
    just_inside = float(torch.nextafter(torch.tensor(r), torch.tensor(0.0)))
    x = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [0.0, r, 0.0], [0.0, 0.0, r], [just_inside, 0.0, 0.0],
                      [0.0, 0.0, -just_inside]])
    e = asymmetric_radius_graph(x, r=r, inside_inds=torch.tensor([0]), convolve_inds=torch.arange(len(x)),
                                batch=torch.zeros(len(x), dtype=torch.long))
    assert sorted(e[0].tolist()) == [4, 5] and e[1].tolist() == [0, 0], e
    raw = radius(x, x[:1], r, max_num_neighbors=10)
    assert sorted(raw[1].tolist()) == [0, 4, 5], raw

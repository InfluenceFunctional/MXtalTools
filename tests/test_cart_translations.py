"""
crystal_building.utils.get_cart_translations: the lattice translations a cluster is built from.

Contract. A translation t = T_fc n (n integer) is returned for a crystal iff it passes the keep test or n = 0, for
EVERY integer n, whatever basis the cell is written in and whatever else is in the batch:
- molecule test (``mol_centroids`` given, as ucell2cluster passes them): some unit-cell molecule k, moved by t, has
  its centroid within rho = 2 r + cutoff + 0.1 of the asymmetric unit's centre cc;
- cell test (no centroids): |cc - t| < 2 r + cutoff + 0.1 + D, D the cell's farthest corner.
Within a crystal the order is ascending |m|_1, m the indices in a reduced basis of the lattice, zero first.

Before 2026-10-18 the integer range was one cube for the whole batch in the given basis, bounded by
R / |lattice vector| and capped by ``supercell_size``. For an oblique cell the lattice planes are much closer than
the vector is long, so translations the test keeps were never examined: a MIPCAS crystal written in a basis with a
17 deg angle needed indices up to 84 on one axis and got 5, and its eLJ moved by 63 raw units with its batch-mates.
``_legacy`` reproduces that search and must fail the completeness checks (control).

Brute force enumerates in the reduced basis the cases were built from (T = T0 N, N unimodular), so it stays small.
CPU only, synthetic cells; a few seconds and well under 1 GB.
"""
import itertools
import math

import numpy as np
import pytest
import torch

import mxtaltools.crystal_building.utils as CBU
from mxtaltools.crystal_building.utils import get_cart_translations

F32 = torch.float32
BAND = 1e-3   # Angstrom: translations this close to a sphere may fall either side through float32 rounding


def _cell(a, b, c, al, be, ga):
    """Fractional-to-Cartesian matrix (columns = lattice vectors), a along x, b in the xy plane."""
    al, be, ga = (math.radians(x) for x in (al, be, ga))
    ca, cb, cg, sg = math.cos(al), math.cos(be), math.cos(ga), math.sin(ga)
    cy = (ca - cb * cg) / sg
    return torch.tensor([[a, b * cg, c * cb], [0, b * sg, c * cy], [0, 0, c * math.sqrt(1 - cb ** 2 - cy ** 2)]],
                        dtype=torch.float64)


def _unimodular(rng, span):
    while True:
        M = rng.integers(-span, span + 1, (3, 3))
        if round(np.linalg.det(M)) == 1:
            return torch.as_tensor(M, dtype=torch.float64)


def _cases(seed=0, n_reduced=5, n_skew=5):
    """dict of T [B,3,3] (float32), T0 and N (float64, T = T0 N), cc [B,3], radius [B], and per crystal 2-4 molecule
    centroids inside the cell (mol [sum Z, 3], Z [B]; molecule 0 is cc). Skewed rows (the last n_skew) are the reduced
    lattices again, in bases that are products of two unimodular matrices with entries in [-2, 2]."""
    rng = np.random.default_rng(seed)
    T0s, Ns = [], []
    for i in range(n_reduced + n_skew):
        if i < n_reduced:
            a, b, c = sorted(rng.uniform(4, 12, 3))
            T0s.append(_cell(a, b, c, *rng.uniform(80, 110, 3)))
            Ns.append(torch.eye(3, dtype=torch.float64))
        else:
            T0s.append(T0s[i % n_reduced])
            Ns.append(_unimodular(rng, 2) @ _unimodular(rng, 2))
    T0, N = torch.stack(T0s), torch.stack(Ns)
    T = T0 @ N
    B = len(T0s)
    cc = torch.einsum('bij,bj->bi', T, torch.as_tensor(rng.uniform(0, 1, (B, 3))))
    Z = torch.as_tensor(rng.integers(2, 5, B))
    mol = torch.cat([torch.einsum('ij,kj->ki', T[b], torch.as_tensor(rng.uniform(0, 1, (int(Z[b]), 3))))
                     for b in range(B)])
    mol[torch.cumsum(Z, 0) - Z] = cc                                  # molecule 0 is the asymmetric unit itself
    radius = torch.as_tensor(rng.uniform(2.5, 4.0, B))
    return dict(T=T.to(F32), T0=T0, N=N, cc=cc.to(F32), radius=radius.to(F32), mol=mol.to(F32), Z=Z)


def _cell_rho(T, radius, cutoff):
    corners = torch.tensor(list(itertools.product((0, 1), repeat=3))[1:], dtype=T.dtype)
    diag = torch.einsum('ki,nji->nkj', corners, T).norm(dim=-1).amax(1)
    return 2 * radius + cutoff + 0.1 + diag


def _brute_force(c, cutoff, molecules):
    """Per crystal (sure, either): the integer n (indices of T) passing the test by more than BAND (or n = 0), and those
    within BAND of the boundary; enumerated in T0's basis around each sphere, float64."""
    T, T0, N = c['T'].double(), c['T0'], c['N']
    if molecules:
        rho = 2 * c['radius'].double() + cutoff + 0.1
        ptr = (torch.cumsum(c['Z'], 0) - c['Z']).tolist()
        centres = [c['cc'][b].double() - c['mol'][ptr[b]:ptr[b] + int(c['Z'][b])].double() for b in range(len(T))]
    else:
        rho = _cell_rho(T, c['radius'].double(), cutoff)
        centres = [c['cc'][b].double()[None] for b in range(len(T))]
    out = []
    for b in range(len(T)):
        T0cf = torch.linalg.inv(T0[b])
        half = (rho[b] * T0cf.square().sum(-1).sqrt()).ceil().long() + 2
        sure, either = {(0, 0, 0)}, set()
        Ninv = torch.linalg.inv(N[b])
        for p in centres[b]:
            m0 = T0cf @ p
            grid = torch.stack(torch.meshgrid(*[torch.arange(int(math.floor(m0[i])) - int(half[i]),
                                                             int(math.ceil(m0[i])) + int(half[i]) + 1)
                                                for i in range(3)], indexing='ij'), -1).reshape(-1, 3)
            dist = (p - grid.double() @ T0[b].T).norm(dim=-1)
            n = (grid.double() @ Ninv.T).round().long()
            sure |= {tuple(x) for x in n[dist < rho[b] - BAND].tolist()}
            either |= {tuple(x) for x in n[(dist - rho[b]).abs() <= BAND].tolist()}
        out.append((sure, either))
    return out


def _run(c, cutoff, molecules, **kw):
    extra = dict(mol_centroids=c['mol'], mols_per_crystal=c['Z']) if molecules else {}
    return get_cart_translations(c['cc'], c['T'], c['radius'], cutoff, **extra, **kw)


def _indices(T, translations, counts):
    """Per crystal, the list of integer n of the returned Cartesian translations, in order."""
    Tcf = torch.linalg.inv(T.double())
    per = torch.split(translations.double(), counts.tolist())
    return [[tuple(int(v) for v in row) for row in torch.einsum('ij,tj->ti', Tcf[b], p).round().long().tolist()]
            for b, p in enumerate(per)]


def _legacy(cc_centroids, T_fc, mol_radii, cutoff, supercell_size=9, rho=None, **_):
    """The pre-2026-10-18 search: one cube for the batch, bound through the lattice-vector length, capped. Given
    ``rho`` (ucell2cluster passes its all-atom reach since 2026-10-23), the cell test's molecule distance is ``rho``
    in place of ``2 * mol_radii + cutoff + 0.1``."""
    cell_vectors = T_fc.permute(0, 2, 1)
    corners = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                           dtype=T_fc.dtype)
    cell_diag = torch.einsum('ki,nij->nkj', corners, cell_vectors).norm(dim=-1).amax(1)
    mol_distance = mol_radii * 2 + cutoff + 0.1 if rho is None else torch.as_tensor(rho, dtype=T_fc.dtype)
    cutoff_distance = mol_distance + cell_diag
    s = min(supercell_size, int(torch.ceil(cutoff_distance[:, None] / torch.linalg.norm(cell_vectors, dim=-1)).amax()))
    frac = CBU.generate_sorted_fractional_translations(s)
    cart = torch.einsum('nij, mi -> nmj', cell_vectors, frac)
    keep = (cc_centroids[:, None, :] - cart).square().sum(-1) < cutoff_distance[:, None].square()
    keep[:, 0] = True
    return cart[keep], keep.sum(1)


@pytest.fixture
def uncapped(monkeypatch):
    """The exact search without the per-crystal cap or the search shrink (the cell test in the skewed bases here keeps
    more than MAX_CELLS_PER_CRYSTAL cells; both limits are tested on their own below)."""
    monkeypatch.setattr(CBU, 'MAX_CELLS_PER_CRYSTAL', 10 ** 8)
    monkeypatch.setattr(CBU, 'MAX_SEARCH_CANDIDATES_PER_CRYSTAL', 2 ** 26)


@pytest.mark.parametrize('molecules', [True, False], ids=['molecule-test', 'cell-test'])
@pytest.mark.parametrize('cutoff', [6.0, 10.0])
def test_every_kept_translation_is_found(cutoff, molecules, uncapped):
    c = _cases()
    tr, counts = _run(c, cutoff, molecules, supercell_size=10)
    got = _indices(c['T'], tr, counts)
    for b, (sure, either) in enumerate(_brute_force(c, cutoff, molecules)):
        assert len(set(got[b])) == len(got[b]), f'crystal {b}: a translation is returned twice'
        missing, extra = sure - set(got[b]), set(got[b]) - sure - either
        assert not missing and not extra, f'crystal {b}: {len(missing)} missing, {len(extra)} extra'


def test_molecule_test_does_not_grow_with_obliqueness():
    """The cell test's radius includes D, the farthest corner of the cell AS WRITTEN; in an oblique basis that admits
    many cells holding nothing within reach. The molecule test does not depend on the basis beyond the molecules'
    own placement in the cell."""
    c = _cases()
    _, mol_counts = _run(c, 10.0, True)
    _, cell_counts = _run(c, 10.0, False)
    assert int(cell_counts[5:].sum()) > 3 * int(mol_counts[5:].sum()), (cell_counts.tolist(), mol_counts.tolist())


def test_legacy_range_misses_translations_in_skewed_bases():
    """Control: the old search loses cell-test translations on the skewed rows."""
    c = _cases()
    tr, counts = _legacy(c['cc'], c['T'], c['radius'], 10.0, supercell_size=10)
    got = _indices(c['T'], tr, counts)
    lost = [b for b, (sure, _) in enumerate(_brute_force(c, 10.0, False)) if sure - set(got[b])]
    assert any(b >= 5 for b in lost), 'the control no longer exercises the skewed-basis gap'


@pytest.mark.parametrize('molecules', [True, False], ids=['molecule-test', 'cell-test'])
def test_result_does_not_depend_on_batch_mates_or_supercell_size(molecules):
    c = _cases()
    tr, counts = _run(c, 10.0, molecules)
    batch = _indices(c['T'], tr, counts)
    ptr = (torch.cumsum(c['Z'], 0) - c['Z']).tolist()
    for b in range(len(c['T'])):
        one = {k: c[k][b:b + 1] for k in ('T', 'cc', 'radius', 'Z')}
        one['mol'] = c['mol'][ptr[b]:ptr[b] + int(c['Z'][b])]
        for s in (1, 10, 50):
            t1, c1 = _run(one, 10.0, molecules, supercell_size=s)
            assert _indices(one['T'], t1, c1)[0] == batch[b], f'crystal {b} alone (supercell_size {s}) differs'
    perm = torch.randperm(len(c['T']), generator=torch.Generator().manual_seed(1))
    shuffled = {k: c[k][perm] for k in ('T', 'cc', 'radius', 'Z')}
    shuffled['mol'] = torch.cat([c['mol'][ptr[b]:ptr[b] + int(c['Z'][b])] for b in perm.tolist()])
    tr_p, counts_p = _run(shuffled, 10.0, molecules)
    got = _indices(shuffled['T'], tr_p, counts_p)
    for i, b in enumerate(perm.tolist()):
        assert got[i] == batch[b], f'crystal {b} differs after shuffling'


def test_order_is_zero_first_then_ascending_l1_in_the_reduced_basis():
    c = _cases()
    tr, counts = _run(c, 10.0, True)
    reduced, _ = CBU._reduce_lattice_bases(c['T'].double())
    for b, t in enumerate(torch.split(tr, counts.tolist())):
        m = torch.linalg.solve(reduced[b], t.double().T).T.round().long()
        assert bool((m[0] == 0).all()), f'crystal {b}: zero translation not first'
        l1 = m.abs().sum(1)
        assert bool((l1[1:] >= l1[:-1]).all()), f'crystal {b}: not ordered by |m|_1 in the reduced basis'


def test_chunking_does_not_change_the_result(monkeypatch):
    c = _cases()
    tr, counts = _run(c, 10.0, True)
    monkeypatch.setattr(CBU, 'TRANSLATION_CANDIDATES_PER_PASS', 500)
    tr2, counts2 = _run(c, 10.0, True)
    assert torch.equal(counts, counts2) and torch.equal(tr, tr2)


def test_reduction_sweep_limit_does_not_change_the_result(monkeypatch, uncapped):
    """Exact after any number of reduction sweeps; only the search boxes grow (the order may follow the basis)."""
    c = _cases()
    want = [set(x) for x in _indices(c['T'], *_run(c, 10.0, True))]
    monkeypatch.setattr(CBU, 'BASIS_REDUCTION_SWEEPS', 0)
    got = [set(x) for x in _indices(c['T'], *_run(c, 10.0, True))]
    assert got == want


def test_scalar_and_tensor_cutoffs_agree():
    c = _cases()
    tr, counts = _run(c, 10.0, True)
    tr2, counts2 = _run(c, torch.full((len(c['T']),), 10.0), True)
    assert torch.equal(counts, counts2) and torch.equal(tr, tr2)


def test_empty_batch():
    tr, counts = get_cart_translations(torch.zeros(0, 3), torch.zeros(0, 3, 3), torch.zeros(0), 10.0)
    assert tr.shape == (0, 3) and counts.shape == (0,)


def test_strongly_oblique_basis_of_an_ordinary_cell_is_searched_cheaply():
    """An ordinary cell in a basis so oblique that a box in that basis would hold 10^8 candidates: still exact, and the
    molecule test keeps what the same molecules keep in the reduced basis."""
    T0 = _cell(6.0, 7.0, 8.0, 95.0, 100.0, 105.0)
    N = torch.tensor([[1.0, 4.0, 9.0], [2.0, 9.0, 21.0], [3.0, 13.0, 31.0]], dtype=torch.float64)
    assert round(float(torch.linalg.det(N))) == 1
    cc = (torch.tensor([[0.3, 0.6, 0.2]], dtype=torch.float64) @ T0.T).float()
    mol = torch.cat([cc, cc + torch.tensor([[1.5, -2.0, 0.7]])])
    c = dict(T=(T0 @ N)[None].float(), T0=T0[None], N=N[None], cc=cc, radius=torch.tensor([3.0]), mol=mol,
             Z=torch.tensor([2]))
    tr, counts = _run(c, 10.0, True)
    sure, either = _brute_force(c, 10.0, True)[0]
    got = set(_indices(c['T'], tr, counts)[0])
    assert not (sure - got) and not (got - sure - either)
    c0 = dict(c, T=T0[None].float(), N=torch.eye(3, dtype=torch.float64)[None])
    _, counts0 = _run(c0, 10.0, True)
    assert abs(int(counts[0]) - int(counts0[0])) <= 2, (int(counts[0]), int(counts0[0]))


def test_translations_carry_the_cell_gradient():
    """The crystal search differentiates energies through T_fc: every returned translation is T_fc n with n integer, and
    its gradient with respect to T_fc must be exactly that."""
    c = _cases()
    T = c['T'][[0, 7]].double().requires_grad_(True)
    tr, counts = get_cart_translations(c['cc'][[0, 7]].double(), T, c['radius'][[0, 7]].double(), 10.0)
    assert tr.requires_grad
    n = torch.linalg.solve(T.detach().repeat_interleave(counts, 0), tr.detach().unsqueeze(-1)).squeeze(-1).round()
    w = torch.randn_like(tr)
    (tr * w).sum().backward()
    graph = torch.repeat_interleave(torch.arange(2), counts)
    want = torch.zeros(2, 3, 3, dtype=torch.float64).index_add_(0, graph, w[:, :, None] * n[:, None, :])
    assert torch.allclose(T.grad, want, atol=1e-9)


@pytest.mark.parametrize('what', ['cell', 'centre', 'radius', 'molecule'])
def test_undefined_crystal_keeps_only_its_zero_translation(what):
    """A crystal with a non-finite cell, centre, radius or molecule centroid has no defined cluster: it keeps only its
    zero translation (with NaN entries for a non-finite cell), with a warning, and the other crystals get exactly what they get
    without it. One bad draw must not stop a training batch; MolCrystalData.analyze turns such rows into NaN."""
    clean = _cases()
    c = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in clean.items()}
    bad = 3 if what != 'molecule' else int(torch.searchsorted(torch.cumsum(c['Z'], 0), 7, right=True))
    if what == 'cell':
        c['T'][3, 0, 0] = float('nan')
    elif what == 'centre':
        c['cc'][3, 1] = float('inf')
    elif what == 'radius':
        c['radius'][3] = float('nan')
    else:
        c['mol'][7, 2] = float('nan')
    with pytest.warns(UserWarning, match='zero translation'):
        tr, counts = _run(c, 10.0, True)
    ref_tr, ref_counts = _run(clean, 10.0, True)
    assert int(counts[bad]) == 1
    keep = torch.arange(len(counts)) != bad
    assert torch.equal(counts[keep], ref_counts[keep])
    per, ref_per = torch.split(tr, counts.tolist()), torch.split(ref_tr, ref_counts.tolist())
    for b in torch.nonzero(keep)[:, 0].tolist():
        assert torch.equal(per[b], ref_per[b])
    assert bool(torch.isnan(per[bad]).any()) if what == 'cell' else bool((per[bad] == 0).all())


def _flat_case(h):
    """A 5 x 5 A cell whose third vector lies h A above the ab plane (volume 25 h A^3), with two molecules."""
    T = _cell(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    T[:, 2] = 0.5 * (T[:, 0] + T[:, 1]) + torch.tensor([0.0, 0.0, h], dtype=torch.float64)
    cc = torch.tensor([[1.0, 2.0, 0.3 * h]])
    mol = torch.cat([cc, cc + torch.tensor([[2.0, -1.0, 0.1 * h]])]).float()
    return T[None].float(), cc.float(), mol


def test_nearly_flat_cell_is_capped_to_its_nearest_cells(monkeypatch):
    """Owner decision 2026-10-18: a crystal keeps at most MAX_CELLS_PER_CRYSTAL translations, those whose nearest molecule
    is closest; the zero translation first; a warning names the capped crystals. The capped set is the cap-nearest part
    of the same search without the cap."""
    T, cc, mol = _flat_case(0.01)
    with pytest.warns(UserWarning, match='nearest cells'):
        tr, counts = get_cart_translations(cc, T, torch.tensor([3.0]), 10.0, mol_centroids=mol,
                                           mols_per_crystal=torch.tensor([2]))
    assert int(counts[0]) == CBU.MAX_CELLS_PER_CRYSTAL
    assert bool((tr[0] == 0).all())
    monkeypatch.setattr(CBU, 'MAX_CELLS_PER_CRYSTAL', 10 ** 8)
    full, n_full = get_cart_translations(cc, T, torch.tensor([3.0]), 10.0, mol_centroids=mol,
                                         mols_per_crystal=torch.tensor([2]))
    assert int(n_full[0]) > int(counts[0])

    def nearest(t):
        return (cc.double()[:, None] - (mol.double()[None] + t.double()[:, None])).norm(dim=-1).amin(1)
    d_kept, d_all = nearest(tr), nearest(full)
    kth = torch.sort(d_all).values[int(counts[0]) - 1]
    assert float(d_kept.max()) <= float(kth) + 1e-6


def test_nearly_flat_cell_search_stays_within_budget(monkeypatch):
    """The search shrinks its radius for a cell whose boxes would exceed MAX_SEARCH_CANDIDATES_PER_CRYSTAL, so the
    pass never examines more than that many candidates for one crystal."""
    T, cc, mol = _flat_case(1e-4)
    seen = []
    real_arange = torch.arange

    def spy(*a, **k):
        out = real_arange(*a, **k)
        seen.append(out.numel())
        return out
    monkeypatch.setattr(torch, 'arange', spy)
    with pytest.warns(UserWarning):
        get_cart_translations(cc, T, torch.tensor([3.0]), 10.0, mol_centroids=mol, mols_per_crystal=torch.tensor([2]))
    assert max(seen) <= CBU.MAX_SEARCH_CANDIDATES_PER_CRYSTAL


def test_singular_cell_keeps_only_its_zero_translation():
    T = _cell(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    singular = T.clone()
    singular[:, 2] = T[:, 0] + T[:, 1]
    with pytest.warns(UserWarning, match='zero translation'):
        tr, counts = get_cart_translations(torch.zeros(2, 3), torch.stack([singular, T]).float(), torch.tensor([3.0, 3.0]),
                                           10.0)
    ref_tr, ref_counts = get_cart_translations(torch.zeros(1, 3), T[None].float(), torch.tensor([3.0]), 10.0)
    assert int(counts[0]) == 1 and bool((tr[0] == 0).all())
    assert int(counts[1]) == int(ref_counts[0]) and torch.equal(tr[1:], ref_tr)


@pytest.mark.parametrize('per', [[2, 3], [2.5, 2.5, 0], [0, 5, 0]])
def test_bad_mols_per_crystal_is_refused(per):
    c = _cases(n_reduced=3, n_skew=0)
    with pytest.raises(ValueError, match='mols_per_crystal'):
        get_cart_translations(c['cc'], c['T'], c['radius'], 10.0, mol_centroids=c['mol'][:5],
                              mols_per_crystal=torch.tensor(per))

"""
analysis/exp6_ff.py: the typed exp-6 intermolecular force field (W99 / FIT).

1. The parameter tables transcribe the published values: W99 C(3), H(1), N(2) equal mol-cspy's w99.pots (eV,
   rho = 1/B) converted to kJ/mol, and every cross term follows the stated combining rules.
2. Typing: acridine is 13 C(3), 1 N(2), 9 H(1) per molecule; small molecules exercise every W99 rule (C by
   connectivity, alcohol / carboxylic acid / N-H hydrogens, nitrile / amine N, carbonyl / ether O); FIT types by
   element and H by partner; an element outside the sets is refused.
3. Short range. exp-6 form: the continuation below the splice is C1 and the pair energy falls monotonically out
   to the well. ELJ form: each pair keeps its exp-6 well (position and depth), is C1 at sigma, falls monotonically
   below the well and is finite at contact; a pair with no dispersion stays the plain exponential.
4. Every H of every periodic image finds its bonded atom through the offset column (the image atom order is the
   asymmetric unit's).
5. The energy is per molecule: a Z'=1 crystal and the same crystal re-described as Z'=2 agree, and a crystal scores the
   same alone and in a batch. The dispersion tail brings the 10 A energy closer to the 16 A one.
6. RDKit charges are cached by topology (rigid motions reuse them) and sum to the molecule's charge. With a SMILES
   the bond orders come from it (a sulfur heteroaromatic whose geometry-perceived bond orders go wrong stays
   neutral); a set that does not sum to the net charge, or a SMILES that does not fit the geometry, is refused; a
   Z' = 2 crystal takes its '|'-joined per-molecule SMILES.
7. Gradients: exact (float64) for this module's pair terms and tail; finite and consistent end to end in the cell
   parameters.

CPU only.
"""
import math
from pathlib import Path

import pytest
import torch

from mxtaltools.analysis import exp6_ff as ff
from mxtaltools.dataset_utils.utils import collate_data_list

ACRIDINE = Path(__file__).resolve().parent / 'datasets' / 'mini_acridine.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']
EV = 96.4853  # kJ/mol per eV


def _strip(c):
    c = c.clone()
    for k in DROP:
        if k in c.keys():
            delattr(c, k)
    return c


@pytest.fixture(scope='module')
def acridine():
    if not ACRIDINE.exists():
        pytest.skip(f'{ACRIDINE} not present')
    data = torch.load(ACRIDINE, weights_only=False)
    zp1 = [_strip(c) for c in data if int(c.sg_ind) == 14 and int(c.z_prime) == 1]
    zp2 = [_strip(c) for c in data if int(c.sg_ind) == 14 and int(c.z_prime) == 2]
    return zp1, zp2


# ------------------------------------------------------------------ 1. parameters
@pytest.mark.parametrize('name, pots', [('C3', (2802.334865, 0.27777778, 17.63857225)),
                                        ('H1', (131.429249, 0.28089888, 2.88532808)),
                                        ('N2', (1061.063155, 0.28735632, 14.49194043))])
def test_w99_matches_mol_cspy(name, pots):
    _, A, B, C = ff.PARAMETER_SETS['w99']['types'][name]
    a_ev, rho, c_ev = pots
    assert A == pytest.approx(a_ev * EV, rel=2e-4)
    assert B == pytest.approx(1 / rho, rel=1e-6)
    assert C == pytest.approx(c_ev * EV, rel=2e-4)


@pytest.mark.parametrize('params', list(ff.PARAMETER_SETS))
def test_cross_terms_follow_the_combining_rules(params):
    t = ff.pair_tables(params, dtype=torch.float64)
    types = ff.PARAMETER_SETS[params]['types']
    for i, a in enumerate(t['names']):
        for j, b in enumerate(t['names']):
            _, Aa, Ba, Ca = types[a]
            _, Ab, Bb, Cb = types[b]
            assert float(t['A'][i, j]) == pytest.approx(math.sqrt(Aa * Ab))
            assert float(t['B'][i, j]) == pytest.approx((Ba + Bb) / 2)
            assert float(t['C'][i, j]) == pytest.approx(math.sqrt(Ca * Cb))
            assert float(t['sqrtC'][i] * t['sqrtC'][j]) == pytest.approx(math.sqrt(Ca * Cb))


# ------------------------------------------------------------------ 2. typing
def _type(smiles, params='w99'):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    types, partner = ff.type_atoms(z, pos, torch.zeros(len(z), dtype=torch.long), params)
    names = list(ff.PARAMETER_SETS[params]['types'])
    return sorted(names[i] for i in types.tolist()), types, partner, z


@pytest.mark.parametrize('smiles, expected', [
    ('CC(=O)O', ['C3', 'C4', 'H1', 'H1', 'H1', 'H3', 'O1', 'O2']),           # acetic acid
    ('CCO', ['C4', 'C4', 'H1', 'H1', 'H1', 'H1', 'H1', 'H2', 'O2']),          # ethanol
    ('CC#N', ['C2', 'C4', 'H1', 'H1', 'H1', 'N1']),                            # acetonitrile
    ('CN', ['C4', 'H1', 'H1', 'H1', 'H4', 'H4', 'N4']),                        # methylamine
    ('CNC', ['C4', 'C4', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H4', 'N3']),    # dimethylamine
    ('c1ccncc1', ['C3'] * 5 + ['H1'] * 5 + ['N2']),                            # pyridine
    ('COC=O', ['C3', 'C4', 'H1', 'H1', 'H1', 'H1', 'O1', 'O2']),              # methyl formate: no acid H
])
def test_w99_rules(smiles, expected):
    assert _type(smiles)[0] == sorted(expected)


def test_fit_types_by_element_and_h_partner():
    assert _type('NCCO', 'fit')[0] == sorted(['C', 'C', 'HC', 'HC', 'HC', 'HC', 'HN', 'HN', 'HO', 'N', 'O'])


def test_h_partner_is_the_bonded_atom():
    _, types, partner, z = _type('CC(=O)O')
    h = z == 1
    assert torch.all(z[partner[h]] != 1) and torch.all(partner[~h] == torch.arange(len(z))[~h])


def test_unsupported_element_is_refused():
    z = torch.tensor([35, 6, 1, 1, 1])
    pos = torch.tensor([[0., 0, 0], [1.94, 0, 0], [2.3, 1.0, 0], [2.3, -0.5, 0.87], [2.3, -0.5, -0.87]])
    with pytest.raises(ValueError, match='35'):
        ff.type_atoms(z, pos, torch.zeros(5, dtype=torch.long))


def test_acridine_types(acridine):
    zp1, zp2 = acridine
    b = collate_data_list([zp2[0].clone()])
    for params, want in (('w99', {'C3': 26, 'N2': 2, 'H1': 18}), ('fit', {'C': 26, 'N': 2, 'HC': 18})):
        types, _ = ff.type_atoms(b.z, b.pos, ff.aunit_molecule_index(b), params)
        names = list(ff.PARAMETER_SETS[params]['types'])
        got = {n: int((types == names.index(n)).sum()) for n in want}
        assert got == want


# ------------------------------------------------------------------ 3. short range
@pytest.mark.parametrize('a, b', [('C3', 'C3'), ('C3', 'H1'), ('H1', 'H1'), ('N2', 'H1'), ('O1', 'H2')])
def test_splice_is_c1_and_monotone(a, b):
    t = ff.pair_tables('w99', dtype=torch.float64)
    i, j = t['names'].index(a), t['names'].index(b)
    rs = float(t['r_s'][i, j])
    r = torch.linspace(0.01, 3.0, 29901, dtype=torch.float64, requires_grad=True)  # r < 1e-3 A is clamped
    ti, tj = torch.full_like(r, i, dtype=torch.long), torch.full_like(r, j, dtype=torch.long)
    rep, disp = ff.exp6_pair_energy(r, ti, tj, t)
    e = rep + disp
    g, = torch.autograd.grad(e.sum(), r)
    assert torch.isfinite(e).all() and torch.isfinite(g).all()
    assert torch.all(g < 0), f'{a}-{b}: the pair energy must fall monotonically below 3 A'
    if rs > 0:
        k = int(torch.searchsorted(r.detach(), torch.tensor(rs, dtype=torch.float64)))
        assert abs(float(g[k] - g[k - 1])) < 1e-2 * abs(float(g[k])), 'slope jumps at the splice'


W99 = ff.pair_tables('w99', dtype=torch.float64)


@pytest.mark.parametrize('a, b', [('C3', 'C3'), ('C3', 'H1'), ('H1', 'H1'), ('N2', 'H1'), ('O1', 'N3'), ('C4', 'Cl')])
def test_elj_form_keeps_each_pairs_exp6_well(a, b):
    i, j = W99['names'].index(a), W99['names'].index(b)
    r = torch.linspace(0.01, 8.0, 159801, dtype=torch.float64, requires_grad=True)
    ti, tj = torch.full_like(r, i, dtype=torch.long), torch.full_like(r, j, dtype=torch.long)
    e = sum(ff.elj_pair_energy(r, ti, tj, W99))
    e6 = sum(ff.exp6_pair_energy(r, ti, tj, W99))
    g, = torch.autograd.grad(e.sum(), r)
    r_ = r.detach()
    k, k6 = int(torch.argmin(e)), int(torch.argmin(e6[r_ > 1.5]) + int((r_ <= 1.5).sum()))
    assert abs(float(r_[k] - r_[k6])) < 1e-3, 'the ELJ form must keep the exp-6 minimum where it is'
    assert float(e[k]) == pytest.approx(float(e6[k6]), rel=1e-4), 'and its depth'
    sig = float(W99['sigma'][i, j])
    ks = int(torch.searchsorted(r_, torch.tensor(sig, dtype=torch.float64)))
    assert abs(float(e[ks])) < 1e-3 * float(W99['eps'][i, j]) * 10, 'zero at sigma'
    assert abs(float(g[ks] - g[ks - 1])) < 1e-2 * abs(float(g[ks])), 'slope jumps at sigma'
    assert torch.all(g[:k] < 0) and torch.isfinite(e).all(), 'monotone and finite below the well'


def test_elj_form_leaves_a_pair_without_dispersion_exponential():
    i, j = W99['names'].index('O1'), W99['names'].index('H2')
    r = torch.linspace(0.5, 6.0, 101, dtype=torch.float64)
    ti, tj = torch.full_like(r, i, dtype=torch.long), torch.full_like(r, j, dtype=torch.long)
    rep, disp = ff.elj_pair_energy(r, ti, tj, W99)
    assert torch.allclose(rep, W99['A'][i, j] * torch.exp(-W99['B'][i, j] * r)) and torch.all(disp == 0)


# ------------------------------------------------------------------ 4. H partners in the cluster
def test_every_image_h_finds_its_partner(acridine):
    _, zp2 = acridine
    b = collate_data_list([c.clone() for c in zp2[:2]])
    cluster = ff.with_ff_columns(b, 'w99', None).mol2cluster(10.0, 10, std_orientation=True)
    off = cluster.x[:, 2].round().long()
    h = cluster.z == 1
    assert torch.equal(h, off != 0)
    d = (cluster.pos[h] - cluster.pos[torch.arange(len(off))[h] + off[h]]).norm(dim=-1)
    assert d.min() > 0.9 and d.max() < 1.2, (float(d.min()), float(d.max()))


# ------------------------------------------------------------------ 5. convention
@pytest.mark.parametrize('form', ['elj', 'exp6'])
def test_per_molecule_zp1_equals_its_zp2_double(acridine, form):
    from mxtaltools.crystal_building.zp_doubling import double_zp1_crystal
    zp1, _ = acridine
    parents = zp1[:4]
    doubled = [double_zp1_crystal(c)[0] for c in parents]
    with torch.no_grad():
        e1 = ff.crystal_exp6_energy(collate_data_list([c.clone() for c in parents]), charges='mmff', form=form)
        e2 = ff.crystal_exp6_energy(collate_data_list([c.clone() for c in doubled]), charges='mmff', form=form)
    assert torch.allclose(e1, e2, rtol=1e-3, atol=1e-2), (e1, e2)


@pytest.mark.parametrize('which', [0, 1])
def test_batch_invariance(acridine, which):
    """Within one Z'. (A batch MIXING Z'=1 and Z'=2 builds its Z'=1 clusters through the Z'>1 path, which finds a few
    more pairs at the cutoff than the Z'=1 path does: 13 of 9440 on ACRDIN crystal 1, 0.05 kJ/mol here and the same
    shift in eLJ. That is cluster building, not this module.)"""
    crystals = acridine[which][:3]
    with torch.no_grad():
        together = ff.crystal_exp6_energy(collate_data_list([c.clone() for c in crystals]), charges='mmff')
        alone = torch.cat([ff.crystal_exp6_energy(collate_data_list([c.clone()]), charges='mmff') for c in crystals])
    assert torch.allclose(together, alone, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize('form', ['elj', 'exp6'])
def test_tail_moves_the_10A_energy_towards_16A(acridine, form):
    zp1, _ = acridine
    b = collate_data_list([zp1[0].clone()])
    with torch.no_grad():
        far = ff.crystal_exp6_energy(b, charges=None, cutoff=16.0, form=form)
        with_tail = ff.crystal_exp6_energy(b, charges=None, cutoff=10.0, form=form)
        without = ff.crystal_exp6_energy(b, charges=None, cutoff=10.0, tail=False, form=form)
    assert abs(float(with_tail - far)) < 0.5 * abs(float(without - far)), (float(far), float(with_tail), float(without))


# ------------------------------------------------------------------ 6. charges
def test_rdkit_charges_cached_by_topology(acridine):
    zp1, _ = acridine
    m = zp1[0]
    q = ff.rdkit_charges(m.z, m.pos, 'mmff')
    n = len(ff._CHARGE_CACHE)
    rot, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(0)))
    assert torch.equal(ff.rdkit_charges(m.z, m.pos @ rot + 3.0, 'mmff'), q)
    assert len(ff._CHARGE_CACHE) == n
    assert abs(float(q.sum())) < 1e-4
    assert float(q[m.z == 7][0]) < -0.5, 'MMFF puts about -0.6 e on the acridine N'


HEYWAK = r'[H]/C(=C1\C(=O)/C(=C(\[H])c2sc([H])c([H])c2[H])C([H])([H])C([H])([H])C1([H])[H])c1sc([H])c([H])c1[H]'


def _embed(smiles, seed=7):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    p = Chem.SmilesParserParams()
    p.removeHs = False
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles, p))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    perm = torch.randperm(mol.GetNumAtoms(), generator=torch.Generator().manual_seed(seed))  # not the SMILES order
    z = torch.tensor([mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in perm])
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)[perm]
    return mol, perm, z, pos


@pytest.mark.parametrize('model', ['mmff', 'gasteiger'])
def test_smiles_bond_orders_keep_a_sulfur_heteroaromatic_neutral(model):
    from rdkit.Chem import AllChem
    mol, perm, z, pos = _embed(HEYWAK)
    q = ff.rdkit_charges(z, pos, model, smiles=HEYWAK)
    assert abs(float(q.sum())) < 1e-3
    if model == 'mmff':  # the same charges RDKit puts on the SMILES molecule, carried to the geometry's atom order
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ref = torch.tensor([props.GetMMFFPartialCharge(int(i)) for i in perm])
        assert torch.allclose(q, ref, atol=1e-4)
    try:  # from the geometry alone RDKit may perceive the thiophenes wrongly, but must never return a charged set
        g = ff.rdkit_charges(z, pos, model)
        assert abs(float(g.sum())) < 1e-3
    except ValueError:
        pass


def test_charges_off_the_net_charge_are_refused():
    with pytest.raises(ValueError, match='net charge'):
        ff._check_charge_sum(torch.tensor([0.3, -0.1, -0.2, -1.0]), 0, 'test')


def test_a_smiles_that_does_not_fit_the_geometry_is_refused():
    _, _, z, pos = _embed('CCO')
    with pytest.raises(ValueError, match='does not match'):
        ff.rdkit_charges(z, pos, 'mmff', smiles='COC')


def test_batch_charges_use_each_molecules_smiles(acridine):
    _, zp2 = acridine
    b = collate_data_list([zp2[0].clone()])
    smi = b.smiles[0] if isinstance(b.smiles, list) else b.smiles
    assert smi.count('|') == 1, smi
    q = ff.batch_charges(b, 'mmff')
    n = len(q) // 2
    assert abs(float(q[:n].sum())) < 1e-3 and abs(float(q[n:].sum())) < 1e-3
    assert any(k[1] == ('smiles', smi.split('|')[0]) for k in ff._CHARGE_CACHE), 'the SMILES path was not taken'
    assert float(q[b.z.long() == 7][0]) < -0.5


# ------------------------------------------------------------------ 7. gradients
@pytest.mark.parametrize('form', ['elj', 'exp6'])
def test_pair_terms_gradient_is_exact(acridine, form):
    """float64, on a fixed cluster and pair list: autograd of the pair sums (exp-6 with the H shift and the envelope,
    and Coulomb) in the atom positions equals a central difference."""
    from mxtaltools.analysis.vdw_analysis import get_intermolecular_dists_dict
    _, zp2 = acridine
    b = ff.with_ff_columns(collate_data_list([zp2[0].clone()]), 'w99', 'mmff')
    with torch.no_grad():
        cluster = b.mol2cluster(10.0, 10, std_orientation=True)
        edges = get_intermolecular_dists_dict(cluster, 10.0)
    cluster.x = cluster.x.double()
    tables = ff.pair_tables('w99', dtype=torch.float64)
    pos0 = cluster.pos.detach().double()

    def energy(pos):
        cluster.pos = pos
        rep, disp, coul = ff.cluster_exp6_terms(cluster, edges, tables, 10.0, h_shift=0.1, envelope=1.0, form=form)
        return (rep + disp + coul).sum()

    pos = pos0.clone().requires_grad_(True)
    g, = torch.autograd.grad(energy(pos), pos)
    inside_h = torch.nonzero((cluster.aux_ind == 0) & (cluster.z == 1))[0, 0]
    image_c = edges['edge_index_inter'][0][(cluster.z[edges['edge_index_inter'][0]] == 6)][0]
    for atom in (int(inside_h), int(image_c), int(inside_h) + int(cluster.x[inside_h, 2])):  # an H, an image C, a
        for d in range(3):                                                               # bonded partner of an H
            dp = torch.zeros_like(pos0)
            dp[atom, d] = 1e-6
            with torch.no_grad():
                fd = (energy(pos0 + dp) - energy(pos0 - dp)) / 2e-6
            assert float(g[atom, d]) == pytest.approx(float(fd), rel=1e-5, abs=1e-6), (atom, d)


@pytest.mark.parametrize('form', ['elj', 'exp6'])
def test_tail_gradient_is_exact(acridine, form):
    _, zp2 = acridine
    b = collate_data_list([zp2[0].clone()])
    types, _ = ff.type_atoms(b.z, b.pos, ff.aunit_molecule_index(b))
    tables = ff.pair_tables('w99', dtype=torch.float64)
    T0 = b.T_fc.detach().double()

    def tail(T):
        return ff.dispersion_tail(types, b.batch, T, b.sym_mult, b.z_prime, tables, 10.0, envelope=1.0,
                                  form=form).sum()

    T = T0.clone().requires_grad_(True)
    g, = torch.autograd.grad(tail(T), T)
    for i, j in ((0, 0), (1, 1), (0, 2)):
        dT = torch.zeros_like(T0)
        dT[0, i, j] = 1e-6
        fd = (tail(T0 + dT) - tail(T0 - dT)) / 2e-6
        assert float(g[0, i, j]) == pytest.approx(float(fd), rel=1e-5, abs=1e-8)


def test_energy_is_differentiable_in_the_cell_parameters(acridine):
    """float32 end to end (cluster building is float32-only): every cell parameter gets a finite gradient that agrees
    with a central difference to within the float32 rounding and curvature of a ~0.01 A step. Catches a detached
    path, not a subtle error (the exact checks are above)."""
    _, zp2 = acridine
    b = collate_data_list([zp2[0].clone()])
    p0 = b.full_cell_parameters().detach().clone()

    def energy(p):
        c = b.clone()
        c.set_cell_parameters(p, skip_box_analysis=False)
        return ff.crystal_exp6_energy(c, charges='mmff', envelope=1.0).sum()

    p = p0.clone().requires_grad_(True)
    g, = torch.autograd.grad(energy(p), p)
    assert torch.isfinite(g).all()
    for k, h in ((0, 1e-2), (1, 1e-2), (2, 1e-2), (7, 1e-3), (12, 1e-3)):
        dp = torch.zeros_like(p0)
        dp[0, k] = h
        with torch.no_grad():
            fd = (energy(p0 + dp) - energy(p0 - dp)) / (2 * h)
        assert float(g[0, k]) == pytest.approx(float(fd), rel=0.05, abs=1.0), (k, float(g[0, k]), float(fd))


def test_envelope_tail_accounts_for_the_switched_shell(acridine):
    """Switching off the last 1 A shell removes dispersion from the pair sum; the tail puts back its homogeneous
    estimate, so the enveloped and sharp totals agree far better than their pair sums do."""
    zp1, _ = acridine
    b = collate_data_list([c.clone() for c in zp1[:3]])
    with torch.no_grad():
        sharp, ts = ff.crystal_exp6_energy(b, charges=None, return_terms=True)
        smooth, tw = ff.crystal_exp6_energy(b, charges=None, envelope=1.0, return_terms=True)
    pair_gap = ((ts['disp'] + ts['rep']) - (tw['disp'] + tw['rep'])).abs()
    total_gap = (sharp - smooth).abs()
    assert torch.all(pair_gap > 0.3) and torch.all(total_gap < 0.2 * pair_gap), (pair_gap, total_gap)

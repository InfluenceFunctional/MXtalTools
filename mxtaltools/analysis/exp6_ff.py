"""Typed exp-6 (Buckingham) intermolecular force field with openly published parameters.

Pair energy between atoms i and j of different molecules, kJ/mol with r in Angstrom:

    E_ij(r) = A_ij exp(-B_ij r) - C_ij / r**6,
    A_ij = sqrt(A_i A_j),  B_ij = (B_i + B_j) / 2,  C_ij = sqrt(C_i C_j),

plus, optionally, shifted-force Coulomb between per-atom point charges, a homogeneous-density dispersion tail beyond
the cutoff, and a C2 switch that takes the exp-6 pair energies smoothly to zero at the cutoff. The combining rules are those of the source papers (checked against the cross terms of mol-cspy's
w99.pots).

Parameter sets, as tabulated in Appendix D of the DMACRYS manual (Price et al., July 2019), which cites:

- ``'w99'``: D. E. Williams, J. Comput. Chem. 22, 1154 (2001). C typed by its number of bonded atoms (2, 3, 4);
  H by what it is bonded to (C; O of an alcohol; O of a carboxylic acid; N); N as triple-bonded (one bonded atom) or
  by its number of bonded H (0, 1, >= 2); O by its number of bonded atoms (1, 2). Polar H carry no dispersion, so
  hydrogen bonds come from the electrostatics. Fitted with point charges, and with every H interaction site moved
  0.1 A into its X-H bond (``h_shift``).
- ``'fit'``: C, N and H on C: Williams & Cox, Acta Cryst. B40, 404 (1984); H on N: Coombes, Price, Willock & Leslie,
  J. Phys. Chem. 100, 7352 (1996); H on O: Beyer & Price, J. Phys. Chem. B 104, 2647 (2000); O: Cox, Hsu & Williams,
  Acta Cryst. A37, 293 (1981); F: Williams & Houpt, Acta Cryst. B42, 286 (1986); Cl: Hsu & Williams, Acta Cryst.
  A36, 277 (1980). Fitted for use with distributed multipoles; H sites at the nuclei.

EXTENSIONS, not part of the published sets: both sets get the S that the manual lists for use with FIT in the blind
tests (C=S and thioether: 401034, 3.30, 5791), and W99, which has no halogens, gets FIT's F and Cl. H bonded to S is
typed as H on C. Any other element raises.

Functional form (``form``). ``'elj'``, the default, is the production eLJ shape with each pair's sigma and eps matched
to its exp-6 minimum: sigma_ij = r_min / 2**(1/6), eps_ij = -E_ij(r_min); the 12-6 LJ 4 eps ((sigma/r)^12 -
(sigma/r)^6) at r >= sigma and, below sigma, eLJ's exponential wall a exp(-k (r - sigma)) - a with k = k_factor /
sigma and a = 24 eps / k_factor (k_factor 2.5 as in ``compute_eLJ_energy``), C1 at sigma and finite at contact. A
pair with C_ij = 0 (W99's polar H) has no well and stays the plain exponential A_ij exp(-B_ij r).

``'exp6'`` evaluates exp-6 itself. It turns over at r ~ 0.8-1.1 A and dives to -inf (the Buckingham catastrophe);
below the inner inflection point r_s of each pair's repulsive branch (0.93-1.27 A for the W99 pairs, far inside any
contact), the pair energy is continued linearly with the slope it has at r_s -- the steepest slope of the branch --
so it is C1, monotone and finite at r = 0. Pairs with C_ij = 0 are plain exponentials and need no repair.

Typing runs once per molecule, on the asymmetric unit, from a covalent bond graph perceived from distances (no RDKit
at run time). The type and the H partner offset ride into the periodic cluster as extra columns of ``x``, which
``_instantiate_cluster`` already replicates from each asymmetric-unit atom to all of its images, exactly as it does
``z``; column 0 stays the partial charge that ``get_intermolecular_dists_dict`` reads.

Energy convention. ``crystal_exp6_energy`` returns the lattice energy PER MOLECULE, kJ/mol: the sum over the pair
list of the asymmetric unit (every Z' molecule against every other molecule within the cutoff, so a pair of
asymmetric-unit molecules appears twice) divided by 2 Z', plus the tail. The eLJ energy of ``compute_eLJ_energy``
is the same pair sum without the 1 / (2 Z').
"""
from functools import lru_cache
from typing import Optional, Union

import torch
from torch_scatter import scatter

KJ_COULOMB = 1389.35457  # e^2 / (4 pi eps0), kJ/mol Angstrom
BOND_TOL = 0.4  # Angstrom added to the covalent radius sum when perceiving bonds
COULOMB_R_MIN = 0.5  # Angstrom: Coulomb distances are clamped here, so coincident sites stay finite

# Cordero et al., Dalton Trans. 2008, 2832 (sp3 C)
COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 16: 1.05, 17: 1.02}

_EXTRA = {'F': (9, 363725.0, 4.16, 844.0),
          'Cl': (17, 924675.0, 3.51, 7740.48),
          'S': (16, 401034.0, 3.30, 5791.0)}

# name: (element, A kJ/mol, B 1/A, C kJ/mol A^6)
PARAMETER_SETS = {
    'w99': {
        'types': {
            'C2': (6, 103235.0, 3.60, 1435.09),  # C bonded to 2 atoms
            'C3': (6, 270363.0, 3.60, 1701.73),  # C bonded to 3 atoms
            'C4': (6, 131571.0, 3.60, 978.36),  # C bonded to 4 atoms
            'H1': (1, 12680.0, 3.56, 278.37),  # H on C (and, as an extension, on S)
            'H2': (1, 361.30, 3.56, 0.0),  # H of an alcohol O
            'H3': (1, 115.70, 3.56, 0.0),  # H of a carboxylic acid O
            'H4': (1, 764.90, 3.56, 0.0),  # H on N
            'N1': (7, 96349.0, 3.48, 1407.57),  # N bonded to 1 atom (triple bond)
            'N2': (7, 102369.0, 3.48, 1398.15),  # other N with no bonded H
            'N3': (7, 191935.0, 3.48, 2376.55),  # N bonded to 1 H
            'N4': (7, 405341.0, 3.48, 5629.82),  # N bonded to >= 2 H
            'O1': (8, 241042.0, 3.96, 1260.73),  # O bonded to 1 atom
            'O2': (8, 284623.0, 3.96, 1285.87),  # O bonded to 2 atoms
            **_EXTRA},
        'h_shift': 0.1,
    },
    'fit': {
        'types': {
            'C': (6, 369743.0, 3.60, 2439.8),
            'HC': (1, 11971.0, 3.74, 136.4),  # H on C (and, as an extension, on S)
            'HN': (1, 5029.68, 4.66, 21.50),
            'HO': (1, 2263.3, 4.66, 21.50),
            'N': (7, 254529.0, 3.78, 1378.4),
            'O': (8, 230064.1, 3.96, 1123.59),
            **_EXTRA},
        'h_shift': 0.0,
    },
}


def perceive_bonds(z: torch.Tensor, pos: torch.Tensor, mol: torch.Tensor, tol: float = BOND_TOL):
    """Covalent bonds within each molecule, from distances: d < r_cov(i) + r_cov(j) + tol.

    Parameters
    ----------
    z : [n] atomic numbers.
    pos : [n, 3] positions (Angstrom).
    mol : [n] molecule index; bonds are only sought between atoms of the same index.
    tol : float, Angstrom.

    Returns
    -------
    [2, n_bonds] long, both directions, no self-loops.
    """
    from torch_cluster import radius_graph
    z = z.long()
    unknown = sorted(set(z.unique().tolist()) - set(COVALENT_RADII))
    if unknown:
        raise ValueError(f'exp6_ff: no covalent radius for elements {unknown}')
    rcov = torch.zeros(int(z.max()) + 1, dtype=pos.dtype, device=pos.device)
    for k, v in COVALENT_RADII.items():
        if k < len(rcov):
            rcov[k] = v
    r_max = 2 * float(rcov[z].max()) + tol
    edge = radius_graph(pos.detach(), r=r_max, batch=mol.long(), loop=False, max_num_neighbors=64)
    d = (pos[edge[0]] - pos[edge[1]]).norm(dim=-1)
    return edge[:, d < rcov[z[edge[0]]] + rcov[z[edge[1]]] + tol]


def type_atoms(z: torch.Tensor, pos: torch.Tensor, mol: torch.Tensor, params: str = 'w99'):
    """Force-field type of every atom, and the bonded heavy atom of every H.

    Parameters
    ----------
    z, pos, mol : as ``perceive_bonds``; one molecule per distinct ``mol`` value, with its own geometry.
    params : ``'w99'`` or ``'fit'``.

    Returns
    -------
    types : [n] long, indices into ``list(PARAMETER_SETS[params]['types'])``.
    partner : [n] long; for an H, the index of the bonded atom nearest to it; for any other atom, its own index.

    Raises
    ------
    ValueError
        For an element the set does not cover, or an H with no bonded atom.
    """
    names = list(PARAMETER_SETS[params]['types'])
    idx = {n: i for i, n in enumerate(names)}
    z = z.long()
    n = len(z)
    dev = z.device
    edge = perceive_bonds(z, pos, mol)
    src, dst = edge
    deg = scatter(torch.ones_like(src), src, dim=0, dim_size=n, reduce='sum')
    n_h = scatter((z[dst] == 1).long(), src, dim=0, dim_size=n, reduce='sum')

    # a carboxylic acid O-H: the O's other neighbour is a C that also carries an O bonded to nothing else
    term_o = (z == 8) & (deg == 1)
    c_with_term_o = scatter(term_o[dst].long(), src, dim=0, dim_size=n, reduce='sum') > 0
    acid_o = (z == 8) & (scatter((c_with_term_o[dst] & (z[dst] == 6)).long(), src, dim=0, dim_size=n,
                                 reduce='sum') > 0)

    # each H's partner: the nearest bonded atom
    is_h = z == 1
    d = (pos[src] - pos[dst]).norm(dim=-1).detach()
    h_edges = is_h[src]
    partner = torch.arange(n, device=dev)
    if h_edges.any():
        hs, hd, hdist = src[h_edges], dst[h_edges], d[h_edges]
        best = scatter(hdist, hs, dim=0, dim_size=n, reduce='min')
        pick = hdist <= best[hs]
        partner[hs[pick]] = hd[pick]
    lonely = is_h & (partner == torch.arange(n, device=dev))
    if lonely.any():
        raise ValueError(f'exp6_ff: {int(lonely.sum())} H atoms have no bonded atom within the covalent cutoff')
    pz = z[partner]

    types = torch.full((n,), -1, dtype=torch.long, device=dev)
    if params == 'w99':
        types[(z == 6) & (deg <= 2)] = idx['C2']
        types[(z == 6) & (deg == 3)] = idx['C3']
        types[(z == 6) & (deg >= 4)] = idx['C4']
        types[is_h & ((pz == 6) | (pz == 16))] = idx['H1']
        types[is_h & (pz == 8) & ~acid_o[partner]] = idx['H2']
        types[is_h & (pz == 8) & acid_o[partner]] = idx['H3']
        types[is_h & (pz == 7)] = idx['H4']
        types[(z == 7) & (deg <= 1)] = idx['N1']
        types[(z == 7) & (deg >= 2) & (n_h == 0)] = idx['N2']
        types[(z == 7) & (deg >= 2) & (n_h == 1)] = idx['N3']
        types[(z == 7) & (deg >= 2) & (n_h >= 2)] = idx['N4']
        types[(z == 8) & (deg <= 1)] = idx['O1']
        types[(z == 8) & (deg >= 2)] = idx['O2']
    elif params == 'fit':
        types[z == 6] = idx['C']
        types[is_h & ((pz == 6) | (pz == 16))] = idx['HC']
        types[is_h & (pz == 7)] = idx['HN']
        types[is_h & (pz == 8)] = idx['HO']
        types[z == 7] = idx['N']
        types[z == 8] = idx['O']
    else:
        raise ValueError(f"exp6_ff: unknown parameter set {params!r}; one of {list(PARAMETER_SETS)}")
    types[z == 9] = idx['F']
    types[z == 17] = idx['Cl']
    types[z == 16] = idx['S']
    if (types < 0).any():
        bad = sorted(set(z[types < 0].tolist()))
        raise ValueError(f'exp6_ff: set {params!r} does not type atoms of elements {bad} '
                         f'(or an H bonded to one of them)')
    return types, partner


def _inner_inflection(A: float, B: float, C: float) -> float:
    """Smallest r > 0 with d2E/dr2 = 0 for E = A exp(-B r) - C / r^6 (0 when C = 0): where the repulsive branch is
    steepest. Bisection on log(A B^2 r^8 exp(-B r)) = log(42 C), whose left side increases up to r = 8 / B."""
    import math
    if C <= 0:
        return 0.0
    f = lambda r: math.log(A * B * B) + 8 * math.log(r) - B * r - math.log(42 * C)
    lo, hi = 1e-3, 8.0 / B
    if f(hi) <= 0:  # the well never turns convex: no physical repulsive branch
        raise ValueError(f'exp6_ff: pair (A={A}, B={B}, C={C}) has no repulsive wall')
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _exp6_minimum(A: float, B: float, C: float, r_s: float) -> float:
    """r of the exp-6 well: the root of dE/dr beyond the inner inflection point, by bisection on
    log(A B) - B r = log(6 C) - 7 log r, whose difference falls monotonically beyond max(r_s, 7 / B)."""
    import math
    f = lambda r: math.log(A * B) - B * r - math.log(6 * C) + 7 * math.log(r)
    lo, hi = max(r_s, 7.0 / B), 30.0
    if not (f(lo) > 0 > f(hi)):
        raise ValueError(f'exp6_ff: pair (A={A}, B={B}, C={C}) has no well to match')
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pair_tables(params: str = 'w99', device='cpu', dtype=torch.float32):
    """Per type-pair tables for a parameter set (built once per set, device and dtype, then cached).

    Returns
    -------
    dict with ``names`` (type names); ``A``, ``B``, ``C``; for the exp-6 form ``r_s`` (splice radius), ``E_s`` and
    ``dE_s`` (pair energy and slope at ``r_s``); for the ELJ form ``sigma`` and ``eps`` (0 for a pair with C = 0) and
    ``C_lj`` = 4 eps sigma^6, its r^-6 coefficient; each [T, T] on ``device``; ``sqrtC`` [T] (C_ij = sqrtC_i
    sqrtC_j) and ``h_shift`` (float). The returned dict is shared: do not modify it.
    """
    return _pair_tables(params, str(torch.device(device)), dtype)


@lru_cache(maxsize=None)
def _pair_tables(params, device, dtype):
    import math
    ps = PARAMETER_SETS[params]
    names = list(ps['types'])
    T = len(names)
    A = torch.zeros(T, T, dtype=torch.float64)
    B, C, r_s, E_s, dE_s, sigma, eps = (torch.zeros_like(A) for _ in range(7))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            _, Aa, Ba, Ca = ps['types'][a]
            _, Ab, Bb, Cb = ps['types'][b]
            Aij, Bij, Cij = math.sqrt(Aa * Ab), 0.5 * (Ba + Bb), math.sqrt(Ca * Cb)
            rs = _inner_inflection(Aij, Bij, Cij)
            A[i, j], B[i, j], C[i, j], r_s[i, j] = Aij, Bij, Cij, rs
            if rs > 0:
                E_s[i, j] = Aij * math.exp(-Bij * rs) - Cij / rs ** 6
                dE_s[i, j] = -Aij * Bij * math.exp(-Bij * rs) + 6 * Cij / rs ** 7
                r_min = _exp6_minimum(Aij, Bij, Cij, rs)
                sigma[i, j] = r_min / 2 ** (1 / 6)
                eps[i, j] = -(Aij * math.exp(-Bij * r_min) - Cij / r_min ** 6)
    sqrtC = torch.tensor([math.sqrt(ps['types'][a][3]) for a in names], dtype=torch.float64)
    out = {k: v.to(device=device, dtype=dtype) for k, v in
           dict(A=A, B=B, C=C, r_s=r_s, E_s=E_s, dE_s=dE_s, sigma=sigma, eps=eps, C_lj=4 * eps * sigma ** 6,
                sqrtC=sqrtC).items()}
    out['names'] = names
    out['h_shift'] = float(ps['h_shift'])
    return out


def exp6_pair_energy(r: torch.Tensor, ti: torch.Tensor, tj: torch.Tensor, tables: dict):
    """(repulsion, dispersion) [n_pairs] kJ/mol for distances ``r`` between atoms of types ``ti``, ``tj``, with the
    linear continuation below each pair's splice radius (its whole value is booked as repulsion there)."""
    A, B, C = tables['A'][ti, tj], tables['B'][ti, tj], tables['C'][ti, tj]
    r_s = tables['r_s'][ti, tj]
    r_eval = torch.maximum(r, r_s).clamp(min=1e-3)
    rep = A * torch.exp(-B * r_eval)
    disp = -C / r_eval ** 6
    inside = r < r_s
    if inside.any():
        lin = tables['E_s'][ti, tj] + tables['dE_s'][ti, tj] * (r - r_s)
        rep = torch.where(inside, lin, rep)
        disp = torch.where(inside, torch.zeros_like(disp), disp)
    return rep, disp


def elj_pair_energy(r: torch.Tensor, ti: torch.Tensor, tj: torch.Tensor, tables: dict, k_factor: float = 2.5):
    """(repulsion, dispersion) [n_pairs] kJ/mol in the ELJ form (module docstring): above sigma the 12-6 terms;
    below it the exponential wall, with the dispersion held at its value at sigma, -4 eps, and the rest booked as
    repulsion. A pair with C = 0 is the plain exponential A exp(-B r), all repulsion."""
    sig, eps = tables['sigma'][ti, tj], tables['eps'][ti, tj]
    well = tables['C'][ti, tj] > 0
    s = torch.where(well, sig, torch.ones_like(sig))
    r = r.clamp(min=1e-3)
    x6 = (s / torch.maximum(r, s)) ** 6
    amp = 24 * eps / k_factor
    below = r < s
    wall = amp * torch.exp(-(k_factor / s) * (r - s)) - amp
    disp = torch.where(below, -4 * eps, -4 * eps * x6)
    rep = torch.where(below, wall + 4 * eps, 4 * eps * x6 ** 2)
    plain = tables['A'][ti, tj] * torch.exp(-tables['B'][ti, tj] * r)
    return torch.where(well, rep, plain), torch.where(well, disp, torch.zeros_like(disp))


def pair_energy(r: torch.Tensor, ti: torch.Tensor, tj: torch.Tensor, tables: dict, form: str = 'elj',
                k_factor: float = 2.5):
    """(repulsion, dispersion) [n_pairs] kJ/mol in the ``'elj'`` or ``'exp6'`` form."""
    if form == 'elj':
        return elj_pair_energy(r, ti, tj, tables, k_factor)
    if form == 'exp6':
        return exp6_pair_energy(r, ti, tj, tables)
    raise ValueError(f"exp6_ff: form must be 'elj' or 'exp6', got {form!r}")


def shifted_force_coulomb(r: torch.Tensor, qi: torch.Tensor, qj: torch.Tensor, cutoff: float):
    """k q_i q_j (1/r - 1/rc + (r - rc)/rc^2), kJ/mol: zero in value and slope at the cutoff; r clamped at
    COULOMB_R_MIN."""
    r = r.clamp(min=COULOMB_R_MIN)
    return KJ_COULOMB * qi * qj * (1 / r - 1 / cutoff + (r - cutoff) / cutoff ** 2)


def aunit_molecule_index(crystal_batch) -> torch.Tensor:
    """[n_atoms] molecule index of each asymmetric-unit atom of a crystal batch, unique over the batch: crystal g's
    atoms are ``z_prime[g]`` contiguous equal blocks."""
    batch, ptr = crystal_batch.batch, crystal_batch.ptr
    zp = crystal_batch.z_prime.long().reshape(-1)
    n_atoms = (ptr[1:] - ptr[:-1])
    per_mol = n_atoms // zp
    local = torch.arange(len(batch), device=batch.device) - ptr[:-1][batch]
    mol_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=batch.device), torch.cumsum(zp, 0)[:-1]])
    return mol_offset[batch] + local // per_mol[batch]


_CHARGE_CACHE = {}


def _mol_on_geometry(z: list, bonds: torch.Tensor, smiles: str):
    """RDKit molecule in the geometry's atom order with bond orders and formal charges from ``smiles``, or None when
    the SMILES graph does not match the distance-derived one. Heavy atoms are matched by graph isomorphism on
    (element, bonded-H count) labels; each heavy atom's H are interchangeable."""
    import networkx as nx
    from networkx.algorithms import isomorphism
    from rdkit import Chem
    n = len(z)
    nbr = [[] for _ in range(n)]
    for a, b in bonds.T.tolist():
        nbr[a].append(b)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    t = Chem.MolFromSmiles(smiles, params)
    if t is None:
        return None
    t = Chem.AddHs(t)
    Chem.Kekulize(t, clearAromaticFlags=True)
    if t.GetNumAtoms() != n:
        return None
    heavy = [k for k in range(n) if z[k] > 1]
    g = nx.Graph()
    g.add_nodes_from((k, {'lab': (z[k], sum(z[j] == 1 for j in nbr[k]))}) for k in heavy)
    g.add_edges_from((a, b) for a in heavy for b in nbr[a] if z[b] > 1 and a < b)
    gt = nx.Graph()
    for a in t.GetAtoms():
        if a.GetAtomicNum() > 1:
            gt.add_node(a.GetIdx(), lab=(a.GetAtomicNum(), sum(x.GetAtomicNum() == 1 for x in a.GetNeighbors())))
    gt.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in t.GetBonds()
                      if b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1)
    gm = isomorphism.GraphMatcher(g, gt, node_match=lambda a, b: a['lab'] == b['lab'])
    if not gm.is_isomorphic():
        return None
    m = dict(gm.mapping)
    for k in heavy:
        mine = [j for j in nbr[k] if z[j] == 1]
        theirs = [x.GetIdx() for x in t.GetAtomWithIdx(m[k]).GetNeighbors() if x.GetAtomicNum() == 1]
        if len(mine) != len(theirs):
            return None
        m.update(zip(mine, theirs))
    rw = Chem.RWMol()
    for k, v in enumerate(z):
        atom = Chem.Atom(v)
        atom.SetFormalCharge(t.GetAtomWithIdx(m[k]).GetFormalCharge())
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for a in range(n):
        for b in nbr[a]:
            if a < b:
                bond = t.GetBondBetweenAtoms(m[a], m[b])
                if bond is None:
                    return None
                rw.AddBond(a, b, bond.GetBondType())
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _check_charge_sum(q: torch.Tensor, expected: float, what: str):
    """Raise unless the charges sum to the molecule's net charge (to 1e-3 e)."""
    if abs(float(q.sum()) - expected) > 1e-3:
        raise ValueError(f'exp6_ff: {what} charges sum to {float(q.sum()):+.3f} e, not the molecule\'s net charge '
                         f'{expected:+.0f} e -- the bond orders were perceived wrongly or the model failed')


def rdkit_charges(z, pos, model: str = 'mmff', total_charge: int = 0, smiles: Optional[str] = None) -> torch.Tensor:
    """Partial charges [n] (e) of ONE molecule in its own atom order, from RDKit: MMFF94 (``'mmff'``) or Gasteiger
    (``'gasteiger'``) charges on a molecule whose bond orders and formal charges come from ``smiles`` when given
    (mapped onto the geometry's atom order; ``total_charge`` is then ignored) and are perceived from the geometry
    (``rdDetermineBonds``, net charge ``total_charge``) otherwise. Both models depend on the bond graph only, so the
    result is cached by (model, bond-order source, atomic numbers, bond list).

    Raises
    ------
    ValueError
        If the SMILES does not match the geometry's bond graph, RDKit has no MMFF94 parameters, or the charges do not
        sum to the molecule's net charge.
    """
    z = torch.as_tensor(z).long().cpu()
    pos = torch.as_tensor(pos).detach().double().cpu()
    bonds = perceive_bonds(z, pos, torch.zeros(len(z), dtype=torch.long))
    source = ('smiles', smiles) if smiles else ('geometry', total_charge)
    key = (model, source, tuple(z.tolist()), tuple(sorted(map(tuple, bonds.T.tolist()))))
    if key not in _CHARGE_CACHE:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDetermineBonds
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        if smiles:
            mol = _mol_on_geometry(z.tolist(), bonds, smiles)
            if mol is None:
                raise ValueError(f'exp6_ff: SMILES {smiles!r} does not match the bond graph of the geometry')
        else:
            pt = Chem.GetPeriodicTable()
            lines = [f'{pt.GetElementSymbol(int(a))} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}'
                     for a, p in zip(z.tolist(), pos.tolist())]
            mol = Chem.MolFromXYZBlock('\n'.join([str(len(z)), ''] + lines))
            rdDetermineBonds.DetermineBonds(mol, charge=total_charge)
            Chem.SanitizeMol(mol)
        if model == 'mmff':
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is None:
                raise ValueError('exp6_ff: RDKit has no MMFF94 parameters for this molecule')
            q = [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
        elif model == 'gasteiger':
            AllChem.ComputeGasteigerCharges(mol)
            q = [a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()]
        else:
            raise ValueError(f"exp6_ff: charge model must be 'mmff' or 'gasteiger', got {model!r}")
        q = torch.tensor(q, dtype=torch.float32)
        _check_charge_sum(q, sum(a.GetFormalCharge() for a in mol.GetAtoms()),
                          f'{model} ({source[0]} bond orders, {CalcMolFormula(mol)})')
        _CHARGE_CACHE[key] = q
    return _CHARGE_CACHE[key]


def batch_charges(crystal_batch, model: str = 'mmff') -> torch.Tensor:
    """[n_atoms] RDKit partial charges of every asymmetric-unit atom of a crystal batch, molecule by molecule
    (``rdkit_charges``), on the batch's device. A crystal's ``smiles`` (Z' > 1: one per molecule joined by '|', in
    molecule order) supplies the bond orders; a crystal without one, or with a different number of parts than its
    Z', has them perceived from its geometry."""
    mol = aunit_molecule_index(crystal_batch)
    bounds = torch.cat([torch.zeros(1, dtype=torch.long, device=mol.device),
                        torch.cumsum(torch.bincount(mol), 0)]).tolist()
    zp = crystal_batch.z_prime.reshape(-1).long().tolist()
    smiles = getattr(crystal_batch, 'smiles', None)
    if isinstance(smiles, str):
        smiles = [smiles]
    per_mol = []
    for g, n in enumerate(zp):
        parts = smiles[g].split('|') if smiles is not None and g < len(smiles) and smiles[g] else []
        per_mol += parts if len(parts) == n else [None] * n
    z, pos = crystal_batch.z.cpu(), crystal_batch.pos.detach().cpu()
    q = torch.cat([rdkit_charges(z[a:b], pos[a:b], model, smiles=s)
                   for a, b, s in zip(bounds[:-1], bounds[1:], per_mol)])
    return q.to(crystal_batch.pos.device)


def with_ff_columns(crystal_batch, params: str, charges: Union[str, torch.Tensor, None]):
    """Clone of ``crystal_batch`` whose ``x`` is [n_atoms, 3]: partial charge, force-field type, and the offset from
    each H to its bonded atom (0 for other atoms), all float.

    ``charges``: ``'mmff'`` or ``'gasteiger'`` computes them in each molecule's own atom order (``batch_charges``,
    bond orders from the batch's SMILES where present); ``'x'`` takes the batch's stored ones (``x`` or ``x[:, 0]``)
    -- CAUTION, the acridine files of 2026-09 store Gasteiger charges attached to the wrong atoms; None writes zeros;
    a tensor [n_atoms] is used as given."""
    types, partner = type_atoms(crystal_batch.z, crystal_batch.pos, aunit_molecule_index(crystal_batch), params)
    n = len(types)
    if isinstance(charges, str):
        if charges == 'x':
            x = crystal_batch.x
            q = x[:, 0] if x.ndim == 2 else x
        elif charges in ('mmff', 'gasteiger'):
            q = batch_charges(crystal_batch, charges)
        else:
            raise ValueError(f"exp6_ff: charges must be 'mmff', 'gasteiger', 'x', None or a tensor, got {charges!r}")
    elif charges is None:
        q = torch.zeros(n, device=crystal_batch.pos.device)
    else:
        q = charges
    q = q.to(crystal_batch.pos.device).float().reshape(-1)
    if len(q) != n:
        raise ValueError(f'exp6_ff: {len(q)} charges for {n} atoms')
    offset = partner - torch.arange(n, device=partner.device)
    out = crystal_batch.clone()
    out.x = torch.stack([q, types.float(), offset.float()], dim=1)
    return out


def cluster_exp6_terms(cluster_batch, edges_dict: dict, tables: dict, cutoff: float, h_shift: float = 0.0,
                       coulomb: bool = True, envelope: Optional[float] = None, form: str = 'elj',
                       k_factor: float = 2.5):
    """Per-graph pair sums (repulsion, dispersion, Coulomb), kJ/mol, over ``edges_dict['edge_index_inter']`` of a
    cluster built from a ``with_ff_columns`` batch (types in ``x[:, 1]``, H partner offsets in ``x[:, 2]``), with the
    pair energy in ``form`` (``pair_energy``). With ``h_shift`` > 0 every H site, for the pair energy and Coulomb
    alike, is moved that far towards its bonded atom. With ``envelope`` (a width, Angstrom) every pair energy is
    multiplied by ``vdw_analysis.lj_cutoff_envelope``, the C2 switch that reaches zero at ``cutoff``; the
    shifted-force Coulomb is already zero in value and slope there."""
    x = cluster_batch.x
    if x is None or x.ndim != 2 or x.shape[1] < 3:
        raise ValueError('exp6_ff: the cluster does not carry force-field columns; build it from with_ff_columns()')
    ei = edges_dict['edge_index_inter']
    pos = cluster_batch.pos
    if h_shift:
        off = x[:, 2].round().long()
        is_h = off != 0
        partner = torch.arange(len(pos), device=pos.device) + off
        v = pos[partner] - pos
        step = h_shift * v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pos = pos + torch.where(is_h[:, None], step, torch.zeros_like(step))
    r = (pos[ei[0]] - pos[ei[1]]).norm(dim=-1)
    ti, tj = x[ei[0], 1].round().long(), x[ei[1], 1].round().long()
    rep, disp = pair_energy(r, ti, tj, tables, form, k_factor)
    if envelope is not None:
        from mxtaltools.analysis.vdw_analysis import lj_cutoff_envelope
        if not 0 < envelope <= cutoff:
            raise ValueError(f'exp6_ff: envelope width must be in (0, cutoff], got {envelope}')
        switch = lj_cutoff_envelope(r, cutoff, envelope)
        rep, disp = rep * switch, disp * switch
    g = cluster_batch.batch[ei[1]]
    ng = cluster_batch.num_graphs
    rep = scatter(rep, g, dim=0, dim_size=ng, reduce='sum')
    disp = scatter(disp, g, dim=0, dim_size=ng, reduce='sum')
    if coulomb:
        coul = scatter(shifted_force_coulomb(r, x[ei[0], 0], x[ei[1], 0], cutoff), g, dim=0, dim_size=ng,
                       reduce='sum')
    else:
        coul = torch.zeros_like(rep)
    return rep, disp, coul


@lru_cache(maxsize=None)
def _tail_integral(cutoff: float, envelope: Optional[float]) -> float:
    """int r^-4 (1 - S(r)) dr over r > 0, S the pair-sum switch: 1/(3 rc^3), plus the part the envelope removes
    inside the cutoff (trapezoid rule, 20001 points)."""
    k = 1.0 / (3.0 * cutoff ** 3)
    if envelope is not None:
        from mxtaltools.analysis.vdw_analysis import lj_cutoff_envelope
        r = torch.linspace(cutoff - envelope, cutoff, 20001, dtype=torch.float64)
        k += float(torch.trapezoid((1 - lj_cutoff_envelope(r, cutoff, envelope)) / r ** 4, r))
    return k


def dispersion_tail(types: torch.Tensor, batch: torch.Tensor, T_fc: torch.Tensor, sym_mult: torch.Tensor,
                    z_prime: torch.Tensor, tables: dict, cutoff: float, envelope: Optional[float] = None,
                    form: str = 'elj') -> torch.Tensor:
    """Per-molecule dispersion missing from the pair sum, for a homogeneous crystal, kJ/mol [num_graphs]:
    -(2 pi sym_mult / (V Z')) K sum_ik C6_ik over the asymmetric-unit atoms i, k (``types``, graph index ``batch``),
    V = |det T_fc|, C6 the form's r^-6 coefficient (``C`` for exp-6, ``C_lj`` for the ELJ form), K = 1/(3 rc^3) for a
    sharp cutoff plus, with an ``envelope``, the switched-off part of the last shell."""
    c6 = tables['C_lj'] if form == 'elj' else tables['C']
    counts = scatter(torch.nn.functional.one_hot(types, len(c6)).to(c6.dtype), batch, dim=0, dim_size=len(T_fc),
                     reduce='sum')
    s = torch.einsum('gi,ij,gj->g', counts, c6, counts)
    vol = torch.linalg.det(T_fc).abs()
    zp = z_prime.reshape(-1).to(s.dtype)
    sym = sym_mult.reshape(-1).to(s.dtype)
    return -2 * torch.pi * sym * s * _tail_integral(float(cutoff), envelope) / (vol * zp)


def cluster_exp6_energy(cluster_batch,
                        edges_dict: Optional[dict] = None,
                        params: str = 'w99',
                        *,
                        coulomb: bool,
                        form: str = 'elj',
                        k_factor: float = 2.5,
                        h_shift: Optional[float] = None,
                        tail: bool = True,
                        envelope: Optional[float] = None,
                        return_terms: bool = False):
    """Lattice energy per molecule, kJ/mol [num_graphs], from a cluster already built -- as ``analyze`` builds it --
    from a ``with_ff_columns`` batch, and its pair list (``edges_dict``, default ``cluster_batch.edges_dict``; the
    cutoff is the list's own). Other arguments as ``crystal_exp6_energy``; ``coulomb`` False drops the Coulomb term."""
    edges = cluster_batch.edges_dict if edges_dict is None else edges_dict
    cutoff = float(edges['cutoff'])
    tables = pair_tables(params, device=cluster_batch.pos.device, dtype=cluster_batch.pos.dtype)
    h = tables['h_shift'] if h_shift is None else float(h_shift)
    rep, disp, coul = cluster_exp6_terms(cluster_batch, edges, tables, cutoff, h_shift=h, coulomb=coulomb,
                                         envelope=envelope, form=form, k_factor=k_factor)
    norm = 2 * cluster_batch.z_prime.reshape(-1).to(rep.dtype)
    terms = {'rep': rep / norm, 'disp': disp / norm, 'coul': coul / norm}
    if tail:
        inside = cluster_batch.aux_ind == 0
        terms['tail'] = dispersion_tail(cluster_batch.x[inside, 1].round().long(), cluster_batch.batch[inside],
                                        cluster_batch.T_fc, cluster_batch.sym_mult, cluster_batch.z_prime, tables,
                                        cutoff, envelope, form)
    else:
        terms['tail'] = torch.zeros_like(rep)
    total = terms['rep'] + terms['disp'] + terms['coul'] + terms['tail']
    return (total, terms) if return_terms else total


def crystal_exp6_energy(crystal_batch,
                        params: str = 'w99',
                        *,
                        charges: Union[str, torch.Tensor, None],
                        form: str = 'elj',
                        k_factor: float = 2.5,
                        cutoff: float = 10.0,
                        h_shift: Optional[float] = None,
                        tail: bool = True,
                        envelope: Optional[float] = None,
                        supercell_size: int = 10,
                        std_orientation: bool = True,
                        max_num_neighbors: int = 10000,
                        return_terms: bool = False):
    """Lattice energy per molecule, kJ/mol [num_graphs], of each crystal in a batch whose cell and asymmetric-unit
    parameters are set (differentiable in them, as ``analyze`` is).

    Parameters
    ----------
    crystal_batch : MolCrystalData batch; not modified.
    params : ``'w99'`` (default) or ``'fit'``.
    charges : required. ``'mmff'`` or ``'gasteiger'`` (RDKit, from each molecule's geometry), ``'x'`` (the batch's
        stored charges; see ``with_ff_columns``), None (no Coulomb term), or a [n_atoms] tensor.
    form : ``'elj'`` (default; each pair's sigma and eps matched to its exp-6 minimum, in the production eLJ shape)
        or ``'exp6'`` (module docstring).
    k_factor : stiffness of the ELJ form's wall below sigma; 2.5 is the production eLJ value.
    cutoff : pair cutoff, Angstrom (the cluster is built to it, as ``analyze`` does).
    h_shift : H site shift into the X-H bond, Angstrom; None takes the set's own (W99 0.1, FIT 0).
    tail : add the dispersion the pair sum leaves out (beyond the cutoff, and inside the envelope window).
    envelope : width (Angstrom) of the C2 switch taking exp-6 pair energies to zero at the cutoff; None, the default,
        is a sharp cutoff (as eLJ in ``analyze``), whose energy steps by the pair energy as a pair crosses it.
    supercell_size, std_orientation : passed to ``mol2cluster``.
    max_num_neighbors : neighbour cap of the pair search; reaching it raises instead of truncating silently.
    return_terms : also return a dict of the per-molecule parts (``rep``, ``disp``, ``coul``, ``tail``).

    Raises
    ------
    RuntimeError
        If an asymmetric-unit atom has ``max_num_neighbors`` neighbours (the list may be truncated).
    """
    from mxtaltools.analysis.vdw_analysis import get_intermolecular_dists_dict
    ff_batch = with_ff_columns(crystal_batch, params, charges)
    cluster = ff_batch.mol2cluster(cutoff, supercell_size, std_orientation=std_orientation)
    edges = get_intermolecular_dists_dict(cluster, cutoff, max_num_neighbors=max_num_neighbors)
    if len(edges['edge_index_inter'][1]):
        counts = torch.bincount(edges['edge_index_inter'][1])
        if int(counts.max()) >= max_num_neighbors:
            raise RuntimeError(f'exp6_ff: an asymmetric-unit atom reached the neighbour cap {max_num_neighbors} at '
                               f'cutoff {cutoff}; the pair list may be truncated')
    return cluster_exp6_energy(cluster, edges, params, coulomb=charges is not None, form=form, k_factor=k_factor,
                               h_shift=h_shift, tail=tail, envelope=envelope, return_terms=return_terms)

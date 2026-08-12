"""
A simple, differentiable intramolecular force field for evaluating the builder.

Deliberately minimal -- harmonic bonds, harmonic angles, and a soft-core Lennard-Jones
over nonbonded pairs.

**The torsion term is opt-in.** ``ff_from_reference`` still omits it, so dihedrals there
are governed entirely by sterics -- that is what makes it a real test of the
parameterisation rather than a tautology, and its zero-at-reference property is what the
builder tests assert against. ``ff_from_graph`` populates it, because a steric-only
target is nearly flat (butanol's basins span ~0.27 kcal/mol against ~6.3 for random
states) and so cannot discriminate mode weighting.

Two choices worth knowing about:

*Bonds and angles run over the whole molecular graph, not the spanning tree.* Ring
closure then falls out of the bond term for free -- a non-tree ring bond is still a
graph bond, so it gets the same harmonic as any other. Restricting to tree terms would
also silently leave H-C-H angles at a methyl unconstrained (only the X-C-H angles are
in the tree, and 1-3 pairs are excluded from LJ), letting the three hydrogens collapse
onto one another at zero cost.

*The repulsive core is exponential, not r^-12.* Below sigma the LJ core is replaced by
an exponential matched in both value and slope, following ``exponential_edgewise_lj_energy``
in mxtaltools. That keeps the energy finite (~107 * epsilon at r=0) and the gradient
bounded, which is the difference between an optimiser that recovers from a bad starting
torsion and one that detonates on the first step.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from mxtaltools.conformers.builder import (BatchedTree, bond_angle, bond_length, dihedral,
                                           nonbonded_pairs)

# van der Waals radii (A), matching mxtaltools.constants.atom_properties.VDW_RADII
VDW_RADII = {
    0: 0.0, 1: 1.2, 2: 1.4, 3: 2.2, 4: 1.9, 5: 1.8, 6: 1.7, 7: 1.6, 8: 1.55, 9: 1.5,
    10: 1.54, 11: 2.4, 12: 2.2, 13: 2.1, 14: 2.1, 15: 1.95, 16: 1.8, 17: 1.8, 18: 1.88,
    19: 2.8, 20: 2.4, 33: 2.05, 34: 1.9, 35: 1.9, 36: 2.02, 52: 2.1, 53: 2.1, 54: 2.16,
}

# sigma = 2^(-1/6) * (R_i + R_j) puts the LJ *minimum* at the sum of vdW radii, the
# usual contact convention. mxtaltools' crystal code sets sigma = R_i + R_j directly,
# which places the minimum 12% further out; harmless between molecules, but for
# intramolecular 1-5 contacts (typically 2.5-3.5 A) it is purely repulsive and inflates
# the molecule. Pass sigma_scale=1.0 to reproduce the crystal-side convention exactly.
CONTACT_SIGMA_SCALE = 2.0 ** (-1.0 / 6.0)


@dataclass
class ForceField:
    """Per-term parameters, aligned with the index arrays they are evaluated on."""

    bond_index: torch.Tensor      # [n_bonds, 2]   all graph bonds, tree and ring-closure
    bond_batch: torch.Tensor
    r0: torch.Tensor              # [n_bonds]
    k_bond: torch.Tensor

    angle_index: torch.Tensor     # [n_angles, 3]  all graph angle triples
    angle_batch: torch.Tensor
    theta0: torch.Tensor          # [n_angles]
    k_angle: torch.Tensor

    pair_index: torch.Tensor      # [n_pairs, 2]
    pair_batch: torch.Tensor
    sigma: torch.Tensor           # [n_pairs]
    epsilon: torch.Tensor         # [n_pairs]

    closure_index: torch.Tensor   # [n_broken, 2] ring bonds absent from the tree
    closure_batch: torch.Tensor
    closure_r0: torch.Tensor      # [n_broken]

    # Proper torsions, OPTIONAL: None means no torsion term, which is what
    # ff_from_reference produces and what the builder tests assume.
    torsion_index: Optional[torch.Tensor] = None    # [n_tors, 4] over the whole graph
    torsion_batch: Optional[torch.Tensor] = None
    tors_v: Optional[torch.Tensor] = None           # [n_tors] PK/IDIVF, kcal/mol
    tors_n: Optional[torch.Tensor] = None           # [n_tors] periodicity
    tors_gamma: Optional[torch.Tensor] = None       # [n_tors] phase, rad

    lj_k_factor: float = 2.5      # soft-core stiffness; 2.5 matches mxtaltools' default
    n_mols: int = 1

    def to(self, device) -> "ForceField":
        moved = {f: getattr(self, f) for f in self.__dataclass_fields__}
        for k, v in moved.items():
            if torch.is_tensor(v):
                moved[k] = v.to(device)
        return ForceField(**moved)


def graph_angle_index(tree: BatchedTree) -> torch.Tensor:
    """Every ``(i, j, k)`` with ``j`` bonded to both ``i`` and ``k``, ``i < k``.

    This is the full redundant angle set over the molecular graph -- strictly larger
    than the tree's N-2 angles.
    """
    bonds = tree.graph_bond_index.cpu().numpy()
    neigh = {}
    for i, j in bonds:
        neigh.setdefault(int(i), []).append(int(j))
        neigh.setdefault(int(j), []).append(int(i))

    triples = []
    for apex, nbrs in neigh.items():
        nbrs = sorted(nbrs)
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                triples.append((nbrs[a], apex, nbrs[b]))
    arr = np.asarray(sorted(triples), dtype=np.int64).reshape(-1, 3)
    return torch.from_numpy(arr).to(tree.device)


def graph_torsion_index(tree: BatchedTree) -> torch.Tensor:
    """Every proper dihedral ``(i, j, k, l)`` with ``j-k`` a graph bond.

    The full redundant set, matching how bonds and angles are handled here rather than
    the tree's minimal one -- a C-C bond in an alkane contributes all 9 of its H/C
    quartets, which is what the AMBER-family divisors assume. Terminal bonds contribute
    nothing, since a degree-1 endpoint has no other substituent to complete the quartet.
    """
    bonds = tree.graph_bond_index.cpu().numpy()
    neigh = {}
    for i, j in bonds:
        neigh.setdefault(int(i), []).append(int(j))
        neigh.setdefault(int(j), []).append(int(i))

    quartets = []
    for j, k in bonds:
        j, k = int(j), int(k)
        for i in sorted(neigh[j]):
            if i == k:
                continue
            for l in sorted(neigh[k]):
                if l == j or l == i:      # l == i closes a 3-ring: degenerate dihedral
                    continue
                quartets.append((i, j, k, l))
    arr = np.asarray(sorted(quartets), dtype=np.int64).reshape(-1, 4)
    return torch.from_numpy(arr).to(tree.device)


def ff_from_reference(tree: BatchedTree,
                      pos: torch.Tensor,
                      k_bond: float = 300.0,
                      k_angle: float = 50.0,
                      epsilon: float = 0.1,
                      min_separation: int = 3,
                      scale_14: float = 0.5,
                      sigma_scale: float = CONTACT_SIGMA_SCALE,
                      lj_k_factor: float = 2.5) -> ForceField:
    """Build a force field whose bonded minima sit at a reference conformer's geometry.

    ``r0`` and ``theta0`` are read off ``pos`` per-term, so the bonded part is exactly
    minimised by the reference local structure and the torsions are left to sterics.
    This stands in for what the 2-/3-body heads would eventually emit.

    Units are kcal/mol, A, rad. Defaults: ``k_bond`` 300 kcal/mol/A^2 and ``k_angle``
    50 kcal/mol/rad^2 are ordinary MMFF-scale constants; ``epsilon`` 0.1 kcal/mol is a
    typical C...C dispersion well. The bonded/nonbonded balance is the knob that decides
    whether sterics can distort local geometry.

    ``min_separation=3`` retains 1-4 pairs scaled by ``scale_14`` (the AMBER/OPLS
    convention). Excluding them entirely leaves short molecules with no torsional signal
    at all, since nothing else in this force field depends on a dihedral.
    """
    bond_index = tree.graph_bond_index
    bond_batch = tree.graph_bond_batch
    angle_index = graph_angle_index(tree)
    angle_batch = tree.batch[angle_index[:, 1]]

    r0 = bond_length(pos[bond_index[:, 0]], pos[bond_index[:, 1]]).detach()
    theta0 = bond_angle(pos[angle_index[:, 0]], pos[angle_index[:, 1]],
                        pos[angle_index[:, 2]]).detach()

    pair_index, pair_batch, separation = nonbonded_pairs(tree, min_separation)

    radii = torch.zeros(max(VDW_RADII) + 1, dtype=pos.dtype, device=pos.device)
    for z, r in VDW_RADII.items():
        radii[z] = r
    z = tree.z.to(pos.device)
    sigma = sigma_scale * (radii[z[pair_index[:, 0]]] + radii[z[pair_index[:, 1]]])

    eps = torch.full_like(sigma, float(epsilon))
    eps = torch.where(separation.to(pos.device) == 3, eps * scale_14, eps)

    closure = tree.broken_bond_index
    closure_r0 = (bond_length(pos[closure[:, 0]], pos[closure[:, 1]]).detach()
                  if closure.numel() else torch.zeros(0, dtype=pos.dtype, device=pos.device))

    return ForceField(
        bond_index=bond_index, bond_batch=bond_batch,
        r0=r0, k_bond=torch.full_like(r0, float(k_bond)),
        angle_index=angle_index, angle_batch=angle_batch,
        theta0=theta0, k_angle=torch.full_like(theta0, float(k_angle)),
        pair_index=pair_index, pair_batch=pair_batch,
        sigma=sigma, epsilon=eps,
        closure_index=closure, closure_batch=tree.broken_bond_batch,
        closure_r0=closure_r0,
        lj_k_factor=lj_k_factor, n_mols=tree.n_mols,
    )


# ---------------------------------------------------- graph-derived parameters

AtomType = Tuple[int, int]      # (atomic number, degree)

# Equilibrium internals keyed by atom TYPE, where a type is ``(element, degree)`` --
# the same coarse typing prior.py uses, with degree standing in for hybridisation.
# Every key is read off the molecular graph, so a force field built from these is a
# deterministic function of the graph and nothing else.
#
# That is the entire point of this table's existence next to ff_from_reference: the
# reference version measures r0/theta0 off whichever conformer RDKit embedded, which
# makes the energy -- and therefore log Z -- depend on an embedding seed. For a single
# unconditional molecule that is an arbitrary but harmless convention. For a CONDITIONAL
# model it is fatal, because log Z(c) acquires a per-molecule offset with no functional
# relationship to the graph, and the Z head cannot fit what the conditioner cannot see.
#
# Values are GAFF-family constants covering the alkane/alcohol subset. Units A, rad,
# kcal/mol, and k is the coefficient of (x - x0)^2 with no 1/2, matching ff_from_reference.
BOND_PARAMS: Dict[Tuple[AtomType, AtomType], Tuple[float, float]] = {
    ((1, 1), (6, 4)): (1.093, 340.0),      # C(sp3)-H
    ((6, 4), (6, 4)): (1.526, 310.0),      # C(sp3)-C(sp3)
    ((6, 4), (8, 2)): (1.410, 320.0),      # C(sp3)-O
    ((1, 1), (8, 2)): (0.960, 553.0),      # O-H
}

# ``(outer, apex, outer)`` with the two outer types sorted.
ANGLE_PARAMS: Dict[Tuple[AtomType, AtomType, AtomType], Tuple[float, float]] = {
    ((1, 1), (6, 4), (1, 1)): (np.deg2rad(109.5), 35.0),
    ((1, 1), (6, 4), (6, 4)): (np.deg2rad(109.5), 46.0),
    ((6, 4), (6, 4), (6, 4)): (np.deg2rad(111.0), 63.0),
    ((1, 1), (6, 4), (8, 2)): (np.deg2rad(109.5), 50.0),
    ((6, 4), (6, 4), (8, 2)): (np.deg2rad(109.5), 67.0),
    ((1, 1), (8, 2), (6, 4)): (np.deg2rad(108.5), 47.0),
}

# Proper torsions, keyed by the sorted pair of CENTRAL atom types -- the AMBER-family
# "X-A-B-X" general form, where the barrier depends only on the rotated bond.
# Values are ``(PK/IDIVF, periodicity, phase_rad)``: the PER-DIHEDRAL coefficient, since
# every quartet about the bond is enumerated and each contributes.
#
# GAFF divides by a FIXED IDIVF (9 for c3-c3, 3 for c3-oh) rather than by the realised
# quartet count. That reproduces experimental barriers when the substituent counts match
# the assumption -- which they do for every bond in an n-alkanol, so this is exact here.
# On a tertiary or quaternary centre the realised count differs and the total barrier
# scales with it, exactly as it does in GAFF itself.
TORSION_PARAMS: Dict[Tuple[AtomType, AtomType], Tuple[float, int, float]] = {
    ((6, 4), (6, 4)): (1.40 / 9.0, 3, 0.0),    # ethane-like, 2.8 kcal/mol total barrier
    ((6, 4), (8, 2)): (0.50 / 3.0, 3, 0.0),    # methanol-like, 1.0 kcal/mol total
}


def atom_types(tree: BatchedTree) -> list:
    """Per-atom ``(element, degree)`` tuples, in placement order.

    Degree counts bonds in the full molecular graph rather than the spanning tree --
    a tree degree would type a ring atom differently depending on which bond the tree
    happened to break.
    """
    z = tree.z.cpu().numpy()
    deg = np.bincount(tree.graph_bond_index.cpu().numpy().reshape(-1),
                      minlength=tree.n_atoms)
    return [(int(zi), int(di)) for zi, di in zip(z, deg)]


def _lookup(keys: list, table: dict, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """Table lookup that reports EVERY missing type at once.

    Raises rather than falling back to a generic constant. A silent default here would
    produce a force field that is quietly wrong for exactly the atoms it could not type,
    and the failure would surface thousands of steps later as an unlearnable Z.
    """
    missing = sorted({k for k in keys if k not in table})
    if missing:
        raise KeyError(
            f"{len(missing)} untyped {kind} type(s) in BOND_PARAMS/ANGLE_PARAMS: "
            f"{missing}. Types are (element, degree). Add them to the table in "
            f"conformers/energy.py -- this raises rather than defaulting because a "
            f"generic constant would silently misparameterise these terms.")
    vals = np.array([table[k] for k in keys], dtype=np.float64).reshape(-1, 2)
    return vals[:, 0], vals[:, 1]


def ff_from_graph(tree: BatchedTree,
                  dtype=torch.float64,
                  epsilon: float = 0.1,
                  min_separation: int = 3,
                  scale_14: float = 0.5,
                  sigma_scale: float = CONTACT_SIGMA_SCALE,
                  lj_k_factor: float = 2.5) -> ForceField:
    """A force field determined entirely by the molecular graph -- no reference conformer.

    Same ``ForceField`` output as ``ff_from_reference``, so it drops into
    ``intramolecular_energy`` unchanged; the difference is where r0/theta0/k come from.
    Here they are typed lookups on ``(element, degree)``, so two runs that embed
    different conformers of the same molecule get bit-identical parameters.

    Still no torsion term -- dihedrals remain governed by sterics, as in
    ``ff_from_reference``. Adding one is the next step, and it is what makes the target
    discriminating rather than nearly flat.
    """
    device = tree.device
    types = atom_types(tree)

    bond_index = tree.graph_bond_index
    bond_pairs = bond_index.cpu().numpy()
    bond_keys = [tuple(sorted((types[i], types[j]))) for i, j in bond_pairs]
    r0_np, kb_np = _lookup(bond_keys, BOND_PARAMS, "bond")

    angle_index = graph_angle_index(tree)
    angle_triples = angle_index.cpu().numpy()
    angle_keys = [(min(types[i], types[k]), types[j], max(types[i], types[k]))
                  for i, j, k in angle_triples]
    th0_np, ka_np = _lookup(angle_keys, ANGLE_PARAMS, "angle")

    tt = lambda a: torch.tensor(a, dtype=dtype, device=device)
    r0, k_bond = tt(r0_np), tt(kb_np)
    theta0, k_angle = tt(th0_np), tt(ka_np)

    # torsions are typed on the CENTRAL bond only, so bonds that generate no quartet
    # (a degree-1 endpoint) never reach the table and need no parameters there
    torsion_index = graph_torsion_index(tree)
    quartets = torsion_index.cpu().numpy()
    tors_keys = [tuple(sorted((types[j], types[k]))) for _, j, k, _ in quartets]
    missing = sorted({k for k in tors_keys if k not in TORSION_PARAMS})
    if missing:
        raise KeyError(
            f"{len(missing)} untyped torsion centre(s): {missing}. Keys are the sorted "
            f"pair of central (element, degree) types; add them to TORSION_PARAMS.")
    tp = np.array([TORSION_PARAMS[k] for k in tors_keys], dtype=np.float64).reshape(-1, 3)

    pair_index, pair_batch, separation = nonbonded_pairs(tree, min_separation)

    radii = torch.zeros(max(VDW_RADII) + 1, dtype=dtype, device=device)
    for z, r in VDW_RADII.items():
        radii[z] = r
    z = tree.z.to(device)
    sigma = sigma_scale * (radii[z[pair_index[:, 0]]] + radii[z[pair_index[:, 1]]])
    eps = torch.full_like(sigma, float(epsilon))
    eps = torch.where(separation.to(device) == 3, eps * scale_14, eps)

    # closure r0 is the TYPED length of that bond, not an as-built one: a ring closure
    # is an ordinary graph bond and gets the ordinary parameter, which is what keeps
    # the closure restraint graph-determined along with everything else.
    closure = tree.broken_bond_index
    if closure.numel():
        ckeys = [tuple(sorted((types[i], types[j]))) for i, j in closure.cpu().numpy()]
        closure_r0 = tt(_lookup(ckeys, BOND_PARAMS, "bond")[0])
    else:
        closure_r0 = torch.zeros(0, dtype=dtype, device=device)

    return ForceField(
        bond_index=bond_index, bond_batch=tree.graph_bond_batch,
        r0=r0, k_bond=k_bond,
        angle_index=angle_index, angle_batch=tree.batch[angle_index[:, 1]],
        theta0=theta0, k_angle=k_angle,
        pair_index=pair_index, pair_batch=pair_batch,
        sigma=sigma, epsilon=eps,
        closure_index=closure, closure_batch=tree.broken_bond_batch,
        closure_r0=closure_r0,
        torsion_index=torsion_index,
        torsion_batch=tree.batch[torsion_index[:, 1]] if torsion_index.numel()
        else torch.zeros(0, dtype=torch.long, device=device),
        tors_v=tt(tp[:, 0]), tors_n=tt(tp[:, 1]), tors_gamma=tt(tp[:, 2]),
        lj_k_factor=lj_k_factor, n_mols=tree.n_mols,
    )


def soft_core_lj(dist: torch.Tensor, sigma: torch.Tensor, epsilon: torch.Tensor,
                 k_factor: float = 2.5) -> torch.Tensor:
    """Lennard-Jones with the repulsive core replaced by a matched exponential.

    Above sigma this is plain 12-6. Below it, ``A exp(-k(r - sigma)) + B`` with ``k =
    k_factor / sigma``, ``A = -F0 / k``, ``B = -A``, chosen so value and slope agree with
    LJ at ``r = sigma`` (both zero and ``-24 eps / sigma`` respectively). Finite at r=0.
    """
    sr6 = (sigma / dist).pow(6)
    lj = 4.0 * epsilon * (sr6 * sr6 - sr6)

    k = k_factor / sigma
    a = 24.0 * epsilon / k_factor          # = -F0 / k with F0 = -24 eps / sigma
    rep = a * torch.exp(-k * (dist - sigma)) - a

    return torch.where(dist < sigma, rep, lj)


def intramolecular_energy(tree: BatchedTree, pos: torch.Tensor, ff: ForceField,
                          components: bool = False):
    """Total energy per molecule, ``[n_mols]``, in kcal/mol.

    Evaluated from Cartesian positions rather than from internal coordinates, so the
    identical function scores an internal-coordinate parameterisation and a Cartesian
    one -- which is what makes comparing them a fair test.
    """
    zeros = lambda: torch.zeros(ff.n_mols, dtype=pos.dtype, device=pos.device)

    r = bond_length(pos[ff.bond_index[:, 0]], pos[ff.bond_index[:, 1]])
    e_bond = zeros().index_add(0, ff.bond_batch, ff.k_bond * (r - ff.r0) ** 2)

    theta = bond_angle(pos[ff.angle_index[:, 0]], pos[ff.angle_index[:, 1]],
                       pos[ff.angle_index[:, 2]])
    e_angle = zeros().index_add(0, ff.angle_batch, ff.k_angle * (theta - ff.theta0) ** 2)

    d = bond_length(pos[ff.pair_index[:, 0]], pos[ff.pair_index[:, 1]])
    e_lj = zeros().index_add(0, ff.pair_batch,
                             soft_core_lj(d, ff.sigma, ff.epsilon, ff.lj_k_factor))

    # V[1 + cos(n phi - gamma)]: unlike the harmonic terms this has no zero at a
    # reference geometry, so a force field carrying it cannot be tested for
    # zero-at-reference the way ff_from_reference is.
    e_tors = zeros()
    if ff.torsion_index is not None and ff.torsion_index.numel():
        t = ff.torsion_index
        phi = dihedral(pos[t[:, 0]], pos[t[:, 1]], pos[t[:, 2]], pos[t[:, 3]])
        e_tors = zeros().index_add(
            0, ff.torsion_batch,
            ff.tors_v * (1.0 + torch.cos(ff.tors_n * phi - ff.tors_gamma)))

    total = e_bond + e_angle + e_lj + e_tors
    if components:
        return total, {"bond": e_bond, "angle": e_angle, "lj": e_lj, "torsion": e_tors}
    return total


def worst_clash(tree: BatchedTree, pos: torch.Tensor, ff: ForceField) -> torch.Tensor:
    """Deepest nonbonded overlap per molecule, ``max(sigma - r)``, in A. Zero if clean."""
    d = bond_length(pos[ff.pair_index[:, 0]], pos[ff.pair_index[:, 1]])
    overlap = torch.relu(ff.sigma - d)
    out = torch.zeros(ff.n_mols, dtype=pos.dtype, device=pos.device)
    return out.scatter_reduce(0, ff.pair_batch, overlap, reduce="amax", include_self=True)


def closure_error(pos: torch.Tensor, ff: ForceField) -> torch.Tensor:
    """Max |r - r0| over ring-closure bonds per molecule, ``[n_mols]``, in A.

    Zero for acyclic molecules. Because the bond term already runs over the whole graph,
    non-tree bonds are already restrained -- this is a diagnostic, not a separate penalty.
    """
    out = torch.zeros(ff.n_mols, dtype=pos.dtype, device=pos.device)
    e = ff.closure_index
    if e.numel() == 0:
        return out
    err = (bond_length(pos[e[:, 0]], pos[e[:, 1]]) - ff.closure_r0).abs()
    return out.scatter_reduce(0, ff.closure_batch, err, reduce="amax", include_self=True)

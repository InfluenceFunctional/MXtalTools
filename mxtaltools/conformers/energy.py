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
from mxtaltools.conformers.geometry import wilson_angle

# --- MMFF94 unit conversions and shape constants. -------------------------------------
# Every one of these was recovered by fitting RDKit's own per-term energies rather than
# transcribed, and each is pinned by test_mmff_matches_rdkit. See ff_from_mmff.
MDYNE_A = 143.9325          # mdyn/A -> kcal/mol/A^2
ANG_PREFACTOR = 0.043844    # MMFF bending prefactor, deg^-2
MMFF_CS = -2.0              # cubic stretch constant, A^-1
MMFF_CB = -0.006981317      # cubic bend constant, deg^-1
MMFF_SB = 2.51210           # stretch-bend prefactor, kcal/(mol A deg)
COULOMB = 332.0716          # e^2/A -> kcal/mol
MMFF_ELE_DELTA = 0.05       # electrostatic buffering distance, A
MMFF_ELE_14_SCALE = 0.75    # MMFF scales 1-4 electrostatics, nothing beyond
MMFF_LINEAR_THETA0 = 179.99  # theta0 at or above this marks an MMFF *linear* centre

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

    # --- MMFF completion. -------------------------------------------------------------
    # Every field below is None on the hand-written force fields, which keeps
    # ff_from_reference and ff_from_graph on exactly the harmonic + soft-core path they
    # were written and tested against. ff_from_mmff populates all of them, and each
    # `is None` branch below is the difference between "this force field does not carry
    # the term" and "this force field carries it as zero" -- do not collapse them.
    bond_cs: Optional[float] = None          # enables MMFF's quartic stretch
    angle_cb: Optional[float] = None         # enables MMFF's cubic bend
    angle_linear: Optional[torch.Tensor] = None   # [n_angles] bool, MMFF linear centres

    sb_index: Optional[torch.Tensor] = None  # [n_sb, 3] stretch-bend triples (i, j, k)
    sb_batch: Optional[torch.Tensor] = None
    sb_k_ijk: Optional[torch.Tensor] = None  # [n_sb] kba for the i-j bond
    sb_k_kji: Optional[torch.Tensor] = None  # [n_sb] kba for the k-j bond
    sb_r0_ij: Optional[torch.Tensor] = None
    sb_r0_kj: Optional[torch.Tensor] = None
    sb_theta0: Optional[torch.Tensor] = None  # [n_sb] radians

    oop_index: Optional[torch.Tensor] = None  # [n_oop, 4] as (i, j CENTRAL, k, l)
    oop_batch: Optional[torch.Tensor] = None
    oop_k: Optional[torch.Tensor] = None      # [n_oop], pre-scaled for chi in RADIANS

    # Electrostatics reuse pair_index, so they inherit its separation rule exactly.
    ele_qq: Optional[torch.Tensor] = None     # [n_pairs] q_i q_j
    ele_scale: Optional[torch.Tensor] = None  # [n_pairs] 0.75 on 1-4, else 1
    ele_delta: float = MMFF_ELE_DELTA
    ele_dielectric: float = 1.0

    # vdW: R*_ij present means MMFF's buffered 14-7 rather than this module's soft-core
    # LJ. vdw_softcore_frac > 0 softens it below that fraction of R* (see buffered_147).
    vdw_rstar: Optional[torch.Tensor] = None
    vdw_softcore_frac: float = 0.0

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


def _repeated_block(rows: np.ndarray, n_mols: int):
    """``(first block, repeat count)`` if ``rows`` is one block repeated, else ``(rows, 1)``.

    MMFF parameters are a function of the atom pair, so a batch that is one molecule
    replicated repeats the first molecule's answers and they can be looked up once and
    tiled -- which is what every other term in ``ff_from_mmff`` already does. Querying
    RDKit per batch row instead made that loop scale with the batch (12.45M pairs for a
    50k-row draw on a 26-atom molecule).

    The repetition is CHECKED, never inferred from ``n_mols``: the index arrays reaching
    here are reduced modulo the molecule size, so a genuinely heterogeneous batch can
    still have a divisible row count, and tiling one there would give every replica the
    first molecule's parameters with nothing to report it.
    """
    if n_mols < 2 or len(rows) % n_mols:
        return rows, 1
    k = len(rows) // n_mols
    first = rows[:k]
    if not np.array_equal(rows.reshape(n_mols, *first.shape),
                          np.broadcast_to(first, (n_mols, *first.shape))):
        return rows, 1
    return first, n_mols


def ff_from_mmff(tree: BatchedTree, rdkit_mol, perm, dtype=torch.float64,
                 min_separation: int = 3, lj_k_factor: float = 2.5,
                 vdw_softcore_frac: float = 0.0, dielectric: float = 1.0):
    """A ForceField whose parameters come from RDKit's MMFF94 typing.

    WHY NOT EXTEND THE HAND-WRITTEN TABLES. ff_from_graph's tables cover four bond types,
    six angle types and two torsion centres -- alkanes and alcohols -- and `_lookup` raises
    on anything else, so a peptide or an aromatic fails at construction. Transcribing a
    published force field by hand is thousands of numbers, and a wrong one produces a
    perfectly plausible energy that nothing downstream can detect. RDKit already carries a
    fully parameterised MMFF94 and will compute the reference energy itself, which makes
    the conversion CHECKABLE rather than asserted -- see the term-by-term test.

    Like ff_from_graph this is a deterministic function of the molecular graph: MMFF typing
    reads connectivity and bond orders, not geometry. So it keeps the property that
    motivated ff_from_graph -- log Z(c) stops depending on which conformer was embedded.

    THIS IS A COMPLETE COPY OF MMFF94: all seven terms, in MMFF's own functional forms.
    test_mmff_matches_rdkit checks each term SEPARATELY against RDKit at a perturbed
    geometry -- perturbed so that a term sitting near zero cannot pass by being small.
    Bond, torsion, out-of-plane, electrostatic and vdW agree to ~1e-14; angle and
    stretch-bend to ~3e-4 kcal/mol, which is not an error in the form but the rounding of
    what RDKit's accessors report (theta0 comes back at three decimals -- the residual
    tracks that rounding across molecule sizes, and is 5e-4 of kT).

    NONE of the unit constants here were transcribed from the literature; each was
    recovered by fitting RDKit's own per-term energies, because a mistyped constant
    produces a plausible energy that nothing downstream can detect. Two were wrong on the
    first pass and were caught only because the check is per-term rather than on the
    total: vdW needs GetMMFFVdWParams' THIRD and FOURTH returns (the donor-acceptor
    rescaled R* and eps), not its first two, which matters on any H-bonding molecule; and
    the torsion sign convention is only pinned by molecules whose dihedrals sit away from
    zero, since at phi=0 the V2 term vanishes under either sign.

    ONE DELIBERATE DEPARTURE, off by default: vdW may be softened below a fraction of R*
    via `vdw_softcore_frac`, since buffered 14-7 reaches ~1.5e9 * eps at r=0 and one
    overlapping pair can otherwise outweigh a whole batch. At the default 0 this is exact
    MMFF. See buffered_147.

    `perm` is TreeSpec.perm -- perm[slot] is the RDKit atom index of that placement slot.
    """
    from rdkit.Chem import AllChem

    props = AllChem.MMFFGetMoleculeProperties(rdkit_mol)
    if props is None:
        raise ValueError('RDKit could not MMFF-type this molecule')
    device = tree.device
    perm = np.asarray(perm)
    slot = np.empty(len(perm), dtype=np.int64)
    slot[perm] = np.arange(len(perm))          # slot[rdkit_index] = placement slot
    n_at, n_mols = len(perm), tree.n_mols
    if tree.n_atoms != n_at * n_mols:
        raise ValueError(
            f'ff_from_mmff expects {n_mols} copies of one {n_at}-atom molecule, got '
            f'{tree.n_atoms} atoms; MMFF parameters are per-molecule and are tiled here')

    def tile_idx(a):
        a = np.asarray(a, dtype=np.int64).reshape(len(a), -1) if len(a) else None
        if a is None:
            return None
        off = (np.arange(n_mols) * n_at)[:, None, None]
        return torch.as_tensor((a[None] + off).reshape(-1, a.shape[1]),
                               dtype=torch.long, device=device)

    def tile_val(v):
        return torch.as_tensor(np.tile(np.asarray(v, dtype=np.float64), n_mols),
                               dtype=dtype, device=device)

    def batch_of(k):
        return torch.repeat_interleave(torch.arange(n_mols, device=device), k)

    # ---- bonds: every graph bond, harmonic with MMFF's kb and r0 ----
    bi, r0, kb = [], [], []
    for b in rdkit_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pr = props.GetMMFFBondStretchParams(rdkit_mol, i, j)
        if pr is None:
            raise ValueError(f'MMFF has no bond parameters for atoms {i}-{j}')
        _, k_b, r_0 = pr
        bi.append([slot[i], slot[j]])
        r0.append(r_0)
        kb.append(143.9325 * k_b / 2.0)        # mdyn/A -> kcal/mol/A^2 in our k(r-r0)^2

    # ---- angles: the full graph set, theta0 in RADIANS, plus the linear-centre mask ----
    ai, th0, ka, a_lin = [], [], [], []
    # ---- stretch-bend, over the same triples: needs both bond r0 as well ----
    sbi, sb1, sb2, sb_rij, sb_rkj, sb_t0 = [], [], [], [], [], []
    for a in rdkit_mol.GetAtoms():
        j = a.GetIdx()
        ns = sorted(x.GetIdx() for x in a.GetNeighbors())
        for u in range(len(ns)):
            for v in range(u + 1, len(ns)):
                i, k = ns[u], ns[v]
                pr = props.GetMMFFAngleBendParams(rdkit_mol, i, j, k)
                if pr is None:
                    continue
                _, k_a, t_0 = pr
                ai.append([slot[i], slot[j], slot[k]])
                th0.append(np.radians(t_0))
                # 0.043844 * ka/2 * dtheta_deg^2 -> our k * dtheta_rad^2
                ka.append(ANG_PREFACTOR * k_a / 2.0 * (180.0 / np.pi) ** 2)
                a_lin.append(t_0 >= MMFF_LINEAR_THETA0)

                sp = props.GetMMFFStretchBendParams(rdkit_mol, i, j, k)
                if sp is None:
                    continue
                b_ij = props.GetMMFFBondStretchParams(rdkit_mol, i, j)
                b_kj = props.GetMMFFBondStretchParams(rdkit_mol, k, j)
                if b_ij is None or b_kj is None:
                    continue
                sbi.append([slot[i], slot[j], slot[k]])
                sb1.append(sp[1])
                sb2.append(sp[2])
                sb_rij.append(b_ij[2])
                sb_rkj.append(b_kj[2])
                sb_t0.append(np.radians(t_0))

    # ---- out-of-plane: all three Wilson angles at every trigonal centre ----
    # RDKit returns the same koop for all three permutations, i.e. MMFF sums over all
    # three rather than picking one, so enumerating all three is required, not redundant.
    oi, ok_ = [], []
    for a in rdkit_mol.GetAtoms():
        j = a.GetIdx()
        ns = sorted(x.GetIdx() for x in a.GetNeighbors())
        if len(ns) != 3:
            continue
        for c in range(3):
            i, k, l = ns[c], ns[(c + 1) % 3], ns[(c + 2) % 3]
            koop = props.GetMMFFOopBendParams(rdkit_mol, i, j, k, l)
            if koop is None:
                continue
            oi.append([slot[i], slot[j], slot[k], slot[l]])
            # 0.043844 * koop/2 * chi_deg^2 -> our k * chi_rad^2
            ok_.append(ANG_PREFACTOR * koop / 2.0 * (180.0 / np.pi) ** 2)

    # ---- torsions: MMFF's 3-term Fourier as three entries of our 1-term form ----
    # 0.5*(V1(1+cos p) + V2(1-cos 2p) + V3(1+cos 3p))
    #   == sum V(1 + cos(n p - gamma)) with (V1/2,1,0), (V2/2,2,pi), (V3/2,3,0)
    ti, tv, tn, tg = [], [], [], []
    for b in rdkit_mol.GetBonds():
        j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        for i in sorted(x.GetIdx() for x in rdkit_mol.GetAtomWithIdx(j).GetNeighbors()):
            if i == k:
                continue
            for l in sorted(x.GetIdx() for x in rdkit_mol.GetAtomWithIdx(k).GetNeighbors()):
                if l == j or l == i:
                    continue
                pr = props.GetMMFFTorsionParams(rdkit_mol, i, j, k, l)
                if pr is None:
                    continue
                _, V1, V2, V3 = pr
                for V, n_per, gam in ((V1, 1, 0.0), (V2, 2, np.pi), (V3, 3, 0.0)):
                    if V == 0.0:
                        continue
                    ti.append([slot[i], slot[j], slot[k], slot[l]])
                    tv.append(V / 2.0)
                    tn.append(float(n_per))
                    tg.append(gam)

    pair_index, pair_batch, separation = nonbonded_pairs(tree, min_separation)
    pi_np = pair_index.detach().cpu().numpy() % n_at            # back to one molecule
    pi_unique, pair_reps = _repeated_block(pi_np, n_mols)
    from rdkit import Chem as _Chem
    path = _Chem.GetDistanceMatrix(rdkit_mol)                   # bonds between atoms
    sig, eps, rstar, qq, qscale = [], [], [], [], []
    for a, b in pi_unique:
        ra, rb = int(perm[int(a)]), int(perm[int(b)])
        # GetMMFFVdWParams returns (R*, eps, R*_da, eps_da): the LAST TWO carry MMFF's
        # donor-acceptor rescaling (DARAD 0.8 / DAEPS 0.5). Using the first two silently
        # mis-scores every hydrogen bond -- it cost 3e-2 kcal/mol on N-methylacetamide
        # while every non-H-bonding molecule still matched to 1e-14.
        _, _, R, e = props.GetMMFFVdWParams(ra, rb)
        rstar.append(R)
        sig.append(R * CONTACT_SIGMA_SCALE)
        eps.append(e)
        qq.append(props.GetMMFFPartialCharge(ra) * props.GetMMFFPartialCharge(rb))
        qscale.append(MMFF_ELE_14_SCALE if path[ra, rb] == 3 else 1.0)

    pair_val = lambda v: torch.as_tensor(
        np.tile(np.asarray(v, dtype=np.float64), pair_reps), dtype=dtype, device=device)

    closure = tree.broken_bond_index
    if closure.numel():
        # same tiling as the pairs: each replica repeats the first molecule's closure bonds
        cl_unique, cl_reps = _repeated_block(closure.detach().cpu().numpy() % n_at, n_mols)
        cr0 = []
        for a, b in cl_unique:
            pr = props.GetMMFFBondStretchParams(rdkit_mol, int(perm[int(a)]),
                                                int(perm[int(b)]))
            cr0.append(pr[2] if pr is not None else 1.5)
        closure_r0 = torch.as_tensor(np.tile(np.asarray(cr0), cl_reps),
                                     dtype=dtype, device=device)
    else:
        closure_r0 = torch.zeros(0, dtype=dtype, device=device)

    return ForceField(
        bond_index=tile_idx(bi), bond_batch=batch_of(len(bi)),
        r0=tile_val(r0), k_bond=tile_val(kb),
        angle_index=tile_idx(ai), angle_batch=batch_of(len(ai)),
        theta0=tile_val(th0), k_angle=tile_val(ka),
        pair_index=pair_index, pair_batch=pair_batch,
        sigma=pair_val(sig), epsilon=pair_val(eps),
        closure_index=closure, closure_batch=tree.broken_bond_batch,
        closure_r0=closure_r0,
        torsion_index=tile_idx(ti) if ti else None,
        torsion_batch=batch_of(len(ti)) if ti else None,
        tors_v=tile_val(tv) if ti else None,
        tors_n=tile_val(tn) if ti else None,
        tors_gamma=tile_val(tg) if ti else None,
        lj_k_factor=lj_k_factor, n_mols=n_mols,
        bond_cs=MMFF_CS, angle_cb=MMFF_CB,
        angle_linear=torch.as_tensor(np.tile(np.asarray(a_lin, dtype=bool), n_mols),
                                     dtype=torch.bool, device=device),
        sb_index=tile_idx(sbi) if sbi else None,
        sb_batch=batch_of(len(sbi)) if sbi else None,
        sb_k_ijk=tile_val(sb1) if sbi else None,
        sb_k_kji=tile_val(sb2) if sbi else None,
        sb_r0_ij=tile_val(sb_rij) if sbi else None,
        sb_r0_kj=tile_val(sb_rkj) if sbi else None,
        sb_theta0=tile_val(sb_t0) if sbi else None,
        oop_index=tile_idx(oi) if oi else None,
        oop_batch=batch_of(len(oi)) if oi else None,
        oop_k=tile_val(ok_) if oi else None,
        ele_qq=pair_val(qq), ele_scale=pair_val(qscale),
        ele_dielectric=dielectric,
        vdw_rstar=pair_val(rstar),
        vdw_softcore_frac=vdw_softcore_frac,
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


def buffered_147(dist: torch.Tensor, rstar: torch.Tensor, epsilon: torch.Tensor,
                 softcore_frac: float = 0.0) -> torch.Tensor:
    """MMFF94's buffered 14-7 van der Waals term.

    ``eps * (1.07 R / (r + 0.07 R))^7 * (1.12 R^7 / (r^7 + 0.12 R^7) - 2)``, exact as
    published. Reproduces RDKit to machine precision when ``softcore_frac`` is 0.

    WHY THE SOFTENING EXISTS. Buffered 14-7 is finite at r=0 but reaches ~1.5e9 * eps
    there, so a single overlapping pair in a sampled batch can carry more energy than
    every other molecule combined -- it is a wall in all but name. With
    ``softcore_frac = f > 0``, below ``r = f * R`` the term is continued LINEARLY,
    matched in value and slope. MMFF is reproduced exactly everywhere above ``f * R``,
    and f should sit well inside the repulsive wall (0.3 or less) so that no thermally
    reachable geometry is touched. It is 0 by default: fidelity first, opt in.

    WHY LINEAR AND NOT AN EXPONENTIAL. soft_core_lj continues 12-6 with an exponential,
    but it can afford to: LJ is zero at sigma, so the decay rate is free to choose. Here
    the switch sits partway up a very steep wall, and an exponential whose rate is fixed
    by matching that slope still reaches ~1.8e5 kcal/mol at r=0 -- it barely helps. A
    line has a CONSTANT gradient by construction, which is the property that actually
    matters for an optimiser recovering from a clash, and it stays finite and monotone.
    """
    r7, R7 = dist.pow(7), rstar.pow(7)
    a = 1.07 * rstar / (dist + 0.07 * rstar)
    e_full = epsilon * a.pow(7) * (1.12 * R7 / (r7 + 0.12 * R7) - 2.0)
    if softcore_frac <= 0.0:
        return e_full

    # value and slope of the 14-7 form at the switch radius, by hand so that this stays
    # a pure function (no autograd graph) and works under torch.no_grad
    rs = softcore_frac * rstar
    rs7 = rs.pow(7)
    a_s = 1.07 * rstar / (rs + 0.07 * rstar)
    g = 1.12 * R7 / (rs7 + 0.12 * R7) - 2.0
    v = epsilon * a_s.pow(7) * g
    da = -7.0 * a_s.pow(7) / (rs + 0.07 * rstar)
    dg = -1.12 * R7 * 7.0 * rs.pow(6) / (rs7 + 0.12 * R7) ** 2
    dv = epsilon * (da * g + a_s.pow(7) * dg)

    return torch.where(dist < rs, v + dv * (dist - rs), e_full)


def intramolecular_energy(tree: BatchedTree, pos: torch.Tensor, ff: ForceField,
                          components: bool = False):
    """Total energy per molecule, ``[n_mols]``, in kcal/mol.

    Evaluated from Cartesian positions rather than from internal coordinates, so the
    identical function scores an internal-coordinate parameterisation and a Cartesian
    one -- which is what makes comparing them a fair test.
    """
    zeros = lambda: torch.zeros(ff.n_mols, dtype=pos.dtype, device=pos.device)
    deg = 180.0 / np.pi

    # --- bonds: harmonic, or MMFF's quartic when bond_cs is set ---
    r = bond_length(pos[ff.bond_index[:, 0]], pos[ff.bond_index[:, 1]])
    dr = r - ff.r0
    e_b = ff.k_bond * dr ** 2
    if ff.bond_cs is not None:
        cs = ff.bond_cs
        e_b = e_b * (1.0 + cs * dr + (7.0 / 12.0) * cs ** 2 * dr ** 2)
    e_bond = zeros().index_add(0, ff.bond_batch, e_b)

    # --- angles: harmonic, or MMFF's cubic bend with the linear centres split out ---
    theta = bond_angle(pos[ff.angle_index[:, 0]], pos[ff.angle_index[:, 1]],
                       pos[ff.angle_index[:, 2]])
    dt = theta - ff.theta0
    e_a = ff.k_angle * dt ** 2
    if ff.angle_cb is not None:
        e_a = e_a * (1.0 + ff.angle_cb * dt * deg)
        if ff.angle_linear is not None and bool(ff.angle_linear.any()):
            # MMFF switches form at a linear centre: 143.9325 ka (1 + cos theta), which
            # has no cusp at 180 deg the way a bend in theta does. k_angle already holds
            # 0.043844 (ka/2) (180/pi)^2, and 0.043844 (180/pi)^2 == 143.9325, so the
            # linear force constant is exactly 2 k_angle -- no separate field needed.
            e_a = torch.where(ff.angle_linear,
                              2.0 * ff.k_angle * (1.0 + torch.cos(theta)), e_a)
    e_angle = zeros().index_add(0, ff.angle_batch, e_a)

    # --- vdW: soft-core 12-6, or MMFF's buffered 14-7 when R* is present ---
    d = bond_length(pos[ff.pair_index[:, 0]], pos[ff.pair_index[:, 1]])
    if ff.vdw_rstar is None:
        e_pair = soft_core_lj(d, ff.sigma, ff.epsilon, ff.lj_k_factor)
    else:
        e_pair = buffered_147(d, ff.vdw_rstar, ff.epsilon, ff.vdw_softcore_frac)
    e_lj = zeros().index_add(0, ff.pair_batch, e_pair)

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

    # --- stretch-bend: bond stretch and angle bend are not independent ---
    e_sb = zeros()
    if ff.sb_index is not None and ff.sb_index.numel():
        s = ff.sb_index
        r_ij = bond_length(pos[s[:, 0]], pos[s[:, 1]])
        r_kj = bond_length(pos[s[:, 2]], pos[s[:, 1]])
        th = bond_angle(pos[s[:, 0]], pos[s[:, 1]], pos[s[:, 2]])
        e_sb = zeros().index_add(
            0, ff.sb_batch,
            MMFF_SB * (ff.sb_k_ijk * (r_ij - ff.sb_r0_ij)
                       + ff.sb_k_kji * (r_kj - ff.sb_r0_kj)) * (th - ff.sb_theta0) * deg)

    # --- out-of-plane: the term that holds an sp2 centre planar ---
    # MMFF's koop can be NEGATIVE (amide N carries -0.02), which deliberately favours a
    # pyramidal centre, so this term is not sign-definite and must not be clamped.
    e_oop = zeros()
    if ff.oop_index is not None and ff.oop_index.numel():
        o = ff.oop_index
        chi = wilson_angle(pos[o[:, 0]], pos[o[:, 1]], pos[o[:, 2]], pos[o[:, 3]])
        e_oop = zeros().index_add(0, ff.oop_batch, ff.oop_k * chi ** 2)

    # --- electrostatics: on a polar molecule this dominates every other term ---
    e_ele = zeros()
    if ff.ele_qq is not None and ff.ele_qq.numel():
        e_ele = zeros().index_add(
            0, ff.pair_batch,
            ff.ele_scale * COULOMB * ff.ele_qq / (ff.ele_dielectric * (d + ff.ele_delta)))

    total = e_bond + e_angle + e_lj + e_tors + e_sb + e_oop + e_ele
    if components:
        return total, {"bond": e_bond, "angle": e_angle, "lj": e_lj, "torsion": e_tors,
                       "stretch_bend": e_sb, "oop": e_oop, "electrostatic": e_ele}
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

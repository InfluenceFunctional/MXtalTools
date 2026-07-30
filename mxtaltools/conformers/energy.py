"""
A simple, differentiable intramolecular force field for evaluating the builder.

Deliberately minimal -- harmonic bonds, harmonic angles, and a soft-core Lennard-Jones
over nonbonded pairs. **No torsion term**: dihedrals are governed entirely by sterics,
which is what makes this a real test of the parameterisation rather than a tautology.

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
from typing import Optional

import numpy as np
import torch

from mxtaltools.conformers.builder import BatchedTree, bond_angle, bond_length, nonbonded_pairs

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

    total = e_bond + e_angle + e_lj
    if components:
        return total, {"bond": e_bond, "angle": e_angle, "lj": e_lj}
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

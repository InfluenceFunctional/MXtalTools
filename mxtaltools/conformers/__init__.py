"""
Exact, batched, differentiable conformer construction from internal coordinates.

    from mxtaltools.conformers import spec_from_smiles, collate, build, measure

    mol, spec = spec_from_smiles("CCOc1ccccc1")
    tree = collate([spec], device="cuda")
    r, theta, phi = measure(tree, positions)     # reference internal coordinates
    pos = build(tree, r, theta, phi)             # exact round trip

Standalone: no dependency on the GFN, the energy models, or mxtaltools.
"""

from .builder import (BatchedTree, build, closure_length, collate, log_jacobian,
                      measure, nonbonded_pairs)
from .geometry import bond_angle, bond_length, dihedral, place_nerf
from .topology import TreeSpec, spec_from_mol, spec_from_smiles

__all__ = [
    "BatchedTree", "TreeSpec",
    "spec_from_mol", "spec_from_smiles", "collate",
    "build", "measure", "log_jacobian", "closure_length", "nonbonded_pairs",
    "bond_length", "bond_angle", "dihedral", "place_nerf",
]

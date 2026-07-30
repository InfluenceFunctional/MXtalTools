"""
Bond perception from coordinates.

Stored ``MolData`` carries ``z``, ``pos``, ``x`` and ``smiles`` but **no bond
topology**, so the molecular graph has to be recovered before a tree can be built.

mxtaltools' existing ``bond_edges_from_distances`` (crystal_rdf.py) uses a flat 1.8 A
cutoff, which is fine for the C/N/O core it was written for but mis-bonds anything
heavier -- C-I is 2.14 A and would be dropped, splitting the molecule. This is
element-aware instead: a pair is bonded when its separation is under
``tolerance * (rcov_i + rcov_j)``.
"""

import numpy as np

# Cordero et al. (2008) covalent radii, A. Values past Kr matter mainly for the
# halogens in drug-like sets; anything absent falls back to CADENCE_FALLBACK.
COVALENT_RADII = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
    9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60,
    23: 1.53, 24: 1.39, 25: 1.50, 26: 1.42, 27: 1.38, 28: 1.24, 29: 1.32,
    30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16,
    47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39,
    54: 1.40, 78: 1.36, 79: 1.36, 80: 1.32, 82: 1.46,
}
CADENCE_FALLBACK = 1.5


def covalent_radii_array(max_z: int = 100) -> np.ndarray:
    radii = np.full(max_z + 1, CADENCE_FALLBACK, dtype=np.float64)
    radii[0] = 0.0
    for z, r in COVALENT_RADII.items():
        if z <= max_z:
            radii[z] = r
    return radii


def infer_bond_index(z, pos, tolerance: float = 1.2) -> np.ndarray:
    """Perceive bonds from geometry. Returns ``[2, E]`` with ``i < j``, sorted.

    ``tolerance`` 1.2 is the usual value: generous enough for strained or
    slightly-distorted geometry, tight enough not to bond 1-3 neighbours.

    Raises if the result is disconnected, because a disconnected graph has no
    spanning tree and would otherwise fail later with a much less obvious message.
    """
    z = np.asarray(z, dtype=np.int64).reshape(-1)
    pos = np.asarray(pos, dtype=np.float64).reshape(len(z), 3)

    radii = covalent_radii_array(int(z.max()))[z]
    cutoff = tolerance * (radii[:, None] + radii[None, :])

    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((delta ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)

    i, j = np.nonzero(np.triu(dist < cutoff, k=1))
    if len(i) == 0:
        raise ValueError("bond perception found no bonds; check units of `pos`")

    n = len(z)
    seen, stack = np.zeros(n, dtype=bool), [0]
    adj = [[] for _ in range(n)]
    for a, b in zip(i, j):
        adj[a].append(b)
        adj[b].append(a)
    seen[0] = True
    while stack:
        for nb in adj[stack.pop()]:
            if not seen[nb]:
                seen[nb] = True
                stack.append(nb)
    if not seen.all():
        raise ValueError(
            f"perceived graph is disconnected ({int((~seen).sum())} of {n} atoms "
            f"unreachable) at tolerance={tolerance}; the molecule may be a complex, "
            f"or `pos` may not be a single connected conformer")

    return np.stack([i, j]).astype(np.int64)

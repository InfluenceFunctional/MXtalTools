"""
Per-molecule spanning-tree specs for internal-coordinate parameterisation.

One ``TreeSpec`` per molecule, built once on CPU from an RDKit mol and cached.
``builder.collate`` turns a list of them into batched index tensors.

Parameterisation
----------------
Atoms are relabelled into *placement order*. The atom at placement slot ``k``
carries ``min(k, 3)`` degrees of freedom:

    k = 0   root, at the origin                          0 DoF
    k = 1   bond only                                    1 DoF   (r)
    k = 2   bond + angle                                 2 DoF   (r, theta)
    k >= 3  bond + angle + dihedral                      3 DoF   (r, theta, phi)

so the total is ``0 + 1 + 2 + 3*(N-3) = 3N - 6`` exactly -- minimal, complete, and
SE(3)-invariant by construction. The 6 external DoF are consumed by the frame
convention in ``geometry.place_seed_*`` and never appear in the parameter vector.

Rings
-----
A spanning tree cannot represent ring-closure bonds. Those edges are recorded in
``broken_bond_index`` and their lengths become *determined* functions of the tree
DoF; closing them is the caller's problem (see ``builder.closure_error``).
"""

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# reference angles outside this band make the dependent dihedral ill-conditioned
_MIN_ANGLE = np.deg2rad(15.0)
_MAX_ANGLE = np.deg2rad(165.0)


@dataclass
class TreeSpec:
    """Internal-coordinate tree for a single molecule, in placement-order indexing."""

    n_atoms: int
    z: np.ndarray                 # [N] atomic numbers, placement order
    perm: np.ndarray              # [N] perm[k] = original RDKit index of slot k
    ref_a: np.ndarray             # [N] reference atoms, placement indexing, -1 = unused
    ref_b: np.ndarray
    ref_c: np.ndarray
    round_id: np.ndarray          # [N] placement round; rounds 0/1/2 are the frame seeds
    bond_index: np.ndarray        # [N-1, 2] (c, n)      aligned with the r vector
    angle_index: np.ndarray       # [N-2, 3] (b, c, n)   aligned with the theta vector
    torsion_index: np.ndarray     # [N-3, 4] (a, b, c, n) aligned with the phi vector
    torsion_is_proper: np.ndarray # [N-3] False where a-b is not a bond (near-root impropers)
    angle_is_linear: np.ndarray   # [N-2] theta ~ pi in the reference geometry
    torsion_frame_is_linear: np.ndarray  # [N-3] angle(a,b,c) ~ pi, so phi is ill-conditioned
    broken_bond_index: np.ndarray # [R, 2] graph bonds absent from the tree
    graph_bond_index: np.ndarray  # [E, 2] every bond, for nonbonded exclusion masks

    @property
    def n_dof(self) -> int:
        return 3 * self.n_atoms - 6

    @property
    def n_rounds(self) -> int:
        return int(self.round_id.max()) + 1


def canonical_rank(z: np.ndarray, neighbours: list, max_iter: int = 0) -> np.ndarray:
    """Weisfeiler-Lehman colour refinement: a canonical label per atom.

    Pure graph function -- no RDKit, no coordinates, no dependence on input atom
    numbering. Every ordering decision in the tree construction breaks ties on this
    rank rather than on atom index, which is what makes the parameterisation a
    property of the *molecule* rather than of how its SMILES happened to parse.

    Atoms that remain tied after refinement are genuinely automorphic (the three
    hydrogens of a methyl, the ortho carbons of a monosubstituted ring). Any choice
    among them yields an isometric tree, so the residual index tie-break is harmless.
    """
    n = len(z)
    colour = np.unique(z, return_inverse=True)[1].astype(np.int64)
    for _ in range(max_iter or n):
        sig = [(int(colour[i]), tuple(sorted(int(colour[j]) for j in neighbours[i])))
               for i in range(n)]
        lut = {s: k for k, s in enumerate(sorted(set(sig)))}
        new = np.fromiter((lut[s] for s in sig), dtype=np.int64, count=n)
        if len(lut) == len(set(colour.tolist())):  # partition stopped refining
            return new
        colour = new
    return colour


def _neighbours(mol: Chem.Mol) -> list:
    return [[nb.GetIdx() for nb in atom.GetNeighbors()] for atom in mol.GetAtoms()]


def choose_root(g: nx.Graph, z: np.ndarray, rank: np.ndarray) -> int:
    """Graph centre, tie-broken toward branching heavy atoms, then canonically.

    Rounds track BFS depth, so rooting at the centre minimises the number of
    sequential kernel launches for the whole batch.
    """
    centre = nx.center(g)
    return max(centre, key=lambda i: (z[i] > 1, g.degree(i), -rank[i], -i))


def _canonical_bfs(neighbours: list, root: int, rank: np.ndarray):
    """BFS visiting neighbours in canonical-rank order.

    ``nx.bfs_tree`` follows adjacency insertion order, which is input-numbering
    dependent; that alone was enough to make the tree non-canonical.
    """
    order, parent = [root], {}
    seen = np.zeros(len(rank), dtype=bool)
    seen[root] = True
    head = 0
    while head < len(order):
        node = order[head]
        head += 1
        for nb in sorted(neighbours[node], key=lambda i: (rank[i], i)):
            if not seen[nb]:
                seen[nb] = True
                parent[nb] = node
                order.append(nb)
    return order, parent


def _angle_ok(pos, i, j, k) -> bool:
    u, v = pos[i] - pos[j], pos[k] - pos[j]
    ang = np.arctan2(np.linalg.norm(np.cross(u, v)), float(u @ v))
    return _MIN_ANGLE < ang < _MAX_ANGLE


def _linear_mask(pos, triples, tol_deg) -> np.ndarray:
    """Rows whose angle exceeds ``tol_deg``; all-False when no geometry is available."""
    if pos is None or len(triples) == 0:
        return np.zeros(len(triples), dtype=bool)
    u = pos[triples[:, 0]] - pos[triples[:, 1]]
    v = pos[triples[:, 2]] - pos[triples[:, 1]]
    ang = np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u * v).sum(-1))
    return ang > np.deg2rad(tol_deg)


def _pick(cands, placed, deg, rank, key_geom=None) -> int:
    """Best placed candidate, preferring branching atoms; -1 if none qualifies.

    When ``key_geom`` is supplied this is *strict*: it returns -1 rather than an
    ill-conditioned reference, so the caller's cascade can fall through to a
    different frame (typically an improper about the parent) before giving up.
    """
    ok = [c for c in cands if placed[c]]
    if key_geom is not None:
        ok = [c for c in ok if key_geom(c)]
    if not ok:
        return -1
    return max(ok, key=lambda c: (deg[c], -rank[c], -c))


def spec_from_graph(z: np.ndarray, edge_index, pos: Optional[np.ndarray] = None,
                    use_geometry: bool = True, linear_tol_deg: float = 175.0) -> TreeSpec:
    """Build a ``TreeSpec`` from a bare molecular graph. Hydrogens must be explicit.

    This is the entry point for graph-native data objects (PyG ``Data`` carries ``z``
    and ``edge_index``, not an RDKit mol). ``edge_index`` may be ``[2, E]`` or ``[E, 2]``
    and may list each bond once or in both directions.

    The tree is **canonical**: every ordering decision breaks ties on a Weisfeiler-Lehman
    rank, so permuting the input atom numbering yields an isometric tree and an identical
    DoF multiset. Without that, the same molecule parsed from a differently-ordered SMILES
    gets a different parameterisation of the same 3D structure.

    ``use_geometry`` additionally steers reference-atom choice away from near-linear
    frames, which is what keeps alkynes and nitriles from producing undefined dihedrals.
    It makes the tree depend on ``pos``, so **anything persisted must be built with
    ``use_geometry=False``** or the spec will not be reproducible at load time.
    """
    z = np.asarray(z, dtype=np.int64).reshape(-1)
    n = len(z)
    if n < 4:
        raise ValueError(f"need at least 4 atoms for a full internal parameterisation, got {n}")

    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2:
        raise ValueError(f"edge_index must be 2-D, got shape {ei.shape}")
    if ei.shape[0] != 2 or (ei.shape[1] == 2 and ei.shape[0] != 2):
        ei = ei.T if ei.shape[1] == 2 else ei
    edges = sorted({(int(min(i, j)), int(max(i, j))) for i, j in zip(ei[0], ei[1]) if i != j})

    g = nx.Graph(edges)
    g.add_nodes_from(range(n))
    if not nx.is_connected(g):
        raise ValueError("molecule graph is disconnected; split components first")

    if pos is not None:
        pos = np.asarray(pos, dtype=np.float64)
    if not use_geometry:
        pos = None

    nbrs = [sorted(g.neighbors(i)) for i in range(n)]
    rank = canonical_rank(z, nbrs)
    z_orig = z

    root = choose_root(g, z_orig, rank)
    order, parent = _canonical_bfs(nbrs, root, rank)

    slot = np.full(n, -1, dtype=np.int64)
    for k, atom in enumerate(order):
        slot[atom] = k

    deg = np.array([g.degree(i) for i in range(n)], dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    ref_a = np.full(n, -1, dtype=np.int64)
    ref_b = np.full(n, -1, dtype=np.int64)
    ref_c = np.full(n, -1, dtype=np.int64)

    for k, atom in enumerate(order):
        if k == 0:
            placed[atom] = True
            continue

        c = parent[atom]
        ref_c[k] = c

        if k >= 2:
            # theta = angle(b, c, atom); prefer the ancestor chain, then any bonded
            # neighbour giving a non-degenerate angle, then whatever is available.
            geom = (lambda cand: _angle_ok(pos, cand, c, atom)) if pos is not None else None
            cands = [x for x in nbrs[c] if x != atom]
            b = parent.get(c, -1)
            if b < 0 or not placed[b] or (geom is not None and not geom(b)):
                b = _pick(cands, placed, deg, rank, geom)
            if b < 0:
                b = _pick(cands, placed, deg, rank)  # terminal linear centre: theta ~ pi, flagged below
            ref_b[k] = b

        if k >= 3:
            # phi = dihedral(a, b, c, atom); the frame is undefined when angle(a, b, c)
            # is linear, so an improper about c beats a collinear proper torsion.
            b = int(ref_b[k])
            geom = (lambda cand: _angle_ok(pos, cand, b, c)) if pos is not None else None
            chain = [x for x in nbrs[b] if x not in (c, atom)]
            improper = [x for x in nbrs[c] if x not in (b, atom)]
            a = parent.get(b, -1)
            if a < 0 or a == c or not placed[a] or (geom is not None and not geom(a)):
                a = _pick(chain, placed, deg, rank, geom)
            if a < 0:
                a = _pick(improper, placed, deg, rank, geom)
            if a < 0:  # nothing well-conditioned exists; take the best proper frame
                a = _pick(chain, placed, deg, rank)
            if a < 0:
                a = _pick(improper, placed, deg, rank)
            if a < 0:  # pathological; any placed atom defines a frame
                a = _pick([x for x in order[:k] if x not in (b, c, atom)], placed, deg, rank)
            ref_a[k] = a

        placed[atom] = True

    # remap everything from input indices into placement-order indices
    def to_slot(arr):
        out = np.where(arr >= 0, slot[np.clip(arr, 0, None)], -1)
        return out.astype(np.int64)

    ref_a, ref_b, ref_c = to_slot(ref_a), to_slot(ref_b), to_slot(ref_c)

    round_id = np.zeros(n, dtype=np.int64)
    for k in range(1, n):
        refs = [r for r in (ref_a[k], ref_b[k], ref_c[k]) if r >= 0]
        round_id[k] = 1 + max(round_id[r] for r in refs)

    slots = np.arange(n, dtype=np.int64)
    bond_index = np.stack([ref_c[1:], slots[1:]], axis=1)
    angle_index = np.stack([ref_b[2:], ref_c[2:], slots[2:]], axis=1)
    torsion_index = np.stack([ref_a[3:], ref_b[3:], ref_c[3:], slots[3:]], axis=1)

    bond_set = {(slot[i], slot[j]) for i, j in edges}
    bond_set |= {(j, i) for i, j in bond_set}
    torsion_is_proper = np.array([(int(a), int(b)) in bond_set for a, b in torsion_index[:, :2]],
                                 dtype=bool)

    # Linear centres (alkynes, nitriles, azides) are a genuine singularity of any
    # atom-tree parameterisation: theta -> pi makes log sin(theta) diverge and leaves
    # the dependent dihedral frame undefined. Reference choice cannot avoid it -- the
    # angle is linear because the molecule is. Flag it so callers can freeze those
    # DoF (physically they are stiff and constant anyway) or filter the molecule.
    pos_ordered = pos[np.asarray(order)] if pos is not None else None
    angle_is_linear = _linear_mask(pos_ordered, angle_index, linear_tol_deg)
    torsion_frame_is_linear = _linear_mask(pos_ordered, torsion_index[:, :3], linear_tol_deg)

    tree_edges = {tuple(sorted((int(c), int(v)))) for c, v in bond_index}
    graph_edges = [tuple(sorted((int(slot[i]), int(slot[j])))) for i, j in edges]
    broken = [e for e in graph_edges if e not in tree_edges]

    return TreeSpec(
        n_atoms=n,
        z=z_orig[order],
        perm=np.asarray(order, dtype=np.int64),
        ref_a=ref_a, ref_b=ref_b, ref_c=ref_c,
        round_id=round_id,
        bond_index=bond_index,
        angle_index=angle_index,
        torsion_index=torsion_index,
        torsion_is_proper=torsion_is_proper,
        angle_is_linear=angle_is_linear,
        torsion_frame_is_linear=torsion_frame_is_linear,
        broken_bond_index=np.asarray(broken, dtype=np.int64).reshape(-1, 2),
        graph_bond_index=np.asarray(graph_edges, dtype=np.int64).reshape(-1, 2),
    )


def spec_from_mol(mol: Chem.Mol, conf_id: int = -1, use_geometry: bool = True,
                  linear_tol_deg: float = 175.0) -> TreeSpec:
    """``spec_from_graph`` for an RDKit mol. Hydrogens must already be explicit."""
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    edges = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
                     dtype=np.int64).reshape(-1, 2).T
    pos = (np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=np.float64)
           if mol.GetNumConformers() else None)
    return spec_from_graph(z, edges, pos, use_geometry=use_geometry,
                           linear_tol_deg=linear_tol_deg)


def spec_from_smiles(smiles: str, embed: bool = True, seed: int = 0xf00d):
    """Convenience wrapper: SMILES -> (mol with explicit Hs and a conformer, TreeSpec).

    The conformer is only used to condition reference-atom choice and to supply
    reference internal coordinates; it is not part of the parameterisation.
    """
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if embed:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"embedding failed: {smiles}")
    return mol, spec_from_mol(mol)


def reference_dihedral(mol: Chem.Mol, spec: TreeSpec, idx: int, conf_id: int = -1) -> float:
    """RDKit's dihedral for torsion row ``idx``; used to cross-check our convention."""
    a, b, c, n = (int(spec.perm[i]) for i in spec.torsion_index[idx])
    return rdMolTransforms.GetDihedralRad(mol.GetConformer(conf_id), a, b, c, n)

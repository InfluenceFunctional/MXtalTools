"""
Batched, differentiable conformer construction from internal coordinates.

The batch is a flat concatenation of molecules (PyG style). Construction proceeds
in *rounds*: every atom whose three reference atoms were placed in earlier rounds
is placed in one vectorised call, across all molecules simultaneously. The number
of sequential kernel launches is therefore the maximum tree depth in the batch
(~10-25 for drug-like molecules), not the atom count -- so 1k molecules cost the
same wall-clock as 1.

Parameter layout
----------------
Three tensors, aligned with the corresponding index arrays on the batch:

    r     [n_bonds]     <-> batch.bond_index    [n_bonds, 2]
    theta [n_angles]    <-> batch.angle_index   [n_angles, 3]
    phi   [n_torsions]  <-> batch.torsion_index [n_torsions, 4]

which is exactly the indexing a 2-/3-/4-body head set on the 2D graph wants to
emit into. Totals satisfy ``n_bonds + n_angles + n_torsions == sum(3N - 6)``.

Semi-rigid variants need no support here: freeze a subset by supplying fixed
values for it and only letting the model emit the rest.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import (bond_angle, bond_length, dihedral, place_nerf,
                       place_seed_second, place_seed_third)
from .topology import TreeSpec


@dataclass
class BatchedTree:
    """Collated index tensors for a batch of molecules. Holds no coordinates."""

    n_mols: int
    n_atoms: int
    z: torch.Tensor                 # [A]
    batch: torch.Tensor             # [A] molecule id per atom
    ref_a: torch.Tensor             # [A] -1 where unused
    ref_b: torch.Tensor
    ref_c: torch.Tensor
    rounds: list                    # list[LongTensor]; rounds[0..2] are the frame seeds
    bond_index: torch.Tensor        # [n_bonds, 2]
    angle_index: torch.Tensor       # [n_angles, 3]
    torsion_index: torch.Tensor     # [n_torsions, 4]
    torsion_is_proper: torch.Tensor # [n_torsions] bool
    angle_is_linear: torch.Tensor   # [n_angles] theta ~ pi; freeze these, see topology.py
    torsion_frame_is_linear: torch.Tensor  # [n_torsions] phi ill-conditioned
    bond_batch: torch.Tensor
    angle_batch: torch.Tensor
    torsion_batch: torch.Tensor
    broken_bond_index: torch.Tensor # [n_broken, 2]
    broken_bond_batch: torch.Tensor
    graph_bond_index: torch.Tensor  # [n_graph_bonds, 2] all bonds, tree and non-tree
    graph_bond_batch: torch.Tensor
    ptr: torch.Tensor               # [n_mols + 1] atom offsets
    perm: torch.Tensor              # [A] original RDKit index within the molecule

    @property
    def device(self):
        return self.z.device

    def to(self, device) -> "BatchedTree":
        move = lambda t: t.to(device)
        return BatchedTree(
            n_mols=self.n_mols, n_atoms=self.n_atoms,
            z=move(self.z), batch=move(self.batch),
            ref_a=move(self.ref_a), ref_b=move(self.ref_b), ref_c=move(self.ref_c),
            rounds=[move(t) for t in self.rounds],
            bond_index=move(self.bond_index), angle_index=move(self.angle_index),
            torsion_index=move(self.torsion_index),
            torsion_is_proper=move(self.torsion_is_proper),
            angle_is_linear=move(self.angle_is_linear),
            torsion_frame_is_linear=move(self.torsion_frame_is_linear),
            bond_batch=move(self.bond_batch), angle_batch=move(self.angle_batch),
            torsion_batch=move(self.torsion_batch),
            broken_bond_index=move(self.broken_bond_index),
            broken_bond_batch=move(self.broken_bond_batch),
            graph_bond_index=move(self.graph_bond_index),
            graph_bond_batch=move(self.graph_bond_batch),
            ptr=move(self.ptr), perm=move(self.perm),
        )


def collate(specs: list, device=None) -> BatchedTree:
    """Concatenate per-molecule ``TreeSpec``s into batched index tensors.

    Pure integer bookkeeping -- cache the result and reuse it across every sample
    drawn for the same molecule set.
    """
    sizes = np.array([s.n_atoms for s in specs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    n_atoms = int(offsets[-1])

    def shift(attr, cols):
        """Offset an index array into the flat batch, preserving -1 sentinels."""
        parts = []
        for s, off in zip(specs, offsets[:-1]):
            arr = np.asarray(getattr(s, attr), dtype=np.int64).reshape(-1, cols)
            parts.append(np.where(arr >= 0, arr + off, -1))
        return torch.from_numpy(np.concatenate(parts, axis=0) if parts
                                else np.zeros((0, cols), dtype=np.int64))

    def seg(attr, cols):
        counts = [np.asarray(getattr(s, attr), dtype=np.int64).reshape(-1, cols).shape[0]
                  for s in specs]
        return torch.from_numpy(np.repeat(np.arange(len(specs), dtype=np.int64), counts))

    refs = {}
    for name in ("ref_a", "ref_b", "ref_c"):
        parts = [np.where(getattr(s, name) >= 0, getattr(s, name) + off, -1)
                 for s, off in zip(specs, offsets[:-1])]
        refs[name] = torch.from_numpy(np.concatenate(parts))

    n_rounds = max(s.n_rounds for s in specs)
    rounds = []
    for t in range(n_rounds):
        idx = np.concatenate([np.flatnonzero(s.round_id == t) + off
                              for s, off in zip(specs, offsets[:-1])])
        rounds.append(torch.from_numpy(np.sort(idx)))

    batched = BatchedTree(
        n_mols=len(specs),
        n_atoms=n_atoms,
        z=torch.from_numpy(np.concatenate([s.z for s in specs])),
        batch=torch.from_numpy(np.repeat(np.arange(len(specs), dtype=np.int64), sizes)),
        ref_a=refs["ref_a"], ref_b=refs["ref_b"], ref_c=refs["ref_c"],
        rounds=rounds,
        bond_index=shift("bond_index", 2),
        angle_index=shift("angle_index", 3),
        torsion_index=shift("torsion_index", 4),
        torsion_is_proper=torch.from_numpy(
            np.concatenate([s.torsion_is_proper for s in specs])),
        angle_is_linear=torch.from_numpy(
            np.concatenate([s.angle_is_linear for s in specs])),
        torsion_frame_is_linear=torch.from_numpy(
            np.concatenate([s.torsion_frame_is_linear for s in specs])),
        bond_batch=seg("bond_index", 2),
        angle_batch=seg("angle_index", 3),
        torsion_batch=seg("torsion_index", 4),
        broken_bond_index=shift("broken_bond_index", 2),
        broken_bond_batch=seg("broken_bond_index", 2),
        graph_bond_index=shift("graph_bond_index", 2),
        graph_bond_batch=seg("graph_bond_index", 2),
        ptr=torch.from_numpy(offsets),
        perm=torch.from_numpy(np.concatenate([s.perm for s in specs])),
    )
    return batched.to(device) if device is not None else batched


def _scatter_to_atoms(values: torch.Tensor, target_atoms: torch.Tensor,
                      n_atoms: int) -> torch.Tensor:
    """Place a compact DoF vector onto its owning atoms; differentiable."""
    full = torch.zeros(n_atoms, dtype=values.dtype, device=values.device)
    return full.index_copy(0, target_atoms, values)


def build(tree: BatchedTree, r: torch.Tensor, theta: torch.Tensor,
          phi: torch.Tensor) -> torch.Tensor:
    """Internal coordinates -> Cartesian positions ``[A, 3]``.

    Output sits in the canonical frame implied by the seed convention (root at the
    origin, second atom on +x, third in the xy half-plane), so it is already
    SE(3)-reduced -- no alignment step is needed anywhere downstream.
    """
    n = tree.n_atoms
    r_full = _scatter_to_atoms(r, tree.bond_index[:, 1], n)
    theta_full = _scatter_to_atoms(theta, tree.angle_index[:, 2], n)
    phi_full = _scatter_to_atoms(phi, tree.torsion_index[:, 3], n)

    pos = torch.zeros(n, 3, dtype=r.dtype, device=r.device)

    for t, slots in enumerate(tree.rounds):
        if t == 0 or slots.numel() == 0:  # roots stay at the origin
            continue
        pc = pos[tree.ref_c[slots]]
        if t == 1:
            new = place_seed_second(pc, r_full[slots])
        elif t == 2:
            new = place_seed_third(pos[tree.ref_b[slots]], pc,
                                   r_full[slots], theta_full[slots])
        else:
            new = place_nerf(pos[tree.ref_a[slots]], pos[tree.ref_b[slots]], pc,
                             r_full[slots], theta_full[slots], phi_full[slots])
        pos = pos.index_copy(0, slots, new)

    return pos


def measure(tree: BatchedTree, pos: torch.Tensor):
    """Cartesian positions -> ``(r, theta, phi)``. Exact inverse of ``build``.

    Fully parallel -- no rounds, since every reference atom is already positioned.
    """
    b = tree.bond_index
    a = tree.angle_index
    d = tree.torsion_index
    return (bond_length(pos[b[:, 0]], pos[b[:, 1]]),
            bond_angle(pos[a[:, 0]], pos[a[:, 1]], pos[a[:, 2]]),
            dihedral(pos[d[:, 0]], pos[d[:, 1]], pos[d[:, 2]], pos[d[:, 3]]))


def log_jacobian(tree: BatchedTree, r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """``log|d(cartesian)/d(internal)|`` per molecule, ``[n_mols]``.

    The bond/angle/torsion volume element is ``prod_i r_i^2 sin(theta_i)``, so a
    density defined over internal coordinates must add this term to correspond to
    a Cartesian Boltzmann distribution. Independent of phi, hence a torsion-only
    parameterisation has unit Jacobian -- but note that with r and theta frozen
    this becomes a *condition-dependent constant*, which shifts log Z(c) and must
    be added back if partition functions are compared across molecules.
    """
    out = torch.zeros(tree.n_mols, dtype=r.dtype, device=r.device)
    out = out.index_add(0, tree.bond_batch, 2.0 * torch.log(r))
    out = out.index_add(0, tree.angle_batch, torch.log(torch.sin(theta)))
    return out


def closure_length(tree: BatchedTree, pos: torch.Tensor) -> torch.Tensor:
    """Realised lengths of the ring-closure bonds the tree could not represent.

    These are *determined* by the tree DoF rather than free parameters; compare
    against target lengths to form the closure penalty.
    """
    e = tree.broken_bond_index
    return bond_length(pos[e[:, 0]], pos[e[:, 1]])


def _pairs_one(adj_edges, size: int, min_separation: int):
    """One molecule's ``(pairs [p, 2], separations [p])``, atom indices from zero.

    ``adj_edges`` is that molecule's bond list already shifted to a zero base.
    """
    adj = np.zeros((size, size), dtype=np.int8)
    adj[adj_edges[:, 0], adj_edges[:, 1]] = 1
    adj[adj_edges[:, 1], adj_edges[:, 0]] = 1

    sep = np.full((size, size), min_separation + 1, dtype=np.int64)
    np.fill_diagonal(sep, 0)
    reach = adj.copy()
    for d in range(1, min_separation + 1):
        hit = (reach > 0) & (sep > d)
        sep[hit] = d
        if d < min_separation:
            reach = ((reach @ adj) > 0).astype(np.int8)

    iu = np.triu_indices(size, k=1)
    keep = sep[iu] >= min_separation
    return np.stack([iu[0][keep], iu[1][keep]], axis=1), sep[iu][keep]


def _replica_count(ptr, bonds, bond_batch, n_mols: int) -> int:
    """``n_mols`` if the batch is one molecule tiled, else 1.

    PROVED, NOT ASSUMED. Equal atom counts are not enough -- a batch of distinct
    isomers passes that and would then inherit the first molecule's exclusion set with
    nothing to report it. This additionally requires every molecule's bond block to be
    the first block shifted by its atom offset, which is exactly what ``collate`` emits
    for repeated copies of one ``TreeSpec``.
    """
    if n_mols < 2:
        return 1
    sizes = np.diff(ptr)
    size = int(sizes[0])
    if not np.all(sizes == size):
        return 1
    counts = np.bincount(bond_batch, minlength=n_mols)
    if not np.all(counts == counts[0]):
        return 1
    # bond_batch must be grouped and ascending for the reshape to mean what it says
    if np.any(np.diff(bond_batch) < 0):
        return 1
    blocks = bonds.reshape(n_mols, int(counts[0]), 2)
    off = (np.arange(n_mols, dtype=np.int64) * size)[:, None, None]
    return n_mols if np.array_equal(blocks, blocks[:1] + off) else 1


def nonbonded_pairs(tree: BatchedTree, min_separation: int = 4):
    """Intramolecular atom pairs separated by at least ``min_separation`` bonds.

    Returns ``(pair_index [P, 2], pair_batch [P], separation [P])`` for the
    intramolecular LJ term. The default excludes 1-2/1-3/1-4; pass ``3`` to retain
    1-4 pairs, which ``separation == 3`` then identifies for the usual scaling.

    Separations are exact up to ``min_separation``; anything beyond is reported as
    ``min_separation + 1``. The extra layer matters: capping at ``min_separation``
    instead would make ``separation == min_separation`` match every retained pair, so
    a 1-4 scaling factor would silently become a global epsilon multiplier.

    Boolean BFS layers per molecule -- O(N^2 * min_separation), run once and cached.

    A batch of IDENTICAL molecules takes a tiled fast path. The BFS reads only the bond
    graph, so replicas produce the same block shifted by the atom offset, and the loop is
    ``n_mols`` repetitions of one answer -- which dominated startup for a large prior
    draw (50k copies of a 26-atom molecule: 154 s, against 2.6 s to collate them). The
    fast path is TAKEN ONLY WHEN THE TILING IS PROVED, by comparing every molecule's bond
    block against the first: a batch that merely happens to be uniform in atom count is
    not a batch of one molecule, and guessing wrong would give every replica the first
    molecule's exclusion set silently.
    """
    device = tree.device
    pair_idx, pair_batch, seps = [], [], []
    ptr = tree.ptr.cpu().numpy()
    bonds = tree.graph_bond_index.cpu().numpy()
    bond_batch = tree.graph_bond_batch.cpu().numpy()

    replicas = _replica_count(ptr, bonds, bond_batch, tree.n_mols)
    if replicas > 1:
        size = int(ptr[1])
        p0, s0 = _pairs_one(bonds[bond_batch == 0], size, min_separation)
        off = (np.arange(replicas, dtype=np.int64) * size)[:, None, None]
        tile = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return (tile((p0[None] + off).reshape(-1, 2)),
                tile(np.repeat(np.arange(replicas, dtype=np.int64), len(p0))),
                tile(np.tile(s0, replicas)))

    for m in range(tree.n_mols):
        lo, hi = int(ptr[m]), int(ptr[m + 1])
        p, s = _pairs_one(bonds[bond_batch == m] - lo, hi - lo, min_separation)
        pair_idx.append(p + lo)
        pair_batch.append(np.full(p.shape[0], m, dtype=np.int64))
        seps.append(s)

    cat = lambda parts, shape: torch.from_numpy(
        np.concatenate(parts) if parts else np.zeros(shape, dtype=np.int64)).to(device)
    return (cat(pair_idx, (0, 2)), cat(pair_batch, (0,)), cat(seps, (0,)))

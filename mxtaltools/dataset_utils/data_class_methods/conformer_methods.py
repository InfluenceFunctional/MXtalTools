"""
Internal-coordinate (Z-matrix) parameterisation for molecular graphs.

The intramolecular counterpart to the crystal DoF machinery in ``crystal_ops.py``:
where a crystal is read out with ``latent_params()`` and written back with
``latent_to_cell_params()``, a conformer is read out with ``internal_dof()`` and
written back with ``set_internal_dof()``. Both leave ``pos`` as the single source of
truth -- the setter *writes* ``pos``, exactly as ``set_cell_parameters`` does, so every
downstream consumer (``pose_aunit``, ``mol2cluster``, ``analyze``) is untouched.

A molecule of N atoms has exactly 3N-6 internal DoF: N-1 bond lengths, N-2 bond angles,
N-3 dihedrals, laid out over a canonical spanning tree. The parameterisation is complete,
minimal, and SE(3)-invariant by construction, so no alignment step is ever needed.

Stored attributes
-----------------
Every index array is named ``*_index`` and stored ``[k, n]`` (edge_index layout), so
``MXtalBase.__inc__``/``__cat_dim__`` batch them with no extra code: a collated batch of
molecules *is* a batched tree. Per-DoF flags carry no ``index`` in their name and so are
concatenated without offsetting, which is what they want.

The tree is built lazily by default rather than in ``__init__``. It needs coordinates
(there is no stored bond topology to build from -- see ``conformers.perception``), costs
0.2-1.4 ms, and can legitimately fail on a disconnected or fully-linear molecule; making
that fire inside every constructor would turn a recoverable condition into an
un-loadable dataset. Pass ``build_tree=True`` to ``MolData`` for eager construction.
"""

from typing import Optional, Union

import numpy as np
import torch

from mxtaltools.conformers.builder import (BatchedTree, build, closure_length,
                                           log_jacobian, measure)
from mxtaltools.conformers.perception import infer_bond_index
from mxtaltools.conformers.topology import spec_from_graph
from mxtaltools.constants.atom_properties import ATOMIC_SYMBOLS

_TREE_KEYS = ('tree_bond_index', 'tree_angle_index', 'tree_torsion_index',
              'tree_broken_bond_index', 'mol_bond_index', 'tree_perm_index',
              'tree_round', 'tree_torsion_is_proper', 'tree_angle_is_linear',
              'tree_torsion_frame_is_linear')


# noinspection PyAttributeOutsideInit
class MolConformerMethods:

    # ------------------------------------------------------------ instantiation

    @property
    def has_conformer_tree(self) -> bool:
        return 'tree_bond_index' in self._store

    def build_conformer_tree(self,
                             force: Optional[bool] = False,
                             bond_tolerance: float = 1.2,
                             use_geometry: bool = False):
        """Attach the canonical internal-coordinate tree. Single molecules only.

        ``use_geometry=False`` (the default) makes the tree a pure function of the
        molecular graph, so it is reproducible at load time from ``z`` and ``pos``
        alone. Setting it True lets reference-atom choice dodge near-linear frames,
        at the cost of the tree depending on the specific conformer -- fine for a
        one-off analysis, wrong for anything persisted.
        """
        if self.has_conformer_tree and not force:
            return
        if self.is_batch:
            raise RuntimeError(
                "build_conformer_tree works on single molecules; build before "
                "collating and the batch inherits the tree automatically")
        if self.pos is None:
            raise RuntimeError("need `pos` to perceive bonds and build the tree")

        z = self.z.detach().cpu().numpy()
        pos = self.pos.detach().cpu().numpy()
        bonds = (self.mol_bond_index.detach().cpu().numpy()
                 if 'mol_bond_index' in self._store
                 else infer_bond_index(z, pos, tolerance=bond_tolerance))
        spec = spec_from_graph(z, bonds, pos, use_geometry=use_geometry)

        dev = self.z.device
        as_t = lambda a, dtype=torch.long: torch.as_tensor(np.ascontiguousarray(a),
                                                           dtype=dtype, device=dev)
        # spec indices live in placement-slot space; map them into this object's own
        # atom numbering so every array indexes `pos` directly and `pos` is never
        # permuted. perm[k] is the original atom placed at slot k.
        perm = spec.perm
        to_atom = lambda a: np.where(a >= 0, perm[np.clip(a, 0, None)], -1)
        round_by_atom = np.empty_like(spec.round_id)
        round_by_atom[perm] = spec.round_id

        # transposed to [k, n]: matches edge_index layout, so __cat_dim__ == -1 works
        self.tree_bond_index = as_t(to_atom(spec.bond_index).T)
        self.tree_angle_index = as_t(to_atom(spec.angle_index).T)
        self.tree_torsion_index = as_t(to_atom(spec.torsion_index).T)
        self.tree_broken_bond_index = as_t(
            to_atom(spec.broken_bond_index).T.reshape(2, -1))
        self.mol_bond_index = as_t(to_atom(spec.graph_bond_index).T)
        self.tree_perm_index = as_t(perm)
        self.tree_round = as_t(round_by_atom)
        self.tree_torsion_is_proper = as_t(spec.torsion_is_proper, torch.bool)
        self.tree_angle_is_linear = as_t(spec.angle_is_linear, torch.bool)
        self.tree_torsion_frame_is_linear = as_t(spec.torsion_frame_is_linear, torch.bool)

        # ring-closure lengths as built. These bonds are absent from the tree, so their
        # lengths are determined by the other DoF rather than settable; recording the
        # reference makes closure drift measurable without needing a force field.
        closure = self.tree_broken_bond_index
        self.tree_closure_r0 = torch.linalg.norm(
            self.pos[closure[0]] - self.pos[closure[1]], dim=-1).detach()

    def _require_tree(self):
        if not self.has_conformer_tree:
            if self.is_batch:
                raise RuntimeError(
                    "batch carries no conformer tree; call build_conformer_tree() on "
                    "the molecules before collating")
            self.build_conformer_tree()

    def as_conformer_tree(self) -> BatchedTree:
        """Adapt the stored attributes into a ``BatchedTree`` for build/measure.

        ``ref_a/b/c`` and the round grouping are reconstructed rather than stored: the
        reference atoms are already carried by the bond/angle/torsion index arrays, and
        storing them separately would mean per-atom index tensors with -1 sentinels,
        which PyG collation would silently offset into valid-looking atom indices.
        """
        self._require_tree()
        n_atoms = int(self.z.shape[0])
        dev = self.z.device
        batch = (self.batch if self.is_batch
                 else torch.zeros(n_atoms, dtype=torch.long, device=dev))
        n_mols = int(self.num_graphs) if self.is_batch else 1
        ptr = (self.ptr if self.is_batch
               else torch.tensor([0, n_atoms], dtype=torch.long, device=dev))

        bond, angle, tors = (self.tree_bond_index, self.tree_angle_index,
                             self.tree_torsion_index)
        ref_a = torch.full((n_atoms,), -1, dtype=torch.long, device=dev)
        ref_b, ref_c = ref_a.clone(), ref_a.clone()
        ref_c[bond[1]] = bond[0]
        ref_b[angle[2]], ref_c[angle[2]] = angle[0], angle[1]
        ref_a[tors[3]], ref_b[tors[3]], ref_c[tors[3]] = tors[0], tors[1], tors[2]

        n_rounds = int(self.tree_round.max()) + 1
        rounds = [torch.nonzero(self.tree_round == t, as_tuple=True)[0]
                  for t in range(n_rounds)]

        return BatchedTree(
            n_mols=n_mols, n_atoms=n_atoms, z=self.z.long(), batch=batch,
            ref_a=ref_a, ref_b=ref_b, ref_c=ref_c, rounds=rounds,
            bond_index=bond.T, angle_index=angle.T, torsion_index=tors.T,
            torsion_is_proper=self.tree_torsion_is_proper,
            angle_is_linear=self.tree_angle_is_linear,
            torsion_frame_is_linear=self.tree_torsion_frame_is_linear,
            bond_batch=batch[bond[1]], angle_batch=batch[angle[2]],
            torsion_batch=batch[tors[3]],
            broken_bond_index=self.tree_broken_bond_index.T,
            broken_bond_batch=batch[self.tree_broken_bond_index[0]],
            graph_bond_index=self.mol_bond_index.T,
            graph_bond_batch=batch[self.mol_bond_index[0]],
            ptr=ptr, perm=self.tree_perm_index,
        )

    # ------------------------------------------------------------------ analysis

    def internal_dof(self, pos: Optional[torch.Tensor] = None):
        """Read ``(r, theta, phi)`` off the current conformer.

        The intramolecular analogue of ``latent_params()``. Lengths in A, angles in rad.
        Aligned with ``tree_bond_index`` / ``tree_angle_index`` / ``tree_torsion_index``,
        which is the indexing a 2-/3-/4-body head set emits into.
        """
        tree = self.as_conformer_tree()
        return measure(tree, self.pos if pos is None else pos)

    def internal_dof_vector(self, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The DoF as one flat ``[3N-6]`` vector, ordered ``[r, theta, phi]``."""
        return torch.cat(self.internal_dof(pos))

    def internal_log_jacobian(self) -> torch.Tensor:
        """``log|d(cartesian)/d(internal)|`` per molecule.

        A density over internal coordinates must carry this term to correspond to a
        Cartesian Boltzmann distribution. It is also what makes log Z independent of
        which spanning tree was chosen -- drop it and the bookkeeping leaks into Z(c).
        """
        tree = self.as_conformer_tree()
        r, theta, _ = measure(tree, self.pos)
        return log_jacobian(tree, r, theta)

    def ring_closure_error(self) -> torch.Tensor:
        """Max drift of any ring-closure bond from its as-built length, per molecule.

        Returns ``[n_mols]``, zero for acyclic molecules. Ring bonds are absent from the
        spanning tree, so their lengths are *determined* by the other DoF rather than set
        directly -- move the ring torsions and the ring opens. This is the diagnostic for
        how far that has gone.
        """
        tree = self.as_conformer_tree()
        out = torch.zeros(tree.n_mols, dtype=self.pos.dtype, device=self.pos.device)
        if tree.broken_bond_index.numel() == 0:
            return out
        err = (closure_length(tree, self.pos) - self.tree_closure_r0.to(self.pos.dtype)).abs()
        return out.scatter_reduce(0, tree.broken_bond_batch, err, reduce="amax",
                                  include_self=True)

    # ---------------------------------------------------------------- assignment

    def set_internal_dof(self,
                         dof: Union[torch.Tensor, tuple, list],
                         recenter: bool = False):
        """Write a conformer from internal coordinates. Mirrors ``set_cell_parameters``.

        Accepts either a flat ``[3N-6]`` vector (batched: the concatenation over
        molecules) or an explicit ``(r, theta, phi)`` triple. **Writes ``pos``**, which
        stays the single source of truth for everything downstream.

        The rebuilt geometry lands in the tree's canonical frame (root at the origin,
        second atom on +x, third in the xy-plane), so it is SE(3)-reduced but *not* in
        the principal-inertial frame the crystal code standardises to. For a crystal,
        follow with ``pose_aunit()``, which applies the orientation and centroid anyway.
        """
        tree = self.as_conformer_tree()
        if isinstance(dof, (tuple, list)):
            r, theta, phi = dof
        else:
            dof = torch.as_tensor(dof, device=self.z.device)
            n_r = tree.bond_index.shape[0]
            n_a = tree.angle_index.shape[0]
            expected = n_r + n_a + tree.torsion_index.shape[0]
            if dof.numel() != expected:
                raise ValueError(f"expected {expected} internal DoF (3N-6), "
                                 f"got {dof.numel()}")
            r, theta, phi = dof[:n_r], dof[n_r:n_r + n_a], dof[n_r + n_a:]

        dtype = self.pos.dtype if self.pos is not None else torch.float32
        new_pos = build(tree, r.to(dtype), theta.to(dtype), phi.to(dtype))
        if recenter:
            centre = torch.zeros(tree.n_mols, 3, dtype=dtype, device=new_pos.device)
            counts = torch.zeros(tree.n_mols, dtype=dtype, device=new_pos.device)
            centre = centre.index_add(0, tree.batch, new_pos)
            counts = counts.index_add(0, tree.batch, torch.ones_like(counts[tree.batch]))
            new_pos = new_pos - (centre / counts[:, None])[tree.batch]
        self.pos = new_pos
        return self.pos

    # ------------------------------------------------------------------- readout

    def print_internal_dof(self, mol_index: int = 0, max_rows: Optional[int] = None):
        """Human-readable dump of the parameterisation for one molecule."""
        print(self.format_internal_dof(mol_index, max_rows))

    def format_internal_dof(self, mol_index: int = 0,
                            max_rows: Optional[int] = None) -> str:
        tree = self.as_conformer_tree()
        r, theta, phi = self.internal_dof()
        z = self.z.detach().cpu().numpy()
        sym = lambda i: f"{ATOMIC_SYMBOLS.get(int(z[i]), '?')}{int(i)}"

        lo = int(tree.ptr[mol_index])
        keep = lambda idx, col: torch.nonzero(
            tree.batch[idx[:, col]] == mol_index, as_tuple=True)[0].cpu().numpy()
        b_rows = keep(tree.bond_index, 1)
        a_rows = keep(tree.angle_index, 2)
        t_rows = keep(tree.torsion_index, 3)

        n_atoms = int(tree.ptr[mol_index + 1]) - lo
        out = [f"internal DoF for molecule {mol_index}: {n_atoms} atoms, "
               f"{len(b_rows) + len(a_rows) + len(t_rows)} DoF (3N-6 = {3 * n_atoms - 6})",
               f"  {len(b_rows)} bonds, {len(a_rows)} angles, {len(t_rows)} dihedrals; "
               f"{tree.broken_bond_index.shape[0]} ring-closure bond(s) not in the tree",
               f"  {'atom':>5}  {'kind':<8} {'reference atoms':<24} {'value':>10}"]

        rows = []
        bi, ai, ti = (x.cpu().numpy() for x in
                      (tree.bond_index, tree.angle_index, tree.torsion_index))
        for k in b_rows:
            rows.append((int(bi[k, 1]), "bond", f"{sym(bi[k,0])}-{sym(bi[k,1])}",
                         f"{r[k].item():9.4f} A", ""))
        for k in a_rows:
            flag = " linear" if bool(tree.angle_is_linear[k]) else ""
            rows.append((int(ai[k, 2]), "angle",
                         f"{sym(ai[k,0])}-{sym(ai[k,1])}-{sym(ai[k,2])}",
                         f"{np.rad2deg(theta[k].item()):8.2f} d", flag))
        for k in t_rows:
            flag = "" if bool(tree.torsion_is_proper[k]) else " improper"
            if bool(tree.torsion_frame_is_linear[k]):
                flag += " ill-conditioned"
            rows.append((int(ti[k, 3]), "dihedral",
                         f"{sym(ti[k,0])}-{sym(ti[k,1])}-{sym(ti[k,2])}-{sym(ti[k,3])}",
                         f"{np.rad2deg(phi[k].item()):8.2f} d", flag))

        rows.sort(key=lambda t: (t[0], t[1]))  # by owning atom, then kind
        shown = rows if max_rows is None else rows[:max_rows]
        for slot, kind, refs, value, flag in shown:
            out.append(f"  {slot - lo:>5}  {kind:<8} {refs:<24} {value:>10}{flag}")
        if max_rows is not None and len(rows) > max_rows:
            out.append(f"  ... {len(rows) - max_rows} more")
        return "\n".join(out)

    # -------------------------------------------------------------------- energy

    def intramolecular_energy(self,
                              force_field=None,
                              components: bool = False,
                              **ff_kwargs):
        """Intramolecular energy: harmonic bonds and angles plus soft-core LJ.

        Without a ``force_field`` one is fitted to the *current* conformer, so the
        bonded terms sit at their minimum and only the nonbonded term is non-zero --
        useful as a clash score, not as a conformer ranking. To rank conformers, fit
        once against a reference and reuse:

            ff = reference_mol.intramolecular_force_field()
            e  = candidate_mol.intramolecular_energy(force_field=ff)

        The repulsive core is exponential below sigma rather than r^-12, matched in
        value and slope, following ``exponential_edgewise_lj_energy`` in vdw_analysis --
        finite at contact, so a bad conformer scores badly instead of overflowing.
        """
        from mxtaltools.conformers.energy import intramolecular_energy

        tree = self.as_conformer_tree()
        ff = force_field if force_field is not None else \
            self.intramolecular_force_field(**ff_kwargs)
        return intramolecular_energy(tree, self.pos, ff, components=components)

    def intramolecular_force_field(self, **kwargs):
        """Fit the simple force field to this conformer's bonded geometry."""
        from mxtaltools.conformers.energy import ff_from_reference

        return ff_from_reference(self.as_conformer_tree(), self.pos, **kwargs)

    def worst_intramolecular_clash(self, force_field=None, **ff_kwargs):
        """Deepest nonbonded overlap ``max(sigma - r)`` per molecule, in A."""
        from mxtaltools.conformers.energy import worst_clash

        ff = force_field if force_field is not None else \
            self.intramolecular_force_field(**ff_kwargs)
        return worst_clash(self.as_conformer_tree(), self.pos, ff)

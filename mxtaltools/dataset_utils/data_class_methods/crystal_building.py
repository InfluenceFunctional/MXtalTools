"""Crystal construction methods mixed into MolCrystalData.

The steps are: pose the asymmetric unit from its cell parameters (pose_aunit), build
the unit cell from the space-group operators (build_unit_cell), and build a finite
cluster of periodic images around the asymmetric unit (build_cluster). mol2ucell and
mol2cluster chain these steps. Crystals with Z' > 1 go through split_to_zp1_batch,
which makes one graph per asymmetric-unit molecule, and one of the ``join_zp1_*`` methods,
which recombines the results per crystal.

Conventions used throughout: ``pos`` and ``unit_cell_pos`` are Cartesian (Angstrom);
``aunit_centroid`` is fractional, [num_graphs, 3 * max_z_prime]; ``aunit_orientation``
holds rotation vectors (angle in radians), [num_graphs, 3 * max_z_prime];
``aunit_handedness`` is [num_graphs, max_z_prime]; ``T_fc`` and ``T_cf`` are the
per-graph [3, 3] fractional-to-Cartesian and Cartesian-to-fractional transforms.
"""

import copy
from typing import Optional

import torch

from mxtaltools.crystal_building.utils import get_aunit_positions, aunit2ucell, ucell2cluster, \
    _unit_layout_unchecked, UNIT_MAP_KEY, reference_molecule_reach
from mxtaltools.dataset_utils.utils import collate_data_list

#: refusal messages name at most this many rows, then count the rest
_ROWS_NAMED = 8


def _rows_named(batch, rows) -> str:
    """'row 3 (ABC), row 7' for refusal messages: each row's index in the batch and, where one is stored, its
    identifier."""
    ident = batch['identifier'] if 'identifier' in batch._store else None
    if isinstance(ident, str):
        ident = [ident]
    names = [f'row {i}' + (f' ({ident[i]})' if isinstance(ident, (list, tuple)) and i < len(ident) else '')
             for i in rows[:_ROWS_NAMED]]
    more = f' and {len(rows) - _ROWS_NAMED} more' if len(rows) > _ROWS_NAMED else ''
    return ', '.join(names) + more


# noinspection PyAttributeOutsideInit
class MolCrystalBuilding:
    """Mixin providing asymmetric-unit posing, unit-cell and cluster construction."""

    def split_to_zp1_batch(self):
        """Split a crystal batch into a batch with one graph per asymmetric-unit molecule.

        Crystal ``i`` becomes ``z_prime[i]`` consecutive graphs, one per unit, as the row's unit map says
        (``crystal_building.utils.unit_layout``: its ``unit_ind``, or ``z_prime`` equal contiguous blocks of
        ``num_atoms[i] // z_prime[i]`` atoms when it carries none). At this stage the units must be equal
        contiguous blocks in unit order either way, so a unit holds ``num_atoms[i] // z_prime[i]`` atoms and the
        split is the equal-block one bit for bit; per-atom tensors (``pos``, ``z``, ``x``) are not reordered, only
        re-batched.

        The result is a clone of ``self``; ``self`` is not modified. On the clone:

        - ``batch``, ``ptr``, ``_num_graphs`` and ``num_atoms`` are rebuilt for the new
          graphs;
        - ``unit_ind`` is zeros: each unit-graph is one unit (stage 2, 2026-10-22);
        - ``sg_ind``, ``sym_mult``, ``nonstandard_symmetry``, ``T_fc``, ``T_cf``,
          ``symmetry_operators`` (a list), ``cell_lengths``, ``cell_angles``,
          ``radius``, ``mol_volume`` and ``mass`` repeat the parent crystal's value;
        - ``z_prime`` is set to ones;
        - ``aunit_centroid`` and ``aunit_orientation`` become [new_num_graphs, 3] and
          ``aunit_handedness`` becomes [new_num_graphs, 1], each taken from that
          molecule's column block of the parent's layout; a batch without
          ``aunit_handedness`` gives a split without it (only ``std_orientation=True``
          reads it, and pose_aunit refuses that without one);
        - ``unit_cell_batch`` is rebuilt when ``unit_cell_pos`` is present and not
          None.

        All other attributes keep the parent's values, including ``max_z_prime``,
        other per-graph properties and the PyG slice/increment bookkeeping, so the
        result is an intermediate for crystal building rather than a complete batch.

        Returns
        -------
        MolCrystalData batch
            The split batch, with ``sum(z_prime)`` graphs.

        Raises
        ------
        ValueError
            If ``self`` is not a batch; if a row has ``z_prime < 1``, more units than
            pose slots (``3 * z_prime`` > the columns of ``aunit_centroid``), or units
            its unit map cannot give at this stage (``unit_layout``: without
            ``unit_ind``, a ``num_atoms`` that does not divide into ``z_prime`` equal
            units; with one, a map that is not ``z_prime`` equal contiguous blocks in
            unit order, e.g. ethanol + methane, refused since 2026-10-22); if the
            stored ``unit_ind`` does not hold one entry per atom; or if the number of
            atoms in ``pos`` differs from ``sum(num_atoms)``. Explicit raises, so they
            hold under ``python -O`` (before 2026-10-22 these were bare ``assert``
            statements, and ``-O`` split 25 atoms 12 + 12).
        """
        # if hasattr(self, "aux_ind"):
        #     assert self.aux_ind is None, "Not implemented for cluster objects"
        # NOTE this is only an intermediate for crystal building, and will not generate all the correct attributes for
        # subunit crystal graphs such as molecule and crystal properties
        if not self.is_batch:
            raise ValueError("split_to_zp1_batch: needs a collated batch, not a single data object "
                             "(collate_data_list([crystal]) first)")
        n_slots = self.aunit_centroid.shape[1] // 3
        try:  # the unit map (the one reader, crystal_building.utils.unit_layout), checked on every row below
            layout, map_bad, map_reason = _unit_layout_unchecked(self)
        except ValueError as e:  # a stored map that does not hold one entry per atom: no row to name
            raise ValueError(f"split_to_zp1_batch: {e}") from e
        slot_bad = (self.z_prime < 1) | (self.z_prime > n_slots)
        bad = slot_bad.reshape(-1) | map_bad
        if bool(bad.any()):  # one device sync on the valid path; the row list only when refusing
            bad = torch.nonzero(bad).flatten().tolist()
            reasons = []
            for i in bad[:_ROWS_NAMED]:
                zp_i = int(self.z_prime[i])
                if zp_i < 1:
                    why = f'z_prime {zp_i}'
                elif zp_i > n_slots:
                    why = f'z_prime {zp_i} with {n_slots} pose slot(s)'
                else:
                    why = map_reason(i)
                reasons.append(f'{_rows_named(self, [i])}: {why}')
            more = f'; and {len(bad) - _ROWS_NAMED} more' if len(bad) > _ROWS_NAMED else ''
            raise ValueError(f"split_to_zp1_batch: {len(bad)} row(s) cannot be split into their units (a unit is "
                             f"what the row's unit_ind says, or one of z_prime equal contiguous blocks without one; "
                             f"units must be equal contiguous blocks in unit order at this stage; one pose slot "
                             f"each): {'; '.join(reasons)}{more}")
        if len(self.pos) != int(self.num_atoms.sum()):
            raise ValueError(f"split_to_zp1_batch: pos holds {len(self.pos)} atoms but the rows' num_atoms sum to "
                             f"{int(self.num_atoms.sum())}; a cluster or another per-atom layout cannot be split")
        new_num_graphs = int(torch.sum(self.z_prime))
        out_graphs_per_in_graph = self.z_prime
        graph_ids = torch.arange(new_num_graphs, dtype=torch.long, device=self.device)
        # each unit's atoms are its row's unit-map block (equal and contiguous at this stage: unit_layout checked)
        rep_index = torch.arange(self.num_graphs, device=self.device).repeat_interleave(self.z_prime)
        atoms_per_subunit = layout.atoms_per_unit[rep_index]
        new_batch = graph_ids.repeat_interleave(atoms_per_subunit)

        zp1_batch = self.clone()
        zp1_batch.num_atoms = atoms_per_subunit
        zp1_batch.batch = new_batch
        zp1_batch.ptr = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.device),
             torch.cumsum(atoms_per_subunit, dim=0)])
        zp1_batch._num_graphs = new_num_graphs
        # copy over relevant crystal properties
        zp1_batch.sg_ind = self.sg_ind[rep_index]
        zp1_batch.sym_mult = self.sym_mult[rep_index]
        zp1_batch.nonstandard_symmetry = self.nonstandard_symmetry[rep_index]
        zp1_batch.T_fc = self.T_fc[rep_index]
        zp1_batch.T_cf = self.T_cf[rep_index]
        zp1_batch.symmetry_operators = [self.symmetry_operators[ind] for ind in rep_index]
        zp1_batch.cell_lengths = self.cell_lengths[rep_index]
        zp1_batch.cell_angles = self.cell_angles[rep_index]
        zp1_batch.num_atoms = layout.atoms_per_unit[rep_index]
        zp1_batch.radius = self.radius[rep_index]
        zp1_batch.mol_volume = self.mol_volume[rep_index]
        zp1_batch.mass = self.mass[rep_index]
        zp1_batch.z_prime = torch.ones(new_num_graphs, device=self.device, dtype=torch.long)
        zp1_batch[UNIT_MAP_KEY] = torch.zeros(len(self.pos), dtype=torch.long, device=self.device)  # one unit each

        if hasattr(zp1_batch,'unit_cell_pos'):
            if zp1_batch.unit_cell_pos is not None:
                atoms_per_sub_ucell = (layout.atoms_per_unit * self.sym_mult.reshape(-1))[rep_index]
                zp1_batch.unit_cell_batch = graph_ids.repeat_interleave(atoms_per_sub_ucell)

        # extra handling for aunit properties
        subunit_index = torch.arange(new_num_graphs, device=self.device) - torch.repeat_interleave(
            torch.cumsum(self.z_prime, 0) - self.z_prime, self.z_prime)
        # columns to index from [num_graphs, 3*Z'] format
        col_base = 3 * subunit_index.unsqueeze(1) + torch.arange(3, device=self.device)
        col_base2 = 1 * subunit_index.unsqueeze(1) + torch.arange(1, device=self.device)

        zp1_batch.aunit_centroid = self.aunit_centroid[rep_index.unsqueeze(1), col_base]
        zp1_batch.aunit_orientation = self.aunit_orientation[rep_index.unsqueeze(1), col_base]
        if self.aunit_handedness is not None:  # absent stays absent: std_orientation=False never reads it
            zp1_batch.aunit_handedness = self.aunit_handedness[rep_index.unsqueeze(1), col_base2]

        return zp1_batch

    def join_zp1_aunit_batch(self, zp1_batch):
        """Copy the atom positions of a split batch back onto this batch, in place.

        Assigns ``self.pos = zp1_batch.pos``; nothing else is changed.

        Parameters
        ----------
        zp1_batch : MolCrystalData batch
            Output of split_to_zp1_batch on this batch (for example after
            pose_aunit), which shares its per-atom order.
        """
        self.pos = zp1_batch.pos

    def join_zp1_ucell_batch(self, zp1_batch):
        """Copy posed positions and unit cells of a split batch back onto this batch, in place.

        Assigns ``self.pos``, ``self.unit_cell_pos`` and ``self.unit_cell_mol_ind``
        from ``zp1_batch``, and rebuilds ``self.unit_cell_batch`` as each crystal's
        index repeated ``num_atoms * sym_mult`` times. The unit-cell atoms of each
        crystal must be contiguous in ``zp1_batch.unit_cell_pos``, which holds when
        ``zp1_batch`` came from split_to_zp1_batch on this batch followed by
        build_unit_cell. ``unit_cell_mol_ind`` keeps the split batch's batch-wide
        molecule numbering.

        Parameters
        ----------
        zp1_batch : MolCrystalData batch
            Output of split_to_zp1_batch after pose_aunit and build_unit_cell.
        """
        atoms_per_ucell = self.num_atoms * self.sym_mult
        combined_ucell_batch = torch.arange(self.num_graphs, device=self.device).repeat_interleave(
            atoms_per_ucell
        )
        self.pos = zp1_batch.pos
        self.unit_cell_pos = zp1_batch.unit_cell_pos
        self.unit_cell_batch = combined_ucell_batch
        self.unit_cell_mol_ind = zp1_batch.unit_cell_mol_ind

    def join_zp1_cluster_batch(self, zp1_batch):
        """Combine the clusters of a split batch into one cluster per original crystal.

        Returns a clone of ``self`` (``self`` is not modified) on which:

        - ``pos``, ``x``, ``z`` and ``aux_ind`` are taken from ``zp1_batch``;
        - ``mol_ind`` is ``zp1_batch.mol_ind`` plus, for each sub-cluster, the number
          of molecules in all preceding sub-clusters of the batch, so molecules from
          different sub-clusters have distinct indices; the index does not restart at
          0 for each crystal;
        - ``batch`` maps each atom to its original crystal and ``ptr`` is rebuilt from
          the resulting atom counts;
        - ``unit_cell_pos`` is a copy of ``zp1_batch.unit_cell_pos`` and
          ``unit_cell_batch`` maps its atoms to their original crystal.

        Each sub-cluster contributes its own ``aux_ind == 0`` molecule, so every
        asymmetric-unit molecule of a crystal has ``aux_ind == 0`` in the result.
        Other attributes, including ``num_atoms`` (the full asymmetric-unit atom count)
        and ``unit_cell_mol_ind``, are those of ``self``, except the unit map
        ``unit_ind``, which is per row atom and is not carried onto the cluster (rows
        only; the cluster's ``mol_ind`` keeps its cluster meaning).

        When no crystal has ``z_prime > 1`` the split was the identity, so ``zp1_batch``
        already holds one cluster per crystal, exactly the one a ``max_z_prime == 1``
        build makes (``mol_ind`` numbered within each cluster). It is returned as built
        (a shallow copy; ``zp1_batch`` is not modified), with only ``aunit_centroid``,
        ``aunit_orientation`` and ``aunit_handedness`` (when present) replaced by copies of
        this batch's own, full-width pose slots, where the split held its slot-0 view.

        Parameters
        ----------
        zp1_batch : MolCrystalData batch
            Cluster built by build_cluster from the output of split_to_zp1_batch on
            this batch; its ``num_atoms`` is the per-molecule atom count.

        Returns
        -------
        MolCrystalData batch
            The combined cluster batch.

        Raises
        ------
        ValueError
            If ``zp1_batch`` does not hold ``sum(z_prime)`` graphs, one per unit of this
            batch (an explicit raise, so it holds under ``python -O``).
        """
        n_units, n_split = int(self.z_prime.sum()), len(zp1_batch.ptr) - 1
        if n_split != n_units:
            raise ValueError(f"join_zp1_cluster_batch: the split cluster holds {n_split} graph(s), but this batch's "
                             f"{self.num_graphs} crystal(s) hold sum(z_prime) = {n_units} units; pass the cluster "
                             f"built from this batch's own split_to_zp1_batch")
        if self.z_prime.amax() > 1:
            cluster_batch = self.clone()
            cluster_batch._store.pop(UNIT_MAP_KEY, None)    # the unit map is per row atom: a cluster carries none
            cluster_batch.pos = zp1_batch.pos
            cluster_batch.x = zp1_batch.x
            cluster_batch.z = zp1_batch.z
            cluster_batch.aux_ind = zp1_batch.aux_ind
            # reindexing molecules properly here is important for intermolecular edge construction
            atoms_per_zp1_crystal = torch.bincount(zp1_batch.batch, minlength=len(zp1_batch.ptr) - 1)
            mols_per_zp1_crystal = atoms_per_zp1_crystal // zp1_batch.num_atoms
            molwise_zp1_ptr = torch.cat([
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.cumsum(mols_per_zp1_crystal[:-1], dim=0)
            ])
            mol_ind_offset = molwise_zp1_ptr.repeat_interleave(atoms_per_zp1_crystal)
            cluster_batch.mol_ind = zp1_batch.mol_ind + mol_ind_offset

            # reindex from Z'=1 to combined structures
            batch_map = torch.arange(self.num_graphs, device=self.device).repeat_interleave(self.z_prime)
            cluster_batch.batch = batch_map[zp1_batch.batch]
            atoms_per_cluster = torch.bincount(cluster_batch.batch)

            cluster_batch.ptr = torch.cat([
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.cumsum(atoms_per_cluster, dim=0)
            ])

            cluster_batch.unit_cell_pos = zp1_batch.unit_cell_pos.clone()
            cluster_batch.unit_cell_batch = batch_map[zp1_batch.unit_cell_batch].clone()

            return cluster_batch

        else:
            # every crystal has Z'=1 (e.g. an all-Z'=1 batch at max_z_prime 2, audit #12): splitting a Z'=1 row is
            # the identity, so zp1_batch is already one cluster per crystal; only its pose slots are the split's
            # slot-0 view, and the batch's own full-width slots go back on
            cluster_batch = copy.copy(zp1_batch)
            for key in ('aunit_centroid', 'aunit_orientation', 'aunit_handedness'):
                value = getattr(self, key)
                if value is not None:  # a batch without handedness: the split carried none either
                    setattr(cluster_batch, key, value.clone())
            return cluster_batch

    # todo add method to just align to the standard orientation
    def pose_aunit(self, std_orientation: Optional[bool] = True,
                   override_handedness=None):
        """Place each molecule at the asymmetric-unit pose given by its cell parameters, in place.

        Assigns ``self.pos`` from ``crystal_building.utils.get_aunit_positions``: each
        molecule is centred on its heavy-atom centroid, optionally aligned to the
        standard axes, rotated by the rotation vector ``aunit_orientation[:, :3]`` and
        translated to the Cartesian image of the fractional ``aunit_centroid[:, :3]``
        under ``T_fc``. Only the first asymmetric-unit parameter block is read, so all
        atoms of a Z' > 1 graph are posed together as one rigid body; split such
        batches first (split_to_zp1_batch). A single object is collated into a
        one-graph batch for the computation.

        Parameters
        ----------
        std_orientation : bool, optional
            If True (default), align each molecule's heavy-atom principal axes to the
            Cartesian axes, with the handedness below, before the rotation. If False,
            the rotation is applied to the recentred current ``pos``.
        override_handedness : torch.Tensor, optional
            [num_graphs, k] handedness used instead of ``self.aunit_handedness``; only
            the first column is read (1 aligns to (x, y, z), -1 to (-x, y, z)). Only
            used when ``std_orientation`` is True.

        Raises
        ------
        ValueError
            If ``std_orientation`` is True and both ``override_handedness`` and
            ``self.aunit_handedness`` are None (an explicit raise, so it holds under
            ``python -O``; get_aunit_positions makes the same check).
        """
        if override_handedness is not None:
            handedness = override_handedness
        else:
            handedness = self.aunit_handedness
        if std_orientation and handedness is None:
            raise ValueError("pose_aunit: std_orientation=True aligns each molecule to the standard axes with its "
                             "handedness, but override_handedness and aunit_handedness are both None")
        if self.is_batch:
            self.pos = get_aunit_positions(
                self,
                std_orientation=std_orientation,
                mol_handedness=handedness,
            )
        else:
            self.pos = get_aunit_positions(
                collate_data_list([self]),
                std_orientation=std_orientation,
                mol_handedness=handedness,
            )

    def build_unit_cell(self):
        """Build the unit cell from the posed asymmetric unit, in place.

        Calls ``crystal_building.utils.aunit2ucell`` and assigns
        ``self.unit_cell_pos`` [sum(num_atoms * sym_mult), 3] (Cartesian),
        ``self.unit_cell_batch`` (graph index of each unit-cell atom) and
        ``self.unit_cell_mol_ind`` (molecule index of each unit-cell atom, counted
        across the whole batch). For each operator in ``symmetry_operators`` the
        molecule's shape is taken from ``pos`` (centred on its heavy-atom centroid and
        rotated in the fractional frame), and its centroid is the operator applied to
        the fractional ``aunit_centroid`` (clipped to [0, 1 - 1e-4]), reduced modulo 1
        except that a value of exactly 1.0 is kept. The absolute location of ``pos`` is
        therefore not used. Each graph is one unit: a Z' = 1 batch or the output of
        split_to_zp1_batch. A batch with more than one pose slot (``aunit_centroid``
        wider than 3 columns, as at ``max_z_prime > 1``) whose rows all have
        ``z_prime == 1`` is built from slot 0, since its other slots
        are pads; it gets the unit cell its width-1 batch gets, bit for bit (since
        2026-10-22: the MLIP input builders' own rebuild branch calls pose_aunit and
        build_unit_cell unsplit whenever ``z_prime.amax() == 1``, and aunit2ucell's
        einsum used to fail on the extra columns). A single object is collated into a
        one-graph batch for the computation.

        Raises
        ------
        ValueError
            If the batch has more than one pose slot and a row's ``z_prime`` is not 1:
            a Z'>1 row must be split first (split_to_zp1_batch; mol2ucell does), and a
            row below 1 holds no unit. Before 2026-10-22 both stopped in torch einsum.
            A batch with one slot is not checked (mol2ucell refuses a Z'>1 row there).
        """
        batch = self if self.is_batch else collate_data_list([self])
        if batch.aunit_centroid.shape[1] > 3:  # more than one pose slot: only a batch of one-unit rows is built
            zp = batch.z_prime.reshape(-1) if batch.z_prime is not None else None
            if zp is None or bool((zp != 1).any()):  # one device sync, only on a wide batch
                bad = list(range(batch.num_graphs)) if zp is None else torch.nonzero(zp != 1).flatten().tolist()
                why = ('no z_prime, so its pad slots are unknown' if zp is None else
                       '; '.join(f'{_rows_named(batch, [i])}: z_prime {int(zp[i])}' for i in bad[:_ROWS_NAMED])
                       + (f'; and {len(bad) - _ROWS_NAMED} more' if len(bad) > _ROWS_NAMED else ''))
                raise ValueError(f"build_unit_cell: builds each graph as one unit from pose slot 0, but this batch "
                                 f"has {batch.aunit_centroid.shape[1] // 3} pose slots and {len(bad)} row(s) that are "
                                 f"not one unit ({why}); split Z'>1 rows first (split_to_zp1_batch; mol2ucell does "
                                 f"this)")
            batch = copy.copy(batch)  # a shallow view: slots past 0 are pads, and self keeps its own slots
            batch.aunit_centroid = batch.aunit_centroid[:, :3]
        self.unit_cell_pos, self.unit_cell_batch, self.unit_cell_mol_ind = aunit2ucell(batch)

    def build_cluster(self, cutoff: float = 6, supercell_size: int = 10, zp_buffer=0):
        """Build a finite cluster of periodic images around each asymmetric unit.

        Calls ``crystal_building.utils.ucell2cluster`` and returns a new cluster batch;
        ``self`` is not modified. For a batch, ``unit_cell_pos`` and
        ``unit_cell_batch`` must already exist (build_unit_cell); the cluster is centred
        on molecule 0 of each graph's unit cell (the first image, at the zero
        translation; ``crystal_building.utils.reference_molecule_reach``). A single object
        is collated into a one-graph batch and its unit cell is built from ``pos`` first.

        A unit-cell translation is instantiated when one of the translated cell's molecules
        has its all-atom centroid within ``cutoff + zp_buffer + 2 * reach + REACH_MARGIN``
        of the centre, ``reach`` the molecule's all-atom reach about its all-atom centroid
        (``crystal_building.utils.molecule_reach``; until 2026-10-23 the heavy-atom
        ``radius`` + 0.1, which hydrogens reach past, audit #16); every such translation is
        found, whatever the cell basis (``crystal_building.utils.get_cart_translations``),
        and the untranslated cell is always kept. A crystal keeps at most
        MAX_CELLS_PER_CRYSTAL cells, the nearest ones (only a cell much smaller than its
        molecule reaches that). A crystal with a non-finite or singular cell, or a
        non-finite centre or molecule centroid, keeps only its untranslated cell, with a
        warning; ``analyze`` returns NaN for it.

        On the returned batch, ``pos``, ``z``, ``x``, ``batch`` and ``ptr`` index the
        cluster atoms; ``mol_ind`` numbers the molecules within each cluster; and
        ``aux_ind`` is 0 for the molecule with ``mol_ind == 0`` (the first molecule of
        the untranslated unit cell), 2 for molecules whose centroid is farther than
        that distance from the centre (they hold no atom within ``cutoff + zp_buffer`` of
        the asymmetric unit), and 1 otherwise. ``num_atoms`` keeps the asymmetric-unit
        atom counts.

        Parameters
        ----------
        cutoff : float, optional
            Interaction cutoff in Angstrom. Default 6.
        supercell_size : int, optional
            Ignored since 2026-10-18; it used to cap the translations tested along each
            cell vector, which dropped needed ones for oblique cells. Kept so existing calls
            work. Default 10.
        zp_buffer : float or torch.Tensor, optional
            Extra distance in Angstrom added to ``cutoff`` for cell selection and for
            the ``aux_ind == 2`` test; a tensor is per graph, [num_graphs]. Default 0.
            mol2cluster passes ``_unit_reach_buffer`` for the unit-graphs of a Z'>1 build.

        Returns
        -------
        MolCrystalData batch
            The cluster batch.

        Raises
        ------
        ValueError
            If any graph has ``z_prime > 1`` (split such batches first; mol2cluster
            does).
        AssertionError
            If the ``mol_ind == 0`` molecule of a cluster is itself farther than the
            ``aux_ind == 2`` threshold from the centre (an explicit raise).
        """
        if self.is_batch:
            return ucell2cluster(self, cutoff=cutoff, supercell_size=supercell_size, zp_buffer=zp_buffer)
        else:
            crystal_batch = collate_data_list([self])
            crystal_batch.build_unit_cell()
            return ucell2cluster(crystal_batch, cutoff=cutoff, supercell_size=supercell_size, zp_buffer=zp_buffer)

    def de_cluster(self):  # todo check and consider rewrite with new methods
        """Reduce a cluster batch to its ``aux_ind == 0`` atoms, in place.

        Keeps only atoms with ``aux_ind == 0`` in ``pos``, ``z`` and (if not None)
        ``x``; rebuilds ``batch`` and ``ptr`` from ``num_atoms``; and sets ``aux_ind``,
        ``mol_ind`` and ``edges_dict`` to None. Other attributes, including
        ``unit_cell_pos``, are unchanged.

        Raises
        ------
        RuntimeError
            If ``aux_ind`` is None.
        """
        # delete cluster information and reset this object as a molecule
        if self.aux_ind is not None:
            aunit_bools = self.aux_ind == 0
            self.pos = self.pos[aunit_bools]
            if self.x is not None:
                self.x = self.x[aunit_bools]
            self.z = self.z[aunit_bools]
            self.batch = torch.arange(self.num_graphs, device=self.device
                                      ).repeat_interleave(self.num_atoms)
            self.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device),
                                  torch.cumsum(self.num_atoms, dim=0)]).long()

            self.aux_ind = None
            self.mol_ind = None
            self.edges_dict = None
        else:
            raise RuntimeError("can't de-cluster - this is already not a cluster")

    def mol2cluster(self, cutoff: float = 6,
                    supercell_size: int = 10,
                    std_orientation: Optional[bool] = True):
        """Pose the asymmetric unit, build the unit cell and return a cluster; ``self`` is not modified.

        If ``max_z_prime == 1``, a clone of ``self`` is posed (pose_aunit), its unit
        cell built (build_unit_cell) and its cluster returned (build_cluster with
        ``zp_buffer = 0``).

        If ``max_z_prime > 1``, every crystal in the batch is split into one graph per
        asymmetric-unit molecule (split_to_zp1_batch); these are posed, built and
        clustered, and the clusters are recombined per crystal
        (join_zp1_cluster_batch; a batch of only Z'=1 rows gets the clusters a
        ``max_z_prime == 1`` build makes). Each unit-graph's cluster holds the images of
        that unit only, so its cutoff is widened by ``zp_buffer`` (_unit_reach_buffer,
        from each unit's molecule 0 in its built unit cell): enough that it holds every image of the unit with an
        atom within ``cutoff`` of any asymmetric-unit molecule of its crystal, and no
        more than the triangle inequality needs. A Z'=1 row gets 0 and the cluster it
        gets alone. Until 2026-10-23 the buffer was the largest distance between the
        crystal's centroid slots, on top of a reach built from the row's summed
        heavy-atom radius (audit #11, #16).

        Parameters
        ----------
        cutoff : float, optional
            Interaction cutoff in Angstrom, passed to build_cluster. Default 6.
        supercell_size : int, optional
            Passed to build_cluster. Default 10.
        std_orientation : bool, optional
            Passed to pose_aunit. Default True.

        Returns
        -------
        MolCrystalData batch
            The cluster batch described in build_cluster, combined as described in
            join_zp1_cluster_batch when ``max_z_prime > 1``.

        Raises
        ------
        ValueError
            If a row's ``z_prime`` exceeds ``max_z_prime`` (it would be posed as one
            rigid body) or is below 1 (it holds no unit), at any width; from
            split_to_zp1_batch if ``max_z_prime > 1`` and a row cannot be split; from
            build_cluster.
        AssertionError
            From build_cluster (an explicit raise; see there).
        """
        self._refuse_z_prime_outside_slots('mol2cluster')
        if self.max_z_prime > 1:
            # if there are any Z'>1 crystals in the batch, we
            # unzip for unit cell generation then re zip
            zp1_batch = self.split_to_zp1_batch()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            # widen each unit's cluster to reach every asymmetric-unit molecule of its own crystal (each unit's
            # molecule 0 in its built unit cell)
            zp_buffer = self._unit_reach_buffer(zp1_batch)
            zp1_cluster = zp1_batch.build_cluster(cutoff=cutoff, supercell_size=supercell_size, zp_buffer=zp_buffer)
            return self.join_zp1_cluster_batch(zp1_cluster)

        else:
            # split batches here to avoid silently mutating the original crystal
            zp1_batch = self.clone()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            return zp1_batch.build_cluster(cutoff, supercell_size)

    def _refuse_z_prime_outside_slots(self, caller: str):
        """Raise ValueError, naming each row and its reason, when a row's ``z_prime`` is outside 1..``max_z_prime``.

        The build dispatches on the batch-wide ``max_z_prime``: at 1 it poses each row's whole graph as one rigid
        body from slot 0, which for a Z'>1 row is wrong (silently, on the unit-cell route; audit #12). Per-row
        dispatch is a later stage; until then such a row is refused. A row with ``z_prime < 1`` holds no unit: at
        width 1 it used to be posed and scored as a Z'=1 crystal while split_to_zp1_batch refused it at width 2, so
        it is refused here at every width. A batch without ``z_prime`` passes.

        At ``max_z_prime == 1`` a row carrying a unit map (``unit_ind``) is also checked against it here, in the same
        device sync (``crystal_building.utils.unit_layout``'s rule: a Z'=1 row's map is all zeros), so a row whose
        map holds more units than its ``z_prime`` (a Z'=2 row relabelled Z'=1) is refused, not posed as one rigid
        body (since 2026-10-22, stage 2). At ``max_z_prime > 1`` the split reads the map and refuses such rows.
        """
        z_prime = self.z_prime
        if z_prime is None:
            return
        zp = z_prime.reshape(-1)
        mz = int(self.max_z_prime)
        bad = (zp > mz) | (zp < 1)
        map_reason = None
        if mz == 1 and UNIT_MAP_KEY in self._store:
            try:
                _, map_bad, map_reason = _unit_layout_unchecked(self)
            except ValueError as e:  # a stored map that does not hold one entry per atom: no row to name
                raise ValueError(f"{caller}: {e}") from e
            bad = bad | map_bad
        if bool(bad.any()):  # one device sync on the valid path; the row list only when refusing
            bad = torch.nonzero(bad).flatten().tolist()
            reasons = []
            for i in bad[:_ROWS_NAMED]:
                zp_i = int(zp[i])
                if zp_i > mz:
                    why = (f'z_prime {zp_i}, more units than slots (it would be posed as one rigid body; collate it '
                           f'at a max_z_prime of at least its z_prime)')
                elif zp_i < 1:
                    why = f'z_prime {zp_i} (a row holds at least one unit)'
                else:
                    why = f'its unit map disagrees with its z_prime: {map_reason(i)}'
                reasons.append(f'{_rows_named(self, [i])}: {why}')
            more = f'; and {len(bad) - _ROWS_NAMED} more' if len(bad) > _ROWS_NAMED else ''
            raise ValueError(f"{caller}: {len(bad)} row(s) with a z_prime outside 1..max_z_prime = {mz} pose "
                             f"slot(s), or one its unit map disagrees with: {'; '.join(reasons)}{more}")

    def _unit_reach_buffer(self, zp1_batch):
        """Extra cutoff (Angstrom) of each unit-graph of a Z'>1 build, [sum(z_prime)], from each unit's molecule 0.

        ``zp1_batch`` is this batch's split_to_zp1_batch after pose_aunit and build_unit_cell. The cluster of
        unit-graph v holds images of unit v only, and its ``aux_ind == 0`` molecule is molecule 0 of v's unit cell.
        An image M of v holds an atom within ``cutoff`` of an atom of unit u's molecule 0 only if
        ``|c_M - c_u| <= cutoff + R_u + R_v`` (c the all-atom centroids, R the all-atom reaches of each unit's
        molecule 0, ``crystal_building.utils.reference_molecule_reach``; not of the posed units, which sit about
        the unclipped ``aunit_centroid``), hence only if
        ``|c_M - c_v| <= cutoff + R_v + max_u (R_u + |c_u - c_v|)``, u over the units of v's crystal, v included.
        ucell2cluster tests ``cutoff + zp_buffer + 2 R_v``, so the buffer is ``max_u (R_u + |c_u - c_v|) - R_v``
        (>= 0). A Z'=1 row gets exactly 0, and a row's buffer reads no other row and no pad slot, so every row
        builds the cluster it builds alone.

        Since 2026-10-23 (audit #11, #16). Before, the buffer was the largest distance between the crystal's
        centroid slots (fractional ``aunit_centroid`` through ``T_fc``, where a collapsed-stack fault applied graph
        0's cell to the whole batch until 2026-08-13), on top of a reach built from the row's summed heavy-atom
        radius: wider than needed at Z'>1 (phenacetin P21/c at Z'=2: 4.2 times the Z'=1 cluster), and short of
        the hydrogens a heavy-atom radius misses.
        """
        centroid, reach = reference_molecule_reach(zp1_batch)                                     # [U, 3], [U]
        n_rows, width = self.num_graphs, int(self.max_z_prime)
        zp = self.z_prime.reshape(-1).long()
        row = torch.arange(n_rows, device=self.device).repeat_interleave(zp)                       # [U]
        slot = torch.arange(len(row), device=self.device) - (torch.cumsum(zp, 0) - zp)[row]        # [U]
        cen = centroid.new_zeros(n_rows, width, 3)
        cen[row, slot] = centroid
        rch = reach.new_full((n_rows, width), float('-inf'))                       # a pad slot is no unit u
        rch[row, slot] = reach
        dist = (cen[:, :, None, :] - cen[:, None, :, :]).norm(dim=-1)              # [n, u, v]
        need = (rch[:, :, None] + dist).amax(dim=1)                                 # [n, v]: max over u
        return need[row, slot] - reach

    def mol2ucell(self,
                  std_orientation: Optional[bool] = True):
        """Pose the asymmetric unit and build the unit cell on ``self``, in place.

        If ``max_z_prime == 1``, calls pose_aunit and build_unit_cell on ``self``. If
        ``max_z_prime > 1``, does the same on the output of split_to_zp1_batch and
        copies ``pos``, ``unit_cell_pos``, ``unit_cell_batch`` and
        ``unit_cell_mol_ind`` back (join_zp1_ucell_batch). In both cases ``self.pos``
        becomes the posed asymmetric unit. Returns None.

        Parameters
        ----------
        std_orientation : bool, optional
            Passed to pose_aunit. Default True.

        Raises
        ------
        ValueError
            If a row's ``z_prime`` exceeds ``max_z_prime`` (it would be posed as one
            rigid body) or is below 1 (it holds no unit), at any width; and from
            split_to_zp1_batch if ``max_z_prime > 1`` and a row cannot be split.
        """
        self._refuse_z_prime_outside_slots('mol2ucell')
        if self.max_z_prime > 1:
            zp1_batch = self.split_to_zp1_batch()
            # forward the flag: this branch used to call pose_aunit() bare, silently
            # substituting std_orientation=True while the Z'=1 branch below and both
            # mol2cluster branches forwarded it.  analyze() routes MLIP computes here
            # and cluster computes (elj/lj/es) through mol2cluster, so a Z'>1 crystal
            # whose `pos` is NOT standard-oriented got its two energies from geometries
            # up to 11 A apart.  Latent wherever `pos` is already standard-oriented --
            # which is an invariant nothing states or checks.
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()

            self.join_zp1_ucell_batch(zp1_batch)

        else:
            # split batches here to avoid silently mutating the original crystal
            self.pose_aunit(std_orientation=std_orientation)
            self.build_unit_cell()


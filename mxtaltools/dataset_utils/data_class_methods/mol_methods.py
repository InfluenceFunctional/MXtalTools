"""Molecule-level methods mixed into MolData.

Geometric properties (radius, mass, volume, principal axes), construction from SMILES,
recentring, orientation and perturbation of positions, a radius graph over each
molecule, visualisation, and a gas-phase UMA energy.
"""

from typing import Optional

import torch
from scipy.spatial.transform import Rotation
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import batch_compute_mol_radius, compute_mol_radius, batch_compute_mol_mass, \
    compute_mol_mass, batch_compute_molecule_volume, center_batch, apply_rotation_to_batch, rotvec2rotmat, \
    rotmat2rotvec, batch_molecule_principal_axes_torch
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes, canonicalize_rotvec
from mxtaltools.dataset_utils.mol_building import smiles2conformer
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import compute_molecule_uma_on_mxt_batch
# from mxtaltools.mlip_interfaces.uma_utils import compute_molecule_uma_on_mxt_batch
from mxtaltools.models.functions.radial_graph import build_radial_graph


# noinspection PyAttributeOutsideInit
class MolDataMethods:
    """Mixin of per-molecule methods for MolData objects and batches.

    ``pos`` holds Cartesian positions (Angstrom) and ``z`` atomic numbers; heavy atoms
    are those with ``z > 1``. Methods that branch on ``is_batch`` treat a single
    object as one molecule.
    """

    def mol_analysis(self,
                     force_reanalysis: Optional[bool] = False,
                     do_intramolecular_energy: Optional[bool] = False,
                     ):
        """Compute and store basic molecular properties, in place.

        Assigns ``self.radius`` (radius_calculation), ``self.mass``
        (mass_calculation) and ``self.mol_volume`` (volume_calculation), each only if
        it is None or ``force_reanalysis`` is True.

        Parameters
        ----------
        force_reanalysis : bool, optional
            Recompute radius, mass and volume even if already set. Default False.
        do_intramolecular_energy : bool, optional
            If True, also assign ``self.intramolecular_pot =
            self.intramolecular_energy()`` and ``self.intramolecular_clash =
            self.worst_intramolecular_clash()``, both using a force field fitted to
            the current conformer. Default False.
        """
        # only piece of analysis
        if self.radius is None or force_reanalysis:
            self.radius = self.radius_calculation()
        if self.mass is None or force_reanalysis:
            self.mass = self.mass_calculation()
        if self.mol_volume is None or force_reanalysis:
            self.mol_volume = self.volume_calculation()

        # opt-in: builds the internal-coordinate tree on first use, and fits the simple
        # force field to *this* conformer, so the bonded terms sit at their minimum and
        # only the nonbonded term is informative. See MolConformerMethods for scoring
        # candidates against a shared reference field instead.
        if do_intramolecular_energy:
            self.intramolecular_pot = self.intramolecular_energy()
            self.intramolecular_clash = self.worst_intramolecular_clash()

    @classmethod
    def from_smiles(cls,
                    smiles: str,
                    protonate: bool = True,
                    minimize: bool = False,
                    scramble_dihedrals: bool = False,
                    allow_methyl_rotations: bool = False,
                    compute_partial_charges: bool = True,
                    pare_to_size: Optional[int] = None,
                    max_pare_iters: int = 10,
                    do_mol_analysis: Optional[bool] = False,
                    ):
        """Build a molecule from a SMILES string via an RDKit conformer.

        The conformer comes from ``dataset_utils.mol_building.smiles2conformer``: RDKit
        embedding, then optional paring, dihedral scrambling and MMFF minimisation,
        then optional Gasteiger partial charges.

        Parameters
        ----------
        smiles : str
            Input SMILES; also stored as ``smiles`` and ``identifier``.
        protonate : bool, optional
            Add explicit hydrogens before embedding; if False, hydrogens are removed.
            Default True.
        minimize : bool, optional
            Run an MMFF optimisation of the final conformer. Default False.
        scramble_dihedrals : bool, optional
            Enables the dihedral-scrambling branch of smiles2conformer. That branch
            currently raises ValueError (see Raises). Default False.
        allow_methyl_rotations : bool, optional
            Passed to the dihedral scrambling; only read when ``scramble_dihedrals``
            is True. Default False.
        compute_partial_charges : bool, optional
            Store Gasteiger charges in ``x``; if False, ``x`` is zeros. Default True.
        pare_to_size : int, optional
            If given, remove fragments at rotatable bonds until at most this many heavy
            atoms remain. Default None.
        max_pare_iters : int, optional
            Maximum number of fragment removals when paring. Default 10.
        do_mol_analysis : bool, optional
            Passed to the constructor; if True, radius, mass and volume are computed.
            Default False.

        Returns
        -------
        cls or None
            A new object with ``z`` [num_atoms] (int64), ``pos`` [num_atoms, 3]
            (float32) and ``x`` [num_atoms] (float32), ``y`` and ``graph_x`` unset; or
            None if the SMILES does not parse, the embedded molecule has no conformer,
            or paring or its re-embedding fails.

        Raises
        ------
        ValueError
            If ``scramble_dihedrals`` is True: smiles2conformer unpacks the three
            values returned by ``scramble_dihedral_angles`` into two names.
        """
        conf_out = smiles2conformer(allow_methyl_rotations,
                                    compute_partial_charges,
                                    max_pare_iters,
                                    minimize,
                                    pare_to_size,
                                    protonate,
                                    scramble_dihedrals,
                                    smiles)
        if conf_out is not None:
            charges, pos, z = conf_out
            return cls(
                z=torch.LongTensor(z),
                pos=torch.tensor(pos, dtype=torch.float32),
                x=torch.tensor(charges, dtype=torch.float32),
                smiles=smiles,
                identifier=smiles,
                y=None,
                graph_x=None,
                do_mol_analysis=do_mol_analysis,
            )
        else:
            return None

    def construct_intra_radial_graph(self, cutoff: float = 6):
        """Build a radius graph over the atoms of each graph, in place.

        Assigns ``self.edges_dict = {'edge_index': ...}`` from
        ``models.functions.radial_graph.build_radial_graph`` without ``aux_ind`` (every
        pair of atoms in the same graph within ``cutoff``, both directions, at most
        10000 neighbours per atom) and ``self.edge_index`` [2, num_edges].

        Parameters
        ----------
        cutoff : float, optional
            Edge cutoff in Angstrom. Default 6.
        """
        self.edges_dict = build_radial_graph(self.pos,
                                             self.batch,
                                             self.ptr,
                                             cutoff,
                                             max_num_neighbors=10000
                                             )
        self.edge_index = self.edges_dict['edge_index']

    def radius_calculation(self, heavy_atoms_only: bool = True):
        """Return the molecular radius: the largest centroid-to-atom distance.

        For a batch with ``heavy_atoms_only``, both the centroid and the distances use
        heavy atoms only; otherwise all atoms. For a single object only heavy atoms are
        used, whatever ``heavy_atoms_only`` is. ``self`` is not modified.

        Parameters
        ----------
        heavy_atoms_only : bool, optional
            Batch only: restrict to heavy atoms. Default True.

        Returns
        -------
        torch.Tensor
            [num_graphs] for a batch, a scalar for a single object, in Angstrom.
        """
        if self.is_batch:
            if heavy_atoms_only:
                # cast BEFORE the scatter: a bool sum-scatter saturates at 1,
                # so `scatter(z > 1, ...).long()` returns ones, not counts
                heavies_per_graph = scatter((self.z > 1).long(), self.batch,
                                            reduce='sum', dim=0, dim_size = self.num_graphs)
                return batch_compute_mol_radius(self.pos[self.z>1],
                                                self.batch[self.z>1],
                                                self.num_graphs,
                                                heavies_per_graph)
            else:
                return batch_compute_mol_radius(self.pos, self.batch, self.num_graphs, self.num_atoms)
        else:
            return compute_mol_radius(self.pos[self.z > 1])

    def mass_calculation(self):
        """Return the molecular mass: the sum of ``ATOM_WEIGHTS[z]`` over all atoms, hydrogens included.

        Returns
        -------
        torch.Tensor
            float32; [num_graphs] for a batch, a scalar for a single object, in the
            units of ``constants.atom_properties.ATOM_WEIGHTS`` (standard atomic
            weights).
        """
        masses_tensor = torch.tensor(list(ATOM_WEIGHTS.values()), device=self.z.device, dtype=torch.float32)
        if self.is_batch:
            return batch_compute_mol_mass(self.z, self.batch, masses_tensor, self.num_graphs)
        else:
            return compute_mol_mass(self.z, masses_tensor)

    def volume_calculation(self):
        """Return an estimate of the molecular van der Waals volume.

        Sum of atomic spheres with radii from ``constants.atom_properties.VDW_RADII``,
        minus pairwise sphere intersections scaled by 0.7272
        (``common.geometry_utils.batch_compute_molecule_volume``).

        Returns
        -------
        torch.Tensor
            [num_graphs] for a batch, a scalar for a single object, in cubic Angstrom.
        """
        vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()),
                                        device=self.z.device,
                                        dtype=torch.float32)
        if self.is_batch:
            return batch_compute_molecule_volume(
                self.z,
                self.pos,
                self.batch,
                self.num_graphs,
                vdw_radii_tensor)
        else:
            return batch_compute_molecule_volume(
                self.z,
                self.pos,
                torch.zeros_like(self.z),
                1,
                vdw_radii_tensor)[0]

    def recenter_molecules(self, center_on_heavy_atoms=True):
        """Translate each molecule so that its centroid is at the origin, in place.

        For a batch, ``self.pos`` is replaced by positions centred on each graph's
        heavy-atom centroid if ``center_on_heavy_atoms``, else its all-atom centroid.
        For a single object the all-atom mean is subtracted from ``self.pos`` in place,
        whatever ``center_on_heavy_atoms`` is.

        Parameters
        ----------
        center_on_heavy_atoms : bool, optional
            Batch only: compute the centroid from heavy atoms. Default True.
        """
        if self.is_batch:
            self.pos = center_batch(self.pos,
                                    self.batch,
                                    self.num_graphs,
                                    self.num_atoms,
                                    center_on_heavy_atoms,
                                    self.z)
        else:
            self.pos -= self.pos.mean(0)

    def noise_positions(self, magnitude: float):
        """Add Gaussian noise with standard deviation ``magnitude`` to ``self.pos``, in place."""
        self.pos += torch.randn_like(self.pos) * magnitude

    def scale_positions(self, affine_scale: float):
        """Scale each graph's positions about the origin by a random factor, in place.

        Each graph's factor is ``clip(affine_scale + N(0, 1) / 10, 0.9, 1.25)`` and
        ``self.pos`` is multiplied by it. Reads ``num_graphs`` and ``num_atoms``.

        Parameters
        ----------
        affine_scale : float
            Mean factor before clipping.
        """
        graph_scale_factor = torch.randn(self.num_graphs, device=self.pos.device, dtype=torch.float32)
        graph_scale_factor = (graph_scale_factor / 10 + affine_scale).clip(min=0.9, max=1.25)
        atomwise_scaling = graph_scale_factor.repeat_interleave(self.num_atoms)
        self.pos *= atomwise_scaling[:, None]

    def visualize(self,
                  sample_inds: Optional[list] = None,
                  **vis_kwargs):
        """Convert graphs to ASE ``Atoms`` objects and open them in the ASE viewer.

        Parameters
        ----------
        sample_inds : list, optional
            Graph indices to convert; if None, all graphs.
        **vis_kwargs
            Forwarded to ``common.ase_interface.data_batch_to_ase_mols_list`` (and from
            there to ``ase_mol_from_crystaldata``).

        Returns
        -------
        list
            One ``ase_mol_from_crystaldata`` result per converted graph: an
            ``ase.Atoms``, or a tuple if ``return_crystal=True`` is passed in
            ``vis_kwargs``.
        """
        from mxtaltools.common.ase_interface import data_batch_to_ase_mols_list
        return data_batch_to_ase_mols_list(
            self,
            specific_inds=sample_inds,
            show_mols=True,
            **vis_kwargs)

    def orient_molecule(self,
                        mode: str,
                        include_inversion: bool = True,
                        target_handedness: Optional[torch.LongTensor] = None,
                        correct_orientation: bool = False,
                        override_random_rotations: Optional[torch.Tensor] = None,
                        ):
        """Recentre and rotate every molecule of a batch, in place.

        Molecules are first centred on their heavy-atom centroids
        (recenter_molecules). ``self.pos`` is then rotated per graph:

        - ``mode`` 'standardized', 'std' or 'standard': heavy-atom principal axes are
          aligned to the Cartesian axes
          (``crystal_building.utils.align_mol_batch_to_standard_axes``);
        - ``mode`` 'random': ``override_random_rotations`` is applied if given;
          otherwise uniformly random rotation matrices, each multiplied by -1 with
          probability 1/2 when ``include_inversion`` is True.

        Parameters
        ----------
        mode : str
            'standardized', 'std', 'standard' or 'random'.
        include_inversion : bool, optional
            Random mode without override only: randomly invert. Default True.
        target_handedness : torch.LongTensor, optional
            Standard mode only: [num_graphs, k] handedness; the first column is read,
            1 aligning to (x, y, z) and -1 to (-x, y, z). If None, all molecules are
            aligned to (x, y, z).
        correct_orientation : bool, optional
            If True and ``self`` has ``aunit_orientation``, replace it with the
            canonicalised rotation vector of
            ``R(aunit_orientation) @ inv(applied_rotation)``, so that the product of
            the new orientation's rotation with the applied rotation equals the old
            orientation's rotation. Default False.
        override_random_rotations : torch.Tensor, optional
            Random mode: [num_graphs, 3, 3] matrices applied instead of random ones.

        Raises
        ------
        AssertionError
            If ``self`` is not a batch; if ``mode`` is not recognised; in random mode
            without override, if ``include_inversion`` and ``correct_orientation`` are
            both True; or if ``correct_orientation`` applies and ``max_z_prime != 1``.
        """
        if self.is_batch:
            # always center the molecules
            self.recenter_molecules()
            if mode == 'standardized' or mode == 'std' or mode == 'standard':
                mol_batch, std_rotation = align_mol_batch_to_standard_axes(self,
                                                                           handedness=target_handedness,
                                                                           return_rot=True)
                self.pos = mol_batch.pos
                applied_rotation = std_rotation  # .permute(0, 2, 1)  # different basis, unfortunately
            elif mode == 'random':  # technically given the below we can do any applied rotation
                if override_random_rotations is not None:
                    random_rotations = override_random_rotations
                else:
                    random_rotations = torch.tensor(
                        Rotation.random(num=self.num_graphs).as_matrix(),
                        device=self.device, dtype=torch.float32)

                    if include_inversion:
                        assert not correct_orientation, "Cannot reconstruct rotations through inversion."
                        invert_inds = torch.randint(2, (self.num_graphs,), device=random_rotations.device) * 2 - 1
                        random_rotations *= invert_inds[:, None, None]

                self.pos = apply_rotation_to_batch(
                    self.pos,
                    random_rotations,
                    self.batch)

                applied_rotation = random_rotations

            else:
                assert False, "Rotation must be standard or random"

            if correct_orientation and hasattr(self, 'aunit_orientation'):
                assert self.max_z_prime == 1, "Mol orientation should be done on unzipped batches of independent Z'=1 crystals, not mixed Z'>1"
                # adjust aunit_orientation such that the final orientation is unchanged
                # new combined rotation should undo the random rotation, then apply the aunit_orientation
                invert_random_rotation = torch.linalg.inv(applied_rotation)
                new_orientation_matrix = rotvec2rotmat(self.aunit_orientation) @ invert_random_rotation
                new_rotvec = rotmat2rotvec(new_orientation_matrix)
                new_rotvec_fixed = canonicalize_rotvec(new_rotvec)
                # correction is successful when this is small
                # err = (rotvec2rotmat(new_rotvec_fixed) @ applied_rotation - rotvec2rotmat(
                #     self.aunit_orientation)).abs().sum(dim=(1, 2))
                # print(err.mean())
                self.aunit_orientation = new_rotvec_fixed

        else:
            assert False, "molecule orientation not implemented for single samples"

    def rotate_embedding(self, rotations):
        """Return ``self.embedding`` rotated per graph; ``self`` is not modified.

        Computes ``einsum('nij, njk -> nik', rotations, self.embedding)``. This assumes
        the embedding is [n, 3, k] with the Cartesian axes on dimension ``j`` and
        feature channels on dimension ``k``.

        Parameters
        ----------
        rotations : torch.Tensor
            [n, 3, 3] rotation matrices.

        Returns
        -------
        torch.Tensor
            [n, 3, k] rotated embedding.
        """
        return torch.einsum('nij, njk -> nik', rotations, self.embedding)

    def deprotonate(self):
        """Remove hydrogen atoms (``z == 1``), in place.

        Filters ``z`` and ``pos``. If ``batch`` is present and not None, also filters
        ``batch`` and rebuilds ``ptr`` and ``num_atoms`` from the remaining atoms;
        otherwise sets ``num_atoms`` to the remaining atom count (a Python int). Sets
        ``num_nodes``. Does nothing if no heavy atom is present.

        Warnings
        --------
        Use with care: this breaks several batching methods. Other per-atom attributes
        (for example ``x``) are not filtered, graph-level properties (``radius``,
        ``mass``, ``mol_volume``) are not recomputed, and the PyG slice/increment
        bookkeeping is not updated. A graph with no heavy atoms is dropped from
        ``ptr`` and ``num_atoms`` but not from the other per-graph attributes.
        """
        heavy_atom_inds = torch.argwhere(self.z != 1).flatten()  # protons are atom type 1
        if len(heavy_atom_inds) > 0:
            self.z = self.z[heavy_atom_inds]
            self.pos = self.pos[heavy_atom_inds]
            if hasattr(self, 'batch') and self.batch is not None:
                self.batch = self.batch[heavy_atom_inds]
                a, b = torch.unique(self.batch, return_counts=True)
                self.ptr = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(b, dim=0)]).long()
                self.num_atoms = torch.diff(self.ptr).long()
            else:
                self.num_atoms = len(self.z)
            self.num_nodes = len(self.z)

    def compute_Ip(self):
        """Return the heavy-atom principal inertial axes and moments of each graph in a batch.

        Uses unit masses on heavy atoms, centred on their centroid
        (``common.geometry_utils.batch_molecule_principal_axes_torch``). When ``pos``
        requires grad, that function adds Gaussian noise of standard deviation 0.05 to
        the coordinates before computing the axes. ``self`` is not modified.

        Returns
        -------
        principal_axes : torch.Tensor
            [num_graphs, 3, 3], one unit axis per row, ordered by ascending moment,
            signed to non-negative overlap with the direction from the centroid to the
            farthest heavy atom; where an overlap vanishes, one axis may be negated
            instead (``common.geometry_utils.correct_Ip_directions``).
        principal_moments : torch.Tensor
            [num_graphs, 3], ascending, in square Angstrom (unit masses).

        Raises
        ------
        AssertionError
            If ``self`` is not a batch, or if ``num_atoms`` does not match the atom
            counts in ``batch``.
        """
        assert self.is_batch

        principal_axes, principal_moments, _ = batch_molecule_principal_axes_torch(
            self.pos,
            self.batch,
            self.num_graphs,
            self.num_atoms,
            heavy_atoms_only=True,
            atom_types=self.z
        )

        return principal_axes, principal_moments

    def compute_molecule_uma(self,
                            predictor,
                            ):
        """Return the UMA energy of each molecule as an isolated, non-periodic system.

        Each graph is converted to an ASE ``Atoms`` object with periodicity off and an
        ``AtomicData`` with task name 'omol', and the batch is evaluated by
        ``predictor`` (``mlip_interfaces.uma_utils.compute_molecule_uma_on_mxt_batch``).
        A single object is collated into a one-graph batch first. ``self`` is not
        modified.

        Parameters
        ----------
        predictor
            UMA predictor, called through ``uma_utils.safe_predict_uma``.

        Returns
        -------
        torch.Tensor
            The predictor's ``'energy'`` output, without unit conversion; float32 NaN
            [num_graphs] if ``safe_predict_uma`` reports a crash.

        Raises
        ------
        RuntimeError
            From ``safe_predict_uma`` (out of memory, or the consecutive-crash bound).
        """
        if self.is_batch:
            return compute_molecule_uma_on_mxt_batch(self,
                                            predictor)
        else:
            return compute_molecule_uma_on_mxt_batch(collate_data_list(self),
                                            predictor)

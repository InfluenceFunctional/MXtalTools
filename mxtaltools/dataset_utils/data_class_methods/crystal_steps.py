"""Structural full-chain steps mixed into MolCrystalData: molecules, pose, rebuild check, validity.

The steps of the ingestion contract (``docs/design/crystal_ingestion.md`` §4, Tables 7 and 8,
in the GFN repository) that need no chemistry backend. They run on batches made by
``MolCrystalData.read`` / ``from_arrays`` (``crystal_io.py``), in this order:

``find_molecules``
    bare -> the deposited set split into molecules: Z', whole (unwrapped) units as equal
    contiguous atom blocks, units 1+ re-indexed onto unit 0's atom order.
``parameterize_pose``
    units -> posed: each unit's own image in the asymmetric-unit box, its centroid,
    orientation and handedness; the state the unit-cell builder consumes.
``check_rebuild``
    posed -> the unit cell built by ``mol2ucell`` matched, element by element, to the
    deposited set expanded by the file's operators.
``check_validity``
    any state -> named thresholds, each applied only when given.

Every step follows one contract:

- it returns a NEW batch and never mutates its input;
- rows whose ``status`` is not ``ok`` on entry pass through bit-identical (their ``done``
  is not changed either);
- every row it runs on gets its bit in ``done`` (``DONE_*``), whatever the outcome; the
  outcome is the row's ``status``;
- a row that fails gets one of the step's status codes (``STATUS``) and a message, never
  an exception, and never changes another row. A ``RuntimeError`` or ``ValueError`` raised
  while processing rows is isolated by retrying on halves down to the single row, which
  gets the step's ``*_failed`` code (``pose_failed`` for the pose); any other exception is
  a programming error and propagates;
- bad ARGUMENTS raise;
- ``find_molecules`` and ``parameterize_pose`` run once per batch (a second pass would take
  their output for their input, and is refused); ``check_rebuild`` and ``check_validity``
  may run again: they re-check the ok rows and pass the others through as above;
- the arguments it ran with are appended to ``steps_run`` (a batch-level list of
  ``{'step': name, 'args': {...}}``), ready for the store's ``save(steps_run=...)``.

Status codes: ``STATUS`` is ``READ_STATUS`` followed by the codes of these steps
(``STEP_STATUS``); ``status`` holds the index. Codes are only ever appended.
"""
import copy
import itertools
from typing import Callable, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from ase.data import chemical_symbols, covalent_radii
from scipy.optimize import linear_sum_assignment
from torch_geometric.data.storage import GlobalStorage
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import (batch_compute_fractional_transform, batch_compute_mol_mass,
                                              batch_compute_mol_radius, batch_compute_molecule_volume,
                                              batch_molecule_principal_axes_torch, compute_Ip_handedness,
                                              extract_rotmat, rotmat2rotvec)
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from mxtaltools.crystal_building.utils import (canonicalize_rotvec, cleanup_invalid_rotvecs,
                                               identify_canonical_asymmetric_unit, unit_layout, equal_block_unit_ind,
                                               unit_map_problem, UNIT_MAP_KEY, NO_UNIT)
from mxtaltools.dataset_utils.construction.cif.composition import parse_formula_moiety
from mxtaltools.dataset_utils.construction.cif.topology import _POLYMER_RE, union_find, unwrap_components
from mxtaltools.dataset_utils.data_class_methods.crystal_io import POSE_KEYS, READ_STATUS

__all__ = ['MolCrystalSteps', 'STATUS', 'STEP_STATUS', 'VALIDITY_CRITERIA',
           'DONE_FIND_MOLECULES', 'DONE_SANITIZE', 'DONE_POSE', 'DONE_REBUILD', 'DONE_VALIDITY']

#: Codes of the structural steps, appended to ``READ_STATUS`` (``z_prime_mismatch`` is
#: already there and is reused by ``find_molecules``). Per step:
#:
#: - ``find_molecules``: ``no_bonds``, ``polymer``, ``special_position``, ``unlisted_bond``,
#:   ``z_prime_fractional``, ``split_molecule``, ``multi_component``, ``z_prime_mismatch``,
#:   ``unit_order_failed``, ``find_molecules_failed``;
#: - ``parameterize_pose``: ``degenerate_axes``, ``pose_failed``;
#: - ``check_rebuild``: ``unmatched_atoms``, ``element_mismatch``, ``rebuild_mismatch``,
#:   ``rebuild_failed``;
#: - ``check_validity``: one per criterion (``VALIDITY_CRITERIA``), and ``validity_failed``.
STEP_STATUS = ('no_bonds', 'polymer', 'special_position', 'z_prime_fractional', 'split_molecule',
               'multi_component', 'unit_order_failed', 'find_molecules_failed',
               'degenerate_axes', 'pose_failed',
               'unmatched_atoms', 'element_mismatch', 'rebuild_mismatch', 'rebuild_failed',
               'atomic_number_above_max', 'heavy_atoms_above_max', 'heavy_nn_out_of_range', 'occupancy_not_one',
               'disordered', 'cell_constraints_violated', 'atoms_too_close', 'validity_failed',
               'unlisted_bond', 'composition_mismatch')
STATUS = tuple(READ_STATUS) + STEP_STATUS
_STATUS = {name: i for i, name in enumerate(STATUS)}
assert len(_STATUS) == len(STATUS), 'a step status repeats a read status'
_OK = _STATUS['ok']

#: ``done`` bits (``DONE_READ`` = 1 is the read's). Sanitize, features and canonicalize
#: are not built; their bits are reserved so the order of the chain stays the bit order.
DONE_FIND_MOLECULES = 1 << 1
DONE_SANITIZE = 1 << 2
DONE_POSE = 1 << 3
DONE_REBUILD = 1 << 4
DONE_VALIDITY = 1 << 5

#: ``check_validity`` criteria, in the order their statuses take precedence; bit ``i`` of
#: ``validity_checked`` says criterion ``i`` ran on the row.
VALIDITY_CRITERIA = ('max_atomic_number', 'max_heavy_atoms', 'heavy_nn_range', 'occupancy', 'disorder',
                     'cell_constraints', 'min_contact', 'composition')
_VALIDITY_CODES = dict(max_atomic_number='atomic_number_above_max', max_heavy_atoms='heavy_atoms_above_max',
                       heavy_nn_range='heavy_nn_out_of_range', occupancy='occupancy_not_one', disorder='disordered',
                       cell_constraints='cell_constraints_violated', min_contact='atoms_too_close',
                       composition='composition_mismatch')

#: Attributes each step writes. ``find_molecules`` and ``parameterize_pose`` refuse a batch
#: that already holds any of theirs (a second pass would overwrite the rows it passes
#: through, and would take its own output for its input).
_FIND_KEYS = ('mol_ind', 'deposit_site', 'deposit_frac', 'mol_bonds', 'z_prime', 'n_components', 'unit_radius',
              'unit_mass', 'unit_volume', 'radius', 'mass', 'mol_volume', UNIT_MAP_KEY)
#: Not ``unit_ind``: a map the batch already carries (an earlier step's) is checked and kept by
#: ``parameterize_pose``, not overwritten; the ``DONE_POSE`` bit refuses a second pass.
_POSE_KEYS = ('aunit_centroid', 'aunit_orientation', 'aunit_handedness', 'aunit_op', 'is_well_defined',
              'asu_ambiguous', 'handedness_unstable', 'max_z_prime', 'x')

#: A unit's heavy-atom centroid within this distance (fractional) of a face of the
#: asymmetric-unit box counts as on the face: which image is "in the box" is then set by
#: rounding (float32 storage rounds fractional coordinates near 1 by about 6e-8).
_ASU_FACE_TOL = 1e-6

#: Largest finite site-symmetry group (a crystallographic point group has at most 48
#: elements); a closure that grows past it is infinite.
_MAX_SITE_GROUP = 48

#: Atom pairs per block in the pairwise searches (bounds memory, not results).
_PAIR_BLOCK = 1 << 21

#: Unit re-indexing when the isomorphism search reaches its bound: at most
#: ``_REFINE_ROUNDS`` refits per rule, each searching only the maps that send every heavy
#: atom to a same-element atom near it after the best fit so far. The rules, in order: the
#: atoms within ``_REFINE_TOL`` (Angstrom) of the nearest one (it separates the methyls of
#: a rotor, about 2.5 A apart); then the ``_REFINE_CANDIDATES`` nearest (for units whose
#: conformations differ by more than the tolerance).
_REFINE_ROUNDS = 3
_REFINE_TOL = 0.5
_REFINE_CANDIDATES = 4
#: Geometry-only starting fits tried by the refinement (``_geometric_fits``), besides the
#: capped search's best.
_GEOMETRIC_STARTS = 2

#: The isomorphism search's work bound, per search: at most ``_ISO_WORK * iso_cap``
#: candidate atom pairs tested (networkx's ``node_match`` calls). Enumerating one
#: isomorphism after the first takes about 10 (measured on rotor-rich chains), and VF2 can
#: backtrack without bound before its first isomorphism, or before proving there is none.
_ISO_WORK = 20

_F64 = torch.float64


class _Refusal(Exception):
    """Internal: one row fails a step; carries the ``STATUS`` name and a message."""

    def __init__(self, code: str, message: str, **info):
        assert code in _STATUS, code
        self.code = code
        self.info = info
        super().__init__(message)


class _SearchBound(Exception):
    """Internal: an isomorphism search reached its work bound (``_ISO_WORK * iso_cap``)."""


# noinspection PyAttributeOutsideInit
class MolCrystalSteps:
    """Mixin providing the structural full-chain steps (module docstring)."""

    def find_molecules(self, *, bonds: str, bond_scale, iso_cap: int, special_tol: float):
        """Split each row's deposited set into molecules: Z', whole units, one atom order.

        Parameters
        ----------
        bonds : {'file', 'distance'}
            Where the bonds within the deposited set come from: the file's ``_geom_bond``
            loop (``file_bonds``, kept by ``read``), or covalent radii, under periodic
            boundaries (atoms i, j bonded when ``|x_i - x_j - t| < s_i r_i + s_j r_j`` for
            some lattice translation t; ASE's covalent radii, ``s`` from ``bond_scale``).
            With 'file', two atoms of different components within that distance of each
            other in the deposited set itself refuse the row (``unlisted_bond``): a loop
            that omits bonds (a long metal-ligand bond; SHELXL writes no X-H bonds without
            ``BOND $H``) would otherwise split one molecule into several. Use 'distance'
            for such files.
        bond_scale : float or (float, float)
            The covalent-radius scale ``s``: one value for every atom, or (heavy atoms,
            hydrogen), e.g. (1.15, 1.35) as the generator benchmark's ingestion uses. With
            ``bonds='file'`` it is still used: for the test of bonds to symmetry images,
            which the file's bond loop does not list, and for ``unlisted_bond``.
        iso_cap : int
            Bounds each graph-isomorphism search when re-indexing a unit onto unit 0: at
            most ``iso_cap`` isomorphisms enumerated and at most ``20 * iso_cap`` candidate
            atom pairs tested. A search that reaches either bound is repeated among the
            maps sending each heavy atom to a same-element atom near it after a fit of
            unit k onto unit 0 (the capped search's best, and the two best fits from
            geometry alone: principal axes under each sign choice, refined by per-element
            assignment): first the atoms within 0.5 A of the nearest, then the 4 nearest;
            at most 3 refits each. If none of those searches completes:
            ``unit_order_failed``.
        special_tol : float
            Angstrom. An image of a deposited atom under a symmetry operation (other than
            the identity) closer than this to a deposited atom marks a special position.

        Returns
        -------
        MolCrystalDataBatch
            A new batch. On rows that pass, the atoms are reordered into ``z_prime`` equal
            contiguous units: unit ``k`` is the ``k``-th bonded component of the deposited
            set (numbered by its lowest deposited atom), unit 0 in deposited order, and
            units 1+ re-indexed onto unit 0's atom order; ``pos`` holds the unwrapped
            units (Cartesian, each molecule whole; its first atom in the file's own
            image). Every per-atom attribute is reordered with the atoms, and so are
            ``atom_labels`` and any other row-level list whose entries are lists of the
            row's atom count (how ``from_arrays`` stores a non-numeric per-atom list);
            ``file_bonds`` is renumbered. Added:

            - ``mol_ind`` [N] long: the unit of each atom (rows that fail keep their
              atom order and hold their component labels where components were found,
              else 0);
            - ``unit_ind`` [N] long: the unit map (``crystal_building.utils.UNIT_MAP_KEY``),
              the single authority for which unit an atom belongs to from here on: the
              units found, equal to ``mol_ind``, on the rows that pass, and ``NO_UNIT``
              (-1) on every other row (no units known), which every reader refuses
              (since stage 3, 2026-10-23; before, ``parameterize_pose`` wrote it, as
              ``sanitize``, which runs between the two, could not fill it for the
              hydrogens it places; it fills it now, and ``parameterize_pose`` checks and
              keeps it). A batch that already carries a map is refused (a second pass);
            - ``deposit_site`` [N] long: the atom's row-local index in the deposited set
              as read (with ``mol_ind``, the stored re-indexing permutation);
            - ``deposit_frac`` [N, 3] float64: the atom's fractional coordinates as
              deposited (before unwrapping), the reference ``check_rebuild`` uses;
            - ``mol_bonds``: list[B] of long [2, E] arrays, the bonds used, in the new atom
              order (None on rows that fail);
            - ``z_prime`` [B] long: units in the deposited set (0 on rows that fail);
              ``n_components`` [B] long: bonded components found (-1 where not reached);
            - ``unit_radius``, ``unit_mass``, ``unit_volume`` [B, Z'max]: per unit, with
              ``MolData.mol_analysis``'s definitions (heavy-atom radius, atomic-weight
              mass, overlap-corrected van der Waals volume), NaN past a row's Z'; and
              their row sums ``radius``, ``mass``, ``mol_volume`` [B], the latent's
              convention (NaN on rows that fail).

        Raises
        ------
        ValueError
            ``bonds`` not 'file' or 'distance', a ``bond_scale`` that is not one or two
            positive finite numbers, ``iso_cap`` not a positive int, ``special_tol`` not
            a positive finite number, a batch that already holds this step's output (any
            row through ``find_molecules``, or any attribute it writes) or a pose, or not
            a batch from ``read`` / ``from_arrays``.

        Notes
        -----
        Status codes, the first that applies naming the row:

        ``find_molecules_failed``
            the row's ``file_bonds`` entry is malformed (not an integer [2, E] array of
            row-local indices in [0, n), or a bond from an atom to itself), checked in
            both bond modes; or processing the row raised a ``RuntimeError`` or
            ``ValueError``.
        ``no_bonds``
            ``bonds='file'`` and the file has no bond loop; or a row of two or more atoms
            with no bond at all; or a row with no atoms.
        ``polymer``
            the moiety formula declares one (``)n``), a molecule bonds to its own lattice
            translate, or a molecule bonds to its own symmetry image under a map that
            generates an infinite chain.
        ``special_position``
            an image of a deposited atom under a symmetry operation other than the
            identity (lattice translates included) lies within ``special_tol`` of a
            deposited atom: an atom on a special position, or a molecule written whole
            on one (it would be duplicated in the cell).
        ``unlisted_bond``
            ``bonds='file'``: two atoms of different components lie within bonding
            distance (and at least ``special_tol`` apart) in the deposited set itself;
            the file's bond loop does not bond them.
        ``z_prime_fractional``
            a molecule bonds to its own symmetry image under maps generating a finite
            group: the molecule is completed by symmetry and Z' < 1 for it.
        ``split_molecule``
            a bonded component bonds to a symmetry image (or lattice translate) of
            another component: one molecule is split across the deposited set.
        ``multi_component``
            the units are not all the same molecule (element-labelled graphs differ):
            salts, solvates, cocrystals; a status for now (owner answer Q3).
        ``z_prime_mismatch``
            ``z_value`` is given and differs from Z' times the number of operators.
        ``unit_order_failed``
            the isomorphism search reached its bounds (``iso_cap``; also among the nearest
            candidates), or the matched order does not carry unit k's bonds onto unit 0's.

        Re-indexing (units 1+): heavy-atom graphs whose bond counts or Weisfeiler-Lehman
        hashes (element-labelled) differ are not isomorphic (``multi_component``, no
        search). Otherwise heavy atoms by an element-labelled graph isomorphism of the
        heavy-atom bond graphs, choosing among at most ``iso_cap`` isomorphisms the one
        with the smallest heavy-atom Kabsch RMSD to unit 0 (a proper or improper fit);
        hydrogens by assignment among the hydrogens of the matched parent atom after that
        fit. Ported from ``order_canon_check.py`` (GFN ``artifacts/fd_design_oct15``).
        """
        _check_input_batch(self)
        if bonds not in ('file', 'distance'):
            raise ValueError(f"bonds must be 'file' or 'distance', got {bonds!r}")
        scales = _bond_scales(bond_scale)
        iso_cap = _positive_int('iso_cap', iso_cap)
        special_tol = _positive_float('special_tol', special_tol)
        posed = [k for k in POSE_KEYS[1:] if k in self._store]
        if posed:
            raise ValueError(f'find_molecules takes a bare batch; this one is posed (holds {posed})')
        _refuse_second_pass(self, DONE_FIND_MOLECULES, _FIND_KEYS, 'find_molecules',
                            'its pos is already unwrapped and reordered, and a second pass would take it as deposited')

        B, ptr = self.num_graphs, self.ptr.detach().to('cpu').long()
        rows = _ok_rows(self)
        z_all = self.z.detach().to('cpu').long().numpy()
        T_fc64, T_cf64 = _transforms64(self)
        frac_all = torch.einsum('nij,nj->ni', T_cf64[_atom_rows(ptr)], self.pos.detach().to('cpu', _F64)).numpy()
        file_bonds = self['file_bonds'] if 'file_bonds' in self._store else [None] * B
        moiety = self['moiety'] if 'moiety' in self._store else [None] * B
        z_value = self['z_value'].tolist() if 'z_value' in self._store else [-1] * B
        if bonds == 'file' and 'file_bonds' not in self._store:
            raise ValueError("bonds='file' needs file_bonds (kept by read); this batch has none")

        def run(idx):
            out = {}
            for b in idx:
                s, e = int(ptr[b]), int(ptr[b + 1])
                try:
                    out[b] = _find_row(z_all[s:e], frac_all[s:e], T_fc64[b].numpy(), self.symmetry_operators[b],
                                       file_bonds[b], moiety[b], int(z_value[b]), bonds=bonds, scales=scales,
                                       iso_cap=iso_cap, special_tol=special_tol)
                except _Refusal as r:
                    out[b] = r
            return out

        results, failed = _isolate(run, rows)
        for b, msg in failed.items():
            results[b] = _Refusal('find_molecules_failed', msg)

        out = _copy_batch(self)
        N = int(ptr[-1])
        perm = torch.arange(N)
        new_frac = torch.as_tensor(frac_all).clone()
        mol_ind = torch.zeros(N, dtype=torch.long)
        z_prime = torch.zeros(B, dtype=torch.long)
        n_components = torch.full((B,), -1, dtype=torch.long)
        mol_bonds = [None] * B
        status = out.status.detach().to('cpu').clone()
        message = list(out.message)
        for b, res in results.items():
            s = int(ptr[b])
            if isinstance(res, _Refusal):
                status[b], message[b] = _STATUS[res.code], str(res)
                comp = res.info.get('components')
                if comp is not None:
                    mol_ind[s:s + len(comp)] = torch.as_tensor(comp)
                    n_components[b] = int(comp.max()) + 1 if len(comp) else 0
                continue
            order = torch.as_tensor(res['order'])
            perm[s:s + len(order)] = s + order
            new_frac[s:s + len(order)] = torch.as_tensor(res['frac'])[order]
            mol_ind[s:s + len(order)] = torch.as_tensor(res['unit'])
            z_prime[b] = res['z_prime']
            n_components[b] = res['z_prime']
            mol_bonds[b] = res['bonds']
            if res['note']:
                message[b] = res['note'] if not message[b] else f"{message[b]}; {res['note']}"

        # rows that fail keep their atom order: only the rows that pass are reordered
        _permute_atoms(out, perm, ptr, {b for b, res in results.items() if not isinstance(res, _Refusal)})
        device, dtype = self.pos.device, self.pos.dtype
        atom_row = _atom_rows(ptr)
        # the reordered atoms moved by whole lattice vectors only (identity on rows that did not
        # pass): add those to the given coordinates, so an atom that does not move keeps them exactly
        shift = torch.round(new_frac - torch.as_tensor(frac_all)[perm])
        moved = torch.einsum('nij,nj->ni', T_fc64[atom_row], shift)
        out['pos'] = (self.pos.detach().to('cpu', _F64)[perm] + moved).to(dtype).to(device)
        unmoved = (shift == 0).all(1).to(device)
        out['pos'][unmoved] = self.pos[perm.to(device)][unmoved]
        _set(out, 'mol_ind', mol_ind.to(device), 'atom')
        # the unit map (stage 3, 2026-10-23; before, parameterize_pose wrote it): the units found, Z' equal contiguous
        # blocks in unit order, on the rows that pass; NO_UNIT on every other row (no units known), which every reader
        # refuses
        passed = torch.zeros(B, dtype=torch.bool)
        for b, res in results.items():
            passed[b] = not isinstance(res, _Refusal)
        _set(out, UNIT_MAP_KEY, torch.where(passed[atom_row], mol_ind, torch.full_like(mol_ind, NO_UNIT)).to(device),
             'atom')
        _set(out, 'deposit_site', (perm - ptr[:-1][atom_row]).to(device), 'atom')
        _set(out, 'deposit_frac', torch.as_tensor(frac_all)[perm].to(device), 'atom')
        _set(out, 'mol_bonds', mol_bonds, 'row')
        _set(out, 'z_prime', z_prime.to(device), 'row')
        _set(out, 'n_components', n_components.to(device), 'row')
        _write_unit_properties(out, ptr, z_prime, mol_ind, dtype, device)
        _finish(out, self, status, message, results, DONE_FIND_MOLECULES, 'find_molecules',
                dict(bonds=bonds, bond_scale=list(scales), iso_cap=iso_cap, special_tol=special_tol))
        return out

    def parameterize_pose(self, *, handedness: str, degenerate_tol: float, handedness_tol: float):
        """Pose each unit: the image in the asymmetric-unit box, centroid, orientation, handedness.

        Per unit, the images under every operator of the row are formed, each shifted by
        a lattice vector so its heavy-atom centroid (unweighted) lies in [0, 1); the image
        whose centroid lies in the space group's asymmetric-unit box (``ASYM_UNITS``) is
        chosen by ``identify_canonical_asymmetric_unit``, as the old featurizer picks it
        (no centroid in the box, or several: the smallest centroid norm). That image is
        the unit's block of ``pos``: every unit keeps its OWN conformer (no copy of unit
        0). Orientation and handedness come from the image's heavy-atom principal axes by
        ``extract_aunit_orientation``'s construction (``batch_molecule_principal_axes_torch``,
        ``compute_Ip_handedness``, ``extract_rotmat``, ``rotmat2rotvec``,
        ``cleanup_invalid_rotvecs``, ``canonicalize_rotvec``), evaluated in float64 (that
        function builds its sign matrix in float32 and refuses float64 positions).

        Parameters
        ----------
        handedness : {'measured'}
            Each unit's handedness is its principal-axis frame's (+1 or -1). Forcing a
            handedness is excluded (D2 of ``fd_tooling.md``): only 'measured' is accepted.
        degenerate_tol : float
            Relative principal-moment gap, ``min(I2 - I1, I3 - I2) / I3``, below which a
            unit's axes are undetermined: ``degenerate_axes``.
        handedness_tol : float
            Dimensionless. A unit's handedness is flagged unstable when an axis's overlap
            (cosine) with the reference direction (to the farthest heavy atom) is below
            it, or when the two farthest heavy atoms are within this relative distance of
            each other: the axis signs are then set by rounding or by noise in the
            structure (planar molecules: the normal's overlap is the out-of-plane noise).

        Returns
        -------
        MolCrystalDataBatch
            A new batch in the posed state. ``pos`` holds the chosen images (Cartesian,
            batch dtype); ``deposit_frac`` keeps the deposited coordinates. Added or
            overwritten (pad: 0, as collate pads; -1 for ``aunit_op``):

            - ``aunit_centroid``, ``aunit_orientation`` [B, 3 Z'max] (batch dtype),
              ``aunit_handedness`` [B, Z'max] long; ``max_z_prime`` (python int, the
              largest Z' among the posed rows, at least 1);
            - ``aunit_op`` [B, Z'max] long: the operator index whose image was chosen;
            - ``is_well_defined`` [B] bool: exactly one image's centroid in the box for
              every unit (the old definition, now computed rather than always True);
            - ``asu_ambiguous`` [B] bool: some unit is not well defined, or one of its
              image centroids lies within 1e-6 (fractional) of a box face, so the choice
              can follow rounding;
            - ``handedness_unstable`` [B] bool (above);
            - ``x`` [N] zeros (batch dtype): charges belong to the features step;
            - ``unit_ind`` [N] long: the unit map, the single authority for which unit an
              atom belongs to (``crystal_building.utils.unit_layout``; since
              2026-10-22). The map the batch carries (``find_molecules`` writes it since
              stage 3, 2026-10-23, and ``sanitize`` fills it for the hydrogens it places)
              is checked on the ok rows and kept as it is. A batch without one (written
              before stage 3) gets it here, from ``mol_ind``: Z' equal contiguous blocks
              on every row whose ``mol_ind`` is those blocks, and ``NO_UNIT`` (-1) on every
              other row (``z_prime < 1``, atoms not divisible, or other units: e.g. a row
              ``find_molecules`` refused), so no row's map contradicts its ``z_prime`` or
              ``mol_ind`` (since the stage-2 review). An ok row whose units (the carried
              map, or ``mol_ind``) are not Z' equal contiguous blocks in unit order gets
              ``pose_failed`` naming them: this step poses those blocks. The unit sizes it
              poses are read through ``unit_layout``.

        Raises
        ------
        ValueError
            ``handedness`` other than 'measured', a threshold that is not a finite number
            >= 0, an ok row that ``find_molecules`` has not processed, a batch that
            already holds a pose (any row through ``parameterize_pose``, or any attribute
            it writes but ``unit_ind``, which is checked and kept: re-posing a posed block
            would lose the operator that maps the deposited atoms onto it), a carried
            ``unit_ind`` (or ``mol_ind``) that is not one integer entry per atom, or not a
            batch from ``read`` / ``from_arrays``.

        Notes
        -----
        Status codes: ``degenerate_axes`` (a unit with fewer than three heavy atoms, a
        relative moment gap below ``degenerate_tol``, or a degenerate frame);
        ``pose_failed`` (no asymmetric-unit box for the space group, or processing the
        row raised a ``RuntimeError`` or ``ValueError``).

        The frame's axis signs and handedness are those of the float32 copy of the chosen
        image, which is what the unit-cell builder standardizes (it runs in float32
        only); the axes themselves and the rotation vector are float64. Where the signs
        are set by rounding (``handedness_unstable``), a float64 frame would differ from
        the one every float32 consumer builds.
        """
        _check_input_batch(self)
        if handedness != 'measured':
            raise ValueError(f"handedness must be 'measured' (forcing a handedness is excluded under D2), "
                             f'got {handedness!r}')
        degenerate_tol = _nonnegative_float('degenerate_tol', degenerate_tol)
        handedness_tol = _nonnegative_float('handedness_tol', handedness_tol)
        _refuse_second_pass(self, DONE_POSE, _POSE_KEYS, 'parameterize_pose',
                            'a posed block re-posed loses the operator that maps the deposited atoms onto it')
        rows = _ok_rows(self)
        _require_done(self, rows, DONE_FIND_MOLECULES, 'parameterize_pose', 'find_molecules')

        B, ptr = self.num_graphs, self.ptr.detach().to('cpu').long()
        T_fc64, T_cf64 = _transforms64(self)
        z_all = self.z.detach().to('cpu').long()
        pos64 = self.pos.detach().to('cpu', _F64)
        zp_all = self['z_prime'].detach().to('cpu').long() if 'z_prime' in self._store else \
            torch.zeros(B, dtype=torch.long)
        sg_all = self.sg_ind.detach().to('cpu').long()

        # the units to pose: z_prime equal contiguous blocks, which must be the units find_molecules found (mol_ind),
        # or the map the batch carries; an ok row whose units are other is not posed as if they were
        counts, N = ptr[1:] - ptr[:-1], int(ptr[-1])
        atom_row = _atom_rows(ptr)
        blocks = equal_block_unit_ind(counts, zp_all, total=N)
        fits = (zp_all >= 1) & (counts % zp_all.clamp(min=1) == 0)
        carried = UNIT_MAP_KEY in self._store
        source = UNIT_MAP_KEY if carried else ('mol_ind' if 'mol_ind' in self._store else None)
        if source is not None:
            units = self[source].detach().to('cpu')
            problem = unit_map_problem(units, N)
            if problem is not None:
                raise ValueError(f'parameterize_pose: {source!r} cannot be read as the units of the atoms: {problem}')
            other_units = torch.zeros(B, dtype=torch.long).index_add_(0, atom_row, (units.long() != blocks).long()) > 0
        else:
            other_units = torch.zeros(B, dtype=torch.bool)
        refused = {}
        for b in rows:
            if bool(fits[b]) and bool(other_units[b]):
                s, e = int(ptr[b]), int(ptr[b + 1])
                sizes = torch.bincount(units[s:e].long().clamp(min=0)).tolist()
                refused[b] = _Refusal('pose_failed', f"its units ({source}: {sizes} atoms) are not z_prime "
                                                     f"{int(zp_all[b])} equal contiguous blocks in unit order, the "
                                                     f"units this step poses")
        rows = [b for b in rows if b not in refused]

        def run(idx):
            # each row's units from the unit map (the one reader; a row whose units cannot be read raises here, and
            # _isolate gives it pose_failed): equal contiguous blocks at this stage
            per_unit = unit_layout(self, 'parameterize_pose', rows=idx).atoms_per_unit.detach().to('cpu')
            return _pose_rows(idx, ptr, z_all, pos64, zp_all, sg_all, T_fc64, T_cf64, self.symmetry_operators,
                              degenerate_tol, handedness_tol, per_unit)

        results, failed = _isolate(run, rows)
        for b, msg in failed.items():
            results[b] = _Refusal('pose_failed', msg)
        results.update(refused)

        out = _copy_batch(self)
        device, dtype = self.pos.device, self.pos.dtype
        posed = [b for b, r in results.items() if not isinstance(r, _Refusal)]
        zmax = max([int(zp_all[b]) for b in posed], default=1)
        centroid = torch.zeros(B, zmax, 3, dtype=_F64)
        orientation = torch.zeros(B, zmax, 3, dtype=_F64)
        hand = torch.zeros(B, zmax, dtype=torch.long)
        op = torch.full((B, zmax), -1, dtype=torch.long)
        well_defined = torch.zeros(B, dtype=torch.bool)
        ambiguous = torch.zeros(B, dtype=torch.bool)
        unstable = torch.zeros(B, dtype=torch.bool)
        new_pos = pos64.clone()
        status = out.status.detach().to('cpu').clone()
        message = list(out.message)
        for b, res in results.items():
            if isinstance(res, _Refusal):
                status[b], message[b] = _STATUS[res.code], str(res)
                continue
            k = int(zp_all[b])
            centroid[b, :k], orientation[b, :k], hand[b, :k], op[b, :k] = (res['centroid'], res['orientation'],
                                                                         res['handedness'], res['op'])
            well_defined[b], ambiguous[b], unstable[b] = res['well_defined'], res['ambiguous'], res['unstable']
            new_pos[int(ptr[b]):int(ptr[b + 1])] = res['pos']
        changed = torch.zeros(len(new_pos), dtype=torch.bool)
        for b in posed:
            changed[int(ptr[b]):int(ptr[b + 1])] = True
        pos = self.pos.clone()
        pos[changed.to(device)] = new_pos[changed].to(dtype).to(device)
        out['pos'] = pos
        _set(out, 'aunit_centroid', centroid.reshape(B, 3 * zmax).to(dtype).to(device), 'row')
        _set(out, 'aunit_orientation', orientation.reshape(B, 3 * zmax).to(dtype).to(device), 'row')
        _set(out, 'aunit_handedness', hand.to(device), 'row')
        _set(out, 'aunit_op', op.to(device), 'row')
        _set(out, 'is_well_defined', well_defined.to(device), 'row')
        _set(out, 'asu_ambiguous', ambiguous.to(device), 'row')
        _set(out, 'handedness_unstable', unstable.to(device), 'row')
        _set(out, 'x', torch.zeros(len(pos), dtype=dtype, device=device), 'atom')
        # the unit map: a carried map (find_molecules writes it since stage 3, sanitize fills it for the hydrogens it
        # places) is kept; a batch written before stage 3 gets find_molecules' units where they are the Z' equal
        # blocks posed, NO_UNIT where the units are not known as such (no row's map contradicts its z_prime or mol_ind)
        if not carried:
            known = fits & ~other_units
            _set(out, UNIT_MAP_KEY, torch.where(known[atom_row], blocks, torch.full_like(blocks, NO_UNIT)).to(device),
                 'atom')
        _set(out, 'max_z_prime', zmax, 'batch')
        _finish(out, self, status, message, results, DONE_POSE, 'parameterize_pose',
                dict(handedness=handedness, degenerate_tol=degenerate_tol, handedness_tol=handedness_tol))
        return out

    def check_rebuild(self, *, max_dev: float = 0.05, mean_dev: float = 0.01, match_elements: bool = True):
        """Build the unit cell from the pose and match it to the deposited set expanded by the operators.

        The posed ok rows are rebuilt with ``mol2ucell`` (std_orientation True, as the
        energies build them) on a float32 copy: the builder runs in float32 only
        (``aunit2ucell`` casts the operators to float32). Each built atom is matched to
        the nearest atom of the reference set under periodic images: the deposited set
        (``deposit_frac``, kept by ``find_molecules``) expanded by the row's operators;
        on a batch posed by ``from_arrays`` (no ``deposit_frac``), the posed asymmetric
        unit itself expanded, so the check then verifies that the pose reproduces the
        given coordinates.

        Parameters
        ----------
        max_dev, mean_dev : float
            Angstrom: the largest and the mean built-atom residual a row may have.
        match_elements : bool
            Match each built atom only to reference atoms of its own element. When False,
            to any atom (elements unchecked, the old featurizer's rule).

        Returns
        -------
        MolCrystalDataBatch
            A new batch with ``rebuild_max_dev``, ``rebuild_mean_dev`` [B] float64 (NaN
            where not checked), ``rebuild_dev`` [N] float64 (per posed atom, the largest
            residual over its images; NaN where not checked), and ``label_flip`` [B]
            bool: the geometry matches, but the labels do not: some built atom's nearest
            reference atom is an image of another posed atom than the one it was built
            from (the reference holds the deposited atoms in the posed order); or
            re-parameterizing the built cell (``reparameterize_unit_cell``) gives another
            handedness or places a unit's atoms (index by index) more than ``max_dev``
            from where the pose put them, e.g. another image or a frame turned onto a
            symmetry-equivalent one. The message says which. A second pass re-checks the
            ok rows and leaves these values unchanged on the rows it passes through.

        Raises
        ------
        ValueError
            A threshold that is not a positive finite number, ``match_elements`` not a
            bool, a batch without a pose, or not a batch from ``read`` / ``from_arrays``.

        Notes
        -----
        Status codes, the first that applies: ``unmatched_atoms`` (the built and
        reference sets differ in size, or the nearest-atom match is not one to one);
        ``element_mismatch`` (with ``match_elements``: the match ignoring elements is one
        to one within ``max_dev`` but the element-aware one is not); ``rebuild_mismatch``
        (one to one, but a residual above ``max_dev`` or a mean above ``mean_dev``);
        ``rebuild_failed`` (processing raised a ``RuntimeError`` or ``ValueError``).
        """
        _check_input_batch(self)
        max_dev = _positive_float('max_dev', max_dev)
        mean_dev = _positive_float('mean_dev', mean_dev)
        if not isinstance(match_elements, bool):
            raise ValueError(f'match_elements must be a bool, got {match_elements!r}')
        missing = [k for k in POSE_KEYS if k not in self._store]
        if missing:
            raise ValueError(f'check_rebuild needs a posed batch; this one lacks {missing}')
        rows = _ok_rows(self)

        def run(idx):
            return _rebuild_rows(self, idx, max_dev, mean_dev, match_elements)

        results, failed = _isolate(run, rows)
        for b, msg in failed.items():
            results[b] = _Refusal('rebuild_failed', msg)

        out = _copy_batch(self)
        B, ptr = self.num_graphs, self.ptr.detach().to('cpu').long()
        device = self.pos.device
        # a second pass keeps the values of the rows it passes through; the rows it checks start blank
        max_d = _previous(self, 'rebuild_max_dev', torch.full((B,), float('nan'), dtype=_F64))
        mean_d = _previous(self, 'rebuild_mean_dev', torch.full((B,), float('nan'), dtype=_F64))
        dev = _previous(self, 'rebuild_dev', torch.full((len(self.pos),), float('nan'), dtype=_F64))
        flip = _previous(self, 'label_flip', torch.zeros(B, dtype=torch.bool))
        for b in results:
            max_d[b] = mean_d[b] = float('nan')
            dev[int(ptr[b]):int(ptr[b + 1])] = float('nan')
            flip[b] = False
        status = out.status.detach().to('cpu').clone()
        message = list(out.message)
        for b, res in results.items():
            info = res.info if isinstance(res, _Refusal) else res
            if 'atom_dev' in info:
                max_d[b], mean_d[b] = info['max_dev'], info['mean_dev']
                dev[int(ptr[b]):int(ptr[b + 1])] = torch.as_tensor(info['atom_dev'])
            if isinstance(res, _Refusal):
                status[b], message[b] = _STATUS[res.code], str(res)
            else:
                flip[b] = res['label_flip']
                if res['note'] and res['note'] not in message[b]:     # a second pass does not repeat it
                    message[b] = res['note'] if not message[b] else f"{message[b]}; {res['note']}"
        _set(out, 'rebuild_max_dev', max_d.to(device), 'row')
        _set(out, 'rebuild_mean_dev', mean_d.to(device), 'row')
        _set(out, 'rebuild_dev', dev.to(device), 'atom')
        _set(out, 'label_flip', flip.to(device), 'row')
        _finish(out, self, status, message, results, DONE_REBUILD, 'check_rebuild',
                dict(max_dev=max_dev, mean_dev=mean_dev, match_elements=match_elements))
        return out

    def check_validity(self, *, max_atomic_number: Optional[int] = None, max_heavy_atoms: Optional[int] = None,
                       heavy_nn_range: Optional[Tuple[float, float]] = None, occupancy: Optional[float] = None,
                       disorder: Optional[bool] = None, cell_constraints: Optional[float] = None,
                       min_contact: Optional[float] = None, composition: Optional[str] = None):
        """Apply named validity thresholds; each None is not applied.

        The criteria of the old featurizer's ``crystal_filter``
        (``featurization_utils.py:273-401``) as explicit thresholds, plus ``min_contact``
        and ``composition``. Each criterion runs on the ok rows that hold its inputs;
        ``validity_checked`` [B] int16 records, bit ``i`` for ``VALIDITY_CRITERIA[i]``,
        which ran on each row.

        Parameters
        ----------
        max_atomic_number : int, optional
            Every atom's atomic number at most this (the old "Too many atoms", which
            tests the atomic number). Any state.
        max_heavy_atoms : int, optional
            Every unit at most this many heavy atoms (Z > 1). Rows with units: through
            ``find_molecules``, or posed by ``from_arrays``.
        heavy_nn_range : (float, float), optional
            Angstrom: every heavy atom's nearest heavy neighbour in its own unit within
            [lo, hi] (old: 0.9, 3.0). A unit with one heavy atom has no neighbour and
            fails, as in the old filter. Rows with units, as above.
        occupancy : float, optional
            Every site's ``|occupancy - 1|`` at most this. Needs ``occupancy`` (kept by
            ``read``). ``read`` already gives sites off 1 by more than 1e-3 the read
            status ``partial_occupancy``, which no step processes.
        disorder : bool, optional
            True: refuse rows flagged disordered. Needs a per-row bool ``has_disorder``,
            which no reader writes yet (CSD exports carry no disorder tags; the database
            flag belongs to the provenance side table, not built).
        cell_constraints : float, optional
            The cell's metric tensor G must be invariant under every operator's rotation
            W (fractional): ``max |W^T G W - G| / max |G|`` at most this. Holds in any
            setting (it replaces the old exact angle tests, which assumed the standard
            unique axis). Any state.
        min_contact : float, optional
            Angstrom: no two atoms of the unit cell (the deposited set, or the posed
            asymmetric unit without one, expanded by the operators; periodic images
            included) closer than this. Superposed atoms: the disorder guard for files
            without disorder tags. Any state.
        composition : {'heavy', 'all'}, optional
            The deposited set's element counts must equal the moiety formula's
            (``_chemical_formula_moiety``, each moiety times its coefficient) times
            ``z_value / n_ops`` (n_ops: the row's operators): 'all' for every element,
            'heavy' for every element but hydrogen (H that a later step regenerates).
            It catches a deposited set that omits a molecule (e.g. a solvent) or
            hydrogens, which every structural step accepts. Runs on rows with a moiety
            that parses to at least one unit and ``z_value`` >= 1 (D and T count as H).
            Any state.

        Returns
        -------
        MolCrystalDataBatch
            A new batch with ``validity_checked``. A failing row gets the code of its
            first failing criterion in ``VALIDITY_CRITERIA`` order; its message names
            every criterion it failed.

        Raises
        ------
        ValueError
            A threshold of the wrong type or range; a criterion requested whose inputs no
            ok row holds (a check that cannot run must not read as passed); or not a
            batch from ``read`` / ``from_arrays``.
        """
        _check_input_batch(self)
        crit = {}
        if max_atomic_number is not None:
            crit['max_atomic_number'] = _positive_int('max_atomic_number', max_atomic_number)
        if max_heavy_atoms is not None:
            crit['max_heavy_atoms'] = _positive_int('max_heavy_atoms', max_heavy_atoms)
        if heavy_nn_range is not None:
            if not (isinstance(heavy_nn_range, (list, tuple)) and len(heavy_nn_range) == 2):
                raise ValueError(f'heavy_nn_range must be (lo, hi), got {heavy_nn_range!r}')
            lo = _nonnegative_float('heavy_nn_range[0]', heavy_nn_range[0])
            hi = _positive_float('heavy_nn_range[1]', heavy_nn_range[1])
            if lo > hi:
                raise ValueError(f'heavy_nn_range must have lo <= hi, got {heavy_nn_range!r}')
            crit['heavy_nn_range'] = [lo, hi]
        if occupancy is not None:
            crit['occupancy'] = _nonnegative_float('occupancy', occupancy)
        if disorder is not None:
            if not isinstance(disorder, bool):
                raise ValueError(f'disorder must be None or a bool, got {disorder!r}')
            if disorder:
                crit['disorder'] = True
        if cell_constraints is not None:
            crit['cell_constraints'] = _nonnegative_float('cell_constraints', cell_constraints)
        if min_contact is not None:
            crit['min_contact'] = _positive_float('min_contact', min_contact)
        if composition is not None:
            if composition not in ('heavy', 'all'):
                raise ValueError(f"composition must be None, 'heavy' or 'all', got {composition!r}")
            crit['composition'] = composition

        rows = _ok_rows(self)
        # rows with units: through find_molecules (mol_ind), posed by from_arrays (Z' equal blocks), or carrying the
        # unit map (since the stage-2 review: a batch with unit_ind and no mol_ind has units); unit sizes are read
        # through unit_layout below
        if 'z_prime' in self._store and (UNIT_MAP_KEY in self._store or 'mol_ind' in self._store):
            zp = self['z_prime'].detach().to('cpu').long()
            found = [b for b in rows if int(zp[b]) >= 1]
        else:
            found = []
        # rows with a declared composition: a moiety that parses, and Z
        expected = {}
        if 'composition' in crit and 'moiety' in self._store and 'z_value' in self._store:
            for b in rows:
                want = _moiety_counts(self['moiety'][b], int(self['z_value'][b]), len(self.symmetry_operators[b]))
                if want is not None:
                    expected[b] = want
        needs = dict(max_heavy_atoms=(found, 'it needs units (find_molecules, or a pose)'),
                     heavy_nn_range=(found, 'it needs units (find_molecules, or a pose)'),
                     occupancy=(rows if 'occupancy' in self._store else [], 'the batch holds no occupancy'),
                     disorder=(rows if 'has_disorder' in self._store else [], 'the batch holds no has_disorder'),
                     composition=(list(expected), 'no ok row holds a moiety that parses and a z_value'))
        for name in crit:
            if name in needs and rows and not needs[name][0]:
                raise ValueError(f'check_validity({name}=...) cannot run on any ok row of this batch: '
                                 f'{needs[name][1]}')
        found_set = set(found)

        def run(idx):
            # the units of the rows that have them, from the unit map (the one reader; equal contiguous blocks at
            # this stage): a row whose units cannot be read raises here, and _isolate gives it validity_failed
            unit_rows = [b for b in idx if b in found_set]
            per_unit = unit_layout(self, 'check_validity', rows=unit_rows).atoms_per_unit.tolist() if unit_rows \
                else None
            return {b: _validity_row(self, b, crit, b in found_set, expected.get(b),
                                     per_unit[b] if b in found_set else None) for b in idx}

        results, failed = _isolate(run, rows)
        for b, msg in failed.items():
            results[b] = _Refusal('validity_failed', msg)

        out = _copy_batch(self)
        checked = torch.zeros(self.num_graphs, dtype=torch.int16)
        status = out.status.detach().to('cpu').clone()
        message = list(out.message)
        for b, res in results.items():
            if isinstance(res, _Refusal):
                status[b], message[b] = _STATUS[res.code], str(res)
                continue
            checked[b] = res['checked']
            if res['failed']:
                first = res['failed'][0]
                status[b] = _STATUS[_VALIDITY_CODES[first]]
                message[b] = '; '.join(res['messages'])
        if 'validity_checked' in self._store:      # a second pass adds to the record of the first
            checked = checked | self['validity_checked'].detach().to('cpu').to(torch.int16)
        _set(out, 'validity_checked', checked.to(self.pos.device), 'row')
        _finish(out, self, status, message, results, DONE_VALIDITY, 'check_validity', crit)
        return out


###############################################################################
# shared machinery
###############################################################################

def _check_input_batch(batch):
    """Raise ValueError unless ``batch`` is a batch with the ingestion bookkeeping and codes in the status table.

    The table is ``crystal_store.status_table()``: this module's ``STATUS`` extended by the
    later chain modules' codes (``crystal_chem``'s), so a row refused by a later step passes
    through these steps like any other non-ok row.
    """
    from mxtaltools.dataset_utils.data_class_methods.crystal_store import status_table  # it imports this module
    if not getattr(batch, 'is_batch', False):
        raise ValueError(f'the steps take a batch from MolCrystalData.read / from_arrays, not a {type(batch).__name__}')
    missing = [k for k in ('ptr', 'batch', 'status', 'message', 'done', 'z', 'pos', 'cell_lengths', 'cell_angles',
                           'sg_ind', 'symmetry_operators') if k not in batch._store]
    if missing or '_slice_dict' not in batch.__dict__:
        raise ValueError(f'the steps take a batch from MolCrystalData.read / from_arrays; this one lacks '
                         f'{missing or ["_slice_dict"]}')
    n_codes = len(status_table())
    bad = sorted({int(s) for s in batch.status.detach().to('cpu').tolist() if not 0 <= int(s) < n_codes})
    if bad:
        raise ValueError(f'status codes {bad} are outside the status table (crystal_store.status_table())')


def _positive_int(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive int, got {value!r}')
    return int(value)


def _positive_float(name, value) -> float:
    v = _nonnegative_float(name, value)
    if v <= 0:
        raise ValueError(f'{name} must be positive, got {value!r}')
    return v


def _nonnegative_float(name, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
            or not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite number >= 0, got {value!r}')
    return float(value)


def _bond_scales(bond_scale) -> Tuple[float, float]:
    """(heavy, hydrogen) covalent-radius scales from one number or a pair."""
    if isinstance(bond_scale, (list, tuple)):
        if len(bond_scale) != 2:
            raise ValueError(f'bond_scale must be one number or (heavy, hydrogen), got {bond_scale!r}')
        return _positive_float('bond_scale[0]', bond_scale[0]), _positive_float('bond_scale[1]', bond_scale[1])
    s = _positive_float('bond_scale', bond_scale)
    return s, s


def _ok_rows(batch) -> list:
    return torch.nonzero(batch.status.detach().to('cpu') == _OK).reshape(-1).tolist()


def _require_done(batch, rows, bit, step, needed):
    done = batch.done.detach().to('cpu').long()
    missing = [b for b in rows if not int(done[b]) & bit]
    if missing:
        raise ValueError(f'{step} needs {needed} on every ok row; rows {missing[:10]} have not been through it')


def _refuse_second_pass(batch, bit, keys, step, why):
    """Raise ValueError when any row (whatever its status) has ``bit`` in ``done``, or the batch holds one of ``keys``.
    """
    done = batch.done.detach().to('cpu').long()
    again = [b for b in range(len(done)) if int(done[b]) & bit]
    held = [k for k in keys if k in batch._store]
    if again or held:
        raise ValueError(f'{step} runs once per batch: '
                         + (f'rows {again[:10]} have been through it already' if again else f'the batch holds {held}')
                         + f' ({why})')


def _previous(batch, key, default):
    """The input's values of ``key`` on the host, as ``default``'s dtype and shape, else ``default``."""
    if key in batch._store and torch.is_tensor(batch[key]) and batch[key].shape == default.shape:
        return batch[key].detach().to('cpu').to(default.dtype).clone()
    return default


def _atom_rows(ptr) -> torch.Tensor:
    return torch.arange(len(ptr) - 1).repeat_interleave(ptr[1:] - ptr[:-1])


def _transforms64(batch):
    """``(T_fc, T_cf)`` [B, 3, 3] float64 on the host, from the stored cell."""
    lengths = batch.cell_lengths.detach().to('cpu', _F64)
    angles = batch.cell_angles.detach().to('cpu', _F64)
    T_fc, T_cf, _ = batch_compute_fractional_transform(lengths, angles)
    return T_fc, T_cf


def _isolate(fn: Callable, rows) -> Tuple[Dict[int, object], Dict[int, str]]:
    """Run ``fn(rows) -> {row: result}``; where it raises ``RuntimeError`` or ``ValueError``,
    split the rows in half and retry, down to single rows (as ``crystal_fd.canon._isolate``).

    Returns ``(results, failed)``: per row its result, or (for a row that raises on its
    own) its message. Any other exception propagates. Every split halves a list of at
    least two rows and a single row is never split, so this ends after at most 2k - 1
    calls for k rows.
    """
    results, failed = {}, {}
    stack = [list(rows)] if len(rows) else []
    while stack:
        idx = stack.pop()
        try:
            results.update(fn(idx))
        except (RuntimeError, ValueError) as e:     # torch's LinAlgError is a RuntimeError
            if len(idx) == 1:
                failed[idx[0]] = f'{type(e).__name__}: {str(e)[:300]}'
            else:
                half = len(idx) // 2
                stack += [idx[half:], idx[:half]]
    return results, failed


def _copy_batch(batch):
    """A new batch equal to ``batch``: tensors cloned, other values deep-copied, bookkeeping copied."""
    out = type(batch).__new__(type(batch))
    out.__dict__['_store'] = GlobalStorage(_parent=out)
    for key, value in batch._store.items():
        out[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
    out.__dict__['_num_graphs'] = batch.__dict__.get('_num_graphs')
    for name in ('_slice_dict', '_inc_dict'):
        entries = batch.__dict__.get(name)
        out.__dict__[name] = None if entries is None else {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in entries.items()}
    return out


def _set(out, key, value, level):
    """Store ``value`` under ``key`` with the slice bookkeeping of its level ('atom', 'row' or 'batch')."""
    out[key] = value
    if level == 'batch':
        out._slice_dict.pop(key, None)
        out._inc_dict.pop(key, None)
        return
    ptr = out.ptr.detach().to('cpu').long()
    B = len(ptr) - 1
    out._slice_dict[key] = ptr.clone() if level == 'atom' else torch.arange(B + 1, dtype=torch.long)
    out._inc_dict[key] = torch.zeros(B, dtype=torch.long) if torch.is_tensor(value) else None


def _finish(out, inp, status, message, results, bit, step, args):
    """Write statuses, messages, the ``done`` bit of the processed rows and the ``steps_run`` record."""
    device = inp.status.device
    out['status'] = status.to(inp.status.dtype).to(device)
    out['message'] = message
    done = inp.done.detach().to('cpu').clone()
    for b in results:
        done[b] = int(done[b]) | bit
    out['done'] = done.to(inp.done.dtype).to(device)
    previous = inp['steps_run'] if 'steps_run' in inp._store else []
    _set(out, 'steps_run', copy.deepcopy(list(previous)) + [dict(step=step, args=args)], 'batch')


def _permute_atoms(out, perm, ptr, rows):
    """Reorder atoms within rows by ``perm`` [N] (global indices; identity outside ``rows``).

    Every per-atom tensor (sliced by ``ptr``) is gathered; a row-level list whose entries
    are lists or tuples of the row's atom count is reordered on ``rows``; ``file_bonds``
    is renumbered on ``rows``.
    """
    N = len(perm)
    if torch.equal(perm, torch.arange(N)):
        return
    B = len(ptr) - 1
    counts = (ptr[1:] - ptr[:-1]).tolist()
    row_slices = torch.arange(B + 1)
    for key in list(out._store.keys()):
        slices = out._slice_dict.get(key)
        value = out[key]
        if slices is None:
            continue
        slices = slices.to('cpu')
        if torch.is_tensor(value) and torch.equal(slices, ptr) and value.shape[:1] == (N,):
            if B == N and torch.equal(slices, row_slices):
                continue                            # one atom per row: no reordering is possible
            out[key] = value[perm.to(value.device)]
        elif isinstance(value, list) and torch.equal(slices, row_slices) and key != 'file_bonds':
            if all(isinstance(v, (list, tuple)) and len(v) == counts[b] for b, v in enumerate(value)) and \
                    any(counts[b] > 1 for b in rows):
                for b in rows:
                    s = int(ptr[b])
                    local = (perm[s:s + counts[b]] - s).tolist()
                    value[b] = type(value[b])(value[b][i] for i in local)
    if 'file_bonds' in out._store:
        fb = out['file_bonds']
        for b in rows:
            if fb[b] is None:
                continue
            s = int(ptr[b])
            local = (perm[s:s + counts[b]] - s).numpy()
            inverse = np.empty_like(local)
            inverse[local] = np.arange(len(local))
            fb[b] = inverse[fb[b]].astype(np.int64)


def _write_unit_properties(out, ptr, z_prime, mol_ind, dtype, device):
    """Per-unit radius, mass and volume [B, Z'max] (NaN past Z') and their row sums [B] (NaN on rows without units).

    ``MolData.mol_analysis``'s definitions, through the same batch functions
    ``from_arrays`` uses: heavy-atom radius about the heavy-atom centroid, atomic-weight
    mass, overlap-corrected van der Waals volume.
    """
    B = len(ptr) - 1
    zmax = max(int(z_prime.max()) if B else 1, 1)
    atom_row = _atom_rows(ptr)
    units_per_row = z_prime
    unit_ptr = torch.cat([units_per_row.new_zeros(1), units_per_row.cumsum(0)])
    n_units = int(unit_ptr[-1])
    has_units = (units_per_row > 0)[atom_row]
    unit_of_atom = (unit_ptr[:-1][atom_row] + mol_ind)[has_units]
    zz = out.z.detach().to('cpu')[has_units]
    pp = out.pos.detach().to('cpu', _F64)[has_units]
    if n_units:
        heavy = zz > 1
        heavies = scatter(heavy.long(), unit_of_atom, dim=0, dim_size=n_units, reduce='sum')
        radius_u = batch_compute_mol_radius(pp[heavy], unit_of_atom[heavy], n_units, heavies)
        masses = torch.tensor(list(ATOM_WEIGHTS.values()), dtype=torch.float32)
        mass_u = batch_compute_mol_mass(zz, unit_of_atom, masses, n_units)
        vdw = torch.tensor(list(VDW_RADII.values()), dtype=torch.float32)
        volume_u = batch_compute_molecule_volume(zz, pp, unit_of_atom, n_units, vdw)
    else:
        radius_u = mass_u = volume_u = torch.zeros(0, dtype=_F64)
    unit_row = torch.arange(B).repeat_interleave(units_per_row)
    slot = torch.arange(n_units) - unit_ptr[:-1][unit_row]
    for unit_key, row_key, values in (('unit_radius', 'radius', radius_u), ('unit_mass', 'mass', mass_u),
                                      ('unit_volume', 'mol_volume', volume_u)):
        per_unit = torch.full((B, zmax), float('nan'), dtype=_F64)
        per_unit[unit_row, slot] = values.to(_F64)
        summed = scatter(values.to(_F64), unit_row, dim=0, dim_size=B, reduce='sum')
        summed = torch.where(units_per_row > 0, summed, torch.full_like(summed, float('nan')))
        _set(out, unit_key, per_unit.to(dtype).to(device), 'row')
        _set(out, row_key, summed.to(dtype).to(device), 'row')


###############################################################################
# find_molecules
###############################################################################

def _find_row(z, frac, T_fc, ops, file_bonds, moiety, z_value, *, bonds, scales, iso_cap, special_tol) -> dict:
    """One row's molecules; raises ``_Refusal`` for a row that fails (see ``find_molecules``)."""
    n = len(z)
    ops = np.asarray(ops, dtype=np.float64)
    if file_bonds is not None:
        _check_file_bonds(file_bonds, n)
    if n == 0:
        raise _Refusal('no_bonds', 'the row holds no atoms')
    if isinstance(moiety, str) and _POLYMER_RE.search(moiety):
        raise _Refusal('polymer', f'the moiety formula {moiety!r} declares a polymer')
    radii = covalent_radii[z] * np.where(z == 1, scales[1], scales[0])

    # 1. bonds within the deposited set, molecules made whole across lattice translations
    if bonds == 'file':
        if file_bonds is None:
            raise _Refusal('no_bonds', 'the file has no _geom_bond loop')
        edges = [tuple(e) for e in np.asarray(file_bonds).T.tolist()]
        if n > 1 and not edges:
            raise _Refusal('no_bonds', "the file's bond loop holds no bond within the deposited set")
        comp = union_find(n, edges)
        x, _ = unwrap_components(frac, edges, n)
        h_listed = bool((z[np.asarray(file_bonds).reshape(-1)] == 1).any())
    else:
        edges, comp, x = _distance_components(z, frac, T_fc, radii)
        if n > 1 and not edges:
            raise _Refusal('no_bonds', 'no two atoms are within bonding distance', components=comp)
        h_listed = True
    n_comp = int(comp.max()) + 1

    # 2. contacts with symmetry images: special positions, bonds the file does not list,
    # molecules completed or chained by symmetry
    _symmetry_contacts(z, x, T_fc, ops, comp, radii, special_tol, h_listed)

    # 3. units: every component the same molecule, re-indexed onto unit 0
    members = [np.nonzero(comp == k)[0] for k in range(n_comp)]
    graph_of = _component_graphs(z, members, edges)
    cart = x @ T_fc.T
    sizes = {len(m) for m in members}
    if len(sizes) > 1 or any(not np.array_equal(np.sort(z[m]), np.sort(z[members[0]])) for m in members[1:]):
        raise _Refusal('multi_component', f'{n_comp} components of different composition '
                                          f'(atom counts {[len(m) for m in members]})', components=comp)
    perms, order_problem = [np.arange(len(members[0]))], None
    for k in range(1, n_comp):
        perm, problem = _match_unit(z[members[0]], cart[members[0]], graph_of[0],
                                    z[members[k]], cart[members[k]], graph_of[k], iso_cap)
        if problem is not None:
            code, text = problem
            if code == 'multi_component':
                raise _Refusal('multi_component', f'unit {k}: {text}', components=comp)
            order_problem = order_problem or f'unit {k}: {text}'
            perm = np.arange(len(members[0]))
        perms.append(perm)
    n_ops = len(ops)
    if z_value >= 1 and z_value != n_comp * n_ops:
        raise _Refusal('z_prime_mismatch', f"z_value {z_value} is not Z' {n_comp} x {n_ops} operators "
                                           f"(the file implies Z' = {z_value / n_ops:g})", components=comp)
    if order_problem is not None:
        raise _Refusal('unit_order_failed', order_problem, components=comp)

    order = np.concatenate([members[k][perms[k]] for k in range(n_comp)])
    new_index = np.empty(n, dtype=np.int64)
    new_index[order] = np.arange(n)
    e = np.array(edges, dtype=np.int64).reshape(-1, 2)
    mol_bonds = np.sort(new_index[e], axis=1)
    mol_bonds = mol_bonds[np.lexsort((mol_bonds[:, 1], mol_bonds[:, 0]))].T.copy()
    note = ''
    moved = int((np.abs(x - frac) > 0.5).any(1).sum())
    if moved:
        note = f'{moved} atom(s) moved by a lattice vector to make their molecule whole'
    return dict(order=order, frac=x, unit=np.repeat(np.arange(n_comp), len(members[0])), z_prime=n_comp,
                bonds=mol_bonds, note=note)


def _check_file_bonds(file_bonds, n):
    """Raise ``_Refusal('find_molecules_failed')`` unless ``file_bonds`` is an integer [2, E] array of indices in [0, n)
    with no bond from an atom to itself (``from_arrays`` takes any value; ``read`` writes only such arrays)."""
    try:
        a = np.asarray(file_bonds)
    except ValueError as e:                     # a ragged nested list
        raise _Refusal('find_molecules_failed', f'file_bonds is not an array: {e}')
    if a.dtype.kind not in 'iu' or a.ndim != 2 or a.shape[0] != 2:
        raise _Refusal('find_molecules_failed', f'file_bonds must be an integer [2, E] array, got dtype {a.dtype}, '
                                                f'shape {list(a.shape)}')
    if a.size and (int(a.min()) < 0 or int(a.max()) >= n):
        raise _Refusal('find_molecules_failed', f'file_bonds holds atom indices outside [0, {n}) '
                                                f'(from {int(a.min())} to {int(a.max())})')
    if (a[0] == a[1]).any():
        raise _Refusal('find_molecules_failed', f'file_bonds bonds atom {int(a[0][a[0] == a[1]][0])} to itself')


def _offset_range(T_fc, cutoff: float) -> np.ndarray:
    """Lattice offsets [k, 3] covering every image within ``cutoff`` (Angstrom) of a point, given the nearest.

    With fractional differences first reduced to [-0.5, 0.5], a vector of length at most
    ``cutoff`` has |component k| <= cutoff / d_k (d_k the spacing of lattice planes k), so
    offsets up to ``floor(cutoff / d_k + 0.5)`` along axis k suffice.
    """
    T_cf = np.linalg.inv(T_fc)
    reach = np.floor(cutoff * np.linalg.norm(T_cf, axis=1) + 0.5).astype(int)
    grids = [np.arange(-r, r + 1) for r in reach]
    return np.stack(np.meshgrid(*grids, indexing='ij'), -1).reshape(-1, 3)


def _contacts(xa, xb, T_fc, cutoff, *, exclude_self: bool):
    """All periodic contacts ``|x_b + t - x_a| < cutoff[a, b]`` (Angstrom).

    ``xa`` [na, 3], ``xb`` [nb, 3] fractional; ``cutoff`` [na, nb], or one number for
    every pair (no pair matrix is formed). Returns arrays ``a``, ``b``, ``t`` [k, 3]
    (integer lattice translation of b) and ``d`` [k]. With ``exclude_self`` (xa is xb),
    the zero translation of an atom to itself is left out. Blocks over ``a`` keep memory
    bounded.
    """
    out_a, out_b, out_t, out_d = [], [], [], []
    cutoff = np.asarray(cutoff, dtype=np.float64)
    scalar = cutoff.ndim == 0
    offsets = _offset_range(T_fc, float(cutoff.max()) if cutoff.size else 0.0)
    nb = len(xb)
    block = max(1, _PAIR_BLOCK // max(1, nb * len(offsets)))
    for s in range(0, len(xa), block):
        diff = xb[None, :, :] - xa[s:s + block, None, :]            # [a, b, 3]
        base = np.round(diff)
        red = diff - base
        cart = (red[:, :, None, :] + offsets[None, None]) @ T_fc.T   # [a, b, o, 3]
        d = np.linalg.norm(cart, axis=-1)
        hit = d < (cutoff if scalar else cutoff[s:s + block, :, None])
        if exclude_self:
            ai = np.arange(s, min(s + block, len(xa)))
            zero = np.nonzero((offsets == 0).all(1))[0][0]
            hit[ai - s, ai, zero] = False
        ia, ib, io = np.nonzero(hit)
        out_a.append(ia + s)
        out_b.append(ib)
        out_t.append((offsets[io] - base[ia, ib]).astype(np.int64))
        out_d.append(d[ia, ib, io])
    if not out_a:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), np.int64), np.zeros(0)
    return np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_t), np.concatenate(out_d)


def _distance_components(z, frac, T_fc, radii):
    """``(edges, comp, x)``: covalent bonds under periodic boundaries, components, unwrapped coordinates.

    Port of ``split_molecules`` (GFN ``eval/day_bench/p1_ingest.py:85-141``) to the
    deposited set: a depth-first walk from each component's lowest atom, which keeps its
    deposited image; each atom reached is placed at the image its bond names. A
    component that reaches an atom at two images bonds to its own lattice translate:
    ``polymer``.
    """
    n = len(z)
    cutoff = radii[:, None] + radii[None, :]
    a, b, t, _ = _contacts(frac, frac, T_fc, cutoff, exclude_self=True)
    self_bond = a == b
    if self_bond.any():
        raise _Refusal('polymer', f'atom {int(a[self_bond][0])} bonds to its own lattice translate '
                                  f'{t[self_bond][0].tolist()}')
    keep = a < b                                # each bond once (the pair search is symmetric)
    a, b, t = a[keep], b[keep], t[keep]
    adj = [[] for _ in range(n)]
    for i, j, tt in zip(a.tolist(), b.tolist(), t):
        adj[i].append((j, tt))
        adj[j].append((i, -tt))
    shift = np.zeros((n, 3), dtype=np.int64)
    comp = np.full(n, -1, dtype=np.int64)
    n_comp = 0
    for root in range(n):                       # every atom is visited once: bounded by n
        if comp[root] >= 0:
            continue
        comp[root] = n_comp
        stack = [root]
        while stack:
            i = stack.pop()
            for j, tt in adj[i]:
                s = shift[i] + tt
                if comp[j] < 0:
                    comp[j], shift[j] = n_comp, s
                    stack.append(j)
                elif not np.array_equal(shift[j], s):
                    raise _Refusal('polymer', f'the molecule of atom {root} bonds to its own lattice translate '
                                              f'{(s - shift[j]).tolist()}')
        n_comp += 1
    edges = sorted({(min(i, j), max(i, j)) for i, j in zip(a.tolist(), b.tolist())})
    return edges, comp, frac + shift


def _is_identity(op) -> bool:
    return np.allclose(op[:3, :3], np.eye(3), atol=1e-6) and np.allclose(op[:3, 3], np.round(op[:3, 3]), atol=1e-6)


def _symmetry_contacts(z, x, T_fc, ops, comp, radii, special_tol, h_listed=True):
    """Raise the first of ``special_position``, ``unlisted_bond``, ``polymer``, ``z_prime_fractional``,
    ``split_molecule`` found among contacts between the deposited set ``x`` (unwrapped) and
    its images under every operator and lattice translation.

    The identity's zero translation is the deposited set itself: there, a contact between
    two components at least ``special_tol`` apart is a bond the bond list omits
    (``unlisted_bond``; with distance bonds none exists, since such atoms share a
    component); nearer pairs are superposed atoms, ``check_validity(min_contact=...)``'s
    concern. ``h_listed`` False (no listed bond touches hydrogen) names the likely cause.
    """
    cutoff = np.maximum(radii[:, None] + radii[None, :], special_tol)
    coincident, self_maps, split, unlisted = [], {}, [], []
    for k, op in enumerate(ops):
        R, tau = op[:3, :3], op[:3, 3]
        y = x @ R.T + tau
        a, b, t, d = _contacts(x, y, T_fc, cutoff, exclude_self=_is_identity(op) and np.allclose(tau, 0))
        if _is_identity(op):
            # the identity's images are lattice translates of the deposited set itself
            t = t + np.round(tau).astype(np.int64)
            inside = (t == 0).all(1)
            gap = inside & (comp[a] != comp[b]) & (d >= special_tol) & (a < b)
            unlisted += list(zip(a[gap].tolist(), b[gap].tolist(), d[gap].tolist()))
            keep = ~inside
            a, b, t, d = a[keep], b[keep], t[keep], d[keep]
        for i, j, tt, dd in zip(a.tolist(), b.tolist(), t, d.tolist()):
            if dd < special_tol:
                coincident.append((i, j, k, dd))
                continue
            ci, cj = int(comp[i]), int(comp[j])
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = R, tau + tt
            if ci == cj:
                self_maps.setdefault(ci, {})[tuple(np.round(M[:3], 6).ravel())] = (M, k)
            else:
                split.append((ci, cj, k))
    if coincident:
        i, j, k, dd = coincident[0]
        raise _Refusal('special_position', f'{len(coincident)} image(s) coincide with deposited atoms, e.g. atom {j} '
                                           f'under operator {k} lies {dd:.3g} A from atom {i}', components=comp)
    if unlisted:
        i, j, dd = unlisted[0]
        hint = '; the bond loop lists no bond to hydrogen' if not h_listed and (z[i] == 1 or z[j] == 1) else ''
        raise _Refusal('unlisted_bond', f'{len(unlisted)} pair(s) of atoms in different components lie within bonding '
                                        f'distance in the deposited set, e.g. atoms {i} and {j} at {dd:.3g} A, but the '
                                        f"file's bond loop does not bond them{hint} (bonds='distance' would)",
                       components=comp)
    fractional = []
    for c, maps in sorted(self_maps.items()):
        gens = [M for M, _ in maps.values()]
        order = _closure_order(gens)
        if order is None:
            raise _Refusal('polymer', f'component {c} bonds to its own symmetry image under operator(s) '
                                      f'{sorted({k for _, k in maps.values()})}, a chain without end', components=comp)
        fractional.append((c, order, sorted({k for _, k in maps.values()})))
    if fractional:
        c, order, ks = fractional[0]
        raise _Refusal('z_prime_fractional', f'component {c} is completed by symmetry: it bonds to its own image under '
                                             f'operator(s) {ks} (site symmetry of order {order})', components=comp)
    if split:
        ci, cj, k = split[0]
        raise _Refusal('split_molecule', f'component {ci} bonds to an image of component {cj} under operator {k}: '
                                         'one molecule is split across the deposited set', components=comp)


def _closure_order(gens) -> Optional[int]:
    """Order of the group the affine maps ``gens`` (4 x 4, fractional) generate, or None when infinite.

    Bounded: elements are added until no new one appears, a non-zero pure translation
    appears (infinite), or the group passes ``_MAX_SITE_GROUP`` elements (infinite: a
    finite site-symmetry group is a point group).
    """
    def key(M):
        return tuple(np.round(M[:3], 6).ravel())

    elems = {key(np.eye(4)): np.eye(4)}
    frontier = [np.eye(4)]
    while frontier:
        new = []
        for e in frontier:
            for g in gens:
                p = g @ e
                if np.allclose(p[:3, :3], np.eye(3), atol=1e-6) and not np.allclose(p[:3, 3], 0, atol=1e-6):
                    return None
                kp = key(p)
                if kp not in elems:
                    elems[kp] = p
                    new.append(p)
                    if len(elems) > _MAX_SITE_GROUP:
                        return None
        frontier = new
    return len(elems)


def _component_graphs(z, members, edges):
    """One networkx graph per component, nodes numbered within the component, attribute ``e`` the element."""
    local = {}
    for k, m in enumerate(members):
        for i, atom in enumerate(m.tolist()):
            local[atom] = (k, i)
    graphs = []
    for k, m in enumerate(members):
        g = nx.Graph()
        g.add_nodes_from((i, dict(e=int(z[atom]))) for i, atom in enumerate(m.tolist()))
        graphs.append(g)
    for a, b in edges:
        ka, ia = local[a]
        kb, ib = local[b]
        if ka == kb:
            graphs[ka].add_edge(ia, ib)
    return graphs


def _kabsch(P, Q, improper: bool):
    """``(rmsd, apply)``: the best rigid fit of points P onto Q (rows), optionally after mirroring x."""
    mirror = np.array([-1.0, 1.0, 1.0]) if improper else np.ones(3)
    mp, mq = P.mean(0), Q.mean(0)
    A = (P - mp) * mirror
    U, _, Vt = np.linalg.svd(A.T @ (Q - mq))
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    fit = A @ R.T + mq
    rms = float(np.sqrt(((fit - Q) ** 2).sum(1).mean()))
    return rms, (lambda X: ((X - mp) * mirror) @ R.T + mq)


def _match_unit(z0, p0, g0, zk, pk, gk, iso_cap):
    """``(perm, problem)``: unit k's atoms reordered onto unit 0's (``zk[perm] == z0``), or a problem.

    ``problem`` is None or ``(code, text)`` with code ``multi_component`` (the units are
    different molecules) or ``unit_order_failed``. Port of ``best_perm`` in
    ``order_canon_check.py``, on the step's own bond graphs.
    """
    h0, hk = np.nonzero(z0 > 1)[0], np.nonzero(zk > 1)[0]
    if len(h0) != len(hk):
        return None, ('multi_component', 'different heavy-atom counts')
    if len(h0) == 0:
        return None, ('unit_order_failed', 'no heavy atoms to match on')
    G0 = nx.convert_node_labels_to_integers(g0.subgraph(h0.tolist()), ordering='sorted')
    Gk = nx.convert_node_labels_to_integers(gk.subgraph(hk.tolist()), ordering='sorted')
    # isomorphic graphs have equal bond counts and equal hashes: a difference settles it without a search
    if G0.number_of_edges() != Gk.number_of_edges() or \
            nx.weisfeiler_lehman_graph_hash(G0, node_attr='e', iterations=6) != \
            nx.weisfeiler_lehman_graph_hash(Gk, node_attr='e', iterations=6):
        return None, ('multi_component', 'the heavy-atom graphs are not isomorphic (bond counts or '
                                         'Weisfeiler-Lehman hashes differ)')
    for g in (G0, Gk):
        nx.set_node_attributes(g, {i: i for i in g.nodes}, 'id')
    best, exhausted = _best_isomorphism(G0, Gk, p0[h0], pk[hk], iso_cap, None)
    if best is None and exhausted:
        return None, ('multi_component', 'the heavy-atom graphs are not isomorphic')
    if not exhausted:
        # a large automorphism group (rotors, carbonyl clusters, perfluoroaryl rings) or a hard
        # first match: search again among the geometrically plausible maps
        best = _refine_isomorphism(G0, Gk, p0[h0], z0[h0], pk[hk], zk[hk], iso_cap, best)
        if best is None:
            return None, ('unit_order_failed', f'the isomorphism search reached its bounds (iso_cap = {iso_cap}), '
                                               'also among the geometrically nearest candidates')
    hp, _, apply, _ = best
    perm = np.full(len(z0), -1, dtype=np.int64)
    perm[h0] = hk[hp]
    H0, Hk = np.nonzero(z0 == 1)[0], np.nonzero(zk == 1)[0]
    if len(H0) != len(Hk):
        return None, ('multi_component', 'different hydrogen counts')
    if len(H0):
        par0 = _hydrogen_parents(H0, h0, p0, g0)
        park = _hydrogen_parents(Hk, hk, pk, gk)
        fitted = apply(pk)
        for i in h0.tolist():
            a, b = H0[par0 == i], Hk[park == perm[i]]
            if len(a) != len(b):
                return None, ('multi_component', f'heavy atom {i} carries {len(a)} H in unit 0, {len(b)} in unit k')
            if len(a):
                rr, cc = linear_sum_assignment(np.linalg.norm(p0[a][:, None] - fitted[b][None], axis=-1))
                perm[a[rr]] = b[cc]
        orphans0, orphansk = H0[par0 < 0], Hk[park < 0]
        if len(orphans0) != len(orphansk):
            return None, ('multi_component', 'different numbers of unbonded hydrogens')
        if len(orphans0):
            rr, cc = linear_sum_assignment(np.linalg.norm(p0[orphans0][:, None] - fitted[orphansk][None], axis=-1))
            perm[orphans0[rr]] = orphansk[cc]
    if (perm < 0).any() or len(np.unique(perm)) != len(perm) or not np.array_equal(zk[perm], z0):
        return None, ('unit_order_failed', 'the hydrogen assignment is not one to one')
    # the order must carry unit k's bonds exactly onto unit 0's
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    edges0 = {tuple(sorted(e)) for e in g0.edges()}
    edgesk = {tuple(sorted((int(inverse[a]), int(inverse[b])))) for a, b in gk.edges()}
    if edges0 != edgesk:
        return None, ('unit_order_failed', f'the matched order maps {len(edgesk - edges0)} bond(s) of unit k onto no '
                                           'bond of unit 0')
    return perm, None


def _better(a, b) -> bool:
    """Whether fit ``a`` beats fit ``b`` (tuples ``(hp, rmsd, apply, improper)``): a lower RMSD by
    more than 1e-9 A, or an equal one with a proper fit against an improper one. A symmetric
    molecule has several exact fits; an earlier one is kept on a tie."""
    if a[1] < b[1] - 1e-9:
        return True
    return abs(a[1] - b[1]) <= 1e-9 and not a[3] and b[3]


def _refine_isomorphism(G0, Gk, p0, z0, pk, zk, iso_cap, capped):
    """The best isomorphism among the maps that send each atom near its partner, or None.

    Starting fits of unit k onto unit 0: the capped search's best (``capped``, may be
    None) and the ``_GEOMETRIC_STARTS`` best fits from geometry alone
    (``_geometric_fits``). From each start, the search runs among the maps sending each
    atom to a same-element atom near it after the fit (``_nearby_candidates``), refitting
    on the result while that lowers the RMSD. Rule 'tolerance' from every start first;
    if no start gives a complete search, rule 'nearest' from every start. Of a rule's
    complete searches the lowest RMSD wins, against ``capped`` when that is lower.
    Bounded: at most ``2 * (_GEOMETRIC_STARTS + 1) * _REFINE_ROUNDS`` searches, each
    bounded by ``iso_cap``.
    """
    starts = ([capped] if capped is not None else []) + _geometric_fits(p0, z0, pk, zk)[:_GEOMETRIC_STARTS]
    for rule in ('tolerance', 'nearest'):
        results = []
        for start in starts:
            current, found = start, None
            for _ in range(_REFINE_ROUNDS):
                allowed = _nearby_candidates(p0, z0, current[2](pk), zk, rule)
                refined, done = _best_isomorphism(G0, Gk, p0, pk, iso_cap, allowed)
                if not done or refined is None:
                    break
                if found is not None and not _better(refined, found):
                    break
                found = current = refined       # exhaustive within the geometrically plausible maps
            if found is not None:
                results.append(found)
        if results:
            best = results[0]
            for r in results[1:]:
                best = r if _better(r, best) else best
            return capped if capped is not None and _better(capped, best) else best
    return None


def _geometric_fits(p0, z0, pk, zk):
    """Fits of unit k onto unit 0 from geometry alone, lowest RMSD first, as ``(None, rmsd, apply, improper)``.

    The two units' principal axes (heavy atoms, float64) matched under each of the 8 sign
    choices (proper and improper), each refined ``_REFINE_ROUNDS`` times by a per-element
    assignment (Hungarian, squared distances) and a Kabsch refit on it.
    """
    c0, ck = p0.mean(0), pk.mean(0)
    _, V0 = np.linalg.eigh((p0 - c0).T @ (p0 - c0))
    _, Vk = np.linalg.eigh((pk - ck).T @ (pk - ck))
    fits = []
    for signs in itertools.product((1.0, -1.0), repeat=3):     # 8 frames
        R = V0 @ np.diag(signs) @ Vk.T
        improper = bool(np.linalg.det(R) < 0)
        rms, apply = np.inf, (lambda X, R=R: (X - ck) @ R.T + c0)
        for _ in range(_REFINE_ROUNDS):
            perm = _assign_by_element(p0, z0, apply(pk), zk)
            rms, apply = _kabsch(pk[perm], p0, improper)
        fits.append((None, rms, apply, improper))
    return sorted(fits, key=lambda f: (f[1], f[3]))


def _assign_by_element(p0, z0, X, zk):
    """``perm`` with ``X[perm]`` matched to ``p0`` within each element by least total squared distance."""
    perm = np.empty(len(p0), dtype=np.int64)
    for e in np.unique(z0).tolist():
        i0, ik = np.nonzero(z0 == e)[0], np.nonzero(zk == e)[0]
        rr, cc = linear_sum_assignment(((p0[i0][:, None] - X[ik][None]) ** 2).sum(-1))
        perm[i0[rr]] = ik[cc]
    return perm


def _nearby_candidates(p0, z0, fitted, zk, rule):
    """Per atom of unit 0, the same-element atoms of unit k (fitted onto it) it may map to:
    those within ``_REFINE_TOL`` of the nearest (``rule='tolerance'``), or the
    ``_REFINE_CANDIDATES`` nearest (``rule='nearest'``)."""
    d = np.linalg.norm(p0[:, None] - fitted[None], axis=-1)
    d[z0[:, None] != zk[None]] = np.inf
    if rule == 'tolerance':
        lim = d.min(1, keepdims=True) + _REFINE_TOL
        return [set(np.nonzero(np.isfinite(d[i]) & (d[i] <= lim[i]))[0].tolist()) for i in range(len(p0))]
    near = np.argsort(d, axis=1)[:, :_REFINE_CANDIDATES]
    return [set(near[i][np.isfinite(d[i, near[i]])].tolist()) for i in range(len(p0))]


def _best_isomorphism(G0, Gk, p0, pk, iso_cap, allowed):
    """``(best, exhausted)``: among at most ``iso_cap`` element-preserving isomorphisms G0 -> Gk
    (restricted to ``allowed[i]`` targets for node i when given), the one whose Kabsch fit of
    ``pk`` onto ``p0`` (proper or improper) has the smallest RMSD (``_better``), as
    ``(hp, rmsd, apply, improper)`` with ``pk[hp]`` index-matched to ``p0``; and whether the
    search ran to its end.

    Bounded twice: by ``iso_cap`` isomorphisms, and by ``_ISO_WORK * iso_cap`` candidate
    pairs tested (VF2 visits a search state only after its candidate pair passes
    ``node_match``, so this bounds the states, each of polynomial cost). A search stopped
    by either bound returns ``exhausted`` False, with the best isomorphism found (or None).
    """
    budget = [_ISO_WORK * iso_cap]

    def node_match(a, b):
        budget[0] -= 1
        if budget[0] < 0:
            raise _SearchBound()
        return a['e'] == b['e'] and (allowed is None or b['id'] in allowed[a['id']])

    it = nx.algorithms.isomorphism.GraphMatcher(G0, Gk, node_match=node_match).isomorphisms_iter()
    best, exhausted = None, False
    try:
        for _ in range(iso_cap):                # bounded by iso_cap (and by the budget inside)
            m = next(it, None)
            if m is None:
                exhausted = True
                break
            hp = np.array([m[i] for i in range(len(p0))])
            for improper in (False, True):
                rms, apply = _kabsch(pk[hp], p0, improper)
                fit = (hp, rms, apply, improper)
                if best is None or _better(fit, best):
                    best = fit
        if not exhausted and next(it, None) is None:
            exhausted = True
    except _SearchBound:
        exhausted = False
    finally:
        it.close()
    return best, exhausted


def _hydrogen_parents(H, heavy, pos, graph):
    """Per hydrogen its parent heavy atom: the one bonded heavy atom, else the nearest heavy atom (-1: none)."""
    out = np.full(len(H), -1, dtype=np.int64)
    heavy_set = set(heavy.tolist())
    for n, h in enumerate(H.tolist()):
        nbrs = [j for j in graph.neighbors(h) if j in heavy_set]
        if len(nbrs) == 1:
            out[n] = nbrs[0]
        elif len(heavy):
            if nbrs:
                cand = np.array(nbrs)
            elif graph.degree(h) == 0:
                continue                            # an unbonded H: matched among the unbonded
            else:
                cand = heavy
            out[n] = cand[np.argmin(np.linalg.norm(pos[cand] - pos[h], axis=1))]
    return out


###############################################################################
# parameterize_pose
###############################################################################

def _pose_rows(idx, ptr, z_all, pos64, zp_all, sg_all, T_fc64, T_cf64, ops_list, degenerate_tol, handedness_tol,
               atoms_per_unit):
    """Pose of every unit of rows ``idx``: per row a result dict or a ``_Refusal``. ``atoms_per_unit`` [B]: each
    row's unit size, from ``unit_layout`` (equal contiguous units at this stage)."""
    out = {}
    unit_pos, unit_z, unit_owner = [], [], []
    picks = {}
    for b in idx:
        s, e = int(ptr[b]), int(ptr[b + 1])
        zp, sg = int(zp_all[b]), int(sg_all[b])
        if str(sg) not in ASYM_UNITS:
            out[b] = _Refusal('pose_failed', f'space group {sg} has no asymmetric-unit box in ASYM_UNITS')
            continue
        n_unit = int(atoms_per_unit[b])
        z_row, p_row = z_all[s:e], pos64[s:e]
        heavy_counts = [int((z_row[u * n_unit:(u + 1) * n_unit] > 1).sum()) for u in range(zp)]
        if min(heavy_counts) < 3:
            out[b] = _Refusal('degenerate_axes', f'a unit has {min(heavy_counts)} heavy atom(s); principal axes need '
                                                 'three')
            continue
        ops = torch.as_tensor(np.asarray(ops_list[b]), dtype=_F64)
        T_fc, T_cf = T_fc64[b], T_cf64[b]
        row = dict(units=[])
        for u in range(zp):
            zu, pu = z_row[u * n_unit:(u + 1) * n_unit], p_row[u * n_unit:(u + 1) * n_unit]
            images, cents = _unit_images(pu, zu, T_fc, T_cf, ops)
            heavy = zu > 1
            chosen, centroids, well = identify_canonical_asymmetric_unit(T_cf, ASYM_UNITS, sg,
                                                                         images[:, heavy])
            k = int(chosen[0])
            # the function wraps its own centroids; move the image by the integer that
            # makes its heavy-atom centroid equal the stored one
            shift = torch.round(centroids[k] - cents[k])
            img = images[k] + (T_fc @ shift)[None]
            # an axis on which the box spans the whole cell cannot be left by rounding (the
            # centroid wraps back in); on a narrower axis, a centroid at the wrap point or at
            # the box's upper face can fall either side
            box = torch.as_tensor(ASYM_UNITS[str(sg)], dtype=_F64)
            narrow = box < 1 - _ASU_FACE_TOL
            near = (centroids < _ASU_FACE_TOL) | (centroids > 1 - _ASU_FACE_TOL) | \
                   ((centroids - box).abs() < _ASU_FACE_TOL)
            near_face = bool((near & narrow).any())
            row['units'].append(dict(pos=img, centroid=centroids[k], op=k, well=bool(well), near_face=near_face))
            unit_pos.append(img)
            unit_z.append(zu)
            unit_owner.append(b)
        picks[b] = row
    if not unit_pos:
        return out
    n_units = len(unit_pos)
    counts = torch.tensor([len(p) for p in unit_pos])
    rotvec, hand, moments, stability = _orientations(torch.cat(unit_pos), torch.cat(unit_z), counts, n_units)
    u = 0
    for b, row in picks.items():
        zp = len(row['units'])
        sl = slice(u, u + zp)
        u += zp
        gaps = torch.minimum(moments[sl, 1] - moments[sl, 0], moments[sl, 2] - moments[sl, 1]) / moments[sl, 2]
        if bool((hand[sl] == 0).any()) or bool((~torch.isfinite(gaps)).any()) or bool((gaps < degenerate_tol).any()):
            out[b] = _Refusal('degenerate_axes', f'relative principal-moment gap {float(gaps.min()):.3g} is below '
                                                 f'degenerate_tol = {degenerate_tol:g} (or the frame is degenerate)')
            continue
        units = row['units']
        out[b] = dict(pos=torch.cat([x['pos'] for x in units]),
                      centroid=torch.stack([x['centroid'] for x in units]),
                      orientation=rotvec[sl], handedness=hand[sl], op=torch.tensor([x['op'] for x in units]),
                      well_defined=all(x['well'] for x in units),
                      ambiguous=any((not x['well']) or x['near_face'] for x in units),
                      unstable=bool((stability[sl] < handedness_tol).any()))
    return out


def _unit_images(pu, zu, T_fc, T_cf, ops):
    """Every operator's image of one unit [m, n, 3] (Cartesian), each shifted so its heavy-atom
    centroid (fractional, unweighted) lies in [0, 1); and those centroids [m, 3]."""
    f = pu @ T_cf.T
    y = torch.einsum('kij,nj->kni', ops[:, :3, :3], f) + ops[:, None, :3, 3]
    c = y[:, zu > 1].mean(1)
    y = y - torch.floor(c)[:, None]
    return y @ T_fc.T, c - torch.floor(c)


def _orientations(pos, z, counts, n_units):
    """Per unit (one graph each): rotation vector, handedness, principal moments, and handedness stability.

    ``extract_aunit_orientation``'s construction in the positions' dtype (that function
    builds its sign matrix in float32), with each axis's sign taken from the frame of the
    float32 copy of the positions: the frame the unit-cell builder computes when it
    standardizes the molecule (``align_mol_batch_to_standard_axes``, float32 only). The
    two agree except where a sign is set by rounding (a tie for the farthest heavy atom,
    an axis normal to it); where an axis differs by more than a sign (|cosine| < 0.9,
    near-equal moments), the float32 axes are taken whole. Stability is the smallest of:
    each axis's overlap (cosine) with the reference direction to the farthest heavy atom,
    and the relative gap between the two farthest heavy-atom distances.
    """
    batch = torch.arange(n_units).repeat_interleave(counts)
    Ip, Ipm, _ = batch_molecule_principal_axes_torch(pos, batch, n_units, counts, heavy_atoms_only=True,
                                                     atom_types=z)
    if pos.dtype != torch.float32:
        Ip32, _, _ = batch_molecule_principal_axes_torch(pos.float(), batch, n_units, counts, heavy_atoms_only=True,
                                                         atom_types=z)
        Ip32 = Ip32.to(pos.dtype)
        cos = (Ip * Ip32).sum(-1)                                       # [units, axis]
        aligned = Ip * torch.where(cos < 0, -1.0, 1.0).to(pos.dtype)[..., None]
        whole = (cos.abs() < 0.9).any(1)
        Ip = torch.where(whole[:, None, None], Ip32, aligned)
    hand = compute_Ip_handedness(Ip).long()
    eye = torch.eye(3, dtype=pos.dtype).repeat(n_units, 1, 1)
    eye[:, 0, 0] = hand.to(pos.dtype)
    rotmat = extract_rotmat(eye, torch.linalg.inv(Ip)).permute(0, 2, 1)
    rotvec = cleanup_invalid_rotvecs(rotmat, rotmat2rotvec(rotmat, warn_on_bad_determinant=False))
    canonicalize_rotvec(rotvec)
    stability = torch.empty(n_units, dtype=pos.dtype)
    heavy = z > 1
    for u in range(n_units):                    # bounded by the number of units
        sel = (batch == u) & heavy
        p = pos[sel] - pos[sel].mean(0)
        dist = p.norm(dim=1)
        top = torch.topk(dist, min(2, len(dist))).values
        gap = (top[0] - top[1]) / top[0] if len(top) == 2 else torch.ones((), dtype=pos.dtype)
        direction = p[torch.argmax(dist)] / dist.max()
        overlap = (Ip[u] @ direction).abs().min()
        stability[u] = torch.minimum(gap, overlap)
    return rotvec, hand, Ipm, stability


###############################################################################
# check_rebuild
###############################################################################

def _rebuild_rows(batch, idx, max_dev, mean_dev, match_elements):
    """Rebuild and match rows ``idx`` of a posed batch; per row a result dict or a ``_Refusal``."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData

    ptr = batch.ptr.detach().to('cpu').long()
    sub = batch.select_rows(torch.tensor(idx, dtype=torch.long))
    cell = torch.cat([sub.cell_lengths, sub.cell_angles], 1).detach().to('cpu', _F64)
    zmax = int(sub.z_prime.max())
    # the rows' unit map goes with them into the work batch, so the build reads the units the batch says (from_arrays
    # flags a row whose map is not its Z' equal blocks, and it is refused below)
    unit_map = {UNIT_MAP_KEY: sub[UNIT_MAP_KEY].detach().to('cpu')} if UNIT_MAP_KEY in sub._store else None
    work = MolCrystalData.from_arrays(
        cell, sub.pos.detach().to('cpu', _F64), sub.z.detach().to('cpu'), ptr=sub.ptr.detach().to('cpu'),
        sg=sub.sg_ind.detach().to('cpu'), ops=list(sub.symmetry_operators), z_prime=sub.z_prime.detach().to('cpu'),
        aunit_centroid=sub.aunit_centroid[:, :3 * zmax].detach().to('cpu', _F64),
        aunit_orientation=sub.aunit_orientation[:, :3 * zmax].detach().to('cpu', _F64),
        aunit_handedness=sub.aunit_handedness.reshape(len(idx), -1)[:, :zmax].detach().to('cpu'),
        atom_attrs=unit_map, dtype=torch.float32)
    bad = (work.status != _OK).nonzero().reshape(-1).tolist()
    if bad:
        raise ValueError(f'rows {[idx[i] for i in bad]} cannot be rebuilt: {[work.message[i] for i in bad]}')
    work.mol2ucell()
    built_pos = work.unit_cell_pos.detach().to('cpu', _F64)
    posed_pos = work.pos.detach().to('cpu', _F64).clone()
    hand_in = work.aunit_handedness.detach().to('cpu').reshape(len(idx), -1)
    _, _, hand_re, _, pos_re = work.reparameterize_unit_cell()
    pos_re = torch.cat(list(pos_re)) if isinstance(pos_re, (list, tuple)) else pos_re
    pos_re = pos_re.detach().to('cpu', _F64)
    hand_re = hand_re.detach().to('cpu').reshape(len(idx), -1)

    # built fractional coordinates: undo the builder's own (float32-cell) transform exactly
    T_cf32 = work.T_cf.detach().to('cpu', _F64)
    T_fc_true, _ = _transforms64(sub)
    have_deposit = 'deposit_frac' in sub._store
    ref_frac_all = sub['deposit_frac'].detach().to('cpu', _F64) if have_deposit else \
        torch.einsum('nij,nj->ni', _transforms64(sub)[1][_atom_rows(sub.ptr.cpu())], sub.pos.detach().to('cpu', _F64))
    sym_mult = work.sym_mult.tolist()
    sub_ptr = sub.ptr.detach().to('cpu').long()
    # each row's units from its unit map (the one reader; equal contiguous blocks at this stage)
    per_unit = unit_layout(sub, 'check_rebuild').atoms_per_unit.tolist()
    out = {}
    ucell_start = 0
    for i, b in enumerate(idx):
        s, e = int(sub_ptr[i]), int(sub_ptr[i + 1])
        n, m, zp = e - s, sym_mult[i], int(sub.z_prime[i])
        built = built_pos[ucell_start:ucell_start + n * m]
        ucell_start += n * m
        ops = torch.as_tensor(np.asarray(sub.symmetry_operators[i]), dtype=_F64)
        zrow = sub.z[s:e].detach().to('cpu')
        # built layout: [unit][operator][atom of the unit]; source posed atom of each built atom
        n_u = per_unit[i]
        u_idx = torch.arange(zp).repeat_interleave(m * n_u)
        a_idx = torch.arange(n_u).repeat(zp * m)
        source = u_idx * n_u + a_idx
        built_f = built @ T_cf32[i].T
        ref = torch.einsum('kij,nj->kni', ops[:, :3, :3], ref_frac_all[s:e]) + ops[:, None, :3, 3]
        ref_f, ref_z = ref.reshape(-1, 3), zrow.repeat(m)
        res = _match_sets(built_f, zrow[source], ref_f, ref_z, T_fc_true[i], match_elements, max_dev)
        atom_dev = torch.zeros(n, dtype=_F64).scatter_reduce(0, source, res['dist'], reduce='amax',
                                                              include_self=False)
        # reference layout [operator][posed atom]: built atom from posed atom i must sit on an image of i
        relabelled = int((res['arg'] % n != source).sum())
        info = dict(max_dev=float(res['dist'].max()) if len(res['dist']) else 0.0,
                    mean_dev=float(res['dist'].mean()) if len(res['dist']) else 0.0, atom_dev=atom_dev)
        if res['problem'] is not None:
            out[b] = _Refusal(res['problem'][0], res['problem'][1], **info)
            continue
        if info['max_dev'] > max_dev or info['mean_dev'] > mean_dev:
            out[b] = _Refusal('rebuild_mismatch', f"rebuilt atoms lie up to {info['max_dev']:.3g} A (mean "
                                                  f"{info['mean_dev']:.3g} A) from the reference, above max_dev = "
                                                  f'{max_dev:g} or mean_dev = {mean_dev:g}', **info)
            continue
        moved = float((pos_re[s:e] - posed_pos[s:e]).norm(dim=1).max()) if n else 0.0
        hand_differs = not torch.equal(hand_re[i, :zp], hand_in[i, :zp])
        reposed = moved > max_dev or hand_differs
        notes = []
        if relabelled:
            notes.append(f'{relabelled} built atom(s) sit on an image of another posed atom than their own: label_flip')
        if reposed:
            notes.append(f're-parameterizing the built cell gives another pose (atoms moved up to {moved:.3g} A'
                         + (', handedness differs' if hand_differs else '') + '): label_flip')
        out[b] = dict(info, label_flip=bool(relabelled) or reposed, note='; '.join(notes))
    return out


def _match_sets(built_f, built_z, ref_f, ref_z, T_fc, match_elements, max_dev):
    """Nearest reference atom of every built atom under periodic images (fractional inputs).

    Returns ``dist`` [nb] (Angstrom, to the matched atom), ``arg`` [nb] (its index in the
    reference) and ``problem`` (None or
    ``(code, text)``): ``unmatched_atoms`` when the sets differ in size or the match is not
    one to one; ``element_mismatch`` when, with ``match_elements``, the match ignoring
    elements is one to one within ``max_dev`` but the element-aware one is not.
    """
    def nearest(same_element_only):
        dist = torch.empty(len(built_f), dtype=_F64)
        arg = torch.empty(len(built_f), dtype=torch.long)
        block = max(1, _PAIR_BLOCK // max(1, len(ref_f)))
        for s in range(0, len(built_f), block):     # bounded blocks
            df = built_f[s:s + block, None] - ref_f[None]
            df = df - torch.round(df)
            d = (df @ T_fc.T).norm(dim=-1)
            if same_element_only:
                d = torch.where(built_z[s:s + block, None] == ref_z[None], d, torch.full_like(d, float('inf')))
            dist[s:s + block], arg[s:s + block] = d.min(1)
        return dist, arg

    if len(built_f) != len(ref_f):
        dist, arg = nearest(match_elements)
        return dict(dist=dist, arg=arg, problem=('unmatched_atoms', f'{len(built_f)} built atoms against '
                                                                    f'{len(ref_f)} in the reference'))
    dist, arg = nearest(match_elements)
    one_to_one = len(torch.unique(arg)) == len(arg)
    if match_elements and (not one_to_one or bool((dist > max_dev).any())):
        d_any, a_any = nearest(False)
        if len(torch.unique(a_any)) == len(a_any) and bool((d_any <= max_dev).all()):
            wrong = int((built_z != ref_z[a_any]).sum())
            return dict(dist=d_any, arg=a_any, problem=('element_mismatch', f'the geometry matches but {wrong} built '
                                                                            'atom(s) sit on atoms of another element'))
    if not one_to_one:
        shared = len(arg) - len(torch.unique(arg))
        return dict(dist=dist, arg=arg, problem=('unmatched_atoms', f'{shared} reference atom(s) are the nearest of '
                                                                    'more than one built atom'))
    return dict(dist=dist, arg=arg, problem=None)


###############################################################################
# check_validity
###############################################################################

def _moiety_counts(moiety, z_value, n_ops) -> Optional[Dict[str, float]]:
    """Element counts the deposited set should hold: the moiety formula (``cif/composition.py``
    ``parse_formula_moiety``; each moiety times its coefficient) times ``z_value / n_ops``,
    with D and T counted as H; None when there is no moiety, it does not parse, or Z < 1."""
    if not isinstance(moiety, str) or not moiety or z_value < 1 or n_ops < 1:
        return None
    try:
        units = parse_formula_moiety(moiety)
    except ValueError:                          # e.g. 'C. H4' (see parse_formula_moiety)
        return None
    if not units:
        return None
    out = {}
    for u in units:
        for sym, count in u.elements.items():
            sym = 'H' if sym in ('D', 'T') else sym
            out[sym] = out.get(sym, 0.0) + u.count * count * z_value / n_ops
    return out


def _validity_row(batch, b, crit, found, expected=None, n_u=None) -> dict:
    """One row's criteria: the ones that ran (bitmask), the ones failed (in order), messages. ``n_u``: the row's
    unit size from ``unit_layout`` (rows with units)."""
    s, e = int(batch.ptr[b]), int(batch.ptr[b + 1])
    z = batch.z[s:e].detach().to('cpu')
    checked, failed, messages = 0, [], []

    def ran(name):
        nonlocal checked
        checked |= 1 << VALIDITY_CRITERIA.index(name)

    def fail(name, text):
        failed.append(name)
        messages.append(f'{name}: {text}')

    if 'max_atomic_number' in crit:
        ran('max_atomic_number')
        if len(z) and int(z.max()) > crit['max_atomic_number']:
            fail('max_atomic_number', f"atomic number {int(z.max())} > {crit['max_atomic_number']}")
    if found and ('max_heavy_atoms' in crit or 'heavy_nn_range' in crit):
        zp = int(batch.z_prime[b])
        n_u = int(n_u)
        pos = batch.pos[s:e].detach().to('cpu', _F64)
        for u in range(zp):
            zu, pu = z[u * n_u:(u + 1) * n_u], pos[u * n_u:(u + 1) * n_u]
            hv = pu[zu > 1]
            if 'max_heavy_atoms' in crit and len(hv) > crit['max_heavy_atoms']:
                fail('max_heavy_atoms', f"unit {u} has {len(hv)} heavy atoms > {crit['max_heavy_atoms']}")
            if 'heavy_nn_range' in crit:
                lo, hi = crit['heavy_nn_range']
                d = torch.cdist(hv, hv) + torch.diag(torch.full((len(hv),), float('inf'), dtype=_F64))
                nn = d.min(1).values if len(hv) else torch.zeros(0, dtype=_F64)
                if bool(((nn < lo) | (nn > hi)).any()):
                    fail('heavy_nn_range', f'unit {u}: nearest heavy neighbour distances span '
                                           f'[{float(nn.min()):.3g}, {float(nn.max()):.3g}] A, outside [{lo:g}, {hi:g}]')
        if 'max_heavy_atoms' in crit:
            ran('max_heavy_atoms')
        if 'heavy_nn_range' in crit:
            ran('heavy_nn_range')
    if 'occupancy' in crit:
        ran('occupancy')
        occ = batch.occupancy[s:e].detach().to('cpu', _F64)
        off = (occ - 1).abs()
        if len(off) and float(off.max()) > crit['occupancy']:
            fail('occupancy', f"a site has occupancy {float(occ[torch.argmax(off)]):.4g}, off 1 by more than "
                              f"{crit['occupancy']:g}")
    if 'disorder' in crit:
        ran('disorder')
        if bool(batch.has_disorder[b]):
            fail('disorder', 'the row is flagged disordered')
    ops = np.asarray(batch.symmetry_operators[b], dtype=np.float64)
    if 'cell_constraints' in crit:
        ran('cell_constraints')
        T_fc = _transforms64(batch)[0][b].numpy()
        G = T_fc.T @ T_fc
        W = ops[:, :3, :3]
        excess = float(np.abs(np.einsum('kji,jl,klm->kim', W, G, W) - G[None]).max() / np.abs(G).max())
        if excess > crit['cell_constraints']:
            fail('cell_constraints', f'the cell metric changes by {excess:.3g} (relative) under the operators, above '
                                     f"{crit['cell_constraints']:g}")
    if 'min_contact' in crit:
        ran('min_contact')
        T_fc = _transforms64(batch)[0][b].numpy()
        if 'deposit_frac' in batch._store:
            f = batch.deposit_frac[s:e].detach().to('cpu', _F64).numpy()
        else:
            f = batch.pos[s:e].detach().to('cpu', _F64).numpy() @ np.linalg.inv(T_fc).T
        cell = (np.einsum('kij,nj->kni', ops[:, :3, :3], f) + ops[:, None, :3, 3]).reshape(-1, 3)
        # one cutoff for every pair: no [n, n] matrix (quadratic in the cell's atoms)
        a, bb, _, d = _contacts(cell, cell, T_fc, crit['min_contact'], exclude_self=True)
        if len(d):
            fail('min_contact', f"{len(d) // 2 or 1} atom pair(s) of the unit cell closer than "
                                f"{crit['min_contact']:g} A (closest {float(d.min()):.3g} A)")
    if 'composition' in crit and expected is not None:
        ran('composition')
        got = {}
        for zz in z.tolist():
            sym = chemical_symbols[int(zz)]
            got[sym] = got.get(sym, 0) + 1
        keys = sorted((set(got) | set(expected)) - ({'H'} if crit['composition'] == 'heavy' else set()))
        off = [f'{k} {got.get(k, 0)} (expected {expected.get(k, 0.0):g})' for k in keys
               if abs(got.get(k, 0) - expected.get(k, 0.0)) > 1e-6]
        if off:
            fail('composition', 'the deposited set holds ' + ', '.join(off) + " from the moiety formula x Z / n_ops")
    return dict(checked=checked, failed=sorted(failed, key=VALIDITY_CRITERIA.index), messages=messages)

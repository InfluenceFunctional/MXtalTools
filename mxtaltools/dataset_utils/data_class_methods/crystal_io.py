"""Batch ingestion mixed into MolCrystalData: read CIF files, or build a batch from arrays.

Both entry points return ONE collated ``MolCrystalDataBatch``, built directly as a batch
(no per-row ``__init__``, no ``collate_data_list``), with a per-row ``status`` instead of
an exception for a bad structure. Bad ARGUMENTS still raise.

Two states (``docs/design/crystal_ingestion.md`` §3, Table 3, in the GFN repository):

- **bare**: the deposited set (the atoms the operators expand, Cartesian), the cell, the
  space group and the file's operators, file metadata and status. Set by ``read`` and by
  ``from_arrays`` without a pose. No consumer of molecular crystals accepts a bare batch:
  the unit-cell builder, the ELJ energy and the FD importer all need a pose.
- **posed**: units, Z', pose, the posed asymmetric unit, per-unit radius / mass / volume
  summed per row, ``x`` zeros. Set by ``from_arrays`` with a pose (and later by the
  ``parameterize_pose`` step, not built here).

Per-row fields both entry points write (B rows, N atoms):

``z`` [N] long, ``pos`` [N, 3] (Angstrom), ``batch`` [N], ``ptr`` [B + 1], ``num_atoms`` [B]
    The atoms; for ``read`` the deposited set as written in the file (not unwrapped:
    joining molecules split across the cell is the job of a later molecule-finding step).
``cell_lengths``, ``cell_angles`` [B, 3]
    Angstrom and radians.
``T_fc``, ``T_cf`` [B, 3, 3], ``cell_volume`` [B]
    From ``batch_compute_fractional_transform`` on the stored (``dtype``) cell.
``sg_ind`` [B] long, ``symmetry_operators`` list[B] of float64 [m, 4, 4], ``sym_mult`` [B]
    The operators as given (for ``read``, the file's own, in file order), never
    regenerated from the number unless the caller asks with ``setting='standard'``.
``setting`` [B] int8
    Index into ``SETTINGS``: 0 the operators equal ``SYM_OPS[sg]`` element-wise in order
    (atol 1e-6), 1 the same set of symmetry elements (rotations equal, translations equal
    modulo 1) in another order or with lattice-shifted translations, 2 anything else;
    -1 where no operators were established (failed ``read`` rows).
``nonstandard_symmetry`` [B] bool
    ``setting != 0``; the flag ``split_to_zp1_batch`` reads.
``status`` [B] int8, ``message`` list[B] of str
    Index into ``READ_STATUS``, and its explanation ('' when there is nothing to say).
``done`` [B] int16
    Bitmask of the ingestion steps that have run; ``DONE_READ`` (bit 0) on every row.

The ``_slice_dict`` / ``_inc_dict`` bookkeeping is written from the DECLARED level of
each attribute (per atom or per crystal), the way ``collate_data_list`` writes it for the
same rows, never inferred from a leading dimension. ``select_rows`` keeps a subset of rows
by reading those levels back for every attribute. ``subsample_new_batch`` reads them where
a tensor's length leaves the level ambiguous (as many atoms as rows, which ``read``
reaches since its failed rows hold no atoms; a one-row or one-atom batch) and keeps the
length rule elsewhere, so on these batches it selects the same rows.
"""
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch_geometric.data.storage import GlobalStorage
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import (batch_compute_fractional_transform, batch_compute_mol_mass,
                                              batch_compute_mol_radius, batch_compute_molecule_volume)
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_building.utils import UNIT_MAP_KEY, NO_UNIT, equal_block_unit_ind, unit_map_problem
from mxtaltools.dataset_utils.construction.cif.errors import (CifInconsistentSymmetryError, CifMissingDataError,
                                                              CifParseError, CifReadError, CifSymmetryError,
                                                              CifUnsupportedError)
# The tag tuples are the reader's own (module-private there); importing them keeps one
# definition of which spellings count, instead of a copy that can drift.
from mxtaltools.dataset_utils.construction.cif.reader import _CELL_ANGLE_TAGS, _CELL_LENGTH_TAGS, _Z_TAGS
from mxtaltools.dataset_utils.construction.cif.rhombohedral import reexpress_rhombohedral_axes
from mxtaltools.dataset_utils.construction.cif.sites import _LABEL_TAG, sites_from_block
from mxtaltools.dataset_utils.construction.cif.symmetry import (_HM_TAGS, _NUM_TAGS, _OP_TAGS, _centring_vectors,
                                                                _factorises_as_centred, space_group_from_block)
from mxtaltools.dataset_utils.construction.cif.tokenizer import as_float, as_int, is_missing, parse_cif
from mxtaltools.dataset_utils.construction.cif.topology import (_BOND_1, _BOND_2, _BOND_SYM_1, _BOND_SYM_2,
                                                                _MOIETY)

__all__ = ['MolCrystalIO', 'READ_STATUS', 'SETTINGS', 'DONE_READ', 'POSE_KEYS']

#: Row status codes; ``status`` holds the index. The first ten are the read-level codes of
#: the ingestion contract (Table 5). ``from_arrays`` adds two: ``z_prime_mismatch`` for a
#: pose whose atom count does not split into Z' equal units (the name the molecule-finding
#: step will also use), and ``bad_coordinates`` for non-finite coordinates (``read`` maps
#: those to ``parse_error``, as a non-numeric value). Codes are only ever appended, so an
#: index keeps its meaning.
READ_STATUS = ('ok', 'unreadable_file', 'parse_error', 'missing_data', 'no_operators',
               'inconsistent_symmetry', 'bad_element', 'duplicate_label', 'bad_cell',
               'partial_occupancy', 'z_prime_mismatch', 'bad_coordinates')
_STATUS = {name: i for i, name in enumerate(READ_STATUS)}

#: ``setting`` codes 0, 1, 2 (index into this tuple); -1 = no operators established.
SETTINGS = ('standard', 'reordered', 'other')

#: ``done`` bit of the read (both entry points set it). Later ingestion steps take
#: further bits when they are built.
DONE_READ = 1

#: Attributes that together attach a pose; all four or none.
POSE_KEYS = ('z_prime', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness')

#: Names ``from_arrays`` computes, so a caller may not pass them in ``attrs`` or
#: ``atom_attrs``: for every batch, and additionally for a posed one.
_COMPUTED = ('z', 'pos', 'batch', 'ptr', 'num_atoms', 'cell_lengths', 'cell_angles', 'T_fc', 'T_cf',
             'cell_volume', 'sg_ind', 'symmetry_operators', 'sym_mult', 'setting', 'nonstandard_symmetry',
             'status', 'message', 'done')
_COMPUTED_POSED = ('max_z_prime', 'mol_ind', 'x', 'radius', 'mass', 'mol_volume')

#: Substrings of attribute names that ``MXtalBase.__inc__`` / ``__cat_dim__`` treat
#: specially on collation (offset per row); ``from_arrays`` refuses such names.
_OFFSET_KEY_PARTS = ('batch', 'index', 'face')

#: The format options ``read(format='cif')`` takes, with their defaults: the native
#: reader's flags that apply at read level. Polymer, Z' and bond-loop refusals belong to
#: the later molecule-finding step, so their flags are not options here.
_CIF_OPTIONS = {'max_atomic_number': 100}

_OP_TOL = 1e-6


class _RowRefusal(Exception):
    """Internal: one row cannot be read; carries the ``READ_STATUS`` name and a message."""

    def __init__(self, code: str, message: str):
        assert code in _STATUS, code
        self.code = code
        super().__init__(message)


# noinspection PyAttributeOutsideInit
class MolCrystalIO:
    """Mixin providing the batch constructors ``read`` and ``from_arrays``."""

    @classmethod
    def read(cls,
             sources,
             format: str = 'cif',
             *,
             identifiers: Optional[Sequence[str]] = None,
             dtype: torch.dtype = torch.float32,
             device: Union[str, torch.device] = 'cpu',
             full=None,
             **format_options):
        """Read crystal files into one bare batch; every ``data_`` block becomes a row.

        A source that cannot be opened or parsed still becomes one row, with a status and
        no atoms, so every source maps to at least one row and a bad file never raises.
        Nothing beyond the read runs (no molecule finding, no Z', no pose) unless ``full``
        names a recipe (the owner's default: ``full=None``, read only).

        Parameters
        ----------
        sources : str, os.PathLike, or list/tuple of them
            A path, a list of paths, or CIF text; a list may mix them. An ``os.PathLike``
            is always a path. A ``str``, stripped of surrounding whitespace and a leading
            byte-order mark, is CIF text when it spans several lines, or when it is one
            line that starts with ``data_`` (any case), contains whitespace, does not end
            in ``.cif`` and names nothing on disk; any other ``str`` is a path (so a
            directory, a missing file of any suffix, or a path read with its newline
            gives ``unreadable_file``, not ``parse_error``). Files are read as UTF-8 with
            a leading byte-order mark and undecodable bytes dropped.
        format : str
            ``'cif'`` (the only format so far).
        identifiers : sequence of str, optional
            One per source. A source with one block gives its row this identifier; a
            source with several gives each row ``f'{identifier}:{block name}'``. When
            None, a row's identifier is its block name, else ``_chemical_name_common``,
            else the file name's stem (``''`` for text that names nothing).
        dtype : torch.dtype
            Floating dtype of positions, cell, transforms and occupancy.
        device : str or torch.device
        full : None, str or Recipe
            None (default): read only. A recipe name (``'csd_v1'``, see
            ``crystal_recipes.RECIPES``) or a ``crystal_recipes.Recipe``: read, then run the
            recipe's steps (``run_chain``). The recipe fixes every ``read`` argument, so
            ``format``, ``dtype`` and the format options must equal its ``read`` step's
            (``csd_v1`` reads float64: pass ``dtype=torch.float64``; the float32 default
            raises ``ValueError``); the ``run_chain`` record then says ``read_checked=True``.
        **format_options
            For ``'cif'``: ``max_atomic_number`` (int or None, default 100), the native
            reader's limit; a site above it gives the row ``bad_element``.

        Returns
        -------
        MolCrystalDataBatch
            The bare state (see the module docstring) plus file metadata:

            - ``identifier``, ``source_path`` (str, or None for text), ``block_name``
              (None when no block was parsed): list[B];
            - ``z_value`` [B] long, ``_cell_formula_units_Z``, -1 when absent (times 3 on
              a row re-expressed from rhombohedral axes, see Notes);
            - ``hm_symbol``, ``hall_symbol``: list[B] of the file's H-M and Hall symbols
              as written (quotes stripped), None when absent; a re-expressed row carries
              the hexagonal-axes Hall symbol;
            - ``moiety``: list[B] of ``_chemical_formula_moiety`` as the native reader
              reads it (``CifCrystal.formula_moiety``), None when absent, ``.`` or ``?``;
              the molecule-finding step reads it for its polymer and multi-component
              checks;
            - ``atom_labels``: list[B] of the site labels, in atom order;
            - ``occupancy`` [N] (1.0 where the file states none) and ``has_occupancy``
              [B] bool, whether the file had an occupancy column;
            - ``file_bonds``: list[B], per row a long array [2, E] of row-local atom
              indices from ``_geom_bond``, or None when the file has no bond loop. A
              bond is kept when both site-symmetry codes are ``1_555`` or ``.`` (the CIF
              default, which SHELXL writes for bonds within the deposited set) or the
              columns are absent; any other code (an image, or ``?``) is not kept and is
              counted in ``message``. ``components_from_block`` drops ``.`` bonds too,
              so on non-CSD files the two can differ.

            A row with a status other than ``ok`` or ``partial_occupancy`` holds no
            atoms, a NaN cell, ``sg_ind`` 0, an empty [0, 4, 4] operator array,
            ``sym_mult`` 0, ``setting`` -1 and no moiety. A ``partial_occupancy`` row
            keeps its atoms and occupancies; later steps skip it. Keep a subset of rows
            with ``select_rows`` (e.g. ``batch.select_rows(batch.status == 0)``), which
            checks the indices and refuses an attribute at a level it cannot split;
            ``subsample_new_batch`` selects the same rows, including when the batch
            holds as many atoms as rows.

        Raises
        ------
        NotImplementedError
            ``full=True``: no recipe is the default (owner answer Q7); name one.
        ValueError
            An unknown ``format``, no sources, ``identifiers`` of the wrong length, an
            unknown recipe name, or read arguments that differ from the recipe's.
        TypeError
            A source that is neither a str nor a path, a non-floating ``dtype``, an unknown
            format option, or a ``full`` that is neither None, a name nor a ``Recipe``.

        Notes
        -----
        Status codes per row (``READ_STATUS``): ``unreadable_file`` (the file cannot be
        opened, including a directory or a path the OS refuses), ``parse_error`` (malformed
        CIF or no ``data_`` block, a non-numeric or non-finite coordinate or occupancy, a
        bond loop naming an absent site), ``missing_data`` (no cell, no atom-site loop, a
        missing coordinate, no space-group number or symbol), ``no_operators`` (no operator
        loop: operators are never guessed from a symbol or number),
        ``inconsistent_symmetry`` (operators that do not fit the IT number, cannot be
        parsed or lack the identity; an invalid number), ``bad_element`` (an unknown
        symbol, a placeholder such as ``X`` that maps to atomic number 0, or one above
        ``max_atomic_number``), ``duplicate_label``, ``bad_cell`` (non-finite or
        non-positive lengths, angles outside (0, 180) degrees, or angles that describe no
        real cell) and ``partial_occupancy``. Z, the bond loop and the Hall symbol are
        not required. The first failing stage names the row (symmetry, then sites, then
        cell, Z and bonds, the native reader's order).

        An R group filed on rhombohedral axes (a third of ``len(SYM_OPS[n])`` operators)
        is not refused: ``cif.rhombohedral.reexpress_rhombohedral_axes`` re-expresses it
        on hexagonal axes, obverse (cell, coordinates, operators ``SYM_OPS[n]``, Z times
        3), checked by rebuilding every site's images, and the row's message starts with
        the note saying so. It is ``inconsistent_symmetry`` when the cell is not
        rhombohedral, the Hall symbol is not the rhombohedral one, or the operators do not
        map onto ``SYM_OPS[n]``.
        """
        recipe = None
        if full is not None:
            # imported here: crystal_recipes imports this module
            from mxtaltools.dataset_utils.data_class_methods.crystal_recipes import (check_read_call, get_recipe,
                                                                                     run_chain_after_read)
            recipe = get_recipe(full)
        if format != 'cif':
            raise ValueError(f"format {format!r} is not supported; only 'cif'")
        _check_float_dtype(dtype)
        unknown = sorted(set(format_options) - set(_CIF_OPTIONS))
        if unknown:
            raise TypeError(f"unknown format option(s) {unknown} for format 'cif'; known: {sorted(_CIF_OPTIONS)}")
        options = {**_CIF_OPTIONS, **format_options}
        if recipe is not None:
            check_read_call(recipe, format=format, dtype=dtype, format_options=options)
            batch = cls.read(sources, format, identifiers=identifiers, dtype=dtype, device=device, **format_options)
            return run_chain_after_read(batch, recipe)          # run_chain, recorded with read_checked=True

        source_list = _as_source_list(sources)
        if identifiers is not None:
            if isinstance(identifiers, str) or len(identifiers) != len(source_list):
                raise ValueError(f'identifiers must be a sequence with one entry per source '
                                 f'({len(source_list)}), got {identifiers!r:.80}')

        rows = []
        for i, src in enumerate(source_list):
            ident = None if identifiers is None else str(identifiers[i])
            rows.extend(_read_cif_source(src, ident, options['max_atomic_number']))

        return _rows_to_batch(cls, rows, dtype, device)

    @classmethod
    def from_arrays(cls,
                    cell,
                    xyz,
                    z,
                    *,
                    ptr,
                    sg,
                    ops=None,
                    setting: Optional[str] = None,
                    atom_attrs: Optional[dict] = None,
                    dtype: torch.dtype = torch.float32,
                    device: Union[str, torch.device] = 'cpu',
                    **attrs):
        """Build one batch from arrays: bare, or posed when ``attrs`` carry a pose.

        Parameters
        ----------
        cell : array-like
            [B, 6] (a, b, c in Angstrom; alpha, beta, gamma in radians), or [B, 3, 3]
            lattice rows (row i is lattice vector i in the frame of ``xyz``). Lattice
            rows are converted to lengths and angles, and each row's ``xyz`` is rotated
            into the ``T_fc`` frame (a along x, b in the xy plane), in float64. Lattice
            rows are refused with a pose, which is defined in the ``T_fc`` frame and is
            not rotated here.
        xyz : array-like
            [N, 3] Cartesian coordinates, Angstrom: the deposited set (bare), or the
            posed asymmetric unit (posed).
        z : array-like
            [N] atomic numbers (integer values; not bool).
        ptr : array-like
            [B + 1] row offsets into the atoms; required, one-dimensional, integer (not
            bool), ``ptr[0] == 0``, non-decreasing, ``ptr[-1] == N``, B >= 1.
        sg : int or array-like
            Space-group number per row, [B], or one int for every row; 1 to 230; not
            bool.
        ops : optional
            Symmetry operators acting on fractional coordinates: a list or tuple of B
            arrays [m_b, 4, 4], one array [m, 4, 4] for every row, or [B, m, 4, 4].
            Required unless ``setting='standard'``. Stored as float64 numpy arrays, as
            ``collate_data_list`` stores them. Their content is checked per row as
            ``read`` checks a file's: finite, affine, holding the identity, and a count
            equal to ``len(SYM_OPS[sg])`` or a centred multiple of it.
        setting : {None, 'standard'}
            ``'standard'`` takes ``SYM_OPS[sg]`` for every row (``ops`` must then be
            None) and records setting 0. There is no default to the standard operators:
            regenerating them from the number moved ELJ on nonstandard rows (-991.0 to
            +4161.2 raw units) without a refusal (ingestion contract §3).
        atom_attrs : dict, optional
            Per-atom attributes, each with leading dimension N. Tensors and numeric
            arrays are stored per atom (moved to ``device``), as are lists of python
            bools, ints or floats, as tensors. Any other list (strings, objects) is
            stored as collate holds per-row lists: a list[B] of each row's slice, at
            row level, so row access and ``select_rows`` split it correctly.
            ``unit_ind``, the unit map (since the stage-2 review, 2026-10-23), is taken
            only with a pose: one integer (not bool) entry per atom, the row-local unit
            of each atom; stored long, per atom, as given. A row whose map is not its Z'
            equal contiguous blocks in unit order (the only units at this stage: e.g.
            units of 6 and 8 atoms at Z'=2) is a ``z_prime_mismatch`` row. Without it the
            pose's own layout is written as the map (below).
        dtype : torch.dtype
            Floating dtype of positions, cell, transforms, pose and computed properties.
        device : str or torch.device
        **attrs
            Per-crystal attributes, each with leading dimension B. A list or tuple of
            python bools, ints or floats becomes a tensor (bool, long, float32) as
            ``collate_data_list`` turns python values into tensors; other lists stay
            lists. All of ``POSE_KEYS`` attach a pose:

            - ``z_prime`` [B] (stored long);
            - ``aunit_centroid``, ``aunit_orientation`` [B, >= 3 * max_z_prime] and
              ``aunit_handedness`` [B, >= max_z_prime] (a [B] vector when
              max_z_prime is 1), truncated to ``max_z_prime`` slots, the largest Z' among
              the rows whose atoms split into Z' units (1 when none do); centroid and
              orientation cast to ``dtype``, handedness kept in its own dtype. Slots past
              a row's Z', and every slot of a ``z_prime_mismatch`` row, are set to 0, the
              pad ``collate_data_list`` uses (not the per-row constructor's 1).

        Returns
        -------
        MolCrystalDataBatch
            The bare fields of the module docstring, the attributes passed, and, when
            posed: ``max_z_prime`` (python int); ``mol_ind`` [N] long, the unit of each
            atom within its row (the atoms of a row are Z' equal contiguous blocks; 0 on
            a ``z_prime_mismatch`` row); ``unit_ind`` [N] long, the unit map
            (``crystal_building.utils.UNIT_MAP_KEY``, per atom): the one ``atom_attrs``
            gives, or else the pose's own layout written out (since stage 3,
            2026-10-23; before, none was written and every reader derived the same
            layout): Z' equal contiguous blocks, equal to ``mol_ind``, on every row whose
            atoms split into Z' units, and ``crystal_building.utils.NO_UNIT`` (-1) on a
            ``z_prime_mismatch`` row, which every reader refuses (as without a map);
            ``radius``, ``mass``, ``mol_volume`` [B] in ``dtype``, each the SUM over the
            row's units of the per-unit value (heavy-atom radius, atomic-weight mass,
            overlap-corrected van der Waals volume: ``MolData.mol_analysis``'s
            definitions), the convention the latent and ``box_analysis`` use; ``x`` [N]
            zeros. ``x`` is computed, not taken: charges are the output of the
            ``features`` step (not built), and ``done`` has no bit for them yet, so a
            caller's charges are refused rather than stored unmarked. The pose itself is
            never checked.

            A value that makes one row unusable gives that row a status (the first found,
            in this order) and keeps its data; a shape, type or bookkeeping error raises.
            ``inconsistent_symmetry``: operators that are non-finite, not affine, lack the
            identity or do not fit the space group. ``bad_element``: an atomic number
            outside 1 to ``len(ATOM_WEIGHTS) - 1`` (99). ``bad_coordinates``: a non-finite
            coordinate. ``bad_cell``: a non-finite or non-positive length, an angle
            outside (0, pi), angles that describe no real cell, or lattice rows that are
            not right-handed (det <= 0; that row's ``xyz`` is not rotated).
            ``z_prime_mismatch``: Z' < 1, fewer atoms than Z', an atom count not
            divisible by Z', or a given unit map that is not the row's Z' equal
            contiguous blocks (that row's ``mol_ind`` is 0). Radius, mass and volume are
            NaN on rows with a Z' mismatch, a bad element or bad coordinates.
            ``is_well_defined`` is not set: pass it in ``attrs`` when a consumer needs it
            (``batch_to_list`` does).

        Raises
        ------
        ValueError
            Shapes that disagree, a bad ``ptr``, a space group outside 1-230, a bool
            ``ptr``, ``sg`` or ``z``, non-integer or non-finite ``z``, ``ops`` missing
            without ``setting='standard'`` or given with it, an unknown ``setting``, a
            partial pose, a pose with [B, 3, 3] lattice rows, pose arrays narrower than Z'
            needs, an attribute with the wrong leading dimension, an attribute named like
            one this method computes, the unit map ``unit_ind`` in ``attrs``, or in
            ``atom_attrs`` without a pose or not one integer entry per atom, or an
            attribute whose name contains 'batch',
            'index' or 'face' (collation offsets those per row, which this method does
            not reproduce).
        TypeError
            A non-floating ``dtype``.
        """
        _check_float_dtype(dtype)
        atom_attrs = dict(atom_attrs or {})

        ptr_cpu = _as_ptr(ptr)
        B, N = len(ptr_cpu) - 1, int(ptr_cpu[-1])
        num_atoms = ptr_cpu[1:] - ptr_cpu[:-1]
        atom_row = torch.arange(B).repeat_interleave(num_atoms)

        xyz = torch.as_tensor(xyz)
        z = torch.as_tensor(z)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != N:
            raise ValueError(f'xyz must be [N, 3] with N = ptr[-1] = {N}, got {tuple(xyz.shape)}')
        if z.ndim != 1 or len(z) != N:
            raise ValueError(f'z must be [N] with N = ptr[-1] = {N}, got {tuple(z.shape)}')
        if z.dtype == torch.bool or z.is_complex() or (
                z.is_floating_point() and not (bool(torch.isfinite(z).all()) and torch.equal(z, z.round()))):
            raise ValueError('z must hold integer atomic numbers')

        sg_cpu = _as_sg(sg, B)
        ops_list, setting_codes, op_problems = _resolve_operators(ops, setting, sg_cpu, B)

        posed_keys = [k for k in POSE_KEYS if k in attrs]
        if posed_keys and len(posed_keys) != len(POSE_KEYS):
            raise ValueError(f'a partial pose: got {posed_keys}, a pose needs all of {list(POSE_KEYS)}')
        posed = bool(posed_keys)
        if posed and torch.as_tensor(cell).ndim == 3:
            # the pose (centroid, orientation) is defined in the T_fc frame; rotating xyz
            # into that frame does not move it, and the builders re-pose every unit from it
            raise ValueError('a pose needs cell as [B, 6] lengths and angles, with the pose and xyz in the T_fc '
                             'frame; [B, 3, 3] lattice rows rotate xyz but not the pose')

        # ---- per-row statuses, the first problem found naming the row
        status = [_STATUS['ok']] * B
        message = [''] * B
        for b, problem in enumerate(op_problems):
            if problem is not None:
                _flag(status, message, b, 'inconsistent_symmetry', problem)
        # atomic numbers must index the property tables (ATOM_WEIGHTS, VDW_RADII: 0-99, 0 unused)
        bad_z = ((z < 1) | (z >= len(ATOM_WEIGHTS))).to('cpu')
        bad_xyz = (~torch.isfinite(xyz).all(dim=1)).to('cpu')
        for code, bad_atom, what in (('bad_element', bad_z, f'atomic numbers outside 1-{len(ATOM_WEIGHTS) - 1}'),
                                     ('bad_coordinates', bad_xyz, 'non-finite coordinates')):
            per_row = torch.zeros(B, dtype=torch.long).index_add_(0, atom_row, bad_atom.long())
            for b in torch.nonzero(per_row).reshape(-1).tolist():
                s, e = int(ptr_cpu[b]), int(ptr_cpu[b + 1])
                local = torch.nonzero(bad_atom[s:e]).reshape(-1)[:5].tolist()
                _flag(status, message, b, code, f'{what} at row-local atom(s) {local}' +
                      (f' (values {z[s:e][local].tolist()})' if code == 'bad_element' else ''))
        atoms_ok = torch.zeros(B, dtype=torch.long).index_add_(0, atom_row, (bad_z | bad_xyz).long()) == 0

        # ---- the cell, and the frame of the coordinates
        cell64, xyz64, lattice_problems = _resolve_cell(cell, xyz, ptr_cpu, B)
        problems = _cell_problems(cell64)
        if lattice_problems is not None:
            problems = [lp or cp for lp, cp in zip(lattice_problems, problems)]
        for b, problem in enumerate(problems):
            if problem is not None:
                _flag(status, message, b, 'bad_cell', problem)

        # ---- attributes: declared levels, reserved names
        reserved = set(_COMPUTED) | (set(_COMPUTED_POSED) if posed else set())
        clash = sorted((set(attrs) | set(atom_attrs)) & reserved)
        if clash:
            raise ValueError(f'{clash} are computed by from_arrays and cannot be passed as attributes')
        both = sorted(set(attrs) & set(atom_attrs))
        if both:
            raise ValueError(f'{both} given both per crystal (attrs) and per atom (atom_attrs)')
        if UNIT_MAP_KEY in attrs:
            raise ValueError(f'{UNIT_MAP_KEY!r} is the unit map, one entry per atom: pass it in atom_attrs')
        unit_map = None
        if UNIT_MAP_KEY in atom_attrs:
            if not posed:
                # a bare batch's units are found by find_molecules, which re-orders the atoms of every unit; a map
                # given here would not follow it (parameterize_pose writes the map of the units it poses)
                raise ValueError(f'{UNIT_MAP_KEY!r} (the unit map) is taken only with a pose; the units of a bare '
                                 f'batch are found by find_molecules')
            unit_map = atom_attrs.pop(UNIT_MAP_KEY)
            unit_map = unit_map if torch.is_tensor(unit_map) else torch.as_tensor(np.asarray(unit_map))
            problem = unit_map_problem(unit_map, N)
            if problem is not None:
                raise ValueError(f'atom_attrs[{UNIT_MAP_KEY!r}] is not a unit map: {problem}')
            unit_map = unit_map.detach().to('cpu').long()
        offset = sorted(k for k in set(attrs) | set(atom_attrs) if any(s in k for s in _OFFSET_KEY_PARTS))
        if offset:
            # MXtalBase.__inc__ makes collate OFFSET these per row ('batch': by the row's max + 1,
            # 'index' / 'face': by its atom count), so whether a batch-level array arrives offset
            # or row-local cannot be told from the array; refused rather than guessed
            raise ValueError(f'{offset}: attribute names containing {list(_OFFSET_KEY_PARTS)} are offset '
                             'per row by collation, which from_arrays does not reproduce')

        device = torch.device(device)
        graph = {}
        node = {}
        pose = {k: attrs.pop(k) for k in POSE_KEYS} if posed else None
        for name, value in attrs.items():
            graph[name] = _declared_level(name, value, B, 'per-crystal (attrs)', device)
        for name, value in atom_attrs.items():
            node[name] = _declared_level(name, value, N, 'per-atom (atom_attrs)', device, ptr_cpu=ptr_cpu)

        pos = xyz64.to(dtype) if xyz64.dtype != dtype else xyz64
        out = _base_fields(cls, cell64=cell64, pos=pos, z=z, ptr_cpu=ptr_cpu, sg_cpu=sg_cpu, ops_list=ops_list,
                           setting_codes=setting_codes, dtype=dtype, device=device)
        if posed:
            _attach_pose(out, pose, num_atoms, ptr_cpu, atoms_ok, status, message, dtype, device, unit_map=unit_map)
        out['status'] = torch.tensor(status, dtype=torch.int8, device=device)
        out['message'] = message
        out['done'] = torch.full((B,), DONE_READ, dtype=torch.int16, device=device)
        for name, value in graph.items():
            out[name] = value
        for name, value in node.items():
            out[name] = value
        # per-atom python lists became per-row lists of lists: row level, as collate holds them
        node_keys = {k for k, v in node.items() if torch.is_tensor(v)} | _NODE_KEYS | (
            {UNIT_MAP_KEY} if posed else set())
        _write_slice_inc(out, ptr_cpu, node_keys=node_keys)
        return out

    def select_rows(self, idx):
        """A new batch of the rows ``idx``, each attribute split at its stored level.

        The row filter for batches from ``read`` and ``from_arrays``, e.g.
        ``batch.select_rows(batch.status == 0)``. It reads every attribute's level from
        ``_slice_dict`` (the level contract of ``MXtalBase``'s Notes, read from the
        bookkeeping alone): sliced by ``ptr``: per atom; by ``arange(B + 1)``: per row
        (when the two coincide, so do the gathers); an 'index' / 'face' tensor key
        (concatenated on its last axis) with an entry describing it and the increments
        collation writes (each row's atom offset): per edge, its rows' columns gathered
        and re-offset to the new atom offsets; a tensor whose entry is the unit pointer of
        ``z_prime``: per unit; any other tensor whose entry describes it: per edge (its own
        counts, values not offset). Stage 3 (2026-10-23) added the unit and edge levels;
        before, such attributes were refused. ``subsample_new_batch`` reads the entry only
        where a tensor's length does not decide its level (as many atoms as rows, as
        ``read`` reaches when failed rows hold no atoms; a one-row or one-atom batch;
        ragged and index values) and otherwise takes the level from the length, so it
        selects the same values and slices on these batches (its increments for list
        attributes are zeros where this method writes None). This method also checks the
        indices (in range, integer) and refuses an attribute stored at a level it cannot
        split, where ``subsample_new_batch`` uses indices as given.

        Parameters
        ----------
        idx : bool mask [B], or integer row indices
            A tensor, array or sequence; rows are kept in the order given, indices may
            repeat and must lie in [0, B).

        Returns
        -------
        MolCrystalDataBatch
            Same class and device, with ``batch``, ``ptr`` and the slice bookkeeping
            rebuilt for the selection. Attributes without a ``_slice_dict`` entry
            (``max_z_prime``) are carried over unchanged, as ``subsample_new_batch``
            carries them.

        Raises
        ------
        ValueError
            ``self`` is not a batch, an index is out of range or not an integer, a mask
            has the wrong length, or an attribute is stored at another level: an entry
            that does not describe it, non-zero collation increments on a key the helpers
            do not re-offset (a collated ``aunit_batch``), an 'index' / 'face' key whose
            increments are not its rows' atom offsets, or a unit-cell value
            (``data_classes.KEY_LEVELS``: ``UNIT_CELL``).
        """
        # the level machinery lives with MXtalBase (data_classes imports this module)
        from mxtaltools.dataset_utils.data_classes import (KEY_LEVELS, UNIT_CELL, _entry_pointer, _gather_segments,
                                                           _offset_by_atoms)
        if '_slice_dict' not in self.__dict__ or '_inc_dict' not in self.__dict__:
            raise ValueError('select_rows needs a batch with _slice_dict / _inc_dict bookkeeping')
        ptr_cpu = self.ptr.detach().to('cpu').long()
        B = len(ptr_cpu) - 1
        idx_cpu = _row_index(idx, B)
        device = self.ptr.device

        counts = ptr_cpu[idx_cpu + 1] - ptr_cpu[idx_cpu]
        new_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        k, n_new = len(idx_cpu), int(new_ptr[-1])
        new_row = torch.arange(k).repeat_interleave(counts)
        node_src = ptr_cpu[idx_cpu][new_row] + torch.arange(n_new) - new_ptr[:-1][new_row]
        row_slices, new_row_slices = torch.arange(B + 1), torch.arange(k + 1)
        zero_inc = torch.zeros(k, dtype=torch.long)

        out = type(self).__new__(type(self))
        out.__dict__['_store'] = GlobalStorage(_parent=out)
        slice_dict, inc_dict = {}, {}
        for key, value in self._store.items():
            if key in ('ptr', 'batch'):
                continue
            if KEY_LEVELS.get(key) == UNIT_CELL:
                raise ValueError(f'{key!r} is per unit-cell atom (build_unit_cell), a level select_rows does not '
                                 f'slice; delete the unit cell before selecting rows and rebuild it on the result')
            slices = self._slice_dict.get(key)
            if slices is None:
                out[key] = value
                continue
            inc = self._inc_dict.get(key)
            if torch.is_tensor(value) and value.ndim > 0 and self.__cat_dim__(key, value) == -1:
                # per edge, values atom indices: collation offset them by each row's atom offset; re-offset here
                pointer = _entry_pointer(self, key, value, slices, B)
                if pointer is None or not _offset_by_atoms(key):
                    raise ValueError(f'{key!r} {tuple(value.shape)} is concatenated on its last axis, but its slice '
                                     f'entry does not describe it (or its name also holds "batch")')
                if not (torch.is_tensor(inc) and torch.equal(inc.detach().to('cpu').long(), ptr_cpu[:-1])):
                    raise ValueError(f"{key!r}'s increments are not its rows' atom offsets (as collation writes "
                                     f'them for an index key), so its values cannot be re-offset')
                out[key], (slice_dict[key], inc_dict[key]) = _gather_segments(self, key, value, pointer, idx_cpu,
                                                                               ptr_cpu, new_ptr)
                continue
            if torch.is_tensor(inc) and bool((inc != 0).any()):
                raise ValueError(f'{key!r} carries collation increments; select_rows does not re-offset them')
            slices = slices.to('cpu')
            if torch.equal(slices, ptr_cpu):
                src, new_slices = node_src, new_ptr
            elif torch.equal(slices, row_slices):
                src, new_slices = idx_cpu, new_row_slices
            else:
                pointer = _entry_pointer(self, key, value, slices, B) if torch.is_tensor(value) else None
                if pointer is None:
                    raise ValueError(f'{key!r} is sliced neither per atom, per row, per unit nor per edge (its slice '
                                     f'entry does not describe it)')
                # per unit (the entry is the unit pointer of z_prime) or per edge (its own counts): either way the
                # selected rows' segments, and the result's entry is their pointer (the result's own unit pointer for
                # a per-unit value, as z_prime is selected with the rows)
                out[key], (slice_dict[key], inc_dict[key]) = _gather_segments(self, key, value, pointer, idx_cpu,
                                                                               ptr_cpu, new_ptr)
                continue
            if torch.is_tensor(value):
                out[key] = value[src.to(value.device)]
            elif isinstance(value, (list, tuple)):
                out[key] = type(value)(value[i] for i in src.tolist())
            else:
                raise ValueError(f'{key!r} is a {type(value).__name__} with a slice entry; cannot split it')
            slice_dict[key] = new_slices
            inc_dict[key] = zero_inc if torch.is_tensor(value) else None
        out['batch'] = new_row.to(device)
        out['ptr'] = new_ptr.to(device)
        out.__dict__['_num_graphs'] = k
        out.__dict__['_slice_dict'] = slice_dict
        out.__dict__['_inc_dict'] = inc_dict
        return out


###############################################################################
# reading
###############################################################################

def _check_float_dtype(dtype):
    """Raise TypeError unless ``dtype`` is a floating torch dtype."""
    if not (isinstance(dtype, torch.dtype) and dtype.is_floating_point):
        raise TypeError(f'dtype must be a floating torch dtype, got {dtype!r}')


def _as_source_list(sources) -> list:
    """``sources`` as a non-empty list of str / PathLike; raises on any other type."""
    if isinstance(sources, (str, os.PathLike)):
        out = [sources]
    elif isinstance(sources, (list, tuple)):
        out = list(sources)
    else:
        raise TypeError(f'sources must be a path, CIF text, or a list of them; got {type(sources).__name__}')
    if not out:
        raise ValueError('no sources given')
    for s in out:
        if not isinstance(s, (str, os.PathLike)):
            raise TypeError(f'a source must be a str or a path, got {type(s).__name__}')
    return out


_BOM = '﻿'


def _source_kind(src):
    """``('path', str path)`` or ``('text', str)``; see ``MolCrystalIO.read`` for the rule."""
    if isinstance(src, os.PathLike):
        return 'path', os.fspath(src)
    text = src.lstrip(_BOM)
    s = text.strip()
    if '\n' in s or '\r' in s:
        return 'text', text
    # one line. A path when it ends in .cif in any case (contract change (f): a '.CIF' path
    # is a path) or names anything on disk (os.path.exists is False, not an error, for a
    # path holding a NUL). Otherwise it is CIF text only if it can be: an inline block
    # starts with 'data_' and holds whitespace before its first tag; one-line text that
    # does not start with 'data_' has no block to parse, so reading it as a path loses
    # nothing and names the problem (no such file) correctly.
    if s.lower().endswith('.cif') or os.path.exists(s):
        return 'path', s
    if s[:5].lower() == 'data_' and any(ch.isspace() for ch in s):
        return 'text', text
    return 'path', s


def _read_cif_source(src, identifier: Optional[str], max_atomic_number) -> list:
    """Every row of one source: one per data block, or one failed row for the source."""
    kind, value = _source_kind(src)
    path = value if kind == 'path' else None
    fallback = identifier if identifier is not None else (Path(path).stem if path else '')
    if kind == 'path':
        try:
            # utf-8-sig drops a leading byte-order mark (Windows tools write one by default)
            text = Path(path).read_text(encoding='utf-8-sig', errors='ignore')
        except (OSError, ValueError) as e:          # ValueError: a path holding a NUL character
            return [_failed_row('unreadable_file', f'{path!r}: {type(e).__name__}: {e}', fallback, path, None)]
    else:
        text = value
    try:
        blocks = parse_cif(text)
    except CifParseError as e:
        return [_failed_row('parse_error', str(e), fallback, path, None)]

    rows = []
    for block in blocks:
        if identifier is not None:
            ident = identifier if len(blocks) == 1 else f'{identifier}:{block.name}'
        else:
            ident = block.name or block.get('_chemical_name_common') or fallback
        try:
            rows.append(_read_cif_block(block, ident, path, max_atomic_number))
        except _RowRefusal as e:
            rows.append(_failed_row(e.code, str(e), ident, path, block.name))
    return rows


def _failed_row(code, message, identifier, path, block_name) -> dict:
    """A row with a status, no atoms and no established symmetry or cell."""
    return dict(status=code, message=message, identifier=identifier, source_path=path, block_name=block_name,
                frac=np.zeros((0, 3)), z=np.zeros(0, dtype=np.int64), labels=[], occupancy=np.zeros(0),
                has_occupancy=False, cell=np.full(6, np.nan), ops=np.zeros((0, 4, 4)), sg=0,
                hm=None, hall=None, z_value=-1, bonds=None, moiety=None)


def _read_cif_block(block, ident, path, max_atomic_number) -> dict:
    """One data block -> one row dict; raises ``_RowRefusal`` for a row that cannot be read.

    An R group filed on rhombohedral axes is read from its hexagonal-axes re-expression
    (``cif/rhombohedral.py``), and the row's message says so.
    """
    block, reexpressed = _block_reexpressed(block, ident)
    symmetry = _block_symmetry(block, ident)
    sites = _block_sites(block, ident, max_atomic_number)
    cell = _block_cell(block, ident)
    z_value = _block_z_value(block, ident)
    bonds, n_image_bonds = _block_bonds(block, sites.labels, ident)

    status, notes = 'ok', []
    if reexpressed is not None:
        notes.append(f'[{ident}] {reexpressed}')
    if sites.has_occupancy:
        partial = np.abs(sites.occupancy - 1.0) > 1e-3
        if partial.any():
            status = 'partial_occupancy'
            bad = [sites.labels[i] for i in np.where(partial)[0][:5]]
            notes.append(f'[{ident}] {int(partial.sum())} site(s) carry partial occupancy (e.g. {bad}); '
                         'kept with their occupancies, later steps skip the row')
    if n_image_bonds:
        notes.append(f'[{ident}] {n_image_bonds} _geom_bond row(s) carry a site-symmetry code other than 1_555 or '
                     "'.' (a symmetry image, or '?') and are not in file_bonds")

    return dict(status=status, message='; '.join(notes), identifier=ident, source_path=path,
                block_name=block.name, frac=sites.frac, z=sites.z, labels=list(sites.labels),
                occupancy=sites.occupancy, has_occupancy=bool(sites.has_occupancy), cell=cell,
                ops=np.stack([np.asarray(o, dtype=np.float64) for o in symmetry.operators]), sg=int(symmetry.number),
                hm=_scalar_symbol(block, _HM_TAGS),
                hall=_scalar_symbol(block, ('_space_group_name_Hall', '_symmetry_space_group_name_Hall')),
                z_value=z_value, bonds=bonds, moiety=_moiety(block))


def _moiety(block):
    """``_chemical_formula_moiety`` as the native reader reads it (``block.get``), None when absent or missing."""
    value = block.get(_MOIETY)
    return None if is_missing(value) else value


def _scalar_symbol(block, tags):
    """The first present tag's value with whitespace and quotes stripped, or None when absent or missing."""
    _, value = block.first_of(*tags)
    if value is None:
        return None
    value = value.strip().strip("'\"")
    return None if is_missing(value) else value


def _block_reexpressed(block, ident):
    """``(block, note)`` from ``reexpress_rhombohedral_axes``; its refusal is ``inconsistent_symmetry``."""
    try:
        return reexpress_rhombohedral_axes(block, ident)
    except CifInconsistentSymmetryError as e:
        raise _RowRefusal('inconsistent_symmetry', str(e))


def _block_symmetry(block, ident):
    """``SymmetryInfo`` of the block, its refusals mapped to status codes."""
    if not _has_operator_rows(block):
        raise _RowRefusal(
            'no_operators', f'[{ident}] no symmetry operators: none of {_OP_TAGS} holds a value. Operators are '
                            'never guessed from the symbol or the IT number')
    try:
        return space_group_from_block(block, ident)
    except CifInconsistentSymmetryError as e:
        raise _RowRefusal('inconsistent_symmetry', str(e))
    except CifParseError as e:                      # a non-integer IT number
        raise _RowRefusal('parse_error', str(e))
    except CifSymmetryError as e:
        if _scalar_symbol(block, _NUM_TAGS) is None and _scalar_symbol(block, _HM_TAGS) is None:
            raise _RowRefusal('missing_data', str(e))
        # an unparseable operator, no identity, an unknown symbol, a number outside 1-230
        raise _RowRefusal('inconsistent_symmetry', str(e))
    except (ValueError, OverflowError) as e:        # an IT number reading as NaN or infinity
        raise _RowRefusal('parse_error', f'[{ident}] {type(e).__name__}: {e}')


def _has_operator_rows(block) -> bool:
    """True when an operator loop with rows, or a non-missing scalar operator, is present.

    Mirrors the lookup order of ``symops_from_block``: the first loop carrying one of
    ``_OP_TAGS`` decides, else a scalar tag.
    """
    for tag in _OP_TAGS:
        loop = block.loop_with(tag)
        if loop is not None:
            return len(loop) > 0
    _, scalar = block.first_of(*_OP_TAGS)
    return scalar is not None and not is_missing(scalar)


def _block_sites(block, ident, max_atomic_number):
    """The deposited set (``SiteSet``), partial occupancy kept, refusals mapped to status codes."""
    try:
        sites = sites_from_block(block, ident, max_atomic_number=max_atomic_number,
                                 refuse_partial_occupancy=False)
    except CifMissingDataError as e:
        raise _RowRefusal('missing_data', str(e))
    except CifUnsupportedError as e:                # reason 'atomic-number'
        raise _RowRefusal('bad_element' if e.reason == 'atomic-number' else 'parse_error', str(e))
    except CifParseError as e:
        # sites_from_block raises CifParseError for exactly two things: duplicate labels
        # (checked first), and a type symbol or label naming no element. Told apart from the
        # data, not the message, which carries the block's identifier
        raise _RowRefusal('duplicate_label' if _has_duplicate_labels(block) else 'bad_element', str(e))
    except (ValueError, OverflowError) as e:        # a non-numeric coordinate or occupancy
        raise _RowRefusal('parse_error', f'[{ident}] {type(e).__name__}: {e}')

    # what float() and the symbol table accept but no atom site can hold: ASE's placeholder
    # 'X' (atomic number 0), and 'nan' / 'inf' / '1e400' as a coordinate or occupancy
    dummy = np.nonzero(sites.z < 1)[0]
    if len(dummy):
        raise _RowRefusal('bad_element', f'[{ident}] site(s) {[sites.labels[i] for i in dummy[:5]]} have atomic '
                                         'number 0 (a placeholder symbol such as X)')
    bad = np.nonzero(~np.isfinite(sites.frac).all(1) | ~np.isfinite(sites.occupancy))[0]
    if len(bad):
        raise _RowRefusal('parse_error', f'[{ident}] non-finite coordinate or occupancy at site(s) '
                                         f'{[sites.labels[i] for i in bad[:5]]}')
    return sites


def _has_duplicate_labels(block) -> bool:
    """Whether the atom-site loop ``sites_from_block`` reads labels from repeats one (its own test)."""
    loop = block.loop_with(_LABEL_TAG)
    if loop is None:
        return False                                # labels are generated, so unique
    labels = [v.strip() for v in loop.column(_LABEL_TAG)]
    return len(set(labels)) != len(labels)


def _block_cell(block, ident) -> np.ndarray:
    """[6] float64: lengths as stated, angles in radians; refusals mapped to status codes."""
    values = []
    for tag in _CELL_LENGTH_TAGS + _CELL_ANGLE_TAGS:
        try:
            v = as_float(block.get(tag), tag, ident)
        except CifParseError as e:
            raise _RowRefusal('parse_error', str(e))
        if v is None:
            raise _RowRefusal('missing_data', f'[{ident}] {tag} is absent')
        values.append(v)
    cell = np.array(values, dtype=np.float64)
    cell[3:] = np.radians(cell[3:])
    problem = _cell_problems(torch.as_tensor(cell[None]))[0]
    if problem is not None:
        raise _RowRefusal('bad_cell', f'[{ident}] {problem}')
    return cell


def _block_z_value(block, ident) -> int:
    """``_cell_formula_units_Z``, or -1 when absent; a malformed value is a ``parse_error``."""
    tag, raw = block.first_of(*_Z_TAGS)
    try:
        value = as_int(raw, tag or _Z_TAGS[0], ident)
    except CifParseError as e:
        raise _RowRefusal('parse_error', str(e))
    except (ValueError, OverflowError) as e:
        raise _RowRefusal('parse_error', f'[{ident}] {tag} {raw!r}: {type(e).__name__}: {e}')
    if value is None:
        return -1
    if value <= 0:
        raise _RowRefusal('parse_error', f'[{ident}] {tag} is {value}, not a count')
    return value


#: Site-symmetry codes of a bond within the deposited set: the identity, and '.' (the CIF default).
_SAME_SITE_CODES = ('1_555', '.')


def _block_bonds(block, labels, ident):
    """``(bonds, n_image)``: the ``_geom_bond`` pairs as a long [2, E] array of site indices, or None.

    A bond is kept when both site-symmetry codes are ``1_555`` or ``.`` (the CIF default:
    SHELXL writes ``.`` for a bond within the deposited set), or the columns are absent.
    Any other code (a symmetry image, or ``?``) is counted, not kept. Unlike
    ``components_from_block``, which also drops ``.``: that rule was measured on CSD
    exports only, which write ``1_555``. A bond naming an absent site, or a loop without
    its second label column, is a ``parse_error``: the file contradicts itself.
    """
    loop = block.loop_with(_BOND_1)
    if loop is None:
        return None, 0
    if _BOND_2 not in loop:
        raise _RowRefusal('parse_error', f'[{ident}] bond loop has {_BOND_1} but no {_BOND_2}')
    index_of = {lab: i for i, lab in enumerate(labels)}
    l1, l2 = loop.column(_BOND_1), loop.column(_BOND_2)
    s1 = loop.column(_BOND_SYM_1) if _BOND_SYM_1 in loop else ['1_555'] * len(l1)
    s2 = loop.column(_BOND_SYM_2) if _BOND_SYM_2 in loop else ['1_555'] * len(l2)
    edges, n_image = [], 0
    for a, b, ca, cb in zip(l1, l2, s1, s2):
        a, b = a.strip(), b.strip()               # a trailing apostrophe is part of a label
        if a not in index_of or b not in index_of:
            raise _RowRefusal('parse_error', f'[{ident}] bond references site '
                                             f'{a if a not in index_of else b!r}, which is not in the atom-site loop')
        if ca.strip() not in _SAME_SITE_CODES or cb.strip() not in _SAME_SITE_CODES:
            n_image += 1
            continue
        edges.append((index_of[a], index_of[b]))
    return np.array(edges, dtype=np.int64).reshape(-1, 2).T.copy(), n_image


def _rows_to_batch(cls, rows, dtype, device):
    """Assemble ``read`` rows into one bare batch."""
    B = len(rows)
    counts = torch.tensor([len(r['z']) for r in rows], dtype=torch.long)
    ptr_cpu = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    cell64 = torch.as_tensor(np.stack([r['cell'] for r in rows]), dtype=torch.float64)
    sg_cpu = torch.tensor([r['sg'] for r in rows], dtype=torch.long)

    # Cartesian deposited set: frac @ T_fc^T per row, in float64 from the float64 cell
    frac = torch.as_tensor(np.concatenate([r['frac'] for r in rows]), dtype=torch.float64)
    T_fc64, _, _ = batch_compute_fractional_transform(cell64[:, :3], cell64[:, 3:])
    atom_row = torch.arange(B).repeat_interleave(counts)
    pos = torch.einsum('nij,nj->ni', T_fc64[atom_row], frac).to(dtype)

    ops_list = [r['ops'] for r in rows]
    setting_codes = [_classify_setting(o, s) if len(o) else -1 for o, s in zip(ops_list, sg_cpu.tolist())]
    z = torch.as_tensor(np.concatenate([r['z'] for r in rows]), dtype=torch.long)

    device = torch.device(device)
    out = _base_fields(cls, cell64=cell64, pos=pos, z=z, ptr_cpu=ptr_cpu, sg_cpu=sg_cpu, ops_list=ops_list,
                       setting_codes=setting_codes, dtype=dtype, device=device)
    out['status'] = torch.tensor([_STATUS[r['status']] for r in rows], dtype=torch.int8, device=device)
    out['message'] = [r['message'] for r in rows]
    out['done'] = torch.full((B,), DONE_READ, dtype=torch.int16, device=device)
    out['identifier'] = [r['identifier'] for r in rows]
    out['source_path'] = [r['source_path'] for r in rows]
    out['block_name'] = [r['block_name'] for r in rows]
    out['z_value'] = torch.tensor([r['z_value'] for r in rows], dtype=torch.long, device=device)
    out['hm_symbol'] = [r['hm'] for r in rows]
    out['hall_symbol'] = [r['hall'] for r in rows]
    out['moiety'] = [r['moiety'] for r in rows]
    out['atom_labels'] = [r['labels'] for r in rows]
    out['occupancy'] = torch.as_tensor(np.concatenate([r['occupancy'] for r in rows]), dtype=dtype).to(device)
    out['has_occupancy'] = torch.tensor([r['has_occupancy'] for r in rows], dtype=torch.bool, device=device)
    out['file_bonds'] = [r['bonds'] for r in rows]
    _write_slice_inc(out, ptr_cpu, node_keys=_NODE_KEYS | {'occupancy'})
    return out


###############################################################################
# arrays
###############################################################################

def _flag(status: list, message: list, b: int, code: str, text: str):
    """Give row ``b`` the status ``code`` unless an earlier check already named the row."""
    if status[b] == _STATUS['ok']:
        status[b], message[b] = _STATUS[code], text


def _is_integer_tensor(t: torch.Tensor) -> bool:
    """An integer dtype that is not bool."""
    return not (t.is_floating_point() or t.is_complex() or t.dtype == torch.bool)


def _as_ptr(ptr) -> torch.Tensor:
    """``ptr`` as a host long tensor, validated."""
    ptr_cpu = torch.as_tensor(ptr).detach().to('cpu')
    if not _is_integer_tensor(ptr_cpu):
        raise ValueError(f'ptr must hold integers (not {ptr_cpu.dtype})')
    if ptr_cpu.ndim != 1:
        raise ValueError(f'ptr must be one-dimensional [B + 1], got shape {tuple(ptr_cpu.shape)}')
    ptr_cpu = ptr_cpu.long()
    if len(ptr_cpu) < 2:
        raise ValueError(f'ptr must have B + 1 >= 2 entries, got {len(ptr_cpu)}')
    if int(ptr_cpu[0]) != 0 or bool((ptr_cpu[1:] < ptr_cpu[:-1]).any()):
        raise ValueError('ptr must start at 0 and be non-decreasing')
    return ptr_cpu


def _as_sg(sg, B) -> torch.Tensor:
    """Space-group numbers as a host long tensor [B], validated (1-230)."""
    sg_cpu = torch.as_tensor(sg).detach().to('cpu')
    if not _is_integer_tensor(sg_cpu):
        raise ValueError(f'sg must hold integers (not {sg_cpu.dtype})')
    sg_cpu = sg_cpu.long()
    if sg_cpu.ndim == 0:
        sg_cpu = sg_cpu.repeat(B)
    sg_cpu = sg_cpu.reshape(-1)
    if len(sg_cpu) != B:
        raise ValueError(f'sg must be one int or [B] with B = {B}, got {len(sg_cpu)} values')
    bad = sorted({int(s) for s in sg_cpu if int(s) not in SYM_OPS})
    if bad:
        raise ValueError(f'space-group numbers {bad} are outside SYM_OPS (1-230)')
    return sg_cpu


def _resolve_operators(ops, setting, sg_cpu, B):
    """``(ops_list, setting_codes, problems)``: float64 [m, 4, 4] arrays per row, their
    ``setting`` codes, and per row None or why the operators are not a usable set."""
    if setting is not None and setting != 'standard':
        raise ValueError(f"setting must be None or 'standard', got {setting!r}")
    if setting == 'standard':
        if ops is not None:
            raise ValueError("pass ops or setting='standard', not both")
        return [np.stack(SYM_OPS[int(s)]) for s in sg_cpu], [0] * B, [None] * B
    if ops is None:
        raise ValueError("ops are required unless setting='standard': operators are never regenerated "
                         'from the space-group number by default (it moved ELJ on nonstandard rows)')

    if isinstance(ops, (list, tuple)):
        per_row = list(ops)
    else:
        arr = ops.detach().cpu().numpy() if torch.is_tensor(ops) else np.asarray(ops)
        if arr.ndim == 3:
            per_row = [arr] * B
        elif arr.ndim == 4:
            per_row = list(arr)
        else:
            raise ValueError(f'ops must be [m, 4, 4], [B, m, 4, 4] or a list of [m, 4, 4]; got shape {arr.shape}')
    if len(per_row) != B:
        raise ValueError(f'ops must give one operator array per row ({B}), got {len(per_row)}')
    out = []
    for b, o in enumerate(per_row):
        o = np.asarray(o.detach().cpu().numpy() if torch.is_tensor(o) else o, dtype=np.float64)
        if o.ndim != 3 or o.shape[1:] != (4, 4) or len(o) == 0:
            raise ValueError(f'row {b}: operators must be [m >= 1, 4, 4], got shape {o.shape}')
        out.append(o)
    sgs = [int(s) for s in sg_cpu]
    return out, [_classify_setting(o, s) for o, s in zip(out, sgs)], [_operator_problem(o, s) for o, s in zip(out, sgs)]


def _operator_problem(ops: np.ndarray, sg: int) -> Optional[str]:
    """None, or why ``ops`` [m, 4, 4] is not a usable operator set for space group ``sg``.

    The checks ``read`` applies to a file's operators (``symops_from_block``,
    ``space_group_from_block``), plus the two a parsed operator cannot fail: finite
    entries and an affine last row. The count must equal ``len(SYM_OPS[sg])``, or be that
    times the number of pure translations with the set factorising as the standard
    operators composed with them (a centred setting), by the reader's own helpers.
    """
    if not np.isfinite(ops).all():
        return 'operators hold non-finite entries'
    if np.abs(ops[:, 3] - np.array([0, 0, 0, 1.0])).max() > _OP_TOL:
        return 'operators are not affine: a last row is not [0, 0, 0, 1]'
    if not any(np.allclose(o, np.eye(4), atol=_OP_TOL, rtol=0) for o in ops):
        return 'operators do not contain the identity'
    expected = len(SYM_OPS[sg])
    if len(ops) != expected:
        op_list = list(ops)
        centring = _centring_vectors(op_list)
        if not (len(ops) == expected * len(centring) and _factorises_as_centred(op_list, SYM_OPS[sg], centring)):
            return (f'{len(ops)} operators do not fit space group {sg}: it has {expected}, and they do not '
                    f'factorise as a centred setting ({len(centring)} centring vector(s))')
    return None


_STD_OPS_CACHE = {}


def _classify_setting(ops: np.ndarray, sg: int) -> int:
    """0, 1 or 2 (see ``SETTINGS``) for operators ``ops`` [m, 4, 4] against ``SYM_OPS[sg]``."""
    if sg not in SYM_OPS:
        return 2
    std = _STD_OPS_CACHE.get(sg)
    if std is None:
        std = _STD_OPS_CACHE.setdefault(sg, np.stack(SYM_OPS[sg]).astype(np.float64))
    if len(ops) != len(std):
        return 2
    if np.allclose(ops, std, atol=_OP_TOL, rtol=0):
        return 0
    # same set of symmetry elements: equal rotations, translations equal modulo 1, a bijection
    if np.abs(ops[:, 3] - np.array([0, 0, 0, 1.0])).max() > _OP_TOL:
        return 2
    same_rot = np.abs(ops[:, None, :3, :3] - std[None, :, :3, :3]).max((-1, -2)) <= _OP_TOL
    dt = ops[:, None, :3, 3] - std[None, :, :3, 3]
    same_t = np.abs(dt - np.round(dt)).max(-1) <= _OP_TOL
    match = same_rot & same_t
    return 1 if bool((match.sum(1) == 1).all() and (match.sum(0) == 1).all()) else 2


def _resolve_cell(cell, xyz, ptr_cpu, B):
    """``(cell64 [B, 6], xyz64 [N, 3], lattice_problems)`` in float64.

    For [B, 3, 3] lattice rows, ``xyz`` is rotated per row by ``R = T_fc @ inv(L^T)``,
    the rotation taking the caller's lattice vectors onto the ``T_fc`` frame; a row
    whose lattice is not right-handed is left unrotated and gets a problem string.
    """
    cell = torch.as_tensor(cell)
    xyz64 = xyz.to(torch.float64)
    if cell.ndim == 2 and cell.shape == (B, 6):
        return cell.to(torch.float64), xyz64, None
    if not (cell.ndim == 3 and cell.shape == (B, 3, 3)):
        raise ValueError(f'cell must be [B, 6] or [B, 3, 3] with B = {B}, got {tuple(cell.shape)}')

    L = cell.to(torch.float64)
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    lengths = torch.stack([a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)], dim=-1)

    def angle(u, v):
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
        return torch.arccos(cos.clamp(-1, 1))

    cell64 = torch.cat([lengths, torch.stack([angle(b, c), angle(a, c), angle(a, b)], dim=-1)], dim=-1)
    T_fc64, _, _ = batch_compute_fractional_transform(cell64[:, :3], cell64[:, 3:])
    det = torch.linalg.det(L).detach()
    problems = [None if bool(d > 0) else f'lattice rows are not right-handed (det {float(d):.4g}); '
                                         'coordinates left unrotated' for d in det]
    good = det > 0
    R = torch.eye(3, dtype=torch.float64, device=L.device).repeat(B, 1, 1)
    if bool(good.any()):
        # R @ L^T = T_fc  ->  R = T_fc @ inv(L^T), via a solve on the transpose: L @ R^T = T_fc^T.
        # Both bases share lengths and angles, so R is orthogonal, and det R = +1 for det L > 0.
        R[good] = torch.linalg.solve(L[good], T_fc64[good].transpose(1, 2)).transpose(1, 2)
    atom_row = torch.arange(B).repeat_interleave(ptr_cpu[1:] - ptr_cpu[:-1]).to(xyz64.device)
    xyz64 = torch.einsum('nij,nj->ni', R.to(xyz64.device)[atom_row], xyz64)
    return cell64, xyz64, problems


def _cell_problems(cell64: torch.Tensor) -> list:
    """Per row of [B, 6] (radians): None, or why the six numbers are not a cell."""
    out = []
    for row in cell64.detach().to('cpu', torch.float64).numpy():
        lengths, angles = row[:3], row[3:]
        if not np.isfinite(row).all():
            out.append(f'non-finite cell parameters {row.tolist()}')
        elif (lengths <= 0).any():
            out.append(f'cell lengths {lengths.tolist()} are not all positive')
        elif ((angles <= 0) | (angles >= np.pi)).any():
            out.append(f'cell angles {np.degrees(angles).tolist()} degrees are not all in (0, 180)')
        else:
            ca = np.cos(angles)
            f = 1 - (ca ** 2).sum() + 2 * ca.prod()
            out.append(None if f > 0 else f'cell angles {np.degrees(angles).tolist()} degrees describe no '
                                          f'real cell (1 - sum cos^2 + 2 prod cos = {f:.3g})')
    return out


def _declared_level(name, value, n, level, device, ptr_cpu=None):
    """An attribute stored at its declared level: leading dimension ``n``, as collate would store it.

    With ``ptr_cpu`` (per-atom attributes), a list that does not become a tensor is split
    into one list per row: collate holds rows' lists as a list[B] at row level, and PyG's
    row access indexes a stored list by row, so a flat per-atom list would hand row i the
    i-th atom's value.
    """
    if torch.is_tensor(value):
        if value.ndim == 0 or value.shape[0] != n:
            raise ValueError(f'{level} attribute {name!r} needs leading dimension {n}, got {tuple(value.shape)}')
        return value.to(device)
    if isinstance(value, np.ndarray) and value.dtype != object:
        if value.ndim == 0 or value.shape[0] != n:
            raise ValueError(f'{level} attribute {name!r} needs leading dimension {n}, got {value.shape}')
        return torch.as_tensor(value).to(device)
    if isinstance(value, (list, tuple, np.ndarray)):
        value = list(value)
        if len(value) != n:
            raise ValueError(f'{level} attribute {name!r} needs {n} entries, got {len(value)}')
        # python numbers become a tensor, as PyG's collate turns them (bool -> bool, int -> long,
        # float -> the default float dtype); anything else stays a list
        if value and all(isinstance(v, bool) for v in value):
            return torch.tensor(value, dtype=torch.bool, device=device)
        if value and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return torch.tensor(value, device=device)
        if ptr_cpu is not None:
            bounds = ptr_cpu.tolist()
            return [value[bounds[b]:bounds[b + 1]] for b in range(len(bounds) - 1)]
        return value
    raise ValueError(f'{level} attribute {name!r} must be a tensor, array, list or tuple with leading '
                     f'dimension {n}; got {type(value).__name__}')


###############################################################################
# shared assembly
###############################################################################

#: Per-atom keys the shared assembly writes (everything else written here is per crystal).
_NODE_KEYS = frozenset({'z', 'pos', 'x', 'mol_ind'})


_BATCH_CLASSES = {}


def _batch_class(cls):
    """The PyG batch class of ``cls`` (``MolCrystalDataBatch``), as ``collate_data_list`` creates it.

    PyG creates it on first use and caches it by name, so this is the same class object a
    collated batch has; one throwaway instance is built per ``cls`` to get it.
    """
    from torch_geometric.data import Batch
    if issubclass(cls, Batch):
        return cls
    if cls not in _BATCH_CLASSES:
        _BATCH_CLASSES[cls] = type(Batch(_base_cls=cls))
    return _BATCH_CLASSES[cls]


def _base_fields(cls, *, cell64, pos, z, ptr_cpu, sg_cpu, ops_list, setting_codes, dtype, device):
    """A new batch holding the fields every bare batch has (no slice bookkeeping yet)."""
    batch_cls = _batch_class(cls)
    out = batch_cls.__new__(batch_cls)
    out.__dict__['_store'] = GlobalStorage(_parent=out)
    B = len(ptr_cpu) - 1
    counts = ptr_cpu[1:] - ptr_cpu[:-1]

    out['z'] = z.to(device=device, dtype=torch.long)
    out['pos'] = pos.to(device)
    out['num_atoms'] = counts.to(device)
    cell = cell64.to(dtype).to(device)
    out['cell_lengths'] = cell[:, :3].contiguous()
    out['cell_angles'] = cell[:, 3:].contiguous()
    # transforms from the STORED (dtype) cell, as box_analysis computes them on stored rows
    out['T_fc'], out['T_cf'], out['cell_volume'] = batch_compute_fractional_transform(out['cell_lengths'],
                                                                                      out['cell_angles'])
    out['sg_ind'] = sg_cpu.to(device)
    out['symmetry_operators'] = list(ops_list)
    out['sym_mult'] = torch.tensor([len(o) for o in ops_list], dtype=torch.long, device=device)
    setting = torch.tensor(setting_codes, dtype=torch.int8, device=device)
    out['setting'] = setting
    out['nonstandard_symmetry'] = setting != 0
    out['batch'] = torch.arange(B, device=device).repeat_interleave(counts.to(device))
    out['ptr'] = ptr_cpu.to(device)
    out.__dict__['_num_graphs'] = B
    return out


def _attach_pose(out, pose, num_atoms, ptr_cpu, atoms_ok, status, message, dtype, device, unit_map=None):
    """Write the posed-state fields onto ``out``; marks ``z_prime_mismatch`` rows in ``status`` / ``message``.

    ``atoms_ok`` [B] bool: the row's atomic numbers index the property tables and its
    coordinates are finite, so its units can be measured. ``unit_map`` [N] long (host), the
    caller's ``atom_attrs['unit_ind']`` when given: stored as it is, and a row whose map is
    not its Z' equal contiguous blocks in unit order (the only units at this stage) is a
    ``z_prime_mismatch`` row.
    """
    B = len(num_atoms)
    z_prime = torch.as_tensor(pose['z_prime']).detach().to('cpu')
    if z_prime.dtype == torch.bool or z_prime.is_complex() or (
            z_prime.is_floating_point() and not (bool(torch.isfinite(z_prime).all())
                                                 and torch.equal(z_prime, z_prime.round()))):
        raise ValueError('z_prime must hold integers')
    z_prime = z_prime.long().reshape(-1)
    if len(z_prime) != B:
        raise ValueError(f'z_prime needs {B} entries, got {len(z_prime)}')
    fits = (z_prime >= 1) & (num_atoms >= z_prime) & (num_atoms % z_prime.clamp(min=1) == 0)
    for b in torch.nonzero(~fits).reshape(-1).tolist():
        _flag(status, message, b, 'z_prime_mismatch',
              f"{int(num_atoms[b])} atoms do not split into Z' = {int(z_prime[b])} equal units")
    N = int(ptr_cpu[-1])
    if unit_map is not None:
        # the caller's map against the units the pose describes: Z' equal contiguous blocks, block k unit k
        atom_row = torch.arange(B).repeat_interleave(num_atoms)
        blocks = equal_block_unit_ind(num_atoms, z_prime, fits, total=N)
        other = torch.zeros(B, dtype=torch.long).index_add_(0, atom_row, (unit_map != blocks).long()) > 0
        for b in torch.nonzero(fits & other).reshape(-1).tolist():
            s, e = int(ptr_cpu[b]), int(ptr_cpu[b + 1])
            u = unit_map[s:e]
            sizes = torch.bincount(u.clamp(min=0)).tolist() if bool((u >= 0).all()) else None
            _flag(status, message, b, 'z_prime_mismatch',
                  f"its unit map (unit_ind) is not Z' = {int(z_prime[b])} equal contiguous blocks in unit order"
                  + (f' (units of {sizes} atoms; units of different sizes are refused until a later stage)'
                     if sizes is not None and len(set(sizes)) > 1 else ''))
        fits = fits & ~other
        out[UNIT_MAP_KEY] = unit_map.to(device)
    else:
        # the units the pose describes, written as the map (stage 3, 2026-10-23; before, none was written and readers
        # derived the same layout): Z' equal contiguous blocks, block k unit k, on the rows whose atoms split into them,
        # and NO_UNIT on the others (z_prime_mismatch rows), so every reader refuses those rows as it did without a map
        rows_of_atoms = torch.arange(B).repeat_interleave(num_atoms)
        blocks = equal_block_unit_ind(num_atoms, z_prime, fits, total=N)
        out[UNIT_MAP_KEY] = torch.where(fits[rows_of_atoms], blocks, torch.full_like(blocks, NO_UNIT)).to(device)
    # the slot count comes from the rows whose atoms split into units, so one bad Z' neither
    # widens every row's pose (consumers dispatch on max_z_prime) nor makes the call raise
    max_zp = int(z_prime[fits].max()) if bool(fits.any()) else 1

    # pose arrays: [B, >= width], truncated to the batch's slots; slots past a row's Z',
    # and every slot of a row whose atoms do not split, zeroed
    slot_used = (torch.arange(max_zp)[None, :] < z_prime[:, None]) & fits[:, None]      # [B, max_zp]
    for key, per_slot in (('aunit_centroid', 3), ('aunit_orientation', 3), ('aunit_handedness', 1)):
        v = torch.as_tensor(pose[key])
        if v.ndim == 1 and per_slot == 1 and max_zp == 1:
            v = v[:, None]
        if v.ndim != 2 or v.shape[0] != B or v.shape[1] < per_slot * max_zp:
            raise ValueError(f'{key} must be [B, >= {per_slot * max_zp}] with B = {B} '
                             f"(max Z' = {max_zp}), got {tuple(v.shape)}")
        v = v[:, :per_slot * max_zp].to(device)
        if key != 'aunit_handedness':
            v = v.to(dtype)
        mask = slot_used.repeat_interleave(per_slot, dim=1).to(device)
        out[key] = torch.where(mask, v, torch.zeros_like(v))
    out['z_prime'] = z_prime.to(device)
    out['max_z_prime'] = max_zp

    # units: Z' equal contiguous blocks per row, the layout of a row without a unit map (the one derivation,
    # crystal_building.utils.equal_block_unit_ind; 0 on a row whose atoms do not split, or whose given map is not
    # those blocks). No unit_ind is written unless the caller gave one: readers derive this same layout from
    # num_atoms and z_prime (unit_layout), and mol_ind keeps its old meaning
    atom_row = torch.arange(B).repeat_interleave(num_atoms)
    mol_ind = equal_block_unit_ind(num_atoms, z_prime, fits, total=N)
    out['mol_ind'] = mol_ind.to(device)

    # per-unit properties only where the units exist and their atoms can be measured: an
    # atomic number outside the tables would index past them (or wrap, if negative)
    measured = fits & atoms_ok
    units_per_row = torch.where(measured, z_prime, torch.zeros_like(z_prime))
    unit_ptr = torch.cat([units_per_row.new_zeros(1), units_per_row.cumsum(0)])
    n_units = int(unit_ptr[-1])
    if n_units > 0:
        # per unit, with MolData.mol_analysis's definitions (batch path)
        atom_sel = measured[atom_row]
        unit_of_atom = (unit_ptr[:-1][atom_row] + mol_ind)[atom_sel].to(device)
        sel = atom_sel.to(device)
        zz, pp = out['z'][sel], out['pos'][sel]
        heavy = zz > 1
        heavies_per_unit = scatter(heavy.long(), unit_of_atom, dim=0, dim_size=n_units, reduce='sum')
        radius_u = batch_compute_mol_radius(pp[heavy], unit_of_atom[heavy], n_units, heavies_per_unit)
        masses = torch.tensor(list(ATOM_WEIGHTS.values()), device=device, dtype=torch.float32)
        mass_u = batch_compute_mol_mass(zz, unit_of_atom, masses, n_units)
        vdw = torch.tensor(list(VDW_RADII.values()), device=device, dtype=torch.float32)
        volume_u = batch_compute_molecule_volume(zz, pp, unit_of_atom, n_units, vdw)
    else:
        radius_u = mass_u = volume_u = torch.zeros(0, dtype=dtype, device=device)

    # row sums over units (the latent's and box_analysis's convention); NaN where the
    # units are not measured
    unit_row = torch.arange(B, device=device).repeat_interleave(units_per_row.to(device))
    for key, per_unit_values in (('radius', radius_u), ('mass', mass_u), ('mol_volume', volume_u)):
        summed = scatter(per_unit_values.to(dtype), unit_row, dim=0, dim_size=B, reduce='sum')
        out[key] = torch.where(measured.to(device), summed, torch.full_like(summed, float('nan')))
    out['x'] = torch.zeros(N, dtype=dtype, device=device)


def _row_index(idx, B) -> torch.Tensor:
    """Row indices [k] (host long) from a bool mask [B] or integer indices, validated."""
    t = torch.as_tensor(idx).detach().to('cpu').reshape(-1)
    if t.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    if t.dtype == torch.bool:
        if len(t) != B:
            raise ValueError(f'a row mask needs {B} entries, got {len(t)}')
        return torch.nonzero(t).reshape(-1)
    if not _is_integer_tensor(t):
        raise ValueError(f'row indices must be integers or a bool mask, got {t.dtype}')
    t = t.long()
    if bool((t < 0).any()) or bool((t >= B).any()):
        raise ValueError(f'row indices must lie in [0, {B}), got {t.tolist()[:10]}')
    return t


def _write_slice_inc(out, ptr_cpu, node_keys):
    """``_slice_dict`` / ``_inc_dict`` from declared levels, as ``collate_data_list`` writes them.

    Per atom: slices ``ptr``; per crystal: ``arange(B + 1)``. Increments are long zeros
    [B] for tensors and None for python lists (collate's value for non-tensor
    sequences). ``batch``, ``ptr`` and ``max_z_prime`` (a batch-level python int) get no
    entry. All on the host, as collate leaves them.
    """
    B = len(ptr_cpu) - 1
    graph_slices = torch.arange(B + 1, dtype=torch.long)
    zero_inc = torch.zeros(B, dtype=torch.long)
    slice_dict, inc_dict = {}, {}
    for key, value in out._store.items():
        if key in ('batch', 'ptr', 'max_z_prime'):
            continue
        slice_dict[key] = ptr_cpu if key in node_keys else graph_slices
        inc_dict[key] = zero_inc if torch.is_tensor(value) else None
    out.__dict__['_slice_dict'] = slice_dict
    out.__dict__['_inc_dict'] = inc_dict

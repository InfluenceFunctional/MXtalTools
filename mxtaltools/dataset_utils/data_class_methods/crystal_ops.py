"""Crystal-parameter operations for molecular crystal data objects.

Defines :class:`MolCrystalOps`, the mixin that gives ``MolCrystalData`` (and its
collated batches) its cell-parameter methods: symmetry and molecule attribute
setup, box analysis, random sampling, the latent <-> cell-parameter transforms,
standardization, cleaning and validation, canonicalization, spglib cell
standardization, plotting and CIF output. Also defines :func:`spglib_hall_number`.
"""
import functools
from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from plotly.subplots import make_subplots

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.ase_interface import ase_write_cif
from mxtaltools.common.geometry_utils import enforce_crystal_system, batch_compute_fractional_transform, \
    cart2sph_rotvec, sph2cart_rotvec, crystal_parameter_distmat, sph_rotvec2lat, lat2sph_rotvec, fractional_transform, \
    rotvec2rotmat, rotmat2rotvec
from mxtaltools.common.utils import softplus_shift, log_rescale_positive, get_point_density
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.space_group_info import (SYM_OPS, LATTICE_TYPE, NORMALIZER_OPS,
                                                   CONTINUOUS_DIMS)
from mxtaltools.crystal_building.random_crystal_sampling import sample_aunit_lengths, sample_cell_angles, \
    sample_aunit_orientations, sample_aunit_centroids
from mxtaltools.crystal_building.utils import parameterize_crystal_batch, canonicalize_aunit_order, canonicalize_rotvec, \
    unit_layout, UNIT_MAP_KEY, unit_cell_z
from mxtaltools.crystal_search.crystal_opt_utils import gradient_descent_optimization
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import get_mol_embedding_for_proxy, enforce_1d_bound
from mxtaltools.reporting.utils import lightweight_one_sided_violin


def _op_set(rotations, translations):
    return {(tuple(np.rint(r).astype(int).ravel()), tuple(np.round(np.mod(t, 1), 4) % 1))
            for r, t in zip(rotations, translations)}


@functools.lru_cache(maxsize=None)
def spglib_hall_number(sg_ind: int) -> int:
    """
    The spglib Hall number whose operators are exactly SYM_OPS[sg_ind].
    spglib standardizes to the first Hall number of each space group unless told otherwise; for the groups
    with two ITA origin choices that is a different origin from SYM_OPS.

    Operator sets are compared with rotations rounded to integers and translations
    taken modulo 1 and rounded to 4 decimals. Results are cached per ``sg_ind``.

    Parameters
    ----------
    sg_ind : int
        Space group number (1-230), a key of ``SYM_OPS``.

    Returns
    -------
    int
        The first spglib Hall number of that space group whose operator set
        equals that of ``SYM_OPS[sg_ind]``.

    Raises
    ------
    ValueError
        If no Hall setting of the space group has the operators of ``SYM_OPS[sg_ind]``.
    """
    import spglib
    ops = np.asarray(SYM_OPS[sg_ind], dtype=np.float64)
    target = _op_set(ops[:, :3, :3], ops[:, :3, 3])
    for hall_number in _spglib_hall_numbers()[sg_ind]:
        db = spglib.get_symmetry_from_database(hall_number)
        if _op_set(db['rotations'], db['translations']) == target:
            return hall_number
    raise ValueError(f"no spglib Hall setting has the operators of SYM_OPS[{sg_ind}]")


@functools.lru_cache(maxsize=None)
def _spglib_hall_numbers():
    import spglib
    halls = {}
    for hall_number in range(1, 531):
        halls.setdefault(spglib.get_spacegroup_type(hall_number).number, []).append(hall_number)
    return halls


#: Row tensors [num_graphs, >= z_prime, ...] holding one entry per asymmetric-unit slot along
#: dimension 1 (written by ``MolCrystalSteps.find_molecules`` and ``parameterize_pose``, and
#: ``MolCrystalChem.features``' packed fingerprints); ``canonicalize_zp_aunits`` moves their
#: entries with the slots.
_UNIT_ROW_KEYS = ('aunit_op', 'unit_radius', 'unit_mass', 'unit_volume', 'unit_fingerprint')
#: Row-level lists whose entries are per-unit lists (length ``z_prime``, or None), written by
#: ``MolCrystalChem.sanitize`` / ``features``; ``canonicalize_zp_aunits`` reorders them with the slots.
_UNIT_LIST_KEYS = ('unit_smiles',)
#: Row-level lists whose entries are [2, E] arrays of row-local atom indices; renumbered when
#: ``canonicalize_zp_aunits`` moves atoms.
_ATOM_INDEX_LIST_KEYS = ('file_bonds', 'mol_bonds')
#: Per-molecule scalars the list constructor (``set_mol_attrs``) sums over the units: one value per molecule, 0-dim
#: as ``MolData`` stores them or [1] as rows taken out of a collated batch carry them (audit #7).
_ROW_SUM_KEYS = ('num_atoms', 'radius', 'mass', 'mol_volume')


def _canonical_unit_order(batch) -> torch.Tensor:
    """The slot order ``canonicalize_aunit_order`` sorts each crystal's asymmetric units into.

    The same sort key and the same ``argsort`` call on the same tensors: used slots by the
    norm of their fractional centroid, unused slots (index >= ``z_prime``) by the norm of
    (1, 1, 1). The two therefore agree slot for slot; ``tests/test_zp_canonical_order.py``
    pins it.

    Parameters
    ----------
    batch : MolCrystalData batch
        Reads ``num_graphs``, ``max_z_prime``, ``z_prime``, ``device`` and
        ``aunit_centroid``. Not modified.

    Returns
    -------
    torch.Tensor
        [num_graphs, max_z_prime] long. Entry ``[i, j]`` is the slot whose unit moves into
        slot ``j``.
    """
    n, k = batch.num_graphs, batch.max_z_prime
    per_aunit_centroids = batch.aunit_centroid.reshape(n, k, 3)
    idx = torch.arange(k, device=batch.device)[None, ...]
    mask = (idx >= batch.z_prime[..., None])[..., None].expand(-1, -1, 3)
    centroids_for_sort = torch.where(mask, torch.ones_like(per_aunit_centroids), per_aunit_centroids)
    return centroids_for_sort.norm(dim=2).argsort(dim=1)


def _unit_block_moves(batch) -> dict:
    """New values of the attributes laid out unit by unit, for the canonical unit order.

    A crystal moves when ``z_prime > 1`` and its used slots change order. Its units are
    read from its unit map (``crystal_building.utils.unit_layout``: ``unit_ind``, or
    ``z_prime`` equal contiguous blocks without one; equal contiguous blocks of
    ``num_atoms // z_prime`` atoms at this stage), block ``k`` posed by slot ``k`` (the
    layout ``split_to_zp1_batch`` reads); new block ``j`` is the old block of the unit that
    moves into slot ``j`` (``_canonical_unit_order``). Atoms of other crystals keep their
    places.

    Parameters
    ----------
    batch : MolCrystalData batch
        Reads ``aunit_centroid``, ``z_prime``, ``max_z_prime``, ``ptr``, ``num_atoms``,
        ``unit_ind``, the bookkeeping ``_slice_dict`` and the attributes listed under
        Returns. Not modified.

    Returns
    -------
    dict
        ``{key: new value}``, new objects, empty when no crystal moves. Otherwise it holds:

        - ``unit_ind`` when stored: gathered with its atoms and relabelled to the new slot
          order (block ``j`` labelled ``j``), so it still says which unit each atom is;
          returned long whatever the stored integer dtype;
        - every tensor whose first dimension is the atom count and that is sliced by
          ``ptr`` (``pos``, ``z`` and ``x`` always), gathered; except ``mol_ind``, which
          numbers the units by position and so already labels the new blocks;
        - every row-level list (length ``num_graphs``) whose entries are lists or tuples
          of their row's atom count, reordered on the moving crystals;
        - ``file_bonds`` and ``mol_bonds`` (``_ATOM_INDEX_LIST_KEYS``), renumbered on the
          moving crystals, and tensors under a key containing ``'index'`` or ``'face'``
          (batch atom indices, as ``MXtalBase.__inc__`` offsets them), renumbered;
        - the per-slot row tensors of ``_UNIT_ROW_KEYS``, with their first ``z_prime``
          entries along dimension 1 reordered on the moving crystals, and the per-unit
          lists of ``_UNIT_LIST_KEYS``, reordered the same way;
        - every ragged per-unit tensor (the ``UNIT`` level of ``MXtalBase``'s level
          contract: one entry per unit, its slice entry the unit pointer of ``z_prime``,
          or its key registered ``UNIT``), its moving crystals' units reordered the same
          way (since stage 3, 2026-10-23).

    Raises
    ------
    ValueError
        If a moving crystal's units cannot be read (``unit_layout``: without a map, an atom
        count not divisible by its ``z_prime``; with one, a map that is not ``z_prime``
        equal contiguous blocks in unit order), or one of its used slots sorts behind an
        unused one (a centroid norm at or above sqrt(3), the unused slots' sort key), or a
        ``_UNIT_ROW_KEYS`` tensor is not [num_graphs, w] with ``w`` at least every moving
        crystal's ``z_prime``. Crystals that do not move are not checked.
    """
    order = _canonical_unit_order(batch)
    B, k = batch.num_graphs, batch.max_z_prime
    device = order.device
    zp = batch.z_prime.reshape(B).long().to(device)
    slot = torch.arange(k, device=device)
    used = slot[None, :] < zp[:, None]
    moved = (zp > 1) & ((order != slot[None, :]) & used).any(dim=1)
    if not bool(moved.any()):
        return {}

    ptr = batch.ptr.long().to(device)
    counts = ptr[1:] - ptr[:-1]
    behind = moved & ((order >= zp[:, None]) & used).any(dim=1)
    if bool(behind.any()):
        raise ValueError(f"canonicalize_zp_aunits: crystals {torch.nonzero(behind).flatten().tolist()} have a used "
                         f"asymmetric-unit slot sorting behind an unused one (fractional centroid norm >= sqrt(3)); "
                         f"their units cannot be put in canonical order")
    moved_rows = torch.nonzero(moved).flatten().tolist()
    # the units of the moving crystals, from the unit map (the one reader; equal contiguous blocks at this stage);
    # a crystal whose atoms are not z_prime equal blocks (or whose map is not) is refused, naming it
    layout = unit_layout(batch, 'canonicalize_zp_aunits', rows=moved_rows)
    stale = moved & (layout.counts.to(device) != counts)
    if bool(stale.any()):   # the atoms are moved by ptr, the unit sizes come from num_atoms: they must agree
        raise ValueError(f"canonicalize_zp_aunits: crystals {torch.nonzero(stale).flatten().tolist()} have a "
                         f"num_atoms that differs from their atom count (ptr)")

    # atom permutation: new atom a takes old atom perm[a]
    N = int(ptr[-1])
    atom_row = torch.repeat_interleave(torch.arange(B, device=device), counts)
    start = ptr[:-1][atom_row]
    per_unit = torch.where(moved, layout.atoms_per_unit.to(device), torch.ones_like(counts))[atom_row]
    local = torch.arange(N, device=device) - start
    unit = torch.div(local, per_unit, rounding_mode='floor')
    source = start + order[atom_row, unit.clamp(max=k - 1)] * per_unit + (local - unit * per_unit)
    perm = torch.where(moved[atom_row], source, torch.arange(N, device=device))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(N, device=device)

    slices = batch.__dict__.get('_slice_dict') or {}
    ptr_cpu = ptr.cpu()
    perm_cpu, inverse_cpu = perm.cpu(), inverse.cpu()
    counts_list = counts.tolist()
    # ragged per-unit values (the UNIT level): new unit j of a moving crystal takes its old unit order[j]
    from mxtaltools.dataset_utils.data_classes import KEY_LEVELS, UNIT, _entry_pointer   # (it imports this module)
    zp_cpu = zp.cpu().clamp(min=0)
    unit_ptr = torch.cat([zp_cpu.new_zeros(1), zp_cpu.cumsum(0)])
    U = int(unit_ptr[-1])
    unit_src = torch.arange(U)
    order_cpu = order.cpu()
    for b in moved_rows:
        n, s = int(zp_cpu[b]), int(unit_ptr[b])
        unit_src[s:s + n] = s + order_cpu[b, :n]

    def per_unit(key, value):
        if value.dim() == 0 or value.shape[0] != U or (value.shape[0] == N and key in slices and
                                                        torch.equal(slices[key].cpu(), ptr_cpu)):
            return False
        if KEY_LEVELS.get(key) == UNIT:
            return True
        entry = _entry_pointer(batch, key, value, slices.get(key), B)
        return entry is not None and torch.equal(entry, unit_ptr)

    out = {}
    if UNIT_MAP_KEY in batch._store:
        # the unit map moves with its atom blocks and is relabelled to the new slot order: block j now holds the unit
        # that moved into slot j (old label order[j]), so it is labelled j (inv_order[order[j]] = j)
        old = batch._store[UNIT_MAP_KEY]
        gathered = old[perm.to(old.device)]
        inv_order = torch.argsort(order, dim=1).to(old.device)
        row = atom_row.to(old.device)
        # long, as the map is defined (a stored int32 map comes out long; unit_layout refused a non-integer one)
        gathered = gathered.long()
        relabelled = inv_order[row, gathered.clamp(min=0, max=k - 1)]
        out[UNIT_MAP_KEY] = torch.where(moved.to(old.device)[row], relabelled, gathered)
    for key, value in batch._store.items():
        if key in ('mol_ind', 'batch', 'ptr', UNIT_MAP_KEY):
            continue
        if torch.is_tensor(value):
            if key in _UNIT_ROW_KEYS:
                out[key] = _gather_unit_columns(key, value, order, zp, moved)
            elif 'index' in key or 'face' in key:
                out[key] = inverse.to(value.device)[value]
            elif value.dim() > 0 and value.shape[0] == N and (
                    key in ('pos', 'z', 'x') or (key in slices and torch.equal(slices[key].cpu(), ptr_cpu))):
                out[key] = value[perm.to(value.device)]
            elif per_unit(key, value):
                out[key] = value[unit_src.to(value.device)]
        elif isinstance(value, list) and len(value) == B:
            if key in _UNIT_LIST_KEYS:
                reordered = list(value)
                for b in moved_rows:
                    if value[b] is None:
                        continue
                    n = int(zp[b])
                    if len(value[b]) < n:
                        raise ValueError(f"canonicalize_zp_aunits: per-unit list {key!r} of crystal {b} has "
                                         f"{len(value[b])} entries for z_prime {n}")
                    slots = order[b, :n].tolist()
                    reordered[b] = type(value[b])([value[b][j] for j in slots] + list(value[b][n:]))
                out[key] = reordered
            elif key in _ATOM_INDEX_LIST_KEYS:
                renumbered = list(value)
                for b in moved_rows:
                    bonds = value[b]
                    if bonds is None:
                        continue
                    s = int(ptr_cpu[b])
                    local_inverse = inverse_cpu[s:s + counts_list[b]] - s  # old row-local index -> new
                    if torch.is_tensor(bonds):
                        renumbered[b] = local_inverse.to(bonds.device)[bonds.long()].to(bonds.dtype)
                    else:
                        bonds = np.asarray(bonds)
                        renumbered[b] = local_inverse.numpy()[bonds].astype(bonds.dtype)
                out[key] = renumbered
            elif all(isinstance(v, (list, tuple)) and len(v) == counts_list[b] for b, v in enumerate(value)):
                reordered = list(value)
                for b in moved_rows:
                    s = int(ptr_cpu[b])
                    reordered[b] = type(value[b])(value[b][i] for i in (perm_cpu[s:s + counts_list[b]] - s).tolist())
                out[key] = reordered
    return out


def _gather_unit_columns(key, value, order, zp, moved) -> torch.Tensor:
    """``value`` [num_graphs, w, ...] with entry ``j < z_prime`` of each moving crystal taken from entry ``order[i, j]``.

    Entries at or beyond a crystal's ``z_prime``, and every entry of a crystal that does
    not move, keep their values. Raises ValueError (see ``_unit_block_moves``) on a tensor
    that is not [num_graphs, w, ...] with ``w`` at least every moving crystal's ``z_prime``.
    """
    B, k = order.shape
    if value.dim() < 2 or value.shape[0] != B or value.shape[1] < int(zp[moved].max()):
        raise ValueError(f"canonicalize_zp_aunits: per-slot row tensor {key!r} has shape {tuple(value.shape)}; "
                         f"expected [{B}, >= {int(zp[moved].max())}, ...]")
    w = value.shape[1]
    m = min(k, w)
    cols = torch.arange(w, device=order.device)
    index = cols[None, :].expand(B, w).clone()
    take = moved[:, None] & (cols[None, :m] < zp[:, None])
    index[:, :m] = torch.where(take, order[:, :m], index[:, :m])
    index = index.reshape((B, w) + (1,) * (value.dim() - 2)).expand(value.shape)
    return torch.gather(value, 1, index.to(value.device))


# noinspection PyAttributeOutsideInit


class MolCrystalOps:
    """Cell-parameter methods mixed into ``MolCrystalData``.

    Methods work on a single crystal or on a collated batch (``self.is_batch``);
    several are batch-only and say so. They use these attributes of ``self``:

    - ``cell_lengths`` : [num_graphs, 3], a, b, c in Angstrom.
    - ``cell_angles`` : [num_graphs, 3], alpha, beta, gamma in radians.
    - ``aunit_centroid`` : [num_graphs, 3 * max_z_prime], fractional (unit-cell)
      coordinates of each asymmetric unit's centroid, one 3-block per unit.
    - ``aunit_orientation`` : [num_graphs, 3 * max_z_prime], one rotation vector
      (axis times angle in radians) per asymmetric unit.
    - ``aunit_handedness`` : [num_graphs, max_z_prime], +1 or -1 per unit.
    - ``sg_ind``, ``sym_mult``, ``z_prime``, ``max_z_prime``, ``symmetry_operators``,
      and molecule attributes such as ``radius``, ``mol_volume`` and ``mass``.

    The full cell-parameter vector (``full_cell_parameters``) is
    ``[cell_lengths | cell_angles | aunit_centroid | aunit_orientation]``, width
    ``6 + 6 * max_z_prime``. The latent vector (``latent_transform``) has the same
    width and block order, with each entry rescaled so that its modelled range
    maps onto [-1, 1].
    """

    def assign_aunit_centroid(self, values, eps=1e-4):
        """Return ``values`` clipped to [0, 1 - eps], so a centroid never touches exactly 1; ``self`` is not modified."""
        return values.clip(min=0, max=1 - eps)

    def set_symmetry_attrs(self,
                           nonstandard_symmetry: bool,
                           sg_ind: Union[int, torch.tensor],
                           symmetry_operators: Optional[list] = None,
                           z_prime: Optional[Union[int, torch.tensor]] = None,
                           ):
        """Assign the symmetry attributes of ``self`` in place.

        Sets ``sg_ind`` (long tensor on ``self.device``), ``symmetry_operators``
        (a numpy array, so batches do not stack it), ``nonstandard_symmetry``,
        ``sym_mult`` ([1] long, the number of symmetry operators) and ``z_prime``.

        Parameters
        ----------
        nonstandard_symmetry : bool
            If True, ``symmetry_operators`` must be given and is used as-is.
        sg_ind : int or torch.Tensor
            Space group number.
        symmetry_operators : numpy.ndarray, optional
            Stack of symmetry operators. For standard symmetry it defaults to
            ``np.stack(SYM_OPS[int(sg_ind)])``.
        z_prime : int or torch.Tensor, optional
            Number of molecules in the asymmetric unit; defaults to a [1] tensor of ones.
            A python (or numpy) int is stored [1], the default's shape, so a row given an int
            collates beside a row given none (since 2026-10-22; it was stored 0-dim, and
            collation failed, audit #7). A tensor keeps its shape.

        Raises
        ------
        RuntimeError
            If ``nonstandard_symmetry`` is True and ``symmetry_operators`` is None.
        AssertionError
            If ``symmetry_operators`` is given and is not a numpy array.
        """
        if not torch.is_tensor(sg_ind):
            self.sg_ind = torch.tensor(sg_ind, dtype=torch.long, device=self.device)
        else:
            self.sg_ind = sg_ind.long().to(self.device)

        if nonstandard_symmetry:  # set as np stack for correct collation behavior (we don't want batches to stack)
            if symmetry_operators is not None:
                assert isinstance(symmetry_operators, np.ndarray)
                self.symmetry_operators = symmetry_operators
            else:
                raise RuntimeError("symmetry_operators must be given for nonstandard symmetry operations")
            self.nonstandard_symmetry = True
        else:  # standard symmetry
            if symmetry_operators is not None:
                assert isinstance(symmetry_operators, np.ndarray)
                self.symmetry_operators = symmetry_operators
            else:
                self.symmetry_operators = np.stack(
                    SYM_OPS[int(sg_ind)])  # if saved as a tensor, we get collation issues
            self.nonstandard_symmetry = False

        # AKA "Z value"
        self.sym_mult = torch.ones(1, dtype=torch.long, device=self.device) * len(self.symmetry_operators)

        if z_prime is not None:
            if torch.is_tensor(z_prime):
                self.z_prime = z_prime.long().to(self.device)
            else:  # a number becomes [1], the default's shape: a 0-dim value does not collate beside it (audit #7)
                z_prime = torch.tensor(z_prime, dtype=torch.long, device=self.device)
                self.z_prime = z_prime.reshape(1) if z_prime.ndim == 0 else z_prime
        else:  # fall back / assume Z'=1 if not stated
            self.z_prime = torch.ones(1, dtype=torch.long, device=self.device)

    def set_mol_attrs(self, molecule):
        """Copy the attributes of ``molecule`` onto ``self`` in place.

        Parameters
        ----------
        molecule : list of MolData, or MolData (single or collated)
            - list (one molecule per asymmetric-unit copy, Z' > 1): the dicts are
              merged key by key. ``identifier`` takes the first molecule's
              identifier (its first element if that is a list of strings, else
              ``'null'``); other strings are joined with ``"|"``; ints and floats are summed;
              0-dim tensors are summed; the row scalars ``num_atoms``, ``radius``, ``mass``
              and ``mol_volume`` are summed when each molecule's is one value, 0-dim or
              [1] as rows taken out of a collated batch carry them, and stored 0-dim,
              ``MolData``'s shape, whatever the inputs' shapes (before 2026-10-22 [1]-shaped
              ones were concatenated, audit #7; until the stage-2 review the sum took
              molecule 0's shape), and refused (ValueError) otherwise; 'index' / 'face'
              tensors (atom indices: ``edge_index``, ``tree_*_index``) are concatenated
              along their last axis with molecule ``k``'s values offset by the atoms of
              molecules ``0 .. k - 1``, as collation offsets them (since stage 3,
              2026-10-23; they were concatenated on dim 0 without an offset, audit #7);
              other tensors are concatenated along dim 0 (a [1, F] per-molecule tensor
              becomes [Z', F], one entry per unit, which collates as a per-unit value).
              Values of any other type are not copied. Every molecule must hold the same
              keys (the unit map aside), else ValueError (a key a later molecule lacked
              raised KeyError; one only later molecules held was dropped). The unit map
              ``unit_ind`` is written: molecule ``k``'s atoms are unit ``k`` (a listed
              molecule carrying a map of more than one unit is refused).
            - batch: dicts and ``torch.nn.Module`` values are skipped; strings and
              lists of strings are copied together with the source batch's
              collation metadata; other non-tensor values are converted to tensors
              (long for a python int, float32 otherwise). 0-dim tensors are set
              directly; an 'index' / 'face' tensor, or any tensor whose source slice entry
              describes it and is neither the source's atom pointer (for a value of the node
              count) nor ``arange`` (for one of the graph count) -- a ``UNIT`` or ``EDGE``
              value of the level contract, whatever its total coincides with -- is set with
              the source's slice and increment entries (since stage 3, 2026-10-23; the
              coincident totals since its review); length-``num_graphs`` values
              are attached with ``add_graph_attr``, length-``num_nodes`` values with
              ``add_node_attr`` (``batch`` is set directly), and anything else is set
              directly. No unit map is written (a batch without ``unit_ind`` has the
              equal-block layout).
            - single molecule and ``self`` not a batch: every attribute is copied, and a
              source without a unit map gets one: zeros for a molecule (one molecule, one
              unit) or a Z'=1 crystal row; for a ``MolCrystalData`` row whose ``z_prime`` is
              above 1 (an older file), its ``z_prime`` equal contiguous blocks, the layout
              every reader gives it (since the stage-2 review, 2026-10-23; it was given one
              unit, and then refused); none for a row whose units cannot be read
              (``z_prime < 1``, atoms not divisible), which the readers refuse as before.
              A single molecule given to a batch ``self`` copies nothing.
        """
        # copy out the data
        if isinstance(molecule, list):  # Z'>1 comes with Z' molecules
            mol_dicts = [mol.to_dict() for mol in molecule]
            for k, d in enumerate(mol_dicts):
                u = d.get(UNIT_MAP_KEY)
                if torch.is_tensor(u) and u.numel() and bool((u != 0).any()):
                    raise ValueError(f'set_mol_attrs: molecule {k} of the list carries a unit map (unit_ind) of '
                                     f'{int(u.max()) + 1} units; each listed molecule is one unit')
            # one key set: a key some molecules lack cannot be merged (it raised KeyError, or was dropped when only
            # later molecules held it; stage 3, 2026-10-23). The unit map is exempt: it is written below.
            keys0 = set(mol_dicts[0]) - {UNIT_MAP_KEY}
            for k, d in enumerate(mol_dicts[1:], start=1):
                differ = sorted(keys0 ^ (set(d) - {UNIT_MAP_KEY}))
                if differ:
                    raise ValueError(f'set_mol_attrs: molecule {k} of the list holds other attributes than molecule 0 '
                                     f'({differ[:8]}); the list constructor merges only keys every molecule holds')
            # atom offsets of the molecules in the merged graph, for the index keys (collation's own rule:
            # MXtalBase.__inc__ offsets 'index' / 'face' keys by the atom count)
            atom_counts = [len(d['z']) if torch.is_tensor(d.get('z')) else
                           len(d['pos']) if torch.is_tensor(d.get('pos')) else int(mol.num_nodes)
                           for d, mol in zip(mol_dicts, molecule)]
            atom_offsets = [sum(atom_counts[:k]) for k in range(len(atom_counts))]

            for key in mol_dicts[0].keys():
                if key == UNIT_MAP_KEY:
                    continue    # written below: molecule k -> unit k
                vals = [d[key] for d in mol_dicts]

                if ('index' in key or 'face' in key) and 'batch' not in key and torch.is_tensor(vals[0]):
                    # atom indices: concatenated along the last axis, molecule k's offset by the atoms before it
                    # (as collation does; audit #7: they were concatenated on dim 0 without an offset)
                    if not all(torch.is_tensor(v) and v.ndim >= 1 and v.ndim == vals[0].ndim and
                               v.shape[:-1] == vals[0].shape[:-1] for v in vals):
                        shapes = [tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for v in vals]
                        raise ValueError(f'set_mol_attrs: index key {key!r} holds shapes {shapes}; the molecules\' '
                                         f'atom indices merge only along their last axis')
                    setattr(self, key, torch.cat([v + off for v, off in zip(vals, atom_offsets)], dim=-1))

                elif key == 'identifier':
                    if isinstance(vals[0], str):
                        setattr(self, key, vals[0])
                    elif isinstance(vals[0][0], str):
                        setattr(self, key, vals[0][0])
                    else:
                        setattr(self, key, 'null')

                elif isinstance(vals[0], str):
                    setattr(self, key, "|".join(vals))

                elif isinstance(vals[0], (int, float)):
                    setattr(self, key, sum(vals))

                elif torch.is_tensor(vals[0]):
                    if key in _ROW_SUM_KEYS:  # one value per molecule, 0-dim or [1]: summed (audit #7)
                        shapes = [tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for v in vals]
                        if not all(torch.is_tensor(v) and v.numel() == 1 for v in vals):
                            raise ValueError(f'set_mol_attrs: {key!r} must be one value per listed molecule to be '
                                             f'summed over the units, got shapes {shapes}')
                        # stored 0-dim, MolData's shape, whatever the inputs' shapes and order: a row built from
                        # collated ([1]) molecules then collates beside a default row
                        setattr(self, key, torch.stack([v.reshape(()) for v in vals]).sum())
                    elif vals[0].ndim == 0:  # scalar tensor → graphwise
                        setattr(self, key, torch.stack(vals).sum())
                    else:  # nodewise tensor → concat
                        setattr(self, key, torch.cat(vals, dim=0))
            z0 = mol_dicts[0].get('z')
            if torch.is_tensor(z0) and all(torch.is_tensor(d.get('z')) for d in mol_dicts):
                sizes = torch.tensor([len(d['z']) for d in mol_dicts], dtype=torch.long, device=z0.device)
                self[UNIT_MAP_KEY] = torch.arange(len(mol_dicts), dtype=torch.long,
                                                  device=z0.device).repeat_interleave(sizes)
        elif molecule.is_batch:
            mol_dict = molecule.to_dict()
            n_graphs, n_nodes = molecule.num_graphs, molecule.num_nodes
            nodes_per_graph = molecule.num_atoms
            slice_dict = torch.arange(0, n_graphs + 1, 1, device=molecule.device)
            inc_dict = torch.zeros(n_graphs, dtype=torch.long, device=molecule.device)
            # node-wise slices are the same for every node-wise key, so build them once
            # here for the same reason slice_dict/inc_dict above are hoisted out of the
            # loop. inc_dict is zeros[n_graphs] either way, so the node path reuses it.
            node_slice_dict = self.node_slice_dict(nodes_per_graph, molecule.device)
            from mxtaltools.dataset_utils.data_classes import _entry_pointer     # (data_classes imports this module)
            source_slices = getattr(molecule, '_slice_dict', None) or {}
            source_atom_ptr = []        # the source's atom pointer, host long, read once on first need

            def atom_ptr():
                if not source_atom_ptr:
                    # the entry its own bookkeeping records for its atoms (collation and the helpers write it on the
                    # host), else its ptr (one host read on a device batch)
                    zs = source_slices.get('z')
                    if torch.is_tensor(zs) and zs.device.type == 'cpu' and torch.is_tensor(mol_dict.get('z')) and \
                            _entry_pointer(molecule, 'z', mol_dict['z'], zs, n_graphs) is not None:
                        source_atom_ptr.append(zs.long())
                    else:
                        source_atom_ptr.append(molecule._ptr_cpu())
                return source_atom_ptr[0]

            def ragged_in_source(key, value):
                """Whether the source batch records ``value`` at a ragged level (``UNIT`` / ``EDGE`` of the level
                contract): its slice entry describes it and is not the atom pointer (for a value of the atom count)
                or ``arange`` (for one of the row count), or it is an 'index' / 'face' key. As
                ``MXtalBase._level_plan``, an entry held on a device is not read for a value of the atom or row
                count (since the stage-3 review, 2026-10-23: a per-edge value whose total equalled the atom count
                was given the atom pointer)."""
                slices = source_slices.get(key)
                if not torch.is_tensor(slices):
                    return False
                index_key = molecule.__cat_dim__(key, value) == -1
                if not index_key and len(value) in (n_graphs, n_nodes) and slices.device.type != 'cpu':
                    return False
                pointer = _entry_pointer(molecule, key, value, slices, n_graphs)
                if pointer is None:
                    return False
                if index_key:
                    return True
                if len(value) == n_graphs and torch.equal(pointer, torch.arange(n_graphs + 1)):
                    return False
                if len(value) == n_nodes and torch.equal(pointer, atom_ptr()):
                    return False
                return True

            for key, value in mol_dict.items():
                if isinstance(value, dict) or isinstance(value, torch.nn.Module):
                    continue

                if isinstance(value, str) or (
                        isinstance(value, list) and len(value) > 0 and isinstance(value[0], str)
                ):  # string attrs (e.g. per-graph identifier lists) can't be tensorized;
                    # copy the source batch's collation metadata so to_data_list() still splits them
                    setattr(self, key, value)
                    src_slices = getattr(molecule, '_slice_dict', {})
                    if key in src_slices:
                        self._slice_dict[key] = src_slices[key]
                        self._inc_dict[key] = molecule._inc_dict.get(key)
                    continue

                if not torch.is_tensor(value):
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                        value = np.asarray(value)

                    value = torch.as_tensor(
                        value,
                        device=self.device,
                        dtype=torch.long if isinstance(value, int) else torch.float32,
                    )

                if len(value.size()) == 0:  # batch variables
                    setattr(self, key, value)
                elif key not in ('batch', 'ptr') and ragged_in_source(key, value):
                    # a ragged value (an 'index' key, a per-unit or per-edge value): copied with the source batch's
                    # own bookkeeping, the level contract's record of it (since stage 3, 2026-10-23; an index tensor
                    # of a 2-graph batch was given a per-row entry by its length, and other lengths none; since its
                    # review, also a ragged value whose total equals the atom or row count)
                    setattr(self, key, value)
                    self._slice_dict[key] = source_slices[key]
                    self._inc_dict[key] = molecule._inc_dict.get(key)
                elif len(value) == n_graphs:
                    self.add_graph_attr(value, key, slice_dict=slice_dict, inc_dict=inc_dict)
                elif len(value) == n_nodes:
                    if key != 'batch':
                        self.add_node_attr(value, key, num_nodes_per_graph=nodes_per_graph,
                                           slice_dict=node_slice_dict, inc_dict=inc_dict)
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)  # batch and ptr come through here for data batch objects
        elif not self.is_batch:
            mol_dict = molecule.to_dict()
            for key, value in mol_dict.items():
                setattr(self, key, value)
            z = mol_dict.get('z')
            if UNIT_MAP_KEY not in mol_dict and torch.is_tensor(z):
                from mxtaltools.dataset_utils.data_classes import MolCrystalData   # (it imports this module)
                # a crystal row's own z_prime says how many units it holds; a molecule's is not read (a molecule is
                # one unit; GFN writes z_prime onto molecule batches when it draws a Z', audit #2)
                zp = mol_dict.get('z_prime') if isinstance(molecule, MolCrystalData) else None
                zp = torch.as_tensor(zp).reshape(-1) if zp is not None else None
                if zp is None or (zp.numel() == 1 and int(zp[0]) == 1):  # one molecule (or a Z'=1 row), one unit
                    self[UNIT_MAP_KEY] = torch.zeros(len(z), dtype=torch.long, device=z.device)
                elif zp.numel() == 1 and int(zp[0]) > 1 and len(z) % int(zp[0]) == 0:
                    # a map-less Z'>1 crystal row (an older file): the layout every reader gives it, z_prime equal
                    # contiguous blocks, written out; not one unit (since the stage-2 review, 2026-10-23)
                    n, k = len(z), int(zp[0])
                    self[UNIT_MAP_KEY] = torch.arange(k, dtype=torch.long, device=z.device).repeat_interleave(n // k)
                # otherwise (z_prime < 1, atoms not divisible, several values) no map: the readers refuse the row
                # exactly as they refuse it without one

    def box_analysis(self):
        """Recompute the box-derived attributes of ``self`` in place from ``cell_lengths`` and ``cell_angles``.

        Assigns ``T_fc`` [num_graphs, 3, 3] (fractional -> Cartesian), ``T_cf``
        [num_graphs, 3, 3] (Cartesian -> fractional), ``cell_volume`` [num_graphs]
        (absolute volume, Angstrom^3), ``packing_coeff`` = ``mol_volume * sym_mult /
        cell_volume`` and ``density`` = ``mass * sym_mult / cell_volume * 1.66054``
        (g/cm^3 from Daltons per Angstrom^3).

        Raises
        ------
        ValueError
            If ``mol_volume`` or ``mass`` is None. ``T_fc``, ``T_cf`` and
            ``cell_volume`` have already been assigned when this is raised.
        """
        self.T_fc, self.T_cf, self.cell_volume = (
            batch_compute_fractional_transform(self.cell_lengths,
                                               self.cell_angles))

        # packing_coeff and density are MEANINGLESS without mol_volume/mass, so say
        # which field is missing rather than raising `TypeError: NoneType * Tensor`
        # from the arithmetic. Not defaulted: a packing coefficient of 0 (or of
        # anything) is a plausible-looking number that no caller could distinguish
        # from a measured one, which is the failure mode this codebase is most
        # prone to. The fix is one keyword at the call site: MolData(...,
        # do_mol_analysis=True), or mol_batch.mol_analysis().
        missing = [name for name in ('mol_volume', 'mass') if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"box_analysis needs {' and '.join(missing)} to compute packing_coeff "
                f"and density, but {'they are' if len(missing) > 1 else 'it is'} None. "
                "Build the molecule with do_mol_analysis=True, or call mol_analysis() "
                "on it first.")

        self.packing_coeff = self.mol_volume * self.sym_mult / self.cell_volume
        self.density = self.mass * self.sym_mult / self.cell_volume * 1.66054  # conversion from D/A^3 to g/cm^3

    def sample_random_cell_lengths(self, target_packing_coeff: Optional = None):
        """
        Sample random cell lengths and assign them to ``self.cell_lengths`` in place.

        NOTE depends on cell angles, so must do those first if resampling both.

        Asymmetric-unit lengths are drawn by ``sample_aunit_lengths`` from the
        current ``cell_angles`` and ``mol_volume`` and converted to unit-cell
        lengths with ``scale_aunit_lengths_to_unit_cell``. A single crystal is
        collated into a one-graph batch for the draw.

        Parameters
        ----------
        target_packing_coeff : float or torch.Tensor, optional
            Forwarded to ``sample_aunit_lengths``. When None, packing coefficients
            are drawn from a normal distribution (mean 0.6226, std 0.0447)
            clipped to [0.4, 1.0].
        """
        if self.is_batch:
            aunit_lengths = sample_aunit_lengths(self.num_graphs,
                                                 self.cell_angles,
                                                 self.mol_volume,
                                                 target_packing_coeff=target_packing_coeff)
        else:
            self_batch = collate_data_list([self])
            aunit_lengths = sample_aunit_lengths(1,
                                                 self_batch.cell_angles,
                                                 self_batch.mol_volume,
                                                 target_packing_coeff=target_packing_coeff)

        self.cell_lengths = self.scale_aunit_lengths_to_unit_cell(aunit_lengths)

    def sample_random_cell_angles(self):
        """Assign ``self.cell_angles`` in place: [num_graphs, 3] (or [1, 3]) radians from a fixed multivariate normal centred at pi/2, unclipped."""
        if self.is_batch:
            self.cell_angles = sample_cell_angles(self.num_graphs).to(self.device)
        else:
            self.cell_angles = sample_cell_angles(1).to(self.device)

    def sample_random_aunit_orientations(self):
        """Assign ``self.aunit_orientation`` in place: [num_graphs, 3 * max_z_prime] random rotation vectors (uniform axis direction, normally distributed angle about pi) with non-negative z components."""
        if self.is_batch:
            self.aunit_orientation = sample_aunit_orientations(
                self.num_graphs, z_prime=self.max_z_prime).to(self.device)
        else:
            self.aunit_orientation = sample_aunit_orientations(
                1, z_prime=self.max_z_prime).to(self.device)

    def sample_random_aunit_centroids(self):
        """Assign ``self.aunit_centroid`` in place: [num_graphs, 3 * max_z_prime] uniform positions in each space group's asymmetric-unit box, as unit-cell fractional coordinates clipped to [0, 1 - 1e-4]."""
        if self.is_batch:
            aunit_centroid = self.assign_aunit_centroid(sample_aunit_centroids(self.num_graphs,
                                                                               z_prime=self.max_z_prime
                                                                               ).to(self.device))
        else:
            aunit_centroid = self.assign_aunit_centroid(sample_aunit_centroids(1,
                                                                               z_prime=self.max_z_prime
                                                                               ).to(self.device))

        self.aunit_centroid = self.scale_centroid_to_unit_cell(aunit_centroid)

    def sample_random_aunit_handedness(self):
        """Assign ``self.aunit_handedness`` in place: [num_graphs, max_z_prime] random values in {-1, +1}, created on the default torch device rather than ``self.device``."""
        # randomize the handedness parameter for the asymmetric unit, including for Zp>12
        self.aunit_handedness = torch.randint(0, 2, (self.num_graphs, self.max_z_prime)) * 2 - 1

    def sample_random_crystal_parameters(self,
                                         target_packing_coeff: Optional = None,
                                         seed: Optional[int] = None,
                                         cleaning_mode: Optional[str] = 'hard'
                                         ):
        """Resample all cell parameters of ``self`` in place.

        Samples cell angles, then cell lengths, aunit orientations and aunit
        centroids; resamples ``aunit_handedness`` only if any ``z_prime > 1``
        (otherwise the existing handedness is kept); then calls
        ``clean_cell_parameters(mode=cleaning_mode)`` and ``box_analysis``.

        Parameters
        ----------
        target_packing_coeff : float or torch.Tensor, optional
            Forwarded to ``sample_random_cell_lengths``.
        seed : int, optional
            If given, passed to ``torch.manual_seed`` (the global torch RNG) before sampling.
        cleaning_mode : str
            ``'hard'`` or ``'soft'``, forwarded to ``clean_cell_parameters``.
        """
        if seed is not None:  # todo this seeding method is wrong
            torch.manual_seed(seed)
        self.sample_random_cell_angles()  # must do this one before cell lengths !!!
        self.sample_random_cell_lengths(target_packing_coeff)
        self.sample_random_aunit_orientations()
        self.sample_random_aunit_centroids()
        if torch.any(self.z_prime > 1):
            self.sample_random_aunit_handedness()
        self.clean_cell_parameters(mode=cleaning_mode)
        self.box_analysis()

    def sample_random_reduced_crystal_parameters(self,
                                                 target_packing_coeff: Optional = None,
                                                 seed: Optional[int] = None,
                                                 oversample: int = 32,
                                                 max_rounds: int = 1000,
                                                 ):
        """
        Sample random crystal parameters whose box has zero cell-reduction penalty, in place.

        Batch only. Rejection sampling per row: each round draws ``oversample`` candidate boxes for every row still
        without one (angles from ``sample_cell_angles``; lengths from ``sample_aunit_lengths`` at the row's molecular
        volume and packing coefficient, converted to unit-cell lengths as ``scale_aunit_lengths_to_unit_cell`` does;
        then ``enforce_crystal_system``), and a row keeps its first candidate whose ``cell_reduction_penalty`` at
        margin 0 is exactly 0. That is the proposal conditioned on a zero penalty, the distribution of the earlier
        loop, which redrew the whole batch until every row had kept one (on MIPCAS sg 2 about 0.5% of draws are
        accepted; 1,000 starts took 6.2 s that way and 0.17 s this way on a CPU, 2026-10-28). Then assigns the kept
        ``cell_lengths`` and ``cell_angles`` (crystal system re-enforced), samples aunit orientations and centroids,
        resamples ``aunit_handedness`` if any ``z_prime > 1``, and calls ``box_analysis``. ``clean_cell_parameters``
        is not called.

        Parameters
        ----------
        target_packing_coeff : float or torch.Tensor, optional
            Forwarded to ``sample_aunit_lengths``: a float, or one value per row.
        seed : int, optional
            If given, passed to ``torch.manual_seed`` once before sampling.
        oversample : int
            Candidates per row and round.
        max_rounds : int
            Rounds before a row still without a zero-penalty box raises.

        Raises
        ------
        AssertionError
            If ``self`` is not a batch.
        RuntimeError
            If some row has no zero-penalty candidate after ``max_rounds`` rounds.
        """
        from mxtaltools.common.sym_utils import cell_reduction_penalty
        if seed is not None:
            torch.manual_seed(seed)
        assert self.is_batch, "Random crystal parameters not setup for single crystals"
        n, dev = self.num_graphs, self.device
        if not hasattr(self, 'asym_unit_dict'):
            self.asym_unit_dict = self.build_asym_unit_dict()
        auv = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind]).to(dev)
        tpc = target_packing_coeff
        per_row = torch.is_tensor(tpc) and tpc.numel() > 1
        if per_row:
            tpc = tpc.reshape(-1).to(dev)
        good_params = torch.zeros((n, 6), dtype=torch.float32, device=dev)
        todo = torch.arange(n, device=dev)
        for _ in range(max_rounds):
            if not len(todo):
                break
            rep = todo.repeat_interleave(oversample)
            angles = sample_cell_angles(len(rep)).to(dev)  # angles first: the lengths depend on them
            lengths = sample_aunit_lengths(len(rep), angles, self.mol_volume[rep],
                                           target_packing_coeff=tpc[rep] if per_row else tpc) / auv[rep]
            lengths, angles = enforce_crystal_system(lengths, angles, self.sg_ind[rep])
            ok = (cell_reduction_penalty(angles, lengths, self.sg_ind[rep], 0.0) == 0).view(len(todo), oversample)
            has = ok.any(dim=1)
            first = torch.arange(len(todo), device=dev) * oversample + ok.float().argmax(dim=1)
            good_params[todo[has]] = torch.cat([lengths, angles], dim=1)[first[has]].to(good_params.dtype)
            todo = todo[~has]
        if len(todo):
            raise RuntimeError(f"sample_random_reduced_crystal_parameters: no zero-penalty cell after {max_rounds} "
                               f"rounds of {oversample} candidates for space groups "
                               f"{sorted(set(self.sg_ind[todo].tolist()))}")

        self.cell_lengths = good_params[:, :3]
        self.cell_angles = good_params[:, 3:]
        self.cell_lengths, self.cell_angles = enforce_crystal_system(
            self.cell_lengths,
            self.cell_angles,
            self.sg_ind
        )
        self.sample_random_aunit_orientations()
        self.sample_random_aunit_centroids()
        if torch.any(self.z_prime > 1):
            self.sample_random_aunit_handedness()
        # enforce agreement with crystal system
        # other cell parameters are valid by explicit construction
        # todo add crystal system conditions to sampling workflow, rather than cleaning up here

        self.box_analysis()

    def latent_to_cell_params(self,
                              latents: torch.tensor,
                              skip_box_analysis: bool = False,
                              skip_enforce_crystal_system: bool = False,
                              widen_to: Optional[torch.Tensor] = None,
                              ):
        # widen_to ([num_graphs, 6 + 6 * max_z_prime], optional): per row and entry, the clamp bounds are widened
        # just enough to include these values (None: unchanged). A hop kick passes the unclipped latents of the
        # crystals it kicks, so a parent lying outside the box stays representable.
        """
        Transform from latent space to physical crystal parameters, assigned to ``self`` in place.

        The latents are first clamped: every entry to [-1, 1], except the three
        length entries (floor -0.99), the polar-angle and rotation-magnitude
        entries of every asymmetric unit's orientation block (floor -0.99), and
        latent entries 0 and 1 (the a and b length entries, ceiling 1 - 1e-4).
        With ``widen_to``, these bounds are widened per row and entry just enough
        to include its values.
        The clamped latents go through ``inv_latent_transform`` and
        ``set_cell_parameters``, which assigns ``cell_lengths``, ``cell_angles``,
        ``aunit_centroid`` and ``aunit_orientation`` and always runs
        ``box_analysis`` once. Unless skipped, ``enforce_crystal_system`` then
        overwrites ``cell_lengths`` and ``cell_angles``, and ``box_analysis`` runs
        again. Returns None.

        Parameters
        ----------
        latents : torch.Tensor
            [num_graphs, 6 + 6 * max_z_prime] latent vectors in the
            ``latent_transform`` layout. Not modified.
        skip_box_analysis : bool
            Skip the ``box_analysis`` call after crystal-system enforcement. The
            call inside ``set_cell_parameters`` still runs, so ``box_analysis``'s
            ValueError on missing ``mol_volume``/``mass`` is still raised, and the
            box attributes describe the cell before crystal-system enforcement.
        skip_enforce_crystal_system : bool
            Skip ``enforce_crystal_system``.
        widen_to : torch.Tensor, optional
            ``[num_graphs, 6 + 6 * max_z_prime]`` values in the latent layout,
            converted to float32 on ``self.device``. Per
            row and entry, the clamp bounds above are widened just enough to
            include these values: the lower bound becomes the smaller of the
            default floor and the value, the upper bound the larger of the default
            ceiling and the value. Default None: the bounds are unchanged.
            ``log_noise_latent_parameters(keep_start_representable=True)`` passes
            the unclipped latents of the crystals it displaces, so a crystal lying
            outside the default bounds stays representable.
        """

        min_vals = -torch.ones(latents.shape[-1], dtype=torch.float32, device=self.device)
        min_vals[:3] = -0.99  # DO NOT allow micro cells to be instantiated
        # also do not allow rotation length = 0 to be instantiated.
        #
        # The layout is [3 lengths | 3 angles | 3*Zp centroids | 3*Zp orientations], with
        # BOTH aunit blocks flattened, so aunit `ind`'s rotation magnitude (component 2 of
        # its rotvec) is at 6 + 3*Zp + 3*ind + 2. The old expression `5 + 6*(1 + ind)`
        # gives 11 + 6*ind, which is correct ONLY at Zp=1 and only by coincidence:
        #   Zp=1: clamps [11]          correct [11]        -- identical, so this is a NO-OP
        #   Zp=2: clamped [11, 17]     correct [14, 17]    -- 11 is centroid[1][z]
        #   Zp=3: clamped [11, 17, 23] correct [17, 20, 23]
        # i.e. only the LAST aunit was ever protected, and at Zp>=2 a centroid row was
        # clamped instead. MEASURED at Zp=2, sg 9: latent row 14 at -1.0 gave |rotvec|
        # EXACTLY 0.0 for aunit 0 -- the r=0 singularity this line exists to prevent, and
        # the one compute_jacobian's log(sin(r/2)) clamps at ~37 nats. Row 17 (the last
        # aunit) correctly floored at 0.0314. It never IndexErrors, because the wrong
        # index 5+6*Zp is exactly width-1: silent by construction.
        for ind in range(self.max_z_prime):
            min_vals[6 + 3 * self.max_z_prime + 3 * ind + 2] = -0.99
            # AND the same guard on the POLAR component (+ 0), for the same reason one row
            # over. lat2sph_rotvec maps theta = (pi/4)x + pi/4, so x = -1 puts theta at
            # EXACTLY 0: the axis pole, where phi is undefined and the crystal -> latent
            # inverse (acos/atan2 on the rotation matrix) has an infinite local derivative.
            # The value survives -- compute_jacobian's log(sin(theta)) clamp caps it -- but
            # d(anything)/d(latent) comes back NON-FINITE for such a row, and one row is
            # enough to poison a whole batch's reward gradient. MEASURED 2026-09-21 on a
            # phase-1 qm9c policy: 3 rows in 2048 non-finite at the -1 floor, 0 at -0.99,
            # with the in-box median |dJ/dx| unchanged (25.30 vs 25.32). Same -0.99 as the
            # magnitude row, which caps log(sin(theta)) near 33 nats against that row's ~37.
            min_vals[6 + 3 * self.max_z_prime + 3 * ind + 0] = -0.99

        max_vals = torch.ones(latents.shape[-1], dtype=torch.float32, device=self.device)
        max_vals[
            0:2] = 1 - 1e-4  # don't let it explicitly touch 1 or it can make an effective orthorhombic cell, and really pisses off ASE
        if widen_to is not None:  # a real crystal outside the box keeps its own value (its kicks stay local)
            w = widen_to.to(device=self.device, dtype=min_vals.dtype)
            min_vals, max_vals = torch.minimum(min_vals, w), torch.maximum(max_vals, w)
        self.set_cell_parameters(self.inv_latent_transform(latents.clamp(min=min_vals, max=max_vals)))

        if not skip_enforce_crystal_system:
            self.cell_lengths, self.cell_angles = enforce_crystal_system(
                self.cell_lengths,
                self.cell_angles,
                self.sg_ind
            )
        if not skip_box_analysis:
            self.box_analysis()

    def latent_params(self, gauge_fix_free_axes: bool = True):
        """
        Transform cell parameters from physical space to latent.

        `gauge_fix_free_axes=False` is for NON-CRYSTAL (toy) problems, where the
        batch is only a container and every latent dim is a real coordinate of
        the energy field: gauge-fixing there rewrites live data (on P1 all THREE
        centroid axes are 'free', so the multiharmonic toy's u,v,w targets were
        silently pinned to 0 from 2026-08-11 until this flag existed -- the
        model was trained toward delta functions the prior does not contain).
        The caller knows crystal-vs-toy (energy_function.is_crystal); this batch
        does not, so it is a parameter.

        This method MUTATES ``self`` before transforming: ``canonicalize_zp_aunits``
        always reassigns ``aunit_centroid``, ``aunit_orientation`` and
        ``aunit_handedness`` (asymmetric units reordered by distance of their
        fractional centroid from the origin) and, on Z' > 1 rows whose order
        changes, moves each unit's atom block (``pos``, ``z``, ``x`` and the other
        per-atom attributes, bond lists and per-unit tensors) with its slot; with
        ``gauge_fix_free_axes=True`` ``canonicalize_free_axes`` may reassign
        ``aunit_centroid``.

        Parameters
        ----------
        gauge_fix_free_axes : bool
            Call ``canonicalize_free_axes`` before transforming.

        Returns
        -------
        torch.Tensor
            [num_graphs, 6 + 6 * max_z_prime] latents from ``latent_transform``,
            clipped to [-1, 1].
        """
        self.canonicalize_zp_aunits()  # latent space is always in the canonical ordering
        # Gauge-fix the free centroid axes to the same canonical value the projected cell
        # angles already take (latent 0). MUST be paired with declaring those rows dead in
        # the GFN: canonicalising here without holding them in the SDE would leave every
        # buffer row at the constant while the policy flowed the dim freely -- which is
        # precisely the D33 defect, in the opposite direction.
        if gauge_fix_free_axes:
            self.canonicalize_free_axes()
        return self.latent_transform(cell_params=self.full_cell_parameters()).clip(min=-1, max=1)

    def latent_transform(self, cell_params):
        """Map cell parameters to latent coordinates, without clipping.

        Reads ``sg_ind``, ``radius``, ``z_prime`` and ``max_z_prime``; builds and
        caches ``self.asym_unit_lut`` if missing. With ``auv`` the space group's
        asymmetric-unit extent (``ASYM_UNITS``) and
        ``r_eff = radius / z_prime ** (2/3)``:

        - lengths: ``cell_lengths * auv / (2 r_eff)`` is mapped log-linearly so that
          [0.075, 3] (a, b) and [0.1, 4] (c) span [-1, 1];
        - angles: [0.2 pi, 0.8 pi] radians is mapped linearly onto [-1, 1];
        - centroids: ``aunit_centroid / auv`` on [0, 1] is mapped onto [-1, 1];
        - orientations: each rotation vector is converted to spherical
          (theta, phi, r) and theta [0, pi/2], phi [-pi, pi], r [0, 2 pi] are mapped
          onto [-1, 1] (``sph_rotvec2lat``).

        Values outside these ranges map outside [-1, 1].

        Parameters
        ----------
        cell_params : torch.Tensor
            [num_graphs, 6 + 6 * max_z_prime] in the ``full_cell_parameters`` layout.

        Returns
        -------
        torch.Tensor
            [num_graphs, 6 + 6 * max_z_prime] latents, blocks
            [lengths | angles | centroids | orientations].
        """
        if not hasattr(self, 'asym_unit_lut'):
            self.build_asym_unit_tensor()

        sg_inds = self.sg_ind
        # 'radius' for Z'>1 stuctures is Z'*radius for downstream reasons. Also it's not intensive so we need a scaling
        radius = self.radius / (self.z_prime ** (2 / 3))
        auvs = self.asym_unit_lut[sg_inds]
        #auvs = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in sg_inds]).to(self.device)

        'convert to latent basis'
        # get aunit lengths
        box_params, aunit_params = torch.tensor_split(cell_params, [6], dim=1)
        cell_lengths = box_params[:, :3]
        cell_angles = box_params[:, 3:]
        aunit_lengths = cell_lengths * auvs
        normed_aunit_lengths = aunit_lengths / (2 * radius[:, None])

        # get aunit-wise fractional centroids
        cell_centroids, cell_orientations = aunit_params.split(3 * self.max_z_prime, dim=1)
        # this is the parallel way to do it over multiple Z'
        aunit_centroids = (
                cell_centroids
                .reshape(-1, self.max_z_prime, 3)  # [n, max_z', 3]
                / auvs.unsqueeze(1)  # [n, 1, 3] → broadcast over Z′
        ).reshape(-1, 3 * self.max_z_prime)  # back to [n, 3 * max_z']

        # get spherical rotvec
        sph_rotvec = cart2sph_rotvec(cell_orientations.reshape(len(cell_orientations) * self.max_z_prime, 3)).reshape(
            len(cell_orientations), self.max_z_prime * 3)

        'rescale everything to [-1,1]'
        # halfpi = torch.pi / 2
        au_range = torch.tensor([[0.075, 0.075, 0.1], [3, 3, 4]], dtype=torch.float32,
                                device=self.device)  # range to scale aunit lengths
        log_au_range = torch.log(au_range)
        ang_range = [0.2 * torch.pi, 0.8 * torch.pi]  # range of allowed cell angles

        # lengths & angles - rescale from [min, max] to [-1,1]
        lat_lengths = ((normed_aunit_lengths.log() - log_au_range[0]) / (log_au_range[1] - log_au_range[0]) - 0.5) * 2
        lat_angles = ((cell_angles - ang_range[0]) / (ang_range[1] - ang_range[0]) - 0.5) * 2

        # aunit centroid [0,1] to [-1,1]
        lat_centroids = (aunit_centroids - 0.5) * 2

        lat_orientations = sph_rotvec2lat(sph_rotvec, self.max_z_prime)

        latents = torch.cat([lat_lengths, lat_angles, lat_centroids, lat_orientations], dim=1)
        return latents

    def inv_latent_transform(self, latents):
        """Map latent coordinates back to cell parameters; the inverse of ``latent_transform``.

        Applies no clamping; builds and caches ``self.asym_unit_lut`` if missing
        and otherwise leaves ``self`` unmodified.

        Parameters
        ----------
        latents : torch.Tensor
            [num_graphs, 6 + 6 * max_z_prime] latents.

        Returns
        -------
        torch.Tensor
            [num_graphs, 6 + 6 * max_z_prime] cell parameters in the
            ``full_cell_parameters`` layout: lengths (Angstrom), angles (radians),
            fractional centroids and rotation vectors.
        """
        if not hasattr(self, 'asym_unit_lut'):
            self.build_asym_unit_tensor()

        sg_inds = self.sg_ind
        radius = self.radius / (self.z_prime ** (
                2 / 3))  # 'radius' for Z'>1 stuctures is Z'*radius for downstream reasons. Also it's not intensive so we need a scaling        auvs = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in sg_inds]).to(self.device)

        auvs = self.asym_unit_lut[sg_inds].to(self.device)

        lat_lengths, lat_angles, lat_centroids, lat_orientations = torch.split(latents, [3, 3, 3 * self.max_z_prime,
                                                                                         3 * self.max_z_prime], dim=1)

        '''reverse bounding transform'''
        sph_rotvec = lat2sph_rotvec(lat_orientations, self.max_z_prime)

        aunit_orientations = sph2cart_rotvec(sph_rotvec.reshape(len(sph_rotvec) * self.max_z_prime, 3)).reshape(
            len(sph_rotvec), self.max_z_prime * 3)

        aunit_centroids = lat_centroids / 2 + 0.5

        cell_centroids = (
                aunit_centroids
                .reshape(-1, self.max_z_prime, 3)  # [n, max_z', 3]
                * auvs.unsqueeze(1)  # [n, 1, 3] → broadcast over Z′
        ).reshape(-1, 3 * self.max_z_prime)  # back to [n, 3 * max_z']

        au_range = torch.tensor([[0.075, 0.075, 0.1], [3, 3, 4]], dtype=torch.float32,
                                device=self.device)  # range to scale aunit lengths
        log_au_range = torch.log(au_range)
        ang_range = [0.2 * torch.pi, 0.8 * torch.pi]  # range of allowed cell angles

        log_span = log_au_range[1] - log_au_range[0]
        logL = (lat_lengths / 2 + 0.5) * log_span + log_au_range[0]
        normed_aunit_lengths = logL.exp()

        # angles
        ang_span = ang_range[1] - ang_range[0]
        cell_angles = (lat_angles / 2 + 0.5) * ang_span + ang_range[0]

        aunit_lengths = normed_aunit_lengths * (2 * radius[:, None])
        cell_lengths = aunit_lengths / auvs

        return torch.cat([cell_lengths, cell_angles, cell_centroids, aunit_orientations], dim=1)

    def latent_distmat(self):
        """Return the [num_graphs, num_graphs] pairwise ``crystal_parameter_distmat`` distances of ``latent_params(gauge_fix_free_axes=False)``, with a zero diagonal.

        ``latent_params`` still reassigns the aunit attributes through ``canonicalize_zp_aunits``.
        """
        # gauge-FREE: distances on gauge-fixed latents collapse the free-axis
        # separation on toy routes (and mutate the batch). Crystal rows are
        # already canonical, so identical there. (P1 2026-08-24)
        return crystal_parameter_distmat(
            self.latent_params(gauge_fix_free_axes=False)).fill_diagonal_(0)

    def sample_reduced_box_vectors(self, target_packing_coeff: Optional[float] = None):
        """
        Disabled: always fails with ``AssertionError`` ("Reduced box sampling is deprecated").

        Parameters
        ----------
        target_packing_coeff : float, optional
            Unused.
        """
        assert False, "Reduced box sampling is deprecated - must be re-implemented with our new reduction methods"

        # if not hasattr(self, 'latent_transform'):
        #     self.init_latent_transform()
        #
        # # use the latent transform for lengths and angles sampling, as its more stable and better tested
        # # we can still use the old samplers for aunit params
        # # latent space is defined on [-1,1]
        # rands = (torch.rand((self.num_graphs, 6 + 6 * self.max_z_prime), device=self.device) - 0.5) * 2
        # temp_params = self.latent_transform.inverse(rands,
        #                                             self.sg_ind,
        #                                             self.radius)
        # cell_lengths = temp_params[:, :3]
        # cell_angles = temp_params[:, 3:6]
        #
        # if target_packing_coeff is not None:
        #     vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
        #     cp1 = self.mol_volume * self.sym_mult / vol1
        #     correction_ratio = (cp1 / target_packing_coeff) ** (1 / 3)
        #     cell_lengths *= correction_ratio[:, None]
        #
        # cell_angles = enforce_niggli_plane(cell_lengths, cell_angles, mode='mirror')
        #
        # self.cell_lengths, self.cell_angles = cell_lengths, cell_angles

    def sample_reasonable_random_parameters(self,
                                            tolerance: float = 1,
                                            target_packing_coeff: Optional = None,
                                            max_attempts: int = 100,
                                            seed: Optional[int] = None,
                                            ):
        """
        Sample random crystal parameters, build the resulting crystals and check their vdW overlaps,
        keeping each crystal's best sample; the result is assigned to ``self`` in place.

        Each attempt calls ``sample_random_crystal_parameters`` and scores the
        batch with ``build_and_analyze(cutoff=3)``'s ``scaled_lj``. Each row keeps
        the ``zp1_cell_parameters`` of its lowest-``scaled_lj`` attempt; the loop
        stops when every row has had an attempt with ``scaled_lj < tolerance`` or
        after ``max_attempts`` attempts. Rows that never meet the tolerance keep
        their best attempt; a row whose score never drops below the initial 1e7
        (e.g. NaN) keeps all-zero parameters. The kept parameters are assigned
        with ``set_cell_parameters`` (which runs ``box_analysis``).
        ``build_and_analyze`` also assigns ``lj``, ``es_pot`` and ``scaled_lj`` on
        ``self``. Rows are copied from ``zp1_cell_parameters`` (12 columns) into a
        tensor of ``full_cell_parameters`` width, so this only works when
        ``max_z_prime == 1``.

        Parameters
        ----------
        tolerance : float
            ``scaled_lj`` threshold below which a row counts as found.
        target_packing_coeff : float or torch.Tensor, optional
            Forwarded to ``sample_random_crystal_parameters``.
        max_attempts : int
            Maximum number of sampling attempts.
        seed : int, optional
            Forwarded to ``sample_random_crystal_parameters``, which reseeds the
            global torch RNG with it on every attempt.
        """
        good_params_tensor = torch.zeros_like(self.full_cell_parameters())
        found_params_flags = torch.zeros(self.num_graphs, dtype=torch.bool)
        best_ljs = torch.ones(self.num_graphs, dtype=torch.float32, device=self.device) * 1e7
        converged = False
        ind = 0
        while not converged and ind < max_attempts:
            self.sample_random_crystal_parameters(target_packing_coeff, seed=seed)

            _, _, scaled_lj = self.build_and_analyze(cutoff=3)
            improved_inds = torch.argwhere(scaled_lj < best_ljs)
            best_ljs[improved_inds] = scaled_lj[improved_inds]
            good_inds = torch.argwhere(scaled_lj < tolerance)
            good_params_tensor[improved_inds] = self.zp1_cell_parameters()[improved_inds]
            found_params_flags[good_inds] = True
            ind += 1
            if torch.all(found_params_flags):
                converged = True

        self.set_cell_parameters(good_params_tensor)
        #
        # if not converged:
        #     print(f"Failed to converge {torch.sum(~found_params_flags)} out of {self.num_graphs} crystals")

    def denorm_by_radius(self, arr):
        """
        Return ``arr * radius`` row-wise (``arr *= mol_radius``); ``self`` is not modified.

        assumes arr shape [n, m]: [num_graphs, m] for a batch, [1, m] for a single crystal.
        """
        if self.is_batch:
            return arr * self.radius[:, None]
        else:
            return arr * self.radius[None]

    def norm_by_radius(self, arr):
        """
        Return ``arr / radius`` row-wise (arr /= mol_radius); ``self`` is not modified.

        assumes arr shape [n, m]: [num_graphs, m] for a batch, [1, m] for a single crystal.
        """
        if self.is_batch:
            return arr / self.radius[:, None]
        else:
            return arr / self.radius[None]

    def build_asym_unit_dict(self,
                             force_all_sgs: Optional[bool] = False
                             ):  # todo add capability to only do this for the necessary keys
        """
        Return a copy of ``ASYM_UNITS`` (keys are space-group numbers as strings)
        in which the relevant entries are converted to float tensors on ``self.device``.

        NOTE this function is extremely slow
        Best used during batch operations

        Entries not converted remain python lists; ``self`` is not modified.

        Parameters
        ----------
        force_all_sgs : bool, optional
            Convert every entry. Otherwise only the space groups present in
            ``self.sg_ind`` are converted.

        Returns
        -------
        dict
            Space-group string -> [3] asymmetric-unit extent along a, b, c
            (fraction of the unit cell).
        """
        asym_unit_dict = ASYM_UNITS.copy()
        if force_all_sgs:
            sgs_to_tensorize = asym_unit_dict.keys()
        else:
            relevant_sgs = self.sg_ind.unique()
            sgs_to_tensorize = [str(int(sg)) for sg in relevant_sgs]
        for key in sgs_to_tensorize:
            asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key]).to(self.device)
        return asym_unit_dict

    def build_asym_unit_tensor(self):
        """Assign ``self.asym_unit_lut``: a [231, 3] float32 table of ``ASYM_UNITS`` extents indexed by space-group number (row 0 is zeros)."""
        max_sg = 230

        asym_unit_by_sg = torch.zeros(
            max_sg + 1, 3,
            dtype=torch.float32,
            device=self.device,
        )

        for k, v in ASYM_UNITS.items():
            asym_unit_by_sg[int(k)] = torch.as_tensor(
                v,
                dtype=torch.float32,
                device=self.device,
            )

        self.asym_unit_lut = asym_unit_by_sg

    def build_sym_mult_tensor(self):
        """Assign ``self.sym_mult_lut``: a [231] long table of the number of ``SYM_OPS`` operators per space-group number (row 0 is zero)."""
        sym_mult_by_sg = torch.zeros(231, dtype=torch.long, device=self.device)

        for sg, sym_ops in SYM_OPS.items():
            sym_mult_by_sg[int(sg)] = len(sym_ops)

        self.sym_mult_lut = sym_mult_by_sg

    def scale_lengths_to_aunit(self):
        """
        scale unit cell lengths to the asymmetric unit, according to the canonical
        asymmetric unit shape within the unit cell.

        Returns ``cell_lengths * auv`` per row, with ``auv`` the space group's
        ``ASYM_UNITS`` extent, as a new [num_graphs, 3] tensor. For a batch,
        ``self.asym_unit_dict`` is built with ``build_asym_unit_dict`` and cached
        on ``self`` if it is missing; the cache is not refreshed if ``sg_ind``
        later changes.
        """
        if self.is_batch:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()

            return self.cell_lengths * torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return self.cell_lengths * torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

    def scale_aunit_lengths_to_unit_cell(self, aunit_lengths):
        """
        Convert asymmetric-unit lengths to unit-cell lengths, ``aunit_lengths / auv`` per row.

        ``auv`` is the space group's ``ASYM_UNITS`` extent; space groups without a
        hand-encoded entry use [1, 1, 1] (the whole cell).
        this approach will not work for asymmetric units which are not neat parallelpipeds.
        For a batch, ``self.asym_unit_dict`` is built and cached if missing.

        Parameters
        ----------
        aunit_lengths : torch.Tensor
            [num_graphs, 3] asymmetric-unit lengths (Angstrom).

        Returns
        -------
        torch.Tensor
            [num_graphs, 3] unit-cell lengths.
        """
        if self.is_batch:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()
            return aunit_lengths / torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return aunit_lengths / torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

    def scale_centroid_to_aunit(self):
        """
        Return ``self.aunit_centroid`` expressed as a fraction of the asymmetric-unit box.

        Divides each 3-block of ``aunit_centroid`` (unit-cell fractional
        coordinates, on [0, auv]) by the space group's ``ASYM_UNITS`` extent
        ``auv``, giving values on [0, 1] for centroids inside the box. Space
        groups without a hand-encoded entry use [1, 1, 1].
        this approach will not work for asymmetric units which are not neat parallelpipeds.
        For a batch, ``self.asym_unit_dict`` is built and cached if missing; for a
        single crystal the divisor has width 3, so broadcasting only works when
        ``max_z_prime == 1``.

        Returns
        -------
        torch.Tensor
            [num_graphs, 3 * max_z_prime] new tensor; ``aunit_centroid`` is not modified.
        """

        if self.is_batch:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()
            scales = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])

            return self.aunit_centroid / scales.repeat(1, self.max_z_prime)
        else:
            return self.aunit_centroid / torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

    def scale_centroid_to_unit_cell(self, normed_centroid):
        """
        Convert asymmetric-unit-fraction centroids to unit-cell fractional coordinates.

        Multiplies each 3-block by the space group's ``ASYM_UNITS`` extent ``auv``
        (space groups without a hand-encoded entry use [1, 1, 1]) and clips the
        result to [0, 1 - 1e-4] with ``assign_aunit_centroid``.
        this approach will not work for asymmetric units which are not neat parallelpipeds.
        For a batch, ``self.asym_unit_dict`` is built and cached if missing;
        otherwise ``self`` is not modified.

        Parameters
        ----------
        normed_centroid : torch.Tensor
            [num_graphs, 3 * max_z_prime] centroids scaled on [0, 1] within the
            asymmetric-unit box.

        Returns
        -------
        torch.Tensor
            [num_graphs, 3 * max_z_prime] fractional centroids.
        """
        if self.is_batch:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()

            scales = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
            return self.assign_aunit_centroid(normed_centroid * scales.repeat(1, self.max_z_prime))
        else:
            return self.assign_aunit_centroid(
                normed_centroid * torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device).repeat(
                    self.max_z_prime))

    def noise_cell_parameters(self, noise_level: float):
        """Add Gaussian noise to the standardized Z'=1 cell parameters and assign the result to ``self`` in place.

        Noise of standard deviation ``noise_level`` is added to
        ``zp1_std_cell_parameters()``; the result is destandardized, assigned with
        ``set_cell_parameters(skip_box_analysis=True)``, cleaned with
        ``clean_cell_parameters(mode='hard')`` and followed by ``box_analysis``.
        The 12-column parameters match ``set_cell_parameters`` only when
        ``max_z_prime == 1``.

        Parameters
        ----------
        noise_level : float
            Standard deviation of the noise, in standardized units.
        """
        # standardize
        std_cell_params = self.zp1_std_cell_parameters()
        # noise
        new_std_cell_params = std_cell_params + torch.randn_like(std_cell_params) * noise_level
        # destandardize
        destandardized_cell_params = self.destandardize_zp1_cell_parameters(new_std_cell_params)
        # assign
        self.set_cell_parameters(destandardized_cell_params, skip_box_analysis=True)
        # clean
        self.clean_cell_parameters(mode='hard')
        # refresh box
        self.box_analysis()

    def noise_latent_parameters(self, noise_level: float):
        """Add Gaussian noise in latent space and assign the result to ``self`` in place.

        Takes ``latent_params(gauge_fix_free_axes=False)``, adds noise of standard
        deviation ``noise_level`` to every entry, and applies
        ``latent_to_cell_params`` (which clamps, enforces the crystal system and
        runs ``box_analysis``).

        Parameters
        ----------
        noise_level : float
            Standard deviation of the noise, in latent units.
        """
        # gauge-FREE: the default MUTATES the batch (canonicalize_free_axes
        # writes aunit_centroid) and noising must start from the batch's ACTUAL
        # latents. Crystal rows are already canonical (SDE-held), so this is
        # behavior-identical there; on toys those dims are real data (P1 2026-08-24).
        latents = self.latent_params(gauge_fix_free_axes=False)
        noised_params = latents + torch.randn_like(latents) * noise_level
        self.latent_to_cell_params(noised_params)

    def log_noise_latent_parameters(self, log_min: float, log_max: float, eps = 1e-6,
                                    keep_start_representable: bool = False):
        """Displace each crystal's latent vector by a random step with a log-uniform length, and assign the result in place.

        For each row of ``latent_params(gauge_fix_free_axes=False)``, draws a
        uniformly random unit direction in latent space and a step length
        ``10 ** u`` with ``u`` uniform on [log_min, log_max], adds the step, clips
        to [-1 + eps, 1 - eps], then applies ``latent_to_cell_params`` and
        ``clean_cell_parameters(mode='hard')``. The draws come from the global
        torch random generator.

        Parameters
        ----------
        log_min, log_max : float
            Base-10 log bounds of the step length, in latent units.
        eps : float
            Margin kept from the latent bounds when clipping.
        keep_start_representable : bool
            Default False (the behaviour above). True: the step is added to the UNCLIPPED latents
            (``latent_transform`` of ``full_cell_parameters()`` after ``canonicalize_zp_aunits``, without the
            [-1, 1] clip of ``latent_params``), and the clip above and the clamp in ``latent_to_cell_params`` are
            widened per entry to include the starting value: the displaced latents are clamped to
            [min(start, -1 + eps), max(start, 1 - eps)] and passed on with ``widen_to=start``. A crystal lying
            outside the latent box (e.g. a long reduced-cell axis) is then displaced from itself rather than from its
            clipped projection, a different crystal. ``clean_cell_parameters(mode='hard')`` still applies its own
            bounds, which are not widened: its cell-angle, centroid and rotation-magnitude bounds lie on or inside
            the latent box, so a value in those entries that is still outside the box after the displacement is
            clamped onto them. A row whose starting latents lie inside every clip and decode bound gets the same
            result as with False (same random draws).

        Returns
        -------
        torch.Tensor
            [num_graphs] the step length drawn for each row (before clipping), in latent units.
        """
        # keep_start_representable (default False = unchanged): the step is added to the UNCLIPPED latents and every
        # clip and clamp bound is widened per entry to include the starting value, so a crystal outside the latent box
        # (e.g. a long reduced-cell axis) is kicked from itself, not from its clipped projection (a different crystal).
        # A row whose starting latents lie inside every clip and decode bound gets the same result either way.
        # gauge-FREE -- same reasoning as noise_latent_parameters (P1 2026-08-24)
        if keep_start_representable:
            self.canonicalize_zp_aunits()  # as latent_params does, without its clip
            start = self.latent_transform(cell_params=self.full_cell_parameters())
            latents = start
        else:
            latents = self.latent_params(gauge_fix_free_axes=False)
        rand_dir = torch.randn_like(latents)
        rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
        u = torch.rand(len(latents), device=self.device)
        rand_magnitude = 10 ** (log_min + (log_max - log_min) * u)
        noised_latents = latents + rand_dir * rand_magnitude[:, None]
        if keep_start_representable:
            noised_latents = noised_latents.clamp(min=torch.clamp(start, max=-1 + eps), max=torch.clamp(start, min=1 - eps))
            self.latent_to_cell_params(noised_latents, widen_to=start)
        else:
            self.latent_to_cell_params(noised_latents.clip(min=-1 + eps, max=1 - eps))
        self.clean_cell_parameters(mode='hard')
        return rand_magnitude  # the step length drawn for each row (latent units)

    def zp1_std_cell_parameters(self):
        """
        Return standardized zp=1 cell parameters (first asymmetric unit only)
        write custom logic for zp>1 crystals

        Concatenates ``standardize_cell_lengths``, ``standardize_cell_angles`` and
        the first 3 columns of ``standardize_aunit_position`` and
        ``standardize_aunit_orientation``. ``self`` is not modified (a batch may
        cache ``asym_unit_dict``).

        Returns
        -------
        torch.Tensor
            [num_graphs, 12] standardized parameters.
        """
        standardized_aunit_lengths = self.standardize_cell_lengths()
        standardized_cell_angles = self.standardize_cell_angles()
        standardized_aunit_centroid = self.standardize_aunit_position()
        standardized_orientation = self.standardize_aunit_orientation()

        return torch.cat([standardized_aunit_lengths,
                          standardized_cell_angles,
                          standardized_aunit_centroid[:, :3],
                          standardized_orientation[:, :3]], dim=1)

    def destandardize_zp1_cell_parameters(self, std_cell_params):
        """Invert ``zp1_std_cell_parameters``.

        Parameters
        ----------
        std_cell_params : torch.Tensor
            [num_graphs, 12] standardized lengths, angles, centroid and orientation.

        Returns
        -------
        torch.Tensor
            [num_graphs, 12] cell lengths (Angstrom), angles (radians), fractional
            centroid (clipped to [0, 1 - 1e-4]) and rotation vector. ``self`` is not
            modified (a batch may cache ``asym_unit_dict``). The centroid step only
            broadcasts when ``max_z_prime == 1``.
        """
        (std_aunit_lengths,
         std_cell_angles,
         std_aunit_positions,
         std_aunit_orientations) = std_cell_params.split(3, dim=1)

        cell_lengths = self.destandardize_cell_lengths(std_aunit_lengths)
        cell_angles = self.destandardize_cell_angles(std_cell_angles)
        aunit_positions = self.destandardize_aunit_position(std_aunit_positions)
        aunit_orientations = self.destandardize_aunit_orientation(std_aunit_orientations)

        return torch.cat([
            cell_lengths, cell_angles, aunit_positions, aunit_orientations
        ], dim=1)

    def crystal_system(self):
        """Return the ``LATTICE_TYPE`` name ('triclinic', 'monoclinic', ...) of ``sg_ind``: a list with one per crystal for a batch, a string for a single crystal."""
        if self.is_batch:
            return [LATTICE_TYPE[int(ind)] for ind in self.sg_ind]
        else:
            return LATTICE_TYPE[int(self.sg_ind)]

    def validate_cell_params(self, check_crystal_system: bool = True):
        """
        Assert that the cell parameters are in range and, optionally, match the crystal system.

        checking crystal system slightly slower, can optionally skip

        Parameters
        ----------
        check_crystal_system : bool
            Also run ``validate_crystal_system`` (batch only).

        Returns
        -------
        bool
            True; a failed check raises instead.

        Raises
        ------
        AssertionError
            From ``validate_cell_params_ranges`` or ``validate_crystal_system``.
        RuntimeError
            From ``validate_crystal_system`` for an unrecognised lattice type.
        """
        self.validate_cell_params_ranges()
        if check_crystal_system:
            self.validate_crystal_system()
        return True

    def validate_cell_params_ranges(self, eps=1e-4):
        """Assert physical ranges on the cell parameters of ``self``; returns None.

        Checks ``0 <= aunit_centroid <= 1 - eps``, ``cell_lengths > 0``,
        ``0 < cell_angles < pi``, the norm of each ``aunit_orientation`` row in
        [0, 2 pi], and the last column of ``aunit_orientation`` non-negative. The
        norm is taken over the whole row and the sign check reads only the last
        column, so for ``max_z_prime > 1`` these two checks span all asymmetric
        units together and only the last unit's z component respectively.

        Raises
        ------
        AssertionError
            On the first failed check.
        """
        # assert valid ranges
        assert torch.all(self.aunit_centroid >= 0), "Aunit centroids must be greater than 0"
        assert torch.all(self.aunit_centroid <= 1 - eps), "Aunit centroids must be less than 0.999"
        assert torch.all(self.cell_lengths > 0), "Cell lengths must be positive"
        assert torch.all(self.cell_angles > 0), "Cell angles must be greater than 0"
        assert torch.all(self.cell_angles < torch.pi), "Cell angles must be less than pi"
        assert torch.all(torch.linalg.norm(self.aunit_orientation,
                                           dim=-1) <= 2 * torch.pi), "Cell orientation rotvec must have length <=2pi"
        assert torch.all(
            torch.linalg.norm(self.aunit_orientation, dim=-1) >= 0), "Cell orientation rotvec must have length >= 0"
        assert torch.all(self.aunit_orientation[:, -1] >= 0), "Cell orientation rotvec z component must be positive"

    def validate_crystal_system(self, atol=1e-3):  # todo cleanup/parallelize
        """Assert, crystal by crystal, that the cell matches the ``LATTICE_TYPE`` of its space group.

        Batch only. Monoclinic requires alpha = gamma = pi/2; orthorhombic all
        angles pi/2; tetragonal all angles pi/2 and a = b; cubic all angles pi/2
        and a = b = c (all exact comparisons). Hexagonal requires, within ``atol``,
        a = b, alpha = beta = pi/2, gamma = 2 pi/3, and c different from a.
        Triclinic is not checked.

        Parameters
        ----------
        atol : float
            Absolute tolerance for the hexagonal checks only.

        Returns
        -------
        bool
            True; a failed check raises instead.

        Raises
        ------
        AssertionError
            On a failed check, or if ``self`` is not a batch.
        RuntimeError
            For a lattice type not listed above.
        """
        lattices = self.crystal_system()
        # enforce agreement with crystal system
        if self.is_batch:
            right_angle = torch.tensor(torch.pi / 2, dtype=torch.float32, device=self.device)
            oblate_angle = torch.tensor(2 * torch.pi / 3, dtype=torch.float32, device=self.device)
            for ind in range(self.num_graphs):
                lattice = lattices[ind]
                cell_lengths = self.cell_lengths[ind]
                cell_angles = self.cell_angles[ind]
                if lattice.lower() == 'triclinic':
                    pass
                elif lattice.lower() == 'monoclinic':  # fix alpha and gamma
                    assert torch.all(cell_angles[0] == torch.pi / 2), "Error in monoclinic alpha angle"
                    assert torch.all(cell_angles[2] == torch.pi / 2), "Error in monoclinic gamma angle"
                elif lattice.lower() == 'orthorhombic':  # fix all angles
                    assert torch.all(cell_angles == torch.ones(3) * torch.pi / 2), "Error in orthorhombic cell angles"
                elif lattice.lower() == 'tetragonal':  # fix all angles and a & b vectors
                    assert torch.all(cell_angles == torch.ones(3) * torch.pi / 2), "Error in tetragonal cell angles"
                    assert torch.all(cell_lengths[0] == cell_lengths[1]), "Error in tetragonal cell lengths"
                elif lattice.lower() == 'hexagonal':  # for rhombohedral, all angles and lengths equal, but not 90.
                    # for truly hexagonal, alpha=90, gamma is 120, a=b!=c
                    # todo check rhobohedral business
                    assert torch.allclose(cell_lengths[0], cell_lengths[1], atol=atol), "hexagonal a ≠ b"
                    assert not torch.allclose(cell_lengths[2], cell_lengths[0], atol=atol), "hexagonal requires c ≠ a"
                    assert torch.allclose(cell_angles[0], right_angle, atol=atol), "hexagonal alpha ≠ 90°"
                    assert torch.allclose(cell_angles[1], right_angle, atol=atol), "hexagonal beta ≠ 90°"
                    assert torch.allclose(cell_angles[2], oblate_angle, atol=atol), "hexagonal gamma ≠ 120°"
                    pass
                elif lattice.lower() == 'cubic':  # all angles 90 all lengths equal
                    assert torch.all(cell_lengths == cell_lengths.mean()), "Error in cubic cell lengths"
                    assert torch.all(cell_angles == torch.pi / 2), "Error in cubic cell angles"
                else:
                    raise RuntimeError(f"{lattice} + ' is not a valid crystal lattice!')")
        else:
            assert False, "Lattice checks not impelemented for single crystals"

        return True

    def reparameterize_unit_cell(self,enforce_right_handedness: bool = False):
        """Recover asymmetric-unit parameters from the built unit cell.

        Requires ``unit_cell_pos``, ``unit_cell_batch``, ``T_cf``, ``z``, ``batch``,
        ``sym_mult`` and ``num_atoms`` (e.g. after ``mol2ucell``). For each crystal,
        ``parameterize_crystal_batch`` picks the canonical copy of the molecule in
        the unit cell (from heavy-atom centroids) and extracts its fractional
        centroid, rotation vector and handedness. For ``max_z_prime == 1`` it runs
        on ``self`` and overwrites ``self.pos`` with the chosen copies' Cartesian
        coordinates; for ``max_z_prime > 1`` it runs on ``split_to_zp1_batch()``
        and the per-unit results are padded out to ``max_z_prime`` slots (padded
        centroid and orientation slots are 0, padded handedness is 0). No other
        attribute of ``self`` is assigned.

        Parameters
        ----------
        enforce_right_handedness : bool
            Forwarded to ``parameterize_crystal_batch``; when True every returned
            handedness is +1.

        Returns
        -------
        aunit_centroid : torch.Tensor
            [num_graphs, 3 * max_z_prime] fractional centroids.
        aunit_orientation : torch.Tensor
            [num_graphs, 3 * max_z_prime] rotation vectors.
        handedness : torch.Tensor
            Long, +1 or -1 (0 for a degenerate principal-axis basis, and in padded
            slots): [num_graphs] when ``max_z_prime == 1``,
            [num_graphs, max_z_prime] otherwise.
        is_well_defined : list of bool or torch.Tensor
            Per crystal: a python list when ``max_z_prime == 1``, a [num_graphs]
            bool tensor (all units well defined) otherwise.
        pos : torch.Tensor
            [total_atoms, 3] Cartesian coordinates of the chosen asymmetric units,
            crystal by crystal.
        """
        if self.max_z_prime > 1:
            zp1_batch = self.split_to_zp1_batch()
            (aunit_centroid, aunit_orientation,
             handedness_list, is_well_defined, pos) = (
                parameterize_crystal_batch(zp1_batch,
                                           ASYM_UNITS,
                                           enforce_right_handedness=enforce_right_handedness,
                                           return_aunit=True))

            # have to pad out these to max_z_prime
            n = self.num_graphs
            mzp = self.max_z_prime
            device = aunit_centroid.device
            zp = self.z_prime.long().to(device)
            crystal_idx = torch.arange(n, device=device).repeat_interleave(zp)
            slot_idx = torch.arange(zp.sum(), device=device) - (torch.cumsum(zp, 0) - zp)[crystal_idx]

            aunit_centroid_p = aunit_centroid.new_zeros(n, mzp, 3)
            aunit_orientation_p = aunit_orientation.new_zeros(n, mzp, 3)
            handedness_p = handedness_list.new_zeros(n, mzp)
            iwd_p = torch.ones(n, mzp, dtype=torch.bool, device=device)

            aunit_centroid_p[crystal_idx, slot_idx] = aunit_centroid
            aunit_orientation_p[crystal_idx, slot_idx] = aunit_orientation
            handedness_p[crystal_idx, slot_idx] = handedness_list
            iwd_p[crystal_idx, slot_idx] = torch.as_tensor(is_well_defined, device=device)

            aunit_centroid = aunit_centroid_p.reshape(n, 3 * mzp)
            aunit_orientation = aunit_orientation_p.reshape(n, 3 * mzp)
            handedness_list = handedness_p
            is_well_defined = iwd_p.all(dim=1)
            combo_pos = []
            counter = 0
            for ind in range(self.num_graphs):
                pos_list = []
                for i in range(self.z_prime[ind].item()):
                    pos_list.extend(pos[counter + i])
                combo_pos.append(torch.stack(pos_list))
                counter += self.z_prime[ind]
            pos = combo_pos

        else:
            (aunit_centroid, aunit_orientation,
             handedness_list, is_well_defined, pos) = (
                parameterize_crystal_batch(self,
                                           ASYM_UNITS,
                                           enforce_right_handedness=enforce_right_handedness,
                                           return_aunit=True))

        return aunit_centroid, aunit_orientation, handedness_list.long(), is_well_defined, torch.cat(pos)

    def _transform_aunit_params(self, centroid_frac, orientation, handedness, R_frac, t_frac, wrap: bool = True):
        """
        Apply one fractional affine operation (R_frac, t_frac) -- a SYM_OPS entry
        or a normalizer coset representative, the math doesn't care which -- to
        this batch's (aunit_centroid, aunit_orientation, aunit_handedness),
        returning the new params describing the same operation applied to every
        atom. Used by get_fd_params to walk the Euclidean-normalizer cosets.
        """
        centroid_frac = centroid_frac[:, :3]
        orientation = orientation[:, :3]
        n = centroid_frac.shape[0]
        device, dtype = centroid_frac.device, centroid_frac.dtype

        if R_frac.ndim == 2:
            R_frac = R_frac[None].expand(n, -1, -1)
        if t_frac.ndim == 1:
            t_frac = t_frac[None].expand(n, -1)
        R_frac = R_frac.to(device=device, dtype=dtype)
        t_frac = t_frac.to(device=device, dtype=dtype)

        # centroid: fractional space is native to SYM_OPS/normalizer generators alike
        new_centroid = torch.einsum('nij,nj->ni', R_frac, centroid_frac) + t_frac
        if wrap:
            new_centroid = new_centroid - torch.floor(new_centroid)

        # orientation/handedness: need the CARTESIAN rotation part. This conjugation
        # is a no-op determinant-wise (det(R_cart) == det(R_frac) always, since
        # T_cf = inv(T_fc)), but it DOES matter for R_orient_new itself whenever
        # R_frac isn't central in O(3) -- i.e. for anything except pure inversion.
        R_cart = self.T_fc @ R_frac @ self.T_cf
        det = torch.linalg.det(R_cart)  # (N,)

        h_old = handedness.reshape(-1).to(dtype)
        h_new = det * h_old

        R_orient_old = rotvec2rotmat(orientation)
        diag_det = torch.zeros(n, 3, 3, device=device, dtype=dtype)
        diag_det[:, 0, 0] = det
        diag_det[:, 1, 1] = 1.
        diag_det[:, 2, 2] = 1.
        R_orient_new = R_cart @ R_orient_old @ diag_det
        orientation_new = rotmat2rotvec(R_orient_new)

        return new_centroid, orientation_new, h_new

    def get_fd_params(self, is_chiral: Optional[torch.Tensor] = None, normalizer_table: dict = NORMALIZER_OPS):
        """
        Reduce this batch's aunit_{centroid,orientation} to a canonical
        representative under the Euclidean normalizer N_E(G); aunit_handedness is
        used to build the candidates but is not returned. Assumes one sg_ind
        for the whole batch, matching the rest of the pipeline's calling convention.

        Algorithm: candidate 0 is the batch's current aunit params (assumed
        already folded into G's box -- true if it came from
        reparameterize_unit_cell or init_batch's sampling path). For each
        additional normalizer coset rep g_i: build the candidate via
        _transform_aunit_params, apply the chirality gate (drop g_i if
        det(W_i) < 0 and the molecule is chiral -- an improper normalizer element
        maps a chiral molecule to its enantiomer, a different crystal, not a
        duplicate description), fold the candidate's centroid back into G's own
        box by actually rebuilding it and calling the existing
        pose_aunit -> build_unit_cell -> reparameterize_unit_cell pipeline (reuses
        identify_canonical_asymmetric_unit exactly as-is, rather than re-deriving
        G's fold-back symbolically), then choose among the valid candidates
        lexicographically: smallest centroid x, then y, then z, then orientation
        x, y, z, each compared with a 1e-6 tolerance; a tie surviving every
        dimension takes the first candidate. Candidates whose fold-back is not
        well defined are excluded.

        Only the first asymmetric unit (columns :3) is used. ``self`` is not
        modified; the fold-back runs on clones. Requires ``T_fc``/``T_cf`` (from
        ``box_analysis``) and ``aunit_handedness``.

        Parameters
        ----------
        is_chiral : torch.Tensor, optional
            [num_graphs] bool, True where the molecule is not superimposable on
            its mirror image. Not computed here -- pull from wherever your
            featurization already flags this (RDKit CIP/stereocenter perception
            on the SMILES, most likely). When None, every improper coset
            representative (det(W) < 0) is rejected for every crystal, i.e. all
            molecules are treated as chiral.
        normalizer_table : dict
            Space-group string -> list of (W, w) coset representatives (3x3
            fractional rotation part, 3-vector translation). Defaults to
            ``NORMALIZER_OPS``.

        Returns
        -------
        centroid : torch.Tensor
            [num_graphs, 3] canonicalized aunit centroid, fractional (C[best, idx]).
        orientation : torch.Tensor
            [num_graphs, 3] canonicalized aunit rotation vector (O[best, idx]).

        Raises
        ------
        AssertionError
            If the batch has more than one ``sg_ind``.
        NotImplementedError
            If ``normalizer_table`` has no coset representatives for the space group.
        """
        sg_ind = int(self.sg_ind[0])
        assert torch.all(self.sg_ind == sg_ind), \
            "get_fd_params assumes one space group per batch, as elsewhere in the pipeline"

        generators = normalizer_table.get(str(sg_ind), [])
        if len(generators) == 0:
            raise NotImplementedError(
                f"No normalizer coset representatives on file for space group {sg_ind}. "
                f"Pull them from https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-norm"
                f"?gnum={sg_ind}&norgens=en (blocks automated fetches -- needs a human) "
                f"or ITA Vol. A1 Table 15.2, then add an entry to NORMALIZER_OPS."
            )

        n = self.num_graphs
        device = self.device
        dtype = self.aunit_centroid.dtype

        cand_centroids = [self.aunit_centroid[:, :3].clone()]
        cand_orients = [self.aunit_orientation[:, :3].clone()]
        cand_valid = [torch.ones(n, dtype=torch.bool, device=device)]

        for (W, w) in generators:
            W_t = torch.as_tensor(W, dtype=dtype, device=device)
            w_t = torch.as_tensor(w, dtype=dtype, device=device)
            det_W = torch.linalg.det(W_t)

            c, o, h = self._transform_aunit_params(
                self.aunit_centroid, self.aunit_orientation, self.aunit_handedness, W_t, w_t,
            )

            # fold this candidate back into G's box, reusing the existing pipeline
            # end to end rather than re-deriving G's own fold-back symbolically
            trial = self.clone()
            trial.aunit_centroid = c
            trial.aunit_orientation = o
            trial.aunit_handedness = h[:, None]
            trial.pose_aunit()
            trial.build_unit_cell()
            c_folded, o_folded, h_folded, well_defined, _ = trial.reparameterize_unit_cell()

            if is_chiral is None:
                chirality_ok = (~True) | (det_W > 0)
            else:
                chirality_ok = (~is_chiral) | (det_W > 0)
            valid = chirality_ok & torch.as_tensor(well_defined, device=device, dtype=torch.bool)

            cand_centroids.append(c_folded[:, :3])
            cand_orients.append(o_folded[:, :3])
            cand_valid.append(valid)

        C = torch.stack(cand_centroids, dim=0)  # (K, n, 3)
        O = torch.stack(cand_orients, dim=0)    # (K, n, 3)
        V = torch.stack(cand_valid, dim=0)      # (K, n)

        # Tie-break: dimension-by-dimension (x, then y, then z, then orientation),
        # matching identify_canonical_asymmetric_unit's own convention -- NOT a
        # joint norm. This matters, not just style: found empirically (7/3) that a
        # joint ||centroid||+||orientation|| norm under-reduces whenever a
        # normalizer generator's G-refold couples two dimensions together (P21/c's
        # 21-screw-mediated fold ties x and z into a single joint flip, (x,z) ->
        # (-x,-z+1/2), rather than touching them independently). A joint norm is
        # symmetric under swapping such coupled dimensions, so it can't
        # consistently resolve which one absorbs the extra reduction -- it picks
        # whichever raw value is smaller, which varies structure to structure and
        # leaves BOTH dimensions spanning their full pre-reduction range in
        # aggregate across a dataset, rather than confining either one. Sequential
        # dimension-ordered comparison fixes this: the first dimension checked
        # gets to use its full available freedom (quartering, for the coupled
        # pair), and whatever's left over resolves the next. Verified numerically
        # against the actual sg-14 orbit structure: joint norm left x and z both
        # spanning the full [0, 0.5) each; x-first lexicographic confined x to
        # [0, 0.25] and z to [0, 0.5) -- exactly the missing extra factor of 2.
        idx = torch.arange(n, device=device)
        keys = torch.cat([C, O], dim=-1)  # (K, n, 6): centroid xyz, then orientation xyz
        keys = torch.where(V[..., None].bool(), keys, torch.full_like(keys, float('inf')))
        still_tied = torch.ones(len(cand_centroids), n, dtype=torch.bool, device=device)
        atol = 1e-6
        for d in range(keys.shape[-1]):
            col = torch.where(still_tied, keys[..., d], torch.full_like(keys[..., d], float('inf')))
            dim_min = col.min(dim=0, keepdim=True).values
            still_tied = still_tied & (col <= dim_min + atol)
        # genuine ties surviving every dimension are measure-zero for generic
        # (non-special-position) structures; take the first survivor for determinism
        best = still_tied.float().argmax(dim=0)  # (n,)

        return C[best, idx], O[best, idx]

    def fold_to_fd(self, is_chiral: Optional[torch.Tensor] = None):
        """
        In-place counterpart to get_fd_params: overwrites this batch's
        aunit_centroid and aunit_orientation with their canonical N_E(G)
        fundamental-domain representatives, each [num_graphs, 3].

        ``aunit_handedness``, the box attributes and ``pos`` are not updated.

        Parameters
        ----------
        is_chiral : torch.Tensor, optional
            Forwarded to ``get_fd_params``.
        """
        self.aunit_centroid, self.aunit_orientation = self.get_fd_params(is_chiral)

    def split_cell_params(self):
        """Return ``full_cell_parameters()`` split into (cell_lengths [num_graphs, 3], cell_angles [num_graphs, 3], aunit centroids [num_graphs, 3 * max_z_prime], aunit orientations [num_graphs, 3 * max_z_prime])."""
        cell_lengths, cell_angles, aunit_positions, aunit_orientations = torch.split(self.full_cell_parameters(),
                                                                                     [3, 3, 3 * self.max_z_prime,
                                                                                      3 * self.max_z_prime], dim=1)
        return cell_lengths, cell_angles, aunit_positions, aunit_orientations

    def clean_cell_parameters(self, mode: str = 'hard',
                              canonicalize_orientations: bool = True,
                              angle_pad: float = 0.6,
                              length_pad: float = 3.0,
                              constrain_z: bool = False,
                              ):
        """
        force cell parameters into physical ranges, in place.

        In order, reassigns on ``self``:

        - ``cell_lengths``: floored at ``length_pad`` ('hard': clip; 'soft':
          softplus-shifted lower bound slightly above ``length_pad``);
        - ``cell_angles``: bounded to pi/2 +/- (pi/2) * ``angle_pad``
          (``enforce_1d_bound``: hardtanh for 'hard', tanh for 'soft');
        - ``aunit_centroid``: bounded to [0, 1] in asymmetric-unit-box units, then
          converted back to fractional coordinates and clipped to [0, 1 - 1e-4];
        - ``aunit_orientation``: if ``constrain_z``, each unit's z component is
          bounded to [0, 2 pi]; then each unit's rotation-vector norm is bounded to
          pi +/- 0.999 pi, keeping its direction;
        - ``cell_lengths`` and ``cell_angles`` again, via ``enforce_crystal_system``;
        - ``aunit_orientation`` again if ``canonicalize_orientations``: each unit's
          vector with z < 0 replaced by the same rotation on the +z half-space
          (``canonicalize_rotvec``);

        and finally calls ``box_analysis``.

        Parameters
        ----------
        mode : str
            ``'hard'`` or ``'soft'``; any other value raises ValueError in
            ``enforce_1d_bound``.
        canonicalize_orientations : bool
            Must be False when ``constrain_z`` is True. Moves each unit's rotation
            vector to the +z half-space without changing the rotation.
        angle_pad : float
            Fraction of pi/2 allowed on either side of pi/2.
        length_pad : float
            Minimum cell length (Angstrom).
        constrain_z : bool
            Bound each unit's rotation-vector z component to [0, 2 pi].

        Raises
        ------
        AssertionError
            If both ``constrain_z`` and ``canonicalize_orientations`` are True.

        Notes
        -----
        The length floor is independent of the length range used by
        ``latent_transform``. The centroid bound, [0, 1] in asymmetric-unit-box
        units, is the latent centroid range, and at the default ``angle_pad=0.6``
        the angle bound [0.2 pi, 0.8 pi] equals the latent angle range.
        """

        # cell lengths have to be positive nonzero
        if mode == 'hard':
            self.cell_lengths = self.cell_lengths.clip(min=length_pad)
        elif mode == 'soft':
            self.cell_lengths = softplus_shift(
                self.cell_lengths - length_pad) + length_pad  # enforces a minimum value of length_pad

        # range from (0,pi) with padding to prevent too-skinny cells
        self.cell_angles = enforce_1d_bound(self.cell_angles,
                                            x_span=torch.pi / 2 * angle_pad,
                                            x_center=torch.pi / 2,
                                            mode=mode)

        # if enforce_niggli:
        #     self.enforce_niggli_box(mode)

        # positions must be on 0->1-eps in the asymmetric unit
        aunit_scaled_pos = self.scale_centroid_to_aunit()
        cleaned_aunit_scaled_pos = enforce_1d_bound(aunit_scaled_pos, x_span=0.5, x_center=0.5, mode=mode)
        self.aunit_centroid = self.assign_aunit_centroid(self.scale_centroid_to_unit_cell(cleaned_aunit_scaled_pos))

        # enforce range on vector norm
        if constrain_z:  # enforce a priori only positive Z possible
            assert not canonicalize_orientations, "Either enforce z positive, or post-canonicalize negative z's"
            # forces Z within the range 0-2pi
            new_orientation = []
            for zp in range(self.max_z_prime):
                fixed_z = enforce_1d_bound(
                    self.aunit_orientation[:, 2 + 3 * zp],
                    x_span=torch.pi,
                    x_center=torch.pi,
                    mode=mode
                )
                new_orientation.append(self.aunit_orientation[:, 3 * zp:3 * zp + 2])
                new_orientation.append(fixed_z[:, None])
            self.aunit_orientation = torch.cat(new_orientation, dim=1)

        new_orientation = []
        for zp in range(self.max_z_prime):
            norm = torch.linalg.norm(self.aunit_orientation[:, zp * 3:3 + zp * 3], dim=1)
            new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode=mode)  # MUST be nonzero
            new_orientation.append(self.aunit_orientation[:, zp * 3:3 + zp * 3] / norm[:, None].clamp(min=1e-6) * new_norm[:, None])
        self.aunit_orientation = torch.cat(new_orientation, dim=1)

        # enforce agreement with crystal system
        self.cell_lengths, self.cell_angles = enforce_crystal_system(self.cell_lengths,
                                                                     self.cell_angles,
                                                                     self.sg_ind,
                                                                     )

        # enforce z component in the upper half-plane
        if canonicalize_orientations:  # converts the vector to its duplicate in the upper half-plane
            new_orientation = []
            for zp in range(self.max_z_prime):  # clone: canonicalize_rotvec writes in place
                new_orientation.append(canonicalize_rotvec(self.aunit_orientation[:, zp * 3:3 + zp * 3].clone()))
            self.aunit_orientation = torch.cat(new_orientation, dim=1)

        # update cell vectors
        self.box_analysis()

    def enforce_niggli_box(self, mode, apply_plane_shift: bool = False):
        """Disabled: always fails with ``AssertionError``; ``mode`` and ``apply_plane_shift`` are unused."""

        assert False, "This needs to be reimplemented with our new reduction workflow"
        # # enforce the scaling factor b/c and a/b in the range [0, 1]
        # a, b, c = self.cell_lengths.split(1, 1)
        # b_scale = enforce_1d_bound((b / c), 0.5, 0.5, mode=mode)
        # a_scale = enforce_1d_bound((a / b), 0.5, 0.5, mode=mode)
        # self.cell_lengths = torch.cat([
        #     a_scale * b_scale * c,
        #     b_scale * c,
        #     c],
        #     dim=1)
        #
        # # enforce the scaling factor cos(alpha) / (b/2c) in the range [0, 1]
        # al_cos, be_cos, ga_cos = self.cell_angles.cos().split(1, 1)
        # a, b, c = self.cell_lengths.split(1, 1)
        # al_cos_max = (b / 2 / c)
        # be_cos_max = (a / 2 / c)
        # ga_cos_max = (a / 2 / b)
        #
        # # now improved - including obtuse cells
        # al_cos_scale = enforce_1d_bound(al_cos / al_cos_max, 1.0, 0.0, mode=mode)
        # be_cos_scale = enforce_1d_bound(be_cos / be_cos_max, 1.0, 0.0, mode=mode)
        # ga_cos_scale = enforce_1d_bound(ga_cos / ga_cos_max, 1.0, 0.0, mode=mode)
        #
        # # limit it here due to instability in arccos
        # al = torch.arccos(al_cos_max * al_cos_scale.clip(min=-0.99, max=0.99))
        # be = torch.arccos(be_cos_max * be_cos_scale.clip(min=-0.99, max=0.99))
        # ga = torch.arccos(ga_cos_max * ga_cos_scale.clip(min=-0.99, max=0.99))
        #
        # self.cell_angles = torch.cat([al, be, ga], dim=1)
        # # here, enforce positivity of niggli overlap
        # if apply_plane_shift:
        #     self.cell_angles = enforce_niggli_plane(self.cell_lengths,
        #                                             self.cell_angles,
        #                                             mode='shift')

    def aunit_volume(self):
        """Return ``cell_volume / sym_mult`` (Angstrom^3 per symmetry copy); requires ``box_analysis`` to have run."""
        return self.cell_volume / self.sym_mult

    def zp1_cell_parameters(self):
        """
        return the zp=1 total cell parameter tensor
        for zp>1 crystals use custom logic

        Returns
        -------
        torch.Tensor
            [num_graphs, 12]: ``cell_lengths``, ``cell_angles`` and the first
            asymmetric unit's ``aunit_centroid`` and ``aunit_orientation`` columns.
        """
        return torch.cat([self.cell_lengths,
                          self.cell_angles,
                          self.aunit_centroid[:, :3],
                          self.aunit_orientation[:, :3]], dim=1)

    def full_cell_parameters(self):
        """Return a new [num_graphs, 6 + 6 * max_z_prime] tensor ``[cell_lengths | cell_angles | aunit_centroid | aunit_orientation]``."""
        return torch.cat([self.cell_lengths,
                          self.cell_angles,
                          self.aunit_centroid,
                          self.aunit_orientation], dim=1)

    def standardize_cell_lengths(self):
        """
        Standardize the cell lengths tensor
        1. convert to asymmetric unit basis
        2. norm by mol radius
        3. standardize by stats
        mean = [1.0563, 0.7978, 1.7570]
        std = [0.6001, 0.5115, 0.7147]

        The per-axis stats are averaged: all three axes use the single mean
        (mean of the three means) and single std (mean of the three stds).
        ``radius`` is used without Z' scaling.

        Returns
        -------
        torch.Tensor
            [num_graphs, 3] standardized lengths; ``self`` is not modified (a
            batch may cache ``asym_unit_dict``).
        """
        aunit_lengths = self.scale_lengths_to_aunit()
        normed_aunit_lengths = self.norm_by_radius(aunit_lengths)
        # flatten stats across length dimensions
        mean = torch.tensor([1.0563, 0.7978, 1.7570], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.6001, 0.5115, 0.7147], dtype=torch.float32, device=self.device)
        mean = mean.mean() * torch.ones_like(mean)
        std = std.mean() * torch.ones_like(std)
        return (normed_aunit_lengths - mean[None, :]) / std[None, :]

    def destandardize_cell_lengths(self, std_aunit_lengths):
        """
        Destandardize the cell lengths tensor
        1. destandardize by stats
        2. denorm by mol radius
        3. convert to unit cell basis
        mean = [1.0563, 0.7978, 1.7570]
        std = [0.6001, 0.5115, 0.7147]

        Uses the same averaged mean and std as ``standardize_cell_lengths``.

        Parameters
        ----------
        std_aunit_lengths : torch.Tensor
            [num_graphs, 3] standardized lengths.

        Returns
        -------
        torch.Tensor
            [num_graphs, 3] unit-cell lengths (Angstrom).
        """
        # flatten stats across length dimensions
        mean = torch.tensor([1.0563, 0.7978, 1.7570], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.6001, 0.5115, 0.7147], dtype=torch.float32, device=self.device)
        mean = mean.mean() * torch.ones_like(mean)
        std = std.mean() * torch.ones_like(std)
        normed_aunit_lengths = std_aunit_lengths * std[None, :] + mean[None, :]
        aunit_lengths = self.denorm_by_radius(normed_aunit_lengths)
        return self.scale_aunit_lengths_to_unit_cell(aunit_lengths)

    def standardize_cell_angles(self):
        """
        Return ``(cell_angles - mean) / std`` per axis, [num_graphs, 3] (angles in radians).

        Using triclinic CSD stats
        mean = [1.5656, 1.5728, 1.5579]
        std = [0.2339, 0.2075, 0.2598]
        """
        mean = torch.tensor([1.5656, 1.5728, 1.5579], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.2339, 0.2075, 0.2598], dtype=torch.float32, device=self.device)
        return (self.cell_angles - mean[None, :]) / std[None, :]

    def destandardize_cell_angles(self, std_cell_angles):
        """
        Return ``std_cell_angles * std + mean`` per axis, [num_graphs, 3] radians; the inverse of ``standardize_cell_angles``.

        Using triclinic CSD stats
        mean = [1.5656, 1.5728, 1.5579]
        std = [0.2339, 0.2075, 0.2598]
        """
        mean = torch.tensor([1.5656, 1.5728, 1.5579], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.2339, 0.2075, 0.2598], dtype=torch.float32, device=self.device)
        return std_cell_angles * std[None, :] + mean[None, :]

    def standardize_aunit_position(self):
        """
        Return ``(scale_centroid_to_aunit() - 0.5) / 0.289``, [num_graphs, 3 * max_z_prime].

        Treats asymmetric-unit-box coordinates as a
        uniform distribution on 0-1
        mean = 0.5
        std = 0.289
        """
        aunit_mean = 0.5
        aunit_std = 0.289
        return (self.scale_centroid_to_aunit() - aunit_mean) / aunit_std

    def destandardize_aunit_position(self, std_aunit_position):
        """
        Return ``scale_centroid_to_unit_cell(std_aunit_position * 0.289 + 0.5)``: fractional centroids clipped to [0, 1 - 1e-4].

        Inverse of ``standardize_aunit_position``, which assumes a
        uniform distribution on 0-1
        mean = 0.5
        std = 0.289
        """
        aunit_mean = 0.5
        aunit_std = 0.289
        return self.scale_centroid_to_unit_cell(std_aunit_position * aunit_std + aunit_mean)

    def standardize_aunit_orientation(self):
        """
        Return ``(aunit_orientation[:, :3] - [0, 0, pi/2]) / [2.08, 2.08, 1.38]``, [num_graphs, 3] (first asymmetric unit only).

        functional only on zp=1 crystals. write custom logic for zp>1

        Notes
        -----
        Recorded derivation of the stats::

            num_graphs = 100000
            random_vectors = torch.randn(size=(num_graphs, 3))

            # set norms uniformly between 0-2pi
            norms = random_vectors.norm(dim=1)
            applied_norms = (torch.rand(num_graphs) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
            random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]

            random_vectors = canonicalize_rotvec(random_vectors)
            std = random_vectors.std(0)
            mean = random_vectors.mean(0)
        """
        orientation_stds = torch.tensor([[2.08, 2.08, 1.38]], dtype=torch.float32, device=self.device)
        orientation_means = torch.tensor([[0, 0, torch.pi / 2]], dtype=torch.float32, device=self.device)
        return (self.aunit_orientation[:, :3] - orientation_means) / orientation_stds

    def destandardize_aunit_orientation(self, std_aunit_orientation):
        # destandardize aunit orientation
        """
        Return ``std_aunit_orientation * [2.08, 2.08, 1.38] + [0, 0, pi/2]``; the inverse of ``standardize_aunit_orientation``.

        Parameters
        ----------
        std_aunit_orientation : torch.Tensor
            [num_graphs, 3] standardized rotation vectors.

        Returns
        -------
        torch.Tensor
            [num_graphs, 3] rotation vectors.

        Notes
        -----
        Recorded derivation of the stats::

            num_graphs = 100000
            random_vectors = torch.randn(size=(num_graphs, 3))

            # set norms uniformly between 0-2pi
            norms = random_vectors.norm(dim=1)
            applied_norms = (torch.rand(num_graphs) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
            random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]

            random_vectors = canonicalize_rotvec(random_vectors)
            std = random_vectors.std(0)
            mean = random_vectors.mean(0)
        """
        orientation_stds = torch.tensor([[2.08, 2.08, 1.38]], dtype=torch.float32, device=self.device)
        orientation_means = torch.tensor([[0, 0, torch.pi / 2]], dtype=torch.float32, device=self.device)
        return std_aunit_orientation * orientation_stds + orientation_means

    def _build_feature_labels(self, space):
        lattice_features = ['a', 'b', 'c',
                            r'$\alpha$', r'$\beta$', r'$\gamma$']
        if self.max_z_prime == 1:
            lattice_features.extend([
                f'u', f'v', f'w',
            ])
            if space == 'latent':
                lattice_features.extend([
                    f'θ', f'φ', f'r'
                ])
            else:
                lattice_features.extend([
                    f'x', f'y', f'z'
                ])

        else:
            for zp in range(self.max_z_prime):
                lattice_features.extend([
                    f'aunit{zp} u', f'aunit{zp} v', f'aunit{zp} w',
                ])
            for zp in range(self.max_z_prime):
                if space == 'latent':
                    lattice_features.extend([
                        f'θ{zp}', f'φ{zp}', f'r{zp}'
                    ])
                else:
                    lattice_features.extend([
                        f'x{zp}', f'y{zp}', f'z{zp}'
                    ])
        return lattice_features

    def _set_cell_ranges(self, space, samples):
        if space == 'latent':
            custom_ranges = {}
            for ind in range(6 + self.max_z_prime * 6):
                custom_ranges.update({ind: [-1.1, 1.1]})

        elif space == 'standard':
            custom_ranges = {
                0: [-1.1, 1.1],  # for cell_a
                1: [-1.1, 1.1],  # for cell_b
                2: [-1.1, 1.1],  # for cell_c
                3: [-1.1, 1.1],  # for cell_alpha
                4: [-1.1, 1.1],  # for cell_beta
                5: [-1.1, 1.1],  # for cell_gamma
            }
            for ind in range(self.max_z_prime * 6):
                custom_ranges.update({6 + ind: [-1.1, 1.1]})

        else:
            custom_ranges = {ind: [np.amin(samples[:, ind]), np.amax(samples[:, ind])]
                             for ind in range(3)}
            custom_ranges.update(
                {3: [np.pi / 10, 0.9 * np.pi],
                 4: [np.pi / 10, 0.9 * np.pi],
                 5: [np.pi / 10, 0.9 * np.pi]
                 }
            )
            for ind in range(self.max_z_prime):
                custom_ranges.update({
                    6 + ind * 3 + 0: [0, 1.1],
                    6 + ind * 3 + 1: [0, 1.1],
                    6 + ind * 3 + 2: [0, 1.1],
                })
            for ind in range(self.max_z_prime):
                custom_ranges.update({
                    6 + self.max_z_prime * 3 + ind * 3 + 0: [-2 * np.pi, 2 * np.pi],
                    6 + self.max_z_prime * 3 + ind * 3 + 1: [-2 * np.pi, 2 * np.pi],
                    6 + self.max_z_prime * 3 + ind * 3 + 2: [0, 2 * np.pi],
                })

        return custom_ranges

    def _collect_sample_dists(self, samples, ref_dist, quantiles, split_by_sg, split_by_zp, aux_dists,
                              override_energy=None):
        num_dists = 1
        dists = []
        dist_names = []
        if ref_dist is not None:  # this should be first
            assert ref_dist.shape[1] == samples.shape[1]
            num_dists += 1
            dist_names += ['Reference']
            if torch.is_tensor(ref_dist):
                dists.append(ref_dist.cpu().detach().numpy())
            else:
                dists.append(ref_dist)

        dist_names.append('Samples')
        dists.append(samples)

        if quantiles is not None:
            if override_energy is not None:
                energies = override_energy
            else:
                energies = self.lj.cpu().detach().numpy()
            for q in quantiles:
                num_dists += 1
                dist_names += [f'{q} Quantile']
                dists.append(samples[energies < np.quantile(energies, q)])

        if split_by_sg:
            for sg in torch.unique(self.sg_ind):
                good_inds = self.sg_ind == sg
                dist_names += [f'SG={int(sg)}']
                dists.append(samples[good_inds])
                num_dists += 1

        if split_by_zp:
            for zp in torch.unique(self.z_prime):
                good_inds = self.z_prime == zp
                dist_names += [f"Z'={int(zp)}"]
                dists.append(samples[good_inds])
                num_dists += 1

        if aux_dists is not None:
            for ind, aux_dist in enumerate(aux_dists):
                dist_names += [f'Aux {ind}']
                dists.append(aux_dist)
                num_dists += 1

        return num_dists, dist_names, dists

    def plot_batch_density_funnel(self,
                                  renderer: Optional[str] = None,
                                  show: bool = True,
                                  return_fig: bool = False,
                                  split_by_sg: bool = False,
                                  split_by_zp: bool = False,
                                  override_energy: Union[str, torch.Tensor] = None,
                                  color_flag: torch.Tensor = None,
                                  show_colorbar: bool = False,
                                  max_y_quantile: Optional[float] = None,
                                  overwrite_yaxis_title: Optional[str] = None,
                                  ):
        """Scatter plot of energy against ``packing_coeff`` with violin marginals (plotly).

        The energy is ``log_rescale_positive`` of ``self.lj`` (set by
        ``build_and_analyze``) unless overridden. Points are coloured by
        ``color_flag`` if given, else by space group, Z', or 2D point density.

        Parameters
        ----------
        renderer : str, optional
            Passed to ``fig.show``.
        show : bool
            Show the figure.
        return_fig : bool
            Return the figure; otherwise return None.
        split_by_sg : bool
            Colour by space group (takes precedence over ``split_by_zp``).
        split_by_zp : bool
            Colour by Z'.
        override_energy : torch.Tensor or str, optional
            Energy tensor, or the name of an attribute of ``self`` to use as energy.
        color_flag : torch.Tensor, optional
            Per-crystal values used as a discrete colour key.
        show_colorbar : bool
            Show the colour bar.
        max_y_quantile : float, optional
            Upper y-axis limit as this quantile of the energies; otherwise
            ``min(10, max + 5% of range)``.
        overwrite_yaxis_title : str, optional
            y-axis title; defaults to 'Energy per Atom /Arb Units'.

        Returns
        -------
        plotly.graph_objects.Figure or None
        """

        if override_energy is None:
            energy = (log_rescale_positive(self.lj)).cpu().detach()
        else:
            if isinstance(override_energy, torch.Tensor):
                energy = (log_rescale_positive(override_energy)).cpu().detach()
            elif isinstance(override_energy, str):
                energy = log_rescale_positive(self[override_energy]).cpu().detach()
            else:
                assert False, "override_energyu must be tensor or string"

        xy = np.vstack([self.packing_coeff.cpu().detach(), energy])
        try:
            z = get_point_density(xy, bins=25)
        except:
            z = np.ones(xy.shape[1])

        scatter_dict = {'energy': energy,
                        'packing_coefficient': self.packing_coeff.cpu().detach(),
                        }
        if color_flag is not None:
            color_tag = 'Custom Flag'
            scatter_dict.update({'Custom Flag': color_flag})
            cscale = px.colors.qualitative.Dark24
        else:
            cscale = px.colors.cyclical.IceFire
            if split_by_sg:
                if split_by_zp:
                    print("Cannot split by both z prime and space group!")
                color_tag = 'Space Group'
                scatter_dict.update({'Space Group': [str(int(sg)) for sg in self.sg_ind]})
            elif split_by_zp:
                color_tag = "Z'"
                scatter_dict.update({"Z'": [str(int(zp)) for zp in self.z_prime]})
            else:
                color_tag = 'Point Density'
                scatter_dict.update({'Point Density': z})

        opacity = max(0.25, 1 - self.num_graphs / 5e4)
        df = pd.DataFrame.from_dict(scatter_dict)

        fig = px.scatter(df,
                         x='packing_coefficient', y='energy',
                         color=color_tag,
                         color_continuous_scale=cscale,
                         marginal_x='violin', marginal_y='violin',
                         color_discrete_sequence=px.colors.qualitative.Set3 if color_tag == 'Space Group' else None,
                         opacity=opacity
                         )

        fig.update_layout(yaxis_title='Energy', xaxis_title='Packing Coeff')
        if max_y_quantile is not None:
            max_y = np.quantile(df['energy'], max_y_quantile)
        else:
            max_y = min(10, np.amax(df['energy']) + np.ptp(df['energy']) * 0.05)

        fig.update_layout(yaxis_range=[np.amin(df['energy']) - np.ptp(df['energy']) * 0.05, max_y],
                          xaxis_range=[max(0, np.amin(df['packing_coefficient']) * 0.95),
                                       min(1, np.amax(df['packing_coefficient']) * 1.05)],
                          )

        if overwrite_yaxis_title is not None:
            yaxis_title = overwrite_yaxis_title
        else:
            yaxis_title = r'Energy per Atom /Arb Units'
        fig.update_layout(
            xaxis_title=r"Packing Coefficient",
            yaxis_title=yaxis_title,
        )
        fig.update_traces(
            marker=dict(
                size=5,
                line=dict(width=0.3, color='rgba(0,0,0,0.3)'),
                opacity=opacity,
            )
        )
        if color_tag == 'Point Density' and show_colorbar:
            fig.update_layout(coloraxis_colorbar=dict(
                title="Point Density (Normed)",
                tickfont=dict(size=10),
                title_font=dict(size=11),
            ))
        fig.update_traces(selector=dict(type='violin'), spanmode='hard')
        fig.update_traces(selector=dict(type='violin'), line=dict(width=0.6, color='black'))
        fig.update_layout(
            font=dict(family="Helvetica", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=20, t=40, b=50),
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.15)',
            gridwidth=0.8,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            row=1, col=1  # target the main scatter subplot
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.15)',
            gridwidth=0.8,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            row=1, col=1  # target the main scatter subplot
        )
        if not show_colorbar:
            fig.update_layout(coloraxis_showscale=False)
        if show:
            fig.show(renderer=renderer)

        if return_fig:
            return fig
        return None

    def plot_batch_cell_params(self, space='real',
                               renderer: Optional[str] = None,
                               quantiles: Optional[Iterable[float]] = None,
                               ref_dist: Optional[torch.Tensor] = None,
                               aux_dists: Optional[list] = None,
                               show: bool = True,
                               return_fig: bool = False,
                               split_by_sg: bool = False,
                               split_by_zp: bool = False,
                               n_kde: int = 200,
                               bw_factor: float = 0.05,
                               override_energy: Optional[torch.Tensor] = None,
                               gauge_fix_free_axes: bool = True, ):
        """Plot one-sided violin (KDE) distributions of every cell parameter (plotly), one subplot per parameter.

        Batch only: for a single crystal it prints a message and returns None.
        ``space='latent'`` goes through ``latent_params``, which reassigns the
        aunit attributes of ``self`` (see ``latent_params``).

        Parameters
        ----------
        space : str
            ``'real'`` (``full_cell_parameters``), ``'latent'`` (``latent_params``)
            or ``'standard'`` (``zp1_std_cell_parameters``, 12 columns).
        renderer : str, optional
            Passed to ``fig.show``.
        quantiles : iterable of float, optional
            For each q, adds the subset of samples with energy below the q-quantile
            (energy is ``self.lj`` unless ``override_energy`` is given).
        ref_dist : torch.Tensor, optional
            Reference samples, same column count, plotted first.
        aux_dists : list, optional
            Additional sample arrays to overlay.
        show : bool
            Show the figure.
        return_fig : bool
            Return the figure; otherwise return None.
        split_by_sg : bool
            Add one distribution per space group.
        split_by_zp : bool
            Add one distribution per Z'.
        n_kde : int
            Number of KDE evaluation points per violin.
        bw_factor : float
            KDE bandwidth factor.
        override_energy : torch.Tensor, optional
            Energies used for ``quantiles`` instead of ``self.lj``.
        gauge_fix_free_axes : bool
            Forwarded to ``latent_params`` when ``space='latent'``.

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        if not self.is_batch:
            print("Cell statistics only works for a batch of crystal data objects")
            return None

        lattice_features = self._build_feature_labels(space=space)
        samples = self._get_samples(space, gauge_fix_free_axes=gauge_fix_free_axes)
        num_dists, dist_names, dists = self._collect_sample_dists(samples, ref_dist, quantiles, split_by_sg,
                                                                  split_by_zp, aux_dists, override_energy)
        # delete or NaN unused higher Z' elements
        # 1d Histograms
        fig = make_subplots(rows=int(2 + 2 * self.max_z_prime),
                            cols=3,
                            subplot_titles=lattice_features)
        colors = self._get_color_set(num_dists)
        ranges = self._set_cell_ranges(space, samples)
        for i in range(len(lattice_features)):
            for j in range(num_dists):
                self._add_violin(
                    fig, dists[j][:, i], dist_names[j], colors[j], i, ranges[i],
                    n_kde, bw_factor
                )

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          violinmode='overlay',
                          legend=dict(
                              orientation="h",
                              yanchor="bottom",
                              y=1.05,
                              xanchor="center",
                              x=0.5,
                              bgcolor='rgba(0,0,0,0)',

                          ),
                          margin=dict(l=40, r=20, t=50, b=50),
                          font=dict(family="Helvetica", size=12, color='black'),
                          )

        for i in range(6 + self.max_z_prime * 6):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.update_xaxes(range=ranges[i], row=row, col=col)
        fig.update_xaxes(
            showgrid=False, zeroline=False, ticks='outside',
            tickwidth=1, mirror=True
        )
        # fig.update_yaxes(
        #     showgrid=False, zeroline=False, ticks='outside',
        #     tickwidth=1, mirror=True
        # )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, ticks='',
            mirror=True
        )
        if len(dists) > 1:
            fig.update_traces(opacity=0.5)
        if show:
            fig.show(renderer=renderer)
        if return_fig:
            return fig
        else:
            return None

    def plot_batch_staircase(self, space='real',
                             renderer=None,
                             show=True,
                             return_fig=False,
                             mode='contour',
                             cmap='icefire',
                             nbins=25,
                             colorbar=False,
                             ref_dist=None,
                             axis_range=None,
                             min_span=None,
                             ):
        """axis_range: (lo, hi) used for EVERY dimension's bins and axes -- pass (-1, 1) in latent space.
        min_span: without axis_range, widen any dimension whose data span is smaller than this, centred on
        its data. Both exist because a near-delta dimension otherwise gets binned over float noise, which
        draws as stripes rather than as a narrow distribution."""
        labels = self._build_feature_labels(space=space)
        samples = self._get_samples(space)
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()
        N, D = samples.shape

        # per-dimension (lo, hi) for binning and axes
        spans = []
        for d in range(D):
            if axis_range is not None:
                lo, hi = float(axis_range[0]), float(axis_range[1])
            else:
                lo, hi = float(np.nanmin(samples[:, d])), float(np.nanmax(samples[:, d]))
                if min_span is not None and (hi - lo) < min_span:
                    mid = 0.5 * (lo + hi)
                    lo, hi = mid - min_span / 2, mid + min_span / 2
                elif hi <= lo:  # a true delta with no min_span: give it a visible, arbitrary width
                    lo, hi = lo - 0.5, hi + 0.5
            spans.append((lo, hi))

        # Create D×D subplots (upper triangle empty)
        fig = make_subplots(
            rows=D, cols=D,
            horizontal_spacing=0.01, vertical_spacing=0.01,
            shared_xaxes=True, shared_yaxes=True,
        )

        # Loop over lower triangle
        for i in range(D):
            for j in range(D):
                if j >= i:
                    continue  # keep lower triangle only

                x = samples[:, j]
                y = samples[:, i]

                if mode == 'contour':
                    xlo, xhi = spans[j]
                    ylo, yhi = spans[i]
                    trace = go.Histogram2dContour(
                        x=x, y=y,
                        ncontours=32,
                        colorscale=cmap,
                        showscale=colorbar and (i == D - 1 and j == 0),
                        contours=dict(coloring='fill', showlines=False, start=0, end=None, size=None),
                        line=dict(smoothing=0.85, width=0),
                        xbins=dict(start=xlo, end=xhi, size=(xhi - xlo) / nbins),
                        ybins=dict(start=ylo, end=yhi, size=(yhi - ylo) / nbins),
                        autobinx=False, autobiny=False,
                    )

                elif mode == 'heatmap':
                    trace = go.Histogram2d(
                        x=x, y=y,
                        nbinsx=nbins, nbinsy=nbins,
                        colorscale=cmap,
                        showscale=colorbar and (i == D - 1 and j == 0),
                    )
                else:
                    raise ValueError("mode must be 'contour' or 'heatmap'")
                fig.add_trace(trace, row=i + 1, col=j + 1)
                if ref_dist is not None:
                    trace = go.Scatter(x=ref_dist[:, j], y=ref_dist[:, i], mode='markers',
                                       marker_color='white', marker_line_width=1.5, opacity=0.8,
                                       marker_line_color='#333333', marker_size=8, showlegend=False)
                    fig.add_trace(trace, row=i + 1, col=j + 1)

        for i in range(D):
            fig.update_xaxes(title_text=labels[i], row=D, col=i + 1)
            fig.update_yaxes(title_text=labels[i], row=i + 1, col=1)
        # axes match the bins, so a narrow distribution sits in a full frame rather than filling it
        for i in range(D):
            for j in range(D):
                if j >= i:
                    continue
                fig.update_xaxes(range=list(spans[j]), row=i + 1, col=j + 1)
                fig.update_yaxes(range=list(spans[i]), row=i + 1, col=j + 1)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            # height=1000,
            # width=1000,
            showlegend=False,
        )
        fig.update_layout(
            font=dict(family="Helvetica", size=12),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=30, r=30, t=20, b=30),
        )
        fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1)
        fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1)
        if show:
            fig.show(renderer=renderer)
        if return_fig:
            return fig
        else:
            return None

    def _get_color_set(self, n):
        from plotly.colors import qualitative
        import plotly.colors as pc

        if n <= len(qualitative.Plotly):
            return qualitative.Plotly[:n]
        elif n <= len(qualitative.Dark24):
            return qualitative.Dark24[:n]
        else:
            return pc.n_colors('rgb(0,0,255)', 'rgb(255,0,0)', n, colortype='rgb')

    def _add_violin(self, fig, samples, name, color, column_index, ranges, n_kde, bw_factor):
        row = column_index // 3 + 1
        col = column_index % 3 + 1
        x_samp, y_samp = lightweight_one_sided_violin(samples,
                                                      n_kde,
                                                      bandwidth_factor=bw_factor,
                                                      data_min=ranges[0],
                                                      data_max=ranges[1])
        fig.add_scatter(
            x=x_samp,
            y=y_samp,
            mode='lines',
            fill='toself',  # 'tonexty' if i == 0 else 'tonexty',  # Fill to next y (which is 0)
            fillcolor=color,
            line=dict(color=color, width=1.2),
            name=name,
            legendgroup=name,
            showlegend=True if column_index == 0 else False,
            row=row, col=col)

    def _get_samples(self, space, gauge_fix_free_axes: bool = True):
        if space == 'real':
            samples = self.full_cell_parameters().detach().cpu().numpy()
        elif space == 'latent':
            samples = self.latent_params(
                gauge_fix_free_axes=gauge_fix_free_axes).detach().cpu().numpy()
        elif space == 'standard':  # todo implement full std method
            samples = self.zp1_std_cell_parameters().detach().cpu().numpy()
        return samples

    def set_cell_parameters(self,
                            cell_parameters,
                            skip_box_analysis: bool = False):
        """Assign ``cell_lengths``, ``cell_angles``, ``aunit_centroid`` and ``aunit_orientation`` of ``self`` in place.

        The attributes are slices (views) of ``cell_parameters``; no cleaning or
        crystal-system enforcement is applied.

        Parameters
        ----------
        cell_parameters : torch.Tensor
            [num_graphs, 6 + 6 * max_z_prime] in the ``full_cell_parameters`` layout.
        skip_box_analysis : bool
            Skip the ``box_analysis`` call that otherwise follows.

        Raises
        ------
        AssertionError
            If the second dimension is not ``6 + 6 * max_z_prime``.
        """
        assert cell_parameters.shape[1] == 6 + self.max_z_prime * 6, (
            f"For crystal batch with max z_prime={self.max_z_prime},"
            f"require {6 + self.max_z_prime * 6} cell params.")

        self.cell_lengths = cell_parameters[:, :3]
        self.cell_angles = cell_parameters[:, 3:6]
        self.aunit_centroid = cell_parameters[:, 6:self.max_z_prime * 3 + 6]
        self.aunit_orientation = cell_parameters[:, self.max_z_prime * 3 + 6:]

        if not skip_box_analysis:
            self.box_analysis()

    def do_embedding(self,
                     embedding_type: str,
                     encoder: Optional = None):
        """Return a per-crystal feature vector: a molecule embedding concatenated with ``zp1_std_cell_parameters()``.

        Batch only. The embedding is computed on a clone of ``self`` by
        ``get_mol_embedding_for_proxy``.

        Parameters
        ----------
        embedding_type : str
            Forwarded to ``get_mol_embedding_for_proxy`` (e.g. ``'autoencoder'``,
            ``'principal_axes'``, ``'principal_moments'``).
        encoder : optional
            Forwarded to ``get_mol_embedding_for_proxy``; used by the
            ``'autoencoder'`` type.

        Returns
        -------
        torch.Tensor
            [num_graphs, embedding_dim + 12].

        Raises
        ------
        AssertionError
            If ``self`` is not a batch.
        """
        if self.is_batch:
            embedding = get_mol_embedding_for_proxy(self.clone(),
                                                    embedding_type,
                                                    encoder
                                                    )
            scaled_params = self.zp1_std_cell_parameters()
            return torch.cat([embedding, scaled_params], dim=1)
        else:
            assert False, "Crystal embedding not implemented for single samples"

    def optimize_crystal_parameters(self,
                                    mol_orientation: Optional[str] = 'standardized',
                                    return_record: Optional[bool] = False,
                                    **opt_kwargs):
        """Locally optimize the cell parameters of a copy of ``self`` with ``gradient_descent_optimization``.

        ``self`` is not modified: a clone (a collated one-graph batch for a single
        crystal) has its molecules re-oriented with ``orient_molecule(mode=mol_orientation,
        target_handedness=self.aunit_handedness)`` and is optimized starting from
        its ``full_cell_parameters()``.

        Parameters
        ----------
        mol_orientation : str
            ``mode`` forwarded to ``orient_molecule``.
        return_record : bool
            Also return the optimization records.
        **opt_kwargs
            Forwarded to ``gradient_descent_optimization``; must include its
            required ``optimizer_func``.

        Returns
        -------
        list or (list, dict)
            The optimized crystals as a list of single data objects (as returned
            by ``gradient_descent_optimization``, which may drop rows), and the
            records dict when ``return_record`` is True.
        """
        if self.is_batch:
            batch_to_optim = self.clone().to(self.device)
        else:
            batch_to_optim = collate_data_list([self]).clone().to(self.device)

        batch_to_optim.orient_molecule(mode=mol_orientation,
                                       target_handedness=self.aunit_handedness)
        opt_samples, opt_record = gradient_descent_optimization(
            batch_to_optim.full_cell_parameters(),
            batch_to_optim,
            **opt_kwargs
        )

        if return_record:
            return opt_samples, opt_record
        else:
            return opt_samples

    def _pad_tensor(self, val, shape, pad_val):
        """Pad a per-Z' parameter out to the max_z_prime width.

        `val` legitimately arrives as a python scalar for Z'=1 -- aunit_handedness
        is documented as accepting an int, and MolCrystalData.__init__ passes it
        straight through.  Reading `.dtype` off that int raised AttributeError, so
        constructing a crystal with an int handedness always failed.
        `out` is allocated on `val`'s device; the previous default-device
        allocation would have thrown on any CUDA input.
        """
        if not torch.is_tensor(val):
            val = torch.as_tensor(val, device=self.device)
        if val.ndim == 0:
            val = val.reshape(1, 1)
        elif val.ndim == 1:
            val = val.reshape(1, -1)
        out = torch.ones(shape, dtype=val.dtype, device=val.device) * pad_val
        out[:, :val.shape[-1]] = val
        return out

    def reset_sg_info(self, sg_ind):
        """Change the space group of ``self`` in place, with standard symmetry operators.

        Assigns ``sg_ind``, ``nonstandard_symmetry`` (False), ``symmetry_operators``
        (from ``SYM_OPS``: a list of arrays, one per crystal, for a batch; one
        array for a single crystal) and ``sym_mult``. For a batch, ``sg_ind``,
        ``nonstandard_symmetry`` and ``sym_mult`` are attached with
        ``add_graph_attr`` and ``self.sym_mult_lut`` is built if missing. Cell
        parameters, ``z_prime`` and the box attributes (``packing_coeff``,
        ``density``, ...) are not recomputed.

        Parameters
        ----------
        sg_ind : int, list of int, or torch.Tensor
            An int is applied to every crystal; a list or tensor gives one space
            group per crystal.

        Raises
        ------
        TypeError
            For any other type.
        """
        if isinstance(sg_ind, int):
            sg_ind_tensor = torch.ones_like(self.sg_ind, device=self.device) * sg_ind
        elif isinstance(sg_ind, list):
            sg_ind_tensor = torch.tensor(sg_ind, dtype=torch.long, device=self.device)
        elif torch.is_tensor(sg_ind):
            sg_ind_tensor = sg_ind * 1
        else:
            raise TypeError("sg_ind must be a tensor or an integer")

        unique_sgs = torch.unique(sg_ind_tensor)
        num_unique_sgs = len(unique_sgs)

        if self.is_batch:
            self.add_graph_attr(sg_ind_tensor, 'sg_ind')
            # reset symmetries will always be standard
            self.add_graph_attr(torch.zeros(self.num_graphs, dtype=torch.bool, device=self.device),
                                'nonstandard_symmetry')
        else:
            setattr(self, 'sg_ind', sg_ind_tensor)
            setattr(self, 'nonstandard_symmetry', False)

        if self.is_batch:
            if num_unique_sgs == 1:
                self.symmetry_operators = [np.stack(SYM_OPS[int(unique_sgs.item())])] * self.num_graphs
            else:
                self.symmetry_operators = [np.stack(SYM_OPS[int(ind)]) for ind in self.sg_ind]

            if not hasattr(self, 'sym_mult_lut'):
                self.build_sym_mult_tensor()

            sym_mult = self.sym_mult_lut[sg_ind_tensor]
            self.add_graph_attr(sym_mult, 'sym_mult')

        else:
            self.symmetry_operators = np.stack(SYM_OPS[int(self.sg_ind)])
            setattr(self, 'sym_mult', torch.tensor(
                len(self.symmetry_operators), dtype=torch.long, device=self.device)[None])

    def build_free_axis_tensor(self):
        """
        [231, 3] bool: which aunit centroid axes are FREE for each space group, i.e.
        the shared +1 eigenspace of G. Derived from cctbx's structure seminvariants
        (CONTINUOUS_DIMS, modulus-0 entries), which agrees 230/230 with the direct
        derivation from SYM_OPS -- see gfn docs/findings.md F-008.

        An axis is marked free when any vector listed for the space group in
        ``CONTINUOUS_DIMS`` has a nonzero component along it. The table is
        assigned to ``self.free_axis_lut`` and returned.
        """
        free_by_sg = torch.zeros(231, 3, dtype=torch.bool, device=self.device)
        for k, vecs in CONTINUOUS_DIMS.items():
            for vec in vecs:
                for axis, comp in enumerate(vec):
                    if comp:
                        free_by_sg[int(k), axis] = True
        self.free_axis_lut = free_by_sg
        return free_by_sg

    def canonicalize_free_axes(self):
        """
        Gauge-fix the FREE aunit centroid axes.

        A free axis is a fractional direction d with R_k d = d for every rotation part
        of G. Moving the aunit centroid along it displaces every symmetry copy by the
        same d, i.e. rigidly translates the whole crystal -- so the energy, the RDF and
        every other observable are EXACTLY invariant, and the coordinate carries no
        information at all. It exists only as an origin choice. 68 of 230 space groups
        have at least one (P1 has three); the centrosymmetric groups have none, which is
        why nothing needed this until the polar/Sohncke groups came into scope.

        CONVENTION: put the centroid at the CENTRE of its aunit box along each free
        axis (frac = auv_d / 2), which is latent 0. Deliberately not 0:

          * latent 0 is already the canonical value the projected cell angles take, so
            the whole dead-row mechanism needs no second constant and `_dead_values`
            stays all-zeros;
          * frac 0 is the box FACE, and for the 52 space groups whose free axes all have
            auv == 1 (26 with a hand-encoded ASYM_UNITS entry, 26 on the default
            [1, 1, 1]) the face is a wrap seam -- pinning a coordinate exactly on the seam
            is asking for trouble that the box centre simply does not have.

        Z' > 1 IS DELIBERATELY NOT HANDLED. The free translation is ONE GLOBAL shift
        shared by every asymmetric unit, so the correct operation there is a COMMON
        offset that lands unit 0 at the centre -- not setting each unit to the centre,
        which would destroy the relative offsets between the Z' molecules along that
        axis. Those offsets are physical. The common shift then pushes other units out
        of [0, auv_d], and re-wrapping a SINGLE unit by auv_d is only a symmetry when
        auv_d == 1 (a true lattice vector), which holds for 26 of the 42 free-axis
        groups that have a hand-encoded ASYM_UNITS entry, and not the rest (the
        other 26 free-axis groups use the default auv = [1, 1, 1]). That is a real
        open question, not a detail, so this
        returns untouched at Z' > 1 rather than guessing. sg 9 Z'=2 is the test case.

        Builds and caches ``self.free_axis_lut`` and ``self.asym_unit_lut`` if
        missing. When it acts, it assigns ``self.aunit_centroid`` ([num_graphs, 3])
        with every free-axis coordinate set to ``auv_d / 2`` and the others unchanged.

        Returns
        -------
        bool
            False (nothing changed) if ``max_z_prime > 1``, any ``z_prime > 1``, or
            no crystal in the batch has a free axis; True after assigning.
        """
        if int(self.max_z_prime) > 1 or bool(torch.any(self.z_prime > 1)):
            return False
        if not hasattr(self, 'free_axis_lut'):
            self.build_free_axis_tensor()
        if not hasattr(self, 'asym_unit_lut'):
            self.build_asym_unit_tensor()

        sg_inds = self.sg_ind.long()
        free = self.free_axis_lut.to(self.device)[sg_inds]            # [N, 3] bool
        if not bool(free.any()):
            return False

        auvs = self.asym_unit_lut.to(self.device)[sg_inds]            # [N, 3]
        centroid = self.aunit_centroid[:, :3]
        target = auvs * 0.5
        self.aunit_centroid = torch.where(free, target, centroid)
        return True

    def canonicalize_zp_aunits(self):
        """Reorder each crystal's asymmetric units into canonical order, in place; atoms move with their slots.

        Uses ``canonicalize_aunit_order``: units are sorted by ascending norm of
        their fractional centroid. Slots beyond a crystal's ``z_prime`` sort last
        and are filled with centroid 0.5 and orientation pi/2 in every component.
        Reassigns ``aunit_centroid``, ``aunit_orientation`` and ``aunit_handedness``
        with new tensors. A single crystal is collated into a one-graph batch first.

        The atoms move with their pose slots (``_unit_block_moves``), so a crystal whose
        units hold different conformers is the same crystal before and after. On a
        crystal with ``z_prime > 1`` whose used slots change order, atom block ``j``
        (of ``z_prime`` contiguous blocks of ``num_atoms // z_prime`` atoms, the layout
        ``split_to_zp1_batch`` reads) becomes the old block of the unit that moves into
        slot ``j``. Reassigned with new objects when any crystal moves: the unit map
        ``unit_ind`` when stored, moved with its atoms and relabelled to the new slot order;
        every per-atom tensor sliced by ``ptr`` (``pos``, ``z``, ``x``, ...) except ``mol_ind``, which
        numbers units by position; row-level lists of per-atom lists; ``file_bonds``,
        ``mol_bonds`` and ``'index'``/``'face'`` tensors, renumbered; the per-slot
        row tensors ``aunit_op``, ``unit_radius``, ``unit_mass``, ``unit_volume`` and
        ``unit_fingerprint``; and the per-unit lists ``unit_smiles``.
        Crystals with ``z_prime <= 1`` and crystals whose order does not change keep
        their atoms in place; when ``max_z_prime == 1`` no atom attribute is touched.
        Unit-cell attributes (``unit_cell_pos``, ``unit_cell_batch``,
        ``unit_cell_mol_ind``) are not reordered: they still hold the same crystal,
        with the units in the old order.

        Raises
        ------
        ValueError
            From ``_unit_block_moves``: a moving crystal whose units cannot be read
            (``unit_layout``; e.g. an atom count not divisible by ``z_prime``, or a unit
            map of unequal units), or with a used slot sorting behind an unused one
            (fractional centroid norm >= sqrt(3)), or a per-slot row tensor narrower than
            a moving crystal's ``z_prime``.
        """
        batch = self if self.is_batch else collate_data_list([self], max_z_prime=self.max_z_prime)
        moves = _unit_block_moves(batch) if batch.max_z_prime > 1 else {}
        self.aunit_centroid, self.aunit_orientation, self.aunit_handedness = canonicalize_aunit_order(batch)
        for key, value in moves.items():
            self[key] = value[0] if (isinstance(value, list) and not self.is_batch) else value

    def canonicalize_orientation(self):
        """Map every asymmetric unit's rotation vector to its equivalent with non-negative z, in place.

        A rotation vector ``v`` with ``v_z < 0`` becomes
        ``-(v / |v|) * (2 pi - |v|)`` (``canonicalize_rotvec``), which describes the
        same rotation. Assigns ``self.aunit_orientation`` ([num_graphs,
        3 * max_z_prime]). ``canonicalize_rotvec`` writes in place into the
        reshaped input, so the previous tensor is also modified when the reshape
        is a view.
        """
        flat_rotvecs = self.aunit_orientation.reshape(self.num_graphs * self.max_z_prime, 3)
        fixed_flat_rotvecs = canonicalize_rotvec(flat_rotvecs)
        fixed_rotvecs = fixed_flat_rotvecs.reshape(self.num_graphs, self.max_z_prime * 3)
        self.aunit_orientation = fixed_rotvecs

    def compute_standard_cell(self, confirm_transform: bool = False,
                              enforce_right_handedness: bool = False,
                              symprec: float = 1e-3):
        """
        Re-express each crystal in spglib's standard cell, in the setting and origin of SYM_OPS[sg_ind].
        symprec is spglib's distance tolerance in Angstrom; at spglib's default (1e-5) float32 crystals are often
        assigned a subgroup. Raises ValueError if spglib's space group for any crystal is not its sg_ind, or if
        the standard cell is not a unimodular change of basis.
        Triclinic (sg 1, 2): the output is spglib's Niggli cell, all angles acute or all obtuse. The reduction penalty uses
        a beta/gamma-obtuse convention instead (sym_utils.tri_niggli_reduction_penalty), so all-acute outputs score > 0 and
        fail the confirm_transform penalty check until (a, b, c) -> (a, -b, -c) is applied.

        Batch only. Calls ``self.mol2ucell()`` first, which reassigns ``pos``,
        ``unit_cell_pos``, ``unit_cell_batch`` and ``unit_cell_mol_ind`` on
        ``self``; requires ``T_cf`` from ``box_analysis``. The cell parameters are
        converted with ``.numpy()`` without ``.cpu()``/``.detach()``, so they must be
        CPU tensors that do not require grad. The result is a clone of ``self`` with
        the standard cell's ``cell_lengths``/``cell_angles`` (box re-analysed),
        unit-cell positions transformed as ``x_s = P x + p``, aunit parameters and
        ``pos`` from ``reparameterize_unit_cell`` (then re-posed with
        ``pose_aunit``), ``is_well_defined``, standard ``symmetry_operators`` and
        ``nonstandard_symmetry`` set False.

        Parameters
        ----------
        confirm_transform : bool
            Also compare RDFs of ``self`` and the result and compute the result's
            ``compute_cell_reduction_penalty``; a crystal succeeds when both are
            below 1e-3. The number of failures is printed.
        enforce_right_handedness : bool
            Forwarded to ``reparameterize_unit_cell``.
        symprec : float
            spglib distance tolerance (Angstrom).

        Returns
        -------
        MolCrystalData batch, or (MolCrystalData batch, torch.Tensor)
            The standardized batch, plus a [num_graphs] bool success mask when
            ``confirm_transform`` is True.

        Raises
        ------
        ValueError
            If spglib finds no symmetry, a different space group, or a transform
            with determinant other than 1 for any crystal.
        AssertionError
            If ``self`` is not a batch.
        """
        assert self.is_batch, "Cell standardization currently only implemented for batch objects"
        import spglib

        """set up unit cell"""
        self.mol2ucell()

        target_params = self.full_cell_parameters()
        std_params = []
        transforms = []
        new_positions = []
        mismatches = []

        """get standard cell from spglib"""
        for ind, p in enumerate(target_params):
            "box params"
            cellpar = p[:6].clone().numpy()
            cellpar[3:] *= (180 / np.pi)
            lattice = cellpar_to_cell(cellpar)

            "fractional positions"
            positions = fractional_transform(self.unit_cell_pos[self.unit_cell_batch == ind],
                                             self.T_cf[ind]).cpu().detach().numpy()

            # the unit cell's layout: each unit tiled sym_mult times, unit by unit
            numbers = unit_cell_z(self.z[self.batch == ind], self.z_prime.reshape(-1)[ind],
                                  self.sym_mult[ind]).cpu().detach().numpy()

            "spglib symmetry, standardized in the SYM_OPS setting"
            cell = (lattice, positions, numbers)
            sg_ind = int(self.sg_ind[ind])
            hall_number = spglib_hall_number(sg_ind)
            dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
            if dataset is not None and dataset.number == sg_ind and dataset.hall_number != hall_number:
                dataset = spglib.get_symmetry_dataset(cell, symprec=symprec, hall_number=hall_number)
            det = None if dataset is None else float(np.linalg.det(dataset.transformation_matrix))
            if dataset is None or dataset.number != sg_ind or abs(det - 1) > 1e-6:
                mismatches.append((ind, sg_ind, None if dataset is None else dataset.number, det))
                continue

            "(a_s b_s c_s) = (a b c) P^-1 and x_s = P x + p, with column vectors"
            P = dataset.transformation_matrix
            lattice_std = np.linalg.inv(P).T @ lattice
            cellpar_std = cell_to_cellpar(lattice_std)

            "manually apply transform to atom positions (avoid breaking mols)"
            T = P.T
            positions_new = positions @ T + dataset.origin_shift
            transforms.append(T)
            new_positions.append(positions_new)
            std_params.append(cellpar_std)

        if mismatches:
            raise ValueError(
                f"compute_standard_cell: spglib (symprec={symprec}) disagrees with the crystal for "
                f"{len(mismatches)} of {self.num_graphs} samples, as (index, sg_ind, spglib number, det P): "
                f"{mismatches[:20]}")

        std_params = np.stack(std_params)

        """instantiate standardized batch object"""
        std_batch = self.clone()
        "new box"
        std_batch.cell_lengths = torch.tensor(std_params[:, :3], dtype=torch.float32, device=self.device)
        std_batch.cell_angles = torch.tensor(std_params[:, 3:] / 180 * np.pi, dtype=torch.float32, device=self.device)
        std_batch.box_analysis()
        "transformed cartesian positions (not properly wrapped)"
        #ucell_pos_frac = torch.as_tensor(np.stack(new_positions), dtype=torch.float32, device=self.device)
        ucell_pos_frac = torch.as_tensor(np.concatenate(new_positions), dtype=torch.float32, device=self.device)
        ucell_batch = torch.arange(self.num_graphs).repeat_interleave((self.num_atoms * self.sym_mult).long())
        #ucell_pos_cart = fractional_transform(ucell_pos_frac.flatten(0, 1), std_batch.T_fc[ucell_batch])
        ucell_pos_cart = fractional_transform(ucell_pos_frac, std_batch.T_fc[ucell_batch])
        std_batch.unit_cell_pos = ucell_pos_cart
        std_batch.unit_cell_batch = ucell_batch

        "parameterize unit cell (wrapping handled here) and update crystal params"
        (aunit_centroid, aunit_orientation,
         handedness_list, is_well_defined, pos) = std_batch.reparameterize_unit_cell(
            enforce_right_handedness=enforce_right_handedness
        )

        std_batch.symmetry_operators = [SYM_OPS[int(ind)] for ind in
                                        std_batch.sg_ind]  # critical - give standard sym ops!
        std_batch.nonstandard_symmetry.fill_(False)

        std_batch.aunit_centroid = aunit_centroid
        std_batch.aunit_orientation = aunit_orientation
        if handedness_list.ndim == 1:
            std_batch.aunit_handedness = handedness_list[:, None]
        else:
            std_batch.aunit_handedness = handedness_list

        std_batch.is_well_defined = torch.tensor(is_well_defined, dtype=torch.bool, device=self.device)

        std_batch.pos = pos
        std_batch.pose_aunit()

        if confirm_transform:
            # estimate magnitude of the transform
            # transform_dist = torch.cdist(torch.tensor(transforms.reshape(-1, 9), dtype=torch.float32), torch.eye(3).flatten()[None, :]).flatten()

            rdf1 = self.analyze(['rdf'])['rdf'][0]
            rdf2 = std_batch.analyze(['rdf'])['rdf'][0]
            diffs = compute_rdf_distance(rdf1, rdf2, torch.linspace(0, 10, rdf1.shape[-1],
                                                                    dtype=torch.float32, device=self.device))

            penalties = std_batch.compute_cell_reduction_penalty()

            succeeded = (diffs < 1e-3) & (penalties < 1e-3)
            if not succeeded.all():
                print((~succeeded).sum().item(), " samples failed standardization")

        if confirm_transform:
            return std_batch, succeeded
        else:
            return std_batch

    def write_cif(self, inds, path, mode='asymmetric unit'):
        """Write one CIF per index in `inds`, as `{path}_{ind}.cif`.

        mode='asymmetric unit' (default) writes space group number, H-M symbol,
        the symmetry operator loop and the asymmetric unit -- so the file can be
        read back into this object.  Every other mode goes to the ASE writer,
        which writes the atoms that mode selects in a P1 cell and records NO
        space group; 'unit cell' is what the
        visualisation and COMPACK callers pass and is unchanged.

        Parameters
        ----------
        inds : iterable of int or torch.Tensor
            Batch indices of the crystals to write.
        path : str or pathlib.Path
            Output prefix; files are named ``{path}_{ind}.cif``.
        mode : str
            ``'asymmetric unit'``, or a mode of ``ase_mol_from_crystaldata``
            (``'unit cell'`` needs ``unit_cell_pos``, e.g. from ``mol2ucell``).

        Returns
        -------
        list of pathlib.Path or None
            The written paths in ``'asymmetric unit'`` mode; None otherwise.
        """
        if mode == 'asymmetric unit':
            from mxtaltools.common.cif_io import write_asymmetric_unit_cif
            return write_asymmetric_unit_cif(self, inds, path)
        ase_write_cif(self, inds, path, mode)